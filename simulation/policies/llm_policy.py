"""LLM-based agent policy using an OpenAI-compatible chat API.

Features:
- Retry + exponential backoff for rate limits and timeouts
- Multi-model fallback strategy with lightweight models first
- Concurrency control via semaphore
- Comprehensive metrics tracking
- Structured error handling and logging
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import time
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from simulation.policies.base import AgentPolicy
from simulation.policies.llm_provider import (
    BaseLLMProvider,
    OllamaProvider,
    RateLimitError,
    TimeoutError as LLMTimeoutError,
    ModelDecommissionedError,
    InvalidAPIKeyError,
    get_llm_models,
)

logger = logging.getLogger(__name__)

# Default values — can be overridden via SimulationConfig or environment variables.
_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
_FALLBACK_ACTION = "defect"  # temporary for debugging
# Maximum number of cached decisions kept in memory per policy instance.
_MAX_CACHE_SIZE = 1024

# Retry configuration - can be overridden via environment
_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
_RETRY_BACKOFF_BASE_RATE_LIMIT = float(os.getenv("LLM_RETRY_BACKOFF_RATE_LIMIT", "2.0"))
_RETRY_BACKOFF_BASE_TIMEOUT = float(os.getenv("LLM_RETRY_BACKOFF_TIMEOUT", "1.5"))


def batch_agents(agents: List[Any], batch_size: int) -> List[List[Any]]:
    """Split agents into fixed-size batches for batched LLM strategy updates."""
    if batch_size <= 0:
        return [list(agents)]
    return [agents[i : i + batch_size] for i in range(0, len(agents), batch_size)]


def summarize_memory(agent, window: int = 10) -> Dict[str, float]:
    """Summarize recent interaction memory into stable strategy features."""
    all_entries = list(getattr(agent, "interaction_memory", []))
    recent = [
        entry
        for entry in all_entries[-window:]
        if "action" in entry and "opponent_action" in entry and "reward" in entry
    ]
    recent_interaction_count = len(recent)
    if recent_interaction_count == 0:
        return {
            "recent_coop_rate": 0.0,
            "avg_reward": 0.0,
            "defect_rate": 0.0,
            "recent_interaction_count": 0,
            "total_interaction_count": sum(
                1
                for entry in all_entries
                if "action" in entry and "opponent_action" in entry and "reward" in entry
            ),
        }

    cooperations = sum(1 for entry in recent if entry.get("action") == "cooperate")
    defections = sum(1 for entry in recent if entry.get("action") == "defect")
    avg_reward = sum(float(entry.get("reward", 0.0)) for entry in recent) / recent_interaction_count
    return {
        "recent_coop_rate": cooperations / recent_interaction_count,
        "avg_reward": avg_reward,
        "defect_rate": defections / recent_interaction_count,
        "recent_interaction_count": recent_interaction_count,
        "total_interaction_count": sum(
            1
            for entry in all_entries
            if "action" in entry and "opponent_action" in entry and "reward" in entry
        ),
    }


class LLMPolicy(AgentPolicy):
    """Policy that asks an LLM to decide whether to cooperate or defect.

    The policy builds a natural-language prompt from the agent's current
    state (resources, trust, memory) and the opponent's state, sends it to
    an OpenAI-compatible chat-completion endpoint, and parses the response
    into ``"cooperate"`` or ``"defect"``.

    API credentials are read from the environment variable ``OPENAI_API_KEY``
    (or ``LLM_API_KEY`` as an alias) — they are never hard-coded.

    Args:
        model: LLM model name to request (default: ``"gpt-4o-mini"``).
        api_base_url: Base URL of the chat-completion API.  Set this to use
            any OpenAI-compatible provider (default:
            ``"https://api.openai.com/v1"``).
        policy_logger: Optional callable ``(record: dict) -> None`` invoked
            after every decision with a structured log record containing the
            agent ID, prompt, raw response, and parsed action.  Falls back to
            the module-level Python logger when not provided.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        model: str = _DEFAULT_MODEL,
        api_base_url: str = _DEFAULT_API_BASE_URL,
        timeout: int = 3,
        policy_logger=None,
        decision_interval: int = 4,
        max_concurrent_llm_calls: int = 4,
        batch_size: int = 8,
        enable_async: bool = True,
        debug_llm: bool = False,
        llm_models: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        # Provider abstraction lets policy logic stay unchanged while backend
        # selection switches between local Ollama and hosted Groq.
        self.provider: BaseLLMProvider = provider
        self._policy_logger = policy_logger
        self.enable_async = enable_async
        self.debug_llm = debug_llm
        # Throttling: only call LLM every decision_interval invocations per agent;
        # reuse the last decision in between to reduce API load.
        self.decision_interval = decision_interval
        self.batch_size = max(1, batch_size)
        self.max_concurrent_llm_calls = max(1, max_concurrent_llm_calls)
        self._semaphore = asyncio.Semaphore(self.max_concurrent_llm_calls)
        self._call_counters: Dict[Any, int] = {}          # agent_id -> total call count
        self._throttle_cache: Dict[Any, str] = {}         # agent_id -> last action
        # Lightweight in-memory decision cache keyed by a minimal state tuple.
        self._decision_cache: Dict[tuple, str] = {}
        
        # Multi-model fallback support
        self.llm_models = llm_models or get_llm_models()
        logger.info("LLMPolicy initialized with models: %s", self.llm_models)
        
        # Diagnostics counters for dashboard-level fallback-rate reporting.
        self._llm_call_count: int = 0
        self._llm_error_count: int = 0
        self._fallback_count: int = 0
        # Retry metrics: track how many times we retry before succeeding or failing
        self._llm_retry_count: int = 0
        # Per-agent-decision counters used to compute an accurate fallback rate.
        # Unlike _llm_call_count (per-request) and _fallback_count (mixed
        # granularity), these are incremented exactly once per agent decision
        # attempt, making fallback_rate = fallback_agent_decisions /
        # total_agent_decisions meaningful.
        self._total_agent_decisions: int = 0
        self._success_agent_decisions: int = 0
        self._fallback_agent_decisions: int = 0
        self._llm_total_latency_seconds: float = 0.0
        self._llm_latency_samples: int = 0
        if self.provider is None:
            # Defensive fallback preserves previous local defaults if no provider
            # is injected by the caller.
            self.provider = OllamaProvider(
                base_url=self.api_base_url,
                model=self.model,
                timeout=float(self.timeout),
            )

    def _log_fallback(self, reason: str) -> None:
        logger.warning("LLM fallback triggered: %s", reason)

    def _record_llm_success(self, strategy: str) -> None:
        """Track one successful LLM-produced strategy and emit diagnostics."""
        self._success_agent_decisions += 1
        model = getattr(self.provider, "model", self.model)
        logger.info(f"LLM SUCCESS: {model} -> {strategy}")

    def _build_provider_error_reason(self, prefix: str, exc: Exception | None = None) -> str:
        provider = getattr(self, "provider", None)
        provider_error = None
        provider_message = None
        if provider is not None and hasattr(provider, "get_health"):
            health = provider.get_health()
            provider_error = health.get("provider_error") or health.get("error_type")
            provider_message = health.get("message")

        if provider_error:
            if provider_message:
                return f"{prefix} provider_error={provider_error} detail={provider_message}"
            return f"{prefix} provider_error={provider_error}"
        if exc is not None:
            return f"{prefix} detail={exc}"
        return prefix

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def decide(self, agent, context) -> str:
        """Return ``"cooperate"`` or ``"defect"`` via an LLM call.

        Falls back to ``"defect"`` if the API call fails so the simulation
        can continue gracefully without a live LLM.

        When ``decision_interval > 1``, the LLM is only queried on every
        *k*-th call per agent; the last result is reused between calls to
        limit API load without removing LLM involvement.  A lightweight
        state-based cache further reduces redundant calls when two agents
        face identical conditions across steps.

        Args:
            agent: The :class:`~simulation.agent.Agent` making the decision.
            context: The opponent agent (or world object).  Agents with an
                ``agent_id`` attribute provide richer prompt context.

        Returns:
            ``"cooperate"`` or ``"defect"``.
        """
        agent_id = agent.agent_id

        # --- Throttle: reuse last decision within the interval window ---
        # call_count is read *before* incrementing so that the first call has
        # call_count=0, which always passes through (0 % N == 0 for any N).
        # Subsequent calls at positions 1, 2, 3 are throttled; position 4
        # (call_count % decision_interval == 0) passes through again, etc.
        call_count = self._call_counters.get(agent_id, 0)
        self._call_counters[agent_id] = call_count + 1

        if call_count % self.decision_interval != 0:
            cached = self._throttle_cache.get(agent_id)
            if cached is not None:
                return cached

        # --- Decision cache: skip LLM when state is identical to a prior call ---
        cache_key = self._make_cache_key(agent, context)
        if cache_key in self._decision_cache:
            action = self._decision_cache[cache_key]
            self._throttle_cache[agent_id] = action
            return action

        # --- Full LLM call path ---
        prompt = self._build_prompt(agent, context)
        raw_response: Optional[str] = None
        action = _FALLBACK_ACTION
        used_fallback = False
        other_agent_id = (
            context.agent_id if context is not None and hasattr(context, "agent_id") else None
        )
        prompt_summary = self._build_prompt_summary(agent, context)

        self._total_agent_decisions += 1
        try:
            if self.debug_llm:
                logger.info("LLM CALLED agent_id=%s", agent.agent_id)
            self._llm_call_count += 1
            started = time.perf_counter()
            raw_response = self._call_llm(prompt)
            self._record_latency(time.perf_counter() - started)
            action, used_fallback = self._parse_response_with_fallback(raw_response)
            if not used_fallback:
                self._record_llm_success(action)
        except (RateLimitError, LLMTimeoutError, ModelDecommissionedError, InvalidAPIKeyError) as exc:
            self._llm_error_count += 1
            used_fallback = True
            self._log_fallback(f"agent_id={agent.agent_id} stage=decision error_type={exc.error_type}")
            action = _FALLBACK_ACTION
        except Exception as exc:  # noqa: BLE001
            self._llm_error_count += 1
            used_fallback = True
            self._log_fallback(
                self._build_provider_error_reason(
                    f"agent_id={agent.agent_id} stage=decision",
                    exc,
                )
            )
            action = _FALLBACK_ACTION

        if used_fallback:
            self._fallback_count += 1
            self._fallback_agent_decisions += 1

        log_record: Dict[str, Any] = {
            "action": "decide_action",
            "policy": "llm",
            "llm_called": True,
            "used_fallback": used_fallback,
            "fallback_action": _FALLBACK_ACTION,
            "agent_id": agent.agent_id,
            "other_agent_id": other_agent_id,
            "prompt": prompt,
            "prompt_summary": prompt_summary,
            "raw_response": raw_response,
            "decision": action,
            "my_resources": agent.resources,
        }

        # Write to the structured policy logger (if provided), then also
        # append a lightweight copy to the agent's own interaction_memory.
        if self._policy_logger is not None:
            self._policy_logger(log_record)
        else:
            logger.info(
                "LLMPolicy decision: agent_id=%s decision=%s",
                agent.agent_id,
                action,
            )
            logger.debug("LLMPolicy full record: %s", json.dumps(log_record))

        agent.interaction_memory.append(log_record)

        # Store results in both caches for future reuse.
        if len(self._decision_cache) < _MAX_CACHE_SIZE:
            self._decision_cache[cache_key] = action
        self._throttle_cache[agent_id] = action

        return action

    def generate_strategy(self, agent) -> str:
        """Determine a standing strategy for *agent* based on its recent history.

        Called periodically (every ``decision_interval`` steps) rather than
        on every interaction, so LLM load stays proportional to the number of
        agents rather than the number of interactions.

        Args:
            agent: The :class:`~simulation.agent.Agent` requesting a strategy
                update.  Uses ``interaction_memory``, ``resources``, and
                ``relationships`` for context.

        Returns:
            ``"cooperate"`` or ``"defect"``.
        """
        prompt = self._build_strategy_prompt(agent)
        action = _FALLBACK_ACTION
        raw_response: Optional[str] = None
        used_fallback = False

        self._total_agent_decisions += 1
        try:
            if self.debug_llm:
                logger.info("LLM CALLED agent_id=%s", agent.agent_id)
            self._llm_call_count += 1
            started = time.perf_counter()
            raw_response = self._call_llm(prompt)
            self._record_latency(time.perf_counter() - started)
            action, used_fallback = self._parse_response_with_fallback(raw_response)
            if not used_fallback:
                self._record_llm_success(action)
        except Exception as exc:  # noqa: BLE001
            self._llm_error_count += 1
            used_fallback = True
            self._log_fallback(
                self._build_provider_error_reason(
                    f"agent_id={agent.agent_id} stage=strategy_update",
                    exc,
                )
            )

        if used_fallback:
            self._fallback_count += 1
            self._fallback_agent_decisions += 1

        log_record: Dict[str, Any] = {
            "action": "strategy_update",
            "policy": "llm",
            "llm_called": True,
            "used_fallback": used_fallback,
            "fallback_action": _FALLBACK_ACTION,
            "agent_id": agent.agent_id,
            "prompt": prompt,
            "raw_response": raw_response,
            "new_strategy": action,
            "my_resources": agent.resources,
        }

        if self._policy_logger is not None:
            self._policy_logger(log_record)
        else:
            logger.info(
                "LLMPolicy strategy update: agent_id=%s new_strategy=%s",
                agent.agent_id,
                action,
            )

        agent.interaction_memory.append(log_record)
        return action

    async def generate_strategy_async(self, agent) -> str:
        """Async strategy generation with timeout, per-task fallback, and latency tracking."""
        prompt = self._build_strategy_prompt(agent)
        action = _FALLBACK_ACTION
        raw_response: Optional[str] = None
        used_fallback = False
        latency_seconds = 0.0

        if self.debug_llm:
            logger.info("LLM CALLED agent_id=%s", agent.agent_id)

        self._total_agent_decisions += 1
        try:
            self._llm_call_count += 1
            started = time.perf_counter()
            raw_response = await self._call_llm_async(prompt)
            latency_seconds = time.perf_counter() - started
            self._record_latency(latency_seconds)
            action, used_fallback = self._parse_response_with_fallback(raw_response)
            if not used_fallback:
                self._record_llm_success(action)
        except asyncio.TimeoutError:
            self._llm_error_count += 1
            used_fallback = True
            self._log_fallback(
                f"agent_id={agent.agent_id} stage=strategy_update_async reason=timeout timeout={self.timeout}s"
            )
        except RateLimitError as exc:
            self._llm_error_count += 1
            used_fallback = True
            self._log_fallback(f"agent_id={agent.agent_id} stage=strategy_update_async error_type=rate_limit")
        except (LLMTimeoutError, ModelDecommissionedError, InvalidAPIKeyError) as exc:
            self._llm_error_count += 1
            used_fallback = True
            self._log_fallback(f"agent_id={agent.agent_id} stage=strategy_update_async error_type={exc.error_type}")
        except Exception as exc:  # noqa: BLE001
            self._llm_error_count += 1
            used_fallback = True
            self._log_fallback(
                self._build_provider_error_reason(
                    f"agent_id={agent.agent_id} stage=strategy_update_async",
                    exc,
                )
            )

        if used_fallback:
            self._fallback_count += 1
            self._fallback_agent_decisions += 1

        log_record: Dict[str, Any] = {
            "action": "strategy_update",
            "policy": "llm",
            "llm_called": True,
            "used_fallback": used_fallback,
            "fallback_action": _FALLBACK_ACTION,
            "agent_id": agent.agent_id,
            "prompt": prompt,
            "raw_response": raw_response,
            "new_strategy": action,
            "my_resources": agent.resources,
            "llm_latency_seconds": latency_seconds,
        }

        if self._policy_logger is not None:
            self._policy_logger(log_record)

        agent.interaction_memory.append(log_record)
        return action

    async def generate_strategies_batch_async(self, agent_batch: List[Any]) -> Dict[Any, str]:
        """Generate strategies for a batch of agents with one LLM call.

        Falls back to per-agent async calls if the batch request fails.
        """
        if not agent_batch:
            return {}

        if self.debug_llm:
            logger.info("LLM BATCH CALLED size=%s", len(agent_batch))

        prompt = self._build_batch_strategy_prompt(agent_batch)
        started = time.perf_counter()
        try:
            self._llm_call_count += 1
            raw_response = await self._call_llm_async(
                prompt,
                max_tokens=self._batch_max_tokens(len(agent_batch)),
                expect_json=True,
            )
            latency_seconds = time.perf_counter() - started
            self._record_latency(latency_seconds)
            mapping = self._parse_batch_response(raw_response, agent_batch)

            missing_agent_ids = [a.agent_id for a in agent_batch if a.agent_id not in mapping]
            if missing_agent_ids:
                raise ValueError(f"Batch response missing agent ids: {missing_agent_ids}")

            # Count every agent in the batch as one decision attempt. Per-agent
            # parse fallbacks were already recorded inside _parse_batch_response.
            self._total_agent_decisions += len(agent_batch)
            return mapping
        except Exception as exc:  # noqa: BLE001
            self._log_fallback(
                self._build_provider_error_reason(
                    f"batch_size={len(agent_batch)} stage=batch_strategy",
                    exc,
                )
            )
            fallback_results: Dict[Any, str] = {}

            async def _fallback(agent) -> Tuple[Any, str]:
                action = await self.generate_strategy_async(agent)
                return agent.agent_id, action

            results = await asyncio.gather(*[_fallback(agent) for agent in agent_batch], return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    continue
                agent_id, action = result
                fallback_results[agent_id] = action

            for agent in agent_batch:
                if agent.agent_id not in fallback_results:
                    # Safety net for the rare case where generate_strategy_async
                    # raised unexpectedly (it normally always returns). These agents
                    # were not counted by generate_strategy_async, so we account for
                    # them here. Note: _total_agent_decisions += len(agent_batch) in
                    # the success path above never runs when we reach this except
                    # block, so there is no risk of double-counting.
                    self._total_agent_decisions += 1
                    self._fallback_count += 1
                    self._fallback_agent_decisions += 1
                    fallback_results[agent.agent_id] = _FALLBACK_ACTION
            return fallback_results

    def generate_strategies_batch(self, agent_batch: List[Any]) -> Dict[Any, str]:
        """Sync wrapper for batch strategy generation for non-async callers."""
        if not self.enable_async:
            return {agent.agent_id: self.generate_strategy(agent) for agent in agent_batch}

        try:
            return asyncio.run(self.generate_strategies_batch_async(agent_batch))
        except RuntimeError:
            # If already inside an event loop, safely degrade to sync path.
            return {agent.agent_id: self.generate_strategy(agent) for agent in agent_batch}

    def _build_strategy_prompt(self, agent) -> str:
        """Build a minimal prompt for a periodic strategy update.

        Intentionally compact — only own state, recent interaction history,
        and average trust — to keep token counts low and latency within the
        configured timeout.

        Args:
            agent: The agent requesting the strategy update.

        Returns:
            A multi-line prompt string.
        """
        resources = agent.resources
        rel_wealth = self._relative_wealth_section(agent)
        memory_stats = summarize_memory(agent, window=10)

        # Recent interactions: at most the last 10 true interaction entries.
        recent = [
            e
            for e in list(getattr(agent, "interaction_memory", []))[-10:]
            if "step" in e and "action" in e and "opponent_action" in e and "reward" in e
        ]
        if recent:
            history_lines = "\n".join(
                f"  - step {e['step']}: I played {e['action']}, "
                f"opponent played {e['opponent_action']}, reward={e['reward']}"
                for e in recent
            )
            history_section = f"Recent interactions:\n{history_lines}\n"
        else:
            history_section = "Recent interactions: none\n"

        # Average trust across all known relationships
        relationships = getattr(agent, "relationships", {})
        if relationships:
            avg_trust = sum(r["trust"] for r in relationships.values()) / len(relationships)
            trust_str = f"{avg_trust:.2f}"
        else:
            trust_str = "no data"

        env_ctx = getattr(agent, "simulation_context", {})
        scarcity = env_ctx.get("scarcity_level", "unknown")

        return (
            "You are setting your strategy for the next several interactions "
            "in a simulated society.\n\n"
            f"Your state:\n"
            f"  - Resources: {resources}\n"
            f"  - Relative wealth: {rel_wealth}\n"
            f"  - Average trust toward others: {trust_str}\n"
            f"  - Scarcity level: {scarcity}\n\n"
            "Recent summary statistics:\n"
            f"  - recent_interaction_count: {memory_stats.get('recent_interaction_count', 0)}\n"
            f"  - total_interaction_count: {memory_stats.get('total_interaction_count', 0)}\n"
            f"  - recent_cooperation_rate: {memory_stats.get('recent_coop_rate', 0.0):.2f}\n"
            f"  - defection_rate: {memory_stats.get('defect_rate', 0.0):.2f}\n"
            f"  - average_reward: {memory_stats.get('avg_reward', 0.0):.2f}\n\n"
            f"{history_section}\n"
            "Choose the strategy that best serves your long-term interests.\n"
            "Respond with ONLY one word: cooperate or defect"
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _make_cache_key(self, agent, context) -> Tuple[Any, Any, int, int, str, float]:
        """Build a minimal state key for the decision cache.

        Uses rounded resource values and the last known outcome/trust so that
        minor floating-point fluctuations do not defeat the cache, while still
        capturing meaningful state differences.

        The *context* object is expected to be an :class:`~simulation.agent.Agent`
        instance (attributes: ``agent_id``, ``resources``).  When context is a
        world object or None, default values are used so the cache still works
        safely in backwards-compatible call paths.
        """
        if context is not None:
            opponent_id = getattr(context, "agent_id", None)
            opp_resources = getattr(context, "resources", 0)
        else:
            opponent_id = None
            opp_resources = 0
        memory_entry = agent.memory.get(opponent_id, {}) if opponent_id is not None else {}
        last_outcome = memory_entry.get("last_outcome", "none")
        trust = round(memory_entry.get("trust", 0.0), 1)
        return (
            agent.agent_id,
            opponent_id,
            round(agent.resources),
            round(opp_resources),
            last_outcome,
            trust,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, agent, context) -> str:
        """Construct the natural-language prompt sent to the LLM.

        Args:
            agent: The deciding agent.
            context: The opponent agent or world object.

        Returns:
            A multi-line prompt string.
        """
        agent_state_section = self._agent_state_section(agent)
        opponent_section = self._opponent_section(agent, context)
        relationship_section = self._relationship_section(agent, context)
        history_section = self._history_section(agent, context)
        environment_section = self._environment_section(agent)

        return (
            "You are making one strategic choice in a simulated society with pairwise interactions.\n\n"
            f"{agent_state_section}"
            f"{opponent_section}"
            f"{relationship_section}"
            f"\n"
            f"{history_section}"
            f"\n"
            f"{environment_section}"
            f"\n"
            "Incentives:\n"
            "  - Cooperation can increase total wealth when both sides cooperate, but you may be exploited if the other defects.\n"
            "  - Defection can protect your own position in the short term, but repeated defection reduces trust and can damage future gains.\n"
            "\n"
            "Based on this context, decide your next action.\n"
            "Respond with ONLY one word: cooperate or defect"
        )

    def _agent_state_section(self, agent) -> str:
        """Return a prompt section describing the deciding agent's own state."""
        resources = agent.resources
        rel_wealth = self._relative_wealth_section(agent)
        return (
            "Your state:\n"
            f"  - Current resources: {resources}\n"
            f"  - Relative wealth: {rel_wealth}\n"
            f"  - Baseline cooperation tendency: {agent.cooperation_tendency:.2f}\n"
            "\n"
        )

    def _relative_wealth_section(self, agent) -> str:
        """Describe wealth relative to peers when population data is available."""
        resources = getattr(agent, "resources", None)
        snapshot = getattr(agent, "population_resources_snapshot", None)
        if resources is None or not snapshot:
            return "unknown"

        avg_resources = mean(snapshot)
        if avg_resources == 0:
            return "at population average"

        ratio = resources / avg_resources
        if ratio >= 1.5:
            label = "well above average"
        elif ratio >= 1.1:
            label = "above average"
        elif ratio <= 0.7:
            label = "well below average"
        elif ratio <= 0.9:
            label = "below average"
        else:
            label = "near average"

        return f"{label} (you={resources:.2f}, avg={avg_resources:.2f}, ratio={ratio:.2f})"

    def _opponent_section(self, agent, context) -> str:
        """Return a prompt section describing the opponent."""
        if context is None or not hasattr(context, "agent_id"):
            return "Opponent state: unknown\n"

        opp_id = context.agent_id

        opp_resources = getattr(context, "resources", "unknown")
        gap = "unknown"
        if isinstance(opp_resources, (int, float)):
            gap = f"{agent.resources - opp_resources:.2f}"
        return (
            f"Opponent (agent {opp_id}) state:\n"
            f"  - Resources: {opp_resources}\n"
            f"  - Resource gap (you - opponent): {gap}\n"
        )

    def _relationship_section(self, agent, context) -> str:
        """Return trust and interaction context for the opponent."""
        if context is None or not hasattr(context, "agent_id"):
            return "Relationship context: unavailable\n"

        opp_id = context.agent_id
        memory_entry = agent.memory.get(opp_id, {})
        relationship_entry = agent.relationships.get(opp_id, {})

        trust_score = memory_entry.get("trust")
        if trust_score is None:
            trust_score = relationship_entry.get("trust")

        interactions = memory_entry.get("interactions")
        if interactions is None:
            interactions = relationship_entry.get("interaction_count", 0)

        last_outcome = memory_entry.get("last_outcome", "none")

        trust_str = f"{trust_score:.2f}" if trust_score is not None else "no prior history"
        return (
            "Relationship context:\n"
            f"  - Your trust toward opponent: {trust_str}\n"
            f"  - Past interactions with opponent: {interactions}\n"
            f"  - Last observed opponent action: {last_outcome}\n"
        )

    def _history_section(self, agent, context) -> str:
        """Return a prompt section summarising recent interaction history."""
        if context is None or not hasattr(context, "agent_id"):
            return "Interaction history with this agent: none\n"

        opp_id = context.agent_id
        memory_entry = agent.memory.get(opp_id, {})
        interactions = memory_entry.get("interactions", 0)
        observed_cooperate = memory_entry.get("cooperated", 0)
        observed_defect = memory_entry.get("defected", 0)

        if interactions == 0:
            return f"Interaction history with agent {opp_id}: none\n"

        return (
            f"Memory summary with agent {opp_id}:\n"
            f"  - Opponent cooperated {observed_cooperate} times\n"
            f"  - Opponent defected {observed_defect} times\n"
            f"  - Total interactions: {interactions}\n"
        )

    def _environment_section(self, agent) -> str:
        """Return environment settings relevant to incentives."""
        context = getattr(agent, "simulation_context", {})
        scarcity = context.get("scarcity_level", "unknown")
        redistribution = context.get("redistribution_strength", "unknown")
        elite_enabled = context.get("enable_elite_advantage", "unknown")
        return (
            "Environment conditions:\n"
            f"  - scarcity_level: {scarcity}\n"
            f"  - redistribution_strength: {redistribution}\n"
            f"  - elite_advantage_enabled: {elite_enabled}\n"
        )

    def _build_prompt_summary(self, agent, context) -> Dict[str, Any]:
        """Build a compact summary for downstream decision analytics logs."""
        opponent_id = context.agent_id if context is not None and hasattr(context, "agent_id") else None
        memory_entry = agent.memory.get(opponent_id, {}) if opponent_id is not None else {}
        recent_memory_summary = summarize_memory(agent, window=10)
        return {
            "agent_resources": agent.resources,
            "relative_wealth": self._relative_wealth_section(agent),
            "opponent_id": opponent_id,
            "opponent_resources": getattr(context, "resources", None) if context is not None else None,
            "trust": memory_entry.get("trust"),
            "interactions": memory_entry.get("interactions", 0),
            "last_outcome": memory_entry.get("last_outcome"),
            "cooperated": memory_entry.get("cooperated", 0),
            "defected": memory_entry.get("defected", 0),
            "recent_memory_summary": recent_memory_summary,
            "environment": getattr(agent, "simulation_context", {}),
        
        }

    def _build_batch_strategy_prompt(self, agent_batch: List[Any]) -> str:
        """Build one prompt that requests strategies for multiple agents."""
        lines = [
            "You are setting strategies for multiple agents in a simulated society.",
            "Return strict JSON only.",
            "Output format:",
            '{"strategies": {"<agent_id>": "cooperate|defect"}}',
            "",
            "Agents:",
        ]
        for agent in agent_batch:
            memory_stats = summarize_memory(agent, window=10)
            relationships = getattr(agent, "relationships", {})
            if relationships:
                avg_trust = sum(r["trust"] for r in relationships.values()) / len(relationships)
            else:
                avg_trust = 0.0
            lines.append(
                (
                    f"- agent_id={agent.agent_id} "
                    f"resources={agent.resources:.2f} "
                    f"recent_coop_rate={memory_stats.get('recent_coop_rate', 0.0):.2f} "
                    f"defect_rate={memory_stats.get('defect_rate', 0.0):.2f} "
                    f"avg_reward={memory_stats.get('avg_reward', 0.0):.2f} "
                    f"avg_trust={avg_trust:.2f}"
                )
            )
        return "\n".join(lines)

    def _parse_batch_response(self, text: str, agent_batch: List[Any]) -> Dict[Any, str]:
        """Parse batched strategy JSON into an agent_id -> action mapping."""
        cleaned = text.strip()
        # Models sometimes wrap JSON in ```json ... ``` despite instructions.
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from batch LLM response: {exc}") from exc

        strategies = payload.get("strategies", {}) if isinstance(payload, dict) else {}
        if not isinstance(strategies, dict):
            raise ValueError("Batch response missing 'strategies' object")

        valid_agent_ids = {str(agent.agent_id): agent.agent_id for agent in agent_batch}
        mapped: Dict[Any, str] = {}
        for raw_agent_id, raw_action in strategies.items():
            if str(raw_agent_id) not in valid_agent_ids:
                continue
            strategy = str(raw_action)
            strategy = strategy.strip().lower()
            action, used_fallback = self._parse_response_with_fallback(strategy)
            if used_fallback:
                self._fallback_count += 1
                self._fallback_agent_decisions += 1
            else:
                self._record_llm_success(action)
            mapped[valid_agent_ids[str(raw_agent_id)]] = action
        return mapped
    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, max_tokens: int = 20, expect_json: bool = False) -> str:
        """Sync helper used by legacy sync policy methods.

        max_tokens is kept for compatibility with existing call sites.
        """
        del max_tokens  # handled internally by providers when needed
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.provider.generate(prompt, expect_json=expect_json))

        # If called inside an existing loop, run provider generation in a
        # dedicated worker thread with its own event loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: asyncio.run(self.provider.generate(prompt, expect_json=expect_json))
            )
            return future.result()

    async def _call_llm_async(
        self,
        prompt: str,
        max_tokens: int = 20,
        expect_json: bool = False,
    ) -> str:
        """Async LLM call with retry, exponential backoff, and multi-model fallback.
        
        Strategy:
        1. Try each model in llm_models with retry+backoff
        2. On RateLimitError or TimeoutError, retry with exponential backoff
        3. On ModelDecommissionedError or InvalidAPIKeyError, skip to next model
        4. Return first successful response
        5. Only fallback to default action after all models and retries exhausted
        """
        del max_tokens  # handled internally by providers when needed
        
        models_to_try = list(self.llm_models) if self.llm_models else ["llama3"]
        last_error = None
        
        for model_idx, model in enumerate(models_to_try):
            # Create provider instance for this model
            if model == self.provider.model:
                provider = self.provider
            else:
                # Switch to alternate model - for now retry with same provider
                # In a real scenario, you might create new provider instances
                # but for simplicity we'll use the current one
                logger.debug("Attempting fallback to model %s", model)
                provider = self.provider
            
            for attempt in range(_MAX_RETRIES):
                try:
                    if self.debug_llm and attempt > 0:
                        logger.info(
                            "LLM retry model=%s attempt=%d/%d",
                            model,
                            attempt + 1,
                            _MAX_RETRIES,
                        )
                    
                    async with self._semaphore:
                        response = await asyncio.wait_for(
                            provider.generate(prompt, expect_json=expect_json),
                            timeout=self.timeout,
                        )
                    
                    if attempt > 0:
                        self._llm_retry_count += 1
                        logger.info(
                            "LLM retry succeeded after %d attempt(s) with model %s",
                            attempt + 1,
                            model,
                        )
                    
                    return response
                
                except RateLimitError as exc:
                    last_error = exc
                    if attempt < _MAX_RETRIES - 1:
                        # Calculate backoff: 2^attempt seconds
                        backoff_seconds = _RETRY_BACKOFF_BASE_RATE_LIMIT ** attempt
                        # Add jitter to avoid thundering herd
                        jitter = 0.1 * attempt
                        total_backoff = backoff_seconds + jitter
                        
                        logger.warning(
                            "Rate limit on model %s. Retry %d/%d after %.2f seconds.",
                            model,
                            attempt + 1,
                            _MAX_RETRIES,
                            total_backoff,
                        )
                        self._llm_retry_count += 1
                        await asyncio.sleep(total_backoff)
                    else:
                        logger.error(
                            "Rate limit exhausted on model %s after %d retries.",
                            model,
                            _MAX_RETRIES,
                        )
                
                except (LLMTimeoutError, asyncio.TimeoutError) as exc:
                    last_error = exc
                    if attempt < _MAX_RETRIES - 1:
                        # Calculate backoff: 1.5^attempt seconds (more conservative)
                        backoff_seconds = _RETRY_BACKOFF_BASE_TIMEOUT ** attempt
                        jitter = 0.05 * attempt
                        total_backoff = backoff_seconds + jitter
                        
                        logger.warning(
                            "Timeout on model %s. Retry %d/%d after %.2f seconds.",
                            model,
                            attempt + 1,
                            _MAX_RETRIES,
                            total_backoff,
                        )
                        self._llm_retry_count += 1
                        await asyncio.sleep(total_backoff)
                    else:
                        logger.error(
                            "Timeout exhausted on model %s after %d retries.",
                            model,
                            _MAX_RETRIES,
                        )
                
                except (ModelDecommissionedError, InvalidAPIKeyError) as exc:
                    last_error = exc
                    logger.warning(
                        "Model %s unavailable (%s). Trying next model in fallback list.",
                        model,
                        exc.error_type,
                    )
                    break  # Break retry loop, try next model
                
                except Exception as exc:
                    last_error = exc
                    logger.error(
                        "Unexpected error on model %s: %s",
                        model,
                        str(exc),
                        exc_info=True,
                    )
                    if attempt < _MAX_RETRIES - 1:
                        await asyncio.sleep(1.0)
                        self._llm_retry_count += 1
        
        # All models and retries exhausted
        logger.error(
            "All LLM models exhausted. Last error: %s",
            str(last_error),
        )
        raise last_error or Exception("LLM call failed: all models exhausted")

    def _batch_max_tokens(self, batch_len: int) -> int:
        """Return a token budget suitable for batched JSON strategy responses."""
        # Keep single-agent calls compact while giving batched JSON enough room.
        return max(128, 24 * max(1, batch_len))

    def _record_latency(self, latency_seconds: float) -> None:
        """Update running latency counters used for dashboard metrics."""
        self._llm_total_latency_seconds += latency_seconds
        self._llm_latency_samples += 1

    def get_avg_llm_latency(self) -> float:
        """Return average LLM call latency in seconds."""
        if self._llm_latency_samples == 0:
            return 0.0
        return self._llm_total_latency_seconds / self._llm_latency_samples
    
    def get_llm_retry_count(self) -> int:
        """Return total number of LLM retries that occurred."""
        return self._llm_retry_count
    
    def get_llm_metrics(self) -> Dict[str, Any]:
        """Return comprehensive LLM metrics for dashboard/logging."""
        total_decisions = self._total_agent_decisions
        success_decisions = self._success_agent_decisions
        fallback_decisions = self._fallback_agent_decisions
        success_rate = (
            (success_decisions / total_decisions)
            if total_decisions > 0
            else 0.0
        )
        fallback_rate = (
            (fallback_decisions / total_decisions) 
            if total_decisions > 0 
            else 0.0
        )

        logger.info(
            f"Agent Decisions: {total_decisions}, Success: {success_decisions}, Fallback: {fallback_decisions}"
        )
        
        return {
            "total_llm_calls": self._llm_call_count,
            "llm_error_count": self._llm_error_count,
            "llm_retry_count": self._llm_retry_count,
            "total_agent_decisions": total_decisions,
            "success_agent_decisions": success_decisions,
            "fallback_agent_decisions": fallback_decisions,
            "success_rate": success_rate,
            "fallback_rate": fallback_rate,
            "avg_llm_latency_seconds": self.get_avg_llm_latency(),
            "configured_models": self.llm_models,
            "max_retries": _MAX_RETRIES,
            "max_concurrency": self.max_concurrent_llm_calls,
        }

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response_with_fallback(self, text: str) -> Tuple[str, bool]:
        """Extract ``"cooperate"`` or ``"defect"`` from *text*.

        The parser first attempts a strict word-boundary match so that
        responses such as ``"I will not cooperate, I will defect"`` resolve to
        ``"defect"`` rather than the first substring hit.  It falls back to a
        simple substring scan when no word-boundary match is found, and
        finally returns ``_FALLBACK_ACTION`` when neither keyword is present.

        Args:
            text: Raw text returned by the LLM.

        Returns:
            Tuple ``(action, used_fallback)`` where action is ``"cooperate"`` or
            ``"defect"``.
        """
        strategy = text.strip().lower()
        normalised = strategy

        # Try to find the *last* occurrence of either keyword as a whole word.
        # "last" is used because LLMs sometimes prepend filler text and end
        # with the actual decision.
        matches = list(re.finditer(r"\b(cooperate|defect)\b", normalised))
        if matches:
            return matches[-1].group(1), False

        self._log_fallback(f"reason=unparseable_response detail={text!r}")
        return _FALLBACK_ACTION, True
