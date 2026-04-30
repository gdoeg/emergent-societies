"""LLM-based agent policy using an OpenAI-compatible chat API."""

import json
import logging
import os
import re
from statistics import mean
from typing import Any, Dict, Optional, Tuple
from urllib import request as urllib_request
from urllib.error import URLError

from simulation.policies.base import AgentPolicy

logger = logging.getLogger(__name__)

# Default values — can be overridden via SimulationConfig or environment variables.
_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
_FALLBACK_ACTION = "cooperate"
# Maximum number of cached decisions kept in memory per policy instance.
_MAX_CACHE_SIZE = 1024


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
        model: str = _DEFAULT_MODEL,
        api_base_url: str = _DEFAULT_API_BASE_URL,
        timeout: int = 4,
        policy_logger=None,
        decision_interval: int = 4,
    ) -> None:
        self.model = model
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self._policy_logger = policy_logger
        # Throttling: only call LLM every decision_interval invocations per agent;
        # reuse the last decision in between to reduce API load.
        self.decision_interval = decision_interval
        self._call_counters: Dict[Any, int] = {}          # agent_id -> total call count
        self._throttle_cache: Dict[Any, str] = {}         # agent_id -> last action
        # Lightweight in-memory decision cache keyed by a minimal state tuple.
        self._decision_cache: Dict[tuple, str] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def decide(self, agent, context) -> str:
        """Return ``"cooperate"`` or ``"defect"`` via an LLM call.

        Falls back to ``"cooperate"`` if the API call fails so the simulation
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
        other_agent_id = (
            context.agent_id if context is not None and hasattr(context, "agent_id") else None
        )
        prompt_summary = self._build_prompt_summary(agent, context)

        try:
            raw_response = self._call_llm(prompt)
            action = self._parse_response(raw_response)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLMPolicy: decision failed for agent %s — falling back to '%s'. "
                "Error: %s",
                agent.agent_id,
                _FALLBACK_ACTION,
                exc,
            )
            action = _FALLBACK_ACTION

        log_record: Dict[str, Any] = {
            "action": "decide_action",
            "policy": "llm",
            "agent_id": agent.agent_id,
            "other_agent_id": other_agent_id,
            "prompt": prompt,
            "prompt_summary": prompt_summary,
            "raw_response": raw_response,
            "decision": action,
            "my_resources": agent.resources,
        }

        # Write to the structured policy logger (if provided), then also
        # append a lightweight copy to the agent's own memory_log.
        if self._policy_logger is not None:
            self._policy_logger(log_record)
        else:
            logger.info(
                "LLMPolicy decision: agent_id=%s decision=%s",
                agent.agent_id,
                action,
            )
            logger.debug("LLMPolicy full record: %s", json.dumps(log_record))

        agent.memory_log.append(log_record)

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

        try:
            raw_response = self._call_llm(prompt)
            action = self._parse_response(raw_response)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLMPolicy.generate_strategy: failed for agent %s — keeping '%s'. Error: %s",
                agent.agent_id,
                _FALLBACK_ACTION,
                exc,
            )

        log_record: Dict[str, Any] = {
            "action": "strategy_update",
            "policy": "llm",
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

        agent.memory_log.append(log_record)
        return action

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

        # Recent interactions: at most the last 10 entries from interaction_memory
        recent = list(getattr(agent, "interaction_memory", []))[-10:]
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
            "environment": getattr(agent, "simulation_context", {}),
        
        }
    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Send *prompt* to the chat-completion API and return the reply text.

        Args:
            prompt: The user-turn message to send.

        Returns:
            The raw text content of the first assistant message.

        Raises:
            EnvironmentError: If no API key is found in the environment.
            URLError: If the HTTP request fails.
            ValueError: If the response cannot be parsed.
        """
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        is_local = self.api_base_url.startswith("http://localhost") or self.api_base_url.startswith(
            "http://127.0.0.1"
        )
        if not api_key and not is_local:
            raise EnvironmentError(
                "No LLM API key found. Set the OPENAI_API_KEY or LLM_API_KEY "
                "environment variable."
            )

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20,
                "temperature": 0.0,
            }
        ).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib_request.Request(
            f"{self.api_base_url}/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )

        with urllib_request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))

        return body["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, text: str) -> str:
        """Extract ``"cooperate"`` or ``"defect"`` from *text*.

        The parser first attempts a strict word-boundary match so that
        responses such as ``"I will not cooperate, I will defect"`` resolve to
        ``"defect"`` rather than the first substring hit.  It falls back to a
        simple substring scan when no word-boundary match is found, and
        finally returns ``_FALLBACK_ACTION`` when neither keyword is present.

        Args:
            text: Raw text returned by the LLM.

        Returns:
            ``"cooperate"`` or ``"defect"``.
        """
        normalised = text.strip().lower()

        # Try to find the *last* occurrence of either keyword as a whole word.
        # "last" is used because LLMs sometimes prepend filler text and end
        # with the actual decision.
        matches = list(re.finditer(r"\b(cooperate|defect)\b", normalised))
        if matches:
            return matches[-1].group(1)

        logger.warning(
            "LLMPolicy: could not parse response %r — defaulting to '%s'",
            text,
            _FALLBACK_ACTION,
        )
        return _FALLBACK_ACTION
