"""LLM-based agent policy using an OpenAI-compatible chat API."""

import json
import logging
import os
from typing import Any, Dict, Optional
from urllib import request as urllib_request
from urllib.error import URLError

from simulation.policies.base import AgentPolicy

logger = logging.getLogger(__name__)

# Default values — can be overridden via SimulationConfig or environment variables.
_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
_FALLBACK_ACTION = "cooperate"


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
        policy_logger=None,
    ) -> None:
        self.model = model
        self.api_base_url = api_base_url.rstrip("/")
        self._policy_logger = policy_logger

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def decide(self, agent, context) -> str:
        """Return ``"cooperate"`` or ``"defect"`` via an LLM call.

        Falls back to ``"cooperate"`` if the API call fails so the simulation
        can continue gracefully without a live LLM.

        Args:
            agent: The :class:`~simulation.agent.Agent` making the decision.
            context: The opponent agent (or world object).  Agents with an
                ``agent_id`` attribute provide richer prompt context.

        Returns:
            ``"cooperate"`` or ``"defect"``.
        """
        prompt = self._build_prompt(agent, context)
        raw_response: Optional[str] = None
        action = _FALLBACK_ACTION

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
            "prompt": prompt,
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

        return action

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
        opponent_section = self._opponent_section(agent, context)
        history_section = self._history_section(agent, context)

        return (
            "You are an agent in a simulated society where agents cooperate or "
            "defect in pairwise interactions.\n\n"
            f"Your current state:\n"
            f"  - Resources: {agent.resources}\n"
            f"  - Cooperation tendency: {agent.cooperation_tendency:.2f}\n"
            f"\n"
            f"{opponent_section}"
            f"\n"
            f"{history_section}"
            f"\n"
            "Based on this context, decide your next action.\n"
            "Respond with ONLY one word: cooperate or defect"
        )

    def _opponent_section(self, agent, context) -> str:
        """Return a prompt section describing the opponent."""
        if context is None or not hasattr(context, "agent_id"):
            return "Opponent state: unknown\n"

        opp_id = context.agent_id
        trust_score: Optional[float] = None

        # Look up trust from agent's memory (range [-1, 1])
        if opp_id in agent.memory:
            trust_score = agent.memory[opp_id]["trust"]
        # Also check relationships dict (range [0, 1])
        elif opp_id in agent.relationships:
            trust_score = agent.relationships[opp_id]["trust"]

        opp_resources = getattr(context, "resources", "unknown")
        trust_str = (
            f"{trust_score:.2f}" if trust_score is not None else "no prior history"
        )
        return (
            f"Opponent (agent {opp_id}) state:\n"
            f"  - Resources: {opp_resources}\n"
            f"  - Your trust toward them: {trust_str}\n"
        )

    def _history_section(self, agent, context) -> str:
        """Return a prompt section summarising recent interaction history."""
        if context is None or not hasattr(context, "agent_id"):
            return "Interaction history with this agent: none\n"

        opp_id = context.agent_id
        relevant = [
            entry
            for entry in agent.memory_log
            if entry.get("other_agent_id") == opp_id
            and entry.get("action") == "decide_action"
        ]

        if not relevant:
            return f"Interaction history with agent {opp_id}: none\n"

        # Summarise the last 5 interactions
        recent = relevant[-5:]
        cooperate_count = sum(1 for e in recent if e.get("decision") == "cooperate")
        defect_count = len(recent) - cooperate_count
        return (
            f"Recent interaction history with agent {opp_id} "
            f"(last {len(recent)} interactions):\n"
            f"  - Times you cooperated: {cooperate_count}\n"
            f"  - Times you defected: {defect_count}\n"
        )

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
        if not api_key:
            raise EnvironmentError(
                "No LLM API key found. Set the OPENAI_API_KEY or LLM_API_KEY "
                "environment variable."
            )

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.0,
            }
        ).encode("utf-8")

        req = urllib_request.Request(
            f"{self.api_base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        with urllib_request.urlopen(req, timeout=15) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))

        return body["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, text: str) -> str:
        """Extract ``"cooperate"`` or ``"defect"`` from *text*.

        The parser is intentionally lenient — it checks whether the response
        contains either keyword (case-insensitive) and returns the first
        match.  Falls back to ``_FALLBACK_ACTION`` when neither is found.

        Args:
            text: Raw text returned by the LLM.

        Returns:
            ``"cooperate"`` or ``"defect"``.
        """
        normalised = text.strip().lower()
        if "cooperate" in normalised:
            return "cooperate"
        if "defect" in normalised:
            return "defect"
        logger.warning(
            "LLMPolicy: could not parse response %r — defaulting to '%s'",
            text,
            _FALLBACK_ACTION,
        )
        return _FALLBACK_ACTION
