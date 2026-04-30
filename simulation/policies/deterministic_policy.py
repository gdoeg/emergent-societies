"""Deterministic (stochastic-tendency) policy — mirrors the original decide_action logic."""

import random

from simulation.policies.base import AgentPolicy


class DeterministicPolicy(AgentPolicy):
    """Replicates the original stochastic decision logic from :class:`~simulation.agent.Agent`.

    Decision is based on:
    - Memory trust toward the opponent (if available), biased around 0.5.
    - Agent's own ``cooperation_tendency`` as a fallback.

    Multipliers are preserved from the original implementation:
    - ``_MEMORY_BIAS_SCALE = 0.3`` — maps memory trust ``[-1, 1]`` to a
      ``±30%`` shift from the 50/50 baseline.
    """

    # Maps memory trust [-1, 1] to a ±30% shift from the 50/50 baseline.
    _MEMORY_BIAS_SCALE = 0.3

    def decide(self, agent, context) -> str:
        """Return ``"cooperate"`` or ``"defect"`` using stochastic tendency logic.

        Args:
            agent: The :class:`~simulation.agent.Agent` making the decision.
            context: The opponent agent (or world object for backwards
                compatibility).  Only agents that have an ``agent_id``
                attribute are used for memory look-ups.

        Returns:
            ``"cooperate"`` or ``"defect"``.
        """
        other_agent_id = None
        if context and hasattr(context, "agent_id"):
            other_agent_id = context.agent_id

        if other_agent_id is not None and other_agent_id in agent.memory:
            trust = agent.memory[other_agent_id]["trust"]
            threshold = max(0.0, min(1.0, 0.5 + trust * self._MEMORY_BIAS_SCALE))
            decision = "cooperate" if random.random() < threshold else "defect"
        else:
            decision = (
                "cooperate"
                if random.random() < agent.cooperation_tendency
                else "defect"
            )

        agent.interaction_memory.append(
            {
                "action": "decide_action",
                "policy": "deterministic",
                "decision": decision,
                "other_agent_id": other_agent_id,
                "my_resources": agent.resources,
            }
        )

        return decision
