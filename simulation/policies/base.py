"""Base policy interface for agent decision-making."""


class AgentPolicy:
    """Abstract base class for agent decision policies.

    Subclasses must implement :meth:`decide` to return either ``"cooperate"``
    or ``"defect"`` given the current agent and interaction context.
    """

    def decide(self, agent, context) -> str:
        """Decide an action for *agent* given *context*.

        Args:
            agent: The :class:`~simulation.agent.Agent` making the decision.
            context: Arbitrary context object passed from the simulation
                (e.g. the opponent agent or world state).

        Returns:
            ``"cooperate"`` or ``"defect"``.

        Raises:
            NotImplementedError: Always — subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement decide()")
