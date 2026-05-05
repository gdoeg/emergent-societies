import random
import asyncio
from typing import Any, Dict, List, TypedDict

# DeterministicPolicy (and all AgentPolicy subclasses) must never import from
# simulation.agent to avoid a circular dependency.  The dependency direction is
# always: agent -> policy, never policy -> agent.
from simulation.policies.deterministic_policy import DeterministicPolicy

class RelationshipRecord(TypedDict):
    """Per-partner relationship state stored in Agent.relationships."""
    trust: float
    interaction_count: int


class Agent:
    """
    Agent class for multi-agent simulation.

    Models autonomous agents interacting in a shared environment.
    Decision-making is delegated to a pluggable :attr:`policy` object so
    that deterministic, stochastic, or LLM-driven strategies can be swapped
    without changing this class.
    """

    def __init__(self, agent_id, resources=10, cooperation_tendency=0.5, policy=None):
        """
        Initialize an Agent.

        Args:
            agent_id: Unique identifier for the agent.
            resources: Initial numeric value representing owned resources (default: 10).
            cooperation_tendency: Float between 0 and 1 representing likelihood to
                cooperate when no memory exists (default: 0.5).
            policy: An :class:`~simulation.policies.base.AgentPolicy` instance that
                implements ``decide(agent, context) -> str``.  Defaults to
                :class:`~simulation.policies.deterministic_policy.DeterministicPolicy`
                when ``None`` is supplied.
        """
        self.agent_id = agent_id
        self.resources = resources
        self.cooperation_tendency = max(0.0, min(1.0, cooperation_tendency))
        self.relationships: Dict[Any, RelationshipRecord] = {}
        self.memory: Dict[Any, Dict[str, Any]] = {}

        # Maintain backwards compatibility
        self.id = agent_id
        self.alive = True

        # Keep policy state isolated per agent by assigning a distinct policy
        # instance whenever one is not explicitly provided.
        self.policy = policy if policy is not None else DeterministicPolicy()

        # Environment-derived context injected by Environment; used by policies
        # to condition decisions without requiring shared policy state.
        self.simulation_context: Dict[str, Any] = {}
        self.population_resources_snapshot = []

        # --- Agent persona (fixed at creation for heterogeneity) ---
        # These traits are injected into LLM prompts to produce consistent,
        # differentiated decision-making behaviour across agents.
        self.risk_tolerance: str = random.choice(["low", "medium", "high"])
        self.social_preference: str = random.choice(["cooperative", "selfish", "mixed"])
        self.memory_bias: str = random.choice(["forgiving", "retaliatory", "neutral"])
        self.goal: str = random.choice(["maximize_wealth", "maintain_fairness", "balance"])

        # --- Periodic strategy model ---
        # Standing strategy used for all interactions until the next LLM update.
        # Stochastic seeding reduces deterministic bias while remaining reproducible
        # when the simulation's random seed is fixed.
        self.strategy: str = "cooperate" if random.random() < self.cooperation_tendency else "defect"
        # Simulation step at which strategy was last refreshed (-1 = never updated).
        self.last_strategy_update_step: int = -1
        # Flat interaction log used as LLM context for strategy updates.
        # Structure per entry: {step, opponent_id, action, opponent_action, reward}
        self.interaction_memory: List[Dict[str, Any]] = []

        # --- Decision tracking metrics ---
        # Sequence of LLM decisions for volatility analysis (oldest first).
        self.decision_history: List[str] = []
        # Sequence of LLM confidence scores aligned with decision_history.
        self.confidence_history: List[float] = []
    
    def get_relationship(self, other_id) -> RelationshipRecord:
        """
        Return the relationship record for the given agent ID, creating a default if absent.
        
        Args:
            other_id: The agent_id of the other agent
            
        Returns:
            RelationshipRecord: {"trust": float, "interaction_count": int}
        """
        if other_id not in self.relationships:
            self.relationships[other_id] = {"trust": 0.5, "interaction_count": 0}
        return self.relationships[other_id]
    
    def update_trust(self, other_id, delta):
        """
        Adjust trust toward other_id by delta, clamped to [0.0, 1.0].
        
        Args:
            other_id: The agent_id of the other agent
            delta: Amount to add to current trust (positive or negative)
        """
        rel = self.get_relationship(other_id)
        rel["trust"] = max(0.0, min(1.0, rel["trust"] + delta))
    
    def record_interaction(self, other_id):
        """
        Increment the interaction count for the given agent ID.
        
        Args:
            other_id: The agent_id of the other agent
        """
        self.get_relationship(other_id)["interaction_count"] += 1
    
    def update_memory(self, other_agent_id, outcome: str):
        """
        Record an interaction outcome and update trust toward the other agent.

        Args:
            other_agent_id: The agent_id of the other agent
            outcome: "cooperate" or "defect" — the action taken by the other agent
        """
        if other_agent_id not in self.memory:
            self.memory[other_agent_id] = {
                "trust": 0.0,
                "interactions": 0,
                "cooperated": 0,
                "defected": 0,
                "last_outcome": None,
            }
        entry = self.memory[other_agent_id]
        entry.setdefault("cooperated", 0)
        entry.setdefault("defected", 0)
        entry.setdefault("last_outcome", None)
        entry["interactions"] += 1
        if outcome == "cooperate":
            entry["trust"] = max(-1.0, min(1.0, entry["trust"] + 0.1))
            entry["cooperated"] += 1
        elif outcome == "defect":
            entry["trust"] = max(-1.0, min(1.0, entry["trust"] - 0.1))
            entry["defected"] += 1
        entry["last_outcome"] = outcome

    def decide_action(self, context):
        """Decide whether to cooperate or defect by delegating to :attr:`policy`.

        Args:
            context: The agent to interact with, or a world object for backwards
                compatibility.  Passed unchanged to
                :meth:`~simulation.policies.base.AgentPolicy.decide`.

        Returns:
            str: ``"cooperate"`` or ``"defect"``.
        """
        if not self.alive:
            return "defect"

        if self.resources <= 0:
            self.alive = False
            return "defect"

        return self.policy.decide(self, context)

    def get_action(self, opponent) -> str:
        """Return the agent's current standing strategy without an LLM call.

        This is the fast interaction path used during every simulation step.
        Strategy is only updated periodically via :meth:`maybe_update_strategy`.

        Args:
            opponent: The other agent in the interaction (unused — strategy is
                not opponent-specific, but accepted for interface consistency).

        Returns:
            ``"cooperate"`` or ``"defect"``.
        """
        if not self.alive or self.resources <= 0:
            self.alive = False
            return "defect"
        return self.strategy

    def maybe_update_strategy(
        self,
        step: int,
        decision_interval: int,
        llm_policy=None,
    ) -> None:
        """Refresh :attr:`strategy` via LLM when the update interval has elapsed.

        LLM calls are concentrated here (rare) rather than on every interaction
        (frequent) so that 100-agent simulations remain tractable.

        Args:
            step: Current simulation step counter.
            decision_interval: Minimum number of steps between LLM strategy
                updates.  Matches ``SimulationConfig.decision_interval``.
            llm_policy: Optional policy override.  When ``None`` (the normal
                case), the agent's own :attr:`policy` is used.  Pass an
                explicit policy when testing with a mock policy or when the
                same shared LLM policy should drive multiple agents.  Must
                implement ``generate_strategy(agent) -> str``.
        """
        if step - self.last_strategy_update_step < decision_interval:
            return

        policy = llm_policy if llm_policy is not None else self.policy
        if hasattr(policy, "generate_strategy"):
            new_strategy = policy.generate_strategy(self)
        else:
            # Fallback for deterministic/non-LLM policies.
            new_strategy = policy.decide(self, None)

        if new_strategy != self.strategy:
            # Lower switch probability increases behavioral persistence.
            if random.random() < 0.3:
                self.strategy = new_strategy
        self.last_strategy_update_step = step

    async def maybe_update_strategy_async(
        self,
        step: int,
        decision_interval: int,
        llm_policy=None,
    ) -> None:
        """Async variant of strategy refresh, preserving sync compatibility."""
        if step - self.last_strategy_update_step < decision_interval:
            return

        policy = llm_policy if llm_policy is not None else self.policy
        if hasattr(policy, "generate_strategy_async"):
            try:
                new_strategy = await policy.generate_strategy_async(self)
            except Exception:
                # Keep simulation non-blocking when async policy call fails.
                new_strategy = self.strategy
        elif hasattr(policy, "generate_strategy"):
            new_strategy = policy.generate_strategy(self)
        else:
            if hasattr(policy, "decide"):
                maybe_value = policy.decide(self, None)
                if asyncio.iscoroutine(maybe_value):
                    new_strategy = await maybe_value
                else:
                    new_strategy = maybe_value
            else:
                new_strategy = self.strategy

        if new_strategy != self.strategy and random.random() < 0.3:
            self.strategy = new_strategy
        self.last_strategy_update_step = step

    def get_strategy_history(self) -> List[str]:
        """Return the sequence of actions from :attr:`interaction_memory`.

        Useful for analysis — shows how the agent actually behaved over time
        regardless of what its current standing strategy is.

        Returns:
            List of ``"cooperate"`` / ``"defect"`` strings, oldest first.
        """
        return [entry["action"] for entry in self.interaction_memory]
    
    def trade(self, other_agent, trade_amount):
        """
        Transfer resources between agents.
        
        Args:
            other_agent: The agent to trade with
            trade_amount: Amount of resources to transfer (positive = give, negative = receive)
                         Zero-amount trades are allowed but create no actual resource transfer
            
        Returns:
            bool: True if trade was successful, False otherwise
        """
        # Validate other_agent is a valid Agent object
        if not other_agent or not hasattr(other_agent, 'resources') or not hasattr(other_agent, 'agent_id'):
            self.interaction_memory.append({
                "action": "trade",
                "status": "failed",
                "reason": "invalid_other_agent",
                "attempted_amount": trade_amount
            })
            return False
        
        # Handle zero-amount trades (no-op but still logged; no relationship update since no
        # actual exchange occurred)
        if trade_amount == 0:
            self.interaction_memory.append({
                "action": "trade",
                "status": "success",
                "other_agent_id": other_agent.agent_id,
                "amount": 0,
                "my_resources_after": self.resources,
                "other_agent_resources_after": other_agent.resources,
                "note": "zero_amount_trade"
            })
            return True
        
        # Validate trade amount
        if trade_amount > self.resources:
            # Cannot trade more resources than owned
            self.interaction_memory.append({
                "action": "trade",
                "status": "failed",
                "reason": "insufficient_resources",
                "other_agent_id": other_agent.agent_id,
                "attempted_amount": trade_amount,
                "my_resources": self.resources
            })
            return False
        
        if trade_amount < 0 and abs(trade_amount) > other_agent.resources:
            # Other agent doesn't have enough resources
            self.interaction_memory.append({
                "action": "trade",
                "status": "failed",
                "reason": "other_agent_insufficient_resources",
                "other_agent_id": other_agent.agent_id,
                "attempted_amount": trade_amount,
                "other_agent_resources": other_agent.resources
            })
            return False
        
        # Execute trade
        self.resources -= trade_amount
        other_agent.resources += trade_amount
        
        # Log successful trade
        self.interaction_memory.append({
            "action": "trade",
            "status": "success",
            "other_agent_id": other_agent.agent_id,
            "amount": trade_amount,
            "my_resources_after": self.resources,
            "other_agent_resources_after": other_agent.resources
        })
        
        other_agent.interaction_memory.append({
            "action": "trade",
            "status": "success",
            "other_agent_id": self.agent_id,
            "amount": -trade_amount,  # From their perspective
            "my_resources_after": other_agent.resources,
            "other_agent_resources_after": self.resources
        })
        
        return True
    
    def receive_resource(self, amount, source=None):
        """
        Add resources to this agent from an external source (e.g. environment reward).
        
        Use this instead of mutating `resources` directly so that every resource
        change is validated and recorded in `interaction_memory`.
        
        Args:
            amount: Positive number of resources to add
            source: Optional label identifying the origin of the reward
        """
        if amount <= 0:
            return
        self.resources += amount
        self.interaction_memory.append({
            "action": "receive_resource",
            "amount": amount,
            "source": source,
            "my_resources_after": self.resources
        })
    
    def communicate(self, other_agent, message):
        """
        Store communication events in interaction_memory.
        
        Args:
            other_agent: The agent to communicate with
            message: The message content to send
        """
        # Log the communication event
        self.interaction_memory.append({
            "action": "communicate",
            "other_agent_id": other_agent.agent_id if (other_agent and hasattr(other_agent, 'agent_id')) else None,
            "message": message,
            "my_resources": self.resources
        })
        
        # Optionally log receipt on other agent's interaction memory.
        if other_agent and hasattr(other_agent, 'interaction_memory') and hasattr(other_agent, 'agent_id'):
            other_agent.interaction_memory.append({
                "action": "received_communication",
                "from_agent_id": self.agent_id,
                "message": message,
                "my_resources": other_agent.resources
            })
