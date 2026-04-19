from typing import Any, Dict, TypedDict

from simulation.policies.deterministic_policy import DeterministicPolicy

# Shared default policy instance — policies are stateless with respect to
# individual agents, so a single instance can safely be reused across all
# agents that do not receive an explicit policy at construction time.
_DEFAULT_POLICY = DeterministicPolicy()


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
        self.memory_log = []
        self.relationships: Dict[Any, RelationshipRecord] = {}
        self.memory: Dict[Any, Dict[str, Any]] = {}

        # Maintain backwards compatibility
        self.id = agent_id
        self.alive = True

        # Assign policy — falls back to the module-level shared default so
        # that no extra allocation is needed for the common case.
        self.policy = policy if policy is not None else _DEFAULT_POLICY
    
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
        old_trust = rel["trust"]
        rel["trust"] = max(0.0, min(1.0, rel["trust"] + delta))
        self.memory_log.append({
            "action": "trust_update",
            "other_agent_id": other_id,
            "trust_before": old_trust,
            "trust_after": rel["trust"],
            "delta": delta
        })
    
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
            self.memory[other_agent_id] = {"trust": 0.0, "interactions": 0}
        entry = self.memory[other_agent_id]
        entry["interactions"] += 1
        if outcome == "cooperate":
            entry["trust"] = max(-1.0, min(1.0, entry["trust"] + 0.1))
        elif outcome == "defect":
            entry["trust"] = max(-1.0, min(1.0, entry["trust"] - 0.1))

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
            self.memory_log.append({
                "action": "trade",
                "status": "failed",
                "reason": "invalid_other_agent",
                "attempted_amount": trade_amount
            })
            return False
        
        # Handle zero-amount trades (no-op but still logged; no relationship update since no
        # actual exchange occurred)
        if trade_amount == 0:
            self.memory_log.append({
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
            self.memory_log.append({
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
            self.memory_log.append({
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
        self.memory_log.append({
            "action": "trade",
            "status": "success",
            "other_agent_id": other_agent.agent_id,
            "amount": trade_amount,
            "my_resources_after": self.resources,
            "other_agent_resources_after": other_agent.resources
        })
        
        other_agent.memory_log.append({
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
        change is validated and recorded in `memory_log`.
        
        Args:
            amount: Positive number of resources to add
            source: Optional label identifying the origin of the reward
        """
        if amount <= 0:
            return
        self.resources += amount
        self.memory_log.append({
            "action": "receive_resource",
            "amount": amount,
            "source": source,
            "my_resources_after": self.resources
        })
    
    def communicate(self, other_agent, message):
        """
        Store communication events in memory_log.
        
        Args:
            other_agent: The agent to communicate with
            message: The message content to send
        """
        # Log the communication event
        self.memory_log.append({
            "action": "communicate",
            "other_agent_id": other_agent.agent_id if (other_agent and hasattr(other_agent, 'agent_id')) else None,
            "message": message,
            "my_resources": self.resources
        })
        
        # Optionally log receipt on other agent's memory
        if other_agent and hasattr(other_agent, 'memory_log') and hasattr(other_agent, 'agent_id'):
            other_agent.memory_log.append({
                "action": "received_communication",
                "from_agent_id": self.agent_id,
                "message": message,
                "my_resources": other_agent.resources
            })
