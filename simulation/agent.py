class Agent:
    """
    Minimal deterministic Agent class for multi-agent simulation.
    
    Models autonomous agents interacting in a shared environment with
    deterministic decision-making based on cooperation tendencies.
    """
    
    def __init__(self, agent_id, resources=10, cooperation_tendency=0.5):
        """
        Initialize an Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            resources: Initial numeric value representing owned resources (default: 10)
            cooperation_tendency: Float between 0 and 1 representing likelihood to cooperate (default: 0.5)
        """
        self.agent_id = agent_id
        self.resources = resources
        self.cooperation_tendency = max(0.0, min(1.0, cooperation_tendency))  # Clamp between 0 and 1
        self.memory_log = []
        
        # Maintain backwards compatibility
        self.id = agent_id
        self.alive = True
    
    def decide_action(self, other_agent):
        """
        Deterministically decide whether to cooperate or defect with another agent.
        
        Args:
            other_agent: The agent to interact with (or world object for backwards compatibility)
            
        Returns:
            str: "cooperate" if cooperation_tendency >= 0.5, "defect" otherwise
        """
        if not self.alive:
            return "defect"
        
        if self.resources <= 0:
            self.alive = False
            return "defect"
        
        # Deterministic decision based on cooperation_tendency
        if self.cooperation_tendency >= 0.5:
            decision = "cooperate"
        else:
            decision = "defect"
        
        # Get other_agent_id safely (handle both Agent objects and other types like World)
        other_agent_id = None
        if other_agent and hasattr(other_agent, 'agent_id'):
            other_agent_id = other_agent.agent_id
        
        # Log the decision
        self.memory_log.append({
            "action": "decide_action",
            "decision": decision,
            "other_agent_id": other_agent_id,
            "my_resources": self.resources,
            "my_cooperation_tendency": self.cooperation_tendency
        })
        
        return decision
    
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
        
        # Handle zero-amount trades (no-op but still logged)
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
