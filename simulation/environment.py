import random


class Environment:
    """
    Environment class for deterministic multi-agent simulation.
    Manages agent population, interactions, and simulation cycles.
    """
    
    def __init__(self, agents, resource_pool=None):
        """
        Initialize the environment with an agent population.
        
        Args:
            agents: list of Agent objects
            resource_pool: optional shared resource pool (default None)
        """
        self.agents = agents
        self.cycle_count = 0
        self.resource_pool = resource_pool
    
    def step(self):
        """
        Execute one simulation cycle:
        - Randomly pair agents
        - Trigger decision-making interactions
        - If both agents cooperate, perform a trade
        - Increment cycle_count
        """
        pairs = self.pair_agents()
        
        for agent1, agent2 in pairs:
            # Trigger decision-making for both agents
            # For simplicity, we'll use the decide_action method if available
            action1 = agent1.decide_action(None) if hasattr(agent1, 'decide_action') else None
            action2 = agent2.decide_action(None) if hasattr(agent2, 'decide_action') else None
            
            # If both agents cooperate (trade), perform the trade
            if action1 == "trade" and action2 == "trade":
                # Simple trade: exchange resources
                agent1.resources += 1
                agent2.resources += 1
        
        self.cycle_count += 1
    
    def pair_agents(self):
        """
        Randomly shuffle agents and return agent pairs for interaction.
        
        Returns:
            list of tuples: agent pairs
        """
        # Create a copy to avoid modifying the original list
        shuffled = self.agents.copy()
        random.shuffle(shuffled)
        
        # Pair agents - if odd number, last agent won't be paired
        pairs = []
        for i in range(0, len(shuffled) - 1, 2):
            pairs.append((shuffled[i], shuffled[i + 1]))
        
        return pairs
    
    def get_resource_distribution(self):
        """
        Return list of agent resource values.
        
        Returns:
            list: resource values for all agents
        """
        return [agent.resources for agent in self.agents]
