import random


class Environment:
    """
    Environment class for multi-agent simulation.
    Manages agent population, interactions, and simulation cycles.
    
    Note: While agent pairing uses randomization, the simulation can be made
    deterministic by setting a random seed before creating the environment.
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
            action1 = agent1.decide_action(agent2) if hasattr(agent1, 'decide_action') else None
            action2 = agent2.decide_action(agent1) if hasattr(agent2, 'decide_action') else None
            
            # Record interaction for both agents
            if hasattr(agent1, 'record_interaction') and hasattr(agent2, 'agent_id'):
                agent1.record_interaction(agent2.agent_id)
            if hasattr(agent2, 'record_interaction') and hasattr(agent1, 'agent_id'):
                agent2.record_interaction(agent1.agent_id)
            
            # Update trust based on observed partner behavior
            # Successful cooperation trust updates are handled centrally by
            # the trade/success path to avoid double-counting here.
            if not (action1 == "cooperate" and action2 == "cooperate"):
                # Handle defection - decrease trust
                if hasattr(agent1, 'update_trust') and hasattr(agent2, 'agent_id'):
                    if action2 == "defect":
                        agent1.update_trust(agent2.agent_id, -0.05)
                if hasattr(agent2, 'update_trust') and hasattr(agent1, 'agent_id'):
                    if action1 == "defect":
                        agent2.update_trust(agent1.agent_id, -0.05)
            
            # If both agents cooperate, perform the trade
            # Each agent gains 1 resource as a simple trade reward
            if action1 == "cooperate" and action2 == "cooperate":
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
