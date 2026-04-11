import random

from simulation.config import SimulationConfig


class Environment:
    """
    Environment class for multi-agent simulation.
    Manages agent population, interactions, and simulation cycles.
    
    Note: While agent pairing uses randomization, the simulation can be made
    deterministic by setting a random seed before creating the environment.
    """
    
    def __init__(self, agents, resource_pool=None, config: SimulationConfig = None):
        """
        Initialize the environment with an agent population.
        
        Args:
            agents: list of Agent objects
            resource_pool: optional shared resource pool (default None)
            config: optional SimulationConfig; controls scarcity_level and
                communication_enabled behaviour
        """
        self.agents = agents
        self.cycle_count = 0
        self.resource_pool = resource_pool
        self.config = config if config is not None else SimulationConfig()
    
    def step(self):
        """
        Execute one simulation cycle:
        - Randomly pair agents
        - Optionally exchange intent signals (gated by communication_enabled)
        - Trigger decision-making interactions
        - If both agents cooperate, attempt to grant a resource reward
          (probability reduced by scarcity_level)
        - Increment cycle_count
        """
        communication_enabled = self.config.communication_enabled
        scarcity_level = self.config.scarcity_level

        pairs = self.pair_agents()
        
        for agent1, agent2 in pairs:
            # Optional pre-decision communication exchange
            if communication_enabled:
                if hasattr(agent1, 'communicate'):
                    agent1.communicate(agent2, "intent")
                if hasattr(agent2, 'communicate'):
                    agent2.communicate(agent1, "intent")

            # Trigger decision-making for both agents
            action1 = agent1.decide_action(agent2) if hasattr(agent1, 'decide_action') else None
            action2 = agent2.decide_action(agent1) if hasattr(agent2, 'decide_action') else None
            
            # Record interaction for both agents
            if hasattr(agent1, 'record_interaction') and hasattr(agent2, 'agent_id'):
                agent1.record_interaction(agent2.agent_id)
            if hasattr(agent2, 'record_interaction') and hasattr(agent1, 'agent_id'):
                agent2.record_interaction(agent1.agent_id)
            
            # Update trust based on observed partner behavior
            if action1 == "cooperate" and action2 == "cooperate":
                # Both cooperated - increase trust
                if hasattr(agent1, 'update_trust') and hasattr(agent2, 'agent_id'):
                    agent1.update_trust(agent2.agent_id, 0.05)
                if hasattr(agent2, 'update_trust') and hasattr(agent1, 'agent_id'):
                    agent2.update_trust(agent1.agent_id, 0.05)
            else:
                # Handle defection - decrease trust toward a partner observed to defect
                if hasattr(agent1, 'update_trust') and hasattr(agent2, 'agent_id'):
                    if action2 == "defect":
                        agent1.update_trust(agent2.agent_id, -0.05)
                if hasattr(agent2, 'update_trust') and hasattr(agent1, 'agent_id'):
                    if action1 == "defect":
                        agent2.update_trust(agent1.agent_id, -0.05)
            
            # If both agents cooperate, grant each a resource reward from the environment.
            # Delegated through Agent.receive_resource() for consistent validation and logging.
            # scarcity_level controls reward availability: higher scarcity = lower probability.
            if action1 == "cooperate" and action2 == "cooperate":
                if random.random() > scarcity_level:
                    if hasattr(agent1, 'receive_resource'):
                        agent1.receive_resource(1, source="cooperation_reward")
                    if hasattr(agent2, 'receive_resource'):
                        agent2.receive_resource(1, source="cooperation_reward")
        
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

    def distribute_resources(self) -> None:
        """
        Set each agent's resources according to the configured distribution.

        ``resource_distribution == "uniform"`` (default): every agent receives
        exactly ``config.initial_resources``.
        ``resource_distribution == "random"``: each agent receives a value
        drawn uniformly from ``[1, config.initial_resources * 2]``.
        """
        initial_resources = self.config.initial_resources
        distribution = self.config.resource_distribution

        for agent in self.agents:
            if distribution == "random":
                agent.resources = random.randint(1, initial_resources * 2)
            else:
                agent.resources = initial_resources
