import random
import logging
from collections import defaultdict
from typing import Any

from simulation.config import SimulationConfig

# Set up logger for environment debugging
logger = logging.getLogger(__name__)


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
        self.interaction_graph: defaultdict[Any, set] = defaultdict(set)
    
    def step(self):
        """
        Execute one simulation cycle:
        - Apply per-step resource decay based on scarcity_level
        - Randomly pair agents
        - Optionally exchange intent signals (gated by communication_enabled)
        - Trigger decision-making interactions
        - If both agents cooperate, attempt to grant a resource reward
          (probability reduced by scarcity_level; richer agent receives an
          amplified reward when enable_elite_advantage is True)
        - Redistribute resources between pairs using redistribution_strength
          (only when gap exceeds trade_threshold)
        - Increment cycle_count
        """
        communication_enabled = self.config.communication_enabled
        scarcity_level = self.config.scarcity_level
        redistribution_strength = self.config.redistribution_strength
        trade_threshold = self.config.trade_threshold
        elite_advantage_factor = self.config.elite_advantage_factor
        enable_elite_advantage = self.config.enable_elite_advantage

        # Per-step scarcity decay: each agent loses a small fraction of a
        # resource unit proportional to scarcity_level.  A decay_factor of 0.1
        # means agents lose at most 0.1 resources per step at full scarcity.
        _DECAY_FACTOR = 0.1
        if scarcity_level > 0.0:
            decay = scarcity_level * _DECAY_FACTOR
            for agent in self.agents:
                agent.resources = max(0.0, agent.resources - decay)

        # Log resource distribution before step (guarded so list build is skipped when DEBUG is off)
        if logger.isEnabledFor(logging.DEBUG):
            resources_before = [agent.resources for agent in self.agents]
            logger.debug("Cycle %d: Resources before step: %s", self.cycle_count, resources_before)
            if resources_before:
                logger.debug(
                    "Cycle %d: Resource stats - min: %s, max: %s, sum: %s",
                    self.cycle_count, min(resources_before), max(resources_before), sum(resources_before),
                )

        pairs = self.pair_agents()
        logger.debug("Cycle %d: Created %d pairs from %d agents", self.cycle_count, len(pairs), len(self.agents))
        
        cooperation_count = 0
        reward_count = 0
        
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
            
            logger.debug(f"Cycle {self.cycle_count}: Agent {agent1.agent_id} -> {action1}, Agent {agent2.agent_id} -> {action2}")
            
            # Record interaction for both agents
            if hasattr(agent1, 'record_interaction') and hasattr(agent2, 'agent_id'):
                agent1.record_interaction(agent2.agent_id)
            if hasattr(agent2, 'record_interaction') and hasattr(agent1, 'agent_id'):
                agent2.record_interaction(agent1.agent_id)

            # Update interaction graph (undirected edge between agent1 and agent2)
            if hasattr(agent1, 'agent_id') and hasattr(agent2, 'agent_id'):
                self.interaction_graph[agent1.agent_id].add(agent2.agent_id)
                self.interaction_graph[agent2.agent_id].add(agent1.agent_id)

            # Update lightweight memory/trust for both agents based on observed behaviour
            if hasattr(agent1, 'update_memory') and action2 is not None and hasattr(agent2, 'agent_id'):
                agent1.update_memory(agent2.agent_id, action2)
            if hasattr(agent2, 'update_memory') and action1 is not None and hasattr(agent1, 'agent_id'):
                agent2.update_memory(agent1.agent_id, action1)
            
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
            # When enable_elite_advantage is True, the wealthier agent receives a bonus on
            # top of the base reward, scaled by elite_advantage_factor.
            if action1 == "cooperate" and action2 == "cooperate":
                cooperation_count += 1
                scarcity_roll = random.random()
                logger.debug(f"Cycle {self.cycle_count}: Both cooperated! Scarcity roll: {scarcity_roll:.3f}, threshold: {scarcity_level}")
                
                if scarcity_roll > scarcity_level:
                    reward_count += 1
                    base_reward = 1

                    if enable_elite_advantage:
                        # Determine which agent is wealthier and apply the advantage factor
                        # to their reward.  A bonus > 1.0 (factor > 2.0) is handled by
                        # granting floor(bonus) guaranteed extra units plus a fractional
                        # probability of one more, so the expected extra reward equals bonus.
                        bonus = elite_advantage_factor - 1.0  # e.g. 0.2 for factor=1.2
                        extra = int(bonus) + (1 if random.random() < (bonus % 1) else 0)
                        if agent1.resources >= agent2.resources:
                            reward1 = base_reward + extra
                            reward2 = base_reward
                        else:
                            reward1 = base_reward
                            reward2 = base_reward + extra
                    else:
                        reward1 = reward2 = base_reward

                    logger.debug(f"Cycle {self.cycle_count}: Granting resources to agents {agent1.agent_id} and {agent2.agent_id}")
                    if hasattr(agent1, 'receive_resource'):
                        agent1.receive_resource(reward1, source="cooperation_reward")
                    if hasattr(agent2, 'receive_resource'):
                        agent2.receive_resource(reward2, source="cooperation_reward")
                else:
                    logger.debug(f"Cycle {self.cycle_count}: Scarcity prevented reward (roll {scarcity_roll:.3f} <= {scarcity_level})")
        
        # Redistribution: transfer resources from the richer agent to the poorer one
        # only when the gap exceeds trade_threshold.  The transfer amount is scaled by
        # redistribution_strength so that 0.0 disables redistribution entirely and 1.0
        # fully halves the gap each step.  When enable_elite_advantage is True, the
        # richer agent effectively loses less by dividing the transfer by
        # elite_advantage_factor.
        for agent1, agent2 in pairs:
            diff = agent1.resources - agent2.resources
            if abs(diff) <= trade_threshold or redistribution_strength == 0.0:
                continue

            transfer = redistribution_strength * (abs(diff) / 2.0)

            if enable_elite_advantage:
                # Elite agents (more resources) retain a larger share: dividing
                # the transfer by elite_advantage_factor reduces the amount taken
                # from the richer side.  This interacts multiplicatively with
                # redistribution_strength; at very low strengths the combined
                # effect can be negligible, which is intentional.
                transfer = transfer / elite_advantage_factor

            if transfer == 0.0:
                continue

            if diff > 0:
                # agent1 is richer — transfer from agent1 to agent2
                actual = min(transfer, agent1.resources)
                agent1.resources -= actual
                agent2.resources += actual
                logger.debug(
                    f"Cycle {self.cycle_count}: Redistribution {agent1.agent_id} -> {agent2.agent_id} "
                    f"amount={actual:.2f} (gap={diff:.2f})"
                )
            else:
                # agent2 is richer — transfer from agent2 to agent1
                actual = min(transfer, agent2.resources)
                agent2.resources -= actual
                agent1.resources += actual
                logger.debug(
                    f"Cycle {self.cycle_count}: Redistribution {agent2.agent_id} -> {agent1.agent_id} "
                    f"amount={actual:.2f} (gap={-diff:.2f})"
                )

        self.cycle_count += 1
        
        # Log resource distribution after step (guarded so list build is skipped when DEBUG is off)
        if logger.isEnabledFor(logging.DEBUG):
            resources_after = [agent.resources for agent in self.agents]
            logger.debug("Cycle %d: Resources after step: %s", self.cycle_count - 1, resources_after)
            if resources_after:
                avg = sum(resources_after) / len(resources_after)
                logger.debug(
                    "Cycle %d: Resource stats - min: %s, max: %s, sum: %s",
                    self.cycle_count - 1, min(resources_after), max(resources_after), sum(resources_after),
                )
                logger.debug(
                    "Cycle %d: Summary - %d cooperation pairs, %d rewards granted",
                    self.cycle_count - 1, cooperation_count, reward_count,
                )
                
                # Compute and log Gini coefficient for this step
                from metrics.economics import compute_gini
                gini = compute_gini(resources_after)
                logger.debug(
                    "Cycle %d: Gini coefficient = %.4f, Total wealth = %d, Avg wealth = %.2f",
                    self.cycle_count - 1, gini, sum(resources_after), avg,
                )
            else:
                logger.debug(
                    "Cycle %d: Summary - %d cooperation pairs, %d rewards granted (no agents)",
                    self.cycle_count - 1, cooperation_count, reward_count,
                )
    
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
