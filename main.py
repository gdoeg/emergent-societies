import random
import logging

from simulation.config import SimulationConfig
from simulation.agent import Agent
from simulation.world import World
from simulation.simulation import Simulation
from simulation.policies.deterministic_policy import DeterministicPolicy
from simulation.policies.llm_policy import LLMPolicy

# Configure logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _make_policy(config: SimulationConfig):
    """Instantiate the policy requested by *config*.

    Args:
        config: Active :class:`~simulation.config.SimulationConfig`.

    Returns:
        An :class:`~simulation.policies.base.AgentPolicy` instance shared by
        all agents (policies are stateless with respect to individual agents).
    """
    if config.policy_type == "llm":
        logger.info(
            "Using LLMPolicy: model=%s api_base_url=%s",
            config.llm_model,
            config.llm_api_base_url,
        )
        return LLMPolicy(model=config.llm_model, api_base_url=config.llm_api_base_url, timeout=config.llm_timeout)
    logger.info("Using DeterministicPolicy")
    return DeterministicPolicy()


def _make_agents(config: SimulationConfig):
    """Create agents with initial resources per the configured distribution."""
    policy = _make_policy(config)

    if config.resource_distribution == "random":
        agents = [
            Agent(i, resources=random.randint(1, config.initial_resources * 2), policy=policy)
            for i in range(config.num_agents)
        ]
    else:
        agents = [
            Agent(i, resources=config.initial_resources, policy=policy)
            for i in range(config.num_agents)
        ]

    logger.info("Created %d agents with resource distribution: %s", len(agents), config.resource_distribution)
    if agents:
        resources = [a.resources for a in agents]
        logger.info(
            "Initial resources - min: %d, max: %d, avg: %.2f",
            min(resources), max(resources), sum(resources) / len(resources),
        )
    return agents


config = SimulationConfig()
logger.info(f"Starting simulation with config: num_agents={config.num_agents}, num_steps={config.num_steps}, scarcity_level={config.scarcity_level}")
agents = _make_agents(config)
world = World(agents)

sim = Simulation(world, config=config)
sim.run()

print("Simulation complete.")
