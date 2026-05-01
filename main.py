import random
import logging

from simulation.config import SimulationConfig
from simulation.agent import Agent
from simulation.world import World
from simulation.simulation import Simulation
from simulation.policies.deterministic_policy import DeterministicPolicy
from simulation.policies.llm_policy import LLMPolicy
from simulation.policies.llm_provider import get_llm_provider
from dotenv import load_dotenv
load_dotenv()

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
        An :class:`~simulation.policies.base.AgentPolicy` instance.
    """
    if config.policy_type == "llm":
        provider = get_llm_provider(
            llm_model=config.llm_model,
            llm_api_base_url=config.llm_api_base_url,
            llm_timeout=config.llm_timeout,
        )
        logger.info(
            "Using LLMPolicy: model=%s api_base_url=%s",
            config.llm_model,
            config.llm_api_base_url,
        )
        return LLMPolicy(
            provider=provider,
            model=config.llm_model,
            api_base_url=config.llm_api_base_url,
            timeout=config.llm_timeout,
            decision_interval=config.decision_interval,
            max_concurrent_llm_calls=config.max_concurrent_llm_calls,
            batch_size=config.llm_batch_size,
            enable_async=config.enable_async_llm,
            debug_llm=config.debug_llm,
        )
    logger.info("Using DeterministicPolicy")
    return DeterministicPolicy()


def _make_agents(config: SimulationConfig):
    """Create agents with initial resources per the configured distribution."""
    if config.policy_type == "llm":
        provider = get_llm_provider(
            llm_model=config.llm_model,
            llm_api_base_url=config.llm_api_base_url,
            llm_timeout=config.llm_timeout,
        )
        logger.info(
            "Creating shared LLMPolicy for agents: model=%s api_base_url=%s",
            config.llm_model,
            config.llm_api_base_url,
        )
        shared_policy = LLMPolicy(
            provider=provider,
            model=config.llm_model,
            api_base_url=config.llm_api_base_url,
            timeout=config.llm_timeout,
            decision_interval=config.decision_interval,
            max_concurrent_llm_calls=config.max_concurrent_llm_calls,
            batch_size=config.llm_batch_size,
            enable_async=config.enable_async_llm,
            debug_llm=config.debug_llm,
        )

        def agent_policy():
            return shared_policy

    else:
        shared_policy = _make_policy(config)

        def agent_policy():
            return shared_policy

    if config.resource_distribution == "random":
        agents = [
            Agent(
                i,
                resources=random.randint(1, config.initial_resources * 2),
                policy=agent_policy(),
            )
            for i in range(config.num_agents)
        ]
    else:
        agents = [
            Agent(i, resources=config.initial_resources, policy=agent_policy())
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
