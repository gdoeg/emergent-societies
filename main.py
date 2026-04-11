import random
import logging

from simulation.config import SimulationConfig
from simulation.agent import Agent
from simulation.world import World
from simulation.simulation import Simulation

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


def _make_agents(config: SimulationConfig):
    """Create agents with initial resources per the configured distribution."""
    if config.resource_distribution == "random":
        agents = [Agent(i, resources=random.randint(1, config.initial_resources * 2)) for i in range(config.num_agents)]
    else:
        agents = [Agent(i, resources=config.initial_resources) for i in range(config.num_agents)]
    
    logger.info(f"Created {len(agents)} agents with resource distribution: {config.resource_distribution}")
    resources = [a.resources for a in agents]
    logger.info(f"Initial resources - min: {min(resources)}, max: {max(resources)}, avg: {sum(resources)/len(resources):.2f}")
    return agents


config = SimulationConfig()
logger.info(f"Starting simulation with config: num_agents={config.num_agents}, num_steps={config.num_steps}, scarcity_level={config.scarcity_level}")
agents = _make_agents(config)
world = World(agents)

sim = Simulation(world, config=config)
sim.run()

print("Simulation complete.")
