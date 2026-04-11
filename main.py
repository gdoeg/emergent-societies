import random

from simulation.config import SimulationConfig
from simulation.agent import Agent
from simulation.world import World
from simulation.simulation import Simulation


def _make_agents(config: SimulationConfig):
    """Create agents with initial resources per the configured distribution."""
    if config.resource_distribution == "random":
        return [Agent(i, resources=random.randint(1, config.initial_resources * 2)) for i in range(config.num_agents)]
    return [Agent(i, resources=config.initial_resources) for i in range(config.num_agents)]


config = SimulationConfig()
agents = _make_agents(config)
world = World(agents)

sim = Simulation(world, config=config)
sim.run()

print("Simulation complete.")
