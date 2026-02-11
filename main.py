from simulation.agent import Agent
from simulation.world import World
from simulation.simulation import Simulation

agents = [Agent(i) for i in range(100)]
world = World(agents)

sim = Simulation(world, steps=500)
sim.run()

print("Simulation complete.")
