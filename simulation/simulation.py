from metrics.economics import MetricsLogger
from simulation.config import SimulationConfig


class Simulation:
    def __init__(self, world, steps=100, config: SimulationConfig = None):
        self.world = world
        self.config = config
        self.steps = config.num_steps if config is not None else steps
        self.metrics_logger = MetricsLogger()

    def run(self):
        for step in range(self.steps):
            self.world.time += 1

            for agent in self.world.agents:
                if not agent.alive:
                    continue

                action = agent.decide_action(self.world)
                if action:
                    self.world.apply_action(agent, action)

            resources = [a.resources for a in self.world.agents if a.alive]
            self.metrics_logger.record(tick=self.world.time, resources=resources)
