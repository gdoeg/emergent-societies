from metrics.economics import MetricsLogger


class Simulation:
    def __init__(self, world, steps=100):
        self.world = world
        self.steps = steps
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
