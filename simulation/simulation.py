from metrics.economics import MetricsLogger, compute_power
from simulation.config import SimulationConfig
import logging

logger = logging.getLogger(__name__)


class Simulation:
    def __init__(self, world, steps=100, config: SimulationConfig = None):
        self.world = world
        self.config = config
        self.steps = config.num_steps if config is not None else steps
        self.metrics_logger = MetricsLogger()

    def step(self) -> dict:
        """Execute a single simulation tick and record metrics.

        Returns:
            The metrics dict recorded for this tick.
        """
        top_n = self.config.top_n_leaders if self.config is not None else 3
        step = self.world.time  # used only for logging

        self.world.time += 1

        for agent in self.world.agents:
            if not agent.alive:
                continue

            action = agent.decide_action(self.world)
            if action:
                self.world.apply_action(agent, action)

        resources = [a.resources for a in self.world.agents if a.alive]

        # Compute power for all living agents and track leadership
        alive_agents = [a for a in self.world.agents if a.alive]
        if alive_agents:
            powers = sorted(alive_agents, key=compute_power, reverse=True)
            max_power = compute_power(powers[0])
            avg_power = sum(compute_power(a) for a in alive_agents) / len(alive_agents)
            top_agent_id = powers[0].agent_id
            logger.info(
                "Step %d: top_agent_id=%s, max_power=%s, avg_power=%.2f",
                step, top_agent_id, max_power, avg_power,
            )
            if logger.isEnabledFor(logging.DEBUG) and len(powers) >= top_n:
                top_ids = [(a.agent_id, compute_power(a)) for a in powers[:top_n]]
                logger.debug("Step %d: top %d agents by power: %s", step, top_n, top_ids)
        else:
            max_power = 0
            avg_power = 0.0

        metrics = self.metrics_logger.record(
            tick=self.world.time,
            resources=resources,
            avg_power=avg_power,
            max_power=max_power,
        )
        logger.debug(f"Step {step}: Recorded metrics - {metrics}")
        return metrics

    def run_steps(self, n_steps: int) -> None:
        """Execute exactly *n_steps* simulation ticks.

        Prefer this over :meth:`run` when you want to process the simulation
        in smaller batches (e.g. to update a dashboard between chunks) rather
        than blocking until all ``num_steps`` are complete.

        Args:
            n_steps: Number of ticks to run.
        """
        for _ in range(n_steps):
            self.step()

    def run(self):
        """Run the full simulation in ``chunk_size`` chunks.

        Chunking keeps individual blocking calls short and makes it easier for
        callers to interleave progress reporting between chunks.
        """
        _DEFAULT_CHUNK_SIZE = 10
        chunk_size = self.config.chunk_size if self.config is not None else _DEFAULT_CHUNK_SIZE
        remaining = self.steps
        while remaining > 0:
            to_run = min(chunk_size, remaining)
            self.run_steps(to_run)
            remaining -= to_run
