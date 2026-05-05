"""Batch experiment runner for emergent-societies simulations.

Runs multiple named experiment configurations automatically, logging each run
to MLflow (when enabled) and the SQLite experiments database, then prints an
aggregate summary.

Usage::

    python scripts/run_experiments.py
    python scripts/run_experiments.py --runs_per_config 3 --steps 50 --agents 20

Environment variables
---------------------
MLFLOW_ENABLED
    Set to ``1``, ``true``, or ``yes`` to enable MLflow tracking.
MLFLOW_TRACKING_URI
    Override the default ``./mlruns`` MLflow tracking URI.
EXPERIMENTS_DB_PATH
    Override the default ``./experiments.db`` SQLite path.
"""

import argparse
import logging
import os
import sys

# Allow imports from the project root regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.agent import Agent
from simulation.config import SimulationConfig
from simulation.experiment_tracking.mlflow_tracker import MLflowTracker
from simulation.policies.deterministic_policy import DeterministicPolicy
from simulation.simulation import Simulation
from simulation.world import World

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIGS = [
    {"name": "baseline", "decision_interval": 15},
    {"name": "fast_decisions", "decision_interval": 5},
    {"name": "slow_decisions", "decision_interval": 30},
    {"name": "high_memory", "memory_size": 100},
]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_simulation(experiment_cfg: dict, num_agents: int, num_steps: int) -> dict:
    """Run a single simulation for the given experiment configuration.

    Creates a fresh :class:`~simulation.config.SimulationConfig` from the
    experiment dict, builds agents and a world, then runs for *num_steps*
    ticks.  Each step's metrics are forwarded to MLflow (when enabled) and
    persisted to the SQLite database by :class:`~simulation.simulation.Simulation`.

    Args:
        experiment_cfg: Dict with a ``"name"`` key plus optional overrides for
            any :class:`~simulation.config.SimulationConfig` field.
        num_agents: Number of agents to create for this run.
        num_steps: Number of simulation ticks to execute.

    Returns:
        The metrics dict from the final simulation tick.
    """
    # Build SimulationConfig with CLI overrides then experiment-level overrides.
    config_kwargs: dict = {
        "config_name": experiment_cfg["name"],
        "num_agents": num_agents,
        "num_steps": num_steps,
        "policy_type": "deterministic",
    }
    for key in ("decision_interval", "memory_size"):
        if key in experiment_cfg:
            config_kwargs[key] = experiment_cfg[key]

    config = SimulationConfig(**config_kwargs)

    # Create agents with the deterministic policy (no LLM required).
    shared_policy = DeterministicPolicy()
    agents = [
        Agent(i, resources=config.initial_resources, policy=shared_policy)
        for i in range(num_agents)
    ]
    world = World(agents)

    # MLflow tracker – no-op when MLFLOW_ENABLED is not set.
    tracker = MLflowTracker()
    mlflow_params = config.to_dict()
    tracker.start_run(mlflow_params)

    # Simulation handles db.init_db(), db.insert_run(), and db.insert_metric()
    # internally for every tick.
    sim = Simulation(world, config=config)

    last_metrics: dict = {}
    for _ in range(num_steps):
        metrics = sim.step()
        tracker.log_metrics(metrics, step=world.time)
        last_metrics = metrics

    tracker.log_final_metrics(last_metrics)
    tracker.end_run()

    return last_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple emergent-societies experiments and store results."
    )
    parser.add_argument(
        "--runs_per_config",
        type=int,
        default=3,
        help="Number of independent runs per experiment configuration (default: 3).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of simulation ticks per run (default: 100).",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=20,
        help="Number of agents per run (default: 20).",
    )
    args = parser.parse_args()
    if args.runs_per_config < 1:
        parser.error("--runs_per_config must be at least 1")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    runs_per_config: int = args.runs_per_config
    num_steps: int = args.steps
    num_agents: int = args.agents

    print(
        f"\nBatch experiment runner starting"
        f"\n  configs      : {len(EXPERIMENT_CONFIGS)}"
        f"\n  runs/config  : {runs_per_config}"
        f"\n  steps/run    : {num_steps}"
        f"\n  agents/run   : {num_agents}"
        f"\n  total runs   : {len(EXPERIMENT_CONFIGS) * runs_per_config}\n"
    )

    # Accumulate per-config aggregate results for the final summary.
    summary: dict[str, dict] = {}

    for config in EXPERIMENT_CONFIGS:
        config_name: str = config["name"]
        gini_values: list[float] = []
        cooperation_values: list[float] = []
        power_values: list[float] = []

        for run_idx in range(runs_per_config):
            print(
                f"  [{config_name}] run {run_idx + 1}/{runs_per_config} ... ",
                end="",
                flush=True,
            )
            metrics = run_simulation(config, num_agents=num_agents, num_steps=num_steps)
            gini_values.append(float(metrics.get("gini", 0.0)))
            cooperation_values.append(float(metrics.get("cooperation_pct", 0.0)))
            power_values.append(float(metrics.get("avg_power", 0.0)))
            print("done")

        summary[config_name] = {
            "avg_gini": sum(gini_values) / len(gini_values),
            "avg_cooperation": sum(cooperation_values) / len(cooperation_values),
            "avg_power": sum(power_values) / len(power_values),
        }

    # Print summary table.
    col_w = 20
    print(f"\n{'=' * 68}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 68}")
    header = (
        f"{'Config':<{col_w}}"
        f"{'Avg Gini':>{col_w}}"
        f"{'Avg Cooperation':>{col_w}}"
        f"{'Avg Power':>{col_w}}"
    )
    print(header)
    print("-" * 68)
    for config_name, stats in summary.items():
        row = (
            f"{config_name:<{col_w}}"
            f"{stats['avg_gini']:>{col_w}.4f}"
            f"{stats['avg_cooperation']:>{col_w}.4f}"
            f"{stats['avg_power']:>{col_w}.4f}"
        )
        print(row)
    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    main()
