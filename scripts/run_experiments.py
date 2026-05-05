"""Batch experiment runner for emergent-societies simulations.

Runs multiple named experiment configurations automatically, logging each run
to MLflow (when enabled) and the SQLite experiments database, then prints an
aggregate summary that includes derived research metrics.

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
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

# Allow imports from the project root regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.agent import Agent
from simulation.config import SimulationConfig
from simulation.experiment_tracking.mlflow_tracker import MLflowTracker
from simulation.experiments.derived_metrics import compute_all, compute_path_dependence
from simulation.policies.deterministic_policy import DeterministicPolicy
from simulation.simulation import Simulation
from simulation.storage import db
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


def run_simulation(
    experiment_cfg: dict,
    num_agents: int,
    num_steps: int,
    seed: Optional[int] = None,
) -> Tuple[dict, dict]:
    """Run a single simulation for the given experiment configuration.

    Creates a fresh :class:`~simulation.config.SimulationConfig` from the
    experiment dict, builds agents and a world, then runs for *num_steps*
    ticks.  Each step's metrics are forwarded to MLflow (when enabled) and
    persisted to the SQLite database by :class:`~simulation.simulation.Simulation`.

    After the run, derived research metrics are computed from the full metrics
    history and stored in the SQLite ``derived_metrics`` table.

    Args:
        experiment_cfg: Dict with a ``"name"`` key plus optional overrides for
            any :class:`~simulation.config.SimulationConfig` field.
        num_agents: Number of agents to create for this run.
        num_steps: Number of simulation ticks to execute.
        seed: Optional random seed for reproducibility.  When provided, both
            :mod:`random` and the initial agent resources use this seed so
            that runs with identical seeds are deterministic.

    Returns:
        A ``(last_metrics, derived)`` tuple where *last_metrics* is the raw
        metrics dict from the final simulation tick and *derived* is the dict
        of computed research metrics (``gini_slope``, ``stability``,
        ``elite_share``, ``switching_rate``, ``network_clustering``).
    """
    if seed is not None:
        random.seed(seed)

    # Build SimulationConfig with CLI overrides then experiment-level overrides.
    config_kwargs: dict = {
        "config_name": experiment_cfg["name"],
        "num_agents": num_agents,
        "num_steps": num_steps,
        "policy_type": "deterministic",
    }
    for key in ("decision_interval", "memory_size", "elite_advantage_factor", "enable_elite_advantage"):
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

        # Enrich metrics with cooperation_pct computed from live agent strategies.
        # This value is added in-place so that the MetricsLogger history, MLflow,
        # TensorBoard, and SQLite all benefit from it without changing Simulation.
        alive_agents = [a for a in world.agents if a.alive]
        if alive_agents:
            cooperating = sum(1 for a in alive_agents if a.strategy == "cooperate")
            metrics["cooperation_pct"] = cooperating / len(alive_agents) * 100.0

        tracker.log_metrics(metrics, step=world.time)
        last_metrics = metrics

    tracker.log_final_metrics(last_metrics)
    tracker.end_run()

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------
    metrics_history = sim.metrics_logger.history
    final_resources = [a.resources for a in world.agents if a.alive]
    derived = compute_all(metrics_history, final_resources)

    if sim._db_run_id is not None:
        db.insert_derived_metrics(sim._db_run_id, derived)

    return last_metrics, derived


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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base random seed for reproducibility.  Each run receives "
            "seed + run_index so runs are independent but reproducible."
        ),
    )
    args = parser.parse_args()
    if args.runs_per_config < 1:
        parser.error("--runs_per_config must be at least 1")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _std(values: List[float]) -> float:
    """Return the population standard deviation of *values*."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def main() -> None:
    args = _parse_args()
    runs_per_config: int = args.runs_per_config
    num_steps: int = args.steps
    num_agents: int = args.agents
    base_seed: Optional[int] = args.seed

    print(
        f"\nBatch experiment runner starting"
        f"\n  configs      : {len(EXPERIMENT_CONFIGS)}"
        f"\n  runs/config  : {runs_per_config}"
        f"\n  steps/run    : {num_steps}"
        f"\n  agents/run   : {num_agents}"
        f"\n  base seed    : {base_seed if base_seed is not None else 'random'}"
        f"\n  total runs   : {len(EXPERIMENT_CONFIGS) * runs_per_config}\n"
    )

    # Ensure derived_metrics and config_aggregates tables exist.
    db.init_db()

    # Derived metric names tracked in the summary.
    derived_keys: List[str] = [
        "gini_slope",
        "stability",
        "elite_share",
        "switching_rate",
    ]

    # Accumulate per-config aggregate results for the final summary.
    summary: dict[str, dict] = {}

    for config in EXPERIMENT_CONFIGS:
        config_name: str = config["name"]
        gini_values: list[float] = []
        cooperation_values: list[float] = []
        power_values: list[float] = []

        # Accumulators for derived metrics across runs of this config.
        derived_accum: Dict[str, List[float]] = {k: [] for k in derived_keys}
        final_ginis: List[float] = []

        for run_idx in range(runs_per_config):
            run_seed = base_seed + run_idx if base_seed is not None else None
            print(
                f"  [{config_name}] run {run_idx + 1}/{runs_per_config} ... ",
                end="",
                flush=True,
            )
            last_metrics, derived = run_simulation(
                config,
                num_agents=num_agents,
                num_steps=num_steps,
                seed=run_seed,
            )
            gini_values.append(float(last_metrics.get("gini", 0.0)))
            cooperation_values.append(float(last_metrics.get("cooperation_pct", 0.0)))
            power_values.append(float(last_metrics.get("avg_power", 0.0)))
            final_ginis.append(float(last_metrics.get("gini", 0.0)))

            for k in derived_keys:
                derived_accum[k].append(float(derived.get(k, 0.0)))

            print("done")

        # Compute path dependence from final Gini values across runs.
        path_dep = compute_path_dependence(final_ginis)

        # Aggregate derived metrics (mean ± std) and persist to SQLite.
        agg: Dict[str, Dict[str, float]] = {}
        for k in derived_keys:
            vals = derived_accum[k]
            m = sum(vals) / len(vals) if vals else 0.0
            s = _std(vals)
            agg[k] = {"mean": m, "std": s}
            db.insert_config_aggregate(config_name, k, m, s, len(vals))

        db.insert_config_aggregate(
            config_name, "path_dependence", path_dep, 0.0, runs_per_config
        )

        summary[config_name] = {
            "avg_gini": sum(gini_values) / len(gini_values),
            "avg_cooperation": sum(cooperation_values) / len(cooperation_values),
            "avg_power": sum(power_values) / len(power_values),
            "agg": agg,
            "path_dependence": path_dep,
        }

    # ------------------------------------------------------------------
    # Print derived-metrics summary table
    # ------------------------------------------------------------------
    cw = 20   # config column width
    mw = 14   # metric column width
    sep_width = cw + mw * len(derived_keys)

    print(f"\n{'=' * sep_width}")
    print("EXPERIMENT SUMMARY — DERIVED METRICS")
    print(f"{'=' * sep_width}")
    header = (
        f"{'Config':<{cw}}"
        f"{'Gini Slope':>{mw}}"
        f"{'Stability':>{mw}}"
        f"{'Elite Share':>{mw}}"
        f"{'Switch Rate':>{mw}}"
    )
    print(header)
    print("-" * sep_width)
    for config_name, stats in summary.items():
        agg = stats["agg"]
        row = (
            f"{config_name:<{cw}}"
            f"{agg['gini_slope']['mean']:>{mw}.4f}"
            f"{agg['stability']['mean']:>{mw}.4f}"
            f"{agg['elite_share']['mean']:>{mw}.4f}"
            f"{agg['switching_rate']['mean']:>{mw}.4f}"
        )
        print(row)
    print(f"{'=' * sep_width}\n")

    # ------------------------------------------------------------------
    # Also print the legacy raw-metric summary
    # ------------------------------------------------------------------
    col_w = 20
    print(f"{'=' * 68}")
    print("EXPERIMENT SUMMARY — RAW METRICS")
    print(f"{'=' * 68}")
    legacy_header = (
        f"{'Config':<{col_w}}"
        f"{'Avg Gini':>{col_w}}"
        f"{'Avg Cooperation':>{col_w}}"
        f"{'Avg Power':>{col_w}}"
    )
    print(legacy_header)
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
