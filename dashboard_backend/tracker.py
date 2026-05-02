"""Simulation tracker for storing and aggregating results across multiple runs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean
from typing import Any


@dataclass
class SimulationRun:
    run_id: str
    metrics_history: list[dict[str, Any]]
    timestamp: str


class SimulationTracker:
    """Stores completed simulation runs in memory and computes aggregate metrics."""

    def __init__(self) -> None:
        self._runs: list[SimulationRun] = []

    def add_run(self, metrics_history: list[dict[str, Any]]) -> SimulationRun:
        """Record a completed simulation run and return the stored run object."""
        run = SimulationRun(
            run_id=str(uuid.uuid4()),
            metrics_history=list(metrics_history),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._runs.append(run)
        return run

    @property
    def runs(self) -> list[SimulationRun]:
        return list(self._runs)

    @property
    def run_count(self) -> int:
        return len(self._runs)

    def clear(self) -> None:
        self._runs.clear()


# Scalar numeric fields that can be averaged across runs at the same step index.
_SCALAR_FIELDS = (
    "tick",
    "total_wealth",
    "avg_wealth",
    "gini",
    "avg_power",
    "max_power",
    "average_degree",
    "network_density",
    "llm_call_count",
    "llm_fallback_count",
    "llm_fallback_rate",
    "total_agent_decisions",
    "fallback_agent_decisions",
    "avg_llm_latency",
    "pct_cooperating",
)


def compute_aggregate_metrics(runs: list[SimulationRun]) -> list[dict[str, Any]]:
    """Average scalar metrics across runs at each step index.

    Handles variable-length runs by only averaging over runs that contain
    data at a given step index.

    Args:
        runs: List of completed simulation runs.

    Returns:
        A list of dicts (one per step index) containing averaged scalar metrics
        and a ``run_count`` field indicating how many runs contributed to that
        step's averages.
    """
    if not runs:
        return []

    max_len = max(len(r.metrics_history) for r in runs)
    result: list[dict[str, Any]] = []

    for idx in range(max_len):
        # Collect snapshots from all runs that have this step index.
        snapshots = [r.metrics_history[idx] for r in runs if idx < len(r.metrics_history)]

        aggregated: dict[str, Any] = {}
        for field_name in _SCALAR_FIELDS:
            values = [s[field_name] for s in snapshots if field_name in s]
            aggregated[field_name] = mean(values) if values else 0.0

        # Preserve wealth_distribution from the last available run's step rather
        # than averaging element-wise, because the distributions may differ in
        # length and an element-wise average would not represent a meaningful
        # single distribution (it would lose the shape information).
        last_snapshot = snapshots[-1] if snapshots else {}
        aggregated["wealth_distribution"] = last_snapshot.get("wealth_distribution", [])

        # Aggregate strategy counts by summing then normalising to an average.
        strategy_keys: set[str] = set()
        for s in snapshots:
            strategy_keys.update((s.get("strategy_counts") or {}).keys())
        if strategy_keys:
            aggregated["strategy_counts"] = {
                k: mean(s.get("strategy_counts", {}).get(k, 0) for s in snapshots)
                for k in strategy_keys
            }
        else:
            aggregated["strategy_counts"] = {}

        aggregated["run_count"] = len(snapshots)
        result.append(aggregated)

    return result
