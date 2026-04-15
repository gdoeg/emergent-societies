"""Economic metrics for the emergent-societies simulation.

Provides :func:`compute_gini` and :class:`MetricsLogger` for tracking wealth
distribution across simulation ticks.
"""

import csv
import json
from typing import Dict, List, Any


def compute_gini(values: List[float]) -> float:
    """Compute the Gini coefficient for a list of wealth values.

    Uses the standard sorted-cumulative-distribution formula, which runs in
    O(n log n) time and is efficient for 100–1000 agents.

    Args:
        values: Non-negative wealth values for each agent.  An empty list or
            a list of all-zeros returns 0.0.

    Returns:
        Gini coefficient in the range [0.0, 1.0], where 0 is perfect equality
        and 1 is maximum inequality.
    """
    n = len(values)
    if n == 0:
        return 0.0

    sorted_values = sorted(values)
    total = sum(sorted_values)
    if total == 0.0:
        return 0.0

    cumulative = 0.0
    gini_numerator = 0.0
    for i, v in enumerate(sorted_values):
        cumulative += v
        gini_numerator += (2 * (i + 1) - n - 1) * v

    return gini_numerator / (n * total)


class MetricsLogger:
    """Accumulates per-tick economic metrics and exports them to JSONL or CSV.

    Each record has the shape::

        {
            "tick": int,
            "gini": float,
            "total_wealth": float,
            "avg_wealth": float,
        }

    Example::

        logger = MetricsLogger()
        logger.record(tick=0, resources=[10.0, 20.0, 5.0])
        logger.to_jsonl("/tmp/metrics.jsonl")
    """

    def __init__(self) -> None:
        """Initialise an empty MetricsLogger."""
        self.history: List[Dict[str, Any]] = []

    def record(self, tick: int, resources: List[float], graph=None, total_agents: int = 0) -> Dict[str, Any]:
        """Compute and store metrics for one simulation tick.

        Args:
            tick: Current simulation step index.
            resources: Resource value for every living agent at this tick.
            graph: Optional interaction graph (``agent_id`` → ``set`` of ids)
                from :attr:`~simulation.environment.Environment.interaction_graph`.
                When provided, ``avg_degree`` and ``network_density`` are included
                in the record.
            total_agents: Total number of agents in the simulation.  Used only
                when *graph* is provided.

        Returns:
            The metrics dict that was appended to :attr:`history`.
        """
        total_wealth = sum(resources)
        n = len(resources)
        avg_wealth = total_wealth / n if n > 0 else 0.0
        entry: Dict[str, Any] = {
            "tick": tick,
            "gini": compute_gini(resources),
            "total_wealth": total_wealth,
            "avg_wealth": avg_wealth,
        }
        if graph is not None:
            from metrics.metrics import average_degree, network_density
            entry["avg_degree"] = average_degree(graph)
            entry["network_density"] = network_density(graph, total_agents)
        self.history.append(entry)
        return entry

    def to_jsonl(self, path: str) -> None:
        """Write all metric records to a JSON Lines file.

        Args:
            path: Filesystem path of the output ``.jsonl`` file.
        """
        with open(path, "w", encoding="utf-8") as fh:
            for entry in self.history:
                fh.write(json.dumps(entry) + "\n")

    def to_csv(self, path: str) -> None:
        """Write all metric records to a CSV file.

        Args:
            path: Filesystem path of the output ``.csv`` file.
        """
        base_fields = ["tick", "gini", "total_wealth", "avg_wealth"]
        network_fields = ["avg_degree", "network_density"]
        has_network = any("avg_degree" in entry for entry in self.history)
        fieldnames = base_fields + (network_fields if has_network else [])
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.history)

    def clear(self) -> None:
        """Remove all records from the history."""
        self.history.clear()
