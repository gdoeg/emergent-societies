"""Network metrics for the emergent-societies simulation.

Provides lightweight graph-based metrics that operate on the
``interaction_graph`` maintained by :class:`~simulation.environment.Environment`.
No external graph libraries are used.
"""

from collections import defaultdict
from typing import Any, DefaultDict, Set


def degree(graph: DefaultDict[Any, Set], agent_id: Any) -> int:
    """Return the number of unique agents that *agent_id* has interacted with.

    Args:
        graph: The interaction graph (``agent_id`` → ``set`` of connected ids).
        agent_id: The agent whose degree to compute.

    Returns:
        Number of connections for *agent_id*, or 0 if not present in the graph.
    """
    return len(graph[agent_id])


def average_degree(graph: DefaultDict[Any, Set]) -> float:
    """Return the average number of connections per agent in *graph*.

    Args:
        graph: The interaction graph (``agent_id`` → ``set`` of connected ids).

    Returns:
        Mean degree across all agents present in the graph, or 0.0 if empty.
    """
    if not graph:
        return 0.0
    return sum(len(neighbors) for neighbors in graph.values()) / len(graph)


def network_density(graph: DefaultDict[Any, Set], total_agents: int) -> float:
    """Return the density of *graph* relative to the total agent population.

    Density is defined as the number of unique undirected edges divided by the
    maximum possible edges for *total_agents* nodes::

        density = edges / (n * (n - 1) / 2)

    Args:
        graph: The interaction graph (``agent_id`` → ``set`` of connected ids).
        total_agents: Total number of agents in the simulation (n).

    Returns:
        Density in the range [0.0, 1.0], or 0.0 when fewer than 2 agents exist.
    """
    possible_edges = total_agents * (total_agents - 1) / 2
    if possible_edges <= 0:
        return 0.0

    # Count unique undirected edges: each edge (A, B) appears in both
    # graph[A] and graph[B], so the sum of degrees equals 2 * edges.
    edges = sum(len(neighbors) for neighbors in graph.values()) / 2
    return edges / possible_edges
