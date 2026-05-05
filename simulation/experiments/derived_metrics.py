"""Derived metrics for the emergent-societies experiment framework.

Each public function accepts a ``metrics_history`` list of per-step metric
dicts (as produced by :class:`~metrics.economics.MetricsLogger`) and returns
a single :class:`float` summarising the run.  Functions that need
per-agent data accept an additional keyword argument.

Metric functions
----------------
compute_gini_slope
    Slope of the Gini coefficient curve (linear regression over steps).
compute_stability
    Variance of the cooperation rate over all steps.
compute_elite_share
    Fraction of total wealth held by the wealthiest 10 % of agents.
compute_switching_rate
    Average absolute change in cooperation rate between consecutive steps.
compute_network_clustering
    Average network density across all steps (proxy for clustering).
compute_path_dependence
    Variance of a scalar (e.g. final Gini) across repeated runs.
compute_all
    Convenience wrapper that calls every metric and returns a dict.

Usage::

    from simulation.experiments.derived_metrics import compute_all

    derived = compute_all(sim.metrics_logger.history, final_resources=[...])
    print(derived)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _linreg_slope(xs: List[float], ys: List[float]) -> float:
    """Return the slope of the ordinary-least-squares line through (xs, ys).

    Args:
        xs: Independent variable values.
        ys: Dependent variable values (same length as *xs*).

    Returns:
        OLS slope, or ``0.0`` if the calculation is degenerate.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0.0:
        return 0.0
    return num / den


def _variance(values: List[float]) -> float:
    """Return the population variance of *values*, or ``0.0`` for ≤1 element.

    Args:
        values: Numeric values to summarise.

    Returns:
        Population variance.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n


def _extract_cooperation(metrics_history: List[Dict[str, Any]]) -> List[float]:
    """Extract cooperation rate values from history, accepting both key aliases.

    Args:
        metrics_history: Per-step metrics dicts.

    Returns:
        List of cooperation rate floats (values where neither alias was present
        are skipped).
    """
    values: List[float] = []
    for m in metrics_history:
        v = m.get("cooperation_pct")
        if v is None:
            v = m.get("pct_cooperating")
        if v is not None:
            values.append(float(v))
    return values


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------


def compute_gini_slope(metrics_history: List[Dict[str, Any]]) -> float:
    """Compute the slope of the Gini coefficient over simulation steps.

    Uses ordinary least-squares regression of Gini values on the step tick
    index.  A positive slope indicates growing inequality; a negative slope
    indicates convergence.

    Args:
        metrics_history: List of per-step metric dicts, each containing at
            least ``"tick"`` and ``"gini"`` keys.

    Returns:
        OLS slope of the Gini curve, or ``0.0`` when fewer than two valid
        data points are available.
    """
    pairs = [
        (float(m["tick"]), float(m["gini"]))
        for m in metrics_history
        if "tick" in m and "gini" in m
    ]
    if len(pairs) < 2:
        return 0.0
    xs, ys = zip(*pairs)
    return _linreg_slope(list(xs), list(ys))


def compute_stability(metrics_history: List[Dict[str, Any]]) -> float:
    """Compute the variance of the cooperation rate over all steps.

    Lower variance means a more stable cooperation equilibrium; higher variance
    indicates oscillation or regime shifts.

    Args:
        metrics_history: List of per-step metric dicts, each optionally
            containing ``"cooperation_pct"`` or ``"pct_cooperating"``.

    Returns:
        Population variance of the cooperation rate, or ``0.0`` when fewer
        than two valid values are present.
    """
    values = _extract_cooperation(metrics_history)
    return _variance(values)


def compute_elite_share(final_resources: List[float]) -> float:
    """Compute the wealth share of the top 10 % of agents.

    Args:
        final_resources: Per-agent resource counts at the end of a run.
            Must be non-negative; an empty list returns ``0.0``.

    Returns:
        Fraction of total wealth held by the wealthiest 10 % of agents,
        in the range ``[0.0, 1.0]``.  Returns ``0.0`` when total wealth is
        zero or fewer than two agents are present.
    """
    if not final_resources:
        return 0.0
    total = sum(final_resources)
    if total <= 0.0:
        return 0.0
    sorted_r = sorted(final_resources, reverse=True)
    top_n = max(1, len(sorted_r) // 10)
    return sum(sorted_r[:top_n]) / total


def compute_switching_rate(metrics_history: List[Dict[str, Any]]) -> float:
    """Compute the average absolute change in cooperation rate per step.

    A high switching rate indicates that agents are frequently revising
    their strategies; a low rate suggests behavioral stability.

    Args:
        metrics_history: List of per-step metric dicts, each optionally
            containing ``"cooperation_pct"`` or ``"pct_cooperating"``.

    Returns:
        Mean absolute per-step change in cooperation rate (same units as
        the cooperation field, typically 0–100), or ``0.0`` when fewer than
        two valid values are present.
    """
    values = _extract_cooperation(metrics_history)
    if len(values) < 2:
        return 0.0
    changes = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    return sum(changes) / len(changes)


def compute_network_clustering(metrics_history: List[Dict[str, Any]]) -> float:
    """Compute the average network density across all steps.

    Network density serves as a proxy for clustering when the full
    interaction graph is not available in the metrics history.

    Args:
        metrics_history: List of per-step metric dicts, each optionally
            containing a ``"network_density"`` key.

    Returns:
        Mean network density across steps that reported it, or ``0.0`` when
        no density values are present.
    """
    values = [
        float(m["network_density"])
        for m in metrics_history
        if m.get("network_density") is not None
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)


def compute_path_dependence(run_final_values: List[float]) -> float:
    """Compute the variance of a scalar across repeated runs of the same config.

    Pass the final value of any metric (e.g. Gini) collected across all runs
    for a single configuration.  Higher variance indicates stronger path
    dependence — the outcome depends heavily on the initial random seed.

    Args:
        run_final_values: Scalar metric value from each independent run
            (e.g. final Gini coefficient).

    Returns:
        Population variance across runs, or ``0.0`` for fewer than two runs.
    """
    return _variance(run_final_values)


def compute_all(
    metrics_history: List[Dict[str, Any]],
    final_resources: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute all derived metrics and return them as a single dict.

    Args:
        metrics_history: List of per-step metric dicts as produced by
            :class:`~metrics.economics.MetricsLogger`.
        final_resources: Optional per-agent resource counts at the end of the
            run.  Required for :func:`compute_elite_share`; when omitted the
            ``"elite_share"`` value is ``0.0``.

    Returns:
        Dict with keys ``gini_slope``, ``stability``, ``elite_share``,
        ``switching_rate``, and ``network_clustering``.
    """
    result: Dict[str, float] = {
        "gini_slope": compute_gini_slope(metrics_history),
        "stability": compute_stability(metrics_history),
        "switching_rate": compute_switching_rate(metrics_history),
        "network_clustering": compute_network_clustering(metrics_history),
        "elite_share": (
            compute_elite_share(final_resources)
            if final_resources is not None
            else 0.0
        ),
    }
    logger.debug("Derived metrics: %s", result)
    return result
