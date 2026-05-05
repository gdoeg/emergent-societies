"""Formal experiment definitions for the emergent-societies simulation.

Each :class:`Experiment` describes a research hypothesis in terms of the
configuration parameters to vary and the derived metrics that should be
computed to evaluate the hypothesis.

Usage::

    from simulation.experiments.experiments import EXPERIMENTS

    for exp in EXPERIMENTS:
        print(exp.name, exp.metrics)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class Experiment:
    """A named, reproducible experiment definition.

    Attributes:
        name: Short machine-readable identifier used in reports and database
            records.
        description: Human-readable explanation of the hypothesis being tested.
        metrics: List of derived-metric names (matching function names in
            :mod:`simulation.experiments.derived_metrics`) that should be
            computed and reported for this experiment.
        parameters: Optional mapping of :class:`~simulation.config.SimulationConfig`
            field overrides applied when running this experiment.  Keys must
            match ``SimulationConfig`` attribute names.
    """

    name: str
    description: str
    metrics: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Experiment(name={self.name!r}, "
            f"metrics={self.metrics!r}, "
            f"parameters={self.parameters!r})"
        )


# ---------------------------------------------------------------------------
# Catalogue of predefined experiments
# ---------------------------------------------------------------------------

EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="inequality_acceleration",
        description=(
            "Measure whether inequality accelerates over time by computing the "
            "slope of the Gini coefficient curve across simulation steps."
        ),
        metrics=["gini_slope"],
        parameters={"decision_interval": 15},
    ),
    Experiment(
        name="cooperation_stability",
        description=(
            "Quantify how stable the cooperation rate is by measuring the "
            "variance of pct_cooperating across all simulation steps."
        ),
        metrics=["stability"],
        parameters={"decision_interval": 15},
    ),
    Experiment(
        name="elite_formation",
        description=(
            "Measure the degree of wealth concentration by computing the share "
            "of total resources held by the top 10 % of agents at the end of "
            "each run.  Elite advantage is enabled to encourage stratification."
        ),
        metrics=["elite_share"],
        parameters={"elite_advantage_factor": 1.5, "enable_elite_advantage": True},
    ),
    Experiment(
        name="strategy_volatility",
        description=(
            "Measure how frequently agents switch strategies by computing the "
            "average absolute change in cooperation rate between consecutive "
            "steps.  A short decision interval increases update frequency."
        ),
        metrics=["switching_rate"],
        parameters={"decision_interval": 5},
    ),
    Experiment(
        name="path_dependence",
        description=(
            "Assess path dependence by running the same configuration multiple "
            "times with different random seeds and measuring the variance of "
            "final Gini values across runs."
        ),
        metrics=["path_dependence", "gini_slope", "stability"],
        parameters={"decision_interval": 15},
    ),
    Experiment(
        name="network_effects",
        description=(
            "Examine how network clustering evolves during the simulation by "
            "averaging the network density metric across all steps."
        ),
        metrics=["network_clustering", "gini_slope"],
        parameters={"decision_interval": 15},
    ),
]
