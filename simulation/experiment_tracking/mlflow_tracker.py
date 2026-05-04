"""MLflow experiment-tracking integration for the emergent-societies simulation.

Enable by setting the ``MLFLOW_ENABLED`` environment variable to ``"true"``::

    MLFLOW_ENABLED=true python main.py

The tracking URI defaults to ``./mlruns`` (local folder) but can be overridden
via the standard ``MLFLOW_TRACKING_URI`` environment variable::

    MLFLOW_TRACKING_URI=http://localhost:5000 python main.py

Usage::

    from simulation.experiment_tracking.mlflow_tracker import MLflowTracker

    tracker = MLflowTracker()
    tracker.start_run(config)
    for step, metrics in enumerate(step_metrics):
        tracker.log_metrics(metrics, step)
    tracker.log_final_metrics(final_metrics)
    tracker.end_run()
"""

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Config keys that are tracked as MLflow params.
_TRACKED_PARAMS = (
    "num_agents",
    "decision_interval",
    "memory_size",
    "scarcity_level",
)

# Metric keys that are logged per step.
_STEP_METRICS = (
    "gini",
    "cooperation_pct",
    "avg_power",
    "max_power",
    "network_density",
)


def _is_enabled() -> bool:
    """Return ``True`` when the ``MLFLOW_ENABLED`` feature flag is set."""
    return os.getenv("MLFLOW_ENABLED", "").strip().lower() in {"1", "true", "yes"}


class MLflowTracker:
    """Thin wrapper around MLflow for tracking simulation runs.

    The tracker is a *no-op* when MLflow is not installed or the
    ``MLFLOW_ENABLED`` environment variable is not set, so the simulation
    continues to work without the dependency.

    Attributes:
        enabled: Whether MLflow tracking is active for this instance.
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self._mlflow: Any = None
        self._active_run: Any = None

        if not _is_enabled():
            logger.debug("MLflow tracking disabled (MLFLOW_ENABLED not set)")
            return

        try:
            import mlflow  # noqa: PLC0415

            self._mlflow = mlflow
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("emergent-societies")
            self.enabled = True
            logger.info("MLflow tracking enabled (uri=%s)", tracking_uri)
        except ImportError:
            logger.warning(
                "mlflow package not installed – experiment tracking disabled. "
                "Install it with: pip install 'mlflow>=3.9.0'"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(self, config: Dict[str, Any]) -> None:
        """Start a new MLflow run and log the tracked configuration params.

        Args:
            config: Mapping of configuration key-value pairs, typically the
                output of :meth:`~simulation.config.SimulationConfig.to_dict`.
                Only the keys listed in :data:`_TRACKED_PARAMS` are recorded.
        """
        if not self.enabled:
            return
        try:
            self._active_run = self._mlflow.start_run()
            params = {k: config[k] for k in _TRACKED_PARAMS if k in config}
            self._mlflow.log_params(params)
            logger.debug("MLflow run started: %s", self._active_run.info.run_id)
        except Exception:  # noqa: BLE001
            logger.exception("MLflow start_run failed; tracking disabled for this run")
            self.enabled = False

    def log_metrics(self, step_metrics: Dict[str, Any], step: int) -> None:
        """Log per-step metrics to the active MLflow run.

        Only the keys listed in :data:`_STEP_METRICS` are forwarded.
        The ``cooperation_pct`` key is looked up under both ``cooperation_pct``
        and ``pct_cooperating`` to stay compatible with the snapshot dict
        produced by ``dashboard_backend.main._snapshot_metrics``.

        Args:
            step_metrics: Metrics dict for the current simulation step.
            step: Current simulation step index (used as the MLflow ``step``).
        """
        if not self.enabled:
            return
        try:
            metrics: Dict[str, float] = {}
            for key in _STEP_METRICS:
                if key == "cooperation_pct":
                    # Accept either naming convention from callers.
                    value = step_metrics.get("cooperation_pct", step_metrics.get("pct_cooperating"))
                else:
                    value = step_metrics.get(key)
                if value is not None:
                    metrics[key] = float(value)
            if metrics:
                self._mlflow.log_metrics(metrics, step=step)
        except Exception:  # noqa: BLE001
            logger.exception("MLflow log_metrics failed at step %d", step)

    def log_final_metrics(self, final_metrics: Dict[str, Any]) -> None:
        """Log summary metrics at the end of a simulation run.

        The same metric names as in :meth:`log_metrics` are recorded but
        prefixed with ``final_`` so they are easy to distinguish in the
        MLflow UI.

        Args:
            final_metrics: Metrics dict for the last simulation step.
        """
        if not self.enabled:
            return
        try:
            summary: Dict[str, float] = {}
            for key in _STEP_METRICS:
                if key == "cooperation_pct":
                    value = final_metrics.get("cooperation_pct", final_metrics.get("pct_cooperating"))
                else:
                    value = final_metrics.get(key)
                if value is not None:
                    summary[f"final_{key}"] = float(value)
            if summary:
                self._mlflow.log_metrics(summary)
        except Exception:  # noqa: BLE001
            logger.exception("MLflow log_final_metrics failed")

    def end_run(self) -> None:
        """End the active MLflow run.

        Safe to call even when no run is active.
        """
        if not self.enabled:
            return
        try:
            self._mlflow.end_run()
            if self._active_run is not None:
                logger.debug("MLflow run ended: %s", self._active_run.info.run_id)
            self._active_run = None
        except Exception:  # noqa: BLE001
            logger.exception("MLflow end_run failed")
