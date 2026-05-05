"""TensorBoard experiment-tracking integration for the emergent-societies simulation.

Enable by setting the ``TENSORBOARD_ENABLED`` environment variable to ``"true"``::

    TENSORBOARD_ENABLED=true python main.py

Logs are written to ``./runs/<run_id>/`` and can be viewed with::

    tensorboard --logdir=./runs

The logger tries ``torch.utils.tensorboard`` first and falls back to
``tensorboardX``.  If neither is installed the logger becomes a no-op so the
simulation continues to work without the dependency.

Usage::

    from simulation.experiment_tracking.tensorboard_logger import TensorBoardLogger

    tb = TensorBoardLogger()
    tb.init_writer(run_id="my_run")
    for step, metrics in enumerate(step_metrics):
        tb.log_metrics(metrics, step)
    tb.close_writer()
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Metric keys logged per simulation step.
_STEP_METRICS = (
    "gini",
    "cooperation_pct",
    "avg_power",
    "max_power",
    "network_density",
)

_LOG_DIR = "./runs"


def _is_enabled() -> bool:
    """Return ``True`` when the ``TENSORBOARD_ENABLED`` feature flag is set."""
    return os.getenv("TENSORBOARD_ENABLED", "false").lower() == "true"


def _import_summary_writer():
    """Return a SummaryWriter class, trying torch then tensorboardX.

    Returns ``None`` when neither package is available.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

        logger.debug("TensorBoard SummaryWriter resolved from torch.utils.tensorboard")
        return SummaryWriter
    except ImportError as exc:
        logger.debug("torch.utils.tensorboard unavailable: %s", exc)

    try:
        from tensorboardX import SummaryWriter  # noqa: PLC0415

        logger.debug("TensorBoard SummaryWriter resolved from tensorboardX")
        return SummaryWriter
    except ImportError as exc:
        logger.debug("tensorboardX unavailable: %s", exc)

    return None


class TensorBoardLogger:
    """Thin wrapper around TensorBoard's SummaryWriter for tracking simulation runs.

    The logger is a *no-op* when neither ``torch.utils.tensorboard`` nor
    ``tensorboardX`` is installed, or when the ``TENSORBOARD_ENABLED``
    environment variable is not set, so the simulation continues to work
    without the dependency.

    Attributes:
        enabled: Whether TensorBoard logging is active for this instance.
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self._writer: Any = None
        self._SummaryWriter: Any = None
        self._log_path: Optional[str] = None

        flag_value = os.getenv("TENSORBOARD_ENABLED", "false")
        logger.debug("Creating TensorBoardLogger with TENSORBOARD_ENABLED=%r", flag_value)

        if not _is_enabled():
            logger.debug("TensorBoard logging disabled (TENSORBOARD_ENABLED=%r)", flag_value)
            return

        writer_cls = _import_summary_writer()
        if writer_cls is None:
            logger.warning(
                "Neither torch.utils.tensorboard nor tensorboardX is available; "
                "TensorBoard logging disabled. Install one with: pip install tensorboardX"
            )
            return

        self._SummaryWriter = writer_cls
        self.enabled = True
        logger.info("TensorBoard logging enabled (base_log_dir=%s)", os.path.abspath(_LOG_DIR))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_writer(self, run_id: str) -> None:
        """Open a SummaryWriter for the given *run_id*.

        Logs are written to ``./runs/<run_id>/``.

        Args:
            run_id: Unique identifier for this simulation run.  Used as the
                subdirectory name under :data:`_LOG_DIR`.
        """
        if not self.enabled:
            logger.debug("Skipping TensorBoard init_writer for run_id=%s because logger is disabled", run_id)
            return
        try:
            log_path = os.path.abspath(os.path.join(_LOG_DIR, run_id))
            os.makedirs(log_path, exist_ok=True)
            logger.debug("Initializing TensorBoard writer for run_id=%s at %s", run_id, log_path)
            self._writer = self._SummaryWriter(log_dir=log_path)
            self._log_path = log_path
            logger.debug("TensorBoard writer opened at %s", log_path)
        except Exception:  # noqa: BLE001
            logger.exception("TensorBoard init_writer failed; logging disabled for this run")
            self.enabled = False

    def log_scalar(self, metric_name: str, value: float, step: int) -> None:
        """Write a single scalar value to TensorBoard.

        Args:
            metric_name: Tag name for the scalar (e.g. ``"gini"``).
            value: Numeric value to record.
            step: Global step index associated with this value.
        """
        if not self.enabled:
            logger.debug("Skipping TensorBoard log_scalar for %s at step %d because logger is disabled", metric_name, step)
            return
        if self._writer is None:
            logger.debug("Skipping TensorBoard log_scalar for %s at step %d because writer is not initialized", metric_name, step)
            return
        try:
            self._writer.add_scalar(metric_name, value, global_step=step)
            logger.debug(
                "TensorBoard logged metric name=%s value=%s step=%d path=%s",
                metric_name,
                value,
                step,
                self._log_path,
            )
        except Exception:  # noqa: BLE001
            logger.exception("TensorBoard log_scalar failed for %r at step %d", metric_name, step)

    def log_metrics(self, step_metrics: Dict[str, Any], step: int) -> None:
        """Log all tracked per-step metrics to TensorBoard.

        Only the keys listed in :data:`_STEP_METRICS` are forwarded.
        ``cooperation_pct`` is also looked up under the alias ``pct_cooperating``
        to remain compatible with dashboard snapshot dicts.

        Args:
            step_metrics: Metrics dict for the current simulation step.
            step: Current simulation step index.
        """
        if not self.enabled:
            logger.debug("Skipping TensorBoard log_metrics at step %d because logger is disabled", step)
            return
        if self._writer is None:
            logger.debug("Skipping TensorBoard log_metrics at step %d because writer is not initialized", step)
            return
        logger.debug("TensorBoard log_metrics called at step %d with keys=%s", step, sorted(step_metrics.keys()))
        for key in _STEP_METRICS:
            if key == "cooperation_pct":
                value = step_metrics.get("cooperation_pct", step_metrics.get("pct_cooperating"))
            else:
                value = step_metrics.get(key)
            if value is not None:
                self.log_scalar(key, float(value), step)

    def close_writer(self) -> None:
        """Flush and close the SummaryWriter.

        Safe to call even when no writer is open.
        """
        if not self.enabled:
            logger.debug("Skipping TensorBoard close_writer because logger is disabled")
            return
        if self._writer is None:
            logger.debug("Skipping TensorBoard close_writer because no writer is open")
            return
        try:
            self._writer.flush()
            self._writer.close()
            logger.debug("TensorBoard writer closed at %s", self._log_path)
        except Exception:  # noqa: BLE001
            logger.exception("TensorBoard close_writer failed")
        finally:
            self._writer = None
