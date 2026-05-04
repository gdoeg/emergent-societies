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
    return os.getenv("TENSORBOARD_ENABLED", "").strip().lower() in {"1", "true", "yes"}


def _import_summary_writer():
    """Return a SummaryWriter class, trying torch then tensorboardX.

    Returns ``None`` when neither package is available.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

        return SummaryWriter
    except ImportError:
        pass

    try:
        from tensorboardX import SummaryWriter  # noqa: PLC0415

        return SummaryWriter
    except ImportError:
        pass

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

        if not _is_enabled():
            logger.debug("TensorBoard logging disabled (TENSORBOARD_ENABLED not set)")
            return

        writer_cls = _import_summary_writer()
        if writer_cls is None:
            logger.warning(
                "Neither torch.utils.tensorboard nor tensorboardX is installed – "
                "TensorBoard logging disabled. "
                "Install one with: pip install tensorboardX"
            )
            return

        self._SummaryWriter = writer_cls
        self.enabled = True
        logger.info("TensorBoard logging enabled (log_dir=%s)", _LOG_DIR)

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
            return
        try:
            log_path = os.path.join(_LOG_DIR, run_id)
            self._writer = self._SummaryWriter(log_dir=log_path)
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
        if not self.enabled or self._writer is None:
            return
        try:
            self._writer.add_scalar(metric_name, value, global_step=step)
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
        if not self.enabled or self._writer is None:
            return
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
        if not self.enabled or self._writer is None:
            return
        try:
            self._writer.close()
            logger.debug("TensorBoard writer closed")
        except Exception:  # noqa: BLE001
            logger.exception("TensorBoard close_writer failed")
        finally:
            self._writer = None
