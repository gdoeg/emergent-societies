"""SQLite persistence layer for simulation runs and per-step metrics.

Database file: ./experiments.db (relative to the working directory).

Tables
------
runs
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    config_name     TEXT
    num_agents      INTEGER
    decision_interval INTEGER
    memory_size     INTEGER
    timestamp       TEXT  (ISO-8601)

metrics
    id              INTEGER PRIMARY KEY AUTOINCREMENT
    run_id          INTEGER  REFERENCES runs(id)
    step            INTEGER
    gini            REAL
    cooperation_pct REAL
    avg_power       REAL
    max_power       REAL
    network_density REAL

derived_metrics
    id                  INTEGER PRIMARY KEY AUTOINCREMENT
    run_id              INTEGER  REFERENCES runs(id)
    gini_slope          REAL
    stability           REAL
    elite_share         REAL
    switching_rate      REAL
    network_clustering  REAL

config_aggregates
    id                  INTEGER PRIMARY KEY AUTOINCREMENT
    config_name         TEXT
    metric_name         TEXT
    mean                REAL
    std                 REAL
    num_runs            INTEGER
    timestamp           TEXT  (ISO-8601)
"""

import sqlite3
import datetime
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("EXPERIMENTS_DB_PATH", "./experiments.db")


def _connect() -> sqlite3.Connection:
    """Return a connection to the experiments database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all simulation tables if they do not yet exist."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                config_name       TEXT,
                num_agents        INTEGER,
                decision_interval INTEGER,
                memory_size       INTEGER,
                timestamp         TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL REFERENCES runs(id),
                step            INTEGER,
                gini            REAL,
                cooperation_pct REAL,
                avg_power       REAL,
                max_power       REAL,
                network_density REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS derived_metrics (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id             INTEGER NOT NULL REFERENCES runs(id),
                gini_slope         REAL,
                stability          REAL,
                elite_share        REAL,
                switching_rate     REAL,
                network_clustering REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS config_aggregates (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                config_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                mean        REAL,
                std         REAL,
                num_runs    INTEGER,
                timestamp   TEXT
            )
            """
        )
        conn.commit()
    logger.debug("Database initialised at %s", DB_PATH)


def insert_run(config) -> int:
    """Insert a new run record derived from *config* and return its row id.

    Args:
        config: A :class:`~simulation.config.SimulationConfig` instance.

    Returns:
        The auto-assigned primary key of the new ``runs`` row.
    """
    config_name = getattr(config, "config_name", None) or getattr(config, "policy_type", "default")
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO runs (config_name, num_agents, decision_interval, memory_size, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                config_name,
                getattr(config, "num_agents", None),
                getattr(config, "decision_interval", None),
                getattr(config, "memory_size", None),
                timestamp,
            ),
        )
        conn.commit()
        run_id = cursor.lastrowid
    logger.debug("Inserted run id=%d config_name=%s", run_id, config_name)
    return run_id


def insert_metric(run_id: int, step: int, metrics: Dict[str, Any]) -> None:
    """Append one step's metrics to the ``metrics`` table.

    Args:
        run_id: Foreign key referencing the ``runs`` row for this experiment.
        step: Simulation step / tick number.
        metrics: Dict as returned by :meth:`~metrics.economics.MetricsLogger.record`.
            Recognised keys: ``gini``, ``cooperation_pct``, ``avg_power``,
            ``max_power``, ``network_density``.  Missing keys are stored as
            ``NULL``.
    """
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO metrics (run_id, step, gini, cooperation_pct, avg_power, max_power, network_density)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                step,
                metrics.get("gini"),
                metrics.get("cooperation_pct"),
                metrics.get("avg_power"),
                metrics.get("max_power"),
                metrics.get("network_density"),
            ),
        )
        conn.commit()


def get_runs() -> List[Dict[str, Any]]:
    """Return all rows from the ``runs`` table as a list of dicts.

    Returns:
        List of dicts with keys: ``id``, ``config_name``, ``num_agents``,
        ``decision_interval``, ``memory_size``, ``timestamp``.
    """
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def get_metrics(run_id: int) -> List[Dict[str, Any]]:
    """Return all metric rows for a given run, ordered by step.

    Args:
        run_id: Primary key of the run in the ``runs`` table.

    Returns:
        List of dicts with keys: ``id``, ``run_id``, ``step``, ``gini``,
        ``cooperation_pct``, ``avg_power``, ``max_power``, ``network_density``.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM metrics WHERE run_id = ? ORDER BY step",
            (run_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def insert_derived_metrics(run_id: int, derived: Dict[str, Any]) -> None:
    """Insert derived metrics for a completed simulation run.

    Args:
        run_id: Foreign key referencing the ``runs`` row.
        derived: Dict as returned by
            :func:`~simulation.experiments.derived_metrics.compute_all`.
            Recognised keys: ``gini_slope``, ``stability``, ``elite_share``,
            ``switching_rate``, ``network_clustering``.  Missing keys are
            stored as ``NULL``.
    """
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO derived_metrics
                (run_id, gini_slope, stability, elite_share, switching_rate, network_clustering)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                derived.get("gini_slope"),
                derived.get("stability"),
                derived.get("elite_share"),
                derived.get("switching_rate"),
                derived.get("network_clustering"),
            ),
        )
        conn.commit()
    logger.debug("Inserted derived_metrics for run_id=%d", run_id)


def insert_config_aggregate(
    config_name: str,
    metric_name: str,
    mean: float,
    std: float,
    num_runs: int,
) -> None:
    """Persist aggregated (mean ± std) derived metrics for a config batch.

    Args:
        config_name: Identifier of the experiment configuration.
        metric_name: Name of the derived metric being aggregated (e.g.
            ``"gini_slope"``).
        mean: Mean value of the metric across all runs.
        std: Standard deviation of the metric across all runs.
        num_runs: Number of runs that were aggregated.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO config_aggregates
                (config_name, metric_name, mean, std, num_runs, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (config_name, metric_name, mean, std, num_runs, timestamp),
        )
        conn.commit()
    logger.debug(
        "Inserted config_aggregate config_name=%s metric=%s mean=%.4f std=%.4f",
        config_name,
        metric_name,
        mean,
        std,
    )


def get_derived_metrics(run_id: int) -> Optional[Dict[str, Any]]:
    """Return the derived metrics row for a given run, or ``None``.

    Args:
        run_id: Primary key of the run in the ``runs`` table.

    Returns:
        Dict with keys ``id``, ``run_id``, ``gini_slope``, ``stability``,
        ``elite_share``, ``switching_rate``, ``network_clustering``, or
        ``None`` when no derived metrics have been stored for the run.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM derived_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def get_config_aggregates(config_name: str) -> List[Dict[str, Any]]:
    """Return all aggregate rows for a given configuration name.

    Args:
        config_name: Configuration identifier to filter by.

    Returns:
        List of dicts with keys ``id``, ``config_name``, ``metric_name``,
        ``mean``, ``std``, ``num_runs``, ``timestamp``.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM config_aggregates WHERE config_name = ? ORDER BY id",
            (config_name,),
        ).fetchall()
    return [dict(row) for row in rows]
