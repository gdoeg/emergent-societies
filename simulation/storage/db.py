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
    """Create the ``runs`` and ``metrics`` tables if they do not yet exist."""
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
