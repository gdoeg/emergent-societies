"""FastAPI backend for the emergent-societies simulation dashboard."""

import datetime
import random
import sys
import os
import logging
from collections import defaultdict
from statistics import mean

# Ensure the repo root is on sys.path so simulation/metrics can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import Body, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from simulation.config import SimulationConfig
from simulation.agent import Agent
from simulation.environment import Environment
from simulation.policies.deterministic_policy import DeterministicPolicy
from simulation.policies.llm_policy import LLMPolicy
from simulation.policies.llm_provider import get_llm_provider
from simulation.storage import db
from metrics.economics import compute_gini, compute_power
from metrics.metrics import average_degree, network_density
from dashboard_backend.tracker import SimulationTracker, compute_aggregate_metrics
from simulation.experiment_tracking.mlflow_tracker import MLflowTracker
from simulation.experiment_tracking.tensorboard_logger import TensorBoardLogger
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard_backend")


def _latest_llm_observation(agent: Agent) -> tuple[float | None, str, str]:
    """Return latest confidence, reasoning, and decision from LLM log records."""
    for entry in reversed(getattr(agent, "interaction_memory", [])):
        if entry.get("action") not in {"decide_action", "strategy_update"}:
            continue
        raw_confidence = entry.get("confidence")
        confidence = None
        if isinstance(raw_confidence, (int, float)):
            confidence = float(raw_confidence)
        reasoning = str(entry.get("reasoning", ""))
        decision = str(entry.get("decision") or entry.get("new_strategy") or "")
        return confidence, reasoning, decision
    return None, "", ""


def _persona_observability_metrics(env: Environment) -> dict:
    """Aggregate persona-conditioned observability metrics from agent histories."""
    social_action_counts: dict[str, int] = defaultdict(int)
    social_coop_counts: dict[str, int] = defaultdict(int)
    social_defect_counts: dict[str, int] = defaultdict(int)

    confidence_by_risk: dict[str, list[float]] = defaultdict(list)
    confidence_by_social: dict[str, list[float]] = defaultdict(list)
    confidence_by_memory: dict[str, list[float]] = defaultdict(list)
    confidence_by_goal: dict[str, list[float]] = defaultdict(list)

    strategy_sequences_by_risk: dict[str, list[str]] = defaultdict(list)

    for agent in env.agents:
        risk_tolerance = str(getattr(agent, "risk_tolerance", "unknown"))
        social_preference = str(getattr(agent, "social_preference", "unknown"))
        memory_bias = str(getattr(agent, "memory_bias", "unknown"))
        goal = str(getattr(agent, "goal", "unknown"))

        for entry in getattr(agent, "interaction_memory", []):
            action = entry.get("action")
            if action in {"cooperate", "defect"}:
                social_action_counts[social_preference] += 1
                if action == "cooperate":
                    social_coop_counts[social_preference] += 1
                else:
                    social_defect_counts[social_preference] += 1

            if action in {"decide_action", "strategy_update"}:
                raw_confidence = entry.get("confidence")
                if isinstance(raw_confidence, (int, float)):
                    confidence = float(raw_confidence)
                    confidence_by_risk[risk_tolerance].append(confidence)
                    confidence_by_social[social_preference].append(confidence)
                    confidence_by_memory[memory_bias].append(confidence)
                    confidence_by_goal[goal].append(confidence)

            if action == "strategy_update":
                new_strategy = str(entry.get("new_strategy", "")).strip().lower()
                if new_strategy in {"cooperate", "defect"}:
                    strategy_sequences_by_risk[risk_tolerance].append(new_strategy)

    def _rate_map(numerator: dict[str, int], denominator: dict[str, int]) -> dict[str, float]:
        keys = set(denominator.keys()) | set(numerator.keys())
        return {
            key: (float(numerator.get(key, 0)) / denominator[key]) if denominator.get(key, 0) > 0 else 0.0
            for key in sorted(keys)
        }

    def _avg_map(values: dict[str, list[float]]) -> dict[str, float]:
        return {
            key: (sum(vals) / len(vals)) if vals else 0.0
            for key, vals in sorted(values.items())
        }

    strategy_switching_rate_by_risk_tolerance: dict[str, float] = {}
    for risk_tolerance, sequence in sorted(strategy_sequences_by_risk.items()):
        if len(sequence) < 2:
            strategy_switching_rate_by_risk_tolerance[risk_tolerance] = 0.0
            continue
        switches = sum(1 for prev, curr in zip(sequence, sequence[1:]) if prev != curr)
        strategy_switching_rate_by_risk_tolerance[risk_tolerance] = switches / (len(sequence) - 1)

    return {
        "cooperation_rate_by_social_preference": _rate_map(social_coop_counts, social_action_counts),
        "defect_rate_by_social_preference": _rate_map(social_defect_counts, social_action_counts),
        "avg_confidence_by_persona_type": {
            "risk_tolerance": _avg_map(confidence_by_risk),
            "social_preference": _avg_map(confidence_by_social),
            "memory_bias": _avg_map(confidence_by_memory),
            "goal": _avg_map(confidence_by_goal),
        },
        "strategy_switching_rate_by_risk_tolerance": strategy_switching_rate_by_risk_tolerance,
    }


def _reset_llm_prompt_sample_logging(env: Environment) -> None:
    """Reset one-shot LLM prompt sample logging for each shared policy instance."""
    seen_policy_ids = set()
    for agent in env.agents:
        policy = getattr(agent, "policy", None)
        if not isinstance(policy, LLMPolicy) or id(policy) in seen_policy_ids:
            continue
        seen_policy_ids.add(id(policy))
        if hasattr(policy, "reset_prompt_debug_sample"):
            policy.reset_prompt_debug_sample()


def _agent_snapshot(agent: Agent) -> dict:
    """Build dashboard agent payload including persona observability fields."""
    latest_confidence, latest_reasoning, latest_decision = _latest_llm_observation(agent)
    decision_history = [str(decision) for decision in getattr(agent, "decision_history", [])]
    confidence_history = [
        float(confidence)
        for confidence in getattr(agent, "confidence_history", [])
        if isinstance(confidence, (int, float))
    ]
    return {
        "id": agent.id,
        "wealth": agent.resources,
        "power": compute_power(agent),
        "strategy": agent.strategy,
        "risk_tolerance": getattr(agent, "risk_tolerance", "unknown"),
        "social_preference": getattr(agent, "social_preference", "unknown"),
        "memory_bias": getattr(agent, "memory_bias", "unknown"),
        "goal": getattr(agent, "goal", "unknown"),
        "latest_decision": latest_decision,
        "latest_confidence": latest_confidence,
        "latest_reasoning": latest_reasoning,
        "decision_history": decision_history,
        "confidence_history": confidence_history,
    }

# W&B is optional – the simulation runs normally if the package is absent.
try:
    import wandb
    WANDB_ENABLED = True
except ImportError:
    WANDB_ENABLED = False
    logger.info("wandb not installed – experiment tracking disabled")


def _wandb_log_step(metrics: dict) -> None:
    """Log per-step metrics to W&B. No-op when W&B is disabled."""
    if not WANDB_ENABLED:
        return
    wandb.log({
        "step": metrics["tick"],
        "gini": metrics["gini"],
        "total_wealth": metrics["total_wealth"],
        "avg_power": metrics["avg_power"],
        "network_density": metrics["network_density"],
        "pct_cooperating": metrics.get("pct_cooperating", 0),
        "llm_fallback_rate": metrics.get("llm_fallback_rate", 0),
    })


def _wandb_finish_run(final_metrics: dict, run_id: str, timestamp: str) -> None:
    """Record summary metrics and close the current W&B run. No-op when disabled."""
    if not WANDB_ENABLED:
        return
    wandb.summary["final_gini"] = final_metrics["gini"]
    wandb.summary["final_total_wealth"] = final_metrics["total_wealth"]
    wandb.summary["final_avg_power"] = final_metrics["avg_power"]
    # Attach the tracker run_id and timestamp for cross-reference.
    wandb.config.update({"run_id": run_id, "timestamp": timestamp})
    wandb.finish()

app = FastAPI(title="Emergent Societies Dashboard")

# MLflow tracker – instantiated once at module load; enabled via MLFLOW_ENABLED env var.
_mlflow_tracker = MLflowTracker()


def _new_tensorboard_logger(run_label: str) -> TensorBoardLogger:
    """Create a TensorBoard logger for a backend-triggered run."""
    run_id = f"backend_{run_label}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    tb_logger = TensorBoardLogger()
    tb_logger.init_writer(run_id)
    logger.info("TensorBoard logger prepared for backend run_id=%s", run_id)
    return tb_logger

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_agents(config: SimulationConfig):
    if config.policy_type == "llm":
        provider = get_llm_provider(
            llm_model=config.llm_model,
            llm_api_base_url=config.llm_api_base_url,
            llm_timeout=config.llm_timeout,
        )
        logger.info(
            "Creating shared LLMPolicy for backend agents: model=%s api_base_url=%s",
            getattr(provider, "model", config.llm_model),
            config.llm_api_base_url,
        )
        shared_policy = LLMPolicy(
            provider=provider,
            model=config.llm_model,
            api_base_url=config.llm_api_base_url,
            timeout=config.llm_timeout,
            decision_interval=config.decision_interval,
            max_concurrent_llm_calls=config.max_concurrent_llm_calls,
            batch_size=config.llm_batch_size,
            enable_async=config.enable_async_llm,
            debug_llm=config.debug_llm,
            llm_models=config.llm_models.split(",") if config.llm_models else None,
            temperature=getattr(config, "llm_temperature", 0.7),
        )

        def agent_policy():
            return shared_policy

    else:
        shared_policy = _make_policy(config)

        def agent_policy():
            return shared_policy

    agents = []
    for i in range(config.num_agents):
        coop = random.uniform(0.2, 0.8)
        if config.resource_distribution == "random":
            res = random.randint(1, config.initial_resources * 2)
        else:
            res = config.initial_resources
        agents.append(Agent(i, resources=res, cooperation_tendency=coop, policy=agent_policy()))
    return agents


def _make_policy(config: SimulationConfig):
    if config.policy_type == "llm":
        provider = get_llm_provider(
            llm_model=config.llm_model,
            llm_api_base_url=config.llm_api_base_url,
            llm_timeout=config.llm_timeout,
        )
        logger.info(
            "Using LLMPolicy in backend: model=%s api_base_url=%s models=%s max_retries=%s",
            getattr(provider, "model", config.llm_model),
            config.llm_api_base_url,
            config.llm_models,
            config.llm_max_retries,
        )
        return LLMPolicy(
            provider=provider,
            model=config.llm_model,
            api_base_url=config.llm_api_base_url,
            timeout=config.llm_timeout,
            decision_interval=config.decision_interval,
            max_concurrent_llm_calls=config.max_concurrent_llm_calls,
            batch_size=config.llm_batch_size,
            enable_async=config.enable_async_llm,
            debug_llm=config.debug_llm,
            llm_models=config.llm_models.split(",") if config.llm_models else None,
            temperature=getattr(config, "llm_temperature", 0.7),
        )

    logger.info("Using DeterministicPolicy in backend")
    return DeterministicPolicy()


def _new_environment():
    config = SimulationConfig(resource_distribution="random", policy_type="llm")
    agents = _make_agents(config)
    env = Environment(agents, config=config)
    logger.info(
        "Environment initialized with %d agents using %s policy",
        config.num_agents,
        config.policy_type,
    )
    return env, config


# Module-level environment instance
_env, _config = _new_environment()
_metrics_history: list = []
_tracker = SimulationTracker()


class RunRequest(BaseModel):
    steps: int = 10


class RunMultipleRequest(BaseModel):
    steps: int = 10
    num_runs: int = 3


@app.on_event("startup")
async def log_startup():
    # Initialize SQLite database for run tracking
    db.init_db()
    logger.info("FastAPI backend started on /metrics, /run, /reset, /aggregate-metrics, /run-multiple")
    seen_policy_ids = set()
    for agent in _env.agents:
        policy = getattr(agent, "policy", None)
        if not isinstance(policy, LLMPolicy) or id(policy) in seen_policy_ids:
            continue
        seen_policy_ids.add(id(policy))

        provider = getattr(policy, "provider", None)
        if provider is None or not hasattr(provider, "validate_connection"):
            continue

        health = await provider.validate_connection()
        if health.get("provider_status") == "error":
            logger.error(
                "LLM provider startup validation failed: provider=%s model=%s error=%s message=%s",
                health.get("provider"),
                health.get("model"),
                health.get("provider_error"),
                health.get("message"),
            )
            logger.warning("LLM provider is unhealthy at startup; simulation requests may fall back until this is fixed")
        else:
            logger.info(
                "LLM provider startup validation succeeded: provider=%s model=%s",
                health.get("provider"),
                health.get("model"),
            )


def _snapshot_metrics(env: Environment, config: SimulationConfig) -> dict:
    """Compute a metrics snapshot from the current environment state."""
    resources = [a.resources for a in env.agents]
    powers = [compute_power(a) for a in env.agents]
    graph = env.interaction_graph
    n = len(env.agents)
    llm_call_count = 0
    llm_fallback_count = 0
    success_agent_decisions = 0
    llm_error_count = 0
    llm_total_latency_seconds = 0.0
    llm_latency_samples = 0
    total_agent_decisions = 0
    fallback_agent_decisions = 0
    llm_provider_health = None
    provider_status = None
    provider_error = None
    seen_policy_ids = set()

    for agent in env.agents:
        policy = getattr(agent, "policy", None)
        if isinstance(policy, LLMPolicy) and id(policy) not in seen_policy_ids:
            seen_policy_ids.add(id(policy))
            llm_call_count += getattr(policy, "_llm_call_count", 0)
            llm_fallback_count += getattr(policy, "_fallback_count", 0)
            success_agent_decisions += getattr(policy, "_success_agent_decisions", 0)
            llm_error_count += getattr(policy, "_llm_error_count", 0)
            llm_total_latency_seconds += getattr(policy, "_llm_total_latency_seconds", 0.0)
            llm_latency_samples += getattr(policy, "_llm_latency_samples", 0)
            total_agent_decisions += getattr(policy, "_total_agent_decisions", 0)
            fallback_agent_decisions += getattr(policy, "_fallback_agent_decisions", 0)
            provider = getattr(policy, "provider", None)
            if provider is not None and hasattr(provider, "get_health"):
                llm_provider_health = provider.get_health()
                provider_status = llm_provider_health.get("provider_status")
                provider_error = llm_provider_health.get("provider_error")

    llm_success_rate = (
        (success_agent_decisions / total_agent_decisions) if total_agent_decisions > 0 else 0.0
    )
    llm_fallback_rate = (
        (fallback_agent_decisions / total_agent_decisions) if total_agent_decisions > 0 else 0.0
    )
    avg_llm_latency = (
        (llm_total_latency_seconds / llm_latency_samples) if llm_latency_samples > 0 else 0.0
    )

    logger.info(
        f"Agent Decisions: {total_agent_decisions}, Success: {success_agent_decisions}, Fallback: {fallback_agent_decisions}"
    )

    # Strategy distribution: count agents currently on each standing strategy.
    cooperating = sum(1 for a in env.agents if getattr(a, "strategy", None) == "cooperate")
    defecting = sum(1 for a in env.agents if getattr(a, "strategy", None) == "defect")
    pct_cooperating = (cooperating / n * 100.0) if n > 0 else 0.0
    persona_metrics = _persona_observability_metrics(env)

    return {
        "tick": env.cycle_count,
        "total_wealth": sum(resources),
        "avg_wealth": mean(resources) if resources else 0.0,
        "gini": compute_gini(resources),
        "avg_power": mean(powers) if powers else 0.0,
        "max_power": max(powers) if powers else 0.0,
        "average_degree": average_degree(graph),
        "network_density": network_density(graph, n),
        "wealth_distribution": resources,
        "llm_call_count": llm_call_count,
        "llm_fallback_count": llm_fallback_count,
        "llm_success_count": success_agent_decisions,
        "llm_success_rate": llm_success_rate,
        "llm_fallback_rate": llm_fallback_rate,
        "total_agent_decisions": total_agent_decisions,
        "success_agent_decisions": success_agent_decisions,
        "fallback_agent_decisions": fallback_agent_decisions,
        "avg_llm_latency": avg_llm_latency,
        "provider_status": provider_status,
        "provider_error": provider_error,
        "llm_provider_health": llm_provider_health,
        # Strategy breakdown for dashboard visualization
        "pct_cooperating": round(pct_cooperating, 2),
        "strategy_counts": {"cooperate": cooperating, "defect": defecting},
        # Per-agent snapshot for interpretability view
        "agents": [
            _agent_snapshot(agent)
            for agent in env.agents
        ],
        "cooperation_rate_by_social_preference": persona_metrics[
            "cooperation_rate_by_social_preference"
        ],
        "defect_rate_by_social_preference": persona_metrics[
            "defect_rate_by_social_preference"
        ],
        "avg_confidence_by_persona_type": persona_metrics[
            "avg_confidence_by_persona_type"
        ],
        "strategy_switching_rate_by_risk_tolerance": persona_metrics[
            "strategy_switching_rate_by_risk_tolerance"
        ],
        # Structured LLM diagnostics
        "llm_stats": {
            "calls": llm_call_count,
            "total_agent_decisions": total_agent_decisions,
            "success": success_agent_decisions,
            # Parse-only fallbacks = total fallbacks minus exception-based errors
            "fallbacks": max(0, llm_fallback_count - llm_error_count),
            "errors": llm_error_count,
            "success_rate": llm_success_rate,
            "fallback_rate": llm_fallback_rate,
            "latency": round(avg_llm_latency * 1000, 2),
        },
    }


def _seed_metrics_history() -> None:
    """Ensure metrics has an initial tick-0 snapshot for immediate dashboard rendering."""
    global _metrics_history
    if not _metrics_history:
        _metrics_history = [_snapshot_metrics(_env, _config)]


@app.get("/metrics")
def get_metrics():
    """Return the full metrics history accumulated across /run calls."""
    logger.info("GET /metrics")
    _seed_metrics_history()
    return _metrics_history


@app.post("/run")
async def run_simulation(
    run_request: RunRequest | None = Body(default=None),
    steps: int | None = Query(default=None),
):
    """Run the simulation for the given number of steps and record metrics.

    The endpoint is async so uvicorn's event loop is not blocked while steps
    execute.  Each step runs in a worker thread (via ``step_async``), which
    allows the internal ``asyncio.run()`` to create its own loop for concurrent
    LLM strategy updates without conflicting with the FastAPI event loop.

    After completion the run is stored in the simulation tracker.
    """
    resolved_steps = steps if steps is not None else (run_request.steps if run_request else 10)
    resolved_steps = max(1, resolved_steps)

    _seed_metrics_history()
    logger.info("POST /run steps=%s", resolved_steps)
    _reset_llm_prompt_sample_logging(_env)

    tb_logger = _new_tensorboard_logger("run")

    # W&B: initialise a new run for this simulation (one run per /run call).
    if WANDB_ENABLED:
        wandb.init(
            project="emergent-societies",
            reinit=True,
            config={
                "num_agents": _config.num_agents,
                "num_steps": resolved_steps,
                "decision_interval": _config.decision_interval,
                "batch_size": _config.llm_batch_size,
                "max_concurrent_llm_calls": _config.max_concurrent_llm_calls,
            },
        )

    # MLflow: start a new run and log config params.
    _mlflow_tracker.start_run(_config.to_dict())

    # SQLite: create a new run record
    db_run_id = db.insert_run(_config)
    logger.debug("SQLite run registered with id=%s", db_run_id)

    run_start_len = len(_metrics_history)
    try:
        for _ in range(resolved_steps):
            await _env.step_async()
            metrics = _snapshot_metrics(_env, _config)
            _metrics_history.append(metrics)

            # W&B: log per-step metrics without blocking the simulation.
            _wandb_log_step(metrics)

            # MLflow: log per-step metrics.
            _mlflow_tracker.log_metrics(metrics, step=metrics["tick"])

            # TensorBoard: log per-step metrics.
            tb_logger.log_metrics(metrics, step=metrics["tick"])

            # SQLite: log per-step metrics.
            db.insert_metric(db_run_id, metrics["tick"], metrics)

        logger.info("Current tick: %d", _env.cycle_count)

        # Store the newly generated portion of history as a completed run.
        run_metrics = _metrics_history[run_start_len:]
        if run_metrics:
            stored = _tracker.add_run(run_metrics)
            logger.info("Stored run %s in tracker (total runs: %d)", stored.run_id, _tracker.run_count)

            # W&B: record final summary metrics and close the run.
            _wandb_finish_run(run_metrics[-1], stored.run_id, stored.timestamp)

            # MLflow: log final summary metrics and end the run.
            _mlflow_tracker.log_final_metrics(run_metrics[-1])
        elif WANDB_ENABLED:
            wandb.finish()
    finally:
        tb_logger.close_writer()
        _mlflow_tracker.end_run()

    return {"status": "ok", "steps_run": resolved_steps, "current_tick": _env.cycle_count}


@app.post("/run-multiple")
async def run_multiple_simulations(
    body: RunMultipleRequest | None = Body(default=None),
):
    """Run the simulation *num_runs* times, each for *steps* steps.

    Each run starts from a freshly initialised environment so the results are
    independent.  All runs are stored in the tracker and the function returns
    the aggregated metrics across all stored runs.
    """
    global _env, _config, _metrics_history

    resolved_steps = body.steps if body else 10
    resolved_steps = max(1, resolved_steps)
    num_runs = body.num_runs if body else 3
    # Cap at 20 runs to prevent unbounded memory growth and very long-running requests.
    num_runs = max(1, min(num_runs, 20))

    logger.info("POST /run-multiple runs=%d steps=%d", num_runs, resolved_steps)

    for run_idx in range(num_runs):
        _env, _config = _new_environment()
        _reset_llm_prompt_sample_logging(_env)
        _metrics_history = [_snapshot_metrics(_env, _config)]
        tb_logger = _new_tensorboard_logger(f"multi_{run_idx}")

        # W&B: each independent run gets its own W&B run.
        if WANDB_ENABLED:
            wandb.init(
                project="emergent-societies",
                reinit=True,
                config={
                    "num_agents": _config.num_agents,
                    "num_steps": resolved_steps,
                    "decision_interval": _config.decision_interval,
                    "batch_size": _config.llm_batch_size,
                    "max_concurrent_llm_calls": _config.max_concurrent_llm_calls,
                    "run_index": run_idx,
                },
            )

        # MLflow: start a new run for each independent simulation.
        _mlflow_tracker.start_run(_config.to_dict())

        # SQLite: create a new run record for this independent simulation
        db_run_id = db.insert_run(_config)
        logger.debug("SQLite run registered for run %d/%d with id=%s", run_idx + 1, num_runs, db_run_id)

        try:
            for _ in range(resolved_steps):
                await _env.step_async()
                metrics = _snapshot_metrics(_env, _config)
                _metrics_history.append(metrics)

                # W&B: log per-step metrics for this independent run.
                _wandb_log_step(metrics)

                # MLflow: log per-step metrics.
                _mlflow_tracker.log_metrics(metrics, step=metrics["tick"])

                # TensorBoard: log per-step metrics.
                tb_logger.log_metrics(metrics, step=metrics["tick"])

                # SQLite: log per-step metrics.
                db.insert_metric(db_run_id, metrics["tick"], metrics)

            stored = _tracker.add_run(_metrics_history)
            logger.info(
                "Run %d/%d complete: run_id=%s (tracker total: %d)",
                run_idx + 1,
                num_runs,
                stored.run_id,
                _tracker.run_count,
            )

            # W&B: record final summary and close this run before starting the next.
            _wandb_finish_run(_metrics_history[-1], stored.run_id, stored.timestamp)

            # MLflow: log final summary and end this run before starting the next.
            _mlflow_tracker.log_final_metrics(_metrics_history[-1])
        finally:
            tb_logger.close_writer()
            _mlflow_tracker.end_run()

    aggregate = compute_aggregate_metrics(_tracker.runs)
    return {
        "status": "ok",
        "runs_completed": num_runs,
        "total_tracked_runs": _tracker.run_count,
        "aggregate_steps": len(aggregate),
    }


@app.get("/aggregate-metrics")
def get_aggregate_metrics():
    """Return averaged metrics across all stored simulation runs."""
    logger.info("GET /aggregate-metrics (runs in tracker: %d)", _tracker.run_count)
    aggregate = compute_aggregate_metrics(_tracker.runs)
    return aggregate


@app.get("/tracker-info")
def get_tracker_info():
    """Return metadata about all stored simulation runs."""
    return {
        "run_count": _tracker.run_count,
        "runs": [
            {"run_id": r.run_id, "timestamp": r.timestamp, "steps": len(r.metrics_history)}
            for r in _tracker.runs
        ],
    }


@app.post("/reset")
def reset_simulation():
    """Reinitialise the environment and clear all metrics."""
    global _env, _config, _metrics_history
    logger.info("POST /reset")
    _env, _config = _new_environment()
    _metrics_history = []
    _tracker.clear()
    _seed_metrics_history()
    return {"status": "ok", "message": "Simulation reset"}
