"""FastAPI backend for the emergent-societies simulation dashboard."""

import random
import sys
import os
import logging
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
from metrics.economics import compute_gini, compute_power
from metrics.metrics import average_degree, network_density
from dashboard_backend.tracker import SimulationTracker, compute_aggregate_metrics
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard_backend")

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
            {
                "id": agent.id,
                "wealth": agent.resources,
                "power": compute_power(agent),
                "strategy": agent.strategy,
            }
            for agent in env.agents
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

    run_start_len = len(_metrics_history)
    for _ in range(resolved_steps):
        await _env.step_async()
        metrics = _snapshot_metrics(_env, _config)
        _metrics_history.append(metrics)

        # W&B: log per-step metrics without blocking the simulation.
        _wandb_log_step(metrics)

    logger.info("Current tick: %d", _env.cycle_count)

    # Store the newly generated portion of history as a completed run.
    run_metrics = _metrics_history[run_start_len:]
    if run_metrics:
        stored = _tracker.add_run(run_metrics)
        logger.info("Stored run %s in tracker (total runs: %d)", stored.run_id, _tracker.run_count)

        # W&B: record final summary metrics and close the run.
        _wandb_finish_run(run_metrics[-1], stored.run_id, stored.timestamp)
    elif WANDB_ENABLED:
        wandb.finish()

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
        _metrics_history = [_snapshot_metrics(_env, _config)]

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

        for _ in range(resolved_steps):
            await _env.step_async()
            metrics = _snapshot_metrics(_env, _config)
            _metrics_history.append(metrics)

            # W&B: log per-step metrics for this independent run.
            _wandb_log_step(metrics)

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
