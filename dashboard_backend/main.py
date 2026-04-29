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
from metrics.economics import compute_gini, compute_power
from metrics.metrics import average_degree, network_density

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard_backend")

app = FastAPI(title="Emergent Societies Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_agents(config: SimulationConfig):
    if config.policy_type == "llm":
        logger.info(
            "Creating independent LLMPolicy instances for backend agents: model=%s api_base_url=%s",
            config.llm_model,
            config.llm_api_base_url,
        )

        def agent_policy():
            return LLMPolicy(
                model=config.llm_model,
                api_base_url=config.llm_api_base_url,
                timeout=config.llm_timeout,
                decision_interval=config.decision_interval,
            )

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
        logger.info(
            "Using LLMPolicy in backend: model=%s api_base_url=%s",
            config.llm_model,
            config.llm_api_base_url,
        )
        return LLMPolicy(
            model=config.llm_model,
            api_base_url=config.llm_api_base_url,
            timeout=config.llm_timeout,
            decision_interval=config.decision_interval,
        )

    logger.info("Using DeterministicPolicy in backend")
    return DeterministicPolicy()


def _new_environment():
    config = SimulationConfig(resource_distribution="random", policy_type="deterministic")
    agents = _make_agents(config)
    env = Environment(agents, config=config)
    logger.info("Environment initialized with %d agents", config.num_agents)
    return env, config


# Module-level environment instance
_env, _config = _new_environment()
_metrics_history: list = []


class RunRequest(BaseModel):
    steps: int = 10


@app.on_event("startup")
def log_startup():
    logger.info("FastAPI backend started on /metrics, /run, /reset")


def _snapshot_metrics(env: Environment, config: SimulationConfig) -> dict:
    """Compute a metrics snapshot from the current environment state."""
    resources = [a.resources for a in env.agents]
    powers = [compute_power(a) for a in env.agents]
    graph = env.interaction_graph
    n = len(env.agents)
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
def run_simulation(
    run_request: RunRequest | None = Body(default=None),
    steps: int | None = Query(default=None),
):
    """Run the simulation for the given number of steps and record metrics."""
    resolved_steps = steps if steps is not None else (run_request.steps if run_request else 10)
    resolved_steps = max(1, resolved_steps)

    _seed_metrics_history()
    logger.info("POST /run steps=%s", resolved_steps)
    for _ in range(resolved_steps):
        _env.step()
        _metrics_history.append(_snapshot_metrics(_env, _config))
    logger.info("Current tick: %d", _env.cycle_count)
    return {"status": "ok", "steps_run": resolved_steps, "current_tick": _env.cycle_count}


@app.post("/reset")
def reset_simulation():
    """Reinitialise the environment and clear all metrics."""
    global _env, _config, _metrics_history
    logger.info("POST /reset")
    _env, _config = _new_environment()
    _metrics_history = []
    _seed_metrics_history()
    return {"status": "ok", "message": "Simulation reset"}
