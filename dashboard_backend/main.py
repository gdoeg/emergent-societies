"""FastAPI backend for the emergent-societies simulation dashboard."""

import random
import sys
import os

# Ensure the repo root is on sys.path so simulation/metrics can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from simulation.config import SimulationConfig
from simulation.agent import Agent
from simulation.world import World
from simulation.simulation import Simulation

app = FastAPI(title="Emergent Societies Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _make_agents(config: SimulationConfig):
    if config.resource_distribution == "random":
        return [Agent(i, resources=random.randint(1, config.initial_resources * 2)) for i in range(config.num_agents)]
    return [Agent(i, resources=config.initial_resources) for i in range(config.num_agents)]


def _new_simulation() -> Simulation:
    config = SimulationConfig()
    agents = _make_agents(config)
    world = World(agents)
    return Simulation(world, config=config)


# Module-level simulation instance
_sim: Simulation = _new_simulation()


@app.get("/metrics")
def get_metrics():
    """Return the full metrics history with wealth_distribution appended per tick."""
    enriched = []
    for entry in _sim.metrics_logger.history:
        record = dict(entry)
        # Provide defaults for optional fields to guarantee a consistent schema
        record.setdefault("avg_power", 0.0)
        record.setdefault("max_power", 0.0)
        record.setdefault("average_degree", 0.0)
        record.setdefault("network_density", 0.0)
        record["wealth_distribution"] = [a.resources for a in _sim.world.agents if a.alive]
        enriched.append(record)
    return enriched


class RunRequest(BaseModel):
    steps: int = 10


@app.post("/run")
def run_simulation(request: RunRequest):
    """Run the simulation for the given number of steps."""
    for _ in range(max(1, request.steps)):
        _sim.step()
    return {"status": "ok", "steps_run": request.steps, "current_tick": _sim.world.time}


@app.post("/reset")
def reset_simulation():
    """Reinitialise the simulation and clear all metrics."""
    global _sim
    _sim = _new_simulation()
    return {"status": "ok", "message": "Simulation reset"}
