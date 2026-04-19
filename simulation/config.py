"""SimulationConfig: centralised configuration for the emergent-societies simulation."""

import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SimulationConfig:
    """Configuration for a simulation run.

    Attributes:
        num_agents: Number of agents to create.
        num_steps: Number of simulation steps (ticks) to run.
        initial_resources: Starting resource count for each agent.
        resource_distribution: How initial resources are spread across agents.
            ``"uniform"`` gives every agent exactly ``initial_resources``.
            ``"random"`` randomises each agent's starting resources in the
            range ``[1, initial_resources * 2]``.
        scarcity_level: Float in ``[0.0, 1.0]`` controlling resource
            availability.  Higher values reduce cooperative rewards and apply
            per-step resource decay to all agents.
        communication_enabled: When ``False``, agents skip the communication
            step before deciding their action.
        trade_threshold: Minimum resource gap required before redistribution
            is triggered between a pair of agents.
        redistribution_strength: Float in ``[0.0, 1.0]`` controlling how much
            of the resource gap is redistributed.  ``0.0`` disables
            redistribution entirely; ``1.0`` fully equalises each pair.
        elite_advantage_factor: Multiplier (>= 1.0) applied to resource gains
            for the wealthier agent when ``enable_elite_advantage`` is
            ``True``.
        enable_elite_advantage: When ``True``, the richer agent in a pairwise
            interaction receives amplified cooperation rewards and loses fewer
            resources during redistribution.
    """

    num_agents: int = 100
    num_steps: int = 500
    initial_resources: int = 10
    resource_distribution: str = "uniform"
    scarcity_level: float = 0.2
    communication_enabled: bool = True
    trade_threshold: int = 5
    top_n_leaders: int = 3
    redistribution_strength: float = 0.5
    elite_advantage_factor: float = 1.2
    enable_elite_advantage: bool = False

    def __post_init__(self) -> None:
        """Validate documented configuration constraints."""
        if self.num_agents < 0:
            raise ValueError("num_agents must be non-negative")
        if self.num_steps < 0:
            raise ValueError("num_steps must be non-negative")

        valid_distributions = {"uniform", "random"}
        if self.resource_distribution not in valid_distributions:
            raise ValueError(
                "resource_distribution must be one of "
                f"{sorted(valid_distributions)}, got {self.resource_distribution!r}"
            )

        if self.resource_distribution == "uniform":
            if self.initial_resources < 0:
                raise ValueError(
                    "initial_resources must be non-negative for "
                    "resource_distribution='uniform'"
                )
        elif self.resource_distribution == "random":
            if self.initial_resources < 1:
                raise ValueError(
                    "initial_resources must be at least 1 for "
                    "resource_distribution='random'"
                )

        if not 0.0 <= self.scarcity_level <= 1.0:
            raise ValueError("scarcity_level must be within [0.0, 1.0]")

        if self.trade_threshold < 0:
            raise ValueError("trade_threshold must be non-negative")

        if not 0.0 <= self.redistribution_strength <= 1.0:
            raise ValueError("redistribution_strength must be within [0.0, 1.0]")

        if self.elite_advantage_factor < 1.0:
            raise ValueError("elite_advantage_factor must be >= 1.0")

        if self.top_n_leaders < 1:
            raise ValueError("top_n_leaders must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation suitable for logging or serialisation."""
        return {
            "num_agents": self.num_agents,
            "num_steps": self.num_steps,
            "initial_resources": self.initial_resources,
            "resource_distribution": self.resource_distribution,
            "scarcity_level": self.scarcity_level,
            "communication_enabled": self.communication_enabled,
            "trade_threshold": self.trade_threshold,
            "top_n_leaders": self.top_n_leaders,
            "redistribution_strength": self.redistribution_strength,
            "elite_advantage_factor": self.elite_advantage_factor,
            "enable_elite_advantage": self.enable_elite_advantage,
        }

    @classmethod
    def from_json(cls, path: str) -> "SimulationConfig":
        """Load a :class:`SimulationConfig` from a JSON file.

        Args:
            path: Path to a JSON file whose keys match the field names of this
                dataclass.  Unknown keys are ignored; missing keys fall back to
                field defaults.

        Returns:
            A new :class:`SimulationConfig` instance.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def __repr__(self) -> str:
        return (
            f"SimulationConfig("
            f"num_agents={self.num_agents}, "
            f"num_steps={self.num_steps}, "
            f"initial_resources={self.initial_resources}, "
            f"resource_distribution={self.resource_distribution!r}, "
            f"scarcity_level={self.scarcity_level}, "
            f"communication_enabled={self.communication_enabled}, "
            f"trade_threshold={self.trade_threshold}, "
            f"top_n_leaders={self.top_n_leaders}, "
            f"redistribution_strength={self.redistribution_strength}, "
            f"elite_advantage_factor={self.elite_advantage_factor}, "
            f"enable_elite_advantage={self.enable_elite_advantage})"
        )
