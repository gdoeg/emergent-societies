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
            availability.  Higher values mean fewer cooperative rewards are
            granted by the environment (probability of reward = ``1 -
            scarcity_level``).
        communication_enabled: When ``False``, agents skip the communication
            step before deciding their action.
    """

    num_agents: int = 100
    num_steps: int = 500
    initial_resources: int = 10
    resource_distribution: str = "uniform"
    scarcity_level: float = 0.5
    communication_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation suitable for logging or serialisation."""
        return {
            "num_agents": self.num_agents,
            "num_steps": self.num_steps,
            "initial_resources": self.initial_resources,
            "resource_distribution": self.resource_distribution,
            "scarcity_level": self.scarcity_level,
            "communication_enabled": self.communication_enabled,
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
        with open(path) as f:
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
            f"communication_enabled={self.communication_enabled})"
        )
