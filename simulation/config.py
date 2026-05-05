"""SimulationConfig: centralised configuration for the emergent-societies simulation."""

import json
import os
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
        policy_type: Which decision policy to assign to agents.
            ``"deterministic"`` uses the built-in stochastic-tendency logic.
            ``"llm"`` delegates decisions to an LLM via the OpenAI-compatible
            chat-completion API.
        llm_model: LLM model name used when ``policy_type`` is ``"llm"``
            (default: ``"llama-3.1-8b-instant"``).
        llm_api_base_url: Base URL for the chat-completion API.  Change this
            to use any OpenAI-compatible provider
            (default: ``"http://localhost:11434/v1"``).
        llm_timeout: HTTP request timeout in seconds for LLM API calls
            (default: ``3``).
        max_concurrent_llm_calls: Maximum in-flight async LLM calls at once
            (default: ``4``).
        llm_batch_size: Number of agents per batched LLM strategy update
            (default: ``8``).
        enable_async_llm: When ``True``, strategy updates run via async gather
            with concurrency controls and batching (default: ``True``).
        debug_llm: Emit detailed LLM debug logs when ``True``.
        decision_interval: LLM agents only call the model every
            ``decision_interval`` steps; the last decision is reused between
            calls to reduce API load (default: ``15``). Can be overridden via
            DECISION_INTERVAL environment variable.
        llm_max_retries: Maximum number of retry attempts for LLM calls
            (default: ``3``). Can be overridden via LLM_MAX_RETRIES env var.
        llm_models: Comma-separated list of LLM models to use in order of
            preference. Models are tried in order with retry/backoff before
            falling back to deterministic strategy. Can be overridden via
            LLM_MODELS environment variable
            (default: ``"llama-3.1-8b-instant,llama-3.3-70b-versatile"``).
        max_pairs_per_step: Maximum number of agent pairs to interact per
            simulation step.  Random sampling is used when the pool of
            shuffled pairs exceeds this limit (default: ``15``).
        chunk_size: Number of simulation steps executed per ``run_steps``
            chunk.  Callers can use ``run_steps`` to avoid blocking for a
            full ``num_steps`` run (default: ``10``).
        memory_size: Maximum number of interaction records kept in each
            agent's ``interaction_memory`` list.  Older entries are evicted
            when the limit is reached (default: ``50``).
    """

    config_name: str = ""
    num_agents: int = 100
    num_steps: int = 100
    initial_resources: int = 10
    resource_distribution: str = "uniform"
    scarcity_level: float = 0.2
    communication_enabled: bool = True
    trade_threshold: int = 5
    top_n_leaders: int = 3
    redistribution_strength: float = 0.5
    elite_advantage_factor: float = 1.2
    enable_elite_advantage: bool = False
    policy_type: str = "llm"
    llm_model: str = "llama-3.1-8b-instant"
    llm_api_base_url: str = "http://localhost:11434/v1"
    # Timeout in seconds for LLM API calls.  5 s gives headroom for the
    # 700–1000 ms Ollama responses seen in practice while still failing fast.
    llm_timeout: float = 5.0
    max_concurrent_llm_calls: int = 4
    llm_batch_size: int = 8
    enable_async_llm: bool = True
    debug_llm: bool = False
    # Strategy update interval: agents ask the LLM for a new strategy every K steps
    decision_interval: int = 15
    # LLM retry configuration for production resilience
    llm_max_retries: int = 3
    # Multi-model fallback strategy: lightweight models first
    llm_models: str = "llama-3.1-8b-instant,llama-3.3-70b-versatile"
    # Cap pairwise interactions per step to bound O(n²) LLM call growth
    max_pairs_per_step: int = 15
    # Chunk size for run_steps(); avoids blocking on a full num_steps run
    chunk_size: int = 10
    # Maximum entries retained in each agent's flat interaction_memory list
    memory_size: int = 50

    def __post_init__(self) -> None:
        """Validate documented configuration constraints and apply environment overrides."""
        # Apply environment variable overrides
        if os.getenv("DECISION_INTERVAL"):
            self.decision_interval = int(os.getenv("DECISION_INTERVAL"))
        if os.getenv("LLM_MAX_RETRIES"):
            self.llm_max_retries = int(os.getenv("LLM_MAX_RETRIES"))
        if os.getenv("LLM_MODELS"):
            self.llm_models = os.getenv("LLM_MODELS")
        if os.getenv("LLM_MAX_CONCURRENCY"):
            self.max_concurrent_llm_calls = int(os.getenv("LLM_MAX_CONCURRENCY"))
        
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

        valid_policy_types = {"deterministic", "llm"}
        if self.policy_type not in valid_policy_types:
            raise ValueError(
                "policy_type must be one of "
                f"{sorted(valid_policy_types)}, got {self.policy_type!r}"
            )

        if self.llm_timeout < 1:
            raise ValueError("llm_timeout must be at least 1 second")

        if self.max_concurrent_llm_calls < 1:
            raise ValueError("max_concurrent_llm_calls must be at least 1")

        if self.llm_batch_size < 1:
            raise ValueError("llm_batch_size must be at least 1")

        if self.decision_interval < 1:
            raise ValueError("decision_interval must be at least 1")

        if self.llm_max_retries < 1:
            raise ValueError("llm_max_retries must be at least 1")

        if self.max_pairs_per_step < 1:
            raise ValueError("max_pairs_per_step must be at least 1")

        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")

        if self.memory_size < 1:
            raise ValueError("memory_size must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation suitable for logging or serialisation."""
        return {
            "config_name": self.config_name,
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
            "policy_type": self.policy_type,
            "llm_model": self.llm_model,
            "llm_api_base_url": self.llm_api_base_url,
            "llm_timeout": self.llm_timeout,
            "max_concurrent_llm_calls": self.max_concurrent_llm_calls,
            "llm_batch_size": self.llm_batch_size,
            "enable_async_llm": self.enable_async_llm,
            "debug_llm": self.debug_llm,
            "decision_interval": self.decision_interval,
            "llm_max_retries": self.llm_max_retries,
            "llm_models": self.llm_models,
            "max_pairs_per_step": self.max_pairs_per_step,
            "chunk_size": self.chunk_size,
            "memory_size": self.memory_size,
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
            f"config_name={self.config_name!r}, "
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
            f"enable_elite_advantage={self.enable_elite_advantage}, "
            f"policy_type={self.policy_type!r}, "
            f"llm_model={self.llm_model!r}, "
            f"llm_api_base_url={self.llm_api_base_url!r}, "
            f"llm_timeout={self.llm_timeout}, "
            f"max_concurrent_llm_calls={self.max_concurrent_llm_calls}, "
            f"llm_batch_size={self.llm_batch_size}, "
            f"enable_async_llm={self.enable_async_llm}, "
            f"debug_llm={self.debug_llm}, "
            f"decision_interval={self.decision_interval}, "
            f"max_pairs_per_step={self.max_pairs_per_step}, "
            f"chunk_size={self.chunk_size}, "
            f"memory_size={self.memory_size})"
        )
