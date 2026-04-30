import random
import logging
import asyncio
from collections import defaultdict
from typing import Any, Dict

from simulation.config import SimulationConfig
from simulation.policies.llm_policy import LLMPolicy, batch_agents

# Set up logger for environment debugging
logger = logging.getLogger(__name__)


class Environment:
    """
    Environment class for multi-agent simulation.
    Manages agent population, interactions, and simulation cycles.
    
    Note: While agent pairing uses randomization, the simulation can be made
    deterministic by setting a random seed before creating the environment.
    """
    
    def __init__(self, agents, resource_pool=None, config: SimulationConfig = None):
        """
        Initialize the environment with an agent population.
        
        Args:
            agents: list of Agent objects
            resource_pool: optional shared resource pool (default None)
            config: optional SimulationConfig; controls scarcity_level and
                communication_enabled behaviour
        """
        self.agents = agents
        self.cycle_count = 0
        self.resource_pool = resource_pool
        self.config = config if config is not None else SimulationConfig()
        self.interaction_graph: defaultdict[Any, set] = defaultdict(set)
        self._inject_agent_context()
        # Spread first-update timestamps so agents don't all become due at the
        # same cycle (thundering herd).  Without this, all 100 agents fire
        # simultaneously at cycle (decision_interval - 1), serialising every
        # LLM batch call into one huge blocking burst.
        self._stagger_strategy_update_offsets()

    def _stagger_strategy_update_offsets(self) -> None:
        """Distribute initial strategy-update timestamps across [0, decision_interval).

        All agents start with ``last_strategy_update_step = -1``, which means
        they all become due at the same cycle (``decision_interval - 1``).  This
        assigns each agent a distinct random offset so that roughly
        ``num_agents / decision_interval`` agents update per cycle instead of all
        at once, spreading LLM load evenly over time.
        """
        decision_interval = getattr(self.config, "decision_interval", 15)
        for agent in self.agents:
            # Negative offset puts the "last update" conceptually in the past;
            # the agent becomes due once cycle_count reaches the gap.
            agent.last_strategy_update_step = -(random.randint(1, decision_interval))

    def _inject_agent_context(self) -> None:
        """Attach environment conditions and population snapshot to each agent."""
        base_context = {
            "scarcity_level": self.config.scarcity_level,
            "redistribution_strength": self.config.redistribution_strength,
            "enable_elite_advantage": self.config.enable_elite_advantage,
        }
        snapshot = [agent.resources for agent in self.agents]
        for agent in self.agents:
            if hasattr(agent, "simulation_context"):
                agent.simulation_context = dict(base_context)
            if hasattr(agent, "population_resources_snapshot"):
                agent.population_resources_snapshot = list(snapshot)

    def _interaction_outcome(self, my_action: str, other_action: str) -> str:
        """Map action pairs to a per-agent interaction outcome label."""
        if my_action == "cooperate" and other_action == "cooperate":
            return "mutual_cooperation"
        if my_action == "defect" and other_action == "defect":
            return "mutual_defection"
        if my_action == "defect" and other_action == "cooperate":
            return "exploited_other"
        if my_action == "cooperate" and other_action == "defect":
            return "was_exploited"
        return "unknown"

    def _payoff_reward(self, my_action: str, other_action: str) -> float:
        """Return per-agent payoff from the standard 2x2 cooperation matrix."""
        if my_action == "cooperate" and other_action == "cooperate":
            return 3.0
        if my_action == "defect" and other_action == "cooperate":
            return 5.0
        if my_action == "cooperate" and other_action == "defect":
            return 0.0
        if my_action == "defect" and other_action == "defect":
            return 1.0
        return 0.0

    def _latest_decision_record(self, agent, other_agent_id):
        """Return the latest decision record for a specific opponent."""
        for record in reversed(agent.interaction_memory):
            if (
                record.get("action") == "decide_action"
                and record.get("other_agent_id") == other_agent_id
            ):
                return record
        return None

    def _log_decision_outcome(self, agent, other_agent, my_action: str, other_action: str) -> None:
        """Log parsed decision and realized outcome for analysis."""
        decision_record = self._latest_decision_record(agent, other_agent.agent_id)
        outcome = self._interaction_outcome(my_action, other_action)
        agent.interaction_memory.append(
            {
                "action": "decision_outcome",
                "policy": decision_record.get("policy") if decision_record else None,
                "agent_id": agent.agent_id,
                "other_agent_id": other_agent.agent_id,
                "prompt_summary": decision_record.get("prompt_summary") if decision_record else None,
                "raw_response": decision_record.get("raw_response") if decision_record else None,
                "parsed_action": my_action,
                "other_action": other_action,
                "resulting_outcome": outcome,
                "cycle": self.cycle_count,
            }
        )

    async def _safe_maybe_update_strategy_async(self, agent, decision_interval: int) -> None:
        """Run one agent strategy update without allowing task exceptions to bubble."""
        try:
            await agent.maybe_update_strategy_async(self.cycle_count, decision_interval)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Async strategy update failed agent_id=%s error=%s",
                getattr(agent, "agent_id", "unknown"),
                exc,
            )

    async def _update_strategies_async(self, decision_interval: int) -> None:
        """Update standing strategies concurrently using batching for shared LLM policies."""
        llm_groups: dict[int, dict[str, Any]] = {}
        non_batch_agents = []

        for agent in self.agents:
            if not hasattr(agent, "maybe_update_strategy_async"):
                continue
            policy = getattr(agent, "policy", None)
            if (
                isinstance(policy, LLMPolicy)
                and getattr(policy, "enable_async", True)
                and hasattr(policy, "generate_strategies_batch_async")
            ):
                group = llm_groups.setdefault(id(policy), {"policy": policy, "agents": []})
                group["agents"].append(agent)
            else:
                non_batch_agents.append(agent)

        tasks = [
            self._safe_maybe_update_strategy_async(agent, decision_interval)
            for agent in non_batch_agents
        ]

        for group in llm_groups.values():
            policy: LLMPolicy = group["policy"]
            agents = group["agents"]
            due_agents = [
                agent
                for agent in agents
                if self.cycle_count - getattr(agent, "last_strategy_update_step", -1) >= decision_interval
            ]
            # Collect all non-empty batches, then dispatch them concurrently.
            # The semaphore inside LLMPolicy caps actual in-flight calls at
            # max_concurrent_llm_calls (default 4), so this never overwhelms
            # Ollama while still eliminating the sequential-await bottleneck.
            batches = [b for b in batch_agents(due_agents, policy.batch_size) if b]
            if not batches:
                continue

            batch_results = await asyncio.gather(
                *[policy.generate_strategies_batch_async(batch) for batch in batches],
                return_exceptions=True,
            )

            for agent_batch, result in zip(batches, batch_results):
                if isinstance(result, Exception):
                    logger.warning("Batch strategy update failed; will fallback per-agent. error=%s", result)
                    mapping: Dict[Any, str] = {}
                else:
                    mapping = result

                for agent in agent_batch:
                    action = mapping.get(agent.agent_id)
                    if action is None:
                        # Agent-level fallback path if batch omitted this agent.
                        tasks.append(self._safe_maybe_update_strategy_async(agent, decision_interval))
                        continue
                    if action != agent.strategy and random.random() < 0.3:
                        agent.strategy = action
                    agent.last_strategy_update_step = self.cycle_count

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def step(self):
        """
        Execute one simulation cycle:
        - Trigger periodic LLM strategy updates for all agents
        - Apply per-step resource decay based on scarcity_level
        - Randomly pair agents (capped at max_pairs_per_step)
        - Optionally exchange intent signals (gated by communication_enabled)
        - Resolve interactions using each agent's standing strategy (no LLM)
        - If both agents cooperate, attempt to grant a resource reward
          (probability reduced by scarcity_level; richer agent receives an
          amplified reward when enable_elite_advantage is True)
        - Log each interaction to agent.interaction_memory (capped at memory_size)
        - Redistribute resources between pairs using redistribution_strength
          (only when gap exceeds trade_threshold)
        - Increment cycle_count
        """
        communication_enabled = self.config.communication_enabled
        scarcity_level = self.config.scarcity_level
        redistribution_strength = self.config.redistribution_strength
        trade_threshold = self.config.trade_threshold
        elite_advantage_factor = self.config.elite_advantage_factor
        enable_elite_advantage = self.config.enable_elite_advantage
        decision_interval = getattr(self.config, "decision_interval", 15)
        memory_size = getattr(self.config, "memory_size", 50)

        # --- Step 1: Periodic strategy updates (only LLM calls happen here) ---
        # Each agent checks whether its update interval has elapsed; if so, the
        # LLM is queried once for a new standing strategy.  All interactions
        # below use that strategy without any further LLM calls.
        async_mode_enabled = getattr(self.config, "enable_async_llm", True)
        if async_mode_enabled:
            try:
                asyncio.run(self._update_strategies_async(decision_interval))
            except RuntimeError:
                # If already in an event loop, keep simulation progressing via sync fallback.
                for agent in self.agents:
                    if hasattr(agent, "maybe_update_strategy"):
                        agent.maybe_update_strategy(self.cycle_count, decision_interval)
        else:
            for agent in self.agents:
                if hasattr(agent, "maybe_update_strategy"):
                    agent.maybe_update_strategy(self.cycle_count, decision_interval)

        # Per-step scarcity decay: each agent loses a small fraction of a
        # resource unit proportional to scarcity_level.  A decay_factor of 0.1
        # means agents lose at most 0.1 resources per step at full scarcity.
        _DECAY_FACTOR = 0.1
        if scarcity_level > 0.0:
            decay = scarcity_level * _DECAY_FACTOR
            for agent in self.agents:
                agent.resources = max(0.0, agent.resources - decay)

        # Log resource distribution before step (guarded so list build is skipped when DEBUG is off)
        if logger.isEnabledFor(logging.DEBUG):
            resources_before = [agent.resources for agent in self.agents]
            logger.debug("Cycle %d: Resources before step: %s", self.cycle_count, resources_before)
            if resources_before:
                logger.debug(
                    "Cycle %d: Resource stats - min: %s, max: %s, sum: %s",
                    self.cycle_count, min(resources_before), max(resources_before), sum(resources_before),
                )

        pairs = self.pair_agents()
        logger.debug("Cycle %d: Created %d pairs from %d agents", self.cycle_count, len(pairs), len(self.agents))
        
        cooperation_count = 0
        reward_count = 0
        
        for agent1, agent2 in pairs:
            # Optional pre-decision communication exchange
            if communication_enabled:
                if hasattr(agent1, 'communicate'):
                    agent1.communicate(agent2, "intent")
                if hasattr(agent2, 'communicate'):
                    agent2.communicate(agent1, "intent")

            # --- Step 2: Fast strategy-based action lookup (no LLM) ---
            # Prefer get_action (strategy model) over decide_action (legacy LLM path).
            if hasattr(agent1, 'get_action'):
                action1 = agent1.get_action(agent2)
            elif hasattr(agent1, 'decide_action'):
                action1 = agent1.decide_action(agent2)
            else:
                action1 = None

            if hasattr(agent2, 'get_action'):
                action2 = agent2.get_action(agent1)
            elif hasattr(agent2, 'decide_action'):
                action2 = agent2.decide_action(agent1)
            else:
                action2 = None
            
            logger.debug(f"Cycle {self.cycle_count}: Agent {agent1.agent_id} -> {action1}, Agent {agent2.agent_id} -> {action2}")
            
            # Record interaction for both agents
            if hasattr(agent1, 'record_interaction') and hasattr(agent2, 'agent_id'):
                agent1.record_interaction(agent2.agent_id)
            if hasattr(agent2, 'record_interaction') and hasattr(agent1, 'agent_id'):
                agent2.record_interaction(agent1.agent_id)

            if action1 is not None and action2 is not None:
                self._log_decision_outcome(agent1, agent2, action1, action2)
                self._log_decision_outcome(agent2, agent1, action2, action1)

            # Update interaction graph (undirected edge between agent1 and agent2)
            if hasattr(agent1, 'agent_id') and hasattr(agent2, 'agent_id'):
                self.interaction_graph[agent1.agent_id].add(agent2.agent_id)
                self.interaction_graph[agent2.agent_id].add(agent1.agent_id)

            # Update lightweight memory/trust for both agents based on observed behaviour
            if hasattr(agent1, 'update_memory') and action2 is not None and hasattr(agent2, 'agent_id'):
                agent1.update_memory(agent2.agent_id, action2)
            if hasattr(agent2, 'update_memory') and action1 is not None and hasattr(agent1, 'agent_id'):
                agent2.update_memory(agent1.agent_id, action1)
            
            # Update trust based on observed partner behavior
            if action1 == "cooperate" and action2 == "cooperate":
                # Both cooperated - increase trust
                if hasattr(agent1, 'update_trust') and hasattr(agent2, 'agent_id'):
                    agent1.update_trust(agent2.agent_id, 0.05)
                if hasattr(agent2, 'update_trust') and hasattr(agent1, 'agent_id'):
                    agent2.update_trust(agent1.agent_id, 0.05)
            else:
                # Handle defection - decrease trust toward a partner observed to defect
                if hasattr(agent1, 'update_trust') and hasattr(agent2, 'agent_id'):
                    if action2 == "defect":
                        agent1.update_trust(agent2.agent_id, -0.05)
                if hasattr(agent2, 'update_trust') and hasattr(agent1, 'agent_id'):
                    if action1 == "defect":
                        agent2.update_trust(agent1.agent_id, -0.05)
            
            # Learning rewards use a full payoff matrix so non-mutual outcomes still
            # carry signal for strategy updates and LLM memory summaries.
            reward1 = self._payoff_reward(action1, action2) if action1 is not None and action2 is not None else 0.0
            reward2 = self._payoff_reward(action2, action1) if action1 is not None and action2 is not None else 0.0

            # If both agents cooperate, grant each a resource reward from the environment.
            # Delegated through Agent.receive_resource() for consistent validation and logging.
            # scarcity_level controls reward availability: higher scarcity = lower probability.
            # When enable_elite_advantage is True, the wealthier agent receives a bonus on
            # top of the base reward, scaled by elite_advantage_factor.
            # resource_gain1/resource_gain2 represent resource gain from this interaction:
            #   mutual cooperation → base_reward (≥1) if scarcity roll passes, else 0.0
            #   any defection scenario → 0.0 (no cooperative surplus is created)
            resource_gain1 = resource_gain2 = 0.0
            if action1 == "cooperate" and action2 == "cooperate":
                cooperation_count += 1
                scarcity_roll = random.random()
                logger.debug(f"Cycle {self.cycle_count}: Both cooperated! Scarcity roll: {scarcity_roll:.3f}, threshold: {scarcity_level}")
                
                if scarcity_roll > scarcity_level:
                    reward_count += 1
                    base_reward = 1

                    if enable_elite_advantage:
                        # Determine which agent is wealthier and apply the advantage factor
                        # to their reward.  A bonus > 1.0 (factor > 2.0) is handled by
                        # granting floor(bonus) guaranteed extra units plus a fractional
                        # probability of one more, so the expected extra reward equals bonus.
                        bonus = elite_advantage_factor - 1.0  # e.g. 0.2 for factor=1.2
                        extra = int(bonus) + (1 if random.random() < (bonus % 1) else 0)
                        if agent1.resources >= agent2.resources:
                            resource_gain1 = base_reward + extra
                            resource_gain2 = base_reward
                        else:
                            resource_gain1 = base_reward
                            resource_gain2 = base_reward + extra
                    else:
                        resource_gain1 = resource_gain2 = base_reward

                    logger.debug(f"Cycle {self.cycle_count}: Granting resources to agents {agent1.agent_id} and {agent2.agent_id}")
                    if hasattr(agent1, 'receive_resource'):
                        agent1.receive_resource(resource_gain1, source="cooperation_reward")
                    if hasattr(agent2, 'receive_resource'):
                        agent2.receive_resource(resource_gain2, source="cooperation_reward")
                else:
                    logger.debug(f"Cycle {self.cycle_count}: Scarcity prevented reward (roll {scarcity_roll:.3f} <= {scarcity_level})")

            # --- Log this interaction to each agent's flat interaction_memory ---
            # Capped at memory_size; oldest entry evicted when limit is reached.
            # Keep explicit reward fields for both sides so learning signals are
            # always present in per-interaction memory entries.
            if action1 is not None and action2 is not None:
                if hasattr(agent1, 'interaction_memory'):
                    agent1.interaction_memory.append({
                        "step": self.cycle_count,
                        "opponent_id": agent2.agent_id,
                        "action": action1,
                        "opponent_action": action2,
                        "reward": reward1,
                    })
                    if len(agent1.interaction_memory) > memory_size:
                        agent1.interaction_memory.pop(0)

                if hasattr(agent2, 'interaction_memory'):
                    agent2.interaction_memory.append({
                        "step": self.cycle_count,
                        "opponent_id": agent1.agent_id,
                        "action": action2,
                        "opponent_action": action1,
                        "reward": reward2,
                    })
                    if len(agent2.interaction_memory) > memory_size:
                        agent2.interaction_memory.pop(0)
        
        # Redistribution: transfer resources from the richer agent to the poorer one
        # only when the gap exceeds trade_threshold.  The transfer amount is scaled by
        # redistribution_strength so that 0.0 disables redistribution entirely and 1.0
        # fully halves the gap each step.  When enable_elite_advantage is True, the
        # richer agent effectively loses less by dividing the transfer by
        # elite_advantage_factor.
        for agent1, agent2 in pairs:
            diff = agent1.resources - agent2.resources
            if abs(diff) <= trade_threshold or redistribution_strength == 0.0:
                continue

            transfer = redistribution_strength * (abs(diff) / 2.0)

            if enable_elite_advantage:
                # Elite agents (more resources) retain a larger share: dividing
                # the transfer by elite_advantage_factor reduces the amount taken
                # from the richer side.  This interacts multiplicatively with
                # redistribution_strength; at very low strengths the combined
                # effect can be negligible, which is intentional.
                transfer = transfer / elite_advantage_factor

            if transfer == 0.0:
                continue

            if diff > 0:
                # agent1 is richer — transfer from agent1 to agent2
                actual = min(transfer, agent1.resources)
                agent1.resources -= actual
                agent2.resources += actual
                logger.debug(
                    f"Cycle {self.cycle_count}: Redistribution {agent1.agent_id} -> {agent2.agent_id} "
                    f"amount={actual:.2f} (gap={diff:.2f})"
                )
            else:
                # agent2 is richer — transfer from agent2 to agent1
                actual = min(transfer, agent2.resources)
                agent2.resources -= actual
                agent1.resources += actual
                logger.debug(
                    f"Cycle {self.cycle_count}: Redistribution {agent2.agent_id} -> {agent1.agent_id} "
                    f"amount={actual:.2f} (gap={-diff:.2f})"
                )

        self.cycle_count += 1
        self._inject_agent_context()
        
        # Log resource distribution after step (guarded so list build is skipped when DEBUG is off)
        if logger.isEnabledFor(logging.DEBUG):
            resources_after = [agent.resources for agent in self.agents]
            logger.debug("Cycle %d: Resources after step: %s", self.cycle_count - 1, resources_after)
            if resources_after:
                avg = sum(resources_after) / len(resources_after)
                logger.debug(
                    "Cycle %d: Resource stats - min: %s, max: %s, sum: %s",
                    self.cycle_count - 1, min(resources_after), max(resources_after), sum(resources_after),
                )
                logger.debug(
                    "Cycle %d: Summary - %d cooperation pairs, %d rewards granted",
                    self.cycle_count - 1, cooperation_count, reward_count,
                )
                
                # Compute and log Gini coefficient for this step
                from metrics.economics import compute_gini
                gini = compute_gini(resources_after)
                logger.debug(
                    "Cycle %d: Gini coefficient = %.4f, Total wealth = %d, Avg wealth = %.2f",
                    self.cycle_count - 1, gini, sum(resources_after), avg,
                )
            else:
                logger.debug(
                    "Cycle %d: Summary - %d cooperation pairs, %d rewards granted (no agents)",
                    self.cycle_count - 1, cooperation_count, reward_count,
                )

    async def step_async(self) -> None:
        """Async variant of ``step()`` for FastAPI / async callers.

        Runs ``step()`` in a worker thread via ``asyncio.to_thread`` so that
        the internal ``asyncio.run()`` call inside ``step()`` can create its
        own event loop without conflicting with the caller's running loop.
        FastAPI's uvicorn event loop is not blocked while strategy updates
        and interactions execute.  The shared ``httpx.Client`` inside
        ``LLMPolicy`` remains alive across steps and reuses TCP connections.
        """
        await asyncio.to_thread(self.step)

    def pair_agents(self):
        """
        Randomly shuffle agents and return agent pairs for interaction.

        When ``config.max_pairs_per_step`` is set, pairing stops once that
        limit is reached so no unnecessary pair tuples are constructed for
        large agent populations.

        Returns:
            list of tuples: agent pairs
        """
        # Create a copy to avoid modifying the original list
        shuffled = self.agents.copy()
        random.shuffle(shuffled)

        # Cap interactions per step to prevent O(n²) LLM call growth.
        # Stop constructing pairs as soon as the limit is reached rather than
        # building all pairs first and then sampling — avoids wasted allocation.
        max_pairs = getattr(self.config, "max_pairs_per_step", None)
        pair_limit = max_pairs if max_pairs is not None else len(shuffled) // 2

        pairs = []
        for i in range(0, len(shuffled) - 1, 2):
            if len(pairs) >= pair_limit:
                break
            pairs.append((shuffled[i], shuffled[i + 1]))

        return pairs
    
    def get_resource_distribution(self):
        """
        Return list of agent resource values.
        
        Returns:
            list: resource values for all agents
        """
        return [agent.resources for agent in self.agents]

    def distribute_resources(self) -> None:
        """
        Set each agent's resources according to the configured distribution.

        ``resource_distribution == "uniform"`` (default): every agent receives
        exactly ``config.initial_resources``.
        ``resource_distribution == "random"``: each agent receives a value
        drawn uniformly from ``[1, config.initial_resources * 2]``.
        """
        initial_resources = self.config.initial_resources
        distribution = self.config.resource_distribution

        for agent in self.agents:
            if distribution == "random":
                agent.resources = random.randint(1, initial_resources * 2)
            else:
                agent.resources = initial_resources
