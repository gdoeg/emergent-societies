# Agent Class Documentation

## Overview

The `Agent` class is a minimal deterministic implementation for multi-agent simulations. It models autonomous agents that can interact in a shared environment through cooperation, resource trading, and communication.

## Key Features

- **Deterministic Behavior**: All decisions are based on fixed parameters, not randomness
- **Resource Management**: Agents own and trade resources with validation
- **Memory System**: All interactions are logged for analysis and future LLM integration
- **Backwards Compatible**: Works with existing simulation code

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `agent_id` | any | Unique identifier for the agent |
| `resources` | int/float | Numeric value representing owned resources (default: 10) |
| `cooperation_tendency` | float | Value between 0 and 1 representing likelihood to cooperate (default: 0.5) |
| `memory_log` | list | List of dictionaries storing past interactions and decisions |
| `id` | any | Alias for `agent_id` (backwards compatibility) |
| `alive` | bool | Whether the agent is active (default: True) |

## Methods

### `__init__(agent_id, resources=10, cooperation_tendency=0.5)`

Initialize a new agent.

**Parameters:**
- `agent_id`: Unique identifier for the agent
- `resources`: Initial resources (default: 10)
- `cooperation_tendency`: Float between 0 and 1 (default: 0.5, clamped to [0, 1])

**Example:**
```python
from simulation.agent import Agent

# Create agent with default values
agent1 = Agent(agent_id=1)

# Create agent with custom parameters
agent2 = Agent(agent_id=2, resources=100, cooperation_tendency=0.8)
```

### `decide_action(other_agent)`

Deterministically decide whether to cooperate or defect.

**Parameters:**
- `other_agent`: The agent to interact with (can be Agent object or None)

**Returns:**
- `"cooperate"` if `cooperation_tendency >= 0.5`
- `"defect"` if `cooperation_tendency < 0.5` or agent is not alive/has no resources

**Example:**
```python
agent1 = Agent(agent_id=1, cooperation_tendency=0.7)
agent2 = Agent(agent_id=2, cooperation_tendency=0.3)

decision1 = agent1.decide_action(agent2)  # Returns "cooperate"
decision2 = agent2.decide_action(agent1)  # Returns "defect"
```

### `trade(other_agent, trade_amount)`

Transfer resources between agents with validation.

**Parameters:**
- `other_agent`: The agent to trade with (must be a valid Agent object)
- `trade_amount`: Amount to transfer (positive = give, negative = receive, zero = no-op)

**Returns:**
- `True` if trade was successful
- `False` if trade failed (insufficient resources or invalid agent)

**Example:**
```python
agent1 = Agent(agent_id=1, resources=100)
agent2 = Agent(agent_id=2, resources=50)

# Agent1 gives 20 resources to Agent2
success = agent1.trade(agent2, 20)
# agent1.resources = 80, agent2.resources = 70

# Agent1 receives 10 resources from Agent2
success = agent1.trade(agent2, -10)
# agent1.resources = 90, agent2.resources = 60

# Failed trade (insufficient resources)
success = agent2.trade(agent1, 100)  # Returns False
# Resources unchanged
```

### `communicate(other_agent, message)`

Send a message to another agent and log the communication.

**Parameters:**
- `other_agent`: The agent to communicate with (can be None for broadcasting)
- `message`: The message content (any type)

**Example:**
```python
agent1 = Agent(agent_id=1)
agent2 = Agent(agent_id=2)

agent1.communicate(agent2, "Let's cooperate!")
# Logs sent message in agent1.memory_log
# Logs received message in agent2.memory_log

agent1.communicate(None, "Broadcasting to all")
# Logs sent message in agent1.memory_log only
```

## Memory Log Structure

All actions are automatically logged in `memory_log` with structured data:

### Decision Log Entry
```python
{
    "action": "decide_action",
    "decision": "cooperate" | "defect",
    "other_agent_id": agent_id | None,
    "my_resources": int/float,
    "my_cooperation_tendency": float
}
```

### Trade Log Entry (Success)
```python
{
    "action": "trade",
    "status": "success",
    "other_agent_id": agent_id,
    "amount": int/float,
    "my_resources_after": int/float,
    "other_agent_resources_after": int/float
}
```

### Trade Log Entry (Failure)
```python
{
    "action": "trade",
    "status": "failed",
    "reason": "insufficient_resources" | "other_agent_insufficient_resources" | "invalid_other_agent",
    "other_agent_id": agent_id | None,
    "attempted_amount": int/float,
    "my_resources": int/float
}
```

### Communication Log Entry (Sent)
```python
{
    "action": "communicate",
    "other_agent_id": agent_id | None,
    "message": any,
    "my_resources": int/float
}
```

### Communication Log Entry (Received)
```python
{
    "action": "received_communication",
    "from_agent_id": agent_id,
    "message": any,
    "my_resources": int/float
}
```

## Complete Example

```python
from simulation.agent import Agent

# Create agents with different cooperation tendencies
alice = Agent(agent_id="Alice", resources=100, cooperation_tendency=0.8)
bob = Agent(agent_id="Bob", resources=80, cooperation_tendency=0.3)

# Decision making
alice_decision = alice.decide_action(bob)  # "cooperate"
bob_decision = bob.decide_action(alice)    # "defect"

# Communication
alice.communicate(bob, "Let's work together!")
bob.communicate(alice, "I prefer to work alone.")

# Trading
alice.trade(bob, 20)  # Alice gives 20 to Bob
# alice.resources = 80, bob.resources = 100

# Review memory
print(f"Alice's memory: {len(alice.memory_log)} entries")
print(f"Bob's memory: {len(bob.memory_log)} entries")

# Inspect specific actions
for entry in alice.memory_log:
    print(f"Action: {entry['action']}")
```

## Design Principles

1. **Deterministic**: No random behavior - same inputs always produce same outputs
2. **Safe**: Extensive validation prevents invalid operations
3. **Transparent**: All actions are logged for inspection and learning
4. **Extensible**: Simple structure allows easy integration with LLM decision-making
5. **Compatible**: Works with existing simulation infrastructure

## Future Extensions

The design supports easy integration with:
- LLM-driven decision making (use memory_log as context)
- Complex game theory strategies (modify decide_action logic)
- Multi-step negotiation protocols (build on communicate method)
- Learning algorithms (analyze memory_log patterns)
