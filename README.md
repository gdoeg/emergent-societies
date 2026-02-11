# Emergent Societies

A multi-agent simulation framework for modeling autonomous agents interacting in a shared environment.

## Features

- **Deterministic Agent Behavior**: Agents make predictable decisions based on cooperation tendencies
- **Resource Management**: Agents can trade resources with validation
- **Communication System**: Agents can exchange messages and log interactions
- **Memory Logging**: All actions are recorded for analysis and future learning
- **Extensible Design**: Easy integration with LLM-driven decision-making

## Quick Start

```python
from simulation.agent import Agent

# Create agents with different characteristics
alice = Agent(agent_id="Alice", resources=100, cooperation_tendency=0.8)
bob = Agent(agent_id="Bob", resources=80, cooperation_tendency=0.3)

# Agents decide whether to cooperate or defect
alice.decide_action(bob)  # "cooperate" (tendency >= 0.5)
bob.decide_action(alice)  # "defect" (tendency < 0.5)

# Agents can communicate
alice.communicate(bob, "Let's work together!")

# Agents can trade resources
alice.trade(bob, 20)  # Alice gives 20 resources to Bob

# Review interaction history
print(alice.memory_log)  # See all of Alice's actions
```

## Documentation

- [Agent Class Documentation](AGENT_DOCUMENTATION.md) - Complete API reference and examples
- Run the simulation: `python main.py`

## Agent Class

The core `Agent` class provides:

- **Attributes**: `agent_id`, `resources`, `cooperation_tendency`, `memory_log`
- **Methods**: `decide_action()`, `trade()`, `communicate()`
- **Deterministic Behavior**: All decisions based on fixed parameters
- **Comprehensive Logging**: Every action recorded in structured format

See [AGENT_DOCUMENTATION.md](AGENT_DOCUMENTATION.md) for detailed documentation.

## Project Structure

```
emergent-societies/
├── simulation/
│   ├── agent.py       # Agent class implementation
│   ├── world.py       # World environment
│   └── simulation.py  # Simulation runner
├── metrics/
│   ├── logger.py      # Logging utilities
│   └── metrics.py     # Metrics collection
├── main.py            # Main simulation entry point
└── README.md          # This file
```

## Requirements

- Python 3.6+
- No external dependencies (uses only standard library)

## License

MIT
