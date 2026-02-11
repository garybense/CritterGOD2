# CritterGOD Quickstart

## Installation

### 1. Install Python dependencies

```bash
# Using pip
pip3 install numpy

# Or install all dependencies at once
pip3 install -r requirements.txt
```

### 2. Verify installation

```bash
# Run the example simulation
python3 examples/simple_network.py
```

### 3. Run tests (after installing pytest)

```bash
pip3 install pytest
python3 -m pytest tests/ -v
```

## What's Implemented (Phase 1)

✅ **Core Neural Engine**:
- Leaky integrate-and-fire neurons
- Bidirectional synapses (excitatory/inhibitory)
- STDP plasticity (Hebbian learning)
- Sensory-motor neuron types
- Random network wiring patterns

## Quick Example

```python
from core.neural import Neuron, Synapse, NeuralNetwork
from core.neural.neuron import NeuronType

# Create network
network = NeuralNetwork(enable_plasticity=True)

# Add neurons
for i in range(100):
    neuron = Neuron(neuron_id=i)
    network.add_neuron(neuron)

# Create synapses
network.create_random_synapses(synapses_per_neuron=40)

# Run simulation
for step in range(1000):
    network.update(dt=1.0)
    activity = network.get_activity_level()
    print(f"Step {step}: activity={activity:.3f}")
```

## Next Steps

See AGENTS.md for:
- Phase 2: Evolution system implementation
- Phase 3: Advanced features (Circuit8, drugs, etc.)
- Phase 4: Optimization strategies

## Project Structure

```
CritterGOD/
├── core/
│   └── neural/          # Neural network implementation
│       ├── neuron.py    # Leaky integrate-and-fire neurons
│       ├── synapse.py   # Synapses with STDP plasticity
│       └── network.py   # Neural network container
├── examples/            # Example simulations
├── tests/              # Unit tests
├── AGENTS.md           # Documentation for AI agents
├── CRITTERS_CODEBASE.md # Analysis of source material
└── requirements.txt    # Python dependencies
```

## Resources

- **AGENTS.md**: Complete development guide
- **CRITTERS_CODEBASE.md**: Analysis of source material (~12K files)
- **ARCHITECTURE.md**: System architecture details
- **DISCOVERY.md**: Source codebase exploration

## Troubleshooting

**Import errors**: Make sure numpy is installed (`pip3 install numpy`)

**No module named 'core'**: Run examples from the project root directory

**Tests fail**: Install pytest (`pip3 install pytest`)
