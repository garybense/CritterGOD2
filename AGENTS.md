# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Status

**Phase 10 COMPLETE** - Research platform with configuration UI and statistics!
**Phase 9d COMPLETE** - Complete sensory-motor integration for emergent behaviors!
**Phase 9c COMPLETE** - Drug-modulated synaptic plasticity!

Core systems operational:
- ✅ Spiking neural networks with STDP plasticity (Phase 1)
- ✅ Genetic evolution with 7 mutation types (Phase 2)
- ✅ Circuit8 telepathic canvas with 1024 depth layers (Phase 3)
- ✅ Psychopharmacology with 5 molecule types (Phase 3)
- ✅ Energy metabolism with starvation mechanics (Phase 3)
- ✅ Collective creature simulation with democratic voting (Phase 3)
- ✅ Audio synthesis from neural activity (Phase 4a)
- ✅ Evolutionary text generation with markov chains (Phase 4b)
- ✅ Visual pattern generation with retinal feedback (Phase 4c)
- ✅ Multi-modal creatures: see, speak, think, create (Phase 4d)
- ✅ Continuous synapse weakening (automatic pruning, Phase 5a)
- ✅ Weight clamping (±5.0 limits, Phase 5a)
- ✅ Bidirectional thresholds (excitation OR inhibition, Phase 5a)
- ✅ Higher inhibitory ratio (30% from CritterGOD4, Phase 5a)
- ✅ Entity introspection (to_dict() debugging, Phase 5b)
- ✅ Adam distance tracking (evolutionary lineage, Phase 5b)
- ✅ Force-directed creature layout (genetic clustering, Phase 5c)
- ✅ Neural network visualization (connectivity-based, Phase 5c)
- ✅ 3D OpenGL visualization with orbital camera (Phase 7-8)
- ✅ Procedural 3D body generation from genetics (Phase 9a)
- ✅ Morphological evolution visible across generations (Phase 9a)
- ✅ Drug-responsive body appearance (pulsing, color shifts, Phase 9a)
- ✅ Food/resource system with Poisson disk sampling (Phase 9b)
- ✅ Hunger-driven food seeking behavior (Phase 9b)
- ✅ Addiction-driven drug seeking with tolerance & withdrawal (Phase 9b)
- ✅ Resource-seeking creatures with emergent survival patterns (Phase 9b)
- ✅ Markov thought generation (genetic language evolution, Phase 9c)
- ✅ Dynamic reproduction with genetic inheritance (Phase 9c)
- ✅ Psychedelic pattern generation to Circuit8 (Phase 9c)
- ✅ Drug-modulated synaptic plasticity (Phase 9c) **THE CRITICAL INTEGRATION**
- ✅ Complete sensory integration: vision, proprioception, interoception, chemoreception (Phase 9d)
- ✅ Retinal vision system (256 visual neurons reading Circuit8, Phase 9d)
- ✅ Multimodal sensory fusion for emergent cross-modal associations (Phase 9d)
- ✅ Audio synthesis from neural activity with drug modulation (Phase 9d)
- ✅ Complete sensory-motor feedback loop operational (Phase 9d)
- ✅ Creature inspector with complete state visualization (Phase 9c)
- ✅ Custom physics engine with Verlet integration (Phase 10a)
- ✅ Neural motor control → physics forces (Phase 10a)
- ✅ Collision-based resource consumption (Phase 10a)
- ✅ Realistic movement with gravity, friction, damping (Phase 10a)
- ✅ Configuration system with 45+ tunable parameters (Phase 10)
- ✅ Real-time statistics tracking with graphs (Phase 10)
- ✅ Event logging in critterdrug style (Phase 10)
- ✅ Configuration profiles (save/load) (Phase 10)
- ✅ Population management with culling (Phase 10)
- ✅ Complete research platform UI (Phase 10)
- ✅ Neural activity visualization Mode 2 (firing neurons + activity rings)
- ✅ Social learning visualization Mode 7 (observation lines + learning circles)

This is a working artificial life system integrating innovations from 15+ years of flamoot's research. See README.md, PHASE4_COMPLETE.md, PHASE5A_COMPLETE.md, PHASE5C_COMPLETE.md, PHASE9A_COMPLETE.md, PHASE9B_COMPLETE.md, PHASE9C_PSYCHOPHARMACOLOGY.md, PHASE10A_COMPLETE.md, and FLAMOOT_DISCOVERIES_SYNTHESIS.md for details.

## Project Overview

CritterGOD is a unified framework for evolutionary artificial life combining:
- Spiking neural networks with synaptic plasticity
- Genetic algorithms and body/brain evolution
- Psychopharmacological simulation (neurotransmitter effects)
- Morphic fields (collective memory via shared canvas)
- Real-time audio-visual synthesis from neural activity
- Generative text systems (Markov chains with genetic selection)
- Multi-scale emergence (atoms → neurons → societies → galaxies)

## Core Architecture

### Modular Layer System

**Layer 1: Core Engine** (`core/`)
- `neural/` - Spiking neural networks, STDP plasticity, dynamic rewiring
- `evolution/` - Genotypes, mutation operators, selection, population management
- `physics/` - Custom physics engine with Verlet integration, collision detection
- `energy/` - Metabolism, energy costs, food sources, per-creature budgets
- `resources/` - Food, drugs, energy zones, Poisson disk sampling
- `behavior/` - Resource-seeking, addiction, hunger, motivation systems

**Layer 2: Generators** (`generators/`)
- `markov/` - Evolutionary text generation with attract/repel word-pair scoring
- `audio/` - Neural activity → audio buffer synthesis
- `visual/` - Procedural pattern generation, retinal rendering
- `mutations/` - Text fuzzing, letter substitution/transposition

**Layer 3: Creatures** (`creatures/`)
- Combined body + brain + energy + genome + physics
- Species management by genetic similarity
- Birth, growth, reproduction, death lifecycle
- Resource-seeking behavior (food, drugs)
- Addiction, tolerance, withdrawal mechanics
- Physics-based movement with neural motor control

**Layer 4: Visualization** (`visualization/`)
- Rendering system with multiple modes (creatures, neurons, patterns, debug)
- Camera controls and UI overlays

### Critical Innovations from Source Material

**Telepathic Canvas (Circuit8)**
- 64x48 pixel shared screen ALL creatures can read/write
- 1024 depth layers per pixel (3D temporal buffer)
- Motor neurons control RGB channels: moreRed, lessRed, moreGreen, lessGreen, moreBlue, lessBlue
- Acts as collective unconscious/morphic field
- Creatures vote on screen movement/scrolling

**Psychopharmacology System**
- 5 molecule types affecting neural dynamics:
  - 0: Inhibitory Antagonist (blocks inhibitory neurons)
  - 1: Inhibitory Agonist (enhances inhibitory neurons)
  - 2: Excitatory Antagonist (blocks excitatory neurons)
  - 3: Excitatory Agonist (enhances excitatory neurons)
  - 4: Potentiator (amplifies all drug effects 10x)
- `tripping[5]` array per creature tracks drug levels
- Drug effects modify neural potential and plasticity

**Neural Network Details**
- Leaky integrate-and-fire model
- Bidirectional synapses (excitatory/inhibitory)
- Hebbian plasticity (Spike-Timing-Dependent Plasticity)
- Dynamic rewiring at runtime
- Sensory neurons: vision (retina), proprioception, energy, touch, morphic field
- Motor neurons: muscles, eating, procreation, screen writing

**Evolutionary Markov Text System**
- Word pairs have attract/repel scores that evolve
- When score > threshold: BREED (mutate line, wire into markov cloud)
- When score < 0.1: KILL (remove from markov table)
- Mutation operators: vowel/consonant substitution, letter increment/decrement, transposition
- Rewards novelty (pairs that haven't occurred recently)

## Technology Decisions

### Language: Python (Recommended for MVP)
- Rapid prototyping and experimentation
- NumPy for vectorized neural operations
- PyGame/SDL2 for graphics
- Path to optimization: Cython, numba, PyPy
- Future: Rewrite critical paths in C++/Rust if needed

### Graphics: SDL2
- Cross-platform, proven technology
- Used in source material (critterding, neural visualizers)
- SDL3 as alternative for modern API

### Physics: Bullet Physics or Custom
- Bullet Physics used in telepathic-critterdrug source
- Custom physics possible for simpler bodies

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Or install with development tools
pip install numpy pygame pytest pytest-cov black ruff mypy
```

### Running Examples
```bash
# Run telepathic collective intelligence simulation
# This demonstrates Circuit8, drugs, energy, and collective behavior
python3 examples/telepathic_simulation.py

# Run evolution demo (intelligence emerges from random networks)
python3 examples/evolution_demo.py

# Run simple neural network example
python3 examples/simple_network.py

# Run Circuit8 visualizer (pygame visualization of collective canvas)
python3 visualization/circuit8_visualizer.py

# Run audio synthesis demo (hear neural activity as sound)
python3 examples/audio_synthesis_demo.py

# Run evolutionary Markov text generation demo
python3 examples/markov_demo.py

# Run Markov demo in interactive mode
python3 examples/markov_demo.py -i

# Run visual pattern generation demo (requires pygame)
python3 examples/visual_patterns_demo.py

# Run multi-modal creature demo (Phase 4d - complete synesthetic beings)
python3 examples/multi_modal_demo.py

# Run Phase 5a demo (neural refinements from flamoot discoveries)
PYTHONPATH=/Users/gspilz/code/CritterGOD python3 examples/phase5a_demo.py

# Run Phase 5b demo (entity introspection and adam distance tracking)
PYTHONPATH=/Users/gspilz/code/CritterGOD python3 examples/phase5b_demo.py

# Run Phase 5c demo (visualization improvements - force-directed layouts)
PYTHONPATH=/Users/gspilz/code/CritterGOD python3 examples/phase5c_demo.py

# Run Phase 6 - COMPLETE EVOLUTIONARY ECOSYSTEM (requires pygame)
# The grand integration: all systems working together
PYTHONPATH=/Users/gspilz/code/CritterGOD python3 examples/evolutionary_ecosystem.py

# Run Phase 7/8 - 3D OpenGL visualization with UI
PYTHONPATH=/Users/gspilz/code/CritterGOD python3 examples/phase7_3d_integrated.py

# Run Phase 9a - Procedural 3D creature bodies (morphological evolution!)
PYTHONPATH=/Users/gspilz/code/CritterGOD python3 examples/phase9a_demo.py

# Run Phase 9b - Resource-seeking behavior (food/drug foraging with addiction!)
PYTHONPATH=/Users/gspilz/code/CritterGOD python3 examples/phase9b_demo.py

# Run Phase 9c - Collective learning with drug-modulated plasticity!
# Complete social artificial life with thoughts, reproduction, and psychedelic vision
PYTHONPATH=/Users/gspilz/code/CritterGOD python3 examples/phase9c_demo.py

# Run Phase 10a - Physics-based artificial life (realistic movement & collisions!)
PYTHONPATH=/Users/gspilz/code/CritterGOD python3 examples/phase10a_demo.py

# Run Phase 10 - RESEARCH PLATFORM with complete configuration UI!
# Runtime parameter tuning, real-time statistics, event console, profiles!
PYTHONPATH=/Users/gspilz/code/CritterGOD python3 examples/research_platform.py
```

### Testing
```bash
# Run full test suite
python3 -m pytest tests/

# Run specific test module
python3 -m pytest tests/test_neural.py

# Run with verbose output
python3 -m pytest tests/ -v

# Run with coverage
python3 -m pytest --cov=core --cov=creatures tests/

# Run single test function
python3 -m pytest tests/test_neural.py::TestNeuron::test_neuron_fires_when_threshold_exceeded
```

### Code Quality
```bash
# Format code (auto-fix)
black core/ creatures/ visualization/ examples/ tests/

# Check linting issues
ruff check .

# Type checking
mypy core/ creatures/

# Check specific file
ruff check core/neural/network.py
```

## Implementation Status

### ✅ Phase 1: Core Neural Engine (COMPLETE)
- `core/neural/neuron.py` - Leaky integrate-and-fire neurons
- `core/neural/synapse.py` - STDP plasticity, bidirectional connections
- `core/neural/network.py` - Network container and execution engine
- `tests/test_neural.py` - Comprehensive unit tests

### ✅ Phase 2: Evolution System (COMPLETE)
- `core/evolution/genotype.py` - Genetic encoding with 7 mutation types
- `core/evolution/phenotype.py` - Genotype→phenotype conversion
- `core/evolution/population.py` - Tournament selection, fitness evaluation
- `examples/evolution_demo.py` - Demonstrated intelligence emergence

### ✅ Phase 3: Revolutionary Features (COMPLETE)
- `core/morphic/circuit8.py` - 64x48 telepathic canvas, 1024 depth layers
- `core/morphic/morphic_field.py` - 6 morphic channels (RuRdGuGdBuBd)
- `core/pharmacology/drugs.py` - 5 molecule types, agonist/antagonist effects
- `core/energy/metabolism.py` - Complete energy system with starvation
- `creatures/creature.py` - Integrated organism (brain+body+energy+drugs)
- `visualization/circuit8_visualizer.py` - Real-time pygame renderer
- `examples/telepathic_simulation.py` - Multi-creature collective simulation

### ✅ Phase 4: Generative Systems (COMPLETE)
- ✅ **Audio Synthesis** - Neural activity → audio (3 modes: potential, firing, mixed)
  - Real-time audio synthesis from network state
  - Interactive pygame demo with visualization
  - Comprehensive test coverage (10 tests)
  - See `PHASE4_AUDIO_COMPLETE.md` for details
- ✅ **Markov Text Generation** - Evolutionary text with attract/repel dynamics
  - Self-organizing text through word pair scoring
  - 7 mutation operators (vowel/consonant substitution, transposition, etc.)
  - Breed/kill mechanics based on usage patterns
  - Interactive demo with statistics
  - See `PHASE4_MARKOV_COMPLETE.md` for details
- ✅ **Visual Pattern Generation** - Neural-driven generative art with retinal feedback
  - Procedural patterns from trigonometric functions
  - Heritage neuron index selection (brain[1591]/brain[1425])
  - Retinal sensor array (100 fingers × 50 neurons = 5000 sensory)
  - Complete feedback loop: network → patterns → sensors → network
  - See `PHASE4_VISUAL_COMPLETE.md` for details
- ✅ **Multi-Modal Creature Integration** - Complete synesthetic beings
  - Visual sensing through retinal arrays
  - Audio generation (voice from brain state)
  - Text generation (thoughts from markov chains)
  - Genetic language inheritance
  - Multi-modal demo with full update cycle
  - See `PHASE4_COMPLETE.md` for comprehensive Phase 4 documentation

### ✅ Phase 5: Neural Refinements (COMPLETE)
- ✅ **Phase 5a**: Continuous synapse weakening, weight clamping, bidirectional thresholds
- ✅ **Phase 5b**: Entity introspection (to_dict()), adam distance tracking
- ✅ **Phase 5c**: Force-directed layouts, neural network visualization
- See `PHASE5A_COMPLETE.md`, `PHASE5B_COMPLETE.md`, `PHASE5C_COMPLETE.md`

### ✅ Phase 7-8: 3D Visualization (COMPLETE)
- ✅ OpenGL 3D renderer with orbital camera
- ✅ Camera controls (mouse drag, zoom, pan)
- ✅ Circuit8 as textured ground plane
- ✅ 2D/3D view toggle
- ✅ UI overlay rendering over 3D scene
- `visualization/opengl_renderer.py` - Complete 3D rendering system
- `visualization/camera.py` - Orbital camera with spherical coordinates
- `visualization/gl_primitives.py` - OpenGL primitives and lighting
- `examples/phase7_3d_integrated.py` - Full 3D integration demo

### ✅ Phase 9a: Procedural 3D Bodies (COMPLETE)
- ✅ **Body Genotype System** - Genetic encoding of 3D morphology
  - Segments (2-10), limbs (0-4 per segment), head, tail
  - 14+ evolvable parameters per creature
  - Body mass calculation for energy costs
- ✅ **Procedural Mesh Generation** - Real-time 3D mesh from genes
  - Low-poly primitives (8 sides) for performance
  - Display list caching by genotype signature
  - Sphere, cylinder, cone generation
- ✅ **Morphological Evolution** - Visible body evolution
  - Mutations change body shape over generations
  - Family resemblance (offspring similar to parents)
  - Segment/limb addition and removal
- ✅ **Drug-Responsive Bodies** - Psychedelic visual effects
  - Body pulsing (size changes)
  - Color shifts (hue rotation)
  - Animation synchronized with trip intensity
- `core/morphology/body_genotype.py` - Body genetics (304 lines)
- `core/morphology/mesh_generator.py` - Mesh generation (410 lines)
- `creatures/morphological_creature.py` - Complete morphological being (267 lines)
- `examples/phase9a_demo.py` - Interactive morphological evolution demo
- See `PHASE9A_COMPLETE.md` for complete documentation

### ✅ Phase 9b-d: Complete Ecosystem (COMPLETE)
- ✅ **Phase 9b**: Food/resource system, resource-seeking behavior, addiction mechanics
- ✅ **Phase 9c**: Collective learning via Circuit8, drug-modulated plasticity
- ✅ **Phase 9d**: Complete sensory-motor integration, retinal vision, audio synthesis

### ✅ Phase 10: Research Platform (COMPLETE)
- ✅ **Configuration System**: 45+ tunable parameters across 7 categories
  - `core/config/parameters.py` - Parameter definitions (327 lines)
  - `core/config/config_manager.py` - Profile management (169 lines)
  - `profiles/default.json` - Default configuration profile
- ✅ **Statistics Tracking**: Real-time metrics with automatic history
  - `core/stats/statistics_tracker.py` - Time-series tracking (152 lines)
  - PopulationStats helper for creature metrics
- ✅ **Event Logging**: Critterdrug-style console output
  - `core/logging/event_logger.py` - Birth/death/reproduction logs (141 lines)
  - Formatted log lines with timesteps and IDs
- ✅ **Population Management**: Automatic culling mechanisms
  - `core/population_manager.py` - Kill_half, cull_oldest, cull_weakest (83 lines)
- ✅ **UI Widgets**: Professional pygame-based interface
  - `visualization/ui/slider_widget.py` - Interactive parameter sliders (246 lines)
  - `visualization/ui/graph_widget.py` - Real-time statistics graphs (270 lines)
  - `visualization/ui/console_widget.py` - Scrolling console output (206 lines)
  - `visualization/ui/config_panel.py` - Scrollable configuration panel (260 lines)
- ✅ **Research Platform**: Complete integration demo
  - `examples/research_platform.py` - Professional research platform (570 lines)
  - Runtime parameter tuning via sliders
  - Real-time statistics graphs (population, neural, FPS)
  - Event console with color-coded logs
  - Configuration profiles (save/load/quick-save)
  - Population control (K key to cull)
  - Auto-save every 1000 timesteps
  - Full 3D visualization with UI overlay

### 🔮 Future Enhancements
- GPU acceleration for 10k+ neuron networks
- Advanced physics (Bullet Physics integration)
- Social language evolution
- Cultural transmission across generations
- Additional configuration profiles (fast_evolution, drug_heavy, minimal)

## Critical Performance Considerations

**Neural Network Updates** (most CPU-intensive)
- Sparse neuron updates (only active neurons)
- Vectorization essential (NumPy arrays)
- Consider GPU acceleration for 10k+ neurons
- Batch operations where possible

**Physics Simulation**
- Spatial hashing for efficient queries
- Limit body complexity initially
- Profile before optimizing

**Rendering**
- Instanced rendering for creatures
- LOD system for distant objects
- Separate render thread if needed

## Configuration System

Profile-based configuration (from critterding heritage):
- All parameters exposed as config options
- Profiles stored in `profiles/` as text files
- Runtime modification support
- Key parameters:
  - Neural network size/topology
  - Mutation rates
  - Energy costs
  - Population size
  - Drug effects
  - Morphic field dimensions

## Philosophical Framework

The code embodies universal patterns that repeat at all scales:

**The Universal Cycle**: Collection → Combination → Heating → Radiation → Cooling → Equilibrium

Examples:
- **Atoms**: Compression → chain reactions → radioactive decay
- **Neurons**: Collect inputs → fire when threshold exceeded → synapses strengthen/weaken
- **Societies**: Cooperation → exponential knowledge growth → self-organization
- **Language**: Words explain each other → combinatorial explosion → compression (cooling)

Implementation connections:
- Neurons = atoms (collect→fire→radiate)
- Synaptic plasticity = heating/cooling dynamics
- Morphic fields = magnetic field loops
- Drug systems = energy injection/dampening
- Evolution = self-organization toward equilibrium
- Collective voting = society-level cooperation

## Source Material

CritterGOD synthesizes from:
- **telepathic-critterdrug**: Telepathy via Circuit8, drugs, voting (C++/Bullet Physics)
- **critterding-beta14**: Stable evolutionary ALife base
- **Neural visualizers** (looser.c, xesu.c, cdd.c, mitre.c, 03Singularity.ogg.599.c): 10k-65k neuron systems with real-time audio/visual
- **markov-attract-repel.py**: Genetic text generation (905 lines)
- **numerolit.py, series.py**: Numerical transformations
- **text-fuzz.py**: Mutation operators

See DISCOVERY.md for complete analysis of source material.

## Key Implementation Files

### Core Neural System
- `core/neural/neuron.py` - Neuron class with leaky integrate-and-fire dynamics
- `core/neural/synapse.py` - Synapse class with STDP plasticity
- `core/neural/network.py` - Network execution engine (update loop inspired by SDL visualizers)

### Evolution System
- `core/evolution/genotype.py` - Genetic encoding (NeuronGene, SynapseGene)
- `core/evolution/phenotype.py` - Convert genotype to phenotype (neural network)
- `core/evolution/population.py` - Population management, tournament selection

### Revolutionary Features
- `core/morphic/circuit8.py` - Telepathic canvas with depth buffer and voting
- `core/morphic/morphic_field.py` - MorphicChannel enum and influence calculation
- `core/pharmacology/drugs.py` - Drug system with 5 molecule types
- `core/energy/metabolism.py` - Energy costs, food consumption, starvation

### Creatures
- `creatures/creature.py` - Complete organism integrating all systems
- `creatures/creature_senses.py` - Visual sensing through retinal arrays
- `creatures/creature_motors.py` - Audio, text, visual generation integration
- `creatures/enhanced_creature.py` - Multi-modal creature with all generative capabilities
- `creatures/genetic_language.py` - Language breeding and genetic inheritance

### Generative Systems
- `generators/audio/neural_audio.py` - Audio synthesis from neural activity
- `examples/audio_synthesis_demo.py` - Interactive audio synthesis demo
- `generators/markov/markov_chain.py` - Basic Markov chain text generation
- `generators/markov/word_pair_score.py` - Evolutionary scoring system
- `generators/markov/mutations.py` - Text mutation operators
- `generators/markov/evolutionary_markov.py` - Self-organizing text system
- `examples/markov_demo.py` - Interactive Markov evolution demo
- `generators/visual/pattern_generators.py` - Trigonometric pattern generation
- `generators/visual/neural_parameters.py` - Extract params from neuron ratios
- `generators/visual/retinal_sensors.py` - Retinal sensor array (5000 neurons)
- `generators/visual/visual_pipeline.py` - Complete feedback loop
- `examples/visual_patterns_demo.py` - Neural art demo
- `examples/multi_modal_demo.py` - Complete synesthetic creatures (Phase 4d)

### Examples & Visualization
- `examples/simple_network.py` - Basic neural network demo
- `examples/evolution_demo.py` - Evolution of intelligent behavior
- `examples/telepathic_simulation.py` - Multi-creature collective intelligence
- `examples/audio_synthesis_demo.py` - Audio synthesis demonstration
- `examples/markov_demo.py` - Evolutionary text generation demo
- `examples/visual_patterns_demo.py` - Neural pattern generation demo
- `examples/multi_modal_demo.py` - Complete synesthetic creatures (Phase 4d)
- `visualization/circuit8_visualizer.py` - Pygame renderer for Circuit8
- `examples/visual_patterns_demo.py` - Neural pattern generation demo
- `examples/multi_modal_demo.py` - Complete synesthetic creatures (Phase 4d)
- `visualization/circuit8_visualizer.py` - Pygame renderer for Circuit8

### Documentation
- `README.md` - Project overview and current status
- `ARCHITECTURE.md` - Detailed system architecture and data flow
- `DISCOVERY.md` - Analysis of source codebase (~3000 files)
- `PHASE3_COMPLETE.md` - Phase 3 completion summary
- `PHASE4_AUDIO_COMPLETE.md` - Phase 4a audio synthesis completion
- `PHASE4_MARKOV_COMPLETE.md` - Phase 4b evolutionary text completion
- `PHASE4_VISUAL_COMPLETE.md` - Phase 4c visual pattern completion
- `PHASE4_COMPLETE.md` - Complete Phase 4 documentation

## Understanding the System Architecture

### Data Flow (Creature Update Cycle)
```
1. Read morphic field (Circuit8) → sensory input
2. Update neural network (neurons fire, synapses propagate)
3. Apply drug effects (modify neural potentials)
4. Extract motor outputs (movement, eating, screen writing)
5. Write to Circuit8 (collective canvas)
6. Vote on collective actions (screen movement)
7. Pay metabolic costs (energy consumption)
8. Check survival (starvation if energy < 0)
```

### Key Design Patterns

**Genotype→Phenotype Separation**
- `Genotype` (core/evolution/genotype.py) = genetic code
- `build_network_from_genotype()` = creates actual neural network
- Mutations operate on genotype, not running network
- See `core/evolution/phenotype.py` for conversion

**Morphic Field Channels**
- Each neuron assigned a channel (0-5): RuRdGuGdBuBd
- Channels read different aspects of RGB values from Circuit8
- Enables creatures to "tune in" to different collective frequencies
- See `MorphicChannel.get_influence()` in core/morphic/morphic_field.py

**Drug Effects on Neurons**
- 5 molecule types stored in `DrugSystem.tripping[5]` array
- Applied during neural update: agonists enhance, antagonists block
- Potentiator (molecule 4) amplifies all effects 10x
- See `DrugSystem.apply_drug_effect()` in core/pharmacology/drugs.py

**Energy-Based Fitness**
- Energy decreases every timestep (neuron costs, synapse costs, firing costs)
- Energy increases by eating food or consuming pills
- Creature dies when energy ≤ 0
- Fitness = energy level (survivors reproduce)

### Parameter Values from Heritage Codebase

**Neural Parameters** (from foodotrope-drug.profile):
- Neuron threshold: 700 + random(0, 8000)
- Inhibitory neuron probability: 0.14 (14%)
- Inhibitory synapse probability: 0.21 (21%)
- Synapses per neuron: 40 (from SDL visualizers)
- Plasticity enabled: 50% of neurons

**Energy Costs** (from core/energy/metabolism.py):
- Per neuron existence: 0.0001/timestep
- Per synapse existence: 0.00001/timestep  
- Per neuron firing: 0.1
- Per motor activity: scales with motor output magnitude

## Testing Strategy

1. **Unit tests**: Individual components (neurons, synapses, mutation operators)
2. **Integration tests**: Neural network + evolution, audio synthesis pipeline
3. **Evolution tests**: Long-run stability, convergence, diversity maintenance
4. **Performance benchmarks**: Neural updates, physics simulation, rendering
5. **Regression tests**: Known emergent behaviors preserved

### Running Specific Test Classes
```bash
# Test just neuron functionality
python3 -m pytest tests/test_neural.py::TestNeuron -v

# Test just synapse functionality  
python3 -m pytest tests/test_neural.py::TestSynapse -v

# Test network execution
python3 -m pytest tests/test_neural.py::TestNeuralNetwork -v

# Test audio synthesis system
python3 -m pytest tests/test_audio_synthesis.py::TestNeuralAudioSynthesizer -v
```

## Revolutionary Aspects to Preserve

1. **Psychedelic computing**: Drug effects on collective intelligence
2. **Morphic engineering**: Shared perception space (Circuit8)
3. **Democratic artificial life**: Emergent voting and collective will
4. **Multi-scale patterns**: Same rules from atoms to galaxies
5. **Evolutionary language**: Genetic algorithms on text itself
6. **Neural-driven aesthetics**: Audio/visual synthesis from brain activity

## Coding Conventions

### Documentation Style
- Every module starts with a docstring explaining its purpose and heritage source
- Reference source material (e.g., "Based on telepathic-critterdrug", "From looser.c")
- Classes have comprehensive docstrings with Attributes section
- Methods document Args and Returns

### Naming Conventions
- Classes: PascalCase (e.g., `NeuralNetwork`, `DrugSystem`)
- Functions/methods: snake_case (e.g., `update_metabolism`, `get_influence`)
- Constants: UPPER_SNAKE_CASE (e.g., `INHIBITORY_ANTAGONIST`)
- Private attributes: prefix with underscore (e.g., `_fired_this_step`)

### Type Hints
- Use type hints for function parameters and return values
- Import from `typing` module (List, Optional, Dict, Tuple)
- Use `float` for numeric values, `int` for indices/counts
- Example: `def update(self, dt: float = 1.0) -> bool:`

### Data Structures
- Use `@dataclass` for simple data containers (e.g., `Pill`, `NeuronGene`)
- Use `IntEnum` for numbered constants (e.g., `MoleculeType`, `ColorChannel`)
- Use `Enum` for symbolic constants (e.g., `NeuronType`, `MutationType`)
- NumPy arrays for vectorized operations (e.g., `tripping = np.zeros(5)`)

### Heritage Parameter Values
**Always preserve values from source codebase:**
- Neuron potential init: `rand() % 5000`
- Neuron threshold: `700 + rand() % 8000`
- Inhibitory neuron probability: 0.14 (from profile)
- Inhibitory synapse probability: 0.21 (from profile)
- Circuit8 dimensions: 64x48 pixels
- Depth buffer: 1024 layers
- Drug decay rate: 0.99 (1% per timestep)
- Potentiator amplification: 10x

### File Organization
- Each module has `__init__.py` that exports key classes
- Keep files focused: one main class per file
- Related functionality grouped in directories (e.g., `core/neural/`, `core/morphic/`)
- Examples in `examples/`, tests in `tests/`
- Documentation at project root

### Error Handling
- Use max/min clamping for array indices (e.g., `x = max(0, min(self.width - 1, x))`)
- Validate data in `__post_init__` for dataclasses
- Check for edge cases (empty lists, zero division, etc.)
- Let NumPy handle vectorized operations efficiently

## Common Development Patterns

### Creating a New Creature
```python
from core.evolution.genotype import Genotype
from core.morphic.circuit8 import Circuit8
from creatures.creature import Creature

# Create shared telepathic canvas
circuit8 = Circuit8(width=64, height=48)

# Create random genotype
genotype = Genotype.create_random(
    n_sensory=10,
    n_motor=10,
    n_hidden_min=50,
    n_hidden_max=200,
    synapses_per_neuron=40
)

# Create creature
creature = Creature(
    genotype=genotype,
    x=0.0, y=0.0,
    initial_energy=1000000.0,
    circuit8=circuit8
)

# Update creature each timestep
alive = creature.update(dt=1.0)
```

### Mutating and Evolving
```python
from core.evolution.genotype import Genotype
from core.evolution.population import Population

# Create population
population = Population(size=100)

# Evaluate fitness (run creatures, measure energy)
for creature in population.creatures:
    # ... run simulation ...
    creature.fitness = creature.energy.energy

# Select survivors and create offspring
population.tournament_selection()
offspring = population.generate_offspring()

# Mutate offspring
for genotype in offspring:
    mutated = genotype.mutate(mutation_rate=0.5)
```

### Visualizing Circuit8
```python
import pygame
from core.morphic.circuit8 import Circuit8

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((640, 480))

# Create circuit8
circuit8 = Circuit8(width=64, height=48)

# In render loop:
for y in range(circuit8.height):
    for x in range(circuit8.width):
        r, g, b = circuit8.read_pixel(x, y)
        pygame.draw.rect(
            screen,
            (r, g, b),
            (x * 10, y * 10, 10, 10)
        )
```

## Debugging Tips

**Neural Network Not Learning?**
- Check neuron firing rates with `network.get_activity_level()`
- Ensure thresholds aren't too high (creatures born with random thresholds)
- Verify STDP is enabled: `network.enable_plasticity = True`
- Check energy levels - dead creatures can't learn

**Circuit8 Not Showing Activity?**
- Verify creatures have screen motor neurons (last 6 neurons)
- Check `creature.screen_motors` array for non-zero values
- Ensure creatures are writing: `circuit8.write_pixel()` being called
- Try blending mode vs direct write

**Evolution Not Improving?**
- Verify fitness is being calculated (energy-based)
- Check selection pressure (tournament size in population.py)
- Ensure mutation rate isn't too high (0.5 = 50% is typical)
- Run for more generations (intelligence emerges slowly)

**Performance Issues?**
- Profile with: `python3 -m cProfile -o profile.stats examples/telepathic_simulation.py`
- Neural updates are O(neurons * synapses) - most expensive
- Consider reducing synapses_per_neuron for testing
- NumPy operations already vectorized where possible

**Audio Synthesis Issues?**
- Check pygame audio initialization: `pygame.mixer.init(frequency=44100)`
- Verify buffer size matches: synthesizer buffer_size = pygame mixer buffer
- Test synthesis modes: 'potential', 'firing', 'mixed'
- Monitor phase continuity: `synth._phase` should stay in [0, 1) range

## Notes for Future Implementation

- Start with simplest working version
- Validate core mechanics before adding complexity
- Profile performance early and often
- Keep modularity - each system should work standalone
- Document emergence when it occurs
- Preserve experimental/playful spirit of source material
- Don't optimize prematurely - measure first
- When adding features, reference DISCOVERY.md for source material details
