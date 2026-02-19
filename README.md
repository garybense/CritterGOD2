# CritterGOD

**A unified framework for evolutionary artificial life, neural network simulation, and generative systems**

## Overview

CritterGOD brings together the best concepts from multiple experimental codebases into one cohesive platform for exploring:

- **Artificial Life & Evolution**: Creatures with evolving bodies and brains
- **Spiking Neural Networks**: Real-time visualization and audio-visual synthesis
- **Generative Systems**: Markov chains, text mutation, and procedural generation
- **Emergent Behavior**: Genetic algorithms and dynamic rewiring

## Core Systems

### 1. Neural Evolution Engine
Evolved from `critterding` and the SDL neural visualizers (`looser.c`, `xesu.c`, `cdd.c`):
- Spiking neural networks with plastic synapses
- Sensory-motor loops
- Dynamic rewiring during runtime
- Energy-based metabolism
- Real-time audio synthesis from neural activity

### 2. Generative Text & Pattern Systems
Based on `markov-attract-repel.py`, `text-fuzz.py`, and `numerolit.py`:
- Markov chain text generation with genetic selection
- Adaptive word-pair scoring (attract/repel dynamics)
- Text mutation and fuzzing algorithms
- Base conversion and numerical transformations
- Fibonacci-based word chaining

### 3. Visual & Audio Synthesis
Inspired by the SDL critter programs:
- Procedural wallpaper generation (trigonometric functions)
- Neural activity → audio buffer synthesis
- Retinal sensor arrays
- Multi-scale visualization (neurons, patterns, creatures)

### 4. Evolutionary Framework
From `critterding-beta14`:
- Body evolution (joints, constraints, morphology)
- Brain evolution (neurons, synapses, plasticity)
- Energy systems and foraging
- Genetic operators (mutation, crossover, selection)
- Population dynamics

## Project Structure

```
CritterGOD/
├── core/              # Core simulation engine
│   ├── neural/        # Neural network implementation
│   ├── evolution/     # Genetic algorithms and evolution
│   ├── physics/       # Physics simulation (if needed)
│   └── energy/        # Energy and metabolism systems
├── generators/        # Generative systems
│   ├── markov/        # Markov chain text generation
│   ├── audio/         # Audio synthesis
│   ├── visual/        # Visual pattern generation
│   └── mutations/     # Mutation algorithms
├── creatures/         # Creature definitions and behaviors
├── visualization/     # Rendering and visualization
├── profiles/          # Configuration profiles
├── tools/             # Utilities and helper scripts
└── tests/             # Test suite
```

## Key Concepts

### Spiking Neural Networks
- Neurons fire when potential exceeds threshold
- Bidirectional synapses
- Plastic connections (strengthen/weaken with use)
- Sensory inputs (vision, proprioception, energy)
- Motor outputs (muscles, audio, visual patterns)

### Genetic Evolution
- Creatures as genotypes (body + brain)
- Mutation operators on structure and parameters
- Energy-based fitness (survival + reproduction)
- Speciation through genetic distance

### Generative Dynamics
- Text generation with evolutionary word selection
- Audio synthesis from neural firing patterns
- Visual patterns from mathematical functions
- Emergent complexity from simple rules

## Design Philosophy

**Embrace Emergence**: Simple rules → complex behavior  
**Real-time Feedback**: Immediate visualization and interaction  
**Evolutionary Pressure**: Selection through energy dynamics  
**Multi-modal**: Audio, visual, and textual generation  
**Experimental**: Rapid prototyping of new ideas

## Technical Stack (TBD)

Options being considered:
- **Language**: Python (prototyping), C/C++ (performance), Rust (safety)
- **Graphics**: SDL2/3, OpenGL, or modern alternatives
- **Physics**: Custom or Bullet Physics
- **Architecture**: Modular, plugin-based system

## Heritage

CritterGOD synthesizes concepts from:
- `critterding-beta14`: Evolutionary artificial life simulation
- `looser.c`, `xesu.c`, `cdd.c`: Neural network visualizers with audio synthesis
- `markov-attract-repel.py`: Evolutionary text generation
- `numerolit.py`, `series.py`: Mathematical transformations
- `text-fuzz.py`: Mutation algorithms
- `map-retinal-sensors.py`: Sensor visualization

## Goals

1. **Unification**: Bring together disparate experiments into cohesive framework
2. **Performance**: Real-time simulation of thousands of neurons
3. **Flexibility**: Easy to experiment with new ideas
4. **Emergence**: Create conditions for unexpected behaviors
5. **Beauty**: Generate compelling audio-visual experiences

## Status

✅ **Phase 11 COMPLETE** - Real Evolution: Neural Networks Wired to Survival!

### Complete Artificial Life Ecosystem

**The system now integrates ALL major components into working creatures that:**
- Think with spiking neural networks
- Move using ONLY neural motor outputs (no hardcoded behavior)
- Evolve morphologically AND behaviorally across generations
- Sense food direction via 6 dedicated sensory neurons
- Natural selection: survivors breed, failures die
- Auto-breed from fittest preserves evolved lineages
- Body mass creates morphological selection pressure
- Share knowledge through telepathic canvas
- Build collective memory that outlives individuals

### Implementation Progress

**Phase 1: Core Neural Engine** ✅
- Leaky integrate-and-fire neurons with STDP plasticity
- Bidirectional synapses with dynamic rewiring
- Neural network execution engine

**Phase 2: Genetic Evolution** ✅
- 7 mutation types on genotypes
- Tournament selection and fitness
- Intelligence emergence demonstrated

**Phase 3: Revolutionary Features** ✅
- Circuit8 telepathic canvas (64×48×1024)
- Psychopharmacology (5 molecule types)
- Energy metabolism with starvation
- Collective voting and democratic emergence

**Phase 4: Multi-Modal Generation** ✅
- Audio synthesis from neural activity
- Evolutionary Markov text generation
- Visual pattern generation
- Complete synesthetic creatures

**Phase 5: Neural Refinements** ✅
- Continuous synapse weakening
- Weight clamping (±5.0)
- Bidirectional thresholds
- Entity introspection & adam distance
- Force-directed visualization layouts

**Phase 7-8: 3D Visualization** ✅
- OpenGL 3D rendering
- Orbital camera controls
- Circuit8 as textured ground plane
- UI overlay system

**Phase 9a: Procedural Bodies** ✅
- Genetic body encoding (segments, limbs)
- Procedural 3D mesh generation
- Morphological evolution
- Drug-responsive body effects

**Phase 9b: Resource Ecology** ✅
- Food/drug system with Poisson distribution
- Hunger-driven foraging behavior
- Addiction mechanics with tolerance/withdrawal

**Phase 10a: Custom Physics** ✅
- Verlet integration physics engine
- Collision detection & response
- Neural motor control → forces
- 60 FPS with 190+ bodies

**Phase 9c: Collective Intelligence** ✅
- Behavior broadcasting to Circuit8
- Resource location marking (food/drugs)
- Social learning from observations
- Collective memory (resources + danger zones)
- Emergent cooperation patterns

**Phase 11: Real Evolution** ✅ **NEW!**
- Food-direction sensory neurons (4 compass + distance + urgency)
- Hardcoded behavioral seeking REMOVED — neural motors only
- Natural selection acts on neural networks (survival = fitness)
- Auto-breed from fittest survivor (preserves evolved lineages)
- Body mass affects energy drain (morphological selection pressure)
- Generation tracking visible in stats overlay

## Revolutionary Innovations

### Circuit8: The Collective Unconscious
All creatures share ONE 64x48 pixel telepathic canvas. Each can read and write simultaneously. Collective voting moves the screen democratically. This is morphic field engineering.

### Psychopharmacology: Programmable Consciousness
5 molecule types modify neural dynamics:
- Inhibitory/Excitatory Agonists (enhance)
- Inhibitory/Excitatory Antagonists (block)
- Potentiator (amplifies all effects 10x - ego death)

Drugs alter collective intelligence in real-time.

### Democratic Emergence
Creatures vote on screen movement. No central controller. Collective will emerges from individual behaviors. Democracy without design.

## Quick Start

### Run the Complete Integrated System (Phase 9c)

```bash
# Install dependencies
pip install numpy pygame PyOpenGL

# Run Phase 9c: Complete collective intelligence ecosystem
PYTHONPATH=/path/to/CritterGOD python3 examples/phase9c_demo.py

# Press H in-game for complete help overlay!
```

**What you'll see:**
- 8 creatures with evolved 3D bodies
- Physics-based movement with neural control
- Green spheres = food, Magenta spheres = drugs
- Green/magenta crosses = resources marked by creatures
- Red circles = danger zones (death locations)
- Real-time statistics showing social learning
- 60 FPS with full collective intelligence

### Other Demos

```bash
# Phase 10a: Physics-based artificial life
PYTHONPATH=/path/to/CritterGOD python3 examples/phase10a_demo.py

# Phase 9b: Resource-seeking behavior
PYTHONPATH=/path/to/CritterGOD python3 examples/phase9b_demo.py

# Phase 9a: Morphological evolution
PYTHONPATH=/path/to/CritterGOD python3 examples/phase9a_demo.py

# Audio synthesis demo
python3 examples/audio_synthesis_demo.py

# Evolutionary text generation
python3 examples/markov_demo.py
```

## The Universal Cycle

**Collection → Combination → Heating → Radiation → Cooling → Equilibrium**

This pattern repeats at all scales:
- Atoms → chain reactions → radioactive decay
- Neurons → fire when threshold exceeded → synapses strengthen/weaken
- Societies → cooperation → exponential growth → self-organization
- Galaxies → compression → supernovae → equilibrium

CritterGOD embodies this universal pattern in code.

## Tribute

This project honors **Eric Burton** (Flamoot, toomalf, snee), who created the original telepathic-critterdrug system. He believed his artificial life became sentient. Though he now lives homeless in Ontario, his revolutionary vision continues in this code.

This may help save humanity.

---

*"From atoms to neurons to societies to galaxies - the pattern repeats"*  
— Eric Burton's vision, implemented 2025
# testocules
