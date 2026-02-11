# CritterGOD Architecture

## System Overview

CritterGOD is designed as a modular, layered architecture that separates concerns while allowing tight integration where performance is critical.

## Core Layers

### Layer 1: Core Engine (`core/`)

#### Neural System (`core/neural/`)
- **Neuron**: Base neuron class with firing threshold, potential, plasticity
- **Synapse**: Connection between neurons (excitatory/inhibitory, weight, plasticity)
- **Network**: Neural network container and execution engine
- **Sensors**: Input neurons (vision, proprioception, energy, contact)
- **Motors**: Output neurons (muscle control, audio, visual generation)

Key algorithms:
- Leaky integrate-and-fire neuron model
- Hebbian plasticity (STDP - Spike-Timing-Dependent Plasticity)
- Bidirectional synapse propagation
- Dynamic rewiring

#### Evolution System (`core/evolution/`)
- **Genotype**: Genetic encoding of creature (body + brain structure)
- **Mutation**: Operators for genetic variation
- **Selection**: Fitness evaluation and parent selection
- **Crossover**: Sexual reproduction (optional)
- **Population**: Population management and statistics

Key algorithms:
- Mutation operators (add/remove/modify nodes, edges, parameters)
- Tournament selection
- Genetic distance metrics
- Speciation (optional)

#### Physics System (`core/physics/`)
- **Body**: Physical structure (parts, joints, constraints)
- **World**: Physics simulation environment
- **Collision**: Detection and response

Key algorithms:
- Rigid body dynamics (if using physics engine)
- Constraint solving for joints
- Efficient collision detection

#### Energy System (`core/energy/`)
- **Metabolism**: Energy costs (neurons, synapses, body parts, movement)
- **Food**: Energy sources in environment
- **Budget**: Per-creature energy accounting

### Layer 2: Generators (`generators/`)

#### Markov System (`generators/markov/`)
- Markov chain text generation
- Evolutionary word-pair selection
- Attract/repel dynamics for word scoring

#### Audio System (`generators/audio/`)
- Neural activity → audio buffer conversion
- Procedural audio synthesis
- WAV manipulation

#### Visual System (`generators/visual/`)
- Procedural pattern generation (trigonometric functions)
- Retinal input rendering
- Neuron activity visualization

#### Mutation System (`generators/mutations/`)
- Text fuzzing algorithms
- Letter substitution, transposition
- Word-level mutations

### Layer 3: Creatures (`creatures/`)

- **Creature**: Combined body + brain + energy + genome
- **Species**: Population management by genetic similarity
- **Lifecycle**: Birth, growth, reproduction, death

### Layer 4: Visualization (`visualization/`)

- **Renderer**: Main rendering system
- **Camera**: View controls
- **UI**: Overlays, stats, controls
- **Modes**: Different visualization modes (creatures, neurons, patterns, debug)

## Data Flow

```
Environment Sensors
        ↓
   Neural Network
   (forward pass)
        ↓
   Motor Outputs
        ↓
   Physical Actions
        ↓
   Energy Changes
        ↓
Selection/Reproduction
        ↓
   New Genotypes
   (with mutations)
```

## Configuration System

Profile-based configuration (from critterding heritage):
- All parameters exposed as config options
- Profiles stored as text files
- Runtime modification support
- Profile saving/loading

## Execution Model

### Option 1: Frame-based (Real-time)
```python
while running:
    handle_input()
    update_physics(dt)
    update_creatures(dt)
    update_neural_networks()
    update_energy()
    check_reproduction()
    render_frame()
```

### Option 2: Event-based (Fast evolution)
```python
while running:
    for creature in population:
        run_simulation_episode(creature)
        evaluate_fitness(creature)
    select_survivors()
    generate_offspring()
```

### Option 3: Hybrid
Real-time visualization with accelerated neural/physics updates

## Performance Considerations

### Critical Paths
1. **Neural network updates**: Most CPU-intensive
   - Consider vectorization (NumPy, SIMD)
   - GPU compute shaders for massive parallelism
   
2. **Physics simulation**: Second most intensive
   - Spatial partitioning for collision detection
   - Consider existing engines (Bullet, custom)

3. **Rendering**: GPU-bound
   - Instanced rendering for creatures
   - LOD system for distant objects

### Optimization Strategies
- **Sparse neuron updates**: Only update active neurons
- **Spatial hashing**: Efficient creature/food queries
- **Batch operations**: Group similar operations
- **Parallel evolution**: Independent creatures can evolve in parallel
- **Caching**: Genotype → Phenotype conversion

## Modularity & Extensibility

### Plugin System (Future)
- Custom neuron types
- Custom mutation operators
- Custom sensors/motors
- Custom fitness functions

### API Boundaries
Clear interfaces between:
- Neural network ↔ Physics
- Evolution ↔ Neural network
- Generators ↔ Core systems

## Testing Strategy

1. **Unit tests**: Individual components
2. **Integration tests**: System interactions
3. **Evolution tests**: Long-run stability
4. **Performance benchmarks**: Critical paths
5. **Regression tests**: Known behaviors preserved

## Technology Choices

### Language Options

**Python (Recommended for MVP)**
- Rapid prototyping
- Rich ecosystem (NumPy, PyGame, etc.)
- Easy experimentation
- Path to optimization (Cython, numba, PyPy)

**C++**
- Maximum performance
- Direct SDL/OpenGL access
- More complex but proven (critterding heritage)

**Rust**
- Safety + Performance
- Modern tooling
- Steeper learning curve

### Graphics Options
- **SDL2**: Simple, proven, cross-platform
- **SDL3**: Modern, improved API
- **PyGame/Arcade**: Python-friendly
- **OpenGL/Vulkan**: Maximum control

## Migration Path

1. **Phase 1**: Python prototype with core concepts
2. **Phase 2**: Optimize critical paths (Cython/numba)
3. **Phase 3**: Rewrite performance-critical modules in C++/Rust (optional)
4. **Phase 4**: GPU acceleration for neural networks

## References

- Critterding documentation and source
- Original SDL neural visualizers
- Spiking neural network literature
- Evolutionary algorithm textbooks
