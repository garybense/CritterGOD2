# CritterGOD4 Analysis

**Discovery Date**: January 24, 2026  
**Location**: `/Users/gspilz/code/critters/crittergod-code-r10/crittergod4`  
**Version**: CritterGOD 0.01 (circa 2010)  
**Author**: crittergod.sourceforge.net  
**License**: GPL v3

## Overview

This is an **early prototype** of CritterGOD from ~2010, written in C++ with Bullet Physics integration. It represents the foundational architecture before the project evolved into telepathic-critterdrug and the later Python/SDL experiments.

This is a **critical discovery** as it shows the original vision and implementation patterns that informed all subsequent work.

## Architecture

### Core Components

1. **Neural System** (`src/neural/`)
   - `Brain.h/cpp` - Main neural network container
   - `Neuron.h/cpp` - Spiking neuron implementation
   - `BrainBuilder.h` - Genetic construction system

2. **Bio System** (`src/bio/`)
   - `BloodBrainInterface.h/cpp` - Nutrient metabolism (early drug system prototype)

3. **Physics** (`src/physics/`)
   - Complete Bullet Physics integration
   - BulletCollision, BulletDynamics, BulletSoftBody
   - Custom creature bodies (SnakeBody, SpiderBody, BoxBody)

4. **Visualization** (`src/video/`, `src/objects/`)
   - `GLWindow` - OpenGL rendering
   - `BrainVis` - Neural network visualizer
   - `Spacetime` - 3D space management

5. **Mathematics** (`src/math/`)
   - Vector2/3 classes
   - Custom math utilities

6. **Graph** (`src/graph/`)
   - Graph data structure (possibly for neural topology)

## Neural Architecture (Brain.h)

### Key Parameters (Heritage Values)

```cpp
Brain(48 inputs, 32 outputs, 16384 neurons, 1-8 synapses/neuron, 0.4 inhibitory)
```

**Critical Parameters**:
- **16,384 neurons** - Large-scale network (matches SDL visualizer heritage)
- **percentChanceInhibitoryNeuron**: 0.4 (40% inhibitory)
- **percentChanceInhibitorySynapses**: 0.4 (40% inhibitory synapses)
- **percentChanceConsistentSynapses**: 0.50 (50% neurons have uniform synapse types)
- **percentChanceOutputNeuron**: 0.25 (25% chance to connect to motor)
- **percentChancePlasticNeuron**: 0.50 (50% have STDP)
- **percentChanceInputSynapse**: 0.25 (25% connect to sensory)

**Plasticity Parameters**:
```cpp
minPlasticityStrengthen = 1.01  (1% strengthen)
maxPlasticityStrengthen = 1.15  (15% strengthen)
minPlasticityWeaken = 0.85      (15% weaken)
maxPlasticityWeaken = 0.99      (1% weaken)
```

**Firing Thresholds**:
```cpp
minFiringThreshold = 0.05
maxFiringThreshold = 0.95
```

### Neuron Model (Neuron.h)

**Binary Spiking Neurons**:
- **Output**: -1 (inhibitory fire), 0 (silent), +1 (excitatory fire)
- **Potential decay**: Configurable per-neuron
- **Synaptic weight**: -1.0 to +1.0 range
- **Max synaptic weight magnitude**: 5.0 (clamping)

**Key Differences from Our Implementation**:
1. **Binary spikes** (-1/0/+1) vs. our continuous potential
2. **Weight clamping** to ±5.0 after plasticity
3. **Consistent synapses** - some neurons have all excitatory or all inhibitory
4. **Potential normalization** - values stay in [-1, 1] range

**Forward Pass Algorithm**:
```cpp
1. Decay potential (potential *= potentialDecay)
2. Weaken all synapses (weight *= plasticityWeaken)  
3. Accumulate synaptic input (potential += weight * input)
4. Check firing threshold (different for excitatory vs inhibitory)
5. If fired: Reset potential to 0, output ±1, strengthen active synapses
6. If not fired: output 0
```

**STDP Implementation**:
- Happens **during firing** (not post-hoc)
- Weakening is **continuous** (every timestep)
- Strengthening is **conditional** (only when both neuron and synapse fire)
- **Asymmetric**: Excitatory and inhibitory have different firing checks

## Blood-Brain Interface (bio/BloodBrainInterface.h)

```cpp
/** how nutrients get absorbed by the brain, metabolized, and influence the neural activity */
class BloodBrainInterface {
```

This is the **proto-drug system**! Comment explicitly mentions:
- Nutrient absorption
- Metabolism
- Influence on neural activity

This became the **psychopharmacology system** in later versions (5 molecule types, agonist/antagonist effects).

## Bullet Physics Bodies

**Main file**: `crittergod4.cpp` shows creature types:

```cpp
SnakeBody* snake1 = new SnakeBody(btVector3(0,0,0), 8);  // 8 segments
SpiderBody* spider = new SpiderBody(6, btVector3(0,0,5));  // 6 legs
BoxBody* box = new BoxBody(btVector3(3,3,3), btVector3(1,0.5, 0.5));
```

Bodies are built with Bullet Physics constraints (joints, hinges, etc.)

## Brain Visualization (BrainVis)

Real-time 3D visualization of neural activity:
- Renders neurons and synapses
- Shows firing patterns
- OpenGL-based

## Comparison with Current CritterGOD Implementation

### What We've Preserved

✅ **Spiking neurons** with potential accumulation  
✅ **STDP plasticity** (strengthen when fire together)  
✅ **Inhibitory neurons** (negative weights)  
✅ **Configurable parameters** (thresholds, plasticity rates)  
✅ **Sensory/motor separation** (InNeuron, OutNeuron, Neuron)  
✅ **Random network generation** with genetic encoding

### Key Differences

**CritterGOD4 (2010)**:
- Binary spikes (-1, 0, +1)
- Continuous weakening every timestep
- Weight clamping to ±5.0
- "Consistent synapses" (all excitatory or all inhibitory per neuron)
- C++/Bullet Physics (3D bodies, rigid body dynamics)
- 16,384 neurons (large scale)
- Blood-brain interface (proto-drug system)

**Current CritterGOD (2026 Python)**:
- Continuous potential values
- STDP based on pre/post spike timing
- Weight initialized randomly, no strict clamping
- Mixed synapse types per neuron
- Python/NumPy (2D physics, energy metabolism)
- Configurable neuron count (typically 100-300)
- Full psychopharmacology (5 molecule types)
- Circuit8 telepathic canvas (64x48 shared screen)
- Evolutionary text generation (Markov chains)
- Audio/visual synthesis from neural activity

### What CritterGOD4 Teaches Us

**1. Binary Spikes Are Powerful**
   - Simplifies computation (-1/0/+1 vs continuous)
   - Clear interpretation (fired or not)
   - Easier to visualize

**2. Continuous Weakening Creates Pruning**
   - Every synapse weakens every timestep
   - Only strengthens when both fire
   - This creates automatic pruning of unused connections
   - **We should consider adding this!**

**3. Weight Clamping Prevents Runaway**
   - Max weight magnitude of 5.0
   - Prevents unbounded growth
   - Our system could benefit from this

**4. Consistent Synapses Create Specialization**
   - 50% of neurons have uniform synapse types
   - Creates clearer functional roles
   - Could add as optional parameter

**5. High Inhibitory Percentage**
   - 40% inhibitory neurons (vs our 14%)
   - Matches biological cortex (~20-30%)
   - **We should experiment with higher values**

**6. Blood-Brain Interface Was Always Planned**
   - Nutrient metabolism affecting neural activity
   - This evolved into our drug system
   - Shows continuity of vision

## Critical Parameters to Consider Adopting

### 1. Continuous Synapse Weakening
```python
# In our Synapse.update() or Neuron.update():
self.weight *= 0.99  # 1% decay per timestep
```

**Benefit**: Automatic pruning, prevents stale connections

### 2. Weight Clamping
```python
MAX_WEIGHT = 5.0
self.weight = np.clip(self.weight, -MAX_WEIGHT, MAX_WEIGHT)
```

**Benefit**: Prevents runaway strengthening, stable long-term dynamics

### 3. Consistent Synapses Option
```python
if consistent_synapses and random.random() < 0.5:
    all_synapses_inhibitory = random.random() < 0.4
    # All synapses for this neuron share sign
```

**Benefit**: Creates functional specialization, clearer roles

### 4. Higher Inhibitory Ratio
```python
INHIBITORY_NEURON_PROB = 0.3  # 30% instead of 14%
INHIBITORY_SYNAPSE_PROB = 0.3
```

**Benefit**: More stable dynamics, better matches biology

### 5. Binary Spike Mode
```python
# Optional mode in addition to continuous:
if self.potential >= self.threshold:
    self.output = 1.0
    self.potential = 0.0
else:
    self.output = 0.0
```

**Benefit**: Simpler, faster, easier to interpret

## Integration Recommendations

### Phase 5a: Neural Enhancements from CritterGOD4

**High Priority** (should integrate):
1. ✅ Continuous synapse weakening (automatic pruning)
2. ✅ Weight clamping (stability)
3. ✅ Configurable inhibitory ratio (experiment with 20-40%)

**Medium Priority** (optional enhancements):
4. ⚠️ Consistent synapses flag (functional specialization)
5. ⚠️ Binary spike mode (performance option)

**Low Priority** (interesting but not critical):
6. 📊 Large-scale networks (10k+ neurons with GPU)
7. 📊 3D Bullet Physics bodies (future Phase 6?)

### Recommended Changes to Current Code

**1. Add to `core/neural/synapse.py`**:
```python
class Synapse:
    def __init__(self, ..., decay_rate=0.99, max_weight=5.0):
        self.decay_rate = decay_rate
        self.max_weight = max_weight
    
    def update_plasticity(self, ...):
        # Continuous weakening
        self.weight *= self.decay_rate
        
        # STDP strengthening (when appropriate)
        if should_strengthen:
            self.weight *= strengthen_factor
        
        # Clamp to range
        self.weight = np.clip(self.weight, -self.max_weight, self.max_weight)
```

**2. Add to `core/evolution/genotype.py`**:
```python
@dataclass
class NeuronGene:
    has_consistent_synapses: bool = False
    consistent_inhibitory: bool = False  # Only used if has_consistent_synapses
```

**3. Update parameter defaults**:
```python
# In Genotype.create_random():
inhibitory_neuron_prob = 0.30  # Up from 0.14
inhibitory_synapse_prob = 0.30  # Up from 0.21
```

## Files of Interest for Deeper Study

**Priority 1** (immediate value):
1. `src/neural/Neuron.cpp` - Complete neuron implementation
2. `src/neural/Brain.cpp` - Network construction and execution
3. `src/bio/BloodBrainInterface.*` - Proto-drug system

**Priority 2** (creature bodies):
4. `src/space/CritterBody.h` - Body construction (if exists)
5. `src/space/SnakeBody.h` - Snake creature (if exists)
6. `src/space/SpiderBody.h` - Spider creature (if exists)

**Priority 3** (visualization):
7. `src/objects/BrainVis.*` - Neural visualization
8. `src/video/Spacetime.*` - 3D space management

## Historical Context

**Timeline**:
- **2010**: CritterGOD4 (this discovery) - C++/Bullet Physics, 16k neurons
- **2010-2015**: Evolution to critterding-beta14 (stable base)
- **2015-2020**: telepathic-critterdrug (Circuit8, drugs, voting)
- **2020-2025**: SDL visualizers (audio synthesis, 65k neurons)
- **2025**: markov-attract-repel.py (evolutionary text)
- **2026**: Current CritterGOD Python implementation (Phase 4 complete)

**Key Insight**: CritterGOD4 shows the original vision was **always** about:
1. Large-scale spiking neural networks
2. Metabolism affecting neural activity (blood-brain interface → drugs)
3. 3D physical bodies with realistic dynamics
4. Real-time visualization of neural activity
5. Evolutionary construction

Our current implementation has **faithfully preserved this vision** while adding:
- Morphic fields (Circuit8)
- Psychopharmacology (5 molecule types)
- Generative systems (audio, text, visual)
- Genetic language evolution

## Conclusion

CritterGOD4 is the **Rosetta Stone** of the project - it shows the foundational architecture and parameter values that informed all subsequent development.

**Key Takeaways**:
1. **Binary spikes** were the original model (worth revisiting)
2. **Continuous weakening** creates automatic pruning (should add)
3. **40% inhibitory** was the original ratio (we use 14%, should experiment)
4. **Blood-brain interface** was always planned (now our drug system)
5. **16k neurons** was the target scale (we're at 100-300)
6. **Weight clamping** was built-in (we should add)

**Recommended Next Steps**:
1. Add continuous synapse weakening to Phase 5
2. Implement weight clamping
3. Experiment with higher inhibitory ratios (20-40%)
4. Consider binary spike mode as performance option
5. Study complete Neuron.cpp implementation
6. Extract creature body construction patterns

This discovery validates our entire approach and provides concrete parameters and patterns to enhance our implementation.

---

**Status**: Analysis complete  
**Recommendation**: Integrate CritterGOD4 enhancements in Phase 5a (Neural Refinements)
