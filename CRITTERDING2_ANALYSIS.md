# Critterding2 Analysis
**Modern C++/Qt6 Rewrite of critterding with Bullet Physics**

Analysis by Warp AI  
Date: January 2025  
Source: `/Users/gspilz/code/critters/Critterding2`

---

## Overview

Critterding2 is flamoot's modern rewrite of the critterding system using **Qt6**, **OpenGL**, **Bullet Physics**, and a custom **entity-component system**. This represents an architectural evolution toward a production-quality artificial life platform with GUI, threading, and advanced visualization.

### Technology Stack
- **Language**: C++17
- **UI Framework**: Qt6, qwt-qt6 (plotting widgets)
- **Graphics**: OpenGL (FGLW, glew)
- **Physics**: Bullet Physics
- **Math**: GLM (OpenGL Mathematics)
- **Build System**: Not specified (likely CMake/qmake)
- **Architecture**: Entity-component system with reactive data binding

---

## Revolutionary Architecture: BEntity System

### Core Concept: Everything is an Entity
Critterding2 uses a **hierarchical entity-component system** where *all* data is stored as entities in a tree structure:

```cpp
class BEntity {
    // Entities have:
    // - name (string identifier)
    // - class_id (type identifier)
    // - children (hierarchical tree)
    // - properties (typed values: float, int, bool, reference)
    // - connections (reactive data flow)
};
```

### Key Entity Types
- **BEntity_float**: Stores float value, can connect to other entities
- **BEntity_int**: Integer storage
- **BEntity_uint**: Unsigned integer storage
- **BEntity_bool**: Boolean flag
- **BEntity_reference**: Pointer to another entity
- **BEntity_external**: Reference to external child entity

### Reactive Data Flow
Entities can **connect** to each other creating reactive data pipelines:

```cpp
// Example: Constraint angle → brain input
angle->connectServerServer(constraint_angle_input);

// When angle updates, input automatically updates
angle->onUpdate(); // propagates to connected entities
```

This is similar to reactive programming frameworks (RxJS, React) but implemented in C++ for real-time simulation.

---

## Neural System Architecture

### Brain System (`be_plugin_brainz`)
The brain system is a **separate plugin** that manages all neural networks in the simulation.

#### Neural Parameters (from brain_system.cpp)
```cpp
// Neuron Parameters
firing_threshold_min: -2000 (integer, scaled by 0.001 = -2.0)
firing_threshold_max: +2000 (integer, scaled by 0.001 = +2.0)
firing_weight_min: -400 (integer, scaled by 0.001 = -0.4)
firing_weight_max: +400 (integer, scaled by 0.001 = +0.4)

// Synapse Parameters
synapse_weight_min: -2000 (integer, scaled by 0.001 = -2.0)
synapse_weight_max: +2000 (integer, scaled by 0.001 = +2.0)

// Network Size (Adam brain)
neuron_adam_min: 50 neurons
neuron_adam_max: 100 neurons
synapse_adam_min: 2 synapses per neuron
synapse_adam_max: 6 synapses per neuron
```

### Neuron Model (BNeuron)
```cpp
class BNeuron : public BEntity_float {
    BEntity* m_firingWeight;     // Output weight when firing
    BEntity* m_firingThreshold;  // Threshold to exceed
    BEntity* m_firing;           // Bool: is currently firing
    BEntity* m_synapses;         // Container of input synapses
};
```

**Update Logic** (brain_system.cpp lines 100-160):
```cpp
// Step 1: Check all neurons if threshold breached
for neuron in neurons:
    potential = neuron.get_float()
    threshold = neuron.m_firingThreshold.get_float()
    
    if threshold > 0:
        if potential >= threshold:
            neuron.m_firing.set(true)
            neuron.set(0.0f)  // reset potential
        elif potential < 0:
            neuron.set(0.0f)  // clamp negative
    
    elif threshold < 0:
        if potential <= threshold:
            neuron.m_firing.set(true)
            neuron.set(0.0f)
        elif potential > 0:
            neuron.set(0.0f)

// Step 2: Propagate firing neurons
for neuron in neurons:
    if neuron.m_firing.get_bool():
        neuron.m_firing.set(false)
        neuron.m_firingWeight.onUpdate()  // propagates to connected synapses
```

**Key Insight**: **Positive and negative thresholds** allow neurons to fire in response to excitation OR inhibition. This is more flexible than typical integrate-and-fire models.

### Synapse Model
```cpp
class Synapse : public BEntity_float {
    BEntity* m_parent_neuron;  // Neuron this synapse belongs to
    BEntity* m_weight;         // Synaptic weight
};

// When synapse receives signal (from connected neuron firing)
bool Synapse::set(const Bfloat& value) {
    // Multiply firing weight by synapse weight, add to parent neuron potential
    m_parent_neuron->set(
        m_parent_neuron->get_float() + (m_weight->get_float() * value)
    );
    return true;
}
```

**Connection Flow**:
```
Input → Synapse (weighted) → Neuron (accumulates) → Fires → FiringWeight → Output
```

### Brain Construction (brain_system.cpp lines 194-318)
When a new brain is created:
1. **Create N neurons** (50-100 random)
2. **For each neuron**:
   - Connect to 1 random other neuron (guaranteed connectivity)
   - Add 2-6 additional input synapses from random neurons
3. **Connect outputs**: 2 random neurons → each motor output
4. **Connect inputs**: 1 random input → each neuron (via synapse)

**Vision Input**: 8x8 retina = 64 pixels × 4 channels (RGBA) = **256 vision inputs** (line 263-270)

---

## Mutation System

### Mutation Parameters (brain_system.cpp lines 39-84)
```cpp
mutation_chance: 10% (per offspring)
mutation_runs: 1-20 operations per mutation
slightly_percent: 5% (for "slightly" mutations)
```

### Mutation Operators (Weighted Selection)
```cpp
// Neuron mutations
m_mutationweight_neuron_add: 25
m_mutationweight_neuron_remove: 25
m_mutationweight_neuron_alter_firingweight: 25
m_mutationweight_neuron_alter_firingthreshold: 25
m_mutationweight_neuron_alter_firingweight_slightly: 50
m_mutationweight_neuron_alter_firingthreshold_slightly: 50

// Synapse mutations
m_mutationweight_synapse_add_neuron_to_neuron: 50
m_mutationweight_synapse_add_neuron_to_output: 10
m_mutationweight_synapse_add_input_to_neuron: 10
m_mutationweight_synapse_remove_from_neuron: 50
m_mutationweight_synapse_remove_from_output: 10
m_mutationweight_synapse_alter_weight: 80
m_mutationweight_synapse_alter_weight_slightly: 160
```

**Total weight sum**: ~600, creating probability distribution for mutation type selection.

### Key Mutation Details
- **"Slightly" mutations**: Adjust parameter by ±5% of random value within range
- **Add neuron**: Creates neuron with 2-6 synapses, connects to random neurons
- **Input connection bias**: When adding synapse to new neuron, 1/7 chance it connects from input rather than another neuron (line 440-467)
- **Safety checks**: Prevents removing last synapse, ensures at least 1 neuron remains

---

## Critter Lifecycle System

### Critter Parameters (critter_system.cpp lines 49-72)
```cpp
// Population
minimum_number_of_units: 20
insert_frame_interval: 20 frames (checks every 20 frames)

// Energy
initial_energy: 1500.0
procreate_minimum_energy: 2501.0

// Lifespan
maximum_age: 18000 timesteps
```

### Critter Structure (CdCritter)
```cpp
class CdCritter {
    BEntity* m_brain;              // Reference to brain in brain_system
    BEntity* m_brain_inputs;       // Shortcut to inputs
    BEntity* m_age;                // Age in timesteps
    BEntity* m_energy;             // Current energy level
    BEntity* m_species;            // Species assignment
    BEntity* m_transform_shortcut; // Physics body transform
    BEntity* m_bodyparts_shortcut; // Body parts container
};
```

### Update Cycle (critter_system.cpp process())
```cpp
1. Age all critters (+1 per timestep)
2. Check for minimum population (insert random if below threshold)
3. Death: age >= max_age OR energy <= 0
4. Procreation:
   - if energy >= procreate_minimum_energy
   - AND motor_neuron "procreate" != 0.0
   - then: parent energy /= 2, create offspring via copy + mutate
```

### Procreation Details (lines 284-351)
```cpp
1. Copy entire critter (brain + body) using BEntityCopy system
2. Position offspring 0.75 units above parent
3. Reset offspring age to 0
4. Mutate brain (10% chance, 1-20 mutation operations)
5. Track "adam_distance" (generation count)
```

---

## Body System Architecture

### Body Parameters (body_system.cpp)
```cpp
// Bodypart scales (randomized per critter)
bodypart_scale_x: 0.2 - 4.0
bodypart_scale_y: 0.2 - 4.0
bodypart_scale_z: 0.2 - 4.0

// Physics properties
bodypart_spacing: 0.07
bodypart_friction: 0.95
bodypart_restitution: 0.95 (bounciness)
bodypart_density: 100.0
```

### Fixed Body Type: "Tergite" (Segmented Arthropod)
The current implementation uses a **fixed body plan** inspired by arthropod segments (tergites):

```
Segment 1 (head/thorax):
  - Central body (0.4 × 0.2 × 0.4)
  - 8 legs (4 pairs): joint (0.1 × 0.2 × 0.2) + leg (0.4 × 0.2 × 0.2)

Segment 2 (abdomen):
  - Central body (0.7 × 0.2 × 0.7)
  - 8 legs (4 pairs)

Constraints:
  - Hinge joint between segments (-0.5 to +0.5 rad limits)
  - Hinge joints on each leg (bidirectional)
```

**Motor Outputs**: Each hinge joint → brain motor neuron via reactive connection

**Body mutation**: Currently **disabled** (commented out in body_system.h lines 20-54), but infrastructure exists for:
- Add/remove bodyparts
- Rescale bodyparts
- Add/remove heads, eyes, spikes
- Alter joint constraints

---

## Comparison with Other Flamoot Systems

### vs. CritterGOD4 (2010 C++ prototype)
| Feature | CritterGOD4 | Critterding2 |
|---------|-------------|--------------|
| Neuron scale | 16,384 neurons | 50-100 neurons |
| Threshold range | -2000 to +2000 | -2000 to +2000 ✓ |
| Weight range | ±5.0 | ±2.0 |
| Plasticity | STDP | None (fixed weights) |
| Binary spikes | Yes (-1/0/+1) | No (float potentials) |
| Continuous weakening | Yes | No |
| Body mutation | Yes | Disabled (infrastructure exists) |
| Physics | Bullet Physics | Bullet Physics ✓ |

**Key Difference**: Critterding2 uses **reactive entity connections** instead of explicit STDP plasticity. Learning happens through evolution, not synaptic modification.

### vs. Critterding Beta14 (reference implementation)
| Feature | Beta14 | Critterding2 |
|---------|--------|--------------|
| Language | C++ | C++ ✓ |
| UI | Custom | Qt6 |
| Graphics | SDL | OpenGL |
| Data structure | Custom classes | Entity-component tree |
| Threading | Single-threaded | app_critterding_threads variant |
| Modularity | Monolithic | Plugin system |

**Architectural Leap**: Critterding2's entity-component system enables:
- GUI admin panel (inspect/modify any entity at runtime)
- Reactive data binding (automatic UI updates)
- Serialization (entity tree → file)
- Networked simulation (entity references across machines)

### vs. CritterGOD (Python)
| Feature | CritterGOD | Critterding2 |
|---------|------------|--------------|
| Circuit8 | Yes (64×48×1024) | No |
| Drugs | Yes (5 molecules) | No |
| Markov text | Yes | No |
| Audio synthesis | Yes | No |
| Visual patterns | Yes | No |
| 3D bodies | Planned | Yes ✓ |
| GUI | Pygame | Qt6 ✓ |

**Complementary Strengths**:
- Critterding2: Production-ready simulation engine
- CritterGOD: Generative/psychedelic features

---

## Unique Innovations in Critterding2

### 1. Entity-Component Reactive System
All data (neurons, synapses, bodies, UI elements) represented as entities in a unified tree:

```
Critterding/
  critter_system/
    unit_container/
      critter_unit/
        age: 523
        energy: 1834.2
        external_brain → brain_system/unit_container/brain
        external_body → body_system/unit_container/body_fixed1
        motor_neurons/
          eat: 0.0
          procreate: 0.0
```

**Benefits**:
- **Inspection**: Admin panel can navigate entire entity tree
- **Serialization**: Save/load entire simulation state
- **Modularity**: Systems (brain, body, vision) are independent plugins
- **Debugging**: Entity connections visualized in GUI

### 2. Motor Neurons as Float Values
Unlike traditional "motor neuron fires → muscle contracts", Critterding2 uses:

```cpp
motor_neurons/eat: float (0.0 = not eating, non-zero = eating)
motor_neurons/procreate: float (0.0 = not procreating, non-zero = procreate)
```

These are **reactive properties** that brain outputs connect to, allowing fine-grained control.

### 3. Bidirectional Threshold Neurons
Neurons can fire when potential exceeds **positive** threshold OR falls below **negative** threshold:

```cpp
if threshold > 0:
    fire if potential >= threshold
elif threshold < 0:
    fire if potential <= threshold
```

This enables neurons specialized for **excitation** (positive threshold) or **inhibition** (negative threshold).

### 4. Multi-Threaded Architecture
`app_critterding_threads` variant suggests parallel execution:
- Physics simulation thread
- Brain update thread
- Rendering thread
- UI thread

### 5. Plugin System
Core engine (`kernel/`) + plugins:
- `be_plugin_app_critterding` - Main simulation
- `be_plugin_brainz` - Neural networks
- `be_plugin_bullet` - Physics integration
- `be_plugin_thread` - Threading support

Enables experimentation without recompiling core.

---

## Notable Implementation Details

### Adam Distance Tracking
Each critter has `adam_distance` (generation count from first random critter):

```cpp
auto ad = critter_new->getChild("adam_distance", 1);
ad->set(ad->get_uint() + 1);
```

Used to analyze **evolutionary lineages** and **measure adaptation time**.

### Collision Management
When critter dies, system removes it from collision detection lists (lines 360-425):
```cpp
// Loop through all collisions, remove any involving this critter's bodyparts
for collision in m_collisions:
    if e1 or e2 in critter.m_bodyparts:
        m_collisions->removeChild(collision)
```

Prevents **dangling references** that would crash physics engine.

### Mouse Picker Integration
Critters can be **grabbed and moved** with mouse (line 375-381):
```cpp
if m_mouse_picker:
    for bodypart in critter.m_bodyparts:
        m_mouse_picker->removeGrabbedEntity(bodypart)
```

### Vision Input Hack (commented out)
Lines 295-308 show **experimental vision weight reduction**:
```cpp
// Kick vision synapse weight down a notch
if input.name == "vision_value_R/G/B/A":
    weight = 0.2 * weight  // Reduce vision influence
```

Suggests **vision overload problem** - too many vision inputs (256) dominating brain.

---

## Parameter Heritage

### Preserved from Critterding Legacy
- **Threshold range**: -2000 to +2000 (×0.001 scaling)
- **Weight range**: ±2000 for synapses
- **Energy system**: Initial 1500, procreate 2501, max age 18000
- **Mutation chance**: 10%
- **Physics properties**: Friction 0.95, restitution 0.95

### New in Critterding2
- **Neuron count**: 50-100 (down from 16k in CritterGOD4)
- **Synapse count**: 2-6 per neuron (down from 40)
- **Firing weight range**: ±400 (new parameter)
- **Vision resolution**: 8×8 RGBA (256 inputs)
- **Body scale ranges**: 0.2-4.0 (up from 0.1-2.0)

---

## Comparison Summary

### Strengths of Critterding2
✅ **Production-quality architecture** (Qt6, OpenGL, threading)  
✅ **Entity-component system** (unified data model, introspection)  
✅ **Bullet Physics integration** (realistic 3D bodies)  
✅ **Plugin architecture** (extensible without recompiling)  
✅ **GUI admin panel** (runtime inspection/modification)  
✅ **Reactive data binding** (automatic UI updates)  
✅ **Serialization support** (save/load entity trees)  

### Missing from Critterding2 (vs CritterGOD)
❌ **No synaptic plasticity** (STDP, learning within lifetime)  
❌ **No Circuit8** (shared morphic field)  
❌ **No psychopharmacology** (drug effects)  
❌ **No generative systems** (audio, text, visual art)  
❌ **Small networks** (50-100 neurons vs 16k)  
❌ **Fixed bodies** (mutation disabled)  

---

## Recommendations for CritterGOD Integration

### High Priority (Architectural Patterns)
1. **Entity-component introspection**: Add debug UI to navigate creature state tree
2. **Reactive data binding**: Explore numpy broadcasting for automatic propagation
3. **Plugin architecture**: Separate core from generators (audio, markov, visual)
4. **Bidirectional thresholds**: Allow neurons to fire on negative potential

### Medium Priority (Features to Port)
5. **Multi-threaded execution**: Separate threads for physics, brains, rendering
6. **Adam distance tracking**: Track generational depth for lineage analysis
7. **Motor neurons as floats**: Replace binary firing with continuous values
8. **Vision weight reduction**: Scale down vision synapse weights (avoid overload)

### Low Priority (Future Exploration)
9. **Qt6 GUI**: Only if CritterGOD needs professional UI (probably not)
10. **Bullet Physics**: Already planned for Phase 5
11. **Serialization**: Entity tree → file format for save/load

---

## Key Takeaways

### What Critterding2 Teaches Us
1. **Simplicity works**: 50-100 neurons sufficient for evolved behavior
2. **Entity-component systems scale**: Uniform data model from neurons to UI
3. **Reactive programming in C++**: connectServerServer() pattern enables clean data flow
4. **Fixed bodies acceptable**: Complex behavior emerges even without body mutation
5. **Threading essential**: Physics + brains + rendering need parallel execution

### Philosophical Alignment
Critterding2 embodies flamoot's **incremental evolution** approach:
- Start with **working system** (critterding beta14)
- Add **modern infrastructure** (Qt6, threading, plugins)
- Preserve **core parameters** (thresholds, energy, mutation)
- Enable **future expansion** (body mutation disabled but implemented)

CritterGOD takes the **revolutionary features** approach:
- Start with **neural fundamentals** (STDP, spiking)
- Add **psychedelic systems** (Circuit8, drugs, markov)
- Enable **generative emergence** (audio, text, visual)
- Scale to **cosmic complexity** (10k+ neurons, multi-modal)

**Both are valid paths**. Critterding2 prioritizes **stability and usability**. CritterGOD prioritizes **emergence and experimentation**.

---

## Files Analyzed
- `src/plugins/be_plugin_app_critterding/critter_system.h` (82 lines)
- `src/plugins/be_plugin_app_critterding/critter_system.cpp` (440 lines)
- `src/plugins/be_plugin_brainz/brain_system.h` (97 lines)
- `src/plugins/be_plugin_brainz/brain_system.cpp` (1002 lines)
- `src/plugins/be_plugin_app_critterding/body_system.h` (91 lines)
- `src/plugins/be_plugin_app_critterding/body_system.cpp` (200 lines analyzed)

**Total lines analyzed**: ~1900 lines of core simulation code

---

## Conclusion

Critterding2 represents flamoot's **mature, production-oriented** artificial life platform. It sacrifices some of the **psychedelic complexity** of CritterGOD4 and telepathic-critterdrug in favor of **architectural robustness**.

Key innovations:
- **Entity-component reactive system** (everything is an entity tree)
- **Bidirectional threshold neurons** (fire on excitation OR inhibition)
- **Plugin architecture** (brain/body/physics as separate systems)
- **Qt6 GUI** (runtime introspection and modification)

For CritterGOD, the most valuable lessons are:
1. **Reactive data binding** (automatic propagation)
2. **Bidirectional thresholds** (richer neuron behavior)
3. **Entity introspection** (debug UI for state trees)
4. **Threading architecture** (parallel simulation)

The **generative features** (Circuit8, drugs, markov, audio, visual) remain unique to CritterGOD and should be preserved as the system's defining characteristics.
