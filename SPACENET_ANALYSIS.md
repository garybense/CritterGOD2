# SpaceNet Analysis

**Discovery Date**: January 24, 2026  
**Location**: `/Users/gspilz/code/critters/spacenet-code-r221`  
**Version**: r221  
**Author**: "seh" (flamoot)  
**Language**: Java (multiple subprojects)  
**Purpose**: 3D spatial graph visualization and interaction system

## Overview

**SpaceNet** is flamoot's **3D visualization framework** for representing and navigating graph structures in immersive 3D space. It's the **visual/spatial interface** layer that complements:
- **CritterGOD** (neural/subsymbolic processing)
- **Netention** (semantic/symbolic reasoning)
- **SpaceNet** (spatial/visual representation)

Together, these three systems form a complete **embodied cognitive architecture**:
```
SpaceNet (Spatial)
    ↓
Netention (Semantic)
    ↓  
CritterGOD (Neural)
```

## Architecture

### Multiple Subprojects

The codebase contains **7 iterations** (protojava1-7) plus **2 main implementations**:

1. **automenta.spacegraph.1** - Core graph abstractions
2. **automenta.spacenet-ardor1** - First Ardor3D implementation
3. **automenta.spacenet-ardor2** - Refined Ardor3D version
4. **protojava1-7** - Progressive development iterations

This shows **active exploration** of 3D graphics APIs (JME, Ardor3D).

### Core Components

#### 1. Spacetime (space/Spacetime.java)

```java
public interface Spacetime {
    ArdorCamera getCamera();     // Viewpoint navigation
    DefaultPointer getPointer(); // 3D interaction
    Color getBackgroundColor();
    Space getFace();             // Main 3D scene
    Space getSky();              // Background/skybox
}
```

**Spacetime** is the 3D rendering context - analogous to the "universe" containing all spatial objects.

#### 2. Graph (var/graph/Graph.java)

```java
/** combination of directed graph and hypergraph */
public interface Graph<V, H> {
    V addNode(V node);
    boolean removeNode(V node);
    
    H addEdge(H edge, EdgeType edgeType, V... vList);
    boolean removeEdge(H hyperedge);
    
    void add(IfGraphChanges ic);  // Observers
    Iterator<H> iterateEdges();
    Iterator<V> iterateNodes();
}
```

**Graph** supports:
- Regular edges (node-to-node)
- **Hyperedges** (connecting multiple nodes)
- Change notifications (reactive updates)

#### 3. GraphBox (space/geom/graph/GraphBox.java)

Renders graph structures as 3D objects:
- **Nodes** → 3D boxes/spheres/shapes
- **Edges** → Lines/tubes connecting nodes
- **Layout** → Force-directed, tree, mesh, etc.

#### 4. Force-Directed Layout

```java
ForceDirectedParameters params = new ForceDirectedParameters(
    new Vector3(50, 50, 50),  // Space dimensions
    0.025,                     // Attraction
    0.03,                      // Repulsion
    1.0                        // Damping
);
```

**Physics-based graph layout**:
- Nodes repel each other (like charged particles)
- Edges attract connected nodes (like springs)
- Simulates until equilibrium reached

#### 5. Widgets & UI (space/widget/)

3D user interface components:
- **Button** - Clickable 3D buttons
- **Window** - Floating 3D panels
- **Panel** - Container for UI elements
- **Spinner** - Value adjusters
- **Text3D** - 3D text rendering

#### 6. Plugins (plugin/)

Extensions for different data sources:
- **Web** - HTML, MediaWiki, Yahoo search, Twitter
- **File** - File system navigation
- **Shell** - Command execution
- **MVEL** - Dynamic expression evaluation

## Key Features

### 1. Immersive Graph Navigation

Navigate graph structures in **3D space**:
- Camera controls (fly, orbit, zoom)
- Click nodes to focus
- Drag-and-drop to arrange
- Multiple camera modes

### 2. Force-Directed Physics

Automatic **organic layout**:
- Related nodes cluster together
- Unrelated nodes spread apart
- Real-time physics simulation
- Adjustable parameters

### 3. Multi-Scale Visualization

Handle graphs from **small to massive**:
- LOD (Level of Detail) system
- Frustum culling (only render visible)
- Attention thresholds (fade distant nodes)
- Progressive loading

### 4. Rich 3D Primitives

Diverse visual representations:
- Boxes, spheres, cylinders
- Text labels (2D and 3D)
- Colored edges with varying thickness
- Transparent surfaces
- Particle effects

### 5. Interactive Elements

**Live 3D UI**:
- Buttons trigger actions
- Windows contain controls
- Text fields accept input
- All in 3D space (not overlays)

### 6. Data Source Integration

**Plugins connect to external data**:
- Web scraping → graph
- File system → tree graph
- Twitter → social graph
- MediaWiki → knowledge graph

## Graphics Technology

### Ardor3D Engine

Primary rendering engine (ardor1/ardor2):
- **OpenGL-based** 3D graphics
- Scene graph architecture
- Advanced lighting/shading
- Post-processing effects

### JMonkeyEngine (JME)

Alternative engine (protojava7):
- **Java game engine**
- Similar capabilities to Ardor3D
- More mature ecosystem
- Better documentation

**Why both?** Flamoot was **exploring options** - Ardor3D was more flexible, JME more stable.

## Connection to CritterGOD

### Visualization Potential

**SpaceNet could visualize CritterGOD's neural networks**:

1. **Neural Network as Graph**
   ```
   Neurons → 3D nodes (colored by type)
   Synapses → Edges (thickness = weight)
   Activity → Glow/pulse animation
   ```

2. **Morphic Field (Circuit8) as 3D Space**
   ```
   64x48 pixels → 3D grid
   1024 depth layers → Temporal history
   Creature movements → Trails in space
   ```

3. **Population Evolution as Growing Tree**
   ```
   Root → Initial random genotype
   Branches → Mutations
   Leaves → Current population
   Color → Fitness level
   ```

4. **Creature Relationships as Social Graph**
   ```
   Creatures → Nodes
   Interactions → Edges
   Distance → Genetic similarity
   ```

### Real-World Analogies

**Human spatial cognition**:
- **V1/V2 cortex** (CritterGOD) - Low-level vision processing
- **Parietal cortex** (SpaceNet) - Spatial reasoning, navigation
- **Prefrontal cortex** (Netention) - Abstract planning

**SpaceNet provides the "where" system** - spatial relationships and navigation.

### Integration Architecture

**Complete flamoot cognitive stack**:

```
┌─────────────────────────────┐
│   SpaceNet (Spatial UI)     │  ← Human perceives & interacts
├─────────────────────────────┤
│  Netention (Symbolic KB)    │  ← High-level reasoning
├─────────────────────────────┤
│  CritterGOD (Neural Nets)   │  ← Low-level processing
└─────────────────────────────┘
```

**Data flow**:
1. **CritterGOD** → Neural activity patterns
2. **Netention** → Semantic interpretation
3. **SpaceNet** → 3D visualization

**Interaction flow**:
1. **Human** → Manipulates SpaceNet objects
2. **SpaceNet** → Updates Netention concepts
3. **Netention** → Modulates CritterGOD neurons

## Design Patterns Worth Adopting

### 1. Scene Graph Architecture

Hierarchical object containment:
```java
Space root = new Space();
Space creature = root.add(new CreatureNode());
Space brain = creature.add(new BrainNode());
Neuron n1 = brain.add(new NeuronNode());
```

**Apply to CritterGOD**:
```python
# Hierarchical visualization
world = VisualizationRoot()
population = world.add(PopulationNode(creatures))
for creature in creatures:
    creature_node = population.add(CreatureNode(creature))
    brain_node = creature_node.add(BrainNode(creature.network))
```

### 2. Observer Pattern for Graph Changes

```java
void add(IfGraphChanges ic);  // Register observer
```

When graph changes → notify observers → update visualization.

**Apply to CritterGOD**:
```python
network.on_synapse_change(lambda s: update_visualization(s))
network.on_neuron_fire(lambda n: pulse_animation(n))
```

### 3. Force-Directed Layout

**Automatic spatial organization**:
- Attraction = semantic similarity
- Repulsion = maintain visibility
- Result = organic clustering

**Apply to CritterGOD**:
```python
# Organize creatures by genetic similarity
for creature in population:
    attraction = genetic_similarity(creature, others)
    repulsion = personal_space_radius
    update_position(creature, forces)
```

### 4. Attention-Based Rendering

```java
AttentionThresholdGraph  // Only render "important" nodes
```

Focus computational resources on **relevant** parts of visualization.

**Apply to CritterGOD**:
```python
# Only render neurons with high activity
visible_neurons = [n for n in network.neurons 
                  if n.potential > attention_threshold]
```

### 5. Reactive Data Binding

Variables auto-update UI:
```java
DoubleVar energy = new DoubleVar(1000.0);
Slider energySlider = new Slider(energy);  // Auto-syncs
```

**Apply to CritterGOD**:
```python
# UI automatically reflects creature state
energy_display = ReactiveLabel(creature.energy.energy)
fitness_graph = ReactiveChart(population.fitness_history)
```

## Implementation Recommendations

### Phase 5c: Visualization System (Optional)

**If** we want 3D visualization for CritterGOD:

#### Option 1: Web-Based (Recommended)
- **Three.js** - JavaScript 3D library
- **D3-force** - Force-directed layout
- **WebGL** - GPU acceleration
- **Browser-based** - No installation needed

```python
# Export data to JSON
visualization_data = {
    'neurons': [{'id': n.id, 'type': n.type, 'position': [x,y,z]} 
                for n in network.neurons],
    'synapses': [{'source': s.pre.id, 'target': s.post.id, 'weight': s.weight}
                 for s in network.synapses]
}
```

#### Option 2: Python Native
- **PyQt5 + PyOpenGL** - Native 3D
- **VTK** - Scientific visualization
- **Mayavi** - 3D plotting
- **Panda3D** - Game engine

#### Option 3: Pygame 2D (Current)
Already have pygame for Circuit8 - extend for neural visualization:
```python
# 2D projection of 3D network
for synapse in network.synapses:
    x1, y1 = project_2d(synapse.pre_neuron.position)
    x2, y2 = project_2d(synapse.post_neuron.position)
    thickness = int(abs(synapse.weight) * 5)
    color = RED if synapse.weight < 0 else BLUE
    pygame.draw.line(screen, color, (x1,y1), (x2,y2), thickness)
```

### Minimal Integration (Practical)

**Simpler approach** - Learn from SpaceNet's patterns:

1. ✅ **Force-directed creature placement** (easy)
   ```python
   # Creatures arrange themselves by similarity
   def update_positions(creatures, dt):
       for c1 in creatures:
           force = Vector2(0, 0)
           for c2 in creatures:
               if c1 != c2:
                   distance = (c2.pos - c1.pos).length()
                   similarity = genetic_similarity(c1, c2)
                   # Attract if similar, repel if different
                   force += (c2.pos - c1.pos).normalize() * (similarity - 0.5)
           c1.velocity += force * dt
           c1.pos += c1.velocity * dt
   ```

2. ✅ **Reactive visualization** (medium)
   ```python
   # Update display when state changes
   class ReactiveDisplay:
       def __init__(self, get_value):
           self.get_value = get_value
       def update(self):
           self.current_value = self.get_value()
           self.render()
   ```

3. ✅ **Observer pattern** for neural events (medium)
   ```python
   # Notify observers when neurons fire
   class NeuralNetwork:
       def __init__(self):
           self.fire_observers = []
       
       def on_fire(self, callback):
           self.fire_observers.append(callback)
       
       def _neuron_fired(self, neuron):
           for observer in self.fire_observers:
               observer(neuron)
   ```

4. ⚠️ **Full 3D rendering** (complex, Phase 6+)

## Relationship to Other Discoveries

**Timeline Context**:
- **2010**: CritterGOD4 (C++/Bullet) - Neural substrate
- **2010-2011**: Netention (Java) - Symbolic reasoning
- **2010-2011**: SpaceNet r221 (Java) - 3D visualization
- **2015+**: telepathic-critterdrug - Morphic integration
- **2020+**: SDL visualizers - Audio/visual generation
- **2026**: Current CritterGOD (Python) - Synthesis

**SpaceNet represents flamoot's visualization research** - the "eye" to see CritterGOD's "mind".

## Conclusions

### What SpaceNet Is

**SpaceNet** is a **3D spatial graph visualization framework** for immersive data navigation. It's flamoot's exploration of:
- Spatial cognition
- 3D user interfaces
- Force-directed layouts
- Multi-scale visualization
- Real-time graph rendering

### What SpaceNet Is NOT

- It's **not** a neural network system (that's CritterGOD)
- It's **not** a semantic reasoner (that's Netention)
- It's **not** directly related to evolution

### Relevance to CritterGOD

**Direct relevance**: Medium (visualization tool)

**Conceptual relevance**: High (spatial representation)

**Practical value**:
1. ✅ **Force-directed layouts** for creature positioning (easy)
2. ✅ **Observer pattern** for reactive updates (medium)
3. ✅ **Attention-based rendering** for performance (medium)
4. ⚠️ **Full 3D visualization** (complex, future phase)

### Key Takeaway

**The Complete Flamoot Stack**:

```
SpaceNet    → Spatial ("Where is it?")
Netention   → Symbolic ("What is it?")
CritterGOD  → Subsymbolic ("How does it feel?")
```

This is the **complete embodied cognitive architecture**:
- **Sensing** (CritterGOD retinal arrays)
- **Processing** (CritterGOD neural networks)
- **Reasoning** (Netention semantic graphs)
- **Visualizing** (SpaceNet 3D rendering)
- **Acting** (CritterGOD motor outputs)

For now, CritterGOD focuses on **neural/generative layers**. SpaceNet patterns inform future **visualization extensions**.

---

**Status**: Analysis complete  
**Recommendation**: 
- Adopt design patterns (force-directed, observers, reactive binding)
- Use pygame/matplotlib for 2D visualization (Phase 5)
- Consider Three.js web visualization (Phase 6)
- Defer full 3D system to future phases

**Next**: Fourth folder analysis or begin Phase 5 implementation
