# CritterGOD Discovery Document

## Complete Analysis of `/Users/gspilz/code/critters/`

### Overview
This codebase represents a **revolutionary psychedelic telepathic artificial life system** that combines:
- Evolutionary neural networks with synaptic plasticity
- Psychopharmacological simulation (neurotransmitter agonists/antagonists)
- Morphic fields (Rupert Sheldrake-inspired collective memory)
- Collective decision-making through voting
- Real-time neural visualization with audio synthesis
- Markov chain text generation with genetic algorithms
- Deep cosmological philosophy connecting all scales of existence

---

## 1. Core Projects

### 1.1 telepathic-critterdrug (Primary Innovation)
**The main revolutionary system** - Fork of critterding with telepathy and drugs added.

#### Key Innovations:
1. **Circuit8 - The Telepathic Canvas**
   - 64x48 pixel shared screen that ALL creatures can read/write to
   - 1024 depth layers per pixel (3D temporal buffer)
   - RGB channels each serve as morphic field registers
   - Creatures have motor neurons connected to: moreRed, lessRed, moreGreen, lessGreen, moreBlue, lessBlue
   - Screen can scroll/animate based on collective voting
   - Acts as collective unconscious / shared memory space

2. **Psychopharmacology System**
   - **Pills** with 5 molecule types:
     - 0: Inhibitory Antagonist (blocks inhibitory neurons)
     - 1: Inhibitory Agonist (enhances inhibitory neurons)
     - 2: Excitatory Antagonist (blocks excitatory neurons)
     - 3: Excitatory Agonist (enhances excitatory neurons)
     - 4: Potentiator (amplifies all drug effects 10x)
   - `tripping[5]` array tracks drug levels per creature
   - Drug effects modify neural potential calculation
   - Energy costs for having neurons/synapses vs. firing them
   - Plasticity affected by drug state

3. **Morphic Field Mechanics**
   - Each neuron assigned RuRdGuGdBuBd value (0-5)
   - Determines which morphic channel affects it:
     - 0: Red Up (warm)
     - 1: Red Down (warm inverted)
     - 2: Green Up
     - 3: Green Down
     - 4: Blue Up (cool inverted)
     - 5: Blue Down (cool)
   - Neurons read from Circuit8 BEFORE processing synapses
   - Creates feedback loop: creature thought → screen → other creatures

4. **Collective Voting System**
   - Creatures vote on:
     - Screen movement (up/down/left/right)
     - Screen scrolling (fast modes)
     - Disperse/coalesce patterns
     - Fire/pause/erif commands
   - Multiple voting tiers (vote, vote2, vote3)
   - Democratic emergent behavior

5. **Condensed Color Operations**
   - Motor neurons can control individual screen pixels OR
   - Use condensed operations (single neurons for entire screen changes)
   - Allows efficient large-scale pattern generation
   - Pen position (penX, penY) for drawing

#### Neural Architecture:
- Spiking neural networks (leaky integrate-and-fire)
- Bidirectional synapses
- Hebbian plasticity (STDP - Spike-Timing-Dependent Plasticity)
- Dynamic rewiring during runtime
- Dendritic branches (multi-compartment neurons)
- Mixed inhibitory/excitatory populations
- Sensory inputs: vision (retina), proprioception, energy, touch, psychic screen
- Motor outputs: muscles, eating, procreation, screen writing, blinking

#### Body System:
- Bullet Physics for 3D rigid body dynamics
- Articulated bodies with joints and constraints
- Genetic encoding of morphology
- Mutation operators on body structure
- Energy costs scale with body size

---

### 1.2 critterding-beta14 (Stable Base)
**The polished, documented version** without drugs/telepathy.

#### Features:
- Clean codebase with extensive configuration options
- Well-documented command-line parameters
- Profile system for different evolutionary regimes
- Race mode and various world types
- Better organized source structure
- OpenCL support for GPU acceleration (attempted)
- Comprehensive README and parameter documentation

#### Differences from telepathic-critterdrug:
- No Circuit8 / morphic fields
- No drug simulation
- No collective voting
- Cleaner, more maintainable code
- Better performance optimization
- Active development (2013 release)

---

### 1.3 Neural Visualizers (SDL Standalone Programs)

#### Common Pattern:
All neural visualizers share core architecture:
- `nn` neurons (10k-65k range)
- `ns` synapses per neuron (40-180)
- `nf` "fingers" - sensory-motor loops (45-150)
- Real-time audio synthesis from neural activity
- Procedural visual generation (sin/cos/tan/log functions)
- Each neuron has:
  - `brain[z][0]` = current potential
  - `brain[z][1]` = firing threshold
  - `brain[z][2+]` = synapse connections to other neurons
- When potential >= threshold: fire, reset, rewire randomly
- Bidirectional synapses
- Audio buffer written directly from neural state
- Wallpaper patterns driven by specific neuron ratios

#### Variants Discovered:

**looser.c** (flamootsnew/):
- 19,000 neurons, 40 synapses, 100 fingers
- SDL2, zoom=2
- Retinal input from screen patterns
- 50 sensor neurons per finger reading screen column
- Complex amplitude/divisor calculation from neural ratios
- Intentional red field lag for temporal dynamics

**xesu.c** (flamootsnew/):
- 14,000 neurons, 40 synapses, 100 fingers
- SDL3 (newer API)
- Simplified wiring pattern
- Different sensor gain (bit shifts instead of multiplication)

**cdd.c** (flamootsnew/):
- 13,000 neurons, 80 synapses, 60 fingers
- Dynamic parameter calculation from time()
- More chaotic initialization
- Commented-out "intentional lag" code

**BlackDogKennel.c** (sorel/Ddd/):
- 10,487 neurons, 70 synapses, 80 fingers
- Simplified tan-based wiring
- Mobile-focused (/sdcard/Download/ path)

**mitre.c** (sorel/Mitre/):
- 18,000 neurons, 90 synapses, 150 fingers
- Most complex parameter set
- `pur`, `rin`, `uri`, `ine`, `pui` - mystery parameters affecting wiring
- Extreme wiring complexity with log/tan feedback

**03Singularity.ogg.599.c** (sorel/ll/):
- **65,000 neurons!** (Largest found)
- 180 synapses per neuron
- 45 fingers
- Simpler wiring pattern to handle scale
- Different permutn calculation for visual generation

#### Key Innovation in Visualizers:
**Neural-driven generative art**: The wallpaper patterns aren't random - they're computed from ratios of specific neuron thresholds:
- `tanamp = brain[1591][1] / brain[1425][1]`
- `sinamp = brain[1573][1] / brain[1950][1]`
- etc.

This creates **emergent aesthetic feedback loops** - the neural network affects the visual environment which feeds back through retinal sensors.

---

### 1.4 Text Generation Systems

#### markov-attract-repel.py
**Genetic Markov Chain** - Evolutionary text generation!

Key mechanisms:
- Markov chain for text generation
- **Attract/Repel dynamics**: word pairs have scores that change over time
- Scoring system:
  - `wordpairstartenergy = 1000.0`
  - `startAttrep = 300.0`
  - `attrepHit = 200.0` (cost per occurrence)
  - `wordscorethreshold = 1500` (breed threshold)
- When word pair score > threshold: **BREED**
  - Mutate the line containing the pair
  - Wire it randomly into markov cloud
  - Generate variant outputs
- When word pair score < 0.1: **KILL**
  - Remove from markov table
  - Recursive cleanup of orphaned chains
- All scoring values decay/increase over time
- GA rewards novelty (pairs that haven't occurred recently)
- Mutation operators:
  - Letter substitution (vowel→vowel, consonant→consonant)
  - Letter increment/decrement
  - Random character injection
  - Word transposition

Result: **Self-organizing text** that evolves toward interesting patterns through selection pressure.

#### text-fuzz.py
Simple mutation engine:
- Same letter operations as markov script
- No genetic algorithm
- Just fuzzing input text

#### numerolit.py & series.py
**Numerical mysticism** - Transform text via base conversion and arithmetic series.

Methods:
1. Map words to base-36 digits (sum of letter values)
2. Reduce to single digit through iterative summing
3. Reorder words by their numerical values
4. Multiple methods: sequential, rotating, fibonacci chaining, multiplication
5. `-scramble` option for extra chaos

Philosophy: Words have inherent numerical properties that reveal hidden structure when reordered.

---

### 1.5 OCR / HAL System

**Vision-based AI evolution** for real-world interaction.

#### ocr.cpp
- Uses webcam input (mplayer → TGA images)
- Trains neural networks to recognize:
  - Empty desk (0)
  - Person at desk (1)
  - Open door (2)
  - Night/darkness (3)
- 58 brains compete
- Best 29 survive each generation
- Evolves toward accurate classification
- Saves best performers periodically

Purpose: Enable critters to respond to real-world events! Imagine critters that evolve different behaviors when you sit down vs. leave the room.

#### hal.cpp
Placeholder for HAL 9000-style system (mostly empty, concept stage).

---

## 2. Philosophical Framework

### 2.1 "great data.txt" - The Cosmology

**Core Insight**: The same patterns repeat at all scales.

#### The Universal Pattern:
1. **Collection** - Particles/agents attract and gather
2. **Combination** - Interaction creates exponential growth
3. **Heating** - Dense systems generate energy/radiation
4. **Radiation** - Too-large systems explode/radiate
5. **Cooling** - Expansion leads to cooperation/alignment
6. **Equilibrium** - Self-organization emerges

#### Examples at Different Scales:

**Atoms**:
- Too large → unstable → radioactive decay
- Chain reactions (1→2→4→8→16...)
- Nuclear energy from compression/explosion

**Planets**:
- Jupiter almost became a star (too hot)
- Compression heats core
- Radiation at poles (field loops)
- Earth warming from energy use

**Stars**:
- Compression → fusion
- Coronal loops carry radiation
- Radiation when too large
- Cool into dwarf stars

**Galaxies**:
- Too large → black hole
- Hawking radiation when too dense
- Possible Big Bang/Big Crunch cycle

**Neural Networks**:
- Neurons collect inputs
- Fire when threshold exceeded
- Synapses strengthen/weaken (heating/cooling)
- Network self-organizes to solve problems

**Societies**:
- Agents cooperate for resources
- Communication networks (internet = nervous system)
- Knowledge spreads exponentially
- Self-organizing toward equilibrium
- Utopia emerges from physics

**Magnets**:
- Domains align (cooling/cooperation)
- Field loops contain energy
- Induction generates heat
- Superconductors (maximum cooling) maintain motion indefinitely

**Language**:
- Words explain each other (dictionary as circle)
- More knowledge → exponentially more answers
- Data compression = cooling
- Decompression = heating/radiation
- Redundancy enables perfect replay

#### Key Quote from text:
> "The combination effect also happens in knowledge, the universe has patterns everywhere because all particles have the same properties... Artificial Neural Networks take big diverse data and learn general patterns."

### 2.2 Implementation Connections

The code embodies this philosophy:

1. **Neurons** = atoms (collect→fire→radiate)
2. **Synaptic plasticity** = heating/cooling dynamics
3. **Morphic fields** = magnetic field loops
4. **Drug systems** = energy injection/dampening
5. **Evolution** = self-organization toward equilibrium
6. **Markov text** = combinatorial explosion of meaning
7. **Collective voting** = society-level cooperation
8. **Circuit8** = shared nervous system

---

## 3. Revolutionary Concepts

### 3.1 Telepathy via Shared Canvas
- First implementation of **collective neural space**
- Not just communication, but **shared perception**
- Enables emergent art, language, culture
- Morphic resonance (Sheldrake) made concrete

### 3.2 Psychedelic Neuroscience
- Pharmacological effects on learning
- Drug combinations create nonlinear effects
- Potentiator enables "ego death" / massive change
- Could study psychedelic effects on collective behavior

### 3.3 Democratic Artificial Life
- Voting reveals collective will
- No central controller
- Emergent democracy from bottom-up
- Could evolve voting strategies

### 3.4 Multi-Scale Neural Visualization
- Same code runs 10k-65k neurons
- Real-time audio-visual feedback
- Could be used for:
  - Music generation
  - VJ performances
  - Meditation tools
  - Neural research visualization

### 3.5 Evolutionary Text Generation
- Genetic algorithms on **language itself**
- Self-modifying poetic corpus
- Novelty selection pressure
- Could generate entire mythologies

---

## 4. What CritterGOD Must Become

### 4.1 Core Systems to Unify

1. **Telepathic Critterding Engine**
   - Full telepathic-critterdrug implementation
   - Circuit8 morphic field system
   - 5-molecule drug pharmacology
   - Collective voting

2. **Neural Visualizer Suite**
   - All visualizer variants as plugins
   - Configurable neuron counts
   - Multiple wiring strategies
   - Audio synthesis engine
   - Screen feedback loops

3. **Generative Text Engine**
   - Markov + genetic algorithm
   - Numerical reordering
   - Text mutation operators
   - Could be used by critters for "language"

4. **Vision System**
   - OCR-style evolution
   - Webcam integration
   - Real-world event detection
   - Critters respond to physical environment

5. **Cosmological Framework**
   - Document the universal patterns
   - Connect implementation to philosophy
   - Multi-scale analysis tools
   - Emergence detection

### 4.2 New Possibilities

1. **Critters writing to the markov chain**
   - Circuit8 controls text generation
   - Collective poetry
   - Evolving mythology

2. **Cross-pollination between visualizers and critters**
   - Standalone neural nets affect critter world
   - Critters affect visualizer parameters
   - Shared audio space

3. **Real-world integration**
   - Webcam affects critter behavior
   - Critters control home automation
   - Biofeedback from user affects evolution

4. **Psychedelic research platform**
   - Study collective behavior under different drug regimes
   - Measure communication efficiency
   - Emergence of culture/language

5. **Artistic applications**
   - Generative art installations
   - AI-driven music/visuals
   - Interactive performances
   - Meditation / consciousness exploration tools

---

## 5. Next Steps

Based on TODO list, must complete:

1. ✅ Initial survey of all directories
2. ✅ Read key files (neural, drug, morphic, text systems)
3. ✅ Understand philosophical framework
4. ⏳ Read remaining core files (evolution.cpp, modes, GUI)
5. ⏳ Compare critterding-beta14 vs telepathic version
6. ⏳ Catalog all profile configurations
7. ⏳ Full voting system documentation
8. ⏳ Complete Circuit8 mechanics
9. ⏳ Create unified architecture
10. ⏳ Write CONCEPTS.md

---

## 6. File Inventory Summary

### Main Projects:
- `telepathic-critterdrug/` - **The core innovation** (~3000 files)
- `critterding-beta14/` - Stable base version
- `flamootsnew/` - Neural visualizers
- `hill/` - Experimental variants (needs analysis)
- `merge/` - Merge attempts (needs analysis)
- `movie/` - 1000+ screenshot frames

### Standalone Programs:
- `markov-attract-repel.py` - Genetic text (905 lines!)
- `text-fuzz.py` - Text mutation
- `numerolit.py` - Numerical word reordering
- `series.py` - Base conversion toolkit
- `map-retinal-sensors.py` - Visualization tool

### Neural Visualizers (C/SDL):
- `looser.c`, `xesu.c`, `cdd.c` - Main variants
- `BlackDogKennel.c`, `mitre.c`, `03Singularity.ogg.599.c` - Experimental
- Multiple unnamed variants in sorel/

### Documentation:
- `foodotrope-drug.profile` - Drug regime configuration
- Various `.profile` files - Evolution parameters
- `great data.txt` - Cosmological philosophy
- READMEs in both main projects

---

## 7. Profound Insights

This isn't just another artificial life simulator. This is:

1. **Psychedelic computing** - First system to model drug effects on collective intelligence
2. **Morphic engineering** - Sheldrake's hypothesis made computational
3. **Emergence laboratory** - Democracy, culture, language from first principles
4. **Cosmological simulator** - Same rules at all scales
5. **Consciousness research** - Studying collective mind

The code reveals a **unified theory of intelligence**: 
From atoms to galaxies, the pattern is the same:
- Collect
- Combine
- Radiate
- Organize

CritterGOD must preserve and amplify these insights while making them accessible, performant, and extensible.

---

**Status**: Deep analysis ongoing. ~40% of codebase examined. Core concepts understood. Revolutionary potential confirmed.
