# CritterGOD Source Material - Complete Analysis

## Overview

This document catalogs ALL critical code from the `/Users/gspilz/code/critters/` directory (~12,000 files analyzed). This is the source material that CritterGOD synthesizes.

---

## 1. ROOT-LEVEL GENERATIVE SYSTEMS (Python)

### 1.1 markov-attract-repel.py (905 lines)
**Revolutionary genetic text generation system**

**Core Algorithm**:
- Markov chain text generation with evolutionary word-pair selection
- **Attract/Repel dynamics**: Word pairs have scores that evolve over time
- `wordpairstartenergy = 1000.0`
- `startAttrep = 300.0` (time since occurrence measure)
- `attrepHit = 200.0` (cost per word pair occurrence)
- `wordscorethreshold = 1500` (breed threshold)

**Evolutionary Mechanics**:
- When `score > threshold`: **BREED** - mutate line, wire into markov cloud
- When `score < 0.1`: **KILL** - remove from markov table recursively
- All scoring values decay/increase over time
- Rewards novelty (pairs that haven't occurred recently)

**Mutation Operators**:
1. Letter substitution (vowel→vowel, consonant→consonant)
2. Letter increment/decrement through alphabet
3. Random character injection (alphanumeric)
4. Letter transposition (1 in 70 words)
5. Word-level transposition (1 in 100 lines)

**Important Parameters**:
- `skipwords` = ['of','that','this','by','and','or','to','the','a','on','them']
- `safewordpairs` = [('.','.'),('.','\\n'),('?','\\n'),('!','\\n')]
- `textodds = 7` (mutation rate: 1 in 7 letters when breeding)

**Usage**: `cat text.txt | ./markov-attract-repel.py [-showkills] [-showbreeds] > output.txt`

### 1.2 text-fuzz.py (63 lines)
**Simple text mutation engine** - same operators as markov without genetic algorithm

**Operators**:
- Same vowel/consonant substitution as markov
- Letter increment/decrement
- Random alphanumeric injection
- `odds = 9` (1 in 9 letters mutated)

### 1.3 numerolit.py (456 lines)
**Numerical mysticism - transform text via base conversion**

**Methods**:
1. **Method 1**: Play words by numerical value (all 0s, then 1s, then 2s...)
2. **Method 2**: Incremental rotating selection from each word rank
3. **Method 3**: Positional-Fibonacci word value chaining (addition)
4. **Method 4**: Same as 3 but multiplication (treats 0s as 1s)

**Process**:
1. Map words to base-36 digits (sum of letter values)
2. Reduce to single digit through iterative summing/multiplying
3. Reorder words by their numerical values
4. Options: `-scramble`, `-mult`, `-base N`, `-method N`

**Usage**: `cat text.txt | ./numerolit.py [-mult] [-scramble] [-base 36] [-method 3] > output.txt`

### 1.4 series.py (683 lines)
**Base conversion and digit manipulation toolkit**

**Commands**:
- `frombase N` - change base input values are treated as
- `tobase N` - change base output values shown as
- `countbase N` - change base computed sums shown as
- `reiterhi` / `reiterlo` - reiterate until all single digit
- `highest` / `lowest` / `random` - filter results

**Core Function** `doSums()`:
- Generate all possible add/subtract combinations of digits
- Support for multiply mode (`-mult`)
- `justSum = 1` for high sets only (faster, larger numbers)

### 1.5 map-retinal-sensors.py (389 lines)
**Visualization tool for critter neural sensor mappings**

**Sensor Types Mapped**:
- Retinal sensors: 50000-100000 (RGBA channels)
- Psychic screen (Circuit8): 100000-110000 
- P-screen write outputs: 50000-79990
- Intensity/colour motors: 79990-80000
- Energy bar: 30000-30010
- Age bar: 40000-40010
- Can procreate sensor: 20000
- Touch sensors: 10000-10002

**Output**: ASCII grid showing sensor/motor neuron connections with:
- Numbers = neuron count per sensor
- Capital letters = neurons with motor outputs

---

## 2. SDL NEURAL VISUALIZERS (C)

All variants share core architecture with different parameter sets:

### Common Pattern
```c
#define nn 10000-65000  // number of neurons
#define ns 40-180       // synapses per neuron
#define nf 45-150       // "fingers" (sensory-motor loops)

Uint16* brain[nn];      // Each neuron has:
// brain[z][0] = current potential
// brain[z][1] = firing threshold
// brain[z][2 to ns-1] = synapse connections
```

**Core Loop**:
1. Read retinal input from procedural wallpaper
2. Process synapses bidirectionally
3. When `potential >= threshold`: fire, reset, optionally rewire
4. Write audio buffer from neural state
5. Update wallpaper from neuron threshold ratios

### 2.1 looser.c (SDL2)
- **19,000 neurons, 40 synapses, 100 fingers**
- zoom=2
- 50 sensor neurons per finger reading screen column
- **Intentional red field lag** for temporal dynamics (500x501 loop)
- Wallpaper parameters from specific neuron ratios

### 2.2 xesu.c (SDL3)
- **14,000 neurons, 40 synapses, 100 fingers**
- zoom=3
- SDL3 API (newer)
- Simplified wiring pattern
- Different sensor gain (bit shifts instead of multiplication)
- Reduced intentional lag (50x51 loop)

### 2.3 cdd.c (SDL2)
- **13,000 neurons, 80 synapses, 60 fingers**
- Dynamic zoom: `zoom=4+(time(NULL)%4)>>1`
- Dynamic parameter calculation from `time()`
- More chaotic initialization
- Commented-out intentional lag code

### 2.4 BlackDogKennel.c (SDL2)
- **10,487 neurons, 70 synapses, 80 fingers**
- zoom=4
- Simplified tan-based wiring
- Mobile-focused: `/sdcard/Download/` path
- Simplest wiring pattern

### 2.5 mitre.c (SDL2)
- **18,000 neurons, 90 synapses, 150 fingers**
- Most complex parameter set
- Mystery parameters: `pur`, `rin`, `uri`, `ine`, `pui` (affect wiring)
- Extreme wiring complexity with log/tan feedback
- Dynamic zoom: `zoom=2+(time(NULL)%10)>>1`

### 2.6 03Singularity.ogg.599.c (SDL2)
- **65,000 NEURONS** (largest!)
- **180 synapses** per neuron
- **45 fingers**
- Simpler wiring pattern to handle scale
- Different `permutn` calculation: `x*x/(((float)(y)*(y)/299.0f))`

### Critical Innovation: Neural-Driven Generative Art

Wallpaper patterns computed from ratios of specific neuron thresholds:
```c
float tanamp = (45.0 * brain[1591][1] / brain[1425][1]);
float sinamp = (45.0 * brain[1573][1] / brain[1950][1]);
float cosamp = (55.0 * brain[1257][1] / brain[1775][1]);
float logamp = (65.0 * brain[1023][1] / brain[1200][1]);
// Similar for divisors...

px[x][y-1] = sin(permutn/sindiv)*sinamp 
           + tan(permutn/tandiv)*tanamp
           + cos(permutn/cosdiv)*cosamp  
           + log(permutn/logdiv)*logamp;
```

This creates **emergent aesthetic feedback loops**: neural network affects visual environment which feeds back through retinal sensors.

---

## 3. TELEPATHIC-CRITTERDRUG (C++)

Core revolutionary artificial life system with:
- Circuit8 (telepathic canvas)
- Psychopharmacology (5 drug types)
- Collective voting
- Bullet Physics for bodies

**Directory Structure**:
```
telepathic-critterdrug/
├── src/
│   ├── brainz/          # Neural network implementation
│   ├── scenes/          # Simulation scenes and modes
│   │   ├── modes/       # Different world types
│   │   ├── gui/         # Interface components
│   │   └── entities/    # Critters, food, pills
│   └── utils/           # Utilities (Bullet, FTGL)
├── ocr/                 # Webcam-based OCR evolution
└── hal/                 # HAL 9000 system (placeholder)
```

### Key Files to Analyze Further:
- `src/brainz/brainz.cpp` - Main neural network
- `src/brainz/neuroninterz.cpp` - Neuron processing
- `src/brainz/synapse.cpp` - Synapse mechanics
- `src/scenes/entities/critter.cpp` - Creature implementation
- `src/scenes/modes/race.cpp` - Evolution modes
- Look for Circuit8 implementation (64x48 pixel morphic field)
- Look for drug system (`tripping[5]` array)
- Look for voting system

---

## 4. CRITTERDING-BETA14 (C++)

Stable base version WITHOUT telepathy/drugs. Cleaner codebase with:
- Extensive configuration options
- Well-documented parameters
- Profile system for different evolutionary regimes
- OpenCL GPU acceleration attempts
- Better performance optimization

**Directory Structure**:
```
critterding-beta14/
├── src/              # Source code
├── profiles/         # Configuration profiles
├── share/            # Resources
└── CMakeLists.txt    # Build system
```

### Comparison to Telepathic Version:
- No Circuit8 / morphic fields
- No drug simulation  
- No collective voting
- Cleaner, more maintainable code
- Better documented
- Active development (2013 release)

---

## 5. PROFILE CONFIGURATION

### foodotrope-drug.profile (Example)
**Critical parameters** (119 lines):

**Neural Network**:
- `brain_maxneurons 250000`
- `brain_maxneuronsatbuildtime 20000`
- `brain_minneuronsatbuildtime 1000`
- `brain_maxsynapses 160`
- `brain_maxsynapsesatbuildtime 5`
- `brain_maxdendridicbranches 500`
- `brain_maxfiringthreshold 1001`
- `brain_minfiringthreshold 258`

**Energy Costs**:
- `brain_costfiringmotorneuron 100`
- `brain_costfiringneuron 5`
- `brain_costhavingneuron 10`
- `brain_costhavingsynapse 3`

**Plasticity**:
- `brain_maxplasticitystrengthen 20000`
- `brain_maxplasticityweaken 100000`
- `brain_minplasticitystrengthen 3`
- `brain_minplasticityweaken 10`

**Probabilities**:
- `brain_percentchanceconsistentsynapses 15`
- `brain_percentchanceinhibitoryneuron 14`
- `brain_percentchanceinhibitorysynapses 21`
- `brain_percentchancemotorneuron 30`
- `brain_percentchanceplasticneuron 50`
- `brain_percentchancesensorysynapse 50`

**Body**:
- `body_maxbodyparts 126`
- `body_maxbodypartsatbuildtime 110`
- `body_maxbodypartsize 250`
- `body_minbodypartsize 10`
- `body_mutationrate 50`

**Critter**:
- `critter_maxenergy 11000`
- `critter_startenergy 4999`
- `critter_minenergyproc 5000` (min energy to procreate)
- `critter_maxlifetime 600000`
- `critter_retinasize 16`
- `critter_sightrange 600`

**Drug System** (Telepathic version only):
- `auto_excit_agon_amt 2500`
- `auto_excit_antag_amt 500`
- `auto_inhib_agon_amt 2600`
- `auto_inhib_antag_amt 300`
- `auto_potentiator_amt 20000`
- `auto_*_every 1363500` (auto-drug interval)

**World**:
- `worldsizeX 25`
- `worldsizeY 94`
- `food_size 260`
- `pill_size 30` (drugs)
- `pill_energylevel 33000`
- `pill_maxtrip 200000`

**Circuit8** (Telepathic version):
- `condensed_colourmotors 1` (enable condensed RGB ops)
- Not all Circuit8 params visible in this profile - need to check code

---

## 6. CRITICAL FEATURES TO IMPLEMENT

### From Telepathic-Critterdrug:

**Circuit8 - Telepathic Canvas**:
- 64x48 pixel shared screen
- 1024 depth layers per pixel (3D temporal buffer)
- RGB channels = morphic field registers
- Motor neurons: moreRed, lessRed, moreGreen, lessGreen, moreBlue, lessBlue
- Creatures vote on screen movement/scrolling
- Acts as collective unconscious

**Psychopharmacology**:
- 5 molecule types:
  - 0: Inhibitory Antagonist
  - 1: Inhibitory Agonist
  - 2: Excitatory Antagonist
  - 3: Excitatory Agonist
  - 4: Potentiator (10x amplifier)
- `tripping[5]` array per creature
- Drug effects modify neural potential
- Energy costs for having vs. firing neurons

**Morphic Field Mechanics**:
- Each neuron assigned RuRdGuGdBuBd value (0-5)
- Determines which morphic channel affects it:
  - 0: Red Up
  - 1: Red Down
  - 2: Green Up
  - 3: Green Down
  - 4: Blue Up
  - 5: Blue Down
- Neurons read from Circuit8 BEFORE processing synapses

**Collective Voting**:
- Screen movement (up/down/left/right)
- Screen scrolling (fast modes)
- Disperse/coalesce patterns
- Fire/pause/erif commands
- Multiple voting tiers

### From Neural Visualizers:

**Audio Synthesis**:
- Direct audio buffer writing from neural state
- WAV callback function
- Real-time generation

**Visual Feedback Loops**:
- Procedural wallpaper from neuron threshold ratios
- Retinal sensors read screen
- Creates emergent aesthetic cycles

**Dynamic Rewiring**:
- Random rewiring on neuron fire
- Bidirectional synapses
- Multiple wiring strategies (see different .c variants)

### From Markov System:

**Evolutionary Text**:
- Attract/repel dynamics
- Self-organizing corpus
- Breed/kill mechanisms
- Could be used for critter "language"

---

## 7. HIDDEN CODE SNIPPETS (To Search For)

Directories still to fully analyze:
- `/hill/` - Experimental variants
- `/merge/` - Integration patterns
- `/mingw/` - Windows-specific code
- `/toons/` - Unknown experiments

**Search Strategy**:
```bash
# Find all C/C++ not in libraries
find /Users/gspilz/code/critters -name "*.c" -o -name "*.cpp" | \
  grep -v "bullet\|ftgl\|gc-7.2alpha4\|autom4te"

# Find all standalone programs
find /Users/gspilz/code/critters -maxdepth 2 -name "*.c"

# Find profile configurations
find /Users/gspilz/code/critters -name "*.profile"
```

---

## 8. PHILOSOPHICAL FRAMEWORK

From `great data.txt` - **The Universal Cycle**:

1. **Collection** - Particles/agents attract and gather
2. **Combination** - Interaction creates exponential growth
3. **Heating** - Dense systems generate energy/radiation
4. **Radiation** - Too-large systems explode/radiate
5. **Cooling** - Expansion leads to cooperation/alignment
6. **Equilibrium** - Self-organization emerges

**Examples at All Scales**:
- **Atoms**: Too large → radioactive decay, chain reactions
- **Neurons**: Collect inputs → fire when threshold exceeded → synapses strengthen/weaken
- **Societies**: Cooperation → exponential knowledge growth → self-organization
- **Language**: Words explain each other → combinatorial explosion → compression

**Implementation Connections**:
- Neurons = atoms (collect→fire→radiate)
- Synaptic plasticity = heating/cooling dynamics
- Morphic fields = magnetic field loops
- Drug systems = energy injection/dampening
- Evolution = self-organization toward equilibrium
- Markov text = combinatorial explosion of meaning

---

## 9. NEXT STEPS FOR COMPLETE ANALYSIS

1. ✅ Root-level Python scripts
2. ✅ SDL neural visualizers (all variants)
3. ⏳ Read telepathic-critterdrug C++ source (brainz, scenes, entities)
4. ⏳ Read critterding-beta14 source (compare architecture)
5. ⏳ Analyze hill/ directory (experimental code)
6. ⏳ Analyze merge/ directory (integration attempts)
7. ⏳ Find ALL profile configurations
8. ⏳ Document complete Circuit8 implementation
9. ⏳ Document complete voting system
10. ⏳ Document complete drug pharmacology

---

## 10. REVOLUTIONARY ASPECTS

This isn't just artificial life. This is:

1. **Psychedelic computing** - First system modeling drug effects on collective intelligence
2. **Morphic engineering** - Sheldrake's hypothesis made computational
3. **Democratic AI** - Emergent voting and collective will from bottom-up
4. **Multi-scale patterns** - Same rules at all scales (atoms to galaxies)
5. **Evolutionary language** - Genetic algorithms on text itself
6. **Neural aesthetics** - Audio/visual synthesis from brain activity

The unified theory: From atoms to galaxies, the pattern is:
**Collect → Combine → Radiate → Organize**

CritterGOD must preserve and amplify these insights.
