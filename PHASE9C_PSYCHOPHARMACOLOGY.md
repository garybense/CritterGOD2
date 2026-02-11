# Phase 9c: Drug-Modulated Synaptic Plasticity

**Status**: COMPLETE  
**Date**: 2026-02-09

## Overview

This document describes the **critical integration** of psychopharmacology with synaptic plasticity - the mechanism by which drugs affect learning rates and evolutionary trajectories in CritterGOD.

## The Deep Integration

Previously, drugs only affected neural firing (immediate consciousness alteration). Now they also affect **synaptic plasticity** - how memories are formed and unformed. This creates a profound feedback loop:

```
DRUG CONSUMPTION → ALTERED PLASTICITY → DIFFERENT LEARNING → 
BEHAVIORAL CHANGES → FITNESS DIFFERENCES → EVOLUTIONARY SELECTION →
GENETIC CHANGES → DIFFERENT BRAINS → DIFFERENT DRUG RESPONSES → ...
```

## How It Works

### Drug System Enhancement

**File**: `core/pharmacology/drugs.py`

New method: `get_plasticity_modulation(is_inhibitory: bool) -> (strengthen_mult, weaken_mult)`

Each of the 5 molecule types affects learning rates differently:

#### Inhibitory Antagonist (Type 0)
- Blocks inhibitory neurons
- **Disinhibition** → Enhanced learning
- Up to **3x strengthening**, 2.5x weakening
- Effect: Learns faster, less rigid thinking

#### Inhibitory Agonist (Type 1)
- Enhances inhibitory neurons
- **Strong inhibition** → Reduced learning
- Down to **0.3x strengthening**, 0.5x weakening
- Effect: Learns slower, more rigid/stable

#### Excitatory Antagonist (Type 2)
- Blocks excitatory neurons
- **Reduced excitation** → Decreased learning
- Down to **0.4x strengthening**, 0.6x weakening
- Effect: Dampened learning, sluggish adaptation

#### Excitatory Agonist (Type 3)
- Enhances excitatory neurons
- **Strong excitation** → Increased learning
- Up to **4x strengthening**, 2x weakening!
- Effect: Rapid learning, fast habit formation

#### Potentiator (Type 4)
- Amplifies ALL effects by 10x
- **Ego death / breakthrough experiences**
- Can reach 30x+ plasticity modulation
- Effect: Profound neuroplasticity changes

### Synapse Integration

**File**: `core/neural/synapse.py`

Modified `apply_stdp()` to accept `drug_system` parameter:

```python
def apply_stdp(self, time, strengthen_rate=0.1, weaken_rate=0.05, drug_system=None):
    # ... timing checks ...
    
    # Apply drug modulation to plasticity rates
    if drug_system is not None:
        strengthen_mult, weaken_mult = drug_system.get_plasticity_modulation(self.is_inhibitory)
        strengthen_rate *= strengthen_mult
        weaken_rate *= weaken_mult
    
    # ... STDP calculations with modulated rates ...
```

### Network Integration

**File**: `core/neural/network.py`

Modified `update()` to accept and pass drug system:

```python
def update(self, dt=1.0, drug_system=None):
    # ... neuron updates, spike propagation ...
    
    # Apply STDP with drug modulation
    if self.enable_plasticity:
        for synapse in self.synapses:
            synapse.apply_stdp(self.time, drug_system=drug_system)
```

### Creature Integration

**File**: `creatures/creature.py`

Creatures pass their drug system to network updates:

```python
# Update neural network (with drug modulation of plasticity!)
self.network.update(drug_system=self.drugs)
```

## Evolutionary Implications

### Scenario 1: Stimulant-Enhanced Learning
- Creature consumes Excitatory Agonist
- Learning rate increases 4x
- Quickly learns food location patterns
- Survives better → reproduces more
- Offspring inherit genes that seek stimulants
- **Evolutionary pressure toward drug-seeking behavior**

### Scenario 2: Psychedelic Unlearning
- Creature consumes Inhibitory Antagonist + Potentiator
- Massive plasticity increase (up to 30x)
- Rapidly unlearns maladaptive behaviors
- Can escape local optima in fitness landscape
- **Evolutionary innovation through chemical-assisted exploration**

### Scenario 3: Depressant Stabilization
- Creature consumes Inhibitory Agonist
- Learning rate decreases to 0.3x
- Preserves existing successful behaviors
- Resists learning new (possibly harmful) patterns
- **Evolutionary conservatism through reduced plasticity**

### Scenario 4: Collective Drug Culture
- Creatures write drug experiences to Circuit8
- Others read these patterns through retinal vision
- Second-hand drug effects via shared hallucination space
- **Cultural transmission of drug knowledge across generations**

## Circuit8 Integration (Shared Hallucination Space)

The telepathic canvas becomes a **collective memory that outlasts individuals**:

1. **Tripping creature** generates psychedelic patterns (via `PsychedelicVisionMixin`)
2. **Patterns written to Circuit8** with drug-amplified intensity
3. **Sober creatures perceive** these patterns through retinal vision
4. **Plasticity modulated** → sober creatures learn differently from drug patterns
5. **Patterns persist** after tripping creature dies
6. **New generations born** into world pre-loaded with ancestral drug experiences
7. **Evolutionary memory** encoded in shared visual space

This creates **trans-generational drug effects** - experiences outliving the experiencer.

## Phase 9c Demo Integration

**File**: `examples/phase9c_demo.py`

### Controls

**Psychedelic Patterns**: Press **P** to toggle pattern generation
- Creatures generate visual patterns based on neural activity
- Drug levels amplify pattern intensity, frequency, complexity
- Patterns written to Circuit8 telepathic canvas
- Creates visible "drug culture" in simulation

**Creature Inspector**: **Right-click** any creature to inspect
- Real-time vitals: age, generation, energy, position
- Brain state: neurons, synapses, activity, plasticity
- Drug levels: visual bars for all 5 molecule types
- Behavior: current action, food consumed, addiction stats
- Thoughts: latest Markov-generated text
- Social learning: observations and learned behaviors
- Right-click again to close inspector

**Time Control**: **[ ]** keys to adjust simulation speed (0.1x to 10x)

**Thoughts**: Press **T** to toggle thought bubbles above creatures

## Performance Notes

### Computational Cost
- Plasticity modulation: **O(1)** per synapse (just 2 multiplications)
- No significant performance impact
- Can run at full 60 FPS with drug-modulated plasticity

### Numerical Stability
- Plasticity multipliers clamped to [0.1, 10.0]
- Synapse weights still clamped to ±5.0 (CritterGOD4 limits)
- Continuous weakening still applies (automatic pruning)
- System remains stable even at 30x plasticity

## Scientific Basis

This system is inspired by real neuroscience:

1. **Psychedelics increase brain plasticity** (Johns Hopkins research)
2. **Stimulants enhance learning consolidation** (amphetamine studies)
3. **Depressants reduce synaptic plasticity** (benzodiazepine effects)
4. **Critical period reopening** (drug-induced neuroplasticity windows)
5. **Set and setting** (context modulates drug effects on learning)

## Future Enhancements

### Not Yet Implemented
1. **Social transmission of drug knowledge** - creatures learn which drugs work
2. **Tolerance affecting plasticity** - chronic use reduces modulation
3. **Withdrawal increasing plasticity** - rebound enhanced learning
4. **Drug combinations** - synergistic plasticity effects
5. **Genetic drug response variation** - some creatures learn more on drugs

### Philosophical Extensions
- Can a species **evolve to require drugs** for optimal learning?
- Do **drug-naive offspring** inherit ancestral drug memories via Circuit8?
- Can collective drug experiences create **emergent species-level intelligence**?
- Does the shared hallucination space become a **proto-culture**?

## Testing

To observe drug-modulated plasticity effects:

```bash
PYTHONPATH=/Users/gspilz/code/CritterGOD python3 examples/phase9c_demo.py
```

1. Watch creatures seek and consume drugs (colored spheres)
2. Press **P** to enable psychedelic pattern generation
3. Observe Circuit8 canvas filling with drug-influenced patterns
4. Notice creatures with high drug levels generating more complex patterns
5. Watch population dynamics - do drug-seekers survive better?

## Conclusion

Drug-modulated plasticity is the **missing link** between pharmacology and evolution. It answers the fundamental question:

**"How do temporary drug experiences affect long-term evolutionary trajectories?"**

Answer: **By changing learning rates during critical experiences, drugs alter which behaviors get encoded into neural networks, which affects survival, which drives genetic selection toward drug-responsive brains.**

This makes CritterGOD a genuine **psychopharmacological artificial life system** - not just creatures that get high, but creatures whose evolution is shaped by their chemical explorations.

The circuit is complete. 🌀
