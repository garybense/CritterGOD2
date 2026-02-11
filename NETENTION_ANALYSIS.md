# Netention Analysis

**Discovery Date**: January 24, 2026  
**Location**: `/Users/gspilz/code/critters/netention-code-r50-proto3`  
**Version**: r50-proto3  
**Author**: "seh" (likely flamoot/same author as CritterGOD)  
**Language**: Java (NetBeans project)  
**Purpose**: Semantic web / knowledge representation system

## Overview

**Netention** is a semantic network system for representing and matching knowledge between distributed agents. It appears to be flamoot's **cognitive/symbolic layer** - a complementary system to CritterGOD's subsymbolic neural networks.

**Key Insight**: While CritterGOD handles **subsymbolic** intelligence (neural networks, emergence, pattern recognition), Netention handles **symbolic** intelligence (knowledge graphs, semantic matching, intentional reasoning).

## Core Architecture

### 1. Network (server/Network.java)
```java
/** an organization of agents and their states that can be associated or matched */ 
public interface Network {
    Map<String, Agent> getAgents();
    Map<String, Node> getNodes();
    Schema getSchema();
}
```

A **network** contains:
- **Agents** - autonomous entities with intentions/needs
- **Nodes** - semantic concepts (analogous to RDF resources)
- **Schema** - ontology defining patterns and properties

### 2. Nodes (server/Node.java)
```java
/** analogous to an RDF resource */
public class Node implements Value {
    private String id;
    private String name;
}
```

Nodes represent **concepts** or **entities** in the semantic web.

### 3. Patterns (server/Pattern.java)

**Patterns** define semantic templates:
```java
abstract public class Pattern extends Node {
    private List<String> ext;  // patterns this extends
    private List<String> props;  // properties defined
    
    // Fluent API for building patterns:
    Pattern withString(String id, String name)
    Pattern withReal(String id, String name, Unit unit)
    Pattern withNode(String id, String name, String patternRestriction)
    Pattern withGeoPoint(String id, String name)
    Pattern withTimePoint(String id, String name)
}
```

Example patterns from schema:
- **PhysicalObject** - has mass, length, width, height
- **Human** - has firstName, lastName, biography
- **Bicycle** - extends built, owned, located; has gearCount, wheelDiameter, bicycleType
- **Event** - extends located; has startTime, endTime

### 4. Agents (implied from schema1.groovy)

**Agents** have **Details** (specific instances of patterns) with **intentions**:

```groovy
Agent ann = network.newAgent("ann", "Ann"); 
    ann.newDetail('Myself', 'Human').
        with('firstName', new StringIs("Ann")).
        with('lastName', new StringIs("Onymous"));

    ann.newDetail('My Bike', 'Bicycle').
        with('gearCount', new IntegerIs(6)).
        with('owner', new NodeIs('ann')).
        with('ownerNext', new NodeNotEquals("ann"));  // Intention: wants to sell bike

Agent bob = network.newAgent("bob", "Bob"); 
    bob.newDetail('Bike I Want', 'Bicycle'). 
        with('gearCount', new RealMoreThan(4)).
        with('ownerNext', new NodeIs('bob')).  // Intention: wants to become owner
        with('owner', new NodeNotEquals('bob'));
```

**Ann has a bike** (with owner='ann', ownerNext≠'ann')  
**Bob wants a bike** (with owner≠'bob', ownerNext='bob')

### 5. Linker (server/linker/)

```java
/** a weaver is a process that semantically links stories in real-time */
public interface Linker {
    void updateNode(Detail n);
    DetailLink getLink(Detail a, Detail b);
}
```

The **Linker** automatically creates connections between compatible intentions:
- **DetailLink** - connection between two details with **strength** (0.0-1.0)
- **LinkReason** - why the link was created (which properties matched)

**Example**: Ann's bike detail matches Bob's bike-want detail → creates link with strength based on compatibility

## Key Concepts

### Intention Matching

The system represents **what agents want** (intentions) as constraints on future states:
- `ownerNext = NodeIs('bob')` - Bob intends to own something
- `ownerNext = NodeNotEquals('ann')` - Ann intends to not own something

The **Linker** matches these complementary intentions to suggest transactions/interactions.

### Semantic Compatibility

Values can be:
- **Definite** (`StringIs("mountain")`) - exact value
- **Constraint** (`RealMoreThan(4)`) - acceptable range
- **Variable** (unconstrained)

Matching finds overlap between constraints:
```
Ann's bike: gearCount=6
Bob's want: gearCount>4
→ Match! (6 > 4)
```

### Pattern Inheritance

Patterns can extend multiple parents:
```groovy
schema.newPattern("bicycle", "Bicycle").
    extending("built", "owned", "located")
```

Bicycle inherits all properties from:
- **built** - builder, serialID, builtWhen
- **owned** - owner, ownerNext
- **located** - location, locationNext

### Property Types

Rich type system:
- **String** - text values (can be "rich" HTML)
- **Real** - floating point with **units** (mass, distance, currency, etc.)
- **Integer** - whole numbers
- **GeoPoint** - lat/lon coordinates
- **TimePoint** - timestamps
- **Node** - references to other nodes/agents

## Connection to CritterGOD

### Complementary Systems

**CritterGOD** (Subsymbolic):
- Spiking neural networks
- Pattern recognition
- Emergence
- Continuous values
- Non-linguistic

**Netention** (Symbolic):
- Knowledge graphs
- Semantic reasoning
- Intentionality
- Discrete concepts
- Linguistic

### Potential Integration: Hybrid AI

**Vision**: CritterGOD creatures with Netention cognition

1. **Neural Substrate** (CritterGOD)
   - Low-level perception (retinal sensors, morphic field)
   - Motor control (movement, eating, procreation)
   - Pattern matching (audio, visual, text generation)

2. **Semantic Layer** (Netention)
   - High-level concepts ("food", "danger", "mate")
   - Intentions ("I want food", "I need energy")
   - Social contracts ("I'll trade this for that")
   - Communication (semantic messages between creatures)

3. **Bridge**
   - Neural → Semantic: Pattern recognition activates concepts
   - Semantic → Neural: Intentions modulate neural activity
   - Example: "hunger" concept → increase food-seeking neuron activity

### Real-World Analogy

**Human cognition**:
- **Basal ganglia/cerebellum** (subsymbolic) - motor patterns, reflexes
- **Cortex** (hybrid) - perceptual processing, recognition
- **Prefrontal cortex** (symbolic) - planning, language, reasoning

**CritterGOD + Netention**:
- **Neural network** (subsymbolic) - sensorimotor processing
- **Morphic field** (hybrid) - collective pattern memory
- **Netention layer** (symbolic) - intentions, communication, trade

## Architectural Patterns Worth Adopting

### 1. Fluent Builder API

Netention's pattern definition syntax is elegant:
```groovy
schema.newPattern("bicycle", "Bicycle").
    extending("built", "owned", "located").
    withInt("gearCount", "Gear Count").
    withReal("wheelDiameter", "Wheel Diameter", Unit.Distance).
    withString("bicycleType", "Bicycle Type", ['mountain', 'street'])
```

**Apply to CritterGOD**:
```python
# Fluent creature configuration
creature = (EnhancedCreature(genotype, circuit8)
    .with_audio(mode='mixed')
    .with_text(seed_text=language)
    .with_vision(resolution=100)
    .with_energy(initial=1000000))
```

### 2. Constraint-Based Matching

Netention's value constraints are powerful:
- `RealMoreThan(4)` - simple inequality
- `RealBetween(3, 7)` - range
- `StringContains("x")` - partial match
- `NodeNotEquals("self")` - exclusion

**Apply to CritterGOD**:
```python
# Creature preferences/intentions
creature.add_preference(
    "food", 
    location=NearBy(distance < 10),
    energy_value=MoreThan(100)
)
```

### 3. Link Strength

DetailLink has **strength** (0.0-1.0) indicating compatibility:
```java
public DetailLink(String fromNode, String toNode, double strength)
```

**Apply to CritterGOD**:
- Synapse weights are already 0.0-1.0 (or ±5.0 with clamping)
- Could add "relationship strength" between creatures
- Social bonds based on interaction history

### 4. Real-Time Linking/Weaving

The **Linker** continuously updates connections as details change:
```java
void updateNode(Detail n);
DetailLink getLink(Detail a, Detail b);
```

**Apply to CritterGOD**:
- Dynamic synapse formation based on co-activation
- Circuit8 as "semantic canvas" where creatures post/read intentions
- Automatic social network formation

## Implementation Recommendations

### Phase 5b: Symbolic Layer (Optional - Advanced)

**If** we want to add symbolic reasoning to CritterGOD:

1. **Concept Nodes** - High-level abstractions
   ```python
   class Concept:
       id: str
       name: str
       activation: float  # From neural activity
   ```

2. **Creature Intentions** - What creature wants
   ```python
   class Intention:
       concept: str  # "food", "mate", "safety"
       constraint: Constraint
       priority: float
   ```

3. **Semantic Matching** - Find compatible intentions
   ```python
   def match_intentions(creature_a, creature_b) -> float:
       # Returns compatibility score 0.0-1.0
       # Based on complementary needs
   ```

4. **Communication Protocol** - Creatures exchange semantic messages
   ```python
   # Instead of raw Circuit8 pixels, creatures write structured data
   creature.post_message(
       concept="food",
       location=GeoPoint(x, y),
       expires=TimePoint(now + 100)
   )
   ```

### Minimal Integration (Practical)

**Simpler approach** - Learn from Netention's design patterns:

1. ✅ **Fluent APIs** for creature configuration (easy)
2. ✅ **Link strength** for social bonds (medium)
3. ✅ **Constraint matching** for creature preferences (medium)
4. ⚠️ **Full semantic layer** (complex, Phase 6+)

## Relationship to Other Discoveries

**Timeline Context**:
- **CritterGOD4** (2010) - Neural substrate (C++/Bullet)
- **Netention** (r50-proto3, ~2010-2011) - Symbolic layer (Java)
- **telepathic-critterdrug** (2015+) - Morphic field integration
- **SDL visualizers** (2020+) - Audio/visual generation
- **Current CritterGOD** (2026) - Python synthesis

**Netention represents flamoot's parallel work on symbolic AI** - the "thinking mind" to CritterGOD's "feeling body".

## Conclusions

### What Netention Is

**Netention** is a **semantic web framework** for distributed agent intention matching. It's flamoot's exploration of:
- Symbolic reasoning
- Knowledge representation
- Social coordination
- Marketplace dynamics (trade, exchange)
- Collective intelligence through semantic compatibility

### What Netention Is NOT

- It's **not** a neural network system (that's CritterGOD)
- It's **not** directly related to creature evolution
- It's **not** a physics simulation

### Relevance to CritterGOD

**Direct relevance**: Low (different paradigm)

**Conceptual relevance**: High (complementary approaches)

**Practical value**:
1. ✅ Design patterns (fluent APIs, constraint matching)
2. ✅ Link strength concept (social bonds)
3. ⚠️ Full integration (future Phase 6: Hybrid Symbolic/Subsymbolic AI)

### Key Takeaway

**Netention shows flamoot was working on BOTH sides of AI**:
- **CritterGOD** = Subsymbolic, emergent, neural
- **Netention** = Symbolic, intentional, semantic

A **complete** flamoot-inspired AI would integrate both:
```
Neural Processing (CritterGOD)
    ↕ (grounding)
Semantic Reasoning (Netention)
```

This is the **holy grail** of AI - symbol grounding and hybrid architectures.

For now, CritterGOD focuses on the **subsymbolic layer** (which is already revolutionary). Netention patterns can inform future symbolic extensions.

---

**Status**: Analysis complete  
**Recommendation**: Adopt design patterns (fluent APIs, link strength) but defer full symbolic integration to future phases

**Next**: Analyze remaining discovered codebases
