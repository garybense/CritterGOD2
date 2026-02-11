"""
Phase 5b Demo: Architectural Patterns from Critterding2

Demonstrates:
1. Entity introspection (to_dict() method for debugging)
2. Adam distance tracking (evolutionary lineage depth)
3. Evolutionary analysis tools
"""

import json
import numpy as np
from core.evolution.genotype import Genotype
from creatures.creature import Creature
from core.morphic.circuit8 import Circuit8


def demo_entity_introspection():
    """Demonstrate entity introspection for debugging."""
    print("=" * 70)
    print("DEMO 1: Entity Introspection (Critterding2 Debug Feature)")
    print("=" * 70)
    
    # Create a creature
    genotype = Genotype.create_random(
        n_sensory=5,
        n_motor=5,
        n_hidden_min=10,
        n_hidden_max=20,
        synapses_per_neuron=10
    )
    
    circuit8 = Circuit8(width=64, height=48)
    creature = Creature(
        genotype=genotype,
        x=32.0,
        y=24.0,
        initial_energy=10000.0,
        circuit8=circuit8,
        adam_distance=0
    )
    
    # Update a few times to generate activity
    for _ in range(5):
        creature.update()
    
    # Get state dictionary (without detailed network)
    state = creature.to_dict(include_network=False)
    
    print("\nCreature state (summary):")
    print(json.dumps(state, indent=2))
    
    print("\n" + "=" * 70)
    print("Result: Complete creature state serialized for debugging/analysis")
    print()


def demo_adam_distance_tracking():
    """Demonstrate adam distance tracking for evolutionary lineage."""
    print("=" * 70)
    print("DEMO 2: Adam Distance Tracking (Critterding2 Lineage)")
    print("=" * 70)
    
    # Create founding creature (adam_distance=0)
    genotype = Genotype.create_random(
        n_sensory=3,
        n_motor=3,
        n_hidden_min=5,
        n_hidden_max=10,
        synapses_per_neuron=5
    )
    
    circuit8 = Circuit8(width=64, height=48)
    founder = Creature(
        genotype=genotype,
        x=0.0,
        y=0.0,
        initial_energy=100000.0,
        circuit8=circuit8,
        adam_distance=0
    )
    
    print(f"Founding creature: {founder}")
    print(f"  Adam distance: {founder.adam_distance}")
    
    # Simulate evolution through several generations
    lineage = [founder]
    current = founder
    
    for gen in range(1, 6):
        # Give creature enough age to reproduce
        current.age = 200
        
        # Create offspring
        offspring = current.reproduce()
        if offspring:
            lineage.append(offspring)
            print(f"\nGeneration {gen}: {offspring}")
            print(f"  Adam distance: {offspring.adam_distance}")
            print(f"  Parent adam: {current.adam_distance}")
            current = offspring
    
    print("\n" + "-" * 70)
    print("Lineage summary:")
    for i, creature in enumerate(lineage):
        print(f"  Gen {i}: adam_distance={creature.adam_distance}, "
              f"neurons={len(creature.network.neurons)}, "
              f"synapses={len(creature.network.synapses)}")
    
    print("\n" + "=" * 70)
    print("Result: Adam distance tracks generational depth from founder")
    print("Useful for:")
    print("  - Measuring adaptation speed")
    print("  - Analyzing evolutionary dynamics")
    print("  - Tracking successful lineages")
    print()


def demo_evolutionary_analysis():
    """Demonstrate using introspection for evolutionary analysis."""
    print("=" * 70)
    print("DEMO 3: Evolutionary Analysis (Combined Features)")
    print("=" * 70)
    
    # Create a population
    circuit8 = Circuit8(width=64, height=48)
    population = []
    
    print("Creating population of 10 creatures...")
    for i in range(10):
        genotype = Genotype.create_random(
            n_sensory=3,
            n_motor=3,
            n_hidden_min=5,
            n_hidden_max=15,
            synapses_per_neuron=8
        )
        creature = Creature(
            genotype=genotype,
            x=np.random.uniform(0, 64),
            y=np.random.uniform(0, 48),
            initial_energy=10000.0,
            circuit8=circuit8,
            adam_distance=0  # All founders
        )
        population.append(creature)
    
    # Simulate for a while
    print("\nSimulating 50 timesteps...")
    for t in range(50):
        for creature in population:
            creature.update()
    
    # Analyze population using introspection
    print("\nPopulation analysis:")
    print("-" * 70)
    
    # Collect statistics
    stats = {
        'ages': [],
        'energies': [],
        'activities': [],
        'neuron_counts': [],
        'synapse_counts': [],
    }
    
    for creature in population:
        state = creature.to_dict()
        stats['ages'].append(state['age'])
        stats['energies'].append(state['energy'])
        stats['activities'].append(state['network_activity'])
        stats['neuron_counts'].append(state['num_neurons'])
        stats['synapse_counts'].append(state['num_synapses'])
    
    print(f"Population size: {len(population)}")
    print(f"Average age: {np.mean(stats['ages']):.1f}")
    print(f"Average energy: {np.mean(stats['energies']):.1f}")
    print(f"Average activity: {np.mean(stats['activities']):.2%}")
    print(f"Average neurons: {np.mean(stats['neuron_counts']):.1f}")
    print(f"Average synapses: {np.mean(stats['synapse_counts']):.1f}")
    
    # Find most active creature
    max_activity_idx = np.argmax(stats['activities'])
    most_active = population[max_activity_idx]
    most_active_state = most_active.to_dict()
    
    print("\nMost active creature:")
    print(f"  Activity: {most_active_state['network_activity']:.2%}")
    print(f"  Neurons: {most_active_state['num_neurons']}")
    print(f"  Synapses: {most_active_state['num_synapses']}")
    print(f"  Energy: {most_active_state['energy']:.1f}")
    
    print("\n" + "=" * 70)
    print("Result: Entity introspection enables population-level analysis")
    print()


def demo_detailed_network_inspection():
    """Demonstrate detailed network inspection."""
    print("=" * 70)
    print("DEMO 4: Detailed Network Inspection")
    print("=" * 70)
    
    # Create small creature for inspection
    genotype = Genotype.create_random(
        n_sensory=2,
        n_motor=2,
        n_hidden_min=3,
        n_hidden_max=5,
        synapses_per_neuron=3
    )
    
    creature = Creature(
        genotype=genotype,
        x=0.0,
        y=0.0,
        initial_energy=10000.0,
        adam_distance=0
    )
    
    # Update to generate activity
    for _ in range(3):
        creature.update()
    
    # Get detailed network state
    print("\nFetching detailed network state (include_network=True)...")
    state = creature.to_dict(include_network=True)
    
    print(f"\nNetwork has {len(state['neurons'])} neurons:")
    print("-" * 70)
    
    # Show first few neurons
    for i, neuron in enumerate(state['neurons'][:5]):
        print(f"  Neuron {neuron['id']}: "
              f"type={neuron['type']}, "
              f"potential={neuron['potential']:.1f}, "
              f"threshold={neuron['threshold']:.1f}, "
              f"fired={neuron['fired']}")
    
    if len(state['neurons']) > 5:
        print(f"  ... and {len(state['neurons']) - 5} more neurons")
    
    print(f"\nNetwork has {len(state['synapses'])} synapses:")
    print("-" * 70)
    
    # Show first few synapses
    for i, synapse in enumerate(state['synapses'][:5]):
        print(f"  Synapse {i}: "
              f"{synapse['pre_id']} -> {synapse['post_id']}, "
              f"weight={synapse['weight']:.2f}, "
              f"inhibitory={synapse['is_inhibitory']}")
    
    if len(state['synapses']) > 5:
        print(f"  ... and {len(state['synapses']) - 5} more synapses")
    
    print("\n" + "=" * 70)
    print("Result: Complete network state available for visualization/debugging")
    print("Use cases:")
    print("  - Neural network visualizer")
    print("  - Debugging network dynamics")
    print("  - Saving/loading creatures")
    print("  - Network topology analysis")
    print()


def demo_save_creature_state():
    """Demonstrate saving creature state to JSON."""
    print("=" * 70)
    print("DEMO 5: Save/Load Creature State (Serialization)")
    print("=" * 70)
    
    # Create creature
    genotype = Genotype.create_random(
        n_sensory=3,
        n_motor=3,
        n_hidden_min=5,
        n_hidden_max=10,
        synapses_per_neuron=5
    )
    
    creature = Creature(
        genotype=genotype,
        x=10.0,
        y=20.0,
        initial_energy=50000.0,
        adam_distance=5  # 5th generation
    )
    
    # Simulate
    for _ in range(10):
        creature.update()
    
    # Save to JSON
    state = creature.to_dict(include_network=True)
    json_str = json.dumps(state, indent=2)
    
    print(f"\nCreature state serialized to JSON ({len(json_str)} chars)")
    print("\nSample of JSON output:")
    print("-" * 70)
    
    # Show first 500 chars
    print(json_str[:500] + "...")
    
    print("\n" + "=" * 70)
    print("Result: Complete creature state serializable to JSON")
    print("Enables:")
    print("  - Save/load creature populations")
    print("  - Share interesting creatures")
    print("  - Checkpoint evolutionary runs")
    print("  - Export for external analysis")
    print()


def main():
    """Run all Phase 5b demos."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Phase 5b: Architectural Patterns from Critterding2".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    print("Integrating:")
    print("  - Entity introspection (to_dict() for debugging)")
    print("  - Adam distance tracking (evolutionary lineage)")
    print("  - Serialization support (save/load)")
    print()
    
    # Run demos
    demo_entity_introspection()
    demo_adam_distance_tracking()
    demo_evolutionary_analysis()
    demo_detailed_network_inspection()
    demo_save_creature_state()
    
    print("=" * 70)
    print("Phase 5b implementation complete! ✓")
    print("=" * 70)
    print()
    print("Next steps (Phase 5c):")
    print("  - Force-directed creature layouts (genetic similarity clustering)")
    print("  - Neural network visualizer (3D connectivity graph)")
    print("  - Bullet Physics integration (3D bodies)")
    print()


if __name__ == "__main__":
    main()
