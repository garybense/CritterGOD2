"""
Phase 5a Demo: Neural Refinements from Flamoot Discoveries

Demonstrates:
1. Continuous synapse weakening (CritterGOD4 automatic pruning)
2. Weight clamping (±5.0 limits)
3. Bidirectional thresholds (Critterding2 innovation)
4. Higher inhibitory ratio (30% vs previous 14%)
"""

import numpy as np
from core.neural.neuron import Neuron, NeuronType
from core.neural.synapse import Synapse
from core.neural.network import NeuralNetwork
from core.evolution.genotype import Genotype
from core.evolution.phenotype import build_network_from_genotype


def demo_continuous_weakening():
    """Demonstrate continuous synapse weakening (automatic pruning)."""
    print("=" * 70)
    print("DEMO 1: Continuous Weakening (CritterGOD4 Automatic Pruning)")
    print("=" * 70)
    
    # Create two neurons
    n1 = Neuron(neuron_id=0, threshold=1000.0)
    n2 = Neuron(neuron_id=1, threshold=1000.0)
    
    # Create synapse with continuous weakening enabled
    synapse = Synapse(n1, n2, weight=3.0, enable_continuous_weakening=True)
    
    print(f"Initial weight: {synapse.weight:.4f}")
    print(f"Weakening factor: {synapse.WEAKENING_FACTOR} (1% decay per timestep)")
    print()
    
    # Simulate 100 timesteps without STDP strengthening
    print("Simulating 100 timesteps with no activity (unused synapse):")
    for t in range(100):
        synapse.update(dt=1.0)
        if t in [0, 9, 49, 99]:
            print(f"  Timestep {t+1:3d}: weight = {synapse.weight:.4f}")
    
    print()
    print("Result: Unused synapses naturally decay toward zero (automatic pruning)")
    print()


def demo_weight_clamping():
    """Demonstrate weight clamping to prevent runaway."""
    print("=" * 70)
    print("DEMO 2: Weight Clamping (CritterGOD4 Stability)")
    print("=" * 70)
    
    # Create neurons
    n1 = Neuron(neuron_id=0, threshold=100.0)
    n2 = Neuron(neuron_id=1, threshold=100.0)
    
    # Create synapse with large initial weight
    synapse = Synapse(n1, n2, weight=10.0, enable_continuous_weakening=False)
    
    print(f"Initial weight: {synapse.weight:.2f}")
    print(f"Weight clamp limits: [{synapse.MIN_WEIGHT_CLAMP}, {synapse.MAX_WEIGHT_CLAMP}]")
    print()
    
    # Attempt to set weight beyond limits
    synapse.weight = 15.0
    synapse._clamp_weight()
    print(f"After setting to 15.0 and clamping: {synapse.weight:.2f}")
    
    synapse.weight = -8.0
    synapse._clamp_weight()
    print(f"After setting to -8.0 and clamping: {synapse.weight:.2f}")
    
    print()
    print("Result: Weights clamped to ±5.0 prevents runaway excitation/inhibition")
    print()


def demo_bidirectional_thresholds():
    """Demonstrate bidirectional thresholds (Critterding2 innovation)."""
    print("=" * 70)
    print("DEMO 3: Bidirectional Thresholds (Critterding2 Innovation)")
    print("=" * 70)
    
    # Create excitatory neuron (positive threshold)
    excitatory = Neuron(neuron_id=0, threshold=1000.0)
    excitatory.potential = 500.0
    
    # Create inhibitory neuron (negative threshold)
    inhibitory = Neuron(neuron_id=1, threshold=-1000.0)
    inhibitory.potential = -500.0
    
    print("Excitatory neuron (positive threshold):")
    print(f"  Threshold: {excitatory.threshold:.1f}")
    print(f"  Initial potential: {excitatory.potential:.1f}")
    
    # Add excitatory input
    excitatory.potential = 1200.0
    fired = excitatory.update(time=1.0)
    print(f"  After excitation (potential=1200): fired={fired}")
    
    print()
    print("Inhibitory neuron (negative threshold):")
    print(f"  Threshold: {inhibitory.threshold:.1f}")
    print(f"  Initial potential: {inhibitory.potential:.1f}")
    
    # Add inhibitory input
    inhibitory.potential = -1200.0
    fired = inhibitory.update(time=1.0)
    print(f"  After inhibition (potential=-1200): fired={fired}")
    
    print()
    print("Result: Neurons can specialize for excitation OR inhibition")
    print()


def demo_higher_inhibitory_ratio():
    """Demonstrate higher inhibitory ratio (30% from CritterGOD4)."""
    print("=" * 70)
    print("DEMO 4: Higher Inhibitory Ratio (30% from CritterGOD4)")
    print("=" * 70)
    
    # Create genotypes with old and new ratios
    print("Creating random genotype with new 30% inhibitory ratio...")
    genotype_new = Genotype.create_random(
        n_sensory=10,
        n_motor=10,
        n_hidden_min=50,
        n_hidden_max=100,
        synapses_per_neuron=40,
        inhibitory_neuron_prob=0.3,
        inhibitory_synapse_prob=0.3
    )
    
    # Count inhibitory neurons
    inhibitory_neurons = sum(1 for n in genotype_new.neuron_genes if n.is_inhibitory)
    total_neurons = len(genotype_new.neuron_genes)
    
    # Count inhibitory synapses
    inhibitory_synapses = sum(1 for s in genotype_new.synapse_genes if s.is_inhibitory)
    total_synapses = len(genotype_new.synapse_genes)
    
    print(f"\nGenotype statistics:")
    print(f"  Total neurons: {total_neurons}")
    print(f"  Inhibitory neurons: {inhibitory_neurons} ({100*inhibitory_neurons/total_neurons:.1f}%)")
    print(f"  Total synapses: {total_synapses}")
    print(f"  Inhibitory synapses: {inhibitory_synapses} ({100*inhibitory_synapses/total_synapses:.1f}%)")
    
    print()
    print("Result: ~30% inhibitory ratio (biologically accurate, from CritterGOD4)")
    print("Previous: 14% inhibitory neurons, 21% inhibitory synapses")
    print()


def demo_integrated_system():
    """Demonstrate all Phase 5a features working together."""
    print("=" * 70)
    print("DEMO 5: Integrated System (All Phase 5a Features)")
    print("=" * 70)
    
    # Create small network
    network = NeuralNetwork(enable_plasticity=True)
    
    # Create neurons with bidirectional thresholds
    n1 = Neuron(neuron_id=0, threshold=1000.0, neuron_type=NeuronType.SENSORY)
    n2 = Neuron(neuron_id=1, threshold=1000.0)
    n3 = Neuron(neuron_id=2, threshold=-1000.0)  # Inhibitory threshold
    n4 = Neuron(neuron_id=3, threshold=1000.0, neuron_type=NeuronType.MOTOR)
    
    network.add_neuron(n1)
    network.add_neuron(n2)
    network.add_neuron(n3)
    network.add_neuron(n4)
    
    # Create synapses with weight clamping and continuous weakening
    s1 = Synapse(n1, n2, weight=2.0)
    s2 = Synapse(n2, n3, weight=2.0, is_inhibitory=True)
    s3 = Synapse(n3, n4, weight=2.0)
    
    network.add_synapse(s1)
    network.add_synapse(s2)
    network.add_synapse(s3)
    
    print(f"Network: {len(network.neurons)} neurons, {len(network.synapses)} synapses")
    print(f"Initial synapse weights: [{s1.weight:.2f}, {s2.weight:.2f}, {s3.weight:.2f}]")
    print()
    
    # Run simulation
    print("Running 50 timesteps with occasional sensory input...")
    for t in range(50):
        # Inject sensory input every 10 steps
        if t % 10 == 0:
            network.inject_sensory_input(0, 2000.0)
        
        # Update network (includes continuous weakening)
        network.update(dt=1.0)
        
        if t in [0, 9, 24, 49]:
            activity = network.get_activity_level()
            weights = [s1.weight, s2.weight, s3.weight]
            print(f"  Step {t+1:2d}: activity={activity:.2%}, weights=[{weights[0]:5.2f}, {weights[1]:6.2f}, {weights[2]:5.2f}]")
    
    print()
    print("Result: All Phase 5a features working together:")
    print("  - Continuous weakening reduces unused synapse weights")
    print("  - Weight clamping prevents runaway (stays within ±5.0)")
    print("  - Bidirectional thresholds allow inhibitory neurons")
    print("  - STDP plasticity strengthens active connections")
    print()


def main():
    """Run all Phase 5a demos."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Phase 5a: Neural Refinements from Flamoot Discoveries".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    print("Integrating innovations from:")
    print("  - CritterGOD4 (2010): Continuous weakening, weight clamping, 40% inhibitory")
    print("  - Critterding2 (2015+): Bidirectional thresholds, entity-component system")
    print()
    
    # Run demos
    demo_continuous_weakening()
    demo_weight_clamping()
    demo_bidirectional_thresholds()
    demo_higher_inhibitory_ratio()
    demo_integrated_system()
    
    print("=" * 70)
    print("Phase 5a implementation complete! ✓")
    print("=" * 70)
    print()
    print("Next steps (Phase 5b):")
    print("  - Entity introspection system (to_dict() methods)")
    print("  - Adam distance tracking (generational lineage)")
    print("  - Reactive data binding (numpy broadcasting)")
    print()


if __name__ == "__main__":
    main()
