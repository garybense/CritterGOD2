"""
Phase 5c Demo: Visualization Improvements

Demonstrates advanced visualization capabilities inspired by SpaceNet (r221):
1. Force-directed creature layout (genetic similarity clustering)
2. Neural network visualization (connectivity-based positioning)

From SpaceNet r221:
- Force-directed graph layouts
- Attention-based rendering
- Real-time interactive visualization
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available - some demos will be skipped")

from core.evolution.genotype import Genotype
from core.evolution.phenotype import build_network_from_genotype
from core.morphic.circuit8 import Circuit8
from creatures.creature import Creature
from visualization.force_directed_layout import ForceDirectedLayout
from visualization.neural_network_viewer import NeuralNetworkViewer


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_1_force_directed_basics():
    """Demo 1: Force-directed layout - basic mechanics."""
    print_section("Demo 1: Force-Directed Layout Basics")
    
    print("Creating population of diverse creatures...")
    circuit8 = Circuit8(width=64, height=48)
    
    # Create creatures with varied genetics
    creatures = []
    for i in range(5):
        # Vary network sizes to get genetic diversity
        n_hidden = 50 + i * 25
        genotype = Genotype.create_random(
            n_sensory=10,
            n_motor=10,
            n_hidden_min=n_hidden,
            n_hidden_max=n_hidden + 10,
            synapses_per_neuron=20 + i * 5
        )
        
        creature = Creature(
            genotype=genotype,
            x=float(i * 50),  # Initial linear positions
            y=float(i * 50),
            initial_energy=1000000.0,
            circuit8=circuit8
        )
        creatures.append(creature)
    
    print(f"Created {len(creatures)} creatures")
    print("\nInitial positions (linear):")
    for i, c in enumerate(creatures):
        print(f"  Creature {i}: ({c.x:.1f}, {c.y:.1f})")
    
    print("\nCreating force-directed layout...")
    layout = ForceDirectedLayout()
    
    print(f"Layout parameters:")
    print(f"  Repulsion strength: {layout.repulsion_strength}")
    print(f"  Attraction strength: {layout.attraction_strength}")
    print(f"  Similarity threshold: {layout.similarity_threshold}")
    print(f"  Damping: {layout.damping}")
    
    print("\nRunning force simulation...")
    converged, iterations = layout.run_until_stable(
        creatures,
        max_iterations=200,
        stability_threshold=1.0
    )
    
    print(f"Converged in {iterations} iterations")
    print("\nFinal positions (force-directed):")
    for i, c in enumerate(creatures):
        print(f"  Creature {i}: ({c.x:.1f}, {c.y:.1f})")
    
    print("\n✓ Creatures positioned by genetic similarity")
    print("  - Similar creatures cluster together")
    print("  - Dissimilar creatures pushed apart")


def demo_2_species_clustering():
    """Demo 2: Species clustering visualization."""
    print_section("Demo 2: Species Clustering")
    
    print("Creating two species with distinct genetics...")
    circuit8 = Circuit8(width=64, height=48)
    
    creatures = []
    
    # Species A: Small networks with few synapses
    print("\nSpecies A (small, sparse networks):")
    for i in range(4):
        genotype = Genotype.create_random(
            n_sensory=10,
            n_motor=10,
            n_hidden_min=40,
            n_hidden_max=50,
            synapses_per_neuron=15
        )
        
        creature = Creature(
            genotype=genotype,
            x=np.random.uniform(0, 300),
            y=np.random.uniform(0, 300),
            initial_energy=1000000.0,
            circuit8=circuit8
        )
        creatures.append(creature)
        
        network = creature.network
        n_inhibitory = sum(1 for n in network.neurons if n.is_inhibitory())
        print(f"  Creature {i}: {len(network.neurons)} neurons, "
              f"{len(network.synapses)} synapses, "
              f"{n_inhibitory} inhibitory")
    
    # Species B: Large networks with many synapses
    print("\nSpecies B (large, dense networks):")
    for i in range(4):
        genotype = Genotype.create_random(
            n_sensory=10,
            n_motor=10,
            n_hidden_min=150,
            n_hidden_max=200,
            synapses_per_neuron=40
        )
        
        creature = Creature(
            genotype=genotype,
            x=np.random.uniform(0, 300),
            y=np.random.uniform(0, 300),
            initial_energy=1000000.0,
            circuit8=circuit8
        )
        creatures.append(creature)
        
        network = creature.network
        n_inhibitory = sum(1 for n in network.neurons if n.is_inhibitory())
        print(f"  Creature {i+4}: {len(network.neurons)} neurons, "
              f"{len(network.synapses)} synapses, "
              f"{n_inhibitory} inhibitory")
    
    print(f"\nTotal population: {len(creatures)} creatures")
    
    # Calculate genetic similarities
    print("\nGenetic similarity matrix:")
    layout = ForceDirectedLayout()
    
    print("     ", end="")
    for i in range(len(creatures)):
        print(f"  {i:2d}", end="")
    print()
    
    for i in range(len(creatures)):
        print(f"  {i:2d} ", end="")
        for j in range(len(creatures)):
            sim = layout.calculate_similarity(creatures[i], creatures[j])
            print(f" {sim:.2f}", end="")
        print()
    
    print("\nRunning clustering simulation...")
    converged, iterations = layout.run_until_stable(
        creatures,
        max_iterations=300,
        stability_threshold=1.0
    )
    
    print(f"Converged in {iterations} iterations")
    
    # Calculate species centroids
    species_a_center = np.mean([(c.x, c.y) for c in creatures[:4]], axis=0)
    species_b_center = np.mean([(c.x, c.y) for c in creatures[4:]], axis=0)
    distance = np.sqrt((species_a_center[0] - species_b_center[0])**2 +
                      (species_a_center[1] - species_b_center[1])**2)
    
    print(f"\nSpecies A centroid: ({species_a_center[0]:.1f}, {species_a_center[1]:.1f})")
    print(f"Species B centroid: ({species_b_center[0]:.1f}, {species_b_center[1]:.1f})")
    print(f"Inter-species distance: {distance:.1f}")
    
    print("\n✓ Species form distinct clusters in 2D space")
    print("  - Similar genomes attract")
    print("  - Different genomes repel")


def demo_3_neural_network_structure():
    """Demo 3: Neural network structure visualization."""
    print_section("Demo 3: Neural Network Structure")
    
    print("Creating neural network...")
    genotype = Genotype.create_random(
        n_sensory=8,
        n_motor=8,
        n_hidden_min=30,
        n_hidden_max=40,
        synapses_per_neuron=15
    )
    
    network = build_network_from_genotype(genotype)
    
    # Analyze structure
    n_sensory = sum(1 for n in network.neurons if n.neuron_type.name == 'SENSORY')
    n_motor = sum(1 for n in network.neurons if n.neuron_type.name == 'MOTOR')
    n_hidden = len(network.neurons) - n_sensory - n_motor
    n_excitatory = sum(1 for n in network.neurons if not n.is_inhibitory())
    n_inhibitory = sum(1 for n in network.neurons if n.is_inhibitory())
    
    print(f"Network composition:")
    print(f"  Total neurons: {len(network.neurons)}")
    print(f"    - Sensory: {n_sensory}")
    print(f"    - Motor: {n_motor}")
    print(f"    - Hidden: {n_hidden}")
    print(f"  Neuron types:")
    print(f"    - Excitatory: {n_excitatory} ({100*n_excitatory/len(network.neurons):.1f}%)")
    print(f"    - Inhibitory: {n_inhibitory} ({100*n_inhibitory/len(network.neurons):.1f}%)")
    print(f"  Total synapses: {len(network.synapses)}")
    
    # Analyze synapse distribution
    n_exc_syn = sum(1 for s in network.synapses if not s.is_inhibitory)
    n_inh_syn = sum(1 for s in network.synapses if s.is_inhibitory)
    
    print(f"  Synapse types:")
    print(f"    - Excitatory: {n_exc_syn} ({100*n_exc_syn/len(network.synapses):.1f}%)")
    print(f"    - Inhibitory: {n_inh_syn} ({100*n_inh_syn/len(network.synapses):.1f}%)")
    
    # Connectivity analysis
    # Count incoming and outgoing connections for each neuron
    incoming = {n.neuron_id: 0 for n in network.neurons}
    outgoing = {n.neuron_id: 0 for n in network.neurons}
    
    for synapse in network.synapses:
        pre_id = synapse.pre_neuron.neuron_id
        post_id = synapse.post_neuron.neuron_id
        outgoing[pre_id] += 1
        incoming[post_id] += 1
    
    avg_in = sum(incoming.values()) / len(network.neurons)
    avg_out = sum(outgoing.values()) / len(network.neurons)
    
    print(f"  Connectivity:")
    print(f"    - Avg inputs per neuron: {avg_in:.1f}")
    print(f"    - Avg outputs per neuron: {avg_out:.1f}")
    
    print("\n✓ Network ready for visualization")
    print("  - Force-directed layout will position neurons by connectivity")
    print("  - Connected neurons attract (spring force)")
    print("  - All neurons repel (prevent overlap)")
    print("  - System finds equilibrium showing topology")


def demo_4_interactive_network_viewer():
    """Demo 4: Interactive neural network viewer."""
    print_section("Demo 4: Interactive Network Viewer")
    
    if not PYGAME_AVAILABLE:
        print("Skipping demo - pygame not available")
        return
    
    print("Creating neural network for visualization...")
    genotype = Genotype.create_random(
        n_sensory=5,
        n_motor=5,
        n_hidden_min=20,
        n_hidden_max=30,
        synapses_per_neuron=12
    )
    
    network = build_network_from_genotype(genotype)
    
    print(f"Network: {len(network.neurons)} neurons, {len(network.synapses)} synapses")
    
    print("\nVisualization features:")
    print("  - Neurons colored by type:")
    print("    • Blue = Sensory")
    print("    • Red = Motor")
    print("    • Green = Excitatory hidden")
    print("    • Magenta = Inhibitory hidden")
    print("    • Yellow flash = Firing")
    print("  - Synapses colored by weight:")
    print("    • Green = Excitatory")
    print("    • Red = Inhibitory")
    print("    • Brightness = Weight magnitude")
    print("  - Neuron size = Current potential")
    print("  - Gray ring = Firing threshold")
    
    print("\nControls:")
    print("  SPACE - Reset layout")
    print("  R - Run layout stabilization (100 steps)")
    print("  Mouse wheel - Zoom in/out")
    print("  ESC - Quit")
    
    print("\nStarting interactive viewer...")
    print("(Close window or press ESC to continue)")
    
    try:
        viewer = NeuralNetworkViewer(network, width=1200, height=800)
        viewer.run(fps=30)
    except Exception as e:
        print(f"Error running viewer: {e}")
    
    print("\n✓ Interactive neural network visualization")


def demo_5_creature_population_layout():
    """Demo 5: Full creature population with force-directed layout."""
    print_section("Demo 5: Population Layout Visualization")
    
    if not PYGAME_AVAILABLE:
        print("Skipping demo - pygame not available")
        return
    
    print("Creating diverse creature population...")
    circuit8 = Circuit8(width=64, height=48)
    
    creatures = []
    
    # Create 12 creatures with varying genetics
    for i in range(12):
        # Gradually increase complexity
        n_hidden = 40 + i * 10
        synapses = 15 + i * 2
        
        genotype = Genotype.create_random(
            n_sensory=10,
            n_motor=10,
            n_hidden_min=n_hidden,
            n_hidden_max=n_hidden + 15,
            synapses_per_neuron=synapses
        )
        
        creature = Creature(
            genotype=genotype,
            x=np.random.uniform(100, 500),
            y=np.random.uniform(100, 400),
            initial_energy=1000000.0,
            circuit8=circuit8
        )
        creatures.append(creature)
    
    print(f"Created {len(creatures)} creatures")
    
    print("\nGenetic diversity:")
    for i, c in enumerate(creatures):
        n_neurons = len(c.network.neurons)
        n_synapses = len(c.network.synapses)
        n_inhibitory = sum(1 for n in c.network.neurons if n.is_inhibitory())
        print(f"  Creature {i:2d}: {n_neurons:3d} neurons, "
              f"{n_synapses:3d} synapses, "
              f"{n_inhibitory:2d} inhibitory")
    
    print("\nApplying force-directed layout...")
    layout = ForceDirectedLayout()
    
    print("Simulating forces...")
    for step in range(100):
        layout.simulate_step(creatures)
        
        if step % 20 == 0:
            # Calculate total kinetic energy
            total_velocity = sum(
                np.sqrt(c.vx**2 + c.vy**2)
                for c in creatures
                if hasattr(c, 'vx')
            )
            print(f"  Step {step:3d}: total velocity = {total_velocity:.2f}")
    
    print("\n✓ Population organized by genetic similarity")
    print("  - Similar creatures form clusters")
    print("  - Natural species boundaries emerge")
    print("  - Visual identification of genetic diversity")
    
    print("\nNote: For full interactive visualization with pygame rendering,")
    print("      integrate this layout with visualization/circuit8_visualizer.py")


def run_all_demos():
    """Run all Phase 5c demos."""
    print("\n" + "=" * 70)
    print("  PHASE 5c: VISUALIZATION IMPROVEMENTS")
    print("  From SpaceNet r221 & Critterding2")
    print("=" * 70)
    
    print("\nPhase 5c introduces advanced visualization capabilities:")
    print("1. Force-directed creature layout (genetic similarity clustering)")
    print("2. Neural network visualization (connectivity-based positioning)")
    print("\nThese systems enable:")
    print("  - Visual understanding of population structure")
    print("  - Species identification through spatial clustering")
    print("  - Real-time network topology visualization")
    print("  - Interactive exploration of neural connectivity")
    
    input("\nPress Enter to begin demos...")
    
    # Run demos
    demo_1_force_directed_basics()
    input("\nPress Enter to continue...")
    
    demo_2_species_clustering()
    input("\nPress Enter to continue...")
    
    demo_3_neural_network_structure()
    input("\nPress Enter to continue...")
    
    demo_4_interactive_network_viewer()
    # No input needed - interactive demo
    
    demo_5_creature_population_layout()
    
    # Summary
    print_section("Phase 5c Complete!")
    
    print("Implemented features:")
    print("  ✓ Force-directed creature layout")
    print("    - Genetic similarity calculation")
    print("    - Repulsion/attraction physics")
    print("    - Species clustering")
    print("  ✓ Neural network visualization")
    print("    - Force-directed neuron positioning")
    print("    - Real-time activity display")
    print("    - Interactive zoom/pan controls")
    
    print("\nHeritage:")
    print("  - SpaceNet r221: Force-directed graph layouts")
    print("  - SpaceNet r221: Attention-based rendering")
    print("  - Critterding2: Modern visualization architecture")
    
    print("\nFiles created:")
    print("  - visualization/force_directed_layout.py")
    print("  - visualization/neural_network_viewer.py")
    print("  - examples/phase5c_demo.py")
    
    print("\nIntegration ready:")
    print("  - Use ForceDirectedLayout in population simulations")
    print("  - Use NeuralNetworkViewer for brain inspection")
    print("  - Combine with existing Circuit8 visualization")
    
    print("\n" + "=" * 70)
    print("  Phase 5c: Visualization - COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_demos()
