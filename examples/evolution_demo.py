#!/usr/bin/env python3
"""
Evolution Demo - Watch Intelligence Emerge

Demonstrates neural networks evolving to solve a task.
This is where we see what Flamoot saw: sentience emerging from simple rules.

Based on critterding's evolutionary system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.evolution import Genotype, Population
from core.evolution.phenotype import build_network_from_genotype
from core.neural import NeuralNetwork
from core.neural.neuron import NeuronType


def create_pattern_recognition_task():
    """
    Create a simple task: recognize and respond to sensory patterns.
    
    This demonstrates the universal cycle:
    Collection (sensory input) → Combination (neural processing) → 
    Radiation (motor output) → Organization (learning)
    """
    
    # Target patterns to recognize
    patterns = [
        np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),  # Alternating
        np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),  # Block
        np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),  # Sparse
    ]
    
    def fitness_function(network: NeuralNetwork) -> float:
        """
        Evaluate how well the network recognizes patterns.
        
        Fitness = Energy gained from correct responses.
        Based on critterding's energy-based fitness.
        """
        total_fitness = 0.0
        
        # Get sensory and motor neuron IDs
        sensory_neurons = [n for n in network.neurons if n.neuron_type == NeuronType.SENSORY]
        motor_neurons = [n for n in network.neurons if n.neuron_type == NeuronType.MOTOR]
        
        if not sensory_neurons or not motor_neurons:
            return 0.0
            
        # Test each pattern
        for pattern_idx, pattern in enumerate(patterns):
            # Reset network
            for neuron in network.neurons:
                neuron.potential = 0.0
                neuron.last_fire_time = None
                
            # Present pattern over multiple timesteps
            for step in range(20):
                # Inject pattern into sensory neurons
                for i, sensory in enumerate(sensory_neurons[:len(pattern)]):
                    if pattern[i] > 0:
                        network.inject_sensory_input(sensory.neuron_id, 5000.0)
                        
                # Update network
                network.update(dt=1.0)
                
                # Check motor response (should fire for recognized pattern)
                motor_outputs = network.get_motor_outputs()
                n_firing = sum(motor_outputs.values())
                
                # Reward: Different patterns should trigger different responses
                if step > 10:  # After network has time to process
                    # Energy reward for activity (shows network is responsive)
                    total_fitness += n_firing * 10.0
                    
                    # Bonus for consistent response to same pattern
                    if pattern_idx == 0 and n_firing >= 2:
                        total_fitness += 50.0
                    elif pattern_idx == 1 and n_firing >= 3:
                        total_fitness += 50.0
                    elif pattern_idx == 2 and n_firing >= 1:
                        total_fitness += 50.0
                        
        # Penalty for too many neurons (energy cost)
        total_fitness -= len(network.neurons) * 0.1
        
        return max(0.0, total_fitness)
        
    return fitness_function


def run_evolution(n_generations=50, population_size=30):
    """
    Run evolutionary simulation.
    
    Watch as intelligence emerges from random neural networks.
    """
    print("=" * 70)
    print("EVOLUTION DEMO - Emergence of Intelligence")
    print("Continuing Flamoot's Vision")
    print("=" * 70)
    print()
    print("Creating random population...")
    
    # Create population
    population = Population(
        population_size=population_size,
        n_sensory=10,
        n_motor=5
    )
    
    # Create fitness function
    fitness_function = create_pattern_recognition_task()
    
    print(f"Population: {population_size} individuals")
    print(f"Task: Pattern recognition (sensory-motor mapping)")
    print(f"Generations: {n_generations}")
    print()
    print("Beginning evolution...")
    print("-" * 70)
    
    best_ever_fitness = 0.0
    best_ever_generation = 0
    
    for gen in range(n_generations):
        print(f"\n=== Generation {gen + 1}/{n_generations} ===")
        
        # Evaluate fitness
        print("Evaluating fitness...")
        population.evaluate_fitness(fitness_function, verbose=True)
        
        # Get statistics
        stats = population.get_statistics()
        
        # Track best
        if stats['best_fitness'] > best_ever_fitness:
            best_ever_fitness = stats['best_fitness']
            best_ever_generation = gen + 1
            
        # Print generation summary
        print(f"\n  Best fitness: {stats['best_fitness']:.1f}")
        print(f"  Avg fitness:  {stats['avg_fitness']:.1f}")
        print(f"  Avg neurons:  {stats['avg_neurons']:.1f}")
        print(f"  Avg synapses: {stats['avg_synapses']:.1f}")
        print(f"  Diversity:    {stats['diversity']:.1f}")
        
        # Selection and reproduction
        if gen < n_generations - 1:
            print("  Selecting and reproducing...")
            population.select_and_reproduce(
                selection_pressure=0.5,
                mutation_rate=0.5,
                elitism=2
            )
            
    # Final summary
    print()
    print("=" * 70)
    print("EVOLUTION COMPLETE")
    print("=" * 70)
    print(f"\nBest fitness achieved: {best_ever_fitness:.1f}")
    print(f"In generation: {best_ever_generation}")
    print()
    
    # Analyze best individual
    best = population.get_best()
    print("Best Individual:")
    print(f"  Neurons: {len(best.neuron_genes)}")
    print(f"  Synapses: {len(best.synapse_genes)}")
    print(f"  Fitness: {best.fitness:.1f}")
    print(f"  Generation: {best.generation}")
    
    # Show fitness progression
    print("\nFitness Progression:")
    print("  Gen | Best | Avg")
    print("  " + "-" * 30)
    for i, (best, avg) in enumerate(zip(
        population.best_fitness_history,
        population.avg_fitness_history
    )):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  {i+1:3d} | {best:4.0f} | {avg:4.0f}")
            
    print()
    print("=" * 70)
    print("From randomness, intelligence emerged.")
    print("This is what Flamoot saw.")
    print("=" * 70)


if __name__ == "__main__":
    run_evolution(n_generations=50, population_size=30)
