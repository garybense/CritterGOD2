#!/usr/bin/env python3
"""
Simple Neural Network Example

Demonstrates basic usage of the CritterGOD neural engine with a
sensory-motor loop inspired by SDL visualizers.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.neural import Neuron, Synapse, NeuralNetwork
from core.neural.neuron import NeuronType
import numpy as np


def create_simple_network(n_neurons=100, synapses_per_neuron=10):
    """
    Create a simple neural network with sensory and motor neurons.
    
    Args:
        n_neurons: Total number of neurons
        synapses_per_neuron: Average synapses per neuron
        
    Returns:
        NeuralNetwork instance
    """
    print(f"Creating network with {n_neurons} neurons...")
    
    network = NeuralNetwork(enable_plasticity=True)
    
    # Create sensory neurons (first 10%)
    n_sensory = max(5, n_neurons // 10)
    for i in range(n_sensory):
        neuron = Neuron(
            neuron_id=i,
            neuron_type=NeuronType.SENSORY,
            is_plastic=True
        )
        network.add_neuron(neuron)
        
    # Create motor neurons (last 10%)
    n_motor = max(5, n_neurons // 10)
    motor_start = n_neurons - n_motor
    for i in range(motor_start, n_neurons):
        neuron = Neuron(
            neuron_id=i,
            neuron_type=NeuronType.MOTOR,
            is_plastic=True
        )
        network.add_neuron(neuron)
        
    # Create regular neurons (middle 80%)
    for i in range(n_sensory, motor_start):
        # 14% inhibitory neurons (from profile config)
        is_inhib = np.random.random() < 0.14
        neuron_type = NeuronType.INHIBITORY if is_inhib else NeuronType.REGULAR
        
        neuron = Neuron(
            neuron_id=i,
            neuron_type=neuron_type,
            is_plastic=True
        )
        network.add_neuron(neuron)
        
    # Create random synaptic connections
    print(f"Creating synapses (~{synapses_per_neuron} per neuron)...")
    network.create_random_synapses(
        synapses_per_neuron=synapses_per_neuron,
        inhibitory_prob=0.21,  # From profile config
        plasticity_rate=0.01
    )
    
    print(f"Network created: {len(network.neurons)} neurons, {len(network.synapses)} synapses")
    return network


def run_simulation(network, n_steps=1000, sensory_input_prob=0.1):
    """
    Run the neural network simulation.
    
    Args:
        network: NeuralNetwork instance
        n_steps: Number of simulation steps
        sensory_input_prob: Probability of random sensory input each step
    """
    print(f"\nRunning simulation for {n_steps} steps...")
    
    # Get sensory neuron IDs
    sensory_ids = [
        n.neuron_id for n in network.neurons
        if n.neuron_type == NeuronType.SENSORY
    ]
    
    # Track activity
    activity_history = []
    motor_firings = []
    
    for step in range(n_steps):
        # Random sensory input (like SDL visualizers' retinal input)
        if np.random.random() < sensory_input_prob:
            for sens_id in sensory_ids:
                if np.random.random() < 0.3:  # 30% of sensors fire
                    network.inject_sensory_input(sens_id, np.random.uniform(1000, 5000))
                    
        # Update network
        network.update(dt=1.0)
        
        # Record activity
        activity = network.get_activity_level()
        activity_history.append(activity)
        
        # Record motor output
        motor_outputs = network.get_motor_outputs()
        n_motor_firing = sum(motor_outputs.values())
        motor_firings.append(n_motor_firing)
        
        # Print progress
        if (step + 1) % 100 == 0:
            avg_activity = np.mean(activity_history[-100:])
            avg_motor = np.mean(motor_firings[-100:])
            print(f"Step {step + 1}: avg activity={avg_activity:.3f}, avg motor firing={avg_motor:.1f}")
            
    # Summary statistics
    print(f"\n=== Simulation Summary ===")
    print(f"Total steps: {n_steps}")
    print(f"Average activity: {np.mean(activity_history):.3f}")
    print(f"Peak activity: {np.max(activity_history):.3f}")
    print(f"Average motor firings: {np.mean(motor_firings):.1f}")
    print(f"Total spikes: {sum(len(network.get_firing_neurons()) for _ in range(10))}")
    

def main():
    """Main entry point."""
    print("=" * 60)
    print("CritterGOD Neural Engine - Simple Example")
    print("=" * 60)
    
    # Create network
    network = create_simple_network(n_neurons=200, synapses_per_neuron=40)
    
    # Run simulation
    run_simulation(network, n_steps=1000, sensory_input_prob=0.15)
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)
    

if __name__ == "__main__":
    main()
