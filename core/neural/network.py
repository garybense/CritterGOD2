"""
Neural Network implementation with execution engine.

Container for neurons and synapses with update loop inspired by SDL visualizers.
"""

import numpy as np
from typing import List, Optional, Dict
from .neuron import Neuron, NeuronType
from .synapse import Synapse


class NeuralNetwork:
    """
    Spiking neural network with plasticity and dynamic rewiring.
    
    Inspired by SDL visualizers' main loop structure and telepathic-critterdrug.
    
    Attributes:
        neurons: List of all neurons in the network
        synapses: List of all synapses
        time: Current simulation time
        enable_plasticity: Whether to apply STDP
        enable_rewiring: Whether to allow dynamic rewiring
    """
    
    def __init__(self, enable_plasticity: bool = True, enable_rewiring: bool = False):
        """
        Initialize an empty neural network.
        
        Args:
            enable_plasticity: Enable STDP plasticity
            enable_rewiring: Enable dynamic rewiring on neuron fire
        """
        self.neurons: List[Neuron] = []
        self.synapses: List[Synapse] = []
        self.time: float = 0.0
        self.enable_plasticity = enable_plasticity
        self.enable_rewiring = enable_rewiring
        
        # Index neurons by ID for fast lookup
        self._neuron_by_id: Dict[int, Neuron] = {}
        
    def add_neuron(self, neuron: Neuron):
        """Add a neuron to the network."""
        self.neurons.append(neuron)
        self._neuron_by_id[neuron.neuron_id] = neuron
        
    def add_synapse(self, synapse: Synapse):
        """Add a synapse to the network."""
        self.synapses.append(synapse)
        
    def create_random_synapses(
        self,
        synapses_per_neuron: int,
        inhibitory_prob: float = 0.2,
        plasticity_rate: float = 0.01
    ):
        """
        Create random synaptic connections.
        
        Inspired by SDL visualizers' wiring patterns.
        
        Args:
            synapses_per_neuron: Average synapses per neuron (ns from visualizers)
            inhibitory_prob: Probability of inhibitory synapse (from profile: 0.21)
            plasticity_rate: Rate of synaptic plasticity
        """
        n_neurons = len(self.neurons)
        
        for neuron in self.neurons:
            # Random number of synapses (from visualizers: ns=40-180)
            n_synapses = max(2, int(np.random.normal(synapses_per_neuron, synapses_per_neuron * 0.2)))
            
            for _ in range(n_synapses):
                # Random target neuron
                target_idx = np.random.randint(0, n_neurons)
                target = self.neurons[target_idx]
                
                # Skip self-connections
                if target.neuron_id == neuron.neuron_id:
                    continue
                    
                # Determine if inhibitory
                is_inhib = np.random.random() < inhibitory_prob
                
                # Create synapse
                synapse = Synapse(
                    pre_neuron=neuron,
                    post_neuron=target,
                    weight=np.random.uniform(50.0, 300.0),
                    is_inhibitory=is_inhib,
                    plasticity_rate=plasticity_rate
                )
                self.add_synapse(synapse)
                
    def update(self, dt: float = 1.0, drug_system=None):
        """
        Execute one timestep of the network.
        
        Based on SDL visualizers' main loop, enhanced with CritterGOD4 features:
        1. Update neurons (leak + fire check)
        2. Propagate spikes through synapses
        3. Apply continuous weakening (CritterGOD4 automatic pruning)
        4. Apply STDP plasticity (modulated by drugs if drug_system provided)
        5. Reset neuron fire states
        
        Args:
            dt: Time delta for this step
            drug_system: Optional DrugSystem for plasticity modulation
        """
        self.time += dt
        
        # 1. Update all neurons (check for firing)
        for neuron in self.neurons:
            neuron.update(self.time)
            
        # 2. Propagate spikes through all synapses
        for synapse in self.synapses:
            synapse.propagate()
            
        # 3. Apply continuous weakening (CritterGOD4: automatic pruning)
        for synapse in self.synapses:
            synapse.update(dt)
            
        # 4. Apply STDP plasticity if enabled (modulated by drugs!)
        if self.enable_plasticity:
            for synapse in self.synapses:
                synapse.apply_stdp(self.time, drug_system=drug_system)
                
        # 5. Reset fire states for next timestep
        for neuron in self.neurons:
            neuron.reset_fire_state()
            
    def get_firing_neurons(self) -> List[int]:
        """Get list of neuron IDs that fired in the most recent step."""
        return [n.neuron_id for n in self.neurons if n.fired_last_step()]
        
    def get_motor_outputs(self) -> Dict[int, bool]:
        """Get last-step firing state of all motor neurons."""
        return {
            n.neuron_id: n.fired_last_step()
            for n in self.neurons
            if n.neuron_type == NeuronType.MOTOR
        }
        
    def inject_sensory_input(self, neuron_id: int, amount: float):
        """
        Inject input into a sensory neuron.
        
        Args:
            neuron_id: ID of sensory neuron
            amount: Input amount
        """
        if neuron_id in self._neuron_by_id:
            neuron = self._neuron_by_id[neuron_id]
            if neuron.neuron_type == NeuronType.SENSORY:
                neuron.add_input(amount)
                
    def get_activity_level(self) -> float:
        """Get fraction of neurons that fired in the most recent step."""
        if not self.neurons:
            return 0.0
        firing = sum(1 for n in self.neurons if n.fired_last_step())
        return firing / len(self.neurons)
        
    def __repr__(self) -> str:
        return (
            f"NeuralNetwork(neurons={len(self.neurons)}, "
            f"synapses={len(self.synapses)}, time={self.time:.1f})"
        )
