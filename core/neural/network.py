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
    
    Dynamic rewiring enables lifelong learning by:
    - Pruning weak synapses (automatic cleanup of unused connections)
    - Sprouting new synapses from active neurons (activity-dependent growth)
    
    Attributes:
        neurons: List of all neurons in the network
        synapses: List of all synapses
        time: Current simulation time
        enable_plasticity: Whether to apply STDP
        enable_rewiring: Whether to allow dynamic rewiring
    """
    
    # Rewiring parameters
    PRUNE_THRESHOLD = 0.1          # Remove synapses with |weight| below this
    SPROUT_FIRE_THRESHOLD = 3      # Neuron must fire this many times to sprout
    SPROUT_PROBABILITY = 0.01      # Probability of sprouting per eligible neuron
    MAX_SYNAPSES_PER_NEURON = 100  # Prevent runaway growth
    REWIRING_INTERVAL = 10         # Only rewire every N timesteps
    
    def __init__(self, enable_plasticity: bool = True, enable_rewiring: bool = True):
        """
        Initialize an empty neural network.
        
        Args:
            enable_plasticity: Enable STDP plasticity
            enable_rewiring: Enable dynamic rewiring (pruning + sprouting)
        """
        self.neurons: List[Neuron] = []
        self.synapses: List[Synapse] = []
        self.time: float = 0.0
        self.enable_plasticity = enable_plasticity
        self.enable_rewiring = enable_rewiring
        
        # Index neurons by ID for fast lookup
        self._neuron_by_id: Dict[int, Neuron] = {}
        
        # Track neuron activity for sprouting decisions
        self._fire_counts: Dict[int, int] = {}  # neuron_id -> fire count since last rewire
        self._outgoing_synapse_count: Dict[int, int] = {}  # neuron_id -> outgoing synapse count
        
    def add_neuron(self, neuron: Neuron):
        """Add a neuron to the network."""
        self.neurons.append(neuron)
        self._neuron_by_id[neuron.neuron_id] = neuron
        
    def add_synapse(self, synapse: Synapse):
        """Add a synapse to the network."""
        self.synapses.append(synapse)
        # Track outgoing synapse count
        pre_id = synapse.pre_neuron.neuron_id
        self._outgoing_synapse_count[pre_id] = self._outgoing_synapse_count.get(pre_id, 0) + 1
        
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
        
        # 5. Dynamic rewiring: prune weak synapses and sprout new ones
        if self.enable_rewiring:
            self._update_fire_counts()
            if int(self.time) % self.REWIRING_INTERVAL == 0:
                self._prune_weak_synapses()
                self._sprout_new_synapses()
                self._reset_fire_counts()
                
        # 6. Reset fire states for next timestep
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
        
    def _update_fire_counts(self):
        """Track which neurons fired for sprouting decisions."""
        for neuron in self.neurons:
            if neuron.did_fire():
                nid = neuron.neuron_id
                self._fire_counts[nid] = self._fire_counts.get(nid, 0) + 1
    
    def _reset_fire_counts(self):
        """Reset fire counts after rewiring."""
        self._fire_counts.clear()
    
    def _prune_weak_synapses(self):
        """
        Remove synapses with weights below threshold.
        
        This implements automatic network pruning - unused connections
        that have decayed through continuous weakening get removed,
        freeing up capacity for new, more useful connections.
        """
        original_count = len(self.synapses)
        
        # Filter out weak synapses
        surviving_synapses = []
        for synapse in self.synapses:
            if abs(synapse.weight) >= self.PRUNE_THRESHOLD:
                surviving_synapses.append(synapse)
            else:
                # Update outgoing synapse count
                pre_id = synapse.pre_neuron.neuron_id
                self._outgoing_synapse_count[pre_id] = max(0, 
                    self._outgoing_synapse_count.get(pre_id, 1) - 1)
        
        self.synapses = surviving_synapses
        
        pruned = original_count - len(self.synapses)
        return pruned
    
    def _sprout_new_synapses(self):
        """
        Create new synapses from highly active neurons.
        
        Activity-dependent growth: neurons that fire frequently
        are likely encoding important information and should have
        more connections to spread that information through the network.
        """
        if len(self.neurons) < 2:
            return 0
        
        sprouted = 0
        
        for neuron in self.neurons:
            nid = neuron.neuron_id
            fire_count = self._fire_counts.get(nid, 0)
            
            # Check if neuron is active enough to sprout
            if fire_count < self.SPROUT_FIRE_THRESHOLD:
                continue
            
            # Check if neuron already has max synapses
            current_synapses = self._outgoing_synapse_count.get(nid, 0)
            if current_synapses >= self.MAX_SYNAPSES_PER_NEURON:
                continue
            
            # Probabilistic sprouting (more active = more likely)
            sprout_prob = self.SPROUT_PROBABILITY * (fire_count / self.SPROUT_FIRE_THRESHOLD)
            if np.random.random() > sprout_prob:
                continue
            
            # Find a target neuron (prefer neurons this one doesn't already connect to)
            existing_targets = set()
            for synapse in self.synapses:
                if synapse.pre_neuron.neuron_id == nid:
                    existing_targets.add(synapse.post_neuron.neuron_id)
            
            # Get potential targets (not self, not already connected)
            potential_targets = [
                n for n in self.neurons 
                if n.neuron_id != nid and n.neuron_id not in existing_targets
            ]
            
            if not potential_targets:
                continue
            
            # Pick random target
            target = potential_targets[np.random.randint(0, len(potential_targets))]
            
            # Create new synapse with small initial weight
            # Inhibitory probability based on pre-neuron type
            is_inhib = neuron.is_inhibitory()
            
            new_synapse = Synapse(
                pre_neuron=neuron,
                post_neuron=target,
                weight=np.random.uniform(0.5, 2.0),  # Start weak, STDP will strengthen if useful
                is_inhibitory=is_inhib,
                plasticity_rate=0.01
            )
            self.add_synapse(new_synapse)
            sprouted += 1
        
        return sprouted
    
    def get_rewiring_stats(self) -> Dict:
        """Get statistics about network connectivity for debugging."""
        if not self.synapses:
            return {
                'total_synapses': 0,
                'avg_weight': 0.0,
                'weak_synapses': 0,
                'strong_synapses': 0
            }
        
        weights = [abs(s.weight) for s in self.synapses]
        return {
            'total_synapses': len(self.synapses),
            'avg_weight': np.mean(weights),
            'weak_synapses': sum(1 for w in weights if w < self.PRUNE_THRESHOLD * 2),
            'strong_synapses': sum(1 for w in weights if w > 1.0),
            'min_weight': min(weights),
            'max_weight': max(weights)
        }
    
    def __repr__(self) -> str:
        return (
            f"NeuralNetwork(neurons={len(self.neurons)}, "
            f"synapses={len(self.synapses)}, time={self.time:.1f})"
        )
