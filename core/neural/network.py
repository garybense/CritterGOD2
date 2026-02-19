"""
Neural Network implementation with execution engine.

Container for neurons and synapses with update loop inspired by SDL visualizers.
"""

import math
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
    
    # Threshold-dependent rewiring (from looser.c line 254)
    # When a neuron fires, redirect one synapse to target = threshold % nn
    THRESHOLD_REWIRE_ENABLED = True
    
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
        
        # Outgoing synapse index: neuron_id -> list of synapses
        # Enables O(1) lookup for threshold rewiring and sprouting (was O(n_synapses))
        self._outgoing_synapses: Dict[int, List[Synapse]] = {}
        
        # Track neuron activity for sprouting decisions
        self._fire_counts: Dict[int, int] = {}  # neuron_id -> fire count since last rewire
        
    def add_neuron(self, neuron: Neuron):
        """Add a neuron to the network."""
        self.neurons.append(neuron)
        self._neuron_by_id[neuron.neuron_id] = neuron
        
    def add_synapse(self, synapse: Synapse):
        """Add a synapse to the network."""
        self.synapses.append(synapse)
        # Update outgoing synapse index
        pre_id = synapse.pre_neuron.neuron_id
        if pre_id not in self._outgoing_synapses:
            self._outgoing_synapses[pre_id] = []
        self._outgoing_synapses[pre_id].append(synapse)
        
    def create_random_synapses(
        self,
        synapses_per_neuron: int,
        inhibitory_prob: float = 0.2,
        plasticity_rate: float = 0.01
    ):
        """
        Create synaptic connections using mixed mathematical wiring strategies.
        
        From looser.c/xesu.c/cdd.c: Multiple wiring patterns create structured-yet-complex
        topology. Each synapse is wired using one of several strategies:
        - Random (uniform): traditional random wiring
        - Local (z/pur): nearby neurons connect (nearest-neighbor)
        - Stride (z*pur % nn): long-range periodic connections
        - Sequential (z+1, z-2): chain-like connections
        - Chaotic (tan(z*y) * fia): nonlinear mathematical wiring
        
        Mystery parameters pur, rin (randomized per network) create unique topologies.
        
        Args:
            synapses_per_neuron: Average synapses per neuron (ns from visualizers)
            inhibitory_prob: Probability of inhibitory synapse (from profile: 0.21)
            plasticity_rate: Rate of synaptic plasticity
        """
        n_neurons = len(self.neurons)
        if n_neurons < 2:
            return
        
        # Mystery parameters from looser.c: randomized per-network wiring topology
        # These create unique network structures each generation
        pur = np.random.randint(3, 13)  # From looser.c: (rand()%10)+4
        rin = np.random.randint(3, 13)  # From looser.c: (rand()%10)+4
        
        # Feedback parameters for chaotic wiring (from looser.c lines 85-87)
        fia = 202555.0  # Starter stock from looser.c
        fib = 202555.0
        fic = 202555.0
        
        for z, neuron in enumerate(self.neurons):
            # Random number of synapses (from visualizers: ns=40-180)
            n_synapses = max(2, int(np.random.normal(synapses_per_neuron, synapses_per_neuron * 0.2)))
            
            for yg in range(n_synapses):
                # Select wiring strategy (mixed, from looser.c lines 69-88)
                r = np.random.random()
                
                if r < 0.25:
                    # Random (traditional uniform random)
                    target_idx = np.random.randint(0, n_neurons)
                elif r < 0.40:
                    # Local: z/pur - nearest-neighbor wiring
                    # From looser.c: brain[z][yg] = z/pur
                    target_idx = (z // max(1, pur)) % n_neurons
                elif r < 0.55:
                    # Stride: z*pur % nn - long-range periodic
                    # From looser.c: brain[z][yg] = (z*pur) % nn
                    target_idx = (z * pur) % n_neurons
                elif r < 0.65:
                    # Sequential forward: z+1
                    # From looser.c: brain[z][yg] = (z+1) % nn
                    target_idx = (z + 1) % n_neurons
                elif r < 0.75:
                    # Sequential backward: z-2
                    # From looser.c: brain[z][yg] = (z-2) % nn
                    target_idx = (z - 2) % n_neurons
                else:
                    # Chaotic: tan-based nonlinear wiring
                    # From looser.c: brain[z][yg] = ((z<<4) + tan(z*yg)*fia) % nn
                    try:
                        tan_val = math.tan((z * (yg + 1)) * 0.001)  # Scale down to avoid overflow
                        target_idx = int(abs((z * 16) + tan_val * fia)) % n_neurons
                    except (ValueError, OverflowError):
                        target_idx = np.random.randint(0, n_neurons)
                
                # Evolve chaotic feedback parameters (from looser.c lines 85-87)
                # These create complex wiring that depends on position in the network
                try:
                    fic = float(n_neurons * 712.0 * math.log(max(abs(fia), 1e-10))) / max(float(synapses_per_neuron), 1.0)
                    fia = float(n_neurons * 885.0 * synapses_per_neuron) * math.tan(fib * 0.0001) * 0.001
                    fib = float(n_neurons * 299.0 * synapses_per_neuron) * math.tan(fic * 0.0001)
                except (ValueError, OverflowError):
                    fia = 202555.0  # Reset on overflow
                    fib = 202555.0
                    fic = 202555.0
                
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
            
        # 2. Propagate spikes through all synapses (with asymmetric bidirectional)
        # From looser.c lines 308-310:
        #   brain[z][0] += df[brain[z][y]] << 1  -- forward: 2x fire value
        #   brain[brain[z][y]][0] += df[z]        -- backward: 1x fire value
        for synapse in self.synapses:
            synapse.propagate()  # Forward propagation (full weight)
            synapse.propagate_reverse()  # Backward propagation (half weight)
            
        # 3. Apply continuous weakening (CritterGOD4: automatic pruning)
        for synapse in self.synapses:
            synapse.update(dt)
            
        # 4. Apply STDP plasticity if enabled (modulated by drugs!)
        if self.enable_plasticity:
            for synapse in self.synapses:
                synapse.apply_stdp(self.time, drug_system=drug_system)
        
        # 5. Threshold-dependent rewiring on fire (from looser.c line 254)
        # When a neuron fires, redirect one of its synapses to a target
        # determined by the neuron's current threshold value.
        # This couples threshold adaptation with network topology.
        if self.THRESHOLD_REWIRE_ENABLED and self.enable_rewiring:
            self._threshold_rewire_on_fire()
        
        # 6. Dynamic rewiring: prune weak synapses and sprout new ones
        if self.enable_rewiring:
            self._update_fire_counts()
            if int(self.time) % self.REWIRING_INTERVAL == 0:
                self._prune_weak_synapses()
                self._sprout_new_synapses()
                self._reset_fire_counts()
                
        # 7. Reset fire states for next timestep
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
        
    _rewire_cursor = 2  # Cycling synapse index for threshold rewiring (from looser.c)
    
    def _threshold_rewire_on_fire(self):
        """
        Threshold-dependent synapse redirection on fire.
        
        From looser.c line 254:
            brain[z][(rewire++)] = ((brain[z][1]) % nn)
        
        When a neuron fires, one of its outgoing synapses is redirected to
        a new target determined by the neuron's current threshold value.
        The rewire cursor cycles through synapse indices.
        
        Uses _outgoing_synapses index for O(1) lookup per neuron.
        """
        n_neurons = len(self.neurons)
        if n_neurons < 2:
            return
        
        for neuron in self.neurons:
            if not neuron.did_fire():
                continue
            
            # O(1) lookup via outgoing synapse index
            nid = neuron.neuron_id
            outgoing = self._outgoing_synapses.get(nid, [])
            if len(outgoing) < 3:  # Keep at least a couple stable connections
                continue
            
            # Select synapse to rewire (cycling cursor like looser.c)
            syn_idx = self._rewire_cursor % len(outgoing)
            synapse_to_rewire = outgoing[syn_idx]
            self._rewire_cursor += 1
            
            # New target: threshold % nn (from looser.c)
            new_target_idx = int(abs(neuron.threshold)) % n_neurons
            new_target = self.neurons[new_target_idx]
            
            # Don't rewire to self
            if new_target.neuron_id == neuron.neuron_id:
                continue
            
            # Redirect the synapse to new target
            synapse_to_rewire.post_neuron = new_target
    
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
                # Remove from outgoing synapse index
                pre_id = synapse.pre_neuron.neuron_id
                if pre_id in self._outgoing_synapses:
                    try:
                        self._outgoing_synapses[pre_id].remove(synapse)
                    except ValueError:
                        pass
        
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
            
            # Check if neuron already has max synapses (using indexed count)
            current_synapses = len(self._outgoing_synapses.get(nid, []))
            if current_synapses >= self.MAX_SYNAPSES_PER_NEURON:
                continue
            
            # Probabilistic sprouting (more active = more likely)
            sprout_prob = self.SPROUT_PROBABILITY * (fire_count / self.SPROUT_FIRE_THRESHOLD)
            if np.random.random() > sprout_prob:
                continue
            
            # Find a target neuron (prefer neurons this one doesn't already connect to)
            # Use outgoing synapse index for O(1) lookup
            existing_targets = set(
                s.post_neuron.neuron_id for s in self._outgoing_synapses.get(nid, [])
            )
            
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
