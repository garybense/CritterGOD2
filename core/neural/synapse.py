"""
Synapse implementation with bidirectional connections and plasticity.

Based on SDL visualizers and telepathic-critterdrug neural architecture.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .neuron import Neuron


class Synapse:
    """
    Bidirectional synapse connecting two neurons with plasticity.
    
    Supports:
    - Excitatory and inhibitory connections
    - Hebbian plasticity (STDP - Spike-Timing-Dependent Plasticity)
    - Dynamic weight adjustment
    - Continuous weakening (automatic pruning from CritterGOD4)
    - Weight clamping (prevents runaway from CritterGOD4)
    
    Attributes:
        pre_neuron: Pre-synaptic neuron
        post_neuron: Post-synaptic neuron
        weight: Connection strength (positive = excitatory, negative = inhibitory)
        is_inhibitory: Whether this is an inhibitory synapse
        plasticity_rate: Rate of synaptic change (0 = no plasticity)
        max_weight: Maximum absolute weight value (±5.0 from CritterGOD4)
        min_weight: Minimum absolute weight value
        weakening_factor: Continuous decay factor (0.99 = 1% decay per timestep)
    """
    
    # Weight limits from CritterGOD4 (±5.0 to prevent runaway)
    MAX_WEIGHT_CLAMP = 5.0
    MIN_WEIGHT_CLAMP = -5.0
    
    # Continuous weakening factor from CritterGOD4 (automatic pruning)
    WEAKENING_FACTOR = 0.99  # 1% decay per timestep
    
    def __init__(
        self,
        pre_neuron: "Neuron",
        post_neuron: "Neuron",
        weight: float = 1.0,
        is_inhibitory: bool = False,
        plasticity_rate: float = 0.01,
        max_weight: float = 1000.0,
        min_weight: float = 1.0,
        enable_continuous_weakening: bool = True,
    ):
        """
        Initialize a synapse.
        
        Args:
            pre_neuron: Pre-synaptic (source) neuron
            post_neuron: Post-synaptic (target) neuron
            weight: Initial weight (absolute value)
            is_inhibitory: Whether this synapse is inhibitory
            plasticity_rate: How fast weights change during plasticity
            max_weight: Maximum weight magnitude
            min_weight: Minimum weight magnitude
            enable_continuous_weakening: Enable automatic pruning (CritterGOD4 feature)
        """
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.is_inhibitory = is_inhibitory
        self.plasticity_rate = plasticity_rate
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.enable_continuous_weakening = enable_continuous_weakening
        
        # Set initial weight (negative if inhibitory)
        self.weight = -abs(weight) if is_inhibitory else abs(weight)
        
        # Apply weight clamping immediately
        self._clamp_weight()
        
    # Asymmetric bidirectional propagation ratio (from looser.c)
    # Forward: full weight, Backward: half weight
    # looser.c: brain[z][0] += df[target] << 1 (forward 2x)
    #           brain[target][0] += df[z]       (backward 1x)
    # We use forward=1.0, backward=0.5 for equivalent 2:1 ratio
    REVERSE_PROPAGATION_RATIO = 0.5
    
    def propagate(self):
        """
        Propagate signal from pre to post neuron (forward direction).
        
        If pre-neuron fired, add weighted input to post-neuron.
        """
        if self.pre_neuron.did_fire():
            self.post_neuron.add_input(self.weight)
    
    def propagate_reverse(self):
        """
        Propagate signal from post to pre neuron (backward direction).
        
        From looser.c lines 308-310: Synapses are bidirectional.
        Forward propagation is at full weight, backward at half weight.
        This asymmetry creates richer signal flow through the network.
        
        If post-neuron fired, add half-weighted input to pre-neuron.
        """
        if self.post_neuron.did_fire():
            self.pre_neuron.add_input(self.weight * self.REVERSE_PROPAGATION_RATIO)
    
    def update(self, dt: float = 1.0):
        """
        Update synapse (apply continuous weakening).
        
        From CritterGOD4: All synapses weaken every timestep (automatic pruning).
        Unused synapses decay to zero, keeping network optimized.
        
        Args:
            dt: Time delta (usually 1.0)
        """
        if self.enable_continuous_weakening:
            # Apply continuous weakening (1% decay per timestep)
            self.weight *= self.WEAKENING_FACTOR
            
            # Clamp to valid range
            self._clamp_weight()
    
    def _clamp_weight(self):
        """
        Clamp weight to valid range (prevents runaway excitation/inhibition).
        
        From CritterGOD4: Weights clamped to ±5.0 for stability.
        """
        self.weight = max(self.MIN_WEIGHT_CLAMP, min(self.MAX_WEIGHT_CLAMP, self.weight))
            
    def apply_stdp(self, time: float, strengthen_rate: float = 0.1, weaken_rate: float = 0.05, drug_system=None):
        """
        Apply Spike-Timing-Dependent Plasticity (STDP).
        
        Hebbian learning: "Neurons that fire together, wire together"
        
        Rules:
        - If pre fires before post: strengthen connection
        - If post fires before pre: weaken connection
        - Magnitude depends on timing difference
        
        CRITICAL: Drug system can modulate plasticity rates!
        This is how psychedelic experiences affect learning and evolutionary trajectories.
        
        Args:
            time: Current simulation time
            strengthen_rate: Base rate of strengthening (from profile: brain_minplasticitystrengthen)
            weaken_rate: Base rate of weakening (from profile: brain_minplasticityweaken)
            drug_system: Optional drug system to modulate plasticity
        """
        # Only apply plasticity if both neurons are plastic
        if not (self.pre_neuron.is_plastic and self.post_neuron.is_plastic):
            return
            
        # Need both neurons to have fired at some point
        if self.pre_neuron.last_fire_time is None or self.post_neuron.last_fire_time is None:
            return
        
        # Apply drug modulation to plasticity rates if drug system available
        if drug_system is not None:
            strengthen_mult, weaken_mult = drug_system.get_plasticity_modulation(self.is_inhibitory)
            strengthen_rate *= strengthen_mult
            weaken_rate *= weaken_mult
            
        # Calculate time difference
        delta_t = self.post_neuron.last_fire_time - self.pre_neuron.last_fire_time
        
        # STDP window (typically ~20-40ms, using dimensionless time units)
        stdp_window = 40.0
        
        if abs(delta_t) < stdp_window:
            if delta_t > 0:
                # Pre fired before post -> strengthen (causal)
                change = strengthen_rate * np.exp(-abs(delta_t) / stdp_window)
                self.weight += change * self.plasticity_rate
            else:
                # Post fired before pre -> weaken (non-causal)
                change = weaken_rate * np.exp(-abs(delta_t) / stdp_window)
                self.weight -= change * self.plasticity_rate
                
            # Clamp weights to ±5.0 (CritterGOD4 limits)
            self._clamp_weight()
                
    def get_strength(self) -> float:
        """Get the absolute strength of this synapse."""
        return abs(self.weight)
        
    def __repr__(self) -> str:
        synapse_type = "inhibitory" if self.is_inhibitory else "excitatory"
        return (
            f"Synapse({synapse_type}, "
            f"{self.pre_neuron.neuron_id}->{self.post_neuron.neuron_id}, "
            f"weight={self.weight:.2f})"
        )
