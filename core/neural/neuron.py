"""
Neuron implementation with leaky integrate-and-fire dynamics.

Based on neural visualizers (looser.c, xesu.c, etc.) and telepathic-critterdrug.
"""

import numpy as np
from enum import Enum
from typing import Optional


class NeuronType(Enum):
    """Types of neurons in the network."""
    REGULAR = 0
    SENSORY = 1
    MOTOR = 2
    INHIBITORY = 3


class Neuron:
    """
    Leaky integrate-and-fire neuron with plasticity and spike-frequency adaptation.
    
    Inspired by SDL neural visualizers where:
    - brain[z][0] = current potential
    - brain[z][1] = firing threshold
    
    Supports bidirectional thresholds (from Critterding2):
    - Positive threshold: fires when potential >= threshold (excitation)
    - Negative threshold: fires when potential <= threshold (inhibition)
    
    Spike-frequency adaptation (from looser.c):
    - Threshold INCREASES by 10% when neuron fires (refractory period)
    - Threshold HALVES when neuron is silent and threshold > min
    - Creates self-regulating neurons: hyperactive ones slow down, silent ones wake up
    
    Attributes:
        neuron_id: Unique identifier
        potential: Current membrane potential
        threshold: Firing threshold (positive or negative)
        initial_threshold: Original threshold (for adaptation bounds)
        neuron_type: Type of neuron (regular, sensory, motor, inhibitory)
        is_plastic: Whether synapses can change strength
        last_fire_time: Last time this neuron fired (for STDP)
        leak_rate: Rate of potential decay (default 0.9)
    """
    
    # Spike-frequency adaptation parameters (from looser.c)
    ADAPTATION_INCREASE = 1.1    # Threshold multiplier on fire (10% increase)
    ADAPTATION_DECAY = 0.5       # Threshold multiplier when silent (halve)
    ADAPTATION_MIN_THRESHOLD = 15.0  # Don't decay below this (from looser.c: if brain[z][1]>15)
    
    def __init__(
        self,
        neuron_id: int,
        threshold: Optional[float] = None,
        neuron_type: NeuronType = NeuronType.REGULAR,
        is_plastic: bool = True,
        leak_rate: float = 0.9,
    ):
        """
        Initialize a neuron.
        
        Args:
            neuron_id: Unique identifier for this neuron
            threshold: Firing threshold (random 700-8700 if None, from looser.c)
            neuron_type: Type of neuron
            is_plastic: Whether this neuron's synapses can change
            leak_rate: Membrane potential leak rate (0-1)
        """
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type
        self.is_plastic = is_plastic
        self.leak_rate = leak_rate
        
        # Initialize potential and threshold
        # From looser.c: brain[z][0]=(rand()%5000); brain[z][1]=700+(rand()%8000)
        self.potential = np.random.uniform(0, 5000)
        self.threshold = threshold if threshold is not None else 700 + np.random.uniform(0, 8000)
        self.initial_threshold = self.threshold  # Store original for adaptation bounds
        
        # For STDP plasticity
        self.last_fire_time: Optional[float] = None
        
        # Track if neuron fired this timestep
        self._fired_this_step = False
        # Snapshot of previous step's fired state (available after network.update())
        self._fired_last_step = False
        
    def add_input(self, amount: float):
        """
        Add input to the neuron's potential.
        
        Args:
            amount: Amount to add (can be negative for inhibitory)
        """
        self.potential += amount
        
    def update(self, time: float) -> bool:
        """
        Update neuron state (leak, check for firing, adapt threshold).
        
        Supports bidirectional thresholds (Critterding2 innovation):
        - Positive threshold: fire when potential >= threshold
        - Negative threshold: fire when potential <= threshold
        
        Spike-frequency adaptation (from looser.c):
        - When neuron fires: threshold *= 1.1 (harder to fire again)
        - When neuron is silent: threshold /= 2 (becomes more excitable)
        - Self-regulating: hyperactive neurons slow down, silent ones wake up
        
        Args:
            time: Current simulation time
            
        Returns:
            True if neuron fired, False otherwise
        """
        # Apply leak (leaky integrate-and-fire)
        self.potential *= self.leak_rate
        
        # Bidirectional threshold check (Critterding2 feature)
        if self.threshold > 0:
            # Positive threshold: fire on excitation
            self._fired_this_step = self.potential >= self.threshold
            
            # Clamp negative accumulation
            if self.potential < 0:
                self.potential = 0.0
        elif self.threshold < 0:
            # Negative threshold: fire on inhibition
            self._fired_this_step = self.potential <= self.threshold
            
            # Clamp positive accumulation
            if self.potential > 0:
                self.potential = 0.0
        else:
            # Zero threshold: never fire (shouldn't happen normally)
            self._fired_this_step = False
        
        if self._fired_this_step:
            self.fire(time)
            # Spike-frequency adaptation: INCREASE threshold on fire
            # From looser.c: brain[z][1] *= 1.1
            # Makes neuron harder to fire again (refractory-like behavior)
            self.threshold *= self.ADAPTATION_INCREASE
        else:
            # Spike-frequency adaptation: DECAY threshold when silent
            # From looser.c: if (brain[z][1] > 15) brain[z][1] >>= 1
            # Makes silent neurons more excitable over time
            abs_threshold = abs(self.threshold)
            if abs_threshold > self.ADAPTATION_MIN_THRESHOLD:
                self.threshold *= self.ADAPTATION_DECAY
            
        return self._fired_this_step
        
    def fire(self, time: float):
        """
        Fire the neuron (reset potential and record time).
        
        Args:
            time: Current simulation time
        """
        self.potential = 0.0
        self.last_fire_time = time
        
    def did_fire(self) -> bool:
        """Check if neuron fired during the current integration step (before reset)."""
        return self._fired_this_step
        
    def fired_last_step(self) -> bool:
        """Check if neuron fired in the most recent completed step (after update)."""
        return self._fired_last_step
        
    def reset_fire_state(self):
        """Reset the fired state (called at end of timestep)."""
        # Preserve snapshot for external consumers after update
        self._fired_last_step = self._fired_this_step
        self._fired_this_step = False
        
    def is_inhibitory(self) -> bool:
        """Check if this is an inhibitory neuron."""
        return self.neuron_type == NeuronType.INHIBITORY
        
    def __repr__(self) -> str:
        return (
            f"Neuron(id={self.neuron_id}, type={self.neuron_type.name}, "
            f"potential={self.potential:.2f}, threshold={self.threshold:.2f})"
        )
