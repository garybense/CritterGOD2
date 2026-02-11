"""
Psychopharmacology System

5 molecule types that affect neural dynamics.
This is psychedelic computing - consciousness alteration in collective systems.

Based on telepathic-critterdrug's drug system.
"""

import numpy as np
from enum import IntEnum
from typing import List, Tuple
from dataclasses import dataclass


class MoleculeType(IntEnum):
    """
    The 5 molecule types from telepathic-critterdrug.
    
    These affect neural potential and plasticity:
    - 0: Inhibitory Antagonist (blocks inhibitory neurons)
    - 1: Inhibitory Agonist (enhances inhibitory neurons)
    - 2: Excitatory Antagonist (blocks excitatory neurons)
    - 3: Excitatory Agonist (enhances excitatory neurons)
    - 4: Potentiator (amplifies ALL drug effects 10x - ego death)
    """
    INHIBITORY_ANTAGONIST = 0  # Blocks inhibition (disinhibition)
    INHIBITORY_AGONIST = 1     # Enhances inhibition (calming)
    EXCITATORY_ANTAGONIST = 2  # Blocks excitation (dampening)
    EXCITATORY_AGONIST = 3     # Enhances excitation (stimulation)
    POTENTIATOR = 4            # Amplifies everything 10x


@dataclass
class Pill:
    """
    A pill containing molecules.
    
    From telepathic-critterdrug profile:
    - pill_size = 30
    - pill_energylevel = 33000 (energy gained from consuming)
    - pill_maxtrip = 200000 (max drug level)
    """
    x: float
    y: float
    molecule_composition: List[int]  # 5 values for 5 molecule types
    energy_value: float = 33000.0
    size: float = 30.0
    
    def __post_init__(self):
        """Ensure composition has 5 values."""
        if len(self.molecule_composition) != 5:
            self.molecule_composition = [0, 0, 0, 0, 0]


class DrugSystem:
    """
    Manage drug effects on a creature.
    
    The 'tripping' array tracks drug levels.
    Drugs modify neural potential during updates.
    
    From telepathic-critterdrug:
    - tripping[5] array per creature
    - auto_*_amt values (how much is auto-administered)
    - auto_*_every values (administration interval)
    """
    
    def __init__(self, max_trip: float = 200000.0):
        """
        Initialize drug system.
        
        Args:
            max_trip: Maximum drug level (from profile: pill_maxtrip)
        """
        # Current drug levels (5 molecule types)
        self.tripping = np.zeros(5, dtype=float)
        self.max_trip = max_trip
        
        # Decay rates (drugs wear off)
        self.decay_rate = 0.99  # 1% decay per timestep
        
    def consume_pill(self, pill: Pill) -> float:
        """
        Consume a pill, adding molecules to system.
        
        Args:
            pill: Pill to consume
            
        Returns:
            Energy gained
        """
        for i in range(5):
            self.tripping[i] += pill.molecule_composition[i]
            # Clamp to max
            self.tripping[i] = min(self.max_trip, self.tripping[i])
            
        return pill.energy_value
        
    def apply_drug_effect(
        self,
        neuron_potential: float,
        is_inhibitory: bool,
        is_firing: bool
    ) -> float:
        """
        Apply drug effects to neuron potential.
        
        This is where consciousness alteration happens.
        
        Args:
            neuron_potential: Current potential
            is_inhibitory: Whether neuron is inhibitory
            is_firing: Whether neuron is currently firing
            
        Returns:
            Modified potential
        """
        # Get potentiator amplification (molecule type 4)
        potentiator_amp = 1.0 + (self.tripping[MoleculeType.POTENTIATOR] / self.max_trip) * 10.0
        
        modified_potential = neuron_potential
        
        if is_inhibitory:
            # Inhibitory Antagonist: blocks inhibitory neurons
            antagonist_effect = self.tripping[MoleculeType.INHIBITORY_ANTAGONIST]
            antagonist_effect *= potentiator_amp
            modified_potential -= antagonist_effect * 0.01
            
            # Inhibitory Agonist: enhances inhibitory neurons
            agonist_effect = self.tripping[MoleculeType.INHIBITORY_AGONIST]
            agonist_effect *= potentiator_amp
            if is_firing:
                modified_potential += agonist_effect * 0.01
        else:
            # Excitatory Antagonist: blocks excitatory neurons
            antagonist_effect = self.tripping[MoleculeType.EXCITATORY_ANTAGONIST]
            antagonist_effect *= potentiator_amp
            modified_potential -= antagonist_effect * 0.01
            
            # Excitatory Agonist: enhances excitatory neurons
            agonist_effect = self.tripping[MoleculeType.EXCITATORY_AGONIST]
            agonist_effect *= potentiator_amp
            if is_firing:
                modified_potential += agonist_effect * 0.01
                
        return modified_potential
    
    def get_plasticity_modulation(self, is_inhibitory: bool) -> Tuple[float, float]:
        """
        Get drug modulation of synaptic plasticity.
        
        THIS IS THE CRITICAL INTEGRATION:
        Drugs affect learning rates, making certain experiences more/less likely
        to be encoded into the neural network. This changes evolutionary trajectories
        through altered learning during drug experiences.
        
        Psychedelic drugs increase plasticity = enhanced learning/unlearning
        Stimulants increase strengthening = faster habit formation
        Depressants decrease plasticity = reduced learning
        Potentiator amplifies all effects = profound neuroplasticity changes
        
        Args:
            is_inhibitory: Whether synapse is inhibitory
            
        Returns:
            (strengthen_multiplier, weaken_multiplier) - multiply baseline STDP rates
        """
        # Get potentiator amplification
        potentiator_amp = 1.0 + (self.tripping[MoleculeType.POTENTIATOR] / self.max_trip) * 10.0
        
        # Base multipliers (1.0 = no change)
        strengthen_mult = 1.0
        weaken_mult = 1.0
        
        if is_inhibitory:
            # Inhibitory Antagonist: blocks inhibition = increases plasticity
            # (disinhibition leads to enhanced learning)
            antagonist = self.tripping[MoleculeType.INHIBITORY_ANTAGONIST] / self.max_trip
            antagonist *= potentiator_amp
            strengthen_mult += antagonist * 2.0  # Up to 3x strengthening
            weaken_mult += antagonist * 1.5      # Up to 2.5x weakening
            
            # Inhibitory Agonist: enhances inhibition = decreases plasticity
            # (strong inhibition reduces learning)
            agonist = self.tripping[MoleculeType.INHIBITORY_AGONIST] / self.max_trip
            agonist *= potentiator_amp
            strengthen_mult *= (1.0 - agonist * 0.7)  # Down to 0.3x
            weaken_mult *= (1.0 - agonist * 0.5)      # Down to 0.5x
        else:
            # Excitatory Antagonist: blocks excitation = decreases plasticity
            # (reduced excitation = less learning)
            antagonist = self.tripping[MoleculeType.EXCITATORY_ANTAGONIST] / self.max_trip
            antagonist *= potentiator_amp
            strengthen_mult *= (1.0 - antagonist * 0.6)  # Down to 0.4x
            weaken_mult *= (1.0 - antagonist * 0.4)      # Down to 0.6x
            
            # Excitatory Agonist: enhances excitation = increases plasticity
            # (strong excitation promotes learning)
            agonist = self.tripping[MoleculeType.EXCITATORY_AGONIST] / self.max_trip
            agonist *= potentiator_amp
            strengthen_mult += agonist * 3.0   # Up to 4x strengthening!
            weaken_mult += agonist * 1.0       # Up to 2x weakening
        
        # Clamp to reasonable bounds (prevent negative or excessive values)
        strengthen_mult = np.clip(strengthen_mult, 0.1, 10.0)
        weaken_mult = np.clip(weaken_mult, 0.1, 10.0)
        
        return (strengthen_mult, weaken_mult)
        
    def update(self):
        """Decay drug levels over time."""
        self.tripping *= self.decay_rate
        
    def get_total_drug_level(self) -> float:
        """Get total amount of drugs in system."""
        return float(np.sum(self.tripping))
        
    def is_sober(self, threshold: float = 10.0) -> bool:
        """Check if essentially sober."""
        return self.get_total_drug_level() < threshold
        
    def __repr__(self) -> str:
        return (
            f"DrugSystem(total={self.get_total_drug_level():.0f}, "
            f"potentiator={self.tripping[4]:.0f})"
        )
