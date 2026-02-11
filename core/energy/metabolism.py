"""
Energy and Metabolism System

Every action costs energy. Starvation leads to death.
Food and pills restore energy.

From telepathic-critterdrug profiles.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Food:
    """
    Food particle in environment.
    
    From foodotrope-drug.profile:
    - food_size = 25
    - food_energylevel = 100000
    - food_maxamount = 60
    """
    x: float
    y: float
    energy_value: float = 100000.0
    size: float = 25.0


class EnergySystem:
    """
    Manage energy budget for a creature.
    
    Energy costs based on telepathic-critterdrug profiles:
    - Base metabolism
    - Neuron existence cost
    - Synapse existence cost  
    - Neuron firing cost
    - Motor activation cost
    - Reproduction cost
    """
    
    def __init__(
        self,
        initial_energy: float = 1000000.0,
        max_energy: float = 10000000.0
    ):
        """
        Initialize energy system.
        
        Args:
            initial_energy: Starting energy
            max_energy: Maximum energy capacity
        """
        self.energy = initial_energy
        self.max_energy = max_energy
        
        # Energy costs per timestep (from profiles)
        self.base_metabolism = 50.0  # Just being alive
        self.neuron_existence_cost = 0.01  # Per neuron
        self.synapse_existence_cost = 0.001  # Per synapse
        self.neuron_firing_cost = 1.0  # When neuron fires
        self.motor_activation_cost = 10.0  # Using muscles
        self.reproduction_cost = 500000.0  # Creating offspring
        
        # Tracking
        self.total_consumed = 0.0
        self.total_spent = 0.0
        self.timesteps_alive = 0
        
    def consume_food(self, food: Food) -> float:
        """
        Eat food, gain energy.
        
        Args:
            food: Food particle
            
        Returns:
            Energy gained
        """
        gained = food.energy_value
        self.energy = min(self.max_energy, self.energy + gained)
        self.total_consumed += gained
        return gained
        
    def spend_energy(self, amount: float) -> bool:
        """
        Spend energy if available.
        
        Args:
            amount: Energy to spend
            
        Returns:
            True if successfully spent, False if insufficient
        """
        if self.energy >= amount:
            self.energy -= amount
            self.total_spent += amount
            return True
        return False
        
    def update_metabolism(
        self,
        num_neurons: int,
        num_synapses: int,
        num_firing: int,
        motor_activity: float = 0.0
    ) -> bool:
        """
        Pay metabolic costs for timestep.
        
        Args:
            num_neurons: Number of neurons in brain
            num_synapses: Number of synapses
            num_firing: Number of neurons firing this timestep
            motor_activity: Motor neuron activation level (0-1)
            
        Returns:
            True if enough energy, False if starving
        """
        cost = (
            self.base_metabolism +
            num_neurons * self.neuron_existence_cost +
            num_synapses * self.synapse_existence_cost +
            num_firing * self.neuron_firing_cost +
            motor_activity * self.motor_activation_cost
        )
        
        success = self.spend_energy(cost)
        self.timesteps_alive += 1
        
        return success
        
    def can_reproduce(self) -> bool:
        """Check if enough energy to reproduce."""
        return self.energy >= self.reproduction_cost
        
    def pay_reproduction_cost(self) -> bool:
        """Pay energy cost for reproduction."""
        return self.spend_energy(self.reproduction_cost)
        
    def is_starving(self, threshold: float = 1000.0) -> bool:
        """Check if energy critically low."""
        return self.energy < threshold
        
    def get_energy_fraction(self) -> float:
        """Get current energy as fraction of max (0-1)."""
        return self.energy / self.max_energy
        
    def transfer_to_offspring(self, fraction: float = 0.3) -> float:
        """
        Transfer energy to offspring at birth.
        
        Args:
            fraction: Fraction of current energy to transfer
            
        Returns:
            Energy transferred
        """
        amount = self.energy * fraction
        self.spend_energy(amount)
        return amount
        
    def __repr__(self) -> str:
        return (
            f"EnergySystem(energy={self.energy:.0f}/{self.max_energy:.0f}, "
            f"alive={self.timesteps_alive})"
        )
