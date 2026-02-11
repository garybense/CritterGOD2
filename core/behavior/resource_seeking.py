"""
Resource-seeking behavior for creatures.

Implements:
- Hunger-driven food seeking
- Drug-seeking with addiction mechanics
- Energy zone attraction
- Breeding ground navigation

Behaviors emerge from neural activation + learned associations.
"""

from typing import Optional, Tuple
import numpy as np
from core.resources.resource import Resource, ResourceType


class ResourceSeekingBehavior:
    """Resource-seeking behavior system.
    
    Attributes:
        hunger_threshold: Energy level that triggers food seeking
        drug_craving_base: Base drug-seeking strength
        addiction_levels: Addiction level for each molecule type (0-4)
        tolerance_levels: Tolerance for each molecule type
        last_drug_time: Timesteps since last drug consumption per type
        drug_seeking_strength: Current drug-seeking motivation
    """
    
    def __init__(self):
        """Initialize resource-seeking behavior."""
        # Hunger parameters
        self.hunger_threshold = 500000.0  # Seek food below this energy
        self.starvation_threshold = 100000.0  # Desperate below this
        
        # Drug-seeking parameters
        self.drug_craving_base = 0.1  # Base craving strength
        self.addiction_levels = np.zeros(5)  # Per molecule type
        self.tolerance_levels = np.zeros(5)  # Per molecule type
        self.last_drug_time = np.zeros(5)  # Timesteps since last use
        self.drug_seeking_strength = np.zeros(5)  # Current motivation
        
        # Addiction/tolerance dynamics
        self.addiction_buildup_rate = 0.01  # Per drug consumption
        self.addiction_decay_rate = 0.001  # Per timestep without drug
        self.tolerance_buildup_rate = 0.02  # Per drug consumption
        self.tolerance_decay_rate = 0.0005  # Per timestep
        self.withdrawal_strength = 0.5  # Withdrawal craving multiplier
        
        # Behavioral stats
        self.food_consumed_count = 0
        self.drugs_consumed_count = 0
        self.withdrawal_episodes = 0
    
    def update(self, dt: float = 1.0) -> None:
        """Update addiction/tolerance/withdrawal dynamics.
        
        Args:
            dt: Time step
        """
        # Update time since last drug use
        self.last_drug_time += dt
        
        # Decay addiction levels
        self.addiction_levels *= (1.0 - self.addiction_decay_rate * dt)
        self.addiction_levels = np.maximum(0, self.addiction_levels)
        
        # Decay tolerance levels
        self.tolerance_levels *= (1.0 - self.tolerance_decay_rate * dt)
        self.tolerance_levels = np.maximum(0, self.tolerance_levels)
        
        # Calculate drug-seeking strength (craving + withdrawal)
        for i in range(5):
            # Base craving from addiction
            craving = self.drug_craving_base + self.addiction_levels[i]
            
            # Withdrawal effects (stronger if recently used but now abstaining)
            if self.last_drug_time[i] > 10.0 and self.addiction_levels[i] > 0.1:
                withdrawal = self.addiction_levels[i] * self.withdrawal_strength
                withdrawal *= min(1.0, self.last_drug_time[i] / 50.0)  # Peaks at 50 timesteps
                craving += withdrawal
                
                # Track withdrawal episodes
                if withdrawal > 0.2:
                    self.withdrawal_episodes += 1
            
            self.drug_seeking_strength[i] = min(1.0, craving)
    
    def consume_drug(self, molecule_type: int, amount: float) -> None:
        """Record drug consumption and update addiction/tolerance.
        
        Args:
            molecule_type: Drug type (0-4)
            amount: Dosage consumed
        """
        if molecule_type < 0 or molecule_type >= 5:
            return
        
        # Build addiction
        self.addiction_levels[molecule_type] += self.addiction_buildup_rate * amount
        self.addiction_levels[molecule_type] = min(1.0, self.addiction_levels[molecule_type])
        
        # Build tolerance
        self.tolerance_levels[molecule_type] += self.tolerance_buildup_rate * amount
        self.tolerance_levels[molecule_type] = min(1.0, self.tolerance_levels[molecule_type])
        
        # Reset last use timer
        self.last_drug_time[molecule_type] = 0.0
        
        # Track consumption
        self.drugs_consumed_count += 1
    
    def get_hunger_level(self, current_energy: float, max_energy: float) -> float:
        """Calculate hunger level (0=satisfied, 1=starving).
        
        Args:
            current_energy: Current energy
            max_energy: Maximum energy capacity
            
        Returns:
            Hunger level 0-1
        """
        if current_energy >= self.hunger_threshold:
            return 0.0
        elif current_energy <= self.starvation_threshold:
            return 1.0
        else:
            # Linear from hunger_threshold to starvation_threshold
            range_size = self.hunger_threshold - self.starvation_threshold
            return 1.0 - ((current_energy - self.starvation_threshold) / range_size)
    
    def get_drug_seeking_level(self, molecule_type: int) -> float:
        """Get current drug-seeking motivation for a molecule type.
        
        Args:
            molecule_type: Drug type (0-4)
            
        Returns:
            Seeking strength 0-1
        """
        if molecule_type < 0 or molecule_type >= 5:
            return 0.0
        
        return self.drug_seeking_strength[molecule_type]
    
    def get_strongest_craving(self) -> Tuple[int, float]:
        """Get most craved drug type and its strength.
        
        Returns:
            (molecule_type, craving_strength) tuple
        """
        max_idx = int(np.argmax(self.drug_seeking_strength))
        max_strength = self.drug_seeking_strength[max_idx]
        return (max_idx, max_strength)
    
    def should_seek_food(self, current_energy: float) -> bool:
        """Determine if creature should seek food.
        
        Args:
            current_energy: Current energy level
            
        Returns:
            True if should seek food
        """
        return current_energy < self.hunger_threshold
    
    def should_seek_drug(self, molecule_type: Optional[int] = None) -> bool:
        """Determine if creature should seek drugs.
        
        Args:
            molecule_type: Specific drug type (None = any drug)
            
        Returns:
            True if should seek drugs
        """
        if molecule_type is None:
            # Seek any drug if any craving is strong
            return np.max(self.drug_seeking_strength) > 0.2
        else:
            return self.drug_seeking_strength[molecule_type] > 0.2
    
    def calculate_resource_motivation(self, resource: Resource,
                                     current_energy: float) -> float:
        """Calculate motivation to pursue a specific resource.
        
        Combines hunger, drug craving, and resource type.
        
        Args:
            resource: Resource to evaluate
            current_energy: Current energy level
            
        Returns:
            Motivation strength 0-1
        """
        if not resource.active:
            return 0.0
        
        motivation = 0.0
        
        if resource.resource_type == ResourceType.FOOD:
            # Motivation from hunger
            hunger = self.get_hunger_level(current_energy, self.hunger_threshold * 10)
            motivation = hunger * 1.0  # Food is high priority when hungry
        
        elif resource.resource_type == ResourceType.DRUG_MUSHROOM:
            # Motivation from drug craving
            if resource.molecule_type is not None:
                craving = self.get_drug_seeking_level(resource.molecule_type)
                motivation = craving * 0.8  # Drugs slightly lower priority than food
        
        elif resource.resource_type == ResourceType.ENERGY_ZONE:
            # Motivation from low energy
            if current_energy < self.hunger_threshold * 2:
                motivation = 0.3  # Moderate attraction to energy zones
        
        elif resource.resource_type == ResourceType.BREEDING_GROUND:
            # Motivation from high energy (ready to reproduce)
            if current_energy > self.hunger_threshold * 5:
                motivation = 0.4  # Moderate attraction when well-fed
        
        return min(1.0, motivation)
    
    def get_movement_direction(self, creature_x: float, creature_y: float,
                              target_resource: Resource) -> Tuple[float, float]:
        """Calculate movement direction toward resource.
        
        Args:
            creature_x, creature_y: Creature position
            target_resource: Resource to move toward
            
        Returns:
            (dx, dy) normalized direction vector
        """
        dx = target_resource.x - creature_x
        dy = target_resource.y - creature_y
        
        # Normalize
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > 0:
            dx /= dist
            dy /= dist
        
        return (dx, dy)
    
    def to_dict(self) -> dict:
        """Export behavior state for debugging.
        
        Returns:
            Dictionary of behavior state
        """
        return {
            'hunger_threshold': self.hunger_threshold,
            'addiction_levels': self.addiction_levels.tolist(),
            'tolerance_levels': self.tolerance_levels.tolist(),
            'drug_seeking_strength': self.drug_seeking_strength.tolist(),
            'last_drug_time': self.last_drug_time.tolist(),
            'food_consumed': self.food_consumed_count,
            'drugs_consumed': self.drugs_consumed_count,
            'withdrawal_episodes': self.withdrawal_episodes,
            'strongest_craving': self.get_strongest_craving()
        }
