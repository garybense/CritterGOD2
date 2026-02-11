"""
Population management utilities for CritterGOD.

Implements population control mechanisms like culling
and automatic population balancing.
"""

from typing import List


class PopulationManager:
    """
    Manages population size and composition.
    
    Provides population control mechanisms to prevent
    resource exhaustion and maintain performance.
    """
    
    def __init__(self):
        """Initialize population manager."""
        self.total_culled = 0
        self.last_cull_timestep = 0
    
    def kill_half_if_needed(self, creatures: List, threshold: int, timestep: int, logger=None) -> List:
        """
        Cull half the population if threshold exceeded.
        
        Removes lowest-fitness creatures (by energy).
        
        Args:
            creatures: List of creatures
            threshold: Population threshold (0 = disabled)
            timestep: Current timestep
            logger: Optional event logger
            
        Returns:
            List of culled creatures
        """
        if threshold <= 0 or len(creatures) <= threshold:
            return []
        
        # Sort by fitness (energy level)
        sorted_creatures = sorted(creatures, key=lambda c: c.energy.energy)
        
        # Kill bottom half
        num_to_kill = len(creatures) // 2
        to_kill = sorted_creatures[:num_to_kill]
        
        # Log culling event
        if logger:
            logger.log_event(timestep, f"Population control: culling {num_to_kill} creatures (threshold: {threshold})")
        
        # Mark creatures as dead
        for creature in to_kill:
            creatures.remove(creature)
            if logger:
                logger.log_death(creature, timestep, "population_control")
        
        self.total_culled += num_to_kill
        self.last_cull_timestep = timestep
        
        return to_kill
    
    def cull_oldest(self, creatures: List, max_age: int, timestep: int, logger=None) -> List:
        """
        Remove creatures that exceed maximum age.
        
        Args:
            creatures: List of creatures
            max_age: Maximum allowed age (0 = no limit)
            timestep: Current timestep
            logger: Optional event logger
            
        Returns:
            List of culled creatures
        """
        if max_age <= 0:
            return []
        
        culled = []
        for creature in creatures[:]:  # Copy to allow removal during iteration
            if creature.age >= max_age:
                creatures.remove(creature)
                culled.append(creature)
                if logger:
                    logger.log_death(creature, timestep, "old_age")
        
        return culled
    
    def cull_weakest(self, creatures: List, min_energy: float, timestep: int, logger=None) -> List:
        """
        Remove creatures below energy threshold.
        
        Args:
            creatures: List of creatures
            min_energy: Minimum energy to survive
            timestep: Current timestep
            logger: Optional event logger
            
        Returns:
            List of culled creatures
        """
        culled = []
        for creature in creatures[:]:
            if creature.energy.energy < min_energy:
                creatures.remove(creature)
                culled.append(creature)
                if logger:
                    logger.log_death(creature, timestep, "starvation")
        
        return culled
    
    def get_stats(self) -> dict:
        """
        Get population management statistics.
        
        Returns:
            Dictionary of stats
        """
        return {
            'total_culled': self.total_culled,
            'last_cull_timestep': self.last_cull_timestep
        }
