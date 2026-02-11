"""
Event logging system for CritterGOD.

Logs birth, death, mutation, and reproduction events
in critterdrug-style console output format.
"""

from collections import deque
from typing import List, Optional


class EventLogger:
    """
    Logs simulation events to console and maintains recent history.
    
    Formats events in critterdrug style:
    - "207836 : 207834 ad: 2 gen: 5 n: 80 s: 2906 brain mutant"
    - "207855 fell in the pit"
    - "207820 procreated with 207801"
    
    Attributes:
        log_lines: Recent log lines
        max_lines: Maximum lines to keep
    """
    
    def __init__(self, max_lines: int = 200):
        """
        Initialize event logger.
        
        Args:
            max_lines: Maximum log lines to keep in memory
        """
        self.log_lines = deque(maxlen=max_lines)
        self.max_lines = max_lines
        
        # Event counters
        self.births = 0
        self.deaths = 0
        self.reproductions = 0
        self.mutations = 0
    
    def log_birth(self, creature, timestep: int, mutations: Optional[List[str]] = None):
        """
        Log creature birth.
        
        Args:
            creature: Newborn creature
            timestep: Current timestep
            mutations: List of mutations (if any)
        """
        self.births += 1
        
        # Get parent ID if available
        parent_id = getattr(creature, 'parent_id', 'unknown')
        
        # Build mutation string
        mutation_str = ""
        if mutations:
            mutation_str = ", ".join(mutations)
            self.mutations += len(mutations)
        
        # Format: "timestep : creature_id ad: X gen: Y n: Z s: W mutations"
        line = (f"{timestep} : {creature.creature_id} "
                f"ad: {getattr(creature, 'adam_distance', 0)} "
                f"gen: {creature.generation} "
                f"n: {len(creature.network.neurons)} "
                f"s: {len(creature.network.synapses)}")
        
        if mutation_str:
            line += f" {mutation_str}"
        
        self.add_line(line)
    
    def log_death(self, creature, timestep: int, cause: str = "unknown"):
        """
        Log creature death.
        
        Args:
            creature: Deceased creature
            timestep: Current timestep
            cause: Cause of death
        """
        self.deaths += 1
        
        # Format: "timestep : creature_id died age: X energy: Y cause: Z"
        line = (f"{timestep} : {creature.creature_id} died "
                f"age: {creature.age} "
                f"energy: {creature.energy.energy:.0f} "
                f"cause: {cause}")
        
        self.add_line(line)
    
    def log_reproduction(self, parent1, parent2, offspring, timestep: int):
        """
        Log reproduction event.
        
        Args:
            parent1: First parent
            parent2: Second parent (or None for asexual)
            offspring: Offspring creature
            timestep: Current timestep
        """
        self.reproductions += 1
        
        if parent2:
            # Sexual reproduction
            line = (f"{timestep} : {parent1.creature_id} procreated with "
                    f"{parent2.creature_id} → offspring: {offspring.creature_id}")
        else:
            # Asexual reproduction
            line = (f"{timestep} : {parent1.creature_id} reproduced "
                    f"→ offspring: {offspring.creature_id}")
        
        self.add_line(line)
    
    def log_mutation(self, creature, timestep: int, mutation_type: str, details: str = ""):
        """
        Log major mutation event.
        
        Args:
            creature: Mutated creature
            timestep: Current timestep
            mutation_type: Type of mutation
            details: Additional details
        """
        self.mutations += 1
        
        line = f"{timestep} : {creature.creature_id} {mutation_type}"
        if details:
            line += f" - {details}"
        
        self.add_line(line)
    
    def log_event(self, timestep: int, message: str):
        """
        Log custom event.
        
        Args:
            timestep: Current timestep
            message: Event message
        """
        line = f"{timestep} : {message}"
        self.add_line(line)
    
    def add_line(self, line: str):
        """
        Add line to log.
        
        Args:
            line: Log line to add
        """
        self.log_lines.append(line)
        print(line)  # Also print to terminal
    
    def get_recent_lines(self, n: int = 20) -> List[str]:
        """
        Get recent log lines.
        
        Args:
            n: Number of lines to return
            
        Returns:
            List of recent log lines
        """
        lines = list(self.log_lines)
        return lines[-n:] if len(lines) > n else lines
    
    def get_all_lines(self) -> List[str]:
        """Get all log lines."""
        return list(self.log_lines)
    
    def clear(self):
        """Clear log history."""
        self.log_lines.clear()
        self.births = 0
        self.deaths = 0
        self.reproductions = 0
        self.mutations = 0
    
    def get_stats(self) -> dict:
        """
        Get event statistics.
        
        Returns:
            Dictionary of event counts
        """
        return {
            'births': self.births,
            'deaths': self.deaths,
            'reproductions': self.reproductions,
            'mutations': self.mutations
        }
