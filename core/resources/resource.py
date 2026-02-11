"""
Resource types for CritterGOD ecosystem.

Resources include:
- Food (energy replenishment)
- Psychedelic mushrooms (drug sources)
- Energy zones (passive energy gain)
- Breeding grounds (safe reproduction)

Heritage from Critterding foodotrope system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class ResourceType(Enum):
    """Types of resources in the world."""
    FOOD = "food"
    DRUG_MUSHROOM = "drug_mushroom"
    ENERGY_ZONE = "energy_zone"
    BREEDING_GROUND = "breeding_ground"


@dataclass
class Resource:
    """Base resource in the world.
    
    Attributes:
        resource_type: Type of resource
        x: X position
        y: Y position
        z: Z position (height)
        amount: Current amount/energy available
        max_amount: Maximum capacity
        regrowth_rate: Amount regenerated per timestep
        radius: Interaction radius
        active: Whether resource is currently available
        respawn_time: Time until respawn if depleted
        molecule_type: Drug type (for mushrooms, 0-4)
    """
    resource_type: ResourceType
    x: float
    y: float
    z: float = 0.0
    amount: float = 100.0
    max_amount: float = 100.0
    regrowth_rate: float = 1.0
    radius: float = 10.0
    active: bool = True
    respawn_time: float = 0.0
    molecule_type: Optional[int] = None  # For drug mushrooms
    
    def update(self, dt: float = 1.0) -> None:
        """Update resource (regrowth, respawn).
        
        Args:
            dt: Time step
        """
        if not self.active:
            # Count down respawn timer
            self.respawn_time -= dt
            if self.respawn_time <= 0:
                self.respawn()
        else:
            # Regrow if not at max
            if self.amount < self.max_amount:
                self.amount = min(self.max_amount, self.amount + self.regrowth_rate * dt)
    
    def consume(self, amount: float) -> float:
        """Consume resource.
        
        Args:
            amount: Amount to consume
            
        Returns:
            Amount actually consumed
        """
        if not self.active:
            return 0.0
        
        consumed = min(amount, self.amount)
        self.amount -= consumed
        
        # Deactivate if fully depleted
        if self.amount <= 0:
            self.deactivate()
        
        return consumed
    
    def deactivate(self, respawn_delay: float = 100.0) -> None:
        """Deactivate resource and set respawn timer.
        
        Args:
            respawn_delay: Time until respawn
        """
        self.active = False
        self.amount = 0.0
        self.respawn_time = respawn_delay
    
    def respawn(self) -> None:
        """Respawn resource at full capacity."""
        self.active = True
        self.amount = self.max_amount
        self.respawn_time = 0.0
    
    def distance_to(self, x: float, y: float, z: float = 0.0) -> float:
        """Calculate distance to a point.
        
        Args:
            x, y, z: Point coordinates
            
        Returns:
            Euclidean distance
        """
        dx = self.x - x
        dy = self.y - y
        dz = self.z - z
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def is_in_range(self, x: float, y: float, z: float = 0.0) -> bool:
        """Check if point is within interaction radius.
        
        Args:
            x, y, z: Point coordinates
            
        Returns:
            True if within radius
        """
        return self.distance_to(x, y, z) <= self.radius


# Factory functions for common resource types

def create_food(x: float, y: float, energy_value: float = 100000.0) -> Resource:
    """Create food resource.
    
    Args:
        x, y: Position
        energy_value: Energy provided
        
    Returns:
        Food resource
    """
    return Resource(
        resource_type=ResourceType.FOOD,
        x=x,
        y=y,
        z=0.0,
        amount=energy_value,
        max_amount=energy_value,
        regrowth_rate=energy_value * 0.01,  # 1% regrowth per timestep
        radius=15.0
    )


def create_drug_mushroom(x: float, y: float, molecule_type: int, 
                        dosage: float = 50.0) -> Resource:
    """Create psychedelic mushroom.
    
    Args:
        x, y: Position
        molecule_type: Drug type (0-4)
        dosage: Drug amount
        
    Returns:
        Drug mushroom resource
    """
    return Resource(
        resource_type=ResourceType.DRUG_MUSHROOM,
        x=x,
        y=y,
        z=0.0,
        amount=dosage,
        max_amount=dosage,
        regrowth_rate=dosage * 0.005,  # Slower regrowth (0.5% per timestep)
        radius=12.0,
        molecule_type=molecule_type
    )


def create_energy_zone(x: float, y: float, radius: float = 50.0,
                       energy_rate: float = 100.0) -> Resource:
    """Create energy zone (passive energy gain).
    
    Args:
        x, y: Position
        radius: Zone radius
        energy_rate: Energy per timestep when inside
        
    Returns:
        Energy zone resource
    """
    return Resource(
        resource_type=ResourceType.ENERGY_ZONE,
        x=x,
        y=y,
        z=0.0,
        amount=float('inf'),  # Infinite energy
        max_amount=float('inf'),
        regrowth_rate=0.0,  # No regrowth needed
        radius=radius
    )


def create_breeding_ground(x: float, y: float, radius: float = 60.0) -> Resource:
    """Create breeding ground (safe reproduction zone).
    
    Args:
        x, y: Position
        radius: Zone radius
        
    Returns:
        Breeding ground resource
    """
    return Resource(
        resource_type=ResourceType.BREEDING_GROUND,
        x=x,
        y=y,
        z=0.0,
        amount=float('inf'),  # Always available
        max_amount=float('inf'),
        regrowth_rate=0.0,
        radius=radius
    )
