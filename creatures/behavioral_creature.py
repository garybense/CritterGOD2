"""
Behavioral creature with resource-seeking.

Extends MorphologicalCreature with:
- Resource-seeking behavior (food, drugs)
- Addiction and tolerance mechanics
- Hunger-driven movement
- Resource interaction

Complete artificial life: brain + body + behavior + resources.
"""

from typing import Optional
import numpy as np
from creatures.morphological_creature import MorphologicalCreature
from core.behavior.resource_seeking import ResourceSeekingBehavior
from core.resources.resource import Resource, ResourceType
from core.resources.resource_manager import ResourceManager
from core.evolution.genotype import Genotype
from core.morphology.body_genotype import BodyGenotype
from core.morphic.circuit8 import Circuit8


class BehavioralCreature(MorphologicalCreature):
    """
    Complete creature with resource-seeking behavior.
    
    Adds to MorphologicalCreature:
    - Resource-seeking behavior system
    - Food consumption from resources
    - Drug consumption from mushrooms
    - Hunger-driven movement
    - Addiction/tolerance/withdrawal
    
    Attributes:
        behavior: Resource-seeking behavior system
        target_resource: Current resource being pursued
        movement_speed: Base movement speed
    """
    
    def __init__(
        self,
        genotype: Genotype,
        body: Optional[BodyGenotype] = None,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        initial_energy: float = 1000000.0,
        circuit8: Optional[Circuit8] = None,
        **kwargs
    ):
        """
        Create behavioral creature.
        
        Args:
            genotype: Neural network genotype
            body: Body genotype
            x, y, z: Initial position
            initial_energy: Starting energy
            circuit8: Shared telepathic canvas
            **kwargs: Additional arguments
        """
        # Initialize morphological creature
        super().__init__(
            genotype=genotype,
            body=body,
            x=x,
            y=y,
            z=z,
            initial_energy=initial_energy,
            circuit8=circuit8,
            **kwargs
        )
        
        # Initialize resource-seeking behavior
        self.behavior = ResourceSeekingBehavior()
        
        # Current target
        self.target_resource: Optional[Resource] = None
        
        # Movement parameters
        self.movement_speed = 2.0  # Units per timestep
        self.detection_range = 100.0  # How far creature can sense resources
    
    def update(self, dt: float = 1.0, resource_manager: Optional[ResourceManager] = None) -> bool:
        """
        Update creature with resource-seeking behavior.
        
        Args:
            dt: Time step
            resource_manager: Resource manager for finding/consuming resources
            
        Returns:
            True if alive, False if dead
        """
        # Update base morphological creature
        alive = super().update(dt)
        if not alive:
            return False
        
        # Update behavior system (addiction, tolerance, withdrawal)
        self.behavior.update(dt)
        
        # Resource-seeking behavior
        if resource_manager is not None:
            self._update_resource_seeking(resource_manager, dt)
        
        return True
    
    def _update_resource_seeking(self, resource_manager: ResourceManager, dt: float) -> None:
        """Update resource-seeking and consumption.
        
        Args:
            resource_manager: Resource manager
            dt: Time step
        """
        # Check if we have a target
        if self.target_resource is None or not self.target_resource.active:
            # Find new target based on needs
            self.target_resource = self._find_best_resource(resource_manager)
        
        # If we have a target, move toward it
        if self.target_resource is not None and self.target_resource.active:
            # Calculate distance
            dist = self.target_resource.distance_to(self.x, self.y, self.z)
            
            # If in range, consume
            if dist <= self.target_resource.radius:
                self._consume_resource(self.target_resource, resource_manager)
                self.target_resource = None  # Find new target next timestep
            else:
                # Move toward target
                dx, dy = self.behavior.get_movement_direction(self.x, self.y, self.target_resource)
                
                # If physics is active, apply force instead of direct position modification
                if hasattr(self, 'rigid_body') and self.rigid_body is not None:
                    # Apply gentle force toward target through physics
                    force = np.array([dx * 2.0, dy * 2.0, 0.0], dtype=np.float32)
                    self.rigid_body.apply_force(force)
                else:
                    # No physics - move directly (fallback)
                    move_dist = self.movement_speed * dt
                    self.x += dx * move_dist
                    self.y += dy * move_dist
                    
                    # Stay in world bounds (simple wrapping for now)
                    self.x = self.x % resource_manager.world_width
                    self.y = self.y % resource_manager.world_height
    
    def _find_best_resource(self, resource_manager: ResourceManager) -> Optional[Resource]:
        """Find most motivating resource within detection range.
        
        Args:
            resource_manager: Resource manager
            
        Returns:
            Best resource to pursue or None
        """
        # Get resources in detection range
        nearby = resource_manager.get_resources_in_range(
            self.x, self.y, self.z,
            radius=self.detection_range
        )
        
        if not nearby:
            return None
        
        # Calculate motivation for each resource
        best_resource = None
        best_motivation = 0.0
        
        for resource in nearby:
            motivation = self.behavior.calculate_resource_motivation(
                resource,
                self.energy.energy
            )
            
            # Factor in distance (closer = more attractive)
            dist = resource.distance_to(self.x, self.y, self.z)
            distance_factor = 1.0 - (dist / self.detection_range)
            adjusted_motivation = motivation * (0.5 + 0.5 * distance_factor)
            
            if adjusted_motivation > best_motivation:
                best_motivation = adjusted_motivation
                best_resource = resource
        
        # Only pursue if motivation is above threshold
        if best_motivation > 0.1:
            return best_resource
        
        return None
    
    def _consume_resource(self, resource: Resource, resource_manager: ResourceManager) -> None:
        """Consume a resource.
        
        Args:
            resource: Resource to consume
            resource_manager: Resource manager
        """
        if not resource.active:
            return
        
        if resource.resource_type == ResourceType.FOOD:
            # Consume food (gain energy)
            amount = resource_manager.consume_resource(resource, resource.amount)
            if amount > 0:
                # Add energy (use food consumption method if available)
                self.energy.energy = min(self.energy.max_energy, self.energy.energy + amount)
                self.behavior.food_consumed_count += 1
                print(f"🍎 Creature ate food (+{amount:.0f} energy)")
        
        elif resource.resource_type == ResourceType.DRUG_MUSHROOM:
            # Consume drug mushroom
            if resource.molecule_type is not None:
                amount = resource_manager.consume_resource(resource, resource.amount)
                if amount > 0:
                    # Apply drug to creature's drug system
                    self.drugs.tripping[resource.molecule_type] += amount
                    
                    # Update behavior addiction/tolerance
                    self.behavior.consume_drug(resource.molecule_type, amount)
                    
                    print(f"🍄 Creature consumed drug type {resource.molecule_type} (dose: {amount:.1f})")
        
        elif resource.resource_type == ResourceType.ENERGY_ZONE:
            # Passive energy gain from being in zone
            energy_gain = 100.0  # Per timestep in zone
            self.energy.energy = min(self.energy.max_energy, self.energy.energy + energy_gain)
    
    def get_behavior_state(self) -> dict:
        """Get current behavioral state.
        
        Returns:
            Dictionary of behavior information
        """
        return {
            'hunger_level': self.behavior.get_hunger_level(self.energy.energy, self.energy.max_energy),
            'should_seek_food': self.behavior.should_seek_food(self.energy.energy),
            'should_seek_drug': self.behavior.should_seek_drug(),
            'strongest_craving': self.behavior.get_strongest_craving(),
            'addiction_levels': self.behavior.addiction_levels.tolist(),
            'tolerance_levels': self.behavior.tolerance_levels.tolist(),
            'has_target': self.target_resource is not None,
            'target_type': self.target_resource.resource_type.value if self.target_resource else None
        }
    
    def to_dict(self) -> dict:
        """Export complete creature state.
        
        Returns:
            Complete state dictionary
        """
        base_dict = super().to_dict()
        
        # Add behavioral information
        base_dict['behavior'] = self.behavior.to_dict()
        base_dict['behavior_state'] = self.get_behavior_state()
        base_dict['target_resource'] = {
            'has_target': self.target_resource is not None,
            'type': self.target_resource.resource_type.value if self.target_resource else None,
            'distance': self.target_resource.distance_to(self.x, self.y, self.z) if self.target_resource else None
        }
        
        return base_dict
