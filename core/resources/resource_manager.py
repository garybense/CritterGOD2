"""
Resource Manager for CritterGOD ecosystem.

Manages:
- Resource spawning and distribution
- Spatial distribution algorithms
- Resource regrowth and respawn
- Resource-creature interactions

Implements Poisson disk sampling for natural resource distribution.
"""

from typing import List, Tuple, Optional
import numpy as np
from core.resources.resource import (
    Resource, ResourceType, 
    create_food, create_drug_mushroom, 
    create_energy_zone, create_breeding_ground
)


class ResourceManager:
    """Manages all resources in the world.
    
    Attributes:
        world_width: World width
        world_height: World height
        resources: List of all resources
        food_density: Food items per 10000 sq units
        drug_density: Drug mushrooms per 10000 sq units
    """
    
    def __init__(self, world_width: float = 1000.0, world_height: float = 1000.0):
        """Initialize resource manager.
        
        Args:
            world_width: Width of world
            world_height: Height of world
        """
        self.world_width = world_width
        self.world_height = world_height
        self.resources: List[Resource] = []
        
        # Density parameters (configurable)
        self.food_density = 20.0  # 20 food per 10000 sq units
        self.drug_density = 10.0   # 10 drug mushrooms per 10000 sq units
        self.min_spacing = 30.0    # Minimum distance between resources
    
    def spawn_initial_resources(self, food_count: Optional[int] = None,
                                drug_count: Optional[int] = None) -> None:
        """Spawn initial resources using Poisson disk sampling.
        
        Args:
            food_count: Number of food items (None = calculate from density)
            drug_count: Number of drug mushrooms (None = calculate from density)
        """
        area = self.world_width * self.world_height
        
        if food_count is None:
            food_count = int((area / 10000.0) * self.food_density)
        if drug_count is None:
            drug_count = int((area / 10000.0) * self.drug_density)
        
        print(f"Spawning {food_count} food items and {drug_count} drug mushrooms...")
        
        # Generate positions using Poisson disk sampling
        positions = self._poisson_disk_sampling(
            food_count + drug_count,
            self.min_spacing
        )
        
        # Create food resources
        for i in range(food_count):
            if i < len(positions):
                x, y = positions[i]
                # Vary energy value slightly
                energy = np.random.uniform(80000.0, 120000.0)
                food = create_food(x, y, energy)
                self.resources.append(food)
        
        # Create drug mushrooms
        for i in range(food_count, food_count + drug_count):
            if i < len(positions):
                x, y = positions[i]
                # Random molecule type (0-4)
                molecule_type = np.random.randint(0, 5)
                # Vary dosage slightly
                dosage = np.random.uniform(40.0, 60.0)
                mushroom = create_drug_mushroom(x, y, molecule_type, dosage)
                self.resources.append(mushroom)
        
        print(f"✓ Spawned {len(self.resources)} resources")
    
    def spawn_energy_zones(self, count: int = 3) -> None:
        """Spawn energy zones (sunlight/heat vents).
        
        Args:
            count: Number of energy zones
        """
        for _ in range(count):
            x = np.random.uniform(50, self.world_width - 50)
            y = np.random.uniform(50, self.world_height - 50)
            radius = np.random.uniform(40.0, 80.0)
            energy_rate = np.random.uniform(50.0, 150.0)
            
            zone = create_energy_zone(x, y, radius, energy_rate)
            self.resources.append(zone)
        
        print(f"✓ Spawned {count} energy zones")
    
    def spawn_breeding_grounds(self, count: int = 2) -> None:
        """Spawn breeding grounds (safe reproduction zones).
        
        Args:
            count: Number of breeding grounds
        """
        for _ in range(count):
            x = np.random.uniform(100, self.world_width - 100)
            y = np.random.uniform(100, self.world_height - 100)
            radius = np.random.uniform(50.0, 100.0)
            
            ground = create_breeding_ground(x, y, radius)
            self.resources.append(ground)
        
        print(f"✓ Spawned {count} breeding grounds")
    
    def update(self, dt: float = 1.0) -> None:
        """Update all resources (regrowth, respawn).
        
        Args:
            dt: Time step
        """
        for resource in self.resources:
            resource.update(dt)
    
    def find_nearest_resource(self, x: float, y: float, z: float = 0.0,
                             resource_type: Optional[ResourceType] = None,
                             active_only: bool = True) -> Optional[Tuple[Resource, float]]:
        """Find nearest resource of a given type.
        
        Args:
            x, y, z: Position to search from
            resource_type: Type filter (None = any type)
            active_only: Only consider active resources
            
        Returns:
            (resource, distance) tuple or None if not found
        """
        nearest = None
        min_distance = float('inf')
        
        for resource in self.resources:
            # Filter by type
            if resource_type is not None and resource.resource_type != resource_type:
                continue
            
            # Filter by active status
            if active_only and not resource.active:
                continue
            
            # Calculate distance
            dist = resource.distance_to(x, y, z)
            if dist < min_distance:
                min_distance = dist
                nearest = resource
        
        if nearest is None:
            return None
        
        return (nearest, min_distance)
    
    def get_resources_in_range(self, x: float, y: float, z: float = 0.0,
                               radius: float = 50.0,
                               resource_type: Optional[ResourceType] = None) -> List[Resource]:
        """Get all resources within a radius.
        
        Args:
            x, y, z: Center position
            radius: Search radius
            resource_type: Type filter (None = any type)
            
        Returns:
            List of resources in range
        """
        in_range = []
        
        for resource in self.resources:
            # Filter by type
            if resource_type is not None and resource.resource_type != resource_type:
                continue
            
            # Check distance
            if resource.distance_to(x, y, z) <= radius:
                in_range.append(resource)
        
        return in_range
    
    def consume_resource(self, resource: Resource, amount: float) -> float:
        """Consume amount from resource.
        
        Args:
            resource: Resource to consume
            amount: Amount to consume
            
        Returns:
            Amount actually consumed
        """
        return resource.consume(amount)
    
    def get_resource_counts(self) -> dict:
        """Get counts of each resource type.
        
        Returns:
            Dictionary of resource type counts
        """
        counts = {
            ResourceType.FOOD: 0,
            ResourceType.DRUG_MUSHROOM: 0,
            ResourceType.ENERGY_ZONE: 0,
            ResourceType.BREEDING_GROUND: 0
        }
        
        active_counts = counts.copy()
        
        for resource in self.resources:
            counts[resource.resource_type] += 1
            if resource.active:
                active_counts[resource.resource_type] += 1
        
        return {
            'total': counts,
            'active': active_counts
        }
    
    def _poisson_disk_sampling(self, n: int, min_dist: float) -> List[Tuple[float, float]]:
        """Generate n positions using Poisson disk sampling.
        
        Ensures minimum distance between points for natural distribution.
        
        Args:
            n: Number of points to generate
            min_dist: Minimum distance between points
            
        Returns:
            List of (x, y) positions
        """
        # Simple Bridson's algorithm implementation
        cell_size = min_dist / np.sqrt(2)
        grid_width = int(np.ceil(self.world_width / cell_size))
        grid_height = int(np.ceil(self.world_height / cell_size))
        
        grid = np.full((grid_width, grid_height), -1, dtype=int)
        points = []
        active = []
        
        # Start with random point
        x0 = np.random.uniform(0, self.world_width)
        y0 = np.random.uniform(0, self.world_height)
        points.append((x0, y0))
        active.append(0)
        
        gx = int(x0 / cell_size)
        gy = int(y0 / cell_size)
        grid[gx, gy] = 0
        
        # Generate points
        k = 30  # Attempts per point
        while active and len(points) < n:
            idx = np.random.randint(0, len(active))
            point_idx = active[idx]
            px, py = points[point_idx]
            
            found = False
            for _ in range(k):
                # Random point in annulus
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(min_dist, 2 * min_dist)
                x = px + radius * np.cos(angle)
                y = py + radius * np.sin(angle)
                
                # Check bounds
                if x < 0 or x >= self.world_width or y < 0 or y >= self.world_height:
                    continue
                
                # Check grid
                gx = int(x / cell_size)
                gy = int(y / cell_size)
                
                # Check neighborhood
                valid = True
                for i in range(max(0, gx - 2), min(grid_width, gx + 3)):
                    for j in range(max(0, gy - 2), min(grid_height, gy + 3)):
                        if grid[i, j] != -1:
                            other_idx = grid[i, j]
                            ox, oy = points[other_idx]
                            dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                            if dist < min_dist:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    new_idx = len(points)
                    points.append((x, y))
                    active.append(new_idx)
                    grid[gx, gy] = new_idx
                    found = True
                    break
            
            if not found:
                active.pop(idx)
        
        return points[:n]
    
    def clear_resources(self) -> None:
        """Remove all resources."""
        self.resources.clear()
    
    def __repr__(self) -> str:
        counts = self.get_resource_counts()
        return (f"ResourceManager({len(self.resources)} resources: "
                f"{counts['active'][ResourceType.FOOD]} food, "
                f"{counts['active'][ResourceType.DRUG_MUSHROOM]} drugs)")
