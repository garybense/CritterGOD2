"""
Force-Directed Layout System

Positions creatures based on genetic similarity using spring-based physics.
From SpaceNet r221 - flamoot's 3D graph visualization framework.

Key concept:
- Similar creatures attract (form species clusters)
- Dissimilar creatures repel (maintain diversity)
- System finds equilibrium through physics simulation
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Vec2:
    """2D vector for positions and velocities."""
    x: float
    y: float
    
    def __add__(self, other: 'Vec2') -> 'Vec2':
        return Vec2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vec2') -> 'Vec2':
        return Vec2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vec2':
        return Vec2(self.x * scalar, self.y * scalar)
    
    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2)
    
    def normalize(self) -> 'Vec2':
        mag = self.magnitude()
        if mag < 1e-10:
            return Vec2(0, 0)
        return Vec2(self.x / mag, self.y / mag)


class ForceDirectedLayout:
    """
    Force-directed layout engine inspired by SpaceNet.
    
    Uses physics simulation to position nodes:
    - Repulsion between all nodes (prevents overlap)
    - Attraction along edges (keeps connected nodes together)
    - Damping (prevents oscillation)
    
    In CritterGOD context:
    - Nodes = creatures
    - Edges = genetic similarity
    - High similarity = strong attraction
    """
    
    def __init__(
        self,
        repulsion_strength: float = 100.0,
        attraction_strength: float = 0.5,
        damping: float = 0.8,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize force-directed layout engine.
        
        Args:
            repulsion_strength: Base repulsion force (K in SpaceNet)
            attraction_strength: Spring constant for attractions
            damping: Velocity damping factor (0-1)
            similarity_threshold: Minimum similarity for attraction
        """
        self.repulsion_strength = repulsion_strength
        self.attraction_strength = attraction_strength
        self.damping = damping
        self.similarity_threshold = similarity_threshold
        
    def calculate_genotype_similarity(self, genotype1, genotype2) -> float:
        """
        Calculate genetic similarity between two genotypes.
        
        Returns similarity in [0, 1]:
        - 1.0 = identical
        - 0.0 = completely different
        
        Based on:
        - Shared neuron count
        - Shared synapse patterns
        - Similar network topology
        """
        # Count neurons by type
        n1_total = len(genotype1.neuron_genes)
        n2_total = len(genotype2.neuron_genes)
        
        if n1_total == 0 or n2_total == 0:
            return 0.0
        
        # Neuron count similarity
        neuron_ratio = min(n1_total, n2_total) / max(n1_total, n2_total)
        
        # Synapse count similarity
        s1_total = len(genotype1.synapse_genes)
        s2_total = len(genotype2.synapse_genes)
        
        if s1_total > 0 and s2_total > 0:
            synapse_ratio = min(s1_total, s2_total) / max(s1_total, s2_total)
        else:
            synapse_ratio = 0.5
        
        # Inhibitory ratio similarity
        n1_inhib = sum(1 for n in genotype1.neuron_genes if n.is_inhibitory)
        n2_inhib = sum(1 for n in genotype2.neuron_genes if n.is_inhibitory)
        
        inhib_ratio1 = n1_inhib / n1_total if n1_total > 0 else 0
        inhib_ratio2 = n2_inhib / n2_total if n2_total > 0 else 0
        
        inhib_similarity = 1.0 - abs(inhib_ratio1 - inhib_ratio2)
        
        # Combined similarity
        similarity = (
            neuron_ratio * 0.4 +
            synapse_ratio * 0.3 +
            inhib_similarity * 0.3
        )
        
        return similarity
    
    def update_forces(
        self,
        creatures: List,
        dt: float = 1.0
    ) -> Dict[int, Tuple[float, float]]:
        """
        Calculate and apply forces to all creatures.
        
        Returns dictionary of {creature_id: (force_x, force_y)}
        
        Args:
            creatures: List of Creature objects
            dt: Time delta for physics step
        """
        n = len(creatures)
        if n == 0:
            return {}
        
        # Initialize forces
        forces = {id(c): Vec2(0, 0) for c in creatures}
        
        # Calculate all pairwise forces
        for i in range(n):
            for j in range(i + 1, n):
                c1 = creatures[i]
                c2 = creatures[j]
                
                # Position vector from c1 to c2
                dx = c2.x - c1.x
                dy = c2.y - c1.y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < 1e-10:
                    # Nudge apart if overlapping
                    dx = np.random.randn() * 0.1
                    dy = np.random.randn() * 0.1
                    distance = np.sqrt(dx**2 + dy**2)
                
                # Normalized direction
                dir_x = dx / distance
                dir_y = dy / distance
                
                # Repulsion force (always present)
                repulsion = self.repulsion_strength / (distance + 1.0)
                
                force_x = -dir_x * repulsion
                force_y = -dir_y * repulsion
                
                # Attraction force (based on genetic similarity)
                similarity = self.calculate_genotype_similarity(
                    c1.genotype, c2.genotype
                )
                
                if similarity > self.similarity_threshold:
                    # Strong attraction for similar creatures
                    attraction = self.attraction_strength * similarity * distance
                    force_x += dir_x * attraction
                    force_y += dir_y * attraction
                
                # Apply forces (Newton's third law)
                forces[id(c1)] = forces[id(c1)] + Vec2(force_x, force_y)
                forces[id(c2)] = forces[id(c2)] + Vec2(-force_x, -force_y)
        
        # Convert to tuples for return
        return {cid: (f.x, f.y) for cid, f in forces.items()}
    
    def apply_forces_to_creatures(
        self,
        creatures: List,
        forces: Dict[int, Tuple[float, float]],
        dt: float = 1.0
    ):
        """
        Apply calculated forces to creature positions.
        
        Args:
            creatures: List of creatures
            forces: Dictionary of forces from update_forces()
            dt: Time delta
        """
        for creature in creatures:
            cid = id(creature)
            if cid not in forces:
                continue
            
            fx, fy = forces[cid]
            
            # Initialize velocity if not present
            if not hasattr(creature, 'vx'):
                creature.vx = 0.0
                creature.vy = 0.0
            
            # Update velocity
            creature.vx += fx * dt
            creature.vy += fy * dt
            
            # Apply damping
            creature.vx *= self.damping
            creature.vy *= self.damping
            
            # Update position
            creature.x += creature.vx * dt
            creature.y += creature.vy * dt
    
    def simulate_step(
        self,
        creatures: List,
        dt: float = 0.1
    ):
        """
        Run one step of force-directed layout simulation.
        
        Args:
            creatures: List of creatures to position
            dt: Time step size
        """
        forces = self.update_forces(creatures, dt)
        self.apply_forces_to_creatures(creatures, forces, dt)
    
    def calculate_similarity(self, creature1, creature2) -> float:
        """
        Calculate genetic similarity between two creatures.
        Convenience wrapper for calculate_genotype_similarity.
        """
        return self.calculate_genotype_similarity(
            creature1.genotype,
            creature2.genotype
        )
    
    def run_until_stable(
        self,
        creatures: List,
        max_iterations: int = 100,
        stability_threshold: float = 0.1
    ) -> Tuple[bool, int]:
        """
        Run simulation until system reaches equilibrium.
        
        Args:
            creatures: List of creatures
            max_iterations: Maximum number of steps
            stability_threshold: Stop when max velocity < this
            
        Returns:
            Tuple of (converged: bool, iterations: int)
        """
        for iteration in range(max_iterations):
            self.simulate_step(creatures, dt=0.1)
            
            # Check if stable
            max_velocity = 0.0
            for creature in creatures:
                if hasattr(creature, 'vx'):
                    velocity = np.sqrt(creature.vx**2 + creature.vy**2)
                    max_velocity = max(max_velocity, velocity)
            
            if max_velocity < stability_threshold:
                return (True, iteration + 1)
        
        return (False, max_iterations)
