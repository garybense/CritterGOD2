"""
Custom physics world for CritterGOD.

Lightweight physics engine optimized for artificial life simulation.
Implements essential features without PyBullet dependency issues.

Features:
- Rigid body dynamics with mass and velocity
- Simple collision detection (sphere-sphere, sphere-plane)
- Friction and damping
- Configurable gravity
- Spatial hashing for efficient collision queries

Based on Verlet integration for stability.
"""

from typing import List, Tuple, Optional, Set, Callable
import numpy as np
from dataclasses import dataclass
from enum import IntEnum


class CollisionShape(IntEnum):
    """Collision shape types."""
    SPHERE = 0
    BOX = 1
    CYLINDER = 2
    COMPOUND = 3


@dataclass
class RigidBody:
    """
    Rigid body for physics simulation.
    
    Uses Verlet integration for stability:
    - position: current position
    - prev_position: previous position (for velocity)
    - acceleration: current acceleration
    - mass: body mass
    - radius: collision radius (simplified sphere collision)
    """
    
    # Identity
    id: int
    
    # Transform
    position: np.ndarray  # (x, y, z)
    prev_position: np.ndarray  # For Verlet integration
    rotation: np.ndarray  # Quaternion (w, x, y, z)
    
    # Physics properties
    mass: float
    radius: float  # Simplified sphere collision
    restitution: float = 0.3  # Bounciness (0-1)
    friction: float = 0.5  # Friction coefficient
    damping: float = 0.98  # Velocity damping per timestep
    
    # Forces
    acceleration: np.ndarray = None  # Current acceleration
    
    # State
    fixed: bool = False  # If True, body doesn't move (infinite mass)
    sleeping: bool = False  # Sleep when velocity is low
    
    # Collision
    collision_group: int = 1  # Collision group bitmask
    collision_mask: int = 0xFFFFFFFF  # Which groups this collides with
    
    # User data
    user_data: object = None  # Reference to creature/resource
    
    def __post_init__(self):
        """Initialize arrays."""
        if self.acceleration is None:
            self.acceleration = np.zeros(3, dtype=np.float32)
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)
        if not isinstance(self.prev_position, np.ndarray):
            self.prev_position = np.array(self.prev_position, dtype=np.float32)
        if not isinstance(self.rotation, np.ndarray):
            self.rotation = np.array(self.rotation, dtype=np.float32)
    
    def get_velocity(self, dt: float = 1.0) -> np.ndarray:
        """Calculate velocity from Verlet positions."""
        return (self.position - self.prev_position) / dt
    
    def set_velocity(self, velocity: np.ndarray, dt: float = 1.0):
        """Set velocity by adjusting previous position."""
        self.prev_position = self.position - velocity * dt
    
    def apply_force(self, force: np.ndarray):
        """Apply force (F = ma, so a += F/m)."""
        if not self.fixed:
            self.acceleration += force / self.mass
    
    def apply_impulse(self, impulse: np.ndarray, dt: float = 1.0):
        """Apply instantaneous impulse."""
        if not self.fixed:
            velocity = self.get_velocity(dt)
            velocity += impulse / self.mass
            self.set_velocity(velocity, dt)


@dataclass
class Collision:
    """Collision between two bodies."""
    body_a: RigidBody
    body_b: RigidBody
    point: np.ndarray  # Collision point
    normal: np.ndarray  # Normal from A to B
    penetration: float  # How much bodies overlap


class PhysicsWorld:
    """
    Physics world with rigid bodies and collision detection.
    
    Uses:
    - Verlet integration for stability
    - Spatial hashing for efficient collision queries
    - Simple sphere-based collision detection
    """
    
    def __init__(
        self,
        gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81),
        world_bounds: Tuple[float, float, float, float] = (-1000.0, -1000.0, 1000.0, 1000.0),
        cell_size: float = 50.0
    ):
        """
        Initialize physics world.
        
        Args:
            gravity: Gravity acceleration (x, y, z)
            world_bounds: (min_x, min_y, max_x, max_y)
            cell_size: Spatial hash cell size
        """
        self.gravity = np.array(gravity, dtype=np.float32)
        self.world_bounds = world_bounds
        self.cell_size = cell_size
        
        # Bodies
        self.bodies: List[RigidBody] = []
        self.next_body_id = 0
        
        # Collision callbacks
        self.collision_callbacks: List[Callable[[Collision], None]] = []
        
        # Spatial hash for efficient collision queries
        self.spatial_hash: dict = {}
        
        # Ground plane (z=0)
        self.ground_enabled = True
        self.ground_friction = 0.7
        self.ground_restitution = 0.2
    
    def add_body(self, body: RigidBody) -> int:
        """
        Add rigid body to world.
        
        Args:
            body: Rigid body to add
            
        Returns:
            Body ID
        """
        body.id = self.next_body_id
        self.next_body_id += 1
        self.bodies.append(body)
        return body.id
    
    def remove_body(self, body_id: int):
        """Remove body from world."""
        self.bodies = [b for b in self.bodies if b.id != body_id]
    
    def create_sphere_body(
        self,
        position: Tuple[float, float, float],
        radius: float,
        mass: float = 1.0,
        **kwargs
    ) -> RigidBody:
        """
        Create and add sphere rigid body.
        
        Args:
            position: Initial position
            radius: Sphere radius
            mass: Body mass
            **kwargs: Additional RigidBody parameters
            
        Returns:
            Created rigid body
        """
        body = RigidBody(
            id=0,  # Will be assigned
            position=np.array(position, dtype=np.float32),
            prev_position=np.array(position, dtype=np.float32),
            rotation=np.array([1, 0, 0, 0], dtype=np.float32),  # Identity quaternion
            mass=mass,
            radius=radius,
            **kwargs
        )
        self.add_body(body)
        return body
    
    def register_collision_callback(self, callback: Callable[[Collision], None]):
        """Register callback for collision events."""
        self.collision_callbacks.append(callback)
    
    def _update_spatial_hash(self):
        """Update spatial hash for efficient collision queries."""
        self.spatial_hash.clear()
        
        for body in self.bodies:
            if body.sleeping:
                continue
            
            # Get cell coordinates
            cell_x = int(body.position[0] / self.cell_size)
            cell_y = int(body.position[1] / self.cell_size)
            
            # Add to cell and neighboring cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell = (cell_x + dx, cell_y + dy)
                    if cell not in self.spatial_hash:
                        self.spatial_hash[cell] = []
                    self.spatial_hash[cell].append(body)
    
    def _get_nearby_bodies(self, position: np.ndarray) -> List[RigidBody]:
        """Get bodies near position using spatial hash."""
        cell_x = int(position[0] / self.cell_size)
        cell_y = int(position[1] / self.cell_size)
        cell = (cell_x, cell_y)
        return self.spatial_hash.get(cell, [])
    
    def _check_collision(self, body_a: RigidBody, body_b: RigidBody) -> Optional[Collision]:
        """
        Check collision between two bodies (sphere-sphere).
        
        Returns:
            Collision object if colliding, None otherwise
        """
        # Check collision groups
        if not (body_a.collision_group & body_b.collision_mask):
            return None
        if not (body_b.collision_group & body_a.collision_mask):
            return None
        
        # Calculate distance
        delta = body_b.position - body_a.position
        distance = np.linalg.norm(delta)
        
        # Check if spheres overlap
        min_distance = body_a.radius + body_b.radius
        if distance < min_distance:
            penetration = min_distance - distance
            normal = delta / distance if distance > 0.0001 else np.array([0, 0, 1], dtype=np.float32)
            point = body_a.position + normal * body_a.radius
            
            return Collision(
                body_a=body_a,
                body_b=body_b,
                point=point,
                normal=normal,
                penetration=penetration
            )
        
        return None
    
    def _resolve_collision(self, collision: Collision, dt: float):
        """Resolve collision between two bodies."""
        body_a = collision.body_a
        body_b = collision.body_b
        
        # Calculate relative velocity
        vel_a = body_a.get_velocity(dt)
        vel_b = body_b.get_velocity(dt)
        relative_vel = vel_b - vel_a
        
        # Separate bodies
        if not body_a.fixed and not body_b.fixed:
            # Both movable - split separation
            separation = collision.normal * (collision.penetration * 0.5)
            body_a.position -= separation
            body_b.position += separation
        elif not body_a.fixed:
            # Only A movable
            separation = collision.normal * collision.penetration
            body_a.position -= separation
        elif not body_b.fixed:
            # Only B movable
            separation = collision.normal * collision.penetration
            body_b.position += separation
        
        # Calculate restitution (bounciness)
        restitution = min(body_a.restitution, body_b.restitution)
        
        # Calculate impulse
        relative_vel_normal = np.dot(relative_vel, collision.normal)
        if relative_vel_normal < 0:  # Bodies moving toward each other
            # Impulse magnitude
            impulse_magnitude = -(1 + restitution) * relative_vel_normal
            if body_a.fixed:
                impulse_magnitude /= (1.0 / body_b.mass)
            elif body_b.fixed:
                impulse_magnitude /= (1.0 / body_a.mass)
            else:
                impulse_magnitude /= (1.0 / body_a.mass + 1.0 / body_b.mass)
            
            impulse = collision.normal * impulse_magnitude
            
            # Apply impulse
            if not body_a.fixed:
                body_a.apply_impulse(-impulse, dt)
            if not body_b.fixed:
                body_b.apply_impulse(impulse, dt)
            
            # Apply friction
            tangent = relative_vel - collision.normal * relative_vel_normal
            tangent_length = np.linalg.norm(tangent)
            if tangent_length > 0.0001:
                tangent /= tangent_length
                friction = min(body_a.friction, body_b.friction)
                friction_impulse = tangent * (-impulse_magnitude * friction)
                
                if not body_a.fixed:
                    body_a.apply_impulse(-friction_impulse, dt)
                if not body_b.fixed:
                    body_b.apply_impulse(friction_impulse, dt)
    
    def _check_ground_collision(self, body: RigidBody, dt: float):
        """Check and resolve collision with ground plane."""
        if not self.ground_enabled or body.fixed:
            return
        
        # Ground is at z = body.radius (so body sits on z=0 plane)
        ground_z = body.radius
        
        if body.position[2] < ground_z:
            # Collision with ground
            penetration = ground_z - body.position[2]
            body.position[2] = ground_z
            
            # Get velocity
            velocity = body.get_velocity(dt)
            
            # Apply restitution (bounce)
            if velocity[2] < 0:
                velocity[2] = -velocity[2] * self.ground_restitution
                body.set_velocity(velocity, dt)
            
            # Apply friction
            horizontal_vel = np.array([velocity[0], velocity[1], 0], dtype=np.float32)
            friction_force = -horizontal_vel * self.ground_friction * body.mass
            body.apply_force(friction_force)
    
    def step(self, dt: float = 1.0/60.0):
        """
        Step physics simulation.
        
        Uses Verlet integration:
        1. Apply forces → acceleration
        2. Integrate: pos_new = 2*pos - pos_old + acc*dt^2
        3. Detect and resolve collisions
        4. Apply damping
        
        Args:
            dt: Timestep in seconds
        """
        # Update spatial hash
        self._update_spatial_hash()
        
        # Integrate bodies (Verlet)
        for body in self.bodies:
            if body.fixed or body.sleeping:
                continue
            
            # Apply gravity
            body.acceleration += self.gravity
            
            # Verlet integration
            velocity = body.get_velocity(dt)
            new_position = body.position + velocity * dt + body.acceleration * (dt * dt * 0.5)
            
            # Update positions
            body.prev_position = body.position.copy()
            body.position = new_position
            
            # Apply damping
            body.prev_position = body.position - (body.position - body.prev_position) * body.damping
            
            # Reset acceleration
            body.acceleration = np.zeros(3, dtype=np.float32)
            
            # Check world bounds (wrap or clamp)
            min_x, min_y, max_x, max_y = self.world_bounds
            body.position[0] = np.clip(body.position[0], min_x, max_x)
            body.position[1] = np.clip(body.position[1], min_y, max_y)
            
            # Check ground collision
            self._check_ground_collision(body, dt)
        
        # Collision detection and resolution
        collisions: List[Collision] = []
        checked_pairs: Set[Tuple[int, int]] = set()
        
        for body_a in self.bodies:
            if body_a.sleeping:
                continue
            
            nearby = self._get_nearby_bodies(body_a.position)
            for body_b in nearby:
                if body_a.id >= body_b.id:
                    continue  # Avoid checking same pair twice
                
                pair = (body_a.id, body_b.id)
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                
                collision = self._check_collision(body_a, body_b)
                if collision:
                    collisions.append(collision)
        
        # Resolve collisions
        for collision in collisions:
            self._resolve_collision(collision, dt)
            
            # Call collision callbacks
            for callback in self.collision_callbacks:
                callback(collision)
    
    def raycast(self, origin: np.ndarray, direction: np.ndarray, max_distance: float = 1000.0) -> Optional[Tuple[RigidBody, float, np.ndarray]]:
        """
        Cast ray and return first hit.
        
        Args:
            origin: Ray origin
            direction: Ray direction (normalized)
            max_distance: Maximum ray distance
            
        Returns:
            (body, distance, hit_point) or None
        """
        closest_hit = None
        closest_distance = max_distance
        
        for body in self.bodies:
            # Sphere raycast
            oc = origin - body.position
            b = 2.0 * np.dot(oc, direction)
            c = np.dot(oc, oc) - body.radius * body.radius
            discriminant = b*b - 4*c
            
            if discriminant >= 0:
                t = (-b - np.sqrt(discriminant)) / 2.0
                if 0 < t < closest_distance:
                    closest_distance = t
                    hit_point = origin + direction * t
                    closest_hit = (body, t, hit_point)
        
        return closest_hit
    
    def query_sphere(self, center: np.ndarray, radius: float) -> List[RigidBody]:
        """
        Find all bodies overlapping sphere.
        
        Args:
            center: Sphere center
            radius: Sphere radius
            
        Returns:
            List of overlapping bodies
        """
        results = []
        for body in self.bodies:
            distance = np.linalg.norm(body.position - center)
            if distance < (radius + body.radius):
                results.append(body)
        return results
    
    def get_body_by_id(self, body_id: int) -> Optional[RigidBody]:
        """Get body by ID."""
        for body in self.bodies:
            if body.id == body_id:
                return body
        return None
