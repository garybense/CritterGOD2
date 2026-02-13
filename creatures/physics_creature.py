"""
Physics-enabled creature.

Extends BehavioralCreature with physics simulation:
- Rigid body physics for movement
- Collision detection and response
- Neural motor control → physics forces
- Resource collision-based consumption
- Creature-creature physical interactions

Complete artificial life with realistic physics.
"""

from typing import Optional
import numpy as np
from creatures.behavioral_creature import BehavioralCreature
from core.physics.physics_world import PhysicsWorld, RigidBody, Collision
from core.resources.resource_manager import ResourceManager
from core.evolution.genotype import Genotype
from core.morphology.body_genotype import BodyGenotype
from core.morphic.circuit8 import Circuit8


class PhysicsCreature(BehavioralCreature):
    """
    Creature with physics simulation.
    
    Adds to BehavioralCreature:
    - Rigid body physics for movement
    - Collision-based resource consumption
    - Neural motor outputs drive physics forces
    - Realistic movement and interactions
    
    Attributes:
        rigid_body: Physics rigid body
        physics_world: Reference to physics world
        motor_force_scale: Scale factor for neural → force conversion
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
        physics_world: Optional[PhysicsWorld] = None,
        **kwargs
    ):
        """
        Create physics-enabled creature.
        
        Args:
            genotype: Neural network genotype
            body: Body genotype
            x, y, z: Initial position
            initial_energy: Starting energy
            circuit8: Shared telepathic canvas
            physics_world: Physics world for simulation
            **kwargs: Additional arguments
        """
        # Initialize behavioral creature
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
        
        # Physics integration
        self.physics_world = physics_world
        self.rigid_body: Optional[RigidBody] = None
        
        # Motor control parameters - reduced for calmer movement
        self.motor_force_scale = 20.0  # Neural output → force conversion (reduced from 100)
        self.max_force = 100.0  # Maximum force per timestep (reduced from 500)
        
        # Collision state
        self.colliding_with_resource = False
        self.colliding_with_creature = False
        
        # Create rigid body if physics world provided
        if self.physics_world is not None:
            self._create_rigid_body()
    
    def _create_rigid_body(self):
        """Create rigid body for physics simulation."""
        if self.physics_world is None:
            return
        
        # Calculate body mass from morphology
        # Base mass + segment/limb contributions
        base_mass = 1.0
        segment_mass = len(self.body.segments) * 0.5
        limb_mass = sum(len(s.limbs) for s in self.body.segments) * 0.2
        total_mass = base_mass + segment_mass + limb_mass
        
        # Calculate collision radius (simplified sphere)
        # Use body's largest dimension
        max_segment_size = max((s.size for s in self.body.segments), default=1.0)
        collision_radius = max_segment_size * 5.0  # Scale for rendering
        
        # Create rigid body
        self.rigid_body = self.physics_world.create_sphere_body(
            position=(self.x, self.y, self.z),
            radius=collision_radius,
            mass=total_mass,
            restitution=0.3,
            friction=0.7,
            damping=0.95,  # Damping for stability
            user_data=self  # Reference back to creature
        )
        
        # Set collision groups (creatures = group 1)
        self.rigid_body.collision_group = 1
        self.rigid_body.collision_mask = 0xFFFFFFFF  # Collide with everything
    
    def sync_from_physics(self):
        """Update creature position from physics simulation."""
        if self.rigid_body is None:
            return
        
        # Copy position from rigid body
        self.x = self.rigid_body.position[0]
        self.y = self.rigid_body.position[1]
        self.z = self.rigid_body.position[2]
    
    def apply_neural_forces(self, dt: float = 1.0):
        """
        Apply forces from neural motor outputs.
        
        Maps motor neuron outputs to physics forces:
        - Motor outputs 0-1: Forward/backward force
        - Motor outputs 2-3: Left/right force
        - Motor outputs 4-5: Up/down force (jump)
        
        Args:
            dt: Time step
        """
        if self.rigid_body is None:
            return
        
        # Get motor outputs from creature (already computed in base update)
        # motor_outputs array is from base Creature class
        if not hasattr(self, 'motor_outputs') or len(self.motor_outputs) < 6:
            return
        
        # Use first 6 motor outputs for physics forces
        motor_outputs = self.motor_outputs[:6]
        
        # Calculate force components
        force = np.zeros(3, dtype=np.float32)
        
        if len(motor_outputs) >= 2:
            # Forward/backward (Y axis in our coordinate system)
            forward = motor_outputs[0]
            backward = motor_outputs[1]
            force[1] = (forward - backward) * self.motor_force_scale
        
        if len(motor_outputs) >= 4:
            # Left/right (X axis)
            right = motor_outputs[2]
            left = motor_outputs[3]
            force[0] = (right - left) * self.motor_force_scale
        
        if len(motor_outputs) >= 6:
            # Up/down (Z axis) - jumping
            up = motor_outputs[4]
            down = motor_outputs[5]
            force[2] = (up - down) * self.motor_force_scale * 0.5  # Less vertical force
        
        # Clamp total force
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > self.max_force:
            force = force * (self.max_force / force_magnitude)
        
        # Apply force to rigid body
        self.rigid_body.apply_force(force)
        
        # Energy cost for applying force
        force_cost = force_magnitude * 0.01  # Energy cost scales with force
        self.energy.energy -= force_cost
    
    def handle_collision(self, collision: Collision):
        """
        Handle collision event.
        
        Args:
            collision: Collision information
        """
        # Determine which body is us
        other_body = collision.body_b if collision.body_a == self.rigid_body else collision.body_a
        
        if other_body.user_data is None:
            return
        
        # Check collision type
        from core.resources.resource import Resource
        from creatures.physics_creature import PhysicsCreature
        
        if isinstance(other_body.user_data, Resource):
            # Colliding with resource
            self.colliding_with_resource = True
            # Resource consumption handled in update loop
        
        elif isinstance(other_body.user_data, PhysicsCreature):
            # Colliding with another creature
            self.colliding_with_creature = True
            # Could trigger social behaviors here
    
    def update(self, dt: float = 1.0, resource_manager: Optional[ResourceManager] = None) -> bool:
        """
        Update physics-enabled creature.
        
        Args:
            dt: Time step
            resource_manager: Resource manager for consumption
            
        Returns:
            True if alive, False if dead
        """
        # Reset collision flags
        self.colliding_with_resource = False
        self.colliding_with_creature = False
        
        # Update base behavioral creature (neural network, behavior, etc.)
        alive = super().update(dt, resource_manager)
        if not alive:
            # Remove rigid body from physics world
            if self.rigid_body and self.physics_world:
                self.physics_world.remove_body(self.rigid_body.id)
            return False
        
        # Physics-specific updates
        if self.physics_world is not None:
            # Apply neural motor forces
            self.apply_neural_forces(dt)
            
            # Sync position from physics simulation
            self.sync_from_physics()
            
            # Check for resource collisions (physics-based consumption)
            if resource_manager is not None and self.rigid_body is not None:
                self._check_resource_collisions(resource_manager)
        
        return True
    
    def _check_resource_collisions(self, resource_manager: ResourceManager):
        """
        Check for and handle resource collisions.
        
        Uses physics collision detection rather than distance checks.
        
        Args:
            resource_manager: Resource manager
        """
        # Query nearby resources using physics world
        nearby_bodies = self.physics_world.query_sphere(
            self.rigid_body.position,
            self.rigid_body.radius * 2.0  # Search radius
        )
        
        for body in nearby_bodies:
            if body == self.rigid_body:
                continue
            
            # Check if body has resource user data
            from core.resources.resource import Resource
            if isinstance(body.user_data, Resource):
                resource = body.user_data
                
                # Calculate actual distance
                distance = np.linalg.norm(body.position - self.rigid_body.position)
                
                # Consume if touching
                if distance < (self.rigid_body.radius + body.radius):
                    self._consume_resource_physics(resource, resource_manager)
    
    def _consume_resource_physics(self, resource, resource_manager: ResourceManager):
        """
        Consume resource via physics collision.
        
        Args:
            resource: Resource to consume
            resource_manager: Resource manager
        """
        # Use parent class consumption logic
        self._consume_resource(resource, resource_manager)
    
    def get_velocity(self) -> np.ndarray:
        """Get creature velocity from physics."""
        if self.rigid_body is None:
            return np.zeros(3, dtype=np.float32)
        return self.rigid_body.get_velocity()
    
    def set_velocity(self, velocity: np.ndarray):
        """Set creature velocity in physics."""
        if self.rigid_body is not None:
            self.rigid_body.set_velocity(velocity)
    
    def apply_impulse(self, impulse: np.ndarray):
        """Apply impulse to creature."""
        if self.rigid_body is not None:
            self.rigid_body.apply_impulse(impulse)
    
    def to_dict(self) -> dict:
        """Export complete creature state with physics."""
        base_dict = super().to_dict()
        
        # Add physics information
        if self.rigid_body:
            base_dict['physics'] = {
                'position': self.rigid_body.position.tolist(),
                'velocity': self.rigid_body.get_velocity().tolist(),
                'mass': self.rigid_body.mass,
                'radius': self.rigid_body.radius,
                'colliding_with_resource': self.colliding_with_resource,
                'colliding_with_creature': self.colliding_with_creature
            }
        
        return base_dict


def create_physics_creatures(
    n_creatures: int,
    physics_world: PhysicsWorld,
    circuit8: Circuit8,
    world_bounds: tuple = (-250.0, -250.0, 250.0, 250.0)
) -> list:
    """
    Create multiple physics creatures with random placement.
    
    Args:
        n_creatures: Number of creatures to create
        physics_world: Physics world
        circuit8: Telepathic canvas
        world_bounds: (min_x, min_y, max_x, max_y)
        
    Returns:
        List of physics creatures
    """
    creatures = []
    
    min_x, min_y, max_x, max_y = world_bounds
    
    for i in range(n_creatures):
        # Random position
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        z = 10.0  # Start above ground
        
        # Random neural genotype
        genotype = Genotype.create_random(
            n_sensory=20,
            n_motor=20,
            n_hidden_min=50,
            n_hidden_max=150,
            synapses_per_neuron=30
        )
        
        # Random body
        body = BodyGenotype.create_random()
        
        # Create physics creature
        creature = PhysicsCreature(
            genotype=genotype,
            body=body,
            x=x,
            y=y,
            z=z,
            initial_energy=1000000.0,
            circuit8=circuit8,
            physics_world=physics_world
        )
        
        creatures.append(creature)
    
    return creatures
