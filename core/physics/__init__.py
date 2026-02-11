"""
Physics module for CritterGOD.

Custom lightweight physics engine for artificial life simulation.
No external dependencies (PyBullet-free).
"""

from core.physics.physics_world import (
    PhysicsWorld,
    RigidBody,
    Collision,
    CollisionShape,
)

__all__ = [
    'PhysicsWorld',
    'RigidBody',
    'Collision',
    'CollisionShape',
]
