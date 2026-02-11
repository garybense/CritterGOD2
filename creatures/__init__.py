"""
Creatures module.

Provides creature classes from basic to complete:
- Creature: Basic creature with brain + body + energy
- EnhancedCreature: Multi-modal (vision, audio, text)
- MorphologicalCreature: 3D procedural bodies
- BehavioralCreature: Resource-seeking behavior (food, drugs)
"""

from creatures.creature import Creature
from creatures.enhanced_creature import EnhancedCreature
from creatures.morphological_creature import MorphologicalCreature
from creatures.behavioral_creature import BehavioralCreature
from creatures.physics_creature import PhysicsCreature, create_physics_creatures

__all__ = [
    'Creature',
    'EnhancedCreature',
    'MorphologicalCreature',
    'BehavioralCreature',
    'PhysicsCreature',
    'create_physics_creatures',
]
