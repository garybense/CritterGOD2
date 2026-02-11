"""
Resources module for CritterGOD ecosystem.

Provides food, drugs, energy zones, and breeding grounds.
"""

from core.resources.resource import (
    Resource,
    ResourceType,
    create_food,
    create_drug_mushroom,
    create_energy_zone,
    create_breeding_ground
)
from core.resources.resource_manager import ResourceManager

__all__ = [
    'Resource',
    'ResourceType',
    'create_food',
    'create_drug_mushroom',
    'create_energy_zone',
    'create_breeding_ground',
    'ResourceManager',
]
