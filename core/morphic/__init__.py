"""
Morphic Field Module

Circuit8 telepathic canvas and morphic resonance system.
Based on telepathic-critterdrug's revolutionary shared perception space.

Enhanced with semantic field types (pheromone trails, danger zones, etc.)
"""

from .circuit8 import Circuit8
from .morphic_field import (
    MorphicChannel,
    FieldType,
    MorphicFieldReader,
    MorphicFieldWriter,
)

__all__ = [
    "Circuit8",
    "MorphicChannel",
    "FieldType",
    "MorphicFieldReader",
    "MorphicFieldWriter",
]
