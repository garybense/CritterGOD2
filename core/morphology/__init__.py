"""
Core morphology module - 3D body generation and evolution.
"""

from core.morphology.body_genotype import BodyGenotype, SegmentGene, LimbGene
from core.morphology.mesh_generator import ProceduralMeshGenerator, Mesh3D

__all__ = [
    'BodyGenotype',
    'SegmentGene', 
    'LimbGene',
    'ProceduralMeshGenerator',
    'Mesh3D'
]
