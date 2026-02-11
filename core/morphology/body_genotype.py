"""
Genetic encoding for 3D creature body morphology.

Based on:
- Critterding articulated body system
- Karl Sims evolved virtual creatures (1994)
- Framsticks genetic encoding

A creature's body is defined by:
- Segments (cylinders with varying size)
- Limbs attached to segments (tapered cylinders)
- Joint positions and angles
- Symmetry parameters
- Color/pattern genetics
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import random
import numpy as np


@dataclass
class LimbGene:
    """Genetic encoding for a single limb.
    
    Attributes:
        length: Limb length relative to body segment (0.5-3.0)
        width: Limb width relative to length (0.1-0.5)
        angle_horizontal: Horizontal joint angle in degrees (0-360)
        angle_vertical: Vertical joint angle in degrees (-90 to 90)
        taper: How much limb tapers from base to tip (0.1-1.0)
    """
    length: float = 1.0
    width: float = 0.2
    angle_horizontal: float = 0.0
    angle_vertical: float = 0.0
    taper: float = 0.5
    
    def mutate(self, mutation_rate: float = 0.1) -> 'LimbGene':
        """Create mutated copy of limb gene."""
        if random.random() > mutation_rate:
            return LimbGene(
                length=self.length,
                width=self.width,
                angle_horizontal=self.angle_horizontal,
                angle_vertical=self.angle_vertical,
                taper=self.taper
            )
        
        # Mutate one parameter
        mutation_type = random.randint(0, 4)
        
        if mutation_type == 0:  # Length
            length = self.length + random.gauss(0, 0.3)
            length = max(0.5, min(3.0, length))
            return LimbGene(length, self.width, self.angle_horizontal, 
                          self.angle_vertical, self.taper)
        elif mutation_type == 1:  # Width
            width = self.width + random.gauss(0, 0.1)
            width = max(0.1, min(0.5, width))
            return LimbGene(self.length, width, self.angle_horizontal,
                          self.angle_vertical, self.taper)
        elif mutation_type == 2:  # Horizontal angle
            angle_h = self.angle_horizontal + random.gauss(0, 30)
            angle_h = angle_h % 360
            return LimbGene(self.length, self.width, angle_h,
                          self.angle_vertical, self.taper)
        elif mutation_type == 3:  # Vertical angle
            angle_v = self.angle_vertical + random.gauss(0, 20)
            angle_v = max(-90, min(90, angle_v))
            return LimbGene(self.length, self.width, self.angle_horizontal,
                          angle_v, self.taper)
        else:  # Taper
            taper = self.taper + random.gauss(0, 0.15)
            taper = max(0.1, min(1.0, taper))
            return LimbGene(self.length, self.width, self.angle_horizontal,
                          self.angle_vertical, taper)


@dataclass
class SegmentGene:
    """Genetic encoding for a body segment.
    
    Attributes:
        size: Segment size (radius for cylinder) (0.3-2.0)
        length: Segment length relative to size (0.5-3.0)
        limbs: List of limbs attached to this segment
    """
    size: float = 1.0
    length: float = 1.0
    limbs: List[LimbGene] = field(default_factory=list)
    
    def mutate(self, mutation_rate: float = 0.1) -> 'SegmentGene':
        """Create mutated copy of segment gene."""
        # Mutate segment parameters
        size = self.size
        length = self.length
        
        if random.random() < mutation_rate:
            mutation_type = random.randint(0, 1)
            if mutation_type == 0:  # Size
                size = self.size + random.gauss(0, 0.2)
                size = max(0.3, min(2.0, size))
            else:  # Length
                length = self.length + random.gauss(0, 0.3)
                length = max(0.5, min(3.0, length))
        
        # Mutate limbs
        mutated_limbs = [limb.mutate(mutation_rate) for limb in self.limbs]
        
        # Possibly add/remove limb
        if random.random() < mutation_rate * 0.3:
            if len(mutated_limbs) > 0 and random.random() < 0.4:
                # Remove random limb
                mutated_limbs.pop(random.randint(0, len(mutated_limbs) - 1))
            elif len(mutated_limbs) < 4:
                # Add random limb
                mutated_limbs.append(LimbGene(
                    length=random.uniform(0.5, 2.0),
                    width=random.uniform(0.1, 0.3),
                    angle_horizontal=random.uniform(0, 360),
                    angle_vertical=random.uniform(-45, 45),
                    taper=random.uniform(0.3, 0.8)
                ))
        
        return SegmentGene(size, length, mutated_limbs)


@dataclass
class BodyGenotype:
    """Complete genetic encoding for creature 3D body morphology.
    
    Attributes:
        segments: List of body segments from head to tail
        head_size: Head size relative to first segment (0.5-2.0)
        tail_length: Tail length relative to last segment (0-3.0)
        symmetry: Body symmetry factor (0=asymmetric, 1=symmetric)
        base_hue: Base color hue (0-360)
        pattern_type: Visual pattern (0=solid, 1=striped, 2=spotted)
        metallic: Metallic sheen factor (0-1.0)
    """
    segments: List[SegmentGene] = field(default_factory=list)
    head_size: float = 1.0
    tail_length: float = 0.5
    symmetry: float = 0.8
    base_hue: float = 180.0
    pattern_type: int = 0
    metallic: float = 0.3
    
    @staticmethod
    def create_random(min_segments: int = 2, max_segments: int = 6) -> 'BodyGenotype':
        """Create random body genotype.
        
        Args:
            min_segments: Minimum number of body segments
            max_segments: Maximum number of body segments
            
        Returns:
            Randomly initialized BodyGenotype
        """
        n_segments = random.randint(min_segments, max_segments)
        segments = []
        
        for i in range(n_segments):
            # Create segment
            size = random.uniform(0.5, 1.5)
            length = random.uniform(0.8, 2.0)
            
            # Add random limbs (0-3 per segment)
            n_limbs = random.randint(0, 3)
            limbs = []
            for _ in range(n_limbs):
                limbs.append(LimbGene(
                    length=random.uniform(0.5, 2.0),
                    width=random.uniform(0.1, 0.3),
                    angle_horizontal=random.uniform(0, 360),
                    angle_vertical=random.uniform(-45, 45),
                    taper=random.uniform(0.3, 0.8)
                ))
            
            segments.append(SegmentGene(size, length, limbs))
        
        return BodyGenotype(
            segments=segments,
            head_size=random.uniform(0.8, 1.5),
            tail_length=random.uniform(0, 2.0),
            symmetry=random.uniform(0.5, 1.0),
            base_hue=random.uniform(0, 360),
            pattern_type=random.randint(0, 2),
            metallic=random.uniform(0, 0.6)
        )
    
    def mutate(self, mutation_rate: float = 0.1) -> 'BodyGenotype':
        """Create mutated copy of body genotype.
        
        Args:
            mutation_rate: Probability of mutation per gene
            
        Returns:
            New mutated BodyGenotype
        """
        # Mutate segments
        mutated_segments = [seg.mutate(mutation_rate) for seg in self.segments]
        
        # Possibly add/remove segment
        if random.random() < mutation_rate * 0.2:
            if len(mutated_segments) > 2 and random.random() < 0.3:
                # Remove random segment (not head or tail)
                if len(mutated_segments) > 3:
                    idx = random.randint(1, len(mutated_segments) - 2)
                    mutated_segments.pop(idx)
            elif len(mutated_segments) < 10:
                # Add segment (duplicate existing one with mutation)
                idx = random.randint(0, len(mutated_segments) - 1)
                new_seg = mutated_segments[idx].mutate(0.5)
                mutated_segments.insert(idx + 1, new_seg)
        
        # Mutate global parameters
        head_size = self.head_size
        tail_length = self.tail_length
        symmetry = self.symmetry
        base_hue = self.base_hue
        pattern_type = self.pattern_type
        metallic = self.metallic
        
        if random.random() < mutation_rate:
            param = random.randint(0, 5)
            if param == 0:  # Head size
                head_size = self.head_size + random.gauss(0, 0.2)
                head_size = max(0.5, min(2.0, head_size))
            elif param == 1:  # Tail length
                tail_length = self.tail_length + random.gauss(0, 0.4)
                tail_length = max(0, min(3.0, tail_length))
            elif param == 2:  # Symmetry
                symmetry = self.symmetry + random.gauss(0, 0.15)
                symmetry = max(0, min(1.0, symmetry))
            elif param == 3:  # Base hue
                base_hue = (self.base_hue + random.gauss(0, 40)) % 360
            elif param == 4:  # Pattern type
                pattern_type = random.randint(0, 2)
            else:  # Metallic
                metallic = self.metallic + random.gauss(0, 0.15)
                metallic = max(0, min(1.0, metallic))
        
        return BodyGenotype(
            segments=mutated_segments,
            head_size=head_size,
            tail_length=tail_length,
            symmetry=symmetry,
            base_hue=base_hue,
            pattern_type=pattern_type,
            metallic=metallic
        )
    
    def get_signature(self) -> str:
        """Get unique signature for caching meshes.
        
        Returns a string that uniquely identifies this body shape
        for mesh caching purposes.
        """
        sig_parts = [
            f"s{len(self.segments)}",
            f"h{self.head_size:.2f}",
            f"t{self.tail_length:.2f}"
        ]
        
        for i, seg in enumerate(self.segments):
            sig_parts.append(f"seg{i}_{seg.size:.2f}_{seg.length:.2f}_l{len(seg.limbs)}")
            for j, limb in enumerate(seg.limbs):
                sig_parts.append(
                    f"l{i}_{j}_{limb.length:.2f}_{limb.width:.2f}_"
                    f"{limb.angle_horizontal:.1f}_{limb.angle_vertical:.1f}"
                )
        
        return "_".join(sig_parts)
    
    def get_total_mass(self) -> float:
        """Estimate total body mass for physics/energy calculations.
        
        Returns:
            Estimated body mass
        """
        mass = 0.0
        
        # Head
        mass += self.head_size ** 3
        
        # Segments
        for seg in self.segments:
            # Segment body (cylinder volume approximation)
            mass += (seg.size ** 2) * seg.length
            
            # Limbs
            for limb in seg.limbs:
                limb_volume = (limb.width * seg.size) ** 2 * (limb.length * seg.size)
                mass += limb_volume
        
        # Tail
        if self.tail_length > 0:
            tail_size = self.segments[-1].size * 0.5 if self.segments else 0.5
            mass += (tail_size ** 2) * self.tail_length
        
        return mass
