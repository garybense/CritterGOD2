"""
Species tracking and clustering system for CritterGOD.

Tracks genetic lineages, calculates similarity between creatures,
and groups them into species based on genetic distance.

Inspired by NEAT's speciation and critterding's adam distance tracking.
"""

from typing import List, Dict, Set, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Species:
    """
    A species cluster of genetically similar creatures.
    
    Attributes:
        species_id: Unique species identifier
        representative_id: ID of representative creature
        member_ids: Set of creature IDs in this species
        birth_generation: Generation when species first appeared
        last_generation: Most recent generation with members
        avg_fitness: Average fitness of species members
        color: RGB color for visualization
    """
    species_id: int
    representative_id: int
    member_ids: Set[int]
    birth_generation: int
    last_generation: int
    avg_fitness: float = 0.0
    color: Tuple[int, int, int] = (255, 255, 255)


class GeneticDistance:
    """
    Calculate genetic distance between creatures.
    
    Uses multiple metrics:
    - Neural topology distance (neuron count, synapse count)
    - Body morphology distance (segment count, limb count)
    - Weight distance (synapse weight differences)
    """
    
    @staticmethod
    def calculate_neural_distance(creature1, creature2) -> float:
        """
        Calculate neural network distance.
        
        Args:
            creature1, creature2: Creatures to compare
            
        Returns:
            Distance metric (0 = identical, higher = more different)
        """
        # Get network sizes
        if hasattr(creature1, 'network'):
            n1_neurons = len(creature1.network.neurons)
            n1_synapses = len(creature1.network.synapses)
        elif hasattr(creature1, 'brain'):
            n1_neurons = len(creature1.brain.neurons)
            n1_synapses = len(creature1.brain.synapses)
        else:
            return float('inf')
        
        if hasattr(creature2, 'network'):
            n2_neurons = len(creature2.network.neurons)
            n2_synapses = len(creature2.network.synapses)
        elif hasattr(creature2, 'brain'):
            n2_neurons = len(creature2.brain.neurons)
            n2_synapses = len(creature2.brain.synapses)
        else:
            return float('inf')
        
        # Topology distance
        neuron_diff = abs(n1_neurons - n2_neurons)
        synapse_diff = abs(n1_synapses - n2_synapses)
        
        # Normalize
        topology_distance = (neuron_diff / 200.0) + (synapse_diff / 5000.0)
        
        return topology_distance
    
    @staticmethod
    def calculate_body_distance(creature1, creature2) -> float:
        """
        Calculate body morphology distance.
        
        Args:
            creature1, creature2: Creatures to compare
            
        Returns:
            Distance metric (0 = identical)
        """
        if not (hasattr(creature1, 'body') and hasattr(creature2, 'body')):
            return 0.0
        
        body1 = creature1.body
        body2 = creature2.body
        
        # Segment count difference
        seg_diff = abs(body1.num_segments - body2.num_segments) / 10.0
        
        # Total limb count difference
        limb1 = sum(body1.limbs_per_segment)
        limb2 = sum(body2.limbs_per_segment)
        limb_diff = abs(limb1 - limb2) / 20.0
        
        # Size differences
        head_diff = abs(body1.head_size - body2.head_size) / 10.0
        
        return seg_diff + limb_diff + head_diff
    
    @staticmethod
    def calculate_total_distance(creature1, creature2) -> float:
        """
        Calculate total genetic distance.
        
        Args:
            creature1, creature2: Creatures to compare
            
        Returns:
            Combined distance metric
        """
        neural_dist = GeneticDistance.calculate_neural_distance(creature1, creature2)
        body_dist = GeneticDistance.calculate_body_distance(creature1, creature2)
        
        # Weighted combination
        return neural_dist * 0.7 + body_dist * 0.3


class SpeciesTracker:
    """
    Track and manage species clusters.
    
    Attributes:
        species: Dictionary mapping species_id to Species
        creature_species: Mapping from creature_id to species_id
        next_species_id: Counter for species IDs
        compatibility_threshold: Distance threshold for same species
        current_generation: Current generation number
    """
    
    def __init__(self, compatibility_threshold: float = 0.5):
        """
        Initialize species tracker.
        
        Args:
            compatibility_threshold: Max distance for same species
        """
        self.species: Dict[int, Species] = {}
        self.creature_species: Dict[int, int] = {}
        self.next_species_id = 0
        self.compatibility_threshold = compatibility_threshold
        self.current_generation = 0
    
    def update_species(self, creatures: List, generation: int):
        """
        Update species clusters for current generation.
        
        Args:
            creatures: List of all creatures
            generation: Current generation number
        """
        self.current_generation = generation
        
        # Clear old memberships
        for species in self.species.values():
            species.member_ids.clear()
        
        # Assign each creature to a species
        for creature in creatures:
            creature_id = creature.creature_id
            
            # Try to find compatible species
            assigned = False
            for species_id, species in self.species.items():
                # Get representative creature
                rep_creature = self._find_creature_by_id(creatures, species.representative_id)
                if rep_creature is None:
                    continue
                
                # Calculate distance
                distance = GeneticDistance.calculate_total_distance(creature, rep_creature)
                
                if distance < self.compatibility_threshold:
                    # Compatible! Add to this species
                    species.member_ids.add(creature_id)
                    self.creature_species[creature_id] = species_id
                    species.last_generation = generation
                    assigned = True
                    break
            
            # Create new species if no match
            if not assigned:
                new_species = Species(
                    species_id=self.next_species_id,
                    representative_id=creature_id,
                    member_ids={creature_id},
                    birth_generation=generation,
                    last_generation=generation,
                    color=self._generate_species_color()
                )
                self.species[self.next_species_id] = new_species
                self.creature_species[creature_id] = self.next_species_id
                self.next_species_id += 1
        
        # Remove extinct species (no members for 10 generations)
        extinct = [
            sid for sid, sp in self.species.items()
            if len(sp.member_ids) == 0 and (generation - sp.last_generation) > 10
        ]
        for sid in extinct:
            del self.species[sid]
        
        # Update fitness statistics
        self._update_fitness_stats(creatures)
    
    def _find_creature_by_id(self, creatures: List, creature_id: int):
        """Find creature by ID in list."""
        for creature in creatures:
            if creature.creature_id == creature_id:
                return creature
        return None
    
    def _update_fitness_stats(self, creatures: List):
        """Update fitness statistics for each species."""
        for species in self.species.values():
            if len(species.member_ids) == 0:
                continue
            
            # Calculate average fitness
            member_creatures = [
                c for c in creatures
                if c.creature_id in species.member_ids
            ]
            
            if member_creatures:
                avg_energy = sum(c.energy.energy for c in member_creatures) / len(member_creatures)
                species.avg_fitness = avg_energy
    
    def _generate_species_color(self) -> Tuple[int, int, int]:
        """Generate a random color for a new species."""
        hue = (self.next_species_id * 137.5) % 360  # Golden angle
        
        # Convert HSV to RGB (S=0.8, V=0.9)
        h = hue / 60.0
        c = 0.8 * 0.9
        x = c * (1 - abs(h % 2 - 1))
        m = 0.9 - c
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))
    
    def get_species_count(self) -> int:
        """Get number of active species."""
        return len([s for s in self.species.values() if len(s.member_ids) > 0])
    
    def get_largest_species(self) -> Optional[Species]:
        """Get species with most members."""
        active_species = [s for s in self.species.values() if len(s.member_ids) > 0]
        if not active_species:
            return None
        return max(active_species, key=lambda s: len(s.member_ids))
    
    def get_oldest_species(self) -> Optional[Species]:
        """Get oldest surviving species."""
        active_species = [s for s in self.species.values() if len(s.member_ids) > 0]
        if not active_species:
            return None
        return min(active_species, key=lambda s: s.birth_generation)
    
    def get_species_for_creature(self, creature_id: int) -> Optional[int]:
        """Get species ID for a creature."""
        return self.creature_species.get(creature_id)
    
    def get_species_diversity(self) -> float:
        """
        Calculate species diversity (Shannon index).
        
        Returns:
            Diversity metric (higher = more diverse)
        """
        active_species = [s for s in self.species.values() if len(s.member_ids) > 0]
        if not active_species:
            return 0.0
        
        total = sum(len(s.member_ids) for s in active_species)
        if total == 0:
            return 0.0
        
        diversity = 0.0
        for species in active_species:
            p = len(species.member_ids) / total
            if p > 0:
                diversity -= p * np.log(p)
        
        return diversity
    
    def get_statistics(self) -> Dict:
        """
        Get species statistics.
        
        Returns:
            Dictionary of statistics
        """
        active_species = [s for s in self.species.values() if len(s.member_ids) > 0]
        
        if not active_species:
            return {
                'total_species': 0,
                'avg_species_size': 0,
                'largest_species_size': 0,
                'oldest_species_age': 0,
                'diversity': 0.0
            }
        
        largest = self.get_largest_species()
        oldest = self.get_oldest_species()
        
        return {
            'total_species': len(active_species),
            'avg_species_size': sum(len(s.member_ids) for s in active_species) / len(active_species),
            'largest_species_size': len(largest.member_ids) if largest else 0,
            'oldest_species_age': self.current_generation - oldest.birth_generation if oldest else 0,
            'diversity': self.get_species_diversity()
        }


class LineageTracker:
    """
    Track family lineages and ancestry.
    
    Maintains parent-child relationships and calculates
    evolutionary distances.
    """
    
    def __init__(self):
        """Initialize lineage tracker."""
        self.parents: Dict[int, List[int]] = {}  # creature_id -> [parent1_id, parent2_id]
        self.children: Dict[int, Set[int]] = defaultdict(set)  # parent_id -> {child_ids}
        self.generations: Dict[int, int] = {}  # creature_id -> generation
    
    def record_birth(self, creature_id: int, parent_ids: List[int], generation: int):
        """
        Record a creature birth.
        
        Args:
            creature_id: New creature ID
            parent_ids: List of parent IDs (1 for asexual, 2 for sexual)
            generation: Generation number
        """
        self.parents[creature_id] = parent_ids
        self.generations[creature_id] = generation
        
        for parent_id in parent_ids:
            self.children[parent_id].add(creature_id)
    
    def get_ancestors(self, creature_id: int, max_depth: int = 10) -> Set[int]:
        """
        Get all ancestors up to max_depth generations.
        
        Args:
            creature_id: Creature to trace
            max_depth: Maximum generations to trace back
            
        Returns:
            Set of ancestor IDs
        """
        ancestors = set()
        current_gen = {creature_id}
        
        for _ in range(max_depth):
            if not current_gen:
                break
            
            next_gen = set()
            for cid in current_gen:
                if cid in self.parents:
                    parent_ids = self.parents[cid]
                    ancestors.update(parent_ids)
                    next_gen.update(parent_ids)
            
            current_gen = next_gen
        
        return ancestors
    
    def get_common_ancestor(self, creature1_id: int, creature2_id: int) -> Optional[int]:
        """
        Find most recent common ancestor.
        
        Args:
            creature1_id, creature2_id: Creatures to compare
            
        Returns:
            Common ancestor ID or None
        """
        ancestors1 = self.get_ancestors(creature1_id)
        ancestors2 = self.get_ancestors(creature2_id)
        
        common = ancestors1 & ancestors2
        if not common:
            return None
        
        # Find most recent (highest generation)
        return max(common, key=lambda cid: self.generations.get(cid, 0))
    
    def get_relatedness(self, creature1_id: int, creature2_id: int) -> float:
        """
        Calculate relatedness coefficient.
        
        Args:
            creature1_id, creature2_id: Creatures to compare
            
        Returns:
            Relatedness (0 = unrelated, 1 = identical)
        """
        # Check if direct parent-child
        if creature1_id in self.parents:
            if creature2_id in self.parents[creature1_id]:
                return 0.5
        if creature2_id in self.parents:
            if creature1_id in self.parents[creature2_id]:
                return 0.5
        
        # Check if siblings
        if creature1_id in self.parents and creature2_id in self.parents:
            parents1 = set(self.parents[creature1_id])
            parents2 = set(self.parents[creature2_id])
            if parents1 & parents2:
                return 0.25  # Half-siblings or full siblings
        
        # Find common ancestor
        common = self.get_common_ancestor(creature1_id, creature2_id)
        if common:
            gen1 = self.generations.get(creature1_id, 0)
            gen2 = self.generations.get(creature2_id, 0)
            gen_common = self.generations.get(common, 0)
            
            dist = (gen1 - gen_common) + (gen2 - gen_common)
            return 0.5 ** dist if dist > 0 else 0.0
        
        return 0.0
