"""
Genotype: Genetic encoding of neural network structure.

Based on critterding-beta14 and telepathic-critterdrug genome structure.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MutationType(Enum):
    """Types of mutations that can occur."""
    ADD_NEURON = "add_neuron"
    REMOVE_NEURON = "remove_neuron"
    ADD_SYNAPSE = "add_synapse"
    REMOVE_SYNAPSE = "remove_synapse"
    MODIFY_THRESHOLD = "modify_threshold"
    MODIFY_WEIGHT = "modify_weight"
    MODIFY_PLASTICITY = "modify_plasticity"


@dataclass
class NeuronGene:
    """
    Genetic encoding of a single neuron.
    
    Based on critterding neuron parameters.
    """
    neuron_id: int
    threshold: float = 1000.0
    is_inhibitory: bool = False
    is_plastic: bool = True
    is_motor: bool = False
    is_sensory: bool = False
    leak_rate: float = 0.9


@dataclass
class SynapseGene:
    """
    Genetic encoding of a synapse.
    
    Based on critterding synapse parameters.
    """
    pre_neuron_id: int
    post_neuron_id: int
    weight: float = 1.0
    is_inhibitory: bool = False
    plasticity_rate: float = 0.01


class Genotype:
    """
    Genetic encoding of a complete neural network.
    
    This genotype can be:
    1. Mutated to create variation
    2. Combined with another genotype (crossover)
    3. Converted to a phenotype (actual neural network)
    
    Based on critterding's genetic encoding system with parameters from
    foodotrope-drug.profile.
    
    Attributes:
        neuron_genes: List of neuron genetic encodings
        synapse_genes: List of synapse genetic encodings
        fitness: Fitness score (energy-based)
        generation: Generation number
    """
    
    def __init__(
        self,
        neuron_genes: Optional[List[NeuronGene]] = None,
        synapse_genes: Optional[List[SynapseGene]] = None,
        generation: int = 0
    ):
        """
        Initialize a genotype.
        
        Args:
            neuron_genes: List of neuron genes
            synapse_genes: List of synapse genes
            generation: Generation number
        """
        self.neuron_genes = neuron_genes or []
        self.synapse_genes = synapse_genes or []
        self.fitness: float = 0.0
        self.generation = generation
        self._next_neuron_id = len(self.neuron_genes)
        
    @classmethod
    def create_random(
        cls,
        n_sensory: int = 10,
        n_motor: int = 10,
        n_hidden_min: int = 50,
        n_hidden_max: int = 200,
        synapses_per_neuron: int = 40,
        inhibitory_neuron_prob: float = 0.3,
        inhibitory_synapse_prob: float = 0.3,
    ) -> "Genotype":
        """
        Create a random genotype.
        
        Based on critterding profile parameters, updated with CritterGOD4 ratios.
        
        Args:
            n_sensory: Number of sensory neurons
            n_motor: Number of motor neurons
            n_hidden_min: Minimum hidden neurons (brain_minneuronsatbuildtime)
            n_hidden_max: Maximum hidden neurons (brain_maxneuronsatbuildtime)
            synapses_per_neuron: Average synapses per neuron
            inhibitory_neuron_prob: Probability of inhibitory neuron (0.3 from CritterGOD4)
            inhibitory_synapse_prob: Probability of inhibitory synapse (0.3 from CritterGOD4)
            
        Returns:
            Random genotype
        """
        neuron_genes = []
        neuron_id = 0
        
        # Create sensory neurons
        for _ in range(n_sensory):
            gene = NeuronGene(
                neuron_id=neuron_id,
                threshold=700 + np.random.uniform(0, 8000),
                is_sensory=True,
                is_plastic=True
            )
            neuron_genes.append(gene)
            neuron_id += 1
            
        # Create motor neurons
        for _ in range(n_motor):
            gene = NeuronGene(
                neuron_id=neuron_id,
                threshold=700 + np.random.uniform(0, 8000),
                is_motor=True,
                is_plastic=True
            )
            neuron_genes.append(gene)
            neuron_id += 1
            
        # Create hidden neurons
        n_hidden = np.random.randint(n_hidden_min, n_hidden_max + 1)
        for _ in range(n_hidden):
            is_inhib = np.random.random() < inhibitory_neuron_prob
            gene = NeuronGene(
                neuron_id=neuron_id,
                threshold=700 + np.random.uniform(0, 8000),
                is_inhibitory=is_inhib,
                is_plastic=np.random.random() < 0.5  # 50% plastic (from profile)
            )
            neuron_genes.append(gene)
            neuron_id += 1
            
        # Create synapses
        synapse_genes = []
        total_neurons = len(neuron_genes)
        
        for pre_gene in neuron_genes:
            # Random number of synapses per neuron
            n_synapses = max(2, int(np.random.normal(synapses_per_neuron, synapses_per_neuron * 0.2)))
            
            for _ in range(n_synapses):
                post_id = np.random.randint(0, total_neurons)
                
                # Skip self-connections
                if post_id == pre_gene.neuron_id:
                    continue
                    
                is_inhib = np.random.random() < inhibitory_synapse_prob
                
                synapse = SynapseGene(
                    pre_neuron_id=pre_gene.neuron_id,
                    post_neuron_id=post_id,
                    weight=np.random.uniform(50.0, 300.0),
                    is_inhibitory=is_inhib,
                    plasticity_rate=np.random.uniform(0.001, 0.05)
                )
                synapse_genes.append(synapse)
                
        genotype = cls(neuron_genes=neuron_genes, synapse_genes=synapse_genes)
        genotype._next_neuron_id = neuron_id
        return genotype
        
    def mutate(
        self,
        mutation_rate: float = 0.5,
        max_mutations: int = 50,
    ) -> "Genotype":
        """
        Create a mutated copy of this genotype.
        
        Based on critterding mutation system with probabilities from profile.
        
        Args:
            mutation_rate: Probability of mutation (brain_mutationrate / 100)
            max_mutations: Maximum mutations per genotype (brain_maxmutations)
            
        Returns:
            Mutated genotype
        """
        # Create deep copy
        new_neuron_genes = [
            NeuronGene(**vars(gene)) for gene in self.neuron_genes
        ]
        new_synapse_genes = [
            SynapseGene(**vars(gene)) for gene in self.synapse_genes
        ]
        
        mutated = Genotype(
            neuron_genes=new_neuron_genes,
            synapse_genes=new_synapse_genes,
            generation=self.generation + 1
        )
        mutated._next_neuron_id = self._next_neuron_id
        
        if np.random.random() > mutation_rate:
            return mutated  # No mutation
            
        # Determine number of mutations
        n_mutations = np.random.randint(1, max_mutations + 1)
        
        for _ in range(n_mutations):
            mutation_type = self._select_mutation_type()
            mutated._apply_mutation(mutation_type)
            
        return mutated
        
    def _select_mutation_type(self) -> MutationType:
        """
        Select a mutation type based on probabilities from profile.
        
        From foodotrope-drug.profile:
        - brain_percentmutateeffectaddneuron: 16
        - brain_percentmutateeffectremoveneuron: 6
        - brain_percentmutateeffectaddsynapse: 16
        - brain_percentmutateeffectremovesynapse: 6
        - brain_percentmutateeffectalterneuron: 12
        """
        mutation_probs = {
            MutationType.ADD_NEURON: 16,
            MutationType.REMOVE_NEURON: 6,
            MutationType.ADD_SYNAPSE: 16,
            MutationType.REMOVE_SYNAPSE: 6,
            MutationType.MODIFY_THRESHOLD: 12,
            MutationType.MODIFY_WEIGHT: 22,
            MutationType.MODIFY_PLASTICITY: 22,
        }
        
        types = list(mutation_probs.keys())
        probs = np.array(list(mutation_probs.values()))
        probs = probs / probs.sum()
        
        return np.random.choice(types, p=probs)
        
    def _apply_mutation(self, mutation_type: MutationType):
        """Apply a specific mutation."""
        if mutation_type == MutationType.ADD_NEURON:
            self._mutate_add_neuron()
        elif mutation_type == MutationType.REMOVE_NEURON:
            self._mutate_remove_neuron()
        elif mutation_type == MutationType.ADD_SYNAPSE:
            self._mutate_add_synapse()
        elif mutation_type == MutationType.REMOVE_SYNAPSE:
            self._mutate_remove_synapse()
        elif mutation_type == MutationType.MODIFY_THRESHOLD:
            self._mutate_modify_threshold()
        elif mutation_type == MutationType.MODIFY_WEIGHT:
            self._mutate_modify_weight()
        elif mutation_type == MutationType.MODIFY_PLASTICITY:
            self._mutate_modify_plasticity()
            
    def _mutate_add_neuron(self):
        """Add a new neuron."""
        # Don't exceed max neurons (250000 from profile)
        if len(self.neuron_genes) >= 250000:
            return
            
        new_gene = NeuronGene(
            neuron_id=self._next_neuron_id,
            threshold=700 + np.random.uniform(0, 8000),
            is_inhibitory=np.random.random() < 0.14,
            is_plastic=np.random.random() < 0.5
        )
        self.neuron_genes.append(new_gene)
        self._next_neuron_id += 1
        
    def _mutate_remove_neuron(self):
        """Remove a random neuron (not sensory/motor)."""
        removable = [g for g in self.neuron_genes if not (g.is_sensory or g.is_motor)]
        if not removable:
            return
            
        to_remove = np.random.choice(removable)
        self.neuron_genes.remove(to_remove)
        
        # Remove synapses connected to this neuron
        self.synapse_genes = [
            s for s in self.synapse_genes
            if s.pre_neuron_id != to_remove.neuron_id and s.post_neuron_id != to_remove.neuron_id
        ]
        
    def _mutate_add_synapse(self):
        """Add a new synapse."""
        if len(self.neuron_genes) < 2:
            return
            
        pre_id = np.random.choice([g.neuron_id for g in self.neuron_genes])
        post_id = np.random.choice([g.neuron_id for g in self.neuron_genes])
        
        if pre_id == post_id:
            return  # No self-connections
            
        new_synapse = SynapseGene(
            pre_neuron_id=pre_id,
            post_neuron_id=post_id,
            weight=np.random.uniform(0.5, 2.0),
            is_inhibitory=np.random.random() < 0.21,
            plasticity_rate=np.random.uniform(0.001, 0.05)
        )
        self.synapse_genes.append(new_synapse)
        
    def _mutate_remove_synapse(self):
        """Remove a random synapse."""
        if self.synapse_genes:
            idx = np.random.randint(0, len(self.synapse_genes))
            self.synapse_genes.pop(idx)
            
    def _mutate_modify_threshold(self):
        """Modify a random neuron's threshold."""
        if self.neuron_genes:
            gene = np.random.choice(self.neuron_genes)
            # Modify by ±20%
            change = gene.threshold * np.random.uniform(-0.2, 0.2)
            gene.threshold = max(258, min(1001, gene.threshold + change))  # From profile limits
            
    def _mutate_modify_weight(self):
        """Modify a random synapse weight."""
        if self.synapse_genes:
            gene = np.random.choice(self.synapse_genes)
            change = gene.weight * np.random.uniform(-0.3, 0.3)
            gene.weight = max(1.0, min(1000.0, gene.weight + change))
            
    def _mutate_modify_plasticity(self):
        """Modify a random synapse's plasticity rate."""
        if self.synapse_genes:
            gene = np.random.choice(self.synapse_genes)
            gene.plasticity_rate *= np.random.uniform(0.5, 2.0)
            gene.plasticity_rate = max(0.001, min(0.1, gene.plasticity_rate))
            
    def __repr__(self) -> str:
        return (
            f"Genotype(neurons={len(self.neuron_genes)}, "
            f"synapses={len(self.synapse_genes)}, "
            f"fitness={self.fitness:.2f}, gen={self.generation})"
        )
