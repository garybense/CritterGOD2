"""
Population: Manage evolutionary dynamics.

Based on critterding population management with energy-based selection.
"""

import numpy as np
from typing import List, Callable, Optional, Tuple
from .genotype import Genotype
from .phenotype import build_network_from_genotype
from core.neural import NeuralNetwork


class Population:
    """
    Manage a population of genotypes undergoing evolution.
    
    Based on critterding's population dynamics with:
    - Energy-based fitness
    - Tournament selection
    - Generational replacement
    - Statistical tracking
    
    Attributes:
        genotypes: Current population
        generation: Current generation number
        best_fitness_history: Track best fitness over time
        avg_fitness_history: Track average fitness
    """
    
    def __init__(
        self,
        population_size: int = 50,
        n_sensory: int = 10,
        n_motor: int = 10,
    ):
        """
        Initialize a random population.
        
        Args:
            population_size: Number of individuals
            n_sensory: Sensory neurons per individual
            n_motor: Motor neurons per individual
        """
        self.population_size = population_size
        self.n_sensory = n_sensory
        self.n_motor = n_motor
        self.generation = 0
        
        # Initialize random population
        self.genotypes: List[Genotype] = []
        for _ in range(population_size):
            genotype = Genotype.create_random(
                n_sensory=n_sensory,
                n_motor=n_motor,
                n_hidden_min=50,
                n_hidden_max=200
            )
            self.genotypes.append(genotype)
            
        # Statistics
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        
    def evaluate_fitness(
        self,
        fitness_function: Callable[[NeuralNetwork], float],
        verbose: bool = False
    ):
        """
        Evaluate fitness of all individuals in population.
        
        Args:
            fitness_function: Function that takes a neural network and returns fitness
            verbose: Print progress
        """
        for i, genotype in enumerate(self.genotypes):
            # Convert genotype to phenotype (neural network)
            network = build_network_from_genotype(genotype)
            
            # Evaluate fitness
            fitness = fitness_function(network)
            genotype.fitness = fitness
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(self.genotypes)} individuals...")
                
    def select_and_reproduce(
        self,
        selection_pressure: float = 0.5,
        mutation_rate: float = 0.5,
        elitism: int = 2,
    ):
        """
        Selection and reproduction to create next generation.
        
        Based on critterding's killhalfat mechanism and tournament selection.
        
        Args:
            selection_pressure: Fraction of population that survives (0.5 = kill half)
            mutation_rate: Probability of mutation
            elitism: Number of best individuals to preserve unchanged
        """
        # Sort by fitness (descending)
        self.genotypes.sort(key=lambda g: g.fitness, reverse=True)
        
        # Track statistics
        best_fitness = self.genotypes[0].fitness
        avg_fitness = np.mean([g.fitness for g in self.genotypes])
        diversity = self._calculate_diversity()
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.diversity_history.append(diversity)
        
        # Determine survivors
        n_survivors = max(elitism, int(self.population_size * selection_pressure))
        survivors = self.genotypes[:n_survivors]
        
        # Create next generation
        next_generation = []
        
        # Elitism: preserve best individuals
        for i in range(min(elitism, len(survivors))):
            next_generation.append(survivors[i])
            
        # Fill rest of population with offspring
        while len(next_generation) < self.population_size:
            # Tournament selection
            parent = self._tournament_select(survivors, tournament_size=3)
            
            # Create offspring through mutation
            offspring = parent.mutate(
                mutation_rate=mutation_rate,
                max_mutations=50
            )
            
            next_generation.append(offspring)
            
        self.genotypes = next_generation
        self.generation += 1
        
    def _tournament_select(
        self,
        candidates: List[Genotype],
        tournament_size: int = 3
    ) -> Genotype:
        """
        Tournament selection: pick best from random subset.
        
        Args:
            candidates: Pool to select from
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected genotype
        """
        tournament = np.random.choice(
            candidates,
            size=min(tournament_size, len(candidates)),
            replace=False
        )
        return max(tournament, key=lambda g: g.fitness)
        
    def _calculate_diversity(self) -> float:
        """
        Calculate genetic diversity (variation in network sizes).
        
        Returns:
            Diversity measure
        """
        sizes = [len(g.neuron_genes) for g in self.genotypes]
        return float(np.std(sizes))
        
    def get_best(self) -> Genotype:
        """Get the best individual in current population."""
        return max(self.genotypes, key=lambda g: g.fitness)
        
    def get_statistics(self) -> dict:
        """
        Get current population statistics.
        
        Returns:
            Dictionary of statistics
        """
        fitnesses = [g.fitness for g in self.genotypes]
        sizes = [len(g.neuron_genes) for g in self.genotypes]
        synapses = [len(g.synapse_genes) for g in self.genotypes]
        
        return {
            "generation": self.generation,
            "best_fitness": max(fitnesses) if fitnesses else 0.0,
            "avg_fitness": np.mean(fitnesses) if fitnesses else 0.0,
            "worst_fitness": min(fitnesses) if fitnesses else 0.0,
            "avg_neurons": np.mean(sizes) if sizes else 0.0,
            "avg_synapses": np.mean(synapses) if synapses else 0.0,
            "diversity": np.std(sizes) if sizes else 0.0,
        }
        
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"Population(gen={self.generation}, "
            f"size={len(self.genotypes)}, "
            f"best={stats['best_fitness']:.2f}, "
            f"avg={stats['avg_fitness']:.2f})"
        )
