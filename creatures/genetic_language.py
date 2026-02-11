"""
Genetic language system for creatures.

Provides utilities for breeding and inheriting language
between creatures. Text becomes heritable like body/brain.

Language evolves through:
- Parent text inheritance
- Mutation of inherited text
- Cross-breeding of word pair scores
"""

import random
from typing import List, Optional
from generators.markov.mutations import (
    mutate_vowel,
    mutate_consonant,
    increment_letter,
    decrement_letter,
    transpose_letters,
    inject_random_char,
    delete_random_char
)


class GeneticLanguage:
    """
    Utilities for genetic text operations.
    
    Enables language to evolve through heredity:
    - Extract text from parent creatures
    - Mutate inherited text
    - Breed text from multiple parents
    
    Language becomes genetic material that evolves
    alongside body and brain.
    """
    
    @staticmethod
    def extract_parent_text(creature) -> str:
        """
        Extract text from creature for inheritance.
        
        Gets representative text from creature's markov chain
        that can be passed to offspring.
        
        Args:
            creature: Creature with motors.markov system
            
        Returns:
            Text string representing creature's language
        """
        if not hasattr(creature, 'motors') or creature.motors.markov is None:
            return ""
        
        # Generate sample text from creature's markov chain
        text_fragments = []
        for _ in range(5):  # Generate 5 fragments
            fragment = creature.motors.markov.generate_and_evolve(
                start_word=None,
                max_length=10
            )
            if fragment:
                text_fragments.append(fragment)
        
        return " ".join(text_fragments)
    
    @staticmethod
    def breed_text(parent1_text: str, parent2_text: str, mutation_rate: float = 0.3) -> str:
        """
        Breed text from two parents.
        
        Combines parental text and applies mutations,
        creating offspring language.
        
        Args:
            parent1_text: Text from first parent
            parent2_text: Text from second parent
            mutation_rate: Probability of mutation per operation
            
        Returns:
            Offspring text
        """
        # Combine parent texts
        words1 = parent1_text.split()
        words2 = parent2_text.split()
        
        # Crossover: randomly select words from each parent
        offspring_words = []
        max_len = max(len(words1), len(words2))
        
        for i in range(max_len):
            if i < len(words1) and i < len(words2):
                # Both parents have word at this position
                if random.random() < 0.5:
                    offspring_words.append(words1[i])
                else:
                    offspring_words.append(words2[i])
            elif i < len(words1):
                offspring_words.append(words1[i])
            elif i < len(words2):
                offspring_words.append(words2[i])
        
        offspring_text = " ".join(offspring_words)
        
        # Mutate offspring text
        return GeneticLanguage.mutate_text(offspring_text, mutation_rate)
    
    @staticmethod
    def mutate_text(text: str, mutation_rate: float = 0.3) -> str:
        """
        Apply mutation operators to text.
        
        Uses 7 mutation types from generators/markov/mutations.py
        
        Args:
            text: Input text to mutate
            mutation_rate: Probability of mutation per word
            
        Returns:
            Mutated text
        """
        if not text or random.random() > mutation_rate:
            return text
        
        # Choose random mutation operator
        mutation_ops = [
            mutate_vowel,
            mutate_consonant,
            increment_letter,
            decrement_letter,
            transpose_letters,
            inject_random_char,
            delete_random_char
        ]
        
        mutation_func = random.choice(mutation_ops)
        
        # Apply mutation to random word
        words = text.split()
        if not words:
            return text
        
        word_idx = random.randint(0, len(words) - 1)
        words[word_idx] = mutation_func(words[word_idx])
        
        return " ".join(words)
    
    @staticmethod
    def create_seed_text_from_parents(parents: List) -> str:
        """
        Create seed text from multiple parents.
        
        Extracts text from all parents and breeds
        combined offspring text.
        
        Args:
            parents: List of parent creatures
            
        Returns:
            Seed text for offspring
        """
        if not parents:
            return GeneticLanguage.generate_random_seed_text()
        
        # Extract text from all parents
        parent_texts = []
        for parent in parents:
            text = GeneticLanguage.extract_parent_text(parent)
            if text:
                parent_texts.append(text)
        
        if not parent_texts:
            return GeneticLanguage.generate_random_seed_text()
        
        # If single parent, mutate their text
        if len(parent_texts) == 1:
            return GeneticLanguage.mutate_text(parent_texts[0])
        
        # If multiple parents, breed their texts
        # Start with first two parents
        offspring_text = GeneticLanguage.breed_text(
            parent_texts[0],
            parent_texts[1]
        )
        
        # Add contributions from additional parents
        for parent_text in parent_texts[2:]:
            offspring_text = GeneticLanguage.breed_text(
                offspring_text,
                parent_text
            )
        
        return offspring_text
    
    @staticmethod
    def generate_random_seed_text() -> str:
        """
        Generate random seed text for creatures without parents.
        
        Creates initial language for first generation.
        
        Returns:
            Random seed text
        """
        # Simple random words - creatures will evolve their own language
        words = [
            "light", "dark", "move", "still", "up", "down",
            "near", "far", "hot", "cold", "eat", "grow",
            "sense", "feel", "think", "dream", "alive", "energy",
            "together", "apart", "rhythm", "pattern", "flow", "change"
        ]
        
        # Generate random text
        n_words = random.randint(10, 20)
        seed_words = [random.choice(words) for _ in range(n_words)]
        
        return " ".join(seed_words)
