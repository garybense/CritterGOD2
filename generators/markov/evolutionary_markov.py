"""
Evolutionary Markov chain text generation system.

Based on markov-attract-repel.py from the critters codebase.
Integrates Markov chains, word pair scoring, and text mutations
to create self-organizing text through evolutionary dynamics.
"""

import random
from typing import Dict, List, Tuple, Optional
from generators.markov.markov_chain import MarkovChain
from generators.markov.word_pair_score import WordPairScore
from generators.markov import mutations


class EvolutionaryMarkov:
    """
    Self-organizing text generation through attract/repel dynamics.
    
    Word pairs accumulate scores based on usage:
    - Frequent usage → depletion → removal (death)
    - Rare/novel usage → reward → breeding (mutation + reinsertion)
    
    This creates an ecosystem of competing phrases that evolves over time.
    
    Attributes:
        chain: MarkovChain for text generation
        scores: Dictionary of (word1, word2) → WordPairScore
        corpus_lines: Original text lines for breeding
        time: Current simulation time
        stats: Generation statistics
        parameters: Evolutionary parameters
    """
    
    def __init__(
        self,
        wordpair_start_energy: float = 1000.0,
        start_attrep: float = 300.0,
        attrep_hit_cost: float = 200.0,
        breed_threshold: float = 1500.0,
        kill_threshold: float = 0.1,
        mutation_rate: float = 0.3,
        decay_rate: float = 0.99,
    ):
        """
        Initialize evolutionary Markov system.
        
        All parameters from markov-attract-repel.py heritage.
        
        Args:
            wordpair_start_energy: Initial energy for new pairs (1000.0)
            start_attrep: Initial attract/repel score (300.0)
            attrep_hit_cost: Energy cost per usage (200.0)
            breed_threshold: Score needed to breed (1500.0)
            kill_threshold: Score below which pair dies (0.1)
            mutation_rate: Probability of word mutation (0.3 = 30%)
            decay_rate: Score decay per timestep (0.99 = 1% decay)
        """
        self.chain = MarkovChain()
        self.scores: Dict[Tuple[str, str], WordPairScore] = {}
        self.corpus_lines: List[str] = []
        self.time: float = 0.0
        
        # Heritage parameters
        self.wordpair_start_energy = wordpair_start_energy
        self.start_attrep = start_attrep
        self.attrep_hit_cost = attrep_hit_cost
        self.breed_threshold = breed_threshold
        self.kill_threshold = kill_threshold
        self.mutation_rate = mutation_rate
        self.decay_rate = decay_rate
        
        # Statistics
        self.stats = {
            'generations': 0,
            'total_bred': 0,
            'total_killed': 0,
            'unique_pairs': 0,
        }
    
    def add_corpus(self, text: str) -> None:
        """
        Add seed text to markov chain and initialize pair scores.
        
        Args:
            text: Input corpus text
        """
        # Store original lines for breeding
        self.corpus_lines.extend([s.strip() for s in text.split('.') if s.strip()])
        
        # Build markov chain
        self.chain.add_text(text)
        
        # Initialize scores for all pairs
        for word1, word2 in self.chain.get_pairs():
            pair_key = (word1, word2)
            if pair_key not in self.scores:
                self.scores[pair_key] = WordPairScore(
                    word1=word1,
                    word2=word2,
                    energy=self.wordpair_start_energy,
                    attrep=self.start_attrep,
                )
        
        self.stats['unique_pairs'] = len(self.scores)
    
    def generate_and_evolve(self, start_word: Optional[str] = None, max_length: int = 50) -> str:
        """
        Generate text and update evolutionary dynamics.
        
        This is the main update cycle:
        1. Generate text from current chain
        2. Extract pairs used in generation
        3. Apply hit() to deplete used pairs
        4. Check for breeding opportunities (high scores)
        5. Check for kill candidates (low scores)
        
        Args:
            start_word: Word to start generation (None = random)
            max_length: Maximum words to generate
            
        Returns:
            Generated text string
        """
        # Generate text
        text = self.chain.generate(start_word=start_word, max_length=max_length)
        
        if not text:
            return ""
        
        # Extract pairs from generated text
        words = text.split()
        pairs_used = [(words[i], words[i+1]) for i in range(len(words)-1)]
        
        # Apply hit cost to used pairs
        for pair in pairs_used:
            if pair in self.scores:
                self.scores[pair].hit(cost=self.attrep_hit_cost, time=self.time)
        
        # Check for breeding opportunities
        to_breed = [
            pair for pair, score in self.scores.items()
            if score.should_breed(threshold=self.breed_threshold)
        ]
        for pair in to_breed:
            self.breed_pair(pair[0], pair[1])
        
        # Check for kill candidates
        to_kill = [
            pair for pair, score in self.scores.items()
            if score.should_kill(threshold=self.kill_threshold)
        ]
        for pair in to_kill:
            self.kill_pair(pair[0], pair[1])
        
        # Increment stats
        self.stats['generations'] += 1
        self.stats['unique_pairs'] = len(self.scores)
        
        return text
    
    def breed_pair(self, word1: str, word2: str) -> None:
        """
        Breed a high-scoring pair through mutation.
        
        1. Find a line from corpus containing this pair
        2. Mutate the line
        3. Add mutated line back to markov chain
        4. Reset pair score
        
        Args:
            word1: First word in pair
            word2: Second word in pair
        """
        pair_key = (word1, word2)
        
        # Find line containing this pair
        target_line = None
        for line in self.corpus_lines:
            if f"{word1} {word2}" in line:
                target_line = line
                break
        
        if not target_line:
            # Fallback: create simple line from pair
            target_line = f"{word1} {word2}"
        
        # Mutate the line
        mutated_line = mutations.mutate_line(target_line, self.mutation_rate)
        
        # Add mutated line to corpus and chain
        self.corpus_lines.append(mutated_line)
        self.chain.add_text(mutated_line)
        
        # Initialize scores for any new pairs
        for new_word1, new_word2 in self.chain.get_pairs():
            new_pair_key = (new_word1, new_word2)
            if new_pair_key not in self.scores:
                self.scores[new_pair_key] = WordPairScore(
                    word1=new_word1,
                    word2=new_word2,
                    energy=self.wordpair_start_energy,
                    attrep=self.start_attrep,
                )
        
        # Reset the bred pair
        if pair_key in self.scores:
            self.scores[pair_key].reset_after_breed()
        
        self.stats['total_bred'] += 1
    
    def kill_pair(self, word1: str, word2: str) -> None:
        """
        Remove a depleted pair from the system.
        
        Args:
            word1: First word in pair
            word2: Second word in pair
        """
        pair_key = (word1, word2)
        
        # Remove from markov chain
        self.chain.remove_pair(word1, word2)
        
        # Remove from scores
        if pair_key in self.scores:
            del self.scores[pair_key]
        
        self.stats['total_killed'] += 1
    
    def update(self, dt: float = 1.0) -> None:
        """
        Update evolutionary dynamics without generation.
        
        Applies decay to all pair scores, allowing recovery.
        
        Args:
            dt: Time delta
        """
        self.time += dt
        
        # Decay all scores
        for score in self.scores.values():
            score.decay(rate=self.decay_rate)
    
    def get_top_pairs(self, n: int = 10) -> List[Tuple[Tuple[str, str], float]]:
        """
        Get pairs with highest scores.
        
        Args:
            n: Number of pairs to return
            
        Returns:
            List of ((word1, word2), score) tuples
        """
        sorted_pairs = sorted(
            self.scores.items(),
            key=lambda item: item[1].get_score(),
            reverse=True
        )
        return [(pair, score.get_score()) for pair, score in sorted_pairs[:n]]
    
    def get_bottom_pairs(self, n: int = 10) -> List[Tuple[Tuple[str, str], float]]:
        """
        Get pairs with lowest scores (near death).
        
        Args:
            n: Number of pairs to return
            
        Returns:
            List of ((word1, word2), score) tuples
        """
        sorted_pairs = sorted(
            self.scores.items(),
            key=lambda item: item[1].get_score()
        )
        return [(pair, score.get_score()) for pair, score in sorted_pairs[:n]]
    
    def get_stats(self) -> Dict:
        """
        Get current evolutionary statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            'current_time': self.time,
            'average_score': sum(s.get_score() for s in self.scores.values()) / len(self.scores) if self.scores else 0,
        }
