"""
Word pair scoring system for evolutionary Markov chains.

Based on attract/repel dynamics from markov-attract-repel.py.
Pairs that are used frequently get depleted (repelled).
Rare pairs get rewarded (attracted).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WordPairScore:
    """
    Tracks evolutionary fitness score for a word pair.
    
    Heritage parameters from markov-attract-repel.py:
    - wordpairstartenergy = 1000.0
    - startAttrep = 300.0
    - attrepHit = 200.0 (cost per use)
    - wordscorethreshold = 1500 (breed)
    - kill threshold = 0.1
    
    Attributes:
        word1: First word in pair
        word2: Second word in pair
        energy: Current energy level (starts at 1000.0)
        attrep: Attract/repel score (starts at 300.0)
        last_used: Timestamp of last occurrence
        use_count: Number of times pair has been used
    """
    
    word1: str
    word2: str
    energy: float = 1000.0
    attrep: float = 300.0
    last_used: float = 0.0
    use_count: int = 0
    
    def hit(self, cost: float = 200.0, time: float = 0.0) -> None:
        """
        Apply usage cost when pair occurs in generation.
        
        This depletes the pair, making it less likely to breed.
        Frequent usage = depletion (repel).
        
        Args:
            cost: Energy cost per usage (attrepHit = 200.0 from heritage)
            time: Current timestamp
        """
        self.attrep -= cost
        self.last_used = time
        self.use_count += 1
    
    def decay(self, rate: float = 0.99) -> None:
        """
        Exponential decay of scores over time.
        
        This allows depleted pairs to recover if unused.
        
        Args:
            rate: Decay multiplier per timestep (0.99 = 1% decay)
        """
        # Attrep recovers toward starting value
        self.attrep *= rate
        if self.attrep < 300.0:
            # Slow recovery toward baseline
            self.attrep += (300.0 - self.attrep) * 0.01
    
    def reward_novelty(self, bonus: float = 50.0) -> None:
        """
        Reward rare/novel pairs with energy boost.
        
        Args:
            bonus: Energy bonus for novelty
        """
        self.attrep += bonus
    
    def should_breed(self, threshold: float = 1500.0) -> bool:
        """
        Check if pair has accumulated enough score to breed.
        
        High-scoring pairs (rare/novel) breed and mutate.
        
        Args:
            threshold: Score threshold for breeding (1500 from heritage)
            
        Returns:
            True if attrep >= threshold
        """
        return self.attrep >= threshold
    
    def should_kill(self, threshold: float = 0.1) -> bool:
        """
        Check if pair is depleted and should be removed.
        
        Low-scoring pairs (overused) die.
        
        Args:
            threshold: Minimum score to survive (0.1 from heritage)
            
        Returns:
            True if attrep <= threshold
        """
        return self.attrep <= threshold
    
    def reset_after_breed(self) -> None:
        """
        Reset score after breeding.
        
        Breeding creates mutations, so reset the pair to baseline.
        """
        self.attrep = 300.0
        self.energy = 1000.0
    
    def get_score(self) -> float:
        """
        Get combined fitness score.
        
        Returns:
            Combined energy + attrep score
        """
        return self.energy + self.attrep
