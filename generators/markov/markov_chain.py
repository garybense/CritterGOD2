"""
Markov chain text generation.

Based on markov-attract-repel.py from the critters codebase.
This is the foundation for evolutionary text generation.
"""

import random
from typing import Dict, List, Tuple, Optional


class MarkovChain:
    """
    Basic Markov chain for text generation.
    
    Stores word transitions and generates text by randomly following chains.
    This is the base system upon which evolutionary dynamics are built.
    
    Attributes:
        chain: Dictionary mapping words to lists of possible next words
        start_words: List of words that can start a sentence
    """
    
    def __init__(self):
        self.chain: Dict[str, List[str]] = {}
        self.start_words: List[str] = []
        
    def add_text(self, text: str) -> None:
        """
        Build Markov chain from input text.
        
        Splits text into words and creates transitions between consecutive words.
        First word of each sentence is added to start_words.
        
        Args:
            text: Input text to process
        """
        # Split into sentences (basic approach - split on periods)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) < 2:
                continue
                
            # Add first word as potential start
            if words[0] not in self.start_words:
                self.start_words.append(words[0])
            
            # Build chain for consecutive word pairs
            for i in range(len(words) - 1):
                word1 = words[i]
                word2 = words[i + 1]
                
                if word1 not in self.chain:
                    self.chain[word1] = []
                self.chain[word1].append(word2)
    
    def generate(self, start_word: Optional[str] = None, max_length: int = 50) -> str:
        """
        Generate text from the Markov chain.
        
        Follows random transitions from word to word until max_length reached
        or no more transitions available.
        
        Args:
            start_word: Word to start generation from (None = random start)
            max_length: Maximum number of words to generate
            
        Returns:
            Generated text string
        """
        if not self.chain:
            return ""
        
        # Pick starting word
        if start_word is None or start_word not in self.chain:
            if not self.start_words:
                start_word = random.choice(list(self.chain.keys()))
            else:
                start_word = random.choice(self.start_words)
        
        words = [start_word]
        current = start_word
        
        # Generate by following random transitions
        for _ in range(max_length - 1):
            if current not in self.chain or not self.chain[current]:
                break
            next_word = random.choice(self.chain[current])
            words.append(next_word)
            current = next_word
        
        return ' '.join(words)
    
    def get_pairs(self) -> List[Tuple[str, str]]:
        """
        Extract all word pairs in the chain.
        
        Returns:
            List of (word1, word2) tuples representing transitions
        """
        pairs = []
        for word1, next_words in self.chain.items():
            for word2 in next_words:
                pairs.append((word1, word2))
        return pairs
    
    def remove_pair(self, word1: str, word2: str) -> None:
        """
        Remove a specific word pair from the chain.
        
        Used when evolutionary system "kills" depleted pairs.
        
        Args:
            word1: First word in pair
            word2: Second word in pair
        """
        if word1 in self.chain and word2 in self.chain[word1]:
            self.chain[word1].remove(word2)
            
            # Clean up empty chains
            if not self.chain[word1]:
                del self.chain[word1]
    
    def get_all_words(self) -> List[str]:
        """
        Get all unique words in the chain.
        
        Returns:
            List of all words that appear in the chain
        """
        words = set(self.chain.keys())
        for next_words in self.chain.values():
            words.update(next_words)
        return list(words)
    
    def has_pair(self, word1: str, word2: str) -> bool:
        """
        Check if a specific word pair exists in the chain.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            True if word1 can transition to word2
        """
        return word1 in self.chain and word2 in self.chain[word1]
