"""
Unit tests for MarkovChain class.
"""

import pytest
from generators.markov import MarkovChain


class TestMarkovChain:
    """Tests for basic Markov chain functionality."""
    
    def test_initialization(self):
        """Test chain initializes empty."""
        chain = MarkovChain()
        assert len(chain.chain) == 0
        assert len(chain.start_words) == 0
    
    def test_add_text_simple(self):
        """Test adding simple text builds chain correctly."""
        chain = MarkovChain()
        chain.add_text("the cat sat on the mat")
        
        assert "the" in chain.chain
        assert "cat" in chain.chain["the"]
        assert "mat" in chain.chain["the"]
        assert "sat" in chain.chain["cat"]
        assert "on" in chain.chain["sat"]
        assert "the" in chain.chain["on"]
    
    def test_add_text_with_sentences(self):
        """Test adding text with multiple sentences."""
        chain = MarkovChain()
        chain.add_text("Hello world. Goodbye world.")
        
        assert "Hello" in chain.start_words
        assert "Goodbye" in chain.start_words
        assert "world" in chain.chain["Hello"]
        assert "world" in chain.chain["Goodbye"]
    
    def test_generate_returns_text(self):
        """Test generation produces non-empty text."""
        chain = MarkovChain()
        chain.add_text("the quick brown fox jumps over the lazy dog")
        
        text = chain.generate(max_length=5)
        assert len(text) > 0
        words = text.split()
        assert len(words) <= 5
    
    def test_generate_with_start_word(self):
        """Test generation starts with specified word."""
        chain = MarkovChain()
        chain.add_text("the cat sat. the dog ran.")
        
        text = chain.generate(start_word="the")
        assert text.startswith("the")
    
    def test_generate_empty_chain(self):
        """Test generation on empty chain returns empty string."""
        chain = MarkovChain()
        text = chain.generate()
        assert text == ""
    
    def test_get_pairs(self):
        """Test extracting all word pairs."""
        chain = MarkovChain()
        chain.add_text("the cat sat on the mat")
        
        pairs = chain.get_pairs()
        assert ("the", "cat") in pairs
        assert ("cat", "sat") in pairs
        assert ("sat", "on") in pairs
        assert ("on", "the") in pairs
        assert ("the", "mat") in pairs
    
    def test_remove_pair(self):
        """Test removing a word pair."""
        chain = MarkovChain()
        chain.add_text("the cat sat. the dog ran.")
        
        # Verify pair exists
        assert chain.has_pair("the", "cat")
        
        # Remove it
        chain.remove_pair("the", "cat")
        
        # Verify it's gone
        assert not chain.has_pair("the", "cat")
        
        # But other "the" transitions still exist
        assert chain.has_pair("the", "dog")
    
    def test_remove_pair_cleans_empty_chains(self):
        """Test removing last pair from a word removes the word."""
        chain = MarkovChain()
        chain.add_text("hello world")
        
        # Only one transition from "hello"
        assert "hello" in chain.chain
        
        chain.remove_pair("hello", "world")
        
        # "hello" should be removed since it has no more transitions
        assert "hello" not in chain.chain
    
    def test_get_all_words(self):
        """Test getting all unique words."""
        chain = MarkovChain()
        chain.add_text("the cat sat on the mat")
        
        words = chain.get_all_words()
        assert "the" in words
        assert "cat" in words
        assert "sat" in words
        assert "on" in words
        assert "mat" in words
        assert len(words) == 5  # Unique words
    
    def test_has_pair(self):
        """Test checking if pair exists."""
        chain = MarkovChain()
        chain.add_text("the cat sat")
        
        assert chain.has_pair("the", "cat")
        assert chain.has_pair("cat", "sat")
        assert not chain.has_pair("cat", "the")
        assert not chain.has_pair("foo", "bar")
    
    def test_multiple_transitions_same_word(self):
        """Test word can have multiple possible next words."""
        chain = MarkovChain()
        chain.add_text("the cat ran. the dog ran. the bird flew.")
        
        # "the" should have multiple possible next words
        assert "cat" in chain.chain["the"]
        assert "dog" in chain.chain["the"]
        assert "bird" in chain.chain["the"]
        assert len(chain.chain["the"]) == 3
