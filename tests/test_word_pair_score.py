"""
Unit tests for WordPairScore evolutionary dynamics.
"""

from generators.markov.word_pair_score import WordPairScore


class TestWordPairScore:
    """Tests for word pair scoring system."""
    
    def test_initialization(self):
        """Test score initializes with heritage values."""
        score = WordPairScore("the", "cat")
        
        assert score.word1 == "the"
        assert score.word2 == "cat"
        assert score.energy == 1000.0
        assert score.attrep == 300.0
        assert score.last_used == 0.0
        assert score.use_count == 0
    
    def test_hit_depletes_score(self):
        """Test hit() depletes attrep score."""
        score = WordPairScore("the", "cat")
        initial_attrep = score.attrep
        
        score.hit(cost=200.0, time=1.0)
        
        assert score.attrep == initial_attrep - 200.0  # 300 - 200 = 100
        assert score.last_used == 1.0
        assert score.use_count == 1
    
    def test_multiple_hits(self):
        """Test multiple hits continue depleting."""
        score = WordPairScore("the", "cat")
        
        score.hit(cost=200.0)
        score.hit(cost=200.0)
        score.hit(cost=200.0)
        
        assert score.attrep == 300.0 - 600.0  # -300
        assert score.use_count == 3
    
    def test_decay_recovers_score(self):
        """Test decay allows recovery over time."""
        score = WordPairScore("the", "cat")
        
        # Deplete it
        score.hit(cost=200.0)
        assert score.attrep == 100.0
        
        # Decay should allow some recovery
        for _ in range(10):
            score.decay(rate=0.99)
        
        # Should be recovering toward 300
        assert score.attrep > 100.0
    
    def test_reward_novelty(self):
        """Test novelty reward increases score."""
        score = WordPairScore("the", "cat")
        initial_attrep = score.attrep
        
        score.reward_novelty(bonus=50.0)
        
        assert score.attrep == initial_attrep + 50.0  # 300 + 50 = 350
    
    def test_should_breed_threshold(self):
        """Test breeding threshold detection."""
        score = WordPairScore("the", "cat")
        
        # Initial state - should not breed
        assert not score.should_breed(threshold=1500.0)
        
        # Boost score above threshold
        score.attrep = 1600.0
        assert score.should_breed(threshold=1500.0)
    
    def test_should_kill_threshold(self):
        """Test kill threshold detection."""
        score = WordPairScore("the", "cat")
        
        # Initial state - should not kill
        assert not score.should_kill(threshold=0.1)
        
        # Deplete below threshold
        score.attrep = 0.05
        assert score.should_kill(threshold=0.1)
    
    def test_reset_after_breed(self):
        """Test breeding resets scores."""
        score = WordPairScore("the", "cat")
        
        # Modify scores
        score.attrep = 1600.0
        score.energy = 500.0
        
        # Reset
        score.reset_after_breed()
        
        assert score.attrep == 300.0
        assert score.energy == 1000.0
    
    def test_get_score(self):
        """Test combined score calculation."""
        score = WordPairScore("the", "cat")
        
        # Default: 1000 + 300 = 1300
        assert score.get_score() == 1300.0
        
        # After modification
        score.energy = 800.0
        score.attrep = 400.0
        assert score.get_score() == 1200.0
    
    def test_depletion_to_kill_threshold(self):
        """Test full depletion cycle to kill."""
        score = WordPairScore("the", "cat")
        
        # Repeatedly hit until below kill threshold
        for i in range(10):
            score.hit(cost=200.0, time=float(i))
            if score.should_kill(threshold=0.1):
                break
        
        # Should be killed
        assert score.should_kill(threshold=0.1)
        assert score.attrep <= 0.1
    
    def test_novelty_to_breed_threshold(self):
        """Test accumulation to breed threshold."""
        score = WordPairScore("the", "cat")
        
        # Repeatedly reward until breed threshold
        while not score.should_breed(threshold=1500.0):
            score.reward_novelty(bonus=100.0)
            if score.attrep > 2000.0:  # Safety limit
                break
        
        # Should be ready to breed
        assert score.should_breed(threshold=1500.0)
