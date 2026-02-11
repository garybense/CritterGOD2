"""
Motor and generative output systems for creatures.

Extends creatures with audio, text, and visual generation capabilities,
creating complete synesthetic beings.
"""

import numpy as np
from typing import Optional
from core.neural.network import NeuralNetwork
from generators.audio.neural_audio import NeuralAudioSynthesizer
from generators.markov.evolutionary_markov import EvolutionaryMarkov
from generators.visual.pattern_generators import PatternGenerator
from generators.visual.neural_parameters import NeuralPatternMapper


class CreatureMotors:
    """
    Generative motor systems for creatures.
    
    Integrates multi-modal generation:
    - Audio synthesis from neural activity
    - Text generation from evolutionary markov chains
    - Visual pattern generation from brain state
    
    Creates synesthetic beings where all generation
    emerges from unified neural source.
    
    Attributes:
        network: Neural network driving generation
        audio_synth: Audio synthesizer (if enabled)
        markov: Evolutionary markov chain (if enabled)
        pattern_gen: Visual pattern generator (if enabled)
        pattern_mapper: Neural parameter mapper (if enabled)
    """
    
    def __init__(
        self,
        network: NeuralNetwork,
        enable_audio: bool = True,
        enable_text: bool = True,
        enable_visual_gen: bool = False,  # Default off (computationally expensive)
        audio_mode: str = 'mixed',
        pattern_width: int = 64,
        pattern_height: int = 48
    ):
        """
        Initialize creature motor/generative systems.
        
        Args:
            network: Neural network for generation
            enable_audio: Enable audio synthesis
            enable_text: Enable text generation
            enable_visual_gen: Enable visual pattern generation
            audio_mode: Audio synthesis mode ('potential', 'firing', 'mixed')
            pattern_width: Generated pattern width (default: 64 like Circuit8)
            pattern_height: Generated pattern height (default: 48 like Circuit8)
        """
        self.network = network
        
        # Audio generation
        self.audio_synth: Optional[NeuralAudioSynthesizer] = None
        if enable_audio:
            self.audio_synth = NeuralAudioSynthesizer(
                sample_rate=44100,
                buffer_size=2048,  # Smaller buffer for lower latency
                mode=audio_mode,
                amplitude_scale=0.2  # Quieter for multiple creatures
            )
        
        # Text generation
        self.markov: Optional[EvolutionaryMarkov] = None
        if enable_text:
            self.markov = EvolutionaryMarkov(
                wordpair_start_energy=1000.0,
                start_attrep=300.0,
                attrep_hit_cost=200.0,
                breed_threshold=1500.0,
                kill_threshold=0.1,
                mutation_rate=0.3,
            )
        
        # Visual pattern generation
        self.pattern_gen: Optional[PatternGenerator] = None
        self.pattern_mapper: Optional[NeuralPatternMapper] = None
        if enable_visual_gen:
            self.pattern_gen = PatternGenerator(
                width=pattern_width,
                height=pattern_height
            )
            self.pattern_mapper = NeuralPatternMapper()
    
    def generate_audio(self, duration_seconds: float = 0.02) -> Optional[np.ndarray]:
        """
        Generate audio from current brain state.
        
        Creates creature's "voice" - unique acoustic signature
        from neural activity patterns.
        
        Args:
            duration_seconds: Audio duration to generate
            
        Returns:
            Audio samples (float32 array) or None if audio disabled
        """
        if self.audio_synth is None:
            return None
        
        return self.audio_synth.synthesize_from_network(
            self.network,
            duration_seconds=duration_seconds
        )
    
    def generate_text(self, max_length: int = 10) -> Optional[str]:
        """
        Generate text from creature's markov chain.
        
        Creates creature's "thoughts" - internal language
        that evolves through attract/repel dynamics.
        
        Args:
            max_length: Maximum words to generate
            
        Returns:
            Generated text string or None if text disabled
        """
        if self.markov is None:
            return None
        
        # Only generate if chain has content
        if not self.markov.chain.chain:
            return ""
        
        return self.markov.generate_and_evolve(
            start_word=None,
            max_length=max_length
        )
    
    def generate_visual_pattern(self) -> Optional[np.ndarray]:
        """
        Generate visual pattern from current brain state.
        
        Creates creature's visual "expression" - unique
        aesthetic signature from neural parameters.
        
        Returns:
            RGB pattern array (height, width, 3) or None if disabled
        """
        if self.pattern_gen is None or self.pattern_mapper is None:
            return None
        
        # Extract parameters from network state
        params = self.pattern_mapper.extract_parameters(self.network)
        
        # Generate pattern
        return self.pattern_gen.generate_pattern(params)
    
    def initialize_language(self, seed_text: str) -> None:
        """
        Initialize creature's language with seed text.
        
        Typically called with genetic text fragment.
        
        Args:
            seed_text: Initial text corpus for markov chain
        """
        if self.markov is not None and seed_text:
            self.markov.add_corpus(seed_text)
    
    def get_language_statistics(self) -> dict:
        """
        Get statistics about creature's language.
        
        Returns:
            Dictionary with language stats
        """
        if self.markov is None:
            return {'text_enabled': False}
        
        stats = self.markov.get_stats()
        return {
            'text_enabled': True,
            **stats
        }
    
    def get_audio_statistics(self) -> dict:
        """
        Get statistics about creature's audio.
        
        Returns:
            Dictionary with audio stats
        """
        if self.audio_synth is None:
            return {'audio_enabled': False}
        
        return {
            'audio_enabled': True,
            'mode': self.audio_synth.mode,
            'sample_rate': self.audio_synth.sample_rate,
        }
