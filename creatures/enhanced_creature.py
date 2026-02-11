"""
Enhanced multi-modal creature with generative capabilities.

Extends base Creature with:
- Visual sensing (retinal array)
- Audio generation (neural synthesis)
- Text generation (evolutionary markov)
- Visual pattern generation (brain-driven aesthetics)

Creates complete synesthetic beings.
"""

from typing import Optional
import numpy as np
from creatures.creature import Creature
from creatures.creature_senses import CreatureSenses
from creatures.creature_motors import CreatureMotors
from core.morphic.circuit8 import Circuit8


class EnhancedCreature(Creature):
    """
    Multi-modal creature with generative capabilities.
    
    Extends base Creature with sensory and motor systems
    that enable seeing, hearing, speaking, and thinking.
    
    Complete synesthetic being:
    - Sees through retinal array
    - Speaks through audio synthesis
    - Thinks through text generation
    - Expresses through visual patterns
    
    Attributes:
        senses: Sensory input systems
        motors: Motor/generative output systems
        last_audio: Most recent audio generation
        last_text: Most recent text generation
        last_pattern: Most recent visual pattern
    """
    
    def __init__(
        self,
        *args,
        enable_audio: bool = True,
        enable_text: bool = True,
        enable_visual_gen: bool = False,
        audio_mode: str = 'mixed',
        seed_text: Optional[str] = None,
        **kwargs
    ):
        """
        Create enhanced creature with multi-modal capabilities.
        
        Args:
            *args: Arguments for base Creature class
            enable_audio: Enable audio synthesis
            enable_text: Enable text generation
            enable_visual_gen: Enable visual pattern generation
            audio_mode: Audio synthesis mode
            seed_text: Initial text for markov chain (genetic language)
            **kwargs: Keyword arguments for base Creature class
        """
        # Initialize base creature
        super().__init__(*args, **kwargs)
        
        # Extract circuit8 dimensions for visual generation
        circuit8_width = 64
        circuit8_height = 48
        if hasattr(self, 'circuit8') and self.circuit8 is not None:
            circuit8_width = self.circuit8.width
            circuit8_height = self.circuit8.height
        
        # Initialize sensory systems
        self.senses = CreatureSenses(
            network=self.network,
            enable_vision=True,
            vision_resolution=100,  # Heritage: 100 sensors
            vision_width=circuit8_width,
            vision_height=circuit8_height
        )
        
        # Initialize motor systems
        self.motors = CreatureMotors(
            network=self.network,
            enable_audio=enable_audio,
            enable_text=enable_text,
            enable_visual_gen=enable_visual_gen,
            audio_mode=audio_mode,
            pattern_width=circuit8_width,
            pattern_height=circuit8_height
        )
        
        # Initialize language with seed text (genetic inheritance)
        if seed_text:
            self.motors.initialize_language(seed_text)
        
        # Storage for most recent generations
        self.last_audio: Optional[np.ndarray] = None
        self.last_text: str = ""
        self.last_pattern: Optional[np.ndarray] = None
        
        # Generation counters
        self.audio_generations = 0
        self.text_generations = 0
        self.pattern_generations = 0
    
    def update(self, dt: float = 1.0) -> bool:
        """
        Update creature with multi-modal sensing and generation.
        
        Extended update cycle:
        1. Read Circuit8 through retinal sensors
        2. Inject visual input into neural network
        3. Run base creature update (network, drugs, energy)
        4. Generate audio from brain state
        5. Generate text from markov chain
        6. Optionally generate visual pattern
        
        Args:
            dt: Time step in seconds
            
        Returns:
            True if creature survived, False if dead
        """
        # Step 1: Visual sensing
        if self.circuit8 is not None:
            visual_input = self.senses.process_visual_input(
                self.circuit8,
                x=int(self.x) % self.circuit8.width,
                y=int(self.y) % self.circuit8.height
            )
            self.senses.inject_visual_input(visual_input)
        
        # Step 2: Base creature update (network, drugs, energy, motors)
        alive = super().update(dt)
        if not alive:
            return False
        
        # Step 3: Audio generation (every timestep for continuous voice)
        self.last_audio = self.motors.generate_audio(duration_seconds=dt * 0.02)
        if self.last_audio is not None:
            self.audio_generations += 1
        
        # Step 4: Text generation (periodic - expensive)
        # Generate text every 10 timesteps
        if self.age % 10 == 0:
            text = self.motors.generate_text(max_length=10)
            if text:
                self.last_text = text
                self.text_generations += 1
        
        # Step 5: Visual pattern generation (if enabled - very expensive)
        # Generate pattern every 5 timesteps
        if self.age % 5 == 0:
            pattern = self.motors.generate_visual_pattern()
            if pattern is not None:
                self.last_pattern = pattern
                self.pattern_generations += 1
        
        return True
    
    def get_generation_statistics(self) -> dict:
        """
        Get statistics about creature's generative output.
        
        Returns:
            Dictionary with generation stats
        """
        stats = {
            'audio_generations': self.audio_generations,
            'text_generations': self.text_generations,
            'pattern_generations': self.pattern_generations,
        }
        
        # Add language statistics
        lang_stats = self.motors.get_language_statistics()
        stats.update(lang_stats)
        
        # Add audio statistics
        audio_stats = self.motors.get_audio_statistics()
        stats.update(audio_stats)
        
        return stats
    
    def get_current_thought(self) -> str:
        """
        Get creature's most recent generated text.
        
        Returns:
            Last generated text or empty string
        """
        return self.last_text
    
    def get_current_pattern(self) -> Optional[np.ndarray]:
        """
        Get creature's most recent visual pattern.
        
        Returns:
            Pattern array (height, width, 3) or None
        """
        return self.last_pattern
    
    def get_audio_buffer(self) -> Optional[np.ndarray]:
        """
        Get creature's most recent audio generation.
        
        Returns:
            Audio samples or None
        """
        return self.last_audio
