"""
Ultimate multi-modal collective creature.

Combines ALL features:
- Collective intelligence (social learning, behavior broadcasting)
- Multi-modal generation (audio, text, visual patterns)
- Retinal vision (seeing the shared hallucination)
- Procedural 3D bodies
- Physics-based movement
- Resource-seeking behavior
- Addiction and drug effects

The pinnacle of artificial life - creatures that see, speak, think,
learn socially, and exist in a shared telepathic dream space.
"""

from typing import Optional
import numpy as np
from creatures.collective_creature import CollectiveCreature
from creatures.creature_senses import CreatureSenses
from creatures.creature_motors import CreatureMotors
from core.evolution.genotype import Genotype
from core.morphology.body_genotype import BodyGenotype
from core.morphic.circuit8 import Circuit8
from core.physics.physics_world import PhysicsWorld
from core.collective import CollectiveMemory


class UltimateCreature(CollectiveCreature):
    """
    Ultimate creature with ALL capabilities integrated.
    
    Combines:
    - CollectiveCreature: Social learning, behavior broadcasting
    - EnhancedCreature: Audio, text, visual generation
    - Retinal vision system
    - Complete synesthetic being in shared reality
    
    Attributes:
        senses: Sensory input systems (retinal vision)
        motors: Generative output systems (audio, text, patterns)
        last_text: Most recent thought
        last_audio: Most recent voice
        last_pattern: Most recent visual expression
    """
    
    def __init__(
        self,
        genotype: Genotype,
        body: Optional[BodyGenotype] = None,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        initial_energy: float = 1000000.0,
        circuit8: Optional[Circuit8] = None,
        physics_world: Optional[PhysicsWorld] = None,
        collective_memory: Optional[CollectiveMemory] = None,
        creature_id: Optional[int] = None,
        # Multi-modal parameters
        enable_audio: bool = False,  # Start disabled for performance
        enable_text: bool = True,   # Thoughts enabled by default
        enable_visual_gen: bool = False,  # Patterns disabled for performance
        audio_mode: str = 'mixed',
        seed_text: Optional[str] = None,
        **kwargs
    ):
        """
        Create ultimate creature with all capabilities.
        
        Args:
            genotype: Neural network genotype
            body: Body genotype
            x, y, z: Initial position
            initial_energy: Starting energy
            circuit8: Shared telepathic canvas
            physics_world: Physics world
            collective_memory: Shared collective memory
            creature_id: Unique creature ID
            enable_audio: Enable audio synthesis (expensive)
            enable_text: Enable text generation (thoughts)
            enable_visual_gen: Enable visual pattern generation (expensive)
            audio_mode: Audio synthesis mode
            seed_text: Initial text for markov chain (genetic language)
            **kwargs: Additional arguments
        """
        # Initialize collective creature base
        super().__init__(
            genotype=genotype,
            body=body,
            x=x,
            y=y,
            z=z,
            initial_energy=initial_energy,
            circuit8=circuit8,
            physics_world=physics_world,
            collective_memory=collective_memory,
            creature_id=creature_id,
            **kwargs
        )
        
        # Extract circuit8 dimensions for visual systems
        circuit8_width = 64
        circuit8_height = 48
        if circuit8 is not None:
            circuit8_width = circuit8.width
            circuit8_height = circuit8.height
        
        # Initialize sensory systems (retinal vision)
        # NOTE: Vision temporarily disabled due to update() signature conflict
        # Will re-enable after refactoring update chain
        self.senses = CreatureSenses(
            network=self.network,
            enable_vision=False,  # TODO: Re-enable after fixing update chain
            vision_resolution=100,
            vision_width=circuit8_width,
            vision_height=circuit8_height
        )
        
        # Initialize motor/generative systems
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
    
    def update(self, dt: float, resource_manager=None) -> bool:
        """
        Update creature with complete multi-modal cycle.
        
        Extended update cycle:
        1. Read Circuit8 through retinal sensors (VISION)
        2. Inject visual input into neural network
        3. Run collective creature update (physics, behavior, social learning)
        4. Generate audio from brain state (VOICE)
        5. Generate text from markov chain (THOUGHTS)
        6. Optionally generate visual pattern (EXPRESSION)
        7. Write visual pattern to Circuit8 (shared hallucination)
        
        Args:
            dt: Time step
            resource_manager: Resource manager for consumption
            
        Returns:
            True if creature survived, False if dead
        """
        # Step 1: Visual sensing - READ the shared hallucination
        # TODO: Re-enable after fixing update chain signature conflicts
        # if self.circuit8 is not None:
        #     visual_input = self.senses.process_visual_input(
        #         self.circuit8,
        #         x=int(self.x) % self.circuit8.width,
        #         y=int(self.y) % self.circuit8.height
        #     )
        #     self.senses.inject_visual_input(visual_input)
        
        # Step 2: Base collective creature update
        # (includes physics, behavior, social learning, broadcasting)
        alive = super().update(dt, resource_manager)
        if not alive:
            return False
        
        # Step 3: Audio generation (continuous voice)
        if self.age % 2 == 0:  # Every 2 timesteps to reduce overhead
            self.last_audio = self.motors.generate_audio(duration_seconds=dt * 0.02)
            if self.last_audio is not None:
                self.audio_generations += 1
        
        # Step 4: Text generation (periodic thoughts)
        # Generate text every 20 timesteps (expensive)
        if self.age % 20 == 0:
            text = self.motors.generate_text(max_length=8)
            if text:
                self.last_text = text
                self.text_generations += 1
        
        # Step 5: Visual pattern generation (if enabled - very expensive)
        # Generate pattern every 10 timesteps
        if self.age % 10 == 0:
            pattern = self.motors.generate_visual_pattern()
            if pattern is not None:
                self.last_pattern = pattern
                self.pattern_generations += 1
                
                # Step 6: Write pattern to Circuit8 - CONTRIBUTE to shared hallucination
                if self.circuit8 is not None:
                    self._write_pattern_to_circuit8(pattern)
        
        return True
    
    def _write_pattern_to_circuit8(self, pattern: np.ndarray):
        """
        Write generated pattern to Circuit8 telepathic canvas.
        
        Creates the shared visual hallucination space.
        
        Args:
            pattern: RGB pattern array (height, width, 3)
        """
        if self.circuit8 is None:
            return
        
        # Write pattern centered on creature's position
        pattern_h, pattern_w = pattern.shape[:2]
        start_x = int(self.x) % self.circuit8.width
        start_y = int(self.y) % self.circuit8.height
        
        # Write with blending to create layered hallucination
        for py in range(min(pattern_h, self.circuit8.height)):
            for px in range(min(pattern_w, self.circuit8.width)):
                cx = (start_x + px) % self.circuit8.width
                cy = (start_y + py) % self.circuit8.height
                
                r, g, b = pattern[py, px]
                self.circuit8.write_pixel(cx, cy, int(r), int(g), int(b), blend=True)
    
    def get_current_thought(self) -> str:
        """Get creature's most recent thought."""
        return self.last_text
    
    def get_audio_buffer(self) -> Optional[np.ndarray]:
        """Get creature's most recent audio generation."""
        return self.last_audio
    
    def get_current_pattern(self) -> Optional[np.ndarray]:
        """Get creature's most recent visual pattern."""
        return self.last_pattern
    
    def get_generation_statistics(self) -> dict:
        """
        Get complete statistics about creature's generative output.
        
        Returns:
            Dictionary with all generation stats
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
    
    def to_dict(self) -> dict:
        """Export complete creature state including generative systems."""
        base_dict = super().to_dict()
        
        # Add generative information
        base_dict['generation'] = self.get_generation_statistics()
        base_dict['current_thought'] = self.last_text
        
        return base_dict


def create_ultimate_creatures(
    n_creatures: int,
    physics_world: PhysicsWorld,
    circuit8: Circuit8,
    collective_memory: CollectiveMemory,
    world_bounds: tuple = (-250.0, -250.0, 250.0, 250.0),
    enable_audio: bool = False,
    enable_text: bool = True,
    enable_visual_gen: bool = False
) -> list:
    """
    Create multiple ultimate creatures sharing reality.
    
    Args:
        n_creatures: Number of creatures
        physics_world: Physics world
        circuit8: Telepathic canvas (shared hallucination)
        collective_memory: Shared collective memory
        world_bounds: (min_x, min_y, max_x, max_y)
        enable_audio: Enable audio synthesis
        enable_text: Enable text generation (thoughts)
        enable_visual_gen: Enable visual pattern generation
        
    Returns:
        List of ultimate creatures
    """
    from creatures.genetic_language import GeneticLanguage
    
    creatures = []
    min_x, min_y, max_x, max_y = world_bounds
    
    for i in range(n_creatures):
        # Random position
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        z = 10.0
        
        # Random genotype
        genotype = Genotype.create_random(
            n_sensory=100,  # Will be overridden by retinal array
            n_motor=20,
            n_hidden_min=80,
            n_hidden_max=200,
            synapses_per_neuron=30
        )
        
        # Random body
        body = BodyGenotype.create_random()
        
        # Random seed text (genetic language)
        seed_text = GeneticLanguage.generate_random_seed_text() if enable_text else None
        
        # Create ultimate creature
        creature = UltimateCreature(
            genotype=genotype,
            body=body,
            x=x,
            y=y,
            z=z,
            initial_energy=1000000.0,
            circuit8=circuit8,
            physics_world=physics_world,
            collective_memory=collective_memory,
            creature_id=i,
            enable_audio=enable_audio,
            enable_text=enable_text,
            enable_visual_gen=enable_visual_gen,
            seed_text=seed_text
        )
        
        creatures.append(creature)
    
    return creatures
