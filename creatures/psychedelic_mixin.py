"""
Psychedelic Vision Mixin

Adds retinal vision and visual pattern generation to creatures
without breaking the inheritance chain. This creates the complete
feedback loop:

PERCEIVE (retinal vision) → PROCESS (neural network) → 
CREATE (pattern generation) → BROADCAST (Circuit8) →
PERCEIVE (other creatures see patterns) → ...

The shared hallucination space emerges from this loop,
especially amplified by psychedelic drugs.
"""

import numpy as np
from typing import Optional
from core.morphic.circuit8 import Circuit8
from generators.visual.pattern_generators import PatternGenerator
from generators.visual.neural_parameters import NeuralPatternMapper


class PsychedelicVisionMixin:
    """
    Mixin to add psychedelic vision capabilities to creatures.
    
    Provides:
    - Retinal vision (reading Circuit8)
    - Visual pattern generation (writing to Circuit8)
    - Drug-amplified visual effects
    
    Can be safely mixed into any creature class.
    """
    
    def init_psychedelic_vision(
        self, 
        enable_vision: bool = False,
        enable_patterns: bool = False
    ):
        """
        Initialize psychedelic vision systems.
        
        Call this from creature __init__ AFTER super().__init__()
        
        Args:
            enable_vision: Enable retinal vision
            enable_patterns: Enable pattern generation
        """
        # Pattern generation
        self.pattern_gen: Optional[PatternGenerator] = None
        self.pattern_mapper: Optional[NeuralPatternMapper] = None
        self.last_pattern: Optional[np.ndarray] = None
        self.pattern_generation_enabled = enable_patterns
        
        if enable_patterns and hasattr(self, 'circuit8') and self.circuit8:
            self.pattern_gen = PatternGenerator(
                width=self.circuit8.width,
                height=self.circuit8.height
            )
            self.pattern_mapper = NeuralPatternMapper()
        
        # Vision flag (retinal sensors would require modifying network)
        self.vision_enabled = enable_vision
        self.retinal_active = False  # Track if retinal sensors are wired
    
    def generate_psychedelic_pattern(self) -> Optional[np.ndarray]:
        """
        Generate visual pattern from current brain state.
        
        Drug effects amplify pattern intensity and modify parameters.
        
        Returns:
            RGB pattern array or None if disabled
        """
        if not self.pattern_generation_enabled or self.pattern_gen is None:
            return None
        
        if not hasattr(self, 'network') or not hasattr(self, 'drugs'):
            return None
        
        # Extract base parameters from network
        params = self.pattern_mapper.extract_parameters(self.network)
        
        # Modify parameters based on drug levels
        trip_level = np.sum(self.drugs.tripping) / self.drugs.max_trip
        
        if trip_level > 0.01:  # Tripping
            # Amplify pattern intensity
            params['amplitude'] *= (1.0 + trip_level * 3.0)
            
            # Increase frequency (more complex patterns)
            params['frequency'] *= (1.0 + trip_level * 2.0)
            
            # More phases (richer visuals)
            params['phase_offset'] += trip_level * np.pi
            
            # Potentiator creates especially wild patterns
            potentiator = self.drugs.tripping[4] / self.drugs.max_trip
            if potentiator > 0.01:
                params['symmetry'] = min(8, int(params.get('symmetry', 4) * (1 + potentiator * 4)))
        
        # Generate pattern
        pattern = self.pattern_gen.generate_pattern(params)
        self.last_pattern = pattern
        
        return pattern
    
    def write_pattern_to_circuit8(self, pattern: np.ndarray, blend: bool = True):
        """
        Write generated pattern to Circuit8 telepathic canvas.
        
        Creates the shared visual hallucination space.
        
        Args:
            pattern: RGB pattern array (height, width, 3)
            blend: Whether to blend with existing content
        """
        if not hasattr(self, 'circuit8') or self.circuit8 is None:
            return
        
        if pattern is None:
            return
        
        pattern_h, pattern_w = pattern.shape[:2]
        
        # Write pattern centered on creature's position
        start_x = int(self.x) % self.circuit8.width
        start_y = int(self.y) % self.circuit8.height
        
        # Drug level affects write intensity
        trip_level = 1.0
        if hasattr(self, 'drugs'):
            trip_level = 1.0 + (np.sum(self.drugs.tripping) / self.drugs.max_trip) * 2.0
        
        # Write with intensity scaling
        for py in range(min(pattern_h, self.circuit8.height)):
            for px in range(min(pattern_w, self.circuit8.width)):
                cx = (start_x + px) % self.circuit8.width
                cy = (start_y + py) % self.circuit8.height
                
                r, g, b = pattern[py, px]
                
                # Scale by trip level
                r = int(np.clip(r * trip_level, 0, 255))
                g = int(np.clip(g * trip_level, 0, 255))
                b = int(np.clip(b * trip_level, 0, 255))
                
                self.circuit8.write_pixel(cx, cy, r, g, b, blend=blend)
    
    def update_psychedelic_vision(self, timestep: int):
        """
        Update psychedelic vision systems.
        
        Call this from creature update() method.
        
        Args:
            timestep: Current timestep counter
        """
        # Generate patterns periodically (expensive operation)
        if self.pattern_generation_enabled and timestep % 10 == 0:
            pattern = self.generate_psychedelic_pattern()
            if pattern is not None:
                self.write_pattern_to_circuit8(pattern, blend=True)
