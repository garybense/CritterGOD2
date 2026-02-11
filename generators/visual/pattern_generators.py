"""
Procedural pattern generation using mathematical functions.

Based on SDL neural visualizers (looser.c, xesu.c, cdd.c, etc.).
Wallpaper patterns emerge from trigonometric combinations driven by neural ratios.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PatternParams:
    """
    Parameters for pattern generation.
    
    In heritage SDL visualizers, these are computed from neuron threshold ratios:
    - tanamp = brain[1591][1] / brain[1425][1]
    - sinamp = brain[1573][1] / brain[1950][1]
    - etc.
    
    Attributes:
        sin_amplitude: Amplitude for sine wave component
        sin_frequency: Frequency for sine wave
        cos_amplitude: Amplitude for cosine wave component
        cos_frequency: Frequency for cosine wave
        tan_amplitude: Amplitude for tangent wave component
        tan_frequency: Frequency for tangent wave
        phase_offset: Global phase offset
        color_rotation: Rotation in RGB color space (0-1)
        brightness: Overall brightness multiplier (0-1)
    """
    
    sin_amplitude: float = 1.0
    sin_frequency: float = 0.01
    cos_amplitude: float = 1.0
    cos_frequency: float = 0.01
    tan_amplitude: float = 0.5
    tan_frequency: float = 0.005
    phase_offset: float = 0.0
    color_rotation: float = 0.0
    brightness: float = 1.0


class PatternGenerator:
    """
    Generates procedural visual patterns from mathematical functions.
    
    Heritage from SDL neural visualizers where wallpaper patterns are computed
    from trigonometric functions using parameters derived from neural state.
    
    Creates emergent aesthetics through:
    - Layered sine/cosine/tangent waves
    - Color mapping from function values
    - Parameter evolution from neural network
    """
    
    def __init__(self, width: int = 640, height: int = 480):
        """
        Initialize pattern generator.
        
        Args:
            width: Pattern width in pixels
            height: Pattern height in pixels
        """
        self.width = width
        self.height = height
        
        # Precompute coordinate grids for efficiency
        self.x_coords, self.y_coords = np.meshgrid(
            np.arange(width),
            np.arange(height)
        )
    
    def generate_pattern(self, params: PatternParams) -> np.ndarray:
        """
        Generate RGB pattern from parameters.
        
        Heritage approach: combine multiple trigonometric functions
        and map results to RGB color space.
        
        Args:
            params: Pattern generation parameters
            
        Returns:
            RGB array of shape (height, width, 3) with uint8 values
        """
        # Compute base wave value from trigonometric combination
        wave = self._compute_wave_function(params)
        
        # Map wave to RGB channels with phase offsets
        rgb = self._wave_to_rgb(wave, params)
        
        # Apply brightness
        rgb = np.clip(rgb * params.brightness, 0, 255).astype(np.uint8)
        
        return rgb
    
    def _compute_wave_function(self, params: PatternParams) -> np.ndarray:
        """
        Compute combined wave function from trigonometric components.
        
        Heritage: Complex combinations like in looser.c:
        val = sin(x * freq + phase) * amp + cos(y * freq) * amp + ...
        
        Args:
            params: Pattern parameters
            
        Returns:
            Wave values array of shape (height, width)
        """
        # Sine component (x-axis)
        sin_wave = (
            np.sin(self.x_coords * params.sin_frequency + params.phase_offset) *
            params.sin_amplitude
        )
        
        # Cosine component (y-axis)
        cos_wave = (
            np.cos(self.y_coords * params.cos_frequency) *
            params.cos_amplitude
        )
        
        # Tangent component (diagonal)
        # Use small epsilon to avoid division by zero
        tan_input = (self.x_coords + self.y_coords) * params.tan_frequency
        tan_wave = np.clip(
            np.tan(tan_input) * params.tan_amplitude,
            -10.0, 10.0  # Clamp to avoid extreme values
        )
        
        # Combine all components
        combined = sin_wave + cos_wave + tan_wave
        
        return combined
    
    def _wave_to_rgb(self, wave: np.ndarray, params: PatternParams) -> np.ndarray:
        """
        Convert wave function to RGB color values.
        
        Heritage approach: Use different phase offsets for R, G, B channels
        to create colorful patterns from single wave function.
        
        Args:
            wave: Wave function values
            params: Pattern parameters for color rotation
            
        Returns:
            RGB array of shape (height, width, 3)
        """
        # Apply color rotation offset to each channel
        rotation = params.color_rotation * 2 * np.pi
        
        # Red channel: base wave
        r = (np.sin(wave + rotation) + 1.0) * 127.5
        
        # Green channel: phase offset by 120 degrees
        g = (np.cos(wave * 1.3 + rotation + np.pi * 2/3) + 1.0) * 127.5
        
        # Blue channel: phase offset by 240 degrees
        b = (np.sin(wave * 1.7 + rotation + np.pi * 4/3) + 1.0) * 127.5
        
        # Stack into RGB array
        rgb = np.stack([r, g, b], axis=-1)
        
        return rgb
    
    def generate_sine_pattern(self, amplitude: float = 1.0, frequency: float = 0.02) -> np.ndarray:
        """
        Generate pure sine wave pattern (for testing).
        
        Args:
            amplitude: Wave amplitude
            frequency: Wave frequency
            
        Returns:
            RGB pattern array
        """
        params = PatternParams(
            sin_amplitude=amplitude,
            sin_frequency=frequency,
            cos_amplitude=0.0,
            tan_amplitude=0.0,
        )
        return self.generate_pattern(params)
    
    def generate_cosine_pattern(self, amplitude: float = 1.0, frequency: float = 0.02) -> np.ndarray:
        """
        Generate pure cosine wave pattern (for testing).
        
        Args:
            amplitude: Wave amplitude
            frequency: Wave frequency
            
        Returns:
            RGB pattern array
        """
        params = PatternParams(
            sin_amplitude=0.0,
            cos_amplitude=amplitude,
            cos_frequency=frequency,
            tan_amplitude=0.0,
        )
        return self.generate_pattern(params)
    
    def generate_tangent_pattern(self, amplitude: float = 0.5, frequency: float = 0.01) -> np.ndarray:
        """
        Generate pure tangent wave pattern (for testing).
        
        Args:
            amplitude: Wave amplitude
            frequency: Wave frequency
            
        Returns:
            RGB pattern array
        """
        params = PatternParams(
            sin_amplitude=0.0,
            cos_amplitude=0.0,
            tan_amplitude=amplitude,
            tan_frequency=frequency,
        )
        return self.generate_pattern(params)
    
    def generate_combined_pattern(
        self,
        sin_amp: float = 1.0,
        cos_amp: float = 1.0,
        tan_amp: float = 0.5,
        frequency: float = 0.01
    ) -> np.ndarray:
        """
        Generate combined multi-function pattern (heritage style).
        
        Args:
            sin_amp: Sine amplitude
            cos_amp: Cosine amplitude
            tan_amp: Tangent amplitude
            frequency: Base frequency for all functions
            
        Returns:
            RGB pattern array
        """
        params = PatternParams(
            sin_amplitude=sin_amp,
            sin_frequency=frequency,
            cos_amplitude=cos_amp,
            cos_frequency=frequency,
            tan_amplitude=tan_amp,
            tan_frequency=frequency * 0.5,
        )
        return self.generate_pattern(params)
