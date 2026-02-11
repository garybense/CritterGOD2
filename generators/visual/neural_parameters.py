"""
Extract pattern generation parameters from neural network state.

Based on SDL neural visualizers where pattern parameters are computed
from ratios of specific neuron thresholds (e.g., brain[1591][1] / brain[1425][1]).

This creates emergent aesthetics - each network generates its own unique visual style.
"""

from typing import List
from generators.visual.pattern_generators import PatternParams
from core.neural.network import NeuralNetwork


class NeuralPatternMapper:
    """
    Maps neural network state to visual pattern parameters.
    
    Heritage approach from SDL visualizers:
    - Select specific neuron indices for each parameter
    - Compute ratios of neuron thresholds
    - Map ratios to parameter ranges
    - Creates network-specific aesthetic signatures
    
    Neuron selection indices from looser.c heritage:
    - tanamp: brain[1591] / brain[1425]
    - sinamp: brain[1573] / brain[1950]
    - etc.
    """
    
    def __init__(
        self,
        sin_amp_indices: tuple = (1573, 1950),
        sin_freq_indices: tuple = (800, 1200),
        cos_amp_indices: tuple = (1200, 1600),
        cos_freq_indices: tuple = (500, 900),
        tan_amp_indices: tuple = (1591, 1425),
        tan_freq_indices: tuple = (300, 700),
        phase_indices: tuple = (100, 500),
        color_indices: tuple = (2000, 2500),
    ):
        """
        Initialize neural pattern mapper with neuron index pairs.
        
        Each parameter is computed from ratio of two neuron thresholds.
        Indices wrap if network is smaller than heritage values.
        
        Args:
            sin_amp_indices: Neuron pair for sine amplitude
            sin_freq_indices: Neuron pair for sine frequency
            cos_amp_indices: Neuron pair for cosine amplitude
            cos_freq_indices: Neuron pair for cosine frequency
            tan_amp_indices: Neuron pair for tangent amplitude (heritage: 1591, 1425)
            tan_freq_indices: Neuron pair for tangent frequency
            phase_indices: Neuron pair for phase offset
            color_indices: Neuron pair for color rotation
        """
        self.sin_amp_indices = sin_amp_indices
        self.sin_freq_indices = sin_freq_indices
        self.cos_amp_indices = cos_amp_indices
        self.cos_freq_indices = cos_freq_indices
        self.tan_amp_indices = tan_amp_indices
        self.tan_freq_indices = tan_freq_indices
        self.phase_indices = phase_indices
        self.color_indices = color_indices
    
    def extract_parameters(self, network: NeuralNetwork) -> PatternParams:
        """
        Extract pattern parameters from network state.
        
        Computes ratios of neuron thresholds and maps to parameter ranges.
        
        Args:
            network: Neural network to extract from
            
        Returns:
            PatternParams for pattern generation
        """
        if not network.neurons:
            # Return default params for empty network
            return PatternParams()
        
        num_neurons = len(network.neurons)
        
        # Extract amplitude parameters (0.1 to 2.0 range)
        sin_amp = self._compute_ratio(
            network, self.sin_amp_indices, num_neurons,
            min_val=0.1, max_val=2.0
        )
        
        cos_amp = self._compute_ratio(
            network, self.cos_amp_indices, num_neurons,
            min_val=0.1, max_val=2.0
        )
        
        tan_amp = self._compute_ratio(
            network, self.tan_amp_indices, num_neurons,
            min_val=0.1, max_val=1.0  # Tangent needs smaller range
        )
        
        # Extract frequency parameters (0.001 to 0.05 range)
        sin_freq = self._compute_ratio(
            network, self.sin_freq_indices, num_neurons,
            min_val=0.001, max_val=0.05
        )
        
        cos_freq = self._compute_ratio(
            network, self.cos_freq_indices, num_neurons,
            min_val=0.001, max_val=0.05
        )
        
        tan_freq = self._compute_ratio(
            network, self.tan_freq_indices, num_neurons,
            min_val=0.001, max_val=0.02  # Tangent needs lower frequency
        )
        
        # Extract phase and color (0 to 1 range)
        phase = self._compute_ratio(
            network, self.phase_indices, num_neurons,
            min_val=0.0, max_val=1.0
        )
        
        color_rotation = self._compute_ratio(
            network, self.color_indices, num_neurons,
            min_val=0.0, max_val=1.0
        )
        
        # Brightness based on average network activity
        brightness = self._compute_brightness(network)
        
        return PatternParams(
            sin_amplitude=sin_amp,
            sin_frequency=sin_freq,
            cos_amplitude=cos_amp,
            cos_frequency=cos_freq,
            tan_amplitude=tan_amp,
            tan_frequency=tan_freq,
            phase_offset=phase * 6.28,  # Convert to radians
            color_rotation=color_rotation,
            brightness=brightness,
        )
    
    def _compute_ratio(
        self,
        network: NeuralNetwork,
        indices: tuple,
        num_neurons: int,
        min_val: float,
        max_val: float
    ) -> float:
        """
        Compute parameter from ratio of two neuron thresholds.
        
        Heritage approach: ratio = brain[idx1][1] / brain[idx2][1]
        
        Args:
            network: Neural network
            indices: Tuple of (idx1, idx2) for neuron selection
            num_neurons: Total neurons in network
            min_val: Minimum output value
            max_val: Maximum output value
            
        Returns:
            Mapped parameter value in [min_val, max_val]
        """
        idx1, idx2 = indices
        
        # Wrap indices if network is smaller than heritage values
        idx1 = idx1 % num_neurons if num_neurons > 0 else 0
        idx2 = idx2 % num_neurons if num_neurons > 0 else 0
        
        # Get neuron thresholds
        threshold1 = network.neurons[idx1].threshold
        threshold2 = network.neurons[idx2].threshold
        
        # Compute ratio (add epsilon to avoid division by zero)
        ratio = threshold1 / (threshold2 + 1.0)
        
        # Normalize ratio to [0, 1] range
        # Typical threshold range: 700-8700 (from heritage)
        # So ratio typically 0.08 to 12
        normalized = (ratio - 0.1) / 12.0
        normalized = max(0.0, min(1.0, normalized))
        
        # Map to desired range
        return min_val + normalized * (max_val - min_val)
    
    def _compute_brightness(self, network: NeuralNetwork) -> float:
        """
        Compute brightness from average network activity.
        
        Higher activity = brighter patterns
        
        Args:
            network: Neural network
            
        Returns:
            Brightness multiplier (0.3 to 1.0)
        """
        if not network.neurons:
            return 0.5
        
        # Average potential across all neurons
        total_potential = sum(n.potential for n in network.neurons)
        avg_potential = total_potential / len(network.neurons)
        
        # Typical potential range: 0-5000 (from heritage)
        normalized = avg_potential / 5000.0
        normalized = max(0.0, min(1.0, normalized))
        
        # Map to brightness range (0.3 to 1.0)
        return 0.3 + normalized * 0.7
