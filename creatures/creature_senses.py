"""
Sensory integration for creatures.

Extends creatures with visual perception through retinal sensors.
Heritage: 100 fingers × 50 neurons = 5000 sensory neurons from SDL visualizers.
"""

import numpy as np
from typing import Optional
from core.neural.network import NeuralNetwork
from core.neural.neuron import NeuronType
from generators.visual.retinal_sensors import RetinalArray


class CreatureSenses:
    """
    Sensory systems for creatures.
    
    Integrates visual perception through retinal sensor arrays,
    mapping visual input to neural sensory neurons.
    
    Heritage from SDL visualizers (looser.c):
    - 100 "fingers" (sensors) reading visual field
    - 50 neurons per finger
    - 5000 total sensory neurons
    
    Attributes:
        network: Neural network receiving sensory input
        retinal_array: Retinal sensor array (if vision enabled)
        hearing_enabled: Whether auditory input enabled (future)
    """
    
    def __init__(
        self,
        network: NeuralNetwork,
        enable_vision: bool = True,
        enable_hearing: bool = False,
        vision_resolution: int = 100,  # Number of sensors
        vision_width: int = 640,
        vision_height: int = 480
    ):
        """
        Initialize creature sensory systems.
        
        Args:
            network: Neural network for sensory injection
            enable_vision: Enable visual perception
            enable_hearing: Enable auditory perception (future)
            vision_resolution: Number of retinal sensors (heritage: 100)
            vision_width: Visual field width
            vision_height: Visual field height
        """
        self.network = network
        self.hearing_enabled = enable_hearing
        
        # Visual system
        self.retinal_array: Optional[RetinalArray] = None
        if enable_vision:
            self.retinal_array = RetinalArray(
                num_sensors=vision_resolution,
                screen_width=vision_width,
                screen_height=vision_height,
                neurons_per_sensor=50  # Heritage value
            )
        
        # Track sensory neuron indices
        self._sensory_neuron_indices = [
            i for i, n in enumerate(network.neurons)
            if n.neuron_type == NeuronType.SENSORY
        ]
    
    def process_visual_input(self, visual_field, x: int = 0, y: int = 0) -> Optional[np.ndarray]:
        """
        Process visual input through retinal sensors.
        
        Converts RGB visual field to neural activation values.
        Can accept either numpy array or Circuit8 object.
        
        Args:
            visual_field: RGB array (height, width, 3) or Circuit8 object
            x: X coordinate for Circuit8 reading (center of view)
            y: Y coordinate for Circuit8 reading (center of view)
            
        Returns:
            Activation values for sensory neurons (5000 values)
            None if vision not enabled
        """
        if self.retinal_array is None:
            return None
        
        # Convert Circuit8 to numpy array if needed
        if hasattr(visual_field, 'screen'):  # Circuit8 object
            visual_array = visual_field.screen.copy()
        else:
            visual_array = visual_field
        
        # Read visual field through retinal sensors
        activations = self.retinal_array.read_screen(visual_array)
        return activations
    
    def inject_visual_input(self, activations: np.ndarray) -> None:
        """
        Inject visual activations into network sensory neurons.
        
        Maps activation array to available sensory neurons.
        If fewer sensory neurons than activations, groups are averaged.
        If more sensory neurons than activations, direct mapping with wraparound.
        
        Args:
            activations: Flat array of activation values from retinal sensors
        """
        if not self._sensory_neuron_indices:
            return
        
        num_sensory = len(self._sensory_neuron_indices)
        num_activations = len(activations)
        
        if num_sensory >= num_activations:
            # More (or equal) neurons than activations - direct mapping with wraparound
            for i, neuron_idx in enumerate(self._sensory_neuron_indices):
                activation_idx = i % num_activations
                self.network.neurons[neuron_idx].add_input(activations[activation_idx])
        else:
            # Fewer neurons than activations - group and average
            activations_per_neuron = num_activations // num_sensory
            for i, neuron_idx in enumerate(self._sensory_neuron_indices):
                start_idx = i * activations_per_neuron
                end_idx = start_idx + activations_per_neuron
                avg_activation = np.mean(activations[start_idx:end_idx])
                self.network.neurons[neuron_idx].add_input(avg_activation)
    
    def get_visual_statistics(self) -> dict:
        """
        Get statistics about visual processing.
        
        Returns:
            Dictionary with visual system stats
        """
        if self.retinal_array is None:
            return {'vision_enabled': False}
        
        return {
            'vision_enabled': True,
            'num_sensors': self.retinal_array.num_sensors,
            'neurons_per_sensor': self.retinal_array.neurons_per_sensor,
            'total_sensory_neurons': self.retinal_array.get_total_neurons(),
            'network_sensory_neurons': len(self._sensory_neuron_indices),
        }
