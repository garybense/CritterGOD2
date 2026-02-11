"""
Visual feedback pipeline integrating pattern generation with retinal sensors.

Creates complete feedback loop from SDL neural visualizers:
network state → pattern parameters → visual pattern → retinal sensors → network input

This is the core of emergent aesthetics - the network literally sees its own thoughts.
"""

import numpy as np
from typing import Optional
from core.neural.network import NeuralNetwork
from core.neural.neuron import NeuronType
from generators.visual.pattern_generators import PatternGenerator
from generators.visual.neural_parameters import NeuralPatternMapper
from generators.visual.retinal_sensors import RetinalArray


class VisualPipeline:
    """
    Complete visual feedback pipeline.
    
    Heritage cycle from SDL visualizers:
    1. Extract parameters from network state (neuron threshold ratios)
    2. Generate visual pattern from parameters
    3. Read pattern through retinal sensors
    4. Inject sensor activations into network
    5. Update network (creates new state for next cycle)
    
    This creates self-sustaining dynamics where the network
    evolves its own unique visual aesthetic.
    
    Attributes:
        network: Neural network driving and receiving visuals
        generator: Pattern generator
        mapper: Neural parameter mapper
        retinal: Retinal sensor array
        screen: Current screen buffer
        width: Screen width
        height: Screen height
    """
    
    def __init__(
        self,
        network: NeuralNetwork,
        width: int = 640,
        height: int = 480,
        num_sensors: int = 100,
        neurons_per_sensor: int = 50
    ):
        """
        Initialize visual pipeline.
        
        Args:
            network: Neural network for feedback loop
            width: Screen width (default: 640)
            height: Screen height (default: 480)
            num_sensors: Number of retinal sensors (heritage: 100)
            neurons_per_sensor: Neurons per sensor (heritage: 50)
        """
        self.network = network
        self.width = width
        self.height = height
        
        # Create components
        self.generator = PatternGenerator(width=width, height=height)
        self.mapper = NeuralPatternMapper()
        self.retinal = RetinalArray(
            num_sensors=num_sensors,
            screen_width=width,
            screen_height=height,
            neurons_per_sensor=neurons_per_sensor
        )
        
        # Screen buffer
        self.screen = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Statistics
        self.cycle_count = 0
    
    def update_cycle(self, inject_input: bool = True) -> np.ndarray:
        """
        Execute one complete visual feedback cycle.
        
        Heritage cycle:
        1. Extract pattern params from network
        2. Generate visual pattern
        3. Read through retinal sensors
        4. Inject into network sensory neurons
        5. Update network
        
        Args:
            inject_input: Whether to inject sensor activations into network
            
        Returns:
            Generated screen pattern (height, width, 3)
        """
        # Step 1: Extract parameters from network state
        params = self.mapper.extract_parameters(self.network)
        
        # Step 2: Generate visual pattern
        self.screen = self.generator.generate_pattern(params)
        
        # Step 3: Read pattern through retinal sensors
        activations = self.retinal.read_screen(self.screen)
        
        # Step 4: Inject into network (if enabled)
        if inject_input:
            self._inject_sensory_input(activations)
        
        # Step 5: Update network (creates new state for next pattern)
        self.network.update(dt=1.0)
        
        self.cycle_count += 1
        
        return self.screen
    
    def _inject_sensory_input(self, activations: np.ndarray) -> None:
        """
        Inject retinal activations into network sensory neurons.
        
        Maps flat activation array to sensory neurons in network.
        If network has fewer sensory neurons than activations,
        distributes activations across available neurons.
        
        Args:
            activations: Flat array of sensor activations
        """
        # Get sensory neurons from network
        sensory_neurons = [
            n for n in self.network.neurons
            if n.neuron_type == NeuronType.SENSORY
        ]
        
        if not sensory_neurons:
            return
        
        num_sensory = len(sensory_neurons)
        num_activations = len(activations)
        
        if num_sensory >= num_activations:
            # More neurons than activations - direct mapping
            for i, activation in enumerate(activations):
                if i < num_sensory:
                    sensory_neurons[i].add_input(activation)
        else:
            # Fewer neurons than activations - average groups
            activations_per_neuron = num_activations // num_sensory
            for i, neuron in enumerate(sensory_neurons):
                start_idx = i * activations_per_neuron
                end_idx = start_idx + activations_per_neuron
                avg_activation = np.mean(activations[start_idx:end_idx])
                neuron.add_input(avg_activation)
    
    def generate_without_feedback(self) -> np.ndarray:
        """
        Generate pattern without network update or sensor feedback.
        
        Useful for visualization without affecting network state.
        
        Returns:
            Generated screen pattern
        """
        params = self.mapper.extract_parameters(self.network)
        return self.generator.generate_pattern(params)
    
    def get_current_params(self):
        """
        Get current pattern parameters from network.
        
        Returns:
            PatternParams extracted from network state
        """
        return self.mapper.extract_parameters(self.network)
    
    def get_sensor_activations(self) -> np.ndarray:
        """
        Get current sensor activations from screen.
        
        Returns:
            Flat array of all sensor activations
        """
        return self.retinal.read_screen(self.screen)
    
    def visualize_sensors(self) -> np.ndarray:
        """
        Create visualization showing sensor positions.
        
        Returns:
            Screen with sensor positions marked
        """
        return self.retinal.visualize_sensor_positions(self.screen)
    
    def reset(self) -> None:
        """Reset pipeline state (clear screen)."""
        self.screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.cycle_count = 0


class VisualPipelineStats:
    """
    Statistics tracker for visual pipeline.
    
    Monitors convergence, diversity, and aesthetic metrics.
    """
    
    def __init__(self):
        self.history = {
            'brightness': [],
            'color_variance': [],
            'pattern_energy': [],
            'sensor_activity': [],
        }
    
    def update(self, screen: np.ndarray, activations: np.ndarray) -> None:
        """
        Update statistics from current cycle.
        
        Args:
            screen: Current screen pattern
            activations: Current sensor activations
        """
        # Average brightness
        brightness = np.mean(screen) / 255.0
        self.history['brightness'].append(brightness)
        
        # Color variance (how colorful vs grayscale)
        r_var = np.var(screen[:, :, 0])
        g_var = np.var(screen[:, :, 1])
        b_var = np.var(screen[:, :, 2])
        color_var = (r_var + g_var + b_var) / 3.0
        self.history['color_variance'].append(color_var)
        
        # Pattern energy (high frequency content)
        gradient = np.gradient(screen.astype(float))
        energy = np.mean([np.abs(g).mean() for g in gradient])
        self.history['pattern_energy'].append(energy)
        
        # Sensor activity level
        sensor_activity = np.mean(activations) / 1000.0
        self.history['sensor_activity'].append(sensor_activity)
    
    def get_stats(self) -> dict:
        """
        Get current statistics summary.
        
        Returns:
            Dictionary of current metrics
        """
        if not self.history['brightness']:
            return {}
        
        return {
            'brightness': self.history['brightness'][-1],
            'color_variance': self.history['color_variance'][-1],
            'pattern_energy': self.history['pattern_energy'][-1],
            'sensor_activity': self.history['sensor_activity'][-1],
            'cycles': len(self.history['brightness']),
        }
