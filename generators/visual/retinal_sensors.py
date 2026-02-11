"""
Retinal sensor system for converting visual patterns to neural input.

Based on SDL neural visualizers (looser.c) with heritage values:
- 100 "fingers" (sensors) reading screen columns
- 50 sensory neurons per finger
- Total: 5000 sensory neurons reading visual field

Creates feedback loop: patterns → sensors → network → patterns
"""

import numpy as np
from typing import List, Tuple


class RetinalSensor:
    """
    Single retinal sensor reading one screen column.
    
    Heritage: Each "finger" has 50 sensory neurons sampling a vertical column.
    Converts RGB pixel values to neural activation potentials.
    
    Attributes:
        x_position: Screen column this sensor reads
        y_start: Top of sensor region
        y_end: Bottom of sensor region
        num_neurons: Number of sensory neurons (heritage: 50)
    """
    
    def __init__(
        self,
        x_position: int,
        y_start: int = 0,
        y_end: int = 480,
        num_neurons: int = 50
    ):
        """
        Initialize retinal sensor.
        
        Args:
            x_position: X coordinate (column) to read
            y_start: Top of sampling region
            y_end: Bottom of sampling region
            num_neurons: Number of sensory neurons (heritage: 50)
        """
        self.x_position = x_position
        self.y_start = y_start
        self.y_end = y_end
        self.num_neurons = num_neurons
        
        # Precompute sampling positions
        self.sample_positions = np.linspace(
            y_start, y_end - 1, num_neurons, dtype=int
        )
    
    def read_column(self, screen: np.ndarray) -> np.ndarray:
        """
        Read screen column and convert to neuron activations.
        
        Heritage approach:
        - Sample pixels along vertical column
        - Convert RGB brightness to activation potential
        - Each neuron gets one sample point
        
        Args:
            screen: RGB screen array of shape (height, width, 3)
            
        Returns:
            Activation values for each neuron (shape: num_neurons)
        """
        height, width = screen.shape[:2]
        
        # Clamp x position to screen bounds
        x = max(0, min(width - 1, self.x_position))
        
        # Sample pixels at neuron positions
        activations = np.zeros(self.num_neurons)
        for i, y in enumerate(self.sample_positions):
            if 0 <= y < height:
                # Get RGB pixel
                rgb = screen[y, x]
                
                # Convert to brightness (activation)
                # Heritage: brightness = (R + G + B) / 3
                brightness = np.mean(rgb)
                
                # Map brightness (0-255) to activation potential (0-1000)
                # Heritage range: neurons have potentials 0-5000
                activation = (brightness / 255.0) * 1000.0
                
                activations[i] = activation
        
        return activations


class RetinalArray:
    """
    Array of retinal sensors covering the visual field.
    
    Heritage: 100 fingers (sensors) reading 100 screen columns.
    Creates 5000 total sensory neurons (100 sensors × 50 neurons each).
    
    Attributes:
        num_sensors: Number of sensors (heritage: 100 fingers)
        screen_width: Width of visual field
        screen_height: Height of visual field
        neurons_per_sensor: Neurons per sensor (heritage: 50)
        sensors: List of RetinalSensor objects
    """
    
    def __init__(
        self,
        num_sensors: int = 100,
        screen_width: int = 640,
        screen_height: int = 480,
        neurons_per_sensor: int = 50
    ):
        """
        Initialize retinal sensor array.
        
        Args:
            num_sensors: Number of sensors (heritage: 100)
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            neurons_per_sensor: Neurons per sensor (heritage: 50)
        """
        self.num_sensors = num_sensors
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.neurons_per_sensor = neurons_per_sensor
        
        # Create sensors evenly spaced across screen width
        self.sensors: List[RetinalSensor] = []
        for i in range(num_sensors):
            x_position = int((i / num_sensors) * screen_width)
            sensor = RetinalSensor(
                x_position=x_position,
                y_start=0,
                y_end=screen_height,
                num_neurons=neurons_per_sensor
            )
            self.sensors.append(sensor)
    
    def read_screen(self, screen: np.ndarray) -> np.ndarray:
        """
        Read entire screen through all sensors.
        
        Returns activations for all sensory neurons as flat array.
        Heritage: 100 sensors × 50 neurons = 5000 total activations.
        
        Args:
            screen: RGB screen array of shape (height, width, 3)
            
        Returns:
            Flat array of all activation values
            Shape: (num_sensors * neurons_per_sensor,)
        """
        all_activations = []
        
        for sensor in self.sensors:
            activations = sensor.read_column(screen)
            all_activations.append(activations)
        
        # Concatenate into flat array
        return np.concatenate(all_activations)
    
    def get_total_neurons(self) -> int:
        """
        Get total number of sensory neurons.
        
        Returns:
            Total neurons (num_sensors * neurons_per_sensor)
        """
        return self.num_sensors * self.neurons_per_sensor
    
    def visualize_sensor_positions(self, screen: np.ndarray) -> np.ndarray:
        """
        Create visualization showing sensor sampling positions.
        
        Draws vertical lines at each sensor position for debugging.
        
        Args:
            screen: Screen to draw on (will be modified)
            
        Returns:
            Modified screen with sensor positions marked
        """
        screen_vis = screen.copy()
        
        for sensor in self.sensors:
            x = sensor.x_position
            
            # Draw vertical line at sensor position
            if 0 <= x < screen_vis.shape[1]:
                screen_vis[:, x] = [255, 0, 0]  # Red line
                
                # Mark sample points
                for y in sensor.sample_positions:
                    if 0 <= y < screen_vis.shape[0]:
                        # Draw small marker
                        screen_vis[y, max(0, x-1):min(screen_vis.shape[1], x+2)] = [0, 255, 0]
        
        return screen_vis


def rgb_to_activation(rgb: Tuple[int, int, int]) -> float:
    """
    Convert RGB pixel to neural activation potential.
    
    Heritage approach: brightness = (R + G + B) / 3
    Maps 0-255 brightness to 0-1000 activation range.
    
    Args:
        rgb: RGB tuple (0-255 each)
        
    Returns:
        Activation potential (0-1000)
    """
    brightness = (rgb[0] + rgb[1] + rgb[2]) / 3.0
    return (brightness / 255.0) * 1000.0


def activation_to_rgb(activation: float) -> Tuple[int, int, int]:
    """
    Convert neural activation to RGB for visualization.
    
    Inverse of rgb_to_activation for debugging/display.
    
    Args:
        activation: Activation potential (0-1000)
        
    Returns:
        Grayscale RGB tuple (0-255 each)
    """
    brightness = int((activation / 1000.0) * 255)
    brightness = max(0, min(255, brightness))
    return (brightness, brightness, brightness)
