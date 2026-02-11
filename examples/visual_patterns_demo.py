"""
Visual pattern generation demo.

Demonstrates neural-driven generative art with retinal feedback loops.
Based on SDL neural visualizers (looser.c, xesu.c, etc.).

REQUIRES: pygame for visualization
Install with: pip install pygame numpy

Controls:
- SPACE: Inject random energy into network
- R: Reset network
- S: Toggle sensor visualization
- Q: Quit
"""

import sys
try:
    import pygame
    import numpy as np
except ImportError:
    print("Error: This demo requires pygame and numpy")
    print("Install with: pip install pygame numpy")
    sys.exit(1)

from core.neural import Neuron, NeuralNetwork
from core.neural.neuron import NeuronType
from generators.visual import VisualPipeline, VisualPipelineStats


def create_network(num_neurons: int = 1000, num_sensory: int = 500) -> NeuralNetwork:
    """
    Create neural network for visual feedback.
    
    Args:
        num_neurons: Total neurons (heritage: 10k-65k)
        num_sensory: Sensory neurons (heritage: 5000)
        
    Returns:
        Neural network with sensory neurons
    """
    network = NeuralNetwork()
    
    # Create sensory neurons
    for i in range(num_sensory):
        neuron = Neuron(
            neuron_id=i,
            neuron_type=NeuronType.SENSORY,
            threshold=1500.0,  # Lower threshold for sensors
            leak_rate=0.95
        )
        network.add_neuron(neuron)
    
    # Create hidden neurons
    for i in range(num_sensory, num_neurons):
        neuron = Neuron(
            neuron_id=i,
            neuron_type=NeuronType.HIDDEN,
            threshold=None,  # Random from heritage range
            leak_rate=0.98
        )
        network.add_neuron(neuron)
    
    # Create random synapses (heritage: 40 per neuron)
    network.create_random_synapses(synapses_per_neuron=40)
    
    # Enable plasticity for evolution
    network.enable_plasticity = True
    
    return network


def run_demo():
    """Run visual pattern demo with pygame."""
    
    # Initialize pygame
    pygame.init()
    width, height = 640, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CritterGOD Visual Patterns - Neural Art")
    clock = pygame.time.Clock()
    
    # Create network
    print("Creating neural network...")
    network = create_network(num_neurons=1000, num_sensory=500)
    
    # Create visual pipeline
    print("Initializing visual pipeline...")
    pipeline = VisualPipeline(
        network=network,
        width=width,
        height=height,
        num_sensors=100,  # Heritage: 100 fingers
        neurons_per_sensor=50  # Heritage: 50 neurons per finger
    )
    
    # Statistics tracker
    stats = VisualPipelineStats()
    
    # State
    show_sensors = False
    running = True
    
    print("\nVisual Pattern Generation Demo")
    print("=" * 50)
    print("Controls:")
    print("  SPACE - Inject random energy")
    print("  R     - Reset network")
    print("  S     - Toggle sensor visualization")
    print("  Q/ESC - Quit")
    print("=" * 50)
    print("\nGenerating patterns...")
    
    # Main loop
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Inject random energy
                    for neuron in network.neurons[:100]:
                        neuron.add_input(np.random.rand() * 2000)
                    print("Energy injected!")
                elif event.key == pygame.K_r:
                    # Reset network
                    for neuron in network.neurons:
                        neuron.potential = np.random.rand() * 5000
                    print("Network reset!")
                elif event.key == pygame.K_s:
                    # Toggle sensor visualization
                    show_sensors = not show_sensors
                    print(f"Sensor visualization: {'ON' if show_sensors else 'OFF'}")
        
        # Update visual feedback cycle
        pattern = pipeline.update_cycle(inject_input=True)
        
        # Get statistics
        activations = pipeline.get_sensor_activations()
        stats.update(pattern, activations)
        
        # Convert numpy array to pygame surface
        if show_sensors:
            # Show sensor positions
            pattern = pipeline.visualize_sensors()
        
        # Transpose for pygame (height, width, 3) -> (width, height, 3)
        pattern_transposed = np.transpose(pattern, (1, 0, 2))
        surface = pygame.surfarray.make_surface(pattern_transposed)
        
        # Render
        screen.blit(surface, (0, 0))
        
        # Display stats overlay
        if pipeline.cycle_count % 30 == 0:  # Update every 30 frames
            current_stats = stats.get_stats()
            print(f"Cycle {pipeline.cycle_count}: "
                  f"Brightness={current_stats['brightness']:.2f}, "
                  f"Energy={current_stats['pattern_energy']:.1f}")
        
        pygame.display.flip()
        clock.tick(30)  # 30 FPS
    
    pygame.quit()
    
    # Final statistics
    print("\n" + "=" * 50)
    print("Final Statistics:")
    final_stats = stats.get_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
