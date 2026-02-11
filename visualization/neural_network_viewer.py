"""
Neural Network Viewer

Real-time visualization of neural network structure and dynamics.
Neurons positioned by connectivity using force-directed layout (from SpaceNet).

Features:
- Neurons colored by type and activity
- Synapses colored by weight (excitatory/inhibitory)
- Real-time firing visualization
- Interactive zoom/pan
"""

import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not available - visualization disabled")

from core.neural.network import NeuralNetwork
from core.neural.neuron import NeuronType
from typing import Dict, Tuple, Optional


class NeuralNetworkViewer:
    """
    Visualize neural network structure and dynamics.
    
    Uses force-directed layout to position neurons:
    - Connected neurons attract
    - All neurons repel (prevent overlap)
    - System reaches equilibrium showing network topology
    """
    
    def __init__(
        self,
        network: NeuralNetwork,
        width: int = 1200,
        height: int = 800
    ):
        """
        Initialize neural network viewer.
        
        Args:
            network: Neural network to visualize
            width: Window width
            height: Window height
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required for visualization")
        
        self.network = network
        self.width = width
        self.height = height
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Neural Network Viewer")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)
        self.running = True
        
        # Layout state
        self.neuron_positions: Dict[int, Tuple[float, float]] = {}
        self.neuron_velocities: Dict[int, Tuple[float, float]] = {}
        
        # Physics parameters (from SpaceNet)
        self.repulsion_strength = 500.0
        self.attraction_strength = 0.1
        self.damping = 0.85
        self.dt = 0.5
        
        # View parameters
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.zoom = 1.0
        
        # Initialize random positions
        self._initialize_positions()
        
        # Run layout to find good initial positions
        for _ in range(50):
            self._update_layout()
    
    def _initialize_positions(self):
        """Initialize neuron positions randomly."""
        for neuron in self.network.neurons:
            # Random position in circle
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(50, 200)
            
            x = self.width / 2 + radius * np.cos(angle)
            y = self.height / 2 + radius * np.sin(angle)
            
            self.neuron_positions[neuron.neuron_id] = (x, y)
            self.neuron_velocities[neuron.neuron_id] = (0.0, 0.0)
    
    def _update_layout(self):
        """Update force-directed layout."""
        forces = {nid: [0.0, 0.0] for nid in self.neuron_positions.keys()}
        
        # Calculate repulsion between all neurons
        neuron_list = list(self.network.neurons)
        n = len(neuron_list)
        
        for i in range(n):
            for j in range(i + 1, n):
                n1 = neuron_list[i]
                n2 = neuron_list[j]
                
                pos1 = self.neuron_positions[n1.neuron_id]
                pos2 = self.neuron_positions[n2.neuron_id]
                
                dx = pos2[0] - pos1[0]
                dy = pos2[1] - pos1[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < 1.0:
                    distance = 1.0
                
                # Repulsion force
                repulsion = self.repulsion_strength / distance
                fx = -dx / distance * repulsion
                fy = -dy / distance * repulsion
                
                forces[n1.neuron_id][0] += fx
                forces[n1.neuron_id][1] += fy
                forces[n2.neuron_id][0] -= fx
                forces[n2.neuron_id][1] -= fy
        
        # Calculate attraction along synapses
        for synapse in self.network.synapses:
            pre_id = synapse.pre_neuron.neuron_id
            post_id = synapse.post_neuron.neuron_id
            
            pos_pre = self.neuron_positions[pre_id]
            pos_post = self.neuron_positions[post_id]
            
            dx = pos_post[0] - pos_pre[0]
            dy = pos_post[1] - pos_pre[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # Spring force (Hooke's law)
            attraction = self.attraction_strength * distance
            fx = dx / distance * attraction if distance > 0 else 0
            fy = dy / distance * attraction if distance > 0 else 0
            
            forces[pre_id][0] += fx
            forces[pre_id][1] += fy
            forces[post_id][0] -= fx
            forces[post_id][1] -= fy
        
        # Apply forces to velocities and positions
        for neuron in self.network.neurons:
            nid = neuron.neuron_id
            fx, fy = forces[nid]
            vx, vy = self.neuron_velocities[nid]
            
            # Update velocity
            vx += fx * self.dt
            vy += fy * self.dt
            
            # Apply damping
            vx *= self.damping
            vy *= self.damping
            
            self.neuron_velocities[nid] = (vx, vy)
            
            # Update position
            px, py = self.neuron_positions[nid]
            px += vx * self.dt
            py += vy * self.dt
            
            self.neuron_positions[nid] = (px, py)
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Reset layout
                    self._initialize_positions()
                elif event.key == pygame.K_r:
                    # Run layout stabilization
                    for _ in range(100):
                        self._update_layout()
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom
                self.zoom *= 1.1 if event.y > 0 else 0.9
                self.zoom = max(0.1, min(5.0, self.zoom))
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int((x - self.camera_x) * self.zoom)
        screen_y = int((y - self.camera_y) * self.zoom)
        return (screen_x, screen_y)
    
    def draw_synapses(self):
        """Draw all synapses as lines."""
        for synapse in self.network.synapses:
            pre_id = synapse.pre_neuron.neuron_id
            post_id = synapse.post_neuron.neuron_id
            
            pos_pre = self.neuron_positions.get(pre_id)
            pos_post = self.neuron_positions.get(post_id)
            
            if pos_pre is None or pos_post is None:
                continue
            
            screen_pre = self.world_to_screen(*pos_pre)
            screen_post = self.world_to_screen(*pos_post)
            
            # Color based on weight
            if synapse.is_inhibitory:
                # Inhibitory = red spectrum
                intensity = min(255, int(abs(synapse.weight) * 50))
                color = (intensity, 0, intensity // 2)
            else:
                # Excitatory = green spectrum
                intensity = min(255, int(abs(synapse.weight) * 50))
                color = (0, intensity, intensity // 2)
            
            # Line width based on weight strength
            width = max(1, int(abs(synapse.weight) * 0.5))
            
            pygame.draw.line(
                self.screen,
                color,
                screen_pre,
                screen_post,
                width
            )
    
    def draw_neurons(self):
        """Draw all neurons as circles."""
        for neuron in self.network.neurons:
            pos = self.neuron_positions.get(neuron.neuron_id)
            if pos is None:
                continue
            
            screen_pos = self.world_to_screen(*pos)
            
            # Base color by neuron type
            if neuron.neuron_type == NeuronType.SENSORY:
                base_color = (100, 100, 255)  # Blue
            elif neuron.neuron_type == NeuronType.MOTOR:
                base_color = (255, 100, 100)  # Red
            elif neuron.is_inhibitory():
                base_color = (255, 100, 255)  # Magenta
            else:
                base_color = (100, 255, 100)  # Green
            
            # Brighten if fired recently
            if neuron.fired_last_step():
                color = (255, 255, 100)  # Bright yellow
            else:
                color = base_color
            
            # Radius based on potential
            base_radius = int(8 * self.zoom)
            potential_radius = int(abs(neuron.potential) * 0.01 * self.zoom)
            radius = max(base_radius, min(base_radius + potential_radius, int(30 * self.zoom)))
            
            pygame.draw.circle(
                self.screen,
                color,
                screen_pos,
                radius
            )
            
            # Draw threshold ring
            threshold_radius = int((base_radius + abs(neuron.threshold) * 0.002) * self.zoom)
            pygame.draw.circle(
                self.screen,
                (150, 150, 150),
                screen_pos,
                threshold_radius,
                1
            )
    
    def draw_stats(self):
        """Draw statistics overlay."""
        y_offset = 10
        
        # Background panel
        pygame.draw.rect(
            self.screen,
            (0, 0, 0, 180),
            (10, 10, 250, 150)
        )
        
        # Stats
        stats = [
            f"Neurons: {len(self.network.neurons)}",
            f"Synapses: {len(self.network.synapses)}",
            f"Activity: {self.network.get_activity_level():.1%}",
            f"Time: {self.network.time:.0f}",
            f"Zoom: {self.zoom:.1f}x",
            "",
            "SPACE: Reset layout",
            "R: Stabilize",
            "ESC: Quit"
        ]
        
        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, (200, 200, 200))
            self.screen.blit(text, (20, y_offset + i * 20))
    
    def render(self):
        """Render one frame."""
        # Clear screen
        self.screen.fill((10, 10, 20))
        
        # Draw network
        self.draw_synapses()
        self.draw_neurons()
        self.draw_stats()
        
        # Update display
        pygame.display.flip()
    
    def run(self, fps: int = 30):
        """
        Run visualization loop.
        
        Args:
            fps: Target frames per second
        """
        while self.running:
            self.handle_events()
            
            # Update network
            self.network.update(dt=1.0)
            
            # Update layout (slow continuous adjustment)
            if np.random.random() < 0.3:  # Update layout 30% of frames
                self._update_layout()
            
            # Render
            self.render()
            
            self.clock.tick(fps)
        
        pygame.quit()


def visualize_network(network: NeuralNetwork):
    """
    Convenience function to visualize a network.
    
    Args:
        network: Neural network to visualize
    """
    viewer = NeuralNetworkViewer(network)
    viewer.run()


if __name__ == "__main__":
    # Demo: create and visualize a random network
    from core.evolution.genotype import Genotype
    from core.evolution.phenotype import build_network_from_genotype
    
    print("Creating random neural network...")
    genotype = Genotype.create_random(
        n_sensory=5,
        n_motor=5,
        n_hidden_min=15,
        n_hidden_max=25,
        synapses_per_neuron=10
    )
    
    network = build_network_from_genotype(genotype)
    
    print(f"Network: {len(network.neurons)} neurons, {len(network.synapses)} synapses")
    print("Starting visualization...")
    print("Controls:")
    print("  SPACE - Reset layout")
    print("  R - Stabilize layout")
    print("  Mouse wheel - Zoom")
    print("  ESC - Quit")
    
    visualize_network(network)
