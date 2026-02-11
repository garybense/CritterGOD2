"""
Circuit8 Visualizer

Real-time visualization of the telepathic canvas.
Shows collective consciousness emerging.

Requires: pygame (pip install pygame)
"""

import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not available - visualization disabled")

from core.morphic.circuit8 import Circuit8
from core.evolution.genotype import Genotype
from creatures.creature import Creature
from typing import List


class Circuit8Visualizer:
    """
    Visualize Circuit8 and creatures in real-time.
    
    Shows:
    - Circuit8 pixel grid (scaled up)
    - Creature positions
    - Voting arrows
    - Statistics overlay
    """
    
    def __init__(
        self,
        circuit8: Circuit8,
        creatures: List[Creature],
        scale: int = 10
    ):
        """
        Initialize visualizer.
        
        Args:
            circuit8: Circuit8 instance to visualize
            creatures: List of creatures to show
            scale: Pixel scale factor (10 = each Circuit8 pixel is 10x10 screen pixels)
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required for visualization")
            
        self.circuit8 = circuit8
        self.creatures = creatures
        self.scale = scale
        
        # Screen size
        self.width = circuit8.width * scale
        self.height = circuit8.height * scale
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width + 300, self.height))
        pygame.display.set_caption("Circuit8 - Collective Unconscious")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.running = True
        
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    
    def draw_circuit8(self):
        """Draw Circuit8 pixel grid."""
        for y in range(self.circuit8.height):
            for x in range(self.circuit8.width):
                r, g, b = self.circuit8.read_pixel(x, y)
                color = (int(r), int(g), int(b))
                
                pygame.draw.rect(
                    self.screen,
                    color,
                    (x * self.scale, y * self.scale, self.scale, self.scale)
                )
                
    def draw_creatures(self):
        """Draw creatures on top of Circuit8."""
        for creature in self.creatures:
            # Map creature position to Circuit8 coordinates
            screen_x = int(creature.x) % self.circuit8.width
            screen_y = int(creature.y) % self.circuit8.height
            
            pixel_x = screen_x * self.scale + self.scale // 2
            pixel_y = screen_y * self.scale + self.scale // 2
            
            # Draw creature as circle
            # Color based on energy level
            energy_frac = creature.energy.get_energy_fraction()
            color = (
                int(255 * (1.0 - energy_frac)),  # Red when low energy
                int(255 * energy_frac),  # Green when high energy
                0
            )
            
            pygame.draw.circle(
                self.screen,
                color,
                (pixel_x, pixel_y),
                self.scale // 2
            )
            
            # Draw voting arrow
            if creature.vote_dx != 0 or creature.vote_dy != 0:
                arrow_len = self.scale
                end_x = pixel_x + creature.vote_dx * arrow_len
                end_y = pixel_y + creature.vote_dy * arrow_len
                
                pygame.draw.line(
                    self.screen,
                    (255, 255, 255),
                    (pixel_x, pixel_y),
                    (end_x, end_y),
                    2
                )
                
    def draw_stats(self, timestep: int):
        """Draw statistics panel."""
        panel_x = self.width + 10
        y_offset = 10
        
        # Background
        pygame.draw.rect(
            self.screen,
            (20, 20, 20),
            (self.width, 0, 300, self.height)
        )
        
        # Title
        title = self.font.render("CIRCUIT8", True, (255, 255, 255))
        self.screen.blit(title, (panel_x, y_offset))
        y_offset += 30
        
        # Timestep
        text = self.font.render(f"Time: {timestep}", True, (200, 200, 200))
        self.screen.blit(text, (panel_x, y_offset))
        y_offset += 25
        
        # Population
        text = self.font.render(f"Pop: {len(self.creatures)}", True, (200, 200, 200))
        self.screen.blit(text, (panel_x, y_offset))
        y_offset += 25
        
        if len(self.creatures) > 0:
            # Average age
            avg_age = np.mean([c.age for c in self.creatures])
            text = self.font.render(f"Avg Age: {avg_age:.0f}", True, (200, 200, 200))
            self.screen.blit(text, (panel_x, y_offset))
            y_offset += 25
            
            # Average energy
            avg_energy = np.mean([c.energy.energy for c in self.creatures])
            text = self.font.render(f"Avg E: {avg_energy:.0f}", True, (200, 200, 200))
            self.screen.blit(text, (panel_x, y_offset))
            y_offset += 25
            
            # Average drug level
            avg_drugs = np.mean([c.drugs.get_total_drug_level() for c in self.creatures])
            text = self.font.render(f"Avg Drug: {avg_drugs:.0f}", True, (200, 200, 200))
            self.screen.blit(text, (panel_x, y_offset))
            y_offset += 25
            
            # Collective vote
            y_offset += 15
            text = self.font.render("Collective Vote:", True, (255, 255, 100))
            self.screen.blit(text, (panel_x, y_offset))
            y_offset += 25
            
            dx_votes = sum(c.vote_dx for c in self.creatures)
            dy_votes = sum(c.vote_dy for c in self.creatures)
            
            text = self.font.render(f"  dx: {dx_votes:+d}", True, (200, 200, 200))
            self.screen.blit(text, (panel_x, y_offset))
            y_offset += 25
            
            text = self.font.render(f"  dy: {dy_votes:+d}", True, (200, 200, 200))
            self.screen.blit(text, (panel_x, y_offset))
            y_offset += 25
            
        # Instructions
        y_offset = self.height - 50
        text = self.font.render("ESC: Quit", True, (150, 150, 150))
        self.screen.blit(text, (panel_x, y_offset))
        
    def render(self, timestep: int):
        """Render one frame."""
        self.screen.fill((0, 0, 0))
        
        self.draw_circuit8()
        self.draw_creatures()
        self.draw_stats(timestep)
        
        pygame.display.flip()
        
    def update(self, timestep: int, fps: int = 30):
        """
        Update visualization.
        
        Args:
            timestep: Current timestep
            fps: Target frames per second
        """
        self.handle_events()
        self.render(timestep)
        self.clock.tick(fps)
        
    def close(self):
        """Close visualizer."""
        pygame.quit()


def demo_visualization():
    """
    Demo Circuit8 visualization with creatures.
    """
    if not PYGAME_AVAILABLE:
        print("pygame not installed - cannot run visualization")
        print("Install with: pip install pygame")
        return
        
    print("Circuit8 Visualization Demo")
    print("Creating creatures sharing telepathic canvas...")
    
    # Create Circuit8
    circuit8 = Circuit8()
    
    # Create creatures
    creatures = []
    for i in range(10):
        genotype = Genotype.create_random(
            n_sensory=10,
            n_motor=16,
            n_hidden_min=50,
            n_hidden_max=100,
            synapses_per_neuron=20
        )
        creature = Creature(
            genotype=genotype,
            x=np.random.uniform(0, 640),
            y=np.random.uniform(0, 480),
            circuit8=circuit8
        )
        creatures.append(creature)
        
    # Create visualizer
    viz = Circuit8Visualizer(circuit8, creatures, scale=10)
    
    print("Visualization running...")
    print("Watch collective consciousness emerge on Circuit8")
    print("Press ESC to quit")
    
    timestep = 0
    while viz.running and timestep < 10000:
        # Update creatures
        for creature in creatures:
            creature.update()
            
        # Apply collective vote
        circuit8.apply_voted_movement()
        circuit8.update_depth_buffer()
        
        # Update visualization
        viz.update(timestep, fps=30)
        
        timestep += 1
        
    viz.close()
    print(f"Simulation ran for {timestep} timesteps")


if __name__ == '__main__':
    demo_visualization()
