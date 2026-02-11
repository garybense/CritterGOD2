"""
Phase 9a Demo: Procedural 3D Creature Bodies

Demonstrates:
- Procedurally generated creature bodies from genetics
- Morphological evolution visible across generations
- Drug-responsive body appearance (pulsing, color shifts)
- Body mass affects energy metabolism

This is Phase 9a: Body Generation (Priority 1)
- Body genotype encoding ✓
- Procedural mesh generation ✓
- Basic rendering (no animation yet) ✓
- Morphological evolution visible ✓
"""

import sys
import pygame
import numpy as np
from OpenGL.GL import *

# Add project root to path
sys.path.insert(0, '/Users/gspilz/code/CritterGOD')

from core.evolution.genotype import Genotype
from core.morphology.body_genotype import BodyGenotype
from core.morphic.circuit8 import Circuit8
from creatures.morphological_creature import MorphologicalCreature
from visualization.opengl_renderer import OpenGL3DRenderer
from core.pharmacology.drugs import Pill


class Phase9ADemo:
    """Phase 9a demo: Procedural 3D bodies."""
    
    def __init__(self):
        """Initialize demo."""
        # Pygame and OpenGL setup
        pygame.init()
        self.width = 1400
        self.height = 900
        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.DOUBLEBUF | pygame.OPENGL
        )
        pygame.display.set_caption("CritterGOD Phase 9a: Procedural 3D Bodies")
        
        # Clock
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        # World
        self.world_width = 1000.0
        self.world_height = 1000.0
        
        # Circuit8 telepathic canvas
        self.circuit8 = Circuit8(width=64, height=48)
        
        # Create diverse creatures with different body morphologies
        self.creatures = []
        print("\n🧬 Creating morphologically diverse creatures...")
        
        for i in range(8):
            # Random brain
            brain = Genotype.create_random(
                n_sensory=10,
                n_motor=10,
                n_hidden_min=30,
                n_hidden_max=60,
                synapses_per_neuron=25
            )
            
            # Random body (different segment counts for diversity)
            min_seg = 2 + (i % 3)
            max_seg = 4 + (i % 4)
            body = BodyGenotype.create_random(min_segments=min_seg, max_segments=max_seg)
            
            # Position in circle
            angle = (i / 8.0) * 2 * np.pi
            x = self.world_width/2 + 200 * np.cos(angle)
            y = self.world_height/2 + 200 * np.sin(angle)
            
            creature = MorphologicalCreature(
                genotype=brain,
                body=body,
                x=x,
                y=y,
                initial_energy=1000000.0,
                circuit8=self.circuit8,
                enable_audio=False,  # Disable for performance
                enable_text=True
            )
            
            self.creatures.append(creature)
            print(f"  Creature {i}: {len(body.segments)} segments, "
                  f"{sum(len(s.limbs) for s in body.segments)} limbs, "
                  f"mass={body.get_total_mass():.2f}")
        
        # Drug pills
        self.pills = []
        
        # Create minimal ecosystem-like object for renderer
        self.ecosystem = type('obj', (object,), {
            'creatures': self.creatures,
            'pills': self.pills,
            'circuit8': self.circuit8,
            'world_width': self.world_width,
            'world_height': self.world_height
        })()
        
        # Setup renderer
        self.renderer = OpenGL3DRenderer(self.ecosystem, self.width, self.height)
        self.renderer.setup_opengl()
        
        # Stats
        self.timestep = 0
        self.show_help = True
        
        print("\n✓ Phase 9a demo initialized")
        print("  Press H for help")
    
    def handle_events(self):
        """Handle user input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_d:
                    self._drop_drug_pills()
                elif event.key == pygame.K_m:
                    self._mutate_random_creature()
                elif event.key == pygame.K_r:
                    self.renderer.camera.reset()
            
            # Camera mouse events
            self.renderer.handle_mouse_event(event)
        
        # Camera keyboard controls
        keys = pygame.key.get_pressed()
        self.renderer.handle_keyboard(keys)
    
    def _drop_drug_pills(self):
        """Drop drug pills at random locations."""
        for _ in range(5):
            x = np.random.uniform(100, self.world_width - 100)
            y = np.random.uniform(100, self.world_height - 100)
            molecule_type = np.random.randint(0, 5)
            composition = [0, 0, 0, 0, 0]
            composition[molecule_type] = 50.0
            pill = Pill(x=x, y=y, molecule_composition=composition)
            self.pills.append(pill)
        print(f"💊 Dropped 5 drug pills (timestep {self.timestep})")
    
    def _mutate_random_creature(self):
        """Create mutated offspring from random creature."""
        if len(self.creatures) > 0 and len(self.creatures) < 20:
            parent = np.random.choice(self.creatures)
            offspring = parent.mutate_body(mutation_rate=0.3)
            self.creatures.append(offspring)
            print(f"🧬 Created mutant offspring: {len(offspring.body.segments)} segments, "
                  f"mass={offspring.body_mass:.2f}")
    
    def update(self):
        """Update simulation."""
        if self.paused:
            return
        
        # Update creatures
        for creature in self.creatures[:]:
            alive = creature.update(dt=1.0)
            if not alive:
                self.creatures.remove(creature)
                print(f"💀 Creature died at timestep {self.timestep}")
        
        # Update Circuit8
        self.circuit8.apply_voted_movement()
        self.circuit8.decay(rate=0.98)  # Slow fade
        
        # Drug pills interaction
        for pill in self.pills[:]:
            for creature in self.creatures:
                dx = creature.x - pill.x
                dy = creature.y - pill.y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 20:
                    creature.drugs.consume_pill(pill)
                    self.pills.remove(pill)
                    print(f"💊 Creature consumed pill at timestep {self.timestep}")
                    break
        
        self.timestep += 1
        
        # Auto-mutate every 100 timesteps
        if self.timestep % 100 == 0 and len(self.creatures) < 15:
            self._mutate_random_creature()
    
    def render(self):
        """Render scene."""
        # 3D scene
        self.renderer.render_3d_scene()
        
        # 2D overlay
        self.renderer.begin_2d_overlay()
        self._render_ui()
        self.renderer.end_2d_overlay()
        
        pygame.display.flip()
    
    def _render_ui(self):
        """Render 2D UI overlay."""
        font = pygame.font.Font(None, 24)
        
        # Stats
        stats_text = [
            f"Phase 9a: Procedural 3D Bodies",
            f"Timestep: {self.timestep}",
            f"Creatures: {len(self.creatures)}",
            f"Pills: {len(self.pills)}",
            f"{'PAUSED' if self.paused else 'Running'}"
        ]
        
        for i, line in enumerate(stats_text):
            surf = font.render(line, True, (255, 255, 255))
            texture_data = pygame.image.tostring(surf, "RGBA", True)
            glRasterPos2f(10, 20 + i * 25)
            glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        
        # Help
        if self.show_help:
            help_text = [
                "Controls:",
                "SPACE - Pause/Resume",
                "D - Drop drug pills",
                "M - Mutate random creature",
                "R - Reset camera",
                "H - Toggle this help",
                "Arrow Keys - Pan camera",
                "Mouse Drag - Rotate camera",
                "Mouse Wheel - Zoom",
                "ESC - Quit"
            ]
            
            for i, line in enumerate(help_text):
                surf = font.render(line, True, (200, 200, 255))
                texture_data = pygame.image.tostring(surf, "RGBA", True)
                glRasterPos2f(10, 200 + i * 25)
                glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    
    def run(self):
        """Main loop."""
        print("\n🚀 Running Phase 9a demo...")
        print("  Watch for:")
        print("  - Different body shapes (segments, limbs)")
        print("  - Drug effects (pulsing, color shifts)")
        print("  - Mutations creating new morphologies")
        
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(30)  # 30 FPS
        
        # Cleanup
        self.renderer.cleanup()
        pygame.quit()
        
        print(f"\n✓ Demo complete after {self.timestep} timesteps")
        print(f"  Final population: {len(self.creatures)} creatures")


if __name__ == "__main__":
    demo = Phase9ADemo()
    demo.run()
