"""
Phase 9b Demo: Food/Resource System & Resource-Seeking Behavior

Demonstrates:
- Food sources with regrowth
- Drug mushrooms (5 molecule types)
- Hunger-driven food seeking
- Addiction-driven drug seeking
- Withdrawal effects
- Resource consumption and energy gain
- Poisson disk sampling for natural distribution

Controls:
- Mouse drag: Rotate camera
- Mouse wheel: Zoom
- Arrow keys: Pan camera
- R: Reset camera
- Space: Pause/unpause
- 1-5: Give drug to creatures (type 0-4)
- F: Spawn food at random location
- D: Spawn random drug mushroom
- S: Print detailed statistics
- Q/Esc: Quit

Phase 9b focuses on resource-seeking behavior as the foundation for more
complex behaviors. Creatures now actively pursue food when hungry and drugs
when addicted, creating emergent survival patterns.
"""

import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from typing import List

# Add parent directory to path
sys.path.insert(0, '/Users/gspilz/code/CritterGOD')

from creatures.behavioral_creature import BehavioralCreature
from core.evolution.genotype import Genotype
from core.morphology.body_genotype import BodyGenotype
from core.morphic.circuit8 import Circuit8
from core.resources.resource_manager import ResourceManager
from core.resources.resource import Resource, ResourceType, create_food, create_drug_mushroom
from visualization.camera import OrbitalCamera
from visualization.gl_primitives import draw_sphere, draw_cylinder, setup_lighting
from core.morphology.mesh_generator import ProceduralMeshGenerator


class Phase9bDemo:
    """Phase 9b interactive demo."""
    
    def __init__(self, n_creatures: int = 8):
        """Initialize demo.
        
        Args:
            n_creatures: Number of creatures
        """
        # Initialize pygame and OpenGL
        pygame.init()
        self.screen_width = 1280
        self.screen_height = 720
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption("Phase 9b: Resource-Seeking Behavior")
        
        # OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.05, 0.05, 0.1, 1.0)
        
        # Setup lighting
        setup_lighting()
        
        # Camera
        self.camera = OrbitalCamera(
            distance=200.0,
            azimuth=45.0,
            elevation=30.0
        )
        
        # World parameters
        self.world_width = 500.0
        self.world_height = 500.0
        
        # Create Circuit8 (telepathic canvas)
        self.circuit8 = Circuit8(width=64, height=48)
        
        # Create resource manager
        self.resource_manager = ResourceManager(
            world_width=self.world_width,
            world_height=self.world_height
        )
        
        # Spawn initial resources
        self.resource_manager.spawn_initial_resources()
        print(f"✅ Spawned {len(self.resource_manager.resources)} initial resources")
        
        # Create creatures
        self.creatures: List[BehavioralCreature] = []
        self._create_creatures(n_creatures)
        
        # Mesh generator for creature bodies
        self.mesh_generator = ProceduralMeshGenerator()
        
        # Simulation state
        self.paused = False
        self.timestep = 0
        self.clock = pygame.time.Clock()
        self.target_fps = 60
        
        # UI font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Mouse state
        self.mouse_dragging = False
        self.last_mouse_pos = None
    
    def _create_creatures(self, n_creatures: int) -> None:
        """Create initial population.
        
        Args:
            n_creatures: Number of creatures to create
        """
        for i in range(n_creatures):
            # Random position
            x = np.random.uniform(-self.world_width/2, self.world_width/2)
            y = np.random.uniform(-self.world_height/2, self.world_height/2)
            z = 0.0
            
            # Create random neural genotype
            genotype = Genotype.create_random(
                n_sensory=20,
                n_motor=20,
                n_hidden_min=50,
                n_hidden_max=150,
                synapses_per_neuron=30
            )
            
            # Create random body
            body = BodyGenotype.create_random()
            
            # Create behavioral creature
            creature = BehavioralCreature(
                genotype=genotype,
                body=body,
                x=x,
                y=y,
                z=z,
                initial_energy=1000000.0,
                circuit8=self.circuit8
            )
            
            self.creatures.append(creature)
            print(f"🧬 Created creature {i+1}: {len(body.segments)} segments, {sum(len(s.limbs) for s in body.segments)} limbs")
    
    def update(self, dt: float = 1.0) -> None:
        """Update simulation.
        
        Args:
            dt: Time step
        """
        if self.paused:
            return
        
        # Update resource manager (regrowth)
        self.resource_manager.update(dt)
        
        # Update creatures
        for creature in self.creatures:
            alive = creature.update(dt, resource_manager=self.resource_manager)
            if not alive:
                print(f"💀 Creature died (energy: {creature.energy.energy:.0f})")
        
        # Remove dead creatures
        self.creatures = [c for c in self.creatures if c.energy.energy > 0]
        
        self.timestep += 1
    
    def render_resources(self) -> None:
        """Render all resources."""
        for resource in self.resource_manager.resources:
            if not resource.active:
                continue
            
            glPushMatrix()
            glTranslatef(resource.x, resource.y, resource.z)
            
            if resource.resource_type == ResourceType.FOOD:
                # Food as green sphere
                glColor4f(0.0, 1.0, 0.0, 0.8)
                draw_sphere(resource.radius, slices=8, stacks=8)
            
            elif resource.resource_type == ResourceType.DRUG_MUSHROOM:
                # Drug mushroom as colored mushroom cap on stem
                # Stem
                glColor4f(0.9, 0.9, 0.8, 0.9)
                glPushMatrix()
                glRotatef(-90, 1, 0, 0)
                draw_cylinder(resource.radius * 0.3, resource.radius * 0.3, resource.radius * 1.5, slices=8)
                glPopMatrix()
                
                # Cap (color based on molecule type)
                colors = [
                    (0.5, 0.0, 1.0),  # Type 0: Purple
                    (1.0, 0.0, 0.5),  # Type 1: Pink
                    (1.0, 0.5, 0.0),  # Type 2: Orange
                    (0.0, 0.5, 1.0),  # Type 3: Cyan
                    (1.0, 1.0, 0.0),  # Type 4: Yellow (potentiator)
                ]
                color = colors[resource.molecule_type] if resource.molecule_type is not None else (1.0, 1.0, 1.0)
                glColor4f(*color, 0.9)
                glPushMatrix()
                glTranslatef(0, 0, resource.radius * 1.5)
                draw_sphere(resource.radius, slices=8, stacks=8)
                glPopMatrix()
            
            elif resource.resource_type == ResourceType.ENERGY_ZONE:
                # Energy zone as pulsing blue sphere
                pulse = 0.8 + 0.2 * np.sin(self.timestep * 0.1)
                glColor4f(0.0, 0.5, 1.0, 0.3 * pulse)
                draw_sphere(resource.radius, slices=12, stacks=12)
            
            glPopMatrix()
    
    def render_creatures(self) -> None:
        """Render all creatures."""
        for creature in self.creatures:
            # Generate or retrieve cached mesh
            if not hasattr(creature, 'mesh') or creature.mesh is None:
                creature.mesh = self.mesh_generator.generate_creature_mesh(creature.body)
            
            # Draw target line if creature has target
            if creature.target_resource is not None and creature.target_resource.active:
                glDisable(GL_LIGHTING)
                glBegin(GL_LINES)
                glColor4f(1.0, 1.0, 0.0, 0.5)
                glVertex3f(creature.x, creature.y, creature.z + 5.0)
                glVertex3f(
                    creature.target_resource.x,
                    creature.target_resource.y,
                    creature.target_resource.z + 5.0
                )
                glEnd()
                glEnable(GL_LIGHTING)
            
            # Draw creature mesh
            glPushMatrix()
            glTranslatef(creature.x, creature.y, creature.z + 5.0)  # Raise above ground
            
            # Get drug-responsive scale
            scale = creature.get_render_scale() if hasattr(creature, 'get_render_scale') else 1.0
            glScalef(scale, scale, scale)
            
            # Base body scale
            body_scale = 5.0
            glScalef(body_scale, body_scale, body_scale)
            
            # Render mesh
            creature.mesh.render()
            
            glPopMatrix()
    
    def render_ground(self) -> None:
        """Render ground plane."""
        glDisable(GL_LIGHTING)
        glColor4f(0.1, 0.15, 0.2, 1.0)
        glBegin(GL_QUADS)
        hw = self.world_width / 2
        hh = self.world_height / 2
        glVertex3f(-hw, -hh, -1)
        glVertex3f(hw, -hh, -1)
        glVertex3f(hw, hh, -1)
        glVertex3f(-hw, hh, -1)
        glEnd()
        glEnable(GL_LIGHTING)
    
    def render_3d(self) -> None:
        """Render 3D scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Setup 3D projection
        self.camera.apply_projection(self.screen_width, self.screen_height)
        
        # Setup camera view
        self.camera.apply_view()
        
        # Render scene
        self.render_ground()
        self.render_resources()
        self.render_creatures()
    
    def render_ui(self) -> None:
        """Render UI overlay."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.screen_width, self.screen_height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Render text using pygame surface -> OpenGL texture
        y_offset = 10
        texts = [
            f"Phase 9b: Resource-Seeking Behavior | Timestep: {self.timestep}",
            f"Creatures: {len(self.creatures)} | Resources: {len([r for r in self.resource_manager.resources if r.active])}",
            f"Food: {len([r for r in self.resource_manager.resources if r.resource_type == ResourceType.FOOD and r.active])} | " +
            f"Drugs: {len([r for r in self.resource_manager.resources if r.resource_type == ResourceType.DRUG_MUSHROOM and r.active])}",
        ]
        
        if self.paused:
            texts.append("⏸ PAUSED")
        
        # Render behavior state for first creature
        if self.creatures:
            creature = self.creatures[0]
            state = creature.get_behavior_state()
            texts.extend([
                "",
                f"Creature 0 Behavior:",
                f"  Hunger: {state['hunger_level']:.2f} | Seeking food: {state['should_seek_food']}",
                f"  Drug craving: {state['strongest_craving']}",
                f"  Target: {state['target_type'] or 'None'}",
            ])
        
        for text in texts:
            self.render_text(text, 10, y_offset, self.small_font)
            y_offset += 20
        
        # Controls at bottom
        controls = [
            "Space: Pause | 1-5: Give drugs | F: Spawn food | D: Spawn drug | S: Stats | R: Reset camera"
        ]
        y_offset = self.screen_height - 30
        for text in controls:
            self.render_text(text, 10, y_offset, self.small_font)
            y_offset += 20
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def render_text(self, text: str, x: int, y: int, font: pygame.font.Font) -> None:
        """Render text to screen.
        
        Args:
            text: Text to render
            x, y: Position
            font: Font to use
        """
        surface = font.render(text, True, (255, 255, 255))
        data = pygame.image.tostring(surface, "RGBA", True)
        
        glRasterPos2i(x, y)
        glDrawPixels(surface.get_width(), surface.get_height(),
                    GL_RGBA, GL_UNSIGNED_BYTE, data)
    
    def render(self) -> None:
        """Render complete frame."""
        self.render_3d()
        self.render_ui()
        pygame.display.flip()
    
    def handle_events(self) -> bool:
        """Handle input events.
        
        Returns:
            False if should quit, True otherwise
        """
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            elif event.type == KEYDOWN:
                if event.key == K_q or event.key == K_ESCAPE:
                    return False
                elif event.key == K_SPACE:
                    self.paused = not self.paused
                elif event.key == K_r:
                    self.camera.reset()
                elif event.key == K_s:
                    self.print_statistics()
                elif event.key == K_f:
                    self.spawn_food()
                elif event.key == K_d:
                    self.spawn_drug()
                elif K_1 <= event.key <= K_5:
                    self.give_drug_to_all(event.key - K_1)
                elif event.key == K_UP:
                    self.camera.handle_pan(0, -1)
                elif event.key == K_DOWN:
                    self.camera.handle_pan(0, 1)
                elif event.key == K_LEFT:
                    self.camera.handle_pan(-1, 0)
                elif event.key == K_RIGHT:
                    self.camera.handle_pan(1, 0)
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:  # Scroll up
                    self.camera.handle_zoom(10)
                elif event.button == 5:  # Scroll down
                    self.camera.handle_zoom(-10)
            
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
            
            elif event.type == MOUSEMOTION:
                if self.mouse_dragging:
                    pos = pygame.mouse.get_pos()
                    if self.last_mouse_pos:
                        dx = pos[0] - self.last_mouse_pos[0]
                        dy = pos[1] - self.last_mouse_pos[1]
                        self.camera.handle_mouse_drag(dx, dy)
                    self.last_mouse_pos = pos
        
        return True
    
    def spawn_food(self) -> None:
        """Spawn food at random location."""
        x = np.random.uniform(-self.world_width/2, self.world_width/2)
        y = np.random.uniform(-self.world_height/2, self.world_height/2)
        food = create_food(x, y, 0.0)
        self.resource_manager.resources.append(food)
        print(f"🍎 Spawned food at ({x:.1f}, {y:.1f})")
    
    def spawn_drug(self) -> None:
        """Spawn random drug mushroom."""
        x = np.random.uniform(-self.world_width/2, self.world_width/2)
        y = np.random.uniform(-self.world_height/2, self.world_height/2)
        molecule_type = np.random.randint(0, 5)
        drug = create_drug_mushroom(x, y, 0.0, molecule_type)
        self.resource_manager.resources.append(drug)
        print(f"🍄 Spawned drug type {molecule_type} at ({x:.1f}, {y:.1f})")
    
    def give_drug_to_all(self, molecule_type: int) -> None:
        """Give drug to all creatures.
        
        Args:
            molecule_type: Drug type (0-4)
        """
        for creature in self.creatures:
            creature.drugs.tripping[molecule_type] += 5.0
        print(f"💊 Gave all creatures drug type {molecule_type} (dose: 5.0)")
    
    def print_statistics(self) -> None:
        """Print detailed simulation statistics."""
        print("\n" + "="*60)
        print(f"Phase 9b Statistics - Timestep {self.timestep}")
        print("="*60)
        
        # Resource statistics
        active_resources = [r for r in self.resource_manager.resources if r.active]
        food_count = len([r for r in active_resources if r.resource_type == ResourceType.FOOD])
        drug_count = len([r for r in active_resources if r.resource_type == ResourceType.DRUG_MUSHROOM])
        
        print(f"\nResources:")
        print(f"  Total active: {len(active_resources)}")
        print(f"  Food: {food_count}")
        print(f"  Drugs: {drug_count}")
        
        # Creature statistics
        print(f"\nCreatures: {len(self.creatures)}")
        for i, creature in enumerate(self.creatures[:3]):  # Show first 3
            state = creature.get_behavior_state()
            print(f"\n  Creature {i}:")
            print(f"    Energy: {creature.energy.energy:.0f}/{creature.energy.max_energy:.0f}")
            print(f"    Hunger: {state['hunger_level']:.2f}")
            print(f"    Seeking food: {state['should_seek_food']}")
            print(f"    Seeking drug: {state['should_seek_drug']}")
            print(f"    Strongest craving: {state['strongest_craving']}")
            print(f"    Target: {state['target_type'] or 'None'}")
            print(f"    Food consumed: {creature.behavior.food_consumed_count}")
            print(f"    Drugs consumed: {sum(creature.behavior.drug_consumed_count)}")
    
    def run(self) -> None:
        """Run demo main loop."""
        print("\n🎮 Phase 9b Demo Controls:")
        print("  Mouse drag: Rotate camera")
        print("  Mouse wheel: Zoom")
        print("  Arrow keys: Pan camera")
        print("  R: Reset camera")
        print("  Space: Pause/unpause")
        print("  1-5: Give drug to creatures (type 0-4)")
        print("  F: Spawn food")
        print("  D: Spawn drug mushroom")
        print("  S: Print detailed statistics")
        print("  Q/Esc: Quit")
        print("\n🧬 Starting simulation with resource-seeking behavior...\n")
        
        running = True
        while running:
            # Handle events
            running = self.handle_events()
            
            # Update
            self.update(dt=1.0)
            
            # Render
            self.render()
            
            # Control framerate
            self.clock.tick(self.target_fps)
        
        pygame.quit()


def main():
    """Run Phase 9b demo."""
    demo = Phase9bDemo(n_creatures=8)
    demo.run()


if __name__ == '__main__':
    main()
