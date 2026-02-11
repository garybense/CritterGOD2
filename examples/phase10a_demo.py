"""
Phase 10a Demo: Physics-Based Artificial Life

Demonstrates:
- Custom physics engine with rigid body dynamics
- Neural motor control → physics forces
- Collision-based resource consumption
- Creature-creature collisions
- Realistic movement and interactions
- Gravity, friction, and damping

Controls:
- Mouse drag: Rotate camera
- Mouse wheel: Zoom
- Arrow keys: Pan camera
- R: Reset camera
- Space: Pause/unpause
- F: Spawn food
- D: Spawn drug mushroom
- 1-5: Give drug to creatures
- I: Apply random impulse to creatures (push them)
- S: Print detailed statistics
- Q/Esc: Quit

Phase 10a introduces realistic physics to CritterGOD, enabling natural
movement, collisions, and interactions based on Newtonian mechanics.
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

from creatures.physics_creature import PhysicsCreature, create_physics_creatures
from core.physics import PhysicsWorld, Collision
from core.morphic.circuit8 import Circuit8
from core.resources.resource_manager import ResourceManager
from core.resources.resource import ResourceType, create_food, create_drug_mushroom
from visualization.camera import OrbitalCamera
from visualization.gl_primitives import draw_sphere, draw_cylinder, setup_lighting
from core.morphology.mesh_generator import ProceduralMeshGenerator


class Phase10aDemo:
    """Phase 10a interactive physics demo."""
    
    def __init__(self, n_creatures: int = 6):
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
        pygame.display.set_caption("Phase 10a: Physics-Based Artificial Life")
        
        # OpenGL setup
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.05, 0.05, 0.1, 1.0)
        
        # Setup lighting
        setup_lighting()
        
        # Camera
        self.camera = OrbitalCamera(
            distance=300.0,
            azimuth=45.0,
            elevation=25.0
        )
        
        # World parameters
        self.world_width = 500.0
        self.world_height = 500.0
        
        # Create physics world
        self.physics_world = PhysicsWorld(
            gravity=(0.0, 0.0, -9.81),
            world_bounds=(-self.world_width/2, -self.world_height/2, 
                         self.world_width/2, self.world_height/2),
            cell_size=50.0
        )
        
        # Register collision callback
        self.physics_world.register_collision_callback(self.on_collision)
        
        # Create Circuit8 (telepathic canvas)
        self.circuit8 = Circuit8(width=64, height=48)
        
        # Create resource manager
        self.resource_manager = ResourceManager(
            world_width=self.world_width,
            world_height=self.world_height
        )
        
        # Spawn initial resources with physics bodies
        self._spawn_initial_resources()
        
        # Create physics creatures
        self.creatures: List[PhysicsCreature] = create_physics_creatures(
            n_creatures=n_creatures,
            physics_world=self.physics_world,
            circuit8=self.circuit8,
            world_bounds=(-self.world_width/2, -self.world_height/2,
                         self.world_width/2, self.world_height/2)
        )
        
        print(f"✅ Created {len(self.creatures)} physics creatures")
        
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
        
        # Collision counter
        self.collision_count = 0
    
    def _spawn_initial_resources(self):
        """Spawn initial resources with physics bodies."""
        self.resource_manager.spawn_initial_resources()
        
        # Create physics bodies for resources
        for resource in self.resource_manager.resources:
            # Create rigid body for resource
            body = self.physics_world.create_sphere_body(
                position=(resource.x, resource.y, resource.z + resource.radius),
                radius=resource.radius,
                mass=0.1,  # Light
                fixed=True,  # Resources don't move
                restitution=0.5,
                user_data=resource  # Link back to resource
            )
            # Resources in collision group 2
            body.collision_group = 2
            body.collision_mask = 1  # Only collide with creatures (group 1)
        
        print(f"✅ Spawned {len(self.resource_manager.resources)} resources with physics")
    
    def on_collision(self, collision: Collision):
        """Handle collision events.
        
        Args:
            collision: Collision information
        """
        self.collision_count += 1
        
        # Handle creature collisions
        from creatures.physics_creature import PhysicsCreature
        if isinstance(collision.body_a.user_data, PhysicsCreature):
            collision.body_a.user_data.handle_collision(collision)
        if isinstance(collision.body_b.user_data, PhysicsCreature):
            collision.body_b.user_data.handle_collision(collision)
    
    def update(self, dt: float = 1.0/60.0) -> None:
        """Update simulation.
        
        Args:
            dt: Time step in seconds
        """
        if self.paused:
            return
        
        # Step physics world
        self.physics_world.step(dt)
        
        # Update resource manager (regrowth)
        self.resource_manager.update(1.0)  # Use unit timestep for gameplay
        
        # Update creatures
        for creature in self.creatures:
            alive = creature.update(dt=1.0, resource_manager=self.resource_manager)
            if not alive:
                print(f"💀 Creature died")
        
        # Remove dead creatures
        self.creatures = [c for c in self.creatures if c.energy.energy > 0]
        
        self.timestep += 1
    
    def render_ground(self) -> None:
        """Render ground plane."""
        glDisable(GL_LIGHTING)
        glColor4f(0.1, 0.15, 0.2, 1.0)
        glBegin(GL_QUADS)
        hw = self.world_width / 2
        hh = self.world_height / 2
        glVertex3f(-hw, -hh, 0)
        glVertex3f(hw, -hh, 0)
        glVertex3f(hw, hh, 0)
        glVertex3f(-hw, hh, 0)
        glEnd()
        
        # Grid lines
        glColor4f(0.2, 0.25, 0.3, 0.5)
        glBegin(GL_LINES)
        grid_spacing = 50.0
        for x in np.arange(-hw, hw + grid_spacing, grid_spacing):
            glVertex3f(x, -hh, 0.1)
            glVertex3f(x, hh, 0.1)
        for y in np.arange(-hh, hh + grid_spacing, grid_spacing):
            glVertex3f(-hw, y, 0.1)
            glVertex3f(hw, y, 0.1)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def render_resources(self) -> None:
        """Render all resources."""
        for resource in self.resource_manager.resources:
            if not resource.active:
                continue
            
            glPushMatrix()
            glTranslatef(resource.x, resource.y, resource.z + resource.radius)
            
            if resource.resource_type == ResourceType.FOOD:
                # Food as green sphere
                glColor4f(0.0, 1.0, 0.0, 0.8)
                draw_sphere(resource.radius, slices=8, stacks=8)
            
            elif resource.resource_type == ResourceType.DRUG_MUSHROOM:
                # Drug mushroom
                glColor4f(0.9, 0.9, 0.8, 0.9)
                glPushMatrix()
                glRotatef(-90, 1, 0, 0)
                draw_cylinder(resource.radius * 0.3, resource.radius * 0.3, 
                            resource.radius * 1.5, slices=8)
                glPopMatrix()
                
                colors = [
                    (0.5, 0.0, 1.0), (1.0, 0.0, 0.5), (1.0, 0.5, 0.0),
                    (0.0, 0.5, 1.0), (1.0, 1.0, 0.0),
                ]
                color = colors[resource.molecule_type] if resource.molecule_type is not None else (1.0, 1.0, 1.0)
                glColor4f(*color, 0.9)
                glPushMatrix()
                glTranslatef(0, 0, resource.radius * 1.5)
                draw_sphere(resource.radius, slices=8, stacks=8)
                glPopMatrix()
            
            glPopMatrix()
    
    def render_creatures(self) -> None:
        """Render all creatures."""
        for creature in self.creatures:
            if not hasattr(creature, 'mesh') or creature.mesh is None:
                creature.mesh = self.mesh_generator.generate_creature_mesh(creature.body)
            
            glPushMatrix()
            glTranslatef(creature.x, creature.y, creature.z)
            
            # Drug-responsive scale
            scale = creature.get_render_scale() if hasattr(creature, 'get_render_scale') else 1.0
            glScalef(scale, scale, scale)
            
            # Base body scale
            body_scale = 5.0
            glScalef(body_scale, body_scale, body_scale)
            
            # Render mesh
            creature.mesh.render()
            
            glPopMatrix()
            
            # Render velocity vector
            if creature.rigid_body:
                vel = creature.get_velocity()
                vel_mag = np.linalg.norm(vel)
                if vel_mag > 0.1:
                    glDisable(GL_LIGHTING)
                    glBegin(GL_LINES)
                    glColor4f(1.0, 1.0, 0.0, 0.7)
                    glVertex3f(creature.x, creature.y, creature.z)
                    glVertex3f(creature.x + vel[0] * 2, creature.y + vel[1] * 2, creature.z + vel[2] * 2)
                    glEnd()
                    glEnable(GL_LIGHTING)
    
    def render_3d(self) -> None:
        """Render 3D scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Setup 3D projection
        self.camera.apply_projection(self.screen_width, self.screen_height)
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
        
        y_offset = 10
        texts = [
            f"Phase 10a: Physics-Based ALife | Time: {self.timestep} | FPS: {int(self.clock.get_fps())}",
            f"Creatures: {len(self.creatures)} | Resources: {len([r for r in self.resource_manager.resources if r.active])}",
            f"Physics Bodies: {len(self.physics_world.bodies)} | Collisions: {self.collision_count}",
        ]
        
        if self.paused:
            texts.append("⏸ PAUSED")
        
        # Show first creature physics state
        if self.creatures:
            creature = self.creatures[0]
            vel = creature.get_velocity()
            vel_mag = np.linalg.norm(vel)
            texts.extend([
                "",
                f"Creature 0 Physics:",
                f"  Position: ({creature.x:.1f}, {creature.y:.1f}, {creature.z:.1f})",
                f"  Velocity: {vel_mag:.2f} m/s",
                f"  Mass: {creature.rigid_body.mass:.2f} kg" if creature.rigid_body else "  No physics body",
                f"  Energy: {creature.energy.energy:.0f}",
            ])
        
        for text in texts:
            self.render_text(text, 10, y_offset, self.small_font)
            y_offset += 20
        
        # Controls at bottom
        controls = [
            "Space: Pause | I: Push creatures | F: Food | D: Drug | S: Stats | R: Reset camera"
        ]
        y_offset = self.screen_height - 30
        for text in controls:
            self.render_text(text, 10, y_offset, self.small_font)
            y_offset += 20
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def render_text(self, text: str, x: int, y: int, font: pygame.font.Font) -> None:
        """Render text to screen."""
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
        """Handle input events."""
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
                elif event.key == K_i:
                    self.apply_random_impulses()
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
                if event.button == 1:
                    self.mouse_dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:
                    self.camera.handle_zoom(10)
                elif event.button == 5:
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
    
    def spawn_food(self):
        """Spawn food with physics body."""
        x = np.random.uniform(-self.world_width/2, self.world_width/2)
        y = np.random.uniform(-self.world_height/2, self.world_height/2)
        food = create_food(x, y, 0.0)
        self.resource_manager.resources.append(food)
        
        # Create physics body
        body = self.physics_world.create_sphere_body(
            position=(x, y, food.radius),
            radius=food.radius,
            mass=0.1,
            fixed=True,
            user_data=food
        )
        body.collision_group = 2
        body.collision_mask = 1
        
        print(f"🍎 Spawned food at ({x:.1f}, {y:.1f})")
    
    def spawn_drug(self):
        """Spawn drug mushroom with physics body."""
        x = np.random.uniform(-self.world_width/2, self.world_width/2)
        y = np.random.uniform(-self.world_height/2, self.world_height/2)
        molecule_type = np.random.randint(0, 5)
        drug = create_drug_mushroom(x, y, 0.0, molecule_type)
        self.resource_manager.resources.append(drug)
        
        # Create physics body
        body = self.physics_world.create_sphere_body(
            position=(x, y, drug.radius),
            radius=drug.radius,
            mass=0.1,
            fixed=True,
            user_data=drug
        )
        body.collision_group = 2
        body.collision_mask = 1
        
        print(f"🍄 Spawned drug type {molecule_type} at ({x:.1f}, {y:.1f})")
    
    def apply_random_impulses(self):
        """Apply random impulses to all creatures (push them)."""
        for creature in self.creatures:
            impulse = np.random.uniform(-500, 500, size=3).astype(np.float32)
            impulse[2] = abs(impulse[2])  # Always push up
            creature.apply_impulse(impulse)
        print("💥 Applied random impulses to all creatures!")
    
    def give_drug_to_all(self, molecule_type: int):
        """Give drug to all creatures."""
        for creature in self.creatures:
            creature.drugs.tripping[molecule_type] += 5.0
        print(f"💊 Gave all creatures drug type {molecule_type}")
    
    def print_statistics(self):
        """Print detailed simulation statistics."""
        print("\n" + "="*60)
        print(f"Phase 10a Statistics - Timestep {self.timestep}")
        print("="*60)
        
        print(f"\nPhysics:")
        print(f"  Total bodies: {len(self.physics_world.bodies)}")
        print(f"  Collisions handled: {self.collision_count}")
        
        print(f"\nCreatures: {len(self.creatures)}")
        for i, creature in enumerate(self.creatures[:3]):
            vel = creature.get_velocity()
            print(f"\n  Creature {i}:")
            print(f"    Position: ({creature.x:.1f}, {creature.y:.1f}, {creature.z:.1f})")
            print(f"    Velocity: {np.linalg.norm(vel):.2f} m/s")
            print(f"    Energy: {creature.energy.energy:.0f}")
            if creature.rigid_body:
                print(f"    Mass: {creature.rigid_body.mass:.2f} kg")
    
    def run(self) -> None:
        """Run demo main loop."""
        print("\n🎮 Phase 10a Demo Controls:")
        print("  Mouse drag: Rotate camera")
        print("  Mouse wheel: Zoom")
        print("  Arrow keys: Pan camera")
        print("  R: Reset camera")
        print("  Space: Pause/unpause")
        print("  I: Apply random impulses (push creatures)")
        print("  F: Spawn food")
        print("  D: Spawn drug mushroom")
        print("  1-5: Give drug to creatures")
        print("  S: Print detailed statistics")
        print("  Q/Esc: Quit")
        print("\n🧬 Starting physics-based simulation...\n")
        
        running = True
        while running:
            running = self.handle_events()
            self.update(dt=1.0/60.0)
            self.render()
            self.clock.tick(self.target_fps)
        
        pygame.quit()


def main():
    """Run Phase 10a demo."""
    demo = Phase10aDemo(n_creatures=6)
    demo.run()


if __name__ == '__main__':
    main()
