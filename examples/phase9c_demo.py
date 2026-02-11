"""
Phase 9c Demo: Collective Learning via Circuit8

Demonstrates emergent social intelligence:
- Creatures broadcast behaviors to Circuit8
- Resource marking (food/drugs)
- Danger warnings
- Social learning from observations
- Collective memory building
- Emergent cooperation patterns

Watch creatures develop shared knowledge through
the telepathic canvas!

Controls:
- Mouse drag: Rotate camera (3D mode)
- Mouse wheel: Zoom
- WASD: Pan camera (2D mode)
- Space: Pause/unpause
- R: Reset simulation
- C: Toggle camera mode (2D/3D)
- F: Toggle fade Circuit8
- M: Toggle collective memory display
- I: Apply random impulses to creatures
- T: Toggle thought bubbles
- P: Toggle psychedelic patterns
- A: Toggle audio synthesis
- 1-8: Toggle render modes
- ESC: Quit

Render modes:
1: Creatures only
2: Neural networks
3: Circuit8 canvas
4: Resources
5: Physics debug
6: Collective signals
7: Social learning
8: All modes
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

from core.morphic.circuit8 import Circuit8
from core.physics.physics_world import PhysicsWorld
from core.resources.resource_manager import ResourceManager
from core.collective import CollectiveMemory, fade_circuit8
from creatures.collective_creature import create_collective_creatures


class Phase9cDemo:
    """Phase 9c demonstration of collective intelligence."""
    
    def __init__(self, width=1280, height=720):
        """Initialize demo."""
        # Pygame and OpenGL setup
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode(
            (width, height),
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption("Phase 9c: Collective Learning via Circuit8")
        
        # Core systems
        self.circuit8 = Circuit8(width=64, height=48)
        
        # Initialize Circuit8 with visible background pattern (shared hallucinatory space seed)
        # This makes the collective unconscious VISIBLE from the start
        for y in range(self.circuit8.height):
            for x in range(self.circuit8.width):
                # Brighter gradient pattern so it's actually visible
                intensity = int(80 * np.sin(x * 0.15) * np.cos(y * 0.15))
                r = max(0, min(255, 60 + intensity))
                g = max(0, min(255, 40 + intensity))
                b = max(0, min(255, 80 + intensity))
                self.circuit8.write_pixel(x, y, r, g, b, blend=False)
        
        self.physics_world = PhysicsWorld(
            world_bounds=(-250, -250, 250, 250),
            gravity=(0, 0, -9.8)
        )
        self.collective_memory = CollectiveMemory()
        
        # Resources
        self.resource_manager = ResourceManager(
            world_width=500.0,
            world_height=500.0
        )
        self.resource_manager.spawn_initial_resources(
            food_count=20,
            drug_count=10
        )
        
        # Create collective creatures (now with thoughts!)
        self.creatures = create_collective_creatures(
            n_creatures=8,
            physics_world=self.physics_world,
            circuit8=self.circuit8,
            collective_memory=self.collective_memory,
            world_bounds=(-200, -200, 200, 200)
        )
        
        # Camera state
        self.camera_distance = 400.0
        self.camera_rotation = 45.0
        self.camera_elevation = 30.0
        self.camera_pan_x = 0.0
        self.camera_pan_y = 0.0
        self.dragging = False
        self.last_mouse_pos = None
        
        # UI state
        self.paused = False
        self.render_mode = 8  # All modes
        self.show_memory = True
        self.fade_enabled = False  # Disable fade so Circuit8 colors accumulate and are visible!
        self.camera_mode = '3d'
        self.show_help = False  # Help overlay
        self.time_speed = 1.0  # Time speed multiplier (0.1x to 10x)
        self.show_thoughts = True  # Show creature thoughts
        self.psychedelic_patterns_enabled = False  # Psychedelic visual patterns
        self.audio_enabled = False  # Audio synthesis
        self.inspected_creature = None  # Currently inspected creature
        
        # Timing
        self.clock = pygame.time.Clock()
        self.timestep = 0
        self.fps_target = 60
        
        # Statistics
        self.stats = {
            'total_observations': 0,
            'total_learnings': 0,
            'resources_marked': 0,
            'danger_warnings': 0,
        }
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                elif event.key == K_SPACE:
                    self.paused = not self.paused
                elif event.key == K_r:
                    self.reset()
                elif event.key == K_c:
                    self.camera_mode = '3d' if self.camera_mode == '2d' else '2d'
                elif event.key == K_f:
                    self.fade_enabled = not self.fade_enabled
                elif event.key == K_m:
                    self.show_memory = not self.show_memory
                elif event.key == K_h:
                    self.show_help = not self.show_help
                elif event.key == K_i:
                    self.apply_random_impulses()
                elif event.key == K_1:
                    self.render_mode = 1
                elif event.key == K_2:
                    self.render_mode = 2
                elif event.key == K_3:
                    self.render_mode = 3
                elif event.key == K_4:
                    self.render_mode = 4
                elif event.key == K_5:
                    self.render_mode = 5
                elif event.key == K_6:
                    self.render_mode = 6
                elif event.key == K_7:
                    self.render_mode = 7
                elif event.key == K_8:
                    self.render_mode = 8
                elif event.key == K_LEFTBRACKET:  # [ key
                    self.time_speed = max(0.1, self.time_speed / 1.5)
                elif event.key == K_RIGHTBRACKET:  # ] key
                    self.time_speed = min(10.0, self.time_speed * 1.5)
                elif event.key == K_t:
                    self.show_thoughts = not self.show_thoughts
                elif event.key == K_p:
                    # Toggle psychedelic patterns for all creatures
                    self.psychedelic_patterns_enabled = not self.psychedelic_patterns_enabled
                    for creature in self.creatures:
                        creature.pattern_generation_enabled = self.psychedelic_patterns_enabled
                        if self.psychedelic_patterns_enabled and not hasattr(creature, 'pattern_gen'):
                            creature.init_psychedelic_vision(enable_patterns=True)
                elif event.key == K_a:
                    # Toggle audio synthesis for all creatures
                    self.audio_enabled = not self.audio_enabled
                    for creature in self.creatures:
                        if self.audio_enabled and not creature.audio_enabled:
                            creature.init_audio_synthesis(enable_audio=True)
                        creature.audio_enabled = self.audio_enabled
            
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Check if clicking on a creature (right click for inspect)
                    self.dragging = True
                    self.last_mouse_pos = event.pos
                elif event.button == 3:  # Right click - inspect creature
                    self.try_inspect_creature(event.pos)
                elif event.button == 4:  # Scroll up
                    self.camera_distance = max(100, self.camera_distance - 20)
                elif event.button == 5:  # Scroll down
                    self.camera_distance = min(1000, self.camera_distance + 20)
            
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
                    self.last_mouse_pos = None
            
            elif event.type == MOUSEMOTION:
                if self.dragging and self.last_mouse_pos:
                    dx = event.pos[0] - self.last_mouse_pos[0]
                    dy = event.pos[1] - self.last_mouse_pos[1]
                    self.camera_rotation += dx * 0.3
                    self.camera_elevation = max(-89, min(89, self.camera_elevation - dy * 0.3))
                    self.last_mouse_pos = event.pos
        
        # Handle held keys for camera pan
        keys = pygame.key.get_pressed()
        if keys[K_w]:
            self.camera_pan_y += 5
        if keys[K_s]:
            self.camera_pan_y -= 5
        if keys[K_a]:
            self.camera_pan_x += 5
        if keys[K_d]:
            self.camera_pan_x -= 5
        
        return True
    
    def apply_random_impulses(self):
        """Apply random impulses to all creatures."""
        for creature in self.creatures:
            if creature.rigid_body:
                impulse = np.random.uniform(-500, 500, size=3).astype(np.float32)
                impulse[2] = abs(impulse[2])  # Up only
                creature.apply_impulse(impulse)
    
    def try_inspect_creature(self, mouse_pos):
        """Try to pick and inspect a creature at mouse position."""
        # Get ray from mouse position
        ray_origin, ray_dir = self.get_ray_from_mouse(mouse_pos)
        
        # Find closest creature hit
        closest_creature = None
        closest_dist = float('inf')
        
        for creature in self.creatures:
            # Simple sphere intersection test
            pos = np.array([creature.x, creature.y, creature.z if hasattr(creature, 'z') else 10.0])
            radius = 15.0  # Approximate creature radius
            
            # Ray-sphere intersection
            oc = ray_origin - pos
            b = 2.0 * np.dot(oc, ray_dir)
            c = np.dot(oc, oc) - radius * radius
            discriminant = b*b - 4*c
            
            if discriminant >= 0:
                t = (-b - np.sqrt(discriminant)) / 2.0
                if 0 < t < closest_dist:
                    closest_dist = t
                    closest_creature = creature
        
        # Toggle inspection
        if closest_creature:
            if self.inspected_creature == closest_creature:
                self.inspected_creature = None  # Close inspector
            else:
                self.inspected_creature = closest_creature
                print(f"\n🔍 Inspecting creature {closest_creature.creature_id}")
    
    def get_ray_from_mouse(self, mouse_pos):
        """Get ray from mouse position in 3D space."""
        # Get viewport and matrices
        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        # Mouse Y is inverted
        win_x = float(mouse_pos[0])
        win_y = float(viewport[3] - mouse_pos[1])
        
        # Unproject near and far points
        near_pt = gluUnProject(win_x, win_y, 0.0, modelview, projection, viewport)
        far_pt = gluUnProject(win_x, win_y, 1.0, modelview, projection, viewport)
        
        # Create ray
        ray_origin = np.array(near_pt, dtype=np.float32)
        ray_dir = np.array(far_pt, dtype=np.float32) - ray_origin
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        return ray_origin, ray_dir
    
    def reset(self):
        """Reset simulation."""
        self.timestep = 0
        
        # Reset Circuit8
        self.circuit8 = Circuit8(width=64, height=48)
        
        # Reset physics world
        self.physics_world = PhysicsWorld(
            world_bounds=(-250, -250, 250, 250),
            gravity=(0, 0, -9.8)
        )
        
        # Reset collective memory
        self.collective_memory = CollectiveMemory()
        
        # Reset resources
        self.resource_manager = ResourceManager(
            world_width=500.0,
            world_height=500.0
        )
        self.resource_manager.spawn_initial_resources(
            food_count=20,
            drug_count=10
        )
        
        # Reset creatures
        self.creatures = create_collective_creatures(
            n_creatures=8,
            physics_world=self.physics_world,
            circuit8=self.circuit8,
            collective_memory=self.collective_memory,
            world_bounds=(-200, -200, 200, 200)
        )
        
        # Reset stats
        self.stats = {
            'total_observations': 0,
            'total_learnings': 0,
            'resources_marked': 0,
            'danger_warnings': 0,
        }
    
    def update(self, dt: float):
        """Update simulation."""
        if self.paused:
            return
        
        # Fade Circuit8 (less aggressive so patterns persist)
        if self.fade_enabled and self.timestep % 10 == 0:
            fade_circuit8(self.circuit8, fade_rate=0.98)
        
        # Update creatures (handle birth and death)
        new_offspring = []
        for creature in self.creatures[:]:
            # Generate audio if enabled (call AFTER update so network is fresh)
            if self.audio_enabled and creature.audio_enabled:
                creature.generate_audio(duration_seconds=dt)
            
            result = creature.update(dt, self.resource_manager)
            
            # Check if result is offspring (reproduction occurred)
            if result and result is not True and result != creature:
                # It's an offspring!
                new_offspring.append(result)
                result = True  # Parent is still alive
            
            # Check if creature died
            if not result:
                self.creatures.remove(creature)
                # Record death for statistics
                print(f"Creature {creature.creature_id} died at age {creature.age}, generation {creature.generation}")
        
        # Add new offspring to population
        if new_offspring:
            print(f"\n🎉 {len(new_offspring)} creature(s) born! Population: {len(self.creatures)} → {len(self.creatures) + len(new_offspring)}")
            self.creatures.extend(new_offspring)
        
        # Update resources
        self.resource_manager.update(dt)
        
        # Update physics
        self.physics_world.step(dt)
        
        # Collect statistics
        self.update_statistics()
        
        self.timestep += 1
    
    def update_statistics(self):
        """Update statistics from creatures."""
        self.stats['total_observations'] = 0
        self.stats['total_learnings'] = 0
        
        for creature in self.creatures:
            self.stats['total_observations'] += len(creature.learner.observation_history)
            self.stats['total_learnings'] += len(creature.learner.behavior_success_rates)
        
        # Memory stats (count directly from lists)
        self.stats['resources_marked'] = len(self.collective_memory.resource_locations)
        self.stats['danger_warnings'] = len(self.collective_memory.danger_zones)
    
    def render(self):
        """Render scene."""
        # Clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Setup camera
        self.setup_camera()
        
        # Render based on mode
        if self.render_mode in [1, 8]:  # Creatures
            for creature in self.creatures:
                self.render_creature(creature)
        
        if self.render_mode in [3, 8]:  # Circuit8
            if self.timestep % 60 == 0:  # Print once per second
                print(f"[DEBUG] Rendering Circuit8 at timestep {self.timestep}")
            self.render_circuit8_ground(self.circuit8)
        
        if self.render_mode in [4, 8]:  # Resources
            for resource in self.resource_manager.resources:
                self.render_resource(resource)
        
        if self.render_mode in [6, 8] and self.show_memory:  # Collective signals
            self.render_collective_signals()
        
        if self.show_thoughts:  # Thought bubbles
            self.render_thoughts()
        
        # Render UI
        self.render_ui()
        
        pygame.display.flip()
    
    def setup_camera(self):
        """Setup 3D camera."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, self.width / self.height, 1.0, 5000.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Calculate camera position from spherical coordinates
        az_rad = np.radians(self.camera_rotation)
        el_rad = np.radians(self.camera_elevation)
        
        cam_x = self.camera_distance * np.cos(el_rad) * np.cos(az_rad)
        cam_y = self.camera_distance * np.sin(el_rad)
        cam_z = self.camera_distance * np.cos(el_rad) * np.sin(az_rad)
        
        gluLookAt(
            cam_x + self.camera_pan_x, cam_y, cam_z + self.camera_pan_y,  # Eye
            self.camera_pan_x, 0, self.camera_pan_y,  # Target
            0, 1, 0  # Up
        )
    
    def render_creature(self, creature):
        """Render a creature with its procedural 3D body."""
        glPushMatrix()
        glTranslatef(creature.x, creature.y, creature.z if hasattr(creature, 'z') else 10.0)
        
        # Color based on energy
        energy_frac = min(creature.energy.energy / 1000000.0, 1.0)
        r = 0.2 + energy_frac * 0.5
        g = 0.5 + energy_frac * 0.3
        b = 0.8
        
        # Apply drug effects to color
        if hasattr(creature, 'drugs') and creature.drugs is not None:
            trip_total = creature.drugs.tripping.sum()
            if trip_total > 0.1:
                # Psychedelic color shift
                r += np.sin(self.timestep * 0.1) * 0.3
                g += np.sin(self.timestep * 0.1 + 2.09) * 0.3
                b += np.sin(self.timestep * 0.1 + 4.18) * 0.3
        
        glColor3f(max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b)))
        
        # Render procedural body if available
        if hasattr(creature, 'body') and creature.body is not None:
            self.render_procedural_body(creature)
        else:
            # Fallback to simple sphere
            radius = creature.rigid_body.radius if hasattr(creature, 'rigid_body') and creature.rigid_body else 5.0
            quad = gluNewQuadric()
            gluSphere(quad, radius, 16, 16)
            gluDeleteQuadric(quad)
        
        glPopMatrix()
    
    def render_procedural_body(self, creature):
        """Render creature's procedural 3D body from genotype."""
        body = creature.body
        
        # Scale for drug effects
        scale = 1.0
        if hasattr(creature, 'drugs') and creature.drugs is not None:
            trip_total = creature.drugs.tripping.sum()
            if trip_total > 0.1:
                scale = 1.0 + 0.2 * np.sin(self.timestep * 0.15 + trip_total)
        
        glScalef(scale, scale, scale)
        
        # Render head
        glPushMatrix()
        glTranslatef(0, 0, body.head_size * 1.5)
        quad = gluNewQuadric()
        gluSphere(quad, body.head_size, 8, 8)
        gluDeleteQuadric(quad)
        glPopMatrix()
        
        # Render body segments
        z_offset = 0
        for i, seg in enumerate(body.segments):
            segment_radius = seg.size
            segment_length = seg.length * seg.size  # Scale length by size
            
            # Segment body (cylinder)
            glPushMatrix()
            glTranslatef(0, 0, z_offset)
            glRotatef(90, 1, 0, 0)  # Orient along Z axis
            quad = gluNewQuadric()
            gluCylinder(quad, segment_radius, segment_radius * 0.8, segment_length, 8, 1)
            gluDeleteQuadric(quad)
            glPopMatrix()
            
            # Limbs attached to this segment
            for limb in seg.limbs:
                # Calculate limb position on segment
                angle_rad = np.radians(limb.angle_horizontal)
                lx = segment_radius * np.cos(angle_rad)
                ly = segment_radius * np.sin(angle_rad)
                
                glPushMatrix()
                glTranslatef(lx, ly, z_offset + segment_length * 0.5)
                
                # Limb orientation
                glRotatef(limb.angle_horizontal, 0, 0, 1)
                glRotatef(limb.angle_vertical, 0, 1, 0)
                
                # Draw limb as tapered cylinder
                limb_length = limb.length * segment_radius
                limb_start_width = limb.width * segment_radius
                limb_end_width = limb_start_width * limb.taper
                
                quad = gluNewQuadric()
                gluCylinder(quad, limb_start_width, limb_end_width, limb_length, 6, 1)
                gluDeleteQuadric(quad)
                glPopMatrix()
            
            z_offset -= segment_length
        
        # Render tail if present
        if body.tail_length > 0:
            tail_radius = body.segments[-1].size * 0.5 if body.segments else 0.5
            glPushMatrix()
            glTranslatef(0, 0, z_offset)
            glRotatef(90, 1, 0, 0)
            quad = gluNewQuadric()
            gluCylinder(quad, tail_radius, tail_radius * 0.2, body.tail_length * tail_radius, 6, 1)
            gluDeleteQuadric(quad)
            glPopMatrix()
    
    def render_circuit8_ground(self, circuit8):
        """
        Render Circuit8 - bright glowing ground plane.
        """
        # Disable depth test so it always shows
        glDisable(GL_DEPTH_TEST)
        
        glPushMatrix()
        glTranslatef(0, 0, 0)  # At origin
        
        # Draw BRIGHT plane that changes color
        size = 500.0
        
        # Sample a few pixels to get color
        r, g, b = circuit8.read_pixel(32, 24)  # Center pixel
        r_f = float(r) / 255.0
        g_f = float(g) / 255.0  
        b_f = float(b) / 255.0
        
        # Brighten it significantly
        r_f = min(1.0, r_f * 3.0 + 0.3)
        g_f = min(1.0, g_f * 3.0 + 0.3)
        b_f = min(1.0, b_f * 3.0 + 0.3)
        
        glColor3f(r_f, g_f, b_f)
        glBegin(GL_QUADS)
        glVertex3f(-size, -size, 0)
        glVertex3f(size, -size, 0)
        glVertex3f(size, size, 0)
        glVertex3f(-size, size, 0)
        glEnd()
        
        # Bright yellow border
        glColor3f(1.0, 1.0, 0.0)
        glLineWidth(10.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-size, -size, 0.1)
        glVertex3f(size, -size, 0.1)
        glVertex3f(size, size, 0.1)
        glVertex3f(-size, size, 0.1)
        glEnd()
        glLineWidth(1.0)
        
        glPopMatrix()
        
        # Re-enable depth test
        glEnable(GL_DEPTH_TEST)
    
    def render_resource(self, resource):
        """Render a resource."""
        from core.resources.resource import ResourceType
        
        # Color by type
        if resource.resource_type == ResourceType.FOOD:
            color = (0.2, 0.8, 0.2)  # Green
        elif resource.resource_type == ResourceType.DRUG_MUSHROOM:
            color = (0.8, 0.2, 0.8)  # Magenta
        else:
            color = (0.5, 0.5, 0.5)  # Gray
        
        # Size by amount
        radius = resource.amount ** 0.333
        
        glPushMatrix()
        glTranslatef(resource.x, resource.y, 5.0)
        glColor3f(*color)
        
        quad = gluNewQuadric()
        gluSphere(quad, radius, 8, 8)
        gluDeleteQuadric(quad)
        
        glPopMatrix()
    
    def render_collective_signals(self):
        """Render collective intelligence visualization."""
        # Draw resource markers from memory
        try:
            for record in self.collective_memory.resource_locations:
                if not isinstance(record, dict) or 'location' not in record:
                    continue
                    
                x, y = record['location']
                resource_type = record.get('resource_type', 'unknown')
                
                # Color by type
                if resource_type == 'food':
                    color = (0.0, 1.0, 0.0, 0.5)  # Green
                else:
                    color = (1.0, 0.0, 1.0, 0.5)  # Magenta
                
                # Draw marker
                glPushMatrix()
                glTranslatef(x, y, 2.0)
                glColor4f(*color)
                glBegin(GL_LINES)
                glVertex3f(-3, 0, 0)
                glVertex3f(3, 0, 0)
                glVertex3f(0, -3, 0)
                glVertex3f(0, 3, 0)
                glEnd()
                glPopMatrix()
        except Exception as e:
            pass  # Skip if collective memory has issues
        
        # Draw danger zones
        try:
            for record in self.collective_memory.danger_zones:
                if not isinstance(record, dict) or 'location' not in record:
                    continue
                    
                x, y = record['location']
                
                glPushMatrix()
                glTranslatef(x, y, 1.0)
                glColor4f(1.0, 0.0, 0.0, 0.3)  # Red
                
                # Draw circle
                glBegin(GL_LINE_LOOP)
                for i in range(16):
                    angle = i * 2 * np.pi / 16
                    glVertex3f(10 * np.cos(angle), 10 * np.sin(angle), 0)
                glEnd()
                glPopMatrix()
        except Exception as e:
            pass  # Skip if collective memory has issues
    
    def render_thoughts(self):
        """Render creature thoughts as floating text."""
        # Collect thoughts to render (in 2D after switching to UI mode)
        if not hasattr(self, '_thoughts_to_render'):
            self._thoughts_to_render = []
        
        self._thoughts_to_render = []
        for creature in self.creatures:
            thought = creature.get_current_thought()
            if thought and len(thought.strip()) > 0:
                # Store position for 2D rendering later
                self._thoughts_to_render.append({
                    'x': creature.x,
                    'y': creature.y,
                    'z': creature.z if hasattr(creature, 'z') else 10.0,
                    'text': thought
                })
    
    def render_ui(self):
        """Render UI overlay."""
        # Switch to 2D for UI
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, self.height, 0)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Create fonts if needed
        if not hasattr(self, 'font'):
            self.font = pygame.font.Font(None, 24)
        if not hasattr(self, 'title_font'):
            self.title_font = pygame.font.Font(None, 32)
        
        # Render status text using texture-based approach (readable!)
        self.render_status_text()
        
        # Creature inspector panel
        if self.inspected_creature:
            self.render_inspector_panel()
        
        # Help overlay or prompt
        if self.show_help:
            self.render_help_overlay()
        
        glEnable(GL_DEPTH_TEST)
        
        # Restore 3D
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def render_status_text(self):
        """Render status text overlay using texture mapping for readability."""
        # Create status lines
        status_lines = [
            f"Phase 9d: Complete Artificial Life",
            f"Timestep: {self.timestep}",
            f"Creatures: {len(self.creatures)} (Dynamic population)",
            f"Resources: {len(self.resource_manager.resources)}",
            f"",
            f"🌌 SHARED HALLUCINATORY SPACE (Circuit8):",
            f"  Ground plane = collective unconscious",
            f"  256 visual neurons per creature",
            f"  Patterns persist across lifetimes",
            f"",
            f"Collective Intelligence:",
            f"  Observations: {self.stats['total_observations']}",
            f"  Learned behaviors: {self.stats['total_learnings']}",
            f"  Resources marked: {self.stats['resources_marked']}",
            f"  Danger zones: {self.stats['danger_warnings']}",
            f"",
            f"Mode: {self.render_mode} | Camera: {self.camera_mode}",
            f"Thoughts: {'ON' if self.show_thoughts else 'OFF'} (T)",
            f"Patterns: {'ON' if self.psychedelic_patterns_enabled else 'OFF'} (P - ENABLE THIS!)",
            f"Audio: {'ON' if self.audio_enabled else 'OFF'} (A)",
            f"Time: {self.time_speed:.1f}x ([ ])",
            f"{'PAUSED' if self.paused else 'RUNNING'}",
            f"",
            f"Press H for Help | Right-click to Inspect",
        ]
        
        # Create transparent surface for status text
        status_surface = pygame.Surface((400, 450), pygame.SRCALPHA)
        status_surface.fill((0, 0, 0, 0))  # Fully transparent
        
        # Render each line to surface
        y = 10
        for line in status_lines:
            if line:  # Only render non-empty lines
                # Different color for help prompt
                if "Press H" in line:
                    color = (255, 255, 0)
                else:
                    color = (255, 255, 255)
                text_surf = self.font.render(line, True, color)
                status_surface.blit(text_surf, (10, y))
            y += 25
        
        # Flip vertically for OpenGL
        status_surface = pygame.transform.flip(status_surface, False, True)
        
        # Convert to texture
        texture_data = pygame.image.tostring(status_surface, "RGBA", False)
        
        # Enable texture mapping
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Create or reuse texture
        if not hasattr(self, 'status_texture_id'):
            self.status_texture_id = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, self.status_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 400, 450, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        
        # Render textured quad in top-left corner
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(0, 0)
        glTexCoord2f(1, 1); glVertex2f(400, 0)
        glTexCoord2f(1, 0); glVertex2f(400, 450)
        glTexCoord2f(0, 0); glVertex2f(0, 450)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        
        # Render thought bubbles if enabled
        if self.show_thoughts and hasattr(self, '_thoughts_to_render'):
            self.render_thought_bubbles()
    
    def render_thought_bubbles(self):
        """Render thought bubbles in 2D UI space."""
        # Project 3D positions to 2D screen space
        from OpenGL.GLU import gluProject
        
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        
        for thought_data in self._thoughts_to_render:
            # Project 3D world position to 2D screen
            try:
                screen_x, screen_y, screen_z = gluProject(
                    thought_data['x'],
                    thought_data['y'],
                    thought_data['z'] + 15.0,  # Above creature
                    modelview,
                    projection,
                    viewport
                )
                
                # Flip Y coordinate (OpenGL vs screen coordinates)
                screen_y = self.height - screen_y
                
                # Only render if in front of camera
                if 0 < screen_z < 1.0:
                    # Render thought text
                    text = thought_data['text'][:30]  # Limit length
                    if len(thought_data['text']) > 30:
                        text += "..."
                    
                    text_surf = self.font.render(text, True, (255, 255, 100))  # Yellow thoughts
                    text_data = pygame.image.tostring(text_surf, "RGBA", True)
                    
                    # Center text above creature
                    text_x = int(screen_x - text_surf.get_width() / 2)
                    text_y = int(screen_y)
                    
                    # Ensure on screen
                    if 0 <= text_x < self.width and 0 <= text_y < self.height:
                        glRasterPos2f(text_x, text_y)
                        glDrawPixels(
                            text_surf.get_width(),
                            text_surf.get_height(),
                            GL_RGBA,
                            GL_UNSIGNED_BYTE,
                            text_data
                        )
            except:
                pass  # Skip if projection fails
    
    def render_inspector_panel(self):
        """Render creature inspector panel in right side of screen."""
        c = self.inspected_creature
        
        # Build inspector info
        inspector_lines = [
            f"🔍 CREATURE {c.creature_id}",
            "",
            f"VITALS:",
            f"  Age: {c.age} | Generation: {c.generation}",
            f"  Energy: {c.energy.energy:,.0f} / {c.energy.max_energy:,.0f}",
            f"  Position: ({c.x:.0f}, {c.y:.0f}, {c.z if hasattr(c, 'z') else 0:.0f})",
            f"  Adam Distance: {c.adam_distance if hasattr(c, 'adam_distance') else 'N/A'}",
            "",
            f"BRAIN:",
            f"  Neurons: {len(c.network.neurons)}",
            f"  Synapses: {len(c.network.synapses)}",
            f"  Activity: {c.network.get_activity_level()*100:.1f}%",
            f"  Plasticity: {'ENABLED' if c.network.enable_plasticity else 'DISABLED'}",
            "",
            f"DRUGS:",
        ]
        
        # Drug levels
        drug_names = ["Inhib Antag", "Inhib Agon", "Excit Antag", "Excit Agon", "POTENTIATOR"]
        for i, name in enumerate(drug_names):
            level = c.drugs.tripping[i]
            if level > 1.0:
                bar_len = int(min(20, level / 5000))  # Scale for display
                bar = "█" * bar_len
                inspector_lines.append(f"  {name}: {level:>8.0f} {bar}")
        
        inspector_lines.extend([
            "",
            f"BEHAVIOR:",
            f"  Current: {c.current_behavior.name if hasattr(c, 'current_behavior') else 'UNKNOWN'}",
            f"  Food consumed: {c.food_consumed if hasattr(c, 'food_consumed') else 0}",
        ])
        
        # Addiction info if present
        if hasattr(c, 'behavior') and hasattr(c.behavior, 'addiction_level'):
            inspector_lines.append(f"  Addiction: {c.behavior.addiction_level:.2f}")
            inspector_lines.append(f"  Tolerance: {c.behavior.tolerance:.2f}")
            inspector_lines.append(f"  Withdrawal: {c.behavior.withdrawal_severity:.2f}")
        
        # Thought if available
        if hasattr(c, 'last_thought') and c.last_thought:
            inspector_lines.extend([
                "",
                f"LAST THOUGHT:",
                f"  {c.last_thought[:40]}",
            ])
        
        # Social learning stats
        if hasattr(c, 'learner'):
            inspector_lines.extend([
                "",
                f"SOCIAL LEARNING:",
                f"  Observations: {len(c.learner.observation_history)}",
                f"  Behaviors learned: {len(c.learner.behavior_success_rates)}",
            ])
        
        inspector_lines.extend([
            "",
            "Right-click to close inspector",
        ])
        
        # Create surface for inspector panel
        panel_width = 450
        panel_height = min(700, len(inspector_lines) * 22 + 30)
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 200))  # Semi-transparent black
        
        # Render each line
        y = 15
        for line in inspector_lines:
            if line:
                # Color code
                if "🔍" in line:
                    color = (255, 255, 0)  # Yellow header
                elif line.endswith(":") and not line.startswith(" "):
                    color = (0, 255, 255)  # Cyan section headers
                elif "POTENTIATOR" in line:
                    color = (255, 0, 255)  # Magenta for potentiator
                elif "█" in line:
                    color = (255, 100, 255)  # Light magenta for drug bars
                else:
                    color = (255, 255, 255)  # White for data
                
                text_surf = self.font.render(line, True, color)
                panel.blit(text_surf, (15, y))
            y += 22
        
        # Flip vertically for OpenGL
        panel = pygame.transform.flip(panel, False, True)
        
        # Convert to texture
        texture_data = pygame.image.tostring(panel, "RGBA", False)
        
        # Enable texture mapping
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Create or reuse texture
        if not hasattr(self, 'inspector_texture_id'):
            self.inspector_texture_id = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, self.inspector_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, panel_width, panel_height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        
        # Render textured quad in right side of screen
        x_start = self.width - panel_width - 10
        y_start = 10
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x_start, y_start)
        glTexCoord2f(1, 1); glVertex2f(x_start + panel_width, y_start)
        glTexCoord2f(1, 0); glVertex2f(x_start + panel_width, y_start + panel_height)
        glTexCoord2f(0, 0); glVertex2f(x_start, y_start + panel_height)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
    
    def render_help_overlay(self):
        """Render help overlay with readable text using OpenGL textures."""
        # Create pygame surface with help text
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 220))  # Semi-transparent black background
        
        # Help text content
        help_lines = [
            ("PHASE 9C: COLLECTIVE LEARNING - CONTROLS", (255, 255, 0)),
            ("", (255, 255, 255)),
            ("CAMERA & INSPECTION:", (0, 255, 255)),
            ("  Mouse Drag         Rotate camera (orbital)", (255, 255, 255)),
            ("  Right Click        Inspect creature (detailed info)", (255, 255, 255)),
            ("  Mouse Wheel        Zoom in/out", (255, 255, 255)),
            ("  W/A/S/D            Pan camera", (255, 255, 255)),
            ("  C                  Toggle 2D/3D camera mode", (255, 255, 255)),
            ("", (255, 255, 255)),
            ("SIMULATION CONTROLS:", (0, 255, 255)),
            ("  Space              Pause/Unpause simulation", (255, 255, 255)),
            ("  R                  Reset simulation", (255, 255, 255)),
            ("  I                  Apply random impulses to creatures", (255, 255, 255)),
            ("  ESC                Quit", (255, 255, 255)),
            ("", (255, 255, 255)),
            ("DISPLAY CONTROLS:", (0, 255, 255)),
            ("  H                  Toggle this help", (255, 255, 255)),
            ("  F                  Toggle Circuit8 fade", (255, 255, 255)),
            ("  M                  Toggle collective memory markers", (255, 255, 255)),
            ("  T                  Toggle creature thoughts", (255, 255, 255)),
            ("  P                  Toggle psychedelic patterns", (255, 255, 255)),
            ("  A                  Toggle audio synthesis", (255, 255, 255)),
            ("  [ ]                Adjust time speed", (255, 255, 255)),
            ("  1-8                Render modes (1=creatures, 8=all)", (255, 255, 255)),
            ("", (255, 255, 255)),
            ("WHAT YOU'RE SEEING:", (0, 255, 255)),
            ("  Gray/White Spheres  Creatures with evolved 3D bodies", (255, 255, 255)),
            ("  Yellow Text         Creature thoughts (T to toggle)", (255, 255, 255)),
            ("  Green Spheres       Food resources", (255, 255, 255)),
            ("  Magenta Spheres     Drug mushrooms", (255, 255, 255)),
            ("  Green Crosses       Food locations marked by creatures", (255, 255, 255)),
            ("  Red Circles         Danger zones", (255, 255, 255)),
            ("", (255, 255, 255)),
            ("Press H to close this help", (255, 255, 0)),
        ]
        
        # Render text to overlay
        y = 50
        for text, color in help_lines:
            if text:  # Only render non-empty lines
                text_surf = self.font.render(text, True, color)
                overlay.blit(text_surf, (50, y))
            y += 25
        
        # Flip surface vertically to fix OpenGL texture orientation
        overlay = pygame.transform.flip(overlay, False, True)
        
        # Convert surface to OpenGL texture
        texture_data = pygame.image.tostring(overlay, "RGBA", False)
        
        # Enable 2D texture mapping
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Create or reuse texture
        if not hasattr(self, 'help_texture_id'):
            self.help_texture_id = glGenTextures(1)
        
        glBindTexture(GL_TEXTURE_2D, self.help_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.width, self.height, 0, 
                     GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        
        # Render full-screen textured quad
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(0, 0)
        glTexCoord2f(1, 1); glVertex2f(self.width, 0)
        glTexCoord2f(1, 0); glVertex2f(self.width, self.height)
        glTexCoord2f(0, 0); glVertex2f(0, self.height)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
    
    def print_help(self):
        """Print help to terminal."""
        print("\n" + "="*70)
        print("PHASE 9C: COLLECTIVE LEARNING - CONTROLS")
        print("="*70)
        print("\nCAMERA CONTROLS:")
        print("  Mouse Drag         Rotate camera (orbital)")
        print("  Mouse Wheel        Zoom in/out")
        print("  W/A/S/D            Pan camera")
        print("  C                  Toggle 2D/3D camera mode")
        print("\nSIMULATION CONTROLS:")
        print("  Space              Pause/Unpause simulation")
        print("  R                  Reset simulation")
        print("  I                  Apply random impulses to creatures")
        print("  ESC                Quit")
        print("\nDISPLAY CONTROLS:")
        print("  H                  Toggle this help")
        print("  F                  Toggle Circuit8 fade")
        print("  M                  Toggle collective memory markers")
        print("  1                  Show creatures only")
        print("  2                  Show neural networks")
        print("  3                  Show Circuit8 canvas")
        print("  4                  Show resources")
        print("  5                  Show physics debug")
        print("  6                  Show collective signals")
        print("  7                  Show social learning")
        print("  8                  Show all modes (default)")
        print("\nWHAT YOU'RE SEEING:")
        print("  Gray/White Spheres  Creatures with evolved 3D bodies")
        print("  Green Spheres       Food resources")
        print("  Magenta Spheres     Drug mushrooms")
        print("  Green Crosses       Food locations marked by creatures")
        print("  Magenta Crosses     Drug locations marked by creatures")
        print("  Red Circles         Danger zones (death locations)")
        print("  Ground Plane        Circuit8 telepathic canvas (64x48)")
        print("\nPress H to hide this help")
        print("="*70 + "\n")
    
    
    def run(self):
        """Run demo loop."""
        print("=" * 60)
        print("Phase 9c: Collective Learning via Circuit8")
        print("=" * 60)
        print("\nWatch creatures develop shared knowledge!")
        print("\nKey behaviors to observe:")
        print("- Creatures mark resources on Circuit8")
        print("- Other creatures navigate to marked locations")
        print("- Success/failure signals propagate")
        print("- Danger warnings cause avoidance")
        print("- Social learning accumulates over time")
        print("\nControls:")
        print("  Space: Pause/unpause")
        print("  R: Reset simulation")
        print("  C: Toggle camera mode")
        print("  F: Toggle Circuit8 fade")
        print("  M: Toggle collective memory display")
        print("  I: Apply random impulses")
        print("  1-8: Toggle render modes")
        print("  ESC: Quit")
        print("\nStarting simulation...")
        print()
        
        running = True
        while running:
            # Handle events
            running = self.handle_events()
            
            # Update (with time speed multiplier)
            dt = (1.0 / self.fps_target) * self.time_speed
            self.update(dt)
            
            # Render
            self.render()
            
            # Maintain framerate
            self.clock.tick(self.fps_target)
        
        pygame.quit()


def main():
    """Main entry point."""
    demo = Phase9cDemo()
    demo.run()


if __name__ == '__main__':
    main()
