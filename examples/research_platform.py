"""
Research Platform Demo - Phase 10 Complete Integration

Professional artificial life research platform with:
- Runtime parameter tuning via sliders
- Real-time statistics graphs
- Event console output
- Configuration profiles
- Population management
- Complete 3D visualization

Controls:
- Mouse: Drag to rotate camera, scroll to zoom
- WASD: Pan camera
- Space: Pause/unpause
- R: Reset camera
- C: Toggle Circuit8 visibility
- I: Inspect selected creature (click to select)
- 1/2/3: Load profile (default/fast_evolution/drug_heavy)
- S: Save current config as profile
- K: Kill half population (cull weakest)
- F5/F9: Quick save/load
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.morphic.circuit8 import Circuit8
from creatures.collective_creature import CollectiveCreature, create_collective_creatures
from core.evolution.genotype import Genotype
from core.resources.resource_manager import ResourceManager
from core.physics.physics_world import PhysicsWorld
from core.collective import CollectiveMemory
from core.config.config_manager import ConfigManager
from core.config.parameters import ParameterCategory
from core.stats.statistics_tracker import StatisticsTracker, PopulationStats
from core.logging.event_logger import EventLogger
from core.population_manager import PopulationManager
from core.morphology.mesh_generator import ProceduralMeshGenerator
from visualization.gl_primitives import setup_lighting, draw_sphere, draw_cylinder
from core.resources.resource import ResourceType, create_food, create_drug_mushroom

# UI widgets
from visualization.ui.config_panel import ConfigPanel
from visualization.ui.graph_widget import MultiGraphPanel
from visualization.ui.console_widget import ConsoleWidget


class ResearchPlatform:
    """Complete research platform with all Phase 10 systems."""
    
    def __init__(self, width=1600, height=900):
        """Initialize research platform."""
        # Pygame setup
        pygame.init()
        self.width = width
        self.height = height
        
        # Create window with OpenGL
        pygame.display.set_mode(
            (width, height),
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption("CritterGOD Research Platform - Phase 10")
        
        # Fonts
        self.font = pygame.font.Font(None, 18)
        self.small_font = pygame.font.Font(None, 14)
        self.title_font = pygame.font.Font(None, 24)
        
        # Core systems
        self.config = ConfigManager()
        self.config.load_profile("default")
        
        self.stats = StatisticsTracker()
        self.logger = EventLogger()
        self.pop_manager = PopulationManager()
        self.max_population = self.config.get_int("creature_kill_half_at")
        
        # Simulation state
        self.paused = False
        self.timestep = 0
        self.clock = pygame.time.Clock()
        self.target_fps = 60
        
        # World
        world_w = int(self.config.get("world_size_x"))
        world_h = int(self.config.get("world_size_y"))
        
        self.circuit8 = Circuit8(width=64, height=48)
        self.physics_world = PhysicsWorld(
            world_bounds=(-world_w/2, -world_h/2, world_w/2, world_h/2),
            gravity=(0, 0, -9.8)
        )
        self.collective_memory = CollectiveMemory()
        self.resource_manager = ResourceManager(
            world_width=world_w,
            world_height=world_h
        )
        # Spawn initial resources (using defaults from ResourceManager)
        self.resource_manager.spawn_initial_resources()
        
        # Creatures
        self.creatures = []
        self._spawn_initial_creatures()
        
        # Mesh generator for creature bodies
        self.mesh_generator = ProceduralMeshGenerator()
        
        # Camera state for 3D  
        self.camera_distance = 400.0
        self.camera_rotation = 45.0
        self.camera_elevation = 30.0
        self.show_circuit8 = True
        
        # UI panels
        self._setup_ui()
        
        # Selected creature
        self.selected_creature = None
        
        # Auto-save interval
        self.autosave_interval = 1000  # Every 1000 timesteps
        
        # Collision tracking
        self.collision_count = 0
        self.physics_world.register_collision_callback(self._on_collision)
        
        # Simulation control
        self.time_speed = 1.0  # Time speed multiplier
        
        # Render modes and toggles
        self.render_mode = 8  # 8 = all modes
        self.show_thoughts = True
        self.psychedelic_patterns_enabled = False
        self.audio_enabled = False
        self.show_help = False
        
        # Setup OpenGL
        glClearColor(0.05, 0.05, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        setup_lighting()
        
        print("Research Platform initialized!")
        print(f"Config: {self.config.profile_name}")
        print(f"Creatures: {len(self.creatures)}")
        print(f"World size: {world_w}x{world_h}")
    
    def _setup_ui(self):
        """Setup UI panels."""
        panel_width = 280
        
        # Config panel (left side)
        self.config_panel = ConfigPanel(
            x=10,
            y=10,
            width=panel_width,
            height=self.height - 20,
            config=self.config,
            on_change=self._on_config_change
        )
        
        # Stats graphs (top right)
        self.graph_panel = MultiGraphPanel(
            x=self.width - panel_width - 10,
            y=10,
            width=panel_width,
            height=400
        )
        
        # Population graph
        pop_graph = self.graph_panel.add_graph("Population", 120)
        pop_graph.add_series("Count", (100, 200, 255))
        pop_graph.add_series("Avg Energy", (255, 200, 100))
        
        # Neural activity graph
        neural_graph = self.graph_panel.add_graph("Neural Activity", 120)
        neural_graph.add_series("Avg Neurons", (150, 255, 150))
        neural_graph.add_series("Avg Synapses", (255, 150, 255))
        
        # FPS graph
        fps_graph = self.graph_panel.add_graph("Performance", 120)
        fps_graph.add_series("FPS", (255, 255, 100))
        
        # Console (bottom)
        self.console = ConsoleWidget(
            x=panel_width + 20,
            y=self.height - 210,
            width=self.width - panel_width * 2 - 40,
            height=200
        )
        
        # Add welcome messages
        self.console.add_line("=== CritterGOD Research Platform - Phase 10 ===")
        self.console.add_line("Configure parameters with sliders (left panel)")
        self.console.add_line("Watch real-time statistics (top right)")
        self.console.add_line("Press Space to pause, R to reset camera")
        self.console.add_line("Press 1/2/3 to load profiles, S to save")
        self.console.add_line("Ready!")
    
    def _spawn_initial_creatures(self):
        """Spawn initial creature population."""
        world_w = int(self.config.get("world_size_x"))
        world_h = int(self.config.get("world_size_y"))
        
        # Use create_collective_creatures helper (start with fewer creatures)
        self.creatures = create_collective_creatures(
            n_creatures=3,  # Start with fewer to test stability
            physics_world=self.physics_world,
            circuit8=self.circuit8,
            collective_memory=self.collective_memory,
            world_bounds=(-world_w/2, -world_h/2, world_w/2, world_h/2)
        )
        
        # Log births
        for creature in self.creatures:
            self.logger.log_birth(creature, self.timestep)
    
    def _on_config_change(self, parameter):
        """Handle configuration parameter change."""
        self.console.add_line(f"Config: {parameter.name} = {parameter.value:.2f}")
        
        # Apply changes to existing systems
        if parameter.category == "resources":
            if parameter.name == "food_density":
                self.resource_manager.food_density = parameter.value
            elif parameter.name == "drug_density":
                self.resource_manager.drug_density = parameter.value
        elif parameter.category == ParameterCategory.CREATURE:
            if parameter.name == "creature_kill_half_at":
                self.max_population = int(parameter.value)
    
    def _on_collision(self, collision):
        """Handle collision events."""
        self.collision_count += 1
        
        # Let creatures handle their own collisions
        from creatures.collective_creature import CollectiveCreature
        if isinstance(collision.body_a.user_data, CollectiveCreature):
            collision.body_a.user_data.handle_collision(collision)
        if isinstance(collision.body_b.user_data, CollectiveCreature):
            collision.body_b.user_data.handle_collision(collision)
    
    def update(self, dt=1.0):
        """Update simulation."""
        if self.paused:
            return
        
        # Apply time speed multiplier
        dt *= self.time_speed
        
        self.timestep += 1
        
        # Update physics world (with error handling)
        try:
            self.physics_world.step(dt * 0.016)  # Convert to seconds (assuming 60fps)
        except Exception as e:
            print(f"Physics update error: {e}")
        
        # Update resources
        try:
            self.resource_manager.update()
        except Exception as e:
            print(f"Resource update error: {e}")
        
        # Update creatures (with error handling to prevent lockups)
        alive_creatures = []
        for creature in self.creatures:
            try:
                if creature.update(dt):
                    alive_creatures.append(creature)
                else:
                    # Log death
                    self.logger.log_death(creature, self.timestep, "starvation")
            except Exception as e:
                print(f"Error updating creature: {e}")
                # Keep creature alive but log error
                alive_creatures.append(creature)
        
        self.creatures = alive_creatures
        
        # Handle reproduction
        self._handle_reproduction()
        
        # Population management (kill_half_if_needed handles creature removal and logging)
        if len(self.creatures) > self.max_population:
            killed = self.pop_manager.kill_half_if_needed(
                self.creatures, self.max_population, self.timestep, self.logger
            )
        
        # Spawn new creature if population too low
        if len(self.creatures) < 5:
            self._spawn_initial_creatures()
        
        # Update statistics
        self._update_statistics()
        
        # Auto-save
        if self.timestep % self.autosave_interval == 0:
            self.config.save_profile("autosave")
            self.console.add_line(f"Auto-saved at timestep {self.timestep}")
    
    def _handle_reproduction(self):
        """Handle creature reproduction."""
        proc_interval = int(self.config.get("creature_proc_interval"))
        
        for creature in self.creatures[:]:  # Copy list to avoid modification issues
            if self.timestep % proc_interval == 0 and creature.can_reproduce():
                # Find nearby mate
                mate = self._find_mate(creature)
                if mate:
                    # Reproduce
                    child_genotype = creature.genotype.mutate(
                        mutation_rate=self.config.get("mutation_rate_neurons")
                    )
                    
                    # Create child using helper
                    children = create_collective_creatures(
                        n_creatures=1,
                        physics_world=self.physics_world,
                        circuit8=self.circuit8,
                        collective_memory=self.collective_memory,
                        world_bounds=None  # Will spawn at center
                    )
                    child = children[0]
                    # Override position and energy
                    child.x = creature.x + np.random.uniform(-10, 10)
                    child.y = creature.y + np.random.uniform(-10, 10)
                    child.energy.energy = creature.energy.energy * 0.5
                    child.genotype = child_genotype
                    child.generation = max(creature.generation, mate.generation) + 1
                    child.adam_distance = (creature.adam_distance + mate.adam_distance) / 2 + 1
                    child.birth_timestep = self.timestep
                    
                    self.creatures.append(child)
                    
                    # Pay energy cost
                    creature.energy.energy *= 0.5
                    mate.energy.energy *= 0.5
                    
                    # Log
                    self.logger.log_reproduction(creature, mate, child, self.timestep)
    
    def _find_mate(self, creature):
        """Find nearby mate for creature."""
        for other in self.creatures:
            if other is creature:
                continue
            
            dist = np.sqrt((creature.x - other.x)**2 + (creature.y - other.y)**2)
            if dist < 50 and other.can_reproduce():
                return other
        
        return None
    
    def _update_statistics(self):
        """Update statistics tracker."""
        # Calculate population stats
        pop_stats = PopulationStats.calculate_stats(self.creatures)
        
        # Add to tracker
        self.stats.record("population", pop_stats['count'], self.timestep)
        self.stats.record("avg_energy", pop_stats['avg_energy'], self.timestep)
        self.stats.record("avg_neurons", pop_stats['avg_neurons'], self.timestep)
        self.stats.record("avg_synapses", pop_stats['avg_synapses'], self.timestep)
        self.stats.record("avg_generation", pop_stats['avg_generation'], self.timestep)
        self.stats.record("fps", self.clock.get_fps(), self.timestep)
        
        # Update graphs
        graphs = self.graph_panel.graphs
        
        # Population graph
        graphs[0].add_data_point(0, pop_stats['count'])
        graphs[0].add_data_point(1, pop_stats['avg_energy'] / 10000)  # Scale down
        
        # Neural graph
        graphs[1].add_data_point(0, pop_stats['avg_neurons'])
        graphs[1].add_data_point(1, pop_stats['avg_synapses'] / 10)  # Scale down
        
        # FPS graph
        graphs[2].add_data_point(0, self.clock.get_fps())
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            # UI events
            if self.config_panel.handle_event(event):
                continue
            if self.console.handle_event(event):
                continue
            
            # Keyboard
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                elif event.key == K_SPACE:
                    self.paused = not self.paused
                    self.console.add_line("Paused" if self.paused else "Unpaused")
                elif event.key == K_r:
                    self.camera_distance = 400.0
                    self.camera_rotation = 45.0
                    self.camera_elevation = 30.0
                    self.console.add_line("Camera reset")
                elif event.key == K_c:
                    self.show_circuit8 = not self.show_circuit8
                    self.console.add_line(f"Circuit8: {'visible' if self.show_circuit8 else 'hidden'}")
                elif event.key == K_k:
                    # Kill half
                    killed = self.pop_manager.kill_half_if_needed(
                        self.creatures, len(self.creatures), self.timestep, self.logger
                    )
                    self.console.add_line(f"Killed {len(killed)} creatures (weakest)")
                elif event.key == K_s:
                    # Save profile
                    self.config.save_profile("user_config")
                    self.console.add_line("Saved profile: user_config")
                elif event.key == K_1:
                    self.config.load_profile("default")
                    self.config_panel._build_sliders()
                    self.console.add_line("Loaded profile: default")
                elif event.key == K_F5:
                    # Quick save
                    self.config.save_profile("quicksave")
                    self.console.add_line("Quick saved!")
                elif event.key == K_F9:
                    # Quick load
                    self.config.load_profile("quicksave")
                    self.config_panel._build_sliders()
                    self.console.add_line("Quick loaded!")
                elif event.key == K_f:
                    # Spawn food
                    self._spawn_food()
                elif event.key == K_d:
                    # Spawn drug mushroom
                    self._spawn_drug()
                elif event.key == K_i:
                    # Apply random impulses to creatures
                    self._apply_random_impulses()
                elif event.key == K_h:
                    # Toggle help overlay
                    self.show_help = not self.show_help
                    self.console.add_line("Help: " + ("visible" if self.show_help else "hidden"))
                elif event.key == K_t:
                    # Toggle thought bubbles
                    self.show_thoughts = not self.show_thoughts
                    self.console.add_line("Thoughts: " + ("visible" if self.show_thoughts else "hidden"))
                elif event.key == K_p:
                    # Toggle psychedelic patterns
                    self.psychedelic_patterns_enabled = not self.psychedelic_patterns_enabled
                    for creature in self.creatures:
                        creature.pattern_generation_enabled = self.psychedelic_patterns_enabled
                    self.console.add_line("Psychedelic patterns: " + ("ON" if self.psychedelic_patterns_enabled else "OFF"))
                elif event.key == K_a:
                    # Toggle audio
                    self.audio_enabled = not self.audio_enabled
                    self.console.add_line("Audio: " + ("ON" if self.audio_enabled else "OFF"))
                elif event.key == K_LEFTBRACKET:
                    # Slow down time
                    self.time_speed = max(0.1, self.time_speed / 1.5)
                    self.console.add_line(f"Time speed: {self.time_speed:.2f}x")
                elif event.key == K_RIGHTBRACKET:
                    # Speed up time
                    self.time_speed = min(10.0, self.time_speed * 1.5)
                    self.console.add_line(f"Time speed: {self.time_speed:.2f}x")
                elif K_1 <= event.key <= K_8:
                    # Switch render mode
                    self.render_mode = event.key - K_0
                    mode_names = ["", "Creatures", "Neural", "Circuit8", "Resources", 
                                  "Physics", "Signals", "Learning", "All"]
                    self.console.add_line(f"Render mode: {mode_names[self.render_mode]}")
                elif K_9 <= event.key <= K_0:
                    # Give drugs to all creatures (9=inhibitory_antagonist, 0=potentiator)
                    molecule_type = (event.key - K_9) if event.key != K_0 else 4
                    if molecule_type < 5:
                        self._give_drug_to_all(molecule_type)
            
            # Camera controls
            elif event.type == MOUSEWHEEL:
                self.camera_distance -= event.y * 20
                self.camera_distance = max(100, min(2000, self.camera_distance))
        
        # WASD camera rotation
        keys = pygame.key.get_pressed()
        if keys[K_a]:
            self.camera_rotation -= 2
        if keys[K_d]:
            self.camera_rotation += 2
        if keys[K_w]:
            self.camera_elevation = min(80, self.camera_elevation + 2)
        if keys[K_s]:
            self.camera_elevation = max(-80, self.camera_elevation - 2)
        
        return True
    
    def render(self):
        """Render complete scene."""
        # 3D scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Setup camera
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 2000.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Apply camera transform
        glTranslatef(0, 0, -self.camera_distance)
        glRotatef(self.camera_elevation, 1, 0, 0)
        glRotatef(self.camera_rotation, 0, 1, 0)
        
        # Render Circuit8 ground plane (if enabled)
        if self.show_circuit8 and self.render_mode in [3, 8]:
            self._render_circuit8_ground()
        else:
            # Regular ground plane
            self._render_ground()
        
        # Render collective signals (if enabled)
        if self.render_mode in [6, 8]:
            self._render_collective_signals()
        
        # Render resources (food & drugs)
        if self.render_mode in [4, 8]:
            self._render_resources()
        
        # Render creatures with procedural meshes
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        for creature in self.creatures:
            # Generate mesh if needed
            if not hasattr(creature, 'mesh') or creature.mesh is None:
                creature.mesh = self.mesh_generator.generate_creature_mesh(creature.body)
            
            glPushMatrix()
            glTranslatef(creature.x, creature.y, 10)  # z=10 for height above ground
            
            # Drug-responsive scale
            scale = creature.get_render_scale() if hasattr(creature, 'get_render_scale') else 1.0
            glScalef(scale, scale, scale)
            
            # Base body scale
            body_scale = 5.0
            glScalef(body_scale, body_scale, body_scale)
            
            # Render mesh
            creature.mesh.render()
            
            glPopMatrix()
        
        # Render velocity vectors
        if self.render_mode in [5, 8]:  # Physics debug mode
            self._render_velocity_vectors()
        
        # Collect thoughts for rendering (if enabled)
        if self.show_thoughts:
            self._collect_thoughts()
        
        # Switch to 2D for UI
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, self.width, self.height, 0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Create pygame surface for UI
        ui_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        ui_surface.fill((0, 0, 0, 0))
        
        # Render UI panels
        self.config_panel.render(ui_surface, self.font, self.small_font)
        self.graph_panel.render(ui_surface, self.font, self.small_font)
        self.console.render(ui_surface, self.small_font)
        
        # Render stats overlay (center top)
        self._render_stats_overlay(ui_surface)
        
        # Render thought bubbles (if enabled)
        if self.show_thoughts and hasattr(self, '_thoughts_to_render'):
            self._render_thought_bubbles(ui_surface)
        
        # Render help overlay (if enabled)
        if self.show_help:
            self._render_help_overlay(ui_surface)
        
        # Blit to OpenGL
        self._blit_pygame_to_opengl(ui_surface)
        
        glEnable(GL_DEPTH_TEST)
        
        pygame.display.flip()
    
    def _render_ground(self):
        """Render ground plane with grid."""
        world_w = int(self.config.get("world_size_x"))
        world_h = int(self.config.get("world_size_y"))
        hw = world_w / 2
        hh = world_h / 2
        
        glDisable(GL_LIGHTING)
        glColor4f(0.1, 0.15, 0.2, 1.0)
        glBegin(GL_QUADS)
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
    
    def _render_resources(self):
        """Render food and drug mushrooms."""
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
                # Drug mushroom stem
                glColor4f(0.9, 0.9, 0.8, 0.9)
                glPushMatrix()
                glRotatef(-90, 1, 0, 0)
                draw_cylinder(resource.radius * 0.3, resource.radius * 0.3, 
                            resource.radius * 1.5, slices=8)
                glPopMatrix()
                
                # Drug mushroom cap (colored by molecule type)
                colors = [
                    (0.5, 0.0, 1.0),  # Purple - Inhibitory Antagonist
                    (1.0, 0.0, 0.5),  # Pink - Inhibitory Agonist
                    (1.0, 0.5, 0.0),  # Orange - Excitatory Antagonist
                    (0.0, 0.5, 1.0),  # Cyan - Excitatory Agonist
                    (1.0, 1.0, 0.0),  # Yellow - Potentiator
                ]
                # Cast molecule_type to int (it may be stored as float)
                mol_type = int(resource.molecule_type) if resource.molecule_type is not None else None
                color = colors[mol_type] if mol_type is not None and 0 <= mol_type < 5 else (1.0, 1.0, 1.0)
                glColor4f(*color, 0.9)
                glPushMatrix()
                glTranslatef(0, 0, resource.radius * 1.5)
                draw_sphere(resource.radius, slices=8, stacks=8)
                glPopMatrix()
            
            glPopMatrix()
    
    def _spawn_food(self):
        """Spawn food at random location."""
        world_w = int(self.config.get("world_size_x"))
        world_h = int(self.config.get("world_size_y"))
        x = np.random.uniform(-world_w/2, world_w/2)
        y = np.random.uniform(-world_h/2, world_h/2)
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
        
        self.console.add_line(f"🍎 Spawned food at ({x:.0f}, {y:.0f})")
    
    def _spawn_drug(self):
        """Spawn drug mushroom at random location."""
        world_w = int(self.config.get("world_size_x"))
        world_h = int(self.config.get("world_size_y"))
        x = np.random.uniform(-world_w/2, world_w/2)
        y = np.random.uniform(-world_h/2, world_h/2)
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
        
        drug_names = ["Inh. Antag.", "Inh. Agon.", "Exc. Antag.", "Exc. Agon.", "Potent."]
        self.console.add_line(f"🍄 Spawned {drug_names[molecule_type]} at ({x:.0f}, {y:.0f})")
    
    def _apply_random_impulses(self):
        """Apply random impulses to all creatures."""
        for creature in self.creatures:
            if hasattr(creature, 'apply_impulse'):
                impulse = np.random.uniform(-500, 500, size=3).astype(np.float32)
                impulse[2] = abs(impulse[2])  # Always push up
                creature.apply_impulse(impulse)
        self.console.add_line("💥 Applied random impulses to all creatures!")
    
    def _give_drug_to_all(self, molecule_type: int):
        """Give drug to all creatures."""
        for creature in self.creatures:
            if hasattr(creature, 'drugs'):
                creature.drugs.tripping[molecule_type] += 5.0
        drug_names = ["Inh. Antag.", "Inh. Agon.", "Exc. Antag.", "Exc. Agon.", "Potent."]
        self.console.add_line(f"💊 Gave all creatures {drug_names[molecule_type]}")
    
    def _render_circuit8_ground(self):
        """Render Circuit8 as glowing ground plane."""
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        glPushMatrix()
        glTranslatef(0, 0, 0)
        
        # Sample center pixel for overall color
        r, g, b = self.circuit8.read_pixel(32, 24)
        r_f = min(1.0, float(r) / 255.0 * 3.0 + 0.3)
        g_f = min(1.0, float(g) / 255.0 * 3.0 + 0.3)
        b_f = min(1.0, float(b) / 255.0 * 3.0 + 0.3)
        
        # Draw colored ground
        size = 500.0
        glColor3f(r_f, g_f, b_f)
        glBegin(GL_QUADS)
        glVertex3f(-size, -size, 0)
        glVertex3f(size, -size, 0)
        glVertex3f(size, size, 0)
        glVertex3f(-size, size, 0)
        glEnd()
        
        # Bright border
        glColor3f(1.0, 1.0, 0.0)
        glLineWidth(5.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-size, -size, 0.1)
        glVertex3f(size, -size, 0.1)
        glVertex3f(size, size, 0.1)
        glVertex3f(-size, size, 0.1)
        glEnd()
        glLineWidth(1.0)
        
        glPopMatrix()
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def _render_collective_signals(self):
        """Render collective memory markers."""
        glDisable(GL_LIGHTING)
        
        # Resource markers
        try:
            for record in self.collective_memory.resource_locations:
                if not isinstance(record, dict) or 'location' not in record:
                    continue
                
                x, y = record['location']
                resource_type = record.get('resource_type', 'unknown')
                
                color = (0.0, 1.0, 0.0, 0.5) if resource_type == 'food' else (1.0, 0.0, 1.0, 0.5)
                
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
        except:
            pass
        
        # Danger zones
        try:
            for record in self.collective_memory.danger_zones:
                if not isinstance(record, dict) or 'location' not in record:
                    continue
                
                x, y = record['location']
                
                glPushMatrix()
                glTranslatef(x, y, 1.0)
                glColor4f(1.0, 0.0, 0.0, 0.3)
                glBegin(GL_LINE_LOOP)
                for i in range(16):
                    angle = i * 2 * np.pi / 16
                    glVertex3f(10 * np.cos(angle), 10 * np.sin(angle), 0)
                glEnd()
                glPopMatrix()
        except:
            pass
        
        glEnable(GL_LIGHTING)
    
    def _collect_thoughts(self):
        """Collect creature thoughts for rendering."""
        self._thoughts_to_render = []
        for creature in self.creatures:
            if hasattr(creature, 'get_current_thought'):
                thought = creature.get_current_thought()
                if thought and len(thought.strip()) > 0:
                    self._thoughts_to_render.append({
                        'x': creature.x,
                        'y': creature.y,
                        'z': 10.0,
                        'text': thought
                    })
    
    def _render_thought_bubbles(self, surface):
        """Render thought bubbles in 2D."""
        # Simple approach: render text near creature positions
        for i, creature in enumerate(self.creatures[:10]):  # Limit to 10
            if hasattr(creature, 'get_current_thought'):
                thought = creature.get_current_thought()
                if thought and len(thought.strip()) > 0:
                    # Truncate long thoughts
                    text = thought[:30]
                    if len(thought) > 30:
                        text += "..."
                    
                    # Render at creature position (approximate)
                    text_surf = self.small_font.render(text, True, (255, 255, 100))
                    x = int(self.width / 2 + creature.x)
                    y = int(self.height / 2 - creature.y - 20)
                    
                    # Background
                    bg = pygame.Surface((text_surf.get_width() + 4, text_surf.get_height() + 2))
                    bg.fill((0, 0, 0))
                    bg.set_alpha(150)
                    surface.blit(bg, (x - 2, y - 1))
                    surface.blit(text_surf, (x, y))
    
    def _render_help_overlay(self, surface):
        """Render help overlay."""
        help_lines = [
            "=== CRITTERGOD RESEARCH PLATFORM ===",
            "",
            "CAMERA:",
            "  Mouse drag: Rotate | Scroll: Zoom | WASD: Pan",
            "  R: Reset camera",
            "",
            "SIMULATION:",
            "  Space: Pause/unpause",
            "  [/]: Slow/speed time (0.1x-10x)",
            "  K: Kill half population",
            "",
            "INTERACTIVE:",
            "  F: Spawn food | D: Spawn drug",
            "  I: Apply impulses (physics test)",
            "  9-0: Give drugs to all (by type)",
            "",
            "VISUALIZATION:",
            "  1-8: Render modes (1=creatures, 8=all)",
            "  C: Toggle Circuit8 | T: Toggle thoughts",
            "  P: Toggle psychedelic | A: Toggle audio",
            "  H: Toggle this help",
            "",
            "CONFIG:",
            "  1: Load default | S: Save config",
            "  F5: Quick save | F9: Quick load",
            "",
            "Press H to close",
        ]
        
        # Semi-transparent background
        help_bg = pygame.Surface((500, len(help_lines) * 22 + 20))
        help_bg.fill((0, 0, 50))
        help_bg.set_alpha(220)
        
        x = self.width // 2 - 250
        y = self.height // 2 - (len(help_lines) * 11 + 10)
        surface.blit(help_bg, (x, y))
        
        # Render help text
        for i, line in enumerate(help_lines):
            if "===" in line:
                text = self.font.render(line, True, (255, 255, 0))
            elif line.strip() and not line.startswith("  "):
                text = self.font.render(line, True, (100, 200, 255))
            else:
                text = self.small_font.render(line, True, (200, 200, 200))
            surface.blit(text, (x + 10, y + 10 + i * 22))
    
    def _render_velocity_vectors(self):
        """Render velocity vectors for moving creatures."""
        for creature in self.creatures:
            if hasattr(creature, 'rigid_body') and creature.rigid_body:
                vel = creature.get_velocity()
                vel_mag = np.linalg.norm(vel)
                if vel_mag > 0.1:
                    glDisable(GL_LIGHTING)
                    glBegin(GL_LINES)
                    glColor4f(1.0, 1.0, 0.0, 0.7)
                    z_pos = creature.z if hasattr(creature, 'z') else 10
                    glVertex3f(creature.x, creature.y, z_pos)
                    glVertex3f(creature.x + vel[0] * 2, creature.y + vel[1] * 2, z_pos + vel[2] * 2)
                    glEnd()
                    glEnable(GL_LIGHTING)
    
    def _render_stats_overlay(self, surface):
        """Render quick stats overlay."""
        stats_text = [
            f"Timestep: {self.timestep}",
            f"Creatures: {len(self.creatures)}",
            f"FPS: {self.clock.get_fps():.1f}",
            f"{'PAUSED' if self.paused else ''}",
        ]
        
        y = 10
        for text in stats_text:
            if not text:
                continue
            text_surface = self.font.render(text, True, (255, 255, 255))
            bg_surface = pygame.Surface(
                (text_surface.get_width() + 10, text_surface.get_height() + 4)
            )
            bg_surface.fill((0, 0, 0))
            bg_surface.set_alpha(180)
            
            x = self.width // 2 - text_surface.get_width() // 2
            surface.blit(bg_surface, (x - 5, y - 2))
            surface.blit(text_surface, (x, y))
            y += 20
    
    def _blit_pygame_to_opengl(self, surface):
        """Blit pygame surface to OpenGL."""
        texture_data = pygame.image.tostring(surface, "RGBA", True)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glRasterPos2f(0, 0)
        glDrawPixels(
            surface.get_width(),
            surface.get_height(),
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            texture_data
        )
        
        glDisable(GL_BLEND)
    
    def run(self):
        """Main loop."""
        running = True
        
        while running:
            # Events
            running = self.handle_events()
            
            # Update
            self.update()
            
            # Render
            self.render()
            
            # Sync recent log lines to console
            recent_logs = self.logger.get_recent_lines(n=5)
            if recent_logs:
                self.console.add_lines(recent_logs)
            
            # Frame rate
            self.clock.tick(self.target_fps)
        
        pygame.quit()


if __name__ == "__main__":
    platform = ResearchPlatform(width=1600, height=900)
    platform.run()
