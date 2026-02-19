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
from PIL import Image, ImageDraw, ImageFont

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
from core.evolution.species import SpeciesTracker, LineageTracker, GeneticDistance
from visualization.force_directed_layout import ForceDirectedLayout
from generators.audio import NeuralAudioSynthesizer


class ResearchPlatform:
    """Complete research platform with all Phase 10 systems."""
    
    def __init__(self, width=1600, height=900):
        """Initialize research platform."""
        # Pygame setup
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        self.width = width
        self.height = height
        
        # Skip pygame.font entirely - it's broken in Python 3.14
        # We'll use PIL for text rendering instead
        print("🔧 Using PIL for text rendering (pygame.font broken in Python 3.14)")
        
        # Force window to top-left corner so it's visible
        os.environ['SDL_VIDEO_WINDOW_POS'] = "50,50"
        
        # Create window with OpenGL
        pygame.display.set_mode(
            (width, height),
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption("🔴 CritterGOD Research Platform - LOOK HERE! 🔴")
        
        # Send macOS notification so user knows window opened
        import subprocess
        subprocess.run([
            'osascript', '-e',
            'display notification "Research Platform window is OPEN! Use Cmd+Tab to switch to it." with title "CritterGOD Started"'
        ], check=False)
        
        # Import UI widgets AFTER display init (avoids pygame.font circular import)
        from visualization.ui.config_panel import ConfigPanel
        from visualization.ui.graph_widget import MultiGraphPanel
        from visualization.ui.console_widget import ConsoleWidget
        self.ConfigPanel = ConfigPanel
        self.MultiGraphPanel = MultiGraphPanel
        self.ConsoleWidget = ConsoleWidget
        
        # Fonts - PIL-based text rendering (bypasses pygame.font)
        class PILFont:
            """pygame.font replacement using PIL/Pillow for text rendering."""
            def __init__(self, size=18):
                self._size = size
                try:
                    # Try to load a monospace font
                    self._pil_font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", size)
                except:
                    # Fall back to default font
                    self._pil_font = ImageFont.load_default()
            
            def render(self, text, antialias, color):
                """Render text using PIL and convert to pygame Surface."""
                if not text:
                    return pygame.Surface((1, 1), pygame.SRCALPHA)
                
                # Get text size using PIL
                dummy_img = Image.new('RGBA', (1, 1))
                draw = ImageDraw.Draw(dummy_img)
                bbox = draw.textbbox((0, 0), text, font=self._pil_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Create PIL image
                img = Image.new('RGBA', (text_width + 4, text_height + 4), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.text((2, 2), text, font=self._pil_font, fill=color)
                
                # Convert to pygame surface
                mode = img.mode
                size = img.size
                data = img.tobytes()
                surf = pygame.image.fromstring(data, size, mode)
                
                return surf
            
            def get_height(self):
                return self._size
            
            def get_linesize(self):
                return self._size
            
            def size(self, text):
                """Get size of rendered text."""
                if not text:
                    return (0, self._size)
                dummy_img = Image.new('RGBA', (1, 1))
                draw = ImageDraw.Draw(dummy_img)
                bbox = draw.textbbox((0, 0), text, font=self._pil_font)
                return (bbox[2] - bbox[0], bbox[3] - bbox[1])
        
        print("🔧 Creating PIL fonts...")
        try:
            self.font = PILFont(12)
            self.small_font = PILFont(10)
            self.title_font = PILFont(14)
            print("✅ PIL fonts created successfully! Text will render properly.")
        except Exception as e:
            print(f"❌ PIL font creation failed: {e}")
            # Ultra fallback
            class DummyFont:
                def __init__(self, size=18):
                    self._size = size
                def render(self, text, aa, color):
                    width = max(len(text) * self._size // 2, 50)
                    surf = pygame.Surface((width, self._size), pygame.SRCALPHA)
                    r, g, b = color[0], color[1], color[2]
                    surf.fill((r, g, b, 100))
                    pygame.draw.rect(surf, color, (0, 0, width, self._size), 1)
                    return surf
                def get_height(self):
                    return self._size
                def get_linesize(self):
                    return self._size
                def size(self, text):
                    return (len(text) * self._size // 2, self._size)
            self.font = DummyFont(12)
            self.small_font = DummyFont(10)
            self.title_font = DummyFont(14)
            print("⚠️  Using colored boxes instead of text")
        
        # Core systems
        self.config = ConfigManager()
        self.config.load_profile("default")
        
        self.stats = StatisticsTracker()
        self.logger = EventLogger()
        self.pop_manager = PopulationManager()
        self.max_population = self.config.get_int("creature_kill_half_at")
        
        # Simulation state
        self.paused = False
        self.extinct = False
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
        
        # Camera state: fixed-ground RTS-style camera
        # Camera looks at (camera_x, camera_y) on the ground from above
        self.camera_x = 0.0       # Look-at X on ground
        self.camera_y = 0.0       # Look-at Y on ground
        self.camera_zoom = 500.0  # Height/distance above ground (zoom level)
        self.camera_angle = 55.0  # Fixed viewing angle (degrees from horizontal, 90=top-down)
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
        
        # Species tracking
        self.species_tracker = SpeciesTracker(compatibility_threshold=0.5)
        self.lineage_tracker = LineageTracker()
        
        # Force-directed layout
        self.force_layout = ForceDirectedLayout(
            repulsion_strength=100.0,
            attraction_strength=0.5,
            similarity_threshold=0.7
        )
        self.use_force_layout = False
        
        # Simulation control
        self.time_speed = 1.0  # Time speed multiplier
        
        # Render modes and toggles
        self.render_mode = 8  # 8 = all modes
        self.show_thoughts = True
        self.psychedelic_patterns_enabled = False
        self.audio_enabled = True  # Audio ON by default
        
        # Audio synthesis
        self.audio_synth = NeuralAudioSynthesizer(
            sample_rate=44100,
            buffer_size=2048,
            mode='mixed',
            amplitude_scale=0.3,
        )
        self.audio_mode_names = ['potential', 'firing', 'mixed', 'flamoot']
        self.audio_mode_index = 2
        self.sound_queue = []
        self.show_help = False
        
        # Mouse drag state for camera
        self.mouse_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Auto-insert creatures (critterding heritage: insertcritterevery)
        self.insert_creature_every = 2000  # Insert new random creature every N timesteps
        
        # Inspector button regions (for click detection)
        self.inspector_buttons = []  # List of (rect, action_name) tuples
        
        # Setup OpenGL
        glClearColor(0.4, 0.6, 0.9, 1.0)  # Sky blue background
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
        self.config_panel = self.ConfigPanel(
            x=10,
            y=10,
            width=panel_width,
            height=self.height - 20,
            config=self.config,
            on_change=self._on_config_change
        )
        
        # Stats graphs (top right)
        self.graph_panel = self.MultiGraphPanel(
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
        self.console = self.ConsoleWidget(
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
        self.console.add_line("Press 1-8 for render modes, H for help")
        self.console.add_line("Ready!")
    
    def _spawn_initial_creatures(self, count=15):
        """Spawn creatures and ADD them to the population (not replace)."""
        world_w = int(self.config.get("world_size_x"))
        world_h = int(self.config.get("world_size_y"))
        
        # Create new creatures and APPEND to existing list
        new_creatures = create_collective_creatures(
            n_creatures=count,
            physics_world=self.physics_world,
            circuit8=self.circuit8,
            collective_memory=self.collective_memory,
            world_bounds=(-world_w/2, -world_h/2, world_w/2, world_h/2)
        )
        self.creatures.extend(new_creatures)
        
        # Log births
        for creature in new_creatures:
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
        
        # Physics timestep - use small fixed timestep for stability
        physics_dt = 0.016 * self.time_speed  # 60fps timestep
        
        # Update physics world (with error handling)
        try:
            self.physics_world.step(physics_dt)
        except Exception as e:
            print(f"Physics update error: {e}")
        
        # Update resources
        try:
            self.resource_manager.update()
        except Exception as e:
            print(f"Resource update error: {e}")
        
        # Update Circuit8: decay fades old activity, voted movement applies collective will
        self.circuit8.decay(rate=0.97)  # 3% fade per frame — keeps it dynamic
        self.circuit8.apply_voted_movement()
        
        # Update creatures (with error handling to prevent lockups)
        # Use same timestep as physics for consistent force application
        alive_creatures = []
        new_offspring = []
        for creature in self.creatures:
            try:
                result = creature.update(physics_dt, resource_manager=self.resource_manager)
                if result is False:
                    # Dead
                    self.logger.log_death(creature, self.timestep, "starvation")
                elif result is True:
                    # Alive, no offspring
                    alive_creatures.append(creature)
                else:
                    # result is an offspring creature
                    alive_creatures.append(creature)
                    new_offspring.append(result)
                    self.logger.log_reproduction(creature, None, result, self.timestep)
            except Exception as e:
                print(f"Error updating creature: {e}")
                # Keep creature alive but log error
                alive_creatures.append(creature)
        
        # Add offspring to population
        alive_creatures.extend(new_offspring)
        
        self.creatures = alive_creatures
        
        # Population management (kill_half_if_needed handles creature removal and logging)
        if len(self.creatures) > self.max_population:
            killed = self.pop_manager.kill_half_if_needed(
                self.creatures, self.max_population, self.timestep, self.logger
            )
        
        # World edge "pit" deaths (critterding heritage)
        # Creatures that wander off the edge die — creates selection pressure
        world_w = int(self.config.get("world_size_x"))
        world_h = int(self.config.get("world_size_y"))
        pit_margin = 20.0
        pit_boundary_x = world_w / 2 + pit_margin
        pit_boundary_y = world_h / 2 + pit_margin
        pit_victims = []
        for creature in self.creatures:
            if (abs(creature.x) > pit_boundary_x or 
                abs(creature.y) > pit_boundary_y):
                pit_victims.append(creature)
        for victim in pit_victims:
            self.creatures.remove(victim)
            self.logger.log_death(victim, self.timestep, "fell in the pit")
        
        # Auto-insert new creatures (critterding: insertcritterevery)
        if (self.insert_creature_every > 0 and 
            self.timestep % self.insert_creature_every == 0 and
            len(self.creatures) < self.max_population and
            len(self.creatures) > 0):  # Don't auto-insert if extinct
            self._spawn_initial_creatures(count=1)
            self.console.add_line(f"Auto-inserted new creature (pop: {len(self.creatures)})")
        
        # Check for extinction
        if len(self.creatures) == 0 and not self.extinct:
            self.extinct = True
            self.paused = True
            self.console.add_line("*** EXTINCTION — All creatures have died ***")
            self.console.add_line("Press N to start a new simulation")
        
        # Update species tracking (every 50 timesteps)
        if self.timestep % 50 == 0 and self.creatures:
            self.species_tracker.update_species(self.creatures, self.timestep)
        
        # Update statistics
        self._update_statistics()
        
        # Audio synthesis
        if self.audio_enabled and self.creatures:
            self._update_audio()
        
        # Auto-save
        if self.timestep % self.autosave_interval == 0:
            self.config.save_profile("autosave")
            self.console.add_line(f"Auto-saved at timestep {self.timestep}")
    
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
                    self.camera_x = 0.0
                    self.camera_y = 0.0
                    self.camera_zoom = 500.0
                    self.camera_angle = 55.0
                    self.console.add_line("Camera reset")
                elif event.key == K_n:
                    # Restart simulation with new creatures
                    self._restart_simulation()
                elif event.key == K_c:
                    self.show_circuit8 = not self.show_circuit8
                    self.console.add_line(f"Circuit8: {'visible' if self.show_circuit8 else 'hidden'}")
                elif event.key == K_k:
                    # Force kill half population (weakest by energy)
                    if len(self.creatures) > 2:
                        sorted_by_energy = sorted(self.creatures, key=lambda c: c.energy.energy)
                        num_to_kill = len(self.creatures) // 2
                        to_kill = sorted_by_energy[:num_to_kill]
                        for c in to_kill:
                            self.creatures.remove(c)
                            self.logger.log_death(c, self.timestep, "manual_cull")
                        self.console.add_line(f"Killed {num_to_kill} creatures (weakest)")
                    else:
                        self.console.add_line("Too few creatures to cull")
                elif event.key == K_v:
                    # Save profile
                    self.config.save_profile("user_config")
                    self.console.add_line("Saved profile: user_config")
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
                elif event.key == K_u:
                    # Toggle audio (changed from A to avoid WASD conflict)
                    self.audio_enabled = not self.audio_enabled
                    if self.audio_enabled:
                        self.console.add_line(f"Audio: ON (mode: {self.audio_synth.mode})")
                    else:
                        self.console.add_line("Audio: OFF")
                elif event.key == K_m:
                    # Cycle audio modes
                    self.audio_mode_index = (self.audio_mode_index + 1) % len(self.audio_mode_names)
                    self.audio_synth.mode = self.audio_mode_names[self.audio_mode_index]
                    self.console.add_line(f"Audio mode: {self.audio_synth.mode}")
                elif event.key == K_LEFTBRACKET:
                    # Slow down time
                    self.time_speed = max(0.1, self.time_speed / 1.5)
                    self.console.add_line(f"Time speed: {self.time_speed:.2f}x")
                elif event.key == K_RIGHTBRACKET:
                    # Speed up time
                    self.time_speed = min(10.0, self.time_speed * 1.5)
                    self.console.add_line(f"Time speed: {self.time_speed:.2f}x")
                elif K_1 <= event.key <= K_8:
                    # Switch render mode (1-8)
                    self.render_mode = event.key - K_0
                    mode_names = ["", "Creatures", "Neural", "Circuit8", "Resources", 
                                  "Physics", "Signals", "Learning", "All"]
                    self.console.add_line(f"Render mode: {mode_names[self.render_mode]}")
                    # 8=All is default, shows everything
                elif event.key == K_9:
                    # Give drug to all (or selected) - Excitatory Agonist
                    if self.selected_creature:
                        self._give_drug_to_creature(self.selected_creature, 3)
                    else:
                        self._give_drug_to_all(3)
                elif event.key == K_0:
                    # Give drug to all (or selected) - Potentiator
                    if self.selected_creature:
                        self._give_drug_to_creature(self.selected_creature, 4)
                    else:
                        self._give_drug_to_all(4)
                elif event.key == K_F1:
                    # Give Inhibitory Antagonist to all or selected
                    if self.selected_creature:
                        self._give_drug_to_creature(self.selected_creature, 0)
                    else:
                        self._give_drug_to_all(0)
                elif event.key == K_F2:
                    # Give Inhibitory Agonist to all or selected
                    if self.selected_creature:
                        self._give_drug_to_creature(self.selected_creature, 1)
                    else:
                        self._give_drug_to_all(1)
                elif event.key == K_F3:
                    # Give Excitatory Antagonist to all or selected
                    if self.selected_creature:
                        self._give_drug_to_creature(self.selected_creature, 2)
                    else:
                        self._give_drug_to_all(2)
                elif event.key == K_F4:
                    # Give Excitatory Agonist to all or selected
                    if self.selected_creature:
                        self._give_drug_to_creature(self.selected_creature, 3)
                    else:
                        self._give_drug_to_all(3)
            
            # Mouse drag for camera PAN (ground stays fixed)
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    # Check inspector buttons first
                    btn_clicked = False
                    if self.selected_creature and self.inspector_buttons:
                        for rect, action in self.inspector_buttons:
                            if rect.collidepoint(event.pos):
                                self._inspector_button_action(action)
                                btn_clicked = True
                                break
                    if not btn_clicked:
                        self.mouse_dragging = True
                        self.last_mouse_x, self.last_mouse_y = event.pos
                elif event.button == 3:  # Right click - inspect creature
                    self._try_inspect_creature(event.pos)
                # Scroll wheel (some pygame versions use button 4/5)
                elif event.button == 4:  # Scroll up - zoom in
                    self.camera_zoom = max(50, self.camera_zoom * 0.9)
                elif event.button == 5:  # Scroll down - zoom out
                    self.camera_zoom = min(2000, self.camera_zoom * 1.1)
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
            elif event.type == MOUSEMOTION:
                if self.mouse_dragging:
                    dx = event.pos[0] - self.last_mouse_x
                    dy = event.pos[1] - self.last_mouse_y
                    # Pan speed scales with zoom level (further out = faster pan)
                    pan_speed = self.camera_zoom / 500.0
                    self.camera_x -= dx * pan_speed
                    self.camera_y += dy * pan_speed  # Inverted Y
                    self.last_mouse_x, self.last_mouse_y = event.pos
            
            # Camera zoom (MOUSEWHEEL event for newer pygame)
            elif event.type == MOUSEWHEEL:
                zoom_factor = 1.0 - event.y * 0.08
                self.camera_zoom = max(50, min(2000, self.camera_zoom * zoom_factor))
        
        # Arrow keys for camera pan
        keys = pygame.key.get_pressed()
        pan_speed = self.camera_zoom / 150.0  # Scale with zoom
        if keys[K_LEFT]:
            self.camera_x -= pan_speed
        if keys[K_RIGHT]:
            self.camera_x += pan_speed
        if keys[K_UP]:
            self.camera_y += pan_speed
        if keys[K_DOWN]:
            self.camera_y -= pan_speed
        
        return True
    
    def render(self):
        """Render complete scene."""
        # 3D scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Render sky gradient (before 3D camera setup)
        self._render_sky()
        
        # Setup camera - fixed ground, pan/zoom style
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 1.0, 5000.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Camera position: offset from look-at point based on angle and zoom
        import math
        angle_rad = math.radians(self.camera_angle)
        cam_height = self.camera_zoom * math.sin(angle_rad)
        cam_back = self.camera_zoom * math.cos(angle_rad)
        
        eye_x = self.camera_x
        eye_y = self.camera_y - cam_back  # Behind the look-at point
        eye_z = cam_height
        
        gluLookAt(
            eye_x, eye_y, eye_z,          # Eye position
            self.camera_x, self.camera_y, 0.0,  # Look-at point on ground
            0.0, 0.0, 1.0                 # Up vector (Z is up)
        )
        
        # Always render regular ground plane
        self._render_ground()
        
        # Render collective signals (if enabled)
        if self.render_mode in [6, 8]:
            self._render_collective_signals()
        
        # Render social learning connections (Mode 7 or 8)
        if self.render_mode in [7, 8]:
            self._render_social_learning()
        
        # Render resources (food & drugs) - ALWAYS visible so user sees food/mushrooms
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
        
        # Render neural activity visualization (Mode 2 or 8)
        if self.render_mode in [2, 8]:
            self._render_neural_activity()
        
        # Collect thoughts for rendering (if enabled)
        if self.show_thoughts:
            self._collect_thoughts()
        
        # Switch to 2D for UI - clear depth buffer and use fresh matrices
        glClear(GL_DEPTH_BUFFER_BIT)  # Clear depth so 2D always renders on top
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, self.height, 0)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)  # Disable depth writes
        glDisable(GL_LIGHTING)
        glDisable(GL_CULL_FACE)  # CRITICAL: Disable backface culling for 2D
        glDisable(GL_TEXTURE_2D)  # Ensure no textures bound
        glBindTexture(GL_TEXTURE_2D, 0)  # Unbind any texture
        
        # Create pygame surface for UI
        ui_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        ui_surface.fill((0, 0, 0, 0))
        
        # Draw VERY VISIBLE backgrounds for UI panels
        # Config panel (left) - Bright blue background
        bg = pygame.Surface((280, self.height - 20), pygame.SRCALPHA)
        bg.fill((50, 100, 200, 200))  # Semi-transparent blue
        ui_surface.blit(bg, (10, 10))
        pygame.draw.rect(ui_surface, (100, 200, 255), (10, 10, 280, self.height - 20), 3)
        
        # Graph panel (top right) - Bright orange background
        bg = pygame.Surface((280, 400), pygame.SRCALPHA)
        bg.fill((200, 100, 50, 200))  # Semi-transparent orange
        ui_surface.blit(bg, (self.width - 290, 10))
        pygame.draw.rect(ui_surface, (255, 150, 100), (self.width - 290, 10, 280, 400), 3)
        
        # Console (bottom center) - Bright green background
        console_w = self.width - 600
        bg = pygame.Surface((console_w, 200), pygame.SRCALPHA)
        bg.fill((50, 200, 100, 200))  # Semi-transparent green
        ui_surface.blit(bg, (300, self.height - 210))
        pygame.draw.rect(ui_surface, (100, 255, 150), (300, self.height - 210, console_w, 200), 3)
        
        # Render UI panels
        self.config_panel.render(ui_surface, self.font, self.small_font)
        self.graph_panel.render(ui_surface, self.font, self.small_font)
        self.console.render(ui_surface, self.small_font)
        
        # Render Circuit8 panel (if enabled) - above the 3D world as a monitor
        if self.show_circuit8:
            self._render_circuit8_panel(ui_surface)
        
        # Render stats overlay (center top)
        self._render_stats_overlay(ui_surface)
        
        # Render thought bubbles (if enabled)
        if self.show_thoughts and hasattr(self, '_thoughts_to_render'):
            self._render_thought_bubbles(ui_surface)
        
        # Render species color table (right side, below graphs)
        self._render_species_panel(ui_surface)
        
        # Render population aggregate stats (below Circuit8)
        self._render_population_stats(ui_surface)
        
        # Render creature inspector (if a creature is selected)
        if self.selected_creature:
            self._render_inspector_panel(ui_surface)
        
        # Render extinction overlay
        if self.extinct:
            self._render_extinction_overlay(ui_surface)
        
        # Render help overlay (if enabled)
        if self.show_help:
            self._render_help_overlay(ui_surface)
        
        # Blit to OpenGL
        self._blit_pygame_to_opengl(ui_surface)
        
        # Restore 3D state (pop matrices from earlier push)
        glDepthMask(GL_TRUE)  # Re-enable depth writes
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)  # Re-enable backface culling for 3D
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        pygame.display.flip()
    
    def _render_sky(self):
        """Render sky gradient background (critterding-style atmosphere)."""
        # Switch to 2D orthographic for sky quad
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, 1, 0, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Sky gradient: deep blue at top -> lighter blue-white at horizon (bottom)
        glBegin(GL_QUADS)
        # Bottom (horizon) - light blue/white
        glColor3f(0.65, 0.78, 0.92)
        glVertex2f(0, 0)
        glVertex2f(1, 0)
        # Top - deeper blue
        glColor3f(0.25, 0.45, 0.85)
        glVertex2f(1, 1)
        glVertex2f(0, 1)
        glEnd()
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        # Clear only depth buffer so sky stays as background
        glClear(GL_DEPTH_BUFFER_BIT)
    
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
        self.console.add_line(f"💊 Gave ALL creatures {drug_names[molecule_type]}")
    
    def _give_drug_to_creature(self, creature, molecule_type: int):
        """Give drug to a single selected creature."""
        if hasattr(creature, 'drugs'):
            creature.drugs.tripping[molecule_type] += 5.0
            drug_names = ["Inh. Antag.", "Inh. Agon.", "Exc. Antag.", "Exc. Agon.", "Potent."]
            cid = getattr(creature, 'creature_id', '?')
            self.console.add_line(f"💊 Dosed creature {cid} with {drug_names[molecule_type]}")
    
    def _update_audio(self):
        """Generate and play audio from creature neural activity."""
        if not self.creatures:
            return
        
        creature = self.creatures[0]
        if not hasattr(creature, 'network'):
            return
        
        try:
            audio_samples = self.audio_synth.synthesize_from_network(
                creature.network, 
                duration_seconds=0.02
            )
            audio_int16 = (audio_samples * 32767).astype(np.int16)
            stereo_audio = np.column_stack((audio_int16, audio_int16))
            sound = pygame.sndarray.make_sound(stereo_audio)
            sound.play()
            self.sound_queue.append(sound)
            if len(self.sound_queue) > 16:
                self.sound_queue.pop(0)
        except Exception:
            pass
    
    def _try_inspect_creature(self, mouse_pos):
        """Try to pick and inspect a creature at mouse position."""
        # Get ray from mouse position
        ray_origin, ray_dir = self._get_ray_from_mouse(mouse_pos)
        
        # Find closest creature hit
        closest_creature = None
        closest_dist = float('inf')
        
        for creature in self.creatures:
            # Simple sphere intersection test
            pos = np.array([creature.x, creature.y, creature.z if hasattr(creature, 'z') else 10.0])
            radius = 15.0  # Approximate creature radius for picking
            
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
            if self.selected_creature == closest_creature:
                self.selected_creature = None  # Close inspector
                self.console.add_line("🔍 Inspector closed")
            else:
                self.selected_creature = closest_creature
                self.console.add_line(f"🔍 Inspecting creature {closest_creature.creature_id}")
        else:
            # Clicked on nothing - close inspector
            if self.selected_creature:
                self.selected_creature = None
                self.console.add_line("🔍 Inspector closed")
    
    def _get_ray_from_mouse(self, mouse_pos):
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
    
    def _render_inspector_panel(self, surface):
        """Render creature inspector panel on the right side of screen."""
        c = self.selected_creature
        if not c:
            return
        
        # Build inspector info lines
        inspector_lines = [
            f"🔍 CREATURE {c.creature_id}",
            "",
            "VITALS:",
            f"  Age: {c.age} | Generation: {c.generation}",
            f"  Energy: {c.energy.energy:,.0f} / {c.energy.max_energy:,.0f}",
            f"  Position: ({c.x:.0f}, {c.y:.0f}, {c.z if hasattr(c, 'z') else 0:.0f})",
        ]
        
        # Adam distance if available
        if hasattr(c, 'adam_distance'):
            inspector_lines.append(f"  Adam Distance: {c.adam_distance}")
        
        inspector_lines.extend([
            "",
            "BRAIN:",
            f"  Neurons: {len(c.network.neurons)}",
            f"  Synapses: {len(c.network.synapses)}",
            f"  Activity: {c.network.get_activity_level()*100:.1f}%",
            f"  Plasticity: {'ON' if c.network.enable_plasticity else 'OFF'}",
            "",
            "DRUGS:",
        ])
        
        # Drug levels
        drug_names = ["Inh.Antag", "Inh.Agon", "Exc.Antag", "Exc.Agon", "POTENT"]
        for i, name in enumerate(drug_names):
            if hasattr(c, 'drugs'):
                level = c.drugs.tripping[i]
                if level > 1.0:
                    bar_len = int(min(15, level / 5000))
                    bar = "█" * bar_len
                    inspector_lines.append(f"  {name}: {level:>7.0f} {bar}")
        
        inspector_lines.extend([
            "",
            "BEHAVIOR:",
        ])
        
        # Current behavior
        if hasattr(c, 'current_behavior'):
            inspector_lines.append(f"  State: {c.current_behavior.name}")
        if hasattr(c, 'food_consumed'):
            inspector_lines.append(f"  Food eaten: {c.food_consumed}")
        
        # Addiction info
        if hasattr(c, 'behavior'):
            if hasattr(c.behavior, 'addiction_level'):
                inspector_lines.append(f"  Addiction: {c.behavior.addiction_level:.2f}")
            if hasattr(c.behavior, 'tolerance'):
                inspector_lines.append(f"  Tolerance: {c.behavior.tolerance:.2f}")
            if hasattr(c.behavior, 'withdrawal_severity'):
                inspector_lines.append(f"  Withdrawal: {c.behavior.withdrawal_severity:.2f}")
        
        # Velocity if available
        if hasattr(c, 'get_velocity'):
            vel = c.get_velocity()
            speed = np.linalg.norm(vel)
            inspector_lines.extend([
                "",
                "PHYSICS:",
                f"  Speed: {speed:.1f}",
                f"  Velocity: ({vel[0]:.1f}, {vel[1]:.1f}, {vel[2]:.1f})",
            ])
        
        # Current thought
        if hasattr(c, 'get_current_thought'):
            thought = c.get_current_thought()
            if thought:
                inspector_lines.extend([
                    "",
                    "LAST THOUGHT:",
                    f"  {thought[:35]}",
                ])
        
        # Social learning stats
        if hasattr(c, 'learner'):
            inspector_lines.extend([
                "",
                "SOCIAL LEARNING:",
                f"  Observations: {len(c.learner.observation_history)}",
                f"  Behaviors: {len(c.learner.behavior_success_rates)}",
            ])
        
        # Create panel surface
        panel_width = 280
        button_section_height = 55  # Space for buttons
        panel_height = min(600, len(inspector_lines) * 16 + button_section_height + 20)
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((20, 40, 60, 220))  # Dark blue semi-transparent
        pygame.draw.rect(panel, (100, 200, 255), (0, 0, panel_width, panel_height), 2)
        
        # Render each line
        y = 10
        for line in inspector_lines:
            if line:
                # Color code
                if "🔍" in line:
                    color = (255, 255, 0)  # Yellow header
                elif line.endswith(":") and not line.startswith(" "):
                    color = (100, 200, 255)  # Cyan section headers
                elif "POTENT" in line or "█" in line:
                    color = (255, 100, 255)  # Magenta for drugs
                else:
                    color = (220, 220, 220)  # White for data
                
                text_surf = self.small_font.render(line, True, color)
                panel.blit(text_surf, (10, y))
            y += 16
        
        # === Action Buttons (critterding heritage) ===
        y += 4
        button_defs = [
            ('kill', (180, 60, 60), 'kill'),
            ('duplicate', (60, 120, 180), 'duplicate'),
            ('feed', (60, 160, 60), 'feed'),
            ('brain mut', (180, 140, 60), 'brain_mutant'),
            ('body mut', (140, 60, 180), 'body_mutant'),
        ]
        
        # Position panel on right side
        x_pos = self.width - 290
        y_pos = 420  # Below the graph panel
        
        # Store button positions for click detection (in screen coords)
        self.inspector_buttons = []
        btn_x = 10
        btn_w = 50
        btn_h = 20
        btn_gap = 3
        for label, color, action in button_defs:
            # Draw button on panel
            pygame.draw.rect(panel, color, (btn_x, y, btn_w, btn_h))
            pygame.draw.rect(panel, (255, 255, 255), (btn_x, y, btn_w, btn_h), 1)
            btn_text = self.small_font.render(label, True, (255, 255, 255))
            panel.blit(btn_text, (btn_x + 3, y + 3))
            
            # Store screen-space rect for click detection
            screen_rect = pygame.Rect(x_pos + btn_x, y_pos + y, btn_w, btn_h)
            self.inspector_buttons.append((screen_rect, action))
            
            btn_x += btn_w + btn_gap
        
        surface.blit(panel, (x_pos, y_pos))
    
    def _render_neural_activity(self):
        """Render neural activity visualization above creatures.
        
        Shows each creature's neural activity as colored particles:
        - Excitatory neurons firing: Yellow/orange points
        - Inhibitory neurons firing: Blue/cyan points
        - Activity level shown by particle density and brightness
        """
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Additive blending for glow effect
        glPointSize(3.0)
        
        for creature in self.creatures:
            if not hasattr(creature, 'network'):
                continue
            
            network = creature.network
            cx, cy = creature.x, creature.y
            cz = creature.z if hasattr(creature, 'z') else 10.0
            
            # Visualization height above creature
            viz_height = 30.0
            viz_radius = 15.0
            
            # Count firing neurons for this frame
            excitatory_firing = []
            inhibitory_firing = []
            
            for i, neuron in enumerate(network.neurons[:min(100, len(network.neurons))]):
                # Check if neuron fired recently (high potential)
                if neuron.potential > neuron.threshold * 0.7:
                    # Position in a circle above creature
                    angle = (i / len(network.neurons)) * 2 * np.pi
                    layer = i % 3  # Multiple layers
                    radius = viz_radius * (0.5 + layer * 0.25)
                    
                    px = cx + radius * np.cos(angle)
                    py = cy + radius * np.sin(angle)
                    pz = cz + viz_height + layer * 5
                    
                    # Brightness based on potential
                    brightness = min(1.0, neuron.potential / neuron.threshold)
                    
                    if neuron.is_inhibitory:
                        inhibitory_firing.append((px, py, pz, brightness))
                    else:
                        excitatory_firing.append((px, py, pz, brightness))
            
            # Render excitatory neurons (yellow-orange)
            if excitatory_firing:
                glBegin(GL_POINTS)
                for px, py, pz, brightness in excitatory_firing:
                    glColor4f(1.0, 0.8 * brightness, 0.2 * brightness, brightness * 0.8)
                    glVertex3f(px, py, pz)
                glEnd()
            
            # Render inhibitory neurons (blue-cyan)
            if inhibitory_firing:
                glBegin(GL_POINTS)
                for px, py, pz, brightness in inhibitory_firing:
                    glColor4f(0.2 * brightness, 0.6 * brightness, 1.0, brightness * 0.8)
                    glVertex3f(px, py, pz)
                glEnd()
            
            # Draw activity ring around creature (based on overall activity)
            activity = network.get_activity_level()
            if activity > 0.01:
                glBegin(GL_LINE_LOOP)
                ring_color = (1.0, 1.0 - activity, 1.0 - activity, activity)
                glColor4f(*ring_color)
                for i in range(16):
                    angle = i * 2 * np.pi / 16
                    glVertex3f(
                        cx + viz_radius * 1.2 * np.cos(angle),
                        cy + viz_radius * 1.2 * np.sin(angle),
                        cz + viz_height - 5
                    )
                glEnd()
        
        glPointSize(1.0)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
    
    def _render_social_learning(self):
        """Render social learning connections between creatures.
        
        Shows:
        - Lines between creatures who have observed each other
        - Color indicates learning success (green=successful, red=failed)
        - Thickness indicates frequency of interaction
        """
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(2.0)
        
        # Track rendered connections to avoid duplicates
        rendered_pairs = set()
        
        for creature in self.creatures:
            if not hasattr(creature, 'learner'):
                continue
            
            cx1 = creature.x
            cy1 = creature.y
            cz1 = creature.z if hasattr(creature, 'z') else 10.0
            
            # Check observation history
            if hasattr(creature.learner, 'observation_history'):
                # Convert deque to list for slicing (deques don't support slice notation)
                history = list(creature.learner.observation_history)
                for observation in history[-5:]:  # Last 5 observations
                    # Find the creature that was observed (BehaviorObservation dataclass)
                    observed_id = getattr(observation, 'creature_id', None)
                    if observed_id is None:
                        continue
                    
                    for other in self.creatures:
                        if getattr(other, 'creature_id', None) == observed_id:
                            # Create pair key to avoid duplicates
                            pair_key = tuple(sorted([id(creature), id(other)]))
                            if pair_key in rendered_pairs:
                                continue
                            rendered_pairs.add(pair_key)
                            
                            cx2 = other.x
                            cy2 = other.y
                            cz2 = other.z if hasattr(other, 'z') else 10.0
                            
                            # Line color based on success
                            outcome = getattr(observation, 'outcome', None)
                            if outcome == 'success':
                                glColor4f(0.0, 1.0, 0.3, 0.4)  # Green for successful learning
                            else:
                                glColor4f(1.0, 0.5, 0.0, 0.3)  # Orange for observation
                            
                            # Draw connection line
                            glBegin(GL_LINES)
                            glVertex3f(cx1, cy1, cz1 + 5)
                            glVertex3f(cx2, cy2, cz2 + 5)
                            glEnd()
            
            # Draw awareness radius circle
            if hasattr(creature, 'sight_range'):
                sight_range = creature.sight_range
            else:
                sight_range = 100.0
            
            # Only draw if creature has learned behaviors
            if hasattr(creature.learner, 'behavior_success_rates'):
                n_behaviors = len(creature.learner.behavior_success_rates)
                if n_behaviors > 0:
                    # Color based on learning progress (more behaviors = more cyan)
                    learning_progress = min(1.0, n_behaviors / 5.0)
                    glColor4f(0.2, 0.5 + 0.5 * learning_progress, 1.0, 0.15)
                    
                    glBegin(GL_LINE_LOOP)
                    for i in range(24):
                        angle = i * 2 * np.pi / 24
                        glVertex3f(
                            cx1 + sight_range * 0.3 * np.cos(angle),
                            cy1 + sight_range * 0.3 * np.sin(angle),
                            cz1 + 2
                        )
                    glEnd()
        
        glLineWidth(1.0)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
    
    def _render_circuit8_panel(self, surface):
        """Render Circuit8 as a 2D UI panel (telepathic canvas monitor).
        
        Displays the 64x48 collective canvas as a scaled-up panel in the UI,
        separate from the 3D world. This is the shared morphic field that
        all creatures read/write to.
        """
        # Panel dimensions and position (top center, between config and graph panels)
        panel_scale = 4  # 64*4=256 wide, 48*4=192 tall
        panel_w = self.circuit8.width * panel_scale
        panel_h = self.circuit8.height * panel_scale
        panel_x = (self.width - panel_w) // 2
        panel_y = 10  # Top of screen
        
        # Background
        bg = pygame.Surface((panel_w + 8, panel_h + 28), pygame.SRCALPHA)
        bg.fill((20, 10, 40, 220))  # Dark purple background
        surface.blit(bg, (panel_x - 4, panel_y - 4))
        
        # Title
        title = self.small_font.render("CIRCUIT8 — Telepathic Canvas", True, (180, 140, 255))
        surface.blit(title, (panel_x + (panel_w - title.get_width()) // 2, panel_y - 2))
        
        # Render Circuit8 pixels to a pygame surface
        c8_surface = pygame.Surface((self.circuit8.width, self.circuit8.height))
        
        # Boost brightness for visibility
        screen = self.circuit8.screen.astype(np.float32)
        boosted = np.clip(screen * 2.0 + 8.0, 0, 255).astype(np.uint8)
        
        # Copy pixel data to surface
        pygame.surfarray.blit_array(c8_surface, boosted.transpose(1, 0, 2))
        
        # Scale up with nearest-neighbor (pixelated look)
        scaled = pygame.transform.scale(c8_surface, (panel_w, panel_h))
        surface.blit(scaled, (panel_x, panel_y + 18))
        
        # Border
        pygame.draw.rect(surface, (120, 80, 200), (panel_x - 2, panel_y + 16, panel_w + 4, panel_h + 4), 2)
    
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
            "  Mouse drag: Pan | Scroll: Zoom | Arrows: Pan",
            "  Right-click: Inspect creature (detailed stats)",
            "  R: Reset camera",
            "",
            "SIMULATION:",
            "  Space: Pause/unpause",
            "  [/]: Slow/speed time (0.1x-10x)",
            "  K: Kill half population",
            "  N: New simulation (restart)",
            "",
            "INTERACTIVE:",
            "  F: Spawn food | D: Spawn drug mushroom",
            "  I: Apply impulses (physics test)",
            "  9: Excitatory Agonist | 0: Potentiator",
            "  F1-F4: Drug types (to all or selected creature)",
            "",
            "VISUALIZATION:",
            "  1-8: Render modes (1=creatures, 8=all)",
            "  C: Toggle Circuit8 | T: Toggle thoughts",
            "  P: Toggle psychedelic | U: Toggle audio",
            "  H: Toggle this help",
            "",
            "CONFIG:",
            "  V: Save config",
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
    
    def _render_extinction_overlay(self, surface):
        """Render extinction message overlay."""
        # Dark overlay
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        surface.blit(overlay, (0, 0))
        
        # Extinction message box
        box_w, box_h = 400, 120
        box_x = (self.width - box_w) // 2
        box_y = (self.height - box_h) // 2
        
        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((80, 10, 10, 230))
        surface.blit(bg, (box_x, box_y))
        pygame.draw.rect(surface, (255, 60, 60), (box_x, box_y, box_w, box_h), 3)
        
        # Text
        title = self.font.render("EXTINCTION", True, (255, 80, 80))
        surface.blit(title, (box_x + (box_w - title.get_width()) // 2, box_y + 15))
        
        msg = self.small_font.render("All creatures have died.", True, (220, 180, 180))
        surface.blit(msg, (box_x + (box_w - msg.get_width()) // 2, box_y + 50))
        
        restart = self.font.render("Press N to start new simulation", True, (255, 255, 100))
        surface.blit(restart, (box_x + (box_w - restart.get_width()) // 2, box_y + 80))
    
    def _restart_simulation(self):
        """Restart simulation with fresh creatures."""
        # Clear old creatures and their physics bodies
        for creature in self.creatures:
            if hasattr(creature, 'rigid_body') and creature.rigid_body and self.physics_world:
                self.physics_world.remove_body(creature.rigid_body.id)
        self.creatures.clear()
        
        # Reset state
        self.timestep = 0
        self.extinct = False
        self.paused = False
        self.collision_count = 0
        
        # Reset Circuit8
        self.circuit8.screen[:] = 0
        
        # Respawn resources
        self.resource_manager.resources.clear()
        self.resource_manager.spawn_initial_resources()
        
        # Spawn fresh creatures
        initial_count = int(self.config.get("initial_population"))
        self._spawn_initial_creatures(count=initial_count)
        
        self.console.add_line(f"=== NEW SIMULATION — {len(self.creatures)} creatures ===")
    
    def _render_species_panel(self, surface):
        """Render species color table (critterding heritage).
        
        Shows each species with color swatch, population count, and adam distance.
        Like the right panel in critterding beta13.
        """
        if not self.species_tracker.species:
            return
        
        panel_width = 260
        row_height = 16
        header_height = 22
        num_species = len(self.species_tracker.species)
        panel_height = min(300, header_height + num_species * row_height + 10)
        
        x_pos = self.width - panel_width - 20
        y_pos = 420  # Below graph panel
        
        # Background
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((15, 15, 30, 210))
        pygame.draw.rect(panel, (80, 120, 180), (0, 0, panel_width, panel_height), 2)
        
        # Header
        header = self.small_font.render("#  Color  Population  Adam Dist", True, (180, 200, 255))
        panel.blit(header, (8, 4))
        
        # Species rows sorted by population
        sorted_species = sorted(
            self.species_tracker.species.values(),
            key=lambda s: len(s.member_ids),
            reverse=True
        )
        
        y = header_height
        for i, species in enumerate(sorted_species[:16]):  # Max 16 rows
            pop = len(species.member_ids)
            if pop == 0:
                continue
            
            # Color swatch
            color = species.color
            pygame.draw.rect(panel, color, (8, y + 2, 12, 12))
            
            # Compute avg adam distance for this species
            avg_ad = 0
            ad_count = 0
            for creature in self.creatures:
                if creature.creature_id in species.member_ids:
                    avg_ad += getattr(creature, 'adam_distance', 0)
                    ad_count += 1
            avg_ad = avg_ad / max(1, ad_count)
            
            # Row text
            row_text = f"{species.species_id:>3}  {'':5}  {pop:>5}       {avg_ad:>6.0f}"
            text_surf = self.small_font.render(row_text, True, (200, 200, 200))
            panel.blit(text_surf, (8, y))
            
            # Population bar
            bar_width = min(40, pop * 3)
            pygame.draw.rect(panel, color, (130, y + 3, bar_width, 10))
            
            y += row_height
        
        surface.blit(panel, (x_pos, y_pos))
    
    def _render_population_stats(self, surface):
        """Render population aggregate statistics (critterding stats panel).
        
        Shows avg neurons, synapses, synapses/neuron, adam distance, body parts.
        """
        if not self.creatures:
            return
        
        # Calculate aggregates
        total_neurons = 0
        total_synapses = 0
        total_ad = 0
        total_body_parts = 0
        total_weight = 0.0
        n = len(self.creatures)
        
        for c in self.creatures:
            if hasattr(c, 'network'):
                total_neurons += len(c.network.neurons)
                total_synapses += len(c.network.synapses)
            total_ad += getattr(c, 'adam_distance', 0)
            if hasattr(c, 'body') and c.body:
                total_body_parts += len(c.body.segments)
                total_weight += sum(s.size * s.length for s in c.body.segments)
        
        avg_neurons = total_neurons / n
        avg_synapses = total_synapses / n
        avg_syn_per_neuron = avg_synapses / max(1, avg_neurons)
        avg_ad = total_ad / n
        avg_body_parts = total_body_parts / n
        avg_weight = total_weight / n
        
        # Render compact stats block
        stats_lines = [
            ("brain", (100, 200, 255)),
            (f"  avg neurons:      {avg_neurons:>8.2f}", (200, 200, 200)),
            (f"  avg synapses:     {avg_synapses:>8.2f}", (200, 200, 200)),
            (f"  avg syn/neuron:   {avg_syn_per_neuron:>8.2f}", (200, 200, 200)),
            (f"  avg adam distance:{avg_ad:>8.2f}", (200, 200, 200)),
            ("", (0, 0, 0)),
            ("body", (100, 200, 255)),
            (f"  avg body parts:   {avg_body_parts:>8.2f}", (200, 200, 200)),
            (f"  avg weight:       {avg_weight:>8.2f}", (200, 200, 200)),
        ]
        
        panel_width = 250
        panel_height = len(stats_lines) * 14 + 12
        
        # Position below Circuit8 panel
        panel_x = (self.width - panel_width) // 2
        panel_y = 235  # Below Circuit8 (which is ~210px from top)
        
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((10, 15, 25, 200))
        pygame.draw.rect(panel, (60, 90, 140), (0, 0, panel_width, panel_height), 1)
        
        y = 6
        for text, color in stats_lines:
            if text:
                text_surf = self.small_font.render(text, True, color)
                panel.blit(text_surf, (8, y))
            y += 14
        
        surface.blit(panel, (panel_x, panel_y))
    
    def _inspector_button_action(self, action: str):
        """Execute an inspector button action on the selected creature."""
        c = self.selected_creature
        if not c or c not in self.creatures:
            self.selected_creature = None
            return
        
        if action == 'kill':
            self.creatures.remove(c)
            self.logger.log_death(c, self.timestep, "manual_kill")
            self.console.add_line(f"Killed creature {c.creature_id}")
            self.selected_creature = None
        
        elif action == 'duplicate':
            offspring_genotype = c.genotype.mutate(mutation_rate=0.5, max_mutations=50)
            offspring_body = c.body.mutate(mutation_rate=0.3) if c.body else None
            from creatures.collective_creature import CollectiveCreature
            clone = CollectiveCreature(
                genotype=offspring_genotype,
                body=offspring_body,
                x=c.x + np.random.uniform(-10, 10),
                y=c.y + np.random.uniform(-10, 10),
                z=c.z if hasattr(c, 'z') else 10.0,
                initial_energy=c.energy.energy * 0.5,
                circuit8=self.circuit8,
                physics_world=self.physics_world,
                collective_memory=self.collective_memory,
            )
            clone.generation = c.generation + 1
            clone.adam_distance = c.adam_distance + 1
            self.creatures.append(clone)
            self.logger.log_reproduction(c, None, clone, self.timestep)
            self.console.add_line(f"Duplicated creature {c.creature_id} -> {clone.creature_id}")
        
        elif action == 'feed':
            c.energy.energy = min(c.energy.max_energy, c.energy.energy + 200000)
            self.console.add_line(f"Fed creature {c.creature_id} (+200k energy)")
        
        elif action == 'brain_mutant':
            c.genotype = c.genotype.mutate(mutation_rate=0.8, max_mutations=100)
            self.logger.log_mutation(c, self.timestep, "brain mutant")
            self.console.add_line(f"Brain mutated creature {c.creature_id}")
        
        elif action == 'body_mutant':
            if c.body:
                c.body = c.body.mutate(mutation_rate=0.8)
                c.mesh = None  # Force mesh regeneration
                self.logger.log_mutation(c, self.timestep, "body mutant")
                self.console.add_line(f"Body mutated creature {c.creature_id}")
    
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
        """Blit pygame surface to OpenGL using texture quad.
        
        Uses the same approach as phase9c_demo.py which works correctly:
        - Flip surface vertically (pygame Y=0 is top, OpenGL Y=0 is bottom)
        - Use texture coordinates that map flipped surface correctly
        """
        width = surface.get_width()
        height = surface.get_height()
        
        # CRITICAL: Flip surface vertically for OpenGL (like phase9c)
        flipped_surface = pygame.transform.flip(surface, False, True)
        texture_data = pygame.image.tostring(flipped_surface, "RGBA", False)
        
        # Create texture if needed
        if not hasattr(self, '_ui_texture'):
            self._ui_texture = glGenTextures(1)
        
        # Bind and upload texture
        glBindTexture(GL_TEXTURE_2D, self._ui_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        
        # Enable texture and blending
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Draw fullscreen quad with texture coordinates matching phase9c
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(0, 0)
        glTexCoord2f(1, 1); glVertex2f(width, 0)
        glTexCoord2f(1, 0); glVertex2f(width, height)
        glTexCoord2f(0, 0); glVertex2f(0, height)
        glEnd()
        
        # Cleanup
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_BLEND)
        glBindTexture(GL_TEXTURE_2D, 0)
    
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
