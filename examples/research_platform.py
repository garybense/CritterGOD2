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
        
        # Use create_collective_creatures helper
        self.creatures = create_collective_creatures(
            n_creatures=10,
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
    
    def update(self, dt=1.0):
        """Update simulation."""
        if self.paused:
            return
        
        self.timestep += 1
        
        # Update resources
        self.resource_manager.update()
        
        # Update creatures
        alive_creatures = []
        for creature in self.creatures:
            if creature.update(dt):
                alive_creatures.append(creature)
            else:
                # Log death
                self.logger.log_death(creature, self.timestep, "starvation")
        
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
        
        # Render simple creatures as spheres
        glEnable(GL_DEPTH_TEST)
        for creature in self.creatures:
            glPushMatrix()
            glTranslatef(creature.x, 10, creature.y)
            # Color by energy
            energy_ratio = min(1.0, creature.energy.energy / 500000)
            glColor3f(0.5 + energy_ratio * 0.5, 0.3, 0.3)
            # Simple sphere (using pygame)
            glPopMatrix()
        
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
        
        # Blit to OpenGL
        self._blit_pygame_to_opengl(ui_surface)
        
        glEnable(GL_DEPTH_TEST)
        
        pygame.display.flip()
    
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
