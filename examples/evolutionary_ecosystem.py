"""
Phase 6: Complete Evolutionary Ecosystem

THE GRAND INTEGRATION - Everything working together:
- Neural networks with STDP plasticity and Phase 5a refinements
- Circuit8 telepathic shared hallucination space
- Psychopharmacology (5 molecule types affecting consciousness)
- Multi-modal creatures: see, speak, think, create
- Audio synthesis (hear the collective consciousness)
- Evolutionary text generation (genetic language)
- Visual patterns (neural art)
- Energy-driven evolution with birth/death cycles
- Force-directed species clustering visualization
- Democratic voting and collective intelligence

This is flamoot's vision realized - psychedelic computing as artificial life.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from typing import List, Optional

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available - visualization disabled")

from core.evolution.genotype import Genotype
from core.morphic.circuit8 import Circuit8
from core.pharmacology.drugs import Pill, MoleculeType
from creatures.enhanced_creature import EnhancedCreature
from visualization.force_directed_layout import ForceDirectedLayout
from generators.audio.neural_audio import NeuralAudioSynthesizer


class EvolutionaryEcosystem:
    """
    Complete artificial life ecosystem.
    
    Features:
    - Self-sustaining population with birth/death
    - Circuit8 shared hallucination space
    - Psychopharmacology system
    - Multi-modal creatures (vision, audio, text)
    - Force-directed species clustering
    - Real-time audio synthesis
    - Energy-driven evolution
    """
    
    def __init__(
        self,
        initial_population: int = 20,
        max_population: int = 50,
        world_width: int = 1200,
        world_height: int = 800,
        enable_audio: bool = True,
        enable_drugs: bool = True
    ):
        """
        Initialize the ecosystem.
        
        Args:
            initial_population: Starting number of creatures
            max_population: Maximum population cap
            world_width: World width in pixels
            world_height: World height in pixels
            enable_audio: Enable audio synthesis
            enable_drugs: Enable psychopharmacology
        """
        self.world_width = world_width
        self.world_height = world_height
        self.max_population = max_population
        self.enable_drugs = enable_drugs
        
        # Shared telepathic canvas (the collective unconscious)
        self.circuit8 = Circuit8(width=64, height=48)
        
        # Population
        self.creatures: List[EnhancedCreature] = []
        self.generation = 0
        self.total_births = 0
        self.total_deaths = 0
        
        # Layout engine for species clustering visualization
        self.layout = ForceDirectedLayout(
            repulsion_strength=100.0,
            attraction_strength=0.5,
            damping=0.8,
            similarity_threshold=0.7
        )
        
        # Audio synthesis (hear the collective consciousness)
        self.audio_enabled = enable_audio and PYGAME_AVAILABLE
        if self.audio_enabled:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
                self.audio_synth = NeuralAudioSynthesizer(
                    sample_rate=22050,
                    buffer_size=512,
                    mode='mixed'
                )
                self.audio_channel = pygame.mixer.Channel(0)
            except:
                print("Audio initialization failed - continuing without audio")
                self.audio_enabled = False
        
        # Drugs scattered in the world (pills for creatures to consume)
        self.pills: List[Pill] = []
        
        # Statistics
        self.stats = {
            'generation': 0,
            'population': 0,
            'births': 0,
            'deaths': 0,
            'avg_energy': 0.0,
            'avg_neurons': 0.0,
            'avg_age': 0.0,
            'species_count': 0,
            'circuit8_activity': 0.0,
            'avg_fitness': 0.0
        }
        
        # Create initial population
        print(f"Creating initial population of {initial_population} creatures...")
        self._create_initial_population(initial_population)
        
        # Scatter some drugs
        if self.enable_drugs:
            self._scatter_drugs(30)
        
        print(f"Ecosystem initialized with {len(self.creatures)} creatures")
    
    def _create_initial_population(self, count: int):
        """Create initial random population."""
        for i in range(count):
            # Vary network complexity
            n_hidden = np.random.randint(40, 120)
            synapses_per = np.random.randint(15, 35)
            
            genotype = Genotype.create_random(
                n_sensory=10,
                n_motor=10,
                n_hidden_min=n_hidden,
                n_hidden_max=n_hidden + 20,
                synapses_per_neuron=synapses_per
            )
            
            # Random position
            x = np.random.uniform(100, self.world_width - 100)
            y = np.random.uniform(100, self.world_height - 100)
            
            creature = EnhancedCreature(
                genotype=genotype,
                x=x,
                y=y,
                initial_energy=500000.0,  # Start with good energy
                circuit8=self.circuit8,
                adam_distance=0  # First generation
            )
            
            self.creatures.append(creature)
            self.total_births += 1
    
    def _scatter_drugs(self, count: int):
        """Scatter drug pills in the world."""
        for _ in range(count):
            x = np.random.uniform(0, self.world_width)
            y = np.random.uniform(0, self.world_height)
            
            # Random molecule type
            molecule = np.random.randint(0, 5)
            amount = np.random.uniform(50.0, 200.0)
            
            # Create molecule composition array
            composition = [0, 0, 0, 0, 0]
            composition[molecule] = amount
            
            pill = Pill(
                x=x,
                y=y,
                molecule_composition=composition
            )
            self.pills.append(pill)
    
    def update(self, dt: float = 1.0):
        """
        Update ecosystem for one timestep.
        
        This is where the magic happens:
        1. Update all creatures (neural networks, drugs, motors)
        2. Handle drug consumption
        3. Remove dead creatures
        4. Reproduce if energy allows
        5. Update force-directed layout
        6. Synthesize collective audio
        7. Update Circuit8 based on voting
        """
        # 1. Update all creatures
        for creature in self.creatures:
            alive = creature.update(dt)
            
            if not alive:
                continue
            
            # Keep in bounds (toroidal world)
            creature.x = creature.x % self.world_width
            creature.y = creature.y % self.world_height
        
        # 2. Drug consumption (if enabled)
        if self.enable_drugs:
            self._handle_drug_consumption()
        
        # 3. Remove dead creatures
        dead = [c for c in self.creatures if c.energy.energy <= 0]
        self.total_deaths += len(dead)
        self.creatures = [c for c in self.creatures if c.energy.energy > 0]
        
        # 4. Reproduction (if population below max and energy allows)
        if len(self.creatures) < self.max_population:
            self._handle_reproduction()
        
        # 5. Update force-directed layout (smooth visual clustering)
        if len(self.creatures) > 1:
            self.layout.simulate_step(self.creatures, dt=0.1)
        
        # 6. Synthesize collective audio (hear the neural symphony)
        if self.audio_enabled and len(self.creatures) > 0:
            self._synthesize_audio()
        
        # 7. Apply collective voting to Circuit8
        self.circuit8.apply_votes()
        
        # 8. Update statistics
        self._update_stats()
    
    def _handle_drug_consumption(self):
        """Creatures consume nearby drugs."""
        for creature in self.creatures:
            for pill in self.pills[:]:  # Copy list to allow removal
                dx = creature.x - pill.x
                dy = creature.y - pill.y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < 30.0:  # Consumption radius
                    creature.consume_drug(pill)
                    self.pills.remove(pill)
                    break  # One pill per creature per timestep
        
        # Scatter new drugs occasionally
        if np.random.random() < 0.01:  # 1% chance per timestep
            self._scatter_drugs(1)
    
    def _handle_reproduction(self):
        """Handle creature reproduction."""
        # Find creatures with enough energy to reproduce
        candidates = [c for c in self.creatures if c.energy.energy > 800000.0]
        
        if not candidates:
            return
        
        # Select one at random
        parent = np.random.choice(candidates)
        
        # Create offspring
        offspring_genotype = parent.genotype.mutate(mutation_rate=0.3)
        
        # Place near parent with some noise
        offspring_x = parent.x + np.random.uniform(-50, 50)
        offspring_y = parent.y + np.random.uniform(-50, 50)
        offspring_x = offspring_x % self.world_width
        offspring_y = offspring_y % self.world_height
        
        # Create offspring
        offspring = EnhancedCreature(
            genotype=offspring_genotype,
            x=offspring_x,
            y=offspring_y,
            initial_energy=300000.0,  # Start with moderate energy
            circuit8=self.circuit8,
            adam_distance=parent.adam_distance + 1  # Track lineage
        )
        
        # Inherit language from parent (genetic transmission)
        if hasattr(parent, 'markov') and hasattr(offspring, 'markov'):
            # Copy some word pairs from parent
            parent_pairs = list(parent.markov.word_pairs.keys())
            if parent_pairs:
                # Inherit ~50% of parent's language
                inherited = np.random.choice(
                    parent_pairs,
                    size=min(len(parent_pairs) // 2, 20),
                    replace=False
                )
                for pair in inherited:
                    if pair in parent.markov.word_pairs:
                        offspring.markov.word_pairs[pair] = parent.markov.word_pairs[pair].copy()
        
        self.creatures.append(offspring)
        self.total_births += 1
        
        # Parent pays reproduction cost
        parent.energy.consume_energy(200000.0)
        
        # New generation if population doubled
        if len(self.creatures) >= 2 * len([c for c in self.creatures if c.adam_distance == 0]):
            self.generation += 1
    
    def _synthesize_audio(self):
        """Synthesize audio from collective neural activity."""
        if not self.creatures:
            return
        
        # Pick a random creature's brain to sonify
        creature = np.random.choice(self.creatures)
        
        # Synthesize audio buffer
        audio_buffer = self.audio_synth.synthesize(creature.network)
        
        # Convert to pygame sound and play
        try:
            sound_array = np.clip(audio_buffer * 32767, -32768, 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(sound_array)
            
            if not self.audio_channel.get_busy():
                self.audio_channel.play(sound)
        except:
            pass  # Audio issues don't stop the simulation
    
    def _update_stats(self):
        """Update ecosystem statistics."""
        if not self.creatures:
            return
        
        self.stats['generation'] = self.generation
        self.stats['population'] = len(self.creatures)
        self.stats['births'] = self.total_births
        self.stats['deaths'] = self.total_deaths
        self.stats['avg_energy'] = np.mean([c.energy.energy for c in self.creatures])
        self.stats['avg_neurons'] = np.mean([len(c.network.neurons) for c in self.creatures])
        self.stats['avg_age'] = np.mean([c.age for c in self.creatures])
        
        # Calculate species count (clusters of similar creatures)
        similarities = []
        for i, c1 in enumerate(self.creatures):
            for c2 in self.creatures[i+1:]:
                sim = self.layout.calculate_similarity(c1, c2)
                if sim > 0.7:
                    similarities.append((c1, c2))
        
        # Rough species estimate (creatures with high similarity)
        self.stats['species_count'] = max(1, len(self.creatures) - len(similarities) // 2)
        
        # Circuit8 activity (how much color change)
        total_intensity = 0
        for y in range(self.circuit8.height):
            for x in range(self.circuit8.width):
                r, g, b = self.circuit8.read_pixel(x, y)
                total_intensity += (r + g + b) / 3.0
        self.stats['circuit8_activity'] = total_intensity / (self.circuit8.width * self.circuit8.height)
        
        self.stats['avg_fitness'] = self.stats['avg_energy']


class EcosystemVisualizer:
    """
    Pygame-based visualization of the complete ecosystem.
    
    Shows:
    - Circuit8 telepathic canvas
    - Creatures as colored circles (clustered by species)
    - Drug pills as small squares
    - Real-time statistics
    - Neural activity indicators
    """
    
    def __init__(self, ecosystem: EvolutionaryEcosystem):
        """Initialize visualizer."""
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required for visualization")
        
        self.ecosystem = ecosystem
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (ecosystem.world_width, ecosystem.world_height)
        )
        pygame.display.set_caption("CritterGOD - Evolutionary Ecosystem")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 18)
        self.font_large = pygame.font.Font(None, 24)
        self.running = True
        
        # Rendering parameters
        self.circuit8_scale = 8  # Scale Circuit8 pixels
        self.show_circuit8 = True
        self.show_stats = True
        self.show_connections = False
        
        # Performance tracking
        self.fps = 30
        self.timestep = 0
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_c:
                    self.show_circuit8 = not self.show_circuit8
                elif event.key == pygame.K_s:
                    self.show_stats = not self.show_stats
                elif event.key == pygame.K_n:
                    self.show_connections = not self.show_connections
                elif event.key == pygame.K_SPACE:
                    # Pause/unpause (toggle FPS)
                    self.fps = 0 if self.fps > 0 else 30
                elif event.key == pygame.K_d:
                    # Drop drug at random location
                    self.ecosystem._scatter_drugs(5)
    
    def render(self):
        """Render one frame."""
        # Clear screen
        self.screen.fill((5, 5, 15))  # Dark blue-black background
        
        # 1. Draw Circuit8 (the shared hallucination)
        if self.show_circuit8:
            self.draw_circuit8()
        
        # 2. Draw drug pills
        self.draw_drugs()
        
        # 3. Draw creatures (colored by genetic similarity)
        self.draw_creatures()
        
        # 4. Draw statistics overlay
        if self.show_stats:
            self.draw_stats()
        
        # 5. Draw controls help
        self.draw_controls()
        
        # Update display
        pygame.display.flip()
    
    def draw_circuit8(self):
        """Draw the telepathic canvas."""
        c8 = self.ecosystem.circuit8
        
        # Position in top-left
        offset_x = 10
        offset_y = 10
        
        for y in range(c8.height):
            for x in range(c8.width):
                r, g, b = c8.read_pixel(x, y)
                
                # Scale to reduce brightness (so it doesn't overwhelm)
                r = int(r * 0.7)
                g = int(g * 0.7)
                b = int(b * 0.7)
                
                pygame.draw.rect(
                    self.screen,
                    (r, g, b),
                    (
                        offset_x + x * self.circuit8_scale,
                        offset_y + y * self.circuit8_scale,
                        self.circuit8_scale,
                        self.circuit8_scale
                    )
                )
        
        # Border
        pygame.draw.rect(
            self.screen,
            (100, 100, 150),
            (offset_x - 2, offset_y - 2,
             c8.width * self.circuit8_scale + 4,
             c8.height * self.circuit8_scale + 4),
            2
        )
        
        # Label
        label = self.font.render("Circuit8 (Telepathic Canvas)", True, (150, 150, 200))
        self.screen.blit(label, (offset_x, offset_y + c8.height * self.circuit8_scale + 5))
    
    def draw_drugs(self):
        """Draw drug pills."""
        for pill in self.ecosystem.pills:
            # Color by molecule type
            colors = [
                (200, 100, 200),  # Inhibitory antagonist - magenta
                (150, 100, 250),  # Inhibitory agonist - purple
                (100, 200, 100),  # Excitatory antagonist - green
                (100, 250, 150),  # Excitatory agonist - cyan
                (255, 255, 100),  # Potentiator - yellow (ego death)
            ]
            color = colors[pill.molecule_type]
            
            pygame.draw.rect(
                self.screen,
                color,
                (int(pill.x) - 3, int(pill.y) - 3, 6, 6)
            )
    
    def draw_creatures(self):
        """Draw creatures as circles."""
        if not self.ecosystem.creatures:
            return
        
        for creature in self.ecosystem.creatures:
            # Color based on adam_distance (evolutionary depth)
            depth = min(creature.adam_distance, 10)
            hue = (depth * 36) % 360  # Cycle through hues
            
            # Convert HSV to RGB
            h = hue / 60.0
            c = 200  # Chroma
            x = c * (1 - abs(h % 2 - 1))
            
            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            # Add 55 to make colors brighter
            color = (int(r) + 55, int(g) + 55, int(b) + 55)
            
            # Size based on energy (but clamped)
            radius = int(5 + min(creature.energy.energy / 100000.0, 15))
            
            # Draw creature
            pygame.draw.circle(
                self.screen,
                color,
                (int(creature.x), int(creature.y)),
                radius
            )
            
            # Draw firing indicator (pulse when neurons active)
            activity = creature.network.get_activity_level()
            if activity > 0.1:
                pulse_radius = radius + int(activity * 10)
                pygame.draw.circle(
                    self.screen,
                    (255, 255, 100, 128),  # Yellow pulse
                    (int(creature.x), int(creature.y)),
                    pulse_radius,
                    1
                )
    
    def draw_stats(self):
        """Draw statistics overlay."""
        stats = self.ecosystem.stats
        
        # Background panel
        panel_x = self.ecosystem.world_width - 250
        panel_y = 10
        panel_width = 240
        panel_height = 220
        
        pygame.draw.rect(
            self.screen,
            (10, 10, 30, 200),
            (panel_x, panel_y, panel_width, panel_height)
        )
        pygame.draw.rect(
            self.screen,
            (80, 80, 120),
            (panel_x, panel_y, panel_width, panel_height),
            2
        )
        
        # Title
        title = self.font_large.render("ECOSYSTEM STATS", True, (200, 200, 255))
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Stats
        y_offset = panel_y + 40
        stat_lines = [
            f"Generation: {stats['generation']}",
            f"Population: {stats['population']}",
            f"Births: {stats['births']}",
            f"Deaths: {stats['deaths']}",
            f"Species: {stats['species_count']}",
            f"Avg Energy: {stats['avg_energy']:.0f}",
            f"Avg Neurons: {stats['avg_neurons']:.0f}",
            f"Avg Age: {stats['avg_age']:.0f}",
            f"C8 Activity: {stats['circuit8_activity']:.1f}",
            f"Timestep: {self.timestep}",
        ]
        
        for line in stat_lines:
            text = self.font.render(line, True, (180, 180, 220))
            self.screen.blit(text, (panel_x + 10, y_offset))
            y_offset += 18
    
    def draw_controls(self):
        """Draw controls help."""
        controls = [
            "C - Toggle Circuit8",
            "S - Toggle Stats",
            "D - Drop Drugs",
            "SPACE - Pause",
            "ESC - Quit"
        ]
        
        y_offset = self.ecosystem.world_height - 100
        for control in controls:
            text = self.font.render(control, True, (120, 120, 150))
            self.screen.blit(text, (10, y_offset))
            y_offset += 18
    
    def run(self):
        """Run visualization loop."""
        print("\n" + "=" * 70)
        print("  CRITTERGOD EVOLUTIONARY ECOSYSTEM")
        print("  The Grand Integration - All Systems Active")
        print("=" * 70)
        print("\nFeatures Active:")
        print("  ✓ Spiking neural networks with STDP plasticity")
        print("  ✓ Circuit8 telepathic shared hallucination")
        print("  ✓ Psychopharmacology (5 molecule types)")
        print("  ✓ Multi-modal creatures (vision, audio, text)")
        print("  ✓ Energy-driven evolution")
        print("  ✓ Force-directed species clustering")
        print("  ✓ Real-time audio synthesis")
        print("  ✓ Democratic collective voting")
        print("\nControls:")
        print("  C - Toggle Circuit8 visualization")
        print("  S - Toggle statistics")
        print("  D - Drop drugs into world")
        print("  SPACE - Pause simulation")
        print("  ESC - Quit")
        print("\nStarting simulation...")
        
        while self.running:
            # Handle input
            self.handle_events()
            
            # Update ecosystem
            if self.fps > 0:  # If not paused
                self.ecosystem.update(dt=1.0)
                self.timestep += 1
            
            # Render
            self.render()
            
            # Control frame rate
            if self.fps > 0:
                self.clock.tick(self.fps)
            else:
                self.clock.tick(10)  # Slow tick when paused
        
        pygame.quit()
        
        # Print final statistics
        print("\n" + "=" * 70)
        print("  SIMULATION COMPLETE")
        print("=" * 70)
        print(f"\nFinal Statistics:")
        print(f"  Total Generations: {self.ecosystem.stats['generation']}")
        print(f"  Total Births: {self.ecosystem.stats['births']}")
        print(f"  Total Deaths: {self.ecosystem.stats['deaths']}")
        print(f"  Final Population: {self.ecosystem.stats['population']}")
        print(f"  Species Count: {self.ecosystem.stats['species_count']}")
        print(f"  Timesteps: {self.timestep}")
        print("\nThe collective consciousness has spoken.\n")


def main():
    """Run the complete evolutionary ecosystem."""
    print("\n" + "=" * 70)
    print("  INITIALIZING CRITTERGOD EVOLUTIONARY ECOSYSTEM")
    print("=" * 70)
    
    if not PYGAME_AVAILABLE:
        print("\nERROR: pygame is required for the ecosystem visualization")
        print("Install with: pip install pygame")
        return
    
    # Create ecosystem
    ecosystem = EvolutionaryEcosystem(
        initial_population=15,
        max_population=40,
        world_width=1200,
        world_height=800,
        enable_audio=True,
        enable_drugs=True
    )
    
    # Create visualizer
    visualizer = EcosystemVisualizer(ecosystem)
    
    # Run
    visualizer.run()


if __name__ == "__main__":
    main()
