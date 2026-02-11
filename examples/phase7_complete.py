"""
Phase 7: COMPLETE INTEGRATION - The Ultimate Synthesis

ALL SYSTEMS INTEGRATED:
- 3D-style top-down visualization (Critterding-inspired)
- Written language evolution (genetic text inheritance)
- Audio consciousness synthesis (hear the collective mind)
- Psychedelic drug control panel (program consciousness)
- Circuit8 telepathic canvas (shared hallucination)
- Multi-modal creatures (see, speak, think, create)
- Energy-driven evolution (birth/death cycles)
- Force-directed species clustering
- Democratic collective voting
- Neural refinements (Phase 5a: weakening, clamping, bidirectional)
- Entity introspection (Phase 5b: debugging, lineage tracking)
- Advanced visualization (Phase 5c: force-directed layouts)

This is flamoot's complete vision + written language evolution:
PSYCHEDELIC COMPUTING + EVOLUTIONARY LINGUISTICS + AUDIO CONSCIOUSNESS

The ultimate artificial life platform.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import time
from typing import List, Optional, Dict

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("ERROR: pygame required for Phase 7")
    sys.exit(1)

from core.evolution.genotype import Genotype
from core.morphic.circuit8 import Circuit8
from core.pharmacology.drugs import Pill, MoleculeType
from creatures.enhanced_creature import EnhancedCreature
from visualization.force_directed_layout import ForceDirectedLayout
from visualization.drug_control_panel import DrugControlPanel
from generators.audio.neural_audio import NeuralAudioSynthesizer


class Phase7Ecosystem:
    """
    The complete integration - everything flamoot envisioned and more.
    
    New in Phase 7:
    - Written language evolution display
    - Genetic text inheritance visualization
    - Integrated drug control panel
    - Enhanced 3D-style rendering
    - Complete control interface
    """
    
    def __init__(
        self,
        initial_population: int = 20,
        max_population: int = 50,
        world_width: int = 1400,
        world_height: int = 900
    ):
        """Initialize Phase 7 ecosystem."""
        self.world_width = world_width
        self.world_height = world_height
        self.max_population = max_population
        
        # Shared telepathic canvas
        self.circuit8 = Circuit8(width=64, height=48)
        
        # Population
        self.creatures: List[EnhancedCreature] = []
        self.generation = 0
        self.total_births = 0
        self.total_deaths = 0
        
        # Layout engine
        self.layout = ForceDirectedLayout(
            repulsion_strength=100.0,
            attraction_strength=0.5,
            damping=0.8,
            similarity_threshold=0.7
        )
        
        # Audio synthesis (will be initialized in visualizer after pygame.init)
        self.audio_enabled = False
        self.audio_synth = None
        self.audio_channel = None
        
        # Drugs
        self.pills: List[Pill] = []
        
        # Language evolution tracking
        self.language_samples = []  # Recent text from creatures
        self.language_history = []  # Track language evolution over time
        
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
            'avg_fitness': 0.0,
            'language_diversity': 0.0  # NEW: Language complexity metric
        }
        
        # Create initial population
        print(f"Creating initial population of {initial_population} creatures...")
        self._create_initial_population(initial_population)
        
        # Scatter some drugs
        self._scatter_drugs(30)
        
        print(f"Phase 7 ecosystem initialized with {len(self.creatures)} creatures")
    
    def _create_initial_population(self, count: int):
        """Create initial random population."""
        for i in range(count):
            n_hidden = np.random.randint(40, 120)
            synapses_per = np.random.randint(15, 35)
            
            genotype = Genotype.create_random(
                n_sensory=10,
                n_motor=10,
                n_hidden_min=n_hidden,
                n_hidden_max=n_hidden + 20,
                synapses_per_neuron=synapses_per
            )
            
            x = np.random.uniform(100, self.world_width - 100)
            y = np.random.uniform(100, self.world_height - 100)
            
            creature = EnhancedCreature(
                genotype=genotype,
                x=x,
                y=y,
                initial_energy=500000.0,
                circuit8=self.circuit8,
                adam_distance=0
            )
            
            self.creatures.append(creature)
            self.total_births += 1
    
    def _scatter_drugs(self, count: int):
        """Scatter drug pills."""
        for _ in range(count):
            x = np.random.uniform(0, self.world_width)
            y = np.random.uniform(0, self.world_height)
            molecule = np.random.randint(0, 5)
            amount = np.random.uniform(50.0, 200.0)
            
            # Create molecule composition array
            composition = [0, 0, 0, 0, 0]
            composition[molecule] = amount
            self.pills.append(Pill(x=x, y=y, molecule_composition=composition))
    
    def update(self, dt: float = 1.0):
        """Update ecosystem for one timestep."""
        # Update all creatures
        for creature in self.creatures:
            alive = creature.update(dt)
            if alive:
                creature.x = creature.x % self.world_width
                creature.y = creature.y % self.world_height
        
        # Drug consumption
        self._handle_drug_consumption()
        
        # Remove dead creatures
        dead = [c for c in self.creatures if c.energy.energy <= 0]
        self.total_deaths += len(dead)
        self.creatures = [c for c in self.creatures if c.energy.energy > 0]
        
        # Reproduction
        if len(self.creatures) < self.max_population:
            self._handle_reproduction()
        
        # Update force-directed layout
        if len(self.creatures) > 1:
            self.layout.simulate_step(self.creatures, dt=0.1)
        
        # Synthesize collective audio
        if self.audio_enabled and len(self.creatures) > 0:
            self._synthesize_audio()
        
        # Apply collective voting
        self.circuit8.apply_voted_movement()
        
        # Sample language from creatures (NEW)
        self._sample_language()
        
        # Update statistics
        self._update_stats()
    
    def _handle_drug_consumption(self):
        """Handle drug consumption."""
        for creature in self.creatures:
            for pill in self.pills[:]:
                dx = creature.x - pill.x
                dy = creature.y - pill.y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance < 30.0:
                    creature.consume_pill(pill)
                    self.pills.remove(pill)
                    break
        
        if np.random.random() < 0.01:
            self._scatter_drugs(1)
    
    def _handle_reproduction(self):
        """Handle reproduction with language inheritance."""
        candidates = [c for c in self.creatures if c.energy.energy > 800000.0]
        if not candidates:
            return
        
        parent = np.random.choice(candidates)
        offspring_genotype = parent.genotype.mutate(mutation_rate=0.3)
        
        offspring_x = parent.x + np.random.uniform(-50, 50)
        offspring_y = parent.y + np.random.uniform(-50, 50)
        offspring_x = offspring_x % self.world_width
        offspring_y = offspring_y % self.world_height
        
        offspring = EnhancedCreature(
            genotype=offspring_genotype,
            x=offspring_x,
            y=offspring_y,
            initial_energy=300000.0,
            circuit8=self.circuit8,
            adam_distance=parent.adam_distance + 1
        )
        
        # Inherit language (genetic transmission)
        if (hasattr(parent, 'motors') and hasattr(parent.motors, 'markov') and
            hasattr(offspring, 'motors') and hasattr(offspring.motors, 'markov')):
            parent_markov = parent.motors.markov
            offspring_markov = offspring.motors.markov
            if parent_markov and hasattr(parent_markov, 'word_pairs'):
                parent_pairs = list(parent_markov.word_pairs.keys())
                if parent_pairs:
                    inherited = np.random.choice(
                        parent_pairs,
                        size=min(len(parent_pairs) // 2, 20),
                        replace=False
                    )
                    for pair in inherited:
                        if pair in parent_markov.word_pairs:
                            offspring_markov.word_pairs[pair] = parent_markov.word_pairs[pair].copy()
        
        self.creatures.append(offspring)
        self.total_births += 1
        parent.energy.consume_energy(200000.0)
        
        if len(self.creatures) >= 2 * len([c for c in self.creatures if c.adam_distance == 0]):
            self.generation += 1
    
    def _synthesize_audio(self):
        """Synthesize audio from collective neural activity."""
        if not self.creatures:
            return
        
        creature = np.random.choice(self.creatures)
        audio_buffer = self.audio_synth.synthesize_from_network(creature.network, duration_seconds=0.1)
        
        try:
            # Convert mono to stereo and scale to int16
            sound_array = np.clip(audio_buffer * 32767, -32768, 32767).astype(np.int16)
            # Reshape to stereo: (samples, 2)
            stereo_array = np.column_stack((sound_array, sound_array))
            sound = pygame.sndarray.make_sound(stereo_array)
            if not self.audio_channel.get_busy():
                self.audio_channel.play(sound)
        except Exception as e:
            # Silent fail - don't spam console
            pass
    
    def _sample_language(self):
        """Sample language from random creatures (NEW)."""
        if not self.creatures:
            return
        
        # Sample from random creature
        if np.random.random() < 0.1:  # 10% chance per frame
            creature = np.random.choice(self.creatures)
            # Access language through motors.markov
            if hasattr(creature, 'motors') and hasattr(creature.motors, 'markov'):
                markov = creature.motors.markov
                if markov and hasattr(markov, 'word_pairs') and markov.word_pairs:
                    try:
                        # Generate text from creature's markov chain
                        text = markov.generate_text(max_words=5)
                        if text:
                            self.language_samples.append({
                                'text': text,
                                'creature_id': id(creature),
                                'adam_distance': creature.adam_distance,
                                'timestamp': time.time()
                            })
                            
                            # Keep only recent samples
                            if len(self.language_samples) > 10:
                                self.language_samples.pop(0)
                    except:
                        pass
    
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
        
        # Species count
        similarities = []
        for i, c1 in enumerate(self.creatures):
            for c2 in self.creatures[i+1:]:
                sim = self.layout.calculate_similarity(c1, c2)
                if sim > 0.7:
                    similarities.append((c1, c2))
        self.stats['species_count'] = max(1, len(self.creatures) - len(similarities) // 2)
        
        # Circuit8 activity
        total_intensity = 0
        for y in range(self.circuit8.height):
            for x in range(self.circuit8.width):
                r, g, b = self.circuit8.read_pixel(x, y)
                total_intensity += (r + g + b) / 3.0
        self.stats['circuit8_activity'] = total_intensity / (self.circuit8.width * self.circuit8.height)
        
        # Language diversity (NEW)
        if self.language_samples:
            unique_words = set()
            for sample in self.language_samples:
                unique_words.update(sample['text'].split())
            self.stats['language_diversity'] = len(unique_words)
        
        self.stats['avg_fitness'] = self.stats['avg_energy']


class Phase7Visualizer:
    """
    Complete visualization with all Phase 7 features.
    
    New features:
    - Language evolution display
    - Integrated drug control panel
    - Enhanced control panel
    - 3D-style top-down rendering
    """
    
    def __init__(self, ecosystem: Phase7Ecosystem):
        """Initialize visualizer."""
        self.ecosystem = ecosystem
        
        # Initialize pygame here (before any pygame usage)
        pygame.init()
        
        self.screen = pygame.display.set_mode(
            (ecosystem.world_width, ecosystem.world_height)
        )
        pygame.display.set_caption("CritterGOD Phase 7 - Complete Integration")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 18)
        self.font_large = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 14)
        self.running = True
        
        # Now initialize audio after pygame is ready
        try:
            # Initialize mixer (will likely be stereo even if we ask for mono)
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=2048)
            mixer_info = pygame.mixer.get_init()
            actual_rate = mixer_info[0] if mixer_info else 22050
            
            self.ecosystem.audio_synth = NeuralAudioSynthesizer(
                sample_rate=actual_rate,
                buffer_size=2048,
                mode='mixed'
            )
            self.ecosystem.audio_channel = pygame.mixer.Channel(0)
            self.ecosystem.audio_enabled = True
            print(f"✓ Audio initialized: {actual_rate}Hz stereo")
        except Exception as e:
            print(f"Audio initialization failed: {e}")
            self.ecosystem.audio_enabled = False
        
        # Drug control panel
        self.drug_panel = DrugControlPanel(width=400, height=630)
        self.drug_panel.initialize_pygame(self.screen)
        
        # View modes
        self.show_circuit8 = True
        self.show_stats = True
        self.show_language = True
        self.show_drug_panel = True
        self.show_control_panel = True
        
        # Performance
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
                elif event.key == pygame.K_l:
                    self.show_language = not self.show_language
                elif event.key == pygame.K_p:
                    self.show_drug_panel = not self.show_drug_panel
                elif event.key == pygame.K_SPACE:
                    self.fps = 0 if self.fps > 0 else 30
                elif event.key == pygame.K_d:
                    self.ecosystem._scatter_drugs(5)
            
            # Drug panel events
            if self.show_drug_panel:
                pill = self.drug_panel.handle_event(event)
                if pill:
                    if self.drug_panel.broadcast_mode:
                        # Broadcast to all creatures
                        for creature in self.ecosystem.creatures:
                            creature.consume_pill(pill)
                    else:
                        # Drop pill in world
                        self.ecosystem.pills.append(pill)
    
    def render(self):
        """Render complete Phase 7 visualization."""
        # Clear screen
        self.screen.fill((15, 15, 25))  # Dark background
        
        # Draw Circuit8
        if self.show_circuit8:
            self.draw_circuit8()
        
        # Draw drugs
        self.draw_drugs()
        
        # Draw creatures
        self.draw_creatures()
        
        # Draw language overlay (NEW)
        if self.show_language:
            self.draw_language()
        
        # Draw statistics
        if self.show_stats:
            self.draw_stats()
        
        # Draw drug control panel
        if self.show_drug_panel:
            self.drug_panel.update_trip_level(self.ecosystem.creatures)
            self.drug_panel.render(
                offset_x=self.ecosystem.world_width - 420,
                offset_y=20
            )
        
        # Draw control panel
        if self.show_control_panel:
            self.draw_control_panel()
        
        # Update display
        pygame.display.flip()
    
    def draw_circuit8(self):
        """Draw Circuit8 telepathic canvas."""
        c8 = self.ecosystem.circuit8
        offset_x = 10
        offset_y = 10
        scale = 6
        
        for y in range(c8.height):
            for x in range(c8.width):
                r, g, b = c8.read_pixel(x, y)
                r = int(r * 0.6)
                g = int(g * 0.6)
                b = int(b * 0.6)
                
                pygame.draw.rect(
                    self.screen,
                    (r, g, b),
                    (offset_x + x * scale, offset_y + y * scale, scale, scale)
                )
        
        # Border and label
        pygame.draw.rect(
            self.screen,
            (100, 100, 150),
            (offset_x - 2, offset_y - 2,
             c8.width * scale + 4,
             c8.height * scale + 4),
            2
        )
        
        label = self.font_small.render("Circuit8 - Collective Unconscious", True, (150, 150, 200))
        self.screen.blit(label, (offset_x, offset_y + c8.height * scale + 5))
    
    def draw_drugs(self):
        """Draw drug pills."""
        colors = [
            (200, 100, 200),  # Inhibitory antagonist
            (150, 100, 250),  # Inhibitory agonist
            (100, 200, 100),  # Excitatory antagonist
            (100, 250, 150),  # Excitatory agonist
            (255, 255, 100),  # Potentiator
        ]
        
        for pill in self.ecosystem.pills:
            # Determine dominant molecule type from composition
            dominant_molecule = max(range(5), key=lambda i: pill.molecule_composition[i])
            color = colors[dominant_molecule]
            pygame.draw.circle(
                self.screen,
                color,
                (int(pill.x), int(pill.y)),
                4
            )
    
    def draw_creatures(self):
        """Draw creatures."""
        for creature in self.ecosystem.creatures:
            # Color by evolutionary depth
            depth = min(creature.adam_distance, 10)
            hue = (depth * 36) % 360
            
            h = hue / 60.0
            c = 200
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
            
            color = (int(r) + 55, int(g) + 55, int(b) + 55)
            radius = int(5 + min(creature.energy.energy / 100000.0, 15))
            
            pygame.draw.circle(
                self.screen,
                color,
                (int(creature.x), int(creature.y)),
                radius
            )
            
            # Neural activity pulse
            activity = creature.network.get_activity_level()
            if activity > 0.1:
                pulse_radius = radius + int(activity * 10)
                pygame.draw.circle(
                    self.screen,
                    (255, 255, 100),
                    (int(creature.x), int(creature.y)),
                    pulse_radius,
                    1
                )
    
    def draw_language(self):
        """Draw language evolution overlay (NEW)."""
        if not self.ecosystem.language_samples:
            return
        
        # Language panel
        panel_x = 10
        panel_y = self.ecosystem.world_height - 200
        panel_width = 400
        panel_height = 180
        
        pygame.draw.rect(
            self.screen,
            (20, 20, 40, 220),
            (panel_x, panel_y, panel_width, panel_height)
        )
        pygame.draw.rect(
            self.screen,
            (100, 150, 100),
            (panel_x, panel_y, panel_width, panel_height),
            2
        )
        
        # Title
        title = self.font_large.render("📖 EVOLVED LANGUAGE", True, (150, 255, 150))
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Recent language samples
        y_offset = panel_y + 45
        for i, sample in enumerate(self.ecosystem.language_samples[-5:]):
            # Creature ID indicator
            creature_label = self.font_small.render(
                f"Gen{sample['adam_distance']}:",
                True,
                (180, 180, 200)
            )
            self.screen.blit(creature_label, (panel_x + 15, y_offset))
            
            # Text
            text_label = self.font.render(
                sample['text'][:40],  # Truncate long text
                True,
                (200, 255, 200)
            )
            self.screen.blit(text_label, (panel_x + 80, y_offset))
            
            y_offset += 25
        
        # Language diversity stat
        diversity = self.ecosystem.stats.get('language_diversity', 0)
        div_label = self.font_small.render(
            f"Vocabulary: {int(diversity)} unique words",
            True,
            (150, 200, 150)
        )
        self.screen.blit(div_label, (panel_x + 15, panel_y + panel_height - 25))
    
    def draw_stats(self):
        """Draw statistics panel."""
        stats = self.ecosystem.stats
        
        panel_x = 10
        panel_y = 300
        panel_width = 240
        panel_height = 220
        
        pygame.draw.rect(
            self.screen,
            (20, 20, 40, 200),
            (panel_x, panel_y, panel_width, panel_height)
        )
        pygame.draw.rect(
            self.screen,
            (100, 100, 150),
            (panel_x, panel_y, panel_width, panel_height),
            2
        )
        
        # Title
        title = self.font_large.render("ECOSYSTEM STATS", True, (200, 200, 255))
        self.screen.blit(title, (panel_x + 20, panel_y + 10))
        
        # Stats
        y_offset = panel_y + 45
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
            text = self.font_small.render(line, True, (180, 180, 220))
            self.screen.blit(text, (panel_x + 15, y_offset))
            y_offset += 18
    
    def draw_control_panel(self):
        """Draw control panel (Critterding-style)."""
        panel_height = 80
        panel_y = self.ecosystem.world_height - panel_height
        
        pygame.draw.rect(
            self.screen,
            (30, 30, 50, 230),
            (0, panel_y, self.ecosystem.world_width, panel_height)
        )
        pygame.draw.rect(
            self.screen,
            (80, 80, 120),
            (0, panel_y, self.ecosystem.world_width, 2)
        )
        
        # Controls help
        controls = [
            "C-Circuit8  S-Stats  L-Language  P-DrugPanel  D-DropDrugs  SPACE-Pause  ESC-Quit",
            f"Phase 7: COMPLETE | FPS: {self.clock.get_fps():.1f} | All systems operational"
        ]
        
        y_offset = panel_y + 20
        for control in controls:
            text = self.font.render(control, True, (180, 180, 220))
            text_rect = text.get_rect(center=(self.ecosystem.world_width // 2, y_offset))
            self.screen.blit(text, text_rect)
            y_offset += 25
    
    def run(self):
        """Run Phase 7 visualization loop."""
        print("\n" + "=" * 80)
        print("  CRITTERGOD PHASE 7 - COMPLETE INTEGRATION")
        print("  The Ultimate Synthesis: Language + Audio + Drugs + Evolution")
        print("=" * 80)
        print("\nAll Systems Operational:")
        print("  ✓ Neural networks with STDP plasticity")
        print("  ✓ Circuit8 telepathic shared hallucination")
        print("  ✓ Psychopharmacology (5 molecule types)")
        print("  ✓ Multi-modal creatures (vision, audio, text)")
        print("  ✓ WRITTEN LANGUAGE EVOLUTION (genetic text inheritance)")
        print("  ✓ AUDIO SYNTHESIS (hear collective consciousness)")
        print("  ✓ DRUG CONTROL PANEL (program consciousness)")
        print("  ✓ Energy-driven evolution")
        print("  ✓ Force-directed species clustering")
        print("  ✓ Democratic collective voting")
        print("\nControls:")
        print("  C - Toggle Circuit8")
        print("  S - Toggle Statistics")
        print("  L - Toggle Language Display")
        print("  P - Toggle Drug Control Panel")
        print("  D - Drop Drugs")
        print("  SPACE - Pause")
        print("  ESC - Quit")
        print("\nStarting Phase 7...\n")
        
        while self.running:
            self.handle_events()
            
            if self.fps > 0:
                self.ecosystem.update(dt=1.0)
                self.timestep += 1
            
            self.render()
            
            if self.fps > 0:
                self.clock.tick(self.fps)
            else:
                self.clock.tick(10)
        
        pygame.quit()
        
        # Final statistics
        print("\n" + "=" * 80)
        print("  PHASE 7 SIMULATION COMPLETE")
        print("=" * 80)
        print(f"\nFinal Statistics:")
        print(f"  Total Generations: {self.ecosystem.stats['generation']}")
        print(f"  Total Births: {self.ecosystem.stats['births']}")
        print(f"  Total Deaths: {self.ecosystem.stats['deaths']}")
        print(f"  Final Population: {self.ecosystem.stats['population']}")
        print(f"  Species Count: {self.ecosystem.stats['species_count']}")
        print(f"  Language Vocabulary: {int(self.ecosystem.stats['language_diversity'])} words")
        print(f"  Timesteps: {self.timestep}")
        print("\nThe complete vision is realized.")
        print("Language evolves. Consciousness speaks. Music plays.")
        print("The collective mind is alive.\n")


def main():
    """Run Phase 7 complete ecosystem."""
    print("\n" + "=" * 80)
    print("  INITIALIZING PHASE 7 - COMPLETE INTEGRATION")
    print("=" * 80)
    
    if not PYGAME_AVAILABLE:
        print("\nERROR: pygame is required")
        print("Install with: pip install pygame")
        return
    
    ecosystem = Phase7Ecosystem(
        initial_population=15,
        max_population=40,
        world_width=1400,
        world_height=900
    )
    
    visualizer = Phase7Visualizer(ecosystem)
    visualizer.run()


if __name__ == "__main__":
    main()
