"""
Drug Control Panel - Psychedelic Interface

Interactive interface for administering psychoactive molecules to the collective consciousness.

Inspired by psychedelic research and flamoot's vision of programmable consciousness:
- 5 molecule types affecting neural dynamics
- Visual feedback of drug effects on Circuit8
- Real-time monitoring of collective consciousness state
- Experimental controls for "ego death" and consciousness modification

This is the interface for the mad scientist experimenting with artificial life consciousness.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import Optional, Dict, List, Tuple

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("pygame not available - drug interface disabled")

from core.pharmacology.drugs import Pill, MoleculeType


class DrugControlPanel:
    """
    Interactive drug administration interface.
    
    Features:
    - Visual molecule selection
    - Dosage control (slider)
    - Targeted vs broadcast administration
    - Real-time effect monitoring
    - Circuit8 activity visualization
    - "Trip level" meter
    - Emergency "ego death" button (potentiator flood)
    """
    
    def __init__(self, width: int = 400, height: int = 600):
        """
        Initialize drug control panel.
        
        Args:
            width: Panel width in pixels
            height: Panel height in pixels
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required for drug panel")
        
        self.width = width
        self.height = height
        
        # Panel state
        self.selected_molecule = 0  # Default to Inhibitory Antagonist
        self.dosage = 100.0  # Default dose
        self.broadcast_mode = False  # False = drop pill, True = dose all creatures
        
        # Molecule info
        self.molecules = [
            {
                'name': 'Inhibitory Antagonist',
                'type': MoleculeType.INHIBITORY_ANTAGONIST,
                'color': (200, 100, 200),
                'description': 'Blocks inhibitory neurons',
                'effect': 'Disinhibition, increased activity'
            },
            {
                'name': 'Inhibitory Agonist',
                'type': MoleculeType.INHIBITORY_AGONIST,
                'color': (150, 100, 250),
                'description': 'Enhances inhibitory neurons',
                'effect': 'Suppression, decreased activity'
            },
            {
                'name': 'Excitatory Antagonist',
                'type': MoleculeType.EXCITATORY_ANTAGONIST,
                'color': (100, 200, 100),
                'description': 'Blocks excitatory neurons',
                'effect': 'Sedation, reduced cognition'
            },
            {
                'name': 'Excitatory Agonist',
                'type': MoleculeType.EXCITATORY_AGONIST,
                'color': (100, 250, 150),
                'description': 'Enhances excitatory neurons',
                'effect': 'Stimulation, hyperactivity'
            },
            {
                'name': '⚠ POTENTIATOR ⚠',
                'type': MoleculeType.POTENTIATOR,
                'color': (255, 255, 100),
                'description': 'AMPLIFIES ALL EFFECTS 10×',
                'effect': '*** EGO DEATH ***'
            }
        ]
        
        # UI elements (will be defined in render)
        self.molecule_buttons = []
        self.dosage_slider = None
        self.broadcast_button = None
        self.administer_button = None
        self.ego_death_button = None
        
        # Panel offset (set during render)
        self.offset_x = 0
        self.offset_y = 0
        
        # Fonts
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        
        # Trip meter state
        self.trip_level = 0.0
        self.trip_history = [0.0] * 100  # Last 100 measurements
    
    def initialize_pygame(self, screen):
        """
        Initialize pygame-specific elements.
        
        Args:
            screen: Pygame screen surface
        """
        self.screen = screen
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Define UI element positions
        button_y = 120
        button_height = 60
        button_spacing = 70
        
        for i in range(5):
            rect = pygame.Rect(20, button_y + i * button_spacing, self.width - 40, button_height)
            self.molecule_buttons.append(rect)
        
        # Dosage slider
        self.dosage_slider = pygame.Rect(20, 550, self.width - 40, 20)
        
        # Mode toggle
        self.broadcast_button = pygame.Rect(20, 480, 180, 40)
        
        # Administer button
        self.administer_button = pygame.Rect(210, 480, 170, 40)
        
        # Emergency ego death button
        self.ego_death_button = pygame.Rect(20, 530, self.width - 40, 50)
    
    def handle_event(self, event) -> Optional[Pill]:
        """
        Handle pygame event.
        
        Args:
            event: Pygame event
            
        Returns:
            Pill object if drug should be administered, None otherwise
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            
            # Check molecule selection (with offset)
            for i, rect in enumerate(self.molecule_buttons):
                adjusted_rect = rect.move(self.offset_x, self.offset_y)
                if adjusted_rect.collidepoint(mouse_pos):
                    self.selected_molecule = i
                    print(f"Selected molecule: {self.molecules[i]['name']}")
                    return None
            
            # Check broadcast toggle (with offset)
            if self.broadcast_button:
                adjusted_rect = self.broadcast_button.move(self.offset_x, self.offset_y)
                if adjusted_rect.collidepoint(mouse_pos):
                    self.broadcast_mode = not self.broadcast_mode
                    return None
            
            # Check administer button (with offset)
            if self.administer_button:
                adjusted_rect = self.administer_button.move(self.offset_x, self.offset_y)
                if adjusted_rect.collidepoint(mouse_pos):
                    print(f"DOSE button clicked! Mode: {'BROADCAST' if self.broadcast_mode else 'TARGETED'}, Molecule: {self.selected_molecule}, Dosage: {self.dosage}")
                    return self._create_pill()
            
            # Check ego death button (with offset)
            if self.ego_death_button:
                adjusted_rect = self.ego_death_button.move(self.offset_x, self.offset_y)
                if adjusted_rect.collidepoint(mouse_pos):
                    print("⚠ EGO DEATH button clicked! Potentiator flood initiated!")
                    return self._create_ego_death_dose()
            
            # Check dosage slider (with offset)
            if self.dosage_slider:
                adjusted_rect = self.dosage_slider.move(self.offset_x, self.offset_y)
                if adjusted_rect.collidepoint(mouse_pos):
                    # Update dosage based on position
                    relative_x = mouse_pos[0] - adjusted_rect.x
                    self.dosage = (relative_x / adjusted_rect.width) * 500.0
                    self.dosage = max(10.0, min(500.0, self.dosage))
                    return None
        
        return None
    
    def _create_pill(self) -> Pill:
        """Create pill with current settings."""
        composition = [0, 0, 0, 0, 0]
        composition[self.selected_molecule] = self.dosage
        return Pill(
            x=np.random.uniform(100, 1100),  # Random position
            y=np.random.uniform(100, 700),
            molecule_composition=composition
        )
    
    def _create_ego_death_dose(self) -> Pill:
        """Create maximum potentiator dose for ego death."""
        composition = [0, 0, 0, 0, 0]
        composition[MoleculeType.POTENTIATOR] = 500.0  # Maximum dose
        return Pill(
            x=np.random.uniform(100, 1100),
            y=np.random.uniform(100, 700),
            molecule_composition=composition
        )
    
    def update_trip_level(self, creatures):
        """
        Calculate current "trip level" based on drug concentrations in population.
        
        Args:
            creatures: List of creatures to analyze
        """
        if not creatures:
            self.trip_level = 0.0
            return
        
        # Average drug concentration across all creatures
        total_drugs = 0.0
        for creature in creatures:
            if hasattr(creature, 'drugs'):
                # Sum all drug levels, weighted by molecule type
                for i, level in enumerate(creature.drugs.tripping):
                    if i == MoleculeType.POTENTIATOR:
                        total_drugs += level * 10.0  # Potentiator counts 10×
                    else:
                        total_drugs += level
        
        # Normalize to 0-1 scale
        self.trip_level = min(1.0, total_drugs / (len(creatures) * 100.0))
        
        # Update history
        self.trip_history.append(self.trip_level)
        if len(self.trip_history) > 100:
            self.trip_history.pop(0)
    
    def render(self, offset_x: int = 0, offset_y: int = 0):
        """
        Render the drug control panel.
        
        Args:
            offset_x: X offset for positioning
            offset_y: Y offset for positioning
        """
        # Store offset for event handling
        self.offset_x = offset_x
        self.offset_y = offset_y
        
        # Background panel
        panel_rect = pygame.Rect(offset_x, offset_y, self.width, self.height)
        pygame.draw.rect(self.screen, (20, 20, 40), panel_rect)
        pygame.draw.rect(self.screen, (100, 100, 150), panel_rect, 3)
        
        # Title
        title = self.font_large.render("⚗ DRUG CONTROL ⚗", True, (200, 200, 255))
        self.screen.blit(title, (offset_x + 50, offset_y + 10))
        
        subtitle = self.font_small.render("Psychedelic Consciousness Interface", True, (150, 150, 200))
        self.screen.blit(subtitle, (offset_x + 70, offset_y + 45))
        
        # Separator
        pygame.draw.line(
            self.screen,
            (100, 100, 150),
            (offset_x + 20, offset_y + 70),
            (offset_x + self.width - 20, offset_y + 70),
            2
        )
        
        # Molecule selection buttons
        for i, (button, molecule) in enumerate(zip(self.molecule_buttons, self.molecules)):
            adjusted_rect = button.move(offset_x, offset_y)
            
            # Highlight selected
            if i == self.selected_molecule:
                pygame.draw.rect(self.screen, molecule['color'], adjusted_rect)
                pygame.draw.rect(self.screen, (255, 255, 255), adjusted_rect, 3)
            else:
                # Dimmed when not selected
                color = tuple(c // 2 for c in molecule['color'])
                pygame.draw.rect(self.screen, color, adjusted_rect)
                pygame.draw.rect(self.screen, (80, 80, 120), adjusted_rect, 2)
            
            # Label
            label = self.font_medium.render(molecule['name'], True, (255, 255, 255))
            label_rect = label.get_rect(center=(adjusted_rect.centerx, adjusted_rect.y + 15))
            self.screen.blit(label, label_rect)
            
            # Effect
            effect = self.font_small.render(molecule['effect'], True, (200, 200, 200))
            effect_rect = effect.get_rect(center=(adjusted_rect.centerx, adjusted_rect.y + 40))
            self.screen.blit(effect, effect_rect)
        
        # Separator
        pygame.draw.line(
            self.screen,
            (100, 100, 150),
            (offset_x + 20, offset_y + 460),
            (offset_x + self.width - 20, offset_y + 460),
            2
        )
        
        # Mode toggle button
        if self.broadcast_button:
            adjusted_rect = self.broadcast_button.move(offset_x, offset_y)
            
            if self.broadcast_mode:
                color = (255, 100, 100)  # Red for broadcast
                text = "BROADCAST"
            else:
                color = (100, 100, 255)  # Blue for targeted
                text = "TARGETED"
            
            pygame.draw.rect(self.screen, color, adjusted_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), adjusted_rect, 2)
            
            label = self.font_small.render(text, True, (255, 255, 255))
            label_rect = label.get_rect(center=adjusted_rect.center)
            self.screen.blit(label, label_rect)
        
        # Administer button
        if self.administer_button:
            adjusted_rect = self.administer_button.move(offset_x, offset_y)
            
            pygame.draw.rect(self.screen, (100, 255, 100), adjusted_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), adjusted_rect, 2)
            
            label = self.font_medium.render("DOSE", True, (0, 0, 0))
            label_rect = label.get_rect(center=adjusted_rect.center)
            self.screen.blit(label, label_rect)
        
        # Dosage label
        dosage_label = self.font_small.render(f"Dosage: {self.dosage:.0f} units", True, (200, 200, 200))
        self.screen.blit(dosage_label, (offset_x + 20, offset_y + 505))
        
        # Dosage slider
        if self.dosage_slider:
            adjusted_rect = self.dosage_slider.move(offset_x, offset_y)
            
            # Slider track
            pygame.draw.rect(self.screen, (60, 60, 80), adjusted_rect)
            pygame.draw.rect(self.screen, (100, 100, 150), adjusted_rect, 1)
            
            # Slider handle
            handle_x = adjusted_rect.x + int((self.dosage / 500.0) * adjusted_rect.width)
            handle_rect = pygame.Rect(handle_x - 5, adjusted_rect.y - 5, 10, 30)
            pygame.draw.rect(self.screen, (200, 200, 255), handle_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), handle_rect, 2)
        
        # Separator
        pygame.draw.line(
            self.screen,
            (100, 100, 150),
            (offset_x + 20, offset_y + 520),
            (offset_x + self.width - 20, offset_y + 520),
            2
        )
        
        # Emergency ego death button
        if self.ego_death_button:
            adjusted_rect = self.ego_death_button.move(offset_x, offset_y)
            
            # Pulsing effect for danger
            pulse = int(127 + 127 * np.sin(pygame.time.get_ticks() * 0.005))
            color = (255, pulse, 0)
            
            pygame.draw.rect(self.screen, color, adjusted_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), adjusted_rect, 3)
            
            label1 = self.font_medium.render("⚠ EGO DEATH ⚠", True, (0, 0, 0))
            label1_rect = label1.get_rect(center=(adjusted_rect.centerx, adjusted_rect.centery - 8))
            self.screen.blit(label1, label1_rect)
            
            label2 = self.font_small.render("(Potentiator Flood)", True, (0, 0, 0))
            label2_rect = label2.get_rect(center=(adjusted_rect.centerx, adjusted_rect.centery + 12))
            self.screen.blit(label2, label2_rect)
        
        # Trip level meter
        self.render_trip_meter(offset_x + 20, offset_y + 590)
    
    def render_trip_meter(self, x: int, y: int):
        """
        Render the collective trip level meter.
        
        Args:
            x: X position
            y: Y position
        """
        meter_width = self.width - 40
        meter_height = 30
        
        # Background
        meter_rect = pygame.Rect(x, y, meter_width, meter_height)
        pygame.draw.rect(self.screen, (40, 40, 60), meter_rect)
        pygame.draw.rect(self.screen, (100, 100, 150), meter_rect, 2)
        
        # Trip level bar (color changes with level)
        if self.trip_level < 0.3:
            color = (100, 255, 100)  # Green - mild
        elif self.trip_level < 0.7:
            color = (255, 255, 100)  # Yellow - moderate
        else:
            color = (255, 100, 100)  # Red - intense
        
        fill_width = int(meter_width * self.trip_level)
        if fill_width > 0:
            fill_rect = pygame.Rect(x, y, fill_width, meter_height)
            pygame.draw.rect(self.screen, color, fill_rect)
        
        # Label
        label = self.font_small.render(f"Collective Trip Level: {self.trip_level:.1%}", True, (255, 255, 255))
        label_rect = label.get_rect(center=(x + meter_width // 2, y + meter_height // 2))
        self.screen.blit(label, label_rect)


def integrate_drug_panel_with_ecosystem(ecosystem, visualizer):
    """
    Helper function to integrate drug panel with ecosystem visualization.
    
    Args:
        ecosystem: EvolutionaryEcosystem instance
        visualizer: EcosystemVisualizer instance
    
    Returns:
        DrugControlPanel instance
    """
    panel = DrugControlPanel(width=400, height=630)
    panel.initialize_pygame(visualizer.screen)
    
    # Position panel on right side
    panel_x = visualizer.ecosystem.world_width - 420
    panel_y = visualizer.ecosystem.world_height - 650
    
    def update_and_render():
        """Update trip level and render panel."""
        panel.update_trip_level(ecosystem.creatures)
        panel.render(offset_x=panel_x, offset_y=panel_y)
    
    return panel, update_and_render


if __name__ == "__main__":
    """Standalone demo of drug control panel."""
    print("Drug Control Panel - Standalone Demo")
    print("=" * 50)
    
    if not PYGAME_AVAILABLE:
        print("ERROR: pygame required")
        exit(1)
    
    pygame.init()
    screen = pygame.display.set_mode((500, 700))
    pygame.display.set_caption("Drug Control Panel Demo")
    
    panel = DrugControlPanel()
    panel.initialize_pygame(screen)
    
    clock = pygame.time.Clock()
    running = True
    
    print("\nControls:")
    print("- Click molecule buttons to select drug type")
    print("- Drag dosage slider to set amount")
    print("- Click TARGETED/BROADCAST to toggle mode")
    print("- Click DOSE to administer selected drug")
    print("- Click EGO DEATH for potentiator flood")
    print("- ESC to quit")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            
            # Handle panel events
            pill = panel.handle_event(event)
            if pill:
                print(f"Administered: {panel.molecules[pill.molecule_type]['name']}, "
                      f"Dose: {pill.amount:.1f}, Mode: {'BROADCAST' if panel.broadcast_mode else 'TARGETED'}")
        
        # Simulate trip level changes
        panel.trip_level = 0.3 + 0.2 * np.sin(pygame.time.get_ticks() * 0.001)
        panel.trip_history.append(panel.trip_level)
        if len(panel.trip_history) > 100:
            panel.trip_history.pop(0)
        
        # Render
        screen.fill((10, 10, 20))
        panel.render(offset_x=50, offset_y=20)
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()
    print("\nDemo complete.")
