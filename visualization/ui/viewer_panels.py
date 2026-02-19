"""
Viewer Panels - Embeddable Mini-Viewers for Research Platform

Provides compact, embeddable versions of:
- Circuit8 canvas viewer (telepathic field visualization)
- Neural network viewer (selected creature's brain)
- Drug control panel (interactive drug administration)

These panels can be toggled on/off and positioned within the main UI.
"""

import numpy as np
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from typing import Optional, List, Tuple, Dict, Any


def _get_font(size: int):
    """Get a pygame font, with fallback for Python 3.14 compatibility issues."""
    try:
        return pygame.font.Font(None, size)
    except (NotImplementedError, ImportError, Exception):
        # Return a DummyFont that renders colored rectangles instead of text
        class DummyFont:
            def __init__(self, sz):
                self._size = sz
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
        return DummyFont(size)


class Circuit8MiniPanel:
    """
    Compact Circuit8 viewer panel.
    
    Shows the telepathic canvas with:
    - Color-coded pixel values
    - Creature positions as dots
    - Field type indicators (food trails, danger zones)
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int = 256,
        height: int = 192,
        scale: int = 4
    ):
        """
        Initialize Circuit8 mini panel.
        
        Args:
            x, y: Panel position
            width, height: Panel dimensions
            scale: Pixel scale factor
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.scale = scale
        self.visible = True
        
        # Lazy initialization (created on first render)
        self.surface = None
        self.font = None
        self.title_font = None
        
        # Field type colors for legend
        self.field_colors = {
            'FOOD_TRAIL': (0, 200, 0),
            'DANGER_ZONE': (255, 0, 0),
            'MATING_SIGNAL': (200, 0, 200),
            'TERRITORY': (0, 0, 200),
            'EXCITEMENT': (200, 200, 0),
            'CALM': (0, 200, 200),
        }
    
    def render(self, screen, circuit8, creatures=None):
        """
        Render the Circuit8 mini panel.
        
        Args:
            screen: Pygame screen surface
            circuit8: Circuit8 instance to visualize
            creatures: Optional list of creatures to show positions
        """
        if not self.visible:
            return
        
        # Lazy init
        if self.surface is None:
            self.surface = pygame.Surface((self.width, self.height))
            self.font = _get_font(16)
            self.title_font = _get_font(20)
        
        # Clear surface
        self.surface.fill((20, 20, 30))
        
        # Draw border
        pygame.draw.rect(self.surface, (80, 80, 120), (0, 0, self.width, self.height), 2)
        
        # Title
        title = self.title_font.render("CIRCUIT8 - Telepathic Canvas", True, (200, 200, 255))
        self.surface.blit(title, (10, 5))
        
        # Canvas area
        canvas_y = 25
        canvas_width = min(self.width - 20, circuit8.width * self.scale)
        canvas_height = min(self.height - 50, circuit8.height * self.scale)
        
        # Draw Circuit8 pixels
        for cy in range(circuit8.height):
            for cx in range(circuit8.width):
                r, g, b = circuit8.read_pixel(cx, cy)
                color = (int(r), int(g), int(b))
                
                px = 10 + cx * self.scale
                py = canvas_y + cy * self.scale
                
                if px < self.width - 10 and py < self.height - 25:
                    pygame.draw.rect(
                        self.surface,
                        color,
                        (px, py, self.scale, self.scale)
                    )
        
        # Draw creature positions
        if creatures:
            for creature in creatures:
                # Map creature position to canvas
                cx = int(creature.x) % circuit8.width
                cy = int(creature.y) % circuit8.height
                
                px = 10 + cx * self.scale + self.scale // 2
                py = canvas_y + cy * self.scale + self.scale // 2
                
                if px < self.width - 10 and py < self.height - 25:
                    # Color based on energy
                    energy_frac = creature.energy.get_energy_fraction()
                    dot_color = (
                        int(255 * (1 - energy_frac)),
                        int(255 * energy_frac),
                        100
                    )
                    pygame.draw.circle(self.surface, dot_color, (px, py), 3)
                    pygame.draw.circle(self.surface, (255, 255, 255), (px, py), 3, 1)
        
        # Info line at bottom
        n_creatures = len(creatures) if creatures else 0
        info = self.font.render(f"Creatures: {n_creatures} | Depth: {circuit8.depth}", True, (150, 150, 150))
        self.surface.blit(info, (10, self.height - 20))
        
        # Blit to screen
        screen.blit(self.surface, (self.x, self.y))
    
    def handle_event(self, event) -> bool:
        """Handle events. Returns True if event was consumed."""
        return False


class NeuralNetworkMiniPanel:
    """
    Compact neural network viewer panel.
    
    Shows a creature's brain with:
    - Neurons as colored dots (by type)
    - Synapses as connecting lines
    - Firing neurons highlighted
    - Activity statistics
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int = 300,
        height: int = 250
    ):
        """Initialize neural network mini panel."""
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.visible = True
        
        # Lazy initialization
        self.surface = None
        self.font = None
        self.title_font = None
        
        # Layout cache
        self._layout_cache = {}
        self._last_network_id = None
        
        # Neuron type colors
        self.neuron_colors = {
            0: (100, 255, 100),  # Regular - green
            1: (100, 100, 255),  # Sensory - blue
            2: (255, 100, 100),  # Motor - red
            3: (255, 100, 255),  # Inhibitory - magenta
        }
    
    def _compute_layout(self, network):
        """Compute force-directed layout for neurons."""
        n = len(network.neurons)
        if n == 0:
            return {}
        
        # Simple circular layout with type clustering
        positions = {}
        
        # Group by type
        sensory = [neu for neu in network.neurons if neu.neuron_type.value == 1]
        motor = [neu for neu in network.neurons if neu.neuron_type.value == 2]
        hidden = [neu for neu in network.neurons if neu.neuron_type.value in (0, 3)]
        
        canvas_x = 30
        canvas_y = 45
        canvas_w = self.width - 60
        canvas_h = self.height - 80
        
        # Position sensory on left
        for i, neu in enumerate(sensory):
            y_frac = (i + 0.5) / max(len(sensory), 1)
            positions[neu.neuron_id] = (
                canvas_x + 20,
                canvas_y + y_frac * canvas_h
            )
        
        # Position motor on right
        for i, neu in enumerate(motor):
            y_frac = (i + 0.5) / max(len(motor), 1)
            positions[neu.neuron_id] = (
                canvas_x + canvas_w - 20,
                canvas_y + y_frac * canvas_h
            )
        
        # Position hidden in middle (grid)
        n_hidden = len(hidden)
        cols = max(1, int(np.sqrt(n_hidden)))
        rows = max(1, (n_hidden + cols - 1) // cols)
        
        for i, neu in enumerate(hidden):
            col = i % cols
            row = i // cols
            x_frac = (col + 0.5) / cols
            y_frac = (row + 0.5) / rows
            positions[neu.neuron_id] = (
                canvas_x + 40 + x_frac * (canvas_w - 80),
                canvas_y + 10 + y_frac * (canvas_h - 20)
            )
        
        return positions
    
    def render(self, screen, creature):
        """
        Render the neural network mini panel.
        
        Args:
            screen: Pygame screen surface
            creature: Creature whose network to visualize
        """
        if not self.visible or creature is None:
            return
        
        # Lazy init
        if self.surface is None:
            self.surface = pygame.Surface((self.width, self.height))
            self.font = _get_font(16)
            self.title_font = _get_font(20)
        
        network = creature.network
        
        # Clear surface
        self.surface.fill((20, 25, 35))
        pygame.draw.rect(self.surface, (80, 80, 120), (0, 0, self.width, self.height), 2)
        
        # Title
        title = self.title_font.render(f"BRAIN - Creature {creature.creature_id}", True, (200, 200, 255))
        self.surface.blit(title, (10, 5))
        
        # Compute or use cached layout
        net_id = id(network)
        if net_id != self._last_network_id:
            self._layout_cache = self._compute_layout(network)
            self._last_network_id = net_id
        
        positions = self._layout_cache
        
        # Draw synapses first (behind neurons)
        for synapse in network.synapses[:500]:  # Limit for performance
            pre_id = synapse.pre_neuron.neuron_id
            post_id = synapse.post_neuron.neuron_id
            
            if pre_id not in positions or post_id not in positions:
                continue
            
            pos1 = positions[pre_id]
            pos2 = positions[post_id]
            
            # Color based on weight
            if synapse.is_inhibitory:
                intensity = min(150, int(abs(synapse.weight) * 30))
                color = (intensity, 0, intensity // 2, 100)
            else:
                intensity = min(150, int(abs(synapse.weight) * 30))
                color = (0, intensity, intensity // 2, 100)
            
            pygame.draw.line(self.surface, color[:3], pos1, pos2, 1)
        
        # Draw neurons
        for neuron in network.neurons:
            pos = positions.get(neuron.neuron_id)
            if pos is None:
                continue
            
            # Base color by type
            base_color = self.neuron_colors.get(neuron.neuron_type.value, (150, 150, 150))
            
            # Brighten if fired
            if neuron.fired_last_step():
                color = (255, 255, 100)
                radius = 6
            else:
                color = base_color
                radius = 4
            
            pygame.draw.circle(self.surface, color, (int(pos[0]), int(pos[1])), radius)
        
        # Stats at bottom
        activity = network.get_activity_level()
        stats = f"Neurons: {len(network.neurons)} | Synapses: {len(network.synapses)} | Activity: {activity*100:.0f}%"
        stats_surf = self.font.render(stats, True, (150, 150, 150))
        self.surface.blit(stats_surf, (10, self.height - 20))
        
        # Legend
        legend_y = 25
        for i, (name, color) in enumerate([("Sensory", (100, 100, 255)), ("Motor", (255, 100, 100)), 
                                            ("Hidden", (100, 255, 100)), ("Inhib", (255, 100, 255))]):
            pygame.draw.circle(self.surface, color, (self.width - 80 + i * 20, legend_y), 5)
        
        # Blit to screen
        screen.blit(self.surface, (self.x, self.y))
    
    def handle_event(self, event) -> bool:
        """Handle events."""
        return False


class DrugControlMiniPanel:
    """
    Compact drug control panel.
    
    Provides:
    - Molecule type selection
    - Dosage control
    - Administer buttons
    - Trip level meter
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int = 280,
        height: int = 350
    ):
        """Initialize drug control mini panel."""
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.visible = True
        
        # Lazy initialization
        self.surface = None
        self.font = None
        self.title_font = None
        
        # Drug state
        self.selected_molecule = 0
        self.dosage = 50.0
        self.trip_level = 0.0
        
        # Molecule info
        self.molecules = [
            {'name': 'Inh. Antagonist', 'color': (200, 100, 200), 'effect': 'Disinhibition'},
            {'name': 'Inh. Agonist', 'color': (150, 100, 250), 'effect': 'Suppression'},
            {'name': 'Exc. Antagonist', 'color': (100, 200, 100), 'effect': 'Sedation'},
            {'name': 'Exc. Agonist', 'color': (100, 250, 150), 'effect': 'Stimulation'},
            {'name': 'POTENTIATOR', 'color': (255, 255, 100), 'effect': '10× Amplify'},
        ]
        
        # Button rects (relative to panel)
        self.molecule_buttons = []
        btn_y = 30
        for i in range(5):
            self.molecule_buttons.append(pygame.Rect(10, btn_y + i * 45, width - 20, 40))
        
        self.dose_button = pygame.Rect(10, 260, width // 2 - 15, 35)
        self.dose_all_button = pygame.Rect(width // 2 + 5, 260, width // 2 - 15, 35)
        self.ego_death_button = pygame.Rect(10, 300, width - 20, 40)
        
        # Callback for drug administration
        self.on_dose_creature = None  # Called with (molecule_type, dosage)
        self.on_dose_all = None       # Called with (molecule_type, dosage)
    
    def update_trip_level(self, creatures):
        """Calculate collective trip level."""
        if not creatures:
            self.trip_level = 0.0
            return
        
        total = 0.0
        for c in creatures:
            if hasattr(c, 'drugs'):
                for i, level in enumerate(c.drugs.tripping):
                    if i == 4:  # Potentiator
                        total += level * 10.0
                    else:
                        total += level
        
        self.trip_level = min(1.0, total / (len(creatures) * 100.0))
    
    def render(self, screen, creatures=None):
        """Render the drug control panel."""
        if not self.visible:
            return
        
        # Lazy init
        if self.surface is None:
            self.surface = pygame.Surface((self.width, self.height))
            self.font = _get_font(16)
            self.title_font = _get_font(20)
        
        # Update trip level
        if creatures:
            self.update_trip_level(creatures)
        
        # Clear surface
        self.surface.fill((25, 20, 35))
        pygame.draw.rect(self.surface, (100, 80, 120), (0, 0, self.width, self.height), 2)
        
        # Title
        title = self.title_font.render("⚗ DRUG CONTROL", True, (200, 200, 255))
        self.surface.blit(title, (10, 5))
        
        # Molecule buttons
        for i, (btn, mol) in enumerate(zip(self.molecule_buttons, self.molecules)):
            # Selected highlight
            if i == self.selected_molecule:
                pygame.draw.rect(self.surface, mol['color'], btn)
                pygame.draw.rect(self.surface, (255, 255, 255), btn, 2)
            else:
                dim_color = tuple(c // 2 for c in mol['color'])
                pygame.draw.rect(self.surface, dim_color, btn)
                pygame.draw.rect(self.surface, (60, 60, 80), btn, 1)
            
            # Label
            label = self.font.render(mol['name'], True, (255, 255, 255))
            self.surface.blit(label, (btn.x + 5, btn.y + 5))
            effect = self.font.render(mol['effect'], True, (200, 200, 200))
            self.surface.blit(effect, (btn.x + 5, btn.y + 22))
        
        # Dose buttons
        pygame.draw.rect(self.surface, (100, 200, 100), self.dose_button)
        pygame.draw.rect(self.surface, (255, 255, 255), self.dose_button, 1)
        label = self.font.render("DOSE ONE", True, (0, 0, 0))
        self.surface.blit(label, (self.dose_button.x + 10, self.dose_button.y + 10))
        
        pygame.draw.rect(self.surface, (200, 150, 100), self.dose_all_button)
        pygame.draw.rect(self.surface, (255, 255, 255), self.dose_all_button, 1)
        label = self.font.render("DOSE ALL", True, (0, 0, 0))
        self.surface.blit(label, (self.dose_all_button.x + 10, self.dose_all_button.y + 10))
        
        # Ego death button (pulsing)
        pulse = int(127 + 127 * np.sin(pygame.time.get_ticks() * 0.005))
        ego_color = (255, pulse, 0)
        pygame.draw.rect(self.surface, ego_color, self.ego_death_button)
        pygame.draw.rect(self.surface, (255, 255, 255), self.ego_death_button, 2)
        label = self.title_font.render("⚠ EGO DEATH ⚠", True, (0, 0, 0))
        label_rect = label.get_rect(center=self.ego_death_button.center)
        self.surface.blit(label, label_rect)
        
        # Blit to screen
        screen.blit(self.surface, (self.x, self.y))
    
    def handle_event(self, event) -> Optional[Tuple[str, int, float]]:
        """
        Handle events.
        
        Returns:
            Tuple of (action, molecule_type, dosage) or None
            Actions: 'dose_one', 'dose_all', 'ego_death'
        """
        if not self.visible:
            return None
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Convert to panel-relative coordinates
            mx, my = event.pos
            rel_x = mx - self.x
            rel_y = my - self.y
            
            # Check if inside panel
            if not (0 <= rel_x <= self.width and 0 <= rel_y <= self.height):
                return None
            
            rel_pos = (rel_x, rel_y)
            
            # Check molecule buttons
            for i, btn in enumerate(self.molecule_buttons):
                if btn.collidepoint(rel_pos):
                    self.selected_molecule = i
                    return None
            
            # Check dose button
            if self.dose_button.collidepoint(rel_pos):
                return ('dose_one', self.selected_molecule, self.dosage)
            
            # Check dose all button
            if self.dose_all_button.collidepoint(rel_pos):
                return ('dose_all', self.selected_molecule, self.dosage)
            
            # Check ego death button
            if self.ego_death_button.collidepoint(rel_pos):
                return ('ego_death', 4, 500.0)  # Potentiator at max dose
        
        return None


class ViewerPanelManager:
    """
    Manages all viewer panels in the research platform.
    
    Handles:
    - Panel visibility toggling
    - Event routing
    - Rendering all visible panels
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        """Initialize panel manager."""
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Create panels with positions
        # Circuit8 - top center
        self.circuit8_panel = Circuit8MiniPanel(
            x=screen_width // 2 - 140,
            y=10,
            width=280,
            height=200
        )
        self.circuit8_panel.visible = False  # Start hidden
        
        # Neural network - middle right
        self.neural_panel = NeuralNetworkMiniPanel(
            x=screen_width - 320,
            y=420,
            width=300,
            height=250
        )
        self.neural_panel.visible = False  # Start hidden
        
        # Drug control - left side
        self.drug_panel = DrugControlMiniPanel(
            x=10,
            y=screen_height - 360,
            width=280,
            height=350
        )
        self.drug_panel.visible = False  # Start hidden
        
        # Track which panels are active
        self.panels = {
            'circuit8': self.circuit8_panel,
            'neural': self.neural_panel,
            'drug': self.drug_panel,
        }
    
    def toggle_panel(self, panel_name: str):
        """Toggle visibility of a panel."""
        if panel_name in self.panels:
            panel = self.panels[panel_name]
            panel.visible = not panel.visible
            return panel.visible
        return False
    
    def render(self, screen, circuit8=None, creatures=None, selected_creature=None):
        """Render all visible panels."""
        # Circuit8 panel
        if self.circuit8_panel.visible and circuit8:
            self.circuit8_panel.render(screen, circuit8, creatures)
        
        # Neural network panel (needs selected creature)
        if self.neural_panel.visible and selected_creature:
            self.neural_panel.render(screen, selected_creature)
        
        # Drug panel
        if self.drug_panel.visible:
            self.drug_panel.render(screen, creatures)
    
    def handle_event(self, event) -> Optional[Tuple[str, Any]]:
        """
        Handle events for all panels.
        
        Returns:
            Tuple of (panel_name, action_data) or None
        """
        # Drug panel events
        if self.drug_panel.visible:
            result = self.drug_panel.handle_event(event)
            if result:
                return ('drug', result)
        
        return None
