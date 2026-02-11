"""
Configuration panel with parameter sliders.

Provides UI for runtime parameter adjustment.
"""

import pygame
from typing import List, Optional, Callable
from core.config.config_manager import ConfigManager
from visualization.ui.slider_widget import LabeledSlider


class ConfigPanel:
    """
    Configuration panel with scrollable parameter sliders.
    
    Displays categories of parameters with collapsible sections.
    
    Attributes:
        x, y: Position
        width, height: Dimensions
        config: ConfigManager instance
        sliders: List of labeled sliders
        on_change: Callback when parameter changes
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        config: ConfigManager,
        on_change: Optional[Callable] = None
    ):
        """
        Initialize config panel.
        
        Args:
            x, y: Position
            width, height: Dimensions
            config: ConfigManager instance
            on_change: Callback when parameter changes
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.config = config
        self.on_change = on_change
        
        # Visual settings
        self.bg_color = (15, 15, 20)
        self.border_color = (80, 80, 100)
        self.title_color = (200, 200, 220)
        self.category_color = (150, 200, 255)
        
        # Scrolling
        self.scroll_offset = 0
        self.content_height = 0
        
        # Build sliders
        self.sliders: List[tuple] = []  # [(category, slider), ...]
        self._build_sliders()
    
    def _build_sliders(self):
        """Build slider widgets from config parameters."""
        self.sliders.clear()
        
        # Define key parameters to expose (most useful for research)
        key_params = [
            # Creature params
            ("creature", "creature_start_energy"),
            ("creature", "creature_max_lifetime"),
            ("creature", "creature_proc_interval"),
            
            # Neural params
            ("neural", "neuron_count_min"),
            ("neural", "neuron_count_max"),
            ("neural", "synapses_per_neuron"),
            ("neural", "plasticity_rate"),
            ("neural", "inhibitory_ratio"),
            
            # Body params
            ("body", "body_max_mutations"),
            ("body", "body_mutation_rate"),
            
            # Drug params
            ("neural", "drug_multiplier_inhibitory_agonist"),
            ("neural", "drug_multiplier_excitatory_agonist"),
            ("neural", "drug_multiplier_potentiator"),
            ("neural", "drug_decay_rate"),
            
            # Resource params
            ("resources", "food_spawn_rate"),
            ("resources", "food_energy_value"),
            ("resources", "drug_spawn_rate"),
            
            # Evolution params
            ("evolution", "mutation_rate"),
            ("evolution", "max_population"),
            
            # Physics params
            ("physics", "physics_friction"),
            ("physics", "physics_damping"),
        ]
        
        y_offset = 50  # Start below title
        
        current_category = None
        for category, param_name in key_params:
            param = self.config.get_parameter(param_name)
            if param is None:
                continue
            
            # Add category header if new category
            if category != current_category:
                self.sliders.append((f"CATEGORY:{category}", None))
                current_category = category
                y_offset += 30
            
            # Create slider
            slider = LabeledSlider(
                self.x + 10,
                self.y + y_offset,
                self.width - 20,
                param,
                on_change=self._on_slider_change
            )
            self.sliders.append((category, slider))
            y_offset += 50  # Spacing between sliders
        
        self.content_height = y_offset
    
    def _on_slider_change(self, parameter):
        """Handle slider value change."""
        if self.on_change:
            self.on_change(parameter)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event.
        
        Args:
            event: Pygame event
            
        Returns:
            True if event was consumed
        """
        # Handle scrolling
        if event.type == pygame.MOUSEWHEEL:
            mouse_pos = pygame.mouse.get_pos()
            if (self.x <= mouse_pos[0] <= self.x + self.width and
                self.y <= mouse_pos[1] <= self.y + self.height):
                self.scroll_offset -= event.y * 20
                self.scroll_offset = max(0, min(
                    max(0, self.content_height - self.height + 20),
                    self.scroll_offset
                ))
                return True
        
        # Handle slider events (account for scroll offset)
        for category, slider in self.sliders:
            if slider is None or category.startswith("CATEGORY:"):
                continue
            
            # Adjust event position for scroll
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
                # Create adjusted event
                adjusted_event = event
                if hasattr(event, 'pos'):
                    adjusted_pos = (event.pos[0], event.pos[1] + self.scroll_offset)
                    # Check if adjusted position is within visible area
                    if not (self.y <= event.pos[1] <= self.y + self.height):
                        continue
                    
                    # Create new event with adjusted position
                    adjusted_event = pygame.event.Event(
                        event.type,
                        {'pos': adjusted_pos, 'button': getattr(event, 'button', 0)}
                    )
                
                if slider.handle_event(adjusted_event):
                    return True
        
        return False
    
    def render(self, surface: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font):
        """
        Render config panel.
        
        Args:
            surface: Pygame surface
            font: Font for title/categories
            small_font: Font for slider labels
        """
        # Background
        pygame.draw.rect(
            surface,
            self.bg_color,
            (self.x, self.y, self.width, self.height)
        )
        
        # Border
        pygame.draw.rect(
            surface,
            self.border_color,
            (self.x, self.y, self.width, self.height),
            2
        )
        
        # Title
        title_surface = font.render("Configuration", True, self.title_color)
        surface.blit(title_surface, (self.x + 10, self.y + 5))
        
        # Create clipping rect for scrollable area
        clip_rect = pygame.Rect(self.x, self.y + 30, self.width, self.height - 30)
        surface.set_clip(clip_rect)
        
        # Render sliders with scroll offset
        for category, slider in self.sliders:
            if category.startswith("CATEGORY:"):
                # Category header
                cat_name = category.split(":")[1].title()
                cat_surface = font.render(cat_name, True, self.category_color)
                cat_y = slider.y - self.scroll_offset if slider else self.y + 50
                # Find the y position from the next slider
                idx = self.sliders.index((category, slider))
                if idx + 1 < len(self.sliders):
                    next_cat, next_slider = self.sliders[idx + 1]
                    if next_slider:
                        cat_y = next_slider.y - 30 - self.scroll_offset
                
                surface.blit(cat_surface, (self.x + 10, cat_y))
            elif slider:
                # Render slider with scroll offset
                original_y = slider.y
                slider.y = original_y - self.scroll_offset
                slider.slider.y = slider.y + 20
                
                # Only render if visible
                if slider.y > self.y + 30 and slider.y < self.y + self.height:
                    slider.render(surface, small_font, small_font)
                
                # Restore original position
                slider.y = original_y
                slider.slider.y = original_y + 20
        
        # Remove clip
        surface.set_clip(None)
        
        # Scrollbar indicator
        if self.content_height > self.height:
            scrollbar_height = max(20, int((self.height / self.content_height) * self.height))
            scrollbar_y = self.y + int((self.scroll_offset / self.content_height) * self.height)
            pygame.draw.rect(
                surface,
                (100, 100, 120),
                (self.x + self.width - 8, scrollbar_y, 6, scrollbar_height)
            )
