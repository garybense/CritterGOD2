"""
Slider widget for parameter control.

Simple pygame-based slider for runtime parameter adjustment.
"""

import pygame
from typing import Optional, Callable
from core.config.parameters import Parameter


class SliderWidget:
    """
    Interactive slider for parameter control.
    
    Allows drag-to-adjust parameter values with visual feedback.
    
    Attributes:
        x, y: Position
        width, height: Dimensions
        parameter: Parameter to control
        dragging: Whether currently being dragged
        on_change: Optional callback when value changes
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        parameter: Parameter,
        on_change: Optional[Callable] = None
    ):
        """
        Initialize slider.
        
        Args:
            x, y: Position
            width, height: Dimensions
            parameter: Parameter to control
            on_change: Callback when value changes
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.parameter = parameter
        self.dragging = False
        self.on_change = on_change
        
        # Visual settings
        self.track_color = (100, 100, 100)
        self.handle_color = (200, 200, 200)
        self.handle_hover_color = (255, 255, 100)
        self.handle_drag_color = (255, 200, 0)
        self.text_color = (255, 255, 255)
        self.value_color = (100, 255, 100)
        
        self.handle_width = 12
        self.handle_height = height + 4
    
    def get_handle_x(self) -> int:
        """Calculate handle X position from parameter value."""
        param_range = self.parameter.max_val - self.parameter.min_val
        if param_range == 0:
            return self.x
        
        normalized = (self.parameter.value - self.parameter.min_val) / param_range
        return int(self.x + normalized * self.width)
    
    def set_value_from_x(self, mouse_x: int):
        """Update parameter value from mouse X position."""
        # Clamp to track
        x_clamped = max(self.x, min(self.x + self.width, mouse_x))
        
        # Calculate normalized position
        normalized = (x_clamped - self.x) / self.width
        
        # Calculate new value
        param_range = self.parameter.max_val - self.parameter.min_val
        new_value = self.parameter.min_val + normalized * param_range
        
        # Update parameter
        old_value = self.parameter.value
        self.parameter.set_value(new_value)
        
        # Callback if value changed
        if self.on_change and old_value != self.parameter.value:
            self.on_change(self.parameter)
    
    def is_mouse_over_handle(self, mouse_pos: tuple) -> bool:
        """Check if mouse is over the handle."""
        handle_x = self.get_handle_x()
        mx, my = mouse_pos
        
        return (handle_x - self.handle_width // 2 <= mx <= handle_x + self.handle_width // 2 and
                self.y - 2 <= my <= self.y + self.height + 2)
    
    def is_mouse_over_track(self, mouse_pos: tuple) -> bool:
        """Check if mouse is over the track."""
        mx, my = mouse_pos
        return (self.x <= mx <= self.x + self.width and
                self.y <= my <= self.y + self.height)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event.
        
        Args:
            event: Pygame event
            
        Returns:
            True if event was consumed
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.is_mouse_over_handle(event.pos) or self.is_mouse_over_track(event.pos):
                self.dragging = True
                self.set_value_from_x(event.pos[0])
                return True
        
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True
        
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.set_value_from_x(event.pos[0])
                return True
        
        return False
    
    def render(self, surface: pygame.Surface, font: pygame.font.Font):
        """
        Render slider to surface.
        
        Args:
            surface: Pygame surface to draw on
            font: Font for text rendering
        """
        # Draw track
        pygame.draw.rect(
            surface,
            self.track_color,
            (self.x, self.y, self.width, self.height)
        )
        
        # Draw filled portion
        handle_x = self.get_handle_x()
        filled_width = handle_x - self.x
        if filled_width > 0:
            pygame.draw.rect(
                surface,
                (80, 150, 80),
                (self.x, self.y, filled_width, self.height)
            )
        
        # Draw handle
        mouse_pos = pygame.mouse.get_pos()
        if self.dragging:
            handle_color = self.handle_drag_color
        elif self.is_mouse_over_handle(mouse_pos):
            handle_color = self.handle_hover_color
        else:
            handle_color = self.handle_color
        
        pygame.draw.rect(
            surface,
            handle_color,
            (handle_x - self.handle_width // 2,
             self.y - 2,
             self.handle_width,
             self.handle_height)
        )
        
        # Draw value text
        if self.parameter.int_only:
            value_text = f"{int(self.parameter.value)}"
        else:
            value_text = f"{self.parameter.value:.3f}"
        
        text_surface = font.render(value_text, True, self.value_color)
        text_x = handle_x - text_surface.get_width() // 2
        text_y = self.y + self.height + 5
        surface.blit(text_surface, (text_x, text_y))


class LabeledSlider:
    """Slider with label and value display."""
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        parameter: Parameter,
        on_change: Optional[Callable] = None
    ):
        """
        Initialize labeled slider.
        
        Args:
            x, y: Position
            width: Total width
            parameter: Parameter to control
            on_change: Callback when value changes
        """
        self.x = x
        self.y = y
        self.width = width
        self.parameter = parameter
        
        # Create slider (below label)
        self.slider = SliderWidget(
            x, y + 20, width, 8, parameter, on_change
        )
        
        self.label_color = (200, 200, 200)
        self.description_color = (150, 150, 150)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle pygame event."""
        return self.slider.handle_event(event)
    
    def render(self, surface: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font):
        """
        Render labeled slider.
        
        Args:
            surface: Pygame surface
            font: Font for label
            small_font: Font for description
        """
        # Draw label
        label_text = self.parameter.name.replace('_', ' ').title()
        label_surface = font.render(label_text, True, self.label_color)
        surface.blit(label_surface, (self.x, self.y))
        
        # Draw slider
        self.slider.render(surface, small_font)
        
        # Draw range info
        range_text = f"[{self.parameter.min_val:.1f} - {self.parameter.max_val:.1f}]"
        range_surface = small_font.render(range_text, True, self.description_color)
        surface.blit(range_surface, (self.x + self.width - range_surface.get_width(), self.y))
