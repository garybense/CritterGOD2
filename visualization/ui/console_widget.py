"""
Console output widget for event logging.

Displays scrolling text output in critterdrug style.
"""

import pygame
from typing import List
from collections import deque


class ConsoleWidget:
    """
    Scrolling console output widget.
    
    Displays event log lines with automatic scrolling.
    Similar to critterdrug's console output.
    
    Attributes:
        x, y: Position
        width, height: Dimensions
        lines: Recent log lines
        max_lines: Maximum lines to display
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        max_lines: int = 200
    ):
        """
        Initialize console widget.
        
        Args:
            x, y: Position
            width, height: Dimensions
            max_lines: Maximum lines to keep in history
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.max_lines = max_lines
        
        # Lines buffer
        self.lines: deque = deque(maxlen=max_lines)
        
        # Visual settings
        self.bg_color = (10, 10, 15)
        self.border_color = (80, 80, 100)
        self.text_color = (180, 180, 200)
        self.title_color = (200, 200, 220)
        
        # Event type colors (critterdrug style)
        self.birth_color = (100, 255, 100)
        self.death_color = (255, 100, 100)
        self.reproduction_color = (255, 255, 100)
        self.mutation_color = (200, 150, 255)
        
        # Scroll state
        self.scroll_offset = 0
        self.auto_scroll = True
    
    def add_line(self, line: str):
        """
        Add a line to console output.
        
        Args:
            line: Text line to add
        """
        self.lines.append(line)
        
        # Auto-scroll to bottom
        if self.auto_scroll:
            self.scroll_offset = 0
    
    def add_lines(self, lines: List[str]):
        """
        Add multiple lines to console.
        
        Args:
            lines: List of text lines
        """
        for line in lines:
            self.add_line(line)
    
    def clear(self):
        """Clear all console output."""
        self.lines.clear()
        self.scroll_offset = 0
    
    def get_line_color(self, line: str) -> tuple:
        """
        Determine color for line based on content.
        
        Args:
            line: Text line
            
        Returns:
            RGB color tuple
        """
        # Detect event type from line content
        if "born" in line.lower() or "ad:" in line:
            return self.birth_color
        elif "died" in line.lower():
            return self.death_color
        elif "procreated" in line.lower() or "→" in line:
            return self.reproduction_color
        elif "mutation" in line.lower():
            return self.mutation_color
        else:
            return self.text_color
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame event.
        
        Args:
            event: Pygame event
            
        Returns:
            True if event was consumed
        """
        if event.type == pygame.MOUSEWHEEL:
            # Check if mouse is over console
            mouse_pos = pygame.mouse.get_pos()
            if (self.x <= mouse_pos[0] <= self.x + self.width and
                self.y <= mouse_pos[1] <= self.y + self.height):
                # Scroll
                self.scroll_offset -= event.y * 3
                self.scroll_offset = max(0, min(len(self.lines) - 1, self.scroll_offset))
                
                # Disable auto-scroll if user scrolled up
                if self.scroll_offset > 0:
                    self.auto_scroll = False
                else:
                    self.auto_scroll = True
                
                return True
        
        return False
    
    def render(self, surface: pygame.Surface, font: pygame.font.Font):
        """
        Render console to surface.
        
        Args:
            surface: Pygame surface
            font: Monospace font for text
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
        title_surface = font.render("Console Output", True, self.title_color)
        surface.blit(title_surface, (self.x + 5, self.y + 5))
        
        # Calculate visible lines
        line_height = font.get_height()
        visible_lines = (self.height - 30) // line_height
        
        # Draw lines (bottom-up, most recent at bottom)
        if len(self.lines) > 0:
            start_index = max(0, len(self.lines) - visible_lines - self.scroll_offset)
            end_index = len(self.lines) - self.scroll_offset
            
            y_pos = self.y + 28
            for i in range(start_index, end_index):
                if i < 0 or i >= len(self.lines):
                    continue
                
                line = self.lines[i]
                color = self.get_line_color(line)
                
                # Truncate if too long
                max_chars = (self.width - 10) // (font.size("X")[0])
                if len(line) > max_chars:
                    line = line[:max_chars - 3] + "..."
                
                line_surface = font.render(line, True, color)
                surface.blit(line_surface, (self.x + 5, y_pos))
                y_pos += line_height
        
        # Scroll indicator
        if not self.auto_scroll:
            scroll_text = f"(scroll: {self.scroll_offset})"
            scroll_surface = font.render(scroll_text, True, (150, 150, 150))
            surface.blit(
                scroll_surface,
                (self.x + self.width - scroll_surface.get_width() - 5, self.y + 5)
            )
