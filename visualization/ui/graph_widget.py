"""
Graph widget for real-time statistics visualization.

Displays time-series data with automatic scaling and multiple series support.
"""

import pygame
from typing import List, Tuple, Optional
from collections import deque


class GraphWidget:
    """
    Time-series graph widget.
    
    Displays scrolling line graphs with automatic Y-axis scaling.
    
    Attributes:
        x, y: Position
        width, height: Dimensions
        max_points: Maximum number of data points to display
        series: List of (name, color, data) tuples
        auto_scale: Whether to automatically scale Y-axis
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        title: str,
        max_points: int = 200,
        auto_scale: bool = True
    ):
        """
        Initialize graph widget.
        
        Args:
            x, y: Position
            width, height: Dimensions
            title: Graph title
            max_points: Maximum points to display
            auto_scale: Auto-scale Y-axis to data
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.title = title
        self.max_points = max_points
        self.auto_scale = auto_scale
        
        # Series: [(name, color, deque)]
        self.series: List[Tuple[str, Tuple[int, int, int], deque]] = []
        
        # Manual Y-axis range (used if auto_scale=False)
        self.y_min = 0.0
        self.y_max = 100.0
        
        # Visual settings
        self.bg_color = (20, 20, 30)
        self.border_color = (100, 100, 120)
        self.grid_color = (40, 40, 50)
        self.title_color = (200, 200, 220)
        self.axis_label_color = (150, 150, 170)
        
        # Padding
        self.padding_left = 50
        self.padding_right = 10
        self.padding_top = 30
        self.padding_bottom = 25
        
        # Graph area
        self.graph_x = self.x + self.padding_left
        self.graph_y = self.y + self.padding_top
        self.graph_width = self.width - self.padding_left - self.padding_right
        self.graph_height = self.height - self.padding_top - self.padding_bottom
    
    def add_series(self, name: str, color: Tuple[int, int, int]):
        """
        Add a new data series.
        
        Args:
            name: Series name for legend
            color: RGB color tuple
        """
        self.series.append((name, color, deque(maxlen=self.max_points)))
    
    def add_data_point(self, series_index: int, value: float):
        """
        Add data point to series.
        
        Args:
            series_index: Index of series
            value: Data value
        """
        if 0 <= series_index < len(self.series):
            _, _, data = self.series[series_index]
            data.append(value)
    
    def get_y_range(self) -> Tuple[float, float]:
        """Calculate Y-axis range from data."""
        if not self.auto_scale:
            return self.y_min, self.y_max
        
        # Find min/max across all series
        all_values = []
        for _, _, data in self.series:
            all_values.extend(data)
        
        if not all_values:
            return 0.0, 1.0
        
        min_val = min(all_values)
        max_val = max(all_values)
        
        # Add 10% padding
        range_val = max_val - min_val
        if range_val < 0.001:
            range_val = 1.0
        
        return min_val - range_val * 0.1, max_val + range_val * 0.1
    
    def value_to_y(self, value: float, y_min: float, y_max: float) -> int:
        """Convert data value to screen Y coordinate."""
        if y_max - y_min < 0.001:
            return self.graph_y + self.graph_height // 2
        
        normalized = (value - y_min) / (y_max - y_min)
        # Flip Y (screen coords are top-down)
        screen_y = self.graph_y + self.graph_height - int(normalized * self.graph_height)
        return max(self.graph_y, min(self.graph_y + self.graph_height, screen_y))
    
    def render(self, surface: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font):
        """
        Render graph to surface.
        
        Args:
            surface: Pygame surface
            font: Font for title
            small_font: Font for labels
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
        title_surface = font.render(self.title, True, self.title_color)
        title_x = self.x + (self.width - title_surface.get_width()) // 2
        surface.blit(title_surface, (title_x, self.y + 5))
        
        # Get Y range
        y_min, y_max = self.get_y_range()
        
        # Draw grid lines
        num_grid_lines = 5
        for i in range(num_grid_lines + 1):
            y = self.graph_y + int(i * self.graph_height / num_grid_lines)
            pygame.draw.line(
                surface,
                self.grid_color,
                (self.graph_x, y),
                (self.graph_x + self.graph_width, y),
                1
            )
            
            # Y-axis label
            value = y_max - i * (y_max - y_min) / num_grid_lines
            if abs(value) < 0.01:
                label = "0"
            elif abs(value) > 1000:
                label = f"{value:.0f}"
            else:
                label = f"{value:.1f}"
            
            label_surface = small_font.render(label, True, self.axis_label_color)
            surface.blit(label_surface, (self.x + 5, y - 7))
        
        # Draw series
        for name, color, data in self.series:
            if len(data) < 2:
                continue
            
            points = []
            for i, value in enumerate(data):
                x = self.graph_x + int(i * self.graph_width / self.max_points)
                y = self.value_to_y(value, y_min, y_max)
                points.append((x, y))
            
            # Draw line
            if len(points) >= 2:
                pygame.draw.lines(surface, color, False, points, 2)
        
        # Draw legend
        legend_y = self.y + self.height - 20
        legend_x = self.graph_x
        for name, color, data in self.series:
            # Color box
            pygame.draw.rect(
                surface,
                color,
                (legend_x, legend_y, 12, 12)
            )
            
            # Name
            name_surface = small_font.render(name, True, self.title_color)
            surface.blit(name_surface, (legend_x + 16, legend_y))
            
            legend_x += 16 + name_surface.get_width() + 15


class MultiGraphPanel:
    """Panel containing multiple stacked graphs."""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Initialize multi-graph panel.
        
        Args:
            x, y: Position
            width, height: Dimensions
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.graphs: List[GraphWidget] = []
    
    def add_graph(self, title: str, height: int, max_points: int = 200) -> GraphWidget:
        """
        Add a new graph to the panel.
        
        Args:
            title: Graph title
            height: Graph height
            max_points: Maximum data points
            
        Returns:
            Created graph widget
        """
        # Stack graphs vertically
        y_offset = sum(g.height for g in self.graphs)
        
        graph = GraphWidget(
            self.x,
            self.y + y_offset,
            self.width,
            height,
            title,
            max_points
        )
        self.graphs.append(graph)
        return graph
    
    def render(self, surface: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font):
        """Render all graphs."""
        for graph in self.graphs:
            graph.render(surface, font, small_font)
