"""
Morphic Field System

How neurons connect to Circuit8 channels.
Based on telepathic-critterdrug's RuRdGuGdBuBd system.

Enhanced with semantic field types for:
- Pheromone trails (food/drug locations)
- Danger zones (predators, starvation areas)
- Mating signals (reproductive readiness)
- Territory markers (claimed areas)
"""

from enum import IntEnum
from typing import Tuple, Dict, Optional
import numpy as np


class MorphicChannel(IntEnum):
    """
    Morphic field channels that neurons can tune into.
    
    From telepathic-critterdrug:
    Each neuron has a RuRdGuGdBuBd value (0-5) determining which
    morphic channel affects it.
    
    Channels:
    - 0: Red Up (warm)
    - 1: Red Down (warm inverted)
    - 2: Green Up
    - 3: Green Down
    - 4: Blue Up (cool inverted)
    - 5: Blue Down (cool)
    """
    RED_UP = 0
    RED_DOWN = 1
    GREEN_UP = 2
    GREEN_DOWN = 3
    BLUE_UP = 4
    BLUE_DOWN = 5
    
    @classmethod
    def get_influence(cls, channel: int, r: int, g: int, b: int) -> float:
        """
        Calculate morphic field influence on a neuron.
        
        Neurons read from Circuit8 BEFORE processing synapses.
        This creates the feedback loop: creature thought → screen → other creatures.
        
        Args:
            channel: Which morphic channel (0-5)
            r: Red value from Circuit8
            g: Green value from Circuit8
            b: Blue value from Circuit8
            
        Returns:
            Influence value (affects neuron potential)
        """
        if channel == cls.RED_UP:
            return float(r) * 10.0  # Warm channel
        elif channel == cls.RED_DOWN:
            return float(255 - r) * 10.0  # Warm inverted
        elif channel == cls.GREEN_UP:
            return float(g) * 10.0
        elif channel == cls.GREEN_DOWN:
            return float(255 - g) * 10.0
        elif channel == cls.BLUE_UP:
            return float(255 - b) * 10.0  # Cool inverted
        elif channel == cls.BLUE_DOWN:
            return float(b) * 10.0  # Cool channel
        else:
            return 0.0


class FieldType(IntEnum):
    """
    Semantic field types mapped to color channels.
    
    These represent different "pheromone" types that creatures
    can emit and sense through the shared Circuit8 canvas.
    """
    FOOD_TRAIL = 0      # Green channel - "food is here"
    DANGER_ZONE = 1     # Red channel - "danger/death here"
    MATING_SIGNAL = 2   # Magenta (R+B) - "ready to mate"
    TERRITORY = 3       # Blue channel - "this is my area"
    EXCITEMENT = 4      # Yellow (R+G) - "something interesting"
    CALM = 5            # Cyan (G+B) - "safe area"


class MorphicFieldReader:
    """
    Reads semantic fields from Circuit8.
    
    Provides high-level interface for creatures to sense
    different types of morphic signals.
    """
    
    # Color signatures for each field type
    FIELD_SIGNATURES = {
        FieldType.FOOD_TRAIL: {'g_min': 150, 'r_max': 100, 'b_max': 100},
        FieldType.DANGER_ZONE: {'r_min': 150, 'g_max': 100, 'b_max': 100},
        FieldType.MATING_SIGNAL: {'r_min': 120, 'b_min': 120, 'g_max': 80},
        FieldType.TERRITORY: {'b_min': 150, 'r_max': 100, 'g_max': 100},
        FieldType.EXCITEMENT: {'r_min': 120, 'g_min': 120, 'b_max': 80},
        FieldType.CALM: {'g_min': 120, 'b_min': 120, 'r_max': 80},
    }
    
    def __init__(self, circuit8):
        """
        Initialize field reader.
        
        Args:
            circuit8: The shared Circuit8 canvas
        """
        self.circuit8 = circuit8
    
    def read_field_at(
        self,
        x: int,
        y: int,
        field_type: FieldType
    ) -> float:
        """
        Read intensity of a specific field type at position.
        
        Args:
            x: X coordinate (world space, will be converted)
            y: Y coordinate (world space, will be converted)
            field_type: Which field to read
            
        Returns:
            Field intensity (0.0 to 1.0)
        """
        # Convert to canvas coords
        px = x % self.circuit8.width
        py = y % self.circuit8.height
        
        r, g, b = self.circuit8.read_pixel(px, py)
        
        return self._calculate_field_intensity(r, g, b, field_type)
    
    def _calculate_field_intensity(
        self,
        r: int,
        g: int,
        b: int,
        field_type: FieldType
    ) -> float:
        """
        Calculate how strongly a pixel matches a field type.
        """
        sig = self.FIELD_SIGNATURES[field_type]
        
        # Check if pixel matches signature
        matches = True
        intensity = 0.0
        
        if 'r_min' in sig and r < sig['r_min']:
            matches = False
        if 'r_max' in sig and r > sig['r_max']:
            matches = False
        if 'g_min' in sig and g < sig['g_min']:
            matches = False
        if 'g_max' in sig and g > sig['g_max']:
            matches = False
        if 'b_min' in sig and b < sig['b_min']:
            matches = False
        if 'b_max' in sig and b > sig['b_max']:
            matches = False
        
        if not matches:
            return 0.0
        
        # Calculate intensity based on field type
        if field_type == FieldType.FOOD_TRAIL:
            intensity = g / 255.0
        elif field_type == FieldType.DANGER_ZONE:
            intensity = r / 255.0
        elif field_type == FieldType.MATING_SIGNAL:
            intensity = (int(r) + int(b)) / 510.0
        elif field_type == FieldType.TERRITORY:
            intensity = b / 255.0
        elif field_type == FieldType.EXCITEMENT:
            intensity = (int(r) + int(g)) / 510.0
        elif field_type == FieldType.CALM:
            intensity = (int(g) + int(b)) / 510.0
        
        return intensity
    
    def read_all_fields_at(self, x: int, y: int) -> Dict[FieldType, float]:
        """
        Read all field types at position.
        
        Args:
            x, y: Position
            
        Returns:
            Dictionary of field type -> intensity
        """
        px = x % self.circuit8.width
        py = y % self.circuit8.height
        r, g, b = self.circuit8.read_pixel(px, py)
        
        return {
            ft: self._calculate_field_intensity(r, g, b, ft)
            for ft in FieldType
        }
    
    def scan_area(
        self,
        center_x: int,
        center_y: int,
        radius: int,
        field_type: FieldType
    ) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Scan area for strongest signal of a field type.
        
        Args:
            center_x, center_y: Center of scan
            radius: Scan radius in pixels
            field_type: Field to search for
            
        Returns:
            (max_intensity, (best_x, best_y)) or (0.0, None)
        """
        max_intensity = 0.0
        best_pos = None
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy > radius*radius:
                    continue  # Circle, not square
                
                x = center_x + dx
                y = center_y + dy
                
                intensity = self.read_field_at(x, y, field_type)
                if intensity > max_intensity:
                    max_intensity = intensity
                    best_pos = (x, y)
        
        return max_intensity, best_pos
    
    def get_field_gradient(
        self,
        x: int,
        y: int,
        field_type: FieldType
    ) -> Tuple[float, float]:
        """
        Calculate gradient direction for a field (which way to move).
        
        Args:
            x, y: Current position
            field_type: Field to follow
            
        Returns:
            (dx, dy) normalized direction toward stronger signal
        """
        # Sample neighbors
        center = self.read_field_at(x, y, field_type)
        left = self.read_field_at(x - 1, y, field_type)
        right = self.read_field_at(x + 1, y, field_type)
        up = self.read_field_at(x, y - 1, field_type)
        down = self.read_field_at(x, y + 1, field_type)
        
        # Calculate gradient
        dx = right - left
        dy = down - up
        
        # Normalize
        mag = np.sqrt(dx*dx + dy*dy)
        if mag > 0.001:
            dx /= mag
            dy /= mag
        
        return dx, dy


class MorphicFieldWriter:
    """
    Writes semantic fields to Circuit8.
    
    Provides high-level interface for creatures to emit
    different types of morphic signals.
    """
    
    # Color values for each field type
    FIELD_COLORS = {
        FieldType.FOOD_TRAIL: (0, 200, 0),      # Green
        FieldType.DANGER_ZONE: (255, 0, 0),     # Red
        FieldType.MATING_SIGNAL: (200, 0, 200), # Magenta
        FieldType.TERRITORY: (0, 0, 200),       # Blue
        FieldType.EXCITEMENT: (200, 200, 0),    # Yellow
        FieldType.CALM: (0, 200, 200),          # Cyan
    }
    
    def __init__(self, circuit8):
        """
        Initialize field writer.
        
        Args:
            circuit8: The shared Circuit8 canvas
        """
        self.circuit8 = circuit8
    
    def emit_field(
        self,
        x: int,
        y: int,
        field_type: FieldType,
        intensity: float = 1.0,
        radius: int = 1
    ):
        """
        Emit a field signal at position.
        
        Args:
            x, y: World position
            field_type: Type of signal to emit
            intensity: Signal strength (0.0 to 1.0)
            radius: Spread radius (0 = single pixel)
        """
        base_color = self.FIELD_COLORS[field_type]
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > radius:
                    continue
                
                # Falloff with distance
                falloff = 1.0 - (dist / (radius + 1))
                local_intensity = intensity * falloff
                
                px = (x + dx) % self.circuit8.width
                py = (y + dy) % self.circuit8.height
                
                # Blend with existing
                r = int(base_color[0] * local_intensity)
                g = int(base_color[1] * local_intensity)
                b = int(base_color[2] * local_intensity)
                
                self.circuit8.write_pixel(px, py, r, g, b, blend=True)
    
    def mark_food_found(self, x: int, y: int, intensity: float = 0.8):
        """Emit food trail signal."""
        self.emit_field(x, y, FieldType.FOOD_TRAIL, intensity, radius=2)
    
    def mark_danger(self, x: int, y: int, intensity: float = 1.0):
        """Emit danger warning signal."""
        self.emit_field(x, y, FieldType.DANGER_ZONE, intensity, radius=3)
    
    def mark_mating_ready(self, x: int, y: int, intensity: float = 0.6):
        """Emit mating readiness signal."""
        self.emit_field(x, y, FieldType.MATING_SIGNAL, intensity, radius=2)
    
    def mark_territory(self, x: int, y: int, intensity: float = 0.5):
        """Emit territory marker."""
        self.emit_field(x, y, FieldType.TERRITORY, intensity, radius=1)
    
    def mark_excitement(self, x: int, y: int, intensity: float = 0.7):
        """Emit excitement/interest signal."""
        self.emit_field(x, y, FieldType.EXCITEMENT, intensity, radius=2)
    
    def mark_calm(self, x: int, y: int, intensity: float = 0.5):
        """Emit calm/safe area signal."""
        self.emit_field(x, y, FieldType.CALM, intensity, radius=2)
