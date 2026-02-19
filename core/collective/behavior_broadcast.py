"""
Behavior broadcasting system for collective intelligence.

Enables creatures to communicate their state and behaviors through Circuit8:
- Encode behaviors as visual patterns (colors/positions)
- Broadcast success/failure states
- Share resource locations
- Warn about dangers
- Signal reproductive readiness

Creates emergent collective intelligence through visual communication.
"""

from typing import Tuple, Optional
import numpy as np
from enum import IntEnum


class BehaviorType(IntEnum):
    """Types of behaviors that can be broadcast."""
    IDLE = 0
    SEEKING_FOOD = 1
    SEEKING_DRUG = 2
    EATING = 3
    FLEEING = 4
    MATING = 5
    EXPLORING = 6
    RESTING = 7


class SignalType(IntEnum):
    """Types of signals broadcast to Circuit8."""
    RESOURCE_FOUND = 0  # Found food or drug
    DANGER_WARNING = 1  # Low energy, collision, threat
    MATING_CALL = 2     # Ready to reproduce
    SUCCESS = 3         # Gained energy/resources
    FAILURE = 4         # Lost energy/resources


class BehaviorBroadcaster:
    """
    Broadcasts creature behaviors to Circuit8.
    
    Encodes behaviors as colors and patterns that other creatures
    can read and learn from.
    
    Attributes:
        creature_id: Unique identifier for creature
        broadcast_intensity: How strongly to write (0-1)
        signal_colors: Color mappings for different signal types
    """
    
    def __init__(self, creature_id: int):
        """
        Initialize behavior broadcaster.
        
        Args:
            creature_id: Unique creature identifier
        """
        self.creature_id = creature_id
        self.broadcast_intensity = 1.0
        
        # Color codes for different signal types
        self.signal_colors = {
            SignalType.RESOURCE_FOUND: (0, 255, 0),      # Green
            SignalType.DANGER_WARNING: (255, 0, 0),      # Red
            SignalType.MATING_CALL: (255, 0, 255),       # Magenta
            SignalType.SUCCESS: (255, 255, 0),           # Yellow
            SignalType.FAILURE: (0, 0, 255),             # Blue
        }
        
        # Behavior colors (softer, for background)
        self.behavior_colors = {
            BehaviorType.IDLE: (50, 50, 50),
            BehaviorType.SEEKING_FOOD: (100, 150, 100),
            BehaviorType.SEEKING_DRUG: (150, 100, 150),
            BehaviorType.EATING: (150, 200, 150),
            BehaviorType.FLEEING: (200, 100, 100),
            BehaviorType.MATING: (200, 150, 200),
            BehaviorType.EXPLORING: (100, 100, 150),
            BehaviorType.RESTING: (80, 80, 120),
        }
    
    def broadcast_signal(
        self,
        circuit8,
        signal_type: SignalType,
        world_x: float,
        world_y: float,
        world_width: float = 500.0,
        world_height: float = 500.0,
        radius: int = 3
    ):
        """
        Broadcast a signal to Circuit8 at world position.
        
        Args:
            circuit8: Circuit8 telepathic canvas
            signal_type: Type of signal to broadcast
            world_x, world_y: World coordinates
            world_width, world_height: World dimensions
            radius: Radius of signal in pixels
        """
        # Convert world coordinates to Circuit8 coordinates
        canvas_x = int((world_x + world_width/2) / world_width * circuit8.width)
        canvas_y = int((world_y + world_height/2) / world_height * circuit8.height)
        
        # Clamp to canvas bounds
        canvas_x = max(0, min(circuit8.width - 1, canvas_x))
        canvas_y = max(0, min(circuit8.height - 1, canvas_y))
        
        # Get signal color
        color = self.signal_colors.get(signal_type, (255, 255, 255))
        
        # Draw signal in radius around position
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    x = canvas_x + dx
                    y = canvas_y + dy
                    if 0 <= x < circuit8.width and 0 <= y < circuit8.height:
                        # Write with blend mode for accumulation
                        circuit8.write_pixel(x, y, *color, blend=True)
    
    def broadcast_behavior(
        self,
        circuit8,
        behavior: BehaviorType,
        world_x: float,
        world_y: float,
        world_width: float = 500.0,
        world_height: float = 500.0
    ):
        """
        Broadcast current behavior to Circuit8.
        
        Args:
            circuit8: Circuit8 telepathic canvas
            behavior: Current behavior type
            world_x, world_y: World coordinates
            world_width, world_height: World dimensions
        """
        # Convert coordinates
        canvas_x = int((world_x + world_width/2) / world_width * circuit8.width)
        canvas_y = int((world_y + world_height/2) / world_height * circuit8.height)
        
        # Clamp
        canvas_x = max(0, min(circuit8.width - 1, canvas_x))
        canvas_y = max(0, min(circuit8.height - 1, canvas_y))
        
        # Get behavior color (subtle)
        color = self.behavior_colors.get(behavior, (100, 100, 100))
        
        # Write single pixel for behavior (less intrusive)
        circuit8.write_pixel(canvas_x, canvas_y, *color, blend=True)
    
    def mark_resource_location(
        self,
        circuit8,
        resource_x: float,
        resource_y: float,
        resource_type: str,  # 'food' or 'drug'
        world_width: float = 500.0,
        world_height: float = 500.0
    ):
        """
        Mark a resource location on Circuit8 for other creatures.
        
        Args:
            circuit8: Circuit8 telepathic canvas
            resource_x, resource_y: Resource world coordinates
            resource_type: 'food' or 'drug'
            world_width, world_height: World dimensions
        """
        # Convert coordinates
        canvas_x = int((resource_x + world_width/2) / world_width * circuit8.width)
        canvas_y = int((resource_y + world_height/2) / world_height * circuit8.height)
        
        # Clamp
        canvas_x = max(0, min(circuit8.width - 1, canvas_x))
        canvas_y = max(0, min(circuit8.height - 1, canvas_y))
        
        # Color based on resource type
        if resource_type == 'food':
            color = (0, 200, 0)  # Bright green
        else:  # drug
            color = (200, 0, 200)  # Bright magenta
        
        # Draw marker (small cross pattern)
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            x = canvas_x + dx
            y = canvas_y + dy
            if 0 <= x < circuit8.width and 0 <= y < circuit8.height:
                circuit8.write_pixel(x, y, *color, blend=False)  # Overwrite for clarity


class BehaviorReader:
    """
    Reads behaviors and signals from Circuit8.
    
    Allows creatures to observe what others are doing and learn
    from their successes and failures.
    
    Attributes:
        detection_radius: How far to look in pixels
        signal_threshold: Minimum intensity to detect signal
    """
    
    def __init__(self, detection_radius: int = 5):
        """
        Initialize behavior reader.
        
        Args:
            detection_radius: Radius in pixels to sample
        """
        self.detection_radius = detection_radius
        self.signal_threshold = 100  # Minimum color intensity
    
    def read_nearby_signals(
        self,
        circuit8,
        world_x: float,
        world_y: float,
        world_width: float = 500.0,
        world_height: float = 500.0
    ) -> list:
        """
        Read signals near world position.
        
        Args:
            circuit8: Circuit8 telepathic canvas
            world_x, world_y: World coordinates
            world_width, world_height: World dimensions
            
        Returns:
            List of detected signals with types and intensities
        """
        # Convert to canvas coordinates
        canvas_x = int((world_x + world_width/2) / world_width * circuit8.width)
        canvas_y = int((world_y + world_height/2) / world_height * circuit8.height)
        
        # Clamp
        canvas_x = max(0, min(circuit8.width - 1, canvas_x))
        canvas_y = max(0, min(circuit8.height - 1, canvas_y))
        
        detected_signals = []
        
        # Sample in radius
        for dy in range(-self.detection_radius, self.detection_radius + 1):
            for dx in range(-self.detection_radius, self.detection_radius + 1):
                x = canvas_x + dx
                y = canvas_y + dy
                
                if 0 <= x < circuit8.width and 0 <= y < circuit8.height:
                    r, g, b = circuit8.read_pixel(x, y)
                    
                    # Detect signal types based on color
                    signal = self._classify_signal(r, g, b)
                    if signal is not None:
                        intensity = (int(r) + int(g) + int(b)) / (3 * 255)
                        detected_signals.append({
                            'type': signal,
                            'intensity': intensity,
                            'position': (x, y),
                            'offset': (dx, dy)
                        })
        
        return detected_signals
    
    def _classify_signal(self, r: int, g: int, b: int) -> Optional[SignalType]:
        """
        Classify pixel color as signal type.
        
        Args:
            r, g, b: Pixel color values
            
        Returns:
            SignalType if recognized, None otherwise
        """
        # Require minimum intensity (cast to int to avoid uint8 overflow)
        if int(r) + int(g) + int(b) < self.signal_threshold:
            return None
        
        # Classify based on dominant color(s)
        if g > r and g > b and g > 150:
            return SignalType.RESOURCE_FOUND  # Green
        elif r > g and r > b and r > 150:
            return SignalType.DANGER_WARNING  # Red
        elif r > 150 and b > 150 and g < 100:
            return SignalType.MATING_CALL  # Magenta
        elif r > 150 and g > 150 and b < 100:
            return SignalType.SUCCESS  # Yellow
        elif b > r and b > g and b > 150:
            return SignalType.FAILURE  # Blue
        
        return None
    
    def find_resource_markers(
        self,
        circuit8,
        world_x: float,
        world_y: float,
        world_width: float = 500.0,
        world_height: float = 500.0,
        search_radius: int = 10
    ) -> list:
        """
        Find resource markers near position.
        
        Args:
            circuit8: Circuit8 canvas
            world_x, world_y: World coordinates
            world_width, world_height: World dimensions
            search_radius: How far to search in pixels
            
        Returns:
            List of resource markers with positions and types
        """
        canvas_x = int((world_x + world_width/2) / world_width * circuit8.width)
        canvas_y = int((world_y + world_height/2) / world_height * circuit8.height)
        
        canvas_x = max(0, min(circuit8.width - 1, canvas_x))
        canvas_y = max(0, min(circuit8.height - 1, canvas_y))
        
        markers = []
        
        # Search in radius
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                x = canvas_x + dx
                y = canvas_y + dy
                
                if 0 <= x < circuit8.width and 0 <= y < circuit8.height:
                    r, g, b = circuit8.read_pixel(x, y)
                    
                    # Detect resource markers
                    resource_type = None
                    if g > 180 and r < 50 and b < 50:
                        resource_type = 'food'
                    elif r > 180 and b > 180 and g < 50:
                        resource_type = 'drug'
                    
                    if resource_type:
                        # Convert back to world coordinates
                        world_marker_x = (x / circuit8.width) * world_width - world_width/2
                        world_marker_y = (y / circuit8.height) * world_height - world_height/2
                        
                        markers.append({
                            'type': resource_type,
                            'world_x': world_marker_x,
                            'world_y': world_marker_y,
                            'canvas_x': x,
                            'canvas_y': y,
                            'distance': np.sqrt(dx*dx + dy*dy)
                        })
        
        return markers


def fade_circuit8(circuit8, fade_rate: float = 0.95):
    """
    Fade Circuit8 over time (old signals disappear).
    
    Args:
        circuit8: Circuit8 canvas
        fade_rate: Fade multiplier (0.95 = 5% fade per step)
    """
    circuit8.screen = (circuit8.screen * fade_rate).astype(np.uint8)
