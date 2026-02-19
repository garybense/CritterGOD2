"""
Circuit8: The Telepathic Canvas

A 64x48 pixel shared screen that ALL creatures can read and write.
This is the collective unconscious - morphic resonance made real.

Based on telepathic-critterdrug's Circuit8 implementation.
"""

import numpy as np
from typing import Tuple, Optional
from enum import IntEnum


class ColorChannel(IntEnum):
    """RGB color channels in Circuit8."""
    RED = 0
    GREEN = 1
    BLUE = 2


class Circuit8:
    """
    The telepathic canvas - a shared perception space.
    
    Key features from telepathic-critterdrug:
    - 64x48 pixels (3,072 pixels total)
    - 1024 depth layers per pixel (temporal/spatial buffer)
    - RGB channels (each creature affects the collective field)
    - All creatures can read AND write
    - Acts as collective memory/unconscious
    
    This is morphic resonance (Rupert Sheldrake) implemented in code.
    
    Attributes:
        width: Screen width (64)
        height: Screen height (48)
        depth: Depth layers per pixel (1024)
        screen: RGB values [height, width, 3]
        depth_buffer: Temporal buffer [height, width, depth, 3]
    """
    
    def __init__(self, width: int = 64, height: int = 48, depth: int = 1024):
        """
        Initialize the telepathic canvas.
        
        Args:
            width: Screen width (default 64 from telepathic-critterdrug)
            height: Screen height (default 48)
            depth: Depth layers (default 1024)
        """
        self.width = width
        self.height = height
        self.depth = depth
        
        # Main screen (what creatures see)
        self.screen = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Depth buffer (temporal/spatial memory)
        self.depth_buffer = np.zeros((height, width, depth, 3), dtype=np.uint8)
        
        # Current depth layer being written
        self.current_depth = 0
        
        # Voting system for screen movement
        self.votes_up = 0
        self.votes_down = 0
        self.votes_left = 0
        self.votes_right = 0
        
    def read_pixel(self, x: int, y: int) -> Tuple[int, int, int]:
        """
        Read RGB values at position.
        
        Args:
            x: X coordinate (0-63)
            y: Y coordinate (0-47)
            
        Returns:
            (R, G, B) tuple
        """
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        
        return tuple(self.screen[y, x])
    
    def read_pixel_array(self, x: int, y: int) -> np.ndarray:
        """
        Read RGB values at position as numpy array (no tuple allocation).
        
        Args:
            x: X coordinate (0-63)
            y: Y coordinate (0-47)
            
        Returns:
            numpy array [R, G, B]
        """
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        return self.screen[y, x]
    
    def read_pixels_batch(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Batch read RGB values at multiple positions (vectorized).
        
        Args:
            xs: Array of X coordinates
            ys: Array of Y coordinates
            
        Returns:
            Array of shape (N, 3) with RGB values
        """
        xs = np.clip(xs, 0, self.width - 1)
        ys = np.clip(ys, 0, self.height - 1)
        return self.screen[ys, xs]
        
    def write_pixel(self, x: int, y: int, r: int, g: int, b: int, blend: bool = True):
        """
        Write RGB values at position.
        
        Motor neurons from creatures write here.
        
        Args:
            x: X coordinate
            y: Y coordinate  
            r: Red value (0-255)
            g: Green value (0-255)
            b: Blue value (0-255)
            blend: Blend with existing values (default True)
        """
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        
        if blend:
            # Blend with existing (creatures affect each other)
            self.screen[y, x, 0] = min(255, int(self.screen[y, x, 0]) + int(r)) // 2
            self.screen[y, x, 1] = min(255, int(self.screen[y, x, 1]) + int(g)) // 2
            self.screen[y, x, 2] = min(255, int(self.screen[y, x, 2]) + int(b)) // 2
        else:
            # Direct write
            self.screen[y, x, 0] = r
            self.screen[y, x, 1] = g
            self.screen[y, x, 2] = b
            
    def modify_channel(
        self,
        x: int,
        y: int,
        channel: ColorChannel,
        delta: int
    ):
        """
        Modify a specific color channel (condensed motor operation).
        
        From telepathic-critterdrug: moreRed, lessRed, moreGreen, etc.
        
        Args:
            x: X coordinate
            y: Y coordinate
            channel: Which channel (R/G/B)
            delta: Change amount (positive or negative)
        """
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        
        current = self.screen[y, x, channel]
        new_value = max(0, min(255, current + delta))
        self.screen[y, x, channel] = new_value
        
    def update_depth_buffer(self):
        """
        Push current screen to depth buffer (temporal memory).
        
        This creates the 1024-layer temporal buffer - the canvas remembers.
        """
        # Shift depth buffer
        self.depth_buffer[:, :, 1:, :] = self.depth_buffer[:, :, :-1, :]
        
        # Add current screen to front
        self.depth_buffer[:, :, 0, :] = self.screen
        
    def read_depth(self, x: int, y: int, depth: int) -> Tuple[int, int, int]:
        """
        Read from depth buffer (temporal memory).
        
        Args:
            x: X coordinate
            y: Y coordinate
            depth: Depth layer (0 = most recent, 1023 = oldest)
            
        Returns:
            (R, G, B) from that time
        """
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        depth = max(0, min(self.depth - 1, depth))
        
        return tuple(self.depth_buffer[y, x, depth])
        
    def vote_movement(self, direction: str, weight: int = 1):
        """
        Vote for screen movement (collective decision).
        
        Supports weighted voting tiers (from telepathic-critterdrug):
        - Tier 1 (weight=1): normal vote, default
        - Tier 2 (weight=3): strong motor output
        - Tier 3 (weight=7): very strong motor output
        
        Creatures with stronger neural conviction get louder voices
        in the collective democratic process.
        
        Args:
            direction: 'up', 'down', 'left', 'right'
            weight: Vote weight (default 1)
        """
        if direction == 'up':
            self.votes_up += weight
        elif direction == 'down':
            self.votes_down += weight
        elif direction == 'left':
            self.votes_left += weight
        elif direction == 'right':
            self.votes_right += weight
            
    def apply_voted_movement(self):
        """
        Apply movement based on collective voting.
        
        This is emergent democracy - the canvas moves based on collective will.
        """
        # Determine dominant direction
        if self.votes_up > max(self.votes_down, self.votes_left, self.votes_right):
            self.scroll(0, -1)
        elif self.votes_down > max(self.votes_up, self.votes_left, self.votes_right):
            self.scroll(0, 1)
        elif self.votes_left > max(self.votes_up, self.votes_down, self.votes_right):
            self.scroll(-1, 0)
        elif self.votes_right > max(self.votes_up, self.votes_down, self.votes_left):
            self.scroll(1, 0)
            
        # Reset votes
        self.votes_up = 0
        self.votes_down = 0
        self.votes_left = 0
        self.votes_right = 0
        
    def scroll(self, dx: int, dy: int):
        """
        Scroll the canvas (shift all pixels).
        
        Args:
            dx: Horizontal shift
            dy: Vertical shift
        """
        self.screen = np.roll(self.screen, (dy, dx), axis=(0, 1))
        
    def decay(self, rate: float = 0.95):
        """
        Decay all pixel values (fade over time).
        
        Args:
            rate: Decay rate (0.95 = 5% fade)
        """
        self.screen = (self.screen * rate).astype(np.uint8)
        
    def get_average_color(self) -> Tuple[float, float, float]:
        """
        Get average color across entire canvas.
        
        Returns:
            (avg_R, avg_G, avg_B)
        """
        return (
            float(np.mean(self.screen[:, :, 0])),
            float(np.mean(self.screen[:, :, 1])),
            float(np.mean(self.screen[:, :, 2]))
        )
        
    def get_local_average(self, x: int, y: int, radius: int = 3) -> Tuple[float, float, float]:
        """
        Get average color in local region.
        
        Args:
            x: Center X
            y: Center Y
            radius: Radius around center
            
        Returns:
            (avg_R, avg_G, avg_B) in region
        """
        x1 = max(0, x - radius)
        x2 = min(self.width, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(self.height, y + radius + 1)
        
        region = self.screen[y1:y2, x1:x2]
        
        return (
            float(np.mean(region[:, :, 0])),
            float(np.mean(region[:, :, 1])),
            float(np.mean(region[:, :, 2]))
        )
        
    def global_color_shift(self, r_delta: int, g_delta: int, b_delta: int):
        """
        Apply a global color shift to the entire canvas.
        
        From telepathic-critterdrug condensed colour motors:
        When a creature's screen motor fires very strongly, it affects
        the entire shared perception field — not just its pixel.
        
        Args:
            r_delta: Red channel shift (-255 to 255)
            g_delta: Green channel shift (-255 to 255)
            b_delta: Blue channel shift (-255 to 255)
        """
        # Use int16 to avoid overflow, then clip
        shifted = self.screen.astype(np.int16)
        shifted[:, :, 0] += r_delta
        shifted[:, :, 1] += g_delta
        shifted[:, :, 2] += b_delta
        self.screen = np.clip(shifted, 0, 255).astype(np.uint8)
    
    def clear(self):
        """Clear the canvas to black."""
        self.screen.fill(0)
        
    def __repr__(self) -> str:
        avg_r, avg_g, avg_b = self.get_average_color()
        return (
            f"Circuit8({self.width}x{self.height}, "
            f"depth={self.depth}, "
            f"avg_color=({avg_r:.0f}, {avg_g:.0f}, {avg_b:.0f}))"
        )
