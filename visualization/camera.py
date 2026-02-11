"""
3D Camera system for OpenGL visualization.

Implements orbital camera with spherical coordinates for natural
rotation around artificial life scenes.
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *


class OrbitalCamera:
    """
    Orbital camera using spherical coordinates.
    
    Camera orbits around a target point, useful for viewing
    artificial life simulations from all angles.
    
    Controls:
    - Mouse drag: Rotate around target
    - Mouse wheel: Zoom in/out
    - Arrow keys: Pan target
    
    Attributes:
        target: Point camera looks at (x, y, z)
        distance: Distance from target
        azimuth: Horizontal rotation angle (degrees)
        elevation: Vertical rotation angle (degrees)
        fov: Field of view (degrees)
        near/far: Clipping planes
    """
    
    def __init__(
        self,
        target=(0.0, 0.0, 0.0),
        distance=500.0,
        azimuth=45.0,
        elevation=30.0,
        fov=60.0,
        near=1.0,
        far=10000.0
    ):
        """
        Initialize orbital camera.
        
        Args:
            target: (x, y, z) point to look at
            distance: Distance from target
            azimuth: Horizontal angle (0-360)
            elevation: Vertical angle (-89 to 89)
            fov: Field of view in degrees
            near: Near clipping plane
            far: Far clipping plane
        """
        self.target = np.array(target, dtype=np.float32)
        self.distance = float(distance)
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)
        self.fov = float(fov)
        self.near = float(near)
        self.far = float(far)
        
        # For smooth mouse control
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_sensitivity = 0.3
        self.zoom_sensitivity = 20.0
        self.pan_sensitivity = 5.0
    
    def get_position(self):
        """
        Calculate camera position from spherical coordinates.
        
        Returns:
            (x, y, z) camera position
        """
        # Convert to radians
        az_rad = np.radians(self.azimuth)
        el_rad = np.radians(self.elevation)
        
        # Spherical to Cartesian
        x = self.distance * np.cos(el_rad) * np.cos(az_rad)
        y = self.distance * np.sin(el_rad)
        z = self.distance * np.cos(el_rad) * np.sin(az_rad)
        
        return self.target + np.array([x, y, z])
    
    def apply_projection(self, width, height):
        """
        Apply perspective projection matrix.
        
        Args:
            width: Viewport width
            height: Viewport height
        """
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = width / height if height > 0 else 1.0
        gluPerspective(self.fov, aspect, self.near, self.far)
    
    def apply_view(self):
        """Apply camera view transformation (lookAt)."""
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        pos = self.get_position()
        gluLookAt(
            pos[0], pos[1], pos[2],  # Eye position
            self.target[0], self.target[1], self.target[2],  # Look at
            0.0, 1.0, 0.0  # Up vector
        )
    
    def handle_mouse_drag(self, dx, dy):
        """
        Rotate camera based on mouse movement.
        
        Args:
            dx: Mouse delta X (pixels)
            dy: Mouse delta Y (pixels)
        """
        self.azimuth += dx * self.mouse_sensitivity
        self.elevation -= dy * self.mouse_sensitivity
        
        # Keep azimuth in 0-360 range
        self.azimuth = self.azimuth % 360.0
        
        # Clamp elevation to prevent gimbal lock
        self.elevation = np.clip(self.elevation, -89.0, 89.0)
    
    def handle_zoom(self, delta):
        """
        Zoom camera in/out.
        
        Args:
            delta: Zoom amount (positive = zoom in)
        """
        self.distance -= delta * self.zoom_sensitivity
        self.distance = np.clip(self.distance, 10.0, 5000.0)
    
    def handle_pan(self, dx, dy):
        """
        Pan target position.
        
        Args:
            dx: Pan delta X
            dy: Pan delta Y
        """
        # Calculate right and up vectors in camera space
        az_rad = np.radians(self.azimuth)
        
        # Right vector (perpendicular to view direction)
        right_x = -np.sin(az_rad)
        right_z = np.cos(az_rad)
        
        # Pan in camera space
        self.target[0] += (right_x * dx - np.cos(az_rad) * dy) * self.pan_sensitivity
        self.target[2] += (right_z * dx - np.sin(az_rad) * dy) * self.pan_sensitivity
    
    def reset(self):
        """Reset camera to default position."""
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.distance = 500.0
        self.azimuth = 45.0
        self.elevation = 30.0
    
    def __repr__(self):
        pos = self.get_position()
        return (
            f"OrbitalCamera(pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}), "
            f"target={tuple(self.target)}, dist={self.distance:.1f}, "
            f"az={self.azimuth:.1f}°, el={self.elevation:.1f}°)"
        )
