"""
OpenGL primitives for 3D visualization.

Provides efficient rendering of spheres, planes, and other 3D shapes
for artificial life creatures and environments.
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *


class SpherePrimitive:
    """
    Reusable sphere primitive with display list caching.
    
    Spheres represent creatures in 3D space, colored by
    generation/species and sized by energy.
    """
    
    def __init__(self, slices=16, stacks=16):
        """
        Create sphere primitive.
        
        Args:
            slices: Number of longitude divisions
            stacks: Number of latitude divisions
        """
        self.slices = slices
        self.stacks = stacks
        self.quadric = gluNewQuadric()
        gluQuadricNormals(self.quadric, GLU_SMOOTH)
        gluQuadricTexture(self.quadric, GL_TRUE)
        
        # Display list for efficient rendering
        self.display_list = None
    
    def create_display_list(self):
        """Create display list for sphere geometry."""
        if self.display_list is None:
            self.display_list = glGenLists(1)
            glNewList(self.display_list, GL_COMPILE)
            gluSphere(self.quadric, 1.0, self.slices, self.stacks)
            glEndList()
    
    def render(self, x, y, z, radius, color):
        """
        Render sphere at position with color.
        
        Args:
            x, y, z: Position
            radius: Sphere radius
            color: (r, g, b) tuple (0-255 range)
        """
        if self.display_list is None:
            self.create_display_list()
        
        glPushMatrix()
        
        # Position and scale
        glTranslatef(x, y, z)
        glScalef(radius, radius, radius)
        
        # Material color
        r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
        glColor3f(r, g, b)
        
        # Set material properties for lighting
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [r, g, b, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 32.0)
        
        # Render
        glCallList(self.display_list)
        
        glPopMatrix()
    
    def cleanup(self):
        """Delete display list."""
        if self.display_list:
            glDeleteLists(self.display_list, 1)
        if self.quadric:
            gluDeleteQuadric(self.quadric)


class GroundPlane:
    """
    Ground plane with optional grid lines.
    
    Provides spatial reference in 3D space.
    """
    
    def __init__(self, size=2000.0, grid_spacing=100.0, show_grid=True):
        """
        Create ground plane.
        
        Args:
            size: Total size of plane
            grid_spacing: Distance between grid lines
            show_grid: Whether to draw grid lines
        """
        self.size = size
        self.grid_spacing = grid_spacing
        self.show_grid = show_grid
        self.ground_color = (0.1, 0.1, 0.15)  # Dark blue-gray
        self.grid_color = (0.2, 0.2, 0.3)  # Lighter blue-gray
    
    def render(self):
        """Render ground plane."""
        half_size = self.size / 2.0
        
        # Draw solid plane
        glDisable(GL_LIGHTING)
        glColor3f(*self.ground_color)
        glBegin(GL_QUADS)
        glNormal3f(0.0, 1.0, 0.0)
        glVertex3f(-half_size, 0.0, -half_size)
        glVertex3f(half_size, 0.0, -half_size)
        glVertex3f(half_size, 0.0, half_size)
        glVertex3f(-half_size, 0.0, half_size)
        glEnd()
        
        # Draw grid lines
        if self.show_grid:
            glColor3f(*self.grid_color)
            glBegin(GL_LINES)
            
            # Lines along X axis
            z = -half_size
            while z <= half_size:
                glVertex3f(-half_size, 0.01, z)
                glVertex3f(half_size, 0.01, z)
                z += self.grid_spacing
            
            # Lines along Z axis
            x = -half_size
            while x <= half_size:
                glVertex3f(x, 0.01, -half_size)
                glVertex3f(x, 0.01, half_size)
                x += self.grid_spacing
            
            glEnd()
        
        glEnable(GL_LIGHTING)


class Circuit8Texture:
    """
    Circuit8 telepathic canvas as OpenGL texture.
    
    Can be displayed as ground plane texture or floating billboard.
    """
    
    def __init__(self, width=64, height=48):
        """
        Create Circuit8 texture.
        
        Args:
            width: Circuit8 width in pixels
            height: Circuit8 height in pixels
        """
        self.width = width
        self.height = height
        self.texture_id = glGenTextures(1)
        self._setup_texture()
    
    def _setup_texture(self):
        """Setup texture parameters."""
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    
    def update_from_circuit8(self, circuit8):
        """
        Update texture from Circuit8 data.
        
        Args:
            circuit8: Circuit8 instance with screen numpy array
        """
        # Convert Circuit8 screen (height, width, 3) to texture format
        # Flip Y axis for OpenGL texture coordinates
        texture_data = np.flip(circuit8.screen, axis=0)
        texture_data = np.ascontiguousarray(texture_data, dtype=np.uint8)
        
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB,
            self.width, self.height, 0,
            GL_RGB, GL_UNSIGNED_BYTE, texture_data
        )
    
    def render_as_ground(self, size=500.0, y=0.5):
        """
        Render Circuit8 as textured ground plane.
        
        Args:
            size: Size of textured plane
            y: Height above ground
        """
        half_size = size / 2.0
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        # Disable lighting for texture
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-half_size, y, -half_size)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(half_size, y, -half_size)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(half_size, y, half_size)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-half_size, y, half_size)
        glEnd()
        
        glEnable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
    
    def cleanup(self):
        """Delete texture."""
        glDeleteTextures([self.texture_id])


def setup_lighting():
    """Setup standard lighting for 3D scene."""
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    
    # Ambient light
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
    
    # Diffuse light (sun from above/side)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
    
    # Specular highlights
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    
    # Light position (above and to the side)
    glLightfv(GL_LIGHT0, GL_POSITION, [1000.0, 2000.0, 1000.0, 0.0])
    
    # Enable depth testing
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    
    # Smooth shading
    glShadeModel(GL_SMOOTH)
    
    # Backface culling for performance
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)


def draw_sphere(radius: float, slices: int = 16, stacks: int = 16) -> None:
    """
    Draw a sphere at current position.
    
    Args:
        radius: Sphere radius
        slices: Number of longitude divisions
        stacks: Number of latitude divisions
    """
    quadric = gluNewQuadric()
    gluQuadricNormals(quadric, GLU_SMOOTH)
    gluSphere(quadric, radius, slices, stacks)
    gluDeleteQuadric(quadric)


def draw_cylinder(base_radius: float, top_radius: float, height: float, slices: int = 16) -> None:
    """
    Draw a cylinder at current position.
    
    Args:
        base_radius: Radius at base
        top_radius: Radius at top
        height: Cylinder height
        slices: Number of subdivisions around axis
    """
    quadric = gluNewQuadric()
    gluQuadricNormals(quadric, GLU_SMOOTH)
    gluCylinder(quadric, base_radius, top_radius, height, slices, 1)
    gluDeleteQuadric(quadric)
