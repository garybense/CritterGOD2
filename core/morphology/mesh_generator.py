"""
Procedural 3D mesh generation from body genotype.

Generates OpenGL-ready vertex data for creature bodies:
- Body segments as cylinders
- Limbs as tapered cylinders
- Joints as spheres
- Head and tail

Performance optimizations:
- Display lists for repeated geometry
- Mesh caching by genotype signature
- Low-poly primitives (8-16 sides)
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import math
from OpenGL.GL import *

from core.morphology.body_genotype import BodyGenotype, SegmentGene, LimbGene


class Mesh3D:
    """Container for 3D mesh data.
    
    Attributes:
        vertices: Nx3 array of vertex positions
        normals: Nx3 array of vertex normals
        colors: Nx3 array of vertex colors (RGB 0-1)
        indices: Array of triangle indices
        display_list: Optional OpenGL display list ID
    """
    
    def __init__(self, vertices: np.ndarray, normals: np.ndarray,
                 colors: np.ndarray, indices: np.ndarray):
        self.vertices = vertices
        self.normals = normals
        self.colors = colors
        self.indices = indices
        self.display_list: Optional[int] = None
    
    def compile_display_list(self):
        """Compile mesh into OpenGL display list for faster rendering."""
        if self.display_list is not None:
            return  # Already compiled
        
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        
        glBegin(GL_TRIANGLES)
        for idx in self.indices:
            glNormal3fv(self.normals[idx])
            glColor3fv(self.colors[idx])
            glVertex3fv(self.vertices[idx])
        glEnd()
        
        glEndList()
    
    def render(self):
        """Render mesh using display list if available."""
        if self.display_list is not None:
            glCallList(self.display_list)
        else:
            # Fallback immediate mode rendering
            glBegin(GL_TRIANGLES)
            for idx in self.indices:
                glNormal3fv(self.normals[idx])
                glColor3fv(self.colors[idx])
                glVertex3fv(self.vertices[idx])
            glEnd()
    
    def delete(self):
        """Free OpenGL display list."""
        if self.display_list is not None:
            glDeleteLists(self.display_list, 1)
            self.display_list = None


class ProceduralMeshGenerator:
    """Generates 3D meshes from body genotypes.
    
    Uses low-poly primitives for performance:
    - Cylinders: 8 sides
    - Spheres: 8x8 lat/lon grid
    """
    
    def __init__(self, cylinder_sides: int = 8, sphere_detail: int = 8):
        """Initialize mesh generator.
        
        Args:
            cylinder_sides: Number of sides for cylinder primitives
            sphere_detail: Detail level for sphere primitives (lat/lon subdivisions)
        """
        self.cylinder_sides = cylinder_sides
        self.sphere_detail = sphere_detail
        self.mesh_cache: Dict[str, Mesh3D] = {}
    
    def generate_creature_mesh(self, body: BodyGenotype) -> Mesh3D:
        """Generate complete creature mesh from body genotype.
        
        Args:
            body: Body genotype to generate mesh from
            
        Returns:
            Complete creature mesh
        """
        # Check cache first
        signature = body.get_signature()
        if signature in self.mesh_cache:
            return self.mesh_cache[signature]
        
        # Generate mesh parts
        all_vertices = []
        all_normals = []
        all_colors = []
        all_indices = []
        vertex_offset = 0
        
        # Generate head
        if len(body.segments) > 0:
            head_radius = body.segments[0].size * body.head_size
            head_verts, head_norms, head_colors, head_inds = self._generate_sphere(
                radius=head_radius,
                position=(0, 0, 0),
                color=self._hue_to_rgb(body.base_hue)
            )
            all_vertices.append(head_verts)
            all_normals.append(head_norms)
            all_colors.append(head_colors)
            all_indices.append(head_inds + vertex_offset)
            vertex_offset += len(head_verts)
        
        # Generate body segments
        z_position = 0.0
        for seg_idx, segment in enumerate(body.segments):
            # Segment position
            seg_start_z = z_position
            seg_end_z = z_position + segment.length * segment.size
            
            # Generate segment cylinder
            seg_verts, seg_norms, seg_colors, seg_inds = self._generate_cylinder(
                radius=segment.size,
                height=segment.length * segment.size,
                position=(0, 0, seg_start_z),
                color=self._hue_to_rgb(body.base_hue, offset=seg_idx * 10)
            )
            all_vertices.append(seg_verts)
            all_normals.append(seg_norms)
            all_colors.append(seg_colors)
            all_indices.append(seg_inds + vertex_offset)
            vertex_offset += len(seg_verts)
            
            # Generate limbs attached to this segment
            limb_attach_z = (seg_start_z + seg_end_z) / 2
            for limb in segment.limbs:
                limb_verts, limb_norms, limb_colors, limb_inds = self._generate_limb(
                    limb=limb,
                    segment_size=segment.size,
                    attach_position=(0, 0, limb_attach_z),
                    color=self._hue_to_rgb(body.base_hue, offset=seg_idx * 10 + 5)
                )
                all_vertices.append(limb_verts)
                all_normals.append(limb_norms)
                all_colors.append(limb_colors)
                all_indices.append(limb_inds + vertex_offset)
                vertex_offset += len(limb_verts)
            
            z_position = seg_end_z
        
        # Generate tail
        if body.tail_length > 0 and len(body.segments) > 0:
            tail_radius = body.segments[-1].size * 0.5
            tail_verts, tail_norms, tail_colors, tail_inds = self._generate_cone(
                base_radius=tail_radius,
                tip_radius=tail_radius * 0.1,
                height=body.tail_length * tail_radius,
                position=(0, 0, z_position),
                color=self._hue_to_rgb(body.base_hue, offset=50)
            )
            all_vertices.append(tail_verts)
            all_normals.append(tail_norms)
            all_colors.append(tail_colors)
            all_indices.append(tail_inds + vertex_offset)
        
        # Combine all mesh parts
        vertices = np.vstack(all_vertices) if all_vertices else np.zeros((0, 3))
        normals = np.vstack(all_normals) if all_normals else np.zeros((0, 3))
        colors = np.vstack(all_colors) if all_colors else np.zeros((0, 3))
        indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=int)
        
        mesh = Mesh3D(vertices, normals, colors, indices)
        mesh.compile_display_list()
        
        # Cache mesh
        self.mesh_cache[signature] = mesh
        
        return mesh
    
    def _generate_sphere(self, radius: float, position: Tuple[float, float, float],
                        color: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate sphere mesh (for head, joints)."""
        vertices = []
        normals = []
        colors = []
        indices = []
        
        # Generate vertices (lat/lon grid)
        for lat in range(self.sphere_detail + 1):
            theta = (lat / self.sphere_detail) * math.pi  # 0 to pi
            for lon in range(self.sphere_detail + 1):
                phi = (lon / self.sphere_detail) * 2 * math.pi  # 0 to 2pi
                
                # Sphere coordinates
                x = radius * math.sin(theta) * math.cos(phi)
                y = radius * math.sin(theta) * math.sin(phi)
                z = radius * math.cos(theta)
                
                # Apply position offset
                vertices.append([x + position[0], y + position[1], z + position[2]])
                
                # Normal (pointing outward)
                normal = np.array([x, y, z])
                if np.linalg.norm(normal) > 0:
                    normal /= np.linalg.norm(normal)
                normals.append(normal)
                
                colors.append(color)
        
        # Generate triangle indices
        for lat in range(self.sphere_detail):
            for lon in range(self.sphere_detail):
                # Two triangles per quad
                i0 = lat * (self.sphere_detail + 1) + lon
                i1 = i0 + 1
                i2 = (lat + 1) * (self.sphere_detail + 1) + lon
                i3 = i2 + 1
                
                indices.extend([i0, i2, i1, i1, i2, i3])
        
        return (np.array(vertices), np.array(normals), 
                np.array(colors), np.array(indices, dtype=int))
    
    def _generate_cylinder(self, radius: float, height: float,
                          position: Tuple[float, float, float],
                          color: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate cylinder mesh (for body segments)."""
        vertices = []
        normals = []
        colors = []
        indices = []
        
        # Generate vertices (two circles + sides)
        for i in range(self.cylinder_sides + 1):
            angle = (i / self.cylinder_sides) * 2 * math.pi
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            # Bottom circle
            vertices.append([x + position[0], y + position[1], position[2]])
            normals.append([x/radius, y/radius, 0])
            colors.append(color)
            
            # Top circle
            vertices.append([x + position[0], y + position[1], position[2] + height])
            normals.append([x/radius, y/radius, 0])
            colors.append(color)
        
        # Generate side triangles
        for i in range(self.cylinder_sides):
            i0 = i * 2
            i1 = i0 + 1
            i2 = (i + 1) * 2
            i3 = i2 + 1
            
            indices.extend([i0, i2, i1, i1, i2, i3])
        
        # Add caps (optional for performance)
        # ...caps would go here if needed...
        
        return (np.array(vertices), np.array(normals),
                np.array(colors), np.array(indices, dtype=int))
    
    def _generate_cone(self, base_radius: float, tip_radius: float, height: float,
                      position: Tuple[float, float, float],
                      color: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate tapered cylinder/cone mesh (for tail)."""
        vertices = []
        normals = []
        colors = []
        indices = []
        
        for i in range(self.cylinder_sides + 1):
            angle = (i / self.cylinder_sides) * 2 * math.pi
            
            # Base
            x_base = base_radius * math.cos(angle)
            y_base = base_radius * math.sin(angle)
            vertices.append([x_base + position[0], y_base + position[1], position[2]])
            normals.append([x_base/base_radius, y_base/base_radius, 0])
            colors.append(color)
            
            # Tip
            x_tip = tip_radius * math.cos(angle)
            y_tip = tip_radius * math.sin(angle)
            vertices.append([x_tip + position[0], y_tip + position[1], position[2] + height])
            normals.append([x_tip/tip_radius if tip_radius > 0 else 0,
                          y_tip/tip_radius if tip_radius > 0 else 0, 0])
            colors.append(color)
        
        # Generate triangles
        for i in range(self.cylinder_sides):
            i0 = i * 2
            i1 = i0 + 1
            i2 = (i + 1) * 2
            i3 = i2 + 1
            
            indices.extend([i0, i2, i1, i1, i2, i3])
        
        return (np.array(vertices), np.array(normals),
                np.array(colors), np.array(indices, dtype=int))
    
    def _generate_limb(self, limb: LimbGene, segment_size: float,
                      attach_position: Tuple[float, float, float],
                      color: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate limb mesh (tapered cylinder with rotation)."""
        # Calculate limb dimensions
        limb_length = limb.length * segment_size
        base_radius = limb.width * segment_size
        tip_radius = base_radius * limb.taper
        
        # Generate cone/cylinder
        verts, norms, cols, inds = self._generate_cone(
            base_radius=base_radius,
            tip_radius=tip_radius,
            height=limb_length,
            position=(0, 0, 0),  # Will transform
            color=color
        )
        
        # Apply limb rotation
        # 1. Rotate around Z axis (horizontal angle)
        angle_h_rad = math.radians(limb.angle_horizontal)
        cos_h = math.cos(angle_h_rad)
        sin_h = math.sin(angle_h_rad)
        
        # 2. Rotate around Y axis (vertical angle)
        angle_v_rad = math.radians(limb.angle_vertical)
        cos_v = math.cos(angle_v_rad)
        sin_v = math.sin(angle_v_rad)
        
        # Apply rotations and translation
        for i in range(len(verts)):
            x, y, z = verts[i]
            
            # Rotate vertical (tilt limb up/down)
            x_new = x * cos_v - z * sin_v
            z_new = x * sin_v + z * cos_v
            x = x_new
            z = z_new
            
            # Rotate horizontal (around body)
            x_new = x * cos_h - y * sin_h
            y_new = x * sin_h + y * cos_h
            x = x_new
            y = y_new
            
            # Translate to attachment point
            verts[i] = [x + attach_position[0], y + attach_position[1], z + attach_position[2]]
        
        return verts, norms, cols, inds
    
    def _hue_to_rgb(self, hue: float, offset: float = 0, saturation: float = 0.7,
                    value: float = 0.8) -> Tuple[float, float, float]:
        """Convert HSV to RGB for creature coloring.
        
        Args:
            hue: Hue value (0-360)
            offset: Hue offset for variation
            saturation: Color saturation (0-1)
            value: Color brightness (0-1)
            
        Returns:
            RGB tuple (0-1 range)
        """
        h = ((hue + offset) % 360) / 60.0
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (r + m, g + m, b + m)
    
    def clear_cache(self):
        """Clear mesh cache and free display lists."""
        for mesh in self.mesh_cache.values():
            mesh.delete()
        self.mesh_cache.clear()
