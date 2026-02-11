"""
OpenGL 3D Renderer for CritterGOD Phase 8.

Renders artificial life ecosystem in 3D with orbital camera,
maintaining all psychedelic computing features.
"""

import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

from visualization.camera import OrbitalCamera
from visualization.gl_primitives import (
    SpherePrimitive, 
    GroundPlane, 
    Circuit8Texture,
    setup_lighting
)
from core.morphology.mesh_generator import ProceduralMeshGenerator


class OpenGL3DRenderer:
    """
    3D OpenGL renderer for CritterGOD ecosystem.
    
    Renders creatures as spheres, Circuit8 as textured ground,
    with full camera controls and lighting.
    """
    
    def __init__(self, ecosystem, width=1400, height=900):
        """
        Initialize 3D renderer.
        
        Args:
            ecosystem: Phase7Ecosystem instance
            width: Window width
            height: Window height
        """
        self.ecosystem = ecosystem
        self.width = width
        self.height = height
        
        # Camera
        world_center_x = ecosystem.world_width / 2.0
        world_center_z = ecosystem.world_height / 2.0
        self.camera = OrbitalCamera(
            target=(world_center_x, 0.0, world_center_z),
            distance=800.0,
            azimuth=45.0,
            elevation=35.0
        )
        
        # 3D primitives
        self.sphere = SpherePrimitive(slices=12, stacks=12)
        self.ground = GroundPlane(size=2000.0, grid_spacing=100.0)
        self.circuit8_tex = Circuit8Texture(
            width=ecosystem.circuit8.width,
            height=ecosystem.circuit8.height
        )
        
        # Mesh generator for procedural creatures
        self.mesh_generator = ProceduralMeshGenerator(cylinder_sides=8, sphere_detail=8)
        
        # Mouse state for camera control
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)
        
        print("✓ 3D OpenGL renderer initialized")
    
    def setup_opengl(self):
        """Setup OpenGL state."""
        glViewport(0, 0, self.width, self.height)
        glClearColor(0.05, 0.05, 0.1, 1.0)  # Dark blue background
        setup_lighting()
    
    def handle_mouse_event(self, event):
        """
        Handle mouse events for camera control.
        
        Args:
            event: pygame event
            
        Returns:
            True if event was handled
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                self.mouse_dragging = True
                self.last_mouse_pos = pygame.mouse.get_pos()
                return True
            elif event.button == 4:  # Mouse wheel up
                self.camera.handle_zoom(1)
                return True
            elif event.button == 5:  # Mouse wheel down
                self.camera.handle_zoom(-1)
                return True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.mouse_dragging = False
                return True
        
        elif event.type == pygame.MOUSEMOTION:
            if self.mouse_dragging:
                mouse_pos = pygame.mouse.get_pos()
                dx = mouse_pos[0] - self.last_mouse_pos[0]
                dy = mouse_pos[1] - self.last_mouse_pos[1]
                self.camera.handle_mouse_drag(dx, dy)
                self.last_mouse_pos = mouse_pos
                return True
        
        return False
    
    def handle_keyboard(self, keys):
        """
        Handle keyboard input for camera panning.
        
        Args:
            keys: pygame.key.get_pressed() result
        """
        # Arrow keys for panning
        if keys[pygame.K_LEFT]:
            self.camera.handle_pan(-1, 0)
        if keys[pygame.K_RIGHT]:
            self.camera.handle_pan(1, 0)
        if keys[pygame.K_UP]:
            self.camera.handle_pan(0, -1)
        if keys[pygame.K_DOWN]:
            self.camera.handle_pan(0, 1)
        
        # R to reset camera
        if keys[pygame.K_r]:
            self.camera.reset()
    
    def render_3d_scene(self):
        """Render complete 3D scene."""
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Setup camera
        self.camera.apply_projection(self.width, self.height)
        self.camera.apply_view()
        
        # Render ground
        self.ground.render()
        
        # Render Circuit8 as textured plane
        self.circuit8_tex.update_from_circuit8(self.ecosystem.circuit8)
        self.circuit8_tex.render_as_ground(size=600.0, y=0.5)
        
        # Render creatures
        for creature in self.ecosystem.creatures:
            # Position (map 2D world to 3D, with Y for height)
            x = creature.x
            z = creature.y  # 2D y becomes 3D z
            y = 10.0  # Height above ground
            
            # Check if creature has 3D body
            if hasattr(creature, 'body'):
                # Render procedural mesh
                self._render_morphological_creature(creature, x, y, z)
            else:
                # Fallback to sphere rendering
                # Size based on energy
                radius = 5.0 + min(creature.energy.energy / 100000.0, 15.0)
                
                # Color based on evolutionary generation
                depth = min(creature.adam_distance, 10)
                hue = (depth * 36) % 360
                h = hue / 60.0
                c = 200
                x_val = c * (1 - abs(h % 2 - 1))
                
                if h < 1:
                    r, g, b = c, x_val, 0
                elif h < 2:
                    r, g, b = x_val, c, 0
                elif h < 3:
                    r, g, b = 0, c, x_val
                elif h < 4:
                    r, g, b = 0, x_val, c
                elif h < 5:
                    r, g, b = x_val, 0, c
                else:
                    r, g, b = c, 0, x_val
                
                color = (int(r) + 55, int(g) + 55, int(b) + 55)
                
                # Render sphere
                self.sphere.render(x, y, z, radius, color)
        
        # Render drug pills
        for pill in self.ecosystem.pills:
            # Determine dominant molecule for color
            dominant_molecule = max(range(5), key=lambda i: pill.molecule_composition[i])
            colors = [
                (200, 100, 200),  # Inhibitory antagonist
                (150, 100, 250),  # Inhibitory agonist
                (100, 200, 100),  # Excitatory antagonist
                (100, 250, 150),  # Excitatory agonist
                (255, 255, 100),  # Potentiator
            ]
            color = colors[dominant_molecule]
            
            # Render as small sphere
            self.sphere.render(pill.x, 2.0, pill.y, 3.0, color)
    
    def _render_morphological_creature(self, creature, x, y, z):
        """Render creature with procedural 3D body mesh.
        
        Args:
            creature: MorphologicalCreature instance
            x, y, z: World position
        """
        # Generate or retrieve cached mesh
        if creature.mesh is None:
            creature.mesh = self.mesh_generator.generate_creature_mesh(creature.body)
        
        # Get body scale (pulsing from drugs)
        scale = creature.get_render_scale() if hasattr(creature, 'get_render_scale') else 1.0
        
        # Get drug-modified color
        if hasattr(creature, 'get_body_color_with_drugs'):
            r, g, b = creature.get_body_color_with_drugs()
            color = (int(r * 255), int(g * 255), int(b * 255))
        else:
            # Fallback color
            color = (200, 150, 100)
        
        # Apply transformations
        glPushMatrix()
        
        # Translate to world position
        glTranslatef(x, y, z)
        
        # Scale body
        glScalef(scale, scale, scale)
        
        # Apply base body scale (normalize to similar size as spheres)
        body_scale = 5.0
        glScalef(body_scale, body_scale, body_scale)
        
        # Render mesh
        creature.mesh.render()
        
        glPopMatrix()
    
    def begin_2d_overlay(self):
        """Switch to 2D orthographic projection for UI overlay."""
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable 3D features for 2D rendering
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glDisable(GL_CULL_FACE)
    
    def end_2d_overlay(self):
        """Restore 3D projection after UI overlay."""
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        
        # Re-enable 3D features
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_CULL_FACE)
    
    def cleanup(self):
        """Cleanup OpenGL resources."""
        self.sphere.cleanup()
        self.circuit8_tex.cleanup()
