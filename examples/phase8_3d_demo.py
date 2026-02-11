"""
Phase 8: 3D OpenGL Visualization Demo

Simple test of 3D rendering before full Phase 7 integration.
Tests camera, spheres, ground plane, and Circuit8 texture.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pygame
from pygame.locals import *
from OpenGL.GL import *

import numpy as np
from examples.phase7_complete import Phase7Ecosystem
from visualization.opengl_renderer import OpenGL3DRenderer


def main():
    """Run Phase 8 3D visualization test."""
    print("\n" + "=" * 80)
    print("  PHASE 8: 3D OPENGL VISUALIZATION TEST")
    print("=" * 80)
    print("\nInitializing...")
    
    # Create ecosystem
    ecosystem = Phase7Ecosystem(
        initial_population=10,
        max_population=30,
        world_width=1400,
        world_height=900
    )
    
    # Initialize pygame with OpenGL
    pygame.init()
    width, height = 1400, 900
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("CritterGOD Phase 8 - 3D View")
    
    # Create 3D renderer
    renderer = OpenGL3DRenderer(ecosystem, width, height)
    renderer.setup_opengl()
    
    clock = pygame.time.Clock()
    running = True
    paused = False
    
    print("\n" + "=" * 80)
    print("  3D VIEW CONTROLS")
    print("=" * 80)
    print("\nCamera:")
    print("  Left Mouse + Drag - Rotate camera")
    print("  Mouse Wheel - Zoom in/out")
    print("  Arrow Keys - Pan camera")
    print("  R - Reset camera")
    print("\nSimulation:")
    print("  SPACE - Pause/Unpause")
    print("  D - Drop drugs")
    print("  ESC - Quit")
    print("\nStarting 3D visualization...\n")
    
    timestep = 0
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_d:
                    ecosystem._scatter_drugs(5)
            
            # Let renderer handle mouse events
            renderer.handle_mouse_event(event)
        
        # Handle keyboard (for continuous input like arrow keys)
        keys = pygame.key.get_pressed()
        renderer.handle_keyboard(keys)
        
        # Update ecosystem
        if not paused:
            ecosystem.update(dt=1.0)
            timestep += 1
        
        # Render 3D scene
        renderer.render_3d_scene()
        
        # Swap buffers
        pygame.display.flip()
        clock.tick(30)
    
    # Cleanup
    renderer.cleanup()
    pygame.quit()
    
    # Final stats
    print("\n" + "=" * 80)
    print("  PHASE 8 TEST COMPLETE")
    print("=" * 80)
    print(f"\nTimesteps: {timestep}")
    print(f"Final population: {len(ecosystem.creatures)}")
    print(f"Births: {ecosystem.total_births}")
    print(f"Deaths: {ecosystem.total_deaths}")
    print("\n3D OpenGL rendering verified!")
    print("Ready for full Phase 7 integration.\n")


if __name__ == "__main__":
    main()
