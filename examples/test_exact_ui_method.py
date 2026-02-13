"""
Test the EXACT rendering method used by research platform.
3D scene, then texture-based 2D overlay.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
import os
import subprocess

print("\n" + "="*60)
print("TESTING RESEARCH PLATFORM RENDERING METHOD")
print("="*60)
print("\nThis mimics EXACTLY what research_platform.py does:")
print("1. Render 3D scene")
print("2. Create pygame surface with colored rectangles")
print("3. Upload surface as OpenGL texture")
print("4. Draw textured quad over 3D scene")
print()
input("Press ENTER to start (window will open for 10 seconds)...")

# Initialize
pygame.init()
os.environ['SDL_VIDEO_WINDOW_POS'] = "50,50"
screen = pygame.display.set_mode((1600, 900), DOUBLEBUF | OPENGL)
pygame.display.set_caption("🔴 UI Overlay Test 🔴")

# Notification
import subprocess
subprocess.run([
    'osascript', '-e',
    'display notification "Look for window with 3D cube and colored panels!" with title "Test Started"'
], check=False)

clock = pygame.time.Clock()
angle = 0
start_time = time.time()

print("\n🔴 WINDOW IS OPEN!")
print("You should see:")
print("  - Spinning 3D CUBE in center")
print("  - BLUE panel on LEFT side")
print("  - ORANGE panel on RIGHT side")
print("  - GREEN panel at BOTTOM")
print("\nWindow will stay open for 10 seconds...")

while time.time() - start_time < 10:
    # Handle events
    for event in pygame.event.get():
        if event.type == QUIT:
            break
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
            break
    
    # === 3D SCENE ===
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # 3D projection
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1600/900, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    
    # Simple lighting
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [5, 5, 5, 1])
    
    glTranslatef(0, 0, -5)
    glRotatef(angle, 1, 1, 1)
    
    # Draw colorful cube
    glBegin(GL_QUADS)
    # Front (red)
    glColor3f(1, 0, 0)
    glVertex3f(-1, -1, 1)
    glVertex3f(1, -1, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)
    # Other faces...
    glColor3f(0, 1, 0)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, -1, -1)
    glEnd()
    
    glDisable(GL_LIGHTING)
    
    # === 2D UI OVERLAY (EXACTLY LIKE RESEARCH PLATFORM) ===
    
    # Switch to 2D ortho
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 1600, 900, 0, -1, 1)  # Note: Y inverted
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    
    # Create pygame surface for UI (EXACTLY like research_platform.py)
    ui_surface = pygame.Surface((1600, 900), pygame.SRCALPHA)
    ui_surface.fill((0, 0, 0, 0))  # Transparent
    
    # Draw BRIGHT colored panels (EXACTLY like research_platform.py)
    # Blue panel (left)
    bg = pygame.Surface((280, 880), pygame.SRCALPHA)
    bg.fill((50, 100, 200, 255))  # OPAQUE blue
    ui_surface.blit(bg, (10, 10))
    pygame.draw.rect(ui_surface, (100, 200, 255), (10, 10, 280, 880), 3)
    
    # Orange panel (right)
    bg = pygame.Surface((280, 400), pygame.SRCALPHA)
    bg.fill((200, 100, 50, 255))  # OPAQUE orange
    ui_surface.blit(bg, (1310, 10))
    pygame.draw.rect(ui_surface, (255, 150, 100), (1310, 10, 280, 400), 3)
    
    # Green panel (bottom)
    bg = pygame.Surface((1000, 200), pygame.SRCALPHA)
    bg.fill((50, 200, 100, 255))  # OPAQUE green
    ui_surface.blit(bg, (300, 690))
    pygame.draw.rect(ui_surface, (100, 255, 150), (300, 690, 1000, 200), 3)
    
    # Convert to texture (EXACTLY like research_platform.py)
    texture_data = pygame.image.tostring(ui_surface, "RGBA", False)
    
    # Upload to OpenGL texture
    if not hasattr(pygame, '_ui_test_texture'):
        pygame._ui_test_texture = glGenTextures(1)
    
    glBindTexture(GL_TEXTURE_2D, pygame._ui_test_texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1600, 900, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    
    # Draw textured quad (EXACTLY like research_platform.py)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_ONE, GL_ZERO)  # Same blend mode we're using now
    
    glColor4f(1.0, 1.0, 1.0, 1.0)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(0, 0)
    glTexCoord2f(1, 0); glVertex2f(1600, 0)
    glTexCoord2f(1, 1); glVertex2f(1600, 900)
    glTexCoord2f(0, 1); glVertex2f(0, 900)
    glEnd()
    
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glBindTexture(GL_TEXTURE_2D, 0)
    
    pygame.display.flip()
    angle += 2
    clock.tick(60)

pygame.quit()

print("\nWindow closed.")
print()
response = input("Did you see COLORED PANELS (blue, orange, green) over the 3D cube? (y/n): ").lower().strip()

print("\n" + "="*60)
if response == 'y':
    print("✅ UI PANELS VISIBLE!")
    print("\nThe rendering method WORKS on your system.")
    print("Something else is wrong with research_platform.py.")
    print("\nPossible issues:")
    print("  1. UI panels rendering BEHIND 3D scene")
    print("  2. Panel alpha too low")
    print("  3. Depth buffer not cleared properly")
else:
    print("❌ UI PANELS NOT VISIBLE!")
    print("\nThe texture-based UI overlay method FAILS on M1.")
    print("\nThis is the core problem. We need to use a different")
    print("rendering approach. Options:")
    print("  1. Use direct glBegin/glEnd for UI (no textures)")
    print("  2. Use different blend mode")
    print("  3. Render UI to separate framebuffer")
    print("  4. Use native macOS UI instead of OpenGL overlay")
