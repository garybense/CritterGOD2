"""
M1 Mac Visual Display Test

Tests if anything actually displays on screen (not just in framebuffer).
Opens visible windows with colored quads - YOU should see them.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import sys


def test_visual_simple_quad():
    """Test 1: Can you SEE a red square?"""
    print("\n" + "="*60)
    print("TEST 1: VISUAL - Simple Red Square")
    print("="*60)
    print("\nA window will open with a BRIGHT RED SQUARE.")
    print("Window will stay open for 3 seconds.")
    print()
    input("Press ENTER to start test...")
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("TEST 1: Do you see a RED SQUARE?")
    
    # Clear to white background
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    
    # Setup 2D
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 800, 600, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    
    # Draw HUGE bright red square
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_QUADS)
    glVertex2f(100, 100)
    glVertex2f(700, 100)
    glVertex2f(700, 500)
    glVertex2f(100, 500)
    glEnd()
    
    pygame.display.flip()
    
    print("\n⏰ Window open for 3 seconds...")
    print("Look at the window NOW!")
    time.sleep(3)
    
    pygame.quit()
    
    response = input("\nDid you see a BRIGHT RED SQUARE? (y/n): ").lower().strip()
    return response == 'y'


def test_visual_multiple_colors():
    """Test 2: Can you SEE colored rectangles?"""
    print("\n" + "="*60)
    print("TEST 2: VISUAL - Multiple Colored Rectangles")
    print("="*60)
    print("\nWindow will show RED, GREEN, BLUE, YELLOW rectangles.")
    print("Window stays open for 5 seconds.")
    print()
    input("Press ENTER to start test...")
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("TEST 2: Do you see COLORED RECTANGLES?")
    
    # Black background
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    
    # Setup 2D
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 800, 600, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    
    # Red rectangle (top left)
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_QUADS)
    glVertex2f(50, 50)
    glVertex2f(350, 50)
    glVertex2f(350, 250)
    glVertex2f(50, 250)
    glEnd()
    
    # Green rectangle (top right)
    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_QUADS)
    glVertex2f(450, 50)
    glVertex2f(750, 50)
    glVertex2f(750, 250)
    glVertex2f(450, 250)
    glEnd()
    
    # Blue rectangle (bottom left)
    glColor3f(0.0, 0.0, 1.0)
    glBegin(GL_QUADS)
    glVertex2f(50, 350)
    glVertex2f(350, 350)
    glVertex2f(350, 550)
    glVertex2f(50, 550)
    glEnd()
    
    # Yellow rectangle (bottom right)
    glColor3f(1.0, 1.0, 0.0)
    glBegin(GL_QUADS)
    glVertex2f(450, 350)
    glVertex2f(750, 350)
    glVertex2f(750, 550)
    glVertex2f(450, 550)
    glEnd()
    
    pygame.display.flip()
    
    print("\n⏰ Window open for 5 seconds...")
    print("Look at the window NOW!")
    print("You should see: RED (top-left), GREEN (top-right),")
    print("                BLUE (bottom-left), YELLOW (bottom-right)")
    time.sleep(5)
    
    pygame.quit()
    
    response = input("\nDid you see 4 COLORED RECTANGLES? (y/n): ").lower().strip()
    return response == 'y'


def test_visual_3d_then_2d():
    """Test 3: 3D cube then 2D overlay - like research platform."""
    print("\n" + "="*60)
    print("TEST 3: VISUAL - 3D Cube + 2D Overlay")
    print("="*60)
    print("\nThis mimics the research platform:")
    print("- Spinning 3D cube (like creatures)")
    print("- 2D colored overlay on top (like UI panels)")
    print("Window stays open for 5 seconds.")
    print()
    input("Press ENTER to start test...")
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("TEST 3: 3D Cube + 2D Overlay")
    clock = pygame.time.Clock()
    
    angle = 0
    start_time = time.time()
    
    print("\n⏰ Rendering for 5 seconds...")
    print("Look for:")
    print("  - Spinning colored CUBE (3D)")
    print("  - GREEN RECTANGLE on left (2D overlay)")
    print("  - BLUE RECTANGLE on right (2D overlay)")
    
    while time.time() - start_time < 5:
        # Handle events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return False
        
        # === 3D SCENE ===
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 3D projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 800/600, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glEnable(GL_DEPTH_TEST)
        glTranslatef(0, 0, -5)
        glRotatef(angle, 1, 1, 1)
        
        # Draw cube faces
        glBegin(GL_QUADS)
        # Front (red)
        glColor3f(1, 0, 0)
        glVertex3f(-1, -1, 1)
        glVertex3f(1, -1, 1)
        glVertex3f(1, 1, 1)
        glVertex3f(-1, 1, 1)
        # Back (green)
        glColor3f(0, 1, 0)
        glVertex3f(-1, -1, -1)
        glVertex3f(-1, 1, -1)
        glVertex3f(1, 1, -1)
        glVertex3f(1, -1, -1)
        # Top (blue)
        glColor3f(0, 0, 1)
        glVertex3f(-1, 1, -1)
        glVertex3f(-1, 1, 1)
        glVertex3f(1, 1, 1)
        glVertex3f(1, 1, -1)
        # Bottom (yellow)
        glColor3f(1, 1, 0)
        glVertex3f(-1, -1, -1)
        glVertex3f(1, -1, -1)
        glVertex3f(1, -1, 1)
        glVertex3f(-1, -1, 1)
        glEnd()
        
        # === 2D OVERLAY ===
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 800, 600, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        
        # Green panel (left)
        glColor4f(0.0, 1.0, 0.0, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(250, 10)
        glVertex2f(250, 590)
        glVertex2f(10, 590)
        glEnd()
        
        # Blue panel (right)
        glColor4f(0.0, 0.5, 1.0, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(550, 10)
        glVertex2f(790, 10)
        glVertex2f(790, 590)
        glVertex2f(550, 590)
        glEnd()
        
        pygame.display.flip()
        angle += 2
        clock.tick(60)
    
    pygame.quit()
    
    print("\n")
    response = input("Did you see:\n  1) A spinning 3D CUBE?\n  2) GREEN and BLUE PANELS on the sides?\n(y/n): ").lower().strip()
    return response == 'y'


def main():
    """Run visual tests."""
    print("\n" + "👁️ "*30)
    print("M1 MAC VISUAL DISPLAY TEST")
    print("👁️ "*30)
    print("\nThis tests if you can SEE anything OpenGL renders.")
    print("Not framebuffer capture - actual visual display!")
    print()
    
    results = {}
    
    # Test 1
    results['simple_quad'] = test_visual_simple_quad()
    
    if not results['simple_quad']:
        print("\n❌ You can't see basic OpenGL rendering!")
        print("This means:")
        print("  - OpenGL context not displaying to screen")
        print("  - Window manager issue")
        print("  - Display driver problem")
        print("\nNo point continuing - basic rendering broken.")
        sys.exit(1)
    
    # Test 2
    results['multi_color'] = test_visual_multiple_colors()
    
    if not results['multi_color']:
        print("\n⚠️  You saw Test 1 but not Test 2!")
        print("This suggests:")
        print("  - Multiple draw calls have issues")
        print("  - Color blending problems")
        sys.exit(2)
    
    # Test 3
    results['3d_and_2d'] = test_visual_3d_then_2d()
    
    # Summary
    print("\n" + "="*60)
    print("VISUAL TEST RESULTS")
    print("="*60)
    
    if results['simple_quad']:
        print("✅ Test 1: Simple quad VISIBLE")
    else:
        print("❌ Test 1: Simple quad INVISIBLE")
    
    if results['multi_color']:
        print("✅ Test 2: Multiple colors VISIBLE")
    else:
        print("❌ Test 2: Multiple colors INVISIBLE")
    
    if results['3d_and_2d']:
        print("✅ Test 3: 3D + 2D overlay VISIBLE")
    else:
        print("❌ Test 3: 3D + 2D overlay INVISIBLE")
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if all(results.values()):
        print("✅ ALL TESTS PASSED!")
        print("\nYou CAN see OpenGL rendering including 2D overlays.")
        print("The research platform UI SHOULD be visible.")
        print("\nIf you still can't see it in research_platform.py:")
        print("  1) UI panels may be rendering BEHIND 3D scene")
        print("  2) Depth test not disabled properly")
        print("  3) UI texture not uploading correctly")
    elif results['simple_quad'] and results['multi_color'] and not results['3d_and_2d']:
        print("⚠️  2D works but 3D+2D overlay FAILS!")
        print("\nThis is EXACTLY the research platform issue!")
        print("When 3D and 2D are mixed, 2D overlay disappears.")
        print("\nLikely causes:")
        print("  1) Depth buffer not cleared between 3D and 2D")
        print("  2) Projection matrix not restored properly")
        print("  3) M1 OpenGL driver quirk with mixed rendering")
    else:
        print("❌ Basic visual rendering broken!")
    
    return results


if __name__ == "__main__":
    results = main()
    
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)
