"""
Comprehensive UI Rendering Test

Tests multiple aspects of UI rendering to diagnose why panels aren't visible.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_1_basic_opengl_ui():
    """Test 1: Basic OpenGL 2D rendering (no research platform)."""
    print("\n" + "="*60)
    print("TEST 1: Basic OpenGL 2D Quad Rendering")
    print("="*60)
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Test 1: Basic 2D OpenGL")
    
    # Clear to dark background
    glClearColor(0.1, 0.1, 0.2, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Setup 2D ortho
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 800, 600, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    glDisable(GL_DEPTH_TEST)
    
    # Draw bright red quad in center
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_QUADS)
    glVertex2f(200, 200)
    glVertex2f(600, 200)
    glVertex2f(600, 400)
    glVertex2f(200, 400)
    glEnd()
    
    pygame.display.flip()
    pygame.time.wait(500)
    
    # Capture and check
    data = glReadPixels(0, 0, 800, 600, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape((600, 800, 3))
    image = np.flipud(image)
    
    # Check center pixel
    center = image[300, 400]
    red_detected = center[0] > 200
    
    print(f"\nCenter pixel RGB: {center}")
    print(f"Red quad visible: {'✅ YES' if red_detected else '❌ NO'}")
    
    pygame.quit()
    return red_detected


def test_2_pygame_surface_to_opengl():
    """Test 2: Pygame surface texture upload to OpenGL."""
    print("\n" + "="*60)
    print("TEST 2: Pygame Surface → OpenGL Texture")
    print("="*60)
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Test 2: Texture Upload")
    
    glClearColor(0.1, 0.1, 0.2, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Create pygame surface with bright pattern
    surf = pygame.Surface((800, 600), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    
    # Draw bright colored rectangles
    pygame.draw.rect(surf, (255, 0, 0, 255), (50, 50, 200, 150))  # Red
    pygame.draw.rect(surf, (0, 255, 0, 255), (300, 50, 200, 150))  # Green
    pygame.draw.rect(surf, (0, 0, 255, 255), (550, 50, 200, 150))  # Blue
    pygame.draw.rect(surf, (255, 255, 0, 255), (200, 250, 400, 300))  # Yellow
    
    # Convert to texture data
    texture_data = pygame.image.tostring(surf, "RGBA", False)
    
    # Upload to OpenGL
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 800, 600, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    
    # Setup 2D ortho
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 800, 600, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Draw textured quad
    glColor4f(1.0, 1.0, 1.0, 1.0)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(0, 0)
    glTexCoord2f(1, 0); glVertex2f(800, 0)
    glTexCoord2f(1, 1); glVertex2f(800, 600)
    glTexCoord2f(0, 1); glVertex2f(0, 600)
    glEnd()
    
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    
    pygame.display.flip()
    pygame.time.wait(500)
    
    # Capture and check
    data = glReadPixels(0, 0, 800, 600, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape((600, 800, 3))
    image = np.flipud(image)
    
    # Check colored regions
    red_px = image[100, 150]
    green_px = image[100, 400]
    blue_px = image[100, 650]
    yellow_px = image[400, 400]
    
    red_ok = red_px[0] > 200
    green_ok = green_px[1] > 200
    blue_ok = blue_px[2] > 200
    yellow_ok = yellow_px[0] > 200 and yellow_px[1] > 200
    
    print(f"\nRed rect pixel: {red_px} - {'✅ OK' if red_ok else '❌ FAIL'}")
    print(f"Green rect pixel: {green_px} - {'✅ OK' if green_ok else '❌ FAIL'}")
    print(f"Blue rect pixel: {blue_px} - {'✅ OK' if blue_ok else '❌ FAIL'}")
    print(f"Yellow rect pixel: {yellow_px} - {'✅ OK' if yellow_ok else '❌ FAIL'}")
    
    all_ok = red_ok and green_ok and blue_ok and yellow_ok
    print(f"\nTexture rendering: {'✅ WORKING' if all_ok else '❌ BROKEN'}")
    
    pygame.quit()
    return all_ok


def test_3_retina_scaling():
    """Test 3: Check for Retina/HiDPI scaling issues."""
    print("\n" + "="*60)
    print("TEST 3: Retina Display Scaling Detection")
    print("="*60)
    
    pygame.init()
    
    # Request specific size
    requested_size = (800, 600)
    screen = pygame.display.set_mode(requested_size, DOUBLEBUF | OPENGL)
    
    # Get actual OpenGL viewport
    viewport = glGetIntegerv(GL_VIEWPORT)
    gl_width = viewport[2]
    gl_height = viewport[3]
    
    # Get pygame surface size
    surf_width, surf_height = pygame.display.get_surface().get_size()
    
    print(f"\nRequested size: {requested_size}")
    print(f"Pygame surface size: ({surf_width}, {surf_height})")
    print(f"OpenGL viewport size: ({gl_width}, {gl_height})")
    
    scale_x = gl_width / requested_size[0]
    scale_y = gl_height / requested_size[1]
    
    print(f"\nScale factors: X={scale_x:.2f}, Y={scale_y:.2f}")
    
    if scale_x > 1.5 or scale_y > 1.5:
        print("⚠️  HIGH DPI DETECTED (Retina display)")
        print("   UI coordinates may need scaling!")
        retina = True
    else:
        print("✅ Normal DPI")
        retina = False
    
    pygame.quit()
    return retina, scale_x, scale_y


def test_4_blend_modes():
    """Test 4: Test different blend modes for UI rendering."""
    print("\n" + "="*60)
    print("TEST 4: Blend Mode Comparison")
    print("="*60)
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
    
    results = {}
    
    for blend_name, (src, dst) in [
        ("SRC_ALPHA/ONE_MINUS_SRC_ALPHA", (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)),
        ("ONE/ONE", (GL_ONE, GL_ONE)),
        ("SRC_ALPHA/ONE", (GL_SRC_ALPHA, GL_ONE)),
        ("ONE/ZERO", (GL_ONE, GL_ZERO)),
    ]:
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Setup 2D
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 800, 600, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(src, dst)
        
        # Draw semi-transparent green quad
        glColor4f(0.0, 1.0, 0.0, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(200, 200)
        glVertex2f(600, 200)
        glVertex2f(600, 400)
        glVertex2f(200, 400)
        glEnd()
        
        glDisable(GL_BLEND)
        pygame.display.flip()
        pygame.time.wait(100)
        
        # Check if visible
        data = glReadPixels(0, 0, 800, 600, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape((600, 800, 3))
        image = np.flipud(image)
        
        center = image[300, 400]
        visible = center[1] > 50  # Check green channel
        
        results[blend_name] = visible
        print(f"{blend_name:40s} - Center pixel: {center} - {'✅ VISIBLE' if visible else '❌ INVISIBLE'}")
    
    pygame.quit()
    return results


def main():
    """Run all diagnostic tests."""
    print("\n" + "🔬"*30)
    print("COMPREHENSIVE UI RENDERING DIAGNOSTIC")
    print("🔬"*30)
    
    results = {}
    
    try:
        results['basic_opengl'] = test_1_basic_opengl_ui()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        results['basic_opengl'] = False
    
    try:
        results['texture_upload'] = test_2_pygame_surface_to_opengl()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        results['texture_upload'] = False
    
    try:
        retina, scale_x, scale_y = test_3_retina_scaling()
        results['retina'] = retina
        results['scale_x'] = scale_x
        results['scale_y'] = scale_y
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        results['retina'] = None
    
    try:
        results['blend_modes'] = test_4_blend_modes()
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        results['blend_modes'] = {}
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    if results.get('basic_opengl'):
        print("✅ Basic OpenGL 2D rendering: WORKING")
    else:
        print("❌ Basic OpenGL 2D rendering: BROKEN")
        print("   → OpenGL driver or context issue!")
    
    if results.get('texture_upload'):
        print("✅ Texture upload & rendering: WORKING")
    else:
        print("❌ Texture upload & rendering: BROKEN")
        print("   → This is why UI overlays aren't visible!")
    
    if results.get('retina'):
        print(f"⚠️  Retina/HiDPI scaling: DETECTED (scale={results.get('scale_x', 1):.1f}x)")
        print("   → UI coordinates may need adjustment!")
    else:
        print("✅ Normal DPI display")
    
    blend_results = results.get('blend_modes', {})
    working_modes = [k for k, v in blend_results.items() if v]
    if working_modes:
        print(f"✅ Working blend modes: {', '.join(working_modes)}")
    else:
        print("❌ No blend modes working!")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if not results.get('basic_opengl'):
        print("❌ CRITICAL: Basic OpenGL not working")
        print("   Likely an OpenGL driver or context issue.")
        print("   Try updating graphics drivers.")
    elif not results.get('texture_upload'):
        print("❌ PROBLEM FOUND: Texture rendering broken")
        print("   This is why UI overlays aren't visible!")
        print("   The research platform uses texture-based UI.")
        print("   Try using glDrawPixels fallback instead.")
    elif results.get('retina'):
        print("⚠️  LIKELY CAUSE: Retina scaling issue")
        print(f"   Display is scaled {results.get('scale_x', 1):.1f}x")
        print("   UI may be rendering but outside visible area.")
        print("   Or coordinates need to be scaled.")
    else:
        print("✅ All tests passed! UI should be visible.")
        print("   If you still can't see it, check:")
        print("   - Window is in focus")
        print("   - Not hidden behind other windows")
        print("   - Display brightness settings")
    
    return results


if __name__ == "__main__":
    results = main()
    
    # Exit code based on results
    if not results.get('basic_opengl'):
        sys.exit(1)
    elif not results.get('texture_upload'):
        sys.exit(2)
    elif results.get('retina'):
        sys.exit(3)
    else:
        sys.exit(0)
