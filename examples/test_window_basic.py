"""
Minimal window creation test for M1 Mac.
Tests if pygame can create ANY window at all.
"""

import pygame
import sys
import time

print("\n" + "="*60)
print("MINIMAL PYGAME WINDOW TEST")
print("="*60)

# Test 1: Basic pygame window (no OpenGL)
print("\nTest 1: Basic pygame window (no OpenGL)...")
try:
    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Test 1: Basic Pygame Window")
    
    # Fill with bright green
    screen.fill((0, 255, 0))
    pygame.display.flip()
    
    print("✅ Window created!")
    print("⏰ Keeping window open for 3 seconds...")
    print("   Look for a GREEN WINDOW!")
    
    start = time.time()
    while time.time() - start < 3:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        time.sleep(0.1)
    
    pygame.quit()
    print("Window closed.")
    
    response = input("\nDid you see a GREEN WINDOW? (y/n): ").lower().strip()
    basic_works = response == 'y'
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    basic_works = False

# Test 2: OpenGL window
print("\n" + "="*60)
print("\nTest 2: OpenGL window...")
try:
    from pygame.locals import DOUBLEBUF, OPENGL
    from OpenGL.GL import *
    
    pygame.init()
    screen = pygame.display.set_mode((400, 300), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Test 2: OpenGL Window")
    
    # Clear to bright red
    glClearColor(1.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    pygame.display.flip()
    
    print("✅ OpenGL window created!")
    print("⏰ Keeping window open for 3 seconds...")
    print("   Look for a RED WINDOW!")
    
    start = time.time()
    while time.time() - start < 3:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        time.sleep(0.1)
    
    pygame.quit()
    print("Window closed.")
    
    response = input("\nDid you see a RED WINDOW? (y/n): ").lower().strip()
    opengl_works = response == 'y'
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    opengl_works = False

# Results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

if basic_works:
    print("✅ Basic pygame windows: WORKING")
else:
    print("❌ Basic pygame windows: BROKEN")
    print("   → pygame can't create windows at all!")

if opengl_works:
    print("✅ OpenGL windows: WORKING")
else:
    print("❌ OpenGL windows: BROKEN")
    print("   → OpenGL context creation failing!")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if not basic_works:
    print("❌ CRITICAL: Pygame can't create windows!")
    print("\nPossible causes:")
    print("  1. Running in headless/SSH session")
    print("  2. Display permissions not granted")
    print("  3. macOS accessibility settings blocking windows")
    print("  4. Running from tmux/screen without proper DISPLAY")
    print("\nTry:")
    print("  - Run from Terminal.app directly (not SSH)")
    print("  - Check System Preferences → Security & Privacy")
    print("  - Grant Python/Terminal screen recording permission")
    
elif not opengl_works:
    print("❌ PROBLEM: Pygame works but OpenGL doesn't!")
    print("\nPossible causes:")
    print("  1. OpenGL not available on M1 in this Python")
    print("  2. PyOpenGL not installed correctly")
    print("  3. M1 requires different OpenGL context settings")
    print("\nTry:")
    print("  - Reinstall PyOpenGL: pip install --upgrade PyOpenGL PyOpenGL_accelerate")
    print("  - Check Python is native ARM64, not Rosetta")
    print("  - Try: python3 -c 'from OpenGL.GL import *; print(\"OK\")'")
    
else:
    print("✅ Both basic and OpenGL windows work!")
    print("\nBut you said you see no window in the other tests?")
    print("This suggests:")
    print("  1. Windows ARE being created but immediately closing")
    print("  2. Windows hidden behind other windows")
    print("  3. Windows created but completely transparent/invisible")
    print("  4. Window manager issue on M1")

sys.exit(0 if (basic_works and opengl_works) else 1)
