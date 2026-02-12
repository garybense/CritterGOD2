"""
UI Visibility Diagnostic Tool

Runs research platform for a few frames, captures screenshot,
and analyzes whether UI overlays are actually visible.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.research_platform import ResearchPlatform


def capture_screenshot(width, height):
    """Capture OpenGL framebuffer as numpy array."""
    # Read pixels from OpenGL
    glReadBuffer(GL_FRONT)
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    
    # Convert to numpy array
    image = np.frombuffer(data, dtype=np.uint8)
    image = image.reshape((height, width, 3))
    
    # Flip vertically (OpenGL origin is bottom-left)
    image = np.flipud(image)
    
    return image


def analyze_ui_visibility(image, width, height):
    """Analyze screenshot to detect if UI panels are visible."""
    print("\n=== UI VISIBILITY ANALYSIS ===\n")
    
    results = {
        'config_panel': False,
        'graph_panel': False,
        'console': False,
        'stats_overlay': False
    }
    
    # Check config panel area (left side: x=10-290, y=10-890)
    config_region = image[10:min(890, height-10), 10:290, :]
    config_variance = np.var(config_region)
    config_blue = np.mean(config_region[:, :, 2])  # Blue channel (border)
    results['config_panel'] = config_variance > 10 or config_blue > 50
    print(f"Config Panel (left):")
    print(f"  Variance: {config_variance:.2f} (>10 = visible)")
    print(f"  Blue channel: {config_blue:.2f} (>50 = border visible)")
    print(f"  Status: {'✅ VISIBLE' if results['config_panel'] else '❌ NOT VISIBLE'}\n")
    
    # Check graph panel area (top right: x=width-290 to width-10, y=10-410)
    if width > 300:
        graph_region = image[10:min(410, height-10), max(0, width-290):width-10, :]
        graph_variance = np.var(graph_region)
        graph_orange = (np.mean(graph_region[:, :, 0]) + np.mean(graph_region[:, :, 1])) / 2
        results['graph_panel'] = graph_variance > 10 or graph_orange > 50
        print(f"Graph Panel (top right):")
        print(f"  Variance: {graph_variance:.2f} (>10 = visible)")
        print(f"  Orange-ish: {graph_orange:.2f} (>50 = border visible)")
        print(f"  Status: {'✅ VISIBLE' if results['graph_panel'] else '❌ NOT VISIBLE'}\n")
    
    # Check console area (bottom center: x=300-width-300, y=height-210 to height-10)
    if height > 220 and width > 600:
        console_region = image[max(0, height-210):height-10, 300:min(width-300, width-10), :]
        console_variance = np.var(console_region)
        console_green = np.mean(console_region[:, :, 1])  # Green channel (border)
        results['console'] = console_variance > 10 or console_green > 50
        print(f"Console (bottom center):")
        print(f"  Variance: {console_variance:.2f} (>10 = visible)")
        print(f"  Green channel: {console_green:.2f} (>50 = border visible)")
        print(f"  Status: {'✅ VISIBLE' if results['console'] else '❌ NOT VISIBLE'}\n")
    
    # Check stats overlay (center top: around x=width/2, y=10-100)
    stats_region = image[10:min(100, height-10), max(0, width//2-100):min(width, width//2+100), :]
    stats_variance = np.var(stats_region)
    results['stats_overlay'] = stats_variance > 10
    print(f"Stats Overlay (center top):")
    print(f"  Variance: {stats_variance:.2f} (>10 = visible)")
    print(f"  Status: {'✅ VISIBLE' if results['stats_overlay'] else '❌ NOT VISIBLE'}\n")
    
    # Overall assessment
    visible_count = sum(results.values())
    total_count = len(results)
    
    print("=" * 40)
    print(f"OVERALL: {visible_count}/{total_count} UI elements visible")
    print("=" * 40 + "\n")
    
    if visible_count == 0:
        print("❌ CRITICAL: NO UI ELEMENTS VISIBLE")
        print("   The texture-based rendering is not working.")
        print("   Possible causes:")
        print("   - Texture upload failing")
        print("   - Blending not enabled")
        print("   - Ortho projection incorrect")
        print("   - Surface data empty")
    elif visible_count < total_count:
        print("⚠️  WARNING: PARTIAL UI VISIBILITY")
        print(f"   {total_count - visible_count} elements not rendering")
    else:
        print("✅ SUCCESS: ALL UI ELEMENTS VISIBLE!")
    
    return results


def save_screenshot(image, filename="ui_test_screenshot.png"):
    """Save screenshot to file using pygame."""
    height, width, channels = image.shape
    
    # Create pygame surface from numpy array
    surface = pygame.Surface((width, height))
    
    # Convert RGB to pygame format
    pygame.surfarray.blit_array(surface, np.transpose(image, (1, 0, 2)))
    
    # Save
    pygame.image.save(surface, filename)
    print(f"\n📸 Screenshot saved: {filename}")


def main():
    """Run diagnostic test."""
    print("🔍 Starting UI Visibility Diagnostic...\n")
    
    # Create platform
    print("Initializing research platform...")
    platform = ResearchPlatform(width=1600, height=900)
    
    # Run for a few frames to ensure everything is initialized
    print("Running simulation for 3 frames...\n")
    for i in range(3):
        platform.handle_events()
        platform.update()
        platform.render()
        pygame.time.wait(100)  # 100ms between frames
    
    # Capture screenshot
    print("Capturing screenshot...")
    image = capture_screenshot(platform.width, platform.height)
    
    # Analyze
    results = analyze_ui_visibility(image, platform.width, platform.height)
    
    # Save screenshot for manual inspection
    save_screenshot(image, "/Users/gspilz/code/CritterGOD/ui_diagnostic.png")
    
    # Cleanup
    pygame.quit()
    
    # Exit with appropriate code
    visible_count = sum(results.values())
    if visible_count == 0:
        print("\n❌ TEST FAILED: No UI visible")
        sys.exit(1)
    elif visible_count < len(results):
        print("\n⚠️  TEST WARNING: Partial visibility")
        sys.exit(2)
    else:
        print("\n✅ TEST PASSED: All UI visible")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
