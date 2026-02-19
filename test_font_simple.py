#!/usr/bin/env python3
"""
Simple font test - verify pygame.font works and can render text.
"""

import pygame
import sys

pygame.init()
pygame.display.set_mode((640, 480), pygame.DOUBLEBUF | pygame.OPENGL)

print("🔧 Testing pygame.font...")

try:
    # Force init
    pygame.font.init()
    
    # Create fonts
    font = pygame.font.Font(None, 24)
    print("✅ pygame.font.Font(None, 24) created successfully!")
    
    # Render text
    text_surface = font.render("Hello World!", True, (255, 255, 255))
    print(f"✅ Text rendered! Surface size: {text_surface.get_size()}")
    
    # Check if surface has content
    import numpy as np
    pixels = pygame.surfarray.array3d(text_surface)
    non_black_pixels = np.sum(pixels > 0)
    print(f"✅ Non-black pixels: {non_black_pixels}")
    
    if non_black_pixels > 0:
        print("✅✅✅ SUCCESS: Text is actually rendering!")
        sys.exit(0)
    else:
        print("❌ FAIL: Text surface is all black")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
