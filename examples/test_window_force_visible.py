"""
Force visible window test - makes window impossible to miss.
"""

import pygame
import sys
import time
import subprocess

print("\n" + "="*60)
print("FORCING WINDOW TO BE VISIBLE")
print("="*60)
print("\nThis test will:")
print("1. Create a FULLSCREEN window (impossible to miss)")
print("2. Show system notification")
print("3. Keep window open for 5 seconds")
print()
input("Press ENTER to start...")

try:
    # Send system notification
    subprocess.run([
        'osascript', '-e',
        'display notification "LOOK FOR FULLSCREEN RED WINDOW NOW!" with title "CritterGOD Test"'
    ], check=False)
    
    pygame.init()
    
    # Get display info
    info = pygame.display.Info()
    screen_w = info.current_w
    screen_h = info.current_h
    print(f"\n📺 Screen size: {screen_w}x{screen_h}")
    
    # Create FULLSCREEN window
    print("Creating FULLSCREEN window...")
    screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
    pygame.display.set_caption("CritterGOD Test - FULLSCREEN")
    
    # Fill with BRIGHT RED
    screen.fill((255, 0, 0))
    
    # Draw huge text
    try:
        font = pygame.font.Font(None, 200)
        text = font.render("CAN YOU SEE THIS?", True, (255, 255, 0))
        text_rect = text.get_rect(center=(screen_w//2, screen_h//2))
        screen.blit(text, text_rect)
    except:
        pass  # Font might fail but red screen should show
    
    pygame.display.flip()
    
    print("\n" + "🔴"*30)
    print("FULLSCREEN RED WINDOW IS NOW OPEN!")
    print("🔴"*30)
    print("\nThe ENTIRE SCREEN should be RED.")
    print("With yellow text: 'CAN YOU SEE THIS?'")
    print("\nWindow will close in 5 seconds...")
    print("Press ESC to close early.")
    
    start = time.time()
    running = True
    while running and time.time() - start < 5:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        time.sleep(0.1)
    
    pygame.quit()
    print("\nWindow closed.")
    
    response = input("\nDid you see a FULLSCREEN RED WINDOW? (y/n): ").lower().strip()
    
    if response == 'y':
        print("\n✅ SUCCESS! Pygame windows ARE visible on your system!")
        print("\nThis means the research platform SHOULD work.")
        print("The issue might be:")
        print("  1. Window appearing behind Warp Terminal")
        print("  2. Window appearing on different desktop/space")
        print("  3. Window too small to notice")
        print("\nTry running research_platform.py and:")
        print("  - Use Cmd+Tab to switch to it")
        print("  - Check Mission Control (F3) for hidden windows")
        print("  - Look in all Spaces/Desktops")
    else:
        print("\n❌ PROBLEM: Even fullscreen window not visible!")
        print("\nThis is very strange. Possible causes:")
        print("  1. macOS Screen Recording permission not granted")
        print("  2. Python blocked from window creation")
        print("  3. Display server issue")
        print("\nTry:")
        print("  1. Open System Settings → Privacy & Security → Screen Recording")
        print("  2. Add Terminal or Python to allowed apps")
        print("  3. Restart Terminal/Warp")
        print("  4. Try: sudo python3 examples/test_window_force_visible.py")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
