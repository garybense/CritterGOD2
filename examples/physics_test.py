"""
Simple physics test - bouncing spheres.

Demonstrates:
- Creating physics world
- Adding rigid bodies
- Gravity and ground collision
- Sphere-sphere collisions
- Collision callbacks
"""

import sys
sys.path.insert(0, '/Users/gspilz/code/CritterGOD')

import numpy as np
from core.physics import PhysicsWorld, RigidBody, Collision


def on_collision(collision: Collision):
    """Collision callback."""
    print(f"💥 Collision: Body {collision.body_a.id} <-> Body {collision.body_b.id}")
    print(f"   Penetration: {collision.penetration:.3f}")
    print(f"   Position: {collision.point}")


def main():
    print("🌍 Physics Test: Bouncing Spheres\n")
    
    # Create physics world
    world = PhysicsWorld(
        gravity=(0.0, 0.0, -9.81),
        world_bounds=(-100.0, -100.0, 100.0, 100.0)
    )
    
    # Register collision callback
    world.register_collision_callback(on_collision)
    
    # Create sphere 1 (drop from height)
    body1 = world.create_sphere_body(
        position=(0.0, 0.0, 50.0),
        radius=5.0,
        mass=1.0,
        restitution=0.8  # Bouncy
    )
    print(f"✅ Created sphere 1 at {body1.position} (r={body1.radius}, m={body1.mass})")
    
    # Create sphere 2 (drop from different height)
    body2 = world.create_sphere_body(
        position=(10.0, 0.0, 30.0),
        radius=4.0,
        mass=0.5,
        restitution=0.6
    )
    print(f"✅ Created sphere 2 at {body2.position} (r={body2.radius}, m={body2.mass})")
    
    # Create fixed sphere (obstacle)
    body3 = world.create_sphere_body(
        position=(0.0, 0.0, 15.0),
        radius=3.0,
        mass=1.0,
        fixed=True  # Doesn't move
    )
    body3.fixed = True
    print(f"✅ Created fixed sphere at {body3.position} (obstacle)\n")
    
    print("Running simulation...")
    print("=" * 60)
    
    # Run simulation
    dt = 1.0 / 60.0  # 60 FPS
    for step in range(600):  # 10 seconds (give time to settle)
        world.step(dt)
        
        # Print status every 120 steps (2 seconds)
        if step % 120 == 0:
            print(f"\n⏱ Time: {step * dt:.1f}s")
            print(f"  Body 1: pos={body1.position}, vel={body1.get_velocity(dt)}")
            print(f"  Body 2: pos={body2.position}, vel={body2.get_velocity(dt)}")
    
    print("\n" + "=" * 60)
    print("✅ Physics test complete!")
    print(f"\nFinal positions:")
    print(f"  Body 1: {body1.position} (z={body1.position[2]:.2f})")
    print(f"  Body 2: {body2.position} (z={body2.position[2]:.2f})")
    print(f"  Body 3: {body3.position} (fixed)")
    
    # Check physics results
    # Body 1 should be resting on fixed obstacle (z = 15 + 3 + 5 = 23)
    expected_z1 = body3.position[2] + body3.radius + body1.radius
    assert abs(body1.position[2] - expected_z1) < 1.0, f"Body 1 should rest on obstacle at z={expected_z1}"
    
    # Body 2 should be on ground (z = radius = 4)
    assert abs(body2.position[2] - body2.radius) < 1.0, "Body 2 should be on ground"
    
    print(f"\n✅ Physics working correctly!")
    print(f"  Body 1 resting on obstacle at z={body1.position[2]:.2f} (expected ~{expected_z1:.2f})")
    print(f"  Body 2 resting on ground at z={body2.position[2]:.2f} (expected ~{body2.radius:.2f})")


if __name__ == '__main__':
    main()
