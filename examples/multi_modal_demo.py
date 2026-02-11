"""
Multi-Modal Creature Demo

Demonstrates EnhancedCreature with complete synesthetic capabilities:
- Visual sensing (retinal array reading Circuit8)
- Audio generation (voice from neural activity)
- Text generation (thoughts from markov chains)
- Visual pattern generation (brain-driven aesthetics)

Creates living beings that truly see, speak, and think.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.evolution.genotype import Genotype
from core.morphic.circuit8 import Circuit8
from creatures.enhanced_creature import EnhancedCreature
from creatures.genetic_language import GeneticLanguage


def create_demo_creature(
    creature_id: int,
    circuit8: Circuit8,
    x: float,
    y: float,
    seed_text: str
) -> EnhancedCreature:
    """
    Create an enhanced creature for demonstration.
    
    Args:
        creature_id: Creature identifier
        circuit8: Shared telepathic canvas
        x: X position
        y: Y position
        seed_text: Initial language seed
        
    Returns:
        Enhanced creature with multi-modal capabilities
    """
    # Create random genotype
    genotype = Genotype.create_random(
        n_sensory=100,  # 100 for retinal array (will be overridden by CreatureSenses)
        n_motor=10,
        n_hidden_min=100,
        n_hidden_max=300,
        synapses_per_neuron=40
    )
    
    # Create enhanced creature
    creature = EnhancedCreature(
        genotype=genotype,
        x=x,
        y=y,
        initial_energy=1000000.0,
        circuit8=circuit8,
        enable_audio=True,
        enable_text=True,
        enable_visual_gen=False,  # Disable visual gen for performance
        audio_mode='mixed',
        seed_text=seed_text
    )
    
    return creature


def demonstrate_sensory_processing(creature: EnhancedCreature, circuit8: Circuit8) -> None:
    """
    Demonstrate visual sensing through retinal array.
    
    Args:
        creature: Creature to demonstrate
        circuit8: Circuit8 canvas to read from
    """
    print("\n=== Visual Sensing ===")
    
    # Write some test patterns to Circuit8
    for x in range(10, 20):
        for y in range(10, 20):
            circuit8.write_pixel(x, y, 255, 0, 0, blend=False)  # Red square
    
    for x in range(30, 40):
        for y in range(10, 20):
            circuit8.write_pixel(x, y, 0, 255, 0, blend=False)  # Green square
    
    # Process visual input
    visual_input = creature.senses.process_visual_input(
        circuit8,
        x=15,
        y=15
    )
    
    print(f"Visual input dimensions: {visual_input.shape}")
    print(f"Visual activation range: [{visual_input.min():.2f}, {visual_input.max():.2f}]")
    print(f"Number of active sensors: {(visual_input > 0.1).sum()}")


def demonstrate_text_generation(creature: EnhancedCreature, n_generations: int = 5) -> None:
    """
    Demonstrate text generation and evolution.
    
    Args:
        creature: Creature to demonstrate
        n_generations: Number of text generations to show
    """
    print("\n=== Text Generation (Creature's Thoughts) ===")
    
    for i in range(n_generations):
        text = creature.motors.generate_text(max_length=10)
        if text:
            print(f"Generation {i+1}: {text}")
    
    # Show language statistics
    stats = creature.motors.get_language_statistics()
    if stats['text_enabled']:
        print(f"\nLanguage Statistics:")
        print(f"  Unique word pairs: {stats.get('unique_pairs', 0)}")
        print(f"  Generations: {stats.get('generations', 0)}")
        print(f"  Total bred: {stats.get('total_bred', 0)}")
        print(f"  Total killed: {stats.get('total_killed', 0)}")


def demonstrate_audio_generation(creature: EnhancedCreature) -> None:
    """
    Demonstrate audio generation from brain state.
    
    Args:
        creature: Creature to demonstrate
    """
    print("\n=== Audio Generation (Creature's Voice) ===")
    
    # Generate audio samples
    audio = creature.motors.generate_audio(duration_seconds=0.1)
    
    if audio is not None:
        print(f"Audio buffer size: {len(audio)} samples")
        print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
        print(f"Audio RMS: {(audio**2).mean()**0.5:.3f}")
    
    # Show audio statistics
    stats = creature.motors.get_audio_statistics()
    if stats['audio_enabled']:
        print(f"\nAudio Statistics:")
        print(f"  Mode: {stats['mode']}")
        print(f"  Sample rate: {stats['sample_rate']} Hz")


def demonstrate_creature_update_cycle(creature: EnhancedCreature, n_timesteps: int = 50) -> None:
    """
    Demonstrate complete update cycle with all systems.
    
    Args:
        creature: Creature to demonstrate
        n_timesteps: Number of timesteps to run
    """
    print("\n=== Complete Update Cycle ===")
    print(f"Running {n_timesteps} timesteps with full multi-modal processing...")
    
    for t in range(n_timesteps):
        alive = creature.update(dt=1.0)
        if not alive:
            print(f"Creature died at timestep {t}")
            break
    
    # Show generation statistics
    stats = creature.get_generation_statistics()
    print(f"\nGeneration Statistics:")
    print(f"  Age: {creature.age} timesteps")
    print(f"  Audio generations: {stats['audio_generations']}")
    print(f"  Text generations: {stats['text_generations']}")
    print(f"  Pattern generations: {stats['pattern_generations']}")
    print(f"  Current energy: {creature.energy.energy:.2f}")
    print(f"  Network activity: {creature.network.get_activity_level():.3f}")
    
    # Show most recent thought
    thought = creature.get_current_thought()
    if thought:
        print(f"\nMost recent thought: \"{thought}\"")


def demonstrate_genetic_language() -> None:
    """
    Demonstrate genetic language breeding and mutation.
    """
    print("\n=== Genetic Language System ===")
    
    # Generate random seed texts
    parent1_text = GeneticLanguage.generate_random_seed_text()
    parent2_text = GeneticLanguage.generate_random_seed_text()
    
    print(f"Parent 1: {parent1_text[:50]}...")
    print(f"Parent 2: {parent2_text[:50]}...")
    
    # Breed texts
    offspring_text = GeneticLanguage.breed_text(parent1_text, parent2_text)
    print(f"Offspring: {offspring_text[:50]}...")
    
    # Mutate offspring
    mutated_text = GeneticLanguage.mutate_text(offspring_text)
    print(f"Mutated: {mutated_text[:50]}...")


def main():
    """
    Run complete multi-modal creature demonstration.
    """
    print("=" * 70)
    print("CRITTERGOD: MULTI-MODAL CREATURE DEMO")
    print("Synesthetic beings that see, speak, and think")
    print("=" * 70)
    
    # Create shared Circuit8
    circuit8 = Circuit8(width=64, height=48)
    
    # Generate seed text for creature
    seed_text = GeneticLanguage.generate_random_seed_text()
    
    # Create enhanced creature
    print("\n=== Creating Enhanced Creature ===")
    creature = create_demo_creature(
        creature_id=1,
        circuit8=circuit8,
        x=32.0,
        y=24.0,
        seed_text=seed_text
    )
    print(f"Creature created with {len(creature.network.neurons)} neurons")
    print(f"Retinal array: {creature.senses.retinal_array.num_sensors} sensors")
    print(f"Seed text: {seed_text[:50]}...")
    
    # Demonstrate each capability
    demonstrate_sensory_processing(creature, circuit8)
    demonstrate_text_generation(creature, n_generations=5)
    demonstrate_audio_generation(creature)
    demonstrate_creature_update_cycle(creature, n_timesteps=50)
    demonstrate_genetic_language()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nThe creature demonstrates:")
    print("✓ Visual sensing through retinal array")
    print("✓ Text generation with evolutionary markov chains")
    print("✓ Audio synthesis from neural activity")
    print("✓ Complete multi-modal update cycle")
    print("✓ Genetic language inheritance")
    print("\nPhase 4d: Multi-Modal Generative Life COMPLETE")


if __name__ == "__main__":
    main()
