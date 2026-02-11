"""
Interactive evolutionary Markov text generation demo.

Demonstrates self-organizing text through attract/repel dynamics.
Watch as word pairs breed and die based on usage patterns.
"""

import time
from generators.markov import EvolutionaryMarkov


# Sample corpus - feel free to replace with your own text
CORPUS = """
The quick brown fox jumps over the lazy dog.
The cat sat on the mat and watched the world go by.
The dog ran fast through the green park.
The fox was clever and quick as lightning.
The world is full of wonder and mystery.
The cat loved to sleep in the warm sunlight.
The park was green and beautiful in spring.
The mystery deepened as night fell.
The lightning struck the old oak tree.
The sunlight filtered through the leaves.
"""


def print_separator():
    """Print visual separator."""
    print("=" * 70)


def print_generation_stats(evo: EvolutionaryMarkov, gen_num: int, text: str):
    """Print generation results and statistics."""
    print(f"\n--- Generation {gen_num} ---")
    print(f"Generated: {text}")
    
    stats = evo.get_stats()
    print(f"\nStats:")
    print(f"  Unique pairs: {stats['unique_pairs']}")
    print(f"  Total bred: {stats['total_bred']}")
    print(f"  Total killed: {stats['total_killed']}")
    print(f"  Avg score: {stats['average_score']:.1f}")


def print_top_pairs(evo: EvolutionaryMarkov, n: int = 5):
    """Print pairs with highest scores (candidates for breeding)."""
    top_pairs = evo.get_top_pairs(n=n)
    print(f"\nTop {n} pairs (breeding candidates):")
    for pair, score in top_pairs:
        print(f"  {pair[0]:12s} -> {pair[1]:12s}  score: {score:.1f}")


def print_bottom_pairs(evo: EvolutionaryMarkov, n: int = 5):
    """Print pairs with lowest scores (near death)."""
    bottom_pairs = evo.get_bottom_pairs(n=n)
    print(f"\nBottom {n} pairs (near death):")
    for pair, score in bottom_pairs:
        print(f"  {pair[0]:12s} -> {pair[1]:12s}  score: {score:.1f}")


def run_demo(num_generations: int = 20, pause_time: float = 0.5):
    """
    Run evolutionary Markov demo.
    
    Args:
        num_generations: Number of generations to run
        pause_time: Seconds to pause between generations
    """
    print_separator()
    print("EVOLUTIONARY MARKOV TEXT GENERATION")
    print("Based on markov-attract-repel.py from critters codebase")
    print_separator()
    
    # Initialize system
    print("\nInitializing evolutionary Markov system...")
    evo = EvolutionaryMarkov(
        wordpair_start_energy=1000.0,
        start_attrep=300.0,
        attrep_hit_cost=200.0,
        breed_threshold=1500.0,
        kill_threshold=0.1,
        mutation_rate=0.3,
        decay_rate=0.99,
    )
    
    # Add corpus
    print("Adding seed corpus...")
    evo.add_corpus(CORPUS)
    print(f"Initial unique pairs: {evo.stats['unique_pairs']}")
    
    # Show initial top pairs
    print_top_pairs(evo, n=5)
    
    print("\nStarting evolution...")
    print("Watch as pairs breed (mutate) and die (deplete)!\n")
    time.sleep(2)
    
    # Run generations
    for gen in range(1, num_generations + 1):
        # Generate text and evolve
        text = evo.generate_and_evolve(max_length=15)
        
        # Print results every few generations
        if gen % 5 == 0 or gen == 1:
            print_separator()
            print_generation_stats(evo, gen, text)
            print_top_pairs(evo, n=3)
            print_bottom_pairs(evo, n=3)
        else:
            # Brief output for intermediate generations
            print(f"Gen {gen:3d}: {text}")
        
        # Apply decay
        evo.update(dt=1.0)
        
        time.sleep(pause_time)
    
    # Final statistics
    print_separator()
    print("\nFINAL STATISTICS")
    print_separator()
    stats = evo.get_stats()
    print(f"Total generations: {stats['generations']}")
    print(f"Final unique pairs: {stats['unique_pairs']}")
    print(f"Total pairs bred: {stats['total_bred']}")
    print(f"Total pairs killed: {stats['total_killed']}")
    print(f"Average score: {stats['average_score']:.1f}")
    
    print("\nTop surviving pairs:")
    print_top_pairs(evo, n=10)
    
    print("\nEvolution complete!")
    print_separator()


def run_interactive():
    """Run interactive mode where user controls generation."""
    print_separator()
    print("INTERACTIVE EVOLUTIONARY MARKOV")
    print_separator()
    
    # Initialize
    evo = EvolutionaryMarkov()
    evo.add_corpus(CORPUS)
    
    print(f"\nInitial pairs: {evo.stats['unique_pairs']}")
    print("\nCommands:")
    print("  g - Generate next generation")
    print("  s - Show statistics")
    print("  t - Show top pairs")
    print("  b - Show bottom pairs")
    print("  q - Quit")
    
    gen_count = 0
    while True:
        cmd = input("\n> ").strip().lower()
        
        if cmd == 'q':
            break
        elif cmd == 'g':
            gen_count += 1
            text = evo.generate_and_evolve(max_length=15)
            print(f"Gen {gen_count}: {text}")
            evo.update(dt=1.0)
        elif cmd == 's':
            stats = evo.get_stats()
            print(f"\nGenerations: {stats['generations']}")
            print(f"Unique pairs: {stats['unique_pairs']}")
            print(f"Bred: {stats['total_bred']}, Killed: {stats['total_killed']}")
            print(f"Avg score: {stats['average_score']:.1f}")
        elif cmd == 't':
            print_top_pairs(evo, n=10)
        elif cmd == 'b':
            print_bottom_pairs(evo, n=10)
    
    print("\nGoodbye!")


if __name__ == "__main__":
    import sys
    
    # Check for interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        run_interactive()
    else:
        # Run automated demo
        run_demo(num_generations=20, pause_time=0.3)
