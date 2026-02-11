"""
Configuration System Demo

Demonstrates:
- Loading profiles
- Profile inheritance
- Command-line overrides
- Creating new profiles
- Attribute-style access to config
"""

import sys
sys.path.insert(0, '/Users/gspilz/code/CritterGOD')

from config import ConfigLoader, Config


def main():
    """Demonstrate configuration system."""
    
    print("=" * 70)
    print("CritterGOD Configuration System Demo")
    print("=" * 70)
    
    # Initialize loader
    loader = ConfigLoader()
    
    # List available profiles
    print("\n📁 Available Profiles:")
    profiles = loader.list_profiles()
    for profile in profiles:
        print(f"  - {profile}")
    
    # Load default profile
    print("\n\n🔧 Loading 'default' profile...")
    config = Config.from_profile('default')
    
    print(f"  Neural threshold: {config.neural.neuron_threshold_min} - {config.neural.neuron_threshold_max}")
    print(f"  Inhibitory ratio: {config.neural.inhibitory_ratio}")
    print(f"  Population size: {config.evolution.population_size}")
    print(f"  Initial energy: {config.energy.initial_energy}")
    print(f"  Drug decay rate: {config.drugs.decay_rate}")
    
    # Load psychedelic profile (demonstrates inheritance)
    print("\n\n💊 Loading 'psychedelic' profile (inherits from default)...")
    psyche_config = Config.from_profile('psychedelic')
    
    print(f"  Drug decay rate: {psyche_config.drugs.decay_rate} (slower!)")
    print(f"  Potentiator multiplier: {psyche_config.drugs.potentiator_multiplier}x")
    print(f"  Drug effect strength: {psyche_config.drugs.drug_effect_strength}x")
    print(f"  Initial energy: {psyche_config.energy.initial_energy} (more!)")
    print(f"  Target FPS: {psyche_config.visualization.target_fps}")
    
    # Load large_brain profile
    print("\n\n🧠 Loading 'large_brain' profile...")
    brain_config = Config.from_profile('large_brain')
    
    print(f"  Hidden neurons: {brain_config.evolution.n_hidden_min} - {brain_config.evolution.n_hidden_max}")
    print(f"  Synapses per neuron: {brain_config.neural.synapses_per_neuron}")
    print(f"  Population size: {brain_config.evolution.population_size} (smaller for big brains)")
    
    # Load fast_evolution profile
    print("\n\n⚡ Loading 'fast_evolution' profile...")
    fast_config = Config.from_profile('fast_evolution')
    
    print(f"  Mutation rate: {fast_config.evolution.mutation_rate} (high!)")
    print(f"  Base metabolism: {fast_config.energy.base_metabolism} (die faster)")
    print(f"  Reproduction cost: {fast_config.energy.reproduction_cost} (cheaper)")
    print(f"  Body mutation rate: {fast_config.morphology.body_mutation_rate}")
    
    # Demonstrate runtime modification
    print("\n\n✏️  Modifying configuration at runtime...")
    config.neural.inhibitory_ratio = 0.35
    config.evolution.population_size = 100
    print(f"  New inhibitory ratio: {config.neural.inhibitory_ratio}")
    print(f"  New population size: {config.evolution.population_size}")
    
    # Demonstrate command-line overrides
    print("\n\n🎯 Applying command-line overrides...")
    overrides = {
        'neural.inhibitory_ratio': '0.25',
        'evolution.mutation_rate': '0.7'
    }
    
    base_config_dict = loader.load_profile('default')
    overridden_config_dict = loader.apply_overrides(base_config_dict, overrides)
    overridden_config = Config(overridden_config_dict)
    
    print(f"  Inhibitory ratio: {overridden_config.neural.inhibitory_ratio} (overridden)")
    print(f"  Mutation rate: {overridden_config.evolution.mutation_rate} (overridden)")
    
    # Demonstrate saving a new profile
    print("\n\n💾 Creating and saving custom profile...")
    custom_config = Config.default()
    custom_config.neural.inhibitory_ratio = 0.40
    custom_config.drugs.decay_rate = 0.97
    custom_config.evolution.mutation_rate = 0.6
    
    # Save to new profile
    custom_config.save('custom_test')
    print(f"  Saved to profiles/custom_test.yaml")
    
    # Verify it was saved
    if 'custom_test' in loader.list_profiles():
        print(f"  ✓ Profile successfully saved and loadable")
        
        # Load it back
        loaded = Config.from_profile('custom_test')
        print(f"  Loaded back - inhibitory ratio: {loaded.neural.inhibitory_ratio}")
    
    # Show configuration comparison
    print("\n\n📊 Configuration Comparison:")
    print(f"{'Profile':<20} {'Inhibit%':<10} {'Mutation':<10} {'PopSize':<10} {'Energy':<12}")
    print("-" * 70)
    
    for profile_name in ['default', 'psychedelic', 'large_brain', 'fast_evolution']:
        try:
            cfg = Config.from_profile(profile_name)
            print(f"{profile_name:<20} {cfg.neural.inhibitory_ratio:<10.2f} "
                  f"{cfg.evolution.mutation_rate:<10.2f} "
                  f"{cfg.evolution.population_size:<10} "
                  f"{cfg.energy.initial_energy:<12.0f}")
        except Exception as e:
            print(f"{profile_name:<20} Error: {e}")
    
    print("\n\n" + "=" * 70)
    print("✅ Configuration system fully operational!")
    print("=" * 70)
    
    print("\n💡 Usage in your code:")
    print("""
    from config import Config
    
    # Load profile
    config = Config.from_profile('psychedelic')
    
    # Access parameters
    threshold = config.neural.neuron_threshold_min
    pop_size = config.evolution.population_size
    
    # Modify at runtime
    config.neural.inhibitory_ratio = 0.35
    
    # Save changes
    config.save('my_experiment')
    """)


if __name__ == "__main__":
    main()
