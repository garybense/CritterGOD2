"""
Telepathic Collective Intelligence Simulation

Multiple creatures sharing Circuit8 telepathic canvas.
Demonstrates emergent collective behavior through morphic fields.

This is the culmination of Eric Burton's vision.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import List
import time

from core.evolution.genotype import Genotype
from core.morphic.circuit8 import Circuit8
from core.pharmacology.drugs import Pill, MoleculeType
from core.energy.metabolism import Food
from creatures.creature import Creature


class TelepathicWorld:
    """
    Environment for creatures sharing Circuit8.
    
    Demonstrates:
    - Collective voting on screen movement
    - Morphic resonance (creatures affected by screen state)
    - Drug effects on collective intelligence
    - Democratic emergence from individual behaviors
    """
    
    def __init__(
        self,
        num_creatures: int = 10,
        world_size: float = 1000.0
    ):
        """
        Initialize telepathic world.
        
        Args:
            num_creatures: Number of creatures
            world_size: Size of world (square)
        """
        self.world_size = world_size
        
        # Shared telepathic canvas
        self.circuit8 = Circuit8()
        # Seed the canvas with faint noise to kickstart morphic feedback
        self.circuit8.screen = np.random.randint(0, 32, size=self.circuit8.screen.shape, dtype=np.uint8)
        
        # Create initial population
        self.creatures: List[Creature] = []
        for i in range(num_creatures):
            genotype = Genotype.create_random(
                n_sensory=10,
                n_motor=16,  # 10 movement + 6 screen motors
                n_hidden_min=50,
                n_hidden_max=100,
                synapses_per_neuron=20
            )
            
            creature = Creature(
                genotype=genotype,
                x=np.random.uniform(0, world_size),
                y=np.random.uniform(0, world_size),
                initial_energy=1000000.0,
                circuit8=self.circuit8
            )
            
            self.creatures.append(creature)
            
        # Resources
        self.food_particles: List[Food] = []
        self.pills: List[Pill] = []
        
        # Spawn initial resources
        self.spawn_food(30)
        self.spawn_pills(10)
        
        # Statistics
        self.timestep = 0
        self.total_births = 0
        self.total_deaths = 0
        
    def spawn_food(self, count: int):
        """Spawn food particles."""
        for _ in range(count):
            food = Food(
                x=np.random.uniform(0, self.world_size),
                y=np.random.uniform(0, self.world_size)
            )
            self.food_particles.append(food)
            
    def spawn_pills(self, count: int):
        """Spawn pills with random molecule compositions."""
        for _ in range(count):
            # Random molecule composition
            composition = [
                np.random.randint(0, 50000),  # Inhibitory antagonist
                np.random.randint(0, 50000),  # Inhibitory agonist
                np.random.randint(0, 50000),  # Excitatory antagonist
                np.random.randint(0, 50000),  # Excitatory agonist
                np.random.randint(0, 5000),   # Potentiator (less common)
            ]
            
            pill = Pill(
                x=np.random.uniform(0, self.world_size),
                y=np.random.uniform(0, self.world_size),
                molecule_composition=composition
            )
            self.pills.append(pill)
            
    def update(self):
        """Update world for one timestep."""
        self.timestep += 1
        
        # Update all creatures
        deaths = []
        for i, creature in enumerate(self.creatures):
            alive = creature.update()
            if not alive:
                deaths.append(i)
                
        # Remove dead creatures
        for i in reversed(deaths):
            self.creatures.pop(i)
            self.total_deaths += 1
            
        # Check food consumption
        consumed_food = []
        for creature in self.creatures:
            for i, food in enumerate(self.food_particles):
                # Simple distance check
                dist = np.sqrt(
                    (creature.x - food.x)**2 + 
                    (creature.y - food.y)**2
                )
                if dist < 50.0:  # Within eating range
                    creature.eat_food(food)
                    consumed_food.append(i)
                    break
                    
        # Remove consumed food
        for i in reversed(consumed_food):
            self.food_particles.pop(i)
            
        # Check pill consumption
        consumed_pills = []
        for creature in self.creatures:
            for i, pill in enumerate(self.pills):
                dist = np.sqrt(
                    (creature.x - pill.x)**2 + 
                    (creature.y - pill.y)**2
                )
                if dist < 50.0:  # Within eating range
                    creature.consume_pill(pill)
                    consumed_pills.append(i)
                    break
                    
        # Remove consumed pills
        for i in reversed(consumed_pills):
            self.pills.pop(i)
            
        # Reproduction
        offspring = []
        for creature in self.creatures:
            if creature.can_reproduce() and np.random.random() < 0.01:
                child = creature.reproduce()
                if child is not None:
                    offspring.append(child)
                    self.total_births += 1
                    
        self.creatures.extend(offspring)
        
        # Apply collective voting to Circuit8
        self.circuit8.apply_voted_movement()
        
        # Update Circuit8 depth buffer
        self.circuit8.update_depth_buffer()
        
        # Replenish resources occasionally
        if self.timestep % 100 == 0:
            if len(self.food_particles) < 30:
                self.spawn_food(5)
            if len(self.pills) < 10:
                self.spawn_pills(2)
                
    def get_statistics(self) -> dict:
        """Get current world statistics."""
        if len(self.creatures) == 0:
            return {
                'timestep': self.timestep,
                'population': 0,
                'total_births': self.total_births,
                'total_deaths': self.total_deaths,
            }
            
        ages = [c.age for c in self.creatures]
        energies = [c.energy.energy for c in self.creatures]
        drug_levels = [c.drugs.get_total_drug_level() for c in self.creatures]
        neurons = [len(c.network.neurons) for c in self.creatures]
        fitnesses = [c.fitness for c in self.creatures]
        activity = [c.network.get_activity_level() for c in self.creatures]
        motor_fire = []
        for c in self.creatures:
            motors = [n for n in c.network.neurons if n.neuron_type.name == 'MOTOR']
            if motors:
                motor_fire.append(sum(1 for n in motors if n.fired_last_step()) / float(len(motors)))
            else:
                motor_fire.append(0.0)
        
        return {
            'timestep': self.timestep,
            'population': len(self.creatures),
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'avg_age': np.mean(ages),
            'max_age': np.max(ages),
            'avg_energy': np.mean(energies),
            'avg_drug_level': np.mean(drug_levels),
            'avg_neurons': np.mean(neurons),
            'avg_fitness': np.mean(fitnesses),
            'max_fitness': np.max(fitnesses),
            'avg_activity': np.mean(activity),
            'avg_motor_firing': np.mean(motor_fire),
            'food_count': len(self.food_particles),
            'pill_count': len(self.pills),
        }
        
    def print_status(self):
        """Print world status."""
        stats = self.get_statistics()
        print(f"Timestep {stats['timestep']:6d} | "
              f"Pop: {stats['population']:3d} | "
              f"Births: {stats['total_births']:3d} | "
              f"Deaths: {stats['total_deaths']:3d} | "
              f"Avg Age: {stats.get('avg_age', 0):6.1f} | "
              f"Avg Energy: {stats.get('avg_energy', 0):8.0f} | "
              f"Avg Fitness: {stats.get('avg_fitness', 0):8.0f} | "
              f"Avg Activity: {stats.get('avg_activity', 0):6.4f} | "
              f"Avg Motor Firing: {stats.get('avg_motor_firing', 0):6.4f}")


def run_simulation(timesteps: int = 1000, print_every: int = 50):
    """
    Run telepathic collective intelligence simulation.
    
    Args:
        timesteps: Number of timesteps to run
        print_every: Print status every N timesteps
    """
    print("=" * 80)
    print("TELEPATHIC COLLECTIVE INTELLIGENCE SIMULATION")
    print("=" * 80)
    print()
    print("Based on Eric Burton's (Flamoot's) revolutionary work:")
    print("- Circuit8 telepathic canvas (64x48 pixels, 1024 depth)")
    print("- Morphic field resonance (6 channels)")
    print("- Psychopharmacology (5 molecule types)")
    print("- Democratic voting (collective screen movement)")
    print("- Energy metabolism and evolution")
    print()
    print("Starting simulation...")
    print()
    
    world = TelepathicWorld(num_creatures=10)
    
    for t in range(timesteps):
        world.update()
        
        if t % print_every == 0:
            world.print_status()
            
        # Early exit if extinction
        if len(world.creatures) == 0:
            print()
            print("EXTINCTION - all creatures died")
            break
            
    print()
    print("=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    
    # Final statistics
    stats = world.get_statistics()
    print()
    print("Final Statistics:")
    print(f"  Total timesteps: {stats['timestep']}")
    print(f"  Final population: {stats['population']}")
    print(f"  Total births: {stats['total_births']}")
    print(f"  Total deaths: {stats['total_deaths']}")
    
    if stats['population'] > 0:
        print(f"  Average age: {stats['avg_age']:.1f}")
        print(f"  Maximum age: {stats['max_age']:.1f}")
        print(f"  Average energy: {stats['avg_energy']:.0f}")
        print(f"  Average fitness: {stats['avg_fitness']:.0f}")
        print(f"  Maximum fitness: {stats['max_fitness']:.0f}")
        print(f"  Average activity: {stats['avg_activity']:.2f}")
        print(f"  Average motor firing: {stats['avg_motor_firing']:.2f}")
        print(f"  Average neurons: {stats['avg_neurons']:.1f}")
        print(f"  Average drug level: {stats['avg_drug_level']:.0f}")
        
    print()
    print("Collective behaviors emerged through:")
    print("  - Morphic field coupling (creatures read/write Circuit8)")
    print("  - Democratic voting (emergent screen movement)")
    print("  - Drug-modified consciousness (psychopharmacology)")
    print("  - Genetic evolution (selection on collective fitness)")
    print()
    print("This is artificial life as Eric Burton envisioned it.")
    print()


if __name__ == '__main__':
    run_simulation(timesteps=1000, print_every=50)
