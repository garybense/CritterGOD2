"""
Creature Class

A complete organism: brain + body + energy + genome + drugs.

This is where all systems come together.
"""

import numpy as np
from typing import Optional, Tuple

from core.neural.network import NeuralNetwork
from core.neural.neuron import NeuronType
from core.evolution.genotype import Genotype
from core.evolution.phenotype import build_network_from_genotype
from core.pharmacology.drugs import DrugSystem, Pill
from core.energy.metabolism import EnergySystem, Food
from core.morphic.circuit8 import Circuit8
from core.morphic.morphic_field import (
    MorphicChannel,
    FieldType,
    MorphicFieldReader,
    MorphicFieldWriter,
)


class Creature:
    """
    A complete artificial life form.
    
    Integrates:
    - Neural network (brain)
    - Genotype (genetic code)
    - Energy system (metabolism)
    - Drug system (psychopharmacology)
    - Morphic field connection (telepathy)
    - Body (position, sensors, motors)
    """
    
    def __init__(
        self,
        genotype: Genotype,
        x: float = 0.0,
        y: float = 0.0,
        initial_energy: float = 1000000.0,
        circuit8: Optional[Circuit8] = None,
        adam_distance: int = 0
    ):
        """
        Initialize creature.
        
        Args:
            genotype: Genetic code
            x: X position
            y: Y position
            initial_energy: Starting energy
            circuit8: Shared telepathic canvas (optional)
            adam_distance: Generational depth from first random creature (Critterding2 feature)
        """
        # Identity
        self.genotype = genotype
        self.age = 0
        self.generation = 0
        self.adam_distance = adam_distance  # Track evolutionary lineage (Critterding2)
        
        # Position
        self.x = x
        self.y = y
        
        # Brain
        self.network = build_network_from_genotype(genotype)
        
        # Systems
        self.energy = EnergySystem(initial_energy=initial_energy)
        self.drugs = DrugSystem()
        
        # Morphic field connection
        self.circuit8 = circuit8
        self.morphic_channel = np.random.randint(0, 6)  # Random channel 0-5
        
        # Semantic field reader/writer (for pheromone trails, danger zones, etc.)
        if circuit8 is not None:
            self.field_reader = MorphicFieldReader(circuit8)
            self.field_writer = MorphicFieldWriter(circuit8)
        else:
            self.field_reader = None
            self.field_writer = None
        
        # Field sensing state (updated each tick)
        self.sensed_fields = {}  # FieldType -> intensity
        
        # Motor outputs (to be read by physics/rendering)
        self.motor_outputs = np.zeros(8)  # 8 motors
        
        # Circuit8 motor outputs (if condensed_colourmotors=1)
        self.screen_motors = np.zeros(6)  # RuRdGuGdBuBd
        
        # Voting
        self.vote_dx = 0
        self.vote_dy = 0
        
        # Fitness tracking
        self.fitness = 0.0
        self.food_consumed = 0
        
        # Unique ID for tracking (useful for debugging)
        self._creature_id = id(self)
        
    def update(self, dt: float = 1.0):
        """
        Update creature for one timestep.
        
        This is the main lifecycle:
        1. Read morphic field
        2. Update neural network
        3. Apply drug effects
        4. Extract motor outputs
        5. Vote on collective actions
        6. Pay metabolic costs
        7. Age
        
        Args:
            dt: Timestep duration
        """
        # Age
        self.age += 1
        
        # === SEMANTIC FIELD SENSING ===
        # Read pheromone trails, danger zones, etc. before neural processing
        self.sense_morphic_fields()
        
        # === COMPLETE SENSORY INTEGRATION (if mixin present) ===
        # This reads ALL senses: vision, proprioception, interoception, chemoreception
        # Call BEFORE neural network processes
        if hasattr(self, 'sense_environment'):
            self.sense_environment()
        
        if hasattr(self, 'inject_senses_to_brain'):
            self.inject_senses_to_brain()
        else:
            # Fallback to old morphic field input (for creatures without sensory mixin)
            morphic_input = 0.0
            if self.circuit8 is not None:
                # Read from current position
                pixel_x = int(self.x) % self.circuit8.width
                pixel_y = int(self.y) % self.circuit8.height
                r, g, b = self.circuit8.read_pixel(pixel_x, pixel_y)
                morphic_input = MorphicChannel.get_influence(
                    self.morphic_channel, r, g, b
                )
                
            # Apply morphic field to input neurons
            # Inject into sensory neurons before network update
            noise = np.random.uniform(0.0, 4000.0)
            for neuron in self.network.neurons:
                if neuron.neuron_type == NeuronType.SENSORY:
                    neuron.add_input(morphic_input + noise)
                elif neuron.neuron_type == NeuronType.MOTOR:
                    # Reduced tonic motor drive for more deliberate movement
                    neuron.add_input(np.random.uniform(100.0, 300.0))
        
        # Update neural network (with drug modulation of plasticity!)
        self.network.update(drug_system=self.drugs)
        
        # Apply drug effects to all neurons based on last-step firing
        for i, neuron in enumerate(self.network.neurons):
            if neuron.fired_last_step():
                modified_potential = self.drugs.apply_drug_effect(
                    neuron.potential,
                    neuron.is_inhibitory(),
                    neuron.fired_last_step()
                )
                neuron.potential = modified_potential
                
        # Extract motor outputs from output neurons
        # Simplified: assume last 8 neurons are motors
        num_neurons = len(self.network.neurons)
        if num_neurons >= 8:
            for i in range(8):
                neuron_idx = num_neurons - 8 + i
                neuron = self.network.neurons[neuron_idx]
                # Normalize potential to [0, 1] range based on threshold
                # Typical neuron threshold is 700-8700, use average of 4700
                normalized = np.clip(neuron.potential / 4700.0, 0.0, 1.0)
                self.motor_outputs[i] = normalized
                
        # Extract screen motors (if condensed_colourmotors)
        # Assume 6 neurons before regular motors are screen motors
        # Normalize to [-1, 1] centered around threshold so channels are balanced
        if num_neurons >= 14:
            for i in range(6):
                neuron_idx = num_neurons - 14 + i
                neuron = self.network.neurons[neuron_idx]
                # Center around threshold: fired = positive output, resting = negative
                self.screen_motors[i] = (neuron.potential - neuron.threshold) / neuron.threshold
                
        # Write to Circuit8 based on screen motors
        if self.circuit8 is not None:
            pixel_x = int(self.x) % self.circuit8.width
            pixel_y = int(self.y) % self.circuit8.height
            
            # RuRdGuGdBuBd control — now centered around 0
            r_change = self.screen_motors[0] - self.screen_motors[1]  # Ru - Rd
            g_change = self.screen_motors[2] - self.screen_motors[3]  # Gu - Gd
            b_change = self.screen_motors[4] - self.screen_motors[5]  # Bu - Bd
            
            # Drug amplification: tripping creatures write brighter patterns
            drug_amp = 1.0
            if hasattr(self, 'drugs'):
                trip_level = np.sum(self.drugs.tripping) / max(1.0, self.drugs.max_trip)
                drug_amp = 1.0 + trip_level * 5.0
            
            # Scale factor: balanced channels with moderate write strength
            scale = 10.0 * drug_amp
            
            # Condensed colour motors (from telepathic-critterdrug):
            # When motor output exceeds threshold, apply GLOBAL color shift
            # instead of just per-pixel write. Strong neural conviction
            # affects the entire collective perception field.
            global_threshold = 1.5  # Strong firing needed for global effect
            if abs(r_change) > global_threshold or abs(g_change) > global_threshold or abs(b_change) > global_threshold:
                # Global shift is weaker (1/4 of local) but affects everything
                g_scale = scale * 0.25
                self.circuit8.global_color_shift(
                    int(r_change * g_scale),
                    int(g_change * g_scale),
                    int(b_change * g_scale)
                )
            else:
                # Normal per-pixel write
                current_r, current_g, current_b = self.circuit8.read_pixel(pixel_x, pixel_y)
                new_r = np.clip(current_r + r_change * scale, 0, 255)
                new_g = np.clip(current_g + g_change * scale, 0, 255)
                new_b = np.clip(current_b + b_change * scale, 0, 255)
                
                self.circuit8.write_pixel(pixel_x, pixel_y, int(new_r), int(new_g), int(new_b), blend=False)
            
        # Collective voting with weighted tiers (from telepathic-critterdrug)
        # Motor output strength determines vote weight:
        #   weak (<0.3): tier 1 (weight 1)
        #   medium (0.3-0.7): tier 2 (weight 3)
        #   strong (>0.7): tier 3 (weight 7)
        motor_x = self.motor_outputs[0] - self.motor_outputs[1]
        motor_y = self.motor_outputs[2] - self.motor_outputs[3]
        self.vote_dx = int(np.sign(motor_x))
        self.vote_dy = int(np.sign(motor_y))
        
        if self.circuit8 is not None:
            # Determine vote weight from motor strength
            x_strength = abs(motor_x)
            y_strength = abs(motor_y)
            x_weight = 7 if x_strength > 0.7 else (3 if x_strength > 0.3 else 1)
            y_weight = 7 if y_strength > 0.7 else (3 if y_strength > 0.3 else 1)
            
            if self.vote_dx > 0:
                self.circuit8.vote_movement('right', weight=x_weight)
            elif self.vote_dx < 0:
                self.circuit8.vote_movement('left', weight=x_weight)
            if self.vote_dy > 0:
                self.circuit8.vote_movement('down', weight=y_weight)
            elif self.vote_dy < 0:
                self.circuit8.vote_movement('up', weight=y_weight)
            
        # Pay metabolic costs
        num_firing = sum(1 for n in self.network.neurons if n.fired_last_step())
        # Motor activity as fraction of motor neurons that fired
        motor_neurons = [n for n in self.network.neurons if n.neuron_type == NeuronType.MOTOR]
        if motor_neurons:
            motor_activity = sum(1 for n in motor_neurons if n.fired_last_step()) / float(len(motor_neurons))
        else:
            motor_activity = 0.0
        
        # Calculate body mass for metabolic cost (heavier = more expensive)
        body_mass = 0.0
        if hasattr(self, 'body') and self.body:
            body_mass = self.body.get_total_mass()
        
        alive = self.energy.update_metabolism(
            num_neurons=len(self.network.neurons),
            num_synapses=len(self.network.synapses),
            num_firing=num_firing,
            motor_activity=motor_activity,
            body_mass=body_mass
        )
        
        # Decay drugs
        self.drugs.update()
        
        # Update fitness (energy-based selection pressure)
        self.calculate_fitness()
        
        # === AUTOMATIC FIELD EMISSIONS ===
        # Emit danger when dying (low energy)
        if self.energy.get_energy_fraction() < 0.1:
            self.emit_danger_signal(intensity=0.5 + 0.5 * (1.0 - self.energy.get_energy_fraction() / 0.1))
        
        # Emit mating signal when ready to reproduce
        if self.can_reproduce():
            self.emit_mating_signal(intensity=0.6)
        
        # Emit excitement when neural activity is high
        activity = self.network.get_activity_level()
        if activity > 0.5:
            self.emit_excitement(intensity=activity * 0.7)
        
        return alive
        
    def eat_food(self, food: Food):
        """Consume food and emit food trail for others."""
        self.energy.consume_food(food)
        self.food_consumed += 1
        # Emit food trail pheromone so others can find food
        self.emit_food_trail(intensity=0.8)
        
    def consume_pill(self, pill: Pill):
        """Consume pill (drugs + energy)."""
        energy_gain = self.drugs.consume_pill(pill)
        self.energy.energy += energy_gain
        
    def can_reproduce(self) -> bool:
        """Check if capable of reproduction."""
        return (
            self.energy.can_reproduce() and
            self.age > 100  # Minimum age
        )
        
    def reproduce(self) -> Optional['Creature']:
        """
        Create offspring through mutation.
        
        Returns:
            New creature (offspring) or None if failed
        """
        if not self.can_reproduce():
            return None
            
        # Pay energy cost
        if not self.energy.pay_reproduction_cost():
            return None
            
        # Create mutated offspring genotype
        offspring_genotype = self.genotype.mutate(
            mutation_rate=0.5,
            max_mutations=50
        )
        
        # Transfer energy to offspring
        offspring_energy = self.energy.transfer_to_offspring(fraction=0.3)
        
        # Create offspring with incremented adam_distance (Critterding2 lineage tracking)
        offspring = Creature(
            genotype=offspring_genotype,
            x=self.x + np.random.randn() * 10.0,  # Near parent
            y=self.y + np.random.randn() * 10.0,
            initial_energy=offspring_energy,
            circuit8=self.circuit8,
            adam_distance=self.adam_distance + 1  # Increment generational depth
        )
        offspring.generation = self.generation + 1
        
        # Emit excitement when reproducing
        self.emit_excitement(intensity=1.0)
        
        return offspring
        
    def calculate_fitness(self) -> float:
        """
        Calculate fitness score.
        
        Based on:
        - Survival time (age)
        - Energy efficiency
        - Food consumed
        """
        self.fitness = (
            self.age * 1.0 +
            self.food_consumed * 100.0 +
            self.energy.get_energy_fraction() * 1000.0
        )
        return self.fitness
        
    def is_alive(self) -> bool:
        """Check if creature is alive."""
        return not self.energy.is_starving()
    
    def to_dict(self, include_network: bool = False) -> dict:
        """
        Serialize creature state to dictionary (Critterding2 entity introspection).
        
        Enables:
        - Debug UI inspection
        - Save/load functionality
        - Network visualization
        - Evolutionary analysis
        
        Args:
            include_network: Include detailed neural network state (expensive)
            
        Returns:
            Dictionary of creature state
        """
        state = {
            # Identity
            'creature_id': self._creature_id,
            'age': self.age,
            'generation': self.generation,
            'adam_distance': self.adam_distance,
            
            # Position
            'x': float(self.x),
            'y': float(self.y),
            
            # Energy
            'energy': float(self.energy.energy),
            'energy_fraction': float(self.energy.get_energy_fraction()),
            'is_alive': self.is_alive(),
            
            # Fitness
            'fitness': float(self.fitness),
            'food_consumed': int(self.food_consumed),
            
            # Network stats
            'num_neurons': len(self.network.neurons),
            'num_synapses': len(self.network.synapses),
            'network_activity': float(self.network.get_activity_level()),
            
            # Drugs
            'tripping': self.drugs.tripping.tolist(),
            
            # Morphic field
            'morphic_channel': int(self.morphic_channel),
            
            # Motors
            'motor_outputs': self.motor_outputs.tolist(),
            'screen_motors': self.screen_motors.tolist(),
            
            # Voting
            'vote_dx': int(self.vote_dx),
            'vote_dy': int(self.vote_dy),
        }
        
        # Optional detailed network state
        if include_network:
            state['neurons'] = [
                {
                    'id': n.neuron_id,
                    'type': n.neuron_type.name,
                    'potential': float(n.potential),
                    'threshold': float(n.threshold),
                    'fired': n.fired_last_step(),
                    'is_inhibitory': n.is_inhibitory(),
                }
                for n in self.network.neurons
            ]
            
            state['synapses'] = [
                {
                    'pre_id': s.pre_neuron.neuron_id,
                    'post_id': s.post_neuron.neuron_id,
                    'weight': float(s.weight),
                    'is_inhibitory': s.is_inhibitory,
                }
                for s in self.network.synapses
            ]
        
        return state
        
    def __repr__(self) -> str:
        return (
            f"Creature(age={self.age}, gen={self.generation}, adam={self.adam_distance}, "
            f"energy={self.energy.energy:.0f}, neurons={len(self.network.neurons)})"
        )
    
    # =========================
    # SEMANTIC FIELD METHODS
    # =========================
    
    def sense_morphic_fields(self):
        """
        Read all semantic field types at current position.
        
        Updates self.sensed_fields with current field intensities.
        Called automatically during update() before neural processing.
        """
        if self.field_reader is None:
            return
        
        x = int(self.x)
        y = int(self.y)
        self.sensed_fields = self.field_reader.read_all_fields_at(x, y)
    
    def sense_field(self, field_type: FieldType) -> float:
        """
        Get intensity of a specific field type at current position.
        
        Args:
            field_type: Type of field to sense
            
        Returns:
            Field intensity (0.0 to 1.0)
        """
        if self.field_reader is None:
            return 0.0
        return self.field_reader.read_field_at(int(self.x), int(self.y), field_type)
    
    def get_field_gradient(self, field_type: FieldType) -> Tuple[float, float]:
        """
        Get direction toward strongest signal of a field type.
        
        Args:
            field_type: Field to follow (e.g., FOOD_TRAIL)
            
        Returns:
            (dx, dy) normalized direction vector
        """
        if self.field_reader is None:
            return (0.0, 0.0)
        return self.field_reader.get_field_gradient(int(self.x), int(self.y), field_type)
    
    def emit_food_trail(self, intensity: float = 0.8):
        """
        Emit food trail pheromone at current position.
        Called when creature finds food.
        """
        if self.field_writer is not None:
            self.field_writer.mark_food_found(int(self.x), int(self.y), intensity)
    
    def emit_danger_signal(self, intensity: float = 1.0):
        """
        Emit danger warning at current position.
        Called when creature is dying or under threat.
        """
        if self.field_writer is not None:
            self.field_writer.mark_danger(int(self.x), int(self.y), intensity)
    
    def emit_mating_signal(self, intensity: float = 0.6):
        """
        Emit mating readiness signal.
        Called when creature is ready to reproduce.
        """
        if self.field_writer is not None:
            self.field_writer.mark_mating_ready(int(self.x), int(self.y), intensity)
    
    def emit_territory_marker(self, intensity: float = 0.5):
        """
        Mark current position as territory.
        """
        if self.field_writer is not None:
            self.field_writer.mark_territory(int(self.x), int(self.y), intensity)
    
    def emit_excitement(self, intensity: float = 0.7):
        """
        Emit excitement/interest signal.
        Called when something interesting happens.
        """
        if self.field_writer is not None:
            self.field_writer.mark_excitement(int(self.x), int(self.y), intensity)
    
    def get_sensed_fields_summary(self) -> dict:
        """
        Get summary of all sensed fields for debug/UI.
        
        Returns:
            Dictionary with field names and intensities
        """
        return {
            ft.name: self.sensed_fields.get(ft, 0.0)
            for ft in FieldType
        }
