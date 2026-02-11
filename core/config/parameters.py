"""
Parameter definitions for CritterGOD configuration system.

Defines all tunable parameters with ranges, defaults, and descriptions.
Inspired by critterdrug's comprehensive parameter control.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
from enum import Enum


class ParameterCategory(Enum):
    """Categories for organizing parameters."""
    WORLD = "world"
    CREATURE = "creature"
    NEURAL = "neural"
    BODY = "body"
    RESOURCE = "resource"
    EVOLUTION = "evolution"
    PHYSICS = "physics"


@dataclass
class Parameter:
    """
    A tunable simulation parameter.
    
    Attributes:
        name: Unique parameter identifier
        value: Current value
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        default: Default value
        description: Human-readable description
        category: Parameter category
        int_only: Whether value should be integer
    """
    name: str
    value: float
    min_val: float
    max_val: float
    default: float
    description: str
    category: ParameterCategory
    int_only: bool = False
    
    def set_value(self, value: float):
        """Set value with clamping to valid range."""
        clamped = max(self.min_val, min(self.max_val, value))
        if self.int_only:
            clamped = float(int(clamped))
        self.value = clamped
    
    def get_value(self) -> float:
        """Get current value."""
        return self.value
    
    def get_int_value(self) -> int:
        """Get value as integer."""
        return int(self.value)
    
    def reset(self):
        """Reset to default value."""
        self.value = self.default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'value': self.value,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'default': self.default,
            'description': self.description,
            'category': self.category.value,
            'int_only': self.int_only
        }


def create_default_parameters() -> Dict[str, Parameter]:
    """
    Create default parameter set.
    
    Returns:
        Dictionary mapping parameter names to Parameter objects
    """
    params = {}
    
    # WORLD PARAMETERS
    params['world_size_x'] = Parameter(
        'world_size_x', 500.0, 100.0, 2000.0, 500.0,
        'World width', ParameterCategory.WORLD
    )
    params['world_size_y'] = Parameter(
        'world_size_y', 500.0, 100.0, 2000.0, 500.0,
        'World height', ParameterCategory.WORLD
    )
    params['gravity_z'] = Parameter(
        'gravity_z', -9.8, -50.0, 50.0, -9.8,
        'Gravity (Z-axis)', ParameterCategory.WORLD
    )
    params['time_speed'] = Parameter(
        'time_speed', 1.0, 0.1, 10.0, 1.0,
        'Time speed multiplier', ParameterCategory.WORLD
    )
    
    # CREATURE PARAMETERS
    params['creature_max_lifetime'] = Parameter(
        'creature_max_lifetime', 10000.0, 1000.0, 100000.0, 10000.0,
        'Maximum creature lifespan', ParameterCategory.CREATURE, int_only=True
    )
    params['creature_start_energy'] = Parameter(
        'creature_start_energy', 1000000.0, 100000.0, 10000000.0, 1000000.0,
        'Starting energy for creatures', ParameterCategory.CREATURE
    )
    params['creature_max_energy'] = Parameter(
        'creature_max_energy', 2000000.0, 500000.0, 20000000.0, 2000000.0,
        'Maximum energy capacity', ParameterCategory.CREATURE
    )
    params['creature_sight_range'] = Parameter(
        'creature_sight_range', 100.0, 10.0, 500.0, 100.0,
        'Visual detection range', ParameterCategory.CREATURE
    )
    params['creature_proc_interval'] = Parameter(
        'creature_proc_interval', 1000.0, 100.0, 10000.0, 1000.0,
        'Timesteps between reproduction attempts', ParameterCategory.CREATURE, int_only=True
    )
    params['creature_min_energy_proc'] = Parameter(
        'creature_min_energy_proc', 800000.0, 100000.0, 5000000.0, 800000.0,
        'Minimum energy to reproduce', ParameterCategory.CREATURE
    )
    params['creature_kill_half_at'] = Parameter(
        'creature_kill_half_at', 100.0, 10.0, 1000.0, 100.0,
        'Population for cull trigger (0=disabled)', ParameterCategory.CREATURE, int_only=True
    )
    
    # NEURAL PARAMETERS
    params['neuron_count_min'] = Parameter(
        'neuron_count_min', 50.0, 10.0, 500.0, 50.0,
        'Minimum neurons per creature', ParameterCategory.NEURAL, int_only=True
    )
    params['neuron_count_max'] = Parameter(
        'neuron_count_max', 200.0, 50.0, 1000.0, 200.0,
        'Maximum neurons per creature', ParameterCategory.NEURAL, int_only=True
    )
    params['synapses_per_neuron'] = Parameter(
        'synapses_per_neuron', 40.0, 5.0, 100.0, 40.0,
        'Average synapses per neuron', ParameterCategory.NEURAL, int_only=True
    )
    params['plasticity_rate'] = Parameter(
        'plasticity_rate', 0.01, 0.001, 0.1, 0.01,
        'STDP learning rate', ParameterCategory.NEURAL
    )
    params['enable_plasticity'] = Parameter(
        'enable_plasticity', 1.0, 0.0, 1.0, 1.0,
        'Enable synaptic plasticity (0=off, 1=on)', ParameterCategory.NEURAL, int_only=True
    )
    
    # DRUG PARAMETERS
    params['drug_multiplier_inhib_antag'] = Parameter(
        'drug_multiplier_inhib_antag', 1.0, 0.0, 5.0, 1.0,
        'Inhibitory antagonist effect', ParameterCategory.NEURAL
    )
    params['drug_multiplier_inhib_agon'] = Parameter(
        'drug_multiplier_inhib_agon', 1.0, 0.0, 5.0, 1.0,
        'Inhibitory agonist effect', ParameterCategory.NEURAL
    )
    params['drug_multiplier_excit_antag'] = Parameter(
        'drug_multiplier_excit_antag', 1.0, 0.0, 5.0, 1.0,
        'Excitatory antagonist effect', ParameterCategory.NEURAL
    )
    params['drug_multiplier_excit_agon'] = Parameter(
        'drug_multiplier_excit_agon', 1.0, 0.0, 5.0, 1.0,
        'Excitatory agonist effect', ParameterCategory.NEURAL
    )
    params['drug_multiplier_potentiator'] = Parameter(
        'drug_multiplier_potentiator', 10.0, 1.0, 50.0, 10.0,
        'Potentiator amplification', ParameterCategory.NEURAL
    )
    
    # BODY PARAMETERS
    params['body_max_mutations'] = Parameter(
        'body_max_mutations', 6.0, 1.0, 20.0, 6.0,
        'Max mutations per reproduction', ParameterCategory.BODY, int_only=True
    )
    params['body_mutation_rate'] = Parameter(
        'body_mutation_rate', 0.3, 0.0, 1.0, 0.3,
        'Probability of mutation', ParameterCategory.BODY
    )
    params['body_max_segments'] = Parameter(
        'body_max_segments', 10.0, 2.0, 30.0, 10.0,
        'Maximum body segments', ParameterCategory.BODY, int_only=True
    )
    params['body_min_segments'] = Parameter(
        'body_min_segments', 2.0, 1.0, 10.0, 2.0,
        'Minimum body segments', ParameterCategory.BODY, int_only=True
    )
    params['body_max_limbs_per_segment'] = Parameter(
        'body_max_limbs_per_segment', 4.0, 0.0, 8.0, 4.0,
        'Max limbs per segment', ParameterCategory.BODY, int_only=True
    )
    params['body_min_head_size'] = Parameter(
        'body_min_head_size', 3.0, 1.0, 10.0, 3.0,
        'Minimum head size', ParameterCategory.BODY
    )
    params['body_max_head_size'] = Parameter(
        'body_max_head_size', 8.0, 5.0, 20.0, 8.0,
        'Maximum head size', ParameterCategory.BODY
    )
    
    # RESOURCE PARAMETERS
    params['food_spawn_rate'] = Parameter(
        'food_spawn_rate', 0.1, 0.0, 1.0, 0.1,
        'Food spawn probability per timestep', ParameterCategory.RESOURCE
    )
    params['food_max_lifetime'] = Parameter(
        'food_max_lifetime', 5000.0, 100.0, 50000.0, 5000.0,
        'Food lifespan before decay', ParameterCategory.RESOURCE, int_only=True
    )
    params['food_energy_value'] = Parameter(
        'food_energy_value', 50000.0, 1000.0, 500000.0, 50000.0,
        'Energy per food item', ParameterCategory.RESOURCE
    )
    params['food_size'] = Parameter(
        'food_size', 5.0, 1.0, 20.0, 5.0,
        'Food visual size', ParameterCategory.RESOURCE
    )
    params['drug_spawn_rate'] = Parameter(
        'drug_spawn_rate', 0.05, 0.0, 1.0, 0.05,
        'Drug spawn probability', ParameterCategory.RESOURCE
    )
    params['drug_potency'] = Parameter(
        'drug_potency', 1000.0, 100.0, 10000.0, 1000.0,
        'Drug effect strength', ParameterCategory.RESOURCE
    )
    
    # EVOLUTION PARAMETERS
    params['mutation_rate_neurons'] = Parameter(
        'mutation_rate_neurons', 0.3, 0.0, 1.0, 0.3,
        'Neuron mutation probability', ParameterCategory.EVOLUTION
    )
    params['mutation_rate_synapses'] = Parameter(
        'mutation_rate_synapses', 0.3, 0.0, 1.0, 0.3,
        'Synapse mutation probability', ParameterCategory.EVOLUTION
    )
    params['mutation_rate_body'] = Parameter(
        'mutation_rate_body', 0.3, 0.0, 1.0, 0.3,
        'Body mutation probability', ParameterCategory.EVOLUTION
    )
    params['selection_pressure'] = Parameter(
        'selection_pressure', 3.0, 2.0, 10.0, 3.0,
        'Tournament selection size', ParameterCategory.EVOLUTION, int_only=True
    )
    params['enable_omnivores'] = Parameter(
        'enable_omnivores', 0.0, 0.0, 1.0, 0.0,
        'Allow creature-eating (0=off, 1=on)', ParameterCategory.EVOLUTION, int_only=True
    )
    
    # PHYSICS PARAMETERS
    params['physics_friction'] = Parameter(
        'physics_friction', 0.98, 0.5, 1.0, 0.98,
        'Velocity damping factor', ParameterCategory.PHYSICS
    )
    params['physics_restitution'] = Parameter(
        'physics_restitution', 0.3, 0.0, 1.0, 0.3,
        'Collision bounciness', ParameterCategory.PHYSICS
    )
    params['physics_max_velocity'] = Parameter(
        'physics_max_velocity', 50.0, 10.0, 200.0, 50.0,
        'Maximum velocity magnitude', ParameterCategory.PHYSICS
    )
    
    return params


def get_parameters_by_category(parameters: Dict[str, Parameter]) -> Dict[ParameterCategory, List[Parameter]]:
    """
    Group parameters by category.
    
    Args:
        parameters: Dictionary of all parameters
        
    Returns:
        Dictionary mapping categories to parameter lists
    """
    categorized = {}
    for category in ParameterCategory:
        categorized[category] = []
    
    for param in parameters.values():
        categorized[param.category].append(param)
    
    # Sort by name within each category
    for category in categorized:
        categorized[category].sort(key=lambda p: p.name)
    
    return categorized
