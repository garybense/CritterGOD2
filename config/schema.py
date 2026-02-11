"""
Configuration schema for CritterGOD.

Defines all configurable parameters with:
- Type validation
- Default values
- Documentation
- Valid ranges

Based on Critterding heritage profiles (foodotrope-drug.profile, etc.)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class ParamType(Enum):
    """Parameter type for validation."""
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"


@dataclass
class Parameter:
    """Configuration parameter definition.
    
    Attributes:
        name: Parameter name
        type: Parameter type
        default: Default value
        min_val: Minimum value (for numeric types)
        max_val: Maximum value (for numeric types)
        description: Documentation string
    """
    name: str
    type: ParamType
    default: Any
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    description: str = ""
    
    def validate(self, value: Any) -> Any:
        """Validate and convert value to correct type.
        
        Args:
            value: Value to validate
            
        Returns:
            Validated and converted value
            
        Raises:
            ValueError: If validation fails
        """
        # Type conversion
        if self.type == ParamType.INT:
            value = int(value)
        elif self.type == ParamType.FLOAT:
            value = float(value)
        elif self.type == ParamType.BOOL:
            if isinstance(value, str):
                value = value.lower() in ('true', 'yes', '1')
            else:
                value = bool(value)
        elif self.type == ParamType.STRING:
            value = str(value)
        
        # Range validation
        if self.min_val is not None and value < self.min_val:
            raise ValueError(f"{self.name}: value {value} below minimum {self.min_val}")
        if self.max_val is not None and value > self.max_val:
            raise ValueError(f"{self.name}: value {value} above maximum {self.max_val}")
        
        return value


class ConfigSchema:
    """Complete configuration schema for CritterGOD.
    
    All parameters organized by subsystem.
    Heritage values from Critterding profiles.
    """
    
    def __init__(self):
        """Initialize schema with all parameters."""
        self.params: Dict[str, Dict[str, Parameter]] = {}
        
        # Neural network parameters (Phase 1)
        self.params['neural'] = {
            'neuron_threshold_min': Parameter(
                'neuron_threshold_min', ParamType.FLOAT, 700.0,
                min_val=0.0, max_val=10000.0,
                description="Minimum neuron firing threshold (heritage: 700)"
            ),
            'neuron_threshold_max': Parameter(
                'neuron_threshold_max', ParamType.FLOAT, 8000.0,
                min_val=0.0, max_val=20000.0,
                description="Maximum neuron firing threshold (heritage: 8000)"
            ),
            'inhibitory_ratio': Parameter(
                'inhibitory_ratio', ParamType.FLOAT, 0.30,
                min_val=0.0, max_val=1.0,
                description="Proportion of inhibitory neurons (heritage: 0.14, CritterGOD4: 0.30)"
            ),
            'inhibitory_synapse_ratio': Parameter(
                'inhibitory_synapse_ratio', ParamType.FLOAT, 0.21,
                min_val=0.0, max_val=1.0,
                description="Proportion of inhibitory synapses (heritage: 0.21)"
            ),
            'synapses_per_neuron': Parameter(
                'synapses_per_neuron', ParamType.INT, 40,
                min_val=1, max_val=1000,
                description="Number of synapses per neuron (heritage: 40)"
            ),
            'enable_plasticity': Parameter(
                'enable_plasticity', ParamType.BOOL, True,
                description="Enable STDP synaptic plasticity"
            ),
            'plasticity_rate': Parameter(
                'plasticity_rate', ParamType.FLOAT, 0.1,
                min_val=0.0, max_val=1.0,
                description="Synaptic plasticity learning rate"
            ),
            'synapse_weakening_rate': Parameter(
                'synapse_weakening_rate', ParamType.FLOAT, 0.001,
                min_val=0.0, max_val=0.1,
                description="Continuous synapse weakening (Phase 5a)"
            ),
            'weight_min': Parameter(
                'weight_min', ParamType.FLOAT, -5.0,
                min_val=-100.0, max_val=0.0,
                description="Minimum synapse weight (Phase 5a)"
            ),
            'weight_max': Parameter(
                'weight_max', ParamType.FLOAT, 5.0,
                min_val=0.0, max_val=100.0,
                description="Maximum synapse weight (Phase 5a)"
            ),
        }
        
        # Evolution parameters (Phase 2)
        self.params['evolution'] = {
            'mutation_rate': Parameter(
                'mutation_rate', ParamType.FLOAT, 0.5,
                min_val=0.0, max_val=1.0,
                description="Probability of mutation per gene"
            ),
            'n_sensory_min': Parameter(
                'n_sensory_min', ParamType.INT, 10,
                min_val=1, max_val=1000,
                description="Minimum sensory neurons"
            ),
            'n_sensory_max': Parameter(
                'n_sensory_max', ParamType.INT, 50,
                min_val=1, max_val=1000,
                description="Maximum sensory neurons"
            ),
            'n_motor_min': Parameter(
                'n_motor_min', ParamType.INT, 10,
                min_val=1, max_val=1000,
                description="Minimum motor neurons"
            ),
            'n_motor_max': Parameter(
                'n_motor_max', ParamType.INT, 50,
                min_val=1, max_val=1000,
                description="Maximum motor neurons"
            ),
            'n_hidden_min': Parameter(
                'n_hidden_min', ParamType.INT, 50,
                min_val=0, max_val=100000,
                description="Minimum hidden neurons"
            ),
            'n_hidden_max': Parameter(
                'n_hidden_max', ParamType.INT, 200,
                min_val=0, max_val=100000,
                description="Maximum hidden neurons"
            ),
            'population_size': Parameter(
                'population_size', ParamType.INT, 30,
                min_val=2, max_val=10000,
                description="Initial population size"
            ),
            'tournament_size': Parameter(
                'tournament_size', ParamType.INT, 5,
                min_val=2, max_val=100,
                description="Tournament selection size"
            ),
        }
        
        # Energy/metabolism parameters (Phase 3)
        self.params['energy'] = {
            'initial_energy': Parameter(
                'initial_energy', ParamType.FLOAT, 1000000.0,
                min_val=0.0,
                description="Starting energy for new creatures"
            ),
            'max_energy': Parameter(
                'max_energy', ParamType.FLOAT, 10000000.0,
                min_val=0.0,
                description="Maximum energy capacity"
            ),
            'base_metabolism': Parameter(
                'base_metabolism', ParamType.FLOAT, 50.0,
                min_val=0.0,
                description="Base metabolic cost per timestep"
            ),
            'neuron_existence_cost': Parameter(
                'neuron_existence_cost', ParamType.FLOAT, 0.01,
                min_val=0.0,
                description="Energy cost per neuron per timestep"
            ),
            'synapse_existence_cost': Parameter(
                'synapse_existence_cost', ParamType.FLOAT, 0.001,
                min_val=0.0,
                description="Energy cost per synapse per timestep"
            ),
            'neuron_firing_cost': Parameter(
                'neuron_firing_cost', ParamType.FLOAT, 1.0,
                min_val=0.0,
                description="Energy cost when neuron fires"
            ),
            'motor_activation_cost': Parameter(
                'motor_activation_cost', ParamType.FLOAT, 10.0,
                min_val=0.0,
                description="Energy cost for motor activation"
            ),
            'food_energy_value': Parameter(
                'food_energy_value', ParamType.FLOAT, 100000.0,
                min_val=0.0,
                description="Energy gained from food (heritage: 100000)"
            ),
            'reproduction_cost': Parameter(
                'reproduction_cost', ParamType.FLOAT, 500000.0,
                min_val=0.0,
                description="Energy cost to reproduce"
            ),
        }
        
        # Drug/pharmacology parameters (Phase 3)
        self.params['drugs'] = {
            'decay_rate': Parameter(
                'decay_rate', ParamType.FLOAT, 0.99,
                min_val=0.0, max_val=1.0,
                description="Drug decay per timestep (heritage: 0.99)"
            ),
            'potentiator_multiplier': Parameter(
                'potentiator_multiplier', ParamType.FLOAT, 10.0,
                min_val=1.0, max_val=100.0,
                description="Potentiator amplification factor (heritage: 10x)"
            ),
            'drug_effect_strength': Parameter(
                'drug_effect_strength', ParamType.FLOAT, 1.0,
                min_val=0.0, max_val=10.0,
                description="Global drug effect multiplier"
            ),
        }
        
        # Circuit8 / morphic field parameters (Phase 3)
        self.params['morphic'] = {
            'circuit8_width': Parameter(
                'circuit8_width', ParamType.INT, 64,
                min_val=8, max_val=1024,
                description="Circuit8 canvas width (heritage: 64)"
            ),
            'circuit8_height': Parameter(
                'circuit8_height', ParamType.INT, 48,
                min_val=8, max_val=1024,
                description="Circuit8 canvas height (heritage: 48)"
            ),
            'circuit8_depth': Parameter(
                'circuit8_depth', ParamType.INT, 1024,
                min_val=1, max_val=10000,
                description="Circuit8 depth buffer layers (heritage: 1024)"
            ),
            'circuit8_decay_rate': Parameter(
                'circuit8_decay_rate', ParamType.FLOAT, 0.95,
                min_val=0.0, max_val=1.0,
                description="Circuit8 pixel fade rate per timestep"
            ),
        }
        
        # Morphology parameters (Phase 9a)
        self.params['morphology'] = {
            'min_segments': Parameter(
                'min_segments', ParamType.INT, 2,
                min_val=1, max_val=50,
                description="Minimum body segments"
            ),
            'max_segments': Parameter(
                'max_segments', ParamType.INT, 10,
                min_val=1, max_val=50,
                description="Maximum body segments"
            ),
            'max_limbs_per_segment': Parameter(
                'max_limbs_per_segment', ParamType.INT, 4,
                min_val=0, max_val=20,
                description="Maximum limbs per segment"
            ),
            'body_mutation_rate': Parameter(
                'body_mutation_rate', ParamType.FLOAT, 0.1,
                min_val=0.0, max_val=1.0,
                description="Body morphology mutation rate"
            ),
            'body_mass_energy_cost': Parameter(
                'body_mass_energy_cost', ParamType.FLOAT, 0.01,
                min_val=0.0,
                description="Energy cost per unit body mass"
            ),
        }
        
        # World/simulation parameters
        self.params['world'] = {
            'world_width': Parameter(
                'world_width', ParamType.FLOAT, 1000.0,
                min_val=100.0, max_val=100000.0,
                description="World width"
            ),
            'world_height': Parameter(
                'world_height', ParamType.FLOAT, 1000.0,
                min_val=100.0, max_val=100000.0,
                description="World height"
            ),
            'timestep': Parameter(
                'timestep', ParamType.FLOAT, 1.0,
                min_val=0.001, max_val=100.0,
                description="Simulation timestep (dt)"
            ),
        }
        
        # Visualization parameters
        self.params['visualization'] = {
            'window_width': Parameter(
                'window_width', ParamType.INT, 1400,
                min_val=640, max_val=7680,
                description="Window width"
            ),
            'window_height': Parameter(
                'window_height', ParamType.INT, 900,
                min_val=480, max_val=4320,
                description="Window height"
            ),
            'target_fps': Parameter(
                'target_fps', ParamType.INT, 30,
                min_val=1, max_val=240,
                description="Target frames per second"
            ),
            'camera_distance': Parameter(
                'camera_distance', ParamType.FLOAT, 800.0,
                min_val=10.0, max_val=10000.0,
                description="Initial camera distance"
            ),
            'camera_azimuth': Parameter(
                'camera_azimuth', ParamType.FLOAT, 45.0,
                min_val=-180.0, max_val=180.0,
                description="Initial camera azimuth angle"
            ),
            'camera_elevation': Parameter(
                'camera_elevation', ParamType.FLOAT, 35.0,
                min_val=-90.0, max_val=90.0,
                description="Initial camera elevation angle"
            ),
        }
    
    def get_param(self, section: str, name: str) -> Optional[Parameter]:
        """Get parameter definition.
        
        Args:
            section: Parameter section (e.g. 'neural')
            name: Parameter name
            
        Returns:
            Parameter definition or None if not found
        """
        return self.params.get(section, {}).get(name)
    
    def get_default_config(self) -> Dict[str, Dict[str, Any]]:
        """Get complete default configuration.
        
        Returns:
            Dictionary of all default values organized by section
        """
        config = {}
        for section, params in self.params.items():
            config[section] = {}
            for name, param in params.items():
                config[section][name] = param.default
        return config
    
    def validate_config(self, config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Validate and normalize configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validated configuration
            
        Raises:
            ValueError: If validation fails
        """
        validated = {}
        for section, params in config.items():
            if section not in self.params:
                raise ValueError(f"Unknown configuration section: {section}")
            
            validated[section] = {}
            for name, value in params.items():
                param = self.get_param(section, name)
                if param is None:
                    raise ValueError(f"Unknown parameter: {section}.{name}")
                
                validated[section][name] = param.validate(value)
        
        return validated
    
    def get_documentation(self) -> str:
        """Generate documentation for all parameters.
        
        Returns:
            Markdown-formatted documentation string
        """
        lines = ["# CritterGOD Configuration Parameters\n"]
        
        for section, params in self.params.items():
            lines.append(f"\n## {section.title()}\n")
            for name, param in params.items():
                lines.append(f"- **{name}**: {param.description}")
                lines.append(f"  - Type: {param.type.value}")
                lines.append(f"  - Default: {param.default}")
                if param.min_val is not None:
                    lines.append(f"  - Min: {param.min_val}")
                if param.max_val is not None:
                    lines.append(f"  - Max: {param.max_val}")
                lines.append("")
        
        return "\n".join(lines)
