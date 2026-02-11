"""
Configuration loader for CritterGOD profiles.

Loads YAML configuration files with:
- Profile inheritance
- Command-line overrides
- Validation
- Merging multiple profiles

Heritage from Critterding's profile system.
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from config.schema import ConfigSchema


class ConfigLoader:
    """Load and manage configuration profiles.
    
    Attributes:
        schema: Configuration schema for validation
        profiles_dir: Directory containing profile files
    """
    
    def __init__(self, profiles_dir: Optional[str] = None):
        """Initialize configuration loader.
        
        Args:
            profiles_dir: Directory containing profiles (default: ./profiles)
        """
        self.schema = ConfigSchema()
        
        if profiles_dir is None:
            # Default to profiles/ directory in project root
            project_root = Path(__file__).parent.parent
            self.profiles_dir = project_root / "profiles"
        else:
            self.profiles_dir = Path(profiles_dir)
        
        # Ensure profiles directory exists
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
    
    def load_profile(self, profile_name: str) -> Dict[str, Dict[str, Any]]:
        """Load a configuration profile from file.
        
        Args:
            profile_name: Name of profile (without .yaml extension)
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If profile file doesn't exist
            ValueError: If profile is invalid
        """
        profile_path = self.profiles_dir / f"{profile_name}.yaml"
        
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")
        
        with open(profile_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            config = {}
        
        # Handle profile inheritance
        if 'inherit' in config:
            base_profile = config.pop('inherit')
            base_config = self.load_profile(base_profile)
            config = self.merge_configs(base_config, config)
        
        # Validate configuration
        config = self.schema.validate_config(config)
        
        return config
    
    def save_profile(self, config: Dict[str, Dict[str, Any]], profile_name: str) -> None:
        """Save configuration to a profile file.
        
        Args:
            config: Configuration dictionary
            profile_name: Name of profile (without .yaml extension)
        """
        profile_path = self.profiles_dir / f"{profile_name}.yaml"
        
        # Validate before saving
        config = self.schema.validate_config(config)
        
        with open(profile_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def merge_configs(self, base: Dict[str, Dict[str, Any]], 
                     override: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Merge two configuration dictionaries.
        
        Override values take precedence over base values.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        merged = {}
        
        # Start with all base sections
        for section in base:
            merged[section] = base[section].copy()
        
        # Override with new values
        for section, params in override.items():
            if section not in merged:
                merged[section] = {}
            merged[section].update(params)
        
        return merged
    
    def apply_overrides(self, config: Dict[str, Dict[str, Any]], 
                       overrides: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Apply command-line overrides to configuration.
        
        Overrides format: {"neural.inhibitory_ratio": "0.25", ...}
        
        Args:
            config: Base configuration
            overrides: Dictionary of overrides
            
        Returns:
            Configuration with overrides applied
            
        Raises:
            ValueError: If override is invalid
        """
        config = config.copy()
        
        for key, value in overrides.items():
            # Parse key (e.g. "neural.inhibitory_ratio")
            parts = key.split('.')
            if len(parts) != 2:
                raise ValueError(f"Invalid override key: {key} (expected section.parameter)")
            
            section, param_name = parts
            
            # Get parameter definition
            param = self.schema.get_param(section, param_name)
            if param is None:
                raise ValueError(f"Unknown parameter: {key}")
            
            # Validate and convert value
            validated_value = param.validate(value)
            
            # Apply override
            if section not in config:
                config[section] = {}
            config[section][param_name] = validated_value
        
        return config
    
    def list_profiles(self) -> List[str]:
        """List available profile names.
        
        Returns:
            List of profile names (without .yaml extension)
        """
        profiles = []
        for path in self.profiles_dir.glob("*.yaml"):
            profiles.append(path.stem)
        return sorted(profiles)
    
    def get_default_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default configuration from schema.
        
        Returns:
            Default configuration dictionary
        """
        return self.schema.get_default_config()
    
    def load_or_default(self, profile_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Load profile or return default if not found.
        
        Args:
            profile_name: Profile name (None for default)
            
        Returns:
            Configuration dictionary
        """
        if profile_name is None:
            return self.get_default_config()
        
        try:
            return self.load_profile(profile_name)
        except FileNotFoundError:
            print(f"Profile '{profile_name}' not found, using defaults")
            return self.get_default_config()


class Config:
    """Configuration wrapper with easy access to parameters.
    
    Usage:
        config = Config.from_profile('psychedelic')
        threshold = config.neural.neuron_threshold_min
        config.neural.inhibitory_ratio = 0.35
    """
    
    def __init__(self, config_dict: Dict[str, Dict[str, Any]]):
        """Initialize configuration wrapper.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
        
        # Create attribute access for each section
        for section, params in config_dict.items():
            setattr(self, section, ConfigSection(section, params))
    
    @classmethod
    def from_profile(cls, profile_name: str, loader: Optional[ConfigLoader] = None) -> 'Config':
        """Load configuration from profile.
        
        Args:
            profile_name: Profile name
            loader: ConfigLoader instance (creates new if None)
            
        Returns:
            Config instance
        """
        if loader is None:
            loader = ConfigLoader()
        config_dict = loader.load_profile(profile_name)
        return cls(config_dict)
    
    @classmethod
    def default(cls) -> 'Config':
        """Get default configuration.
        
        Returns:
            Config instance with default values
        """
        loader = ConfigLoader()
        config_dict = loader.get_default_config()
        return cls(config_dict)
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config
    
    def save(self, profile_name: str, loader: Optional[ConfigLoader] = None) -> None:
        """Save configuration to profile.
        
        Args:
            profile_name: Profile name
            loader: ConfigLoader instance (creates new if None)
        """
        if loader is None:
            loader = ConfigLoader()
        loader.save_profile(self._config, profile_name)


class ConfigSection:
    """Configuration section with attribute access.
    
    Usage:
        section = ConfigSection('neural', {'inhibitory_ratio': 0.3})
        ratio = section.inhibitory_ratio  # 0.3
        section.inhibitory_ratio = 0.35   # Modify
    """
    
    def __init__(self, name: str, params: Dict[str, Any]):
        """Initialize configuration section.
        
        Args:
            name: Section name
            params: Parameter dictionary
        """
        self._name = name
        self._params = params
    
    def __getattr__(self, name: str) -> Any:
        """Get parameter value.
        
        Args:
            name: Parameter name
            
        Returns:
            Parameter value
            
        Raises:
            AttributeError: If parameter doesn't exist
        """
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        if name not in self._params:
            raise AttributeError(f"Parameter '{name}' not found in section '{self._name}'")
        
        return self._params[name]
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set parameter value.
        
        Args:
            name: Parameter name
            value: New value
        """
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._params[name] = value
    
    def __repr__(self) -> str:
        return f"ConfigSection({self._name}, {self._params})"
