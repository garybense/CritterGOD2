"""
Configuration Manager for CritterGOD.

Central system for managing all tunable parameters with
save/load functionality inspired by critterdrug profiles.
"""

import json
from typing import Dict, Optional, List
from pathlib import Path

from core.config.parameters import (
    Parameter,
    ParameterCategory,
    create_default_parameters,
    get_parameters_by_category
)


class ConfigManager:
    """
    Central configuration management system.
    
    Manages all tunable parameters, profile loading/saving,
    and runtime parameter updates.
    
    Attributes:
        parameters: Dictionary of all parameters
        profile_name: Currently loaded profile name
        profile_dir: Directory containing profile files
    """
    
    def __init__(self, profile_dir: str = "profiles"):
        """
        Initialize configuration manager.
        
        Args:
            profile_dir: Directory for profile files
        """
        self.parameters: Dict[str, Parameter] = create_default_parameters()
        self.profile_name = "default"
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(exist_ok=True)
        
        # Track which parameters have changed since last save
        self.dirty_params: set = set()
    
    def get(self, name: str) -> float:
        """
        Get parameter value.
        
        Args:
            name: Parameter name
            
        Returns:
            Parameter value
            
        Raises:
            KeyError: If parameter doesn't exist
        """
        return self.parameters[name].get_value()
    
    def get_int(self, name: str) -> int:
        """
        Get parameter value as integer.
        
        Args:
            name: Parameter name
            
        Returns:
            Parameter value as int
        """
        return self.parameters[name].get_int_value()
    
    def set(self, name: str, value: float):
        """
        Set parameter value.
        
        Args:
            name: Parameter name
            value: New value (will be clamped to valid range)
        """
        if name in self.parameters:
            self.parameters[name].set_value(value)
            self.dirty_params.add(name)
    
    def reset(self, name: str):
        """Reset parameter to default value."""
        if name in self.parameters:
            self.parameters[name].reset()
            self.dirty_params.add(name)
    
    def reset_all(self):
        """Reset all parameters to defaults."""
        for param in self.parameters.values():
            param.reset()
        self.dirty_params = set(self.parameters.keys())
    
    def get_parameter(self, name: str) -> Optional[Parameter]:
        """
        Get Parameter object.
        
        Args:
            name: Parameter name
            
        Returns:
            Parameter object or None if not found
        """
        return self.parameters.get(name)
    
    def get_all_parameters(self) -> Dict[str, Parameter]:
        """Get all parameters."""
        return self.parameters.copy()
    
    def get_parameters_by_category(self) -> Dict[ParameterCategory, List[Parameter]]:
        """Get parameters grouped by category."""
        return get_parameters_by_category(self.parameters)
    
    def save_profile(self, profile_name: str, description: str = ""):
        """
        Save current configuration to profile.
        
        Args:
            profile_name: Name for the profile
            description: Optional description
        """
        profile_data = {
            'name': profile_name,
            'description': description,
            'parameters': {}
        }
        
        for name, param in self.parameters.items():
            profile_data['parameters'][name] = param.value
        
        profile_path = self.profile_dir / f"{profile_name}.json"
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        self.profile_name = profile_name
        self.dirty_params.clear()
        print(f"✅ Saved profile: {profile_name}")
    
    def load_profile(self, profile_name: str) -> bool:
        """
        Load configuration from profile.
        
        Args:
            profile_name: Name of profile to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        profile_path = self.profile_dir / f"{profile_name}.json"
        
        if not profile_path.exists():
            print(f"❌ Profile not found: {profile_name}")
            return False
        
        try:
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)
            
            # Load parameter values
            loaded_count = 0
            for name, value in profile_data.get('parameters', {}).items():
                if name in self.parameters:
                    self.parameters[name].set_value(value)
                    loaded_count += 1
            
            self.profile_name = profile_name
            self.dirty_params.clear()
            
            print(f"✅ Loaded profile: {profile_name} ({loaded_count} parameters)")
            return True
            
        except Exception as e:
            print(f"❌ Error loading profile {profile_name}: {e}")
            return False
    
    def list_profiles(self) -> List[str]:
        """
        List available profiles.
        
        Returns:
            List of profile names (without .json extension)
        """
        profiles = []
        for path in self.profile_dir.glob("*.json"):
            profiles.append(path.stem)
        return sorted(profiles)
    
    def export_to_dict(self) -> Dict:
        """
        Export configuration to dictionary.
        
        Returns:
            Dictionary with all parameters
        """
        return {
            'profile_name': self.profile_name,
            'parameters': {
                name: param.to_dict()
                for name, param in self.parameters.items()
            }
        }
    
    def has_unsaved_changes(self) -> bool:
        """Check if there are unsaved parameter changes."""
        return len(self.dirty_params) > 0
    
    def get_dirty_params(self) -> List[str]:
        """Get list of parameters that have changed."""
        return list(self.dirty_params)
