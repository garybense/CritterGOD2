"""Configuration system for CritterGOD."""

from core.config.parameters import Parameter, ParameterCategory, create_default_parameters
from core.config.config_manager import ConfigManager

__all__ = ['Parameter', 'ParameterCategory', 'create_default_parameters', 'ConfigManager']
