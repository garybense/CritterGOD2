"""
Configuration system for CritterGOD.

Profile-based configuration with YAML files.
Heritage from Critterding's profile system.
"""

from config.schema import ConfigSchema, Parameter, ParamType
from config.loader import ConfigLoader, Config, ConfigSection

__all__ = [
    'ConfigSchema',
    'Parameter',
    'ParamType',
    'ConfigLoader',
    'Config',
    'ConfigSection',
]
