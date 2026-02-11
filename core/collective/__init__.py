"""
Collective intelligence module.

Enables social learning and communication through Circuit8:
- Behavior broadcasting
- Social learning from observations
- Resource location sharing
- Collective memory
"""

from core.collective.behavior_broadcast import (
    BehaviorBroadcaster,
    BehaviorReader,
    BehaviorType,
    SignalType,
    fade_circuit8,
)

from core.collective.social_learning import (
    SocialLearner,
    CollectiveMemory,
    BehaviorObservation,
)

__all__ = [
    'BehaviorBroadcaster',
    'BehaviorReader',
    'BehaviorType',
    'SignalType',
    'fade_circuit8',
    'SocialLearner',
    'CollectiveMemory',
    'BehaviorObservation',
]
