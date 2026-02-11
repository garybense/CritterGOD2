"""
Social learning system for collective intelligence.

Enables creatures to learn from each other's successes and failures:
- Observe successful behaviors
- Copy effective strategies
- Avoid failed strategies
- Build collective knowledge over time

Creates emergent cultural evolution.
"""

from typing import Dict, List, Optional
import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class BehaviorObservation:
    """
    Observation of another creature's behavior.
    
    Attributes:
        creature_id: ID of observed creature
        behavior: Behavior type observed
        outcome: Success/failure result
        energy_change: Energy gained/lost
        timestep: When observed
        location: Where observed (x, y)
    """
    creature_id: int
    behavior: int  # BehaviorType
    outcome: str  # 'success' or 'failure'
    energy_change: float
    timestep: int
    location: tuple  # (x, y)


class SocialLearner:
    """
    Learns from observations of other creatures.
    
    Tracks successful behaviors and adapts own behavior based
    on what works for others.
    
    Attributes:
        observation_history: Recent observations
        behavior_success_rates: Success rate by behavior type
        learning_rate: How quickly to adapt (0-1)
        memory_size: How many observations to remember
    """
    
    def __init__(self, creature_id: int, memory_size: int = 50):
        """
        Initialize social learner.
        
        Args:
            creature_id: This creature's ID
            memory_size: Max observations to remember
        """
        self.creature_id = creature_id
        self.memory_size = memory_size
        
        # Observation history (FIFO queue)
        self.observation_history: deque = deque(maxlen=memory_size)
        
        # Behavior success tracking
        self.behavior_success_rates: Dict[int, Dict] = {}
        
        # Learning parameters
        self.learning_rate = 0.1  # How fast to learn
        self.min_observations = 3  # Min observations before learning
        
        # Imitation probability (increases with observations)
        self.base_imitation_rate = 0.1
        self.max_imitation_rate = 0.8
    
    def observe_behavior(self, observation: BehaviorObservation):
        """
        Record observation of another creature's behavior.
        
        Args:
            observation: Behavior observation
        """
        # Add to history
        self.observation_history.append(observation)
        
        # Update success rates
        behavior = observation.behavior
        if behavior not in self.behavior_success_rates:
            self.behavior_success_rates[behavior] = {
                'successes': 0,
                'failures': 0,
                'total_energy_gained': 0.0,
                'total_energy_lost': 0.0,
                'observations': 0
            }
        
        stats = self.behavior_success_rates[behavior]
        stats['observations'] += 1
        
        if observation.outcome == 'success':
            stats['successes'] += 1
            stats['total_energy_gained'] += observation.energy_change
        else:
            stats['failures'] += 1
            stats['total_energy_lost'] += abs(observation.energy_change)
    
    def get_behavior_success_rate(self, behavior: int) -> float:
        """
        Get success rate for behavior type.
        
        Args:
            behavior: Behavior type
            
        Returns:
            Success rate (0-1), or 0.5 if unknown
        """
        if behavior not in self.behavior_success_rates:
            return 0.5  # Neutral if unknown
        
        stats = self.behavior_success_rates[behavior]
        if stats['observations'] < self.min_observations:
            return 0.5  # Not enough data
        
        total = stats['successes'] + stats['failures']
        if total == 0:
            return 0.5
        
        return stats['successes'] / total
    
    def should_imitate_behavior(self, behavior: int) -> bool:
        """
        Decide whether to imitate observed behavior.
        
        Args:
            behavior: Behavior type
            
        Returns:
            True if should imitate
        """
        success_rate = self.get_behavior_success_rate(behavior)
        
        # Calculate imitation probability based on success rate
        if success_rate < 0.3:
            # Low success - unlikely to imitate
            imitation_prob = self.base_imitation_rate * 0.5
        elif success_rate > 0.7:
            # High success - very likely to imitate
            imitation_prob = min(
                self.max_imitation_rate,
                self.base_imitation_rate + (success_rate - 0.5) * 2.0
            )
        else:
            # Medium success - base probability
            imitation_prob = self.base_imitation_rate
        
        return np.random.random() < imitation_prob
    
    def get_recommended_behavior(self, current_energy: float) -> Optional[int]:
        """
        Get recommended behavior based on social learning.
        
        Args:
            current_energy: Creature's current energy
            
        Returns:
            Recommended behavior type, or None
        """
        if not self.behavior_success_rates:
            return None
        
        # Filter behaviors with enough observations
        viable_behaviors = [
            (behavior, self.get_behavior_success_rate(behavior))
            for behavior, stats in self.behavior_success_rates.items()
            if stats['observations'] >= self.min_observations
        ]
        
        if not viable_behaviors:
            return None
        
        # Weight by success rate and energy gain
        best_behavior = None
        best_score = -float('inf')
        
        for behavior, success_rate in viable_behaviors:
            stats = self.behavior_success_rates[behavior]
            
            # Calculate average energy gain
            if stats['successes'] > 0:
                avg_gain = stats['total_energy_gained'] / stats['successes']
            else:
                avg_gain = 0.0
            
            # Score = success_rate * avg_energy_gain
            score = success_rate * avg_gain
            
            # Bonus if low energy (prioritize energy-gaining behaviors)
            if current_energy < 500000:
                score *= 1.5
            
            if score > best_score:
                best_score = score
                best_behavior = behavior
        
        return best_behavior
    
    def get_collective_knowledge_summary(self) -> Dict:
        """
        Get summary of collective knowledge learned.
        
        Returns:
            Dictionary of learned behavior patterns
        """
        summary = {}
        
        for behavior, stats in self.behavior_success_rates.items():
            if stats['observations'] >= self.min_observations:
                success_rate = self.get_behavior_success_rate(behavior)
                
                summary[behavior] = {
                    'success_rate': success_rate,
                    'observations': stats['observations'],
                    'avg_energy_gained': (
                        stats['total_energy_gained'] / max(1, stats['successes'])
                    ),
                    'confidence': min(1.0, stats['observations'] / 20.0)
                }
        
        return summary


class CollectiveMemory:
    """
    Shared collective memory across all creatures.
    
    Stores knowledge that persists beyond individual lifetimes,
    enabling cultural transmission.
    
    Attributes:
        global_behavior_stats: Aggregated behavior statistics
        resource_locations: Known resource locations
        danger_zones: Known dangerous areas
    """
    
    def __init__(self):
        """Initialize collective memory."""
        # Global behavior statistics
        self.global_behavior_stats: Dict[int, Dict] = {}
        
        # Known resource locations (fades over time)
        self.resource_locations: List[Dict] = []
        
        # Danger zones (areas where creatures died)
        self.danger_zones: List[Dict] = []
        
        # Success stories (highly successful behaviors)
        self.success_stories: List[Dict] = []
    
    def record_behavior_outcome(
        self,
        behavior: int,
        outcome: str,
        energy_change: float,
        location: tuple
    ):
        """
        Record behavior outcome to collective memory.
        
        Args:
            behavior: Behavior type
            outcome: 'success' or 'failure'
            energy_change: Energy change
            location: Where it happened
        """
        if behavior not in self.global_behavior_stats:
            self.global_behavior_stats[behavior] = {
                'successes': 0,
                'failures': 0,
                'total_observations': 0,
                'avg_energy_gain': 0.0
            }
        
        stats = self.global_behavior_stats[behavior]
        stats['total_observations'] += 1
        
        if outcome == 'success':
            stats['successes'] += 1
            # Update running average
            n = stats['successes']
            stats['avg_energy_gain'] = (
                stats['avg_energy_gain'] * (n - 1) / n +
                energy_change / n
            )
        else:
            stats['failures'] += 1
    
    def record_resource_location(
        self,
        resource_type: str,
        x: float,
        y: float,
        timestep: int
    ):
        """
        Record resource location in collective memory.
        
        Args:
            resource_type: 'food' or 'drug'
            x, y: Location
            timestep: When discovered
        """
        self.resource_locations.append({
            'type': resource_type,
            'x': x,
            'y': y,
            'timestep': timestep,
            'confidence': 1.0
        })
        
        # Limit size
        if len(self.resource_locations) > 100:
            self.resource_locations = self.resource_locations[-100:]
    
    def record_danger_zone(self, x: float, y: float, radius: float = 50.0):
        """
        Record dangerous area.
        
        Args:
            x, y: Location
            radius: Danger radius
        """
        self.danger_zones.append({
            'x': x,
            'y': y,
            'radius': radius,
            'strength': 1.0
        })
        
        # Limit size
        if len(self.danger_zones) > 50:
            self.danger_zones = self.danger_zones[-50:]
    
    def is_dangerous_location(self, x: float, y: float) -> bool:
        """
        Check if location is known to be dangerous.
        
        Args:
            x, y: Location to check
            
        Returns:
            True if dangerous
        """
        for zone in self.danger_zones:
            dist_sq = (x - zone['x'])**2 + (y - zone['y'])**2
            if dist_sq < zone['radius']**2:
                return True
        return False
    
    def get_nearby_known_resources(
        self,
        x: float,
        y: float,
        radius: float = 100.0,
        max_age: int = 1000
    ) -> List[Dict]:
        """
        Get known resource locations near position.
        
        Args:
            x, y: Search center
            radius: Search radius
            max_age: Max timesteps old
            
        Returns:
            List of nearby resources
        """
        from time import time
        current_time = int(time())  # Approximation
        
        nearby = []
        for resource in self.resource_locations:
            age = current_time - resource['timestep']
            if age > max_age:
                continue
            
            dist_sq = (x - resource['x'])**2 + (y - resource['y'])**2
            if dist_sq < radius**2:
                nearby.append({
                    **resource,
                    'distance': np.sqrt(dist_sq),
                    'age': age
                })
        
        return sorted(nearby, key=lambda r: r['distance'])
    
    def get_global_success_rate(self, behavior: int) -> float:
        """
        Get global success rate for behavior.
        
        Args:
            behavior: Behavior type
            
        Returns:
            Success rate (0-1)
        """
        if behavior not in self.global_behavior_stats:
            return 0.5
        
        stats = self.global_behavior_stats[behavior]
        total = stats['successes'] + stats['failures']
        if total == 0:
            return 0.5
        
        return stats['successes'] / total
