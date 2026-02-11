"""
Statistics tracking system for CritterGOD.

Tracks performance metrics, population statistics, and evolutionary data
for real-time visualization and analysis.
"""

from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import time


class StatisticsTracker:
    """
    Tracks all simulation metrics.
    
    Maintains time-series data for visualization and analysis.
    Automatically manages history size to prevent memory bloat.
    
    Attributes:
        metrics: Time-series data for each metric
        max_history: Maximum data points to keep
        start_time: Simulation start timestamp
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize statistics tracker.
        
        Args:
            max_history: Maximum data points per metric
        """
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.max_history = max_history
        self.start_time = time.time()
        
        # Performance tracking
        self.frame_times = deque(maxlen=60)  # Last 60 frames for FPS
        self.last_frame_time = time.time()
    
    def record(self, metric: str, value: float, timestep: int):
        """
        Record a metric value.
        
        Args:
            metric: Metric name
            value: Metric value
            timestep: Current simulation timestep
        """
        self.metrics[metric].append((timestep, value))
    
    def record_batch(self, values: Dict[str, float], timestep: int):
        """
        Record multiple metrics at once.
        
        Args:
            values: Dictionary of metric names to values
            timestep: Current simulation timestep
        """
        for metric, value in values.items():
            self.record(metric, value, timestep)
    
    def get_latest(self, metric: str) -> Optional[float]:
        """
        Get most recent value for metric.
        
        Args:
            metric: Metric name
            
        Returns:
            Latest value or None if no data
        """
        if metric in self.metrics and len(self.metrics[metric]) > 0:
            return self.metrics[metric][-1][1]
        return None
    
    def get_history(self, metric: str, last_n: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Get historical data for metric.
        
        Args:
            metric: Metric name
            last_n: Number of recent values (None = all)
            
        Returns:
            List of (timestep, value) tuples
        """
        if metric not in self.metrics:
            return []
        
        data = list(self.metrics[metric])
        if last_n is not None:
            data = data[-last_n:]
        return data
    
    def get_average(self, metric: str, last_n: int = 100) -> float:
        """
        Get average value over recent history.
        
        Args:
            metric: Metric name
            last_n: Number of recent values to average
            
        Returns:
            Average value or 0.0 if no data
        """
        history = self.get_history(metric, last_n)
        if not history:
            return 0.0
        values = [v for _, v in history]
        return sum(values) / len(values)
    
    def get_min_max(self, metric: str, last_n: int = 100) -> Tuple[float, float]:
        """
        Get min and max values over recent history.
        
        Args:
            metric: Metric name
            last_n: Number of recent values
            
        Returns:
            (min_value, max_value) tuple
        """
        history = self.get_history(metric, last_n)
        if not history:
            return (0.0, 0.0)
        values = [v for _, v in history]
        return (min(values), max(values))
    
    def update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.frame_times.append(frame_time)
        self.last_frame_time = current_time
    
    def get_fps(self) -> float:
        """
        Get current FPS.
        
        Returns:
            Frames per second
        """
        if len(self.frame_times) == 0:
            return 0.0
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        if avg_frame_time > 0:
            return 1.0 / avg_frame_time
        return 0.0
    
    def get_runtime(self) -> float:
        """
        Get simulation runtime in seconds.
        
        Returns:
            Seconds since start
        """
        return time.time() - self.start_time
    
    def clear(self):
        """Clear all statistics."""
        self.metrics.clear()
        self.frame_times.clear()
        self.start_time = time.time()
    
    def get_all_metrics(self) -> List[str]:
        """Get list of all tracked metric names."""
        return sorted(list(self.metrics.keys()))
    
    def export_to_dict(self) -> Dict:
        """
        Export all statistics to dictionary.
        
        Returns:
            Dictionary with all metrics and metadata
        """
        return {
            'runtime': self.get_runtime(),
            'fps': self.get_fps(),
            'metrics': {
                name: list(data)
                for name, data in self.metrics.items()
            }
        }


class PopulationStats:
    """Helper for calculating population statistics."""
    
    @staticmethod
    def calculate_stats(creatures) -> Dict[str, float]:
        """
        Calculate population statistics.
        
        Args:
            creatures: List of creatures
            
        Returns:
            Dictionary of statistics
        """
        if not creatures:
            return {
                'count': 0,
                'avg_energy': 0.0,
                'avg_age': 0.0,
                'avg_generation': 0.0,
                'avg_neurons': 0.0,
                'avg_synapses': 0.0,
                'avg_adam_distance': 0.0
            }
        
        stats = {
            'count': len(creatures),
            'avg_energy': sum(c.energy.energy for c in creatures) / len(creatures),
            'avg_age': sum(c.age for c in creatures) / len(creatures),
            'avg_generation': sum(c.generation for c in creatures) / len(creatures),
            'avg_neurons': sum(len(c.network.neurons) for c in creatures) / len(creatures),
            'avg_synapses': sum(len(c.network.synapses) for c in creatures) / len(creatures),
        }
        
        # Adam distance (if available)
        adam_distances = [c.adam_distance for c in creatures if hasattr(c, 'adam_distance')]
        if adam_distances:
            stats['avg_adam_distance'] = sum(adam_distances) / len(adam_distances)
        else:
            stats['avg_adam_distance'] = 0.0
        
        return stats
