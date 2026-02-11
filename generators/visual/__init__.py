"""
Visual pattern generation for neural-driven generative art.

Based on SDL neural visualizers (looser.c, xesu.c, etc.).
Creates procedural patterns from neural network state with retinal feedback.
"""

from generators.visual.pattern_generators import PatternGenerator, PatternParams
from generators.visual.neural_parameters import NeuralPatternMapper
from generators.visual.retinal_sensors import RetinalSensor, RetinalArray
from generators.visual.visual_pipeline import VisualPipeline, VisualPipelineStats

__all__ = [
    'PatternGenerator',
    'PatternParams',
    'NeuralPatternMapper',
    'RetinalSensor',
    'RetinalArray',
    'VisualPipeline',
    'VisualPipelineStats',
]
