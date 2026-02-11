"""
Neural Network Module

Spiking neural networks with STDP plasticity and dynamic rewiring.
"""

from .neuron import Neuron
from .synapse import Synapse
from .network import NeuralNetwork

__all__ = ["Neuron", "Synapse", "NeuralNetwork"]
