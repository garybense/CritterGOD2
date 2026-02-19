"""
Neural Network Module

Spiking neural networks with STDP plasticity and dynamic rewiring.

Phase 10b adds GPU acceleration for 10,000+ neuron networks:
- GPUNeuralNetwork: Vectorized, backend-agnostic neural network
- Supports CUDA (via CuPy), OpenCL, and NumPy backends
- Automatic device selection and memory management

Usage:
    # CPU version (original)
    from core.neural import NeuralNetwork
    
    # GPU-accelerated version (Phase 10b)
    from core.neural.gpu import GPUNeuralNetwork, GPUConfig
"""

from .neuron import Neuron, NeuronType
from .synapse import Synapse
from .network import NeuralNetwork

# GPU module available via: from core.neural.gpu import ...
# Not imported here to avoid requiring GPU dependencies

__all__ = ["Neuron", "NeuronType", "Synapse", "NeuralNetwork"]
