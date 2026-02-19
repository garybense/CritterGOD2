"""
GPU-Accelerated Neural Network Module

Phase 10b: High-performance neural network computation for 10,000+ neuron networks.

Supported backends:
- CUDA (via CuPy) - NVIDIA GPUs
- OpenCL (via PyOpenCL) - Cross-platform GPUs
- NumPy - CPU fallback (vectorized)

Usage:
    from core.neural.gpu import GPUNeuralNetwork, get_available_backends, GPUConfig

    # Check available backends
    backends = get_available_backends()

    # Create GPU-accelerated network
    config = GPUConfig(backend='cuda', device_id=0)
    network = GPUNeuralNetwork(n_neurons=10000, config=config)
"""

from .config import GPUConfig, get_available_backends, get_best_backend
from .gpu_network import GPUNeuralNetwork
from .backend import Backend, get_backend

__all__ = [
    "GPUConfig",
    "GPUNeuralNetwork",
    "Backend",
    "get_available_backends",
    "get_best_backend",
    "get_backend",
]
