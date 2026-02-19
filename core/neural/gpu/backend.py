"""
Backend Abstraction Layer

Provides a unified interface for array operations across:
- NumPy (CPU)
- CuPy (CUDA)
- PyOpenCL (OpenCL)

This allows the neural network code to be backend-agnostic.
"""

import numpy as np
from typing import Optional, Tuple, Any
from abc import ABC, abstractmethod
import logging

from .config import BackendType, GPUConfig

logger = logging.getLogger(__name__)


class Backend(ABC):
    """
    Abstract base class for compute backends.
    
    Provides array operations and neural network primitives.
    """
    
    def __init__(self, config: GPUConfig):
        self.config = config
        self.dtype = np.float16 if config.use_fp16 else np.float32
        
    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        pass
    
    @abstractmethod
    def array(self, data, dtype=None) -> Any:
        """Create an array on the device."""
        pass
    
    @abstractmethod
    def zeros(self, shape, dtype=None) -> Any:
        """Create a zero-filled array."""
        pass
    
    @abstractmethod
    def ones(self, shape, dtype=None) -> Any:
        """Create a one-filled array."""
        pass
    
    @abstractmethod
    def empty(self, shape, dtype=None) -> Any:
        """Create an uninitialized array."""
        pass
    
    @abstractmethod
    def random_uniform(self, low: float, high: float, size: Tuple[int, ...]) -> Any:
        """Create array with uniform random values."""
        pass
    
    @abstractmethod
    def to_cpu(self, arr) -> np.ndarray:
        """Transfer array to CPU memory."""
        pass
    
    @abstractmethod
    def synchronize(self):
        """Wait for all pending operations to complete."""
        pass
    
    # Neural network primitives
    
    @abstractmethod
    def neuron_update(
        self,
        potentials: Any,
        thresholds: Any,
        leak_rate: float,
        fired: Any,
        last_fire_time: Any,
        time: float
    ) -> None:
        """
        Update neuron states in-place.
        
        1. Apply leak: potential *= leak_rate
        2. Check firing: fired = (potential >= threshold) for positive thresholds
        3. Reset fired neurons: potential[fired] = 0
        4. Update last_fire_time for fired neurons
        """
        pass
    
    @abstractmethod
    def synapse_propagate(
        self,
        pre_indices: Any,
        post_indices: Any,
        weights: Any,
        pre_fired: Any,
        post_input: Any
    ) -> None:
        """
        Propagate spikes through synapses.
        
        For each synapse where pre_neuron fired, add weight to post_neuron input.
        Uses atomic operations for thread safety.
        """
        pass
    
    @abstractmethod
    def synapse_stdp(
        self,
        pre_indices: Any,
        post_indices: Any,
        weights: Any,
        pre_fire_time: Any,
        post_fire_time: Any,
        plasticity_rate: float,
        strengthen_rate: float,
        weaken_rate: float,
        stdp_window: float,
        weight_clamp: Tuple[float, float]
    ) -> None:
        """
        Apply STDP plasticity to synapse weights.
        """
        pass
    
    @abstractmethod
    def synapse_decay(
        self,
        weights: Any,
        decay_factor: float,
        weight_clamp: Tuple[float, float]
    ) -> None:
        """
        Apply continuous weakening to synapse weights.
        """
        pass


class NumPyBackend(Backend):
    """
    NumPy-based CPU backend with vectorized operations.
    
    This is always available and serves as the reference implementation.
    """
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.NUMPY
    
    def array(self, data, dtype=None) -> np.ndarray:
        dtype = dtype or self.dtype
        return np.array(data, dtype=dtype)
    
    def zeros(self, shape, dtype=None) -> np.ndarray:
        dtype = dtype or self.dtype
        return np.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=None) -> np.ndarray:
        dtype = dtype or self.dtype
        return np.ones(shape, dtype=dtype)
    
    def empty(self, shape, dtype=None) -> np.ndarray:
        dtype = dtype or self.dtype
        return np.empty(shape, dtype=dtype)
    
    def random_uniform(self, low: float, high: float, size: Tuple[int, ...]) -> np.ndarray:
        return np.random.uniform(low, high, size).astype(self.dtype)
    
    def to_cpu(self, arr) -> np.ndarray:
        return np.asarray(arr)
    
    def synchronize(self):
        pass  # No-op for CPU
    
    def neuron_update(
        self,
        potentials: np.ndarray,
        thresholds: np.ndarray,
        leak_rate: float,
        fired: np.ndarray,
        last_fire_time: np.ndarray,
        time: float
    ) -> None:
        # Apply leak
        potentials *= leak_rate
        
        # Bidirectional threshold check
        pos_thresh = thresholds > 0
        neg_thresh = thresholds < 0
        
        # Positive threshold: fire on excitation
        fired_pos = pos_thresh & (potentials >= thresholds)
        # Negative threshold: fire on inhibition
        fired_neg = neg_thresh & (potentials <= thresholds)
        
        # Combine firing
        np.logical_or(fired_pos, fired_neg, out=fired)
        
        # Clamp potentials based on threshold sign
        potentials[pos_thresh & (potentials < 0)] = 0.0
        potentials[neg_thresh & (potentials > 0)] = 0.0
        
        # Reset fired neurons
        potentials[fired] = 0.0
        last_fire_time[fired] = time
    
    def synapse_propagate(
        self,
        pre_indices: np.ndarray,
        post_indices: np.ndarray,
        weights: np.ndarray,
        pre_fired: np.ndarray,
        post_input: np.ndarray
    ) -> None:
        # Find synapses where pre-neuron fired
        active_mask = pre_fired[pre_indices]
        
        # Add weights to post-neuron inputs (use bincount for efficiency)
        if np.any(active_mask):
            active_posts = post_indices[active_mask]
            active_weights = weights[active_mask]
            np.add.at(post_input, active_posts, active_weights)
    
    def synapse_stdp(
        self,
        pre_indices: np.ndarray,
        post_indices: np.ndarray,
        weights: np.ndarray,
        pre_fire_time: np.ndarray,
        post_fire_time: np.ndarray,
        plasticity_rate: float,
        strengthen_rate: float,
        weaken_rate: float,
        stdp_window: float,
        weight_clamp: Tuple[float, float]
    ) -> None:
        # Get fire times for pre and post neurons of each synapse
        pre_times = pre_fire_time[pre_indices]
        post_times = post_fire_time[post_indices]
        
        # Calculate time difference
        delta_t = post_times - pre_times
        
        # Only apply STDP within window
        in_window = np.abs(delta_t) < stdp_window
        
        # Strengthen where pre fired before post (causal)
        strengthen_mask = in_window & (delta_t > 0)
        if np.any(strengthen_mask):
            change = strengthen_rate * np.exp(-np.abs(delta_t[strengthen_mask]) / stdp_window)
            weights[strengthen_mask] += change * plasticity_rate
        
        # Weaken where post fired before pre (anti-causal)
        weaken_mask = in_window & (delta_t < 0)
        if np.any(weaken_mask):
            change = weaken_rate * np.exp(-np.abs(delta_t[weaken_mask]) / stdp_window)
            weights[weaken_mask] -= change * plasticity_rate
        
        # Clamp weights
        np.clip(weights, weight_clamp[0], weight_clamp[1], out=weights)
    
    def synapse_decay(
        self,
        weights: np.ndarray,
        decay_factor: float,
        weight_clamp: Tuple[float, float]
    ) -> None:
        weights *= decay_factor
        np.clip(weights, weight_clamp[0], weight_clamp[1], out=weights)


class CuPyBackend(Backend):
    """
    CuPy-based CUDA backend for NVIDIA GPUs.
    
    Provides massive parallelism for large neural networks.
    """
    
    def __init__(self, config: GPUConfig):
        super().__init__(config)
        import cupy as cp
        self.cp = cp
        
        # Set device
        cp.cuda.Device(config.device_id).use()
        
        # Compile custom kernels for neural operations
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile CUDA kernels for neural network operations."""
        self._propagate_kernel = self.cp.RawKernel(r'''
        extern "C" __global__
        void synapse_propagate(
            const int* pre_indices,
            const int* post_indices,
            const float* weights,
            const bool* pre_fired,
            float* post_input,
            int n_synapses
        ) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx < n_synapses) {
                int pre_idx = pre_indices[idx];
                if (pre_fired[pre_idx]) {
                    int post_idx = post_indices[idx];
                    atomicAdd(&post_input[post_idx], weights[idx]);
                }
            }
        }
        ''', 'synapse_propagate')
        
        self._neuron_update_kernel = self.cp.RawKernel(r'''
        extern "C" __global__
        void neuron_update(
            float* potentials,
            const float* thresholds,
            float leak_rate,
            bool* fired,
            float* last_fire_time,
            float time,
            int n_neurons
        ) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx < n_neurons) {
                // Apply leak
                potentials[idx] *= leak_rate;
                
                float pot = potentials[idx];
                float thresh = thresholds[idx];
                bool did_fire = false;
                
                if (thresh > 0) {
                    // Positive threshold: fire on excitation
                    did_fire = pot >= thresh;
                    if (pot < 0) potentials[idx] = 0.0f;
                } else if (thresh < 0) {
                    // Negative threshold: fire on inhibition
                    did_fire = pot <= thresh;
                    if (pot > 0) potentials[idx] = 0.0f;
                }
                
                fired[idx] = did_fire;
                if (did_fire) {
                    potentials[idx] = 0.0f;
                    last_fire_time[idx] = time;
                }
            }
        }
        ''', 'neuron_update')
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.CUDA
    
    def array(self, data, dtype=None):
        dtype = dtype or self.dtype
        return self.cp.array(data, dtype=dtype)
    
    def zeros(self, shape, dtype=None):
        dtype = dtype or self.dtype
        return self.cp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=None):
        dtype = dtype or self.dtype
        return self.cp.ones(shape, dtype=dtype)
    
    def empty(self, shape, dtype=None):
        dtype = dtype or self.dtype
        return self.cp.empty(shape, dtype=dtype)
    
    def random_uniform(self, low: float, high: float, size: Tuple[int, ...]):
        return self.cp.random.uniform(low, high, size).astype(self.dtype)
    
    def to_cpu(self, arr) -> np.ndarray:
        return self.cp.asnumpy(arr)
    
    def synchronize(self):
        self.cp.cuda.Stream.null.synchronize()
    
    def neuron_update(
        self,
        potentials,
        thresholds,
        leak_rate: float,
        fired,
        last_fire_time,
        time: float
    ) -> None:
        n_neurons = len(potentials)
        block_size = self.config.block_size
        grid_size = (n_neurons + block_size - 1) // block_size
        
        self._neuron_update_kernel(
            (grid_size,), (block_size,),
            (potentials, thresholds, np.float32(leak_rate), 
             fired, last_fire_time, np.float32(time), np.int32(n_neurons))
        )
        
        if self.config.sync_after_kernel:
            self.synchronize()
    
    def synapse_propagate(
        self,
        pre_indices,
        post_indices,
        weights,
        pre_fired,
        post_input
    ) -> None:
        n_synapses = len(pre_indices)
        if n_synapses == 0:
            return
            
        block_size = self.config.block_size
        grid_size = (n_synapses + block_size - 1) // block_size
        
        self._propagate_kernel(
            (grid_size,), (block_size,),
            (pre_indices, post_indices, weights, pre_fired, post_input, np.int32(n_synapses))
        )
        
        if self.config.sync_after_kernel:
            self.synchronize()
    
    def synapse_stdp(
        self,
        pre_indices,
        post_indices,
        weights,
        pre_fire_time,
        post_fire_time,
        plasticity_rate: float,
        strengthen_rate: float,
        weaken_rate: float,
        stdp_window: float,
        weight_clamp: Tuple[float, float]
    ) -> None:
        # Use vectorized CuPy operations (fast enough for most cases)
        pre_times = pre_fire_time[pre_indices]
        post_times = post_fire_time[post_indices]
        
        delta_t = post_times - pre_times
        in_window = self.cp.abs(delta_t) < stdp_window
        
        # Strengthen
        strengthen_mask = in_window & (delta_t > 0)
        change = strengthen_rate * self.cp.exp(-self.cp.abs(delta_t) / stdp_window)
        weights += self.cp.where(strengthen_mask, change * plasticity_rate, 0)
        
        # Weaken
        weaken_mask = in_window & (delta_t < 0)
        change = weaken_rate * self.cp.exp(-self.cp.abs(delta_t) / stdp_window)
        weights -= self.cp.where(weaken_mask, change * plasticity_rate, 0)
        
        # Clamp
        self.cp.clip(weights, weight_clamp[0], weight_clamp[1], out=weights)
    
    def synapse_decay(
        self,
        weights,
        decay_factor: float,
        weight_clamp: Tuple[float, float]
    ) -> None:
        weights *= decay_factor
        self.cp.clip(weights, weight_clamp[0], weight_clamp[1], out=weights)


class OpenCLBackend(Backend):
    """
    PyOpenCL-based backend for cross-platform GPU support.
    
    Works with AMD, Intel, and NVIDIA GPUs.
    """
    
    def __init__(self, config: GPUConfig):
        super().__init__(config)
        import pyopencl as cl
        import pyopencl.array as cl_array
        self.cl = cl
        self.cl_array = cl_array
        
        # Initialize OpenCL context
        platforms = cl.get_platforms()
        gpu_devices = []
        for platform in platforms:
            gpu_devices.extend(platform.get_devices(device_type=cl.device_type.GPU))
        
        if config.device_id >= len(gpu_devices):
            raise ValueError(f"OpenCL device {config.device_id} not found")
        
        self.device = gpu_devices[config.device_id]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        
        # Compile kernels
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile OpenCL kernels."""
        kernel_src = '''
        __kernel void neuron_update(
            __global float* potentials,
            __global const float* thresholds,
            float leak_rate,
            __global int* fired,
            __global float* last_fire_time,
            float time,
            int n_neurons
        ) {
            int idx = get_global_id(0);
            if (idx < n_neurons) {
                potentials[idx] *= leak_rate;
                
                float pot = potentials[idx];
                float thresh = thresholds[idx];
                int did_fire = 0;
                
                if (thresh > 0) {
                    did_fire = pot >= thresh ? 1 : 0;
                    if (pot < 0) potentials[idx] = 0.0f;
                } else if (thresh < 0) {
                    did_fire = pot <= thresh ? 1 : 0;
                    if (pot > 0) potentials[idx] = 0.0f;
                }
                
                fired[idx] = did_fire;
                if (did_fire) {
                    potentials[idx] = 0.0f;
                    last_fire_time[idx] = time;
                }
            }
        }
        
        __kernel void synapse_propagate(
            __global const int* pre_indices,
            __global const int* post_indices,
            __global const float* weights,
            __global const int* pre_fired,
            __global float* post_input,
            int n_synapses
        ) {
            int idx = get_global_id(0);
            if (idx < n_synapses) {
                int pre_idx = pre_indices[idx];
                if (pre_fired[pre_idx]) {
                    int post_idx = post_indices[idx];
                    // Note: atomic_add for float may need extensions
                    post_input[post_idx] += weights[idx];
                }
            }
        }
        '''
        self.program = self.cl.Program(self.ctx, kernel_src).build()
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.OPENCL
    
    def array(self, data, dtype=None):
        dtype = dtype or self.dtype
        return self.cl_array.to_device(self.queue, np.array(data, dtype=dtype))
    
    def zeros(self, shape, dtype=None):
        dtype = dtype or self.dtype
        return self.cl_array.zeros(self.queue, shape, dtype=dtype)
    
    def ones(self, shape, dtype=None):
        dtype = dtype or self.dtype
        arr = self.cl_array.zeros(self.queue, shape, dtype=dtype)
        arr.fill(1.0)
        return arr
    
    def empty(self, shape, dtype=None):
        dtype = dtype or self.dtype
        return self.cl_array.empty(self.queue, shape, dtype=dtype)
    
    def random_uniform(self, low: float, high: float, size: Tuple[int, ...]):
        data = np.random.uniform(low, high, size).astype(self.dtype)
        return self.cl_array.to_device(self.queue, data)
    
    def to_cpu(self, arr) -> np.ndarray:
        return arr.get()
    
    def synchronize(self):
        self.queue.finish()
    
    def neuron_update(
        self,
        potentials,
        thresholds,
        leak_rate: float,
        fired,
        last_fire_time,
        time: float
    ) -> None:
        n_neurons = len(potentials)
        global_size = ((n_neurons + self.config.block_size - 1) // self.config.block_size) * self.config.block_size
        
        self.program.neuron_update(
            self.queue, (global_size,), (self.config.block_size,),
            potentials.data, thresholds.data,
            np.float32(leak_rate),
            fired.data, last_fire_time.data,
            np.float32(time), np.int32(n_neurons)
        )
        
        if self.config.sync_after_kernel:
            self.synchronize()
    
    def synapse_propagate(
        self,
        pre_indices,
        post_indices,
        weights,
        pre_fired,
        post_input
    ) -> None:
        n_synapses = len(pre_indices)
        if n_synapses == 0:
            return
            
        global_size = ((n_synapses + self.config.block_size - 1) // self.config.block_size) * self.config.block_size
        
        self.program.synapse_propagate(
            self.queue, (global_size,), (self.config.block_size,),
            pre_indices.data, post_indices.data, weights.data,
            pre_fired.data, post_input.data, np.int32(n_synapses)
        )
        
        if self.config.sync_after_kernel:
            self.synchronize()
    
    def synapse_stdp(
        self,
        pre_indices,
        post_indices,
        weights,
        pre_fire_time,
        post_fire_time,
        plasticity_rate: float,
        strengthen_rate: float,
        weaken_rate: float,
        stdp_window: float,
        weight_clamp: Tuple[float, float]
    ) -> None:
        # Fall back to CPU for STDP (complex operation)
        weights_cpu = self.to_cpu(weights)
        pre_times_cpu = self.to_cpu(pre_fire_time)[self.to_cpu(pre_indices)]
        post_times_cpu = self.to_cpu(post_fire_time)[self.to_cpu(post_indices)]
        
        delta_t = post_times_cpu - pre_times_cpu
        in_window = np.abs(delta_t) < stdp_window
        
        strengthen_mask = in_window & (delta_t > 0)
        if np.any(strengthen_mask):
            change = strengthen_rate * np.exp(-np.abs(delta_t[strengthen_mask]) / stdp_window)
            weights_cpu[strengthen_mask] += change * plasticity_rate
        
        weaken_mask = in_window & (delta_t < 0)
        if np.any(weaken_mask):
            change = weaken_rate * np.exp(-np.abs(delta_t[weaken_mask]) / stdp_window)
            weights_cpu[weaken_mask] -= change * plasticity_rate
        
        np.clip(weights_cpu, weight_clamp[0], weight_clamp[1], out=weights_cpu)
        weights.set(weights_cpu)
    
    def synapse_decay(
        self,
        weights,
        decay_factor: float,
        weight_clamp: Tuple[float, float]
    ) -> None:
        weights *= decay_factor
        # Clamp on CPU
        weights_cpu = self.to_cpu(weights)
        np.clip(weights_cpu, weight_clamp[0], weight_clamp[1], out=weights_cpu)
        weights.set(weights_cpu)


def get_backend(config: GPUConfig) -> Backend:
    """
    Get the appropriate backend based on configuration.
    
    Args:
        config: GPU configuration
        
    Returns:
        Backend instance
    """
    backend_type = config.backend.lower()
    
    if backend_type == "auto":
        from .config import get_best_backend
        best = get_best_backend()
        backend_type = best.backend.value
        logger.info(f"Auto-selected backend: {best}")
    
    if backend_type == "cuda":
        try:
            return CuPyBackend(config)
        except ImportError:
            logger.warning("CuPy not available, falling back to NumPy")
            return NumPyBackend(config)
    elif backend_type == "opencl":
        try:
            return OpenCLBackend(config)
        except ImportError:
            logger.warning("PyOpenCL not available, falling back to NumPy")
            return NumPyBackend(config)
    else:
        return NumPyBackend(config)
