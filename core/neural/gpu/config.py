"""
GPU Configuration and Device Selection

Handles backend selection, device enumeration, and performance tuning parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available compute backends."""
    NUMPY = "numpy"      # CPU vectorized (always available)
    CUDA = "cuda"        # NVIDIA GPU via CuPy
    OPENCL = "opencl"    # Cross-platform GPU via PyOpenCL


@dataclass
class DeviceInfo:
    """Information about a compute device."""
    backend: BackendType
    device_id: int
    name: str
    memory_bytes: int
    compute_units: int
    is_available: bool
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        mem_gb = self.memory_bytes / (1024**3)
        return f"{self.backend.value}:{self.device_id} - {self.name} ({mem_gb:.1f}GB, {self.compute_units} CUs)"


@dataclass
class GPUConfig:
    """
    Configuration for GPU-accelerated neural network.
    
    Performance tuning parameters:
    - batch_size: Number of neurons processed per kernel launch
    - block_size: CUDA/OpenCL work group size (usually 256 or 512)
    - use_fp16: Use half precision for memory bandwidth (experimental)
    - async_transfer: Overlap CPU/GPU data transfer with computation
    - sparse_threshold: Switch to sparse representation above this sparsity
    """
    # Backend selection
    backend: str = "auto"  # 'auto', 'cuda', 'opencl', 'numpy'
    device_id: int = 0
    
    # Performance tuning
    batch_size: int = 1024         # Neurons per batch
    block_size: int = 256          # CUDA block size / OpenCL work group
    use_fp16: bool = False         # Half precision (experimental)
    async_transfer: bool = True    # Async memory transfers
    sparse_threshold: float = 0.3  # Use sparse ops if >30% zeros
    
    # Memory management
    preallocate_memory: bool = True     # Preallocate GPU buffers
    max_neurons: int = 100000           # Max neurons to preallocate for
    max_synapses: int = 10000000        # Max synapses to preallocate for
    
    # Precision settings
    potential_dtype: str = "float32"    # Neuron potential precision
    weight_dtype: str = "float32"       # Synapse weight precision
    
    # Debug options
    sync_after_kernel: bool = False     # Force sync for debugging
    profile_kernels: bool = False       # Enable kernel profiling
    
    def validate(self):
        """Validate configuration values."""
        if self.batch_size < 32:
            logger.warning("batch_size < 32 may reduce GPU efficiency")
        if self.block_size not in [64, 128, 256, 512, 1024]:
            logger.warning(f"block_size={self.block_size} is unusual, consider 256 or 512")


def get_available_backends() -> List[DeviceInfo]:
    """
    Enumerate all available compute backends and devices.
    
    Returns:
        List of DeviceInfo for all available devices
    """
    devices = []
    
    # NumPy is always available
    import numpy as np
    devices.append(DeviceInfo(
        backend=BackendType.NUMPY,
        device_id=0,
        name="NumPy CPU (vectorized)",
        memory_bytes=0,  # System RAM
        compute_units=1,
        is_available=True,
        extra={"numpy_version": np.__version__}
    ))
    
    # Check for CuPy (CUDA)
    try:
        import cupy as cp
        for i in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(i):
                props = cp.cuda.runtime.getDeviceProperties(i)
                devices.append(DeviceInfo(
                    backend=BackendType.CUDA,
                    device_id=i,
                    name=props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
                    memory_bytes=props["totalGlobalMem"],
                    compute_units=props["multiProcessorCount"],
                    is_available=True,
                    extra={
                        "cuda_version": cp.cuda.runtime.runtimeGetVersion(),
                        "compute_capability": (props["major"], props["minor"]),
                    }
                ))
    except ImportError:
        logger.debug("CuPy not installed - CUDA backend unavailable")
    except Exception as e:
        logger.debug(f"CUDA initialization failed: {e}")
    
    # Check for PyOpenCL
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        device_idx = 0
        for platform in platforms:
            for device in platform.get_devices(device_type=cl.device_type.GPU):
                devices.append(DeviceInfo(
                    backend=BackendType.OPENCL,
                    device_id=device_idx,
                    name=device.name,
                    memory_bytes=device.global_mem_size,
                    compute_units=device.max_compute_units,
                    is_available=True,
                    extra={
                        "platform": platform.name,
                        "opencl_version": device.opencl_c_version,
                    }
                ))
                device_idx += 1
    except ImportError:
        logger.debug("PyOpenCL not installed - OpenCL backend unavailable")
    except Exception as e:
        logger.debug(f"OpenCL initialization failed: {e}")
    
    return devices


def get_best_backend() -> DeviceInfo:
    """
    Select the best available backend automatically.
    
    Priority: CUDA > OpenCL > NumPy
    
    Returns:
        DeviceInfo for the best available device
    """
    devices = get_available_backends()
    
    # Prefer CUDA
    cuda_devices = [d for d in devices if d.backend == BackendType.CUDA]
    if cuda_devices:
        # Return device with most memory
        return max(cuda_devices, key=lambda d: d.memory_bytes)
    
    # Then OpenCL
    opencl_devices = [d for d in devices if d.backend == BackendType.OPENCL]
    if opencl_devices:
        return max(opencl_devices, key=lambda d: d.memory_bytes)
    
    # Fall back to NumPy
    return devices[0]


def estimate_memory_usage(n_neurons: int, n_synapses: int, config: GPUConfig) -> int:
    """
    Estimate GPU memory usage in bytes.
    
    Args:
        n_neurons: Number of neurons
        n_synapses: Number of synapses
        config: GPU configuration
        
    Returns:
        Estimated memory usage in bytes
    """
    dtype_size = 2 if config.use_fp16 else 4
    
    # Neuron arrays: potential, threshold, type, fired_state, last_fire_time
    neuron_bytes = n_neurons * dtype_size * 5
    
    # Synapse arrays: pre_idx, post_idx, weight, is_inhibitory
    synapse_bytes = n_synapses * (4 + 4 + dtype_size + 1)
    
    # Work buffers (input accumulation, temp arrays)
    work_bytes = n_neurons * dtype_size * 4
    
    # Overhead
    overhead = 1024 * 1024 * 10  # 10MB overhead
    
    return neuron_bytes + synapse_bytes + work_bytes + overhead
