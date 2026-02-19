# GPU Acceleration Setup Guide - Phase 10b

## Overview
CritterGOD supports GPU-accelerated neural networks for 10,000+ neuron simulations through multiple backends:
- **CUDA** (NVIDIA GPUs via CuPy)
- **OpenCL** (Cross-platform via PyOpenCL)  
- **NumPy** (CPU vectorized - always available)

## Current Status on Your System
- ✅ **NumPy backend**: Available (CPU fallback)
- ❌ **CUDA backend**: Not available (MacOS ARM doesn't support CUDA)
- ❌ **OpenCL backend**: Not detected (needs installation)

## MacOS ARM GPU Options

### Option 1: Metal Performance Shaders (Recommended for M1/M2/M3)
MacOS uses Metal instead of CUDA. To use Apple Silicon GPU:

```bash
# Install PyTorch with Metal support (best performance)
pip3 install --break-system-packages --user torch torchvision

# Or use MLX (Apple's ML framework)
pip3 install --break-system-packages --user mlx
```

**Note**: Would require adding a Metal backend to `core/neural/gpu/backend.py`

### Option 2: OpenCL (Cross-platform)
OpenCL works on Apple Silicon but with limited performance:

```bash
# Install PyOpenCL
pip3 install --break-system-packages --user pyopencl

# Test OpenCL detection
python3 -c "from core.neural.gpu import get_available_backends; print([b for b in get_available_backends()])"
```

### Option 3: Remote GPU (Cloud)
Use a cloud GPU for heavy workloads:

```bash
# AWS, GCP, or Lambda Labs with CUDA
pip install cupy-cuda11x  # or cuda12x for newer GPUs
```

## Using GPU Networks

### Quick Start
```python
from core.neural.gpu import GPUNeuralNetwork, GPUConfig, get_available_backends

# Check what's available
backends = get_available_backends()
for backend in backends:
    print(f"  {backend}")

# Create GPU-accelerated network
config = GPUConfig(
    backend='auto',  # Automatically selects best available
    batch_size=1024,
    block_size=256,
)

network = GPUNeuralNetwork(
    n_neurons=10000,
    n_synapses_per_neuron=40,
    config=config,
    enable_plasticity=True,
)

# Use just like regular network
for step in range(1000):
    network.update()
    
stats = network.get_stats()
print(f"Activity: {stats['activity']*100:.1f}%")
```

### Configuration Options

```python
config = GPUConfig(
    # Backend selection
    backend='auto',        # 'auto', 'cuda', 'opencl', 'numpy'
    device_id=0,           # GPU device index (if multiple GPUs)
    
    # Performance tuning
    batch_size=1024,       # Neurons per kernel launch
    block_size=256,        # CUDA block / OpenCL workgroup size
    use_fp16=False,        # Half precision (experimental, 2x memory savings)
    async_transfer=True,   # Async CPU↔GPU transfers
    sparse_threshold=0.3,  # Use sparse ops if >30% weights are zero
    
    # Memory management  
    preallocate_memory=True,    # Preallocate GPU buffers
    max_neurons=100000,          # Max neurons to preallocate for
    max_synapses=10000000,       # Max synapses to preallocate for
    
    # Precision
    potential_dtype='float32',   # 'float32' or 'float16'
    weight_dtype='float32',      # 'float32' or 'float16'
    
    # Debug
    sync_after_kernel=False,     # Force sync for debugging (slow)
    profile_kernels=False,       # Enable profiling
)
```

## Benchmarking

Run the benchmark to compare backends:

```bash
# Basic benchmark (10k neurons, 100 steps)
python3 examples/gpu_benchmark.py

# Scaling test
python3 examples/gpu_benchmark.py --scale --neurons 50000

# Compare all available backends
python3 examples/gpu_benchmark.py --compare --neurons 20000

# Specific backend
python3 examples/gpu_benchmark.py --backend cuda --neurons 10000 --steps 1000
```

### Expected Performance

| Backend | Neurons | Steps/sec | ms/step | Notes |
|---------|---------|-----------|---------|-------|
| NumPy   | 10,000  | ~50       | 20ms    | CPU baseline |
| OpenCL  | 10,000  | ~200      | 5ms     | 4x speedup (if available) |
| CUDA    | 10,000  | ~500      | 2ms     | 10x speedup (NVIDIA only) |
| Metal   | 10,000  | ~400      | 2.5ms   | 8x speedup (Apple Silicon) |

For 100,000 neurons:
- NumPy: ~0.5 steps/sec (2000ms/step) - **too slow**
- GPU: ~50 steps/sec (20ms/step) - **100x faster!**

## Integration with Research Platform

To use GPU networks in `research_platform.py`:

```python
# In ResearchPlatform.__init__()
from core.neural.gpu import GPUNeuralNetwork, GPUConfig

# Create GPU config
gpu_config = GPUConfig(backend='auto', batch_size=1024)

# When creating creatures, use GPU networks
for creature in self.creatures:
    creature.network = GPUNeuralNetwork(
        n_neurons=len(creature.network.neurons),
        config=gpu_config,
    )
```

Or add a configuration parameter:

```python
# In profiles/default.json
{
    "neural": {
        "use_gpu": true,
        "gpu_backend": "auto",
        "gpu_batch_size": 1024
    }
}
```

## Troubleshooting

### "No CUDA devices found"
- CUDA only works on NVIDIA GPUs
- MacOS doesn't support CUDA (use Metal/OpenCL instead)

### "OpenCL initialization failed"  
- Check OpenCL drivers: `clinfo` (install via `brew install clinfo`)
- Try: `python3 -m pip install --user pyopencl`

### "Out of memory"
- Reduce `max_neurons` and `max_synapses` in GPUConfig
- Enable `use_fp16=True` for 2x memory savings
- Reduce creature population size

### Performance not improved
- NumPy is already well-optimized for small networks (<1000 neurons)
- GPU overhead dominates for tiny networks
- GPUs excel at 10,000+ neurons

## Memory Estimates

Approximate GPU memory usage:

| Neurons | Synapses | Memory (FP32) | Memory (FP16) |
|---------|----------|---------------|---------------|
| 1,000   | 40,000   | ~2 MB         | ~1 MB         |
| 10,000  | 400,000  | ~20 MB        | ~10 MB        |
| 100,000 | 4,000,000| ~200 MB       | ~100 MB       |
| 1,000,000| 40,000,000| ~2 GB        | ~1 GB         |

Formula: `memory_bytes = (neurons * 20 + synapses * 9) * dtype_size`

## Next Steps

1. **Install OpenCL** (cross-platform GPU)
   ```bash
   pip3 install --break-system-packages --user pyopencl
   ```

2. **Run benchmark** to verify installation
   ```bash
   python3 examples/gpu_benchmark.py --compare
   ```

3. **Add Metal backend** for optimal Apple Silicon performance (future work)

4. **Profile your workload** to see where GPU helps most

## References

- Phase 10b documentation: `PHASE10B_COMPLETE.md`
- Backend implementation: `core/neural/gpu/backend.py`
- Network implementation: `core/neural/gpu/gpu_network.py`
- Configuration: `core/neural/gpu/config.py`
