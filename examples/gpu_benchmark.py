#!/usr/bin/env python3
"""
GPU Neural Network Benchmark

Phase 10b: Demonstrates GPU acceleration for large neural networks.

This benchmark compares performance between:
- NumPy (CPU, vectorized)
- CuPy (CUDA, if available)
- PyOpenCL (if available)

Run with:
    PYTHONPATH=/Users/gspilz/code/CritterGOD python3 examples/gpu_benchmark.py
"""

import time
import argparse
import numpy as np
from typing import Dict, List, Tuple

# Import GPU neural network module
from core.neural.gpu import (
    GPUNeuralNetwork,
    GPUConfig,
    get_available_backends,
    get_best_backend,
)


def benchmark_network(
    n_neurons: int,
    n_steps: int,
    backend: str,
    device_id: int = 0,
) -> Dict:
    """
    Benchmark a neural network configuration.
    
    Returns timing and statistics.
    """
    config = GPUConfig(
        backend=backend,
        device_id=device_id,
        batch_size=1024,
        block_size=256,
    )
    
    print(f"\n{'='*60}")
    print(f"Benchmark: {n_neurons} neurons, {backend} backend")
    print(f"{'='*60}")
    
    # Create network
    print("Creating network...")
    t0 = time.perf_counter()
    network = GPUNeuralNetwork(
        n_neurons=n_neurons,
        n_synapses_per_neuron=40,
        config=config,
        enable_plasticity=True,
        enable_rewiring=False,  # Disable for consistent timing
    )
    creation_time = time.perf_counter() - t0
    print(f"  Created in {creation_time:.3f}s")
    print(f"  Neurons: {network.n_neurons:,}")
    print(f"  Synapses: {network.n_synapses:,}")
    print(f"  Backend: {network.backend.backend_type.value}")
    
    # Warm up
    print("Warming up...")
    for _ in range(10):
        network.update()
    network.synchronize()
    
    # Benchmark
    print(f"Running {n_steps} steps...")
    t0 = time.perf_counter()
    for _ in range(n_steps):
        network.update()
    network.synchronize()
    total_time = time.perf_counter() - t0
    
    # Calculate metrics
    steps_per_second = n_steps / total_time
    ms_per_step = (total_time / n_steps) * 1000
    neurons_per_second = n_neurons * steps_per_second
    synapses_per_second = network.n_synapses * steps_per_second
    
    stats = network.get_stats()
    
    result = {
        "backend": backend,
        "n_neurons": n_neurons,
        "n_synapses": network.n_synapses,
        "n_steps": n_steps,
        "total_time": total_time,
        "creation_time": creation_time,
        "steps_per_second": steps_per_second,
        "ms_per_step": ms_per_step,
        "neurons_per_second": neurons_per_second,
        "synapses_per_second": synapses_per_second,
        "activity": stats["activity"],
    }
    
    print(f"\nResults:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Steps/second: {steps_per_second:.1f}")
    print(f"  ms/step: {ms_per_step:.3f}")
    print(f"  Neurons/second: {neurons_per_second:,.0f}")
    print(f"  Synapses/second: {synapses_per_second:,.0f}")
    print(f"  Final activity: {stats['activity']*100:.1f}%")
    
    return result


def run_scaling_test(backend: str, max_neurons: int = 50000) -> List[Dict]:
    """Test how performance scales with network size."""
    results = []
    
    sizes = [100, 500, 1000, 2000, 5000, 10000]
    if max_neurons >= 20000:
        sizes.append(20000)
    if max_neurons >= 50000:
        sizes.append(50000)
    
    for n in sizes:
        if n > max_neurons:
            break
        try:
            result = benchmark_network(n, n_steps=100, backend=backend)
            results.append(result)
        except Exception as e:
            print(f"  Failed at {n} neurons: {e}")
            break
    
    return results


def print_comparison_table(results: List[Dict]):
    """Print a comparison table of results."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Backend':<10} {'Neurons':>10} {'Synapses':>12} {'Steps/s':>10} {'ms/step':>10} {'M neurons/s':>12}")
    print("-"*80)
    
    for r in results:
        print(
            f"{r['backend']:<10} "
            f"{r['n_neurons']:>10,} "
            f"{r['n_synapses']:>12,} "
            f"{r['steps_per_second']:>10.1f} "
            f"{r['ms_per_step']:>10.3f} "
            f"{r['neurons_per_second']/1e6:>12.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="GPU Neural Network Benchmark")
    parser.add_argument("--neurons", type=int, default=10000, help="Number of neurons")
    parser.add_argument("--steps", type=int, default=100, help="Number of timesteps")
    parser.add_argument("--backend", type=str, default="auto", help="Backend: auto, numpy, cuda, opencl")
    parser.add_argument("--scale", action="store_true", help="Run scaling test")
    parser.add_argument("--compare", action="store_true", help="Compare all backends")
    args = parser.parse_args()
    
    print("="*60)
    print("CritterGOD GPU Neural Network Benchmark")
    print("Phase 10b: High-Performance Computing")
    print("="*60)
    
    # Show available backends
    print("\nAvailable backends:")
    backends = get_available_backends()
    for device in backends:
        print(f"  {device}")
    
    best = get_best_backend()
    print(f"\nBest available: {best}")
    
    if args.compare:
        # Compare all available backends
        all_results = []
        
        # Always test NumPy
        results = run_scaling_test("numpy", max_neurons=args.neurons)
        all_results.extend(results)
        
        # Test CUDA if available
        cuda_devices = [d for d in backends if d.backend.value == "cuda"]
        if cuda_devices:
            results = run_scaling_test("cuda", max_neurons=args.neurons)
            all_results.extend(results)
        
        # Test OpenCL if available
        opencl_devices = [d for d in backends if d.backend.value == "opencl"]
        if opencl_devices:
            results = run_scaling_test("opencl", max_neurons=args.neurons)
            all_results.extend(results)
        
        print_comparison_table(all_results)
        
    elif args.scale:
        # Run scaling test for selected backend
        results = run_scaling_test(args.backend, max_neurons=args.neurons)
        print_comparison_table(results)
        
    else:
        # Single benchmark
        benchmark_network(args.neurons, args.steps, args.backend)
    
    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)


if __name__ == "__main__":
    main()
