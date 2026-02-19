#!/usr/bin/env python3
"""
GPU-Accelerated Research Platform
Phase 10b: Demonstrates GPU neural networks in the research platform

This extends research_platform.py with GPU acceleration for large-scale simulations.

Controls: Same as research_platform.py
Additional:
- G: Toggle GPU mode on/off
- B: Show GPU backend info

Run with:
    PYTHONPATH=/Users/gspilz/code/CritterGOD python3 examples/gpu_research_platform.py
    
    # Or with specific backend
    PYTHONPATH=/Users/gspilz/code/CritterGOD python3 examples/gpu_research_platform.py --gpu-backend cuda
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# Import base research platform
from examples.research_platform import ResearchPlatform

# Import GPU modules
from core.neural.gpu import (
    GPUNeuralNetwork,
    GPUConfig,
    get_available_backends,
    get_best_backend,
)


class GPUResearchPlatform(ResearchPlatform):
    """Research platform with GPU-accelerated neural networks."""
    
    def __init__(self, width=1600, height=900, gpu_backend='auto', gpu_enabled=True):
        """
        Initialize GPU research platform.
        
        Args:
            width, height: Window dimensions
            gpu_backend: 'auto', 'cuda', 'opencl', 'numpy'
            gpu_enabled: Start with GPU enabled
        """
        # Store GPU settings before parent init
        self.gpu_enabled = gpu_enabled
        self.gpu_backend_name = gpu_backend
        
        # Create GPU config
        self.gpu_config = GPUConfig(
            backend=gpu_backend,
            device_id=0,
            batch_size=1024,
            block_size=256,
            preallocate_memory=True,
        )
        
        # Detect available backends
        self.available_backends = get_available_backends()
        best_backend = get_best_backend()
        
        print("=" * 60)
        print("GPU-Accelerated Research Platform - Phase 10b")
        print("=" * 60)
        print(f"\nAvailable GPU backends:")
        for backend in self.available_backends:
            marker = "✓" if backend == best_backend else " "
            print(f"  [{marker}] {backend}")
        print(f"\nSelected: {self.gpu_backend_name}")
        print(f"GPU enabled: {self.gpu_enabled}")
        print("=" * 60 + "\n")
        
        # Initialize parent (this creates creatures with regular networks)
        super().__init__(width, height)
        
        # Convert creatures to GPU networks if enabled
        if self.gpu_enabled:
            self._convert_to_gpu_networks()
    
    def _convert_to_gpu_networks(self):
        """Convert existing creatures to use GPU networks."""
        print("Converting creatures to GPU networks...")
        converted = 0
        
        for creature in self.creatures:
            if not hasattr(creature, 'network'):
                continue
            
            try:
                # Get current network parameters
                old_network = creature.network
                n_neurons = len(old_network.neurons)
                n_synapses = len(old_network.synapses)
                
                # Create GPU network
                gpu_network = GPUNeuralNetwork(
                    n_neurons=n_neurons,
                    n_synapses_per_neuron=40,  # Standard connectivity
                    config=self.gpu_config,
                    enable_plasticity=old_network.enable_plasticity,
                    enable_rewiring=False,  # Can add later
                )
                
                # Copy neuron states
                for i, neuron in enumerate(old_network.neurons):
                    if i < gpu_network.n_neurons:
                        # GPU networks use array indexing
                        gpu_network.backend.set_neuron_potential(i, neuron.potential)
                        gpu_network.backend.set_neuron_threshold(i, neuron.threshold)
                
                # Replace network
                creature.network = gpu_network
                converted += 1
                
            except Exception as e:
                print(f"  Warning: Failed to convert creature {creature.creature_id}: {e}")
        
        print(f"✓ Converted {converted}/{len(self.creatures)} creatures to GPU")
        if converted > 0:
            self.console.add_line(f"🚀 GPU mode: {converted} creatures using {self.gpu_config.backend}")
    
    def toggle_gpu_mode(self):
        """Toggle between GPU and CPU networks."""
        self.gpu_enabled = not self.gpu_enabled
        
        if self.gpu_enabled:
            # Convert to GPU
            self._convert_to_gpu_networks()
            self.console.add_line("🚀 GPU mode enabled")
        else:
            # Convert back to CPU
            print("Converting creatures back to CPU networks...")
            # This would require re-creating CPU networks
            # For now, just disable GPU updates
            self.console.add_line("⚠️  GPU mode disabled (creatures keep GPU networks)")
    
    def show_gpu_info(self):
        """Display GPU backend information in console."""
        self.console.add_line("=" * 40)
        self.console.add_line("GPU Backend Information:")
        self.console.add_line(f"  Enabled: {self.gpu_enabled}")
        self.console.add_line(f"  Backend: {self.gpu_config.backend}")
        
        # Show device info
        for backend in self.available_backends:
            if backend.backend.value == self.gpu_config.backend or self.gpu_config.backend == 'auto':
                mem_gb = backend.memory_bytes / (1024**3) if backend.memory_bytes > 0 else 0
                self.console.add_line(f"  Device: {backend.name}")
                if mem_gb > 0:
                    self.console.add_line(f"  Memory: {mem_gb:.1f} GB")
                self.console.add_line(f"  Compute units: {backend.compute_units}")
                break
        
        # Show GPU config
        self.console.add_line(f"  Batch size: {self.gpu_config.batch_size}")
        self.console.add_line(f"  Block size: {self.gpu_config.block_size}")
        
        # Show network stats
        if self.creatures and hasattr(self.creatures[0], 'network'):
            net = self.creatures[0].network
            if isinstance(net, GPUNeuralNetwork):
                stats = net.get_stats()
                self.console.add_line(f"  Network type: GPU")
                self.console.add_line(f"  Backend: {net.backend.backend_type.value}")
            else:
                self.console.add_line(f"  Network type: CPU")
        
        self.console.add_line("=" * 40)
    
    def handle_events(self):
        """Handle events including GPU controls."""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            
            # Let parent handle most events
            if event.type == KEYDOWN:
                if event.key == K_g:
                    # Toggle GPU mode
                    self.toggle_gpu_mode()
                    continue
                elif event.key == K_b:
                    # Show GPU backend info
                    self.show_gpu_info()
                    continue
        
        # Let parent handle the rest
        return super().handle_events()
    
    def _render_stats_overlay(self, surface):
        """Render stats overlay with GPU info."""
        # Call parent
        super()._render_stats_overlay(surface)
        
        # Add GPU indicator
        if self.gpu_enabled:
            gpu_text = f"GPU: {self.gpu_config.backend}"
            text_surface = self.small_font.render(gpu_text, True, (100, 255, 100))
            bg_surface = pygame.Surface(
                (text_surface.get_width() + 10, text_surface.get_height() + 4)
            )
            bg_surface.fill((0, 0, 0))
            bg_surface.set_alpha(180)
            
            x = 10
            y = self.height - 30
            surface.blit(bg_surface, (x - 5, y - 2))
            surface.blit(text_surface, (x, y))


def main():
    parser = argparse.ArgumentParser(description="GPU Research Platform")
    parser.add_argument("--width", type=int, default=1600, help="Window width")
    parser.add_argument("--height", type=int, default=900, help="Window height")
    parser.add_argument("--gpu-backend", type=str, default="auto",
                        help="GPU backend: auto, cuda, opencl, numpy")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Start with GPU disabled")
    args = parser.parse_args()
    
    platform = GPUResearchPlatform(
        width=args.width,
        height=args.height,
        gpu_backend=args.gpu_backend,
        gpu_enabled=not args.no_gpu,
    )
    platform.run()


if __name__ == "__main__":
    main()
