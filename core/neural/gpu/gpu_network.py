"""
GPU-Accelerated Neural Network

High-performance spiking neural network for 10,000+ neuron networks.
Uses vectorized array operations that can run on CPU (NumPy), CUDA (CuPy), or OpenCL.

This is a Structure-of-Arrays (SoA) implementation for GPU efficiency,
contrasted with the Array-of-Structures (AoS) approach in the original network.py.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Any
import logging

from .config import GPUConfig, estimate_memory_usage
from .backend import Backend, get_backend

logger = logging.getLogger(__name__)


class GPUNeuralNetwork:
    """
    GPU-accelerated spiking neural network.
    
    Key differences from CPU version:
    - Structure-of-Arrays layout (SoA) for GPU memory coalescing
    - Vectorized operations for all neural updates
    - Batch processing of synaptic propagation
    - Supports CUDA, OpenCL, and CPU (NumPy) backends
    
    Attributes:
        n_neurons: Number of neurons
        n_synapses: Number of synapses
        config: GPU configuration
        backend: Compute backend (CUDA/OpenCL/NumPy)
    """
    
    # Network parameters (matching original)
    WEIGHT_CLAMP = (-5.0, 5.0)     # From CritterGOD4
    DECAY_FACTOR = 0.99           # Continuous weakening
    STDP_WINDOW = 40.0            # Plasticity window
    PRUNE_THRESHOLD = 0.1         # For dynamic rewiring
    LEAK_RATE = 0.9               # Neuron leak
    
    def __init__(
        self,
        n_neurons: int = 1000,
        n_synapses_per_neuron: int = 40,
        config: Optional[GPUConfig] = None,
        enable_plasticity: bool = True,
        enable_rewiring: bool = True,
        inhibitory_ratio: float = 0.3,
    ):
        """
        Initialize GPU neural network.
        
        Args:
            n_neurons: Number of neurons
            n_synapses_per_neuron: Average synapses per neuron
            config: GPU configuration (auto-detect if None)
            enable_plasticity: Enable STDP plasticity
            enable_rewiring: Enable dynamic rewiring
            inhibitory_ratio: Fraction of inhibitory neurons (0.3 from CritterGOD4)
        """
        self.config = config or GPUConfig()
        self.config.validate()
        
        # Initialize backend
        self.backend = get_backend(self.config)
        logger.info(f"Using backend: {self.backend.backend_type.value}")
        
        # Network state
        self.n_neurons = n_neurons
        self.enable_plasticity = enable_plasticity
        self.enable_rewiring = enable_rewiring
        self.time = 0.0
        
        # Estimate synapse count
        estimated_synapses = n_neurons * n_synapses_per_neuron
        
        # Check memory
        mem_estimate = estimate_memory_usage(n_neurons, estimated_synapses, self.config)
        logger.info(f"Estimated memory usage: {mem_estimate / 1e6:.1f} MB")
        
        # Initialize neuron arrays (SoA layout)
        self._init_neurons(n_neurons, inhibitory_ratio)
        
        # Initialize synapse arrays
        self._init_synapses(n_synapses_per_neuron)
        
        # Work buffers
        self._input_buffer = self.backend.zeros((n_neurons,))
        
        # Statistics
        self._update_count = 0
        self._last_activity = 0.0
    
    def _init_neurons(self, n_neurons: int, inhibitory_ratio: float):
        """Initialize neuron state arrays."""
        # Potentials: from looser.c - rand() % 5000
        self.potentials = self.backend.random_uniform(0, 5000, (n_neurons,))
        
        # Thresholds: from looser.c - 700 + rand() % 8000
        # Some neurons get negative thresholds (bidirectional from Critterding2)
        thresholds = np.random.uniform(700, 8700, n_neurons).astype(np.float32)
        # Make some thresholds negative for inhibitory-fire neurons
        neg_mask = np.random.random(n_neurons) < 0.1  # 10% negative threshold
        thresholds[neg_mask] *= -1
        self.thresholds = self.backend.array(thresholds)
        
        # Neuron types: 0=regular, 1=sensory, 2=motor, 3=inhibitory
        types = np.zeros(n_neurons, dtype=np.int32)
        # First 10% sensory, last 10% motor, 30% inhibitory (from CritterGOD4)
        n_sensory = n_neurons // 10
        n_motor = n_neurons // 10
        types[:n_sensory] = 1  # Sensory
        types[-n_motor:] = 2   # Motor
        # Mark inhibitory neurons
        n_inhibitory = int(n_neurons * inhibitory_ratio)
        inhibitory_indices = np.random.choice(
            np.arange(n_sensory, n_neurons - n_motor),
            size=min(n_inhibitory, n_neurons - n_sensory - n_motor),
            replace=False
        )
        types[inhibitory_indices] = 3
        self.neuron_types = self.backend.array(types, dtype=np.int32)
        
        # Firing state
        self.fired = self.backend.zeros((n_neurons,), dtype=np.bool_)
        self.fired_last = self.backend.zeros((n_neurons,), dtype=np.bool_)
        
        # Last fire time (for STDP)
        self.last_fire_time = self.backend.zeros((n_neurons,))
        
        # Plasticity flags
        plasticity = np.random.random(n_neurons) < 0.5  # 50% plastic
        self.is_plastic = self.backend.array(plasticity, dtype=np.bool_)
    
    def _init_synapses(self, synapses_per_neuron: int):
        """Initialize synapse arrays with random connectivity."""
        n = self.n_neurons
        
        # Generate random synapses
        pre_list = []
        post_list = []
        weight_list = []
        inhib_list = []
        
        # Get inhibitory neuron mask
        types_cpu = self.backend.to_cpu(self.neuron_types)
        
        for i in range(n):
            # Variable number of synapses per neuron
            n_syn = max(2, int(np.random.normal(synapses_per_neuron, synapses_per_neuron * 0.2)))
            
            for _ in range(n_syn):
                target = np.random.randint(0, n)
                if target == i:
                    continue  # No self-connections
                
                pre_list.append(i)
                post_list.append(target)
                
                # Initial weight: 50-300 from original
                weight = np.random.uniform(50.0, 300.0)
                
                # Inhibitory if pre-neuron is inhibitory type
                is_inhib = types_cpu[i] == 3 or np.random.random() < 0.21
                if is_inhib:
                    weight = -weight
                
                weight_list.append(weight)
                inhib_list.append(is_inhib)
        
        self.n_synapses = len(pre_list)
        logger.info(f"Created {self.n_synapses} synapses ({self.n_synapses / self.n_neurons:.1f} per neuron)")
        
        # Convert to GPU arrays
        self.pre_indices = self.backend.array(pre_list, dtype=np.int32)
        self.post_indices = self.backend.array(post_list, dtype=np.int32)
        self.weights = self.backend.array(weight_list)
        self.is_inhibitory = self.backend.array(inhib_list, dtype=np.bool_)
        
        # Clamp initial weights
        weights_cpu = self.backend.to_cpu(self.weights)
        np.clip(weights_cpu, self.WEIGHT_CLAMP[0], self.WEIGHT_CLAMP[1], out=weights_cpu)
        self.weights = self.backend.array(weights_cpu)
    
    def update(self, dt: float = 1.0, drug_system=None) -> None:
        """
        Execute one timestep of the network.
        
        GPU-accelerated version of the main update loop:
        1. Reset input buffer
        2. Update neurons (leak + fire check)
        3. Propagate spikes through synapses
        4. Add inputs to potentials
        5. Apply continuous weakening
        6. Apply STDP plasticity
        7. Update state
        
        Args:
            dt: Time delta
            drug_system: Optional drug system for plasticity modulation
        """
        self.time += dt
        self._update_count += 1
        
        # Save previous fired state
        fired_cpu = self.backend.to_cpu(self.fired)
        self.fired_last = self.backend.array(fired_cpu, dtype=np.bool_)
        
        # 1. Reset input buffer
        self._input_buffer = self.backend.zeros((self.n_neurons,))
        
        # 2. Update neurons (leak + fire)
        self.backend.neuron_update(
            self.potentials,
            self.thresholds,
            self.LEAK_RATE,
            self.fired,
            self.last_fire_time,
            self.time
        )
        
        # 3. Propagate spikes
        self.backend.synapse_propagate(
            self.pre_indices,
            self.post_indices,
            self.weights,
            self.fired,
            self._input_buffer
        )
        
        # 4. Add inputs to potentials
        potentials_cpu = self.backend.to_cpu(self.potentials)
        inputs_cpu = self.backend.to_cpu(self._input_buffer)
        potentials_cpu += inputs_cpu
        self.potentials = self.backend.array(potentials_cpu)
        
        # 5. Apply continuous weakening
        self.backend.synapse_decay(
            self.weights,
            self.DECAY_FACTOR,
            self.WEIGHT_CLAMP
        )
        
        # 6. Apply STDP plasticity
        if self.enable_plasticity:
            # Get plasticity modulation from drugs
            strengthen_rate = 0.1
            weaken_rate = 0.05
            
            if drug_system is not None:
                # Average modulation across inhibitory/excitatory
                s_exc, w_exc = drug_system.get_plasticity_modulation(False)
                s_inh, w_inh = drug_system.get_plasticity_modulation(True)
                strengthen_rate *= (s_exc + s_inh) / 2
                weaken_rate *= (w_exc + w_inh) / 2
            
            self.backend.synapse_stdp(
                self.pre_indices,
                self.post_indices,
                self.weights,
                self.last_fire_time,
                self.last_fire_time,  # Use same array for post
                plasticity_rate=0.01,
                strengthen_rate=strengthen_rate,
                weaken_rate=weaken_rate,
                stdp_window=self.STDP_WINDOW,
                weight_clamp=self.WEIGHT_CLAMP
            )
        
        # 7. Dynamic rewiring (every 10 timesteps)
        if self.enable_rewiring and self._update_count % 10 == 0:
            self._prune_weak_synapses()
        
        # Update activity statistic
        fired_cpu = self.backend.to_cpu(self.fired)
        self._last_activity = np.sum(fired_cpu) / self.n_neurons
    
    def _prune_weak_synapses(self):
        """Remove synapses with weights below threshold."""
        weights_cpu = self.backend.to_cpu(self.weights)
        mask = np.abs(weights_cpu) >= self.PRUNE_THRESHOLD
        
        if np.all(mask):
            return  # Nothing to prune
        
        # Keep only strong synapses
        pre_cpu = self.backend.to_cpu(self.pre_indices)
        post_cpu = self.backend.to_cpu(self.post_indices)
        inhib_cpu = self.backend.to_cpu(self.is_inhibitory)
        
        self.pre_indices = self.backend.array(pre_cpu[mask], dtype=np.int32)
        self.post_indices = self.backend.array(post_cpu[mask], dtype=np.int32)
        self.weights = self.backend.array(weights_cpu[mask])
        self.is_inhibitory = self.backend.array(inhib_cpu[mask], dtype=np.bool_)
        
        pruned = self.n_synapses - int(np.sum(mask))
        self.n_synapses = int(np.sum(mask))
        
        if pruned > 0:
            logger.debug(f"Pruned {pruned} weak synapses, {self.n_synapses} remaining")
    
    def inject_input(self, neuron_indices: np.ndarray, amounts: np.ndarray):
        """
        Inject input to specific neurons.
        
        Args:
            neuron_indices: Array of neuron indices
            amounts: Array of input amounts
        """
        potentials_cpu = self.backend.to_cpu(self.potentials)
        potentials_cpu[neuron_indices] += amounts
        self.potentials = self.backend.array(potentials_cpu)
    
    def inject_sensory_input(self, neuron_id: int, amount: float):
        """Inject input to a single sensory neuron (compatibility method)."""
        if neuron_id < self.n_neurons:
            types_cpu = self.backend.to_cpu(self.neuron_types)
            if types_cpu[neuron_id] == 1:  # Sensory
                potentials_cpu = self.backend.to_cpu(self.potentials)
                potentials_cpu[neuron_id] += amount
                self.potentials = self.backend.array(potentials_cpu)
    
    def get_activity_level(self) -> float:
        """Get fraction of neurons that fired last step."""
        return self._last_activity
    
    def get_firing_neurons(self) -> List[int]:
        """Get list of neuron IDs that fired last step."""
        fired_cpu = self.backend.to_cpu(self.fired_last)
        return list(np.where(fired_cpu)[0])
    
    def get_motor_outputs(self) -> Dict[int, bool]:
        """Get firing state of motor neurons."""
        types_cpu = self.backend.to_cpu(self.neuron_types)
        fired_cpu = self.backend.to_cpu(self.fired_last)
        
        motor_mask = types_cpu == 2
        motor_indices = np.where(motor_mask)[0]
        
        return {int(idx): bool(fired_cpu[idx]) for idx in motor_indices}
    
    def get_potentials(self) -> np.ndarray:
        """Get neuron potentials as CPU array."""
        return self.backend.to_cpu(self.potentials)
    
    def get_weights(self) -> np.ndarray:
        """Get synapse weights as CPU array."""
        return self.backend.to_cpu(self.weights)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        weights_cpu = self.backend.to_cpu(self.weights)
        potentials_cpu = self.backend.to_cpu(self.potentials)
        
        return {
            "n_neurons": self.n_neurons,
            "n_synapses": self.n_synapses,
            "activity": self._last_activity,
            "time": self.time,
            "backend": self.backend.backend_type.value,
            "avg_potential": float(np.mean(potentials_cpu)),
            "avg_weight": float(np.mean(np.abs(weights_cpu))),
            "max_weight": float(np.max(np.abs(weights_cpu))),
            "weak_synapses": int(np.sum(np.abs(weights_cpu) < self.PRUNE_THRESHOLD * 2)),
        }
    
    def synchronize(self):
        """Force GPU synchronization (for timing)."""
        self.backend.synchronize()
    
    def __repr__(self) -> str:
        return (
            f"GPUNeuralNetwork(neurons={self.n_neurons}, synapses={self.n_synapses}, "
            f"backend={self.backend.backend_type.value}, time={self.time:.1f})"
        )


def create_gpu_network_from_genotype(genotype, config: Optional[GPUConfig] = None) -> GPUNeuralNetwork:
    """
    Create a GPU neural network from a genotype.
    
    This allows using GPU acceleration with the existing evolution system.
    
    Args:
        genotype: Genotype from evolution system
        config: GPU configuration
        
    Returns:
        GPUNeuralNetwork initialized from genotype
    """
    # Count neurons and synapses from genotype
    n_neurons = len(genotype.neuron_genes)
    avg_synapses = len(genotype.synapse_genes) / max(1, n_neurons)
    
    # Create network
    network = GPUNeuralNetwork(
        n_neurons=n_neurons,
        n_synapses_per_neuron=int(avg_synapses),
        config=config,
        enable_plasticity=True,
        enable_rewiring=True
    )
    
    # Override with genotype values
    potentials = []
    thresholds = []
    types = []
    
    for gene in genotype.neuron_genes:
        potentials.append(gene.initial_potential)
        thresholds.append(gene.threshold)
        types.append(gene.neuron_type.value)
    
    network.potentials = network.backend.array(potentials)
    network.thresholds = network.backend.array(thresholds)
    network.neuron_types = network.backend.array(types, dtype=np.int32)
    
    # Override synapses
    if genotype.synapse_genes:
        pre_list = []
        post_list = []
        weight_list = []
        inhib_list = []
        
        for gene in genotype.synapse_genes:
            pre_list.append(gene.pre_neuron_idx)
            post_list.append(gene.post_neuron_idx)
            weight_list.append(gene.weight if not gene.is_inhibitory else -gene.weight)
            inhib_list.append(gene.is_inhibitory)
        
        network.pre_indices = network.backend.array(pre_list, dtype=np.int32)
        network.post_indices = network.backend.array(post_list, dtype=np.int32)
        network.weights = network.backend.array(weight_list)
        network.is_inhibitory = network.backend.array(inhib_list, dtype=np.bool_)
        network.n_synapses = len(pre_list)
    
    return network
