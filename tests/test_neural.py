"""
Unit tests for neural network components.
"""

import pytest
import numpy as np
from core.neural import Neuron, Synapse, NeuralNetwork
from core.neural.neuron import NeuronType


class TestNeuron:
    """Tests for Neuron class."""
    
    def test_neuron_initialization(self):
        """Test neuron is initialized correctly."""
        neuron = Neuron(neuron_id=0, threshold=1000.0)
        assert neuron.neuron_id == 0
        assert neuron.threshold == 1000.0
        assert neuron.potential >= 0
        assert neuron.last_fire_time is None
        
    def test_neuron_fires_when_threshold_exceeded(self):
        """Test neuron fires when potential exceeds threshold."""
        neuron = Neuron(neuron_id=0, threshold=100.0, leak_rate=1.0)
        neuron.add_input(150.0)
        fired = neuron.update(time=1.0)
        assert fired
        assert neuron.potential == 0.0  # Reset after firing
        assert neuron.last_fire_time == 1.0
        
    def test_neuron_does_not_fire_below_threshold(self):
        """Test neuron doesn't fire when below threshold."""
        neuron = Neuron(neuron_id=0, threshold=100.0, leak_rate=1.0)
        neuron.potential = 0.0  # Set initial potential to known value
        neuron.add_input(50.0)
        fired = neuron.update(time=1.0)
        assert not fired
        assert neuron.last_fire_time is None
        
    def test_neuron_leak(self):
        """Test membrane potential leaks over time."""
        neuron = Neuron(neuron_id=0, threshold=1000.0, leak_rate=0.9)
        neuron.add_input(100.0)
        initial_potential = neuron.potential
        neuron.update(time=1.0)
        # Potential should have decreased due to leak
        assert neuron.potential < initial_potential
        

class TestSynapse:
    """Tests for Synapse class."""
    
    def test_synapse_initialization(self):
        """Test synapse is initialized correctly."""
        pre = Neuron(neuron_id=0)
        post = Neuron(neuron_id=1)
        synapse = Synapse(pre, post, weight=2.0, is_inhibitory=False)
        assert synapse.weight > 0  # Excitatory
        assert not synapse.is_inhibitory
        
    def test_inhibitory_synapse(self):
        """Test inhibitory synapse has negative weight."""
        pre = Neuron(neuron_id=0)
        post = Neuron(neuron_id=1)
        synapse = Synapse(pre, post, weight=2.0, is_inhibitory=True)
        assert synapse.weight < 0  # Inhibitory
        assert synapse.is_inhibitory
        
    def test_synapse_propagation(self):
        """Test synapse propagates signal when pre-neuron fires."""
        pre = Neuron(neuron_id=0, threshold=10.0, leak_rate=1.0)
        post = Neuron(neuron_id=1, threshold=1000.0, leak_rate=1.0)
        synapse = Synapse(pre, post, weight=5.0)
        
        # Make pre-neuron fire
        pre.add_input(20.0)
        pre.update(time=1.0)
        
        initial_post_potential = post.potential
        synapse.propagate()
        
        # Post neuron should have received input
        assert post.potential > initial_post_potential
        
    def test_stdp_strengthening(self):
        """Test STDP strengthens synapse when pre fires before post."""
        pre = Neuron(neuron_id=0)
        post = Neuron(neuron_id=1)
        synapse = Synapse(pre, post, weight=1.0, plasticity_rate=0.1)
        
        # Pre fires first
        pre.last_fire_time = 10.0
        post.last_fire_time = 15.0  # Post fires 5 units later
        
        initial_weight = synapse.weight
        synapse.apply_stdp(time=20.0, strengthen_rate=1.0, weaken_rate=0.1)
        
        # Weight should have increased
        assert synapse.weight > initial_weight
        

class TestNeuralNetwork:
    """Tests for NeuralNetwork class."""
    
    def test_network_initialization(self):
        """Test network initializes empty."""
        network = NeuralNetwork()
        assert len(network.neurons) == 0
        assert len(network.synapses) == 0
        assert network.time == 0.0
        
    def test_add_neurons(self):
        """Test adding neurons to network."""
        network = NeuralNetwork()
        neuron1 = Neuron(neuron_id=0)
        neuron2 = Neuron(neuron_id=1)
        network.add_neuron(neuron1)
        network.add_neuron(neuron2)
        assert len(network.neurons) == 2
        
    def test_create_random_synapses(self):
        """Test random synapse creation."""
        network = NeuralNetwork()
        for i in range(10):
            network.add_neuron(Neuron(neuron_id=i))
        network.create_random_synapses(synapses_per_neuron=5)
        assert len(network.synapses) > 0
        
    def test_network_update(self):
        """Test network update loop."""
        network = NeuralNetwork()
        for i in range(10):
            network.add_neuron(Neuron(neuron_id=i))
        network.create_random_synapses(synapses_per_neuron=3)
        
        initial_time = network.time
        network.update(dt=1.0)
        assert network.time > initial_time
        
    def test_sensory_input_injection(self):
        """Test injecting input into sensory neurons."""
        network = NeuralNetwork()
        sensory = Neuron(neuron_id=0, neuron_type=NeuronType.SENSORY, threshold=100.0, leak_rate=1.0)
        network.add_neuron(sensory)
        
        network.inject_sensory_input(neuron_id=0, amount=200.0)
        network.update(dt=1.0)
        
        # Sensory neuron should have fired
        assert sensory.last_fire_time is not None
