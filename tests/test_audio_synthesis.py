"""
Tests for neural audio synthesis system.
"""

import numpy as np

from core.neural import NeuralNetwork, Neuron
from generators.audio import NeuralAudioSynthesizer


class TestNeuralAudioSynthesizer:
    """Tests for NeuralAudioSynthesizer class."""

    def test_synthesizer_initialization(self):
        """Test synthesizer initializes with correct parameters."""
        synth = NeuralAudioSynthesizer(
            sample_rate=44100,
            buffer_size=4096,
            mode='mixed',
            amplitude_scale=0.5,
        )
        assert synth.sample_rate == 44100
        assert synth.buffer_size == 4096
        assert synth.mode == 'mixed'
        assert synth.amplitude_scale == 0.5

    def test_synthesize_from_network_returns_correct_shape(self):
        """Test synthesis returns correct number of samples."""
        network = NeuralNetwork()
        for i in range(10):
            network.add_neuron(Neuron(neuron_id=i, threshold=1000.0))

        synth = NeuralAudioSynthesizer(sample_rate=44100)

        # Generate 0.1 seconds of audio
        samples = synth.synthesize_from_network(network, duration_seconds=0.1)

        expected_samples = int(0.1 * 44100)
        assert len(samples) == expected_samples
        assert samples.dtype == np.float32

    def test_synthesize_from_network_range(self):
        """Test audio samples are in valid range (-1 to 1)."""
        network = NeuralNetwork()
        for i in range(10):
            neuron = Neuron(neuron_id=i, threshold=100.0, leak_rate=1.0)
            neuron.add_input(5000.0)  # High energy
            network.add_neuron(neuron)

        synth = NeuralAudioSynthesizer(sample_rate=44100, amplitude_scale=0.3)
        samples = synth.synthesize_from_network(network, duration_seconds=0.01)

        # Samples should be within [-1, 1] range
        assert np.all(samples >= -1.0)
        assert np.all(samples <= 1.0)

    def test_potential_mode(self):
        """Test potential synthesis mode produces output."""
        network = NeuralNetwork()
        for i in range(10):
            neuron = Neuron(neuron_id=i, threshold=1000.0)
            neuron.potential = 2000.0  # High potential
            network.add_neuron(neuron)

        synth = NeuralAudioSynthesizer(mode='potential')
        samples = synth.synthesize_from_network(network, duration_seconds=0.01)

        # Should produce non-zero output
        assert np.any(samples != 0)

    def test_firing_mode(self):
        """Test firing synthesis mode produces output on firing."""
        network = NeuralNetwork()
        for i in range(10):
            neuron = Neuron(neuron_id=i, threshold=100.0, leak_rate=1.0)
            network.add_neuron(neuron)

        synth = NeuralAudioSynthesizer(mode='firing')

        # First call - no previous firing state
        samples1 = synth.synthesize_from_network(network, duration_seconds=0.01)

        # Make neurons fire and call synthesizer BEFORE network resets state
        for neuron in network.neurons:
            neuron.add_input(200.0)
            neuron.update(time=1.0)  # Update neurons directly so they fire

        # Second call - should detect new firings (neurons still have _fired_this_step=True)
        samples2 = synth.synthesize_from_network(network, duration_seconds=0.01)

        # Should produce output after firing
        assert np.any(samples2 != 0)

    def test_mixed_mode(self):
        """Test mixed mode combines potential and firing."""
        network = NeuralNetwork()
        for i in range(10):
            neuron = Neuron(neuron_id=i, threshold=100.0, leak_rate=1.0)
            neuron.potential = 1000.0
            network.add_neuron(neuron)

        synth = NeuralAudioSynthesizer(mode='mixed')
        samples = synth.synthesize_from_network(network, duration_seconds=0.01)

        # Should produce non-zero output
        assert np.any(samples != 0)

    def test_synthesize_from_activity(self):
        """Test synthesis from raw activity metrics."""
        synth = NeuralAudioSynthesizer(sample_rate=44100)

        samples = synth.synthesize_from_activity(
            firing_count=50,
            total_potential=50000.0,
            num_samples=1000,
        )

        assert len(samples) == 1000
        assert samples.dtype == np.float32
        assert np.any(samples != 0)

    def test_reset(self):
        """Test synthesizer reset clears state."""
        synth = NeuralAudioSynthesizer()

        # Set some state
        synth._phase = 0.5
        synth._last_firing_neurons = {1, 2, 3}

        # Reset
        synth.reset()

        # State should be cleared
        assert synth._phase == 0.0
        assert len(synth._last_firing_neurons) == 0

    def test_amplitude_scaling(self):
        """Test amplitude scale affects output volume."""
        network = NeuralNetwork()
        for i in range(10):
            neuron = Neuron(neuron_id=i, threshold=100.0)
            neuron.potential = 2000.0
            network.add_neuron(neuron)

        # Low amplitude
        synth1 = NeuralAudioSynthesizer(mode='potential', amplitude_scale=0.1)
        samples1 = synth1.synthesize_from_network(network, duration_seconds=0.01)

        # High amplitude
        synth2 = NeuralAudioSynthesizer(mode='potential', amplitude_scale=0.5)
        samples2 = synth2.synthesize_from_network(network, duration_seconds=0.01)

        # Higher amplitude should produce louder output
        assert np.max(np.abs(samples2)) > np.max(np.abs(samples1))

    def test_phase_continuity(self):
        """Test phase continues smoothly across calls."""
        network = NeuralNetwork()
        for i in range(10):
            neuron = Neuron(neuron_id=i)
            neuron.potential = 2000.0
            network.add_neuron(neuron)

        synth = NeuralAudioSynthesizer(mode='potential')

        # Generate multiple buffers
        phase_before = synth._phase
        samples1 = synth.synthesize_from_network(network, duration_seconds=0.01)
        phase_after = synth._phase

        # Phase should have advanced
        assert phase_after != phase_before

        # Should be in [0, 1) range
        assert 0.0 <= phase_after < 1.0
