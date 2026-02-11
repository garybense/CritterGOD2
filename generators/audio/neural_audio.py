"""
Neural audio synthesis - convert neural activity to sound.

Inspired by SDL neural visualizers where firing patterns directly
generate audio buffers. The audio reflects the collective state of
the neural network in real-time.
"""

from typing import Optional

import numpy as np

from core.neural.network import NeuralNetwork


class NeuralAudioSynthesizer:
    """
    Generates audio waveforms from neural network activity.
    
    Based on SDL neural visualizers (looser.c, xesu.c, etc.) where:
    - Each neuron's potential contributes to audio amplitude
    - Firing events create transients/attacks
    - Multiple synthesis modes create different sonic textures
    
    Attributes:
        sample_rate: Audio sample rate (Hz)
        buffer_size: Audio buffer size in samples
        mode: Synthesis mode ('potential', 'firing', 'mixed')
        amplitude_scale: Overall volume scaling factor
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        buffer_size: int = 4096,
        mode: str = 'mixed',
        amplitude_scale: float = 0.3,
    ):
        """
        Initialize audio synthesizer.
        
        Args:
            sample_rate: Audio sample rate in Hz (default 44100)
            buffer_size: Number of samples per buffer (default 4096)
            mode: Synthesis mode - 'potential', 'firing', or 'mixed'
            amplitude_scale: Volume scaling (0.0-1.0, default 0.3)
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.mode = mode
        self.amplitude_scale = amplitude_scale

        # Internal state
        self._phase = 0.0
        self._last_firing_neurons = set()

    def synthesize_from_network(
        self,
        network: NeuralNetwork,
        duration_seconds: float = 0.1,
    ) -> np.ndarray:
        """
        Generate audio buffer from current network state.
        
        Args:
            network: Neural network to sonify
            duration_seconds: Duration of audio to generate
            
        Returns:
            Audio samples as float32 array (-1.0 to 1.0)
        """
        num_samples = int(duration_seconds * self.sample_rate)

        if self.mode == 'potential':
            return self._synthesize_potential(network, num_samples)
        elif self.mode == 'firing':
            return self._synthesize_firing(network, num_samples)
        else:  # mixed
            return self._synthesize_mixed(network, num_samples)

    def _synthesize_potential(
        self,
        network: NeuralNetwork,
        num_samples: int,
    ) -> np.ndarray:
        """
        Synthesize audio from neuron potentials.
        
        Maps total network potential to audio amplitude.
        Creates smooth, droning textures.
        """
        # Sum all neuron potentials
        total_potential = sum(n.potential for n in network.neurons)

        # Normalize to reasonable range (assuming typical potential 0-5000)
        max_expected_potential = len(network.neurons) * 2500
        normalized = total_potential / max(max_expected_potential, 1.0)

        # Generate sine wave at frequency proportional to potential
        base_freq = 200.0  # Hz
        freq = base_freq + normalized * 400.0  # 200-600 Hz range

        samples = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            samples[i] = np.sin(self._phase * 2.0 * np.pi) * normalized
            self._phase += freq / self.sample_rate
            self._phase %= 1.0  # Keep phase in [0, 1)

        return samples * self.amplitude_scale

    def _synthesize_firing(
        self,
        network: NeuralNetwork,
        num_samples: int,
    ) -> np.ndarray:
        """
        Synthesize audio from firing events.
        
        Creates percussive attacks when neurons fire.
        More sparse, rhythmic texture.
        """
        # Find neurons that fired this step
        firing_neurons = {
            n.neuron_id for n in network.neurons
            if n.did_fire()
        }

        # Detect new firing events (wasn't firing last check)
        new_firings = firing_neurons - self._last_firing_neurons
        self._last_firing_neurons = firing_neurons

        # Generate attack transient for each new firing
        samples = np.zeros(num_samples, dtype=np.float32)

        if new_firings:
            # Create percussive envelope
            attack_samples = min(num_samples, int(self.sample_rate * 0.01))  # 10ms attack
            decay_samples = num_samples - attack_samples

            # Attack phase (0 to 1)
            if attack_samples > 0:
                samples[:attack_samples] = np.linspace(0, 1, attack_samples)

            # Decay phase (exponential)
            if decay_samples > 0:
                decay = np.exp(-3.0 * np.linspace(0, 1, decay_samples))
                samples[attack_samples:] = decay

            # Scale by number of firings (more firings = louder)
            amplitude = min(1.0, len(new_firings) / 100.0)
            samples *= amplitude

        return samples * self.amplitude_scale

    def _synthesize_mixed(
        self,
        network: NeuralNetwork,
        num_samples: int,
    ) -> np.ndarray:
        """
        Synthesize audio mixing potential and firing modes.
        
        Combines smooth drone with percussive attacks.
        Most sonically interesting mode.
        """
        potential_audio = self._synthesize_potential(network, num_samples)
        firing_audio = self._synthesize_firing(network, num_samples)

        # Mix: 60% potential, 40% firing
        return 0.6 * potential_audio + 0.4 * firing_audio

    def synthesize_from_activity(
        self,
        firing_count: int,
        total_potential: float,
        num_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate audio from raw activity metrics.
        
        Useful when you don't have direct network access.
        
        Args:
            firing_count: Number of neurons that fired
            total_potential: Sum of all neuron potentials
            num_samples: Number of samples to generate (uses buffer_size if None)
            
        Returns:
            Audio samples as float32 array
        """
        if num_samples is None:
            num_samples = self.buffer_size

        # Generate frequency from potential
        normalized_potential = min(1.0, total_potential / 100000.0)
        freq = 200.0 + normalized_potential * 400.0

        # Generate amplitude from firing rate
        amplitude = min(1.0, firing_count / 100.0)

        samples = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            samples[i] = np.sin(self._phase * 2.0 * np.pi) * amplitude
            self._phase += freq / self.sample_rate
            self._phase %= 1.0

        return samples * self.amplitude_scale

    def reset(self):
        """Reset synthesizer state."""
        self._phase = 0.0
        self._last_firing_neurons = set()
