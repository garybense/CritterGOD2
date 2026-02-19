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

        # Normalize using average potential per neuron
        # Typical range is 0-500 for average potential
        avg_potential = total_potential / max(len(network.neurons), 1)
        normalized = min(1.0, avg_potential / 300.0)  # Scale so 300 avg = 1.0

        # Generate sine wave at frequency proportional to potential
        base_freq = 200.0  # Hz
        freq = base_freq + normalized * 400.0  # 200-600 Hz range

        # Use minimum amplitude floor (0.3) so audio is always audible
        # Plus modulation from network activity (0.0-0.7)
        min_amplitude = 0.3
        amplitude = min_amplitude + normalized * 0.7

        samples = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            samples[i] = np.sin(self._phase * 2.0 * np.pi) * amplitude
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
        
        Creates percussive/buzzy tones based on firing rate.
        More active networks produce brighter, noisier sound.
        """
        # Count neurons that fired (use fired_last_step for post-update state)
        firing_count = sum(1 for n in network.neurons if n.fired_last_step())
        
        # If no firings detected, try did_fire() as fallback
        if firing_count == 0:
            firing_count = sum(1 for n in network.neurons if n.did_fire())
        
        # Calculate firing ratio
        firing_ratio = firing_count / max(len(network.neurons), 1)
        
        # Generate buzzy tone - frequency based on firing ratio
        # More firing = higher, brighter frequency
        base_freq = 150.0  # Hz
        freq = base_freq + firing_ratio * 500.0  # 150-650 Hz range
        
        # Add some harmonics for texture based on firing rate
        samples = np.zeros(num_samples, dtype=np.float32)
        
        # Minimum amplitude 0.3, scales up with more firings
        amplitude = 0.3 + min(0.7, firing_ratio * 2.0)
        
        for i in range(num_samples):
            # Fundamental + harmonics for richer sound
            sample = np.sin(self._phase * 2.0 * np.pi)
            sample += 0.3 * np.sin(self._phase * 4.0 * np.pi)  # 2nd harmonic
            sample += 0.1 * np.sin(self._phase * 6.0 * np.pi)  # 3rd harmonic
            samples[i] = sample * amplitude * 0.5  # Scale down due to harmonics
            self._phase += freq / self.sample_rate
            self._phase %= 1.0

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
