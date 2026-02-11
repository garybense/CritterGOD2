"""
Audio synthesis mixin for creatures.

Creatures generate sound from their neural activity using the
NeuralAudioSynthesizer from Phase 4a. Each creature has a unique
voice that changes with:
- Neural firing patterns
- Drug effects (altered frequencies, timbre)
- Energy levels (volume)
- Behavioral states

This completes the sensory-motor loop: creatures can "hear" each other
through visual patterns on Circuit8, and their sounds contribute to
the collective experience.
"""

from typing import Optional
import numpy as np
from generators.audio.neural_audio import NeuralAudioSynthesizer


class AudioSynthesisMixin:
    """
    Audio synthesis for creatures.
    
    Adds voice generation from neural activity:
    - Real-time audio from brain state
    - Drug-modulated synthesis
    - Energy-scaled amplitude
    - Per-creature unique timbre
    
    Audio is generated but not automatically played - the simulation
    can choose to mix/play creature audio or just track it for
    visualization.
    """
    
    def init_audio_synthesis(
        self,
        enable_audio: bool = True,
        sample_rate: int = 44100,
        mode: str = 'mixed',
        base_amplitude: float = 0.1
    ):
        """
        Initialize audio synthesis system.
        
        Args:
            enable_audio: Enable audio generation
            sample_rate: Audio sample rate (Hz)
            mode: Synthesis mode ('potential', 'firing', 'mixed')
            base_amplitude: Base volume (0.0-1.0)
        """
        self.audio_enabled = enable_audio
        self.audio_synthesizer: Optional[NeuralAudioSynthesizer] = None
        self.last_audio_buffer = None
        self.audio_mode = mode
        
        if enable_audio:
            self.audio_synthesizer = NeuralAudioSynthesizer(
                sample_rate=sample_rate,
                buffer_size=2048,  # Small buffer for responsiveness
                mode=mode,
                amplitude_scale=base_amplitude
            )
            print(f"🔊 Audio: {mode} synthesis @ {sample_rate}Hz")
    
    def generate_audio(self, duration_seconds: float = 0.05) -> Optional[np.ndarray]:
        """
        Generate audio buffer from current neural state.
        
        Call this each frame to generate real-time audio.
        
        Args:
            duration_seconds: Duration of audio to generate
            
        Returns:
            Audio samples as float32 array, or None if disabled
        """
        if not self.audio_enabled or not self.audio_synthesizer:
            return None
        
        if not hasattr(self, 'network'):
            return None
        
        # Scale amplitude by energy level
        energy_scale = 1.0
        if hasattr(self, 'energy'):
            energy_ratio = self.energy.energy / self.energy.max_energy
            energy_scale = max(0.1, energy_ratio)  # Quieter when low energy
        
        # Adjust synthesis amplitude
        original_amplitude = self.audio_synthesizer.amplitude_scale
        self.audio_synthesizer.amplitude_scale = original_amplitude * energy_scale
        
        # Synthesize from network
        audio = self.audio_synthesizer.synthesize_from_network(
            self.network,
            duration_seconds=duration_seconds
        )
        
        # Apply drug effects to audio (frequency/timbre modulation)
        if hasattr(self, 'drugs'):
            audio = self._apply_drug_audio_effects(audio)
        
        # Restore original amplitude
        self.audio_synthesizer.amplitude_scale = original_amplitude
        
        # Cache for visualization
        self.last_audio_buffer = audio
        
        return audio
    
    def _apply_drug_audio_effects(self, audio: np.ndarray) -> np.ndarray:
        """
        Modulate audio based on drug effects.
        
        Different drugs create different sonic signatures:
        - Inhibitory: Lower frequencies, muted
        - Excitatory: Higher frequencies, sharper
        - Potentiator: Amplified, distorted
        
        Args:
            audio: Original audio buffer
            
        Returns:
            Drug-modulated audio
        """
        if not hasattr(self, 'drugs'):
            return audio
        
        modified = audio.copy()
        
        # Get drug levels
        from core.pharmacology.drugs import MoleculeType
        inhibitory = (
            self.drugs.tripping[MoleculeType.INHIBITORY_ANTAGONIST] +
            self.drugs.tripping[MoleculeType.INHIBITORY_AGONIST]
        )
        excitatory = (
            self.drugs.tripping[MoleculeType.EXCITATORY_ANTAGONIST] +
            self.drugs.tripping[MoleculeType.EXCITATORY_AGONIST]
        )
        potentiator = self.drugs.tripping[MoleculeType.POTENTIATOR]
        
        # Normalize to 0-1 range
        total_max = self.drugs.max_trip * 2  # Max for combined drugs
        inhibitory_norm = min(1.0, inhibitory / total_max)
        excitatory_norm = min(1.0, excitatory / total_max)
        potentiator_norm = min(1.0, potentiator / self.drugs.max_trip)
        
        # Inhibitory drugs: low-pass filter effect (dampen high frequencies)
        if inhibitory_norm > 0.1:
            # Simple smoothing = low-pass
            smoothed = np.convolve(modified, np.ones(3)/3, mode='same')
            modified = modified * (1 - inhibitory_norm) + smoothed * inhibitory_norm
        
        # Excitatory drugs: high-pass emphasis (sharpen)
        if excitatory_norm > 0.1:
            # Emphasize changes = high-pass
            diff = np.diff(modified, prepend=modified[0])
            modified = modified + diff * excitatory_norm * 0.5
        
        # Potentiator: amplify and add harmonic distortion
        if potentiator_norm > 0.1:
            # Soft clipping for distortion
            gain = 1.0 + potentiator_norm * 3.0
            modified = modified * gain
            modified = np.tanh(modified)  # Soft clipping
        
        return modified
    
    def get_audio_waveform_display(self, num_points: int = 100) -> Optional[np.ndarray]:
        """
        Get simplified waveform for visualization.
        
        Args:
            num_points: Number of points for display
            
        Returns:
            Downsampled waveform or None
        """
        if self.last_audio_buffer is None:
            return None
        
        # Downsample for display
        step = max(1, len(self.last_audio_buffer) // num_points)
        return self.last_audio_buffer[::step]
    
    def get_audio_energy(self) -> float:
        """
        Get current audio energy (RMS) for visualization.
        
        Returns:
            RMS energy (0.0-1.0)
        """
        if self.last_audio_buffer is None:
            return 0.0
        
        rms = np.sqrt(np.mean(self.last_audio_buffer ** 2))
        return float(rms)
    
    def set_audio_mode(self, mode: str):
        """
        Change audio synthesis mode.
        
        Args:
            mode: 'potential', 'firing', or 'mixed'
        """
        if self.audio_synthesizer:
            self.audio_synthesizer.mode = mode
            self.audio_mode = mode
