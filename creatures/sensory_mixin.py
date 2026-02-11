"""
Complete Multimodal Sensory Integration

THE KEY TO EMERGENCE: Multiple sensory streams feeding into a single
neural substrate allow unexpected cross-modal associations. The brain
can discover novel correlations between vision, proprioception, drugs,
and social cues that were never explicitly programmed.

This creates the conditions for genuinely surprising behaviors.
"""

import numpy as np
from typing import Optional
from generators.visual.retinal_sensors import RetinalArray


class CompleteSensoryMixin:
    """
    Complete sensory integration across all modalities.
    
    Sensory Modalities:
    1. VISION: Retinal array reading Circuit8 patterns
    2. PROPRIOCEPTION: Body state (velocity, energy, size)
    3. CHEMORECEPTION: Drug levels, internal chemistry  
    4. INTEROCEPTION: Hunger, pain, pleasure signals
    5. SOCIAL: Awareness of nearby creatures (via visual)
    
    The magic happens when these combine in the neural network.
    Example emergent behaviors:
    - Visual drug-seeking (seeing patterns → craving)
    - Energy-driven exploration (low energy → seek bright areas)
    - Social mimicry (seeing movement → copying motor patterns)
    - Drug-modulated vision (altered perception → altered behavior)
    """
    
    def init_complete_senses(
        self,
        enable_vision: bool = True,
        vision_sensors: int = 32,  # Performance-optimized
        neurons_per_sensor: int = 8
    ):
        """
        Initialize complete sensory apparatus.
        
        Args:
            enable_vision: Enable retinal vision system
            vision_sensors: Number of retinal sensors
            neurons_per_sensor: Neurons sampling each sensor column
        """
        # Vision system
        self.vision_enabled = enable_vision
        self.retinal_array: Optional[RetinalArray] = None
        self.visual_input = None
        
        if enable_vision and hasattr(self, 'circuit8') and self.circuit8:
            self.retinal_array = RetinalArray(
                num_sensors=vision_sensors,
                screen_width=self.circuit8.width,
                screen_height=self.circuit8.height,
                neurons_per_sensor=neurons_per_sensor
            )
            total_visual_neurons = self.retinal_array.get_total_neurons()
            print(f"👁️  Vision: {total_visual_neurons} neurons ({vision_sensors}×{neurons_per_sensor})")
        
        # Proprioception state
        self.body_state = {
            'velocity': 0.0,
            'energy_ratio': 0.0,
            'mass': 1.0,
            'z_position': 0.0
        }
        
        # Interoception (internal sensations)
        self.internal_state = {
            'hunger': 0.0,      # 0=sated, 1=starving
            'pain': 0.0,        # Damage, withdrawal
            'pleasure': 0.0,    # Drug effects, success
            'arousal': 0.0      # General excitation level
        }
    
    def sense_environment(self):
        """
        Complete sensory update - call BEFORE network.update().
        
        This gathers all sensory information and prepares it for
        injection into the neural network.
        """
        # Visual perception
        if self.vision_enabled and self.retinal_array:
            self._sense_vision()
        
        # Body awareness
        self._sense_proprioception()
        
        # Internal states
        self._sense_interoception()
        
        # Chemical senses
        self._sense_chemistry()
    
    def inject_senses_to_brain(self):
        """
        Inject ALL sensory data into neural network.
        
        This is where cross-modal magic happens. Visual neurons next
        to drug neurons next to proprioceptive neurons = unexpected
        associations form through STDP.
        
        Call AFTER sense_environment(), BEFORE network.update().
        """
        if not hasattr(self, 'network'):
            return
        
        from core.neural.neuron import NeuronType
        sensory_neurons = [n for n in self.network.neurons if n.neuron_type == NeuronType.SENSORY]
        
        if not sensory_neurons:
            return
        
        idx = 0
        
        # 1. VISION (majority of sensory neurons)
        if self.visual_input is not None:
            visual_count = min(len(sensory_neurons) - idx, len(self.visual_input))
            for i in range(visual_count):
                sensory_neurons[idx].add_input(self.visual_input[i])
                idx += 1
        
        # 2. PROPRIOCEPTION (body state)
        props = [
            self.body_state['velocity'] * 100.0,
            self.body_state['energy_ratio'] * 1000.0,
            self.body_state['mass'] * 10.0,
            self.body_state['z_position'],
        ]
        for val in props:
            if idx >= len(sensory_neurons):
                break
            sensory_neurons[idx].add_input(val)
            idx += 1
        
        # 3. INTEROCEPTION (internal feelings)
        interos = [
            self.internal_state['hunger'] * 800.0,
            self.internal_state['pain'] * 600.0,
            self.internal_state['pleasure'] * 400.0,
            self.internal_state['arousal'] * 300.0,
        ]
        for val in interos:
            if idx >= len(sensory_neurons):
                break
            sensory_neurons[idx].add_input(val)
            idx += 1
        
        # 4. CHEMORECEPTION (drugs)
        if hasattr(self, 'drugs'):
            for drug_level in self.drugs.tripping:
                if idx >= len(sensory_neurons):
                    break
                activation = (drug_level / self.drugs.max_trip) * 500.0
                sensory_neurons[idx].add_input(activation)
                idx += 1
        
        # 5. Fill remaining with noise (prevents dead neurons, adds stochasticity)
        while idx < len(sensory_neurons):
            noise = np.random.uniform(50.0, 250.0)
            sensory_neurons[idx].add_input(noise)
            idx += 1
    
    def _sense_vision(self):
        """Read visual environment through retinal array."""
        if not hasattr(self, 'circuit8'):
            return
        
        # Convert Circuit8 to screen format
        screen = np.zeros((self.circuit8.height, self.circuit8.width, 3), dtype=np.uint8)
        for y in range(self.circuit8.height):
            for x in range(self.circuit8.width):
                r, g, b = self.circuit8.read_pixel(x, y)
                screen[y, x] = [r, g, b]
        
        # Read through retinal sensors
        self.visual_input = self.retinal_array.read_screen(screen)
    
    def _sense_proprioception(self):
        """Sense body state."""
        # Velocity
        if hasattr(self, 'rigid_body') and self.rigid_body:
            vel = self.rigid_body.get_velocity()
            self.body_state['velocity'] = np.linalg.norm(vel)
            self.body_state['z_position'] = self.rigid_body.position[2]
        
        # Energy ratio
        if hasattr(self, 'energy'):
            self.body_state['energy_ratio'] = self.energy.energy / self.energy.max_energy
        
        # Mass
        if hasattr(self, 'body') and self.body:
            self.body_state['mass'] = self.body.get_total_mass()
    
    def _sense_interoception(self):
        """Sense internal states."""
        # Hunger increases as energy decreases
        if hasattr(self, 'energy'):
            energy_ratio = self.energy.energy / self.energy.max_energy
            self.internal_state['hunger'] = 1.0 - energy_ratio
        
        # Pain from withdrawal
        if hasattr(self, 'behavior') and hasattr(self.behavior, 'withdrawal_severity'):
            self.internal_state['pain'] = self.behavior.withdrawal_severity
        
        # Pleasure from drugs
        if hasattr(self, 'drugs'):
            total_drugs = np.sum(self.drugs.tripping)
            self.internal_state['pleasure'] = min(1.0, total_drugs / 10000.0)
        
        # Arousal from network activity
        if hasattr(self, 'network'):
            activity = self.network.get_activity_level()
            self.internal_state['arousal'] = activity
    
    def _sense_chemistry(self):
        """Sense chemical environment (drugs available in self.drugs)."""
        # Drug levels automatically available through self.drugs.tripping
        # This gets injected in inject_senses_to_brain()
        pass
    
    def get_visual_focus(self) -> Optional[tuple]:
        """
        Get the (x,y) point in Circuit8 with highest visual activation.
        
        Returns:
            (x, y) coordinates or None
        """
        if self.visual_input is None or not self.retinal_array:
            return None
        
        neurons_per = self.retinal_array.neurons_per_sensor
        num_sensors = self.retinal_array.num_sensors
        
        max_activation = -1
        max_sensor_idx = 0
        
        for i in range(num_sensors):
            start = i * neurons_per
            end = start + neurons_per
            avg_activation = np.mean(self.visual_input[start:end])
            
            if avg_activation > max_activation:
                max_activation = avg_activation
                max_sensor_idx = i
        
        # Map to Circuit8 coordinates
        sensor = self.retinal_array.sensors[max_sensor_idx]
        x = sensor.x_position
        y = (sensor.y_start + sensor.y_end) // 2
        
        return (x, y)
    
    def visualize_sensory_state(self) -> dict:
        """
        Get complete sensory state for visualization/debugging.
        
        Returns:
            Dictionary of all sensory values
        """
        visual_focus = self.get_visual_focus()
        
        return {
            'vision': {
                'enabled': self.vision_enabled,
                'neurons': len(self.visual_input) if self.visual_input is not None else 0,
                'focus_point': visual_focus,
                'avg_activation': np.mean(self.visual_input) if self.visual_input is not None else 0.0
            },
            'proprioception': self.body_state.copy(),
            'interoception': self.internal_state.copy(),
            'drugs': {
                'tripping': list(self.drugs.tripping) if hasattr(self, 'drugs') else []
            }
        }
