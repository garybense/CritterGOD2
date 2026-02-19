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
    
    # Critterdrug heritage sensor counts (from map-retinal-sensors.py)
    ENERGY_BAR_SENSORS = 10   # 10 segmented energy bar sensors (IDs 30000-30009)
    AGE_BAR_SENSORS = 10      # 10 segmented age bar sensors (IDs 40000-40009)
    # Touch sensors: touchingfood=10000, touchingcritter=10001, touchingpill=10002
    # canProcreate=20000
    NUM_HERITAGE_SENSORS = ENERGY_BAR_SENSORS + AGE_BAR_SENSORS + 3 + 1  # 24 total
    
    def init_complete_senses(
        self,
        enable_vision: bool = True,
        vision_sensors: int = 32,  # Performance-optimized
        neurons_per_sensor: int = 8
    ):
        """
        Initialize complete sensory apparatus.
        
        Includes critterdrug heritage sensors (from map-retinal-sensors.py):
        - 10 energy bar sensors (segmented battery display)
        - 10 age bar sensors (life progress)
        - Touch sensors: touchingFood, touchingCreature, touchingPill
        - canProcreate sensor
        
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
        
        # Critterdrug heritage sensors (from map-retinal-sensors.py)
        # These give the neural network much more specific inputs to learn from
        self.heritage_sensors = {
            'energy_bar': np.zeros(self.ENERGY_BAR_SENSORS),  # Segmented energy display
            'age_bar': np.zeros(self.AGE_BAR_SENSORS),        # Segmented age display
            'touching_food': 0.0,      # Binary: is creature touching food?
            'touching_creature': 0.0,  # Binary: is creature touching another creature?
            'touching_pill': 0.0,      # Binary: is creature touching a drug mushroom?
            'can_procreate': 0.0,      # Binary: can creature reproduce right now?
        }
        
        # Pen-position motor system (from looser.c lines 227-246)
        # Each creature has a drawing cursor on Circuit8
        # Motor neurons control pen position; sensors read at pen position
        self.pen_x = 0.0  # Pen X position on Circuit8 (0..width)
        self.pen_y = 0.0  # Pen Y position on Circuit8 (0..height)
    
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
        
        # Critterdrug heritage sensors
        self._sense_heritage()
    
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
        
        # 5. HERITAGE SENSORS (critterdrug: energy bar, age bar, touch, procreation)
        # Energy bar: 10 segments, each lights up as energy fills
        for val in self.heritage_sensors['energy_bar']:
            if idx >= len(sensory_neurons):
                break
            sensory_neurons[idx].add_input(val * 1000.0)
            idx += 1
        
        # Age bar: 10 segments
        for val in self.heritage_sensors['age_bar']:
            if idx >= len(sensory_neurons):
                break
            sensory_neurons[idx].add_input(val * 1000.0)
            idx += 1
        
        # Touch sensors (binary, strong signal)
        touch_vals = [
            self.heritage_sensors['touching_food'] * 5000.0,
            self.heritage_sensors['touching_creature'] * 5000.0,
            self.heritage_sensors['touching_pill'] * 5000.0,
            self.heritage_sensors['can_procreate'] * 5000.0,
        ]
        for val in touch_vals:
            if idx >= len(sensory_neurons):
                break
            sensory_neurons[idx].add_input(val)
            idx += 1
        
        # 6. Fill remaining with noise (prevents dead neurons, adds stochasticity)
        while idx < len(sensory_neurons):
            noise = np.random.uniform(50.0, 250.0)
            sensory_neurons[idx].add_input(noise)
            idx += 1
    
    def _sense_vision(self):
        """Read visual environment through retinal array.
        
        Performance: self.circuit8.screen IS already [height, width, 3] uint8.
        Pass it directly — no pixel-by-pixel copy needed.
        """
        if not hasattr(self, 'circuit8') or self.circuit8 is None:
            return
        
        # Pass the screen array directly (it's already [h, w, 3] uint8)
        self.visual_input = self.retinal_array.read_screen(self.circuit8.screen)
    
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
    
    def _sense_heritage(self):
        """
        Update critterdrug heritage sensors.
        
        From map-retinal-sensors.py:
        - Energy bar: 10 segments, each lights up as energy fills (like a battery)
        - Age bar: 10 segments, fills as creature ages toward max age
        - Touch sensors: binary signals for touching food/creature/pill
        - canProcreate: binary, can this creature reproduce right now?
        
        These give the neural network very specific, dedicated inputs that
        are easier to evolve useful behaviors around than generic values.
        """
        # Energy bar: segmented battery display (10 segments)
        # Each segment lights up as energy increases
        if hasattr(self, 'energy'):
            energy_ratio = self.energy.energy / max(1.0, self.energy.max_energy)
            for i in range(self.ENERGY_BAR_SENSORS):
                # Segment i lights up when energy > i/10
                threshold = (i + 1) / self.ENERGY_BAR_SENSORS
                self.heritage_sensors['energy_bar'][i] = 1.0 if energy_ratio >= threshold else 0.0
        
        # Age bar: fills as creature approaches max age (18000 from heritage)
        max_age = 18000.0  # From critterdrug heritage
        if hasattr(self, 'age'):
            age_ratio = min(1.0, self.age / max_age)
            for i in range(self.AGE_BAR_SENSORS):
                threshold = (i + 1) / self.AGE_BAR_SENSORS
                self.heritage_sensors['age_bar'][i] = 1.0 if age_ratio >= threshold else 0.0
        
        # Touch sensors (from collision state)
        if hasattr(self, 'colliding_with_resource') and self.colliding_with_resource:
            # Check resource type
            self.heritage_sensors['touching_food'] = 1.0
        else:
            self.heritage_sensors['touching_food'] = 0.0
        
        if hasattr(self, 'colliding_with_creature') and self.colliding_with_creature:
            self.heritage_sensors['touching_creature'] = 1.0
        else:
            self.heritage_sensors['touching_creature'] = 0.0
        
        # Touching pill (drug mushroom) - detected via drug system
        if hasattr(self, 'drugs'):
            # If any drug level increased recently, we're touching a pill
            total_trip = np.sum(self.drugs.tripping)
            self.heritage_sensors['touching_pill'] = 1.0 if total_trip > 0.5 else 0.0
        
        # Can procreate sensor
        if hasattr(self, 'can_reproduce'):
            self.heritage_sensors['can_procreate'] = 1.0 if self.can_reproduce() else 0.0
    
    def update_pen_position(self):
        """
        Update pen position from motor neurons and draw on Circuit8.
        
        From looser.c lines 227-246: Each creature has a pen (x,y) cursor
        on Circuit8. Motor neurons control pen movement:
        - 2 motors: pen X movement (add/subtract from pen_x)
        - 2 motors: pen Y movement (add/subtract from pen_y)
        
        The pen draws at its position on Circuit8 while retinal sensors
        read from the pen position. This creates a more expressive
        draw-and-read feedback loop than writing at the creature's
        world position.
        
        Call AFTER network.update(), as it reads motor outputs.
        """
        if not hasattr(self, 'circuit8') or self.circuit8 is None:
            return
        if not hasattr(self, 'motor_outputs') or len(self.motor_outputs) < 8:
            return
        
        # Use motor outputs 4-7 for pen control (0-3 used for physical movement)
        # Motor 4: pen X forward, Motor 5: pen X backward
        # Motor 6: pen Y forward, Motor 7: pen Y backward
        pen_dx = (self.motor_outputs[4] - self.motor_outputs[5]) * 2.0
        pen_dy = (self.motor_outputs[6] - self.motor_outputs[7]) * 2.0
        
        self.pen_x = (self.pen_x + pen_dx) % self.circuit8.width
        self.pen_y = (self.pen_y + pen_dy) % self.circuit8.height
        
        # Draw at pen position using screen motors (condensed color ops)
        px = int(self.pen_x) % self.circuit8.width
        py = int(self.pen_y) % self.circuit8.height
        
        if hasattr(self, 'screen_motors') and len(self.screen_motors) >= 6:
            # RuRdGuGdBuBd control from screen motors
            r_change = self.screen_motors[0] - self.screen_motors[1]
            g_change = self.screen_motors[2] - self.screen_motors[3]
            b_change = self.screen_motors[4] - self.screen_motors[5]
            
            # Drug amplification
            drug_amp = 1.0
            if hasattr(self, 'drugs'):
                trip_level = np.sum(self.drugs.tripping) / max(1.0, self.drugs.max_trip)
                drug_amp = 1.0 + trip_level * 5.0
            
            scale = 15.0 * drug_amp
            
            current_r, current_g, current_b = self.circuit8.read_pixel(px, py)
            new_r = np.clip(current_r + r_change * scale, 0, 255)
            new_g = np.clip(current_g + g_change * scale, 0, 255)
            new_b = np.clip(current_b + b_change * scale, 0, 255)
            
            self.circuit8.write_pixel(px, py, int(new_r), int(new_g), int(new_b), blend=False)
    
    def read_at_pen_position(self) -> Optional[np.ndarray]:
        """
        Read Circuit8 at pen position (retinal sensors at pen location).
        
        From looser.c lines 281-292: Each finger reads a vertical strip
        of the screen at its pen position. This returns the RGB values
        at the pen position for injection into sensory neurons.
        
        Returns:
            RGB array at pen position or None
        """
        if not hasattr(self, 'circuit8') or self.circuit8 is None:
            return None
        
        px = int(self.pen_x) % self.circuit8.width
        py = int(self.pen_y) % self.circuit8.height
        return self.circuit8.screen[py, px].astype(np.float32)
    
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
