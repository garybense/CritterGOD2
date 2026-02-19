"""
Collective creature with social intelligence.

Extends PhysicsCreature with collective learning and communication:
- Broadcasts behaviors to Circuit8
- Reads signals from other creatures
- Learns from others' successes/failures
- Marks resource locations for others
- Responds to danger warnings
- Participates in collective memory

Complete social artificial life.
"""

from typing import Optional
import numpy as np
from creatures.physics_creature import PhysicsCreature
from core.collective import (
    BehaviorBroadcaster,
    BehaviorReader,
    BehaviorType,
    SignalType,
    SocialLearner,
    BehaviorObservation,
    CollectiveMemory,
)
from core.physics.physics_world import PhysicsWorld
from core.resources.resource_manager import ResourceManager
from core.evolution.genotype import Genotype
from core.morphology.body_genotype import BodyGenotype
from core.morphic.circuit8 import Circuit8
from creatures.genetic_language import GeneticLanguage
from creatures.psychedelic_mixin import PsychedelicVisionMixin
from creatures.sensory_mixin import CompleteSensoryMixin
from creatures.audio_mixin import AudioSynthesisMixin


class CollectiveCreature(PhysicsCreature, PsychedelicVisionMixin, CompleteSensoryMixin, AudioSynthesisMixin):
    """
    Creature with collective intelligence.
    
    Adds to PhysicsCreature:
    - Broadcasts behaviors to Circuit8
    - Reads and responds to signals
    - Social learning from observations
    - Resource location marking
    - Danger warning system
    - Collective memory access
    
    Attributes:
        broadcaster: Behavior broadcasting system
        reader: Signal reading system
        learner: Social learning system
        collective_memory: Shared memory (optional)
        current_behavior: Current behavior being performed
    """
    
    def __init__(
        self,
        genotype: Genotype,
        body: Optional[BodyGenotype] = None,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        initial_energy: float = 1000000.0,
        circuit8: Optional[Circuit8] = None,
        physics_world: Optional[PhysicsWorld] = None,
        collective_memory: Optional[CollectiveMemory] = None,
        creature_id: Optional[int] = None,
        **kwargs
    ):
        """
        Create collective creature.
        
        Args:
            genotype: Neural network genotype
            body: Body genotype
            x, y, z: Initial position
            initial_energy: Starting energy
            circuit8: Shared telepathic canvas
            physics_world: Physics world
            collective_memory: Shared collective memory
            creature_id: Unique creature ID
            **kwargs: Additional arguments
        """
        # Initialize physics creature
        super().__init__(
            genotype=genotype,
            body=body,
            x=x,
            y=y,
            z=z,
            initial_energy=initial_energy,
            circuit8=circuit8,
            physics_world=physics_world,
            **kwargs
        )
        
        # Generate unique ID if not provided
        if creature_id is None:
            creature_id = np.random.randint(0, 1000000)
        self.creature_id = creature_id
        
        # Collective intelligence systems
        self.broadcaster = BehaviorBroadcaster(self.creature_id)
        self.reader = BehaviorReader(detection_radius=5)
        self.learner = SocialLearner(self.creature_id, memory_size=50)
        self.collective_memory = collective_memory
        
        # Current state
        self.current_behavior = BehaviorType.IDLE
        self.previous_energy = initial_energy
        
        # Broadcasting parameters
        self.broadcast_frequency = 5  # Broadcast every N timesteps
        self.timestep_counter = 0
        
        # Social learning parameters
        self.enable_social_learning = True
        self.enable_resource_marking = True
        self.enable_danger_warnings = True
        
        # Thought generation (Markov text)
        from generators.markov.evolutionary_markov import EvolutionaryMarkov
        self.markov = EvolutionaryMarkov(
            wordpair_start_energy=1000.0,
            start_attrep=300.0,
            attrep_hit_cost=200.0,
            breed_threshold=1500.0,
            kill_threshold=0.1,
            mutation_rate=0.3,
        )
        self.last_thought = ""
        
        # Initialize with random seed text (genetic language)
        seed_text = GeneticLanguage.generate_random_seed_text()
        if seed_text:
            self.markov.add_corpus(seed_text)
        
        # Initialize psychedelic vision (pattern generation)
        # Start disabled for performance, can enable via demo toggle
        self.init_psychedelic_vision(
            enable_vision=False,  # Retinal vision (handled by CompleteSensoryMixin)
            enable_patterns=False  # Visual pattern generation (can enable with P key)
        )
        
        # Initialize COMPLETE sensory integration for maximum emergence
        # This wires: vision, proprioception, interoception, chemoreception
        self.init_complete_senses(
            enable_vision=True,      # Retinal vision ON by default
            vision_sensors=32,        # Reasonable performance
            neurons_per_sensor=8      # 32*8 = 256 visual neurons
        )
        
        # Initialize audio synthesis (voice from neural activity)
        # Start disabled for performance, can enable via demo toggle
        self.init_audio_synthesis(
            enable_audio=False,      # Audio OFF by default (toggle with A key)
            sample_rate=44100,
            mode='mixed',            # Best sonic texture
            base_amplitude=0.1       # Quiet by default
        )
    
    def determine_current_behavior(self) -> BehaviorType:
        """
        Determine current behavior based on state.
        
        Returns:
            Current behavior type
        """
        # Check hunger
        if self.behavior.should_seek_food(self.energy.energy):
            if self.target_resource is not None:
                return BehaviorType.EATING if self.colliding_with_resource else BehaviorType.SEEKING_FOOD
            return BehaviorType.SEEKING_FOOD
        
        # Check drug craving
        if self.behavior.should_seek_drug():
            return BehaviorType.SEEKING_DRUG
        
        # Check velocity (moving = exploring)
        if self.rigid_body:
            vel_mag = np.linalg.norm(self.get_velocity())
            if vel_mag > 1.0:
                return BehaviorType.EXPLORING
        
        # Check energy level
        if self.energy.energy < 300000:
            return BehaviorType.FLEEING
        
        # Default
        return BehaviorType.IDLE
    
    def broadcast_current_state(self):
        """Broadcast current behavior and signals to Circuit8."""
        if self.circuit8 is None:
            return
        
        # Determine current behavior
        self.current_behavior = self.determine_current_behavior()
        
        # Broadcast behavior
        self.broadcaster.broadcast_behavior(
            self.circuit8,
            self.current_behavior,
            self.x,
            self.y
        )
        
        # Check for significant events to broadcast
        energy_change = self.energy.energy - self.previous_energy
        
        # Success signal (gained significant energy)
        if energy_change > 10000:
            self.broadcaster.broadcast_signal(
                self.circuit8,
                SignalType.SUCCESS,
                self.x,
                self.y
            )
            
            # Record to collective memory
            if self.collective_memory:
                self.collective_memory.record_behavior_outcome(
                    self.current_behavior,
                    'success',
                    energy_change,
                    (self.x, self.y)
                )
        
        # Failure signal (lost significant energy)
        elif energy_change < -10000:
            self.broadcaster.broadcast_signal(
                self.circuit8,
                SignalType.FAILURE,
                self.x,
                self.y
            )
            
            if self.collective_memory:
                self.collective_memory.record_behavior_outcome(
                    self.current_behavior,
                    'failure',
                    energy_change,
                    (self.x, self.y)
                )
        
        # Danger warning (low energy)
        if self.enable_danger_warnings and self.energy.energy < 200000:
            self.broadcaster.broadcast_signal(
                self.circuit8,
                SignalType.DANGER_WARNING,
                self.x,
                self.y
            )
        
        self.previous_energy = self.energy.energy
    
    def mark_resource_found(self, resource):
        """
        Mark resource location on Circuit8 for other creatures.
        
        Args:
            resource: Resource that was found
        """
        if not self.enable_resource_marking or self.circuit8 is None:
            return
        
        # Determine resource type
        from core.resources.resource import ResourceType
        if resource.resource_type == ResourceType.FOOD:
            resource_type = 'food'
        elif resource.resource_type == ResourceType.DRUG_MUSHROOM:
            resource_type = 'drug'
        else:
            return
        
        # Mark on Circuit8
        self.broadcaster.mark_resource_location(
            self.circuit8,
            resource.x,
            resource.y,
            resource_type
        )
        
        # Broadcast resource found signal
        self.broadcaster.broadcast_signal(
            self.circuit8,
            SignalType.RESOURCE_FOUND,
            resource.x,
            resource.y,
            radius=5
        )
        
        # Record to collective memory
        if self.collective_memory:
            self.collective_memory.record_resource_location(
                resource_type,
                resource.x,
                resource.y,
                self.timestep_counter
            )
    
    def read_nearby_signals(self):
        """Read signals from Circuit8 and respond."""
        if self.circuit8 is None:
            return
        
        # Read signals
        signals = self.reader.read_nearby_signals(
            self.circuit8,
            self.x,
            self.y
        )
        
        # Process signals
        for signal in signals:
            signal_type = signal['type']
            
            # Learn from success/failure signals
            if signal_type in [SignalType.SUCCESS, SignalType.FAILURE]:
                # Create observation (approximate)
                outcome = 'success' if signal_type == SignalType.SUCCESS else 'failure'
                observation = BehaviorObservation(
                    creature_id=-1,  # Unknown creature
                    behavior=self.current_behavior,  # Assume similar behavior
                    outcome=outcome,
                    energy_change=signal['intensity'] * 50000,  # Estimate
                    timestep=self.timestep_counter,
                    location=(self.x, self.y)
                )
                self.learner.observe_behavior(observation)
            
            # Respond to danger warnings (avoid area)
            elif signal_type == SignalType.DANGER_WARNING:
                if self.rigid_body and np.random.random() < 0.3:
                    # Apply impulse away from danger
                    offset = signal['offset']
                    if offset[0] != 0 or offset[1] != 0:
                        away_dir = np.array([-offset[0], -offset[1], 0], dtype=np.float32)
                        away_dir = away_dir / np.linalg.norm(away_dir)
                        self.apply_impulse(away_dir * 2.0)
    
    def check_resource_markers(self):
        """Check for resource markers on Circuit8."""
        if self.circuit8 is None or not self.enable_social_learning:
            return
        
        # Only check if currently seeking resources
        if self.current_behavior not in [BehaviorType.SEEKING_FOOD, BehaviorType.SEEKING_DRUG]:
            return
        
        # Find nearby markers
        markers = self.reader.find_resource_markers(
            self.circuit8,
            self.x,
            self.y,
            search_radius=15
        )
        
        if markers:
            # Use closest marker
            closest = markers[0]
            
            # Move toward marked location (if no current target)
            if self.target_resource is None and self.rigid_body:
                dx = closest['world_x'] - self.x
                dy = closest['world_y'] - self.y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0.1:
                    # Apply force toward marker
                    force_dir = np.array([dx, dy, 0], dtype=np.float32) / dist
                    self.rigid_body.apply_force(force_dir * 2.0)
    
    def apply_social_learning(self):
        """Apply learned behaviors from observations."""
        if not self.enable_social_learning:
            return
        
        # Get recommended behavior
        recommended = self.learner.get_recommended_behavior(self.energy.energy)
        
        if recommended is not None:
            # Check if should adopt recommended behavior
            if self.learner.should_imitate_behavior(recommended):
                # Could modify target resource selection here
                # For now, just influence exploration direction
                if self.rigid_body and np.random.random() < 0.1:
                    # Random exploration influenced by learned behavior
                    random_force = np.random.uniform(-2.0, 2.0, size=3).astype(np.float32)
                    random_force[2] *= 0.1  # Much less vertical
                    self.rigid_body.apply_force(random_force)
    
    def update(self, dt: float = 1.0, resource_manager: Optional[ResourceManager] = None) -> bool:
        """
        Update collective creature.
        
        Args:
            dt: Time step
            resource_manager: Resource manager
            
        Returns:
            True if alive, False if dead
        """
        # Update physics creature
        alive = super().update(dt, resource_manager)
        if not alive:
            # Record death location as danger zone
            if self.collective_memory:
                self.collective_memory.record_danger_zone(self.x, self.y)
            return False
        
        self.timestep_counter += 1
        
        # Broadcast state periodically
        if self.timestep_counter % self.broadcast_frequency == 0:
            self.broadcast_current_state()
        
        # Read and respond to signals
        self.read_nearby_signals()
        
        # Check for resource markers
        self.check_resource_markers()
        
        # Apply social learning
        self.apply_social_learning()
        
        # Generate thoughts (every 20 timesteps)
        if self.timestep_counter % 20 == 0 and self.markov:
            thought = self.markov.generate_and_evolve(max_length=8)
            if thought:
                self.last_thought = thought
        
        # Auto-enable psychedelic pattern generation when tripping
        # This creates the shared hallucination space on Circuit8
        if hasattr(self, 'drugs'):
            trip_level = np.sum(self.drugs.tripping) / max(1.0, self.drugs.max_trip)
            if trip_level > 0.05 and not self.pattern_generation_enabled:
                self.pattern_generation_enabled = True
                if self.pattern_gen is None and hasattr(self, 'circuit8') and self.circuit8:
                    from generators.visual.pattern_generators import PatternGenerator
                    from generators.visual.neural_parameters import NeuralPatternMapper
                    self.pattern_gen = PatternGenerator(
                        width=self.circuit8.width,
                        height=self.circuit8.height
                    )
                    self.pattern_mapper = NeuralPatternMapper()
        
        # Update psychedelic vision (pattern generation to Circuit8)
        self.update_psychedelic_vision(self.timestep_counter)
        
        # Check for reproduction opportunity
        offspring = self._attempt_reproduction()
        if offspring:
            return offspring  # Return offspring to be added to population
        
        return True
    
    def _consume_resource(self, resource, resource_manager: ResourceManager):
        """
        Override to mark resource location when found.
        
        Args:
            resource: Resource to consume
            resource_manager: Resource manager
        """
        # Mark resource for others
        self.mark_resource_found(resource)
        
        # Consume normally
        super()._consume_resource(resource, resource_manager)
    
    def get_collective_state(self) -> dict:
        """
        Get collective intelligence state.
        
        Returns:
            Dictionary of collective state
        """
        return {
            'creature_id': self.creature_id,
            'current_behavior': self.current_behavior.name,
            'observations_count': len(self.learner.observation_history),
            'learned_behaviors': len(self.learner.behavior_success_rates),
            'collective_knowledge': self.learner.get_collective_knowledge_summary(),
        }
    
    def _attempt_reproduction(self):
        """
        Attempt to reproduce if conditions are met.
        
        Returns:
            Offspring creature or None
        """
        # Check if capable of reproduction
        if not self.can_reproduce():
            return None
        
        # Broadcast mating signal periodically when ready
        if self.timestep_counter % 10 == 0:
            self.broadcaster.broadcast_signal(
                self.circuit8,
                SignalType.MATING_CALL,
                self.x,
                self.y
            )
        
        # Reproduce asexually (for now - sexual reproduction requires finding mates)
        # Only reproduce if energy is high enough
        if self.energy.energy > 800000:  # 80% of max energy
            offspring = self._create_offspring()
            return offspring
        
        return None
    
    def _create_offspring(self):
        """
        Create offspring with genetic inheritance.
        
        Returns:
            New CollectiveCreature offspring
        """
        # Pay energy cost
        if not self.energy.pay_reproduction_cost():
            return None
        
        # Create mutated offspring genotype
        offspring_genotype = self.genotype.mutate(
            mutation_rate=0.5,
            max_mutations=50
        )
        
        # Mutate body
        offspring_body = self.body.mutate(mutation_rate=0.3) if self.body else None
        
        # Transfer energy to offspring  
        offspring_energy = self.energy.transfer_to_offspring(fraction=0.3)
        
        # Inherit language (genetic text)
        seed_text = None
        if self.markov and self.markov.chain.chain:
            seed_text = self.markov.generate_and_evolve(max_length=20)
        
        # Create offspring
        offspring = CollectiveCreature(
            genotype=offspring_genotype,
            body=offspring_body,
            x=self.x + np.random.uniform(-15, 15),
            y=self.y + np.random.uniform(-15, 15),
            z=self.z if hasattr(self, 'z') else 10.0,
            initial_energy=offspring_energy,
            circuit8=self.circuit8,
            physics_world=self.physics_world,
            collective_memory=self.collective_memory,
            creature_id=None  # Will get new ID
        )
        
        # Inherit lineage
        offspring.generation = self.generation + 1
        offspring.adam_distance = self.adam_distance + 1
        
        # Inherit parent's language
        if seed_text:
            offspring.markov.add_corpus(seed_text)
        
        return offspring
    
    def get_current_thought(self) -> str:
        """Get creature's most recent thought."""
        return self.last_thought
    
    def to_dict(self) -> dict:
        """Export complete creature state with collective intelligence."""
        base_dict = super().to_dict()
        
        # Add collective information
        base_dict['collective'] = self.get_collective_state()
        base_dict['current_thought'] = self.last_thought
        
        return base_dict


def create_collective_creatures(
    n_creatures: int,
    physics_world: PhysicsWorld,
    circuit8: Circuit8,
    collective_memory: CollectiveMemory,
    world_bounds: tuple = (-250.0, -250.0, 250.0, 250.0)
) -> list:
    """
    Create multiple collective creatures sharing memory.
    
    Args:
        n_creatures: Number of creatures
        physics_world: Physics world
        circuit8: Telepathic canvas
        collective_memory: Shared collective memory
        world_bounds: (min_x, min_y, max_x, max_y)
        
    Returns:
        List of collective creatures
    """
    creatures = []
    
    min_x, min_y, max_x, max_y = world_bounds
    
    for i in range(n_creatures):
        # Random position
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        z = 10.0
        
        # Random genotype
        genotype = Genotype.create_random(
            n_sensory=20,
            n_motor=20,
            n_hidden_min=50,
            n_hidden_max=150,
            synapses_per_neuron=30
        )
        
        # Random body
        body = BodyGenotype.create_random()
        
        # Create collective creature
        creature = CollectiveCreature(
            genotype=genotype,
            body=body,
            x=x,
            y=y,
            z=z,
            initial_energy=1000000.0,
            circuit8=circuit8,
            physics_world=physics_world,
            collective_memory=collective_memory,
            creature_id=i
        )
        
        creatures.append(creature)
    
    return creatures
