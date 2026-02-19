"""
Neural audio synthesis demonstration.

Creates a small neural network and generates audio from its activity
in real-time using pygame's audio system.

Based on SDL neural visualizers (looser.c, xesu.c, cdd.c).
"""

import numpy as np
import pygame

from core.neural import NeuralNetwork, Neuron
from generators.audio import NeuralAudioSynthesizer


def create_demo_network(n_neurons: int = 100) -> NeuralNetwork:
    """Create a small self-exciting neural network."""
    network = NeuralNetwork()

    # Add neurons with LOW thresholds so chain reactions can sustain
    for i in range(n_neurons):
        threshold = 150 + np.random.uniform(0, 350)  # 150-500 range (much lower)
        neuron = Neuron(
            neuron_id=i,
            threshold=threshold,
            leak_rate=0.92,  # 8% leak - faster decay creates rhythm
        )
        neuron.potential = np.random.uniform(0, 100)  # Start low
        network.add_neuron(neuron)

    # Create DENSE connections for self-sustaining activity
    network.create_random_synapses(
        synapses_per_neuron=60,  # More connections
        inhibitory_prob=0.15,   # Fewer inhibitory = more activity
    )

    return network


def audio_callback(synth, network, userdata, stream):
    """
    Pygame audio callback - generates audio from neural activity.
    
    This gets called by pygame's audio system whenever it needs more samples.
    """
    # Generate audio from current network state
    samples = synth.synthesize_from_network(
        network,
        duration_seconds=len(stream) / synth.sample_rate,
    )

    # Convert float32 (-1 to 1) to int16 for pygame
    samples_int16 = (samples * 32767).astype(np.int16)

    # Copy to stream buffer
    stream[:] = samples_int16.tobytes()


def main():
    """Run audio synthesis demo."""
    print("CritterGOD Neural Audio Synthesis Demo")
    print("=" * 50)
    print()
    print("Generating audio from neural network activity...")
    print("Press SPACE to inject random energy")
    print("Press 1-3 to change synthesis mode")
    print("Press Q to quit")
    print()

    # Initialize pygame
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=4096)

    # Create screen (small window for control)
    screen = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Neural Audio Synthesis")

    # Create network and synthesizer
    network = create_demo_network(n_neurons=150)
    synth = NeuralAudioSynthesizer(
        sample_rate=44100,
        buffer_size=4096,
        mode='mixed',
        amplitude_scale=0.2,
    )

    # Give strong initial energy to ensure neurons fire immediately
    for neuron in network.neurons:
        neuron.add_input(np.random.uniform(1000, 3000))  # All neurons get energy

    # Main loop
    clock = pygame.time.Clock()
    running = True
    frame_count = 0
    energy_injection_rate = 5  # Inject energy frequently to maintain activity
    
    # Keep references to recent sounds to prevent garbage collection
    sound_queue = []

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Inject MASSIVE energy burst - affects most of the network
                    injected = 0
                    for neuron in network.neurons:
                        if np.random.random() < 0.7:  # 70% of neurons
                            neuron.add_input(np.random.uniform(3000, 8000))
                            injected += 1
                    print(f"  [BURST! Energy injected into {injected} neurons]")
                elif event.key == pygame.K_1:
                    synth.mode = 'potential'
                    print("  Mode: POTENTIAL (smooth drone)")
                elif event.key == pygame.K_2:
                    synth.mode = 'firing'
                    print("  Mode: FIRING (percussive)")
                elif event.key == pygame.K_3:
                    synth.mode = 'mixed'
                    print("  Mode: MIXED (drone + percussion)")

        # Continuous energy injection to sustain activity
        if frame_count % energy_injection_rate == 0:
            # Inject energy into multiple neurons (simulates continuous sensory input)
            for _ in range(10):  # More neurons
                neuron = np.random.choice(network.neurons)
                neuron.add_input(np.random.uniform(200, 600))  # More energy

        # Update network
        network.update(dt=1.0)

        # Generate and play audio
        audio_samples = synth.synthesize_from_network(network, duration_seconds=0.02)

        # Convert to pygame Sound and play (stereo format required)
        audio_int16 = (audio_samples * 32767).astype(np.int16)
        stereo_audio = np.column_stack((audio_int16, audio_int16))
        sound = pygame.sndarray.make_sound(stereo_audio)
        sound.play()
        
        # Keep reference to prevent garbage collection, limit queue size
        sound_queue.append(sound)
        if len(sound_queue) > 16:
            sound_queue.pop(0)

        # Display network stats
        if frame_count % 30 == 0:
            # Use fired_last_step() - did_fire() is cleared after update()
            firing_count = sum(1 for n in network.neurons if n.fired_last_step())
            total_potential = sum(n.potential for n in network.neurons)
            avg_potential = total_potential / len(network.neurons)

            print(f"Frame {frame_count:5d} | "
                  f"Firing: {firing_count:3d}/{len(network.neurons)} | "
                  f"Avg potential: {avg_potential:7.1f} | "
                  f"Mode: {synth.mode}")

        # Clear screen and draw simple visualization
        screen.fill((0, 0, 0))

        # Draw firing neurons as dots (use fired_last_step for correct timing)
        for i, neuron in enumerate(network.neurons):
            if neuron.fired_last_step():
                x = (i % 20) * 20 + 10
                y = (i // 20) * 20 + 10
                pygame.draw.circle(screen, (255, 255, 0), (x, y), 3)

        # Draw info text (with font fallback for Python 3.14 compatibility)
        try:
            font = pygame.font.Font(None, 24)
            texts = [
                f"Neurons: {len(network.neurons)}",
                f"Mode: {synth.mode}",
                "SPACE: inject energy",
                "1-3: change mode",
                "Q: quit",
            ]
            for i, text in enumerate(texts):
                surface = font.render(text, True, (200, 200, 200))
                screen.blit(surface, (10, 200 + i * 20))
        except (NotImplementedError, ImportError):
            pass  # Font not available on Python 3.14

        pygame.display.flip()

        frame_count += 1
        clock.tick(50)  # 50 FPS

    pygame.quit()
    print()
    print("Demo complete!")


if __name__ == "__main__":
    main()
