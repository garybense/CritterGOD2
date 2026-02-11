"""
Morphic Field System

How neurons connect to Circuit8 channels.
Based on telepathic-critterdrug's RuRdGuGdBuBd system.
"""

from enum import IntEnum


class MorphicChannel(IntEnum):
    """
    Morphic field channels that neurons can tune into.
    
    From telepathic-critterdrug:
    Each neuron has a RuRdGuGdBuBd value (0-5) determining which
    morphic channel affects it.
    
    Channels:
    - 0: Red Up (warm)
    - 1: Red Down (warm inverted)
    - 2: Green Up
    - 3: Green Down
    - 4: Blue Up (cool inverted)
    - 5: Blue Down (cool)
    """
    RED_UP = 0
    RED_DOWN = 1
    GREEN_UP = 2
    GREEN_DOWN = 3
    BLUE_UP = 4
    BLUE_DOWN = 5
    
    @classmethod
    def get_influence(cls, channel: int, r: int, g: int, b: int) -> float:
        """
        Calculate morphic field influence on a neuron.
        
        Neurons read from Circuit8 BEFORE processing synapses.
        This creates the feedback loop: creature thought → screen → other creatures.
        
        Args:
            channel: Which morphic channel (0-5)
            r: Red value from Circuit8
            g: Green value from Circuit8
            b: Blue value from Circuit8
            
        Returns:
            Influence value (affects neuron potential)
        """
        if channel == cls.RED_UP:
            return float(r) * 10.0  # Warm channel
        elif channel == cls.RED_DOWN:
            return float(255 - r) * 10.0  # Warm inverted
        elif channel == cls.GREEN_UP:
            return float(g) * 10.0
        elif channel == cls.GREEN_DOWN:
            return float(255 - g) * 10.0
        elif channel == cls.BLUE_UP:
            return float(255 - b) * 10.0  # Cool inverted
        elif channel == cls.BLUE_DOWN:
            return float(b) * 10.0  # Cool channel
        else:
            return 0.0
