"""
Phenotype: Convert genotype to actual neural network.

Genotype → Phenotype conversion (genetic encoding → working brain).
"""

from typing import Dict
from core.neural import Neuron, Synapse, NeuralNetwork
from core.neural.neuron import NeuronType
from .genotype import Genotype, NeuronGene, SynapseGene


def genotype_to_phenotype(genotype: Genotype) -> NeuralNetwork:
    """
    Convert a genotype into a working neural network (phenotype).
    
    This is an alias for build_network_from_genotype.
    """
    return build_network_from_genotype(genotype)


def build_network_from_genotype(genotype: Genotype) -> NeuralNetwork:
    """
    Convert a genotype into a working neural network (phenotype).
    
    This is where the genetic encoding becomes a living, functioning brain.
    
    Args:
        genotype: Genetic encoding
        
    Returns:
        Functional neural network
    """
    network = NeuralNetwork(enable_plasticity=True)
    
    # Create neuron ID to Neuron object mapping
    neuron_map: Dict[int, Neuron] = {}
    
    # Build all neurons from genes
    for gene in genotype.neuron_genes:
        # Determine neuron type
        if gene.is_sensory:
            neuron_type = NeuronType.SENSORY
        elif gene.is_motor:
            neuron_type = NeuronType.MOTOR
        elif gene.is_inhibitory:
            neuron_type = NeuronType.INHIBITORY
        else:
            neuron_type = NeuronType.REGULAR
            
        # Create neuron
        neuron = Neuron(
            neuron_id=gene.neuron_id,
            threshold=gene.threshold,
            neuron_type=neuron_type,
            is_plastic=gene.is_plastic,
            leak_rate=gene.leak_rate
        )
        
        network.add_neuron(neuron)
        neuron_map[gene.neuron_id] = neuron
        
    # Build all synapses from genes
    for gene in genotype.synapse_genes:
        # Get pre and post neurons
        if gene.pre_neuron_id not in neuron_map or gene.post_neuron_id not in neuron_map:
            continue  # Skip if neurons don't exist
            
        pre_neuron = neuron_map[gene.pre_neuron_id]
        post_neuron = neuron_map[gene.post_neuron_id]
        
        # Create synapse
        synapse = Synapse(
            pre_neuron=pre_neuron,
            post_neuron=post_neuron,
            weight=gene.weight,
            is_inhibitory=gene.is_inhibitory,
            plasticity_rate=gene.plasticity_rate
        )
        
        network.add_synapse(synapse)
        
    return network
