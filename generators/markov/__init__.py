"""
Evolutionary Markov chain text generation.

Based on markov-attract-repel.py from the critters codebase.
Implements self-organizing text through attract/repel dynamics.
"""

from generators.markov.markov_chain import MarkovChain
from generators.markov.word_pair_score import WordPairScore
from generators.markov.evolutionary_markov import EvolutionaryMarkov
from generators.markov import mutations

__all__ = ['MarkovChain', 'WordPairScore', 'EvolutionaryMarkov', 'mutations']
