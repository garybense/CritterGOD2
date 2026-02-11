"""
Text mutation operators for evolutionary Markov chains.

Based on text-fuzz.py from the critters codebase.
Mutations operate on letters and words to create genetic variation.
"""

import random
import string
from typing import List


# Vowels and consonants for targeted mutations
VOWELS = 'aeiouAEIOU'
CONSONANTS = 'bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ'


def is_vowel(char: str) -> bool:
    """Check if character is a vowel."""
    return char in VOWELS


def is_consonant(char: str) -> bool:
    """Check if character is a consonant."""
    return char in CONSONANTS


def mutate_vowel(word: str) -> str:
    """
    Substitute a random vowel with another random vowel.
    
    Args:
        word: Word to mutate
        
    Returns:
        Mutated word (or original if no vowels)
    """
    if not word:
        return word
    
    vowel_positions = [i for i, c in enumerate(word) if is_vowel(c)]
    if not vowel_positions:
        return word
    
    # Pick random vowel position
    pos = random.choice(vowel_positions)
    
    # Replace with different random vowel
    is_upper = word[pos].isupper()
    new_vowel = random.choice(VOWELS.lower())
    if is_upper:
        new_vowel = new_vowel.upper()
    
    return word[:pos] + new_vowel + word[pos+1:]


def mutate_consonant(word: str) -> str:
    """
    Substitute a random consonant with another random consonant.
    
    Args:
        word: Word to mutate
        
    Returns:
        Mutated word (or original if no consonants)
    """
    if not word:
        return word
    
    consonant_positions = [i for i, c in enumerate(word) if is_consonant(c)]
    if not consonant_positions:
        return word
    
    # Pick random consonant position
    pos = random.choice(consonant_positions)
    
    # Replace with different random consonant
    is_upper = word[pos].isupper()
    new_consonant = random.choice(CONSONANTS.lower())
    if is_upper:
        new_consonant = new_consonant.upper()
    
    return word[:pos] + new_consonant + word[pos+1:]


def increment_letter(word: str, pos: int = -1) -> str:
    """
    Increment a letter alphabetically (a→b, b→c, ..., z→a).
    
    Args:
        word: Word to mutate
        pos: Position to increment (-1 = random position)
        
    Returns:
        Mutated word
    """
    if not word:
        return word
    
    if pos == -1:
        pos = random.randint(0, len(word) - 1)
    
    if pos < 0 or pos >= len(word):
        return word
    
    char = word[pos]
    if not char.isalpha():
        return word
    
    # Increment with wraparound
    if char == 'z':
        new_char = 'a'
    elif char == 'Z':
        new_char = 'A'
    else:
        new_char = chr(ord(char) + 1)
    
    return word[:pos] + new_char + word[pos+1:]


def decrement_letter(word: str, pos: int = -1) -> str:
    """
    Decrement a letter alphabetically (b→a, c→b, ..., a→z).
    
    Args:
        word: Word to mutate
        pos: Position to decrement (-1 = random position)
        
    Returns:
        Mutated word
    """
    if not word:
        return word
    
    if pos == -1:
        pos = random.randint(0, len(word) - 1)
    
    if pos < 0 or pos >= len(word):
        return word
    
    char = word[pos]
    if not char.isalpha():
        return word
    
    # Decrement with wraparound
    if char == 'a':
        new_char = 'z'
    elif char == 'A':
        new_char = 'Z'
    else:
        new_char = chr(ord(char) - 1)
    
    return word[:pos] + new_char + word[pos+1:]


def transpose_letters(word: str) -> str:
    """
    Swap two adjacent letters (dyslexic mutation).
    
    Args:
        word: Word to mutate
        
    Returns:
        Mutated word (or original if too short)
    """
    if len(word) < 2:
        return word
    
    # Pick random position to transpose
    pos = random.randint(0, len(word) - 2)
    
    # Swap adjacent letters
    chars = list(word)
    chars[pos], chars[pos+1] = chars[pos+1], chars[pos]
    
    return ''.join(chars)


def inject_random_char(word: str) -> str:
    """
    Insert a random character at a random position.
    
    Args:
        word: Word to mutate
        
    Returns:
        Mutated word with extra character
    """
    if not word:
        return word
    
    pos = random.randint(0, len(word))
    random_char = random.choice(string.ascii_lowercase)
    
    return word[:pos] + random_char + word[pos:]


def delete_random_char(word: str) -> str:
    """
    Remove a random character.
    
    Args:
        word: Word to mutate
        
    Returns:
        Mutated word with one less character (or original if too short)
    """
    if len(word) <= 1:
        return word
    
    pos = random.randint(0, len(word) - 1)
    return word[:pos] + word[pos+1:]


def mutate_word(word: str, mutation_rate: float = 1.0) -> str:
    """
    Apply random mutation to a word.
    
    Mutation types from text-fuzz.py:
    - Vowel substitution
    - Consonant substitution
    - Letter increment/decrement
    - Transposition
    - Random character injection
    - Random character deletion
    
    Args:
        word: Word to mutate
        mutation_rate: Probability of mutation (0.0-1.0)
        
    Returns:
        Mutated word (or original if no mutation occurs)
    """
    if random.random() > mutation_rate:
        return word
    
    # Pick random mutation type
    mutations = [
        mutate_vowel,
        mutate_consonant,
        increment_letter,
        decrement_letter,
        transpose_letters,
        inject_random_char,
        delete_random_char,
    ]
    
    mutation_func = random.choice(mutations)
    return mutation_func(word)


def mutate_line(line: str, mutation_rate: float = 0.3) -> str:
    """
    Apply mutations to words in a line of text.
    
    Heritage parameter: 0.3 = 30% of words mutated.
    
    Args:
        line: Line of text to mutate
        mutation_rate: Probability each word mutates (0.3 from heritage)
        
    Returns:
        Mutated line
    """
    words = line.split()
    mutated_words = [mutate_word(w, mutation_rate) for w in words]
    return ' '.join(mutated_words)
