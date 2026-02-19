"""
Fibonacci text reordering from flamoot's numerolit.py.

Assigns each word a base-36 numerical value (sum of letter values),
reduces to a single digit, then reorders words using fibonacci-chaining
where the next tier = (val1 + val2) % base.

This creates hidden mathematical structure in text — words with related
numerical properties cluster together, revealing patterns invisible to
normal reading.

Four methods from the original:
1. Sequential pools: all words of tier 0, then tier 1, etc.
2. Rotating selection: first word from each tier in round-robin
3. Fibonacci chaining: next tier determined by sum of previous two values
4. Fibonacci multiply: like 3 but using product instead of sum

From flamoot's numerolit.py (2010-2015)
"""

from typing import List, Optional
import random


# Base-36 character to value mapping
CHAR_VALUES = {}
for i in range(10):
    CHAR_VALUES[str(i)] = i
for i, c in enumerate('abcdefghijklmnopqrstuvwxyz'):
    CHAR_VALUES[c] = 10 + i

# Value to character mapping
VALUE_CHARS = {v: k for k, v in CHAR_VALUES.items()}


def word_to_digit(word: str, base: int = 36) -> str:
    """
    Reduce a word to a single base-N digit.
    
    Sum all character values (base-36), then iteratively sum digits
    until we get a single digit.
    
    From numerolit.py: the core numerological operation.
    
    Args:
        word: Input word (will be lowercased, non-alphanumeric stripped)
        base: Number base for reduction (default 36)
        
    Returns:
        Single character representing the word's numerical value
    """
    # Clean word
    word = word.lower()
    word = ''.join(c for c in word if c in CHAR_VALUES)
    
    if not word:
        return '0'
    
    # Sum character values
    total = sum(CHAR_VALUES.get(c, 0) for c in word)
    
    # Iteratively reduce to single digit in given base
    while total >= base:
        digits = []
        while total > 0:
            digits.append(total % base)
            total //= base
        total = sum(digits)
    
    return VALUE_CHARS.get(total, '0')


def fibonacci_reorder(words: List[str], method: int = 3, base: int = 36,
                      scramble: bool = False) -> List[str]:
    """
    Reorder words using fibonacci-chaining on numerical values.
    
    From flamoot's numerolit.py — the main entry point.
    
    Args:
        words: List of words to reorder
        method: Reordering method (1-4)
        base: Number base for digit reduction
        scramble: Whether to shuffle within tiers
        
    Returns:
        Reordered list of words
    """
    if not words or len(words) < 2:
        return words
    
    # Compute digit value for each word
    digits = [word_to_digit(w, base) for w in words]
    
    # Build tier pools: digit -> [(position, word), ...]
    pools = {}
    for i, (word, digit) in enumerate(zip(words, digits)):
        if digit not in pools:
            pools[digit] = []
        pools[digit].append((i, word))
    
    # Track consumption position per tier
    position = {d: 0 for d in pools}
    
    if method == 1:
        return _method_sequential(pools, position, base, scramble)
    elif method == 2:
        return _method_rotating(pools, position, base, scramble)
    elif method == 3:
        return _method_fibonacci_sum(pools, position, words, digits, base, scramble)
    elif method == 4:
        return _method_fibonacci_mult(pools, position, words, digits, base, scramble)
    else:
        return _method_fibonacci_sum(pools, position, words, digits, base, scramble)


def _method_sequential(pools, position, base, scramble) -> List[str]:
    """Method 1: Play all words of tier 0, then tier 1, etc."""
    result = []
    for val in range(base):
        digit = VALUE_CHARS.get(val, str(val))
        if digit in pools:
            tier_words = [w for _, w in pools[digit]]
            if scramble:
                random.shuffle(tier_words)
            result.extend(tier_words)
    return result


def _method_rotating(pools, position, base, scramble) -> List[str]:
    """Method 2: Take first word from each tier, then second, etc."""
    result = []
    max_len = max(len(v) for v in pools.values()) if pools else 0
    
    for round_idx in range(max_len):
        round_words = []
        for val in range(base):
            digit = VALUE_CHARS.get(val, str(val))
            if digit in pools and round_idx < len(pools[digit]):
                round_words.append(pools[digit][round_idx][1])
        if scramble:
            random.shuffle(round_words)
        result.extend(round_words)
    return result


def _method_fibonacci_sum(pools, position, words, digits, base, scramble) -> List[str]:
    """
    Method 3: Fibonacci value chaining with sum.
    
    Start with first two words. Next tier = (val1 + val2) % base.
    Pull the next available word from that tier. Repeat.
    If a tier is exhausted, try next tier.
    """
    if len(words) < 2:
        return list(words)
    
    result = [words[0]]
    
    val1 = CHAR_VALUES.get(digits[0], 0)
    val2 = CHAR_VALUES.get(digits[1], 0)
    
    # Mark first words as consumed
    consumed = {0}
    
    for _ in range(len(words) - 1):
        next_val = (val1 + val2) % base
        
        # Try to find an unconsumed word in this tier
        word_found = False
        for tries in range(base):
            target_digit = VALUE_CHARS.get((next_val + tries) % base, '0')
            if target_digit in pools:
                for idx, (orig_pos, word) in enumerate(pools[target_digit]):
                    if orig_pos not in consumed:
                        result.append(word)
                        consumed.add(orig_pos)
                        val1 = val2
                        val2 = CHAR_VALUES.get(target_digit, 0)
                        word_found = True
                        break
            if word_found:
                break
        
        if not word_found:
            break
    
    # Append any remaining unconsumed words
    for i, word in enumerate(words):
        if i not in consumed:
            result.append(word)
    
    return result


def _method_fibonacci_mult(pools, position, words, digits, base, scramble) -> List[str]:
    """
    Method 4: Fibonacci value chaining with multiplication.
    
    Same as method 3 but next tier = (val1 * val2) % base.
    Zeroes are treated as ones (from numerolit.py).
    """
    if len(words) < 2:
        return list(words)
    
    result = [words[0]]
    
    val1 = max(1, CHAR_VALUES.get(digits[0], 1))
    val2 = max(1, CHAR_VALUES.get(digits[1], 1))
    
    consumed = {0}
    
    for _ in range(len(words) - 1):
        next_val = (val1 * val2) % base
        
        word_found = False
        for tries in range(base):
            target_digit = VALUE_CHARS.get((next_val + tries) % base, '0')
            if target_digit in pools:
                for idx, (orig_pos, word) in enumerate(pools[target_digit]):
                    if orig_pos not in consumed:
                        result.append(word)
                        consumed.add(orig_pos)
                        val1 = val2
                        val2 = max(1, CHAR_VALUES.get(target_digit, 1))
                        word_found = True
                        break
            if word_found:
                break
        
        if not word_found:
            break
    
    for i, word in enumerate(words):
        if i not in consumed:
            result.append(word)
    
    return result


def reorder_text(text: str, method: int = 3, base: int = 36) -> str:
    """
    Convenience function: reorder a text string.
    
    Args:
        text: Input text
        method: Reordering method (1-4)
        base: Number base
        
    Returns:
        Reordered text
    """
    words = text.split()
    if not words:
        return text
    reordered = fibonacci_reorder(words, method=method, base=base)
    return ' '.join(reordered)
