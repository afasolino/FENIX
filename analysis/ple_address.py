"""Independent reference implementation of Qwen PLE row addressing.

This module intentionally imports neither vLLM nor SGLang. It is the
independent oracle used to verify physical PLE row IDs emitted by the runtime.
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
PLE_LAYER_PRIME = 10007

_MILLER_RABIN_BASES_64 = (
    2,
    325,
    9375,
    28178,
    450775,
    9780504,
    1795265022,
)
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def splitmix64(value: int) -> int:
    """Return the SplitMix64 transform used by the upstream PLE implementation."""

    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def is_prime_64(value: int) -> bool:
    """Deterministically test primality for an unsigned 64-bit integer."""

    if value < 2:
        return False

    for prime in _SMALL_PRIMES:
        if value % prime == 0:
            return value == prime

    remainder = value - 1
    power_of_two = 0
    while remainder % 2 == 0:
        remainder //= 2
        power_of_two += 1

    for base in _MILLER_RABIN_BASES_64:
        if base % value == 0:
            continue
        witness = pow(base, remainder, value)
        if witness in (1, value - 1):
            continue

        for _ in range(power_of_two - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False

    return True


def nth_prime_after(start: int, count: int) -> int:
    """Return the ``count``-th prime strictly greater than ``start``."""

    if count < 0:
        raise ValueError("count cannot be negative")

    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue

        if candidate % 2 == 0:
            candidate += 1
        while not is_prime_64(candidate):
            candidate += 2
        prime = candidate

    return prime


def layer_multipliers(
    ngram_size: int,
    unigram_vocab_size: int,
    seed: int,
    ple_dense_layer_id: int,
) -> list[int]:
    """Return deterministic per-position PLE hash multipliers."""

    max_multiplier = ((1 << 63) - 1) // unigram_vocab_size
    half_bound = max(1, max_multiplier // 2)
    base_seed = seed + PLE_LAYER_PRIME * ple_dense_layer_id

    return [
        2
        * (
            splitmix64(base_seed + SPLITMIX_GAMMA * (index + 1))
            % half_bound
        )
        + 1
        for index in range(ngram_size)
    ]


def vocab_layout(
    base_size: int,
    heads: int,
    ple_dense_layer_id: int,
) -> tuple[list[int], list[int], int]:
    """Return per-head table sizes, offsets, and total concatenated rows."""

    sizes: list[int] = []
    offsets: list[int] = []
    offset = 0

    for local_head in range(heads):
        global_head = ple_dense_layer_id * heads + local_head
        size = nth_prime_after(base_size - 1, global_head + 1)
        sizes.append(size)
        offsets.append(offset)
        offset += size

    return sizes, offsets, offset


def rows_for_history(
    history: list[int],
    *,
    ngram_size: int = 3,
    heads_per_ngram: int = 8,
    ngram_vocab_size_base: int = 20_000_000,
    unigram_vocab_size: int = 248_320,
    seed: int = 1234,
    ple_dense_layer_id: int = 0,
    eos_token_id: int = 248_044,
) -> list[int]:
    """Compute all physical PLE rows addressed for the newest token in history."""

    if ngram_size < 2:
        raise ValueError("ngram_size must be at least 2")
    if heads_per_ngram <= 0:
        raise ValueError("heads_per_ngram must be positive")

    total_heads = (ngram_size - 1) * heads_per_ngram
    sizes, offsets, _ = vocab_layout(
        ngram_vocab_size_base,
        total_heads,
        ple_dense_layer_id,
    )
    multipliers = layer_multipliers(
        ngram_size,
        unigram_vocab_size,
        seed,
        ple_dense_layer_id,
    )

    shifted = [
        history[-1 - index] if len(history) > index else eos_token_id
        for index in range(ngram_size)
    ]

    rows: list[int] = []
    for order in range(2, ngram_size + 1):
        mixed = shifted[0] * multipliers[0]
        for index in range(1, order):
            mixed ^= shifted[index] * multipliers[index]

        first_head = (order - 2) * heads_per_ngram
        for head in range(first_head, first_head + heads_per_ngram):
            rows.append((mixed % sizes[head]) + offsets[head])

    return rows
