"""MinHash signatures and LSH banding in NumPy (Broder 1997; Indyk and Motwani 1998).

Each shingle string is hashed once with keyed BLAKE2b to 64 bits; every permutation ``i`` is then
a SplitMix64 finalizer applied to ``hash XOR seed_i`` (Steele, Lea and Flood 2014), computed for
the whole shingle vocabulary at once. A record's signature entry is the minimum over its shingles,
evaluated with ``np.minimum.reduceat`` in chunks so memory stays bounded. For two sets the
probability that one signature entry agrees approximates their Jaccard similarity, and with
``b`` bands of ``r`` entries a pair becomes a candidate with probability ``1 - (1 - J^r)^b``.

Signatures do not depend on which other records are present, because shingle hashes are derived
from the shingle text rather than from its position in a vocabulary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "EMPTY",
    "MinHasher",
    "band_keys",
    "candidate_probability",
    "jaccard",
    "shingle_sets",
]

UInt32Array = npt.NDArray[np.uint32]
UInt64Array = npt.NDArray[np.uint64]
Int64Array = npt.NDArray[np.int64]

EMPTY = np.uint32(0xFFFFFFFF)  # signature value of an empty shingle set
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def _splitmix(values: UInt64Array) -> UInt64Array:
    z = values.copy()
    z ^= z >> np.uint64(30)
    z *= _MIX1
    z ^= z >> np.uint64(27)
    z *= _MIX2
    z ^= z >> np.uint64(31)
    return z


def shingle_sets(
    strings: Sequence[str], q: int = 2, *, pad: str = "#"
) -> tuple[Int64Array, Int64Array, tuple[str, ...]]:
    """CSR representation of each string's distinct q-grams: ``(indptr, indices, vocabulary)``.

    Strings are padded with ``pad`` so leading and trailing characters produce their own q-grams.
    Empty strings get an empty set.
    """
    if q < 1:
        raise ValueError("q must be >= 1")
    cache: dict[str, tuple[str, ...]] = {}
    per_string: list[tuple[str, ...]] = []
    for text in strings:
        grams = cache.get(text)
        if grams is None:
            if text:
                padded = pad + text + pad
                grams = tuple(sorted({padded[i : i + q] for i in range(len(padded) - q + 1)}))
            else:
                grams = ()
            cache[text] = grams
        per_string.append(grams)
    vocabulary = tuple(sorted({gram for grams in cache.values() for gram in grams}))
    index = {gram: i for i, gram in enumerate(vocabulary)}
    lengths = np.fromiter((len(g) for g in per_string), dtype=np.int64, count=len(per_string))
    indptr = np.zeros(len(per_string) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    indices = np.fromiter(
        (index[gram] for grams in per_string for gram in grams),
        dtype=np.int64,
        count=int(indptr[-1]),
    )
    return indptr, indices, vocabulary


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def candidate_probability(similarity: float, bands: int, rows: int) -> float:
    """Probability that LSH with ``bands`` x ``rows`` proposes a pair of Jaccard ``similarity``."""
    return float(1.0 - (1.0 - similarity**rows) ** bands)


class MinHasher:
    """``num_perm`` independent min-wise hash functions derived from ``seed``."""

    def __init__(self, num_perm: int = 120, seed: int = 7) -> None:
        if num_perm < 1:
            raise ValueError("num_perm must be >= 1")
        self.num_perm = num_perm
        self.seed = seed
        rng = np.random.default_rng([seed, 0xC0FFEE])
        self._salts = rng.integers(0, np.iinfo(np.uint64).max, size=num_perm, dtype=np.uint64)
        self._key = hashlib.blake2b(f"minhash:{seed}".encode(), digest_size=32).digest()

    def shingle_hashes(self, vocabulary: Sequence[str]) -> UInt64Array:
        return np.fromiter(
            (
                int.from_bytes(
                    hashlib.blake2b(gram.encode("utf-8"), digest_size=8, key=self._key).digest(),
                    "little",
                )
                for gram in vocabulary
            ),
            dtype=np.uint64,
            count=len(vocabulary),
        )

    def permutation_table(self, vocabulary: Sequence[str]) -> UInt32Array:
        """``(num_perm, len(vocabulary))`` 32-bit hash values; ``EMPTY`` is never produced."""
        base = self.shingle_hashes(vocabulary)
        with np.errstate(over="ignore"):
            mixed = _splitmix(base[None, :] ^ (self._salts[:, None] * _GOLDEN))
        table = (mixed & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        table[table == EMPTY] = EMPTY - np.uint32(1)
        return table

    def signatures(
        self,
        indptr: Int64Array,
        indices: Int64Array,
        vocabulary: Sequence[str],
        *,
        chunk_rows: int = 8192,
    ) -> UInt32Array:
        """``(n, num_perm)`` signatures; rows with no shingles are all ``EMPTY``."""
        n = indptr.size - 1
        signatures = np.full((n, self.num_perm), EMPTY, dtype=np.uint32)
        if n == 0 or not vocabulary:
            return signatures
        table = self.permutation_table(vocabulary)
        lengths = np.diff(indptr)
        for start in range(0, n, chunk_rows):
            stop = min(n, start + chunk_rows)
            rows = np.flatnonzero(lengths[start:stop]) + start
            if rows.size == 0:
                continue
            low, high = int(indptr[rows[0]]), int(indptr[rows[-1] + 1])
            values = table[:, indices[low:high]]
            offsets = indptr[rows] - low
            signatures[rows] = np.minimum.reduceat(values, offsets, axis=1).T
        return signatures


def band_keys(signatures: UInt32Array, bands: int) -> Iterator[tuple[int, UInt64Array]]:
    """Yield ``(band, key per record)``; records with an empty signature get no valid key.

    Callers must drop rows whose signature is ``EMPTY`` (see :func:`blocking.lsh_pairs`).
    """
    n, num_perm = signatures.shape
    if bands < 1 or num_perm % bands:
        raise ValueError("num_perm must be a positive multiple of bands")
    rows = num_perm // bands
    for band in range(bands):
        block = signatures[:, band * rows : (band + 1) * rows].astype(np.uint64)
        key = np.full(n, np.uint64(band + 1), dtype=np.uint64)
        with np.errstate(over="ignore"):
            key *= _GOLDEN
            for column in range(rows):
                key ^= block[:, column]
                key = _splitmix(key)
        yield band, key
