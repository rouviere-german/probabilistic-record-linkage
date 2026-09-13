"""Synthetic name, surname, street and city vocabularies built from Zipfian syllable models.

No external name lists are used. Each synthetic *name-origin group* has its own syllable inventory
(onsets, nuclei, codas) and syllable-count distribution, so groups differ in name length and in
how concentrated their surnames are. Those differences are what the per-group error rates in
MODEL_CARD.md measure: short names with few distinct surnames collide more often.

Syllable parts are drawn with Zipf weights (a few parts are very common), and every finished
vocabulary is given Zipf-Mandelbrot frequencies ``p_k ∝ 1 / (k + q)^s`` over its rank ``k``
(Mandelbrot 1953). The offset ``q`` flattens the head so the most common name is a few percent of
the population rather than a third of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from probabilistic_record_linkage.linkage.nicknames import NicknameTable, derive_nicknames

__all__ = [
    "STREET_SUFFIXES",
    "Gazetteer",
    "NameVocabulary",
    "SyllableModel",
    "World",
    "build_world",
    "generate_words",
    "zipf_mandelbrot",
]

FloatArray = npt.NDArray[np.float64]

# (short form, long form); sources print either, normalization maps both to the short form.
STREET_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("st", "street"),
    ("ave", "avenue"),
    ("rd", "road"),
    ("ln", "lane"),
    ("way", "way"),
    ("ct", "court"),
    ("pl", "place"),
    ("dr", "drive"),
)


def zipf_mandelbrot(n: int, exponent: float, offset: float = 0.0) -> FloatArray:
    """Normalized Zipf-Mandelbrot weights over ranks 1..n."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    if offset < 0:
        raise ValueError("offset must be >= 0")
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = 1.0 / (ranks + offset) ** exponent
    normalized: FloatArray = weights / weights.sum()
    return normalized


@dataclass(frozen=True, slots=True)
class SyllableModel:
    """A syllable inventory with Zipf-weighted parts."""

    onsets: tuple[str, ...]
    nuclei: tuple[str, ...]
    codas: tuple[str, ...]
    syllable_counts: tuple[int, ...]
    syllable_count_weights: tuple[float, ...]
    part_exponent: float = 1.05

    def __post_init__(self) -> None:
        if not (self.onsets and self.nuclei and self.codas):
            raise ValueError("onsets, nuclei and codas must be non-empty")
        if len(self.syllable_counts) != len(self.syllable_count_weights):
            raise ValueError("syllable_counts and weights must have the same length")
        if min(self.syllable_counts) < 1:
            raise ValueError("syllable counts must be >= 1")


def generate_words(
    model: SyllableModel,
    n: int,
    rng: np.random.Generator,
    *,
    exclude: set[str] | None = None,
    min_length: int = 2,
    max_length: int = 14,
    suffixes: Sequence[str] = (),
    max_rounds: int = 400,
) -> list[str]:
    """Draw ``n`` distinct words (lowercase ASCII) from ``model``, skipping ``exclude``.

    Words are generated in vectorized batches: one ``rng.choice`` per part type per batch, then
    joined in Python. Raises ``ValueError`` when the inventory cannot produce ``n`` distinct words.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    seen = set(exclude or ())
    words: list[str] = []
    max_syllables = max(model.syllable_counts)
    count_p = np.asarray(model.syllable_count_weights, dtype=np.float64)
    count_p = count_p / count_p.sum()
    onset_p = zipf_mandelbrot(len(model.onsets), model.part_exponent, 1.0)
    nucleus_p = zipf_mandelbrot(len(model.nuclei), model.part_exponent, 1.0)
    coda_p = zipf_mandelbrot(len(model.codas), model.part_exponent, 1.0)
    for _ in range(max_rounds):
        if len(words) >= n:
            break
        batch = max(256, 2 * (n - len(words)))
        counts = rng.choice(np.asarray(model.syllable_counts), size=batch, p=count_p)
        onsets = rng.choice(len(model.onsets), size=(batch, max_syllables), p=onset_p)
        nuclei = rng.choice(len(model.nuclei), size=(batch, max_syllables), p=nucleus_p)
        codas = rng.choice(len(model.codas), size=(batch, max_syllables), p=coda_p)
        suffix_idx = rng.integers(0, max(1, len(suffixes)), size=batch)
        for b in range(batch):
            parts = [
                model.onsets[onsets[b, k]] + model.nuclei[nuclei[b, k]] + model.codas[codas[b, k]]
                for k in range(int(counts[b]))
            ]
            word = "".join(parts) + (suffixes[suffix_idx[b]] if suffixes else "")
            if min_length <= len(word) <= max_length and word not in seen:
                seen.add(word)
                words.append(word)
                if len(words) == n:
                    break
    if len(words) < n:
        raise ValueError(f"syllable model exhausted after {len(words)} of {n} distinct words")
    return words


@dataclass(frozen=True, slots=True)
class NameVocabulary:
    """First names (split by sex) and surnames of one name-origin group, with frequencies."""

    group: str
    female_first: tuple[str, ...]
    female_first_p: FloatArray
    male_first: tuple[str, ...]
    male_first_p: FloatArray
    surnames: tuple[str, ...]
    surname_p: FloatArray


@dataclass(frozen=True, slots=True)
class Gazetteer:
    """Cities with Zipf sizes, postcodes per city and street names per postcode."""

    cities: tuple[str, ...]
    city_p: FloatArray
    postcode_city: tuple[int, ...]  # postcode index -> city index
    postcodes: tuple[str, ...]
    postcodes_by_city: tuple[tuple[int, ...], ...]
    streets: tuple[str, ...]  # street stems, e.g. "brandor"
    streets_by_postcode: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class World:
    """Everything the generator draws people from. Deterministic given the seed."""

    seed: int
    groups: tuple[NameVocabulary, ...]
    group_shares: FloatArray
    gazetteer: Gazetteer
    nicknames: NicknameTable

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(group.group for group in self.groups)

    def first_name_frequencies(self) -> dict[str, float]:
        """Population share of every first name: the "public name table" an attacker may know."""
        frequencies: dict[str, float] = {}
        for share, vocab in zip(self.group_shares.tolist(), self.groups, strict=True):
            for names, probs in (
                (vocab.female_first, vocab.female_first_p),
                (vocab.male_first, vocab.male_first_p),
            ):
                for name, prob in zip(names, probs.tolist(), strict=True):
                    frequencies[name] = frequencies.get(name, 0.0) + share * 0.5 * prob
        return frequencies


_GROUP_MODELS: dict[str, SyllableModel] = {
    # Consonant clusters, closed syllables, mostly two syllables.
    "north": SyllableModel(
        onsets=(
            "b",
            "d",
            "k",
            "m",
            "st",
            "h",
            "br",
            "g",
            "l",
            "r",
            "t",
            "w",
            "f",
            "gr",
            "dr",
            "s",
            "th",
            "v",
            "kr",
            "n",
            "p",
            "",
        ),
        nuclei=("a", "e", "o", "i", "u", "ae", "ei", "oo"),
        codas=("", "n", "r", "l", "s", "nd", "rt", "ck", "m", "st", "ld", ""),
        syllable_counts=(1, 2, 3),
        syllable_count_weights=(0.15, 0.6, 0.25),
    ),
    # Open syllables and longer names.
    "south": SyllableModel(
        onsets=("l", "m", "r", "s", "t", "b", "c", "d", "f", "g", "n", "p", "v", "z", ""),
        nuclei=("a", "o", "e", "i", "u", "ia", "io", "ea"),
        codas=("", "", "", "n", "l", "r", "s"),
        syllable_counts=(2, 3, 4),
        syllable_count_weights=(0.35, 0.5, 0.15),
    ),
    # Short names from a small inventory: many near-collisions.
    "east": SyllableModel(
        onsets=(
            "l",
            "w",
            "zh",
            "ch",
            "h",
            "j",
            "sh",
            "y",
            "m",
            "t",
            "b",
            "d",
            "k",
            "n",
            "p",
            "s",
            "z",
        ),
        nuclei=("a", "i", "u", "e", "o", "ai", "ao", "ei", "ou", "ia"),
        codas=("", "", "n", "ng"),
        syllable_counts=(1, 2),
        syllable_count_weights=(0.45, 0.55),
    ),
}

# (first names per sex, first-name offset, surnames, surname exponent, surname offset)
_GROUP_SIZES: dict[str, tuple[int, float, int, float, float]] = {
    "north": (1200, 12.0, 9000, 1.0, 60.0),
    "south": (800, 10.0, 5000, 1.0, 40.0),
    "east": (400, 6.0, 600, 1.0, 3.0),
}
_GROUP_SHARES = {"north": 0.6, "south": 0.25, "east": 0.15}


def build_world(seed: int, *, n_cities: int = 250) -> World:
    """Build the synthetic world (vocabularies, gazetteer, nickname table) for ``seed``."""
    rng = np.random.default_rng([seed, 0x5EED])
    used: set[str] = set()
    groups: list[NameVocabulary] = []
    all_first: list[str] = []
    for name, model in _GROUP_MODELS.items():
        n_first, first_offset, n_surnames, surname_exp, surname_offset = _GROUP_SIZES[name]
        first = generate_words(model, 2 * n_first, rng, exclude=used, min_length=3, max_length=11)
        used.update(first)
        surnames = generate_words(model, n_surnames, rng, exclude=used, min_length=3)
        used.update(surnames)
        all_first.extend(first)
        first_p = zipf_mandelbrot(n_first, 1.0, first_offset)
        groups.append(
            NameVocabulary(
                group=name,
                female_first=tuple(first[0::2]),
                female_first_p=first_p,
                male_first=tuple(first[1::2]),
                male_first_p=first_p.copy(),
                surnames=tuple(surnames),
                surname_p=zipf_mandelbrot(n_surnames, surname_exp, surname_offset),
            )
        )
    shares = np.asarray([_GROUP_SHARES[g.group] for g in groups], dtype=np.float64)

    city_model = _GROUP_MODELS["north"]
    cities = generate_words(
        city_model,
        n_cities,
        rng,
        exclude=used,
        min_length=4,
        max_length=16,
        suffixes=("ton", "ford", "burg", "field", "dale", "mouth", "by"),
    )
    used.update(cities)
    city_p = zipf_mandelbrot(n_cities, 1.1, 2.0)
    postcodes_per_city = np.clip(np.rint(city_p * 900.0), 1, 60).astype(np.int64)
    n_postcodes = int(postcodes_per_city.sum())
    codes = rng.choice(np.arange(10_000, 100_000), size=n_postcodes, replace=False)
    postcodes = tuple(f"{code:05d}" for code in codes)
    postcode_city: list[int] = []
    by_city: list[tuple[int, ...]] = []
    cursor = 0
    for city_index, count in enumerate(postcodes_per_city.tolist()):
        members = tuple(range(cursor, cursor + count))
        by_city.append(members)
        postcode_city.extend([city_index] * count)
        cursor += count

    street_stems = generate_words(
        _GROUP_MODELS["south"], 4000, rng, exclude=used, min_length=4, max_length=12
    )
    streets_by_postcode = tuple(
        tuple(int(s) for s in rng.choice(len(street_stems), size=int(k), replace=False))
        for k in rng.integers(6, 16, size=n_postcodes)
    )
    gazetteer = Gazetteer(
        cities=tuple(cities),
        city_p=city_p,
        postcode_city=tuple(postcode_city),
        postcodes=postcodes,
        postcodes_by_city=tuple(by_city),
        streets=tuple(street_stems),
        streets_by_postcode=streets_by_postcode,
    )
    nicknames = derive_nicknames(all_first, rng, forbidden=used)
    return World(
        seed=seed,
        groups=tuple(groups),
        group_shares=shares / shares.sum(),
        gazetteer=gazetteer,
        nicknames=nicknames,
    )
