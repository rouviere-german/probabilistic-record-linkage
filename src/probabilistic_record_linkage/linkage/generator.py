"""Seeded synthetic population and multi-source record generator (milestone M1).

People are drawn from the synthetic world (:mod:`vocab`), grouped into twins and households that
act as hard negatives, and emitted into several sources with independent corruption
(:mod:`corruption`). True entity ids never appear in a source: sources and ground truth are
separate frames, written to separate directories, and the source contract rejects any extra
column (``tests/unit/test_contracts.py``). Record ids are assigned after shuffling each source,
so an id does not reveal which records were generated together.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from probabilistic_record_linkage.core.canonical import canonical_json, sha256_hex
from probabilistic_record_linkage.core.fileio import atomic_write_text
from probabilistic_record_linkage.linkage.corruption import (
    CorruptionConfig,
    date_typo,
    digit_typo,
    keyboard_typo,
    swap_day_month,
)
from probabilistic_record_linkage.linkage.nicknames import NicknameTable
from probabilistic_record_linkage.linkage.vocab import STREET_SUFFIXES, World, build_world
from probabilistic_record_linkage.ml.manifest import sha256_file

__all__ = [
    "KIND_NAMES",
    "SOURCE_COLUMNS",
    "SOURCE_SCHEMA",
    "TRUTH_SCHEMA",
    "GeneratorConfig",
    "Population",
    "SyntheticDataset",
    "draw_population",
    "emit_source",
    "entity_labels",
    "generate",
    "household_chains",
    "load_nicknames",
    "load_sources",
    "load_truth",
    "make_bridge_records",
    "write_dataset",
]

IntArray = npt.NDArray[np.int64]
Int32Array = npt.NDArray[np.int32]
KIND_NAMES = ("normal", "twin", "household", "bridge")
SOURCE_COLUMNS = (
    "record_id",
    "first_name",
    "last_name",
    "dob",
    "sex",
    "street",
    "city",
    "postcode",
)
SOURCE_SCHEMA: dict[str, type[pl.DataType]] = {
    "record_id": pl.Utf8,
    "first_name": pl.Utf8,
    "last_name": pl.Utf8,
    "dob": pl.Date,
    "sex": pl.Utf8,
    "street": pl.Utf8,
    "city": pl.Utf8,
    "postcode": pl.Utf8,
}
TRUTH_SCHEMA: dict[str, type[pl.DataType]] = {
    "record_id": pl.Utf8,
    "source": pl.Utf8,
    "entity_id": pl.Int64,
    "kind": pl.Utf8,
    "group": pl.Utf8,
    "household_id": pl.Int64,
    "twin_of": pl.Int64,
    "sloppy": pl.Boolean,
}
_SOURCE_NAME = re.compile(r"^[a-z][a-z0-9]{0,7}$")
_N_UNIFORMS = 18
_ENTITY_COLUMNS = (
    "group", "sex", "first", "surname", "year", "month", "day", "house", "street", "postcode",
    "alt_house", "alt_street", "alt_postcode", "kind", "household", "twin_of",
)  # fmt: skip


class GeneratorConfig(BaseModel):
    """Everything that determines a synthetic dataset. Its hash goes into every manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 20260913
    n_records: int = Field(default=200_000, ge=20, le=2_000_000)
    sources: tuple[str, ...] = ("s1", "s2", "s3", "s4", "s5")
    source_inclusion: tuple[float, ...] = (0.55, 0.50, 0.45, 0.50, 0.40)
    duplicate_rate: float = Field(default=0.03, ge=0.0, le=0.5)
    twin_rate: float = Field(default=0.01, ge=0.0, le=0.2)
    household_rate: float = Field(default=0.05, ge=0.0, le=0.5)
    birth_years: tuple[int, int] = (1936, 2006)
    upper_case_share: float = Field(default=0.05, ge=0.0, le=1.0)
    corruption: CorruptionConfig = Field(default_factory=CorruptionConfig)

    @field_validator("sources")
    @classmethod
    def _check_sources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("sources must be non-empty and unique")
        if not all(_SOURCE_NAME.fullmatch(name) for name in value):
            raise ValueError("source names must match ^[a-z][a-z0-9]{0,7}$")
        return value

    @model_validator(mode="after")
    def _check_inclusion(self) -> GeneratorConfig:
        if len(self.source_inclusion) != len(self.sources):
            raise ValueError("source_inclusion needs one probability per source")
        if not all(0.0 < p <= 1.0 for p in self.source_inclusion):
            raise ValueError("inclusion probabilities must be in (0, 1]")
        if self.birth_years[0] >= self.birth_years[1] or self.birth_years[0] < 1900:
            raise ValueError("birth_years must be an increasing range starting at 1900 or later")
        return self

    def expected_records_per_entity(self) -> float:
        p = np.asarray(self.source_inclusion, dtype=np.float64)
        p_none = float(np.prod(1.0 - p))
        return (float(p.sum()) + p_none) * (1.0 + self.duplicate_rate)

    def sha256(self) -> str:
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


@dataclass(frozen=True, slots=True)
class Population:
    """One row per true entity (column arrays). Indices point into the world's vocabularies."""

    group: npt.NDArray[np.int8]
    sex: npt.NDArray[np.int8]  # 0 female, 1 male
    first: Int32Array
    surname: Int32Array
    year: Int32Array
    month: Int32Array
    day: Int32Array
    house: Int32Array
    street: Int32Array
    postcode: Int32Array
    alt_house: Int32Array
    alt_street: Int32Array
    alt_postcode: Int32Array
    kind: npt.NDArray[np.int8]
    household: IntArray
    twin_of: IntArray

    def __len__(self) -> int:
        return int(self.group.size)

    def first_name(self, world: World, entity: int) -> str:
        vocab = world.groups[int(self.group[entity])]
        names = vocab.female_first if self.sex[entity] == 0 else vocab.male_first
        return names[int(self.first[entity])]

    def last_name(self, world: World, entity: int) -> str:
        return world.groups[int(self.group[entity])].surnames[int(self.surname[entity])]

    def dob(self, entity: int) -> date:
        return date(int(self.year[entity]), int(self.month[entity]), int(self.day[entity]))


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """Sources (no entity ids) plus the separately held ground truth."""

    config: GeneratorConfig
    world: World
    population: Population
    sources: dict[str, pl.DataFrame]
    truth: pl.DataFrame

    @property
    def n_records(self) -> int:
        return sum(frame.height for frame in self.sources.values())


def _days_in_month(year: IntArray, month: IntArray) -> IntArray:
    leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    table = np.asarray([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int64)
    result: IntArray = np.where((month == 2) & leap, 29, table[month - 1]).astype(np.int64)
    return result


def _draw_dates(
    n: int, years: tuple[int, int], rng: np.random.Generator
) -> tuple[IntArray, IntArray, IntArray]:
    year = rng.integers(years[0], years[1] + 1, size=n, dtype=np.int64)
    month = rng.integers(1, 13, size=n, dtype=np.int64)
    day = 1 + np.floor(rng.random(n) * _days_in_month(year, month)).astype(np.int64)
    return year, month, day


def _draw_address(
    world: World, n: int, rng: np.random.Generator, *, city: IntArray | None = None
) -> tuple[IntArray, IntArray, IntArray]:
    gaz = world.gazetteer
    if city is None:
        city = rng.choice(len(gaz.cities), size=n, p=gaz.city_p).astype(np.int64)
    counts = np.asarray([len(codes) for codes in gaz.postcodes_by_city], dtype=np.int64)
    starts = np.asarray([codes[0] for codes in gaz.postcodes_by_city], dtype=np.int64)
    postcode = starts[city] + np.floor(rng.random(n) * counts[city]).astype(np.int64)
    street_counts = np.asarray([len(s) for s in gaz.streets_by_postcode], dtype=np.int64)
    slot = np.floor(rng.random(n) * street_counts[postcode]).astype(np.int64)
    pairs = zip(postcode.tolist(), slot.tolist(), strict=True)
    street = np.fromiter((gaz.streets_by_postcode[p][s] for p, s in pairs), dtype=np.int64, count=n)
    house = rng.integers(1, 301, size=n, dtype=np.int64)
    return house, street, postcode


def _pick(rng: np.random.Generator, size: int, p: npt.NDArray[np.float64], avoid: int = -1) -> int:
    for _ in range(32):
        choice = int(rng.choice(size, p=p))
        if choice != avoid:
            return choice
    return (avoid + 1) % size


class _PopulationBuilder:
    """Draws base people in bulk, then appends twins and household members of earlier draws."""

    def __init__(
        self, world: World, config: GeneratorConfig, rng: np.random.Generator, n: int
    ) -> None:
        self.world = world
        self.config = config
        self.rng = rng
        self.n = n
        self.base = self._draw_base()
        self.columns: dict[str, list[int]] = {name: [] for name in _ENTITY_COLUMNS}

    def _draw_base(self) -> dict[str, IntArray]:
        world, rng, n = self.world, self.rng, self.n
        group = rng.choice(len(world.groups), size=n, p=world.group_shares).astype(np.int64)
        sex = rng.integers(0, 2, size=n, dtype=np.int64)
        first = np.zeros(n, dtype=np.int64)
        surname = np.zeros(n, dtype=np.int64)
        for g, vocab in enumerate(world.groups):
            by_sex = (
                (vocab.female_first, vocab.female_first_p),
                (vocab.male_first, vocab.male_first_p),
            )
            for sex_code, (names, probs) in enumerate(by_sex):
                mask = (group == g) & (sex == sex_code)
                first[mask] = rng.choice(len(names), size=int(mask.sum()), p=probs)
            in_group = group == g
            surname[in_group] = rng.choice(
                len(vocab.surnames), size=int(in_group.sum()), p=vocab.surname_p
            )
        year, month, day = _draw_dates(n, self.config.birth_years, rng)
        house, street, postcode = _draw_address(world, n, rng)
        alt_house, alt_street, alt_postcode = self._draw_alternative(postcode)
        return {
            "group": group,
            "sex": sex,
            "first": first,
            "surname": surname,
            "year": year,
            "month": month,
            "day": day,
            "house": house,
            "street": street,
            "postcode": postcode,
            "alt_house": alt_house,
            "alt_street": alt_street,
            "alt_postcode": alt_postcode,
            "kind": np.zeros(n, dtype=np.int64),
            "household": np.full(n, -1, dtype=np.int64),
            "twin_of": np.full(n, -1, dtype=np.int64),
        }

    def _draw_alternative(self, postcode: IntArray) -> tuple[IntArray, IntArray, IntArray]:
        """An address the person moved to or from: half in the same city, never same postcode."""
        gaz, rng, n = self.world.gazetteer, self.rng, postcode.size
        primary_city = np.asarray(gaz.postcode_city, dtype=np.int64)[postcode]
        moved_city = rng.choice(len(gaz.cities), size=n, p=gaz.city_p).astype(np.int64)
        single = np.asarray([len(c) == 1 for c in gaz.postcodes_by_city], dtype=bool)[primary_city]
        same_city = (rng.random(n) < 0.5) & ~single
        alt_city = np.where(same_city, primary_city, moved_city).astype(np.int64)
        alt_house, alt_street, alt_postcode = _draw_address(self.world, n, rng, city=alt_city)
        clash = alt_postcode == postcode
        while clash.any():
            count = int(clash.sum())
            other = rng.choice(len(gaz.cities), size=count, p=gaz.city_p).astype(np.int64)
            new_house, new_street, new_postcode = _draw_address(self.world, count, rng, city=other)
            alt_house[clash] = new_house
            alt_street[clash] = new_street
            alt_postcode[clash] = new_postcode
            clash = alt_postcode == postcode
        return alt_house, alt_street, alt_postcode

    @property
    def size(self) -> int:
        return len(self.columns["group"])

    def append(self, source: int, **overrides: int) -> int:
        for name in _ENTITY_COLUMNS:
            self.columns[name].append(overrides.get(name, int(self.base[name][source])))
        return self.size - 1

    def _first_name_index(self, head: int, sex: int) -> int:
        vocab = self.world.groups[self.columns["group"][head]]
        names, probs = (
            (vocab.female_first, vocab.female_first_p)
            if sex == 0
            else (vocab.male_first, vocab.male_first_p)
        )
        avoid = self.columns["first"][head] if sex == self.columns["sex"][head] else -1
        return _pick(self.rng, len(names), probs, avoid)

    def add_twin(self, base: int, head: int) -> None:
        sex = int(self.rng.integers(0, 2))
        first = self._first_name_index(head, sex)
        twin = self.append(base, sex=sex, first=first, kind=1, twin_of=head)
        self.columns["twin_of"][head] = twin

    def add_household(self, base: int, head: int, members: int) -> None:
        self.columns["household"][head] = head
        base_year = self.columns["year"][head]
        for member in range(members):
            if self.size >= self.n:
                break
            sex = int(self.rng.integers(0, 2))
            first = self._first_name_index(head, sex)
            if member == 0:
                offset = int(self.rng.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]))
            else:
                offset = int(self.rng.integers(20, 39))
            year = base_year + offset
            if year > self.config.birth_years[1]:
                year = base_year - offset
            year = max(year, 1900)
            month = int(self.rng.integers(1, 13))
            days = _days_in_month(
                np.asarray([year], dtype=np.int64), np.asarray([month], dtype=np.int64)
            )
            day = 1 + int(self.rng.random() * int(days[0]))
            self.append(
                base, sex=sex, first=first, year=year, month=month, day=day, kind=2, household=head
            )

    def build(self) -> Population:
        roles = self.rng.random((self.n, 3))
        base = 0
        while self.size < self.n and base < self.n:
            head = self.append(base)
            if roles[base, 0] < self.config.twin_rate and self.size < self.n:
                self.add_twin(base, head)
            if roles[base, 1] < self.config.household_rate:
                self.add_household(base, head, 1 + int(roles[base, 2] * 3))
            base += 1

        def int32(name: str) -> Int32Array:
            return np.asarray(self.columns[name][: self.n], dtype=np.int32)

        def int8(name: str) -> npt.NDArray[np.int8]:
            return np.asarray(self.columns[name][: self.n], dtype=np.int8)

        def int64(name: str) -> IntArray:
            return np.asarray(self.columns[name][: self.n], dtype=np.int64)

        return Population(
            group=int8("group"),
            sex=int8("sex"),
            first=int32("first"),
            surname=int32("surname"),
            year=int32("year"),
            month=int32("month"),
            day=int32("day"),
            house=int32("house"),
            street=int32("street"),
            postcode=int32("postcode"),
            alt_house=int32("alt_house"),
            alt_street=int32("alt_street"),
            alt_postcode=int32("alt_postcode"),
            kind=int8("kind"),
            household=int64("household"),
            twin_of=int64("twin_of"),
        )


def draw_population(
    world: World, n_entities: int, config: GeneratorConfig, rng: np.random.Generator
) -> Population:
    """Draw ``n_entities`` people, including twins and household members of earlier draws."""
    if n_entities < 1:
        raise ValueError("n_entities must be >= 1")
    return _PopulationBuilder(world, config, rng, n_entities).build()


@dataclass(frozen=True, slots=True)
class _RecordContext:
    world: World
    population: Population
    corruption: CorruptionConfig
    rng: np.random.Generator


def _display_name(value: str, upper: bool) -> str:
    return value.upper() if upper else value.capitalize()


def _first_name(
    ctx: _RecordContext, entity: int, scale: float, row: list[float], upper: bool
) -> str | None:
    c = ctx.corruption
    if row[0] < scale * c.first_name_missing:
        return None
    canonical = ctx.population.first_name(ctx.world, entity)
    options = ctx.world.nicknames.nicknames_of(canonical)
    if options and row[1] < scale * c.first_name_nickname:
        value = options[int(row[2] * len(options))]
    elif row[3] < scale * c.first_name_typo:
        value = keyboard_typo(canonical, ctx.rng, transposition_share=c.transposition_share)
    else:
        value = canonical
    return _display_name(value, upper)


def _last_name(
    ctx: _RecordContext, entity: int, scale: float, row: list[float], upper: bool
) -> str | None:
    c = ctx.corruption
    if row[4] < scale * c.last_name_missing:
        return None
    surname = ctx.population.last_name(ctx.world, entity)
    if row[5] < scale * c.last_name_typo:
        surname = keyboard_typo(surname, ctx.rng, transposition_share=c.transposition_share)
    return _display_name(surname, upper)


def _birth_date(ctx: _RecordContext, entity: int, scale: float, row: list[float]) -> date | None:
    c = ctx.corruption
    if row[6] < scale * c.dob_missing:
        return None
    born = ctx.population.dob(entity)
    if row[7] < scale * c.dob_swap:
        return swap_day_month(born)
    if row[7] < scale * (c.dob_swap + c.dob_typo):
        return date_typo(born, ctx.rng)
    return born


def _sex(ctx: _RecordContext, entity: int, scale: float, row: list[float]) -> str | None:
    c = ctx.corruption
    if row[8] < scale * c.sex_missing:
        return None
    female = bool(ctx.population.sex[entity] == 0)
    if row[9] < scale * c.sex_flip:
        female = not female
    return "F" if female else "M"


def _address(
    ctx: _RecordContext, entity: int, scale: float, row: list[float]
) -> tuple[str, str | None, str | None]:
    c, pop, gaz = ctx.corruption, ctx.population, ctx.world.gazetteer
    moved = row[10] < scale * c.address_move
    house = int(pop.alt_house[entity] if moved else pop.house[entity])
    street_index = int(pop.alt_street[entity] if moved else pop.street[entity])
    postcode_index = int(pop.alt_postcode[entity] if moved else pop.postcode[entity])
    stem = gaz.streets[street_index]
    if row[11] < scale * c.street_typo:
        stem = keyboard_typo(stem, ctx.rng, transposition_share=c.transposition_share)
    short, long = STREET_SUFFIXES[street_index % len(STREET_SUFFIXES)]
    suffix = long if row[12] < 0.5 else short
    street = f"{house} {stem.capitalize()} {suffix.capitalize()}"
    postcode: str | None = None
    if row[13] >= scale * c.postcode_missing:
        postcode = gaz.postcodes[postcode_index]
        if row[14] < scale * c.postcode_typo:
            postcode = digit_typo(postcode, ctx.rng)
    city: str | None = None
    if row[15] >= scale * c.city_missing:
        city = gaz.cities[gaz.postcode_city[postcode_index]].capitalize()
    return street, city, postcode


def emit_source(
    world: World,
    population: Population,
    entities: IntArray,
    source: str,
    *,
    corruption: CorruptionConfig,
    rng: np.random.Generator,
    upper_case_share: float = 0.05,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Emit one corrupted record per entry of ``entities``; repeats are within-source duplicates."""
    order = rng.permutation(entities.size)
    entity_of = entities[order]
    n = int(entity_of.size)
    mixture = corruption.multipliers()
    if len(mixture) == 2:
        sloppy = rng.random(n) < mixture[0][0]
        scale = np.where(sloppy, mixture[0][1], mixture[1][1])
    else:
        sloppy = np.zeros(n, dtype=bool)
        scale = np.ones(n)
    uniforms = rng.random((n, _N_UNIFORMS))
    ctx = _RecordContext(world, population, corruption, rng)
    columns: dict[str, list[object]] = {name: [] for name in SOURCE_COLUMNS[1:]}
    for r in range(n):
        entity = int(entity_of[r])
        factor = float(scale[r])
        row: list[float] = uniforms[r].tolist()
        upper = row[17] < upper_case_share
        columns["first_name"].append(_first_name(ctx, entity, factor, row, upper))
        columns["last_name"].append(_last_name(ctx, entity, factor, row, upper))
        columns["dob"].append(_birth_date(ctx, entity, factor, row))
        columns["sex"].append(_sex(ctx, entity, factor, row))
        street, city, postcode = _address(ctx, entity, factor, row)
        columns["street"].append(street)
        columns["city"].append(city)
        columns["postcode"].append(postcode)

    record_ids = [f"{source}-{i:07d}" for i in range(n)]
    frame = pl.DataFrame({"record_id": record_ids, **columns}, schema=SOURCE_SCHEMA)
    group_names = world.group_names
    truth = pl.DataFrame(
        {
            "record_id": record_ids,
            "source": [source] * n,
            "entity_id": entity_of.astype(np.int64),
            "kind": [KIND_NAMES[int(k)] for k in population.kind[entity_of]],
            "group": [group_names[int(g)] for g in population.group[entity_of]],
            "household_id": population.household[entity_of],
            "twin_of": population.twin_of[entity_of],
            "sloppy": sloppy,
        },
        schema=TRUTH_SCHEMA,
    )
    return frame, truth


def generate(config: GeneratorConfig, world: World | None = None) -> SyntheticDataset:
    """Generate all sources and the ground truth for ``config`` (deterministic)."""
    world = world if world is not None else build_world(config.seed)
    rng = np.random.default_rng([config.seed, 1])
    n_entities = max(2, round(config.n_records / config.expected_records_per_entity()))
    population = draw_population(world, n_entities, config, rng)
    p = np.asarray(config.source_inclusion, dtype=np.float64)
    included = rng.random((n_entities, p.size)) < p
    empty = ~included.any(axis=1)
    forced = rng.choice(p.size, size=int(empty.sum()), p=p / p.sum())
    included[np.flatnonzero(empty), forced] = True
    duplicated = included & (rng.random((n_entities, p.size)) < config.duplicate_rate)
    counts = included.astype(np.int64) + duplicated.astype(np.int64)
    sources: dict[str, pl.DataFrame] = {}
    truths: list[pl.DataFrame] = []
    entity_ids = np.arange(n_entities, dtype=np.int64)
    for s, name in enumerate(config.sources):
        frame, truth = emit_source(
            world,
            population,
            np.repeat(entity_ids, counts[:, s]),
            name,
            corruption=config.corruption,
            rng=rng,
            upper_case_share=config.upper_case_share,
        )
        sources[name] = frame
        truths.append(truth)
    return SyntheticDataset(config, world, population, sources, pl.concat(truths))


def household_chains(
    dataset: SyntheticDataset, *, min_members: int = 3, min_records: int = 2
) -> list[list[int]]:
    """Households with at least ``min_members`` people who each have ``min_records`` records."""
    records = dataset.truth.group_by("entity_id").len()
    per_entity = dict(zip(records["entity_id"].to_list(), records["len"].to_list(), strict=True))
    households: dict[int, list[int]] = {}
    for entity, household in enumerate(dataset.population.household.tolist()):
        if household >= 0:
            households.setdefault(int(household), []).append(entity)
    return [
        sorted(members)
        for _, members in sorted(households.items())
        if len(members) >= min_members and all(per_entity.get(m, 0) >= min_records for m in members)
    ]


def make_bridge_records(
    dataset: SyntheticDataset, chains: Sequence[Sequence[int]], *, source: str = "adv"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Adversarial bridge records that link consecutive entities of each chain.

    Bridge ``t`` copies the first name, surname and sex of entity ``t`` and the date of birth and
    primary address of entity ``t + 1``. Between household members (shared surname and address)
    such a record scores as a match against both, so without the over-merge guard a chain of
    ``L`` people collapses into one cluster of ``2L - 1`` true identities.
    """
    world, pop = dataset.world, dataset.population
    gaz = world.gazetteer
    next_entity = len(pop)
    rows: dict[str, list[object]] = {name: [] for name in SOURCE_COLUMNS}
    truth_rows: dict[str, list[object]] = {name: [] for name in TRUTH_SCHEMA}
    index = 0
    for chain in chains:
        for left, right in pairwise(chain):
            record_id = f"{source}-{index:07d}"
            street_index = int(pop.street[right])
            short, _ = STREET_SUFFIXES[street_index % len(STREET_SUFFIXES)]
            stem = gaz.streets[street_index].capitalize()
            postcode_index = int(pop.postcode[right])
            rows["record_id"].append(record_id)
            rows["first_name"].append(pop.first_name(world, left).capitalize())
            rows["last_name"].append(pop.last_name(world, left).capitalize())
            rows["dob"].append(pop.dob(right))
            rows["sex"].append("F" if pop.sex[left] == 0 else "M")
            rows["street"].append(f"{int(pop.house[right])} {stem} {short.capitalize()}")
            rows["city"].append(gaz.cities[gaz.postcode_city[postcode_index]].capitalize())
            rows["postcode"].append(gaz.postcodes[postcode_index])
            values = (record_id, source, next_entity, "bridge", "", -1, -1, False)
            for name, value in zip(TRUTH_SCHEMA, values, strict=True):
                truth_rows[name].append(value)
            truth_rows["group"][-1] = world.group_names[int(pop.group[left])]
            next_entity += 1
            index += 1
    return pl.DataFrame(rows, schema=SOURCE_SCHEMA), pl.DataFrame(truth_rows, schema=TRUTH_SCHEMA)


def write_dataset(dataset: SyntheticDataset, root: Path) -> dict[str, str]:
    """Write sources, truth (separate directory), nickname reference data and a hash manifest."""
    for directory in ("sources", "truth", "reference"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, frame in sorted(dataset.sources.items()):
        path = root / "sources" / f"{name}.parquet"
        frame.write_parquet(path, statistics=False)
        hashes[f"sources/{name}.parquet"] = sha256_file(path)
    truth_path = root / "truth" / "truth.parquet"
    dataset.truth.write_parquet(truth_path, statistics=False)
    hashes["truth/truth.parquet"] = sha256_file(truth_path)
    nick_path = root / "reference" / "nicknames.json"
    atomic_write_text(nick_path, dataset.world.nicknames.to_json() + "\n")
    hashes["reference/nicknames.json"] = sha256_file(nick_path)
    manifest = {
        "generator": dataset.config.model_dump(mode="json"),
        "generator_sha256": dataset.config.sha256(),
        "corruption_sha256": dataset.config.corruption.sha256(),
        "files": hashes,
        "n_records": dataset.n_records,
        "n_entities": len(dataset.population),
    }
    atomic_write_text(root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return hashes


def load_sources(root: Path) -> dict[str, pl.DataFrame]:
    """Read ``root/sources/*.parquet`` only; ground truth lives elsewhere and is never read here."""
    paths = sorted((root / "sources").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no sources under {root / 'sources'}")
    return {path.stem: pl.read_parquet(path) for path in paths}


def load_truth(root: Path) -> pl.DataFrame:
    return pl.read_parquet(root / "truth" / "truth.parquet")


def load_nicknames(root: Path) -> NicknameTable:
    text = (root / "reference" / "nicknames.json").read_text(encoding="utf-8")
    return NicknameTable.from_json(text)


def entity_labels(truth: pl.DataFrame, record_ids: Sequence[str]) -> IntArray:
    """True entity id per record id (in the given order); raises on unknown ids."""
    lookup: Mapping[str, int] = dict(
        zip(truth["record_id"].to_list(), truth["entity_id"].to_list(), strict=True)
    )
    return np.fromiter((lookup[r] for r in record_ids), dtype=np.int64, count=len(record_ids))
