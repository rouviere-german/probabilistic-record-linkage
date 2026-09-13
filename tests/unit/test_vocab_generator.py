from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from probabilistic_record_linkage.linkage.generator import (
    GeneratorConfig,
    SyntheticDataset,
    generate,
    household_chains,
    load_nicknames,
    load_sources,
    load_truth,
    make_bridge_records,
    write_dataset,
)
from probabilistic_record_linkage.linkage.vocab import (
    SyllableModel,
    World,
    build_world,
    generate_words,
    zipf_mandelbrot,
)
from probabilistic_record_linkage.ml.contracts import validate_source
from probabilistic_record_linkage.ml.manifest import sha256_file

pytestmark = pytest.mark.unit


def test_zipf_mandelbrot_weights_are_normalized_and_decreasing() -> None:
    weights = zipf_mandelbrot(100, 1.0, 5.0)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(np.diff(weights) < 0)
    for args in ((0, 1.0, 0.0), (5, 0.0, 0.0), (5, 1.0, -1.0)):
        with pytest.raises(ValueError, match="must"):
            zipf_mandelbrot(*args)


def test_generated_words_are_distinct_and_exhaustion_is_reported() -> None:
    model = SyllableModel(("b",), ("a",), ("",), (1,), (1.0,))
    rng = np.random.default_rng(0)
    assert generate_words(model, 1, rng) == ["ba"]
    with pytest.raises(ValueError, match="exhausted"):
        generate_words(model, 2, rng, max_rounds=3)
    with pytest.raises(ValueError, match=">= 0"):
        generate_words(model, -1, rng)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "onsets": (),
            "nuclei": ("a",),
            "codas": ("",),
            "syllable_counts": (1,),
            "syllable_count_weights": (1.0,),
        },
        {
            "onsets": ("b",),
            "nuclei": ("a",),
            "codas": ("",),
            "syllable_counts": (1, 2),
            "syllable_count_weights": (1.0,),
        },
        {
            "onsets": ("b",),
            "nuclei": ("a",),
            "codas": ("",),
            "syllable_counts": (0,),
            "syllable_count_weights": (1.0,),
        },
    ],
)
def test_syllable_models_are_validated(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must"):
        SyllableModel(**kwargs)  # type: ignore[arg-type]


def test_world_is_deterministic_for_a_seed(world: World) -> None:
    again = build_world(world.seed)
    assert again.groups[2].surnames[:50] == world.groups[2].surnames[:50]
    assert again.gazetteer.postcodes == world.gazetteer.postcodes
    assert again.nicknames == world.nicknames
    frequencies = world.first_name_frequencies()
    assert sum(frequencies.values()) == pytest.approx(1.0)


def test_generation_is_deterministic_and_sources_satisfy_the_contract(
    small_dataset: SyntheticDataset,
) -> None:
    again = generate(small_dataset.config, small_dataset.world)
    for name, frame in small_dataset.sources.items():
        assert frame.equals(again.sources[name])
        validate_source(frame)
        assert "entity_id" not in frame.columns
    ids = [rid for frame in small_dataset.sources.values() for rid in frame["record_id"].to_list()]
    assert len(ids) == len(set(ids))
    assert sorted(small_dataset.truth["record_id"].to_list()) == sorted(ids)


def test_record_ids_do_not_reveal_entity_order(small_dataset: SyntheticDataset) -> None:
    truth = small_dataset.truth.filter(pl.col("source") == "s1")
    entities = truth["entity_id"].to_numpy()
    rank_correlation = np.corrcoef(np.arange(entities.size), np.argsort(np.argsort(entities)))[0, 1]
    assert abs(rank_correlation) < 0.2


def test_twins_and_household_members_are_hard_negatives(small_dataset: SyntheticDataset) -> None:
    pop, world = small_dataset.population, small_dataset.world
    twins = [(e, int(t)) for e, t in enumerate(pop.twin_of.tolist()) if t > e]
    assert twins
    for a, b in twins:
        assert pop.last_name(world, a) == pop.last_name(world, b)
        assert pop.dob(a) == pop.dob(b)
        assert (pop.street[a], pop.postcode[a]) == (pop.street[b], pop.postcode[b])
        assert pop.first_name(world, a) != pop.first_name(world, b)
    members = [e for e, h in enumerate(pop.household.tolist()) if h >= 0 and h != e]
    assert members
    for e in members:
        head = int(pop.household[e])
        assert pop.last_name(world, e) == pop.last_name(world, head)
        assert pop.postcode[e] == pop.postcode[head]
        assert pop.dob(e) != pop.dob(head)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sources": ("s1",), "source_inclusion": (0.5, 0.5)},
        {"sources": ("S-1",), "source_inclusion": (0.5,)},
        {"sources": ("s1", "s1"), "source_inclusion": (0.5, 0.5)},
        {"sources": ("s1",), "source_inclusion": (0.0,)},
        {"birth_years": (2000, 1990)},
    ],
)
def test_invalid_generator_configurations_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        GeneratorConfig.model_validate(kwargs)


def test_datasets_round_trip_through_disk_with_hashes(
    tmp_path: Path, small_dataset: SyntheticDataset
) -> None:
    hashes = write_dataset(small_dataset, tmp_path)
    for relative, digest in hashes.items():
        assert sha256_file(tmp_path / relative) == digest
    loaded = load_sources(tmp_path)
    assert set(loaded) == set(small_dataset.sources)
    for name, frame in loaded.items():
        assert frame.equals(small_dataset.sources[name])
    assert load_truth(tmp_path).equals(small_dataset.truth)
    assert load_nicknames(tmp_path) == small_dataset.world.nicknames
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator_sha256"] == small_dataset.config.sha256()
    assert manifest["corruption_sha256"] == small_dataset.config.corruption.sha256()
    with pytest.raises(FileNotFoundError):
        load_sources(tmp_path / "missing")


def test_bridge_records_copy_fields_from_consecutive_entities(
    small_dataset: SyntheticDataset,
) -> None:
    chains = household_chains(small_dataset, min_members=2, min_records=1)[:3]
    assert chains
    frame, truth = make_bridge_records(small_dataset, chains)
    assert frame.height == sum(len(chain) - 1 for chain in chains)
    pop, world = small_dataset.population, small_dataset.world
    left, right = chains[0][0], chains[0][1]
    first = frame.row(0, named=True)
    assert first["first_name"] == pop.first_name(world, left).capitalize()
    assert first["dob"] == pop.dob(right)
    assert set(truth["kind"].to_list()) == {"bridge"}
    assert truth["entity_id"].min() >= len(pop)  # type: ignore[operator]
    validate_source(frame)
