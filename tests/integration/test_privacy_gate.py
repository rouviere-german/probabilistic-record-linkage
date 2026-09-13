"""The frequency-alignment attack gates the privacy-mode hardening defaults."""

from __future__ import annotations

import pytest

from probabilistic_record_linkage.linkage.attack import frequency_alignment_attack
from probabilistic_record_linkage.linkage.generator import GeneratorConfig, generate
from probabilistic_record_linkage.linkage.normalize import normalize_sources
from probabilistic_record_linkage.linkage.privacy import PrivacyConfig, encode_records
from probabilistic_record_linkage.linkage.vocab import World

pytestmark = pytest.mark.integration


def test_default_hardening_defeats_the_frequency_attack(world: World) -> None:
    dataset = generate(GeneratorConfig(seed=world.seed, n_records=20_000), world)
    normalized = normalize_sources(dataset.sources)
    public = world.first_name_frequencies()
    years = range(dataset.config.birth_years[0], dataset.config.birth_years[1] + 1)
    salts = {str(year): 1.0 / len(years) for year in years}
    defaults = PrivacyConfig().hardening
    assert defaults.salt_names_with_birth_year
    assert defaults.xor_fold
    assert defaults.balance
    worst: dict[str, float] = {}
    for label, config in (("plain", PrivacyConfig.plain()), ("hardened", PrivacyConfig())):
        encoded = encode_records(normalized, b"gate-key-" + b"0" * 24, config)
        reports = frequency_alignment_attack(
            encoded.fields["first_name"],
            encoded.present["first_name"],
            normalized["first_name"].to_list(),
            public,
            bits=encoded.bits["first_name"],
            hashes=config.name_hashes,
            q=config.q,
            top_k=100,
            salt_frequencies=salts,
        )
        worst[label] = max(report.rate for report in reports)
    assert worst["plain"] >= 0.08
    assert worst["hardened"] <= 0.05
