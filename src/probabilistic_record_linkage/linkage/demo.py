"""Offline end-to-end demo (``uv run python tasks.py demo``).

1. Generate a seeded multi-source dataset (sources and ground truth in separate directories).
2. Link it through the service (DuckDB store, audited review queue) and score the result against
   the ground truth: pair completeness, reduction ratio, pairwise and B-cubed F1, EM recovery,
   twin and household false-merge rates.
3. Register the model with its reproducibility manifest and check readiness.
4. Find a twin pair that the ``possible_twin`` review rule routed to review, and resolve it with
   the CLI (``review decide``); verify the audit chain.
5. Privacy mode: keyed Bloom-filter linkage with plain and hardened encodings, and the
   frequency-alignment attack against both.
6. Drift: comparison-pattern PSI of a new source with the same corruption versus a source with
   three times the corruption rates.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import JsonValue

from probabilistic_record_linkage.linkage.analytic import analytic_exact_agreement
from probabilistic_record_linkage.linkage.attack import frequency_alignment_attack
from probabilistic_record_linkage.linkage.audit import verify_file
from probabilistic_record_linkage.linkage.blocking import pair_completeness, reduction_ratio
from probabilistic_record_linkage.linkage.evaluation import (
    bcubed_scores,
    merge_rate,
    pairwise_scores,
)
from probabilistic_record_linkage.linkage.generator import (
    GeneratorConfig,
    SyntheticDataset,
    emit_source,
    entity_labels,
    generate,
    load_nicknames,
    load_sources,
    load_truth,
    write_dataset,
)
from probabilistic_record_linkage.linkage.model import load_model, register_run
from probabilistic_record_linkage.linkage.normalize import normalize_sources
from probabilistic_record_linkage.linkage.pipeline import LinkageRun, link_encoded, link_sources
from probabilistic_record_linkage.linkage.privacy import (
    PrivacyConfig,
    encode_records,
    load_key_file,
)
from probabilistic_record_linkage.linkage.service import default_audit_path
from probabilistic_record_linkage.ml.registry import ModelRegistry

__all__ = ["entity_pairs", "run_demo", "write_key_file"]


def write_key_file(path: Path) -> None:
    """Create a random Bloom-filter key file readable only by its owner (demo only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(secrets.token_hex(32).encode("ascii"))


def entity_pairs(dataset: SyntheticDataset) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(twin entity pairs, household member pairs) from the generator's population."""
    twins = [(e, int(t)) for e, t in enumerate(dataset.population.twin_of.tolist()) if t > e]
    households: dict[int, list[int]] = {}
    for entity, household in enumerate(dataset.population.household.tolist()):
        if household >= 0:
            households.setdefault(int(household), []).append(entity)
    members = [
        (a, b) for group in households.values() for i, a in enumerate(group) for b in group[i + 1 :]
    ]
    return twins, members


def _evaluate(run: LinkageRun, dataset: SyntheticDataset, truth: np.ndarray) -> dict[str, Any]:
    blocking = run.blocking
    pairwise = pairwise_scores(run.resolution.labels, truth)
    _, _, b3_f1 = bcubed_scores(run.resolution.labels, truth)
    analytic = analytic_exact_agreement(dataset)
    recovery = [
        abs(float(run.model.params.m[f][spec.exact_level]) - analytic[spec.name])
        for f, spec in enumerate(run.comparator.space.fields)
    ]
    twins, households = entity_pairs(dataset)
    return {
        "blocking": {
            "candidate_pairs": blocking.n_pairs,
            "pair_completeness": round(pair_completeness(blocking.left, blocking.right, truth), 4),
            "reduction_ratio": round(reduction_ratio(blocking.n_pairs, run.n_records), 6),
        },
        "linkage": {
            "pairwise": pairwise.as_dict(),
            "bcubed_f1": round(b3_f1, 4),
            "em_max_abs_m_error": round(max(recovery), 4),
            "twin_merge_rate": round(merge_rate(run.resolution.labels, truth, twins)[0], 4),
            "household_merge_rate": round(
                merge_rate(run.resolution.labels, truth, households)[0], 4
            ),
            "review_rule_hits": dict(run.rule_hits),
            "pending_records": int(run.resolution.pending.sum()),
        },
    }


def _link_register_and_find_twin(
    dataset: SyntheticDataset, data_dir: Path, state_dir: Path, summary: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    from probabilistic_record_linkage.api.app import workdir_service  # noqa: PLC0415

    sources = load_sources(data_dir)
    truth_frame = load_truth(data_dir)
    service = workdir_service(state_dir)
    try:
        job_id = service.submit("demo", sources, wait=True)
        run = service.run(job_id)
        summary.update(_evaluate(run, dataset, entity_labels(truth_frame, list(run.record_ids))))
        registry = service.registry
        if registry is None:
            raise RuntimeError("the demo service has no model registry")
        manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        version = register_run(
            registry,
            run,
            name=service.model_name,
            generator_manifest={
                key: manifest[key] for key in ("generator_sha256", "corruption_sha256", "files")
            },
            lockfile=Path("uv.lock"),
            datasets=sorted((data_dir / "sources").glob("*.parquet")),
        )
        registry.set_alias(service.model_name, "production", version.digest)
        summary["model_version"] = version.digest
        summary["run_hash"] = run.run_hash()
        summary["readiness"] = {
            name: bool(check()) for name, check in service.readiness_checks().items()
        }
        entity_of = dict(
            zip(truth_frame["record_id"].to_list(), truth_frame["entity_id"].to_list(), strict=True)
        )
        twin_of = dataset.population.twin_of
        for item in service.review_queue(job_id, include_pii=False, limit=500):
            left_entity = entity_of[item["left_record_id"]]
            right_entity = entity_of[item["right_record_id"]]
            if item["reason"] == "rule_possible_twin" and twin_of[left_entity] == right_entity:
                return job_id, item
        return job_id, None
    finally:
        service.shutdown()
        service.store.close()


def _resolve_with_cli(state_dir: Path, job_id: str, item: dict[str, Any]) -> dict[str, Any]:
    from probabilistic_record_linkage import cli  # noqa: PLC0415 - the demo drives the CLI
    from probabilistic_record_linkage.api.app import workdir_service  # noqa: PLC0415

    exit_code = cli.main(
        [
            "review", "decide", "--workdir", str(state_dir), "--job", job_id,
            "--item", str(item["item_id"]), "--decision", "non_match",
            "--reviewer", "demo-reviewer",
        ]
    )  # fmt: skip
    service = workdir_service(state_dir)
    try:
        left = service.entity(job_id, str(item["left_record_id"]), include_pii=False)
        right = service.entity(job_id, str(item["right_record_id"]), include_pii=False)
    finally:
        service.shutdown()
        service.store.close()
    return {
        "cli_exit_code": exit_code,
        "same_entity_after_decision": left["entity_id"] == right["entity_id"],
    }


def _privacy_report(
    dataset: SyntheticDataset, data_dir: Path, key_path: Path, plaintext_f1: float
) -> dict[str, Any]:
    write_key_file(key_path)
    key = load_key_file(key_path)
    normalized = normalize_sources(load_sources(data_dir))
    truth = entity_labels(load_truth(data_dir), normalized["record_id"].to_list())
    public_names = dataset.world.first_name_frequencies()
    years = range(dataset.config.birth_years[0], dataset.config.birth_years[1] + 1)
    salt_frequencies = {str(year): 1.0 / len(years) for year in years}
    report: dict[str, Any] = {}
    for label, config in (("plain_bloom", PrivacyConfig.plain()), ("hardened", PrivacyConfig())):
        encoded = encode_records(normalized, key, config)
        private_run = link_encoded(encoded, normalized["source"].to_list())
        scores = pairwise_scores(private_run.resolution.labels, truth)
        attacks = frequency_alignment_attack(
            encoded.fields["first_name"],
            encoded.present["first_name"],
            normalized["first_name"].to_list(),
            public_names,
            bits=encoded.bits["first_name"],
            hashes=config.name_hashes,
            q=config.q,
            top_k=100,
            salt_frequencies=salt_frequencies,
        )
        report[label] = {
            "f1": round(scores.f1, 4),
            "f1_drop_pp": round(100 * (plaintext_f1 - scores.f1), 2),
            "attack": [attack.as_dict() for attack in attacks],
            "worst_attack_rate": max(attack.rate for attack in attacks),
        }
    return report


def _drift_report(
    dataset: SyntheticDataset, data_dir: Path, state_dir: Path, seed: int
) -> dict[str, Any]:
    sources = load_sources(data_dir)
    nicknames = load_nicknames(data_dir)
    model, _ = load_model(ModelRegistry(state_dir / "registry"), "person-linkage")
    monitor = model.drift_monitor()
    rng = np.random.default_rng([seed, 99])
    report: dict[str, Any] = {}
    corruption = dataset.config.corruption
    for label, scenario in (
        ("same_corruption", corruption),
        ("shifted_corruption", corruption.scaled(3.0)),
    ):
        size = max(2, dataset.n_records // 5)
        entities = rng.choice(len(dataset.population), size=size, replace=False).astype(np.int64)
        frame, _ = emit_source(
            dataset.world, dataset.population, entities, "new", corruption=scenario, rng=rng
        )
        new_run = link_sources(
            {**sources, "new": frame},
            nicknames=nicknames,
            config=model.pipeline,
            fitted=model.fitted(),
            thresholds=model.thresholds,
        )
        report[label] = monitor.check(new_run.drift_patterns(involving="new")).as_dict()
    return report


def _all_checks_pass(summary: dict[str, Any]) -> bool:
    privacy, drift = summary["privacy"], summary["drift"]
    resolution = summary.get("twin_resolution") or {}
    return bool(
        summary["linkage"]["pairwise"]["f1"] >= 0.9
        and summary["audit"]["ok"]
        and all(summary["readiness"].values())
        and resolution.get("cli_exit_code") == 0
        and resolution.get("same_entity_after_decision") is False
        and privacy["hardened"]["worst_attack_rate"] < privacy["plain_bloom"]["worst_attack_rate"]
        and drift["shifted_corruption"]["drifted"]
        and not drift["same_corruption"]["drifted"]
    )


def run_demo(workdir: Path, *, records: int = 20_000, seed: int = 20260913) -> dict[str, JsonValue]:
    for child in ("data", "state", "secrets"):
        shutil.rmtree(workdir / child, ignore_errors=True)
    data_dir, state_dir = workdir / "data", workdir / "state"
    dataset = generate(GeneratorConfig(seed=seed, n_records=records))
    write_dataset(dataset, data_dir)
    (state_dir / "reference").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        data_dir / "reference" / "nicknames.json", state_dir / "reference" / "nicknames.json"
    )
    summary: dict[str, Any] = {"records": dataset.n_records, "entities": len(dataset.population)}
    job_id, twin_item = _link_register_and_find_twin(dataset, data_dir, state_dir, summary)
    summary["twin_review_item"] = twin_item
    if twin_item is not None:
        summary["twin_resolution"] = _resolve_with_cli(state_dir, job_id, twin_item)
    verification = verify_file(default_audit_path(state_dir))
    summary["audit"] = {
        "ok": verification.ok,
        "entries": verification.entries,
        "head": verification.head,
    }
    plaintext_f1 = float(summary["linkage"]["pairwise"]["f1"])
    summary["privacy"] = _privacy_report(
        dataset, data_dir, workdir / "secrets" / "pprl.key", plaintext_f1
    )
    summary["drift"] = _drift_report(dataset, data_dir, state_dir, seed)
    summary["ok"] = _all_checks_pass(summary)
    result: dict[str, JsonValue] = json.loads(json.dumps(summary, default=str))
    return result
