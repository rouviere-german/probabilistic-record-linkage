from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace as otel_trace

from probabilistic_record_linkage import cli
from probabilistic_record_linkage.core import tracing
from probabilistic_record_linkage.core.config import get_settings
from probabilistic_record_linkage.core.security import JwtVerifier, hash_api_key
from probabilistic_record_linkage.linkage.generator import load_sources

pytestmark = pytest.mark.integration


def _objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        obj, index = decoder.raw_decode(text, index)
        objects.append(obj)
    return objects


def _last(capsys: pytest.CaptureFixture[str]) -> Any:
    return _objects(capsys.readouterr().out)[-1]


@pytest.mark.usefixtures("restore_logging")
def test_generate_link_review_entity_and_audit_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data, state = tmp_path / "data", tmp_path / "state"
    assert cli.main(["generate", "--records", "3000", "--seed", "11", "--out", str(data)]) == 0
    assert _last(capsys)["records"] > 2_000
    assert cli.main(["link", "--data", str(data), "--workdir", str(state), "--register"]) == 0
    linked = _last(capsys)
    job = linked["job_id"]
    assert linked["status"] == "succeeded"
    assert len(linked["model_version"]) == 64

    assert cli.main(["review", "list", "--workdir", str(state), "--job", job, "--limit", "5"]) == 0
    items = _last(capsys)
    assert items
    assert "left_fields" not in items[0]
    decide = [
        "review", "decide", "--workdir", str(state), "--job", job, "--item", items[0]["item_id"],
        "--decision", "match", "--reviewer", "tester",
    ]  # fmt: skip
    assert cli.main(decide) == 0
    assert _last(capsys)["same_entity"] is True
    assert cli.main(decide) == 1
    capsys.readouterr()

    record = load_sources(data)["s1"]["record_id"][0]
    entity_args = ["entity", "--workdir", str(state), "--job", job, "--record", record]
    assert cli.main(entity_args) == 0
    assert "fields" not in _last(capsys)["members"][0]
    assert cli.main([*entity_args, "--pii"]) == 0
    assert "fields" in _last(capsys)["members"][0]
    assert (
        cli.main(["entity", "--workdir", str(state), "--job", job, "--record", "s1-9999999"]) == 1
    )

    audit_path = state / "audit" / "review-decisions.jsonl"
    assert cli.main(["audit", "verify", str(audit_path)]) == 0
    assert cli.main(["audit", "verify", str(audit_path), "--expect-head", "0" * 64]) == 1
    capsys.readouterr()

    assert (
        cli.main(["link", "--data", str(data), "--workdir", str(state), "--use-registered-model"])
        == 0
    )
    assert _last(capsys)["summary"]["used_registered_model"] is True

    assert cli.main(["audit", "verify", "--workdir", str(state)]) == 0
    assert _last(capsys)["entries"] == 1
    audit_path.write_bytes(b"")  # truncated to nothing: only the anchored head reveals it
    assert cli.main(["audit", "verify", str(audit_path)]) == 0
    assert cli.main(["audit", "verify", "--workdir", str(state)]) == 1
    assert "anchored head" in _last(capsys)["error"]
    assert cli.main(["audit", "verify"]) == 2
    assert cli.main(["audit", "verify", "--workdir", str(tmp_path / "nowhere")]) == 2


def test_issue_token_needs_a_secret_and_signs_verifiable_tokens(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "cli-test-secret-" + "q" * 20
    get_settings.cache_clear()
    try:
        assert cli.main(["issue-token", "--subject", "alice", "--scope", "entity:read"]) == 2
        monkeypatch.setenv("APP_JWT_SECRET", secret)
        get_settings.cache_clear()
        args = [
            "issue-token",
            "--subject",
            "alice",
            "--scope",
            "entity:read",
            "--scope",
            "pii:read",
        ]
        assert cli.main(args) == 0
    finally:
        get_settings.cache_clear()
    token = capsys.readouterr().out.strip().splitlines()[-1]
    verifier = JwtVerifier(
        secret.encode(), issuer="probabilistic-record-linkage", audience="linkage-api"
    )
    principal = verifier.verify(token)
    assert principal is not None
    assert principal.scopes == {"entity:read", "pii:read"}


def test_generate_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate-api-key"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["api_key_sha256"] == hash_api_key(payload["api_key"])


@pytest.mark.usefixtures("restore_logging")
def test_serve_configures_uvicorn_safely(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import uvicorn

    captured: dict[str, Any] = {}

    def fake_run(app: Any, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        assert cli.main(["serve", "--port", "8123"]) == 0
    finally:
        get_settings.cache_clear()
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False
    service = captured["app"].state.linkage_service
    service.shutdown()
    service.store.close()


def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["probabilistic_record_linkage", "generate-api-key"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("probabilistic_record_linkage", run_name="__main__")
    assert exit_info.value.code == 0
    assert "api_key_sha256" in capsys.readouterr().out


@pytest.mark.usefixtures("restore_logging")
def test_demo_runs_offline_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["demo", "--workdir", str(tmp_path), "--records", "3000"])
    summary = _last(capsys)
    assert summary["demo"] == "record-linkage"
    assert summary["audit"]["ok"] is True
    assert all(summary["readiness"].values())
    assert summary["blocking"]["pair_completeness"] >= 0.95
    privacy = summary["privacy"]
    assert privacy["hardened"]["worst_attack_rate"] <= privacy["plain_bloom"]["worst_attack_rate"]
    assert code == (0 if summary["ok"] else 1)
