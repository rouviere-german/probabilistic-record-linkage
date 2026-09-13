"""Command line entry point (``probabilistic-record-linkage`` or ``python -m ...``)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["main"]

_SCOPES = ("link:run", "entity:read", "review:write", "pii:read")


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _service(state_dir: Path) -> Any:
    from probabilistic_record_linkage.api.app import workdir_service  # noqa: PLC0415

    return workdir_service(state_dir)


def _close(service: Any) -> None:
    service.shutdown()
    service.store.close()


def _serve(args: argparse.Namespace) -> int:
    import uvicorn  # noqa: PLC0415 - only needed when serving

    from probabilistic_record_linkage.api.app import create_linkage_app  # noqa: PLC0415
    from probabilistic_record_linkage.core.config import get_settings  # noqa: PLC0415
    from probabilistic_record_linkage.core.logging import configure_logging  # noqa: PLC0415
    from probabilistic_record_linkage.core.tracing import (  # noqa: PLC0415
        install_global_tracer_provider,
    )

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)
    app = create_linkage_app(settings)
    install_global_tracer_provider(app.state.tracer_provider)
    uvicorn.run(
        app,
        host=args.host or settings.host,
        port=args.port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )
    return 0


def _generate_api_key(args: argparse.Namespace) -> int:
    del args
    from probabilistic_record_linkage.core.security import generate_api_key  # noqa: PLC0415

    raw, digest = generate_api_key()
    _print({"api_key": raw, "api_key_sha256": digest})
    print("Store only api_key_sha256 in APP_API_KEY_SHA256.", file=sys.stderr)
    return 0


def _issue_token(args: argparse.Namespace) -> int:
    from probabilistic_record_linkage.core.config import get_settings  # noqa: PLC0415
    from probabilistic_record_linkage.core.security import issue_token  # noqa: PLC0415

    settings = get_settings()
    if settings.jwt_secret is None:
        print("issue-token needs APP_JWT_SECRET (at least 32 characters)", file=sys.stderr)
        return 2
    token = issue_token(
        settings.jwt_secret.get_secret_value().encode("utf-8"),
        subject=args.subject,
        scopes=args.scope,
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
        ttl_seconds=min(args.ttl, settings.jwt_max_ttl_seconds),
    )
    print(token)
    return 0


def _generate(args: argparse.Namespace) -> int:
    from probabilistic_record_linkage.linkage.generator import (  # noqa: PLC0415
        GeneratorConfig,
        generate,
        write_dataset,
    )

    dataset = generate(GeneratorConfig(seed=args.seed, n_records=args.records))
    hashes = write_dataset(dataset, args.out)
    _print({"records": dataset.n_records, "entities": len(dataset.population), "files": hashes})
    return 0


def _link(args: argparse.Namespace) -> int:
    from probabilistic_record_linkage.linkage.generator import load_sources  # noqa: PLC0415
    from probabilistic_record_linkage.linkage.model import register_run  # noqa: PLC0415

    data: Path = args.data
    state_dir: Path = args.workdir
    nick_source = data / "reference" / "nicknames.json"
    if nick_source.exists():
        (state_dir / "reference").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(nick_source, state_dir / "reference" / "nicknames.json")
    service = _service(state_dir)
    try:
        job_id = service.submit(
            "cli", load_sources(data), use_registered_model=args.use_registered_model, wait=True
        )
        job = service.job(job_id)
        output: dict[str, Any] = {
            "job_id": job_id,
            "status": job["status"],
            "summary": job["summary"],
        }
        if job["status"] == "succeeded" and args.register:
            manifest_path = data / "manifest.json"
            generator = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists()
                else {}
            )
            version = register_run(
                service.registry,
                service.run(job_id),
                name=service.model_name,
                generator_manifest=generator,
                lockfile=Path("uv.lock"),
                datasets=sorted((data / "sources").glob("*.parquet")),
            )
            service.registry.set_alias(service.model_name, "production", version.digest)
            output["model_version"] = version.digest
        _print(output)
        return 0 if job["status"] == "succeeded" else 1
    finally:
        _close(service)


def _review(args: argparse.Namespace) -> int:
    from probabilistic_record_linkage.linkage.service import JobNotFoundError  # noqa: PLC0415
    from probabilistic_record_linkage.linkage.store import ReviewConflictError  # noqa: PLC0415

    service = _service(args.workdir)
    try:
        if args.review_command == "list":
            _print(service.review_queue(args.job, include_pii=args.pii, limit=args.limit))
            return 0
        try:
            result = service.decide(
                args.job, args.item, decision=args.decision, reviewer=args.reviewer
            )
        except (ReviewConflictError, JobNotFoundError) as exc:
            print(f"review decision refused: {type(exc).__name__}", file=sys.stderr)
            return 1
        _print(result)
        return 0
    finally:
        _close(service)


def _entity(args: argparse.Namespace) -> int:
    from probabilistic_record_linkage.linkage.service import JobNotFoundError  # noqa: PLC0415

    service = _service(args.workdir)
    try:
        entity = service.entity(args.job, args.record, include_pii=args.pii)
    except JobNotFoundError:
        print("unknown job or record", file=sys.stderr)
        return 1
    else:
        _print(entity)
        return 0
    finally:
        _close(service)


def _audit(args: argparse.Namespace) -> int:
    from probabilistic_record_linkage.linkage.audit import verify_file  # noqa: PLC0415
    from probabilistic_record_linkage.linkage.service import (  # noqa: PLC0415
        REVIEW_AUDIT_ANCHOR,
        default_audit_path,
    )
    from probabilistic_record_linkage.linkage.store import LinkageStore  # noqa: PLC0415

    path: Path | None = args.path
    anchor = None
    if args.workdir is not None:
        database = args.workdir / "linkage.duckdb"
        if not database.exists():
            print("audit verify: no linkage store in --workdir", file=sys.stderr)
            return 2
        store = LinkageStore(database)
        try:
            anchor = store.load_audit_anchor(REVIEW_AUDIT_ANCHOR)
        finally:
            store.close()
        path = path or default_audit_path(args.workdir)
    if path is None:
        print("audit verify needs a log path or --workdir", file=sys.stderr)
        return 2
    result = verify_file(path, expect_head=args.expect_head, anchor=anchor)
    _print({"ok": result.ok, "entries": result.entries, "head": result.head, "error": result.error})
    return 0 if result.ok else 1


def _demo(args: argparse.Namespace) -> int:
    from probabilistic_record_linkage.core.logging import configure_logging  # noqa: PLC0415
    from probabilistic_record_linkage.linkage.demo import run_demo  # noqa: PLC0415

    configure_logging("WARNING", json_output=True, stream=sys.stderr)
    summary = run_demo(args.workdir, records=args.records)
    _print({"demo": "record-linkage", **summary})
    return 0 if summary.get("ok") else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probabilistic-record-linkage")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("generate-api-key", help="print a new API key and its SHA-256 digest")

    token = sub.add_parser("issue-token", help="issue a scoped JWT signed with APP_JWT_SECRET")
    token.add_argument("--subject", required=True)
    token.add_argument("--scope", action="append", choices=_SCOPES, required=True)
    token.add_argument("--ttl", type=int, default=900)

    generate = sub.add_parser("generate", help="write a synthetic multi-source dataset")
    generate.add_argument("--records", type=int, default=20_000)
    generate.add_argument("--seed", type=int, default=20260913)
    generate.add_argument("--out", type=Path, required=True)

    link = sub.add_parser("link", help="link a generated dataset into a local state directory")
    link.add_argument("--data", type=Path, required=True)
    link.add_argument("--workdir", type=Path, required=True)
    link.add_argument("--register", action="store_true", help="register the model as production")
    link.add_argument("--use-registered-model", action="store_true")

    review = sub.add_parser("review", help="review queue tools")
    review_sub = review.add_subparsers(dest="review_command", required=True)
    listing = review_sub.add_parser("list", help="list open review items of a job")
    decide = review_sub.add_parser("decide", help="decide a review item (audited)")
    for command in (listing, decide):
        command.add_argument("--workdir", type=Path, required=True)
        command.add_argument("--job", required=True)
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--pii", action="store_true", help="include record fields")
    decide.add_argument("--item", required=True)
    decide.add_argument("--decision", choices=["match", "non_match"], required=True)
    decide.add_argument("--reviewer", required=True)

    entity = sub.add_parser("entity", help="look up the resolved entity of a record")
    entity.add_argument("--workdir", type=Path, required=True)
    entity.add_argument("--job", required=True)
    entity.add_argument("--record", required=True)
    entity.add_argument("--pii", action="store_true")

    audit = sub.add_parser("audit", help="audit log tools")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify", help="verify the hash-chained review audit log")
    verify.add_argument("path", type=Path, nargs="?", default=None)
    verify.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="also check the head anchored in this state directory's DuckDB store",
    )
    verify.add_argument("--expect-head", default=None)

    demo = sub.add_parser("demo", help="run the offline end-to-end demo")
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))
    demo.add_argument("--records", type=int, default=20_000)
    return parser


_COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "serve": _serve,
    "generate-api-key": _generate_api_key,
    "issue-token": _issue_token,
    "generate": _generate,
    "link": _link,
    "review": _review,
    "entity": _entity,
    "audit": _audit,
    "demo": _demo,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "serve":
        from probabilistic_record_linkage.core.logging import configure_logging  # noqa: PLC0415

        # stdout carries the JSON results; logs go to stderr.
        configure_logging("WARNING", json_output=True, stream=sys.stderr)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
