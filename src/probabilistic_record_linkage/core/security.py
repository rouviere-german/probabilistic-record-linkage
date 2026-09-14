"""Authentication: API keys (operators, metrics) and scoped JWTs (linkage API users).

API keys are high-entropy random tokens, so a plain SHA-256 digest is an adequate at-rest form.
The presented key is hashed first and compared against *every* configured digest with
``hmac.compare_digest`` and no early exit, so timing reveals neither key prefixes nor which key
matched.

JWTs are verified with PyJWT pinned to HS256 (``alg: none`` and algorithm confusion are rejected),
and must carry ``sub``, ``iat``, ``exp``, ``iss`` and ``aud``; tokens whose lifetime exceeds the
configured maximum are refused. Scopes come from the space-separated ``scope`` claim; scopes this
service does not define are ignored. An optional ``tenant`` claim groups subjects that share link
jobs; a malformed one invalidates the token. Protected routes fail closed when no secret is
configured.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Protocol

import jwt
from fastapi import Depends, Request

from probabilistic_record_linkage.core.errors import AppError

__all__ = [
    "API_KEY_HEADER",
    "SCOPES",
    "TENANT_PATTERN",
    "ApiKeyVerifier",
    "AuthenticatedPrincipal",
    "CompositeVerifier",
    "JwtVerifier",
    "Principal",
    "PrincipalVerifier",
    "extract_api_key",
    "extract_bearer",
    "generate_api_key",
    "hash_api_key",
    "issue_token",
    "require_api_key",
    "require_scopes",
]

API_KEY_HEADER = "X-API-Key"
SCOPES = frozenset({"link:run", "entity:read", "review:write", "pii:read"})
TENANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MAX_KEY_LENGTH = 512
_MAX_TOKEN_LENGTH = 4096
_ALGORITHM = "HS256"


def hash_api_key(raw_key: str) -> str:
    """Lowercase hex SHA-256 of an API key, as stored in ``APP_API_KEY_SHA256``."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Return a new (raw_key, sha256_hex) pair. Show the raw key once; store only the digest."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_api_key(raw)


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated caller. ``key_id`` is a non-secret pseudonym safe for logs and buckets."""

    key_id: str
    subject: str | None = None
    scopes: frozenset[str] = field(default_factory=frozenset)
    tenant: str | None = None

    @property
    def owner(self) -> str:
        """Who owns the link jobs this caller creates and may read or review.

        Callers from the same ``tenant`` claim share jobs (for example a service account that
        submits batches and the reviewers who work its queue); without the claim, only the same
        token subject does.
        """
        return f"tenant:{self.tenant}" if self.tenant else f"principal:{self.key_id}"


class PrincipalVerifier(Protocol):
    def verify(self, presented: str | None) -> Principal | None: ...


class ApiKeyVerifier:
    def __init__(self, sha256_hex_digests: Iterable[str]) -> None:
        self._digests = tuple(bytes.fromhex(item) for item in sha256_hex_digests)

    @property
    def configured(self) -> bool:
        return bool(self._digests)

    def verify(self, presented: str | None) -> Principal | None:
        """Return the principal for a valid key, else None. Fails closed with no keys set."""
        if not presented or len(presented) > _MAX_KEY_LENGTH:
            return None
        digest = hashlib.sha256(presented.encode("utf-8")).digest()
        matched: bytes | None = None
        for candidate in self._digests:
            if hmac.compare_digest(digest, candidate):
                matched = candidate
        return Principal(key_id=matched.hex()[:12]) if matched is not None else None


def issue_token(
    secret: bytes,
    *,
    subject: str,
    scopes: Sequence[str],
    issuer: str,
    audience: str,
    ttl_seconds: int = 900,
    now: float | None = None,
    tenant: str | None = None,
) -> str:
    """Create an HS256 token (for operators and tests; production tokens come from an IdP)."""
    unknown = set(scopes) - SCOPES
    if unknown:
        raise ValueError(f"unknown scopes: {sorted(unknown)}")
    if tenant is not None and not TENANT_PATTERN.fullmatch(tenant):
        raise ValueError("tenant must match " + TENANT_PATTERN.pattern)
    issued = int(now if now is not None else time.time())
    claims: dict[str, object] = {
        "sub": subject,
        "iss": issuer,
        "aud": audience,
        "iat": issued,
        "exp": issued + ttl_seconds,
        "scope": " ".join(sorted(set(scopes))),
    }
    if tenant is not None:
        claims["tenant"] = tenant
    return jwt.encode(claims, secret, algorithm=_ALGORITHM)


class JwtVerifier:
    def __init__(
        self,
        secret: bytes | None,
        *,
        issuer: str,
        audience: str,
        max_ttl_seconds: int = 3600,
        leeway_seconds: int = 10,
    ) -> None:
        self._secret = secret
        self._issuer = issuer
        self._audience = audience
        self._max_ttl = max_ttl_seconds
        self._leeway = leeway_seconds
        self._pseudonym_key = (
            hmac.new(secret, b"principal-pseudonym", hashlib.sha256).digest() if secret else b""
        )

    @property
    def configured(self) -> bool:
        return bool(self._secret)

    def _identity(self, claims: Mapping[str, object]) -> tuple[str, str | None] | None:
        """``(subject, tenant)`` of well-formed claims within the lifetime cap, else ``None``."""
        subject = claims.get("sub")
        issued, expires = claims.get("iat"), claims.get("exp")
        if not isinstance(subject, str) or not 1 <= len(subject) <= 128:
            return None
        if not isinstance(issued, int | float) or not isinstance(expires, int | float):
            return None
        if expires - issued > self._max_ttl:
            return None
        tenant = claims.get("tenant")
        if tenant is not None and (
            not isinstance(tenant, str) or not TENANT_PATTERN.fullmatch(tenant)
        ):
            return None  # a malformed ownership claim must not fall back to another owner
        return subject, tenant

    def verify(self, presented: str | None) -> Principal | None:
        if not self._secret or not presented or len(presented) > _MAX_TOKEN_LENGTH:
            return None
        try:
            claims = jwt.decode(
                presented,
                self._secret,
                algorithms=[_ALGORITHM],
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway,
                options={"require": ["exp", "iat", "sub", "iss", "aud"]},
            )
        except jwt.PyJWTError:
            return None
        identity = self._identity(claims)
        if identity is None:
            return None
        subject, tenant = identity
        scope = claims.get("scope", "")
        granted = frozenset(scope.split()) & SCOPES if isinstance(scope, str) else frozenset()
        key_id = hmac.new(self._pseudonym_key, subject.encode("utf-8"), hashlib.sha256).hexdigest()[
            :12
        ]
        return Principal(key_id=key_id, subject=subject, scopes=granted, tenant=tenant)


class CompositeVerifier:
    """Try several verifiers in order (API key first, then JWT)."""

    def __init__(self, *verifiers: PrincipalVerifier) -> None:
        self._verifiers = verifiers

    def verify(self, presented: str | None) -> Principal | None:
        for verifier in self._verifiers:
            principal = verifier.verify(presented)
            if principal is not None:
                return principal
        return None


def extract_api_key(headers: Mapping[str, str]) -> str | None:
    """Read the key from ``X-API-Key`` or ``Authorization: Bearer <key>``.

    Starlette's ``Headers`` is case-insensitive; plain mappings must use lowercase names.
    """
    value = headers.get("x-api-key")
    if value:
        return value
    return extract_bearer(headers)


def extract_bearer(headers: Mapping[str, str]) -> str | None:
    auth = headers.get("authorization")
    if auth and auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    return None


def require_api_key(request: Request) -> Principal:
    """FastAPI dependency: authenticate the caller or raise a uniform 401."""
    verifier: ApiKeyVerifier = request.app.state.api_key_verifier
    principal = verifier.verify(extract_api_key(request.headers))
    if principal is None:
        request.app.state.metrics.auth_failures_total.inc()
        raise AppError(
            401,
            "unauthorized",
            "A valid API key is required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return principal


def require_scopes(*required: str) -> Callable[[Request], Principal]:
    """FastAPI dependency factory: a valid JWT carrying every scope in ``required``."""
    unknown = set(required) - SCOPES
    if unknown:
        raise ValueError(f"unknown scopes: {sorted(unknown)}")

    def dependency(request: Request) -> Principal:
        verifier: JwtVerifier = request.app.state.jwt_verifier
        principal = verifier.verify(extract_bearer(request.headers))
        if principal is None:
            request.app.state.metrics.auth_failures_total.inc()
            raise AppError(
                401,
                "unauthorized",
                "A valid bearer token is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not set(required) <= principal.scopes:
            raise AppError(403, "forbidden", "The token lacks a required scope.")
        return principal

    return dependency


AuthenticatedPrincipal = Annotated[Principal, Depends(require_api_key)]
