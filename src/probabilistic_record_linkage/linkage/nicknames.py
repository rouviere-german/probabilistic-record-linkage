"""Nickname table for the synthetic first-name vocabulary.

Real nickname lists do not apply to synthetic names, so the table is derived deterministically from
the vocabulary when the world is built (short stems such as ``bran``/``branny`` for ``brandor``)
and shipped with every generated dataset as reference data (``reference/nicknames.json``). The
generator uses it to corrupt first names; the linker loads the shipped file, the same way a
production linker loads a curated nickname list. Nicknames never coincide with a vocabulary word,
so a nickname is never mistaken for someone else's given name, but one nickname may belong to
several canonical names (``bran`` for ``brandor`` and ``branwen``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from probabilistic_record_linkage.core.canonical import canonical_json, sha256_hex

__all__ = ["NicknameTable", "derive_nicknames"]

_MAX_NICKNAMES = 3
_VOWELS = frozenset("aeiou")


class NicknameFormatError(ValueError):
    """A nickname file is not a JSON object of name -> list of names."""


@dataclass(frozen=True)
class NicknameTable:
    """``canonical name -> nicknames`` with a reverse index. All names are normalized lowercase."""

    nicknames: Mapping[str, tuple[str, ...]]
    _canonical_of: dict[str, tuple[str, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        reverse: dict[str, set[str]] = {}
        for canonical, nicks in self.nicknames.items():
            reverse.setdefault(canonical, set()).add(canonical)
            for nick in nicks:
                if nick == canonical:
                    raise ValueError("a nickname must differ from its canonical name")
                reverse.setdefault(nick, set()).add(canonical)
        object.__setattr__(
            self, "_canonical_of", {key: tuple(sorted(value)) for key, value in reverse.items()}
        )

    def canonical_names(self, value: str) -> tuple[str, ...]:
        """Canonical names ``value`` may stand for (empty for unknown values)."""
        return self._canonical_of.get(value, ())

    def nicknames_of(self, canonical: str) -> tuple[str, ...]:
        return tuple(self.nicknames.get(canonical, ()))

    def to_json(self) -> str:
        return canonical_json({k: list(v) for k, v in sorted(self.nicknames.items())})

    def sha256(self) -> str:
        return sha256_hex(self.to_json())

    @classmethod
    def from_json(cls, text: str) -> NicknameTable:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise NicknameFormatError("nickname table must be a JSON object")
        table: dict[str, tuple[str, ...]] = {}
        for key, value in data.items():
            if not isinstance(key, str) or not isinstance(value, list):
                raise NicknameFormatError("nickname table entries must map a name to a list")
            if not all(isinstance(item, str) for item in value):
                raise NicknameFormatError("nicknames must be strings")
            table[key] = tuple(value)
        return cls(table)


def _stem(name: str) -> str:
    """First syllable-ish stem: through the first vowel run and one following consonant."""
    index = 0
    while index < len(name) and name[index] not in _VOWELS:
        index += 1
    while index < len(name) and name[index] in _VOWELS:
        index += 1
    return name[: min(len(name), index + 1)]


def derive_nicknames(
    first_names: Iterable[str],
    rng: np.random.Generator,
    *,
    forbidden: Iterable[str] = (),
    share_with_nicknames: float = 0.6,
    min_name_length: int = 5,
) -> NicknameTable:
    """Derive 1-3 nicknames for about ``share_with_nicknames`` of the names of length >= 5."""
    names = list(first_names)
    blocked = set(forbidden) | set(names)
    table: dict[str, tuple[str, ...]] = {}
    coins = rng.random(len(names))
    for name, coin in zip(names, coins, strict=True):
        if len(name) < min_name_length or coin >= share_with_nicknames:
            continue
        stem = _stem(name)
        if len(stem) < 2 or stem == name:
            continue
        candidates = [stem, stem + "y", (stem[-1] in _VOWELS and stem + "n") or stem + "o"]
        chosen: list[str] = []
        for candidate in candidates:
            if candidate != name and candidate not in blocked and candidate not in chosen:
                chosen.append(candidate)
            if len(chosen) == _MAX_NICKNAMES:
                break
        if chosen:
            table[name] = tuple(chosen)
    return NicknameTable(table)
