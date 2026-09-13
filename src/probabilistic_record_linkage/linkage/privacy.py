"""Privacy-preserving linkage with keyed Bloom-filter encodings (M5).

Each party encodes every field of every record as a Bloom filter (Schnell, Bachteler and Reiher
2009): the field's padded q-grams are hashed into ``bits`` positions with ``hashes`` HMAC-SHA256
based hash functions under a secret key shared by the parties but not by the linkage unit. Hash
positions come from HMAC output directly ("random hashing", Niedermeyer et al. 2014), not from
double hashing, whose structure enables known attacks. The linkage unit only sees bit vectors and
compares them with the Dice coefficient. Dates are encoded as component tokens (year, month, day
and the unordered month/day pair) instead of q-grams, so a one-component error or a day/month swap
stays partially similar while unrelated dates do not.

Hardening (Schnell, Borgs and Encarnacao 2016; Christen et al. 2017, "Pattern-mining based
cryptanalysis of Bloom filters"), on by default for the variable-length, frequency-skewed string
fields (names, street, city):

- per-field keys derived from the master key, and name fields salted with the record's birth year,
  so identical names of people born in different years produce unrelated encodings and encoding
  frequencies no longer mirror name frequencies;
- XOR-folding: the two halves of a filter are XORed, breaking the one-to-one map from q-grams to
  bit positions;
- balancing: the filter is concatenated with its complement and permuted with a keyed
  permutation, so every encoding of a field has the same Hamming weight and weights no longer
  reveal name length.

Blocking in privacy mode uses keyed tokens (HMAC of the date of birth, and of Soundex codes with the
birth year) instead of MinHash over the encodings: balanced filters have half their bits set, which
destroys the Jaccard signal LSH relies on. The key is read only from a mounted secret file
(:func:`load_key_file`); there is deliberately no environment-variable or inline-key path.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from probabilistic_record_linkage.linkage.blocking import bucket_pairs, dedupe_pairs
from probabilistic_record_linkage.linkage.comparison import (
    ComparisonSpace,
    FieldSpec,
    exact_agreement_rate,
    random_pairs,
    u_from_random_pairs,
)
from probabilistic_record_linkage.linkage.phonetic import soundex

__all__ = [
    "PRIVATE_SPACE",
    "BloomComparator",
    "EncodedRecords",
    "HardeningConfig",
    "InsecureKeyFileError",
    "PrivacyConfig",
    "date_tokens",
    "derive_key",
    "dice_pairs",
    "encode_records",
    "load_key_file",
    "name_salts",
    "token_blocking",
]

UInt64Array = npt.NDArray[np.uint64]
Int64Array = npt.NDArray[np.int64]
Int32Array = npt.NDArray[np.int32]
Int8Array = npt.NDArray[np.int8]
BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]

MIN_KEY_BYTES = 32
MAX_KEY_BYTES = 4096
_CHUNK = 200_000

PRIVATE_SPACE = ComparisonSpace(
    (
        FieldSpec("first_name", ("else", "dice-low", "dice-high", "identical")),
        FieldSpec("last_name", ("else", "dice-low", "dice-high", "identical")),
        FieldSpec("dob", ("else", "dice-high", "identical")),
        FieldSpec("sex", ("else", "identical")),
        FieldSpec("address", ("else", "same city", "same postcode", "street similar", "identical")),
    )
)


class InsecureKeyFileError(PermissionError):
    """The Bloom-filter key file is missing, malformed or readable by other users."""


def load_key_file(path: Path) -> bytes:
    """Read the shared Bloom-filter key from a mounted secret file.

    Refuses symlinks, non-regular files, files outside ``[32, 4096]`` bytes after stripping
    whitespace, and (on POSIX) files that group or others can read or write.
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise InsecureKeyFileError("key file not found") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise InsecureKeyFileError("key file must be a regular file, not a link or device")
    if os.name == "posix" and info.st_mode & 0o077:
        raise InsecureKeyFileError("key file must not be accessible by group or others (chmod 600)")
    if info.st_size > MAX_KEY_BYTES * 2:
        raise InsecureKeyFileError("key file is too large")
    key = path.read_bytes().strip()
    if not MIN_KEY_BYTES <= len(key) <= MAX_KEY_BYTES:
        raise InsecureKeyFileError(f"key must be between {MIN_KEY_BYTES} and {MAX_KEY_BYTES} bytes")
    return key


def derive_key(master: bytes, label: str) -> bytes:
    """Independent per-purpose key (HMAC-SHA256 as a PRF)."""
    return hmac.new(master, b"prl-v1:" + label.encode("utf-8"), hashlib.sha256).digest()


class HardeningConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    salt_names_with_birth_year: bool = True
    # Record attributes combined into the name salt when salting is on. More parts flatten
    # encoding frequencies further, but a typo in any part also breaks name similarity.
    name_salt_parts: tuple[Literal["birth_year", "birth_month", "sex"], ...] = ("birth_year",)
    xor_fold: bool = True
    balance: bool = True
    # Variable-length, frequency-skewed strings are hardened; fixed-format fields (date, sex,
    # postcode) keep plain keyed encodings so their similarity levels stay usable.
    hardened_fields: tuple[str, ...] = ("first_name", "last_name", "street", "city")

    @property
    def enabled(self) -> bool:
        return self.salt_names_with_birth_year or self.xor_fold or self.balance

    def folds(self, field: str) -> bool:
        return self.xor_fold and field in self.hardened_fields

    def balances(self, field: str) -> bool:
        return self.balance and field in self.hardened_fields


class PrivacyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name_bits: int = Field(default=512, ge=64, le=8192)
    name_hashes: int = Field(default=12, ge=1, le=64)
    other_bits: int = Field(default=256, ge=64, le=8192)
    other_hashes: int = Field(default=10, ge=1, le=64)
    q: int = Field(default=2, ge=1, le=4)
    hardening: HardeningConfig = Field(default_factory=HardeningConfig)
    max_bucket: int = Field(default=300, ge=2)
    bucket_window: int = Field(default=10, ge=1)
    # Calibrated on Dice distributions of matching versus non-matching candidate pairs.
    plain_cutpoints: tuple[float, float] = (0.55, 0.70)
    balanced_cutpoints: tuple[float, float] = (0.70, 0.80)
    dob_cutpoint: float = 0.45

    @classmethod
    def plain(cls) -> PrivacyConfig:
        """Unhardened encodings: only for measuring what hardening buys."""
        return cls(
            hardening=HardeningConfig(
                salt_names_with_birth_year=False, xor_fold=False, balance=False
            )
        )

    def dice_cutpoints(self, field: str) -> tuple[float, float]:
        """(low, high) Dice thresholds; balancing compresses Dice towards [0.5, 1]."""
        if field == "dob":
            return (self.dob_cutpoint, self.dob_cutpoint)
        return self.balanced_cutpoints if self.hardening.balances(field) else self.plain_cutpoints


@dataclass(frozen=True, slots=True)
class EncodedRecords:
    """What a linkage unit receives: record ids, packed encodings and keyed blocking tokens."""

    record_ids: tuple[str, ...]
    fields: dict[str, UInt64Array]  # (n, words)
    present: dict[str, BoolArray]
    popcounts: dict[str, Int32Array]
    tokens: dict[str, Int64Array]  # -1 where the token is undefined
    bits: dict[str, int]  # encoded length per field after folding and balancing
    config: PrivacyConfig

    @property
    def n(self) -> int:
        return len(self.record_ids)


def _positions(key: bytes, salt: str, gram: str, bits: int, hashes: int) -> list[int]:
    positions: list[int] = []
    counter = 0
    width = max(1, math.ceil(math.log2(bits) / 8)) + 1
    while len(positions) < hashes:
        message = f"{salt}\x1f{gram}\x1f{counter}".encode()
        digest = hmac.new(key, message, hashlib.sha256).digest()
        for offset in range(0, len(digest) - width + 1, width):
            positions.append(int.from_bytes(digest[offset : offset + width], "little") % bits)
            if len(positions) == hashes:
                break
        counter += 1
    return positions


def _qgrams(value: str, q: int) -> set[str]:
    padded = f"#{value}#" if q > 1 else value
    return {padded[i : i + q] for i in range(max(1, len(padded) - q + 1))}


def date_tokens(value: str) -> set[str]:
    """Tokens of a ``YYYYMMDD`` date: year, month, day and the unordered month/day pair.

    Identical dates share all four tokens (Dice 1); a one-component error or a day/month swap
    shares two or three (Dice 0.5 to 0.75); unrelated dates rarely share more than one.
    """
    year, month, day = value[:4], value[4:6], value[6:8]
    low, high = sorted((month, day))
    return {f"y{year}", f"m{month}", f"d{day}", f"md{low}{high}"}


class _FieldEncoder:
    def __init__(
        self, master: bytes, field: str, bits: int, hashes: int, config: PrivacyConfig
    ) -> None:
        self.key = derive_key(master, f"bloom:{field}")
        self.field = field
        self.bits = bits
        self.hashes = hashes
        self.q = config.q
        self.fold = config.hardening.folds(field)
        self.balance = config.hardening.balances(field)
        self._gram_cache: dict[tuple[str, str], list[int]] = {}
        seed = int.from_bytes(derive_key(master, f"permutation:{field}")[:8], "little")
        out_bits = bits // 2 if self.fold else bits
        self.out_bits = out_bits * 2 if self.balance else out_bits
        self.permutation = np.random.default_rng(seed).permutation(self.out_bits)
        self.words = math.ceil(self.out_bits / 64)

    def encode(self, value: str, salt: str) -> UInt64Array:
        vector = np.zeros(self.bits, dtype=bool)
        grams = date_tokens(value) if self.field == "dob" else _qgrams(value, self.q)
        for gram in grams:
            cache_key = (salt, gram)
            positions = self._gram_cache.get(cache_key)
            if positions is None:
                positions = _positions(self.key, salt, gram, self.bits, self.hashes)
                self._gram_cache[cache_key] = positions
            vector[positions] = True
        if self.fold:
            half = self.bits // 2
            vector = vector[:half] ^ vector[half : 2 * half]
        if self.balance:
            vector = np.concatenate([vector, ~vector])[self.permutation]
        padded = np.zeros(self.words * 64, dtype=bool)
        padded[: vector.size] = vector
        packed: UInt64Array = np.frombuffer(np.packbits(padded).tobytes(), dtype=">u8").astype(
            np.uint64
        )
        return packed


def _hmac_token(key: bytes, text: str) -> int:
    digest = hmac.new(key, text.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF


def name_salts(normalized: pl.DataFrame, hardening: HardeningConfig) -> list[str]:
    """Per-record salt for name fields ("" when salting is off or a salt part is missing)."""
    n = normalized.height
    if not hardening.salt_names_with_birth_year:
        return [""] * n
    dob = normalized["dob"]
    columns: list[list[str | None]] = []
    for part in hardening.name_salt_parts:
        if part == "birth_year":
            columns.append([str(y) if y is not None else None for y in dob.dt.year().to_list()])
        elif part == "birth_month":
            columns.append(
                [f"{m:02d}" if m is not None else None for m in dob.dt.month().to_list()]
            )
        else:
            columns.append(normalized["sex"].to_list())
    return [
        "" if any(v is None for v in row) else "|".join(row) for row in zip(*columns, strict=True)
    ]


@dataclass(frozen=True, slots=True)
class _FieldPlan:
    name: str
    values: list[str | None]
    salts: list[str]
    bits: int
    hashes: int


def _field_plans(normalized: pl.DataFrame, config: PrivacyConfig) -> list[_FieldPlan]:
    n = normalized.height
    dob = normalized["dob"]
    dob_text = [d.strftime("%Y%m%d") if d is not None else None for d in dob.to_list()]
    salts = name_salts(normalized, config.hardening)
    unsalted = [""] * n
    sex = [s.lower() if s is not None else None for s in normalized["sex"].to_list()]
    name_size = (config.name_bits, config.name_hashes)
    other_size = (config.other_bits, config.other_hashes)
    return [
        _FieldPlan("first_name", normalized["first_name"].to_list(), salts, *name_size),
        _FieldPlan("last_name", normalized["last_name"].to_list(), salts, *name_size),
        _FieldPlan("dob", dob_text, unsalted, *other_size),
        _FieldPlan("sex", sex, unsalted, *other_size),
        _FieldPlan("street", normalized["street"].to_list(), unsalted, *name_size),
        _FieldPlan("postcode", normalized["postcode"].to_list(), unsalted, *other_size),
        _FieldPlan("city", normalized["city"].to_list(), unsalted, *other_size),
    ]


def _encode_field(
    master_key: bytes, plan: _FieldPlan, config: PrivacyConfig
) -> tuple[UInt64Array, BoolArray, Int32Array, int]:
    """Encode one field; each distinct (salt, value) is encoded once."""
    encoder = _FieldEncoder(master_key, plan.name, plan.bits, plan.hashes, config)
    cache: dict[tuple[str, str], int] = {}
    rows: list[UInt64Array] = []
    index = np.full(len(plan.values), -1, dtype=np.int64)
    for row, (value, salt) in enumerate(zip(plan.values, plan.salts, strict=True)):
        if value is None:
            continue
        combo = (salt, value)
        position = cache.get(combo)
        if position is None:
            position = len(rows)
            cache[combo] = position
            rows.append(encoder.encode(value, salt))
        index[row] = position
    table = np.vstack(rows) if rows else np.zeros((1, encoder.words), dtype=np.uint64)
    matrix = np.zeros((len(plan.values), encoder.words), dtype=np.uint64)
    mask = index >= 0
    matrix[mask] = table[index[mask]]
    popcount = np.bitwise_count(matrix).sum(axis=1).astype(np.int32)
    return matrix, mask, popcount, encoder.out_bits


def _blocking_tokens(normalized: pl.DataFrame, master_key: bytes) -> dict[str, Int64Array]:
    token_key = derive_key(master_key, "blocking-tokens")
    dob = normalized["dob"]
    years: list[int | None] = dob.dt.year().to_list()
    dob_text = [d.strftime("%Y%m%d") if d is not None else None for d in dob.to_list()]
    sx_first = [soundex(v) if v else "" for v in normalized["first_name"].to_list()]
    sx_last = [soundex(v) if v else "" for v in normalized["last_name"].to_list()]
    specs: dict[str, list[str | None]] = {
        "dob": [f"dob|{d}" if d else None for d in dob_text],
        "names": [
            f"sx|{a}|{b}" if a and b else None for a, b in zip(sx_first, sx_last, strict=True)
        ],
        "last_year": [
            f"ly|{b}|{y}" if b and y is not None else None
            for b, y in zip(sx_last, years, strict=True)
        ],
        "first_year": [
            f"fy|{a}|{y}" if a and y is not None else None
            for a, y in zip(sx_first, years, strict=True)
        ],
    }
    tokens: dict[str, Int64Array] = {}
    for name, texts in specs.items():
        cache: dict[str, int] = {}
        out = np.full(len(texts), -1, dtype=np.int64)
        for row, text in enumerate(texts):
            if text is None:
                continue
            token = cache.get(text)
            if token is None:
                token = _hmac_token(token_key, text)
                cache[text] = token
            out[row] = token
        tokens[name] = out
    return tokens


def encode_records(
    normalized: pl.DataFrame, master_key: bytes, config: PrivacyConfig
) -> EncodedRecords:
    """Encode a normalized frame (see :func:`normalize.normalize_sources`) for privacy mode."""
    if len(master_key) < MIN_KEY_BYTES:
        raise InsecureKeyFileError(f"key must be at least {MIN_KEY_BYTES} bytes")
    fields: dict[str, UInt64Array] = {}
    present: dict[str, BoolArray] = {}
    popcounts: dict[str, Int32Array] = {}
    bits: dict[str, int] = {}
    for plan in _field_plans(normalized, config):
        matrix, mask, popcount, out_bits = _encode_field(master_key, plan, config)
        fields[plan.name] = matrix
        present[plan.name] = mask
        popcounts[plan.name] = popcount
        bits[plan.name] = out_bits
    return EncodedRecords(
        record_ids=tuple(normalized["record_id"].to_list()),
        fields=fields,
        present=present,
        popcounts=popcounts,
        tokens=_blocking_tokens(normalized, master_key),
        bits=bits,
        config=config,
    )


def token_blocking(encoded: EncodedRecords) -> tuple[Int64Array, Int64Array]:
    """Candidate pairs sharing any keyed blocking token (large buckets use a sorted window)."""
    lefts: list[Int64Array] = []
    rights: list[Int64Array] = []
    secondary = encoded.tokens["dob"]
    for name in sorted(encoded.tokens):
        keys = encoded.tokens[name]
        left, right = bucket_pairs(
            keys,
            keys >= 0,
            max_bucket=encoded.config.max_bucket,
            window=encoded.config.bucket_window,
            secondary=secondary,
        )
        lefts.append(left)
        rights.append(right)
    return dedupe_pairs(np.concatenate(lefts), np.concatenate(rights), encoded.n)


def dice_pairs(
    encoded: EncodedRecords, field: str, left: Int64Array, right: Int64Array
) -> FloatArray:
    """Dice coefficient ``2 |a & b| / (|a| + |b|)`` for each pair (0 where a side is missing)."""
    matrix = encoded.fields[field]
    counts = encoded.popcounts[field]
    out = np.zeros(left.size, dtype=np.float64)
    for start in range(0, left.size, _CHUNK):
        stop = min(left.size, start + _CHUNK)
        a, b = left[start:stop], right[start:stop]
        inter = np.bitwise_count(matrix[a] & matrix[b]).sum(axis=1)
        denominator = counts[a] + counts[b]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[start:stop] = np.where(denominator > 0, 2.0 * inter / denominator, 0.0)
    return out


class BloomComparator:
    """Agreement levels from Dice similarities of encodings (``PRIVATE_SPACE``); no TF terms."""

    def __init__(self, encoded: EncodedRecords) -> None:
        self.encoded = encoded
        self.cutpoints = {name: encoded.config.dice_cutpoints(name) for name in encoded.fields}

    @property
    def space(self) -> ComparisonSpace:
        return PRIVATE_SPACE

    @property
    def n_records(self) -> int:
        return self.encoded.n

    def _both(self, field: str, left: Int64Array, right: Int64Array) -> BoolArray:
        present = self.encoded.present[field]
        both: BoolArray = present[left] & present[right]
        return both

    def _identical(self, field: str, left: Int64Array, right: Int64Array) -> BoolArray:
        matrix = self.encoded.fields[field]
        same = np.ones(left.size, dtype=bool)
        for start in range(0, left.size, _CHUNK):
            stop = min(left.size, start + _CHUNK)
            same[start:stop] = np.all(matrix[left[start:stop]] == matrix[right[start:stop]], axis=1)
        return same & self._both(field, left, right)

    def _name_level(self, field: str, left: Int64Array, right: Int64Array) -> Int8Array:
        low, high = self.cutpoints[field]
        dice = dice_pairs(self.encoded, field, left, right)
        level = np.ones(left.size, dtype=np.int8)
        level[dice >= low] = 2
        level[dice >= high] = 3
        level[self._identical(field, left, right)] = 4
        level[~self._both(field, left, right)] = 0
        return level

    def _address_level(self, left: Int64Array, right: Int64Array) -> Int8Array:
        same_street = self._identical("street", left, right)
        same_postcode = self._identical("postcode", left, right)
        same_city = self._identical("city", left, right)
        street_dice = dice_pairs(self.encoded, "street", left, right)
        present = self.encoded.present
        any_a = present["street"][left] | present["postcode"][left] | present["city"][left]
        any_b = present["street"][right] | present["postcode"][right] | present["city"][right]
        similar = same_postcode & (street_dice >= self.cutpoints["street"][1])
        address = np.ones(left.size, dtype=np.int8)
        address[same_city] = 2
        address[same_postcode] = 3
        address[similar | (same_street & ~same_postcode)] = 4
        address[same_street & same_postcode] = 5
        address[~any_a | ~any_b] = 0
        return address

    def levels(self, left: Int64Array, right: Int64Array) -> Int8Array:
        out = np.zeros((left.size, len(PRIVATE_SPACE.fields)), dtype=np.int8)
        out[:, 0] = self._name_level("first_name", left, right)
        out[:, 1] = self._name_level("last_name", left, right)
        dice = dice_pairs(self.encoded, "dob", left, right)
        dob = np.ones(left.size, dtype=np.int8)
        dob[dice >= self.cutpoints["dob"][0]] = 2
        dob[self._identical("dob", left, right)] = 3
        dob[~self._both("dob", left, right)] = 0
        out[:, 2] = dob
        sex = np.where(self._identical("sex", left, right), 2, 1).astype(np.int8)
        sex[~self._both("sex", left, right)] = 0
        out[:, 3] = sex
        out[:, 4] = self._address_level(left, right)
        return out

    def tf_adjustment(self, left: Int64Array, right: Int64Array, levels: Int8Array) -> FloatArray:
        del right, levels
        return np.zeros(left.size, dtype=np.float64)

    def identity_codes(self, field: str) -> Int64Array:
        """Equal codes for identical encodings; -1 where the field is missing."""
        present = self.encoded.present[field]
        codes = np.full(self.encoded.n, -1, dtype=np.int64)
        if present.any():
            _, inverse = np.unique(self.encoded.fields[field][present], axis=0, return_inverse=True)
            codes[present] = inverse.reshape(-1)
        return codes

    def random_u(self, n_pairs: int, rng: np.random.Generator) -> tuple[FloatArray, ...]:
        left, right = random_pairs(self.encoded.n, n_pairs, rng)
        sampled = self.levels(left, right)
        names = ("first_name", "last_name", "dob", "sex", "street", "postcode", "city")
        codes = {name: self.identity_codes(name) for name in names}
        both = (codes["street"] >= 0) & (codes["postcode"] >= 0)
        any_present = (codes["street"] >= 0) | (codes["postcode"] >= 0) | (codes["city"] >= 0)
        combined = codes["street"] * (int(codes["postcode"].max()) + 2) + codes["postcode"]
        share_both = float(both.mean()) if both.size else 0.0
        share_any = float(any_present.mean()) if any_present.size else 1.0
        rates = {
            name: exact_agreement_rate(codes[name], codes[name] >= 0)
            for name in ("first_name", "last_name", "dob", "sex")
        }
        rates["address"] = (
            exact_agreement_rate(combined, both) * share_both**2 / max(share_any**2, 1e-12)
        )
        return u_from_random_pairs(PRIVATE_SPACE, sampled, rates)
