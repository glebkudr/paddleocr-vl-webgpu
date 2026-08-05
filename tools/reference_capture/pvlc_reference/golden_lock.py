from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from blake3 import blake3

from .model_lock import PINNED_REVISION
from .trace_bundle import (
    TRACE_SCHEMA_VERSION,
    BundleFormatError,
    BundleIntegrityError,
    TraceLevel,
    verify_bundle,
)


PINNED_GOLDEN_LOCK_DIGEST = (
    "40947f87eec2ac0f75ce671ca9226bb335adbe6254e5c1858f5c2ae6310450c9"
)

_DIGEST_RE = re.compile(r"^blake3:[0-9a-f]{64}$")
_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class GoldenLockError(ValueError):
    def __init__(
        self, code: str, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True, slots=True)
class GoldenEntry:
    case_id: str
    trace_level: TraceLevel
    artifact_path: str
    bundle_digest: str
    semantic_fingerprint: str
    generated_tokens: tuple[int, ...]
    decoded_text: str
    repeat_count: int

    def validate(self) -> None:
        if not isinstance(self.case_id, str) or _CASE_ID_RE.fullmatch(self.case_id) is None:
            raise GoldenLockError("invalid_case_id", "invalid golden case ID")
        if not isinstance(self.trace_level, TraceLevel):
            raise GoldenLockError("invalid_trace_level", "invalid golden trace level")
        _validate_artifact_path(self.artifact_path)
        for value in (self.bundle_digest, self.semantic_fingerprint):
            if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
                raise GoldenLockError("invalid_digest", "invalid golden digest")
        if not isinstance(self.generated_tokens, tuple) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.generated_tokens
        ):
            raise GoldenLockError(
                "invalid_generated_tokens", "generated tokens must be nonnegative integers"
            )
        if self.trace_level in {TraceLevel.L2, TraceLevel.L3} and not self.generated_tokens:
            raise GoldenLockError(
                "invalid_generated_tokens", "stage/deep goldens require generated tokens"
            )
        if not isinstance(self.decoded_text, str):
            raise GoldenLockError("invalid_decoded_text", "decoded text must be a string")
        if (
            isinstance(self.repeat_count, bool)
            or not isinstance(self.repeat_count, int)
            or self.repeat_count < 2
        ):
            raise GoldenLockError(
                "insufficient_repeat_count", "golden publication requires two captures"
            )


@dataclass(frozen=True, slots=True)
class GoldenLock:
    format_version: int
    model_revision: str
    trace_schema_version: int
    bundles: tuple[GoldenEntry, ...]

    def validate(self) -> None:
        if self.format_version != 1 or isinstance(self.format_version, bool):
            raise GoldenLockError(
                "unsupported_format_version", "unsupported golden lock format"
            )
        if self.model_revision != PINNED_REVISION:
            raise GoldenLockError(
                "wrong_model_identity", "golden lock identifies another model revision"
            )
        if (
            self.trace_schema_version != TRACE_SCHEMA_VERSION
            or isinstance(self.trace_schema_version, bool)
        ):
            raise GoldenLockError(
                "unsupported_trace_schema", "unsupported golden trace schema"
            )
        if not isinstance(self.bundles, tuple) or not self.bundles:
            raise GoldenLockError("empty_golden_lock", "golden lock has no bundles")

        identities: set[tuple[str, TraceLevel]] = set()
        paths: set[str] = set()
        for entry in self.bundles:
            if not isinstance(entry, GoldenEntry):
                raise GoldenLockError("invalid_bundle", "invalid golden entry")
            entry.validate()
            identity = (entry.case_id, entry.trace_level)
            if identity in identities or entry.artifact_path in paths:
                raise GoldenLockError(
                    "duplicate_bundle", "duplicate golden identity or artifact path"
                )
            identities.add(identity)
            paths.add(entry.artifact_path)

    def canonicalized(self) -> GoldenLock:
        return GoldenLock(
            format_version=self.format_version,
            model_revision=self.model_revision,
            trace_schema_version=self.trace_schema_version,
            bundles=tuple(
                sorted(
                    self.bundles,
                    key=lambda entry: (entry.case_id, entry.trace_level.value),
                )
            ),
        )

    def canonical_bytes(self) -> bytes:
        self.validate()
        lock = self.canonicalized()
        lines = [
            f"format_version = {lock.format_version}",
            f"model_revision = {json.dumps(lock.model_revision)}",
            f"trace_schema_version = {lock.trace_schema_version}",
        ]
        for entry in lock.bundles:
            tokens = ", ".join(str(token) for token in entry.generated_tokens)
            lines.extend(
                [
                    "",
                    "[[bundles]]",
                    f"case_id = {json.dumps(entry.case_id, ensure_ascii=False)}",
                    f"trace_level = {json.dumps(entry.trace_level.value)}",
                    f"artifact_path = {json.dumps(entry.artifact_path)}",
                    f"bundle_digest = {json.dumps(entry.bundle_digest)}",
                    f"semantic_fingerprint = {json.dumps(entry.semantic_fingerprint)}",
                    f"generated_tokens = [{tokens}]",
                    f"decoded_text = {json.dumps(entry.decoded_text, ensure_ascii=False)}",
                    f"repeat_count = {entry.repeat_count}",
                ]
            )
        return ("\n".join(lines) + "\n").encode("utf-8")

    @classmethod
    def parse_bytes(cls, payload: bytes) -> GoldenLock:
        try:
            decoded = payload.decode("utf-8")
            parsed = tomllib.loads(decoded)
        except (UnicodeError, tomllib.TOMLDecodeError) as error:
            raise GoldenLockError(
                "invalid_toml",
                f"cannot parse golden lock: {error}",
            ) from error
        expected_top = {
            "format_version",
            "model_revision",
            "trace_schema_version",
            "bundles",
        }
        _validate_exact_keys(parsed, expected_top)
        raw_bundles = parsed["bundles"]
        if not isinstance(raw_bundles, list):
            raise GoldenLockError("invalid_bundle", "bundles must be an array of tables")
        expected_entry = {
            "case_id",
            "trace_level",
            "artifact_path",
            "bundle_digest",
            "semantic_fingerprint",
            "generated_tokens",
            "decoded_text",
            "repeat_count",
        }
        entries: list[GoldenEntry] = []
        for raw in raw_bundles:
            if not isinstance(raw, dict):
                raise GoldenLockError("invalid_bundle", "golden entry must be a table")
            _validate_exact_keys(raw, expected_entry)
            raw_tokens = raw["generated_tokens"]
            if not isinstance(raw_tokens, list):
                raise GoldenLockError(
                    "invalid_generated_tokens", "generated_tokens must be an array"
                )
            try:
                trace_level = TraceLevel(raw["trace_level"])
            except (TypeError, ValueError) as error:
                raise GoldenLockError(
                    "invalid_trace_level", "unknown golden trace level"
                ) from error
            entry = GoldenEntry(
                case_id=raw["case_id"],
                trace_level=trace_level,
                artifact_path=raw["artifact_path"],
                bundle_digest=raw["bundle_digest"],
                semantic_fingerprint=raw["semantic_fingerprint"],
                generated_tokens=tuple(raw_tokens),
                decoded_text=raw["decoded_text"],
                repeat_count=raw["repeat_count"],
            )
            entry.validate()
            entries.append(entry)
        lock = cls(
            format_version=parsed["format_version"],
            model_revision=parsed["model_revision"],
            trace_schema_version=parsed["trace_schema_version"],
            bundles=tuple(entries),
        )
        lock.validate()
        return lock.canonicalized()

    @classmethod
    def load(cls, path: Path) -> GoldenLock:
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise GoldenLockError(
                "invalid_toml",
                f"cannot parse golden lock: {error}",
            ) from error
        return cls.parse_bytes(payload)


@dataclass(frozen=True, slots=True)
class GoldenVerificationSummary:
    verified: tuple[str, ...]
    missing: tuple[str, ...]


def load_pinned_golden_lock(path: Path) -> GoldenLock:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise GoldenLockError("missing_golden_lock", str(error)) from error
    actual = blake3(payload).hexdigest()
    if actual != PINNED_GOLDEN_LOCK_DIGEST:
        raise GoldenLockError(
            "wrong_golden_lock_digest",
            "golden lock bytes differ from the reviewed pin",
            details={"actual": actual, "expected": PINNED_GOLDEN_LOCK_DIGEST},
        )
    return GoldenLock.parse_bytes(payload)


def verify_locked_bundles(
    lock: GoldenLock, root: Path, *, require_all: bool
) -> GoldenVerificationSummary:
    lock.validate()
    verified: list[str] = []
    missing: list[str] = []
    for entry in lock.canonicalized().bundles:
        bundle = root / PurePosixPath(entry.artifact_path)
        if not bundle.is_dir() or bundle.is_symlink():
            missing.append(entry.artifact_path)
            continue
        try:
            report = verify_bundle(
                bundle, expected_bundle_digest=entry.bundle_digest
            )
        except (BundleFormatError, BundleIntegrityError, OSError, ValueError) as error:
            raise GoldenLockError(
                "locked_bundle_invalid",
                f"locked bundle failed verification: {entry.artifact_path}: {error}",
                details={"artifact_path": entry.artifact_path},
            ) from error
        if report.case.case_id != entry.case_id:
            raise GoldenLockError(
                "locked_bundle_case_mismatch",
                "locked case ID differs from bundle case ID",
                details={"artifact_path": entry.artifact_path},
            )
        try:
            manifest = json.loads((bundle / "manifest.json").read_bytes())
            bundle_level = manifest["trace_level"]
        except (OSError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise GoldenLockError(
                "locked_bundle_invalid",
                "cannot read verified bundle trace level",
                details={"artifact_path": entry.artifact_path},
            ) from error
        if bundle_level != entry.trace_level.value:
            raise GoldenLockError(
                "locked_bundle_trace_mismatch",
                "locked trace level differs from bundle manifest",
                details={"artifact_path": entry.artifact_path},
            )
        verified.append(entry.artifact_path)

    if missing and require_all:
        raise GoldenLockError(
            "missing_bundle",
            "one or more locked golden bundles are unavailable",
            details={"missing": missing},
        )
    return GoldenVerificationSummary(
        verified=tuple(verified),
        missing=tuple(missing),
    )


def _validate_artifact_path(value: object) -> None:
    if not isinstance(value, str) or not value or value == "." or value.startswith("/"):
        raise GoldenLockError("invalid_artifact_path", "unsafe golden artifact path")
    if "\\" in value or "\x00" in value:
        raise GoldenLockError("invalid_artifact_path", "unsafe golden artifact path")
    parts = value.split("/")
    parsed = PurePosixPath(value)
    if (
        any(part in {"", ".", ".."} for part in parts)
        or parsed.is_absolute()
        or parsed.as_posix() != value
    ):
        raise GoldenLockError("invalid_artifact_path", "unsafe golden artifact path")


def _validate_exact_keys(value: dict[str, Any], expected: set[str]) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise GoldenLockError(
            "unknown_field", f"unknown golden lock fields: {unknown}"
        )
    if missing:
        raise GoldenLockError(
            "missing_field", f"missing golden lock fields: {missing}"
        )
