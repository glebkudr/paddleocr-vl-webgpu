from __future__ import annotations

import json
import os
import re
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from blake3 import blake3


PINNED_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
PINNED_REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e"
PINNED_LOCK_DIGEST = "c2ffa46821fdda1a7e4cd6bad63cdcdc2f775b8fed250b0679cc7d5411577d10"
PINNED_FILE_NAMES = frozenset(
    {
        ".gitattributes",
        "LICENSE",
        "README.md",
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "configuration_paddleocr_vl.py",
        "generation_config.json",
        "image_processing_paddleocr_vl.py",
        "inference.yml",
        "model.safetensors",
        "modeling_paddleocr_vl.py",
        "preprocessor_config.json",
        "processing_paddleocr_vl.py",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    }
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_BLAKE3_RE = re.compile(r"^blake3:[0-9a-f]{64}$")


class LockFormatError(ValueError):
    def __init__(
        self, code: str, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    changed: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    unsafe: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not (self.changed or self.missing or self.unexpected or self.unsafe)


class IntegrityError(RuntimeError):
    def __init__(self, report: IntegrityReport) -> None:
        super().__init__(
            "model snapshot integrity check failed: "
            f"changed={list(report.changed)}, missing={list(report.missing)}, "
            f"unexpected={list(report.unexpected)}, unsafe={list(report.unsafe)}"
        )
        self.report = report


@dataclass(frozen=True, slots=True)
class ModelFile:
    path: str
    digest: str
    size: int

    def validate(self) -> None:
        _validate_relative_path(self.path)
        if not isinstance(self.digest, str) or _BLAKE3_RE.fullmatch(self.digest) is None:
            raise LockFormatError(
                "invalid_digest",
                f"invalid BLAKE3 digest for {self.path!r}",
                details={"path": self.path},
            )
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise LockFormatError(
                "invalid_size",
                f"invalid byte size for {self.path!r}",
                details={"path": self.path, "size": self.size},
            )


@dataclass(frozen=True, slots=True)
class ModelLock:
    format_version: int
    model_id: str
    revision: str
    compiler_model_abi: int
    files: tuple[ModelFile, ...]

    def validate(self) -> None:
        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, int)
            or self.format_version != 1
        ):
            raise LockFormatError(
                "unsupported_format_version",
                f"unsupported model lock format version: {self.format_version!r}",
            )
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise LockFormatError("invalid_model_id", "model_id must be a non-empty string")
        if not isinstance(self.revision, str) or _SHA_RE.fullmatch(self.revision) is None:
            raise LockFormatError(
                "invalid_revision",
                "revision must be a full lowercase commit SHA",
            )
        if (
            isinstance(self.compiler_model_abi, bool)
            or not isinstance(self.compiler_model_abi, int)
            or self.compiler_model_abi < 1
        ):
            raise LockFormatError(
                "invalid_abi", "compiler_model_abi must be a positive integer"
            )
        if not self.files:
            raise LockFormatError("empty_inventory", "model lock file inventory is empty")

        seen: set[str] = set()
        for entry in self.files:
            entry.validate()
            if entry.path in seen:
                raise LockFormatError(
                    "duplicate_path",
                    f"duplicate model path: {entry.path}",
                    details={"path": entry.path},
                )
            seen.add(entry.path)

    def canonicalized(self) -> ModelLock:
        return ModelLock(
            format_version=self.format_version,
            model_id=self.model_id,
            revision=self.revision,
            compiler_model_abi=self.compiler_model_abi,
            files=tuple(sorted(self.files, key=lambda entry: entry.path)),
        )

    def canonical_bytes(self) -> bytes:
        self.validate()
        canonical = self.canonicalized()
        lines = [
            f"format_version = {canonical.format_version}",
            f"model_id = {json.dumps(canonical.model_id, ensure_ascii=False)}",
            f"revision = {json.dumps(canonical.revision)}",
            f"compiler_model_abi = {canonical.compiler_model_abi}",
            "",
            "[files]",
        ]
        for entry in canonical.files:
            raw_digest = entry.digest.removeprefix("blake3:")
            lines.append(
                f"{json.dumps(entry.path, ensure_ascii=False)} = "
                f'{{ blake3 = "{raw_digest}", size = {entry.size} }}'
            )
        return ("\n".join(lines) + "\n").encode("utf-8")

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.canonical_bytes())

    @classmethod
    def load(cls, path: Path) -> ModelLock:
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
            raise LockFormatError("invalid_toml", f"cannot parse model lock: {error}") from error

        expected_top = {
            "format_version",
            "model_id",
            "revision",
            "compiler_model_abi",
            "files",
        }
        actual_top = set(payload)
        unknown = sorted(actual_top - expected_top)
        missing = sorted(expected_top - actual_top)
        if unknown:
            raise LockFormatError(
                "unknown_field",
                f"unknown model lock fields: {unknown}",
                details={"fields": unknown},
            )
        if missing:
            raise LockFormatError(
                "missing_field",
                f"missing model lock fields: {missing}",
                details={"fields": missing},
            )

        raw_files = payload["files"]
        if not isinstance(raw_files, dict):
            raise LockFormatError("invalid_file_schema", "files must be a TOML table")

        entries: list[ModelFile] = []
        for name, metadata in raw_files.items():
            if not isinstance(metadata, dict) or set(metadata) != {"blake3", "size"}:
                raise LockFormatError(
                    "invalid_file_schema",
                    f"file metadata for {name!r} must contain only blake3 and size",
                    details={"path": name},
                )
            digest_value = metadata["blake3"]
            digest = (
                f"blake3:{digest_value}" if isinstance(digest_value, str) else digest_value
            )
            entries.append(ModelFile(path=name, digest=digest, size=metadata["size"]))

        lock = cls(
            format_version=payload["format_version"],
            model_id=payload["model_id"],
            revision=payload["revision"],
            compiler_model_abi=payload["compiler_model_abi"],
            files=tuple(entries),
        )
        lock.validate()
        return lock.canonicalized()


@dataclass(frozen=True, slots=True)
class VerifiedModel:
    model_id: str
    revision: str
    lock_digest: str
    snapshot_digest: str
    verified_files: tuple[str, ...]


def load_pinned_paddleocr_vl_16_lock(path: Path) -> ModelLock:
    lock = ModelLock.load(path)
    if lock.model_id != PINNED_MODEL_ID or lock.revision != PINNED_REVISION:
        raise LockFormatError(
            "wrong_model_identity",
            "model lock does not identify the pinned PaddleOCR-VL-1.6 snapshot",
            details={"model_id": lock.model_id, "revision": lock.revision},
        )
    actual_names = {entry.path for entry in lock.files}
    if actual_names != PINNED_FILE_NAMES:
        raise LockFormatError(
            "wrong_model_inventory",
            "model lock inventory differs from the pinned upstream snapshot",
            details={
                "missing": sorted(PINNED_FILE_NAMES - actual_names),
                "unexpected": sorted(actual_names - PINNED_FILE_NAMES),
            },
        )
    actual_lock_digest = blake3(path.read_bytes()).hexdigest()
    if actual_lock_digest != PINNED_LOCK_DIGEST:
        raise LockFormatError(
            "wrong_lock_digest",
            "pinned model lock bytes do not match the reviewed lock",
            details={"actual": actual_lock_digest, "expected": PINNED_LOCK_DIGEST},
        )
    return lock


def verify_model_directory(lock: ModelLock, snapshot: Path) -> VerifiedModel:
    lock.validate()
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise IntegrityError(IntegrityReport(unsafe=(".",)))

    changed: set[str] = set()
    missing: set[str] = set()
    unsafe: set[str] = set()
    actual_metadata: dict[str, tuple[int, str]] = {}

    for entry in lock.files:
        status = _inspect_locked_path(snapshot, entry.path)
        if status == "missing":
            missing.add(entry.path)
            continue
        if status == "unsafe":
            unsafe.add(entry.path)
            continue

        target = snapshot / PurePosixPath(entry.path)
        actual_size = target.stat().st_size
        actual_digest = _digest_file(target)
        actual_metadata[entry.path] = (actual_size, actual_digest)
        if actual_size != entry.size or actual_digest != entry.digest:
            changed.add(entry.path)

    locked_paths = {entry.path for entry in lock.files}
    locked_prefixes = {
        "/".join(parts[:index])
        for path in locked_paths
        for parts in [path.split("/")]
        for index in range(1, len(parts))
    }
    discovered = _discover_snapshot_entries(snapshot)
    unexpected = {
        path
        for path in discovered
        if path not in locked_paths and path not in locked_prefixes
    }

    report = IntegrityReport(
        changed=tuple(sorted(changed)),
        missing=tuple(sorted(missing)),
        unexpected=tuple(sorted(unexpected)),
        unsafe=tuple(sorted(unsafe)),
    )
    if not report.ok:
        raise IntegrityError(report)

    snapshot_description = [
        {
            "path": path,
            "size": actual_metadata[path][0],
            "blake3": actual_metadata[path][1].removeprefix("blake3:"),
        }
        for path in sorted(actual_metadata)
    ]
    snapshot_payload = json.dumps(
        snapshot_description, sort_keys=True, separators=(",", ":")
    ).encode()
    return VerifiedModel(
        model_id=lock.model_id,
        revision=lock.revision,
        lock_digest=f"blake3:{blake3(lock.canonical_bytes()).hexdigest()}",
        snapshot_digest=f"blake3:{blake3(snapshot_payload).hexdigest()}",
        verified_files=tuple(sorted(locked_paths)),
    )


def _validate_relative_path(value: str) -> None:
    if not isinstance(value, str) or not value or value == ".":
        raise LockFormatError("invalid_path", "model file path must be a safe relative path")
    if "\\" in value or "\x00" in value or value.startswith("/"):
        raise LockFormatError(
            "invalid_path", f"model file path is not a safe relative path: {value!r}"
        )
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise LockFormatError(
            "invalid_path", f"model file path is not canonical: {value!r}"
        )
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed.as_posix() != value:
        raise LockFormatError(
            "invalid_path", f"model file path is not a safe relative path: {value!r}"
        )


def _digest_file(path: Path) -> str:
    hasher = blake3()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return f"blake3:{hasher.hexdigest()}"


def _inspect_locked_path(root: Path, relative: str) -> str:
    current = root
    parts = relative.split("/")
    for index, part in enumerate(parts):
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            return "missing"
        if stat.S_ISLNK(mode):
            return "unsafe"
        if index < len(parts) - 1:
            if not stat.S_ISDIR(mode):
                return "unsafe"
        elif not stat.S_ISREG(mode):
            return "unsafe"
    return "regular"


def _discover_snapshot_entries(root: Path) -> set[str]:
    discovered: set[str] = set()

    def visit(directory: Path, prefix: str) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = f"{prefix}/{entry.name}" if prefix else entry.name
                if entry.is_dir(follow_symlinks=False):
                    visit(Path(entry.path), relative)
                else:
                    discovered.add(relative)

    visit(root, "")
    return discovered
