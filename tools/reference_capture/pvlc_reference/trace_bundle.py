from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping

from blake3 import blake3

from .model_lock import PINNED_MODEL_ID, PINNED_REVISION


TRACE_SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = 1
COMPATIBILITY_SHIM_ID = "paddleocr-vl-1.6/transformers-v5-abi@1"

_DIGEST_RE = re.compile(r"^blake3:[0-9a-f]{64}$")
_SEMANTIC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_CASE_ID_RE = _SEMANTIC_ID_RE
_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9A-Za-z_-]+)+$")

_TASK_CONTRACT = {
    "ocr": ("OCR:", 1_003_520),
    "table": ("Table Recognition:", 1_003_520),
    "formula": ("Formula Recognition:", 1_003_520),
    "chart": ("Chart Recognition:", 1_003_520),
    "spotting": ("Spotting:", 1_605_632),
    "seal": ("Seal Recognition:", 1_003_520),
}
_SOURCE_MEDIA_TYPES = {
    "image/png",
    "image/jpeg",
    "image/x-portable-pixmap",
    "application/x-canonical-rgb8",
}
_TENSOR_DTYPES = {
    "float32",
    "float16",
    "bfloat16",
    "int64",
    "int32",
    "int8",
    "uint8",
}


class BundleFormatError(ValueError):
    def __init__(
        self, code: str, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class BundleIntegrityError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        changed: tuple[str, ...] = (),
        missing: tuple[str, ...] = (),
        unexpected: tuple[str, ...] = (),
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.changed = changed
        self.missing = missing
        self.unexpected = unexpected
        self.details = details or {}


class TraceLevel(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


_REQUIRED_BY_LEVEL: dict[TraceLevel, frozenset[str]] = {
    TraceLevel.L0: frozenset(
        {"case.json", "source-image.bin", "tensor-stats.jsonl"}
    ),
    TraceLevel.L1: frozenset(
        {"case.json", "source-image.bin", "tensor-stats.jsonl", "probes.bin"}
    ),
    TraceLevel.L2: frozenset(
        {
            "case.json",
            "source-image.bin",
            "tensor-stats.jsonl",
            "probes.bin",
            "processor.safetensors",
            "stage-checkpoints.safetensors",
            "token-trace.jsonl",
        }
    ),
    TraceLevel.L3: frozenset(
        {
            "case.json",
            "source-image.bin",
            "tensor-stats.jsonl",
            "probes.bin",
            "processor.safetensors",
            "stage-checkpoints.safetensors",
            "deep-checkpoints.safetensors",
            "token-trace.jsonl",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class CaseSpec:
    case_id: str
    task: str
    prompt: str
    source_image_hash: str
    source_media_type: str
    width: int
    height: int
    max_new_tokens: int
    do_sample: bool
    max_pixels: int

    def validate(self) -> None:
        if not isinstance(self.case_id, str) or _CASE_ID_RE.fullmatch(self.case_id) is None:
            raise BundleFormatError("invalid_case", "case_id is not canonical")
        if self.task not in _TASK_CONTRACT:
            raise BundleFormatError("invalid_case", f"unsupported task: {self.task!r}")
        expected_prompt, expected_pixels = _TASK_CONTRACT[self.task]
        if self.prompt != expected_prompt:
            raise BundleFormatError(
                "invalid_case",
                f"prompt does not match the official {self.task} contract",
            )
        if self.max_pixels != expected_pixels:
            raise BundleFormatError(
                "invalid_case",
                f"pixel budget does not match the official {self.task} contract",
            )
        _validate_digest(self.source_image_hash)
        if self.source_media_type not in _SOURCE_MEDIA_TYPES:
            raise BundleFormatError(
                "invalid_case", f"unsupported source media type: {self.source_media_type!r}"
            )
        for name, value in (
            ("width", self.width),
            ("height", self.height),
            ("max_new_tokens", self.max_new_tokens),
            ("max_pixels", self.max_pixels),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise BundleFormatError("invalid_case", f"{name} must be a positive integer")
        if not isinstance(self.do_sample, bool) or self.do_sample:
            raise BundleFormatError(
                "invalid_case", "M0 golden capture requires do_sample=false"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task": self.task,
            "prompt": self.prompt,
            "source_image_hash": self.source_image_hash,
            "source_media_type": self.source_media_type,
            "width": self.width,
            "height": self.height,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "max_pixels": self.max_pixels,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CaseSpec:
        expected = {
            "case_id",
            "task",
            "prompt",
            "source_image_hash",
            "source_media_type",
            "width",
            "height",
            "max_new_tokens",
            "do_sample",
            "max_pixels",
        }
        _validate_exact_keys(value, expected)
        case = cls(**{key: value[key] for key in expected})
        case.validate()
        return case

    @classmethod
    def load(cls, path: Path) -> CaseSpec:
        return cls.from_dict(_load_json(path))


@dataclass(frozen=True, slots=True)
class CaptureProvenance:
    model_id: str
    model_revision: str
    model_lock_hash: str
    trace_schema_version: int
    capture_tool_version: str
    compatibility_shims: tuple[str, ...]
    python_version: str
    torch_version: str
    transformers_version: str
    device: str
    dtype: str
    deterministic_algorithms: bool

    def validate(self) -> None:
        if self.model_id != PINNED_MODEL_ID or self.model_revision != PINNED_REVISION:
            raise BundleFormatError(
                "wrong_model_identity", "capture provenance identifies another model"
            )
        _validate_digest(self.model_lock_hash)
        if (
            isinstance(self.trace_schema_version, bool)
            or not isinstance(self.trace_schema_version, int)
            or self.trace_schema_version != TRACE_SCHEMA_VERSION
        ):
            raise BundleFormatError(
                "unsupported_trace_schema",
                f"unsupported trace schema: {self.trace_schema_version!r}",
            )
        if self.compatibility_shims != (COMPATIBILITY_SHIM_ID,):
            raise BundleFormatError(
                "invalid_provenance", "unreviewed compatibility shim set"
            )
        for name, version in (
            ("capture_tool_version", self.capture_tool_version),
            ("python_version", self.python_version),
            ("torch_version", self.torch_version),
            ("transformers_version", self.transformers_version),
        ):
            if not isinstance(version, str) or _VERSION_RE.fullmatch(version) is None:
                raise BundleFormatError(
                    "invalid_provenance", f"invalid {name}: {version!r}"
                )
        if self.device not in {"cpu", "mps"}:
            raise BundleFormatError("invalid_provenance", "unsupported capture device")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise BundleFormatError("invalid_provenance", "unsupported capture dtype")
        if not isinstance(self.deterministic_algorithms, bool) or not self.deterministic_algorithms:
            raise BundleFormatError(
                "nondeterministic_capture", "deterministic algorithms must be enabled"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_lock_hash": self.model_lock_hash,
            "trace_schema_version": self.trace_schema_version,
            "capture_tool_version": self.capture_tool_version,
            "compatibility_shims": list(self.compatibility_shims),
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "transformers_version": self.transformers_version,
            "device": self.device,
            "dtype": self.dtype,
            "deterministic_algorithms": self.deterministic_algorithms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CaptureProvenance:
        expected = {
            "model_id",
            "model_revision",
            "model_lock_hash",
            "trace_schema_version",
            "capture_tool_version",
            "compatibility_shims",
            "python_version",
            "torch_version",
            "transformers_version",
            "device",
            "dtype",
            "deterministic_algorithms",
        }
        _validate_exact_keys(value, expected)
        raw_shims = value["compatibility_shims"]
        if not isinstance(raw_shims, list) or not all(
            isinstance(item, str) for item in raw_shims
        ):
            raise BundleFormatError(
                "invalid_provenance", "compatibility_shims must be a string list"
            )
        provenance = cls(
            model_id=value["model_id"],
            model_revision=value["model_revision"],
            model_lock_hash=value["model_lock_hash"],
            trace_schema_version=value["trace_schema_version"],
            capture_tool_version=value["capture_tool_version"],
            compatibility_shims=tuple(raw_shims),
            python_version=value["python_version"],
            torch_version=value["torch_version"],
            transformers_version=value["transformers_version"],
            device=value["device"],
            dtype=value["dtype"],
            deterministic_algorithms=value["deterministic_algorithms"],
        )
        provenance.validate()
        return provenance


@dataclass(frozen=True, slots=True)
class TensorSummary:
    semantic_id: str
    shape: tuple[int, ...]
    dtype: str
    byte_order: str
    layout: str
    contiguous: bool
    minimum: float
    maximum: float
    mean: float
    std: float
    l1: float
    l2: float
    nan_count: int
    inf_count: int
    raw_hash: str
    probe_seed: int

    def validate(self) -> None:
        if (
            not isinstance(self.semantic_id, str)
            or _SEMANTIC_ID_RE.fullmatch(self.semantic_id) is None
        ):
            raise BundleFormatError("invalid_tensor", "invalid semantic_id")
        if not isinstance(self.shape, tuple) or not self.shape:
            raise BundleFormatError("invalid_tensor", "shape must be a non-empty tuple")
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in self.shape
        ):
            raise BundleFormatError("invalid_tensor", "shape dimensions must be positive integers")
        if self.dtype not in _TENSOR_DTYPES:
            raise BundleFormatError("invalid_tensor", "unsupported tensor dtype")
        if self.byte_order not in {"little", "big", "not-applicable"}:
            raise BundleFormatError("invalid_tensor", "unsupported byte order")
        if self.layout not in {"row-major", "packed"}:
            raise BundleFormatError("invalid_tensor", "unsupported tensor layout")
        if not isinstance(self.contiguous, bool):
            raise BundleFormatError("invalid_tensor", "contiguous must be boolean")
        numeric = {
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "l1": self.l1,
            "l2": self.l2,
        }
        for name, value in numeric.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise BundleFormatError("invalid_tensor", f"{name} must be finite")
        if self.minimum > self.maximum:
            raise BundleFormatError("invalid_tensor", "min exceeds max")
        if self.std < 0 or self.l1 < 0 or self.l2 < 0:
            raise BundleFormatError("invalid_tensor", "norms and std must be nonnegative")
        for name, value in (
            ("nan_count", self.nan_count),
            ("inf_count", self.inf_count),
            ("probe_seed", self.probe_seed),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BundleFormatError("invalid_tensor", f"{name} must be a nonnegative integer")
        _validate_digest(self.raw_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_id": self.semantic_id,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "byte_order": self.byte_order,
            "layout": self.layout,
            "contiguous": self.contiguous,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "l1": self.l1,
            "l2": self.l2,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "raw_hash": self.raw_hash,
            "probe_seed": self.probe_seed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TensorSummary:
        expected = {
            "semantic_id",
            "shape",
            "dtype",
            "byte_order",
            "layout",
            "contiguous",
            "min",
            "max",
            "mean",
            "std",
            "l1",
            "l2",
            "nan_count",
            "inf_count",
            "raw_hash",
            "probe_seed",
        }
        _validate_exact_keys(value, expected)
        raw_shape = value["shape"]
        shape = tuple(raw_shape) if isinstance(raw_shape, list) else raw_shape
        summary = cls(
            semantic_id=value["semantic_id"],
            shape=shape,
            dtype=value["dtype"],
            byte_order=value["byte_order"],
            layout=value["layout"],
            contiguous=value["contiguous"],
            minimum=value["min"],
            maximum=value["max"],
            mean=value["mean"],
            std=value["std"],
            l1=value["l1"],
            l2=value["l2"],
            nan_count=value["nan_count"],
            inf_count=value["inf_count"],
            raw_hash=value["raw_hash"],
            probe_seed=value["probe_seed"],
        )
        summary.validate()
        return summary


@dataclass(frozen=True, slots=True)
class BundleBuildResult:
    bundle_digest: str
    artifacts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BundleVerificationReport:
    bundle_digest: str
    case: CaseSpec
    provenance: CaptureProvenance
    verified_artifacts: tuple[str, ...]


class GoldenBundleBuilder:
    def __init__(
        self,
        root: Path,
        case: CaseSpec,
        trace_level: TraceLevel,
        provenance: CaptureProvenance,
    ) -> None:
        case.validate()
        provenance.validate()
        if not isinstance(trace_level, TraceLevel):
            raise BundleFormatError("invalid_trace_level", "invalid trace level")
        _assert_safe_output_root(root)
        self.root = root
        self.case = case
        self.trace_level = trace_level
        self.provenance = provenance
        self._artifacts: dict[str, bytes] = {}
        self._tensor_summaries: list[TensorSummary] = []
        self._semantic_ids: set[str] = set()
        self._finished = False

    def add_bytes(self, path: str, payload: bytes) -> None:
        self._ensure_open()
        _validate_bundle_path(path)
        if path in {"manifest.json", "hashes.json", "case.json", "tensor-stats.jsonl"}:
            raise BundleFormatError("reserved_path", f"builder owns {path}")
        if path in self._artifacts:
            raise BundleFormatError("duplicate_artifact", f"duplicate artifact: {path}")
        if not isinstance(payload, bytes):
            raise BundleFormatError("invalid_artifact", "artifact payload must be bytes")
        self._artifacts[path] = payload

    def add_tensor_summaries(self, summaries: Iterable[TensorSummary]) -> None:
        self._ensure_open()
        for summary in summaries:
            if not isinstance(summary, TensorSummary):
                raise BundleFormatError("invalid_tensor", "expected TensorSummary")
            summary.validate()
            if summary.semantic_id in self._semantic_ids:
                raise BundleFormatError(
                    "duplicate_semantic_id",
                    f"duplicate semantic ID: {summary.semantic_id}",
                )
            self._semantic_ids.add(summary.semantic_id)
            self._tensor_summaries.append(summary)

    def finish(self) -> BundleBuildResult:
        self._ensure_open()
        artifacts = dict(self._artifacts)
        if self._tensor_summaries:
            artifacts["tensor-stats.jsonl"] = b"".join(
                canonical_json_bytes(summary.to_dict())
                for summary in self._tensor_summaries
            )

        present = set(artifacts) | {"case.json"}
        missing = sorted(_REQUIRED_BY_LEVEL[self.trace_level] - present)
        if missing:
            raise BundleFormatError(
                "incomplete_bundle",
                f"bundle is missing required artifacts: {missing}",
                details={"missing": missing},
            )

        source = artifacts["source-image.bin"]
        actual_source_hash = f"blake3:{blake3(source).hexdigest()}"
        if actual_source_hash != self.case.source_image_hash:
            raise BundleFormatError(
                "source_hash_mismatch", "source image bytes do not match CaseSpec"
            )
        _validate_source_shape(self.case, source)

        case_bytes = canonical_json_bytes(self.case.to_dict())
        required = sorted(_REQUIRED_BY_LEVEL[self.trace_level])
        manifest = {
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "case_id": self.case.case_id,
            "provenance": self.provenance.to_dict(),
            "required_artifacts": required,
            "trace_level": self.trace_level.value,
        }
        manifest_bytes = canonical_json_bytes(manifest)
        payloads = {
            "case.json": case_bytes,
            **artifacts,
            "manifest.json": manifest_bytes,
        }
        hash_entries = {
            name: {
                "blake3": blake3(payload).hexdigest(),
                "size": len(payload),
            }
            for name, payload in sorted(payloads.items())
        }
        hashes_bytes = canonical_json_bytes(
            {
                "algorithm": "blake3",
                "artifacts": hash_entries,
                "format_version": 1,
            }
        )
        payloads["hashes.json"] = hashes_bytes
        bundle_digest = f"blake3:{blake3(hashes_bytes).hexdigest()}"

        self._write_atomically(payloads)
        self._finished = True
        return BundleBuildResult(
            bundle_digest=bundle_digest,
            artifacts=tuple(sorted(name for name in payloads if name != "hashes.json")),
        )

    def _ensure_open(self) -> None:
        if self._finished:
            raise BundleFormatError("builder_finished", "bundle builder is already finished")

    def _write_atomically(self, payloads: Mapping[str, bytes]) -> None:
        _assert_safe_output_root(self.root)
        self.root.parent.mkdir(parents=True, exist_ok=True)
        if self.root.exists() or self.root.is_symlink():
            raise BundleFormatError("output_exists", f"output already exists: {self.root}")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{self.root.name}.tmp-", dir=self.root.parent)
        )
        try:
            for name, payload in payloads.items():
                target = staging / PurePosixPath(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            os.replace(staging, self.root)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def verify_bundle(
    root: Path, *, expected_bundle_digest: str | None = None
) -> BundleVerificationReport:
    if root.is_symlink() or not root.is_dir():
        raise BundleIntegrityError("unsafe_bundle", "bundle root is not a regular directory")
    hashes_path = root / "hashes.json"
    if not hashes_path.is_file() or hashes_path.is_symlink():
        raise BundleIntegrityError(
            "missing_hash_manifest",
            "bundle hashes.json is missing or unsafe",
            missing=("hashes.json",),
        )
    hashes_bytes = hashes_path.read_bytes()
    bundle_digest = f"blake3:{blake3(hashes_bytes).hexdigest()}"
    if expected_bundle_digest is not None:
        _validate_digest(expected_bundle_digest)
        if bundle_digest != expected_bundle_digest:
            raise BundleIntegrityError(
                "bundle_digest_mismatch",
                "bundle digest does not match its external pin",
                changed=("hashes.json",),
                details={"actual": bundle_digest, "expected": expected_bundle_digest},
            )

    hashes = _load_json_bytes(hashes_bytes, "hashes.json")
    if hashes_bytes != canonical_json_bytes(hashes):
        raise BundleFormatError("noncanonical_json", "hashes.json is not canonical")
    _validate_exact_keys(hashes, {"algorithm", "artifacts", "format_version"})
    if hashes["algorithm"] != "blake3" or hashes["format_version"] != 1:
        raise BundleFormatError("invalid_hash_manifest", "unsupported hashes.json format")
    raw_entries = hashes["artifacts"]
    if not isinstance(raw_entries, dict) or not raw_entries:
        raise BundleFormatError("invalid_hash_manifest", "empty artifact hash map")

    expected_entries: dict[str, tuple[int, str]] = {}
    for name, metadata in raw_entries.items():
        _validate_bundle_path(name)
        if not isinstance(metadata, dict) or set(metadata) != {"blake3", "size"}:
            raise BundleFormatError("invalid_hash_manifest", "invalid artifact hash entry")
        raw_hash = metadata["blake3"]
        size = metadata["size"]
        digest = f"blake3:{raw_hash}" if isinstance(raw_hash, str) else raw_hash
        _validate_digest(digest)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BundleFormatError("invalid_hash_manifest", "invalid artifact size")
        expected_entries[name] = (size, digest)

    discovered, unsafe = _discover_bundle_files(root)
    expected_names = set(expected_entries)
    missing = tuple(sorted(expected_names - discovered))
    unexpected = tuple(sorted(discovered - expected_names - {"hashes.json"}))
    changed: set[str] = set(unsafe & expected_names)
    for name in sorted(expected_names & discovered - unsafe):
        target = root / PurePosixPath(name)
        payload = target.read_bytes()
        expected_size, expected_hash = expected_entries[name]
        if len(payload) != expected_size or f"blake3:{blake3(payload).hexdigest()}" != expected_hash:
            changed.add(name)
    if missing or unexpected or changed:
        raise BundleIntegrityError(
            "artifact_integrity_failure",
            "one or more bundle artifacts failed integrity verification",
            changed=tuple(sorted(changed)),
            missing=missing,
            unexpected=unexpected,
        )

    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = _load_json_bytes(manifest_bytes, "manifest.json")
    if manifest_bytes != canonical_json_bytes(manifest):
        raise BundleFormatError("noncanonical_json", "manifest.json is not canonical")
    _validate_exact_keys(
        manifest,
        {
            "bundle_schema_version",
            "case_id",
            "provenance",
            "required_artifacts",
            "trace_level",
        },
    )
    if manifest["bundle_schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise BundleFormatError("unsupported_bundle_schema", "unsupported bundle schema")
    try:
        trace_level = TraceLevel(manifest["trace_level"])
    except (TypeError, ValueError) as error:
        raise BundleFormatError("invalid_trace_level", "invalid trace level") from error
    required = sorted(_REQUIRED_BY_LEVEL[trace_level])
    if manifest["required_artifacts"] != required:
        raise BundleFormatError("invalid_manifest", "required artifact contract differs")

    case_bytes = (root / "case.json").read_bytes()
    case_data = _load_json_bytes(case_bytes, "case.json")
    if case_bytes != canonical_json_bytes(case_data):
        raise BundleFormatError("noncanonical_json", "case.json is not canonical")
    case = CaseSpec.from_dict(case_data)
    if manifest["case_id"] != case.case_id:
        raise BundleFormatError("invalid_manifest", "manifest case_id differs from case.json")
    provenance = CaptureProvenance.from_dict(manifest["provenance"])

    source = (root / "source-image.bin").read_bytes()
    actual_source_hash = f"blake3:{blake3(source).hexdigest()}"
    if actual_source_hash != case.source_image_hash:
        raise BundleIntegrityError(
            "source_hash_mismatch",
            "source image does not match the hash in case.json",
            changed=("case.json", "source-image.bin"),
        )
    _validate_source_shape(case, source)
    _validate_tensor_stats(root / "tensor-stats.jsonl")
    return BundleVerificationReport(
        bundle_digest=bundle_digest,
        case=case,
        provenance=provenance,
        verified_artifacts=tuple(sorted(expected_names)),
    )


def _validate_tensor_stats(path: Path) -> None:
    semantic_ids: set[str] = set()
    for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line:
            raise BundleFormatError("invalid_tensor_stats", "blank JSONL record")
        record = _load_json_bytes(line, f"tensor-stats.jsonl:{line_number}")
        if line + b"\n" != canonical_json_bytes(record):
            raise BundleFormatError("noncanonical_json", "tensor stats record is not canonical")
        summary = TensorSummary.from_dict(record)
        if summary.semantic_id in semantic_ids:
            raise BundleFormatError("duplicate_semantic_id", "duplicate tensor semantic ID")
        semantic_ids.add(summary.semantic_id)


def _validate_exact_keys(value: Mapping[str, Any], expected: set[str]) -> None:
    if not isinstance(value, Mapping):
        raise BundleFormatError("invalid_schema", "expected an object")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise BundleFormatError(
            "missing_field", f"missing fields: {missing}", details={"fields": missing}
        )
    if unknown:
        raise BundleFormatError(
            "unknown_field", f"unknown fields: {unknown}", details={"fields": unknown}
        )


def _validate_digest(value: Any) -> None:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise BundleFormatError("invalid_digest", f"invalid BLAKE3 digest: {value!r}")


def canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise BundleFormatError("invalid_json_value", str(error)) from error
    return text.encode("utf-8") + b"\n"


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise BundleFormatError("invalid_json", f"cannot parse {path}: {error}") from error


def _load_json_bytes(payload: bytes, name: str) -> Any:
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise BundleFormatError("invalid_json", f"cannot parse {name}: {error}") from error


def _validate_bundle_path(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value == "."
        or value.startswith("/")
        or "\\" in value
        or "\x00" in value
    ):
        raise BundleFormatError("invalid_path", f"invalid artifact path: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise BundleFormatError("invalid_path", f"noncanonical artifact path: {value!r}")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed.as_posix() != value:
        raise BundleFormatError("invalid_path", f"invalid artifact path: {value!r}")


def _assert_safe_output_root(root: Path) -> None:
    absolute = root.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if not os.path.lexists(current):
            continue
        if stat.S_ISLNK(os.lstat(current).st_mode):
            raise BundleFormatError(
                "unsafe_output_path", f"output path crosses a symlink: {current}"
            )


def _validate_source_shape(case: CaseSpec, payload: bytes) -> None:
    if case.source_media_type == "application/x-canonical-rgb8":
        if len(payload) != case.width * case.height * 3:
            raise BundleFormatError("invalid_source_shape", "RGB8 byte length mismatch")
    elif case.source_media_type == "image/png":
        if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[12:16] != b"IHDR":
            raise BundleFormatError("invalid_source_image", "invalid PNG header")
        width, height = struct.unpack(">II", payload[16:24])
        if (width, height) != (case.width, case.height):
            raise BundleFormatError("invalid_source_shape", "PNG dimensions differ from case")
    elif case.source_media_type == "image/x-portable-pixmap":
        header = payload.split(maxsplit=4)
        if len(header) < 4 or header[0] not in {b"P3", b"P6"}:
            raise BundleFormatError("invalid_source_image", "invalid PPM header")
        if (int(header[1]), int(header[2])) != (case.width, case.height):
            raise BundleFormatError("invalid_source_shape", "PPM dimensions differ from case")


def _discover_bundle_files(root: Path) -> tuple[set[str], set[str]]:
    discovered: set[str] = set()
    unsafe: set[str] = set()

    def visit(directory: Path, prefix: str) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = f"{prefix}/{entry.name}" if prefix else entry.name
                if entry.is_dir(follow_symlinks=False):
                    visit(Path(entry.path), relative)
                else:
                    discovered.add(relative)
                    mode = os.lstat(entry.path).st_mode
                    if not stat.S_ISREG(mode):
                        unsafe.add(relative)

    visit(root, "")
    return discovered, unsafe
