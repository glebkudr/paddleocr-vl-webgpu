from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from blake3 import blake3
from safetensors.torch import save as save_safetensors

from .capture import GenerationTrace
from .trace_bundle import (
    BundleBuildResult,
    CaptureProvenance,
    CaseSpec,
    GoldenBundleBuilder,
    TensorSummary,
    TraceLevel,
    canonical_json_bytes,
)


PROBE_MAGIC = b"PVLCPRB1"
_PROBE_HEADER = struct.Struct("<QI")
_PROBE_RECORD_PREFIX = struct.Struct("<H")
_PROBE_TYPE_AND_RANK = struct.Struct("<BB")
_PROBE_SAMPLE_COUNT = struct.Struct("<I")
_PROBE_SAMPLE = struct.Struct("<Qd")

_TORCH_TO_SCHEMA_DTYPE = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int8: "int8",
    torch.uint8: "uint8",
}
_SCHEMA_DTYPE_TO_CODE = {
    "float32": 1,
    "float16": 2,
    "bfloat16": 3,
    "int64": 4,
    "int32": 5,
    "int8": 6,
    "uint8": 7,
}
_CODE_TO_SCHEMA_DTYPE = {
    code: dtype for dtype, code in _SCHEMA_DTYPE_TO_CODE.items()
}


class CaptureArtifactError(ValueError):
    def __init__(
        self, code: str, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True, slots=True)
class CapturedArtifacts:
    processor_tensors: Mapping[str, torch.Tensor]
    stage_tensors: Mapping[str, torch.Tensor]
    deep_tensors: Mapping[str, torch.Tensor]
    token_trace: GenerationTrace | None


@dataclass(frozen=True, slots=True)
class ProbeRecord:
    semantic_id: str
    dtype: str
    shape: tuple[int, ...]
    indices: tuple[int, ...]
    values: tuple[float, ...]


def _canonical_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise CaptureArtifactError("invalid_tensor", "capture value is not a tensor")
    if tensor.dtype not in _TORCH_TO_SCHEMA_DTYPE:
        raise CaptureArtifactError(
            "unsupported_tensor_dtype", f"unsupported tensor dtype: {tensor.dtype}"
        )
    if tensor.ndim == 0 or tensor.numel() == 0 or any(axis <= 0 for axis in tensor.shape):
        raise CaptureArtifactError(
            "invalid_tensor_shape", "captured tensors must be nonempty and non-scalar"
        )
    return tensor.detach().cpu().contiguous()


def _storage_bytes(tensor: torch.Tensor) -> bytes:
    canonical = _canonical_tensor(tensor)
    return canonical.view(torch.uint8).numpy().tobytes()


def summarize_tensor(
    semantic_id: str, tensor: torch.Tensor, *, probe_seed: int
) -> TensorSummary:
    if (
        isinstance(probe_seed, bool)
        or not isinstance(probe_seed, int)
        or probe_seed < 0
        or probe_seed > (2**64 - 1)
    ):
        raise CaptureArtifactError("invalid_probe_seed", "probe seed must be uint64")
    canonical = _canonical_tensor(tensor)
    numeric = canonical.to(torch.float64).reshape(-1).numpy()
    finite = numeric[np.isfinite(numeric)]
    nan_count = int(np.isnan(numeric).sum())
    inf_count = int(np.isinf(numeric).sum())
    if finite.size:
        minimum = float(finite.min())
        maximum = float(finite.max())
        mean = float(finite.mean())
        std = float(finite.std())
        l1 = float(np.abs(finite).sum())
        l2 = float(np.linalg.vector_norm(finite))
    else:
        minimum = maximum = mean = std = l1 = l2 = 0.0

    summary = TensorSummary(
        semantic_id=semantic_id,
        shape=tuple(int(axis) for axis in canonical.shape),
        dtype=_TORCH_TO_SCHEMA_DTYPE[canonical.dtype],
        byte_order=sys.byteorder,
        layout="row-major",
        contiguous=True,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        l1=l1,
        l2=l2,
        nan_count=nan_count,
        inf_count=inf_count,
        raw_hash=f"blake3:{blake3(_storage_bytes(canonical)).hexdigest()}",
        probe_seed=probe_seed,
    )
    summary.validate()
    return summary


def _probe_indices(
    semantic_id: str, element_count: int, sample_count: int, seed: int
) -> tuple[int, ...]:
    target = min(element_count, sample_count)
    chosen: set[int] = set()
    for index in (0, element_count // 2, element_count - 1):
        if len(chosen) == target:
            break
        chosen.add(index)
    counter = 0
    seed_bytes = struct.pack("<Q", seed)
    name_bytes = semantic_id.encode("utf-8")
    while len(chosen) < target:
        material = seed_bytes + name_bytes + struct.pack("<Q", counter)
        candidate = int.from_bytes(
            blake3(material).digest(length=8), "little"
        ) % element_count
        chosen.add(candidate)
        counter += 1
    return tuple(sorted(chosen))


def _validated_tensor_mapping(
    tensors: Mapping[str, torch.Tensor], *, allow_empty: bool = False
) -> dict[str, torch.Tensor]:
    if not isinstance(tensors, Mapping) or (not tensors and not allow_empty):
        raise CaptureArtifactError(
            "missing_tensors", "captured tensor mapping must be nonempty"
        )
    canonical: dict[str, torch.Tensor] = {}
    for semantic_id, tensor in sorted(tensors.items()):
        if not isinstance(semantic_id, str):
            raise CaptureArtifactError("invalid_semantic_id", "tensor ID must be a string")
        normalized = _canonical_tensor(tensor)
        # TensorSummary owns the stable SemanticId grammar validation.
        summarize_tensor(semantic_id, normalized, probe_seed=0)
        canonical[semantic_id] = normalized
    return canonical


def build_probe_bundle(
    tensors: Mapping[str, torch.Tensor], *, seed: int, samples_per_tensor: int
) -> bytes:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > (2**64 - 1)
    ):
        raise CaptureArtifactError("invalid_probe_seed", "probe seed must be uint64")
    if (
        isinstance(samples_per_tensor, bool)
        or not isinstance(samples_per_tensor, int)
        or samples_per_tensor <= 0
    ):
        raise CaptureArtifactError(
            "invalid_probe_count", "samples_per_tensor must be positive"
        )
    canonical = _validated_tensor_mapping(tensors)
    payload = bytearray(PROBE_MAGIC)
    payload.extend(_PROBE_HEADER.pack(seed, len(canonical)))
    for semantic_id, tensor in canonical.items():
        name = semantic_id.encode("utf-8")
        if len(name) > (2**16 - 1) or tensor.ndim > (2**8 - 1):
            raise CaptureArtifactError(
                "probe_schema_overflow", "probe metadata exceeds the binary schema"
            )
        dtype = _TORCH_TO_SCHEMA_DTYPE[tensor.dtype]
        flat = tensor.to(torch.float64).reshape(-1)
        indices = _probe_indices(
            semantic_id, flat.numel(), samples_per_tensor, seed
        )
        payload.extend(_PROBE_RECORD_PREFIX.pack(len(name)))
        payload.extend(name)
        payload.extend(_PROBE_TYPE_AND_RANK.pack(_SCHEMA_DTYPE_TO_CODE[dtype], tensor.ndim))
        for axis in tensor.shape:
            payload.extend(struct.pack("<Q", int(axis)))
        payload.extend(_PROBE_SAMPLE_COUNT.pack(len(indices)))
        for index in indices:
            payload.extend(_PROBE_SAMPLE.pack(index, float(flat[index].item())))
    return bytes(payload)


class _ProbeReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def take(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.payload):
            raise CaptureArtifactError(
                "invalid_probe_bundle", "probe bundle is truncated"
            )
        value = self.payload[self.offset : end]
        self.offset = end
        return value

    def unpack(self, schema: struct.Struct) -> tuple[Any, ...]:
        return schema.unpack(self.take(schema.size))


def parse_probe_bundle(payload: bytes) -> tuple[ProbeRecord, ...]:
    if not isinstance(payload, bytes):
        raise CaptureArtifactError("invalid_probe_bundle", "probe bundle must be bytes")
    reader = _ProbeReader(payload)
    if reader.take(len(PROBE_MAGIC)) != PROBE_MAGIC:
        raise CaptureArtifactError("invalid_probe_bundle", "unsupported probe magic")
    _, record_count = reader.unpack(_PROBE_HEADER)
    records: list[ProbeRecord] = []
    previous_id: str | None = None
    for _ in range(record_count):
        (name_size,) = reader.unpack(_PROBE_RECORD_PREFIX)
        try:
            semantic_id = reader.take(name_size).decode("utf-8")
        except UnicodeDecodeError as error:
            raise CaptureArtifactError(
                "invalid_probe_bundle", "probe SemanticId is not UTF-8"
            ) from error
        dtype_code, rank = reader.unpack(_PROBE_TYPE_AND_RANK)
        if dtype_code not in _CODE_TO_SCHEMA_DTYPE or rank == 0:
            raise CaptureArtifactError(
                "invalid_probe_bundle", "unknown probe dtype or rank"
            )
        shape = tuple(struct.unpack("<Q", reader.take(8))[0] for _ in range(rank))
        if any(axis == 0 for axis in shape):
            raise CaptureArtifactError("invalid_probe_bundle", "zero probe dimension")
        (sample_count,) = reader.unpack(_PROBE_SAMPLE_COUNT)
        indices: list[int] = []
        values: list[float] = []
        element_count = int(np.prod(shape))
        for _ in range(sample_count):
            index, value = reader.unpack(_PROBE_SAMPLE)
            if index >= element_count:
                raise CaptureArtifactError(
                    "invalid_probe_bundle", "probe index is out of bounds"
                )
            indices.append(index)
            values.append(value)
        if tuple(indices) != tuple(sorted(set(indices))):
            raise CaptureArtifactError(
                "invalid_probe_bundle", "probe indices are not canonical"
            )
        if previous_id is not None and semantic_id <= previous_id:
            raise CaptureArtifactError(
                "invalid_probe_bundle", "probe records are not canonically ordered"
            )
        previous_id = semantic_id
        records.append(
            ProbeRecord(
                semantic_id=semantic_id,
                dtype=_CODE_TO_SCHEMA_DTYPE[dtype_code],
                shape=shape,
                indices=tuple(indices),
                values=tuple(values),
            )
        )
    if reader.offset != len(payload):
        raise CaptureArtifactError(
            "invalid_probe_bundle", "probe bundle has trailing bytes"
        )
    return tuple(records)


def serialize_safetensors(tensors: Mapping[str, torch.Tensor]) -> bytes:
    canonical = _validated_tensor_mapping(tensors)
    owned = {name: tensor.clone() for name, tensor in canonical.items()}
    return save_safetensors(owned)


def serialize_token_trace(trace: GenerationTrace) -> bytes:
    if not isinstance(trace, GenerationTrace):
        raise CaptureArtifactError(
            "missing_token_trace", "stage capture requires a generation trace"
        )
    trace.validate()
    return b"".join(canonical_json_bytes(step.to_dict()) for step in trace.steps)


def _merge_tensor_groups(
    *groups: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for group in groups:
        for semantic_id, tensor in group.items():
            if semantic_id in merged:
                raise CaptureArtifactError(
                    "duplicate_semantic_id",
                    f"tensor appears in multiple capture groups: {semantic_id}",
                )
            merged[semantic_id] = tensor
    return dict(sorted(merged.items()))


def export_golden_bundle(
    *,
    root: Any,
    case: CaseSpec,
    source_image: bytes,
    provenance: CaptureProvenance,
    trace_level: TraceLevel,
    captured: CapturedArtifacts,
    probe_seed: int,
) -> BundleBuildResult:
    if not isinstance(captured, CapturedArtifacts):
        raise CaptureArtifactError("invalid_capture", "missing captured artifacts")
    processor = _validated_tensor_mapping(
        captured.processor_tensors, allow_empty=trace_level in {TraceLevel.L0, TraceLevel.L1}
    )
    stage = _validated_tensor_mapping(
        captured.stage_tensors, allow_empty=trace_level in {TraceLevel.L0, TraceLevel.L1}
    )
    deep = _validated_tensor_mapping(captured.deep_tensors, allow_empty=True)
    if trace_level is TraceLevel.L3 and not deep:
        raise CaptureArtifactError(
            "missing_deep_checkpoints", "L3 requires deep checkpoint tensors"
        )
    summary_tensors = _merge_tensor_groups(
        processor,
        stage,
        deep if trace_level is TraceLevel.L3 else {},
    )
    if not summary_tensors:
        raise CaptureArtifactError("missing_tensors", "bundle has no tensors to summarize")

    builder = GoldenBundleBuilder(
        root=root,
        case=case,
        trace_level=trace_level,
        provenance=provenance,
    )
    builder.add_bytes("source-image.bin", source_image)
    builder.add_tensor_summaries(
        summarize_tensor(semantic_id, tensor, probe_seed=probe_seed)
        for semantic_id, tensor in summary_tensors.items()
    )

    if trace_level in {TraceLevel.L1, TraceLevel.L2, TraceLevel.L3}:
        builder.add_bytes(
            "probes.bin",
            build_probe_bundle(
                summary_tensors,
                seed=probe_seed,
                samples_per_tensor=64,
            ),
        )
    if trace_level in {TraceLevel.L2, TraceLevel.L3}:
        if captured.token_trace is None:
            raise CaptureArtifactError(
                "missing_token_trace", "L2 and L3 require a token trace"
            )
        builder.add_bytes("processor.safetensors", serialize_safetensors(processor))
        builder.add_bytes(
            "stage-checkpoints.safetensors", serialize_safetensors(stage)
        )
        builder.add_bytes(
            "token-trace.jsonl", serialize_token_trace(captured.token_trace)
        )
    if trace_level is TraceLevel.L3:
        builder.add_bytes(
            "deep-checkpoints.safetensors", serialize_safetensors(deep)
        )
    return builder.finish()
