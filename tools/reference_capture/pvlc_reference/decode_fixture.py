from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import re
import stat
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import torch
from blake3 import blake3
from safetensors import SafetensorError
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from .capture import GenerationStep, GenerationTrace
from .capture_artifacts import CapturedArtifacts
from .golden_lock import GoldenEntry, GoldenLock, GoldenLockError
from .trace_bundle import (
    BundleFormatError,
    BundleIntegrityError,
    CaptureProvenance,
    TraceLevel,
    canonical_json_bytes,
    verify_bundle,
)


PREFIX_TOKENS = 332
LAYERS = 18
KEY_VALUE_HEADS = 2
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
VOCAB_SIZE = 103424
QUERY_HEADS = 16

PREFILL_TOKEN = 94013
DECODE_TOKEN = 898
OFFICIAL_CASE_ID = "ocr.clean_latin.0001"
OFFICIAL_DECODED_TEXT = "JUL"
PINNED_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
PINNED_MODEL_REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e"
PINNED_MODEL_LOCK = (
    "blake3:c2ffa46821fdda1a7e4cd6bad63cdcdc2f775b8fed250b0679cc7d5411577d10"
)
PINNED_LAYER0_FIXTURE = (
    "blake3:30eed2f7a4d9336f8b3429d7294065c45214c4425f7559e8e2059ebd54c89522"
)
PINNED_COMPATIBILITY_SHIMS = (
    "paddleocr-vl-1.6/transformers-v5-abi@1",
)
_DIGEST_RE = re.compile(r"^blake3:[0-9a-f]{64}$")


class DecodeFixtureError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True, slots=True)
class DecodeFixtureSource:
    captured: CapturedArtifacts
    provenance: CaptureProvenance
    publication: GoldenEntry
    publication_lock_blake3: str


@dataclass(frozen=True, slots=True)
class _CreatedDirectory:
    path: Path
    identity: tuple[int, int]


def _fail(code: str, message: str, **details: Any) -> None:
    raise DecodeFixtureError(code, message, details=details)


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def _validate_publication(source: DecodeFixtureSource) -> None:
    publication = source.publication
    if not isinstance(publication, GoldenEntry):
        _fail("invalid_source_publication", "source publication is not a golden entry")
    if not _is_digest(publication.bundle_digest) or not _is_digest(
        publication.semantic_fingerprint
    ):
        _fail("invalid_source_digest", "source publication contains a malformed digest")
    if (
        isinstance(publication.repeat_count, bool)
        or not isinstance(publication.repeat_count, int)
        or publication.repeat_count < 2
    ):
        _fail(
            "insufficient_repeat_count",
            "source publication requires two deterministic captures",
        )
    try:
        publication.validate()
    except ValueError as error:
        _fail("invalid_source_publication", f"invalid source publication: {error}")
    if (
        publication.case_id != OFFICIAL_CASE_ID
        or publication.trace_level is not TraceLevel.L3
        or publication.generated_tokens != (PREFILL_TOKEN, DECODE_TOKEN)
        or publication.decoded_text != OFFICIAL_DECODED_TEXT
        or publication.repeat_count != 2
    ):
        _fail(
            "invalid_source_publication",
            "source publication does not identify the reviewed decode capture",
        )
    if not _is_digest(source.publication_lock_blake3):
        _fail("invalid_source_digest", "publication lock digest is malformed")
    lock = GoldenLock(
        format_version=1,
        model_revision=PINNED_MODEL_REVISION,
        trace_schema_version=1,
        bundles=(publication,),
    )
    expected = f"blake3:{blake3(lock.canonical_bytes()).hexdigest()}"
    if source.publication_lock_blake3 != expected:
        _fail(
            "publication_lock_digest_mismatch",
            "publication lock digest does not authenticate the selected entry",
            actual=source.publication_lock_blake3,
            expected=expected,
        )


def _validate_provenance(provenance: CaptureProvenance) -> None:
    if not isinstance(provenance, CaptureProvenance):
        _fail("invalid_provenance", "source provenance is missing")
    try:
        provenance.validate()
    except ValueError as error:
        _fail("invalid_provenance", f"invalid capture provenance: {error}")
    expected = {
        "model_id": PINNED_MODEL_ID,
        "model_revision": PINNED_MODEL_REVISION,
        "model_lock_hash": PINNED_MODEL_LOCK,
        "trace_schema_version": 1,
        "capture_tool_version": "0.1.0",
        "compatibility_shims": PINNED_COMPATIBILITY_SHIMS,
        "python_version": "3.12.13",
        "torch_version": "2.13.0",
        "transformers_version": "5.14.1",
        "device": "mps",
        "dtype": "bfloat16",
        "deterministic_algorithms": True,
    }
    if any(getattr(provenance, name) != value for name, value in expected.items()):
        _fail(
            "invalid_provenance",
            "capture provenance differs from the reviewed oracle environment",
        )


TensorSpec = tuple[Mapping[str, torch.Tensor], str, tuple[int, ...], torch.dtype]


def _tensor_specs(captured: CapturedArtifacts) -> tuple[TensorSpec, ...]:
    processor = captured.processor_tensors
    stage = captured.stage_tensors
    deep = captured.deep_tensors
    prefix = "decoder.decode.00"
    specs: list[TensorSpec] = [
        (processor, "processor.input_ids", (1, PREFIX_TOKENS), torch.int64),
        (
            processor,
            "processor.attention_mask",
            (1, PREFIX_TOKENS),
            torch.int64,
        ),
        (
            stage,
            "decoder.prefill.logits.last",
            (1, VOCAB_SIZE),
            torch.bfloat16,
        ),
        (stage, "decoder.mrope.delta", (1, 1), torch.int64),
        (
            stage,
            "decoder.mrope.index",
            (3, 1, PREFIX_TOKENS),
            torch.int64,
        ),
        (deep, f"{prefix}.attention_mask", (1, PREFIX_TOKENS + 1), torch.int64),
        (deep, f"{prefix}.cache_position", (1,), torch.int64),
        (deep, f"{prefix}.position_ids", (3, 1, 1), torch.int64),
        (deep, f"{prefix}.rope.cos", (3, 1, 1, HEAD_DIM), torch.bfloat16),
        (deep, f"{prefix}.rope.sin", (3, 1, 1, HEAD_DIM), torch.bfloat16),
        (deep, f"{prefix}.layer.00.input", (1, 1, HIDDEN_SIZE), torch.bfloat16),
        (deep, f"{prefix}.layer.00.norm1", (1, 1, HIDDEN_SIZE), torch.bfloat16),
        (
            deep,
            f"{prefix}.layer.00.q",
            (1, 1, QUERY_HEADS * HEAD_DIM),
            torch.bfloat16,
        ),
        (
            deep,
            f"{prefix}.layer.00.k",
            (1, 1, KEY_VALUE_HEADS * HEAD_DIM),
            torch.bfloat16,
        ),
        (
            deep,
            f"{prefix}.layer.00.v",
            (1, 1, KEY_VALUE_HEADS * HEAD_DIM),
            torch.bfloat16,
        ),
        (
            deep,
            f"{prefix}.layer.00.mrope.q",
            (1, QUERY_HEADS, 1, HEAD_DIM),
            torch.bfloat16,
        ),
        (
            deep,
            f"{prefix}.layer.00.mrope.k",
            (1, KEY_VALUE_HEADS, 1, HEAD_DIM),
            torch.bfloat16,
        ),
        (
            deep,
            f"{prefix}.layer.00.attention.context",
            (1, 1, QUERY_HEADS * HEAD_DIM),
            torch.bfloat16,
        ),
    ]
    for suffix in ("attention.output", "attention.residual", "norm2"):
        specs.append(
            (deep, f"{prefix}.layer.00.{suffix}", (1, 1, HIDDEN_SIZE), torch.bfloat16)
        )
    for suffix in ("mlp.gate", "mlp.up", "mlp.activation"):
        specs.append(
            (
                deep,
                f"{prefix}.layer.00.{suffix}",
                (1, 1, INTERMEDIATE_SIZE),
                torch.bfloat16,
            )
        )
    specs.append(
        (deep, f"{prefix}.layer.00.mlp.down", (1, 1, HIDDEN_SIZE), torch.bfloat16)
    )
    specs.extend(
        (
            (deep, f"{prefix}.final_norm", (1, 1, HIDDEN_SIZE), torch.bfloat16),
            (deep, f"{prefix}.logits", (1, 1, VOCAB_SIZE), torch.bfloat16),
        )
    )
    for layer_index in range(LAYERS):
        layer = f"{layer_index:02d}"
        specs.append(
            (deep, f"{prefix}.layer.{layer}.output", (1, 1, HIDDEN_SIZE), torch.bfloat16)
        )
    for layer_index in range(LAYERS):
        layer = f"{layer_index:02d}"
        for kind in ("key", "value"):
            specs.append(
                (
                    deep,
                    f"decoder.layer.{layer}.kv.{kind}",
                    (1, KEY_VALUE_HEADS, PREFIX_TOKENS, HEAD_DIM),
                    torch.bfloat16,
                )
            )
            specs.append(
                (
                    deep,
                    f"{prefix}.layer.{layer}.kv.{kind}",
                    (1, KEY_VALUE_HEADS, PREFIX_TOKENS + 1, HEAD_DIM),
                    torch.bfloat16,
                )
            )
    return tuple(specs)


def _validate_tensor_contracts(captured: CapturedArtifacts) -> None:
    if not isinstance(captured, CapturedArtifacts):
        _fail("missing_tensor", "captured artifacts are missing")
    specs = _tensor_specs(captured)
    tensors: list[tuple[str, torch.Tensor]] = []
    for mapping, semantic_id, expected_shape, expected_dtype in specs:
        tensor = mapping.get(semantic_id)
        if not isinstance(tensor, torch.Tensor):
            _fail("missing_tensor", f"required tensor is missing: {semantic_id}")
        if tuple(tensor.shape) != expected_shape:
            _fail(
                "invalid_tensor_shape",
                f"tensor has the wrong shape: {semantic_id}",
                actual=tuple(tensor.shape),
                expected=expected_shape,
            )
        if tensor.dtype != expected_dtype:
            _fail(
                "invalid_tensor_dtype",
                f"tensor has the wrong dtype: {semantic_id}",
                actual=str(tensor.dtype),
                expected=str(expected_dtype),
            )
        tensors.append((semantic_id, tensor))
    for semantic_id, tensor in tensors:
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            _fail("nonfinite_tensor", f"tensor contains NaN or infinity: {semantic_id}")


def _top_tokens(logits: torch.Tensor, count: int) -> tuple[tuple[int, float], ...]:
    scores, token_ids = torch.topk(logits, k=count)
    entries = [
        (int(token_id), float(score))
        for token_id, score in zip(token_ids.tolist(), scores.tolist())
    ]
    entries.sort(key=lambda item: (-item[1], item[0]))
    return tuple(entries)


def _validate_semantics(source: DecodeFixtureSource) -> None:
    captured = source.captured
    processor = captured.processor_tensors
    stage = captured.stage_tensors
    deep = captured.deep_tensors
    prefix = "decoder.decode.00"
    trace = captured.token_trace
    if not isinstance(trace, GenerationTrace):
        _fail("invalid_generation_trace", "decode source has no generation trace")
    try:
        trace.validate()
    except ValueError as error:
        _fail("invalid_generation_trace", f"invalid generation trace: {error}")
    if trace.tokens != source.publication.generated_tokens or len(trace.steps) != 2:
        _fail("invalid_generation_trace", "trace tokens differ from the publication")
    prefill_step, decode_step = trace.steps

    input_ids = processor["processor.input_ids"]
    if int(input_ids[0, -1].item()) != prefill_step.input_token:
        _fail("invalid_generation_trace", "prefill input token is not the processor terminal")
    prefill_logits = stage["decoder.prefill.logits.last"][0]
    if int(torch.argmax(prefill_logits).item()) != prefill_step.chosen_token:
        _fail("invalid_generation_trace", "prefill trace token is not the logits argmax")
    if _top_tokens(prefill_logits, len(prefill_step.top_tokens)) != prefill_step.top_tokens:
        _fail("invalid_generation_trace", "prefill top-token trace differs from logits")

    decode_logits = deep[f"{prefix}.logits"][0, 0]
    if int(torch.argmax(decode_logits).item()) != decode_step.chosen_token:
        _fail("logits_argmax_mismatch", "decode trace token is not the logits argmax")
    if _top_tokens(decode_logits, len(decode_step.top_tokens)) != decode_step.top_tokens:
        _fail("invalid_generation_trace", "decode top-token trace differs from logits")

    if prefill_step.cache_position != PREFIX_TOKENS:
        _fail("cache_position_mismatch", "prefill cache position is not canonical")
    cache_position = int(deep[f"{prefix}.cache_position"][0].item())
    if cache_position != prefill_step.cache_position:
        _fail("cache_position_mismatch", "decode cache position differs from trace")
    expected_positions = tuple(int(value) for value in deep[f"{prefix}.position_ids"][:, 0, 0])
    if expected_positions != prefill_step.position_ids or expected_positions != (42, 42, 42):
        _fail("position_id_mismatch", "decode position IDs differ from the trace")
    terminal_positions = tuple(
        int(value) for value in stage["decoder.mrope.index"][:, 0, -1]
    )
    expected_terminal_positions = tuple(
        position - 1 for position in prefill_step.position_ids
    )
    if terminal_positions != expected_terminal_positions:
        _fail("position_id_mismatch", "MRoPE terminal position differs from the trace")
    rope_delta = int(stage["decoder.mrope.delta"][0, 0].item())
    if rope_delta != prefill_step.rope_delta or rope_delta != -290:
        _fail("rope_delta_mismatch", "MRoPE delta differs from the trace")

    processor_mask = processor["processor.attention_mask"]
    decode_mask = deep[f"{prefix}.attention_mask"]
    if not bool(torch.all((decode_mask == 0) | (decode_mask == 1)).item()):
        _fail("invalid_attention_mask", "decode attention mask is not binary")
    if not torch.equal(decode_mask[:, :-1], processor_mask) or int(
        decode_mask[0, -1].item()
    ) != 1:
        _fail("invalid_attention_mask", "decode attention mask is not the exact prefix extension")

    actual_prefill_shapes: list[tuple[int, int, int, int]] = []
    actual_post_shapes: list[tuple[int, int, int, int]] = []
    for layer_index in range(LAYERS):
        layer = f"{layer_index:02d}"
        prefill_key = deep[f"decoder.layer.{layer}.kv.key"]
        post_key = deep[f"{prefix}.layer.{layer}.kv.key"]
        actual_prefill_shapes.append(tuple(int(axis) for axis in prefill_key.shape))
        actual_post_shapes.append(tuple(int(axis) for axis in post_key.shape))
        for kind in ("key", "value"):
            prefill = deep[f"decoder.layer.{layer}.kv.{kind}"]
            post = deep[f"{prefix}.layer.{layer}.kv.{kind}"]
            if not torch.equal(post[:, :, :PREFIX_TOKENS, :], prefill):
                _fail(
                    "cache_prefix_mismatch",
                    f"post-decode cache changed its verified prefix: layer {layer} {kind}",
                )
    if tuple(actual_prefill_shapes) != prefill_step.kv_shapes or tuple(
        actual_post_shapes
    ) != decode_step.kv_shapes:
        _fail("cache_shape_mismatch", "generation trace cache geometry differs from tensors")

    layer0_key_append = deep[f"{prefix}.layer.00.kv.key"][:, :, -1:, :]
    if not torch.equal(layer0_key_append, deep[f"{prefix}.layer.00.mrope.k"]):
        _fail("cache_append_mismatch", "layer-zero key cache append differs from MRoPE key")
    raw_value = (
        deep[f"{prefix}.layer.00.v"]
        .view(1, 1, KEY_VALUE_HEADS, HEAD_DIM)
        .transpose(1, 2)
        .contiguous()
    )
    layer0_value_append = deep[f"{prefix}.layer.00.kv.value"][:, :, -1:, :]
    if not torch.equal(layer0_value_append, raw_value):
        _fail("cache_append_mismatch", "layer-zero value cache append differs from raw value")


def _owned(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous().clone()


def _fixture_tensors(source: DecodeFixtureSource) -> dict[str, torch.Tensor]:
    captured = source.captured
    deep = captured.deep_tensors
    trace = captured.token_trace
    assert trace is not None
    prefix = "decoder.decode.00"
    output: dict[str, torch.Tensor] = {}
    for kind in ("key", "value"):
        layers: list[torch.Tensor] = []
        for layer_index in range(LAYERS):
            layer = f"{layer_index:02d}"
            prefill = deep[f"decoder.layer.{layer}.kv.{kind}"][0].permute(1, 0, 2)
            append = deep[f"{prefix}.layer.{layer}.kv.{kind}"][
                0, :, PREFIX_TOKENS, :
            ].unsqueeze(0)
            layers.append(torch.cat((_owned(prefill), _owned(append)), dim=0))
        output[f"{prefix}.kv.{kind}.layer_token_major"] = torch.stack(layers)

    output.update(
        {
            f"{prefix}.attention_mask": _owned(deep[f"{prefix}.attention_mask"]),
            f"{prefix}.cache_position": _owned(deep[f"{prefix}.cache_position"]),
            f"{prefix}.input_token_id": torch.tensor(
                [[trace.steps[1].input_token]], dtype=torch.int64
            ),
            f"{prefix}.position_ids": _owned(deep[f"{prefix}.position_ids"]),
            "decoder.mrope.delta": _owned(
                captured.stage_tensors["decoder.mrope.delta"]
            ),
            f"{prefix}.rope.cos.axis_major": _owned(
                deep[f"{prefix}.rope.cos"].squeeze(1)
            ),
            f"{prefix}.rope.sin.axis_major": _owned(
                deep[f"{prefix}.rope.sin"].squeeze(1)
            ),
        }
    )
    for suffix in (
        "input",
        "norm1",
        "q",
        "k",
        "v",
        "attention.output",
        "attention.residual",
        "norm2",
        "mlp.gate",
        "mlp.up",
        "mlp.activation",
        "mlp.down",
    ):
        semantic_id = f"{prefix}.layer.00.{suffix}"
        output[semantic_id] = _owned(deep[semantic_id].squeeze(0))
    for suffix in ("mrope.q", "mrope.k"):
        source_id = f"{prefix}.layer.00.{suffix}"
        output[f"{source_id}.token_major"] = _owned(
            deep[source_id].squeeze(0).permute(1, 0, 2)
        )
    context_id = f"{prefix}.layer.00.attention.context"
    output[f"{context_id}.token_major"] = _owned(
        deep[context_id].reshape(1, QUERY_HEADS, HEAD_DIM)
    )
    for layer_index in range(LAYERS):
        semantic_id = f"{prefix}.layer.{layer_index:02d}.output"
        output[semantic_id] = _owned(deep[semantic_id].squeeze(0))
    output[f"{prefix}.final_norm"] = _owned(deep[f"{prefix}.final_norm"].squeeze(0))
    output[f"{prefix}.logits"] = _owned(deep[f"{prefix}.logits"].squeeze(0))
    return dict(sorted(output.items()))


def _fixture_metadata(source: DecodeFixtureSource) -> dict[str, str]:
    publication = source.publication
    provenance = source.provenance
    trace = source.captured.token_trace
    assert trace is not None
    metadata = {
        "bias": "false",
        "cache_layout": "layer_token_major",
        "cache_position": str(PREFIX_TOKENS),
        "cache_tokens": str(PREFIX_TOKENS + 1),
        "capture_repeat_count": str(publication.repeat_count),
        "capture_tool_version": provenance.capture_tool_version,
        "case_id": publication.case_id,
        "decode_input_token": str(trace.steps[1].input_token),
        "decode_next_token": str(trace.steps[1].chosen_token),
        "decode_step": "1",
        "decode_tokens": "1",
        "decoded_text": publication.decoded_text,
        "device": provenance.device,
        "dtype": provenance.dtype,
        "fixture_schema": "pvlc.decoder_decode.official.v1",
        "generated_tokens": ",".join(str(token) for token in publication.generated_tokens),
        "head_dim": str(HEAD_DIM),
        "hidden_size": str(HIDDEN_SIZE),
        "intermediate_size": str(INTERMEDIATE_SIZE),
        "key_value_heads": str(KEY_VALUE_HEADS),
        "layer0_weights_fixture_blake3": PINNED_LAYER0_FIXTURE,
        "layers": str(LAYERS),
        "model_id": provenance.model_id,
        "model_lock_blake3": provenance.model_lock_hash,
        "model_revision": provenance.model_revision,
        "mrope_sections": "16,24,24",
        "oracle": "TransformersOracle pinned remote code",
        "prefix_tokens": str(PREFIX_TOKENS),
        "query_heads": str(QUERY_HEADS),
        "rms_norm_epsilon": "1e-5",
        "rope_delta": str(trace.steps[0].rope_delta),
        "source_bundle_digest": publication.bundle_digest,
        "source_publication_lock_blake3": source.publication_lock_blake3,
        "source_semantic_fingerprint": publication.semantic_fingerprint,
        "torch_version": provenance.torch_version,
        "trace_level": publication.trace_level.value,
        "transformers_version": provenance.transformers_version,
        "vocab_size": str(VOCAB_SIZE),
    }
    return dict(sorted(metadata.items()))


def _validate_source(source: DecodeFixtureSource) -> None:
    if not isinstance(source, DecodeFixtureSource):
        _fail("invalid_source_publication", "decode fixture source has the wrong type")
    _validate_publication(source)
    _validate_provenance(source.provenance)
    _validate_tensor_contracts(source.captured)
    _validate_semantics(source)


def serialize_decode_fixture(source: DecodeFixtureSource) -> bytes:
    _validate_source(source)
    tensors = _fixture_tensors(source)
    metadata = _fixture_metadata(source)
    if len(tensors) != 44 or len(metadata) != 38:
        _fail("invalid_fixture_inventory", "internal decode fixture inventory is incomplete")
    serialized = save_safetensors(tensors, metadata=metadata)
    (header_size,) = struct.unpack("<Q", serialized[:8])
    header = json.loads(serialized[8 : 8 + header_size])
    canonical_header = json.dumps(
        header,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    canonical_header += b" " * (-len(canonical_header) % 8)
    return (
        struct.pack("<Q", len(canonical_header))
        + canonical_header
        + serialized[8 + header_size :]
    )


def _parse_token_trace(payload: bytes) -> GenerationTrace:
    if not isinstance(payload, bytes):
        _fail("invalid_generation_trace", "generation trace payload is not bytes")
    if not payload:
        _fail("invalid_generation_trace", "generation trace is empty")
    steps: list[GenerationStep] = []
    expected_keys = {
        "step",
        "input_token",
        "position_ids",
        "cache_position",
        "rope_delta",
        "top_tokens",
        "chosen_token",
        "kv_shapes",
    }
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line:
            _fail("invalid_generation_trace", "generation trace contains a blank record")
        try:
            raw = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            _fail(
                "invalid_generation_trace",
                f"cannot parse generation trace record {line_number}: {error}",
            )
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            _fail(
                "invalid_generation_trace",
                f"generation trace record {line_number} has the wrong schema",
            )
        try:
            canonical_line = canonical_json_bytes(raw)
        except (TypeError, ValueError, OverflowError) as error:
            _fail(
                "invalid_generation_trace",
                f"generation trace record {line_number} is not canonical: {error}",
            )
        if line != canonical_line:
            _fail(
                "invalid_generation_trace",
                f"generation trace record {line_number} is not canonical",
            )
        raw_positions = raw["position_ids"]
        raw_top_tokens = raw["top_tokens"]
        raw_kv_shapes = raw["kv_shapes"]
        if (
            not isinstance(raw_positions, list)
            or not isinstance(raw_top_tokens, list)
            or not isinstance(raw_kv_shapes, list)
            or any(not isinstance(item, list) or len(item) != 2 for item in raw_top_tokens)
            or any(not isinstance(item, list) for item in raw_kv_shapes)
        ):
            _fail(
                "invalid_generation_trace",
                f"generation trace record {line_number} contains invalid arrays",
            )
        try:
            step = GenerationStep(
                step=raw["step"],
                input_token=raw["input_token"],
                position_ids=tuple(raw_positions),
                cache_position=raw["cache_position"],
                rope_delta=raw["rope_delta"],
                top_tokens=tuple(
                    (item[0], item[1]) for item in raw_top_tokens
                ),
                chosen_token=raw["chosen_token"],
                kv_shapes=tuple(tuple(item) for item in raw_kv_shapes),
            )
            step.validate()
        except (TypeError, ValueError) as error:
            _fail(
                "invalid_generation_trace",
                f"generation trace record {line_number} is invalid: {error}",
            )
        steps.append(step)
    trace = GenerationTrace(
        tokens=tuple(step.chosen_token for step in steps),
        steps=tuple(steps),
    )
    try:
        trace.validate()
    except (TypeError, ValueError) as error:
        _fail("invalid_generation_trace", f"generation trace is invalid: {error}")
    return trace


def _load_lock(
    path: Path,
    expected_publication_lock_blake3: str,
) -> tuple[GoldenEntry, bytes]:
    if not _is_digest(expected_publication_lock_blake3):
        _fail("invalid_source_digest", "expected publication lock digest is malformed")
    try:
        descriptor = _open_path_regular_descriptor(path)
    except OSError as error:
        _fail("invalid_source_publication", f"cannot read publication lock: {error}")
    try:
        raw_bytes = _read_descriptor_snapshot(
            descriptor,
            artifact_name="publication lock",
        )
    except OSError as error:
        _fail("invalid_source_publication", f"cannot read publication lock: {error}")
    finally:
        _safe_close(descriptor)
    actual_digest = f"blake3:{blake3(raw_bytes).hexdigest()}"
    if actual_digest != expected_publication_lock_blake3:
        _fail(
            "publication_lock_digest_mismatch",
            "publication lock bytes differ from the external pin",
            actual=actual_digest,
            expected=expected_publication_lock_blake3,
        )
    try:
        lock = GoldenLock.parse_bytes(raw_bytes)
    except GoldenLockError as error:
        if error.code == "insufficient_repeat_count":
            _fail("insufficient_repeat_count", str(error))
        _fail("invalid_source_publication", f"invalid publication lock: {error}")
    if raw_bytes != lock.canonical_bytes() or len(lock.bundles) != 1:
        _fail(
            "invalid_source_publication",
            "publication lock must be canonical and select exactly one bundle",
        )
    return lock.bundles[0], raw_bytes


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _parse_canonical_bundle_json(payload: bytes, artifact_name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
        canonical = canonical_json_bytes(value)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as error:
        _fail(
            "invalid_source_bundle",
            f"cannot parse authenticated {artifact_name}: {error}",
        )
    if not isinstance(value, dict) or payload != canonical:
        _fail(
            "invalid_source_bundle",
            f"authenticated {artifact_name} is not a canonical JSON object",
        )
    return value


def _snapshot_authenticated_artifacts(
    bundle_descriptor: int,
    publication: GoldenEntry,
) -> dict[str, bytes]:
    try:
        hashes_bytes = _snapshot_relative_artifact(
            "hashes.json",
            directory_fd=bundle_descriptor,
        )
    except OSError as error:
        _fail("invalid_source_bundle", f"cannot snapshot hashes.json: {error}")
    snapshot_digest = f"blake3:{blake3(hashes_bytes).hexdigest()}"
    if snapshot_digest != publication.bundle_digest:
        _fail(
            "invalid_source_bundle",
            "post-verification hashes.json differs from the publication digest",
            actual=snapshot_digest,
            expected=publication.bundle_digest,
        )
    hashes = _parse_canonical_bundle_json(hashes_bytes, "hashes.json")
    if set(hashes) != {"algorithm", "artifacts", "format_version"}:
        _fail("invalid_source_bundle", "authenticated hashes.json has the wrong schema")
    if hashes["algorithm"] != "blake3" or hashes["format_version"] != 1:
        _fail("invalid_source_bundle", "authenticated hashes.json uses an unknown format")
    raw_artifacts = hashes["artifacts"]
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        _fail("invalid_source_bundle", "authenticated hashes.json has no artifacts")

    authenticated_entries: dict[str, tuple[int, str]] = {}
    for artifact_name, raw_entry in raw_artifacts.items():
        path = PurePosixPath(artifact_name) if isinstance(artifact_name, str) else None
        if (
            path is None
            or path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or not isinstance(raw_entry, dict)
            or set(raw_entry) != {"blake3", "size"}
        ):
            _fail("invalid_source_bundle", "authenticated artifact hash entry is invalid")
        raw_digest = raw_entry["blake3"]
        raw_size = raw_entry["size"]
        if (
            not isinstance(raw_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", raw_digest) is None
            or isinstance(raw_size, bool)
            or not isinstance(raw_size, int)
            or raw_size < 0
        ):
            _fail("invalid_source_bundle", "authenticated artifact hash metadata is invalid")
        authenticated_entries[artifact_name] = (raw_size, raw_digest)

    consumed_artifacts = (
        "manifest.json",
        "processor.safetensors",
        "stage-checkpoints.safetensors",
        "deep-checkpoints.safetensors",
        "token-trace.jsonl",
    )
    snapshot: dict[str, bytes] = {}
    for artifact_name in consumed_artifacts:
        expected = authenticated_entries.get(artifact_name)
        if expected is None:
            _fail(
                "invalid_source_bundle",
                f"authenticated bundle omits required payload: {artifact_name}",
            )
        try:
            payload = _snapshot_relative_artifact(
                artifact_name,
                directory_fd=bundle_descriptor,
            )
        except OSError as error:
            _fail(
                "invalid_source_bundle",
                f"cannot snapshot authenticated payload {artifact_name}: {error}",
            )
        expected_size, expected_digest = expected
        if len(payload) != expected_size or blake3(payload).hexdigest() != expected_digest:
            _fail(
                "invalid_source_bundle",
                f"authenticated payload changed after verification: {artifact_name}",
            )
        snapshot[artifact_name] = payload
    return snapshot


def _parse_authenticated_manifest(
    payload: bytes,
) -> tuple[str, TraceLevel, CaptureProvenance]:
    manifest = _parse_canonical_bundle_json(payload, "manifest.json")
    expected_keys = {
        "bundle_schema_version",
        "case_id",
        "provenance",
        "required_artifacts",
        "trace_level",
    }
    if set(manifest) != expected_keys or manifest["bundle_schema_version"] != 1:
        _fail("invalid_source_bundle", "authenticated manifest has the wrong schema")
    try:
        trace_level = TraceLevel(manifest["trace_level"])
        provenance = CaptureProvenance.from_dict(manifest["provenance"])
    except (BundleFormatError, KeyError, TypeError, ValueError) as error:
        _fail("invalid_source_bundle", f"authenticated manifest is invalid: {error}")
    case_id = manifest["case_id"]
    if not isinstance(case_id, str):
        _fail("invalid_source_bundle", "authenticated manifest case ID is invalid")
    return case_id, trace_level, provenance


def load_decode_fixture_source(
    bundle_root: Path | str,
    *,
    publication_lock_path: Path | str,
    expected_publication_lock_blake3: str,
) -> DecodeFixtureSource:
    lock_path = Path(publication_lock_path)
    publication, _ = _load_lock(
        lock_path,
        expected_publication_lock_blake3,
    )
    bundle_path = Path(bundle_root)
    selected_path = lock_path.parent / PurePosixPath(publication.artifact_path)
    if (
        _absolute_lexical(bundle_path) != _absolute_lexical(selected_path)
    ):
        _fail(
            "invalid_source_bundle",
            "bundle path is not the regular directory selected by the publication lock",
        )
    bundle_descriptor: int | None = None
    try:
        bundle_descriptor = _open_directory_snapshot_descriptor(bundle_path)
        bundle_identity = _stat_identity(os.fstat(bundle_descriptor))
        _assert_lexical_directory_matches(bundle_path, bundle_identity)
        report = verify_bundle(
            bundle_path,
            expected_bundle_digest=publication.bundle_digest,
        )
        _assert_lexical_directory_matches(bundle_path, bundle_identity)
    except (BundleFormatError, BundleIntegrityError, OSError, ValueError) as error:
        _safe_close(bundle_descriptor)
        _fail("invalid_source_bundle", f"source bundle verification failed: {error}")
    try:
        snapshot = _snapshot_authenticated_artifacts(
            bundle_descriptor,
            publication,
        )
    finally:
        _safe_close(bundle_descriptor)
    manifest_case_id, manifest_trace_level, manifest_provenance = (
        _parse_authenticated_manifest(snapshot["manifest.json"])
    )
    if (
        report.case.case_id != publication.case_id
        or manifest_case_id != report.case.case_id
        or manifest_trace_level is not publication.trace_level
        or publication.case_id != OFFICIAL_CASE_ID
        or publication.trace_level is not TraceLevel.L3
    ):
        _fail(
            "invalid_source_bundle",
            "verified bundle identity differs from the reviewed L3 OCR source",
        )
    if manifest_provenance != report.provenance:
        _fail(
            "invalid_source_bundle",
            "authenticated manifest provenance differs from the verification report",
        )
    _validate_provenance(manifest_provenance)
    try:
        processor = load_safetensors(snapshot["processor.safetensors"])
        stage = load_safetensors(snapshot["stage-checkpoints.safetensors"])
        deep = load_safetensors(snapshot["deep-checkpoints.safetensors"])
    except (SafetensorError, OSError, RuntimeError, ValueError) as error:
        _fail("invalid_source_bundle", f"cannot load verified source tensors: {error}")
    trace = _parse_token_trace(snapshot["token-trace.jsonl"])
    source = DecodeFixtureSource(
        captured=CapturedArtifacts(
            processor_tensors=processor,
            stage_tensors=stage,
            deep_tensors=deep,
            token_trace=trace,
        ),
        provenance=manifest_provenance,
        publication=publication,
        publication_lock_blake3=expected_publication_lock_blake3,
    )
    _validate_source(source)
    return source


def _validate_output_path(output_path: Path) -> None:
    if not output_path.name or any(part == ".." for part in output_path.parts):
        _fail("unsafe_output_path", "output path is not canonical")
    if os.path.lexists(output_path):
        if output_path.is_symlink() or output_path.is_file():
            _fail("output_exists", f"output already exists: {output_path}")
        _fail("unsafe_output_path", f"output path is not a regular file target: {output_path}")
    current = output_path.parent
    while True:
        if os.path.lexists(current) and (current.is_symlink() or not current.is_dir()):
            _fail("unsafe_output_path", f"output parent is unsafe: {current}")
        if current.parent == current:
            break
        current = current.parent


def _stat_identity(metadata: os.stat_result) -> tuple[int, int]:
    return (metadata.st_dev, metadata.st_ino)


def _open_directory_descriptor(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    return os.open(path, flags)


def _open_path_regular_descriptor(path: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags)
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise OSError(errno.EIO, "path is not a regular file")
        return descriptor
    except Exception:
        _safe_close(descriptor)
        raise


def _open_directory_snapshot_descriptor(path: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags)
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(descriptor_stat.st_mode):
            raise OSError(errno.ENOTDIR, f"path is not a directory: {path}")
        return descriptor
    except Exception:
        _safe_close(descriptor)
        raise


def _safe_close(descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


def _lexical_parent_matches(parent: Path, expected_identity: tuple[int, int]) -> bool:
    try:
        parent_stat = os.stat(parent, follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISDIR(parent_stat.st_mode) and _stat_identity(parent_stat) == expected_identity


def _lexical_directory_matches(
    path: Path,
    expected_identity: tuple[int, int],
) -> bool:
    try:
        directory_stat = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISDIR(directory_stat.st_mode) and _stat_identity(directory_stat) == (
        expected_identity
    )


def _assert_lexical_directory_matches(
    path: Path,
    expected_identity: tuple[int, int],
) -> None:
    if not _lexical_directory_matches(path, expected_identity):
        _fail(
            "invalid_source_bundle",
            "bundle path changed during verification",
        )


def _assert_lexical_parent_matches(parent: Path, expected_identity: tuple[int, int]) -> None:
    if not _lexical_parent_matches(parent, expected_identity):
        _fail(
            "publication_failed",
            f"output parent changed during publication: {parent}",
        )


def _create_parent_chain(parent: Path, created: list[_CreatedDirectory]) -> None:
    missing: list[Path] = []
    current = parent
    while not os.path.lexists(current):
        missing.append(current)
        if current.parent == current:
            _fail("unsafe_output_path", "cannot locate an existing output ancestor")
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        _fail("unsafe_output_path", f"output ancestor is unsafe: {current}")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if directory.is_symlink() or not directory.is_dir():
                _fail("unsafe_output_path", f"output parent is unsafe: {directory}")
        else:
            directory_stat = os.stat(directory, follow_symlinks=False)
            if not stat.S_ISDIR(directory_stat.st_mode):
                _fail("unsafe_output_path", f"output parent is unsafe: {directory}")
            created.append(
                _CreatedDirectory(path=directory, identity=_stat_identity(directory_stat))
            )


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError(errno.EIO, "short write while staging decode fixture")
        remaining = remaining[written:]


def _unsupported_secure_commit_error() -> OSError:
    unsupported_errno = getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)
    return OSError(
        unsupported_errno,
        "secure fd-bound decode-fixture publication is unsupported on this platform",
    )


def _raise_native_errno(function_name: str) -> None:
    native_errno = ctypes.get_errno() or errno.EIO
    raise OSError(native_errno, f"{function_name} failed: {os.strerror(native_errno)}")


def _open_libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _validate_output_basename(output_name: str) -> None:
    if not output_name or output_name in {".", ".."} or "\x00" in output_name:
        raise OSError(errno.EINVAL, "output name must be a regular basename")
    if Path(output_name).name != output_name:
        raise OSError(errno.EINVAL, "output name must not contain path separators")


def _assert_regular_descriptor(descriptor: int) -> os.stat_result:
    descriptor_stat = os.fstat(descriptor)
    if not stat.S_ISREG(descriptor_stat.st_mode):
        raise OSError(errno.EIO, "staging descriptor is not a regular file")
    return descriptor_stat


def _assert_anonymous_regular_descriptor(descriptor: int) -> os.stat_result:
    descriptor_stat = _assert_regular_descriptor(descriptor)
    if descriptor_stat.st_nlink != 0:
        raise OSError(errno.EIO, "staging descriptor is not anonymous")
    return descriptor_stat


def _create_darwin_anonymous_staging_descriptor(parent_fd: int) -> int:
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(32):
        staging_name = f".decode-fixture-staging-{os.urandom(8).hex()}"
        descriptor: int | None = None
        try:
            descriptor = os.open(staging_name, flags, 0o600, dir_fd=parent_fd)
            os.unlink(staging_name, dir_fd=parent_fd)
            _assert_anonymous_regular_descriptor(descriptor)
            return descriptor
        except FileExistsError:
            if descriptor is not None:
                _safe_close(descriptor)
            continue
        except OSError as error:
            if descriptor is not None:
                _safe_close(descriptor)
            if error.errno == errno.EEXIST:
                continue
            raise
    raise OSError(errno.EEXIST, "cannot allocate an anonymous staging descriptor")


def _create_linux_anonymous_staging_descriptor(parent_fd: int) -> int:
    open_tmpfile_flag = getattr(os, "O_TMPFILE", None)
    if open_tmpfile_flag is None:
        raise _unsupported_secure_commit_error()
    flags = os.O_RDWR | open_tmpfile_flag | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(".", flags, 0o600, dir_fd=parent_fd)
    try:
        _assert_anonymous_regular_descriptor(descriptor)
        return descriptor
    except Exception:
        _safe_close(descriptor)
        raise


def _create_anonymous_staging_descriptor(parent_fd: int) -> int:
    if sys.platform == "darwin":
        return _create_darwin_anonymous_staging_descriptor(parent_fd)
    if sys.platform == "linux":
        return _create_linux_anonymous_staging_descriptor(parent_fd)
    raise _unsupported_secure_commit_error()


def _stat_relative_entry(name: str, *, directory_fd: int) -> os.stat_result:
    return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)


def _unlink_owned_relative_entry(
    name: str | None,
    *,
    directory_fd: int | None,
    expected_identity: tuple[int, int] | None,
) -> bool:
    if name is None or directory_fd is None or expected_identity is None:
        return False
    try:
        entry_stat = _stat_relative_entry(name, directory_fd=directory_fd)
    except OSError:
        return False
    if not stat.S_ISREG(entry_stat.st_mode) or _stat_identity(entry_stat) != expected_identity:
        return False
    try:
        os.unlink(name, dir_fd=directory_fd)
    except OSError:
        return False
    return True


def _open_relative_regular_descriptor(name: str, *, directory_fd: int) -> int:
    return _open_relative_existing_regular_descriptor(
        name,
        directory_fd=directory_fd,
        descriptor_label="published output",
    )


def _open_relative_existing_regular_descriptor(
    name: str,
    *,
    directory_fd: int,
    descriptor_label: str,
) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise OSError(
                errno.EIO,
                f"{descriptor_label} is not a regular file: {name}",
            )
        return descriptor
    except Exception:
        _safe_close(descriptor)
        raise


def _read_descriptor_snapshot(
    descriptor: int,
    *,
    artifact_name: str,
) -> bytes:
    initial_stat = os.fstat(descriptor)
    if not stat.S_ISREG(initial_stat.st_mode):
        raise OSError(errno.EIO, f"{artifact_name} is not a regular file")
    expected_size = int(initial_stat.st_size)
    payload = bytearray(expected_size)
    offset = 0
    while offset < expected_size:
        chunk = os.pread(descriptor, min(1024 * 1024, expected_size - offset), offset)
        if not chunk:
            raise OSError(errno.EIO, f"short read while reading {artifact_name}")
        payload[offset : offset + len(chunk)] = chunk
        offset += len(chunk)
    final_stat = os.fstat(descriptor)
    if (
        not stat.S_ISREG(final_stat.st_mode)
        or _stat_identity(final_stat) != _stat_identity(initial_stat)
        or int(final_stat.st_size) != expected_size
    ):
        raise OSError(errno.EIO, f"{artifact_name} changed while reading")
    return bytes(payload)

def _snapshot_relative_artifact(
    name: str,
    *,
    directory_fd: int,
) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = _open_relative_existing_regular_descriptor(
            name,
            directory_fd=directory_fd,
            descriptor_label="authenticated bundle payload",
        )
        return _read_descriptor_snapshot(descriptor, artifact_name=name)
    finally:
        _safe_close(descriptor)


def _descriptor_size_and_blake3(descriptor: int) -> tuple[int, str]:
    descriptor_stat = os.fstat(descriptor)
    size = int(descriptor_stat.st_size)
    remaining = size
    offset = 0
    digest = blake3()
    while remaining:
        chunk = os.pread(descriptor, min(1024 * 1024, remaining), offset)
        if not chunk:
            raise OSError(errno.EIO, "short read while authenticating published output")
        digest.update(chunk)
        read_size = len(chunk)
        offset += read_size
        remaining -= read_size
    return (size, digest.hexdigest())


def _descriptors_have_identical_content(source_fd: int, destination_fd: int) -> bool:
    source_size, source_digest = _descriptor_size_and_blake3(source_fd)
    destination_size, destination_digest = _descriptor_size_and_blake3(destination_fd)
    return source_size == destination_size and source_digest == destination_digest


def _authenticate_published_destination(
    staging_fd: int,
    *,
    parent_fd: int,
    output_name: str,
) -> tuple[int, int]:
    _assert_regular_descriptor(staging_fd)
    destination_stat = _stat_relative_entry(output_name, directory_fd=parent_fd)
    if not stat.S_ISREG(destination_stat.st_mode):
        raise OSError(errno.EIO, "published output is not a regular file")
    destination_identity = _stat_identity(destination_stat)
    destination_fd: int | None = None
    try:
        destination_fd = _open_relative_regular_descriptor(
            output_name,
            directory_fd=parent_fd,
        )
        opened_destination_stat = os.fstat(destination_fd)
        if _stat_identity(opened_destination_stat) != destination_identity:
            raise OSError(errno.EIO, "published output changed during authentication")
        if not _descriptors_have_identical_content(staging_fd, destination_fd):
            raise OSError(errno.EIO, "published output bytes differ from staged bytes")
        return destination_identity
    finally:
        _safe_close(destination_fd)


def _rollback_authenticated_destination(
    staging_fd: int,
    *,
    parent_fd: int,
    output_name: str,
) -> bool:
    destination_fd: int | None = None
    try:
        destination_fd = _open_relative_regular_descriptor(
            output_name,
            directory_fd=parent_fd,
        )
        destination_stat = os.fstat(destination_fd)
        if not _descriptors_have_identical_content(staging_fd, destination_fd):
            return False
        destination_identity = _stat_identity(destination_stat)
    except OSError:
        return False
    finally:
        _safe_close(destination_fd)
    return _unlink_owned_relative_entry(
        output_name,
        directory_fd=parent_fd,
        expected_identity=destination_identity,
    )


def _darwin_fclonefileat(source_fd: int, destination_parent_fd: int, destination_name: str):
    try:
        function = _open_libc().fclonefileat
    except AttributeError as error:
        raise _unsupported_secure_commit_error() from error
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint32,
    )
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(
        source_fd,
        destination_parent_fd,
        os.fsencode(destination_name),
        0,
    )
    if result != 0:
        _raise_native_errno("fclonefileat")


def _linux_linkat_anonymous(source_fd: int, destination_parent_fd: int, destination_name: str):
    try:
        function = _open_libc().linkat
    except AttributeError as error:
        raise _unsupported_secure_commit_error() from error
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    function.restype = ctypes.c_int
    destination_name_bytes = os.fsencode(destination_name)
    empty_path = getattr(os, "AT_EMPTY_PATH", 0x1000)
    follow_symlink = getattr(os, "AT_SYMLINK_FOLLOW", 0x400)
    ctypes.set_errno(0)
    result = function(
        source_fd,
        b"",
        destination_parent_fd,
        destination_name_bytes,
        empty_path,
    )
    if result == 0:
        return
    initial_errno = ctypes.get_errno() or errno.EIO
    if initial_errno not in {errno.EPERM, errno.ENOENT}:
        raise OSError(initial_errno, f"linkat failed: {os.strerror(initial_errno)}")
    proc_source = f"/proc/self/fd/{source_fd}"
    ctypes.set_errno(0)
    result = function(
        getattr(os, "AT_FDCWD", -100),
        os.fsencode(proc_source),
        destination_parent_fd,
        destination_name_bytes,
        follow_symlink,
    )
    if result != 0:
        _raise_native_errno("linkat")


def _commit_staging_descriptor(
    staging_fd: int,
    *,
    parent_fd: int,
    output_name: str,
) -> tuple[int, int]:
    _validate_output_basename(output_name)
    _assert_anonymous_regular_descriptor(staging_fd)
    if sys.platform == "darwin":
        commit_primitive = _darwin_fclonefileat
    elif sys.platform == "linux":
        commit_primitive = _linux_linkat_anonymous
    else:
        raise _unsupported_secure_commit_error()
    commit_primitive(staging_fd, parent_fd, output_name)
    try:
        return _authenticate_published_destination(
            staging_fd,
            parent_fd=parent_fd,
            output_name=output_name,
        )
    except OSError:
        if _rollback_authenticated_destination(
            staging_fd,
            parent_fd=parent_fd,
            output_name=output_name,
        ):
            try:
                os.fsync(parent_fd)
            except OSError:
                pass
        raise


def _cleanup_created_directories(created_directories: Sequence[_CreatedDirectory]) -> None:
    for directory in reversed(created_directories):
        try:
            directory_stat = os.stat(directory.path, follow_symlinks=False)
        except OSError:
            continue
        if (
            not stat.S_ISDIR(directory_stat.st_mode)
            or _stat_identity(directory_stat) != directory.identity
        ):
            continue
        try:
            directory.path.rmdir()
        except OSError:
            pass


def _cleanup_publication(
    *,
    parent_fd: int | None,
    output_name: str,
    destination_identity: tuple[int, int] | None,
    created_directories: Sequence[_CreatedDirectory],
) -> None:
    parent_dirty = _unlink_owned_relative_entry(
        output_name,
        directory_fd=parent_fd,
        expected_identity=destination_identity,
    )
    if parent_dirty and parent_fd is not None:
        try:
            os.fsync(parent_fd)
        except OSError:
            pass
    _cleanup_created_directories(created_directories)


def export_decode_fixture(source: DecodeFixtureSource, output_path: Path | str) -> None:
    output = Path(output_path)
    serialized = serialize_decode_fixture(source)
    _validate_output_path(output)

    created_directories: list[_CreatedDirectory] = []
    parent_descriptor: int | None = None
    parent_identity: tuple[int, int] | None = None
    staging_descriptor: int | None = None
    destination_identity: tuple[int, int] | None = None
    try:
        _create_parent_chain(output.parent, created_directories)
        _validate_output_path(output)
        parent_descriptor = _open_directory_descriptor(output.parent)
        parent_stat = os.fstat(parent_descriptor)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise OSError(errno.ENOTDIR, f"output parent is not a directory: {output.parent}")
        parent_identity = _stat_identity(parent_stat)
        _assert_lexical_parent_matches(output.parent, parent_identity)

        staging_descriptor = _create_anonymous_staging_descriptor(parent_descriptor)
        _write_all(staging_descriptor, serialized)
        os.fsync(staging_descriptor)
        _assert_lexical_parent_matches(output.parent, parent_identity)

        try:
            destination_identity = _commit_staging_descriptor(
                staging_descriptor,
                parent_fd=parent_descriptor,
                output_name=output.name,
            )
        except OSError as error:
            if isinstance(error, FileExistsError) or error.errno == errno.EEXIST:
                if parent_identity is not None and _lexical_parent_matches(
                    output.parent, parent_identity
                ):
                    _fail("output_exists", f"output already exists: {output}")
            _fail("publication_failed", f"cannot commit decode fixture: {error}")
        _assert_lexical_parent_matches(output.parent, parent_identity)
        os.fsync(parent_descriptor)
    except DecodeFixtureError:
        _cleanup_publication(
            parent_fd=parent_descriptor,
            output_name=output.name,
            destination_identity=destination_identity,
            created_directories=created_directories,
        )
        raise
    except OSError as error:
        _cleanup_publication(
            parent_fd=parent_descriptor,
            output_name=output.name,
            destination_identity=destination_identity,
            created_directories=created_directories,
        )
        raise DecodeFixtureError(
            "publication_failed",
            f"cannot publish decode fixture: {error}",
        ) from error
    except Exception:
        _cleanup_publication(
            parent_fd=parent_descriptor,
            output_name=output.name,
            destination_identity=destination_identity,
            created_directories=created_directories,
        )
        raise
    finally:
        _safe_close(staging_descriptor)
        _safe_close(parent_descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pvlc-reference-decode-fixture",
        description="Export the reviewed decoder decode-step fixture",
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--publication-lock", required=True, type=Path)
    parser.add_argument("--expected-publication-lock-blake3", required=True)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        source = load_decode_fixture_source(
            arguments.bundle,
            publication_lock_path=arguments.publication_lock,
            expected_publication_lock_blake3=(
                arguments.expected_publication_lock_blake3
            ),
        )
        export_decode_fixture(source, arguments.out)
    except DecodeFixtureError as error:
        print(f"{error.code}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
