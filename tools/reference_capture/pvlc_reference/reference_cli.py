from __future__ import annotations

import argparse
import platform
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import transformers
from blake3 import blake3

from .capture import CaptureSettings
from .capture_artifacts import CapturedArtifacts, export_golden_bundle
from .model_lock import ModelLock, load_pinned_paddleocr_vl_16_lock
from .trace_bundle import (
    COMPATIBILITY_SHIM_ID,
    TRACE_SCHEMA_VERSION,
    CaptureProvenance,
    CaseSpec,
    TraceLevel,
    canonical_json_bytes,
    verify_bundle,
)
from .transformers_compat import SUPPORTED_TRANSFORMERS_VERSION
from .transformers_oracle import OracleCaptureResult, TransformersOracle


CAPTURE_TOOL_VERSION = "0.1.0"
_TRACE_LEVEL_NAMES = {
    "metadata": TraceLevel.L0,
    "probes": TraceLevel.L1,
    "stage": TraceLevel.L2,
    "deep": TraceLevel.L3,
    "l0": TraceLevel.L0,
    "l1": TraceLevel.L1,
    "l2": TraceLevel.L2,
    "l3": TraceLevel.L3,
}


class ReferenceCaptureCliError(ValueError):
    def __init__(
        self, code: str, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True, slots=True)
class CaptureCliResult:
    case_id: str
    bundle_digest: str
    generated_tokens: tuple[int, ...]
    decoded_text: str
    repeat_count: int
    semantic_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "bundle_digest": self.bundle_digest,
            "generated_tokens": list(self.generated_tokens),
            "decoded_text": self.decoded_text,
            "repeat_count": self.repeat_count,
            "semantic_fingerprint": self.semantic_fingerprint,
        }


def parse_trace_level(value: str) -> TraceLevel:
    if not isinstance(value, str) or value.lower() not in _TRACE_LEVEL_NAMES:
        raise ReferenceCaptureCliError(
            "invalid_trace_level", f"unknown trace level: {value!r}"
        )
    return _TRACE_LEVEL_NAMES[value.lower()]


def _tensor_fingerprint_records(
    captured: CapturedArtifacts,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for group_name, group in (
        ("processor", captured.processor_tensors),
        ("stage", captured.stage_tensors),
        ("deep", captured.deep_tensors),
    ):
        for semantic_id, tensor in sorted(group.items()):
            if not isinstance(tensor, torch.Tensor):
                raise ReferenceCaptureCliError(
                    "invalid_oracle_capture", f"{semantic_id} is not a tensor"
                )
            canonical = tensor.detach().cpu().contiguous()
            raw = canonical.view(torch.uint8).numpy().tobytes()
            records.append(
                {
                    "group": group_name,
                    "semantic_id": semantic_id,
                    "dtype": str(canonical.dtype).removeprefix("torch."),
                    "shape": list(canonical.shape),
                    "raw_hash": f"blake3:{blake3(raw).hexdigest()}",
                }
            )
    return records


def oracle_capture_fingerprint(
    result: OracleCaptureResult, trace_level: TraceLevel
) -> str:
    if not isinstance(result, OracleCaptureResult):
        raise ReferenceCaptureCliError(
            "invalid_oracle_capture", "oracle returned an unexpected result"
        )
    if not isinstance(trace_level, TraceLevel):
        raise ReferenceCaptureCliError("invalid_trace_level", "unknown trace level")
    token_trace = result.captured.token_trace
    if token_trace is None or token_trace != result.comparison.manual_trace:
        raise ReferenceCaptureCliError(
            "inconsistent_oracle_capture",
            "captured token trace differs from the comparison trace",
        )
    result.comparison.manual_trace.validate()
    token_trace.validate()
    payload = {
        "trace_level": trace_level.value,
        "processor": result.comparison.processor.to_dict(),
        "generate_tokens": list(result.comparison.generate_tokens),
        "manual_trace": result.comparison.manual_trace.to_dict(),
        "captured_token_trace": token_trace.to_dict(),
        "decoded_text": result.comparison.decoded_text,
        "tensors": _tensor_fingerprint_records(result.captured),
    }
    return f"blake3:{blake3(canonical_json_bytes(payload)).hexdigest()}"


def _capture_provenance(
    model_lock_path: Path,
    model_lock: ModelLock,
    *,
    device: str,
    dtype: str,
) -> CaptureProvenance:
    provenance = CaptureProvenance(
        model_id=model_lock.model_id,
        model_revision=model_lock.revision,
        model_lock_hash=f"blake3:{blake3(model_lock_path.read_bytes()).hexdigest()}",
        trace_schema_version=TRACE_SCHEMA_VERSION,
        capture_tool_version=CAPTURE_TOOL_VERSION,
        compatibility_shims=(COMPATIBILITY_SHIM_ID,),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        transformers_version=str(transformers.__version__),
        device=device,
        dtype=dtype,
        deterministic_algorithms=True,
    )
    provenance.validate()
    return provenance


def capture_case(
    *,
    model_lock_path: Path,
    snapshot: Path,
    case_path: Path,
    image_path: Path,
    output: Path,
    device: str,
    dtype: str,
    trace_level: TraceLevel,
    max_new_tokens: int,
    repeat: int,
    probe_seed: int,
    oracle_factory: Callable[..., Any] = TransformersOracle,
    publisher: Callable[..., Any] = export_golden_bundle,
) -> CaptureCliResult:
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat <= 0:
        raise ReferenceCaptureCliError("invalid_repeat", "repeat must be positive")
    if not isinstance(trace_level, TraceLevel):
        raise ReferenceCaptureCliError("invalid_trace_level", "unknown trace level")
    case = CaseSpec.load(case_path)
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
        or max_new_tokens > case.max_new_tokens
    ):
        raise ReferenceCaptureCliError(
            "invalid_generation_limit", "max_new_tokens exceeds the case contract"
        )
    if output.exists() or output.is_symlink():
        raise ReferenceCaptureCliError("output_exists", f"output exists: {output}")
    if not image_path.is_file() or image_path.is_symlink():
        raise ReferenceCaptureCliError("invalid_source", f"invalid image: {image_path}")

    model_lock = load_pinned_paddleocr_vl_16_lock(model_lock_path)
    CaptureSettings(
        model_id=model_lock.model_id,
        revision=model_lock.revision,
        device=device,
        dtype=dtype,
        seed=probe_seed,
        deterministic_algorithms=True,
        inference_mode=True,
    ).validate()
    if str(transformers.__version__) != SUPPORTED_TRANSFORMERS_VERSION:
        raise ReferenceCaptureCliError(
            "unsupported_transformers_version",
            f"expected Transformers {SUPPORTED_TRANSFORMERS_VERSION}",
        )

    oracle = oracle_factory(
        snapshot,
        model_lock,
        device=device,
        dtype=dtype,
    )
    consensus_result: OracleCaptureResult | None = None
    consensus_fingerprint: str | None = None
    for index in range(repeat):
        current = oracle.capture_artifacts(
            case,
            image_path,
            max_new_tokens=max_new_tokens,
            trace_level=trace_level,
        )
        fingerprint = oracle_capture_fingerprint(current, trace_level)
        if consensus_fingerprint is None:
            consensus_result = current
            consensus_fingerprint = fingerprint
        elif fingerprint != consensus_fingerprint:
            raise ReferenceCaptureCliError(
                "nondeterministic_oracle_capture",
                f"oracle repeat {index + 1} differs from the first capture",
                details={"repeat": index + 1},
            )

    assert consensus_result is not None and consensus_fingerprint is not None
    provenance = _capture_provenance(
        model_lock_path,
        model_lock,
        device=device,
        dtype=dtype,
    )
    build_result = publisher(
        root=output,
        case=case,
        source_image=image_path.read_bytes(),
        provenance=provenance,
        trace_level=trace_level,
        captured=consensus_result.captured,
        probe_seed=probe_seed,
    )
    report = verify_bundle(
        output,
        expected_bundle_digest=build_result.bundle_digest,
    )
    if report.case.case_id != case.case_id:
        raise ReferenceCaptureCliError(
            "published_bundle_mismatch", "published bundle identifies another case"
        )
    return CaptureCliResult(
        case_id=case.case_id,
        bundle_digest=build_result.bundle_digest,
        generated_tokens=consensus_result.comparison.generate_tokens,
        decoded_text=consensus_result.comparison.decoded_text,
        repeat_count=repeat,
        semantic_fingerprint=consensus_fingerprint,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a deterministic PaddleOCR-VL-1.6 golden trace bundle"
    )
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--trace-level",
        choices=("metadata", "probes", "stage", "deep", "L0", "L1", "L2", "L3"),
        default="stage",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--probe-seed", type=int, default=12_345)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _infer_image_path(case_path: Path) -> Path:
    if case_path.parent.name == "cases":
        candidate = case_path.parent.parent / "assets" / f"{case_path.stem}.png"
        if candidate.is_file():
            return candidate
    raise ReferenceCaptureCliError(
        "image_required", "--image is required when it cannot be inferred from the case"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        model_lock = load_pinned_paddleocr_vl_16_lock(args.model_lock)
        case = CaseSpec.load(args.case)
        snapshot = (
            args.snapshot
            if args.snapshot is not None
            else args.model_lock.parent / "snapshots" / model_lock.revision
        )
        image_path = args.image if args.image is not None else _infer_image_path(args.case)
        dtype = args.dtype or ("float32" if args.device == "cpu" else "bfloat16")
        max_new_tokens = args.max_new_tokens or case.max_new_tokens
        result = capture_case(
            model_lock_path=args.model_lock,
            snapshot=snapshot,
            case_path=args.case,
            image_path=image_path,
            output=args.out,
            device=args.device,
            dtype=dtype,
            trace_level=parse_trace_level(args.trace_level),
            max_new_tokens=max_new_tokens,
            repeat=args.repeat,
            probe_seed=args.probe_seed,
        )
    except (OSError, ValueError, RuntimeError) as error:
        code = getattr(error, "code", error.__class__.__name__)
        print(f"capture failed [{code}]: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(result.to_dict()))
    return 0
