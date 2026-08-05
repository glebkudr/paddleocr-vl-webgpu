from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Protocol

from blake3 import blake3

from .model_lock import PINNED_MODEL_ID, PINNED_REVISION
from .trace_bundle import BundleFormatError, CaseSpec, canonical_json_bytes


CAPTURE_SCHEMA_VERSION = 1
_MISSING = object()


class CaptureContractError(ValueError):
    def __init__(
        self, code: str, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True, slots=True)
class CaptureSettings:
    model_id: str
    revision: str
    device: str
    dtype: str
    seed: int
    deterministic_algorithms: bool
    inference_mode: bool

    def validate(self) -> None:
        if self.model_id != PINNED_MODEL_ID or self.revision != PINNED_REVISION:
            raise CaptureContractError(
                "wrong_model_identity", "capture must use the pinned model snapshot"
            )
        if self.device not in {"cpu", "mps"}:
            raise CaptureContractError(
                "unsupported_oracle_device",
                f"unsupported reference device: {self.device!r}",
            )
        allowed_dtypes = {
            "cpu": {"float32"},
            "mps": {"float32", "bfloat16"},
        }
        if self.dtype not in allowed_dtypes[self.device]:
            raise CaptureContractError(
                "unsupported_oracle_precision",
                f"{self.dtype!r} is not approved on {self.device}",
            )
        if not _is_nonnegative_int(self.seed) or self.seed > (2**63 - 1):
            raise CaptureContractError("invalid_seed", "seed must be an unsigned int63")
        if self.deterministic_algorithms is not True or self.inference_mode is not True:
            raise CaptureContractError(
                "nondeterministic_capture",
                "deterministic algorithms and inference mode are mandatory",
            )


@dataclass(frozen=True, slots=True)
class ProcessorObservation:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    image_grid_thw: tuple[int, int, int]
    pixel_values_shape: tuple[int, int, int, int]
    spatial_merge_size: int
    image_token_count: int
    placeholder_count: int

    def validate(self) -> None:
        if (
            not self.input_ids
            or any(not _is_nonnegative_int(token) for token in self.input_ids)
            or len(self.attention_mask) != len(self.input_ids)
            or any(mask not in {0, 1} or isinstance(mask, bool) for mask in self.attention_mask)
        ):
            raise CaptureContractError(
                "processor_shape_mismatch",
                "input_ids and attention_mask must have the same valid sequence length",
            )
        if len(self.image_grid_thw) != 3 or any(
            not _is_positive_int(axis) for axis in self.image_grid_thw
        ):
            raise CaptureContractError(
                "invalid_image_grid", "image_grid_thw must contain three positive axes"
            )
        if len(self.pixel_values_shape) != 4 or any(
            not _is_positive_int(axis) for axis in self.pixel_values_shape
        ):
            raise CaptureContractError(
                "processor_shape_mismatch", "pixel_values must be a positive rank-4 tensor"
            )

        patch_count = math.prod(self.image_grid_thw)
        if self.pixel_values_shape[0] != patch_count:
            raise CaptureContractError(
                "patch_grid_mismatch",
                "flattened pixel patch count differs from image_grid_thw",
            )
        if not _is_positive_int(self.spatial_merge_size):
            raise CaptureContractError(
                "placeholder_mismatch", "spatial_merge_size must be positive"
            )
        merge_area = self.spatial_merge_size**2
        if patch_count % merge_area != 0:
            raise CaptureContractError(
                "placeholder_mismatch", "patch grid is not divisible by merge area"
            )
        expected_image_tokens = patch_count // merge_area
        if (
            not _is_nonnegative_int(self.image_token_count)
            or not _is_nonnegative_int(self.placeholder_count)
            or self.image_token_count != expected_image_tokens
            or self.placeholder_count != expected_image_tokens
        ):
            raise CaptureContractError(
                "placeholder_mismatch",
                "processor placeholder count differs from merged image token count",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "image_grid_thw": list(self.image_grid_thw),
            "pixel_values_shape": list(self.pixel_values_shape),
            "spatial_merge_size": self.spatial_merge_size,
            "image_token_count": self.image_token_count,
            "placeholder_count": self.placeholder_count,
        }


@dataclass(frozen=True, slots=True)
class GenerationStep:
    step: int
    input_token: int
    position_ids: tuple[int, int, int]
    cache_position: int
    rope_delta: int
    top_tokens: tuple[tuple[int, float], ...]
    chosen_token: int
    kv_shapes: tuple[tuple[int, int, int, int], ...]

    def validate(self) -> None:
        if not _is_nonnegative_int(self.step):
            raise CaptureContractError("invalid_step_index", "step must be nonnegative")
        if not _is_nonnegative_int(self.input_token) or not _is_nonnegative_int(
            self.chosen_token
        ):
            raise CaptureContractError("invalid_token_id", "token IDs must be nonnegative")
        if len(self.position_ids) != 3 or any(
            not _is_nonnegative_int(position) for position in self.position_ids
        ):
            raise CaptureContractError(
                "invalid_position_ids", "position_ids must contain three nonnegative axes"
            )
        if not _is_nonnegative_int(self.cache_position):
            raise CaptureContractError(
                "invalid_cache_position", "cache_position must be nonnegative"
            )
        if not isinstance(self.rope_delta, int) or isinstance(self.rope_delta, bool):
            raise CaptureContractError("invalid_rope_delta", "rope_delta must be an integer")

        if not self.top_tokens:
            raise CaptureContractError("invalid_top_tokens", "top token set is empty")
        seen: set[int] = set()
        previous: tuple[float, int] | None = None
        for token, score in self.top_tokens:
            if (
                not _is_nonnegative_int(token)
                or token in seen
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                raise CaptureContractError(
                    "invalid_top_tokens", "top token entries must be unique and finite"
                )
            ordering = (-float(score), token)
            if previous is not None and ordering < previous:
                raise CaptureContractError(
                    "invalid_top_tokens",
                    "top tokens must be sorted by score then token ID",
                )
            previous = ordering
            seen.add(token)
        if self.top_tokens[0][0] != self.chosen_token:
            raise CaptureContractError(
                "chosen_token_mismatch", "chosen token is not the deterministic argmax"
            )

        if not self.kv_shapes:
            raise CaptureContractError("invalid_kv_shape", "KV shape list is empty")
        for shape in self.kv_shapes:
            if len(shape) != 4 or any(not _is_positive_int(axis) for axis in shape):
                raise CaptureContractError(
                    "invalid_kv_shape", "each KV cache shape must be positive rank-4"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "input_token": self.input_token,
            "position_ids": list(self.position_ids),
            "cache_position": self.cache_position,
            "rope_delta": self.rope_delta,
            "top_tokens": [[token, score] for token, score in self.top_tokens],
            "chosen_token": self.chosen_token,
            "kv_shapes": [list(shape) for shape in self.kv_shapes],
        }


@dataclass(frozen=True, slots=True)
class GenerationTrace:
    tokens: tuple[int, ...]
    steps: tuple[GenerationStep, ...]

    def validate(self) -> None:
        if any(not _is_nonnegative_int(token) for token in self.tokens):
            raise CaptureContractError("invalid_token_id", "generated token IDs are invalid")
        if len(self.steps) != len(self.tokens):
            raise CaptureContractError(
                "generation_trace_mismatch", "one diagnostic step is required per token"
            )

        previous: GenerationStep | None = None
        for index, (token, step) in enumerate(zip(self.tokens, self.steps, strict=True)):
            step.validate()
            if step.step != index:
                raise CaptureContractError(
                    "invalid_step_index", "generation step indexes must be contiguous"
                )
            if step.chosen_token != token:
                raise CaptureContractError(
                    "chosen_token_mismatch", "trace token differs from chosen token"
                )
            if previous is not None:
                if step.input_token != previous.chosen_token:
                    raise CaptureContractError(
                        "input_token_mismatch", "decode input is not the preceding token"
                    )
                if step.cache_position != previous.cache_position + 1:
                    raise CaptureContractError(
                        "cache_position_mismatch", "cache position must advance by one"
                    )
                expected_positions = tuple(position + 1 for position in previous.position_ids)
                if step.position_ids != expected_positions:
                    raise CaptureContractError(
                        "position_id_mismatch", "decode position IDs must advance by one"
                    )
                if step.rope_delta != previous.rope_delta:
                    raise CaptureContractError(
                        "rope_delta_mismatch", "rope delta must remain stable during decode"
                    )
            previous = step

    def to_dict(self) -> dict[str, object]:
        return {
            "tokens": list(self.tokens),
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class CaptureResult:
    case: CaseSpec
    processor: ProcessorObservation
    generate_tokens: tuple[int, ...]
    manual_trace: GenerationTrace

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "case_id": self.case.case_id,
            "source_image_hash": self.case.source_image_hash,
            "processor": self.processor.to_dict(),
            "generate_tokens": list(self.generate_tokens),
            "manual_trace": self.manual_trace.to_dict(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def semantic_digest(self) -> str:
        return f"blake3:{blake3(self.canonical_bytes()).hexdigest()}"


class ReferenceOracle(Protocol):
    def load(self, capture_settings: CaptureSettings) -> None: ...

    def eval(self) -> None: ...

    def enable_determinism(self, seed: int) -> None: ...

    def process(self, source: bytes, case: CaseSpec) -> ProcessorObservation: ...

    def synchronize(self) -> None: ...

    def generate(
        self, processed: ProcessorObservation, *, max_new_tokens: int, do_sample: bool
    ) -> tuple[int, ...]: ...

    def manual_generate(
        self, processed: ProcessorObservation, *, max_new_tokens: int, do_sample: bool
    ) -> GenerationTrace: ...


def _validate_generated_tokens(tokens: object, max_new_tokens: int) -> tuple[int, ...]:
    if (
        not isinstance(tokens, tuple)
        or len(tokens) > max_new_tokens
        or any(not _is_nonnegative_int(token) for token in tokens)
    ):
        raise CaptureContractError(
            "invalid_generation", "oracle returned an invalid generated token sequence"
        )
    return tokens


def assert_generation_parity(
    generated: tuple[int, ...], manual: tuple[int, ...]
) -> None:
    for index, (generate_token, manual_token) in enumerate(
        zip_longest(generated, manual, fillvalue=_MISSING)
    ):
        if generate_token != manual_token:
            raise CaptureContractError(
                "generation_mismatch",
                f"generate and manual decode first diverge at step {index}",
                details={
                    "first_divergent_step": index,
                    "generate_token": (
                        None if generate_token is _MISSING else generate_token
                    ),
                    "manual_token": None if manual_token is _MISSING else manual_token,
                },
            )


class ReferenceCaptureRunner:
    def __init__(self, oracle: ReferenceOracle, settings: CaptureSettings) -> None:
        self.oracle = oracle
        self.settings = settings

    def capture(self, case: CaseSpec, source_image: bytes) -> CaptureResult:
        self.settings.validate()
        try:
            case.validate()
        except BundleFormatError as error:
            raise CaptureContractError(
                error.code, str(error), details=error.details
            ) from error
        if not isinstance(source_image, bytes):
            raise CaptureContractError("invalid_source", "source image must be bytes")
        actual_hash = f"blake3:{blake3(source_image).hexdigest()}"
        if actual_hash != case.source_image_hash:
            raise CaptureContractError(
                "source_hash_mismatch",
                "source bytes do not match the case hash",
                details={"expected": case.source_image_hash, "actual": actual_hash},
            )

        self.oracle.load(self.settings)
        self.oracle.eval()
        self.oracle.enable_determinism(self.settings.seed)

        processed = self.oracle.process(source_image, case)
        processed.validate()
        self.oracle.synchronize()

        generated = self.oracle.generate(
            processed,
            max_new_tokens=case.max_new_tokens,
            do_sample=case.do_sample,
        )
        self.oracle.synchronize()
        generated = _validate_generated_tokens(generated, case.max_new_tokens)

        manual_trace = self.oracle.manual_generate(
            processed,
            max_new_tokens=case.max_new_tokens,
            do_sample=case.do_sample,
        )
        self.oracle.synchronize()
        manual_trace.validate()
        if len(manual_trace.tokens) > case.max_new_tokens:
            raise CaptureContractError(
                "invalid_generation", "manual decode exceeded max_new_tokens"
            )
        assert_generation_parity(generated, manual_trace.tokens)

        return CaptureResult(
            case=case,
            processor=processed,
            generate_tokens=generated,
            manual_trace=manual_trace,
        )
