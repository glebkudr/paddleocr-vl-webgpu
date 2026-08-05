from __future__ import annotations

import io
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from blake3 import blake3
from PIL import Image
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache

from .capture import (
    CaptureContractError,
    GenerationStep,
    GenerationTrace,
    assert_generation_parity,
)
from .capture_artifacts import CapturedArtifacts
from .model_lock import (
    PINNED_MODEL_ID,
    PINNED_REVISION,
    ModelLock,
    verify_model_directory,
)
from .trace_bundle import (
    BundleFormatError,
    CaseSpec,
    TraceLevel,
    canonical_json_bytes,
)
from .transformers_compat import (
    initial_cache_position,
    install_transformers_compat,
)


_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
_DEFAULT_SEED = 12_345
_TOP_TOKEN_COUNT = 32


@dataclass(frozen=True, slots=True)
class ProcessorCapture:
    case_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    image_grid_thw: tuple[int, int, int]
    pixel_values_shape: tuple[int, int, int, int]
    placeholder_id: int
    placeholder_count: int
    spatial_merge_size: int
    pixel_values_digest: str
    pixel_min: float
    pixel_max: float
    pixel_mean: float
    pixel_std: float

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "image_grid_thw": list(self.image_grid_thw),
            "pixel_values_shape": list(self.pixel_values_shape),
            "placeholder_id": self.placeholder_id,
            "placeholder_count": self.placeholder_count,
            "spatial_merge_size": self.spatial_merge_size,
            "pixel_values_digest": self.pixel_values_digest,
            "pixel_min": self.pixel_min,
            "pixel_max": self.pixel_max,
            "pixel_mean": self.pixel_mean,
            "pixel_std": self.pixel_std,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class GenerationComparison:
    processor: ProcessorCapture
    generate_tokens: tuple[int, ...]
    manual_trace: GenerationTrace
    decoded_text: str


@dataclass(frozen=True, slots=True)
class OracleCaptureResult:
    comparison: GenerationComparison
    captured: CapturedArtifacts


def _extract_hook_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "last_hidden_state"):
        return _extract_hook_tensor(value.last_hidden_state)
    if isinstance(value, (tuple, list)):
        tensors = [
            _extract_hook_tensor(item)
            for item in value
            if isinstance(item, torch.Tensor)
            or isinstance(item, (tuple, list))
            or hasattr(item, "last_hidden_state")
        ]
        if len(tensors) == 1:
            return tensors[0]
        if tensors and all(
            tensor.ndim == tensors[0].ndim
            and tuple(tensor.shape[1:]) == tuple(tensors[0].shape[1:])
            for tensor in tensors
        ):
            return torch.cat(tensors, dim=0)
    raise CaptureContractError(
        "invalid_trace_hook", f"cannot normalize hook value of type {type(value)!r}"
    )


class _TensorCallAccumulator:
    def __init__(self, semantic_id: str) -> None:
        self.semantic_id = semantic_id
        self.tensors: list[torch.Tensor] = []

    def append(self, value: object) -> None:
        tensor = _extract_hook_tensor(value)
        if tensor.ndim == 0 or tensor.shape[0] == 0:
            raise CaptureContractError(
                "invalid_trace_shape",
                f"{self.semantic_id} must have a nonempty leading image-token axis",
                details={"shape": list(tensor.shape)},
            )
        if self.tensors:
            expected = self.tensors[0]
            if tensor.ndim != expected.ndim or tuple(tensor.shape[1:]) != tuple(
                expected.shape[1:]
            ):
                raise CaptureContractError(
                    "invalid_trace_shape",
                    f"{self.semantic_id} image calls disagree on tensor shape",
                    details={
                        "expected_tail": list(expected.shape[1:]),
                        "actual": list(tensor.shape),
                    },
                )
            if tensor.dtype != expected.dtype:
                raise CaptureContractError(
                    "invalid_trace_dtype",
                    f"{self.semantic_id} image calls disagree on dtype",
                    details={
                        "expected": str(expected.dtype),
                        "actual": str(tensor.dtype),
                    },
                )
            if tensor.device != expected.device:
                raise CaptureContractError(
                    "invalid_trace_device",
                    f"{self.semantic_id} image calls disagree on device",
                    details={
                        "expected": str(expected.device),
                        "actual": str(tensor.device),
                    },
                )
        self.tensors.append(tensor.detach())

    @property
    def call_count(self) -> int:
        return len(self.tensors)

    def finish(self) -> torch.Tensor:
        if not self.tensors:
            raise CaptureContractError(
                "incomplete_deep_trace",
                f"{self.semantic_id} was not captured",
            )
        if len(self.tensors) == 1:
            return self.tensors[0]
        return torch.cat(self.tensors, dim=0)


class _ProjectorTraceCapture:
    SEMANTIC_IDS = (
        "projector.pre_norm",
        "projector.merge",
        "projector.linear1",
        "projector.gelu",
        "projector.linear2",
    )

    def __init__(self) -> None:
        self._accumulators = {
            semantic_id: _TensorCallAccumulator(semantic_id)
            for semantic_id in self.SEMANTIC_IDS
        }
        self._active = False
        self._completed = False
        self._registered = False

    def _start(
        self, module: object, arguments: tuple[object, ...]
    ) -> None:
        del module, arguments
        if self._active:
            raise CaptureContractError(
                "invalid_trace_hook", "projector trace does not support reentrant forwards"
            )
        self._active = not self._completed

    def _complete(
        self,
        module: object,
        arguments: tuple[object, ...],
        output: object,
    ) -> None:
        del module, arguments, output
        if self._active:
            self._active = False
            self._completed = True

    def _output_hook(
        self, semantic_id: str
    ) -> Callable[[object, tuple[object, ...], object], None]:
        def hook(module: object, arguments: tuple[object, ...], output: object) -> None:
            del module, arguments
            if self._active:
                self._accumulators[semantic_id].append(output)

        return hook

    def _input_hook(
        self, semantic_id: str
    ) -> Callable[[object, tuple[object, ...]], None]:
        def hook(module: object, arguments: tuple[object, ...]) -> None:
            del module
            if not self._active:
                return
            if len(arguments) != 1:
                raise CaptureContractError(
                    "invalid_trace_hook",
                    f"{semantic_id} expected exactly one positional tensor input",
                    details={"argument_count": len(arguments)},
                )
            self._accumulators[semantic_id].append(arguments[0])

        return hook

    def register(self, projector: object) -> list[Any]:
        if self._registered:
            raise CaptureContractError(
                "invalid_trace_hook", "projector trace was registered more than once"
            )
        required = ("pre_norm", "linear_1", "act", "linear_2")
        if any(not hasattr(projector, name) for name in required):
            raise CaptureContractError(
                "invalid_trace_hook", "projector modules do not match the pinned topology"
            )
        self._registered = True
        return [
            projector.register_forward_pre_hook(self._start),
            projector.pre_norm.register_forward_hook(
                self._output_hook("projector.pre_norm")
            ),
            projector.linear_1.register_forward_pre_hook(
                self._input_hook("projector.merge")
            ),
            projector.linear_1.register_forward_hook(
                self._output_hook("projector.linear1")
            ),
            projector.act.register_forward_hook(
                self._output_hook("projector.gelu")
            ),
            projector.linear_2.register_forward_hook(
                self._output_hook("projector.linear2")
            ),
            projector.register_forward_hook(self._complete),
        ]

    def finish(self) -> dict[str, torch.Tensor]:
        if not self._completed or self._active:
            raise CaptureContractError(
                "incomplete_deep_trace", "projector trace did not complete its first forward"
            )
        counts = {
            semantic_id: accumulator.call_count
            for semantic_id, accumulator in self._accumulators.items()
        }
        if not counts or len(set(counts.values())) != 1 or next(iter(counts.values())) == 0:
            raise CaptureContractError(
                "incomplete_deep_trace",
                "projector stages captured different image-call counts",
                details={"call_counts": counts},
            )
        tensors = {
            semantic_id: accumulator.finish()
            for semantic_id, accumulator in self._accumulators.items()
        }
        pre_norm = tensors["projector.pre_norm"]
        merged = tensors["projector.merge"]
        linear1 = tensors["projector.linear1"]
        gelu = tensors["projector.gelu"]
        linear2 = tensors["projector.linear2"]
        if (
            pre_norm.ndim != 2
            or merged.ndim != 2
            or linear1.ndim != 2
            or gelu.ndim != 2
            or linear2.ndim != 2
            or pre_norm.shape[0] != merged.shape[0] * 4
            or merged.shape[1] != pre_norm.shape[1] * 4
            or linear1.shape != merged.shape
            or gelu.shape != linear1.shape
            or linear2.shape[0] != gelu.shape[0]
        ):
            raise CaptureContractError(
                "invalid_trace_shape",
                "projector stage shapes do not match the pinned 2x2 topology",
                details={
                    semantic_id: list(tensor.shape)
                    for semantic_id, tensor in tensors.items()
                },
            )
        if len({tensor.dtype for tensor in tensors.values()}) != 1:
            raise CaptureContractError(
                "invalid_trace_dtype", "projector stages disagree on dtype"
            )
        if len({tensor.device for tensor in tensors.values()}) != 1:
            raise CaptureContractError(
                "invalid_trace_device", "projector stages disagree on device"
            )
        return dict(sorted(tensors.items()))


class _MultimodalTraceCapture:
    SEMANTIC_IDS = (
        "decoder.embedding",
        "multimodal.image_token_indices",
        "multimodal.inputs_embeds",
        "decoder.mrope.index",
        "decoder.mrope.delta",
    )

    def __init__(self) -> None:
        self._tensors: dict[str, torch.Tensor] = {}
        self._input_ids: torch.Tensor | None = None
        self._image_token_id: int | None = None
        self._active = False
        self._completed = False
        self._registered = False

    @staticmethod
    def _exact_tensor(semantic_id: str, value: object) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise CaptureContractError(
                "invalid_trace_hook",
                f"{semantic_id} hook value must be a tensor",
                details={"type": type(value).__name__},
            )
        return value.detach().clone()

    def _capture_once(self, semantic_id: str, value: object) -> None:
        if semantic_id in self._tensors:
            raise CaptureContractError(
                "invalid_trace_hook",
                f"{semantic_id} was captured more than once in one outer forward",
            )
        self._tensors[semantic_id] = self._exact_tensor(semantic_id, value)

    def _start(
        self,
        module: object,
        arguments: tuple[object, ...],
        keyword_arguments: dict[str, object],
    ) -> None:
        if self._active:
            raise CaptureContractError(
                "invalid_trace_hook",
                "multimodal trace does not support reentrant outer forwards",
            )
        if self._completed:
            return

        input_ids_value = keyword_arguments.get("input_ids")
        if input_ids_value is None and arguments:
            input_ids_value = arguments[0]
        input_ids = self._exact_tensor("processor.input_ids", input_ids_value)
        if (
            input_ids.dtype != torch.int64
            or input_ids.ndim != 2
            or input_ids.shape[0] == 0
            or input_ids.shape[1] == 0
        ):
            raise CaptureContractError(
                "invalid_trace_shape",
                "outer forward input_ids must be a nonempty rank-2 int64 tensor",
                details={
                    "shape": list(input_ids.shape),
                    "dtype": str(input_ids.dtype),
                },
            )

        config = getattr(module, "config", None)
        image_token_id = getattr(config, "image_token_id", None)
        if not isinstance(image_token_id, int) or isinstance(image_token_id, bool):
            raise CaptureContractError(
                "invalid_trace_hook",
                "outer model config.image_token_id must be an integer",
            )
        image_indices = torch.argwhere(input_ids == image_token_id)
        if image_indices.ndim != 2 or tuple(image_indices.shape[1:]) != (2,):
            raise CaptureContractError(
                "invalid_trace_shape",
                "image-token argwhere must produce a rank-2 coordinate tensor",
                details={"shape": list(image_indices.shape)},
            )
        if image_indices.shape[0] == 0:
            raise CaptureContractError(
                "invalid_trace_shape",
                "multimodal trace requires at least one image placeholder",
            )

        self._input_ids = input_ids
        self._image_token_id = image_token_id
        self._tensors["multimodal.image_token_indices"] = image_indices.detach().clone()
        self._active = True

    def _embedding_hook(
        self,
        module: object,
        arguments: tuple[object, ...],
        output: object,
    ) -> None:
        del module, arguments
        if self._active:
            self._capture_once("decoder.embedding", output)

    def _decoder_hook(
        self,
        module: object,
        arguments: tuple[object, ...],
        keyword_arguments: dict[str, object],
    ) -> None:
        del module, arguments
        if not self._active:
            return
        for semantic_id, keyword in (
            ("multimodal.inputs_embeds", "inputs_embeds"),
            ("decoder.mrope.index", "position_ids"),
        ):
            if keyword not in keyword_arguments:
                raise CaptureContractError(
                    "invalid_trace_hook",
                    f"decoder prefill did not receive {keyword}",
                )
            self._capture_once(semantic_id, keyword_arguments[keyword])

    def _complete(
        self,
        module: object,
        arguments: tuple[object, ...],
        keyword_arguments: dict[str, object],
        output: object,
    ) -> None:
        del module, arguments, keyword_arguments
        if not self._active:
            return
        if not hasattr(output, "rope_deltas"):
            raise CaptureContractError(
                "invalid_trace_hook", "outer model output did not contain rope_deltas"
            )
        self._capture_once("decoder.mrope.delta", output.rope_deltas)
        self._active = False
        self._completed = True

    def register(self, model: object) -> list[Any]:
        if self._registered:
            raise CaptureContractError(
                "invalid_trace_hook", "multimodal trace was registered more than once"
            )
        decoder = getattr(model, "model", None)
        embed_tokens = getattr(decoder, "embed_tokens", None)
        if (
            not hasattr(model, "register_forward_pre_hook")
            or not hasattr(model, "register_forward_hook")
            or not hasattr(decoder, "register_forward_pre_hook")
            or not hasattr(embed_tokens, "register_forward_hook")
            or not hasattr(model, "config")
        ):
            raise CaptureContractError(
                "invalid_trace_hook",
                "outer model modules do not match the pinned multimodal topology",
            )
        self._registered = True
        return [
            model.register_forward_pre_hook(self._start, with_kwargs=True),
            embed_tokens.register_forward_hook(self._embedding_hook),
            decoder.register_forward_pre_hook(self._decoder_hook, with_kwargs=True),
            model.register_forward_hook(self._complete, with_kwargs=True),
        ]

    def finish(self) -> dict[str, torch.Tensor]:
        if not self._completed or self._active:
            raise CaptureContractError(
                "incomplete_stage_trace",
                "multimodal trace did not complete its first outer forward",
            )
        if self._input_ids is None or self._image_token_id is None:
            raise CaptureContractError(
                "incomplete_stage_trace", "multimodal input metadata was not captured"
            )
        if set(self._tensors) != set(self.SEMANTIC_IDS):
            raise CaptureContractError(
                "incomplete_stage_trace",
                "multimodal hook set is incomplete",
                details={"captured": sorted(self._tensors)},
            )

        input_ids = self._input_ids
        embedding = self._tensors["decoder.embedding"]
        image_indices = self._tensors["multimodal.image_token_indices"]
        inputs_embeds = self._tensors["multimodal.inputs_embeds"]
        position_ids = self._tensors["decoder.mrope.index"]
        rope_deltas = self._tensors["decoder.mrope.delta"]
        batch_size, sequence_length = input_ids.shape
        expected_embedding_shape = (batch_size, sequence_length)
        if (
            embedding.ndim != 3
            or inputs_embeds.ndim != 3
            or tuple(embedding.shape[:2]) != expected_embedding_shape
            or tuple(inputs_embeds.shape) != tuple(embedding.shape)
            or embedding.shape[2] == 0
            or tuple(position_ids.shape) != (3, batch_size, sequence_length)
            or tuple(rope_deltas.shape) != (batch_size, 1)
            or image_indices.ndim != 2
            or tuple(image_indices.shape[1:]) != (2,)
            or image_indices.shape[0] == 0
        ):
            raise CaptureContractError(
                "invalid_trace_shape",
                "multimodal stage shapes are incoherent",
                details={
                    "input_ids": list(input_ids.shape),
                    **{
                        semantic_id: list(tensor.shape)
                        for semantic_id, tensor in self._tensors.items()
                    },
                },
            )
        if (
            not embedding.is_floating_point()
            or inputs_embeds.dtype != embedding.dtype
        ):
            raise CaptureContractError(
                "invalid_trace_dtype",
                "decoder embedding stages must share one floating dtype",
            )
        metadata = (input_ids, image_indices, position_ids, rope_deltas)
        if any(tensor.dtype != torch.int64 for tensor in metadata):
            raise CaptureContractError(
                "invalid_trace_dtype", "multimodal metadata tensors must be int64"
            )
        if inputs_embeds.device != embedding.device:
            raise CaptureContractError(
                "invalid_trace_device", "decoder embedding stages disagree on device"
            )
        metadata_devices = {tensor.device for tensor in metadata}
        if len(metadata_devices) != 1 or input_ids.device != embedding.device:
            raise CaptureContractError(
                "invalid_trace_device", "multimodal stage tensors disagree on device"
            )

        index_pairs = [
            (int(batch), int(sequence))
            for batch, sequence in image_indices.detach().cpu().tolist()
        ]
        if (
            len(set(index_pairs)) != len(index_pairs)
            or any(
                batch < 0
                or batch >= batch_size
                or sequence < 0
                or sequence >= sequence_length
                for batch, sequence in index_pairs
            )
        ):
            raise CaptureContractError(
                "invalid_trace_shape",
                "image-token coordinates are duplicated or out of range",
            )
        expected_indices = torch.argwhere(input_ids == self._image_token_id)
        if not torch.equal(image_indices, expected_indices):
            raise CaptureContractError(
                "invalid_trace_shape",
                "image-token coordinates disagree with outer input_ids",
                details={
                    "expected_count": int(expected_indices.shape[0]),
                    "actual_count": int(image_indices.shape[0]),
                },
            )
        return dict(
            sorted(
                (semantic_id, tensor.detach())
                for semantic_id, tensor in self._tensors.items()
            )
        )


class _DecoderPrefillTraceCapture:
    _DECODE_PREFIX = "decoder.decode.00"
    _LAYER0_SEMANTIC_IDS = (
        "decoder.layer.00.input",
        "decoder.layer.00.norm1",
        "decoder.layer.00.q",
        "decoder.layer.00.k",
        "decoder.layer.00.v",
        "decoder.layer.00.mrope.q",
        "decoder.layer.00.mrope.k",
        "decoder.layer.00.kv.key",
        "decoder.layer.00.kv.value",
        "decoder.layer.00.attention.context",
        "decoder.layer.00.attention.output",
        "decoder.layer.00.attention.residual",
        "decoder.layer.00.norm2",
        "decoder.layer.00.mlp.gate",
        "decoder.layer.00.mlp.up",
        "decoder.layer.00.mlp.activation",
        "decoder.layer.00.mlp.down",
    )
    _ACTIVE_TRACE: _DecoderPrefillTraceCapture | None = None

    def __init__(self) -> None:
        self._registered = False
        self._handles: list[Any] = []
        self._tensors: dict[str, torch.Tensor] = {}
        self._layer_count: int | None = None
        self._decoder: object | None = None
        self._owner_module: object | None = None
        self._original_rotary_fn: Callable[..., object] | None = None
        self._current_forward_kind: str | None = None
        self._layer0_attention_active = False
        self._prefill_complete = False
        self._cache_complete = False
        self._manual_cache_snapshots: tuple[
            tuple[torch.Tensor, torch.Tensor], ...
        ] | None = None
        self._manual_attention_mask: torch.Tensor | None = None
        self._manual_terminal_position_ids: torch.Tensor | None = None
        self._pending_manual_attention_mask: torch.Tensor | None = None
        self._pending_manual_position_ids: torch.Tensor | None = None
        self._decode_decoder_position_ids: torch.Tensor | None = None
        self._decode_tensors: dict[str, torch.Tensor] = {}
        self._decode_started = False
        self._decode_complete = False
        self._decode_logits_complete = False
        self._awaiting_decode_logits = False
        self._model: object | None = None
        self._lm_head: object | None = None
        self._shadowed_modules: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []

    @staticmethod
    def _snapshot_tensor(semantic_id: str, value: object) -> torch.Tensor:
        tensor = _extract_hook_tensor(value)
        if tensor.device.type == "meta":
            raise CaptureContractError(
                "invalid_trace_device",
                f"{semantic_id} cannot be captured from the meta device",
            )
        return tensor.detach().clone()

    def _capture_once(self, semantic_id: str, value: object) -> None:
        if semantic_id in self._tensors:
            raise CaptureContractError(
                "invalid_trace_hook",
                f"{semantic_id} was captured more than once in one decoder prefill",
            )
        self._tensors[semantic_id] = self._snapshot_tensor(semantic_id, value)

    def _sanitize_module_cycles(self, decoder: torch.nn.Module, owner: torch.nn.Module) -> None:
        for module in decoder.modules():
            child = module._modules.get("_owner")
            if child is owner:
                module._modules.pop("_owner")
                object.__setattr__(module, "_owner", child)
                self._shadowed_modules.append((module, "_owner", child))

    def _restore_module_cycles(self) -> None:
        while self._shadowed_modules:
            module, name, child = self._shadowed_modules.pop()
            if name in module.__dict__:
                del module.__dict__[name]
            setattr(module, name, child)

    def _legacy_expected_semantic_ids(self) -> set[str]:
        if self._layer_count is None:
            raise CaptureContractError(
                "incomplete_deep_trace", "decoder prefill trace was not registered"
            )
        expected = {
            "decoder.rope.cos",
            "decoder.rope.sin",
            *self._LAYER0_SEMANTIC_IDS,
            "decoder.final_norm",
        }
        expected.update(
            f"decoder.layer.{index:02d}.output" for index in range(self._layer_count)
        )
        return expected

    def _expected_semantic_ids(self) -> set[str]:
        expected = self._legacy_expected_semantic_ids()
        if not self._decode_logits_complete:
            return expected
        assert self._layer_count is not None
        expected.update(
            f"decoder.layer.{index:02d}.kv.{kind}"
            for index in range(self._layer_count)
            for kind in ("key", "value")
        )
        expected.update(
            f"{self._DECODE_PREFIX}.{semantic_id.removeprefix('decoder.')}"
            for semantic_id in self._legacy_expected_semantic_ids()
            if ".kv." not in semantic_id
        )
        expected.update(
            f"{self._DECODE_PREFIX}.layer.{index:02d}.kv.{kind}"
            for index in range(self._layer_count)
            for kind in ("key", "value")
        )
        expected.update(
            {
                f"{self._DECODE_PREFIX}.attention_mask",
                f"{self._DECODE_PREFIX}.cache_position",
                f"{self._DECODE_PREFIX}.position_ids",
                f"{self._DECODE_PREFIX}.logits",
            }
        )
        return expected

    @property
    def semantic_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._expected_semantic_ids()))

    @staticmethod
    def _require_shape(
        semantic_id: str,
        tensor: torch.Tensor,
        expected_shape: tuple[int, ...],
    ) -> None:
        actual_shape = tuple(int(axis) for axis in tensor.shape)
        if actual_shape != expected_shape:
            raise CaptureContractError(
                "invalid_trace_shape",
                f"{semantic_id} shape does not match the pinned decoder ABI",
                details={
                    "expected": list(expected_shape),
                    "actual": list(actual_shape),
                },
            )

    @staticmethod
    def _require_floating(semantic_id: str, tensor: torch.Tensor) -> None:
        if not tensor.is_floating_point():
            raise CaptureContractError(
                "invalid_trace_dtype",
                f"{semantic_id} must be floating-point",
                details={"dtype": str(tensor.dtype)},
            )

    @classmethod
    def _require_same_dtype(
        cls,
        reference_id: str,
        reference: torch.Tensor,
        semantic_id: str,
        tensor: torch.Tensor,
    ) -> None:
        cls._require_floating(semantic_id, tensor)
        if tensor.dtype != reference.dtype:
            raise CaptureContractError(
                "invalid_trace_dtype",
                f"{semantic_id} dtype does not match {reference_id}",
                details={
                    "expected": str(reference.dtype),
                    "actual": str(tensor.dtype),
                },
            )

    @staticmethod
    def _require_same_device(
        reference_id: str,
        reference: torch.Tensor,
        semantic_id: str,
        tensor: torch.Tensor,
    ) -> None:
        if tensor.device != reference.device:
            raise CaptureContractError(
                "invalid_trace_device",
                f"{semantic_id} device does not match {reference_id}",
                details={
                    "expected": str(reference.device),
                    "actual": str(tensor.device),
                },
            )
        if tensor.device.type == "meta":
            raise CaptureContractError(
                "invalid_trace_device",
                f"{semantic_id} cannot live on the meta device",
            )

    @staticmethod
    def _require_finite(semantic_id: str, tensor: torch.Tensor) -> None:
        if not bool(torch.isfinite(tensor).all().item()):
            raise CaptureContractError(
                "invalid_trace_value",
                f"{semantic_id} must contain only finite values",
            )

    @classmethod
    def _validate_integer_tensor(
        cls,
        semantic_id: str,
        value: object,
        *,
        expected_shape: tuple[int, ...],
        expected_device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise CaptureContractError(
                "invalid_trace_hook",
                f"{semantic_id} must be a tensor",
                details={"type": type(value).__name__},
            )
        if value.device.type == "meta" or value.device != expected_device:
            raise CaptureContractError(
                "invalid_trace_device",
                f"{semantic_id} device does not match the decoder",
                details={
                    "expected": str(expected_device),
                    "actual": str(value.device),
                },
            )
        if value.dtype != torch.int64:
            raise CaptureContractError(
                "invalid_trace_dtype",
                f"{semantic_id} must be int64",
                details={"actual": str(value.dtype)},
            )
        cls._require_shape(semantic_id, value, expected_shape)
        return value

    @classmethod
    def _validate_binary_mask(
        cls,
        semantic_id: str,
        value: object,
        *,
        expected_shape: tuple[int, ...],
        expected_device: torch.device,
    ) -> torch.Tensor:
        mask = cls._validate_integer_tensor(
            semantic_id,
            value,
            expected_shape=expected_shape,
            expected_device=expected_device,
        )
        if not bool(((mask == 0) | (mask == 1)).all().item()):
            raise CaptureContractError(
                "invalid_trace_value",
                f"{semantic_id} must contain only binary values",
            )
        return mask

    def _validate_cache_layers(
        self,
        cache: object,
        *,
        phase: str,
        sequence_length: int,
        reference: torch.Tensor,
        expected_snapshots: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if type(cache) is not DynamicCache:
            raise CaptureContractError(
                "invalid_kv_shape",
                f"{phase} must use an exact DynamicCache",
                details={"type": type(cache).__name__},
            )
        if self._layer_count is None or len(cache.layers) != self._layer_count:
            raise CaptureContractError(
                "invalid_kv_shape",
                f"{phase} cache must cover exactly every decoder layer",
                details={
                    "expected_layers": self._layer_count,
                    "actual_layers": len(cache.layers),
                },
            )
        if expected_snapshots is not None and len(expected_snapshots) != self._layer_count:
            raise CaptureContractError(
                "incomplete_deep_trace",
                "saved manual-prefill cache snapshot is incomplete",
            )

        expected_shape = (
            int(reference.shape[0]),
            2,
            sequence_length,
            128,
        )
        validated: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, layer in enumerate(cache.layers):
            semantic_prefix = f"{phase} layer {layer_index}"
            if not hasattr(layer, "keys") or not hasattr(layer, "values"):
                raise CaptureContractError(
                    "invalid_kv_shape",
                    f"{semantic_prefix} is missing keys/values",
                )
            keys = layer.keys
            values = layer.values
            if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
                raise CaptureContractError(
                    "invalid_kv_shape",
                    f"{semantic_prefix} keys/values must be tensors",
                )
            for kind, tensor in (("key", keys), ("value", values)):
                semantic_id = f"{semantic_prefix} {kind}"
                if tensor.device.type == "meta":
                    raise CaptureContractError(
                        "invalid_trace_device",
                        f"{semantic_id} cannot live on the meta device",
                    )
                self._require_shape(semantic_id, tensor, expected_shape)
                self._require_same_dtype(
                    "decoder.layer.00.input",
                    reference,
                    semantic_id,
                    tensor,
                )
                self._require_same_device(
                    "decoder.layer.00.input",
                    reference,
                    semantic_id,
                    tensor,
                )
                self._require_finite(semantic_id, tensor)
            if expected_snapshots is not None:
                expected_keys, expected_values = expected_snapshots[layer_index]
                if not torch.equal(keys, expected_keys) or not torch.equal(
                    values, expected_values
                ):
                    raise CaptureContractError(
                        "invalid_kv_value",
                        f"{semantic_prefix} differs from the saved manual-prefill cache",
                    )
            validated.append((keys, values))
        return tuple(validated)

    @staticmethod
    def _snapshot_cache_layers(
        layers: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(
            (keys.detach().clone(), values.detach().clone())
            for keys, values in layers
        )

    @staticmethod
    def _decoder_query_tensor(
        arguments: tuple[object, ...],
        keyword_arguments: dict[str, object],
    ) -> torch.Tensor:
        value = keyword_arguments.get("inputs_embeds")
        if value is None:
            value = keyword_arguments.get("hidden_states")
        if value is None and arguments:
            value = arguments[0]
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise CaptureContractError(
                "invalid_trace_shape",
                "decoder forward must receive a rank-3 hidden-state tensor",
            )
        return value

    def _phase_semantic_id(self, semantic_id: str) -> str:
        if self._current_forward_kind == "decode":
            return f"{self._DECODE_PREFIX}.{semantic_id.removeprefix('decoder.')}"
        return semantic_id

    def _capture_phase_once(self, semantic_id: str, value: object) -> None:
        target_id = self._phase_semantic_id(semantic_id)
        target = self._decode_tensors if self._current_forward_kind == "decode" else self._tensors
        if target_id in target:
            raise CaptureContractError(
                "invalid_trace_hook",
                f"{target_id} was captured more than once in one decoder forward",
            )
        target[target_id] = self._snapshot_tensor(target_id, value)

    @staticmethod
    def _module_hookable(module: object) -> bool:
        return hasattr(module, "register_forward_hook") and hasattr(
            module, "register_forward_pre_hook"
        )

    def _reference_tensor(self) -> torch.Tensor:
        reference = self._tensors.get("decoder.layer.00.input")
        if reference is None:
            raise CaptureContractError(
                "incomplete_deep_trace",
                "decoder trace has no prefill reference tensor",
            )
        return reference

    def _prepare_manual_prefill(
        self,
        query: torch.Tensor,
        keyword_arguments: dict[str, object],
    ) -> None:
        reference = self._reference_tensor()
        batch_size, sequence_length, _ = reference.shape
        self._require_shape(
            "manual-prefill input",
            query,
            (int(batch_size), int(sequence_length), 1024),
        )
        self._require_same_dtype(
            "decoder.layer.00.input",
            reference,
            "manual-prefill input",
            query,
        )
        self._require_same_device(
            "decoder.layer.00.input",
            reference,
            "manual-prefill input",
            query,
        )
        attention_mask = keyword_arguments.get("attention_mask")
        if attention_mask is None:
            self._pending_manual_attention_mask = None
        else:
            validated_mask = self._validate_binary_mask(
                "manual-prefill attention_mask",
                attention_mask,
                expected_shape=(int(batch_size), int(sequence_length)),
                expected_device=reference.device,
            )
            self._pending_manual_attention_mask = validated_mask.detach().clone()
        position_ids = self._validate_integer_tensor(
            "manual-prefill position_ids",
            keyword_arguments.get("position_ids"),
            expected_shape=(3, int(batch_size), int(sequence_length)),
            expected_device=reference.device,
        )
        self._pending_manual_position_ids = position_ids.detach().clone()

    def _prepare_decode(
        self,
        query: torch.Tensor,
        keyword_arguments: dict[str, object],
    ) -> None:
        reference = self._reference_tensor()
        if self._manual_cache_snapshots is None:
            raise CaptureContractError(
                "incomplete_deep_trace",
                "decode started before manual-prefill cache capture",
            )
        batch_size = int(reference.shape[0])
        sequence_length = int(reference.shape[1])
        self._require_shape(
            f"{self._DECODE_PREFIX}.layer.00.input",
            query,
            (batch_size, 1, 1024),
        )
        self._require_same_dtype(
            "decoder.layer.00.input",
            reference,
            f"{self._DECODE_PREFIX}.layer.00.input",
            query,
        )
        self._require_same_device(
            "decoder.layer.00.input",
            reference,
            f"{self._DECODE_PREFIX}.layer.00.input",
            query,
        )
        self._validate_cache_layers(
            keyword_arguments.get("past_key_values"),
            phase="decode input",
            sequence_length=sequence_length,
            reference=reference,
            expected_snapshots=self._manual_cache_snapshots,
        )
        if self._manual_attention_mask is None:
            raise CaptureContractError(
                "invalid_trace_hook",
                "decode requires a saved manual-prefill attention mask",
            )
        attention_mask = self._validate_binary_mask(
            f"{self._DECODE_PREFIX}.attention_mask",
            keyword_arguments.get("attention_mask"),
            expected_shape=(batch_size, sequence_length + 1),
            expected_device=reference.device,
        )
        if not torch.equal(attention_mask[:, :-1], self._manual_attention_mask):
            raise CaptureContractError(
                "invalid_trace_value",
                "decode attention-mask prefix differs from manual prefill",
            )
        if not bool((attention_mask[:, -1:] == 1).all().item()):
            raise CaptureContractError(
                "invalid_trace_value",
                "decode attention mask must append one active token",
            )
        decoder_position_ids = self._validate_integer_tensor(
            f"{self._DECODE_PREFIX}.position_ids",
            keyword_arguments.get("position_ids"),
            expected_shape=(3, batch_size, 1),
            expected_device=reference.device,
        )
        self._decode_tensors[f"{self._DECODE_PREFIX}.attention_mask"] = (
            attention_mask.detach().clone()
        )
        self._decode_decoder_position_ids = decoder_position_ids.detach().clone()
        self._decode_started = True

    def _decoder_start(
        self,
        module: object,
        arguments: tuple[object, ...],
        keyword_arguments: dict[str, object],
    ) -> None:
        del module
        try:
            if self._current_forward_kind is not None:
                raise CaptureContractError(
                    "invalid_trace_hook",
                    "decoder trace does not support reentrant forwards",
                )
            query = self._decoder_query_tensor(arguments, keyword_arguments)
            query_length = int(query.shape[1])
            use_cache = bool(keyword_arguments.get("use_cache", False))
            past_key_values = keyword_arguments.get("past_key_values")

            self._current_forward_kind = "ignore"
            if not self._prefill_complete:
                if not use_cache and query_length > 1:
                    self._current_forward_kind = "prefill"
                return

            reference_length = int(self._reference_tensor().shape[1])
            if not self._cache_complete:
                if not use_cache:
                    return
                if past_key_values is None and query_length == reference_length:
                    self._current_forward_kind = "manual_prefill"
                    self._prepare_manual_prefill(query, keyword_arguments)
                    return
                raise CaptureContractError(
                    "invalid_kv_shape",
                    "cached decode started before a valid manual prefill",
                )

            if self._awaiting_decode_logits:
                raise CaptureContractError(
                    "incomplete_deep_trace",
                    "another decoder forward started before target decode logits",
                )
            if not use_cache or self._decode_started:
                return
            if past_key_values is None and query_length == reference_length:
                return
            if type(past_key_values) is not DynamicCache:
                raise CaptureContractError(
                    "invalid_kv_shape",
                    "one-token cached decode requires an exact DynamicCache input",
                    details={"type": type(past_key_values).__name__},
                )
            if query_length != 1:
                raise CaptureContractError(
                    "invalid_trace_shape",
                    "cached decode target must contain exactly one query token",
                    details={"query_length": query_length},
                )
            self._current_forward_kind = "decode"
            self._prepare_decode(query, keyword_arguments)
        except BaseException:
            self.close()
            raise

    def _rotary_hook(
        self,
        module: object,
        arguments: tuple[object, ...],
        output: object,
    ) -> None:
        del module, arguments
        try:
            if self._current_forward_kind not in ("prefill", "decode"):
                return
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise CaptureContractError(
                    "invalid_trace_hook",
                    "decoder rotary embedding must return a (cos, sin) pair",
                )
            cos, sin = output
            self._capture_phase_once("decoder.rope.cos", cos)
            self._capture_phase_once("decoder.rope.sin", sin)
        except BaseException:
            self.close()
            raise

    def _layer0_self_attn_start(
        self, module: object, arguments: tuple[object, ...], keyword_arguments: dict[str, object]
    ) -> None:
        del module, arguments
        try:
            if self._current_forward_kind == "prefill":
                self._layer0_attention_active = True
                return
            if self._current_forward_kind != "decode":
                return
            reference = self._reference_tensor()
            batch_size = int(reference.shape[0])
            sequence_length = int(reference.shape[1])
            cache_position = self._validate_integer_tensor(
                f"{self._DECODE_PREFIX}.cache_position",
                keyword_arguments.get("cache_position"),
                expected_shape=(1,),
                expected_device=reference.device,
            )
            expected_cache_position = torch.tensor(
                [sequence_length],
                dtype=torch.int64,
                device=reference.device,
            )
            if not torch.equal(cache_position, expected_cache_position):
                raise CaptureContractError(
                    "invalid_trace_value",
                    "effective decode cache_position does not append the prompt",
                )
            position_ids = self._validate_integer_tensor(
                f"{self._DECODE_PREFIX}.position_ids",
                keyword_arguments.get("position_ids"),
                expected_shape=(3, batch_size, 1),
                expected_device=reference.device,
            )
            if self._decode_decoder_position_ids is None:
                raise CaptureContractError(
                    "incomplete_deep_trace",
                    "decoder prehook did not save decode position_ids",
                )
            if not torch.equal(position_ids, self._decode_decoder_position_ids):
                raise CaptureContractError(
                    "invalid_trace_value",
                    "self-attention position_ids differ from decoder position_ids",
                )
            if self._manual_terminal_position_ids is None or not torch.equal(
                position_ids,
                self._manual_terminal_position_ids + 1,
            ):
                raise CaptureContractError(
                    "invalid_trace_value",
                    "decode position_ids do not append manual-prefill M-RoPE positions",
                )
            self._capture_phase_once("decoder.cache_position", cache_position)
            self._capture_phase_once("decoder.position_ids", position_ids)
            self._layer0_attention_active = True
        except BaseException:
            self.close()
            raise

    def _layer0_self_attn_complete(
        self,
        module: object,
        arguments: tuple[object, ...],
        keyword_arguments: dict[str, object],
        output: object,
    ) -> None:
        del module, arguments, keyword_arguments, output
        self._layer0_attention_active = False

    def _input_hook(
        self, semantic_id: str
    ) -> Callable[[object, tuple[object, ...]], None]:
        def hook(module: object, arguments: tuple[object, ...]) -> None:
            del module
            try:
                if self._current_forward_kind not in ("prefill", "decode"):
                    return
                if len(arguments) != 1:
                    raise CaptureContractError(
                        "invalid_trace_hook",
                        f"{semantic_id} expected exactly one positional tensor input",
                        details={"argument_count": len(arguments)},
                    )
                self._capture_phase_once(semantic_id, arguments[0])
            except BaseException:
                self.close()
                raise

        return hook

    def _output_hook(
        self, semantic_id: str
    ) -> Callable[[object, tuple[object, ...], object], None]:
        def hook(module: object, arguments: tuple[object, ...], output: object) -> None:
            del module, arguments
            try:
                if self._current_forward_kind in ("prefill", "decode"):
                    self._capture_phase_once(semantic_id, output)
            except BaseException:
                self.close()
                raise

        return hook

    def _extract_cache(self, output: object) -> DynamicCache:
        cache: object | None = None
        if hasattr(output, "past_key_values"):
            cache = output.past_key_values
        elif isinstance(output, (tuple, list)):
            for item in output:
                if item is None:
                    continue
                if isinstance(item, DynamicCache):
                    cache = item
                    break
                if hasattr(item, "past_key_values"):
                    cache = item.past_key_values
                    break
            if cache is None and len(output) >= 2:
                cache = output[1]
        if not isinstance(cache, DynamicCache):
            raise CaptureContractError(
                "invalid_kv_shape", "decoder prefill did not produce a DynamicCache"
            )
        return cache

    def _capture_manual_cache(self, output: object) -> None:
        reference = self._reference_tensor()
        sequence_length = int(reference.shape[1])
        layers = self._validate_cache_layers(
            self._extract_cache(output),
            phase="manual prefill",
            sequence_length=sequence_length,
            reference=reference,
        )
        snapshots = self._snapshot_cache_layers(layers)
        if self._pending_manual_position_ids is None:
            raise CaptureContractError(
                "incomplete_deep_trace",
                "manual-prefill position_ids were not captured",
            )
        self._manual_cache_snapshots = snapshots
        self._manual_attention_mask = self._pending_manual_attention_mask
        self._manual_terminal_position_ids = (
            self._pending_manual_position_ids[:, :, -1:].detach().clone()
        )
        self._tensors["decoder.layer.00.kv.key"] = snapshots[0][0]
        self._tensors["decoder.layer.00.kv.value"] = snapshots[0][1]
        self._pending_manual_attention_mask = None
        self._pending_manual_position_ids = None
        self._cache_complete = True

    def _capture_decode_cache(self, output: object) -> None:
        reference = self._reference_tensor()
        sequence_length = int(reference.shape[1])
        if self._manual_cache_snapshots is None:
            raise CaptureContractError(
                "incomplete_deep_trace",
                "decode cache validation has no manual-prefill snapshot",
            )
        layers = self._validate_cache_layers(
            self._extract_cache(output),
            phase="decode output",
            sequence_length=sequence_length + 1,
            reference=reference,
        )
        for layer_index, ((keys, values), (prefill_keys, prefill_values)) in enumerate(
            zip(layers, self._manual_cache_snapshots, strict=True)
        ):
            if not torch.equal(keys[:, :, :-1, :], prefill_keys) or not torch.equal(
                values[:, :, :-1, :], prefill_values
            ):
                raise CaptureContractError(
                    "invalid_kv_value",
                    f"decode output layer {layer_index} mutated its cache prefix",
                )

        decode_key = self._decode_tensors.get(
            f"{self._DECODE_PREFIX}.layer.00.mrope.k"
        )
        decode_value = self._decode_tensors.get(f"{self._DECODE_PREFIX}.layer.00.v")
        if decode_key is None or decode_value is None:
            raise CaptureContractError(
                "incomplete_deep_trace",
                "decode cache arrived before live layer-0 K/V checkpoints",
            )
        batch_size = int(reference.shape[0])
        canonical_value = (
            decode_value.view(batch_size, 1, 2, 128)
            .transpose(1, 2)
            .contiguous()
        )
        layer0_keys, layer0_values = layers[0]
        if not torch.equal(layer0_keys[:, :, -1:, :], decode_key):
            raise CaptureContractError(
                "invalid_kv_value",
                "decode output appended layer-0 key differs from live M-RoPE key",
            )
        if not torch.equal(layer0_values[:, :, -1:, :], canonical_value):
            raise CaptureContractError(
                "invalid_kv_value",
                "decode output appended layer-0 value differs from canonical V",
            )
        for layer_index, (keys, values) in enumerate(layers):
            self._decode_tensors[
                f"{self._DECODE_PREFIX}.layer.{layer_index:02d}.kv.key"
            ] = keys.detach().clone()
            self._decode_tensors[
                f"{self._DECODE_PREFIX}.layer.{layer_index:02d}.kv.value"
            ] = values.detach().clone()

    def _decoder_complete(
        self,
        module: object,
        arguments: tuple[object, ...],
        keyword_arguments: dict[str, object],
        output: object,
    ) -> None:
        del module, arguments, keyword_arguments
        if output is None:
            self.close()
            return
        try:
            if self._current_forward_kind == "prefill":
                self._prefill_complete = True
            elif self._current_forward_kind == "manual_prefill":
                self._capture_manual_cache(output)
            elif self._current_forward_kind == "decode":
                self._capture_decode_cache(output)
                self._decode_complete = True
                self._awaiting_decode_logits = True
        except BaseException:
            self.close()
            raise
        finally:
            self._current_forward_kind = None
            self._layer0_attention_active = False

    @classmethod
    def _wrapped_rotary(
        cls, *args: object, **kwargs: object
    ) -> object:
        trace = cls._ACTIVE_TRACE
        if trace is None or trace._original_rotary_fn is None:
            raise CaptureContractError(
                "invalid_trace_hook", "decoder rotary wrapper lost its owning trace"
            )
        try:
            result = trace._original_rotary_fn(*args, **kwargs)
            if trace._layer0_attention_active and trace._current_forward_kind in (
                "prefill",
                "decode",
            ):
                if not isinstance(result, tuple) or len(result) != 2:
                    raise CaptureContractError(
                        "invalid_trace_hook",
                        "multimodal rotary function must return a (q, k) pair",
                    )
                q_rot, k_rot = result
                trace._capture_phase_once("decoder.layer.00.mrope.q", q_rot)
                trace._capture_phase_once("decoder.layer.00.mrope.k", k_rot)
            return result
        except BaseException:
            trace.close()
            raise

    def _lm_head_complete(
        self,
        module: object,
        arguments: tuple[object, ...],
        output: object,
    ) -> None:
        del arguments
        if output is None:
            self.close()
            return
        if not self._awaiting_decode_logits:
            return
        try:
            reference = self._reference_tensor()
            logits = _extract_hook_tensor(output)
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
                raise CaptureContractError(
                    "invalid_trace_hook",
                    "lm_head must expose a rank-2 weight tensor",
                )
            expected_shape = (int(reference.shape[0]), 1, int(weight.shape[0]))
            semantic_id = f"{self._DECODE_PREFIX}.logits"
            self._require_shape(semantic_id, logits, expected_shape)
            self._require_same_dtype(
                "decoder.layer.00.input",
                reference,
                semantic_id,
                logits,
            )
            self._require_same_device(
                "decoder.layer.00.input",
                reference,
                semantic_id,
                logits,
            )
            self._require_finite(semantic_id, logits)
            self._decode_tensors[semantic_id] = logits.detach().clone()
            self._awaiting_decode_logits = False
            self._decode_logits_complete = True
        except BaseException:
            self.close()
            raise

    def register(self, model: object) -> list[Any]:
        if self._registered:
            raise CaptureContractError(
                "invalid_trace_hook", "decoder prefill trace was registered more than once"
            )
        if _DecoderPrefillTraceCapture._ACTIVE_TRACE is not None:
            raise CaptureContractError(
                "invalid_trace_hook",
                "decoder prefill trace already owns the global rotary wrapper",
            )
        decoder = getattr(model, "model", None)
        lm_head = getattr(model, "lm_head", None)
        layers = getattr(decoder, "layers", None)
        norm = getattr(decoder, "norm", None)
        rotary = getattr(decoder, "rotary_emb", None)
        if (
            decoder is None
            or not self._module_hookable(decoder)
            or lm_head is None
            or not self._module_hookable(lm_head)
            or rotary is None
            or not self._module_hookable(rotary)
            or norm is None
            or not self._module_hookable(norm)
            or not isinstance(layers, (list, tuple, torch.nn.ModuleList))
            or len(layers) == 0
        ):
            raise CaptureContractError(
                "invalid_trace_hook",
                "decoder modules do not match the pinned prefill topology",
            )
        layer0 = layers[0]
        self_attn = getattr(layer0, "self_attn", None)
        mlp = getattr(layer0, "mlp", None)
        required_modules = (
            getattr(layer0, "input_layernorm", None),
            self_attn,
            getattr(self_attn, "q_proj", None) if self_attn is not None else None,
            getattr(self_attn, "k_proj", None) if self_attn is not None else None,
            getattr(self_attn, "v_proj", None) if self_attn is not None else None,
            getattr(self_attn, "o_proj", None) if self_attn is not None else None,
            getattr(layer0, "post_attention_layernorm", None),
            mlp,
            getattr(mlp, "gate_proj", None) if mlp is not None else None,
            getattr(mlp, "up_proj", None) if mlp is not None else None,
            getattr(mlp, "down_proj", None) if mlp is not None else None,
        )
        if any(module is None or not self._module_hookable(module) for module in required_modules):
            raise CaptureContractError(
                "invalid_trace_hook",
                "decoder layer-0 modules do not match the pinned prefill topology",
            )
        owner_module = sys.modules.get(type(decoder).__module__)
        original_rotary_fn = (
            getattr(owner_module, "apply_multimodal_rotary_pos_emb", None)
            if owner_module is not None
            else None
        )
        if owner_module is None or not callable(original_rotary_fn):
            raise CaptureContractError(
                "invalid_trace_hook",
                "decoder owner module does not expose apply_multimodal_rotary_pos_emb",
            )

        input_norm = layer0.input_layernorm
        post_norm = layer0.post_attention_layernorm
        q_proj = self_attn.q_proj
        k_proj = self_attn.k_proj
        v_proj = self_attn.v_proj
        o_proj = self_attn.o_proj
        gate_proj = mlp.gate_proj
        up_proj = mlp.up_proj
        down_proj = mlp.down_proj

        try:
            self._decoder = decoder
            self._model = model
            self._lm_head = lm_head
            self._layer_count = len(layers)
            self._owner_module = owner_module
            self._original_rotary_fn = original_rotary_fn
            if isinstance(decoder, torch.nn.Module) and isinstance(model, torch.nn.Module):
                self._sanitize_module_cycles(decoder, model)
            _DecoderPrefillTraceCapture._ACTIVE_TRACE = self
            setattr(owner_module, "apply_multimodal_rotary_pos_emb", self._wrapped_rotary)
            self._handles.append(
                decoder.register_forward_pre_hook(self._decoder_start, with_kwargs=True)
            )
            self._handles.append(rotary.register_forward_hook(self._rotary_hook))
            self._handles.append(
                self_attn.register_forward_pre_hook(
                    self._layer0_self_attn_start, with_kwargs=True
                )
            )
            self._handles.append(
                self_attn.register_forward_hook(
                    self._layer0_self_attn_complete, with_kwargs=True
                )
            )
            self._handles.append(
                input_norm.register_forward_pre_hook(
                    self._input_hook("decoder.layer.00.input")
                )
            )
            self._handles.append(
                input_norm.register_forward_hook(
                    self._output_hook("decoder.layer.00.norm1")
                )
            )
            self._handles.append(
                q_proj.register_forward_hook(self._output_hook("decoder.layer.00.q"))
            )
            self._handles.append(
                k_proj.register_forward_hook(self._output_hook("decoder.layer.00.k"))
            )
            self._handles.append(
                v_proj.register_forward_hook(self._output_hook("decoder.layer.00.v"))
            )
            self._handles.append(
                o_proj.register_forward_pre_hook(
                    self._input_hook("decoder.layer.00.attention.context")
                )
            )
            self._handles.append(
                o_proj.register_forward_hook(
                    self._output_hook("decoder.layer.00.attention.output")
                )
            )
            self._handles.append(
                post_norm.register_forward_pre_hook(
                    self._input_hook("decoder.layer.00.attention.residual")
                )
            )
            self._handles.append(
                post_norm.register_forward_hook(
                    self._output_hook("decoder.layer.00.norm2")
                )
            )
            self._handles.append(
                gate_proj.register_forward_hook(
                    self._output_hook("decoder.layer.00.mlp.gate")
                )
            )
            self._handles.append(
                up_proj.register_forward_hook(
                    self._output_hook("decoder.layer.00.mlp.up")
                )
            )
            self._handles.append(
                down_proj.register_forward_pre_hook(
                    self._input_hook("decoder.layer.00.mlp.activation")
                )
            )
            self._handles.append(
                down_proj.register_forward_hook(
                    self._output_hook("decoder.layer.00.mlp.down")
                )
            )
            for index, layer in enumerate(layers):
                self._handles.append(
                    layer.register_forward_hook(
                        self._output_hook(f"decoder.layer.{index:02d}.output")
                    )
                )
            self._handles.append(
                norm.register_forward_hook(self._output_hook("decoder.final_norm"))
            )
            self._handles.append(
                decoder.register_forward_hook(
                    self._decoder_complete,
                    with_kwargs=True,
                    always_call=True,
                )
            )
            self._handles.append(
                lm_head.register_forward_hook(
                    self._lm_head_complete,
                    always_call=True,
                )
            )
            self._registered = True
        except Exception:
            self.close()
            raise

        return list(self._handles)

    def close(self) -> None:
        while self._handles:
            self._handles.pop().remove()
        if _DecoderPrefillTraceCapture._ACTIVE_TRACE is self:
            assert self._owner_module is not None
            assert self._original_rotary_fn is not None
            setattr(
                self._owner_module,
                "apply_multimodal_rotary_pos_emb",
                self._original_rotary_fn,
            )
            _DecoderPrefillTraceCapture._ACTIVE_TRACE = None
        self._restore_module_cycles()
        self._current_forward_kind = None
        self._layer0_attention_active = False

    def _finish_legacy_tensors(self) -> dict[str, torch.Tensor]:
        expected = self._legacy_expected_semantic_ids()
        if not self._prefill_complete or not self._cache_complete:
            raise CaptureContractError(
                "incomplete_deep_trace",
                "decoder prefill trace did not capture both prefill checkpoints and cache",
                details={
                    "prefill_complete": self._prefill_complete,
                    "cache_complete": self._cache_complete,
                },
            )
        if set(self._tensors) != expected:
            raise CaptureContractError(
                "incomplete_deep_trace",
                "decoder prefill hook set is incomplete",
                details={"captured": sorted(self._tensors)},
            )
        layer0_input = self._tensors["decoder.layer.00.input"]
        if layer0_input.ndim != 3:
            raise CaptureContractError(
                "invalid_trace_shape",
                "decoder.layer.00.input must be a rank-3 tensor",
                details={"actual": list(layer0_input.shape)},
            )
        batch_size, sequence_length, hidden_size = (
            int(layer0_input.shape[0]),
            int(layer0_input.shape[1]),
            int(layer0_input.shape[2]),
        )
        if batch_size <= 0 or sequence_length <= 0 or hidden_size != 1024:
            raise CaptureContractError(
                "invalid_trace_shape",
                "decoder.layer.00.input does not match the pinned decoder ABI",
                details={"actual": [batch_size, sequence_length, hidden_size]},
            )
        self._require_floating("decoder.layer.00.input", layer0_input)
        if layer0_input.device.type == "meta":
            raise CaptureContractError(
                "invalid_trace_device",
                "decoder.layer.00.input cannot live on the meta device",
            )
        self._require_finite("decoder.layer.00.input", layer0_input)
        reference_id = "decoder.layer.00.input"
        shape_groups = {
            "decoder.rope.cos": (3, batch_size, sequence_length, 128),
            "decoder.rope.sin": (3, batch_size, sequence_length, 128),
            "decoder.layer.00.norm1": (batch_size, sequence_length, 1024),
            "decoder.layer.00.q": (batch_size, sequence_length, 2048),
            "decoder.layer.00.k": (batch_size, sequence_length, 256),
            "decoder.layer.00.v": (batch_size, sequence_length, 256),
            "decoder.layer.00.mrope.q": (batch_size, 16, sequence_length, 128),
            "decoder.layer.00.mrope.k": (batch_size, 2, sequence_length, 128),
            "decoder.layer.00.kv.key": (batch_size, 2, sequence_length, 128),
            "decoder.layer.00.kv.value": (batch_size, 2, sequence_length, 128),
            "decoder.layer.00.attention.context": (batch_size, sequence_length, 2048),
            "decoder.layer.00.attention.output": (batch_size, sequence_length, 1024),
            "decoder.layer.00.attention.residual": (batch_size, sequence_length, 1024),
            "decoder.layer.00.norm2": (batch_size, sequence_length, 1024),
            "decoder.layer.00.mlp.gate": (batch_size, sequence_length, 3072),
            "decoder.layer.00.mlp.up": (batch_size, sequence_length, 3072),
            "decoder.layer.00.mlp.activation": (batch_size, sequence_length, 3072),
            "decoder.layer.00.mlp.down": (batch_size, sequence_length, 1024),
            "decoder.final_norm": (batch_size, sequence_length, 1024),
        }
        for semantic_id, expected_shape in shape_groups.items():
            tensor = self._tensors[semantic_id]
            self._require_shape(semantic_id, tensor, expected_shape)
            self._require_same_dtype(reference_id, layer0_input, semantic_id, tensor)
            self._require_same_device(reference_id, layer0_input, semantic_id, tensor)
            self._require_finite(semantic_id, tensor)
        for index in range(self._layer_count or 0):
            semantic_id = f"decoder.layer.{index:02d}.output"
            tensor = self._tensors[semantic_id]
            self._require_shape(
                semantic_id, tensor, (batch_size, sequence_length, 1024)
            )
            self._require_same_dtype(reference_id, layer0_input, semantic_id, tensor)
            self._require_same_device(reference_id, layer0_input, semantic_id, tensor)
            self._require_finite(semantic_id, tensor)
        return {
            semantic_id: tensor.detach().contiguous().clone()
            for semantic_id, tensor in sorted(self._tensors.items())
        }

    def _validate_extended_tensors(
        self,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        reference = self._reference_tensor()
        batch_size = int(reference.shape[0])
        sequence_length = int(reference.shape[1])
        prefix = self._DECODE_PREFIX
        decode_input_id = f"{prefix}.layer.00.input"
        decode_input = tensors[decode_input_id]
        self._require_shape(decode_input_id, decode_input, (batch_size, 1, 1024))
        self._require_same_dtype(
            "decoder.layer.00.input",
            reference,
            decode_input_id,
            decode_input,
        )
        self._require_same_device(
            "decoder.layer.00.input",
            reference,
            decode_input_id,
            decode_input,
        )
        self._require_finite(decode_input_id, decode_input)

        decode_shapes = {
            f"{prefix}.rope.cos": (3, batch_size, 1, 128),
            f"{prefix}.rope.sin": (3, batch_size, 1, 128),
            f"{prefix}.layer.00.norm1": (batch_size, 1, 1024),
            f"{prefix}.layer.00.q": (batch_size, 1, 2048),
            f"{prefix}.layer.00.k": (batch_size, 1, 256),
            f"{prefix}.layer.00.v": (batch_size, 1, 256),
            f"{prefix}.layer.00.mrope.q": (batch_size, 16, 1, 128),
            f"{prefix}.layer.00.mrope.k": (batch_size, 2, 1, 128),
            f"{prefix}.layer.00.attention.context": (batch_size, 1, 2048),
            f"{prefix}.layer.00.attention.output": (batch_size, 1, 1024),
            f"{prefix}.layer.00.attention.residual": (batch_size, 1, 1024),
            f"{prefix}.layer.00.norm2": (batch_size, 1, 1024),
            f"{prefix}.layer.00.mlp.gate": (batch_size, 1, 3072),
            f"{prefix}.layer.00.mlp.up": (batch_size, 1, 3072),
            f"{prefix}.layer.00.mlp.activation": (batch_size, 1, 3072),
            f"{prefix}.layer.00.mlp.down": (batch_size, 1, 1024),
            f"{prefix}.final_norm": (batch_size, 1, 1024),
        }
        for semantic_id, expected_shape in decode_shapes.items():
            tensor = tensors[semantic_id]
            self._require_shape(semantic_id, tensor, expected_shape)
            self._require_same_dtype(
                "decoder.layer.00.input",
                reference,
                semantic_id,
                tensor,
            )
            self._require_same_device(
                "decoder.layer.00.input",
                reference,
                semantic_id,
                tensor,
            )
            self._require_finite(semantic_id, tensor)

        for layer_index in range(self._layer_count or 0):
            output_id = f"{prefix}.layer.{layer_index:02d}.output"
            output = tensors[output_id]
            self._require_shape(output_id, output, (batch_size, 1, 1024))
            self._require_same_dtype(
                "decoder.layer.00.input", reference, output_id, output
            )
            self._require_same_device(
                "decoder.layer.00.input", reference, output_id, output
            )
            self._require_finite(output_id, output)
            for cache_prefix, cache_length in (
                ("decoder", sequence_length),
                (prefix, sequence_length + 1),
            ):
                for kind in ("key", "value"):
                    cache_id = (
                        f"{cache_prefix}.layer.{layer_index:02d}.kv.{kind}"
                    )
                    cache_tensor = tensors[cache_id]
                    self._require_shape(
                        cache_id,
                        cache_tensor,
                        (batch_size, 2, cache_length, 128),
                    )
                    self._require_same_dtype(
                        "decoder.layer.00.input",
                        reference,
                        cache_id,
                        cache_tensor,
                    )
                    self._require_same_device(
                        "decoder.layer.00.input",
                        reference,
                        cache_id,
                        cache_tensor,
                    )
                    self._require_finite(cache_id, cache_tensor)

        attention_mask = self._validate_binary_mask(
            f"{prefix}.attention_mask",
            tensors[f"{prefix}.attention_mask"],
            expected_shape=(batch_size, sequence_length + 1),
            expected_device=reference.device,
        )
        if self._manual_attention_mask is None or not torch.equal(
            attention_mask[:, :-1], self._manual_attention_mask
        ):
            raise CaptureContractError(
                "invalid_trace_value",
                "captured decode attention mask lost its manual-prefill prefix",
            )
        if not bool((attention_mask[:, -1:] == 1).all().item()):
            raise CaptureContractError(
                "invalid_trace_value",
                "captured decode attention mask did not append an active token",
            )
        cache_position = self._validate_integer_tensor(
            f"{prefix}.cache_position",
            tensors[f"{prefix}.cache_position"],
            expected_shape=(1,),
            expected_device=reference.device,
        )
        expected_cache_position = torch.tensor(
            [sequence_length], dtype=torch.int64, device=reference.device
        )
        if not torch.equal(cache_position, expected_cache_position):
            raise CaptureContractError(
                "invalid_trace_value",
                "captured decode cache_position is not the append position",
            )
        position_ids = self._validate_integer_tensor(
            f"{prefix}.position_ids",
            tensors[f"{prefix}.position_ids"],
            expected_shape=(3, batch_size, 1),
            expected_device=reference.device,
        )
        if self._manual_terminal_position_ids is None or not torch.equal(
            position_ids, self._manual_terminal_position_ids + 1
        ):
            raise CaptureContractError(
                "invalid_trace_value",
                "captured decode position_ids do not append manual M-RoPE positions",
            )

        logits_id = f"{prefix}.logits"
        logits = tensors[logits_id]
        weight = getattr(self._lm_head, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise CaptureContractError(
                "invalid_trace_hook", "lm_head weight is unavailable at finish"
            )
        self._require_shape(
            logits_id,
            logits,
            (batch_size, 1, int(weight.shape[0])),
        )
        self._require_same_dtype(
            "decoder.layer.00.input", reference, logits_id, logits
        )
        self._require_same_device(
            "decoder.layer.00.input", reference, logits_id, logits
        )
        self._require_finite(logits_id, logits)

    def finish(self) -> dict[str, torch.Tensor]:
        try:
            legacy = self._finish_legacy_tensors()
            if self._decode_started and not self._decode_logits_complete:
                raise CaptureContractError(
                    "incomplete_deep_trace",
                    "target decode did not complete both decoder and lm_head",
                    details={
                        "decoder_complete": self._decode_complete,
                        "logits_complete": self._decode_logits_complete,
                    },
                )
            if not self._decode_logits_complete:
                return legacy
            if not self._decode_complete or self._manual_cache_snapshots is None:
                raise CaptureContractError(
                    "incomplete_deep_trace",
                    "extended decode trace state is incomplete",
                )

            tensors = dict(self._tensors)
            for layer_index, (keys, values) in enumerate(
                self._manual_cache_snapshots
            ):
                tensors[f"decoder.layer.{layer_index:02d}.kv.key"] = keys
                tensors[f"decoder.layer.{layer_index:02d}.kv.value"] = values
            tensors.update(self._decode_tensors)
            expected = self._expected_semantic_ids()
            if set(tensors) != expected:
                raise CaptureContractError(
                    "incomplete_deep_trace",
                    "extended decoder hook set is incomplete",
                    details={"captured": sorted(tensors)},
                )
            self._validate_extended_tensors(tensors)
            return {
                semantic_id: tensor.detach().contiguous().clone()
                for semantic_id, tensor in sorted(tensors.items())
            }
        except BaseException:
            self.close()
            raise


class _ModelTraceRecorder:
    _VISION_LAYERS = (0, 1, 13, 26)
    _VISION_EMBEDDINGS = (
        "vision.embeddings.patch",
        "vision.embeddings.output",
    )
    _VISION_LAYER_ZERO = (
        "vision.layer.00.attention.context",
        "vision.layer.00.attention.output",
        "vision.layer.00.attention.residual",
        "vision.layer.00.k",
        "vision.layer.00.mlp.activation",
        "vision.layer.00.mlp.fc1",
        "vision.layer.00.mlp.output",
        "vision.layer.00.norm1",
        "vision.layer.00.norm2",
        "vision.layer.00.q",
        "vision.layer.00.v",
        "vision.rope.frequencies",
    )

    def __init__(self, model: Any, trace_level: TraceLevel) -> None:
        self.model = model
        self.trace_level = trace_level
        self._handles: list[Any] = []
        self._stage: dict[str, torch.Tensor] = {}
        self._deep: dict[str, torch.Tensor] = {}
        self._projector_trace = _ProjectorTraceCapture()
        self._multimodal_trace = _MultimodalTraceCapture()
        self._decoder_trace = _DecoderPrefillTraceCapture()

    def _save(
        self,
        target: dict[str, torch.Tensor],
        semantic_id: str,
        value: object,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if semantic_id in target:
            return
        tensor = _extract_hook_tensor(value)
        if transform is not None:
            tensor = transform(tensor)
        target[semantic_id] = tensor.detach()

    def _output_hook(
        self,
        target: dict[str, torch.Tensor],
        semantic_id: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> Callable[[object, tuple[object, ...], object], None]:
        def hook(module: object, arguments: tuple[object, ...], output: object) -> None:
            del module, arguments
            self._save(target, semantic_id, output, transform)

        return hook

    def _input_hook(
        self,
        target: dict[str, torch.Tensor],
        semantic_id: str,
    ) -> Callable[[object, tuple[object, ...]], None]:
        def hook(module: object, arguments: tuple[object, ...]) -> None:
            del module
            if len(arguments) != 1:
                raise CaptureContractError(
                    "invalid_trace_hook",
                    f"{semantic_id} expected exactly one positional tensor input",
                    details={"argument_count": len(arguments)},
                )
            self._save(target, semantic_id, arguments[0])

        return hook

    @staticmethod
    def _canonical_patch_embedding(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim != 4 or tuple(tensor.shape[-2:]) != (1, 1):
            raise CaptureContractError(
                "invalid_trace_shape",
                "patch embedding hook must produce one vector per processor patch",
                details={"shape": list(tensor.shape)},
            )
        return tensor.flatten(-2).squeeze(-1).unsqueeze(0)

    def __enter__(self) -> _ModelTraceRecorder:
        try:
            self._handles.extend(self._multimodal_trace.register(self.model))
            self._handles.extend(
                [
                    self.model.visual.register_forward_hook(
                        self._output_hook(self._stage, "vision.final")
                    ),
                    self.model.mlp_AR.register_forward_hook(
                        self._output_hook(self._stage, "projector.final")
                    ),
                    self.model.lm_head.register_forward_hook(
                        self._output_hook(
                            self._stage,
                            "decoder.prefill.logits.last",
                            lambda tensor: tensor[:, -1, :],
                        )
                    ),
                ]
            )
            if self.trace_level is TraceLevel.L3:
                vision_embeddings = self.model.visual.vision_model.embeddings
                vision_encoder = self.model.visual.vision_model.encoder
                layer_zero = vision_encoder.layers[0]
                self._decoder_trace.register(self.model)
                self._handles.extend(self._projector_trace.register(self.model.mlp_AR))
                self._handles.extend(
                    [
                        vision_embeddings.patch_embedding.register_forward_hook(
                            self._output_hook(
                                self._deep,
                                "vision.embeddings.patch",
                                self._canonical_patch_embedding,
                            )
                        ),
                        vision_embeddings.register_forward_hook(
                            self._output_hook(
                                self._deep, "vision.embeddings.output"
                            )
                        ),
                        vision_encoder.rotary_pos_emb.register_forward_hook(
                            self._output_hook(
                                self._deep, "vision.rope.frequencies"
                            )
                        ),
                        layer_zero.layer_norm1.register_forward_hook(
                            self._output_hook(self._deep, "vision.layer.00.norm1")
                        ),
                        layer_zero.self_attn.q_proj.register_forward_hook(
                            self._output_hook(self._deep, "vision.layer.00.q")
                        ),
                        layer_zero.self_attn.k_proj.register_forward_hook(
                            self._output_hook(self._deep, "vision.layer.00.k")
                        ),
                        layer_zero.self_attn.v_proj.register_forward_hook(
                            self._output_hook(self._deep, "vision.layer.00.v")
                        ),
                        layer_zero.self_attn.out_proj.register_forward_pre_hook(
                            self._input_hook(
                                self._deep, "vision.layer.00.attention.context"
                            )
                        ),
                        layer_zero.self_attn.out_proj.register_forward_hook(
                            self._output_hook(
                                self._deep, "vision.layer.00.attention.output"
                            )
                        ),
                        layer_zero.layer_norm2.register_forward_pre_hook(
                            self._input_hook(
                                self._deep, "vision.layer.00.attention.residual"
                            )
                        ),
                        layer_zero.layer_norm2.register_forward_hook(
                            self._output_hook(self._deep, "vision.layer.00.norm2")
                        ),
                        layer_zero.mlp.fc1.register_forward_hook(
                            self._output_hook(
                                self._deep, "vision.layer.00.mlp.fc1"
                            )
                        ),
                        layer_zero.mlp.activation_fn.register_forward_hook(
                            self._output_hook(
                                self._deep, "vision.layer.00.mlp.activation"
                            )
                        ),
                        layer_zero.mlp.fc2.register_forward_hook(
                            self._output_hook(
                                self._deep, "vision.layer.00.mlp.output"
                            )
                        ),
                    ]
                )
                for index in self._VISION_LAYERS:
                    self._handles.append(
                        self.model.visual.vision_model.encoder.layers[
                            index
                        ].register_forward_hook(
                            self._output_hook(
                                self._deep, f"vision.layer.{index:02d}.output"
                            )
                        )
                    )
        except Exception:
            self.close()
            raise
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()

    def close(self) -> None:
        while self._handles:
            self._handles.pop().remove()
        self._decoder_trace.close()

    @staticmethod
    def _to_cpu(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            semantic_id: tensor.detach().cpu().contiguous().clone()
            for semantic_id, tensor in sorted(tensors.items())
        }

    def finish(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        stage = {**self._stage, **self._multimodal_trace.finish()}
        expected_stage = {
            "vision.final",
            "projector.final",
            "decoder.prefill.logits.last",
            *_MultimodalTraceCapture.SEMANTIC_IDS,
        }
        if set(stage) != expected_stage:
            raise CaptureContractError(
                "incomplete_stage_trace",
                "stage hook set is incomplete",
                details={"captured": sorted(stage)},
            )
        expected_deep = {
            *self._VISION_EMBEDDINGS,
            *self._VISION_LAYER_ZERO,
            *_ProjectorTraceCapture.SEMANTIC_IDS,
            *(f"vision.layer.{index:02d}.output" for index in self._VISION_LAYERS),
        }
        deep = self._deep
        if self.trace_level is TraceLevel.L3:
            deep = {
                **deep,
                **self._projector_trace.finish(),
                **self._decoder_trace.finish(),
            }
            expected_deep = {*expected_deep, *self._decoder_trace.semantic_ids}
            if set(deep) != expected_deep:
                raise CaptureContractError(
                    "incomplete_deep_trace",
                    "deep hook set is incomplete",
                    details={"captured": sorted(deep)},
                )
        return self._to_cpu(stage), self._to_cpu(deep)


class TransformersOracle:
    def __init__(
        self,
        snapshot: Path,
        model_lock: ModelLock,
        *,
        device: str,
        dtype: str,
    ) -> None:
        self.snapshot = snapshot.expanduser().resolve(strict=True)
        self.model_lock = model_lock
        if (
            model_lock.model_id != PINNED_MODEL_ID
            or model_lock.revision != PINNED_REVISION
            or self.snapshot.name != PINNED_REVISION
        ):
            raise CaptureContractError(
                "wrong_model_identity", "oracle requires the pinned PaddleOCR-VL-1.6"
            )
        if device not in {"cpu", "mps"}:
            raise CaptureContractError(
                "unsupported_oracle_device", f"unsupported oracle device: {device!r}"
            )
        if dtype not in _DTYPES or (device == "cpu" and dtype != "float32"):
            raise CaptureContractError(
                "unsupported_oracle_precision",
                f"unsupported {device} oracle precision: {dtype!r}",
            )
        if device == "mps" and not torch.backends.mps.is_available():
            raise CaptureContractError("unsupported_oracle_device", "MPS is unavailable")

        self.device = torch.device(device)
        self.dtype_name = dtype
        self.torch_dtype = _DTYPES[dtype]
        self.verified_model = verify_model_directory(model_lock, self.snapshot)
        self._processor: Any | None = None
        self._model: Any | None = None

    def _load_processor(self) -> Any:
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.snapshot,
                trust_remote_code=True,
                local_files_only=True,
            )
        return self._processor

    def _load_model(self) -> Any:
        if self._model is None:
            installation = install_transformers_compat(self.snapshot)
            self._model = installation.model_class.from_pretrained(
                self.snapshot,
                local_files_only=True,
                use_safetensors=True,
                dtype=self.torch_dtype,
            )
            self._model.to(self.device)
            self._model.eval()
        return self._model

    @staticmethod
    def _validate_case(case: CaseSpec) -> None:
        try:
            case.validate()
        except BundleFormatError as error:
            raise CaptureContractError(
                error.code, str(error), details=error.details
            ) from error

    def _load_case_image(self, case: CaseSpec, image_path: Path) -> Image.Image:
        self._validate_case(case)
        payload = image_path.read_bytes()
        actual_hash = f"blake3:{blake3(payload).hexdigest()}"
        if actual_hash != case.source_image_hash:
            raise CaptureContractError(
                "source_hash_mismatch",
                "image bytes do not match the smoke case",
                details={"expected": case.source_image_hash, "actual": actual_hash},
            )
        try:
            image = Image.open(io.BytesIO(payload)).convert("RGB")
        except (OSError, ValueError) as error:
            raise CaptureContractError("invalid_source", str(error)) from error
        if image.size != (case.width, case.height):
            raise CaptureContractError(
                "source_shape_mismatch",
                f"case expects {(case.width, case.height)}, got {image.size}",
            )
        if case.task == "spotting" and case.width < 1500 and case.height < 1500:
            image = image.resize(
                (case.width * 2, case.height * 2), Image.Resampling.LANCZOS
            )
        return image

    def _prepare(self, case: CaseSpec, image_path: Path) -> tuple[ProcessorCapture, Any]:
        processor = self._load_processor()
        image = self._load_case_image(case, image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": case.prompt},
                ],
            }
        ]
        batch = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": processor.image_processor.min_pixels,
                    "longest_edge": case.max_pixels,
                }
            },
        )

        input_ids_tensor = batch["input_ids"].detach().cpu()
        attention_tensor = batch["attention_mask"].detach().cpu()
        grid_tensor = batch["image_grid_thw"].detach().cpu()
        pixels = batch["pixel_values"].detach().cpu().contiguous()
        if (
            tuple(input_ids_tensor.shape[:1]) != (1,)
            or tuple(attention_tensor.shape) != tuple(input_ids_tensor.shape)
            or tuple(grid_tensor.shape) != (1, 3)
            or pixels.ndim != 4
        ):
            raise CaptureContractError(
                "processor_shape_mismatch", "official processor returned a non-singleton batch"
            )

        input_ids = tuple(int(value) for value in input_ids_tensor[0].tolist())
        placeholder_id = int(processor.tokenizer.image_token_id)
        merge_size = int(processor.image_processor.merge_size)
        pixel_array = pixels.numpy()
        capture = ProcessorCapture(
            case_id=case.case_id,
            input_ids=input_ids,
            attention_mask=tuple(int(value) for value in attention_tensor[0].tolist()),
            image_grid_thw=tuple(int(value) for value in grid_tensor[0].tolist()),
            pixel_values_shape=tuple(int(value) for value in pixels.shape),
            placeholder_id=placeholder_id,
            placeholder_count=input_ids.count(placeholder_id),
            spatial_merge_size=merge_size,
            pixel_values_digest=f"blake3:{blake3(pixel_array.tobytes()).hexdigest()}",
            pixel_min=float(pixel_array.min()),
            pixel_max=float(pixel_array.max()),
            pixel_mean=float(pixel_array.mean()),
            pixel_std=float(pixel_array.std()),
        )
        self._validate_processor_capture(capture)
        return capture, batch

    @staticmethod
    def _validate_processor_capture(capture: ProcessorCapture) -> None:
        patch_count = int(np.prod(capture.image_grid_thw))
        expected_placeholders = patch_count // (capture.spatial_merge_size**2)
        if capture.pixel_values_shape[0] != patch_count:
            raise CaptureContractError(
                "patch_grid_mismatch", "processor patch tensor differs from image grid"
            )
        if capture.placeholder_count != expected_placeholders:
            raise CaptureContractError(
                "placeholder_mismatch", "processor image placeholder expansion is inconsistent"
            )

    def capture_processor(
        self, case: CaseSpec, image_path: Path
    ) -> ProcessorCapture:
        capture, _ = self._prepare(case, image_path)
        return capture

    def _enable_determinism(self) -> None:
        random.seed(_DEFAULT_SEED)
        np.random.seed(_DEFAULT_SEED)
        torch.manual_seed(_DEFAULT_SEED)
        torch.use_deterministic_algorithms(True)

    def _synchronize(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()

    def _device_batch(self, batch: Any) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    @staticmethod
    def _kv_shapes(cache: object) -> tuple[tuple[int, int, int, int], ...]:
        shapes: list[tuple[int, int, int, int]] = []
        for layer in getattr(cache, "layers", ()):
            keys = getattr(layer, "keys", None)
            if keys is None or keys.ndim != 4 or keys.numel() == 0:
                continue
            shapes.append(tuple(int(axis) for axis in keys.shape))
        if not shapes:
            raise CaptureContractError(
                "invalid_kv_shape", "manual decode did not produce a DynamicCache"
            )
        return tuple(shapes)

    @staticmethod
    def _top_tokens(logits: torch.Tensor) -> tuple[tuple[int, float], ...]:
        count = min(_TOP_TOKEN_COUNT, int(logits.shape[-1]))
        values, indexes = torch.topk(logits.float(), k=count, dim=-1)
        entries = [
            (int(token), float(score))
            for token, score in zip(
                indexes.detach().cpu().tolist(),
                values.detach().cpu().tolist(),
                strict=True,
            )
        ]
        entries.sort(key=lambda item: (-item[1], item[0]))
        return tuple(entries)

    def _manual_generate(
        self,
        model: Any,
        batch: dict[str, torch.Tensor],
        *,
        max_new_tokens: int,
    ) -> GenerationTrace:
        prompt_ids = batch["input_ids"]
        prompt_length = int(prompt_ids.shape[-1])
        attention_mask = batch["attention_mask"]
        pixel_values = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"]

        model.rope_deltas = None
        past_key_values = None
        previous_token: torch.Tensor | None = None
        tokens: list[int] = []
        steps: list[GenerationStep] = []

        for step_index in range(max_new_tokens):
            if step_index == 0:
                input_ids = prompt_ids
                cache_position = initial_cache_position(prompt_length, self.device)
                step_pixels = pixel_values
            else:
                assert previous_token is not None
                input_ids = previous_token.view(1, 1)
                cache_position = torch.tensor(
                    [prompt_length + step_index - 1],
                    dtype=torch.int64,
                    device=self.device,
                )
                step_pixels = None

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                pixel_values=step_pixels,
                image_grid_thw=image_grid_thw,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[0, -1]
            chosen = int(torch.argmax(logits).item())
            top_tokens = self._top_tokens(logits)
            rope_delta = int(outputs.rope_deltas[0, 0].item())
            token_cache_position = prompt_length + step_index
            token_position = token_cache_position + rope_delta
            step = GenerationStep(
                step=step_index,
                input_token=(
                    int(prompt_ids[0, -1].item())
                    if step_index == 0
                    else tokens[-1]
                ),
                position_ids=(token_position, token_position, token_position),
                cache_position=token_cache_position,
                rope_delta=rope_delta,
                top_tokens=top_tokens,
                chosen_token=chosen,
                kv_shapes=self._kv_shapes(outputs.past_key_values),
            )
            step.validate()
            tokens.append(chosen)
            steps.append(step)
            past_key_values = outputs.past_key_values
            previous_token = torch.tensor(chosen, dtype=torch.int64, device=self.device)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=-1,
            )

        trace = GenerationTrace(tokens=tuple(tokens), steps=tuple(steps))
        trace.validate()
        return trace

    @staticmethod
    def _validate_generation_limit(case: CaseSpec, max_new_tokens: int) -> None:
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
            or max_new_tokens > case.max_new_tokens
        ):
            raise CaptureContractError(
                "invalid_generation", "max_new_tokens exceeds the case contract"
            )

    def _compare_prepared(
        self,
        processor_capture: ProcessorCapture,
        host_batch: Any,
        *,
        max_new_tokens: int,
    ) -> GenerationComparison:
        processor = self._load_processor()
        model = self._load_model()
        batch = self._device_batch(host_batch)
        prompt_length = int(batch["input_ids"].shape[-1])

        self._enable_determinism()
        model.rope_deltas = None
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,
                cache_position=initial_cache_position(prompt_length, self.device),
            )
            self._synchronize()
            generate_tokens = tuple(
                int(token) for token in generated[0, prompt_length:].detach().cpu().tolist()
            )

            self._enable_determinism()
            manual_trace = self._manual_generate(
                model,
                batch,
                max_new_tokens=max_new_tokens,
            )
            self._synchronize()

        assert_generation_parity(generate_tokens, manual_trace.tokens)
        decoded = processor.decode(generate_tokens, skip_special_tokens=True)
        return GenerationComparison(
            processor=processor_capture,
            generate_tokens=generate_tokens,
            manual_trace=manual_trace,
            decoded_text=decoded,
        )

    def compare_generate_and_manual(
        self,
        case: CaseSpec,
        image_path: Path,
        *,
        max_new_tokens: int,
    ) -> GenerationComparison:
        self._validate_generation_limit(case, max_new_tokens)
        processor_capture, host_batch = self._prepare(case, image_path)
        return self._compare_prepared(
            processor_capture,
            host_batch,
            max_new_tokens=max_new_tokens,
        )

    def capture_artifacts(
        self,
        case: CaseSpec,
        image_path: Path,
        *,
        max_new_tokens: int,
        trace_level: TraceLevel,
    ) -> OracleCaptureResult:
        self._validate_generation_limit(case, max_new_tokens)
        if not isinstance(trace_level, TraceLevel):
            raise CaptureContractError("invalid_trace_level", "unknown trace level")
        processor_capture, host_batch = self._prepare(case, image_path)
        model = self._load_model()
        recorder = _ModelTraceRecorder(model, trace_level)
        with recorder:
            comparison = self._compare_prepared(
                processor_capture,
                host_batch,
                max_new_tokens=max_new_tokens,
            )
            self._synchronize()
            stage_tensors, deep_tensors = recorder.finish()

        processor_tensors = {
            f"processor.{name}": host_batch[name]
            .detach()
            .cpu()
            .contiguous()
            .clone()
            for name in (
                "attention_mask",
                "image_grid_thw",
                "input_ids",
                "pixel_values",
            )
        }
        captured = CapturedArtifacts(
            processor_tensors=dict(sorted(processor_tensors.items())),
            stage_tensors=stage_tensors,
            deep_tensors=deep_tensors,
            token_trace=comparison.manual_trace,
        )
        return OracleCaptureResult(comparison=comparison, captured=captured)
