from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import transformers
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .capture import CaptureContractError
from .model_lock import PINNED_REVISION
from .trace_bundle import COMPATIBILITY_SHIM_ID


SUPPORTED_TRANSFORMERS_VERSION = "5.14.1"
_REMOTE_MODEL_REFERENCE = (
    "modeling_paddleocr_vl.PaddleOCRVLForConditionalGeneration"
)


@dataclass(frozen=True, slots=True)
class CompatibilityInstallation:
    shim_id: str
    model_class: type


def assert_supported_transformers_version(version: str | None = None) -> None:
    actual = transformers.__version__ if version is None else version
    if actual != SUPPORTED_TRANSFORMERS_VERSION:
        raise CaptureContractError(
            "unsupported_transformers_version",
            f"compatibility shim is reviewed only for Transformers "
            f"{SUPPORTED_TRANSFORMERS_VERSION}, got {actual}",
            details={
                "expected": SUPPORTED_TRANSFORMERS_VERSION,
                "actual": actual,
            },
        )


def compute_default_rope_parameters(
    config: object | None = None,
    device: str | torch.device | None = None,
    seq_len: int | None = None,
    **rope_kwargs: object,
) -> tuple[torch.Tensor, float]:
    """Compute the unscaled RoPE frequencies used by the pinned remote model.

    Transformers v5 removed the ``default`` registry entry while this snapshot
    still selects it.  The formula is deliberately local and minimal: no
    scaling policy and no inference from a mixture of config and legacy kwargs.
    ``seq_len`` remains accepted because the remote dynamic-update hook passes it.
    """

    del seq_len
    if config is not None and rope_kwargs:
        raise CaptureContractError(
            "ambiguous_rope_parameters",
            "RoPE parameters cannot mix a config with legacy keyword values",
            details={"keywords": sorted(rope_kwargs)},
        )
    if config is None:
        expected = {"base", "dim"}
        if set(rope_kwargs) != expected:
            raise CaptureContractError(
                "ambiguous_rope_parameters",
                "legacy RoPE parameters require exactly base and dim",
                details={"keywords": sorted(rope_kwargs)},
            )
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    else:
        base = getattr(config, "rope_theta", None)
        dim = getattr(config, "head_dim", None)
        if dim is None:
            hidden_size = getattr(config, "hidden_size", None)
            num_heads = getattr(config, "num_attention_heads", None)
            if (
                not isinstance(hidden_size, int)
                or isinstance(hidden_size, bool)
                or not isinstance(num_heads, int)
                or isinstance(num_heads, bool)
                or num_heads <= 0
                or hidden_size % num_heads != 0
            ):
                raise CaptureContractError(
                    "ambiguous_rope_parameters", "cannot derive the RoPE head dimension"
                )
            dim = hidden_size // num_heads
        partial_factor = getattr(config, "partial_rotary_factor", 1.0)
        if isinstance(partial_factor, bool) or not isinstance(
            partial_factor, (int, float)
        ):
            raise CaptureContractError(
                "ambiguous_rope_parameters", "partial_rotary_factor must be numeric"
            )
        dim = int(dim * partial_factor) if isinstance(dim, int) else None

    if (
        isinstance(base, bool)
        or not isinstance(base, (int, float))
        or base <= 0
        or isinstance(dim, bool)
        or not isinstance(dim, int)
        or dim <= 0
        or dim % 2 != 0
    ):
        raise CaptureContractError(
            "ambiguous_rope_parameters", "default RoPE needs a positive base and even dim"
        )

    indexes = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()
    inv_freq = 1.0 / (float(base) ** (indexes / dim))
    return inv_freq, 1.0


def adapt_create_causal_mask(current_mask: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(current_mask, "_pvlc_compat_shim", None) == COMPATIBILITY_SHIM_ID:
        return current_mask

    def compat_create_causal_mask(*args: object, **kwargs: object) -> Any:
        forwarded = dict(kwargs)
        forwarded.pop("cache_position", None)
        return current_mask(*args, **forwarded)

    compat_create_causal_mask._pvlc_compat_shim = COMPATIBILITY_SHIM_ID  # type: ignore[attr-defined]
    return compat_create_causal_mask


def initial_cache_position(
    sequence_length: int, device: str | torch.device
) -> torch.Tensor:
    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length <= 0
    ):
        raise CaptureContractError(
            "invalid_cache_position", "prompt sequence length must be positive"
        )
    return torch.arange(sequence_length, dtype=torch.int64, device=device)


def _validate_snapshot_identity(snapshot: Path) -> Path:
    resolved = snapshot.expanduser().resolve(strict=True)
    if resolved.name != PINNED_REVISION:
        raise CaptureContractError(
            "wrong_model_identity", "compatibility shim only supports the pinned revision"
        )
    config_path = resolved / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CaptureContractError(
            "wrong_model_identity", f"cannot inspect pinned model config: {error}"
        ) from error
    if (
        config.get("model_type") != "paddleocr_vl"
        or config.get("auto_map", {}).get("AutoModelForCausalLM")
        != _REMOTE_MODEL_REFERENCE
    ):
        raise CaptureContractError(
            "wrong_model_identity", "snapshot has an unexpected remote model mapping"
        )
    return resolved


def install_transformers_compat(snapshot: Path) -> CompatibilityInstallation:
    assert_supported_transformers_version()
    resolved = _validate_snapshot_identity(snapshot)
    model_class = get_class_from_dynamic_module(
        _REMOTE_MODEL_REFERENCE,
        str(resolved),
        local_files_only=True,
    )
    module = importlib.import_module(model_class.__module__)

    module.ROPE_INIT_FUNCTIONS["default"] = compute_default_rope_parameters
    module.RotaryEmbedding.compute_default_rope_parameters = staticmethod(
        compute_default_rope_parameters
    )
    module.create_causal_mask = adapt_create_causal_mask(module.create_causal_mask)

    return CompatibilityInstallation(
        shim_id=COMPATIBILITY_SHIM_ID,
        model_class=model_class,
    )
