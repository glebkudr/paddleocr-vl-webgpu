from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pvlc_reference.capture import CaptureContractError
from pvlc_reference.transformers_compat import (
    COMPATIBILITY_SHIM_ID,
    SUPPORTED_TRANSFORMERS_VERSION,
    adapt_create_causal_mask,
    assert_supported_transformers_version,
    compute_default_rope_parameters,
    initial_cache_position,
    install_transformers_compat,
)


REPO_ROOT = Path(__file__).parents[3]
SNAPSHOT = (
    REPO_ROOT
    / "models"
    / "snapshots"
    / "66317acc4c9fc17bd154591ce650735cd2855f3e"
)


@dataclass(frozen=True)
class TinyRopeConfig:
    rope_theta: float = 10_000.0
    hidden_size: int = 8
    num_attention_heads: int = 1
    head_dim: int = 8
    partial_rotary_factor: float = 1.0


def test_default_rope_shim_is_the_independent_standard_formula() -> None:
    frequencies, attention_factor = compute_default_rope_parameters(
        TinyRopeConfig(), device="cpu"
    )

    assert attention_factor == 1.0
    torch.testing.assert_close(
        frequencies,
        torch.tensor([1.0, 0.1, 0.01, 0.001], dtype=torch.float32),
        rtol=1e-6,
        atol=1e-7,
    )


def test_default_rope_shim_rejects_ambiguous_config_and_keyword_inputs() -> None:
    with pytest.raises(CaptureContractError) as caught:
        compute_default_rope_parameters(
            TinyRopeConfig(), device="cpu", base=10_000.0, dim=8
        )

    assert caught.value.code == "ambiguous_rope_parameters"


def test_mask_adapter_drops_only_removed_cache_position_argument() -> None:
    received: dict[str, object] = {}

    def current_mask(**kwargs: object) -> str:
        received.update(kwargs)
        return "mask"

    adapter = adapt_create_causal_mask(current_mask)
    result = adapter(
        config="config",
        inputs_embeds="embeddings",
        attention_mask="attention",
        past_key_values="cache",
        position_ids="positions",
        cache_position="legacy-cache-position",
    )

    assert result == "mask"
    assert received == {
        "config": "config",
        "inputs_embeds": "embeddings",
        "attention_mask": "attention",
        "past_key_values": "cache",
        "position_ids": "positions",
    }


def test_initial_cache_position_is_exact_contiguous_prompt_range() -> None:
    positions = initial_cache_position(sequence_length=5, device="cpu")

    assert positions.dtype == torch.int64
    assert positions.tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("version", ["4.57.6", "5.14.0", "5.15.0"])
def test_shim_refuses_unreviewed_transformers_versions(version: str) -> None:
    with pytest.raises(CaptureContractError) as caught:
        assert_supported_transformers_version(version)

    assert caught.value.code == "unsupported_transformers_version"
    assert SUPPORTED_TRANSFORMERS_VERSION == "5.14.1"


@pytest.mark.oracle
def test_installation_patches_only_the_pinned_remote_module_and_is_idempotent() -> None:
    if not SNAPSHOT.exists():
        pytest.skip("pinned model snapshot is not present")

    first = install_transformers_compat(SNAPSHOT)
    second = install_transformers_compat(SNAPSHOT)
    module = importlib.import_module(first.model_class.__module__)

    assert first == second
    assert first.shim_id == COMPATIBILITY_SHIM_ID
    assert first.shim_id == "paddleocr-vl-1.6/transformers-v5-abi@1"
    assert first.model_class.__name__ == "PaddleOCRVLForConditionalGeneration"
    assert first.model_class.__module__.startswith("transformers_modules.")
    assert module.RotaryEmbedding.compute_default_rope_parameters is not None
    assert module.create_causal_mask.__name__ == "compat_create_causal_mask"
