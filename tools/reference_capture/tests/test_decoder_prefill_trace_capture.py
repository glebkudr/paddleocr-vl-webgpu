import importlib
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from pvlc_reference.transformers_oracle import _DecoderPrefillTraceCapture


_BATCH = 1
_SEQ = 2
_HIDDEN = 1024
_Q_HEADS = 16
_KV_HEADS = 2
_HEAD_DIM = 128
_Q_FLAT = _Q_HEADS * _HEAD_DIM
_KV_FLAT = _KV_HEADS * _HEAD_DIM
_INTERMEDIATE = 3072
_MROPE_SECTION = [16, 24, 24]
_MROPE_SECTIONS = _MROPE_SECTION * 2
_FAKE_LAYER_COUNT = 2
_FAKE_VOCAB_SIZE = 17
_DROP_CACHE_OUTPUT = object()
_EXPECTED_CAPTURE_KEYS = {
    "decoder.rope.cos",
    "decoder.rope.sin",
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
    "decoder.final_norm",
} | {f"decoder.layer.{index:02d}.output" for index in range(_FAKE_LAYER_COUNT)}
_ALL_PREFILL_CACHE_KEYS = {
    f"decoder.layer.{index:02d}.kv.{kind}"
    for index in range(_FAKE_LAYER_COUNT)
    for kind in ("key", "value")
}
_DECODE_PIPELINE_KEYS = {
    f"decoder.decode.00.{semantic_id.removeprefix('decoder.')}"
    for semantic_id in _EXPECTED_CAPTURE_KEYS
    if ".kv." not in semantic_id
}
_DECODE_CACHE_KEYS = {
    f"decoder.decode.00.layer.{index:02d}.kv.{kind}"
    for index in range(_FAKE_LAYER_COUNT)
    for kind in ("key", "value")
}
_DECODE_METADATA_KEYS = {
    "decoder.decode.00.attention_mask",
    "decoder.decode.00.cache_position",
    "decoder.decode.00.position_ids",
}
_EXPECTED_M6_CAPTURE_KEYS = (
    _EXPECTED_CAPTURE_KEYS
    | _ALL_PREFILL_CACHE_KEYS
    | _DECODE_PIPELINE_KEYS
    | _DECODE_CACHE_KEYS
    | _DECODE_METADATA_KEYS
    | {"decoder.decode.00.logits"}
)
_M6_SCENARIOS = (
    {
        "id": "ordinal_3_dense_prefix",
        "seed": 41,
        "input_offset": 0.0,
        "axis_starts": (40, 140, 240),
        "prompt_mask": (1, 1),
        "prefill_decoys": (),
        "post_prefill_decoys": (),
        "decode_ordinal": 3,
    },
    {
        "id": "ordinal_5_left_padded",
        "seed": 73,
        "input_offset": 17.0,
        "axis_starts": (70, 170, 370),
        "prompt_mask": (0, 1),
        "prefill_decoys": ("uncached",),
        "post_prefill_decoys": ("manual",),
        "decode_ordinal": 5,
    },
    {
        "id": "ordinal_6_reordered_decoys",
        "seed": 101,
        "input_offset": 31.0,
        "axis_starts": (15, 215, 515),
        "prompt_mask": (1, 1),
        "prefill_decoys": (),
        "post_prefill_decoys": ("manual", "uncached", "manual"),
        "decode_ordinal": 6,
    },
)
_DEFAULT_M6_SCENARIO = _M6_SCENARIOS[0]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section,
    unsqueeze_dim: int = 1,
):
    assert list(mrope_section) == _MROPE_SECTION
    cos = torch.cat(
        [chunk[index % 3] for index, chunk in enumerate(cos.split(_MROPE_SECTIONS, dim=-1))],
        dim=-1,
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [chunk[index % 3] for index, chunk in enumerate(sin.split(_MROPE_SECTIONS, dim=-1))],
        dim=-1,
    ).unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def _reshape_q(x: torch.Tensor) -> torch.Tensor:
    batch, sequence, _ = x.shape
    return x.view(batch, sequence, _Q_HEADS, _HEAD_DIM).transpose(1, 2).contiguous()


def _reshape_kv(x: torch.Tensor) -> torch.Tensor:
    batch, sequence, _ = x.shape
    return x.view(batch, sequence, _KV_HEADS, _HEAD_DIM).transpose(1, 2).contiguous()


def _flatten_context(x: torch.Tensor) -> torch.Tensor:
    batch, _, sequence, _ = x.shape
    return x.transpose(1, 2).contiguous().view(batch, sequence, _Q_FLAT)


def _fake_causal_mask(
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    *,
    batch: int,
    key_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if (
        not isinstance(attention_mask, torch.Tensor)
        or attention_mask.device.type == "meta"
        or attention_mask.ndim != 2
        or tuple(attention_mask.shape) != (batch, key_length)
        or attention_mask.dtype not in (torch.int32, torch.int64, torch.bool)
    ):
        key_is_active = torch.ones((batch, key_length), dtype=torch.bool, device=device)
    else:
        key_is_active = attention_mask.to(device=device, dtype=torch.bool)
    key_indexes = torch.arange(key_length, device=device).view(1, 1, 1, key_length)
    query_positions = cache_position.to(device=device, dtype=torch.int64).view(1, 1, -1, 1)
    allowed = (key_indexes <= query_positions) & key_is_active.view(batch, 1, 1, key_length)
    if bool(allowed.all()):
        return None
    return torch.where(
        allowed,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), -10_000.0, dtype=dtype, device=device),
    )


def _clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone()


def _cache_snapshot(cache: DynamicCache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple((_clone(layer.keys), _clone(layer.values)) for layer in cache.layers)


def _clone_cache(cache: DynamicCache) -> DynamicCache:
    clone = DynamicCache()
    for layer_index, (keys, values) in enumerate(_cache_snapshot(cache)):
        clone.update(keys, values, layer_index)
    return clone


class _Recorder:
    def __init__(self):
        self.by_tag: dict[str, dict[str, torch.Tensor]] = {}
        self.source_by_tag: dict[str, dict[str, torch.Tensor]] = {}

    def record(self, tag: str, key: str, tensor: torch.Tensor) -> None:
        self.by_tag.setdefault(tag, {})[key] = _clone(tensor)
        self.source_by_tag.setdefault(tag, {})[key] = tensor


class _FakeRMSNorm(torch.nn.Module):
    def __init__(self, weight_offset: float):
        super().__init__()
        weight = torch.linspace(1.0 + weight_offset, 2.0 + weight_offset, _HIDDEN, dtype=torch.float32)
        self.register_buffer("weight", weight)
        self.variance_epsilon = 1e-5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class _FakeRotaryEmbedding(torch.nn.Module):
    def __init__(self, recorder: _Recorder, owner):
        super().__init__()
        self._recorder = recorder
        self._owner = owner

    def forward(self, hidden_states: torch.Tensor, position_ids=None):
        if not isinstance(position_ids, torch.Tensor):
            raise AssertionError("fake rotary requires effective position_ids")
        base = hidden_states.sum(dim=-1, keepdim=True).unsqueeze(0) / 1_000.0
        frequency = (
            torch.arange(
                1,
                _HEAD_DIM + 1,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            ).view(1, 1, 1, _HEAD_DIM)
            / 100.0
        )
        positions = position_ids.to(dtype=hidden_states.dtype).unsqueeze(-1)
        axis_offsets = torch.tensor(
            [100.0, 200.0, 300.0],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).view(3, 1, 1, 1)
        cos = base + axis_offsets + (positions * frequency)
        sin = base + (axis_offsets * 2.0) - (positions * (frequency + 0.5))
        if self._owner.active_tag is not None:
            self._recorder.record(self._owner.active_tag, "decoder.rope.cos", cos)
            self._recorder.record(self._owner.active_tag, "decoder.rope.sin", sin)
        return cos, sin


class _FakeMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(_HIDDEN, _INTERMEDIATE, bias=False)
        self.up_proj = torch.nn.Linear(_HIDDEN, _INTERMEDIATE, bias=False)
        self.act_fn = torch.nn.SiLU()
        self.down_proj = torch.nn.Linear(_INTERMEDIATE, _HIDDEN, bias=False)


class _FakeSelfAttention(torch.nn.Module):
    def __init__(self, layer_idx: int, recorder: _Recorder, owner):
        super().__init__()
        self.layer_idx = layer_idx
        self._recorder = recorder
        self._owner = owner
        self.q_proj = torch.nn.Linear(_HIDDEN, _Q_FLAT, bias=False)
        self.k_proj = torch.nn.Linear(_HIDDEN, _KV_FLAT, bias=False)
        self.v_proj = torch.nn.Linear(_HIDDEN, _KV_FLAT, bias=False)
        self.o_proj = torch.nn.Linear(_Q_FLAT, _HIDDEN, bias=False)

    def _record(self, key: str, tensor: torch.Tensor) -> None:
        if self._owner.active_tag is None:
            return
        self._recorder.record(self._owner.active_tag, key, tensor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        if self._owner.raise_in_layer0 and self.layer_idx == 0:
            raise RuntimeError("layer0 boom")
        if self.layer_idx == 0 and self._owner.active_tag is not None:
            metadata = {
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "position_ids": position_ids,
            }
            self._owner.self_attn_metadata_source_by_tag[self._owner.active_tag] = metadata
            self._owner.self_attn_metadata_by_tag[self._owner.active_tag] = {
                name: (_clone(tensor) if isinstance(tensor, torch.Tensor) else tensor)
                for name, tensor in metadata.items()
            }

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q_heads = _reshape_q(q)
        k_heads = _reshape_kv(k)
        v_heads = _reshape_kv(v)
        cos, sin = position_embeddings

        q_rot, k_rot = apply_multimodal_rotary_pos_emb(
            q_heads,
            k_heads,
            cos,
            sin,
            _MROPE_SECTION,
            unsqueeze_dim=1,
        )
        if self._owner.double_mrope_layer0 and self.layer_idx == 0:
            apply_multimodal_rotary_pos_emb(
                q_heads,
                k_heads,
                cos,
                sin,
                _MROPE_SECTION,
                unsqueeze_dim=1,
            )

        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            attention_k, attention_v = past_key_values.update(
                k_rot,
                v_heads,
                self.layer_idx,
            )
        else:
            attention_k, attention_v = k_rot, v_heads

        expanded_k = attention_k.repeat_interleave(_Q_HEADS // _KV_HEADS, dim=1)
        expanded_v = attention_v.repeat_interleave(_Q_HEADS // _KV_HEADS, dim=1)
        scores = torch.matmul(q_rot, expanded_k.transpose(-1, -2)) / (_HEAD_DIM**0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = torch.softmax(scores, dim=-1)
        context_4d = torch.matmul(weights, expanded_v)
        context_flat = _flatten_context(context_4d)
        output = self.o_proj(context_flat)

        if self.layer_idx == 0:
            self._record("decoder.layer.00.q", q)
            self._record("decoder.layer.00.k", k)
            self._record("decoder.layer.00.v", v)
            self._record("decoder.layer.00.mrope.q", q_rot)
            self._record("decoder.layer.00.mrope.k", k_rot)
            self._record("decoder.layer.00.attention.context", context_flat)
            self._record("decoder.layer.00.attention.output", output)

        return output, None


class _FakeDecoderLayer(torch.nn.Module):
    def __init__(self, layer_idx: int, recorder: _Recorder, owner):
        super().__init__()
        self.layer_idx = layer_idx
        self._recorder = recorder
        self._owner = owner
        self.input_layernorm = _FakeRMSNorm(weight_offset=0.1 + (layer_idx * 0.01))
        self.self_attn = _FakeSelfAttention(layer_idx=layer_idx, recorder=recorder, owner=owner)
        self.post_attention_layernorm = _FakeRMSNorm(weight_offset=0.6 + (layer_idx * 0.01))
        self.mlp = _FakeMLP()

    def _record(self, key: str, tensor: torch.Tensor) -> None:
        if self._owner.active_tag is None:
            return
        self._recorder.record(self._owner.active_tag, key, tensor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        norm1 = self.input_layernorm(hidden_states)
        attn_output, _ = self.self_attn(
            norm1,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
        )
        attention_residual = residual + attn_output
        norm2 = self.post_attention_layernorm(attention_residual)
        gate = self.mlp.gate_proj(norm2)
        up = self.mlp.up_proj(norm2)
        activation = self.mlp.act_fn(gate) * up
        down = self.mlp.down_proj(activation)
        output = attention_residual + down

        if self.layer_idx == 0:
            self._record("decoder.layer.00.input", residual)
            self._record("decoder.layer.00.norm1", norm1)
            self._record("decoder.layer.00.attention.residual", attention_residual)
            self._record("decoder.layer.00.norm2", norm2)
            self._record("decoder.layer.00.mlp.gate", gate)
            self._record("decoder.layer.00.mlp.up", up)
            self._record("decoder.layer.00.mlp.activation", activation)
            self._record("decoder.layer.00.mlp.down", down)
        self._record(f"decoder.layer.{self.layer_idx:02d}.output", output)

        return output


class _FakeDecoderModel(torch.nn.Module):
    def __init__(self, recorder: _Recorder, owner):
        super().__init__()
        self._recorder = recorder
        self._owner = owner
        self.rotary_emb = _FakeRotaryEmbedding(recorder=recorder, owner=owner)
        self.layers = torch.nn.ModuleList(
            [_FakeDecoderLayer(layer_idx=index, recorder=recorder, owner=owner) for index in range(2)]
        )
        self.norm = _FakeRMSNorm(weight_offset=0.9)

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        if (hidden_states is None) == (inputs_embeds is None):
            raise AssertionError("fake decoder requires exactly one hidden-state input")
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        assert hidden_states is not None
        if cache_position is not None:
            raise AssertionError("pinned outer ABI must not pass cache_position to inner decoder")
        if self._owner.input_cache_repair is not None:
            past_key_values = self._owner.input_cache_repair(past_key_values)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past_length = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        effective_cache_position = torch.arange(
            past_length,
            past_length + hidden_states.shape[1],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        if position_ids is None:
            position_ids = effective_cache_position.view(1, 1, -1).expand(
                3,
                hidden_states.shape[0],
                -1,
            )
        if self._owner.active_tag is not None:
            decoder_metadata = {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            self._owner.decoder_metadata_source_by_tag[self._owner.active_tag] = decoder_metadata
            self._owner.decoder_metadata_by_tag[self._owner.active_tag] = {
                name: (_clone(tensor) if isinstance(tensor, torch.Tensor) else tensor)
                for name, tensor in decoder_metadata.items()
            }
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        key_length = past_length + hidden_states.shape[1]
        causal_mask = _fake_causal_mask(
            attention_mask,
            effective_cache_position,
            batch=hidden_states.shape[0],
            key_length=key_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self_attn_metadata = {
            "attention_mask": causal_mask,
            "cache_position": effective_cache_position,
            "position_ids": position_ids,
        }
        if self._owner.self_attn_metadata_mutator is not None:
            self._owner.self_attn_metadata_mutator(self_attn_metadata)
        layer_sequence = [0, 0, 1] if self._owner.repeat_layer0_once else [0, 1]

        for layer_index in layer_sequence:
            hidden_states = self.layers[layer_index](
                hidden_states,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **self_attn_metadata,
            )

        hidden_states = self.norm(hidden_states)
        if self._owner.active_tag is not None:
            self._recorder.record(self._owner.active_tag, "decoder.final_norm", hidden_states)

        if use_cache and self._owner.cache_mutator is not None:
            replacement_cache = self._owner.cache_mutator(past_key_values)
            if replacement_cache is _DROP_CACHE_OUTPUT:
                past_key_values = None
            elif replacement_cache is not None:
                past_key_values = replacement_cache

        return hidden_states, past_key_values


class _FakeOuterModel(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.recorder = _Recorder()
        self.active_tag: str | None = None
        self.cache_mutator = None
        self.input_cache_repair = None
        self.self_attn_metadata_mutator = None
        self.repeat_layer0_once = False
        self.double_mrope_layer0 = False
        self.raise_in_layer0 = False
        self.position_axis_starts = (0, 100, 200)
        self.decoder_forward_count = 0
        self.decoder_forward_ordinal_by_tag: dict[str, int] = {}
        self.decoder_metadata_by_tag: dict[str, dict[str, torch.Tensor]] = {}
        self.decoder_metadata_source_by_tag: dict[str, dict[str, torch.Tensor]] = {}
        self.self_attn_metadata_by_tag: dict[str, dict[str, torch.Tensor]] = {}
        self.self_attn_metadata_source_by_tag: dict[str, dict[str, torch.Tensor]] = {}
        self.model = _FakeDecoderModel(recorder=self.recorder, owner=self)
        self.lm_head = torch.nn.Linear(_HIDDEN, _FAKE_VOCAB_SIZE, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        self.decoder_forward_count += 1
        if self.active_tag is not None:
            self.decoder_forward_ordinal_by_tag[self.active_tag] = self.decoder_forward_count
        if position_ids is None:
            if cache_position is None:
                start = (
                    past_key_values.get_seq_length()
                    if past_key_values is not None
                    else 0
                )
                cache_position = torch.arange(
                    start,
                    start + hidden_states.shape[1],
                    dtype=torch.int64,
                    device=hidden_states.device,
                )
            axis_starts = torch.tensor(
                self.position_axis_starts,
                dtype=torch.int64,
                device=hidden_states.device,
            ).view(3, 1, 1)
            position_ids = axis_starts + cache_position.view(1, 1, -1)
        last_hidden_state, past_key_values = self.model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
        )
        logits = self.lm_head(last_hidden_state)
        if self.active_tag is not None:
            self.recorder.record(self.active_tag, "decoder.logits", logits)
        return types.SimpleNamespace(
            last_hidden_state=last_hidden_state,
            past_key_values=past_key_values,
            logits=logits,
        )


def _contract_error_type():
    module = importlib.import_module("pvlc_reference.transformers_oracle")
    return module.CaptureContractError


def _build_inputs(*, offset: float = 0.0):
    base = torch.arange(1, 1 + (_BATCH * _SEQ * _HIDDEN), dtype=torch.float32).view(_BATCH, _SEQ, _HIDDEN)
    input_a = (base / 10_000.0) + offset
    input_b = ((base + 100_000.0) / 10_000.0) + offset
    input_c = ((base + 200_000.0) / 10_000.0) + offset
    input_d = ((base + 300_000.0) / 10_000.0) + offset
    return input_a, input_b, input_c, input_d


def _owner_module(model: _FakeOuterModel):
    return sys.modules[model.__class__.__module__]


def _hook_count(model: torch.nn.Module) -> int:
    total = 0
    for module in model.modules():
        total += len(getattr(module, "_forward_hooks", {}))
        total += len(getattr(module, "_forward_pre_hooks", {}))
        total += len(getattr(module, "_backward_hooks", {}))
        total += len(getattr(module, "_backward_pre_hooks", {}))
    return total


def _assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def _assert_cache_layer_equal(actual_layer, expected_layer) -> None:
    _assert_tensor_equal(actual_layer.keys, expected_layer.keys)
    _assert_tensor_equal(actual_layer.values, expected_layer.values)


def _mutate_valid_layer0_cache(cache: DynamicCache) -> None:
    cache.layers[0].keys = cache.layers[0].keys + 0.25
    cache.layers[0].values = cache.layers[0].values - 0.5


def _set_active(
    model: _FakeOuterModel,
    tag: str,
    tensor: torch.Tensor,
    *,
    use_cache: bool,
    past_key_values: DynamicCache | None = None,
    attention_mask: torch.Tensor | None = None,
    cache_position: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
):
    model.active_tag = tag
    return model(
        tensor,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_ids=position_ids,
    )


def _discard_recorded_tag(model: _FakeOuterModel, tag: str) -> None:
    model.recorder.by_tag.pop(tag, None)
    model.recorder.source_by_tag.pop(tag, None)
    model.decoder_metadata_by_tag.pop(tag, None)
    model.decoder_metadata_source_by_tag.pop(tag, None)
    model.self_attn_metadata_by_tag.pop(tag, None)
    model.self_attn_metadata_source_by_tag.pop(tag, None)


def _run_m6_until_cached_prefill(
    model: _FakeOuterModel,
    *,
    scenario=_DEFAULT_M6_SCENARIO,
):
    model.position_axis_starts = scenario["axis_starts"]
    prompt, uncached_prompt, cached_prefill_decoy, later_prompt = _build_inputs(
        offset=scenario["input_offset"]
    )
    first_decode = uncached_prompt[:, :1].clone()
    later_decode = later_prompt[:, :1].clone()
    prompt_mask = torch.tensor([scenario["prompt_mask"]], dtype=torch.int64)
    prompt_cache_position = torch.arange(_SEQ, dtype=torch.int64)

    uncached_result = _set_active(
        model,
        "A",
        prompt,
        use_cache=False,
        attention_mask=prompt_mask,
        cache_position=prompt_cache_position,
    )
    assert uncached_result.past_key_values is None

    def _run_decoy(kind: str, tag: str, index: int) -> None:
        if kind == "uncached":
            result = _set_active(
                model,
                tag,
                first_decode + 50.0 + index,
                use_cache=False,
                attention_mask=torch.ones((_BATCH, 1), dtype=torch.int64),
                cache_position=torch.tensor([777 + index], dtype=torch.int64),
            )
            assert result.past_key_values is None
        elif kind == "manual":
            result = _set_active(
                model,
                tag,
                cached_prefill_decoy + index,
                use_cache=True,
                past_key_values=None,
                attention_mask=prompt_mask,
                cache_position=prompt_cache_position,
            )
            assert result.past_key_values is not None
        else:
            raise AssertionError(f"unknown decoy kind: {kind}")
        _discard_recorded_tag(model, tag)

    decoy_tags: list[str] = []
    for index, kind in enumerate(scenario["prefill_decoys"]):
        tag = f"B{index:02d}"
        _run_decoy(kind, tag, index)
        decoy_tags.append(tag)

    prefill_result = _set_active(
        model,
        "C",
        prompt.clone(),
        use_cache=True,
        past_key_values=None,
        attention_mask=prompt_mask,
        cache_position=prompt_cache_position,
    )
    assert prefill_result.past_key_values is not None
    prefill_cache_sources = tuple(
        (layer.keys, layer.values) for layer in prefill_result.past_key_values.layers
    )
    prefill_cache = _cache_snapshot(prefill_result.past_key_values)
    prompt_position_ids = _clone(
        model.self_attn_metadata_by_tag["C"]["position_ids"]
    )

    for index, kind in enumerate(scenario["post_prefill_decoys"]):
        tag = f"C2{index:02d}"
        _run_decoy(kind, tag, index + len(decoy_tags))
        decoy_tags.append(tag)

    return types.SimpleNamespace(
        scenario=scenario,
        prompt=prompt,
        prompt_mask=prompt_mask,
        uncached_result=uncached_result,
        prefill_result=prefill_result,
        decoy_tags=tuple(decoy_tags),
        prompt_position_ids=prompt_position_ids,
        prefill_cache_sources=prefill_cache_sources,
        prefill_cache=prefill_cache,
        first_decode=first_decode,
        later_decode=later_decode,
    )


def _run_m6_call_sequence(
    model: _FakeOuterModel,
    *,
    scenario=_DEFAULT_M6_SCENARIO,
    include_later_decode: bool = True,
):
    prefill = _run_m6_until_cached_prefill(model, scenario=scenario)
    requested_first_decode_metadata = {
        "attention_mask": torch.cat(
            (prefill.prompt_mask, torch.ones((_BATCH, 1), dtype=torch.int64)),
            dim=1,
        ),
        "cache_position": torch.tensor([_SEQ], dtype=torch.int64),
    }
    first_decode_result = _set_active(
        model,
        "D",
        prefill.first_decode,
        use_cache=True,
        past_key_values=prefill.prefill_result.past_key_values,
        **requested_first_decode_metadata,
    )
    assert first_decode_result.past_key_values is not None
    first_decode_metadata_sources = {
        "attention_mask": model.decoder_metadata_source_by_tag["D"]["attention_mask"],
        "cache_position": model.self_attn_metadata_source_by_tag["D"]["cache_position"],
        "position_ids": model.self_attn_metadata_source_by_tag["D"]["position_ids"],
    }
    first_decode_metadata = {
        name: _clone(tensor) for name, tensor in first_decode_metadata_sources.items()
    }
    first_decode_cache_sources = tuple(
        (layer.keys, layer.values) for layer in first_decode_result.past_key_values.layers
    )
    first_decode_cache = _cache_snapshot(first_decode_result.past_key_values)

    requested_later_decode_metadata = None
    later_decode_result = None
    if include_later_decode:
        requested_later_decode_metadata = {
            "attention_mask": torch.cat(
                (prefill.prompt_mask, torch.ones((_BATCH, 2), dtype=torch.int64)),
                dim=1,
            ),
            "cache_position": torch.tensor([_SEQ + 1], dtype=torch.int64),
        }
        later_decode_result = _set_active(
            model,
            "E",
            prefill.later_decode,
            use_cache=True,
            past_key_values=first_decode_result.past_key_values,
            **requested_later_decode_metadata,
        )

    return types.SimpleNamespace(
        scenario=scenario,
        prompt=prefill.prompt,
        prompt_mask=prefill.prompt_mask,
        uncached_result=prefill.uncached_result,
        prefill_result=prefill.prefill_result,
        decoy_tags=prefill.decoy_tags,
        prompt_position_ids=prefill.prompt_position_ids,
        prefill_cache_sources=prefill.prefill_cache_sources,
        prefill_cache=prefill.prefill_cache,
        first_decode=prefill.first_decode,
        first_decode_result=first_decode_result,
        first_decode_cache_sources=first_decode_cache_sources,
        first_decode_cache=first_decode_cache,
        requested_first_decode_metadata=requested_first_decode_metadata,
        first_decode_metadata_sources=first_decode_metadata_sources,
        first_decode_metadata=first_decode_metadata,
        later_decode=prefill.later_decode,
        later_decode_result=later_decode_result,
        requested_later_decode_metadata=requested_later_decode_metadata,
        first_decode_ordinal=model.decoder_forward_ordinal_by_tag["D"],
    )


def _m6_expected_tensors(model: _FakeOuterModel, sequence) -> dict[str, torch.Tensor]:
    expected = {
        semantic_id: _clone(model.recorder.by_tag["A"][semantic_id])
        for semantic_id in _EXPECTED_CAPTURE_KEYS
        if ".kv." not in semantic_id
    }
    for layer_index, (keys, values) in enumerate(sequence.prefill_cache):
        expected[f"decoder.layer.{layer_index:02d}.kv.key"] = _clone(keys)
        expected[f"decoder.layer.{layer_index:02d}.kv.value"] = _clone(values)

    for semantic_id, tensor in model.recorder.by_tag["D"].items():
        if semantic_id == "decoder.logits":
            decode_id = "decoder.decode.00.logits"
        else:
            decode_id = f"decoder.decode.00.{semantic_id.removeprefix('decoder.')}"
        expected[decode_id] = _clone(tensor)
    for layer_index, (keys, values) in enumerate(sequence.first_decode_cache):
        expected[f"decoder.decode.00.layer.{layer_index:02d}.kv.key"] = _clone(keys)
        expected[f"decoder.decode.00.layer.{layer_index:02d}.kv.value"] = _clone(values)
    for name, tensor in sequence.first_decode_metadata.items():
        expected[f"decoder.decode.00.{name}"] = _clone(tensor)

    assert set(expected) == _EXPECTED_M6_CAPTURE_KEYS
    return expected


def _assert_exact_cache_append(
    prefill_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    decode_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    *,
    appended_layer0_key: torch.Tensor,
    appended_layer0_value: torch.Tensor,
) -> None:
    assert len(prefill_cache) == _FAKE_LAYER_COUNT
    assert len(decode_cache) == _FAKE_LAYER_COUNT
    for (prefill_key, prefill_value), (decode_key, decode_value) in zip(
        prefill_cache,
        decode_cache,
        strict=True,
    ):
        assert decode_key.shape[2] == prefill_key.shape[2] + 1
        assert decode_value.shape[2] == prefill_value.shape[2] + 1
        _assert_tensor_equal(decode_key[:, :, :-1, :], prefill_key)
        _assert_tensor_equal(decode_value[:, :, :-1, :], prefill_value)
    _assert_tensor_equal(decode_cache[0][0][:, :, -1:, :], appended_layer0_key)
    _assert_tensor_equal(decode_cache[0][1][:, :, -1:, :], appended_layer0_value)


def _valid_first_decode_metadata() -> dict[str, torch.Tensor | None]:
    return {
        "attention_mask": torch.ones((_BATCH, _SEQ + 1), dtype=torch.int64),
        "cache_position": torch.tensor([_SEQ], dtype=torch.int64),
        "position_ids": None,
    }


_INVALID_DECODE_MASK_CONTENT_CASES = (
    "wrong_prefix_bit",
    "appended_zero",
    "integer_two",
    "integer_negative_one",
)


def _invalid_decode_attention_mask(
    prompt_mask: torch.Tensor,
    case: str,
) -> torch.Tensor:
    mask = torch.cat(
        (prompt_mask.clone(), torch.ones((_BATCH, 1), dtype=torch.int64)),
        dim=1,
    )
    if case == "wrong_prefix_bit":
        mask[0, 0] = 1 - mask[0, 0]
    elif case == "appended_zero":
        mask[0, -1] = 0
    elif case == "integer_two":
        mask[0, -1] = 2
    elif case == "integer_negative_one":
        mask[0, -1] = -1
    else:
        raise AssertionError(f"unknown attention-mask content case: {case}")
    return mask


def _invoke_first_decode(
    model: _FakeOuterModel,
    prefill,
    *,
    past_key_values,
    metadata: dict[str, torch.Tensor | None] | None = None,
    query: torch.Tensor | None = None,
):
    if metadata is None:
        metadata = _valid_first_decode_metadata()
    return _set_active(
        model,
        "D",
        prefill.first_decode if query is None else query,
        use_cache=True,
        past_key_values=past_key_values,
        attention_mask=metadata.get("attention_mask"),
        cache_position=metadata.get("cache_position"),
        position_ids=metadata.get("position_ids"),
    )


_DECODE_CACHE_TENSOR_DEFECTS = (
    "rank",
    "batch",
    "heads",
    "sequence",
    "head_dim",
    "dtype",
    "device",
    "nan",
    "positive_infinity",
    "negative_infinity",
)
_DECODE_CACHE_TENSOR_CASES = (
    tuple(
        (
            f"layer{layer_index}_{side}_{defect}",
            layer_index,
            side,
            defect,
        )
        for layer_index in (0, _FAKE_LAYER_COUNT - 1)
        for side in ("keys", "values")
        for defect in _DECODE_CACHE_TENSOR_DEFECTS
    )
    + tuple(
        (
            f"layer{_FAKE_LAYER_COUNT - 1}_joint_{defect}",
            _FAKE_LAYER_COUNT - 1,
            "joint",
            defect,
        )
        for defect in _DECODE_CACHE_TENSOR_DEFECTS
    )
)
_CACHE_LAYER_SIDE_CASES = tuple(
    (f"layer{layer_index}_{side}", layer_index, side)
    for layer_index in (0, _FAKE_LAYER_COUNT - 1)
    for side in ("keys", "values")
)


def _apply_decode_cache_tensor_defect(
    cache: DynamicCache,
    *,
    layer_index: int,
    side: str,
    defect: str,
    stage: str,
) -> None:
    if side == "joint":
        for member in ("keys", "values"):
            _apply_decode_cache_tensor_defect(
                cache,
                layer_index=layer_index,
                side=member,
                defect=defect,
                stage=stage,
            )
        keys = cache.layers[layer_index].keys
        values = cache.layers[layer_index].values
        assert keys.shape == values.shape
        assert keys.dtype == values.dtype
        assert keys.device == values.device
        if defect in ("nan", "positive_infinity", "negative_infinity"):
            assert not torch.isfinite(keys).all()
            assert not torch.isfinite(values).all()
        return
    tensor = getattr(cache.layers[layer_index], side)
    if defect == "rank":
        replacement = tensor[0]
    elif defect == "batch":
        replacement = tensor.repeat(2, 1, 1, 1)
    elif defect == "heads":
        replacement = tensor.repeat(1, 2, 1, 1)
    elif defect == "sequence":
        replacement = (
            torch.cat((tensor, tensor[:, :, :1, :]), dim=2)
            if stage == "input"
            else tensor[:, :, :-1, :]
        )
    elif defect == "head_dim":
        replacement = tensor[..., :-1]
    elif defect == "dtype":
        replacement = tensor.to(torch.float64)
    elif defect == "device":
        replacement = torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")
    elif defect in ("nan", "positive_infinity", "negative_infinity"):
        replacement = tensor.clone()
        value = {
            "nan": torch.nan,
            "positive_infinity": torch.inf,
            "negative_infinity": -torch.inf,
        }[defect]
        sequence_index = 0 if stage == "input" else -1
        replacement[0, 0, sequence_index, 0] = value
    else:
        raise AssertionError(f"unknown cache defect: {defect}")
    setattr(cache.layers[layer_index], side, replacement)


def _add_finite_cache_drift(
    cache: DynamicCache,
    *,
    layer_index: int,
    side: str,
    sequence_index: int,
) -> None:
    tensor = getattr(cache.layers[layer_index], side)
    replacement = tensor.clone()
    with torch.no_grad():
        replacement[0, 0, sequence_index, 0].add_(0.25)
    assert replacement.shape == tensor.shape
    assert replacement.dtype == tensor.dtype
    assert replacement.device == tensor.device
    assert torch.isfinite(replacement).all()
    setattr(cache.layers[layer_index], side, replacement)


def _make_decode_cache_structure_case(cache, case: str):
    if case == "missing":
        return None
    if case == "empty":
        cache.layers.clear()
        return cache
    if case == "partial":
        cache.layers.pop()
        return cache
    if case == "extra":
        cache.update(
            cache.layers[-1].keys.clone(),
            cache.layers[-1].values.clone(),
            _FAKE_LAYER_COUNT,
        )
        return cache
    if case == "lookalike":
        return types.SimpleNamespace(layers=cache.layers)
    raise AssertionError(f"unknown cache structure case: {case}")


def _make_post_decode_cache_mutators():
    def _missing(_cache):
        return _DROP_CACHE_OUTPUT

    def _lookalike(cache):
        return types.SimpleNamespace(layers=cache.layers)

    def _incomplete_layers(cache):
        cache.layers.pop()

    def _not_appended(cache):
        for layer in cache.layers:
            layer.keys = layer.keys[:, :, :-1, :]
            layer.values = layer.values[:, :, :-1, :]

    def _appended_twice(cache):
        for layer in cache.layers:
            layer.keys = torch.cat((layer.keys, layer.keys[:, :, -1:, :]), dim=2)
            layer.values = torch.cat((layer.values, layer.values[:, :, -1:, :]), dim=2)

    def _prefix_mutated(cache):
        for layer in cache.layers:
            layer.keys = layer.keys.clone()
            layer.values = layer.values.clone()
            layer.keys[:, :, 0, 0].add_(0.25)
            layer.values[:, :, 0, 0].sub_(0.5)

    def _wrong_shape(cache):
        cache.layers[0].keys = cache.layers[0].keys[..., :-1]
        cache.layers[0].values = cache.layers[0].values[..., :-1]

    def _key_value_shape_disagreement(cache):
        cache.layers[0].values = cache.layers[0].values[:, :, :-1, :]

    def _wrong_dtype(cache):
        cache.layers[0].keys = cache.layers[0].keys.to(torch.float64)
        cache.layers[0].values = cache.layers[0].values.to(torch.float64)

    def _wrong_device(cache):
        shape = cache.layers[0].keys.shape
        dtype = cache.layers[0].keys.dtype
        cache.layers[0].keys = torch.empty(shape, dtype=dtype, device="meta")
        cache.layers[0].values = torch.empty(shape, dtype=dtype, device="meta")

    return [
        ("missing", _missing),
        ("lookalike", _lookalike),
        ("incomplete_layers", _incomplete_layers),
        ("not_appended", _not_appended),
        ("appended_twice", _appended_twice),
        ("prefix_mutated", _prefix_mutated),
        ("wrong_shape", _wrong_shape),
        ("key_value_shape_disagreement", _key_value_shape_disagreement),
        ("wrong_dtype", _wrong_dtype),
        ("wrong_device", _wrong_device),
    ]


def _make_decode_metadata_mutators():
    def _missing(name: str):
        return lambda metadata: metadata.__setitem__(name, None)

    def _replace(name: str, value: torch.Tensor):
        return lambda metadata: metadata.__setitem__(name, value)

    return [
        ("missing_attention_mask", "decoder", _missing("attention_mask")),
        (
            "short_attention_mask",
            "decoder",
            _replace("attention_mask", torch.ones((_BATCH, _SEQ), dtype=torch.int64)),
        ),
        (
            "long_attention_mask",
            "decoder",
            _replace("attention_mask", torch.ones((_BATCH, _SEQ + 2), dtype=torch.int64)),
        ),
        (
            "rank1_attention_mask",
            "decoder",
            _replace("attention_mask", torch.ones((_SEQ + 1,), dtype=torch.int64)),
        ),
        (
            "non_integer_attention_mask",
            "decoder",
            _replace(
                "attention_mask",
                torch.ones((_BATCH, _SEQ + 1), dtype=torch.float32),
            ),
        ),
        (
            "meta_attention_mask",
            "decoder",
            _replace(
                "attention_mask",
                torch.empty((_BATCH, _SEQ + 1), dtype=torch.int64, device="meta"),
            ),
        ),
        ("missing_cache_position", "self_attn", _missing("cache_position")),
        (
            "wrong_cache_position",
            "self_attn",
            _replace("cache_position", torch.tensor([_SEQ + 1], dtype=torch.int64)),
        ),
        (
            "rank2_cache_position",
            "self_attn",
            _replace("cache_position", torch.tensor([[_SEQ]], dtype=torch.int64)),
        ),
        (
            "non_integer_cache_position",
            "self_attn",
            _replace("cache_position", torch.tensor([float(_SEQ)], dtype=torch.float32)),
        ),
        (
            "meta_cache_position",
            "self_attn",
            _replace("cache_position", torch.empty((1,), dtype=torch.int64, device="meta")),
        ),
        ("missing_position_ids", "self_attn", _missing("position_ids")),
        (
            "one_wrong_position_axis",
            "self_attn",
            lambda metadata: metadata.__setitem__(
                "position_ids",
                metadata["position_ids"].clone().index_add(
                    0,
                    torch.tensor([1], dtype=torch.int64),
                    torch.ones((1, _BATCH, 1), dtype=torch.int64),
                ),
            ),
        ),
        (
            "wrong_position_ids_shape",
            "self_attn",
            _replace("position_ids", torch.full((1, _BATCH, 1), 42, dtype=torch.int64)),
        ),
        (
            "non_integer_position_ids",
            "self_attn",
            _replace("position_ids", torch.full((3, _BATCH, 1), 42.0, dtype=torch.float32)),
        ),
        (
            "meta_position_ids",
            "self_attn",
            _replace(
                "position_ids",
                torch.empty((3, _BATCH, 1), dtype=torch.int64, device="meta"),
            ),
        ),
    ]


def _register_trace(trace, model: _FakeOuterModel, *, expected_baseline_hooks: int = 0):
    owner = _owner_module(model)
    original_fn = owner.apply_multimodal_rotary_pos_emb
    baseline_hooks = _hook_count(model)
    assert baseline_hooks == expected_baseline_hooks
    handles = trace.register(model)
    active_hooks = _hook_count(model)
    assert handles
    assert owner.apply_multimodal_rotary_pos_emb is not original_fn
    assert active_hooks > baseline_hooks
    return owner, original_fn, baseline_hooks, active_hooks


def _assert_closed(trace, model: _FakeOuterModel, owner, original_fn, baseline_hooks: int):
    trace.close()
    assert owner.apply_multimodal_rotary_pos_emb is original_fn
    assert _hook_count(model) == baseline_hooks
    trace.close()
    assert owner.apply_multimodal_rotary_pos_emb is original_fn
    assert _hook_count(model) == baseline_hooks


def _tensor_storage_identity(tensor: torch.Tensor) -> tuple[int, str]:
    return (tensor.untyped_storage().data_ptr(), str(tensor.device))


def _mutate_dense_tensor_observably(tensor: torch.Tensor) -> None:
    assert tensor.layout == torch.strided
    assert tensor.numel() > 0
    if tensor.dtype != torch.bool and not (
        tensor.is_floating_point() or tensor.is_complex()
    ):
        torch.iinfo(tensor.dtype)

    before = _clone(tensor)
    tensor.zero_()
    if torch.equal(tensor, before):
        tensor.fill_(1)
    assert not torch.equal(tensor, before)


def _assert_captured_tensors_detached_and_storage_isolated(
    captured: dict[str, torch.Tensor],
    source_tensors: list[torch.Tensor],
    input_a: torch.Tensor,
    cache_tensors: list[torch.Tensor],
) -> None:
    storage_identities = {
        name: _tensor_storage_identity(tensor) for name, tensor in captured.items()
    }
    assert len(set(storage_identities.values())) == len(storage_identities)
    source_storage_identities = {
        _tensor_storage_identity(tensor)
        for tensor in (*source_tensors, input_a, *cache_tensors)
    }
    assert set(storage_identities.values()).isdisjoint(source_storage_identities)
    for tensor in captured.values():
        assert not tensor.requires_grad
        assert tensor.grad_fn is None

    snapshots = {name: _clone(tensor) for name, tensor in captured.items()}

    def _mutate_and_assert(original: torch.Tensor) -> None:
        with torch.no_grad():
            before = _clone(original)
            _mutate_dense_tensor_observably(original)
            for name, expected in snapshots.items():
                _assert_tensor_equal(captured[name], expected)
            original.copy_(before)
            _assert_tensor_equal(original, before)

    for tensor in source_tensors:
        _mutate_and_assert(tensor)
    _mutate_and_assert(input_a)
    for tensor in cache_tensors:
        _mutate_and_assert(tensor)

    for name, tensor in captured.items():
        with torch.no_grad():
            before = _clone(tensor)
            _mutate_dense_tensor_observably(tensor)
            for other_name, expected in snapshots.items():
                if other_name == name:
                    continue
                _assert_tensor_equal(captured[other_name], expected)
            tensor.copy_(before)
            _assert_tensor_equal(tensor, before)


_OBSERVABLE_MUTATION_DTYPES = (
    torch.bool,
    *(getattr(torch, name) for name in ("int8", "int16", "int32", "int64")),
    *(
        getattr(torch, name)
        for name in ("uint8", "uint16", "uint32", "uint64")
        if hasattr(torch, name)
    ),
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    *(
        getattr(torch, name)
        for name in dir(torch)
        if name.startswith("float8_") and isinstance(getattr(torch, name), torch.dtype)
    ),
    torch.complex64,
    torch.complex128,
)


@pytest.mark.parametrize("fill_value", (0, 1), ids=("zero", "nonzero"))
@pytest.mark.parametrize(
    "dtype",
    _OBSERVABLE_MUTATION_DTYPES,
    ids=lambda dtype: str(dtype).removeprefix("torch."),
)
def test_mutate_dense_tensor_observably_supports_numeric_dtype_family(
    dtype: torch.dtype,
    fill_value: int,
) -> None:
    tensor = torch.full((2, 3), fill_value, dtype=dtype)
    before = _clone(tensor)

    with torch.no_grad():
        _mutate_dense_tensor_observably(tensor)

    assert not torch.equal(tensor, before)


def test_storage_isolation_helper_rejects_disjoint_sibling_view_alias() -> None:
    backing = torch.arange(8, dtype=torch.float32)
    source_view = backing[:4]
    captured_sibling_view = backing[4:]
    input_a = torch.arange(4, dtype=torch.float32) + 20
    source_before = _clone(source_view)
    input_before = _clone(input_a)

    assert source_view.data_ptr() != captured_sibling_view.data_ptr()
    assert _tensor_storage_identity(source_view) == _tensor_storage_identity(
        captured_sibling_view
    )
    with pytest.raises(AssertionError):
        _assert_captured_tensors_detached_and_storage_isolated(
            {"capture": captured_sibling_view},
            [source_view],
            input_a,
            [],
        )

    disjoint_capture = _clone(captured_sibling_view)
    capture_before = _clone(disjoint_capture)
    _assert_captured_tensors_detached_and_storage_isolated(
        {"capture": disjoint_capture},
        [source_view],
        input_a,
        [],
    )
    _assert_tensor_equal(disjoint_capture, capture_before)
    _assert_tensor_equal(source_view, source_before)
    _assert_tensor_equal(input_a, input_before)


def _assert_raw_mrope_axes(
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    sequence: int = _SEQ,
) -> None:
    assert cos.shape == (3, _BATCH, sequence, _HEAD_DIM)
    assert sin.shape == (3, _BATCH, sequence, _HEAD_DIM)
    assert not torch.equal(cos[0], cos[1])
    assert not torch.equal(cos[1], cos[2])
    assert not torch.equal(sin[0], sin[1])
    assert not torch.equal(sin[1], sin[2])

    selected_cos = torch.cat(
        [chunk[index % 3] for index, chunk in enumerate(cos.split(_MROPE_SECTIONS, dim=-1))],
        dim=-1,
    )
    selected_sin = torch.cat(
        [chunk[index % 3] for index, chunk in enumerate(sin.split(_MROPE_SECTIONS, dim=-1))],
        dim=-1,
    )

    assert selected_cos.shape == (_BATCH, sequence, _HEAD_DIM)
    assert selected_sin.shape == (_BATCH, sequence, _HEAD_DIM)
    _assert_tensor_equal(selected_cos[..., 0:16], cos[0, ..., 0:16])
    _assert_tensor_equal(selected_cos[..., 16:40], cos[1, ..., 16:40])
    _assert_tensor_equal(selected_cos[..., 40:64], cos[2, ..., 40:64])
    _assert_tensor_equal(selected_cos[..., 64:80], cos[0, ..., 64:80])
    _assert_tensor_equal(selected_cos[..., 80:104], cos[1, ..., 80:104])
    _assert_tensor_equal(selected_cos[..., 104:128], cos[2, ..., 104:128])
    _assert_tensor_equal(selected_sin[..., 0:16], sin[0, ..., 0:16])
    _assert_tensor_equal(selected_sin[..., 16:40], sin[1, ..., 16:40])
    _assert_tensor_equal(selected_sin[..., 40:64], sin[2, ..., 40:64])
    _assert_tensor_equal(selected_sin[..., 64:80], sin[0, ..., 64:80])
    _assert_tensor_equal(selected_sin[..., 80:104], sin[1, ..., 80:104])
    _assert_tensor_equal(selected_sin[..., 104:128], sin[2, ..., 104:128])


def _assert_contract_error(error: BaseException) -> None:
    message = str(error)
    assert message
    assert getattr(error, "code")


def _expect_contract_error_from_operation(trace, contract_error, operation) -> None:
    try:
        operation()
    except contract_error as error:
        _assert_contract_error(error)
    else:
        with pytest.raises(contract_error) as exc_info:
            trace.finish()
        _assert_contract_error(exc_info.value)


def _make_cache_layer_mutators():
    def _mutate_cache_entry(cache, which: str, mutator):
        setattr(cache.layers[0], which, mutator(getattr(cache.layers[0], which)))

    def _mutate_joint(cache, key_mutator, value_mutator):
        cache.layers[0].keys = key_mutator(cache.layers[0].keys)
        cache.layers[0].values = value_mutator(cache.layers[0].values)

    def _empty_layers(cache):
        cache.layers.clear()

    def _lookalike_cache(cache):
        return types.SimpleNamespace(layers=cache.layers)

    def _delete_layer0_keys(cache):
        delattr(cache.layers[0], "keys")

    def _delete_layer0_values(cache):
        delattr(cache.layers[0], "values")

    def _layer0_non_tensor_keys(cache):
        _mutate_cache_entry(cache, "keys", lambda _: "not-a-tensor")

    def _layer0_non_tensor_values(cache):
        _mutate_cache_entry(cache, "values", lambda _: {"bad": True})

    def _layer0_wrong_key_batch(cache):
        _mutate_cache_entry(cache, "keys", lambda tensor: tensor.repeat(2, 1, 1, 1))

    def _layer0_wrong_value_batch(cache):
        _mutate_cache_entry(cache, "values", lambda tensor: tensor.repeat(2, 1, 1, 1))

    def _layer0_wrong_key_kv_heads(cache):
        _mutate_cache_entry(cache, "keys", lambda tensor: tensor.repeat(1, 2, 1, 1))

    def _layer0_wrong_value_kv_heads(cache):
        _mutate_cache_entry(cache, "values", lambda tensor: tensor.repeat(1, 2, 1, 1))

    def _layer0_wrong_key_seq(cache):
        _mutate_cache_entry(
            cache,
            "keys",
            lambda tensor: torch.cat((tensor, tensor[:, :, :1, :]), dim=2),
        )

    def _layer0_wrong_value_seq(cache):
        _mutate_cache_entry(
            cache,
            "values",
            lambda tensor: torch.cat((tensor, tensor[:, :, :1, :]), dim=2),
        )

    def _layer0_wrong_key_head_dim(cache):
        _mutate_cache_entry(cache, "keys", lambda tensor: tensor[..., :-1])

    def _layer0_wrong_value_head_dim(cache):
        _mutate_cache_entry(cache, "values", lambda tensor: tensor[..., :-1])

    def _layer0_wrong_key_rank3(cache):
        _mutate_cache_entry(cache, "keys", lambda tensor: tensor[0])

    def _layer0_wrong_value_rank3(cache):
        _mutate_cache_entry(cache, "values", lambda tensor: tensor[0])

    def _layer0_wrong_key_dtype(cache):
        _mutate_cache_entry(cache, "keys", lambda tensor: tensor.to(torch.float16))

    def _layer0_wrong_value_dtype(cache):
        _mutate_cache_entry(cache, "values", lambda tensor: tensor.to(torch.float16))

    def _layer0_wrong_key_device(cache):
        _mutate_cache_entry(
            cache,
            "keys",
            lambda tensor: torch.empty(tensor.shape, dtype=tensor.dtype, device="meta"),
        )

    def _layer0_wrong_value_device(cache):
        _mutate_cache_entry(
            cache,
            "values",
            lambda tensor: torch.empty(tensor.shape, dtype=tensor.dtype, device="meta"),
        )

    def _layer0_kv_disagreement(cache):
        cache.layers[0].values = torch.cat((cache.layers[0].values, cache.layers[0].values[:, :, :1, :]), dim=2)

    def _layer0_joint_wrong_rank(cache):
        _mutate_joint(cache, lambda tensor: tensor[0], lambda tensor: tensor[0])

    def _layer0_joint_wrong_batch(cache):
        _mutate_joint(
            cache,
            lambda tensor: tensor.repeat(2, 1, 1, 1),
            lambda tensor: tensor.repeat(2, 1, 1, 1),
        )

    def _layer0_joint_wrong_kv_heads(cache):
        _mutate_joint(
            cache,
            lambda tensor: tensor.repeat(1, 2, 1, 1),
            lambda tensor: tensor.repeat(1, 2, 1, 1),
        )

    def _layer0_joint_wrong_seq(cache):
        _mutate_joint(
            cache,
            lambda tensor: torch.cat((tensor, tensor[:, :, :1, :]), dim=2),
            lambda tensor: torch.cat((tensor, tensor[:, :, :1, :]), dim=2),
        )

    def _layer0_joint_wrong_head_dim(cache):
        _mutate_joint(cache, lambda tensor: tensor[..., :-1], lambda tensor: tensor[..., :-1])

    def _layer0_joint_wrong_dtype(cache):
        _mutate_joint(
            cache,
            lambda tensor: tensor.to(torch.float16),
            lambda tensor: tensor.to(torch.float16),
        )

    def _layer0_joint_wrong_device(cache):
        _mutate_joint(
            cache,
            lambda tensor: torch.empty(tensor.shape, dtype=tensor.dtype, device="meta"),
            lambda tensor: torch.empty(tensor.shape, dtype=tensor.dtype, device="meta"),
        )

    return [
        ("empty_layers", _empty_layers),
        ("lookalike_cache", _lookalike_cache),
        ("delete_layer0_keys", _delete_layer0_keys),
        ("delete_layer0_values", _delete_layer0_values),
        ("layer0_non_tensor_keys", _layer0_non_tensor_keys),
        ("layer0_non_tensor_values", _layer0_non_tensor_values),
        ("layer0_wrong_key_batch", _layer0_wrong_key_batch),
        ("layer0_wrong_value_batch", _layer0_wrong_value_batch),
        ("layer0_wrong_key_kv_heads", _layer0_wrong_key_kv_heads),
        ("layer0_wrong_value_kv_heads", _layer0_wrong_value_kv_heads),
        ("layer0_wrong_key_seq", _layer0_wrong_key_seq),
        ("layer0_wrong_value_seq", _layer0_wrong_value_seq),
        ("layer0_wrong_key_head_dim", _layer0_wrong_key_head_dim),
        ("layer0_wrong_value_head_dim", _layer0_wrong_value_head_dim),
        ("layer0_wrong_key_rank3", _layer0_wrong_key_rank3),
        ("layer0_wrong_value_rank3", _layer0_wrong_value_rank3),
        ("layer0_wrong_key_dtype", _layer0_wrong_key_dtype),
        ("layer0_wrong_value_dtype", _layer0_wrong_value_dtype),
        ("layer0_wrong_key_device", _layer0_wrong_key_device),
        ("layer0_wrong_value_device", _layer0_wrong_value_device),
        ("layer0_kv_disagreement", _layer0_kv_disagreement),
        ("layer0_joint_wrong_rank", _layer0_joint_wrong_rank),
        ("layer0_joint_wrong_batch", _layer0_joint_wrong_batch),
        ("layer0_joint_wrong_kv_heads", _layer0_joint_wrong_kv_heads),
        ("layer0_joint_wrong_seq", _layer0_joint_wrong_seq),
        ("layer0_joint_wrong_head_dim", _layer0_joint_wrong_head_dim),
        ("layer0_joint_wrong_dtype", _layer0_joint_wrong_dtype),
        ("layer0_joint_wrong_device", _layer0_joint_wrong_device),
    ]


def test_decoder_prefill_trace_capture_happy_path_uses_independent_baseline_and_exact_snapshots(
    monkeypatch: pytest.MonkeyPatch,
):
    seed = 7
    baseline_model = _FakeOuterModel(seed=seed)
    baseline_a, baseline_b, baseline_c, baseline_d = _build_inputs()
    baseline_model.cache_mutator = _mutate_valid_layer0_cache

    baseline_result_a = _set_active(baseline_model, "A", baseline_a, use_cache=False)
    baseline_result_b = _set_active(baseline_model, "B", baseline_b, use_cache=False)
    baseline_result_c = _set_active(baseline_model, "C", baseline_c, use_cache=True)
    baseline_result_d = _set_active(baseline_model, "D", baseline_d, use_cache=True)
    assert baseline_result_a.past_key_values is None
    assert baseline_result_b.past_key_values is None
    assert baseline_result_c.past_key_values is not None
    assert baseline_result_d.past_key_values is not None
    assert len(baseline_result_c.past_key_values.layers) == 2
    assert len(baseline_result_d.past_key_values.layers) == 2

    baseline_expected = {
        "decoder.rope.cos": baseline_model.recorder.by_tag["A"]["decoder.rope.cos"],
        "decoder.rope.sin": baseline_model.recorder.by_tag["A"]["decoder.rope.sin"],
        "decoder.layer.00.input": baseline_model.recorder.by_tag["A"]["decoder.layer.00.input"],
        "decoder.layer.00.norm1": baseline_model.recorder.by_tag["A"]["decoder.layer.00.norm1"],
        "decoder.layer.00.q": baseline_model.recorder.by_tag["A"]["decoder.layer.00.q"],
        "decoder.layer.00.k": baseline_model.recorder.by_tag["A"]["decoder.layer.00.k"],
        "decoder.layer.00.v": baseline_model.recorder.by_tag["A"]["decoder.layer.00.v"],
        "decoder.layer.00.mrope.q": baseline_model.recorder.by_tag["A"]["decoder.layer.00.mrope.q"],
        "decoder.layer.00.mrope.k": baseline_model.recorder.by_tag["A"]["decoder.layer.00.mrope.k"],
        "decoder.layer.00.kv.key": _clone(baseline_result_c.past_key_values.layers[0].keys),
        "decoder.layer.00.kv.value": _clone(baseline_result_c.past_key_values.layers[0].values),
        "decoder.layer.00.attention.context": baseline_model.recorder.by_tag["A"]["decoder.layer.00.attention.context"],
        "decoder.layer.00.attention.output": baseline_model.recorder.by_tag["A"]["decoder.layer.00.attention.output"],
        "decoder.layer.00.attention.residual": baseline_model.recorder.by_tag["A"]["decoder.layer.00.attention.residual"],
        "decoder.layer.00.norm2": baseline_model.recorder.by_tag["A"]["decoder.layer.00.norm2"],
        "decoder.layer.00.mlp.gate": baseline_model.recorder.by_tag["A"]["decoder.layer.00.mlp.gate"],
        "decoder.layer.00.mlp.up": baseline_model.recorder.by_tag["A"]["decoder.layer.00.mlp.up"],
        "decoder.layer.00.mlp.activation": baseline_model.recorder.by_tag["A"]["decoder.layer.00.mlp.activation"],
        "decoder.layer.00.mlp.down": baseline_model.recorder.by_tag["A"]["decoder.layer.00.mlp.down"],
        "decoder.final_norm": baseline_model.recorder.by_tag["A"]["decoder.final_norm"],
    }
    for layer_index, _ in enumerate(baseline_model.model.layers):
        baseline_expected[f"decoder.layer.{layer_index:02d}.output"] = baseline_model.recorder.by_tag["A"][
            f"decoder.layer.{layer_index:02d}.output"
        ]
    baseline_unmutated_c_key = baseline_model.recorder.by_tag["C"]["decoder.layer.00.mrope.k"]
    baseline_unmutated_c_value = _reshape_kv(baseline_model.recorder.by_tag["C"]["decoder.layer.00.v"])
    _assert_raw_mrope_axes(
        baseline_expected["decoder.rope.cos"],
        baseline_expected["decoder.rope.sin"],
    )

    instrumented_model = _FakeOuterModel(seed=seed)
    instrumented_model.cache_mutator = _mutate_valid_layer0_cache
    input_a, input_b, input_c, input_d = _build_inputs()
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, instrumented_model)

    try:
        instrumented_result_a = _set_active(instrumented_model, "A", input_a, use_cache=False)
        instrumented_result_b = _set_active(instrumented_model, "B", input_b, use_cache=False)
        instrumented_result_c = _set_active(instrumented_model, "C", input_c, use_cache=True)
        decoder = instrumented_model.model
        layer0 = decoder.layers[0]
        preserved_input_a = _clone(input_a)
        preserved_decoder_state = {
            name: _clone(tensor) for name, tensor in decoder.state_dict().items()
        }

        def _raise_if_finish_recomputes(*args, **kwargs):
            raise AssertionError("finish must not recompute decoder checkpoints")

        try:
            with monkeypatch.context() as patch_context:
                input_a.add_(12_345.0)
                for parameter in decoder.parameters():
                    parameter.data.add_(7.0)
                for buffer in decoder.buffers():
                    buffer.data.mul_(3.0)

                for module in (
                    decoder,
                    decoder.rotary_emb,
                    decoder.norm,
                    *decoder.layers,
                    layer0.input_layernorm,
                    layer0.self_attn,
                    layer0.self_attn.q_proj,
                    layer0.self_attn.k_proj,
                    layer0.self_attn.v_proj,
                    layer0.self_attn.o_proj,
                    layer0.post_attention_layernorm,
                    layer0.mlp.gate_proj,
                    layer0.mlp.up_proj,
                    layer0.mlp.act_fn,
                    layer0.mlp.down_proj,
                ):
                    patch_context.setattr(module, "forward", _raise_if_finish_recomputes)

                captured = trace.finish()
        finally:
            input_a.copy_(preserved_input_a)
            decoder.load_state_dict(preserved_decoder_state)

        instrumented_result_d = _set_active(instrumented_model, "D", input_d, use_cache=True)
    finally:
        _assert_closed(trace, instrumented_model, owner, original_fn, baseline_hooks)

    assert instrumented_result_a.past_key_values is None
    assert instrumented_result_b.past_key_values is None
    assert instrumented_result_c.past_key_values is not None
    assert instrumented_result_d.past_key_values is not None
    assert len(instrumented_result_c.past_key_values.layers) == 2

    _assert_tensor_equal(instrumented_result_a.last_hidden_state, baseline_result_a.last_hidden_state)
    _assert_tensor_equal(instrumented_result_b.last_hidden_state, baseline_result_b.last_hidden_state)
    _assert_tensor_equal(instrumented_result_c.last_hidden_state, baseline_result_c.last_hidden_state)
    _assert_tensor_equal(instrumented_result_d.last_hidden_state, baseline_result_d.last_hidden_state)
    _assert_cache_layer_equal(instrumented_result_c.past_key_values.layers[0], baseline_result_c.past_key_values.layers[0])
    _assert_cache_layer_equal(instrumented_result_c.past_key_values.layers[1], baseline_result_c.past_key_values.layers[1])
    _assert_cache_layer_equal(instrumented_result_d.past_key_values.layers[0], baseline_result_d.past_key_values.layers[0])
    _assert_cache_layer_equal(instrumented_result_d.past_key_values.layers[1], baseline_result_d.past_key_values.layers[1])

    assert set(captured) == _EXPECTED_CAPTURE_KEYS
    for key, expected_tensor in baseline_expected.items():
        _assert_tensor_equal(captured[key], expected_tensor)
    _assert_raw_mrope_axes(captured["decoder.rope.cos"], captured["decoder.rope.sin"])

    assert captured["decoder.layer.00.attention.context"].shape == (_BATCH, _SEQ, _Q_FLAT)
    assert not torch.equal(captured["decoder.layer.00.norm1"], captured["decoder.layer.00.input"])
    assert not torch.equal(captured["decoder.layer.00.norm2"], captured["decoder.layer.00.attention.residual"])
    assert torch.equal(
        captured["decoder.layer.00.attention.residual"],
        captured["decoder.layer.00.input"] + captured["decoder.layer.00.attention.output"],
    )
    assert torch.equal(
        captured["decoder.layer.00.mlp.activation"],
        F.silu(captured["decoder.layer.00.mlp.gate"]) * captured["decoder.layer.00.mlp.up"],
    )
    assert torch.equal(
        captured["decoder.layer.00.output"],
        captured["decoder.layer.00.attention.residual"] + captured["decoder.layer.00.mlp.down"],
    )
    assert not torch.equal(captured["decoder.layer.00.input"], instrumented_result_b.last_hidden_state)
    assert not torch.equal(captured["decoder.layer.00.input"], instrumented_result_d.last_hidden_state)
    assert torch.equal(captured["decoder.layer.00.kv.key"], instrumented_result_c.past_key_values.layers[0].keys)
    assert torch.equal(captured["decoder.layer.00.kv.value"], instrumented_result_c.past_key_values.layers[0].values)
    assert not torch.equal(captured["decoder.layer.00.kv.key"], instrumented_result_c.past_key_values.layers[1].keys)
    assert not torch.equal(captured["decoder.layer.00.kv.value"], instrumented_result_c.past_key_values.layers[1].values)
    assert not torch.equal(captured["decoder.layer.00.kv.key"], baseline_unmutated_c_key)
    assert not torch.equal(captured["decoder.layer.00.kv.value"], baseline_unmutated_c_value)
    for layer_index, _ in enumerate(instrumented_model.model.layers):
        _assert_tensor_equal(
            captured[f"decoder.layer.{layer_index:02d}.output"],
            baseline_expected[f"decoder.layer.{layer_index:02d}.output"],
        )
    _assert_tensor_equal(captured["decoder.final_norm"], baseline_expected["decoder.final_norm"])
    _assert_captured_tensors_detached_and_storage_isolated(
        captured,
        list(instrumented_model.recorder.source_by_tag["A"].values()),
        input_a,
        [
            instrumented_result_c.past_key_values.layers[0].keys,
            instrumented_result_c.past_key_values.layers[0].values,
        ],
    )


def test_decoder_trace_capture_scenarios_cannot_pass_with_a_fixed_decode_ordinal_or_position():
    baselines = []
    for scenario in _M6_SCENARIOS:
        model = _FakeOuterModel(seed=scenario["seed"])
        sequence = _run_m6_call_sequence(model, scenario=scenario)
        expected = _m6_expected_tensors(model, sequence)
        baselines.append((scenario, sequence, expected))

        assert sequence.first_decode_ordinal == scenario["decode_ordinal"]
        expected_position_ids = torch.tensor(
            [axis_start + _SEQ for axis_start in scenario["axis_starts"]],
            dtype=torch.int64,
        ).view(3, _BATCH, 1)
        _assert_tensor_equal(
            sequence.first_decode_metadata["position_ids"],
            expected_position_ids,
        )
        _assert_tensor_equal(
            sequence.first_decode_metadata["position_ids"],
            sequence.prompt_position_ids[:, :, -1:] + 1,
        )
        assert len(set(expected_position_ids[:, 0, 0].tolist())) == 3
        assert sequence.first_decode_metadata["attention_mask"].tolist() == [
            [*scenario["prompt_mask"], 1]
        ]
        _assert_raw_mrope_axes(
            expected["decoder.decode.00.rope.cos"],
            expected["decoder.decode.00.rope.sin"],
            sequence=1,
        )

    assert {sequence.first_decode_ordinal for _, sequence, _ in baselines} == {3, 5, 6}
    differing_pipeline_ids = (
        "decoder.decode.00.rope.cos",
        "decoder.decode.00.rope.sin",
        "decoder.decode.00.layer.00.input",
        "decoder.decode.00.layer.00.mrope.q",
        "decoder.decode.00.layer.00.output",
        "decoder.decode.00.final_norm",
        "decoder.decode.00.logits",
    )
    for left_index, (_, left_sequence, left_expected) in enumerate(baselines):
        for _, right_sequence, right_expected in baselines[left_index + 1 :]:
            for semantic_id in differing_pipeline_ids:
                assert not torch.equal(
                    left_expected[semantic_id],
                    right_expected[semantic_id],
                )
            for left_pair, right_pair in zip(
                left_sequence.first_decode_cache,
                right_sequence.first_decode_cache,
                strict=True,
            ):
                for left_tensor, right_tensor in zip(left_pair, right_pair, strict=True):
                    assert not torch.equal(left_tensor, right_tensor)


@pytest.mark.parametrize(
    "scenario",
    _M6_SCENARIOS,
    ids=lambda scenario: scenario["id"],
)
def test_decoder_trace_capture_first_cached_decode_is_semantic_exact_append_only_and_detached(
    scenario,
):
    seed = scenario["seed"]
    baseline_model = _FakeOuterModel(seed=seed)
    baseline_sequence = _run_m6_call_sequence(
        baseline_model,
        scenario=scenario,
    )
    baseline_expected = _m6_expected_tensors(baseline_model, baseline_sequence)

    assert baseline_sequence.uncached_result.past_key_values is None
    assert baseline_sequence.first_decode_ordinal == scenario["decode_ordinal"]
    assert "position_ids" not in baseline_sequence.requested_first_decode_metadata
    assert len(baseline_sequence.prefill_cache) == _FAKE_LAYER_COUNT
    assert len(baseline_sequence.first_decode_cache) == _FAKE_LAYER_COUNT
    _assert_tensor_equal(
        baseline_sequence.prefill_cache[0][0],
        baseline_model.recorder.by_tag["A"]["decoder.layer.00.mrope.k"],
    )
    _assert_tensor_equal(
        baseline_sequence.prefill_cache[0][1],
        _reshape_kv(baseline_model.recorder.by_tag["A"]["decoder.layer.00.v"]),
    )
    _assert_exact_cache_append(
        baseline_sequence.prefill_cache,
        baseline_sequence.first_decode_cache,
        appended_layer0_key=baseline_model.recorder.by_tag["D"]["decoder.layer.00.mrope.k"],
        appended_layer0_value=_reshape_kv(
            baseline_model.recorder.by_tag["D"]["decoder.layer.00.v"]
        ),
    )
    _assert_tensor_equal(
        baseline_model.recorder.by_tag["D"]["decoder.logits"],
        baseline_model.lm_head(baseline_model.recorder.by_tag["D"]["decoder.final_norm"]),
    )
    assert not torch.equal(
        baseline_model.recorder.by_tag["D"]["decoder.layer.00.input"],
        baseline_model.recorder.by_tag["E"]["decoder.layer.00.input"],
    )
    assert not torch.equal(
        baseline_model.recorder.by_tag["D"]["decoder.logits"],
        baseline_model.recorder.by_tag["E"]["decoder.logits"],
    )

    instrumented_model = _FakeOuterModel(seed=seed)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, instrumented_model)
    try:
        instrumented_sequence = _run_m6_call_sequence(
            instrumented_model,
            scenario=scenario,
        )
        # Instrumentation must be observational before finish validates snapshots.
        for actual, expected in (
            (instrumented_sequence.uncached_result, baseline_sequence.uncached_result),
            (instrumented_sequence.prefill_result, baseline_sequence.prefill_result),
            (instrumented_sequence.first_decode_result, baseline_sequence.first_decode_result),
            (instrumented_sequence.later_decode_result, baseline_sequence.later_decode_result),
        ):
            _assert_tensor_equal(actual.last_hidden_state, expected.last_hidden_state)
            _assert_tensor_equal(actual.logits, expected.logits)
        for actual_cache, expected_cache in (
            (instrumented_sequence.prefill_cache, baseline_sequence.prefill_cache),
            (instrumented_sequence.first_decode_cache, baseline_sequence.first_decode_cache),
        ):
            for actual_pair, expected_pair in zip(actual_cache, expected_cache, strict=True):
                for actual, expected in zip(actual_pair, expected_pair, strict=True):
                    _assert_tensor_equal(actual, expected)
        for tag in ("A", "C", "D", "E"):
            assert set(instrumented_model.recorder.by_tag[tag]) == set(
                baseline_model.recorder.by_tag[tag]
            )
            for semantic_id, actual in instrumented_model.recorder.by_tag[tag].items():
                _assert_tensor_equal(actual, baseline_model.recorder.by_tag[tag][semantic_id])
        assert instrumented_sequence.first_decode_ordinal == scenario["decode_ordinal"]
        captured = trace.finish()
    finally:
        _assert_closed(trace, instrumented_model, owner, original_fn, baseline_hooks)

    assert set(trace.semantic_ids) == _EXPECTED_M6_CAPTURE_KEYS
    assert set(captured) == _EXPECTED_M6_CAPTURE_KEYS
    for semantic_id, expected in baseline_expected.items():
        _assert_tensor_equal(captured[semantic_id], expected)

    _assert_raw_mrope_axes(
        captured["decoder.decode.00.rope.cos"],
        captured["decoder.decode.00.rope.sin"],
        sequence=1,
    )
    assert captured["decoder.decode.00.attention_mask"].dtype == torch.int64
    assert captured["decoder.decode.00.cache_position"].dtype == torch.int64
    assert captured["decoder.decode.00.position_ids"].dtype == torch.int64
    assert captured["decoder.decode.00.attention_mask"].tolist() == [
        [*scenario["prompt_mask"], 1]
    ]
    assert captured["decoder.decode.00.cache_position"].tolist() == [_SEQ]
    assert captured["decoder.decode.00.position_ids"].shape == (3, _BATCH, 1)
    expected_decode_position_ids = instrumented_sequence.prompt_position_ids[:, :, -1:] + 1
    _assert_tensor_equal(
        captured["decoder.decode.00.position_ids"],
        expected_decode_position_ids,
    )
    assert captured["decoder.decode.00.layer.00.input"].shape == (_BATCH, 1, _HIDDEN)
    assert captured["decoder.decode.00.layer.00.attention.context"].shape == (
        _BATCH,
        1,
        _Q_FLAT,
    )
    assert captured["decoder.decode.00.final_norm"].shape == (_BATCH, 1, _HIDDEN)
    assert captured["decoder.decode.00.logits"].shape == (_BATCH, 1, _FAKE_VOCAB_SIZE)
    assert torch.equal(
        captured["decoder.decode.00.layer.00.attention.residual"],
        captured["decoder.decode.00.layer.00.input"]
        + captured["decoder.decode.00.layer.00.attention.output"],
    )
    assert torch.equal(
        captured["decoder.decode.00.layer.00.mlp.activation"],
        F.silu(captured["decoder.decode.00.layer.00.mlp.gate"])
        * captured["decoder.decode.00.layer.00.mlp.up"],
    )
    assert torch.equal(
        captured["decoder.decode.00.layer.00.output"],
        captured["decoder.decode.00.layer.00.attention.residual"]
        + captured["decoder.decode.00.layer.00.mlp.down"],
    )

    captured_prefill_cache = tuple(
        (
            captured[f"decoder.layer.{layer_index:02d}.kv.key"],
            captured[f"decoder.layer.{layer_index:02d}.kv.value"],
        )
        for layer_index in range(_FAKE_LAYER_COUNT)
    )
    captured_decode_cache = tuple(
        (
            captured[f"decoder.decode.00.layer.{layer_index:02d}.kv.key"],
            captured[f"decoder.decode.00.layer.{layer_index:02d}.kv.value"],
        )
        for layer_index in range(_FAKE_LAYER_COUNT)
    )
    _assert_exact_cache_append(
        captured_prefill_cache,
        captured_decode_cache,
        appended_layer0_key=captured["decoder.decode.00.layer.00.mrope.k"],
        appended_layer0_value=_reshape_kv(captured["decoder.decode.00.layer.00.v"]),
    )
    assert all(
        not tensor.requires_grad and tensor.grad_fn is None for tensor in captured.values()
    )

    _assert_captured_tensors_detached_and_storage_isolated(
        captured,
        [
            *instrumented_model.recorder.source_by_tag["A"].values(),
            *instrumented_model.recorder.source_by_tag["D"].values(),
            *instrumented_sequence.first_decode_metadata_sources.values(),
        ],
        instrumented_sequence.first_decode,
        [
            tensor
            for cache_sources in (
                instrumented_sequence.prefill_cache_sources,
                instrumented_sequence.first_decode_cache_sources,
            )
            for key_value_pair in cache_sources
            for tensor in key_value_pair
        ],
    )

    # The second cached call already mutated the source cache to length prompt+2.
    # Captured post-decode snapshots must remain the first decode at prompt+1.
    assert instrumented_sequence.later_decode_result.past_key_values is not None
    assert all(
        layer.keys.shape[2] == _SEQ + 2
        for layer in instrumented_sequence.later_decode_result.past_key_values.layers
    )
    frozen = {semantic_id: _clone(tensor) for semantic_id, tensor in captured.items()}
    for layer in instrumented_sequence.later_decode_result.past_key_values.layers:
        layer.keys.add_(19.0)
        layer.values.sub_(23.0)
    instrumented_sequence.first_decode_metadata_sources["attention_mask"].zero_()
    instrumented_sequence.first_decode_metadata_sources["cache_position"].add_(100)
    instrumented_sequence.first_decode_metadata_sources["position_ids"].add_(100)
    instrumented_sequence.first_decode.add_(100.0)
    for semantic_id, expected in frozen.items():
        _assert_tensor_equal(captured[semantic_id], expected)


def _mutate_unique_sources_before_finish(tensors: list[torch.Tensor]) -> None:
    seen: set[tuple[int, str]] = set()
    with torch.no_grad():
        for index, tensor in enumerate(tensors):
            identity = _tensor_storage_identity(tensor)
            if identity in seen:
                continue
            seen.add(identity)
            tensor.add_(7 + index)


def test_decoder_trace_capture_owns_all_snapshots_before_finish_without_second_decode():
    scenario = _DEFAULT_M6_SCENARIO
    baseline_model = _FakeOuterModel(seed=scenario["seed"])
    baseline_sequence = _run_m6_call_sequence(
        baseline_model,
        scenario=scenario,
        include_later_decode=False,
    )
    baseline_expected = _m6_expected_tensors(baseline_model, baseline_sequence)

    model = _FakeOuterModel(seed=scenario["seed"])
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        sequence = _run_m6_call_sequence(
            model,
            scenario=scenario,
            include_later_decode=False,
        )
        assert sequence.later_decode_result is None
        assert model.decoder_forward_count == scenario["decode_ordinal"]
        for actual, expected in (
            (sequence.uncached_result, baseline_sequence.uncached_result),
            (sequence.prefill_result, baseline_sequence.prefill_result),
            (sequence.first_decode_result, baseline_sequence.first_decode_result),
        ):
            _assert_tensor_equal(actual.last_hidden_state, expected.last_hidden_state)
            _assert_tensor_equal(actual.logits, expected.logits)
        for actual_cache, expected_cache in (
            (sequence.prefill_cache, baseline_sequence.prefill_cache),
            (sequence.first_decode_cache, baseline_sequence.first_decode_cache),
        ):
            for actual_pair, expected_pair in zip(actual_cache, expected_cache, strict=True):
                for actual, expected in zip(actual_pair, expected_pair, strict=True):
                    _assert_tensor_equal(actual, expected)

        _mutate_unique_sources_before_finish(
            [
                *model.recorder.source_by_tag["A"].values(),
                *model.recorder.source_by_tag["D"].values(),
                *sequence.first_decode_metadata_sources.values(),
                sequence.first_decode,
                *[
                    tensor
                    for cache_sources in (
                        sequence.prefill_cache_sources,
                        sequence.first_decode_cache_sources,
                    )
                    for key_value_pair in cache_sources
                    for tensor in key_value_pair
                ],
            ]
        )
        captured = trace.finish()
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)

    assert set(captured) == _EXPECTED_M6_CAPTURE_KEYS
    for semantic_id, expected in baseline_expected.items():
        _assert_tensor_equal(captured[semantic_id], expected)


@pytest.mark.parametrize(
    ("cache_case", "layer_index", "side", "defect"),
    _DECODE_CACHE_TENSOR_CASES,
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_decoder_trace_capture_rejects_every_input_cache_layer_side_tensor_defect(
    cache_case,
    layer_index,
    side,
    defect,
):
    del cache_case
    contract_error = _contract_error_type()
    model = _FakeOuterModel(seed=56)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        prefill = _run_m6_until_cached_prefill(model)
        valid_cache = _clone_cache(prefill.prefill_result.past_key_values)
        malformed_cache = _clone_cache(valid_cache)
        _apply_decode_cache_tensor_defect(
            malformed_cache,
            layer_index=layer_index,
            side=side,
            defect=defect,
            stage="input",
        )
        # The decoder pre-hook sees malformed_cache; fake attention receives a
        # repaired clone only so a rejection cannot come from test arithmetic.
        model.input_cache_repair = lambda _cache: _clone_cache(valid_cache)

        def _malformed_decode():
            _invoke_first_decode(
                model,
                prefill,
                past_key_values=malformed_cache,
            )

        _expect_contract_error_from_operation(trace, contract_error, _malformed_decode)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)


@pytest.mark.parametrize(
    ("cache_case", "layer_index", "side", "defect"),
    _DECODE_CACHE_TENSOR_CASES,
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_decoder_trace_capture_rejects_every_post_cache_layer_side_tensor_defect(
    cache_case,
    layer_index,
    side,
    defect,
):
    del cache_case
    contract_error = _contract_error_type()
    model = _FakeOuterModel(seed=57)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        prefill = _run_m6_until_cached_prefill(model)

        def _mutate_post_cache(cache):
            _apply_decode_cache_tensor_defect(
                cache,
                layer_index=layer_index,
                side=side,
                defect=defect,
                stage="post",
            )

        model.cache_mutator = _mutate_post_cache

        def _malformed_decode():
            _invoke_first_decode(
                model,
                prefill,
                past_key_values=prefill.prefill_result.past_key_values,
            )

        _expect_contract_error_from_operation(trace, contract_error, _malformed_decode)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)


@pytest.mark.parametrize(
    ("cache_case", "layer_index", "side"),
    _CACHE_LAYER_SIDE_CASES,
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_decoder_trace_capture_rejects_finite_same_shape_input_cache_prefix_drift(
    cache_case,
    layer_index,
    side,
):
    del cache_case
    contract_error = _contract_error_type()
    model = _FakeOuterModel(seed=65)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        prefill = _run_m6_until_cached_prefill(model)
        valid_cache = _clone_cache(prefill.prefill_result.past_key_values)
        malformed_cache = _clone_cache(valid_cache)
        _add_finite_cache_drift(
            malformed_cache,
            layer_index=layer_index,
            side=side,
            sequence_index=0,
        )
        malformed_tensor = getattr(malformed_cache.layers[layer_index], side)
        valid_tensor = getattr(valid_cache.layers[layer_index], side)
        assert not torch.equal(malformed_tensor, valid_tensor)
        model.input_cache_repair = lambda _cache: _clone_cache(valid_cache)

        def _drifted_input_cache():
            _invoke_first_decode(
                model,
                prefill,
                past_key_values=malformed_cache,
            )

        _expect_contract_error_from_operation(trace, contract_error, _drifted_input_cache)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)


@pytest.mark.parametrize("side", ("keys", "values"))
def test_decoder_trace_capture_rejects_only_corrupted_appended_layer0_cache_value(side):
    contract_error = _contract_error_type()
    model = _FakeOuterModel(seed=66)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        prefill = _run_m6_until_cached_prefill(model)

        def _mutate_only_appended_layer0(cache):
            before_cache = _cache_snapshot(cache)
            before = _clone(getattr(cache.layers[0], side))
            _add_finite_cache_drift(
                cache,
                layer_index=0,
                side=side,
                sequence_index=-1,
            )
            after = getattr(cache.layers[0], side)
            _assert_tensor_equal(after[:, :, :-1, :], before[:, :, :-1, :])
            for layer_index, (before_keys, before_values) in enumerate(before_cache):
                for member, expected in (("keys", before_keys), ("values", before_values)):
                    if layer_index == 0 and member == side:
                        continue
                    _assert_tensor_equal(
                        getattr(cache.layers[layer_index], member),
                        expected,
                    )
            canonical_append = (
                model.recorder.by_tag["D"]["decoder.layer.00.mrope.k"]
                if side == "keys"
                else _reshape_kv(model.recorder.by_tag["D"]["decoder.layer.00.v"])
            )
            assert not torch.equal(after[:, :, -1:, :], canonical_append)

        model.cache_mutator = _mutate_only_appended_layer0

        def _corrupted_append():
            _invoke_first_decode(
                model,
                prefill,
                past_key_values=prefill.prefill_result.past_key_values,
            )

        _expect_contract_error_from_operation(trace, contract_error, _corrupted_append)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)


@pytest.mark.parametrize("stage", ("input", "post"))
@pytest.mark.parametrize(
    "structure_case",
    ("missing", "empty", "partial", "extra", "lookalike"),
)
def test_decoder_trace_capture_requires_exact_real_decode_cache_structure(
    stage,
    structure_case,
):
    contract_error = _contract_error_type()
    model = _FakeOuterModel(seed=58)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        prefill = _run_m6_until_cached_prefill(model)
        valid_cache = _clone_cache(prefill.prefill_result.past_key_values)
        if stage == "input":
            malformed_cache = _make_decode_cache_structure_case(
                _clone_cache(valid_cache),
                structure_case,
            )
            model.input_cache_repair = lambda _cache: _clone_cache(valid_cache)

            def _malformed_decode():
                _invoke_first_decode(
                    model,
                    prefill,
                    past_key_values=malformed_cache,
                )

        else:
            def _mutate_post_structure(cache):
                replacement = _make_decode_cache_structure_case(cache, structure_case)
                return _DROP_CACHE_OUTPUT if replacement is None else replacement

            model.cache_mutator = _mutate_post_structure

            def _malformed_decode():
                _invoke_first_decode(
                    model,
                    prefill,
                    past_key_values=prefill.prefill_result.past_key_values,
                )

        _expect_contract_error_from_operation(trace, contract_error, _malformed_decode)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)


@pytest.mark.parametrize(
    ("cache_case", "layer_index", "side", "defect"),
    _DECODE_CACHE_TENSOR_CASES,
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_decoder_prefill_trace_capture_rejects_every_manual_prefill_cache_layer_side_defect(
    cache_case,
    layer_index,
    side,
    defect,
):
    del cache_case
    contract_error = _contract_error_type()
    input_a, _, input_c, _ = _build_inputs()
    model = _FakeOuterModel(seed=60)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        _set_active(model, "A", input_a, use_cache=False)

        def _mutate_prefill_cache(cache):
            _apply_decode_cache_tensor_defect(
                cache,
                layer_index=layer_index,
                side=side,
                defect=defect,
                stage="prefill",
            )

        model.cache_mutator = _mutate_prefill_cache

        def _malformed_prefill():
            _set_active(model, "C", input_c, use_cache=True)

        _expect_contract_error_from_operation(trace, contract_error, _malformed_prefill)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)


@pytest.mark.parametrize(
    "structure_case",
    ("missing", "empty", "partial", "extra", "lookalike"),
)
def test_decoder_prefill_trace_capture_requires_exact_real_manual_prefill_cache_structure(
    structure_case,
):
    contract_error = _contract_error_type()
    input_a, _, input_c, _ = _build_inputs()
    model = _FakeOuterModel(seed=61)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        _set_active(model, "A", input_a, use_cache=False)

        def _mutate_prefill_structure(cache):
            replacement = _make_decode_cache_structure_case(cache, structure_case)
            return _DROP_CACHE_OUTPUT if replacement is None else replacement

        model.cache_mutator = _mutate_prefill_structure

        def _malformed_prefill():
            _set_active(model, "C", input_c, use_cache=True)

        _expect_contract_error_from_operation(trace, contract_error, _malformed_prefill)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)


def test_decoder_prefill_trace_capture_finish_remains_compatible_without_decode():
    seed = 62
    input_a, _, input_c, _ = _build_inputs()
    baseline_model = _FakeOuterModel(seed=seed)
    baseline_a = _set_active(baseline_model, "A", input_a, use_cache=False)
    baseline_c = _set_active(baseline_model, "C", input_c, use_cache=True)
    baseline_expected = {
        semantic_id: _clone(baseline_model.recorder.by_tag["A"][semantic_id])
        for semantic_id in _EXPECTED_CAPTURE_KEYS
        if ".kv." not in semantic_id
    }
    baseline_expected["decoder.layer.00.kv.key"] = _clone(
        baseline_c.past_key_values.layers[0].keys
    )
    baseline_expected["decoder.layer.00.kv.value"] = _clone(
        baseline_c.past_key_values.layers[0].values
    )

    model = _FakeOuterModel(seed=seed)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        result_a = _set_active(model, "A", input_a.clone(), use_cache=False)
        result_c = _set_active(model, "C", input_c.clone(), use_cache=True)
        _assert_tensor_equal(result_a.last_hidden_state, baseline_a.last_hidden_state)
        _assert_tensor_equal(result_a.logits, baseline_a.logits)
        _assert_tensor_equal(result_c.last_hidden_state, baseline_c.last_hidden_state)
        _assert_tensor_equal(result_c.logits, baseline_c.logits)
        for actual_layer, expected_layer in zip(
            result_c.past_key_values.layers,
            baseline_c.past_key_values.layers,
            strict=True,
        ):
            _assert_cache_layer_equal(actual_layer, expected_layer)
        captured = trace.finish()
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)

    assert model.decoder_forward_count == 2
    assert set(trace.semantic_ids) == _EXPECTED_CAPTURE_KEYS
    assert set(captured) == _EXPECTED_CAPTURE_KEYS
    for semantic_id, expected in baseline_expected.items():
        _assert_tensor_equal(captured[semantic_id], expected)


@pytest.mark.parametrize(
    ("mutator_name", "mutator"),
    _make_post_decode_cache_mutators(),
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_decoder_trace_capture_fails_closed_for_malformed_post_decode_cache(
    mutator_name,
    mutator,
):
    del mutator_name
    contract_error = _contract_error_type()
    model = _FakeOuterModel(seed=51)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        prefill = _run_m6_until_cached_prefill(model)
        model.cache_mutator = mutator

        def _malformed_decode():
            _invoke_first_decode(
                model,
                prefill,
                past_key_values=prefill.prefill_result.past_key_values,
            )

        _expect_contract_error_from_operation(trace, contract_error, _malformed_decode)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)


@pytest.mark.parametrize(
    "cache_case",
    ("missing", "empty", "partial", "short", "non_finite"),
)
def test_decoder_trace_capture_fails_closed_for_missing_partial_or_malformed_input_cache(
    cache_case,
):
    contract_error = _contract_error_type()
    model = _FakeOuterModel(seed=52)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        prefill = _run_m6_until_cached_prefill(model)
        valid_cache = _clone_cache(prefill.prefill_result.past_key_values)
        input_cache = _clone_cache(valid_cache)
        if cache_case == "missing":
            input_cache = None
        elif cache_case == "empty":
            input_cache.layers.clear()
        elif cache_case == "partial":
            input_cache.layers.pop()
        elif cache_case == "short":
            for layer in input_cache.layers:
                layer.keys = layer.keys[:, :, :-1, :]
                layer.values = layer.values[:, :, :-1, :]
        elif cache_case == "non_finite":
            input_cache.layers[0].keys[0, 0, 0, 0] = torch.nan
        else:
            raise AssertionError(f"unknown input cache case: {cache_case}")
        model.input_cache_repair = lambda _cache: _clone_cache(valid_cache)

        def _malformed_decode():
            _invoke_first_decode(
                model,
                prefill,
                past_key_values=input_cache,
            )

        _expect_contract_error_from_operation(trace, contract_error, _malformed_decode)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)


@pytest.mark.parametrize(
    ("metadata_case", "metadata_source", "mutator"),
    _make_decode_metadata_mutators(),
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_decoder_trace_capture_fails_closed_for_malformed_decode_metadata(
    metadata_case,
    metadata_source,
    mutator,
):
    del metadata_case
    contract_error = _contract_error_type()
    model = _FakeOuterModel(seed=53)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        prefill = _run_m6_until_cached_prefill(model)
        metadata = _valid_first_decode_metadata()
        if metadata_source == "decoder":
            mutator(metadata)
        elif metadata_source == "self_attn":
            model.self_attn_metadata_mutator = mutator
        else:
            raise AssertionError(f"unknown metadata source: {metadata_source}")

        def _malformed_decode():
            _invoke_first_decode(
                model,
                prefill,
                past_key_values=prefill.prefill_result.past_key_values,
                metadata=metadata,
            )

        _expect_contract_error_from_operation(trace, contract_error, _malformed_decode)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)


@pytest.mark.parametrize("mask_case", _INVALID_DECODE_MASK_CONTENT_CASES)
def test_decoder_trace_capture_rejects_semantically_invalid_same_shape_integer_mask(
    mask_case,
):
    contract_error = _contract_error_type()
    model = _FakeOuterModel(seed=64)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        prefill = _run_m6_until_cached_prefill(model)
        metadata = _valid_first_decode_metadata()
        invalid_mask = _invalid_decode_attention_mask(prefill.prompt_mask, mask_case)
        assert invalid_mask.dtype == torch.int64
        assert tuple(invalid_mask.shape) == (_BATCH, _SEQ + 1)
        if mask_case == "wrong_prefix_bit":
            assert not torch.equal(invalid_mask[:, :-1], prefill.prompt_mask)
            assert invalid_mask[:, -1:].tolist() == [[1]]
        else:
            _assert_tensor_equal(invalid_mask[:, :-1], prefill.prompt_mask)
        metadata["attention_mask"] = invalid_mask

        def _malformed_decode():
            _invoke_first_decode(
                model,
                prefill,
                past_key_values=prefill.prefill_result.past_key_values,
                metadata=metadata,
            )

        _expect_contract_error_from_operation(trace, contract_error, _malformed_decode)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)


@pytest.mark.parametrize("target", ("pipeline", "logits"))
@pytest.mark.parametrize(
    ("nonfinite_case", "nonfinite_value"),
    (("nan", torch.nan), ("positive_infinity", torch.inf), ("negative_infinity", -torch.inf)),
)
def test_decoder_trace_capture_rejects_nonfinite_decode_pipeline_and_logits(
    target,
    nonfinite_case,
    nonfinite_value,
):
    del nonfinite_case
    contract_error = _contract_error_type()
    model = _FakeOuterModel(seed=59)
    target_module = (
        model.model.layers[0].self_attn.q_proj
        if target == "pipeline"
        else model.lm_head
    )
    finite_output: dict[str, torch.Tensor] = {}

    def _corrupt_decode_output(_module, _arguments, output):
        if model.active_tag != "D":
            return output
        finite_output["tensor"] = output
        corrupted = output.clone()
        corrupted.reshape(-1)[0] = nonfinite_value
        return corrupted

    corruption_handle = target_module.register_forward_hook(_corrupt_decode_output)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(
        trace,
        model,
        expected_baseline_hooks=1,
    )
    repair_handle = None
    if target == "pipeline":
        def _repair_downstream_output(_module, _arguments, output):
            if model.active_tag == "D":
                assert not torch.isfinite(output).all()
                return finite_output["tensor"]
            return output

        # Registered after capture: the trace sees corruption, attention does not.
        repair_handle = target_module.register_forward_hook(_repair_downstream_output)
    try:
        prefill = _run_m6_until_cached_prefill(model)

        def _nonfinite_decode():
            _invoke_first_decode(
                model,
                prefill,
                past_key_values=prefill.prefill_result.past_key_values,
            )

        _expect_contract_error_from_operation(trace, contract_error, _nonfinite_decode)
    finally:
        if repair_handle is not None:
            repair_handle.remove()
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)
        corruption_handle.remove()
        assert _hook_count(model) == 0


@pytest.mark.parametrize("tensor_defect", ("shape", "dtype", "device"))
def test_decoder_trace_capture_rejects_repaired_decode_q_abi_corruption(tensor_defect):
    contract_error = _contract_error_type()
    model = _FakeOuterModel(seed=67)
    q_proj = model.model.layers[0].self_attn.q_proj
    finite_output: dict[str, torch.Tensor] = {}

    def _corrupt_decode_q(_module, _arguments, output):
        if model.active_tag != "D":
            return output
        finite_output["tensor"] = output
        if tensor_defect == "shape":
            return output[..., :-1]
        if tensor_defect == "dtype":
            return output.to(torch.float64)
        if tensor_defect == "device":
            return torch.empty(output.shape, dtype=output.dtype, device="meta")
        raise AssertionError(f"unknown q defect: {tensor_defect}")

    corruption_handle = q_proj.register_forward_hook(_corrupt_decode_q)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(
        trace,
        model,
        expected_baseline_hooks=1,
    )

    def _repair_q_for_attention(_module, _arguments, output):
        if model.active_tag == "D":
            return finite_output["tensor"]
        return output

    repair_handle = q_proj.register_forward_hook(_repair_q_for_attention)
    try:
        prefill = _run_m6_until_cached_prefill(model)

        def _corrupted_q_decode():
            _invoke_first_decode(
                model,
                prefill,
                past_key_values=prefill.prefill_result.past_key_values,
            )

        _expect_contract_error_from_operation(trace, contract_error, _corrupted_q_decode)
    finally:
        repair_handle.remove()
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)
        corruption_handle.remove()
        assert _hook_count(model) == 0


def test_decoder_trace_capture_fails_closed_for_cached_query_longer_than_one():
    contract_error = _contract_error_type()
    model = _FakeOuterModel(seed=54)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        prefill = _run_m6_until_cached_prefill(model)
        two_token_query = torch.cat((prefill.first_decode, prefill.first_decode + 1.0), dim=1)
        coherent_two_token_metadata = {
            "attention_mask": torch.cat(
                (
                    prefill.prompt_mask,
                    torch.ones((_BATCH, 2), dtype=torch.int64),
                ),
                dim=1,
            ),
            "cache_position": torch.tensor([_SEQ, _SEQ + 1], dtype=torch.int64),
            "position_ids": None,
        }

        def _malformed_decode():
            _invoke_first_decode(
                model,
                prefill,
                past_key_values=prefill.prefill_result.past_key_values,
                metadata=coherent_two_token_metadata,
                query=two_token_query,
            )

        _expect_contract_error_from_operation(trace, contract_error, _malformed_decode)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)


def test_decoder_trace_capture_restores_hooks_after_cached_decode_forward_exception():
    model = _FakeOuterModel(seed=55)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        prefill = _run_m6_until_cached_prefill(model)
        model.raise_in_layer0 = True
        with pytest.raises(RuntimeError, match="layer0 boom"):
            _invoke_first_decode(
                model,
                prefill,
                past_key_values=prefill.prefill_result.past_key_values,
            )
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)

    model.raise_in_layer0 = False
    replacement_trace = _DecoderPrefillTraceCapture()
    replacement_owner, replacement_fn, replacement_hooks, _ = _register_trace(
        replacement_trace,
        model,
    )
    _assert_closed(
        replacement_trace,
        model,
        replacement_owner,
        replacement_fn,
        replacement_hooks,
    )


def test_decoder_trace_runtime_error_is_immediately_reusable_before_explicit_close():
    model = _FakeOuterModel(seed=68)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    replacement_trace = None
    try:
        prefill = _run_m6_until_cached_prefill(model)
        model.raise_in_layer0 = True
        with pytest.raises(RuntimeError, match="layer0 boom"):
            _invoke_first_decode(
                model,
                prefill,
                past_key_values=prefill.prefill_result.past_key_values,
            )

        assert owner.apply_multimodal_rotary_pos_emb is original_fn
        assert _hook_count(model) == baseline_hooks
        model.raise_in_layer0 = False
        replacement_trace = _DecoderPrefillTraceCapture()
        replacement_owner, replacement_fn, replacement_hooks, _ = _register_trace(
            replacement_trace,
            model,
        )
        _assert_closed(
            replacement_trace,
            model,
            replacement_owner,
            replacement_fn,
            replacement_hooks,
        )
        replacement_trace = None
    finally:
        if replacement_trace is not None:
            replacement_trace.close()
        trace.close()
        assert owner.apply_multimodal_rotary_pos_emb is original_fn
        assert _hook_count(model) == baseline_hooks


def test_decoder_trace_contract_error_is_immediately_reusable_before_explicit_close():
    contract_error = _contract_error_type()
    model = _FakeOuterModel(seed=63)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    replacement_trace = None
    try:
        prefill = _run_m6_until_cached_prefill(model)
        metadata = _valid_first_decode_metadata()
        metadata["attention_mask"] = None
        try:
            _invoke_first_decode(
                model,
                prefill,
                past_key_values=prefill.prefill_result.past_key_values,
                metadata=metadata,
            )
        except contract_error as error:
            _assert_contract_error(error)
        else:
            with pytest.raises(contract_error) as exc_info:
                trace.finish()
            _assert_contract_error(exc_info.value)

        assert owner.apply_multimodal_rotary_pos_emb is original_fn
        assert _hook_count(model) == baseline_hooks
        replacement_trace = _DecoderPrefillTraceCapture()
        replacement_owner, replacement_fn, replacement_hooks, _ = _register_trace(
            replacement_trace,
            model,
        )
        _assert_closed(
            replacement_trace,
            model,
            replacement_owner,
            replacement_fn,
            replacement_hooks,
        )
        replacement_trace = None
    finally:
        if replacement_trace is not None:
            replacement_trace.close()
        trace.close()
        assert owner.apply_multimodal_rotary_pos_emb is original_fn
        assert _hook_count(model) == baseline_hooks


def test_decoder_prefill_trace_capture_fail_closed_for_missing_cache():
    contract_error = _contract_error_type()
    input_a, _, _, _ = _build_inputs()

    model = _FakeOuterModel(seed=11)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        _set_active(model, "A", input_a, use_cache=False)
        with pytest.raises(contract_error) as exc_info:
            trace.finish()
        _assert_contract_error(exc_info.value)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)


@pytest.mark.parametrize(
    ("mutator_name", "mutator"),
    _make_cache_layer_mutators(),
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_decoder_prefill_trace_capture_fail_closed_for_malformed_cache(mutator_name, mutator):
    del mutator_name
    contract_error = _contract_error_type()
    input_a, _, input_c, _ = _build_inputs()

    model = _FakeOuterModel(seed=12)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        _set_active(model, "A", input_a, use_cache=False)
        model.cache_mutator = mutator
        def _malformed_operation():
            _set_active(model, "C", input_c, use_cache=True)
        _expect_contract_error_from_operation(trace, contract_error, _malformed_operation)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)


def test_decoder_prefill_trace_capture_duplicate_register_and_cleanup_keep_first_trace_intact():
    contract_error = _contract_error_type()
    input_a, _, input_c, _ = _build_inputs()

    model1 = _FakeOuterModel(seed=13)
    model2 = _FakeOuterModel(seed=23)
    trace1 = _DecoderPrefillTraceCapture()
    owner1, original_fn, baseline_hooks1, active_hooks1 = _register_trace(trace1, model1)
    owner2 = _owner_module(model2)
    baseline_hooks2 = _hook_count(model2)
    assert owner1 is owner2
    assert baseline_hooks2 == 0
    first_wrapper = owner1.apply_multimodal_rotary_pos_emb

    with pytest.raises(contract_error) as exc_info:
        trace1.register(model1)
    _assert_contract_error(exc_info.value)
    assert owner1.apply_multimodal_rotary_pos_emb is first_wrapper
    assert _hook_count(model1) == active_hooks1
    assert _hook_count(model2) == baseline_hooks2

    second_trace_same_model = _DecoderPrefillTraceCapture()
    with pytest.raises(contract_error) as exc_info:
        second_trace_same_model.register(model1)
    _assert_contract_error(exc_info.value)
    assert owner1.apply_multimodal_rotary_pos_emb is first_wrapper
    assert _hook_count(model1) == active_hooks1
    assert _hook_count(model2) == baseline_hooks2
    second_trace_same_model.close()
    assert owner1.apply_multimodal_rotary_pos_emb is first_wrapper
    assert _hook_count(model1) == active_hooks1
    assert _hook_count(model2) == baseline_hooks2

    second_trace = _DecoderPrefillTraceCapture()
    with pytest.raises(contract_error) as exc_info:
        second_trace.register(model2)
    _assert_contract_error(exc_info.value)

    assert owner1.apply_multimodal_rotary_pos_emb is first_wrapper
    assert _hook_count(model1) == active_hooks1
    assert _hook_count(model2) == baseline_hooks2
    second_trace.close()
    assert owner1.apply_multimodal_rotary_pos_emb is first_wrapper
    assert _hook_count(model1) == active_hooks1
    assert _hook_count(model2) == baseline_hooks2

    try:
        _set_active(model1, "A", input_a, use_cache=False)
        _set_active(model1, "C", input_c, use_cache=True)
        captured = trace1.finish()
    finally:
        _assert_closed(trace1, model1, owner1, original_fn, baseline_hooks1)

    assert set(captured) == _EXPECTED_CAPTURE_KEYS


def test_decoder_prefill_trace_capture_partial_register_failure_restores_global_and_hooks(
    monkeypatch: pytest.MonkeyPatch,
):
    contract_error = _contract_error_type()
    original_register_forward_hook = torch.nn.Module.register_forward_hook
    original_register_forward_pre_hook = torch.nn.Module.register_forward_pre_hook

    def _run_registration_probe(*, fail_after: int | None):
        model = _FakeOuterModel(seed=31)
        trace = _DecoderPrefillTraceCapture()
        owner = _owner_module(model)
        original_fn = owner.apply_multimodal_rotary_pos_emb
        baseline_hooks = _hook_count(model)
        state = {
            "successful_registrations": 0,
            "saw_partial_registration": False,
        }

        def _register_forward_hook_wrapper(module, hook, *args, **kwargs):
            if fail_after is not None and state["successful_registrations"] >= fail_after:
                state["saw_partial_registration"] = (
                    owner.apply_multimodal_rotary_pos_emb is not original_fn
                    or _hook_count(model) > baseline_hooks
                )
                raise contract_error(
                    "forced_partial_register_failure",
                    "forced partial register failure",
                )
            state["successful_registrations"] += 1
            return original_register_forward_hook(module, hook, *args, **kwargs)

        def _register_forward_pre_hook_wrapper(module, hook, *args, **kwargs):
            if fail_after is not None and state["successful_registrations"] >= fail_after:
                state["saw_partial_registration"] = (
                    owner.apply_multimodal_rotary_pos_emb is not original_fn
                    or _hook_count(model) > baseline_hooks
                )
                raise contract_error(
                    "forced_partial_register_failure",
                    "forced partial register failure",
                )
            state["successful_registrations"] += 1
            return original_register_forward_pre_hook(module, hook, *args, **kwargs)

        with monkeypatch.context() as patch_context:
            patch_context.setattr(
                torch.nn.Module,
                "register_forward_hook",
                _register_forward_hook_wrapper,
            )
            patch_context.setattr(
                torch.nn.Module,
                "register_forward_pre_hook",
                _register_forward_pre_hook_wrapper,
            )

            if fail_after is None:
                _register_trace(trace, model)
                assert state["successful_registrations"] >= 1
                _assert_closed(trace, model, owner, original_fn, baseline_hooks)
                return state["successful_registrations"]

            with pytest.raises(contract_error) as exc_info:
                trace.register(model)
            _assert_contract_error(exc_info.value)
            assert state["successful_registrations"] == fail_after
            assert owner.apply_multimodal_rotary_pos_emb is original_fn
            assert _hook_count(model) == baseline_hooks
            if fail_after >= 1:
                assert state["saw_partial_registration"]
            _assert_closed(trace, model, owner, original_fn, baseline_hooks)
            return state["successful_registrations"]

    total_registrations = _run_registration_probe(fail_after=None)
    assert total_registrations >= 2
    for fail_after in range(0, total_registrations):
        _run_registration_probe(fail_after=fail_after)


def test_decoder_prefill_trace_capture_cleans_up_after_duplicate_events_and_forward_exception():
    contract_error = _contract_error_type()
    input_a, _, _, _ = _build_inputs()

    model = _FakeOuterModel(seed=14)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        model.repeat_layer0_once = True
        with pytest.raises(contract_error) as exc_info:
            _set_active(model, "A", input_a, use_cache=False)
        _assert_contract_error(exc_info.value)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)

    model = _FakeOuterModel(seed=15)
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        model.double_mrope_layer0 = True
        with pytest.raises(contract_error) as exc_info:
            _set_active(model, "A", input_a, use_cache=False)
        _assert_contract_error(exc_info.value)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)

    model = _FakeOuterModel(seed=16)
    model.raise_in_layer0 = True
    trace = _DecoderPrefillTraceCapture()
    owner, original_fn, baseline_hooks, _ = _register_trace(trace, model)
    try:
        with pytest.raises(RuntimeError, match="layer0 boom"):
            _set_active(model, "A", input_a, use_cache=False)
    finally:
        _assert_closed(trace, model, owner, original_fn, baseline_hooks)
