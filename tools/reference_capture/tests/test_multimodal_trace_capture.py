from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pvlc_reference.capture import CaptureContractError
from pvlc_reference.transformers_oracle import _MultimodalTraceCapture


IMAGE_TOKEN_ID = 7


class _FakeDecoder(torch.nn.Module):
    def __init__(self, embedding_table: torch.Tensor) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding.from_pretrained(
            embedding_table, freeze=True
        )

    def forward(self, **keyword_arguments: torch.Tensor) -> torch.Tensor | None:
        return keyword_arguments.get("inputs_embeds")


class _FakeOuterModel(torch.nn.Module):
    def __init__(self, embedding_table: torch.Tensor) -> None:
        super().__init__()
        self.config = SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.model = _FakeDecoder(embedding_table)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        image_embeddings: torch.Tensor,
        position_ids: torch.Tensor,
        rope_deltas: torch.Tensor,
        omit_decoder_key: str | None = None,
        omit_rope_deltas: bool = False,
        extra_embedding_call: bool = False,
    ) -> SimpleNamespace:
        embedding = self.model.embed_tokens(input_ids)
        if extra_embedding_call:
            self.model.embed_tokens(input_ids)
        assembled = embedding.clone()
        assembled[input_ids == self.config.image_token_id] = image_embeddings
        decoder_kwargs = {
            "inputs_embeds": assembled,
            "position_ids": position_ids,
        }
        if omit_decoder_key is not None:
            decoder_kwargs.pop(omit_decoder_key)
        hidden = self.model(**decoder_kwargs)
        output = SimpleNamespace(last_hidden_state=hidden)
        if not omit_rope_deltas:
            output.rope_deltas = rope_deltas
        return output


def _hook_count(module: torch.nn.Module) -> int:
    return sum(
        len(child._forward_hooks) + len(child._forward_pre_hooks)
        for child in module.modules()
    )


def _remove_handles(model: torch.nn.Module, handles: list[object]) -> None:
    while handles:
        handles.pop().remove()
    assert _hook_count(model) == 0


def _valid_forward_arguments() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(
            [[1, IMAGE_TOKEN_ID, 2, IMAGE_TOKEN_ID], [IMAGE_TOKEN_ID, 3, 4, 5]],
            dtype=torch.int64,
        ),
        "image_embeddings": torch.tensor(
            [[101.0, 102.0, 103.0], [201.0, 202.0, 203.0], [301.0, 302.0, 303.0]]
        ),
        "position_ids": torch.tensor(
            [
                [[0, 1, 2, 3], [4, 5, 6, 7]],
                [[10, 11, 12, 13], [14, 15, 16, 17]],
                [[20, 21, 22, 23], [24, 25, 26, 27]],
            ],
            dtype=torch.int64,
        ),
        "rope_deltas": torch.tensor([[-2], [-1]], dtype=torch.int64),
    }


def test_multimodal_trace_captures_exact_first_completed_outer_forward() -> None:
    embedding_table = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    model = _FakeOuterModel(embedding_table)
    arguments = _valid_forward_arguments()
    trace = _MultimodalTraceCapture()
    handles = trace.register(model)
    try:
        model(**arguments)
        model(
            input_ids=torch.full((2, 4), IMAGE_TOKEN_ID, dtype=torch.int64),
            image_embeddings=torch.full((8, 3), 9_999.0),
            position_ids=torch.full((3, 2, 4), 9_999, dtype=torch.int64),
            rope_deltas=torch.full((2, 1), 9_999, dtype=torch.int64),
        )
        captured = trace.finish()
    finally:
        _remove_handles(model, handles)

    input_ids = arguments["input_ids"]
    expected_embedding = embedding_table[input_ids]
    expected_assembled = expected_embedding.clone()
    expected_assembled[input_ids == IMAGE_TOKEN_ID] = arguments["image_embeddings"]
    expected_indices = torch.tensor([[0, 1], [0, 3], [1, 0]], dtype=torch.int64)

    assert set(captured) == {
        "decoder.embedding",
        "multimodal.image_token_indices",
        "multimodal.inputs_embeds",
        "decoder.mrope.index",
        "decoder.mrope.delta",
    }
    assert {
        name: tuple(tensor.shape) for name, tensor in captured.items()
    } == {
        "decoder.embedding": (2, 4, 3),
        "multimodal.image_token_indices": (3, 2),
        "multimodal.inputs_embeds": (2, 4, 3),
        "decoder.mrope.index": (3, 2, 4),
        "decoder.mrope.delta": (2, 1),
    }
    assert captured["decoder.embedding"].dtype == torch.float32
    assert captured["multimodal.inputs_embeds"].dtype == torch.float32
    assert captured["multimodal.image_token_indices"].dtype == torch.int64
    assert captured["decoder.mrope.index"].dtype == torch.int64
    assert captured["decoder.mrope.delta"].dtype == torch.int64
    assert torch.equal(captured["decoder.embedding"], expected_embedding)
    assert torch.equal(captured["multimodal.image_token_indices"], expected_indices)
    assert torch.equal(captured["multimodal.inputs_embeds"], expected_assembled)
    assert not torch.equal(
        captured["decoder.embedding"], captured["multimodal.inputs_embeds"]
    )
    assert torch.equal(captured["decoder.mrope.index"], arguments["position_ids"])
    assert torch.equal(captured["decoder.mrope.delta"], arguments["rope_deltas"])


def test_multimodal_trace_fails_closed_and_removes_all_handles() -> None:
    with pytest.raises(CaptureContractError):
        _MultimodalTraceCapture().finish()

    embedding_table = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    model = _FakeOuterModel(embedding_table)
    trace = _MultimodalTraceCapture()
    handles = trace.register(model)
    try:
        with pytest.raises(CaptureContractError):
            trace.register(model)
    finally:
        _remove_handles(model, handles)

    invalid_arguments: list[dict[str, object]] = [
        {"omit_decoder_key": "inputs_embeds"},
        {"omit_decoder_key": "position_ids"},
        {"omit_rope_deltas": True},
        {
            "position_ids": torch.zeros((2, 2, 4), dtype=torch.int64),
        },
        {
            "position_ids": torch.zeros((3, 2, 3), dtype=torch.int64),
        },
        {
            "rope_deltas": torch.zeros((1, 1), dtype=torch.int64),
        },
        {
            "input_ids": torch.tensor([1, IMAGE_TOKEN_ID, 2], dtype=torch.int64),
            "image_embeddings": torch.zeros((1, 3)),
            "position_ids": torch.zeros((3, 1, 3), dtype=torch.int64),
            "rope_deltas": torch.zeros((1, 1), dtype=torch.int64),
        },
        {"extra_embedding_call": True},
    ]
    for overrides in invalid_arguments:
        model = _FakeOuterModel(embedding_table)
        trace = _MultimodalTraceCapture()
        handles = trace.register(model)
        arguments: dict[str, object] = _valid_forward_arguments()
        arguments.update(overrides)
        try:
            with pytest.raises(CaptureContractError):
                model(**arguments)
                trace.finish()
        finally:
            _remove_handles(model, handles)
