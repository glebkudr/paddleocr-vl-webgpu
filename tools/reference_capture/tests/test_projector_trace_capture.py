from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pvlc_reference.capture import CaptureContractError
from pvlc_reference.transformers_oracle import _ProjectorTraceCapture


class _TakeFirstThree(torch.nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, :3]


class _NarrowSecondCall(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return values if self.calls == 1 else values[:, :-1]


class _FakeProjector(torch.nn.Module):
    def __init__(
        self,
        *,
        skip_second_gelu: bool = False,
        linear_1: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.skip_second_gelu = skip_second_gelu
        self.pre_norm = torch.nn.Identity()
        self.linear_1 = linear_1 or torch.nn.Identity()
        self.act = torch.nn.Identity()
        self.linear_2 = _TakeFirstThree()

    def forward(self, image_features: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []
        for index, image in enumerate(image_features):
            normalized = self.pre_norm(image)
            merged = normalized.reshape(-1, 8)
            hidden = self.linear_1(merged)
            if not (self.skip_second_gelu and index == 1):
                hidden = self.act(hidden)
            outputs.append(self.linear_2(hidden))
        return outputs


def test_projector_trace_concatenates_images_but_captures_only_first_top_level_call() -> None:
    projector = _FakeProjector()
    trace = _ProjectorTraceCapture()
    handles = trace.register(projector)
    first_image = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    second_image = torch.arange(16, dtype=torch.float32).reshape(8, 2) + 100
    try:
        first_output = projector([first_image, second_image])
        projector([torch.full((4, 2), 9_999.0)])
        captured = trace.finish()
    finally:
        for handle in reversed(handles):
            handle.remove()

    assert set(captured) == {
        "projector.pre_norm",
        "projector.merge",
        "projector.linear1",
        "projector.gelu",
        "projector.linear2",
    }
    assert torch.equal(
        captured["projector.pre_norm"], torch.cat([first_image, second_image])
    )
    expected_merge = torch.cat(
        [first_image.reshape(1, 8), second_image.reshape(2, 8)]
    )
    assert torch.equal(captured["projector.merge"], expected_merge)
    assert torch.equal(captured["projector.linear1"], expected_merge)
    assert torch.equal(captured["projector.gelu"], expected_merge)
    assert torch.equal(captured["projector.linear2"], torch.cat(first_output))
    assert all(torch.count_nonzero(tensor == 9_999).item() == 0 for tensor in captured.values())


def test_projector_trace_fails_closed_on_incomplete_or_inconsistent_image_calls() -> None:
    with pytest.raises(CaptureContractError) as incomplete:
        _ProjectorTraceCapture().finish()
    assert incomplete.value.code == "incomplete_deep_trace"

    projector = _FakeProjector()
    trace = _ProjectorTraceCapture()
    handles = trace.register(projector)
    try:
        with pytest.raises(CaptureContractError) as captured:
            projector(
                [
                    torch.zeros((4, 2), dtype=torch.float32),
                    torch.zeros((4, 3), dtype=torch.float32),
                ]
            )
        assert captured.value.code == "invalid_trace_shape"
    finally:
        for handle in reversed(handles):
            handle.remove()

    projector = _FakeProjector(skip_second_gelu=True)
    trace = _ProjectorTraceCapture()
    handles = trace.register(projector)
    try:
        projector([torch.zeros((4, 2)), torch.zeros((8, 2))])
        with pytest.raises(CaptureContractError) as incomplete_image:
            trace.finish()
        assert incomplete_image.value.code == "incomplete_deep_trace"
    finally:
        for handle in reversed(handles):
            handle.remove()

    projector = _FakeProjector(linear_1=_NarrowSecondCall())
    trace = _ProjectorTraceCapture()
    handles = trace.register(projector)
    try:
        with pytest.raises(CaptureContractError) as late_shape:
            projector([torch.zeros((4, 2)), torch.zeros((8, 2))])
        assert late_shape.value.code == "invalid_trace_shape"
    finally:
        for handle in reversed(handles):
            handle.remove()


@pytest.mark.parametrize(
    ("second", "error_code"),
    [
        (torch.zeros((4, 2), dtype=torch.float64), "invalid_trace_dtype"),
        (torch.zeros((4, 2), device="meta"), "invalid_trace_device"),
    ],
)
def test_projector_trace_rejects_cross_image_dtype_and_device_promotion(
    second: torch.Tensor, error_code: str
) -> None:
    projector = _FakeProjector()
    trace = _ProjectorTraceCapture()
    handles = trace.register(projector)
    try:
        with pytest.raises(CaptureContractError) as captured:
            projector([torch.zeros((4, 2), dtype=torch.float32), second])
        assert captured.value.code == error_code
    finally:
        for handle in reversed(handles):
            handle.remove()
