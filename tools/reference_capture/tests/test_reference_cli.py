from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pvlc_reference.capture import GenerationStep, GenerationTrace
from pvlc_reference.capture_artifacts import CapturedArtifacts
from pvlc_reference.capture_artifacts import export_golden_bundle
from pvlc_reference.reference_cli import (
    ReferenceCaptureCliError,
    capture_case,
    oracle_capture_fingerprint,
    parse_trace_level,
)
from pvlc_reference.trace_bundle import TraceLevel, verify_bundle
from pvlc_reference.transformers_oracle import (
    GenerationComparison,
    OracleCaptureResult,
    ProcessorCapture,
)


pytestmark = pytest.mark.oracle
REPO_ROOT = Path(__file__).parents[3]
REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e"
LOCK_PATH = REPO_ROOT / "models" / "paddleocr-vl-1.6.lock"
SNAPSHOT = REPO_ROOT / "models" / "snapshots" / REVISION
CASE_PATH = REPO_ROOT / "cases" / "smoke" / "cases" / "ocr-clean-latin.json"
IMAGE_PATH = REPO_ROOT / "cases" / "smoke" / "assets" / "ocr-clean-latin.png"


def trace() -> GenerationTrace:
    return GenerationTrace(
        tokens=(94013, 898),
        steps=(
            GenerationStep(
                step=0,
                input_token=23,
                position_ids=(68, 68, 68),
                cache_position=332,
                rope_delta=-264,
                top_tokens=((94013, 8.0), (898, 1.0)),
                chosen_token=94013,
                kv_shapes=((1, 2, 332, 128),),
            ),
            GenerationStep(
                step=1,
                input_token=94013,
                position_ids=(69, 69, 69),
                cache_position=333,
                rope_delta=-264,
                top_tokens=((898, 7.0), (94013, 0.5)),
                chosen_token=898,
                kv_shapes=((1, 2, 333, 128),),
            ),
        ),
    )


def oracle_result(stage_offset: float = 0.0) -> OracleCaptureResult:
    processor = ProcessorCapture(
        case_id="ocr.clean_latin.0001",
        input_ids=(1, 2, 3),
        attention_mask=(1, 1, 1),
        image_grid_thw=(1, 2, 2),
        pixel_values_shape=(4, 3, 2, 2),
        placeholder_id=2,
        placeholder_count=1,
        spatial_merge_size=2,
        pixel_values_digest=(
            "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ),
        pixel_min=-1.0,
        pixel_max=1.0,
        pixel_mean=0.0,
        pixel_std=0.5,
    )
    comparison = GenerationComparison(
        processor=processor,
        generate_tokens=(94013, 898),
        manual_trace=trace(),
        decoded_text="JUL",
    )
    captured = CapturedArtifacts(
        processor_tensors={
            "processor.input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            "processor.pixel_values": torch.arange(48, dtype=torch.float32).reshape(
                4, 3, 2, 2
            ),
        },
        stage_tensors={
            "vision.final": torch.full(
                (1, 2, 4), stage_offset, dtype=torch.bfloat16
            ),
            "decoder.prefill.logits.last": torch.tensor(
                [[1.0, 2.0]], dtype=torch.bfloat16
            ),
        },
        deep_tensors={},
        token_trace=trace(),
    )
    return OracleCaptureResult(comparison=comparison, captured=captured)


def mutate_result(
    result: OracleCaptureResult, component: str
) -> OracleCaptureResult:
    captured = result.captured
    if component == "processor":
        changed = dict(captured.processor_tensors)
        changed["processor.input_ids"] = changed["processor.input_ids"] + 1
        return replace(result, captured=replace(captured, processor_tensors=changed))
    if component == "stage":
        changed = dict(captured.stage_tensors)
        changed["vision.final"] = changed["vision.final"] + 1
        return replace(result, captured=replace(captured, stage_tensors=changed))
    if component == "trace":
        original_trace = result.comparison.manual_trace
        changed_trace = replace(
            original_trace,
            steps=(
                replace(
                    original_trace.steps[0],
                    top_tokens=((94013, 9.0), (898, 1.0)),
                ),
                original_trace.steps[1],
            ),
        )
        return replace(
            result,
            comparison=replace(result.comparison, manual_trace=changed_trace),
            captured=replace(captured, token_trace=changed_trace),
        )
    if component == "decoded_text":
        return replace(
            result,
            comparison=replace(result.comparison, decoded_text="different"),
        )
    if component == "generate_tokens":
        return replace(
            result,
            comparison=replace(result.comparison, generate_tokens=(94013, 999)),
        )
    if component == "processor_capture":
        return replace(
            result,
            comparison=replace(
                result.comparison,
                processor=replace(result.comparison.processor, pixel_mean=0.25),
            ),
        )
    if component == "deep":
        return replace(
            result,
            captured=replace(
                captured,
                deep_tensors={
                    "vision.layer.00.output": torch.ones(
                        (1, 2, 4), dtype=torch.bfloat16
                    )
                },
            ),
        )
    if component == "key":
        changed = dict(captured.stage_tensors)
        changed["vision.changed"] = changed.pop("vision.final")
        return replace(result, captured=replace(captured, stage_tensors=changed))
    if component == "shape":
        changed = dict(captured.stage_tensors)
        changed["vision.final"] = changed["vision.final"].reshape(1, 1, 8)
        return replace(result, captured=replace(captured, stage_tensors=changed))
    if component == "dtype":
        changed = dict(captured.stage_tensors)
        changed["vision.final"] = changed["vision.final"].float()
        return replace(result, captured=replace(captured, stage_tensors=changed))
    if component == "inconsistent_trace":
        changed_trace = replace(
            captured.token_trace,
            steps=(
                replace(
                    captured.token_trace.steps[0],
                    top_tokens=((94013, 9.0), (898, 1.0)),
                ),
                captured.token_trace.steps[1],
            ),
        )
        return replace(result, captured=replace(captured, token_trace=changed_trace))
    raise AssertionError(component)


class FakeOracle:
    def __init__(
        self,
        results: tuple[OracleCaptureResult, ...],
        *,
        unpublished_output: Path | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.results = iter(results)
        self.calls: list[tuple[object, ...]] = []
        self.unpublished_output = unpublished_output
        self.events = events

    def capture_artifacts(
        self,
        case: object,
        image_path: Path,
        *,
        max_new_tokens: int,
        trace_level: TraceLevel,
    ) -> OracleCaptureResult:
        if self.unpublished_output is not None:
            assert not self.unpublished_output.exists()
        if self.events is not None:
            self.events.append("capture")
        self.calls.append(
            ("capture", getattr(case, "case_id"), image_path, max_new_tokens, trace_level)
        )
        return next(self.results)


class RecordingFactory:
    def __init__(self, oracle: FakeOracle) -> None:
        self.oracle = oracle
        self.calls: list[tuple[object, ...]] = []

    def __call__(
        self,
        snapshot: Path,
        model_lock: object,
        *,
        device: str,
        dtype: str,
    ) -> FakeOracle:
        self.calls.append(
            (
                snapshot,
                getattr(model_lock, "revision"),
                device,
                dtype,
            )
        )
        return self.oracle


class RecordingPublisher:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.events.append("publish")
        self.calls.append(kwargs)
        return export_golden_bundle(**kwargs)


@pytest.mark.parametrize(
    ("value", "level"),
    [
        ("metadata", TraceLevel.L0),
        ("probes", TraceLevel.L1),
        ("stage", TraceLevel.L2),
        ("deep", TraceLevel.L3),
        ("L2", TraceLevel.L2),
    ],
)
def test_cli_trace_level_names_are_stable(value: str, level: TraceLevel) -> None:
    assert parse_trace_level(value) is level


def test_capture_case_repeat_checks_before_publishing_verified_bundle(
    tmp_path: Path,
) -> None:
    output = tmp_path / "golden"
    events: list[str] = []
    oracle = FakeOracle(
        (oracle_result(), oracle_result()),
        unpublished_output=output,
        events=events,
    )
    factory = RecordingFactory(oracle)
    publisher = RecordingPublisher(events)

    result = capture_case(
        model_lock_path=LOCK_PATH,
        snapshot=SNAPSHOT,
        case_path=CASE_PATH,
        image_path=IMAGE_PATH,
        output=output,
        device="mps",
        dtype="bfloat16",
        trace_level=TraceLevel.L2,
        max_new_tokens=2,
        repeat=2,
        probe_seed=12_345,
        oracle_factory=factory,
        publisher=publisher,
    )

    assert factory.calls == [(SNAPSHOT, REVISION, "mps", "bfloat16")]
    assert oracle.calls == [
        ("capture", "ocr.clean_latin.0001", IMAGE_PATH, 2, TraceLevel.L2),
        ("capture", "ocr.clean_latin.0001", IMAGE_PATH, 2, TraceLevel.L2),
    ]
    assert events == ["capture", "capture", "publish"]
    assert len(publisher.calls) == 1
    assert result.case_id == "ocr.clean_latin.0001"
    assert result.decoded_text == "JUL"
    assert result.generated_tokens == (94013, 898)
    report = verify_bundle(output, expected_bundle_digest=result.bundle_digest)
    assert report.case.case_id == result.case_id
    assert report.provenance.device == "mps"
    assert report.provenance.dtype == "bfloat16"


@pytest.mark.parametrize(
    ("component", "level"),
    [
        ("processor", TraceLevel.L2),
        ("stage", TraceLevel.L2),
        ("trace", TraceLevel.L2),
        ("decoded_text", TraceLevel.L2),
        ("generate_tokens", TraceLevel.L2),
        ("processor_capture", TraceLevel.L2),
        ("key", TraceLevel.L2),
        ("shape", TraceLevel.L2),
        ("dtype", TraceLevel.L2),
        ("deep", TraceLevel.L3),
    ],
)
def test_repeat_mismatch_does_not_publish_partial_bundle(
    tmp_path: Path, component: str, level: TraceLevel
) -> None:
    first = oracle_result()
    output = tmp_path / "must-not-exist"
    events: list[str] = []
    oracle = FakeOracle(
        (first, mutate_result(first, component)),
        unpublished_output=output,
        events=events,
    )
    publisher = RecordingPublisher(events)

    with pytest.raises(ReferenceCaptureCliError) as caught:
        capture_case(
            model_lock_path=LOCK_PATH,
            snapshot=SNAPSHOT,
            case_path=CASE_PATH,
            image_path=IMAGE_PATH,
            output=output,
            device="mps",
            dtype="bfloat16",
            trace_level=level,
            max_new_tokens=2,
            repeat=2,
            probe_seed=12_345,
            oracle_factory=RecordingFactory(oracle),
            publisher=publisher,
        )

    assert caught.value.code == "nondeterministic_oracle_capture"
    assert caught.value.details == {"repeat": 2}
    assert not output.exists()
    assert events == ["capture", "capture"]
    assert publisher.calls == []


def test_inconsistent_trace_is_rejected_in_capture_pipeline_before_publish(
    tmp_path: Path,
) -> None:
    output = tmp_path / "inconsistent"
    events: list[str] = []
    oracle = FakeOracle(
        (mutate_result(oracle_result(), "inconsistent_trace"),),
        unpublished_output=output,
        events=events,
    )
    publisher = RecordingPublisher(events)

    with pytest.raises(ReferenceCaptureCliError) as caught:
        capture_case(
            model_lock_path=LOCK_PATH,
            snapshot=SNAPSHOT,
            case_path=CASE_PATH,
            image_path=IMAGE_PATH,
            output=output,
            device="mps",
            dtype="bfloat16",
            trace_level=TraceLevel.L2,
            max_new_tokens=2,
            repeat=1,
            probe_seed=12_345,
            oracle_factory=RecordingFactory(oracle),
            publisher=publisher,
        )

    assert caught.value.code == "inconsistent_oracle_capture"
    assert events == ["capture"]
    assert publisher.calls == []
    assert not output.exists()


@pytest.mark.parametrize(
    ("component", "level"),
    [
        ("processor", TraceLevel.L2),
        ("stage", TraceLevel.L2),
        ("trace", TraceLevel.L2),
        ("decoded_text", TraceLevel.L2),
        ("generate_tokens", TraceLevel.L2),
        ("processor_capture", TraceLevel.L2),
        ("deep", TraceLevel.L3),
        ("key", TraceLevel.L2),
        ("shape", TraceLevel.L2),
        ("dtype", TraceLevel.L2),
    ],
)
def test_full_oracle_fingerprint_is_sensitive_to_every_semantic_component(
    component: str, level: TraceLevel
) -> None:
    base = oracle_result()

    assert oracle_capture_fingerprint(base, level) != oracle_capture_fingerprint(
        mutate_result(base, component), level
    )


def test_oracle_fingerprint_rejects_disagreement_between_trace_views() -> None:
    with pytest.raises(ReferenceCaptureCliError) as caught:
        oracle_capture_fingerprint(
            mutate_result(oracle_result(), "inconsistent_trace"), TraceLevel.L2
        )

    assert caught.value.code == "inconsistent_oracle_capture"


def test_invalid_repeat_or_generation_limit_is_rejected_before_oracle_factory(
    tmp_path: Path,
) -> None:
    factory = RecordingFactory(FakeOracle((oracle_result(),)))

    for repeat, max_new_tokens in ((0, 2), (1, 0), (1, 129)):
        with pytest.raises(ReferenceCaptureCliError):
            capture_case(
                model_lock_path=LOCK_PATH,
                snapshot=SNAPSHOT,
                case_path=CASE_PATH,
                image_path=IMAGE_PATH,
                output=tmp_path / f"invalid-{repeat}-{max_new_tokens}",
                device="mps",
                dtype="bfloat16",
                trace_level=TraceLevel.L2,
                max_new_tokens=max_new_tokens,
                repeat=repeat,
                probe_seed=12_345,
                oracle_factory=factory,
            )

    assert factory.calls == []
