from __future__ import annotations

from dataclasses import replace

import pytest

from pvlc_reference.capture import (
    CaptureContractError,
    CaptureSettings,
    GenerationStep,
    GenerationTrace,
    ProcessorObservation,
    ReferenceCaptureRunner,
    assert_generation_parity,
)
from pvlc_reference.trace_bundle import CaseSpec


MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e"
SOURCE = bytes(range(12))
EXPECTED_CAPTURE_BYTES = (
    b'{"case_id":"ocr.synthetic.0001","generate_tokens":[701,702],'
    b'"manual_trace":{"steps":[{"cache_position":4,"chosen_token":701,'
    b'"input_token":1,"kv_shapes":[[1,2,5,64]],"position_ids":[4,4,4],'
    b'"rope_delta":0,"step":0,"top_tokens":[[701,8.0],[702,2.0]]},'
    b'{"cache_position":5,"chosen_token":702,"input_token":701,'
    b'"kv_shapes":[[1,2,6,64]],"position_ids":[5,5,5],"rope_delta":0,'
    b'"step":1,"top_tokens":[[702,7.0],[703,1.0]]}],"tokens":[701,702]},'
    b'"processor":{"attention_mask":[1,1,1,1],"image_grid_thw":[1,2,2],'
    b'"image_token_count":1,"input_ids":[1,101,102,2],'
    b'"pixel_values_shape":[4,3,28,28],"placeholder_count":1,'
    b'"spatial_merge_size":2},"schema_version":1,'
    b'"source_image_hash":"blake3:46771fd4c72c26de414671d8c8634b327fd12ba240e3739e011b4e64bfbb898c"}\n'
)
EXPECTED_CAPTURE_DIGEST = (
    "blake3:37c671b71fe75536d346df21c90cb6aa21a73e1dfc539cd855bf64f4b0d3e74d"
)


def case_spec() -> CaseSpec:
    return CaseSpec(
        case_id="ocr.synthetic.0001",
        task="ocr",
        prompt="OCR:",
        source_image_hash="blake3:46771fd4c72c26de414671d8c8634b327fd12ba240e3739e011b4e64bfbb898c",
        source_media_type="application/x-canonical-rgb8",
        width=2,
        height=2,
        max_new_tokens=4,
        do_sample=False,
        max_pixels=1_003_520,
    )


def settings() -> CaptureSettings:
    return CaptureSettings(
        model_id=MODEL_ID,
        revision=REVISION,
        device="cpu",
        dtype="float32",
        seed=12_345,
        deterministic_algorithms=True,
        inference_mode=True,
    )


def processor_observation() -> ProcessorObservation:
    return ProcessorObservation(
        input_ids=(1, 101, 102, 2),
        attention_mask=(1, 1, 1, 1),
        image_grid_thw=(1, 2, 2),
        pixel_values_shape=(4, 3, 28, 28),
        spatial_merge_size=2,
        image_token_count=1,
        placeholder_count=1,
    )


def generation_trace(tokens: tuple[int, ...] = (701, 702)) -> GenerationTrace:
    return GenerationTrace(
        tokens=tokens,
        steps=tuple(
            GenerationStep(
                step=index,
                input_token=1 if index == 0 else tokens[index - 1],
                position_ids=(4 + index, 4 + index, 4 + index),
                cache_position=4 + index,
                rope_delta=0,
                top_tokens=((token, 8.0 - index), (token + 1, 2.0 - index)),
                chosen_token=token,
                kv_shapes=((1, 2, 5 + index, 64),),
            )
            for index, token in enumerate(tokens)
        ),
    )


class RecordingOracle:
    def __init__(self, manual_tokens: tuple[int, ...] = (701, 702)) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.training = True
        self.manual_tokens = manual_tokens

    def load(self, capture_settings: CaptureSettings) -> None:
        self.calls.append(
            (
                "load",
                capture_settings.model_id,
                capture_settings.revision,
                capture_settings.device,
                capture_settings.dtype,
            )
        )

    def eval(self) -> None:
        self.training = False
        self.calls.append(("eval",))

    def enable_determinism(self, seed: int) -> None:
        self.calls.append(("deterministic", seed))

    def process(self, source: bytes, case: CaseSpec) -> ProcessorObservation:
        self.calls.append(("process", case.prompt, case.max_pixels, source))
        return processor_observation()

    def synchronize(self) -> None:
        self.calls.append(("synchronize",))

    def generate(
        self, processed: ProcessorObservation, *, max_new_tokens: int, do_sample: bool
    ) -> tuple[int, ...]:
        self.calls.append(("generate", max_new_tokens, do_sample))
        return (701, 702)

    def manual_generate(
        self, processed: ProcessorObservation, *, max_new_tokens: int, do_sample: bool
    ) -> GenerationTrace:
        self.calls.append(("manual_generate", max_new_tokens, do_sample))
        return generation_trace(self.manual_tokens)


def test_capture_runner_enforces_official_deterministic_oracle_sequence() -> None:
    oracle = RecordingOracle()
    runner = ReferenceCaptureRunner(oracle=oracle, settings=settings())

    result = runner.capture(case=case_spec(), source_image=SOURCE)

    assert oracle.training is False
    assert oracle.calls == [
        ("load", MODEL_ID, REVISION, "cpu", "float32"),
        ("eval",),
        ("deterministic", 12_345),
        ("process", "OCR:", 1_003_520, SOURCE),
        ("synchronize",),
        ("generate", 4, False),
        ("synchronize",),
        ("manual_generate", 4, False),
        ("synchronize",),
    ]
    assert result.processor == processor_observation()
    assert result.generate_tokens == (701, 702)
    assert result.manual_trace == generation_trace()


def test_repeat_capture_has_identical_semantic_result() -> None:
    first = ReferenceCaptureRunner(RecordingOracle(), settings()).capture(
        case_spec(), SOURCE
    )
    second = ReferenceCaptureRunner(RecordingOracle(), settings()).capture(
        case_spec(), SOURCE
    )

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.semantic_digest == second.semantic_digest


def test_capture_result_has_independently_fixed_canonical_payload_and_digest() -> None:
    result = ReferenceCaptureRunner(RecordingOracle(), settings()).capture(
        case_spec(), SOURCE
    )

    assert result.canonical_bytes() == EXPECTED_CAPTURE_BYTES
    assert result.semantic_digest == EXPECTED_CAPTURE_DIGEST


def test_each_semantic_capture_component_changes_payload_and_digest() -> None:
    result = ReferenceCaptureRunner(RecordingOracle(), settings()).capture(
        case_spec(), SOURCE
    )
    variants = (
        replace(
            result,
            processor=replace(result.processor, input_ids=(1, 101, 103, 2)),
        ),
        replace(result, generate_tokens=(701, 703)),
        replace(result, manual_trace=generation_trace((701, 703))),
    )

    for variant in variants:
        assert variant.canonical_bytes() != result.canonical_bytes()
        assert variant.semantic_digest != result.semantic_digest


def test_manual_loop_mismatch_reports_first_divergent_decode_step() -> None:
    oracle = RecordingOracle(manual_tokens=(701, 999))
    runner = ReferenceCaptureRunner(oracle=oracle, settings=settings())

    with pytest.raises(CaptureContractError) as caught:
        runner.capture(case=case_spec(), source_image=SOURCE)

    assert caught.value.code == "generation_mismatch"
    assert caught.value.details == {
        "first_divergent_step": 1,
        "generate_token": 702,
        "manual_token": 999,
    }


@pytest.mark.parametrize(
    ("generated", "manual", "step"),
    [
        ((1, 2), (1,), 1),
        ((1,), (1, 2), 1),
        ((1, 2, 3), (1, 9, 3), 1),
    ],
)
def test_generation_parity_includes_length_mismatches(
    generated: tuple[int, ...], manual: tuple[int, ...], step: int
) -> None:
    with pytest.raises(CaptureContractError) as caught:
        assert_generation_parity(generated, manual)

    assert caught.value.code == "generation_mismatch"
    assert caught.value.details["first_divergent_step"] == step


def test_capture_rejects_source_hash_mismatch_before_oracle_access() -> None:
    oracle = RecordingOracle()
    wrong_case = replace(
        case_spec(),
        source_image_hash=(
            "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ),
    )

    with pytest.raises(CaptureContractError) as caught:
        ReferenceCaptureRunner(oracle, settings()).capture(wrong_case, SOURCE)

    assert caught.value.code == "source_hash_mismatch"
    assert oracle.calls == []


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("model_id", "PaddlePaddle/PaddleOCR-VL-1.5", "wrong_model_identity"),
        ("revision", "76317acc4c9fc17bd154591ce650735cd2855f3e", "wrong_model_identity"),
        ("device", "cuda", "unsupported_oracle_device"),
        ("dtype", "float16", "unsupported_oracle_precision"),
        ("seed", True, "invalid_seed"),
        ("deterministic_algorithms", False, "nondeterministic_capture"),
        ("inference_mode", False, "nondeterministic_capture"),
    ],
)
def test_capture_settings_reject_semantic_drift(
    field: str, value: object, code: str
) -> None:
    candidate = replace(settings(), **{field: value})

    with pytest.raises(CaptureContractError) as caught:
        candidate.validate()

    assert caught.value.code == code


def test_runner_validates_settings_before_oracle_access() -> None:
    oracle = RecordingOracle()
    invalid = replace(settings(), deterministic_algorithms=False)

    with pytest.raises(CaptureContractError) as caught:
        ReferenceCaptureRunner(oracle, invalid).capture(case_spec(), SOURCE)

    assert caught.value.code == "nondeterministic_capture"
    assert oracle.calls == []


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("attention_mask", (1, 1), "processor_shape_mismatch"),
        ("image_grid_thw", (1, 0, 2), "invalid_image_grid"),
        ("pixel_values_shape", (3, 3, 28, 28), "patch_grid_mismatch"),
        ("image_token_count", 3, "placeholder_mismatch"),
        ("placeholder_count", 2, "placeholder_mismatch"),
    ],
)
def test_processor_observation_enforces_exact_multimodal_contract(
    field: str, value: object, code: str
) -> None:
    candidate = replace(processor_observation(), **{field: value})

    with pytest.raises(CaptureContractError) as caught:
        candidate.validate()

    assert caught.value.code == code


def test_generation_trace_rejects_missing_or_inconsistent_steps() -> None:
    trace = generation_trace()
    missing = replace(trace, steps=trace.steps[:1])
    wrong_choice = replace(
        trace,
        steps=(trace.steps[0], replace(trace.steps[1], chosen_token=999)),
    )

    for candidate in (missing, wrong_choice):
        with pytest.raises(CaptureContractError):
            candidate.validate()


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"step": -1}, "invalid_step_index"),
        ({"input_token": -1}, "invalid_token_id"),
        ({"position_ids": (4, 4)}, "invalid_position_ids"),
        ({"cache_position": -1}, "invalid_cache_position"),
        ({"rope_delta": True}, "invalid_rope_delta"),
        ({"top_tokens": ()}, "invalid_top_tokens"),
        ({"top_tokens": ((701, float("nan")),)}, "invalid_top_tokens"),
        ({"top_tokens": ((701, 8.0), (701, 2.0))}, "invalid_top_tokens"),
        ({"top_tokens": ((701, 2.0), (702, 8.0))}, "invalid_top_tokens"),
        ({"kv_shapes": ()}, "invalid_kv_shape"),
        ({"kv_shapes": ((1, 2, 0, 64),)}, "invalid_kv_shape"),
    ],
)
def test_generation_step_rejects_ambiguous_diagnostics(
    changes: dict[str, object], code: str
) -> None:
    step = replace(generation_trace().steps[0], **changes)

    with pytest.raises(CaptureContractError) as caught:
        step.validate()

    assert caught.value.code == code


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"step": 2}, "invalid_step_index"),
        ({"input_token": 999}, "input_token_mismatch"),
        ({"cache_position": 4}, "cache_position_mismatch"),
        ({"position_ids": (6, 5, 5)}, "position_id_mismatch"),
        ({"rope_delta": 1}, "rope_delta_mismatch"),
    ],
)
def test_generation_trace_requires_stepwise_decode_continuity(
    changes: dict[str, object], code: str
) -> None:
    trace = generation_trace()
    candidate = replace(
        trace,
        steps=(trace.steps[0], replace(trace.steps[1], **changes)),
    )

    with pytest.raises(CaptureContractError) as caught:
        candidate.validate()

    assert caught.value.code == code


class InvalidProcessorOracle(RecordingOracle):
    def process(self, source: bytes, case: CaseSpec) -> ProcessorObservation:
        observation = super().process(source, case)
        return replace(observation, attention_mask=(1, 1))


class InvalidTraceOracle(RecordingOracle):
    def manual_generate(
        self, processed: ProcessorObservation, *, max_new_tokens: int, do_sample: bool
    ) -> GenerationTrace:
        trace = super().manual_generate(
            processed, max_new_tokens=max_new_tokens, do_sample=do_sample
        )
        return replace(
            trace,
            steps=(trace.steps[0], replace(trace.steps[1], chosen_token=999)),
        )


@pytest.mark.parametrize(
    ("oracle", "code"),
    [
        (InvalidProcessorOracle(), "processor_shape_mismatch"),
        (InvalidTraceOracle(), "chosen_token_mismatch"),
    ],
)
def test_runner_rejects_invalid_oracle_outputs(
    oracle: RecordingOracle, code: str
) -> None:
    with pytest.raises(CaptureContractError) as caught:
        ReferenceCaptureRunner(oracle, settings()).capture(case_spec(), SOURCE)

    assert caught.value.code == code
