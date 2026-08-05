from __future__ import annotations

import math
import json
import struct
from pathlib import Path

import pytest
from blake3 import blake3
import torch
import safetensors.torch as safetensors_torch

from pvlc_reference.capture import GenerationStep, GenerationTrace
from pvlc_reference.capture_artifacts import (
    CaptureArtifactError,
    CapturedArtifacts,
    build_probe_bundle,
    export_golden_bundle,
    parse_probe_bundle,
    serialize_safetensors,
    serialize_token_trace,
    summarize_tensor,
)
from pvlc_reference.trace_bundle import (
    CaptureProvenance,
    CaseSpec,
    TraceLevel,
    verify_bundle,
)


SOURCE = bytes(range(12))
SOURCE_HASH = f"blake3:{blake3(SOURCE).hexdigest()}"
pytestmark = pytest.mark.oracle

EXPECTED_BUNDLE_DIGESTS = {
    TraceLevel.L0: "blake3:4b6c82ad45ec2712911ade7671be5259c885b22f82b4d60f6a1bdf9966b85d83",
    TraceLevel.L1: "blake3:24036852ac85d95e7233f8a01b618c259f1417a2669df4cd466bc059e0927a5a",
    TraceLevel.L2: "blake3:ec9eab2d64f589f7a0dbab50bfce5b937ccd0978328d5b481bd0fe64e75a3214",
    TraceLevel.L3: "blake3:083ff4fea932e503491e8e20fb99a719560c74c8cdf9a3a1edc13ba3510ec515",
}
EXPECTED_STANDALONE_PROBE_HEX = (
    "50564c43505242313930000000000000020000000800612e74656e736f720401"
    "0300000000000000030000000000000000000000000000000000104001000000"
    "0000000000000000000014400200000000000000000000000000184008007a2e"
    "74656e736f720101140000000000000007000000000000000000000000000000"
    "000000000100000000000000000000000000f03f070000000000000000000000"
    "00001c40090000000000000000000000000022400a0000000000000000000000"
    "0000244012000000000000000000000000003240130000000000000000000000"
    "00003340"
)


def case_spec() -> CaseSpec:
    return CaseSpec(
        case_id="ocr.artifact.0001",
        task="ocr",
        prompt="OCR:",
        source_image_hash=SOURCE_HASH,
        source_media_type="application/x-canonical-rgb8",
        width=2,
        height=2,
        max_new_tokens=2,
        do_sample=False,
        max_pixels=1_003_520,
    )


def provenance() -> CaptureProvenance:
    return CaptureProvenance(
        model_id="PaddlePaddle/PaddleOCR-VL-1.6",
        model_revision="66317acc4c9fc17bd154591ce650735cd2855f3e",
        model_lock_hash=(
            "blake3:c2ffa46821fdda1a7e4cd6bad63cdcdc2f775b8fed250b0679cc7d5411577d10"
        ),
        trace_schema_version=1,
        capture_tool_version="0.1.0",
        compatibility_shims=("paddleocr-vl-1.6/transformers-v5-abi@1",),
        python_version="3.12.13",
        torch_version="2.13.0",
        transformers_version="5.14.1",
        device="mps",
        dtype="bfloat16",
        deterministic_algorithms=True,
    )


def generation_trace() -> GenerationTrace:
    return GenerationTrace(
        tokens=(7, 8),
        steps=(
            GenerationStep(
                step=0,
                input_token=6,
                position_ids=(9, 9, 9),
                cache_position=12,
                rope_delta=-3,
                top_tokens=((7, 4.0), (9, 1.0)),
                chosen_token=7,
                kv_shapes=((1, 2, 12, 4),),
            ),
            GenerationStep(
                step=1,
                input_token=7,
                position_ids=(10, 10, 10),
                cache_position=13,
                rope_delta=-3,
                top_tokens=((8, 3.0), (9, 1.5)),
                chosen_token=8,
                kv_shapes=((1, 2, 13, 4),),
            ),
        ),
    )


def artifacts() -> CapturedArtifacts:
    return CapturedArtifacts(
        processor_tensors={
            "processor.input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            "processor.pixel_values": torch.arange(12, dtype=torch.float32).reshape(
                1, 3, 2, 2
            ),
        },
        stage_tensors={
            "vision.final": torch.arange(8, dtype=torch.bfloat16).reshape(1, 2, 4),
            "decoder.prefill.logits.last": torch.tensor(
                [[-1.0, 2.0, 0.5]], dtype=torch.bfloat16
            ),
        },
        deep_tensors={
            "vision.layer.00.output": torch.ones((1, 2, 4), dtype=torch.bfloat16)
        },
        token_trace=generation_trace(),
    )


def expected_tensors(level: TraceLevel) -> dict[str, torch.Tensor]:
    captured = artifacts()
    tensors = {**captured.processor_tensors, **captured.stage_tensors}
    if level is TraceLevel.L3:
        tensors.update(captured.deep_tensors)
    return tensors


def raw_tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def assert_tensor_mapping_exact(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    assert tuple(sorted(actual)) == tuple(sorted(expected))
    for semantic_id, expected_tensor in expected.items():
        assert actual[semantic_id].dtype == expected_tensor.dtype
        assert tuple(actual[semantic_id].shape) == tuple(expected_tensor.shape)
        assert raw_tensor_bytes(actual[semantic_id]) == raw_tensor_bytes(expected_tensor)


def parse_probes_independently(
    payload: bytes,
) -> tuple[int, dict[str, tuple[int, tuple[int, ...], tuple[int, ...], tuple[float, ...]]]]:
    assert payload[:8] == b"PVLCPRB1"
    offset = 8
    seed, record_count = struct.unpack_from("<QI", payload, offset)
    offset += struct.calcsize("<QI")
    records = {}
    for _ in range(record_count):
        (name_size,) = struct.unpack_from("<H", payload, offset)
        offset += 2
        semantic_id = payload[offset : offset + name_size].decode("utf-8")
        offset += name_size
        dtype_code, rank = struct.unpack_from("<BB", payload, offset)
        offset += 2
        shape = struct.unpack_from(f"<{rank}Q", payload, offset)
        offset += rank * 8
        (sample_count,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        indices = []
        values = []
        for _ in range(sample_count):
            index, value = struct.unpack_from("<Qd", payload, offset)
            offset += 16
            indices.append(index)
            values.append(value)
        records[semantic_id] = (
            dtype_code,
            tuple(shape),
            tuple(indices),
            tuple(values),
        )
    assert offset == len(payload)
    return seed, records


def test_float_tensor_summary_has_independently_computed_population_statistics() -> None:
    tensor = torch.tensor([[1.0, -2.0], [3.0, 0.0]], dtype=torch.float32)
    summary = summarize_tensor("unit.float", tensor, probe_seed=12_345)
    expected_raw = struct.pack("<4f", 1.0, -2.0, 3.0, 0.0)

    assert summary.shape == (2, 2)
    assert summary.dtype == "float32"
    assert summary.byte_order == "little"
    assert summary.layout == "row-major"
    assert summary.contiguous is True
    assert summary.minimum == -2.0
    assert summary.maximum == 3.0
    assert summary.mean == 0.5
    assert summary.std == pytest.approx(math.sqrt(3.25), abs=1e-12)
    assert summary.l1 == 6.0
    assert summary.l2 == pytest.approx(math.sqrt(14.0), abs=1e-12)
    assert summary.nan_count == 0
    assert summary.inf_count == 0
    assert summary.raw_hash == f"blake3:{blake3(expected_raw).hexdigest()}"


def test_summary_counts_nonfinite_values_but_stats_cover_only_finite_values() -> None:
    tensor = torch.tensor([float("nan"), float("inf"), -float("inf"), 2.0])

    summary = summarize_tensor("unit.nonfinite", tensor, probe_seed=9)

    assert summary.nan_count == 1
    assert summary.inf_count == 2
    assert summary.minimum == summary.maximum == summary.mean == 2.0
    assert summary.std == 0.0
    assert summary.l1 == summary.l2 == 2.0
    summary.validate()


def test_bfloat16_summary_hashes_storage_bytes_without_float_conversion() -> None:
    tensor = torch.tensor([1.0, -2.0, 0.5], dtype=torch.bfloat16)
    expected = tensor.contiguous().view(torch.uint8).numpy().tobytes()

    summary = summarize_tensor("unit.bfloat16", tensor, probe_seed=1)

    assert summary.dtype == "bfloat16"
    assert summary.raw_hash == f"blake3:{blake3(expected).hexdigest()}"


def test_probe_binary_is_canonical_decodable_and_sensitive_to_tensor_values() -> None:
    tensors = {
        "z.tensor": torch.arange(20, dtype=torch.float32),
        "a.tensor": torch.tensor([4, 5, 6], dtype=torch.int64),
    }

    first = build_probe_bundle(tensors, seed=12_345, samples_per_tensor=7)
    reordered = build_probe_bundle(
        dict(reversed(tuple(tensors.items()))), seed=12_345, samples_per_tensor=7
    )
    changed = build_probe_bundle(
        {**tensors, "z.tensor": tensors["z.tensor"] + 1},
        seed=12_345,
        samples_per_tensor=7,
    )
    changed_seed = build_probe_bundle(tensors, seed=54_321, samples_per_tensor=7)
    records = parse_probe_bundle(first)
    independent_seed, independent_records = parse_probes_independently(first)

    assert first.startswith(b"PVLCPRB1")
    assert first.hex() == EXPECTED_STANDALONE_PROBE_HEX
    assert first == reordered
    assert first != changed
    assert first != changed_seed
    assert tuple(record.semantic_id for record in records) == ("a.tensor", "z.tensor")
    assert records[0].indices == (0, 1, 2)
    assert records[0].values == (4.0, 5.0, 6.0)
    assert len(records[1].indices) == 7
    assert records[1].indices == (0, 1, 7, 9, 10, 18, 19)
    assert records[1].values == tuple(float(index) for index in records[1].indices)
    assert independent_seed == 12_345
    assert independent_records["z.tensor"][2] == (0, 1, 7, 9, 10, 18, 19)
    changed_seed_records = parse_probes_independently(changed_seed)[1]
    assert changed_seed_records["z.tensor"][2] == (0, 4, 7, 10, 14, 18, 19)


def test_safetensors_serialization_is_order_independent_and_roundtrips_exactly() -> None:
    tensors = artifacts().processor_tensors
    first = serialize_safetensors(tensors)
    second = serialize_safetensors(dict(reversed(tuple(tensors.items()))))

    assert first == second
    restored = safetensors_torch.load(first)
    assert tuple(sorted(restored)) == tuple(sorted(tensors))
    for name, expected in tensors.items():
        torch.testing.assert_close(restored[name], expected)


def test_token_trace_jsonl_is_one_canonical_record_per_decode_step() -> None:
    payload = serialize_token_trace(generation_trace())

    assert payload == (
        b'{"cache_position":12,"chosen_token":7,"input_token":6,'
        b'"kv_shapes":[[1,2,12,4]],"position_ids":[9,9,9],"rope_delta":-3,'
        b'"step":0,"top_tokens":[[7,4.0],[9,1.0]]}\n'
        b'{"cache_position":13,"chosen_token":8,"input_token":7,'
        b'"kv_shapes":[[1,2,13,4]],"position_ids":[10,10,10],"rope_delta":-3,'
        b'"step":1,"top_tokens":[[8,3.0],[9,1.5]]}\n'
    )


@pytest.mark.parametrize(
    ("level", "expected_artifacts"),
    [
        (
            TraceLevel.L0,
            {
                "case.json",
                "hashes.json",
                "manifest.json",
                "source-image.bin",
                "tensor-stats.jsonl",
            },
        ),
        (
            TraceLevel.L1,
            {
                "case.json",
                "hashes.json",
                "manifest.json",
                "probes.bin",
                "source-image.bin",
                "tensor-stats.jsonl",
            },
        ),
        (
            TraceLevel.L2,
            {
                "case.json",
                "hashes.json",
                "manifest.json",
                "probes.bin",
                "processor.safetensors",
                "source-image.bin",
                "stage-checkpoints.safetensors",
                "tensor-stats.jsonl",
                "token-trace.jsonl",
            },
        ),
        (
            TraceLevel.L3,
            {
                "case.json",
                "deep-checkpoints.safetensors",
                "hashes.json",
                "manifest.json",
                "probes.bin",
                "processor.safetensors",
                "source-image.bin",
                "stage-checkpoints.safetensors",
                "tensor-stats.jsonl",
                "token-trace.jsonl",
            },
        ),
    ],
)
def test_exported_bundle_contains_exact_artifact_set_and_verifies(
    tmp_path: Path, level: TraceLevel, expected_artifacts: set[str]
) -> None:
    target = tmp_path / level.value

    result = export_golden_bundle(
        root=target,
        case=case_spec(),
        source_image=SOURCE,
        provenance=provenance(),
        trace_level=level,
        captured=artifacts(),
        probe_seed=12_345,
    )

    assert {path.name for path in target.iterdir()} == expected_artifacts
    assert result.bundle_digest == EXPECTED_BUNDLE_DIGESTS[level]
    report = verify_bundle(
        target, expected_bundle_digest=EXPECTED_BUNDLE_DIGESTS[level]
    )
    assert report.case == case_spec()
    assert report.provenance == provenance()
    expected = expected_tensors(level)
    stats = [
        json.loads(line)
        for line in (target / "tensor-stats.jsonl").read_text().splitlines()
    ]
    assert tuple(record["semantic_id"] for record in stats) == tuple(sorted(expected))
    for record in stats:
        tensor = expected[record["semantic_id"]]
        assert record["shape"] == list(tensor.shape)
        assert record["raw_hash"] == f"blake3:{blake3(raw_tensor_bytes(tensor)).hexdigest()}"

    if level in {TraceLevel.L1, TraceLevel.L2, TraceLevel.L3}:
        seed, probes = parse_probes_independently((target / "probes.bin").read_bytes())
        assert seed == 12_345
        assert tuple(sorted(probes)) == tuple(sorted(expected))
        for semantic_id, (_, shape, indices, values) in probes.items():
            tensor = expected[semantic_id].to(torch.float64).flatten()
            assert shape == tuple(expected[semantic_id].shape)
            assert values == tuple(float(tensor[index]) for index in indices)

    captured = artifacts()
    if level in {TraceLevel.L2, TraceLevel.L3}:
        assert_tensor_mapping_exact(
            safetensors_torch.load((target / "processor.safetensors").read_bytes()),
            captured.processor_tensors,
        )
        assert_tensor_mapping_exact(
            safetensors_torch.load(
                (target / "stage-checkpoints.safetensors").read_bytes()
            ),
            captured.stage_tensors,
        )
        assert (target / "token-trace.jsonl").read_bytes() == serialize_token_trace(
            captured.token_trace
        )
    if level is TraceLevel.L3:
        assert_tensor_mapping_exact(
            safetensors_torch.load(
                (target / "deep-checkpoints.safetensors").read_bytes()
            ),
            captured.deep_tensors,
        )


def test_l2_export_is_byte_reproducible_and_excludes_unrequested_deep_tensors(
    tmp_path: Path,
) -> None:
    digests = []
    base = artifacts()
    reordered = CapturedArtifacts(
        processor_tensors=dict(reversed(tuple(base.processor_tensors.items()))),
        stage_tensors=dict(reversed(tuple(base.stage_tensors.items()))),
        deep_tensors=dict(reversed(tuple(base.deep_tensors.items()))),
        token_trace=base.token_trace,
    )
    for name, captured in (("first", base), ("second", reordered)):
        result = export_golden_bundle(
            root=tmp_path / name,
            case=case_spec(),
            source_image=SOURCE,
            provenance=provenance(),
            trace_level=TraceLevel.L2,
            captured=captured,
            probe_seed=12_345,
        )
        digests.append(result.bundle_digest)

    assert digests[0] == digests[1]
    first_files = tuple(sorted(path.name for path in (tmp_path / "first").iterdir()))
    second_files = tuple(sorted(path.name for path in (tmp_path / "second").iterdir()))
    assert first_files == second_files
    for artifact in first_files:
        assert (tmp_path / "first" / artifact).read_bytes() == (
            tmp_path / "second" / artifact
        ).read_bytes()
    assert not (tmp_path / "first" / "deep-checkpoints.safetensors").exists()


def test_l3_requires_nonempty_deep_checkpoints(tmp_path: Path) -> None:
    captured = artifacts()
    without_deep = CapturedArtifacts(
        processor_tensors=captured.processor_tensors,
        stage_tensors=captured.stage_tensors,
        deep_tensors={},
        token_trace=captured.token_trace,
    )

    with pytest.raises(CaptureArtifactError) as caught:
        export_golden_bundle(
            root=tmp_path / "deep",
            case=case_spec(),
            source_image=SOURCE,
            provenance=provenance(),
            trace_level=TraceLevel.L3,
            captured=without_deep,
            probe_seed=12_345,
        )

    assert caught.value.code == "missing_deep_checkpoints"
