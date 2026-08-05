from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest
from blake3 import blake3

from pvlc_reference.trace_bundle import (
    BundleFormatError,
    BundleIntegrityError,
    CaptureProvenance,
    CaseSpec,
    GoldenBundleBuilder,
    TensorSummary,
    TraceLevel,
    verify_bundle,
)


FIXTURE = Path(__file__).parent / "fixtures" / "golden_l0"
FIXTURE_BUNDLE_DIGEST = "blake3:faa6ceff05cd755f43bc798c9a6cb12d487142f559bbbb445ea22401c0c5bb62"
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e"
MODEL_LOCK_HASH = "blake3:c2ffa46821fdda1a7e4cd6bad63cdcdc2f775b8fed250b0679cc7d5411577d10"
SOURCE_BYTES = (FIXTURE / "source-image.bin").read_bytes()


def digest(data: bytes) -> str:
    return f"blake3:{blake3(data).hexdigest()}"


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"


def valid_case() -> CaseSpec:
    return CaseSpec(
        case_id="ocr.synthetic.0001",
        task="ocr",
        prompt="OCR:",
        source_image_hash=digest(SOURCE_BYTES),
        source_media_type="image/x-portable-pixmap",
        width=2,
        height=1,
        max_new_tokens=32,
        do_sample=False,
        max_pixels=1_003_520,
    )


def valid_provenance() -> CaptureProvenance:
    return CaptureProvenance(
        model_id=MODEL_ID,
        model_revision=REVISION,
        model_lock_hash=MODEL_LOCK_HASH,
        trace_schema_version=1,
        capture_tool_version="0.1.0",
        compatibility_shims=("paddleocr-vl-1.6/transformers-v5-abi@1",),
        python_version="3.12.13",
        torch_version="2.13.0",
        transformers_version="5.14.1",
        device="cpu",
        dtype="float32",
        deterministic_algorithms=True,
    )


def valid_summary() -> TensorSummary:
    return TensorSummary(
        semantic_id="vision.layer.00.output",
        shape=(1, 4, 8),
        dtype="float32",
        byte_order="little",
        layout="row-major",
        contiguous=True,
        minimum=-1.0,
        maximum=1.0,
        mean=0.0,
        std=0.5,
        l1=8.0,
        l2=2.0,
        nan_count=0,
        inf_count=0,
        raw_hash="blake3:209684608a8f8e329bb4f4c2d1a6d04bf840656b83f9dc1f0cd2864d845bc54c",
        probe_seed=12_345,
    )


def build_minimal_bundle(root: Path) -> str:
    builder = GoldenBundleBuilder(
        root=root,
        case=valid_case(),
        trace_level=TraceLevel.L0,
        provenance=valid_provenance(),
    )
    builder.add_bytes("source-image.bin", SOURCE_BYTES)
    builder.add_tensor_summaries([valid_summary()])
    return builder.finish().bundle_digest


def copy_fixture(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    shutil.copytree(FIXTURE, bundle)
    return bundle


def rewrite_hashes_without_production_code(bundle: Path) -> str:
    artifacts = {}
    for path in sorted(bundle.iterdir()):
        if path.name == "hashes.json":
            continue
        payload = path.read_bytes()
        artifacts[path.name] = {
            "blake3": blake3(payload).hexdigest(),
            "size": len(payload),
        }
    payload = canonical_json(
        {"algorithm": "blake3", "artifacts": artifacts, "format_version": 1}
    )
    (bundle / "hashes.json").write_bytes(payload)
    return digest(payload)


def test_independently_authored_bundle_fixture_is_valid_and_externally_pinned() -> None:
    report = verify_bundle(FIXTURE, expected_bundle_digest=FIXTURE_BUNDLE_DIGEST)

    assert report.bundle_digest == FIXTURE_BUNDLE_DIGEST
    assert report.case.case_id == "ocr.synthetic.0001"
    assert report.provenance == valid_provenance()
    assert report.verified_artifacts == (
        "case.json",
        "manifest.json",
        "source-image.bin",
        "tensor-stats.jsonl",
    )


def test_builder_matches_independent_fixture_byte_for_byte(tmp_path: Path) -> None:
    bundle = tmp_path / "generated"

    bundle_digest = build_minimal_bundle(bundle)

    assert bundle_digest == FIXTURE_BUNDLE_DIGEST
    assert [path.name for path in sorted(bundle.iterdir())] == [
        path.name for path in sorted(FIXTURE.iterdir())
    ]
    for expected in sorted(FIXTURE.iterdir()):
        assert (bundle / expected.name).read_bytes() == expected.read_bytes()


@pytest.mark.parametrize(
    "artifact",
    [
        "case.json",
        "manifest.json",
        "source-image.bin",
        "tensor-stats.jsonl",
        "hashes.json",
    ],
)
def test_external_bundle_pin_detects_mutation_of_every_control_or_data_file(
    tmp_path: Path, artifact: str
) -> None:
    bundle = copy_fixture(tmp_path)
    with (bundle / artifact).open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(BundleIntegrityError):
        verify_bundle(bundle, expected_bundle_digest=FIXTURE_BUNDLE_DIGEST)


def test_bundle_verification_classifies_missing_and_unexpected_artifacts(
    tmp_path: Path,
) -> None:
    bundle = copy_fixture(tmp_path)
    (bundle / "tensor-stats.jsonl").unlink()
    (bundle / "unlocked.bin").write_bytes(b"not declared")

    with pytest.raises(BundleIntegrityError) as caught:
        verify_bundle(bundle)

    assert caught.value.missing == ("tensor-stats.jsonl",)
    assert caught.value.unexpected == ("unlocked.bin",)


def test_source_hash_must_match_actual_source_even_if_internal_hashes_are_rewritten(
    tmp_path: Path,
) -> None:
    bundle = copy_fixture(tmp_path)
    case = json.loads((bundle / "case.json").read_bytes())
    case["source_image_hash"] = digest(b"some-other-image")
    (bundle / "case.json").write_bytes(canonical_json(case))
    rewrite_hashes_without_production_code(bundle)

    with pytest.raises(BundleIntegrityError) as caught:
        verify_bundle(bundle)

    assert caught.value.code == "source_hash_mismatch"


def test_external_pin_detects_coordinated_artifact_and_hash_manifest_rewrite(
    tmp_path: Path,
) -> None:
    bundle = copy_fixture(tmp_path)
    (bundle / "source-image.bin").write_bytes(b"P3\n1 1\n255\n0 0 0\n")
    rewrite_hashes_without_production_code(bundle)

    with pytest.raises(BundleIntegrityError) as caught:
        verify_bundle(bundle, expected_bundle_digest=FIXTURE_BUNDLE_DIGEST)

    assert caught.value.code == "bundle_digest_mismatch"


@pytest.mark.parametrize(
    "path",
    [
        "manifest.json",
        "hashes.json",
        "../outside.bin",
        "/tmp/outside.bin",
        "a/../outside.bin",
        "a//b.bin",
        r"a\b.bin",
        r"C:\outside.bin",
    ],
)
def test_bundle_builder_rejects_reserved_noncanonical_or_escaping_paths(
    tmp_path: Path, path: str
) -> None:
    builder = GoldenBundleBuilder(
        root=tmp_path / "bundle",
        case=valid_case(),
        trace_level=TraceLevel.L0,
        provenance=valid_provenance(),
    )

    with pytest.raises(BundleFormatError) as caught:
        builder.add_bytes(path, b"unsafe")

    assert caught.value.code in {"reserved_path", "invalid_path"}


def test_bundle_builder_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(BundleFormatError) as caught:
        GoldenBundleBuilder(
            root=linked / "bundle",
            case=valid_case(),
            trace_level=TraceLevel.L0,
            provenance=valid_provenance(),
        )

    assert caught.value.code == "unsafe_output_path"
    assert not (outside / "bundle").exists()


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("model_id", "PaddlePaddle/PaddleOCR-VL-1.5", "wrong_model_identity"),
        ("model_revision", "76317acc4c9fc17bd154591ce650735cd2855f3e", "wrong_model_identity"),
        ("model_lock_hash", "sha256:" + "a" * 64, "invalid_digest"),
        ("trace_schema_version", 2, "unsupported_trace_schema"),
        ("deterministic_algorithms", False, "nondeterministic_capture"),
        ("device", "", "invalid_provenance"),
        ("dtype", "int4", "invalid_provenance"),
    ],
)
def test_provenance_enforces_exact_model_schema_and_deterministic_environment(
    field: str, value: object, code: str
) -> None:
    values = valid_provenance().to_dict()
    values[field] = value

    with pytest.raises(BundleFormatError) as caught:
        CaptureProvenance.from_dict(values)

    assert caught.value.code == code


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_provenance_rejects_missing_or_unknown_fields(mutation: str) -> None:
    values = valid_provenance().to_dict()
    if mutation == "missing":
        values.pop("torch_version")
    else:
        values["unreviewed_backend"] = "yes"

    with pytest.raises(BundleFormatError) as caught:
        CaptureProvenance.from_dict(values)

    assert caught.value.code in {"missing_field", "unknown_field"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "contains spaces"),
        ("task", "unknown"),
        ("prompt", "Table Recognition:"),
        ("source_image_hash", "sha256:" + "a" * 64),
        ("source_media_type", "application/octet-stream"),
        ("width", 0),
        ("height", True),
        ("max_new_tokens", 0),
        ("do_sample", True),
        ("max_pixels", -1),
    ],
)
def test_deterministic_capture_rejects_invalid_case_contract(
    field: str, value: object
) -> None:
    values = valid_case().to_dict()
    values[field] = value

    with pytest.raises(BundleFormatError):
        CaseSpec.from_dict(values)


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_case_contract_rejects_missing_or_unknown_fields(mutation: str) -> None:
    values = valid_case().to_dict()
    if mutation == "missing":
        values.pop("prompt")
    else:
        values["processor_override"] = True

    with pytest.raises(BundleFormatError) as caught:
        CaseSpec.from_dict(values)

    assert caught.value.code in {"missing_field", "unknown_field"}


@pytest.mark.parametrize(
    ("task", "prompt", "max_pixels"),
    [
        ("ocr", "OCR:", 1_003_520),
        ("table", "Table Recognition:", 1_003_520),
        ("formula", "Formula Recognition:", 1_003_520),
        ("chart", "Chart Recognition:", 1_003_520),
        ("spotting", "Spotting:", 1_605_632),
        ("seal", "Seal Recognition:", 1_003_520),
    ],
)
def test_case_contract_pins_official_task_prompt_and_pixel_budget(
    task: str, prompt: str, max_pixels: int
) -> None:
    values = valid_case().to_dict()
    values.update(task=task, prompt=prompt, max_pixels=max_pixels)

    case = CaseSpec.from_dict(values)

    assert case.task == task
    assert case.prompt == prompt
    assert case.max_pixels == max_pixels


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("semantic_id", "vision layer 00"),
        ("shape", [1, -1, 8]),
        ("shape", [1, True, 8]),
        ("dtype", "object"),
        ("byte_order", "native"),
        ("layout", "unknown"),
        ("contiguous", 1),
        ("min", 2.0),
        ("mean", float("nan")),
        ("std", -0.1),
        ("l1", -1.0),
        ("l2", float("inf")),
        ("nan_count", -1),
        ("inf_count", True),
        ("raw_hash", "blake3:short"),
        ("probe_seed", -1),
        ("probe_seed", 1.5),
    ],
)
def test_tensor_summary_rejects_ambiguous_or_inconsistent_metadata(
    field: str, value: object
) -> None:
    values = valid_summary().to_dict()
    values[field] = value
    if field == "min":
        values["max"] = 1.0

    with pytest.raises(BundleFormatError):
        TensorSummary.from_dict(values)


def test_tensor_summary_rejects_missing_unknown_and_nonfinite_statistics() -> None:
    for mutation in ("missing", "unknown"):
        values = valid_summary().to_dict()
        if mutation == "missing":
            values.pop("layout")
        else:
            values["stride"] = [32, 8, 1]
        with pytest.raises(BundleFormatError):
            TensorSummary.from_dict(values)

    for field in ("min", "max", "mean", "std", "l1", "l2"):
        values = valid_summary().to_dict()
        values[field] = math.nan
        with pytest.raises(BundleFormatError):
            TensorSummary.from_dict(values)


def test_duplicate_semantic_ids_are_rejected(tmp_path: Path) -> None:
    builder = GoldenBundleBuilder(
        root=tmp_path / "bundle",
        case=valid_case(),
        trace_level=TraceLevel.L0,
        provenance=valid_provenance(),
    )
    builder.add_bytes("source-image.bin", SOURCE_BYTES)

    with pytest.raises(BundleFormatError) as caught:
        builder.add_tensor_summaries([valid_summary(), valid_summary()])

    assert caught.value.code == "duplicate_semantic_id"


@pytest.mark.parametrize(
    ("level", "missing"),
    [
        (TraceLevel.L1, {"probes.bin"}),
        (
            TraceLevel.L2,
            {
                "probes.bin",
                "processor.safetensors",
                "stage-checkpoints.safetensors",
                "token-trace.jsonl",
            },
        ),
        (
            TraceLevel.L3,
            {
                "probes.bin",
                "processor.safetensors",
                "stage-checkpoints.safetensors",
                "deep-checkpoints.safetensors",
                "token-trace.jsonl",
            },
        ),
    ],
)
def test_each_trace_level_has_an_explicit_required_artifact_contract(
    tmp_path: Path, level: TraceLevel, missing: set[str]
) -> None:
    builder = GoldenBundleBuilder(
        root=tmp_path / level.value,
        case=valid_case(),
        trace_level=level,
        provenance=valid_provenance(),
    )
    builder.add_bytes("source-image.bin", SOURCE_BYTES)
    builder.add_tensor_summaries([valid_summary()])

    with pytest.raises(BundleFormatError) as caught:
        builder.finish()

    assert caught.value.code == "incomplete_bundle"
    assert set(caught.value.details["missing"]) == missing


def test_builder_lifecycle_prevents_duplicate_or_post_finish_mutation(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    builder = GoldenBundleBuilder(
        root=bundle,
        case=valid_case(),
        trace_level=TraceLevel.L0,
        provenance=valid_provenance(),
    )
    builder.add_bytes("source-image.bin", SOURCE_BYTES)
    with pytest.raises(BundleFormatError) as duplicate:
        builder.add_bytes("source-image.bin", SOURCE_BYTES)
    assert duplicate.value.code == "duplicate_artifact"

    builder.add_tensor_summaries([valid_summary()])
    builder.finish()
    with pytest.raises(BundleFormatError) as second_finish:
        builder.finish()
    with pytest.raises(BundleFormatError) as add_after_finish:
        builder.add_bytes("late.bin", b"late")
    assert second_finish.value.code == "builder_finished"
    assert add_after_finish.value.code == "builder_finished"


def test_canonical_json_fixture_has_sorted_keys_compact_separators_and_one_newline() -> None:
    expected_case = canonical_json(valid_case().to_dict())
    expected_tensor = canonical_json(valid_summary().to_dict())

    assert (FIXTURE / "case.json").read_bytes() == expected_case
    assert (FIXTURE / "tensor-stats.jsonl").read_bytes() == expected_tensor
    assert expected_case.endswith(b"\n") and not expected_case.endswith(b"\n\n")
    assert b'": "' not in expected_case
