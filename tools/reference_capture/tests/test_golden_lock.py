from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from blake3 import blake3

from pvlc_reference.golden_lock import (
    PINNED_GOLDEN_LOCK_DIGEST,
    GoldenEntry,
    GoldenLock,
    GoldenLockError,
    load_pinned_golden_lock,
    verify_locked_bundles,
)
from pvlc_reference.trace_bundle import TraceLevel, verify_bundle


REPO_ROOT = Path(__file__).parents[3]
LOCK_PATH = REPO_ROOT / "goldens" / "golden.lock"
L0_FIXTURE = Path(__file__).parent / "fixtures" / "golden_l0"
L0_DIGEST = "blake3:faa6ceff05cd755f43bc798c9a6cb12d487142f559bbbb445ea22401c0c5bb62"


def fixture_entry(path: str = "bundles/l0") -> GoldenEntry:
    return GoldenEntry(
        case_id="ocr.synthetic.0001",
        trace_level=TraceLevel.L0,
        artifact_path=path,
        bundle_digest=L0_DIGEST,
        semantic_fingerprint=(
            "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ),
        generated_tokens=(),
        decoded_text="",
        repeat_count=2,
    )


def fixture_lock(entry: GoldenEntry | None = None) -> GoldenLock:
    return GoldenLock(
        format_version=1,
        model_revision="66317acc4c9fc17bd154591ce650735cd2855f3e",
        trace_schema_version=1,
        bundles=(entry or fixture_entry(),),
    )


def test_checked_in_golden_lock_is_canonical_and_externally_pinned() -> None:
    lock = load_pinned_golden_lock(LOCK_PATH)

    assert blake3(LOCK_PATH.read_bytes()).hexdigest() == (
        "40947f87eec2ac0f75ce671ca9226bb335adbe6254e5c1858f5c2ae6310450c9"
    )
    assert PINNED_GOLDEN_LOCK_DIGEST == (
        "40947f87eec2ac0f75ce671ca9226bb335adbe6254e5c1858f5c2ae6310450c9"
    )
    assert lock.canonical_bytes() == LOCK_PATH.read_bytes()
    assert tuple((entry.case_id, entry.trace_level) for entry in lock.bundles) == (
        ("ocr.clean_latin.0001", TraceLevel.L3),
        ("table.simple.0001", TraceLevel.L2),
    )
    assert tuple(entry.bundle_digest for entry in lock.bundles) == (
        "blake3:35572ac07da5fddee97becb46fb866f906200f433f4c5677ad20fdf36440acf9",
        "blake3:4cecc8b2abc37030920acf8860bb317084125dbbbcae5af394fccd0c8044c842",
    )
    assert tuple(entry.semantic_fingerprint for entry in lock.bundles) == (
        "blake3:632cbe2de6f47f8f84c764e3a551bf99f352d7459c98d939eb89725357f952b4",
        "blake3:91f72ffdf81070c2c9ca31628605064bdc02bf19417482cefce41d6a27aa5404",
    )
    assert lock.bundles[0].generated_tokens == (94013, 898)
    assert lock.bundles[0].decoded_text == "JUL"
    assert lock.bundles[1].generated_tokens == (101309, 93933)
    assert lock.bundles[1].decoded_text == "<fcel>m"


def test_pinned_loader_rejects_parseable_whitespace_change(tmp_path: Path) -> None:
    changed = tmp_path / "golden.lock"
    changed.write_bytes(LOCK_PATH.read_bytes() + b"\n")

    with pytest.raises(GoldenLockError) as caught:
        load_pinned_golden_lock(changed)

    assert caught.value.code == "wrong_golden_lock_digest"


@pytest.mark.oracle
def test_checked_in_lock_verifies_both_real_local_bundles() -> None:
    lock = load_pinned_golden_lock(LOCK_PATH)
    if any(not (REPO_ROOT / entry.artifact_path).is_dir() for entry in lock.bundles):
        pytest.skip("locally captured M0 goldens are not present")

    summary = verify_locked_bundles(lock, REPO_ROOT, require_all=True)

    assert summary.verified == (
        "artifacts/goldens/ocr.clean_latin.0001-l3",
        "artifacts/goldens/table.simple.0001-l2",
    )
    assert summary.missing == ()


def test_lock_to_bundle_verification_uses_external_pin_and_cross_fields(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundles" / "l0"
    shutil.copytree(L0_FIXTURE, bundle)

    summary = verify_locked_bundles(fixture_lock(), tmp_path, require_all=True)

    assert summary.verified == ("bundles/l0",)
    assert summary.missing == ()


def test_missing_bundle_can_be_reported_or_made_fatal(tmp_path: Path) -> None:
    summary = verify_locked_bundles(fixture_lock(), tmp_path, require_all=False)

    assert summary.verified == ()
    assert summary.missing == ("bundles/l0",)
    with pytest.raises(GoldenLockError) as caught:
        verify_locked_bundles(fixture_lock(), tmp_path, require_all=True)
    assert caught.value.code == "missing_bundle"


def test_coordinated_bundle_mutation_is_rejected_by_lock_pin(tmp_path: Path) -> None:
    bundle = tmp_path / "bundles" / "l0"
    shutil.copytree(L0_FIXTURE, bundle)
    changed_source = b"P3\n2 1\n255\n0 0 0 255 255 255\n"
    (bundle / "source-image.bin").write_bytes(changed_source)
    case = json.loads((bundle / "case.json").read_bytes())
    case["source_image_hash"] = f"blake3:{blake3(changed_source).hexdigest()}"
    (bundle / "case.json").write_bytes(
        json.dumps(case, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    artifacts = {}
    for artifact in sorted(bundle.iterdir()):
        if artifact.name == "hashes.json":
            continue
        payload = artifact.read_bytes()
        artifacts[artifact.name] = {
            "blake3": blake3(payload).hexdigest(),
            "size": len(payload),
        }
    hashes = json.dumps(
        {"algorithm": "blake3", "artifacts": artifacts, "format_version": 1},
        sort_keys=True,
        separators=(",", ":"),
    ).encode() + b"\n"
    (bundle / "hashes.json").write_bytes(hashes)
    rewritten = verify_bundle(bundle)
    assert rewritten.bundle_digest != L0_DIGEST

    with pytest.raises(GoldenLockError) as caught:
        verify_locked_bundles(fixture_lock(), tmp_path, require_all=True)

    assert caught.value.code == "locked_bundle_invalid"
    assert caught.value.details["artifact_path"] == "bundles/l0"


@pytest.mark.parametrize(
    ("entry", "code"),
    [
        (
            replace(fixture_entry(), case_id="ocr.wrong.0001"),
            "locked_bundle_case_mismatch",
        ),
        (
            replace(fixture_entry(), trace_level=TraceLevel.L1),
            "locked_bundle_trace_mismatch",
        ),
    ],
)
def test_verifier_rejects_cross_field_mismatch_with_valid_digest(
    tmp_path: Path, entry: GoldenEntry, code: str
) -> None:
    bundle = tmp_path / "bundles" / "l0"
    shutil.copytree(L0_FIXTURE, bundle)

    with pytest.raises(GoldenLockError) as caught:
        verify_locked_bundles(fixture_lock(entry), tmp_path, require_all=True)

    assert caught.value.code == code


@pytest.mark.parametrize(
    ("entry", "code"),
    [
        (replace(fixture_entry(), artifact_path="../escape"), "invalid_artifact_path"),
        (replace(fixture_entry(), artifact_path="/absolute"), "invalid_artifact_path"),
        (replace(fixture_entry(), repeat_count=1), "insufficient_repeat_count"),
        (replace(fixture_entry(), generated_tokens=(-1,)), "invalid_generated_tokens"),
        (
            replace(fixture_entry(), bundle_digest="blake3:" + ("A" * 64)),
            "invalid_digest",
        ),
    ],
)
def test_entry_rejects_unsafe_or_semantically_weak_pins(
    entry: GoldenEntry, code: str
) -> None:
    with pytest.raises(GoldenLockError) as caught:
        entry.validate()

    assert caught.value.code == code


def test_lock_rejects_duplicate_case_level_or_artifact_path() -> None:
    first = fixture_entry()
    duplicate_case = replace(first, artifact_path="bundles/other")
    duplicate_path = replace(
        first,
        case_id="ocr.other.0001",
        trace_level=TraceLevel.L1,
    )

    for duplicate in (duplicate_case, duplicate_path):
        with pytest.raises(GoldenLockError) as caught:
            replace(fixture_lock(), bundles=(first, duplicate)).validate()
        assert caught.value.code == "duplicate_bundle"


def test_parser_rejects_unknown_top_level_field_in_correct_scope(tmp_path: Path) -> None:
    path = tmp_path / "golden.lock"
    payload = fixture_lock().canonical_bytes().replace(
        b"trace_schema_version = 1\n",
        b"trace_schema_version = 1\nunknown_top_level = 1\n",
        1,
    )
    path.write_bytes(payload)

    with pytest.raises(GoldenLockError) as caught:
        GoldenLock.load(path)
    assert caught.value.code == "unknown_field"


def test_parser_rejects_unknown_bundle_field_in_complete_entry(tmp_path: Path) -> None:
    path = tmp_path / "golden.lock"
    payload = fixture_lock().canonical_bytes().replace(
        b'case_id = "ocr.synthetic.0001"\n',
        b'case_id = "ocr.synthetic.0001"\nunknown_entry = 1\n',
        1,
    )
    path.write_bytes(payload)

    with pytest.raises(GoldenLockError) as caught:
        GoldenLock.load(path)
    assert caught.value.code == "unknown_field"


def test_parser_rejects_unreviewed_format_version(tmp_path: Path) -> None:
    path = tmp_path / "golden.lock"
    path.write_bytes(
        fixture_lock().canonical_bytes().replace(
            b"format_version = 1", b"format_version = 2", 1
        )
    )

    with pytest.raises(GoldenLockError) as caught:
        GoldenLock.load(path)
    assert caught.value.code == "unsupported_format_version"
