from __future__ import annotations

import os
from pathlib import Path

import pytest
from blake3 import blake3

from pvlc_reference.model_lock import (
    IntegrityError,
    LockFormatError,
    ModelFile,
    ModelLock,
    load_pinned_paddleocr_vl_16_lock,
    verify_model_directory,
)


REPO_ROOT = Path(__file__).parents[3]
PINNED_LOCK = REPO_ROOT / "models" / "paddleocr-vl-1.6.lock"
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e"
PINNED_LOCK_DIGEST = "c2ffa46821fdda1a7e4cd6bad63cdcdc2f775b8fed250b0679cc7d5411577d10"

# Independently transcribed from the Hugging Face tree API at REVISION and
# verified against the downloaded files. This is intentionally not imported
# from production code: it is the Gate 1 oracle for the checked-in lock.
EXPECTED_UPSTREAM_FILES = {
    ".gitattributes": (1570, "31f6f58a80e0bd3c74f070c66897db328f87a209ced487b130fd59c05904c193"),
    "LICENSE": (11376, "31ebad5a844da183c3264de01d35c65d7232cee1a9be9ee5e057e50d8d123f16"),
    "README.md": (21466, "7c34a83b8b2bf1d399e728ee76af4d40f0c249dbadd678137603a6e74e1ad199"),
    "added_tokens.json": (25381, "d2f2522dfd396c5866f99c4cecd88c0925ff2b2841dc6cc97cf998c563cb8863"),
    "chat_template.jinja": (1474, "6a19b0424bafd095b24bf997940aae59dea3b36659bc25cec26b051c0d2e919b"),
    "config.json": (2059, "bd249ac20e7b0458fe9d8c89662e3bd6ad0125864226bce912e0b62739b9a0f5"),
    "configuration_paddleocr_vl.py": (8104, "d0f984a9fad14d94bcfaa5ad9a89df94ce48a6ee1eb2d08977a4d88262e328f5"),
    "generation_config.json": (133, "78c47d8168d89991b41518b1ec92efdea064e3292967312a8526ec299d4f4952"),
    "image_processing_paddleocr_vl.py": (25032, "a96d24e2bf35391c3130232e7d08667f805029c34d29d291a239e6b70b02fc54"),
    "inference.yml": (43, "9aba04ed9d3a0385aed79e2c9dcddb84e2eb7acba4a1165665351a2a4b493a6d"),
    "model.safetensors": (1917255968, "4dc3ab13685a0c0a701f77f9c5ebafdc0004074247e00bde6b7e7b04279a41fc"),
    "modeling_paddleocr_vl.py": (103889, "d3b366856f054c2884640c0a80230c77d2220bb998a2c07d3a4ade453803d893"),
    "preprocessor_config.json": (641, "06a17b64a56e696acc447ca8002286dde7cc2900f57378e478178c39927cf70e"),
    "processing_paddleocr_vl.py": (12253, "7a4a80b9a7576c9354ec1edddfdfe9b90d6a407656efd4638d571d38f90b6804"),
    "processor_config.json": (137, "58e67e1528ff5730ec83d9d356ff25e22169a6f11eeaa6896b310ba1b9c73590"),
    "special_tokens_map.json": (1151, "700353952faf58cb96f7e27eaddc40f033aff8b3e371071651ed8de7f22ceac3"),
    "tokenizer.json": (11189060, "664e6c2425fd92e710a67a919753493657005ddcd1cb839737b6678db3edf3c3"),
    "tokenizer.model": (1614363, "b2ab4f3c0e402033fcba3884ff694d5e8584c3c1078ee6c6cd06aeac922fae8d"),
    "tokenizer_config.json": (186947, "c6f2522b04bdbc341bddc187edf259b7ea063c3546fe71e8eb32326da1d09cc6"),
}


def digest(data: bytes) -> str:
    return f"blake3:{blake3(data).hexdigest()}"


def handwritten_lock(files: dict[str, bytes]) -> bytes:
    lines = [
        "format_version = 1",
        f'model_id = "{MODEL_ID}"',
        f'revision = "{REVISION}"',
        "compiler_model_abi = 1",
        "",
        "[files]",
    ]
    for name, data in sorted(files.items()):
        lines.append(
            f'"{name}" = {{ blake3 = "{blake3(data).hexdigest()}", size = {len(data)} }}'
        )
    return ("\n".join(lines) + "\n").encode()


def load_test_lock(tmp_path: Path, files: dict[str, bytes]) -> ModelLock:
    lock_path = tmp_path / "model.lock"
    lock_path.write_bytes(handwritten_lock(files))
    return ModelLock.load(lock_path)


def test_checked_in_lock_is_the_complete_exact_upstream_snapshot() -> None:
    assert blake3(PINNED_LOCK.read_bytes()).hexdigest() == PINNED_LOCK_DIGEST

    lock = load_pinned_paddleocr_vl_16_lock(PINNED_LOCK)
    actual = {
        entry.path: (entry.size, entry.digest.removeprefix("blake3:"))
        for entry in lock.files
    }

    assert lock.model_id == MODEL_ID
    assert lock.revision == REVISION
    assert lock.compiler_model_abi == 1
    assert len(actual) == 19
    assert actual == EXPECTED_UPSTREAM_FILES


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (MODEL_ID, "PaddlePaddle/PaddleOCR-VL-1.5"),
        (REVISION, "76317acc4c9fc17bd154591ce650735cd2855f3e"),
    ],
)
def test_specialized_capture_boundary_rejects_another_valid_identity(
    tmp_path: Path, old: str, new: str
) -> None:
    changed = PINNED_LOCK.read_text().replace(old, new, 1)
    candidate = tmp_path / "candidate.lock"
    candidate.write_text(changed)

    with pytest.raises(LockFormatError) as caught:
        load_pinned_paddleocr_vl_16_lock(candidate)

    assert caught.value.code == "wrong_model_identity"


def test_specialized_capture_boundary_rejects_a_subset_inventory(tmp_path: Path) -> None:
    changed = "\n".join(
        line
        for line in PINNED_LOCK.read_text().splitlines()
        if not line.startswith('"README.md"')
    ) + "\n"
    candidate = tmp_path / "candidate.lock"
    candidate.write_text(changed)

    with pytest.raises(LockFormatError) as caught:
        load_pinned_paddleocr_vl_16_lock(candidate)

    assert caught.value.code == "wrong_model_inventory"


def test_valid_snapshot_is_accepted_with_reproducible_provenance(tmp_path: Path) -> None:
    files = {
        "config.json": b'{"model_type":"paddleocr_vl"}\n',
        "tokenizer/tokenizer.json": b'{"version":"1.0"}\n',
        "model.safetensors": b"small-test-weight-shard",
    }
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name, data in files.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    lock = load_test_lock(tmp_path, files)

    first = verify_model_directory(lock, snapshot)
    second = verify_model_directory(lock, snapshot)

    assert first.model_id == MODEL_ID
    assert first.revision == REVISION
    assert first.lock_digest == digest(handwritten_lock(files))
    assert first.snapshot_digest == second.snapshot_digest
    assert first.verified_files == tuple(sorted(files))


def test_integrity_failure_classifies_all_failure_kinds_in_stable_order(
    tmp_path: Path,
) -> None:
    locked_files = {
        "a-config.json": b"expected-config",
        "m-tokenizer.json": b"expected-tokenizer",
        "z-model.safetensors": b"expected-weights",
    }
    lock = load_test_lock(tmp_path, locked_files)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "a-config.json").write_bytes(b"tampered-config")
    (snapshot / "z-model.safetensors").write_bytes(locked_files["z-model.safetensors"])
    (snapshot / "z-unlocked-code.py").write_text("raise SystemExit\n")
    (snapshot / "a-unlocked-code.py").write_text("raise SystemExit\n")

    with pytest.raises(IntegrityError) as caught:
        verify_model_directory(lock, snapshot)

    assert caught.value.report.changed == ("a-config.json",)
    assert caught.value.report.missing == ("m-tokenizer.json",)
    assert caught.value.report.unexpected == (
        "a-unlocked-code.py",
        "z-unlocked-code.py",
    )
    assert caught.value.report.unsafe == ()


@pytest.mark.parametrize(
    "revision",
    ["main", "66317acc", "g" * 40, "66317ACC4C9FC17BD154591CE650735CD2855F3E"],
)
def test_lock_rejects_any_revision_that_is_not_a_full_lowercase_commit_sha(
    tmp_path: Path, revision: str
) -> None:
    payload = handwritten_lock({"config.json": b"x"}).decode().replace(REVISION, revision)
    path = tmp_path / "model.lock"
    path.write_text(payload)

    with pytest.raises(LockFormatError) as caught:
        ModelLock.load(path)

    assert caught.value.code == "invalid_revision"


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside",
        "/absolute/file",
        "nested/../../outside",
        "",
        ".",
        "a/./b",
        "a//b",
        "a/../b",
        r"a\b",
        r"C:\absolute\file",
        r"\\server\share\file",
    ],
)
def test_lock_rejects_noncanonical_or_escaping_paths(unsafe_path: str) -> None:
    lock = ModelLock(
        format_version=1,
        model_id=MODEL_ID,
        revision=REVISION,
        compiler_model_abi=1,
        files=(ModelFile(path=unsafe_path, digest=digest(b"x"), size=1),),
    )

    with pytest.raises(LockFormatError) as caught:
        lock.validate()

    assert caught.value.code == "invalid_path"


def test_verifier_rejects_leaf_symlinks_even_when_target_is_inside_snapshot(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "real.json").write_bytes(b"expected")
    (snapshot / "config.json").symlink_to(snapshot / "real.json")
    lock = load_test_lock(tmp_path, {"config.json": b"expected", "real.json": b"expected"})

    with pytest.raises(IntegrityError) as caught:
        verify_model_directory(lock, snapshot)

    assert caught.value.report.unsafe == ("config.json",)


def test_verifier_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    (real / "config.json").write_bytes(b"expected")
    (snapshot / "nested").symlink_to(real, target_is_directory=True)
    lock = load_test_lock(tmp_path, {"nested/config.json": b"expected"})

    with pytest.raises(IntegrityError) as caught:
        verify_model_directory(lock, snapshot)

    assert caught.value.report.unsafe == ("nested/config.json",)


def test_verifier_rejects_non_regular_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    os.mkfifo(snapshot / "config.json")
    lock = load_test_lock(tmp_path, {"config.json": b"expected"})

    with pytest.raises(IntegrityError) as caught:
        verify_model_directory(lock, snapshot)

    assert caught.value.report.unsafe == ("config.json",)


@pytest.mark.parametrize(
    ("old", "new", "code"),
    [
        ('size = 1', 'size = -1', "invalid_size"),
        ('size = 1', 'size = true', "invalid_size"),
        ('blake3 = "', 'sha256 = "', "invalid_file_schema"),
        ('compiler_model_abi = 1', 'compiler_model_abi = true', "invalid_abi"),
        ('format_version = 1', 'format_version = 2', "unsupported_format_version"),
        ('[files]', 'unknown = "field"\n\n[files]', "unknown_field"),
    ],
)
def test_lock_schema_rejects_ambiguous_or_unsupported_values(
    tmp_path: Path, old: str, new: str, code: str
) -> None:
    payload = handwritten_lock({"config.json": b"x"}).decode().replace(old, new, 1)
    path = tmp_path / "model.lock"
    path.write_text(payload)

    with pytest.raises(LockFormatError) as caught:
        ModelLock.load(path)

    assert caught.value.code == code


def test_empty_inventory_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.lock"
    path.write_text(
        f'format_version = 1\nmodel_id = "{MODEL_ID}"\nrevision = "{REVISION}"\n'
        "compiler_model_abi = 1\n\n[files]\n"
    )

    with pytest.raises(LockFormatError) as caught:
        ModelLock.load(path)

    assert caught.value.code == "empty_inventory"


def test_lock_writer_matches_an_independently_authored_canonical_fixture(
    tmp_path: Path,
) -> None:
    entries = (
        ModelFile(path="z-last.json", digest=digest(b"z"), size=1),
        ModelFile(path="a-first.json", digest=digest(b"a"), size=1),
    )
    lock = ModelLock(
        format_version=1,
        model_id=MODEL_ID,
        revision=REVISION,
        compiler_model_abi=1,
        files=entries,
    )
    target = tmp_path / "model.lock"
    lock.write(target)

    expected = handwritten_lock({"a-first.json": b"a", "z-last.json": b"z"})
    assert target.read_bytes() == expected
    assert ModelLock.load(target).canonical_bytes() == expected
