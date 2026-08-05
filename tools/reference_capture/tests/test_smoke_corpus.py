from __future__ import annotations

import json
import struct
from pathlib import Path

from blake3 import blake3

from pvlc_reference.trace_bundle import CaseSpec


REPO_ROOT = Path(__file__).parents[3]
CORPUS_ROOT = REPO_ROOT / "cases" / "smoke"


def png_dimensions(payload: bytes) -> tuple[int, int]:
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert payload[12:16] == b"IHDR"
    return struct.unpack(">II", payload[16:24])


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"


def test_m0_smoke_corpus_has_twelve_real_files_and_all_official_tasks() -> None:
    manifest = json.loads((CORPUS_ROOT / "manifest.json").read_bytes())

    assert manifest["corpus_version"] == 1
    assert len(manifest["cases"]) == 12
    assert len({entry["category"] for entry in manifest["cases"]}) == 12

    cases: list[CaseSpec] = []
    source_hashes: set[str] = set()
    for entry in manifest["cases"]:
        case_path = (CORPUS_ROOT / entry["case"]).resolve()
        image_path = (CORPUS_ROOT / entry["image"]).resolve()
        assert case_path.is_relative_to(CORPUS_ROOT.resolve())
        assert image_path.is_relative_to(CORPUS_ROOT.resolve())

        case_payload = json.loads(case_path.read_bytes())
        case = CaseSpec.from_dict(case_payload)
        image = image_path.read_bytes()
        width, height = png_dimensions(image)

        assert case_path.read_bytes() == canonical_json(case_payload)
        assert case.source_media_type == "image/png"
        assert (case.width, case.height) == (width, height)
        assert case.source_image_hash == f"blake3:{blake3(image).hexdigest()}"
        assert case.source_image_hash not in source_hashes
        source_hashes.add(case.source_image_hash)
        cases.append(case)

    assert {case.task for case in cases} == {
        "ocr",
        "table",
        "formula",
        "chart",
        "spotting",
        "seal",
    }
    assert any(case.width / case.height > 5 for case in cases)
    assert any(case.height / case.width > 2 for case in cases)
    assert next(case for case in cases if case.task == "spotting").max_pixels == 1_605_632
    assert all(case.do_sample is False for case in cases)


def test_smoke_cases_cover_the_declared_correctness_risk_categories() -> None:
    manifest = json.loads((CORPUS_ROOT / "manifest.json").read_bytes())
    categories = {entry["category"] for entry in manifest["cases"]}

    assert {
        "clean-print-latin-digits",
        "cyrillic-digits-currency",
        "table-simple-grid",
        "table-merged-header",
        "formula-indices-greek",
        "chart-bar-labels",
        "seal-multilingual",
        "spotting-multiple-regions",
        "low-contrast",
        "skew",
        "extreme-wide",
        "extreme-tall",
    } <= categories
