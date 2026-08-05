use std::path::{Path, PathBuf};

use pvlc_cpu_ref::projector_merge_2x2_f32;
use pvlc_safetensors::SafetensorsCatalog;

const HIDDEN_SIZE: usize = 1_152;
const MERGED_WIDTH: usize = HIDDEN_SIZE * 4;
const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const GOLDEN_LOCK: &str = include_str!("../../../goldens/golden.lock");

struct OfficialMergeCase {
    directory: &'static str,
    artifact_path: &'static str,
    case_id: &'static str,
    trace_level: &'static str,
    bundle_digest: &'static str,
    semantic_fingerprint: &'static str,
    grid: [usize; 3],
    output_tokens: usize,
    expected_blake3: &'static str,
    first_boundary_values: [f32; 8],
    last_boundary_values: [f32; 8],
}

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn hash_f32_le(values: &[f32]) -> String {
    let mut hasher = blake3::Hasher::new();
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn selected_boundaries(values: &[f32], row: usize) -> [f32; 8] {
    let start = row * MERGED_WIDTH;
    [
        values[start],
        values[start + 1_151],
        values[start + 1_152],
        values[start + 2_303],
        values[start + 2_304],
        values[start + 3_455],
        values[start + 3_456],
        values[start + 4_607],
    ]
}

fn assert_exact_golden_identity(case: &OfficialMergeCase) {
    let lock = toml::from_str::<toml::Table>(GOLDEN_LOCK).unwrap();
    assert_eq!(lock["model_revision"].as_str(), Some(MODEL_REVISION));
    assert_eq!(lock["trace_schema_version"].as_integer(), Some(1));
    let matches = lock["bundles"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|bundle| bundle["case_id"].as_str() == Some(case.case_id))
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "case_id={}", case.case_id);
    let bundle = matches[0];
    assert_eq!(bundle["trace_level"].as_str(), Some(case.trace_level));
    assert_eq!(bundle["artifact_path"].as_str(), Some(case.artifact_path));
    assert_eq!(bundle["bundle_digest"].as_str(), Some(case.bundle_digest));
    assert_eq!(
        bundle["semantic_fingerprint"].as_str(),
        Some(case.semantic_fingerprint)
    );
}

#[test]
fn pinned_official_shapes_have_exact_projector_merge_2x2_anchors() {
    let cases = [
        OfficialMergeCase {
            directory: "ocr.clean_latin.0001-l3",
            artifact_path: "artifacts/goldens/ocr.clean_latin.0001-l3",
            case_id: "ocr.clean_latin.0001",
            trace_level: "L3",
            bundle_digest: "blake3:35572ac07da5fddee97becb46fb866f906200f433f4c5677ad20fdf36440acf9",
            semantic_fingerprint: "blake3:632cbe2de6f47f8f84c764e3a551bf99f352d7459c98d939eb89725357f952b4",
            grid: [1, 22, 58],
            output_tokens: 319,
            expected_blake3: "e2d53992b7144e94e8f950e36be8fda8fe1ec88e67bb18d99ac4f86de507344a",
            first_boundary_values: [
                -0.23339844,
                0.13769531,
                -0.3046875,
                0.15136719,
                -0.26953125,
                -0.08251953,
                -0.24121094,
                -0.07910156,
            ],
            last_boundary_values: [
                -0.20703125,
                1.546875,
                -0.31054688,
                1.6484375,
                -0.5,
                1.671875,
                -0.42382813,
                1.78125,
            ],
        },
        OfficialMergeCase {
            directory: "table.simple.0001-l2",
            artifact_path: "artifacts/goldens/table.simple.0001-l2",
            case_id: "table.simple.0001",
            trace_level: "L2",
            bundle_digest: "blake3:4cecc8b2abc37030920acf8860bb317084125dbbbcae5af394fccd0c8044c842",
            semantic_fingerprint: "blake3:91f72ffdf81070c2c9ca31628605064bdc02bf19417482cefce41d6a27aa5404",
            grid: [1, 30, 58],
            output_tokens: 435,
            expected_blake3: "8fc722021088cc5f8126c2e144c798d26362183c0ea6aa70ad20ffc30ddbed93",
            first_boundary_values: [
                0.25,
                0.44335938,
                0.20117188,
                0.32226563,
                -0.57421875,
                0.012817383,
                -0.69921875,
                0.18457031,
            ],
            last_boundary_values: [
                0.41210938, 1.5, 0.359375, 1.6640625, 0.13671875, 1.5703125, 0.11035156, 1.625,
            ],
        },
    ];

    if cases.iter().any(|case| {
        !repository()
            .join("artifacts/goldens")
            .join(case.directory)
            .join("stage-checkpoints.safetensors")
            .is_file()
    }) {
        eprintln!("skipping official projector anchors: stage fixtures are not distributed");
        return;
    }

    for case in cases {
        assert_exact_golden_identity(&case);

        let catalog = SafetensorsCatalog::open(
            repository()
                .join("artifacts/goldens")
                .join(case.directory)
                .join("stage-checkpoints.safetensors"),
        )
        .unwrap();
        let input_tokens = case.grid.into_iter().product::<usize>();
        assert_eq!(
            catalog.tensor("vision.final").unwrap().shape,
            [input_tokens as u64, HIDDEN_SIZE as u64]
        );
        assert_eq!(
            catalog.tensor("projector.final").unwrap().shape,
            [case.output_tokens as u64, 1_024]
        );
        assert_eq!(input_tokens / 4, case.output_tokens);

        let vision_final = catalog.load_tensor_f32("vision.final").unwrap();
        let merged = projector_merge_2x2_f32(&vision_final, HIDDEN_SIZE, &[case.grid]).unwrap();
        assert_eq!(merged.len(), case.output_tokens * MERGED_WIDTH);
        assert_eq!(selected_boundaries(&merged, 0), case.first_boundary_values);
        assert_eq!(
            selected_boundaries(&merged, case.output_tokens - 1),
            case.last_boundary_values
        );
        assert_eq!(hash_f32_le(&merged), case.expected_blake3);
    }
}
