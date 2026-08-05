use std::path::{Path, PathBuf};

use pvlc_cpu_ref::{add_interpolated_position_embedding_f32, patch_projection_f32};
use pvlc_safetensors::SafetensorsCatalog;
use pvlc_testkit::{ComparisonAxes, ComparisonPolicy, compare_f32};

const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const PATCHES: usize = 1_276;
const CHANNELS: usize = 3;
const PATCH_SIZE: usize = 14;
const HIDDEN_SIZE: usize = 1_152;
const SAMPLED_PATCHES: [usize; 7] = [0, 178, 190, 237, 244, 253, 1_275];
const MODEL_WEIGHT: &str = "visual.vision_model.embeddings.patch_embedding.weight";
const MODEL_BIAS: &str = "visual.vision_model.embeddings.patch_embedding.bias";
const MODEL_POSITION: &str = "visual.vision_model.embeddings.position_embedding.weight";
const PROCESSOR_PIXELS: &str = "processor.pixel_values";
const OFFICIAL_PATCH: &str = "vision.embeddings.patch";
const OFFICIAL_OUTPUT: &str = "vision.embeddings.output";
const GOLDEN_LOCK: &str = include_str!("../../../goldens/golden.lock");

struct OfficialInputs {
    pixels: Vec<f32>,
    weights: Vec<f32>,
    bias: Vec<f32>,
    positions: Vec<f32>,
    official: Vec<f32>,
    official_output: Vec<f32>,
}

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn require_model_checkpoint() -> Option<PathBuf> {
    let path = repository()
        .join("models/snapshots")
        .join(MODEL_REVISION)
        .join("model.safetensors");
    if path.is_file() {
        return Some(path);
    }
    assert_ne!(
        std::env::var("PVLC_REQUIRE_MODEL").as_deref(),
        Ok("1"),
        "PVLC_REQUIRE_MODEL=1 but pinned checkpoint is absent at {}",
        path.display()
    );
    eprintln!(
        "skipping local patch-embedding oracle: {} is absent",
        path.display()
    );
    None
}

fn load_official_inputs() -> Option<OfficialInputs> {
    let model_path = require_model_checkpoint()?;
    assert!(GOLDEN_LOCK.contains(
        "bundle_digest = \"blake3:35572ac07da5fddee97becb46fb866f906200f433f4c5677ad20fdf36440acf9\""
    ));
    assert!(GOLDEN_LOCK.contains(
        "semantic_fingerprint = \"blake3:632cbe2de6f47f8f84c764e3a551bf99f352d7459c98d939eb89725357f952b4\""
    ));

    let model = SafetensorsCatalog::open(model_path).unwrap();
    assert_eq!(
        model.tensor(MODEL_WEIGHT).unwrap().shape,
        [
            HIDDEN_SIZE as u64,
            CHANNELS as u64,
            PATCH_SIZE as u64,
            PATCH_SIZE as u64
        ]
    );
    assert_eq!(
        model.tensor(MODEL_BIAS).unwrap().shape,
        [HIDDEN_SIZE as u64]
    );
    let weights = model.load_tensor_f32(MODEL_WEIGHT).unwrap();
    let bias = model.load_tensor_f32(MODEL_BIAS).unwrap();
    assert_eq!(
        model.tensor(MODEL_POSITION).unwrap().shape,
        [27 * 27, HIDDEN_SIZE as u64]
    );
    let positions = model.load_tensor_f32(MODEL_POSITION).unwrap();

    let golden = repository().join("artifacts/goldens/ocr.clean_latin.0001-l3");
    let processor = SafetensorsCatalog::open(golden.join("processor.safetensors")).unwrap();
    assert_eq!(
        processor.tensor(PROCESSOR_PIXELS).unwrap().shape,
        [
            PATCHES as u64,
            CHANNELS as u64,
            PATCH_SIZE as u64,
            PATCH_SIZE as u64
        ]
    );
    let pixels = processor.load_tensor_f32(PROCESSOR_PIXELS).unwrap();

    let deep = SafetensorsCatalog::open(golden.join("deep-checkpoints.safetensors")).unwrap();
    assert_eq!(
        deep.tensor(OFFICIAL_PATCH).unwrap().shape,
        [1, PATCHES as u64, HIDDEN_SIZE as u64]
    );
    let official = deep.load_tensor_f32(OFFICIAL_PATCH).unwrap();
    assert_eq!(official[0], 1.171875);
    assert_eq!(
        deep.tensor(OFFICIAL_OUTPUT).unwrap().shape,
        [1, PATCHES as u64, HIDDEN_SIZE as u64]
    );
    let official_output = deep.load_tensor_f32(OFFICIAL_OUTPUT).unwrap();
    assert_eq!(official_output[0], 1.125);
    Some(OfficialInputs {
        pixels,
        weights,
        bias,
        positions,
        official,
        official_output,
    })
}

fn selected_rows(values: &[f32], width: usize, rows: &[usize]) -> Vec<f32> {
    rows.iter()
        .flat_map(|row| values[row * width..(row + 1) * width].iter().copied())
        .collect()
}

fn sampled_policy() -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs: 0.016,
        max_mean_abs: 0.0011,
        max_p99_abs: 0.0042,
        max_relative_l2: 0.002,
        min_cosine_similarity: 0.999_998,
        max_per_token_relative_l2: Some(0.002),
        max_per_channel_relative_l2: None,
    }
}

fn full_policy() -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs: 0.032,
        max_mean_abs: 0.0019,
        max_p99_abs: 0.0043,
        max_relative_l2: 0.002,
        min_cosine_similarity: 0.999_997_5,
        max_per_token_relative_l2: Some(0.0023),
        max_per_channel_relative_l2: None,
    }
}

fn position_policy() -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs: 0.56,
        max_mean_abs: 0.002,
        max_p99_abs: 0.0056,
        max_relative_l2: 0.0021,
        min_cosine_similarity: 0.999_997_5,
        max_per_token_relative_l2: Some(0.0036),
        max_per_channel_relative_l2: None,
    }
}

fn combined_embedding_policy() -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs: 0.56,
        max_mean_abs: 0.0026,
        max_p99_abs: 0.008,
        max_relative_l2: 0.0029,
        min_cosine_similarity: 0.999_996_3,
        max_per_token_relative_l2: Some(0.0038),
        max_per_channel_relative_l2: None,
    }
}

#[test]
fn real_patch_projection_matches_official_mps_bfloat16_on_content_and_border_patches() {
    let Some(inputs) = load_official_inputs() else {
        return;
    };
    let input_width = CHANNELS * PATCH_SIZE * PATCH_SIZE;
    let sampled_pixels = selected_rows(&inputs.pixels, input_width, &SAMPLED_PATCHES);
    assert!(sampled_pixels.iter().any(|value| *value <= -0.95));
    assert!(sampled_pixels.contains(&1.0));
    let expected = selected_rows(&inputs.official, HIDDEN_SIZE, &SAMPLED_PATCHES);

    let actual = patch_projection_f32(
        &sampled_pixels,
        SAMPLED_PATCHES.len(),
        CHANNELS,
        PATCH_SIZE,
        &inputs.weights,
        &inputs.bias,
        HIDDEN_SIZE,
    )
    .unwrap();
    let report = compare_f32(
        &expected,
        &actual,
        &[SAMPLED_PATCHES.len(), HIDDEN_SIZE],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap();
    let verdict = report.assess(&sampled_policy()).unwrap();
    assert!(verdict.passed(), "{report:#?}\n{verdict:#?}");

    let mut wrong_channel_order = actual;
    for row in wrong_channel_order.chunks_exact_mut(HIDDEN_SIZE) {
        row.rotate_left(1);
    }
    let wrong_report = compare_f32(
        &expected,
        &wrong_channel_order,
        &[SAMPLED_PATCHES.len(), HIDDEN_SIZE],
        ComparisonAxes::default(),
    )
    .unwrap();
    assert!(!wrong_report.assess(&sampled_policy()).unwrap().passed());
}

#[test]
fn builtin_transformers_position_geometry_rejects_the_legacy_remote_fixture() {
    let Some(inputs) = load_official_inputs() else {
        return;
    };
    let actual = add_interpolated_position_embedding_f32(
        &inputs.official,
        HIDDEN_SIZE,
        &inputs.positions,
        27,
        27,
        &[[1, 22, 58]],
    )
    .unwrap();
    for (target_token, source_position) in [
        (0, 0),
        (57, 26),
        (21 * 58, 26 * 27),
        (22 * 58 - 1, 27 * 27 - 1),
    ] {
        for channel in 0..HIDDEN_SIZE {
            let target = target_token * HIDDEN_SIZE + channel;
            let source = source_position * HIDDEN_SIZE + channel;
            assert_eq!(
                actual[target],
                inputs.official[target] + inputs.positions[source],
                "endpoint-aligned builtin interpolation drifted at token {target_token}, channel {channel}",
            );
        }
    }

    let report = compare_f32(
        &inputs.official_output,
        &actual,
        &[PATCHES, HIDDEN_SIZE],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap();
    let verdict = report.assess(&position_policy()).unwrap();
    assert!(
        !verdict.passed(),
        "the legacy remote-code fixture unexpectedly became a valid builtin Transformers oracle"
    );
}

#[test]
#[ignore = "M3d hard gate evaluates full patch projection and positional interpolation"]
fn full_real_patch_and_position_embedding_match_official_l3_checkpoints() {
    let Some(inputs) = load_official_inputs() else {
        return;
    };
    let actual = patch_projection_f32(
        &inputs.pixels,
        PATCHES,
        CHANNELS,
        PATCH_SIZE,
        &inputs.weights,
        &inputs.bias,
        HIDDEN_SIZE,
    )
    .unwrap();
    let report = compare_f32(
        &inputs.official,
        &actual,
        &[PATCHES, HIDDEN_SIZE],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap();
    let verdict = report.assess(&full_policy()).unwrap();
    assert!(verdict.passed(), "{report:#?}\n{verdict:#?}");

    let embedding = add_interpolated_position_embedding_f32(
        &actual,
        HIDDEN_SIZE,
        &inputs.positions,
        27,
        27,
        &[[1, 22, 58]],
    )
    .unwrap();
    let embedding_report = compare_f32(
        &inputs.official_output,
        &embedding,
        &[PATCHES, HIDDEN_SIZE],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap();
    let embedding_verdict = embedding_report
        .assess(&combined_embedding_policy())
        .unwrap();
    assert!(
        embedding_verdict.passed(),
        "{embedding_report:#?}\n{embedding_verdict:#?}"
    );
}
