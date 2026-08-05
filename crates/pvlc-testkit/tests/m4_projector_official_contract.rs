use std::path::{Path, PathBuf};

use pvlc_cpu_ref::{
    LayerNormParameters, LinearParameters, ProjectorParameters, gelu_pytorch_tanh, projector_f32,
};
use pvlc_safetensors::SafetensorsCatalog;
use pvlc_testkit::{ComparisonAxes, ComparisonPolicy, ComparisonReport, compare_f32};

const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const HIDDEN_SIZE: usize = 1_152;
const MERGED_WIDTH: usize = HIDDEN_SIZE * 4;
const OUTPUT_WIDTH: usize = 1_024;
// Pinned remote Projector.__init__: LayerNorm(hidden_size, eps=1e-05).
const EPSILON: f32 = 1.0e-5;
const GOLDEN_LOCK: &str = include_str!("../../../goldens/golden.lock");

struct OfficialCase {
    directory: &'static str,
    artifact_path: &'static str,
    case_id: &'static str,
    trace_level: &'static str,
    bundle_digest: &'static str,
    semantic_fingerprint: &'static str,
    grid: [usize; 3],
    selected_output_rows: [usize; 5],
}

const L3: OfficialCase = OfficialCase {
    directory: "ocr.clean_latin.0001-l3",
    artifact_path: "artifacts/goldens/ocr.clean_latin.0001-l3",
    case_id: "ocr.clean_latin.0001",
    trace_level: "L3",
    bundle_digest: "blake3:35572ac07da5fddee97becb46fb866f906200f433f4c5677ad20fdf36440acf9",
    semantic_fingerprint: "blake3:632cbe2de6f47f8f84c764e3a551bf99f352d7459c98d939eb89725357f952b4",
    grid: [1, 22, 58],
    selected_output_rows: [0, 28, 29, 159, 318],
};

const L2: OfficialCase = OfficialCase {
    directory: "table.simple.0001-l2",
    artifact_path: "artifacts/goldens/table.simple.0001-l2",
    case_id: "table.simple.0001",
    trace_level: "L2",
    bundle_digest: "blake3:4cecc8b2abc37030920acf8860bb317084125dbbbcae5af394fccd0c8044c842",
    semantic_fingerprint: "blake3:91f72ffdf81070c2c9ca31628605064bdc02bf19417482cefce41d6a27aa5404",
    grid: [1, 30, 58],
    selected_output_rows: [0, 28, 29, 217, 434],
};

#[derive(Clone)]
struct OwnedParameters {
    pre_norm_weight: Vec<f32>,
    pre_norm_bias: Vec<f32>,
    linear1_weight: Vec<f32>,
    linear1_bias: Vec<f32>,
    linear2_weight: Vec<f32>,
    linear2_bias: Vec<f32>,
}

impl OwnedParameters {
    fn load(model: &SafetensorsCatalog) -> Self {
        Self {
            pre_norm_weight: load_tensor(model, "mlp_AR.pre_norm.weight", &[HIDDEN_SIZE as u64]),
            pre_norm_bias: load_tensor(model, "mlp_AR.pre_norm.bias", &[HIDDEN_SIZE as u64]),
            linear1_weight: load_tensor(
                model,
                "mlp_AR.linear_1.weight",
                &[MERGED_WIDTH as u64, MERGED_WIDTH as u64],
            ),
            linear1_bias: load_tensor(model, "mlp_AR.linear_1.bias", &[MERGED_WIDTH as u64]),
            linear2_weight: load_tensor(
                model,
                "mlp_AR.linear_2.weight",
                &[OUTPUT_WIDTH as u64, MERGED_WIDTH as u64],
            ),
            linear2_bias: load_tensor(model, "mlp_AR.linear_2.bias", &[OUTPUT_WIDTH as u64]),
        }
    }

    fn borrowed(&self) -> ProjectorParameters<'_> {
        ProjectorParameters {
            pre_norm: LayerNormParameters {
                weight: &self.pre_norm_weight,
                bias: &self.pre_norm_bias,
            },
            linear1: LinearParameters {
                weight: &self.linear1_weight,
                bias: &self.linear1_bias,
            },
            linear2: LinearParameters {
                weight: &self.linear2_weight,
                bias: &self.linear2_bias,
            },
        }
    }
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
    eprintln!("skipping local projector oracle: {}", path.display());
    None
}

fn load_tensor(catalog: &SafetensorsCatalog, name: &str, shape: &[u64]) -> Vec<f32> {
    assert_eq!(catalog.tensor(name).unwrap().shape, shape, "tensor={name}");
    catalog.load_tensor_f32(name).unwrap()
}

fn assert_exact_golden_identity(case: &OfficialCase) {
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

fn source_rows(case: &OfficialCase) -> Vec<usize> {
    let merged_width = case.grid[2] / 2;
    case.selected_output_rows
        .iter()
        .flat_map(|output| {
            let (merged_y, merged_x) = (output / merged_width, output % merged_width);
            let top_left = merged_y * 2 * case.grid[2] + merged_x * 2;
            [
                top_left,
                top_left + 1,
                top_left + case.grid[2],
                top_left + case.grid[2] + 1,
            ]
        })
        .collect()
}

fn selected_rows(values: &[f32], width: usize, rows: &[usize]) -> Vec<f32> {
    rows.iter()
        .flat_map(|row| values[row * width..(row + 1) * width].iter().copied())
        .collect()
}

fn policy(
    max_abs: f64,
    max_mean_abs: f64,
    max_p99_abs: f64,
    max_relative_l2: f64,
    min_cosine_similarity: f64,
    max_per_token_relative_l2: f64,
) -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs,
        max_mean_abs,
        max_p99_abs,
        max_relative_l2,
        min_cosine_similarity,
        max_per_token_relative_l2: Some(max_per_token_relative_l2),
        max_per_channel_relative_l2: None,
    }
}

fn pre_norm_policy() -> ComparisonPolicy {
    policy(0.07, 0.0008, 0.006, 0.0018, 0.999_998, 0.0025)
}

fn merge_policy() -> ComparisonPolicy {
    policy(0.07, 0.0008, 0.006, 0.0018, 0.999_998, 0.0023)
}

fn linear1_policy() -> ComparisonPolicy {
    policy(0.035, 0.0028, 0.013, 0.0032, 0.999_995, 0.004)
}

fn gelu_policy() -> ComparisonPolicy {
    policy(0.035, 0.002, 0.013, 0.0042, 0.999_992, 0.005)
}

fn linear2_policy() -> ComparisonPolicy {
    policy(0.045, 0.0042, 0.018, 0.004, 0.999_992, 0.006)
}

fn l2_linear2_policy() -> ComparisonPolicy {
    policy(0.05, 0.0045, 0.019, 0.004, 0.999_992, 0.006)
}

fn report(expected: &[f32], actual: &[f32], rows: usize, width: usize) -> ComparisonReport {
    compare_f32(
        expected,
        actual,
        &[rows, width],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap()
}

fn assert_stage(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    rows: usize,
    width: usize,
    comparison_policy: &ComparisonPolicy,
) {
    let report = report(expected, actual, rows, width);
    let verdict = report.assess(comparison_policy).unwrap();
    assert!(
        verdict.passed(),
        "{label}\n{report:#?}\nviolations={:?}",
        verdict.violations()
    );
    eprintln!(
        "{label}: max_abs={:.9} mean_abs={:.9} p99_abs={:.9} rel_l2={:.9} cosine={:.9} max_token_rel={:.9}",
        report.max_abs,
        report.mean_abs,
        report.p99_abs,
        report.relative_l2,
        report.cosine_similarity,
        report
            .per_token_relative_l2
            .as_deref()
            .unwrap()
            .iter()
            .copied()
            .fold(0.0_f64, f64::max),
    );
}

fn assert_rejected(
    label: &str,
    expected: &[f32],
    wrong: &[f32],
    rows: usize,
    width: usize,
    comparison_policy: &ComparisonPolicy,
) {
    let report = report(expected, wrong, rows, width);
    assert!(
        !report.assess(comparison_policy).unwrap().passed(),
        "negative control unexpectedly passed: {label}\n{report:#?}"
    );
}

fn load_case_input(case: &OfficialCase) -> (SafetensorsCatalog, Vec<f32>, Vec<usize>) {
    assert_exact_golden_identity(case);
    let stage = SafetensorsCatalog::open(
        repository()
            .join("artifacts/goldens")
            .join(case.directory)
            .join("stage-checkpoints.safetensors"),
    )
    .unwrap();
    let input_tokens = case.grid.into_iter().product::<usize>();
    let vision = load_tensor(
        &stage,
        "vision.final",
        &[input_tokens as u64, HIDDEN_SIZE as u64],
    );
    let rows = source_rows(case);
    let packed = selected_rows(&vision, HIDDEN_SIZE, &rows);
    (stage, packed, rows)
}

#[test]
fn pinned_l3_projector_matches_every_official_stage_and_rejects_stage_substitutions() {
    let Some(model_path) = require_model_checkpoint() else {
        return;
    };
    let model = SafetensorsCatalog::open(model_path).unwrap();
    let parameters = OwnedParameters::load(&model);
    let (stage, input, input_rows) = load_case_input(&L3);
    assert_eq!(
        input_rows,
        [
            0, 1, 58, 59, 56, 57, 114, 115, 116, 117, 174, 175, 608, 609, 666, 667, 1216, 1217,
            1274, 1275
        ]
    );
    let trace = projector_f32(
        &input,
        HIDDEN_SIZE,
        &[[1, 2, 2]; 5],
        parameters.borrowed(),
        EPSILON,
    )
    .unwrap();
    let deep = SafetensorsCatalog::open(
        repository()
            .join("artifacts/goldens")
            .join(L3.directory)
            .join("deep-checkpoints.safetensors"),
    )
    .unwrap();

    let expected_pre_norm = selected_rows(
        &load_tensor(&deep, "projector.pre_norm", &[1_276, HIDDEN_SIZE as u64]),
        HIDDEN_SIZE,
        &input_rows,
    );
    let expected_merge = selected_rows(
        &load_tensor(&deep, "projector.merge", &[319, MERGED_WIDTH as u64]),
        MERGED_WIDTH,
        &L3.selected_output_rows,
    );
    let expected_linear1 = selected_rows(
        &load_tensor(&deep, "projector.linear1", &[319, MERGED_WIDTH as u64]),
        MERGED_WIDTH,
        &L3.selected_output_rows,
    );
    let expected_gelu = selected_rows(
        &load_tensor(&deep, "projector.gelu", &[319, MERGED_WIDTH as u64]),
        MERGED_WIDTH,
        &L3.selected_output_rows,
    );
    let expected_linear2 = selected_rows(
        &load_tensor(&deep, "projector.linear2", &[319, OUTPUT_WIDTH as u64]),
        OUTPUT_WIDTH,
        &L3.selected_output_rows,
    );
    let expected_final = selected_rows(
        &load_tensor(&stage, "projector.final", &[319, OUTPUT_WIDTH as u64]),
        OUTPUT_WIDTH,
        &L3.selected_output_rows,
    );
    assert_eq!(expected_linear2, expected_final);

    assert_stage(
        "projector.pre_norm",
        &expected_pre_norm,
        &trace.pre_norm,
        20,
        HIDDEN_SIZE,
        &pre_norm_policy(),
    );
    assert_stage(
        "projector.merge",
        &expected_merge,
        &trace.merged,
        5,
        MERGED_WIDTH,
        &merge_policy(),
    );
    assert_stage(
        "projector.linear1",
        &expected_linear1,
        &trace.linear1,
        5,
        MERGED_WIDTH,
        &linear1_policy(),
    );
    assert_stage(
        "projector.gelu",
        &expected_gelu,
        &trace.activation,
        5,
        MERGED_WIDTH,
        &gelu_policy(),
    );
    assert_stage(
        "projector.linear2",
        &expected_linear2,
        &trace.output,
        5,
        OUTPUT_WIDTH,
        &linear2_policy(),
    );

    assert_rejected(
        "skipped pre_norm",
        &expected_pre_norm,
        &input,
        20,
        HIDDEN_SIZE,
        &pre_norm_policy(),
    );
    let wrong_merge = trace
        .pre_norm
        .chunks_exact(MERGED_WIDTH)
        .flat_map(|block| {
            [0, 2, 1, 3]
                .into_iter()
                .flat_map(|patch| {
                    block[patch * HIDDEN_SIZE..(patch + 1) * HIDDEN_SIZE]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_rejected(
        "wrong 2x2 patch order",
        &expected_merge,
        &wrong_merge,
        5,
        MERGED_WIDTH,
        &merge_policy(),
    );
    assert_rejected(
        "skipped linear1",
        &expected_linear1,
        &trace.merged,
        5,
        MERGED_WIDTH,
        &linear1_policy(),
    );
    assert_rejected(
        "skipped GELU",
        &expected_gelu,
        &trace.linear1,
        5,
        MERGED_WIDTH,
        &gelu_policy(),
    );
    let tanh_activation = trace
        .linear1
        .iter()
        .copied()
        .map(gelu_pytorch_tanh)
        .collect::<Vec<_>>();
    assert!(
        trace
            .activation
            .iter()
            .zip(&tanh_activation)
            .any(|(exact, tanh)| (*exact - *tanh).abs() > 1.0e-5),
        "the official exact GELU path must remain distinguishable from tanh approximation"
    );
    let mut rotated_output = trace.output.clone();
    rotated_output.rotate_left(OUTPUT_WIDTH);
    assert_rejected(
        "rotated output tokens",
        &expected_linear2,
        &rotated_output,
        5,
        OUTPUT_WIDTH,
        &linear2_policy(),
    );
}

#[test]
fn pinned_l2_shape_generalization_matches_official_projector_final() {
    let Some(model_path) = require_model_checkpoint() else {
        return;
    };
    let model = SafetensorsCatalog::open(model_path).unwrap();
    let parameters = OwnedParameters::load(&model);
    let (stage, input, input_rows) = load_case_input(&L2);
    assert_eq!(
        input_rows,
        [
            0, 1, 58, 59, 56, 57, 114, 115, 116, 117, 174, 175, 840, 841, 898, 899, 1680, 1681,
            1738, 1739
        ]
    );
    let trace = projector_f32(
        &input,
        HIDDEN_SIZE,
        &[[1, 2, 2]; 5],
        parameters.borrowed(),
        EPSILON,
    )
    .unwrap();
    let expected = selected_rows(
        &load_tensor(&stage, "projector.final", &[435, OUTPUT_WIDTH as u64]),
        OUTPUT_WIDTH,
        &L2.selected_output_rows,
    );
    assert_stage(
        "table.simple projector.final",
        &expected,
        &trace.output,
        5,
        OUTPUT_WIDTH,
        &l2_linear2_policy(),
    );

    let mut rotated = trace.output;
    rotated.rotate_left(OUTPUT_WIDTH);
    assert_rejected(
        "table.simple rotated output tokens",
        &expected,
        &rotated,
        5,
        OUTPUT_WIDTH,
        &l2_linear2_policy(),
    );
}
