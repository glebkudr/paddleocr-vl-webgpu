use std::path::{Path, PathBuf};

use pvlc_cpu_ref::{
    LayerNormParameters as CpuLayerNormParameters, LinearParameters as CpuLinearParameters,
    ProjectorParameters as CpuProjectorParameters, projector_f32 as cpu_projector_f32,
};
use pvlc_runtime_core::{
    ProjectorInvocation, ProjectorParameters, ProjectorReadback, ProjectorStage,
    VisionLayerNormParameters, VisionLinearParameters,
};
use pvlc_runtime_native::{BackendKind, NativeOptions, NativeRuntime};
use pvlc_safetensors::SafetensorsCatalog;
use pvlc_testkit::{ComparisonAxes, ComparisonPolicy, compare_f32};

const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const HIDDEN_SIZE: usize = 1_152;
const MERGED_WIDTH: usize = HIDDEN_SIZE * 4;
const OUTPUT_WIDTH: usize = 1_024;
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

    fn runtime(&self) -> ProjectorParameters<'_> {
        ProjectorParameters {
            pre_norm: VisionLayerNormParameters {
                weight: &self.pre_norm_weight,
                bias: &self.pre_norm_bias,
            },
            linear1: VisionLinearParameters {
                weight: &self.linear1_weight,
                bias: &self.linear1_bias,
            },
            linear2: VisionLinearParameters {
                weight: &self.linear2_weight,
                bias: &self.linear2_bias,
            },
        }
    }

    fn cpu(&self) -> CpuProjectorParameters<'_> {
        CpuProjectorParameters {
            pre_norm: CpuLayerNormParameters {
                weight: &self.pre_norm_weight,
                bias: &self.pre_norm_bias,
            },
            linear1: CpuLinearParameters {
                weight: &self.linear1_weight,
                bias: &self.linear1_bias,
            },
            linear2: CpuLinearParameters {
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
    eprintln!("skipping local native projector oracle: {}", path.display());
    None
}

fn require_runtime() -> Option<NativeRuntime> {
    match NativeRuntime::new(NativeOptions::default()) {
        Ok(runtime) => {
            if std::env::var("PVLC_REQUIRE_M4_METAL").as_deref() == Ok("1") {
                assert_eq!(runtime.capabilities().backend, BackendKind::Metal);
                assert!(runtime.capabilities().adapter_name.contains("M4 Pro"));
            }
            Some(runtime)
        }
        Err(error)
            if std::env::var("PVLC_REQUIRE_NATIVE_GPU").as_deref() == Ok("1")
                || std::env::var("PVLC_REQUIRE_M4_METAL").as_deref() == Ok("1") =>
        {
            panic!("native GPU is required: {error}")
        }
        Err(error) => {
            eprintln!("skipping native official projector oracle: {error}");
            None
        }
    }
}

fn load_tensor(catalog: &SafetensorsCatalog, name: &str, shape: &[u64]) -> Vec<f32> {
    assert_eq!(catalog.tensor(name).unwrap().shape, shape, "tensor={name}");
    catalog.load_tensor_f32(name).unwrap()
}

fn assert_exact_golden_identity(case: &OfficialCase) {
    let lock = toml::from_str::<toml::Table>(GOLDEN_LOCK).unwrap();
    assert_eq!(lock["model_revision"].as_str(), Some(MODEL_REVISION));
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
    case.selected_output_rows
        .iter()
        .flat_map(|output| {
            let (merged_y, merged_x) = (output / (case.grid[2] / 2), output % (case.grid[2] / 2));
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
    mean_abs: f64,
    p99_abs: f64,
    relative_l2: f64,
    cosine: f64,
    token_l2: f64,
) -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs,
        max_mean_abs: mean_abs,
        max_p99_abs: p99_abs,
        max_relative_l2: relative_l2,
        min_cosine_similarity: cosine,
        max_per_token_relative_l2: Some(token_l2),
        max_per_channel_relative_l2: None,
    }
}

fn assert_stage(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    rows: usize,
    width: usize,
    comparison_policy: ComparisonPolicy,
) {
    let report = compare_f32(
        expected,
        actual,
        &[rows, width],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap();
    let verdict = report.assess(&comparison_policy).unwrap();
    assert!(
        verdict.passed(),
        "{label}\n{report:#?}\nviolations={:?}",
        verdict.violations()
    );
    let max_abs = report.max_abs;
    let mean_abs = report.mean_abs;
    let p99_abs = report.p99_abs;
    let relative_l2 = report.relative_l2;
    let cosine_similarity = report.cosine_similarity;
    eprintln!(
        "native {label}: max_abs={max_abs:.9} mean_abs={mean_abs:.9} p99_abs={p99_abs:.9} rel_l2={relative_l2:.9} cosine={cosine_similarity:.9}"
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
    let input = selected_rows(&vision, HIDDEN_SIZE, &rows);
    (stage, input, rows)
}

#[test]
fn native_resident_projector_matches_cpu_and_both_pinned_official_shapes() {
    let Some(model_path) = require_model_checkpoint() else {
        return;
    };
    let Some(runtime) = require_runtime() else {
        return;
    };
    let model = SafetensorsCatalog::open(model_path).unwrap();
    let parameters = OwnedParameters::load(&model);

    let (l3_stage, l3_input, l3_input_rows) = load_case_input(&L3);
    let grids = [[1_u32, 2, 2]; 5];
    let invocation = ProjectorInvocation {
        hidden_size: HIDDEN_SIZE as u32,
        output_size: OUTPUT_WIDTH as u32,
        layer_norm_epsilon: EPSILON,
        input: &l3_input,
        image_grid_thw: &grids,
        parameters: parameters.runtime(),
    };
    let native = runtime
        .run_projector(&invocation, ProjectorReadback::AllStages)
        .unwrap();
    let cpu = cpu_projector_f32(
        &l3_input,
        HIDDEN_SIZE,
        &[[1_usize, 2, 2]; 5],
        parameters.cpu(),
        EPSILON,
    )
    .unwrap();
    for (stage, cpu_values) in [
        (ProjectorStage::PreNorm, cpu.pre_norm.as_slice()),
        (ProjectorStage::Merge, cpu.merged.as_slice()),
        (ProjectorStage::Linear1, cpu.linear1.as_slice()),
        (ProjectorStage::Activation, cpu.activation.as_slice()),
        (ProjectorStage::Linear2, cpu.output.as_slice()),
    ] {
        let native_values = &native.checkpoints[&stage];
        let max_abs = native_values
            .iter()
            .zip(cpu_values)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs <= 2.0e-4,
            "stage={stage:?} native-vs-cpu max_abs={max_abs}"
        );
    }
    assert_eq!(native.diagnostics.submission_count, 1);
    assert_eq!(native.diagnostics.compute_pass_count, 1);
    assert_eq!(native.diagnostics.dispatch_count, 5);
    assert!(native.diagnostics.captured_errors.is_empty());

    let deep = SafetensorsCatalog::open(
        repository()
            .join("artifacts/goldens")
            .join(L3.directory)
            .join("deep-checkpoints.safetensors"),
    )
    .unwrap();
    let references = [
        (
            ProjectorStage::PreNorm,
            selected_rows(
                &load_tensor(&deep, "projector.pre_norm", &[1_276, HIDDEN_SIZE as u64]),
                HIDDEN_SIZE,
                &l3_input_rows,
            ),
            20,
            HIDDEN_SIZE,
            policy(0.08, 0.001, 0.007, 0.0022, 0.999_997, 0.003),
        ),
        (
            ProjectorStage::Merge,
            selected_rows(
                &load_tensor(&deep, "projector.merge", &[319, MERGED_WIDTH as u64]),
                MERGED_WIDTH,
                &L3.selected_output_rows,
            ),
            5,
            MERGED_WIDTH,
            policy(0.08, 0.001, 0.007, 0.0022, 0.999_997, 0.003),
        ),
        (
            ProjectorStage::Linear1,
            selected_rows(
                &load_tensor(&deep, "projector.linear1", &[319, MERGED_WIDTH as u64]),
                MERGED_WIDTH,
                &L3.selected_output_rows,
            ),
            5,
            MERGED_WIDTH,
            policy(0.045, 0.0032, 0.016, 0.0038, 0.999_993, 0.005),
        ),
        (
            ProjectorStage::Activation,
            selected_rows(
                &load_tensor(&deep, "projector.gelu", &[319, MERGED_WIDTH as u64]),
                MERGED_WIDTH,
                &L3.selected_output_rows,
            ),
            5,
            MERGED_WIDTH,
            policy(0.045, 0.0024, 0.016, 0.005, 0.999_988, 0.006),
        ),
        (
            ProjectorStage::Linear2,
            selected_rows(
                &load_tensor(&deep, "projector.linear2", &[319, OUTPUT_WIDTH as u64]),
                OUTPUT_WIDTH,
                &L3.selected_output_rows,
            ),
            5,
            OUTPUT_WIDTH,
            policy(0.06, 0.0055, 0.022, 0.005, 0.999_988, 0.007),
        ),
    ];
    for (stage, expected, rows, width, comparison_policy) in references {
        assert_stage(
            stage.as_str(),
            &expected,
            &native.checkpoints[&stage],
            rows,
            width,
            comparison_policy,
        );
    }
    let l3_final = selected_rows(
        &load_tensor(&l3_stage, "projector.final", &[319, OUTPUT_WIDTH as u64]),
        OUTPUT_WIDTH,
        &L3.selected_output_rows,
    );
    assert_eq!(
        l3_final,
        selected_rows(
            &load_tensor(&deep, "projector.linear2", &[319, OUTPUT_WIDTH as u64]),
            OUTPUT_WIDTH,
            &L3.selected_output_rows,
        )
    );

    let (l2_stage, l2_input, _) = load_case_input(&L2);
    let l2_invocation = ProjectorInvocation {
        input: &l2_input,
        ..invocation
    };
    let l2_native = runtime
        .run_projector(&l2_invocation, ProjectorReadback::OutputOnly)
        .unwrap();
    let l2_expected = selected_rows(
        &load_tensor(&l2_stage, "projector.final", &[435, OUTPUT_WIDTH as u64]),
        OUTPUT_WIDTH,
        &L2.selected_output_rows,
    );
    assert_stage(
        "table.simple projector.final",
        &l2_expected,
        &l2_native.checkpoints[&ProjectorStage::Linear2],
        5,
        OUTPUT_WIDTH,
        policy(0.065, 0.006, 0.024, 0.005, 0.999_988, 0.007),
    );
    assert_eq!(l2_native.checkpoints.len(), 1);
    assert_eq!(
        l2_native.diagnostics.readback_bytes,
        (5 * OUTPUT_WIDTH * 4) as u64
    );
}
