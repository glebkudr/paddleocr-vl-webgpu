use std::{env, path::Path, sync::OnceLock};

use pvlc_cpu_ref::patch_projection_f32;
use pvlc_runtime_core::{KernelId, KernelInvocation};
use pvlc_runtime_native::{BackendKind, ErrorScopeKind, NativeOptions, NativeRuntime};
use pvlc_safetensors::SafetensorsCatalog;
use pvlc_testkit::{ComparisonAxes, ComparisonPolicy, compare_f32};

const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const PATCHES: usize = 1_276;
const INPUT_WIDTH: usize = 3 * 14 * 14;
const OUTPUT_WIDTH: usize = 1_152;
const SAMPLED_PATCHES: [usize; 7] = [0, 178, 190, 237, 244, 253, 1_275];
const MODEL_WEIGHT: &str = "visual.vision_model.embeddings.patch_embedding.weight";
const MODEL_BIAS: &str = "visual.vision_model.embeddings.patch_embedding.bias";
const GOLDEN_LOCK: &str = include_str!("../../../goldens/golden.lock");

static RUNTIME: OnceLock<Result<NativeRuntime, String>> = OnceLock::new();

struct RealInputs {
    pixels: Vec<f32>,
    weight: Vec<f32>,
    bias: Vec<f32>,
    official: Vec<f32>,
}

fn env_flag(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1" | "true"))
}

fn hardware_required() -> bool {
    env_flag("PVLC_REQUIRE_NATIVE_GPU") || env_flag("PVLC_REQUIRE_M4_METAL")
}

fn runtime() -> Option<&'static NativeRuntime> {
    match RUNTIME.get_or_init(|| {
        NativeRuntime::new(NativeOptions::default()).map_err(|error| error.to_string())
    }) {
        Ok(runtime) => {
            if env_flag("PVLC_REQUIRE_M4_METAL") {
                assert_eq!(runtime.capabilities().backend, BackendKind::Metal);
                assert!(runtime.capabilities().adapter_name.contains("M4 Pro"));
            }
            Some(runtime)
        }
        Err(error) if hardware_required() => panic!("native GPU is required: {error}"),
        Err(error) => {
            eprintln!("skipping native patch projection: {error}");
            None
        }
    }
}

fn values(length: usize, seed: u32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let phase = (index as f32 + 1.0) * (seed as f32 + 3.0) * 0.017;
            phase.sin() * 0.75 + phase.cos() * 0.25
        })
        .collect()
}

fn invocation(
    patch_count: usize,
    input_width: usize,
    output_width: usize,
    input: Vec<f32>,
    weight: Vec<f32>,
    bias: Vec<f32>,
) -> KernelInvocation {
    KernelInvocation::VisionPatchProjectionF32 {
        patch_count: patch_count as u32,
        input_width: input_width as u32,
        output_width: output_width as u32,
        input,
        weight,
        bias,
    }
}

fn gpu_policy() -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs: 1.0e-4,
        max_mean_abs: 2.0e-5,
        max_p99_abs: 6.0e-5,
        max_relative_l2: 3.0e-5,
        min_cosine_similarity: 0.999_99,
        max_per_token_relative_l2: None,
        max_per_channel_relative_l2: None,
    }
}

fn official_sample_policy() -> ComparisonPolicy {
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

fn official_full_policy() -> ComparisonPolicy {
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

fn run(runtime: &NativeRuntime, invocation: &KernelInvocation) -> Vec<f32> {
    let execution = runtime.run(invocation).unwrap();
    assert_eq!(
        execution.diagnostics.kernel,
        KernelId::VisionPatchProjectionF32
    );
    let source = pvlc_wgsl::module(KernelId::VisionPatchProjectionF32)
        .unwrap()
        .source;
    assert_eq!(
        execution.diagnostics.shader_blake3,
        *blake3::hash(source.as_bytes()).as_bytes()
    );
    assert_eq!(
        execution.diagnostics.checked_error_scopes,
        [
            ErrorScopeKind::Validation,
            ErrorScopeKind::OutOfMemory,
            ErrorScopeKind::Internal,
        ]
    );
    assert!(execution.diagnostics.captured_errors.is_empty());
    assert!(execution.diagnostics.queue_wall_time_ns > 0);
    if env_flag("PVLC_REQUIRE_TIMESTAMP_QUERY") {
        assert!(runtime.capabilities().timestamp_query);
    }
    if runtime.capabilities().timestamp_query {
        let timestamp = execution.diagnostics.timestamp.unwrap();
        assert!(timestamp.end_ticks > timestamp.begin_ticks);
        assert!(timestamp.duration_ns.is_finite() && timestamp.duration_ns > 0.0);
    } else {
        assert!(execution.diagnostics.timestamp.is_none());
    }
    execution.values
}

fn assert_comparison(
    expected: &[f32],
    actual: &[f32],
    shape: &[usize],
    axes: ComparisonAxes,
    policy: ComparisonPolicy,
) {
    let report = compare_f32(expected, actual, shape, axes).unwrap();
    let verdict = report.assess(&policy).unwrap();
    assert!(verdict.passed(), "{report:#?}\n{verdict:#?}");
}

#[test]
fn native_patch_projection_matches_cpu_across_both_dispatch_tails_and_asymmetric_shapes() {
    let Some(runtime) = runtime() else { return };
    for (case, (patch_count, channels, patch_size, output_width)) in
        [(1, 1, 1, 1), (7, 1, 3, 9), (9, 3, 2, 7), (17, 2, 4, 17)]
            .into_iter()
            .enumerate()
    {
        let input_width = channels * patch_size * patch_size;
        let input = values(patch_count * input_width, case as u32 + 1);
        let weight = values(output_width * input_width, case as u32 + 11);
        let bias = values(output_width, case as u32 + 23);
        let expected = patch_projection_f32(
            &input,
            patch_count,
            channels,
            patch_size,
            &weight,
            &bias,
            output_width,
        )
        .unwrap();
        let actual = run(
            runtime,
            &invocation(patch_count, input_width, output_width, input, weight, bias),
        );
        assert_comparison(
            &expected,
            &actual,
            &[patch_count, output_width],
            ComparisonAxes::default(),
            gpu_policy(),
        );
    }
}

fn require_model() -> Option<std::path::PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/snapshots")
        .join(MODEL_REVISION)
        .join("model.safetensors");
    if path.is_file() {
        Some(path)
    } else if env_flag("PVLC_REQUIRE_MODEL") {
        panic!("pinned model is required at {}", path.display());
    } else {
        eprintln!(
            "skipping real patch projection: {} is absent",
            path.display()
        );
        None
    }
}

fn real_inputs() -> Option<RealInputs> {
    assert!(GOLDEN_LOCK.contains(
        "bundle_digest = \"blake3:35572ac07da5fddee97becb46fb866f906200f433f4c5677ad20fdf36440acf9\""
    ));
    assert!(GOLDEN_LOCK.contains(
        "semantic_fingerprint = \"blake3:632cbe2de6f47f8f84c764e3a551bf99f352d7459c98d939eb89725357f952b4\""
    ));
    let model = SafetensorsCatalog::open(require_model()?).unwrap();
    assert_eq!(
        model.tensor(MODEL_WEIGHT).unwrap().shape,
        [OUTPUT_WIDTH as u64, 3, 14, 14]
    );
    assert_eq!(
        model.tensor(MODEL_BIAS).unwrap().shape,
        [OUTPUT_WIDTH as u64]
    );
    let weight = model.load_tensor_f32(MODEL_WEIGHT).unwrap();
    let bias = model.load_tensor_f32(MODEL_BIAS).unwrap();
    let golden = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../artifacts/goldens/ocr.clean_latin.0001-l3");
    let processor = SafetensorsCatalog::open(golden.join("processor.safetensors")).unwrap();
    assert_eq!(
        processor.tensor("processor.pixel_values").unwrap().shape,
        [PATCHES as u64, 3, 14, 14]
    );
    let pixels = processor.load_tensor_f32("processor.pixel_values").unwrap();
    let deep = SafetensorsCatalog::open(golden.join("deep-checkpoints.safetensors")).unwrap();
    assert_eq!(
        deep.tensor("vision.embeddings.patch").unwrap().shape,
        [1, PATCHES as u64, OUTPUT_WIDTH as u64]
    );
    let official = deep.load_tensor_f32("vision.embeddings.patch").unwrap();
    Some(RealInputs {
        pixels,
        weight,
        bias,
        official,
    })
}

fn selected_rows(values: &[f32], width: usize, rows: &[usize]) -> Vec<f32> {
    rows.iter()
        .flat_map(|row| values[row * width..(row + 1) * width].iter().copied())
        .collect()
}

#[test]
fn native_real_checkpoint_projection_matches_cpu_and_official_content_patches() {
    let Some(runtime) = runtime() else { return };
    let Some(inputs) = real_inputs() else {
        return;
    };
    let input = selected_rows(&inputs.pixels, INPUT_WIDTH, &SAMPLED_PATCHES);
    assert!(input.iter().any(|value| *value <= -0.95));
    assert!(input.contains(&1.0));
    let expected_cpu = patch_projection_f32(
        &input,
        SAMPLED_PATCHES.len(),
        3,
        14,
        &inputs.weight,
        &inputs.bias,
        OUTPUT_WIDTH,
    )
    .unwrap();
    let expected_official = selected_rows(&inputs.official, OUTPUT_WIDTH, &SAMPLED_PATCHES);
    let actual = run(
        runtime,
        &invocation(
            SAMPLED_PATCHES.len(),
            INPUT_WIDTH,
            OUTPUT_WIDTH,
            input,
            inputs.weight,
            inputs.bias,
        ),
    );
    let axes = ComparisonAxes {
        token_axis: Some(0),
        channel_axis: Some(1),
    };
    assert_comparison(
        &expected_cpu,
        &actual,
        &[SAMPLED_PATCHES.len(), OUTPUT_WIDTH],
        axes,
        gpu_policy(),
    );
    assert_comparison(
        &expected_official,
        &actual,
        &[SAMPLED_PATCHES.len(), OUTPUT_WIDTH],
        axes,
        official_sample_policy(),
    );
}

#[test]
#[ignore = "M3d native hard gate dispatches all 1,276 real patches"]
fn native_full_real_checkpoint_projection_matches_cpu_and_official_l3() {
    let Some(runtime) = runtime() else { return };
    let Some(inputs) = real_inputs() else {
        return;
    };
    let expected_cpu = patch_projection_f32(
        &inputs.pixels,
        PATCHES,
        3,
        14,
        &inputs.weight,
        &inputs.bias,
        OUTPUT_WIDTH,
    )
    .unwrap();
    let actual = run(
        runtime,
        &invocation(
            PATCHES,
            INPUT_WIDTH,
            OUTPUT_WIDTH,
            inputs.pixels,
            inputs.weight,
            inputs.bias,
        ),
    );
    let axes = ComparisonAxes {
        token_axis: Some(0),
        channel_axis: Some(1),
    };
    assert_comparison(
        &expected_cpu,
        &actual,
        &[PATCHES, OUTPUT_WIDTH],
        axes,
        gpu_policy(),
    );
    assert_comparison(
        &inputs.official,
        &actual,
        &[PATCHES, OUTPUT_WIDTH],
        axes,
        official_full_policy(),
    );
}
