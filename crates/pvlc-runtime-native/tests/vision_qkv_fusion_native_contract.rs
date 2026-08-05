use std::{
    env,
    sync::{Mutex, MutexGuard, OnceLock},
};

use pvlc_runtime_core::{
    KernelId, KernelInvocation, VisionQkvFusedInvocation, VisionQkvFusedTargetLimits,
};
use pvlc_runtime_native::{
    ErrorScopeKind, KernelExecution, NativeOptions, NativeRuntime, RuntimeErrorCode,
};

static RUNTIME: OnceLock<Result<NativeRuntime, String>> = OnceLock::new();
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Clone)]
struct Projection {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

#[derive(Clone)]
struct Case {
    tokens: usize,
    input_width: usize,
    output_width: usize,
    input: Vec<f32>,
    projections: [Projection; 3],
}

impl Case {
    fn invocation(&self) -> VisionQkvFusedInvocation<'_> {
        VisionQkvFusedInvocation {
            tokens: self.tokens as u32,
            input_width: self.input_width as u32,
            output_width: self.output_width as u32,
            input: &self.input,
            query_weight: &self.projections[0].weight,
            query_bias: &self.projections[0].bias,
            key_weight: &self.projections[1].weight,
            key_bias: &self.projections[1].bias,
            value_weight: &self.projections[2].weight,
            value_bias: &self.projections[2].bias,
        }
    }

    fn seeded(tokens: usize, input_width: usize, output_width: usize, seed: u32) -> Self {
        let values = |length: usize, salt: u32| {
            (0..length)
                .map(|index| {
                    let integer = ((index as u32 * 37 + salt * 19) % 113) as i32 - 56;
                    integer as f32 / 32.0
                })
                .collect()
        };
        Self {
            tokens,
            input_width,
            output_width,
            input: values(tokens * input_width, seed),
            projections: std::array::from_fn(|projection| Projection {
                weight: values(
                    output_width * input_width,
                    seed + 11 + projection as u32 * 17,
                ),
                bias: values(output_width, seed + 71 + projection as u32 * 23),
            }),
        }
    }

    fn one_hot_transpose_detector() -> Self {
        let tokens = 2;
        let input_width = 3;
        let output_width = 5;
        Self {
            tokens,
            input_width,
            output_width,
            input: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            projections: std::array::from_fn(|projection| Projection {
                weight: (0..output_width)
                    .flat_map(|output| {
                        (0..input_width)
                            .map(move |depth| (1_000 * projection + 100 * output + depth) as f32)
                    })
                    .collect(),
                bias: (0..output_width)
                    .map(|output| (10 * projection + output) as f32)
                    .collect(),
            }),
        }
    }

    fn bias_only_permutation_detector() -> Self {
        let mut case = Self::seeded(3, 2, 5, 101);
        for (projection, parameters) in case.projections.iter_mut().enumerate() {
            parameters.weight.fill(0.0);
            parameters.bias = (0..case.output_width)
                .map(|output| (100 * (projection + 1) + output) as f32)
                .collect();
        }
        case
    }

    fn ordered_cancellation_detector() -> Self {
        Self {
            tokens: 1,
            input_width: 4,
            output_width: 1,
            input: vec![1.0e20, 1.0, -1.0e20, 2.0],
            projections: [
                Projection {
                    weight: vec![1.0; 4],
                    bias: vec![3.0],
                },
                Projection {
                    weight: vec![1.0; 4],
                    bias: vec![-7.0],
                },
                Projection {
                    weight: vec![1.0; 4],
                    bias: vec![11.0],
                },
            ],
        }
    }

    fn empty_geometry(tokens: u32, input_width: u32, output_width: u32) -> Self {
        let empty = || Projection {
            weight: Vec::new(),
            bias: Vec::new(),
        };
        Self {
            tokens: usize::try_from(tokens).unwrap(),
            input_width: usize::try_from(input_width).unwrap(),
            output_width: usize::try_from(output_width).unwrap(),
            input: Vec::new(),
            projections: [empty(), empty(), empty()],
        }
    }
}

fn env_flag(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1" | "true"))
}

fn runtime() -> Option<&'static NativeRuntime> {
    match RUNTIME.get_or_init(|| {
        NativeRuntime::new(NativeOptions::default()).map_err(|error| error.to_string())
    }) {
        Ok(runtime) => Some(runtime),
        Err(error) if env_flag("PVLC_REQUIRE_NATIVE_GPU") => {
            panic!("native GPU is required: {error}")
        }
        Err(error) => {
            eprintln!("skipping fused QKV native contract: {error}");
            None
        }
    }
}

fn target_limits(runtime: &NativeRuntime) -> VisionQkvFusedTargetLimits {
    let capabilities = runtime.capabilities();
    VisionQkvFusedTargetLimits {
        min_storage_buffer_offset_alignment: capabilities.min_storage_buffer_offset_alignment,
        max_storage_buffers_per_shader_stage: capabilities.max_storage_buffers_per_shader_stage,
        max_storage_buffer_binding_size: capabilities.max_storage_buffer_binding_size,
        max_buffer_size: capabilities.max_buffer_size,
        max_compute_workgroups_per_dimension: capabilities.max_compute_workgroups_per_dimension,
    }
}

fn ordered_cpu_projection(case: &Case, projection: usize) -> Vec<f32> {
    let parameters = &case.projections[projection];
    let mut output = vec![0.0; case.tokens * case.output_width];
    for token in 0..case.tokens {
        for channel in 0..case.output_width {
            let mut accumulator = parameters.bias[channel];
            for depth in 0..case.input_width {
                accumulator += case.input[token * case.input_width + depth]
                    * parameters.weight[channel * case.input_width + depth];
            }
            output[token * case.output_width + channel] = accumulator;
        }
    }
    output
}

fn legacy_gpu_projection(runtime: &NativeRuntime, case: &Case, projection: usize) -> Vec<f32> {
    runtime
        .run(&KernelInvocation::VisionPatchProjectionF32 {
            patch_count: case.tokens as u32,
            input_width: case.input_width as u32,
            output_width: case.output_width as u32,
            input: case.input.clone(),
            weight: case.projections[projection].weight.clone(),
            bias: case.projections[projection].bias.clone(),
        })
        .unwrap()
        .values
}

fn plane(values: &[f32], offset_bytes: u64, size_bytes: u64) -> &[f32] {
    let start = usize::try_from(offset_bytes / 4).unwrap();
    let length = usize::try_from(size_bytes / 4).unwrap();
    &values[start..start + length]
}

fn assert_close(expected: &[f32], actual: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        let tolerance = 2.0e-5 * (1.0 + expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context} mismatch at {index}: actual={actual} expected={expected} tolerance={tolerance}"
        );
    }
}

fn bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.iter()
        .zip(right)
        .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn all_semantic_planes_match(
    execution: &KernelExecution,
    plan: &pvlc_runtime_core::VisionQkvFusedPlan,
    expected: &[Vec<f32>; 3],
) -> bool {
    [
        plan.output_layout.query,
        plan.output_layout.key,
        plan.output_layout.value,
    ]
    .into_iter()
    .enumerate()
    .all(|(projection, slice)| {
        plane(&execution.values, slice.offset, slice.size)
            .iter()
            .zip(&expected[projection])
            .all(|(&actual, &expected): (&f32, &f32)| {
                (actual - expected).abs() <= 2.0e-5 * (1.0 + expected.abs())
            })
    })
}

fn assert_diagnostics(
    runtime: &NativeRuntime,
    before_submissions: u64,
    execution: &KernelExecution,
    source: &str,
) {
    assert_eq!(runtime.counters().submissions - before_submissions, 1);
    assert_eq!(execution.diagnostics.kernel, KernelId::VisionQkvFusedF32);
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
    assert!(
        execution.diagnostics.timestamp.is_none(),
        "fused single-submission execution must not expose shared or stale timestamp diagnostics"
    );
    assert_eq!(
        execution.diagnostics.shader_blake3,
        *blake3::hash(source.as_bytes()).as_bytes()
    );
}

fn run_fused(
    runtime: &NativeRuntime,
    case: &Case,
) -> Option<(KernelExecution, pvlc_runtime_core::VisionQkvFusedPlan)> {
    if runtime.capabilities().max_storage_buffers_per_shader_stage < 8 {
        return None;
    }
    let invocation = case.invocation();
    let plan = invocation.plan(target_limits(runtime)).unwrap();
    let source = pvlc_wgsl::module(KernelId::VisionQkvFusedF32)
        .unwrap()
        .source;
    let before = runtime.counters().submissions;
    let execution = runtime.run_vision_qkv_fused(&invocation).unwrap();
    assert_diagnostics(runtime, before, &execution, source);
    assert_ne!(
        execution.diagnostics.shader_blake3,
        *blake3::hash(
            pvlc_wgsl::module(KernelId::VisionPatchProjectionF32)
                .unwrap()
                .source
                .as_bytes()
        )
        .as_bytes()
    );
    assert_eq!(execution.values.len(), plan.invocation.output_elements);
    Some((execution, plan))
}

#[test]
fn fused_timestamp_is_always_none_and_the_next_legacy_timestamp_is_fresh() {
    let _serial = serial();
    let Some(runtime) = runtime() else { return };
    if runtime.capabilities().max_storage_buffers_per_shader_stage < 8 {
        return;
    }

    let legacy = KernelInvocation::SiluF32 {
        values: vec![0.25; 4_096],
    };
    let legacy_before = runtime.run(&legacy).unwrap();

    let case = Case::seeded(3, 3, 5, 211);
    let source = pvlc_wgsl::module(KernelId::VisionQkvFusedF32)
        .unwrap()
        .source;
    let before_fused_submissions = runtime.counters().submissions;
    let fused = runtime.run_vision_qkv_fused(&case.invocation()).unwrap();
    assert_diagnostics(runtime, before_fused_submissions, &fused, source);
    assert!(
        fused.diagnostics.timestamp.is_none(),
        "fused timestamp suppression is unconditional, including timestamp-capable adapters"
    );

    let legacy_after = runtime.run(&legacy).unwrap();
    if runtime.capabilities().timestamp_query {
        let before = legacy_before
            .diagnostics
            .timestamp
            .expect("legacy RequireFresh must timestamp on a capable adapter");
        let after = legacy_after
            .diagnostics
            .timestamp
            .expect("legacy RequireFresh must recover immediately after fused execution");
        for timestamp in [before, after] {
            assert!(timestamp.end_ticks > timestamp.begin_ticks);
            assert!(timestamp.period_ns.is_finite() && timestamp.period_ns > 0.0);
            assert!(timestamp.duration_ns.is_finite() && timestamp.duration_ns > 0.0);
        }
        assert_ne!(
            (before.begin_ticks, before.end_ticks),
            (after.begin_ticks, after.end_ticks),
            "legacy RequireFresh returned the pre-fused timestamp pair"
        );
    } else {
        assert!(legacy_before.diagnostics.timestamp.is_none());
        assert!(legacy_after.diagnostics.timestamp.is_none());
    }
}

#[test]
fn fused_qkv_matches_independent_ordered_cpu_and_three_legacy_gpu_projections() {
    let _serial = serial();
    let Some(runtime) = runtime() else { return };
    if runtime.capabilities().max_storage_buffers_per_shader_stage < 8 {
        return;
    }
    let cases = [
        Case::seeded(7, 5, 9, 1),
        Case::seeded(9, 7, 5, 2),
        Case::one_hot_transpose_detector(),
        Case::bias_only_permutation_detector(),
        Case::ordered_cancellation_detector(),
    ];
    for case in cases {
        let arithmetic_budget = 3 * case.tokens * case.output_width * case.input_width;
        assert!(
            arithmetic_budget <= 10_000,
            "the independent CPU oracle must remain a bounded correctness gate"
        );
        let expected: [Vec<f32>; 3] =
            std::array::from_fn(|projection| ordered_cpu_projection(&case, projection));
        let legacy: [Vec<f32>; 3] =
            std::array::from_fn(|projection| legacy_gpu_projection(runtime, &case, projection));
        let (execution, plan) = run_fused(runtime, &case).unwrap();
        let layout = plan.output_layout;
        for (projection, slice) in [layout.query, layout.key, layout.value]
            .into_iter()
            .enumerate()
        {
            let fused = plane(&execution.values, slice.offset, slice.size);
            assert_close(&expected[projection], fused, "fused versus independent CPU");
            assert_close(&legacy[projection], fused, "fused versus legacy GPU");
            if bits_equal(&legacy[projection], &expected[projection]) {
                assert!(
                    bits_equal(fused, &legacy[projection]),
                    "when the backend preserves the ordered CPU arithmetic, fused and legacy GPU results must be bitwise equal"
                );
            }
        }
    }
}

fn assert_padding_bits(
    values: &[f32],
    layout: pvlc_runtime_core::VisionQkvFusedOutputLayout,
    expected_bits: u32,
) {
    for (start, end) in [
        (layout.query.offset + layout.query.size, layout.key.offset),
        (layout.key.offset + layout.key.size, layout.value.offset),
        (
            layout.value.offset + layout.value.size,
            layout.physical_bytes,
        ),
    ] {
        for value in plane(values, start, end - start) {
            assert_eq!(value.to_bits(), expected_bits, "padding was modified");
        }
    }
}

fn assert_native_rejection_has_no_effects(runtime: &NativeRuntime, label: &str, case: &Case) {
    let before = runtime.counters();
    let error = runtime
        .run_vision_qkv_fused(&case.invocation())
        .expect_err(label);
    assert_eq!(error.code(), RuntimeErrorCode::InvalidInvocation, "{label}");
    assert_eq!(
        runtime.counters(),
        before,
        "{label} reached native allocation or submission before validation"
    );
}

#[test]
fn actual_adapter_limits_gate_before_effects_and_padding_preserves_zero_and_canary_bits() {
    let _serial = serial();
    let Some(runtime) = runtime() else { return };
    let case = Case::seeded(3, 3, 5, 31);
    if runtime.capabilities().max_storage_buffers_per_shader_stage < 8 {
        assert_native_rejection_has_no_effects(
            runtime,
            "actual adapter has fewer than 8 slots",
            &case,
        );
        return;
    }

    let mut long = case.clone();
    long.input.push(0.0);
    let mut nonfinite = case.clone();
    nonfinite.projections[1].weight[nonfinite.input_width] = f32::NAN;
    let dispatch_limit = runtime.capabilities().max_compute_workgroups_per_dimension;
    let dispatch_overflow_dimension = dispatch_limit
        .checked_mul(8)
        .and_then(|dimension| dimension.checked_add(1))
        .expect("portable WebGPU dispatch limit must leave a representable negative boundary");
    let invalid_cases = [
        ("zero dimension", Case::empty_geometry(0, 3, 5)),
        ("long operand", long),
        ("nonfinite operand", nonfinite),
        (
            "host arithmetic overflow",
            Case::empty_geometry(u32::MAX, u32::MAX, u32::MAX),
        ),
        (
            "WGSL u32 address overflow",
            Case::empty_geometry(65_536, 1, 65_536),
        ),
        (
            "dispatch x overflow",
            Case::empty_geometry(1, 1, dispatch_overflow_dimension),
        ),
        (
            "dispatch y overflow",
            Case::empty_geometry(dispatch_overflow_dimension, 1, 1),
        ),
    ];
    for (label, invalid) in &invalid_cases {
        assert_native_rejection_has_no_effects(runtime, label, invalid);
    }

    let (zeroed, plan) = run_fused(runtime, &case).unwrap();
    assert_padding_bits(&zeroed.values, plan.output_layout, 0.0_f32.to_bits());
    if plan.output_layout.plane_stride_bytes == plan.output_layout.plane_bytes {
        return;
    }

    const CANARY_BITS: u32 = 0xC2F6_E979;
    let source = pvlc_wgsl::module(KernelId::VisionQkvFusedF32)
        .unwrap()
        .source;
    let before = runtime.counters().submissions;
    let execution = runtime
        .run_vision_qkv_fused_with_shader(
            &case.invocation(),
            "vision-qkv-canary",
            source,
            "main",
            CANARY_BITS,
        )
        .unwrap();
    assert_diagnostics(runtime, before, &execution, source);
    assert_padding_bits(&execution.values, plan.output_layout, CANARY_BITS);
}

const EXECUTABLE_BASELINE_WGSL: &str = r#"struct F32Buffer { data: array<f32>, }
struct Params {
    tokens: u32,
    input_width: u32,
    output_width: u32,
    plane_stride_elements: u32,
}
@group(0) @binding(0) var<storage, read> input: F32Buffer;
@group(0) @binding(1) var<storage, read> query_weight: F32Buffer;
@group(0) @binding(2) var<storage, read> query_bias: F32Buffer;
@group(0) @binding(3) var<storage, read> key_weight: F32Buffer;
@group(0) @binding(4) var<storage, read> key_bias: F32Buffer;
@group(0) @binding(5) var<storage, read> value_weight: F32Buffer;
@group(0) @binding(6) var<storage, read> value_bias: F32Buffer;
@group(0) @binding(7) var<storage, read_write> output: F32Buffer;
@group(0) @binding(8) var<uniform> params: Params;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let channel = global_id.x;
    let token = global_id.y;
    let projection = global_id.z;
    if token >= params.tokens || channel >= params.output_width || projection >= 3u { return; }
    var accumulator = value_bias.data[channel];
    if projection == 0u {
        accumulator = query_bias.data[channel];
    } else if projection == 1u {
        accumulator = key_bias.data[channel];
    }
    for (var depth = 0u; depth < params.input_width; depth = depth + 1u) {
        var coefficient = value_weight.data[channel * params.input_width + depth];
        if projection == 0u {
            coefficient = query_weight.data[channel * params.input_width + depth];
        } else if projection == 1u {
            coefficient = key_weight.data[channel * params.input_width + depth];
        }
        let input_value = input.data[token * params.input_width + depth];
        accumulator = accumulator + input_value * coefficient;
    }
    let index = projection * params.plane_stride_elements + token * params.output_width + channel;
    output.data[index] = accumulator;
}
"#;

fn replace_once(source: &str, from: &str, to: &str) -> String {
    assert_eq!(source.matches(from).count(), 1);
    source.replacen(from, to, 1)
}

fn run_shader_override(
    runtime: &NativeRuntime,
    case: &Case,
    label: &str,
    source: &str,
) -> KernelExecution {
    let before = runtime.counters().submissions;
    let execution = runtime
        .run_vision_qkv_fused_with_shader(
            &case.invocation(),
            label,
            source,
            "main",
            0.0_f32.to_bits(),
        )
        .unwrap();
    assert_diagnostics(runtime, before, &execution, source);
    execution
}

#[test]
fn hidden_shader_hook_executes_supplied_source_and_kills_routing_and_ordered_sum_mutants() {
    let _serial = serial();
    let Some(runtime) = runtime() else { return };
    if runtime.capabilities().max_storage_buffers_per_shader_stage < 8 {
        return;
    }
    let case = Case::one_hot_transpose_detector();
    let plan = case.invocation().plan(target_limits(runtime)).unwrap();
    let expected: [Vec<f32>; 3] =
        std::array::from_fn(|projection| ordered_cpu_projection(&case, projection));

    let baseline = run_shader_override(
        runtime,
        &case,
        "vision-qkv-independent-baseline",
        EXECUTABLE_BASELINE_WGSL,
    );
    for (projection, slice) in [
        plan.output_layout.query,
        plan.output_layout.key,
        plan.output_layout.value,
    ]
    .into_iter()
    .enumerate()
    {
        assert_close(
            &expected[projection],
            plane(&baseline.values, slice.offset, slice.size),
            "independently authored executable baseline",
        );
    }

    let routing_mutant = replace_once(
        EXECUTABLE_BASELINE_WGSL,
        "if projection == 0u {\n        accumulator = query_bias.data[channel];",
        "if projection == 2u {\n        accumulator = query_bias.data[channel];",
    );
    let arithmetic_mutant = replace_once(
        EXECUTABLE_BASELINE_WGSL,
        "accumulator = accumulator + input_value * coefficient;",
        "accumulator = input_value * coefficient;",
    );
    for (label, source) in [
        ("vision-qkv-routing-mutant", routing_mutant),
        ("vision-qkv-arithmetic-mutant", arithmetic_mutant),
    ] {
        let mutant = run_shader_override(runtime, &case, label, &source);
        assert_ne!(mutant.values, baseline.values, "mutant was CPU-shadowed");
        assert!(
            !all_semantic_planes_match(&mutant, &plan, &expected),
            "{label} unexpectedly matched all three independent CPU planes"
        );
        assert_ne!(
            mutant.diagnostics.shader_blake3, baseline.diagnostics.shader_blake3,
            "diagnostics did not identify the executed mutant"
        );
    }

    let cancellation = Case::ordered_cancellation_detector();
    let cancellation_plan = cancellation
        .invocation()
        .plan(target_limits(runtime))
        .unwrap();
    let cancellation_expected: [Vec<f32>; 3] =
        std::array::from_fn(|projection| ordered_cpu_projection(&cancellation, projection));
    let cancellation_baseline = run_shader_override(
        runtime,
        &cancellation,
        "vision-qkv-cancellation-baseline",
        EXECUTABLE_BASELINE_WGSL,
    );
    assert!(all_semantic_planes_match(
        &cancellation_baseline,
        &cancellation_plan,
        &cancellation_expected
    ));
    let reverse_depth_mutant = replace_once(
        EXECUTABLE_BASELINE_WGSL,
        "for (var depth = 0u; depth < params.input_width; depth = depth + 1u) {",
        "for (var reverse_depth = params.input_width; reverse_depth > 0u; reverse_depth = reverse_depth - 1u) {\n        let depth = reverse_depth - 1u;",
    );
    let reverse = run_shader_override(
        runtime,
        &cancellation,
        "vision-qkv-reverse-depth-mutant",
        &reverse_depth_mutant,
    );
    assert!(
        !all_semantic_planes_match(&reverse, &cancellation_plan, &cancellation_expected),
        "reverse-depth mutant unexpectedly preserved ordered CPU semantics"
    );
}
