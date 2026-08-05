use std::{
    env,
    sync::{Arc, Mutex, MutexGuard, OnceLock},
};

use pvlc_cpu_ref::{
    apply_rope_neox, gelu_pytorch_tanh, gemm_f32, layer_norm_f32, rms_norm_f32, silu,
};
use pvlc_runtime_core::{KernelId, KernelInvocation};
use pvlc_runtime_native::{
    BackendKind, ErrorScopeDriver, ErrorScopeKind, NativeOptions, NativeRuntime, RuntimeErrorCode,
    RuntimeEvent, RuntimeObserver, drive_error_scopes,
};
use pvlc_testkit::{ComparisonAxes, ComparisonPolicy, compare_f32};

static RUNTIME: OnceLock<Result<NativeRuntime, String>> = OnceLock::new();
static SERIAL: Mutex<()> = Mutex::new(());
const BOUNDARIES: [u32; 25] = [
    1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513,
    1023, 1024,
];

#[derive(Default)]
struct RecordingObserver {
    events: Mutex<Vec<RuntimeEvent>>,
}

impl RuntimeObserver for RecordingObserver {
    fn on_event(&self, event: RuntimeEvent) {
        self.events.lock().unwrap().push(event);
    }
}

impl RecordingObserver {
    fn take(&self) -> Vec<RuntimeEvent> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }
}

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn env_flag(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1" | "true"))
}

fn hardware_required() -> bool {
    env_flag("PVLC_REQUIRE_NATIVE_GPU")
}

fn runtime() -> Option<&'static NativeRuntime> {
    match RUNTIME.get_or_init(|| {
        NativeRuntime::new(NativeOptions::default()).map_err(|error| error.to_string())
    }) {
        Ok(runtime) => Some(runtime),
        Err(error) if hardware_required() => {
            panic!("PVLC_REQUIRE_NATIVE_GPU is set but no native WebGPU adapter is usable: {error}")
        }
        Err(error) => {
            eprintln!("skipping native GPU contract because no adapter is available: {error}");
            None
        }
    }
}

fn values(length: usize, seed: u32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let value = ((index as u32 * 37 + seed * 17) % 101) as i32 - 50;
            value as f32 / 32.0
        })
        .collect()
}

fn scalar_input_families(length: usize) -> Vec<(&'static str, Vec<f32>)> {
    let mut impulse = vec![0.0; length];
    impulse[length / 2] = 1.0;
    vec![
        ("zeros", vec![0.0; length]),
        ("ones", vec![1.0; length]),
        (
            "alternating-signs",
            (0..length)
                .map(|index| if index.is_multiple_of(2) { -1.0 } else { 1.0 })
                .collect(),
        ),
        ("tiny", vec![1.0e-30; length]),
        (
            "near-fp16-limit",
            (0..length)
                .map(|index| {
                    if index.is_multiple_of(2) {
                        -65_504.0
                    } else {
                        65_504.0
                    }
                })
                .collect(),
        ),
        ("impulse", impulse),
        (
            "repeated-pattern",
            (0..length)
                .map(|index| [0.25, -0.5, 1.0][index % 3])
                .collect(),
        ),
    ]
}

fn policy(max_abs: f64, relative_l2: f64) -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs,
        max_mean_abs: max_abs,
        max_p99_abs: max_abs,
        max_relative_l2: relative_l2,
        min_cosine_similarity: 0.999_99,
        max_per_token_relative_l2: None,
        max_per_channel_relative_l2: None,
    }
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

fn assert_gpu_close(
    reference: &[f32],
    actual: &[f32],
    shape: &[usize],
    tolerance: ComparisonPolicy,
) {
    let report = compare_f32(reference, actual, shape, ComparisonAxes::default()).unwrap();
    let verdict = report.assess(&tolerance).unwrap();
    assert!(
        verdict.passed(),
        "GPU comparison failed: {report:?}; violations: {:?}",
        verdict.violations()
    );
}

fn run(
    runtime: &NativeRuntime,
    invocation: KernelInvocation,
) -> pvlc_runtime_native::KernelExecution {
    let expected_kernel = invocation.kernel_id();
    let execution = runtime.run(&invocation).unwrap();
    assert_eq!(execution.diagnostics.kernel, expected_kernel);
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
    execution
}

#[test]
fn native_context_is_the_expected_backend_and_all_pipelines_validate() {
    let _serial = serial();
    let Some(runtime) = runtime() else { return };
    let capabilities = runtime.capabilities();
    assert!(!capabilities.adapter_name.is_empty());
    assert!(capabilities.max_storage_buffer_binding_size >= 128 * 1024 * 1024);
    assert!(capabilities.max_compute_workgroups_per_dimension >= 65_535);
    if env_flag("PVLC_REQUIRE_M4_METAL") {
        assert_eq!(capabilities.backend, BackendKind::Metal);
        assert!(capabilities.adapter_name.contains("M4 Pro"));
    }

    let report = runtime.validate_all_pipelines().unwrap();
    assert_eq!(
        report.validated_kernels.as_slice(),
        KernelId::ALL.as_slice()
    );
    assert!(report.captured_errors.is_empty());
}

#[test]
fn gemm_and_dedicated_gemv_match_the_cpu_oracle_around_tile_boundaries() {
    let _serial = serial();
    let Some(runtime) = runtime() else { return };
    let mut case = 0_u32;
    for axis in 0..3 {
        for edge in BOUNDARIES {
            let (rows, inner, columns) = match axis {
                0 => (edge, 7, 9),
                1 => (7, edge, 9),
                2 => (7, 9, edge),
                _ => unreachable!(),
            };
            let left = values((rows * inner) as usize, case + 1);
            let right = values((inner * columns) as usize, case + 11);
            let expected = gemm_f32(
                &left,
                rows as usize,
                inner as usize,
                &right,
                columns as usize,
            )
            .unwrap();
            let execution = run(
                runtime,
                KernelInvocation::GemmF32 {
                    rows,
                    inner,
                    columns,
                    left,
                    right,
                },
            );
            assert_gpu_close(
                &expected,
                &execution.values,
                &[rows as usize, columns as usize],
                policy(5.0e-5, 2.0e-5),
            );
            case += 1;
        }
    }

    for (case, rows) in BOUNDARIES.into_iter().enumerate() {
        let columns = [1_u32, 3, 17, 33][case % 4];
        let matrix = values((rows * columns) as usize, case as u32 + 19);
        let vector = values(columns as usize, case as u32 + 29);
        let expected = gemm_f32(&matrix, rows as usize, columns as usize, &vector, 1).unwrap();
        let execution = run(
            runtime,
            KernelInvocation::GemvF32 {
                rows,
                columns,
                matrix,
                vector,
            },
        );
        assert_gpu_close(
            &expected,
            &execution.values,
            &[rows as usize],
            policy(5.0e-5, 2.0e-5),
        );
    }
    for (case, columns) in BOUNDARIES.into_iter().enumerate() {
        let rows = 7;
        let matrix = values((rows * columns) as usize, case as u32 + 131);
        let vector = values(columns as usize, case as u32 + 149);
        let expected = gemm_f32(&matrix, rows as usize, columns as usize, &vector, 1).unwrap();
        let execution = run(
            runtime,
            KernelInvocation::GemvF32 {
                rows,
                columns,
                matrix,
                vector,
            },
        );
        assert_gpu_close(
            &expected,
            &execution.values,
            &[rows as usize],
            policy(8.0e-5, 3.0e-5),
        );
    }
}

#[test]
fn gemm_covers_zero_one_sign_tiny_fp16_impulse_and_repeated_row_inputs() {
    let _serial = serial();
    let Some(runtime) = runtime() else { return };
    let rows = 4_u32;
    let inner = 65_u32;
    let columns = 3_u32;
    let mut impulse_left = vec![0.0; (rows * inner) as usize];
    impulse_left[(2 * inner + 31) as usize] = 1.0;
    let mut impulse_right = vec![0.0; (inner * columns) as usize];
    impulse_right[(31 * columns + 1) as usize] = 1.0;
    let repeated_row = values(inner as usize, 211);
    let mut repeated_rows = Vec::with_capacity((rows * inner) as usize);
    for _ in 0..rows {
        repeated_rows.extend_from_slice(&repeated_row);
    }
    let cases = [
        (
            "zeros",
            vec![0.0; (rows * inner) as usize],
            vec![1.0; (inner * columns) as usize],
        ),
        (
            "ones",
            vec![1.0; (rows * inner) as usize],
            vec![1.0; (inner * columns) as usize],
        ),
        (
            "alternating-signs",
            (0..rows * inner)
                .map(|index| if index.is_multiple_of(2) { -1.0 } else { 1.0 })
                .collect(),
            values((inner * columns) as usize, 223),
        ),
        (
            "tiny",
            vec![1.0e-30; (rows * inner) as usize],
            vec![-1.0e-5; (inner * columns) as usize],
        ),
        (
            "near-fp16-limit",
            (0..rows * inner)
                .map(|index| {
                    if index.is_multiple_of(2) {
                        -65_504.0
                    } else {
                        65_504.0
                    }
                })
                .collect(),
            vec![1.0e-4; (inner * columns) as usize],
        ),
        ("impulse", impulse_left, impulse_right),
        (
            "repeated-rows",
            repeated_rows,
            values((inner * columns) as usize, 227),
        ),
    ];

    for (name, left, right) in cases {
        let expected = gemm_f32(
            &left,
            rows as usize,
            inner as usize,
            &right,
            columns as usize,
        )
        .unwrap();
        let actual = run(
            runtime,
            KernelInvocation::GemmF32 {
                rows,
                inner,
                columns,
                left,
                right,
            },
        );
        let report = compare_f32(
            &expected,
            &actual.values,
            &[rows as usize, columns as usize],
            ComparisonAxes::default(),
        )
        .unwrap();
        assert!(
            report.assess(&policy(0.1, 3.0e-5)).unwrap().passed(),
            "input family {name} failed: {report:?}"
        );
    }
}

#[test]
fn gemv_covers_each_required_input_family_in_matrix_and_vector_operands() {
    let _serial = serial();
    let Some(runtime) = runtime() else { return };
    let rows = 4_u32;
    let columns = 65_u32;
    let fixed_matrix = values((rows * columns) as usize, 229);
    let fixed_vector = values(columns as usize, 233);

    let mut matrix_families = scalar_input_families((rows * columns) as usize);
    let repeated_row = values(columns as usize, 239);
    let mut repeated_rows = Vec::with_capacity((rows * columns) as usize);
    for _ in 0..rows {
        repeated_rows.extend_from_slice(&repeated_row);
    }
    matrix_families.push(("repeated-rows", repeated_rows));

    for (name, matrix) in matrix_families {
        let expected =
            gemm_f32(&matrix, rows as usize, columns as usize, &fixed_vector, 1).unwrap();
        let actual = run(
            runtime,
            KernelInvocation::GemvF32 {
                rows,
                columns,
                matrix,
                vector: fixed_vector.clone(),
            },
        );
        let report = compare_f32(
            &expected,
            &actual.values,
            &[rows as usize],
            ComparisonAxes::default(),
        )
        .unwrap();
        assert!(
            report.assess(&policy(1.0, 5.0e-5)).unwrap().passed(),
            "GEMV matrix family {name} failed: {report:?}"
        );
    }

    for (name, vector) in scalar_input_families(columns as usize) {
        let expected =
            gemm_f32(&fixed_matrix, rows as usize, columns as usize, &vector, 1).unwrap();
        let actual = run(
            runtime,
            KernelInvocation::GemvF32 {
                rows,
                columns,
                matrix: fixed_matrix.clone(),
                vector,
            },
        );
        let report = compare_f32(
            &expected,
            &actual.values,
            &[rows as usize],
            ComparisonAxes::default(),
        )
        .unwrap();
        assert!(
            report.assess(&policy(1.0, 5.0e-5)).unwrap().passed(),
            "GEMV vector family {name} failed: {report:?}"
        );
    }
}

#[test]
fn layer_and_rms_norm_match_cpu_on_width_boundaries_and_input_families() {
    let _serial = serial();
    let Some(runtime) = runtime() else { return };
    for (case, width) in BOUNDARIES.into_iter().enumerate() {
        let rows = 3_u32;
        let mut input = values((rows * width) as usize, case as u32 + 41);
        match case % 7 {
            0 => input.fill(0.0),
            1 => input.fill(1.0),
            2 => {
                for (index, value) in input.iter_mut().enumerate() {
                    *value = if index.is_multiple_of(2) { -0.25 } else { 0.25 };
                }
            }
            3 => input.fill(1.0e-30),
            4 => input.fill(65_504.0),
            5 => {
                input.fill(0.0);
                input[width as usize / 2] = 1.0;
            }
            6 => {
                let row = values(width as usize, case as u32 + 43);
                for target in input.chunks_exact_mut(width as usize) {
                    target.copy_from_slice(&row);
                }
            }
            _ => {}
        }
        let weight = values(width as usize, case as u32 + 47)
            .into_iter()
            .map(|value| 1.0 + value * 0.1)
            .collect::<Vec<_>>();
        let bias = values(width as usize, case as u32 + 53)
            .into_iter()
            .map(|value| value * 0.05)
            .collect::<Vec<_>>();
        let epsilon = [1.0e-6, 1.0e-5, 1.0e-3, 0.5][case % 4];

        let expected_layer = layer_norm_f32(
            &input,
            rows as usize,
            width as usize,
            &weight,
            &bias,
            epsilon,
        )
        .unwrap();
        let layer = run(
            runtime,
            KernelInvocation::LayerNormF32 {
                rows,
                width,
                input: input.clone(),
                weight: weight.clone(),
                bias,
                epsilon,
            },
        );
        assert_gpu_close(
            &expected_layer,
            &layer.values,
            &[rows as usize, width as usize],
            policy(2.0e-4, 1.0e-4),
        );

        let expected_rms =
            rms_norm_f32(&input, rows as usize, width as usize, &weight, epsilon).unwrap();
        let rms = run(
            runtime,
            KernelInvocation::RmsNormF32 {
                rows,
                width,
                input,
                weight,
                epsilon,
            },
        );
        assert_gpu_close(
            &expected_rms,
            &rms.values,
            &[rows as usize, width as usize],
            policy(2.0e-4, 1.0e-4),
        );
    }

    for (case, rows) in BOUNDARIES.into_iter().enumerate() {
        let width = 17;
        let input = values((rows * width) as usize, case as u32 + 157);
        let weight = values(width as usize, case as u32 + 163)
            .into_iter()
            .map(|value| 1.0 + value * 0.1)
            .collect::<Vec<_>>();
        let bias = vec![0.125; width as usize];
        let epsilon = [1.0e-6, 1.0e-4, 0.25][case % 3];
        let expected_layer = layer_norm_f32(
            &input,
            rows as usize,
            width as usize,
            &weight,
            &bias,
            epsilon,
        )
        .unwrap();
        let layer = run(
            runtime,
            KernelInvocation::LayerNormF32 {
                rows,
                width,
                input: input.clone(),
                weight: weight.clone(),
                bias,
                epsilon,
            },
        );
        assert_gpu_close(
            &expected_layer,
            &layer.values,
            &[rows as usize, width as usize],
            policy(2.0e-4, 1.0e-4),
        );

        let expected_rms =
            rms_norm_f32(&input, rows as usize, width as usize, &weight, epsilon).unwrap();
        let rms = run(
            runtime,
            KernelInvocation::RmsNormF32 {
                rows,
                width,
                input,
                weight,
                epsilon,
            },
        );
        assert_gpu_close(
            &expected_rms,
            &rms.values,
            &[rows as usize, width as usize],
            policy(2.0e-4, 1.0e-4),
        );
    }
}

#[test]
fn activation_and_rope_kernels_match_cpu_for_extremes_positions_and_tail_values() {
    let _serial = serial();
    let Some(runtime) = runtime() else { return };
    for (case, length) in BOUNDARIES
        .into_iter()
        .map(|value| value as usize)
        .enumerate()
    {
        let mut input = values(length, case as u32 + 71);
        if length >= 7 {
            input[..7].copy_from_slice(&[-65_504.0, -20.0, -1.0e-7, 0.0, 1.0e-7, 20.0, 65_504.0]);
        }

        let expected_silu: Vec<_> = input.iter().copied().map(silu).collect();
        let actual_silu = run(
            runtime,
            KernelInvocation::SiluF32 {
                values: input.clone(),
            },
        );
        assert_gpu_close(
            &expected_silu,
            &actual_silu.values,
            &[length],
            policy(2.0e-5, 2.0e-5),
        );

        let expected_gelu: Vec<_> = input.iter().copied().map(gelu_pytorch_tanh).collect();
        let actual_gelu = run(runtime, KernelInvocation::GeluTanhF32 { values: input });
        assert_gpu_close(
            &expected_gelu,
            &actual_gelu.values,
            &[length],
            policy(3.0e-5, 2.0e-5),
        );
    }

    for (name, input) in scalar_input_families(65) {
        let expected_silu: Vec<_> = input.iter().copied().map(silu).collect();
        let actual_silu = run(
            runtime,
            KernelInvocation::SiluF32 {
                values: input.clone(),
            },
        );
        let silu_report = compare_f32(
            &expected_silu,
            &actual_silu.values,
            &[input.len()],
            ComparisonAxes::default(),
        )
        .unwrap();
        assert!(
            silu_report.assess(&policy(0.05, 5.0e-5)).unwrap().passed(),
            "SiLU family {name} failed: {silu_report:?}"
        );

        let expected_gelu: Vec<_> = input.iter().copied().map(gelu_pytorch_tanh).collect();
        let actual_gelu = run(runtime, KernelInvocation::GeluTanhF32 { values: input });
        let gelu_report = compare_f32(
            &expected_gelu,
            &actual_gelu.values,
            &[expected_gelu.len()],
            ComparisonAxes::default(),
        )
        .unwrap();
        assert!(
            gelu_report.assess(&policy(0.05, 5.0e-5)).unwrap().passed(),
            "GELU family {name} failed: {gelu_report:?}"
        );
    }

    for (case, rotary_dim) in [2_u32, 4, 8, 16, 32, 64].into_iter().enumerate() {
        let rows = 4_u32;
        let width = rotary_dim + 3;
        let positions = vec![0, 1, 17, 4096];
        let input = values((rows * width) as usize, case as u32 + 89);
        let mut expected = input.clone();
        let base = [2.0, 10_000.0, 500_000.0, 1_000_000.0][case % 4];
        apply_rope_neox(
            &mut expected,
            rows as usize,
            width as usize,
            rotary_dim as usize,
            &positions,
            base,
        )
        .unwrap();
        let actual = run(
            runtime,
            KernelInvocation::RopeNeoxF32 {
                rows,
                width,
                rotary_dim,
                positions,
                base,
                values: input.clone(),
            },
        );
        assert_gpu_close(
            &expected,
            &actual.values,
            &[rows as usize, width as usize],
            policy(3.0e-4, 2.0e-4),
        );
        assert_eq!(
            &actual.values[..width as usize],
            &input[..width as usize],
            "the complete position-zero row must be bit-exact"
        );
        for row in 0..rows as usize {
            let start = row * width as usize + rotary_dim as usize;
            let end = (row + 1) * width as usize;
            assert_eq!(
                &actual.values[start..end],
                &input[start..end],
                "every untouched RoPE tail must be bit-exact"
            );
        }
    }

    for (case, rows) in BOUNDARIES.into_iter().enumerate() {
        let width = 5;
        let rotary_dim = 2;
        let positions: Vec<_> = (0..rows).map(|row| row * 7).collect();
        let input = values((rows * width) as usize, case as u32 + 181);
        let base = [2.0, 10_000.0, 500_000.0][case % 3];
        let mut expected = input.clone();
        apply_rope_neox(
            &mut expected,
            rows as usize,
            width as usize,
            rotary_dim as usize,
            &positions,
            base,
        )
        .unwrap();
        let actual = run(
            runtime,
            KernelInvocation::RopeNeoxF32 {
                rows,
                width,
                rotary_dim,
                positions,
                base,
                values: input.clone(),
            },
        );
        assert_gpu_close(
            &expected,
            &actual.values,
            &[rows as usize, width as usize],
            policy(3.0e-4, 2.0e-4),
        );
        for row in 0..rows as usize {
            let start = row * width as usize + rotary_dim as usize;
            let end = (row + 1) * width as usize;
            assert_eq!(&actual.values[start..end], &input[start..end]);
        }
    }

    let rows = 4_u32;
    let width = 10_u32;
    let rotary_dim = 8_u32;
    let positions = vec![0, 1, 17, 127];
    let mut rope_families = scalar_input_families((rows * width) as usize);
    let repeated_row = values(width as usize, 251);
    let mut repeated_rows = Vec::with_capacity((rows * width) as usize);
    for _ in 0..rows {
        repeated_rows.extend_from_slice(&repeated_row);
    }
    rope_families.push(("repeated-rows", repeated_rows));
    for (case, (name, input)) in rope_families.into_iter().enumerate() {
        let base = [2.0, 10_000.0, 500_000.0][case % 3];
        let mut expected = input.clone();
        apply_rope_neox(
            &mut expected,
            rows as usize,
            width as usize,
            rotary_dim as usize,
            &positions,
            base,
        )
        .unwrap();
        let actual = run(
            runtime,
            KernelInvocation::RopeNeoxF32 {
                rows,
                width,
                rotary_dim,
                positions: positions.clone(),
                base,
                values: input.clone(),
            },
        );
        let report = compare_f32(
            &expected,
            &actual.values,
            &[rows as usize, width as usize],
            ComparisonAxes::default(),
        )
        .unwrap();
        assert!(
            report.assess(&policy(1.0, 5.0e-5)).unwrap().passed(),
            "RoPE family {name} failed: {report:?}"
        );
        assert_eq!(&actual.values[..width as usize], &input[..width as usize]);
        for row in 0..rows as usize {
            let start = row * width as usize + rotary_dim as usize;
            let end = (row + 1) * width as usize;
            assert_eq!(&actual.values[start..end], &input[start..end]);
        }
    }
}

#[test]
fn gelu_crosses_the_maximum_one_dimensional_dispatch_without_gaps_or_aliasing() {
    let _serial = serial();
    let Some(runtime) = runtime() else { return };
    const LENGTH: usize = 65_535 * 64 + 1;
    const ROW_STRIDE: usize = 32_768 * 64;
    let sentinels = [
        (0, -3.0_f32),
        (ROW_STRIDE - 1, -1.0),
        (ROW_STRIDE, 0.5),
        (ROW_STRIDE + 1, 2.0),
        (LENGTH - 1, 4.0),
    ];
    let mut input = vec![0.0; LENGTH];
    for &(index, value) in &sentinels {
        input[index] = value;
    }
    let execution = run(runtime, KernelInvocation::GeluTanhF32 { values: input });
    assert_eq!(execution.values.len(), LENGTH);
    for (index, value) in sentinels {
        let expected = gelu_pytorch_tanh(value);
        assert!(
            (execution.values[index] - expected).abs() <= 3.0e-5,
            "2D activation dispatch lost or aliased index {index}: actual={} expected={expected}",
            execution.values[index]
        );
    }
    for index in [1, ROW_STRIDE - 2, ROW_STRIDE + 2, LENGTH - 2] {
        assert_eq!(execution.values[index], 0.0);
    }
}

#[test]
fn readback_is_tied_to_the_test_supplied_wgsl_not_a_cpu_shadow_implementation() {
    let _serial = serial();
    let Some(runtime) = runtime() else { return };
    let source = r#"
struct F32Buffer { data: array<f32> }
struct Params {
    length: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}
@group(0) @binding(0) var<storage, read> input: F32Buffer;
@group(0) @binding(1) var<storage, read_write> output: F32Buffer;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= params.length {
        return;
    }
    output.data[global_id.x] = input.data[global_id.x] * 3.25 + 7.0;
}
"#;
    let input = vec![-8.0, -1.0, 0.0, 2.0, 9.0];
    let expected: Vec<_> = input.iter().map(|value| value * 3.25 + 7.0).collect();
    let invocation = KernelInvocation::SiluF32 { values: input };
    let execution = runtime
        .run_with_shader(&invocation, "test-nonce-arithmetic", source, "main")
        .unwrap();
    assert_eq!(execution.values, expected);
    assert_eq!(
        execution.diagnostics.shader_blake3,
        *blake3::hash(source.as_bytes()).as_bytes()
    );
    assert_eq!(execution.diagnostics.kernel, KernelId::SiluF32);
}

#[test]
fn error_scopes_capture_pipeline_validation_and_runtime_recovers_afterward() {
    let _serial = serial();
    let Some(_) = runtime() else { return };
    let observer = Arc::new(RecordingObserver::default());
    let runtime = NativeRuntime::new(NativeOptions {
        observer: Some(observer.clone()),
    })
    .unwrap();
    observer.take();

    let success = run(
        &runtime,
        KernelInvocation::SiluF32 {
            values: vec![-1.0, 0.0, 1.0],
        },
    );
    assert_eq!(success.values.len(), 3);
    let success_events = observer.take();
    let push_events: Vec<_> = success_events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::ScopePushed(scope) => Some(*scope),
            _ => None,
        })
        .collect();
    assert_eq!(
        push_events,
        [
            ErrorScopeKind::Internal,
            ErrorScopeKind::OutOfMemory,
            ErrorScopeKind::Validation,
        ]
    );
    let submission_index = success_events
        .iter()
        .position(|event| matches!(event, RuntimeEvent::SubmissionQueued { .. }))
        .expect("the observed run must really submit a command buffer");
    let last_push = success_events
        .iter()
        .rposition(|event| matches!(event, RuntimeEvent::ScopePushed(_)))
        .unwrap();
    let first_pop = success_events
        .iter()
        .position(|event| matches!(event, RuntimeEvent::ScopePopped { .. }))
        .unwrap();
    assert!(last_push < submission_index && submission_index < first_pop);
    let pop_events: Vec<_> = success_events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::ScopePopped {
                scope,
                captured_error,
            } => Some((*scope, *captured_error)),
            _ => None,
        })
        .collect();
    assert_eq!(
        pop_events,
        [
            (ErrorScopeKind::Validation, false),
            (ErrorScopeKind::OutOfMemory, false),
            (ErrorScopeKind::Internal, false),
        ]
    );

    let source = pvlc_wgsl::module(KernelId::SiluF32).unwrap().source;
    let error = runtime
        .validate_pipeline_source("intentional-missing-entry", source, "missing")
        .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::Validation);
    assert_eq!(error.scope(), Some(ErrorScopeKind::Validation));
    assert!(error.to_string().contains("intentional-missing-entry"));
    let failure_events = observer.take();
    assert!(failure_events.contains(&RuntimeEvent::ScopePopped {
        scope: ErrorScopeKind::Validation,
        captured_error: true,
    }));
    assert!(failure_events.contains(&RuntimeEvent::ScopePopped {
        scope: ErrorScopeKind::OutOfMemory,
        captured_error: false,
    }));
    assert!(failure_events.contains(&RuntimeEvent::ScopePopped {
        scope: ErrorScopeKind::Internal,
        captured_error: false,
    }));

    let recovered = run(
        &runtime,
        KernelInvocation::SiluF32 {
            values: vec![-1.0, 0.0, 1.0],
        },
    );
    assert_eq!(recovered.values.len(), 3);
}

#[test]
fn scope_driver_propagates_validation_oom_and_internal_errors_and_always_unwinds() {
    #[derive(Default)]
    struct FakeDriver {
        injected: Option<ErrorScopeKind>,
        events: Vec<(bool, ErrorScopeKind)>,
    }

    impl ErrorScopeDriver for FakeDriver {
        fn push_scope(&mut self, scope: ErrorScopeKind) {
            self.events.push((true, scope));
        }

        fn pop_scope(&mut self, scope: ErrorScopeKind) -> Option<String> {
            self.events.push((false, scope));
            (self.injected == Some(scope)).then(|| format!("injected {scope:?}"))
        }
    }

    let _serial = serial();
    for (scope, code) in [
        (ErrorScopeKind::Validation, RuntimeErrorCode::Validation),
        (ErrorScopeKind::OutOfMemory, RuntimeErrorCode::OutOfMemory),
        (ErrorScopeKind::Internal, RuntimeErrorCode::Internal),
    ] {
        let mut driver = FakeDriver {
            injected: Some(scope),
            events: Vec::new(),
        };
        let error = drive_error_scopes(&mut driver, || {
            Ok::<_, pvlc_runtime_native::RuntimeError>(7)
        })
        .unwrap_err();
        assert_eq!(error.code(), code);
        assert_eq!(error.scope(), Some(scope));
        assert_eq!(
            driver.events,
            [
                (true, ErrorScopeKind::Internal),
                (true, ErrorScopeKind::OutOfMemory),
                (true, ErrorScopeKind::Validation),
                (false, ErrorScopeKind::Validation),
                (false, ErrorScopeKind::OutOfMemory),
                (false, ErrorScopeKind::Internal),
            ]
        );

        driver.injected = None;
        driver.events.clear();
        assert_eq!(
            drive_error_scopes(&mut driver, || Ok::<_, pvlc_runtime_native::RuntimeError>(
                11
            ))
            .unwrap(),
            11
        );
    }

    let expected_unwind = [
        (true, ErrorScopeKind::Internal),
        (true, ErrorScopeKind::OutOfMemory),
        (true, ErrorScopeKind::Validation),
        (false, ErrorScopeKind::Validation),
        (false, ErrorScopeKind::OutOfMemory),
        (false, ErrorScopeKind::Internal),
    ];
    let mut operation_failure = FakeDriver::default();
    let error = drive_error_scopes(&mut operation_failure, || {
        Err::<(), _>(pvlc_runtime_native::RuntimeError::operation(
            "sentinel-operation",
        ))
    })
    .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::Operation);
    assert_eq!(error.scope(), None);
    assert!(error.to_string().contains("sentinel-operation"));
    assert_eq!(operation_failure.events, expected_unwind);

    // Captured GPU errors have priority, but the triggering operation error remains in context.
    let mut simultaneous = FakeDriver {
        injected: Some(ErrorScopeKind::Validation),
        events: Vec::new(),
    };
    let error = drive_error_scopes(&mut simultaneous, || {
        Err::<(), _>(pvlc_runtime_native::RuntimeError::operation(
            "sentinel-operation",
        ))
    })
    .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::Validation);
    assert_eq!(error.scope(), Some(ErrorScopeKind::Validation));
    assert!(error.to_string().contains("injected Validation"));
    assert!(error.to_string().contains("sentinel-operation"));
    assert_eq!(simultaneous.events, expected_unwind);
}

#[test]
fn timestamp_diagnostics_are_readable_when_the_adapter_advertises_support() {
    let _serial = serial();
    let Some(runtime) = runtime() else { return };
    let capabilities = runtime.capabilities();
    if env_flag("PVLC_REQUIRE_TIMESTAMP_QUERY") {
        assert!(
            capabilities.timestamp_query,
            "the explicitly required timestamp-query laboratory must expose the feature"
        );
    }

    let execution = run(
        runtime,
        KernelInvocation::GemmF32 {
            rows: 256,
            inner: 256,
            columns: 256,
            left: values(256 * 256, 101),
            right: values(256 * 256, 103),
        },
    );
    assert!(execution.diagnostics.queue_wall_time_ns > 0);
    if capabilities.timestamp_query {
        let timestamp = execution
            .diagnostics
            .timestamp
            .expect("timestamp-capable adapters must emit timing diagnostics");
        assert!(timestamp.end_ticks > timestamp.begin_ticks);
        assert!(timestamp.period_ns.is_finite() && timestamp.period_ns > 0.0);
        assert!(timestamp.duration_ns.is_finite() && timestamp.duration_ns > 0.0);
        let expected = (timestamp.end_ticks - timestamp.begin_ticks) as f64 * timestamp.period_ns;
        assert_close(
            timestamp.duration_ns,
            expected,
            timestamp.period_ns.max(expected * 1.0e-12),
        );
    } else {
        assert!(execution.diagnostics.timestamp.is_none());
        assert!(
            execution.diagnostics.queue_wall_time_ns > 0,
            "queue-completion wall time is the required fallback"
        );
    }
}

#[test]
fn malformed_invocations_are_reported_before_dispatch() {
    let _serial = serial();
    let Some(runtime) = runtime() else { return };
    let before = runtime.counters();
    let error = runtime
        .run(&KernelInvocation::GemmF32 {
            rows: 1,
            inner: 2,
            columns: 1,
            left: vec![1.0],
            right: vec![1.0, 2.0],
        })
        .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::InvalidInvocation);
    assert_eq!(error.scope(), None);
    let after = runtime.counters();
    assert_eq!(after.buffer_allocations, before.buffer_allocations);
    assert_eq!(after.submissions, before.submissions);
}
