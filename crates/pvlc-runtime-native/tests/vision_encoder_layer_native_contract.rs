use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use pvlc_cpu_ref::{
    KvBlockOrder, LayerNormParameters as CpuLayerNormParameters,
    LinearParameters as CpuLinearParameters, VisionEncoderLayerConfig as CpuLayerConfig,
    VisionEncoderLayerParameters as CpuLayerParameters, VisionEncoderLayerTrace,
    VisionEncoderStackConfig as CpuStackConfig, vision_encoder_layer_identity_rope_f32,
    vision_encoder_stack_identity_rope_f32,
};
use pvlc_runtime_core::{
    KernelId, VisionEncoderLayerInvocation, VisionEncoderLayerParameters, VisionEncoderLayerStage,
    VisionEncoderStackInvocation, VisionLayerNormParameters, VisionLinearParameters,
    VisionRopeSpecialization, VisionStackActivationLayoutConfig, VisionStackActivationStrategy,
};
use pvlc_runtime_native::{
    BackendKind, ErrorScopeKind, NativeOptions, NativeRuntime, RuntimeErrorCode, RuntimeEvent,
    RuntimeObserver, VisionLayerReadback,
};
use pvlc_testkit::{ComparisonAxes, ComparisonPolicy, compare_f32};

const TOKENS: usize = 9;
const HIDDEN: usize = 18;
const HEADS: usize = 3;
const HEAD_DIM: usize = 6;
const INTERMEDIATE: usize = 23;
const EPSILON: f32 = 1.0e-6;

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

#[derive(Clone)]
struct Fixture {
    boundaries: Vec<u32>,
    input: Vec<f32>,
    norm1_weight: Vec<f32>,
    norm1_bias: Vec<f32>,
    query_weight: Vec<f32>,
    query_bias: Vec<f32>,
    key_weight: Vec<f32>,
    key_bias: Vec<f32>,
    value_weight: Vec<f32>,
    value_bias: Vec<f32>,
    attention_output_weight: Vec<f32>,
    attention_output_bias: Vec<f32>,
    norm2_weight: Vec<f32>,
    norm2_bias: Vec<f32>,
    mlp_fc1_weight: Vec<f32>,
    mlp_fc1_bias: Vec<f32>,
    mlp_fc2_weight: Vec<f32>,
    mlp_fc2_bias: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            boundaries: vec![0, 3, TOKENS as u32],
            input: values(TOKENS * HIDDEN, 1),
            norm1_weight: shifted_values(HIDDEN, 2, 1.0),
            norm1_bias: values(HIDDEN, 3),
            query_weight: values(HIDDEN * HIDDEN, 4),
            query_bias: values(HIDDEN, 5),
            key_weight: values(HIDDEN * HIDDEN, 6),
            key_bias: values(HIDDEN, 7),
            value_weight: values(HIDDEN * HIDDEN, 8),
            value_bias: values(HIDDEN, 9),
            attention_output_weight: values(HIDDEN * HIDDEN, 10),
            attention_output_bias: values(HIDDEN, 11),
            norm2_weight: shifted_values(HIDDEN, 12, 1.0),
            norm2_bias: values(HIDDEN, 13),
            mlp_fc1_weight: values(INTERMEDIATE * HIDDEN, 14),
            mlp_fc1_bias: values(INTERMEDIATE, 15),
            mlp_fc2_weight: values(HIDDEN * INTERMEDIATE, 16),
            mlp_fc2_bias: values(HIDDEN, 17),
        }
    }

    fn invocation(&self) -> VisionEncoderLayerInvocation<'_> {
        VisionEncoderLayerInvocation {
            tokens: TOKENS as u32,
            hidden_size: HIDDEN as u32,
            attention_heads: HEADS as u32,
            head_dim: HEAD_DIM as u32,
            intermediate_size: INTERMEDIATE as u32,
            layer_norm_epsilon: EPSILON,
            input: &self.input,
            cu_seqlens: &self.boundaries,
            parameters: VisionEncoderLayerParameters {
                norm1: VisionLayerNormParameters {
                    weight: &self.norm1_weight,
                    bias: &self.norm1_bias,
                },
                query: VisionLinearParameters {
                    weight: &self.query_weight,
                    bias: &self.query_bias,
                },
                key: VisionLinearParameters {
                    weight: &self.key_weight,
                    bias: &self.key_bias,
                },
                value: VisionLinearParameters {
                    weight: &self.value_weight,
                    bias: &self.value_bias,
                },
                attention_output: VisionLinearParameters {
                    weight: &self.attention_output_weight,
                    bias: &self.attention_output_bias,
                },
                norm2: VisionLayerNormParameters {
                    weight: &self.norm2_weight,
                    bias: &self.norm2_bias,
                },
                mlp_fc1: VisionLinearParameters {
                    weight: &self.mlp_fc1_weight,
                    bias: &self.mlp_fc1_bias,
                },
                mlp_fc2: VisionLinearParameters {
                    weight: &self.mlp_fc2_weight,
                    bias: &self.mlp_fc2_bias,
                },
            },
        }
    }

    fn cpu_parameters(&self) -> CpuLayerParameters<'_> {
        CpuLayerParameters {
            norm1: CpuLayerNormParameters {
                weight: &self.norm1_weight,
                bias: &self.norm1_bias,
            },
            query: CpuLinearParameters {
                weight: &self.query_weight,
                bias: &self.query_bias,
            },
            key: CpuLinearParameters {
                weight: &self.key_weight,
                bias: &self.key_bias,
            },
            value: CpuLinearParameters {
                weight: &self.value_weight,
                bias: &self.value_bias,
            },
            attention_output: CpuLinearParameters {
                weight: &self.attention_output_weight,
                bias: &self.attention_output_bias,
            },
            norm2: CpuLayerNormParameters {
                weight: &self.norm2_weight,
                bias: &self.norm2_bias,
            },
            mlp_fc1: CpuLinearParameters {
                weight: &self.mlp_fc1_weight,
                bias: &self.mlp_fc1_bias,
            },
            mlp_fc2: CpuLinearParameters {
                weight: &self.mlp_fc2_weight,
                bias: &self.mlp_fc2_bias,
            },
        }
    }

    fn cpu_trace(&self) -> VisionEncoderLayerTrace {
        let boundaries = self
            .boundaries
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>();
        vision_encoder_layer_identity_rope_f32(
            &self.input,
            CpuLayerConfig {
                tokens: TOKENS,
                hidden_size: HIDDEN,
                attention_heads: HEADS,
                head_dim: HEAD_DIM,
                intermediate_size: INTERMEDIATE,
                layer_norm_epsilon: EPSILON,
                attention_key_tile: 4,
                attention_order: KvBlockOrder::Forward,
            },
            &boundaries,
            self.cpu_parameters(),
        )
        .unwrap()
    }

    fn runtime_parameters(&self) -> VisionEncoderLayerParameters<'_> {
        self.invocation().parameters
    }
}

fn values(length: usize, seed: u32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let phase = (index as f32 + 1.0) * (seed as f32 + 0.5) * 0.013;
            phase.sin() * 0.2 - phase.cos() * 0.07
        })
        .collect()
}

fn shifted_values(length: usize, seed: u32, shift: f32) -> Vec<f32> {
    values(length, seed)
        .into_iter()
        .map(|value| value + shift)
        .collect()
}

fn env_flag(name: &str) -> bool {
    matches!(std::env::var(name).as_deref(), Ok("1" | "true"))
}

fn runtime(observer: Arc<RecordingObserver>) -> Option<NativeRuntime> {
    match NativeRuntime::new(NativeOptions {
        observer: Some(observer),
    }) {
        Ok(runtime) => {
            if env_flag("PVLC_REQUIRE_M4_METAL") {
                assert_eq!(runtime.capabilities().backend, BackendKind::Metal);
                assert!(runtime.capabilities().adapter_name.contains("M4 Pro"));
            }
            Some(runtime)
        }
        Err(error) if env_flag("PVLC_REQUIRE_NATIVE_GPU") || env_flag("PVLC_REQUIRE_M4_METAL") => {
            panic!("native GPU is required: {error}")
        }
        Err(error) => {
            eprintln!("skipping resident vision-layer contract: {error}");
            None
        }
    }
}

fn policy() -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs: 2.0e-4,
        max_mean_abs: 2.0e-5,
        max_p99_abs: 1.0e-4,
        max_relative_l2: 1.0e-4,
        min_cosine_similarity: 0.999_999,
        max_per_token_relative_l2: Some(2.0e-4),
        max_per_channel_relative_l2: None,
    }
}

fn expected_stages(
    trace: &VisionEncoderLayerTrace,
) -> [(VisionEncoderLayerStage, &[f32], usize); 12] {
    [
        (VisionEncoderLayerStage::Norm1, &trace.norm1, HIDDEN),
        (VisionEncoderLayerStage::Query, &trace.query, HIDDEN),
        (VisionEncoderLayerStage::Key, &trace.key, HIDDEN),
        (VisionEncoderLayerStage::Value, &trace.value, HIDDEN),
        (
            VisionEncoderLayerStage::AttentionContext,
            &trace.attention_context,
            HIDDEN,
        ),
        (
            VisionEncoderLayerStage::AttentionOutput,
            &trace.attention_output,
            HIDDEN,
        ),
        (
            VisionEncoderLayerStage::AttentionResidual,
            &trace.attention_residual,
            HIDDEN,
        ),
        (VisionEncoderLayerStage::Norm2, &trace.norm2, HIDDEN),
        (
            VisionEncoderLayerStage::MlpFc1,
            &trace.mlp_fc1,
            INTERMEDIATE,
        ),
        (
            VisionEncoderLayerStage::MlpActivation,
            &trace.mlp_activation,
            INTERMEDIATE,
        ),
        (
            VisionEncoderLayerStage::MlpOutput,
            &trace.mlp_output,
            HIDDEN,
        ),
        (VisionEncoderLayerStage::Output, &trace.output, HIDDEN),
    ]
}

const SHADER_NONCE: f32 = 0.125;

fn nonce_shader(kernel: KernelId) -> String {
    let source = pvlc_wgsl::module(kernel).unwrap().source;
    let (original, replacement) = match kernel {
        KernelId::LayerNormF32 => (
            "output.data[row_start + column] = (input.data[row_start + column] - mean) * inverse_stddev * weight.data[column] + bias.data[column];",
            "output.data[row_start + column] = (input.data[row_start + column] - mean) * inverse_stddev * weight.data[column] + bias.data[column] + 0.125;",
        ),
        KernelId::VisionPatchProjectionF32 => (
            "accumulated[output_offset];",
            "accumulated[output_offset] + 0.125;",
        ),
        KernelId::VisionAttentionF32 => (
            "let normalized = attention_output[vector_index] / running_denominator;",
            "let normalized = attention_output[vector_index] / running_denominator + vec4<f32>(0.125);",
        ),
        KernelId::GeluTanhF32 => (
            "output.data[index] = 0.5 * value * (1.0 + tanh(argument));",
            "output.data[index] = 0.5 * value * (1.0 + tanh(argument)) + 0.125;",
        ),
        KernelId::AddF32 => (
            "output.data[index] = left.data[index] + right.data[index];",
            "output.data[index] = left.data[index] + right.data[index] + 0.125;",
        ),
        _ => panic!("{kernel:?} is not a physical vision-layer kernel"),
    };
    assert_eq!(
        source.matches(original).count(),
        1,
        "nonce injection missed kernel {kernel}"
    );
    let nonce = source.replace(original, replacement);
    assert_ne!(nonce, source);
    nonce
}

fn max_abs_difference(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0, f32::max)
}

#[test]
fn native_layer_keeps_all_intermediates_on_gpu_in_one_submission_and_matches_every_cpu_stage() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();
    let fixture = Fixture::new();
    let expected = fixture.cpu_trace();
    let before = runtime.counters();
    let execution = runtime
        .run_vision_encoder_layer_identity_rope(
            &fixture.invocation(),
            VisionLayerReadback::AllStages,
        )
        .unwrap();
    let after = runtime.counters();
    let events = observer.take();

    assert_eq!(after.submissions - before.submissions, 1);
    assert_eq!(execution.diagnostics.submission_count, 1);
    assert_eq!(execution.diagnostics.command_buffer_count, 1);
    assert_eq!(
        after.buffer_allocations - before.buffer_allocations,
        execution.diagnostics.buffer_allocation_count
    );
    assert!(execution.diagnostics.buffer_allocation_count <= 40);
    assert_eq!(execution.diagnostics.readback_buffer_count, 1);
    assert_eq!(
        execution.diagnostics.dispatch_stages,
        VisionEncoderLayerStage::ALL
    );
    assert_eq!(
        execution.diagnostics.rope_specialization,
        VisionRopeSpecialization::Identity
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

    let readback_allocations = events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::BufferAllocated { label, bytes } if label.contains("readback") => {
                Some(*bytes)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let submitted_command_buffers = events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::SubmissionQueued {
                command_buffers, ..
            } => Some(*command_buffers),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(submitted_command_buffers, vec![1]);
    let submission_position = events
        .iter()
        .position(|event| matches!(event, RuntimeEvent::SubmissionQueued { .. }))
        .unwrap();
    assert!(
        events[..submission_position]
            .iter()
            .any(|event| matches!(event, RuntimeEvent::BufferAllocated { .. }))
    );
    assert!(
        !events[submission_position + 1..]
            .iter()
            .any(|event| matches!(event, RuntimeEvent::BufferAllocated { .. }))
    );

    let expected_readback_bytes = expected_stages(&expected)
        .iter()
        .map(|(_, values, _)| values.len() as u64 * 4)
        .sum::<u64>();
    assert_eq!(readback_allocations, vec![expected_readback_bytes]);
    assert_eq!(
        execution.diagnostics.readback_bytes,
        expected_readback_bytes
    );
    assert_eq!(
        execution.checkpoints.len(),
        VisionEncoderLayerStage::ALL.len()
    );
    for (stage, reference, width) in expected_stages(&expected) {
        let actual = execution.checkpoints.get(&stage).unwrap();
        let report = compare_f32(
            reference,
            actual,
            &[TOKENS, width],
            ComparisonAxes {
                token_axis: Some(0),
                channel_axis: None,
            },
        )
        .unwrap();
        let verdict = report.assess(&policy()).unwrap();
        assert!(
            verdict.passed(),
            "stage {stage:?} failed: {report:#?}\n{verdict:#?}"
        );
    }

    let expected_kernels = [
        KernelId::LayerNormF32,
        KernelId::VisionPatchProjectionF32,
        KernelId::VisionAttentionF32,
        KernelId::AddF32,
        KernelId::GeluTanhF32,
    ];
    assert_eq!(
        execution.diagnostics.shader_blake3.len(),
        expected_kernels.len()
    );
    for kernel in expected_kernels {
        let source = pvlc_wgsl::module(kernel).unwrap().source;
        assert_eq!(
            execution.diagnostics.shader_blake3.get(&kernel),
            Some(blake3::hash(source.as_bytes()).as_bytes())
        );
    }
    if runtime.capabilities().timestamp_query {
        let timestamp = execution.diagnostics.timestamp.unwrap();
        assert!(timestamp.end_ticks > timestamp.begin_ticks);
        assert!(timestamp.duration_ns.is_finite() && timestamp.duration_ns > 0.0);
    } else {
        assert!(execution.diagnostics.timestamp.is_none());
    }

    let before_output_only = runtime.counters();
    let output_only = runtime
        .run_vision_encoder_layer_identity_rope(
            &fixture.invocation(),
            VisionLayerReadback::OutputOnly,
        )
        .unwrap();
    let after_output_only = runtime.counters();
    let output_only_events = observer.take();
    assert_eq!(
        after_output_only.submissions - before_output_only.submissions,
        1
    );
    assert_eq!(output_only.checkpoints.len(), 1);
    assert_eq!(
        output_only
            .checkpoints
            .get(&VisionEncoderLayerStage::Output),
        execution.checkpoints.get(&VisionEncoderLayerStage::Output)
    );
    assert_eq!(output_only.diagnostics.readback_buffer_count, 1);
    assert_eq!(output_only.diagnostics.command_buffer_count, 1);
    assert_eq!(
        output_only.diagnostics.readback_bytes,
        (TOKENS * HIDDEN * 4) as u64
    );
    let output_only_readbacks = output_only_events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::BufferAllocated { label, bytes } if label.contains("readback") => {
                Some(*bytes)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(output_only_readbacks, vec![(TOKENS * HIDDEN * 4) as u64]);
    let output_only_command_buffers = output_only_events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::SubmissionQueued {
                command_buffers, ..
            } => Some(*command_buffers),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(output_only_command_buffers, vec![1]);
    let output_only_submission = output_only_events
        .iter()
        .position(|event| matches!(event, RuntimeEvent::SubmissionQueued { .. }))
        .unwrap();
    assert!(
        !output_only_events[output_only_submission + 1..]
            .iter()
            .any(|event| matches!(event, RuntimeEvent::BufferAllocated { .. }))
    );
}

#[test]
fn every_physical_shader_is_proven_to_drive_its_checkpoint_and_the_final_output() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(observer) else {
        return;
    };
    let fixture = Fixture::new();
    let baseline = runtime
        .run_vision_encoder_layer_identity_rope(
            &fixture.invocation(),
            VisionLayerReadback::AllStages,
        )
        .unwrap();
    let baseline_output = baseline
        .checkpoints
        .get(&VisionEncoderLayerStage::Output)
        .unwrap();

    for (kernel, directly_written_stage) in [
        (KernelId::LayerNormF32, VisionEncoderLayerStage::Norm1),
        (
            KernelId::VisionPatchProjectionF32,
            VisionEncoderLayerStage::Query,
        ),
        (
            KernelId::VisionAttentionF32,
            VisionEncoderLayerStage::AttentionContext,
        ),
        (KernelId::AddF32, VisionEncoderLayerStage::AttentionResidual),
        (
            KernelId::GeluTanhF32,
            VisionEncoderLayerStage::MlpActivation,
        ),
    ] {
        let source = nonce_shader(kernel);
        let overrides = BTreeMap::from([(kernel, source.clone())]);
        let nonce_execution = runtime
            .run_vision_encoder_layer_identity_rope_with_shader_overrides(
                &fixture.invocation(),
                VisionLayerReadback::AllStages,
                &overrides,
            )
            .unwrap();
        let canonical_checkpoint = baseline.checkpoints.get(&directly_written_stage).unwrap();
        let nonce_checkpoint = nonce_execution
            .checkpoints
            .get(&directly_written_stage)
            .unwrap();
        let expected_checkpoint = canonical_checkpoint
            .iter()
            .map(|value| *value + SHADER_NONCE)
            .collect::<Vec<_>>();
        for (index, ((&canonical, &actual), &expected)) in canonical_checkpoint
            .iter()
            .zip(nonce_checkpoint)
            .zip(&expected_checkpoint)
            .enumerate()
        {
            let tolerance = 4.0 * f32::EPSILON * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{kernel:?}[{index}] must add the local {SHADER_NONCE} nonce: canonical={canonical}, actual={actual}, expected={expected}, tolerance={tolerance}"
            );
            assert!(
                ((actual - canonical) - SHADER_NONCE).abs() <= tolerance,
                "{kernel:?}[{index}] nonce delta is not locally observable"
            );
        }
        let nonce_output = nonce_execution
            .checkpoints
            .get(&VisionEncoderLayerStage::Output)
            .unwrap();
        assert!(
            max_abs_difference(baseline_output, nonce_output) > 1.0e-5,
            "{kernel:?} nonce must propagate to the final output"
        );
        assert_eq!(
            nonce_execution.diagnostics.shader_blake3.get(&kernel),
            Some(blake3::hash(source.as_bytes()).as_bytes())
        );
        assert_eq!(nonce_execution.diagnostics.submission_count, 1);
        assert_eq!(nonce_execution.diagnostics.command_buffer_count, 1);
    }
}

#[test]
fn resident_layer_preserves_packed_image_isolation_end_to_end() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(observer) else {
        return;
    };
    let baseline_fixture = Fixture::new();
    let baseline = runtime
        .run_vision_encoder_layer_identity_rope(
            &baseline_fixture.invocation(),
            VisionLayerReadback::OutputOnly,
        )
        .unwrap();

    let mut changed_fixture = baseline_fixture.clone();
    for (index, value) in changed_fixture.input[3 * HIDDEN..].iter_mut().enumerate() {
        *value = (index as f32 + 1.0) * 0.37 - 11.0;
    }
    let isolated = runtime
        .run_vision_encoder_layer_identity_rope(
            &changed_fixture.invocation(),
            VisionLayerReadback::OutputOnly,
        )
        .unwrap();
    let mut joint_fixture = changed_fixture.clone();
    joint_fixture.boundaries = vec![0, TOKENS as u32];
    let joint = runtime
        .run_vision_encoder_layer_identity_rope(
            &joint_fixture.invocation(),
            VisionLayerReadback::OutputOnly,
        )
        .unwrap();

    let first_segment = 3 * HIDDEN;
    let baseline_output = baseline
        .checkpoints
        .get(&VisionEncoderLayerStage::Output)
        .unwrap();
    let isolated_output = isolated
        .checkpoints
        .get(&VisionEncoderLayerStage::Output)
        .unwrap();
    let joint_output = joint
        .checkpoints
        .get(&VisionEncoderLayerStage::Output)
        .unwrap();
    assert_eq!(
        &baseline_output[..first_segment],
        &isolated_output[..first_segment]
    );
    assert!(
        baseline_output[..first_segment]
            .iter()
            .zip(&joint_output[..first_segment])
            .any(|(baseline, joint)| (baseline - joint).abs() > 1.0e-4)
    );
}

#[test]
fn invalid_layer_is_rejected_before_gpu_allocation_or_submission() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();
    let mut fixture = Fixture::new();
    fixture.query_weight[7] = f32::NAN;
    let before = runtime.counters();
    let error = runtime
        .run_vision_encoder_layer_identity_rope(
            &fixture.invocation(),
            VisionLayerReadback::AllStages,
        )
        .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::InvalidInvocation);
    assert_eq!(runtime.counters(), before);
    assert!(observer.take().is_empty());
}

fn stack_fixtures(layers: usize) -> Vec<Fixture> {
    (0..layers)
        .map(|layer| {
            let mut fixture = Fixture::new();
            let delta = layer as f32 * 0.003_125;
            for (operand_index, operand) in [
                &mut fixture.norm1_bias,
                &mut fixture.query_bias,
                &mut fixture.key_bias,
                &mut fixture.value_bias,
                &mut fixture.attention_output_bias,
                &mut fixture.norm2_bias,
                &mut fixture.mlp_fc1_bias,
                &mut fixture.mlp_fc2_bias,
            ]
            .into_iter()
            .enumerate()
            {
                for value in operand {
                    *value += delta * (operand_index as f32 + 1.0);
                }
            }
            fixture
        })
        .collect()
}

fn stack_runtime_parameters(fixtures: &[Fixture]) -> Vec<VisionEncoderLayerParameters<'_>> {
    fixtures.iter().map(Fixture::runtime_parameters).collect()
}

fn stack_invocation<'a>(
    fixtures: &'a [Fixture],
    layer_parameters: &'a [VisionEncoderLayerParameters<'a>],
    post_weight: &'a [f32],
    post_bias: &'a [f32],
) -> VisionEncoderStackInvocation<'a> {
    VisionEncoderStackInvocation {
        tokens: TOKENS as u32,
        hidden_size: HIDDEN as u32,
        attention_heads: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        intermediate_size: INTERMEDIATE as u32,
        layer_norm_epsilon: EPSILON,
        input: &fixtures[0].input,
        cu_seqlens: &fixtures[0].boundaries,
        layer_parameters,
        post_norm: VisionLayerNormParameters {
            weight: post_weight,
            bias: post_bias,
        },
    }
}

fn cpu_stack(
    fixtures: &[Fixture],
    checkpoints: &[usize],
    post_weight: &[f32],
    post_bias: &[f32],
) -> pvlc_cpu_ref::VisionEncoderStackTrace {
    let boundaries = fixtures[0]
        .boundaries
        .iter()
        .map(|value| *value as usize)
        .collect::<Vec<_>>();
    vision_encoder_stack_identity_rope_f32(
        &fixtures[0].input,
        CpuStackConfig {
            tokens: TOKENS,
            hidden_size: HIDDEN,
            layers: fixtures.len(),
            layer_norm_epsilon: EPSILON,
        },
        checkpoints,
        CpuLayerNormParameters {
            weight: post_weight,
            bias: post_bias,
        },
        |layer, input| {
            vision_encoder_layer_identity_rope_f32(
                input,
                CpuLayerConfig {
                    tokens: TOKENS,
                    hidden_size: HIDDEN,
                    attention_heads: HEADS,
                    head_dim: HEAD_DIM,
                    intermediate_size: INTERMEDIATE,
                    layer_norm_epsilon: EPSILON,
                    attention_key_tile: 4,
                    attention_order: KvBlockOrder::Forward,
                },
                &boundaries,
                fixtures[layer].cpu_parameters(),
            )
            .map(|trace| trace.output)
        },
    )
    .unwrap()
}

fn assert_stack_close(expected: &[f32], actual: &[f32]) {
    let report = compare_f32(
        expected,
        actual,
        &[TOKENS, HIDDEN],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap();
    let verdict = report.assess(&policy()).unwrap();
    assert!(verdict.passed(), "{report:#?}\n{verdict:#?}");
}

fn activation_allocation_bytes(events: &[RuntimeEvent]) -> BTreeMap<String, u64> {
    let allocations = raw_activation_allocations(events);
    assert_eq!(
        allocations.len(),
        allocations
            .iter()
            .map(|(label, _)| label.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        "duplicate activation allocation labels are not allowed"
    );
    allocations.into_iter().collect()
}

fn raw_activation_allocations(events: &[RuntimeEvent]) -> Vec<(String, u64)> {
    events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::BufferAllocated { label, bytes }
                if label.starts_with("vision-stack-activation-") =>
            {
                Some((label.clone(), *bytes))
            }
            _ => None,
        })
        .collect()
}

fn submission_command_buffers(events: &[RuntimeEvent]) -> Vec<u32> {
    events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::SubmissionQueued {
                command_buffers, ..
            } => Some(*command_buffers),
            _ => None,
        })
        .collect()
}

fn readback_labels(events: &[RuntimeEvent]) -> Vec<(String, u64)> {
    events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::ReadbackMapRequested { label, bytes } => Some((label.clone(), *bytes)),
            _ => None,
        })
        .collect()
}

fn activation_layout_config(
    runtime: &NativeRuntime,
    allow_aliasing: bool,
) -> VisionStackActivationLayoutConfig {
    let alignment = u64::from(runtime.capabilities().min_storage_buffer_offset_alignment);
    VisionStackActivationLayoutConfig {
        allow_aliasing,
        storage_buffer_offset_alignment: alignment,
        arena_alignment: alignment,
    }
}

#[test]
fn native_stack_chains_layers_and_post_norm_in_one_submission_with_selected_readback() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();
    let fixtures = stack_fixtures(4);
    let layer_parameters = stack_runtime_parameters(&fixtures);
    let post_weight = shifted_values(HIDDEN, 91, 1.0);
    let post_bias = values(HIDDEN, 92);
    let selected = [0_usize, 2, 3];
    let expected = cpu_stack(&fixtures, &selected, &post_weight, &post_bias);
    let unchained_last = vision_encoder_layer_identity_rope_f32(
        &fixtures[0].input,
        CpuLayerConfig {
            tokens: TOKENS,
            hidden_size: HIDDEN,
            attention_heads: HEADS,
            head_dim: HEAD_DIM,
            intermediate_size: INTERMEDIATE,
            layer_norm_epsilon: EPSILON,
            attention_key_tile: 4,
            attention_order: KvBlockOrder::Forward,
        },
        &[0, 3, TOKENS],
        fixtures[3].cpu_parameters(),
    )
    .unwrap()
    .output;
    assert!(
        max_abs_difference(expected.checkpoint(3).unwrap(), &unchained_last) > 1.0e-3,
        "fixture must distinguish a correctly chained stack from layer 3 on the initial input"
    );
    assert!(
        max_abs_difference(expected.checkpoint(3).unwrap(), &expected.output) > 1.0e-3,
        "fixture must make post-layernorm observable"
    );

    let before = runtime.counters();
    let execution = runtime
        .run_vision_encoder_stack_identity_rope(
            &stack_invocation(&fixtures, &layer_parameters, &post_weight, &post_bias),
            &selected,
        )
        .unwrap();
    let after = runtime.counters();
    let events = observer.take();

    assert_eq!(execution.checkpoints.len(), selected.len());
    for layer in selected {
        assert_stack_close(
            expected.checkpoint(layer).unwrap(),
            execution.checkpoints.get(&layer).unwrap(),
        );
    }
    assert_stack_close(&expected.output, &execution.output);
    assert_eq!(execution.diagnostics.layer_count, 4);
    assert_eq!(execution.diagnostics.checkpoint_layers, selected);
    assert_eq!(execution.diagnostics.dispatch_count, 4 * 12 + 1);
    assert_eq!(execution.diagnostics.compute_pass_count, 5);
    assert_eq!(execution.diagnostics.submission_count, 1);
    assert_eq!(execution.diagnostics.command_buffer_count, 1);
    assert_eq!(after.submissions - before.submissions, 1);
    assert_eq!(execution.diagnostics.activation_buffer_count, 13);
    assert_eq!(execution.diagnostics.weight_buffer_count, 4 * 16 + 2);
    assert_eq!(execution.diagnostics.readback_buffer_count, 1);
    assert_eq!(execution.diagnostics.readback_map_count, 1);
    assert_eq!(
        execution.diagnostics.readback_bytes,
        ((selected.len() + 1) * TOKENS * HIDDEN * 4) as u64
    );
    assert!(execution.diagnostics.captured_errors.is_empty());
    assert_eq!(
        execution.diagnostics.checked_error_scopes,
        [
            ErrorScopeKind::Validation,
            ErrorScopeKind::OutOfMemory,
            ErrorScopeKind::Internal,
        ]
    );
    assert_eq!(
        execution.diagnostics.rope_specialization,
        VisionRopeSpecialization::Identity
    );
    assert_eq!(
        after.buffer_allocations - before.buffer_allocations,
        execution.diagnostics.buffer_allocation_count
    );
    let activation_allocations = events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::BufferAllocated { label, bytes }
                if label.starts_with("vision-stack-activation-") =>
            {
                Some(*bytes)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let expected_activation_arena_bytes =
        ((11 * TOKENS * HIDDEN + 2 * TOKENS * INTERMEDIATE) * 4) as u64;
    assert_eq!(activation_allocations.len(), 13);
    assert_eq!(
        activation_allocations.iter().sum::<u64>(),
        expected_activation_arena_bytes
    );
    assert_eq!(
        execution.diagnostics.activation_arena_bytes,
        expected_activation_arena_bytes
    );
    let maps = events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::ReadbackMapRequested { label, bytes } => Some((label.as_str(), *bytes)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        maps,
        [(
            "vision-stack-readback",
            execution.diagnostics.readback_bytes
        )]
    );
    let submitted_command_buffers = events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::SubmissionQueued {
                command_buffers, ..
            } => Some(*command_buffers),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(submitted_command_buffers, [1]);
    let submission_position = events
        .iter()
        .position(|event| matches!(event, RuntimeEvent::SubmissionQueued { .. }))
        .unwrap();
    let map_position = events
        .iter()
        .position(|event| matches!(event, RuntimeEvent::ReadbackMapRequested { .. }))
        .unwrap();
    assert!(submission_position < map_position);
    assert!(
        !events[submission_position + 1..]
            .iter()
            .any(|event| matches!(event, RuntimeEvent::BufferAllocated { .. }))
    );
}

#[test]
fn native_stack_activation_allocations_are_depth_constant() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();
    let post_weight = shifted_values(HIDDEN, 91, 1.0);
    let post_bias = values(HIDDEN, 92);
    let expected_activation_arena_bytes =
        ((11 * TOKENS * HIDDEN + 2 * TOKENS * INTERMEDIATE) * 4) as u64;
    let mut diagnostics = Vec::new();
    for depth in [1_usize, 16] {
        let fixtures = stack_fixtures(depth);
        let layer_parameters = stack_runtime_parameters(&fixtures);
        let execution = runtime
            .run_vision_encoder_stack_identity_rope(
                &stack_invocation(&fixtures, &layer_parameters, &post_weight, &post_bias),
                &[],
            )
            .unwrap();
        let events = observer.take();
        let activation_allocations = events
            .iter()
            .filter_map(|event| match event {
                RuntimeEvent::BufferAllocated { label, bytes }
                    if label.starts_with("vision-stack-activation-") =>
                {
                    Some(*bytes)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(activation_allocations.len(), 13);
        assert_eq!(
            activation_allocations.iter().sum::<u64>(),
            expected_activation_arena_bytes
        );
        assert_eq!(
            execution.diagnostics.activation_arena_bytes,
            expected_activation_arena_bytes
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, RuntimeEvent::ReadbackMapRequested { .. }))
                .count(),
            1
        );
        assert_eq!(
            events
                .iter()
                .filter_map(|event| match event {
                    RuntimeEvent::SubmissionQueued {
                        command_buffers, ..
                    } => Some(*command_buffers),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            [1]
        );
        assert_eq!(execution.diagnostics.weight_buffer_count, depth * 16 + 2);
        assert_eq!(
            execution.diagnostics.readback_bytes,
            (TOKENS * HIDDEN * 4) as u64
        );
        diagnostics.push(execution.diagnostics);
    }
    assert_eq!(diagnostics[0].activation_buffer_count, 13);
    assert_eq!(diagnostics[1].activation_buffer_count, 13);
    assert_eq!(
        diagnostics[0].activation_arena_bytes,
        diagnostics[1].activation_arena_bytes
    );
}

#[test]
fn invalid_stack_is_rejected_before_gpu_allocation_or_submission() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();
    let post_weight = shifted_values(HIDDEN, 91, 1.0);
    let post_bias = values(HIDDEN, 92);
    let assert_invalid =
        |fixtures: &[Fixture], checkpoints: &[usize], post_weight: &[f32], post_bias: &[f32]| {
            let layer_parameters = stack_runtime_parameters(fixtures);
            let before = runtime.counters();
            let error = runtime
                .run_vision_encoder_stack_identity_rope(
                    &stack_invocation(fixtures, &layer_parameters, post_weight, post_bias),
                    checkpoints,
                )
                .unwrap_err();
            assert_eq!(error.code(), RuntimeErrorCode::InvalidInvocation);
            assert_eq!(runtime.counters(), before);
            assert!(observer.take().is_empty());
        };

    let fixtures = stack_fixtures(4);
    assert_invalid(&fixtures, &[2, 1], &post_weight, &post_bias);
    let mut bad_post_bias = post_bias.clone();
    bad_post_bias[HIDDEN - 1] = f32::INFINITY;
    assert_invalid(&fixtures, &[], &post_weight, &bad_post_bias);

    let mut malformed_first = fixtures.clone();
    malformed_first[0].query_weight.pop();
    assert_invalid(&malformed_first, &[], &post_weight, &post_bias);
    let mut malformed_middle = fixtures.clone();
    malformed_middle[2].norm2_bias[HIDDEN / 2] = f32::INFINITY;
    assert_invalid(&malformed_middle, &[], &post_weight, &post_bias);
    let mut malformed_last = fixtures;
    malformed_last[3].mlp_fc2_bias[HIDDEN - 1] = f32::NAN;
    assert_invalid(&malformed_last, &[], &post_weight, &post_bias);
}

#[test]
fn native_stack_activation_strategies_match_cpu_and_separate_buffers_with_auditable_allocations() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();

    let fixtures = stack_fixtures(4);
    let layer_parameters = stack_runtime_parameters(&fixtures);
    let post_weight = shifted_values(HIDDEN, 91, 1.0);
    let post_bias = values(HIDDEN, 92);
    let checkpoints = [0_usize, 1, 2, 3];
    let invocation = stack_invocation(&fixtures, &layer_parameters, &post_weight, &post_bias);
    let expected = cpu_stack(&fixtures, &checkpoints, &post_weight, &post_bias);
    let plan = invocation.plan(&checkpoints).unwrap();
    let no_alias_layout = plan
        .activation_layout(activation_layout_config(&runtime, false))
        .unwrap();
    let alias_layout = plan
        .activation_layout(activation_layout_config(&runtime, true))
        .unwrap();

    let separate = runtime
        .run_vision_encoder_stack_identity_rope(&invocation, &checkpoints)
        .unwrap();
    let separate_events = observer.take();
    let separate_raw_allocations = raw_activation_allocations(&separate_events);
    assert_eq!(separate_raw_allocations.len(), 13);
    let separate_allocations = activation_allocation_bytes(&separate_events);

    let separate_via_strategy = runtime
        .run_vision_encoder_stack_identity_rope_with_activation_strategy(
            &invocation,
            &checkpoints,
            VisionStackActivationStrategy::SeparateBuffers,
        )
        .unwrap();
    let separate_strategy_events = observer.take();
    let separate_strategy_raw_allocations = raw_activation_allocations(&separate_strategy_events);
    assert_eq!(separate_strategy_raw_allocations.len(), 13);

    let no_alias = runtime
        .run_vision_encoder_stack_identity_rope_with_activation_strategy(
            &invocation,
            &checkpoints,
            VisionStackActivationStrategy::StaticArenaNoAlias,
        )
        .unwrap();
    let no_alias_events = observer.take();
    let no_alias_raw_allocations = raw_activation_allocations(&no_alias_events);
    assert_eq!(no_alias_raw_allocations.len(), 3);

    let alias = runtime
        .run_vision_encoder_stack_identity_rope_with_activation_strategy(
            &invocation,
            &checkpoints,
            VisionStackActivationStrategy::StaticArenaAlias,
        )
        .unwrap();
    let alias_events = observer.take();
    let alias_raw_allocations = raw_activation_allocations(&alias_events);
    assert_eq!(alias_raw_allocations.len(), 3);

    for execution in [&separate, &separate_via_strategy, &no_alias, &alias] {
        for layer in checkpoints {
            assert_stack_close(
                expected.checkpoint(layer).unwrap(),
                execution.checkpoints.get(&layer).unwrap(),
            );
        }
        assert_stack_close(&expected.output, &execution.output);
        assert_eq!(execution.diagnostics.submission_count, 1);
        assert_eq!(execution.diagnostics.command_buffer_count, 1);
        assert_eq!(execution.diagnostics.readback_buffer_count, 1);
        assert_eq!(execution.diagnostics.readback_map_count, 1);
        assert!(execution.diagnostics.captured_errors.is_empty());
    }

    assert_eq!(separate.checkpoints, separate_via_strategy.checkpoints);
    assert_eq!(separate.output, separate_via_strategy.output);
    assert_eq!(separate.checkpoints, no_alias.checkpoints);
    assert_eq!(separate.output, no_alias.output);
    assert_eq!(separate.checkpoints, alias.checkpoints);
    assert_eq!(separate.output, alias.output);

    assert_eq!(
        separate_allocations.keys().cloned().collect::<Vec<_>>(),
        vec![
            "vision-stack-activation-main-a".to_owned(),
            "vision-stack-activation-main-b".to_owned(),
            "vision-stack-activation-scratch-attention-context".to_owned(),
            "vision-stack-activation-scratch-attention-output".to_owned(),
            "vision-stack-activation-scratch-attention-residual".to_owned(),
            "vision-stack-activation-scratch-key".to_owned(),
            "vision-stack-activation-scratch-mlp-activation".to_owned(),
            "vision-stack-activation-scratch-mlp-fc1".to_owned(),
            "vision-stack-activation-scratch-mlp-output".to_owned(),
            "vision-stack-activation-scratch-norm1".to_owned(),
            "vision-stack-activation-scratch-norm2".to_owned(),
            "vision-stack-activation-scratch-query".to_owned(),
            "vision-stack-activation-scratch-value".to_owned(),
        ]
    );
    assert_eq!(separate_allocations.len(), 13);

    for (execution, events, strategy, layout) in [
        (
            &separate_via_strategy,
            &separate_strategy_events,
            VisionStackActivationStrategy::SeparateBuffers,
            None,
        ),
        (
            &no_alias,
            &no_alias_events,
            VisionStackActivationStrategy::StaticArenaNoAlias,
            Some(&no_alias_layout),
        ),
        (
            &alias,
            &alias_events,
            VisionStackActivationStrategy::StaticArenaAlias,
            Some(&alias_layout),
        ),
    ] {
        assert_eq!(execution.diagnostics.activation_strategy, strategy);
        assert_eq!(submission_command_buffers(events), vec![1]);
        assert_eq!(
            readback_labels(events),
            vec![(
                "vision-stack-readback".to_owned(),
                execution.diagnostics.readback_bytes
            )]
        );
        match layout {
            None => {
                assert_eq!(execution.diagnostics.activation_buffer_count, 13);
                assert_eq!(raw_activation_allocations(events).len(), 13);
                assert_eq!(
                    activation_allocation_bytes(events)
                        .values()
                        .copied()
                        .sum::<u64>(),
                    execution.diagnostics.activation_arena_bytes
                );
            }
            Some(layout) => {
                assert_eq!(raw_activation_allocations(events).len(), 3);
                let allocations = activation_allocation_bytes(events);
                assert_eq!(
                    allocations,
                    BTreeMap::from([
                        (
                            "vision-stack-activation-main-a".to_owned(),
                            u64::try_from(TOKENS * HIDDEN * 4).unwrap()
                        ),
                        (
                            "vision-stack-activation-main-b".to_owned(),
                            u64::try_from(TOKENS * HIDDEN * 4).unwrap()
                        ),
                        (
                            "vision-stack-activation-scratch-arena".to_owned(),
                            layout.scratch_arena_bytes
                        ),
                    ])
                );
                assert_eq!(execution.diagnostics.activation_buffer_count, 3);
                assert_eq!(
                    execution.diagnostics.scratch_arena_bytes,
                    layout.scratch_arena_bytes
                );
                assert_eq!(
                    execution.diagnostics.main_buffers_bytes,
                    layout.main_buffers_bytes
                );
                assert_eq!(
                    execution.diagnostics.activation_arena_bytes,
                    layout.total_activation_bytes
                );
                assert_eq!(
                    allocations.values().copied().sum::<u64>(),
                    execution.diagnostics.activation_arena_bytes
                );
                assert_eq!(
                    execution.diagnostics.scratch_allocations,
                    layout.scratch_allocations
                );
            }
        }
    }

    assert_eq!(separate.diagnostics.activation_buffer_count, 13);
    assert_eq!(
        separate.diagnostics.activation_arena_bytes,
        separate_allocations.values().copied().sum::<u64>()
    );
    assert_eq!(no_alias.diagnostics.activation_buffer_count, 3);
    assert_eq!(alias.diagnostics.activation_buffer_count, 3);
    assert!(alias.diagnostics.scratch_arena_bytes < no_alias.diagnostics.scratch_arena_bytes);
    assert!(alias.diagnostics.activation_arena_bytes < no_alias.diagnostics.activation_arena_bytes);
    assert!(alias.diagnostics.activation_arena_bytes < separate.diagnostics.activation_arena_bytes);
}

#[test]
fn native_stack_static_alias_layout_is_depth_invariant_and_cpu_exact() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();

    let post_weight = shifted_values(HIDDEN, 91, 1.0);
    let post_bias = values(HIDDEN, 92);
    let mut executions = Vec::new();
    let mut layouts = Vec::new();
    for depth in [1_usize, 16] {
        let fixtures = stack_fixtures(depth);
        let layer_parameters = stack_runtime_parameters(&fixtures);
        let invocation = stack_invocation(&fixtures, &layer_parameters, &post_weight, &post_bias);
        let expected = cpu_stack(&fixtures, &[], &post_weight, &post_bias);
        let plan = invocation.plan(&[]).unwrap();
        let layout = plan
            .activation_layout(activation_layout_config(&runtime, true))
            .unwrap();
        let execution = runtime
            .run_vision_encoder_stack_identity_rope_with_activation_strategy(
                &invocation,
                &[],
                VisionStackActivationStrategy::StaticArenaAlias,
            )
            .unwrap();
        let events = observer.take();
        assert_eq!(raw_activation_allocations(&events).len(), 3);
        assert_stack_close(&expected.output, &execution.output);
        assert_eq!(
            execution.diagnostics.activation_strategy,
            VisionStackActivationStrategy::StaticArenaAlias
        );
        assert_eq!(execution.diagnostics.activation_buffer_count, 3);
        assert_eq!(
            execution.diagnostics.activation_arena_bytes,
            layout.total_activation_bytes
        );
        assert_eq!(
            execution.diagnostics.scratch_arena_bytes,
            layout.scratch_arena_bytes
        );
        assert_eq!(
            execution.diagnostics.main_buffers_bytes,
            layout.main_buffers_bytes
        );
        assert_eq!(
            execution.diagnostics.scratch_allocations,
            layout.scratch_allocations
        );
        assert_eq!(
            activation_allocation_bytes(&events),
            BTreeMap::from([
                (
                    "vision-stack-activation-main-a".to_owned(),
                    u64::try_from(TOKENS * HIDDEN * 4).unwrap()
                ),
                (
                    "vision-stack-activation-main-b".to_owned(),
                    u64::try_from(TOKENS * HIDDEN * 4).unwrap()
                ),
                (
                    "vision-stack-activation-scratch-arena".to_owned(),
                    layout.scratch_arena_bytes
                ),
            ])
        );
        executions.push(execution);
        layouts.push(layout);
    }

    assert_eq!(layouts[0], layouts[1]);
    assert_eq!(executions[0].diagnostics.activation_buffer_count, 3);
    assert_eq!(executions[1].diagnostics.activation_buffer_count, 3);
    assert_eq!(
        executions[0].diagnostics.activation_arena_bytes,
        executions[1].diagnostics.activation_arena_bytes
    );
    assert_eq!(
        executions[0].diagnostics.scratch_allocations,
        executions[1].diagnostics.scratch_allocations
    );
}

#[test]
fn native_stack_static_alias_is_request_isolated_and_deterministic() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();

    let post_weight = shifted_values(HIDDEN, 91, 1.0);
    let post_bias = values(HIDDEN, 92);
    let fixtures_a = stack_fixtures(4);
    let params_a = stack_runtime_parameters(&fixtures_a);
    let invocation_a = stack_invocation(&fixtures_a, &params_a, &post_weight, &post_bias);

    let mut fixtures_b = stack_fixtures(4);
    for value in &mut fixtures_b[3].mlp_fc2_bias {
        *value += 0.125;
    }
    let params_b = stack_runtime_parameters(&fixtures_b);
    let invocation_b = stack_invocation(&fixtures_b, &params_b, &post_weight, &post_bias);

    let first_a = runtime
        .run_vision_encoder_stack_identity_rope_with_activation_strategy(
            &invocation_a,
            &[0, 1, 2, 3],
            VisionStackActivationStrategy::StaticArenaAlias,
        )
        .unwrap();
    observer.take();
    let b = runtime
        .run_vision_encoder_stack_identity_rope_with_activation_strategy(
            &invocation_b,
            &[0, 1, 2, 3],
            VisionStackActivationStrategy::StaticArenaAlias,
        )
        .unwrap();
    observer.take();
    let second_a = runtime
        .run_vision_encoder_stack_identity_rope_with_activation_strategy(
            &invocation_a,
            &[0, 1, 2, 3],
            VisionStackActivationStrategy::StaticArenaAlias,
        )
        .unwrap();

    assert_eq!(first_a.output, second_a.output);
    assert_eq!(first_a.checkpoints, second_a.checkpoints);
    assert_ne!(first_a.output, b.output);
}
