use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use pvlc_cpu_ref::{
    LayerNormParameters as CpuLayerNormParameters, LinearParameters as CpuLinearParameters,
    ProjectorParameters as CpuProjectorParameters, ProjectorTrace, gelu_erf_f32,
    projector_f32 as cpu_projector_f32, projector_merge_2x2_f32,
};
use pvlc_runtime_core::{
    KernelId, KernelInvocation, ProjectorInvocation, ProjectorParameters, ProjectorReadback,
    ProjectorStage, VisionLayerNormParameters, VisionLinearParameters,
};
use pvlc_runtime_native::{
    BackendKind, NativeOptions, NativeRuntime, RuntimeErrorCode, RuntimeEvent, RuntimeObserver,
};

const HIDDEN: usize = 3;
const MERGED: usize = HIDDEN * 4;
const OUTPUT: usize = 5;
const EPSILON: f32 = 1.0e-5;
const GRIDS: [[u32; 3]; 2] = [[1, 2, 4], [2, 2, 2]];

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
    input: Vec<f32>,
    pre_norm_weight: Vec<f32>,
    pre_norm_bias: Vec<f32>,
    linear1_weight: Vec<f32>,
    linear1_bias: Vec<f32>,
    linear2_weight: Vec<f32>,
    linear2_bias: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            input: values(16 * HIDDEN, 1),
            pre_norm_weight: shifted_values(HIDDEN, 2, 1.0),
            pre_norm_bias: values(HIDDEN, 3),
            linear1_weight: values(MERGED * MERGED, 4),
            linear1_bias: values(MERGED, 5),
            linear2_weight: values(OUTPUT * MERGED, 6),
            linear2_bias: values(OUTPUT, 7),
        }
    }

    fn invocation(&self) -> ProjectorInvocation<'_> {
        ProjectorInvocation {
            hidden_size: HIDDEN as u32,
            output_size: OUTPUT as u32,
            layer_norm_epsilon: EPSILON,
            input: &self.input,
            image_grid_thw: &GRIDS,
            parameters: ProjectorParameters {
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
            },
        }
    }

    fn cpu_trace(&self) -> ProjectorTrace {
        let grids = GRIDS.map(|grid| grid.map(|dimension| dimension as usize));
        cpu_projector_f32(
            &self.input,
            HIDDEN,
            &grids,
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
            },
            EPSILON,
        )
        .unwrap()
    }
}

fn values(length: usize, seed: u32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let phase = (index as f32 + 1.0) * (seed as f32 + 0.25) * 0.031;
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
        observer: Some(observer.clone()),
    }) {
        Ok(runtime) => {
            if env_flag("PVLC_REQUIRE_M4_METAL") {
                assert_eq!(runtime.capabilities().backend, BackendKind::Metal);
                assert!(runtime.capabilities().adapter_name.contains("M4 Pro"));
            }
            observer.take();
            Some(runtime)
        }
        Err(error) if env_flag("PVLC_REQUIRE_NATIVE_GPU") || env_flag("PVLC_REQUIRE_M4_METAL") => {
            panic!("native GPU is required: {error}")
        }
        Err(error) => {
            eprintln!("skipping native projector contract: {error}");
            None
        }
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index={index} actual={actual:?} expected={expected:?} tolerance={tolerance:?}"
        );
    }
}

fn expected_stage(trace: &ProjectorTrace, stage: ProjectorStage) -> &[f32] {
    match stage {
        ProjectorStage::PreNorm => &trace.pre_norm,
        ProjectorStage::Merge => &trace.merged,
        ProjectorStage::Linear1 => &trace.linear1,
        ProjectorStage::Activation => &trace.activation,
        ProjectorStage::Linear2 => &trace.output,
    }
}

#[test]
fn native_merge_and_exact_gelu_kernels_match_cpu_on_order_boundaries_and_tails() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(observer) else {
        return;
    };

    let grids = GRIDS.map(|grid| grid.map(|dimension| dimension as usize));
    let input = (0..16 * HIDDEN)
        .map(|value| value as f32 * 0.25 - 3.0)
        .collect::<Vec<_>>();
    let expected_merge = projector_merge_2x2_f32(&input, HIDDEN, &grids).unwrap();
    let merge = runtime
        .run(&KernelInvocation::ProjectorMerge2x2F32 {
            output_tokens: 4,
            hidden_size: HIDDEN as u32,
            input,
            source_token_indices: vec![0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        })
        .unwrap();
    assert_eq!(merge.values, expected_merge);
    assert_eq!(merge.diagnostics.kernel, KernelId::ProjectorMerge2x2F32);
    assert!(merge.diagnostics.captured_errors.is_empty());

    let values = [
        -100.0_f32, -10.0, -3.0, -1.0, -0.1, -0.0, 0.0, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0,
    ];
    let expected = values.into_iter().map(gelu_erf_f32).collect::<Vec<_>>();
    let gelu = runtime
        .run(&KernelInvocation::GeluErfF32 {
            values: values.to_vec(),
        })
        .unwrap();
    assert_close(&gelu.values, &expected, 1.0e-6);
    assert_eq!(gelu.values[5].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(gelu.values[6], 0.0);
    assert_eq!(gelu.diagnostics.kernel, KernelId::GeluErfF32);
    assert!(gelu.diagnostics.captured_errors.is_empty());
}

#[test]
fn resident_projector_matches_every_cpu_stage_in_one_submission_and_output_only_is_bounded() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(observer.clone()) else {
        return;
    };
    let fixture = Fixture::new();
    let expected = fixture.cpu_trace();
    let execution = runtime
        .run_projector(&fixture.invocation(), ProjectorReadback::AllStages)
        .unwrap();

    assert_eq!(execution.checkpoints.len(), 5);
    for stage in ProjectorStage::ALL {
        assert_close(
            &execution.checkpoints[&stage],
            expected_stage(&expected, stage),
            match stage {
                ProjectorStage::PreNorm | ProjectorStage::Merge => 3.0e-6,
                ProjectorStage::Linear1 => 8.0e-6,
                ProjectorStage::Activation => 1.0e-5,
                ProjectorStage::Linear2 => 2.0e-5,
            },
        );
    }
    let diagnostics = &execution.diagnostics;
    assert_eq!(diagnostics.dispatch_stages, ProjectorStage::ALL);
    assert_eq!(
        diagnostics
            .shader_blake3
            .keys()
            .copied()
            .collect::<Vec<_>>(),
        [
            KernelId::LayerNormF32,
            KernelId::GeluErfF32,
            KernelId::VisionPatchProjectionF32,
            KernelId::ProjectorMerge2x2F32,
        ]
    );
    assert!(diagnostics.captured_errors.is_empty());
    assert_eq!(diagnostics.submission_count, 1);
    assert_eq!(diagnostics.command_buffer_count, 1);
    assert_eq!(diagnostics.compute_pass_count, 1);
    assert_eq!(diagnostics.dispatch_count, 5);
    assert_eq!(diagnostics.buffer_allocation_count, 15);
    assert_eq!(diagnostics.readback_buffer_count, 1);
    assert_eq!(diagnostics.readback_map_count, 1);
    assert_eq!(diagnostics.readback_bytes, 848);
    assert_eq!(diagnostics.resident_intermediate_bytes, 848);
    assert_eq!(diagnostics.resident_weight_bytes, 908);
    let events = observer.take();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, RuntimeEvent::SubmissionQueued { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, RuntimeEvent::ReadbackMapRequested { bytes: 848, .. }))
            .count(),
        1
    );

    let output_only = runtime
        .run_projector(&fixture.invocation(), ProjectorReadback::OutputOnly)
        .unwrap();
    assert_eq!(
        output_only.checkpoints.keys().copied().collect::<Vec<_>>(),
        [ProjectorStage::Linear2]
    );
    assert_close(
        &output_only.checkpoints[&ProjectorStage::Linear2],
        &expected.output,
        2.0e-5,
    );
    assert_eq!(output_only.diagnostics.readback_bytes, 80);
    assert_eq!(output_only.diagnostics.submission_count, 1);
    assert_eq!(output_only.diagnostics.command_buffer_count, 1);
    assert_eq!(output_only.diagnostics.compute_pass_count, 1);
    assert_eq!(output_only.diagnostics.dispatch_count, 5);
    assert_eq!(output_only.diagnostics.buffer_allocation_count, 15);
    assert_eq!(output_only.diagnostics.readback_buffer_count, 1);
    assert_eq!(output_only.diagnostics.readback_map_count, 1);
    assert_eq!(output_only.diagnostics.resident_intermediate_bytes, 848);
    assert_eq!(output_only.diagnostics.resident_weight_bytes, 908);
    assert!(output_only.diagnostics.captured_errors.is_empty());
    let output_only_events = observer.take();
    assert_eq!(
        output_only_events
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
    assert_eq!(
        output_only_events
            .iter()
            .filter_map(|event| match event {
                RuntimeEvent::ReadbackMapRequested { bytes, .. } => Some(*bytes),
                _ => None,
            })
            .collect::<Vec<_>>(),
        [80]
    );
}

#[test]
fn resident_projector_preserves_bidirectional_packed_image_isolation_at_every_stage() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(observer) else {
        return;
    };
    let baseline_fixture = Fixture::new();
    let baseline = runtime
        .run_projector(&baseline_fixture.invocation(), ProjectorReadback::AllStages)
        .unwrap();

    for poison_first in [true, false] {
        let mut poisoned = baseline_fixture.clone();
        let index = if poison_first { 1 } else { 8 * HIDDEN + 1 };
        poisoned.input[index] += 2.0;
        let actual = runtime
            .run_projector(&poisoned.invocation(), ProjectorReadback::AllStages)
            .unwrap();
        for stage in ProjectorStage::ALL {
            let first_image_elements = match stage {
                ProjectorStage::PreNorm => 8 * HIDDEN,
                ProjectorStage::Merge | ProjectorStage::Linear1 | ProjectorStage::Activation => {
                    2 * MERGED
                }
                ProjectorStage::Linear2 => 2 * OUTPUT,
            };
            let expected = &baseline.checkpoints[&stage];
            let observed = &actual.checkpoints[&stage];
            if poison_first {
                assert_ne!(
                    &observed[..first_image_elements],
                    &expected[..first_image_elements],
                    "stage={stage:?}"
                );
                assert_eq!(
                    &observed[first_image_elements..],
                    &expected[first_image_elements..],
                    "stage={stage:?}"
                );
            } else {
                assert_eq!(
                    &observed[..first_image_elements],
                    &expected[..first_image_elements],
                    "stage={stage:?}"
                );
                assert_ne!(
                    &observed[first_image_elements..],
                    &expected[first_image_elements..],
                    "stage={stage:?}"
                );
            }
        }
    }
}

fn nonce_source(kernel: KernelId) -> String {
    let source = pvlc_wgsl::module(kernel).unwrap().source;
    let mutated = match kernel {
        KernelId::LayerNormF32 => source.replace(
            " + bias.data[column];",
            " + bias.data[column] + 0.125;",
        ),
        KernelId::ProjectorMerge2x2F32 => source.replace(
            "output.data[index] = input.data[source_token * params.hidden_size + channel];",
            "output.data[index] = input.data[source_token * params.hidden_size + channel] + 0.125;",
        ),
        KernelId::VisionPatchProjectionF32 => source.replace(
            "accumulated[output_offset];",
            "accumulated[output_offset] + 0.125;",
        ),
        KernelId::GeluErfF32 => source.replace(
            "output.data[index] = gelu;",
            "output.data[index] = gelu + 0.125;",
        ),
        _ => panic!("kernel {kernel} is not a physical projector kernel"),
    };
    assert_ne!(mutated, source, "nonce injection missed kernel {kernel}");
    mutated
}

#[test]
fn every_physical_projector_shader_drives_its_direct_stage_and_final_output() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(observer) else {
        return;
    };
    let fixture = Fixture::new();
    let baseline = runtime
        .run_projector(&fixture.invocation(), ProjectorReadback::AllStages)
        .unwrap();
    for (kernel, direct_stage) in [
        (KernelId::LayerNormF32, ProjectorStage::PreNorm),
        (KernelId::ProjectorMerge2x2F32, ProjectorStage::Merge),
        (KernelId::VisionPatchProjectionF32, ProjectorStage::Linear1),
        (KernelId::GeluErfF32, ProjectorStage::Activation),
    ] {
        let overrides = BTreeMap::from([(kernel, nonce_source(kernel))]);
        let mutated = runtime
            .run_projector_with_shader_overrides(
                &fixture.invocation(),
                ProjectorReadback::AllStages,
                &overrides,
            )
            .unwrap();
        let target = ProjectorStage::ALL
            .iter()
            .position(|stage| *stage == direct_stage)
            .unwrap();
        for stage in &ProjectorStage::ALL[..target] {
            assert_eq!(
                mutated.checkpoints[stage], baseline.checkpoints[stage],
                "kernel={kernel} changed predecessor {stage:?}"
            );
        }
        assert_ne!(
            mutated.checkpoints[&direct_stage], baseline.checkpoints[&direct_stage],
            "kernel={kernel} did not drive {direct_stage:?}"
        );
        assert_ne!(
            mutated.checkpoints[&ProjectorStage::Linear2],
            baseline.checkpoints[&ProjectorStage::Linear2],
            "kernel={kernel} did not propagate to final output"
        );
        assert_ne!(
            mutated.diagnostics.shader_blake3[&kernel],
            baseline.diagnostics.shader_blake3[&kernel]
        );
    }
}

#[test]
fn shared_projection_shader_is_executed_again_for_linear2_after_activation() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(observer) else {
        return;
    };
    let fixture = Fixture::new();
    let baseline = runtime
        .run_projector(&fixture.invocation(), ProjectorReadback::AllStages)
        .unwrap();
    let kernel = KernelId::VisionPatchProjectionF32;
    let source = pvlc_wgsl::module(kernel).unwrap().source;
    let mutated_source = source.replace(
        "output.data[output_row * params.output_width + output_column] =\n                        accumulated[output_offset];",
        "let linear2_nonce = select(0.0, 0.125, params.output_width == 5u);\n                    output.data[output_row * params.output_width + output_column] =\n                        accumulated[output_offset] + linear2_nonce;",
    );
    assert_ne!(mutated_source, source, "linear2 nonce injection missed");
    let overrides = BTreeMap::from([(kernel, mutated_source)]);
    let mutated = runtime
        .run_projector_with_shader_overrides(
            &fixture.invocation(),
            ProjectorReadback::AllStages,
            &overrides,
        )
        .unwrap();

    for stage in &ProjectorStage::ALL[..4] {
        assert_eq!(
            mutated.checkpoints[stage], baseline.checkpoints[stage],
            "linear2-only nonce changed predecessor {stage:?}"
        );
    }
    assert_ne!(
        mutated.checkpoints[&ProjectorStage::Linear2],
        baseline.checkpoints[&ProjectorStage::Linear2]
    );
    assert_ne!(
        mutated.diagnostics.shader_blake3[&kernel],
        baseline.diagnostics.shader_blake3[&kernel]
    );
}

#[test]
fn invalid_projector_is_rejected_before_gpu_allocation_submission_or_scope_activity() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(observer.clone()) else {
        return;
    };
    let mut fixture = Fixture::new();
    let last = fixture.linear2_weight.len() - 1;
    fixture.linear2_weight[last] = f32::NAN;
    let before = runtime.counters();
    let error = runtime
        .run_projector(&fixture.invocation(), ProjectorReadback::AllStages)
        .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::InvalidInvocation);
    assert_eq!(runtime.counters(), before);
    assert!(observer.take().is_empty());
}
