use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    sync::{Arc, Mutex},
};

use pvlc_cpu_ref::{
    KvBlockOrder, LayerNormParameters as CpuLayerNormParameters,
    LinearParameters as CpuLinearParameters, VisionEncoderLayerConfig as CpuLayerConfig,
    VisionEncoderLayerParameters as CpuLayerParameters, VisionEncoderLayerTrace,
    VisionEncoderStackConfig as CpuStackConfig, VisionEncoderStackTrace,
    vision_encoder_layer_identity_rope_f32, vision_encoder_stack_identity_rope_f32,
};
use pvlc_ir::SemanticGraph;
use pvlc_model_schema::{TensorDtype, TensorSpec};
use pvlc_passes::{
    VisionQkvStackOverlayErrorCode, VisionQkvStackSelection,
    build_verified_vision_qkv_stack_overlay, select_vision_qkv_stack_overlay,
};
use pvlc_runtime_core::{
    KernelId, KernelInvocation, VisionEncoderLayerGeometry, VisionEncoderLayerParameters,
    VisionEncoderStackInvocation, VisionLayerNormParameters, VisionLinearParameters,
    VisionQkvCanaryKind, VisionQkvCopyPurpose, VisionQkvExecutionPolicy, VisionQkvFusedInvocation,
    VisionQkvFusedTargetLimits, VisionQkvMapPurpose, VisionQkvSelectionOutcome,
    VisionQkvStackStage, VisionStackActivationLayoutConfig, VisionStackActivationStrategy,
};
use pvlc_runtime_native::{
    NativeOptions, NativeRuntime, RuntimeCounters, RuntimeErrorCode, RuntimeEvent, RuntimeObserver,
};
use pvlc_testkit::{ComparisonAxes, ComparisonPolicy, compare_f32};

const TOKENS: usize = 3;
const HIDDEN: usize = 4;
const HEADS: usize = 2;
const HEAD_DIM: usize = 2;
const INTERMEDIATE: usize = 7;
const EPSILON: f32 = 1.0e-5;

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
struct LayerFixture {
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

impl LayerFixture {
    fn seeded(layer: usize) -> Self {
        let seed = 101 + u32::try_from(layer).unwrap() * 97;
        Self {
            norm1_weight: shifted_values(HIDDEN, seed, 1.0, 0.025),
            norm1_bias: values(HIDDEN, seed + 1, 0.0125),
            query_weight: matrix(HIDDEN, HIDDEN, seed + 2, 0.075),
            query_bias: routed_bias(HIDDEN, layer, 0),
            key_weight: matrix(HIDDEN, HIDDEN, seed + 3, -0.0625),
            key_bias: routed_bias(HIDDEN, layer, 1),
            value_weight: matrix(HIDDEN, HIDDEN, seed + 4, 0.05),
            value_bias: routed_bias(HIDDEN, layer, 2),
            attention_output_weight: matrix(HIDDEN, HIDDEN, seed + 5, 0.04),
            attention_output_bias: values(HIDDEN, seed + 6, 0.01),
            norm2_weight: shifted_values(HIDDEN, seed + 7, 1.0, 0.02),
            norm2_bias: values(HIDDEN, seed + 8, 0.01),
            mlp_fc1_weight: matrix(INTERMEDIATE, HIDDEN, seed + 9, 0.035),
            mlp_fc1_bias: values(INTERMEDIATE, seed + 10, 0.01),
            mlp_fc2_weight: matrix(HIDDEN, INTERMEDIATE, seed + 11, 0.03),
            mlp_fc2_bias: values(HIDDEN, seed + 12, 0.01),
        }
    }

    fn runtime_parameters(&self) -> VisionEncoderLayerParameters<'_> {
        VisionEncoderLayerParameters {
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
}

struct StackFixture {
    input: Vec<f32>,
    boundaries: Vec<u32>,
    layers: Vec<LayerFixture>,
    post_weight: Vec<f32>,
    post_bias: Vec<f32>,
}

struct CpuStackEvidence {
    stack: VisionEncoderStackTrace,
    layers: Vec<VisionEncoderLayerTrace>,
}

fn values(length: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let integer = ((u32::try_from(index).unwrap() * 37 + seed * 19) % 113) as i32 - 56;
            integer as f32 * scale
        })
        .collect()
}

fn shifted_values(length: usize, seed: u32, shift: f32, scale: f32) -> Vec<f32> {
    values(length, seed, scale)
        .into_iter()
        .map(|value| value + shift)
        .collect()
}

fn matrix(rows: usize, columns: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..rows)
        .flat_map(|row| {
            (0..columns).map(move |column| {
                let routed = (row * 29 + column * 11 + usize::try_from(seed).unwrap() * 7) % 41;
                (routed as f32 - 20.0) * scale + row as f32 * 0.007 - column as f32 * 0.003
            })
        })
        .collect()
}

fn routed_bias(length: usize, layer: usize, projection: usize) -> Vec<f32> {
    (0..length)
        .map(|channel| {
            0.125 * (projection + 1) as f32 + 0.03125 * channel as f32 + 0.0078125 * layer as f32
        })
        .collect()
}

fn fixture(depth: usize) -> StackFixture {
    StackFixture {
        input: matrix(TOKENS, HIDDEN, 17, 0.08),
        boundaries: vec![0, 1, TOKENS as u32],
        layers: (0..depth).map(LayerFixture::seeded).collect(),
        post_weight: shifted_values(HIDDEN, 809, 1.0, 0.02),
        post_bias: values(HIDDEN, 811, 0.01),
    }
}

fn runtime_parameters(fixtures: &[LayerFixture]) -> Vec<VisionEncoderLayerParameters<'_>> {
    fixtures
        .iter()
        .map(LayerFixture::runtime_parameters)
        .collect()
}

fn stack_invocation<'a>(
    fixture: &'a StackFixture,
    parameters: &'a [VisionEncoderLayerParameters<'a>],
) -> VisionEncoderStackInvocation<'a> {
    VisionEncoderStackInvocation {
        tokens: TOKENS as u32,
        hidden_size: HIDDEN as u32,
        attention_heads: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        intermediate_size: INTERMEDIATE as u32,
        layer_norm_epsilon: EPSILON,
        input: &fixture.input,
        cu_seqlens: &fixture.boundaries,
        layer_parameters: parameters,
        post_norm: VisionLayerNormParameters {
            weight: &fixture.post_weight,
            bias: &fixture.post_bias,
        },
    }
}

fn cpu_stack(fixture: &StackFixture, checkpoints: &[usize]) -> CpuStackEvidence {
    let boundaries = fixture
        .boundaries
        .iter()
        .map(|value| *value as usize)
        .collect::<Vec<_>>();
    let mut layers = Vec::with_capacity(fixture.layers.len());
    let stack = vision_encoder_stack_identity_rope_f32(
        &fixture.input,
        CpuStackConfig {
            tokens: TOKENS,
            hidden_size: HIDDEN,
            layers: fixture.layers.len(),
            layer_norm_epsilon: EPSILON,
        },
        checkpoints,
        CpuLayerNormParameters {
            weight: &fixture.post_weight,
            bias: &fixture.post_bias,
        },
        |layer, input| {
            let trace = vision_encoder_layer_identity_rope_f32(
                input,
                CpuLayerConfig {
                    tokens: TOKENS,
                    hidden_size: HIDDEN,
                    attention_heads: HEADS,
                    head_dim: HEAD_DIM,
                    intermediate_size: INTERMEDIATE,
                    layer_norm_epsilon: EPSILON,
                    attention_key_tile: 2,
                    attention_order: KvBlockOrder::Forward,
                },
                &boundaries,
                fixture.layers[layer].cpu_parameters(),
            )?;
            let output = trace.output.clone();
            layers.push(trace);
            Ok(output)
        },
    )
    .unwrap();
    CpuStackEvidence { stack, layers }
}

fn layer_plan() -> pvlc_runtime_core::VisionEncoderLayerPlan {
    VisionEncoderLayerGeometry {
        tokens: TOKENS as u32,
        hidden_size: HIDDEN as u32,
        attention_heads: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        intermediate_size: INTERMEDIATE as u32,
        layer_norm_epsilon: EPSILON,
        cu_seqlens: &[0, 1, TOKENS as u32],
    }
    .plan()
    .unwrap()
}

fn alternate_token_layer_plan() -> pvlc_runtime_core::VisionEncoderLayerPlan {
    VisionEncoderLayerGeometry {
        tokens: 2,
        hidden_size: HIDDEN as u32,
        attention_heads: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        intermediate_size: INTERMEDIATE as u32,
        layer_norm_epsilon: EPSILON,
        cu_seqlens: &[0, 2],
    }
    .plan()
    .unwrap()
}

#[derive(Clone, Copy)]
enum Role {
    Query,
    Key,
    Value,
}

impl Role {
    const ALL: [Self; 3] = [Self::Query, Self::Key, Self::Value];

    const fn letter(self) -> &'static str {
        match self {
            Self::Query => "q",
            Self::Key => "k",
            Self::Value => "v",
        }
    }

    const fn projection(self) -> &'static str {
        match self {
            Self::Query => "q_proj",
            Self::Key => "k_proj",
            Self::Value => "v_proj",
        }
    }
}

fn tensor_catalog(depth: usize) -> Vec<TensorSpec> {
    let mut catalog = Vec::with_capacity(depth * 6);
    for layer in 0..depth {
        for role in Role::ALL {
            let prefix = format!(
                "visual.vision_model.encoder.layers.{layer}.self_attn.{}",
                role.projection()
            );
            let semantic = format!("vision.layer.{layer:02}.attention.{}", role.letter());
            catalog.push(TensorSpec {
                name: format!("{prefix}.weight"),
                dtype: TensorDtype::BFloat16,
                shape: vec![HIDDEN as u64, HIDDEN as u64],
                semantic_id: format!("{semantic}.weight"),
            });
            catalog.push(TensorSpec {
                name: format!("{prefix}.bias"),
                dtype: TensorDtype::BFloat16,
                shape: vec![HIDDEN as u64],
                semantic_id: format!("{semantic}.bias"),
            });
        }
    }
    catalog
}

fn target(runtime: &NativeRuntime) -> VisionQkvFusedTargetLimits {
    let capabilities = runtime.capabilities();
    VisionQkvFusedTargetLimits {
        min_storage_buffer_offset_alignment: capabilities.min_storage_buffer_offset_alignment,
        max_storage_buffers_per_shader_stage: capabilities.max_storage_buffers_per_shader_stage,
        max_storage_buffer_binding_size: capabilities.max_storage_buffer_binding_size,
        max_buffer_size: capabilities.max_buffer_size,
        max_compute_workgroups_per_dimension: capabilities.max_compute_workgroups_per_dimension,
    }
}

fn fused_selection_with_policy(
    runtime: &NativeRuntime,
    depth: usize,
    policy: VisionQkvExecutionPolicy,
) -> VisionQkvStackSelection {
    assert!(matches!(
        policy,
        VisionQkvExecutionPolicy::Preferred | VisionQkvExecutionPolicy::Required
    ));
    let graph = SemanticGraph::paddleocr_vl_16();
    let catalog = tensor_catalog(depth);
    let selection = select_vision_qkv_stack_overlay(policy, || {
        build_verified_vision_qkv_stack_overlay(
            &graph,
            depth,
            &layer_plan(),
            &catalog,
            target(runtime),
        )
    })
    .unwrap();
    assert_eq!(selection.policy(), policy);
    assert_eq!(selection.outcome(), VisionQkvSelectionOutcome::Fused);
    selection
}

fn fused_selection(runtime: &NativeRuntime, depth: usize) -> VisionQkvStackSelection {
    fused_selection_with_policy(runtime, depth, VisionQkvExecutionPolicy::Required)
}

fn selected_plan_identities(selection: &VisionQkvStackSelection) -> Vec<String> {
    selection
        .overlay()
        .into_iter()
        .flat_map(|overlay| overlay.layers())
        .map(|layer| layer.canonical_plan_blake3_hex().to_owned())
        .collect()
}

fn differing_sufficient_target(runtime: &NativeRuntime) -> VisionQkvFusedTargetLimits {
    let actual = target(runtime);
    VisionQkvFusedTargetLimits {
        min_storage_buffer_offset_alignment: actual.min_storage_buffer_offset_alignment,
        max_storage_buffers_per_shader_stage: actual.max_storage_buffers_per_shader_stage.min(8),
        max_storage_buffer_binding_size: actual.max_storage_buffer_binding_size.min(4_096),
        max_buffer_size: actual.max_buffer_size.min(4_096),
        max_compute_workgroups_per_dimension: actual
            .max_compute_workgroups_per_dimension
            .min(1_024),
    }
}

fn disabled_selection() -> VisionQkvStackSelection {
    select_vision_qkv_stack_overlay(VisionQkvExecutionPolicy::Disabled, || {
        panic!("Disabled must not inspect poisoned compiler input")
    })
    .unwrap()
}

fn env_flag(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1" | "true"))
}

fn runtime(observer: Arc<RecordingObserver>) -> Option<NativeRuntime> {
    match NativeRuntime::new(NativeOptions {
        observer: Some(observer),
    }) {
        Ok(runtime) => Some(runtime),
        Err(error) if env_flag("PVLC_REQUIRE_NATIVE_GPU") => {
            panic!("native GPU is required: {error}")
        }
        Err(error) => {
            eprintln!("skipping native QKV stack-overlay contract: {error}");
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

fn assert_stack_close(expected: &[f32], actual: &[f32], context: &str) {
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
    assert!(verdict.passed(), "{context}: {report:#?}\n{verdict:#?}");
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExpectedDispatch {
    layer: Option<usize>,
    stage: VisionQkvStackStage,
    kernel: KernelId,
    workgroups: [u32; 3],
}

fn expected_binding_count(stage: VisionQkvStackStage) -> usize {
    match stage {
        VisionQkvStackStage::Norm1
        | VisionQkvStackStage::Query
        | VisionQkvStackStage::Key
        | VisionQkvStackStage::Value
        | VisionQkvStackStage::AttentionOutput
        | VisionQkvStackStage::Norm2
        | VisionQkvStackStage::MlpFc1
        | VisionQkvStackStage::MlpOutput
        | VisionQkvStackStage::PostNorm => 5,
        VisionQkvStackStage::QkvFused => 9,
        VisionQkvStackStage::AttentionContext => 6,
        VisionQkvStackStage::AttentionResidual | VisionQkvStackStage::Output => 4,
        VisionQkvStackStage::MlpActivation => 3,
    }
}

fn expected_dispatches(depth: usize, fused: bool) -> Vec<ExpectedDispatch> {
    let plan = layer_plan();
    let mut expected = Vec::with_capacity((if fused { 10 } else { 12 }) * depth + 1);
    for layer in 0..depth {
        let legacy = plan.dispatches;
        expected.push(ExpectedDispatch {
            layer: Some(layer),
            stage: VisionQkvStackStage::Norm1,
            kernel: legacy[0].invocation.kernel,
            workgroups: legacy[0].invocation.dispatch,
        });
        if fused {
            expected.push(ExpectedDispatch {
                layer: Some(layer),
                stage: VisionQkvStackStage::QkvFused,
                kernel: KernelId::VisionQkvFusedF32,
                workgroups: [(HIDDEN as u32).div_ceil(8), (TOKENS as u32).div_ceil(8), 3],
            });
        } else {
            for (index, stage) in [
                VisionQkvStackStage::Query,
                VisionQkvStackStage::Key,
                VisionQkvStackStage::Value,
            ]
            .into_iter()
            .enumerate()
            {
                expected.push(ExpectedDispatch {
                    layer: Some(layer),
                    stage,
                    kernel: legacy[index + 1].invocation.kernel,
                    workgroups: legacy[index + 1].invocation.dispatch,
                });
            }
        }
        for (index, stage) in [
            VisionQkvStackStage::AttentionContext,
            VisionQkvStackStage::AttentionOutput,
            VisionQkvStackStage::AttentionResidual,
            VisionQkvStackStage::Norm2,
            VisionQkvStackStage::MlpFc1,
            VisionQkvStackStage::MlpActivation,
            VisionQkvStackStage::MlpOutput,
            VisionQkvStackStage::Output,
        ]
        .into_iter()
        .enumerate()
        {
            let legacy_index = index + 4;
            expected.push(ExpectedDispatch {
                layer: Some(layer),
                stage,
                kernel: legacy[legacy_index].invocation.kernel,
                workgroups: legacy[legacy_index].invocation.dispatch,
            });
        }
    }
    expected.push(ExpectedDispatch {
        layer: None,
        stage: VisionQkvStackStage::PostNorm,
        kernel: plan.dispatches[0].invocation.kernel,
        workgroups: plan.dispatches[0].invocation.dispatch,
    });
    expected
}

fn checkpoints(depth: usize) -> Vec<usize> {
    [0, depth / 2, depth - 1]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn activation_allocations(events: &[RuntimeEvent]) -> BTreeMap<String, u64> {
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

fn observer_topology(events: &[RuntimeEvent]) -> (Vec<u32>, Vec<(String, u64)>) {
    let submissions = events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::SubmissionQueued {
                command_buffers, ..
            } => Some(*command_buffers),
            _ => None,
        })
        .collect();
    let maps = events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::ReadbackMapRequested { label, bytes } => Some((label.clone(), *bytes)),
            _ => None,
        })
        .collect();
    (submissions, maps)
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExpectedSlice {
    binding: u32,
    offset: u64,
    length: u64,
}

fn independent_qkv_layout(runtime: &NativeRuntime) -> (u64, u64, [ExpectedSlice; 3]) {
    let alignment = u64::from(runtime.capabilities().min_storage_buffer_offset_alignment);
    let plane_bytes = u64::try_from(TOKENS * HIDDEN * 4).unwrap();
    let stride = align_up(plane_bytes, alignment);
    let semantic_bytes = stride * 3;
    let slices = std::array::from_fn(|binding| ExpectedSlice {
        binding: u32::try_from(binding).unwrap(),
        offset: u64::try_from(binding).unwrap() * stride,
        length: plane_bytes,
    });
    (stride, semantic_bytes, slices)
}

fn assert_single_submit_and_final_map(events: &[RuntimeEvent]) {
    let submission_positions = events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| {
            matches!(event, RuntimeEvent::SubmissionQueued { .. }).then_some(index)
        })
        .collect::<Vec<_>>();
    let map_positions = events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| {
            matches!(event, RuntimeEvent::ReadbackMapRequested { .. }).then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(submission_positions.len(), 1);
    assert_eq!(map_positions.len(), 1);
    assert!(submission_positions[0] < map_positions[0]);
    assert!(
        !events[map_positions[0] + 1..]
            .iter()
            .any(|event| matches!(event, RuntimeEvent::SubmissionQueued { .. }))
    );
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ObservedPipelineCreation {
    kernel: KernelId,
    shader_blake3: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ObservedBufferBinding {
    binding: u32,
    buffer_identity: u64,
    byte_offset: u64,
    byte_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ObservedBindGroupCreation {
    layer: Option<usize>,
    stage: VisionQkvStackStage,
    bindings: Vec<ObservedBufferBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ObservedDispatchEncoding {
    ordinal: usize,
    layer: Option<usize>,
    stage: VisionQkvStackStage,
    kernel: KernelId,
    workgroups: [u32; 3],
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ObservedCopyEncoding {
    ordinal: usize,
    source_buffer_identity: u64,
    source_offset: u64,
    destination_buffer_identity: u64,
    destination_offset: u64,
    byte_length: u64,
    purpose: VisionQkvCopyPurpose,
    after_dispatch_ordinal: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ObservedMapRequest {
    purpose: VisionQkvMapPurpose,
    buffer_identity: u64,
    byte_offset: u64,
    byte_length: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct OperationSiteTrace {
    pipelines: Vec<ObservedPipelineCreation>,
    bind_groups: Vec<ObservedBindGroupCreation>,
    command_encoders: Vec<String>,
    dispatches: Vec<ObservedDispatchEncoding>,
    copies: Vec<ObservedCopyEncoding>,
    maps: Vec<ObservedMapRequest>,
}

fn observed_binding(
    binding: &pvlc_runtime_core::VisionQkvBufferBindingEvidence,
) -> ObservedBufferBinding {
    ObservedBufferBinding {
        binding: binding.binding,
        buffer_identity: binding.buffer_identity,
        byte_offset: binding.byte_offset,
        byte_length: binding.byte_length,
    }
}

fn operation_site_trace(events: &[RuntimeEvent]) -> OperationSiteTrace {
    let mut trace = OperationSiteTrace::default();
    for event in events {
        match event {
            RuntimeEvent::PipelineCreated {
                kernel,
                shader_blake3,
            } => trace.pipelines.push(ObservedPipelineCreation {
                kernel: *kernel,
                shader_blake3: *shader_blake3,
            }),
            RuntimeEvent::BindGroupCreated {
                layer,
                stage,
                bindings,
            } => trace.bind_groups.push(ObservedBindGroupCreation {
                layer: *layer,
                stage: *stage,
                bindings: bindings.iter().map(observed_binding).collect(),
            }),
            RuntimeEvent::CommandEncoderCreated { label } => {
                trace.command_encoders.push(label.clone());
            }
            RuntimeEvent::DispatchEncoded {
                ordinal,
                layer,
                stage,
                kernel,
                workgroups,
            } => trace.dispatches.push(ObservedDispatchEncoding {
                ordinal: *ordinal,
                layer: *layer,
                stage: *stage,
                kernel: *kernel,
                workgroups: *workgroups,
            }),
            RuntimeEvent::BufferCopyEncoded {
                ordinal,
                source_buffer_identity,
                source_offset,
                destination_buffer_identity,
                destination_offset,
                byte_length,
                purpose,
                after_dispatch_ordinal,
            } => trace.copies.push(ObservedCopyEncoding {
                ordinal: *ordinal,
                source_buffer_identity: *source_buffer_identity,
                source_offset: *source_offset,
                destination_buffer_identity: *destination_buffer_identity,
                destination_offset: *destination_offset,
                byte_length: *byte_length,
                purpose: *purpose,
                after_dispatch_ordinal: *after_dispatch_ordinal,
            }),
            RuntimeEvent::MapRequested {
                purpose,
                buffer_identity,
                byte_offset,
                byte_length,
            } => trace.maps.push(ObservedMapRequest {
                purpose: *purpose,
                buffer_identity: *buffer_identity,
                byte_offset: *byte_offset,
                byte_length: *byte_length,
            }),
            _ => {}
        }
    }
    trace
}

fn assert_operation_event_order(events: &[RuntimeEvent]) {
    let encoder = events
        .iter()
        .position(|event| matches!(event, RuntimeEvent::CommandEncoderCreated { .. }))
        .unwrap();
    let submission = events
        .iter()
        .position(|event| matches!(event, RuntimeEvent::SubmissionQueued { .. }))
        .unwrap();
    let bind_groups = events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| {
            matches!(event, RuntimeEvent::BindGroupCreated { .. }).then_some(index)
        })
        .collect::<Vec<_>>();
    let dispatches = events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| {
            matches!(event, RuntimeEvent::DispatchEncoded { .. }).then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(bind_groups.len(), dispatches.len());
    for (bind_group, dispatch) in bind_groups.into_iter().zip(dispatches) {
        assert!(bind_group < dispatch);
        assert!(encoder < dispatch);
        assert!(dispatch < submission);
    }
    assert!(events.iter().enumerate().all(|(index, event)| {
        !matches!(event, RuntimeEvent::BufferCopyEncoded { .. })
            || (encoder < index && index < submission)
    }));
    assert!(events.iter().enumerate().all(|(index, event)| {
        !matches!(event, RuntimeEvent::MapRequested { .. }) || submission < index
    }));
}

fn assert_raw_copy_temporal_causality(events: &[RuntimeEvent]) {
    let submission_position = events
        .iter()
        .position(|event| matches!(event, RuntimeEvent::SubmissionQueued { .. }))
        .expect("copy causality requires the physical submission event");
    let final_attention_position = events
        .iter()
        .rposition(|event| {
            matches!(
                event,
                RuntimeEvent::DispatchEncoded {
                    stage: VisionQkvStackStage::AttentionContext,
                    ..
                }
            )
        })
        .expect("copy causality requires a raw attention dispatch event");

    for (copy_position, event) in events.iter().enumerate() {
        let RuntimeEvent::BufferCopyEncoded {
            purpose,
            after_dispatch_ordinal,
            ..
        } = event
        else {
            continue;
        };

        assert!(
            copy_position < submission_position,
            "every encoded copy must precede the physical submission"
        );
        let latest_preceding_dispatch_ordinal = events[..copy_position]
            .iter()
            .rev()
            .find_map(|preceding| match preceding {
                RuntimeEvent::DispatchEncoded { ordinal, .. } => Some(*ordinal),
                _ => None,
            })
            .expect("every encoded copy must have a preceding raw dispatch event");
        assert_eq!(
            *after_dispatch_ordinal, latest_preceding_dispatch_ordinal,
            "returned copy metadata must agree with raw event order, not define it"
        );
        if *purpose == VisionQkvCopyPurpose::CanaryEvidence {
            assert!(
                copy_position > final_attention_position,
                "canary evidence must be copied strictly after the final raw attention dispatch"
            );
        }
    }
}

fn evidence_trace(execution: &pvlc_runtime_native::VisionQkvStackExecution) -> OperationSiteTrace {
    OperationSiteTrace {
        pipelines: execution
            .evidence
            .pipeline_creations
            .iter()
            .map(|pipeline| ObservedPipelineCreation {
                kernel: pipeline.kernel,
                shader_blake3: pipeline.shader_blake3,
            })
            .collect(),
        bind_groups: execution
            .evidence
            .bind_group_creations
            .iter()
            .map(|group| ObservedBindGroupCreation {
                layer: group.layer,
                stage: group.stage,
                bindings: group.bindings.iter().map(observed_binding).collect(),
            })
            .collect(),
        command_encoders: execution
            .evidence
            .command_encoder_creations
            .iter()
            .map(|encoder| encoder.label.clone())
            .collect(),
        dispatches: execution
            .evidence
            .encoded_dispatches
            .iter()
            .map(|dispatch| ObservedDispatchEncoding {
                ordinal: dispatch.ordinal,
                layer: dispatch.layer,
                stage: dispatch.stage,
                kernel: dispatch.kernel,
                workgroups: dispatch.workgroups,
            })
            .collect(),
        copies: execution
            .evidence
            .encoded_copies
            .iter()
            .map(|copy| ObservedCopyEncoding {
                ordinal: copy.ordinal,
                source_buffer_identity: copy.source_buffer_identity,
                source_offset: copy.source_offset,
                destination_buffer_identity: copy.destination_buffer_identity,
                destination_offset: copy.destination_offset,
                byte_length: copy.byte_length,
                purpose: copy.purpose,
                after_dispatch_ordinal: copy.after_dispatch_ordinal,
            })
            .collect(),
        maps: execution
            .evidence
            .map_requests
            .iter()
            .map(|map| ObservedMapRequest {
                purpose: map.purpose,
                buffer_identity: map.buffer_identity,
                byte_offset: map.byte_offset,
                byte_length: map.byte_length,
            })
            .collect(),
    }
}

fn assert_effect_counter_delta(
    before: RuntimeCounters,
    after: RuntimeCounters,
    operations: &OperationSiteTrace,
    execution: &pvlc_runtime_native::VisionQkvStackExecution,
) {
    assert_eq!(
        after.pipeline_creations - before.pipeline_creations,
        u64::try_from(operations.pipelines.len()).unwrap()
    );
    assert_eq!(
        after.bind_group_creations - before.bind_group_creations,
        u64::try_from(operations.bind_groups.len()).unwrap()
    );
    assert_eq!(
        after.command_encoder_creations - before.command_encoder_creations,
        u64::try_from(operations.command_encoders.len()).unwrap()
    );
    assert_eq!(
        after.dispatch_encodings - before.dispatch_encodings,
        u64::try_from(operations.dispatches.len()).unwrap()
    );
    assert_eq!(
        after.buffer_copy_encodings - before.buffer_copy_encodings,
        u64::try_from(operations.copies.len()).unwrap()
    );
    assert_eq!(
        after.map_requests - before.map_requests,
        u64::try_from(operations.maps.len()).unwrap()
    );
    assert_eq!(after.submissions - before.submissions, 1);
    assert_eq!(
        after.buffer_allocations - before.buffer_allocations,
        execution.diagnostics.buffer_allocation_count
    );
}

fn assert_error_effect_counter_delta(
    before: RuntimeCounters,
    after: RuntimeCounters,
    operations: &OperationSiteTrace,
) {
    assert_eq!(
        after.pipeline_creations - before.pipeline_creations,
        u64::try_from(operations.pipelines.len()).unwrap()
    );
    assert_eq!(
        after.bind_group_creations - before.bind_group_creations,
        u64::try_from(operations.bind_groups.len()).unwrap()
    );
    assert_eq!(
        after.command_encoder_creations - before.command_encoder_creations,
        u64::try_from(operations.command_encoders.len()).unwrap()
    );
    assert_eq!(
        after.dispatch_encodings - before.dispatch_encodings,
        u64::try_from(operations.dispatches.len()).unwrap()
    );
    assert_eq!(
        after.buffer_copy_encodings - before.buffer_copy_encodings,
        u64::try_from(operations.copies.len()).unwrap()
    );
    assert_eq!(
        after.map_requests - before.map_requests,
        u64::try_from(operations.maps.len()).unwrap()
    );
    assert_eq!(after.submissions - before.submissions, 1);
}

fn assert_operation_site_evidence(
    runtime: &NativeRuntime,
    before: RuntimeCounters,
    after: RuntimeCounters,
    execution: &pvlc_runtime_native::VisionQkvStackExecution,
    events: &[RuntimeEvent],
    expected: &[ExpectedDispatch],
    context: &str,
) {
    let observed = operation_site_trace(events);
    assert_operation_event_order(events);
    assert_raw_copy_temporal_causality(events);
    assert_eq!(
        observed,
        evidence_trace(execution),
        "{context}: returned evidence diverged from independent operation-site events"
    );
    assert_effect_counter_delta(before, after, &observed, execution);
    let expected_kernels = expected
        .iter()
        .map(|dispatch| dispatch.kernel)
        .collect::<BTreeSet<_>>();
    assert!(
        observed
            .pipelines
            .iter()
            .all(|pipeline| expected_kernels.contains(&pipeline.kernel))
    );
    assert_eq!(
        observed
            .pipelines
            .iter()
            .map(|pipeline| pipeline.kernel)
            .collect::<BTreeSet<_>>()
            .len(),
        observed.pipelines.len(),
        "one physical pipeline creation per kernel/source is expected in one request"
    );
    for pipeline in &observed.pipelines {
        assert_eq!(
            execution.diagnostics.shader_blake3.get(&pipeline.kernel),
            Some(&pipeline.shader_blake3)
        );
    }
    assert_eq!(observed.command_encoders.len(), 1);
    assert!(!observed.command_encoders[0].is_empty());
    assert_eq!(observed.bind_groups.len(), expected.len());
    assert_eq!(observed.dispatches.len(), expected.len());
    for (group, dispatch) in observed.bind_groups.iter().zip(expected) {
        assert_eq!(group.layer, dispatch.layer);
        assert_eq!(group.stage, dispatch.stage);
        assert_eq!(group.bindings.len(), expected_binding_count(dispatch.stage));
        assert_eq!(
            group
                .bindings
                .iter()
                .map(|binding| binding.binding)
                .collect::<Vec<_>>(),
            (0..u32::try_from(group.bindings.len()).unwrap()).collect::<Vec<_>>()
        );
        assert!(group.bindings.iter().all(|binding| {
            binding.buffer_identity != 0
                && binding.byte_length > 0
                && binding
                    .byte_offset
                    .checked_add(binding.byte_length)
                    .is_some()
        }));
    }
    assert_eq!(
        observed
            .dispatches
            .iter()
            .map(|dispatch| ExpectedDispatch {
                layer: dispatch.layer,
                stage: dispatch.stage,
                kernel: dispatch.kernel,
                workgroups: dispatch.workgroups,
            })
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(
        observed
            .dispatches
            .iter()
            .map(|dispatch| dispatch.ordinal)
            .collect::<Vec<_>>(),
        (0..expected.len()).collect::<Vec<_>>()
    );
    assert_eq!(
        observed
            .copies
            .iter()
            .map(|copy| copy.ordinal)
            .collect::<Vec<_>>(),
        (0..observed.copies.len()).collect::<Vec<_>>()
    );
    assert!(observed.copies.iter().all(|copy| {
        copy.source_buffer_identity != 0
            && copy.destination_buffer_identity != 0
            && copy.byte_length > 0
            && copy.source_offset.checked_add(copy.byte_length).is_some()
            && copy
                .destination_offset
                .checked_add(copy.byte_length)
                .is_some()
    }));
    let depth = expected
        .iter()
        .filter_map(|dispatch| dispatch.layer)
        .max()
        .unwrap()
        + 1;
    assert_eq!(
        observed
            .copies
            .iter()
            .filter(|copy| copy.purpose == VisionQkvCopyPurpose::Checkpoint)
            .count(),
        checkpoints(depth).len()
    );
    assert_eq!(
        observed
            .copies
            .iter()
            .filter(|copy| copy.purpose == VisionQkvCopyPurpose::SemanticOutput)
            .count(),
        1
    );
    let fused = expected
        .iter()
        .any(|dispatch| dispatch.stage == VisionQkvStackStage::QkvFused);
    let (stride, _, slices) = independent_qkv_layout(runtime);
    let internal_padding_regions = usize::from(stride > slices[0].length) * 3;
    assert_eq!(
        observed
            .copies
            .iter()
            .filter(|copy| copy.purpose == VisionQkvCopyPurpose::CanaryEvidence)
            .count(),
        if fused {
            2 + internal_padding_regions
        } else {
            0
        }
    );
    assert_eq!(
        observed
            .maps
            .iter()
            .filter(|map| map.purpose == VisionQkvMapPurpose::SemanticOutput)
            .count(),
        1
    );
    assert_eq!(
        observed
            .maps
            .iter()
            .filter(|map| map.purpose == VisionQkvMapPurpose::TimestampQuery)
            .count(),
        usize::from(runtime.capabilities().timestamp_query)
    );
    assert_eq!(
        observed.maps.len(),
        1 + usize::from(runtime.capabilities().timestamp_query)
    );
    assert_eq!(observed.maps.len(), execution.evidence.map_count);
    assert_eq!(
        observed
            .copies
            .iter()
            .filter(|copy| copy.purpose == VisionQkvCopyPurpose::TimestampQuery)
            .count(),
        usize::from(runtime.capabilities().timestamp_query)
    );
    if let Some(timestamp_map) = observed
        .maps
        .iter()
        .find(|map| map.purpose == VisionQkvMapPurpose::TimestampQuery)
    {
        assert_eq!(timestamp_map.byte_offset, 0);
        assert_eq!(timestamp_map.byte_length, 16);
        let timestamp_copy = observed
            .copies
            .iter()
            .find(|copy| copy.purpose == VisionQkvCopyPurpose::TimestampQuery)
            .unwrap();
        assert_eq!(
            timestamp_copy.destination_buffer_identity,
            timestamp_map.buffer_identity
        );
        assert_eq!(timestamp_copy.destination_offset, timestamp_map.byte_offset);
        assert_eq!(timestamp_copy.byte_length, timestamp_map.byte_length);
    }
    assert_eq!(execution.evidence.command_buffer_count, 1);
    assert_eq!(execution.evidence.submission_count, 1);
    assert_eq!(execution.diagnostics.command_buffer_count, 1);
    assert_eq!(execution.diagnostics.submission_count, 1);
    assert_eq!(execution.diagnostics.readback_map_count, 1);
    assert!(execution.diagnostics.captured_errors.is_empty());
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, RuntimeEvent::SubmissionQueued { .. }))
            .count(),
        1
    );
}

fn assert_fused_workspace_evidence(
    runtime: &NativeRuntime,
    depth: usize,
    execution: &pvlc_runtime_native::VisionQkvStackExecution,
    events: &[RuntimeEvent],
) {
    let alignment = u64::from(runtime.capabilities().min_storage_buffer_offset_alignment);
    let (stride, semantic_bytes, relative_slices) = independent_qkv_layout(runtime);
    let workspace = execution
        .evidence
        .workspace
        .as_ref()
        .expect("fused selection must report its separate physical workspace");
    assert!(!workspace.logical_buffer_id.is_empty());
    assert!(workspace.semantic_base > 0);
    assert_eq!(workspace.semantic_base % alignment, 0);
    assert_eq!(workspace.semantic_bytes, semantic_bytes);
    let semantic_end = workspace
        .semantic_base
        .checked_add(workspace.semantic_bytes)
        .unwrap();
    assert!(workspace.allocation_bytes > semantic_end);
    assert!(workspace.buffer_identity != 0);

    let workspace_allocations = events
        .iter()
        .filter_map(|event| match event {
            RuntimeEvent::BufferAllocated { label, bytes }
                if label == &workspace.logical_buffer_id =>
            {
                Some(*bytes)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(workspace_allocations, [workspace.allocation_bytes]);

    let expected_slices = relative_slices.map(|slice| ExpectedSlice {
        binding: slice.binding,
        offset: workspace.semantic_base + slice.offset,
        length: slice.length,
    });

    assert_eq!(execution.evidence.attention_bindings.len(), depth * 3);
    for layer in 0..depth {
        let bindings = &execution.evidence.attention_bindings[layer * 3..layer * 3 + 3];
        for (actual, expected) in bindings.iter().zip(expected_slices) {
            assert_eq!(actual.layer, layer);
            assert_eq!(actual.binding, expected.binding);
            assert_eq!(actual.buffer_identity, workspace.buffer_identity);
            assert_eq!(actual.byte_offset, expected.offset);
            assert_eq!(actual.byte_length, expected.length);
            assert_ne!(actual.byte_length, workspace.semantic_bytes);
            assert_ne!(actual.byte_length, workspace.allocation_bytes);
        }
        assert_eq!(
            bindings
                .iter()
                .map(|binding| binding.buffer_identity)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([workspace.buffer_identity])
        );
        assert!(bindings[0].byte_offset + bindings[0].byte_length <= bindings[1].byte_offset);
        assert!(bindings[1].byte_offset + bindings[1].byte_length <= bindings[2].byte_offset);
        assert!(bindings[2].byte_offset + bindings[2].byte_length <= semantic_end);
    }

    let plane_bytes = expected_slices[0].length;
    let expected_canaries = [
        (VisionQkvCanaryKind::Prefix, 0, workspace.semantic_base),
        (
            VisionQkvCanaryKind::InternalPadding { plane: 0 },
            expected_slices[0].offset + plane_bytes,
            stride - plane_bytes,
        ),
        (
            VisionQkvCanaryKind::InternalPadding { plane: 1 },
            expected_slices[1].offset + plane_bytes,
            stride - plane_bytes,
        ),
        (
            VisionQkvCanaryKind::InternalPadding { plane: 2 },
            expected_slices[2].offset + plane_bytes,
            stride - plane_bytes,
        ),
        (
            VisionQkvCanaryKind::Suffix,
            semantic_end,
            workspace.allocation_bytes - semantic_end,
        ),
    ]
    .into_iter()
    .filter(|(_, _, length)| *length != 0)
    .collect::<Vec<_>>();
    assert_eq!(execution.evidence.canaries.len(), expected_canaries.len());
    let expected_canary_bits = execution
        .evidence
        .canaries
        .first()
        .expect("nonzero prefix and suffix require canary records")
        .expected_bits;
    assert_ne!(expected_canary_bits, 0);
    for (actual, (kind, offset, byte_length)) in
        execution.evidence.canaries.iter().zip(expected_canaries)
    {
        assert_eq!(actual.kind, kind);
        assert_eq!(actual.byte_offset, offset);
        assert_eq!(actual.byte_length, byte_length);
        assert_eq!(actual.expected_bits, expected_canary_bits);
        assert!(actual.passed);
        assert!(actual.byte_length > 0);
        assert!(
            actual
                .byte_offset
                .checked_add(actual.byte_length)
                .is_some_and(|end| end <= workspace.allocation_bytes)
        );
    }

    for pair in execution.evidence.canaries.windows(2) {
        assert!(pair[0].byte_offset + pair[0].byte_length <= pair[1].byte_offset);
    }

    let operations = operation_site_trace(events);
    let canary_copies = operations
        .copies
        .iter()
        .filter(|copy| copy.purpose == VisionQkvCopyPurpose::CanaryEvidence)
        .collect::<Vec<_>>();
    assert_eq!(canary_copies.len(), execution.evidence.canaries.len());
    assert_eq!(
        canary_copies
            .iter()
            .map(|copy| copy.byte_length)
            .sum::<u64>(),
        execution
            .evidence
            .canaries
            .iter()
            .map(|canary| canary.byte_length)
            .sum::<u64>()
    );
    for (copy, canary) in canary_copies.iter().zip(&execution.evidence.canaries) {
        assert_eq!(copy.source_buffer_identity, workspace.buffer_identity);
        assert_eq!(copy.source_offset, canary.byte_offset);
        assert_eq!(copy.byte_length, canary.byte_length);
    }
    let semantic_map = operations
        .maps
        .iter()
        .find(|map| map.purpose == VisionQkvMapPurpose::SemanticOutput)
        .unwrap();
    for copy in &canary_copies {
        assert_eq!(
            copy.destination_buffer_identity,
            semantic_map.buffer_identity
        );
        assert!(copy.destination_offset >= semantic_map.byte_offset);
        assert!(
            copy.destination_offset + copy.byte_length
                <= semantic_map.byte_offset + semantic_map.byte_length
        );
    }
    for copies in canary_copies.windows(2) {
        assert!(
            copies[0].destination_offset + copies[0].byte_length <= copies[1].destination_offset,
            "canary evidence destinations must be in-order and pairwise disjoint"
        );
    }
    assert_eq!(
        execution
            .evidence
            .encoded_copies
            .iter()
            .filter(|copy| copy.purpose == VisionQkvCopyPurpose::Checkpoint)
            .count(),
        checkpoints(depth).len()
    );
    assert_eq!(
        execution
            .evidence
            .encoded_copies
            .iter()
            .filter(|copy| copy.purpose == VisionQkvCopyPurpose::SemanticOutput)
            .count(),
        1
    );
    assert_eq!(
        execution
            .evidence
            .encoded_copies
            .iter()
            .filter(|copy| copy.purpose == VisionQkvCopyPurpose::TimestampQuery)
            .count(),
        usize::from(runtime.capabilities().timestamp_query)
    );
    assert!(
        execution
            .evidence
            .encoded_copies
            .iter()
            .all(|copy| matches!(
                copy.purpose,
                VisionQkvCopyPurpose::Checkpoint
                    | VisionQkvCopyPurpose::SemanticOutput
                    | VisionQkvCopyPurpose::CanaryEvidence
                    | VisionQkvCopyPurpose::TimestampQuery
            ))
    );
    assert!(
        !execution
            .evidence
            .encoded_dispatches
            .iter()
            .any(|dispatch| {
                matches!(
                    dispatch.stage,
                    VisionQkvStackStage::Query
                        | VisionQkvStackStage::Key
                        | VisionQkvStackStage::Value
                )
            })
    );

    let attention_groups = operations
        .bind_groups
        .iter()
        .filter(|group| group.stage == VisionQkvStackStage::AttentionContext)
        .collect::<Vec<_>>();
    assert_eq!(attention_groups.len(), depth);
    for (layer, group) in attention_groups.into_iter().enumerate() {
        assert_eq!(group.layer, Some(layer));
        assert!(group.bindings.len() > 3);
        for (actual, expected) in group.bindings[..3].iter().zip(expected_slices) {
            assert_eq!(actual.binding, expected.binding);
            assert_eq!(actual.buffer_identity, workspace.buffer_identity);
            assert_eq!(actual.byte_offset, expected.offset);
            assert_eq!(actual.byte_length, expected.length);
        }
    }
}

fn layout_config(
    runtime: &NativeRuntime,
    strategy: VisionStackActivationStrategy,
) -> VisionStackActivationLayoutConfig {
    let alignment = u64::from(runtime.capabilities().min_storage_buffer_offset_alignment);
    VisionStackActivationLayoutConfig {
        allow_aliasing: strategy == VisionStackActivationStrategy::StaticArenaAlias,
        storage_buffer_offset_alignment: alignment,
        arena_alignment: alignment,
    }
}

#[derive(Clone, Copy)]
struct ProjectionRef<'a> {
    weight: &'a [f32],
    bias: &'a [f32],
}

fn layer_projections(layer: &LayerFixture) -> [ProjectionRef<'_>; 3] {
    [
        ProjectionRef {
            weight: &layer.query_weight,
            bias: &layer.query_bias,
        },
        ProjectionRef {
            weight: &layer.key_weight,
            bias: &layer.key_bias,
        },
        ProjectionRef {
            weight: &layer.value_weight,
            bias: &layer.value_bias,
        },
    ]
}

fn ordered_projection(
    input: &[f32],
    tokens: usize,
    input_width: usize,
    output_width: usize,
    projection: ProjectionRef<'_>,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; tokens * output_width];
    for token in 0..tokens {
        for channel in 0..output_width {
            let mut accumulator = projection.bias[channel];
            for depth in 0..input_width {
                accumulator += input[token * input_width + depth]
                    * projection.weight[channel * input_width + depth];
            }
            output[token * output_width + channel] = accumulator;
        }
    }
    output
}

fn transposed_projection(
    input: &[f32],
    tokens: usize,
    width: usize,
    projection: ProjectionRef<'_>,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; tokens * width];
    for token in 0..tokens {
        for channel in 0..width {
            let mut accumulator = projection.bias[channel];
            for depth in 0..width {
                accumulator +=
                    input[token * width + depth] * projection.weight[depth * width + channel];
            }
            output[token * width + channel] = accumulator;
        }
    }
    output
}

fn legacy_projection(
    runtime: &NativeRuntime,
    input: &[f32],
    tokens: usize,
    input_width: usize,
    output_width: usize,
    projection: ProjectionRef<'_>,
) -> Vec<f32> {
    runtime
        .run(&KernelInvocation::VisionPatchProjectionF32 {
            patch_count: tokens as u32,
            input_width: input_width as u32,
            output_width: output_width as u32,
            input: input.to_vec(),
            weight: projection.weight.to_vec(),
            bias: projection.bias.to_vec(),
        })
        .unwrap()
        .values
}

fn assert_projection_close(expected: &[f32], actual: &[f32], context: &str) {
    assert_eq!(expected.len(), actual.len(), "{context}");
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        let tolerance = 2.0e-5 * (1.0 + expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context} at {index}: expected={expected}, actual={actual}, tolerance={tolerance}"
        );
    }
}

fn fused_plane(values: &[f32], offset: u64, size: u64) -> &[f32] {
    let start = usize::try_from(offset / 4).unwrap();
    let length = usize::try_from(size / 4).unwrap();
    &values[start..start + length]
}

fn run_qkv_oracle_case(
    runtime: &NativeRuntime,
    input: &[f32],
    tokens: usize,
    width: usize,
    projections: [ProjectionRef<'_>; 3],
) -> [Vec<f32>; 3] {
    let invocation = VisionQkvFusedInvocation {
        tokens: tokens as u32,
        input_width: width as u32,
        output_width: width as u32,
        input,
        query_weight: projections[0].weight,
        query_bias: projections[0].bias,
        key_weight: projections[1].weight,
        key_bias: projections[1].bias,
        value_weight: projections[2].weight,
        value_bias: projections[2].bias,
    };
    let plan = invocation.plan(target(runtime)).unwrap();
    let expected: [Vec<f32>; 3] = std::array::from_fn(|projection| {
        ordered_projection(input, tokens, width, width, projections[projection])
    });
    let legacy: [Vec<f32>; 3] = std::array::from_fn(|projection| {
        legacy_projection(
            runtime,
            input,
            tokens,
            width,
            width,
            projections[projection],
        )
    });
    let fused = runtime.run_vision_qkv_fused(&invocation).unwrap();
    for (projection, slice) in [
        plan.output_layout.query,
        plan.output_layout.key,
        plan.output_layout.value,
    ]
    .into_iter()
    .enumerate()
    {
        let actual = fused_plane(&fused.values, slice.offset, slice.size);
        assert_projection_close(&expected[projection], actual, "ordered CPU/fused QKV");
        assert_projection_close(&legacy[projection], actual, "legacy GPU/fused QKV");
    }
    expected
}

#[test]
fn fused_qkv_arithmetic_uses_ordered_oracle_and_kills_transpose_permutation_bias_and_reassociation()
{
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(observer) else {
        return;
    };
    if runtime.capabilities().max_storage_buffers_per_shader_stage < 8 {
        return;
    }

    let fixture = fixture(3);
    let cpu = cpu_stack(&fixture, &[0, 1, 2]);
    for (layer, trace) in cpu.layers.iter().enumerate() {
        let projections = layer_projections(&fixture.layers[layer]);
        let expected = run_qkv_oracle_case(&runtime, &trace.norm1, TOKENS, HIDDEN, projections);
        for (actual, expected) in [
            (&trace.query, &expected[0]),
            (&trace.key, &expected[1]),
            (&trace.value, &expected[2]),
        ] {
            assert_projection_close(expected, actual, "existing CPU stack QKV oracle");
        }
        assert_ne!(
            expected[0], expected[1],
            "Q/K permutation must be observable"
        );
        assert_ne!(
            expected[1], expected[2],
            "K/V permutation must be observable"
        );
        assert_ne!(
            expected[0],
            transposed_projection(&trace.norm1, TOKENS, HIDDEN, projections[0]),
            "weight transpose must be observable"
        );
        let no_bias = ProjectionRef {
            weight: projections[2].weight,
            bias: &[0.0; HIDDEN],
        };
        assert_ne!(
            expected[2],
            ordered_projection(&trace.norm1, TOKENS, HIDDEN, HIDDEN, no_bias),
            "bias routing must be observable"
        );
    }

    let cancellation_input = vec![
        1.0e20, 1.0, -1.0e20, 2.0, -1.0e20, 3.0, 1.0e20, 4.0, 1.0e20, 5.0, -1.0e20, 6.0,
    ];
    let cancellation_weights = [
        vec![1.0; HIDDEN * HIDDEN],
        vec![0.5; HIDDEN * HIDDEN],
        vec![-0.25; HIDDEN * HIDDEN],
    ];
    let cancellation_biases = [
        vec![3.0, 5.0, 7.0, 11.0],
        vec![-2.0, -3.0, -5.0, -7.0],
        vec![13.0, 17.0, 19.0, 23.0],
    ];
    let projections = std::array::from_fn(|index| ProjectionRef {
        weight: &cancellation_weights[index],
        bias: &cancellation_biases[index],
    });
    let forward = run_qkv_oracle_case(&runtime, &cancellation_input, TOKENS, HIDDEN, projections);
    let mut reversed = cancellation_input.clone();
    for token in reversed.chunks_exact_mut(HIDDEN) {
        token.reverse();
    }
    let reverse_order = ordered_projection(&reversed, TOKENS, HIDDEN, HIDDEN, projections[0]);
    assert_ne!(
        forward[0], reverse_order,
        "the cancellation fixture must detect reassociated/reversed accumulation"
    );
}

#[test]
fn preferred_fused_accepts_same_alignment_with_different_sufficient_maxima_and_has_exact_operations()
 {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();
    let graph = SemanticGraph::paddleocr_vl_16();
    let catalog = tensor_catalog(3);
    let sufficient = differing_sufficient_target(&runtime);
    let actual = target(&runtime);
    assert_eq!(
        sufficient.min_storage_buffer_offset_alignment,
        actual.min_storage_buffer_offset_alignment
    );
    assert_ne!(
        sufficient.max_buffer_size, actual.max_buffer_size,
        "fixture must prove that sufficient maxima are requirements, not adapter identity"
    );
    let before_selection = runtime.counters();
    let selection = select_vision_qkv_stack_overlay(VisionQkvExecutionPolicy::Preferred, || {
        build_verified_vision_qkv_stack_overlay(&graph, 3, &layer_plan(), &catalog, sufficient)
    })
    .unwrap();
    assert_eq!(runtime.counters(), before_selection);
    assert!(observer.take().is_empty());
    assert_eq!(selection.policy(), VisionQkvExecutionPolicy::Preferred);
    assert_eq!(selection.outcome(), VisionQkvSelectionOutcome::Fused);

    let fixture = fixture(3);
    let parameters = runtime_parameters(&fixture.layers);
    let invocation = stack_invocation(&fixture, &parameters);
    let checkpoints = checkpoints(3);
    let cpu = cpu_stack(&fixture, &checkpoints);
    let before = runtime.counters();
    let execution = runtime
        .run_vision_encoder_stack_identity_rope_with_qkv_selection(
            &invocation,
            &checkpoints,
            VisionStackActivationStrategy::StaticArenaAlias,
            &selection,
        )
        .unwrap();
    let after = runtime.counters();
    let events = observer.take();
    assert_eq!(
        execution.evidence.policy,
        VisionQkvExecutionPolicy::Preferred
    );
    assert_eq!(execution.evidence.outcome, VisionQkvSelectionOutcome::Fused);
    assert_eq!(
        execution.evidence.canonical_layer_plan_blake3,
        selected_plan_identities(&selection)
    );
    assert_operation_site_evidence(
        &runtime,
        before,
        after,
        &execution,
        &events,
        &expected_dispatches(3, true),
        "Preferred/Fused",
    );
    assert_fused_workspace_evidence(&runtime, 3, &execution, &events);
    assert_stack_close(&cpu.stack.output, &execution.output, "Preferred/Fused CPU");
}

#[test]
fn policy_fallback_and_every_failure_are_whole_stack_and_finish_before_gpu_effects() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();
    let graph = SemanticGraph::paddleocr_vl_16();
    let fixture = fixture(3);
    let parameters = runtime_parameters(&fixture.layers);
    let invocation = stack_invocation(&fixture, &parameters);
    let checkpoints = checkpoints(3);
    let cpu = cpu_stack(&fixture, &checkpoints);
    let catalog = tensor_catalog(3);

    let mut unsupported = target(&runtime);
    unsupported.max_storage_buffers_per_shader_stage = 7;
    let before_preferred = runtime.counters();
    let preferred = select_vision_qkv_stack_overlay(VisionQkvExecutionPolicy::Preferred, || {
        build_verified_vision_qkv_stack_overlay(&graph, 3, &layer_plan(), &catalog, unsupported)
    })
    .unwrap();
    assert_eq!(runtime.counters(), before_preferred);
    assert!(observer.take().is_empty());
    assert_eq!(preferred.policy(), VisionQkvExecutionPolicy::Preferred);
    assert_eq!(
        preferred.outcome(),
        VisionQkvSelectionOutcome::FallbackUnsupportedTarget
    );
    assert!(preferred.overlay().is_none());
    assert_eq!(
        preferred.fallback_error_code(),
        Some(VisionQkvStackOverlayErrorCode::UnsupportedTarget)
    );

    let before_fallback = runtime.counters();
    let fallback = runtime
        .run_vision_encoder_stack_identity_rope_with_qkv_selection(
            &invocation,
            &checkpoints,
            VisionStackActivationStrategy::StaticArenaAlias,
            &preferred,
        )
        .unwrap();
    let after_fallback = runtime.counters();
    let fallback_events = observer.take();
    assert_eq!(
        fallback.evidence.policy,
        VisionQkvExecutionPolicy::Preferred
    );
    assert_eq!(
        fallback.evidence.outcome,
        VisionQkvSelectionOutcome::FallbackUnsupportedTarget
    );
    assert!(fallback.evidence.canonical_layer_plan_blake3.is_empty());
    assert!(fallback.evidence.workspace.is_none());
    assert!(fallback.evidence.attention_bindings.is_empty());
    assert!(fallback.evidence.canaries.is_empty());
    assert_operation_site_evidence(
        &runtime,
        before_fallback,
        after_fallback,
        &fallback,
        &fallback_events,
        &expected_dispatches(3, false),
        "Preferred/UnsupportedTarget whole-stack fallback",
    );
    assert_single_submit_and_final_map(&fallback_events);
    assert_stack_close(
        &cpu.stack.output,
        &fallback.output,
        "Preferred fallback/CPU",
    );

    let before_required = runtime.counters();
    let required_error =
        select_vision_qkv_stack_overlay(VisionQkvExecutionPolicy::Required, || {
            build_verified_vision_qkv_stack_overlay(&graph, 3, &layer_plan(), &catalog, unsupported)
        })
        .unwrap_err();
    assert_eq!(
        required_error.code(),
        VisionQkvStackOverlayErrorCode::UnsupportedTarget
    );
    assert_eq!(runtime.counters(), before_required);
    assert!(observer.take().is_empty());

    let mut missing_tensor = catalog.clone();
    missing_tensor.remove(3);
    for policy in [
        VisionQkvExecutionPolicy::Preferred,
        VisionQkvExecutionPolicy::Required,
    ] {
        let before = runtime.counters();
        let error = select_vision_qkv_stack_overlay(policy, || {
            build_verified_vision_qkv_stack_overlay(
                &graph,
                3,
                &layer_plan(),
                &missing_tensor,
                target(&runtime),
            )
        })
        .unwrap_err();
        assert_eq!(
            error.code(),
            VisionQkvStackOverlayErrorCode::SemanticOrTensorIdentity
        );
        assert_eq!(runtime.counters(), before);
        assert!(observer.take().is_empty());
    }

    for policy in [
        VisionQkvExecutionPolicy::Preferred,
        VisionQkvExecutionPolicy::Required,
    ] {
        let before_selections = runtime.counters();
        let wrong_depth = fused_selection_with_policy(&runtime, 1, policy);
        let wrong_geometry = select_vision_qkv_stack_overlay(policy, || {
            build_verified_vision_qkv_stack_overlay(
                &graph,
                3,
                &alternate_token_layer_plan(),
                &catalog,
                target(&runtime),
            )
        })
        .unwrap();
        let mut stale_target = target(&runtime);
        stale_target.min_storage_buffer_offset_alignment = stale_target
            .min_storage_buffer_offset_alignment
            .checked_mul(2)
            .unwrap();
        let stale_target = select_vision_qkv_stack_overlay(policy, || {
            build_verified_vision_qkv_stack_overlay(
                &graph,
                3,
                &layer_plan(),
                &catalog,
                stale_target,
            )
        })
        .unwrap();
        assert_eq!(runtime.counters(), before_selections);
        assert!(observer.take().is_empty());

        for (context, strategy, incompatible_selection) in [
            (
                "overlay depth versus invocation depth",
                VisionStackActivationStrategy::SeparateBuffers,
                wrong_depth,
            ),
            (
                "overlay token geometry versus invocation geometry",
                VisionStackActivationStrategy::SeparateBuffers,
                wrong_geometry,
            ),
            (
                "overlay alignment versus runtime alignment",
                VisionStackActivationStrategy::StaticArenaNoAlias,
                stale_target,
            ),
        ] {
            assert_eq!(incompatible_selection.policy(), policy);
            assert_eq!(
                incompatible_selection.outcome(),
                VisionQkvSelectionOutcome::Fused
            );
            let before = runtime.counters();
            let error = runtime
                .run_vision_encoder_stack_identity_rope_with_qkv_selection(
                    &invocation,
                    &checkpoints,
                    strategy,
                    &incompatible_selection,
                )
                .unwrap_err();
            assert_eq!(
                error.code(),
                RuntimeErrorCode::InvalidInvocation,
                "{policy:?}: {context}"
            );
            assert_eq!(runtime.counters(), before, "{policy:?}: {context}");
            assert!(observer.take().is_empty(), "{policy:?}: {context}");
        }
    }
}

fn assert_invalid_stack_is_rejected_pre_effect(
    runtime: &NativeRuntime,
    observer: &RecordingObserver,
    fixture: &StackFixture,
    checkpoints: &[usize],
    selection: &VisionQkvStackSelection,
    context: &str,
) {
    let parameters = runtime_parameters(&fixture.layers);
    let invocation = stack_invocation(fixture, &parameters);
    let before = runtime.counters();
    let error = runtime
        .run_vision_encoder_stack_identity_rope_with_qkv_selection(
            &invocation,
            checkpoints,
            VisionStackActivationStrategy::SeparateBuffers,
            selection,
        )
        .unwrap_err();
    assert_eq!(
        error.code(),
        RuntimeErrorCode::InvalidInvocation,
        "{context}"
    );
    assert_eq!(runtime.counters(), before, "{context}");
    assert!(observer.take().is_empty(), "{context}");
}

#[test]
fn optimized_entrypoint_validates_malformed_stack_before_effects_for_every_policy() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();

    for policy in [
        VisionQkvExecutionPolicy::Disabled,
        VisionQkvExecutionPolicy::Preferred,
        VisionQkvExecutionPolicy::Required,
    ] {
        let before_selection = runtime.counters();
        let selection = if policy == VisionQkvExecutionPolicy::Disabled {
            disabled_selection()
        } else {
            fused_selection_with_policy(&runtime, 3, policy)
        };
        assert_eq!(runtime.counters(), before_selection);
        assert!(observer.take().is_empty());
        assert_eq!(selection.policy(), policy);
        assert_eq!(
            selection.outcome(),
            if policy == VisionQkvExecutionPolicy::Disabled {
                VisionQkvSelectionOutcome::Disabled
            } else {
                VisionQkvSelectionOutcome::Fused
            }
        );

        let mut short_input = fixture(3);
        short_input.input.pop();
        assert_invalid_stack_is_rejected_pre_effect(
            &runtime,
            &observer,
            &short_input,
            &checkpoints(3),
            &selection,
            &format!("{policy:?}: malformed stack input length"),
        );

        let mut nonfinite_input = fixture(3);
        nonfinite_input.input[0] = f32::NAN;
        assert_invalid_stack_is_rejected_pre_effect(
            &runtime,
            &observer,
            &nonfinite_input,
            &checkpoints(3),
            &selection,
            &format!("{policy:?}: non-finite stack input"),
        );

        let mut short_layer_operand = fixture(3);
        short_layer_operand.layers[1].query_weight.pop();
        assert_invalid_stack_is_rejected_pre_effect(
            &runtime,
            &observer,
            &short_layer_operand,
            &checkpoints(3),
            &selection,
            &format!("{policy:?}: malformed layer operand length"),
        );

        assert_invalid_stack_is_rejected_pre_effect(
            &runtime,
            &observer,
            &fixture(3),
            &[2, 1],
            &selection,
            &format!("{policy:?}: invalid checkpoint ordering"),
        );
    }
}

fn max_abs_difference(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0, f32::max)
}

#[test]
fn existing_legacy_entry_point_stays_legacy_and_matches_disabled_opt_in() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();
    let fixture = fixture(3);
    let parameters = runtime_parameters(&fixture.layers);
    let invocation = stack_invocation(&fixture, &parameters);
    let checkpoints = checkpoints(3);

    let old = runtime
        .run_vision_encoder_stack_identity_rope_with_activation_strategy(
            &invocation,
            &checkpoints,
            VisionStackActivationStrategy::StaticArenaNoAlias,
        )
        .unwrap();
    let old_events = observer.take();
    let disabled = disabled_selection();
    let before_traced = runtime.counters();
    let traced = runtime
        .run_vision_encoder_stack_identity_rope_with_qkv_selection(
            &invocation,
            &checkpoints,
            VisionStackActivationStrategy::StaticArenaNoAlias,
            &disabled,
        )
        .unwrap();
    let after_traced = runtime.counters();
    let traced_events = observer.take();

    assert_eq!(old.output, traced.output);
    assert_eq!(old.checkpoints, traced.checkpoints);
    assert_eq!(old.diagnostics.dispatch_count, 12 * 3 + 1);
    assert_eq!(traced.evidence.encoded_dispatches.len(), 12 * 3 + 1);
    assert_operation_site_evidence(
        &runtime,
        before_traced,
        after_traced,
        &traced,
        &traced_events,
        &expected_dispatches(3, false),
        "Disabled optimized entry point",
    );
    assert_eq!(
        old.diagnostics.activation_strategy,
        traced.diagnostics.activation_strategy
    );
    assert_eq!(
        old.diagnostics.activation_arena_bytes,
        traced.diagnostics.activation_arena_bytes
    );
    assert_eq!(
        old.diagnostics.scratch_allocations,
        traced.diagnostics.scratch_allocations
    );
    assert_eq!(
        activation_allocations(&old_events),
        activation_allocations(&traced_events)
    );
    assert_single_submit_and_final_map(&old_events);
    assert_single_submit_and_final_map(&traced_events);
}

fn qkv_nonce_shader() -> String {
    let source = pvlc_wgsl::module(KernelId::VisionQkvFusedF32)
        .unwrap()
        .source;
    let original = "output.data[output_index] = accumulator;";
    let replacement = "output.data[output_index] = accumulator + 0.125;";
    assert_eq!(source.matches(original).count(), 1);
    source.replacen(original, replacement, 1)
}

fn qkv_padding_corrupt_shader(runtime: &NativeRuntime) -> String {
    let (stride, _, slices) = independent_qkv_layout(runtime);
    assert!(
        stride > slices[0].length,
        "the compact native fixture requires a real internal padding element"
    );
    let source = pvlc_wgsl::module(KernelId::VisionQkvFusedF32)
        .unwrap()
        .source;
    let original = "output.data[output_index] = accumulator;";
    let replacement = r#"output.data[output_index] = accumulator;
    if projection == 0u && token == 0u && output_channel == 0u {
        output.data[params.tokens * params.output_width] = 0.0;
    }"#;
    assert_eq!(source.matches(original).count(), 1);
    source.replacen(original, replacement, 1)
}

#[test]
fn abi_valid_fused_shader_mutant_executes_and_canonical_cache_recovers() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();
    let fixture = fixture(3);
    let parameters = runtime_parameters(&fixture.layers);
    let invocation = stack_invocation(&fixture, &parameters);
    let checkpoints = checkpoints(3);
    let cpu = cpu_stack(&fixture, &checkpoints);
    let selection = fused_selection(&runtime, 3);

    let counters_before_canonical = runtime.counters();
    let canonical_before = runtime
        .run_vision_encoder_stack_identity_rope_with_qkv_selection(
            &invocation,
            &checkpoints,
            VisionStackActivationStrategy::StaticArenaAlias,
            &selection,
        )
        .unwrap();
    let counters_after_canonical = runtime.counters();
    let before_events = observer.take();
    assert_operation_site_evidence(
        &runtime,
        counters_before_canonical,
        counters_after_canonical,
        &canonical_before,
        &before_events,
        &expected_dispatches(3, true),
        "canonical before mutant",
    );
    assert_single_submit_and_final_map(&before_events);
    assert_stack_close(
        &cpu.stack.output,
        &canonical_before.output,
        "canonical-before/CPU",
    );

    let mutant_source = qkv_nonce_shader();
    let overrides = BTreeMap::from([(KernelId::VisionQkvFusedF32, mutant_source.clone())]);
    let before_mutant = runtime.counters();
    let mutant = runtime
        .run_vision_encoder_stack_identity_rope_with_qkv_selection_and_shader_overrides(
            &invocation,
            &checkpoints,
            VisionStackActivationStrategy::StaticArenaAlias,
            &selection,
            &overrides,
        )
        .unwrap();
    let after_mutant = runtime.counters();
    let mutant_events = observer.take();
    assert_operation_site_evidence(
        &runtime,
        before_mutant,
        after_mutant,
        &mutant,
        &mutant_events,
        &expected_dispatches(3, true),
        "ABI-valid semantic mutant",
    );
    assert_single_submit_and_final_map(&mutant_events);
    assert_eq!(mutant.evidence.outcome, VisionQkvSelectionOutcome::Fused);
    assert_eq!(mutant.evidence.encoded_dispatches.len(), 10 * 3 + 1);
    assert!(
        max_abs_difference(&mutant.output, &canonical_before.output) > 1.0e-4,
        "ABI-valid QKV mutant was not causally executed by the stack"
    );
    assert_eq!(
        mutant
            .diagnostics
            .shader_blake3
            .get(&KernelId::VisionQkvFusedF32),
        Some(blake3::hash(mutant_source.as_bytes()).as_bytes())
    );

    let before_recovery = runtime.counters();
    let canonical_after = runtime
        .run_vision_encoder_stack_identity_rope_with_qkv_selection(
            &invocation,
            &checkpoints,
            VisionStackActivationStrategy::StaticArenaAlias,
            &selection,
        )
        .unwrap();
    let after_recovery = runtime.counters();
    let after_events = observer.take();
    assert_operation_site_evidence(
        &runtime,
        before_recovery,
        after_recovery,
        &canonical_after,
        &after_events,
        &expected_dispatches(3, true),
        "canonical after mutant",
    );
    assert_single_submit_and_final_map(&after_events);
    assert_eq!(canonical_after.output, canonical_before.output);
    assert_eq!(canonical_after.checkpoints, canonical_before.checkpoints);
    let canonical_source = pvlc_wgsl::module(KernelId::VisionQkvFusedF32)
        .unwrap()
        .source;
    assert_eq!(
        canonical_after
            .diagnostics
            .shader_blake3
            .get(&KernelId::VisionQkvFusedF32),
        Some(blake3::hash(canonical_source.as_bytes()).as_bytes())
    );
    assert_ne!(
        mutant
            .diagnostics
            .shader_blake3
            .get(&KernelId::VisionQkvFusedF32),
        canonical_after
            .diagnostics
            .shader_blake3
            .get(&KernelId::VisionQkvFusedF32),
        "test-only mutant must not alias or poison the canonical pipeline cache"
    );
}

#[test]
fn padding_corrupting_fused_shader_fails_canary_without_retry_and_canonical_cache_recovers() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();
    let fixture = fixture(3);
    let parameters = runtime_parameters(&fixture.layers);
    let invocation = stack_invocation(&fixture, &parameters);
    let checkpoints = checkpoints(3);
    let selection = fused_selection_with_policy(&runtime, 3, VisionQkvExecutionPolicy::Preferred);
    assert_eq!(selection.policy(), VisionQkvExecutionPolicy::Preferred);
    assert_eq!(selection.outcome(), VisionQkvSelectionOutcome::Fused);

    let before_canonical = runtime.counters();
    let canonical_before = runtime
        .run_vision_encoder_stack_identity_rope_with_qkv_selection(
            &invocation,
            &checkpoints,
            VisionStackActivationStrategy::StaticArenaAlias,
            &selection,
        )
        .unwrap();
    let after_canonical = runtime.counters();
    let canonical_before_events = observer.take();
    assert_operation_site_evidence(
        &runtime,
        before_canonical,
        after_canonical,
        &canonical_before,
        &canonical_before_events,
        &expected_dispatches(3, true),
        "canonical before padding mutant",
    );
    assert_fused_workspace_evidence(&runtime, 3, &canonical_before, &canonical_before_events);

    let corrupt_source = qkv_padding_corrupt_shader(&runtime);
    let overrides = BTreeMap::from([(KernelId::VisionQkvFusedF32, corrupt_source)]);
    let before_corrupt = runtime.counters();
    let error = runtime
        .run_vision_encoder_stack_identity_rope_with_qkv_selection_and_shader_overrides(
            &invocation,
            &checkpoints,
            VisionStackActivationStrategy::StaticArenaAlias,
            &selection,
            &overrides,
        )
        .unwrap_err();
    let after_corrupt = runtime.counters();
    let corrupt_events = observer.take();
    assert_eq!(error.code(), RuntimeErrorCode::Operation);
    let operations = operation_site_trace(&corrupt_events);
    assert_operation_event_order(&corrupt_events);
    assert_raw_copy_temporal_causality(&corrupt_events);
    assert_error_effect_counter_delta(before_corrupt, after_corrupt, &operations);
    let fused_kernels = expected_dispatches(3, true)
        .into_iter()
        .map(|dispatch| dispatch.kernel)
        .collect::<BTreeSet<_>>();
    assert!(
        operations
            .pipelines
            .iter()
            .all(|pipeline| fused_kernels.contains(&pipeline.kernel))
    );
    assert_eq!(
        operations
            .pipelines
            .iter()
            .map(|pipeline| pipeline.kernel)
            .collect::<BTreeSet<_>>()
            .len(),
        operations.pipelines.len()
    );
    assert_eq!(operations.command_encoders.len(), 1);
    assert_eq!(operations.bind_groups.len(), 10 * 3 + 1);
    assert_eq!(operations.dispatches.len(), 10 * 3 + 1);
    assert_eq!(
        operations
            .dispatches
            .iter()
            .map(|dispatch| ExpectedDispatch {
                layer: dispatch.layer,
                stage: dispatch.stage,
                kernel: dispatch.kernel,
                workgroups: dispatch.workgroups,
            })
            .collect::<Vec<_>>(),
        expected_dispatches(3, true)
    );
    assert_eq!(
        operations
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.stage == VisionQkvStackStage::QkvFused)
            .count(),
        3
    );
    assert!(operations.dispatches.iter().all(|dispatch| !matches!(
        dispatch.stage,
        VisionQkvStackStage::Query | VisionQkvStackStage::Key | VisionQkvStackStage::Value
    )));
    assert_eq!(
        corrupt_events
            .iter()
            .filter(|event| matches!(event, RuntimeEvent::SubmissionQueued { .. }))
            .count(),
        1,
        "canary failure must not retry or fall back"
    );
    assert_eq!(
        operations
            .maps
            .iter()
            .filter(|map| map.purpose == VisionQkvMapPurpose::SemanticOutput)
            .count(),
        1
    );
    assert_eq!(
        operations
            .maps
            .iter()
            .filter(|map| map.purpose == VisionQkvMapPurpose::TimestampQuery)
            .count(),
        usize::from(runtime.capabilities().timestamp_query)
    );
    assert_eq!(
        operations.maps.len(),
        1 + usize::from(runtime.capabilities().timestamp_query)
    );

    let (workspace_identity, checked_canaries) = corrupt_events
        .iter()
        .find_map(|event| match event {
            RuntimeEvent::CanaryChecked {
                buffer_identity,
                canaries,
            } => Some((*buffer_identity, canaries.clone())),
            _ => None,
        })
        .expect("corrupt execution must report operation-site canary results");
    assert!(checked_canaries.iter().any(|canary| {
        matches!(canary.kind, VisionQkvCanaryKind::InternalPadding { .. }) && !canary.passed
    }));
    let stable_bits = checked_canaries[0].expected_bits;
    assert_ne!(stable_bits, 0);
    assert!(
        checked_canaries
            .iter()
            .all(|canary| canary.expected_bits == stable_bits)
    );
    let canary_copies = operations
        .copies
        .iter()
        .filter(|copy| copy.purpose == VisionQkvCopyPurpose::CanaryEvidence)
        .collect::<Vec<_>>();
    assert_eq!(canary_copies.len(), checked_canaries.len());
    for (copy, canary) in canary_copies.iter().zip(&checked_canaries) {
        assert_eq!(copy.source_buffer_identity, workspace_identity);
        assert_eq!(copy.source_offset, canary.byte_offset);
        assert_eq!(copy.byte_length, canary.byte_length);
    }
    for pair in canary_copies.windows(2) {
        assert!(pair[0].destination_offset + pair[0].byte_length <= pair[1].destination_offset);
    }
    let semantic_map = operations
        .maps
        .iter()
        .find(|map| map.purpose == VisionQkvMapPurpose::SemanticOutput)
        .unwrap();
    assert!(canary_copies.iter().all(|copy| {
        copy.destination_buffer_identity == semantic_map.buffer_identity
            && copy.destination_offset >= semantic_map.byte_offset
            && copy.destination_offset + copy.byte_length
                <= semantic_map.byte_offset + semantic_map.byte_length
    }));
    assert_eq!(
        after_corrupt.buffer_allocations - before_corrupt.buffer_allocations,
        u64::try_from(
            corrupt_events
                .iter()
                .filter(|event| matches!(event, RuntimeEvent::BufferAllocated { .. }))
                .count()
        )
        .unwrap()
    );

    let before_recovery = runtime.counters();
    let canonical_after = runtime
        .run_vision_encoder_stack_identity_rope_with_qkv_selection(
            &invocation,
            &checkpoints,
            VisionStackActivationStrategy::StaticArenaAlias,
            &selection,
        )
        .unwrap();
    let after_recovery = runtime.counters();
    let recovery_events = observer.take();
    assert_operation_site_evidence(
        &runtime,
        before_recovery,
        after_recovery,
        &canonical_after,
        &recovery_events,
        &expected_dispatches(3, true),
        "canonical after padding mutant",
    );
    assert_eq!(canonical_after.output, canonical_before.output);
    assert_eq!(canonical_after.checkpoints, canonical_before.checkpoints);
    assert!(
        canonical_after
            .evidence
            .canaries
            .iter()
            .all(|canary| canary.passed)
    );
}

#[test]
fn paired_legacy_and_fused_depths_and_activation_strategies_match_cpu_and_actual_encoding_trace() {
    let observer = Arc::new(RecordingObserver::default());
    let Some(runtime) = runtime(Arc::clone(&observer)) else {
        return;
    };
    observer.take();

    for depth in [1_usize, 3, 16] {
        let fixture = fixture(depth);
        let parameters = runtime_parameters(&fixture.layers);
        let invocation = stack_invocation(&fixture, &parameters);
        let checkpoints = checkpoints(depth);
        let cpu = cpu_stack(&fixture, &checkpoints);
        let fused_selection = fused_selection(&runtime, depth);
        let fused_plan_identities = selected_plan_identities(&fused_selection);

        for strategy in [
            VisionStackActivationStrategy::SeparateBuffers,
            VisionStackActivationStrategy::StaticArenaNoAlias,
            VisionStackActivationStrategy::StaticArenaAlias,
        ] {
            let disabled = disabled_selection();
            let before_legacy = runtime.counters();
            let legacy = runtime
                .run_vision_encoder_stack_identity_rope_with_qkv_selection(
                    &invocation,
                    &checkpoints,
                    strategy,
                    &disabled,
                )
                .unwrap();
            let after_legacy = runtime.counters();
            let legacy_events = observer.take();

            let before_fused = runtime.counters();
            let fused = runtime
                .run_vision_encoder_stack_identity_rope_with_qkv_selection(
                    &invocation,
                    &checkpoints,
                    strategy,
                    &fused_selection,
                )
                .unwrap();
            let after_fused = runtime.counters();
            let fused_events = observer.take();

            assert_eq!(disabled.policy(), VisionQkvExecutionPolicy::Disabled);
            assert_eq!(legacy.evidence.outcome, VisionQkvSelectionOutcome::Disabled);
            assert_eq!(fused.evidence.outcome, VisionQkvSelectionOutcome::Fused);
            assert_eq!(legacy.evidence.policy, VisionQkvExecutionPolicy::Disabled);
            assert_eq!(fused.evidence.policy, VisionQkvExecutionPolicy::Required);
            assert!(legacy.evidence.canonical_layer_plan_blake3.is_empty());
            assert_eq!(
                fused.evidence.canonical_layer_plan_blake3, fused_plan_identities,
                "returned identities must exactly preserve selected overlay order and values"
            );
            assert_eq!(legacy.evidence.encoded_dispatches.len(), 12 * depth + 1);
            assert_eq!(fused.evidence.encoded_dispatches.len(), 10 * depth + 1);
            assert_eq!(legacy.evidence.compute_pass_count, depth + 1);
            assert_eq!(fused.evidence.compute_pass_count, depth + 1);

            assert_operation_site_evidence(
                &runtime,
                before_legacy,
                after_legacy,
                &legacy,
                &legacy_events,
                &expected_dispatches(depth, false),
                &format!("legacy depth={depth} strategy={strategy:?}"),
            );
            assert_operation_site_evidence(
                &runtime,
                before_fused,
                after_fused,
                &fused,
                &fused_events,
                &expected_dispatches(depth, true),
                &format!("fused depth={depth} strategy={strategy:?}"),
            );

            for layer in &checkpoints {
                let expected = cpu.stack.checkpoint(*layer).unwrap();
                assert_stack_close(
                    expected,
                    legacy.checkpoints.get(layer).unwrap(),
                    "CPU/legacy checkpoint",
                );
                assert_stack_close(
                    expected,
                    fused.checkpoints.get(layer).unwrap(),
                    "CPU/fused checkpoint",
                );
                assert_stack_close(
                    legacy.checkpoints.get(layer).unwrap(),
                    fused.checkpoints.get(layer).unwrap(),
                    "legacy/fused checkpoint",
                );
            }
            assert_stack_close(&cpu.stack.output, &legacy.output, "CPU/legacy output");
            assert_stack_close(&cpu.stack.output, &fused.output, "CPU/fused output");
            assert_stack_close(&legacy.output, &fused.output, "legacy/fused output");

            assert_eq!(legacy.diagnostics.activation_strategy, strategy);
            assert_eq!(fused.diagnostics.activation_strategy, strategy);
            assert_eq!(
                legacy.diagnostics.activation_buffer_count,
                fused.diagnostics.activation_buffer_count
            );
            assert_eq!(
                legacy.diagnostics.activation_arena_bytes,
                fused.diagnostics.activation_arena_bytes
            );
            assert_eq!(
                legacy.diagnostics.scratch_arena_bytes,
                fused.diagnostics.scratch_arena_bytes
            );
            assert_eq!(
                legacy.diagnostics.main_buffers_bytes,
                fused.diagnostics.main_buffers_bytes
            );
            assert_eq!(
                legacy.diagnostics.weight_buffer_count,
                fused.diagnostics.weight_buffer_count
            );
            assert_eq!(legacy.diagnostics.readback_buffer_count, 1);
            assert_eq!(fused.diagnostics.readback_buffer_count, 1);
            assert_eq!(
                legacy.diagnostics.scratch_allocations,
                fused.diagnostics.scratch_allocations
            );
            assert_eq!(
                activation_allocations(&legacy_events),
                activation_allocations(&fused_events)
            );
            assert!(legacy.evidence.workspace.is_none());
            assert!(legacy.evidence.attention_bindings.is_empty());
            assert!(legacy.evidence.canaries.is_empty());
            assert_fused_workspace_evidence(&runtime, depth, &fused, &fused_events);

            if strategy != VisionStackActivationStrategy::SeparateBuffers {
                let layout = invocation
                    .plan(&checkpoints)
                    .unwrap()
                    .activation_layout(layout_config(&runtime, strategy))
                    .unwrap();
                assert_eq!(
                    fused.diagnostics.activation_arena_bytes,
                    layout.total_activation_bytes
                );
                assert_eq!(
                    fused.diagnostics.scratch_allocations,
                    layout.scratch_allocations
                );
            }

            assert_eq!(after_legacy.submissions - before_legacy.submissions, 1);
            assert_eq!(after_fused.submissions - before_fused.submissions, 1);
            assert_eq!(
                after_legacy.buffer_allocations - before_legacy.buffer_allocations,
                legacy.diagnostics.buffer_allocation_count
            );
            assert_eq!(
                after_fused.buffer_allocations - before_fused.buffer_allocations,
                fused.diagnostics.buffer_allocation_count
            );
            assert_eq!(
                fused.diagnostics.buffer_allocation_count,
                legacy.diagnostics.buffer_allocation_count + 1,
                "the fused workspace is the only additional physical allocation"
            );
            assert_eq!(observer_topology(&legacy_events).0, [1]);
            assert_eq!(observer_topology(&fused_events).0, [1]);
            assert_eq!(observer_topology(&legacy_events).1.len(), 1);
            assert_eq!(observer_topology(&fused_events).1.len(), 1);
            assert_single_submit_and_final_map(&legacy_events);
            assert_single_submit_and_final_map(&fused_events);
        }
    }
}

fn source_braced_item<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("native source is missing {signature:?}"));
    let opening = start
        + source[start..]
            .find('{')
            .unwrap_or_else(|| panic!("native source item {signature:?} has no body"));
    let mut depth = 0_usize;
    for (offset, byte) in source.as_bytes()[opening..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth = depth
                    .checked_sub(1)
                    .unwrap_or_else(|| panic!("native source item {signature:?} has bad braces"));
                if depth == 0 {
                    return &source[start..=opening + offset];
                }
            }
            _ => {}
        }
    }
    panic!("native source item {signature:?} has an unterminated body");
}

fn without_ascii_whitespace(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect()
}

fn bound_call_result_name(function_body: &str, call: &str) -> String {
    let call_offset = function_body
        .find(call)
        .unwrap_or_else(|| panic!("source function must call {call}"));
    let prefix = &function_body[..call_offset];
    let let_offset = prefix
        .rfind("let ")
        .unwrap_or_else(|| panic!("result of {call} must be bound before use"));
    let binding_and_equals = &prefix[let_offset + "let ".len()..];
    let equals = binding_and_equals
        .find('=')
        .unwrap_or_else(|| panic!("result binding for {call} has no equals sign"));
    let binding = binding_and_equals[..equals]
        .trim()
        .strip_prefix("mut ")
        .unwrap_or_else(|| binding_and_equals[..equals].trim());
    assert!(
        binding != "_"
            && !binding.is_empty()
            && binding
                .chars()
                .all(|character| character == '_' || character.is_ascii_alphanumeric()),
        "result of {call} must be retained in a named local, found {binding:?}"
    );
    binding.to_owned()
}

#[test]
fn native_optimized_executor_does_not_replan_verified_qkv_geometry_and_preflights_before_effects() {
    let source = include_str!("../src/lib.rs");
    assert!(
        !source.contains("plan_vision_qkv_fused_geometry"),
        "native optimized execution must consume the opaque verified Q/K/V descriptor and must not import, call, or otherwise reference plan_vision_qkv_fused_geometry"
    );

    let prepared_state = source_braced_item(source, "struct PreparedVisionQkvExecution {");
    for field in [
        "total_readback_bytes",
        "readback_f32_elements",
        "workspace_u32_words",
    ] {
        assert!(
            prepared_state.contains(field),
            "prepared optimized execution state must retain checked preflight field {field}"
        );
    }

    let prepare_body = source_braced_item(source, "fn prepare_vision_qkv_execution(");
    assert!(
        prepare_body.contains("preflight_vision_qkv_execution_allocations("),
        "prepare_vision_qkv_execution must invoke the pure allocation/readback preflight before any optimized shader validation or execution"
    );
    let preflight_binding =
        bound_call_result_name(prepare_body, "preflight_vision_qkv_execution_allocations(");
    let compact_prepare = without_ascii_whitespace(prepare_body);
    for field in [
        "total_readback_bytes",
        "readback_f32_elements",
        "workspace_u32_words",
    ] {
        let causal_reference = format!("{preflight_binding}.{field}");
        assert!(
            compact_prepare.contains(&causal_reference),
            "prepare_vision_qkv_execution must copy checked helper output {causal_reference} into prepared execution state"
        );
    }

    let entrypoint_body = source_braced_item(
        source,
        "pub fn run_vision_encoder_stack_identity_rope_with_qkv_selection_and_shader_overrides(",
    );
    let prepare_call = entrypoint_body
        .find("self.prepare_vision_qkv_execution(")
        .expect("optimized entry point must call preparation");
    let sources_call = entrypoint_body
        .find("self.validated_vision_qkv_stack_sources(")
        .expect("optimized entry point must validate shader sources");
    let execute_call = entrypoint_body
        .find("self.execute_vision_stack_once_optimized(")
        .expect("optimized entry point must call the one-shot executor");
    assert!(
        prepare_call < sources_call && sources_call < execute_call,
        "allocation/readback preflight must finish before optimized shader validation and the one-shot GPU executor"
    );

    let optimized_wrapper = source_braced_item(source, "fn execute_vision_stack_once_optimized(");
    assert!(
        optimized_wrapper.contains("self.execute_vision_stack_once_common("),
        "optimized executor wrapper must delegate the prepared state to the common one-shot executor"
    );
    let common_executor = source_braced_item(source, "fn execute_vision_stack_once_common(");
    for field_reference in [
        "prepared.total_readback_bytes",
        "prepared.readback_f32_elements",
        "prepared.workspace_u32_words",
    ] {
        assert!(
            common_executor.contains(field_reference),
            "common executor must causally consume checked preflight value {field_reference}"
        );
    }

    let compact_common = without_ascii_whitespace(common_executor);
    for (forbidden, reason) in [
        (
            "usize::try_from(workspace.allocation_bytes/4).unwrap_or(0)",
            "workspace initialization must use prepared.workspace_u32_words instead of a lossy late conversion",
        ),
        (
            "letcanary_readback_bytes=",
            "common execution must not locally re-sum the canary tail after preflight",
        ),
        (
            ".canaries.iter().map(|canary|canary.byte_length).sum()",
            "common execution must not recompute canary readback bytes",
        ),
        (
            "plan.readback_bytes.checked_add(canary_readback_bytes)",
            "readback allocation must use prepared.total_readback_bytes instead of a late checked_add",
        ),
        (
            "usize::try_from(total_readback_bytes/4)",
            "readback decoding must use prepared.readback_f32_elements instead of a late host conversion",
        ),
    ] {
        assert!(
            !compact_common.contains(forbidden),
            "{reason}; forbidden source pattern: {forbidden}"
        );
    }
}
