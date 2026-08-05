use pvlc_runtime_core::{
    InvocationErrorCode, KernelId, VisionEncoderLayerDispatch, VisionEncoderLayerInvocation,
    VisionEncoderLayerGeometry, VisionEncoderLayerParameters, VisionEncoderLayerPlan,
    VisionEncoderLayerStage, VisionEncoderStackInvocation, VisionEncoderStackPlan,
    VisionLayerNormParameters, VisionLinearParameters, VisionRopeSpecialization,
    VisionStackActivationLayout, VisionStackActivationLayoutConfig, VisionStackScratchAllocation,
};

const TOKENS: u32 = 9;
const HIDDEN: u32 = 18;
const HEADS: u32 = 3;
const HEAD_DIM: u32 = 6;
const INTERMEDIATE: u32 = 23;
const EPSILON: f32 = 1.0e-6;

#[derive(Clone)]
struct Fixture {
    tokens: u32,
    hidden: u32,
    heads: u32,
    head_dim: u32,
    intermediate: u32,
    epsilon: f32,
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
            tokens: TOKENS,
            hidden: HIDDEN,
            heads: HEADS,
            head_dim: HEAD_DIM,
            intermediate: INTERMEDIATE,
            epsilon: EPSILON,
            boundaries: vec![0, 3, TOKENS],
            input: values((TOKENS * HIDDEN) as usize, 1),
            norm1_weight: values(HIDDEN as usize, 2),
            norm1_bias: values(HIDDEN as usize, 3),
            query_weight: values((HIDDEN * HIDDEN) as usize, 4),
            query_bias: values(HIDDEN as usize, 5),
            key_weight: values((HIDDEN * HIDDEN) as usize, 6),
            key_bias: values(HIDDEN as usize, 7),
            value_weight: values((HIDDEN * HIDDEN) as usize, 8),
            value_bias: values(HIDDEN as usize, 9),
            attention_output_weight: values((HIDDEN * HIDDEN) as usize, 10),
            attention_output_bias: values(HIDDEN as usize, 11),
            norm2_weight: values(HIDDEN as usize, 12),
            norm2_bias: values(HIDDEN as usize, 13),
            mlp_fc1_weight: values((INTERMEDIATE * HIDDEN) as usize, 14),
            mlp_fc1_bias: values(INTERMEDIATE as usize, 15),
            mlp_fc2_weight: values((HIDDEN * INTERMEDIATE) as usize, 16),
            mlp_fc2_bias: values(HIDDEN as usize, 17),
        }
    }

    fn invocation(&self) -> VisionEncoderLayerInvocation<'_> {
        VisionEncoderLayerInvocation {
            tokens: self.tokens,
            hidden_size: self.hidden,
            attention_heads: self.heads,
            head_dim: self.head_dim,
            intermediate_size: self.intermediate,
            layer_norm_epsilon: self.epsilon,
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

    fn floating_operand_mut(&mut self, index: usize) -> &mut Vec<f32> {
        match index {
            0 => &mut self.input,
            1 => &mut self.norm1_weight,
            2 => &mut self.norm1_bias,
            3 => &mut self.query_weight,
            4 => &mut self.query_bias,
            5 => &mut self.key_weight,
            6 => &mut self.key_bias,
            7 => &mut self.value_weight,
            8 => &mut self.value_bias,
            9 => &mut self.attention_output_weight,
            10 => &mut self.attention_output_bias,
            11 => &mut self.norm2_weight,
            12 => &mut self.norm2_bias,
            13 => &mut self.mlp_fc1_weight,
            14 => &mut self.mlp_fc1_bias,
            15 => &mut self.mlp_fc2_weight,
            16 => &mut self.mlp_fc2_bias,
            _ => panic!("fixture has exactly seventeen floating operands"),
        }
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

fn assert_error(fixture: &Fixture, expected: InvocationErrorCode) {
    assert_eq!(fixture.invocation().plan().unwrap_err().code(), expected);
}

fn assert_dispatch(
    actual: VisionEncoderLayerDispatch,
    stage: VisionEncoderLayerStage,
    kernel: KernelId,
    output_elements: usize,
    dispatch: [u32; 3],
    uniform_words: [u32; 4],
) {
    assert_eq!(actual.stage, stage);
    assert_eq!(actual.invocation.kernel, kernel);
    assert_eq!(actual.invocation.output_elements, output_elements);
    assert_eq!(actual.invocation.output_bytes, (output_elements * 4) as u64);
    assert_eq!(actual.invocation.dispatch, dispatch);
    assert_eq!(actual.uniform_words, uniform_words);
}

#[test]
fn identity_rope_plan_is_exactly_twelve_resident_dispatches_with_no_rope_kernel() {
    let fixture = Fixture::new();
    let plan = fixture.invocation().plan().unwrap();
    let hidden_elements = (TOKENS * HIDDEN) as usize;
    let intermediate_elements = (TOKENS * INTERMEDIATE) as usize;

    assert_eq!(plan.rope_specialization, VisionRopeSpecialization::Identity);
    assert_eq!(plan.dispatches.len(), 12);
    assert_eq!(VisionEncoderLayerStage::ALL.len(), 12);
    assert_eq!(
        plan.dispatches.map(|dispatch| dispatch.stage),
        VisionEncoderLayerStage::ALL
    );
    assert!(
        plan.dispatches
            .iter()
            .all(|dispatch| dispatch.invocation.kernel != KernelId::RopeNeoxF32),
        "the pinned remote-code zero-frequency RoPE must compile away"
    );

    let linear_hidden = [1, 1, 1];
    let linear_intermediate = [1, 1, 1];
    let norm_uniform = [TOKENS, HIDDEN, EPSILON.to_bits(), 0];
    assert_dispatch(
        plan.dispatches[0],
        VisionEncoderLayerStage::Norm1,
        KernelId::LayerNormF32,
        hidden_elements,
        [1, 1, 1],
        norm_uniform,
    );
    for (index, stage) in [
        VisionEncoderLayerStage::Query,
        VisionEncoderLayerStage::Key,
        VisionEncoderLayerStage::Value,
    ]
    .into_iter()
    .enumerate()
    {
        assert_dispatch(
            plan.dispatches[index + 1],
            stage,
            KernelId::VisionPatchProjectionF32,
            hidden_elements,
            linear_hidden,
            [TOKENS, HIDDEN, HIDDEN, 0],
        );
    }
    assert_dispatch(
        plan.dispatches[4],
        VisionEncoderLayerStage::AttentionContext,
        KernelId::VisionAttentionF32,
        hidden_elements,
        [TOKENS.div_ceil(128), HEADS, 1],
        [TOKENS, HEADS, HEAD_DIM, 2],
    );
    assert_dispatch(
        plan.dispatches[5],
        VisionEncoderLayerStage::AttentionOutput,
        KernelId::VisionPatchProjectionF32,
        hidden_elements,
        linear_hidden,
        [TOKENS, HIDDEN, HIDDEN, 0],
    );
    assert_dispatch(
        plan.dispatches[6],
        VisionEncoderLayerStage::AttentionResidual,
        KernelId::AddF32,
        hidden_elements,
        [3, 1, 1],
        [hidden_elements as u32, 0, 0, 0],
    );
    assert_dispatch(
        plan.dispatches[7],
        VisionEncoderLayerStage::Norm2,
        KernelId::LayerNormF32,
        hidden_elements,
        [1, 1, 1],
        norm_uniform,
    );
    assert_dispatch(
        plan.dispatches[8],
        VisionEncoderLayerStage::MlpFc1,
        KernelId::VisionPatchProjectionF32,
        intermediate_elements,
        linear_intermediate,
        [TOKENS, HIDDEN, INTERMEDIATE, 0],
    );
    assert_dispatch(
        plan.dispatches[9],
        VisionEncoderLayerStage::MlpActivation,
        KernelId::GeluTanhF32,
        intermediate_elements,
        [4, 1, 1],
        [intermediate_elements as u32, 0, 0, 0],
    );
    assert_dispatch(
        plan.dispatches[10],
        VisionEncoderLayerStage::MlpOutput,
        KernelId::VisionPatchProjectionF32,
        hidden_elements,
        linear_hidden,
        [TOKENS, INTERMEDIATE, HIDDEN, 0],
    );
    assert_dispatch(
        plan.dispatches[11],
        VisionEncoderLayerStage::Output,
        KernelId::AddF32,
        hidden_elements,
        [3, 1, 1],
        [hidden_elements as u32, 0, 0, 0],
    );

    assert_eq!(
        plan.resident_intermediate_bytes,
        ((10 * hidden_elements + 2 * intermediate_elements) * 4) as u64,
        "semantic-debug residency is O(tokens * (hidden + intermediate)), never O(tokens^2)"
    );
}

#[test]
fn layer_plan_tiles_a_large_mlp_activation_without_oversized_webgpu_dispatches() {
    let mut fixture = Fixture::new();
    fixture.tokens = 233;
    fixture.hidden = 1;
    fixture.heads = 1;
    fixture.head_dim = 1;
    fixture.intermediate = 18_002;
    fixture.boundaries = vec![0, fixture.tokens];
    fixture.input = vec![0.25; fixture.tokens as usize];
    fixture.norm1_weight = vec![1.0];
    fixture.norm1_bias = vec![0.0];
    fixture.query_weight = vec![0.5];
    fixture.query_bias = vec![0.0];
    fixture.key_weight = vec![0.25];
    fixture.key_bias = vec![0.0];
    fixture.value_weight = vec![0.75];
    fixture.value_bias = vec![0.0];
    fixture.attention_output_weight = vec![1.0];
    fixture.attention_output_bias = vec![0.0];
    fixture.norm2_weight = vec![1.0];
    fixture.norm2_bias = vec![0.0];
    fixture.mlp_fc1_weight = vec![0.01; fixture.intermediate as usize];
    fixture.mlp_fc1_bias = vec![0.0; fixture.intermediate as usize];
    fixture.mlp_fc2_weight = vec![0.01; fixture.intermediate as usize];
    fixture.mlp_fc2_bias = vec![0.0];

    let plan = fixture.invocation().plan().unwrap();
    let activation_elements = fixture.tokens * fixture.intermediate;
    assert_eq!(activation_elements, 4_194_466);
    assert_eq!(
        plan.dispatches[9].stage,
        VisionEncoderLayerStage::MlpActivation
    );
    assert_eq!(plan.dispatches[9].invocation.dispatch, [32_770, 2, 1]);
    assert_eq!(
        plan.dispatches[9].uniform_words,
        [activation_elements, 32_770 * 64, 0, 0]
    );
    assert!(
        plan.dispatches
            .iter()
            .flat_map(|dispatch| dispatch.invocation.dispatch)
            .all(|dimension| dimension <= 65_535)
    );
}

#[test]
fn portrait_ocr_geometry_tiles_both_residual_adds_within_webgpu_limits() {
    // The supplied 960×1280 receipt is smart-resized to grid [1,82,60]:
    // 4,920 patches. This is a normal admitted OCR input, and its
    // 4,920×1,152 residual planes must not be rejected by the browser's
    // 65,535-workgroup-per-axis limit.
    let tokens = 4_920;
    let hidden = 1_152;
    let hidden_elements = tokens * hidden;
    let expected_dispatch = [44_280, 2, 1];
    let expected_row_stride = 44_280 * 64;
    let plan = VisionEncoderLayerGeometry {
        tokens,
        hidden_size: hidden,
        attention_heads: 16,
        head_dim: 72,
        intermediate_size: 4_304,
        layer_norm_epsilon: 1.0e-6,
        cu_seqlens: &[0, tokens],
    }
    .plan()
    .unwrap();

    for stage in [
        VisionEncoderLayerStage::AttentionResidual,
        VisionEncoderLayerStage::Output,
    ] {
        let dispatch = plan
            .dispatches
            .iter()
            .find(|dispatch| dispatch.stage == stage)
            .unwrap();
        assert_eq!(dispatch.invocation.kernel, KernelId::AddF32);
        assert_eq!(dispatch.invocation.dispatch, expected_dispatch);
        assert_eq!(
            dispatch.uniform_words,
            [hidden_elements, expected_row_stride, 0, 0]
        );
    }
    assert!(
        plan.dispatches
            .iter()
            .flat_map(|dispatch| dispatch.invocation.dispatch)
            .all(|dimension| dimension <= 65_535),
        "every stage of the admitted portrait OCR geometry must fit WebGPU"
    );
}

#[test]
fn layer_plan_rejects_geometry_epsilon_and_boundaries_before_execution() {
    for mutate in [
        |fixture: &mut Fixture| fixture.tokens = 0,
        |fixture: &mut Fixture| fixture.hidden = 0,
        |fixture: &mut Fixture| fixture.heads = 0,
        |fixture: &mut Fixture| fixture.head_dim = 0,
        |fixture: &mut Fixture| fixture.intermediate = 0,
    ] {
        let mut fixture = Fixture::new();
        mutate(&mut fixture);
        assert_error(&fixture, InvocationErrorCode::ZeroDimension);
    }

    let mut fixture = Fixture::new();
    fixture.heads = 4;
    fixture.head_dim = 4;
    assert_error(&fixture, InvocationErrorCode::InvalidVisionGeometry);

    let mut fixture = Fixture::new();
    fixture.hidden = 73;
    fixture.heads = 1;
    fixture.head_dim = 73;
    assert_error(&fixture, InvocationErrorCode::UnsupportedHeadDimension);

    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut fixture = Fixture::new();
        fixture.epsilon = epsilon;
        assert_error(&fixture, InvocationErrorCode::InvalidEpsilon);
    }

    for boundaries in [
        vec![],
        vec![0],
        vec![1, TOKENS],
        vec![0, 3, 3, TOKENS],
        vec![0, TOKENS - 1],
        vec![0, TOKENS, TOKENS + 1],
    ] {
        let mut fixture = Fixture::new();
        fixture.boundaries = boundaries;
        assert_error(&fixture, InvocationErrorCode::InvalidSequenceBoundaries);
    }

    let mut fixture = Fixture::new();
    fixture.tokens = u32::MAX;
    fixture.hidden = u32::MAX;
    fixture.heads = u32::MAX;
    fixture.head_dim = 1;
    fixture.intermediate = 1;
    assert_error(&fixture, InvocationErrorCode::ArithmeticOverflow);
}

#[test]
fn layer_plan_validates_every_floating_operand_for_length_and_finiteness() {
    for operand in 0..17 {
        for oversized in [false, true] {
            let mut fixture = Fixture::new();
            if oversized {
                fixture.floating_operand_mut(operand).push(0.0);
            } else {
                fixture.floating_operand_mut(operand).pop();
            }
            assert_error(&fixture, InvocationErrorCode::LengthMismatch);
        }

        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let length = Fixture::new().floating_operand_mut(operand).len();
            for position in [0, length / 2, length - 1] {
                let mut fixture = Fixture::new();
                fixture.floating_operand_mut(operand)[position] = nonfinite;
                assert_error(&fixture, InvocationErrorCode::NonFiniteInput);
            }
        }
    }
}

fn stack_parameters(fixtures: &[Fixture]) -> Vec<VisionEncoderLayerParameters<'_>> {
    fixtures
        .iter()
        .map(|fixture| fixture.invocation().parameters)
        .collect()
}

fn stack_invocation<'a>(
    fixture: &'a Fixture,
    layer_parameters: &'a [VisionEncoderLayerParameters<'a>],
    post_weight: &'a [f32],
    post_bias: &'a [f32],
) -> VisionEncoderStackInvocation<'a> {
    VisionEncoderStackInvocation {
        tokens: fixture.tokens,
        hidden_size: fixture.hidden,
        attention_heads: fixture.heads,
        head_dim: fixture.head_dim,
        intermediate_size: fixture.intermediate,
        layer_norm_epsilon: fixture.epsilon,
        input: &fixture.input,
        cu_seqlens: &fixture.boundaries,
        layer_parameters,
        post_norm: VisionLayerNormParameters {
            weight: post_weight,
            bias: post_bias,
        },
    }
}

fn layer_parameter_elements(fixture: &Fixture) -> usize {
    [
        fixture.norm1_weight.len(),
        fixture.norm1_bias.len(),
        fixture.query_weight.len(),
        fixture.query_bias.len(),
        fixture.key_weight.len(),
        fixture.key_bias.len(),
        fixture.value_weight.len(),
        fixture.value_bias.len(),
        fixture.attention_output_weight.len(),
        fixture.attention_output_bias.len(),
        fixture.norm2_weight.len(),
        fixture.norm2_bias.len(),
        fixture.mlp_fc1_weight.len(),
        fixture.mlp_fc1_bias.len(),
        fixture.mlp_fc2_weight.len(),
        fixture.mlp_fc2_bias.len(),
    ]
    .into_iter()
    .sum()
}

fn compact_hidden_bytes() -> u64 {
    u64::from(TOKENS) * u64::from(HIDDEN) * 4
}

fn compact_intermediate_bytes() -> u64 {
    u64::from(TOKENS) * u64::from(INTERMEDIATE) * 4
}

fn compact_main_buffer_bytes() -> u64 {
    2 * compact_hidden_bytes()
}

fn compact_post_norm() -> (Vec<f32>, Vec<f32>) {
    (values(HIDDEN as usize, 91), values(HIDDEN as usize, 92))
}

fn compact_stack_plan(layer_count: usize) -> VisionEncoderStackPlan {
    let fixtures = vec![Fixture::new(); layer_count];
    let layer_parameters = stack_parameters(&fixtures);
    let (post_weight, post_bias) = compact_post_norm();
    stack_invocation(&fixtures[0], &layer_parameters, &post_weight, &post_bias)
        .plan(&[])
        .unwrap()
}

fn compact_layer_plan() -> VisionEncoderLayerPlan {
    Fixture::new().invocation().plan().unwrap()
}

fn layout_config(
    allow_aliasing: bool,
    storage_buffer_offset_alignment: u64,
    arena_alignment: u64,
) -> VisionStackActivationLayoutConfig {
    VisionStackActivationLayoutConfig {
        allow_aliasing,
        storage_buffer_offset_alignment,
        arena_alignment,
    }
}

fn expected_scratch_allocation(
    stage: VisionEncoderLayerStage,
    offset: u64,
    size: u64,
    alignment: u64,
    first_write: u32,
    last_use: u32,
) -> VisionStackScratchAllocation {
    VisionStackScratchAllocation {
        stage,
        offset,
        size,
        alignment,
        first_write,
        last_use,
    }
}

fn expected_no_alias_scratch_allocations() -> Vec<VisionStackScratchAllocation> {
    let hidden = compact_hidden_bytes();
    let intermediate = compact_intermediate_bytes();
    vec![
        expected_scratch_allocation(VisionEncoderLayerStage::Norm1, 0, hidden, 4, 0, 3),
        expected_scratch_allocation(VisionEncoderLayerStage::Query, hidden, hidden, 4, 1, 4),
        expected_scratch_allocation(VisionEncoderLayerStage::Key, 2 * hidden, hidden, 4, 2, 4),
        expected_scratch_allocation(VisionEncoderLayerStage::Value, 3 * hidden, hidden, 4, 3, 4),
        expected_scratch_allocation(
            VisionEncoderLayerStage::AttentionContext,
            4 * hidden,
            hidden,
            4,
            4,
            5,
        ),
        expected_scratch_allocation(
            VisionEncoderLayerStage::AttentionOutput,
            5 * hidden,
            hidden,
            4,
            5,
            6,
        ),
        expected_scratch_allocation(
            VisionEncoderLayerStage::AttentionResidual,
            6 * hidden,
            hidden,
            4,
            6,
            11,
        ),
        expected_scratch_allocation(VisionEncoderLayerStage::Norm2, 7 * hidden, hidden, 4, 7, 8),
        expected_scratch_allocation(
            VisionEncoderLayerStage::MlpFc1,
            8 * hidden,
            intermediate,
            4,
            8,
            9,
        ),
        expected_scratch_allocation(
            VisionEncoderLayerStage::MlpActivation,
            8 * hidden + intermediate,
            intermediate,
            4,
            9,
            10,
        ),
        expected_scratch_allocation(
            VisionEncoderLayerStage::MlpOutput,
            8 * hidden + 2 * intermediate,
            hidden,
            4,
            10,
            11,
        ),
    ]
}

fn expected_alias_scratch_allocations() -> Vec<VisionStackScratchAllocation> {
    let hidden = compact_hidden_bytes();
    let intermediate = compact_intermediate_bytes();
    vec![
        expected_scratch_allocation(VisionEncoderLayerStage::Norm1, 0, hidden, 4, 0, 3),
        expected_scratch_allocation(VisionEncoderLayerStage::Query, hidden, hidden, 4, 1, 4),
        expected_scratch_allocation(VisionEncoderLayerStage::Key, 2 * hidden, hidden, 4, 2, 4),
        expected_scratch_allocation(VisionEncoderLayerStage::Value, 3 * hidden, hidden, 4, 3, 4),
        expected_scratch_allocation(
            VisionEncoderLayerStage::AttentionContext,
            0,
            hidden,
            4,
            4,
            5,
        ),
        expected_scratch_allocation(
            VisionEncoderLayerStage::AttentionOutput,
            hidden,
            hidden,
            4,
            5,
            6,
        ),
        expected_scratch_allocation(
            VisionEncoderLayerStage::AttentionResidual,
            1_656,
            hidden,
            4,
            6,
            11,
        ),
        expected_scratch_allocation(VisionEncoderLayerStage::Norm2, 828, hidden, 4, 7, 8),
        expected_scratch_allocation(VisionEncoderLayerStage::MlpFc1, 0, intermediate, 4, 8, 9),
        expected_scratch_allocation(
            VisionEncoderLayerStage::MlpActivation,
            828,
            intermediate,
            4,
            9,
            10,
        ),
        expected_scratch_allocation(VisionEncoderLayerStage::MlpOutput, 0, hidden, 4, 10, 11),
    ]
}

fn live_at(allocation: &VisionStackScratchAllocation, point: u32) -> bool {
    allocation.first_write <= point && point <= allocation.last_use
}

fn end_offset(allocation: &VisionStackScratchAllocation) -> u64 {
    allocation.offset + allocation.size
}

fn assert_layout_matches_expected(
    layout: &VisionStackActivationLayout,
    expected_allocations: &[VisionStackScratchAllocation],
    expected_scratch_arena_bytes: u64,
    expected_total_activation_bytes: u64,
) {
    // These exact offsets are the accepted shared pvlc-memory planner result
    // for this schedule, not a handwritten local packing preference.
    assert_eq!(layout.scratch_allocations, expected_allocations);
    assert_eq!(layout.scratch_arena_bytes, expected_scratch_arena_bytes);
    assert_eq!(layout.main_buffers_bytes, compact_main_buffer_bytes());
    assert_eq!(
        layout.total_activation_bytes,
        expected_total_activation_bytes
    );
    assert_eq!(layout.physical_buffer_count, 3);
    assert_eq!(
        layout
            .scratch_allocations
            .iter()
            .map(|allocation| allocation.stage.as_str())
            .collect::<Vec<_>>(),
        vec![
            "norm1",
            "query",
            "key",
            "value",
            "attention-context",
            "attention-output",
            "attention-residual",
            "norm2",
            "mlp-fc1",
            "mlp-activation",
            "mlp-output",
        ]
    );

    for allocation in &layout.scratch_allocations {
        assert_eq!(allocation.offset % allocation.alignment, 0);
        assert!(end_offset(allocation) <= layout.scratch_arena_bytes);
    }

    for point in 0..=11 {
        let live = layout
            .scratch_allocations
            .iter()
            .filter(|allocation| live_at(allocation, point))
            .collect::<Vec<_>>();
        for (index, left) in live.iter().enumerate() {
            for right in &live[index + 1..] {
                assert!(
                    end_offset(left) <= right.offset || end_offset(right) <= left.offset,
                    "scratch overlap at point {point}: {} vs {}",
                    left.stage.as_str(),
                    right.stage.as_str()
                );
            }
        }
    }
}

fn assert_layout_error(
    plan: &VisionEncoderStackPlan,
    config: VisionStackActivationLayoutConfig,
    expected: InvocationErrorCode,
) {
    assert_eq!(plan.activation_layout(config).unwrap_err().code(), expected);
}

#[test]
fn stack_plan_composes_every_layer_and_post_norm_with_exact_readback_selection() {
    let fixtures = vec![Fixture::new(); 4];
    let layer_parameters = stack_parameters(&fixtures);
    let post_weight = values(HIDDEN as usize, 91);
    let post_bias = values(HIDDEN as usize, 92);
    let plan = stack_invocation(&fixtures[0], &layer_parameters, &post_weight, &post_bias)
        .plan(&[0, 2, 3])
        .unwrap();
    let layer_plan = fixtures[0].invocation().plan().unwrap();
    let hidden_elements = usize::try_from(TOKENS * HIDDEN).unwrap();

    assert_eq!(plan.layer_count, 4);
    assert_eq!(plan.layer_dispatches, layer_plan.dispatches);
    assert_eq!(plan.rope_specialization, VisionRopeSpecialization::Identity);
    assert_eq!(plan.checkpoint_layers, [0, 2, 3]);
    assert_eq!(plan.dispatch_count, 4 * 12 + 1);
    assert_eq!(plan.compute_pass_count, 5);
    assert_eq!(plan.post_norm_dispatch.kernel, KernelId::LayerNormF32);
    assert_eq!(plan.post_norm_dispatch.output_elements, hidden_elements);
    assert_eq!(
        plan.post_norm_dispatch.output_bytes,
        (hidden_elements * 4) as u64
    );
    assert_eq!(
        plan.post_norm_uniform_words,
        [TOKENS, HIDDEN, EPSILON.to_bits(), 0]
    );
    assert_eq!(plan.readback_bytes, (4 * hidden_elements * 4) as u64);
}

#[test]
fn stack_plan_keeps_activation_arena_constant_when_depth_grows() {
    let post_weight = values(HIDDEN as usize, 91);
    let post_bias = values(HIDDEN as usize, 92);
    let short_fixtures = vec![Fixture::new()];
    let short_parameters = stack_parameters(&short_fixtures);
    let short = stack_invocation(
        &short_fixtures[0],
        &short_parameters,
        &post_weight,
        &post_bias,
    )
    .plan(&[])
    .unwrap();
    let long_fixtures = vec![Fixture::new(); 64];
    let long_parameters = stack_parameters(&long_fixtures);
    let long = stack_invocation(
        &long_fixtures[0],
        &long_parameters,
        &post_weight,
        &post_bias,
    )
    .plan(&[0, 63])
    .unwrap();
    let hidden_elements = usize::try_from(TOKENS * HIDDEN).unwrap();
    let intermediate_elements = usize::try_from(TOKENS * INTERMEDIATE).unwrap();
    let expected_arena = ((11 * hidden_elements + 2 * intermediate_elements) * 4) as u64;
    let layer_weight_bytes = (layer_parameter_elements(&long_fixtures[0]) * 4) as u64;
    let post_weight_bytes = u64::from(HIDDEN) * 2 * 4;

    assert_eq!(short.activation_buffer_count, 13);
    assert_eq!(long.activation_buffer_count, 13);
    assert_eq!(short.activation_arena_bytes, expected_arena);
    assert_eq!(long.activation_arena_bytes, expected_arena);
    assert_eq!(
        short.resident_weight_bytes,
        layer_weight_bytes + post_weight_bytes
    );
    assert_eq!(
        long.resident_weight_bytes,
        64 * layer_weight_bytes + post_weight_bytes
    );
    assert_eq!(long.dispatch_count, 64 * 12 + 1);
    assert_eq!(long.compute_pass_count, 65);
    assert_eq!(long.readback_bytes, (3 * hidden_elements * 4) as u64);
}

#[test]
fn stack_plan_rejects_every_layer_post_norm_and_checkpoint_drift_before_execution() {
    let post_weight = values(HIDDEN as usize, 91);
    let post_bias = values(HIDDEN as usize, 92);
    let no_parameters = [];
    assert_eq!(
        stack_invocation(&Fixture::new(), &no_parameters, &post_weight, &post_bias,)
            .plan(&[])
            .unwrap_err()
            .code(),
        InvocationErrorCode::ZeroDimension
    );

    let fixtures = vec![Fixture::new(); 4];
    let parameters = stack_parameters(&fixtures);
    for checkpoints in [&[1, 0][..], &[1, 1][..], &[0, 4][..]] {
        assert_eq!(
            stack_invocation(&fixtures[0], &parameters, &post_weight, &post_bias,)
                .plan(checkpoints)
                .unwrap_err()
                .code(),
            InvocationErrorCode::InvalidCheckpointSelection
        );
    }

    assert_eq!(
        stack_invocation(
            &fixtures[0],
            &parameters,
            &post_weight[..HIDDEN as usize - 1],
            &post_bias,
        )
        .plan(&[])
        .unwrap_err()
        .code(),
        InvocationErrorCode::LengthMismatch
    );
    let mut nonfinite_post_bias = post_bias.clone();
    nonfinite_post_bias[HIDDEN as usize - 1] = f32::INFINITY;
    assert_eq!(
        stack_invocation(
            &fixtures[0],
            &parameters,
            &post_weight,
            &nonfinite_post_bias,
        )
        .plan(&[])
        .unwrap_err()
        .code(),
        InvocationErrorCode::NonFiniteInput
    );

    for layer in 0..4 {
        let operand = 1 + (layer * 5) % 16;
        let mut short_fixtures = vec![Fixture::new(); 4];
        short_fixtures[layer].floating_operand_mut(operand).pop();
        let short_parameters = stack_parameters(&short_fixtures);
        assert_eq!(
            stack_invocation(
                &short_fixtures[0],
                &short_parameters,
                &post_weight,
                &post_bias,
            )
            .plan(&[])
            .unwrap_err()
            .code(),
            InvocationErrorCode::LengthMismatch,
            "stack skipped length validation for layer {layer}, operand {operand}"
        );

        let mut nonfinite_fixtures = vec![Fixture::new(); 4];
        let values = nonfinite_fixtures[layer].floating_operand_mut(operand);
        let position = [0, values.len() / 2, values.len() - 1][layer % 3];
        values[position] = f32::NAN;
        let nonfinite_parameters = stack_parameters(&nonfinite_fixtures);
        assert_eq!(
            stack_invocation(
                &nonfinite_fixtures[0],
                &nonfinite_parameters,
                &post_weight,
                &post_bias,
            )
            .plan(&[])
            .unwrap_err()
            .code(),
            InvocationErrorCode::NonFiniteInput,
            "stack skipped finite validation for layer {layer}, operand {operand}"
        );
    }
}

#[test]
fn stack_activation_layout_freezes_exact_compact_alias_and_noalias_scratch_offsets() {
    let short = compact_stack_plan(1);
    let long = compact_stack_plan(64);
    let no_alias_expected = expected_no_alias_scratch_allocations();
    let alias_expected = expected_alias_scratch_allocations();

    assert_eq!(short.activation_buffer_count, 13);
    assert_eq!(long.activation_buffer_count, 13);
    assert_eq!(short.activation_arena_bytes, 8_784);
    assert_eq!(long.activation_arena_bytes, 8_784);

    let short_no_alias = short.activation_layout(layout_config(false, 4, 4)).unwrap();
    let short_alias = short.activation_layout(layout_config(true, 4, 4)).unwrap();
    let long_no_alias = long.activation_layout(layout_config(false, 4, 4)).unwrap();
    let long_alias = long.activation_layout(layout_config(true, 4, 4)).unwrap();

    assert_layout_matches_expected(&short_no_alias, &no_alias_expected, 7_488, 8_784);
    assert_layout_matches_expected(&short_alias, &alias_expected, 2_592, 3_888);
    assert_layout_matches_expected(&long_no_alias, &no_alias_expected, 7_488, 8_784);
    assert_layout_matches_expected(&long_alias, &alias_expected, 2_592, 3_888);

    assert_eq!(
        short_no_alias.scratch_allocations,
        long_no_alias.scratch_allocations
    );
    assert_eq!(
        short_alias.scratch_allocations,
        long_alias.scratch_allocations
    );
    assert_eq!(
        short_no_alias.scratch_arena_bytes,
        long_no_alias.scratch_arena_bytes
    );
    assert_eq!(
        short_alias.scratch_arena_bytes,
        long_alias.scratch_arena_bytes
    );
    assert_eq!(short_no_alias.physical_buffer_count, 3);
    assert_eq!(short_alias.physical_buffer_count, 3);
    assert!(short_alias.scratch_arena_bytes < short_no_alias.scratch_arena_bytes);
    assert!(short_alias.total_activation_bytes < short_no_alias.total_activation_bytes);
}

#[test]
fn stack_activation_layout_respects_storage_alignment_rounding_and_repeatability() {
    let plan = compact_stack_plan(4);
    let first = plan
        .activation_layout(layout_config(false, 256, 512))
        .unwrap();
    let second = plan
        .activation_layout(layout_config(false, 256, 512))
        .unwrap();

    assert_eq!(first.scratch_allocations, second.scratch_allocations);
    assert_eq!(first.scratch_arena_bytes, second.scratch_arena_bytes);
    assert_eq!(first.scratch_arena_bytes, 9_216);
    assert_eq!(
        first
            .scratch_allocations
            .iter()
            .map(|allocation| allocation.offset)
            .collect::<Vec<_>>(),
        vec![
            0, 768, 1_536, 2_304, 3_072, 3_840, 4_608, 5_376, 6_144, 7_168, 8_192
        ]
    );
    for allocation in &first.scratch_allocations {
        assert_eq!(allocation.alignment, 256);
        assert_eq!(allocation.offset % 256, 0);
        assert!(end_offset(allocation) <= first.scratch_arena_bytes);
    }
    assert_eq!(first.scratch_arena_bytes % 512, 0);
}

#[test]
fn stack_activation_layout_rejects_invalid_alignments_and_u64_overflow() {
    let plan = compact_stack_plan(1);

    for config in [
        layout_config(false, 0, 4),
        layout_config(false, 4, 0),
        layout_config(false, 12, 4),
        layout_config(false, 4, 24),
    ] {
        assert_layout_error(&plan, config, InvocationErrorCode::InvalidActivationLayout);
    }

    assert_layout_error(
        &plan,
        layout_config(false, 1_u64 << 63, 1_u64 << 63),
        InvocationErrorCode::ArithmeticOverflow,
    );
}

#[test]
fn layer_stack_activation_layout_matches_stack_plan_for_compact_geometry_and_base_alignments() {
    let layer = compact_layer_plan();
    let stack = compact_stack_plan(1);

    for config in [
        layout_config(false, 32, 32),
        layout_config(true, 32, 32),
        layout_config(false, 256, 256),
        layout_config(true, 256, 256),
    ] {
        assert_eq!(
            layer.stack_activation_layout(config).unwrap(),
            stack.activation_layout(config).unwrap()
        );
    }
}
