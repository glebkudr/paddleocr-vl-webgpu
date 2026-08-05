use pvlc_runtime_core::{
    InvocationErrorCode, KernelId, VISION_QKV_FUSED_STORAGE_BINDING_COUNT,
    VisionQkvFusedInvocation, VisionQkvFusedTargetLimits, plan_vision_qkv_fused_geometry,
};

#[derive(Clone)]
struct Operands {
    input: Vec<f32>,
    query_weight: Vec<f32>,
    query_bias: Vec<f32>,
    key_weight: Vec<f32>,
    key_bias: Vec<f32>,
    value_weight: Vec<f32>,
    value_bias: Vec<f32>,
}

impl Operands {
    fn asymmetric() -> Self {
        const TOKENS: usize = 3;
        const INPUT_WIDTH: usize = 3;
        const OUTPUT_WIDTH: usize = 5;

        let sequence = |length: usize, salt: usize| {
            (0..length)
                .map(|index| ((index * 17 + salt * 13) as f32 - 31.0) / 19.0)
                .collect()
        };
        Self {
            input: sequence(TOKENS * INPUT_WIDTH, 1),
            query_weight: sequence(OUTPUT_WIDTH * INPUT_WIDTH, 2),
            query_bias: sequence(OUTPUT_WIDTH, 3),
            key_weight: sequence(OUTPUT_WIDTH * INPUT_WIDTH, 4),
            key_bias: sequence(OUTPUT_WIDTH, 5),
            value_weight: sequence(OUTPUT_WIDTH * INPUT_WIDTH, 6),
            value_bias: sequence(OUTPUT_WIDTH, 7),
        }
    }

    fn invocation(&self) -> VisionQkvFusedInvocation<'_> {
        VisionQkvFusedInvocation {
            tokens: 3,
            input_width: 3,
            output_width: 5,
            input: &self.input,
            query_weight: &self.query_weight,
            query_bias: &self.query_bias,
            key_weight: &self.key_weight,
            key_bias: &self.key_bias,
            value_weight: &self.value_weight,
            value_bias: &self.value_bias,
        }
    }

    fn operand_mut(&mut self, index: usize) -> &mut Vec<f32> {
        match index {
            0 => &mut self.input,
            1 => &mut self.query_weight,
            2 => &mut self.query_bias,
            3 => &mut self.key_weight,
            4 => &mut self.key_bias,
            5 => &mut self.value_weight,
            6 => &mut self.value_bias,
            _ => panic!("operand index must be in the frozen seven-input ABI"),
        }
    }
}

fn limits(alignment: u32) -> VisionQkvFusedTargetLimits {
    VisionQkvFusedTargetLimits {
        min_storage_buffer_offset_alignment: alignment,
        max_storage_buffers_per_shader_stage: 8,
        max_storage_buffer_binding_size: 1_u64 << 34,
        max_buffer_size: 1_u64 << 34,
        max_compute_workgroups_per_dimension: 65_535,
    }
}

fn assert_error(
    invocation: VisionQkvFusedInvocation<'_>,
    target: VisionQkvFusedTargetLimits,
    expected: InvocationErrorCode,
) {
    let error = invocation
        .plan(target)
        .expect_err("invalid fused QKV invocation must fail before GPU effects");
    assert_eq!(error.code(), expected, "unexpected error: {error}");
}

#[test]
fn fused_qkv_is_an_additive_kernel_without_changing_the_frozen_legacy_prefix() {
    let legacy = [
        KernelId::GemmF32,
        KernelId::GemvF32,
        KernelId::LayerNormF32,
        KernelId::RmsNormF32,
        KernelId::SiluF32,
        KernelId::GeluTanhF32,
        KernelId::RopeNeoxF32,
        KernelId::VisionAttentionF32,
        KernelId::VisionPatchProjectionF32,
        KernelId::AddF32,
        KernelId::GeluErfF32,
        KernelId::ProjectorMerge2x2F32,
    ];
    assert_eq!(KernelId::M2_PRIMITIVES, legacy[..7]);
    assert_eq!(&KernelId::ALL[..legacy.len()], legacy);
    assert_eq!(KernelId::ALL[legacy.len()], KernelId::VisionQkvFusedF32);
    assert_eq!(
        &KernelId::ALL[legacy.len() + 1..legacy.len() + 11],
        [
            KernelId::DecoderKvAppendF32,
            KernelId::DecoderGqaF32,
            KernelId::DecoderGqaSplitPartialF32,
            KernelId::DecoderGqaSplitMergeF32,
            KernelId::DecoderMropeF32,
            KernelId::DecoderSwigluF32,
            KernelId::DecoderPrefillGqaF32,
            KernelId::DecoderPrefillMropeF32,
            KernelId::DecoderKvAppendRangeF32,
            KernelId::GemvTiledF32,
        ]
    );
    assert_eq!(KernelId::VisionQkvFusedF32.as_str(), "vision_qkv_fused_f32");
    assert_eq!(KernelId::DecoderMropeF32.as_str(), "decoder_mrope_f32");
    assert_eq!(VISION_QKV_FUSED_STORAGE_BINDING_COUNT, 8);
}

#[test]
fn plan_freezes_the_eight_storage_input_order_and_exact_padded_three_plane_layout() {
    let operands = Operands::asymmetric();
    let invocation = operands.invocation();
    let inputs = invocation.inputs();
    assert_eq!(inputs.len(), 7);
    assert_eq!(inputs[0], operands.input);
    assert_eq!(inputs[1], operands.query_weight);
    assert_eq!(inputs[2], operands.query_bias);
    assert_eq!(inputs[3], operands.key_weight);
    assert_eq!(inputs[4], operands.key_bias);
    assert_eq!(inputs[5], operands.value_weight);
    assert_eq!(inputs[6], operands.value_bias);

    let plan = invocation.plan(limits(32)).unwrap();
    assert_eq!(plan.invocation.kernel, KernelId::VisionQkvFusedF32);
    assert_eq!(plan.invocation.workgroup_size, [8, 8, 1]);
    assert_eq!(plan.invocation.dispatch, [1, 1, 3]);
    assert_eq!(plan.invocation.output_elements, 48);
    assert_eq!(plan.invocation.output_bytes, 192);
    assert_eq!(plan.uniform_words, [3, 3, 5, 16]);

    let layout = plan.output_layout;
    assert_eq!(layout.plane_elements, 15);
    assert_eq!(layout.plane_bytes, 60);
    assert_eq!(layout.plane_stride_bytes, 64);
    assert_eq!(layout.physical_bytes, 192);
    assert_eq!((layout.query.offset, layout.query.size), (0, 60));
    assert_eq!((layout.key.offset, layout.key.size), (64, 60));
    assert_eq!((layout.value.offset, layout.value.size), (128, 60));

    for slice in [layout.query, layout.key, layout.value] {
        assert_eq!(slice.offset % 32, 0);
        assert!(slice.offset + slice.size <= layout.physical_bytes);
    }
    assert!(layout.query.offset + layout.query.size <= layout.key.offset);
    assert!(layout.key.offset + layout.key.size <= layout.value.offset);
    assert_eq!(
        layout.physical_bytes - (layout.value.offset + layout.value.size),
        4
    );
}

#[test]
fn geometry_only_planner_freezes_both_official_shapes_for_32_and_256_alignment() {
    for alignment in [32, 256] {
        let l3 = plan_vision_qkv_fused_geometry(1_276, 1_152, 1_152, limits(alignment))
            .expect("official L3 fused QKV geometry must fit portable WebGPU limits");
        assert_eq!(l3.output_layout.plane_elements, 1_469_952);
        assert_eq!(l3.output_layout.plane_bytes, 5_879_808);
        assert_eq!(l3.output_layout.plane_stride_bytes, 5_879_808);
        assert_eq!(l3.output_layout.physical_bytes, 17_639_424);
        assert_eq!(l3.invocation.output_elements, 4_409_856);
        assert_eq!(l3.invocation.dispatch, [144, 160, 3]);
        assert_eq!(l3.uniform_words, [1_276, 1_152, 1_152, 1_469_952]);

        let l2 = plan_vision_qkv_fused_geometry(1_740, 1_152, 1_152, limits(alignment))
            .expect("official L2 fused QKV geometry must fit portable WebGPU limits");
        assert_eq!(l2.output_layout.plane_elements, 2_004_480);
        assert_eq!(l2.output_layout.plane_bytes, 8_017_920);
        assert_eq!(l2.output_layout.plane_stride_bytes, 8_017_920);
        assert_eq!(l2.output_layout.physical_bytes, 24_053_760);
        assert_eq!(l2.invocation.output_elements, 6_013_440);
        assert_eq!(l2.invocation.dispatch, [144, 218, 3]);
        assert_eq!(l2.uniform_words, [1_740, 1_152, 1_152, 2_004_480]);
    }
}

#[test]
fn tile_boundaries_and_alignment_padding_are_derived_without_shape_shortcuts() {
    for (tokens, output_width, expected_dispatch) in [
        (7, 7, [1, 1, 3]),
        (8, 8, [1, 1, 3]),
        (9, 9, [2, 2, 3]),
        (17, 33, [5, 3, 3]),
    ] {
        let plan = plan_vision_qkv_fused_geometry(tokens, 3, output_width, limits(256)).unwrap();
        let plane_bytes = u64::from(tokens) * u64::from(output_width) * 4;
        let stride = plane_bytes.div_ceil(256) * 256;
        assert_eq!(plan.invocation.dispatch, expected_dispatch);
        assert_eq!(plan.output_layout.plane_bytes, plane_bytes);
        assert_eq!(plan.output_layout.plane_stride_bytes, stride);
        assert_eq!(plan.output_layout.physical_bytes, stride * 3);
        assert_eq!(plan.uniform_words[3], u32::try_from(stride / 4).unwrap());
    }
}

#[test]
fn every_f32_compatible_power_of_two_alignment_derives_stride_from_the_target() {
    let plane_bytes = 3_u64 * 5 * 4;
    for alignment in [4, 8, 16, 32, 64, 128, 256] {
        let plan = plan_vision_qkv_fused_geometry(3, 3, 5, limits(alignment)).unwrap();
        let expected_stride = plane_bytes.div_ceil(u64::from(alignment)) * u64::from(alignment);
        assert_eq!(plan.output_layout.plane_bytes, plane_bytes);
        assert_eq!(plan.output_layout.plane_stride_bytes, expected_stride);
        assert_eq!(plan.output_layout.physical_bytes, expected_stride * 3);
        assert_eq!(
            plan.uniform_words[3],
            u32::try_from(expected_stride / 4).unwrap()
        );
    }
}

#[test]
fn target_limit_eight_is_portable_and_malformed_alignment_fails_closed() {
    let operands = Operands::asymmetric();
    operands.invocation().plan(limits(32)).unwrap();

    let mut target = limits(32);
    target.max_storage_buffers_per_shader_stage = 7;
    assert_error(
        operands.invocation(),
        target,
        InvocationErrorCode::InvalidFusionTarget,
    );

    for alignment in [0, 1, 2, 3, 6, 12, 24, 48] {
        assert_error(
            operands.invocation(),
            limits(alignment),
            InvocationErrorCode::InvalidFusionTarget,
        );
    }
}

#[test]
fn target_dispatch_limit_checks_x_and_y_independently_at_the_exact_boundary() {
    let mut target = limits(4);
    target.max_compute_workgroups_per_dimension = 3;

    let x_boundary = plan_vision_qkv_fused_geometry(1, 1, 17, target).unwrap();
    assert_eq!(x_boundary.invocation.dispatch, [3, 1, 3]);
    let y_boundary = plan_vision_qkv_fused_geometry(17, 1, 1, target).unwrap();
    assert_eq!(y_boundary.invocation.dispatch, [1, 3, 3]);
    let both_boundaries = plan_vision_qkv_fused_geometry(17, 1, 17, target).unwrap();
    assert_eq!(both_boundaries.invocation.dispatch, [3, 3, 3]);

    let x_error = plan_vision_qkv_fused_geometry(17, 1, 25, target).unwrap_err();
    assert_eq!(x_error.code(), InvocationErrorCode::InvalidFusionTarget);
    let y_error = plan_vision_qkv_fused_geometry(25, 1, 17, target).unwrap_err();
    assert_eq!(y_error.code(), InvocationErrorCode::InvalidFusionTarget);
}

#[test]
fn storage_binding_and_buffer_limits_accept_exact_sizes_and_reject_each_dominant_class() {
    for (tokens, input_width, output_width, exact_largest_bytes) in [
        (2, 100, 1, 800_u64), // input dominates
        (1, 100, 2, 800_u64), // each projection weight dominates
        (3, 1, 5, 180_u64),   // padded three-plane output dominates
    ] {
        let mut exact = limits(4);
        exact.max_storage_buffer_binding_size = exact_largest_bytes;
        exact.max_buffer_size = exact_largest_bytes;
        plan_vision_qkv_fused_geometry(tokens, input_width, output_width, exact).unwrap();

        let mut short_binding = limits(4);
        short_binding.max_storage_buffer_binding_size = exact_largest_bytes - 1;
        let binding_error =
            plan_vision_qkv_fused_geometry(tokens, input_width, output_width, short_binding)
                .unwrap_err();
        assert_eq!(
            binding_error.code(),
            InvocationErrorCode::InvalidFusionTarget
        );

        let mut short_buffer = limits(4);
        short_buffer.max_buffer_size = exact_largest_bytes - 1;
        let buffer_error =
            plan_vision_qkv_fused_geometry(tokens, input_width, output_width, short_buffer)
                .unwrap_err();
        assert_eq!(
            buffer_error.code(),
            InvocationErrorCode::InvalidFusionTarget
        );
    }
}

#[test]
fn zero_geometry_and_all_short_or_long_operands_are_rejected_with_frozen_precedence() {
    let operands = Operands::asymmetric();
    for field in 0..3 {
        let mut invocation = operands.invocation();
        match field {
            0 => invocation.tokens = 0,
            1 => invocation.input_width = 0,
            2 => invocation.output_width = 0,
            _ => unreachable!(),
        }
        assert_error(invocation, limits(32), InvocationErrorCode::ZeroDimension);
    }

    for operand_index in 0..7 {
        let mut short = operands.clone();
        short.operand_mut(operand_index).pop();
        assert_error(
            short.invocation(),
            limits(32),
            InvocationErrorCode::LengthMismatch,
        );

        let mut long = operands.clone();
        long.operand_mut(operand_index).push(0.0);
        assert_error(
            long.invocation(),
            limits(32),
            InvocationErrorCode::LengthMismatch,
        );
    }
}

#[test]
fn every_nonfinite_kind_and_position_in_all_seven_inputs_is_rejected() {
    let operands = Operands::asymmetric();
    for operand_index in 0..7 {
        let length = operands.clone().operand_mut(operand_index).len();
        let positions = [0, length / 2, length - 1];
        for position in positions {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut invalid = operands.clone();
                invalid.operand_mut(operand_index)[position] = value;
                assert_error(
                    invalid.invocation(),
                    limits(32),
                    InvocationErrorCode::NonFiniteInput,
                );
            }
        }
    }
}

#[test]
fn checked_host_wgsl_and_dispatch_arithmetic_fails_before_operand_lengths() {
    let empty = Operands {
        input: vec![],
        query_weight: vec![],
        query_bias: vec![],
        key_weight: vec![],
        key_bias: vec![],
        value_weight: vec![],
        value_bias: vec![],
    };
    let invocation = |tokens, input_width, output_width| VisionQkvFusedInvocation {
        tokens,
        input_width,
        output_width,
        input: &empty.input,
        query_weight: &empty.query_weight,
        query_bias: &empty.query_bias,
        key_weight: &empty.key_weight,
        key_bias: &empty.key_bias,
        value_weight: &empty.value_weight,
        value_bias: &empty.value_bias,
    };

    assert_error(
        invocation(u32::MAX, u32::MAX, u32::MAX),
        limits(256),
        InvocationErrorCode::ArithmeticOverflow,
    );
    assert_error(
        invocation(65_536, 1, 65_536),
        limits(256),
        InvocationErrorCode::ArithmeticOverflow,
    );
    assert_error(
        invocation(65_535 * 8 + 1, 1, 1),
        limits(256),
        InvocationErrorCode::ArithmeticOverflow,
    );
    assert_error(
        invocation(1, 1, 65_535 * 8 + 1),
        limits(256),
        InvocationErrorCode::ArithmeticOverflow,
    );
}
