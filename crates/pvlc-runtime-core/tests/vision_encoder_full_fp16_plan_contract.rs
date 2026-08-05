use pvlc_runtime_core::{
    DecoderWeightStorage, KernelId, LinearWeightLayout, VisionEncoderLayerGeometry,
    VisionEncoderLayerStage, VisionEncoderPrecision,
};

fn geometry() -> VisionEncoderLayerGeometry<'static> {
    VisionEncoderLayerGeometry {
        tokens: 9,
        hidden_size: 40,
        attention_heads: 5,
        head_dim: 8,
        intermediate_size: 64,
        layer_norm_epsilon: 1.0e-6,
        cu_seqlens: &[0, 4, 9],
    }
}

fn full_fp16() -> VisionEncoderPrecision {
    VisionEncoderPrecision {
        matrix_weight_storage: DecoderWeightStorage::F16,
        matrix_weight_layout: LinearWeightLayout::InputMajor,
        vector_weight_storage: DecoderWeightStorage::F16,
        activation_storage: DecoderWeightStorage::F16,
    }
}

#[test]
fn full_fp16_plan_keeps_all_twelve_layer_stages_in_half_storage() {
    let plan = geometry().plan_with_precision(full_fp16()).unwrap();
    let expected = [
        (VisionEncoderLayerStage::Norm1, KernelId::LayerNormF16),
        (
            VisionEncoderLayerStage::Query,
            KernelId::LinearProjectionF16,
        ),
        (VisionEncoderLayerStage::Key, KernelId::LinearProjectionF16),
        (
            VisionEncoderLayerStage::Value,
            KernelId::LinearProjectionF16,
        ),
        (
            VisionEncoderLayerStage::AttentionContext,
            KernelId::VisionAttentionF16,
        ),
        (
            VisionEncoderLayerStage::AttentionOutput,
            KernelId::LinearProjectionF16,
        ),
        (VisionEncoderLayerStage::AttentionResidual, KernelId::AddF16),
        (VisionEncoderLayerStage::Norm2, KernelId::LayerNormF16),
        (
            VisionEncoderLayerStage::MlpFc1,
            KernelId::LinearProjectionF16,
        ),
        (
            VisionEncoderLayerStage::MlpActivation,
            KernelId::GeluTanhF16,
        ),
        (
            VisionEncoderLayerStage::MlpOutput,
            KernelId::LinearProjectionF16,
        ),
        (VisionEncoderLayerStage::Output, KernelId::AddF16),
    ];
    assert_eq!(
        plan.dispatches
            .map(|dispatch| (dispatch.stage, dispatch.invocation.kernel)),
        expected,
    );
    assert!(plan.dispatches.iter().all(|dispatch| {
        dispatch.invocation.output_bytes == dispatch.invocation.output_elements as u64 * 2
    }));
    assert_eq!(
        plan.resident_intermediate_bytes,
        9 * 40 * 2 * 10 + 9 * 64 * 2 * 2,
    );

    for dispatch in plan.dispatches.iter().filter(|dispatch| {
        matches!(
            dispatch.stage,
            VisionEncoderLayerStage::Query
                | VisionEncoderLayerStage::Key
                | VisionEncoderLayerStage::Value
                | VisionEncoderLayerStage::AttentionOutput
                | VisionEncoderLayerStage::MlpFc1
                | VisionEncoderLayerStage::MlpOutput
        )
    }) {
        assert_eq!(dispatch.uniform_words[3], 1);
        assert_eq!(
            dispatch.invocation.dispatch,
            [dispatch.uniform_words[2].div_ceil(32), 1, 1],
        );
    }
}

#[test]
fn full_fp16_plan_vectorizes_elementwise_dispatches_without_changing_semantic_lengths() {
    let plan = geometry().plan_with_precision(full_fp16()).unwrap();
    for stage in [
        VisionEncoderLayerStage::AttentionResidual,
        VisionEncoderLayerStage::MlpActivation,
        VisionEncoderLayerStage::Output,
    ] {
        let dispatch = plan
            .dispatches
            .iter()
            .find(|dispatch| dispatch.stage == stage)
            .unwrap();
        let semantic_elements = dispatch.invocation.output_elements;
        assert_eq!(dispatch.uniform_words[0], semantic_elements as u32);
        assert_eq!(
            dispatch.invocation.dispatch,
            [(semantic_elements as u32).div_ceil(4).div_ceil(64), 1, 1],
            "one invocation must process one packed vec4<f16>",
        );
    }
}

#[test]
fn runtime_rejects_partial_profiles_before_any_kernel_can_be_dispatched() {
    for precision in [
        VisionEncoderPrecision {
            vector_weight_storage: DecoderWeightStorage::F32,
            ..full_fp16()
        },
        VisionEncoderPrecision {
            activation_storage: DecoderWeightStorage::F32,
            ..full_fp16()
        },
        VisionEncoderPrecision {
            matrix_weight_storage: DecoderWeightStorage::F32,
            matrix_weight_layout: LinearWeightLayout::OutputMajor,
            ..full_fp16()
        },
        VisionEncoderPrecision {
            matrix_weight_storage: DecoderWeightStorage::F32,
            matrix_weight_layout: LinearWeightLayout::OutputMajor,
            vector_weight_storage: DecoderWeightStorage::F32,
            activation_storage: DecoderWeightStorage::F16,
        },
        VisionEncoderPrecision {
            matrix_weight_layout: LinearWeightLayout::OutputMajor,
            ..full_fp16()
        },
    ] {
        let error = geometry()
            .plan_with_precision(precision)
            .expect_err("partial FP16 profiles have incompatible buffer ABIs");
        assert_eq!(
            error.code(),
            pvlc_runtime_core::InvocationErrorCode::InvalidVisionGeometry,
        );
        assert!(error.to_string().contains("full FP16"));
    }
}

#[test]
fn legacy_mixed_fp16_planner_remains_the_existing_f32_activation_path() {
    let legacy = geometry()
        .plan_with_matrix_weight_storage_and_layout(
            DecoderWeightStorage::F16,
            LinearWeightLayout::InputMajor,
        )
        .unwrap();
    let explicit = geometry()
        .plan_with_precision(VisionEncoderPrecision {
            matrix_weight_storage: DecoderWeightStorage::F16,
            matrix_weight_layout: LinearWeightLayout::InputMajor,
            vector_weight_storage: DecoderWeightStorage::F32,
            activation_storage: DecoderWeightStorage::F32,
        })
        .unwrap();
    assert_eq!(legacy, explicit);
    assert!(legacy.dispatches.iter().all(|dispatch| {
        dispatch.invocation.output_bytes == dispatch.invocation.output_elements as u64 * 4
    }));
}

#[test]
fn production_shape_covers_multiple_row_workgroups_and_both_projection_tails() {
    let plan = VisionEncoderLayerGeometry {
        tokens: 1_836,
        hidden_size: 1_152,
        attention_heads: 16,
        head_dim: 72,
        intermediate_size: 4_304,
        layer_norm_epsilon: 1.0e-6,
        cu_seqlens: &[0, 1_836],
    }
    .plan_with_precision(full_fp16())
    .unwrap();

    for dispatch in plan.dispatches.iter().filter(|dispatch| {
        matches!(
            dispatch.stage,
            VisionEncoderLayerStage::Query
                | VisionEncoderLayerStage::Key
                | VisionEncoderLayerStage::Value
                | VisionEncoderLayerStage::AttentionOutput
                | VisionEncoderLayerStage::MlpFc1
                | VisionEncoderLayerStage::MlpOutput
        )
    }) {
        let output_width = dispatch.uniform_words[2];
        assert_eq!(
            dispatch.invocation.dispatch,
            [output_width.div_ceil(32), 58, 1],
            "1,836 rows require 58 row tiles and a 12-row tail",
        );
    }
    let mlp_fc1 = plan
        .dispatches
        .iter()
        .find(|dispatch| dispatch.stage == VisionEncoderLayerStage::MlpFc1)
        .unwrap();
    assert_eq!(
        mlp_fc1.invocation.dispatch[0], 135,
        "4,304 columns require a partial final 32-column tile",
    );

    for stage in [
        VisionEncoderLayerStage::AttentionResidual,
        VisionEncoderLayerStage::MlpActivation,
        VisionEncoderLayerStage::Output,
    ] {
        let dispatch = plan
            .dispatches
            .iter()
            .find(|dispatch| dispatch.stage == stage)
            .unwrap();
        let packed_elements = (dispatch.invocation.output_elements as u32).div_ceil(4);
        let capacity = dispatch.invocation.dispatch[0]
            * dispatch.invocation.dispatch[1]
            * dispatch.invocation.workgroup_size[0];
        assert!(capacity >= packed_elements);
        assert!(capacity - packed_elements < dispatch.invocation.workgroup_size[0]);
        assert!(
            dispatch.invocation.dispatch[0] > 1,
            "production elementwise work must span many workgroups",
        );
    }
}
