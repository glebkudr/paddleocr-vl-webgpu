use pvlc_runtime_core::{
    DecoderWeightStorage, KernelId, LinearWeightLayout, VisionEncoderLayerGeometry,
    VisionEncoderLayerStage,
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

fn plan(storage: DecoderWeightStorage) -> pvlc_runtime_core::VisionEncoderLayerPlan {
    geometry().plan_with_matrix_weight_storage(storage).unwrap()
}

#[test]
fn fp16_storage_selects_fp16_only_for_the_six_matrix_projections() {
    let f16 = plan(DecoderWeightStorage::F16);
    let f32 = plan(DecoderWeightStorage::F32);
    let expected = [
        (VisionEncoderLayerStage::Norm1, KernelId::LayerNormF32),
        (
            VisionEncoderLayerStage::Query,
            KernelId::LinearProjectionF16Weights,
        ),
        (
            VisionEncoderLayerStage::Key,
            KernelId::LinearProjectionF16Weights,
        ),
        (
            VisionEncoderLayerStage::Value,
            KernelId::LinearProjectionF16Weights,
        ),
        (
            VisionEncoderLayerStage::AttentionContext,
            KernelId::VisionAttentionF32,
        ),
        (
            VisionEncoderLayerStage::AttentionOutput,
            KernelId::LinearProjectionF16Weights,
        ),
        (VisionEncoderLayerStage::AttentionResidual, KernelId::AddF32),
        (VisionEncoderLayerStage::Norm2, KernelId::LayerNormF32),
        (
            VisionEncoderLayerStage::MlpFc1,
            KernelId::LinearProjectionF16Weights,
        ),
        (
            VisionEncoderLayerStage::MlpActivation,
            KernelId::GeluTanhF32,
        ),
        (
            VisionEncoderLayerStage::MlpOutput,
            KernelId::LinearProjectionF16Weights,
        ),
        (VisionEncoderLayerStage::Output, KernelId::AddF32),
    ];

    assert_eq!(
        f16.dispatches
            .map(|dispatch| (dispatch.stage, dispatch.invocation.kernel)),
        expected
    );
    assert!(
        f16.dispatches
            .iter()
            .all(|dispatch| dispatch.invocation.output_bytes
                == dispatch.invocation.output_elements as u64 * 4),
        "activations and attention remain F32 while checkpoint matrices use F16 storage",
    );
    assert_eq!(f16.rope_specialization, f32.rope_specialization);
    assert_eq!(
        f16.resident_intermediate_bytes,
        f32.resident_intermediate_bytes
    );
    for (f16_dispatch, f32_dispatch) in f16.dispatches.iter().zip(f32.dispatches) {
        assert_eq!(f16_dispatch.stage, f32_dispatch.stage);
        assert_eq!(
            f16_dispatch.invocation.output_elements,
            f32_dispatch.invocation.output_elements,
        );
        assert_eq!(
            f16_dispatch.invocation.output_bytes,
            f32_dispatch.invocation.output_bytes,
        );
        assert_eq!(
            f16_dispatch.invocation.workgroup_size,
            f32_dispatch.invocation.workgroup_size,
        );
        assert_eq!(f16_dispatch.uniform_words, f32_dispatch.uniform_words);
        if matches!(
            f16_dispatch.stage,
            VisionEncoderLayerStage::Query
                | VisionEncoderLayerStage::Key
                | VisionEncoderLayerStage::Value
                | VisionEncoderLayerStage::AttentionOutput
                | VisionEncoderLayerStage::MlpFc1
                | VisionEncoderLayerStage::MlpOutput
        ) {
            assert_eq!(
                f16_dispatch.invocation.kernel,
                KernelId::LinearProjectionF16Weights,
            );
            assert_eq!(
                f32_dispatch.invocation.kernel,
                KernelId::VisionPatchProjectionF32,
            );
            let output_width: u32 = match f16_dispatch.stage {
                VisionEncoderLayerStage::MlpFc1 => 64,
                _ => 40,
            };
            assert_eq!(
                f16_dispatch.invocation.dispatch,
                [output_width.div_ceil(32), 1, 1],
                "each FP16 workgroup must produce a full 32x32 output tile",
            );
            assert_eq!(
                f32_dispatch.invocation.dispatch,
                [output_width.div_ceil(32), 1, 1],
                "the F32 projection must use the same 32x32 tiled output topology",
            );
        } else {
            assert_eq!(
                f16_dispatch.invocation.kernel,
                f32_dispatch.invocation.kernel,
            );
            assert_eq!(
                f16_dispatch.invocation.dispatch,
                f32_dispatch.invocation.dispatch,
            );
        }
    }
}

#[test]
fn fp16_matrix_storage_rejects_widths_that_cannot_use_packed_vec4_io() {
    let unaligned_intermediate = VisionEncoderLayerGeometry {
        intermediate_size: 65,
        ..geometry()
    }
    .plan_with_matrix_weight_storage(DecoderWeightStorage::F16)
    .expect_err("an unpacked FP16 MLP matrix must not reach the packed shader");
    assert_eq!(
        unaligned_intermediate.code(),
        pvlc_runtime_core::InvocationErrorCode::InvalidVisionGeometry,
    );
    assert!(
        unaligned_intermediate
            .to_string()
            .contains("multiple of 4"),
    );

    let unaligned_hidden = VisionEncoderLayerGeometry {
        hidden_size: 42,
        attention_heads: 7,
        head_dim: 6,
        ..geometry()
    }
    .plan_with_matrix_weight_storage(DecoderWeightStorage::F16)
    .expect_err("an unpacked FP16 attention matrix must not reach the packed shader");
    assert_eq!(
        unaligned_hidden.code(),
        pvlc_runtime_core::InvocationErrorCode::InvalidVisionGeometry,
    );

    assert!(
        VisionEncoderLayerGeometry {
            intermediate_size: 65,
            ..geometry()
        }
        .plan_with_matrix_weight_storage(DecoderWeightStorage::F32)
        .is_ok(),
        "the scalar F32 compatibility path must retain an unaligned MLP width",
    );
    assert!(
        VisionEncoderLayerGeometry {
            hidden_size: 42,
            attention_heads: 7,
            head_dim: 6,
            intermediate_size: 65,
            ..geometry()
        }
        .plan_with_matrix_weight_storage(DecoderWeightStorage::F32)
        .is_ok(),
        "the scalar F32 compatibility path must retain valid unaligned hidden and MLP widths",
    );
}

#[test]
fn input_major_fp16_layout_is_explicit_only_on_the_six_projection_uniforms() {
    let output_major = plan(DecoderWeightStorage::F16);
    let explicit_output_major = geometry()
        .plan_with_matrix_weight_storage_and_layout(
            DecoderWeightStorage::F16,
            LinearWeightLayout::OutputMajor,
        )
        .unwrap();
    let input_major = geometry()
        .plan_with_matrix_weight_storage_and_layout(
            DecoderWeightStorage::F16,
            LinearWeightLayout::InputMajor,
        )
        .unwrap();
    assert_eq!(
        explicit_output_major, output_major,
        "the new layout-aware API must preserve the legacy F16 output-major plan",
    );

    for (output_dispatch, input_dispatch) in
        output_major.dispatches.iter().zip(input_major.dispatches)
    {
        assert_eq!(output_dispatch.stage, input_dispatch.stage);
        assert_eq!(output_dispatch.invocation, input_dispatch.invocation);
        assert_eq!(
            &output_dispatch.uniform_words[..3],
            &input_dispatch.uniform_words[..3],
        );
        let is_projection = matches!(
            input_dispatch.stage,
            VisionEncoderLayerStage::Query
                | VisionEncoderLayerStage::Key
                | VisionEncoderLayerStage::Value
                | VisionEncoderLayerStage::AttentionOutput
                | VisionEncoderLayerStage::MlpFc1
                | VisionEncoderLayerStage::MlpOutput
        );
        if is_projection {
            assert_eq!(output_dispatch.uniform_words[3], 0);
            assert_eq!(
                input_dispatch.uniform_words[3], 1,
                "each FP16 matrix projection must select input-major addressing",
            );
        } else {
            assert_eq!(
                input_dispatch.uniform_words, output_dispatch.uniform_words,
                "non-projection uniforms must remain byte-for-byte unchanged",
            );
        }
    }

    assert_eq!(
        geometry()
            .plan_with_matrix_weight_storage_and_layout(
                DecoderWeightStorage::F32,
                LinearWeightLayout::InputMajor,
            )
            .unwrap_err()
            .code(),
        pvlc_runtime_core::InvocationErrorCode::InvalidVisionGeometry,
        "the scalar F32 compatibility shader only accepts output-major weights",
    );
}

#[test]
fn the_existing_plan_entrypoint_remains_the_f32_compatibility_path() {
    let geometry = geometry();
    let legacy = geometry.plan().unwrap();
    let explicit = plan(DecoderWeightStorage::F32);

    assert_eq!(legacy, explicit);
    for stage in [
        VisionEncoderLayerStage::Query,
        VisionEncoderLayerStage::Key,
        VisionEncoderLayerStage::Value,
        VisionEncoderLayerStage::AttentionOutput,
        VisionEncoderLayerStage::MlpFc1,
        VisionEncoderLayerStage::MlpOutput,
    ] {
        assert_eq!(
            legacy
                .dispatches
                .iter()
                .find(|dispatch| dispatch.stage == stage)
                .unwrap()
                .invocation
                .kernel,
            KernelId::VisionPatchProjectionF32,
        );
    }
}

#[test]
fn production_ocr_shape_uses_fifty_eight_projection_tiles_and_fifteen_query_tiles() {
    let plan = VisionEncoderLayerGeometry {
        tokens: 1_836,
        hidden_size: 1_152,
        attention_heads: 16,
        head_dim: 72,
        intermediate_size: 4_304,
        layer_norm_epsilon: 1.0e-6,
        cu_seqlens: &[0, 1_836],
    }
    .plan_with_matrix_weight_storage_and_layout(
        DecoderWeightStorage::F16,
        LinearWeightLayout::InputMajor,
    )
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
            "production projections must cover 1,836 rows in 32-row tiles",
        );
    }

    let attention = plan
        .dispatches
        .iter()
        .find(|dispatch| dispatch.stage == VisionEncoderLayerStage::AttentionContext)
        .unwrap();
    assert_eq!(attention.invocation.workgroup_size, [128, 1, 1]);
    assert_eq!(attention.invocation.dispatch, [15, 16, 1]);
}
