use pvlc_runtime_core::{
    InvocationErrorCode, KernelId, VisionEncoderLayerExecutionStage, VisionEncoderLayerGeometry,
    VisionRope2dDescriptor, VisionRopeSpecialization,
};

#[test]
fn spatial_vision_rope_is_part_of_every_real_layer_and_reuses_one_table_upload() {
    let tokens = 1_836_u32;
    let heads = 16_u32;
    let head_dim = 72_u32;
    let table_elements = tokens * head_dim / 2;
    let cos = vec![1.0; table_elements as usize];
    let sin = vec![0.0; table_elements as usize];

    let rope_plan = VisionRope2dDescriptor {
        tokens,
        heads,
        head_dim,
        cos: &cos,
        sin: &sin,
    }
    .plan()
    .unwrap();
    let base_layer = VisionEncoderLayerGeometry {
        tokens,
        hidden_size: heads * head_dim,
        attention_heads: heads,
        head_dim,
        intermediate_size: 4_304,
        layer_norm_epsilon: 1.0e-6,
        cu_seqlens: &[0, tokens],
    }
    .plan()
    .unwrap();
    let layer_plan = base_layer.with_spatial_rope(rope_plan).unwrap();

    assert_eq!(
        layer_plan.rope_specialization,
        VisionRopeSpecialization::Spatial2d
    );
    assert_eq!(
        layer_plan.execution_stages,
        [
            VisionEncoderLayerExecutionStage::Norm1,
            VisionEncoderLayerExecutionStage::Query,
            VisionEncoderLayerExecutionStage::Key,
            VisionEncoderLayerExecutionStage::Value,
            VisionEncoderLayerExecutionStage::SpatialRope,
            VisionEncoderLayerExecutionStage::AttentionContext,
            VisionEncoderLayerExecutionStage::AttentionOutput,
            VisionEncoderLayerExecutionStage::AttentionResidual,
            VisionEncoderLayerExecutionStage::Norm2,
            VisionEncoderLayerExecutionStage::MlpFc1,
            VisionEncoderLayerExecutionStage::MlpActivation,
            VisionEncoderLayerExecutionStage::MlpOutput,
            VisionEncoderLayerExecutionStage::Output,
        ]
    );
    assert_eq!(layer_plan.rope.invocation.kernel, KernelId::VisionRope2dF32);
    assert_eq!(layer_plan.rope.invocation.workgroup_size, [64, 1, 1]);
    assert_eq!(
        layer_plan.rope.invocation.dispatch,
        [(tokens * heads * (head_dim / 2)).div_ceil(64), 1, 1]
    );
    assert_eq!(
        layer_plan.rope.invocation.output_elements,
        2 * tokens as usize * heads as usize * head_dim as usize
    );
    assert_eq!(
        layer_plan.rope.invocation.output_bytes,
        2 * tokens as u64 * heads as u64 * head_dim as u64 * 4
    );
    assert_eq!(layer_plan.rope.table_elements, table_elements as usize);
    assert_eq!(layer_plan.rope.table_bytes, table_elements as u64 * 4);
    assert_eq!(layer_plan.rope.uniform_words, [tokens, heads, head_dim, 0]);

    let stack_plan = layer_plan.stack_plan(27).unwrap();
    assert_eq!(stack_plan.layer_count, 27);
    assert_eq!(stack_plan.layer_dispatch_count, 27 * 13);
    assert_eq!(stack_plan.rope_dispatch_count, 27);
    assert_eq!(stack_plan.post_norm_dispatch_count, 1);
    assert_eq!(stack_plan.dispatch_count, 27 * 13 + 1);
    assert_eq!(stack_plan.rope_table_buffer_count, 2);
    assert_eq!(stack_plan.rope_table_upload_count, 2);
}

#[test]
fn spatial_vision_rope_plan_rejects_wrong_tables_geometry_and_nonfinite_values() {
    let valid = vec![0.0; 4];
    let invalid_geometry = [
        (
            0,
            1,
            8,
            &valid[..],
            &valid[..],
            InvocationErrorCode::ZeroDimension,
        ),
        (
            1,
            0,
            8,
            &valid[..],
            &valid[..],
            InvocationErrorCode::ZeroDimension,
        ),
        (
            1,
            1,
            6,
            &valid[..3],
            &valid[..3],
            InvocationErrorCode::InvalidRotaryDimension,
        ),
        (
            1,
            1,
            76,
            &valid[..],
            &valid[..],
            InvocationErrorCode::UnsupportedHeadDimension,
        ),
    ];
    for (tokens, heads, head_dim, cos, sin, expected) in invalid_geometry {
        assert_eq!(
            VisionRope2dDescriptor {
                tokens,
                heads,
                head_dim,
                cos,
                sin,
            }
            .plan()
            .unwrap_err()
            .code(),
            expected
        );
    }

    for (cos, sin, expected) in [
        (&valid[..3], &valid[..], InvocationErrorCode::LengthMismatch),
        (&valid[..], &valid[..3], InvocationErrorCode::LengthMismatch),
        (
            &[f32::NAN, 0.0, 0.0, 0.0][..],
            &valid[..],
            InvocationErrorCode::NonFiniteInput,
        ),
        (
            &valid[..],
            &[0.0, 0.0, f32::INFINITY, 0.0][..],
            InvocationErrorCode::NonFiniteInput,
        ),
    ] {
        assert_eq!(
            VisionRope2dDescriptor {
                tokens: 1,
                heads: 1,
                head_dim: 8,
                cos,
                sin,
            }
            .plan()
            .unwrap_err()
            .code(),
            expected
        );
    }
}

#[test]
fn spatial_vision_rope_cannot_be_attached_to_a_different_layer_geometry() {
    let base_layer = VisionEncoderLayerGeometry {
        tokens: 2,
        hidden_size: 8,
        attention_heads: 1,
        head_dim: 8,
        intermediate_size: 16,
        layer_norm_epsilon: 1.0e-6,
        cu_seqlens: &[0, 2],
    }
    .plan()
    .unwrap();
    let tables = [1.0; 4];
    let mismatched_rope = VisionRope2dDescriptor {
        tokens: 1,
        heads: 1,
        head_dim: 8,
        cos: &tables,
        sin: &tables,
    }
    .plan()
    .unwrap();
    assert_eq!(
        base_layer
            .with_spatial_rope(mismatched_rope)
            .unwrap_err()
            .code(),
        InvocationErrorCode::InvalidVisionGeometry
    );

    let tables = [1.0; 8];
    let rope = VisionRope2dDescriptor {
        tokens: 2,
        heads: 1,
        head_dim: 8,
        cos: &tables,
        sin: &tables,
    }
    .plan()
    .unwrap();
    let layer = base_layer.with_spatial_rope(rope).unwrap();
    assert_eq!(
        layer.stack_plan(0).unwrap_err().code(),
        InvocationErrorCode::ZeroDimension
    );
}
