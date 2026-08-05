use pvlc_runtime_core::{
    KernelId, VisionQkvFusedTargetLimits, plan_vision_qkv_fused_f16_weight_geometry,
    plan_vision_qkv_fused_geometry,
};

fn target(alignment: u32) -> VisionQkvFusedTargetLimits {
    VisionQkvFusedTargetLimits {
        max_storage_buffer_binding_size: 1 << 30,
        max_buffer_size: 1 << 30,
        max_compute_workgroups_per_dimension: 65_535,
        max_storage_buffers_per_shader_stage: 8,
        min_storage_buffer_offset_alignment: alignment,
    }
}

#[test]
fn tiled_fp16_qkv_plans_one_dispatch_plane_and_three_aligned_output_slices() {
    let tokens = 1_836;
    let width = 1_152;
    let plan =
        plan_vision_qkv_fused_f16_weight_geometry(tokens, width, width, target(256)).unwrap();
    let hidden_bytes = u64::from(tokens) * u64::from(width) * 4;

    assert_eq!(plan.invocation.kernel, KernelId::VisionQkvFusedF16Weights);
    assert_eq!(plan.invocation.workgroup_size, [8, 8, 1]);
    assert_eq!(plan.invocation.dispatch, [36, 115, 1]);
    assert_eq!(
        plan.uniform_words,
        [tokens, width, width, (hidden_bytes / 4) as u32]
    );
    assert_eq!(plan.output_layout.plane_bytes, hidden_bytes);
    assert_eq!(plan.output_layout.plane_stride_bytes, hidden_bytes);
    assert_eq!(plan.output_layout.query.offset, 0);
    assert_eq!(plan.output_layout.key.offset, hidden_bytes);
    assert_eq!(plan.output_layout.value.offset, hidden_bytes * 2);
    assert_eq!(plan.output_layout.physical_bytes, hidden_bytes * 3);
}

#[test]
fn tiled_fp16_qkv_handles_mixed_row_column_depth_tails_and_alignment_padding() {
    let plan = plan_vision_qkv_fused_f16_weight_geometry(33, 36, 68, target(256)).unwrap();
    let plane_bytes = 33_u64 * 68 * 4;
    let plane_stride = plane_bytes.next_multiple_of(256);

    assert_eq!(plan.invocation.dispatch, [3, 3, 1]);
    assert_eq!(plan.output_layout.plane_bytes, plane_bytes);
    assert_eq!(plan.output_layout.plane_stride_bytes, plane_stride);
    assert_eq!(plan.output_layout.query.offset, 0);
    assert_eq!(plan.output_layout.key.offset, plane_stride);
    assert_eq!(plan.output_layout.value.offset, plane_stride * 2);
    assert_eq!(plan.uniform_words[3], (plane_stride / 4) as u32);
}

#[test]
fn tiled_fp16_qkv_rejects_unpacked_widths_and_keeps_the_f32_plan_unchanged() {
    assert!(
        plan_vision_qkv_fused_f16_weight_geometry(33, 35, 68, target(256)).is_err(),
        "packed vec4 input requires an input width divisible by four",
    );
    assert!(
        plan_vision_qkv_fused_f16_weight_geometry(33, 36, 67, target(256)).is_err(),
        "packed vec4 weights and biases require an output width divisible by four",
    );

    let f32 = plan_vision_qkv_fused_geometry(33, 36, 68, target(256)).unwrap();
    assert_eq!(f32.invocation.kernel, KernelId::VisionQkvFusedF32);
    assert_eq!(f32.invocation.dispatch, [9, 5, 3]);
}

#[test]
fn tiled_fp16_qkv_requires_only_the_portable_eight_storage_bindings() {
    let mut insufficient = target(256);
    insufficient.max_storage_buffers_per_shader_stage = 7;
    assert!(plan_vision_qkv_fused_f16_weight_geometry(33, 36, 68, insufficient).is_err());
    assert!(plan_vision_qkv_fused_f16_weight_geometry(33, 36, 68, target(256)).is_ok());
}
