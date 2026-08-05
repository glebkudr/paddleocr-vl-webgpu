use pvlc_runtime_core::{
    ComputeDispatchLimits, InvocationError, InvocationErrorCode, InvocationPlan, KernelId,
    VisionQkvReadbackRequirements, plan_vision_qkv_readback_layout,
};

fn fused_plan(workgroup_size: [u32; 3], dispatch: [u32; 3]) -> InvocationPlan {
    InvocationPlan {
        kernel: KernelId::VisionQkvFusedF32,
        output_elements: 96,
        output_bytes: 384,
        workgroup_size,
        dispatch,
    }
}

fn compute_limits() -> ComputeDispatchLimits {
    ComputeDispatchLimits {
        max_workgroup_size: [8, 8, 1],
        max_invocations_per_workgroup: 64,
        max_workgroups_per_dimension: 65_535,
    }
}

fn assert_error_code<T: std::fmt::Debug>(
    result: Result<T, InvocationError>,
    expected: InvocationErrorCode,
) {
    let error = result.expect_err("invalid preflight input must fail before adapter effects");
    assert_eq!(error.code(), expected, "unexpected error: {error}");
}

fn readback_requirements(
    semantic_readback_bytes: u64,
    scratch_canary_readback_bytes: u64,
    qkv_canary_readback_bytes: u64,
    workspace_allocation_bytes: u64,
    max_buffer_size: u64,
    max_host_elements: u64,
) -> VisionQkvReadbackRequirements {
    VisionQkvReadbackRequirements {
        semantic_readback_bytes,
        scratch_canary_readback_bytes,
        qkv_canary_readback_bytes,
        workspace_allocation_bytes,
        max_buffer_size,
        max_host_elements,
    }
}

#[test]
fn common_compute_limits_accept_exact_boundaries_and_reject_every_axis_before_execution() {
    let limits = compute_limits();
    limits
        .validate(&fused_plan([8, 8, 1], [65_535, 65_535, 65_535]))
        .expect("every exact adapter limit must be accepted");

    for (max_workgroup_size, workgroup_size) in [
        ([8, u32::MAX, u32::MAX], [9, 1, 1]),
        ([u32::MAX, 8, u32::MAX], [1, 9, 1]),
        ([u32::MAX, u32::MAX, 1], [1, 1, 2]),
    ] {
        let axis_limits = ComputeDispatchLimits {
            max_workgroup_size,
            max_invocations_per_workgroup: u32::MAX,
            max_workgroups_per_dimension: u32::MAX,
        };
        assert_error_code(
            axis_limits.validate(&fused_plan(workgroup_size, [1, 1, 1])),
            InvocationErrorCode::InvalidFusionTarget,
        );
    }
    for dispatch in [[65_536, 1, 1], [1, 65_536, 1], [1, 1, 65_536]] {
        assert_error_code(
            limits.validate(&fused_plan([8, 8, 1], dispatch)),
            InvocationErrorCode::InvalidFusionTarget,
        );
    }
    for workgroup_size in [[0, 8, 1], [8, 0, 1], [8, 8, 0]] {
        assert_error_code(
            limits.validate(&fused_plan(workgroup_size, [1, 1, 1])),
            InvocationErrorCode::InvalidFusionTarget,
        );
    }
    for dispatch in [[0, 1, 1], [1, 0, 1], [1, 1, 0]] {
        assert_error_code(
            limits.validate(&fused_plan([8, 8, 1], dispatch)),
            InvocationErrorCode::InvalidFusionTarget,
        );
    }
}

#[test]
fn common_compute_limits_check_invocation_product_without_wrapping() {
    let mut too_few_invocations = compute_limits();
    too_few_invocations.max_invocations_per_workgroup = 63;
    assert_error_code(
        too_few_invocations.validate(&fused_plan([8, 8, 1], [1, 1, 1])),
        InvocationErrorCode::InvalidFusionTarget,
    );

    let unbounded_axes = ComputeDispatchLimits {
        max_workgroup_size: [u32::MAX; 3],
        max_invocations_per_workgroup: u32::MAX,
        max_workgroups_per_dimension: u32::MAX,
    };
    assert_error_code(
        unbounded_axes.validate(&fused_plan([u32::MAX, u32::MAX, 2], [1, 1, 1])),
        InvocationErrorCode::ArithmeticOverflow,
    );
}

#[test]
fn common_readback_plan_freezes_semantic_scratch_and_qkv_layout_and_host_sizes() {
    let layout = plan_vision_qkv_readback_layout(readback_requirements(16, 8, 12, 40, 40, 10))
        .expect("exact buffer and host limits must be accepted");

    assert_eq!(layout.semantic_offset(), 0);
    assert_eq!(layout.semantic_readback_bytes(), 16);
    assert_eq!(layout.scratch_canary_offset(), 16);
    assert_eq!(layout.scratch_canary_readback_bytes(), 8);
    assert_eq!(layout.qkv_canary_offset(), 24);
    assert_eq!(layout.qkv_canary_readback_bytes(), 12);
    assert_eq!(layout.total_readback_bytes(), 36);
    assert_eq!(layout.workspace_allocation_bytes(), 40);
    assert_eq!(layout.readback_f32_elements(), 9);
    assert_eq!(layout.workspace_u32_words(), 10);
}

#[test]
fn common_readback_plan_rejects_each_checked_add_overflow_with_a_stable_code() {
    assert_error_code(
        plan_vision_qkv_readback_layout(readback_requirements(
            u64::MAX - 3,
            4,
            0,
            4,
            u64::MAX,
            u64::MAX,
        )),
        InvocationErrorCode::ArithmeticOverflow,
    );
    assert_error_code(
        plan_vision_qkv_readback_layout(readback_requirements(
            4,
            u64::MAX - 7,
            4,
            4,
            u64::MAX,
            u64::MAX,
        )),
        InvocationErrorCode::ArithmeticOverflow,
    );
}

#[test]
fn common_readback_plan_enforces_exact_buffer_alignment_and_host_boundaries() {
    plan_vision_qkv_readback_layout(readback_requirements(8, 4, 4, 16, 16, 4))
        .expect("exact readback, workspace, and host boundaries must be accepted");

    for requirements in [
        readback_requirements(8, 4, 4, 12, 15, 4),
        readback_requirements(4, 4, 4, 20, 16, 5),
        readback_requirements(8, 4, 4, 12, 16, 3),
        readback_requirements(4, 4, 4, 16, 16, 3),
        readback_requirements(6, 4, 4, 16, 32, 8),
        readback_requirements(8, 2, 4, 16, 32, 8),
        readback_requirements(8, 4, 2, 16, 32, 8),
        readback_requirements(8, 4, 4, 14, 32, 8),
    ] {
        assert_error_code(
            plan_vision_qkv_readback_layout(requirements),
            InvocationErrorCode::InvalidFusionTarget,
        );
    }

    let wasm_sized_host_limit = u64::from(u32::MAX);
    let too_many_host_elements = wasm_sized_host_limit + 1;
    assert_error_code(
        plan_vision_qkv_readback_layout(readback_requirements(
            too_many_host_elements * 4,
            0,
            0,
            4,
            u64::MAX,
            wasm_sized_host_limit,
        )),
        InvocationErrorCode::InvalidFusionTarget,
    );
}
