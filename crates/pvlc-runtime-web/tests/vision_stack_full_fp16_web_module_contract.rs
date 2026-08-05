//! Production wiring contract for the browser's full-FP16 vision path.
//!
//! Planner and WGSL tests own geometry and math. This gate proves that the
//! wasm adapter allocates those byte sizes, builds the full kernel catalog,
//! binds each real stage, applies FP16 spatial RoPE, and reads back the FP16
//! post-norm result without routing through an F32-only special case.

const WEB_RUNTIME: &str = include_str!("../src/web.rs");

fn between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    source
        .split_once(start)
        .and_then(|(_, tail)| tail.split_once(end).map(|(body, _)| body))
        .unwrap_or_else(|| panic!("missing source section {start:?} .. {end:?}"))
}

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

#[test]
fn authenticated_activation_storage_selects_the_complete_production_shader_family() {
    let preparation = between(
        WEB_RUNTIME,
        "fn prepare_browser_stack",
        "fn preflight_vision_stack_shard",
    );
    let compact_preparation = compact(preparation);
    assert!(
        compact_preparation.contains("weight_plan.activation_storage"),
        "shader admission must use the authenticated precision field",
    );
    assert!(
        compact_preparation.contains("weight_plan.rope_kernel"),
        "spatial RoPE admission must come from the same weight/activation plan",
    );

    let sources = between(
        WEB_RUNTIME,
        "fn vision_stack_shader_sources",
        "fn vision_qkv_stack_shader_sources",
    );
    let precision_match = between(
        sources,
        "let kernels = match activation_storage {",
        "};\n    kernels",
    );
    let fp16_arm = between(
        precision_match,
        "DecoderWeightStorage::F16 => vec![",
        "DecoderWeightStorage::F32 => vec![",
    );
    for kernel in [
        "KernelId::LayerNormF16",
        "KernelId::LinearProjectionF16",
        "KernelId::VisionAttentionF16",
        "KernelId::AddF16",
        "KernelId::GeluTanhF16",
        "KernelId::VisionRope2dF16",
    ] {
        assert!(
            fp16_arm.contains(kernel),
            "the activation_storage == F16 arm omits {kernel}",
        );
    }
    for forbidden in [
        "KernelId::LayerNormF32",
        "KernelId::LinearProjectionF16Weights",
        "KernelId::VisionAttentionF32",
        "KernelId::AddF32",
        "KernelId::GeluTanhF32",
        "KernelId::VisionRope2dF32",
    ] {
        assert!(
            !fp16_arm.contains(forbidden),
            "the activation_storage == F16 arm retained {forbidden}",
        );
    }
    let mixed_arm = precision_match
        .split_once("DecoderWeightStorage::F32 => vec![")
        .map(|(_, arm)| arm)
        .expect("mixed-precision shader arm is missing");
    for kernel in [
        "KernelId::LayerNormF32",
        "projection_kernel",
        "KernelId::VisionAttentionF32",
        "KernelId::AddF32",
        "KernelId::GeluTanhF32",
    ] {
        assert!(
            mixed_arm.contains(kernel),
            "the existing F32-activation arm lost {kernel}",
        );
    }
    for forbidden in [
        "KernelId::LayerNormF16",
        "KernelId::LinearProjectionF16,",
        "KernelId::VisionAttentionF16",
        "KernelId::AddF16",
        "KernelId::GeluTanhF16",
    ] {
        assert!(
            !mixed_arm.contains(forbidden),
            "the legacy mixed path accidentally selects {forbidden}",
        );
    }
    assert!(
        precision_match.contains("DecoderWeightStorage::F16 => vec![")
            && precision_match.contains("DecoderWeightStorage::F32 => vec!["),
        "shader selection must exhaustively branch on authenticated activation storage",
    );
}

#[test]
fn gpu_allocation_and_readback_take_sizes_from_the_full_fp16_plan() {
    let allocation = between(
        WEB_RUNTIME,
        "fn allocate_vision_stack_gpu",
        "fn enqueue_vision_stack_sharded_layer",
    );
    assert!(allocation.contains("let hidden_bytes = session.plan.hidden_bytes;"));
    assert!(
        allocation.contains("dispatch.invocation.output_bytes"),
        "scratch buffers must use each F16 dispatch's two-byte output size",
    );
    assert!(
        allocation.contains("session.plan.readback_bytes"),
        "checkpoint allocation must use the F16 plan instead of an F32 formula",
    );
    assert!(
        allocation.contains("bytes: hidden_bytes"),
        "the authenticated F16 input upload must be bounded by the planned input bytes",
    );

    let layer = between(
        WEB_RUNTIME,
        "fn encode_and_submit_vision_stack_layer",
        "async fn finish_vision_stack_sharded",
    );
    assert!(
        layer.contains("session.plan.hidden_bytes"),
        "layer checkpoints must copy the planned F16 plane size",
    );
    assert!(
        !layer.contains("checked_mul(4)"),
        "the execution loop must not reconstruct an F32 activation size",
    );

    let finish = between(
        WEB_RUNTIME,
        "async fn execute_vision_stack_post_norm",
        "fn create_vision_stack_bind_group",
    );
    assert!(finish.contains("let hidden_bytes = session.plan.hidden_bytes;"));
    assert!(finish.contains("let dispatch = layer_plan.dispatches[0];"));
    assert!(finish.contains("gpu.pipelines[&dispatch.invocation.kernel]"));
    assert!(
        !finish.contains("KernelId::LayerNormF32"),
        "post-norm must use the authenticated full-FP16 layer-norm kernel",
    );
}

#[test]
fn every_real_layer_stage_uses_planned_kernels_and_fp16_rope_without_cpu_copies() {
    let layer = between(
        WEB_RUNTIME,
        "fn encode_and_submit_vision_stack_layer",
        "fn encode_and_submit_vision_stack_fp16_qkv_layer",
    );
    let disabled = between(
        layer,
        "VisionQkvSelectionOutcome::Disabled",
        "gpu.current_main = next_index;",
    );
    assert_eq!(
        disabled
            .matches("self.create_vision_stack_bind_group(")
            .count(),
        12,
        "all 12 semantic stages need direct GPU bind groups",
    );
    let compact_disabled = compact(disabled);
    for index in 0..12 {
        assert!(
            compact_disabled.contains(&format!("&layer_plan,gpu,{index},")),
            "stage {index} is not bound through its planned ABI",
        );
    }
    assert!(
        disabled.contains("for index in 0..4 {"),
        "norm plus Q/K/V must dispatch exactly stages 0..4",
    );
    assert!(
        disabled.contains("for index in 4..layer_plan.dispatches.len() {"),
        "attention through output must dispatch every remaining planned stage",
    );
    assert_eq!(
        disabled
            .matches("let dispatch = layer_plan.dispatches[index];")
            .count(),
        2,
        "both exact loop partitions must obtain their kernel and geometry from the planner",
    );
    assert_eq!(
        disabled
            .matches("gpu.pipelines[&dispatch.invocation.kernel]")
            .count(),
        2,
        "both loop partitions must bind planned pipelines",
    );
    assert!(
        compact(layer).contains("letrope_kernel=session.weight_plan.rope_kernel;")
            && compact(layer).contains("gpu.pipelines[&rope_kernel]"),
        "spatial RoPE must use VisionRope2dF16 for full-FP16 sessions",
    );
    assert!(
        !layer.contains("KernelId::VisionRope2dF32"),
        "the common layer encoder must not hard-code an F32 RoPE pipeline",
    );
    let stage_execution = disabled
        .split_once("if let Some(slot) = checkpoint_slot")
        .map(|(execution, _)| execution)
        .expect("checkpoint boundary is missing from the layer encoder");
    for forbidden in [
        "copy_buffer_to_buffer",
        "map_async",
        "get_mapped_range",
        "readback",
        "queue.write_buffer",
        "upload",
        "to_vec",
        "await_queue_completion",
    ] {
        assert!(
            !stage_execution.contains(forbidden),
            "the 12-stage GPU path round-trips an intermediate through {forbidden}",
        );
    }
}

#[test]
fn spatial_rope_configuration_and_bind_group_layout_follow_the_authenticated_kernel() {
    let configuration = between(
        WEB_RUNTIME,
        "fn configure_vision_stack_spatial_rope",
        "fn prepare_browser_stack",
    );
    assert!(
        compact(configuration).contains("letrope_kernel=session.weight_plan.rope_kernel;"),
        "RoPE module admission must use the authenticated precision plan",
    );
    assert!(configuration.contains("pvlc_wgsl::module(rope_kernel)"));
    assert!(
        compact(configuration)
            .contains("session.shader_sources.insert(rope_kernel,module.source.to_owned())"),
    );
    assert!(!configuration.contains("KernelId::VisionRope2dF32"));

    let bind_group = between(
        WEB_RUNTIME,
        "fn create_vision_stack_rope_bind_group",
        "fn validate_vision_stack_capabilities",
    );
    assert!(
        compact(bind_group).contains("get(&rope_kernel)"),
        "the real RoPE bind group must use the same authenticated kernel",
    );
    assert!(!bind_group.contains("KernelId::VisionRope2dF32"));
}
