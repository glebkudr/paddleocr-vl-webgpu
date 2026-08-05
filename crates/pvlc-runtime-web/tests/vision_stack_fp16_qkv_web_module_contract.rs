//! Browser adapter contract for the automatic tiled FP16-weight QKV path.
//!
//! Numerical shader/layout semantics are owned by the pvlc-wgsl contract.
//! This gate pins the wasm-only resource lifetime and direct GPU dataflow.

const WEB_RUNTIME: &str = include_str!("../src/web.rs");

fn source_between(start: &str, end: &str) -> &'static str {
    between(WEB_RUNTIME, start, end)
}

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
fn fp16_qkv_workspace_is_allocated_once_per_stack_and_not_once_per_layer() {
    let state = source_between(
        "struct BrowserVisionStackGpuState",
        "enum BrowserVisionStackScratch",
    );
    assert!(state.contains("fp16_qkv_workspace: Option<wgpu::Buffer>"));

    let allocation = source_between(
        "fn allocate_vision_stack_gpu",
        "fn enqueue_vision_stack_sharded_layer",
    );
    assert_eq!(
        allocation
            .matches("\"vision-stack-fp16-qkv-workspace\"")
            .count(),
        1,
    );
    assert!(allocation.contains("session.fp16_qkv_plan.as_ref()"));
    assert!(allocation.contains("output_layout.physical_bytes"));

    let layer = source_between(
        "fn encode_and_submit_vision_stack_layer",
        "async fn finish_vision_stack_sharded",
    );
    assert!(
        !layer.contains("create_runtime_buffer("),
        "a layer must reuse the stack-owned QKV workspace",
    );
}

#[test]
fn fp16_qkv_layer_uses_one_dispatch_and_feeds_workspace_slices_directly_to_rope_and_attention() {
    let router = source_between(
        "fn encode_and_submit_vision_stack_layer",
        "fn encode_and_submit_vision_stack_fp16_qkv_layer",
    );
    assert!(router.contains("session.fp16_qkv_plan.is_some()"));
    assert!(router.contains("encode_and_submit_vision_stack_fp16_qkv_layer"));

    let layer = source_between(
        "fn encode_and_submit_vision_stack_fp16_qkv_layer",
        "async fn finish_vision_stack_sharded",
    );
    for required in [
        "KernelId::VisionQkvFusedF16Weights",
        "create_vision_stack_fp16_qkv_bind_group",
        "vision_stack_fp16_qkv_workspace_bindings",
        "weight_bindings[2]",
        "weight_bindings[3]",
        "weight_bindings[4]",
        "weight_bindings[5]",
        "weight_bindings[6]",
        "weight_bindings[7]",
        "fp16_qkv_workspace",
        "qkv_plan.invocation.dispatch",
    ] {
        assert!(layer.contains(required), "missing {required}");
    }
    assert!(
        layer.contains("let [query, key, value] ="),
        "the three plane slices must remain typed GPU bindings",
    );
    assert!(
        layer.contains("&[query, key, value, boundary]"),
        "attention must consume Q/K/V workspace slices without copies",
    );
    assert!(
        compact(layer).contains(
            "create_vision_stack_rope_bind_group(spatial_plan,gpu,rope_kernel,query,key)",
        ),
        "RoPE must consume query/key workspace slices without copies",
    );
    assert_eq!(
        layer.matches("KernelId::VisionQkvFusedF16Weights").count(),
        1,
        "the specialized method must encode exactly one fused QKV dispatch",
    );
    let fused_dispatch = between(
        layer,
        "pass.set_pipeline(&gpu.pipelines[&KernelId::VisionQkvFusedF16Weights]);",
        "if let (Some(spatial_plan), Some(bind_group))",
    );
    assert_eq!(fused_dispatch.matches("pass.set_bind_group(").count(), 1);
    assert_eq!(
        fused_dispatch.matches("pass.dispatch_workgroups(").count(),
        1
    );
    assert_eq!(
        fused_dispatch
            .matches("qkv_plan.invocation.dispatch[")
            .count(),
        3,
        "the single fused dispatch must take all three dimensions from the sealed planner",
    );
    for rejected in [
        "layer_plan.dispatches[1]",
        "layer_plan.dispatches[2]",
        "layer_plan.dispatches[3]",
        "for index in 0..4",
    ] {
        assert!(
            !layer.contains(rejected),
            "the specialized method retained legacy Q/K/V dispatch authority: {rejected}",
        );
    }
    assert!(
        !layer.contains("copy_buffer_to_buffer(\n                        fp16_qkv_workspace"),
        "Q/K/V may not round-trip through scratch copies",
    );
}

#[test]
fn specialized_kernel_is_admitted_only_for_fp16_input_major_sessions() {
    let sources = source_between(
        "fn vision_stack_shader_sources",
        "fn vision_qkv_stack_shader_sources",
    );
    assert!(sources.contains("KernelId::VisionQkvFusedF16Weights"));
    assert!(sources.contains("LinearWeightLayout::InputMajor"));
    assert!(sources.contains("DecoderWeightStorage::F16"));

    let preparation = source_between(
        "fn prepare_browser_stack",
        "fn preflight_vision_stack_shard",
    );
    assert!(
        compact(preparation).contains("weight_plan.tiled_fp16_qkv_kernel"),
        "shader admission must come from the authenticated weight plan",
    );
    assert!(
        preparation.contains("plan_vision_qkv_fused_f16_weight_geometry"),
        "the browser must use the adapter-neutral dispatch/workspace planner",
    );
}

#[test]
fn measured_regression_keeps_the_fused_candidate_out_of_the_default_path() {
    assert!(
        WEB_RUNTIME.contains("const ENABLE_TILED_FP16_QKV: bool = false;"),
        "the 0.925-0.965 s fused candidates must not replace the 0.887 s default path",
    );
    let preparation = compact(source_between(
        "fn prepare_browser_stack",
        "fn preflight_vision_stack_shard",
    ));
    assert!(
        preparation.contains("weight_plan.tiled_fp16_qkv_kernel.filter(|_|ENABLE_TILED_FP16_QKV)"),
        "runtime selection must pass through the measured performance gate",
    );
}
