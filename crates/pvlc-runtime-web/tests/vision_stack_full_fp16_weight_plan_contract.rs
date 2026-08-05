use pvlc_model_schema::{COMPILER_MODEL_ABI, MODEL_ID, MODEL_REVISION};
use pvlc_pack::{
    VISION_STACK_SHARD_MANIFEST_SCHEMA_VERSION, VisionStackShardDescriptor, VisionStackShardKind,
    VisionStackShardManifest, VisionStackShardOracle,
    vision_stack_layer_weight_ranges_with_vector_storage,
};
use pvlc_runtime_core::{
    DecoderWeightStorage, KernelId, LinearWeightLayout, VisionEncoderLayerStage,
};
use pvlc_runtime_web::{
    plan_browser_vision_stack_layer_weights, prepare_browser_vision_stack_execution,
};

fn descriptor(
    id: &str,
    kind: VisionStackShardKind,
    layer_index: Option<u32>,
    bytes: u64,
) -> VisionStackShardDescriptor {
    VisionStackShardDescriptor {
        id: id.to_owned(),
        kind,
        layer_index,
        bytes,
        blake3: "1".repeat(64),
    }
}

fn manifest() -> VisionStackShardManifest {
    let ranges = vision_stack_layer_weight_ranges_with_vector_storage(
        40,
        64,
        DecoderWeightStorage::F16,
        DecoderWeightStorage::F16,
    )
    .unwrap();
    let layer_bytes = ranges.last().unwrap().offset + ranges.last().unwrap().bytes;
    VisionStackShardManifest {
        schema_version: VISION_STACK_SHARD_MANIFEST_SCHEMA_VERSION,
        oracle: VisionStackShardOracle::Synthetic,
        case_id: "synthetic.browser_full_fp16".to_owned(),
        model_id: MODEL_ID.to_owned(),
        model_revision: MODEL_REVISION.to_owned(),
        compiler_model_abi: COMPILER_MODEL_ABI,
        compiler_build: "0".repeat(64),
        golden_bundle_digest: None,
        semantic_fingerprint: None,
        matrix_weight_storage: DecoderWeightStorage::F16,
        matrix_weight_layout: LinearWeightLayout::InputMajor,
        vector_weight_storage: DecoderWeightStorage::F16,
        activation_storage: DecoderWeightStorage::F16,
        tokens: 9,
        hidden_size: 40,
        attention_heads: 5,
        head_dim: 8,
        intermediate_size: 64,
        layer_norm_epsilon: 1.0e-6,
        cu_seqlens: vec![0, 4, 9],
        layer_count: 1,
        checkpoint_layers: vec![],
        shards: vec![
            descriptor("input.embeddings", VisionStackShardKind::Input, None, 720),
            descriptor(
                "weights.vision_layer.00",
                VisionStackShardKind::Layer,
                Some(0),
                layer_bytes,
            ),
            descriptor(
                "weights.vision_post_norm",
                VisionStackShardKind::PostNorm,
                None,
                160,
            ),
        ],
    }
}

#[test]
fn browser_preparation_selects_the_complete_full_fp16_kernel_family() {
    let manifest = manifest();
    let weights = plan_browser_vision_stack_layer_weights(&manifest).unwrap();
    assert_eq!(weights.vector_weight_storage, DecoderWeightStorage::F16);
    assert_eq!(weights.activation_storage, DecoderWeightStorage::F16);
    assert_eq!(weights.projection_kernel, KernelId::LinearProjectionF16);
    assert_eq!(weights.rope_kernel, KernelId::VisionRope2dF16);
    assert!(weights.requires_shader_f16);
    assert_eq!(
        weights.tiled_fp16_qkv_kernel, None,
        "the measured-regressing F32-activation candidate is ABI-incompatible with full FP16",
    );
    assert_eq!(
        weights.ranges.map(|range| range.storage),
        [DecoderWeightStorage::F16; 16],
    );

    let prepared = prepare_browser_vision_stack_execution(
        &manifest,
        pvlc_runtime_core::VisionQkvSelectionOutcome::Disabled,
        true,
    )
    .unwrap();
    assert_eq!(
        prepared
            .layer_plan
            .dispatches
            .map(|dispatch| (dispatch.stage, dispatch.invocation.kernel)),
        [
            (VisionEncoderLayerStage::Norm1, KernelId::LayerNormF16),
            (
                VisionEncoderLayerStage::Query,
                KernelId::LinearProjectionF16
            ),
            (VisionEncoderLayerStage::Key, KernelId::LinearProjectionF16),
            (
                VisionEncoderLayerStage::Value,
                KernelId::LinearProjectionF16
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
                KernelId::LinearProjectionF16
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
        ],
    );
}

#[test]
fn browser_full_fp16_requires_shader_f16_before_gpu_allocation() {
    let error = prepare_browser_vision_stack_execution(
        &manifest(),
        pvlc_runtime_core::VisionQkvSelectionOutcome::Disabled,
        false,
    )
    .unwrap_err();
    assert_eq!(
        error.code(),
        pvlc_runtime_web::BrowserVisionStackWeightPlanErrorCode::MissingShaderF16,
    );
}
