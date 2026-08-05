use pvlc_model_schema::{COMPILER_MODEL_ABI, MODEL_ID, MODEL_REVISION};
use pvlc_pack::{
    VISION_STACK_SHARD_MANIFEST_SCHEMA_VERSION, VisionStackShardDescriptor, VisionStackShardKind,
    VisionStackShardManifest, VisionStackShardOracle,
};
use pvlc_runtime_core::{
    DecoderWeightStorage, KernelId, LinearWeightLayout, VisionEncoderLayerStage,
    VisionQkvSelectionOutcome,
};
use pvlc_runtime_web::{
    BrowserVisionStackWeightPlanErrorCode, plan_browser_vision_stack_layer_weights,
    prepare_browser_vision_stack_execution, vision_stack_resident_weight_key,
};

fn manifest(matrix_weight_storage: DecoderWeightStorage) -> VisionStackShardManifest {
    manifest_with_layout(matrix_weight_storage, LinearWeightLayout::OutputMajor)
}

fn manifest_with_layout(
    matrix_weight_storage: DecoderWeightStorage,
    matrix_weight_layout: LinearWeightLayout,
) -> VisionStackShardManifest {
    let layer_bytes = match matrix_weight_storage {
        DecoderWeightStorage::F32 => 688,
        DecoderWeightStorage::F16 => 432,
    };
    VisionStackShardManifest {
        schema_version: VISION_STACK_SHARD_MANIFEST_SCHEMA_VERSION,
        oracle: VisionStackShardOracle::Synthetic,
        case_id: "synthetic.browser_fp16_weight_plan".to_owned(),
        model_id: MODEL_ID.to_owned(),
        model_revision: MODEL_REVISION.to_owned(),
        compiler_model_abi: COMPILER_MODEL_ABI,
        compiler_build: "0".repeat(64),
        golden_bundle_digest: None,
        semantic_fingerprint: None,
        matrix_weight_storage,
        matrix_weight_layout,
        vector_weight_storage: DecoderWeightStorage::F32,
        activation_storage: DecoderWeightStorage::F32,
        tokens: 2,
        hidden_size: 4,
        attention_heads: 1,
        head_dim: 4,
        intermediate_size: 8,
        layer_norm_epsilon: 1.0e-6,
        cu_seqlens: vec![0, 2],
        layer_count: 1,
        checkpoint_layers: vec![],
        shards: vec![
            shard("input.embeddings", VisionStackShardKind::Input, None, 32),
            shard(
                "weights.vision_layer.00",
                VisionStackShardKind::Layer,
                Some(0),
                layer_bytes,
            ),
            shard(
                "weights.vision_post_norm",
                VisionStackShardKind::PostNorm,
                None,
                32,
            ),
        ],
    }
}

fn shard(
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
        blake3: "0".repeat(64),
    }
}

fn resident_manifest() -> VisionStackShardManifest {
    let mut manifest =
        manifest_with_layout(DecoderWeightStorage::F16, LinearWeightLayout::InputMajor);
    manifest.layer_count = 27;
    manifest.shards = std::iter::once(shard(
        "input.embeddings",
        VisionStackShardKind::Input,
        None,
        32,
    ))
    .chain((0..27).map(|layer| {
        let mut descriptor = shard(
            &format!("weights.vision_layer.{layer:02}"),
            VisionStackShardKind::Layer,
            Some(layer),
            432,
        );
        descriptor.blake3 = format!("{layer:02x}{}", "0".repeat(62));
        descriptor
    }))
    .chain(std::iter::once(shard(
        "weights.vision_post_norm",
        VisionStackShardKind::PostNorm,
        None,
        32,
    )))
    .collect();
    manifest
}

#[test]
fn resident_weight_identity_binds_every_model_and_authenticated_layout_input() {
    let manifest = resident_manifest();
    let baseline = vision_stack_resident_weight_key(&manifest).unwrap();
    assert_eq!(
        baseline,
        vision_stack_resident_weight_key(&manifest.clone()).unwrap(),
    );

    let mut changed_digest = manifest.clone();
    changed_digest.shards[14].blake3 = "f".repeat(64);
    assert_ne!(
        baseline,
        vision_stack_resident_weight_key(&changed_digest).unwrap(),
        "one authenticated layer digest must invalidate resident reuse",
    );

    let mut changed_layout = manifest.clone();
    changed_layout.matrix_weight_layout = LinearWeightLayout::OutputMajor;
    assert_ne!(
        baseline,
        vision_stack_resident_weight_key(&changed_layout).unwrap(),
        "matrix layout must be part of the resident identity",
    );

    let one_layer = manifest_with_layout(DecoderWeightStorage::F16, LinearWeightLayout::InputMajor);
    assert_ne!(
        baseline,
        vision_stack_resident_weight_key(&one_layer).unwrap(),
        "layer count and complete shard directory must be part of the resident identity",
    );

    let mut changed_build = manifest.clone();
    changed_build.compiler_build = "1".repeat(64);
    assert_ne!(
        baseline,
        vision_stack_resident_weight_key(&changed_build).unwrap(),
        "compiler/model artifact identity must invalidate resident reuse",
    );

    let mut changed_input = manifest.clone();
    changed_input.case_id = "another-image/same-weights".to_owned();
    changed_input.tokens = 3;
    changed_input.cu_seqlens = vec![0, 3];
    changed_input.shards[0].bytes = 48;
    changed_input.shards[0].blake3 = "e".repeat(64);
    assert_eq!(
        baseline,
        vision_stack_resident_weight_key(&changed_input).unwrap(),
        "image identity, token geometry, and input digest must not invalidate identical weights",
    );

    let mut wrong_revision = manifest;
    wrong_revision.model_revision = "not-the-admitted-model-revision".to_owned();
    assert!(
        vision_stack_resident_weight_key(&wrong_revision).is_err(),
        "an inadmissible model revision must never receive a reusable cache key",
    );
}

#[test]
fn browser_preparation_propagates_input_major_layout_to_projection_uniforms() {
    let input_major_manifest =
        manifest_with_layout(DecoderWeightStorage::F16, LinearWeightLayout::InputMajor);
    let prepared = prepare_browser_vision_stack_execution(
        &input_major_manifest,
        VisionQkvSelectionOutcome::Disabled,
        true,
    )
    .unwrap();
    let output_major_manifest = manifest(DecoderWeightStorage::F16);
    let output_major = prepare_browser_vision_stack_execution(
        &output_major_manifest,
        VisionQkvSelectionOutcome::Disabled,
        true,
    )
    .unwrap();

    assert_eq!(
        prepared.weights.matrix_weight_layout,
        LinearWeightLayout::InputMajor,
    );
    for (input_dispatch, output_dispatch) in prepared
        .layer_plan
        .dispatches
        .iter()
        .zip(output_major.layer_plan.dispatches)
    {
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
            assert_eq!(input_dispatch.uniform_words[3], 1);
        } else {
            assert_eq!(input_dispatch.uniform_words, output_dispatch.uniform_words);
        }
    }
}

#[test]
fn browser_weight_plan_propagates_manifest_storage_into_exact_gpu_bindings() {
    let f16 =
        plan_browser_vision_stack_layer_weights(&manifest(DecoderWeightStorage::F16)).unwrap();
    let f16_input_major = plan_browser_vision_stack_layer_weights(&manifest_with_layout(
        DecoderWeightStorage::F16,
        LinearWeightLayout::InputMajor,
    ))
    .unwrap();
    let f32 =
        plan_browser_vision_stack_layer_weights(&manifest(DecoderWeightStorage::F32)).unwrap();
    let f32_input_major = plan_browser_vision_stack_layer_weights(&manifest_with_layout(
        DecoderWeightStorage::F32,
        LinearWeightLayout::InputMajor,
    ));

    assert_eq!(f16.matrix_weight_storage, DecoderWeightStorage::F16);
    assert_eq!(f16.projection_kernel, KernelId::LinearProjectionF16Weights);
    assert!(f16.requires_shader_f16);
    assert!(!f16.fused_qkv_supported);
    assert_eq!(
        f16.tiled_fp16_qkv_kernel, None,
        "the tiled fused kernel accepts only offline-transposed input-major matrices",
    );
    assert_eq!(
        f16_input_major.tiled_fp16_qkv_kernel,
        Some(KernelId::VisionQkvFusedF16Weights),
        "the browser FP16 checkpoint must select its dedicated tiled QKV kernel",
    );
    assert_eq!(
        f16.ranges
            .map(|range| (range.offset, range.bytes, range.storage)),
        [
            (0, 16, DecoderWeightStorage::F32),
            (16, 16, DecoderWeightStorage::F32),
            (32, 32, DecoderWeightStorage::F16),
            (64, 16, DecoderWeightStorage::F32),
            (80, 32, DecoderWeightStorage::F16),
            (112, 16, DecoderWeightStorage::F32),
            (128, 32, DecoderWeightStorage::F16),
            (160, 16, DecoderWeightStorage::F32),
            (176, 32, DecoderWeightStorage::F16),
            (208, 16, DecoderWeightStorage::F32),
            (224, 16, DecoderWeightStorage::F32),
            (240, 16, DecoderWeightStorage::F32),
            (256, 64, DecoderWeightStorage::F16),
            (320, 32, DecoderWeightStorage::F32),
            (352, 64, DecoderWeightStorage::F16),
            (416, 16, DecoderWeightStorage::F32),
        ],
    );

    assert_eq!(f32.matrix_weight_storage, DecoderWeightStorage::F32);
    assert_eq!(f32.projection_kernel, KernelId::VisionPatchProjectionF32);
    assert!(!f32.requires_shader_f16);
    assert!(f32.fused_qkv_supported);
    assert_eq!(f32.tiled_fp16_qkv_kernel, None);
    assert!(
        f32_input_major.is_err(),
        "the manifest schema must reject F32 input-major matrices before kernel selection",
    );
    assert_eq!(
        f32.ranges
            .map(|range| (range.offset, range.bytes, range.storage)),
        [
            (0, 16, DecoderWeightStorage::F32),
            (16, 16, DecoderWeightStorage::F32),
            (32, 64, DecoderWeightStorage::F32),
            (96, 16, DecoderWeightStorage::F32),
            (112, 64, DecoderWeightStorage::F32),
            (176, 16, DecoderWeightStorage::F32),
            (192, 64, DecoderWeightStorage::F32),
            (256, 16, DecoderWeightStorage::F32),
            (272, 64, DecoderWeightStorage::F32),
            (336, 16, DecoderWeightStorage::F32),
            (352, 16, DecoderWeightStorage::F32),
            (368, 16, DecoderWeightStorage::F32),
            (384, 128, DecoderWeightStorage::F32),
            (512, 32, DecoderWeightStorage::F32),
            (544, 128, DecoderWeightStorage::F32),
            (672, 16, DecoderWeightStorage::F32),
        ],
    );
}

#[test]
fn browser_rejects_missing_shader_f16_and_f32_fused_qkv_for_fp16_weights() {
    let f16 =
        plan_browser_vision_stack_layer_weights(&manifest(DecoderWeightStorage::F16)).unwrap();
    let f32 =
        plan_browser_vision_stack_layer_weights(&manifest(DecoderWeightStorage::F32)).unwrap();

    assert!(f16.validate_capabilities(true).is_ok());
    assert_eq!(
        f16.validate_capabilities(false).unwrap_err().code(),
        BrowserVisionStackWeightPlanErrorCode::MissingShaderF16,
    );
    assert!(
        f16.validate_qkv_outcome(VisionQkvSelectionOutcome::Disabled)
            .is_ok()
    );
    assert_eq!(
        f16.validate_qkv_outcome(VisionQkvSelectionOutcome::Fused)
            .unwrap_err()
            .code(),
        BrowserVisionStackWeightPlanErrorCode::UnsupportedFusedQkv,
    );
    assert!(f32.validate_capabilities(false).is_ok());
    assert!(
        f32.validate_qkv_outcome(VisionQkvSelectionOutcome::Fused)
            .is_ok()
    );
}

#[test]
fn callable_browser_execution_preparation_composes_manifest_storage_with_layer_planning() {
    let f16_manifest = manifest(DecoderWeightStorage::F16);
    let prepared = prepare_browser_vision_stack_execution(
        &f16_manifest,
        VisionQkvSelectionOutcome::Disabled,
        true,
    )
    .unwrap();

    assert_eq!(
        prepared.weights.matrix_weight_storage,
        DecoderWeightStorage::F16,
    );
    assert_eq!(
        prepared.weights,
        plan_browser_vision_stack_layer_weights(&f16_manifest).unwrap(),
        "callable browser preparation must preserve the complete reviewed binding plan",
    );
    assert_eq!(
        prepared.layer_plan.dispatches[1].invocation.kernel,
        KernelId::LinearProjectionF16Weights,
    );
    assert_eq!(
        prepared.weights.ranges[2].bytes, 32,
        "the browser preparation must bind the F16 query matrix at half the F32 byte width",
    );

    assert_eq!(
        prepare_browser_vision_stack_execution(
            &f16_manifest,
            VisionQkvSelectionOutcome::Fused,
            true,
        )
        .unwrap_err()
        .code(),
        BrowserVisionStackWeightPlanErrorCode::UnsupportedFusedQkv,
    );
    assert_eq!(
        prepare_browser_vision_stack_execution(
            &f16_manifest,
            VisionQkvSelectionOutcome::Disabled,
            false,
        )
        .unwrap_err()
        .code(),
        BrowserVisionStackWeightPlanErrorCode::MissingShaderF16,
    );
}
