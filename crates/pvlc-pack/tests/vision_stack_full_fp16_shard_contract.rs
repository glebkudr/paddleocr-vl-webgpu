use pvlc_model_schema::{COMPILER_MODEL_ABI, MODEL_ID, MODEL_REVISION};
use pvlc_pack::{
    VISION_STACK_SHARD_MANIFEST_SCHEMA_VERSION, VisionStackLayerTensorRange,
    VisionStackShardDescriptor, VisionStackShardErrorCode, VisionStackShardKind,
    VisionStackShardManifest, VisionStackShardOracle, canonical_vision_stack_shard_manifest_bytes,
    inspect_vision_stack_shard, parse_vision_stack_shard_manifest,
    vision_stack_layer_weight_ranges_with_vector_storage,
};
use pvlc_runtime_core::{DecoderWeightStorage, LinearWeightLayout};

const TOKENS: u32 = 2;
const HIDDEN: u32 = 4;
const INTERMEDIATE: u32 = 8;

fn full_fp16_ranges() -> [VisionStackLayerTensorRange; 16] {
    vision_stack_layer_weight_ranges_with_vector_storage(
        HIDDEN,
        INTERMEDIATE,
        DecoderWeightStorage::F16,
        DecoderWeightStorage::F16,
    )
    .unwrap()
}

fn finite_f16(bytes: usize) -> Vec<u8> {
    (0..bytes / 2)
        .flat_map(|_| 0x3c00_u16.to_le_bytes())
        .collect()
}

fn descriptor(
    id: &str,
    kind: VisionStackShardKind,
    layer_index: Option<u32>,
    payload: &[u8],
) -> VisionStackShardDescriptor {
    VisionStackShardDescriptor {
        id: id.to_owned(),
        kind,
        layer_index,
        bytes: payload.len() as u64,
        blake3: blake3::hash(payload).to_hex().to_string(),
    }
}

fn manifest(
    matrix_weight_storage: DecoderWeightStorage,
    vector_weight_storage: DecoderWeightStorage,
    activation_storage: DecoderWeightStorage,
) -> (VisionStackShardManifest, Vec<u8>, Vec<u8>, Vec<u8>) {
    let ranges = vision_stack_layer_weight_ranges_with_vector_storage(
        HIDDEN,
        INTERMEDIATE,
        matrix_weight_storage,
        vector_weight_storage,
    )
    .unwrap();
    let layer_bytes = ranges.last().unwrap().offset + ranges.last().unwrap().bytes;
    let input =
        finite_f16((TOKENS * HIDDEN * activation_storage.bytes_per_element() as u32) as usize);
    let layer = finite_f16(layer_bytes as usize);
    let post_norm =
        finite_f16((HIDDEN * 2 * vector_weight_storage.bytes_per_element() as u32) as usize);
    let manifest = VisionStackShardManifest {
        schema_version: VISION_STACK_SHARD_MANIFEST_SCHEMA_VERSION,
        oracle: VisionStackShardOracle::Synthetic,
        case_id: "synthetic.full_fp16_vision_stack".to_owned(),
        model_id: MODEL_ID.to_owned(),
        model_revision: MODEL_REVISION.to_owned(),
        compiler_model_abi: COMPILER_MODEL_ABI,
        compiler_build: "0".repeat(64),
        golden_bundle_digest: None,
        semantic_fingerprint: None,
        matrix_weight_storage,
        matrix_weight_layout: if matrix_weight_storage == DecoderWeightStorage::F16 {
            LinearWeightLayout::InputMajor
        } else {
            LinearWeightLayout::OutputMajor
        },
        vector_weight_storage,
        activation_storage,
        tokens: TOKENS,
        hidden_size: HIDDEN,
        attention_heads: 1,
        head_dim: HIDDEN,
        intermediate_size: INTERMEDIATE,
        layer_norm_epsilon: 1.0e-6,
        cu_seqlens: vec![0, TOKENS],
        layer_count: 1,
        checkpoint_layers: vec![],
        shards: vec![
            descriptor(
                "input.embeddings",
                VisionStackShardKind::Input,
                None,
                &input,
            ),
            descriptor(
                "weights.vision_layer.00",
                VisionStackShardKind::Layer,
                Some(0),
                &layer,
            ),
            descriptor(
                "weights.vision_post_norm",
                VisionStackShardKind::PostNorm,
                None,
                &post_norm,
            ),
        ],
    };
    (manifest, input, layer, post_norm)
}

#[test]
fn full_fp16_manifest_halves_every_vision_tensor_and_activation() {
    let (manifest, _, _, _) = manifest(
        DecoderWeightStorage::F16,
        DecoderWeightStorage::F16,
        DecoderWeightStorage::F16,
    );
    let plan = manifest.plan().unwrap();

    assert_eq!(
        full_fp16_ranges().map(|range| range.storage),
        [DecoderWeightStorage::F16; 16],
    );
    assert_eq!(
        full_fp16_ranges().map(|range| (range.offset, range.bytes)),
        [
            (0, 8),
            (8, 8),
            (16, 32),
            (48, 8),
            (56, 32),
            (88, 8),
            (96, 32),
            (128, 8),
            (136, 32),
            (168, 8),
            (176, 8),
            (184, 8),
            (192, 64),
            (256, 16),
            (272, 64),
            (336, 8),
        ],
    );
    assert_eq!(plan.vector_weight_storage, DecoderWeightStorage::F16);
    assert_eq!(plan.activation_storage, DecoderWeightStorage::F16);
    assert_eq!(plan.input_bytes, 16);
    assert_eq!(plan.hidden_bytes, 16);
    assert_eq!(plan.intermediate_bytes, 32);
    assert_eq!(plan.layer_weight_bytes, 344);
    assert_eq!(plan.post_norm_bytes, 16);
    assert_eq!(plan.transport_bytes, 376);
    assert_eq!(plan.activation_arena_bytes, 240);
    assert_eq!(plan.readback_bytes, 16);
    assert_eq!(plan.peak_gpu_data_bytes, 600);

    let canonical = canonical_vision_stack_shard_manifest_bytes(&manifest).unwrap();
    let text = std::str::from_utf8(&canonical).unwrap();
    assert!(text.contains(r#""vector_weight_storage":"f16""#));
    assert!(text.contains(r#""activation_storage":"f16""#));
    let reparsed = parse_vision_stack_shard_manifest(&canonical).unwrap();
    assert_eq!(reparsed, manifest);
}

#[test]
fn full_fp16_inspection_uses_half_precision_for_input_vectors_and_post_norm() {
    let (manifest, mut input, mut layer, mut post_norm) = manifest(
        DecoderWeightStorage::F16,
        DecoderWeightStorage::F16,
        DecoderWeightStorage::F16,
    );
    for (id, payload) in [
        ("input.embeddings", &input),
        ("weights.vision_layer.00", &layer),
        ("weights.vision_post_norm", &post_norm),
    ] {
        assert!(
            inspect_vision_stack_shard(&manifest, id, payload)
                .unwrap()
                .all_finite
        );
    }

    input[0..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert!(
        !inspect_vision_stack_shard(&manifest, "input.embeddings", &input)
            .unwrap()
            .all_finite
    );

    layer[0..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert!(
        !inspect_vision_stack_shard(&manifest, "weights.vision_layer.00", &layer)
            .unwrap()
            .all_finite
    );

    post_norm[0..2].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert!(
        !inspect_vision_stack_shard(&manifest, "weights.vision_post_norm", &post_norm)
            .unwrap()
            .all_finite
    );
}

#[test]
fn half_activations_require_one_coherent_full_fp16_profile() {
    for (matrix, vectors, activations) in [
        (
            DecoderWeightStorage::F16,
            DecoderWeightStorage::F32,
            DecoderWeightStorage::F16,
        ),
        (
            DecoderWeightStorage::F32,
            DecoderWeightStorage::F16,
            DecoderWeightStorage::F16,
        ),
        (
            DecoderWeightStorage::F32,
            DecoderWeightStorage::F32,
            DecoderWeightStorage::F16,
        ),
        (
            DecoderWeightStorage::F16,
            DecoderWeightStorage::F16,
            DecoderWeightStorage::F32,
        ),
    ] {
        let (manifest, _, _, _) = manifest(matrix, vectors, activations);
        let error = manifest
            .plan()
            .expect_err("a partial profile must not silently bind incompatible kernels");
        assert_eq!(error.code(), VisionStackShardErrorCode::InvalidManifest);
        assert!(error.to_string().contains("full FP16"));
    }
}
