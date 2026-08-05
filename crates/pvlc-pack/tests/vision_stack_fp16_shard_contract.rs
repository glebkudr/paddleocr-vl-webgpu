use pvlc_model_schema::{COMPILER_MODEL_ABI, MODEL_ID, MODEL_REVISION};
use pvlc_pack::{
    VISION_STACK_SHARD_MANIFEST_SCHEMA_VERSION, VisionStackLayerTensorRange,
    VisionStackShardDescriptor, VisionStackShardErrorCode, VisionStackShardKind,
    VisionStackShardManifest, VisionStackShardOracle, canonical_vision_stack_shard_manifest_bytes,
    inspect_vision_stack_shard, parse_vision_stack_shard_manifest,
    vision_stack_layer_weight_ranges,
};
use pvlc_runtime_core::{DecoderWeightStorage, LinearWeightLayout};

const TOKENS: u32 = 2;
const HIDDEN: u32 = 4;
const INTERMEDIATE: u32 = 8;

fn ranges(storage: DecoderWeightStorage) -> [VisionStackLayerTensorRange; 16] {
    vision_stack_layer_weight_ranges(HIDDEN, INTERMEDIATE, storage).unwrap()
}

fn finite_layer_payload(storage: DecoderWeightStorage) -> Vec<u8> {
    let ranges = ranges(storage);
    let payload_bytes = ranges
        .last()
        .map(|range| range.offset + range.bytes)
        .unwrap();
    let mut payload = vec![0_u8; payload_bytes as usize];
    for range in ranges {
        let bytes = &mut payload[range.offset as usize..(range.offset + range.bytes) as usize];
        match range.storage {
            DecoderWeightStorage::F16 => {
                for value in bytes.chunks_exact_mut(2) {
                    value.copy_from_slice(&0x3c00_u16.to_le_bytes());
                }
            }
            DecoderWeightStorage::F32 => {
                for value in bytes.chunks_exact_mut(4) {
                    value.copy_from_slice(&1.0_f32.to_le_bytes());
                }
            }
        }
    }
    payload
}

fn manifest(
    matrix_weight_storage: DecoderWeightStorage,
    layer_payload: &[u8],
) -> VisionStackShardManifest {
    manifest_with_layout(
        matrix_weight_storage,
        LinearWeightLayout::OutputMajor,
        layer_payload,
    )
}

fn manifest_with_layout(
    matrix_weight_storage: DecoderWeightStorage,
    matrix_weight_layout: LinearWeightLayout,
    layer_payload: &[u8],
) -> VisionStackShardManifest {
    let input = vec![0_u8; (TOKENS * HIDDEN * 4) as usize];
    let post_norm = vec![0_u8; (HIDDEN * 2 * 4) as usize];
    VisionStackShardManifest {
        schema_version: VISION_STACK_SHARD_MANIFEST_SCHEMA_VERSION,
        oracle: VisionStackShardOracle::Synthetic,
        case_id: "synthetic.fp16_vision_stack".to_owned(),
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
                layer_payload,
            ),
            descriptor(
                "weights.vision_post_norm",
                VisionStackShardKind::PostNorm,
                None,
                &post_norm,
            ),
        ],
    }
}

#[test]
fn input_major_layout_is_canonical_only_for_fp16_matrices() {
    let payload = finite_layer_payload(DecoderWeightStorage::F16);
    let manifest = manifest_with_layout(
        DecoderWeightStorage::F16,
        LinearWeightLayout::InputMajor,
        &payload,
    );
    let canonical = canonical_vision_stack_shard_manifest_bytes(&manifest).unwrap();
    let text = std::str::from_utf8(&canonical).unwrap();
    assert!(text.contains(r#""matrix_weight_layout":"input_major""#));

    let reparsed = parse_vision_stack_shard_manifest(&canonical).unwrap();
    assert_eq!(
        reparsed.matrix_weight_layout,
        LinearWeightLayout::InputMajor,
    );
    assert_eq!(
        reparsed.plan().unwrap().matrix_weight_layout,
        LinearWeightLayout::InputMajor,
    );

    let f32_payload = finite_layer_payload(DecoderWeightStorage::F32);
    assert_eq!(
        manifest_with_layout(
            DecoderWeightStorage::F32,
            LinearWeightLayout::InputMajor,
            &f32_payload,
        )
        .plan()
        .unwrap_err()
        .code(),
        VisionStackShardErrorCode::InvalidManifest,
        "input-major is unsupported by the scalar F32 projection shader",
    );
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

#[test]
fn fp16_manifest_halves_only_the_six_large_matrix_payloads() {
    let f32_ranges = ranges(DecoderWeightStorage::F32);
    let f16_ranges = ranges(DecoderWeightStorage::F16);

    assert_eq!(
        f16_ranges.map(|range| range.storage),
        [
            DecoderWeightStorage::F32,
            DecoderWeightStorage::F32,
            DecoderWeightStorage::F16,
            DecoderWeightStorage::F32,
            DecoderWeightStorage::F16,
            DecoderWeightStorage::F32,
            DecoderWeightStorage::F16,
            DecoderWeightStorage::F32,
            DecoderWeightStorage::F16,
            DecoderWeightStorage::F32,
            DecoderWeightStorage::F32,
            DecoderWeightStorage::F32,
            DecoderWeightStorage::F16,
            DecoderWeightStorage::F32,
            DecoderWeightStorage::F16,
            DecoderWeightStorage::F32,
        ]
    );
    assert_eq!(
        f16_ranges.map(|range| (range.offset, range.bytes)),
        [
            (0, 16),
            (16, 16),
            (32, 32),
            (64, 16),
            (80, 32),
            (112, 16),
            (128, 32),
            (160, 16),
            (176, 32),
            (208, 16),
            (224, 16),
            (240, 16),
            (256, 64),
            (320, 32),
            (352, 64),
            (416, 16),
        ]
    );
    assert_eq!(
        f32_ranges.last().unwrap().offset + f32_ranges.last().unwrap().bytes,
        688
    );
    assert_eq!(
        f16_ranges.last().unwrap().offset + f16_ranges.last().unwrap().bytes,
        432
    );

    let payload = finite_layer_payload(DecoderWeightStorage::F16);
    let plan = manifest(DecoderWeightStorage::F16, &payload)
        .plan()
        .unwrap();
    assert_eq!(plan.matrix_weight_storage, DecoderWeightStorage::F16);
    assert_eq!(plan.layer_weight_bytes, 432);
    assert_eq!(plan.transport_bytes, 496);
}

#[test]
fn mixed_precision_inspection_checks_each_tensor_using_its_declared_storage() {
    let mut payload = finite_layer_payload(DecoderWeightStorage::F16);
    let manifest = manifest(DecoderWeightStorage::F16, &payload);
    // Each four-byte discriminator is finite as one little-endian F32
    // (0x00007c00), but its low F16 half is +infinity. This proves both a
    // LayerNorm vector and a linear bias are inspected with F32 semantics.
    let f32_not_f16_discriminator = [0x00, 0x7c, 0x00, 0x00];
    payload[0..4].copy_from_slice(&f32_not_f16_discriminator);
    payload[64..68].copy_from_slice(&f32_not_f16_discriminator);
    let observed =
        inspect_vision_stack_shard(&manifest, "weights.vision_layer.00", &payload).unwrap();
    assert!(observed.all_finite);

    // First matrix coefficient: IEEE-F16 +infinity.
    payload[32..34].copy_from_slice(&0x7c00_u16.to_le_bytes());
    assert!(
        !inspect_vision_stack_shard(&manifest, "weights.vision_layer.00", &payload,)
            .unwrap()
            .all_finite
    );

    payload[32..34].copy_from_slice(&0x3c00_u16.to_le_bytes());
    // First query bias: IEEE-F32 +infinity. Biases remain F32 so both paths
    // consume the exact widened value from the shared FP16 checkpoint.
    payload[64..68].copy_from_slice(&f32::INFINITY.to_le_bytes());
    assert!(
        !inspect_vision_stack_shard(&manifest, "weights.vision_layer.00", &payload,)
            .unwrap()
            .all_finite
    );
}

#[test]
fn the_legacy_f32_layout_and_observation_remain_unchanged() {
    let mut payload = finite_layer_payload(DecoderWeightStorage::F32);
    let manifest = manifest(DecoderWeightStorage::F32, &payload);
    let plan = manifest.plan().unwrap();
    assert_eq!(plan.matrix_weight_storage, DecoderWeightStorage::F32);
    assert_eq!(plan.layer_weight_bytes, 688);
    assert!(
        inspect_vision_stack_shard(&manifest, "weights.vision_layer.00", &payload,)
            .unwrap()
            .all_finite
    );

    payload[32..36].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(
        !inspect_vision_stack_shard(&manifest, "weights.vision_layer.00", &payload,)
            .unwrap()
            .all_finite
    );
}

#[test]
fn legacy_canonical_schema_v1_bytes_without_the_new_field_remain_byte_exact() {
    let payload = finite_layer_payload(DecoderWeightStorage::F32);
    let manifest = manifest(DecoderWeightStorage::F32, &payload);
    let canonical = canonical_vision_stack_shard_manifest_bytes(&manifest).unwrap();
    let text = std::str::from_utf8(&canonical).unwrap();

    assert!(
        !text.contains("matrix_weight_storage"),
        "the default F32 field must stay absent so persisted schema-v1 bytes remain canonical",
    );
    assert!(
        !text.contains("matrix_weight_layout"),
        "the default output-major layout must stay absent from legacy schema-v1 bytes",
    );
    let reparsed = parse_vision_stack_shard_manifest(&canonical).unwrap();
    assert_eq!(reparsed.matrix_weight_storage, DecoderWeightStorage::F32);
    assert_eq!(
        reparsed.matrix_weight_layout,
        LinearWeightLayout::OutputMajor,
    );
    assert_eq!(
        canonical_vision_stack_shard_manifest_bytes(&reparsed).unwrap(),
        canonical
    );
}

#[test]
fn legacy_fp16_output_major_manifest_omits_layout_and_roundtrips_unchanged() {
    let payload = finite_layer_payload(DecoderWeightStorage::F16);
    let manifest = manifest(DecoderWeightStorage::F16, &payload);
    let canonical = canonical_vision_stack_shard_manifest_bytes(&manifest).unwrap();
    let text = std::str::from_utf8(&canonical).unwrap();

    assert!(text.contains(r#""matrix_weight_storage":"f16""#));
    assert!(
        !text.contains("matrix_weight_layout"),
        "legacy FP16 output-major bytes must not gain a default layout field",
    );
    let reparsed = parse_vision_stack_shard_manifest(&canonical).unwrap();
    assert_eq!(reparsed.matrix_weight_storage, DecoderWeightStorage::F16,);
    assert_eq!(
        reparsed.matrix_weight_layout,
        LinearWeightLayout::OutputMajor,
    );
    assert_eq!(
        canonical_vision_stack_shard_manifest_bytes(&reparsed).unwrap(),
        canonical,
    );
}

#[test]
fn manifest_rejects_a_layer_payload_sized_for_the_other_storage_mode() {
    for (declared, actual) in [
        (DecoderWeightStorage::F16, DecoderWeightStorage::F32),
        (DecoderWeightStorage::F32, DecoderWeightStorage::F16),
    ] {
        let payload = finite_layer_payload(actual);
        assert_eq!(
            manifest(declared, &payload).plan().unwrap_err().code(),
            VisionStackShardErrorCode::LengthMismatch,
            "declared {declared:?}, payload {actual:?}",
        );
    }
}

#[test]
fn mixed_precision_inspection_rejects_a_truncated_or_unaligned_layer() {
    let payload = finite_layer_payload(DecoderWeightStorage::F16);
    let manifest = manifest(DecoderWeightStorage::F16, &payload);
    assert_eq!(
        inspect_vision_stack_shard(
            &manifest,
            "weights.vision_layer.00",
            &payload[..payload.len() - 1],
        )
        .unwrap_err()
        .code(),
        VisionStackShardErrorCode::LengthMismatch,
    );
}

#[test]
fn fp16_matrix_manifest_still_inspects_input_and_post_norm_as_f32() {
    let layer = finite_layer_payload(DecoderWeightStorage::F16);
    let manifest = manifest(DecoderWeightStorage::F16, &layer);
    let mut input = vec![0_u8; (TOKENS * HIDDEN * 4) as usize];
    let mut post_norm = vec![0_u8; (HIDDEN * 2 * 4) as usize];
    let f32_not_f16_discriminator = [0x00, 0x7c, 0x00, 0x00];
    input[0..4].copy_from_slice(&f32_not_f16_discriminator);
    post_norm[0..4].copy_from_slice(&f32_not_f16_discriminator);

    assert!(
        inspect_vision_stack_shard(&manifest, "input.embeddings", &input)
            .unwrap()
            .all_finite
    );
    assert!(
        inspect_vision_stack_shard(&manifest, "weights.vision_post_norm", &post_norm)
            .unwrap()
            .all_finite
    );

    input[0..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
    post_norm[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(
        !inspect_vision_stack_shard(&manifest, "input.embeddings", &input)
            .unwrap()
            .all_finite
    );
    assert!(
        !inspect_vision_stack_shard(&manifest, "weights.vision_post_norm", &post_norm)
            .unwrap()
            .all_finite
    );
}
