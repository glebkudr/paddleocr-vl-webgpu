use std::collections::BTreeMap;

use pvlc_pack::{
    VisionStackShardAcceptance, VisionStackShardDescriptor, VisionStackShardErrorCode,
    VisionStackShardKind, VisionStackShardManifest, VisionStackShardObservation,
    VisionStackShardOracle, VisionStackShardProtocol, VisionStackShardProtocolPhase,
    canonical_vision_stack_shard_manifest_bytes, inspect_vision_stack_f32_shard,
    parse_vision_stack_shard_manifest,
};

const MODEL_ID: &str = "PaddlePaddle/PaddleOCR-VL-1.6";
const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const COMPILER_BUILD: &str = "4e40a34cd64f347a14270d339ab61b131ae3120ca566ff140c81d101a3aa4a2c";

fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
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

fn synthetic_fixture() -> (VisionStackShardManifest, BTreeMap<String, Vec<u8>>) {
    let input = f32_bytes((0..12).map(|index| index as f32 / 13.0 - 0.25));
    let layer_payload = |layer: u32| {
        f32_bytes(
            (0..145).map(move |index| (index as f32 + 1.0) * (layer as f32 + 1.0) / 257.0 - 0.4),
        )
    };
    let post_norm = f32_bytes([1.0, 1.1, 0.9, 1.2, 0.01, -0.02, 0.03, -0.04]);
    let mut payloads = BTreeMap::from([("input.embeddings".to_owned(), input)]);
    for layer in 0..3 {
        payloads.insert(
            format!("weights.vision_layer.{layer:02}"),
            layer_payload(layer),
        );
    }
    payloads.insert("weights.vision_post_norm".to_owned(), post_norm);

    let shards = [
        "input.embeddings",
        "weights.vision_layer.00",
        "weights.vision_layer.01",
        "weights.vision_layer.02",
        "weights.vision_post_norm",
    ]
    .into_iter()
    .enumerate()
    .map(|(position, id)| {
        let (kind, layer_index) = match position {
            0 => (VisionStackShardKind::Input, None),
            1..=3 => (VisionStackShardKind::Layer, Some((position - 1) as u32)),
            _ => (VisionStackShardKind::PostNorm, None),
        };
        descriptor(id, kind, layer_index, &payloads[id])
    })
    .collect();
    let manifest = VisionStackShardManifest {
        schema_version: 1,
        oracle: VisionStackShardOracle::Synthetic,
        case_id: "synthetic.vision_stack/bounded_streaming".to_owned(),
        model_id: MODEL_ID.to_owned(),
        model_revision: MODEL_REVISION.to_owned(),
        compiler_model_abi: 1,
        compiler_build: COMPILER_BUILD.to_owned(),
        golden_bundle_digest: None,
        semantic_fingerprint: None,
        matrix_weight_storage: pvlc_runtime_core::DecoderWeightStorage::F32,
        matrix_weight_layout: pvlc_runtime_core::LinearWeightLayout::OutputMajor,
        vector_weight_storage: pvlc_runtime_core::DecoderWeightStorage::F32,
        activation_storage: pvlc_runtime_core::DecoderWeightStorage::F32,
        tokens: 3,
        hidden_size: 4,
        attention_heads: 2,
        head_dim: 2,
        intermediate_size: 5,
        layer_norm_epsilon: 1.0e-6,
        cu_seqlens: vec![0, 1, 3],
        layer_count: 3,
        checkpoint_layers: vec![0, 2],
        shards,
    };
    (manifest, payloads)
}

fn official_manifest() -> VisionStackShardManifest {
    let digest = "a".repeat(64);
    let mut shards = Vec::with_capacity(29);
    shards.push(VisionStackShardDescriptor {
        id: "input.embeddings".to_owned(),
        kind: VisionStackShardKind::Input,
        layer_index: None,
        bytes: 5_879_808,
        blake3: digest.clone(),
    });
    for layer in 0..27 {
        shards.push(VisionStackShardDescriptor {
            id: format!("weights.vision_layer.{layer:02}"),
            kind: VisionStackShardKind::Layer,
            layer_index: Some(layer),
            bytes: 60_958_016,
            blake3: digest.clone(),
        });
    }
    shards.push(VisionStackShardDescriptor {
        id: "weights.vision_post_norm".to_owned(),
        kind: VisionStackShardKind::PostNorm,
        layer_index: None,
        bytes: 9_216,
        blake3: digest,
    });
    VisionStackShardManifest {
        schema_version: 1,
        oracle: VisionStackShardOracle::OfficialMpsBf16,
        case_id: "ocr.clean_latin.0001/vision.stack.27".to_owned(),
        model_id: MODEL_ID.to_owned(),
        model_revision: MODEL_REVISION.to_owned(),
        compiler_model_abi: 1,
        compiler_build: COMPILER_BUILD.to_owned(),
        golden_bundle_digest: Some(
            "blake3:35572ac07da5fddee97becb46fb866f906200f433f4c5677ad20fdf36440acf9".to_owned(),
        ),
        semantic_fingerprint: Some(
            "blake3:632cbe2de6f47f8f84c764e3a551bf99f352d7459c98d939eb89725357f952b4".to_owned(),
        ),
        matrix_weight_storage: pvlc_runtime_core::DecoderWeightStorage::F32,
        matrix_weight_layout: pvlc_runtime_core::LinearWeightLayout::OutputMajor,
        vector_weight_storage: pvlc_runtime_core::DecoderWeightStorage::F32,
        activation_storage: pvlc_runtime_core::DecoderWeightStorage::F32,
        tokens: 1_276,
        hidden_size: 1_152,
        attention_heads: 16,
        head_dim: 72,
        intermediate_size: 4_304,
        layer_norm_epsilon: 1.0e-6,
        cu_seqlens: vec![0, 1_276],
        layer_count: 27,
        checkpoint_layers: vec![0, 1, 13, 26],
        shards,
    }
}

fn table_l2_official_manifest() -> VisionStackShardManifest {
    let mut manifest = official_manifest();
    manifest.case_id = "table.simple.0001/vision.stack.27".to_owned();
    manifest.golden_bundle_digest =
        Some("blake3:4cecc8b2abc37030920acf8860bb317084125dbbbcae5af394fccd0c8044c842".to_owned());
    manifest.semantic_fingerprint =
        Some("blake3:91f72ffdf81070c2c9ca31628605064bdc02bf19417482cefce41d6a27aa5404".to_owned());
    manifest.tokens = 1_740;
    manifest.cu_seqlens = vec![0, 1_740];
    manifest.checkpoint_layers.clear();
    manifest.shards[0].bytes = 8_017_920;
    manifest
}

fn assert_manifest_error(
    mut manifest: VisionStackShardManifest,
    mutate: impl FnOnce(&mut VisionStackShardManifest),
    code: VisionStackShardErrorCode,
) {
    mutate(&mut manifest);
    let error = canonical_vision_stack_shard_manifest_bytes(&manifest).unwrap_err();
    assert_eq!(error.code(), code, "{error}");
}

#[test]
fn canonical_manifest_roundtrips_and_derives_the_exact_bounded_execution_plan() {
    let (manifest, _) = synthetic_fixture();
    let first = canonical_vision_stack_shard_manifest_bytes(&manifest).unwrap();
    let second = canonical_vision_stack_shard_manifest_bytes(&manifest).unwrap();
    assert_eq!(first, second, "compiler output must be byte-reproducible");
    assert_eq!(first.last(), Some(&b'\n'));
    assert_eq!(first.iter().filter(|byte| **byte == b'\n').count(), 1);
    assert_eq!(parse_vision_stack_shard_manifest(&first).unwrap(), manifest);

    let plan = manifest.plan().unwrap();
    assert_eq!(plan.layer_count, 3);
    assert_eq!(plan.shard_count, 5);
    assert_eq!(plan.hidden_bytes, 48);
    assert_eq!(plan.intermediate_bytes, 60);
    assert_eq!(plan.layer_weight_bytes, 580);
    assert_eq!(plan.post_norm_bytes, 32);
    assert_eq!(plan.transport_bytes, 48 + 3 * 580 + 32);
    assert_eq!(plan.activation_buffer_count, 13);
    assert_eq!(plan.activation_arena_bytes, 11 * 48 + 2 * 60);
    assert_eq!(plan.readback_bytes, 3 * 48);
    assert_eq!(plan.peak_gpu_data_bytes, 11 * 48 + 2 * 60 + 3 * 48 + 580);
    assert_eq!(plan.submission_count, 4);
    assert_eq!(plan.compute_pass_count, 4);
    assert_eq!(plan.dispatch_count, 37);

    let mut noncanonical = first.clone();
    noncanonical.insert(0, b' ');
    assert_eq!(
        parse_vision_stack_shard_manifest(&noncanonical)
            .unwrap_err()
            .code(),
        VisionStackShardErrorCode::NonCanonicalManifest,
    );
    let mut unknown: serde_json::Value = serde_json::from_slice(&first).unwrap();
    unknown["shadow"] = true.into();
    let mut unknown_bytes = serde_json::to_vec(&unknown).unwrap();
    unknown_bytes.push(b'\n');
    assert_eq!(
        parse_vision_stack_shard_manifest(&unknown_bytes)
            .unwrap_err()
            .code(),
        VisionStackShardErrorCode::InvalidManifest,
    );
}

#[test]
fn manifest_rejects_identity_geometry_selection_and_every_shard_directory_drift() {
    let (manifest, _) = synthetic_fixture();
    type ManifestMutation = (
        Box<dyn FnOnce(&mut VisionStackShardManifest)>,
        VisionStackShardErrorCode,
    );
    let cases: Vec<ManifestMutation> = vec![
        (
            Box::new(|value| value.schema_version += 1),
            VisionStackShardErrorCode::InvalidManifest,
        ),
        (
            Box::new(|value| value.model_id = "other/model".to_owned()),
            VisionStackShardErrorCode::ModelIdentityMismatch,
        ),
        (
            Box::new(|value| value.model_revision = "0".repeat(40)),
            VisionStackShardErrorCode::ModelIdentityMismatch,
        ),
        (
            Box::new(|value| value.compiler_model_abi += 1),
            VisionStackShardErrorCode::ModelIdentityMismatch,
        ),
        (
            Box::new(|value| value.compiler_build = "A".repeat(64)),
            VisionStackShardErrorCode::InvalidManifest,
        ),
        (
            Box::new(|value| value.tokens = 0),
            VisionStackShardErrorCode::InvalidGeometry,
        ),
        (
            Box::new(|value| value.hidden_size = 5),
            VisionStackShardErrorCode::InvalidGeometry,
        ),
        (
            Box::new(|value| value.layer_norm_epsilon = f32::NAN),
            VisionStackShardErrorCode::InvalidGeometry,
        ),
        (
            Box::new(|value| value.cu_seqlens = vec![0, 3, 3]),
            VisionStackShardErrorCode::InvalidGeometry,
        ),
        (
            Box::new(|value| value.layer_count = 0),
            VisionStackShardErrorCode::InvalidGeometry,
        ),
        (
            Box::new(|value| value.checkpoint_layers = vec![0, 0]),
            VisionStackShardErrorCode::InvalidCheckpointSelection,
        ),
        (
            Box::new(|value| value.checkpoint_layers = vec![0, 3]),
            VisionStackShardErrorCode::InvalidCheckpointSelection,
        ),
        (
            Box::new(|value| value.checkpoint_layers = vec![2, 1]),
            VisionStackShardErrorCode::InvalidCheckpointSelection,
        ),
        (
            Box::new(|value| {
                value.shards.swap(1, 2);
            }),
            VisionStackShardErrorCode::InvalidShardDirectory,
        ),
        (
            Box::new(|value| {
                value.shards.pop();
            }),
            VisionStackShardErrorCode::InvalidShardDirectory,
        ),
        (
            Box::new(|value| value.shards.push(value.shards[0].clone())),
            VisionStackShardErrorCode::InvalidShardDirectory,
        ),
        (
            Box::new(|value| value.shards[0].id = "input.other".to_owned()),
            VisionStackShardErrorCode::InvalidShardDirectory,
        ),
        (
            Box::new(|value| value.shards[0].kind = VisionStackShardKind::Layer),
            VisionStackShardErrorCode::InvalidShardDirectory,
        ),
        (
            Box::new(|value| value.shards[1].layer_index = Some(2)),
            VisionStackShardErrorCode::InvalidShardDirectory,
        ),
        (
            Box::new(|value| value.shards[1].bytes -= 4),
            VisionStackShardErrorCode::LengthMismatch,
        ),
        (
            Box::new(|value| value.shards[4].bytes += 4),
            VisionStackShardErrorCode::LengthMismatch,
        ),
        (
            Box::new(|value| value.shards[2].blake3 = "A".repeat(64)),
            VisionStackShardErrorCode::InvalidShardDirectory,
        ),
        (
            Box::new(|value| value.golden_bundle_digest = Some("blake3:bad".to_owned())),
            VisionStackShardErrorCode::InvalidManifest,
        ),
    ];
    for (mutate, code) in cases {
        assert_manifest_error(manifest.clone(), mutate, code);
    }

    let mut official = official_manifest();
    official.golden_bundle_digest = None;
    assert_eq!(
        canonical_vision_stack_shard_manifest_bytes(&official)
            .unwrap_err()
            .code(),
        VisionStackShardErrorCode::OfficialIdentityMismatch,
    );
    let mut official = official_manifest();
    official.golden_bundle_digest = Some(format!("blake3:{}", "0".repeat(64)));
    assert_eq!(
        canonical_vision_stack_shard_manifest_bytes(&official)
            .unwrap_err()
            .code(),
        VisionStackShardErrorCode::OfficialIdentityMismatch,
    );
    let mut official = official_manifest();
    official.semantic_fingerprint = Some(format!("blake3:{}", "0".repeat(64)));
    assert_eq!(
        canonical_vision_stack_shard_manifest_bytes(&official)
            .unwrap_err()
            .code(),
        VisionStackShardErrorCode::OfficialIdentityMismatch,
    );
}

#[test]
fn protocol_requires_a_complete_gpu_free_preflight_then_the_same_exact_execution_sequence() {
    let (manifest, payloads) = synthetic_fixture();
    let mut protocol = VisionStackShardProtocol::new(manifest).unwrap();
    assert_eq!(protocol.phase(), VisionStackShardProtocolPhase::Preflight);
    assert_eq!(protocol.next_shard_id(), Some("input.embeddings"));

    let order = [
        "input.embeddings",
        "weights.vision_layer.00",
        "weights.vision_layer.01",
        "weights.vision_layer.02",
        "weights.vision_post_norm",
    ];
    for (position, id) in order.into_iter().enumerate() {
        let observed = inspect_vision_stack_f32_shard(id, &payloads[id]);
        let accepted = protocol.accept_preflight(&observed).unwrap();
        assert_eq!(accepted.phase, VisionStackShardProtocolPhase::Preflight);
        assert_eq!(accepted.id, id);
        assert_eq!(accepted.checkpoint_slot, None);
        if position + 1 == order.len() {
            assert_eq!(protocol.phase(), VisionStackShardProtocolPhase::Ready);
        }
    }
    assert_eq!(protocol.next_shard_id(), Some("input.embeddings"));

    let input = inspect_vision_stack_f32_shard(order[0], &payloads[order[0]]);
    let accepted = protocol.accept_execution(&input).unwrap();
    assert_eq!(accepted.kind, VisionStackShardKind::Input);
    assert_eq!(accepted.checkpoint_slot, None);
    assert_eq!(protocol.phase(), VisionStackShardProtocolPhase::Executing);

    for (layer, checkpoint_slot) in [(0, Some(0)), (1, None), (2, Some(1))] {
        let id = format!("weights.vision_layer.{layer:02}");
        let observed = inspect_vision_stack_f32_shard(&id, &payloads[&id]);
        let accepted = protocol.accept_execution(&observed).unwrap();
        assert_eq!(
            accepted,
            VisionStackShardAcceptance {
                phase: VisionStackShardProtocolPhase::Executing,
                id,
                kind: VisionStackShardKind::Layer,
                layer_index: Some(layer),
                checkpoint_slot,
            },
        );
    }
    let post_norm = inspect_vision_stack_f32_shard(order[4], &payloads[order[4]]);
    let accepted = protocol.accept_execution(&post_norm).unwrap();
    assert_eq!(accepted.kind, VisionStackShardKind::PostNorm);
    assert_eq!(
        accepted.checkpoint_slot,
        Some(2),
        "final output follows selected depths"
    );
    assert_eq!(protocol.phase(), VisionStackShardProtocolPhase::Complete);
    assert_eq!(protocol.next_shard_id(), None);
}

#[test]
fn deferred_preflight_uses_manifest_order_but_execution_still_authenticates_every_payload() {
    let (manifest, payloads) = synthetic_fixture();
    let order = [
        "input.embeddings",
        "weights.vision_layer.00",
        "weights.vision_layer.01",
        "weights.vision_layer.02",
        "weights.vision_post_norm",
    ];
    let mut protocol = VisionStackShardProtocol::new(manifest).unwrap();

    assert_eq!(
        protocol
            .accept_deferred_preflight(order[1])
            .unwrap_err()
            .code(),
        VisionStackShardErrorCode::WrongShardOrder,
    );
    assert_eq!(protocol.next_shard_id(), Some(order[0]));
    for (position, id) in order.iter().copied().enumerate() {
        let accepted = protocol.accept_deferred_preflight(id).unwrap();
        assert_eq!(accepted.phase, VisionStackShardProtocolPhase::Preflight);
        assert_eq!(accepted.id, id);
        assert_eq!(accepted.checkpoint_slot, None);
        if position + 1 == order.len() {
            assert_eq!(protocol.phase(), VisionStackShardProtocolPhase::Ready);
        }
    }
    assert_eq!(
        protocol
            .accept_deferred_preflight(order[0])
            .unwrap_err()
            .code(),
        VisionStackShardErrorCode::InvalidPhase,
    );

    let assert_execution_rejected =
        |protocol: &mut VisionStackShardProtocol,
         observation: VisionStackShardObservation,
         code: VisionStackShardErrorCode| {
            let expected = protocol.next_shard_id().map(str::to_owned);
            assert_eq!(
                protocol.accept_execution(&observation).unwrap_err().code(),
                code,
            );
            assert_eq!(
                protocol.next_shard_id().map(str::to_owned),
                expected,
                "deferred preflight allowed rejected execution to advance",
            );
        };

    let mut mutated_input = payloads[order[0]].clone();
    mutated_input[0] ^= 1;
    assert_execution_rejected(
        &mut protocol,
        inspect_vision_stack_f32_shard(order[0], &mutated_input),
        VisionStackShardErrorCode::DigestMismatch,
    );
    protocol
        .accept_execution(&inspect_vision_stack_f32_shard(
            order[0],
            &payloads[order[0]],
        ))
        .unwrap();

    let layer0 = protocol.manifest().shards[1].clone();
    assert_execution_rejected(
        &mut protocol,
        VisionStackShardObservation {
            id: layer0.id,
            bytes: layer0.bytes,
            blake3: layer0.blake3,
            all_finite: false,
        },
        VisionStackShardErrorCode::NonFinitePayload,
    );
    protocol
        .accept_execution(&inspect_vision_stack_f32_shard(
            order[1],
            &payloads[order[1]],
        ))
        .unwrap();

    assert_execution_rejected(
        &mut protocol,
        inspect_vision_stack_f32_shard(order[2], &payloads[order[2]][..576]),
        VisionStackShardErrorCode::LengthMismatch,
    );
    for id in &order[2..] {
        protocol
            .accept_execution(&inspect_vision_stack_f32_shard(id, &payloads[*id]))
            .unwrap();
    }
    assert_eq!(protocol.phase(), VisionStackShardProtocolPhase::Complete);
}

#[test]
fn rejected_payload_or_out_of_order_call_never_advances_the_protocol() {
    let (manifest, payloads) = synthetic_fixture();
    let mut protocol = VisionStackShardProtocol::new(manifest).unwrap();
    let expected = protocol.next_shard_id().unwrap().to_owned();

    let wrong_order = inspect_vision_stack_f32_shard(
        "weights.vision_layer.00",
        &payloads["weights.vision_layer.00"],
    );
    assert_eq!(
        protocol.accept_preflight(&wrong_order).unwrap_err().code(),
        VisionStackShardErrorCode::WrongShardOrder,
    );
    assert_eq!(protocol.next_shard_id(), Some(expected.as_str()));

    let short =
        inspect_vision_stack_f32_shard("input.embeddings", &payloads["input.embeddings"][..44]);
    assert_eq!(
        protocol.accept_preflight(&short).unwrap_err().code(),
        VisionStackShardErrorCode::LengthMismatch,
    );
    assert_eq!(protocol.next_shard_id(), Some(expected.as_str()));

    let corrupt = VisionStackShardObservation {
        id: "input.embeddings".to_owned(),
        bytes: 48,
        blake3: "0".repeat(64),
        all_finite: true,
    };
    assert_eq!(
        protocol.accept_preflight(&corrupt).unwrap_err().code(),
        VisionStackShardErrorCode::DigestMismatch,
    );
    assert_eq!(protocol.next_shard_id(), Some(expected.as_str()));

    let mut nan_payload = payloads["input.embeddings"].clone();
    nan_payload[8..12].copy_from_slice(&f32::NAN.to_le_bytes());
    let nan = inspect_vision_stack_f32_shard("input.embeddings", &nan_payload);
    assert_eq!(
        protocol.accept_preflight(&nan).unwrap_err().code(),
        VisionStackShardErrorCode::DigestMismatch,
        "digest mismatch is checked before semantic payload acceptance",
    );
    let nan_reanchored = VisionStackShardObservation {
        blake3: payloads
            .get("input.embeddings")
            .map(|_| protocol.manifest().shards[0].blake3.clone())
            .unwrap(),
        ..nan
    };
    assert_eq!(
        protocol
            .accept_preflight(&nan_reanchored)
            .unwrap_err()
            .code(),
        VisionStackShardErrorCode::NonFinitePayload,
    );
    assert_eq!(protocol.next_shard_id(), Some(expected.as_str()));

    for id in [
        "input.embeddings",
        "weights.vision_layer.00",
        "weights.vision_layer.01",
        "weights.vision_layer.02",
        "weights.vision_post_norm",
    ] {
        protocol
            .accept_preflight(&inspect_vision_stack_f32_shard(id, &payloads[id]))
            .unwrap();
    }
    assert_eq!(
        protocol
            .accept_preflight(&inspect_vision_stack_f32_shard(
                "input.embeddings",
                &payloads["input.embeddings"],
            ))
            .unwrap_err()
            .code(),
        VisionStackShardErrorCode::InvalidPhase,
    );
}

#[test]
fn execution_revalidates_mutated_input_middle_layer_and_post_norm_without_advancing() {
    let (manifest, payloads) = synthetic_fixture();
    let mut protocol = VisionStackShardProtocol::new(manifest).unwrap();
    for id in [
        "input.embeddings",
        "weights.vision_layer.00",
        "weights.vision_layer.01",
        "weights.vision_layer.02",
        "weights.vision_post_norm",
    ] {
        protocol
            .accept_preflight(&inspect_vision_stack_f32_shard(id, &payloads[id]))
            .unwrap();
    }

    let assert_rejected = |protocol: &mut VisionStackShardProtocol,
                           observation: VisionStackShardObservation,
                           code: VisionStackShardErrorCode| {
        let expected = protocol.next_shard_id().map(str::to_owned);
        assert_eq!(
            protocol.accept_execution(&observation).unwrap_err().code(),
            code,
        );
        assert_eq!(
            protocol.next_shard_id().map(str::to_owned),
            expected,
            "a rejected execution shard advanced the state machine",
        );
    };

    let mut mutated_input = payloads["input.embeddings"].clone();
    mutated_input[0] ^= 1;
    assert_rejected(
        &mut protocol,
        inspect_vision_stack_f32_shard("input.embeddings", &mutated_input),
        VisionStackShardErrorCode::DigestMismatch,
    );
    protocol
        .accept_execution(&inspect_vision_stack_f32_shard(
            "input.embeddings",
            &payloads["input.embeddings"],
        ))
        .unwrap();

    assert_rejected(
        &mut protocol,
        inspect_vision_stack_f32_shard(
            "weights.vision_layer.01",
            &payloads["weights.vision_layer.01"],
        ),
        VisionStackShardErrorCode::WrongShardOrder,
    );
    protocol
        .accept_execution(&inspect_vision_stack_f32_shard(
            "weights.vision_layer.00",
            &payloads["weights.vision_layer.00"],
        ))
        .unwrap();

    let middle_id = "weights.vision_layer.01";
    assert_rejected(
        &mut protocol,
        inspect_vision_stack_f32_shard(middle_id, &payloads[middle_id][..576]),
        VisionStackShardErrorCode::LengthMismatch,
    );
    let mut mutated_middle = payloads[middle_id].clone();
    mutated_middle[24] ^= 1;
    assert_rejected(
        &mut protocol,
        inspect_vision_stack_f32_shard(middle_id, &mutated_middle),
        VisionStackShardErrorCode::DigestMismatch,
    );
    let non_finite_middle = VisionStackShardObservation {
        id: middle_id.to_owned(),
        bytes: payloads[middle_id].len() as u64,
        blake3: protocol.manifest().shards[2].blake3.clone(),
        all_finite: false,
    };
    assert_rejected(
        &mut protocol,
        non_finite_middle,
        VisionStackShardErrorCode::NonFinitePayload,
    );
    for id in [middle_id, "weights.vision_layer.02"] {
        protocol
            .accept_execution(&inspect_vision_stack_f32_shard(id, &payloads[id]))
            .unwrap();
    }

    let post_id = "weights.vision_post_norm";
    let mut mutated_post = payloads[post_id].clone();
    let final_byte = mutated_post.len() - 1;
    mutated_post[final_byte] ^= 1;
    assert_rejected(
        &mut protocol,
        inspect_vision_stack_f32_shard(post_id, &mutated_post),
        VisionStackShardErrorCode::DigestMismatch,
    );
    protocol
        .accept_execution(&inspect_vision_stack_f32_shard(post_id, &payloads[post_id]))
        .unwrap();
    assert_eq!(protocol.phase(), VisionStackShardProtocolPhase::Complete);
}

#[test]
fn official_27_layer_plan_proves_bounded_memory_without_allocating_the_model() {
    let manifest = official_manifest();
    let plan = manifest.plan().unwrap();
    assert_eq!(plan.layer_count, 27);
    assert_eq!(plan.shard_count, 29);
    assert_eq!(plan.input_bytes, 5_879_808);
    assert_eq!(plan.layer_weight_bytes, 60_958_016);
    assert_eq!(plan.post_norm_bytes, 9_216);
    assert_eq!(plan.transport_bytes, 1_651_755_456);
    assert_eq!(plan.activation_buffer_count, 13);
    assert_eq!(plan.activation_arena_bytes, 108_613_120);
    assert_eq!(plan.readback_bytes, 29_399_040);
    assert_eq!(plan.peak_gpu_data_bytes, 198_970_176);
    assert_eq!(plan.submission_count, 28);
    assert_eq!(plan.compute_pass_count, 28);
    assert_eq!(plan.dispatch_count, 325);
    assert!(plan.peak_gpu_data_bytes * 8 < plan.transport_bytes);
    assert!(
        plan.layer_weight_bytes < 64 * 1024 * 1024,
        "one streamed layer must remain a bounded browser payload",
    );
    canonical_vision_stack_shard_manifest_bytes(&manifest).unwrap();
}

#[test]
fn official_table_l2_profile_is_allowlisted_as_a_final_only_second_shape() {
    let manifest = table_l2_official_manifest();
    let plan = manifest.plan().unwrap();
    assert_eq!(plan.layer_count, 27);
    assert_eq!(plan.shard_count, 29);
    assert_eq!(plan.hidden_bytes, 8_017_920);
    assert_eq!(plan.intermediate_bytes, 29_955_840);
    assert_eq!(plan.layer_weight_bytes, 60_958_016);
    assert_eq!(plan.transport_bytes, 1_653_893_568);
    assert_eq!(plan.activation_arena_bytes, 148_108_800);
    assert_eq!(plan.readback_bytes, 8_017_920);
    assert_eq!(plan.peak_gpu_data_bytes, 217_084_736);
    assert_eq!(plan.submission_count, 28);
    assert_eq!(plan.compute_pass_count, 28);
    assert_eq!(plan.dispatch_count, 325);

    let canonical = canonical_vision_stack_shard_manifest_bytes(&manifest).unwrap();
    assert_eq!(
        parse_vision_stack_shard_manifest(&canonical).unwrap(),
        manifest
    );
}

#[test]
fn official_profiles_are_an_exact_identity_tuple_not_mix_and_match_fields() {
    type ManifestMutation = Box<dyn FnOnce(&mut VisionStackShardManifest)>;

    let table = table_l2_official_manifest();
    let l3 = official_manifest();
    let l3_bundle = l3.golden_bundle_digest.unwrap();
    let l3_semantic = l3.semantic_fingerprint.unwrap();
    let mutations: Vec<ManifestMutation> = vec![
        Box::new(|manifest| {
            manifest.tokens = 1_276;
            manifest.cu_seqlens = vec![0, 1_276];
        }),
        Box::new(|manifest| manifest.cu_seqlens = vec![0, 870, 1_740]),
        Box::new(|manifest| manifest.checkpoint_layers = vec![0, 1, 13, 26]),
        Box::new(|manifest| manifest.case_id = "ocr.clean_latin.0001/vision.stack.27".to_owned()),
        Box::new(move |manifest| manifest.golden_bundle_digest = Some(l3_bundle)),
        Box::new(move |manifest| manifest.semantic_fingerprint = Some(l3_semantic)),
    ];
    for mutate in mutations {
        assert_manifest_error(
            table.clone(),
            mutate,
            VisionStackShardErrorCode::OfficialIdentityMismatch,
        );
    }
}
