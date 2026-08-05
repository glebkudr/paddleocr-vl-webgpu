use std::{collections::BTreeMap, path::PathBuf};

use pvlc_pack::{
    OFFICIAL_VISION_LAYER_EXPECTED_BLAKE3, OFFICIAL_VISION_LAYER_EXPECTED_BYTES,
    OFFICIAL_VISION_LAYER_SELF_TEST_CASE_ID, OFFICIAL_VISION_LAYER_WEIGHTS_BLAKE3,
    OFFICIAL_VISION_LAYER_WEIGHTS_BYTES, PackBuilder, PackReader, PackSection, SectionKind,
    VISION_LAYER_SELF_TEST_DESCRIPTOR_ID, VISION_LAYER_SELF_TEST_EXPECTED_ID,
    VISION_LAYER_SELF_TEST_SCHEMA_VERSION, VISION_LAYER_SELF_TEST_WEIGHTS_ID,
    VisionLayerSelfTestErrorCode, VisionLayerSelfTestPack, VisionLayerSelfTestSource,
    build_vision_layer_self_test_pack,
};
use pvlc_runtime_core::{
    OwnedVisionEncoderLayerInvocation, OwnedVisionEncoderLayerParameters,
    OwnedVisionLayerNormParameters, OwnedVisionLinearParameters, VisionEncoderLayerStage,
};
use pvlc_safetensors::SafetensorsCatalog;
use pvlc_testkit::m3_vision_layer_corpus;

const COMPILER_BUILD: &str = "4e40a34cd64f347a14270d339ab61b131ae3120ca566ff140c81d101a3aa4a2c";
const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const GOLDEN_BUNDLE_DIGEST: &str =
    "blake3:35572ac07da5fddee97becb46fb866f906200f433f4c5677ad20fdf36440acf9";
const SEMANTIC_FINGERPRINT: &str =
    "blake3:632cbe2de6f47f8f84c764e3a551bf99f352d7459c98d939eb89725357f952b4";
const OFFICIAL_WEIGHTS_BLAKE3: &str =
    "8bbc38f130818d26cf7996e8c78055a022665e5772fee52c27d80a2efdd7d5c0";
const OFFICIAL_EXPECTED_BLAKE3: &str =
    "df6d0d65ea94559f68b36f0e8e85de364265d0fb719bff1eced439af11a63cba";
const OFFICIAL_WEIGHTS_BYTES: u64 = 66_837_824;
const OFFICIAL_EXPECTED_BYTES: u64 = 102_733_312;

type Checkpoints = BTreeMap<VisionEncoderLayerStage, Vec<f32>>;
type JsonMutation = Box<dyn Fn(&mut serde_json::Value)>;
type NamedJsonMutation = (&'static str, JsonMutation);

fn synthetic_fixture() -> (OwnedVisionEncoderLayerInvocation, Checkpoints) {
    let case = &m3_vision_layer_corpus().unwrap().cases[0];
    let invocation = case.invocation().unwrap();
    let expected = VisionEncoderLayerStage::ALL
        .into_iter()
        .map(|stage| (stage, case.expected.stage(stage).to_vec()))
        .collect();
    (invocation, expected)
}

fn build_synthetic_fixture() -> Vec<u8> {
    let (invocation, expected) = synthetic_fixture();
    build_vision_layer_self_test_pack(
        COMPILER_BUILD,
        VisionLayerSelfTestSource::synthetic(
            "synthetic.vision_layer/baseline",
            invocation.borrowed(),
            &expected,
        ),
    )
    .unwrap()
}

fn raw_weight_elements(invocation: &OwnedVisionEncoderLayerInvocation) -> usize {
    let tokens = invocation.tokens as usize;
    let hidden = invocation.hidden_size as usize;
    let intermediate = invocation.intermediate_size as usize;
    tokens * hidden + 4 * hidden * hidden + 2 * hidden * intermediate + 9 * hidden + intermediate
}

fn expected_elements(invocation: &OwnedVisionEncoderLayerInvocation) -> usize {
    let tokens = invocation.tokens as usize;
    let hidden = invocation.hidden_size as usize;
    let intermediate = invocation.intermediate_size as usize;
    10 * tokens * hidden + 2 * tokens * intermediate
}

fn canonical_json(value: &serde_json::Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(value).unwrap();
    bytes.push(b'\n');
    bytes
}

#[derive(Clone)]
struct SectionRecord {
    kind: SectionKind,
    alignment: u32,
    bytes: Vec<u8>,
}

fn sections(pack: &[u8]) -> (pvlc_pack::PackManifest, BTreeMap<String, SectionRecord>) {
    let reader = PackReader::open(pack).unwrap();
    let records = reader
        .entries()
        .iter()
        .map(|entry| {
            (
                entry.id.clone(),
                SectionRecord {
                    kind: entry.kind,
                    alignment: entry.alignment,
                    bytes: reader.section(&entry.id).unwrap().to_vec(),
                },
            )
        })
        .collect();
    (reader.manifest().clone(), records)
}

fn rebuild(manifest: pvlc_pack::PackManifest, records: BTreeMap<String, SectionRecord>) -> Vec<u8> {
    let mut builder = PackBuilder::new(manifest);
    for (id, record) in records {
        builder
            .add_section(PackSection::new(
                id,
                record.kind,
                record.alignment,
                record.bytes,
            ))
            .unwrap();
    }
    builder.build().unwrap()
}

fn mutate_descriptor(pack: &[u8], mutate: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    let (manifest, mut records) = sections(pack);
    let descriptor = &mut records
        .get_mut(VISION_LAYER_SELF_TEST_DESCRIPTOR_ID)
        .unwrap()
        .bytes;
    let mut value: serde_json::Value = serde_json::from_slice(descriptor).unwrap();
    mutate(&mut value);
    *descriptor = canonical_json(&value);
    rebuild(manifest, records)
}

fn mutate_payload(pack: &[u8], id: &str, mutate: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let (manifest, mut records) = sections(pack);
    mutate(&mut records.get_mut(id).unwrap().bytes);
    rebuild(manifest, records)
}

fn mutate_payload_and_reanchor(
    pack: &[u8],
    id: &str,
    mutate: impl FnOnce(&mut Vec<u8>),
) -> Vec<u8> {
    let (manifest, mut records) = sections(pack);
    let payload = &mut records.get_mut(id).unwrap().bytes;
    mutate(payload);
    let byte_len = payload.len() as u64;
    let digest = blake3::hash(payload).to_hex().to_string();
    let descriptor = &mut records
        .get_mut(VISION_LAYER_SELF_TEST_DESCRIPTOR_ID)
        .unwrap()
        .bytes;
    let mut value: serde_json::Value = serde_json::from_slice(descriptor).unwrap();
    let prefix = if id == VISION_LAYER_SELF_TEST_WEIGHTS_ID {
        "weights"
    } else {
        "expected"
    };
    value[format!("{prefix}_bytes")] = byte_len.into();
    value[format!("{prefix}_blake3")] = digest.into();
    *descriptor = canonical_json(&value);
    rebuild(manifest, records)
}

fn assert_f32_segment(bytes: &[u8], cursor: &mut usize, values: &[f32], label: &str) {
    for (index, value) in values.iter().enumerate() {
        let end = *cursor + 4;
        assert_eq!(
            &bytes[*cursor..end],
            &value.to_le_bytes(),
            "{label}[{index}] at byte {}",
            *cursor
        );
        *cursor = end;
    }
}

#[test]
fn binary_self_test_pack_roundtrips_every_operand_and_all_twelve_checkpoints() {
    let (invocation, expected) = synthetic_fixture();
    let first = build_synthetic_fixture();
    let second = build_synthetic_fixture();
    assert_eq!(first, second, "the browser artifact must be reproducible");

    let generic = PackReader::open(&first).unwrap();
    assert_eq!(generic.manifest().model_revision, MODEL_REVISION);
    assert_eq!(generic.manifest().compiler_build, COMPILER_BUILD);
    let mandatory = [
        (
            VISION_LAYER_SELF_TEST_DESCRIPTOR_ID,
            SectionKind::SemanticIr,
        ),
        (VISION_LAYER_SELF_TEST_EXPECTED_ID, SectionKind::SelfTest),
        (VISION_LAYER_SELF_TEST_WEIGHTS_ID, SectionKind::WeightShard),
    ];
    for (id, kind) in mandatory {
        let entry = generic
            .entries()
            .iter()
            .find(|entry| entry.id == id)
            .unwrap();
        assert_eq!(entry.kind, kind);
        assert_eq!(entry.offset % 4, 0, "{id} is not Float32Array aligned");
        assert_eq!(entry.offset % u64::from(entry.alignment), 0, "{id}");
    }

    let descriptor_bytes = generic
        .section(VISION_LAYER_SELF_TEST_DESCRIPTOR_ID)
        .unwrap();
    assert!(
        descriptor_bytes.len() < 4_096,
        "tensor data leaked into JSON"
    );
    assert_eq!(descriptor_bytes.last(), Some(&b'\n'));
    let descriptor_value: serde_json::Value = serde_json::from_slice(descriptor_bytes).unwrap();
    assert_eq!(descriptor_bytes, canonical_json(&descriptor_value));
    let descriptor_text = std::str::from_utf8(descriptor_bytes).unwrap();
    for forbidden in ["\"input\"", "\"parameters\"", "\"seed\"", "\"values\""] {
        assert!(
            !descriptor_text.contains(forbidden),
            "descriptor contains {forbidden}"
        );
    }

    let weights = generic.section(VISION_LAYER_SELF_TEST_WEIGHTS_ID).unwrap();
    let expected_bytes = generic.section(VISION_LAYER_SELF_TEST_EXPECTED_ID).unwrap();
    assert_eq!(weights.len(), raw_weight_elements(&invocation) * 4);
    assert_eq!(expected_bytes.len(), expected_elements(&invocation) * 4);

    let specialized = VisionLayerSelfTestPack::open(&first).unwrap();
    let descriptor = specialized.descriptor();
    assert_eq!(
        descriptor.schema_version,
        VISION_LAYER_SELF_TEST_SCHEMA_VERSION
    );
    assert_eq!(descriptor.case_id, "synthetic.vision_layer/baseline");
    assert_eq!(descriptor.model_revision, MODEL_REVISION);
    assert_eq!(descriptor.stage_order, VisionEncoderLayerStage::ALL);
    assert_eq!(descriptor.cu_seqlens, invocation.cu_seqlens);
    assert_eq!(specialized.invocation(), &invocation);
    specialized.invocation().borrowed().plan().unwrap();
    for stage in VisionEncoderLayerStage::ALL {
        assert_eq!(specialized.expected(stage), expected[&stage], "{stage:?}");
    }
}

#[test]
fn raw_sections_have_an_independently_proven_little_endian_operand_and_stage_order() {
    let (invocation, expected) = synthetic_fixture();
    let pack = build_synthetic_fixture();
    let reader = PackReader::open(&pack).unwrap();
    let weights = reader.section(VISION_LAYER_SELF_TEST_WEIGHTS_ID).unwrap();
    let parameters = &invocation.parameters;
    let mut cursor = 0;
    for (label, values) in [
        ("input", &invocation.input[..]),
        ("norm1.weight", &parameters.norm1.weight[..]),
        ("norm1.bias", &parameters.norm1.bias[..]),
        ("query.weight", &parameters.query.weight[..]),
        ("query.bias", &parameters.query.bias[..]),
        ("key.weight", &parameters.key.weight[..]),
        ("key.bias", &parameters.key.bias[..]),
        ("value.weight", &parameters.value.weight[..]),
        ("value.bias", &parameters.value.bias[..]),
        (
            "attention_output.weight",
            &parameters.attention_output.weight[..],
        ),
        (
            "attention_output.bias",
            &parameters.attention_output.bias[..],
        ),
        ("norm2.weight", &parameters.norm2.weight[..]),
        ("norm2.bias", &parameters.norm2.bias[..]),
        ("mlp_fc1.weight", &parameters.mlp_fc1.weight[..]),
        ("mlp_fc1.bias", &parameters.mlp_fc1.bias[..]),
        ("mlp_fc2.weight", &parameters.mlp_fc2.weight[..]),
        ("mlp_fc2.bias", &parameters.mlp_fc2.bias[..]),
    ] {
        assert_f32_segment(weights, &mut cursor, values, label);
    }
    assert_eq!(cursor, weights.len(), "unclaimed weight bytes remain");

    let checkpoint_bytes = reader.section(VISION_LAYER_SELF_TEST_EXPECTED_ID).unwrap();
    cursor = 0;
    for stage in VisionEncoderLayerStage::ALL {
        assert_f32_segment(
            checkpoint_bytes,
            &mut cursor,
            &expected[&stage],
            stage.as_str(),
        );
    }
    assert_eq!(
        cursor,
        checkpoint_bytes.len(),
        "unclaimed checkpoint bytes remain"
    );
}

#[test]
fn specialized_reader_rejects_identity_schema_layout_lengths_and_overflow() {
    let pristine = build_synthetic_fixture();
    let mutations: Vec<NamedJsonMutation> = vec![
        (
            "unknown field",
            Box::new(|value| value["shadow"] = 1.into()),
        ),
        (
            "schema",
            Box::new(|value| value["schema_version"] = 2.into()),
        ),
        (
            "revision",
            Box::new(|value| value["model_revision"] = "0".repeat(40).into()),
        ),
        (
            "case id",
            Box::new(|value| value["case_id"] = "../shadow".into()),
        ),
        (
            "geometry",
            Box::new(|value| value["hidden_size"] = 19.into()),
        ),
        (
            "overflow geometry",
            Box::new(|value| {
                value["tokens"] = u32::MAX.into();
                value["hidden_size"] = u32::MAX.into();
                value["intermediate_size"] = u32::MAX.into();
            }),
        ),
        (
            "boundaries",
            Box::new(|value| value["cu_seqlens"] = serde_json::json!([0, 8])),
        ),
        (
            "stage order",
            Box::new(|value| value["stage_order"][0] = "query".into()),
        ),
        (
            "weights u64 overflow",
            Box::new(|value| value["weights_bytes"] = u64::MAX.into()),
        ),
        (
            "expected u64 overflow",
            Box::new(|value| value["expected_bytes"] = u64::MAX.into()),
        ),
        (
            "weights digest",
            Box::new(|value| value["weights_blake3"] = "0".repeat(64).into()),
        ),
        (
            "expected digest",
            Box::new(|value| value["expected_blake3"] = "0".repeat(64).into()),
        ),
        (
            "false official provenance",
            Box::new(|value| value["oracle"] = "official_l3".into()),
        ),
    ];
    for (label, mutation) in mutations {
        let changed = mutate_descriptor(&pristine, mutation);
        assert!(
            VisionLayerSelfTestPack::open(&changed).is_err(),
            "descriptor mutation {label} was accepted"
        );
    }

    let noncanonical = mutate_payload(&pristine, VISION_LAYER_SELF_TEST_DESCRIPTOR_ID, |bytes| {
        bytes.push(b' ')
    });
    assert_eq!(
        VisionLayerSelfTestPack::open(&noncanonical)
            .unwrap_err()
            .code(),
        VisionLayerSelfTestErrorCode::InvalidDescriptor
    );
}

#[test]
fn specialized_reader_rejects_short_long_non_f32_and_digest_substituted_payloads() {
    let pristine = build_synthetic_fixture();
    for section in [
        VISION_LAYER_SELF_TEST_WEIGHTS_ID,
        VISION_LAYER_SELF_TEST_EXPECTED_ID,
    ] {
        for (label, mutation) in [
            ("short-f32", 0_i8),
            ("short-byte", 1),
            ("trailing-byte", 2),
            ("trailing-f32", 3),
        ] {
            let changed = mutate_payload(&pristine, section, |bytes| match mutation {
                0 => bytes.truncate(bytes.len() - 4),
                1 => bytes.truncate(bytes.len() - 1),
                2 => bytes.push(0),
                3 => bytes.extend_from_slice(&0_f32.to_le_bytes()),
                _ => unreachable!(),
            });
            assert_eq!(
                VisionLayerSelfTestPack::open(&changed).unwrap_err().code(),
                VisionLayerSelfTestErrorCode::LengthMismatch,
                "{section} {label}"
            );
        }

        let corrupt = mutate_payload(&pristine, section, |bytes| {
            let offset = bytes.len() / 2 / 4 * 4;
            bytes[offset] ^= 1;
        });
        assert_eq!(
            VisionLayerSelfTestPack::open(&corrupt).unwrap_err().code(),
            VisionLayerSelfTestErrorCode::DigestMismatch,
            "{section}"
        );
    }
}

#[test]
fn specialized_reader_revalidates_semantics_after_inner_and_outer_hashes_are_rebuilt() {
    let pristine = build_synthetic_fixture();
    let nonfinite_weights =
        mutate_payload_and_reanchor(&pristine, VISION_LAYER_SELF_TEST_WEIGHTS_ID, |bytes| {
            bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes())
        });
    assert_eq!(
        VisionLayerSelfTestPack::open(&nonfinite_weights)
            .unwrap_err()
            .code(),
        VisionLayerSelfTestErrorCode::InvalidInvocation
    );

    let nonfinite_expected =
        mutate_payload_and_reanchor(&pristine, VISION_LAYER_SELF_TEST_EXPECTED_ID, |bytes| {
            let start = bytes.len() - 4;
            bytes[start..].copy_from_slice(&f32::INFINITY.to_le_bytes());
        });
    assert_eq!(
        VisionLayerSelfTestPack::open(&nonfinite_expected)
            .unwrap_err()
            .code(),
        VisionLayerSelfTestErrorCode::InvalidCheckpoint
    );
}

#[test]
fn specialized_reader_requires_mandatory_section_kinds_but_allows_future_sections() {
    let pristine = build_synthetic_fixture();
    for missing in [
        VISION_LAYER_SELF_TEST_DESCRIPTOR_ID,
        VISION_LAYER_SELF_TEST_WEIGHTS_ID,
        VISION_LAYER_SELF_TEST_EXPECTED_ID,
    ] {
        let (manifest, mut records) = sections(&pristine);
        records.remove(missing);
        assert_eq!(
            VisionLayerSelfTestPack::open(&rebuild(manifest, records))
                .unwrap_err()
                .code(),
            VisionLayerSelfTestErrorCode::MissingSection,
            "{missing}"
        );
    }

    for (id, wrong_kind) in [
        (VISION_LAYER_SELF_TEST_DESCRIPTOR_ID, SectionKind::SelfTest),
        (VISION_LAYER_SELF_TEST_WEIGHTS_ID, SectionKind::SelfTest),
        (VISION_LAYER_SELF_TEST_EXPECTED_ID, SectionKind::WeightShard),
    ] {
        let (manifest, mut records) = sections(&pristine);
        records.get_mut(id).unwrap().kind = wrong_kind;
        assert_eq!(
            VisionLayerSelfTestPack::open(&rebuild(manifest, records))
                .unwrap_err()
                .code(),
            VisionLayerSelfTestErrorCode::WrongSectionKind,
            "{id}"
        );
    }

    let (manifest, mut records) = sections(&pristine);
    records.insert(
        "schema.future".into(),
        SectionRecord {
            kind: SectionKind::ModelSchema,
            alignment: 16,
            bytes: b"{}\n".to_vec(),
        },
    );
    VisionLayerSelfTestPack::open(&rebuild(manifest, records)).unwrap();
}

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load_tensor(catalog: &SafetensorsCatalog, name: &str, shape: &[u64]) -> Vec<f32> {
    assert_eq!(catalog.tensor(name).unwrap().shape, shape);
    catalog.load_tensor_f32(name).unwrap()
}

fn official_fixture() -> (OwnedVisionEncoderLayerInvocation, Checkpoints) {
    const TOKENS: u64 = 1_276;
    const HIDDEN: u64 = 1_152;
    const INTERMEDIATE: u64 = 4_304;
    const PREFIX: &str = "visual.vision_model.encoder.layers.0";
    let model = SafetensorsCatalog::open(
        repository()
            .join("models/snapshots")
            .join(MODEL_REVISION)
            .join("model.safetensors"),
    )
    .unwrap();
    let deep = SafetensorsCatalog::open(
        repository()
            .join("artifacts/goldens/ocr.clean_latin.0001-l3")
            .join("deep-checkpoints.safetensors"),
    )
    .unwrap();
    let linear =
        |weight: &str, bias: &str, shape: &[u64], bias_shape: &[u64]| OwnedVisionLinearParameters {
            weight: load_tensor(&model, &format!("{PREFIX}.{weight}"), shape),
            bias: load_tensor(&model, &format!("{PREFIX}.{bias}"), bias_shape),
        };
    let invocation = OwnedVisionEncoderLayerInvocation {
        tokens: TOKENS as u32,
        hidden_size: HIDDEN as u32,
        attention_heads: 16,
        head_dim: 72,
        intermediate_size: INTERMEDIATE as u32,
        layer_norm_epsilon: 1.0e-6,
        input: load_tensor(&deep, "vision.embeddings.output", &[1, TOKENS, HIDDEN]),
        cu_seqlens: vec![0, TOKENS as u32],
        parameters: OwnedVisionEncoderLayerParameters {
            norm1: OwnedVisionLayerNormParameters {
                weight: load_tensor(&model, &format!("{PREFIX}.layer_norm1.weight"), &[HIDDEN]),
                bias: load_tensor(&model, &format!("{PREFIX}.layer_norm1.bias"), &[HIDDEN]),
            },
            query: linear(
                "self_attn.q_proj.weight",
                "self_attn.q_proj.bias",
                &[HIDDEN, HIDDEN],
                &[HIDDEN],
            ),
            key: linear(
                "self_attn.k_proj.weight",
                "self_attn.k_proj.bias",
                &[HIDDEN, HIDDEN],
                &[HIDDEN],
            ),
            value: linear(
                "self_attn.v_proj.weight",
                "self_attn.v_proj.bias",
                &[HIDDEN, HIDDEN],
                &[HIDDEN],
            ),
            attention_output: linear(
                "self_attn.out_proj.weight",
                "self_attn.out_proj.bias",
                &[HIDDEN, HIDDEN],
                &[HIDDEN],
            ),
            norm2: OwnedVisionLayerNormParameters {
                weight: load_tensor(&model, &format!("{PREFIX}.layer_norm2.weight"), &[HIDDEN]),
                bias: load_tensor(&model, &format!("{PREFIX}.layer_norm2.bias"), &[HIDDEN]),
            },
            mlp_fc1: linear(
                "mlp.fc1.weight",
                "mlp.fc1.bias",
                &[INTERMEDIATE, HIDDEN],
                &[INTERMEDIATE],
            ),
            mlp_fc2: linear(
                "mlp.fc2.weight",
                "mlp.fc2.bias",
                &[HIDDEN, INTERMEDIATE],
                &[HIDDEN],
            ),
        },
    };
    let names = [
        "vision.layer.00.norm1",
        "vision.layer.00.q",
        "vision.layer.00.k",
        "vision.layer.00.v",
        "vision.layer.00.attention.context",
        "vision.layer.00.attention.output",
        "vision.layer.00.attention.residual",
        "vision.layer.00.norm2",
        "vision.layer.00.mlp.fc1",
        "vision.layer.00.mlp.activation",
        "vision.layer.00.mlp.output",
        "vision.layer.00.output",
    ];
    let expected = VisionEncoderLayerStage::ALL
        .into_iter()
        .zip(names)
        .map(|(stage, name)| {
            let width = if matches!(
                stage,
                VisionEncoderLayerStage::MlpFc1 | VisionEncoderLayerStage::MlpActivation
            ) {
                INTERMEDIATE
            } else {
                HIDDEN
            };
            (stage, load_tensor(&deep, name, &[1, TOKENS, width]))
        })
        .collect();
    invocation.borrowed().plan().unwrap();
    (invocation, expected)
}

#[test]
#[ignore = "M3g official browser-pack gate materializes the pinned 161.7 MiB F32 artifact"]
fn official_full_size_pack_is_anchored_to_the_pinned_model_and_every_l3_checkpoint() {
    assert_eq!(std::env::var("PVLC_REQUIRE_MODEL").as_deref(), Ok("1"));
    assert_eq!(
        OFFICIAL_VISION_LAYER_SELF_TEST_CASE_ID,
        "ocr.clean_latin.0001/vision.layer.00"
    );
    assert_eq!(OFFICIAL_VISION_LAYER_WEIGHTS_BYTES, OFFICIAL_WEIGHTS_BYTES);
    assert_eq!(
        OFFICIAL_VISION_LAYER_EXPECTED_BYTES,
        OFFICIAL_EXPECTED_BYTES
    );
    assert_eq!(
        OFFICIAL_VISION_LAYER_WEIGHTS_BLAKE3,
        OFFICIAL_WEIGHTS_BLAKE3
    );
    assert_eq!(
        OFFICIAL_VISION_LAYER_EXPECTED_BLAKE3,
        OFFICIAL_EXPECTED_BLAKE3
    );

    let (invocation, expected) = official_fixture();
    let build = || {
        build_vision_layer_self_test_pack(
            COMPILER_BUILD,
            VisionLayerSelfTestSource::official_layer_zero(invocation.borrowed(), &expected),
        )
        .unwrap()
    };
    let pack = build();
    let repeated = build();
    assert_eq!(pack.len(), repeated.len());
    assert_eq!(blake3::hash(&pack), blake3::hash(&repeated));
    assert!(
        pack == repeated,
        "official pack bytes are not deterministic"
    );
    drop(repeated);
    let reader = PackReader::open(&pack).unwrap();
    let descriptor_bytes = reader
        .section(VISION_LAYER_SELF_TEST_DESCRIPTOR_ID)
        .unwrap();
    assert!(
        descriptor_bytes.len() < 4_096,
        "official tensor data leaked into its descriptor"
    );
    assert_eq!(descriptor_bytes.last(), Some(&b'\n'));
    let descriptor_value: serde_json::Value = serde_json::from_slice(descriptor_bytes).unwrap();
    assert_eq!(descriptor_bytes, canonical_json(&descriptor_value));
    assert_eq!(
        descriptor_value,
        serde_json::json!({
            "attention_heads": 16,
            "case_id": OFFICIAL_VISION_LAYER_SELF_TEST_CASE_ID,
            "cu_seqlens": [0, 1_276],
            "expected_blake3": OFFICIAL_EXPECTED_BLAKE3,
            "expected_bytes": OFFICIAL_EXPECTED_BYTES,
            "golden_bundle_digest": GOLDEN_BUNDLE_DIGEST,
            "head_dim": 72,
            "hidden_size": 1_152,
            "intermediate_size": 4_304,
            "layer_norm_epsilon": f64::from(1.0e-6_f32),
            "model_revision": MODEL_REVISION,
            "oracle": "official_l3",
            "schema_version": VISION_LAYER_SELF_TEST_SCHEMA_VERSION,
            "semantic_fingerprint": SEMANTIC_FINGERPRINT,
            "stage_order": [
                "norm1", "query", "key", "value", "attention_context",
                "attention_output", "attention_residual", "norm2", "mlp_fc1",
                "mlp_activation", "mlp_output", "output"
            ],
            "tokens": 1_276,
            "weights_blake3": OFFICIAL_WEIGHTS_BLAKE3,
            "weights_bytes": OFFICIAL_WEIGHTS_BYTES,
        })
    );
    let descriptor_text = std::str::from_utf8(descriptor_bytes).unwrap();
    for forbidden in ["\"input\"", "\"parameters\"", "\"seed\"", "\"values\""] {
        assert!(
            !descriptor_text.contains(forbidden),
            "official descriptor contains {forbidden}"
        );
    }
    assert_eq!(
        reader
            .section(VISION_LAYER_SELF_TEST_WEIGHTS_ID)
            .unwrap()
            .len() as u64,
        OFFICIAL_WEIGHTS_BYTES
    );
    assert_eq!(
        reader
            .section(VISION_LAYER_SELF_TEST_EXPECTED_ID)
            .unwrap()
            .len() as u64,
        OFFICIAL_EXPECTED_BYTES
    );
    assert_eq!(
        blake3::hash(reader.section(VISION_LAYER_SELF_TEST_WEIGHTS_ID).unwrap())
            .to_hex()
            .as_str(),
        OFFICIAL_WEIGHTS_BLAKE3
    );
    assert_eq!(
        blake3::hash(reader.section(VISION_LAYER_SELF_TEST_EXPECTED_ID).unwrap())
            .to_hex()
            .as_str(),
        OFFICIAL_EXPECTED_BLAKE3
    );
    let specialized = VisionLayerSelfTestPack::open(&pack).unwrap();
    assert_eq!(
        specialized.descriptor().case_id,
        OFFICIAL_VISION_LAYER_SELF_TEST_CASE_ID
    );
    assert_eq!(
        specialized.descriptor().golden_bundle_digest.as_deref(),
        Some(GOLDEN_BUNDLE_DIGEST)
    );
    assert_eq!(
        specialized.descriptor().semantic_fingerprint.as_deref(),
        Some(SEMANTIC_FINGERPRINT)
    );
    for stage in VisionEncoderLayerStage::ALL {
        assert_eq!(specialized.expected(stage), expected[&stage], "{stage:?}");
    }
    drop(specialized);
    drop(reader);

    let substituted_weights =
        mutate_payload_and_reanchor(&pack, VISION_LAYER_SELF_TEST_WEIGHTS_ID, |bytes| {
            let original = f32::from_le_bytes(bytes[..4].try_into().unwrap());
            bytes[..4].copy_from_slice(&(original + 1.0).to_le_bytes());
        });
    assert_eq!(
        VisionLayerSelfTestPack::open(&substituted_weights)
            .unwrap_err()
            .code(),
        VisionLayerSelfTestErrorCode::OfficialPayloadMismatch
    );
    drop(substituted_weights);
    let substituted_expected =
        mutate_payload_and_reanchor(&pack, VISION_LAYER_SELF_TEST_EXPECTED_ID, |bytes| {
            let original = f32::from_le_bytes(bytes[..4].try_into().unwrap());
            bytes[..4].copy_from_slice(&(original + 1.0).to_le_bytes());
        });
    assert_eq!(
        VisionLayerSelfTestPack::open(&substituted_expected)
            .unwrap_err()
            .code(),
        VisionLayerSelfTestErrorCode::OfficialPayloadMismatch
    );
    drop(substituted_expected);

    let identity_mutations: Vec<JsonMutation> = vec![
        Box::new(|value| value["case_id"] = "synthetic.vision_layer/baseline".into()),
        Box::new(|value| value["oracle"] = "synthetic".into()),
        Box::new(|value| value["model_revision"] = "0".repeat(40).into()),
        Box::new(|value| {
            value["golden_bundle_digest"] = format!("blake3:{}", "0".repeat(64)).into();
        }),
        Box::new(|value| {
            value["semantic_fingerprint"] = format!("blake3:{}", "0".repeat(64)).into();
        }),
        Box::new(|value| value["tokens"] = 1_275.into()),
        Box::new(|value| value["hidden_size"] = 1_153.into()),
        Box::new(|value| value["attention_heads"] = 15.into()),
        Box::new(|value| value["head_dim"] = 71.into()),
        Box::new(|value| value["intermediate_size"] = 4_303.into()),
        Box::new(|value| {
            value["layer_norm_epsilon"] = f64::from(1.0e-5_f32).into();
        }),
        Box::new(|value| value["cu_seqlens"] = serde_json::json!([0, 1_275])),
    ];
    for mutation in identity_mutations {
        let changed = mutate_descriptor(&pack, mutation);
        assert_eq!(
            VisionLayerSelfTestPack::open(&changed).unwrap_err().code(),
            VisionLayerSelfTestErrorCode::OfficialIdentityMismatch
        );
    }
    let wrong_schema = mutate_descriptor(&pack, |value| value["schema_version"] = 2.into());
    assert_eq!(
        VisionLayerSelfTestPack::open(&wrong_schema)
            .unwrap_err()
            .code(),
        VisionLayerSelfTestErrorCode::InvalidDescriptor
    );
}
