use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use pvlc_cpu_ref::{
    LayerNormParameters as CpuLayerNormParameters, LinearParameters as CpuLinearParameters,
    ProjectorParameters as CpuProjectorParameters, projector_f32,
};
use pvlc_pack::{
    OFFICIAL_PROJECTOR_L2_EXPECTED_BLAKE3, OFFICIAL_PROJECTOR_L2_EXPECTED_BYTES,
    OFFICIAL_PROJECTOR_L2_EXPECTED_ID, OFFICIAL_PROJECTOR_L2_INPUT_BLAKE3,
    OFFICIAL_PROJECTOR_L2_INPUT_BYTES, OFFICIAL_PROJECTOR_L2_INPUT_ID,
    OFFICIAL_PROJECTOR_L2_PROFILE, OFFICIAL_PROJECTOR_L3_EXPECTED_BLAKE3,
    OFFICIAL_PROJECTOR_L3_EXPECTED_BYTES, OFFICIAL_PROJECTOR_L3_EXPECTED_ID,
    OFFICIAL_PROJECTOR_L3_INPUT_BLAKE3, OFFICIAL_PROJECTOR_L3_INPUT_BYTES,
    OFFICIAL_PROJECTOR_L3_INPUT_ID, OFFICIAL_PROJECTOR_L3_PROFILE,
    OFFICIAL_PROJECTOR_WEIGHTS_BLAKE3, OFFICIAL_PROJECTOR_WEIGHTS_BYTES,
    PROJECTOR_SELF_TEST_DESCRIPTOR_ID, PROJECTOR_SELF_TEST_SCHEMA_VERSION,
    PROJECTOR_SELF_TEST_WEIGHTS_ID, PackBuilder, PackReader, PackSection,
    ProjectorSelfTestCaseSource, ProjectorSelfTestErrorCode, ProjectorSelfTestOracle,
    ProjectorSelfTestPack, ProjectorSelfTestSource, SectionKind, build_projector_self_test_pack,
};
use pvlc_runtime_core::{
    OwnedProjectorParameters, OwnedVisionLayerNormParameters, OwnedVisionLinearParameters,
    ProjectorReadback, ProjectorStage,
};
use pvlc_safetensors::SafetensorsCatalog;

const COMPILER_BUILD: &str = "4e40a34cd64f347a14270d339ab61b131ae3120ca566ff140c81d101a3aa4a2c";
const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const HIDDEN: usize = 3;
const MERGED: usize = HIDDEN * 4;
const OUTPUT: usize = 5;
const EPSILON: f32 = 1.0e-5;
const L3_BUNDLE: &str = "blake3:35572ac07da5fddee97becb46fb866f906200f433f4c5677ad20fdf36440acf9";
const L3_SEMANTIC: &str = "blake3:632cbe2de6f47f8f84c764e3a551bf99f352d7459c98d939eb89725357f952b4";
const L2_BUNDLE: &str = "blake3:4cecc8b2abc37030920acf8860bb317084125dbbbcae5af394fccd0c8044c842";
const L2_SEMANTIC: &str = "blake3:91f72ffdf81070c2c9ca31628605064bdc02bf19417482cefce41d6a27aa5404";
const EXPECTED_DESCRIPTOR_ID: &str = "ir.projector.official";
const EXPECTED_WEIGHTS_ID: &str = "weights.projector";
const EXPECTED_L3_PROFILE: &str = "ocr-clean-latin-l3";
const EXPECTED_L2_PROFILE: &str = "table-simple-l2";
const EXPECTED_L3_CASE_ID: &str = "ocr.clean_latin.0001/projector";
const EXPECTED_L2_CASE_ID: &str = "table.simple.0001/projector";
const EXPECTED_L3_INPUT_ID: &str = "input.projector.ocr-clean-latin-l3";
const EXPECTED_L2_INPUT_ID: &str = "input.projector.table-simple-l2";
const EXPECTED_L3_CHECKPOINT_ID: &str = "self_test.projector.ocr-clean-latin-l3";
const EXPECTED_L2_CHECKPOINT_ID: &str = "self_test.projector.table-simple-l2";
const EXPECTED_WEIGHTS_BYTES: u64 = 103_840_768;
const EXPECTED_WEIGHTS_BLAKE3: &str =
    "bca2e52ec0a24bb3643141aa467bb265b2617b6ebbc5591c884c93351cf08a64";
const EXPECTED_L3_INPUT_BYTES: u64 = 5_879_808;
const EXPECTED_L3_INPUT_BLAKE3: &str =
    "fd6bbb0ccc67ba679f5c06c0fbb4074f8970bb54aa4115b3b39a814bfff48663";
const EXPECTED_L2_INPUT_BYTES: u64 = 8_017_920;
const EXPECTED_L2_INPUT_BLAKE3: &str =
    "fcd101b25a04e1b4e0984e5d712094630f11c22d4fc57abdf743e9fd7a79aed9";
const EXPECTED_L3_CHECKPOINT_BYTES: u64 = 24_825_856;
const EXPECTED_L3_CHECKPOINT_BLAKE3: &str =
    "cc27c875fdeb691a44a466cf7a4fd3fdb9cf9c218e53236a59f968abbb92ad88";
const EXPECTED_L2_CHECKPOINT_BYTES: u64 = 1_781_760;
const EXPECTED_L2_CHECKPOINT_BLAKE3: &str =
    "e31d337fd75b3ae5a95edc7c3a7ae88d5c0433e6549e82533cafcbbe9f12aac7";

type Checkpoints = BTreeMap<ProjectorStage, Vec<f32>>;
type DescriptorMutation = Box<dyn Fn(&mut serde_json::Value)>;

fn values(length: usize, seed: u32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let residue = (index as u32 * 17 + seed * 13) % 97;
            (residue as i32 - 48) as f32 / 64.0
        })
        .collect()
}

fn parameters() -> OwnedProjectorParameters {
    OwnedProjectorParameters {
        pre_norm: OwnedVisionLayerNormParameters {
            weight: values(HIDDEN, 11)
                .into_iter()
                .map(|value| value + 1.0)
                .collect(),
            bias: values(HIDDEN, 12),
        },
        linear1: OwnedVisionLinearParameters {
            weight: values(MERGED * MERGED, 13),
            bias: values(MERGED, 14),
        },
        linear2: OwnedVisionLinearParameters {
            weight: values(OUTPUT * MERGED, 15),
            bias: values(OUTPUT, 16),
        },
    }
}

fn cpu_checkpoints(
    input: &[f32],
    grids: &[[u32; 3]],
    parameters: &OwnedProjectorParameters,
) -> Checkpoints {
    let grids = grids
        .iter()
        .map(|grid| grid.map(|dimension| dimension as usize))
        .collect::<Vec<_>>();
    let trace = projector_f32(
        input,
        HIDDEN,
        &grids,
        CpuProjectorParameters {
            pre_norm: CpuLayerNormParameters {
                weight: &parameters.pre_norm.weight,
                bias: &parameters.pre_norm.bias,
            },
            linear1: CpuLinearParameters {
                weight: &parameters.linear1.weight,
                bias: &parameters.linear1.bias,
            },
            linear2: CpuLinearParameters {
                weight: &parameters.linear2.weight,
                bias: &parameters.linear2.bias,
            },
        },
        EPSILON,
    )
    .unwrap();
    BTreeMap::from([
        (ProjectorStage::PreNorm, trace.pre_norm),
        (ProjectorStage::Merge, trace.merged),
        (ProjectorStage::Linear1, trace.linear1),
        (ProjectorStage::Activation, trace.activation),
        (ProjectorStage::Linear2, trace.output),
    ])
}

struct SyntheticFixture {
    parameters: OwnedProjectorParameters,
    l3_input: Vec<f32>,
    l3_grids: Vec<[u32; 3]>,
    l3_expected: Checkpoints,
    l2_input: Vec<f32>,
    l2_grids: Vec<[u32; 3]>,
    l2_expected: Checkpoints,
}

fn synthetic_fixture() -> SyntheticFixture {
    let parameters = parameters();
    let l3_grids = vec![[1, 2, 4], [2, 2, 2]];
    let l3_input = values(16 * HIDDEN, 1);
    let l3_expected = cpu_checkpoints(&l3_input, &l3_grids, &parameters);
    let l2_grids = vec![[1, 2, 2]];
    let l2_input = values(4 * HIDDEN, 2);
    let mut l2_expected = cpu_checkpoints(&l2_input, &l2_grids, &parameters);
    l2_expected.retain(|stage, _| *stage == ProjectorStage::Linear2);
    SyntheticFixture {
        parameters,
        l3_input,
        l3_grids,
        l3_expected,
        l2_input,
        l2_grids,
        l2_expected,
    }
}

fn build_synthetic_fixture() -> Vec<u8> {
    let fixture = synthetic_fixture();
    let cases = [
        ProjectorSelfTestCaseSource::synthetic(
            "synthetic-l3",
            "synthetic.projector/l3",
            &fixture.l3_input,
            &fixture.l3_grids,
            ProjectorReadback::AllStages,
            &fixture.l3_expected,
        ),
        ProjectorSelfTestCaseSource::synthetic(
            "synthetic-l2",
            "synthetic.projector/l2",
            &fixture.l2_input,
            &fixture.l2_grids,
            ProjectorReadback::OutputOnly,
            &fixture.l2_expected,
        ),
    ];
    build_projector_self_test_pack(
        COMPILER_BUILD,
        ProjectorSelfTestSource::synthetic(
            HIDDEN as u32,
            OUTPUT as u32,
            EPSILON,
            fixture.parameters.borrowed(),
            &cases,
        ),
    )
    .unwrap()
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

fn canonical_json(value: &serde_json::Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(value).unwrap();
    bytes.push(b'\n');
    bytes
}

fn mutate_descriptor(pack: &[u8], mutate: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    let (manifest, mut records) = sections(pack);
    let descriptor = &mut records
        .get_mut(PROJECTOR_SELF_TEST_DESCRIPTOR_ID)
        .unwrap()
        .bytes;
    let mut value: serde_json::Value = serde_json::from_slice(descriptor).unwrap();
    mutate(&mut value);
    *descriptor = canonical_json(&value);
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
    let bytes = payload.len() as u64;
    let digest = blake3::hash(payload).to_hex().to_string();
    let descriptor = &mut records
        .get_mut(PROJECTOR_SELF_TEST_DESCRIPTOR_ID)
        .unwrap()
        .bytes;
    let mut value: serde_json::Value = serde_json::from_slice(descriptor).unwrap();
    if id == PROJECTOR_SELF_TEST_WEIGHTS_ID {
        value["weights"]["bytes"] = bytes.into();
        value["weights"]["blake3"] = digest.into();
    } else {
        let cases = value["cases"].as_array_mut().unwrap();
        let (case, field) = cases
            .iter_mut()
            .find_map(|case| {
                for field in ["input", "expected"] {
                    if case[field]["section_id"].as_str() == Some(id) {
                        return Some((case, field));
                    }
                }
                None
            })
            .unwrap();
        case[field]["bytes"] = bytes.into();
        case[field]["blake3"] = digest.into();
    }
    *descriptor = canonical_json(&value);
    rebuild(manifest, records)
}

fn assert_f32_segment(bytes: &[u8], cursor: &mut usize, values: &[f32], label: &str) {
    for (index, value) in values.iter().enumerate() {
        let end = *cursor + 4;
        assert_eq!(
            &bytes[*cursor..end],
            &value.to_le_bytes(),
            "{label}[{index}]"
        );
        *cursor = end;
    }
}

#[test]
fn exported_official_projector_anchors_equal_independent_literal_oracles() {
    assert_eq!(PROJECTOR_SELF_TEST_DESCRIPTOR_ID, EXPECTED_DESCRIPTOR_ID);
    assert_eq!(PROJECTOR_SELF_TEST_WEIGHTS_ID, EXPECTED_WEIGHTS_ID);
    assert_eq!(OFFICIAL_PROJECTOR_L3_PROFILE, EXPECTED_L3_PROFILE);
    assert_eq!(OFFICIAL_PROJECTOR_L2_PROFILE, EXPECTED_L2_PROFILE);
    assert_eq!(OFFICIAL_PROJECTOR_L3_INPUT_ID, EXPECTED_L3_INPUT_ID);
    assert_eq!(OFFICIAL_PROJECTOR_L2_INPUT_ID, EXPECTED_L2_INPUT_ID);
    assert_eq!(OFFICIAL_PROJECTOR_L3_EXPECTED_ID, EXPECTED_L3_CHECKPOINT_ID);
    assert_eq!(OFFICIAL_PROJECTOR_L2_EXPECTED_ID, EXPECTED_L2_CHECKPOINT_ID);
    assert_eq!(OFFICIAL_PROJECTOR_WEIGHTS_BYTES, EXPECTED_WEIGHTS_BYTES);
    assert_eq!(OFFICIAL_PROJECTOR_WEIGHTS_BLAKE3, EXPECTED_WEIGHTS_BLAKE3);
    assert_eq!(OFFICIAL_PROJECTOR_L3_INPUT_BYTES, EXPECTED_L3_INPUT_BYTES);
    assert_eq!(OFFICIAL_PROJECTOR_L3_INPUT_BLAKE3, EXPECTED_L3_INPUT_BLAKE3);
    assert_eq!(OFFICIAL_PROJECTOR_L2_INPUT_BYTES, EXPECTED_L2_INPUT_BYTES);
    assert_eq!(OFFICIAL_PROJECTOR_L2_INPUT_BLAKE3, EXPECTED_L2_INPUT_BLAKE3);
    assert_eq!(
        OFFICIAL_PROJECTOR_L3_EXPECTED_BYTES,
        EXPECTED_L3_CHECKPOINT_BYTES
    );
    assert_eq!(
        OFFICIAL_PROJECTOR_L3_EXPECTED_BLAKE3,
        EXPECTED_L3_CHECKPOINT_BLAKE3
    );
    assert_eq!(
        OFFICIAL_PROJECTOR_L2_EXPECTED_BYTES,
        EXPECTED_L2_CHECKPOINT_BYTES
    );
    assert_eq!(
        OFFICIAL_PROJECTOR_L2_EXPECTED_BLAKE3,
        EXPECTED_L2_CHECKPOINT_BLAKE3
    );
}

#[test]
fn binary_projector_pack_is_deterministic_shared_and_roundtrips_both_readback_modes() {
    let fixture = synthetic_fixture();
    let first = build_synthetic_fixture();
    let second = build_synthetic_fixture();
    assert_eq!(
        first, second,
        "projector self-test pack is not reproducible"
    );

    let generic = PackReader::open(&first).unwrap();
    assert_eq!(generic.manifest().model_revision, MODEL_REVISION);
    assert_eq!(generic.manifest().compiler_build, COMPILER_BUILD);
    let exact_layout = [
        (
            "input.projector.synthetic-l2",
            SectionKind::WeightShard,
            256,
        ),
        (
            "input.projector.synthetic-l3",
            SectionKind::WeightShard,
            256,
        ),
        ("ir.projector.official", SectionKind::SemanticIr, 64),
        (
            "self_test.projector.synthetic-l2",
            SectionKind::SelfTest,
            256,
        ),
        (
            "self_test.projector.synthetic-l3",
            SectionKind::SelfTest,
            256,
        ),
        ("weights.projector", SectionKind::WeightShard, 256),
    ];
    assert_eq!(
        generic
            .entries()
            .iter()
            .map(|entry| (entry.id.as_str(), entry.kind, entry.alignment))
            .collect::<Vec<_>>(),
        exact_layout,
    );
    for entry in generic.entries() {
        assert_eq!(entry.offset % u64::from(entry.alignment), 0, "{}", entry.id);
        if entry.kind != SectionKind::SemanticIr {
            assert_eq!(
                entry.offset % 4,
                0,
                "{} is not Float32Array aligned",
                entry.id
            );
            assert_eq!(
                entry.byte_len % 4,
                0,
                "{} is not a complete F32 payload",
                entry.id
            );
        }
    }
    assert_eq!(
        generic
            .entries()
            .iter()
            .filter(|entry| entry.id == PROJECTOR_SELF_TEST_WEIGHTS_ID)
            .count(),
        1,
        "shared projector weights were duplicated per profile",
    );

    let descriptor_bytes = generic.section(PROJECTOR_SELF_TEST_DESCRIPTOR_ID).unwrap();
    assert!(
        descriptor_bytes.len() < 8_192,
        "tensor data leaked into descriptor JSON"
    );
    assert_eq!(descriptor_bytes.last(), Some(&b'\n'));
    let descriptor_value: serde_json::Value = serde_json::from_slice(descriptor_bytes).unwrap();
    assert_eq!(descriptor_bytes, canonical_json(&descriptor_value));
    for forbidden in ["\"values\"", "\"parameters\"", "\"seed\""] {
        assert!(
            !std::str::from_utf8(descriptor_bytes)
                .unwrap()
                .contains(forbidden)
        );
    }

    let specialized = ProjectorSelfTestPack::open(&first).unwrap();
    assert_eq!(
        specialized.descriptor().schema_version,
        PROJECTOR_SELF_TEST_SCHEMA_VERSION
    );
    assert_eq!(
        specialized.descriptor().oracle,
        ProjectorSelfTestOracle::Synthetic
    );
    assert_eq!(specialized.descriptor().cases.len(), 2);
    assert_eq!(specialized.parameters(), &fixture.parameters);

    let l3 = specialized.invocation("synthetic-l3").unwrap();
    assert_eq!(l3.input, fixture.l3_input);
    assert_eq!(l3.image_grid_thw, fixture.l3_grids);
    l3.plan().unwrap();
    for stage in ProjectorStage::ALL {
        assert_eq!(
            specialized.expected("synthetic-l3", stage).unwrap(),
            fixture.l3_expected[&stage]
        );
    }
    let l2 = specialized.invocation("synthetic-l2").unwrap();
    assert_eq!(l2.input, fixture.l2_input);
    assert_eq!(l2.image_grid_thw, fixture.l2_grids);
    l2.plan().unwrap();
    assert_eq!(
        specialized
            .expected("synthetic-l2", ProjectorStage::Linear2)
            .unwrap(),
        fixture.l2_expected[&ProjectorStage::Linear2],
    );
    for stage in ProjectorStage::ALL
        .into_iter()
        .filter(|stage| *stage != ProjectorStage::Linear2)
    {
        assert!(specialized.expected("synthetic-l2", stage).is_none());
    }
}

#[test]
fn raw_sections_have_independently_proven_operand_input_and_checkpoint_order() {
    let fixture = synthetic_fixture();
    let pack = build_synthetic_fixture();
    let reader = PackReader::open(&pack).unwrap();
    let weights = reader.section(PROJECTOR_SELF_TEST_WEIGHTS_ID).unwrap();
    let mut cursor = 0;
    for (label, tensor) in [
        ("pre_norm.weight", &fixture.parameters.pre_norm.weight[..]),
        ("pre_norm.bias", &fixture.parameters.pre_norm.bias[..]),
        ("linear1.weight", &fixture.parameters.linear1.weight[..]),
        ("linear1.bias", &fixture.parameters.linear1.bias[..]),
        ("linear2.weight", &fixture.parameters.linear2.weight[..]),
        ("linear2.bias", &fixture.parameters.linear2.bias[..]),
    ] {
        assert_f32_segment(weights, &mut cursor, tensor, label);
    }
    assert_eq!(cursor, weights.len(), "unclaimed weight bytes remain");

    for (profile, input, expected, stages) in [
        (
            "synthetic-l3",
            &fixture.l3_input,
            &fixture.l3_expected,
            ProjectorStage::ALL.as_slice(),
        ),
        (
            "synthetic-l2",
            &fixture.l2_input,
            &fixture.l2_expected,
            [ProjectorStage::Linear2].as_slice(),
        ),
    ] {
        let input_bytes = reader
            .section(&format!("input.projector.{profile}"))
            .unwrap();
        cursor = 0;
        assert_f32_segment(input_bytes, &mut cursor, input, profile);
        assert_eq!(cursor, input_bytes.len());
        let expected_bytes = reader
            .section(&format!("self_test.projector.{profile}"))
            .unwrap();
        cursor = 0;
        for stage in stages {
            assert_f32_segment(
                expected_bytes,
                &mut cursor,
                &expected[stage],
                stage.as_str(),
            );
        }
        assert_eq!(
            cursor,
            expected_bytes.len(),
            "{profile} expected payload has a tail"
        );
    }
}

#[test]
fn specialized_reader_rejects_descriptor_sections_and_reanchored_payload_drift() {
    let pristine = build_synthetic_fixture();
    for mutate in [
        |value: &mut serde_json::Value| value["schema_version"] = 2.into(),
        |value: &mut serde_json::Value| value["model_revision"] = "0".repeat(40).into(),
        |value: &mut serde_json::Value| value["hidden_size"] = 4.into(),
        |value: &mut serde_json::Value| value["cases"][0]["profile"] = "../escape".into(),
        |value: &mut serde_json::Value| value["cases"][0]["image_grid_thw"][0][1] = 3.into(),
        |value: &mut serde_json::Value| value["cases"][0]["stage_order"][0] = "merge".into(),
        |value: &mut serde_json::Value| value["cases"].as_array_mut().unwrap().reverse(),
        |value: &mut serde_json::Value| value["shadow"] = true.into(),
    ] {
        assert!(ProjectorSelfTestPack::open(&mutate_descriptor(&pristine, mutate)).is_err());
    }

    let mandatory = [
        (PROJECTOR_SELF_TEST_DESCRIPTOR_ID, SectionKind::SemanticIr),
        (PROJECTOR_SELF_TEST_WEIGHTS_ID, SectionKind::WeightShard),
        ("input.projector.synthetic-l2", SectionKind::WeightShard),
        ("input.projector.synthetic-l3", SectionKind::WeightShard),
        ("self_test.projector.synthetic-l2", SectionKind::SelfTest),
        ("self_test.projector.synthetic-l3", SectionKind::SelfTest),
    ];
    for (id, expected_kind) in mandatory {
        let (manifest, mut records) = sections(&pristine);
        assert!(records.remove(id).is_some());
        assert_eq!(
            ProjectorSelfTestPack::open(&rebuild(manifest, records))
                .unwrap_err()
                .code(),
            ProjectorSelfTestErrorCode::MissingSection,
            "missing {id}",
        );

        let (manifest, mut records) = sections(&pristine);
        let record = records.get_mut(id).unwrap();
        assert_eq!(record.kind, expected_kind);
        record.kind = if expected_kind == SectionKind::SelfTest {
            SectionKind::WeightShard
        } else {
            SectionKind::SelfTest
        };
        assert_eq!(
            ProjectorSelfTestPack::open(&rebuild(manifest, records))
                .unwrap_err()
                .code(),
            ProjectorSelfTestErrorCode::WrongSectionKind,
            "mistyped {id}",
        );

        let (manifest, mut records) = sections(&pristine);
        records.get_mut(id).unwrap().alignment = 1;
        assert_eq!(
            ProjectorSelfTestPack::open(&rebuild(manifest, records))
                .unwrap_err()
                .code(),
            ProjectorSelfTestErrorCode::InvalidAlignment,
            "misaligned {id}",
        );
    }

    let nonfinite_weight =
        mutate_payload_and_reanchor(&pristine, PROJECTOR_SELF_TEST_WEIGHTS_ID, |bytes| {
            bytes[0..4].copy_from_slice(&f32::NAN.to_le_bytes())
        });
    assert_eq!(
        ProjectorSelfTestPack::open(&nonfinite_weight)
            .unwrap_err()
            .code(),
        ProjectorSelfTestErrorCode::InvalidInvocation,
    );
    let nonfinite_input =
        mutate_payload_and_reanchor(&pristine, "input.projector.synthetic-l3", |bytes| {
            bytes[4..8].copy_from_slice(&f32::INFINITY.to_le_bytes())
        });
    assert_eq!(
        ProjectorSelfTestPack::open(&nonfinite_input)
            .unwrap_err()
            .code(),
        ProjectorSelfTestErrorCode::InvalidInvocation,
    );
    let nonfinite_expected =
        mutate_payload_and_reanchor(&pristine, "self_test.projector.synthetic-l3", |bytes| {
            bytes[8..12].copy_from_slice(&f32::NEG_INFINITY.to_le_bytes())
        });
    assert_eq!(
        ProjectorSelfTestPack::open(&nonfinite_expected)
            .unwrap_err()
            .code(),
        ProjectorSelfTestErrorCode::InvalidCheckpoint,
    );
    let short_input =
        mutate_payload_and_reanchor(&pristine, "input.projector.synthetic-l2", |bytes| {
            bytes.truncate(bytes.len() - 4)
        });
    assert_eq!(
        ProjectorSelfTestPack::open(&short_input)
            .unwrap_err()
            .code(),
        ProjectorSelfTestErrorCode::InvalidInvocation,
    );
}

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load_tensor(catalog: &SafetensorsCatalog, name: &str, shape: &[u64]) -> Vec<f32> {
    assert_eq!(catalog.tensor(name).unwrap().shape, shape, "tensor={name}");
    catalog.load_tensor_f32(name).unwrap()
}

#[test]
#[ignore = "materializes the shared 137.7 MiB official L3/L2 projector browser pack"]
fn official_full_projector_pack_is_anchored_to_both_pinned_profiles() {
    assert_eq!(
        std::env::var("PVLC_REQUIRE_MODEL").as_deref(),
        Ok("1"),
        "the explicit official pack gate must require the pinned model",
    );
    let model_path = repository()
        .join("models/snapshots")
        .join(MODEL_REVISION)
        .join("model.safetensors");
    assert!(
        model_path.is_file(),
        "pinned model is absent at {}",
        model_path.display()
    );
    let model = SafetensorsCatalog::open(model_path).unwrap();
    let hidden = 1_152_u64;
    let merged = hidden * 4;
    let output = 1_024_u64;
    let parameters = OwnedProjectorParameters {
        pre_norm: OwnedVisionLayerNormParameters {
            weight: load_tensor(&model, "mlp_AR.pre_norm.weight", &[hidden]),
            bias: load_tensor(&model, "mlp_AR.pre_norm.bias", &[hidden]),
        },
        linear1: OwnedVisionLinearParameters {
            weight: load_tensor(&model, "mlp_AR.linear_1.weight", &[merged, merged]),
            bias: load_tensor(&model, "mlp_AR.linear_1.bias", &[merged]),
        },
        linear2: OwnedVisionLinearParameters {
            weight: load_tensor(&model, "mlp_AR.linear_2.weight", &[output, merged]),
            bias: load_tensor(&model, "mlp_AR.linear_2.bias", &[output]),
        },
    };
    let l3_root = repository().join("artifacts/goldens/ocr.clean_latin.0001-l3");
    let l3_stage = SafetensorsCatalog::open(l3_root.join("stage-checkpoints.safetensors")).unwrap();
    let l3_deep = SafetensorsCatalog::open(l3_root.join("deep-checkpoints.safetensors")).unwrap();
    let l3_input = load_tensor(&l3_stage, "vision.final", &[1_276, hidden]);
    let l3_expected = BTreeMap::from([
        (
            ProjectorStage::PreNorm,
            load_tensor(&l3_deep, "projector.pre_norm", &[1_276, hidden]),
        ),
        (
            ProjectorStage::Merge,
            load_tensor(&l3_deep, "projector.merge", &[319, merged]),
        ),
        (
            ProjectorStage::Linear1,
            load_tensor(&l3_deep, "projector.linear1", &[319, merged]),
        ),
        (
            ProjectorStage::Activation,
            load_tensor(&l3_deep, "projector.gelu", &[319, merged]),
        ),
        (
            ProjectorStage::Linear2,
            load_tensor(&l3_deep, "projector.linear2", &[319, output]),
        ),
    ]);
    let l2_root = repository().join("artifacts/goldens/table.simple.0001-l2");
    let l2_stage = SafetensorsCatalog::open(l2_root.join("stage-checkpoints.safetensors")).unwrap();
    let l2_input = load_tensor(&l2_stage, "vision.final", &[1_740, hidden]);
    let l2_expected = BTreeMap::from([(
        ProjectorStage::Linear2,
        load_tensor(&l2_stage, "projector.final", &[435, output]),
    )]);
    let l3_grid = [[1, 22, 58]];
    let l2_grid = [[1, 30, 58]];
    let cases = [
        ProjectorSelfTestCaseSource::official_l3(&l3_input, &l3_grid, &l3_expected),
        ProjectorSelfTestCaseSource::official_l2(&l2_input, &l2_grid, &l2_expected),
    ];
    let build = || {
        build_projector_self_test_pack(
            COMPILER_BUILD,
            ProjectorSelfTestSource::official(parameters.borrowed(), &cases),
        )
        .unwrap()
    };
    let pack = build();
    let repeated = build();
    assert_eq!(
        pack, repeated,
        "official L3/L2 projector pack is not byte reproducible"
    );
    drop(repeated);
    let reader = PackReader::open(&pack).unwrap();
    let exact_sections = [
        (EXPECTED_L3_INPUT_ID, SectionKind::WeightShard, 256),
        (EXPECTED_L2_INPUT_ID, SectionKind::WeightShard, 256),
        (EXPECTED_DESCRIPTOR_ID, SectionKind::SemanticIr, 64),
        (EXPECTED_L3_CHECKPOINT_ID, SectionKind::SelfTest, 256),
        (EXPECTED_L2_CHECKPOINT_ID, SectionKind::SelfTest, 256),
        (EXPECTED_WEIGHTS_ID, SectionKind::WeightShard, 256),
    ];
    assert_eq!(reader.entries().len(), exact_sections.len());
    assert_eq!(
        reader
            .entries()
            .iter()
            .map(|entry| (entry.id.as_str(), entry.kind, entry.alignment))
            .collect::<Vec<_>>(),
        exact_sections,
    );
    for entry in reader.entries() {
        assert_eq!(entry.offset % u64::from(entry.alignment), 0, "{}", entry.id);
        if entry.kind != SectionKind::SemanticIr {
            assert_eq!(
                entry.offset % 4,
                0,
                "{} is not Float32Array aligned",
                entry.id
            );
        }
    }
    assert_eq!(
        reader
            .entries()
            .iter()
            .filter(|entry| entry.id == EXPECTED_WEIGHTS_ID)
            .count(),
        1,
        "official profiles duplicated their shared projector weights",
    );
    let expected_sections = [
        (
            EXPECTED_L3_INPUT_ID,
            SectionKind::WeightShard,
            EXPECTED_L3_INPUT_BYTES,
            EXPECTED_L3_INPUT_BLAKE3,
        ),
        (
            EXPECTED_L2_INPUT_ID,
            SectionKind::WeightShard,
            EXPECTED_L2_INPUT_BYTES,
            EXPECTED_L2_INPUT_BLAKE3,
        ),
        (
            EXPECTED_WEIGHTS_ID,
            SectionKind::WeightShard,
            EXPECTED_WEIGHTS_BYTES,
            EXPECTED_WEIGHTS_BLAKE3,
        ),
        (
            EXPECTED_L3_CHECKPOINT_ID,
            SectionKind::SelfTest,
            EXPECTED_L3_CHECKPOINT_BYTES,
            EXPECTED_L3_CHECKPOINT_BLAKE3,
        ),
        (
            EXPECTED_L2_CHECKPOINT_ID,
            SectionKind::SelfTest,
            EXPECTED_L2_CHECKPOINT_BYTES,
            EXPECTED_L2_CHECKPOINT_BLAKE3,
        ),
    ];
    for (id, kind, bytes, digest) in expected_sections {
        let entry = reader
            .entries()
            .iter()
            .find(|entry| entry.id == id)
            .unwrap();
        assert_eq!(entry.kind, kind, "{id}");
        assert_eq!(entry.byte_len, bytes, "{id}");
        assert_eq!(
            blake3::hash(reader.section(id).unwrap()).to_hex().as_str(),
            digest,
            "{id}"
        );
    }
    let descriptor_value: serde_json::Value =
        serde_json::from_slice(reader.section(EXPECTED_DESCRIPTOR_ID).unwrap()).unwrap();
    assert_eq!(
        descriptor_value,
        serde_json::json!({
            "schema_version": 1,
            "oracle": "official_mps_bf16",
            "model_revision": MODEL_REVISION,
            "hidden_size": 1_152,
            "output_size": 1_024,
            "layer_norm_epsilon": EPSILON,
            "weights": {
                "section_id": EXPECTED_WEIGHTS_ID,
                "bytes": EXPECTED_WEIGHTS_BYTES,
                "blake3": EXPECTED_WEIGHTS_BLAKE3,
            },
            "cases": [
                {
                    "profile": EXPECTED_L3_PROFILE,
                    "case_id": EXPECTED_L3_CASE_ID,
                    "trace_level": "L3",
                    "golden_bundle_digest": L3_BUNDLE,
                    "semantic_fingerprint": L3_SEMANTIC,
                    "image_grid_thw": [[1, 22, 58]],
                    "readback": "all_stages",
                    "input": {
                        "section_id": EXPECTED_L3_INPUT_ID,
                        "bytes": EXPECTED_L3_INPUT_BYTES,
                        "blake3": EXPECTED_L3_INPUT_BLAKE3,
                    },
                    "expected": {
                        "section_id": EXPECTED_L3_CHECKPOINT_ID,
                        "bytes": EXPECTED_L3_CHECKPOINT_BYTES,
                        "blake3": EXPECTED_L3_CHECKPOINT_BLAKE3,
                    },
                    "stage_order": ["pre_norm", "merge", "linear1", "activation", "linear2"],
                },
                {
                    "profile": EXPECTED_L2_PROFILE,
                    "case_id": EXPECTED_L2_CASE_ID,
                    "trace_level": "L2",
                    "golden_bundle_digest": L2_BUNDLE,
                    "semantic_fingerprint": L2_SEMANTIC,
                    "image_grid_thw": [[1, 30, 58]],
                    "readback": "output_only",
                    "input": {
                        "section_id": EXPECTED_L2_INPUT_ID,
                        "bytes": EXPECTED_L2_INPUT_BYTES,
                        "blake3": EXPECTED_L2_INPUT_BLAKE3,
                    },
                    "expected": {
                        "section_id": EXPECTED_L2_CHECKPOINT_ID,
                        "bytes": EXPECTED_L2_CHECKPOINT_BYTES,
                        "blake3": EXPECTED_L2_CHECKPOINT_BLAKE3,
                    },
                    "stage_order": ["linear2"],
                },
            ],
        }),
    );
    let identity_mutations: Vec<(&str, DescriptorMutation)> = vec![
        (
            "model revision",
            Box::new(|value| value["model_revision"] = "0".repeat(40).into()),
        ),
        (
            "hidden size",
            Box::new(|value| value["hidden_size"] = 1_154.into()),
        ),
        (
            "output size",
            Box::new(|value| value["output_size"] = 1_026.into()),
        ),
        (
            "epsilon",
            Box::new(|value| value["layer_norm_epsilon"] = 2.0e-5.into()),
        ),
        (
            "L3 profile",
            Box::new(|value| value["cases"][0]["profile"] = "ocr-clean-latin-alt-l3".into()),
        ),
        (
            "L3 case",
            Box::new(|value| {
                value["cases"][0]["case_id"] = "ocr.clean_latin.0002/projector".into()
            }),
        ),
        (
            "L3 trace",
            Box::new(|value| value["cases"][0]["trace_level"] = "L2".into()),
        ),
        (
            "L3 bundle",
            Box::new(|value| {
                value["cases"][0]["golden_bundle_digest"] =
                    format!("blake3:{}", "0".repeat(64)).into()
            }),
        ),
        (
            "L3 semantic",
            Box::new(|value| {
                value["cases"][0]["semantic_fingerprint"] =
                    format!("blake3:{}", "0".repeat(64)).into()
            }),
        ),
        (
            "L3 grid",
            Box::new(|value| {
                value["cases"][0]["image_grid_thw"] = serde_json::json!([[1, 20, 58]])
            }),
        ),
        (
            "L3 readback",
            Box::new(|value| value["cases"][0]["readback"] = "output_only".into()),
        ),
        (
            "L3 stages",
            Box::new(|value| value["cases"][0]["stage_order"] = serde_json::json!(["linear2"])),
        ),
        (
            "L2 profile",
            Box::new(|value| value["cases"][1]["profile"] = "table-simple-alt-l2".into()),
        ),
        (
            "L2 case",
            Box::new(|value| value["cases"][1]["case_id"] = "table.simple.0002/projector".into()),
        ),
        (
            "L2 trace",
            Box::new(|value| value["cases"][1]["trace_level"] = "L3".into()),
        ),
        (
            "L2 bundle",
            Box::new(|value| {
                value["cases"][1]["golden_bundle_digest"] =
                    format!("blake3:{}", "0".repeat(64)).into()
            }),
        ),
        (
            "L2 semantic",
            Box::new(|value| {
                value["cases"][1]["semantic_fingerprint"] =
                    format!("blake3:{}", "0".repeat(64)).into()
            }),
        ),
        (
            "L2 grid",
            Box::new(|value| {
                value["cases"][1]["image_grid_thw"] = serde_json::json!([[1, 32, 58]])
            }),
        ),
        (
            "L2 readback",
            Box::new(|value| value["cases"][1]["readback"] = "all_stages".into()),
        ),
        (
            "L2 stages",
            Box::new(|value| {
                value["cases"][1]["stage_order"] =
                    serde_json::json!(["pre_norm", "merge", "linear1", "activation", "linear2"])
            }),
        ),
    ];
    for (label, mutate) in identity_mutations {
        let changed = mutate_descriptor(&pack, |value| mutate(value));
        assert_eq!(
            ProjectorSelfTestPack::open(&changed).unwrap_err().code(),
            ProjectorSelfTestErrorCode::OfficialIdentityMismatch,
            "official descriptor-only substitution passed for {label}",
        );
    }
    let specialized = ProjectorSelfTestPack::open(&pack).unwrap();
    let descriptor = specialized.descriptor();
    assert_eq!(
        descriptor.schema_version,
        PROJECTOR_SELF_TEST_SCHEMA_VERSION
    );
    assert_eq!(descriptor.model_revision, MODEL_REVISION);
    assert_eq!(descriptor.hidden_size, 1_152);
    assert_eq!(descriptor.output_size, 1_024);
    assert_eq!(descriptor.layer_norm_epsilon, EPSILON);
    assert_eq!(descriptor.oracle, ProjectorSelfTestOracle::OfficialMpsBf16);
    assert_eq!(descriptor.cases[0].profile, EXPECTED_L3_PROFILE);
    assert_eq!(descriptor.cases[0].case_id, EXPECTED_L3_CASE_ID);
    assert_eq!(descriptor.cases[0].readback, ProjectorReadback::AllStages);
    assert_eq!(descriptor.cases[0].image_grid_thw, [[1, 22, 58]]);
    assert_eq!(
        descriptor.cases[0].golden_bundle_digest.as_deref(),
        Some(L3_BUNDLE)
    );
    assert_eq!(
        descriptor.cases[0].semantic_fingerprint.as_deref(),
        Some(L3_SEMANTIC)
    );
    assert_eq!(descriptor.cases[1].profile, EXPECTED_L2_PROFILE);
    assert_eq!(descriptor.cases[1].case_id, EXPECTED_L2_CASE_ID);
    assert_eq!(descriptor.cases[1].readback, ProjectorReadback::OutputOnly);
    assert_eq!(descriptor.cases[1].image_grid_thw, [[1, 30, 58]]);
    assert_eq!(
        descriptor.cases[1].golden_bundle_digest.as_deref(),
        Some(L2_BUNDLE)
    );
    assert_eq!(
        descriptor.cases[1].semantic_fingerprint.as_deref(),
        Some(L2_SEMANTIC)
    );
    assert_eq!(descriptor.cases[0].stage_order, ProjectorStage::ALL);
    assert_eq!(descriptor.cases[1].stage_order, [ProjectorStage::Linear2]);
    assert_eq!(
        specialized
            .invocation(EXPECTED_L3_PROFILE)
            .unwrap()
            .plan()
            .unwrap()
            .output_tokens,
        319
    );
    assert_eq!(
        specialized
            .invocation(EXPECTED_L2_PROFILE)
            .unwrap()
            .plan()
            .unwrap()
            .output_tokens,
        435
    );

    for id in [
        EXPECTED_WEIGHTS_ID,
        EXPECTED_L3_INPUT_ID,
        EXPECTED_L2_INPUT_ID,
        EXPECTED_L3_CHECKPOINT_ID,
        EXPECTED_L2_CHECKPOINT_ID,
    ] {
        let changed = mutate_payload_and_reanchor(&pack, id, |bytes| {
            let original = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
            assert!(original.is_finite());
            let replacement = original + 1.0;
            assert!(replacement.is_finite());
            assert_ne!(original.to_bits(), replacement.to_bits());
            bytes[0..4].copy_from_slice(&replacement.to_le_bytes());
        });
        assert_eq!(
            ProjectorSelfTestPack::open(&changed).unwrap_err().code(),
            ProjectorSelfTestErrorCode::OfficialPayloadMismatch,
            "re-anchored finite substitution passed for {id}",
        );
    }
}
