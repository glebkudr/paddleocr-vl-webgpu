use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use pvlc_model_schema::MODEL_REVISION;
use pvlc_runtime_core::{
    OwnedProjectorInvocation, OwnedProjectorParameters, OwnedVisionLayerNormParameters,
    OwnedVisionLinearParameters, ProjectorInvocation, ProjectorParameters, ProjectorReadback,
    ProjectorStage,
};
use serde::{Deserialize, Serialize};

use crate::{PackBuilder, PackManifest, PackReader, PackSection, SectionKind};

pub const PROJECTOR_SELF_TEST_SCHEMA_VERSION: u32 = 1;
pub const PROJECTOR_SELF_TEST_DESCRIPTOR_ID: &str = "ir.projector.official";
pub const PROJECTOR_SELF_TEST_WEIGHTS_ID: &str = "weights.projector";

pub const OFFICIAL_PROJECTOR_L3_PROFILE: &str = "ocr-clean-latin-l3";
pub const OFFICIAL_PROJECTOR_L2_PROFILE: &str = "table-simple-l2";
pub const OFFICIAL_PROJECTOR_L3_INPUT_ID: &str = "input.projector.ocr-clean-latin-l3";
pub const OFFICIAL_PROJECTOR_L2_INPUT_ID: &str = "input.projector.table-simple-l2";
pub const OFFICIAL_PROJECTOR_L3_EXPECTED_ID: &str = "self_test.projector.ocr-clean-latin-l3";
pub const OFFICIAL_PROJECTOR_L2_EXPECTED_ID: &str = "self_test.projector.table-simple-l2";

pub const OFFICIAL_PROJECTOR_WEIGHTS_BYTES: u64 = 103_840_768;
pub const OFFICIAL_PROJECTOR_WEIGHTS_BLAKE3: &str =
    "bca2e52ec0a24bb3643141aa467bb265b2617b6ebbc5591c884c93351cf08a64";
pub const OFFICIAL_PROJECTOR_L3_INPUT_BYTES: u64 = 5_879_808;
pub const OFFICIAL_PROJECTOR_L3_INPUT_BLAKE3: &str =
    "fd6bbb0ccc67ba679f5c06c0fbb4074f8970bb54aa4115b3b39a814bfff48663";
pub const OFFICIAL_PROJECTOR_L2_INPUT_BYTES: u64 = 8_017_920;
pub const OFFICIAL_PROJECTOR_L2_INPUT_BLAKE3: &str =
    "fcd101b25a04e1b4e0984e5d712094630f11c22d4fc57abdf743e9fd7a79aed9";
pub const OFFICIAL_PROJECTOR_L3_EXPECTED_BYTES: u64 = 24_825_856;
pub const OFFICIAL_PROJECTOR_L3_EXPECTED_BLAKE3: &str =
    "cc27c875fdeb691a44a466cf7a4fd3fdb9cf9c218e53236a59f968abbb92ad88";
pub const OFFICIAL_PROJECTOR_L2_EXPECTED_BYTES: u64 = 1_781_760;
pub const OFFICIAL_PROJECTOR_L2_EXPECTED_BLAKE3: &str =
    "e31d337fd75b3ae5a95edc7c3a7ae88d5c0433e6549e82533cafcbbe9f12aac7";

const OFFICIAL_HIDDEN_SIZE: u32 = 1_152;
const OFFICIAL_OUTPUT_SIZE: u32 = 1_024;
const OFFICIAL_LAYER_NORM_EPSILON: f32 = 1.0e-5;
const OFFICIAL_L3_CASE_ID: &str = "ocr.clean_latin.0001/projector";
const OFFICIAL_L2_CASE_ID: &str = "table.simple.0001/projector";
const OFFICIAL_L3_BUNDLE: &str =
    "blake3:35572ac07da5fddee97becb46fb866f906200f433f4c5677ad20fdf36440acf9";
const OFFICIAL_L3_SEMANTIC: &str =
    "blake3:632cbe2de6f47f8f84c764e3a551bf99f352d7459c98d939eb89725357f952b4";
const OFFICIAL_L2_BUNDLE: &str =
    "blake3:4cecc8b2abc37030920acf8860bb317084125dbbbcae5af394fccd0c8044c842";
const OFFICIAL_L2_SEMANTIC: &str =
    "blake3:91f72ffdf81070c2c9ca31628605064bdc02bf19417482cefce41d6a27aa5404";
const OFFICIAL_L3_GRID: [[u32; 3]; 1] = [[1, 22, 58]];
const OFFICIAL_L2_GRID: [[u32; 3]; 1] = [[1, 30, 58]];
const DESCRIPTOR_ALIGNMENT: u32 = 64;
const F32_PAYLOAD_ALIGNMENT: u32 = 256;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectorSelfTestOracle {
    Synthetic,
    OfficialMpsBf16,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectorPayloadDescriptor {
    pub section_id: String,
    pub bytes: u64,
    pub blake3: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectorSelfTestCaseDescriptor {
    pub profile: String,
    pub case_id: String,
    pub trace_level: String,
    pub golden_bundle_digest: Option<String>,
    pub semantic_fingerprint: Option<String>,
    pub image_grid_thw: Vec<[u32; 3]>,
    pub readback: ProjectorReadback,
    pub input: ProjectorPayloadDescriptor,
    pub expected: ProjectorPayloadDescriptor,
    pub stage_order: Vec<ProjectorStage>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectorSelfTestDescriptor {
    pub schema_version: u32,
    pub oracle: ProjectorSelfTestOracle,
    pub model_revision: String,
    pub hidden_size: u32,
    pub output_size: u32,
    pub layer_norm_epsilon: f32,
    pub weights: ProjectorPayloadDescriptor,
    pub cases: Vec<ProjectorSelfTestCaseDescriptor>,
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectorSelfTestCaseSource<'a> {
    profile: &'a str,
    case_id: &'a str,
    trace_level: &'a str,
    golden_bundle_digest: Option<&'a str>,
    semantic_fingerprint: Option<&'a str>,
    input: &'a [f32],
    image_grid_thw: &'a [[u32; 3]],
    readback: ProjectorReadback,
    expected: &'a BTreeMap<ProjectorStage, Vec<f32>>,
}

impl<'a> ProjectorSelfTestCaseSource<'a> {
    #[must_use]
    pub const fn synthetic(
        profile: &'a str,
        case_id: &'a str,
        input: &'a [f32],
        image_grid_thw: &'a [[u32; 3]],
        readback: ProjectorReadback,
        expected: &'a BTreeMap<ProjectorStage, Vec<f32>>,
    ) -> Self {
        Self {
            profile,
            case_id,
            trace_level: "synthetic",
            golden_bundle_digest: None,
            semantic_fingerprint: None,
            input,
            image_grid_thw,
            readback,
            expected,
        }
    }

    #[must_use]
    pub const fn official_l3(
        input: &'a [f32],
        image_grid_thw: &'a [[u32; 3]],
        expected: &'a BTreeMap<ProjectorStage, Vec<f32>>,
    ) -> Self {
        Self {
            profile: OFFICIAL_PROJECTOR_L3_PROFILE,
            case_id: OFFICIAL_L3_CASE_ID,
            trace_level: "L3",
            golden_bundle_digest: Some(OFFICIAL_L3_BUNDLE),
            semantic_fingerprint: Some(OFFICIAL_L3_SEMANTIC),
            input,
            image_grid_thw,
            readback: ProjectorReadback::AllStages,
            expected,
        }
    }

    #[must_use]
    pub const fn official_l2(
        input: &'a [f32],
        image_grid_thw: &'a [[u32; 3]],
        expected: &'a BTreeMap<ProjectorStage, Vec<f32>>,
    ) -> Self {
        Self {
            profile: OFFICIAL_PROJECTOR_L2_PROFILE,
            case_id: OFFICIAL_L2_CASE_ID,
            trace_level: "L2",
            golden_bundle_digest: Some(OFFICIAL_L2_BUNDLE),
            semantic_fingerprint: Some(OFFICIAL_L2_SEMANTIC),
            input,
            image_grid_thw,
            readback: ProjectorReadback::OutputOnly,
            expected,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectorSelfTestSource<'a> {
    oracle: ProjectorSelfTestOracle,
    hidden_size: u32,
    output_size: u32,
    layer_norm_epsilon: f32,
    parameters: ProjectorParameters<'a>,
    cases: &'a [ProjectorSelfTestCaseSource<'a>],
}

impl<'a> ProjectorSelfTestSource<'a> {
    #[must_use]
    pub const fn synthetic(
        hidden_size: u32,
        output_size: u32,
        layer_norm_epsilon: f32,
        parameters: ProjectorParameters<'a>,
        cases: &'a [ProjectorSelfTestCaseSource<'a>],
    ) -> Self {
        Self {
            oracle: ProjectorSelfTestOracle::Synthetic,
            hidden_size,
            output_size,
            layer_norm_epsilon,
            parameters,
            cases,
        }
    }

    #[must_use]
    pub const fn official(
        parameters: ProjectorParameters<'a>,
        cases: &'a [ProjectorSelfTestCaseSource<'a>],
    ) -> Self {
        Self {
            oracle: ProjectorSelfTestOracle::OfficialMpsBf16,
            hidden_size: OFFICIAL_HIDDEN_SIZE,
            output_size: OFFICIAL_OUTPUT_SIZE,
            layer_norm_epsilon: OFFICIAL_LAYER_NORM_EPSILON,
            parameters,
            cases,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectorSelfTestErrorCode {
    PackFormat,
    MissingSection,
    WrongSectionKind,
    InvalidAlignment,
    InvalidDescriptor,
    DigestMismatch,
    LengthMismatch,
    InvalidInvocation,
    InvalidCheckpoint,
    OfficialIdentityMismatch,
    OfficialPayloadMismatch,
}

#[derive(Debug)]
pub struct ProjectorSelfTestError {
    code: ProjectorSelfTestErrorCode,
    message: String,
}

impl ProjectorSelfTestError {
    #[must_use]
    pub const fn code(&self) -> ProjectorSelfTestErrorCode {
        self.code
    }
}

impl fmt::Display for ProjectorSelfTestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "projector self-test {:?}: {}",
            self.code, self.message
        )
    }
}

impl Error for ProjectorSelfTestError {}

#[derive(Debug)]
struct DecodedProjectorCase {
    input: Vec<f32>,
    image_grid_thw: Vec<[u32; 3]>,
    expected: BTreeMap<ProjectorStage, Vec<f32>>,
}

#[derive(Debug)]
pub struct ProjectorSelfTestPack {
    descriptor: ProjectorSelfTestDescriptor,
    parameters: OwnedProjectorParameters,
    cases: BTreeMap<String, DecodedProjectorCase>,
}

impl ProjectorSelfTestPack {
    pub fn open(bytes: &[u8]) -> Result<Self, ProjectorSelfTestError> {
        let reader = PackReader::open(bytes)
            .map_err(|error| pack_error(format!("cannot open outer pack: {error}")))?;
        let descriptor_bytes = required_section(
            &reader,
            PROJECTOR_SELF_TEST_DESCRIPTOR_ID,
            SectionKind::SemanticIr,
            DESCRIPTOR_ALIGNMENT,
        )?;
        let descriptor = parse_projector_self_test_descriptor(descriptor_bytes)?;
        let weights = required_section(
            &reader,
            &descriptor.weights.section_id,
            SectionKind::WeightShard,
            F32_PAYLOAD_ALIGNMENT,
        )?;
        let parameters = decode_parameters(&descriptor, weights)?;
        let mut expected_ids = BTreeSet::from([
            PROJECTOR_SELF_TEST_DESCRIPTOR_ID.to_owned(),
            descriptor.weights.section_id.clone(),
        ]);
        let mut cases = BTreeMap::new();
        for case in &descriptor.cases {
            let input_bytes = required_section(
                &reader,
                &case.input.section_id,
                SectionKind::WeightShard,
                F32_PAYLOAD_ALIGNMENT,
            )?;
            let expected_bytes = required_section(
                &reader,
                &case.expected.section_id,
                SectionKind::SelfTest,
                F32_PAYLOAD_ALIGNMENT,
            )?;
            expected_ids.insert(case.input.section_id.clone());
            expected_ids.insert(case.expected.section_id.clone());
            let input = decode_input(&descriptor, case, input_bytes)?;
            validate_invocation(&descriptor, case, &input, &parameters)?;
            let expected = decode_checkpoints(&descriptor, case, expected_bytes)?;
            cases.insert(
                case.profile.clone(),
                DecodedProjectorCase {
                    input,
                    image_grid_thw: case.image_grid_thw.clone(),
                    expected,
                },
            );
        }
        let actual_ids = reader
            .entries()
            .iter()
            .map(|entry| entry.id.clone())
            .collect::<BTreeSet<_>>();
        if actual_ids != expected_ids || reader.entries().len() != expected_ids.len() {
            return Err(pack_error("projector pack section set is not exact"));
        }
        Ok(Self {
            descriptor,
            parameters,
            cases,
        })
    }

    #[must_use]
    pub const fn descriptor(&self) -> &ProjectorSelfTestDescriptor {
        &self.descriptor
    }

    #[must_use]
    pub const fn parameters(&self) -> &OwnedProjectorParameters {
        &self.parameters
    }

    #[must_use]
    pub fn invocation(&self, profile: &str) -> Option<ProjectorInvocation<'_>> {
        let case = self.cases.get(profile)?;
        Some(ProjectorInvocation {
            hidden_size: self.descriptor.hidden_size,
            output_size: self.descriptor.output_size,
            layer_norm_epsilon: self.descriptor.layer_norm_epsilon,
            input: &case.input,
            image_grid_thw: &case.image_grid_thw,
            parameters: self.parameters.borrowed(),
        })
    }

    #[must_use]
    pub fn expected(&self, profile: &str, stage: ProjectorStage) -> Option<&[f32]> {
        self.cases
            .get(profile)?
            .expected
            .get(&stage)
            .map(Vec::as_slice)
    }
}

struct EncodedCase {
    descriptor: ProjectorSelfTestCaseDescriptor,
    input: Vec<u8>,
    expected: Vec<u8>,
}

pub fn build_projector_self_test_pack(
    compiler_build: &str,
    source: ProjectorSelfTestSource<'_>,
) -> Result<Vec<u8>, ProjectorSelfTestError> {
    if source.cases.is_empty() {
        return Err(invalid_descriptor("projector self-test case set is empty"));
    }
    let weights = encode_parameters(source.parameters);
    let weights_descriptor = payload_descriptor(PROJECTOR_SELF_TEST_WEIGHTS_ID, &weights);
    let mut encoded_cases = Vec::with_capacity(source.cases.len());
    for case in source.cases {
        validate_profile(case.profile)?;
        validate_case_id(case.case_id)?;
        let invocation = ProjectorInvocation {
            hidden_size: source.hidden_size,
            output_size: source.output_size,
            layer_norm_epsilon: source.layer_norm_epsilon,
            input: case.input,
            image_grid_thw: case.image_grid_thw,
            parameters: source.parameters,
        };
        let plan = invocation
            .plan()
            .map_err(|error| invalid_invocation(format!("{} is invalid: {error}", case.profile)))?;
        let input = encode_f32(case.input);
        let expected = encode_checkpoints(invocation, &plan, case.readback, case.expected)?;
        let input_id = format!("input.projector.{}", case.profile);
        let expected_id = format!("self_test.projector.{}", case.profile);
        encoded_cases.push(EncodedCase {
            descriptor: ProjectorSelfTestCaseDescriptor {
                profile: case.profile.to_owned(),
                case_id: case.case_id.to_owned(),
                trace_level: case.trace_level.to_owned(),
                golden_bundle_digest: case.golden_bundle_digest.map(str::to_owned),
                semantic_fingerprint: case.semantic_fingerprint.map(str::to_owned),
                image_grid_thw: case.image_grid_thw.to_vec(),
                readback: case.readback,
                input: payload_descriptor(&input_id, &input),
                expected: payload_descriptor(&expected_id, &expected),
                stage_order: readback_stages(case.readback).to_vec(),
            },
            input,
            expected,
        });
    }
    encoded_cases.sort_by(|left, right| left.descriptor.profile.cmp(&right.descriptor.profile));
    let descriptor = ProjectorSelfTestDescriptor {
        schema_version: PROJECTOR_SELF_TEST_SCHEMA_VERSION,
        oracle: source.oracle,
        model_revision: MODEL_REVISION.to_owned(),
        hidden_size: source.hidden_size,
        output_size: source.output_size,
        layer_norm_epsilon: source.layer_norm_epsilon,
        weights: weights_descriptor,
        cases: encoded_cases
            .iter()
            .map(|case| case.descriptor.clone())
            .collect(),
    };
    validate_descriptor(&descriptor)?;
    let descriptor_bytes = canonical_descriptor_bytes(&descriptor)?;
    let mut manifest = PackManifest::paddleocr_vl_16(compiler_build)
        .map_err(|error| pack_error(format!("cannot create pack manifest: {error}")))?;
    manifest.resolution_buckets = descriptor
        .cases
        .iter()
        .flat_map(|case| case.image_grid_thw.iter().map(|grid| [grid[1], grid[2]]))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut builder = PackBuilder::new(manifest);
    builder
        .add_section(PackSection::new(
            PROJECTOR_SELF_TEST_DESCRIPTOR_ID,
            SectionKind::SemanticIr,
            DESCRIPTOR_ALIGNMENT,
            descriptor_bytes,
        ))
        .map_err(|error| pack_error(format!("cannot add projector descriptor: {error}")))?;
    builder
        .add_section(PackSection::new(
            PROJECTOR_SELF_TEST_WEIGHTS_ID,
            SectionKind::WeightShard,
            F32_PAYLOAD_ALIGNMENT,
            weights,
        ))
        .map_err(|error| pack_error(format!("cannot add projector weights: {error}")))?;
    for case in encoded_cases {
        builder
            .add_section(PackSection::new(
                case.descriptor.input.section_id,
                SectionKind::WeightShard,
                F32_PAYLOAD_ALIGNMENT,
                case.input,
            ))
            .map_err(|error| pack_error(format!("cannot add projector input: {error}")))?;
        builder
            .add_section(PackSection::new(
                case.descriptor.expected.section_id,
                SectionKind::SelfTest,
                F32_PAYLOAD_ALIGNMENT,
                case.expected,
            ))
            .map_err(|error| pack_error(format!("cannot add projector checkpoints: {error}")))?;
    }
    builder
        .build()
        .map_err(|error| pack_error(format!("cannot build projector self-test pack: {error}")))
}

pub fn parse_projector_self_test_descriptor(
    json: &[u8],
) -> Result<ProjectorSelfTestDescriptor, ProjectorSelfTestError> {
    let value: serde_json::Value = serde_json::from_slice(json)
        .map_err(|error| invalid_descriptor(format!("descriptor JSON is invalid: {error}")))?;
    let descriptor: ProjectorSelfTestDescriptor = serde_json::from_value(value)
        .map_err(|error| invalid_descriptor(format!("descriptor schema is invalid: {error}")))?;
    validate_descriptor(&descriptor)?;
    if canonical_descriptor_bytes(&descriptor)? != json {
        return Err(invalid_descriptor(
            "descriptor JSON is not in canonical byte form",
        ));
    }
    Ok(descriptor)
}

pub fn decode_projector_self_test_invocation(
    descriptor: &ProjectorSelfTestDescriptor,
    profile: &str,
    input: &[u8],
    weights: &[u8],
) -> Result<OwnedProjectorInvocation, ProjectorSelfTestError> {
    validate_descriptor(descriptor)?;
    let case = descriptor_case(descriptor, profile)?;
    let parameters = decode_parameters(descriptor, weights)?;
    let input = decode_input(descriptor, case, input)?;
    validate_invocation(descriptor, case, &input, &parameters)?;
    Ok(OwnedProjectorInvocation {
        hidden_size: descriptor.hidden_size,
        output_size: descriptor.output_size,
        layer_norm_epsilon: descriptor.layer_norm_epsilon,
        input,
        image_grid_thw: case.image_grid_thw.clone(),
        parameters,
    })
}

fn validate_descriptor(
    descriptor: &ProjectorSelfTestDescriptor,
) -> Result<(), ProjectorSelfTestError> {
    if descriptor.schema_version != PROJECTOR_SELF_TEST_SCHEMA_VERSION {
        return Err(invalid_descriptor(format!(
            "unsupported schema version {}",
            descriptor.schema_version
        )));
    }
    if descriptor.oracle == ProjectorSelfTestOracle::OfficialMpsBf16 {
        validate_official_identity(descriptor)?;
        validate_official_payloads(descriptor)?;
    }
    if descriptor.model_revision != MODEL_REVISION {
        return Err(invalid_descriptor(
            "model revision does not match the compiler",
        ));
    }
    if descriptor.hidden_size == 0
        || descriptor.output_size == 0
        || !descriptor.layer_norm_epsilon.is_finite()
        || descriptor.layer_norm_epsilon <= 0.0
    {
        return Err(invalid_descriptor(
            "projector geometry or epsilon is invalid",
        ));
    }
    validate_payload_descriptor(&descriptor.weights, PROJECTOR_SELF_TEST_WEIGHTS_ID)?;
    let required_weight_bytes =
        derived_weight_bytes(descriptor.hidden_size, descriptor.output_size)?;
    if descriptor.weights.bytes != required_weight_bytes {
        return Err(error(
            ProjectorSelfTestErrorCode::LengthMismatch,
            format!(
                "weight payload has {} bytes, geometry requires {required_weight_bytes}",
                descriptor.weights.bytes
            ),
        ));
    }
    if descriptor.cases.is_empty() {
        return Err(invalid_descriptor("projector case set is empty"));
    }
    let mut previous_profile: Option<&str> = None;
    let mut section_ids = BTreeSet::from([
        PROJECTOR_SELF_TEST_DESCRIPTOR_ID.to_owned(),
        PROJECTOR_SELF_TEST_WEIGHTS_ID.to_owned(),
    ]);
    for case in &descriptor.cases {
        validate_profile(&case.profile)?;
        validate_case_id(&case.case_id)?;
        if previous_profile.is_some_and(|previous| previous >= case.profile.as_str()) {
            return Err(invalid_descriptor(
                "projector profiles are not strictly canonical",
            ));
        }
        previous_profile = Some(&case.profile);
        if case.trace_level.is_empty() || case.trace_level.len() > 32 {
            return Err(invalid_descriptor("projector trace level is invalid"));
        }
        validate_payload_descriptor(&case.input, &format!("input.projector.{}", case.profile))?;
        validate_payload_descriptor(
            &case.expected,
            &format!("self_test.projector.{}", case.profile),
        )?;
        if !section_ids.insert(case.input.section_id.clone())
            || !section_ids.insert(case.expected.section_id.clone())
        {
            return Err(invalid_descriptor("projector payload section IDs collide"));
        }
        let geometry = case_geometry(&case.image_grid_thw, descriptor.hidden_size)?;
        if case.input.bytes != geometry.input_bytes {
            return Err(invalid_invocation(format!(
                "{} input byte length drifted",
                case.profile
            )));
        }
        let expected_stages = readback_stages(case.readback);
        if case.stage_order != expected_stages {
            return Err(invalid_descriptor(format!(
                "{} stage order disagrees with readback",
                case.profile
            )));
        }
        let expected_bytes = expected_checkpoint_bytes(
            descriptor.hidden_size,
            descriptor.output_size,
            geometry.input_tokens,
            geometry.output_tokens,
            expected_stages,
        )?;
        if case.expected.bytes != expected_bytes {
            return Err(error(
                ProjectorSelfTestErrorCode::LengthMismatch,
                format!("{} expected byte length drifted", case.profile),
            ));
        }
        if descriptor.oracle == ProjectorSelfTestOracle::Synthetic
            && (case.golden_bundle_digest.is_some()
                || case.semantic_fingerprint.is_some()
                || matches!(
                    case.profile.as_str(),
                    OFFICIAL_PROJECTOR_L3_PROFILE | OFFICIAL_PROJECTOR_L2_PROFILE
                ))
        {
            return Err(error(
                ProjectorSelfTestErrorCode::OfficialIdentityMismatch,
                "synthetic projector descriptor claims official provenance",
            ));
        }
    }
    Ok(())
}

fn validate_official_identity(
    descriptor: &ProjectorSelfTestDescriptor,
) -> Result<(), ProjectorSelfTestError> {
    let identity_matches = descriptor.model_revision == MODEL_REVISION
        && descriptor.hidden_size == OFFICIAL_HIDDEN_SIZE
        && descriptor.output_size == OFFICIAL_OUTPUT_SIZE
        && descriptor.layer_norm_epsilon.to_bits() == OFFICIAL_LAYER_NORM_EPSILON.to_bits()
        && descriptor.cases.len() == 2
        && official_case_identity_matches(
            &descriptor.cases[0],
            OFFICIAL_PROJECTOR_L3_PROFILE,
            OFFICIAL_L3_CASE_ID,
            "L3",
            OFFICIAL_L3_BUNDLE,
            OFFICIAL_L3_SEMANTIC,
            &OFFICIAL_L3_GRID,
            ProjectorReadback::AllStages,
            &ProjectorStage::ALL,
        )
        && official_case_identity_matches(
            &descriptor.cases[1],
            OFFICIAL_PROJECTOR_L2_PROFILE,
            OFFICIAL_L2_CASE_ID,
            "L2",
            OFFICIAL_L2_BUNDLE,
            OFFICIAL_L2_SEMANTIC,
            &OFFICIAL_L2_GRID,
            ProjectorReadback::OutputOnly,
            &[ProjectorStage::Linear2],
        );
    if !identity_matches {
        return Err(error(
            ProjectorSelfTestErrorCode::OfficialIdentityMismatch,
            "official projector descriptor identity, geometry, or readback drifted",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn official_case_identity_matches(
    case: &ProjectorSelfTestCaseDescriptor,
    profile: &str,
    case_id: &str,
    trace_level: &str,
    bundle: &str,
    semantic: &str,
    grid: &[[u32; 3]],
    readback: ProjectorReadback,
    stage_order: &[ProjectorStage],
) -> bool {
    case.profile == profile
        && case.case_id == case_id
        && case.trace_level == trace_level
        && case.golden_bundle_digest.as_deref() == Some(bundle)
        && case.semantic_fingerprint.as_deref() == Some(semantic)
        && case.image_grid_thw == grid
        && case.readback == readback
        && case.stage_order == stage_order
}

fn validate_official_payloads(
    descriptor: &ProjectorSelfTestDescriptor,
) -> Result<(), ProjectorSelfTestError> {
    let expected_weights = ProjectorPayloadDescriptor {
        section_id: PROJECTOR_SELF_TEST_WEIGHTS_ID.to_owned(),
        bytes: OFFICIAL_PROJECTOR_WEIGHTS_BYTES,
        blake3: OFFICIAL_PROJECTOR_WEIGHTS_BLAKE3.to_owned(),
    };
    let expected_l3_input = ProjectorPayloadDescriptor {
        section_id: OFFICIAL_PROJECTOR_L3_INPUT_ID.to_owned(),
        bytes: OFFICIAL_PROJECTOR_L3_INPUT_BYTES,
        blake3: OFFICIAL_PROJECTOR_L3_INPUT_BLAKE3.to_owned(),
    };
    let expected_l3_checkpoints = ProjectorPayloadDescriptor {
        section_id: OFFICIAL_PROJECTOR_L3_EXPECTED_ID.to_owned(),
        bytes: OFFICIAL_PROJECTOR_L3_EXPECTED_BYTES,
        blake3: OFFICIAL_PROJECTOR_L3_EXPECTED_BLAKE3.to_owned(),
    };
    let expected_l2_input = ProjectorPayloadDescriptor {
        section_id: OFFICIAL_PROJECTOR_L2_INPUT_ID.to_owned(),
        bytes: OFFICIAL_PROJECTOR_L2_INPUT_BYTES,
        blake3: OFFICIAL_PROJECTOR_L2_INPUT_BLAKE3.to_owned(),
    };
    let expected_l2_checkpoints = ProjectorPayloadDescriptor {
        section_id: OFFICIAL_PROJECTOR_L2_EXPECTED_ID.to_owned(),
        bytes: OFFICIAL_PROJECTOR_L2_EXPECTED_BYTES,
        blake3: OFFICIAL_PROJECTOR_L2_EXPECTED_BLAKE3.to_owned(),
    };
    let matches = descriptor.weights == expected_weights
        && descriptor.cases[0].input == expected_l3_input
        && descriptor.cases[0].expected == expected_l3_checkpoints
        && descriptor.cases[1].input == expected_l2_input
        && descriptor.cases[1].expected == expected_l2_checkpoints;
    if !matches {
        return Err(error(
            ProjectorSelfTestErrorCode::OfficialPayloadMismatch,
            "official projector payload references do not match pinned anchors",
        ));
    }
    Ok(())
}

fn required_section<'a>(
    reader: &'a PackReader<'_>,
    id: &str,
    expected_kind: SectionKind,
    expected_alignment: u32,
) -> Result<&'a [u8], ProjectorSelfTestError> {
    let entry = reader
        .entries()
        .iter()
        .find(|entry| entry.id == id)
        .ok_or_else(|| {
            error(
                ProjectorSelfTestErrorCode::MissingSection,
                format!("missing {id}"),
            )
        })?;
    if entry.kind != expected_kind {
        return Err(error(
            ProjectorSelfTestErrorCode::WrongSectionKind,
            format!("{id} has {:?}, expected {expected_kind:?}", entry.kind),
        ));
    }
    if entry.alignment != expected_alignment
        || entry.offset % u64::from(expected_alignment) != 0
        || (expected_kind != SectionKind::SemanticIr
            && (!entry.offset.is_multiple_of(4) || !entry.byte_len.is_multiple_of(4)))
    {
        return Err(error(
            ProjectorSelfTestErrorCode::InvalidAlignment,
            format!("{id} does not satisfy its browser alignment ABI"),
        ));
    }
    reader.section(id).ok_or_else(|| {
        error(
            ProjectorSelfTestErrorCode::MissingSection,
            format!("cannot read {id}"),
        )
    })
}

fn descriptor_case<'a>(
    descriptor: &'a ProjectorSelfTestDescriptor,
    profile: &str,
) -> Result<&'a ProjectorSelfTestCaseDescriptor, ProjectorSelfTestError> {
    let matches = descriptor
        .cases
        .iter()
        .filter(|case| case.profile == profile)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(invalid_descriptor(format!(
            "profile {profile:?} is missing or duplicated"
        )));
    }
    Ok(matches[0])
}

fn payload_descriptor(id: &str, bytes: &[u8]) -> ProjectorPayloadDescriptor {
    ProjectorPayloadDescriptor {
        section_id: id.to_owned(),
        bytes: bytes.len() as u64,
        blake3: blake3::hash(bytes).to_hex().to_string(),
    }
}

fn validate_payload_descriptor(
    payload: &ProjectorPayloadDescriptor,
    expected_id: &str,
) -> Result<(), ProjectorSelfTestError> {
    validate_section_id(&payload.section_id)?;
    if payload.section_id != expected_id {
        return Err(invalid_descriptor(format!(
            "payload section {} is not canonical {expected_id}",
            payload.section_id
        )));
    }
    if payload.bytes == 0 || !payload.bytes.is_multiple_of(4) {
        return Err(error(
            ProjectorSelfTestErrorCode::LengthMismatch,
            format!("{} byte length is not nonzero F32", payload.section_id),
        ));
    }
    if !is_blake3_hex(&payload.blake3) {
        return Err(invalid_descriptor(format!(
            "{} BLAKE3 is invalid",
            payload.section_id
        )));
    }
    Ok(())
}

fn validate_profile(profile: &str) -> Result<(), ProjectorSelfTestError> {
    if profile.is_empty()
        || profile.len() > 64
        || profile.starts_with('-')
        || profile.ends_with('-')
        || !profile
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(invalid_descriptor(
            "projector profile is unsafe or malformed",
        ));
    }
    Ok(())
}

fn validate_case_id(case_id: &str) -> Result<(), ProjectorSelfTestError> {
    if case_id.is_empty()
        || case_id.len() > 128
        || case_id.starts_with('/')
        || case_id.contains("..")
        || case_id.contains("//")
        || !case_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'/'))
    {
        return Err(invalid_descriptor(
            "projector case ID is unsafe or malformed",
        ));
    }
    Ok(())
}

fn validate_section_id(section_id: &str) -> Result<(), ProjectorSelfTestError> {
    if section_id.is_empty()
        || section_id.len() > 128
        || section_id.starts_with('/')
        || section_id.contains("..")
        || section_id.contains("//")
        || !section_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'/'))
    {
        return Err(invalid_descriptor(
            "projector section ID is unsafe or malformed",
        ));
    }
    Ok(())
}

fn canonical_descriptor_bytes(
    descriptor: &ProjectorSelfTestDescriptor,
) -> Result<Vec<u8>, ProjectorSelfTestError> {
    let value = serde_json::to_value(descriptor)
        .map_err(|error| invalid_descriptor(format!("cannot serialize descriptor: {error}")))?;
    let mut bytes = serde_json::to_vec(&value)
        .map_err(|error| invalid_descriptor(format!("cannot encode descriptor: {error}")))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn encode_parameters(parameters: ProjectorParameters<'_>) -> Vec<u8> {
    let tensors = [
        parameters.pre_norm.weight,
        parameters.pre_norm.bias,
        parameters.linear1.weight,
        parameters.linear1.bias,
        parameters.linear2.weight,
        parameters.linear2.bias,
    ];
    let elements = tensors.iter().map(|tensor| tensor.len()).sum::<usize>();
    let mut bytes = Vec::with_capacity(elements.saturating_mul(4));
    for tensor in tensors {
        append_f32(&mut bytes, tensor);
    }
    bytes
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    append_f32(&mut bytes, values);
    bytes
}

fn encode_checkpoints(
    invocation: ProjectorInvocation<'_>,
    plan: &pvlc_runtime_core::ProjectorPlan,
    readback: ProjectorReadback,
    checkpoints: &BTreeMap<ProjectorStage, Vec<f32>>,
) -> Result<Vec<u8>, ProjectorSelfTestError> {
    let stages = readback_stages(readback);
    if checkpoints.len() != stages.len()
        || checkpoints.keys().copied().collect::<Vec<_>>() != stages
    {
        return Err(invalid_checkpoint(
            "projector checkpoint stage set is not exact",
        ));
    }
    let expected_bytes = plan.readback_bytes(readback);
    let mut bytes = Vec::with_capacity(
        usize::try_from(expected_bytes)
            .map_err(|_| invalid_checkpoint("checkpoint payload exceeds usize"))?,
    );
    for stage in stages {
        let values = checkpoints
            .get(stage)
            .ok_or_else(|| invalid_checkpoint(format!("missing {stage:?}")))?;
        let expected = stage_elements(
            invocation.hidden_size,
            invocation.output_size,
            plan.input_tokens,
            plan.output_tokens,
            *stage,
        )?;
        if values.len() as u64 != expected || values.iter().any(|value| !value.is_finite()) {
            return Err(invalid_checkpoint(format!(
                "{stage:?} has invalid length or non-finite values"
            )));
        }
        append_f32(&mut bytes, values);
    }
    if bytes.len() as u64 != expected_bytes {
        return Err(invalid_checkpoint("encoded checkpoint byte count drifted"));
    }
    Ok(bytes)
}

fn decode_parameters(
    descriptor: &ProjectorSelfTestDescriptor,
    bytes: &[u8],
) -> Result<OwnedProjectorParameters, ProjectorSelfTestError> {
    require_payload(bytes, &descriptor.weights)?;
    let hidden = u64::from(descriptor.hidden_size);
    let merged = hidden
        .checked_mul(4)
        .ok_or_else(|| invalid_invocation("projector merged width overflowed"))?;
    let output = u64::from(descriptor.output_size);
    let mut decoder = F32Decoder::new(bytes);
    let parameters = OwnedProjectorParameters {
        pre_norm: OwnedVisionLayerNormParameters {
            weight: decoder.take(hidden, "pre_norm.weight")?,
            bias: decoder.take(hidden, "pre_norm.bias")?,
        },
        linear1: OwnedVisionLinearParameters {
            weight: decoder.take(
                checked_mul(merged, merged, "linear1.weight")?,
                "linear1.weight",
            )?,
            bias: decoder.take(merged, "linear1.bias")?,
        },
        linear2: OwnedVisionLinearParameters {
            weight: decoder.take(
                checked_mul(output, merged, "linear2.weight")?,
                "linear2.weight",
            )?,
            bias: decoder.take(output, "linear2.bias")?,
        },
    };
    if !decoder.is_finished() {
        return Err(error(
            ProjectorSelfTestErrorCode::LengthMismatch,
            "projector weight payload has trailing values",
        ));
    }
    Ok(parameters)
}

fn decode_input(
    descriptor: &ProjectorSelfTestDescriptor,
    case: &ProjectorSelfTestCaseDescriptor,
    bytes: &[u8],
) -> Result<Vec<f32>, ProjectorSelfTestError> {
    require_payload(bytes, &case.input)?;
    let geometry = case_geometry(&case.image_grid_thw, descriptor.hidden_size)?;
    let mut decoder = F32Decoder::new(bytes);
    let input = decoder.take(
        checked_mul(
            u64::from(geometry.input_tokens),
            u64::from(descriptor.hidden_size),
            "projector input",
        )?,
        "input",
    )?;
    if !decoder.is_finished() {
        return Err(error(
            ProjectorSelfTestErrorCode::LengthMismatch,
            "projector input payload has trailing values",
        ));
    }
    Ok(input)
}

fn validate_invocation(
    descriptor: &ProjectorSelfTestDescriptor,
    case: &ProjectorSelfTestCaseDescriptor,
    input: &[f32],
    parameters: &OwnedProjectorParameters,
) -> Result<(), ProjectorSelfTestError> {
    ProjectorInvocation {
        hidden_size: descriptor.hidden_size,
        output_size: descriptor.output_size,
        layer_norm_epsilon: descriptor.layer_norm_epsilon,
        input,
        image_grid_thw: &case.image_grid_thw,
        parameters: parameters.borrowed(),
    }
    .plan()
    .map_err(|error| invalid_invocation(format!("decoded {} is invalid: {error}", case.profile)))?;
    Ok(())
}

fn decode_checkpoints(
    descriptor: &ProjectorSelfTestDescriptor,
    case: &ProjectorSelfTestCaseDescriptor,
    bytes: &[u8],
) -> Result<BTreeMap<ProjectorStage, Vec<f32>>, ProjectorSelfTestError> {
    require_payload(bytes, &case.expected)?;
    let geometry = case_geometry(&case.image_grid_thw, descriptor.hidden_size)?;
    let mut decoder = F32Decoder::new(bytes);
    let mut checkpoints = BTreeMap::new();
    for stage in &case.stage_order {
        let values = decoder
            .take(
                stage_elements(
                    descriptor.hidden_size,
                    descriptor.output_size,
                    geometry.input_tokens,
                    geometry.output_tokens,
                    *stage,
                )?,
                stage.as_str(),
            )
            .map_err(|error| invalid_checkpoint(error.to_string()))?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid_checkpoint(format!(
                "{} contains a non-finite value",
                stage.as_str()
            )));
        }
        checkpoints.insert(*stage, values);
    }
    if !decoder.is_finished() {
        return Err(error(
            ProjectorSelfTestErrorCode::LengthMismatch,
            "projector checkpoint payload has trailing values",
        ));
    }
    Ok(checkpoints)
}

fn require_payload(
    bytes: &[u8],
    descriptor: &ProjectorPayloadDescriptor,
) -> Result<(), ProjectorSelfTestError> {
    if bytes.len() as u64 != descriptor.bytes {
        return Err(error(
            ProjectorSelfTestErrorCode::LengthMismatch,
            format!(
                "{} has {} bytes, expected {}",
                descriptor.section_id,
                bytes.len(),
                descriptor.bytes
            ),
        ));
    }
    if blake3::hash(bytes).to_hex().as_str() != descriptor.blake3 {
        return Err(error(
            ProjectorSelfTestErrorCode::DigestMismatch,
            format!("{} BLAKE3 mismatch", descriptor.section_id),
        ));
    }
    Ok(())
}

struct F32Decoder<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> F32Decoder<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn take(&mut self, elements: u64, label: &str) -> Result<Vec<f32>, ProjectorSelfTestError> {
        let byte_len = elements
            .checked_mul(4)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| invalid_invocation(format!("{label} byte length overflowed")))?;
        let end = self
            .cursor
            .checked_add(byte_len)
            .ok_or_else(|| invalid_invocation(format!("{label} offset overflowed")))?;
        let source = self
            .bytes
            .get(self.cursor..end)
            .ok_or_else(|| invalid_invocation(format!("{label} is truncated")))?;
        self.cursor = end;
        Ok(source
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect())
    }

    const fn is_finished(&self) -> bool {
        self.cursor == self.bytes.len()
    }
}

fn append_f32(bytes: &mut Vec<u8>, values: &[f32]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CaseGeometry {
    input_tokens: u32,
    output_tokens: u32,
    input_bytes: u64,
}

fn case_geometry(
    image_grid_thw: &[[u32; 3]],
    hidden_size: u32,
) -> Result<CaseGeometry, ProjectorSelfTestError> {
    if image_grid_thw.is_empty() {
        return Err(invalid_invocation(
            "image_grid_thw must contain at least one image",
        ));
    }
    let mut input_tokens = 0_u32;
    let mut output_tokens = 0_u32;
    for &[temporal, height, width] in image_grid_thw {
        if temporal == 0 || height == 0 || width == 0 {
            return Err(invalid_invocation(
                "every projector grid dimension must be nonzero",
            ));
        }
        if !height.is_multiple_of(2) || !width.is_multiple_of(2) {
            return Err(invalid_invocation(
                "projector grid height and width must be even",
            ));
        }
        let grid_input = checked_mul(
            checked_mul(u64::from(temporal), u64::from(height), "grid input")?,
            u64::from(width),
            "grid input",
        )?;
        let grid_output = checked_mul(
            checked_mul(u64::from(temporal), u64::from(height / 2), "grid output")?,
            u64::from(width / 2),
            "grid output",
        )?;
        input_tokens = input_tokens
            .checked_add(
                u32::try_from(grid_input)
                    .map_err(|_| invalid_invocation("projector input token count exceeds u32"))?,
            )
            .ok_or_else(|| invalid_invocation("projector input token count overflowed"))?;
        output_tokens = output_tokens
            .checked_add(
                u32::try_from(grid_output)
                    .map_err(|_| invalid_invocation("projector output token count exceeds u32"))?,
            )
            .ok_or_else(|| invalid_invocation("projector output token count overflowed"))?;
    }
    let input_elements = checked_mul(
        u64::from(input_tokens),
        u64::from(hidden_size),
        "projector input",
    )?;
    let input_bytes = checked_mul(input_elements, 4, "projector input bytes")?;
    Ok(CaseGeometry {
        input_tokens,
        output_tokens,
        input_bytes,
    })
}

fn derived_weight_bytes(hidden_size: u32, output_size: u32) -> Result<u64, ProjectorSelfTestError> {
    let hidden = u64::from(hidden_size);
    let output = u64::from(output_size);
    let merged = checked_mul(hidden, 4, "projector merged width")?;
    let norm = checked_mul(hidden, 2, "projector pre-norm parameters")?;
    let linear1 = checked_mul(merged, merged, "projector linear1 weights")?
        .checked_add(merged)
        .ok_or_else(|| invalid_invocation("projector linear1 parameters overflowed"))?;
    let linear2 = checked_mul(output, merged, "projector linear2 weights")?
        .checked_add(output)
        .ok_or_else(|| invalid_invocation("projector linear2 parameters overflowed"))?;
    let elements = norm
        .checked_add(linear1)
        .and_then(|elements| elements.checked_add(linear2))
        .ok_or_else(|| invalid_invocation("projector parameter count overflowed"))?;
    checked_mul(elements, 4, "projector weight bytes")
}

fn expected_checkpoint_bytes(
    hidden_size: u32,
    output_size: u32,
    input_tokens: u32,
    output_tokens: u32,
    stages: &[ProjectorStage],
) -> Result<u64, ProjectorSelfTestError> {
    let elements = stages.iter().try_fold(0_u64, |elements, stage| {
        elements
            .checked_add(stage_elements(
                hidden_size,
                output_size,
                input_tokens,
                output_tokens,
                *stage,
            )?)
            .ok_or_else(|| invalid_checkpoint("projector checkpoint element count overflowed"))
    })?;
    checked_mul(elements, 4, "projector checkpoint bytes")
        .map_err(|error| invalid_checkpoint(error.to_string()))
}

fn stage_elements(
    hidden_size: u32,
    output_size: u32,
    input_tokens: u32,
    output_tokens: u32,
    stage: ProjectorStage,
) -> Result<u64, ProjectorSelfTestError> {
    let width = match stage {
        ProjectorStage::PreNorm => hidden_size,
        ProjectorStage::Merge | ProjectorStage::Linear1 | ProjectorStage::Activation => hidden_size
            .checked_mul(4)
            .ok_or_else(|| invalid_checkpoint("projector merged width overflowed"))?,
        ProjectorStage::Linear2 => output_size,
    };
    let tokens = if stage == ProjectorStage::PreNorm {
        input_tokens
    } else {
        output_tokens
    };
    checked_mul(u64::from(tokens), u64::from(width), stage.as_str())
        .map_err(|error| invalid_checkpoint(error.to_string()))
}

const fn readback_stages(readback: ProjectorReadback) -> &'static [ProjectorStage] {
    match readback {
        ProjectorReadback::OutputOnly => &[ProjectorStage::Linear2],
        ProjectorReadback::AllStages => &ProjectorStage::ALL,
    }
}

fn checked_mul(left: u64, right: u64, label: &str) -> Result<u64, ProjectorSelfTestError> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_invocation(format!("{label} overflowed")))
}

fn is_blake3_hex(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn error(code: ProjectorSelfTestErrorCode, message: impl Into<String>) -> ProjectorSelfTestError {
    ProjectorSelfTestError {
        code,
        message: message.into(),
    }
}

fn pack_error(message: impl Into<String>) -> ProjectorSelfTestError {
    error(ProjectorSelfTestErrorCode::PackFormat, message)
}

fn invalid_descriptor(message: impl Into<String>) -> ProjectorSelfTestError {
    error(ProjectorSelfTestErrorCode::InvalidDescriptor, message)
}

fn invalid_invocation(message: impl Into<String>) -> ProjectorSelfTestError {
    error(ProjectorSelfTestErrorCode::InvalidInvocation, message)
}

fn invalid_checkpoint(message: impl Into<String>) -> ProjectorSelfTestError {
    error(ProjectorSelfTestErrorCode::InvalidCheckpoint, message)
}
