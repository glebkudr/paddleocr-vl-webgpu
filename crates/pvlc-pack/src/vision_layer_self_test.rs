use std::{collections::BTreeMap, error::Error, fmt};

use pvlc_model_schema::MODEL_REVISION;
use pvlc_runtime_core::{
    OwnedVisionEncoderLayerInvocation, OwnedVisionEncoderLayerParameters,
    OwnedVisionLayerNormParameters, OwnedVisionLinearParameters, VisionEncoderLayerInvocation,
    VisionEncoderLayerStage,
};
use serde::{Deserialize, Serialize};

use crate::{PackBuilder, PackManifest, PackReader, PackSection, SectionKind};

pub const VISION_LAYER_SELF_TEST_SCHEMA_VERSION: u32 = 1;
pub const VISION_LAYER_SELF_TEST_DESCRIPTOR_ID: &str = "ir.vision_layer_00";
pub const VISION_LAYER_SELF_TEST_WEIGHTS_ID: &str = "weights.vision_layer_00";
pub const VISION_LAYER_SELF_TEST_EXPECTED_ID: &str = "self_test.vision_layer_00";

pub const OFFICIAL_VISION_LAYER_SELF_TEST_CASE_ID: &str = "ocr.clean_latin.0001/vision.layer.00";
pub const OFFICIAL_VISION_LAYER_WEIGHTS_BYTES: u64 = 66_837_824;
pub const OFFICIAL_VISION_LAYER_EXPECTED_BYTES: u64 = 102_733_312;
pub const OFFICIAL_VISION_LAYER_WEIGHTS_BLAKE3: &str =
    "8bbc38f130818d26cf7996e8c78055a022665e5772fee52c27d80a2efdd7d5c0";
pub const OFFICIAL_VISION_LAYER_EXPECTED_BLAKE3: &str =
    "df6d0d65ea94559f68b36f0e8e85de364265d0fb719bff1eced439af11a63cba";
pub const OFFICIAL_VISION_LAYER_GOLDEN_BUNDLE_DIGEST: &str =
    "blake3:35572ac07da5fddee97becb46fb866f906200f433f4c5677ad20fdf36440acf9";
pub const OFFICIAL_VISION_LAYER_SEMANTIC_FINGERPRINT: &str =
    "blake3:632cbe2de6f47f8f84c764e3a551bf99f352d7459c98d939eb89725357f952b4";

const OFFICIAL_TOKENS: u32 = 1_276;
const OFFICIAL_HIDDEN_SIZE: u32 = 1_152;
const OFFICIAL_ATTENTION_HEADS: u32 = 16;
const OFFICIAL_HEAD_DIM: u32 = 72;
const OFFICIAL_INTERMEDIATE_SIZE: u32 = 4_304;
const OFFICIAL_LAYER_NORM_EPSILON: f32 = 1.0e-6;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionLayerSelfTestOracle {
    Synthetic,
    OfficialL3,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VisionLayerSelfTestDescriptor {
    pub schema_version: u32,
    pub oracle: VisionLayerSelfTestOracle,
    pub case_id: String,
    pub model_revision: String,
    pub golden_bundle_digest: Option<String>,
    pub semantic_fingerprint: Option<String>,
    pub tokens: u32,
    pub hidden_size: u32,
    pub attention_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub layer_norm_epsilon: f32,
    pub cu_seqlens: Vec<u32>,
    pub weights_bytes: u64,
    pub weights_blake3: String,
    pub expected_bytes: u64,
    pub expected_blake3: String,
    pub stage_order: [VisionEncoderLayerStage; 12],
}

#[derive(Clone, Copy, Debug)]
pub struct VisionLayerSelfTestSource<'a> {
    oracle: VisionLayerSelfTestOracle,
    case_id: &'a str,
    invocation: VisionEncoderLayerInvocation<'a>,
    expected: &'a BTreeMap<VisionEncoderLayerStage, Vec<f32>>,
}

impl<'a> VisionLayerSelfTestSource<'a> {
    #[must_use]
    pub const fn synthetic(
        case_id: &'a str,
        invocation: VisionEncoderLayerInvocation<'a>,
        expected: &'a BTreeMap<VisionEncoderLayerStage, Vec<f32>>,
    ) -> Self {
        Self {
            oracle: VisionLayerSelfTestOracle::Synthetic,
            case_id,
            invocation,
            expected,
        }
    }

    #[must_use]
    pub const fn official_layer_zero(
        invocation: VisionEncoderLayerInvocation<'a>,
        expected: &'a BTreeMap<VisionEncoderLayerStage, Vec<f32>>,
    ) -> Self {
        Self {
            oracle: VisionLayerSelfTestOracle::OfficialL3,
            case_id: OFFICIAL_VISION_LAYER_SELF_TEST_CASE_ID,
            invocation,
            expected,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionLayerSelfTestErrorCode {
    PackFormat,
    MissingSection,
    WrongSectionKind,
    InvalidDescriptor,
    DigestMismatch,
    LengthMismatch,
    InvalidInvocation,
    InvalidCheckpoint,
    OfficialIdentityMismatch,
    OfficialPayloadMismatch,
}

#[derive(Debug)]
pub struct VisionLayerSelfTestError {
    code: VisionLayerSelfTestErrorCode,
    message: String,
}

impl VisionLayerSelfTestError {
    #[must_use]
    pub const fn code(&self) -> VisionLayerSelfTestErrorCode {
        self.code
    }
}

impl fmt::Display for VisionLayerSelfTestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "vision-layer self-test {:?}: {}",
            self.code, self.message
        )
    }
}

impl Error for VisionLayerSelfTestError {}

#[derive(Debug)]
pub struct VisionLayerSelfTestPack {
    descriptor: VisionLayerSelfTestDescriptor,
    invocation: OwnedVisionEncoderLayerInvocation,
    expected: BTreeMap<VisionEncoderLayerStage, Vec<f32>>,
}

impl VisionLayerSelfTestPack {
    pub fn open(bytes: &[u8]) -> Result<Self, VisionLayerSelfTestError> {
        let reader = PackReader::open(bytes)
            .map_err(|error| pack_error(format!("cannot open outer pack: {error}")))?;
        let descriptor_bytes = required_section(
            &reader,
            VISION_LAYER_SELF_TEST_DESCRIPTOR_ID,
            SectionKind::SemanticIr,
        )?;
        let weights = required_section(
            &reader,
            VISION_LAYER_SELF_TEST_WEIGHTS_ID,
            SectionKind::WeightShard,
        )?;
        let expected_bytes = required_section(
            &reader,
            VISION_LAYER_SELF_TEST_EXPECTED_ID,
            SectionKind::SelfTest,
        )?;
        let descriptor = parse_vision_layer_self_test_descriptor(descriptor_bytes)?;
        let invocation = decode_invocation(&descriptor, weights)?;
        let expected = decode_checkpoints(&descriptor, expected_bytes)?;
        Ok(Self {
            descriptor,
            invocation,
            expected,
        })
    }

    #[must_use]
    pub const fn descriptor(&self) -> &VisionLayerSelfTestDescriptor {
        &self.descriptor
    }

    #[must_use]
    pub const fn invocation(&self) -> &OwnedVisionEncoderLayerInvocation {
        &self.invocation
    }

    #[must_use]
    pub fn expected(&self, stage: VisionEncoderLayerStage) -> &[f32] {
        self.expected
            .get(&stage)
            .expect("validated self-test packs contain every stage")
    }
}

pub fn build_vision_layer_self_test_pack(
    compiler_build: &str,
    source: VisionLayerSelfTestSource<'_>,
) -> Result<Vec<u8>, VisionLayerSelfTestError> {
    source
        .invocation
        .plan()
        .map_err(|error| invalid_invocation(format!("source invocation is invalid: {error}")))?;
    validate_case_id(source.case_id)?;
    let weights = encode_invocation(source.invocation);
    let expected = encode_checkpoints(source.invocation, source.expected)?;
    let descriptor = VisionLayerSelfTestDescriptor {
        schema_version: VISION_LAYER_SELF_TEST_SCHEMA_VERSION,
        oracle: source.oracle,
        case_id: source.case_id.to_owned(),
        model_revision: MODEL_REVISION.to_owned(),
        golden_bundle_digest: (source.oracle == VisionLayerSelfTestOracle::OfficialL3)
            .then(|| OFFICIAL_VISION_LAYER_GOLDEN_BUNDLE_DIGEST.to_owned()),
        semantic_fingerprint: (source.oracle == VisionLayerSelfTestOracle::OfficialL3)
            .then(|| OFFICIAL_VISION_LAYER_SEMANTIC_FINGERPRINT.to_owned()),
        tokens: source.invocation.tokens,
        hidden_size: source.invocation.hidden_size,
        attention_heads: source.invocation.attention_heads,
        head_dim: source.invocation.head_dim,
        intermediate_size: source.invocation.intermediate_size,
        layer_norm_epsilon: source.invocation.layer_norm_epsilon,
        cu_seqlens: source.invocation.cu_seqlens.to_vec(),
        weights_bytes: weights.len() as u64,
        weights_blake3: blake3::hash(&weights).to_hex().to_string(),
        expected_bytes: expected.len() as u64,
        expected_blake3: blake3::hash(&expected).to_hex().to_string(),
        stage_order: VisionEncoderLayerStage::ALL,
    };
    validate_descriptor(&descriptor)?;
    let descriptor_bytes = canonical_descriptor_bytes(&descriptor)?;
    let manifest = PackManifest::paddleocr_vl_16(compiler_build)
        .map_err(|error| pack_error(format!("cannot create pack manifest: {error}")))?;
    let mut builder = PackBuilder::new(manifest);
    for section in [
        PackSection::new(
            VISION_LAYER_SELF_TEST_DESCRIPTOR_ID,
            SectionKind::SemanticIr,
            64,
            descriptor_bytes,
        ),
        PackSection::new(
            VISION_LAYER_SELF_TEST_EXPECTED_ID,
            SectionKind::SelfTest,
            256,
            expected,
        ),
        PackSection::new(
            VISION_LAYER_SELF_TEST_WEIGHTS_ID,
            SectionKind::WeightShard,
            256,
            weights,
        ),
    ] {
        builder
            .add_section(section)
            .map_err(|error| pack_error(format!("cannot add self-test section: {error}")))?;
    }
    builder
        .build()
        .map_err(|error| pack_error(format!("cannot build self-test pack: {error}")))
}

pub fn parse_vision_layer_self_test_descriptor(
    json: &[u8],
) -> Result<VisionLayerSelfTestDescriptor, VisionLayerSelfTestError> {
    let value: serde_json::Value = serde_json::from_slice(json)
        .map_err(|error| invalid_descriptor(format!("descriptor JSON is invalid: {error}")))?;
    let descriptor: VisionLayerSelfTestDescriptor = serde_json::from_value(value)
        .map_err(|error| invalid_descriptor(format!("descriptor schema is invalid: {error}")))?;
    if canonical_descriptor_bytes(&descriptor)? != json {
        return Err(invalid_descriptor(
            "descriptor JSON is not in canonical byte form",
        ));
    }
    validate_descriptor(&descriptor)?;
    Ok(descriptor)
}

pub fn decode_vision_layer_self_test_invocation(
    descriptor: &VisionLayerSelfTestDescriptor,
    weights: &[u8],
) -> Result<OwnedVisionEncoderLayerInvocation, VisionLayerSelfTestError> {
    validate_descriptor(descriptor)?;
    decode_invocation(descriptor, weights)
}

fn required_section<'a>(
    reader: &'a PackReader<'_>,
    id: &str,
    expected_kind: SectionKind,
) -> Result<&'a [u8], VisionLayerSelfTestError> {
    let entry = reader
        .entries()
        .iter()
        .find(|entry| entry.id == id)
        .ok_or_else(|| {
            error(
                VisionLayerSelfTestErrorCode::MissingSection,
                format!("missing {id}"),
            )
        })?;
    if entry.kind != expected_kind {
        return Err(error(
            VisionLayerSelfTestErrorCode::WrongSectionKind,
            format!("{id} has {:?}, expected {expected_kind:?}", entry.kind),
        ));
    }
    reader.section(id).ok_or_else(|| {
        error(
            VisionLayerSelfTestErrorCode::MissingSection,
            format!("cannot read {id}"),
        )
    })
}

fn validate_descriptor(
    descriptor: &VisionLayerSelfTestDescriptor,
) -> Result<(), VisionLayerSelfTestError> {
    if descriptor.schema_version != VISION_LAYER_SELF_TEST_SCHEMA_VERSION {
        return Err(invalid_descriptor(format!(
            "unsupported schema version {}",
            descriptor.schema_version
        )));
    }
    validate_case_id(&descriptor.case_id)?;
    if descriptor.stage_order != VisionEncoderLayerStage::ALL {
        return Err(invalid_descriptor("semantic stage order is not canonical"));
    }
    if descriptor.oracle == VisionLayerSelfTestOracle::Synthetic
        && descriptor.model_revision != MODEL_REVISION
    {
        return Err(invalid_descriptor(
            "model revision does not match the compiler",
        ));
    }
    for (label, digest) in [
        ("weights", descriptor.weights_blake3.as_str()),
        ("expected", descriptor.expected_blake3.as_str()),
    ] {
        if !is_blake3_hex(digest) {
            return Err(invalid_descriptor(format!("{label} BLAKE3 is invalid")));
        }
    }
    if !descriptor.weights_bytes.is_multiple_of(4) || !descriptor.expected_bytes.is_multiple_of(4) {
        return Err(error(
            VisionLayerSelfTestErrorCode::LengthMismatch,
            "F32 payload lengths must be multiples of four",
        ));
    }
    if descriptor.oracle == VisionLayerSelfTestOracle::OfficialL3 {
        validate_official_descriptor(descriptor)?;
    } else if descriptor.golden_bundle_digest.is_some()
        || descriptor.semantic_fingerprint.is_some()
        || descriptor.case_id == OFFICIAL_VISION_LAYER_SELF_TEST_CASE_ID
    {
        return Err(error(
            VisionLayerSelfTestErrorCode::OfficialIdentityMismatch,
            "official descriptor oracle or provenance drifted",
        ));
    }
    let weights_bytes = derived_weights_bytes(descriptor)?;
    let expected_bytes = derived_expected_bytes(descriptor)?;
    if descriptor.weights_bytes != weights_bytes || descriptor.expected_bytes != expected_bytes {
        return Err(error(
            VisionLayerSelfTestErrorCode::LengthMismatch,
            format!(
                "descriptor payload lengths are {}, {}; geometry requires {weights_bytes}, {expected_bytes}",
                descriptor.weights_bytes, descriptor.expected_bytes
            ),
        ));
    }
    Ok(())
}

fn validate_official_descriptor(
    descriptor: &VisionLayerSelfTestDescriptor,
) -> Result<(), VisionLayerSelfTestError> {
    let identity_matches = descriptor.case_id == OFFICIAL_VISION_LAYER_SELF_TEST_CASE_ID
        && descriptor.model_revision == MODEL_REVISION
        && descriptor.golden_bundle_digest.as_deref()
            == Some(OFFICIAL_VISION_LAYER_GOLDEN_BUNDLE_DIGEST)
        && descriptor.semantic_fingerprint.as_deref()
            == Some(OFFICIAL_VISION_LAYER_SEMANTIC_FINGERPRINT)
        && descriptor.tokens == OFFICIAL_TOKENS
        && descriptor.hidden_size == OFFICIAL_HIDDEN_SIZE
        && descriptor.attention_heads == OFFICIAL_ATTENTION_HEADS
        && descriptor.head_dim == OFFICIAL_HEAD_DIM
        && descriptor.intermediate_size == OFFICIAL_INTERMEDIATE_SIZE
        && descriptor.layer_norm_epsilon.to_bits() == OFFICIAL_LAYER_NORM_EPSILON.to_bits()
        && descriptor.cu_seqlens == [0, OFFICIAL_TOKENS];
    if !identity_matches {
        return Err(error(
            VisionLayerSelfTestErrorCode::OfficialIdentityMismatch,
            "official L3 descriptor identity or geometry drifted",
        ));
    }
    if descriptor.weights_bytes != OFFICIAL_VISION_LAYER_WEIGHTS_BYTES
        || descriptor.expected_bytes != OFFICIAL_VISION_LAYER_EXPECTED_BYTES
        || descriptor.weights_blake3 != OFFICIAL_VISION_LAYER_WEIGHTS_BLAKE3
        || descriptor.expected_blake3 != OFFICIAL_VISION_LAYER_EXPECTED_BLAKE3
    {
        return Err(error(
            VisionLayerSelfTestErrorCode::OfficialPayloadMismatch,
            "official L3 payload does not match its independently pinned anchor",
        ));
    }
    Ok(())
}

fn validate_case_id(case_id: &str) -> Result<(), VisionLayerSelfTestError> {
    if case_id.is_empty()
        || case_id.len() > 128
        || case_id.starts_with('/')
        || case_id.contains("..")
        || case_id.contains("//")
        || !case_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'/'))
    {
        return Err(invalid_descriptor("case ID is unsafe or malformed"));
    }
    Ok(())
}

fn canonical_descriptor_bytes(
    descriptor: &VisionLayerSelfTestDescriptor,
) -> Result<Vec<u8>, VisionLayerSelfTestError> {
    let value = serde_json::to_value(descriptor)
        .map_err(|error| invalid_descriptor(format!("cannot serialize descriptor: {error}")))?;
    let mut bytes = serde_json::to_vec(&value)
        .map_err(|error| invalid_descriptor(format!("cannot encode descriptor: {error}")))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn encode_invocation(invocation: VisionEncoderLayerInvocation<'_>) -> Vec<u8> {
    let parameters = invocation.parameters;
    let values = [
        invocation.input,
        parameters.norm1.weight,
        parameters.norm1.bias,
        parameters.query.weight,
        parameters.query.bias,
        parameters.key.weight,
        parameters.key.bias,
        parameters.value.weight,
        parameters.value.bias,
        parameters.attention_output.weight,
        parameters.attention_output.bias,
        parameters.norm2.weight,
        parameters.norm2.bias,
        parameters.mlp_fc1.weight,
        parameters.mlp_fc1.bias,
        parameters.mlp_fc2.weight,
        parameters.mlp_fc2.bias,
    ];
    let elements: usize = values.iter().map(|values| values.len()).sum();
    let mut bytes = Vec::with_capacity(elements * 4);
    for tensor in values {
        append_f32(&mut bytes, tensor);
    }
    bytes
}

fn encode_checkpoints(
    invocation: VisionEncoderLayerInvocation<'_>,
    checkpoints: &BTreeMap<VisionEncoderLayerStage, Vec<f32>>,
) -> Result<Vec<u8>, VisionLayerSelfTestError> {
    if checkpoints.len() != VisionEncoderLayerStage::ALL.len() {
        return Err(invalid_checkpoint("checkpoint set is incomplete"));
    }
    let expected_bytes = derived_expected_bytes(&descriptor_geometry(invocation)?)?;
    let mut bytes = Vec::with_capacity(
        usize::try_from(expected_bytes)
            .map_err(|_| invalid_checkpoint("checkpoint payload exceeds usize"))?,
    );
    for stage in VisionEncoderLayerStage::ALL {
        let values = checkpoints
            .get(&stage)
            .ok_or_else(|| invalid_checkpoint(format!("missing {stage:?}")))?;
        let expected = stage_elements(
            invocation.tokens,
            invocation.hidden_size,
            invocation.intermediate_size,
            stage,
        )?;
        if values.len() as u64 != expected || values.iter().any(|value| !value.is_finite()) {
            return Err(invalid_checkpoint(format!(
                "{stage:?} has invalid length or non-finite values"
            )));
        }
        append_f32(&mut bytes, values);
    }
    Ok(bytes)
}

fn decode_invocation(
    descriptor: &VisionLayerSelfTestDescriptor,
    weights: &[u8],
) -> Result<OwnedVisionEncoderLayerInvocation, VisionLayerSelfTestError> {
    require_payload(
        weights,
        descriptor.weights_bytes,
        &descriptor.weights_blake3,
        "weights",
    )?;
    let hidden = u64::from(descriptor.hidden_size);
    let intermediate = u64::from(descriptor.intermediate_size);
    let mut decoder = F32Decoder::new(weights);
    let mut take = |elements: u64, label: &str| decoder.take(elements, label);
    let invocation = OwnedVisionEncoderLayerInvocation {
        tokens: descriptor.tokens,
        hidden_size: descriptor.hidden_size,
        attention_heads: descriptor.attention_heads,
        head_dim: descriptor.head_dim,
        intermediate_size: descriptor.intermediate_size,
        layer_norm_epsilon: descriptor.layer_norm_epsilon,
        input: take(
            checked_mul(u64::from(descriptor.tokens), hidden, "input")?,
            "input",
        )?,
        cu_seqlens: descriptor.cu_seqlens.clone(),
        parameters: OwnedVisionEncoderLayerParameters {
            norm1: OwnedVisionLayerNormParameters {
                weight: take(hidden, "norm1.weight")?,
                bias: take(hidden, "norm1.bias")?,
            },
            query: decode_linear(&mut take, hidden, hidden, "query")?,
            key: decode_linear(&mut take, hidden, hidden, "key")?,
            value: decode_linear(&mut take, hidden, hidden, "value")?,
            attention_output: decode_linear(&mut take, hidden, hidden, "attention_output")?,
            norm2: OwnedVisionLayerNormParameters {
                weight: take(hidden, "norm2.weight")?,
                bias: take(hidden, "norm2.bias")?,
            },
            mlp_fc1: decode_linear(&mut take, hidden, intermediate, "mlp_fc1")?,
            mlp_fc2: decode_linear(&mut take, intermediate, hidden, "mlp_fc2")?,
        },
    };
    if !decoder.is_finished() {
        return Err(error(
            VisionLayerSelfTestErrorCode::LengthMismatch,
            "weight payload has trailing values",
        ));
    }
    invocation
        .borrowed()
        .plan()
        .map_err(|error| invalid_invocation(format!("decoded invocation is invalid: {error}")))?;
    Ok(invocation)
}

fn decode_linear(
    take: &mut impl FnMut(u64, &str) -> Result<Vec<f32>, VisionLayerSelfTestError>,
    input: u64,
    output: u64,
    label: &str,
) -> Result<OwnedVisionLinearParameters, VisionLayerSelfTestError> {
    Ok(OwnedVisionLinearParameters {
        weight: take(
            checked_mul(input, output, label)?,
            &format!("{label}.weight"),
        )?,
        bias: take(output, &format!("{label}.bias"))?,
    })
}

fn decode_checkpoints(
    descriptor: &VisionLayerSelfTestDescriptor,
    bytes: &[u8],
) -> Result<BTreeMap<VisionEncoderLayerStage, Vec<f32>>, VisionLayerSelfTestError> {
    require_payload(
        bytes,
        descriptor.expected_bytes,
        &descriptor.expected_blake3,
        "expected",
    )?;
    let mut decoder = F32Decoder::new(bytes);
    let mut checkpoints = BTreeMap::new();
    for stage in VisionEncoderLayerStage::ALL {
        let elements = stage_elements(
            descriptor.tokens,
            descriptor.hidden_size,
            descriptor.intermediate_size,
            stage,
        )?;
        let values = decoder
            .take(elements, stage.as_str())
            .map_err(|error| invalid_checkpoint(error.to_string()))?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid_checkpoint(format!(
                "{} contains a non-finite value",
                stage.as_str()
            )));
        }
        checkpoints.insert(stage, values);
    }
    if !decoder.is_finished() {
        return Err(error(
            VisionLayerSelfTestErrorCode::LengthMismatch,
            "checkpoint payload has trailing values",
        ));
    }
    Ok(checkpoints)
}

fn require_payload(
    bytes: &[u8],
    expected_bytes: u64,
    expected_digest: &str,
    label: &str,
) -> Result<(), VisionLayerSelfTestError> {
    if bytes.len() as u64 != expected_bytes {
        return Err(error(
            VisionLayerSelfTestErrorCode::LengthMismatch,
            format!(
                "{label} payload has {} bytes, expected {expected_bytes}",
                bytes.len()
            ),
        ));
    }
    if blake3::hash(bytes).to_hex().as_str() != expected_digest {
        return Err(error(
            VisionLayerSelfTestErrorCode::DigestMismatch,
            format!("{label} payload BLAKE3 mismatch"),
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

    fn take(&mut self, elements: u64, label: &str) -> Result<Vec<f32>, VisionLayerSelfTestError> {
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

fn descriptor_geometry(
    invocation: VisionEncoderLayerInvocation<'_>,
) -> Result<VisionLayerSelfTestDescriptor, VisionLayerSelfTestError> {
    Ok(VisionLayerSelfTestDescriptor {
        schema_version: VISION_LAYER_SELF_TEST_SCHEMA_VERSION,
        oracle: VisionLayerSelfTestOracle::Synthetic,
        case_id: "synthetic.geometry".to_owned(),
        model_revision: MODEL_REVISION.to_owned(),
        golden_bundle_digest: None,
        semantic_fingerprint: None,
        tokens: invocation.tokens,
        hidden_size: invocation.hidden_size,
        attention_heads: invocation.attention_heads,
        head_dim: invocation.head_dim,
        intermediate_size: invocation.intermediate_size,
        layer_norm_epsilon: invocation.layer_norm_epsilon,
        cu_seqlens: invocation.cu_seqlens.to_vec(),
        weights_bytes: derived_weights_bytes_from_geometry(
            invocation.tokens,
            invocation.hidden_size,
            invocation.intermediate_size,
        )?,
        weights_blake3: "0".repeat(64),
        expected_bytes: derived_expected_bytes_from_geometry(
            invocation.tokens,
            invocation.hidden_size,
            invocation.intermediate_size,
        )?,
        expected_blake3: "0".repeat(64),
        stage_order: VisionEncoderLayerStage::ALL,
    })
}

fn derived_weights_bytes(
    descriptor: &VisionLayerSelfTestDescriptor,
) -> Result<u64, VisionLayerSelfTestError> {
    derived_weights_bytes_from_geometry(
        descriptor.tokens,
        descriptor.hidden_size,
        descriptor.intermediate_size,
    )
}

fn derived_weights_bytes_from_geometry(
    tokens: u32,
    hidden: u32,
    intermediate: u32,
) -> Result<u64, VisionLayerSelfTestError> {
    let tokens = u64::from(tokens);
    let hidden = u64::from(hidden);
    let intermediate = u64::from(intermediate);
    let input = checked_mul(tokens, hidden, "input")?;
    let hidden_square = checked_mul(hidden, hidden, "hidden square")?;
    let hidden_intermediate = checked_mul(hidden, intermediate, "hidden intermediate")?;
    let elements = input
        .checked_add(
            hidden_square
                .checked_mul(4)
                .ok_or_else(|| invalid_descriptor("four hidden matrices overflowed"))?,
        )
        .and_then(|value| value.checked_add(hidden_intermediate.checked_mul(2)?))
        .and_then(|value| value.checked_add(hidden.checked_mul(9)?))
        .and_then(|value| value.checked_add(intermediate))
        .ok_or_else(|| invalid_descriptor("weight element count overflowed"))?;
    elements
        .checked_mul(4)
        .ok_or_else(|| invalid_descriptor("weight byte count overflowed"))
}

fn derived_expected_bytes(
    descriptor: &VisionLayerSelfTestDescriptor,
) -> Result<u64, VisionLayerSelfTestError> {
    derived_expected_bytes_from_geometry(
        descriptor.tokens,
        descriptor.hidden_size,
        descriptor.intermediate_size,
    )
}

fn derived_expected_bytes_from_geometry(
    tokens: u32,
    hidden: u32,
    intermediate: u32,
) -> Result<u64, VisionLayerSelfTestError> {
    let hidden_elements = checked_mul(u64::from(tokens), u64::from(hidden), "hidden stage")?;
    let intermediate_elements = checked_mul(
        u64::from(tokens),
        u64::from(intermediate),
        "intermediate stage",
    )?;
    hidden_elements
        .checked_mul(10)
        .and_then(|value| value.checked_add(intermediate_elements.checked_mul(2)?))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| invalid_descriptor("checkpoint byte count overflowed"))
}

fn stage_elements(
    tokens: u32,
    hidden: u32,
    intermediate: u32,
    stage: VisionEncoderLayerStage,
) -> Result<u64, VisionLayerSelfTestError> {
    checked_mul(
        u64::from(tokens),
        u64::from(
            if matches!(
                stage,
                VisionEncoderLayerStage::MlpFc1 | VisionEncoderLayerStage::MlpActivation
            ) {
                intermediate
            } else {
                hidden
            },
        ),
        stage.as_str(),
    )
}

fn checked_mul(left: u64, right: u64, label: &str) -> Result<u64, VisionLayerSelfTestError> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_descriptor(format!("{label} element count overflowed")))
}

fn is_blake3_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn error(
    code: VisionLayerSelfTestErrorCode,
    message: impl Into<String>,
) -> VisionLayerSelfTestError {
    VisionLayerSelfTestError {
        code,
        message: message.into(),
    }
}

fn pack_error(message: impl Into<String>) -> VisionLayerSelfTestError {
    error(VisionLayerSelfTestErrorCode::PackFormat, message)
}

fn invalid_descriptor(message: impl Into<String>) -> VisionLayerSelfTestError {
    error(VisionLayerSelfTestErrorCode::InvalidDescriptor, message)
}

fn invalid_invocation(message: impl Into<String>) -> VisionLayerSelfTestError {
    error(VisionLayerSelfTestErrorCode::InvalidInvocation, message)
}

fn invalid_checkpoint(message: impl Into<String>) -> VisionLayerSelfTestError {
    error(VisionLayerSelfTestErrorCode::InvalidCheckpoint, message)
}
