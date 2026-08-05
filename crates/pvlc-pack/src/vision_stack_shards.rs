use std::{error::Error, fmt};

use pvlc_model_schema::{COMPILER_MODEL_ABI, MODEL_ID, MODEL_REVISION};
use pvlc_runtime_core::MAX_VISION_HEAD_DIM;
pub use pvlc_runtime_core::{DecoderWeightStorage, LinearWeightLayout};
use serde::{Deserialize, Serialize};

use crate::vision_layer_self_test::{
    OFFICIAL_VISION_LAYER_GOLDEN_BUNDLE_DIGEST, OFFICIAL_VISION_LAYER_SEMANTIC_FINGERPRINT,
};

pub const VISION_STACK_SHARD_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const OFFICIAL_VISION_STACK_CASE_ID: &str = "ocr.clean_latin.0001/vision.stack.27";
pub const OFFICIAL_VISION_STACK_TABLE_L2_CASE_ID: &str = "table.simple.0001/vision.stack.27";
pub const OFFICIAL_VISION_STACK_TABLE_L2_GOLDEN_BUNDLE_DIGEST: &str =
    "blake3:4cecc8b2abc37030920acf8860bb317084125dbbbcae5af394fccd0c8044c842";
pub const OFFICIAL_VISION_STACK_TABLE_L2_SEMANTIC_FINGERPRINT: &str =
    "blake3:91f72ffdf81070c2c9ca31628605064bdc02bf19417482cefce41d6a27aa5404";

const MAX_MANIFEST_BYTES: usize = 1_048_576;
const MAX_LAYER_COUNT: u32 = 256;
const OFFICIAL_TOKENS: u32 = 1_276;
const OFFICIAL_HIDDEN_SIZE: u32 = 1_152;
const OFFICIAL_ATTENTION_HEADS: u32 = 16;
const OFFICIAL_HEAD_DIM: u32 = 72;
const OFFICIAL_INTERMEDIATE_SIZE: u32 = 4_304;
const OFFICIAL_LAYER_NORM_EPSILON_BITS: u32 = 1.0e-6_f32.to_bits();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OfficialVisionStackProfile {
    OcrCleanLatinL3,
    TableSimpleL2,
}

impl OfficialVisionStackProfile {
    #[must_use]
    pub const fn case_id(self) -> &'static str {
        match self {
            Self::OcrCleanLatinL3 => OFFICIAL_VISION_STACK_CASE_ID,
            Self::TableSimpleL2 => OFFICIAL_VISION_STACK_TABLE_L2_CASE_ID,
        }
    }

    #[must_use]
    pub const fn golden_bundle_digest(self) -> &'static str {
        match self {
            Self::OcrCleanLatinL3 => OFFICIAL_VISION_LAYER_GOLDEN_BUNDLE_DIGEST,
            Self::TableSimpleL2 => OFFICIAL_VISION_STACK_TABLE_L2_GOLDEN_BUNDLE_DIGEST,
        }
    }

    #[must_use]
    pub const fn semantic_fingerprint(self) -> &'static str {
        match self {
            Self::OcrCleanLatinL3 => OFFICIAL_VISION_LAYER_SEMANTIC_FINGERPRINT,
            Self::TableSimpleL2 => OFFICIAL_VISION_STACK_TABLE_L2_SEMANTIC_FINGERPRINT,
        }
    }

    #[must_use]
    pub const fn tokens(self) -> u32 {
        match self {
            Self::OcrCleanLatinL3 => OFFICIAL_TOKENS,
            Self::TableSimpleL2 => 1_740,
        }
    }

    #[must_use]
    pub const fn checkpoint_layers(self) -> &'static [u32] {
        match self {
            Self::OcrCleanLatinL3 => &[0, 1, 13, 26],
            Self::TableSimpleL2 => &[],
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionStackShardOracle {
    Synthetic,
    OfficialMpsBf16,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionStackShardKind {
    Input,
    Layer,
    PostNorm,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VisionStackShardDescriptor {
    pub id: String,
    pub kind: VisionStackShardKind,
    pub layer_index: Option<u32>,
    pub bytes: u64,
    pub blake3: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VisionStackShardManifest {
    pub schema_version: u32,
    pub oracle: VisionStackShardOracle,
    pub case_id: String,
    pub model_id: String,
    pub model_revision: String,
    pub compiler_model_abi: u32,
    pub compiler_build: String,
    pub golden_bundle_digest: Option<String>,
    pub semantic_fingerprint: Option<String>,
    #[serde(
        default = "default_matrix_weight_storage",
        skip_serializing_if = "matrix_weight_storage_is_f32"
    )]
    pub matrix_weight_storage: DecoderWeightStorage,
    #[serde(default, skip_serializing_if = "matrix_weight_layout_is_output_major")]
    pub matrix_weight_layout: LinearWeightLayout,
    #[serde(
        default = "default_matrix_weight_storage",
        skip_serializing_if = "matrix_weight_storage_is_f32"
    )]
    pub vector_weight_storage: DecoderWeightStorage,
    #[serde(
        default = "default_matrix_weight_storage",
        skip_serializing_if = "matrix_weight_storage_is_f32"
    )]
    pub activation_storage: DecoderWeightStorage,
    pub tokens: u32,
    pub hidden_size: u32,
    pub attention_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub layer_norm_epsilon: f32,
    pub cu_seqlens: Vec<u32>,
    pub layer_count: u32,
    pub checkpoint_layers: Vec<u32>,
    pub shards: Vec<VisionStackShardDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct VisionStackShardPlan {
    pub matrix_weight_storage: DecoderWeightStorage,
    pub matrix_weight_layout: LinearWeightLayout,
    pub vector_weight_storage: DecoderWeightStorage,
    pub activation_storage: DecoderWeightStorage,
    pub layer_count: u32,
    pub shard_count: usize,
    pub input_bytes: u64,
    pub hidden_bytes: u64,
    pub intermediate_bytes: u64,
    pub layer_weight_bytes: u64,
    pub post_norm_bytes: u64,
    pub transport_bytes: u64,
    pub activation_buffer_count: u32,
    pub activation_arena_bytes: u64,
    pub readback_bytes: u64,
    pub peak_gpu_data_bytes: u64,
    pub submission_count: u32,
    pub compute_pass_count: u32,
    pub dispatch_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionStackLayerTensorRange {
    pub offset: u64,
    pub bytes: u64,
    pub storage: DecoderWeightStorage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionStackShardErrorCode {
    InvalidManifest,
    NonCanonicalManifest,
    ModelIdentityMismatch,
    OfficialIdentityMismatch,
    InvalidGeometry,
    InvalidCheckpointSelection,
    InvalidShardDirectory,
    LengthMismatch,
    DigestMismatch,
    NonFinitePayload,
    WrongShardOrder,
    InvalidPhase,
    ArithmeticOverflow,
}

#[derive(Debug)]
pub struct VisionStackShardError {
    code: VisionStackShardErrorCode,
    message: String,
}

impl VisionStackShardError {
    #[must_use]
    pub const fn code(&self) -> VisionStackShardErrorCode {
        self.code
    }
}

impl fmt::Display for VisionStackShardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "vision-stack shard {:?}: {}",
            self.code, self.message
        )
    }
}

impl Error for VisionStackShardError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionStackShardObservation {
    pub id: String,
    pub bytes: u64,
    pub blake3: String,
    pub all_finite: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionStackShardProtocolPhase {
    Preflight,
    Ready,
    Executing,
    Complete,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionStackShardAcceptance {
    pub phase: VisionStackShardProtocolPhase,
    pub id: String,
    pub kind: VisionStackShardKind,
    pub layer_index: Option<u32>,
    pub checkpoint_slot: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct VisionStackShardProtocol {
    manifest: VisionStackShardManifest,
    phase: VisionStackShardProtocolPhase,
    next_index: usize,
}

impl VisionStackShardManifest {
    pub fn plan(&self) -> Result<VisionStackShardPlan, VisionStackShardError> {
        validate_manifest_header(self)?;
        let hidden_elements = checked_mul(u64::from(self.tokens), u64::from(self.hidden_size))?;
        let intermediate_elements =
            checked_mul(u64::from(self.tokens), u64::from(self.intermediate_size))?;
        let hidden_bytes =
            checked_mul(hidden_elements, self.activation_storage.bytes_per_element())?;
        let intermediate_bytes = checked_mul(
            intermediate_elements,
            self.activation_storage.bytes_per_element(),
        )?;
        let hidden = u64::from(self.hidden_size);
        let layer_ranges = vision_stack_layer_weight_ranges_with_vector_storage(
            self.hidden_size,
            self.intermediate_size,
            self.matrix_weight_storage,
            self.vector_weight_storage,
        )?;
        let last_layer_range = layer_ranges
            .last()
            .expect("vision-stack layer layout has sixteen tensors");
        let layer_weight_bytes = checked_add(last_layer_range.offset, last_layer_range.bytes)?;
        let post_norm_bytes = checked_mul(
            checked_mul(hidden, 2)?,
            self.vector_weight_storage.bytes_per_element(),
        )?;
        validate_shard_directory(self, hidden_bytes, layer_weight_bytes, post_norm_bytes)?;

        let layer_transport = checked_mul(u64::from(self.layer_count), layer_weight_bytes)?;
        let transport_bytes =
            checked_add(checked_add(hidden_bytes, layer_transport)?, post_norm_bytes)?;
        let activation_arena_bytes = checked_add(
            checked_mul(hidden_bytes, 11)?,
            checked_mul(intermediate_bytes, 2)?,
        )?;
        let readback_copies = u64::try_from(self.checkpoint_layers.len())
            .map_err(|_| overflow("checkpoint count"))?
            .checked_add(1)
            .ok_or_else(|| overflow("checkpoint count"))?;
        let readback_bytes = checked_mul(hidden_bytes, readback_copies)?;
        let maximum_shard_bytes = hidden_bytes.max(layer_weight_bytes).max(post_norm_bytes);
        let peak_gpu_data_bytes = checked_add(
            checked_add(activation_arena_bytes, readback_bytes)?,
            maximum_shard_bytes,
        )?;
        let submission_count = self
            .layer_count
            .checked_add(1)
            .ok_or_else(|| overflow("submission count"))?;
        let dispatch_count = self
            .layer_count
            .checked_mul(12)
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| overflow("dispatch count"))?;

        Ok(VisionStackShardPlan {
            matrix_weight_storage: self.matrix_weight_storage,
            matrix_weight_layout: self.matrix_weight_layout,
            vector_weight_storage: self.vector_weight_storage,
            activation_storage: self.activation_storage,
            layer_count: self.layer_count,
            shard_count: self.shards.len(),
            input_bytes: hidden_bytes,
            hidden_bytes,
            intermediate_bytes,
            layer_weight_bytes,
            post_norm_bytes,
            transport_bytes,
            activation_buffer_count: 13,
            activation_arena_bytes,
            readback_bytes,
            peak_gpu_data_bytes,
            submission_count,
            compute_pass_count: submission_count,
            dispatch_count,
        })
    }
}

impl VisionStackShardProtocol {
    pub fn new(manifest: VisionStackShardManifest) -> Result<Self, VisionStackShardError> {
        manifest.plan()?;
        Ok(Self {
            manifest,
            phase: VisionStackShardProtocolPhase::Preflight,
            next_index: 0,
        })
    }

    #[must_use]
    pub const fn manifest(&self) -> &VisionStackShardManifest {
        &self.manifest
    }

    #[must_use]
    pub const fn phase(&self) -> VisionStackShardProtocolPhase {
        self.phase
    }

    #[must_use]
    pub fn next_shard_id(&self) -> Option<&str> {
        match self.phase {
            VisionStackShardProtocolPhase::Preflight | VisionStackShardProtocolPhase::Executing => {
                self.manifest
                    .shards
                    .get(self.next_index)
                    .map(|shard| shard.id.as_str())
            }
            VisionStackShardProtocolPhase::Ready => {
                self.manifest.shards.first().map(|shard| shard.id.as_str())
            }
            VisionStackShardProtocolPhase::Complete => None,
        }
    }

    pub fn accept_preflight(
        &mut self,
        observation: &VisionStackShardObservation,
    ) -> Result<VisionStackShardAcceptance, VisionStackShardError> {
        self.require_preflight_phase()?;
        self.validate_next(observation)?;
        Ok(self.advance_preflight())
    }

    /// Accepts only the manifest-declared shard identity during preflight.
    ///
    /// This avoids reading large payloads twice in streaming callers. It does
    /// not authenticate bytes: every execution call must still supply a full
    /// [`VisionStackShardObservation`] to [`Self::accept_execution`].
    pub fn accept_deferred_preflight(
        &mut self,
        shard_id: &str,
    ) -> Result<VisionStackShardAcceptance, VisionStackShardError> {
        self.require_preflight_phase()?;
        let descriptor = &self.manifest.shards[self.next_index];
        if shard_id != descriptor.id {
            return Err(error(
                VisionStackShardErrorCode::WrongShardOrder,
                format!("received {shard_id}, expected {}", descriptor.id),
            ));
        }
        Ok(self.advance_preflight())
    }

    fn require_preflight_phase(&self) -> Result<(), VisionStackShardError> {
        if self.phase != VisionStackShardProtocolPhase::Preflight {
            return Err(error(
                VisionStackShardErrorCode::InvalidPhase,
                "preflight shard supplied outside preflight",
            ));
        }
        Ok(())
    }

    fn advance_preflight(&mut self) -> VisionStackShardAcceptance {
        let accepted = acceptance(
            VisionStackShardProtocolPhase::Preflight,
            &self.manifest.shards[self.next_index],
            None,
        );
        self.next_index += 1;
        if self.next_index == self.manifest.shards.len() {
            self.phase = VisionStackShardProtocolPhase::Ready;
            self.next_index = 0;
        }
        accepted
    }

    pub fn accept_execution(
        &mut self,
        observation: &VisionStackShardObservation,
    ) -> Result<VisionStackShardAcceptance, VisionStackShardError> {
        if !matches!(
            self.phase,
            VisionStackShardProtocolPhase::Ready | VisionStackShardProtocolPhase::Executing
        ) {
            return Err(error(
                VisionStackShardErrorCode::InvalidPhase,
                "execution shard supplied before complete preflight or after completion",
            ));
        }
        let descriptor = self.validate_next(observation)?;
        let checkpoint_slot = match descriptor.kind {
            VisionStackShardKind::Layer => descriptor.layer_index.and_then(|layer| {
                self.manifest
                    .checkpoint_layers
                    .iter()
                    .position(|selected| *selected == layer)
            }),
            VisionStackShardKind::PostNorm => Some(self.manifest.checkpoint_layers.len()),
            VisionStackShardKind::Input => None,
        };
        let accepted = acceptance(
            VisionStackShardProtocolPhase::Executing,
            descriptor,
            checkpoint_slot,
        );
        self.next_index += 1;
        if self.next_index == self.manifest.shards.len() {
            self.phase = VisionStackShardProtocolPhase::Complete;
        } else {
            self.phase = VisionStackShardProtocolPhase::Executing;
        }
        Ok(accepted)
    }

    fn validate_next(
        &self,
        observation: &VisionStackShardObservation,
    ) -> Result<&VisionStackShardDescriptor, VisionStackShardError> {
        let descriptor = &self.manifest.shards[self.next_index];
        if observation.id != descriptor.id {
            return Err(error(
                VisionStackShardErrorCode::WrongShardOrder,
                format!("received {}, expected {}", observation.id, descriptor.id),
            ));
        }
        if observation.bytes != descriptor.bytes {
            return Err(error(
                VisionStackShardErrorCode::LengthMismatch,
                format!(
                    "{} has {} bytes, expected {}",
                    descriptor.id, observation.bytes, descriptor.bytes
                ),
            ));
        }
        if observation.blake3 != descriptor.blake3 {
            return Err(error(
                VisionStackShardErrorCode::DigestMismatch,
                format!("{} BLAKE3 does not match the manifest", descriptor.id),
            ));
        }
        if !observation.all_finite {
            return Err(error(
                VisionStackShardErrorCode::NonFinitePayload,
                format!("{} contains a non-finite stored value", descriptor.id),
            ));
        }
        Ok(descriptor)
    }
}

pub fn canonical_vision_stack_shard_manifest_bytes(
    manifest: &VisionStackShardManifest,
) -> Result<Vec<u8>, VisionStackShardError> {
    manifest.plan()?;
    let mut bytes = serde_json::to_vec(manifest).map_err(|source| {
        error(
            VisionStackShardErrorCode::InvalidManifest,
            format!("cannot serialize canonical manifest: {source}"),
        )
    })?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn parse_vision_stack_shard_manifest(
    bytes: &[u8],
) -> Result<VisionStackShardManifest, VisionStackShardError> {
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(error(
            VisionStackShardErrorCode::InvalidManifest,
            "manifest exceeds its byte bound",
        ));
    }
    let manifest: VisionStackShardManifest = serde_json::from_slice(bytes).map_err(|source| {
        error(
            VisionStackShardErrorCode::InvalidManifest,
            format!("manifest JSON or schema is invalid: {source}"),
        )
    })?;
    let canonical = canonical_vision_stack_shard_manifest_bytes(&manifest)?;
    if canonical != bytes {
        return Err(error(
            VisionStackShardErrorCode::NonCanonicalManifest,
            "manifest JSON is not in canonical byte form",
        ));
    }
    Ok(manifest)
}

#[must_use]
pub fn inspect_vision_stack_f32_shard(id: &str, bytes: &[u8]) -> VisionStackShardObservation {
    let all_finite = finite_storage_bytes(DecoderWeightStorage::F32, bytes);
    VisionStackShardObservation {
        id: id.to_owned(),
        bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        blake3: blake3::hash(bytes).to_hex().to_string(),
        all_finite,
    }
}

pub fn inspect_vision_stack_shard(
    manifest: &VisionStackShardManifest,
    id: &str,
    bytes: &[u8],
) -> Result<VisionStackShardObservation, VisionStackShardError> {
    manifest.plan()?;
    let descriptor = manifest
        .shards
        .iter()
        .find(|descriptor| descriptor.id == id)
        .ok_or_else(|| {
            error(
                VisionStackShardErrorCode::InvalidShardDirectory,
                format!("vision-stack shard {id} is not declared"),
            )
        })?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != descriptor.bytes {
        return Err(error(
            VisionStackShardErrorCode::LengthMismatch,
            format!(
                "{} has {} bytes, expected {}",
                descriptor.id,
                bytes.len(),
                descriptor.bytes
            ),
        ));
    }
    let all_finite = match descriptor.kind {
        VisionStackShardKind::Input | VisionStackShardKind::PostNorm => {
            let storage = match descriptor.kind {
                VisionStackShardKind::Input => manifest.activation_storage,
                VisionStackShardKind::PostNorm => manifest.vector_weight_storage,
                VisionStackShardKind::Layer => unreachable!(),
            };
            finite_storage_bytes(storage, bytes)
        }
        VisionStackShardKind::Layer => {
            let ranges = vision_stack_layer_weight_ranges_with_vector_storage(
                manifest.hidden_size,
                manifest.intermediate_size,
                manifest.matrix_weight_storage,
                manifest.vector_weight_storage,
            )?;
            ranges.into_iter().all(|range| {
                let begin = usize::try_from(range.offset).ok();
                let end = range
                    .offset
                    .checked_add(range.bytes)
                    .and_then(|end| usize::try_from(end).ok());
                begin
                    .zip(end)
                    .and_then(|(begin, end)| bytes.get(begin..end))
                    .is_some_and(|payload| finite_storage_bytes(range.storage, payload))
            })
        }
    };
    Ok(VisionStackShardObservation {
        id: id.to_owned(),
        bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        blake3: blake3::hash(bytes).to_hex().to_string(),
        all_finite,
    })
}

pub fn vision_stack_layer_weight_ranges(
    hidden_size: u32,
    intermediate_size: u32,
    matrix_weight_storage: DecoderWeightStorage,
) -> Result<[VisionStackLayerTensorRange; 16], VisionStackShardError> {
    vision_stack_layer_weight_ranges_with_vector_storage(
        hidden_size,
        intermediate_size,
        matrix_weight_storage,
        DecoderWeightStorage::F32,
    )
}

pub fn vision_stack_layer_weight_ranges_with_vector_storage(
    hidden_size: u32,
    intermediate_size: u32,
    matrix_weight_storage: DecoderWeightStorage,
    vector_weight_storage: DecoderWeightStorage,
) -> Result<[VisionStackLayerTensorRange; 16], VisionStackShardError> {
    if hidden_size == 0 || intermediate_size == 0 {
        return Err(error(
            VisionStackShardErrorCode::InvalidGeometry,
            "vision-stack layer tensor dimensions must be positive",
        ));
    }
    let hidden = u64::from(hidden_size);
    let intermediate = u64::from(intermediate_size);
    let vector = |elements| checked_mul(elements, vector_weight_storage.bytes_per_element());
    let matrix = |elements| checked_mul(elements, matrix_weight_storage.bytes_per_element());
    let hidden_vector = vector(hidden)?;
    let hidden_matrix = matrix(checked_mul(hidden, hidden)?)?;
    let fc1_matrix = matrix(checked_mul(intermediate, hidden)?)?;
    let intermediate_vector = vector(intermediate)?;
    let fc2_matrix = matrix(checked_mul(hidden, intermediate)?)?;
    let specifications = [
        (hidden_vector, vector_weight_storage),
        (hidden_vector, vector_weight_storage),
        (hidden_matrix, matrix_weight_storage),
        (hidden_vector, vector_weight_storage),
        (hidden_matrix, matrix_weight_storage),
        (hidden_vector, vector_weight_storage),
        (hidden_matrix, matrix_weight_storage),
        (hidden_vector, vector_weight_storage),
        (hidden_matrix, matrix_weight_storage),
        (hidden_vector, vector_weight_storage),
        (hidden_vector, vector_weight_storage),
        (hidden_vector, vector_weight_storage),
        (fc1_matrix, matrix_weight_storage),
        (intermediate_vector, vector_weight_storage),
        (fc2_matrix, matrix_weight_storage),
        (hidden_vector, vector_weight_storage),
    ];
    let mut ranges = [VisionStackLayerTensorRange {
        offset: 0,
        bytes: 0,
        storage: DecoderWeightStorage::F32,
    }; 16];
    let mut offset = 0_u64;
    for (slot, (bytes, storage)) in specifications.into_iter().enumerate() {
        ranges[slot] = VisionStackLayerTensorRange {
            offset,
            bytes,
            storage,
        };
        offset = checked_add(offset, bytes)?;
    }
    Ok(ranges)
}

fn finite_storage_bytes(storage: DecoderWeightStorage, bytes: &[u8]) -> bool {
    storage.validate_finite_bytes(bytes).is_ok()
}

const fn default_matrix_weight_storage() -> DecoderWeightStorage {
    DecoderWeightStorage::F32
}

const fn matrix_weight_storage_is_f32(storage: &DecoderWeightStorage) -> bool {
    matches!(storage, DecoderWeightStorage::F32)
}

const fn matrix_weight_layout_is_output_major(layout: &LinearWeightLayout) -> bool {
    matches!(layout, LinearWeightLayout::OutputMajor)
}

fn validate_manifest_header(
    manifest: &VisionStackShardManifest,
) -> Result<(), VisionStackShardError> {
    if manifest.schema_version != VISION_STACK_SHARD_MANIFEST_SCHEMA_VERSION {
        return Err(error(
            VisionStackShardErrorCode::InvalidManifest,
            format!("unsupported schema version {}", manifest.schema_version),
        ));
    }
    let legacy_f32_activations = manifest.vector_weight_storage == DecoderWeightStorage::F32
        && manifest.activation_storage == DecoderWeightStorage::F32;
    let full_f16 = manifest.matrix_weight_storage == DecoderWeightStorage::F16
        && manifest.matrix_weight_layout == LinearWeightLayout::InputMajor
        && manifest.vector_weight_storage == DecoderWeightStorage::F16
        && manifest.activation_storage == DecoderWeightStorage::F16;
    if !legacy_f32_activations && !full_f16 {
        return Err(error(
            VisionStackShardErrorCode::InvalidManifest,
            "vision execution must use either legacy F32 activations/vectors or one coherent full FP16 matrix/vector/activation profile",
        ));
    }
    if manifest.matrix_weight_layout == LinearWeightLayout::InputMajor
        && manifest.matrix_weight_storage != DecoderWeightStorage::F16
    {
        return Err(error(
            VisionStackShardErrorCode::InvalidManifest,
            "input-major linear weights require F16 matrix storage",
        ));
    }
    if manifest.model_id != MODEL_ID
        || manifest.model_revision != MODEL_REVISION
        || manifest.compiler_model_abi != COMPILER_MODEL_ABI
    {
        return Err(error(
            VisionStackShardErrorCode::ModelIdentityMismatch,
            "manifest does not identify the pinned PaddleOCR-VL compiler ABI",
        ));
    }
    if !is_lower_hex_64(&manifest.compiler_build)
        || manifest.case_id.is_empty()
        || manifest.case_id.len() > 256
        || !manifest
            .case_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._/-".contains(&byte))
    {
        return Err(error(
            VisionStackShardErrorCode::InvalidManifest,
            "case ID or compiler build is invalid",
        ));
    }
    for provenance in [
        manifest.golden_bundle_digest.as_deref(),
        manifest.semantic_fingerprint.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if provenance
            .strip_prefix("blake3:")
            .is_none_or(|digest| !is_lower_hex_64(digest))
        {
            return Err(error(
                VisionStackShardErrorCode::InvalidManifest,
                "provenance digest is invalid",
            ));
        }
    }
    validate_geometry(manifest)?;
    validate_checkpoint_selection(manifest)?;
    validate_oracle_identity(manifest)
}

fn validate_geometry(manifest: &VisionStackShardManifest) -> Result<(), VisionStackShardError> {
    if manifest.tokens == 0
        || manifest.hidden_size == 0
        || manifest.attention_heads == 0
        || manifest.head_dim == 0
        || manifest.intermediate_size == 0
        || manifest.layer_count == 0
        || manifest.layer_count > MAX_LAYER_COUNT
        || !manifest.layer_norm_epsilon.is_finite()
        || manifest.layer_norm_epsilon <= 0.0
        || manifest.head_dim > MAX_VISION_HEAD_DIM
        || manifest.attention_heads.checked_mul(manifest.head_dim) != Some(manifest.hidden_size)
    {
        return Err(error(
            VisionStackShardErrorCode::InvalidGeometry,
            "vision-stack geometry, epsilon, or layer count is invalid",
        ));
    }
    if manifest.cu_seqlens.len() < 2
        || manifest.cu_seqlens.first() != Some(&0)
        || manifest.cu_seqlens.last() != Some(&manifest.tokens)
        || manifest
            .cu_seqlens
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(error(
            VisionStackShardErrorCode::InvalidGeometry,
            "sequence boundaries must be strictly increasing from zero to tokens",
        ));
    }
    Ok(())
}

fn validate_checkpoint_selection(
    manifest: &VisionStackShardManifest,
) -> Result<(), VisionStackShardError> {
    if manifest
        .checkpoint_layers
        .iter()
        .any(|layer| *layer >= manifest.layer_count)
        || manifest
            .checkpoint_layers
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
    {
        return Err(error(
            VisionStackShardErrorCode::InvalidCheckpointSelection,
            "checkpoint layers must be unique, increasing, and in range",
        ));
    }
    Ok(())
}

fn validate_oracle_identity(
    manifest: &VisionStackShardManifest,
) -> Result<(), VisionStackShardError> {
    match manifest.oracle {
        VisionStackShardOracle::Synthetic => {
            if manifest.golden_bundle_digest.is_some() || manifest.semantic_fingerprint.is_some() {
                return Err(error(
                    VisionStackShardErrorCode::InvalidManifest,
                    "synthetic manifest cannot claim official provenance",
                ));
            }
        }
        VisionStackShardOracle::OfficialMpsBf16 => {
            let profile_matches = [
                OfficialVisionStackProfile::OcrCleanLatinL3,
                OfficialVisionStackProfile::TableSimpleL2,
            ]
            .into_iter()
            .any(|profile| official_profile_matches(manifest, profile));
            if !profile_matches
                || manifest.hidden_size != OFFICIAL_HIDDEN_SIZE
                || manifest.attention_heads != OFFICIAL_ATTENTION_HEADS
                || manifest.head_dim != OFFICIAL_HEAD_DIM
                || manifest.intermediate_size != OFFICIAL_INTERMEDIATE_SIZE
                || manifest.layer_norm_epsilon.to_bits() != OFFICIAL_LAYER_NORM_EPSILON_BITS
                || manifest.layer_count != 27
            {
                return Err(error(
                    VisionStackShardErrorCode::OfficialIdentityMismatch,
                    "official stack identity or provenance drifted",
                ));
            }
        }
    }
    Ok(())
}

fn official_profile_matches(
    manifest: &VisionStackShardManifest,
    profile: OfficialVisionStackProfile,
) -> bool {
    manifest.case_id == profile.case_id()
        && manifest.golden_bundle_digest.as_deref() == Some(profile.golden_bundle_digest())
        && manifest.semantic_fingerprint.as_deref() == Some(profile.semantic_fingerprint())
        && manifest.tokens == profile.tokens()
        && manifest.cu_seqlens == [0, profile.tokens()]
        && manifest.checkpoint_layers == profile.checkpoint_layers()
}

fn validate_shard_directory(
    manifest: &VisionStackShardManifest,
    input_bytes: u64,
    layer_weight_bytes: u64,
    post_norm_bytes: u64,
) -> Result<(), VisionStackShardError> {
    let expected_count = usize::try_from(manifest.layer_count)
        .map_err(|_| overflow("shard count"))?
        .checked_add(2)
        .ok_or_else(|| overflow("shard count"))?;
    if manifest.shards.len() != expected_count {
        return Err(error(
            VisionStackShardErrorCode::InvalidShardDirectory,
            format!(
                "manifest has {} shards, expected {expected_count}",
                manifest.shards.len()
            ),
        ));
    }
    validate_shard(
        &manifest.shards[0],
        "input.embeddings",
        VisionStackShardKind::Input,
        None,
        input_bytes,
    )?;
    for layer in 0..manifest.layer_count {
        let position = usize::try_from(layer)
            .map_err(|_| overflow("layer position"))?
            .checked_add(1)
            .ok_or_else(|| overflow("layer position"))?;
        validate_shard(
            &manifest.shards[position],
            &format!("weights.vision_layer.{layer:02}"),
            VisionStackShardKind::Layer,
            Some(layer),
            layer_weight_bytes,
        )?;
    }
    validate_shard(
        manifest
            .shards
            .last()
            .expect("validated shard directory is nonempty"),
        "weights.vision_post_norm",
        VisionStackShardKind::PostNorm,
        None,
        post_norm_bytes,
    )
}

fn validate_shard(
    shard: &VisionStackShardDescriptor,
    id: &str,
    kind: VisionStackShardKind,
    layer_index: Option<u32>,
    bytes: u64,
) -> Result<(), VisionStackShardError> {
    if shard.id != id || shard.kind != kind || shard.layer_index != layer_index {
        return Err(error(
            VisionStackShardErrorCode::InvalidShardDirectory,
            format!("shard {} identity, kind, index, or order drifted", shard.id),
        ));
    }
    if shard.bytes != bytes {
        return Err(error(
            VisionStackShardErrorCode::LengthMismatch,
            format!("{} has {} bytes, expected {bytes}", shard.id, shard.bytes),
        ));
    }
    if !is_lower_hex_64(&shard.blake3) {
        return Err(error(
            VisionStackShardErrorCode::InvalidShardDirectory,
            format!("{} BLAKE3 is not canonical lowercase hex", shard.id),
        ));
    }
    Ok(())
}

fn acceptance(
    phase: VisionStackShardProtocolPhase,
    descriptor: &VisionStackShardDescriptor,
    checkpoint_slot: Option<usize>,
) -> VisionStackShardAcceptance {
    VisionStackShardAcceptance {
        phase,
        id: descriptor.id.clone(),
        kind: descriptor.kind,
        layer_index: descriptor.layer_index,
        checkpoint_slot,
    }
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn checked_mul(left: u64, right: u64) -> Result<u64, VisionStackShardError> {
    left.checked_mul(right)
        .ok_or_else(|| overflow("manifest arithmetic"))
}

fn checked_add(left: u64, right: u64) -> Result<u64, VisionStackShardError> {
    left.checked_add(right)
        .ok_or_else(|| overflow("manifest arithmetic"))
}

fn overflow(context: &str) -> VisionStackShardError {
    error(
        VisionStackShardErrorCode::ArithmeticOverflow,
        format!("{context} overflowed"),
    )
}

fn error(code: VisionStackShardErrorCode, message: impl Into<String>) -> VisionStackShardError {
    VisionStackShardError {
        code,
        message: message.into(),
    }
}
