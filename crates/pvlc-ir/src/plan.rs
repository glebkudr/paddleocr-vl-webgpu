use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use pvlc_runtime_core::{InvocationPlan, KernelId};
use serde::de::Error as _;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::SemanticId;

const PLAN_SCHEMA_VERSION: u32 = 1;
const MAX_PLAN_ID_BYTES: usize = 128;

#[derive(Clone, Copy)]
struct Producer<'a> {
    node_index: Option<usize>,
    buffer_id: &'a str,
}

macro_rules! plan_id {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn parse(value: &str) -> Result<Self, PlanIdError> {
                validate_plan_id(value).map(|()| Self(value.to_owned()))
            }

            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(&self.0)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                Self::parse(&value).map_err(D::Error::custom)
            }
        }
    };
}

plan_id!(PlanNodeId);
plan_id!(PlanValueId);
plan_id!(PlanBufferId);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanIdError(String);

impl fmt::Display for PlanIdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid PlanIR identifier {:?}", self.0)
    }
}

impl Error for PlanIdError {}

fn validate_plan_id(value: &str) -> Result<(), PlanIdError> {
    if value.is_empty() || value.len() > MAX_PLAN_ID_BYTES {
        return Err(PlanIdError(value.to_owned()));
    }
    for segment in value.split('.') {
        if segment.is_empty() || !valid_id_segment(segment) {
            return Err(PlanIdError(value.to_owned()));
        }
    }
    Ok(())
}

fn valid_id_segment(segment: &str) -> bool {
    if segment.bytes().all(|byte| byte.is_ascii_digit()) {
        return segment.len() == 2;
    }
    let mut bytes = segment.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    first.is_ascii_lowercase()
        && bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

impl Serialize for SemanticId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for SemanticId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value).map_err(D::Error::custom)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum PlanDtype {
    #[serde(rename = "bf16")]
    BFloat16,
    #[serde(rename = "f16")]
    Float16,
    #[serde(rename = "f32")]
    Float32,
}

impl PlanDtype {
    #[must_use]
    pub const fn byte_width(self) -> u64 {
        match self {
            Self::BFloat16 | Self::Float16 => 2,
            Self::Float32 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum PlanFeature {
    #[serde(rename = "shader_f16")]
    ShaderF16,
    #[serde(rename = "timestamp_query")]
    TimestampQuery,
}

impl PlanFeature {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ShaderF16 => "shader_f16",
            Self::TimestampQuery => "timestamp_query",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanBindingAccess {
    ReadOnlyStorage,
    ReadWriteStorage,
    Uniform,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanTensorResource {
    pub physical_name: String,
    pub semantic_id: SemanticId,
    pub dtype: PlanDtype,
    pub shape: Vec<u64>,
    pub storage_format: PlanDtype,
    pub buffer_id: PlanBufferId,
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanOutputBuffer {
    pub buffer_id: PlanBufferId,
    pub byte_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanUniformResource {
    pub words: [u32; 4],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanBindingResource {
    Value(PlanValueId),
    Tensor(PlanTensorResource),
    OutputBuffer(PlanOutputBuffer),
    UniformWords(PlanUniformResource),
}

impl Serialize for PlanBindingResource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Value(value_id) => {
                let mut state = serializer.serialize_struct("PlanBindingResource", 2)?;
                state.serialize_field("kind", "value")?;
                state.serialize_field("value_id", value_id)?;
                state.end()
            }
            Self::Tensor(tensor) => {
                let mut state = serializer.serialize_struct("PlanBindingResource", 9)?;
                state.serialize_field("kind", "tensor")?;
                state.serialize_field("physical_name", &tensor.physical_name)?;
                state.serialize_field("semantic_id", &tensor.semantic_id)?;
                state.serialize_field("dtype", &tensor.dtype)?;
                state.serialize_field("shape", &tensor.shape)?;
                state.serialize_field("storage_format", &tensor.storage_format)?;
                state.serialize_field("buffer_id", &tensor.buffer_id)?;
                state.serialize_field("byte_offset", &tensor.byte_offset)?;
                state.serialize_field("byte_length", &tensor.byte_length)?;
                state.end()
            }
            Self::OutputBuffer(output) => {
                let mut state = serializer.serialize_struct("PlanBindingResource", 3)?;
                state.serialize_field("kind", "output_buffer")?;
                state.serialize_field("buffer_id", &output.buffer_id)?;
                state.serialize_field("byte_length", &output.byte_length)?;
                state.end()
            }
            Self::UniformWords(uniform) => {
                let mut state = serializer.serialize_struct("PlanBindingResource", 2)?;
                state.serialize_field("kind", "uniform_words")?;
                state.serialize_field("words", &uniform.words)?;
                state.end()
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
enum PlanBindingResourceWire {
    #[serde(rename = "value")]
    Value { value_id: PlanValueId },
    #[serde(rename = "tensor")]
    Tensor {
        physical_name: String,
        semantic_id: SemanticId,
        dtype: PlanDtype,
        shape: Vec<u64>,
        storage_format: PlanDtype,
        buffer_id: PlanBufferId,
        byte_offset: u64,
        byte_length: u64,
    },
    #[serde(rename = "output_buffer")]
    OutputBuffer {
        buffer_id: PlanBufferId,
        byte_length: u64,
    },
    #[serde(rename = "uniform_words")]
    UniformWords { words: [u32; 4] },
}

impl<'de> Deserialize<'de> for PlanBindingResource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match PlanBindingResourceWire::deserialize(deserializer)? {
            PlanBindingResourceWire::Value { value_id } => Self::Value(value_id),
            PlanBindingResourceWire::Tensor {
                physical_name,
                semantic_id,
                dtype,
                shape,
                storage_format,
                buffer_id,
                byte_offset,
                byte_length,
            } => Self::Tensor(PlanTensorResource {
                physical_name,
                semantic_id,
                dtype,
                shape,
                storage_format,
                buffer_id,
                byte_offset,
                byte_length,
            }),
            PlanBindingResourceWire::OutputBuffer {
                buffer_id,
                byte_length,
            } => Self::OutputBuffer(PlanOutputBuffer {
                buffer_id,
                byte_length,
            }),
            PlanBindingResourceWire::UniformWords { words } => {
                Self::UniformWords(PlanUniformResource { words })
            }
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanBinding {
    pub number: u32,
    pub access: PlanBindingAccess,
    pub resource: PlanBindingResource,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanOutput {
    pub id: PlanValueId,
    pub dtype: PlanDtype,
    pub shape: Vec<u64>,
    pub buffer_id: PlanBufferId,
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanExternalValue {
    pub id: PlanValueId,
    pub dtype: PlanDtype,
    pub shape: Vec<u64>,
    pub buffer_id: PlanBufferId,
    pub byte_offset: u64,
    pub byte_length: u64,
}

mod invocation_serde {
    use super::*;

    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct InvocationPlanWire {
        kernel: KernelId,
        output_elements: usize,
        output_bytes: u64,
        workgroup_size: [u32; 3],
        dispatch: [u32; 3],
    }

    pub fn serialize<S>(invocation: &InvocationPlan, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        InvocationPlanWire {
            kernel: invocation.kernel,
            output_elements: invocation.output_elements,
            output_bytes: invocation.output_bytes,
            workgroup_size: invocation.workgroup_size,
            dispatch: invocation.dispatch,
        }
        .serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<InvocationPlan, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = InvocationPlanWire::deserialize(deserializer)?;
        Ok(InvocationPlan {
            kernel: wire.kernel,
            output_elements: wire.output_elements,
            output_bytes: wire.output_bytes,
            workgroup_size: wire.workgroup_size,
            dispatch: wire.dispatch,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanNodeSnapshot {
    pub id: PlanNodeId,
    #[serde(with = "invocation_serde")]
    pub invocation: InvocationPlan,
    pub uniform_words: [u32; 4],
    pub bindings: Vec<PlanBinding>,
    pub outputs: Vec<PlanOutput>,
    pub diagnostic_label: String,
    pub timestamp_label: Option<String>,
    pub source_semantic_ids: Vec<SemanticId>,
}

impl PlanNodeSnapshot {
    pub fn canonical_node_bytes(&self) -> Result<Vec<u8>, PlanError> {
        PlanNode {
            id: self.id.clone(),
            invocation: self.invocation,
            uniform_words: self.uniform_words,
            bindings: self.bindings.clone(),
            outputs: self.outputs.clone(),
            diagnostic_label: self.diagnostic_label.clone(),
            timestamp_label: self.timestamp_label.clone(),
            source_semantic_ids: self.source_semantic_ids.clone(),
            rewrite_provenance: None,
        }
        .canonical_bytes()
    }

    pub fn canonical_node_blake3_hex(&self) -> Result<String, PlanError> {
        Ok(blake3::hash(&self.canonical_node_bytes()?)
            .to_hex()
            .to_string())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanConsumedNode {
    pub role: String,
    pub original: PlanNodeSnapshot,
    pub canonical_blake3: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanRewriteProvenance {
    pub pass_id: String,
    pub source_semantic_ids: Vec<SemanticId>,
    pub consumed: Vec<PlanConsumedNode>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanNode {
    pub id: PlanNodeId,
    #[serde(with = "invocation_serde")]
    pub invocation: InvocationPlan,
    pub uniform_words: [u32; 4],
    pub bindings: Vec<PlanBinding>,
    pub outputs: Vec<PlanOutput>,
    pub diagnostic_label: String,
    pub timestamp_label: Option<String>,
    pub source_semantic_ids: Vec<SemanticId>,
    pub rewrite_provenance: Option<PlanRewriteProvenance>,
}

impl PlanNode {
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, PlanError> {
        let mut bytes = serde_json::to_vec(self).map_err(PlanError::serialization)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    #[must_use]
    pub fn snapshot(&self) -> PlanNodeSnapshot {
        PlanNodeSnapshot {
            id: self.id.clone(),
            invocation: self.invocation,
            uniform_words: self.uniform_words,
            bindings: self.bindings.clone(),
            outputs: self.outputs.clone(),
            diagnostic_label: self.diagnostic_label.clone(),
            timestamp_label: self.timestamp_label.clone(),
            source_semantic_ids: self.source_semantic_ids.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanRequirements {
    pub storage_binding_count: u32,
    pub uniform_binding_count: u32,
    pub required_storage_buffer_offset_alignment: u64,
    pub largest_storage_binding_bytes: u64,
    pub largest_buffer_bytes: u64,
    pub max_workgroup_size: [u32; 3],
    pub max_dispatch: [u32; 3],
    pub required_features: Vec<PlanFeature>,
}

impl PlanRequirements {
    pub fn derive(
        external_values: &[PlanExternalValue],
        nodes: &[PlanNode],
        required_storage_buffer_offset_alignment: u64,
        required_features: &[PlanFeature],
    ) -> Result<Self, PlanError> {
        verify_feature_order(required_features)?;

        let mut value_lengths = BTreeMap::new();
        let mut largest_buffer_bytes = 0_u64;
        for external in external_values {
            value_lengths.insert(external.id.as_str(), external.byte_length);
            largest_buffer_bytes = largest_buffer_bytes.max(checked_range_end(
                external.byte_offset,
                external.byte_length,
            )?);
        }
        for node in nodes {
            for output in &node.outputs {
                value_lengths.insert(output.id.as_str(), output.byte_length);
            }
        }

        let mut storage_binding_count = 0_u32;
        let mut uniform_binding_count = 0_u32;
        let mut largest_storage_binding_bytes = 0_u64;
        let mut max_workgroup_size = [0_u32; 3];
        let mut max_dispatch = [0_u32; 3];

        for node in nodes {
            let mut node_storage = 0_u32;
            let mut node_uniform = 0_u32;
            for binding in &node.bindings {
                let binding_bytes = match &binding.resource {
                    PlanBindingResource::Value(value_id) => {
                        node_storage = node_storage.checked_add(1).ok_or_else(|| {
                            PlanError::new(
                                PlanErrorCode::ArithmeticOverflow,
                                "storage binding count overflow",
                            )
                        })?;
                        value_lengths.get(value_id.as_str()).copied().unwrap_or(0)
                    }
                    PlanBindingResource::Tensor(tensor) => {
                        node_storage = node_storage.checked_add(1).ok_or_else(|| {
                            PlanError::new(
                                PlanErrorCode::ArithmeticOverflow,
                                "storage binding count overflow",
                            )
                        })?;
                        largest_buffer_bytes = largest_buffer_bytes
                            .max(checked_range_end(tensor.byte_offset, tensor.byte_length)?);
                        tensor.byte_length
                    }
                    PlanBindingResource::OutputBuffer(output) => {
                        node_storage = node_storage.checked_add(1).ok_or_else(|| {
                            PlanError::new(
                                PlanErrorCode::ArithmeticOverflow,
                                "storage binding count overflow",
                            )
                        })?;
                        largest_buffer_bytes = largest_buffer_bytes.max(output.byte_length);
                        output.byte_length
                    }
                    PlanBindingResource::UniformWords(_) => {
                        node_uniform = node_uniform.checked_add(1).ok_or_else(|| {
                            PlanError::new(
                                PlanErrorCode::ArithmeticOverflow,
                                "uniform binding count overflow",
                            )
                        })?;
                        0
                    }
                };
                largest_storage_binding_bytes = largest_storage_binding_bytes.max(binding_bytes);
            }
            storage_binding_count = storage_binding_count.max(node_storage);
            uniform_binding_count = uniform_binding_count.max(node_uniform);
            for dimension in 0..3 {
                max_workgroup_size[dimension] =
                    max_workgroup_size[dimension].max(node.invocation.workgroup_size[dimension]);
                max_dispatch[dimension] =
                    max_dispatch[dimension].max(node.invocation.dispatch[dimension]);
            }
        }

        Ok(Self {
            storage_binding_count,
            uniform_binding_count,
            required_storage_buffer_offset_alignment,
            largest_storage_binding_bytes,
            largest_buffer_bytes,
            max_workgroup_size,
            max_dispatch,
            required_features: required_features.to_vec(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanIr {
    pub schema_version: u32,
    pub external_values: Vec<PlanExternalValue>,
    pub nodes: Vec<PlanNode>,
    pub outputs: Vec<PlanValueId>,
    pub requirements: PlanRequirements,
}

impl PlanIr {
    pub fn verify(&self) -> Result<(), PlanError> {
        if self.schema_version != PLAN_SCHEMA_VERSION {
            return Err(PlanError::new(
                PlanErrorCode::UnsupportedSchemaVersion,
                "unsupported PlanIR schema version",
            ));
        }

        let mut node_ids = BTreeSet::new();
        for node in &self.nodes {
            if !node_ids.insert(node.id.as_str()) {
                return Err(PlanError::new(
                    PlanErrorCode::DuplicateNodeId,
                    format!("duplicate node {}", node.id),
                ));
            }
        }

        let mut producers = BTreeMap::new();
        for external in &self.external_values {
            let elements = verify_shape(&external.shape)?;
            checked_range_end(external.byte_offset, external.byte_length)?;
            verify_logical_value_byte_length(external.dtype, elements, external.byte_length)?;
            if producers
                .insert(
                    external.id.as_str(),
                    Producer {
                        node_index: None,
                        buffer_id: external.buffer_id.as_str(),
                    },
                )
                .is_some()
            {
                return Err(PlanError::new(
                    PlanErrorCode::DuplicateValueProducer,
                    format!("duplicate value producer {}", external.id),
                ));
            }
        }
        for (index, node) in self.nodes.iter().enumerate() {
            for output in &node.outputs {
                if producers
                    .insert(
                        output.id.as_str(),
                        Producer {
                            node_index: Some(index),
                            buffer_id: output.buffer_id.as_str(),
                        },
                    )
                    .is_some()
                {
                    return Err(PlanError::new(
                        PlanErrorCode::DuplicateValueProducer,
                        format!("duplicate value producer {}", output.id),
                    ));
                }
            }
        }

        for (index, node) in self.nodes.iter().enumerate() {
            verify_node(node, index, &producers)?;
        }

        verify_cross_node_storage_ranges(&self.nodes)?;

        let mut plan_outputs = BTreeSet::new();
        for output in &self.outputs {
            if !plan_outputs.insert(output.as_str()) {
                return Err(PlanError::new(
                    PlanErrorCode::DuplicatePlanOutput,
                    format!("plan output {output} is exported more than once"),
                ));
            }
            if !producers.contains_key(output.as_str()) {
                return Err(PlanError::new(
                    PlanErrorCode::OutputWithoutProducer,
                    format!("plan output {output} has no producer"),
                ));
            }
        }

        let derived = PlanRequirements::derive(
            &self.external_values,
            &self.nodes,
            self.requirements.required_storage_buffer_offset_alignment,
            &self.requirements.required_features,
        )?;
        if derived != self.requirements {
            return Err(PlanError::new(
                PlanErrorCode::RequirementsMismatch,
                "stored requirements do not match selected resources",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, PlanError> {
        self.verify()?;
        let mut bytes = serde_json::to_vec(self).map_err(PlanError::serialization)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    pub fn canonical_blake3_hex(&self) -> Result<String, PlanError> {
        Ok(blake3::hash(&self.canonical_bytes()?).to_hex().to_string())
    }

    pub fn parse_canonical(bytes: &[u8]) -> Result<Self, PlanError> {
        let plan: Self = serde_json::from_slice(bytes).map_err(|error| {
            let message = error.to_string();
            if message.contains("unknown field") {
                PlanError::new(PlanErrorCode::UnknownField, message)
            } else {
                PlanError::new(PlanErrorCode::NonCanonicalEncoding, message)
            }
        })?;
        let canonical = plan.canonical_bytes()?;
        if canonical != bytes {
            return Err(PlanError::new(
                PlanErrorCode::NonCanonicalEncoding,
                "input bytes are not the canonical PlanIR encoding",
            ));
        }
        Ok(plan)
    }
}

fn verify_node(
    node: &PlanNode,
    index: usize,
    producers: &BTreeMap<&str, Producer<'_>>,
) -> Result<(), PlanError> {
    verify_sources(&node.source_semantic_ids)?;

    let mut binding_numbers = BTreeSet::new();
    let mut previous_number = None;
    let mut input_buffers = BTreeSet::new();
    let mut parameter_buffers = BTreeSet::new();
    let mut output_buffers: BTreeMap<&str, u64> = BTreeMap::new();
    let mut uniform_count = 0_usize;

    for binding in &node.bindings {
        if !binding_numbers.insert(binding.number) {
            return Err(PlanError::new(
                PlanErrorCode::DuplicateBindingNumber,
                format!("node {} duplicates binding {}", node.id, binding.number),
            ));
        }
        if previous_number.is_some_and(|previous| binding.number <= previous) {
            return Err(PlanError::new(
                PlanErrorCode::NonCanonicalOrder,
                format!("node {} bindings are not ordered", node.id),
            ));
        }
        previous_number = Some(binding.number);

        let access_matches = matches!(
            (&binding.access, &binding.resource),
            (
                PlanBindingAccess::ReadOnlyStorage,
                PlanBindingResource::Value(_) | PlanBindingResource::Tensor(_)
            ) | (
                PlanBindingAccess::ReadWriteStorage,
                PlanBindingResource::OutputBuffer(_)
            ) | (
                PlanBindingAccess::Uniform,
                PlanBindingResource::UniformWords(_)
            )
        );
        if !access_matches {
            return Err(PlanError::new(
                PlanErrorCode::BindingResourceMismatch,
                format!("node {} binding access does not match resource", node.id),
            ));
        }

        match &binding.resource {
            PlanBindingResource::Value(value_id) => {
                let Some(producer) = producers.get(value_id.as_str()) else {
                    return Err(PlanError::new(
                        PlanErrorCode::DanglingInputValue,
                        format!("node {} consumes missing value {value_id}", node.id),
                    ));
                };
                if producer
                    .node_index
                    .is_some_and(|producer| producer >= index)
                {
                    return Err(PlanError::new(
                        PlanErrorCode::InvalidTopology,
                        format!("node {} consumes a self or later value", node.id),
                    ));
                }
                // Buffer aliases are checked by the plan-level helper below.
            }
            PlanBindingResource::Tensor(tensor) => {
                verify_shape(&tensor.shape)?;
                checked_range_end(tensor.byte_offset, tensor.byte_length)?;
                parameter_buffers.insert(tensor.buffer_id.as_str());
            }
            PlanBindingResource::OutputBuffer(output) => {
                if output_buffers
                    .insert(output.buffer_id.as_str(), output.byte_length)
                    .is_some()
                {
                    return Err(PlanError::new(
                        PlanErrorCode::InvocationResourceMismatch,
                        format!("node {} binds an output buffer more than once", node.id),
                    ));
                }
            }
            PlanBindingResource::UniformWords(uniform) => {
                uniform_count += 1;
                if uniform.words != node.uniform_words {
                    return Err(PlanError::new(
                        PlanErrorCode::InvocationResourceMismatch,
                        format!(
                            "node {} uniform resource differs from uniform words",
                            node.id
                        ),
                    ));
                }
            }
        }
    }

    // Resolve input buffer identities only after structural value checks.
    for binding in &node.bindings {
        if let PlanBindingResource::Value(value_id) = &binding.resource
            && let Some(producer) = producers.get(value_id.as_str())
        {
            input_buffers.insert(producer.buffer_id);
        }
    }

    if node.invocation.workgroup_size.contains(&0)
        || node.invocation.dispatch.contains(&0)
        || uniform_count != 1
        || output_buffers.len() != 1
    {
        return Err(PlanError::new(
            PlanErrorCode::InvocationResourceMismatch,
            format!("node {} invocation resources are incomplete", node.id),
        ));
    }
    let invocation_elements = u64::try_from(node.invocation.output_elements).map_err(|_| {
        PlanError::new(
            PlanErrorCode::ArithmeticOverflow,
            "invocation element count does not fit u64",
        )
    })?;
    let invocation_bytes = invocation_elements.checked_mul(4).ok_or_else(|| {
        PlanError::new(
            PlanErrorCode::ArithmeticOverflow,
            "invocation output byte count overflow",
        )
    })?;
    let (&output_buffer, &output_buffer_bytes) = output_buffers.iter().next().expect("one output");
    if invocation_bytes != node.invocation.output_bytes
        || output_buffer_bytes != node.invocation.output_bytes
    {
        return Err(PlanError::new(
            PlanErrorCode::InvocationResourceMismatch,
            format!(
                "node {} invocation output metadata is inconsistent",
                node.id
            ),
        ));
    }

    let mut ranges = Vec::new();
    let mut logical_outputs = Vec::new();
    for output in &node.outputs {
        let elements = verify_shape(&output.shape)?;
        if output.dtype != PlanDtype::Float32 {
            return Err(PlanError::new(
                PlanErrorCode::InvocationOutputDtypeMismatch,
                format!("node {} invocation output is not F32", node.id),
            ));
        }
        let end = checked_range_end(output.byte_offset, output.byte_length)?;
        let Some(&buffer_bytes) = output_buffers.get(output.buffer_id.as_str()) else {
            return Err(PlanError::new(
                PlanErrorCode::InvocationResourceMismatch,
                format!(
                    "node {} output is not backed by its output binding",
                    node.id
                ),
            ));
        };
        if end > buffer_bytes {
            return Err(PlanError::new(
                PlanErrorCode::SliceOutOfBounds,
                format!("node {} output slice exceeds its buffer", node.id),
            ));
        }
        ranges.push((output.buffer_id.as_str(), output.byte_offset, end));
        logical_outputs.push((output.dtype, elements, output.byte_length));
    }
    for left in 0..ranges.len() {
        for right in (left + 1)..ranges.len() {
            let (left_buffer, left_start, left_end) = ranges[left];
            let (right_buffer, right_start, right_end) = ranges[right];
            if left_buffer == right_buffer && left_start < right_end && right_start < left_end {
                return Err(PlanError::new(
                    PlanErrorCode::OverlappingOutputSlices,
                    format!("node {} has overlapping output slices", node.id),
                ));
            }
        }
    }
    for (dtype, elements, byte_length) in logical_outputs {
        verify_logical_value_byte_length(dtype, elements, byte_length)?;
    }

    if input_buffers.contains(output_buffer)
        || parameter_buffers.contains(output_buffer)
        || input_buffers
            .iter()
            .any(|buffer| parameter_buffers.contains(buffer))
    {
        return Err(PlanError::new(
            PlanErrorCode::IllegalBufferAlias,
            format!(
                "node {} aliases input, parameter, or output storage",
                node.id
            ),
        ));
    }

    if let Some(provenance) = &node.rewrite_provenance {
        verify_rewrite_provenance(node, provenance)?;
    }
    Ok(())
}

fn verify_rewrite_provenance(
    node: &PlanNode,
    provenance: &PlanRewriteProvenance,
) -> Result<(), PlanError> {
    if provenance.pass_id.is_empty()
        || provenance.source_semantic_ids != node.source_semantic_ids
        || provenance.consumed.is_empty()
    {
        return Err(PlanError::new(
            PlanErrorCode::InvalidRewriteProvenance,
            format!("node {} has incomplete rewrite provenance", node.id),
        ));
    }
    verify_sources(&provenance.source_semantic_ids).map_err(|_| {
        PlanError::new(
            PlanErrorCode::InvalidRewriteProvenance,
            format!("node {} has invalid rewrite source provenance", node.id),
        )
    })?;
    let mut roles = BTreeSet::new();
    for consumed in &provenance.consumed {
        if consumed.role.is_empty() || !roles.insert(consumed.role.as_str()) {
            return Err(PlanError::new(
                PlanErrorCode::InvalidRewriteProvenance,
                format!("node {} has invalid consumed roles", node.id),
            ));
        }
        verify_sources(&consumed.original.source_semantic_ids).map_err(|_| {
            PlanError::new(
                PlanErrorCode::InvalidRewriteProvenance,
                format!("node {} has invalid consumed source provenance", node.id),
            )
        })?;
        let expected = consumed.original.canonical_node_blake3_hex()?;
        if consumed.canonical_blake3 != expected {
            return Err(PlanError::new(
                PlanErrorCode::InvalidRewriteProvenance,
                format!("node {} has a corrupt consumed-node hash", node.id),
            ));
        }
    }
    Ok(())
}

fn verify_sources(sources: &[SemanticId]) -> Result<(), PlanError> {
    if sources.is_empty() {
        return Err(PlanError::new(
            PlanErrorCode::EmptySourceProvenance,
            "source semantic provenance is empty",
        ));
    }
    let mut seen = BTreeSet::new();
    for source in sources {
        if !seen.insert(source.as_str()) {
            return Err(PlanError::new(
                PlanErrorCode::DuplicateSourceProvenance,
                format!("duplicate source semantic ID {source}"),
            ));
        }
    }
    Ok(())
}

fn verify_logical_value_byte_length(
    dtype: PlanDtype,
    elements: u64,
    byte_length: u64,
) -> Result<(), PlanError> {
    if dtype != PlanDtype::Float32 {
        return Ok(());
    }
    let expected = elements
        .checked_mul(PlanDtype::Float32.byte_width())
        .ok_or_else(|| {
            PlanError::new(
                PlanErrorCode::ArithmeticOverflow,
                "F32 logical value byte length overflow",
            )
        })?;
    if byte_length != expected {
        return Err(PlanError::new(
            PlanErrorCode::ValueByteLengthMismatch,
            format!("F32 logical value requires {expected} bytes, found {byte_length}"),
        ));
    }
    Ok(())
}

fn verify_cross_node_storage_ranges(nodes: &[PlanNode]) -> Result<(), PlanError> {
    let mut parameters = Vec::new();
    let mut outputs = Vec::new();
    for (node_index, node) in nodes.iter().enumerate() {
        for binding in &node.bindings {
            if let PlanBindingResource::Tensor(tensor) = &binding.resource {
                parameters.push((
                    node_index,
                    tensor.buffer_id.as_str(),
                    tensor.byte_offset,
                    checked_range_end(tensor.byte_offset, tensor.byte_length)?,
                ));
            }
        }
        for output in &node.outputs {
            outputs.push((
                node_index,
                output.buffer_id.as_str(),
                output.byte_offset,
                checked_range_end(output.byte_offset, output.byte_length)?,
            ));
        }
    }

    for left in 0..parameters.len() {
        for right in (left + 1)..parameters.len() {
            let (left_node, left_buffer, left_start, left_end) = parameters[left];
            let (right_node, right_buffer, right_start, right_end) = parameters[right];
            if left_node != right_node
                && left_buffer == right_buffer
                && left_start < right_end
                && right_start < left_end
            {
                return Err(PlanError::new(
                    PlanErrorCode::OverlappingParameterSlices,
                    "parameter slices overlap across PlanIR nodes",
                ));
            }
        }
    }

    for left in 0..outputs.len() {
        for right in (left + 1)..outputs.len() {
            let (left_node, left_buffer, left_start, left_end) = outputs[left];
            let (right_node, right_buffer, right_start, right_end) = outputs[right];
            if left_node != right_node
                && left_buffer == right_buffer
                && left_start < right_end
                && right_start < left_end
            {
                return Err(PlanError::new(
                    PlanErrorCode::OverlappingOutputSlices,
                    "output slices overlap across PlanIR nodes",
                ));
            }
        }
    }
    Ok(())
}

fn verify_shape(shape: &[u64]) -> Result<u64, PlanError> {
    let mut elements = 1_u64;
    for &dimension in shape {
        if dimension == 0 {
            return Err(PlanError::new(
                PlanErrorCode::ZeroShapeDimension,
                "shape contains a zero dimension",
            ));
        }
        elements = elements.checked_mul(dimension).ok_or_else(|| {
            PlanError::new(
                PlanErrorCode::ArithmeticOverflow,
                "shape element count overflow",
            )
        })?;
    }
    Ok(elements)
}

fn checked_range_end(offset: u64, length: u64) -> Result<u64, PlanError> {
    offset.checked_add(length).ok_or_else(|| {
        PlanError::new(
            PlanErrorCode::ByteRangeOverflow,
            "byte offset plus length overflow",
        )
    })
}

fn verify_feature_order(features: &[PlanFeature]) -> Result<(), PlanError> {
    let mut previous = None;
    for &feature in features {
        if previous == Some(feature) {
            return Err(PlanError::new(
                PlanErrorCode::DuplicateRequiredFeature,
                format!("duplicate required feature {}", feature.as_str()),
            ));
        }
        if previous.is_some_and(|item| feature < item) {
            return Err(PlanError::new(
                PlanErrorCode::NonCanonicalOrder,
                "required features are not canonically ordered",
            ));
        }
        previous = Some(feature);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanErrorCode {
    UnsupportedSchemaVersion,
    DuplicateNodeId,
    DuplicateValueProducer,
    DuplicateBindingNumber,
    BindingResourceMismatch,
    DanglingInputValue,
    OutputWithoutProducer,
    InvalidTopology,
    EmptySourceProvenance,
    DuplicateSourceProvenance,
    NonCanonicalOrder,
    ZeroShapeDimension,
    ArithmeticOverflow,
    ByteRangeOverflow,
    SliceOutOfBounds,
    OverlappingOutputSlices,
    InvocationResourceMismatch,
    IllegalBufferAlias,
    RequirementsMismatch,
    DuplicateRequiredFeature,
    ValueByteLengthMismatch,
    InvocationOutputDtypeMismatch,
    DuplicatePlanOutput,
    OverlappingParameterSlices,
    InvalidRewriteProvenance,
    UnknownField,
    NonCanonicalEncoding,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanError {
    code: PlanErrorCode,
    message: String,
}

impl PlanError {
    #[must_use]
    pub const fn code(&self) -> PlanErrorCode {
        self.code
    }

    fn new(code: PlanErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    fn serialization(error: serde_json::Error) -> Self {
        Self::new(PlanErrorCode::NonCanonicalEncoding, error.to_string())
    }
}

impl fmt::Display for PlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "PlanIR error {:?}: {}", self.code, self.message)
    }
}

impl Error for PlanError {}
