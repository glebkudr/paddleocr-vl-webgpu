use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use pvlc_ir::{
    PlanBinding, PlanBindingAccess, PlanBindingResource, PlanBufferId, PlanConsumedNode, PlanDtype,
    PlanErrorCode, PlanExternalValue, PlanIr, PlanNode, PlanNodeId, PlanNodeSnapshot, PlanOutput,
    PlanOutputBuffer, PlanRequirements, PlanRewriteProvenance, PlanTensorResource,
    PlanUniformResource, PlanValueId, SemanticGraph, SemanticId, SemanticOp,
};
use pvlc_model_schema::{TensorDtype, TensorSpec};
use pvlc_runtime_core::{
    InvocationPlan, KernelId, LINEAR_PROJECTION_TILE, VisionEncoderLayerDispatch,
    VisionEncoderLayerPlan, VisionEncoderLayerStage, VisionQkvFusedPlan,
    VisionQkvFusedTargetLimits, plan_vision_qkv_fused_geometry,
};

const PASS_ID: &str = "vision-qkv-fusion-v1";
const VISION_LAYER_COUNT: usize = 27;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionQkvFusionOptions {
    pub enabled: bool,
    pub target: VisionQkvFusedTargetLimits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionQkvPassStatus {
    UnchangedDisabled,
    UnchangedNoMatch,
    UnchangedAlreadyFused,
    Fused,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvPassResult {
    pub status: VisionQkvPassStatus,
    pub plan: PlanIr,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionQkvPassErrorCode {
    InvalidPlan,
    InvalidProvenance,
    InvalidLayer,
    InvalidGeometry,
    SemanticMismatch,
    MissingTensor,
    DuplicateTensor,
    TensorIdentityMismatch,
    MalformedCandidate,
    IncompleteCandidate,
    DuplicateCandidateRole,
    NonCanonicalCandidateOrder,
    NonContiguousCandidate,
    AmbiguousCandidate,
    MixedCandidate,
    LegacyAbiMismatch,
    TensorBindingMismatch,
    IllegalCandidateDataflow,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvPassError {
    code: VisionQkvPassErrorCode,
    message: String,
    fused_mismatch: Option<VisionQkvFusedMismatchKind>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VisionQkvFusedMismatchKind {
    Structural,
    SemanticOrTensor,
}

impl VisionQkvPassError {
    #[must_use]
    pub const fn code(&self) -> VisionQkvPassErrorCode {
        self.code
    }

    fn new(code: VisionQkvPassErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            fused_mismatch: None,
        }
    }

    fn fused_mismatch(kind: VisionQkvFusedMismatchKind, message: impl Into<String>) -> Self {
        Self {
            code: VisionQkvPassErrorCode::InvalidProvenance,
            message: message.into(),
            fused_mismatch: Some(kind),
        }
    }

    pub(crate) const fn fused_mismatch_kind(&self) -> Option<VisionQkvFusedMismatchKind> {
        self.fused_mismatch
    }
}

impl fmt::Display for VisionQkvPassError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "vision Q/K/V pass error {:?}: {}",
            self.code, self.message
        )
    }
}

impl Error for VisionQkvPassError {}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Role {
    Query,
    Key,
    Value,
}

impl Role {
    pub(crate) const ALL: [Self; 3] = [Self::Query, Self::Key, Self::Value];

    const fn name(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Key => "key",
            Self::Value => "value",
        }
    }

    pub(crate) const fn letter(self) -> &'static str {
        match self {
            Self::Query => "q",
            Self::Key => "k",
            Self::Value => "v",
        }
    }

    pub(crate) const fn projection(self) -> &'static str {
        match self {
            Self::Query => "q_proj",
            Self::Key => "k_proj",
            Self::Value => "v_proj",
        }
    }

    const fn stage(self) -> VisionEncoderLayerStage {
        match self {
            Self::Query => VisionEncoderLayerStage::Query,
            Self::Key => VisionEncoderLayerStage::Key,
            Self::Value => VisionEncoderLayerStage::Value,
        }
    }
}

#[derive(Clone, Copy)]
struct LegacyCandidate {
    layer: usize,
    indices: [usize; 3],
}

pub fn lower_vision_qkv_fragment(
    graph: &SemanticGraph,
    layer: usize,
    geometry: &VisionEncoderLayerPlan,
    catalog: &[TensorSpec],
) -> Result<PlanIr, VisionQkvPassError> {
    if layer >= VISION_LAYER_COUNT {
        return Err(VisionQkvPassError::new(
            VisionQkvPassErrorCode::InvalidLayer,
            format!("vision layer {layer} is outside 00..26"),
        ));
    }
    verify_semantic_source(graph, layer)?;
    let legacy_dispatches = verify_geometry(geometry)?;
    let uniform_words = legacy_dispatches[0].uniform_words;
    let [tokens, input_width, output_width, _] = uniform_words;
    let tensors = resolve_tensors(catalog, layer, input_width, output_width)?;

    let input_bytes = checked_f32_bytes(tokens, input_width)?;
    let activation = activation_value(layer);
    let external_values = vec![PlanExternalValue {
        id: value_id(&activation),
        dtype: PlanDtype::Float32,
        shape: vec![u64::from(tokens), u64::from(input_width)],
        buffer_id: buffer_id(&format!("activation.{activation}")),
        byte_offset: 0,
        byte_length: input_bytes,
    }];

    let mut nodes = Vec::with_capacity(3);
    for (role_index, role) in Role::ALL.into_iter().enumerate() {
        nodes.push(lower_projection_node(
            layer,
            role,
            legacy_dispatches[role_index],
            tensors[role_index * 2],
            tensors[role_index * 2 + 1],
        )?);
    }
    let outputs = Role::ALL
        .into_iter()
        .map(|role| value_id(&output_value(layer, role)))
        .collect::<Vec<_>>();
    let requirements = PlanRequirements::derive(&external_values, &nodes, 4, &[])
        .map_err(|error| invalid_geometry(error.to_string()))?;
    let plan = PlanIr {
        schema_version: 1,
        external_values,
        nodes,
        outputs,
        requirements,
    };
    plan.verify()
        .map_err(|error| invalid_geometry(error.to_string()))?;
    Ok(plan)
}

pub fn fuse_vision_qkv(
    plan: &PlanIr,
    graph: &SemanticGraph,
    options: VisionQkvFusionOptions,
) -> Result<VisionQkvPassResult, VisionQkvPassError> {
    let deferred_requirements_error = match plan.verify() {
        Ok(()) => None,
        Err(error)
            if matches!(
                error.code(),
                PlanErrorCode::RequirementsMismatch | PlanErrorCode::OverlappingParameterSlices
            ) =>
        {
            Some(error)
        }
        Err(error) => {
            let code = if error.code() == PlanErrorCode::InvalidRewriteProvenance {
                VisionQkvPassErrorCode::InvalidProvenance
            } else {
                VisionQkvPassErrorCode::InvalidPlan
            };
            return Err(VisionQkvPassError::new(code, error.to_string()));
        }
    };
    graph.verify().map_err(|error| {
        VisionQkvPassError::new(VisionQkvPassErrorCode::SemanticMismatch, error.to_string())
    })?;

    if !options.enabled {
        if let Some(error) = deferred_requirements_error {
            return Err(VisionQkvPassError::new(
                VisionQkvPassErrorCode::InvalidPlan,
                error.to_string(),
            ));
        }
        return Ok(VisionQkvPassResult {
            status: VisionQkvPassStatus::UnchangedDisabled,
            plan: plan.clone(),
        });
    }

    let fused_indices = plan
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            (node.invocation.kernel == KernelId::VisionQkvFusedF32).then_some(index)
        })
        .collect::<Vec<_>>();
    if plan.nodes.iter().any(|node| {
        node.invocation.kernel != KernelId::VisionQkvFusedF32
            && has_qkv_rewrite_provenance_evidence(node)
    }) {
        return Err(invalid_provenance(
            "rewrite provenance is attached to a non-fused Q/K/V node",
        ));
    }
    let legacy_evidence = collect_legacy_evidence(plan)?;

    if !fused_indices.is_empty() && !legacy_evidence.is_empty() {
        return Err(VisionQkvPassError::new(
            VisionQkvPassErrorCode::MixedCandidate,
            "plan contains both fused and legacy Q/K/V candidate evidence",
        ));
    }
    if !fused_indices.is_empty() {
        if let Some(error) = deferred_requirements_error {
            return Err(VisionQkvPassError::new(
                VisionQkvPassErrorCode::InvalidPlan,
                error.to_string(),
            ));
        }
        if fused_indices.len() != 1 {
            return Err(VisionQkvPassError::new(
                VisionQkvPassErrorCode::AmbiguousCandidate,
                "plan contains multiple fused Q/K/V candidates",
            ));
        }
        verify_existing_fused(plan, graph, fused_indices[0], options.target)?;
        return Ok(VisionQkvPassResult {
            status: VisionQkvPassStatus::UnchangedAlreadyFused,
            plan: plan.clone(),
        });
    }

    if legacy_evidence.is_empty() {
        if let Some(error) = deferred_requirements_error {
            return Err(VisionQkvPassError::new(
                VisionQkvPassErrorCode::InvalidPlan,
                error.to_string(),
            ));
        }
        return Ok(VisionQkvPassResult {
            status: VisionQkvPassStatus::UnchangedNoMatch,
            plan: plan.clone(),
        });
    }

    let candidate = classify_legacy_candidate(plan, &legacy_evidence)?;
    verify_semantic_source(graph, candidate.layer)?;
    let nodes = candidate_nodes(plan, candidate);
    verify_legacy_triple(plan, candidate, nodes)?;
    verify_live_qkv_dataflow(plan, candidate.layer, &candidate.indices)?;
    if let Some(error) = deferred_requirements_error {
        return Err(VisionQkvPassError::new(
            VisionQkvPassErrorCode::InvalidPlan,
            error.to_string(),
        ));
    }

    let [tokens, input_width, output_width, _] = nodes[0].uniform_words;
    let fused_geometry =
        plan_vision_qkv_fused_geometry(tokens, input_width, output_width, options.target)
            .map_err(|error| invalid_geometry(error.to_string()))?;
    let fused_node = construct_fused_node(candidate.layer, nodes, fused_geometry)?;

    let mut rewritten_nodes = Vec::with_capacity(plan.nodes.len() - 2);
    for (index, node) in plan.nodes.iter().enumerate() {
        if index == candidate.indices[0] {
            rewritten_nodes.push(fused_node.clone());
        } else if !candidate.indices.contains(&index) {
            rewritten_nodes.push(node.clone());
        }
    }
    let requirements = PlanRequirements::derive(
        &plan.external_values,
        &rewritten_nodes,
        u64::from(options.target.min_storage_buffer_offset_alignment),
        &plan.requirements.required_features,
    )
    .map_err(|error| invalid_geometry(error.to_string()))?;
    let rewritten = PlanIr {
        schema_version: plan.schema_version,
        external_values: plan.external_values.clone(),
        nodes: rewritten_nodes,
        outputs: plan.outputs.clone(),
        requirements,
    };
    rewritten.verify().map_err(|error| {
        VisionQkvPassError::new(VisionQkvPassErrorCode::InvalidPlan, error.to_string())
    })?;
    Ok(VisionQkvPassResult {
        status: VisionQkvPassStatus::Fused,
        plan: rewritten,
    })
}

fn has_qkv_rewrite_provenance_evidence(node: &PlanNode) -> bool {
    let Some(provenance) = &node.rewrite_provenance else {
        return false;
    };
    provenance.pass_id == PASS_ID
        || provenance
            .source_semantic_ids
            .iter()
            .any(|source| parse_semantic_source(source.as_str()).is_some())
        || provenance
            .consumed
            .iter()
            .any(|consumed| snapshot_has_qkv_evidence(&consumed.original))
}

fn snapshot_has_qkv_evidence(snapshot: &PlanNodeSnapshot) -> bool {
    parse_projection_id(snapshot.id.as_str()).is_some()
        || parse_fused_id(snapshot.id.as_str()).is_some()
        || snapshot.invocation.kernel == KernelId::VisionQkvFusedF32
        || snapshot
            .source_semantic_ids
            .iter()
            .any(|source| parse_semantic_source(source.as_str()).is_some())
        || snapshot
            .outputs
            .iter()
            .any(|output| parse_projection_id(output.id.as_str()).is_some())
        || snapshot.bindings.iter().any(|binding| {
            matches!(
                &binding.resource,
                PlanBindingResource::Tensor(tensor)
                    if parse_tensor_semantic(tensor.semantic_id.as_str()).is_some()
            )
        })
}

fn verify_geometry(
    geometry: &VisionEncoderLayerPlan,
) -> Result<[&VisionEncoderLayerDispatch; 3], VisionQkvPassError> {
    let mut selected = Vec::with_capacity(3);
    for role in Role::ALL {
        let matches = geometry
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.stage == role.stage())
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(invalid_geometry(format!(
                "geometry must contain one {} dispatch",
                role.name()
            )));
        }
        selected.push(matches[0]);
    }
    let selected: [&VisionEncoderLayerDispatch; 3] = selected
        .try_into()
        .expect("three projection roles were selected");
    let [tokens, input_width, output_width, reserved] = selected[0].uniform_words;
    if reserved != 0 || tokens == 0 || input_width == 0 || output_width == 0 {
        return Err(invalid_geometry("invalid projection uniform dimensions"));
    }
    let output_elements_u64 = u64::from(tokens)
        .checked_mul(u64::from(output_width))
        .ok_or_else(|| invalid_geometry("projection element count overflow"))?;
    let output_elements = usize::try_from(output_elements_u64)
        .map_err(|_| invalid_geometry("projection element count does not fit usize"))?;
    let output_bytes = output_elements_u64
        .checked_mul(4)
        .ok_or_else(|| invalid_geometry("projection byte count overflow"))?;
    let expected_invocation = InvocationPlan {
        kernel: KernelId::VisionPatchProjectionF32,
        output_elements,
        output_bytes,
        workgroup_size: [8, 8, 1],
        dispatch: [
            output_width.div_ceil(LINEAR_PROJECTION_TILE),
            tokens.div_ceil(LINEAR_PROJECTION_TILE),
            1,
        ],
    };
    for dispatch in selected {
        if dispatch.uniform_words != [tokens, input_width, output_width, 0]
            || dispatch.invocation != expected_invocation
        {
            return Err(invalid_geometry(
                "Q/K/V projection dispatches do not match the legacy ABI",
            ));
        }
    }
    Ok(selected)
}

fn resolve_tensors(
    catalog: &[TensorSpec],
    layer: usize,
    input_width: u32,
    output_width: u32,
) -> Result<[&TensorSpec; 6], VisionQkvPassError> {
    let mut resolved = Vec::with_capacity(6);
    for role in Role::ALL {
        for suffix in ["weight", "bias"] {
            let semantic = tensor_semantic(layer, role, suffix);
            let matches = catalog
                .iter()
                .filter(|tensor| tensor.semantic_id == semantic)
                .collect::<Vec<_>>();
            for tensor in &matches {
                let expected_shape = if suffix == "weight" {
                    vec![u64::from(output_width), u64::from(input_width)]
                } else {
                    vec![u64::from(output_width)]
                };
                if tensor.name != tensor_physical(layer, role, suffix)
                    || tensor.dtype != TensorDtype::BFloat16
                    || tensor.shape != expected_shape
                {
                    return Err(VisionQkvPassError::new(
                        VisionQkvPassErrorCode::TensorIdentityMismatch,
                        format!("tensor identity for {semantic} does not match the checkpoint"),
                    ));
                }
            }
            match matches.len() {
                0 => {
                    return Err(VisionQkvPassError::new(
                        VisionQkvPassErrorCode::MissingTensor,
                        format!("missing tensor {semantic}"),
                    ));
                }
                1 => resolved.push(matches[0]),
                _ => {
                    return Err(VisionQkvPassError::new(
                        VisionQkvPassErrorCode::DuplicateTensor,
                        format!("duplicate tensor {semantic}"),
                    ));
                }
            }
        }
    }
    Ok(resolved
        .try_into()
        .expect("all six projection tensors were resolved"))
}

fn lower_projection_node(
    layer: usize,
    role: Role,
    dispatch: &VisionEncoderLayerDispatch,
    weight: &TensorSpec,
    bias: &TensorSpec,
) -> Result<PlanNode, VisionQkvPassError> {
    let [tokens, _input_width, output_width, _] = dispatch.uniform_words;
    let output_bytes = checked_f32_bytes(tokens, output_width)?;
    let output_buffer_id = buffer_id(&output_buffer(layer, role));
    let uniform_words = dispatch.uniform_words;
    Ok(PlanNode {
        id: node_id(&projection_node_id(layer, role)),
        invocation: dispatch.invocation,
        uniform_words,
        bindings: vec![
            PlanBinding {
                number: 0,
                access: PlanBindingAccess::ReadOnlyStorage,
                resource: PlanBindingResource::Value(value_id(&activation_value(layer))),
            },
            PlanBinding {
                number: 1,
                access: PlanBindingAccess::ReadOnlyStorage,
                resource: PlanBindingResource::Tensor(tensor_resource(
                    layer, role, "weight", weight,
                )?),
            },
            PlanBinding {
                number: 2,
                access: PlanBindingAccess::ReadOnlyStorage,
                resource: PlanBindingResource::Tensor(tensor_resource(layer, role, "bias", bias)?),
            },
            PlanBinding {
                number: 3,
                access: PlanBindingAccess::ReadWriteStorage,
                resource: PlanBindingResource::OutputBuffer(PlanOutputBuffer {
                    buffer_id: output_buffer_id.clone(),
                    byte_length: output_bytes,
                }),
            },
            PlanBinding {
                number: 4,
                access: PlanBindingAccess::Uniform,
                resource: PlanBindingResource::UniformWords(PlanUniformResource {
                    words: uniform_words,
                }),
            },
        ],
        outputs: vec![PlanOutput {
            id: value_id(&output_value(layer, role)),
            dtype: PlanDtype::Float32,
            shape: vec![u64::from(tokens), u64::from(output_width)],
            buffer_id: output_buffer_id,
            byte_offset: 0,
            byte_length: output_bytes,
        }],
        diagnostic_label: projection_node_id(layer, role),
        timestamp_label: Some(projection_node_id(layer, role)),
        source_semantic_ids: vec![semantic_id(&semantic_source(layer))],
        rewrite_provenance: None,
    })
}

fn tensor_resource(
    layer: usize,
    role: Role,
    suffix: &str,
    tensor: &TensorSpec,
) -> Result<PlanTensorResource, VisionQkvPassError> {
    let elements = tensor.shape.iter().try_fold(1_u64, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or_else(|| invalid_geometry("tensor element count overflow"))
    })?;
    let byte_length = elements
        .checked_mul(4)
        .ok_or_else(|| invalid_geometry("tensor executable byte count overflow"))?;
    Ok(PlanTensorResource {
        physical_name: tensor.name.clone(),
        semantic_id: semantic_id(&tensor.semantic_id),
        dtype: PlanDtype::BFloat16,
        shape: tensor.shape.clone(),
        storage_format: PlanDtype::Float32,
        buffer_id: buffer_id(&tensor_buffer(layer, role, suffix)),
        byte_offset: 0,
        byte_length,
    })
}

#[derive(Clone, Copy)]
struct NodeEvidence {
    index: usize,
    layer: Option<usize>,
    role: Option<Role>,
    exact_id: bool,
    exact_source: bool,
    exact_tensor_role: bool,
}

fn collect_legacy_evidence(plan: &PlanIr) -> Result<Vec<NodeEvidence>, VisionQkvPassError> {
    let mut evidence = Vec::new();
    for (index, node) in plan.nodes.iter().enumerate() {
        if node.invocation.kernel == KernelId::VisionQkvFusedF32 {
            continue;
        }
        let id = parse_projection_id(node.id.as_str());
        let source_layers = node
            .source_semantic_ids
            .iter()
            .filter_map(|source| parse_semantic_source(source.as_str()))
            .collect::<BTreeSet<_>>();
        if source_layers.len() > 1 {
            return Err(malformed_candidate(
                "candidate node has conflicting canonical semantic sources",
            ));
        }
        let source_layer = source_layers.iter().next().copied();
        let tensor_signals = tensor_role_signals(node);
        if tensor_signals.len() > 1 && id.is_none() {
            return Err(malformed_candidate(
                "candidate node has conflicting Q/K/V tensor identities",
            ));
        }
        let tensor = if tensor_signals.len() == 1 {
            tensor_signals.iter().next().copied()
        } else {
            id
        };
        let output_signals = node
            .outputs
            .iter()
            .filter_map(|output| parse_projection_id(output.id.as_str()))
            .collect::<BTreeSet<_>>();
        if output_signals.len() > 1 {
            return Err(malformed_candidate(
                "candidate node has conflicting canonical Q/K/V output IDs",
            ));
        }
        let output = output_signals.iter().next().copied();
        let has_signal =
            id.is_some() || source_layer.is_some() || tensor.is_some() || output.is_some();
        if !has_signal {
            continue;
        }
        let layer = id
            .map(|(layer, _)| layer)
            .or_else(|| tensor.map(|(layer, _)| layer))
            .or_else(|| output.map(|(layer, _)| layer))
            .or(source_layer);
        let role = id
            .map(|(_, role)| role)
            .or_else(|| tensor.map(|(_, role)| role))
            .or_else(|| output.map(|(_, role)| role));
        evidence.push(NodeEvidence {
            index,
            layer,
            role,
            exact_id: id.is_some(),
            exact_source: source_layer.is_some_and(|source| Some(source) == layer),
            exact_tensor_role: tensor_signals.len() == 1
                && tensor.is_some_and(|item| Some(item) == layer.zip(role)),
        });
    }
    Ok(evidence)
}

fn classify_legacy_candidate(
    plan: &PlanIr,
    evidence: &[NodeEvidence],
) -> Result<LegacyCandidate, VisionQkvPassError> {
    let mut groups: BTreeMap<usize, Vec<NodeEvidence>> = BTreeMap::new();
    for item in evidence {
        let Some(layer) = item.layer else {
            return Err(malformed_candidate(
                "candidate evidence has no layer identity",
            ));
        };
        groups.entry(layer).or_default().push(*item);
    }
    if groups.len() > 1 {
        return Err(VisionQkvPassError::new(
            VisionQkvPassErrorCode::AmbiguousCandidate,
            "plan contains candidate evidence for multiple Q/K/V clusters",
        ));
    }
    let (&layer, items) = groups.iter().next().expect("evidence is non-empty");
    let mut role_indices: BTreeMap<Role, Vec<usize>> = BTreeMap::new();
    for item in items {
        let Some(role) = item.role else {
            return Err(malformed_candidate("candidate evidence has no Q/K/V role"));
        };
        role_indices.entry(role).or_default().push(item.index);
    }
    if role_indices.values().any(|indices| indices.len() > 1) {
        return Err(VisionQkvPassError::new(
            VisionQkvPassErrorCode::DuplicateCandidateRole,
            "Q/K/V candidate contains a duplicate projection role",
        ));
    }
    if role_indices.len() != 3 {
        let evidence_is_exact = items.iter().all(|item| {
            item.exact_id && item.exact_source && item.exact_tensor_role && item.role.is_some()
        });
        return Err(if evidence_is_exact {
            VisionQkvPassError::new(
                VisionQkvPassErrorCode::IncompleteCandidate,
                "Q/K/V candidate is missing a projection role",
            )
        } else {
            malformed_candidate("isolated Q/K/V signal does not form a candidate")
        });
    }
    let indices = Role::ALL.map(|role| role_indices[&role][0]);
    if !(indices[0] < indices[1] && indices[1] < indices[2]) {
        return Err(VisionQkvPassError::new(
            VisionQkvPassErrorCode::NonCanonicalCandidateOrder,
            "Q/K/V candidate is not ordered Query, Key, Value",
        ));
    }
    if indices[1] != indices[0] + 1 || indices[2] != indices[1] + 1 {
        return Err(VisionQkvPassError::new(
            VisionQkvPassErrorCode::NonContiguousCandidate,
            "Q/K/V candidate nodes are not contiguous",
        ));
    }
    if items.len() != 3 || indices.iter().any(|&index| index >= plan.nodes.len()) {
        return Err(malformed_candidate(
            "candidate contains unexplained evidence",
        ));
    }
    Ok(LegacyCandidate { layer, indices })
}

fn candidate_nodes(plan: &PlanIr, candidate: LegacyCandidate) -> [&PlanNode; 3] {
    candidate.indices.map(|index| &plan.nodes[index])
}

fn verify_legacy_triple(
    plan: &PlanIr,
    candidate: LegacyCandidate,
    nodes: [&PlanNode; 3],
) -> Result<(), VisionQkvPassError> {
    let source = semantic_id(&semantic_source(candidate.layer));
    let [tokens, input_width, output_width, reserved] = nodes[0].uniform_words;
    if tokens == 0 || input_width == 0 || output_width == 0 || reserved != 0 {
        return Err(legacy_abi("invalid legacy Q/K/V dimensions"));
    }
    let output_elements_u64 = u64::from(tokens)
        .checked_mul(u64::from(output_width))
        .ok_or_else(|| legacy_abi("legacy Q/K/V output element overflow"))?;
    let output_elements = usize::try_from(output_elements_u64)
        .map_err(|_| legacy_abi("legacy Q/K/V output elements do not fit usize"))?;
    let output_bytes = output_elements_u64
        .checked_mul(4)
        .ok_or_else(|| legacy_abi("legacy Q/K/V output byte overflow"))?;
    let expected_invocation = InvocationPlan {
        kernel: KernelId::VisionPatchProjectionF32,
        output_elements,
        output_bytes,
        workgroup_size: [8, 8, 1],
        dispatch: [
            output_width.div_ceil(LINEAR_PROJECTION_TILE),
            tokens.div_ceil(LINEAR_PROJECTION_TILE),
            1,
        ],
    };
    let activation = value_id(&activation_value(candidate.layer));

    for (node, role) in nodes.into_iter().zip(Role::ALL) {
        if node.id != node_id(&projection_node_id(candidate.layer, role))
            || node.source_semantic_ids != [source.clone()]
        {
            return Err(VisionQkvPassError::new(
                VisionQkvPassErrorCode::SemanticMismatch,
                "legacy node identity or source provenance differs from canonical SemanticIR",
            ));
        }
        if node.invocation != expected_invocation
            || node.uniform_words != [tokens, input_width, output_width, 0]
            || node.outputs.len() != 1
            || node.rewrite_provenance.is_some()
            || node.diagnostic_label != projection_node_id(candidate.layer, role)
            || node.timestamp_label.as_deref()
                != Some(projection_node_id(candidate.layer, role).as_str())
        {
            return Err(legacy_abi(format!(
                "{} projection does not match the legacy invocation ABI",
                role.name()
            )));
        }
        if node.bindings.len() < 3 {
            return Err(tensor_binding_mismatch(
                "legacy projection is missing weight or bias bindings",
            ));
        }
        if node.bindings.len() != 5 {
            verify_tensor_binding(
                &node.bindings[1],
                candidate.layer,
                role,
                "weight",
                &[u64::from(output_width), u64::from(input_width)],
            )?;
            verify_tensor_binding(
                &node.bindings[2],
                candidate.layer,
                role,
                "bias",
                &[u64::from(output_width)],
            )?;
            return Err(tensor_binding_mismatch(
                "legacy projection has an unexpected tensor binding cardinality",
            ));
        }
        if node.bindings.iter().enumerate().any(|(number, binding)| {
            binding.number != u32::try_from(number).expect("five bindings fit u32")
        }) {
            return Err(legacy_abi("legacy binding numbers are not 0..4"));
        }
        if !matches!(
            &node.bindings[0],
            PlanBinding {
                access: PlanBindingAccess::ReadOnlyStorage,
                resource: PlanBindingResource::Value(value),
                ..
            } if value == &activation
        ) {
            return Err(legacy_abi(
                "legacy projections do not share the canonical activation",
            ));
        }
        verify_tensor_binding(
            &node.bindings[1],
            candidate.layer,
            role,
            "weight",
            &[u64::from(output_width), u64::from(input_width)],
        )?;
        verify_tensor_binding(
            &node.bindings[2],
            candidate.layer,
            role,
            "bias",
            &[u64::from(output_width)],
        )?;
        let expected_buffer = buffer_id(&output_buffer(candidate.layer, role));
        if !matches!(
            &node.bindings[3],
            PlanBinding {
                access: PlanBindingAccess::ReadWriteStorage,
                resource: PlanBindingResource::OutputBuffer(output),
                ..
            } if output.buffer_id == expected_buffer && output.byte_length == output_bytes
        ) || !matches!(
            &node.bindings[4],
            PlanBinding {
                access: PlanBindingAccess::Uniform,
                resource: PlanBindingResource::UniformWords(uniform),
                ..
            } if uniform.words == [tokens, input_width, output_width, 0]
        ) {
            return Err(legacy_abi(
                "legacy output or uniform binding differs from the ABI",
            ));
        }
        let output = &node.outputs[0];
        if output.id != value_id(&output_value(candidate.layer, role))
            || output.dtype != PlanDtype::Float32
            || output.shape != [u64::from(tokens), u64::from(output_width)]
            || output.buffer_id != expected_buffer
            || output.byte_offset != 0
            || output.byte_length != output_bytes
        {
            return Err(legacy_abi("legacy logical output differs from the ABI"));
        }
    }

    let Some(external) = plan
        .external_values
        .iter()
        .find(|external| external.id == activation)
    else {
        return Err(legacy_abi("canonical activation external value is missing"));
    };
    let input_bytes = checked_f32_bytes(tokens, input_width)
        .map_err(|_| legacy_abi("activation byte count overflow"))?;
    if external.dtype != PlanDtype::Float32
        || external.shape != [u64::from(tokens), u64::from(input_width)]
        || external.buffer_id
            != buffer_id(&format!("activation.{}", activation_value(candidate.layer)))
        || external.byte_offset != 0
        || external.byte_length != input_bytes
    {
        return Err(legacy_abi(
            "canonical activation resource differs from the ABI",
        ));
    }
    Ok(())
}

fn verify_tensor_binding(
    binding: &PlanBinding,
    layer: usize,
    role: Role,
    suffix: &str,
    shape: &[u64],
) -> Result<(), VisionQkvPassError> {
    let PlanBinding {
        access: PlanBindingAccess::ReadOnlyStorage,
        resource: PlanBindingResource::Tensor(tensor),
        ..
    } = binding
    else {
        return Err(tensor_binding_mismatch("missing tensor binding"));
    };
    let expected_byte_length = shape
        .iter()
        .try_fold(1_u64, |elements, &dimension| {
            elements
                .checked_mul(dimension)
                .ok_or_else(|| tensor_binding_mismatch("tensor shape element count overflows u64"))
        })?
        .checked_mul(PlanDtype::Float32.byte_width())
        .ok_or_else(|| tensor_binding_mismatch("tensor executable byte count overflows u64"))?;
    if tensor.physical_name != tensor_physical(layer, role, suffix)
        || tensor.semantic_id != semantic_id(&tensor_semantic(layer, role, suffix))
        || tensor.dtype != PlanDtype::BFloat16
        || tensor.shape != shape
        || tensor.storage_format != PlanDtype::Float32
        || tensor.buffer_id != buffer_id(&tensor_buffer(layer, role, suffix))
        || tensor.byte_offset != 0
        || tensor.byte_length != expected_byte_length
    {
        return Err(tensor_binding_mismatch(format!(
            "{} {suffix} tensor identity differs from the checkpoint",
            role.name()
        )));
    }
    Ok(())
}

fn verify_live_qkv_dataflow(
    plan: &PlanIr,
    layer: usize,
    producer_indices: &[usize],
) -> Result<(), VisionQkvPassError> {
    let output_ids = Role::ALL.map(|role| output_value(layer, role));
    let mut positions = [0_usize; 3];
    for (role_index, output_id) in output_ids.iter().enumerate() {
        let matches = plan
            .outputs
            .iter()
            .enumerate()
            .filter_map(|(index, output)| (output.as_str() == output_id).then_some(index))
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(VisionQkvPassError::new(
                VisionQkvPassErrorCode::IllegalCandidateDataflow,
                "each Q/K/V value must be exported exactly once",
            ));
        }
        positions[role_index] = matches[0];
    }
    if !(positions[0] < positions[1] && positions[1] < positions[2]) {
        return Err(VisionQkvPassError::new(
            VisionQkvPassErrorCode::IllegalCandidateDataflow,
            "Q/K/V exports are not in canonical relative order",
        ));
    }
    for (index, node) in plan.nodes.iter().enumerate() {
        if producer_indices.contains(&index) {
            continue;
        }
        if node.bindings.iter().any(|binding| {
            matches!(
                &binding.resource,
                PlanBindingResource::Value(value)
                    if output_ids.iter().any(|output| output == value.as_str())
            )
        }) {
            return Err(VisionQkvPassError::new(
                VisionQkvPassErrorCode::IllegalCandidateDataflow,
                "legacy Q/K/V value has an internal consumer",
            ));
        }
    }
    Ok(())
}

fn construct_fused_node(
    layer: usize,
    nodes: [&PlanNode; 3],
    fused: VisionQkvFusedPlan,
) -> Result<PlanNode, VisionQkvPassError> {
    let shared_buffer = buffer_id(&format!("output.{}.qkv", semantic_layer_prefix(layer)));
    let mut bindings = vec![nodes[0].bindings[0].clone()];
    for node in nodes {
        bindings.extend_from_slice(&node.bindings[1..=2]);
    }
    bindings.push(PlanBinding {
        number: 7,
        access: PlanBindingAccess::ReadWriteStorage,
        resource: PlanBindingResource::OutputBuffer(PlanOutputBuffer {
            buffer_id: shared_buffer.clone(),
            byte_length: fused.output_layout.physical_bytes,
        }),
    });
    bindings.push(PlanBinding {
        number: 8,
        access: PlanBindingAccess::Uniform,
        resource: PlanBindingResource::UniformWords(PlanUniformResource {
            words: fused.uniform_words,
        }),
    });
    for (number, binding) in bindings.iter_mut().enumerate() {
        binding.number = u32::try_from(number).expect("nine bindings fit u32");
    }

    let slices = [
        fused.output_layout.query,
        fused.output_layout.key,
        fused.output_layout.value,
    ];
    let outputs = nodes
        .into_iter()
        .zip(slices)
        .map(|(node, slice)| {
            let mut output = node.outputs[0].clone();
            output.buffer_id = shared_buffer.clone();
            output.byte_offset = slice.offset;
            output.byte_length = slice.size;
            output
        })
        .collect::<Vec<_>>();
    let consumed = nodes
        .into_iter()
        .zip(Role::ALL)
        .map(|(node, role)| {
            let original = node.snapshot();
            let canonical_blake3 = original
                .canonical_node_blake3_hex()
                .map_err(|error| invalid_provenance(error.to_string()))?;
            Ok(PlanConsumedNode {
                role: role.name().to_owned(),
                original,
                canonical_blake3,
            })
        })
        .collect::<Result<Vec<_>, VisionQkvPassError>>()?;
    let source = semantic_id(&semantic_source(layer));
    Ok(PlanNode {
        id: node_id(&format!("{}.qkv_fused", semantic_layer_prefix(layer))),
        invocation: fused.invocation,
        uniform_words: fused.uniform_words,
        bindings,
        outputs,
        diagnostic_label: format!("{}.qkv.fused", semantic_layer_prefix(layer)),
        timestamp_label: None,
        source_semantic_ids: vec![source.clone()],
        rewrite_provenance: Some(PlanRewriteProvenance {
            pass_id: PASS_ID.to_owned(),
            source_semantic_ids: vec![source],
            consumed,
        }),
    })
}

fn verify_existing_fused(
    plan: &PlanIr,
    graph: &SemanticGraph,
    index: usize,
    target: VisionQkvFusedTargetLimits,
) -> Result<(), VisionQkvPassError> {
    let node = &plan.nodes[index];
    let Some(layer) = parse_fused_id(node.id.as_str()) else {
        return Err(malformed_candidate(
            "fused kernel has no canonical fused node ID",
        ));
    };
    verify_semantic_source(graph, layer)?;
    let Some(provenance) = &node.rewrite_provenance else {
        return Err(malformed_candidate(
            "fused candidate has no rewrite provenance",
        ));
    };
    if provenance.pass_id != PASS_ID
        || provenance.source_semantic_ids != [semantic_id(&semantic_source(layer))]
        || provenance.consumed.len() != 3
        || provenance
            .consumed
            .iter()
            .zip(Role::ALL)
            .any(|(consumed, role)| consumed.role != role.name())
    {
        return Err(invalid_provenance(
            "fused rewrite provenance is not canonical",
        ));
    }
    for consumed in &provenance.consumed {
        let canonical_blake3 = consumed
            .original
            .canonical_node_blake3_hex()
            .map_err(|error| invalid_provenance(error.to_string()))?;
        if consumed.canonical_blake3 != canonical_blake3 {
            return Err(invalid_provenance(
                "consumed snapshot canonical BLAKE3 does not match its contents",
            ));
        }
    }
    let legacy_nodes = provenance
        .consumed
        .iter()
        .map(|consumed| PlanNode {
            id: consumed.original.id.clone(),
            invocation: consumed.original.invocation,
            uniform_words: consumed.original.uniform_words,
            bindings: consumed.original.bindings.clone(),
            outputs: consumed.original.outputs.clone(),
            diagnostic_label: consumed.original.diagnostic_label.clone(),
            timestamp_label: consumed.original.timestamp_label.clone(),
            source_semantic_ids: consumed.original.source_semantic_ids.clone(),
            rewrite_provenance: None,
        })
        .collect::<Vec<_>>();
    let legacy_nodes: [PlanNode; 3] = legacy_nodes.try_into().map_err(|_| {
        invalid_provenance("fused provenance must contain exactly three consumed snapshots")
    })?;
    let [query, key, value] = legacy_nodes.each_ref();
    let references = [query, key, value];
    let snapshot_outputs = references
        .iter()
        .map(|legacy| match legacy.outputs.as_slice() {
            [output] => Ok(output.id.clone()),
            _ => Err(invalid_provenance(
                "each consumed snapshot must contain exactly one logical output",
            )),
        })
        .collect::<Result<Vec<_>, VisionQkvPassError>>()?;
    let temporary = PlanIr {
        schema_version: 1,
        external_values: plan.external_values.clone(),
        nodes: legacy_nodes.to_vec(),
        outputs: snapshot_outputs,
        requirements: PlanRequirements::derive(&plan.external_values, &legacy_nodes, 4, &[])
            .map_err(|error| invalid_provenance(error.to_string()))?,
    };
    let candidate = LegacyCandidate {
        layer,
        indices: [0, 1, 2],
    };
    verify_legacy_triple(&temporary, candidate, references)
        .map_err(|error| invalid_provenance(error.to_string()))?;
    let [tokens, input_width, output_width, _] = query.uniform_words;
    let geometry = plan_vision_qkv_fused_geometry(tokens, input_width, output_width, target)
        .map_err(|error| invalid_provenance(error.to_string()))?;
    let expected = construct_fused_node(layer, references, geometry)?;
    if &expected != node {
        return Err(VisionQkvPassError::fused_mismatch(
            classify_fused_mismatch(&expected, node),
            "fused node does not reproduce its consumed legacy snapshots",
        ));
    }
    verify_live_qkv_dataflow(plan, layer, &[index])?;
    Ok(())
}

fn classify_fused_mismatch(expected: &PlanNode, actual: &PlanNode) -> VisionQkvFusedMismatchKind {
    let output_identity_mismatch = expected.outputs.len() != actual.outputs.len()
        || expected
            .outputs
            .iter()
            .zip(&actual.outputs)
            .any(|(expected, actual)| expected.id != actual.id);
    let tensor_identity_mismatch = (1..=6).any(|binding_number| {
        let expected = expected
            .bindings
            .iter()
            .find(|binding| binding.number == binding_number)
            .map(|binding| &binding.resource);
        let actual = actual
            .bindings
            .iter()
            .find(|binding| binding.number == binding_number)
            .map(|binding| &binding.resource);
        expected != actual
    });
    if output_identity_mismatch || tensor_identity_mismatch {
        VisionQkvFusedMismatchKind::SemanticOrTensor
    } else {
        VisionQkvFusedMismatchKind::Structural
    }
}

/// Executor-ready fields extracted only after `fuse_vision_qkv` has accepted an
/// already-fused node. This is deliberately crate-private: callers cannot use
/// it to bypass the existing fused verifier.
pub(crate) struct VerifiedVisionQkvFusedDescriptorParts {
    pub(crate) layer: usize,
    pub(crate) invocation: InvocationPlan,
    pub(crate) uniform_words: [u32; 4],
    pub(crate) shared_output_bytes: u64,
    pub(crate) outputs: [PlanOutput; 3],
}

pub(crate) fn extract_verified_vision_qkv_fused_descriptor(
    plan: &PlanIr,
) -> Result<VerifiedVisionQkvFusedDescriptorParts, VisionQkvPassError> {
    let mut fused = plan
        .nodes
        .iter()
        .filter(|node| node.invocation.kernel == KernelId::VisionQkvFusedF32);
    let node = fused.next().ok_or_else(|| {
        VisionQkvPassError::new(
            VisionQkvPassErrorCode::InvalidPlan,
            "verified plan has no fused Q/K/V node",
        )
    })?;
    if fused.next().is_some() {
        return Err(VisionQkvPassError::new(
            VisionQkvPassErrorCode::InvalidPlan,
            "verified plan has more than one fused Q/K/V node",
        ));
    }
    let layer = parse_fused_id(node.id.as_str()).ok_or_else(|| {
        VisionQkvPassError::new(
            VisionQkvPassErrorCode::InvalidPlan,
            "verified fused Q/K/V node has no canonical layer identity",
        )
    })?;
    let shared_output_bytes = node
        .bindings
        .iter()
        .find_map(|binding| match (&binding.number, &binding.resource) {
            (7, PlanBindingResource::OutputBuffer(output)) => Some(output.byte_length),
            _ => None,
        })
        .ok_or_else(|| {
            VisionQkvPassError::new(
                VisionQkvPassErrorCode::InvalidPlan,
                "verified fused Q/K/V node has no shared output binding",
            )
        })?;
    let outputs = node.outputs.clone().try_into().map_err(|_| {
        VisionQkvPassError::new(
            VisionQkvPassErrorCode::InvalidPlan,
            "verified fused Q/K/V node must expose exactly three outputs",
        )
    })?;
    Ok(VerifiedVisionQkvFusedDescriptorParts {
        layer,
        invocation: node.invocation,
        uniform_words: node.uniform_words,
        shared_output_bytes,
        outputs,
    })
}

fn verify_semantic_source(graph: &SemanticGraph, layer: usize) -> Result<(), VisionQkvPassError> {
    graph.verify().map_err(|error| {
        VisionQkvPassError::new(VisionQkvPassErrorCode::SemanticMismatch, error.to_string())
    })?;
    let source = semantic_id(&semantic_source(layer));
    let expected_input = semantic_id(&activation_value(layer));
    let Some(node) = graph.node(&source) else {
        return Err(VisionQkvPassError::new(
            VisionQkvPassErrorCode::SemanticMismatch,
            format!("canonical SemanticIR node {source} is missing"),
        ));
    };
    if node.op != SemanticOp::VisionQkv
        || node.inputs != [expected_input]
        || node.source_ids != [source.clone()]
    {
        return Err(VisionQkvPassError::new(
            VisionQkvPassErrorCode::SemanticMismatch,
            format!("canonical SemanticIR node {source} does not match Q/K/V"),
        ));
    }
    Ok(())
}

fn tensor_role_signals(node: &PlanNode) -> BTreeSet<(usize, Role)> {
    let mut signals = BTreeSet::new();
    for binding in &node.bindings {
        let PlanBindingResource::Tensor(tensor) = &binding.resource else {
            continue;
        };
        let Some((layer, role)) = parse_tensor_semantic(tensor.semantic_id.as_str()) else {
            continue;
        };
        signals.insert((layer, role));
    }
    signals
}

fn parse_projection_id(value: &str) -> Option<(usize, Role)> {
    let segments = value.split('.').collect::<Vec<_>>();
    if segments.len() != 4 || segments[0] != "vision" || segments[1] != "layer" {
        return None;
    }
    let layer = parse_layer(segments[2])?;
    let role = match segments[3] {
        "query" => Role::Query,
        "key" => Role::Key,
        "value" => Role::Value,
        _ => return None,
    };
    Some((layer, role))
}

fn parse_fused_id(value: &str) -> Option<usize> {
    let segments = value.split('.').collect::<Vec<_>>();
    (segments.len() == 4
        && segments[0] == "vision"
        && segments[1] == "layer"
        && segments[3] == "qkv_fused")
        .then(|| parse_layer(segments[2]))?
}

fn parse_semantic_source(value: &str) -> Option<usize> {
    let segments = value.split('.').collect::<Vec<_>>();
    (segments.len() == 4
        && segments[0] == "vision"
        && segments[1] == "layer"
        && segments[3] == "qkv")
        .then(|| parse_layer(segments[2]))?
}

fn parse_tensor_semantic(value: &str) -> Option<(usize, Role)> {
    let segments = value.split('.').collect::<Vec<_>>();
    if segments.len() != 6
        || segments[0] != "vision"
        || segments[1] != "layer"
        || segments[3] != "attention"
        || !matches!(segments[5], "weight" | "bias")
    {
        return None;
    }
    let layer = parse_layer(segments[2])?;
    let role = match segments[4] {
        "q" => Role::Query,
        "k" => Role::Key,
        "v" => Role::Value,
        _ => return None,
    };
    Some((layer, role))
}

fn parse_layer(value: &str) -> Option<usize> {
    (value.len() == 2 && value.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| value.parse().ok())?
}

fn checked_f32_bytes(rows: u32, columns: u32) -> Result<u64, VisionQkvPassError> {
    u64::from(rows)
        .checked_mul(u64::from(columns))
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| invalid_geometry("F32 byte count overflow"))
}

fn semantic_layer_prefix(layer: usize) -> String {
    format!("vision.layer.{layer:02}")
}

fn semantic_source(layer: usize) -> String {
    format!("{}.qkv", semantic_layer_prefix(layer))
}

fn activation_value(layer: usize) -> String {
    format!("{}.norm1", semantic_layer_prefix(layer))
}

fn output_value(layer: usize, role: Role) -> String {
    format!("{}.{}", semantic_layer_prefix(layer), role.name())
}

fn projection_node_id(layer: usize, role: Role) -> String {
    output_value(layer, role)
}

pub(crate) fn tensor_semantic(layer: usize, role: Role, suffix: &str) -> String {
    format!(
        "{}.attention.{}.{suffix}",
        semantic_layer_prefix(layer),
        role.letter()
    )
}

pub(crate) fn tensor_physical(layer: usize, role: Role, suffix: &str) -> String {
    format!(
        "visual.vision_model.encoder.layers.{layer}.self_attn.{}.{suffix}",
        role.projection()
    )
}

fn tensor_buffer(layer: usize, role: Role, suffix: &str) -> String {
    format!("tensor.vision.layer.{layer:02}.{}_{suffix}", role.letter())
}

fn output_buffer(layer: usize, role: Role) -> String {
    format!("output.vision.layer.{layer:02}.{}", role.name())
}

fn node_id(value: &str) -> PlanNodeId {
    PlanNodeId::parse(value).expect("internally generated PlanNodeId must be valid")
}

fn value_id(value: &str) -> PlanValueId {
    PlanValueId::parse(value).expect("internally generated PlanValueId must be valid")
}

fn buffer_id(value: &str) -> PlanBufferId {
    PlanBufferId::parse(value).expect("internally generated PlanBufferId must be valid")
}

fn semantic_id(value: &str) -> SemanticId {
    SemanticId::parse(value).expect("internally generated SemanticId must be valid")
}

fn invalid_geometry(message: impl Into<String>) -> VisionQkvPassError {
    VisionQkvPassError::new(VisionQkvPassErrorCode::InvalidGeometry, message)
}

fn malformed_candidate(message: impl Into<String>) -> VisionQkvPassError {
    VisionQkvPassError::new(VisionQkvPassErrorCode::MalformedCandidate, message)
}

fn legacy_abi(message: impl Into<String>) -> VisionQkvPassError {
    VisionQkvPassError::new(VisionQkvPassErrorCode::LegacyAbiMismatch, message)
}

fn tensor_binding_mismatch(message: impl Into<String>) -> VisionQkvPassError {
    VisionQkvPassError::new(VisionQkvPassErrorCode::TensorBindingMismatch, message)
}

fn invalid_provenance(message: impl Into<String>) -> VisionQkvPassError {
    VisionQkvPassError::new(VisionQkvPassErrorCode::InvalidProvenance, message)
}
