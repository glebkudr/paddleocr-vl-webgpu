//! Concrete SemanticIR for PaddleOCR-VL-1.6.
//!
//! This is intentionally not a generic operator graph. Stable IDs and explicit
//! operation families survive later fusion/lowering and form the trace ABI.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

use serde::Serialize;

mod plan;

pub use plan::*;

const MAX_SEMANTIC_ID_BYTES: usize = 128;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SemanticId(String);

impl SemanticId {
    pub fn parse(value: &str) -> Result<Self, SemanticIdError> {
        if value.is_empty() || value.len() > MAX_SEMANTIC_ID_BYTES {
            return Err(SemanticIdError(value.to_owned()));
        }
        for segment in value.split('.') {
            if segment.is_empty() || !valid_segment(segment) {
                return Err(SemanticIdError(value.to_owned()));
            }
        }
        Ok(Self(value.to_owned()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SemanticId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticIdError(String);

impl fmt::Display for SemanticIdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid SemanticId {:?}", self.0)
    }
}

impl Error for SemanticIdError {}

fn valid_segment(segment: &str) -> bool {
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

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum SemanticOpKind {
    ImageSmartResize,
    NormalizeRgb,
    PatchProjection,
    VisionPositionEmbedding,
    VisionLayerNorm,
    VisionQkv,
    VisionRope,
    VisionAttention,
    VisionOutProjection,
    VisionMlp,
    ResidualAdd,
    ProjectorMerge2x2,
    ProjectorMlp,
    TokenEmbedding,
    MultimodalAssemble,
    MRopeIndex,
    DecoderRmsNorm,
    DecoderQkv,
    DecoderMRope,
    DecoderPrefillAttention,
    DecoderKvAppend,
    DecoderDecodeAttention,
    DecoderOutProjection,
    DecoderSwiGlu,
    FinalRmsNorm,
    LmHead,
    TopK,
    Sampling,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticOp {
    ImageSmartResize,
    NormalizeRgb,
    PatchProjection,
    VisionPositionEmbedding,
    VisionLayerNorm,
    VisionQkv,
    VisionRope,
    VisionAttention,
    VisionOutProjection,
    VisionMlp,
    ResidualAdd,
    ProjectorMerge2x2,
    ProjectorLayerNorm,
    ProjectorLinear,
    Gelu,
    TokenEmbedding,
    MultimodalAssemble,
    MRopeIndex,
    DecoderRmsNorm,
    DecoderQkv,
    DecoderMRope,
    DecoderPrefillAttention,
    DecoderKvAppend,
    DecoderDecodeAttention,
    DecoderOutProjection,
    DecoderSwiGlu,
    FinalRmsNorm,
    LmHead,
    TopK,
    Sampling,
}

impl SemanticOp {
    #[must_use]
    pub const fn stable_name(self) -> &'static str {
        match self {
            Self::ImageSmartResize => "image_smart_resize",
            Self::NormalizeRgb => "normalize_rgb",
            Self::PatchProjection => "patch_projection",
            Self::VisionPositionEmbedding => "vision_position_embedding",
            Self::VisionLayerNorm => "vision_layer_norm",
            Self::VisionQkv => "vision_qkv",
            Self::VisionRope => "vision_rope",
            Self::VisionAttention => "vision_attention",
            Self::VisionOutProjection => "vision_out_projection",
            Self::VisionMlp => "vision_mlp",
            Self::ResidualAdd => "residual_add",
            Self::ProjectorMerge2x2 => "projector_merge_2x2",
            Self::ProjectorLayerNorm => "projector_layer_norm",
            Self::ProjectorLinear => "projector_linear",
            Self::Gelu => "gelu",
            Self::TokenEmbedding => "token_embedding",
            Self::MultimodalAssemble => "multimodal_assemble",
            Self::MRopeIndex => "m_rope_index",
            Self::DecoderRmsNorm => "decoder_rms_norm",
            Self::DecoderQkv => "decoder_qkv",
            Self::DecoderMRope => "decoder_m_rope",
            Self::DecoderPrefillAttention => "decoder_prefill_attention",
            Self::DecoderKvAppend => "decoder_kv_append",
            Self::DecoderDecodeAttention => "decoder_decode_attention",
            Self::DecoderOutProjection => "decoder_out_projection",
            Self::DecoderSwiGlu => "decoder_swi_glu",
            Self::FinalRmsNorm => "final_rms_norm",
            Self::LmHead => "lm_head",
            Self::TopK => "top_k",
            Self::Sampling => "sampling",
        }
    }

    #[must_use]
    pub const fn kind(self) -> SemanticOpKind {
        match self {
            Self::ImageSmartResize => SemanticOpKind::ImageSmartResize,
            Self::NormalizeRgb => SemanticOpKind::NormalizeRgb,
            Self::PatchProjection => SemanticOpKind::PatchProjection,
            Self::VisionPositionEmbedding => SemanticOpKind::VisionPositionEmbedding,
            Self::VisionLayerNorm => SemanticOpKind::VisionLayerNorm,
            Self::VisionQkv => SemanticOpKind::VisionQkv,
            Self::VisionRope => SemanticOpKind::VisionRope,
            Self::VisionAttention => SemanticOpKind::VisionAttention,
            Self::VisionOutProjection => SemanticOpKind::VisionOutProjection,
            Self::VisionMlp => SemanticOpKind::VisionMlp,
            Self::ResidualAdd => SemanticOpKind::ResidualAdd,
            Self::ProjectorMerge2x2 => SemanticOpKind::ProjectorMerge2x2,
            Self::ProjectorLayerNorm | Self::ProjectorLinear | Self::Gelu => {
                SemanticOpKind::ProjectorMlp
            }
            Self::TokenEmbedding => SemanticOpKind::TokenEmbedding,
            Self::MultimodalAssemble => SemanticOpKind::MultimodalAssemble,
            Self::MRopeIndex => SemanticOpKind::MRopeIndex,
            Self::DecoderRmsNorm => SemanticOpKind::DecoderRmsNorm,
            Self::DecoderQkv => SemanticOpKind::DecoderQkv,
            Self::DecoderMRope => SemanticOpKind::DecoderMRope,
            Self::DecoderPrefillAttention => SemanticOpKind::DecoderPrefillAttention,
            Self::DecoderKvAppend => SemanticOpKind::DecoderKvAppend,
            Self::DecoderDecodeAttention => SemanticOpKind::DecoderDecodeAttention,
            Self::DecoderOutProjection => SemanticOpKind::DecoderOutProjection,
            Self::DecoderSwiGlu => SemanticOpKind::DecoderSwiGlu,
            Self::FinalRmsNorm => SemanticOpKind::FinalRmsNorm,
            Self::LmHead => SemanticOpKind::LmHead,
            Self::TopK => SemanticOpKind::TopK,
            Self::Sampling => SemanticOpKind::Sampling,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticNode {
    pub id: SemanticId,
    pub op: SemanticOp,
    pub inputs: Vec<SemanticId>,
    /// Original semantic operations represented by this node after fusion.
    pub source_ids: Vec<SemanticId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticGraph {
    nodes: Vec<SemanticNode>,
}

impl SemanticGraph {
    #[must_use]
    pub fn from_nodes(nodes: Vec<SemanticNode>) -> Self {
        Self { nodes }
    }

    #[must_use]
    pub fn paddleocr_vl_16() -> Self {
        Self {
            nodes: build_model_graph(),
        }
    }

    #[must_use]
    pub fn nodes(&self) -> &[SemanticNode] {
        &self.nodes
    }

    #[must_use]
    pub fn node(&self, id: &SemanticId) -> Option<&SemanticNode> {
        self.nodes.iter().find(|node| &node.id == id)
    }

    pub fn verify(&self) -> Result<(), GraphError> {
        let mut node_indices = BTreeMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            if node_indices.insert(node.id.as_str(), index).is_some() {
                return Err(GraphError::node(
                    GraphErrorCode::DuplicateSemanticId,
                    &node.id,
                    "semantic node ID is duplicated",
                ));
            }
        }

        let mut globally_claimed_sources: BTreeMap<&str, &SemanticId> = BTreeMap::new();
        for node in &self.nodes {
            if node.source_ids.is_empty() {
                return Err(GraphError::node(
                    GraphErrorCode::EmptySourceIds,
                    &node.id,
                    "node has no source semantic provenance",
                ));
            }
            let mut local_sources = BTreeSet::new();
            for source_id in &node.source_ids {
                if !local_sources.insert(source_id.as_str()) {
                    return Err(GraphError::node(
                        GraphErrorCode::DuplicateSourceId,
                        &node.id,
                        "source semantic ID is duplicated within the node",
                    ));
                }
                if let Some(previous_node) =
                    globally_claimed_sources.insert(source_id.as_str(), &node.id)
                {
                    return Err(GraphError::node(
                        GraphErrorCode::SourceIdClaimedByMultipleNodes,
                        &node.id,
                        format!(
                            "source {} is already represented by {}",
                            source_id, previous_node
                        ),
                    ));
                }
            }

            let mut inputs = BTreeSet::new();
            for input in &node.inputs {
                if !inputs.insert(input.as_str()) {
                    return Err(GraphError::node(
                        GraphErrorCode::DuplicateInput,
                        &node.id,
                        format!("input {input} is duplicated"),
                    ));
                }
            }
        }

        for node in &self.nodes {
            for input in &node.inputs {
                if !node_indices.contains_key(input.as_str()) {
                    return Err(GraphError::node(
                        GraphErrorCode::DanglingInput,
                        &node.id,
                        format!("input {input} does not exist"),
                    ));
                }
            }
        }
        verify_acyclic(&self.nodes, &node_indices)?;

        for node in &self.nodes {
            if !valid_arity(node.op, node.inputs.len()) {
                return Err(GraphError::node(
                    GraphErrorCode::InvalidInputArity,
                    &node.id,
                    format!(
                        "operation {} does not accept {} inputs",
                        node.op.stable_name(),
                        node.inputs.len()
                    ),
                ));
            }
            let input_ops: Vec<_> = node
                .inputs
                .iter()
                .map(|input| self.nodes[node_indices[input.as_str()]].op)
                .collect();
            if !valid_input_ops(node.op, &input_ops) {
                return Err(GraphError::node(
                    GraphErrorCode::InvalidInputKind,
                    &node.id,
                    format!(
                        "operation {} has incompatible inputs",
                        node.op.stable_name()
                    ),
                ));
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, GraphError> {
        self.verify()?;
        #[derive(Serialize)]
        struct Record<'a> {
            id: &'a str,
            inputs: Vec<&'a str>,
            op: &'a str,
            source_ids: Vec<&'a str>,
        }
        let records: Vec<_> = self
            .nodes
            .iter()
            .map(|node| Record {
                id: node.id.as_str(),
                inputs: node.inputs.iter().map(SemanticId::as_str).collect(),
                op: node.op.stable_name(),
                source_ids: node.source_ids.iter().map(SemanticId::as_str).collect(),
            })
            .collect();
        let mut bytes = serde_json::to_vec(&records)
            .expect("verified SemanticIR records are always serializable");
        bytes.push(b'\n');
        Ok(bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphErrorCode {
    DuplicateSemanticId,
    DanglingInput,
    Cycle,
    EmptySourceIds,
    DuplicateSourceId,
    SourceIdClaimedByMultipleNodes,
    DuplicateInput,
    InvalidInputArity,
    InvalidInputKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphError {
    code: GraphErrorCode,
    node_id: Option<SemanticId>,
    message: String,
}

impl GraphError {
    #[must_use]
    pub const fn code(&self) -> GraphErrorCode {
        self.code
    }

    fn node(code: GraphErrorCode, node_id: &SemanticId, message: impl Into<String>) -> Self {
        Self {
            code,
            node_id: Some(node_id.clone()),
            message: message.into(),
        }
    }

    fn graph(code: GraphErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            node_id: None,
            message: message.into(),
        }
    }
}

impl fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "SemanticIR error {:?}: {}",
            self.code, self.message
        )?;
        if let Some(node_id) = &self.node_id {
            write!(formatter, " (node {node_id})")?;
        }
        Ok(())
    }
}

impl Error for GraphError {}

fn verify_acyclic(
    nodes: &[SemanticNode],
    node_indices: &BTreeMap<&str, usize>,
) -> Result<(), GraphError> {
    let mut indegree: Vec<_> = nodes.iter().map(|node| node.inputs.len()).collect();
    let mut dependents = vec![Vec::new(); nodes.len()];
    for (node_index, node) in nodes.iter().enumerate() {
        for input in &node.inputs {
            dependents[node_indices[input.as_str()]].push(node_index);
        }
    }
    let mut queue: VecDeque<_> = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect();
    let mut visited = 0;
    while let Some(index) = queue.pop_front() {
        visited += 1;
        for dependent in &dependents[index] {
            indegree[*dependent] -= 1;
            if indegree[*dependent] == 0 {
                queue.push_back(*dependent);
            }
        }
    }
    if visited != nodes.len() {
        return Err(GraphError::graph(
            GraphErrorCode::Cycle,
            "semantic graph contains a dependency cycle",
        ));
    }
    Ok(())
}

fn valid_arity(op: SemanticOp, arity: usize) -> bool {
    match op {
        SemanticOp::ImageSmartResize | SemanticOp::TokenEmbedding => arity == 0,
        SemanticOp::VisionLayerNorm | SemanticOp::DecoderRmsNorm | SemanticOp::DecoderSwiGlu => {
            matches!(arity, 1 | 2)
        }
        SemanticOp::ResidualAdd => arity >= 2,
        SemanticOp::VisionAttention
        | SemanticOp::MultimodalAssemble
        | SemanticOp::DecoderMRope
        | SemanticOp::DecoderDecodeAttention
        | SemanticOp::DecoderOutProjection => arity == 2,
        _ => arity == 1,
    }
}

fn valid_input_ops(op: SemanticOp, inputs: &[SemanticOp]) -> bool {
    use SemanticOp as Op;
    matches!(
        (op, inputs),
        (Op::ImageSmartResize | Op::TokenEmbedding, [])
            | (Op::NormalizeRgb, [Op::ImageSmartResize])
            | (Op::PatchProjection, [Op::NormalizeRgb])
            | (Op::VisionPositionEmbedding, [Op::PatchProjection])
            | (
                Op::VisionLayerNorm,
                [Op::VisionPositionEmbedding | Op::ResidualAdd]
            )
            | (
                Op::VisionLayerNorm,
                [
                    Op::VisionPositionEmbedding | Op::ResidualAdd,
                    Op::VisionOutProjection,
                ]
            )
            | (Op::VisionQkv, [Op::VisionLayerNorm])
            | (Op::VisionRope, [Op::VisionQkv])
            | (Op::VisionAttention, [Op::VisionQkv, Op::VisionRope])
            | (Op::VisionOutProjection, [Op::VisionAttention])
            | (Op::VisionMlp, [Op::VisionLayerNorm])
            | (
                Op::ResidualAdd,
                [
                    Op::VisionPositionEmbedding | Op::ResidualAdd,
                    Op::VisionOutProjection,
                    Op::VisionMlp,
                ]
            )
            | (
                Op::ResidualAdd,
                [
                    Op::MultimodalAssemble | Op::ResidualAdd,
                    Op::DecoderOutProjection,
                    Op::DecoderSwiGlu,
                ]
            )
            | (Op::ProjectorMerge2x2, [Op::VisionLayerNorm])
            | (Op::ProjectorLayerNorm, [Op::ProjectorMerge2x2])
            | (Op::ProjectorLinear, [Op::ProjectorLayerNorm | Op::Gelu])
            | (Op::Gelu, [Op::ProjectorLinear])
            | (
                Op::MultimodalAssemble,
                [Op::TokenEmbedding, Op::ProjectorLinear]
            )
            | (Op::MRopeIndex, [Op::MultimodalAssemble])
            | (
                Op::DecoderRmsNorm,
                [Op::MultimodalAssemble | Op::ResidualAdd]
            )
            | (
                Op::DecoderRmsNorm,
                [
                    Op::MultimodalAssemble | Op::ResidualAdd,
                    Op::DecoderOutProjection,
                ]
            )
            | (Op::DecoderQkv, [Op::DecoderRmsNorm])
            | (Op::DecoderMRope, [Op::DecoderQkv, Op::MRopeIndex])
            | (Op::DecoderPrefillAttention, [Op::DecoderMRope])
            | (Op::DecoderKvAppend, [Op::DecoderMRope])
            | (
                Op::DecoderDecodeAttention,
                [Op::DecoderMRope, Op::DecoderKvAppend]
            )
            | (
                Op::DecoderOutProjection,
                [Op::DecoderPrefillAttention, Op::DecoderDecodeAttention]
            )
            | (Op::DecoderSwiGlu, [Op::DecoderRmsNorm | Op::DecoderSwiGlu])
            | (Op::DecoderSwiGlu, [Op::DecoderSwiGlu, Op::DecoderSwiGlu])
            | (Op::FinalRmsNorm, [Op::ResidualAdd])
            | (Op::LmHead, [Op::FinalRmsNorm])
            | (Op::TopK, [Op::LmHead])
            | (Op::Sampling, [Op::TopK])
    )
}

fn sid(value: impl AsRef<str>) -> SemanticId {
    SemanticId::parse(value.as_ref()).expect("built-in SemanticId is valid")
}

fn push(nodes: &mut Vec<SemanticNode>, id: impl AsRef<str>, op: SemanticOp, inputs: Vec<String>) {
    let id = sid(id);
    nodes.push(SemanticNode {
        source_ids: vec![id.clone()],
        id,
        op,
        inputs: inputs.into_iter().map(sid).collect(),
    });
}

fn build_model_graph() -> Vec<SemanticNode> {
    use SemanticOp as Op;

    let mut nodes = Vec::with_capacity(467);
    push(
        &mut nodes,
        "preprocess.resize",
        Op::ImageSmartResize,
        vec![],
    );
    push(
        &mut nodes,
        "preprocess.normalize",
        Op::NormalizeRgb,
        vec!["preprocess.resize".into()],
    );
    push(
        &mut nodes,
        "vision.embeddings.patch",
        Op::PatchProjection,
        vec!["preprocess.normalize".into()],
    );
    push(
        &mut nodes,
        "vision.embeddings.position",
        Op::VisionPositionEmbedding,
        vec!["vision.embeddings.patch".into()],
    );

    let mut previous = "vision.embeddings.position".to_owned();
    for layer in 0..27 {
        let prefix = format!("vision.layer.{layer:02}");
        let norm1 = format!("{prefix}.norm1");
        let qkv = format!("{prefix}.qkv");
        let rope = format!("{prefix}.rope");
        let attention = format!("{prefix}.attention");
        let out = format!("{prefix}.out");
        let norm2 = format!("{prefix}.norm2");
        let mlp = format!("{prefix}.mlp");
        let output = format!("{prefix}.output");

        push(
            &mut nodes,
            &norm1,
            Op::VisionLayerNorm,
            vec![previous.clone()],
        );
        push(&mut nodes, &qkv, Op::VisionQkv, vec![norm1.clone()]);
        push(&mut nodes, &rope, Op::VisionRope, vec![qkv.clone()]);
        push(&mut nodes, &attention, Op::VisionAttention, vec![qkv, rope]);
        push(&mut nodes, &out, Op::VisionOutProjection, vec![attention]);
        push(
            &mut nodes,
            &norm2,
            Op::VisionLayerNorm,
            vec![previous.clone(), out.clone()],
        );
        push(&mut nodes, &mlp, Op::VisionMlp, vec![norm2]);
        push(
            &mut nodes,
            &output,
            Op::ResidualAdd,
            vec![previous, out, mlp],
        );
        previous = output;
    }

    push(
        &mut nodes,
        "vision.post_norm",
        Op::VisionLayerNorm,
        vec![previous],
    );
    push(
        &mut nodes,
        "projector.merge",
        Op::ProjectorMerge2x2,
        vec!["vision.post_norm".into()],
    );
    push(
        &mut nodes,
        "projector.pre_norm",
        Op::ProjectorLayerNorm,
        vec!["projector.merge".into()],
    );
    push(
        &mut nodes,
        "projector.linear1",
        Op::ProjectorLinear,
        vec!["projector.pre_norm".into()],
    );
    push(
        &mut nodes,
        "projector.gelu",
        Op::Gelu,
        vec!["projector.linear1".into()],
    );
    push(
        &mut nodes,
        "projector.linear2",
        Op::ProjectorLinear,
        vec!["projector.gelu".into()],
    );

    push(&mut nodes, "decoder.embedding", Op::TokenEmbedding, vec![]);
    push(
        &mut nodes,
        "multimodal.inputs_embeds",
        Op::MultimodalAssemble,
        vec!["decoder.embedding".into(), "projector.linear2".into()],
    );
    push(
        &mut nodes,
        "decoder.mrope.index",
        Op::MRopeIndex,
        vec!["multimodal.inputs_embeds".into()],
    );

    previous = "multimodal.inputs_embeds".to_owned();
    for layer in 0..18 {
        let prefix = format!("decoder.layer.{layer:02}");
        let norm1 = format!("{prefix}.norm1");
        let qkv = format!("{prefix}.qkv");
        let mrope = format!("{prefix}.mrope");
        let prefill = format!("{prefix}.attention.prefill");
        let kv_append = format!("{prefix}.kv_append");
        let decode = format!("{prefix}.attention.decode");
        let attention_out = format!("{prefix}.attention.out");
        let norm2 = format!("{prefix}.norm2");
        let gate = format!("{prefix}.mlp.gate");
        let up = format!("{prefix}.mlp.up");
        let activation = format!("{prefix}.mlp.activation");
        let down = format!("{prefix}.mlp.down");
        let output = format!("{prefix}.output");

        push(
            &mut nodes,
            &norm1,
            Op::DecoderRmsNorm,
            vec![previous.clone()],
        );
        push(&mut nodes, &qkv, Op::DecoderQkv, vec![norm1]);
        push(
            &mut nodes,
            &mrope,
            Op::DecoderMRope,
            vec![qkv, "decoder.mrope.index".into()],
        );
        push(
            &mut nodes,
            &prefill,
            Op::DecoderPrefillAttention,
            vec![mrope.clone()],
        );
        push(
            &mut nodes,
            &kv_append,
            Op::DecoderKvAppend,
            vec![mrope.clone()],
        );
        push(
            &mut nodes,
            &decode,
            Op::DecoderDecodeAttention,
            vec![mrope, kv_append],
        );
        push(
            &mut nodes,
            &attention_out,
            Op::DecoderOutProjection,
            vec![prefill, decode],
        );
        push(
            &mut nodes,
            &norm2,
            Op::DecoderRmsNorm,
            vec![previous.clone(), attention_out.clone()],
        );
        push(&mut nodes, &gate, Op::DecoderSwiGlu, vec![norm2.clone()]);
        push(&mut nodes, &up, Op::DecoderSwiGlu, vec![norm2]);
        push(&mut nodes, &activation, Op::DecoderSwiGlu, vec![gate]);
        push(&mut nodes, &down, Op::DecoderSwiGlu, vec![activation, up]);
        push(
            &mut nodes,
            &output,
            Op::ResidualAdd,
            vec![previous, attention_out, down],
        );
        previous = output;
    }

    push(
        &mut nodes,
        "decoder.final_norm",
        Op::FinalRmsNorm,
        vec![previous],
    );
    push(
        &mut nodes,
        "lm_head",
        Op::LmHead,
        vec!["decoder.final_norm".into()],
    );
    push(&mut nodes, "top_k", Op::TopK, vec!["lm_head".into()]);
    push(&mut nodes, "sampling", Op::Sampling, vec!["top_k".into()]);

    debug_assert_eq!(nodes.len(), 467);
    nodes
}
