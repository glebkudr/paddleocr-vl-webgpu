use std::collections::BTreeSet;

use pvlc_ir::{
    GraphErrorCode, SemanticGraph, SemanticId, SemanticNode, SemanticOp, SemanticOpKind,
};

const EXPECTED_GRAPH_NODE_COUNT: usize = 467;
const EXPECTED_GRAPH_CANONICAL_LEN: usize = 68_833;
const EXPECTED_GRAPH_BLAKE3: &str =
    "2b2556c363545dcef569e3e6d0db01967973a081706c8483e1c5af3c7dc5bf73";

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExpectedNode {
    id: String,
    op: &'static str,
    inputs: Vec<String>,
    source_ids: Vec<String>,
}

fn push(
    out: &mut Vec<ExpectedNode>,
    id: impl Into<String>,
    op: &'static str,
    inputs: impl IntoIterator<Item = impl Into<String>>,
) {
    let id = id.into();
    out.push(ExpectedNode {
        source_ids: vec![id.clone()],
        id,
        op,
        inputs: inputs.into_iter().map(Into::into).collect(),
    });
}

/// Independent, model-specific graph fixture. It intentionally repeats the
/// topology instead of sharing a builder with production code.
fn expected_graph() -> Vec<ExpectedNode> {
    let mut nodes = Vec::new();
    push(
        &mut nodes,
        "preprocess.resize",
        "image_smart_resize",
        Vec::<String>::new(),
    );
    push(
        &mut nodes,
        "preprocess.normalize",
        "normalize_rgb",
        ["preprocess.resize"],
    );
    push(
        &mut nodes,
        "vision.embeddings.patch",
        "patch_projection",
        ["preprocess.normalize"],
    );
    push(
        &mut nodes,
        "vision.embeddings.position",
        "vision_position_embedding",
        ["vision.embeddings.patch"],
    );

    let mut previous = "vision.embeddings.position".to_owned();
    for layer in 0..27 {
        let p = format!("vision.layer.{layer:02}");
        let norm1 = format!("{p}.norm1");
        let qkv = format!("{p}.qkv");
        let rope = format!("{p}.rope");
        let attention = format!("{p}.attention");
        let out = format!("{p}.out");
        let norm2 = format!("{p}.norm2");
        let mlp = format!("{p}.mlp");
        let output = format!("{p}.output");

        push(&mut nodes, &norm1, "vision_layer_norm", [&previous]);
        push(&mut nodes, &qkv, "vision_qkv", [&norm1]);
        push(&mut nodes, &rope, "vision_rope", [&qkv]);
        push(&mut nodes, &attention, "vision_attention", [&qkv, &rope]);
        push(&mut nodes, &out, "vision_out_projection", [&attention]);
        push(&mut nodes, &norm2, "vision_layer_norm", [&previous, &out]);
        push(&mut nodes, &mlp, "vision_mlp", [&norm2]);
        push(&mut nodes, &output, "residual_add", [&previous, &out, &mlp]);
        previous = output;
    }

    push(
        &mut nodes,
        "vision.post_norm",
        "vision_layer_norm",
        [&previous],
    );
    push(
        &mut nodes,
        "projector.merge",
        "projector_merge_2x2",
        ["vision.post_norm"],
    );
    push(
        &mut nodes,
        "projector.pre_norm",
        "projector_layer_norm",
        ["projector.merge"],
    );
    push(
        &mut nodes,
        "projector.linear1",
        "projector_linear",
        ["projector.pre_norm"],
    );
    push(&mut nodes, "projector.gelu", "gelu", ["projector.linear1"]);
    push(
        &mut nodes,
        "projector.linear2",
        "projector_linear",
        ["projector.gelu"],
    );

    push(
        &mut nodes,
        "decoder.embedding",
        "token_embedding",
        Vec::<String>::new(),
    );
    push(
        &mut nodes,
        "multimodal.inputs_embeds",
        "multimodal_assemble",
        ["decoder.embedding", "projector.linear2"],
    );
    push(
        &mut nodes,
        "decoder.mrope.index",
        "m_rope_index",
        ["multimodal.inputs_embeds"],
    );

    previous = "multimodal.inputs_embeds".to_owned();
    for layer in 0..18 {
        let p = format!("decoder.layer.{layer:02}");
        let norm1 = format!("{p}.norm1");
        let qkv = format!("{p}.qkv");
        let mrope = format!("{p}.mrope");
        let prefill = format!("{p}.attention.prefill");
        let kv_append = format!("{p}.kv_append");
        let decode = format!("{p}.attention.decode");
        let attention_out = format!("{p}.attention.out");
        let norm2 = format!("{p}.norm2");
        let gate = format!("{p}.mlp.gate");
        let up = format!("{p}.mlp.up");
        let activation = format!("{p}.mlp.activation");
        let down = format!("{p}.mlp.down");
        let output = format!("{p}.output");

        push(&mut nodes, &norm1, "decoder_rms_norm", [&previous]);
        push(&mut nodes, &qkv, "decoder_qkv", [&norm1]);
        push(
            &mut nodes,
            &mrope,
            "decoder_m_rope",
            [&qkv, "decoder.mrope.index"],
        );
        push(&mut nodes, &prefill, "decoder_prefill_attention", [&mrope]);
        push(&mut nodes, &kv_append, "decoder_kv_append", [&mrope]);
        push(
            &mut nodes,
            &decode,
            "decoder_decode_attention",
            [&mrope, &kv_append],
        );
        // SemanticIR retains both mutually-exclusive execution branches. PlanIR
        // specializes this join to prefill or decode later.
        push(
            &mut nodes,
            &attention_out,
            "decoder_out_projection",
            [&prefill, &decode],
        );
        push(
            &mut nodes,
            &norm2,
            "decoder_rms_norm",
            [&previous, &attention_out],
        );
        push(&mut nodes, &gate, "decoder_swi_glu", [&norm2]);
        push(&mut nodes, &up, "decoder_swi_glu", [&norm2]);
        push(&mut nodes, &activation, "decoder_swi_glu", [&gate]);
        push(&mut nodes, &down, "decoder_swi_glu", [&activation, &up]);
        push(
            &mut nodes,
            &output,
            "residual_add",
            [&previous, &attention_out, &down],
        );
        previous = output;
    }

    push(
        &mut nodes,
        "decoder.final_norm",
        "final_rms_norm",
        [&previous],
    );
    push(&mut nodes, "lm_head", "lm_head", ["decoder.final_norm"]);
    push(&mut nodes, "top_k", "top_k", ["lm_head"]);
    push(&mut nodes, "sampling", "sampling", ["top_k"]);

    assert_eq!(nodes.len(), EXPECTED_GRAPH_NODE_COUNT);
    nodes
}

fn expected_canonical_bytes() -> Vec<u8> {
    let records: Vec<_> = expected_graph()
        .into_iter()
        .map(|node| {
            serde_json::json!({
                "id": node.id,
                "inputs": node.inputs,
                "op": node.op,
                "source_ids": node.source_ids,
            })
        })
        .collect();
    let mut bytes = serde_json::to_vec(&records).expect("serializing test fixture cannot fail");
    bytes.push(b'\n');
    bytes
}

#[test]
fn semantic_id_parser_enforces_the_stable_grammar() {
    for valid in [
        "vision.embeddings.patch",
        "vision.layer.00.norm1",
        "decoder.layer.17.mlp.down",
        "multimodal.inputs_embeds",
        "lm_head",
        "sampling",
    ] {
        let id = SemanticId::parse(valid).unwrap_or_else(|e| panic!("{valid}: {e}"));
        assert_eq!(id.as_str(), valid);
        assert_eq!(id.to_string(), valid);
    }

    for invalid in [
        "",
        ".vision",
        "vision.",
        "vision..layer",
        "Vision.layer",
        "vision-layer",
        "vision layer",
        "vision/layer",
        "vision.layer.0.norm1",
        "vision.layer.000.norm1",
        "_vision.layer",
    ] {
        assert!(SemanticId::parse(invalid).is_err(), "accepted {invalid:?}");
    }

    let too_long = format!("vision.{}", "a".repeat(122));
    assert!(too_long.len() > 128);
    assert!(SemanticId::parse(&too_long).is_err());
}

#[test]
fn canonical_model_graph_matches_the_independent_topology() {
    let graph = SemanticGraph::paddleocr_vl_16();
    graph.verify().expect("built-in graph must verify");
    let actual = graph.nodes();
    let expected = expected_graph();

    assert_eq!(actual.len(), EXPECTED_GRAPH_NODE_COUNT);
    assert_eq!(
        actual
            .iter()
            .map(|n| n.id.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        actual.len()
    );
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.id.as_str(), expected.id);
        assert_eq!(actual.op.stable_name(), expected.op, "{}", expected.id);
        assert_eq!(
            actual
                .inputs
                .iter()
                .map(SemanticId::as_str)
                .collect::<Vec<_>>(),
            expected.inputs,
            "{}",
            expected.id
        );
        assert_eq!(
            actual
                .source_ids
                .iter()
                .map(SemanticId::as_str)
                .collect::<Vec<_>>(),
            expected.source_ids,
            "{}",
            expected.id
        );
    }
}

#[test]
fn canonical_graph_snapshot_is_byte_reproducible() {
    let graph = SemanticGraph::paddleocr_vl_16();
    let actual = graph.canonical_bytes().expect("valid graph serializes");
    let expected = expected_canonical_bytes();

    assert_eq!(actual, expected);
    assert_eq!(actual.len(), EXPECTED_GRAPH_CANONICAL_LEN);
    assert_eq!(
        blake3::hash(&actual).to_hex().as_str(),
        EXPECTED_GRAPH_BLAKE3
    );
    assert_eq!(actual.last(), Some(&b'\n'));
}

#[test]
fn all_documented_semantic_op_families_are_present() {
    let actual: BTreeSet<_> = SemanticGraph::paddleocr_vl_16()
        .nodes()
        .iter()
        .map(|node| node.op.kind())
        .collect();
    let required = BTreeSet::from([
        SemanticOpKind::ImageSmartResize,
        SemanticOpKind::NormalizeRgb,
        SemanticOpKind::PatchProjection,
        SemanticOpKind::VisionPositionEmbedding,
        SemanticOpKind::VisionLayerNorm,
        SemanticOpKind::VisionQkv,
        SemanticOpKind::VisionRope,
        SemanticOpKind::VisionAttention,
        SemanticOpKind::VisionOutProjection,
        SemanticOpKind::VisionMlp,
        SemanticOpKind::ProjectorMerge2x2,
        SemanticOpKind::ProjectorMlp,
        SemanticOpKind::TokenEmbedding,
        SemanticOpKind::MultimodalAssemble,
        SemanticOpKind::MRopeIndex,
        SemanticOpKind::DecoderRmsNorm,
        SemanticOpKind::DecoderQkv,
        SemanticOpKind::DecoderMRope,
        SemanticOpKind::DecoderPrefillAttention,
        SemanticOpKind::DecoderKvAppend,
        SemanticOpKind::DecoderDecodeAttention,
        SemanticOpKind::DecoderSwiGlu,
        SemanticOpKind::FinalRmsNorm,
        SemanticOpKind::LmHead,
        SemanticOpKind::TopK,
        SemanticOpKind::Sampling,
    ]);
    assert!(
        required.is_subset(&actual),
        "missing: {:?}",
        required.difference(&actual).collect::<Vec<_>>()
    );
}

#[test]
fn selected_semantic_ids_and_operations_do_not_drift() {
    let graph = SemanticGraph::paddleocr_vl_16();
    let op = |id: &str| {
        graph
            .node(&SemanticId::parse(id).unwrap())
            .unwrap_or_else(|| panic!("missing {id}"))
            .op
    };

    assert_eq!(op("vision.layer.00.norm1"), SemanticOp::VisionLayerNorm);
    assert_eq!(op("vision.layer.26.mlp"), SemanticOp::VisionMlp);
    assert_eq!(op("projector.merge"), SemanticOp::ProjectorMerge2x2);
    assert_eq!(
        op("decoder.layer.17.attention.prefill"),
        SemanticOp::DecoderPrefillAttention
    );
    assert_eq!(
        op("decoder.layer.17.kv_append"),
        SemanticOp::DecoderKvAppend
    );
    assert_eq!(op("decoder.layer.17.mlp.down"), SemanticOp::DecoderSwiGlu);
    assert_eq!(op("lm_head"), SemanticOp::LmHead);
    assert_eq!(op("sampling"), SemanticOp::Sampling);
}

fn id(value: &str) -> SemanticId {
    SemanticId::parse(value).unwrap()
}

fn node(id_value: &str, inputs: &[&str]) -> SemanticNode {
    node_with_op(id_value, SemanticOp::ResidualAdd, inputs)
}

fn node_with_op(id_value: &str, op: SemanticOp, inputs: &[&str]) -> SemanticNode {
    SemanticNode {
        id: id(id_value),
        op,
        inputs: inputs.iter().map(|value| id(value)).collect(),
        source_ids: vec![id(id_value)],
    }
}

#[test]
fn verifier_rejects_duplicate_ids_dangling_edges_and_cycles() {
    let duplicate = SemanticGraph::from_nodes(vec![node("test.a", &[]), node("test.a", &[])]);
    assert_eq!(
        duplicate.verify().unwrap_err().code(),
        GraphErrorCode::DuplicateSemanticId
    );

    let dangling = SemanticGraph::from_nodes(vec![node("test.a", &["test.missing"])]);
    assert_eq!(
        dangling.verify().unwrap_err().code(),
        GraphErrorCode::DanglingInput
    );

    let cyclic = SemanticGraph::from_nodes(vec![
        node("test.a", &["test.b"]),
        node("test.b", &["test.a"]),
    ]);
    assert_eq!(cyclic.verify().unwrap_err().code(), GraphErrorCode::Cycle);
}

#[test]
fn verifier_rejects_duplicate_or_empty_source_provenance() {
    let empty = SemanticGraph::from_nodes(vec![SemanticNode {
        source_ids: vec![],
        ..node("test.a", &[])
    }]);
    assert_eq!(
        empty.verify().unwrap_err().code(),
        GraphErrorCode::EmptySourceIds
    );

    let duplicate = SemanticGraph::from_nodes(vec![SemanticNode {
        source_ids: vec![id("source.a"), id("source.a")],
        ..node("test.a", &[])
    }]);
    assert_eq!(
        duplicate.verify().unwrap_err().code(),
        GraphErrorCode::DuplicateSourceId
    );
}

#[test]
fn verifier_enforces_operation_arity_unique_inputs_and_input_kinds() {
    let representative_wrong_arities: [(SemanticOp, Vec<&str>); 10] = [
        (SemanticOp::ImageSmartResize, vec!["test.source"]),
        (SemanticOp::PatchProjection, vec![]),
        (SemanticOp::VisionAttention, vec!["test.source"]),
        (SemanticOp::ProjectorMerge2x2, vec![]),
        (SemanticOp::MultimodalAssemble, vec!["test.source"]),
        (SemanticOp::DecoderMRope, vec!["test.source"]),
        (SemanticOp::DecoderDecodeAttention, vec!["test.source"]),
        (SemanticOp::ResidualAdd, vec!["test.source"]),
        (SemanticOp::LmHead, vec![]),
        (SemanticOp::Sampling, vec![]),
    ];
    for (index, (op, inputs)) in representative_wrong_arities.into_iter().enumerate() {
        let target = format!("test.target_{index}");
        let mut nodes = vec![node_with_op("test.source", SemanticOp::TokenEmbedding, &[])];
        nodes.push(SemanticNode {
            id: id(&target),
            op,
            inputs: inputs.into_iter().map(id).collect(),
            source_ids: vec![id(&target)],
        });
        let graph = SemanticGraph::from_nodes(nodes);
        assert_eq!(
            graph.verify().unwrap_err().code(),
            GraphErrorCode::InvalidInputArity,
            "operation {}",
            op.stable_name()
        );
    }

    let duplicate_input = SemanticGraph::from_nodes(vec![
        node_with_op("test.input", SemanticOp::TokenEmbedding, &[]),
        node_with_op(
            "test.output",
            SemanticOp::ResidualAdd,
            &["test.input", "test.input"],
        ),
    ]);
    assert_eq!(
        duplicate_input.verify().unwrap_err().code(),
        GraphErrorCode::DuplicateInput
    );

    for (target_op, source_op) in [
        (SemanticOp::NormalizeRgb, SemanticOp::TokenEmbedding),
        (SemanticOp::TopK, SemanticOp::TokenEmbedding),
    ] {
        let wrong_input_kind = SemanticGraph::from_nodes(vec![
            node_with_op("test.source", source_op, &[]),
            node_with_op("test.target", target_op, &["test.source"]),
        ]);
        assert_eq!(
            wrong_input_kind.verify().unwrap_err().code(),
            GraphErrorCode::InvalidInputKind,
            "{} must reject {}",
            target_op.stable_name(),
            source_op.stable_name()
        );
    }
}

#[test]
fn verifier_prevents_two_nodes_from_claiming_the_same_source_semantics() {
    let shared = id("source.original");
    let graph = SemanticGraph::from_nodes(vec![
        SemanticNode {
            source_ids: vec![shared.clone()],
            ..node_with_op("test.a", SemanticOp::ImageSmartResize, &[])
        },
        SemanticNode {
            source_ids: vec![shared],
            ..node_with_op("test.b", SemanticOp::ImageSmartResize, &[])
        },
    ]);
    assert_eq!(
        graph.verify().unwrap_err().code(),
        GraphErrorCode::SourceIdClaimedByMultipleNodes
    );
}
