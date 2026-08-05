mod support;

use std::panic::{AssertUnwindSafe, catch_unwind};

use pvlc_ir::{
    PlanBindingResource, PlanConsumedNode, PlanDtype, PlanError, PlanErrorCode, PlanExternalValue,
    PlanIr, PlanRewriteProvenance, PlanTensorResource, SemanticGraph, SemanticNode, SemanticOp,
};
use pvlc_model_schema::{PaddleOcrVl16Schema, TensorDtype, TensorSpec};
use pvlc_passes::{
    VisionQkvFusionOptions, VisionQkvPassError, VisionQkvPassErrorCode, VisionQkvPassResult,
    VisionQkvPassStatus, fuse_vision_qkv, lower_vision_qkv_fragment,
};
use pvlc_runtime_core::{
    KernelId, VisionEncoderLayerGeometry, VisionEncoderLayerStage, plan_vision_qkv_fused_geometry,
};

use support::*;

const SEMANTIC_IR_CANONICAL_LEN: usize = 68_833;
const SEMANTIC_IR_BLAKE3: &str = "2b2556c363545dcef569e3e6d0db01967973a081706c8483e1c5af3c7dc5bf73";
const OTHER_PASS_ID: &str = "other-pass-v1";

fn options(enabled: bool, alignment: u32) -> VisionQkvFusionOptions {
    VisionQkvFusionOptions {
        enabled,
        target: limits(alignment),
    }
}

fn raw_bytes(plan: &PlanIr) -> Vec<u8> {
    serde_json::to_vec(plan).expect("PlanIR records must remain serializable even when invalid")
}

fn inject_unknown(canonical: &[u8], scope: &str, needle: &str) -> Vec<u8> {
    let canonical = std::str::from_utf8(canonical).unwrap();
    let position = canonical
        .find(needle)
        .unwrap_or_else(|| panic!("missing {scope} unknown-field injection point"))
        + needle.len();
    let mut mutant = canonical.to_owned();
    mutant.insert_str(position, "\"unknown\":true,");
    mutant.into_bytes()
}

fn assert_error_atomic_result(
    name: &str,
    plan: &PlanIr,
    graph: &SemanticGraph,
    expected: VisionQkvPassErrorCode,
) {
    let before = raw_bytes(plan);
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let result: Result<VisionQkvPassResult, VisionQkvPassError> =
            fuse_vision_qkv(plan, graph, options(true, 32));
        result
    }));
    assert_eq!(raw_bytes(plan), before, "{name}: input mutated on error");
    let result = outcome.unwrap_or_else(|_| {
        panic!("{name}: fusion panicked instead of returning stable {expected:?}")
    });
    match result {
        Err(error) => assert_eq!(error.code(), expected, "{name}: unexpected error: {error}"),
        Ok(result) => panic!(
            "{name}: candidate evidence returned {:?}, expected {expected:?}",
            result.status
        ),
    }
}

fn tensor(plan: &PlanIr, node_index: usize, binding_index: usize) -> &PlanTensorResource {
    let PlanBindingResource::Tensor(tensor) =
        &plan.nodes[node_index].bindings[binding_index].resource
    else {
        panic!("expected tensor binding {binding_index} on node {node_index}");
    };
    tensor
}

#[test]
fn independent_unfused_fixture_matches_lowering_and_exact_legacy_abi() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_catalog(0);
    let expected = compact_unfused_plan();
    expected
        .verify()
        .expect("hand-authored fixture must verify");
    assert_eq!(
        expected.requirements,
        exact_unfused_requirements(TOKENS, INPUT_WIDTH, OUTPUT_WIDTH)
    );

    let actual = lower_vision_qkv_fragment(&graph, 0, &geometry, &catalog)
        .expect("valid compact Q/K/V fragment must lower");
    assert_eq!(
        actual.canonical_bytes().unwrap(),
        expected.canonical_bytes().unwrap()
    );
    assert_eq!(actual.nodes.len(), 3);
    assert_eq!(actual.outputs, expected.outputs);

    for (index, role) in Role::ALL.into_iter().enumerate() {
        let node = &actual.nodes[index];
        assert_eq!(node.id.as_str(), projection_node_id(0, role));
        assert_eq!(node.invocation.kernel, KernelId::VisionPatchProjectionF32);
        assert_eq!(node.invocation.workgroup_size, [8, 8, 1]);
        assert_eq!(node.invocation.dispatch, [1, 1, 1]);
        assert_eq!(node.invocation.output_elements, 9);
        assert_eq!(node.invocation.output_bytes, PLANE_BYTES);
        assert_eq!(node.uniform_words, [TOKENS, INPUT_WIDTH, OUTPUT_WIDTH, 0]);
        assert_eq!(
            node.bindings
                .iter()
                .map(|binding| binding.number)
                .collect::<Vec<_>>(),
            (0..5).collect::<Vec<_>>()
        );
        assert_eq!(node.outputs.len(), 1);
        assert_eq!(node.outputs[0].id.as_str(), output_value(0, role));
        assert_eq!(node.outputs[0].shape, [3, 3]);
        assert_eq!(node.source_semantic_ids, [semantic_id(&semantic_source(0))]);
        assert!(node.rewrite_provenance.is_none());
        let expected_label = projection_node_id(0, role);
        assert_eq!(
            node.timestamp_label.as_deref(),
            Some(expected_label.as_str())
        );

        for (binding, suffix) in [(1, "weight"), (2, "bias")] {
            let tensor = tensor(&actual, index, binding);
            assert_eq!(tensor.physical_name, tensor_physical(0, role, suffix));
            assert_eq!(
                tensor.semantic_id.as_str(),
                tensor_semantic(0, role, suffix)
            );
            assert_eq!(tensor.dtype, PlanDtype::BFloat16);
            assert_eq!(tensor.storage_format, PlanDtype::Float32);
        }
    }
}

#[test]
fn unfused_projection_contract_distinguishes_32_tiles_from_the_old_8_tiles() {
    let graph = canonical_graph();
    let geometry = VisionEncoderLayerGeometry {
        tokens: 65,
        hidden_size: 1_152,
        attention_heads: 16,
        head_dim: 72,
        intermediate_size: 4_304,
        layer_norm_epsilon: 1.0e-5,
        cu_seqlens: &[0, 65],
    }
    .plan()
    .expect("official-width boundary geometry must plan");
    let catalog = PaddleOcrVl16Schema::tensor_specs();

    let unfused = lower_vision_qkv_fragment(&graph, 0, &geometry, &catalog)
        .expect("production lowering must accept 32x32 tiled projection dispatch");
    assert_eq!(unfused.nodes.len(), 3);
    for node in &unfused.nodes {
        assert_eq!(node.invocation.kernel, KernelId::VisionPatchProjectionF32);
        assert_eq!(node.invocation.workgroup_size, [8, 8, 1]);
        assert_eq!(
            node.invocation.dispatch,
            [36, 3, 1],
            "1,152 columns by 65 tokens require 36x3 shared 32x32 tiles",
        );
    }
    assert_eq!(unfused.requirements.max_dispatch, [36, 3, 1]);

    let result = fuse_vision_qkv(&unfused, &graph, options(true, 32))
        .expect("production fusion must accept the tiled unfused plan");
    assert_eq!(result.status, VisionQkvPassStatus::Fused);
    assert_eq!(result.plan.nodes.len(), 1);
    let fused = &result.plan.nodes[0];
    assert_eq!(fused.invocation.kernel, KernelId::VisionQkvFusedF32);
    assert_eq!(fused.invocation.workgroup_size, [8, 8, 1]);
    assert_eq!(
        fused.invocation.dispatch,
        [144, 9, 3],
        "the distinct fused QKV kernel must retain its existing 8x8x3 topology",
    );
}

#[test]
fn lowering_resolves_every_layer_from_a_shuffled_catalog_deterministically() {
    let graph = canonical_graph();
    let geometry = official_layer_plan();
    let canonical_catalog = PaddleOcrVl16Schema::tensor_specs();
    let mut shuffled = canonical_catalog.clone();
    shuffled.reverse();
    shuffled.rotate_left(137);

    for layer in 0..27 {
        let expected = unfused_plan_for(layer, 1, 1_152, 1_152);
        let canonical = lower_vision_qkv_fragment(&graph, layer, &geometry, &canonical_catalog)
            .unwrap_or_else(|error| panic!("layer {layer:02} canonical lowering failed: {error}"));
        let shuffled_once = lower_vision_qkv_fragment(&graph, layer, &geometry, &shuffled)
            .unwrap_or_else(|error| panic!("layer {layer:02} shuffled lowering failed: {error}"));
        let shuffled_twice = lower_vision_qkv_fragment(&graph, layer, &geometry, &shuffled)
            .unwrap_or_else(|error| panic!("layer {layer:02} repeat lowering failed: {error}"));

        let expected_bytes = expected.canonical_bytes().unwrap();
        assert_eq!(
            canonical.canonical_bytes().unwrap(),
            expected_bytes,
            "layer {layer:02}"
        );
        assert_eq!(
            shuffled_once.canonical_bytes().unwrap(),
            expected_bytes,
            "layer {layer:02}"
        );
        assert_eq!(
            shuffled_twice.canonical_bytes().unwrap(),
            expected_bytes,
            "layer {layer:02}"
        );
        for (index, role) in Role::ALL.into_iter().enumerate() {
            assert_eq!(
                canonical.nodes[index].id.as_str(),
                projection_node_id(layer, role)
            );
            assert_eq!(
                tensor(&canonical, index, 1).semantic_id.as_str(),
                tensor_semantic(layer, role, "weight")
            );
            assert_eq!(
                tensor(&canonical, index, 2).semantic_id.as_str(),
                tensor_semantic(layer, role, "bias")
            );
        }
    }
}

#[test]
fn independent_fused_arithmetic_matches_literal_oracle_and_m7c1a_only_as_cross_check() {
    let graph = canonical_graph();
    let input = compact_unfused_plan();

    for alignment in [32, 256] {
        let independent = independent_layout(alignment);
        assert_eq!(independent.plane_elements, 9);
        assert_eq!(independent.plane_bytes, 36);
        assert_eq!(
            independent.plane_stride_bytes,
            if alignment == 32 { 64 } else { 256 }
        );
        assert_eq!(
            independent.physical_bytes,
            if alignment == 32 { 192 } else { 768 }
        );
        assert_eq!(
            independent.offsets,
            if alignment == 32 {
                [0, 64, 128]
            } else {
                [0, 256, 512]
            }
        );

        let expected = expected_fused_plan(alignment);
        let result = fuse_vision_qkv(&input, &graph, options(true, alignment)).unwrap();
        assert_eq!(result.status, VisionQkvPassStatus::Fused);
        assert_eq!(result.plan, expected);
        assert_eq!(
            result.plan.requirements,
            exact_fused_requirements(alignment)
        );
        assert_eq!(result.plan.requirements.storage_binding_count, 8);
        assert_eq!(result.plan.requirements.uniform_binding_count, 1);
        assert_eq!(
            result
                .plan
                .requirements
                .required_storage_buffer_offset_alignment,
            u64::from(alignment)
        );
        assert_eq!(
            result.plan.requirements.largest_storage_binding_bytes,
            independent.physical_bytes
        );
        assert_eq!(
            result.plan.requirements.largest_buffer_bytes,
            independent.physical_bytes
        );
        assert_eq!(result.plan.requirements.max_workgroup_size, [8, 8, 1]);
        assert_eq!(result.plan.requirements.max_dispatch, [1, 1, 3]);
        assert!(result.plan.requirements.required_features.is_empty());
        assert_eq!(result.plan.nodes.len(), 1);
        let fused = &result.plan.nodes[0];
        assert_eq!(fused.invocation.kernel, KernelId::VisionQkvFusedF32);
        assert_eq!(fused.invocation.workgroup_size, [8, 8, 1]);
        assert_eq!(fused.invocation.dispatch, independent.dispatch);
        assert_eq!(fused.invocation.output_bytes, independent.physical_bytes);
        assert_eq!(fused.uniform_words, independent.uniform_words);
        assert_eq!(
            fused
                .bindings
                .iter()
                .map(|binding| binding.number)
                .collect::<Vec<_>>(),
            (0..9).collect::<Vec<_>>()
        );
        assert_eq!(
            fused
                .outputs
                .iter()
                .map(|output| output.id.clone())
                .collect::<Vec<_>>(),
            input.outputs
        );
        assert_eq!(
            fused
                .outputs
                .iter()
                .map(|output| output.byte_offset)
                .collect::<Vec<_>>(),
            independent.offsets
        );
        assert!(
            fused
                .outputs
                .iter()
                .all(|output| output.byte_length == PLANE_BYTES)
        );
        assert_eq!(fused.timestamp_label, None);

        let accepted =
            plan_vision_qkv_fused_geometry(TOKENS, INPUT_WIDTH, OUTPUT_WIDTH, limits(alignment))
                .expect("accepted M7c1a planner must accept the independent compact geometry");
        assert_eq!(
            accepted.output_layout.plane_elements,
            independent.plane_elements
        );
        assert_eq!(accepted.output_layout.plane_bytes, independent.plane_bytes);
        assert_eq!(
            accepted.output_layout.plane_stride_bytes,
            independent.plane_stride_bytes
        );
        assert_eq!(
            accepted.output_layout.physical_bytes,
            independent.physical_bytes
        );
        assert_eq!(
            [
                accepted.output_layout.query.offset,
                accepted.output_layout.key.offset,
                accepted.output_layout.value.offset,
            ],
            independent.offsets
        );
        assert_eq!(accepted.uniform_words, independent.uniform_words);
        assert_eq!(accepted.invocation.dispatch, independent.dispatch);

        let larger = fuse_vision_qkv(
            &input,
            &graph,
            VisionQkvFusionOptions {
                enabled: true,
                target: larger_limits(alignment),
            },
        )
        .unwrap();
        assert_eq!(
            larger.plan.canonical_bytes().unwrap(),
            result.plan.canonical_bytes().unwrap(),
            "permissive adapter maxima must not enter canonical PlanIR"
        );
    }
}

#[test]
fn fused_canonical_bytes_hash_and_complete_consumed_snapshots_are_frozen() {
    let graph = canonical_graph();
    let unfused = compact_unfused_plan();
    let result = fuse_vision_qkv(&unfused, &graph, options(true, 32)).unwrap();
    let bytes = result.plan.canonical_bytes().unwrap();
    assert_eq!(bytes, EXPECTED_FUSED_CANONICAL);
    assert_eq!(bytes.len(), EXPECTED_FUSED_CANONICAL_LEN);
    assert_eq!(bytes.last(), Some(&b'\n'));
    assert_eq!(
        blake3::hash(&bytes).to_hex().as_str(),
        EXPECTED_FUSED_BLAKE3
    );
    assert_eq!(
        result.plan.canonical_blake3_hex().unwrap(),
        EXPECTED_FUSED_BLAKE3
    );

    let parsed = PlanIr::parse_canonical(EXPECTED_FUSED_CANONICAL).unwrap();
    assert_eq!(parsed.canonical_bytes().unwrap(), EXPECTED_FUSED_CANONICAL);
    assert_eq!(parsed, result.plan);

    let provenance = parsed.nodes[0].rewrite_provenance.as_ref().unwrap();
    assert_eq!(provenance.pass_id, PASS_ID);
    assert_eq!(
        provenance.source_semantic_ids,
        [semantic_id(&semantic_source(0))]
    );
    assert_eq!(provenance.consumed.len(), 3);
    for ((consumed, original), role) in provenance
        .consumed
        .iter()
        .zip(&unfused.nodes)
        .zip(Role::ALL)
    {
        let consumed: &PlanConsumedNode = consumed;
        assert_eq!(consumed.role, role.name());
        assert_eq!(consumed.original, snapshot(original));
        let original_bytes = consumed.original.canonical_node_bytes().unwrap();
        assert_eq!(
            blake3::hash(&original_bytes).to_hex().as_str(),
            consumed.canonical_blake3
        );
        assert_eq!(
            consumed.original.canonical_node_blake3_hex().unwrap(),
            consumed.canonical_blake3
        );
    }
}

#[test]
fn sentinels_outcomes_and_idempotence_preserve_unrelated_bytes() {
    let graph = canonical_graph();
    let input = compact_unfused_with_sentinels();
    let input_bytes = input.canonical_bytes().unwrap();
    let before_sentinel = input.nodes[0].canonical_bytes().unwrap();
    let after_sentinel = input.nodes[4].canonical_bytes().unwrap();

    let disabled = fuse_vision_qkv(&input, &graph, options(false, 32)).unwrap();
    assert_eq!(disabled.status, VisionQkvPassStatus::UnchangedDisabled);
    assert_eq!(disabled.plan.canonical_bytes().unwrap(), input_bytes);

    let unrelated = unrelated_plan();
    let unrelated_bytes = unrelated.canonical_bytes().unwrap();
    let no_match = fuse_vision_qkv(&unrelated, &graph, options(true, 32)).unwrap();
    assert_eq!(no_match.status, VisionQkvPassStatus::UnchangedNoMatch);
    assert_eq!(no_match.plan.canonical_bytes().unwrap(), unrelated_bytes);

    let first = fuse_vision_qkv(&input, &graph, options(true, 32)).unwrap();
    assert_eq!(first.status, VisionQkvPassStatus::Fused);
    assert_eq!(first.plan.nodes.len(), 3);
    assert_eq!(
        first.plan.nodes[0].canonical_bytes().unwrap(),
        before_sentinel
    );
    assert_eq!(
        first.plan.nodes[2].canonical_bytes().unwrap(),
        after_sentinel
    );
    assert_eq!(first.plan.outputs[0], input.outputs[0]);
    assert_eq!(first.plan.outputs[4], input.outputs[4]);

    let first_bytes = first.plan.canonical_bytes().unwrap();
    let second = fuse_vision_qkv(&first.plan, &graph, options(true, 32)).unwrap();
    assert_eq!(second.status, VisionQkvPassStatus::UnchangedAlreadyFused);
    assert_eq!(second.plan.canonical_bytes().unwrap(), first_bytes);

    let already = expected_fused_plan(32);
    let already_bytes = already.canonical_bytes().unwrap();
    let unchanged = fuse_vision_qkv(&already, &graph, options(true, 32)).unwrap();
    assert_eq!(unchanged.status, VisionQkvPassStatus::UnchangedAlreadyFused);
    assert_eq!(unchanged.plan.canonical_bytes().unwrap(), already_bytes);
}

#[test]
fn duplicate_candidate_exports_have_stable_generic_and_pass_errors() {
    let graph = canonical_graph();
    let mut outcomes = Vec::new();
    for role in Role::ALL {
        let mut duplicate = compact_unfused_plan();
        duplicate.outputs.push(value_id(&output_value(0, role)));
        let before = raw_bytes(&duplicate);
        let generic_code = duplicate
            .verify()
            .err()
            .map(|error| format!("{:?}", error.code()));
        let pass_outcome = catch_unwind(AssertUnwindSafe(|| {
            fuse_vision_qkv(&duplicate, &graph, options(true, 32))
        }));
        assert_eq!(
            raw_bytes(&duplicate),
            before,
            "duplicate {} export input mutated",
            role.name()
        );
        let pass_code = pass_outcome
            .unwrap_or_else(|_| panic!("duplicate {} export validation panicked", role.name()))
            .err()
            .map(|error| error.code());
        outcomes.push((role.name(), generic_code, pass_code));
    }
    assert_eq!(
        outcomes,
        Role::ALL
            .map(|role| {
                (
                    role.name(),
                    Some("DuplicatePlanOutput".to_owned()),
                    Some(VisionQkvPassErrorCode::InvalidPlan),
                )
            })
            .to_vec(),
        "duplicate Q/K/V exports need stable generic and pass error classes"
    );
}

#[test]
fn reordered_candidate_exports_fail_closed_atomically() {
    let graph = canonical_graph();
    let mut reordered = compact_unfused_with_sentinels();
    let query = reordered
        .outputs
        .iter()
        .position(|output| output.as_str() == output_value(0, Role::Query))
        .unwrap();
    let key = reordered
        .outputs
        .iter()
        .position(|output| output.as_str() == output_value(0, Role::Key))
        .unwrap();
    reordered.outputs.swap(query, key);
    reordered
        .verify()
        .expect("Q/K/V export order is a pass-level semantic constraint");
    assert_error_atomic_result(
        "reordered Q/K/V exports",
        &reordered,
        &graph,
        VisionQkvPassErrorCode::IllegalCandidateDataflow,
    );
}

#[test]
fn unrelated_exports_may_interleave_without_changing_qkv_relative_order() {
    let graph = canonical_graph();
    let mut input = compact_unfused_with_sentinels();
    let unrelated_after = input.outputs.remove(4);
    input.outputs.insert(2, unrelated_after);
    input
        .verify()
        .expect("interleaved unrelated exports must remain structural PlanIR");
    let expected_outputs = input.outputs.clone();

    let result = fuse_vision_qkv(&input, &graph, options(true, 32)).unwrap();
    assert_eq!(result.status, VisionQkvPassStatus::Fused);
    assert_eq!(result.plan.outputs, expected_outputs);
    let qkv_positions = Role::ALL.map(|role| {
        result
            .plan
            .outputs
            .iter()
            .position(|output| output.as_str() == output_value(0, role))
            .unwrap()
    });
    assert!(qkv_positions[0] < qkv_positions[1] && qkv_positions[1] < qkv_positions[2]);
}

#[test]
fn already_fused_missing_export_is_rejected_atomically() {
    let graph = canonical_graph();
    let mut outcomes = Vec::new();
    for role in Role::ALL {
        let mut missing_export = expected_fused_plan(32);
        let index = missing_export
            .outputs
            .iter()
            .position(|output| output.as_str() == output_value(0, role))
            .unwrap();
        missing_export.outputs.remove(index);
        missing_export
            .verify()
            .unwrap_or_else(|error| panic!("missing {} export must verify: {error}", role.name()));
        let before = raw_bytes(&missing_export);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            fuse_vision_qkv(&missing_export, &graph, options(true, 32))
        }));
        assert_eq!(
            raw_bytes(&missing_export),
            before,
            "missing {} export input mutated",
            role.name()
        );
        let code = outcome
            .unwrap_or_else(|_| panic!("missing {} export validation panicked", role.name()))
            .err()
            .map(|error| error.code());
        outcomes.push((role.name(), code));
    }
    assert_eq!(
        outcomes,
        Role::ALL
            .map(|role| {
                (
                    role.name(),
                    Some(VisionQkvPassErrorCode::IllegalCandidateDataflow),
                )
            })
            .to_vec()
    );
}

#[test]
fn already_fused_live_consumer_is_rejected_atomically() {
    let graph = canonical_graph();
    let mut consumed = expected_fused_plan(32);
    let (unused_external, mut consumer, consumer_output) =
        sentinel_node("fused_consumer", "postprocess.consumer");
    consumer.bindings[0].resource =
        PlanBindingResource::Value(value_id(&output_value(0, Role::Value)));
    consumed.external_values.push(unused_external);
    consumed.nodes.push(consumer);
    consumed.outputs.push(consumer_output);
    refresh_requirements(&mut consumed, 32);
    consumed
        .verify()
        .expect("an unrelated live consumer must remain structural PlanIR");
    assert_error_atomic_result(
        "already-fused internal value consumer",
        &consumed,
        &graph,
        VisionQkvPassErrorCode::IllegalCandidateDataflow,
    );
}

fn partial(plan: &mut PlanIr) {
    plan.nodes.pop();
    plan.outputs.pop();
    refresh_requirements(plan, 4);
}

fn duplicate_role(plan: &mut PlanIr) {
    let mut duplicate = plan.nodes[0].clone();
    duplicate.id = node_id("vision.layer.00.query_duplicate");
    duplicate.outputs[0].id = value_id("vision.layer.00.query_duplicate");
    duplicate.outputs[0].buffer_id = buffer_id("output.vision.layer.00.query_duplicate");
    output_buffer_mut(&mut duplicate).buffer_id =
        buffer_id("output.vision.layer.00.query_duplicate");
    plan.outputs.push(duplicate.outputs[0].id.clone());
    plan.nodes.push(duplicate);
    refresh_requirements(plan, 4);
}

fn reordered(plan: &mut PlanIr) {
    plan.nodes.swap(0, 1);
}

fn noncontiguous(plan: &mut PlanIr) {
    let (external, node, output) = sentinel_node("between", "preprocess.resize");
    plan.external_values.push(external);
    plan.nodes.insert(1, node);
    plan.outputs.push(output);
    refresh_requirements(plan, 4);
}

fn ambiguous(plan: &mut PlanIr) {
    append_cluster(plan, 1);
}

fn mixed(plan: &mut PlanIr) {
    *plan = expected_fused_plan(32);
    append_cluster(plan, 1);
    refresh_requirements(plan, 32);
}

fn source_evidence_only(plan: &mut PlanIr) {
    plan.nodes[0].source_semantic_ids = vec![semantic_id("vision.layer.00.qkv")];
}

fn role_evidence_only(plan: &mut PlanIr) {
    plan.nodes[0].id = node_id("vision.layer.00.query");
}

fn tensor_evidence_only(plan: &mut PlanIr) {
    plan.nodes[0].bindings[0].resource =
        compact_unfused_plan().nodes[0].bindings[1].resource.clone();
    refresh_requirements(plan, 4);
}

fn fused_kernel_evidence_only(plan: &mut PlanIr) {
    plan.nodes[0].invocation.kernel = KernelId::VisionQkvFusedF32;
}

#[test]
fn partial_duplicate_reordered_noncontiguous_ambiguous_and_mixed_candidates_error() {
    let graph = canonical_graph();
    let cases: [(&str, fn(&mut PlanIr), VisionQkvPassErrorCode); 6] = [
        (
            "partial",
            partial,
            VisionQkvPassErrorCode::IncompleteCandidate,
        ),
        (
            "duplicate role",
            duplicate_role,
            VisionQkvPassErrorCode::DuplicateCandidateRole,
        ),
        (
            "reordered",
            reordered,
            VisionQkvPassErrorCode::NonCanonicalCandidateOrder,
        ),
        (
            "noncontiguous",
            noncontiguous,
            VisionQkvPassErrorCode::NonContiguousCandidate,
        ),
        (
            "ambiguous",
            ambiguous,
            VisionQkvPassErrorCode::AmbiguousCandidate,
        ),
        ("mixed", mixed, VisionQkvPassErrorCode::MixedCandidate),
    ];
    for (name, mutate, expected) in cases {
        let mut mutant = compact_unfused_plan();
        mutate(&mut mutant);
        assert_error_atomic_result(name, &mutant, &graph, expected);
    }
}

#[test]
fn every_isolated_candidate_signal_errors_instead_of_reporting_no_match() {
    let graph = canonical_graph();
    let cases: [(&str, fn(&mut PlanIr)); 4] = [
        ("canonical source", source_evidence_only),
        ("legacy role", role_evidence_only),
        ("tensor identity", tensor_evidence_only),
        ("fused kernel", fused_kernel_evidence_only),
    ];
    for (name, mutate) in cases {
        let mut mutant = unrelated_plan();
        mutate(&mut mutant);
        assert_error_atomic_result(
            name,
            &mutant,
            &graph,
            VisionQkvPassErrorCode::MalformedCandidate,
        );
    }
}

fn conflicting_tensor_evidence(plan: &mut PlanIr) {
    let canonical = compact_unfused_plan();
    let query_weight = canonical.nodes[0].bindings[1].clone();
    let key_weight = canonical.nodes[1].bindings[1].clone();
    plan.nodes[0].bindings.insert(1, query_weight);
    plan.nodes[0].bindings.insert(2, key_weight);
    for (number, binding) in plan.nodes[0].bindings.iter_mut().enumerate() {
        binding.number = u32::try_from(number).unwrap();
    }
    refresh_requirements(plan, 4);
}

fn rewrite_provenance_evidence(plan: &mut PlanIr) {
    let original = snapshot(&compact_unfused_plan().nodes[0]);
    let canonical_blake3 = original.canonical_node_blake3_hex().unwrap();
    let source_semantic_ids = plan.nodes[0].source_semantic_ids.clone();
    plan.nodes[0].rewrite_provenance = Some(PlanRewriteProvenance {
        pass_id: PASS_ID.to_owned(),
        source_semantic_ids,
        consumed: vec![PlanConsumedNode {
            role: Role::Query.name().to_owned(),
            original,
            canonical_blake3,
        }],
    });
}

fn attach_unrelated_other_pass_provenance(node: &mut pvlc_ir::PlanNode) {
    let original = snapshot(node);
    let original_bytes = original.canonical_node_bytes().unwrap();
    let original_text = std::str::from_utf8(&original_bytes).unwrap();
    assert!(!original_text.contains("vision.layer"));
    assert!(!original_text.contains("qkv"));
    let canonical_blake3 = original.canonical_node_blake3_hex().unwrap();
    let source_semantic_ids = node.source_semantic_ids.clone();
    node.diagnostic_label.push_str(".other_pass");
    node.rewrite_provenance = Some(PlanRewriteProvenance {
        pass_id: OTHER_PASS_ID.to_owned(),
        source_semantic_ids,
        consumed: vec![PlanConsumedNode {
            role: "sentinel_input".to_owned(),
            original,
            canonical_blake3,
        }],
    });
}

#[test]
fn unrelated_other_pass_provenance_is_no_match_and_byte_stable() {
    let graph = canonical_graph();
    let mut plan = unrelated_plan();
    plan.verify()
        .expect("the original sentinel must be structurally valid");
    attach_unrelated_other_pass_provenance(&mut plan.nodes[0]);
    plan.verify()
        .expect("self-consistent foreign provenance must remain valid PlanIR");
    let provenance = plan.nodes[0].rewrite_provenance.as_ref().unwrap();
    assert_eq!(provenance.pass_id, OTHER_PASS_ID);
    assert_eq!(
        provenance.source_semantic_ids,
        plan.nodes[0].source_semantic_ids
    );
    assert_eq!(provenance.consumed.len(), 1);
    assert!(!provenance.consumed[0].original.outputs.is_empty());

    let before_bytes = plan.canonical_bytes().unwrap();
    let before_hash = plan.canonical_blake3_hex().unwrap();
    let result = fuse_vision_qkv(&plan, &graph, options(true, 32)).unwrap_or_else(|error| {
        panic!("genuinely unrelated foreign provenance must not block this pass: {error}")
    });
    assert_eq!(result.status, VisionQkvPassStatus::UnchangedNoMatch);
    assert_eq!(result.plan.canonical_bytes().unwrap(), before_bytes);
    assert_eq!(result.plan.canonical_blake3_hex().unwrap(), before_hash);
    assert_eq!(plan.canonical_bytes().unwrap(), before_bytes);
    assert_eq!(plan.canonical_blake3_hex().unwrap(), before_hash);
}

#[test]
fn fusion_preserves_an_unrelated_rewritten_sentinel_byte_for_byte() {
    let graph = canonical_graph();
    let mut input = compact_unfused_with_sentinels();
    input
        .verify()
        .expect("the Q/K/V fragment and sentinels must start structurally valid");
    attach_unrelated_other_pass_provenance(&mut input.nodes[0]);
    input
        .verify()
        .expect("foreign sentinel provenance must compose with the Q/K/V fragment");

    let rewritten_sentinel_id = input.nodes[0].id.clone();
    let rewritten_sentinel_bytes = input.nodes[0].canonical_bytes().unwrap();
    let trailing_sentinel_id = input.nodes[4].id.clone();
    let trailing_sentinel_bytes = input.nodes[4].canonical_bytes().unwrap();
    let expected_outputs = input.outputs.clone();

    let result = fuse_vision_qkv(&input, &graph, options(true, 32)).unwrap_or_else(|error| {
        panic!("unrelated foreign provenance must not block Q/K/V fusion: {error}")
    });
    assert_eq!(result.status, VisionQkvPassStatus::Fused);
    assert_eq!(result.plan.nodes.len(), 3);
    assert_eq!(result.plan.nodes[0].id, rewritten_sentinel_id);
    assert_eq!(
        result.plan.nodes[0].canonical_bytes().unwrap(),
        rewritten_sentinel_bytes
    );
    assert_eq!(
        result.plan.nodes[0]
            .rewrite_provenance
            .as_ref()
            .unwrap()
            .pass_id,
        OTHER_PASS_ID
    );
    assert_eq!(
        result.plan.nodes[1].invocation.kernel,
        KernelId::VisionQkvFusedF32
    );
    assert_eq!(result.plan.nodes[2].id, trailing_sentinel_id);
    assert_eq!(
        result.plan.nodes[2].canonical_bytes().unwrap(),
        trailing_sentinel_bytes
    );
    assert_eq!(result.plan.outputs, expected_outputs);
}

#[test]
fn conflicting_tensor_evidence_fails_closed() {
    let graph = canonical_graph();

    let mut conflicting = unrelated_plan();
    conflicting_tensor_evidence(&mut conflicting);
    conflicting
        .verify()
        .expect("conflicting Q/K tensor identities must remain structural PlanIR");
    assert_error_atomic_result(
        "conflicting Q/K tensor evidence",
        &conflicting,
        &graph,
        VisionQkvPassErrorCode::MalformedCandidate,
    );
}

#[test]
fn isolated_qkv_output_id_evidence_fails_closed() {
    let graph = canonical_graph();
    let mut outcomes = Vec::new();
    for role in Role::ALL {
        let mut output_evidence = unrelated_plan();
        let canonical_output = value_id(&output_value(0, role));
        output_evidence.nodes[0].outputs[0].id = canonical_output.clone();
        output_evidence.outputs[0] = canonical_output;
        output_evidence
            .verify()
            .unwrap_or_else(|error| panic!("{} output evidence must verify: {error}", role.name()));
        let before = raw_bytes(&output_evidence);
        let result = fuse_vision_qkv(&output_evidence, &graph, options(true, 32));
        assert_eq!(raw_bytes(&output_evidence), before, "{}", role.name());
        outcomes.push((role.name(), result.err().map(|error| error.code())));
    }
    assert_eq!(
        outcomes,
        Role::ALL
            .map(|role| {
                (
                    role.name(),
                    Some(VisionQkvPassErrorCode::MalformedCandidate),
                )
            })
            .to_vec()
    );
}

#[test]
fn isolated_rewrite_provenance_evidence_fails_closed() {
    let graph = canonical_graph();
    let mut provenance_evidence = unrelated_plan();
    rewrite_provenance_evidence(&mut provenance_evidence);
    provenance_evidence
        .verify()
        .expect("isolated canonical rewrite provenance must remain structural PlanIR");
    assert_error_atomic_result(
        "isolated rewrite provenance evidence",
        &provenance_evidence,
        &graph,
        VisionQkvPassErrorCode::InvalidProvenance,
    );
}

fn wrong_kernel(plan: &mut PlanIr) {
    plan.nodes[0].invocation.kernel = KernelId::AddF32;
}

fn wrong_workgroup(plan: &mut PlanIr) {
    plan.nodes[0].invocation.workgroup_size = [4, 8, 1];
}

fn wrong_dispatch(plan: &mut PlanIr) {
    plan.nodes[0].invocation.dispatch = [2, 1, 1];
}

fn wrong_uniform(plan: &mut PlanIr) {
    plan.nodes[0].uniform_words[2] = 4;
    uniform_mut(&mut plan.nodes[0]).words[2] = 4;
}

fn wrong_dimension(plan: &mut PlanIr) {
    let node = &mut plan.nodes[0];
    node.invocation.output_elements = 6;
    node.invocation.output_bytes = 24;
    node.uniform_words[2] = 2;
    uniform_mut(node).words[2] = 2;
    tensor_mut(node, 1).shape = vec![2, 3];
    tensor_mut(node, 1).byte_length = 24;
    tensor_mut(node, 2).shape = vec![2];
    tensor_mut(node, 2).byte_length = 8;
    output_buffer_mut(node).byte_length = 24;
    node.outputs[0].shape = vec![3, 2];
    node.outputs[0].byte_length = 24;
    refresh_requirements(plan, 4);
}

fn wrong_common_input(plan: &mut PlanIr) {
    plan.external_values.push(PlanExternalValue {
        id: value_id("vision.layer.00.norm1_alternate"),
        dtype: PlanDtype::Float32,
        shape: vec![3, 3],
        buffer_id: buffer_id("activation.vision.layer.00.norm1_alternate"),
        byte_offset: 0,
        byte_length: 36,
    });
    plan.nodes[0].bindings[0].resource =
        PlanBindingResource::Value(value_id("vision.layer.00.norm1_alternate"));
    refresh_requirements(plan, 4);
}

#[test]
fn legacy_kernel_workgroup_dispatch_uniform_dimension_and_common_input_are_exact() {
    let graph = canonical_graph();
    let cases: [(&str, fn(&mut PlanIr)); 6] = [
        ("kernel", wrong_kernel),
        ("workgroup", wrong_workgroup),
        ("dispatch", wrong_dispatch),
        ("uniform", wrong_uniform),
        ("dimension", wrong_dimension),
        ("common input", wrong_common_input),
    ];
    for (name, mutate) in cases {
        let mut mutant = compact_unfused_plan();
        mutate(&mut mutant);
        assert_error_atomic_result(
            name,
            &mutant,
            &graph,
            VisionQkvPassErrorCode::LegacyAbiMismatch,
        );
    }
}

fn swapped_roles(plan: &mut PlanIr) {
    let (query, key_and_value) = plan.nodes.split_at_mut(1);
    std::mem::swap(
        &mut query[0].bindings[1].resource,
        &mut key_and_value[0].bindings[1].resource,
    );
}

fn swapped_bias_roles(plan: &mut PlanIr) {
    let (query, key_and_value) = plan.nodes.split_at_mut(1);
    std::mem::swap(
        &mut query[0].bindings[2].resource,
        &mut key_and_value[0].bindings[2].resource,
    );
}

fn missing_tensor(plan: &mut PlanIr) {
    plan.nodes[0].bindings.remove(1);
    for (number, binding) in plan.nodes[0].bindings.iter_mut().enumerate() {
        binding.number = u32::try_from(number).unwrap();
    }
    refresh_requirements(plan, 4);
}

fn missing_bias(plan: &mut PlanIr) {
    plan.nodes[0].bindings.remove(2);
    for (number, binding) in plan.nodes[0].bindings.iter_mut().enumerate() {
        binding.number = u32::try_from(number).unwrap();
    }
    refresh_requirements(plan, 4);
}

fn duplicate_tensor(plan: &mut PlanIr) {
    let duplicate = plan.nodes[0].bindings[1].clone();
    plan.nodes[0].bindings.insert(2, duplicate);
    for (number, binding) in plan.nodes[0].bindings.iter_mut().enumerate() {
        binding.number = u32::try_from(number).unwrap();
    }
    refresh_requirements(plan, 4);
}

fn duplicate_bias(plan: &mut PlanIr) {
    let duplicate = plan.nodes[0].bindings[2].clone();
    plan.nodes[0].bindings.insert(3, duplicate);
    for (number, binding) in plan.nodes[0].bindings.iter_mut().enumerate() {
        binding.number = u32::try_from(number).unwrap();
    }
    refresh_requirements(plan, 4);
}

fn wrong_checkpoint_dtype(plan: &mut PlanIr) {
    tensor_mut(&mut plan.nodes[0], 1).dtype = PlanDtype::Float16;
}

fn wrong_storage_format(plan: &mut PlanIr) {
    tensor_mut(&mut plan.nodes[0], 1).storage_format = PlanDtype::BFloat16;
}

fn wrong_tensor_shape(plan: &mut PlanIr) {
    tensor_mut(&mut plan.nodes[0], 1).shape = vec![3, 4];
}

fn wrong_physical_name(plan: &mut PlanIr) {
    tensor_mut(&mut plan.nodes[0], 1)
        .physical_name
        .push_str(".drift");
}

fn wrong_tensor_semantic(plan: &mut PlanIr) {
    tensor_mut(&mut plan.nodes[0], 1).semantic_id =
        semantic_id("vision.layer.00.attention.k.weight");
}

fn wrong_weight_buffer(plan: &mut PlanIr) {
    tensor_mut(&mut plan.nodes[0], 1).buffer_id =
        buffer_id("tensor.vision.layer.00.q_weight_drift");
    refresh_requirements(plan, 4);
}

fn wrong_bias_buffer(plan: &mut PlanIr) {
    tensor_mut(&mut plan.nodes[0], 2).buffer_id = buffer_id("tensor.vision.layer.00.q_bias_drift");
    refresh_requirements(plan, 4);
}

fn wrong_weight_offset(plan: &mut PlanIr) {
    tensor_mut(&mut plan.nodes[0], 1).byte_offset = 4;
    refresh_requirements(plan, 4);
}

fn wrong_bias_offset(plan: &mut PlanIr) {
    tensor_mut(&mut plan.nodes[0], 2).byte_offset = 4;
    refresh_requirements(plan, 4);
}

fn wrong_weight_length(plan: &mut PlanIr) {
    tensor_mut(&mut plan.nodes[0], 1).byte_length += 4;
    refresh_requirements(plan, 4);
}

fn wrong_bias_length(plan: &mut PlanIr) {
    tensor_mut(&mut plan.nodes[0], 2).byte_length += 4;
    refresh_requirements(plan, 4);
}

#[test]
fn tensor_roles_presence_identity_checkpoint_dtype_and_storage_format_fail_closed() {
    let graph = canonical_graph();
    let cases: [(&str, fn(&mut PlanIr)); 11] = [
        ("weight role swap", swapped_roles),
        ("bias role swap", swapped_bias_roles),
        ("missing weight", missing_tensor),
        ("missing bias", missing_bias),
        ("duplicate weight", duplicate_tensor),
        ("duplicate bias", duplicate_bias),
        ("checkpoint dtype", wrong_checkpoint_dtype),
        ("storage format", wrong_storage_format),
        ("shape", wrong_tensor_shape),
        ("physical name", wrong_physical_name),
        ("semantic ID", wrong_tensor_semantic),
    ];
    for (name, mutate) in cases {
        let mut mutant = compact_unfused_plan();
        mutate(&mut mutant);
        assert_error_atomic_result(
            name,
            &mutant,
            &graph,
            VisionQkvPassErrorCode::TensorBindingMismatch,
        );
    }
}

#[test]
fn weight_and_bias_buffer_offset_and_length_drifts_remain_structural_but_never_match() {
    let graph = canonical_graph();
    let cases: [(&str, fn(&mut PlanIr)); 6] = [
        ("weight buffer", wrong_weight_buffer),
        ("bias buffer", wrong_bias_buffer),
        ("weight offset", wrong_weight_offset),
        ("bias offset", wrong_bias_offset),
        ("weight length", wrong_weight_length),
        ("bias length", wrong_bias_length),
    ];
    for (name, mutate) in cases {
        let mut mutant = compact_unfused_plan();
        mutate(&mut mutant);
        mutant
            .verify()
            .unwrap_or_else(|error| panic!("{name} must remain structural PlanIR: {error}"));
        assert_error_atomic_result(
            name,
            &mutant,
            &graph,
            VisionQkvPassErrorCode::TensorBindingMismatch,
        );
    }
}

#[test]
fn enormous_structural_tensor_shape_returns_tensor_mismatch_without_panicking() {
    let mut mutant = compact_unfused_plan();
    let width = u32::MAX;
    let output_elements = usize::try_from(u64::from(width)).unwrap();
    let output_bytes = u64::from(width) * 4;
    let dispatch_x = width.div_ceil(32);

    mutant.external_values[0].shape = vec![1, u64::from(width)];
    mutant.external_values[0].byte_length = output_bytes;
    for node in &mut mutant.nodes {
        node.invocation.output_elements = output_elements;
        node.invocation.output_bytes = output_bytes;
        node.invocation.dispatch = [dispatch_x, 1, 1];
        node.uniform_words = [1, width, width, 0];
        uniform_mut(node).words = [1, width, width, 0];
        output_buffer_mut(node).byte_length = output_bytes;
        node.outputs[0].shape = vec![1, u64::from(width)];
        node.outputs[0].byte_length = output_bytes;

        tensor_mut(node, 1).shape = vec![u64::from(width), u64::from(width)];
        tensor_mut(node, 2).shape = vec![u64::from(width)];
        assert!(
            tensor_mut(node, 1).byte_length < output_bytes,
            "the tensor range must stay deliberately small"
        );
    }
    refresh_requirements(&mut mutant, 4);
    mutant
        .verify()
        .expect("large tensor shapes with nonzero small ranges must remain structural PlanIR");

    assert_error_atomic_result(
        "overflow-safe specialized tensor validation",
        &mutant,
        &canonical_graph(),
        VisionQkvPassErrorCode::TensorBindingMismatch,
    );
}

fn input_output_alias(plan: &mut PlanIr) {
    let alias = plan.external_values[0].buffer_id.clone();
    plan.nodes[0].outputs[0].buffer_id = alias.clone();
    output_buffer_mut(&mut plan.nodes[0]).buffer_id = alias;
}

fn parameter_output_alias(plan: &mut PlanIr) {
    let alias = plan.nodes[0].outputs[0].buffer_id.clone();
    tensor_mut(&mut plan.nodes[0], 1).buffer_id = alias;
}

fn overlapping_outputs(plan: &mut PlanIr) {
    *plan = expected_fused_plan(32);
    plan.nodes[0].outputs[1].byte_offset = 32;
}

fn out_of_bounds_output(plan: &mut PlanIr) {
    *plan = expected_fused_plan(32);
    plan.nodes[0].outputs[2].byte_offset = 180;
}

fn extra_consumer(plan: &mut PlanIr) {
    let (external, mut consumer, output) = sentinel_node("consumer", "preprocess.resize");
    consumer.bindings[0].resource =
        PlanBindingResource::Value(value_id(&output_value(0, Role::Query)));
    plan.external_values.push(external);
    plan.nodes.push(consumer);
    plan.outputs.push(output);
    refresh_requirements(plan, 4);
}

#[test]
fn aliases_overlap_oob_and_internal_consumers_are_rejected_atomically() {
    let graph = canonical_graph();
    for (name, mutate) in [
        ("input/output alias", input_output_alias as fn(&mut PlanIr)),
        ("parameter/output alias", parameter_output_alias),
        ("overlap", overlapping_outputs),
        ("out of bounds", out_of_bounds_output),
    ] {
        let mut mutant = compact_unfused_plan();
        mutate(&mut mutant);
        assert_error_atomic_result(name, &mutant, &graph, VisionQkvPassErrorCode::InvalidPlan);
    }

    let mut consumer = compact_unfused_plan();
    extra_consumer(&mut consumer);
    assert_error_atomic_result(
        "internal consumer",
        &consumer,
        &graph,
        VisionQkvPassErrorCode::IllegalCandidateDataflow,
    );
}

fn mutate_graph(mutator: impl FnOnce(&mut pvlc_ir::SemanticNode)) -> SemanticGraph {
    let graph = canonical_graph();
    let mut nodes = graph.nodes().to_vec();
    let qkv = nodes
        .iter_mut()
        .find(|node| node.id.as_str() == "vision.layer.00.qkv")
        .unwrap();
    mutator(qkv);
    let graph = SemanticGraph::from_nodes(nodes);
    graph
        .verify()
        .expect("semantic matcher mutant must pass generic graph verification");
    graph
}

fn semantic_node_mut<'a>(nodes: &'a mut [SemanticNode], id: &str) -> &'a mut SemanticNode {
    nodes
        .iter_mut()
        .find(|node| node.id.as_str() == id)
        .unwrap_or_else(|| panic!("missing semantic fixture node {id}"))
}

fn verified_wrong_qkv_op_graph() -> SemanticGraph {
    let graph = canonical_graph();
    let mut nodes = graph.nodes().to_vec();

    let qkv = semantic_node_mut(&mut nodes, "vision.layer.00.qkv");
    qkv.op = SemanticOp::VisionMlp;
    qkv.inputs = vec![semantic_id("vision.layer.00.norm2")];

    let mlp = semantic_node_mut(&mut nodes, "vision.layer.00.mlp");
    mlp.op = SemanticOp::VisionQkv;
    mlp.inputs = vec![semantic_id("vision.layer.00.norm1")];

    semantic_node_mut(&mut nodes, "vision.layer.00.rope").inputs =
        vec![semantic_id("vision.layer.00.mlp")];
    semantic_node_mut(&mut nodes, "vision.layer.00.attention").inputs = vec![
        semantic_id("vision.layer.00.mlp"),
        semantic_id("vision.layer.00.rope"),
    ];
    let output = semantic_node_mut(&mut nodes, "vision.layer.00.output");
    let mlp_input = output
        .inputs
        .iter_mut()
        .find(|input| input.as_str() == "vision.layer.00.mlp")
        .expect("canonical layer output must consume its MLP");
    *mlp_input = semantic_id("vision.layer.00.qkv");

    let graph = SemanticGraph::from_nodes(nodes);
    graph
        .verify()
        .expect("wrong-QKV-op graph must remain generically valid");
    graph
}

fn verified_wrong_qkv_input_graph() -> SemanticGraph {
    let graph = canonical_graph();
    let mut nodes = graph.nodes().to_vec();
    let qkv_index = nodes
        .iter()
        .position(|node| node.id.as_str() == "vision.layer.00.qkv")
        .unwrap();
    let alternate_id = semantic_id("vision.layer.00.norm1_alternate");
    nodes.insert(
        qkv_index,
        SemanticNode {
            id: alternate_id.clone(),
            op: SemanticOp::VisionLayerNorm,
            inputs: vec![semantic_id("vision.embeddings.position")],
            source_ids: vec![alternate_id.clone()],
        },
    );
    semantic_node_mut(&mut nodes, "vision.layer.00.qkv").inputs = vec![alternate_id];

    let graph = SemanticGraph::from_nodes(nodes);
    graph
        .verify()
        .expect("wrong-QKV-input graph must remain generically valid");
    graph
}

#[test]
fn semantic_op_layer_input_and_source_provenance_must_match_the_canonical_graph() {
    let input = compact_unfused_plan();

    let wrong_op = verified_wrong_qkv_op_graph();
    assert_error_atomic_result(
        "wrong semantic op",
        &input,
        &wrong_op,
        VisionQkvPassErrorCode::SemanticMismatch,
    );

    let wrong_input = verified_wrong_qkv_input_graph();
    assert_error_atomic_result(
        "wrong semantic input",
        &input,
        &wrong_input,
        VisionQkvPassErrorCode::SemanticMismatch,
    );

    let wrong_graph_source =
        mutate_graph(|qkv| qkv.source_ids = vec![semantic_id("vision.layer.00.qkv_wrong")]);
    assert_error_atomic_result(
        "wrong graph source provenance",
        &input,
        &wrong_graph_source,
        VisionQkvPassErrorCode::SemanticMismatch,
    );

    let mut wrong_plan_source = input.clone();
    for node in &mut wrong_plan_source.nodes {
        node.source_semantic_ids = vec![semantic_id("vision.layer.00.rope")];
    }
    assert_error_atomic_result(
        "wrong plan source provenance",
        &wrong_plan_source,
        &canonical_graph(),
        VisionQkvPassErrorCode::SemanticMismatch,
    );

    let mut wrong_layer = input;
    for node in &mut wrong_layer.nodes {
        node.source_semantic_ids = vec![semantic_id("vision.layer.01.qkv")];
    }
    assert_error_atomic_result(
        "wrong layer",
        &wrong_layer,
        &canonical_graph(),
        VisionQkvPassErrorCode::SemanticMismatch,
    );
}

fn assert_lowering_error(
    name: &str,
    graph: &SemanticGraph,
    layer: usize,
    geometry: &pvlc_runtime_core::VisionEncoderLayerPlan,
    catalog: &[TensorSpec],
    expected: VisionQkvPassErrorCode,
) {
    let catalog_before = catalog.to_vec();
    let error = lower_vision_qkv_fragment(graph, layer, geometry, catalog)
        .expect_err("invalid lowering input must fail");
    assert_eq!(error.code(), expected, "{name}: unexpected error: {error}");
    assert_eq!(
        catalog,
        catalog_before.as_slice(),
        "{name}: catalog mutated on error"
    );
}

#[test]
fn lowering_fails_closed_for_graph_geometry_and_every_tensor_catalog_drift() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_catalog(0);

    assert_lowering_error(
        "layer range",
        &graph,
        27,
        &geometry,
        &catalog,
        VisionQkvPassErrorCode::InvalidLayer,
    );

    let wrong_op = verified_wrong_qkv_op_graph();
    assert_lowering_error(
        "graph op",
        &wrong_op,
        0,
        &geometry,
        &catalog,
        VisionQkvPassErrorCode::SemanticMismatch,
    );

    let wrong_input = verified_wrong_qkv_input_graph();
    assert_lowering_error(
        "graph input",
        &wrong_input,
        0,
        &geometry,
        &catalog,
        VisionQkvPassErrorCode::SemanticMismatch,
    );

    let mut wrong_geometry = geometry.clone();
    let query = wrong_geometry
        .dispatches
        .iter_mut()
        .find(|dispatch| dispatch.stage == VisionEncoderLayerStage::Query)
        .unwrap();
    query.invocation.dispatch[0] += 1;
    assert_lowering_error(
        "geometry",
        &graph,
        0,
        &wrong_geometry,
        &catalog,
        VisionQkvPassErrorCode::InvalidGeometry,
    );

    let mut missing = catalog.clone();
    missing.remove(0);
    assert_lowering_error(
        "missing tensor",
        &graph,
        0,
        &geometry,
        &missing,
        VisionQkvPassErrorCode::MissingTensor,
    );

    let mut duplicate = catalog.clone();
    duplicate.push(duplicate[0].clone());
    assert_lowering_error(
        "duplicate tensor",
        &graph,
        0,
        &geometry,
        &duplicate,
        VisionQkvPassErrorCode::DuplicateTensor,
    );

    let mut swapped = catalog.clone();
    let query_semantic = swapped[0].semantic_id.clone();
    swapped[0].semantic_id = swapped[2].semantic_id.clone();
    swapped[2].semantic_id = query_semantic;
    assert_lowering_error(
        "swapped roles",
        &graph,
        0,
        &geometry,
        &swapped,
        VisionQkvPassErrorCode::TensorIdentityMismatch,
    );

    let mut wrong_dtype = catalog.clone();
    wrong_dtype[0].dtype = TensorDtype::Float16;
    assert_lowering_error(
        "dtype",
        &graph,
        0,
        &geometry,
        &wrong_dtype,
        VisionQkvPassErrorCode::TensorIdentityMismatch,
    );

    let mut wrong_shape = catalog.clone();
    wrong_shape[0].shape = vec![3, 4];
    assert_lowering_error(
        "shape",
        &graph,
        0,
        &geometry,
        &wrong_shape,
        VisionQkvPassErrorCode::TensorIdentityMismatch,
    );

    let mut wrong_physical = catalog;
    wrong_physical[0].name.push_str(".drift");
    assert_lowering_error(
        "physical name",
        &graph,
        0,
        &geometry,
        &wrong_physical,
        VisionQkvPassErrorCode::TensorIdentityMismatch,
    );
}

#[test]
fn consumed_hash_order_and_original_slices_are_verified_after_parse() {
    let graph = canonical_graph();

    let mut wrong_hash = expected_fused_plan(32);
    wrong_hash.nodes[0]
        .rewrite_provenance
        .as_mut()
        .unwrap()
        .consumed[0]
        .canonical_blake3 = "f".repeat(64);
    let verify_error = wrong_hash.verify().unwrap_err();
    assert_eq!(verify_error.code(), PlanErrorCode::InvalidRewriteProvenance);
    assert_error_atomic_result(
        "consumed hash",
        &wrong_hash,
        &graph,
        VisionQkvPassErrorCode::InvalidProvenance,
    );

    let mut wrong_order = expected_fused_plan(32);
    wrong_order.nodes[0]
        .rewrite_provenance
        .as_mut()
        .unwrap()
        .consumed
        .swap(0, 1);
    assert_error_atomic_result(
        "consumed order",
        &wrong_order,
        &graph,
        VisionQkvPassErrorCode::InvalidProvenance,
    );

    let mut wrong_slice = expected_fused_plan(32);
    let consumed = &mut wrong_slice.nodes[0]
        .rewrite_provenance
        .as_mut()
        .unwrap()
        .consumed[0];
    consumed.original.outputs[0].byte_offset = 4;
    consumed.canonical_blake3 = consumed
        .original
        .canonical_node_blake3_hex()
        .expect("mutated snapshot itself remains canonical");
    assert_error_atomic_result(
        "consumed original slice",
        &wrong_slice,
        &graph,
        VisionQkvPassErrorCode::InvalidProvenance,
    );

    let mut wrong_output_order = expected_fused_plan(32);
    wrong_output_order.nodes[0].outputs.swap(0, 1);
    assert_error_atomic_result(
        "fused output slice order",
        &wrong_output_order,
        &graph,
        VisionQkvPassErrorCode::InvalidProvenance,
    );

    let corrupt_bytes = std::str::from_utf8(EXPECTED_FUSED_CANONICAL)
        .unwrap()
        .replacen(EXPECTED_QUERY_NODE_BLAKE3, &"f".repeat(64), 1);
    let parse_error = PlanIr::parse_canonical(corrupt_bytes.as_bytes()).unwrap_err();
    assert_eq!(parse_error.code(), PlanErrorCode::InvalidRewriteProvenance);
}

#[test]
fn empty_consumed_snapshot_outputs_return_invalid_provenance_without_panicking() {
    let graph = canonical_graph();
    let mut mutant = expected_fused_plan(32);
    let consumed = &mut mutant.nodes[0]
        .rewrite_provenance
        .as_mut()
        .unwrap()
        .consumed[0];
    consumed.original.outputs.clear();
    consumed.canonical_blake3 = consumed
        .original
        .canonical_node_blake3_hex()
        .expect("the deliberately incomplete snapshot must remain rehashable");
    mutant
        .verify()
        .expect("generic PlanIR verifies the rehashed nonrecursive snapshot");

    assert_error_atomic_result(
        "consumed snapshot without an output",
        &mutant,
        &graph,
        VisionQkvPassErrorCode::InvalidProvenance,
    );
}

#[test]
fn strict_parser_rejects_unknown_rewrite_consumed_and_snapshot_fields() {
    for (scope, needle) in [
        (
            "rewrite provenance",
            "\"rewrite_provenance\":{\"pass_id\":\"vision-qkv-fusion-v1\",",
        ),
        ("consumed record", "\"consumed\":[{\"role\":\"query\","),
        (
            "original snapshot",
            "\"original\":{\"id\":\"vision.layer.00.query\",",
        ),
    ] {
        let mutant = inject_unknown(EXPECTED_FUSED_CANONICAL, scope, needle);
        let error: PlanError = match PlanIr::parse_canonical(&mutant) {
            Ok(_) => panic!("accepted unknown field inside {scope}"),
            Err(error) => error,
        };
        assert_eq!(error.code(), PlanErrorCode::UnknownField, "{scope}");
    }
}

#[test]
fn semantic_ir_and_kernel_id_compatibility_anchors_do_not_move() {
    let semantic_bytes = canonical_graph().canonical_bytes().unwrap();
    assert_eq!(semantic_bytes.len(), SEMANTIC_IR_CANONICAL_LEN);
    assert_eq!(
        blake3::hash(&semantic_bytes).to_hex().as_str(),
        SEMANTIC_IR_BLAKE3
    );

    let legacy = [
        KernelId::GemmF32,
        KernelId::GemvF32,
        KernelId::LayerNormF32,
        KernelId::RmsNormF32,
        KernelId::SiluF32,
        KernelId::GeluTanhF32,
        KernelId::RopeNeoxF32,
        KernelId::VisionAttentionF32,
        KernelId::VisionPatchProjectionF32,
        KernelId::AddF32,
        KernelId::GeluErfF32,
        KernelId::ProjectorMerge2x2F32,
    ];
    assert_eq!(KernelId::M2_PRIMITIVES, legacy[..7]);
    assert_eq!(&KernelId::ALL[..legacy.len()], legacy);
    assert_eq!(KernelId::ALL[legacy.len()], KernelId::VisionQkvFusedF32);
    assert_eq!(KernelId::VisionQkvFusedF32.as_str(), "vision_qkv_fused_f32");
}
