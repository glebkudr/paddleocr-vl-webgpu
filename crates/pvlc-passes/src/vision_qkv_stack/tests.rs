// This crate-local suite is wired by the future `vision_qkv_stack.rs` with:
// `#[cfg(test)] mod tests;`. It deliberately exercises private verifier inputs;
// none of these raw construction helpers are part of the public crate API.

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::*;
use crate::{
    VisionQkvFusionOptions, VisionQkvPassStatus, fuse_vision_qkv, lower_vision_qkv_fragment,
};
use pvlc_ir::{PlanBindingResource, PlanIr, PlanNode, PlanRewriteProvenance, SemanticGraph};
use pvlc_model_schema::TensorSpec;
use pvlc_runtime_core::{
    KernelId, VisionEncoderLayerPlan, VisionQkvExecutionPolicy, VisionQkvFusedTargetLimits,
    VisionQkvSelectionOutcome,
};

#[path = "../../tests/support/mod.rs"]
mod support;
use support::*;

fn compact_stack_catalog(depth: usize) -> Vec<TensorSpec> {
    (0..depth).flat_map(compact_catalog).collect()
}

fn accepted_fused_plan(
    layer: usize,
    geometry: &VisionEncoderLayerPlan,
    catalog: &[TensorSpec],
    graph: &SemanticGraph,
    target: VisionQkvFusedTargetLimits,
) -> (PlanIr, Vec<u8>) {
    let unfused = lower_vision_qkv_fragment(graph, layer, geometry, catalog)
        .unwrap_or_else(|error| panic!("layer {layer:02} lowering failed: {error}"));
    let fused = fuse_vision_qkv(
        &unfused,
        graph,
        VisionQkvFusionOptions {
            enabled: true,
            target,
        },
    )
    .unwrap_or_else(|error| panic!("layer {layer:02} fusion failed: {error}"));
    assert_eq!(fused.status, VisionQkvPassStatus::Fused);
    let bytes = fused.plan.canonical_bytes().unwrap();
    let parsed = PlanIr::parse_canonical(&bytes).unwrap();
    let verified = fuse_vision_qkv(
        &parsed,
        graph,
        VisionQkvFusionOptions {
            enabled: true,
            target,
        },
    )
    .unwrap_or_else(|error| panic!("layer {layer:02} existing-fused verification failed: {error}"));
    assert_eq!(verified.status, VisionQkvPassStatus::UnchangedAlreadyFused);
    assert_eq!(verified.plan.canonical_bytes().unwrap(), bytes);
    (parsed, bytes)
}

fn canonical_stack(
    depth: usize,
    geometry: &VisionEncoderLayerPlan,
    catalog: &[TensorSpec],
    graph: &SemanticGraph,
    target: VisionQkvFusedTargetLimits,
) -> Vec<Vec<u8>> {
    (0..depth)
        .map(|layer| accepted_fused_plan(layer, geometry, catalog, graph, target).1)
        .collect()
}

fn raw_bytes(plan: &PlanIr) -> Vec<u8> {
    let mut bytes =
        serde_json::to_vec(plan).expect("invalid PlanIR fixture must remain serializable");
    bytes.push(b'\n');
    bytes
}

fn inject_unknown(canonical: &[u8], needle: &str) -> Vec<u8> {
    let mut text = std::str::from_utf8(canonical).unwrap().to_owned();
    let index = text.find(needle).expect("unknown-field injection point") + needle.len();
    text.insert_str(index, "\"unknown\":true,");
    text.into_bytes()
}

fn assert_overlay_error<T>(
    name: &str,
    expected: VisionQkvStackOverlayErrorCode,
    call: impl FnOnce() -> Result<T, VisionQkvStackOverlayError>,
) {
    let outcome = catch_unwind(AssertUnwindSafe(call));
    let result = outcome.unwrap_or_else(|_| {
        panic!("{name}: verifier panicked instead of returning stable {expected:?}")
    });
    match result {
        Err(error) => assert_eq!(error.code(), expected, "{name}: unexpected error: {error}"),
        Ok(_) => panic!("{name}: invalid input was accepted; expected {expected:?}"),
    }
}

fn assert_canonical_error(
    name: &str,
    expected_layers: &[usize],
    plans: &[Vec<u8>],
    graph: &SemanticGraph,
    target: VisionQkvFusedTargetLimits,
    expected: VisionQkvStackOverlayErrorCode,
) {
    let before = plans.to_vec();
    assert_overlay_error(name, expected, || {
        verify_canonical_vision_qkv_stack_overlay(expected_layers, plans, graph, target)
    });
    assert_eq!(plans, before, "{name}: canonical source bytes mutated");
}

fn binding_candidates(layer: usize, alignment: u32) -> Vec<VisionQkvAttentionBindingCandidate> {
    let offsets = match alignment {
        32 => [0, 64, 128],
        256 => [0, 256, 512],
        _ => panic!("the hand-authored bridge oracle covers alignments 32 and 256 only"),
    };
    support::Role::ALL
        .into_iter()
        .enumerate()
        .zip(offsets)
        .map(
            |((binding, role), byte_offset)| VisionQkvAttentionBindingCandidate {
                binding: u32::try_from(binding).unwrap(),
                value_id: output_value(layer, role),
                buffer_id: shared_output_buffer(layer),
                byte_offset,
                byte_length: PLANE_BYTES,
            },
        )
        .collect()
}

fn refresh_consumed_hash(consumed: &mut pvlc_ir::PlanConsumedNode) {
    consumed.canonical_blake3 = consumed.original.canonical_node_blake3_hex().unwrap();
}

fn rewrite_provenance(plan: &mut PlanIr, mutate: impl FnOnce(&mut PlanRewriteProvenance)) {
    let provenance = plan.nodes[0]
        .rewrite_provenance
        .as_mut()
        .expect("accepted fused plan must have provenance");
    mutate(provenance);
}

fn fused_node_mutant(canonical: &[u8], mutate: impl FnOnce(&mut PlanNode)) -> Vec<u8> {
    let mut plan = PlanIr::parse_canonical(canonical).unwrap();
    mutate(&mut plan.nodes[0]);
    raw_bytes(&plan)
}

fn fused_plan_mutant(canonical: &[u8], mutate: impl FnOnce(&mut PlanIr)) -> Vec<u8> {
    let mut plan = PlanIr::parse_canonical(canonical).unwrap();
    mutate(&mut plan);
    raw_bytes(&plan)
}

#[test]
fn private_canonical_boundary_reverifies_existing_fused_plans_and_matches_public_builder() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_stack_catalog(3);
    let canonical = canonical_stack(3, &geometry, &catalog, &graph, limits(32));
    let before = canonical.clone();
    let private =
        verify_canonical_vision_qkv_stack_overlay(&[0, 1, 2], &canonical, &graph, limits(32))
            .unwrap();
    let public =
        build_verified_vision_qkv_stack_overlay(&graph, 3, &geometry, &catalog, limits(32))
            .unwrap();
    assert_eq!(canonical, before);
    assert_eq!(private.layer_count(), public.layer_count());
    for (private, public) in private.layers().iter().zip(public.layers()) {
        assert_eq!(private.layer_index(), public.layer_index());
        assert_eq!(
            private.canonical_plan_blake3_hex(),
            public.canonical_plan_blake3_hex()
        );
        assert_eq!(private.invocation(), public.invocation());
        assert_eq!(private.uniform_words(), public.uniform_words());
        assert_eq!(private.shared_output_bytes(), public.shared_output_bytes());
    }
}

#[test]
fn private_layer_set_order_and_mixed_fragment_mutants_fail_closed() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_stack_catalog(3);
    let canonical = canonical_stack(3, &geometry, &catalog, &graph, limits(32));
    let cases = vec![
        ("zero layers", vec![], vec![]),
        ("count mismatch", vec![0, 1, 2], canonical[..2].to_vec()),
        ("duplicate layer", vec![0, 1, 1], canonical.clone()),
        ("out-of-range layer", vec![0, 1, 27], canonical.clone()),
        (
            "reordered fragments",
            vec![0, 1, 2],
            vec![
                canonical[1].clone(),
                canonical[0].clone(),
                canonical[2].clone(),
            ],
        ),
        (
            "duplicated fragment",
            vec![0, 1, 2],
            vec![
                canonical[0].clone(),
                canonical[1].clone(),
                canonical[1].clone(),
            ],
        ),
        ("mixed expected layer", vec![0], vec![canonical[1].clone()]),
    ];
    for (name, expected_layers, plans) in cases {
        assert_canonical_error(
            name,
            &expected_layers,
            &plans,
            &graph,
            limits(32),
            VisionQkvStackOverlayErrorCode::LayerSetOrOrder,
        );
    }
}

#[test]
fn private_schema_canonical_and_structural_mutants_have_stable_classes() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_catalog(0);
    let canonical = canonical_stack(1, &geometry, &catalog, &graph, limits(32));

    let mut corrupt = canonical.clone();
    corrupt[0][0] = b'!';
    let mut noncanonical = canonical.clone();
    noncanonical[0].pop();
    let mut unknown = canonical.clone();
    unknown[0] = inject_unknown(&unknown[0], "{");
    for (name, plans) in [
        ("corrupt JSON", corrupt),
        ("noncanonical encoding", noncanonical),
        ("unknown field", unknown),
    ] {
        assert_canonical_error(
            name,
            &[0],
            &plans,
            &graph,
            limits(32),
            VisionQkvStackOverlayErrorCode::CanonicalEncoding,
        );
    }

    let mut wrong_schema = PlanIr::parse_canonical(&canonical[0]).unwrap();
    wrong_schema.schema_version = 2;
    assert_canonical_error(
        "wrong schema",
        &[0],
        &[raw_bytes(&wrong_schema)],
        &graph,
        limits(32),
        VisionQkvStackOverlayErrorCode::SchemaVersion,
    );

    let mut duplicate_output = PlanIr::parse_canonical(&canonical[0]).unwrap();
    duplicate_output
        .outputs
        .push(duplicate_output.outputs[0].clone());
    assert_canonical_error(
        "duplicate plan output",
        &[0],
        &[raw_bytes(&duplicate_output)],
        &graph,
        limits(32),
        VisionQkvStackOverlayErrorCode::StructuralPlan,
    );
}

#[test]
fn private_provenance_role_hash_snapshot_and_mixed_layer_mutants_are_rejected() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_stack_catalog(2);
    let canonical = canonical_stack(2, &geometry, &catalog, &graph, limits(32));
    let base = PlanIr::parse_canonical(&canonical[0]).unwrap();
    let layer1 = PlanIr::parse_canonical(&canonical[1]).unwrap();
    let layer1_key = layer1.nodes[0]
        .rewrite_provenance
        .as_ref()
        .unwrap()
        .consumed[1]
        .clone();

    let cases: Vec<(&str, Box<dyn Fn(&mut PlanIr)>)> = vec![
        (
            "wrong pass id",
            Box::new(|plan| rewrite_provenance(plan, |p| p.pass_id = "other-pass-v1".into())),
        ),
        (
            "missing consumed role",
            Box::new(|plan| {
                rewrite_provenance(plan, |p| {
                    p.consumed.pop();
                })
            }),
        ),
        (
            "duplicate consumed role",
            Box::new(|plan| rewrite_provenance(plan, |p| p.consumed[1].role = "query".into())),
        ),
        (
            "reordered consumed roles",
            Box::new(|plan| rewrite_provenance(plan, |p| p.consumed.swap(0, 1))),
        ),
        (
            "stale consumed hash",
            Box::new(|plan| {
                rewrite_provenance(plan, |p| p.consumed[0].canonical_blake3 = "00".repeat(32))
            }),
        ),
        (
            "wrong consumed tensor",
            Box::new(|plan| {
                rewrite_provenance(plan, |p| {
                    p.consumed[0].original.bindings[1].resource =
                        p.consumed[1].original.bindings[1].resource.clone();
                    refresh_consumed_hash(&mut p.consumed[0]);
                })
            }),
        ),
        (
            "wrong consumed role output",
            Box::new(|plan| {
                rewrite_provenance(plan, |p| {
                    p.consumed[0].original.outputs[0].id = value_id("vision.layer.00.query_other");
                    refresh_consumed_hash(&mut p.consumed[0]);
                })
            }),
        ),
        (
            "mixed layer consumed snapshot",
            Box::new(move |plan| rewrite_provenance(plan, |p| p.consumed[1] = layer1_key.clone())),
        ),
        (
            "mixed layer source provenance",
            Box::new(|plan| {
                let mixed_sources = vec![
                    semantic_id("vision.layer.00.qkv"),
                    semantic_id("vision.layer.01.qkv"),
                ];
                plan.nodes[0].source_semantic_ids = mixed_sources.clone();
                rewrite_provenance(plan, |p| p.source_semantic_ids = mixed_sources);
            }),
        ),
    ];

    for (name, mutate) in cases {
        let mut mutant = base.clone();
        mutate(&mut mutant);
        assert_canonical_error(
            name,
            &[0],
            &[raw_bytes(&mutant)],
            &graph,
            limits(32),
            VisionQkvStackOverlayErrorCode::RewriteProvenance,
        );
    }
}

#[test]
fn private_stale_legacy_dispatch_evidence_beside_fused_evidence_is_rejected() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_catalog(0);
    let canonical = canonical_stack(1, &geometry, &catalog, &graph, limits(32));
    let mut mixed = PlanIr::parse_canonical(&canonical[0]).unwrap();
    append_cluster(&mut mixed, 1);
    refresh_requirements(&mut mixed, 32);
    mixed
        .verify()
        .expect("fused plus stale legacy evidence is structural PlanIR");
    assert_canonical_error(
        "stale legacy evidence",
        &[0],
        &[mixed.canonical_bytes().unwrap()],
        &graph,
        limits(32),
        VisionQkvStackOverlayErrorCode::StructuralPlan,
    );
}

#[test]
fn private_live_descriptor_identity_layout_and_tensor_mutants_never_yield_overlay() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_catalog(0);
    let canonical = canonical_stack(1, &geometry, &catalog, &graph, limits(32));
    let accepted = &canonical[0];

    let wrong_kernel = fused_node_mutant(accepted, |node| {
        node.invocation.kernel = KernelId::VisionPatchProjectionF32
    });
    assert_canonical_error(
        "rewrite provenance attached to wrong kernel",
        &[0],
        &[wrong_kernel],
        &graph,
        limits(32),
        VisionQkvStackOverlayErrorCode::RewriteProvenance,
    );

    let structural = vec![
        (
            "wrong invocation bytes",
            fused_node_mutant(accepted, |node| node.invocation.output_bytes += 4),
        ),
        (
            "wrong uniform",
            fused_node_mutant(accepted, |node| node.uniform_words[3] += 1),
        ),
        (
            "wrong shared binding size",
            fused_node_mutant(accepted, |node| {
                let PlanBindingResource::OutputBuffer(output) = &mut node.bindings[7].resource
                else {
                    panic!()
                };
                output.byte_length += 32;
            }),
        ),
        (
            "split output buffer",
            fused_node_mutant(accepted, |node| {
                node.outputs[1].buffer_id = buffer_id("output.vision.layer.00.other")
            }),
        ),
        (
            "misaligned slice",
            fused_node_mutant(accepted, |node| node.outputs[1].byte_offset = 65),
        ),
        (
            "overlapping slice",
            fused_node_mutant(accepted, |node| node.outputs[1].byte_offset = 32),
        ),
        (
            "overflowing slice",
            fused_node_mutant(accepted, |node| node.outputs[2].byte_offset = u64::MAX - 8),
        ),
        (
            "out-of-bounds slice",
            fused_node_mutant(accepted, |node| node.outputs[2].byte_offset = 180),
        ),
        (
            "wrong slice size",
            fused_node_mutant(accepted, |node| node.outputs[0].byte_length = 32),
        ),
        (
            "whole-buffer slice",
            fused_node_mutant(accepted, |node| node.outputs[0].byte_length = 192),
        ),
    ];
    for (name, bytes) in structural {
        assert_canonical_error(
            name,
            &[0],
            &[bytes],
            &graph,
            limits(32),
            VisionQkvStackOverlayErrorCode::StructuralPlan,
        );
    }

    let semantic_or_tensor = vec![
        (
            "swapped Q/K value identities",
            fused_plan_mutant(accepted, |plan| {
                let query = plan.nodes[0].outputs[0].id.clone();
                plan.nodes[0].outputs[0].id = plan.nodes[0].outputs[1].id.clone();
                plan.nodes[0].outputs[1].id = query;
            }),
        ),
        (
            "swapped Q/K tensor bindings",
            fused_node_mutant(accepted, |node| {
                let query_weight = node.bindings[1].resource.clone();
                node.bindings[1].resource = node.bindings[3].resource.clone();
                node.bindings[3].resource = query_weight;
            }),
        ),
        (
            "wrong live tensor physical identity",
            fused_node_mutant(accepted, |node| {
                tensor_mut(node, 1).physical_name =
                    "visual.vision_model.encoder.layers.0.self_attn.k_proj.weight".into();
            }),
        ),
    ];
    for (name, bytes) in semantic_or_tensor {
        assert_canonical_error(
            name,
            &[0],
            &[bytes],
            &graph,
            limits(32),
            VisionQkvStackOverlayErrorCode::SemanticOrTensorIdentity,
        );
    }

    let tensor_drifts: [(&str, usize, fn(&mut pvlc_ir::PlanTensorResource)); 6] = [
        ("weight buffer id", 1, |tensor| {
            tensor.buffer_id = buffer_id("tensor.vision.layer.00.q_weight_drift")
        }),
        ("bias buffer id", 2, |tensor| {
            tensor.buffer_id = buffer_id("tensor.vision.layer.00.q_bias_drift")
        }),
        ("weight offset", 1, |tensor| tensor.byte_offset += 4),
        ("bias offset", 2, |tensor| tensor.byte_offset += 4),
        ("weight length", 1, |tensor| tensor.byte_length -= 4),
        ("bias length", 2, |tensor| tensor.byte_length -= 4),
    ];
    for (name, binding, mutate) in tensor_drifts {
        let mut plan = PlanIr::parse_canonical(accepted).unwrap();
        mutate(tensor_mut(&mut plan.nodes[0], binding));
        plan.verify()
            .unwrap_or_else(|error| panic!("{name} must remain structural PlanIR: {error}"));
        assert_canonical_error(
            name,
            &[0],
            &[plan.canonical_bytes().unwrap()],
            &graph,
            limits(32),
            VisionQkvStackOverlayErrorCode::SemanticOrTensorIdentity,
        );
    }
}

#[test]
fn private_attention_bridge_rejects_every_untrusted_binding_and_slice_mutant() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_catalog(0);
    let overlay =
        build_verified_vision_qkv_stack_overlay(&graph, 1, &geometry, &catalog, limits(32))
            .unwrap();
    let layer = &overlay.layers()[0];

    let mut cases = Vec::new();
    let mut missing = binding_candidates(0, 32);
    missing.pop();
    cases.push(("missing binding", missing));
    let mut duplicate = binding_candidates(0, 32);
    duplicate[1].binding = 0;
    cases.push(("duplicate binding", duplicate));
    let mut reordered = binding_candidates(0, 32);
    reordered.swap(0, 1);
    cases.push(("reordered binding", reordered));
    let mut wrong_value = binding_candidates(0, 32);
    wrong_value[0].value_id = output_value(0, support::Role::Key);
    cases.push(("wrong value", wrong_value));
    let mut split_buffers = binding_candidates(0, 32);
    split_buffers[1].buffer_id = "output.vision.layer.00.key".into();
    cases.push(("different physical buffer", split_buffers));
    let mut wrong_offset = binding_candidates(0, 32);
    wrong_offset[1].byte_offset = 0;
    cases.push(("wrong offset", wrong_offset));
    let mut misaligned = binding_candidates(0, 32);
    misaligned[1].byte_offset = 65;
    cases.push(("misaligned slice", misaligned));
    let mut overlapping = binding_candidates(0, 32);
    overlapping[1].byte_offset = 32;
    cases.push(("overlapping slice", overlapping));
    let mut overflowing = binding_candidates(0, 32);
    overflowing[2].byte_offset = u64::MAX - 8;
    cases.push(("overflowing slice", overflowing));
    let mut out_of_bounds = binding_candidates(0, 32);
    out_of_bounds[2].byte_offset = 180;
    cases.push(("out-of-bounds slice", out_of_bounds));
    let mut wrong_size = binding_candidates(0, 32);
    wrong_size[0].byte_length = 35;
    cases.push(("wrong slice size", wrong_size));
    let mut whole_buffer = binding_candidates(0, 32);
    whole_buffer[0].byte_length = 192;
    cases.push(("whole-buffer binding", whole_buffer));

    for (name, candidates) in cases {
        let before = candidates.clone();
        assert_overlay_error(name, VisionQkvStackOverlayErrorCode::ConsumerBridge, || {
            verify_vision_qkv_attention_bridge(layer, &candidates)
        });
        assert_eq!(candidates, before, "{name}: candidate input mutated");
    }
}

#[test]
fn private_stale_target_identity_is_not_classified_as_unsupported_target() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_catalog(0);
    let canonical32 = canonical_stack(1, &geometry, &catalog, &graph, limits(32));
    assert_canonical_error(
        "alignment-stale canonical plan",
        &[0],
        &canonical32,
        &graph,
        limits(256),
        VisionQkvStackOverlayErrorCode::StructuralPlan,
    );
}

fn result_for_non_target_error(
    expected: VisionQkvStackOverlayErrorCode,
) -> Result<VerifiedVisionQkvStackOverlay, VisionQkvStackOverlayError> {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_catalog(0);
    let canonical = canonical_stack(1, &geometry, &catalog, &graph, limits(32));
    match expected {
        VisionQkvStackOverlayErrorCode::SchemaVersion => {
            let mut plan = PlanIr::parse_canonical(&canonical[0]).unwrap();
            plan.schema_version = 2;
            verify_canonical_vision_qkv_stack_overlay(&[0], &[raw_bytes(&plan)], &graph, limits(32))
        }
        VisionQkvStackOverlayErrorCode::CanonicalEncoding => {
            let mut corrupt = canonical;
            corrupt[0][0] = b'!';
            verify_canonical_vision_qkv_stack_overlay(&[0], &corrupt, &graph, limits(32))
        }
        VisionQkvStackOverlayErrorCode::StructuralPlan => {
            let mut plan = PlanIr::parse_canonical(&canonical[0]).unwrap();
            plan.outputs.push(plan.outputs[0].clone());
            verify_canonical_vision_qkv_stack_overlay(&[0], &[raw_bytes(&plan)], &graph, limits(32))
        }
        VisionQkvStackOverlayErrorCode::RewriteProvenance => {
            let mut plan = PlanIr::parse_canonical(&canonical[0]).unwrap();
            rewrite_provenance(&mut plan, |p| {
                p.consumed[0].canonical_blake3 = "00".repeat(32)
            });
            verify_canonical_vision_qkv_stack_overlay(&[0], &[raw_bytes(&plan)], &graph, limits(32))
        }
        VisionQkvStackOverlayErrorCode::SemanticOrTensorIdentity => {
            build_verified_vision_qkv_stack_overlay(&graph, 1, &geometry, &[], limits(32))
        }
        VisionQkvStackOverlayErrorCode::LayerSetOrOrder => {
            verify_canonical_vision_qkv_stack_overlay(&[], &[], &graph, limits(32))
        }
        VisionQkvStackOverlayErrorCode::ConsumerBridge => {
            let overlay = build_verified_vision_qkv_stack_overlay(
                &graph,
                1,
                &geometry,
                &catalog,
                limits(32),
            )?;
            let mut candidates = binding_candidates(0, 32);
            candidates.pop();
            verify_vision_qkv_attention_bridge(&overlay.layers()[0], &candidates)?;
            Ok(overlay)
        }
        VisionQkvStackOverlayErrorCode::UnsupportedTarget => {
            panic!("unsupported target is tested as the only fallback separately")
        }
    }
}

#[test]
fn preferred_and_required_propagate_every_non_target_error_class_without_retry() {
    const NON_TARGET: [VisionQkvStackOverlayErrorCode; 7] = [
        VisionQkvStackOverlayErrorCode::SchemaVersion,
        VisionQkvStackOverlayErrorCode::CanonicalEncoding,
        VisionQkvStackOverlayErrorCode::StructuralPlan,
        VisionQkvStackOverlayErrorCode::RewriteProvenance,
        VisionQkvStackOverlayErrorCode::SemanticOrTensorIdentity,
        VisionQkvStackOverlayErrorCode::LayerSetOrOrder,
        VisionQkvStackOverlayErrorCode::ConsumerBridge,
    ];
    for expected in NON_TARGET {
        for policy in [
            VisionQkvExecutionPolicy::Preferred,
            VisionQkvExecutionPolicy::Required,
        ] {
            let calls = Cell::new(0);
            assert_overlay_error("selection propagation", expected, || {
                select_vision_qkv_stack_overlay(policy, || {
                    calls.set(calls.get() + 1);
                    result_for_non_target_error(expected)
                })
            });
            assert_eq!(calls.get(), 1, "{policy:?}/{expected:?}: retry detected");
        }
    }

    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_catalog(0);
    let unsupported = VisionQkvFusedTargetLimits {
        max_storage_buffers_per_shader_stage: 7,
        ..limits(32)
    };
    let fallback = select_vision_qkv_stack_overlay(VisionQkvExecutionPolicy::Preferred, || {
        build_verified_vision_qkv_stack_overlay(&graph, 1, &geometry, &catalog, unsupported)
    })
    .unwrap();
    assert_eq!(
        fallback.outcome(),
        VisionQkvSelectionOutcome::FallbackUnsupportedTarget
    );
    assert_eq!(
        fallback.fallback_error_code(),
        Some(VisionQkvStackOverlayErrorCode::UnsupportedTarget)
    );
}

#[path = "m7c2b_tests.rs"]
mod m7c2b_tests;
