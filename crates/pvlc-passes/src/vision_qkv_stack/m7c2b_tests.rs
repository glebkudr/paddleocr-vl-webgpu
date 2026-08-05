//! M7c2b crate-private hostile tests for the shared prepared-execution boundary.
//!
//! This child module is intentionally wired from `tests.rs`: it needs access to
//! private verified descriptor fields while keeping those mutation surfaces out
//! of the public compiler API.

use std::panic::{AssertUnwindSafe, catch_unwind};

use super::super::*;
use super::{
    compact_stack_catalog,
    support::{self, *},
};
use pvlc_runtime_core::{KernelId, VisionQkvFusedTargetLimits};

fn accepted_overlay(depth: usize, alignment: u32) -> VerifiedVisionQkvStackOverlay {
    build_verified_vision_qkv_stack_overlay(
        &canonical_graph(),
        depth,
        &compact_layer_plan(),
        &compact_stack_catalog(depth),
        limits(alignment),
    )
    .expect("private hostile fixture must start from an accepted overlay")
}

fn assert_prepared_error(
    label: &str,
    expected: VisionQkvPreparedExecutionErrorCode,
    overlay: &VerifiedVisionQkvStackOverlay,
    depth: usize,
    target: VisionQkvFusedTargetLimits,
) {
    let before = overlay.clone();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        prepare_vision_qkv_stack_execution(overlay, depth, &compact_layer_plan(), target)
    }));
    let result = outcome.unwrap_or_else(|_| {
        panic!("{label}: prepared verifier panicked instead of returning {expected:?}")
    });
    let error = match result {
        Ok(_) => panic!("{label}: hostile descriptor was accepted; expected {expected:?}"),
        Err(error) => error,
    };
    assert_eq!(
        error.code(),
        expected,
        "{label}: wrong stable error: {error}"
    );
    assert_eq!(
        overlay, &before,
        "{label}: verifier mutated its overlay input"
    );
}

#[test]
fn prepared_view_rejects_depth_order_and_cross_layer_drift_without_partial_output() {
    let baseline = accepted_overlay(3, 32);

    assert_prepared_error(
        "manifest depth shorter than overlay",
        VisionQkvPreparedExecutionErrorCode::LayerSetOrOrder,
        &baseline,
        2,
        limits(32),
    );
    assert_prepared_error(
        "manifest depth longer than overlay",
        VisionQkvPreparedExecutionErrorCode::LayerSetOrOrder,
        &baseline,
        4,
        limits(32),
    );

    let mut reordered = baseline.clone();
    reordered.layers.swap(0, 1);
    assert_prepared_error(
        "reordered layer descriptors",
        VisionQkvPreparedExecutionErrorCode::LayerSetOrOrder,
        &reordered,
        3,
        limits(32),
    );

    let mut duplicate = baseline.clone();
    duplicate.layers[2].layer_index = 1;
    assert_prepared_error(
        "duplicate layer identity",
        VisionQkvPreparedExecutionErrorCode::LayerSetOrOrder,
        &duplicate,
        3,
        limits(32),
    );

    let mut coherent_abi_drift = baseline.clone();
    let drifted = &mut coherent_abi_drift.layers[2];
    drifted.uniform_words[3] = 24;
    drifted.invocation.output_elements = 72;
    drifted.invocation.output_bytes = 288;
    drifted.shared_output_bytes = 288;
    for (binding, byte_offset) in drifted
        .attention_bridge
        .bindings
        .iter_mut()
        .zip([0_u64, 96, 192])
    {
        binding.byte_offset = byte_offset;
    }
    assert_eq!(drifted.invocation.output_elements * 4, 288);
    assert_eq!(drifted.invocation.output_bytes, drifted.shared_output_bytes);
    assert_eq!(u64::from(drifted.uniform_words[3]) * 4 * 3, 288);
    assert!(
        drifted
            .attention_bridge
            .bindings
            .iter()
            .all(|binding| binding.byte_offset % 32 == 0
                && binding.byte_offset + binding.byte_length <= 288)
    );
    assert_prepared_error(
        "coherent cross-layer execution ABI drift",
        VisionQkvPreparedExecutionErrorCode::CrossLayerDrift,
        &coherent_abi_drift,
        3,
        limits(32),
    );
}

#[test]
fn prepared_view_rejects_kernel_invocation_uniform_dispatch_and_output_drift_stably() {
    let baseline = accepted_overlay(1, 32);
    let cases: [(
        &str,
        VisionQkvPreparedExecutionErrorCode,
        fn(&mut VerifiedVisionQkvLayerDescriptor),
    ); 14] = [
        (
            "wrong kernel",
            VisionQkvPreparedExecutionErrorCode::Kernel,
            |layer| {
                layer.invocation.kernel = KernelId::VisionPatchProjectionF32;
            },
        ),
        (
            "wrong workgroup x",
            VisionQkvPreparedExecutionErrorCode::Invocation,
            |layer| {
                layer.invocation.workgroup_size = [4, 8, 1];
            },
        ),
        (
            "wrong workgroup y",
            VisionQkvPreparedExecutionErrorCode::Invocation,
            |layer| {
                layer.invocation.workgroup_size = [8, 4, 1];
            },
        ),
        (
            "wrong workgroup z",
            VisionQkvPreparedExecutionErrorCode::Invocation,
            |layer| {
                layer.invocation.workgroup_size = [8, 8, 2];
            },
        ),
        (
            "wrong output elements",
            VisionQkvPreparedExecutionErrorCode::OutputBytes,
            |layer| {
                layer.invocation.output_elements -= 1;
            },
        ),
        (
            "wrong invocation bytes",
            VisionQkvPreparedExecutionErrorCode::OutputBytes,
            |layer| {
                layer.invocation.output_bytes -= 4;
            },
        ),
        (
            "wrong semantic bytes",
            VisionQkvPreparedExecutionErrorCode::OutputBytes,
            |layer| {
                layer.shared_output_bytes -= 4;
            },
        ),
        (
            "wrong dispatch x",
            VisionQkvPreparedExecutionErrorCode::Dispatch,
            |layer| {
                layer.invocation.dispatch[0] += 1;
            },
        ),
        (
            "wrong dispatch y",
            VisionQkvPreparedExecutionErrorCode::Dispatch,
            |layer| {
                layer.invocation.dispatch[1] += 1;
            },
        ),
        (
            "wrong dispatch z",
            VisionQkvPreparedExecutionErrorCode::Dispatch,
            |layer| {
                layer.invocation.dispatch[2] = 2;
            },
        ),
        (
            "wrong uniform tokens",
            VisionQkvPreparedExecutionErrorCode::Uniform,
            |layer| {
                layer.uniform_words[0] += 1;
            },
        ),
        (
            "wrong uniform input width",
            VisionQkvPreparedExecutionErrorCode::Uniform,
            |layer| {
                layer.uniform_words[1] += 1;
            },
        ),
        (
            "wrong uniform width",
            VisionQkvPreparedExecutionErrorCode::Uniform,
            |layer| {
                layer.uniform_words[2] += 1;
            },
        ),
        (
            "wrong uniform stride",
            VisionQkvPreparedExecutionErrorCode::Uniform,
            |layer| {
                layer.uniform_words[3] -= 1;
            },
        ),
    ];

    for (label, expected, mutate) in cases {
        let mut mutant = baseline.clone();
        mutate(&mut mutant.layers[0]);
        assert_prepared_error(label, expected, &mutant, 1, limits(32));
    }
}

#[test]
fn prepared_view_rechecks_kernel_uniform_dispatch_and_output_on_the_last_layer() {
    let baseline = accepted_overlay(3, 32);
    let cases: [(
        &str,
        VisionQkvPreparedExecutionErrorCode,
        fn(&mut VerifiedVisionQkvLayerDescriptor),
    ); 4] = [
        (
            "last-layer kernel",
            VisionQkvPreparedExecutionErrorCode::Kernel,
            |layer| layer.invocation.kernel = KernelId::VisionPatchProjectionF32,
        ),
        (
            "last-layer uniform",
            VisionQkvPreparedExecutionErrorCode::Uniform,
            |layer| layer.uniform_words[3] -= 1,
        ),
        (
            "last-layer dispatch",
            VisionQkvPreparedExecutionErrorCode::Dispatch,
            |layer| layer.invocation.dispatch[0] += 1,
        ),
        (
            "last-layer output bytes",
            VisionQkvPreparedExecutionErrorCode::OutputBytes,
            |layer| layer.invocation.output_bytes -= 4,
        ),
    ];

    for (label, expected, mutate) in cases {
        let mut mutant = baseline.clone();
        mutate(&mut mutant.layers[2]);
        assert_prepared_error(label, expected, &mutant, 3, limits(32));
    }
}

#[test]
fn prepared_view_rechecks_all_five_fields_for_every_qkv_binding_on_a_middle_layer() {
    const MIDDLE_LAYER: usize = 1;
    for binding_index in 0..support::Role::ALL.len() {
        for (field, expected) in [
            (
                "binding number",
                VisionQkvPreparedExecutionErrorCode::ConsumerBridge,
            ),
            (
                "value identity",
                VisionQkvPreparedExecutionErrorCode::ConsumerBridge,
            ),
            (
                "buffer identity",
                VisionQkvPreparedExecutionErrorCode::ConsumerBridge,
            ),
            (
                "byte offset",
                VisionQkvPreparedExecutionErrorCode::WorkspaceLayout,
            ),
            (
                "byte length",
                VisionQkvPreparedExecutionErrorCode::WorkspaceLayout,
            ),
        ] {
            let baseline = accepted_overlay(3, 32);
            prepare_vision_qkv_stack_execution(
                &baseline,
                3,
                &compact_layer_plan(),
                limits(32),
            )
            .unwrap_or_else(|error| {
                panic!(
                    "role {binding_index} / {field}: isolated baseline must remain valid: {error}"
                )
            });

            let mut mutant = baseline.clone();
            let binding = &mut mutant.layers[MIDDLE_LAYER].attention_bridge.bindings[binding_index];
            match field {
                "binding number" => binding.binding = ((binding_index + 1) % 3) as u32,
                "value identity" => {
                    binding.value_id =
                        output_value(MIDDLE_LAYER, support::Role::ALL[(binding_index + 1) % 3]);
                }
                "buffer identity" => {
                    binding.buffer_id = format!("other-middle-workspace-{binding_index}");
                }
                "byte offset" => binding.byte_offset += 4,
                "byte length" => binding.byte_length -= 4,
                _ => unreachable!(),
            }
            if field == "byte offset" || field == "byte length" {
                let bridge = &mutant.layers[MIDDLE_LAYER].attention_bridge.bindings;
                assert!(bridge.iter().all(|binding| binding.byte_offset % 4 == 0));
                assert!(bridge.iter().all(|binding| binding.byte_length > 0));
                assert!(bridge.windows(2).all(|pair| {
                    pair[0].byte_offset + pair[0].byte_length <= pair[1].byte_offset
                }));
                assert!(bridge.iter().all(|binding| {
                    binding.byte_offset + binding.byte_length
                        <= mutant.layers[MIDDLE_LAYER].shared_output_bytes
                }));
            }
            assert_prepared_error(
                &format!("middle-layer role {binding_index} {field}"),
                expected,
                &mutant,
                3,
                limits(32),
            );
        }
    }
}

#[test]
fn prepared_view_rejects_every_hostile_attention_bridge_shape() {
    let baseline = accepted_overlay(1, 32);
    let mut cases: Vec<(
        &str,
        VisionQkvPreparedExecutionErrorCode,
        VerifiedVisionQkvStackOverlay,
    )> = Vec::new();

    let mut missing = baseline.clone();
    missing.layers[0].attention_bridge.bindings.pop();
    cases.push((
        "missing binding",
        VisionQkvPreparedExecutionErrorCode::ConsumerBridge,
        missing,
    ));

    let mut duplicate = baseline.clone();
    duplicate.layers[0].attention_bridge.bindings[1].binding = 0;
    cases.push((
        "duplicate binding",
        VisionQkvPreparedExecutionErrorCode::ConsumerBridge,
        duplicate,
    ));

    let mut extra = baseline.clone();
    let fourth_binding = extra.layers[0].attention_bridge.bindings[2].clone();
    extra.layers[0]
        .attention_bridge
        .bindings
        .push(fourth_binding);
    cases.push((
        "extra fourth binding",
        VisionQkvPreparedExecutionErrorCode::ConsumerBridge,
        extra,
    ));

    let mut reordered = baseline.clone();
    reordered.layers[0].attention_bridge.bindings.swap(0, 1);
    cases.push((
        "reordered binding",
        VisionQkvPreparedExecutionErrorCode::ConsumerBridge,
        reordered,
    ));

    let mut split = baseline.clone();
    split.layers[0].attention_bridge.bindings[1].buffer_id = "other-workspace".into();
    cases.push((
        "different buffer",
        VisionQkvPreparedExecutionErrorCode::ConsumerBridge,
        split,
    ));

    let mut wrong_value = baseline.clone();
    wrong_value.layers[0].attention_bridge.bindings[0].value_id =
        output_value(0, support::Role::Key);
    cases.push((
        "wrong value identity",
        VisionQkvPreparedExecutionErrorCode::ConsumerBridge,
        wrong_value,
    ));

    let mut whole = baseline.clone();
    whole.layers[0].attention_bridge.bindings[0].byte_length = 192;
    cases.push((
        "whole workspace",
        VisionQkvPreparedExecutionErrorCode::WorkspaceLayout,
        whole,
    ));

    let mut misaligned = baseline.clone();
    misaligned.layers[0].attention_bridge.bindings[1].byte_offset = 65;
    cases.push((
        "misaligned slice",
        VisionQkvPreparedExecutionErrorCode::WorkspaceLayout,
        misaligned,
    ));

    let mut overlap = baseline.clone();
    overlap.layers[0].attention_bridge.bindings[1].byte_offset = 32;
    cases.push((
        "overlapping slice",
        VisionQkvPreparedExecutionErrorCode::WorkspaceLayout,
        overlap,
    ));

    let mut out_of_bounds = baseline.clone();
    out_of_bounds.layers[0].attention_bridge.bindings[2].byte_offset = 180;
    cases.push((
        "out-of-bounds slice",
        VisionQkvPreparedExecutionErrorCode::WorkspaceLayout,
        out_of_bounds,
    ));

    let mut overflow = baseline.clone();
    overflow.layers[0].attention_bridge.bindings[2].byte_offset = u64::MAX - 31;
    cases.push((
        "overflowing slice",
        VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
        overflow,
    ));

    for (label, expected, mutant) in cases {
        assert_prepared_error(label, expected, &mutant, 1, limits(32));
    }
}

#[test]
fn prepared_view_rechecks_actual_target_requirements_without_treating_maxima_as_identity() {
    let baseline = accepted_overlay(1, 32);

    prepare_vision_qkv_stack_execution(&baseline, 1, &compact_layer_plan(), larger_limits(32))
        .expect("different sufficient maxima must remain compatible");

    let targets = [
        (
            "stale alignment",
            VisionQkvPreparedExecutionErrorCode::TargetAlignment,
            limits(256),
        ),
        (
            "invalid alignment",
            VisionQkvPreparedExecutionErrorCode::TargetAlignment,
            VisionQkvFusedTargetLimits {
                min_storage_buffer_offset_alignment: 3,
                ..limits(32)
            },
        ),
        (
            "insufficient storage bindings",
            VisionQkvPreparedExecutionErrorCode::TargetStorageBindings,
            VisionQkvFusedTargetLimits {
                max_storage_buffers_per_shader_stage: 7,
                ..limits(32)
            },
        ),
        (
            "insufficient binding size",
            VisionQkvPreparedExecutionErrorCode::TargetBindingSize,
            VisionQkvFusedTargetLimits {
                max_storage_buffer_binding_size: 191,
                ..limits(32)
            },
        ),
        (
            "insufficient workspace buffer size",
            VisionQkvPreparedExecutionErrorCode::TargetBufferSize,
            VisionQkvFusedTargetLimits {
                max_buffer_size: 255,
                ..limits(32)
            },
        ),
        (
            "insufficient dispatch limit",
            VisionQkvPreparedExecutionErrorCode::TargetDispatchLimit,
            VisionQkvFusedTargetLimits {
                max_compute_workgroups_per_dimension: 0,
                ..limits(32)
            },
        ),
    ];
    for (label, expected, target) in targets {
        assert_prepared_error(label, expected, &baseline, 1, target);
    }
}

#[test]
fn prepared_view_uses_cloned_authority_after_the_overlay_is_dropped() {
    let prepared = {
        let overlay = accepted_overlay(3, 32);
        prepare_vision_qkv_stack_execution(&overlay, 3, &compact_layer_plan(), limits(32))
            .expect("accepted overlay must prepare")
    };

    assert_eq!(prepared.layer_count(), 3);
    assert_eq!(prepared.layers().len(), 3);
    assert_eq!(prepared.layers()[2].layer_index(), 2);
    assert_eq!(prepared.workspace().semantic_base(), 32);
}
