mod support;

use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};

use pvlc_ir::SemanticGraph;
use pvlc_model_schema::{PaddleOcrVl16Schema, TensorSpec};
use pvlc_passes::{
    VerifiedVisionQkvLayerDescriptor, VisionQkvStackOverlayError, VisionQkvStackOverlayErrorCode,
    build_verified_vision_qkv_stack_overlay, select_vision_qkv_stack_overlay,
};
use pvlc_runtime_core::{
    KernelId, VisionQkvExecutionPolicy, VisionQkvFusedTargetLimits, VisionQkvSelectionOutcome,
};

use support::*;

const COMPACT_ALIGN32_BLAKE3: [&str; 27] = [
    "11e7a2ea2a5e602e9f3415e74a2190df6df522452fbafad7d62ceb67318da481",
    "fe4608a21ea003507de4b21f42c546b77810637a99bdb1c118a601722eab1427",
    "ab920d6b363f713fce64ef0980052f4dfb7adf248864e05ab22eb5ef1c96ebc3",
    "73cd742f3e8e7e3b5d2fdfc172a5a3c035e5dae88dbe21a6b06b44ad83d4e8dd",
    "5636f4b1561e55dcb9c3c3e87c055a1347b261d8ce0d2c1ed2fae83cb25e4499",
    "ae592e5062e7d48a65052c93c9267cd1f29b20dfa766bec2aff960264be48110",
    "66adc2f8820b2e4276b7a8e0ec4dcd0b961f1827d5188f21f01e48fbcd662a5b",
    "f98f2597c0b1475460e88e55e00285282ea59b4186cd99d135d4b0bf1f2ad1b7",
    "bb9aa875731afb8c9a483527147d1c4a3e6080e21a671d6f9d6832f0cbe0b079",
    "cc114b95563ac5cdd550cacada45d1df5c2a86990b913173d311a9167a4caa49",
    "3f07d2cf7798a0e2d1a099985d1e52967bc68d22a341f8dfaee37da6240bd1b9",
    "ae6a1a87614d2ec8fdde1ccb31b2394f52afce1779ceea069a6326d728f8121a",
    "faa446081bc0f902c2215dfa5ef74a37c24db1d18298080f61422a8d8e8dd8a7",
    "288863b2724dc7a67aa367b75dc92ce4de4247d577b18c4b6869ef1096b625bd",
    "8a7c57710c7da46a4ab8697a74e543bfcd013794bfef996cc556cf21837fb705",
    "33aecdd9695b8345e0ea8eb4c51d40e0bc6608f281f0f45f2b29cdc1e19dc700",
    "1e92449760ac1e8135ac3914f0659642ffe6307e4e2118fe617eda89e7f52a4b",
    "3751b61a7b73e9149b552c1274d989f409f60f9e4bf20c15b5fa25523f606b8e",
    "27c911ed66a97c6c43feca7aba0e088c0b9755c406a5148160b67de8deffa45d",
    "e4196600e081a1fa0eea2ce26b228a8c2dcb145e8ee4440785dab6f0c7763c5b",
    "f2165c6805fa3753192420cc1e839bf53855b493a562bcfecd35d48681ca6dfe",
    "2278e73ed67d290d1d7cd1e75dbb06e4cbd8f9e50d63eb7f050ad59ab7729d58",
    "f3f6f0e88cc44829f3fffe4d1754b560b4dd846264150be09bb4a86fdfd4be2a",
    "28f1f0109ee72c699c4701daa1aed0b681caaca67f46b8ee3e3ea5d1bc4cba42",
    "2ac1f39dd8a0cf5e3563415de540434fa84e309639f12a9fd1e9805a02356b25",
    "2c91a061544fe53867266d564aeff7b9fab5d8c475dd4defe3e528f1f40f7eec",
    "16f4d8f4729633a03a8837fa4e166dff24b4b04dcf9812f0b71ede177d9fe3dd",
];

const COMPACT_LAYER00_ALIGN256_BLAKE3: &str =
    "361a0a1c71de94c6ec30aafd725edbff5ddac65ac0d01f66dc7668c4d1d8688b";

const OFFICIAL_ALIGN256_BLAKE3: [&str; 27] = [
    "aef174a939b9674473def594c71f439d7aa1889da96e83b55fea9bacd58aba02",
    "116a499ea3dc8a4fe9f972365bc0cac89f638f28710b187df55a6d62d879f6ee",
    "3ae7c47cec96d773cb07ffbaadf28d22baa4393bf8c169d850016f52f0359965",
    "d7f5005ad0e9e3d16d09351f73a1ab2297d036c077ca928e6c729322edb0140f",
    "79dcdb426aeecfa70371b9a39a2a17ca0fe839632c3a232b575afdc266426a25",
    "8d6b6f96ca10d6296dbe2b8c7aec6c539f85b889f421eb7b777a1da401db140b",
    "c19da73a322a0a1fea11559bfbbce7d2174dcebac17817c55e4eaca2d4c17d07",
    "ff52537b6d536eabcb4fd9c35cfdd948bac6e1e74f4f10f2a2d84645400acdf5",
    "36f4c71a7d7fe8298df9ed98e441e75e229c55b5a2f2c301f02503bffb347210",
    "5f29927f13b40a2c58bfd708fd5342326966136243ea5843fd706e1bb9e62d79",
    "5a95341071256730ef25d94a823563ac751700317f092497f946c32adf6e1640",
    "fe6085808152a7bc5dcd2d8855ffae741fbd9166f8ba960860f1e958266b5b23",
    "8136372b002cd287efd892c8c049d074e0774d4ba538c3a77d1ef270660ac3fd",
    "f35730da22ac3c38135cf52c0af8f865fbeec16c3a2f757e2422e3265e9373ba",
    "38e41c7f9a4bf642800ea2808a9a47befc8aa896e10aa0eabc3c70f1a5643d3c",
    "3f2aad3149f63f5cdc4d8ce43697f4cf24b16ae5d7fa0ad244794480ec82056b",
    "fd9cab048c81c81e97d3a41d0c6cf48786e082218a36584f56ccd033842fc6b9",
    "4b5cac407040a4cd2e6d93aa9b1166903074a2756e942db6c43853edf53e2a9f",
    "d65e0a7e2a8c7c20dd9a26b5b6025b1eb721bfefb6a469b12755e0be91b43697",
    "f02383a35fb7a73fedded7611c9e200fdf353ed9f57ac6b6a5f170f417153924",
    "9753b64de317fdc8c2f70ddf3fe75300665a0792792eeac8decf61a894955223",
    "86558dbea04dd054c4eb3f169a7df0fac8753acf172c8b2bf7c7cfebda5040c2",
    "1793e3409b6cf10dbb1f9567e685cb25a598628561e4aa8305aef2f3a01733d3",
    "28d2ad5bab768cc43500e2984f921f5744d160dfd2c0bb13456b56ce973bfe2d",
    "b55bfe91a03ec32120d38212291337025f3138e7a6d72c64e854186e4e7dd857",
    "ea575f2b745bfa941cacc770d62eb8cf80eb04dedb0c8c679c4335c357ababe7",
    "e5b978efcdd85cbbe3b8c211ae0c79b18d20927330a4c707586713456c595655",
];

fn compact_stack_catalog(depth: usize) -> Vec<TensorSpec> {
    (0..depth).flat_map(compact_catalog).collect()
}

fn assert_overlay_error<T>(
    name: &str,
    expected: VisionQkvStackOverlayErrorCode,
    call: impl FnOnce() -> Result<T, VisionQkvStackOverlayError>,
) {
    let outcome = catch_unwind(AssertUnwindSafe(call));
    let result = outcome.unwrap_or_else(|_| {
        panic!("{name}: validation panicked instead of returning stable {expected:?}")
    });
    match result {
        Err(error) => assert_eq!(error.code(), expected, "{name}: unexpected error: {error}"),
        Ok(_) => panic!("{name}: invalid input was accepted; expected {expected:?}"),
    }
}

fn assert_exact_compact_layer(
    layer: &VerifiedVisionQkvLayerDescriptor,
    expected_layer: usize,
    alignment: u32,
    expected_blake3: &str,
) {
    let expected = independent_layout(alignment);
    assert_eq!(layer.layer_index(), expected_layer);
    assert_eq!(layer.canonical_plan_blake3_hex(), expected_blake3);
    assert_eq!(layer.invocation().kernel, KernelId::VisionQkvFusedF32);
    assert_eq!(
        layer.invocation().output_elements,
        usize::try_from(expected.physical_bytes / 4).unwrap()
    );
    assert_eq!(layer.invocation().output_bytes, expected.physical_bytes);
    assert_eq!(layer.invocation().workgroup_size, [8, 8, 1]);
    assert_eq!(layer.invocation().dispatch, expected.dispatch);
    assert_eq!(layer.uniform_words(), expected.uniform_words);
    assert_eq!(layer.shared_output_bytes(), expected.physical_bytes);

    let bridge = layer.attention_bridge();
    assert_eq!(bridge.bindings().len(), 3);
    for ((binding, role), offset) in bridge
        .bindings()
        .iter()
        .zip(Role::ALL)
        .zip(expected.offsets)
    {
        let number = match role {
            Role::Query => 0,
            Role::Key => 1,
            Role::Value => 2,
        };
        assert_eq!(binding.binding(), number);
        assert_eq!(binding.value_id(), output_value(expected_layer, role));
        assert_eq!(binding.buffer_id(), shared_output_buffer(expected_layer));
        assert_eq!(binding.byte_offset(), offset);
        assert_eq!(binding.byte_length(), PLANE_BYTES);
    }
}

fn assert_same_execution_identity(
    left: &VerifiedVisionQkvLayerDescriptor,
    right: &VerifiedVisionQkvLayerDescriptor,
) {
    assert_eq!(left.layer_index(), right.layer_index());
    assert_eq!(
        left.canonical_plan_blake3_hex(),
        right.canonical_plan_blake3_hex()
    );
    assert_eq!(left.invocation(), right.invocation());
    assert_eq!(left.uniform_words(), right.uniform_words());
    assert_eq!(left.shared_output_bytes(), right.shared_output_bytes());
    assert_eq!(
        left.attention_bridge().bindings().len(),
        right.attention_bridge().bindings().len()
    );
    for (left, right) in left
        .attention_bridge()
        .bindings()
        .iter()
        .zip(right.attention_bridge().bindings())
    {
        assert_eq!(left.binding(), right.binding());
        assert_eq!(left.value_id(), right.value_id());
        assert_eq!(left.buffer_id(), right.buffer_id());
        assert_eq!(left.byte_offset(), right.byte_offset());
        assert_eq!(left.byte_length(), right.byte_length());
    }
}

#[test]
fn stack_builder_depths_1_3_16_are_ordered_exact_deterministic_and_immutable() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    for depth in [1, 3, 16] {
        let catalog = compact_stack_catalog(depth);
        let graph_before = graph.clone();
        let catalog_before = catalog.clone();
        let first =
            build_verified_vision_qkv_stack_overlay(&graph, depth, &geometry, &catalog, limits(32))
                .unwrap_or_else(|error| panic!("depth {depth} failed: {error}"));
        let second =
            build_verified_vision_qkv_stack_overlay(&graph, depth, &geometry, &catalog, limits(32))
                .unwrap();

        assert_eq!(graph, graph_before, "depth {depth}: SemanticIR mutated");
        assert_eq!(catalog, catalog_before, "depth {depth}: catalog mutated");
        assert_eq!(first.layer_count(), depth);
        assert_eq!(first.layers().len(), depth);
        assert_eq!(first.target_limits(), limits(32));
        assert_eq!(second.layer_count(), first.layer_count());
        for (layer, expected_hash) in COMPACT_ALIGN32_BLAKE3
            .iter()
            .copied()
            .enumerate()
            .take(depth)
        {
            assert_exact_compact_layer(&first.layers()[layer], layer, 32, expected_hash);
            assert_same_execution_identity(&first.layers()[layer], &second.layers()[layer]);
        }
    }
}

#[test]
fn all_27_official_layers_resolve_from_shuffled_catalog_to_literal_hashes() {
    let graph = canonical_graph();
    let geometry = official_layer_plan();
    let mut shuffled = PaddleOcrVl16Schema::tensor_specs();
    shuffled.reverse();
    shuffled.rotate_left(137);
    let shuffled_before = shuffled.clone();

    let overlay =
        build_verified_vision_qkv_stack_overlay(&graph, 27, &geometry, &shuffled, limits(256))
            .expect("the complete official vision stack must build");
    assert_eq!(shuffled, shuffled_before);
    assert_eq!(overlay.layer_count(), 27);
    assert_eq!(overlay.layers().len(), 27);
    for (layer, descriptor) in overlay.layers().iter().enumerate() {
        assert_eq!(descriptor.layer_index(), layer);
        assert_eq!(
            descriptor.canonical_plan_blake3_hex(),
            OFFICIAL_ALIGN256_BLAKE3[layer]
        );
        assert_eq!(descriptor.invocation().kernel, KernelId::VisionQkvFusedF32);
        assert_eq!(descriptor.uniform_words(), [1, 1_152, 1_152, 1_152]);
        assert_eq!(descriptor.shared_output_bytes(), 13_824);
        for ((actual, role), expected_offset) in descriptor
            .attention_bridge()
            .bindings()
            .iter()
            .zip(Role::ALL)
            .zip([0, 4_608, 9_216])
        {
            let expected_binding = match role {
                Role::Query => 0,
                Role::Key => 1,
                Role::Value => 2,
            };
            assert_eq!(actual.binding(), expected_binding);
            assert_eq!(actual.value_id(), output_value(layer, role));
            assert_eq!(actual.buffer_id(), shared_output_buffer(layer));
            assert_eq!(actual.byte_offset(), expected_offset);
            assert_eq!(actual.byte_length(), 4_608);
        }
    }
}

#[test]
fn compact_read_only_bridge_is_exact_for_alignments_32_and_256() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_catalog(0);
    for (alignment, expected_hash) in [
        (32, COMPACT_ALIGN32_BLAKE3[0]),
        (256, COMPACT_LAYER00_ALIGN256_BLAKE3),
    ] {
        let overlay = build_verified_vision_qkv_stack_overlay(
            &graph,
            1,
            &geometry,
            &catalog,
            limits(alignment),
        )
        .unwrap();
        assert_exact_compact_layer(&overlay.layers()[0], 0, alignment, expected_hash);
    }
}

#[test]
fn sufficient_target_maxima_do_not_enter_overlay_execution_identity() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_stack_catalog(3);
    let baseline =
        build_verified_vision_qkv_stack_overlay(&graph, 3, &geometry, &catalog, limits(32))
            .unwrap();
    let permissive =
        build_verified_vision_qkv_stack_overlay(&graph, 3, &geometry, &catalog, larger_limits(32))
            .unwrap();

    assert_eq!(baseline.target_limits(), limits(32));
    assert_eq!(permissive.target_limits(), larger_limits(32));
    assert_ne!(baseline.target_limits(), permissive.target_limits());
    for (baseline, permissive) in baseline.layers().iter().zip(permissive.layers()) {
        assert_same_execution_identity(baseline, permissive);
    }
}

#[test]
fn public_builder_rejects_invalid_layer_graph_and_catalog_inputs_atomically() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_catalog(0);
    let graph_before = graph.clone();
    let catalog_before = catalog.clone();

    assert_overlay_error(
        "zero layers",
        VisionQkvStackOverlayErrorCode::LayerSetOrOrder,
        || build_verified_vision_qkv_stack_overlay(&graph, 0, &geometry, &[], limits(32)),
    );
    let complete_catalog = compact_stack_catalog(27);
    assert_overlay_error(
        "layer count beyond canonical model",
        VisionQkvStackOverlayErrorCode::LayerSetOrOrder,
        || {
            build_verified_vision_qkv_stack_overlay(
                &graph,
                28,
                &geometry,
                &complete_catalog,
                limits(32),
            )
        },
    );

    let empty_graph = SemanticGraph::from_nodes(vec![]);
    assert_overlay_error(
        "missing semantic source",
        VisionQkvStackOverlayErrorCode::SemanticOrTensorIdentity,
        || {
            build_verified_vision_qkv_stack_overlay(
                &empty_graph,
                1,
                &geometry,
                &catalog,
                limits(32),
            )
        },
    );
    let mut missing = catalog.clone();
    missing.pop();
    assert_overlay_error(
        "missing tensor",
        VisionQkvStackOverlayErrorCode::SemanticOrTensorIdentity,
        || build_verified_vision_qkv_stack_overlay(&graph, 1, &geometry, &missing, limits(32)),
    );
    let mut duplicate = catalog.clone();
    duplicate.push(duplicate[0].clone());
    assert_overlay_error(
        "duplicate tensor",
        VisionQkvStackOverlayErrorCode::SemanticOrTensorIdentity,
        || build_verified_vision_qkv_stack_overlay(&graph, 1, &geometry, &duplicate, limits(32)),
    );
    let mut wrong_shape = catalog.clone();
    wrong_shape[0].shape[0] += 1;
    assert_overlay_error(
        "wrong tensor shape",
        VisionQkvStackOverlayErrorCode::SemanticOrTensorIdentity,
        || build_verified_vision_qkv_stack_overlay(&graph, 1, &geometry, &wrong_shape, limits(32)),
    );

    assert_eq!(graph, graph_before);
    assert_eq!(catalog, catalog_before);
}

#[test]
fn unsupported_target_maxima_are_independently_classified_before_selection() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_catalog(0);
    let unsupported_targets = [
        VisionQkvFusedTargetLimits {
            min_storage_buffer_offset_alignment: 3,
            ..limits(32)
        },
        VisionQkvFusedTargetLimits {
            max_storage_buffers_per_shader_stage: 7,
            ..limits(32)
        },
        VisionQkvFusedTargetLimits {
            max_storage_buffer_binding_size: 191,
            ..limits(32)
        },
        VisionQkvFusedTargetLimits {
            max_buffer_size: 191,
            ..limits(32)
        },
        VisionQkvFusedTargetLimits {
            max_compute_workgroups_per_dimension: 2,
            ..limits(32)
        },
    ];
    for target in unsupported_targets {
        assert_overlay_error(
            "unsupported target limit",
            VisionQkvStackOverlayErrorCode::UnsupportedTarget,
            || build_verified_vision_qkv_stack_overlay(&graph, 1, &geometry, &catalog, target),
        );
    }
}

#[test]
fn public_selection_is_lazy_and_only_unsupported_target_falls_back() {
    let graph = canonical_graph();
    let geometry = compact_layer_plan();
    let catalog = compact_catalog(0);

    let disabled_calls = Cell::new(0);
    let invalid_graph = SemanticGraph::from_nodes(vec![]);
    let disabled = select_vision_qkv_stack_overlay(VisionQkvExecutionPolicy::Disabled, || {
        disabled_calls.set(disabled_calls.get() + 1);
        build_verified_vision_qkv_stack_overlay(&invalid_graph, 1, &geometry, &[], limits(32))
    })
    .unwrap();
    assert_eq!(disabled_calls.get(), 0);
    assert_eq!(disabled.policy(), VisionQkvExecutionPolicy::Disabled);
    assert_eq!(disabled.outcome(), VisionQkvSelectionOutcome::Disabled);
    assert!(disabled.overlay().is_none());
    assert_eq!(disabled.fallback_error_code(), None);

    for policy in [
        VisionQkvExecutionPolicy::Preferred,
        VisionQkvExecutionPolicy::Required,
    ] {
        let calls = Cell::new(0);
        let selected = select_vision_qkv_stack_overlay(policy, || {
            calls.set(calls.get() + 1);
            build_verified_vision_qkv_stack_overlay(&graph, 1, &geometry, &catalog, limits(32))
        })
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(selected.policy(), policy);
        assert_eq!(selected.outcome(), VisionQkvSelectionOutcome::Fused);
        assert_eq!(selected.overlay().unwrap().layer_count(), 1);
        assert_eq!(selected.fallback_error_code(), None);
    }

    let unsupported = VisionQkvFusedTargetLimits {
        max_storage_buffers_per_shader_stage: 7,
        ..limits(32)
    };
    let calls = Cell::new(0);
    let fallback = select_vision_qkv_stack_overlay(VisionQkvExecutionPolicy::Preferred, || {
        calls.set(calls.get() + 1);
        build_verified_vision_qkv_stack_overlay(&graph, 1, &geometry, &catalog, unsupported)
    })
    .unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(fallback.policy(), VisionQkvExecutionPolicy::Preferred);
    assert_eq!(
        fallback.outcome(),
        VisionQkvSelectionOutcome::FallbackUnsupportedTarget
    );
    assert!(fallback.overlay().is_none());
    assert_eq!(
        fallback.fallback_error_code(),
        Some(VisionQkvStackOverlayErrorCode::UnsupportedTarget)
    );

    assert_overlay_error(
        "Required unsupported target",
        VisionQkvStackOverlayErrorCode::UnsupportedTarget,
        || {
            select_vision_qkv_stack_overlay(VisionQkvExecutionPolicy::Required, || {
                build_verified_vision_qkv_stack_overlay(&graph, 1, &geometry, &catalog, unsupported)
            })
        },
    );

    for policy in [
        VisionQkvExecutionPolicy::Preferred,
        VisionQkvExecutionPolicy::Required,
    ] {
        assert_overlay_error(
            "semantic failures never fall back",
            VisionQkvStackOverlayErrorCode::SemanticOrTensorIdentity,
            || {
                select_vision_qkv_stack_overlay(policy, || {
                    build_verified_vision_qkv_stack_overlay(
                        &invalid_graph,
                        1,
                        &geometry,
                        &catalog,
                        limits(32),
                    )
                })
            },
        );
    }
}
