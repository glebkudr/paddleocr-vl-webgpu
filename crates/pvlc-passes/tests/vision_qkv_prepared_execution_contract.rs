//! Public M7c2b contract for the compiler-owned, adapter-neutral Q/K/V handoff.

mod support;

use std::panic::{AssertUnwindSafe, catch_unwind};

use pvlc_ir::SemanticGraph;
use pvlc_model_schema::PaddleOcrVl16Schema;
use pvlc_passes::{
    PreparedVisionQkvStackExecution, VisionQkvPhysicalBindingErrorCode,
    VisionQkvPhysicalExecutionSpec, VisionQkvPreparedExecutionError,
    VisionQkvPreparedExecutionErrorCode, bind_vision_qkv_physical_execution,
    build_verified_vision_qkv_stack_overlay, canonical_synthetic_vision_qkv_tensor_catalog,
    prepare_vision_qkv_stack_execution,
};
use pvlc_runtime_core::{
    KernelId, VISION_QKV_CANARY_U32, VisionQkvCanaryKind, VisionQkvFusedTargetLimits,
    VisionQkvReadbackLayout, VisionQkvReadbackRequirements, plan_vision_qkv_readback_layout,
};

use support::*;

const COMPACT_ALIGN32_LAYER_HASHES: [&str; 27] = [
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
const COMPACT_ALIGN256_LAYER_HASHES: [&str; 27] = [
    "361a0a1c71de94c6ec30aafd725edbff5ddac65ac0d01f66dc7668c4d1d8688b",
    "9d4496cd963f536ec8358e79ff66e2c848b53c246e6a8aba3352dbe333edec5a",
    "7e4de849fdc92ee3896d62ca9c19491990e035d27ccd3d7aee9c6f3a24bf15e5",
    "5e0794d6325bd5686dc67eb84a3ec71437ef2ee1bab55594c4ac9ffd6b61248a",
    "9a61f941824f5da32f159b9e4def115df1b2c12d4f9a112cc9486e597abc0039",
    "98ff559f1fc4c6f8c56ac1675def78e9f668fb091de35bdd3ff05e5ab1979769",
    "96fdfc703ff303c53d57ab1494f4d911c6e29b54588261e68aaf9e20de31e480",
    "aa761fc523a6d3c88110cdebb17acdad998ba0fe46bd023dfeee5fd13f444971",
    "79ad8a36c868b7ab754d2b04df32096e5c810b0dfb727000b332e8153f283ffb",
    "002bc70d5d513d39a0700c968af3721281a5ee79629248b09f9ee91b96e443c1",
    "d2fe25776c908937df8fcbfb5e9cf83e20456b9a78d061663ae628c451aa958d",
    "5c03b7edc76b96f0b7c660b6cdb827379710aab20160ed849a9fe56bb09b98b0",
    "552000f986e9af39dfd3f2b55a3ca9e58f6423dbcac54e0c49596481801a97a3",
    "fe36d9a685dc038325046d1938ed705aaf7498a7bb49db4583aef2ef75fbbe63",
    "437fd4f9f8d8e29fe2e941adf068ebe67a3548020256b4e6d0d8077a1c033f85",
    "f1acb7a25723f4c52eaaf49dc8e75a078133671a7ff3bba8edc212c92b999c88",
    "5e11476a7e12a65eb864fa8532da94f2e82a35d72e6491f969ef52fffcb5498a",
    "6cbef29c4841df2b0d1a20545ccfb7f78deac973fddd98d7430420ec0d08109a",
    "4d6b3daeac89ac20bac5fb939190df6f0ec1410f81be553879890b6e71aae537",
    "21cd00d525630c70b174f27dadb003ae29947b0843235aad907590e8c0deb12e",
    "a95739a1fa28df689eb466df067e877d18748a78cd8875d6adbd3a1269924ba0",
    "a6e5a3960e0719d832e37353fbd5fb3fb18728828bbf4910be38d46ec1e65071",
    "2e3cfea6c589b6b965fb4ec262a53ba83cdba98a6556ce00221ceadc0b01adf8",
    "ccd0d9f60b5742eb1e456d0d83b550319c6038b91c9f8a67c88ceeacec29becf",
    "0c4a5c3e836e2543a584cfc18255c2d12ac168a2839dc6f5fe4efca0d04f21d4",
    "76dd112de1efbfe65acecdabc2ddd76d9e82cd035e6f2fcac34a549f8a6df37a",
    "ea60cde4672c2edc6bbe4e462ef83282705becf91de702d44170435ac3498c85",
];
const OFFICIAL_ALIGN32_LAYER_HASHES: [&str; 27] = [
    "26af4d09c5e4015a58967095466420d7288169fdb3426ec60e78a3980de94b34",
    "b71969eed0cf2987638a3d5620358a3341411e8087a04fc2a0423f5851c2905b",
    "ba30b83e6943cfc10f9c3d7914f4f517a10e80bb1f5cfd986169281293d45ebb",
    "a992481d7ef4999657f49826507e559ad89157a07e59fc1f99fe76275019e56c",
    "a967d501777af856d9fbf5d5365d08436d94b0e8f611ef011e79aef1b4c8d133",
    "e25a9b816d1fc220534b118b4b9c49efbf6c8ae02ce2fb89e8b4baaa2bcfa6d8",
    "028384ffb95bf939efa43b9295fc5e4baa68ed353c69dabe2fce6f84cdbae6a2",
    "1dcb5801a275dc7a13e4fe0b44600bf91ec6ee0971d40b9fc2b5b1247500df85",
    "956fda484a03df097e432d52590541ca77f618b91254b122388d130014790374",
    "b0e7d5bf46b357d7eef96e1b40fc33888fb452ee627f63f21efb14e7d752c1d1",
    "67e1aa8d244b89d5b2db83e7ae916ded10498fab71a7d1e11f28850d25f7c4d8",
    "5c18b7c28beefb6f8242677531aefa03b9c808e9d8b342edbe9e95655183d36f",
    "7ab8ae86fbfbde7ea8d371c5d522303d927752b6a5e2ffb14dd9009ad7baa839",
    "cacf38e4a067b140675a568df3a9976ea07747d59d379c37b593b1dff1458c69",
    "336985033c3fc302e597c906f6a958b0bc473f1bd229093617790eeded82822d",
    "048dedbbd815b6b46766a3004cead2cb64cc6d153e36cf26ee19736099407e52",
    "497610835dd6e1672cd5f4efc96f0430516d908927604f9d4b02df7d1fec3476",
    "a50b998ad72e42cdb9c3d75616feb1d72dcd735f1caa44cf8d22a9418c82e0d9",
    "3e16364699dda512484c37f5b1ad58ef428f3b48fd2c38abd0f5c7625bd9daf3",
    "5fee41bdc682ba2ad6a5fa96a9ebe088e936cbb2f19b98c599c41dd4cdd2d1a2",
    "37bfe43a1da5202778d27ea19590d63f30a3b9356e04fd8191998591de783ae9",
    "cacbd825c3fac2c74b9fb7d0a51f7a411e1d1d81a3cdf7a989198d34a9415c97",
    "aee7ff44a8f32dd26660ad5d6082d77e1ff74549e7d6f2a20f48138c8d0771fb",
    "35aa3a90611e8dfc2030549ebab7b172c435ad5a2f9977ed9f700ab4391b6d8f",
    "5e2f887af220619f45910e48f3e9ac987ecb6913160ca73e3ae691b85183fc29",
    "30345811b3fb056031af0388ba6933ad0c69d095c6cb3acdfedc289532260b2a",
    "40c9edfb339c0b3a2e3fce4a1d440fb4e8adecb0292b7823f331ba5f15bfe8d8",
];
const OFFICIAL_ALIGN256_LAYER_HASHES: [&str; 27] = [
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

fn compact_stack_catalog(depth: usize) -> Vec<pvlc_model_schema::TensorSpec> {
    (0..depth).flat_map(compact_catalog).collect()
}

fn prepare_compact(depth: usize, alignment: u32) -> PreparedVisionQkvStackExecution {
    let geometry = compact_layer_plan();
    let overlay = build_verified_vision_qkv_stack_overlay(
        &canonical_graph(),
        depth,
        &geometry,
        &canonical_synthetic_vision_qkv_tensor_catalog(depth, OUTPUT_WIDTH)
            .expect("canonical compact synthetic catalog"),
        limits(alignment),
    )
    .expect("compact overlay");
    prepare_vision_qkv_stack_execution(&overlay, depth, &geometry, limits(alignment))
        .expect("compact prepared execution")
}

fn physical_readback_layout(
    prepared: &PreparedVisionQkvStackExecution,
    semantic_readback_bytes: u64,
    scratch_canary_readback_bytes: u64,
) -> VisionQkvReadbackLayout {
    let qkv_canary_readback_bytes = prepared
        .workspace()
        .canaries()
        .iter()
        .map(|range| range.byte_length())
        .sum();
    let workspace_allocation_bytes = prepared.workspace().allocation_bytes();
    plan_vision_qkv_readback_layout(VisionQkvReadbackRequirements {
        semantic_readback_bytes,
        scratch_canary_readback_bytes,
        qkv_canary_readback_bytes,
        workspace_allocation_bytes,
        max_buffer_size: 1_u64 << 34,
        max_host_elements: 1_u64 << 32,
    })
    .expect("physical-spec fixture has a valid core layout")
}

fn assert_physical_spec_type(_: &VisionQkvPhysicalExecutionSpec) {}

fn assert_prepared_error(
    label: &str,
    expected: VisionQkvPreparedExecutionErrorCode,
    call: impl FnOnce() -> Result<PreparedVisionQkvStackExecution, VisionQkvPreparedExecutionError>,
) {
    let outcome = catch_unwind(AssertUnwindSafe(call));
    let result = outcome.unwrap_or_else(|_| {
        panic!("{label}: preparation panicked instead of returning {expected:?}")
    });
    match result {
        Err(error) => assert_eq!(error.code(), expected, "{label}: {error}"),
        Ok(_) => panic!("{label}: invalid handoff was accepted"),
    }
}

#[test]
fn canonical_synthetic_catalog_is_exact_deterministic_and_separate_from_official_catalog() {
    for depth in [1, 3, 16] {
        let first = canonical_synthetic_vision_qkv_tensor_catalog(depth, OUTPUT_WIDTH)
            .expect("valid synthetic catalog");
        let second = canonical_synthetic_vision_qkv_tensor_catalog(depth, OUTPUT_WIDTH)
            .expect("repeat synthetic catalog");
        assert_eq!(first, second);
        assert_eq!(first, compact_stack_catalog(depth));
        assert_eq!(first.len(), depth * 6);
        assert!(
            first
                .iter()
                .all(|tensor| tensor.shape == vec![3, 3] || tensor.shape == vec![3])
        );
    }

    let synthetic_official_width = canonical_synthetic_vision_qkv_tensor_catalog(27, 1_152)
        .expect("shape-authenticated synthetic official-width catalog");
    let official = PaddleOcrVl16Schema::tensor_specs();
    assert_ne!(synthetic_official_width, official);
    assert!(official.len() > synthetic_official_width.len());

    for (depth, hidden) in [(0, 3), (28, 3), (1, 0)] {
        assert!(
            canonical_synthetic_vision_qkv_tensor_catalog(depth, hidden).is_err(),
            "invalid synthetic geometry {depth}/{hidden} was accepted",
        );
    }
}

#[test]
fn prepared_compact_depths_are_owned_exact_and_deterministic_at_both_alignments() {
    assert_eq!(VISION_QKV_CANARY_U32, 0x7fc0_51a7);
    for depth in [1, 3, 16, 27] {
        for alignment in [32, 256] {
            let first = prepare_compact(depth, alignment);
            let second = prepare_compact(depth, alignment);
            let oracle = independent_layout(alignment);
            let expected_hashes = match alignment {
                32 => &COMPACT_ALIGN32_LAYER_HASHES,
                256 => &COMPACT_ALIGN256_LAYER_HASHES,
                _ => unreachable!(),
            };
            assert_eq!(first, second, "depth {depth}, alignment {alignment}");
            assert_eq!(first.layer_count(), depth);
            assert_eq!(first.layers().len(), depth);
            for (index, layer) in first.layers().iter().enumerate() {
                assert_eq!(layer.layer_index(), index);
                assert_eq!(
                    layer.canonical_plan_blake3_hex(),
                    expected_hashes[index],
                    "depth {depth}, alignment {alignment}, layer {index}",
                );
                assert_eq!(layer.invocation().kernel, KernelId::VisionQkvFusedF32);
                assert_eq!(
                    layer.invocation().output_elements,
                    usize::try_from(oracle.physical_bytes / 4).unwrap(),
                );
                assert_eq!(layer.invocation().output_bytes, oracle.physical_bytes);
                assert_eq!(layer.invocation().workgroup_size, [8, 8, 1]);
                assert_eq!(layer.invocation().dispatch, [1, 1, 3]);
                assert_eq!(layer.uniform_words(), oracle.uniform_words);
                assert_eq!(layer.shared_output_bytes(), oracle.physical_bytes);
            }
        }
    }
}

#[test]
fn physical_execution_spec_moves_and_preserves_only_the_exact_prepared_and_core_layout_authority() {
    for depth in [1, 3] {
        for alignment in [32, 256] {
            for (semantic_bytes, scratch_bytes) in [(16, 8), (28, 20)] {
                let prepared = prepare_compact(depth, alignment);
                let layout = physical_readback_layout(&prepared, semantic_bytes, scratch_bytes);
                let expected_prepared = prepared.clone();
                let expected_layout = layout.clone();
                let spec = bind_vision_qkv_physical_execution(prepared, layout)
                    .expect("matching prepared/layout authority must bind");
                assert_physical_spec_type(&spec);
                assert_eq!(
                    spec.prepared_execution(),
                    &expected_prepared,
                    "depth {depth}, alignment {alignment}: prepared authority was reconstructed",
                );
                assert_eq!(
                    spec.readback_layout(),
                    &expected_layout,
                    "depth {depth}, alignment {alignment}: core layout was reconstructed",
                );
                assert_eq!(spec.prepared_execution().layer_count(), depth);
                assert_eq!(
                    spec.prepared_execution().workspace().semantic_base(),
                    u64::from(alignment),
                );
                assert_eq!(
                    spec.readback_layout().semantic_readback_bytes(),
                    semantic_bytes,
                );
                assert_eq!(
                    spec.readback_layout().scratch_canary_readback_bytes(),
                    scratch_bytes,
                );
                assert_eq!(
                    spec.readback_layout().workspace_allocation_bytes(),
                    expected_prepared.workspace().allocation_bytes(),
                );
            }
        }
    }
}

#[test]
fn physical_execution_spec_rejects_incompatible_workspace_and_qkv_canary_layouts_stably() {
    let prepared = prepare_compact(3, 256);
    let expected_qkv_canary_bytes = prepared
        .workspace()
        .canaries()
        .iter()
        .map(|range| range.byte_length())
        .sum::<u64>();
    let expected_workspace_bytes = prepared.workspace().allocation_bytes();
    for (label, expected, qkv_canary_readback_bytes, workspace_allocation_bytes) in [
        (
            "workspace allocation mismatch",
            VisionQkvPhysicalBindingErrorCode::WorkspaceAllocationMismatch,
            expected_qkv_canary_bytes,
            expected_workspace_bytes + 4,
        ),
        (
            "QKV canary readback mismatch",
            VisionQkvPhysicalBindingErrorCode::QkvCanaryReadbackMismatch,
            expected_qkv_canary_bytes + 4,
            expected_workspace_bytes,
        ),
    ] {
        let layout = plan_vision_qkv_readback_layout(VisionQkvReadbackRequirements {
            semantic_readback_bytes: 16,
            scratch_canary_readback_bytes: 8,
            qkv_canary_readback_bytes,
            workspace_allocation_bytes,
            max_buffer_size: 1_u64 << 34,
            max_host_elements: 1_u64 << 32,
        })
        .expect("hostile pairing still has an independently valid core layout");
        let error = bind_vision_qkv_physical_execution(prepared.clone(), layout)
            .expect_err("incompatible opaque authorities must not bind");
        assert_eq!(error.code(), expected, "{label}: {error}");
    }
}

#[test]
fn compact_prepared_workspace_and_attention_slices_are_literal_for_32_and_256() {
    for (alignment, semantic_bytes, allocation_bytes, expected_ranges) in [
        (
            32,
            192,
            256,
            vec![
                (VisionQkvCanaryKind::Prefix, 0, 32),
                (VisionQkvCanaryKind::InternalPadding { plane: 0 }, 68, 28),
                (VisionQkvCanaryKind::InternalPadding { plane: 1 }, 132, 28),
                (VisionQkvCanaryKind::InternalPadding { plane: 2 }, 196, 28),
                (VisionQkvCanaryKind::Suffix, 224, 32),
            ],
        ),
        (
            256,
            768,
            1_280,
            vec![
                (VisionQkvCanaryKind::Prefix, 0, 256),
                (VisionQkvCanaryKind::InternalPadding { plane: 0 }, 292, 220),
                (VisionQkvCanaryKind::InternalPadding { plane: 1 }, 548, 220),
                (VisionQkvCanaryKind::InternalPadding { plane: 2 }, 804, 220),
                (VisionQkvCanaryKind::Suffix, 1_024, 256),
            ],
        ),
    ] {
        let prepared = prepare_compact(3, alignment);
        let oracle = independent_layout(alignment);
        let workspace = prepared.workspace();
        assert_eq!(workspace.semantic_base(), u64::from(alignment));
        assert_eq!(workspace.semantic_bytes(), semantic_bytes);
        assert_eq!(workspace.allocation_bytes(), allocation_bytes);
        assert_eq!(
            workspace.canary_readback_bytes(),
            allocation_bytes - 3 * PLANE_BYTES
        );
        let actual = workspace
            .canaries()
            .iter()
            .map(|range| (range.kind(), range.byte_offset(), range.byte_length()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected_ranges);

        for layer in prepared.layers() {
            for (((binding, role), expected_binding), expected_offset) in layer
                .attention_bridge()
                .bindings()
                .iter()
                .zip(Role::ALL)
                .zip([0, 1, 2])
                .zip(oracle.offsets)
            {
                assert_eq!(binding.binding(), expected_binding);
                assert_eq!(binding.value_id(), output_value(layer.layer_index(), role));
                assert_eq!(
                    binding.buffer_id(),
                    shared_output_buffer(layer.layer_index())
                );
                assert_eq!(binding.byte_offset(), expected_offset);
                assert_eq!(binding.byte_length(), PLANE_BYTES);
                assert_ne!(binding.byte_length(), workspace.semantic_bytes());
                assert_eq!(
                    workspace.semantic_base() + binding.byte_offset(),
                    u64::from(alignment) + expected_offset,
                );
            }
        }
    }
}

#[test]
fn official_27_layer_prepared_view_uses_full_schema_and_all_ordered_identities() {
    let graph = SemanticGraph::paddleocr_vl_16();
    let geometry = official_layer_plan();
    let official = PaddleOcrVl16Schema::tensor_specs();
    let before = official.clone();
    for alignment in [32, 256] {
        let expected_hashes = match alignment {
            32 => &OFFICIAL_ALIGN32_LAYER_HASHES,
            256 => &OFFICIAL_ALIGN256_LAYER_HASHES,
            _ => unreachable!(),
        };
        let overlay = build_verified_vision_qkv_stack_overlay(
            &graph,
            27,
            &geometry,
            &official,
            limits(alignment),
        )
        .expect("official overlay");
        let prepared =
            prepare_vision_qkv_stack_execution(&overlay, 27, &geometry, larger_limits(alignment))
                .expect("official prepared view");
        assert_eq!(prepared.layer_count(), 27);
        assert_eq!(prepared.layers()[0].layer_index(), 0);
        assert_eq!(prepared.layers()[26].layer_index(), 26);
        for (index, layer) in prepared.layers().iter().enumerate() {
            assert_eq!(layer.layer_index(), index);
            assert_eq!(layer.canonical_plan_blake3_hex(), expected_hashes[index]);
            assert_eq!(layer.invocation().kernel, KernelId::VisionQkvFusedF32);
            assert_eq!(layer.invocation().output_elements, 3_456);
            assert_eq!(layer.invocation().output_bytes, 13_824);
            assert_eq!(layer.invocation().workgroup_size, [8, 8, 1]);
            assert_eq!(layer.invocation().dispatch, [144, 1, 3]);
            assert_eq!(layer.uniform_words(), [1, 1_152, 1_152, 1_152]);
            assert_eq!(layer.shared_output_bytes(), 13_824);
            for (((binding, role), expected_binding), expected_offset) in layer
                .attention_bridge()
                .bindings()
                .iter()
                .zip(Role::ALL)
                .zip([0, 1, 2])
                .zip([0, 4_608, 9_216])
            {
                assert_eq!(binding.binding(), expected_binding);
                assert_eq!(binding.value_id(), output_value(index, role));
                assert_eq!(binding.buffer_id(), shared_output_buffer(index));
                assert_eq!(binding.byte_offset(), expected_offset);
                assert_eq!(binding.byte_length(), 4_608);
            }
        }
        assert_eq!(prepared.workspace().semantic_bytes(), 13_824);
        assert_eq!(prepared.workspace().semantic_base(), u64::from(alignment));
        assert_eq!(
            prepared.workspace().allocation_bytes(),
            13_824 + 2 * u64::from(alignment)
        );
        assert_eq!(prepared.workspace().canaries().len(), 2);
    }
    assert_eq!(official, before, "official schema catalog input mutated");
}

#[test]
fn official_dominant_weight_accepts_exact_binding_and_buffer_limits_separately() {
    const OFFICIAL_DOMINANT_WEIGHT_BYTES: u64 = 1_152 * 1_152 * 4;
    assert_eq!(OFFICIAL_DOMINANT_WEIGHT_BYTES, 5_308_416);

    let graph = SemanticGraph::paddleocr_vl_16();
    let geometry = official_layer_plan();
    let official = PaddleOcrVl16Schema::tensor_specs();
    let overlay =
        build_verified_vision_qkv_stack_overlay(&graph, 27, &geometry, &official, limits(32))
            .expect("official overlay");
    let exact = VisionQkvFusedTargetLimits {
        max_storage_buffer_binding_size: OFFICIAL_DOMINANT_WEIGHT_BYTES,
        max_buffer_size: OFFICIAL_DOMINANT_WEIGHT_BYTES,
        ..limits(32)
    };
    prepare_vision_qkv_stack_execution(&overlay, 27, &geometry, exact)
        .expect("the exact official dominant-weight boundary must be accepted");

    assert_prepared_error(
        "one-less official binding-size limit",
        VisionQkvPreparedExecutionErrorCode::TargetBindingSize,
        || {
            prepare_vision_qkv_stack_execution(
                &overlay,
                27,
                &geometry,
                VisionQkvFusedTargetLimits {
                    max_storage_buffer_binding_size: OFFICIAL_DOMINANT_WEIGHT_BYTES - 1,
                    ..exact
                },
            )
        },
    );
    assert_prepared_error(
        "one-less official buffer-size limit",
        VisionQkvPreparedExecutionErrorCode::TargetBufferSize,
        || {
            prepare_vision_qkv_stack_execution(
                &overlay,
                27,
                &geometry,
                VisionQkvFusedTargetLimits {
                    max_buffer_size: OFFICIAL_DOMINANT_WEIGHT_BYTES - 1,
                    ..exact
                },
            )
        },
    );

    prepare_vision_qkv_stack_execution(&overlay, 27, &geometry, larger_limits(32))
        .expect("larger compatible official maxima must remain reusable");
}

#[test]
fn prepared_view_accepts_sufficient_maxima_but_rejects_depth_alignment_and_limits() {
    let geometry = compact_layer_plan();
    let overlay = build_verified_vision_qkv_stack_overlay(
        &canonical_graph(),
        3,
        &geometry,
        &compact_stack_catalog(3),
        limits(32),
    )
    .unwrap();
    prepare_vision_qkv_stack_execution(&overlay, 3, &geometry, larger_limits(32))
        .expect("larger sufficient maxima are compatible");

    assert_prepared_error(
        "cross-depth reuse",
        VisionQkvPreparedExecutionErrorCode::LayerSetOrOrder,
        || prepare_vision_qkv_stack_execution(&overlay, 2, &geometry, limits(32)),
    );
    assert_prepared_error(
        "stale alignment",
        VisionQkvPreparedExecutionErrorCode::TargetAlignment,
        || prepare_vision_qkv_stack_execution(&overlay, 3, &geometry, limits(256)),
    );
    for (expected, target) in [
        (
            VisionQkvPreparedExecutionErrorCode::TargetStorageBindings,
            VisionQkvFusedTargetLimits {
                max_storage_buffers_per_shader_stage: 7,
                ..limits(32)
            },
        ),
        (
            VisionQkvPreparedExecutionErrorCode::TargetBindingSize,
            VisionQkvFusedTargetLimits {
                max_storage_buffer_binding_size: 191,
                ..limits(32)
            },
        ),
        (
            VisionQkvPreparedExecutionErrorCode::TargetBufferSize,
            VisionQkvFusedTargetLimits {
                max_buffer_size: 255,
                ..limits(32)
            },
        ),
        (
            VisionQkvPreparedExecutionErrorCode::TargetDispatchLimit,
            VisionQkvFusedTargetLimits {
                max_compute_workgroups_per_dimension: 0,
                ..limits(32)
            },
        ),
    ] {
        assert_prepared_error("insufficient target", expected, || {
            prepare_vision_qkv_stack_execution(&overlay, 3, &geometry, target)
        });
    }
}

#[test]
fn prepared_view_is_fully_owned_and_does_not_expose_the_fused_replanner_boundary() {
    let prepared = {
        let geometry = compact_layer_plan();
        let overlay = build_verified_vision_qkv_stack_overlay(
            &canonical_graph(),
            3,
            &geometry,
            &compact_stack_catalog(3),
            limits(32),
        )
        .unwrap();
        prepare_vision_qkv_stack_execution(&overlay, 3, &geometry, limits(32)).unwrap()
    };
    assert_eq!(prepared.layers()[2].layer_index(), 2);

    let passes_source = include_str!("../src/vision_qkv_stack.rs");
    let native_source = include_str!("../../pvlc-runtime-native/src/lib.rs");
    assert!(
        !passes_source.contains("plan_vision_qkv_fused_geometry"),
        "shared preparation must validate the verified descriptor without invoking the fused planner",
    );
    assert!(
        native_source.contains("prepare_vision_qkv_stack_execution("),
        "native must consume the same passes-owned prepared view",
    );
    assert!(
        !native_source.contains("struct PreparedVisionQkvWorkspace"),
        "native must not retain its divergent local workspace planner",
    );
}
