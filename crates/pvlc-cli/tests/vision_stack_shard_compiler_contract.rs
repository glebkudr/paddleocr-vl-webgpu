mod support;

use std::{fs, path::Path, process::Command};

use pvlc_cli::{
    CompileOfficialVisionStackOptions, OfficialVisionStackProfile, SourceErrorCode,
    VisionStackTensorCatalog, compile_official_vision_stack_shards,
    official_vision_stack_tensor_program,
};
use pvlc_pack::{
    OFFICIAL_VISION_STACK_CASE_ID, VisionStackShardKind, VisionStackShardOracle,
    VisionStackShardProtocol, inspect_vision_stack_f32_shard, parse_vision_stack_shard_manifest,
};
use support::official_vision_stack_shard_anchors;

const COMPILER_BUILD: &str = "4e40a34cd64f347a14270d339ab61b131ae3120ca566ff140c81d101a3aa4a2c";
const EXPECTED_CHECKPOINT_BYTES: u64 = 29_399_040;
const EXPECTED_CHECKPOINT_BLAKE3: &str =
    "6949b4d783f2a65f653e52f9f5dc29380834bb2c5eee5a7d646b07abd70c3f4a";

#[test]
fn official_tensor_program_covers_exact_catalogs_shapes_and_parameter_order() {
    let program = official_vision_stack_tensor_program();
    assert_eq!(program.shards.len(), 29);
    assert_eq!(program.shards[0].id, "input.embeddings");
    assert_eq!(program.shards[0].kind, VisionStackShardKind::Input);
    assert_eq!(program.shards[0].tensors.len(), 1);
    assert_eq!(
        program.shards[0].tensors[0].catalog,
        VisionStackTensorCatalog::DeepCheckpoints
    );
    assert_eq!(
        program.shards[0].tensors[0].name,
        "vision.embeddings.output"
    );
    assert_eq!(program.shards[0].tensors[0].shape, [1, 1_276, 1_152]);

    let layer_suffixes = [
        "layer_norm1.weight",
        "layer_norm1.bias",
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.out_proj.weight",
        "self_attn.out_proj.bias",
        "layer_norm2.weight",
        "layer_norm2.bias",
        "mlp.fc1.weight",
        "mlp.fc1.bias",
        "mlp.fc2.weight",
        "mlp.fc2.bias",
    ];
    for layer in 0..27 {
        let shard = &program.shards[layer + 1];
        assert_eq!(shard.id, format!("weights.vision_layer.{layer:02}"));
        assert_eq!(shard.kind, VisionStackShardKind::Layer);
        assert_eq!(shard.layer_index, Some(layer as u32));
        assert_eq!(shard.tensors.len(), layer_suffixes.len());
        for (tensor, suffix) in shard.tensors.iter().zip(layer_suffixes) {
            assert_eq!(tensor.catalog, VisionStackTensorCatalog::Model);
            assert_eq!(
                tensor.name,
                format!("visual.vision_model.encoder.layers.{layer}.{suffix}")
            );
        }
        let shapes = shard
            .tensors
            .iter()
            .map(|tensor| tensor.shape.as_slice())
            .collect::<Vec<_>>();
        assert_eq!(
            shapes,
            [
                &[1_152][..],
                &[1_152],
                &[1_152, 1_152],
                &[1_152],
                &[1_152, 1_152],
                &[1_152],
                &[1_152, 1_152],
                &[1_152],
                &[1_152, 1_152],
                &[1_152],
                &[1_152],
                &[1_152],
                &[4_304, 1_152],
                &[4_304],
                &[1_152, 4_304],
                &[1_152],
            ]
        );
    }
    let post = &program.shards[28];
    assert_eq!(post.id, "weights.vision_post_norm");
    assert_eq!(post.kind, VisionStackShardKind::PostNorm);
    assert_eq!(post.layer_index, None);
    assert!(
        post.tensors
            .iter()
            .all(|tensor| tensor.catalog == VisionStackTensorCatalog::Model)
    );
    assert_eq!(
        post.tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>(),
        [
            "visual.vision_model.post_layernorm.weight",
            "visual.vision_model.post_layernorm.bias",
        ]
    );
    assert_eq!(
        post.tensors
            .iter()
            .map(|tensor| tensor.shape.as_slice())
            .collect::<Vec<_>>(),
        [&[1_152][..], &[1_152],]
    );
    assert_eq!(
        program
            .expected
            .iter()
            .map(|tensor| (tensor.catalog, tensor.name.as_str()))
            .collect::<Vec<_>>(),
        [
            (
                VisionStackTensorCatalog::DeepCheckpoints,
                "vision.layer.00.output"
            ),
            (
                VisionStackTensorCatalog::DeepCheckpoints,
                "vision.layer.01.output"
            ),
            (
                VisionStackTensorCatalog::DeepCheckpoints,
                "vision.layer.13.output"
            ),
            (
                VisionStackTensorCatalog::DeepCheckpoints,
                "vision.layer.26.output"
            ),
            (VisionStackTensorCatalog::StageCheckpoints, "vision.final"),
        ]
    );
    assert_eq!(
        program
            .expected
            .iter()
            .map(|tensor| tensor.shape.as_slice())
            .collect::<Vec<_>>(),
        [
            &[1, 1_276, 1_152][..],
            &[1, 1_276, 1_152],
            &[1, 1_276, 1_152],
            &[1, 1_276, 1_152],
            &[1_276, 1_152],
        ]
    );
}

#[test]
fn invalid_compiler_build_is_rejected_before_any_source_or_output_access() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("existing-output");
    fs::create_dir(&output).unwrap();
    fs::write(output.join("user-sentinel"), b"preserve me").unwrap();
    let missing = directory.path().join("missing");
    let error = compile_official_vision_stack_shards(
        &missing,
        &missing,
        &missing,
        &output,
        &CompileOfficialVisionStackOptions {
            compiler_build: "invalid".to_owned(),
            profile: OfficialVisionStackProfile::OcrCleanLatinL3,
        },
    )
    .unwrap_err();
    assert_eq!(error.code(), SourceErrorCode::InvalidCompilerBuild);
    assert_eq!(
        fs::read(output.join("user-sentinel")).unwrap(),
        b"preserve me"
    );
    assert_eq!(fs::read_dir(&output).unwrap().count(), 1);
}

#[test]
fn cli_exposes_official_shard_compilation_and_rejects_build_before_output_access() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("existing-output");
    fs::create_dir(&output).unwrap();
    fs::write(output.join("user-sentinel"), b"preserve me").unwrap();
    let missing = directory.path().join("missing");
    let result = Command::new(env!("CARGO_BIN_EXE_pvlc"))
        .args([
            "compile-official-vision-stack-shards",
            "--profile",
            "ocr-clean-latin-l3",
            "--lock",
            missing.to_str().unwrap(),
            "--model-dir",
            missing.to_str().unwrap(),
            "--golden-dir",
            missing.to_str().unwrap(),
            "--output-dir",
            output.to_str().unwrap(),
            "--compiler-build",
            "invalid",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8(result.stderr)
            .unwrap()
            .contains("InvalidCompilerBuild")
    );
    assert_eq!(
        fs::read(output.join("user-sentinel")).unwrap(),
        b"preserve me"
    );
    assert_eq!(fs::read_dir(&output).unwrap().count(), 1);
}

#[test]
#[ignore = "writes and independently rehashes all 1.65 GB of pinned official vision shards"]
fn official_compiler_emits_every_independently_anchored_shard_and_checkpoint() {
    assert_eq!(std::env::var("PVLC_REQUIRE_MODEL").as_deref(), Ok("1"));
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output_parent = root.join("output/compiler-tests");
    fs::create_dir_all(&output_parent).unwrap();
    let output = tempfile::tempdir_in(output_parent).unwrap();
    let report = compile_official_vision_stack_shards(
        root.join("models/paddleocr-vl-1.6.lock"),
        root.join("models/snapshots/66317acc4c9fc17bd154591ce650735cd2855f3e"),
        root.join("artifacts/goldens/ocr.clean_latin.0001-l3"),
        output.path(),
        &CompileOfficialVisionStackOptions {
            compiler_build: COMPILER_BUILD.to_owned(),
            profile: OfficialVisionStackProfile::OcrCleanLatinL3,
        },
    )
    .unwrap();
    assert_eq!(report.expected_bytes, EXPECTED_CHECKPOINT_BYTES);
    assert_eq!(report.expected_blake3, EXPECTED_CHECKPOINT_BLAKE3);

    let manifest_bytes = fs::read(output.path().join("manifest.json")).unwrap();
    let manifest = parse_vision_stack_shard_manifest(&manifest_bytes).unwrap();
    assert_eq!(manifest.oracle, VisionStackShardOracle::OfficialMpsBf16);
    assert_eq!(manifest.case_id, OFFICIAL_VISION_STACK_CASE_ID);
    assert_eq!(manifest.checkpoint_layers, [0, 1, 13, 26]);
    let anchors = official_vision_stack_shard_anchors();
    let mut protocol = VisionStackShardProtocol::new(manifest.clone()).unwrap();
    for descriptor in &manifest.shards {
        let (bytes, digest) = anchors[descriptor.id.as_str()];
        assert_eq!(
            (descriptor.bytes, descriptor.blake3.as_str()),
            (bytes, digest)
        );
        let payload = fs::read(output.path().join(format!("{}.f32", descriptor.id))).unwrap();
        protocol
            .accept_preflight(&inspect_vision_stack_f32_shard(&descriptor.id, &payload))
            .unwrap();
    }
    let expected = fs::read(output.path().join("expected.checkpoints.f32")).unwrap();
    assert_eq!(expected.len() as u64, EXPECTED_CHECKPOINT_BYTES);
    assert_eq!(
        blake3::hash(&expected).to_hex().as_str(),
        EXPECTED_CHECKPOINT_BLAKE3
    );
    assert_eq!(manifest.plan().unwrap().transport_bytes, 1_651_755_456);
}
