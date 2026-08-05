mod support;

use std::{fs, path::Path, process::Command};

use pvlc_cli::{
    CompileOfficialVisionStackOptions, OfficialVisionStackProfile, SourceErrorCode,
    VisionStackInputMaterialization, VisionStackTensorCatalog,
    compile_official_vision_stack_shards, official_vision_stack_tensor_program_for,
};
use pvlc_pack::{
    VisionStackShardOracle, VisionStackShardProtocol, inspect_vision_stack_f32_shard,
    parse_vision_stack_shard_manifest,
};
use support::official_vision_stack_shard_anchors;

const COMPILER_BUILD: &str = "4e40a34cd64f347a14270d339ab61b131ae3120ca566ff140c81d101a3aa4a2c";
const TABLE_CASE_ID: &str = "table.simple.0001/vision.stack.27";
const TABLE_BUNDLE_DIGEST: &str =
    "blake3:4cecc8b2abc37030920acf8860bb317084125dbbbcae5af394fccd0c8044c842";
const TABLE_SEMANTIC_FINGERPRINT: &str =
    "blake3:91f72ffdf81070c2c9ca31628605064bdc02bf19417482cefce41d6a27aa5404";
const TABLE_INPUT_BYTES: u64 = 8_017_920;
const TABLE_INPUT_BLAKE3: &str = "645e12596caffcd4b394202a1b790acbb51d242cf6f616c0bade5d012eece742";
const TABLE_EXPECTED_BYTES: u64 = 8_017_920;
const TABLE_EXPECTED_BLAKE3: &str =
    "fcd101b25a04e1b4e0984e5d712094630f11c22d4fc57abdf743e9fd7a79aed9";

#[test]
fn table_l2_program_materializes_input_from_pinned_pixels_patch_weights_and_grid() {
    let program =
        official_vision_stack_tensor_program_for(OfficialVisionStackProfile::TableSimpleL2);
    assert_eq!(program.shards.len(), 29);
    assert_eq!(program.shards[0].id, "input.embeddings");
    assert_eq!(
        program.input_materialization,
        VisionStackInputMaterialization::PatchProjectionWithInterpolatedPosition {
            channels: 3,
            patch_size: 14,
            source_height: 27,
            source_width: 27,
            image_grid_thw: [1, 30, 58],
        }
    );
    assert_eq!(
        program.shards[0]
            .tensors
            .iter()
            .map(|tensor| (
                tensor.catalog,
                tensor.name.as_str(),
                tensor.shape.as_slice()
            ))
            .collect::<Vec<_>>(),
        [
            (
                VisionStackTensorCatalog::Processor,
                "processor.pixel_values",
                &[1_740, 3, 14, 14][..],
            ),
            (
                VisionStackTensorCatalog::Model,
                "visual.vision_model.embeddings.patch_embedding.weight",
                &[1_152, 3, 14, 14],
            ),
            (
                VisionStackTensorCatalog::Model,
                "visual.vision_model.embeddings.patch_embedding.bias",
                &[1_152],
            ),
            (
                VisionStackTensorCatalog::Model,
                "visual.vision_model.embeddings.position_embedding.weight",
                &[729, 1_152],
            ),
        ]
    );
    assert_eq!(program.expected.len(), 1);
    assert_eq!(
        (
            program.expected[0].catalog,
            program.expected[0].name.as_str(),
            program.expected[0].shape.as_slice(),
        ),
        (
            VisionStackTensorCatalog::StageCheckpoints,
            "vision.final",
            &[1_740, 1_152][..],
        )
    );

    let l3 = official_vision_stack_tensor_program_for(OfficialVisionStackProfile::OcrCleanLatinL3);
    assert_eq!(
        l3.input_materialization,
        VisionStackInputMaterialization::CapturedEmbedding
    );
    assert_eq!(l3.shards[0].tensors.len(), 1);
    assert_eq!(l3.shards[0].tensors[0].name, "vision.embeddings.output");
    assert_eq!(l3.expected.len(), 5);
    assert_eq!(program.shards[1..], l3.shards[1..]);
}

#[test]
fn cli_requires_an_allowlisted_profile_before_touching_sources_or_output() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("existing-output");
    fs::create_dir(&output).unwrap();
    fs::write(output.join("user-sentinel"), b"preserve me").unwrap();
    let missing = directory.path().join("missing");
    let assert_output_untouched = || {
        assert_eq!(
            fs::read(output.join("user-sentinel")).unwrap(),
            b"preserve me"
        );
        assert_eq!(fs::read_dir(&output).unwrap().count(), 1);
    };

    let missing_profile = Command::new(env!("CARGO_BIN_EXE_pvlc"))
        .args([
            "compile-official-vision-stack-shards",
            "--lock",
            missing.to_str().unwrap(),
            "--model-dir",
            missing.to_str().unwrap(),
            "--golden-dir",
            missing.to_str().unwrap(),
            "--output-dir",
            output.to_str().unwrap(),
            "--compiler-build",
            COMPILER_BUILD,
        ])
        .output()
        .unwrap();
    assert!(!missing_profile.status.success());
    assert!(
        String::from_utf8(missing_profile.stderr)
            .unwrap()
            .contains("--profile")
    );
    assert_output_untouched();

    let invalid_build = Command::new(env!("CARGO_BIN_EXE_pvlc"))
        .args([
            "compile-official-vision-stack-shards",
            "--profile",
            "table-simple-l2",
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
    assert!(!invalid_build.status.success());
    assert!(
        String::from_utf8(invalid_build.stderr)
            .unwrap()
            .contains("InvalidCompilerBuild")
    );
    assert_output_untouched();

    let unknown_profile = Command::new(env!("CARGO_BIN_EXE_pvlc"))
        .args([
            "compile-official-vision-stack-shards",
            "--profile",
            "unreviewed-shape",
            "--lock",
            missing.to_str().unwrap(),
            "--model-dir",
            missing.to_str().unwrap(),
            "--golden-dir",
            missing.to_str().unwrap(),
            "--output-dir",
            output.to_str().unwrap(),
            "--compiler-build",
            COMPILER_BUILD,
        ])
        .output()
        .unwrap();
    assert!(!unknown_profile.status.success());
    let stderr = String::from_utf8(unknown_profile.stderr).unwrap();
    assert!(stderr.contains("ocr-clean-latin-l3"));
    assert!(stderr.contains("table-simple-l2"));
    assert_output_untouched();
}

#[test]
fn invalid_build_for_the_second_profile_is_rejected_before_any_io() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("missing");
    let error = compile_official_vision_stack_shards(
        &missing,
        &missing,
        &missing,
        directory.path().join("output"),
        &CompileOfficialVisionStackOptions {
            compiler_build: "invalid".to_owned(),
            profile: OfficialVisionStackProfile::TableSimpleL2,
        },
    )
    .unwrap_err();
    assert_eq!(error.code(), SourceErrorCode::InvalidCompilerBuild);
    assert!(!directory.path().join("output").exists());
}

#[test]
#[ignore = "authenticates the pinned model before proving a profile/golden mismatch is atomic"]
fn mismatched_profile_and_golden_leave_existing_output_untouched() {
    assert_eq!(std::env::var("PVLC_REQUIRE_MODEL").as_deref(), Ok("1"));
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("output");
    fs::create_dir(&output).unwrap();
    fs::write(output.join("user-sentinel"), b"preserve me").unwrap();
    let error = compile_official_vision_stack_shards(
        root.join("models/paddleocr-vl-1.6.lock"),
        root.join("models/snapshots/66317acc4c9fc17bd154591ce650735cd2855f3e"),
        root.join("artifacts/goldens/ocr.clean_latin.0001-l3"),
        &output,
        &CompileOfficialVisionStackOptions {
            compiler_build: COMPILER_BUILD.to_owned(),
            profile: OfficialVisionStackProfile::TableSimpleL2,
        },
    )
    .unwrap_err();
    assert_eq!(error.code(), SourceErrorCode::GoldenMismatch);
    assert_eq!(
        fs::read(output.join("user-sentinel")).unwrap(),
        b"preserve me"
    );
    assert!(!output.join("manifest.json").exists());
    assert_eq!(fs::read_dir(&output).unwrap().count(), 1);
}

#[test]
#[ignore = "materializes and independently rehashes all second-shape official shards"]
fn table_l2_compiler_emits_anchored_input_final_and_bounded_manifest() {
    assert_eq!(std::env::var("PVLC_REQUIRE_MODEL").as_deref(), Ok("1"));
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output_parent = root.join("output/compiler-tests");
    fs::create_dir_all(&output_parent).unwrap();
    let output = tempfile::tempdir_in(output_parent).unwrap();
    let report = compile_official_vision_stack_shards(
        root.join("models/paddleocr-vl-1.6.lock"),
        root.join("models/snapshots/66317acc4c9fc17bd154591ce650735cd2855f3e"),
        root.join("artifacts/goldens/table.simple.0001-l2"),
        output.path(),
        &CompileOfficialVisionStackOptions {
            compiler_build: COMPILER_BUILD.to_owned(),
            profile: OfficialVisionStackProfile::TableSimpleL2,
        },
    )
    .unwrap();
    assert_eq!(report.expected_bytes, TABLE_EXPECTED_BYTES);
    assert_eq!(report.expected_blake3, TABLE_EXPECTED_BLAKE3);

    let manifest_bytes = fs::read(output.path().join("manifest.json")).unwrap();
    let manifest = parse_vision_stack_shard_manifest(&manifest_bytes).unwrap();
    assert_eq!(manifest.oracle, VisionStackShardOracle::OfficialMpsBf16);
    assert_eq!(manifest.case_id, TABLE_CASE_ID);
    assert_eq!(
        manifest.golden_bundle_digest.as_deref(),
        Some(TABLE_BUNDLE_DIGEST)
    );
    assert_eq!(
        manifest.semantic_fingerprint.as_deref(),
        Some(TABLE_SEMANTIC_FINGERPRINT)
    );
    assert_eq!(manifest.tokens, 1_740);
    assert_eq!(manifest.cu_seqlens, [0, 1_740]);
    assert!(manifest.checkpoint_layers.is_empty());
    let plan = manifest.plan().unwrap();
    assert_eq!(plan.transport_bytes, 1_653_893_568);
    assert_eq!(plan.activation_arena_bytes, 148_108_800);
    assert_eq!(plan.readback_bytes, 8_017_920);
    assert_eq!(plan.peak_gpu_data_bytes, 217_084_736);

    let mut anchors = official_vision_stack_shard_anchors();
    anchors.insert("input.embeddings", (TABLE_INPUT_BYTES, TABLE_INPUT_BLAKE3));
    let mut protocol = VisionStackShardProtocol::new(manifest.clone()).unwrap();
    for descriptor in &manifest.shards {
        assert_eq!(
            (descriptor.bytes, descriptor.blake3.as_str()),
            anchors[descriptor.id.as_str()],
            "{} emitted with an unreviewed anchor",
            descriptor.id
        );
        let payload = fs::read(output.path().join(format!("{}.f32", descriptor.id))).unwrap();
        assert_eq!(payload.len() as u64, anchors[descriptor.id.as_str()].0);
        assert_eq!(
            blake3::hash(&payload).to_hex().as_str(),
            anchors[descriptor.id.as_str()].1
        );
        protocol
            .accept_preflight(&inspect_vision_stack_f32_shard(&descriptor.id, &payload))
            .unwrap();
    }
    assert_eq!(manifest.shards[0].bytes, TABLE_INPUT_BYTES);
    assert_eq!(manifest.shards[0].blake3, TABLE_INPUT_BLAKE3);
    let input = fs::read(output.path().join("input.embeddings.f32")).unwrap();
    assert_eq!(input.len() as u64, TABLE_INPUT_BYTES);
    assert_eq!(blake3::hash(&input).to_hex().as_str(), TABLE_INPUT_BLAKE3);

    let expected = fs::read(output.path().join("expected.checkpoints.f32")).unwrap();
    assert_eq!(expected.len() as u64, TABLE_EXPECTED_BYTES);
    assert_eq!(
        blake3::hash(&expected).to_hex().as_str(),
        TABLE_EXPECTED_BLAKE3
    );
    assert_eq!(fs::read_dir(output.path()).unwrap().count(), 31);
}
