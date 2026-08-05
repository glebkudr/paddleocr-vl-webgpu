use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use pvlc_cli::{
    CompileTinyOptions, ModelLock, SourceErrorCode, compile_tiny_pack, verify_model_source,
};
use pvlc_ir::SemanticGraph;
use pvlc_model_schema::{COMPILER_MODEL_ABI, MODEL_ID, MODEL_REVISION, PaddleOcrVl16Schema};
use pvlc_pack::{PackReader, PrecisionProfile};

const EXPECTED_FIXTURE_LOCK_BLAKE3: &str =
    "07a471f15c5e49d808f634d976e66019a16c0c2974c837bc22478dc883fbee29";
const EXPECTED_FIXTURE_SOURCE_FINGERPRINT: &str =
    "080efe42ee9136a0bd6e79618adb1f4cf946e236f8864868dc541f5d47c0c4cb";

fn fixture_files() -> [(&'static str, &'static [u8]); 3] {
    [
        ("config.json", br#"{"model_type":"paddleocr_vl"}\n"#),
        ("model.safetensors", b"tiny-weight-fixture\x00\xff"),
        ("tokenizer.json", br#"{"version":"1.0"}\n"#),
    ]
}

fn model_lock_text(files: &[(&str, &[u8])]) -> String {
    let mut text = format!(
        "format_version = 1\nmodel_id = {MODEL_ID:?}\nrevision = {MODEL_REVISION:?}\ncompiler_model_abi = {COMPILER_MODEL_ABI}\n\n[files]\n"
    );
    let mut files = files.to_vec();
    files.sort_by_key(|(path, _)| *path);
    for (path, bytes) in files {
        text.push_str(&format!(
            "{path:?} = {{ blake3 = {:?}, size = {} }}\n",
            blake3::hash(bytes).to_hex().as_str(),
            bytes.len()
        ));
    }
    text
}

fn source_fixture() -> (tempfile::TempDir, PathBuf, PathBuf, ModelLock) {
    let dir = tempfile::tempdir().unwrap();
    let model_dir = dir.path().join("model");
    fs::create_dir(&model_dir).unwrap();
    let files = fixture_files();
    for (name, bytes) in files {
        fs::write(model_dir.join(name), bytes).unwrap();
    }
    let lock_text = model_lock_text(&files);
    let lock_path = dir.path().join("model.lock");
    fs::write(&lock_path, &lock_text).unwrap();
    let lock = ModelLock::parse(lock_text.as_bytes()).unwrap();
    (dir, model_dir, lock_path, lock)
}

fn independent_source_fingerprint(lock: &ModelLock) -> [u8; 32] {
    let files: Vec<_> = lock
        .files()
        .iter()
        .map(|(path, file)| {
            serde_json::json!({
                "blake3": file.blake3,
                "path": path,
                "size": file.size,
            })
        })
        .collect();
    let value = serde_json::json!({
        "compiler_model_abi": lock.compiler_model_abi,
        "files": files,
        "model_id": lock.model_id,
        "revision": lock.revision,
    });
    let mut canonical = serde_json::to_vec(&value).unwrap();
    canonical.push(b'\n');
    *blake3::hash(&canonical).as_bytes()
}

#[test]
fn checked_in_lock_has_exact_identity_inventory_and_raw_digest() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/paddleocr-vl-1.6.lock");
    let lock = ModelLock::from_path(path).unwrap();
    assert_eq!(lock.format_version, 1);
    assert_eq!(lock.model_id, MODEL_ID);
    assert_eq!(lock.revision, MODEL_REVISION);
    assert_eq!(lock.compiler_model_abi, COMPILER_MODEL_ABI);
    assert_eq!(lock.files().len(), 19);
    assert_eq!(
        lock.raw_blake3_hex(),
        "c2ffa46821fdda1a7e4cd6bad63cdcdc2f775b8fed250b0679cc7d5411577d10"
    );
    let weights = lock.file("model.safetensors").unwrap();
    assert_eq!(weights.size, 1_917_255_968);
    assert_eq!(
        weights.blake3,
        "4dc3ab13685a0c0a701f77f9c5ebafdc0004074247e00bde6b7e7b04279a41fc"
    );
}

#[test]
fn verifier_accepts_exact_files_and_returns_a_stable_source_fingerprint() {
    let (_dir, model_dir, _lock_path, lock) = source_fixture();
    let first = verify_model_source(&lock, &model_dir).unwrap();
    let second = verify_model_source(&lock, &model_dir).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.files.len(), 3);
    assert_eq!(
        first
            .files
            .iter()
            .map(|file| file.path.as_str())
            .collect::<Vec<_>>(),
        ["config.json", "model.safetensors", "tokenizer.json"]
    );
    assert_eq!(first.lock_blake3, lock.raw_blake3());
    assert_eq!(lock.raw_blake3_hex(), EXPECTED_FIXTURE_LOCK_BLAKE3);
    assert_eq!(
        first.source_fingerprint,
        independent_source_fingerprint(&lock)
    );
    assert_eq!(
        blake3::Hash::from_bytes(first.source_fingerprint)
            .to_hex()
            .as_str(),
        EXPECTED_FIXTURE_SOURCE_FINGERPRINT
    );
}

#[test]
fn fingerprints_and_packs_ignore_filesystem_location_but_change_with_locked_content() {
    let (_first_dir, first_model_dir, _first_lock_path, first_lock) = source_fixture();
    let (_second_dir, second_model_dir, _second_lock_path, second_lock) = source_fixture();
    let options = CompileTinyOptions {
        compiler_build: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".into(),
        precision_profile: PrecisionProfile::Balanced,
        resolution_buckets: vec![[28, 56]],
        context_limit: 96,
    };
    let first_verified = verify_model_source(&first_lock, &first_model_dir).unwrap();
    let second_verified = verify_model_source(&second_lock, &second_model_dir).unwrap();
    assert_eq!(
        first_verified.source_fingerprint,
        second_verified.source_fingerprint
    );
    assert_eq!(
        compile_tiny_pack(&first_lock, &first_model_dir, &options).unwrap(),
        compile_tiny_pack(&second_lock, &second_model_dir, &options).unwrap()
    );

    let mut changed_config = fixture_files()[0].1.to_vec();
    changed_config[0] ^= 1;
    fs::write(second_model_dir.join("config.json"), &changed_config).unwrap();
    let base = fixture_files();
    let changed_files = [("config.json", changed_config.as_slice()), base[1], base[2]];
    let changed_lock_text = model_lock_text(&changed_files);
    let changed_lock = ModelLock::parse(changed_lock_text.as_bytes()).unwrap();
    let changed_verified = verify_model_source(&changed_lock, &second_model_dir).unwrap();
    assert_ne!(
        first_verified.source_fingerprint,
        changed_verified.source_fingerprint
    );
    assert_eq!(
        changed_verified.source_fingerprint,
        independent_source_fingerprint(&changed_lock)
    );
    assert_ne!(
        compile_tiny_pack(&first_lock, &first_model_dir, &options).unwrap(),
        compile_tiny_pack(&changed_lock, &second_model_dir, &options).unwrap()
    );
}

#[test]
fn identity_gate_runs_before_any_source_file_access() {
    let (_dir, _model_dir, _lock_path, lock) = source_fixture();
    for (mut wrong, expected) in [
        (lock.clone(), SourceErrorCode::WrongModelId),
        (lock.clone(), SourceErrorCode::WrongModelRevision),
        (lock.clone(), SourceErrorCode::WrongCompilerModelAbi),
    ] {
        match expected {
            SourceErrorCode::WrongModelId => wrong.model_id = "other/model".into(),
            SourceErrorCode::WrongModelRevision => wrong.revision = "0".repeat(40),
            SourceErrorCode::WrongCompilerModelAbi => wrong.compiler_model_abi += 1,
            _ => unreachable!(),
        }
        let error = verify_model_source(&wrong, "/definitely/not/a/model/directory").unwrap_err();
        assert_eq!(error.code(), expected);
    }
}

#[test]
fn parser_rejects_unknown_duplicate_invalid_hash_and_unsafe_path_contracts() {
    let valid = model_lock_text(&[("config.json", b"{}")]);

    let unknown = valid.replace("format_version = 1", "format_version = 1\nsurprise = true");
    assert_eq!(
        ModelLock::parse(unknown.as_bytes()).unwrap_err().code(),
        SourceErrorCode::InvalidLock
    );

    let duplicate = valid.replace(
        "format_version = 1",
        "format_version = 1\nformat_version = 1",
    );
    assert_eq!(
        ModelLock::parse(duplicate.as_bytes()).unwrap_err().code(),
        SourceErrorCode::InvalidLock
    );

    let expected_hash = blake3::hash(b"{}").to_hex().to_string();
    for replacement in [
        "A".repeat(64),
        "a".repeat(63),
        "a".repeat(65),
        "g".repeat(64),
    ] {
        let bad_hash = valid.replacen(&expected_hash, &replacement, 1);
        assert_eq!(
            ModelLock::parse(bad_hash.as_bytes()).unwrap_err().code(),
            SourceErrorCode::InvalidHash,
            "digest {replacement:?}"
        );
    }

    let unknown_file_field = valid.replace("size = 2 }", "size = 2, surprise = true }");
    assert_eq!(
        ModelLock::parse(unknown_file_field.as_bytes())
            .unwrap_err()
            .code(),
        SourceErrorCode::InvalidLock
    );

    for unsafe_name in [
        "../config.json",
        "/absolute/config.json",
        "nested/../config.json",
        "./config.json",
        r"C:\config.json",
    ] {
        let unsafe_path = model_lock_text(&[(unsafe_name, b"{}")]);
        assert_eq!(
            ModelLock::parse(unsafe_path.as_bytes()).unwrap_err().code(),
            SourceErrorCode::UnsafePath,
            "path {unsafe_name:?}"
        );
    }

    let unsupported = valid.replace("format_version = 1", "format_version = 2");
    assert_eq!(
        ModelLock::parse(unsupported.as_bytes()).unwrap_err().code(),
        SourceErrorCode::UnsupportedFormatVersion
    );

    let no_files = format!(
        "format_version = 1\nmodel_id = {MODEL_ID:?}\nrevision = {MODEL_REVISION:?}\ncompiler_model_abi = 1\n\n[files]\n"
    );
    assert_eq!(
        ModelLock::parse(no_files.as_bytes()).unwrap_err().code(),
        SourceErrorCode::EmptyFileSet
    );
}

#[test]
fn verifier_rejects_missing_unexpected_wrong_size_and_wrong_hash_files() {
    let (_dir, model_dir, _lock_path, lock) = source_fixture();

    fs::remove_file(model_dir.join("config.json")).unwrap();
    let error = verify_model_source(&lock, &model_dir).unwrap_err();
    assert_eq!(error.code(), SourceErrorCode::MissingFile);
    assert_eq!(error.path(), Some("config.json"));
    fs::write(
        model_dir.join("config.json"),
        br#"{"model_type":"paddleocr_vl"}\n"#,
    )
    .unwrap();

    fs::write(model_dir.join("unexpected.bin"), b"x").unwrap();
    let error = verify_model_source(&lock, &model_dir).unwrap_err();
    assert_eq!(error.code(), SourceErrorCode::UnexpectedFile);
    assert_eq!(error.path(), Some("unexpected.bin"));
    fs::remove_file(model_dir.join("unexpected.bin")).unwrap();

    fs::write(model_dir.join("config.json"), b"short").unwrap();
    let error = verify_model_source(&lock, &model_dir).unwrap_err();
    assert_eq!(error.code(), SourceErrorCode::SizeMismatch);
    assert_eq!(error.path(), Some("config.json"));

    let expected_size = lock.file("config.json").unwrap().size as usize;
    fs::write(model_dir.join("config.json"), vec![b'x'; expected_size]).unwrap();
    let error = verify_model_source(&lock, &model_dir).unwrap_err();
    assert_eq!(error.code(), SourceErrorCode::HashMismatch);
    assert_eq!(error.path(), Some("config.json"));
}

#[test]
fn every_locked_file_is_hashed_and_corruption_blocks_library_and_cli_compilation() {
    let (dir, model_dir, lock_path, lock) = source_fixture();
    let options = CompileTinyOptions {
        compiler_build: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".into(),
        precision_profile: PrecisionProfile::Fidelity,
        resolution_buckets: vec![[28, 28]],
        context_limit: 64,
    };

    for (path, original) in fixture_files() {
        let mut corrupted = original.to_vec();
        let middle = corrupted.len() / 2;
        corrupted[middle] ^= 0x5a;
        fs::write(model_dir.join(path), &corrupted).unwrap();

        let error = verify_model_source(&lock, &model_dir).unwrap_err();
        assert_eq!(error.code(), SourceErrorCode::HashMismatch, "{path}");
        assert_eq!(error.path(), Some(path));
        let error = compile_tiny_pack(&lock, &model_dir, &options).unwrap_err();
        assert_eq!(error.code(), SourceErrorCode::HashMismatch, "{path}");
        assert_eq!(error.path(), Some(path));

        let output = dir.path().join(format!("{path}.pvlc"));
        let sentinel = b"existing-output-must-survive";
        fs::write(&output, sentinel).unwrap();
        let status = Command::new(env!("CARGO_BIN_EXE_pvlc"))
            .args([
                "compile-tiny",
                "--lock",
                lock_path.to_str().unwrap(),
                "--model-dir",
                model_dir.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
                "--compiler-build",
                &options.compiler_build,
            ])
            .status()
            .unwrap();
        assert!(!status.success(), "{path}");
        assert_eq!(fs::read(output).unwrap(), sentinel, "{path}");

        fs::write(model_dir.join(path), original).unwrap();
    }
}

#[cfg(unix)]
#[test]
fn verifier_rejects_symlinked_model_files() {
    use std::os::unix::fs::symlink;

    let (_dir, model_dir, _lock_path, lock) = source_fixture();
    let real = model_dir.join("config.real");
    fs::rename(model_dir.join("config.json"), &real).unwrap();
    symlink(&real, model_dir.join("config.json")).unwrap();
    let error = verify_model_source(&lock, &model_dir).unwrap_err();
    assert_eq!(error.code(), SourceErrorCode::NotRegularFile);
    assert_eq!(error.path(), Some("config.json"));
}

#[test]
fn deterministic_tiny_compilation_embeds_schema_semantic_ir_and_source_provenance() {
    let (_dir, model_dir, _lock_path, lock) = source_fixture();
    let options = CompileTinyOptions {
        compiler_build: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
        precision_profile: PrecisionProfile::Fidelity,
        resolution_buckets: vec![[28, 28]],
        context_limit: 64,
    };
    let first = compile_tiny_pack(&lock, &model_dir, &options).unwrap();
    let second = compile_tiny_pack(&lock, &model_dir, &options).unwrap();
    assert_eq!(first, second);

    let pack = PackReader::open(&first).unwrap();
    assert_eq!(pack.manifest().compiler_build, options.compiler_build);
    assert_eq!(pack.manifest().precision_profile, options.precision_profile);
    assert_eq!(
        pack.manifest().resolution_buckets,
        options.resolution_buckets
    );
    assert_eq!(pack.manifest().context_limit, options.context_limit);
    assert_eq!(
        pack.section("model.schema").unwrap(),
        PaddleOcrVl16Schema::canonical_catalog_bytes()
    );
    assert_eq!(
        pack.section("model.semantic_map").unwrap(),
        PaddleOcrVl16Schema::canonical_semantic_map_bytes()
    );
    assert_eq!(
        pack.section("ir.semantic").unwrap(),
        SemanticGraph::paddleocr_vl_16().canonical_bytes().unwrap()
    );
    let provenance = pack.section("self_test.source_provenance").unwrap();
    assert_eq!(provenance.len(), 64);
    let verified = verify_model_source(&lock, &model_dir).unwrap();
    assert_eq!(&provenance[..32], &verified.lock_blake3);
    assert_eq!(&provenance[32..], &verified.source_fingerprint);

    let mut variants = Vec::new();
    let mut changed_build = options.clone();
    changed_build.compiler_build =
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".into();
    variants.push(changed_build);
    let mut changed_precision = options.clone();
    changed_precision.precision_profile = PrecisionProfile::Turbo;
    variants.push(changed_precision);
    let mut changed_buckets = options.clone();
    changed_buckets.resolution_buckets = vec![[56, 56], [84, 28]];
    variants.push(changed_buckets);
    let mut changed_context = options.clone();
    changed_context.context_limit = 65;
    variants.push(changed_context);

    for variant in variants {
        let bytes = compile_tiny_pack(&lock, &model_dir, &variant).unwrap();
        assert_ne!(bytes, first);
        let manifest = PackReader::open(&bytes).unwrap().manifest().clone();
        assert_eq!(manifest.compiler_build, variant.compiler_build);
        assert_eq!(manifest.precision_profile, variant.precision_profile);
        assert_eq!(manifest.resolution_buckets, variant.resolution_buckets);
        assert_eq!(manifest.context_limit, variant.context_limit);
    }

    for invalid in [
        "a".repeat(63),
        "a".repeat(65),
        "A".repeat(64),
        "g".repeat(64),
    ] {
        let mut invalid_build = options.clone();
        invalid_build.compiler_build = invalid.clone();
        assert_eq!(
            compile_tiny_pack(&lock, &model_dir, &invalid_build)
                .unwrap_err()
                .code(),
            SourceErrorCode::InvalidCompilerBuild,
            "compiler build {invalid:?}"
        );
    }
}

#[test]
fn cli_compile_tiny_is_atomic_and_rejects_the_wrong_revision() {
    let (dir, model_dir, lock_path, _lock) = source_fixture();
    let output = dir.path().join("tiny.pvlc");
    let compiler_build = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    fs::write(&output, b"old-pack-must-be-atomically-replaced").unwrap();
    #[cfg(unix)]
    let inode_before = {
        use std::os::unix::fs::MetadataExt;
        fs::metadata(&output).unwrap().ino()
    };
    let status = Command::new(env!("CARGO_BIN_EXE_pvlc"))
        .args([
            "compile-tiny",
            "--lock",
            lock_path.to_str().unwrap(),
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--compiler-build",
            compiler_build,
        ])
        .status()
        .unwrap();
    assert!(status.success());
    PackReader::open(&fs::read(&output).unwrap()).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_ne!(
            fs::metadata(&output).unwrap().ino(),
            inode_before,
            "successful output must be installed by rename, not in-place truncation"
        );
    }

    let wrong_lock_path = dir.path().join("wrong.lock");
    let wrong = fs::read_to_string(&lock_path)
        .unwrap()
        .replace(MODEL_REVISION, &"0".repeat(40));
    fs::write(&wrong_lock_path, wrong).unwrap();
    let sentinel = b"must-survive-failed-compile";
    fs::write(&output, sentinel).unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_pvlc"))
        .args([
            "compile-tiny",
            "--lock",
            wrong_lock_path.to_str().unwrap(),
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--compiler-build",
            compiler_build,
        ])
        .status()
        .unwrap();
    assert!(!status.success());
    assert_eq!(fs::read(output).unwrap(), sentinel);
}

#[cfg(unix)]
#[test]
fn cli_refuses_to_follow_an_output_symlink_and_preserves_its_victim() {
    use std::os::unix::fs::symlink;

    let (dir, model_dir, lock_path, _lock) = source_fixture();
    let victim = dir.path().join("victim.bin");
    let output = dir.path().join("output.pvlc");
    let sentinel = b"symlink-victim-must-never-be-overwritten";
    fs::write(&victim, sentinel).unwrap();
    symlink(&victim, &output).unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_pvlc"))
        .args([
            "compile-tiny",
            "--lock",
            lock_path.to_str().unwrap(),
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--compiler-build",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        ])
        .status()
        .unwrap();
    assert!(!status.success());
    assert_eq!(fs::read(&victim).unwrap(), sentinel);
    assert!(
        fs::symlink_metadata(&output)
            .unwrap()
            .file_type()
            .is_symlink()
    );
}

#[cfg(unix)]
#[test]
fn cli_write_stage_failure_preserves_the_previous_output_atomically() {
    let (dir, model_dir, lock_path, _lock) = source_fixture();
    let output = dir.path().join("limited-output.pvlc");
    let sentinel = b"old-output-survives-failed-staged-write";
    fs::write(&output, sentinel).unwrap();

    // POSIX `ulimit -f 1` caps newly written files at one 512-byte block. The
    // tiny pack is much larger. A direct truncate/unlink+write destroys the old
    // output before SIGXFSZ/EFBIG; write-temp-then-rename leaves it untouched.
    let status = Command::new("/bin/sh")
        .arg("-c")
        .arg("ulimit -f 1; exec \"$@\"")
        .arg("pvlc-rlimit-wrapper")
        .arg(env!("CARGO_BIN_EXE_pvlc"))
        .args([
            "compile-tiny",
            "--lock",
            lock_path.to_str().unwrap(),
            "--model-dir",
            model_dir.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--compiler-build",
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        ])
        .status()
        .unwrap();
    assert!(!status.success());
    assert_eq!(fs::read(output).unwrap(), sentinel);
}

#[test]
fn real_pinned_source_verifies_when_the_local_snapshot_is_available() {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let lock = ModelLock::from_path(repository.join("models/paddleocr-vl-1.6.lock")).unwrap();
    let model_dir = repository.join("models/snapshots").join(MODEL_REVISION);
    if !model_dir.join("model.safetensors").is_file() {
        assert_ne!(
            std::env::var("PVLC_REQUIRE_MODEL").as_deref(),
            Ok("1"),
            "PVLC_REQUIRE_MODEL=1 but the complete snapshot at {} is absent",
            model_dir.display()
        );
        eprintln!(
            "skipping local source verification: the complete snapshot at {} is absent",
            model_dir.display()
        );
        return;
    }
    let verified = verify_model_source(&lock, model_dir).unwrap();
    assert_eq!(verified.files.len(), 19);
}
