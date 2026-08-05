use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use pvlc_runtime_core::DecoderKvSessionStep;
use pvlc_runtime_native::{
    DecoderKvSessionSnapshot, DecoderKvSessionStepExecution, NativeDecoderKvSession, RuntimeError,
};

fn crate_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn newest_rlib(prefix: &str) -> PathBuf {
    let dependencies = std::env::current_exe()
        .expect("current integration-test executable path")
        .parent()
        .expect("integration tests execute from Cargo's dependency directory")
        .to_path_buf();
    fs::read_dir(&dependencies)
        .expect("read Cargo dependency directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(prefix) && name.ends_with(".rlib"))
        })
        .max_by_key(|path| {
            fs::metadata(path)
                .and_then(|metadata| metadata.modified())
                .ok()
        })
        .unwrap_or_else(|| panic!("Cargo must build dependency rlib {prefix}"))
}

fn compile_failure_stderr(fixture_name: &str) -> String {
    let fixture = crate_path(&format!("tests/ui/{fixture_name}.rs"));
    let runtime_rlib = newest_rlib("libpvlc_runtime_native");
    let dependencies = runtime_rlib
        .parent()
        .expect("runtime rlib has a dependency directory");
    let output = std::env::temp_dir().join(format!(
        "pvlc-decoder-kv-session-{fixture_name}-{}",
        std::process::id(),
    ));
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let result = Command::new(rustc)
        .arg("--edition=2024")
        .arg("--crate-name")
        .arg(fixture_name)
        .arg("--emit=metadata")
        .arg(&fixture)
        .arg("--extern")
        .arg(format!("pvlc_runtime_native={}", runtime_rlib.display()))
        .arg("-L")
        .arg(format!("dependency={}", dependencies.display()))
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap_or_else(|error| panic!("run compile-fail fixture {fixture_name}: {error}"));
    let _ = fs::remove_file(&output);
    assert!(
        !result.status.success(),
        "{fixture_name} unexpectedly compiled"
    );
    let stderr = String::from_utf8_lossy(&result.stderr).into_owned();
    assert!(
        !stderr.contains("can't find crate") && !stderr.contains("incompatible version"),
        "{fixture_name} failed before checking session authority:\n{stderr}",
    );
    stderr
}

fn assert_compile_fails(fixture_name: &str, expected_stderr: &[&str]) {
    let stderr = compile_failure_stderr(fixture_name);
    for expected in expected_stderr {
        assert!(
            stderr.contains(expected),
            "{fixture_name} did not fail for `{expected}`:\n{stderr}",
        );
    }
}

fn assert_compile_fails_with_any(fixture_name: &str, alternatives: &[&str]) {
    let stderr = compile_failure_stderr(fixture_name);
    assert!(
        alternatives
            .iter()
            .any(|alternative| stderr.contains(alternative)),
        "{fixture_name} did not fail for any of {alternatives:?}:\n{stderr}",
    );
}

#[test]
fn public_session_surface_is_lifetime_bound_and_read_only_except_for_step() {
    fn observe(session: &NativeDecoderKvSession<'_>) {
        let _: u32 = session.cache_tokens();
        let _: u32 = session.cache_capacity();
        let _ = session.creation_diagnostics();
    }
    fn execute(
        session: &mut NativeDecoderKvSession<'_>,
        step: &DecoderKvSessionStep<'_>,
    ) -> Result<DecoderKvSessionStepExecution, RuntimeError> {
        session.step(step)
    }
    fn finish(
        session: NativeDecoderKvSession<'_>,
    ) -> Result<DecoderKvSessionSnapshot, RuntimeError> {
        session.finish()
    }

    let _ = observe;
    let _ = execute;
    let _ = finish;
}

#[test]
fn external_code_cannot_forge_clone_default_or_open_the_session_authority() {
    assert_compile_fails_with_any(
        "decoder_kv_session_field_escape",
        &["is private", "no field"],
    );
    assert_compile_fails(
        "decoder_kv_session_struct_literal",
        &["cannot construct `NativeDecoderKvSession", "private fields"],
    );
    assert_compile_fails(
        "decoder_kv_session_clone",
        &["trait bound", "NativeDecoderKvSession", "Clone"],
    );
    assert_compile_fails(
        "decoder_kv_session_default",
        &["trait bound", "Default", "NativeDecoderKvSession"],
    );
    assert_compile_fails(
        "decoder_kv_session_parts_constructor",
        &["no function or associated item named `from_parts`"],
    );
    assert_compile_fails(
        "decoder_kv_session_private_module_escape",
        &["module `decoder_kv_session` is private"],
    );
    assert_compile_fails(
        "decoder_kv_session_shared_step",
        &["cannot borrow", "as mutable"],
    );
    assert_compile_fails(
        "decoder_kv_session_finish_reuse",
        &["moved value", "session"],
    );
}

#[test]
fn session_cannot_outlive_the_runtime_that_owns_its_device_authority() {
    assert_compile_fails(
        "decoder_kv_session_runtime_lifetime_escape",
        &["cannot return value referencing local variable `runtime`"],
    );
}
