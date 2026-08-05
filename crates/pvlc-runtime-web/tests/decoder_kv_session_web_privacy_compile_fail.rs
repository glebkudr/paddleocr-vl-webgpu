use std::{fs, process::Command};

fn workspace_path(relative: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

#[test]
fn decoder_authority_field_is_present_but_inaccessible_outside_the_crate() {
    let fixture_root = workspace_path("target/m6e3-decoder-authority-compile-fail/fixture");
    let fixture_src = fixture_root.join("src");
    let target_dir = workspace_path("target/m6e3-decoder-authority-compile-fail/build");
    if fixture_root.exists() {
        fs::remove_dir_all(&fixture_root).expect("stale compile-fail fixture must be removable");
    }
    fs::create_dir_all(&fixture_src).expect("compile-fail fixture directory must be creatable");
    fs::copy(
        workspace_path("crates/pvlc-runtime-web/tests/compile_fail/decoder_authority_privacy.rs"),
        fixture_src.join("lib.rs"),
    )
    .expect("compile-fail source must be copied");
    fs::write(
        fixture_root.join("Cargo.toml"),
        format!(
            r#"[package]
name = "pvlc-decoder-authority-privacy-probe"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
pvlc-runtime-web = {{ path = {:?} }}

[workspace]
"#,
            workspace_path("crates/pvlc-runtime-web"),
        ),
    )
    .expect("compile-fail manifest must be written");
    let manifest = fixture_root.join("Cargo.toml");
    let output = Command::new(env!("CARGO"))
        .args([
            "check",
            "--quiet",
            "--target",
            "wasm32-unknown-unknown",
            "--manifest-path",
        ])
        .arg(&manifest)
        .arg("--target-dir")
        .arg(&target_dir)
        .output()
        .expect("compile-fail fixture must launch cargo check");
    assert!(
        !output.status.success(),
        "external crate unexpectedly accessed WebRuntime decoder authority"
    );
    let stderr = String::from_utf8(output.stderr).expect("cargo diagnostics must be UTF-8");
    assert!(
        stderr.contains("error[E0616]")
            && stderr.contains("decoder_kv_session")
            && stderr.contains("private"),
        "fixture must fail specifically because the real decoder authority field is private:\n{stderr}"
    );
    assert!(
        !stderr.contains("error[E0609]"),
        "unknown-field failure does not prove privacy of a real authority field:\n{stderr}"
    );
}
