use std::fs::{self, File};
use std::io::Write;
use std::process::Command;

use serde_json::Value;

fn source_fixture(directory: &std::path::Path) -> std::path::PathBuf {
    let source = directory.join("source.safetensors");
    let header_json =
        br#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}}"#;
    let mut header = header_json.to_vec();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = File::create(&source).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    for bits in [0x3f80_u16, 0xbf80, 0x3800] {
        file.write_all(&bits.to_le_bytes()).unwrap();
    }
    file.sync_all().unwrap();
    source
}

#[test]
fn convert_checkpoint_fp16_cli_emits_one_machine_readable_artifact_identity() {
    let directory = tempfile::tempdir().unwrap();
    let source = source_fixture(directory.path());
    let output = directory.path().join("model.fp16.safetensors");
    let source_bytes = fs::read(&source).unwrap();

    let result = Command::new(env!("CARGO_BIN_EXE_pvlc"))
        .args([
            "convert-checkpoint-fp16",
            "--source",
            source.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        result.stderr.is_empty(),
        "successful conversion wrote stderr: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        report.get("conversion").and_then(Value::as_str),
        Some("bf16_to_ieee_f16_rne_v1")
    );
    assert_eq!(report.get("tensor_count").and_then(Value::as_u64), Some(1));
    assert_eq!(report.get("element_count").and_then(Value::as_u64), Some(3));
    assert_eq!(
        report.get("source_blake3").and_then(Value::as_str),
        Some(blake3::hash(&source_bytes).to_hex().as_str())
    );
    let output_bytes = fs::read(&output).unwrap();
    assert_eq!(
        report.get("output_blake3").and_then(Value::as_str),
        Some(blake3::hash(&output_bytes).to_hex().as_str())
    );
    assert_eq!(
        report.get("output_bytes").and_then(Value::as_u64),
        Some(fs::metadata(&output).unwrap().len())
    );

    let before = fs::read(&output).unwrap();
    let repeated = Command::new(env!("CARGO_BIN_EXE_pvlc"))
        .args([
            "convert-checkpoint-fp16",
            "--source",
            source.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!repeated.status.success());
    assert!(
        String::from_utf8_lossy(&repeated.stderr).contains("already exists"),
        "stderr: {}",
        String::from_utf8_lossy(&repeated.stderr)
    );
    assert_eq!(fs::read(output).unwrap(), before);
}
