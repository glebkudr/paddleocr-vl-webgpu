use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use pvlc_pack::{
    OFFICIAL_PROJECTOR_L2_PROFILE, OFFICIAL_PROJECTOR_L3_PROFILE,
    PROJECTOR_SELF_TEST_DESCRIPTOR_ID, PROJECTOR_SELF_TEST_WEIGHTS_ID, PackReader,
    ProjectorSelfTestOracle, ProjectorSelfTestPack,
};
use pvlc_runtime_core::{ProjectorReadback, ProjectorStage};

fn harness() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pvlc-browser-harness"))
}

fn temporary_pack(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("pvlc-{label}-{}-{nonce}.pvlc", std::process::id()))
}

struct Cleanup(PathBuf);

impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[test]
fn cli_exposes_a_dedicated_official_projector_pack_command() {
    let output = harness()
        .args(["prepare-official-projector", "--help"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("--pack"));
    assert!(!stdout.contains("--corpus"));
    assert!(!stdout.contains("--native-baseline"));
}

#[test]
fn official_projector_prepare_fails_closed_without_the_model_gate() {
    for gate in [None, Some("0"), Some("true")] {
        let pack = temporary_pack(&format!("projector-rejected-gate-{gate:?}"));
        let _cleanup = Cleanup(pack.clone());
        let mut command = harness();
        command.env_remove("PVLC_REQUIRE_MODEL");
        if let Some(value) = gate {
            command.env("PVLC_REQUIRE_MODEL", value);
        }
        let output = command
            .args(["prepare-official-projector", "--pack"])
            .arg(&pack)
            .output()
            .unwrap();
        assert!(!output.status.success(), "gate={gate:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("PVLC_REQUIRE_MODEL"),
            "gate={gate:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!pack.exists(), "gate={gate:?} left a rejected pack behind");
    }
}

#[test]
#[ignore = "materializes and reopens the shared 137.7 MiB official projector pack"]
fn prepared_official_projector_pack_reopens_with_both_exact_profiles() {
    let pack_path = temporary_pack("projector-official");
    let _cleanup = Cleanup(pack_path.clone());
    let output = harness()
        .env("PVLC_REQUIRE_MODEL", "1")
        .args(["prepare-official-projector", "--pack"])
        .arg(&pack_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = fs::read(&pack_path).unwrap();
    let reader = PackReader::open(&bytes).unwrap();
    assert_eq!(reader.entries().len(), 6);
    assert_eq!(
        reader
            .entries()
            .iter()
            .filter(|entry| entry.id == PROJECTOR_SELF_TEST_WEIGHTS_ID)
            .count(),
        1
    );
    assert!(reader.section(PROJECTOR_SELF_TEST_DESCRIPTOR_ID).is_some());

    let pack = ProjectorSelfTestPack::open(&bytes).unwrap();
    assert_eq!(
        pack.descriptor().oracle,
        ProjectorSelfTestOracle::OfficialMpsBf16
    );
    assert_eq!(
        pack.descriptor()
            .cases
            .iter()
            .map(|case| (case.profile.as_str(), case.readback))
            .collect::<Vec<_>>(),
        [
            (OFFICIAL_PROJECTOR_L3_PROFILE, ProjectorReadback::AllStages),
            (OFFICIAL_PROJECTOR_L2_PROFILE, ProjectorReadback::OutputOnly),
        ]
    );
    assert_eq!(
        pack.invocation(OFFICIAL_PROJECTOR_L3_PROFILE)
            .unwrap()
            .plan()
            .unwrap()
            .output_tokens,
        319
    );
    assert_eq!(
        pack.invocation(OFFICIAL_PROJECTOR_L2_PROFILE)
            .unwrap()
            .plan()
            .unwrap()
            .output_tokens,
        435
    );
    for stage in ProjectorStage::ALL {
        assert!(
            pack.expected(OFFICIAL_PROJECTOR_L3_PROFILE, stage)
                .is_some()
        );
    }
    assert!(
        pack.expected(OFFICIAL_PROJECTOR_L2_PROFILE, ProjectorStage::Linear2)
            .is_some()
    );
}
