use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[allow(unused_imports)]
use pvlc_bench_collector::{
    CollectorError, NativeBenchmarkCohortFailurePhaseV1, NativeBenchmarkCohortFailureV1,
    NativeBenchmarkCohortSuccessV1, NativeBenchmarkEnvironmentProbeV1, NativeBenchmarkLeafPlanV1,
    NativeBenchmarkVisionStackValidationReferenceV1, NativeBenchmarkVisionStackValidatorV1,
    run_native_public_legacy_benchmark_cohort_v1, run_native_public_qkv_benchmark_cohort_v1,
};
use pvlc_passes::VisionQkvStackSelection;
use pvlc_runtime_core::{VisionEncoderStackInvocation, VisionStackActivationStrategy};
use pvlc_runtime_native::NativeRuntime;

trait AmbiguousIfDeserialize<Marker> {
    fn marker() {}
}

impl<T: ?Sized> AmbiguousIfDeserialize<()> for T {}
impl<T: serde::de::DeserializeOwned> AmbiguousIfDeserialize<u8> for T {}

fn crate_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn newest_collector_rlib() -> PathBuf {
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
                .is_some_and(|name| {
                    name.starts_with("libpvlc_bench_collector") && name.ends_with(".rlib")
                })
        })
        .max_by_key(|path| {
            fs::metadata(path)
                .and_then(|metadata| metadata.modified())
                .ok()
        })
        .expect("Cargo must build a host pvlc-bench-collector rlib")
}

fn newest_dependency_rlib(dependencies: &Path, prefix: &str) -> PathBuf {
    fs::read_dir(dependencies)
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

fn assert_compile_fails(fixture_name: &str, expected_stderr: &[&str]) {
    let fixture = crate_path(&format!("tests/ui/{fixture_name}.rs"));
    let collector_rlib = newest_collector_rlib();
    let dependencies = collector_rlib.parent().unwrap();
    let passes_rlib = newest_dependency_rlib(dependencies, "libpvlc_passes");
    let runtime_core_rlib = newest_dependency_rlib(dependencies, "libpvlc_runtime_core");
    let runtime_native_rlib = newest_dependency_rlib(dependencies, "libpvlc_runtime_native");
    let output = std::env::temp_dir().join(format!(
        "pvlc-bench-cohort-{fixture_name}-{}",
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
        .arg(format!("pvlc_bench_collector={}", collector_rlib.display()))
        .arg("--extern")
        .arg(format!("pvlc_passes={}", passes_rlib.display()))
        .arg("--extern")
        .arg(format!("pvlc_runtime_core={}", runtime_core_rlib.display()))
        .arg("--extern")
        .arg(format!(
            "pvlc_runtime_native={}",
            runtime_native_rlib.display()
        ))
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
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        !stderr.contains("can't find crate") && !stderr.contains("incompatible version"),
        "{fixture_name} failed before checking authority:\n{stderr}",
    );
    for expected in expected_stderr {
        assert!(
            stderr.contains(expected),
            "{fixture_name} did not fail for `{expected}`:\n{stderr}",
        );
    }
}

#[test]
fn public_observation_surface_is_read_only() {
    fn require_legacy_entry(
        _entry: fn(
            &NativeRuntime,
            NativeBenchmarkLeafPlanV1,
            &VisionEncoderStackInvocation<'_>,
            &[usize],
            VisionStackActivationStrategy,
            &mut NativeBenchmarkEnvironmentProbeV1,
            &mut NativeBenchmarkVisionStackValidatorV1,
        )
            -> Result<NativeBenchmarkCohortSuccessV1, NativeBenchmarkCohortFailureV1>,
    ) {
    }
    fn require_qkv_entry(
        _entry: fn(
            &NativeRuntime,
            NativeBenchmarkLeafPlanV1,
            &VisionEncoderStackInvocation<'_>,
            &[usize],
            VisionStackActivationStrategy,
            &VisionQkvStackSelection,
            &mut NativeBenchmarkEnvironmentProbeV1,
            &mut NativeBenchmarkVisionStackValidatorV1,
        )
            -> Result<NativeBenchmarkCohortSuccessV1, NativeBenchmarkCohortFailureV1>,
    ) {
    }
    fn observe_success(success: &NativeBenchmarkCohortSuccessV1) {
        let _: &str = success.run_id();
        let _: usize = success.attempt_count();
        let _ = success.assembled();
    }
    fn observe_failure(failure: &NativeBenchmarkCohortFailureV1) {
        let _: &str = failure.run_id();
        let _: NativeBenchmarkCohortFailurePhaseV1 = failure.phase();
        let _: &str = failure.code();
        let _: u64 = failure.expected_attempt_count();
        let _ = failure.attempt_log();
        let _: Vec<u8> = failure.canonical_bytes();
    }
    fn author_plan(plan: &mut NativeBenchmarkLeafPlanV1) {
        let _ = &mut plan.run_id;
        let _ = &mut plan.passport;
        let _ = &mut plan.workload;
        let _ = &mut plan.correctness_anchor;
        let _: &mut NativeBenchmarkVisionStackValidationReferenceV1 =
            &mut plan.validation_reference;
        let _ = &mut plan.protocol;
        let _ = &mut plan.load_or_compile;
        let _ = &mut plan.base_descriptor;
    }
    fn require_data_only_authority_constructors(
        _probe: fn() -> NativeBenchmarkEnvironmentProbeV1,
        _validator: fn(
            &NativeBenchmarkLeafPlanV1,
        ) -> Result<NativeBenchmarkVisionStackValidatorV1, CollectorError>,
    ) {
    }

    let _ = observe_success;
    let _ = observe_failure;
    let _ = author_plan;
    require_legacy_entry(run_native_public_legacy_benchmark_cohort_v1);
    require_qkv_entry(run_native_public_qkv_benchmark_cohort_v1);
    require_data_only_authority_constructors(
        NativeBenchmarkEnvironmentProbeV1::system,
        NativeBenchmarkVisionStackValidatorV1::from_leaf,
    );
}

#[test]
fn opaque_cohort_results_reject_external_construction_and_mutation() {
    for (fixture, field) in [
        ("cohort_success_field_escape", "run_id"),
        ("cohort_success_count_field_escape", "attempt_count"),
        ("cohort_success_assembly_field_escape", "assembled"),
        ("cohort_failure_field_escape", "run_id"),
        (
            "cohort_failure_count_field_escape",
            "expected_attempt_count",
        ),
        ("cohort_failure_journal_field_escape", "attempt_log"),
        ("cohort_failure_code_field_escape", "failure_code"),
        ("cohort_failure_phase_field_escape", "phase"),
        ("cohort_failure_bytes_field_escape", "canonical_bytes"),
    ] {
        let privacy = format!("field `{field}` of struct");
        assert_compile_fails(fixture, &[&privacy, "is private"]);
    }
    assert_compile_fails(
        "cohort_failure_journal_mutation",
        &[
            "cannot borrow data in a `&` reference as mutable",
            "attempt_log().clear()",
        ],
    );
    assert_compile_fails(
        "cohort_success_summary_mutation",
        &[
            "cannot assign to data in a `&` reference",
            "summary().count",
        ],
    );
    assert_compile_fails(
        "cohort_success_hash_mutation",
        &[
            "cannot borrow data in a `&` reference as mutable",
            "assembly_blake3()",
        ],
    );
    assert_compile_fails(
        "cohort_probe_callback_injection",
        &["NativeBenchmarkEnvironmentProbeV1"],
    );
    assert_compile_fails(
        "cohort_validator_callback_injection",
        &["NativeBenchmarkVisionStackValidatorV1"],
    );
    assert_compile_fails(
        "cohort_qkv_probe_callback_injection",
        &["NativeBenchmarkEnvironmentProbeV1"],
    );
    assert_compile_fails(
        "cohort_qkv_validator_callback_injection",
        &["NativeBenchmarkVisionStackValidatorV1"],
    );
    assert_compile_fails(
        "cohort_test_kernel_escape",
        &["run_native_public_legacy_benchmark_cohort_test_kernel_v1"],
    );
    assert_compile_fails(
        "cohort_engine_escape",
        &["run_native_benchmark_cohort_engine_v1"],
    );
    assert_compile_fails("cohort_environment_default_construction", &["Default"]);
    assert_compile_fails("cohort_validator_default_construction", &["Default"]);
    assert_compile_fails("cohort_success_default_construction", &["Default"]);
    assert_compile_fails("cohort_failure_default_construction", &["Default"]);
    assert_compile_fails(
        "cohort_environment_callback_constructor",
        &[
            "no function or associated item named `from_callback`",
            "NativeBenchmarkEnvironmentProbeV1",
        ],
    );
    assert_compile_fails(
        "cohort_environment_parts_constructor",
        &[
            "no function or associated item named `from_parts`",
            "NativeBenchmarkEnvironmentProbeV1",
        ],
    );
    assert_compile_fails(
        "cohort_validator_callback_constructor",
        &[
            "no function or associated item named `from_callback`",
            "NativeBenchmarkVisionStackValidatorV1",
        ],
    );
    assert_compile_fails(
        "cohort_validator_parts_constructor",
        &[
            "no function or associated item named `from_parts`",
            "NativeBenchmarkVisionStackValidatorV1",
        ],
    );
}

#[test]
fn physical_authorities_and_results_cannot_be_deserialized_around_the_sealed_runner() {
    let _ = <NativeBenchmarkCohortSuccessV1 as AmbiguousIfDeserialize<_>>::marker;
    let _ = <NativeBenchmarkCohortFailureV1 as AmbiguousIfDeserialize<_>>::marker;
    let _ = <NativeBenchmarkEnvironmentProbeV1 as AmbiguousIfDeserialize<_>>::marker;
    let _ = <NativeBenchmarkVisionStackValidatorV1 as AmbiguousIfDeserialize<_>>::marker;
}
