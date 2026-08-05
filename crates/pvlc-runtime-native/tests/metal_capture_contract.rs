use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::Mutex,
};

use pvlc_runtime_core::KernelInvocation;
use pvlc_runtime_native::{BackendKind, NativeOptions, NativeRuntime, RuntimeErrorCode};

static SERIAL: Mutex<()> = Mutex::new(());

fn env_flag(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1" | "true"))
}

fn runtime() -> Option<NativeRuntime> {
    match NativeRuntime::new(NativeOptions::default()) {
        Ok(runtime) => Some(runtime),
        Err(error) if env_flag("PVLC_REQUIRE_METAL_CAPTURE") => {
            panic!("Metal capture is required but no native runtime is available: {error}")
        }
        Err(error) => {
            eprintln!("skipping Metal capture contract: {error}");
            None
        }
    }
}

fn capture_invocation() -> KernelInvocation {
    let size = 64_u32;
    KernelInvocation::GemmF32 {
        rows: size,
        inner: size,
        columns: size,
        left: vec![0.25; (size * size) as usize],
        right: vec![0.5; (size * size) as usize],
    }
}

fn filesystem_inventory(path: &Path) -> (u64, u64) {
    let metadata = fs::symlink_metadata(path).unwrap();
    if metadata.is_file() {
        return (1, metadata.len());
    }
    assert!(
        metadata.is_dir(),
        "capture artifact must be a file or package"
    );
    let mut file_count = 0_u64;
    let mut byte_count = 0_u64;
    let mut pending = vec![path.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let entry = entry.unwrap();
            let metadata = entry.metadata().unwrap();
            if metadata.is_dir() {
                pending.push(entry.path());
            } else if metadata.is_file() {
                file_count += 1;
                byte_count += metadata.len();
            } else {
                panic!(
                    "capture package contains a non-file entry: {:?}",
                    entry.path()
                );
            }
        }
    }
    (file_count, byte_count)
}

#[test]
fn capture_target_is_rejected_before_the_operation_or_gpu_submission() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let Some(runtime) = runtime() else { return };
    let temporary = tempfile::tempdir().unwrap();
    let existing = temporary.path().join("existing.gputrace");
    fs::write(&existing, b"do-not-overwrite").unwrap();
    let missing_parent = temporary
        .path()
        .join("missing-parent")
        .join("capture.gputrace");
    let relative = PathBuf::from("relative.gputrace");
    let wrong_extension = temporary.path().join("capture.trace");

    for path in [relative, wrong_extension, missing_parent, existing.clone()] {
        let before = runtime.counters();
        let mut called = false;
        let error = runtime
            .capture_to_gputrace(&path, |_| {
                called = true;
                Ok(())
            })
            .unwrap_err();
        assert_eq!(error.code(), RuntimeErrorCode::Capture);
        assert_eq!(error.scope(), None);
        assert!(!called, "invalid targets must fail before the operation");
        assert_eq!(runtime.counters(), before);
    }
    assert_eq!(fs::read(existing).unwrap(), b"do-not-overwrite");
}

#[test]
fn required_metal_capture_contains_the_real_wgpu_dispatch_and_is_nonempty() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !env_flag("PVLC_REQUIRE_METAL_CAPTURE") {
        eprintln!(
            "skipping durable Metal capture; rerun with MTL_CAPTURE_ENABLED=1 and PVLC_REQUIRE_METAL_CAPTURE=1"
        );
        return;
    }
    assert_eq!(env::var("MTL_CAPTURE_ENABLED").as_deref(), Ok("1"));
    let runtime = runtime().expect("required capture runtime");
    assert_eq!(runtime.capabilities().backend, BackendKind::Metal);
    let target = PathBuf::from(
        env::var("PVLC_METAL_CAPTURE_OUTPUT")
            .expect("required capture must set a durable PVLC_METAL_CAPTURE_OUTPUT"),
    );
    assert!(target.is_absolute());
    assert_eq!(
        target.extension().and_then(|value| value.to_str()),
        Some("gputrace")
    );
    assert!(target.parent().unwrap().is_dir());
    assert!(!target.exists(), "capture target must not already exist");
    let before = runtime.counters();
    let (execution, artifact) = runtime
        .capture_to_gputrace(&target, |runtime| {
            assert!(runtime.is_metal_capture_active()?);
            runtime.run(&capture_invocation())
        })
        .unwrap();

    assert_eq!(execution.values, vec![8.0; 64 * 64]);
    assert_eq!(artifact.path, target);
    assert!(artifact.file_count > 0);
    assert!(artifact.byte_count > 0);
    assert!(artifact.path.exists());
    assert_eq!(
        (artifact.file_count, artifact.byte_count),
        filesystem_inventory(&target)
    );
    assert!(runtime.counters().submissions > before.submissions);
    assert!(!runtime.is_metal_capture_active().unwrap());
}

#[test]
fn operation_failure_stops_capture_removes_partial_output_and_allows_recovery() {
    let _serial = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !env_flag("PVLC_REQUIRE_METAL_CAPTURE") {
        return;
    }
    assert_eq!(env::var("MTL_CAPTURE_ENABLED").as_deref(), Ok("1"));
    let runtime = runtime().expect("required capture runtime");
    assert_eq!(runtime.capabilities().backend, BackendKind::Metal);
    let temporary = tempfile::tempdir().unwrap();
    let failed_target = temporary.path().join("failed.gputrace");
    let before_failure = runtime.counters();
    let error = runtime
        .capture_to_gputrace(&failed_target, |runtime| {
            assert!(runtime.is_metal_capture_active()?);
            Err::<(), _>(pvlc_runtime_native::RuntimeError::operation(
                "sentinel-captured-operation",
            ))
        })
        .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::Operation);
    assert!(error.to_string().contains("sentinel-captured-operation"));
    assert!(!runtime.is_metal_capture_active().unwrap());
    assert!(!failed_target.exists());
    assert_eq!(runtime.counters(), before_failure);

    let recovered_target = temporary.path().join("recovered.gputrace");
    let (execution, artifact) = runtime
        .capture_to_gputrace(&recovered_target, |runtime| {
            assert!(runtime.is_metal_capture_active()?);
            runtime.run(&capture_invocation())
        })
        .unwrap();
    assert_eq!(execution.values, vec![8.0; 64 * 64]);
    assert!(artifact.file_count > 0);
    assert!(artifact.byte_count > 0);
    assert_eq!(
        (artifact.file_count, artifact.byte_count),
        filesystem_inventory(&recovered_target)
    );
    assert!(!runtime.is_metal_capture_active().unwrap());
}
