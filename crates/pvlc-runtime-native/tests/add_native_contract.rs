use std::{env, sync::OnceLock};

use pvlc_cpu_ref::add_vectors_f32;
use pvlc_runtime_core::{KernelId, KernelInvocation};
use pvlc_runtime_native::{BackendKind, ErrorScopeKind, NativeOptions, NativeRuntime};

static RUNTIME: OnceLock<Result<NativeRuntime, String>> = OnceLock::new();

fn env_flag(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1" | "true"))
}

fn runtime() -> Option<&'static NativeRuntime> {
    match RUNTIME.get_or_init(|| {
        NativeRuntime::new(NativeOptions::default()).map_err(|error| error.to_string())
    }) {
        Ok(runtime) => {
            if env_flag("PVLC_REQUIRE_M4_METAL") {
                assert_eq!(runtime.capabilities().backend, BackendKind::Metal);
                assert!(runtime.capabilities().adapter_name.contains("M4 Pro"));
            }
            Some(runtime)
        }
        Err(error) if env_flag("PVLC_REQUIRE_NATIVE_GPU") || env_flag("PVLC_REQUIRE_M4_METAL") => {
            panic!("native GPU is required: {error}")
        }
        Err(error) => {
            eprintln!("skipping native add: {error}");
            None
        }
    }
}

fn values(length: usize, seed: f32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let phase = (index as f32 + 1.0) * seed;
            phase.sin() * 0.75 - phase.cos() * 0.25
        })
        .collect()
}

#[test]
fn native_add_matches_cpu_exactly_across_workgroup_tails_and_signs() {
    let Some(runtime) = runtime() else { return };
    for length in [1, 63, 64, 65, 257] {
        let left = values(length, 0.017);
        let right = values(length, 0.031);
        if length > 1 {
            assert!(left.iter().any(|value| *value < 0.0));
            assert!(left.iter().any(|value| *value > 0.0));
            assert!(right.iter().any(|value| *value < 0.0));
            assert!(right.iter().any(|value| *value > 0.0));
        }
        let expected = add_vectors_f32(&left, &right).unwrap();
        if length > 1 {
            assert!(expected.iter().any(|value| *value < 0.0));
            assert!(expected.iter().any(|value| *value > 0.0));
        }
        let execution = runtime
            .run(&KernelInvocation::AddF32 { left, right })
            .unwrap();
        assert_eq!(execution.values, expected);
        assert_eq!(execution.diagnostics.kernel, KernelId::AddF32);
        assert_eq!(
            execution.diagnostics.checked_error_scopes,
            [
                ErrorScopeKind::Validation,
                ErrorScopeKind::OutOfMemory,
                ErrorScopeKind::Internal,
            ]
        );
        assert!(execution.diagnostics.captured_errors.is_empty());
        assert!(execution.diagnostics.queue_wall_time_ns > 0);
        let source = pvlc_wgsl::module(KernelId::AddF32).unwrap().source;
        assert_eq!(
            execution.diagnostics.shader_blake3,
            *blake3::hash(source.as_bytes()).as_bytes()
        );
        if env_flag("PVLC_REQUIRE_TIMESTAMP_QUERY") {
            assert!(runtime.capabilities().timestamp_query);
        }
        if runtime.capabilities().timestamp_query {
            let timestamp = execution.diagnostics.timestamp.unwrap();
            assert!(timestamp.end_ticks > timestamp.begin_ticks);
            assert!(timestamp.duration_ns.is_finite() && timestamp.duration_ns > 0.0);
        } else {
            assert!(execution.diagnostics.timestamp.is_none());
        }
    }
}
