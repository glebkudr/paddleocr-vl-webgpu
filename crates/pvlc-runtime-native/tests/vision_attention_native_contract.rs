use std::{env, sync::OnceLock};

use pvlc_cpu_ref::{KvBlockOrder, streaming_segmented_attention_f32};
use pvlc_runtime_core::{KernelId, KernelInvocation};
use pvlc_runtime_native::{BackendKind, ErrorScopeKind, NativeOptions, NativeRuntime};
use pvlc_testkit::{ComparisonAxes, ComparisonPolicy, compare_f32};

static RUNTIME: OnceLock<Result<NativeRuntime, String>> = OnceLock::new();

fn env_flag(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1" | "true"))
}

fn hardware_required() -> bool {
    env_flag("PVLC_REQUIRE_NATIVE_GPU") || env_flag("PVLC_REQUIRE_M4_METAL")
}

fn runtime() -> Option<&'static NativeRuntime> {
    match RUNTIME.get_or_init(|| {
        NativeRuntime::new(NativeOptions::default()).map_err(|error| error.to_string())
    }) {
        Ok(runtime) => {
            if env_flag("PVLC_REQUIRE_M4_METAL") {
                assert_eq!(runtime.capabilities().backend, BackendKind::Metal);
                assert!(
                    runtime.capabilities().adapter_name.contains("M4 Pro"),
                    "expected an M4 Pro adapter, got {}",
                    runtime.capabilities().adapter_name
                );
            }
            Some(runtime)
        }
        Err(error) if hardware_required() => panic!("native GPU is required: {error}"),
        Err(error) => {
            eprintln!("skipping native GPU contract because no adapter is available: {error}");
            None
        }
    }
}

fn fixture(tokens: usize, heads: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = tokens * heads * head_dim;
    let query = (0..len)
        .map(|index| ((index * 17 + 3) as f32 * 0.071).sin())
        .collect();
    let key = (0..len)
        .map(|index| ((index * 29 + 11) as f32 * 0.037).cos())
        .collect();
    let value = (0..len)
        .map(|index| ((index * 13 + 5) as f32 * 0.053).sin() * 2.0)
        .collect();
    (query, key, value)
}

fn invocation(
    tokens: usize,
    heads: usize,
    head_dim: usize,
    cu_seqlens: &[usize],
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
) -> KernelInvocation {
    KernelInvocation::VisionAttentionF32 {
        tokens: tokens as u32,
        heads: heads as u32,
        head_dim: head_dim as u32,
        query,
        key,
        value,
        cu_seqlens: cu_seqlens.iter().map(|value| *value as u32).collect(),
    }
}

fn policy() -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs: 3.0e-4,
        max_mean_abs: 8.0e-5,
        max_p99_abs: 2.0e-4,
        max_relative_l2: 1.0e-4,
        min_cosine_similarity: 0.999_99,
        max_per_token_relative_l2: None,
        max_per_channel_relative_l2: None,
    }
}

fn run(runtime: &NativeRuntime, invocation: &KernelInvocation) -> Vec<f32> {
    let execution = runtime.run(invocation).unwrap();
    assert_eq!(execution.diagnostics.kernel, KernelId::VisionAttentionF32);
    let source = pvlc_wgsl::module(KernelId::VisionAttentionF32)
        .unwrap()
        .source;
    assert_eq!(
        execution.diagnostics.shader_blake3,
        *blake3::hash(source.as_bytes()).as_bytes()
    );
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
    if env_flag("PVLC_REQUIRE_TIMESTAMP_QUERY") {
        assert!(
            runtime.capabilities().timestamp_query,
            "the required laboratory adapter must expose timestamp queries"
        );
    }
    if runtime.capabilities().timestamp_query {
        let timestamp = execution
            .diagnostics
            .timestamp
            .expect("timestamp-capable adapters must emit timing diagnostics");
        assert!(timestamp.end_ticks > timestamp.begin_ticks);
        assert!(timestamp.period_ns.is_finite() && timestamp.period_ns > 0.0);
        assert!(timestamp.duration_ns.is_finite() && timestamp.duration_ns > 0.0);
    } else {
        assert!(execution.diagnostics.timestamp.is_none());
    }
    execution.values
}

#[test]
fn native_streaming_attention_matches_cpu_for_required_sequences_and_dispatch_tails() {
    let Some(runtime) = runtime() else { return };
    for tokens in [8, 16, 31, 64, 127, 256] {
        let heads = 16;
        let head_dim = 72;
        let boundaries = [0, tokens];
        let (query, key, value) = fixture(tokens, heads, head_dim);
        let expected = streaming_segmented_attention_f32(
            &query,
            &key,
            &value,
            tokens,
            heads,
            head_dim,
            &boundaries,
            17,
            KvBlockOrder::Forward,
        )
        .unwrap();
        assert_eq!(expected.len(), tokens * heads * head_dim);
        let actual = run(
            runtime,
            &invocation(tokens, heads, head_dim, &boundaries, query, key, value),
        );
        assert_eq!(actual.len(), tokens * heads * head_dim);
        let report = compare_f32(
            &expected,
            &actual,
            &[tokens, heads, head_dim],
            ComparisonAxes::default(),
        )
        .unwrap();
        let verdict = report.assess(&policy()).unwrap();
        assert!(
            verdict.passed(),
            "S={tokens} failed: {report:?}; {:?}",
            verdict.violations()
        );
    }
}

#[test]
fn native_streaming_attention_runs_the_real_vision_head_geometry_and_packed_segments() {
    let Some(runtime) = runtime() else { return };
    let (tokens, heads, head_dim) = (31, 16, 72);
    let boundaries = [0, 3, 11, 31];
    let (query, key, value) = fixture(tokens, heads, head_dim);
    let expected = streaming_segmented_attention_f32(
        &query,
        &key,
        &value,
        tokens,
        heads,
        head_dim,
        &boundaries,
        7,
        KvBlockOrder::Forward,
    )
    .unwrap();
    assert_eq!(expected.len(), tokens * heads * head_dim);
    let actual = run(
        runtime,
        &invocation(tokens, heads, head_dim, &boundaries, query, key, value),
    );
    assert_eq!(actual.len(), tokens * heads * head_dim);
    let report = compare_f32(
        &expected,
        &actual,
        &[tokens, heads, head_dim],
        ComparisonAxes::default(),
    )
    .unwrap();
    assert!(report.assess(&policy()).unwrap().passed(), "{report:?}");
}

#[test]
fn native_segment_boundaries_are_bidirectionally_isolated() {
    let Some(runtime) = runtime() else { return };
    let (tokens, heads, head_dim) = (17, 2, 8);
    let boundaries = [0, 3, 9, 17];
    let (query, key, value) = fixture(tokens, heads, head_dim);
    let baseline = run(
        runtime,
        &invocation(
            tokens,
            heads,
            head_dim,
            &boundaries,
            query.clone(),
            key.clone(),
            value.clone(),
        ),
    );
    assert_eq!(baseline.len(), tokens * heads * head_dim);

    for poisoned_segment in 0..boundaries.len() - 1 {
        let mut poisoned_key = key.clone();
        let mut poisoned_value = value.clone();
        let start = boundaries[poisoned_segment] * heads * head_dim;
        let end = boundaries[poisoned_segment + 1] * heads * head_dim;
        for index in start..end {
            poisoned_key[index] = poisoned_key[index] * -31.0 + 7.0;
            poisoned_value[index] = poisoned_value[index] * 47.0 - 11.0;
        }
        let poisoned = run(
            runtime,
            &invocation(
                tokens,
                heads,
                head_dim,
                &boundaries,
                query.clone(),
                poisoned_key,
                poisoned_value,
            ),
        );
        assert_eq!(poisoned.len(), tokens * heads * head_dim);
        for segment in 0..boundaries.len() - 1 {
            let start = boundaries[segment] * heads * head_dim;
            let end = boundaries[segment + 1] * heads * head_dim;
            if segment == poisoned_segment {
                assert_ne!(&baseline[start..end], &poisoned[start..end]);
            } else {
                assert_eq!(&baseline[start..end], &poisoned[start..end]);
            }
        }
    }
}
