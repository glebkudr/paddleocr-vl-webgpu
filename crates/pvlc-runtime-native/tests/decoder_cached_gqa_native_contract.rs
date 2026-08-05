use std::{
    collections::BTreeMap,
    env,
    sync::{Mutex, OnceLock},
};

use pvlc_cpu_ref::decode_gqa_f32;
use pvlc_runtime_core::{DecoderCachedGqaInvocation, DecoderCachedGqaStage, KernelId};
use pvlc_runtime_native::{
    BackendKind, DecoderCachedGqaBufferEvidence, DecoderCachedGqaBufferRole,
    DecoderCachedGqaCopyPurpose, DecoderCachedGqaOperationEvidence, ErrorScopeKind, NativeOptions,
    NativeRuntime, RuntimeErrorCode,
};
use pvlc_testkit::{ComparisonAxes, ComparisonPolicy, compare_f32};

const QUERY_HEADS: usize = 16;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const KV_WIDTH: usize = KEY_VALUE_HEADS * HEAD_DIM;
const QUERY_WIDTH: usize = QUERY_HEADS * HEAD_DIM;
const POISON_BITS: u32 = 0x7fc0_51a7;

static RUNTIME: OnceLock<Result<NativeRuntime, String>> = OnceLock::new();
static TEST_LOCK: Mutex<()> = Mutex::new(());

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

fn serial_test_guard() -> std::sync::MutexGuard<'static, ()> {
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Clone)]
struct Case {
    prefix_tokens: usize,
    cache_capacity: usize,
    query: Vec<f32>,
    appended_key: Vec<f32>,
    appended_value: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

impl Case {
    fn new(prefix_tokens: usize, cache_capacity: usize, seed: usize) -> Self {
        assert!(prefix_tokens > 0);
        assert!(cache_capacity > prefix_tokens);
        let valid_prefix_elements = prefix_tokens * KV_WIDTH;
        let cache_elements = cache_capacity * KV_WIDTH;
        let mut key_cache = vec![f32::from_bits(POISON_BITS); cache_elements];
        let mut value_cache = vec![f32::from_bits(POISON_BITS); cache_elements];
        for index in 0..valid_prefix_elements {
            key_cache[index] = patterned(index, seed + 3, 0.019);
            value_cache[index] = patterned(index, seed + 11, 0.023) * 1.7;
        }
        Self {
            prefix_tokens,
            cache_capacity,
            query: (0..QUERY_WIDTH)
                .map(|index| patterned(index, seed + 17, 0.013))
                .collect(),
            appended_key: (0..KV_WIDTH)
                .map(|index| patterned(index, seed + 23, 0.017))
                .collect(),
            appended_value: (0..KV_WIDTH)
                .map(|index| patterned(index, seed + 31, 0.029) * 1.3)
                .collect(),
            key_cache,
            value_cache,
        }
    }

    fn invocation(&self) -> DecoderCachedGqaInvocation<'_> {
        DecoderCachedGqaInvocation {
            query_heads: QUERY_HEADS as u32,
            key_value_heads: KEY_VALUE_HEADS as u32,
            head_dim: HEAD_DIM as u32,
            prefix_tokens: self.prefix_tokens as u32,
            cache_capacity: self.cache_capacity as u32,
            query: &self.query,
            appended_key: &self.appended_key,
            appended_value: &self.appended_value,
            key_cache: &self.key_cache,
            value_cache: &self.value_cache,
        }
    }

    fn expected_caches(&self) -> (Vec<f32>, Vec<f32>) {
        let mut keys = self.key_cache.clone();
        let mut values = self.value_cache.clone();
        let append_start = self.prefix_tokens * KV_WIDTH;
        keys[append_start..append_start + KV_WIDTH].copy_from_slice(&self.appended_key);
        values[append_start..append_start + KV_WIDTH].copy_from_slice(&self.appended_value);
        (keys, values)
    }

    fn expected_attention(&self) -> Vec<f32> {
        let (keys, values) = self.expected_caches();
        let valid_elements = (self.prefix_tokens + 1) * KV_WIDTH;
        decode_gqa_f32(
            &self.query,
            &keys[..valid_elements],
            &values[..valid_elements],
            self.prefix_tokens + 1,
            QUERY_HEADS,
            KEY_VALUE_HEADS,
            HEAD_DIM,
        )
        .unwrap()
    }
}

fn patterned(index: usize, seed: usize, scale: f32) -> f32 {
    (((index * 37 + seed * 19 + 7) % 4_093) as f32 * scale).sin()
}

fn policy() -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs: 4.0e-4,
        max_mean_abs: 6.0e-5,
        max_p99_abs: 2.5e-4,
        max_relative_l2: 1.5e-4,
        min_cosine_similarity: 0.999_99,
        max_per_token_relative_l2: None,
        max_per_channel_relative_l2: None,
    }
}

fn assert_bits_eq(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label}[{index}] differs: actual={actual:?}, expected={expected:?}"
        );
    }
}

#[test]
fn native_append_then_direct_gqa_matches_cpu_and_never_reads_or_clobbers_poison_tail() {
    let _guard = serial_test_guard();
    let Some(runtime) = runtime() else { return };

    for (case_index, prefix_tokens) in [1, 2, 15, 16, 17, 31, 32, 33, 127, 332, 2_047]
        .into_iter()
        .enumerate()
    {
        let case = Case::new(prefix_tokens, prefix_tokens + 4, case_index + 1);
        let original_query = case.query.clone();
        let original_appended_key = case.appended_key.clone();
        let original_appended_value = case.appended_value.clone();
        let original_key_cache = case.key_cache.clone();
        let original_value_cache = case.value_cache.clone();
        let (expected_keys, expected_values) = case.expected_caches();
        let expected_attention = case.expected_attention();
        let valid_tokens = prefix_tokens + 1;
        let valid_elements = valid_tokens * KV_WIDTH;
        let repeated_keys =
            explicitly_repeat_kv_heads(&expected_keys[..valid_elements], valid_tokens);
        let repeated_values =
            explicitly_repeat_kv_heads(&expected_values[..valid_elements], valid_tokens);
        let explicit_repeat_attention = decode_gqa_f32(
            &case.query,
            &repeated_keys,
            &repeated_values,
            valid_tokens,
            QUERY_HEADS,
            QUERY_HEADS,
            HEAD_DIM,
        )
        .unwrap();
        assert_bits_eq(
            &expected_attention,
            &explicit_repeat_attention,
            "direct CPU GQA vs explicit KV repeat",
        );

        let execution = runtime.run_decoder_cached_gqa(&case.invocation()).unwrap();
        assert_eq!(execution.cache_tokens, prefix_tokens as u32 + 1);
        assert_eq!(execution.cache_capacity, (prefix_tokens + 4) as u32);
        assert_bits_eq(&execution.key_cache, &expected_keys, "key cache");
        assert_bits_eq(&execution.value_cache, &expected_values, "value cache");

        let report = compare_f32(
            &expected_attention,
            &execution.attention,
            &[1, QUERY_HEADS, HEAD_DIM],
            ComparisonAxes::default(),
        )
        .unwrap();
        let verdict = report.assess(&policy()).unwrap();
        assert!(
            verdict.passed(),
            "prefix={prefix_tokens} failed: {report:?}; {:?}",
            verdict.violations()
        );

        assert_bits_eq(&case.query, &original_query, "host query");
        assert_bits_eq(
            &case.appended_key,
            &original_appended_key,
            "host appended key",
        );
        assert_bits_eq(
            &case.appended_value,
            &original_appended_value,
            "host appended value",
        );
        assert_bits_eq(&case.key_cache, &original_key_cache, "host key cache");
        assert_bits_eq(&case.value_cache, &original_value_cache, "host value cache");
        assert!(
            execution.attention.iter().all(|value| value.is_finite()),
            "poison outside cache_tokens must not reach attention"
        );

        let mut swapped_keys = expected_keys.clone();
        let mut swapped_values = expected_values.clone();
        for token in 0..=prefix_tokens {
            let row = token * KV_WIDTH;
            let (left, right) = swapped_keys[row..row + KV_WIDTH].split_at_mut(HEAD_DIM);
            left.swap_with_slice(right);
            let (left, right) = swapped_values[row..row + KV_WIDTH].split_at_mut(HEAD_DIM);
            left.swap_with_slice(right);
        }
        let wrong_grouping = decode_gqa_f32(
            &case.query,
            &swapped_keys[..valid_elements],
            &swapped_values[..valid_elements],
            prefix_tokens + 1,
            QUERY_HEADS,
            KEY_VALUE_HEADS,
            HEAD_DIM,
        )
        .unwrap();
        let correct_error = max_abs_diff(&execution.attention, &expected_attention);
        let wrong_error = max_abs_diff(&execution.attention, &wrong_grouping);
        assert!(
            wrong_error > correct_error * 100.0 + 1.0e-3,
            "all contiguous query-head groups must select the direct KV head: correct={correct_error}, wrong={wrong_error}"
        );
    }
}

#[test]
fn one_call_has_two_ordered_dispatches_one_submit_one_map_and_repeated_requests_are_isolated() {
    let _guard = serial_test_guard();
    let Some(runtime) = runtime() else { return };
    let first = Case::new(17, 23, 101);
    let other = Case::new(5, 11, 909);

    let before = runtime.counters();
    let first_execution = runtime.run_decoder_cached_gqa(&first.invocation()).unwrap();
    let after = runtime.counters();
    assert_eq!(after.submissions - before.submissions, 1);
    assert_eq!(
        after.command_encoder_creations - before.command_encoder_creations,
        1
    );
    assert_eq!(after.dispatch_encodings - before.dispatch_encodings, 2);
    assert_eq!(
        after.buffer_copy_encodings - before.buffer_copy_encodings,
        3
    );
    assert_eq!(after.map_requests - before.map_requests, 1);
    assert_eq!(after.bind_group_creations - before.bind_group_creations, 2);
    assert_eq!(after.buffer_allocations - before.buffer_allocations, 9);

    let diagnostics = &first_execution.diagnostics;
    assert_eq!(
        diagnostics.dispatch_stages,
        [
            DecoderCachedGqaStage::AppendKeyValue,
            DecoderCachedGqaStage::DirectGqa,
        ]
    );
    assert_eq!(diagnostics.dispatch_count, 2);
    assert_eq!(diagnostics.compute_pass_count, 2);
    assert_eq!(diagnostics.command_buffer_count, 1);
    assert_eq!(diagnostics.submission_count, 1);
    assert_eq!(diagnostics.readback_buffer_count, 1);
    assert_eq!(diagnostics.readback_map_count, 1);
    assert_eq!(
        diagnostics.readback_bytes,
        ((QUERY_WIDTH + 2 * first.cache_capacity * KV_WIDTH) * size_of::<f32>()) as u64
    );
    assert_eq!(
        diagnostics.checked_error_scopes,
        [
            ErrorScopeKind::Validation,
            ErrorScopeKind::OutOfMemory,
            ErrorScopeKind::Internal,
        ]
    );
    assert!(diagnostics.captured_errors.is_empty());
    assert!(diagnostics.queue_wall_time_ns > 0);
    assert_eq!(diagnostics.shader_blake3.len(), 2);
    for kernel in [KernelId::DecoderKvAppendF32, KernelId::DecoderGqaF32] {
        let source = pvlc_wgsl::module(kernel).unwrap().source;
        assert_eq!(
            diagnostics.shader_blake3[&kernel],
            *blake3::hash(source.as_bytes()).as_bytes()
        );
    }
    assert_operation_evidence(&diagnostics.operation_evidence, &first);

    let other_execution = runtime.run_decoder_cached_gqa(&other.invocation()).unwrap();
    let repeated = runtime.run_decoder_cached_gqa(&first.invocation()).unwrap();
    assert_attention_matches(
        &first.expected_attention(),
        &first_execution.attention,
        "first request",
    );
    assert_attention_matches(
        &other.expected_attention(),
        &other_execution.attention,
        "intervening request",
    );
    assert_attention_matches(
        &first.expected_attention(),
        &repeated.attention,
        "repeated request",
    );
    assert_bits_eq(
        &repeated.attention,
        &first_execution.attention,
        "repeat attention",
    );
    assert_bits_eq(
        &repeated.key_cache,
        &first_execution.key_cache,
        "repeat key cache",
    );
    assert_bits_eq(
        &repeated.value_cache,
        &first_execution.value_cache,
        "repeat value cache",
    );
    assert_ne!(
        first_execution
            .attention
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        other_execution
            .attention
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "an intervening request must execute its own data"
    );
}

#[test]
fn shader_overrides_prove_that_both_gpu_dispatches_are_causal_not_dummy_or_cpu_shadow_work() {
    let _guard = serial_test_guard();
    let Some(runtime) = runtime() else { return };
    let case = Case::new(3, 7, 77);
    let append_module = pvlc_wgsl::module(KernelId::DecoderKvAppendF32).unwrap();
    let attention_module = pvlc_wgsl::module(KernelId::DecoderGqaF32).unwrap();

    let append_source = append_module
        .source
        .replace(
            "key_cache.data[cache_index] = appended_key.data[linear];",
            "key_cache.data[cache_index] = appended_key.data[linear] + 4.0;",
        )
        .replace(
            "value_cache.data[cache_index] = appended_value.data[linear];",
            "value_cache.data[cache_index] = appended_value.data[linear] - 3.0;",
        );
    assert_ne!(append_source, append_module.source);
    validate_override(append_module, &append_source);

    let append_only = BTreeMap::from([(KernelId::DecoderKvAppendF32, append_source.clone())]);
    let append_execution = runtime
        .run_decoder_cached_gqa_with_shader_overrides(&case.invocation(), &append_only)
        .unwrap();
    let (mut expected_keys, mut expected_values) = case.expected_caches();
    let append_start = case.prefix_tokens * KV_WIDTH;
    for index in 0..KV_WIDTH {
        expected_keys[append_start + index] = case.appended_key[index] + 4.0;
        expected_values[append_start + index] = case.appended_value[index] - 3.0;
    }
    assert_bits_eq(
        &append_execution.key_cache,
        &expected_keys,
        "append-override key cache",
    );
    assert_bits_eq(
        &append_execution.value_cache,
        &expected_values,
        "append-override value cache",
    );
    let valid_elements = (case.prefix_tokens + 1) * KV_WIDTH;
    let append_oracle = decode_gqa_f32(
        &case.query,
        &expected_keys[..valid_elements],
        &expected_values[..valid_elements],
        case.prefix_tokens + 1,
        QUERY_HEADS,
        KEY_VALUE_HEADS,
        HEAD_DIM,
    )
    .unwrap();
    assert_attention_matches(
        &append_oracle,
        &append_execution.attention,
        "append-only nonce must feed canonical GQA",
    );
    assert!(
        max_abs_diff(&append_execution.attention, &case.expected_attention()) > 1.0e-2,
        "append-only shader mutation must causally alter canonical GQA"
    );
    assert_eq!(
        append_execution.diagnostics.shader_blake3[&KernelId::DecoderKvAppendF32],
        *blake3::hash(append_source.as_bytes()).as_bytes()
    );
    assert_eq!(
        append_execution.diagnostics.shader_blake3[&KernelId::DecoderGqaF32],
        *blake3::hash(attention_module.source.as_bytes()).as_bytes()
    );

    let attention_source = attention_module.source.replace(
        "output.data[output_base + dimension] = weighted[dimension] / denominator;",
        "output.data[output_base + dimension] = 8192.0 + f32(query_head * 128u + dimension);",
    );
    assert_ne!(attention_source, attention_module.source);
    validate_override(attention_module, &attention_source);

    let attention_only = BTreeMap::from([(KernelId::DecoderGqaF32, attention_source.clone())]);
    let attention_execution = runtime
        .run_decoder_cached_gqa_with_shader_overrides(&case.invocation(), &attention_only)
        .unwrap();
    let (canonical_keys, canonical_values) = case.expected_caches();
    assert_bits_eq(
        &attention_execution.key_cache,
        &canonical_keys,
        "attention-override key cache",
    );
    assert_bits_eq(
        &attention_execution.value_cache,
        &canonical_values,
        "attention-override value cache",
    );
    for (index, actual) in attention_execution.attention.iter().enumerate() {
        assert_eq!(actual.to_bits(), (8192.0 + index as f32).to_bits());
    }
    assert_eq!(
        attention_execution.diagnostics.shader_blake3[&KernelId::DecoderKvAppendF32],
        *blake3::hash(append_module.source.as_bytes()).as_bytes()
    );
    assert_eq!(
        attention_execution.diagnostics.shader_blake3[&KernelId::DecoderGqaF32],
        *blake3::hash(attention_source.as_bytes()).as_bytes()
    );

    let missing_scale_source = attention_module.source.replace(
        "let attention_scale = inverseSqrt(f32(params.head_dim));",
        "let attention_scale = 1.0;",
    );
    assert_ne!(missing_scale_source, attention_module.source);
    validate_override(attention_module, &missing_scale_source);
    let missing_scale = BTreeMap::from([(KernelId::DecoderGqaF32, missing_scale_source)]);
    let wrong = runtime
        .run_decoder_cached_gqa_with_shader_overrides(&case.invocation(), &missing_scale)
        .unwrap();
    assert!(
        max_abs_diff(&wrong.attention, &case.expected_attention()) > 1.0e-2,
        "missing attention scale must be rejected by the behavioral oracle"
    );
}

#[test]
fn every_validation_class_fails_before_any_gpu_side_effect() {
    let _guard = serial_test_guard();
    let Some(runtime) = runtime() else { return };
    let case = Case::new(3, 7, 41);
    let base = case.invocation();

    for invocation in [
        DecoderCachedGqaInvocation {
            query_heads: 0,
            ..base
        },
        DecoderCachedGqaInvocation {
            prefix_tokens: base.cache_capacity,
            ..base
        },
    ] {
        assert_preflight_no_effect(runtime, invocation);
    }
    for (query_heads, key_value_heads, head_dim) in [
        (8, KEY_VALUE_HEADS as u32, HEAD_DIM as u32),
        (QUERY_HEADS as u32, 1, HEAD_DIM as u32),
        (QUERY_HEADS as u32, KEY_VALUE_HEADS as u32, 64),
        (
            QUERY_HEADS as u32,
            KEY_VALUE_HEADS as u32,
            HEAD_DIM as u32 + 1,
        ),
    ] {
        assert_sized_geometry_preflight_no_effect(runtime, query_heads, key_value_heads, head_dim);
    }
    assert_preflight_no_effect(
        runtime,
        DecoderCachedGqaInvocation {
            cache_capacity: u32::MAX,
            key_cache: &[],
            value_cache: &[],
            ..base
        },
    );

    for operand in 0..5 {
        let original = match operand {
            0 => &case.query,
            1 => &case.appended_key,
            2 => &case.appended_value,
            3 => &case.key_cache,
            4 => &case.value_cache,
            _ => unreachable!(),
        };
        for values in wrong_lengths(original) {
            let invocation = match operand {
                0 => DecoderCachedGqaInvocation {
                    query: &values,
                    ..base
                },
                1 => DecoderCachedGqaInvocation {
                    appended_key: &values,
                    ..base
                },
                2 => DecoderCachedGqaInvocation {
                    appended_value: &values,
                    ..base
                },
                3 => DecoderCachedGqaInvocation {
                    key_cache: &values,
                    ..base
                },
                4 => DecoderCachedGqaInvocation {
                    value_cache: &values,
                    ..base
                },
                _ => unreachable!(),
            };
            assert_preflight_no_effect(runtime, invocation);
        }
    }

    for operand in 0..5 {
        let mut poisoned = case.clone();
        match operand {
            0 => poisoned.query[9] = f32::NAN,
            1 => poisoned.appended_key[11] = f32::INFINITY,
            2 => poisoned.appended_value[13] = f32::NEG_INFINITY,
            3 => poisoned.key_cache[case.prefix_tokens * KV_WIDTH - 1] = f32::NAN,
            4 => poisoned.value_cache[case.prefix_tokens * KV_WIDTH - 1] = f32::INFINITY,
            _ => unreachable!(),
        }
        assert_preflight_no_effect(runtime, poisoned.invocation());
    }
}

fn assert_operation_evidence(evidence: &DecoderCachedGqaOperationEvidence, case: &Case) {
    use DecoderCachedGqaBufferRole::{
        AppendUniform, AppendedKey, AppendedValue, AttentionOutput, AttentionUniform, KeyCache,
        Query, Readback, ValueCache,
    };

    assert_eq!(
        evidence
            .buffers
            .iter()
            .map(|buffer| buffer.role)
            .collect::<Vec<_>>(),
        [
            Query,
            AppendedKey,
            AppendedValue,
            KeyCache,
            ValueCache,
            AttentionOutput,
            AppendUniform,
            AttentionUniform,
            Readback,
        ]
    );
    let mut identities = evidence
        .buffers
        .iter()
        .map(|buffer| buffer.buffer_identity)
        .collect::<Vec<_>>();
    assert!(identities.iter().all(|identity| *identity != 0));
    identities.sort_unstable();
    identities.dedup();
    assert_eq!(identities.len(), 9);

    let query_bytes = (QUERY_WIDTH * size_of::<f32>()) as u64;
    let kv_row_bytes = (KV_WIDTH * size_of::<f32>()) as u64;
    let cache_bytes = (case.cache_capacity * KV_WIDTH * size_of::<f32>()) as u64;
    let readback_bytes = query_bytes + 2 * cache_bytes;
    for (role, bytes) in [
        (Query, query_bytes),
        (AppendedKey, kv_row_bytes),
        (AppendedValue, kv_row_bytes),
        (KeyCache, cache_bytes),
        (ValueCache, cache_bytes),
        (AttentionOutput, query_bytes),
        (AppendUniform, 16),
        (AttentionUniform, 16),
        (Readback, readback_bytes),
    ] {
        assert_eq!(buffer(evidence, role).allocation_bytes, bytes, "{role:?}");
    }

    let append_key = buffer(evidence, AppendedKey).buffer_identity;
    let append_value = buffer(evidence, AppendedValue).buffer_identity;
    let key_cache = buffer(evidence, KeyCache).buffer_identity;
    let value_cache = buffer(evidence, ValueCache).buffer_identity;
    let query = buffer(evidence, Query).buffer_identity;
    let attention_output = buffer(evidence, AttentionOutput).buffer_identity;
    let append_uniform = buffer(evidence, AppendUniform).buffer_identity;
    let attention_uniform = buffer(evidence, AttentionUniform).buffer_identity;
    let readback = buffer(evidence, Readback).buffer_identity;

    assert_eq!(evidence.bind_groups.len(), 2);
    assert_eq!(
        evidence.bind_groups[0]
            .bindings
            .iter()
            .map(|binding| {
                (
                    binding.binding,
                    binding.buffer_identity,
                    binding.byte_offset,
                    binding.byte_length,
                )
            })
            .collect::<Vec<_>>(),
        [
            (0, append_key, 0, kv_row_bytes),
            (1, append_value, 0, kv_row_bytes),
            (2, key_cache, 0, cache_bytes),
            (3, value_cache, 0, cache_bytes),
            (4, append_uniform, 0, 16),
        ]
    );
    assert_eq!(
        evidence.bind_groups[0].stage,
        DecoderCachedGqaStage::AppendKeyValue
    );
    assert_eq!(
        evidence.bind_groups[1]
            .bindings
            .iter()
            .map(|binding| {
                (
                    binding.binding,
                    binding.buffer_identity,
                    binding.byte_offset,
                    binding.byte_length,
                )
            })
            .collect::<Vec<_>>(),
        [
            (0, query, 0, query_bytes),
            (1, key_cache, 0, cache_bytes),
            (2, value_cache, 0, cache_bytes),
            (3, attention_output, 0, query_bytes),
            (4, attention_uniform, 0, 16),
        ]
    );
    assert_eq!(
        evidence.bind_groups[1].stage,
        DecoderCachedGqaStage::DirectGqa
    );

    assert_eq!(
        evidence
            .dispatches
            .iter()
            .map(|dispatch| {
                (
                    dispatch.ordinal,
                    dispatch.stage,
                    dispatch.kernel,
                    dispatch.workgroups,
                )
            })
            .collect::<Vec<_>>(),
        [
            (
                1,
                DecoderCachedGqaStage::AppendKeyValue,
                KernelId::DecoderKvAppendF32,
                [4, 1, 1],
            ),
            (
                2,
                DecoderCachedGqaStage::DirectGqa,
                KernelId::DecoderGqaF32,
                [1, 1, 1],
            ),
        ]
    );

    assert_eq!(
        evidence
            .copies
            .iter()
            .map(|copy| {
                (
                    copy.ordinal,
                    copy.source_buffer_identity,
                    copy.source_offset,
                    copy.destination_buffer_identity,
                    copy.destination_offset,
                    copy.byte_length,
                    copy.purpose,
                    copy.after_dispatch_ordinal,
                )
            })
            .collect::<Vec<_>>(),
        [
            (
                1,
                attention_output,
                0,
                readback,
                0,
                query_bytes,
                DecoderCachedGqaCopyPurpose::Attention,
                2,
            ),
            (
                2,
                key_cache,
                0,
                readback,
                query_bytes,
                cache_bytes,
                DecoderCachedGqaCopyPurpose::KeyCache,
                2,
            ),
            (
                3,
                value_cache,
                0,
                readback,
                query_bytes + cache_bytes,
                cache_bytes,
                DecoderCachedGqaCopyPurpose::ValueCache,
                2,
            ),
        ]
    );
    assert_eq!(evidence.maps.len(), 1);
    assert_eq!(evidence.maps[0].buffer_identity, readback);
    assert_eq!(evidence.maps[0].byte_offset, 0);
    assert_eq!(evidence.maps[0].byte_length, readback_bytes);
    assert_eq!(evidence.maps[0].after_copy_ordinal, 3);
}

fn buffer(
    evidence: &DecoderCachedGqaOperationEvidence,
    role: DecoderCachedGqaBufferRole,
) -> &DecoderCachedGqaBufferEvidence {
    evidence
        .buffers
        .iter()
        .find(|buffer| buffer.role == role)
        .unwrap()
}

fn assert_attention_matches(expected: &[f32], actual: &[f32], label: &str) {
    let report = compare_f32(
        expected,
        actual,
        &[1, QUERY_HEADS, HEAD_DIM],
        ComparisonAxes::default(),
    )
    .unwrap();
    let verdict = report.assess(&policy()).unwrap();
    assert!(
        verdict.passed(),
        "{label} failed: {report:?}; {:?}",
        verdict.violations()
    );
}

fn assert_preflight_no_effect(runtime: &NativeRuntime, invocation: DecoderCachedGqaInvocation<'_>) {
    let before = runtime.counters();
    let error = runtime.run_decoder_cached_gqa(&invocation).unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::InvalidInvocation);
    assert_eq!(
        runtime.counters(),
        before,
        "invalid invocation reached an observable GPU side effect: {error}"
    );
}

fn assert_sized_geometry_preflight_no_effect(
    runtime: &NativeRuntime,
    query_heads: u32,
    key_value_heads: u32,
    head_dim: u32,
) {
    let prefix_tokens = 3_u32;
    let cache_capacity = 7_u32;
    let query = vec![0.25; query_heads as usize * head_dim as usize];
    let appended_key = vec![0.5; key_value_heads as usize * head_dim as usize];
    let appended_value = vec![-0.75; key_value_heads as usize * head_dim as usize];
    let cache_elements = cache_capacity as usize * key_value_heads as usize * head_dim as usize;
    let key_cache = vec![0.125; cache_elements];
    let value_cache = vec![-0.375; cache_elements];
    assert_preflight_no_effect(
        runtime,
        DecoderCachedGqaInvocation {
            query_heads,
            key_value_heads,
            head_dim,
            prefix_tokens,
            cache_capacity,
            query: &query,
            appended_key: &appended_key,
            appended_value: &appended_value,
            key_cache: &key_cache,
            value_cache: &value_cache,
        },
    );
}

fn wrong_lengths(values: &[f32]) -> [Vec<f32>; 2] {
    let mut short = values.to_vec();
    short.pop();
    let mut long = values.to_vec();
    long.push(0.0);
    [short, long]
}

fn validate_override(module: &pvlc_wgsl::KernelModule, source: &str) {
    pvlc_wgsl::validate_source_contract(&module.spec, source).unwrap();
}

fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

fn explicitly_repeat_kv_heads(cache: &[f32], tokens: usize) -> Vec<f32> {
    let query_heads_per_kv = QUERY_HEADS / KEY_VALUE_HEADS;
    let mut repeated = vec![0.0; tokens * QUERY_HEADS * HEAD_DIM];
    for token in 0..tokens {
        for query_head in 0..QUERY_HEADS {
            let key_value_head = query_head / query_heads_per_kv;
            let source = (token * KEY_VALUE_HEADS + key_value_head) * HEAD_DIM;
            let destination = (token * QUERY_HEADS + query_head) * HEAD_DIM;
            repeated[destination..destination + HEAD_DIM]
                .copy_from_slice(&cache[source..source + HEAD_DIM]);
        }
    }
    repeated
}
