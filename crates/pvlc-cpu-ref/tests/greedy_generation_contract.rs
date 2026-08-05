//! M6c2 tests-first contract for greedy, one-token-at-a-time generation.
//!
//! Decode chunks, random split scheduling, and WebGPU/runtime execution are
//! intentionally outside this contract.

use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, DecoderPrefillKvCache, GreedyDecodeStep, GreedyGenerationConfig,
    GreedyGenerationTrace, GreedyStopReason, TopKEntry, greedy_generate_f32,
    pinned_greedy_generate_f32, top_k,
};

fn entry(index: usize, value: f32) -> TopKEntry {
    TopKEntry { index, value }
}

fn config(layers: usize, max_new_tokens: usize, eos_token_id: usize) -> GreedyGenerationConfig {
    GreedyGenerationConfig {
        layers,
        vocab_size: 8,
        key_value_heads: 1,
        head_dim: 2,
        max_new_tokens,
        eos_token_id,
    }
}

fn cache(tokens: usize, keys: &[f32], values: &[f32]) -> DecoderPrefillKvCache {
    assert_eq!(keys.len(), tokens * 2);
    assert_eq!(values.len(), tokens * 2);
    DecoderPrefillKvCache {
        keys: keys.to_vec(),
        values: values.to_vec(),
        tokens,
        key_value_heads: 1,
        head_dim: 2,
    }
}

fn synthetic_cache(tokens: usize, layer: usize, generation: usize) -> DecoderPrefillKvCache {
    shaped_cache(tokens, 1, 2, layer, generation)
}

fn shaped_cache(
    tokens: usize,
    key_value_heads: usize,
    head_dim: usize,
    layer: usize,
    generation: usize,
) -> DecoderPrefillKvCache {
    let len = tokens * key_value_heads * head_dim;
    let base = (generation * 100 + layer * 20) as f32;
    DecoderPrefillKvCache {
        keys: (0..len).map(|index| base + index as f32 / 16.0).collect(),
        values: (0..len).map(|index| -base - index as f32 / 32.0).collect(),
        tokens,
        key_value_heads,
        head_dim,
    }
}

fn synthetic_caches(tokens: usize, layers: usize, generation: usize) -> Vec<DecoderPrefillKvCache> {
    (0..layers)
        .map(|layer| synthetic_cache(tokens, layer, generation))
        .collect()
}

fn shaped_caches(
    tokens: usize,
    layers: usize,
    key_value_heads: usize,
    head_dim: usize,
    generation: usize,
) -> Vec<DecoderPrefillKvCache> {
    (0..layers)
        .map(|layer| shaped_cache(tokens, key_value_heads, head_dim, layer, generation))
        .collect()
}

fn full_config(
    layers: usize,
    vocab_size: usize,
    key_value_heads: usize,
    head_dim: usize,
    max_new_tokens: usize,
    eos_token_id: usize,
) -> GreedyGenerationConfig {
    GreedyGenerationConfig {
        layers,
        vocab_size,
        key_value_heads,
        head_dim,
        max_new_tokens,
        eos_token_id,
    }
}

fn assert_error_code<T>(case: &str, result: Result<T, CpuRefError>, expected: CpuRefErrorCode) {
    match result {
        Err(error) => assert_eq!(error.code(), expected, "{case}"),
        Ok(_) => panic!("{case}: expected {expected:?}"),
    }
}

fn assert_rejected_before_callback(
    case: &str,
    prefill_top_k: &[TopKEntry],
    initial_kv_caches: Vec<DecoderPrefillKvCache>,
    generation_config: GreedyGenerationConfig,
    expected: CpuRefErrorCode,
) {
    let mut calls = 0;
    let result = greedy_generate_f32(
        prefill_top_k,
        initial_kv_caches,
        generation_config,
        |_, _, _| -> Result<GreedyDecodeStep, CpuRefError> {
            calls += 1;
            panic!("{case}: invalid request reached callback")
        },
    );
    assert_eq!(calls, 0, "{case}");
    assert_error_code(case, result, expected);
}

fn update_usize(hasher: &mut blake3::Hasher, value: usize) {
    hasher.update(&(value as u64).to_le_bytes());
}

fn update_f32_bits(hasher: &mut blake3::Hasher, values: &[f32]) {
    update_usize(hasher, values.len());
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

// Digest schema: domain; generated-token count and token IDs; final-cache
// count; for each layer, token/head/dimension metadata plus key/value lengths
// and exact f32 bits; stop discriminant (0=max, 1=eos); decode-step count.
fn trace_digest(trace: &GreedyGenerationTrace) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"pvlc-greedy-generation-v1\0");
    update_usize(&mut hasher, trace.generated_tokens.len());
    for token in &trace.generated_tokens {
        update_usize(&mut hasher, *token);
    }
    update_usize(&mut hasher, trace.kv_caches.len());
    for layer in &trace.kv_caches {
        update_usize(&mut hasher, layer.tokens);
        update_usize(&mut hasher, layer.key_value_heads);
        update_usize(&mut hasher, layer.head_dim);
        update_f32_bits(&mut hasher, &layer.keys);
        update_f32_bits(&mut hasher, &layer.values);
    }
    hasher.update(&[match trace.stop_reason {
        GreedyStopReason::MaxNewTokens => 0,
        GreedyStopReason::EosToken => 1,
    }]);
    update_usize(&mut hasher, trace.decode_steps);
    hasher.finalize().to_hex().to_string()
}

#[test]
fn literal_four_token_trace_matches_exact_oracle_and_fixed_digest() {
    let initial = vec![
        cache(2, &[0.10, 0.11, 0.12, 0.13], &[1.10, 1.11, 1.12, 1.13]),
        cache(2, &[2.10, 2.11, 2.12, 2.13], &[3.10, 3.11, 3.12, 3.13]),
    ];
    let after_five = vec![
        cache(
            3,
            &[0.20, 0.21, 0.22, 0.23, 0.24, 0.25],
            &[1.20, 1.21, 1.22, 1.23, 1.24, 1.25],
        ),
        cache(
            3,
            &[2.20, 2.21, 2.22, 2.23, 2.24, 2.25],
            &[3.20, 3.21, 3.22, 3.23, 3.24, 3.25],
        ),
    ];
    let after_one = vec![
        cache(
            4,
            &[0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37],
            &[1.30, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37],
        ),
        cache(
            4,
            &[2.30, 2.31, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37],
            &[3.30, 3.31, 3.32, 3.33, 3.34, 3.35, 3.36, 3.37],
        ),
    ];
    let after_six = vec![
        cache(
            5,
            &[0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49],
            &[1.40, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49],
        ),
        cache(
            5,
            &[2.40, 2.41, 2.42, 2.43, 2.44, 2.45, 2.46, 2.47, 2.48, 2.49],
            &[3.40, 3.41, 3.42, 3.43, 3.44, 3.45, 3.46, 3.47, 3.48, 3.49],
        ),
    ];
    let expected_inputs = [initial.clone(), after_five.clone(), after_one.clone()];
    let expected_outputs = [after_five, after_one, after_six.clone()];
    let expected_input_tokens = [3, 5, 1];
    let expected_top_k = [
        vec![entry(5, 9.5), entry(4, 8.0)],
        vec![entry(1, 7.25), entry(0, 6.0)],
        vec![entry(6, 5.5), entry(2, 4.0)],
    ];
    let mut calls = 0;

    let trace = greedy_generate_f32(
        &[entry(3, 10.0), entry(1, 9.0)],
        initial,
        config(2, 4, 7),
        |step_index: usize, input_token: usize, current_caches: &[DecoderPrefillKvCache]| {
            assert_eq!(step_index, calls);
            assert_eq!(input_token, expected_input_tokens[step_index]);
            assert_eq!(current_caches, expected_inputs[step_index]);
            calls += 1;
            Ok(GreedyDecodeStep {
                top_k: expected_top_k[step_index].clone(),
                kv_caches: expected_outputs[step_index].clone(),
            })
        },
    )
    .unwrap();

    assert_eq!(calls, 3);
    assert_eq!(trace.generated_tokens, [3, 5, 1, 6]);
    assert_eq!(trace.kv_caches, after_six);
    assert_eq!(trace.decode_steps, 3);
    assert_eq!(trace.stop_reason, GreedyStopReason::MaxNewTokens);
    assert_eq!(
        trace_digest(&trace),
        "f830d7fe54c0fb87f45e6a0ae1ba8f2fd9d0ffdfbcfe9bc786f194dd6baf80ba"
    );
}

#[test]
fn max_one_returns_prefill_top_one_and_owned_initial_caches_without_callback() {
    let prefill = vec![entry(4, 3.0), entry(2, 1.0)];
    let preserved_prefill = prefill.clone();
    let initial = synthetic_caches(3, 2, 0);
    let key_pointers: Vec<_> = initial.iter().map(|layer| layer.keys.as_ptr()).collect();
    let value_pointers: Vec<_> = initial.iter().map(|layer| layer.values.as_ptr()).collect();

    let trace = greedy_generate_f32(&prefill, initial, config(2, 1, 7), |_, _, _| {
        panic!("max_new_tokens=1 must not decode")
    })
    .unwrap();

    assert_eq!(prefill, preserved_prefill);
    assert_eq!(trace.generated_tokens, [4]);
    assert_eq!(trace.decode_steps, 0);
    assert_eq!(trace.stop_reason, GreedyStopReason::MaxNewTokens);
    for (layer_index, layer) in trace.kv_caches.iter().enumerate() {
        assert_eq!(layer.keys.as_ptr(), key_pointers[layer_index]);
        assert_eq!(layer.values.as_ptr(), value_pointers[layer_index]);
    }
}

#[test]
fn prefill_eos_stops_before_decode_and_returns_initial_caches() {
    let initial = synthetic_caches(1, 2, 0);
    let expected = initial.clone();
    let mut calls = 0;
    let trace = greedy_generate_f32(
        &[entry(2, 5.0), entry(1, 4.0)],
        initial,
        config(2, 8, 2),
        |_, _, _| -> Result<GreedyDecodeStep, CpuRefError> {
            calls += 1;
            unreachable!("prefill EOS must stop generation")
        },
    )
    .unwrap();

    assert_eq!(calls, 0);
    assert_eq!(trace.generated_tokens, [2]);
    assert_eq!(trace.kv_caches, expected);
    assert_eq!(trace.decode_steps, 0);
    assert_eq!(trace.stop_reason, GreedyStopReason::EosToken);
}

#[test]
fn decode_eos_is_appended_and_stops_without_an_extra_callback() {
    let initial = synthetic_caches(2, 1, 0);
    let mut final_caches = Some(synthetic_caches(3, 1, 1));
    let expected_final = final_caches.as_ref().unwrap().clone();
    let mut calls = 0;
    let trace = greedy_generate_f32(
        &[entry(4, 5.0)],
        initial,
        config(1, 6, 2),
        |step_index: usize, input_token: usize, caches: &[DecoderPrefillKvCache]| {
            calls += 1;
            assert_eq!((step_index, input_token), (0, 4));
            assert_eq!(caches[0].tokens, 2);
            Ok(GreedyDecodeStep {
                top_k: vec![entry(2, 7.0), entry(1, 6.0)],
                kv_caches: final_caches.take().unwrap(),
            })
        },
    )
    .unwrap();

    assert_eq!(calls, 1);
    assert_eq!(trace.generated_tokens, [4, 2]);
    assert_eq!(trace.kv_caches, expected_final);
    assert_eq!(trace.decode_steps, 1);
    assert_eq!(trace.stop_reason, GreedyStopReason::EosToken);
}

#[test]
fn callback_observes_exact_zero_based_step_token_and_current_cache_chain() {
    let initial = synthetic_caches(2, 2, 0);
    let first = synthetic_caches(3, 2, 1);
    let second = synthetic_caches(4, 2, 2);
    let expected_inputs = [initial.clone(), first.clone()];
    let outputs = [first, second.clone()];
    let next_tokens = [6, 1];
    let mut observations = Vec::new();

    let trace = greedy_generate_f32(
        &[entry(4, 5.0)],
        initial,
        config(2, 3, 7),
        |step_index: usize, input_token: usize, caches: &[DecoderPrefillKvCache]| {
            observations.push((
                step_index,
                input_token,
                caches[0].tokens,
                caches[0].keys[0].to_bits(),
                caches[1].values[0].to_bits(),
            ));
            assert_eq!(caches, expected_inputs[step_index]);
            Ok(GreedyDecodeStep {
                top_k: vec![entry(next_tokens[step_index], 10.0 - step_index as f32)],
                kv_caches: outputs[step_index].clone(),
            })
        },
    )
    .unwrap();

    assert_eq!(
        observations,
        [
            (0, 4, 2, 0.0_f32.to_bits(), (-20.0_f32).to_bits()),
            (1, 6, 3, 100.0_f32.to_bits(), (-120.0_f32).to_bits()),
        ]
    );
    assert_eq!(trace.generated_tokens, [4, 6, 1]);
    assert_eq!(trace.kv_caches, second);
    assert_eq!(trace.decode_steps, 2);
    assert_eq!(trace.stop_reason, GreedyStopReason::MaxNewTokens);
}

#[test]
fn callback_error_is_propagated_exactly_and_prevents_later_calls() {
    let expected_error = top_k(&[], 1).unwrap_err();
    let returned_error = expected_error.clone();
    let mut calls = 0;
    let result = greedy_generate_f32(
        &[entry(4, 5.0)],
        synthetic_caches(2, 1, 0),
        config(1, 5, 7),
        |step_index, input_token, caches| -> Result<GreedyDecodeStep, CpuRefError> {
            calls += 1;
            assert_eq!((step_index, input_token, caches[0].tokens), (0, 4, 2));
            Err(returned_error.clone())
        },
    );

    assert_eq!(calls, 1);
    assert_eq!(result.unwrap_err(), expected_error);
}

#[test]
fn malformed_config_and_prefill_candidates_are_rejected_before_callback() {
    let valid_prefill = [entry(4, 5.0), entry(1, 3.0)];
    let invalid_configs = [
        ("zero layers", full_config(0, 8, 1, 2, 2, 7)),
        ("zero vocabulary", full_config(2, 0, 1, 2, 2, 0)),
        ("zero KV heads", full_config(2, 8, 0, 2, 2, 7)),
        ("zero head dimension", full_config(2, 8, 1, 0, 2, 7)),
        ("zero max tokens", full_config(2, 8, 1, 2, 0, 7)),
        ("EOS equals vocabulary", full_config(2, 8, 1, 2, 2, 8)),
        ("EOS exceeds vocabulary", full_config(2, 8, 1, 2, 2, 9)),
    ];
    for (case, invalid_config) in invalid_configs {
        assert_rejected_before_callback(
            case,
            &valid_prefill,
            synthetic_caches(2, 2, 0),
            invalid_config,
            CpuRefErrorCode::DimensionMismatch,
        );
    }

    let malformed_candidates = vec![
        ("empty", vec![], CpuRefErrorCode::DimensionMismatch),
        (
            "longer than vocabulary",
            (0..9)
                .map(|token| entry(token, 20.0 - token as f32))
                .collect(),
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "duplicate ID",
            vec![entry(1, 4.0), entry(1, 3.0)],
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "out of range",
            vec![entry(8, 4.0)],
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "ascending score",
            vec![entry(1, 3.0), entry(2, 4.0)],
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "descending ID for tied score",
            vec![entry(2, 4.0), entry(1, 4.0)],
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "NaN score",
            vec![entry(1, f32::NAN)],
            CpuRefErrorCode::NonFiniteInput,
        ),
        (
            "positive infinite score",
            vec![entry(1, f32::INFINITY)],
            CpuRefErrorCode::NonFiniteInput,
        ),
        (
            "negative infinite score",
            vec![entry(1, f32::NEG_INFINITY)],
            CpuRefErrorCode::NonFiniteInput,
        ),
    ];
    for (case, candidates, expected) in malformed_candidates {
        assert_rejected_before_callback(
            case,
            &candidates,
            synthetic_caches(2, 2, 0),
            config(2, 2, 7),
            expected,
        );
    }
}

#[test]
fn malformed_initial_cache_sets_are_rejected_before_callback() {
    let mut cases = vec![
        (
            "short layer count",
            synthetic_caches(2, 1, 0),
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "long layer count",
            synthetic_caches(2, 3, 0),
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "zero tokens",
            synthetic_caches(0, 2, 0),
            CpuRefErrorCode::DimensionMismatch,
        ),
    ];

    let mut nonuniform = synthetic_caches(2, 2, 0);
    nonuniform[1] = synthetic_cache(3, 1, 0);
    cases.push((
        "nonuniform tokens",
        nonuniform,
        CpuRefErrorCode::DimensionMismatch,
    ));

    let mut wrong_heads = synthetic_caches(2, 2, 0);
    wrong_heads[0] = shaped_cache(2, 2, 2, 0, 0);
    cases.push((
        "shape-consistent wrong KV heads",
        wrong_heads,
        CpuRefErrorCode::DimensionMismatch,
    ));

    let mut wrong_head_dim = synthetic_caches(2, 2, 0);
    wrong_head_dim[1] = shaped_cache(2, 1, 3, 1, 0);
    cases.push((
        "shape-consistent wrong head dimension",
        wrong_head_dim,
        CpuRefErrorCode::DimensionMismatch,
    ));

    let mut short_keys = synthetic_caches(2, 2, 0);
    short_keys[0].keys.pop();
    cases.push((
        "short key storage",
        short_keys,
        CpuRefErrorCode::DimensionMismatch,
    ));

    let mut long_values = synthetic_caches(2, 2, 0);
    long_values[1].values.push(0.0);
    cases.push((
        "long value storage",
        long_values,
        CpuRefErrorCode::DimensionMismatch,
    ));

    let mut nonfinite_keys = synthetic_caches(2, 2, 0);
    nonfinite_keys[0].keys[2] = f32::NAN;
    cases.push((
        "nonfinite keys",
        nonfinite_keys,
        CpuRefErrorCode::NonFiniteInput,
    ));

    let mut nonfinite_values = synthetic_caches(2, 2, 0);
    nonfinite_values[1].values[1] = f32::NEG_INFINITY;
    cases.push((
        "nonfinite values",
        nonfinite_values,
        CpuRefErrorCode::NonFiniteInput,
    ));

    let overflowing_cache = DecoderPrefillKvCache {
        keys: vec![],
        values: vec![],
        tokens: usize::MAX,
        key_value_heads: 1,
        head_dim: 2,
    };
    cases.push((
        "checked cache length overflow",
        vec![overflowing_cache.clone(), overflowing_cache],
        CpuRefErrorCode::DimensionMismatch,
    ));

    for (case, caches, expected) in cases {
        assert_rejected_before_callback(case, &[entry(4, 5.0)], caches, config(2, 2, 7), expected);
    }
}

#[test]
fn malformed_callback_candidates_fail_before_any_later_callback() {
    let cases = vec![
        ("empty", vec![], CpuRefErrorCode::DimensionMismatch),
        (
            "duplicate",
            vec![entry(3, 4.0), entry(3, 2.0)],
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "ascending score",
            vec![entry(3, 2.0), entry(4, 3.0)],
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "wrong tied-ID order",
            vec![entry(4, 3.0), entry(3, 3.0)],
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "out of range",
            vec![entry(8, 3.0)],
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "nonfinite",
            vec![entry(3, f32::NAN)],
            CpuRefErrorCode::NonFiniteInput,
        ),
    ];

    for (case, malformed_top_k, expected) in cases {
        let mut calls = 0;
        let result = greedy_generate_f32(
            &[entry(4, 5.0)],
            synthetic_caches(1, 2, 0),
            config(2, 3, 7),
            |step_index: usize, input_token: usize, caches: &[DecoderPrefillKvCache]| {
                calls += 1;
                assert_eq!((step_index, input_token, caches[0].tokens), (0, 4, 1));
                Ok(GreedyDecodeStep {
                    top_k: malformed_top_k.clone(),
                    kv_caches: synthetic_caches(2, 2, 1),
                })
            },
        );
        assert_eq!(calls, 1, "{case}");
        assert_error_code(case, result, expected);
    }
}

#[test]
fn malformed_callback_cache_sets_fail_before_any_later_callback() {
    let mut cases = vec![
        (
            "wrong count",
            synthetic_caches(2, 1, 1),
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "tokens did not advance",
            synthetic_caches(1, 2, 1),
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "tokens advanced twice",
            synthetic_caches(3, 2, 1),
            CpuRefErrorCode::DimensionMismatch,
        ),
    ];

    let mut nonuniform = synthetic_caches(2, 2, 1);
    nonuniform[1] = synthetic_cache(3, 1, 1);
    cases.push((
        "nonuniform tokens",
        nonuniform,
        CpuRefErrorCode::DimensionMismatch,
    ));

    let mut wrong_heads = synthetic_caches(2, 2, 1);
    wrong_heads[0] = shaped_cache(2, 2, 2, 0, 1);
    cases.push((
        "wrong KV heads",
        wrong_heads,
        CpuRefErrorCode::DimensionMismatch,
    ));

    let mut wrong_head_dim = synthetic_caches(2, 2, 1);
    wrong_head_dim[1] = shaped_cache(2, 1, 3, 1, 1);
    cases.push((
        "wrong head dimension",
        wrong_head_dim,
        CpuRefErrorCode::DimensionMismatch,
    ));

    let mut bad_keys = synthetic_caches(2, 2, 1);
    bad_keys[0].keys.pop();
    cases.push((
        "bad key length",
        bad_keys,
        CpuRefErrorCode::DimensionMismatch,
    ));

    let mut bad_values = synthetic_caches(2, 2, 1);
    bad_values[1].values.push(0.0);
    cases.push((
        "bad value length",
        bad_values,
        CpuRefErrorCode::DimensionMismatch,
    ));

    let mut nonfinite = synthetic_caches(2, 2, 1);
    nonfinite[1].values[2] = f32::INFINITY;
    cases.push((
        "nonfinite cache",
        nonfinite,
        CpuRefErrorCode::NonFiniteInput,
    ));

    for (case, malformed_caches, expected) in cases {
        let mut calls = 0;
        let result = greedy_generate_f32(
            &[entry(4, 5.0)],
            synthetic_caches(1, 2, 0),
            config(2, 3, 7),
            |step_index: usize, input_token: usize, caches: &[DecoderPrefillKvCache]| {
                calls += 1;
                assert_eq!((step_index, input_token, caches[0].tokens), (0, 4, 1));
                Ok(GreedyDecodeStep {
                    top_k: vec![entry(5, 4.0)],
                    kv_caches: malformed_caches.clone(),
                })
            },
        );
        assert_eq!(calls, 1, "{case}");
        assert_error_code(case, result, expected);
    }
}

#[test]
fn cache_ownership_moves_through_callbacks_and_only_latest_buffers_are_returned() {
    let prefill = vec![entry(4, 5.0), entry(3, 4.0)];
    let preserved_prefill = prefill.clone();
    let initial = synthetic_caches(1, 2, 0);
    let initial_key_pointers: Vec<_> = initial.iter().map(|cache| cache.keys.as_ptr()).collect();
    let first = synthetic_caches(2, 2, 1);
    let first_key_pointers: Vec<_> = first.iter().map(|cache| cache.keys.as_ptr()).collect();
    let second = synthetic_caches(3, 2, 2);
    let second_key_pointers: Vec<_> = second.iter().map(|cache| cache.keys.as_ptr()).collect();
    let expected_final = second.clone();
    let mut first = Some(first);
    let mut second = Some(second);

    let trace = greedy_generate_f32(
        &prefill,
        initial,
        config(2, 3, 7),
        |step_index: usize, input_token: usize, caches: &[DecoderPrefillKvCache]| match step_index {
            0 => {
                assert_eq!(input_token, 4);
                for (layer, pointer) in caches.iter().zip(&initial_key_pointers) {
                    assert_eq!(layer.keys.as_ptr(), *pointer);
                }
                Ok(GreedyDecodeStep {
                    top_k: vec![entry(5, 4.0)],
                    kv_caches: first.take().unwrap(),
                })
            }
            1 => {
                assert_eq!(input_token, 5);
                for (layer, pointer) in caches.iter().zip(&first_key_pointers) {
                    assert_eq!(layer.keys.as_ptr(), *pointer);
                }
                Ok(GreedyDecodeStep {
                    top_k: vec![entry(6, 3.0)],
                    kv_caches: second.take().unwrap(),
                })
            }
            _ => panic!("generation called past its exact maximum"),
        },
    )
    .unwrap();

    assert_eq!(prefill, preserved_prefill);
    assert_eq!(trace.generated_tokens, [4, 5, 6]);
    assert_eq!(trace.kv_caches, expected_final);
    assert_eq!(trace.kv_caches.len(), 2);
    for (layer, pointer) in trace.kv_caches.iter().zip(&second_key_pointers) {
        assert_eq!(layer.keys.as_ptr(), *pointer);
    }
}

fn run_repeatable_request() -> GreedyGenerationTrace {
    greedy_generate_f32(
        &[entry(3, 8.0), entry(1, 7.0)],
        synthetic_caches(2, 2, 0),
        config(2, 3, 7),
        |step_index: usize, input_token: usize, caches: &[DecoderPrefillKvCache]| {
            assert_eq!(input_token, [3, 5][step_index]);
            assert_eq!(caches[0].tokens, 2 + step_index);
            Ok(GreedyDecodeStep {
                top_k: vec![entry([5, 6][step_index], 6.0 - step_index as f32)],
                kv_caches: synthetic_caches(3 + step_index, 2, step_index + 1),
            })
        },
    )
    .unwrap()
}

#[test]
fn repeated_requests_are_exact_and_do_not_share_generation_state() {
    let mut first = run_repeatable_request();
    let second = run_repeatable_request();

    assert_eq!(first.generated_tokens, second.generated_tokens);
    assert_eq!(first.kv_caches, second.kv_caches);
    assert_eq!(first.decode_steps, second.decode_steps);
    assert_eq!(&first.stop_reason, &second.stop_reason);
    assert_eq!(trace_digest(&first), trace_digest(&second));
    assert_ne!(
        first.kv_caches[0].keys.as_ptr(),
        second.kv_caches[0].keys.as_ptr()
    );

    let preserved_second_bit = second.kv_caches[0].keys[0].to_bits();
    first.kv_caches[0].keys[0] = -999.0;
    assert_eq!(second.kv_caches[0].keys[0].to_bits(), preserved_second_bit);
}

#[test]
fn pinned_wrapper_freezes_shape_vocab_and_eos_contract() {
    const LAYERS: usize = 18;
    const VOCAB: usize = 103_424;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;

    let initial = shaped_caches(1, LAYERS, KV_HEADS, HEAD_DIM, 0);
    let final_caches = shaped_caches(2, LAYERS, KV_HEADS, HEAD_DIM, 1);
    let expected_final = final_caches.clone();
    let mut final_caches = Some(final_caches);
    let mut calls = 0;
    let trace = pinned_greedy_generate_f32(
        &[entry(VOCAB - 1, 9.0)],
        initial,
        2,
        |step_index: usize, input_token: usize, caches: &[DecoderPrefillKvCache]| {
            calls += 1;
            assert_eq!(
                (step_index, input_token, caches.len()),
                (0, VOCAB - 1, LAYERS)
            );
            assert!(caches.iter().all(|cache| {
                cache.tokens == 1 && cache.key_value_heads == KV_HEADS && cache.head_dim == HEAD_DIM
            }));
            Ok(GreedyDecodeStep {
                top_k: vec![entry(2, 8.0)],
                kv_caches: final_caches.take().unwrap(),
            })
        },
    )
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(trace.generated_tokens, [VOCAB - 1, 2]);
    assert_eq!(trace.kv_caches, expected_final);
    assert_eq!(trace.stop_reason, GreedyStopReason::EosToken);

    let mut malformed_sets = vec![
        (
            "short pinned layer count",
            shaped_caches(1, LAYERS - 1, KV_HEADS, HEAD_DIM, 0),
        ),
        (
            "long pinned layer count",
            shaped_caches(1, LAYERS + 1, KV_HEADS, HEAD_DIM, 0),
        ),
    ];
    let mut wrong_heads = shaped_caches(1, LAYERS, KV_HEADS, HEAD_DIM, 0);
    wrong_heads[3] = shaped_cache(1, KV_HEADS + 1, HEAD_DIM, 3, 0);
    malformed_sets.push(("shape-consistent pinned KV heads", wrong_heads));
    let mut wrong_dim = shaped_caches(1, LAYERS, KV_HEADS, HEAD_DIM, 0);
    wrong_dim[5] = shaped_cache(1, KV_HEADS, HEAD_DIM - 1, 5, 0);
    malformed_sets.push(("shape-consistent pinned head dimension", wrong_dim));

    for (case, caches) in malformed_sets {
        let mut invalid_calls = 0;
        let result = pinned_greedy_generate_f32(
            &[entry(4, 5.0)],
            caches,
            2,
            |_, _, _| -> Result<GreedyDecodeStep, CpuRefError> {
                invalid_calls += 1;
                panic!("{case}: malformed pinned request reached callback")
            },
        );
        assert_eq!(invalid_calls, 0, "{case}");
        assert_error_code(case, result, CpuRefErrorCode::DimensionMismatch);
    }

    let mut out_of_range_calls = 0;
    let out_of_range = pinned_greedy_generate_f32(
        &[entry(VOCAB, 5.0)],
        shaped_caches(1, LAYERS, KV_HEADS, HEAD_DIM, 0),
        2,
        |_, _, _| -> Result<GreedyDecodeStep, CpuRefError> {
            out_of_range_calls += 1;
            panic!("out-of-range pinned token reached callback")
        },
    );
    assert_eq!(out_of_range_calls, 0);
    assert_error_code(
        "pinned vocabulary upper bound",
        out_of_range,
        CpuRefErrorCode::DimensionMismatch,
    );

    let pinned_eos = pinned_greedy_generate_f32(
        &[entry(2, 5.0)],
        shaped_caches(1, LAYERS, KV_HEADS, HEAD_DIM, 0),
        4,
        |_, _, _| -> Result<GreedyDecodeStep, CpuRefError> {
            panic!("pinned prefill EOS reached callback")
        },
    )
    .unwrap();
    assert_eq!(pinned_eos.generated_tokens, [2]);
    assert_eq!(pinned_eos.decode_steps, 0);
    assert_eq!(pinned_eos.stop_reason, GreedyStopReason::EosToken);
}

#[test]
fn unbounded_maximum_with_prefill_eos_does_not_reserve_unbounded_storage() {
    let initial = synthetic_caches(1, 1, 0);
    let expected = initial.clone();
    let trace = greedy_generate_f32(
        &[entry(2, 5.0)],
        initial,
        full_config(1, 8, 1, 2, usize::MAX, 2),
        |_, _, _| -> Result<GreedyDecodeStep, CpuRefError> {
            panic!("immediate EOS reached callback")
        },
    )
    .unwrap();

    assert_eq!(trace.generated_tokens, [2]);
    assert_eq!(trace.kv_caches, expected);
    assert_eq!(trace.decode_steps, 0);
    assert_eq!(trace.stop_reason, GreedyStopReason::EosToken);
}
