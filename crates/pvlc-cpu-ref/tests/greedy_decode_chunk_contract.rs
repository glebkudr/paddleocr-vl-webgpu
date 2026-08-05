//! M6d tests-first contract for chunking already-defined greedy decode steps.
//! Random nondeterminism, decoder arithmetic, WebGPU, and performance claims
//! are intentionally outside this compact state-machine contract.

use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, DecoderPrefillKvCache, GreedyChunkConfig,
    GreedyChunkedGenerationTrace, GreedyDecodeChunk, GreedyDecodeStep, GreedyGenerationConfig,
    GreedyGenerationTrace, GreedyStopReason, TopKEntry, greedy_generate_chunked_f32,
    greedy_generate_f32, pinned_greedy_generate_chunked_f32, top_k,
};

const LAYERS: usize = 2;
const VOCAB: usize = 64;
const INITIAL_TOKENS: usize = 2;
const PREFILL_TOKEN: usize = 4;
const EOS_TOKEN: usize = 63;
const NEXT_TOKENS: [usize; 8] = [5, 6, 7, 8, 9, 10, 11, 12];

fn entry(index: usize, value: f32) -> TopKEntry {
    TopKEntry { index, value }
}

fn generation_config(max_new_tokens: usize) -> GreedyGenerationConfig {
    full_generation_config(LAYERS, VOCAB, 1, 2, max_new_tokens, EOS_TOKEN)
}

fn full_generation_config(
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

fn chunk_config(max_new_tokens: usize, decode_chunk_size: usize) -> GreedyChunkConfig {
    GreedyChunkConfig {
        generation: generation_config(max_new_tokens),
        decode_chunk_size,
    }
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

fn caches(tokens: usize, generation: usize) -> Vec<DecoderPrefillKvCache> {
    shaped_caches(tokens, LAYERS, 1, 2, generation)
}

fn cache_buffer_pointers(caches: &[DecoderPrefillKvCache]) -> Vec<(*const f32, *const f32)> {
    caches
        .iter()
        .map(|cache| (cache.keys.as_ptr(), cache.values.as_ptr()))
        .collect()
}

fn decode_step(next_token: usize, decoded_index: usize) -> GreedyDecodeStep {
    GreedyDecodeStep {
        top_k: vec![entry(next_token, 100.0 - decoded_index as f32)],
        kv_caches: caches(INITIAL_TOKENS + decoded_index + 1, decoded_index + 1),
    }
}

fn assert_generation_literal(trace: &GreedyGenerationTrace) {
    assert_eq!(
        trace.generated_tokens,
        [PREFILL_TOKEN, 5, 6, 7, 8, 9, 10, 11, 12]
    );
    assert_eq!(trace.kv_caches, caches(INITIAL_TOKENS + 8, 8));
    assert_eq!(trace.decode_steps, 8);
    assert_eq!(trace.stop_reason, GreedyStopReason::MaxNewTokens);
}

fn run_single_step_baseline() -> GreedyGenerationTrace {
    greedy_generate_f32(
        &[entry(PREFILL_TOKEN, 101.0)],
        caches(INITIAL_TOKENS, 0),
        generation_config(9),
        |step_index: usize, input_token: usize, current: &[DecoderPrefillKvCache]| {
            assert_eq!(
                input_token,
                if step_index == 0 {
                    PREFILL_TOKEN
                } else {
                    NEXT_TOKENS[step_index - 1]
                }
            );
            assert_eq!(current, caches(INITIAL_TOKENS + step_index, step_index));
            Ok(decode_step(NEXT_TOKENS[step_index], step_index))
        },
    )
    .unwrap()
}

fn run_exact_chunks(decode_chunk_size: usize) -> GreedyChunkedGenerationTrace {
    let mut decoded = 0;
    let mut chunks = 0;
    let trace = greedy_generate_chunked_f32(
        &[entry(PREFILL_TOKEN, 101.0)],
        caches(INITIAL_TOKENS, 0),
        chunk_config(9, decode_chunk_size),
        |chunk_index: usize,
         input_token: usize,
         current: &[DecoderPrefillKvCache],
         requested_steps: usize| {
            assert_eq!(chunk_index, chunks);
            assert_eq!(
                input_token,
                if decoded == 0 {
                    PREFILL_TOKEN
                } else {
                    NEXT_TOKENS[decoded - 1]
                }
            );
            assert_eq!(current, caches(INITIAL_TOKENS + decoded, decoded));
            assert_eq!(
                requested_steps,
                decode_chunk_size.min(NEXT_TOKENS.len() - decoded)
            );
            let steps = (0..requested_steps)
                .map(|offset| decode_step(NEXT_TOKENS[decoded + offset], decoded + offset))
                .collect();
            decoded += requested_steps;
            chunks += 1;
            Ok(GreedyDecodeChunk { steps })
        },
    )
    .unwrap();
    assert_eq!(decoded, NEXT_TOKENS.len());
    assert_eq!(trace.decode_chunks, chunks);
    trace
}

fn update_usize(hasher: &mut blake3::Hasher, value: usize) {
    hasher.update(&(value as u64).to_le_bytes());
}

fn update_f32(hasher: &mut blake3::Hasher, values: &[f32]) {
    update_usize(hasher, values.len());
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn trace_digest(trace: &GreedyChunkedGenerationTrace) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"pvlc-greedy-decode-chunk-v1\0");
    update_usize(&mut hasher, trace.generation.generated_tokens.len());
    for token in &trace.generation.generated_tokens {
        update_usize(&mut hasher, *token);
    }
    update_usize(&mut hasher, trace.generation.kv_caches.len());
    for cache in &trace.generation.kv_caches {
        update_usize(&mut hasher, cache.tokens);
        update_usize(&mut hasher, cache.key_value_heads);
        update_usize(&mut hasher, cache.head_dim);
        update_f32(&mut hasher, &cache.keys);
        update_f32(&mut hasher, &cache.values);
    }
    hasher.update(&[match trace.generation.stop_reason {
        GreedyStopReason::MaxNewTokens => 0,
        GreedyStopReason::EosToken => 1,
    }]);
    update_usize(&mut hasher, trace.generation.decode_steps);
    update_usize(&mut hasher, trace.decode_chunks);
    hasher.finalize().to_hex().to_string()
}

fn assert_error<T>(case: &str, result: Result<T, CpuRefError>, expected: CpuRefErrorCode) {
    match result {
        Err(error) => assert_eq!(error.code(), expected, "{case}"),
        Ok(_) => panic!("{case}: expected {expected:?}"),
    }
}

fn assert_rejected_before_callback(
    case: &str,
    prefill_top_k: &[TopKEntry],
    initial_caches: Vec<DecoderPrefillKvCache>,
    config: GreedyChunkConfig,
    expected: CpuRefErrorCode,
) {
    let mut calls = 0;
    let result = greedy_generate_chunked_f32(
        prefill_top_k,
        initial_caches,
        config,
        |_, _, _, _| -> Result<GreedyDecodeChunk, CpuRefError> {
            calls += 1;
            panic!("{case}: malformed request reached callback")
        },
    );
    assert_eq!(calls, 0, "{case}");
    assert_error(case, result, expected);
}

#[test]
fn literal_nine_token_three_chunk_trace_matches_fixed_digest() {
    let trace = run_exact_chunks(3);
    assert_generation_literal(&trace.generation);
    assert_eq!(trace.decode_chunks, 3);
    assert_eq!(
        trace_digest(&trace),
        "2bd4512d55393ab9f9115b0827edf7821a54d6ea6be963d0be5caabe88e60259"
    );
}

#[test]
fn chunk_sizes_one_two_four_and_eight_equal_single_step_greedy_and_literal_oracle() {
    let baseline = run_single_step_baseline();
    assert_generation_literal(&baseline);
    for (chunk_size, expected_chunks) in [(1, 8), (2, 4), (4, 2), (8, 1)] {
        let chunked = run_exact_chunks(chunk_size);
        assert_generation_literal(&chunked.generation);
        assert_eq!(chunked.generation, baseline, "chunk size {chunk_size}");
        assert_eq!(chunked.decode_chunks, expected_chunks);
    }
}

#[test]
fn short_chunks_follow_exact_two_one_four_one_split_protocol_and_literal_trace() {
    const SPLITS: [usize; 4] = [2, 1, 4, 1];
    const REQUESTED: [usize; 4] = [4, 4, 4, 1];
    let mut decoded = 0;
    let mut calls = Vec::new();
    let trace = greedy_generate_chunked_f32(
        &[entry(PREFILL_TOKEN, 101.0)],
        caches(INITIAL_TOKENS, 0),
        chunk_config(9, 4),
        |chunk_index: usize,
         input_token: usize,
         current: &[DecoderPrefillKvCache],
         requested_steps: usize| {
            let returned = SPLITS[chunk_index];
            assert_eq!(requested_steps, REQUESTED[chunk_index]);
            assert!(returned <= requested_steps);
            assert_eq!(
                input_token,
                if decoded == 0 {
                    PREFILL_TOKEN
                } else {
                    NEXT_TOKENS[decoded - 1]
                }
            );
            assert_eq!(current, caches(INITIAL_TOKENS + decoded, decoded));
            calls.push((chunk_index, input_token, requested_steps, returned));
            let steps = (0..returned)
                .map(|offset| decode_step(NEXT_TOKENS[decoded + offset], decoded + offset))
                .collect();
            decoded += returned;
            Ok(GreedyDecodeChunk { steps })
        },
    )
    .unwrap();

    assert_eq!(
        calls,
        [(0, 4, 4, 2), (1, 6, 4, 1), (2, 7, 4, 4), (3, 11, 1, 1)]
    );
    assert_eq!(decoded, NEXT_TOKENS.len());
    assert_generation_literal(&trace.generation);
    assert_eq!(trace.decode_chunks, 4);
}

#[test]
fn eos_as_final_returned_step_at_first_middle_or_last_position_stops_one_chunk_exactly() {
    for eos_position in [1_usize, 2, 4] {
        let mut calls = 0;
        let mut emitted_steps = 0;
        let trace = greedy_generate_chunked_f32(
            &[entry(PREFILL_TOKEN, 101.0)],
            caches(INITIAL_TOKENS, 0),
            chunk_config(9, 4),
            |chunk_index: usize,
             input_token: usize,
             current: &[DecoderPrefillKvCache],
             requested_steps: usize| {
                calls += 1;
                assert_eq!(chunk_index, 0);
                assert_eq!(input_token, PREFILL_TOKEN);
                assert_eq!(current, caches(INITIAL_TOKENS, 0));
                assert_eq!(requested_steps, 4);
                let steps = (0..eos_position)
                    .map(|index| {
                        let token = if index + 1 == eos_position {
                            EOS_TOKEN
                        } else {
                            NEXT_TOKENS[index]
                        };
                        decode_step(token, index)
                    })
                    .collect();
                emitted_steps += eos_position;
                Ok(GreedyDecodeChunk { steps })
            },
        )
        .unwrap();

        let mut expected_tokens = vec![PREFILL_TOKEN];
        expected_tokens.extend_from_slice(&NEXT_TOKENS[..eos_position - 1]);
        expected_tokens.push(EOS_TOKEN);
        assert_eq!(calls, 1, "EOS position {eos_position}");
        assert_eq!(emitted_steps, eos_position, "EOS position {eos_position}");
        assert_eq!(trace.generation.generated_tokens, expected_tokens);
        assert_eq!(
            trace.generation.kv_caches,
            caches(INITIAL_TOKENS + eos_position, eos_position)
        );
        assert_eq!(trace.generation.decode_steps, eos_position);
        assert_eq!(trace.generation.stop_reason, GreedyStopReason::EosToken);
        assert_eq!(trace.decode_chunks, 1);
    }
}

#[test]
fn returned_step_after_eos_is_rejected_after_one_callback() {
    let mut calls = 0;
    let result = greedy_generate_chunked_f32(
        &[entry(PREFILL_TOKEN, 101.0)],
        caches(INITIAL_TOKENS, 0),
        chunk_config(9, 4),
        |chunk_index: usize,
         input_token: usize,
         current: &[DecoderPrefillKvCache],
         requested_steps: usize| {
            calls += 1;
            assert_eq!(
                (chunk_index, input_token, requested_steps),
                (0, PREFILL_TOKEN, 4)
            );
            assert_eq!(current, caches(INITIAL_TOKENS, 0));
            Ok(GreedyDecodeChunk {
                steps: vec![decode_step(EOS_TOKEN, 0), decode_step(5, 1)],
            })
        },
    );
    assert_eq!(calls, 1);
    assert_error("step after EOS", result, CpuRefErrorCode::DimensionMismatch);
}

#[test]
fn returned_steps_beyond_requested_and_exact_max_are_rejected_before_later_callback() {
    let mut calls = 0;
    let result = greedy_generate_chunked_f32(
        &[entry(PREFILL_TOKEN, 101.0)],
        caches(INITIAL_TOKENS, 0),
        chunk_config(3, 8),
        |chunk_index: usize,
         input_token: usize,
         current: &[DecoderPrefillKvCache],
         requested_steps: usize| {
            calls += 1;
            assert_eq!(
                (chunk_index, input_token, requested_steps),
                (0, PREFILL_TOKEN, 2)
            );
            assert_eq!(current, caches(INITIAL_TOKENS, 0));
            Ok(GreedyDecodeChunk {
                steps: vec![decode_step(5, 0), decode_step(6, 1), decode_step(7, 2)],
            })
        },
    );
    assert_eq!(calls, 1);
    assert_error(
        "steps beyond requested and max",
        result,
        CpuRefErrorCode::DimensionMismatch,
    );
}

#[test]
fn empty_returned_chunk_is_rejected_before_later_callback() {
    let mut calls = 0;
    let result = greedy_generate_chunked_f32(
        &[entry(PREFILL_TOKEN, 101.0)],
        caches(INITIAL_TOKENS, 0),
        chunk_config(9, 4),
        |chunk_index: usize,
         input_token: usize,
         current: &[DecoderPrefillKvCache],
         requested_steps: usize| {
            calls += 1;
            assert_eq!(
                (chunk_index, input_token, requested_steps),
                (0, PREFILL_TOKEN, 4)
            );
            assert_eq!(current, caches(INITIAL_TOKENS, 0));
            Ok(GreedyDecodeChunk { steps: vec![] })
        },
    );
    assert_eq!(calls, 1);
    assert_error("empty chunk", result, CpuRefErrorCode::DimensionMismatch);
}

#[test]
fn zero_chunk_and_malformed_base_inputs_are_rejected_before_callback() {
    assert_rejected_before_callback(
        "zero decode chunk",
        &[entry(PREFILL_TOKEN, 101.0)],
        caches(INITIAL_TOKENS, 0),
        chunk_config(9, 0),
        CpuRefErrorCode::DimensionMismatch,
    );

    assert_rejected_before_callback(
        "zero base maximum",
        &[entry(PREFILL_TOKEN, 101.0)],
        caches(INITIAL_TOKENS, 0),
        GreedyChunkConfig {
            generation: generation_config(0),
            decode_chunk_size: 4,
        },
        CpuRefErrorCode::DimensionMismatch,
    );

    assert_rejected_before_callback(
        "duplicate prefill token",
        &[entry(PREFILL_TOKEN, 101.0), entry(PREFILL_TOKEN, 100.0)],
        caches(INITIAL_TOKENS, 0),
        chunk_config(9, 4),
        CpuRefErrorCode::DimensionMismatch,
    );

    assert_rejected_before_callback(
        "wrong prefill tie order",
        &[entry(7, 101.0), entry(6, 101.0)],
        caches(INITIAL_TOKENS, 0),
        chunk_config(9, 4),
        CpuRefErrorCode::DimensionMismatch,
    );

    assert_rejected_before_callback(
        "nonfinite prefill score",
        &[entry(PREFILL_TOKEN, f32::NAN)],
        caches(INITIAL_TOKENS, 0),
        chunk_config(9, 4),
        CpuRefErrorCode::NonFiniteInput,
    );

    assert_rejected_before_callback(
        "wrong initial cache count",
        &[entry(PREFILL_TOKEN, 101.0)],
        shaped_caches(INITIAL_TOKENS, LAYERS - 1, 1, 2, 0),
        chunk_config(9, 4),
        CpuRefErrorCode::DimensionMismatch,
    );

    let mut nonuniform_initial = caches(INITIAL_TOKENS, 0);
    nonuniform_initial[1] = shaped_cache(INITIAL_TOKENS + 1, 1, 2, 1, 0);
    assert_rejected_before_callback(
        "later initial layer has nonuniform tokens",
        &[entry(PREFILL_TOKEN, 101.0)],
        nonuniform_initial,
        chunk_config(9, 4),
        CpuRefErrorCode::DimensionMismatch,
    );

    let mut mismatched_initial_metadata = caches(INITIAL_TOKENS, 0);
    mismatched_initial_metadata[1] = shaped_cache(INITIAL_TOKENS, 2, 2, 1, 0);
    assert_rejected_before_callback(
        "later initial layer has mismatched KV metadata",
        &[entry(PREFILL_TOKEN, 101.0)],
        mismatched_initial_metadata,
        chunk_config(9, 4),
        CpuRefErrorCode::DimensionMismatch,
    );
}

#[test]
fn invalid_second_returned_step_candidates_and_caches_fail_in_first_chunk() {
    let valid_second = decode_step(6, 1);
    let mut cases = Vec::new();

    let mut empty_candidate = valid_second.clone();
    empty_candidate.top_k.clear();
    cases.push((
        "empty second candidates",
        empty_candidate,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut duplicate_candidate = valid_second.clone();
    duplicate_candidate.top_k = vec![entry(6, 90.0), entry(6, 80.0)];
    cases.push((
        "duplicate second candidate",
        duplicate_candidate,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut unsorted_candidate = valid_second.clone();
    unsorted_candidate.top_k = vec![entry(6, 80.0), entry(7, 90.0)];
    cases.push((
        "unsorted second candidates",
        unsorted_candidate,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut wrong_tie_order = valid_second.clone();
    wrong_tie_order.top_k = vec![entry(7, 90.0), entry(6, 90.0)];
    cases.push((
        "wrong second candidate tie order",
        wrong_tie_order,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut out_of_range_candidate = valid_second.clone();
    out_of_range_candidate.top_k = vec![entry(VOCAB, 90.0)];
    cases.push((
        "out-of-range second candidate",
        out_of_range_candidate,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut nonfinite_candidate = valid_second.clone();
    nonfinite_candidate.top_k = vec![entry(6, f32::NAN)];
    cases.push((
        "nonfinite second candidate",
        nonfinite_candidate,
        CpuRefErrorCode::NonFiniteInput,
    ));

    let mut wrong_count = valid_second.clone();
    wrong_count.kv_caches.pop();
    cases.push((
        "wrong second cache count",
        wrong_count,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut wrong_metadata = valid_second.clone();
    wrong_metadata.kv_caches[0] = shaped_cache(4, 2, 2, 0, 2);
    cases.push((
        "wrong second cache metadata",
        wrong_metadata,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut wrong_progression = valid_second.clone();
    wrong_progression.kv_caches = caches(3, 2);
    cases.push((
        "wrong second cache token progression",
        wrong_progression,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut later_nonuniform = valid_second.clone();
    later_nonuniform.kv_caches[1] = shaped_cache(5, 1, 2, 1, 2);
    cases.push((
        "later second cache has nonuniform tokens",
        later_nonuniform,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut later_metadata = valid_second.clone();
    later_metadata.kv_caches[1] = shaped_cache(4, 1, 3, 1, 2);
    cases.push((
        "later second cache has mismatched head metadata",
        later_metadata,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut wrong_length = valid_second.clone();
    wrong_length.kv_caches[1].keys.pop();
    cases.push((
        "wrong second cache length",
        wrong_length,
        CpuRefErrorCode::DimensionMismatch,
    ));
    let mut nonfinite_cache = valid_second;
    nonfinite_cache.kv_caches[0].values[1] = f32::INFINITY;
    cases.push((
        "nonfinite second cache",
        nonfinite_cache,
        CpuRefErrorCode::NonFiniteInput,
    ));

    for (case, invalid_second, expected) in cases {
        let mut calls = 0;
        let result = greedy_generate_chunked_f32(
            &[entry(PREFILL_TOKEN, 101.0)],
            caches(INITIAL_TOKENS, 0),
            chunk_config(9, 4),
            |chunk_index: usize,
             input_token: usize,
             current: &[DecoderPrefillKvCache],
             requested_steps: usize| {
                calls += 1;
                assert_eq!(
                    (chunk_index, input_token, requested_steps),
                    (0, PREFILL_TOKEN, 4)
                );
                assert_eq!(current, caches(INITIAL_TOKENS, 0));
                Ok(GreedyDecodeChunk {
                    steps: vec![decode_step(5, 0), invalid_second.clone()],
                })
            },
        );
        assert_eq!(calls, 1, "{case}");
        assert_error(case, result, expected);
    }
}

#[test]
fn callback_error_cancels_request_without_leaking_into_fresh_exact_request() {
    let baseline = run_exact_chunks(3);
    let expected_error = top_k(&[], 1).unwrap_err();
    let returned_error = expected_error.clone();
    let mut calls = 0;
    let canceled = greedy_generate_chunked_f32(
        &[entry(PREFILL_TOKEN, 101.0)],
        caches(INITIAL_TOKENS, 0),
        chunk_config(9, 3),
        |chunk_index: usize,
         input_token: usize,
         current: &[DecoderPrefillKvCache],
         requested_steps: usize|
         -> Result<GreedyDecodeChunk, CpuRefError> {
            calls += 1;
            assert_eq!(
                (chunk_index, input_token, requested_steps),
                (0, PREFILL_TOKEN, 3)
            );
            assert_eq!(current, caches(INITIAL_TOKENS, 0));
            Err(returned_error.clone())
        },
    );
    assert_eq!(calls, 1);
    assert_eq!(canceled.unwrap_err(), expected_error);

    let fresh = run_exact_chunks(3);
    assert_eq!(fresh, baseline);
    assert_eq!(trace_digest(&fresh), trace_digest(&baseline));
    for (fresh_cache, baseline_cache) in fresh
        .generation
        .kv_caches
        .iter()
        .zip(&baseline.generation.kv_caches)
    {
        assert_ne!(fresh_cache.keys.as_ptr(), baseline_cache.keys.as_ptr());
        assert_ne!(fresh_cache.values.as_ptr(), baseline_cache.values.as_ptr());
    }
}

#[test]
fn cache_prefix_boundaries_end_at_exact_start_plus_eight_for_all_chunk_sizes() {
    const STARTS: [usize; 14] = [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129];
    for start_tokens in STARTS {
        for chunk_size in [1_usize, 2, 4, 8] {
            let mut decoded = 0;
            let trace = greedy_generate_chunked_f32(
                &[entry(PREFILL_TOKEN, 101.0)],
                shaped_caches(start_tokens, 1, 1, 1, 0),
                GreedyChunkConfig {
                    generation: full_generation_config(1, VOCAB, 1, 1, 9, EOS_TOKEN),
                    decode_chunk_size: chunk_size,
                },
                |chunk_index: usize,
                 input_token: usize,
                 current: &[DecoderPrefillKvCache],
                 requested_steps: usize| {
                    assert_eq!(chunk_index, decoded / chunk_size);
                    assert_eq!(
                        input_token,
                        if decoded == 0 {
                            PREFILL_TOKEN
                        } else {
                            NEXT_TOKENS[decoded - 1]
                        }
                    );
                    assert_eq!(current.len(), 1);
                    assert_eq!(current[0].tokens, start_tokens + decoded);
                    assert_eq!(current[0].keys.len(), start_tokens + decoded);
                    assert_eq!(requested_steps, chunk_size.min(NEXT_TOKENS.len() - decoded));
                    let steps = (0..requested_steps)
                        .map(|offset| {
                            let index = decoded + offset;
                            GreedyDecodeStep {
                                top_k: vec![entry(NEXT_TOKENS[index], 100.0 - index as f32)],
                                kv_caches: shaped_caches(
                                    start_tokens + index + 1,
                                    1,
                                    1,
                                    1,
                                    index + 1,
                                ),
                            }
                        })
                        .collect();
                    decoded += requested_steps;
                    Ok(GreedyDecodeChunk { steps })
                },
            )
            .unwrap();

            assert_eq!(
                trace.generation.generated_tokens,
                [PREFILL_TOKEN, 5, 6, 7, 8, 9, 10, 11, 12]
            );
            assert_eq!(trace.generation.kv_caches.len(), 1);
            assert_eq!(trace.generation.kv_caches[0].tokens, start_tokens + 8);
            assert_eq!(trace.generation.kv_caches[0].keys.len(), start_tokens + 8);
            assert_eq!(trace.generation.kv_caches[0].values.len(), start_tokens + 8);
            assert_eq!(trace.generation.decode_steps, 8);
            assert_eq!(trace.generation.stop_reason, GreedyStopReason::MaxNewTokens);
            assert_eq!(trace.decode_chunks, 8_usize.div_ceil(chunk_size));
        }
    }
}

#[test]
fn only_last_step_buffers_move_to_next_callback_and_final_trace() {
    let prefill = vec![entry(PREFILL_TOKEN, 101.0)];
    let preserved_prefill = prefill.clone();
    let initial = caches(INITIAL_TOKENS, 0);
    let initial_pointers = cache_buffer_pointers(&initial);
    let first = caches(INITIAL_TOKENS + 1, 1);
    let second = caches(INITIAL_TOKENS + 2, 2);
    let third = caches(INITIAL_TOKENS + 3, 3);
    let fourth = caches(INITIAL_TOKENS + 4, 4);
    let first_pointers = cache_buffer_pointers(&first);
    let second_pointers = cache_buffer_pointers(&second);
    let third_pointers = cache_buffer_pointers(&third);
    let fourth_pointers = cache_buffer_pointers(&fourth);
    assert_ne!(first_pointers[0], second_pointers[0]);
    assert_ne!(second_pointers[0], third_pointers[0]);
    assert_ne!(third_pointers[0], fourth_pointers[0]);
    let expected_final = fourth.clone();
    let mut first = Some(first);
    let mut second = Some(second);
    let mut third = Some(third);
    let mut fourth = Some(fourth);
    let mut calls = 0;

    let trace = greedy_generate_chunked_f32(
        &prefill,
        initial,
        chunk_config(5, 3),
        |chunk_index: usize,
         input_token: usize,
         current: &[DecoderPrefillKvCache],
         requested_steps: usize| {
            calls += 1;
            match chunk_index {
                0 => {
                    assert_eq!((input_token, requested_steps), (PREFILL_TOKEN, 3));
                    assert_eq!(cache_buffer_pointers(current), initial_pointers);
                    Ok(GreedyDecodeChunk {
                        steps: vec![
                            GreedyDecodeStep {
                                top_k: vec![entry(5, 90.0)],
                                kv_caches: first.take().unwrap(),
                            },
                            GreedyDecodeStep {
                                top_k: vec![entry(6, 80.0)],
                                kv_caches: second.take().unwrap(),
                            },
                            GreedyDecodeStep {
                                top_k: vec![entry(7, 70.0)],
                                kv_caches: third.take().unwrap(),
                            },
                        ],
                    })
                }
                1 => {
                    assert_eq!((input_token, requested_steps), (7, 1));
                    assert_eq!(cache_buffer_pointers(current), third_pointers);
                    assert_ne!(cache_buffer_pointers(current), first_pointers);
                    assert_ne!(cache_buffer_pointers(current), second_pointers);
                    Ok(GreedyDecodeChunk {
                        steps: vec![GreedyDecodeStep {
                            top_k: vec![entry(8, 60.0)],
                            kv_caches: fourth.take().unwrap(),
                        }],
                    })
                }
                _ => panic!("generation called after exact maximum"),
            }
        },
    )
    .unwrap();

    assert_eq!(calls, 2);
    assert_eq!(prefill, preserved_prefill);
    assert_eq!(trace.generation.generated_tokens, [4, 5, 6, 7, 8]);
    assert_eq!(trace.generation.kv_caches, expected_final);
    assert_eq!(
        cache_buffer_pointers(&trace.generation.kv_caches),
        fourth_pointers
    );
    assert_eq!(trace.generation.decode_steps, 4);
    assert_eq!(trace.decode_chunks, 2);
}

#[test]
fn pinned_wrapper_accepts_exact_topology_and_rejects_wrong_shapes_before_callback() {
    const PINNED_LAYERS: usize = 18;
    const PINNED_VOCAB: usize = 103_424;
    const PINNED_KV: usize = 2;
    const PINNED_DIM: usize = 128;
    let initial = shaped_caches(1, PINNED_LAYERS, PINNED_KV, PINNED_DIM, 0);
    let expected_final = shaped_caches(5, PINNED_LAYERS, PINNED_KV, PINNED_DIM, 4);
    let mut calls = 0;
    let trace = pinned_greedy_generate_chunked_f32(
        &[entry(PINNED_VOCAB - 1, 101.0)],
        initial,
        5,
        4,
        |chunk_index: usize,
         input_token: usize,
         current: &[DecoderPrefillKvCache],
         requested_steps: usize| {
            calls += 1;
            assert_eq!(
                (chunk_index, input_token, requested_steps),
                (0, PINNED_VOCAB - 1, 4)
            );
            assert_eq!(current.len(), PINNED_LAYERS);
            assert!(current.iter().all(|cache| {
                cache.tokens == 1
                    && cache.key_value_heads == PINNED_KV
                    && cache.head_dim == PINNED_DIM
            }));
            Ok(GreedyDecodeChunk {
                steps: [5, 6, 7, 2]
                    .into_iter()
                    .enumerate()
                    .map(|(index, token)| GreedyDecodeStep {
                        top_k: vec![entry(token, 90.0 - index as f32)],
                        kv_caches: shaped_caches(
                            index + 2,
                            PINNED_LAYERS,
                            PINNED_KV,
                            PINNED_DIM,
                            index + 1,
                        ),
                    })
                    .collect(),
            })
        },
    )
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(
        trace.generation.generated_tokens,
        [PINNED_VOCAB - 1, 5, 6, 7, 2]
    );
    assert_eq!(trace.generation.kv_caches, expected_final);
    assert_eq!(trace.generation.decode_steps, 4);
    assert_eq!(trace.generation.stop_reason, GreedyStopReason::EosToken);
    assert_eq!(trace.decode_chunks, 1);

    let mut malformed = vec![
        (
            "short pinned layer count",
            vec![entry(4, 10.0)],
            shaped_caches(1, PINNED_LAYERS - 1, PINNED_KV, PINNED_DIM, 0),
        ),
        (
            "long pinned layer count",
            vec![entry(4, 10.0)],
            shaped_caches(1, PINNED_LAYERS + 1, PINNED_KV, PINNED_DIM, 0),
        ),
        (
            "pinned vocabulary upper bound",
            vec![entry(PINNED_VOCAB, 10.0)],
            shaped_caches(1, PINNED_LAYERS, PINNED_KV, PINNED_DIM, 0),
        ),
    ];
    let mut wrong_kv = shaped_caches(1, PINNED_LAYERS, PINNED_KV, PINNED_DIM, 0);
    wrong_kv[3] = shaped_cache(1, PINNED_KV + 1, PINNED_DIM, 3, 0);
    malformed.push(("wrong pinned KV metadata", vec![entry(4, 10.0)], wrong_kv));
    let mut wrong_dim = shaped_caches(1, PINNED_LAYERS, PINNED_KV, PINNED_DIM, 0);
    wrong_dim[5] = shaped_cache(1, PINNED_KV, PINNED_DIM - 1, 5, 0);
    malformed.push((
        "wrong pinned head dimension",
        vec![entry(4, 10.0)],
        wrong_dim,
    ));

    for (case, prefill, caches) in malformed {
        let mut invalid_calls = 0;
        let result = pinned_greedy_generate_chunked_f32(
            &prefill,
            caches,
            2,
            4,
            |_, _, _, _| -> Result<GreedyDecodeChunk, CpuRefError> {
                invalid_calls += 1;
                panic!("{case}: malformed pinned request reached callback")
            },
        );
        assert_eq!(invalid_calls, 0, "{case}");
        assert_error(case, result, CpuRefErrorCode::DimensionMismatch);
    }
}

#[test]
fn unbounded_max_and_chunk_remain_bounded_for_decode_eos_and_prefill_eos() {
    let generation = full_generation_config(1, 8, 1, 1, usize::MAX, 2);
    let initial = shaped_caches(1, 1, 1, 1, 0);
    let final_caches = shaped_caches(2, 1, 1, 1, 1);
    let expected_final = final_caches.clone();
    let mut final_caches = Some(final_caches);
    let mut calls = 0;
    let trace = greedy_generate_chunked_f32(
        &[entry(4, 10.0)],
        initial,
        GreedyChunkConfig {
            generation,
            decode_chunk_size: usize::MAX,
        },
        |chunk_index: usize,
         input_token: usize,
         current: &[DecoderPrefillKvCache],
         requested_steps: usize| {
            calls += 1;
            assert_eq!((chunk_index, input_token), (0, 4));
            assert_eq!(requested_steps, usize::MAX - 1);
            assert_eq!(current[0].tokens, 1);
            Ok(GreedyDecodeChunk {
                steps: vec![GreedyDecodeStep {
                    top_k: vec![entry(2, 9.0)],
                    kv_caches: final_caches.take().unwrap(),
                }],
            })
        },
    )
    .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(trace.generation.generated_tokens, [4, 2]);
    assert_eq!(trace.generation.kv_caches, expected_final);
    assert_eq!(trace.generation.decode_steps, 1);
    assert_eq!(trace.generation.stop_reason, GreedyStopReason::EosToken);
    assert_eq!(trace.decode_chunks, 1);

    let eos_initial = shaped_caches(1, 1, 1, 1, 0);
    let expected_eos_initial = eos_initial.clone();
    let eos_trace = greedy_generate_chunked_f32(
        &[entry(2, 10.0)],
        eos_initial,
        GreedyChunkConfig {
            generation,
            decode_chunk_size: usize::MAX,
        },
        |_, _, _, _| -> Result<GreedyDecodeChunk, CpuRefError> {
            panic!("prefill EOS must not invoke a chunk callback")
        },
    )
    .unwrap();
    assert_eq!(eos_trace.generation.generated_tokens, [2]);
    assert_eq!(eos_trace.generation.kv_caches, expected_eos_initial);
    assert_eq!(eos_trace.generation.decode_steps, 0);
    assert_eq!(eos_trace.generation.stop_reason, GreedyStopReason::EosToken);
    assert_eq!(eos_trace.decode_chunks, 0);
}
