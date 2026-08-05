use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, DecoderPrefillKvCache, causal_gqa_f32,
    pinned_decoder_causal_gqa_f32, write_pinned_decoder_prefill_kv_f32,
};

const TOKENS: usize = 5;
const QUERY_HEADS: usize = 4;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIM: usize = 3;
const COMPACT_BLAKE3: &str = "edf2b814cef41ad64d1635cc496f5855462043bcf1923aaafab47a39ef9eda3e";

type Inputs = [Vec<f32>; 3];

fn fixture(tokens: usize, query_heads: usize, key_value_heads: usize, head_dim: usize) -> Inputs {
    [
        (0..tokens * query_heads * head_dim)
            .map(|index| (((index * 17 + 3) % 37) as f32 - 18.0) / 11.0)
            .collect(),
        (0..tokens * key_value_heads * head_dim)
            .map(|index| (((index * 13 + 7) % 31) as f32 - 15.0) / 9.0)
            .collect(),
        (0..tokens * key_value_heads * head_dim)
            .map(|index| (((index * 19 + 5) % 41) as f32 - 20.0) / 7.0)
            .collect(),
    ]
}

fn index(token: usize, head: usize, dim: usize, heads: usize, head_dim: usize) -> usize {
    (token * heads + head) * head_dim + dim
}

fn physically_repeat_kv_heads(
    input: &[f32],
    tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let group = query_heads / key_value_heads;
    let mut repeated = Vec::with_capacity(tokens * query_heads * head_dim);
    for token in 0..tokens {
        for kv_head in 0..key_value_heads {
            let start = index(token, kv_head, 0, key_value_heads, head_dim);
            for _ in 0..group {
                repeated.extend_from_slice(&input[start..start + head_dim]);
            }
        }
    }
    repeated
}

fn independent_causal_gqa(
    inputs: &Inputs,
    tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let repeated_key =
        physically_repeat_kv_heads(&inputs[1], tokens, query_heads, key_value_heads, head_dim);
    let repeated_value =
        physically_repeat_kv_heads(&inputs[2], tokens, query_heads, key_value_heads, head_dim);
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0; inputs[0].len()];
    for query_token in 0..tokens {
        for head in 0..query_heads {
            let mut logits = Vec::with_capacity(query_token + 1);
            for key_token in 0..=query_token {
                let mut dot = 0.0_f32;
                for dim in 0..head_dim {
                    dot += inputs[0][index(query_token, head, dim, query_heads, head_dim)]
                        * repeated_key[index(key_token, head, dim, query_heads, head_dim)];
                }
                logits.push(dot * scale);
            }
            let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probabilities = logits
                .into_iter()
                .map(|logit| (logit - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = probabilities.iter().sum::<f32>();
            probabilities
                .iter_mut()
                .for_each(|probability| *probability /= denominator);
            for dim in 0..head_dim {
                let mut weighted = 0.0_f32;
                for (key_token, probability) in probabilities.iter().copied().enumerate() {
                    weighted += probability
                        * repeated_value[index(key_token, head, dim, query_heads, head_dim)];
                }
                output[index(query_token, head, dim, query_heads, head_dim)] = weighted;
            }
        }
    }
    output
}

fn invoke(
    inputs: &Inputs,
    tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, CpuRefError> {
    causal_gqa_f32(
        &inputs[0],
        &inputs[1],
        &inputs[2],
        tokens,
        query_heads,
        key_value_heads,
        head_dim,
    )
}

fn hash_f32_le(values: &[f32]) -> String {
    let mut hasher = blake3::Hasher::new();
    values.iter().for_each(|value| {
        hasher.update(&value.to_le_bytes());
    });
    hasher.finalize().to_hex().to_string()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (offset, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-6,
            "{label}[{offset}] actual={actual:?} expected={expected:?}"
        );
    }
}

fn assert_error<T>(result: Result<T, CpuRefError>, expected: CpuRefErrorCode) {
    match result {
        Err(error) => assert_eq!(error.code(), expected),
        Ok(_) => panic!("expected {expected:?}"),
    }
}

#[test]
fn asymmetric_gqa_matches_physical_repeat_oracle_digest_and_contiguous_grouping() {
    let inputs = fixture(TOKENS, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
    let preserved = inputs.clone();
    let expected = independent_causal_gqa(&inputs, TOKENS, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
    assert_eq!(hash_f32_le(&expected), COMPACT_BLAKE3);
    let actual = invoke(&inputs, TOKENS, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM).unwrap();
    assert_eq!(hash_f32_le(&actual), COMPACT_BLAKE3);
    assert_eq!(actual, expected);
    assert_eq!(inputs, preserved);

    for (query_head, kv_head) in [0, 0, 1, 1].into_iter().enumerate() {
        let output_start = index(0, query_head, 0, QUERY_HEADS, HEAD_DIM);
        let value_start = index(0, kv_head, 0, KEY_VALUE_HEADS, HEAD_DIM);
        assert_eq!(
            &actual[output_start..output_start + HEAD_DIM],
            &inputs[2][value_start..value_start + HEAD_DIM]
        );
    }
}

#[test]
fn generic_gqa_supports_group_one_and_single_kv_head_and_rejects_swapped_mapping() {
    let tokens = 6;
    for (query_heads, key_value_heads) in [(2, 2), (4, 1)] {
        let inputs = fixture(tokens, query_heads, key_value_heads, HEAD_DIM);
        let expected =
            independent_causal_gqa(&inputs, tokens, query_heads, key_value_heads, HEAD_DIM);
        let actual = invoke(&inputs, tokens, query_heads, key_value_heads, HEAD_DIM).unwrap();
        assert_close(
            &actual,
            &expected,
            &format!("qh={query_heads} kvh={key_value_heads}"),
        );
    }

    let inputs = fixture(TOKENS, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
    let expected = independent_causal_gqa(&inputs, TOKENS, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
    let mut swapped = inputs.clone();
    for operand in swapped.iter_mut().skip(1) {
        for token in 0..TOKENS {
            for dim in 0..HEAD_DIM {
                let first = index(token, 0, dim, KEY_VALUE_HEADS, HEAD_DIM);
                let second = index(token, 1, dim, KEY_VALUE_HEADS, HEAD_DIM);
                operand.swap(first, second);
            }
        }
    }
    let swapped_mapping =
        independent_causal_gqa(&swapped, TOKENS, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
    assert_ne!(
        bits(&expected),
        bits(&swapped_mapping),
        "mapping q-head groups to the opposite KV head must be observable"
    );
}

#[test]
fn pinned_gqa_accepts_synthetic_single_and_multi_token_lengths() {
    const PINNED_QUERY_HEADS: usize = 16;
    const PINNED_KEY_VALUE_HEADS: usize = 2;
    const PINNED_HEAD_DIM: usize = 128;

    for tokens in [1, 4] {
        let inputs = fixture(
            tokens,
            PINNED_QUERY_HEADS,
            PINNED_KEY_VALUE_HEADS,
            PINNED_HEAD_DIM,
        );
        let preserved = inputs.clone();
        let expected = independent_causal_gqa(
            &inputs,
            tokens,
            PINNED_QUERY_HEADS,
            PINNED_KEY_VALUE_HEADS,
            PINNED_HEAD_DIM,
        );
        let actual =
            pinned_decoder_causal_gqa_f32(&inputs[0], &inputs[1], &inputs[2], tokens).unwrap();
        assert_close(&actual, &expected, &format!("pinned tokens={tokens}"));
        assert_eq!(inputs, preserved, "pinned tokens={tokens}");
    }
}

#[test]
fn causal_prefix_is_bitwise_immune_to_future_kv_poison() {
    let tokens = 7;
    let inputs = fixture(tokens, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
    let baseline = invoke(&inputs, tokens, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM).unwrap();
    let poison_start = 3;
    let mut poisoned = inputs.clone();
    for token in poison_start..tokens {
        for head in 0..KEY_VALUE_HEADS {
            for dim in 0..HEAD_DIM {
                let offset = index(token, head, dim, KEY_VALUE_HEADS, HEAD_DIM);
                poisoned[1][offset] = if (token + head + dim).is_multiple_of(2) {
                    80.0
                } else {
                    -80.0
                };
                poisoned[2][offset] = 500.0 + offset as f32;
            }
        }
    }
    let poisoned_output =
        invoke(&poisoned, tokens, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM).unwrap();
    let prefix = poison_start * QUERY_HEADS * HEAD_DIM;
    assert_eq!(bits(&baseline[..prefix]), bits(&poisoned_output[..prefix]));
    for token in poison_start..tokens {
        let start = token * QUERY_HEADS * HEAD_DIM;
        let end = start + QUERY_HEADS * HEAD_DIM;
        assert_ne!(
            bits(&baseline[start..end]),
            bits(&poisoned_output[start..end])
        );
    }
}

#[test]
fn gqa_matches_oracle_across_boundaries_and_stabilizes_bounded_extreme_logits() {
    for tokens in [1, 2, 15, 16, 17, 31, 32, 33] {
        let inputs = fixture(tokens, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
        let expected =
            independent_causal_gqa(&inputs, tokens, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
        let actual = invoke(&inputs, tokens, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM).unwrap();
        assert_close(&actual, &expected, &format!("tokens={tokens}"));
    }

    let tokens = 3;
    let mut extreme = fixture(tokens, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
    let [extreme_query, extreme_key, _] = &mut extreme;
    for (offset, value) in extreme_query.iter_mut().chain(extreme_key).enumerate() {
        *value = if offset.is_multiple_of(2) {
            1.0e10
        } else {
            -1.0e10
        };
    }
    let expected = independent_causal_gqa(&extreme, tokens, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
    let actual = invoke(&extreme, tokens, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM).unwrap();
    assert_close(&actual, &expected, "extreme logits");
    assert!(actual.iter().all(|value| value.is_finite()));
}

#[test]
fn gqa_fail_closes_for_geometry_lengths_overflow_and_nonfinite_inputs() {
    let query_len = TOKENS * QUERY_HEADS * HEAD_DIM;
    let kv_len = TOKENS * KEY_VALUE_HEADS * HEAD_DIM;
    #[rustfmt::skip]
    let invalid_geometry = [
        (0, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM, [0, 0, 0]),
        (TOKENS, 0, KEY_VALUE_HEADS, HEAD_DIM, [0, kv_len, kv_len]),
        (TOKENS, QUERY_HEADS, 0, HEAD_DIM, [query_len, 0, 0]),
        (TOKENS, QUERY_HEADS, KEY_VALUE_HEADS, 0, [0, 0, 0]),
        (TOKENS, 3, KEY_VALUE_HEADS, HEAD_DIM, [TOKENS * 3 * HEAD_DIM, kv_len, kv_len]),
        (usize::MAX, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM, [0, 0, 0]),
        (1, usize::MAX, 1, 2, [0, 0, 0]),
    ];
    for (tokens, query_heads, key_value_heads, head_dim, lengths) in invalid_geometry {
        let inputs = lengths.map(|length| vec![0.0; length]);
        assert_error(
            invoke(&inputs, tokens, query_heads, key_value_heads, head_dim),
            CpuRefErrorCode::DimensionMismatch,
        );
    }

    let finite = fixture(TOKENS, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
    for operand in 0..3 {
        let mut short = finite.clone();
        short[operand].pop();
        assert_error(
            invoke(&short, TOKENS, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM),
            CpuRefErrorCode::DimensionMismatch,
        );
        let mut long = finite.clone();
        long[operand].push(0.0);
        assert_error(
            invoke(&long, TOKENS, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM),
            CpuRefErrorCode::DimensionMismatch,
        );
    }
    for operand in 0..3 {
        let len = finite[operand].len();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for offset in [0, len / 2, len - 1] {
                let mut inputs = finite.clone();
                inputs[operand][offset] = value;
                assert_error(
                    invoke(&inputs, TOKENS, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM),
                    CpuRefErrorCode::NonFiniteInput,
                );
            }
        }
    }
}

fn assert_pinned_cache(cache: &DecoderPrefillKvCache, tokens: usize) {
    assert_eq!(cache.tokens, tokens);
    assert_eq!(cache.key_value_heads, 2);
    assert_eq!(cache.head_dim, 128);
}

#[test]
fn pinned_prefill_kv_write_detaches_and_isolates_exact_storage() {
    let tokens = 3;
    let len = tokens * 2 * 128;
    let mut key = (0..len)
        .map(|index| (index as f32 - 311.0) / 97.0)
        .collect::<Vec<_>>();
    let mut value = (0..len)
        .map(|index| (503.0 - index as f32) / 83.0)
        .collect::<Vec<_>>();
    let expected_key = key.clone();
    let expected_value = value.clone();
    let mut cache = write_pinned_decoder_prefill_kv_f32(&key, &value, tokens).unwrap();
    assert_pinned_cache(&cache, tokens);
    assert_eq!(cache.keys, expected_key);
    assert_eq!(cache.values, expected_value);

    key.fill(f32::NAN);
    value.fill(f32::INFINITY);
    assert_eq!(cache.keys, expected_key);
    assert_eq!(cache.values, expected_value);
    let value_zero = cache.values[0];
    cache.keys[0] = 1234.0;
    assert_eq!(cache.values[0], value_zero);
    let key_one = cache.keys[1];
    cache.values[1] = -5678.0;
    assert_eq!(cache.keys[1], key_one);
}

#[test]
fn pinned_prefill_kv_write_rejects_malformed_and_nonfinite_inputs() {
    let tokens = 2;
    let len = tokens * 2 * 128;
    let finite = [vec![0.25; len], vec![-0.5; len]];
    assert_error(
        write_pinned_decoder_prefill_kv_f32(&finite[0], &finite[1], 0),
        CpuRefErrorCode::DimensionMismatch,
    );
    assert_error(
        write_pinned_decoder_prefill_kv_f32(&[], &[], usize::MAX),
        CpuRefErrorCode::DimensionMismatch,
    );
    for operand in 0..2 {
        let mut short = finite.clone();
        short[operand].pop();
        assert_error(
            write_pinned_decoder_prefill_kv_f32(&short[0], &short[1], tokens),
            CpuRefErrorCode::DimensionMismatch,
        );
        let mut long = finite.clone();
        long[operand].push(0.0);
        assert_error(
            write_pinned_decoder_prefill_kv_f32(&long[0], &long[1], tokens),
            CpuRefErrorCode::DimensionMismatch,
        );
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for offset in [0, len / 2, len - 1] {
                let mut inputs = finite.clone();
                inputs[operand][offset] = value;
                assert_error(
                    write_pinned_decoder_prefill_kv_f32(&inputs[0], &inputs[1], tokens),
                    CpuRefErrorCode::NonFiniteInput,
                );
            }
        }
    }
}
