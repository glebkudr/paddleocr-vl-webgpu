use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, DecoderLayerConfig, DecoderLayerParameters,
    DecoderLayerPrefillTrace, DecoderPrefillKvCache, add_vectors_f32, apply_multimodal_rope_f32,
    causal_gqa_f32, decoder_layer_prefill_f32, linear_f32, rms_norm_f32, silu,
};

const TOKENS: usize = 3;
const HIDDEN: usize = 5;
const INTERMEDIATE: usize = 7;
const QUERY_HEADS: usize = 4;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIM: usize = 6;
const EPSILON: f32 = 1.0e-5;
const SECTIONS: [usize; 3] = [1, 1, 1];
// Derived once on 2026-07-19 from `independent_trace`, while the composed API
// was still absent, using only the already accepted public primitives below.
// The expected digest is deliberately a literal, never a runtime self-anchor.
const COMBINED_BLAKE3: &str = "61ab6647b035c78971b02d51c4600f5e78d92686deda863809ea0e7f9c35740e";

#[derive(Clone, Copy, Debug)]
enum Operand {
    Input,
    RawCos,
    RawSin,
    InputNormWeight,
    QueryWeight,
    KeyWeight,
    ValueWeight,
    AttentionOutputWeight,
    PostAttentionNormWeight,
    GateWeight,
    UpWeight,
    DownWeight,
}

const OPERANDS: [Operand; 12] = [
    Operand::Input,
    Operand::RawCos,
    Operand::RawSin,
    Operand::InputNormWeight,
    Operand::QueryWeight,
    Operand::KeyWeight,
    Operand::ValueWeight,
    Operand::AttentionOutputWeight,
    Operand::PostAttentionNormWeight,
    Operand::GateWeight,
    Operand::UpWeight,
    Operand::DownWeight,
];

impl Operand {
    const fn label(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::RawCos => "raw_cos",
            Self::RawSin => "raw_sin",
            Self::InputNormWeight => "input_norm_weight",
            Self::QueryWeight => "query_weight",
            Self::KeyWeight => "key_weight",
            Self::ValueWeight => "value_weight",
            Self::AttentionOutputWeight => "attention_output_weight",
            Self::PostAttentionNormWeight => "post_attention_norm_weight",
            Self::GateWeight => "gate_weight",
            Self::UpWeight => "up_weight",
            Self::DownWeight => "down_weight",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct OwnedParameters {
    input_norm_weight: Vec<f32>,
    query_weight: Vec<f32>,
    key_weight: Vec<f32>,
    value_weight: Vec<f32>,
    attention_output_weight: Vec<f32>,
    post_attention_norm_weight: Vec<f32>,
    gate_weight: Vec<f32>,
    up_weight: Vec<f32>,
    down_weight: Vec<f32>,
}

impl OwnedParameters {
    fn borrowed(&self) -> DecoderLayerParameters<'_> {
        DecoderLayerParameters {
            input_norm_weight: &self.input_norm_weight,
            query_weight: &self.query_weight,
            key_weight: &self.key_weight,
            value_weight: &self.value_weight,
            attention_output_weight: &self.attention_output_weight,
            post_attention_norm_weight: &self.post_attention_norm_weight,
            gate_weight: &self.gate_weight,
            up_weight: &self.up_weight,
            down_weight: &self.down_weight,
        }
    }

    fn operand_mut(&mut self, operand: Operand) -> &mut Vec<f32> {
        match operand {
            Operand::InputNormWeight => &mut self.input_norm_weight,
            Operand::QueryWeight => &mut self.query_weight,
            Operand::KeyWeight => &mut self.key_weight,
            Operand::ValueWeight => &mut self.value_weight,
            Operand::AttentionOutputWeight => &mut self.attention_output_weight,
            Operand::PostAttentionNormWeight => &mut self.post_attention_norm_weight,
            Operand::GateWeight => &mut self.gate_weight,
            Operand::UpWeight => &mut self.up_weight,
            Operand::DownWeight => &mut self.down_weight,
            Operand::Input | Operand::RawCos | Operand::RawSin => {
                panic!("{} is not a weight", operand.label())
            }
        }
    }
}

fn operand_mut<'a>(
    operand: Operand,
    input: &'a mut Vec<f32>,
    raw_cos: &'a mut Vec<f32>,
    raw_sin: &'a mut Vec<f32>,
    parameters: &'a mut OwnedParameters,
) -> &'a mut Vec<f32> {
    match operand {
        Operand::Input => input,
        Operand::RawCos => raw_cos,
        Operand::RawSin => raw_sin,
        weight => parameters.operand_mut(weight),
    }
}

fn config(tokens: usize) -> DecoderLayerConfig {
    DecoderLayerConfig {
        tokens,
        hidden_size: HIDDEN,
        intermediate_size: INTERMEDIATE,
        query_heads: QUERY_HEADS,
        key_value_heads: KEY_VALUE_HEADS,
        head_dim: HEAD_DIM,
        rms_norm_epsilon: EPSILON,
        mrope_sections: SECTIONS,
    }
}

fn dense(len: usize, mul: usize, add: usize, modulus: usize, divisor: f32) -> Vec<f32> {
    (0..len)
        .map(|index| (((index * mul + add) % modulus) as f32 - (modulus / 2) as f32) / divisor)
        .collect()
}

fn parameters() -> OwnedParameters {
    OwnedParameters {
        input_norm_weight: (0..HIDDEN)
            .map(|index| 0.7 + ((index * 3 + 1) % 7) as f32 / 20.0)
            .collect(),
        query_weight: dense(QUERY_HEADS * HEAD_DIM * HIDDEN, 7, 3, 29, 19.0),
        key_weight: dense(KEY_VALUE_HEADS * HEAD_DIM * HIDDEN, 11, 5, 31, 23.0),
        value_weight: dense(KEY_VALUE_HEADS * HEAD_DIM * HIDDEN, 13, 7, 37, 29.0),
        attention_output_weight: dense(HIDDEN * QUERY_HEADS * HEAD_DIM, 17, 2, 41, 31.0),
        post_attention_norm_weight: (0..HIDDEN)
            .map(|index| 0.65 + ((index * 5 + 2) % 9) as f32 / 24.0)
            .collect(),
        gate_weight: dense(INTERMEDIATE * HIDDEN, 19, 1, 43, 37.0),
        up_weight: dense(INTERMEDIATE * HIDDEN, 23, 4, 47, 41.0),
        down_weight: dense(HIDDEN * INTERMEDIATE, 29, 6, 53, 43.0),
    }
}

fn input(tokens: usize) -> Vec<f32> {
    dense(tokens * HIDDEN, 17, 3, 29, 11.0)
}

fn raw_tables(tokens: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = Vec::with_capacity(3 * tokens * HEAD_DIM);
    let mut sine = Vec::with_capacity(3 * tokens * HEAD_DIM);
    for axis in 0..3 {
        for token in 0..tokens {
            for dim in 0..HEAD_DIM {
                cosine.push(0.55 + axis as f32 * 0.13 + token as f32 * 0.02 + dim as f32 * 0.007);
                sine.push(-0.3 - axis as f32 * 0.11 + token as f32 * 0.015 - dim as f32 * 0.005);
            }
        }
    }
    (cosine, sine)
}

fn independent_trace(
    input: &[f32],
    config: DecoderLayerConfig,
    raw_cos: &[f32],
    raw_sin: &[f32],
    parameters: &OwnedParameters,
) -> DecoderLayerPrefillTrace {
    let query_width = config.query_heads * config.head_dim;
    let key_value_width = config.key_value_heads * config.head_dim;
    let zero_query = vec![0.0; query_width];
    let zero_key_value = vec![0.0; key_value_width];
    let zero_hidden = vec![0.0; config.hidden_size];
    let zero_intermediate = vec![0.0; config.intermediate_size];
    let norm1 = rms_norm_f32(
        input,
        config.tokens,
        config.hidden_size,
        &parameters.input_norm_weight,
        config.rms_norm_epsilon,
    )
    .unwrap();
    let query = linear_f32(
        &norm1,
        config.tokens,
        config.hidden_size,
        &parameters.query_weight,
        &zero_query,
        query_width,
    )
    .unwrap();
    let key = linear_f32(
        &norm1,
        config.tokens,
        config.hidden_size,
        &parameters.key_weight,
        &zero_key_value,
        key_value_width,
    )
    .unwrap();
    let value = linear_f32(
        &norm1,
        config.tokens,
        config.hidden_size,
        &parameters.value_weight,
        &zero_key_value,
        key_value_width,
    )
    .unwrap();
    let (mrope_query, mrope_key) = apply_multimodal_rope_f32(
        &query,
        &key,
        config.tokens,
        config.query_heads,
        config.key_value_heads,
        config.head_dim,
        raw_cos,
        raw_sin,
        config.mrope_sections,
    )
    .unwrap();
    let kv_cache = DecoderPrefillKvCache {
        keys: mrope_key.clone(),
        values: value.clone(),
        tokens: config.tokens,
        key_value_heads: config.key_value_heads,
        head_dim: config.head_dim,
    };
    let attention_context = causal_gqa_f32(
        &mrope_query,
        &kv_cache.keys,
        &kv_cache.values,
        config.tokens,
        config.query_heads,
        config.key_value_heads,
        config.head_dim,
    )
    .unwrap();
    let attention_output = linear_f32(
        &attention_context,
        config.tokens,
        query_width,
        &parameters.attention_output_weight,
        &zero_hidden,
        config.hidden_size,
    )
    .unwrap();
    let attention_residual = add_vectors_f32(input, &attention_output).unwrap();
    let norm2 = rms_norm_f32(
        &attention_residual,
        config.tokens,
        config.hidden_size,
        &parameters.post_attention_norm_weight,
        config.rms_norm_epsilon,
    )
    .unwrap();
    let mlp_gate = linear_f32(
        &norm2,
        config.tokens,
        config.hidden_size,
        &parameters.gate_weight,
        &zero_intermediate,
        config.intermediate_size,
    )
    .unwrap();
    let mlp_up = linear_f32(
        &norm2,
        config.tokens,
        config.hidden_size,
        &parameters.up_weight,
        &zero_intermediate,
        config.intermediate_size,
    )
    .unwrap();
    let mlp_activation = mlp_gate
        .iter()
        .zip(&mlp_up)
        .map(|(&gate, &up)| silu(gate) * up)
        .collect::<Vec<_>>();
    let mlp_down = linear_f32(
        &mlp_activation,
        config.tokens,
        config.intermediate_size,
        &parameters.down_weight,
        &zero_hidden,
        config.hidden_size,
    )
    .unwrap();
    let output = add_vectors_f32(&attention_residual, &mlp_down).unwrap();
    DecoderLayerPrefillTrace {
        norm1,
        query,
        key,
        value,
        mrope_query,
        mrope_key,
        kv_cache,
        attention_context,
        attention_output,
        attention_residual,
        norm2,
        mlp_gate,
        mlp_up,
        mlp_activation,
        mlp_down,
        output,
    }
}

fn invoke(
    input: &[f32],
    config: DecoderLayerConfig,
    raw_cos: &[f32],
    raw_sin: &[f32],
    parameters: &OwnedParameters,
) -> Result<DecoderLayerPrefillTrace, CpuRefError> {
    decoder_layer_prefill_f32(input, config, raw_cos, raw_sin, parameters.borrowed())
}

fn assert_f32_bits(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label}[{index}]: actual={actual:?}, expected={expected:?}"
        );
    }
}

fn assert_trace_exact(actual: &DecoderLayerPrefillTrace, expected: &DecoderLayerPrefillTrace) {
    let fields: [(&str, &[f32], &[f32]); 17] = [
        ("norm1", &actual.norm1, &expected.norm1),
        ("query", &actual.query, &expected.query),
        ("key", &actual.key, &expected.key),
        ("value", &actual.value, &expected.value),
        ("mrope_query", &actual.mrope_query, &expected.mrope_query),
        ("mrope_key", &actual.mrope_key, &expected.mrope_key),
        (
            "kv_cache.keys",
            &actual.kv_cache.keys,
            &expected.kv_cache.keys,
        ),
        (
            "kv_cache.values",
            &actual.kv_cache.values,
            &expected.kv_cache.values,
        ),
        (
            "attention_context",
            &actual.attention_context,
            &expected.attention_context,
        ),
        (
            "attention_output",
            &actual.attention_output,
            &expected.attention_output,
        ),
        (
            "attention_residual",
            &actual.attention_residual,
            &expected.attention_residual,
        ),
        ("norm2", &actual.norm2, &expected.norm2),
        ("mlp_gate", &actual.mlp_gate, &expected.mlp_gate),
        ("mlp_up", &actual.mlp_up, &expected.mlp_up),
        (
            "mlp_activation",
            &actual.mlp_activation,
            &expected.mlp_activation,
        ),
        ("mlp_down", &actual.mlp_down, &expected.mlp_down),
        ("output", &actual.output, &expected.output),
    ];
    for (label, actual, expected) in fields {
        assert_f32_bits(label, actual, expected);
    }
    assert_eq!(actual.kv_cache.tokens, expected.kv_cache.tokens);
    assert_eq!(
        actual.kv_cache.key_value_heads,
        expected.kv_cache.key_value_heads
    );
    assert_eq!(actual.kv_cache.head_dim, expected.kv_cache.head_dim);
}

// Independent pre-production anchor order: norm1, q, k, v, mrope q/k,
// cache keys/values, cache tokens/KV-heads/head-dim as u64 LE, then context,
// attention output/residual, norm2, gate, up, activation, down, final output.
fn combined_digest(trace: &DecoderLayerPrefillTrace) -> String {
    let mut hasher = blake3::Hasher::new();
    let prefix: [&[f32]; 8] = [
        &trace.norm1,
        &trace.query,
        &trace.key,
        &trace.value,
        &trace.mrope_query,
        &trace.mrope_key,
        &trace.kv_cache.keys,
        &trace.kv_cache.values,
    ];
    for values in prefix {
        update_f32_digest(&mut hasher, values);
    }
    for metadata in [
        trace.kv_cache.tokens,
        trace.kv_cache.key_value_heads,
        trace.kv_cache.head_dim,
    ] {
        hasher.update(&(metadata as u64).to_le_bytes());
    }
    let suffix: [&[f32]; 9] = [
        &trace.attention_context,
        &trace.attention_output,
        &trace.attention_residual,
        &trace.norm2,
        &trace.mlp_gate,
        &trace.mlp_up,
        &trace.mlp_activation,
        &trace.mlp_down,
        &trace.output,
    ];
    for values in suffix {
        update_f32_digest(&mut hasher, values);
    }
    hasher.finalize().to_hex().to_string()
}

fn update_f32_digest(hasher: &mut blake3::Hasher, values: &[f32]) {
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
}

fn tensor_digest(values: &[f32]) -> String {
    let mut hasher = blake3::Hasher::new();
    update_f32_digest(&mut hasher, values);
    hasher.finalize().to_hex().to_string()
}

fn assert_digest_differs(label: &str, wrong: &[f32], correct: &[f32]) {
    assert_ne!(tensor_digest(wrong), tensor_digest(correct), "{label}");
}

fn assert_error<T>(case: &str, result: Result<T, CpuRefError>, expected: CpuRefErrorCode) {
    match result {
        Err(error) => assert_eq!(error.code(), expected, "{case}"),
        Ok(_) => panic!("{case}: expected {expected:?}"),
    }
}

fn assert_cache_detached_and_isolated(trace: &mut DecoderLayerPrefillTrace) {
    assert_f32_bits(
        "cache equals rotated key",
        &trace.kv_cache.keys,
        &trace.mrope_key,
    );
    assert_f32_bits("cache equals value", &trace.kv_cache.values, &trace.value);
    assert_ne!(trace.kv_cache.keys.as_ptr(), trace.mrope_key.as_ptr());
    assert_ne!(trace.kv_cache.values.as_ptr(), trace.value.as_ptr());
    assert_ne!(trace.kv_cache.keys.as_ptr(), trace.kv_cache.values.as_ptr());

    let rotated_key_zero = trace.mrope_key[0];
    let value_zero = trace.value[0];
    let cached_value_zero = trace.kv_cache.values[0];
    trace.kv_cache.keys[0] = 123.25;
    assert_eq!(trace.mrope_key[0].to_bits(), rotated_key_zero.to_bits());
    assert_eq!(
        trace.kv_cache.values[0].to_bits(),
        cached_value_zero.to_bits()
    );
    trace.kv_cache.values[0] = -456.5;
    assert_eq!(trace.value[0].to_bits(), value_zero.to_bits());
    assert_eq!(trace.kv_cache.keys[0].to_bits(), 123.25_f32.to_bits());
}

#[test]
fn tiny_layer_matches_independent_composition_anchor_and_preserves_inputs() {
    let config = config(TOKENS);
    let input = input(TOKENS);
    let (raw_cos, raw_sin) = raw_tables(TOKENS);
    let parameters = parameters();
    let preserved = (
        input.clone(),
        raw_cos.clone(),
        raw_sin.clone(),
        parameters.clone(),
    );
    let expected = independent_trace(&input, config, &raw_cos, &raw_sin, &parameters);
    assert_eq!(combined_digest(&expected), COMBINED_BLAKE3);
    let mut actual = invoke(&input, config, &raw_cos, &raw_sin, &parameters).unwrap();
    assert_trace_exact(&actual, &expected);
    assert_eq!(combined_digest(&actual), COMBINED_BLAKE3);
    assert_eq!(input, preserved.0);
    assert_eq!(raw_cos, preserved.1);
    assert_eq!(raw_sin, preserved.2);
    assert_eq!(parameters, preserved.3);

    assert_cache_detached_and_isolated(&mut actual);
}

#[test]
fn gate_norm_and_residual_negative_controls_diverge() {
    let config = config(TOKENS);
    let input = input(TOKENS);
    let (raw_cos, raw_sin) = raw_tables(TOKENS);
    let parameters = parameters();
    let expected = independent_trace(&input, config, &raw_cos, &raw_sin, &parameters);
    let zero_hidden = vec![0.0; HIDDEN];
    let zero_intermediate = vec![0.0; INTERMEDIATE];

    let swapped_activation = expected
        .mlp_gate
        .iter()
        .zip(&expected.mlp_up)
        .map(|(&gate, &up)| silu(up) * gate)
        .collect::<Vec<_>>();
    let swapped_down = linear_f32(
        &swapped_activation,
        TOKENS,
        INTERMEDIATE,
        &parameters.down_weight,
        &zero_hidden,
        HIDDEN,
    )
    .unwrap();
    let swapped_output = add_vectors_f32(&expected.attention_residual, &swapped_down).unwrap();
    assert_digest_differs(
        "SiLU(up) * gate activation digest",
        &swapped_activation,
        &expected.mlp_activation,
    );
    assert_digest_differs(
        "SiLU(up) * gate final digest",
        &swapped_output,
        &expected.output,
    );

    let skipped_gate = linear_f32(
        &expected.attention_residual,
        TOKENS,
        HIDDEN,
        &parameters.gate_weight,
        &zero_intermediate,
        INTERMEDIATE,
    )
    .unwrap();
    let skipped_up = linear_f32(
        &expected.attention_residual,
        TOKENS,
        HIDDEN,
        &parameters.up_weight,
        &zero_intermediate,
        INTERMEDIATE,
    )
    .unwrap();
    let skipped_activation = skipped_gate
        .iter()
        .zip(skipped_up)
        .map(|(&gate, up)| silu(gate) * up)
        .collect::<Vec<_>>();
    let skipped_down = linear_f32(
        &skipped_activation,
        TOKENS,
        INTERMEDIATE,
        &parameters.down_weight,
        &zero_hidden,
        HIDDEN,
    )
    .unwrap();
    let skipped_output = add_vectors_f32(&expected.attention_residual, &skipped_down).unwrap();
    assert_digest_differs(
        "skipped second RMSNorm stage digest",
        &expected.attention_residual,
        &expected.norm2,
    );
    assert_digest_differs(
        "skipped second RMSNorm activation digest",
        &skipped_activation,
        &expected.mlp_activation,
    );
    assert_digest_differs(
        "skipped second RMSNorm final digest",
        &skipped_output,
        &expected.output,
    );

    let wrong_residual = add_vectors_f32(&input, &expected.mlp_down).unwrap();
    assert_digest_differs(
        "wrong final residual source digest",
        &wrong_residual,
        &expected.output,
    );
}

#[test]
fn generic_layer_accepts_single_and_small_multi_token_lengths() {
    let parameters = parameters();
    for tokens in [1, 4] {
        let config = config(tokens);
        let input = input(tokens);
        let (raw_cos, raw_sin) = raw_tables(tokens);
        let expected = independent_trace(&input, config, &raw_cos, &raw_sin, &parameters);
        let mut actual = invoke(&input, config, &raw_cos, &raw_sin, &parameters).unwrap();
        assert_trace_exact(&actual, &expected);
        assert_eq!(actual.kv_cache.tokens, tokens);
        assert_eq!(actual.kv_cache.key_value_heads, KEY_VALUE_HEADS);
        assert_eq!(actual.kv_cache.head_dim, HEAD_DIM);
        assert_cache_detached_and_isolated(&mut actual);
    }
}

fn assert_prefix_bits(left: &[f32], right: &[f32], tokens: usize, width: usize, label: &str) {
    let end = tokens * width;
    let left = left[..end]
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    let right = right[..end]
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(left, right, "{label}");
}

fn assert_row_bits_differ(label: &str, left: &[f32], right: &[f32], token: usize, width: usize) {
    let range = token * width..(token + 1) * width;
    let left_bits = left[range.clone()]
        .iter()
        .map(|value: &f32| value.to_bits())
        .collect::<Vec<_>>();
    let right_bits = right[range]
        .iter()
        .map(|value: &f32| value.to_bits())
        .collect::<Vec<_>>();
    assert_ne!(left_bits, right_bits, "{label} token {token}");
}

fn assert_raw_prefix_unchanged(
    label: &str,
    baseline: &[f32],
    poisoned: &[f32],
    tokens: usize,
    prefix_tokens: usize,
) {
    for axis in 0..3 {
        let start = axis * tokens * HEAD_DIM;
        let end = start + prefix_tokens * HEAD_DIM;
        assert_f32_bits(
            &format!("{label} axis {axis} prefix"),
            &poisoned[start..end],
            &baseline[start..end],
        );
    }
}

#[test]
fn future_input_and_rope_poison_cannot_change_any_prefix_stage() {
    let tokens = 5;
    let poison_start = 3;
    let config = config(tokens);
    let parameters = parameters();
    let input = input(tokens);
    let (raw_cos, raw_sin) = raw_tables(tokens);
    let preserved = (
        input.clone(),
        raw_cos.clone(),
        raw_sin.clone(),
        parameters.clone(),
    );
    let baseline = invoke(&input, config, &raw_cos, &raw_sin, &parameters).unwrap();
    let mut poisoned_input = input.clone();
    let mut poisoned_cos = raw_cos.clone();
    let mut poisoned_sin = raw_sin.clone();
    poisoned_input[poison_start * HIDDEN..].fill(77.0);
    for axis in 0..3 {
        for token in poison_start..tokens {
            let start = (axis * tokens + token) * HEAD_DIM;
            poisoned_cos[start..start + HEAD_DIM].fill(33.0 + axis as f32);
            poisoned_sin[start..start + HEAD_DIM].fill(-44.0 - axis as f32);
        }
    }
    let preserved_poison = (
        poisoned_input.clone(),
        poisoned_cos.clone(),
        poisoned_sin.clone(),
    );
    let poisoned = invoke(
        &poisoned_input,
        config,
        &poisoned_cos,
        &poisoned_sin,
        &parameters,
    )
    .unwrap();
    assert_prefix_bits(
        &input,
        &poisoned_input,
        poison_start,
        HIDDEN,
        "poisoned input prefix",
    );
    assert_raw_prefix_unchanged(
        "poisoned raw cosine",
        &raw_cos,
        &poisoned_cos,
        tokens,
        poison_start,
    );
    assert_raw_prefix_unchanged(
        "poisoned raw sine",
        &raw_sin,
        &poisoned_sin,
        tokens,
        poison_start,
    );
    for (label, left, right, width) in [
        ("norm1", &baseline.norm1, &poisoned.norm1, HIDDEN),
        (
            "query",
            &baseline.query,
            &poisoned.query,
            QUERY_HEADS * HEAD_DIM,
        ),
        (
            "key",
            &baseline.key,
            &poisoned.key,
            KEY_VALUE_HEADS * HEAD_DIM,
        ),
        (
            "value",
            &baseline.value,
            &poisoned.value,
            KEY_VALUE_HEADS * HEAD_DIM,
        ),
        (
            "mrope_query",
            &baseline.mrope_query,
            &poisoned.mrope_query,
            QUERY_HEADS * HEAD_DIM,
        ),
        (
            "mrope_key",
            &baseline.mrope_key,
            &poisoned.mrope_key,
            KEY_VALUE_HEADS * HEAD_DIM,
        ),
        (
            "cache_key",
            &baseline.kv_cache.keys,
            &poisoned.kv_cache.keys,
            KEY_VALUE_HEADS * HEAD_DIM,
        ),
        (
            "cache_value",
            &baseline.kv_cache.values,
            &poisoned.kv_cache.values,
            KEY_VALUE_HEADS * HEAD_DIM,
        ),
        (
            "context",
            &baseline.attention_context,
            &poisoned.attention_context,
            QUERY_HEADS * HEAD_DIM,
        ),
        (
            "attention_output",
            &baseline.attention_output,
            &poisoned.attention_output,
            HIDDEN,
        ),
        (
            "attention_residual",
            &baseline.attention_residual,
            &poisoned.attention_residual,
            HIDDEN,
        ),
        ("norm2", &baseline.norm2, &poisoned.norm2, HIDDEN),
        ("gate", &baseline.mlp_gate, &poisoned.mlp_gate, INTERMEDIATE),
        ("up", &baseline.mlp_up, &poisoned.mlp_up, INTERMEDIATE),
        (
            "activation",
            &baseline.mlp_activation,
            &poisoned.mlp_activation,
            INTERMEDIATE,
        ),
        ("down", &baseline.mlp_down, &poisoned.mlp_down, HIDDEN),
        ("output", &baseline.output, &poisoned.output, HIDDEN),
    ] {
        assert_prefix_bits(left, right, poison_start, width, label);
    }
    for token in poison_start..tokens {
        assert_row_bits_differ(
            "poisoned final suffix",
            &baseline.output,
            &poisoned.output,
            token,
            HIDDEN,
        );
    }
    assert_row_bits_differ(
        "poisoned row-local norm1 suffix",
        &baseline.norm1,
        &poisoned.norm1,
        poison_start,
        HIDDEN,
    );
    assert_row_bits_differ(
        "poisoned M-RoPE query suffix",
        &baseline.mrope_query,
        &poisoned.mrope_query,
        poison_start,
        QUERY_HEADS * HEAD_DIM,
    );
    assert_eq!(input, preserved.0);
    assert_eq!(raw_cos, preserved.1);
    assert_eq!(raw_sin, preserved.2);
    assert_eq!(parameters, preserved.3);
    assert_eq!(poisoned_input, preserved_poison.0);
    assert_eq!(poisoned_cos, preserved_poison.1);
    assert_eq!(poisoned_sin, preserved_poison.2);
}

#[test]
fn layer_fail_closes_for_geometry_epsilon_overflow_lengths_and_nonfinite_operands() {
    let base = config(TOKENS);
    let input = input(TOKENS);
    let (raw_cos, raw_sin) = raw_tables(TOKENS);
    let parameters = parameters();

    let mut invalid: Vec<(&str, DecoderLayerConfig)> = Vec::new();
    for field in 0..6 {
        let mut config = base;
        let label = match field {
            0 => {
                config.tokens = 0;
                "zero tokens"
            }
            1 => {
                config.hidden_size = 0;
                "zero hidden_size"
            }
            2 => {
                config.intermediate_size = 0;
                "zero intermediate_size"
            }
            3 => {
                config.query_heads = 0;
                "zero query_heads"
            }
            4 => {
                config.key_value_heads = 0;
                "zero key_value_heads"
            }
            5 => {
                config.head_dim = 0;
                "zero head_dim"
            }
            _ => unreachable!(),
        };
        invalid.push((label, config));
    }
    let mut not_divisible = base;
    not_divisible.query_heads = 3;
    invalid.push(("query heads not divisible by KV heads", not_divisible));
    let mut kv_wider = base;
    kv_wider.query_heads = 2;
    kv_wider.key_value_heads = 4;
    invalid.push(("KV heads wider than query heads", kv_wider));
    for (label, sections) in [
        ("zero temporal section", [0, 1, 2]),
        ("zero height section", [1, 0, 2]),
        ("zero width section", [1, 2, 0]),
    ] {
        let mut zero_section = base;
        zero_section.mrope_sections = sections;
        invalid.push((label, zero_section));
    }
    let mut wrong_sections = base;
    wrong_sections.mrope_sections = [1, 1, 2];
    invalid.push((
        "section sum doubled does not equal head_dim",
        wrong_sections,
    ));
    let mut section_sum_overflow = base;
    section_sum_overflow.mrope_sections = [usize::MAX, 1, 1];
    invalid.push(("M-RoPE section sum overflow", section_sum_overflow));
    let mut doubled_section_overflow = base;
    doubled_section_overflow.mrope_sections = [usize::MAX / 2, 1, 1];
    invalid.push((
        "M-RoPE doubled section sum overflow",
        doubled_section_overflow,
    ));
    for (case, config) in invalid {
        assert_error(
            case,
            invoke(&input, config, &raw_cos, &raw_sin, &parameters),
            CpuRefErrorCode::DimensionMismatch,
        );
    }
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut config = base;
        config.rms_norm_epsilon = epsilon;
        assert_error(
            &format!("invalid epsilon {epsilon:?}"),
            invoke(&input, config, &raw_cos, &raw_sin, &parameters),
            CpuRefErrorCode::NonPositiveEpsilon,
        );
    }

    let mut token_overflow = base;
    token_overflow.tokens = usize::MAX;
    let mut hidden_overflow = base;
    hidden_overflow.hidden_size = usize::MAX;
    let mut intermediate_overflow = base;
    intermediate_overflow.intermediate_size = usize::MAX;
    let mut query_width_overflow = base;
    query_width_overflow.query_heads = usize::MAX - 1;
    let mut key_value_width_overflow = base;
    key_value_width_overflow.query_heads = usize::MAX;
    key_value_width_overflow.key_value_heads = usize::MAX;
    for (case, overflowing) in [
        ("tokens times hidden_size overflow", token_overflow),
        (
            "tokens times hidden_size overflow via width",
            hidden_overflow,
        ),
        (
            "intermediate weight size multiplication overflow",
            intermediate_overflow,
        ),
        ("query_heads times head_dim overflow", query_width_overflow),
        (
            "key_value_heads times head_dim overflow",
            key_value_width_overflow,
        ),
    ] {
        assert_error(
            case,
            invoke(&[], overflowing, &[], &[], &parameters),
            CpuRefErrorCode::DimensionMismatch,
        );
    }

    for operand in OPERANDS {
        for long in [false, true] {
            let mut input = input.clone();
            let mut raw_cos = raw_cos.clone();
            let mut raw_sin = raw_sin.clone();
            let mut parameters = parameters.clone();
            let values = operand_mut(
                operand,
                &mut input,
                &mut raw_cos,
                &mut raw_sin,
                &mut parameters,
            );
            if long {
                values.push(0.0);
            } else {
                values.pop();
            }
            let case = format!(
                "{} {} length",
                operand.label(),
                if long { "long" } else { "short" }
            );
            assert_error(
                &case,
                invoke(&input, base, &raw_cos, &raw_sin, &parameters),
                CpuRefErrorCode::DimensionMismatch,
            );
        }
    }

    for operand in OPERANDS {
        let len = {
            let mut input = input.clone();
            let mut raw_cos = raw_cos.clone();
            let mut raw_sin = raw_sin.clone();
            let mut parameters = parameters.clone();
            operand_mut(
                operand,
                &mut input,
                &mut raw_cos,
                &mut raw_sin,
                &mut parameters,
            )
            .len()
        };
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for offset in [0, len / 2, len - 1] {
                let mut input = input.clone();
                let mut raw_cos = raw_cos.clone();
                let mut raw_sin = raw_sin.clone();
                let mut parameters = parameters.clone();
                operand_mut(
                    operand,
                    &mut input,
                    &mut raw_cos,
                    &mut raw_sin,
                    &mut parameters,
                )[offset] = value;
                let case = format!(
                    "{} non-finite {value:?} at offset {offset}",
                    operand.label()
                );
                assert_error(
                    &case,
                    invoke(&input, base, &raw_cos, &raw_sin, &parameters),
                    CpuRefErrorCode::NonFiniteInput,
                );
            }
        }
    }

    let mut extreme_input = vec![0.0; input.len()];
    for token in 0..TOKENS {
        extreme_input[token * HIDDEN] = 1.0;
    }
    let mut late_malformed = parameters.clone();
    late_malformed.input_norm_weight.fill(f32::MAX);
    late_malformed.query_weight.fill(f32::MAX);
    late_malformed.down_weight.pop();
    let overflowing_prefix = rms_norm_f32(
        &extreme_input,
        TOKENS,
        HIDDEN,
        &late_malformed.input_norm_weight,
        EPSILON,
    )
    .unwrap();
    assert!(
        overflowing_prefix.iter().any(|value| !value.is_finite()),
        "the finite early operands must overflow if execution starts"
    );
    let zero_query_bias = vec![0.0; QUERY_HEADS * HEAD_DIM];
    assert_error(
        "the independently executed early prefix becomes non-finite",
        linear_f32(
            &overflowing_prefix,
            TOKENS,
            HIDDEN,
            &late_malformed.query_weight,
            &zero_query_bias,
            QUERY_HEADS * HEAD_DIM,
        ),
        CpuRefErrorCode::NonFiniteInput,
    );
    assert_error(
        "late malformed down_weight wins before any arithmetic",
        invoke(&extreme_input, base, &raw_cos, &raw_sin, &late_malformed),
        CpuRefErrorCode::DimensionMismatch,
    );
}
