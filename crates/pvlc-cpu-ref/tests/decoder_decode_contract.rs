use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, DecoderLayerConfig, DecoderLayerDecodeTrace,
    DecoderLayerParameters, DecoderLayerPrefillTrace, DecoderPrefillKvCache, decode_gqa_f32,
    decoder_layer_decode_f32, decoder_layer_prefill_f32, pinned_decoder_decode_gqa_f32,
};

const QUERY_HEADS: usize = 4;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIM: usize = 6;
const HIDDEN_SIZE: usize = 7;
const INTERMEDIATE_SIZE: usize = 9;
const RMS_NORM_EPSILON: f32 = 1.0e-5;
const MROPE_SECTIONS: [usize; 3] = [1, 1, 1];
// Derived independently on 2026-07-20 in a throwaway temp Rust helper that
// hashed each boundary case as `cache_tokens` encoded as u64 LE followed by the
// independent oracle output tensor encoded as concatenated `f32::to_le_bytes()`.
const DIRECT_BOUNDARY_BLAKE3: &str =
    "9ff23303286a2301bc4ca4367c4429b8913d0323bc97d193dfc3281088a9b5db";

const DIRECT_BOUNDARY_LENGTHS: [usize; 8] = [1, 2, 15, 16, 17, 31, 32, 33];
const PINNED_QUERY_HEADS: usize = 16;
const PINNED_KEY_VALUE_HEADS: usize = 2;
const PINNED_HEAD_DIM: usize = 128;
// The pinned decode-layer wrapper is intentionally omitted from this compact
// contract increment. Its behavioral coverage moves to the next official
// fixture-backed decode test before production implementation lands.

#[derive(Clone, Copy, Debug)]
enum DirectOperand {
    Query,
    Keys,
    Values,
}

#[derive(Clone, Copy, Debug)]
enum LayerOperand {
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

const LAYER_OPERANDS: [LayerOperand; 12] = [
    LayerOperand::Input,
    LayerOperand::RawCos,
    LayerOperand::RawSin,
    LayerOperand::InputNormWeight,
    LayerOperand::QueryWeight,
    LayerOperand::KeyWeight,
    LayerOperand::ValueWeight,
    LayerOperand::AttentionOutputWeight,
    LayerOperand::PostAttentionNormWeight,
    LayerOperand::GateWeight,
    LayerOperand::UpWeight,
    LayerOperand::DownWeight,
];

impl DirectOperand {
    const fn label(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Keys => "keys",
            Self::Values => "values",
        }
    }
}

impl LayerOperand {
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
struct DirectDecodeFixture {
    query: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
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

#[derive(Clone, Debug, PartialEq)]
struct LayerCaseInputs {
    input: Vec<f32>,
    raw_cos: Vec<f32>,
    raw_sin: Vec<f32>,
    parameters: OwnedParameters,
}

#[derive(Clone, Copy, Debug)]
struct LayerLengths {
    input: usize,
    raw_table: usize,
    query_weight: usize,
    key_weight: usize,
    value_weight: usize,
    attention_output_weight: usize,
    gate_weight: usize,
    up_weight: usize,
    down_weight: usize,
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

    fn operand_mut(&mut self, operand: LayerOperand) -> &mut Vec<f32> {
        match operand {
            LayerOperand::InputNormWeight => &mut self.input_norm_weight,
            LayerOperand::QueryWeight => &mut self.query_weight,
            LayerOperand::KeyWeight => &mut self.key_weight,
            LayerOperand::ValueWeight => &mut self.value_weight,
            LayerOperand::AttentionOutputWeight => &mut self.attention_output_weight,
            LayerOperand::PostAttentionNormWeight => &mut self.post_attention_norm_weight,
            LayerOperand::GateWeight => &mut self.gate_weight,
            LayerOperand::UpWeight => &mut self.up_weight,
            LayerOperand::DownWeight => &mut self.down_weight,
            LayerOperand::Input | LayerOperand::RawCos | LayerOperand::RawSin => {
                panic!("{} is not a parameter weight", operand.label())
            }
        }
    }
}

fn layer_operand_mut<'a>(
    operand: LayerOperand,
    input: &'a mut Vec<f32>,
    raw_cos: &'a mut Vec<f32>,
    raw_sin: &'a mut Vec<f32>,
    parameters: &'a mut OwnedParameters,
) -> &'a mut Vec<f32> {
    match operand {
        LayerOperand::Input => input,
        LayerOperand::RawCos => raw_cos,
        LayerOperand::RawSin => raw_sin,
        weight => parameters.operand_mut(weight),
    }
}

fn layer_config(tokens: usize) -> DecoderLayerConfig {
    DecoderLayerConfig {
        tokens,
        hidden_size: HIDDEN_SIZE,
        intermediate_size: INTERMEDIATE_SIZE,
        query_heads: QUERY_HEADS,
        key_value_heads: KEY_VALUE_HEADS,
        head_dim: HEAD_DIM,
        rms_norm_epsilon: RMS_NORM_EPSILON,
        mrope_sections: MROPE_SECTIONS,
    }
}

fn dense(len: usize, mul: usize, add: usize, modulus: usize, divisor: f32) -> Vec<f32> {
    (0..len)
        .map(|index| (((index * mul + add) % modulus) as f32 - (modulus / 2) as f32) / divisor)
        .collect()
}

fn layer_parameters() -> OwnedParameters {
    OwnedParameters {
        input_norm_weight: (0..HIDDEN_SIZE)
            .map(|index| 0.75 + ((index * 5 + 2) % 11) as f32 / 23.0)
            .collect(),
        query_weight: dense(QUERY_HEADS * HEAD_DIM * HIDDEN_SIZE, 7, 3, 29, 19.0),
        key_weight: dense(KEY_VALUE_HEADS * HEAD_DIM * HIDDEN_SIZE, 11, 5, 31, 23.0),
        value_weight: dense(KEY_VALUE_HEADS * HEAD_DIM * HIDDEN_SIZE, 13, 7, 37, 29.0),
        attention_output_weight: dense(HIDDEN_SIZE * QUERY_HEADS * HEAD_DIM, 17, 2, 41, 31.0),
        post_attention_norm_weight: (0..HIDDEN_SIZE)
            .map(|index| 0.7 + ((index * 3 + 1) % 13) as f32 / 29.0)
            .collect(),
        gate_weight: dense(INTERMEDIATE_SIZE * HIDDEN_SIZE, 19, 1, 43, 37.0),
        up_weight: dense(INTERMEDIATE_SIZE * HIDDEN_SIZE, 23, 4, 47, 41.0),
        down_weight: dense(HIDDEN_SIZE * INTERMEDIATE_SIZE, 29, 6, 53, 43.0),
    }
}

fn layer_input(tokens: usize) -> Vec<f32> {
    dense(tokens * HIDDEN_SIZE, 17, 3, 29, 11.0)
}

fn layer_raw_tables(tokens: usize) -> (Vec<f32>, Vec<f32>) {
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

fn layer_lengths(config: DecoderLayerConfig) -> Option<LayerLengths> {
    let query_width = config.query_heads.checked_mul(config.head_dim)?;
    let key_value_width = config.key_value_heads.checked_mul(config.head_dim)?;
    Some(LayerLengths {
        input: config.tokens.checked_mul(config.hidden_size)?,
        raw_table: config.tokens.checked_mul(config.head_dim)?.checked_mul(3)?,
        query_weight: query_width.checked_mul(config.hidden_size)?,
        key_weight: key_value_width.checked_mul(config.hidden_size)?,
        value_weight: key_value_width.checked_mul(config.hidden_size)?,
        attention_output_weight: config.hidden_size.checked_mul(query_width)?,
        gate_weight: config.intermediate_size.checked_mul(config.hidden_size)?,
        up_weight: config.intermediate_size.checked_mul(config.hidden_size)?,
        down_weight: config.hidden_size.checked_mul(config.intermediate_size)?,
    })
}

fn generic_layer_case_inputs(config: DecoderLayerConfig) -> Option<LayerCaseInputs> {
    let lengths = layer_lengths(config)?;
    let mut raw_cos = Vec::with_capacity(lengths.raw_table);
    let mut raw_sin = Vec::with_capacity(lengths.raw_table);
    for axis in 0..3 {
        for token in 0..config.tokens {
            for dim in 0..config.head_dim {
                raw_cos.push(0.55 + axis as f32 * 0.13 + token as f32 * 0.02 + dim as f32 * 0.007);
                raw_sin.push(-0.3 - axis as f32 * 0.11 + token as f32 * 0.015 - dim as f32 * 0.005);
            }
        }
    }
    Some(LayerCaseInputs {
        input: dense(lengths.input, 17, 3, 29, 11.0),
        raw_cos,
        raw_sin,
        parameters: OwnedParameters {
            input_norm_weight: (0..config.hidden_size)
                .map(|index| 0.75 + ((index * 5 + 2) % 11) as f32 / 23.0)
                .collect(),
            query_weight: dense(lengths.query_weight, 7, 3, 29, 19.0),
            key_weight: dense(lengths.key_weight, 11, 5, 31, 23.0),
            value_weight: dense(lengths.value_weight, 13, 7, 37, 29.0),
            attention_output_weight: dense(lengths.attention_output_weight, 17, 2, 41, 31.0),
            post_attention_norm_weight: (0..config.hidden_size)
                .map(|index| 0.7 + ((index * 3 + 1) % 13) as f32 / 29.0)
                .collect(),
            gate_weight: dense(lengths.gate_weight, 19, 1, 43, 37.0),
            up_weight: dense(lengths.up_weight, 23, 4, 47, 41.0),
            down_weight: dense(lengths.down_weight, 29, 6, 53, 43.0),
        },
    })
}

fn direct_fixture(
    cache_tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> DirectDecodeFixture {
    DirectDecodeFixture {
        query: (0..query_heads * head_dim)
            .map(|index| (((index * 17 + cache_tokens * 5 + 3) % 37) as f32 - 18.0) / 11.0)
            .collect(),
        keys: (0..cache_tokens * key_value_heads * head_dim)
            .map(|index| (((index * 13 + cache_tokens * 7 + 5) % 31) as f32 - 15.0) / 9.0)
            .collect(),
        values: (0..cache_tokens * key_value_heads * head_dim)
            .map(|index| (((index * 19 + cache_tokens * 11 + 7) % 41) as f32 - 20.0) / 7.0)
            .collect(),
    }
}

fn query_index(head: usize, dim: usize, head_dim: usize) -> usize {
    head * head_dim + dim
}

fn cache_index(token: usize, head: usize, dim: usize, heads: usize, head_dim: usize) -> usize {
    (token * heads + head) * head_dim + dim
}

fn repeat_kv_heads_contiguously(
    input: &[f32],
    cache_tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let group = query_heads / key_value_heads;
    let mut repeated = Vec::with_capacity(cache_tokens * query_heads * head_dim);
    for token in 0..cache_tokens {
        for kv_head in 0..key_value_heads {
            let start = cache_index(token, kv_head, 0, key_value_heads, head_dim);
            for _ in 0..group {
                repeated.extend_from_slice(&input[start..start + head_dim]);
            }
        }
    }
    repeated
}

fn repeat_kv_heads_with_rotated_groups(
    input: &[f32],
    cache_tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let group = query_heads / key_value_heads;
    let mut repeated = Vec::with_capacity(cache_tokens * query_heads * head_dim);
    for token in 0..cache_tokens {
        for kv_head in 0..key_value_heads {
            let wrong_head = (kv_head + 1) % key_value_heads;
            let start = cache_index(token, wrong_head, 0, key_value_heads, head_dim);
            for _ in 0..group {
                repeated.extend_from_slice(&input[start..start + head_dim]);
            }
        }
    }
    repeated
}

fn direct_decode_from_repeated_kv(
    query: &[f32],
    repeated_keys: &[f32],
    repeated_values: &[f32],
    cache_tokens: usize,
    query_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; query.len()];
    for head in 0..query_heads {
        let mut logits = Vec::with_capacity(cache_tokens);
        for token in 0..cache_tokens {
            let mut dot = 0.0_f32;
            for dim in 0..head_dim {
                dot += query[query_index(head, dim, head_dim)]
                    * repeated_keys[cache_index(token, head, dim, query_heads, head_dim)];
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
            for (token, probability) in probabilities.iter().copied().enumerate() {
                weighted += probability
                    * repeated_values[cache_index(token, head, dim, query_heads, head_dim)];
            }
            output[query_index(head, dim, head_dim)] = weighted;
        }
    }
    output
}

fn independent_decode_gqa(
    fixture: &DirectDecodeFixture,
    cache_tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let repeated_keys = repeat_kv_heads_contiguously(
        &fixture.keys,
        cache_tokens,
        query_heads,
        key_value_heads,
        head_dim,
    );
    let repeated_values = repeat_kv_heads_contiguously(
        &fixture.values,
        cache_tokens,
        query_heads,
        key_value_heads,
        head_dim,
    );
    direct_decode_from_repeated_kv(
        &fixture.query,
        &repeated_keys,
        &repeated_values,
        cache_tokens,
        query_heads,
        head_dim,
        (head_dim as f32).sqrt().recip(),
    )
}

fn missing_scale_decode_gqa(
    fixture: &DirectDecodeFixture,
    cache_tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let repeated_keys = repeat_kv_heads_contiguously(
        &fixture.keys,
        cache_tokens,
        query_heads,
        key_value_heads,
        head_dim,
    );
    let repeated_values = repeat_kv_heads_contiguously(
        &fixture.values,
        cache_tokens,
        query_heads,
        key_value_heads,
        head_dim,
    );
    direct_decode_from_repeated_kv(
        &fixture.query,
        &repeated_keys,
        &repeated_values,
        cache_tokens,
        query_heads,
        head_dim,
        1.0,
    )
}

fn wrong_grouping_decode_gqa(
    fixture: &DirectDecodeFixture,
    cache_tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let repeated_keys = repeat_kv_heads_with_rotated_groups(
        &fixture.keys,
        cache_tokens,
        query_heads,
        key_value_heads,
        head_dim,
    );
    let repeated_values = repeat_kv_heads_with_rotated_groups(
        &fixture.values,
        cache_tokens,
        query_heads,
        key_value_heads,
        head_dim,
    );
    direct_decode_from_repeated_kv(
        &fixture.query,
        &repeated_keys,
        &repeated_values,
        cache_tokens,
        query_heads,
        head_dim,
        (head_dim as f32).sqrt().recip(),
    )
}

fn bounded_extreme_fixture() -> DirectDecodeFixture {
    DirectDecodeFixture {
        query: vec![1_000.0, -1_000.0],
        keys: vec![100.0, 99.0, -100.0],
        values: vec![1.0, 2.0, 3.0],
    }
}

fn invoke_direct_decode(
    fixture: &DirectDecodeFixture,
    cache_tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, CpuRefError> {
    decode_gqa_f32(
        &fixture.query,
        &fixture.keys,
        &fixture.values,
        cache_tokens,
        query_heads,
        key_value_heads,
        head_dim,
    )
}

fn direct_boundary_digest() -> String {
    let mut hasher = blake3::Hasher::new();
    for cache_tokens in DIRECT_BOUNDARY_LENGTHS {
        let fixture = direct_fixture(cache_tokens, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
        hasher.update(&(cache_tokens as u64).to_le_bytes());
        update_f32_digest(
            &mut hasher,
            &independent_decode_gqa(
                &fixture,
                cache_tokens,
                QUERY_HEADS,
                KEY_VALUE_HEADS,
                HEAD_DIM,
            ),
        );
    }
    hasher.finalize().to_hex().to_string()
}

fn update_f32_digest(hasher: &mut blake3::Hasher, values: &[f32]) {
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn assert_f32_bits(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label}[{index}] actual={actual:?} expected={expected:?}"
        );
    }
}

fn assert_cache_exact(
    label: &str,
    actual: &DecoderPrefillKvCache,
    expected: &DecoderPrefillKvCache,
) {
    assert_eq!(actual.tokens, expected.tokens, "{label} tokens");
    assert_eq!(
        actual.key_value_heads, expected.key_value_heads,
        "{label} KV heads"
    );
    assert_eq!(actual.head_dim, expected.head_dim, "{label} head dim");
    assert_f32_bits(&format!("{label} keys"), &actual.keys, &expected.keys);
    assert_f32_bits(&format!("{label} values"), &actual.values, &expected.values);
}

fn slice_row(values: &[f32], row: usize, width: usize) -> Vec<f32> {
    values[row * width..(row + 1) * width].to_vec()
}

fn slice_last_row(values: &[f32], tokens: usize, width: usize) -> Vec<f32> {
    slice_row(values, tokens - 1, width)
}

fn slice_last_raw_axis_major_row(values: &[f32], tokens: usize, head_dim: usize) -> Vec<f32> {
    let mut row = Vec::with_capacity(3 * head_dim);
    for axis in 0..3 {
        let start = (axis * tokens + (tokens - 1)) * head_dim;
        row.extend_from_slice(&values[start..start + head_dim]);
    }
    row
}

fn slice_prefix_raw_axis_major(
    values: &[f32],
    tokens: usize,
    prefix_tokens: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut prefix = Vec::with_capacity(3 * prefix_tokens * head_dim);
    for axis in 0..3 {
        let start = axis * tokens * head_dim;
        let end = start + prefix_tokens * head_dim;
        prefix.extend_from_slice(&values[start..end]);
    }
    prefix
}

fn assert_decode_trace_matches_last_prefill_row(
    actual: &DecoderLayerDecodeTrace,
    expected: &DecoderLayerPrefillTrace,
    full_tokens: usize,
) {
    let query_width = QUERY_HEADS * HEAD_DIM;
    let key_value_width = KEY_VALUE_HEADS * HEAD_DIM;
    let intermediate_width = INTERMEDIATE_SIZE;

    let fields: [(&str, &[f32], Vec<f32>); 15] = [
        (
            "norm1",
            &actual.norm1,
            slice_last_row(&expected.norm1, full_tokens, HIDDEN_SIZE),
        ),
        (
            "query",
            &actual.query,
            slice_last_row(&expected.query, full_tokens, query_width),
        ),
        (
            "key",
            &actual.key,
            slice_last_row(&expected.key, full_tokens, key_value_width),
        ),
        (
            "value",
            &actual.value,
            slice_last_row(&expected.value, full_tokens, key_value_width),
        ),
        (
            "mrope_query",
            &actual.mrope_query,
            slice_last_row(&expected.mrope_query, full_tokens, query_width),
        ),
        (
            "mrope_key",
            &actual.mrope_key,
            slice_last_row(&expected.mrope_key, full_tokens, key_value_width),
        ),
        (
            "attention_context",
            &actual.attention_context,
            slice_last_row(&expected.attention_context, full_tokens, query_width),
        ),
        (
            "attention_output",
            &actual.attention_output,
            slice_last_row(&expected.attention_output, full_tokens, HIDDEN_SIZE),
        ),
        (
            "attention_residual",
            &actual.attention_residual,
            slice_last_row(&expected.attention_residual, full_tokens, HIDDEN_SIZE),
        ),
        (
            "norm2",
            &actual.norm2,
            slice_last_row(&expected.norm2, full_tokens, HIDDEN_SIZE),
        ),
        (
            "mlp_gate",
            &actual.mlp_gate,
            slice_last_row(&expected.mlp_gate, full_tokens, intermediate_width),
        ),
        (
            "mlp_up",
            &actual.mlp_up,
            slice_last_row(&expected.mlp_up, full_tokens, intermediate_width),
        ),
        (
            "mlp_activation",
            &actual.mlp_activation,
            slice_last_row(&expected.mlp_activation, full_tokens, intermediate_width),
        ),
        (
            "mlp_down",
            &actual.mlp_down,
            slice_last_row(&expected.mlp_down, full_tokens, HIDDEN_SIZE),
        ),
        (
            "output",
            &actual.output,
            slice_last_row(&expected.output, full_tokens, HIDDEN_SIZE),
        ),
    ];
    for (label, actual, expected) in fields {
        assert_f32_bits(label, actual, &expected);
    }
}

fn assert_direct_error<T>(result: Result<T, CpuRefError>, expected: CpuRefErrorCode, case: &str) {
    match result {
        Err(error) => assert_eq!(error.code(), expected, "{case}"),
        Ok(_) => panic!("{case}: expected {expected:?}"),
    }
}

fn assert_layer_error<T>(result: Result<T, CpuRefError>, expected: CpuRefErrorCode, case: &str) {
    match result {
        Err(error) => assert_eq!(error.code(), expected, "{case}"),
        Ok(_) => panic!("{case}: expected {expected:?}"),
    }
}

#[test]
fn direct_decode_matches_independent_boundary_oracle_and_fixed_digest() {
    assert_eq!(direct_boundary_digest(), DIRECT_BOUNDARY_BLAKE3);

    for cache_tokens in DIRECT_BOUNDARY_LENGTHS {
        let fixture = direct_fixture(cache_tokens, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
        let preserved = fixture.clone();
        let expected = independent_decode_gqa(
            &fixture,
            cache_tokens,
            QUERY_HEADS,
            KEY_VALUE_HEADS,
            HEAD_DIM,
        );
        let actual = invoke_direct_decode(
            &fixture,
            cache_tokens,
            QUERY_HEADS,
            KEY_VALUE_HEADS,
            HEAD_DIM,
        )
        .unwrap();
        assert_f32_bits(&format!("cache_tokens={cache_tokens}"), &actual, &expected);
        assert_eq!(fixture, preserved, "cache_tokens={cache_tokens}");
    }
}

#[test]
fn direct_decode_supports_group_size_one_and_single_kv_head() {
    for (cache_tokens, query_heads, key_value_heads, head_dim) in [(7, 2, 2, 5), (9, 4, 1, 3)] {
        let fixture = direct_fixture(cache_tokens, query_heads, key_value_heads, head_dim);
        let expected = independent_decode_gqa(
            &fixture,
            cache_tokens,
            query_heads,
            key_value_heads,
            head_dim,
        );
        let actual = invoke_direct_decode(
            &fixture,
            cache_tokens,
            query_heads,
            key_value_heads,
            head_dim,
        )
        .unwrap();
        assert_f32_bits(
            &format!("cache_tokens={cache_tokens} qh={query_heads} kvh={key_value_heads}"),
            &actual,
            &expected,
        );
    }
}

#[test]
fn direct_decode_oracle_fixture_is_sensitive_to_wrong_grouping_missing_scale_and_truncated_prefix()
{
    let cache_tokens = 33;
    let fixture = direct_fixture(cache_tokens, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
    let expected = independent_decode_gqa(
        &fixture,
        cache_tokens,
        QUERY_HEADS,
        KEY_VALUE_HEADS,
        HEAD_DIM,
    );
    let wrong_grouping = wrong_grouping_decode_gqa(
        &fixture,
        cache_tokens,
        QUERY_HEADS,
        KEY_VALUE_HEADS,
        HEAD_DIM,
    );
    let missing_scale = missing_scale_decode_gqa(
        &fixture,
        cache_tokens,
        QUERY_HEADS,
        KEY_VALUE_HEADS,
        HEAD_DIM,
    );
    let truncated = DirectDecodeFixture {
        query: fixture.query.clone(),
        keys: fixture.keys[KEY_VALUE_HEADS * HEAD_DIM..].to_vec(),
        values: fixture.values[KEY_VALUE_HEADS * HEAD_DIM..].to_vec(),
    };
    let truncated_expected = independent_decode_gqa(
        &truncated,
        cache_tokens - 1,
        QUERY_HEADS,
        KEY_VALUE_HEADS,
        HEAD_DIM,
    );
    assert_ne!(bits(&wrong_grouping), bits(&expected));
    assert_ne!(bits(&missing_scale), bits(&expected));
    assert_ne!(bits(&truncated_expected), bits(&expected));
}

#[test]
fn pinned_direct_decode_matches_practical_boundary_oracle_and_preserves_inputs() {
    for cache_tokens in [1, 17, 33] {
        let fixture = direct_fixture(
            cache_tokens,
            PINNED_QUERY_HEADS,
            PINNED_KEY_VALUE_HEADS,
            PINNED_HEAD_DIM,
        );
        let preserved = fixture.clone();
        let expected = independent_decode_gqa(
            &fixture,
            cache_tokens,
            PINNED_QUERY_HEADS,
            PINNED_KEY_VALUE_HEADS,
            PINNED_HEAD_DIM,
        );
        let actual = pinned_decoder_decode_gqa_f32(
            &fixture.query,
            &fixture.keys,
            &fixture.values,
            cache_tokens,
        )
        .unwrap();
        assert_f32_bits(
            &format!("pinned cache_tokens={cache_tokens}"),
            &actual,
            &expected,
        );
        assert_eq!(fixture, preserved, "pinned cache_tokens={cache_tokens}");
    }
}

#[test]
fn direct_decode_matches_bounded_extreme_oracle_without_overflow() {
    let cache_tokens = 3;
    let query_heads = 2;
    let key_value_heads = 1;
    let head_dim = 1;
    let fixture = bounded_extreme_fixture();
    let repeated_keys = repeat_kv_heads_contiguously(
        &fixture.keys,
        cache_tokens,
        query_heads,
        key_value_heads,
        head_dim,
    );
    let mut saw_naive_exp_overflow = false;
    for head in 0..query_heads {
        for token in 0..cache_tokens {
            let logit = fixture.query[query_index(head, 0, head_dim)]
                * repeated_keys[cache_index(token, head, 0, query_heads, head_dim)];
            saw_naive_exp_overflow |= logit.exp().is_infinite();
        }
    }
    assert!(saw_naive_exp_overflow);

    let expected = independent_decode_gqa(
        &fixture,
        cache_tokens,
        query_heads,
        key_value_heads,
        head_dim,
    );
    assert!(expected.iter().all(|value| value.is_finite()));
    assert_f32_bits("bounded extreme oracle", &expected, &[1.0, 3.0]);

    let actual = invoke_direct_decode(
        &fixture,
        cache_tokens,
        query_heads,
        key_value_heads,
        head_dim,
    )
    .unwrap();
    assert_f32_bits("bounded extreme actual", &actual, &expected);
    assert!(actual.iter().all(|value| value.is_finite()));
}

#[test]
fn compact_decode_layer_matches_full_recompute_last_row_and_complete_cache() {
    let prefix_tokens = 4;
    let full_tokens = prefix_tokens + 1;
    let parameters = layer_parameters();
    let full_input = layer_input(full_tokens);
    let prefix_input = full_input[..prefix_tokens * HIDDEN_SIZE].to_vec();
    let (full_raw_cos, full_raw_sin) = layer_raw_tables(full_tokens);
    let prefix_raw_cos =
        slice_prefix_raw_axis_major(&full_raw_cos, full_tokens, prefix_tokens, HEAD_DIM);
    let prefix_raw_sin =
        slice_prefix_raw_axis_major(&full_raw_sin, full_tokens, prefix_tokens, HEAD_DIM);
    let decode_input = slice_last_row(&full_input, full_tokens, HIDDEN_SIZE);
    let decode_raw_cos = slice_last_raw_axis_major_row(&full_raw_cos, full_tokens, HEAD_DIM);
    let decode_raw_sin = slice_last_raw_axis_major_row(&full_raw_sin, full_tokens, HEAD_DIM);
    let full_trace = decoder_layer_prefill_f32(
        &full_input,
        layer_config(full_tokens),
        &full_raw_cos,
        &full_raw_sin,
        parameters.borrowed(),
    )
    .unwrap();
    let prefix_trace = decoder_layer_prefill_f32(
        &prefix_input,
        layer_config(prefix_tokens),
        &prefix_raw_cos,
        &prefix_raw_sin,
        parameters.borrowed(),
    )
    .unwrap();
    let preserved = (
        decode_input.clone(),
        decode_raw_cos.clone(),
        decode_raw_sin.clone(),
        prefix_trace.kv_cache.clone(),
        parameters.clone(),
    );
    let mut actual = decoder_layer_decode_f32(
        &decode_input,
        layer_config(1),
        &decode_raw_cos,
        &decode_raw_sin,
        &prefix_trace.kv_cache,
        parameters.borrowed(),
    )
    .unwrap();

    assert_decode_trace_matches_last_prefill_row(&actual, &full_trace, full_tokens);
    assert_cache_exact(
        "decode output cache",
        &actual.kv_cache,
        &full_trace.kv_cache,
    );
    assert_cache_exact(
        "prefix cache remains exact",
        &prefix_trace.kv_cache,
        &preserved.3,
    );
    assert_eq!(decode_input, preserved.0);
    assert_eq!(decode_raw_cos, preserved.1);
    assert_eq!(decode_raw_sin, preserved.2);
    assert_eq!(parameters, preserved.4);

    let cached_key_row = slice_last_row(
        &actual.kv_cache.keys,
        full_tokens,
        KEY_VALUE_HEADS * HEAD_DIM,
    );
    let cached_value_row = slice_last_row(
        &actual.kv_cache.values,
        full_tokens,
        KEY_VALUE_HEADS * HEAD_DIM,
    );
    assert_f32_bits(
        "cached appended rotated key",
        &cached_key_row,
        &actual.mrope_key,
    );
    assert_f32_bits(
        "cached appended raw value",
        &cached_value_row,
        &actual.value,
    );

    let preserved_prefix = prefix_trace.kv_cache.clone();
    let preserved_key = actual.mrope_key.clone();
    let preserved_value = actual.value.clone();
    actual.kv_cache.keys[0] = 123.5;
    actual.kv_cache.values[1] = -456.25;
    assert_cache_exact(
        "prefix cache detached from decode output cache",
        &prefix_trace.kv_cache,
        &preserved_prefix,
    );
    assert_f32_bits(
        "decode key detached from output cache",
        &actual.mrope_key,
        &preserved_key,
    );
    assert_f32_bits(
        "decode value detached from output cache",
        &actual.value,
        &preserved_value,
    );
}

#[test]
fn compact_decode_layer_matches_full_recompute_cache_and_output_across_prefix_boundaries() {
    let parameters = layer_parameters();
    for prefix_tokens in [1, 15, 16, 17, 31, 32, 33] {
        let full_tokens = prefix_tokens + 1;
        let full_input = layer_input(full_tokens);
        let prefix_input = full_input[..prefix_tokens * HIDDEN_SIZE].to_vec();
        let (full_raw_cos, full_raw_sin) = layer_raw_tables(full_tokens);
        let prefix_raw_cos =
            slice_prefix_raw_axis_major(&full_raw_cos, full_tokens, prefix_tokens, HEAD_DIM);
        let prefix_raw_sin =
            slice_prefix_raw_axis_major(&full_raw_sin, full_tokens, prefix_tokens, HEAD_DIM);
        let decode_input = slice_last_row(&full_input, full_tokens, HIDDEN_SIZE);
        let decode_raw_cos = slice_last_raw_axis_major_row(&full_raw_cos, full_tokens, HEAD_DIM);
        let decode_raw_sin = slice_last_raw_axis_major_row(&full_raw_sin, full_tokens, HEAD_DIM);
        let full_trace = decoder_layer_prefill_f32(
            &full_input,
            layer_config(full_tokens),
            &full_raw_cos,
            &full_raw_sin,
            parameters.borrowed(),
        )
        .unwrap();
        let prefix_trace = decoder_layer_prefill_f32(
            &prefix_input,
            layer_config(prefix_tokens),
            &prefix_raw_cos,
            &prefix_raw_sin,
            parameters.borrowed(),
        )
        .unwrap();
        let actual = decoder_layer_decode_f32(
            &decode_input,
            layer_config(1),
            &decode_raw_cos,
            &decode_raw_sin,
            &prefix_trace.kv_cache,
            parameters.borrowed(),
        )
        .unwrap();
        assert_f32_bits(
            &format!("prefix_tokens={prefix_tokens} output"),
            &actual.output,
            &slice_last_row(&full_trace.output, full_tokens, HIDDEN_SIZE),
        );
        assert_cache_exact(
            &format!("prefix_tokens={prefix_tokens} cache"),
            &actual.kv_cache,
            &full_trace.kv_cache,
        );
    }
}

#[test]
fn decode_layer_rejects_config_tokens_not_equal_one_with_shape_consistent_operands() {
    let prefix_tokens = 3;
    let parameters = layer_parameters();
    let prefix_input = layer_input(prefix_tokens);
    let (prefix_raw_cos, prefix_raw_sin) = layer_raw_tables(prefix_tokens);
    let prefix_trace = decoder_layer_prefill_f32(
        &prefix_input,
        layer_config(prefix_tokens),
        &prefix_raw_cos,
        &prefix_raw_sin,
        parameters.borrowed(),
    )
    .unwrap();
    for tokens in [0, 2] {
        let config = layer_config(tokens);
        let case = generic_layer_case_inputs(config).unwrap();
        assert_layer_error(
            decoder_layer_decode_f32(
                &case.input,
                config,
                &case.raw_cos,
                &case.raw_sin,
                &prefix_trace.kv_cache,
                case.parameters.borrowed(),
            ),
            CpuRefErrorCode::DimensionMismatch,
            &format!("config tokens={tokens}"),
        );
    }
}

#[test]
fn decode_layer_rejects_malformed_config_lengths_nonfinite_cache_and_nonfinite_operands() {
    let prefix_tokens = 3;
    let parameters = layer_parameters();
    let prefix_input = layer_input(prefix_tokens);
    let (prefix_raw_cos, prefix_raw_sin) = layer_raw_tables(prefix_tokens);
    let prefix_trace = decoder_layer_prefill_f32(
        &prefix_input,
        layer_config(prefix_tokens),
        &prefix_raw_cos,
        &prefix_raw_sin,
        parameters.borrowed(),
    )
    .unwrap();
    let base_config = layer_config(1);
    let base_case = generic_layer_case_inputs(base_config).unwrap();

    for operand in LAYER_OPERANDS {
        for long in [false, true] {
            let mut input = base_case.input.clone();
            let mut raw_cos = base_case.raw_cos.clone();
            let mut raw_sin = base_case.raw_sin.clone();
            let mut parameters = base_case.parameters.clone();
            let values = layer_operand_mut(
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
            assert_layer_error(
                decoder_layer_decode_f32(
                    &input,
                    base_config,
                    &raw_cos,
                    &raw_sin,
                    &prefix_trace.kv_cache,
                    parameters.borrowed(),
                ),
                CpuRefErrorCode::DimensionMismatch,
                &format!(
                    "{} {}",
                    operand.label(),
                    if long { "long" } else { "short" }
                ),
            );
        }
    }

    let cache_width = KEY_VALUE_HEADS * HEAD_DIM;
    let malformed_caches = [
        (
            "cache keys short",
            DecoderPrefillKvCache {
                keys: prefix_trace.kv_cache.keys[..prefix_trace.kv_cache.keys.len() - 1].to_vec(),
                values: prefix_trace.kv_cache.values.clone(),
                tokens: prefix_trace.kv_cache.tokens,
                key_value_heads: KEY_VALUE_HEADS,
                head_dim: HEAD_DIM,
            },
        ),
        (
            "cache keys long",
            DecoderPrefillKvCache {
                keys: {
                    let mut values = prefix_trace.kv_cache.keys.clone();
                    values.push(0.0);
                    values
                },
                values: prefix_trace.kv_cache.values.clone(),
                tokens: prefix_trace.kv_cache.tokens,
                key_value_heads: KEY_VALUE_HEADS,
                head_dim: HEAD_DIM,
            },
        ),
        (
            "cache values short",
            DecoderPrefillKvCache {
                keys: prefix_trace.kv_cache.keys.clone(),
                values: prefix_trace.kv_cache.values[..prefix_trace.kv_cache.values.len() - 1]
                    .to_vec(),
                tokens: prefix_trace.kv_cache.tokens,
                key_value_heads: KEY_VALUE_HEADS,
                head_dim: HEAD_DIM,
            },
        ),
        (
            "cache values long",
            DecoderPrefillKvCache {
                keys: prefix_trace.kv_cache.keys.clone(),
                values: {
                    let mut values = prefix_trace.kv_cache.values.clone();
                    values.push(0.0);
                    values
                },
                tokens: prefix_trace.kv_cache.tokens,
                key_value_heads: KEY_VALUE_HEADS,
                head_dim: HEAD_DIM,
            },
        ),
        (
            "cache metadata token mismatch smaller",
            DecoderPrefillKvCache {
                keys: prefix_trace.kv_cache.keys.clone(),
                values: prefix_trace.kv_cache.values.clone(),
                tokens: prefix_tokens - 1,
                key_value_heads: KEY_VALUE_HEADS,
                head_dim: HEAD_DIM,
            },
        ),
        (
            "cache metadata token mismatch larger",
            DecoderPrefillKvCache {
                keys: prefix_trace.kv_cache.keys.clone(),
                values: prefix_trace.kv_cache.values.clone(),
                tokens: prefix_tokens + 1,
                key_value_heads: KEY_VALUE_HEADS,
                head_dim: HEAD_DIM,
            },
        ),
        (
            "cache metadata key value heads mismatch",
            DecoderPrefillKvCache {
                keys: prefix_trace.kv_cache.keys.clone(),
                values: prefix_trace.kv_cache.values.clone(),
                tokens: prefix_tokens,
                key_value_heads: 1,
                head_dim: HEAD_DIM,
            },
        ),
        (
            "cache metadata head dim mismatch",
            DecoderPrefillKvCache {
                keys: prefix_trace.kv_cache.keys.clone(),
                values: prefix_trace.kv_cache.values.clone(),
                tokens: prefix_tokens,
                key_value_heads: KEY_VALUE_HEADS,
                head_dim: HEAD_DIM + 2,
            },
        ),
        (
            "cache tokens zero",
            DecoderPrefillKvCache {
                keys: Vec::new(),
                values: Vec::new(),
                tokens: 0,
                key_value_heads: KEY_VALUE_HEADS,
                head_dim: HEAD_DIM,
            },
        ),
        (
            "cache tokens overflow",
            DecoderPrefillKvCache {
                keys: Vec::new(),
                values: Vec::new(),
                tokens: usize::MAX,
                key_value_heads: KEY_VALUE_HEADS,
                head_dim: HEAD_DIM,
            },
        ),
    ];
    for (case, cache) in malformed_caches {
        assert_layer_error(
            decoder_layer_decode_f32(
                &base_case.input,
                base_config,
                &base_case.raw_cos,
                &base_case.raw_sin,
                &cache,
                base_case.parameters.borrowed(),
            ),
            CpuRefErrorCode::DimensionMismatch,
            case,
        );
    }

    let finite_invalid_configs = [
        (
            "zero hidden_size",
            DecoderLayerConfig {
                hidden_size: 0,
                ..base_config
            },
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "zero intermediate_size",
            DecoderLayerConfig {
                intermediate_size: 0,
                ..base_config
            },
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "zero query_heads",
            DecoderLayerConfig {
                query_heads: 0,
                ..base_config
            },
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "zero key_value_heads",
            DecoderLayerConfig {
                key_value_heads: 0,
                ..base_config
            },
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "zero head_dim",
            DecoderLayerConfig {
                head_dim: 0,
                ..base_config
            },
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "query heads not divisible by key value heads",
            DecoderLayerConfig {
                query_heads: 3,
                key_value_heads: 2,
                head_dim: 6,
                mrope_sections: [1, 1, 1],
                ..base_config
            },
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "zero M-RoPE section",
            DecoderLayerConfig {
                mrope_sections: [1, 0, 2],
                head_dim: 6,
                ..base_config
            },
            CpuRefErrorCode::DimensionMismatch,
        ),
        (
            "M-RoPE section sum mismatch",
            DecoderLayerConfig {
                mrope_sections: [1, 1, 2],
                head_dim: 6,
                ..base_config
            },
            CpuRefErrorCode::DimensionMismatch,
        ),
    ];
    for (case, config, expected) in finite_invalid_configs {
        let inputs = generic_layer_case_inputs(config).unwrap();
        assert_layer_error(
            decoder_layer_decode_f32(
                &inputs.input,
                config,
                &inputs.raw_cos,
                &inputs.raw_sin,
                &prefix_trace.kv_cache,
                inputs.parameters.borrowed(),
            ),
            expected,
            case,
        );
    }

    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let config = DecoderLayerConfig {
            rms_norm_epsilon: epsilon,
            ..base_config
        };
        let inputs = generic_layer_case_inputs(config).unwrap();
        assert_layer_error(
            decoder_layer_decode_f32(
                &inputs.input,
                config,
                &inputs.raw_cos,
                &inputs.raw_sin,
                &prefix_trace.kv_cache,
                inputs.parameters.borrowed(),
            ),
            CpuRefErrorCode::NonPositiveEpsilon,
            &format!("invalid epsilon {epsilon:?}"),
        );
    }

    // These configurations only exist to trip checked-multiplication guards; a
    // shape-consistent operand allocation is impossible at these sizes.
    let overflow_configs = [
        (
            "hidden_size overflow",
            DecoderLayerConfig {
                hidden_size: usize::MAX,
                ..base_config
            },
        ),
        (
            "intermediate_size overflow",
            DecoderLayerConfig {
                intermediate_size: usize::MAX,
                ..base_config
            },
        ),
        (
            "query width overflow",
            DecoderLayerConfig {
                query_heads: usize::MAX,
                head_dim: 2,
                key_value_heads: 1,
                ..base_config
            },
        ),
    ];
    for (case, config) in overflow_configs {
        assert!(generic_layer_case_inputs(config).is_none(), "{case}");
        assert_layer_error(
            decoder_layer_decode_f32(
                &[],
                config,
                &[],
                &[],
                &prefix_trace.kv_cache,
                base_case.parameters.borrowed(),
            ),
            CpuRefErrorCode::DimensionMismatch,
            case,
        );
    }

    for operand in LAYER_OPERANDS {
        let len = match operand {
            LayerOperand::Input => base_case.input.len(),
            LayerOperand::RawCos | LayerOperand::RawSin => 3 * HEAD_DIM,
            LayerOperand::InputNormWeight | LayerOperand::PostAttentionNormWeight => HIDDEN_SIZE,
            LayerOperand::QueryWeight => QUERY_HEADS * HEAD_DIM * HIDDEN_SIZE,
            LayerOperand::KeyWeight | LayerOperand::ValueWeight => {
                KEY_VALUE_HEADS * HEAD_DIM * HIDDEN_SIZE
            }
            LayerOperand::AttentionOutputWeight => HIDDEN_SIZE * QUERY_HEADS * HEAD_DIM,
            LayerOperand::GateWeight | LayerOperand::UpWeight => INTERMEDIATE_SIZE * HIDDEN_SIZE,
            LayerOperand::DownWeight => HIDDEN_SIZE * INTERMEDIATE_SIZE,
        };
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for offset in [0, len / 2, len - 1] {
                let mut input = base_case.input.clone();
                let mut raw_cos = base_case.raw_cos.clone();
                let mut raw_sin = base_case.raw_sin.clone();
                let mut parameters = base_case.parameters.clone();
                layer_operand_mut(
                    operand,
                    &mut input,
                    &mut raw_cos,
                    &mut raw_sin,
                    &mut parameters,
                )[offset] = value;
                assert_layer_error(
                    decoder_layer_decode_f32(
                        &input,
                        base_config,
                        &raw_cos,
                        &raw_sin,
                        &prefix_trace.kv_cache,
                        parameters.borrowed(),
                    ),
                    CpuRefErrorCode::NonFiniteInput,
                    &format!("layer {} nonfinite", operand.label()),
                );
            }
        }
    }

    for operand in [DirectOperand::Keys, DirectOperand::Values] {
        let len = prefix_tokens * cache_width;
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for offset in [0, len / 2, len - 1] {
                let mut cache = prefix_trace.kv_cache.clone();
                match operand {
                    DirectOperand::Keys => cache.keys[offset] = value,
                    DirectOperand::Values => cache.values[offset] = value,
                    DirectOperand::Query => unreachable!(),
                }
                assert_layer_error(
                    decoder_layer_decode_f32(
                        &base_case.input,
                        base_config,
                        &base_case.raw_cos,
                        &base_case.raw_sin,
                        &cache,
                        base_case.parameters.borrowed(),
                    ),
                    CpuRefErrorCode::NonFiniteInput,
                    &format!("layer cache {} nonfinite", operand.label()),
                );
            }
        }
    }
}

#[test]
fn direct_decode_rejects_invalid_geometry_malformed_operands_and_nonfinite_inputs() {
    let invalid_geometry = [
        (
            "zero cache tokens",
            0,
            QUERY_HEADS,
            KEY_VALUE_HEADS,
            HEAD_DIM,
            true,
        ),
        ("zero query heads", 1, 0, KEY_VALUE_HEADS, HEAD_DIM, true),
        ("zero key value heads", 1, QUERY_HEADS, 0, HEAD_DIM, true),
        ("zero head dim", 1, QUERY_HEADS, KEY_VALUE_HEADS, 0, true),
        (
            "query heads not divisible by key value heads",
            1,
            3,
            KEY_VALUE_HEADS,
            HEAD_DIM,
            true,
        ),
        (
            "cache token overflow",
            usize::MAX,
            QUERY_HEADS,
            KEY_VALUE_HEADS,
            HEAD_DIM,
            false,
        ),
        ("query width overflow", 1, usize::MAX, 1, 2, false),
        ("head dim overflow", 1, 4, 2, usize::MAX, false),
    ];
    for (case, cache_tokens, query_heads, key_value_heads, head_dim, finite_shape) in
        invalid_geometry
    {
        let fixture = if finite_shape {
            direct_fixture(cache_tokens, query_heads, key_value_heads, head_dim)
        } else {
            // These shapes are intentionally empty because checked length
            // multiplication overflows before any consistent operand allocation.
            DirectDecodeFixture {
                query: Vec::new(),
                keys: Vec::new(),
                values: Vec::new(),
            }
        };
        assert_direct_error(
            invoke_direct_decode(
                &fixture,
                cache_tokens,
                query_heads,
                key_value_heads,
                head_dim,
            ),
            CpuRefErrorCode::DimensionMismatch,
            case,
        );
    }

    let cache_tokens = 17;
    let baseline = direct_fixture(cache_tokens, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM);
    for operand in [
        DirectOperand::Query,
        DirectOperand::Keys,
        DirectOperand::Values,
    ] {
        for long in [false, true] {
            let mut fixture = baseline.clone();
            let values = match operand {
                DirectOperand::Query => &mut fixture.query,
                DirectOperand::Keys => &mut fixture.keys,
                DirectOperand::Values => &mut fixture.values,
            };
            if long {
                values.push(0.0);
            } else {
                values.pop();
            }
            assert_direct_error(
                invoke_direct_decode(
                    &fixture,
                    cache_tokens,
                    QUERY_HEADS,
                    KEY_VALUE_HEADS,
                    HEAD_DIM,
                ),
                CpuRefErrorCode::DimensionMismatch,
                &format!(
                    "direct {} {}",
                    operand.label(),
                    if long { "long" } else { "short" }
                ),
            );
        }
    }

    for operand in [
        DirectOperand::Query,
        DirectOperand::Keys,
        DirectOperand::Values,
    ] {
        let len = match operand {
            DirectOperand::Query => baseline.query.len(),
            DirectOperand::Keys => baseline.keys.len(),
            DirectOperand::Values => baseline.values.len(),
        };
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for offset in [0, len / 2, len - 1] {
                let mut fixture = baseline.clone();
                match operand {
                    DirectOperand::Query => fixture.query[offset] = value,
                    DirectOperand::Keys => fixture.keys[offset] = value,
                    DirectOperand::Values => fixture.values[offset] = value,
                }
                assert_direct_error(
                    invoke_direct_decode(
                        &fixture,
                        cache_tokens,
                        QUERY_HEADS,
                        KEY_VALUE_HEADS,
                        HEAD_DIM,
                    ),
                    CpuRefErrorCode::NonFiniteInput,
                    &format!("direct {} nonfinite", operand.label()),
                );
            }
        }
    }
}
