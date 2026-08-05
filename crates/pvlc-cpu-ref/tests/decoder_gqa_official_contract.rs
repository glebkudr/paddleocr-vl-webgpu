use std::fs;
use std::path::PathBuf;

use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, DecoderPrefillKvCache, pinned_decoder_causal_gqa_f32,
    write_pinned_decoder_prefill_kv_f32,
};
use pvlc_safetensors::SafetensorsCatalog;

const TOKENS: usize = 332;
const QUERY_HEADS: usize = 16;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const FIXTURE_BLAKE3: &str = "4ce7feeafb64a7a3494b5b54f2fbfb7cd783cfe080a3a696028c5cf1bbc83f98";

const QUERY: &str = "decoder.layer.00.mrope.q.token_major";
const KEY: &str = "decoder.layer.00.kv.key.token_major";
const VALUE: &str = "decoder.layer.00.kv.value.token_major";
const EXPECTED: &str = "decoder.layer.00.attention.context.token_major";

type Inputs = [Vec<f32>; 3];

struct TensorSpec {
    name: &'static str,
    shape: &'static [u64],
    raw_blake3: &'static str,
}

// Pinned semantics:
// models/snapshots/66317acc4c9fc17bd154591ce650735cd2855f3e/
// modeling_paddleocr_vl.py:234-275 (`repeat_kv`, `eager_attention_forward_ernie`)
// and :358-448 (`Ernie4_5Attention.forward`). The official integration proof is
// tools/reference_capture/tests/test_transformers_oracle_integration.py:648-668,
// with semantic IDs decoder.layer.00.{mrope.q,kv.key,kv.value,attention.context}.
#[rustfmt::skip]
const TENSORS: [TensorSpec; 4] = [
    TensorSpec { name: EXPECTED, shape: &[332, 16, 128], raw_blake3: "8e7a9de666991e9320c6909e84f2f9b0fcb5e5f1dac5d192a4eb80a179077b01" },
    TensorSpec { name: KEY, shape: &[332, 2, 128], raw_blake3: "d0674f66af35df6bc48a6e60e3098bcaf9fa5e22ba9e20afe29bad44e82140f0" },
    TensorSpec { name: VALUE, shape: &[332, 2, 128], raw_blake3: "fc6f30bc2fc420c6166a0380c29c349c71caf1944dab6671aca50c5eb5f27202" },
    TensorSpec { name: QUERY, shape: &[332, 16, 128], raw_blake3: "053c3f08e2034c513c7f8bafaa9216d6b4a29992ef7dc984b1d4b54db4527273" },
];

#[rustfmt::skip]
const METADATA: [(&str, &str); 16] = [
    ("case_id", "ocr.clean_latin.0001"),
    ("causal", "true"),
    ("decoded_text", "JUL"),
    ("device", "mps"),
    ("dtype", "bfloat16"),
    ("fixture_schema", "pvlc.decoder_gqa.official.v1"),
    ("generated_tokens", "94013,898"),
    ("head_dim", "128"),
    ("key_value_heads", "2"),
    ("model_id", "PaddlePaddle/PaddleOCR-VL-1.6"),
    ("model_revision", "66317acc4c9fc17bd154591ce650735cd2855f3e"),
    ("oracle", "TransformersOracle pinned remote code"),
    ("query_heads", "16"),
    ("scale", "head_dim^-0.5"),
    ("tokens", "332"),
    ("trace_level", "L3"),
];

#[derive(Clone, Copy, Debug)]
struct Policy {
    max_abs: f64,
    mean_abs: f64,
    p99_abs: f64,
    rel_l2: f64,
}

const POLICY: Policy = Policy {
    max_abs: 0.003_6,
    mean_abs: 0.000_042,
    p99_abs: 0.000_23,
    rel_l2: 0.002_7,
};

#[derive(Clone, Copy, Debug)]
struct Metrics {
    max_abs: f64,
    mean_abs: f64,
    p99_abs: f64,
    rel_l2: f64,
}

impl Policy {
    fn accepts(self, metrics: Metrics) -> bool {
        metrics.max_abs <= self.max_abs
            && metrics.mean_abs <= self.mean_abs
            && metrics.p99_abs <= self.p99_abs
            && metrics.rel_l2 <= self.rel_l2
    }
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/decoder-gqa-official-v1.safetensors")
}

fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn tensor_bytes(catalog: &SafetensorsCatalog, name: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    catalog.copy_tensor_to(name, &mut bytes).unwrap();
    bytes
}

fn load_inputs(catalog: &SafetensorsCatalog) -> Inputs {
    [QUERY, KEY, VALUE].map(|name| catalog.load_tensor_f32(name).unwrap())
}

fn invoke(inputs: &Inputs, tokens: usize) -> Result<Vec<f32>, CpuRefError> {
    pinned_decoder_causal_gqa_f32(&inputs[0], &inputs[1], &inputs[2], tokens)
}

fn assert_error<T>(result: Result<T, CpuRefError>, expected: CpuRefErrorCode) {
    match result {
        Err(error) => assert_eq!(error.code(), expected),
        Ok(_) => panic!("expected {expected:?}"),
    }
}

fn error_metrics(actual: &[f32], expected: &[f32]) -> Metrics {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    assert!(actual.iter().chain(expected).all(|value| value.is_finite()));
    let mut ordered_abs = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| (f64::from(actual) - f64::from(expected)).abs())
        .collect::<Vec<_>>();
    let mean_abs = ordered_abs.iter().sum::<f64>() / ordered_abs.len() as f64;
    let squared_error = actual
        .iter()
        .zip(expected)
        .map(|(&actual, &expected)| (f64::from(actual) - f64::from(expected)).powi(2))
        .sum::<f64>();
    let expected_energy = expected
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>();
    assert!(expected_energy > 0.0);
    ordered_abs.sort_by(f64::total_cmp);
    // Nearest-rank p99: ascending errors at one-based rank ceil(0.99 * N).
    let p99_index = (99 * ordered_abs.len()).div_ceil(100) - 1;
    Metrics {
        max_abs: *ordered_abs.last().unwrap(),
        mean_abs,
        p99_abs: ordered_abs[p99_index],
        rel_l2: (squared_error / expected_energy).sqrt(),
    }
}

fn assert_policy(actual: &[f32], expected: &[f32]) {
    let metrics = error_metrics(actual, expected);
    assert!(
        POLICY.accepts(metrics),
        "official GQA metrics {metrics:?} exceed fixed policy {POLICY:?}"
    );
}

fn collapse_to_kv_head_zero(input: &[f32]) -> Vec<f32> {
    let mut collapsed = vec![0.0; input.len()];
    for token in 0..TOKENS {
        let source = token * KEY_VALUE_HEADS * HEAD_DIM;
        for head in 0..KEY_VALUE_HEADS {
            let destination = (token * KEY_VALUE_HEADS + head) * HEAD_DIM;
            collapsed[destination..destination + HEAD_DIM]
                .copy_from_slice(&input[source..source + HEAD_DIM]);
        }
    }
    collapsed
}

#[test]
fn authenticates_official_gqa_fixture_metadata_headers_and_raw_payloads() {
    let path = fixture_path();
    assert_eq!(hash_bytes(&fs::read(&path).unwrap()), FIXTURE_BLAKE3);
    let catalog = SafetensorsCatalog::open(path).unwrap();
    let metadata = catalog
        .metadata()
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(metadata, METADATA);
    assert_eq!(catalog.tensors().len(), TENSORS.len());
    for (tensor, spec) in catalog.tensors().iter().zip(&TENSORS) {
        assert_eq!(tensor.name, spec.name);
        assert_eq!(tensor.shape, spec.shape);
        assert_eq!(tensor.dtype.safetensors_name(), "BF16");
        assert_eq!(
            hash_bytes(&tensor_bytes(&catalog, spec.name)),
            spec.raw_blake3
        );
    }
}

#[test]
fn pinned_gqa_matches_official_capture_and_kv_head_collapse_fails_policy() {
    let catalog = SafetensorsCatalog::open(fixture_path()).unwrap();
    let inputs = load_inputs(&catalog);
    let actual = invoke(&inputs, TOKENS).unwrap();
    let expected = catalog.load_tensor_f32(EXPECTED).unwrap();
    assert_eq!(actual.len(), TOKENS * QUERY_HEADS * HEAD_DIM);
    // Independent observed metrics are max=0.003350, mean=0.00003830,
    // nearest-rank p99=0.0002054, rel_l2=0.002524.
    assert_policy(&actual, &expected);

    let collapsed_key = collapse_to_kv_head_zero(&inputs[1]);
    let collapsed_value = collapse_to_kv_head_zero(&inputs[2]);
    let negative = invoke(&[inputs[0].clone(), collapsed_key, collapsed_value], TOKENS).unwrap();
    assert!(negative.iter().all(|value: &f32| value.is_finite()));
    assert!(!POLICY.accepts(error_metrics(&negative, &expected)));
}

#[test]
fn pinned_kv_write_preserves_official_payloads_and_storage_isolation() {
    let catalog = SafetensorsCatalog::open(fixture_path()).unwrap();
    let inputs = load_inputs(&catalog);
    let mut key = inputs[1].clone();
    let mut value = inputs[2].clone();
    let expected_key = key.clone();
    let expected_value = value.clone();
    let mut cache: DecoderPrefillKvCache =
        write_pinned_decoder_prefill_kv_f32(&key, &value, TOKENS).unwrap();
    assert_eq!(cache.tokens, TOKENS);
    assert_eq!(cache.key_value_heads, KEY_VALUE_HEADS);
    assert_eq!(cache.head_dim, HEAD_DIM);
    assert_eq!(cache.keys, expected_key);
    assert_eq!(cache.values, expected_value);

    key[0] = f32::NAN;
    value[0] = f32::INFINITY;
    assert_eq!(cache.keys, expected_key);
    assert_eq!(cache.values, expected_value);
    let value_zero = cache.values[0];
    cache.keys[0] = 42.0;
    assert_eq!(cache.values[0], value_zero);
    let key_one = cache.keys[1];
    cache.values[1] = -42.0;
    assert_eq!(cache.keys[1], key_one);
}

#[test]
fn pinned_gqa_fail_closes_for_tokens_lengths_and_nonfinite_inputs() {
    let catalog = SafetensorsCatalog::open(fixture_path()).unwrap();
    let finite = load_inputs(&catalog);
    assert_error(invoke(&finite, 0), CpuRefErrorCode::DimensionMismatch);
    assert_error(
        invoke(&finite, usize::MAX),
        CpuRefErrorCode::DimensionMismatch,
    );

    for operand in 0..3 {
        let mut short = finite.clone();
        short[operand].pop();
        assert_error(invoke(&short, TOKENS), CpuRefErrorCode::DimensionMismatch);
        let mut long = finite.clone();
        long[operand].push(0.0);
        assert_error(invoke(&long, TOKENS), CpuRefErrorCode::DimensionMismatch);
    }

    for operand in 0..3 {
        let len = finite[operand].len();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for offset in [0, len / 2, len - 1] {
                let mut inputs = finite.clone();
                inputs[operand][offset] = value;
                assert_error(invoke(&inputs, TOKENS), CpuRefErrorCode::NonFiniteInput);
            }
        }
    }
}
