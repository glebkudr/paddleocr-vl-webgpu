use std::fs;
use std::path::PathBuf;

use pvlc_cpu_ref::{CpuRefError, CpuRefErrorCode, apply_pinned_decoder_multimodal_rope_f32};
use pvlc_safetensors::SafetensorsCatalog;

const TOKENS: usize = 332;
const QUERY_HEADS: usize = 16;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const RAW_AXIS_LEN: usize = TOKENS * HEAD_DIM;
const FIXTURE_BLAKE3: &str = "c815b6e8e5806f1617b2b93b6798fc2b3ff9afea032f8e9fc9cbcbde12474497";

const QUERY: &str = "decoder.layer.00.q.token_major";
const KEY: &str = "decoder.layer.00.k.token_major";
const COS: &str = "decoder.rope.cos.axis_major";
const SIN: &str = "decoder.rope.sin.axis_major";
const EXPECTED_QUERY: &str = "decoder.layer.00.mrope.q.token_major";
const EXPECTED_KEY: &str = "decoder.layer.00.mrope.k.token_major";

type Inputs = [Vec<f32>; 4];

struct TensorSpec {
    name: &'static str,
    shape: &'static [u64],
    raw_blake3: &'static str,
}

// Provenance: the pinned remote-code oracle is
// models/snapshots/66317acc4c9fc17bd154591ce650735cd2855f3e/
// modeling_paddleocr_vl.py:313 `apply_multimodal_rotary_pos_emb`.
// The official Python integration proof is
// tools/reference_capture/tests/test_transformers_oracle_integration.py:647-665,
// using semantic IDs decoder.layer.00.{q,k,mrope.q,mrope.k} and
// decoder.rope.{cos,sin} with sections [16,24,24].
#[rustfmt::skip]
const TENSORS: [TensorSpec; 6] = [
    TensorSpec { name: KEY, shape: &[332, 256], raw_blake3: "e729075dff364dca4699edb9c1e9e96ea856cffc2c5e091d96899a642eb0c02a" },
    TensorSpec { name: EXPECTED_KEY, shape: &[332, 2, 128], raw_blake3: "d0674f66af35df6bc48a6e60e3098bcaf9fa5e22ba9e20afe29bad44e82140f0" },
    TensorSpec { name: EXPECTED_QUERY, shape: &[332, 16, 128], raw_blake3: "053c3f08e2034c513c7f8bafaa9216d6b4a29992ef7dc984b1d4b54db4527273" },
    TensorSpec { name: QUERY, shape: &[332, 2048], raw_blake3: "888a4232edc8b6e404f34f494961b2e4645af3b6cebbd17f06ec66058b70b111" },
    TensorSpec { name: COS, shape: &[3, 332, 128], raw_blake3: "096287f2c2ee912105fbc747def39441b541c50b87b1330a8b3b3647b2b49654" },
    TensorSpec { name: SIN, shape: &[3, 332, 128], raw_blake3: "d34eff803104785331690d7f263c4f7ce44838f6083c5f2fb5ed987de613d310" },
];

#[rustfmt::skip]
const METADATA: [(&str, &str); 14] = [
    ("case_id", "ocr.clean_latin.0001"),
    ("decoded_text", "JUL"),
    ("device", "mps"),
    ("dtype", "bfloat16"),
    ("fixture_schema", "pvlc.decoder_mrope.official.v1"),
    ("generated_tokens", "94013,898"),
    ("head_dim", "128"),
    ("key_value_heads", "2"),
    ("model_id", "PaddlePaddle/PaddleOCR-VL-1.6"),
    ("model_revision", "66317acc4c9fc17bd154591ce650735cd2855f3e"),
    ("mrope_sections", "16,24,24"),
    ("oracle", "TransformersOracle pinned remote code"),
    ("query_heads", "16"),
    ("trace_level", "L3"),
];

#[derive(Clone, Copy, Debug)]
struct Policy {
    max_abs: f64,
    mean_abs: f64,
    p99_abs: f64,
    rel_l2: f64,
}

// Fixed from independent f64 analysis of the FP32 equation against the MPS BF16 capture.
const QUERY_POLICY: Policy = Policy {
    max_abs: 0.036,
    mean_abs: 0.000_61,
    p99_abs: 0.004_1,
    rel_l2: 0.001_66,
};
const KEY_POLICY: Policy = Policy {
    max_abs: 0.021_1,
    mean_abs: 0.000_93,
    p99_abs: 0.005_7,
    rel_l2: 0.001_36,
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
        .join("tests/fixtures/decoder-mrope-official-v1.safetensors")
}

fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn hash_f32_le(values: &[f32]) -> String {
    let mut hasher = blake3::Hasher::new();
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn tensor_bytes(catalog: &SafetensorsCatalog, name: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    catalog.copy_tensor_to(name, &mut bytes).unwrap();
    bytes
}

fn load_inputs(catalog: &SafetensorsCatalog) -> Inputs {
    [QUERY, KEY, COS, SIN].map(|name| catalog.load_tensor_f32(name).unwrap())
}

fn invoke(inputs: &Inputs, tokens: usize) -> Result<(Vec<f32>, Vec<f32>), CpuRefError> {
    apply_pinned_decoder_multimodal_rope_f32(&inputs[0], &inputs[1], tokens, &inputs[2], &inputs[3])
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

fn assert_policy(label: &str, actual: &[f32], expected: &[f32], policy: Policy) {
    let metrics = error_metrics(actual, expected);
    assert!(
        policy.accepts(metrics),
        "{label} metrics {metrics:?} exceed fixed policy {policy:?}"
    );
}

#[test]
fn authenticates_official_fixture_metadata_headers_and_raw_payloads() {
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
fn pinned_wrapper_matches_official_mps_bf16_capture_and_rejects_axis_collapse() {
    let catalog = SafetensorsCatalog::open(fixture_path()).unwrap();
    let inputs = load_inputs(&catalog);
    let (actual_query, actual_key) = invoke(&inputs, TOKENS).unwrap();
    assert_eq!(
        hash_f32_le(&actual_query),
        "c752b4514e76e2e4941a3c1f8fc74d9c2bec890772d368814bdc03109fccf972"
    );
    assert_eq!(
        hash_f32_le(&actual_key),
        "ab9a1e1792309ce0d7a8af0a6838b2f5e38b1bb7829ae16dacfb51c06d668d30"
    );
    let expected_query = catalog.load_tensor_f32(EXPECTED_QUERY).unwrap();
    let expected_key = catalog.load_tensor_f32(EXPECTED_KEY).unwrap();
    assert_policy("query", &actual_query, &expected_query, QUERY_POLICY);
    assert_policy("key", &actual_key, &expected_key, KEY_POLICY);

    let mut collapsed = inputs.clone();
    collapsed[2] = inputs[2][..RAW_AXIS_LEN].repeat(3);
    collapsed[3] = inputs[3][..RAW_AXIS_LEN].repeat(3);
    let (bad_query, bad_key) = invoke(&collapsed, TOKENS).unwrap();
    assert!(
        bad_query
            .iter()
            .chain(&bad_key)
            .all(|value| value.is_finite())
    );
    assert!(!QUERY_POLICY.accepts(error_metrics(&bad_query, &expected_query)));
    assert!(!KEY_POLICY.accepts(error_metrics(&bad_key, &expected_key)));
}

#[test]
fn pinned_wrapper_fail_closes_for_lengths_tokens_overflow_and_nonfinite_inputs() {
    let catalog = SafetensorsCatalog::open(fixture_path()).unwrap();
    let finite = load_inputs(&catalog);
    for operand in 0..4 {
        let mut short = finite.clone();
        short[operand].pop();
        assert_error(invoke(&short, TOKENS), CpuRefErrorCode::DimensionMismatch);
        let mut long = finite.clone();
        long[operand].push(0.0);
        assert_error(invoke(&long, TOKENS), CpuRefErrorCode::DimensionMismatch);
    }
    assert_error(invoke(&finite, 0), CpuRefErrorCode::DimensionMismatch);
    assert_error(
        invoke(&finite, usize::MAX),
        CpuRefErrorCode::DimensionMismatch,
    );

    for operand in 0..4 {
        let len = finite[operand].len();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for index in [0, len / 2, len - 1] {
                let mut inputs = finite.clone();
                inputs[operand][index] = value;
                assert_error(invoke(&inputs, TOKENS), CpuRefErrorCode::NonFiniteInput);
            }
        }
    }
    assert_eq!(finite[0].len(), TOKENS * QUERY_HEADS * HEAD_DIM);
    assert_eq!(finite[1].len(), TOKENS * KEY_VALUE_HEADS * HEAD_DIM);
}
