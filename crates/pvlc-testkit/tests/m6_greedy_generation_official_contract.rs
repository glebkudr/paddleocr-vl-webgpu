//! Ordinary M6c2 official gate assembled only from checked-in compact fixtures.
//! It deliberately does not load the 1.9 GB model payload or execute decoder weights.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, DecoderPrefillKvCache, GreedyDecodeStep, GreedyGenerationTrace,
    GreedyStopReason, TopKEntry, pinned_greedy_generate_f32, top_k,
};
use pvlc_safetensors::SafetensorsCatalog;

const MODEL_ID: &str = "PaddlePaddle/PaddleOCR-VL-1.6";
const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const MODEL_LOCK_BYTES: usize = 2_385;
const MODEL_LOCK_BLAKE3: &str = "c2ffa46821fdda1a7e4cd6bad63cdcdc2f775b8fed250b0679cc7d5411577d10";
const MODEL_FILE_BYTES: i64 = 1_917_255_968;
const MODEL_FILE_BLAKE3: &str = "4dc3ab13685a0c0a701f77f9c5ebafdc0004074247e00bde6b7e7b04279a41fc";
const GENERATION_CONFIG_BYTES: usize = 133;
const GENERATION_CONFIG_BLAKE3: &str =
    "78c47d8168d89991b41518b1ec92efdea064e3292967312a8526ec299d4f4952";
const PREFILL_FIXTURE_BYTES: usize = 887_680;
const PREFILL_FIXTURE_BLAKE3: &str =
    "f73291c81f8251715bf7ec64a749338c928930add8406845463705ca1468334d";
const DECODE_FIXTURE_BYTES: usize = 6_438_376;
const DECODE_FIXTURE_BLAKE3: &str =
    "386735089b35b2a1fc50ad578678689eed98b3e62f2a0c18d5e2890d0e6a8ebf";
const DECODE_KEY_RAW_BLAKE3: &str =
    "6902c1a29d014177981eaa5daa12390141f0e16331b78dbac0653c45faa3692f";
const DECODE_VALUE_RAW_BLAKE3: &str =
    "e396c182612fc2a6e8660dbeb1a5d59d05f5267f07df13dab39c5802acf530f3";

const LAYERS: usize = 18;
const PREFIX_TOKENS: usize = 332;
const FULL_TOKENS: usize = 333;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const VOCAB_SIZE: usize = 103_424;
const PREFILL_TOKEN: usize = 94_013;
const DECODE_TOKEN: usize = 898;
const EOS_TOKEN: usize = 2;
const PREFILL_LOGITS: &str = "decoder.prefill.logits.last";
const DECODE_LOGITS: &str = "decoder.decode.00.logits";
const STACKED_KEYS: &str = "decoder.decode.00.kv.key.layer_token_major";
const STACKED_VALUES: &str = "decoder.decode.00.kv.value.layer_token_major";

const MODEL_LOCK: &str = include_str!("../../../models/paddleocr-vl-1.6.lock");
const GOLDEN_LOCK: &str = include_str!("../../../goldens/golden.lock");
const GENERATION_CONFIG: &str = include_str!(concat!(
    "../../../models/snapshots/",
    "66317acc4c9fc17bd154591ce650735cd2855f3e/",
    "generation_config.json"
));

fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn tensor_bytes(catalog: &SafetensorsCatalog, name: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    catalog.copy_tensor_to(name, &mut bytes).unwrap();
    bytes
}

fn metadata(catalog: &SafetensorsCatalog) -> BTreeMap<String, String> {
    catalog
        .metadata()
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn assert_common_metadata(values: &BTreeMap<String, String>) {
    assert_eq!(values.get("model_id").map(String::as_str), Some(MODEL_ID));
    assert_eq!(
        values.get("model_revision").map(String::as_str),
        Some(MODEL_REVISION)
    );
    assert_eq!(
        values.get("generated_tokens").map(String::as_str),
        Some("94013,898")
    );
    assert_eq!(values.get("decoded_text").map(String::as_str), Some("JUL"));
}

fn assert_model_and_generation_locks() {
    assert_eq!(MODEL_LOCK.len(), MODEL_LOCK_BYTES);
    assert_eq!(hash_bytes(MODEL_LOCK.as_bytes()), MODEL_LOCK_BLAKE3);
    let model: toml::Value = toml::from_str(MODEL_LOCK).unwrap();
    assert_eq!(model["format_version"].as_integer(), Some(1));
    assert_eq!(model["compiler_model_abi"].as_integer(), Some(1));
    assert_eq!(model["model_id"].as_str(), Some(MODEL_ID));
    assert_eq!(model["revision"].as_str(), Some(MODEL_REVISION));
    let model_file = model["files"]["model.safetensors"].as_table().unwrap();
    assert_eq!(model_file["size"].as_integer(), Some(MODEL_FILE_BYTES));
    assert_eq!(model_file["blake3"].as_str(), Some(MODEL_FILE_BLAKE3));
    let generation_file = model["files"]["generation_config.json"].as_table().unwrap();
    assert_eq!(
        generation_file["size"].as_integer(),
        Some(GENERATION_CONFIG_BYTES as i64)
    );
    assert_eq!(
        generation_file["blake3"].as_str(),
        Some(GENERATION_CONFIG_BLAKE3)
    );

    assert_eq!(GENERATION_CONFIG.len(), GENERATION_CONFIG_BYTES);
    assert_eq!(
        hash_bytes(GENERATION_CONFIG.as_bytes()),
        GENERATION_CONFIG_BLAKE3
    );
    let generation: serde_json::Value = serde_json::from_str(GENERATION_CONFIG).unwrap();
    assert_eq!(generation["eos_token_id"].as_u64(), Some(EOS_TOKEN as u64));

    let golden: toml::Value = toml::from_str(GOLDEN_LOCK).unwrap();
    assert_eq!(golden["model_revision"].as_str(), Some(MODEL_REVISION));
    let matches = golden["bundles"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|bundle| {
            bundle["case_id"].as_str() == Some("ocr.clean_latin.0001")
                && bundle["trace_level"].as_str() == Some("L3")
        })
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0]["decoded_text"].as_str(), Some("JUL"));
    assert_eq!(
        matches[0]["generated_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|token| token.as_integer().unwrap())
            .collect::<Vec<_>>(),
        [PREFILL_TOKEN as i64, DECODE_TOKEN as i64]
    );
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        let bits = value.to_bits();
        assert_eq!(bits & 0xffff, 0, "fixture value is not exactly BF16");
        bytes.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
    }
    bytes
}

fn slice_layer_caches(
    stacked_keys: &[f32],
    stacked_values: &[f32],
    retained_tokens: usize,
) -> Vec<DecoderPrefillKvCache> {
    let row_width = KV_HEADS * HEAD_DIM;
    let full_layer_len = FULL_TOKENS * row_width;
    let retained_layer_len = retained_tokens * row_width;
    assert!(retained_tokens <= FULL_TOKENS);
    assert_eq!(stacked_keys.len(), LAYERS * full_layer_len);
    assert_eq!(stacked_values.len(), LAYERS * full_layer_len);

    (0..LAYERS)
        .map(|layer| {
            let start = layer * full_layer_len;
            DecoderPrefillKvCache {
                keys: stacked_keys[start..start + retained_layer_len].to_vec(),
                values: stacked_values[start..start + retained_layer_len].to_vec(),
                tokens: retained_tokens,
                key_value_heads: KV_HEADS,
                head_dim: HEAD_DIM,
            }
        })
        .collect()
}

fn load_official_cache_pair(
    decode: &SafetensorsCatalog,
) -> (Vec<DecoderPrefillKvCache>, Vec<DecoderPrefillKvCache>) {
    let expected_shape = [
        LAYERS as u64,
        FULL_TOKENS as u64,
        KV_HEADS as u64,
        HEAD_DIM as u64,
    ];
    for name in [STACKED_KEYS, STACKED_VALUES] {
        let tensor = decode.tensor(name).unwrap();
        assert_eq!(tensor.shape, expected_shape, "{name}");
        assert_eq!(tensor.dtype.safetensors_name(), "BF16", "{name}");
    }

    let raw_keys = tensor_bytes(decode, STACKED_KEYS);
    let raw_values = tensor_bytes(decode, STACKED_VALUES);
    assert_eq!(hash_bytes(&raw_keys), DECODE_KEY_RAW_BLAKE3);
    assert_eq!(hash_bytes(&raw_values), DECODE_VALUE_RAW_BLAKE3);
    let stacked_keys = decode.load_tensor_f32(STACKED_KEYS).unwrap();
    let stacked_values = decode.load_tensor_f32(STACKED_VALUES).unwrap();
    let prefix = slice_layer_caches(&stacked_keys, &stacked_values, PREFIX_TOKENS);
    let full = slice_layer_caches(&stacked_keys, &stacked_values, FULL_TOKENS);

    let row_bytes = KV_HEADS * HEAD_DIM * 2;
    let raw_full_layer_len = FULL_TOKENS * row_bytes;
    let raw_prefix_layer_len = PREFIX_TOKENS * row_bytes;
    for layer in 0..LAYERS {
        let raw_start = layer * raw_full_layer_len;
        let raw_key_layer = &raw_keys[raw_start..raw_start + raw_full_layer_len];
        let raw_value_layer = &raw_values[raw_start..raw_start + raw_full_layer_len];
        let prefix_key_bytes = bf16_bytes(&prefix[layer].keys);
        let prefix_value_bytes = bf16_bytes(&prefix[layer].values);
        let full_key_bytes = bf16_bytes(&full[layer].keys);
        let full_value_bytes = bf16_bytes(&full[layer].values);
        assert_eq!(full_key_bytes, raw_key_layer, "full key layer {layer}");
        assert_eq!(
            full_value_bytes, raw_value_layer,
            "full value layer {layer}"
        );
        assert_eq!(
            prefix_key_bytes,
            raw_key_layer[..raw_prefix_layer_len],
            "prefix key layer {layer}"
        );
        assert_eq!(
            prefix_value_bytes,
            raw_value_layer[..raw_prefix_layer_len],
            "prefix value layer {layer}"
        );
        assert_eq!(prefix_key_bytes, full_key_bytes[..raw_prefix_layer_len]);
        assert_eq!(prefix_value_bytes, full_value_bytes[..raw_prefix_layer_len]);
    }
    (prefix, full)
}

fn update_usize(hasher: &mut blake3::Hasher, value: usize) {
    hasher.update(&(value as u64).to_le_bytes());
}

fn update_values(hasher: &mut blake3::Hasher, values: &[f32]) {
    update_usize(hasher, values.len());
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn trace_digest(trace: &GreedyGenerationTrace) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"pvlc-m6c2-official-generation-v1\0");
    update_usize(&mut hasher, trace.generated_tokens.len());
    for token in &trace.generated_tokens {
        update_usize(&mut hasher, *token);
    }
    update_usize(&mut hasher, trace.kv_caches.len());
    for cache in &trace.kv_caches {
        update_usize(&mut hasher, cache.tokens);
        update_usize(&mut hasher, cache.key_value_heads);
        update_usize(&mut hasher, cache.head_dim);
        update_values(&mut hasher, &cache.keys);
        update_values(&mut hasher, &cache.values);
    }
    hasher.update(&[match trace.stop_reason {
        GreedyStopReason::MaxNewTokens => 0,
        GreedyStopReason::EosToken => 1,
    }]);
    update_usize(&mut hasher, trace.decode_steps);
    hasher.finalize().to_hex().to_string()
}

fn run_official_request(
    prefill_top: &[TopKEntry],
    decode_top: &[TopKEntry],
    prefix: &[DecoderPrefillKvCache],
    full: &[DecoderPrefillKvCache],
) -> GreedyGenerationTrace {
    let mut returned_full = Some(full.to_vec());
    let mut calls = 0;
    let trace = pinned_greedy_generate_f32(
        prefill_top,
        prefix.to_vec(),
        2,
        |step_index: usize, input_token: usize, caches: &[DecoderPrefillKvCache]| {
            calls += 1;
            assert_eq!((step_index, input_token), (0, PREFILL_TOKEN));
            assert_eq!(caches, prefix);
            Ok(GreedyDecodeStep {
                top_k: decode_top.to_vec(),
                kv_caches: returned_full.take().unwrap(),
            })
        },
    )
    .unwrap();
    assert_eq!(calls, 1);
    trace
}

#[test]
fn authenticates_compact_official_generation_sources_and_top_one_tokens() {
    assert_model_and_generation_locks();

    let prefill_path = fixture_path("prefill-lm-head-official-v1.safetensors");
    let prefill_bytes = fs::read(&prefill_path).unwrap();
    assert_eq!(prefill_bytes.len(), PREFILL_FIXTURE_BYTES);
    assert_eq!(hash_bytes(&prefill_bytes), PREFILL_FIXTURE_BLAKE3);
    let prefill = SafetensorsCatalog::open(prefill_path).unwrap();
    assert_eq!(prefill.tensors().len(), 2);
    assert_eq!(prefill.metadata().len(), 19);
    assert_common_metadata(&metadata(&prefill));
    let prefill_logits_tensor = prefill.tensor(PREFILL_LOGITS).unwrap();
    assert_eq!(prefill_logits_tensor.shape, [1, VOCAB_SIZE as u64]);
    assert_eq!(prefill_logits_tensor.dtype.safetensors_name(), "BF16");
    let prefill_top = top_k(&prefill.load_tensor_f32(PREFILL_LOGITS).unwrap(), 1).unwrap();
    assert_eq!(
        prefill_top,
        [TopKEntry {
            index: PREFILL_TOKEN,
            value: 8.75
        }]
    );

    let decode_path = fixture_path("decoder-decode-official-v1.safetensors");
    let decode_bytes = fs::read(&decode_path).unwrap();
    assert_eq!(decode_bytes.len(), DECODE_FIXTURE_BYTES);
    assert_eq!(hash_bytes(&decode_bytes), DECODE_FIXTURE_BLAKE3);
    let decode = SafetensorsCatalog::open(decode_path).unwrap();
    assert_eq!(decode.tensors().len(), 44);
    assert_eq!(decode.metadata().len(), 38);
    assert_common_metadata(&metadata(&decode));
    let decode_logits_tensor = decode.tensor(DECODE_LOGITS).unwrap();
    assert_eq!(decode_logits_tensor.shape, [1, VOCAB_SIZE as u64]);
    assert_eq!(decode_logits_tensor.dtype.safetensors_name(), "BF16");
    let decode_top = top_k(&decode.load_tensor_f32(DECODE_LOGITS).unwrap(), 1).unwrap();
    assert_eq!(
        decode_top,
        [TopKEntry {
            index: DECODE_TOKEN,
            value: 8.5
        }]
    );

    let (prefix_caches, full_caches) = load_official_cache_pair(&decode);
    let mut first = run_official_request(&prefill_top, &decode_top, &prefix_caches, &full_caches);
    let second = run_official_request(&prefill_top, &decode_top, &prefix_caches, &full_caches);

    assert_eq!(first.generated_tokens, [PREFILL_TOKEN, DECODE_TOKEN]);
    assert_eq!(first.decode_steps, 1);
    assert_eq!(first.stop_reason, GreedyStopReason::MaxNewTokens);
    assert_eq!(first.kv_caches, full_caches);
    assert_eq!(second.generated_tokens, first.generated_tokens);
    assert_eq!(second.decode_steps, first.decode_steps);
    assert_eq!(&second.stop_reason, &first.stop_reason);
    assert_eq!(second.kv_caches, first.kv_caches);
    assert_eq!(trace_digest(&second), trace_digest(&first));
    for (first_layer, second_layer) in first.kv_caches.iter().zip(&second.kv_caches) {
        assert_ne!(first_layer.keys.as_ptr(), second_layer.keys.as_ptr());
        assert_ne!(first_layer.values.as_ptr(), second_layer.values.as_ptr());
    }

    let preserved_second_key = second.kv_caches[0].keys[0].to_bits();
    let preserved_second_value = second.kv_caches[0].values[0].to_bits();
    first.kv_caches[0].keys[0] = -123.0;
    first.kv_caches[0].values[0] = 456.0;
    assert_eq!(second.kv_caches[0].keys[0].to_bits(), preserved_second_key);
    assert_eq!(
        second.kv_caches[0].values[0].to_bits(),
        preserved_second_value
    );
}

#[test]
fn official_generation_rejects_a_reset_cache_after_exactly_one_callback() {
    let decode_path = fixture_path("decoder-decode-official-v1.safetensors");
    let decode_bytes = fs::read(&decode_path).unwrap();
    assert_eq!(decode_bytes.len(), DECODE_FIXTURE_BYTES);
    assert_eq!(hash_bytes(&decode_bytes), DECODE_FIXTURE_BLAKE3);
    let decode = SafetensorsCatalog::open(decode_path).unwrap();
    let (prefix_caches, _) = load_official_cache_pair(&decode);
    let mut reset_caches = Some(prefix_caches.clone());
    let mut calls = 0;

    let result: Result<GreedyGenerationTrace, CpuRefError> = pinned_greedy_generate_f32(
        &[TopKEntry {
            index: PREFILL_TOKEN,
            value: 8.75,
        }],
        prefix_caches.clone(),
        3,
        |step_index: usize, input_token: usize, caches: &[DecoderPrefillKvCache]| {
            calls += 1;
            assert_eq!((step_index, input_token), (0, PREFILL_TOKEN));
            assert_eq!(caches, prefix_caches);
            Ok(GreedyDecodeStep {
                top_k: vec![TopKEntry {
                    index: DECODE_TOKEN,
                    value: 8.5,
                }],
                kv_caches: reset_caches.take().unwrap(),
            })
        },
    );

    assert_eq!(calls, 1);
    let error: CpuRefError = match result {
        Err(error) => error,
        Ok(_) => panic!("reset caches must be rejected before any later callback"),
    };
    assert_eq!(error.code(), CpuRefErrorCode::DimensionMismatch);
}
