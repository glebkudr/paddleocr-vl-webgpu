//! Ordinary M6d decode-chunking gate over checked-in compact fixtures only.
//! The pinned 1.9 GB model payload and decoder weights are never opened.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, DecoderPrefillKvCache, GreedyChunkedGenerationTrace,
    GreedyDecodeChunk, GreedyDecodeStep, GreedyStopReason, TopKEntry,
    pinned_greedy_generate_chunked_f32, top_k,
};
use pvlc_safetensors::SafetensorsCatalog;

const MODEL_ID: &str = "PaddlePaddle/PaddleOCR-VL-1.6";
const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const MODEL_LOCK_BYTES: usize = 2_385;
const MODEL_LOCK_BLAKE3: &str = "c2ffa46821fdda1a7e4cd6bad63cdcdc2f775b8fed250b0679cc7d5411577d10";
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
const TRACE_BLAKE3: &str = "84a4cf8aff4ed9364a9be881d831a3171561b29e835cdbd2fd577d59bf37b68b";

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

struct OfficialCase {
    prefill_top: Vec<TopKEntry>,
    decode_top: Vec<TopKEntry>,
    prefix_caches: Vec<DecoderPrefillKvCache>,
    full_caches: Vec<DecoderPrefillKvCache>,
}

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

fn assert_lock_anchors() {
    assert_eq!(MODEL_LOCK.len(), MODEL_LOCK_BYTES);
    assert_eq!(hash_bytes(MODEL_LOCK.as_bytes()), MODEL_LOCK_BLAKE3);
    let model: toml::Value = toml::from_str(MODEL_LOCK).unwrap();
    assert_eq!(model["model_id"].as_str(), Some(MODEL_ID));
    assert_eq!(model["revision"].as_str(), Some(MODEL_REVISION));
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

fn slice_caches(
    stacked_keys: &[f32],
    stacked_values: &[f32],
    retained_tokens: usize,
) -> Vec<DecoderPrefillKvCache> {
    let row_width = KV_HEADS * HEAD_DIM;
    let full_layer_len = FULL_TOKENS * row_width;
    let retained_layer_len = retained_tokens * row_width;
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

fn load_cache_pair(
    decode: &SafetensorsCatalog,
) -> (Vec<DecoderPrefillKvCache>, Vec<DecoderPrefillKvCache>) {
    let shape = [
        LAYERS as u64,
        FULL_TOKENS as u64,
        KV_HEADS as u64,
        HEAD_DIM as u64,
    ];
    for name in [STACKED_KEYS, STACKED_VALUES] {
        let tensor = decode.tensor(name).unwrap();
        assert_eq!(tensor.shape, shape, "{name}");
        assert_eq!(tensor.dtype.safetensors_name(), "BF16", "{name}");
    }
    let raw_keys = tensor_bytes(decode, STACKED_KEYS);
    let raw_values = tensor_bytes(decode, STACKED_VALUES);
    assert_eq!(hash_bytes(&raw_keys), DECODE_KEY_RAW_BLAKE3);
    assert_eq!(hash_bytes(&raw_values), DECODE_VALUE_RAW_BLAKE3);
    let stacked_keys = decode.load_tensor_f32(STACKED_KEYS).unwrap();
    let stacked_values = decode.load_tensor_f32(STACKED_VALUES).unwrap();
    let prefix = slice_caches(&stacked_keys, &stacked_values, PREFIX_TOKENS);
    let full = slice_caches(&stacked_keys, &stacked_values, FULL_TOKENS);

    let full_layer_bytes = FULL_TOKENS * KV_HEADS * HEAD_DIM * 2;
    let prefix_layer_bytes = PREFIX_TOKENS * KV_HEADS * HEAD_DIM * 2;
    for layer in 0..LAYERS {
        let start = layer * full_layer_bytes;
        let raw_key_layer = &raw_keys[start..start + full_layer_bytes];
        let raw_value_layer = &raw_values[start..start + full_layer_bytes];
        let prefix_keys = bf16_bytes(&prefix[layer].keys);
        let prefix_values = bf16_bytes(&prefix[layer].values);
        let full_keys = bf16_bytes(&full[layer].keys);
        let full_values = bf16_bytes(&full[layer].values);
        assert_eq!(full_keys, raw_key_layer, "full key layer {layer}");
        assert_eq!(full_values, raw_value_layer, "full value layer {layer}");
        assert_eq!(prefix_keys, raw_key_layer[..prefix_layer_bytes]);
        assert_eq!(prefix_values, raw_value_layer[..prefix_layer_bytes]);
        assert_eq!(prefix_keys, full_keys[..prefix_layer_bytes]);
        assert_eq!(prefix_values, full_values[..prefix_layer_bytes]);
    }
    (prefix, full)
}

fn load_official_case() -> OfficialCase {
    assert_lock_anchors();
    let prefill_path = fixture_path("prefill-lm-head-official-v1.safetensors");
    let prefill_bytes = fs::read(&prefill_path).unwrap();
    assert_eq!(prefill_bytes.len(), PREFILL_FIXTURE_BYTES);
    assert_eq!(hash_bytes(&prefill_bytes), PREFILL_FIXTURE_BLAKE3);
    let prefill = SafetensorsCatalog::open(prefill_path).unwrap();
    assert_eq!(prefill.tensors().len(), 2);
    assert_eq!(prefill.metadata().len(), 19);
    assert_common_metadata(&metadata(&prefill));
    let prefill_tensor = prefill.tensor(PREFILL_LOGITS).unwrap();
    assert_eq!(prefill_tensor.shape, [1, VOCAB_SIZE as u64]);
    assert_eq!(prefill_tensor.dtype.safetensors_name(), "BF16");
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
    let decode_tensor = decode.tensor(DECODE_LOGITS).unwrap();
    assert_eq!(decode_tensor.shape, [1, VOCAB_SIZE as u64]);
    assert_eq!(decode_tensor.dtype.safetensors_name(), "BF16");
    let decode_top = top_k(&decode.load_tensor_f32(DECODE_LOGITS).unwrap(), 1).unwrap();
    assert_eq!(
        decode_top,
        [TopKEntry {
            index: DECODE_TOKEN,
            value: 8.5
        }]
    );
    let (prefix_caches, full_caches) = load_cache_pair(&decode);
    OfficialCase {
        prefill_top,
        decode_top,
        prefix_caches,
        full_caches,
    }
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

fn trace_digest(trace: &GreedyChunkedGenerationTrace) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"pvlc-m6d-official-decode-chunk-v1\0");
    update_usize(&mut hasher, trace.generation.generated_tokens.len());
    for token in &trace.generation.generated_tokens {
        update_usize(&mut hasher, *token);
    }
    update_usize(&mut hasher, trace.generation.kv_caches.len());
    for cache in &trace.generation.kv_caches {
        update_usize(&mut hasher, cache.tokens);
        update_usize(&mut hasher, cache.key_value_heads);
        update_usize(&mut hasher, cache.head_dim);
        update_values(&mut hasher, &cache.keys);
        update_values(&mut hasher, &cache.values);
    }
    hasher.update(&[match trace.generation.stop_reason {
        GreedyStopReason::MaxNewTokens => 0,
        GreedyStopReason::EosToken => 1,
    }]);
    update_usize(&mut hasher, trace.generation.decode_steps);
    update_usize(&mut hasher, trace.decode_chunks);
    hasher.finalize().to_hex().to_string()
}

fn run_official_chunk(
    case: &OfficialCase,
    decode_chunk_size: usize,
) -> GreedyChunkedGenerationTrace {
    let owned_prefix = case.prefix_caches.clone();
    let input_pointers = owned_prefix
        .iter()
        .map(|cache| (cache.keys.as_ptr(), cache.values.as_ptr()))
        .collect::<Vec<_>>();
    let returned_full = case.full_caches.clone();
    let output_pointers = returned_full
        .iter()
        .map(|cache| (cache.keys.as_ptr(), cache.values.as_ptr()))
        .collect::<Vec<_>>();
    let mut returned_full = Some(returned_full);
    let mut calls = 0;
    let trace = pinned_greedy_generate_chunked_f32(
        &case.prefill_top,
        owned_prefix,
        2,
        decode_chunk_size,
        |chunk_index: usize,
         input_token: usize,
         current: &[DecoderPrefillKvCache],
         requested_steps: usize| {
            calls += 1;
            assert_eq!(
                (chunk_index, input_token, requested_steps),
                (0, PREFILL_TOKEN, 1)
            );
            assert_eq!(current, case.prefix_caches);
            for ((cache, pointers), canonical) in
                current.iter().zip(&input_pointers).zip(&case.prefix_caches)
            {
                assert_eq!((cache.keys.as_ptr(), cache.values.as_ptr()), *pointers);
                assert_ne!(cache.keys.as_ptr(), canonical.keys.as_ptr());
                assert_ne!(cache.values.as_ptr(), canonical.values.as_ptr());
            }
            Ok(GreedyDecodeChunk {
                steps: vec![GreedyDecodeStep {
                    top_k: case.decode_top.clone(),
                    kv_caches: returned_full.take().unwrap(),
                }],
            })
        },
    )
    .unwrap();
    assert_eq!(calls, 1);
    for (cache, pointers) in trace.generation.kv_caches.iter().zip(&output_pointers) {
        assert_eq!((cache.keys.as_ptr(), cache.values.as_ptr()), *pointers);
    }
    trace
}

fn extend_full_caches(full: &[DecoderPrefillKvCache]) -> Vec<DecoderPrefillKvCache> {
    full.iter()
        .map(|cache| {
            let mut extended = cache.clone();
            extended
                .keys
                .extend(std::iter::repeat_n(0.0, KV_HEADS * HEAD_DIM));
            extended
                .values
                .extend(std::iter::repeat_n(0.0, KV_HEADS * HEAD_DIM));
            extended.tokens = FULL_TOKENS + 1;
            extended
        })
        .collect()
}

#[test]
fn official_one_step_trace_is_identical_for_all_chunk_sizes_and_fixed_digest() {
    let case = load_official_case();
    let mut baseline: Option<GreedyChunkedGenerationTrace> = None;
    for decode_chunk_size in [1, 2, 4, 8, usize::MAX] {
        let trace = run_official_chunk(&case, decode_chunk_size);
        assert_eq!(
            trace.generation.generated_tokens,
            [PREFILL_TOKEN, DECODE_TOKEN]
        );
        assert_eq!(trace.generation.kv_caches, case.full_caches);
        assert_eq!(trace.generation.decode_steps, 1);
        assert_eq!(trace.generation.stop_reason, GreedyStopReason::MaxNewTokens);
        assert_eq!(trace.decode_chunks, 1);
        assert_eq!(trace_digest(&trace), TRACE_BLAKE3);
        for (actual, canonical) in trace.generation.kv_caches.iter().zip(&case.full_caches) {
            assert_ne!(actual.keys.as_ptr(), canonical.keys.as_ptr());
            assert_ne!(actual.values.as_ptr(), canonical.values.as_ptr());
        }
        if let Some(expected) = &baseline {
            assert_eq!(&trace, expected, "chunk size {decode_chunk_size}");
            for (actual, prior) in trace
                .generation
                .kv_caches
                .iter()
                .zip(&expected.generation.kv_caches)
            {
                assert_ne!(actual.keys.as_ptr(), prior.keys.as_ptr());
                assert_ne!(actual.values.as_ptr(), prior.values.as_ptr());
            }
        } else {
            baseline = Some(trace);
        }
    }
}

#[test]
fn official_callback_over_return_is_rejected_after_exactly_one_chunk() {
    let case = load_official_case();
    let full = case.full_caches.clone();
    let extra = extend_full_caches(&full);
    let mut full = Some(full);
    let mut extra = Some(extra);
    let mut calls = 0;
    let result: Result<GreedyChunkedGenerationTrace, CpuRefError> =
        pinned_greedy_generate_chunked_f32(
            &case.prefill_top,
            case.prefix_caches.clone(),
            2,
            8,
            |chunk_index: usize,
             input_token: usize,
             current: &[DecoderPrefillKvCache],
             requested_steps: usize| {
                calls += 1;
                assert_eq!(
                    (chunk_index, input_token, requested_steps),
                    (0, PREFILL_TOKEN, 1)
                );
                assert_eq!(current, case.prefix_caches);
                Ok(GreedyDecodeChunk {
                    steps: vec![
                        GreedyDecodeStep {
                            top_k: case.decode_top.clone(),
                            kv_caches: full.take().unwrap(),
                        },
                        GreedyDecodeStep {
                            top_k: vec![TopKEntry {
                                index: DECODE_TOKEN + 1,
                                value: 7.0,
                            }],
                            kv_caches: extra.take().unwrap(),
                        },
                    ],
                })
            },
        );
    assert_eq!(calls, 1);
    let error: CpuRefError = match result {
        Err(error) => error,
        Ok(_) => panic!("over-returned official chunk must be rejected"),
    };
    assert_eq!(error.code(), CpuRefErrorCode::DimensionMismatch);
}
