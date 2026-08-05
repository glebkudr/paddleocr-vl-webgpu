use std::{
    fs,
    path::{Path, PathBuf},
};

use pvlc_cpu_ref::top_k;
use pvlc_safetensors::SafetensorsCatalog;

const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const MODEL_FILE_BYTES: u64 = 1_917_255_968;
const MODEL_FILE_BLAKE3: &str = "4dc3ab13685a0c0a701f77f9c5ebafdc0004074247e00bde6b7e7b04279a41fc";
const FIXTURE_BYTES: usize = 887_680;
const FIXTURE_BLAKE3: &str = "f73291c81f8251715bf7ec64a749338c928930add8406845463705ca1468334d";
const FINAL_NORM_RAW_BLAKE3: &str =
    "2a69c20f24b2a517170611a29cb71fa46a0c3e8ee758805031d3fe9ea2318ac9";
const LOGITS_RAW_BLAKE3: &str = "d661fb880ccfcc073581609745c19bc512d37b67a79430f8072449a762288b8b";
const TOKENS: usize = 332;
const HIDDEN_SIZE: usize = 1_024;
const VOCAB_SIZE: usize = 103_424;
const FINAL_NORM: &str = "decoder.final_norm";
const LOGITS: &str = "decoder.prefill.logits.last";
const MODEL_LOCK: &str = include_str!("../../../models/paddleocr-vl-1.6.lock");
const GOLDEN_LOCK: &str = include_str!("../../../goldens/golden.lock");

const REFERENCE_TOP_32: [usize; 32] = [
    94_013, 93_992, 820, 13_476, 1_369, 5_859, 1_704, 93_965, 588, 763, 6_934, 93_969, 1_315,
    93_966, 93_971, 1_502, 93_955, 93_967, 5_083, 93_982, 2_668, 9_063, 93_970, 729, 3_448, 707,
    93_957, 93_987, 93_981, 4_373, 19_280, 36_561,
];

#[rustfmt::skip]
const METADATA: [(&str, &str); 19] = [
    ("bias", "false"),
    ("case_id", "ocr.clean_latin.0001"),
    ("decoded_text", "JUL"),
    ("device", "mps"),
    ("dtype", "bfloat16"),
    ("final_norm_raw_blake3", FINAL_NORM_RAW_BLAKE3),
    ("fixture_schema", "pvlc.prefill_lm_head.official.v1"),
    ("generated_tokens", "94013,898"),
    ("hidden_size", "1024"),
    ("lm_head_layout", "output-major[vocab,hidden]"),
    ("logits_raw_blake3", LOGITS_RAW_BLAKE3),
    ("model_id", "PaddlePaddle/PaddleOCR-VL-1.6"),
    ("model_revision", MODEL_REVISION),
    ("oracle", "TransformersOracle pinned remote code"),
    ("prefill_top1", "94013"),
    ("selected_token", "last"),
    ("tokens", "332"),
    ("trace_level", "L3"),
    ("vocab_size", "103424"),
];

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/prefill-lm-head-official-v1.safetensors")
}

fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn tensor_bytes(catalog: &SafetensorsCatalog, name: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    catalog.copy_tensor_to(name, &mut bytes).unwrap();
    bytes
}

fn assert_lock_identity() {
    let model: toml::Value = toml::from_str(MODEL_LOCK).unwrap();
    assert_eq!(model["format_version"].as_integer(), Some(1));
    assert_eq!(model["compiler_model_abi"].as_integer(), Some(1));
    assert_eq!(
        model["model_id"].as_str(),
        Some("PaddlePaddle/PaddleOCR-VL-1.6")
    );
    assert_eq!(model["revision"].as_str(), Some(MODEL_REVISION));
    let model_file = model["files"]["model.safetensors"].as_table().unwrap();
    assert_eq!(model_file.len(), 2);
    assert_eq!(model_file["blake3"].as_str(), Some(MODEL_FILE_BLAKE3));
    assert_eq!(
        model_file["size"].as_integer(),
        Some(MODEL_FILE_BYTES as i64)
    );

    let golden: toml::Value = toml::from_str(GOLDEN_LOCK).unwrap();
    assert_eq!(golden["model_revision"].as_str(), Some(MODEL_REVISION));
    let matching = golden["bundles"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|bundle| {
            bundle["case_id"].as_str() == Some("ocr.clean_latin.0001")
                && bundle["trace_level"].as_str() == Some("L3")
        })
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1);
    assert_eq!(matching[0]["decoded_text"].as_str(), Some("JUL"));
    assert_eq!(
        matching[0]["generated_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|token| token.as_integer().unwrap())
            .collect::<Vec<_>>(),
        [94_013, 898]
    );
}

#[test]
fn authenticates_canonical_official_prefill_fixture_and_source_anchors() {
    assert_lock_identity();
    let path = fixture_path();
    let bytes = fs::read(&path).unwrap();
    assert_eq!(bytes.len(), FIXTURE_BYTES);
    assert_eq!(hash_bytes(&bytes), FIXTURE_BLAKE3);

    let fixture = SafetensorsCatalog::open(path).unwrap();
    let metadata = fixture
        .metadata()
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(metadata, METADATA);
    assert_eq!(fixture.tensors().len(), 2);

    let final_norm_tensor = fixture.tensor(FINAL_NORM).unwrap();
    assert_eq!(
        final_norm_tensor.shape,
        [1, TOKENS as u64, HIDDEN_SIZE as u64]
    );
    assert_eq!(final_norm_tensor.dtype.safetensors_name(), "BF16");
    assert_eq!(
        hash_bytes(&tensor_bytes(&fixture, FINAL_NORM)),
        FINAL_NORM_RAW_BLAKE3
    );
    let logits_tensor = fixture.tensor(LOGITS).unwrap();
    assert_eq!(logits_tensor.shape, [1, VOCAB_SIZE as u64]);
    assert_eq!(logits_tensor.dtype.safetensors_name(), "BF16");
    assert_eq!(
        hash_bytes(&tensor_bytes(&fixture, LOGITS)),
        LOGITS_RAW_BLAKE3
    );

    let final_norm = fixture.load_tensor_f32(FINAL_NORM).unwrap();
    let logits = fixture.load_tensor_f32(LOGITS).unwrap();
    assert_eq!(final_norm.len(), TOKENS * HIDDEN_SIZE);
    assert_eq!(logits.len(), VOCAB_SIZE);
    assert_eq!(final_norm[0], 3.515625);
    assert_eq!(logits[0], -3.3125);
    assert!(
        final_norm
            .iter()
            .chain(&logits)
            .all(|value| value.is_finite())
    );

    let top = top_k(&logits, 32).unwrap();
    assert_eq!(
        top.iter().map(|entry| entry.index).collect::<Vec<_>>(),
        REFERENCE_TOP_32
    );
    assert_eq!(top[0].index, 94_013);
    assert_eq!(top[0].value, 8.75);
    assert_eq!(top[1].value, 7.5);
    assert_eq!(top[0].value - top[1].value, 1.25);
}
