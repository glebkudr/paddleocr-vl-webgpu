//! M6c final-norm-to-logits/top-k evidence only. This target deliberately
//! does not execute the 18-layer stack, generation, or decode chunks.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

use pvlc_cpu_ref::{
    TopKEntry, pinned_chunked_lm_head_top_k_f32, pinned_prefill_last_logits_f32, top_k,
};
use pvlc_safetensors::SafetensorsCatalog;
use pvlc_testkit::{
    ComparisonAxes, ComparisonPolicy, ComparisonReport, compare_f32, compare_logits,
};

const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const MODEL_LOCK_BYTES: usize = 2_385;
const MODEL_LOCK_BLAKE3: &str = "c2ffa46821fdda1a7e4cd6bad63cdcdc2f775b8fed250b0679cc7d5411577d10";
const MODEL_FILE_BYTES: u64 = 1_917_255_968;
const MODEL_FILE_BLAKE3: &str = "4dc3ab13685a0c0a701f77f9c5ebafdc0004074247e00bde6b7e7b04279a41fc";
const MODEL_TENSOR_COUNT: usize = 620;
const FIXTURE_BYTES: usize = 6_438_376;
const FIXTURE_BLAKE3: &str = "386735089b35b2a1fc50ad578678689eed98b3e62f2a0c18d5e2890d0e6a8ebf";
const FIXTURE_TENSOR_COUNT: usize = 44;
const FIXTURE_METADATA_COUNT: usize = 38;
const HIDDEN_SIZE: usize = 1_024;
const VOCAB_SIZE: usize = 103_424;
const TOKENS: usize = 1;
const FINAL_NORM: &str = "decoder.decode.00.final_norm";
const LOGITS: &str = "decoder.decode.00.logits";
const LM_HEAD_WEIGHT: &str = "lm_head.weight";
const FINAL_NORM_RAW_BLAKE3: &str =
    "16cc97ce46e4839948e6425bcc444d333cac45c0425a88297e29518ca796f6e7";
const LOGITS_RAW_BLAKE3: &str = "68b42601be700f15701546b3b9e5c011da5e528b87dc825e01051d56068b5dbf";
const LM_HEAD_RAW_BLAKE3: &str = "784ffd4944c3b72292fa62a8f6044485aef55be16479ac7946eaf0e7ba3e08dc";
const ACCEPTED_LOGITS_F32_BLAKE3: &str =
    "8225e50495271f11291079e2b9fbb08a7f5bf4b0fe129fb45d67d62bcf4e3fe3";
const MODEL_LOCK: &str = include_str!("../../../models/paddleocr-vl-1.6.lock");

const REFERENCE_TOP_32: [usize; 32] = [
    898, 820, 1_352, 48_798, 93_937, 1_924, 2_987, 40_930, 41_630, 6_188, 613, 1_502, 1_096,
    11_706, 790, 17_610, 2_131, 763, 84_289, 13_001, 2, 245, 8_388, 23, 1_470, 23_259, 3_448,
    6_777, 719, 2_015, 8_504, 16_487,
];

const REFERENCE_TOP_32_VALUES: [f32; 32] = [
    8.5, 8.1875, 7.40625, 6.71875, 6.71875, 6.5625, 6.4375, 6.09375, 6.03125, 5.96875, 5.9375,
    5.875, 5.84375, 5.75, 5.71875, 5.59375, 5.53125, 5.5, 5.5, 5.46875, 5.1875, 5.125, 5.03125,
    5.0, 4.96875, 4.875, 4.84375, 4.8125, 4.75, 4.6875, 4.6875, 4.625,
];

const ACCEPTED_TOP_32: [usize; 32] = [
    898, 820, 1_352, 48_798, 93_937, 1_924, 2_987, 40_930, 41_630, 6_188, 613, 1_502, 1_096,
    11_706, 790, 17_610, 2_131, 84_289, 763, 13_001, 2, 245, 8_388, 23, 1_470, 23_259, 3_448,
    6_777, 719, 2_015, 8_504, 16_487,
];

const ACCEPTED_TOP_32_VALUE_BITS: [u32; 32] = [
    1_091_028_468,
    1_090_738_227,
    1_089_294_351,
    1_087_858_869,
    1_087_819_439,
    1_087_526_765,
    1_087_235_533,
    1_086_549_950,
    1_086_393_454,
    1_086_235_275,
    1_086_172_151,
    1_086_059_710,
    1_085_992_235,
    1_085_772_767,
    1_085_705_945,
    1_085_505_371,
    1_085_338_421,
    1_085_274_008,
    1_085_254_960,
    1_085_180_023,
    1_084_641_246,
    1_084_514_834,
    1_084_301_545,
    1_084_205_403,
    1_084_130_692,
    1_083_976_940,
    1_083_923_592,
    1_083_824_694,
    1_083_695_791,
    1_083_575_449,
    1_083_565_058,
    1_083_417_193,
];

// Calibration observed max/mean/p99/relative-L2
// .031193733215/.004949844414/.015244007111/.001664192272. Every bound
// below is rounded upward beyond 1.25x the observation; 1-cosine has the
// same headroom rule.
const OFFICIAL_POLICY: ComparisonPolicy = ComparisonPolicy {
    require_finite: true,
    max_abs: 0.0390,
    max_mean_abs: 0.00619,
    max_p99_abs: 0.0191,
    max_relative_l2: 0.00209,
    min_cosine_similarity: 0.999_998_2,
    max_per_token_relative_l2: Some(0.00209),
    max_per_channel_relative_l2: None,
};
const MAX_KL_REFERENCE_TO_CANDIDATE: f64 = 0.000_119;
const MAX_JENSEN_SHANNON_DIVERGENCE: f64 = 0.000_029_8;
const RELEASE_CHUNK_SIZES: [usize; 8] = [1, 31, 32, 33, 1_024, 103_423, 103_424, 103_425];

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/decoder-decode-official-v1.safetensors")
}

fn release_model_path() -> PathBuf {
    assert_eq!(
        std::env::var("PVLC_REQUIRE_MODEL").as_deref(),
        Ok("1"),
        "the ignored M6 logits/top-k release gate must run with PVLC_REQUIRE_MODEL=1"
    );
    let path = repository()
        .join("models/snapshots")
        .join(MODEL_REVISION)
        .join("model.safetensors");
    assert!(
        path.is_file(),
        "pinned checkpoint is absent at {}",
        path.display()
    );
    path
}

fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn hash_file(path: &Path) -> String {
    let mut source = File::open(path).unwrap();
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; 128 * 1_024];
    loop {
        let read = source.read(&mut buffer).unwrap();
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    hasher.finalize().to_hex().to_string()
}

fn hash_f32(values: &[f32]) -> String {
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

fn compare_official(expected: &[f32], actual: &[f32]) -> ComparisonReport {
    compare_f32(
        expected,
        actual,
        &[1, VOCAB_SIZE],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap()
}

fn metadata_map(catalog: &SafetensorsCatalog) -> BTreeMap<String, String> {
    catalog
        .metadata()
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn assert_model_lock_identity() {
    assert_eq!(MODEL_LOCK.len(), MODEL_LOCK_BYTES);
    assert_eq!(hash_bytes(MODEL_LOCK.as_bytes()), MODEL_LOCK_BLAKE3);

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
}

fn row_reversed_lm_head_weight(weights: &[f32]) -> Vec<f32> {
    let mut reversed = weights.to_vec();
    for row in reversed.chunks_exact_mut(HIDDEN_SIZE) {
        row.reverse();
    }
    reversed
}

fn assert_top_32(entries: &[TopKEntry]) {
    assert_eq!(entries.len(), REFERENCE_TOP_32.len());
    assert_eq!(
        entries.iter().map(|entry| entry.index).collect::<Vec<_>>(),
        REFERENCE_TOP_32
    );
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.value.to_bits())
            .collect::<Vec<_>>(),
        REFERENCE_TOP_32_VALUES
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn authenticates_decode_logits_fixture_and_frozen_top_32_without_loading_model_payload() {
    assert_model_lock_identity();

    let path = fixture_path();
    let bytes = fs::read(&path).unwrap();
    assert_eq!(bytes.len(), FIXTURE_BYTES);
    assert_eq!(hash_bytes(&bytes), FIXTURE_BLAKE3);

    let fixture = SafetensorsCatalog::open(path).unwrap();
    assert_eq!(fixture.tensors().len(), FIXTURE_TENSOR_COUNT);
    assert_eq!(fixture.metadata().len(), FIXTURE_METADATA_COUNT);

    let metadata = metadata_map(&fixture);
    assert_eq!(
        metadata.get("fixture_schema").map(String::as_str),
        Some("pvlc.decoder_decode.official.v1")
    );
    assert_eq!(
        metadata.get("generated_tokens").map(String::as_str),
        Some("94013,898")
    );
    assert_eq!(
        metadata.get("decode_input_token").map(String::as_str),
        Some("94013")
    );
    assert_eq!(
        metadata.get("decode_next_token").map(String::as_str),
        Some("898")
    );
    assert_eq!(
        metadata.get("model_revision").map(String::as_str),
        Some(MODEL_REVISION)
    );

    let final_norm_tensor = fixture.tensor(FINAL_NORM).unwrap();
    assert_eq!(final_norm_tensor.shape, [1, HIDDEN_SIZE as u64]);
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
    assert_eq!(final_norm.len(), HIDDEN_SIZE);
    assert_eq!(logits.len(), VOCAB_SIZE);
    assert!(
        final_norm
            .iter()
            .chain(&logits)
            .all(|value| value.is_finite())
    );

    let top = top_k(&logits, 32).unwrap();
    assert_top_32(&top);
    assert_eq!(top[0].index, 898);
    assert_eq!(top[0].value, 8.5);
    assert_eq!(top[1].value, 8.1875);
    assert_eq!(top[0].value - top[1].value, 0.3125);
}

#[test]
#[ignore = "release-only official decode logits/top-k gate with pinned 1.9 GB checkpoint"]
fn full_and_chunked_lm_head_match_decode_logits_and_reject_wrong_inputs() {
    let model_path = release_model_path();
    assert_eq!(fs::metadata(&model_path).unwrap().len(), MODEL_FILE_BYTES);
    assert_eq!(hash_file(&model_path), MODEL_FILE_BLAKE3);

    let fixture_path = fixture_path();
    assert_eq!(
        fs::metadata(&fixture_path).unwrap().len(),
        FIXTURE_BYTES as u64
    );
    assert_eq!(hash_file(&fixture_path), FIXTURE_BLAKE3);

    let fixture = SafetensorsCatalog::open(fixture_path).unwrap();
    let model = SafetensorsCatalog::open(&model_path).unwrap();
    assert_eq!(model.tensors().len(), MODEL_TENSOR_COUNT);
    let lm_head_tensor = model.tensor(LM_HEAD_WEIGHT).unwrap();
    assert_eq!(
        lm_head_tensor.shape,
        [VOCAB_SIZE as u64, HIDDEN_SIZE as u64]
    );
    assert_eq!(lm_head_tensor.dtype.safetensors_name(), "BF16");
    assert_eq!(
        blake3::hash(&tensor_bytes(&model, LM_HEAD_WEIGHT))
            .to_hex()
            .to_string(),
        LM_HEAD_RAW_BLAKE3
    );

    let final_norm = fixture.load_tensor_f32(FINAL_NORM).unwrap();
    let captured = fixture.load_tensor_f32(LOGITS).unwrap();
    let lm_head_weight = model.load_tensor_f32(LM_HEAD_WEIGHT).unwrap();
    assert_eq!(final_norm.len(), TOKENS * HIDDEN_SIZE);
    assert_eq!(captured.len(), VOCAB_SIZE);
    assert_eq!(lm_head_weight.len(), VOCAB_SIZE * HIDDEN_SIZE);
    let preserved_final_norm = hash_f32(&final_norm);
    let preserved_weight = hash_f32(&lm_head_weight);

    let started = Instant::now();
    let full_logits = pinned_prefill_last_logits_f32(&final_norm, TOKENS, &lm_head_weight).unwrap();
    let runtime = started.elapsed();
    let report = compare_official(&captured, &full_logits);
    let topology = compare_logits(&captured, &full_logits, 32).unwrap();
    let full_top = top_k(&full_logits, 32).unwrap();
    let (reversed_report, reversed_topology) = {
        let reversed_weight = row_reversed_lm_head_weight(&lm_head_weight);
        let reversed =
            pinned_prefill_last_logits_f32(&final_norm, TOKENS, &reversed_weight).unwrap();
        let reversed_top = top_k(&reversed, REFERENCE_TOP_32.len()).unwrap();
        let reversed_chunked = pinned_chunked_lm_head_top_k_f32(
            &final_norm,
            REFERENCE_TOP_32.len(),
            32,
            &reversed_weight,
        )
        .unwrap();
        assert_eq!(reversed_chunked, reversed_top);
        let report = compare_official(&captured, &reversed);
        let topology = compare_logits(&captured, &reversed, 32).unwrap();
        drop(reversed_weight);
        (report, topology)
    };
    let wrong_final_norm = vec![0.0_f32; HIDDEN_SIZE];
    let wrong_final_norm_logits =
        pinned_prefill_last_logits_f32(&wrong_final_norm, TOKENS, &lm_head_weight).unwrap();
    let wrong_final_norm_top = top_k(&wrong_final_norm_logits, REFERENCE_TOP_32.len()).unwrap();
    let wrong_final_norm_chunked = pinned_chunked_lm_head_top_k_f32(
        &wrong_final_norm,
        REFERENCE_TOP_32.len(),
        32,
        &lm_head_weight,
    )
    .unwrap();
    assert_eq!(wrong_final_norm_chunked, wrong_final_norm_top);
    let wrong_final_norm_report = compare_official(&captured, &wrong_final_norm_logits);
    let wrong_final_norm_topology =
        compare_logits(&captured, &wrong_final_norm_logits, 32).unwrap();

    assert_eq!(full_logits.len(), VOCAB_SIZE);
    assert!(full_logits.iter().all(|value| value.is_finite()));
    assert_eq!(hash_f32(&full_logits), ACCEPTED_LOGITS_F32_BLAKE3);
    assert_eq!(
        full_top.iter().map(|entry| entry.index).collect::<Vec<_>>(),
        ACCEPTED_TOP_32
    );
    assert_eq!(
        full_top
            .iter()
            .map(|entry| entry.value.to_bits())
            .collect::<Vec<_>>(),
        ACCEPTED_TOP_32_VALUE_BITS
    );

    let verdict = report.assess(&OFFICIAL_POLICY).unwrap();
    assert!(verdict.passed(), "{report:#?}\n{verdict:#?}");
    assert_eq!(topology.reference_top1, 898);
    assert_eq!(topology.candidate_top1, 898);
    assert!(topology.top1_agreement);
    assert_eq!(topology.reference_top_k, REFERENCE_TOP_32);
    assert_eq!(topology.candidate_top_k, ACCEPTED_TOP_32);
    assert_eq!(topology.top_k_overlap, 32);
    assert_eq!(topology.top_k_overlap_fraction, 1.0);
    assert_eq!(topology.reference_margin, 0.3125);
    assert!(
        topology.kl_reference_to_candidate <= MAX_KL_REFERENCE_TO_CANDIDATE,
        "{topology:#?}"
    );
    assert!(
        topology.jensen_shannon_divergence <= MAX_JENSEN_SHANNON_DIVERGENCE,
        "{topology:#?}"
    );
    assert!(!reversed_report.assess(&OFFICIAL_POLICY).unwrap().passed());
    assert_eq!(reversed_topology.candidate_top1, 96_249);
    assert_eq!(reversed_topology.top_k_overlap, 1);
    assert!(reversed_report.relative_l2 > 0.83);
    assert!(
        !wrong_final_norm_report
            .assess(&OFFICIAL_POLICY)
            .unwrap()
            .passed()
    );
    assert!(wrong_final_norm_report.relative_l2 > 0.99);
    assert_eq!(wrong_final_norm_topology.candidate_top1, 0);
    assert_eq!(wrong_final_norm_topology.top_k_overlap, 2);

    for chunk_size in RELEASE_CHUNK_SIZES {
        let chunked = pinned_chunked_lm_head_top_k_f32(
            &final_norm,
            REFERENCE_TOP_32.len(),
            chunk_size,
            &lm_head_weight,
        )
        .unwrap();
        assert_eq!(chunked, full_top, "chunk_size={chunk_size}");
    }

    assert_eq!(hash_f32(&final_norm), preserved_final_norm);
    assert_eq!(hash_f32(&lm_head_weight), preserved_weight);

    eprintln!(
        "M6 logits accepted runtime={runtime:?} logits={} max={:.12} mean={:.12} p99={:.12} rel_l2={:.12} cosine={:.12} top1_ref={} top1_full={} overlap={}/32 margin={:.12} kl={:.12} js={:.12}",
        hash_f32(&full_logits),
        report.max_abs,
        report.mean_abs,
        report.p99_abs,
        report.relative_l2,
        report.cosine_similarity,
        topology.reference_top1,
        topology.candidate_top1,
        topology.top_k_overlap,
        topology.reference_margin,
        topology.kl_reference_to_candidate,
        topology.jensen_shannon_divergence,
    );
    eprintln!(
        "M6 logits full top32 idx={:?} bits={:?}",
        full_top.iter().map(|entry| entry.index).collect::<Vec<_>>(),
        full_top
            .iter()
            .map(|entry| entry.value.to_bits())
            .collect::<Vec<_>>(),
    );
    eprintln!(
        "M6 logits reversed rel_l2={:.12} cosine={:.12} top1={} overlap={}/32",
        reversed_report.relative_l2,
        reversed_report.cosine_similarity,
        reversed_topology.candidate_top1,
        reversed_topology.top_k_overlap,
    );
    eprintln!(
        "M6 logits wrong-final-norm rel_l2={:.12} top1={} overlap={}/32",
        wrong_final_norm_report.relative_l2,
        wrong_final_norm_topology.candidate_top1,
        wrong_final_norm_topology.top_k_overlap,
    );
}
