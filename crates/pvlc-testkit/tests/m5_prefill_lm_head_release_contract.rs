use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

use pvlc_cpu_ref::pinned_prefill_last_logits_f32;
use pvlc_safetensors::SafetensorsCatalog;
use pvlc_testkit::{
    ComparisonAxes, ComparisonPolicy, ComparisonReport, compare_f32, compare_logits,
};

const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const MODEL_FILE_BYTES: u64 = 1_917_255_968;
const MODEL_FILE_BLAKE3: &str = "4dc3ab13685a0c0a701f77f9c5ebafdc0004074247e00bde6b7e7b04279a41fc";
const LM_HEAD_RAW_BLAKE3: &str = "784ffd4944c3b72292fa62a8f6044485aef55be16479ac7946eaf0e7ba3e08dc";
const FIXTURE_BYTES: usize = 887_680;
const FIXTURE_BLAKE3: &str = "f73291c81f8251715bf7ec64a749338c928930add8406845463705ca1468334d";
const ACCEPTED_LOGITS_F32_BLAKE3: &str =
    "a0a1708214355e7d4e43193599ac2c0cc11b4e7f4e2abd8fd5c01e7a2185bc92";
const TOKENS: usize = 332;
const HIDDEN_SIZE: usize = 1_024;
const VOCAB_SIZE: usize = 103_424;
const FINAL_NORM: &str = "decoder.final_norm";
const LOGITS: &str = "decoder.prefill.logits.last";
const LM_HEAD_WEIGHT: &str = "lm_head.weight";

const REFERENCE_TOP_32: [usize; 32] = [
    94_013, 93_992, 820, 13_476, 1_369, 5_859, 1_704, 93_965, 588, 763, 6_934, 93_969, 1_315,
    93_966, 93_971, 1_502, 93_955, 93_967, 5_083, 93_982, 2_668, 9_063, 93_970, 729, 3_448, 707,
    93_957, 93_987, 93_981, 4_373, 19_280, 36_561,
];
const ACCEPTED_TOP_32: [usize; 32] = [
    94_013, 93_992, 820, 13_476, 5_859, 1_369, 1_704, 93_965, 588, 763, 6_934, 93_969, 93_971,
    1_315, 93_966, 1_502, 93_955, 93_967, 93_982, 5_083, 2_668, 9_063, 93_970, 729, 3_448, 707,
    93_957, 93_987, 93_981, 4_373, 36_561, 19_280,
];

// Preproduction accepted-linear calibration observed max/mean/p99/relative-L2
// .03083229065/.004353132782/.01477479935/.001632113768. Each bound rounds
// upward beyond 1.25x the observation; 1-cosine has equivalent headroom.
const OFFICIAL_POLICY: ComparisonPolicy = ComparisonPolicy {
    require_finite: true,
    max_abs: 0.0386,
    max_mean_abs: 0.00545,
    max_p99_abs: 0.0185,
    max_relative_l2: 0.00205,
    min_cosine_similarity: 0.999_998_3,
    max_per_token_relative_l2: Some(0.00205),
    max_per_channel_relative_l2: None,
};
const MAX_KL_REFERENCE_TO_CANDIDATE: f64 = 0.000_047_6;
const MAX_JENSEN_SHANNON_DIVERGENCE: f64 = 0.000_011_9;

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/prefill-lm-head-official-v1.safetensors")
}

fn release_model_path() -> PathBuf {
    assert_eq!(
        std::env::var("PVLC_REQUIRE_MODEL").as_deref(),
        Ok("1"),
        "the ignored M5d release gate must run with PVLC_REQUIRE_MODEL=1"
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

fn reversed_hidden_weight_logits(input: &[f32], weights: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; VOCAB_SIZE];
    for (token_id, value) in output.iter_mut().enumerate() {
        let weight = &weights[token_id * HIDDEN_SIZE..(token_id + 1) * HIDDEN_SIZE];
        let mut accumulator = 0.0_f32;
        for hidden_index in 0..HIDDEN_SIZE {
            accumulator += input[hidden_index] * weight[HIDDEN_SIZE - 1 - hidden_index];
        }
        *value = accumulator;
    }
    output
}

#[test]
#[ignore = "release-only pinned 103424 x 1024 LM head with 1.9 GB checkpoint"]
fn full_official_prefill_lm_head_matches_logits_and_rejects_wrong_token_or_layout() {
    let model_path = release_model_path();
    assert_eq!(fs::metadata(&model_path).unwrap().len(), MODEL_FILE_BYTES);
    assert_eq!(hash_file(&model_path), MODEL_FILE_BLAKE3);
    let fixture_path = fixture_path();
    assert_eq!(
        fs::metadata(&fixture_path).unwrap().len(),
        FIXTURE_BYTES as u64
    );
    assert_eq!(hash_file(&fixture_path), FIXTURE_BLAKE3);

    let model = SafetensorsCatalog::open(&model_path).unwrap();
    assert_eq!(model.tensors().len(), 620);
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

    let fixture = SafetensorsCatalog::open(fixture_path).unwrap();
    let final_norm = fixture.load_tensor_f32(FINAL_NORM).unwrap();
    let expected = fixture.load_tensor_f32(LOGITS).unwrap();
    let lm_head_weight = model.load_tensor_f32(LM_HEAD_WEIGHT).unwrap();
    assert_eq!(final_norm.len(), TOKENS * HIDDEN_SIZE);
    assert_eq!(expected.len(), VOCAB_SIZE);
    assert_eq!(lm_head_weight.len(), VOCAB_SIZE * HIDDEN_SIZE);
    let preserved_final_norm = hash_f32(&final_norm);
    let preserved_lm_head_weight = hash_f32(&lm_head_weight);

    let started = Instant::now();
    let actual = pinned_prefill_last_logits_f32(&final_norm, TOKENS, &lm_head_weight).unwrap();
    let accepted_runtime = started.elapsed();
    assert_eq!(actual.len(), VOCAB_SIZE);
    assert!(actual.iter().all(|value: &f32| value.is_finite()));
    assert_eq!(hash_f32(&actual), ACCEPTED_LOGITS_F32_BLAKE3);

    let report = compare_official(&expected, &actual);
    let verdict = report.assess(&OFFICIAL_POLICY).unwrap();
    assert!(verdict.passed(), "{report:#?}\n{verdict:#?}");
    let topology = compare_logits(&expected, &actual, 32).unwrap();
    assert_eq!(topology.reference_top1, 94_013);
    assert_eq!(topology.candidate_top1, 94_013);
    assert!(topology.top1_agreement);
    assert_eq!(topology.reference_top_k, REFERENCE_TOP_32);
    assert_eq!(topology.candidate_top_k, ACCEPTED_TOP_32);
    assert_eq!(topology.top_k_overlap, 32);
    assert_eq!(topology.top_k_overlap_fraction, 1.0);
    assert_eq!(topology.reference_margin, 1.25);
    assert!(
        topology.kl_reference_to_candidate <= MAX_KL_REFERENCE_TO_CANDIDATE,
        "{topology:#?}"
    );
    assert!(
        topology.jensen_shannon_divergence <= MAX_JENSEN_SHANNON_DIVERGENCE,
        "{topology:#?}"
    );

    let mut wrong_final_norm = final_norm.clone();
    let previous_start = (TOKENS - 2) * HIDDEN_SIZE;
    let last_start = (TOKENS - 1) * HIDDEN_SIZE;
    wrong_final_norm.copy_within(previous_start..last_start, last_start);
    let wrong_last =
        pinned_prefill_last_logits_f32(&wrong_final_norm, TOKENS, &lm_head_weight).unwrap();
    let wrong_last_report = compare_official(&expected, &wrong_last);
    assert!(!wrong_last_report.assess(&OFFICIAL_POLICY).unwrap().passed());
    let wrong_last_topology = compare_logits(&expected, &wrong_last, 32).unwrap();
    assert_eq!(wrong_last_topology.top_k_overlap, 19);
    assert!(wrong_last_report.relative_l2 > 0.30);

    let wrong_layout = reversed_hidden_weight_logits(&final_norm[last_start..], &lm_head_weight);
    let wrong_layout_report = compare_official(&expected, &wrong_layout);
    assert!(
        !wrong_layout_report
            .assess(&OFFICIAL_POLICY)
            .unwrap()
            .passed()
    );
    let wrong_layout_topology = compare_logits(&expected, &wrong_layout, 32).unwrap();
    assert_eq!(wrong_layout_topology.candidate_top1, 94_077);
    assert_eq!(wrong_layout_topology.top_k_overlap, 0);
    assert!(wrong_layout_report.relative_l2 > 0.74);

    assert_eq!(hash_f32(&final_norm), preserved_final_norm);
    assert_eq!(hash_f32(&lm_head_weight), preserved_lm_head_weight);
    eprintln!(
        "M5d accepted runtime={accepted_runtime:?} max={:.9} mean={:.9} p99={:.9} rel_l2={:.9} cosine={:.9} top32={}/32 margin={:.3} fixture={} output={}",
        report.max_abs,
        report.mean_abs,
        report.p99_abs,
        report.relative_l2,
        report.cosine_similarity,
        topology.top_k_overlap,
        topology.reference_margin,
        FIXTURE_BLAKE3,
        ACCEPTED_LOGITS_F32_BLAKE3,
    );
    eprintln!(
        "M5d negatives previous-row rel_l2={:.6} overlap={}/32; reversed-weight rel_l2={:.6} top1={} overlap={}/32",
        wrong_last_report.relative_l2,
        wrong_last_topology.top_k_overlap,
        wrong_layout_report.relative_l2,
        wrong_layout_topology.candidate_top1,
        wrong_layout_topology.top_k_overlap,
    );
}
