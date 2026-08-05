use std::{
    cell::{Cell, RefCell},
    collections::HashSet,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};

use pvlc_cpu_ref::{
    CpuRefError, DecoderLayerConfig, DecoderLayerParameters, DecoderLayerPrefillTrace,
    DecoderStackConfig, DecoderStackPrefillTrace, decoder_layer_prefill_f32,
    decoder_stack_prefill_f32, pinned_decoder_stack_prefill_f32, pinned_prefill_last_logits_f32,
    rms_norm_f32,
};
use pvlc_safetensors::SafetensorsCatalog;
use pvlc_testkit::{
    ComparisonAxes, ComparisonPolicy, ComparisonReport, compare_f32, compare_logits,
};

const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const MODEL_FILE_BYTES: u64 = 1_917_255_968;
const MODEL_FILE_BLAKE3: &str = "4dc3ab13685a0c0a701f77f9c5ebafdc0004074247e00bde6b7e7b04279a41fc";
const FIXTURE_BYTES: usize = 14_113_480;
const FIXTURE_BLAKE3: &str = "c109097d10f26f33ac48699d152fad483e9d0fd1c726535ef3d70024b820c4b4";
const LOGITS_FIXTURE_BYTES: usize = 887_680;
const LOGITS_FIXTURE_BLAKE3: &str =
    "f73291c81f8251715bf7ec64a749338c928930add8406845463705ca1468334d";
const LOGITS_FINAL_NORM_RAW_BLAKE3: &str =
    "2a69c20f24b2a517170611a29cb71fa46a0c3e8ee758805031d3fe9ea2318ac9";
const LOGITS_RAW_BLAKE3: &str = "d661fb880ccfcc073581609745c19bc512d37b67a79430f8072449a762288b8b";
const LM_HEAD_RAW_BLAKE3: &str = "784ffd4944c3b72292fa62a8f6044485aef55be16479ac7946eaf0e7ba3e08dc";
const CAPTURED_LOGITS_F32_BLAKE3: &str =
    "a0a1708214355e7d4e43193599ac2c0cc11b4e7f4e2abd8fd5c01e7a2185bc92";
const INTEGRATED_LOGITS_F32_BLAKE3: &str =
    "9a3a46516a17903b9e931005fe2253b787454b3a1bd6bdb3aa3f105f09bbd01d";
const SKIPPED_FINAL_NORM_LOGITS_F32_BLAKE3: &str =
    "d0fea64389734e83487674e5ecf41d3dda14c69797c957aa01a55d85c1f6f638";
const TOKENS: usize = 332;
const LAYERS: usize = 18;
const HIDDEN_SIZE: usize = 1_024;
const VOCAB_SIZE: usize = 103_424;
const INTERMEDIATE_SIZE: usize = 3_072;
const QUERY_HEADS: usize = 16;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const KEY_VALUE_WIDTH: usize = KEY_VALUE_HEADS * HEAD_DIM;
const RMS_NORM_EPSILON: f32 = 1.0e-5;
const MROPE_SECTIONS: [usize; 3] = [16, 24, 24];
const INPUT: &str = "decoder.layer.00.input";
const RAW_COS: &str = "decoder.rope.cos.axis_major";
const RAW_SIN: &str = "decoder.rope.sin.axis_major";
const FINAL_NORM: &str = "decoder.final_norm";
const FINAL_NORM_WEIGHT: &str = "model.norm.weight";
const LAST_LOGITS: &str = "decoder.prefill.logits.last";
const LM_HEAD_WEIGHT: &str = "lm_head.weight";
const GOLDEN_LOCK: &str = include_str!("../../../goldens/golden.lock");
const MODEL_LOCK: &str = include_str!("../../../models/paddleocr-vl-1.6.lock");

struct TensorSpec {
    name: &'static str,
    shape: &'static [u64],
    raw_blake3: &'static str,
}

// Provenance: all captured tensors below came from the live pinned remote-code
// TransformersOracle.capture_artifacts MPS/BF16 L3 path exercised in
// tools/reference_capture/tests/test_transformers_oracle_integration.py:136+.
// Before this fixture was written, each captured raw payload was checked
// against the complete decoder span of literal expected_value_anchors at lines
// 306-462 and the semantic decoder checks at lines 647-685. The final norm weight is the
// exact BF16 model.norm.weight payload from the pinned model.safetensors.
#[rustfmt::skip]
const TENSORS: [TensorSpec; 23] = [
    TensorSpec { name: FINAL_NORM, shape: &[332, 1024], raw_blake3: "2a69c20f24b2a517170611a29cb71fa46a0c3e8ee758805031d3fe9ea2318ac9" },
    TensorSpec { name: INPUT, shape: &[332, 1024], raw_blake3: "8b46524fa1d413be6ee140b8af80c18547c3d505fd8f61c9781962e957f2da52" },
    TensorSpec { name: "decoder.layer.00.output", shape: &[332, 1024], raw_blake3: "7130fabeb187b3b9dc463fa0aea6c39775a674ae6497513f0b48b0216c9cac6e" },
    TensorSpec { name: "decoder.layer.01.output", shape: &[332, 1024], raw_blake3: "616ae03b02d5cc638f6c3f8546529ae19b933fa454fbc4cd1ede5bca05b351cc" },
    TensorSpec { name: "decoder.layer.02.output", shape: &[332, 1024], raw_blake3: "371450c422d098b0d449394fade6194edcdfbc61632be6e872377810dbfc8728" },
    TensorSpec { name: "decoder.layer.03.output", shape: &[332, 1024], raw_blake3: "d025392f602ced5948a4892a78dc6a77e965ff78848333f1f6c37ae3c1ae2763" },
    TensorSpec { name: "decoder.layer.04.output", shape: &[332, 1024], raw_blake3: "a35191b2b032d877eafdfacfe042154d919ba65a0814de29bff0ff1b7e9ad103" },
    TensorSpec { name: "decoder.layer.05.output", shape: &[332, 1024], raw_blake3: "c9b136d5891238bdc9df6f2535c82179a92e7428dc5c13386d2b35ab81fc5655" },
    TensorSpec { name: "decoder.layer.06.output", shape: &[332, 1024], raw_blake3: "2e1e6dee728eb412b40e4dfc7c509ccf251ebaa90b14c3838b649558bbbfd3a4" },
    TensorSpec { name: "decoder.layer.07.output", shape: &[332, 1024], raw_blake3: "dafdf38e87de7f87b9f742b97153a51fed1854c5e40501c3af07155478747745" },
    TensorSpec { name: "decoder.layer.08.output", shape: &[332, 1024], raw_blake3: "2d1172cb3ef44008b86ca171547963637695fc55655125f7557af5047319d52f" },
    TensorSpec { name: "decoder.layer.09.output", shape: &[332, 1024], raw_blake3: "51b0b2b6c7d57d0bcc3c062ef093aa902c4527da83acaafbc96098ffab184241" },
    TensorSpec { name: "decoder.layer.10.output", shape: &[332, 1024], raw_blake3: "b1f36a6cedb0e4a52f2b8ba35e5906434e3b7575c638f910e380dd70d0c29eb1" },
    TensorSpec { name: "decoder.layer.11.output", shape: &[332, 1024], raw_blake3: "9ea87539784ead4957837f8f2773fcaa327db36f7508f93744c3144dffbac9fc" },
    TensorSpec { name: "decoder.layer.12.output", shape: &[332, 1024], raw_blake3: "cf8630fe39cb849e8e3756d389c332078aa2a197c30a912bfcb4eef6709c7b62" },
    TensorSpec { name: "decoder.layer.13.output", shape: &[332, 1024], raw_blake3: "02bd1711796f8723c89cb7492556553af81f8f55fdda58c5ff891d1f226a8aa3" },
    TensorSpec { name: "decoder.layer.14.output", shape: &[332, 1024], raw_blake3: "82493b17983fc4c564dd8282fb8c88f1367b27826738df4380878ef85aa2c7ef" },
    TensorSpec { name: "decoder.layer.15.output", shape: &[332, 1024], raw_blake3: "30e4e90ad96c8e90076bd9207bf2b3317d854cc8187c5c617116118527f61a28" },
    TensorSpec { name: "decoder.layer.16.output", shape: &[332, 1024], raw_blake3: "450f9361fe27e0084145124767c1a294a9a33f4fd40460311a2ca58d9f1f814c" },
    TensorSpec { name: "decoder.layer.17.output", shape: &[332, 1024], raw_blake3: "6e21cbbaa94f6e7dd979d8e039b59cdf86140569b3262d30489eaf6eb091ba20" },
    TensorSpec { name: RAW_COS, shape: &[3, 332, 128], raw_blake3: "096287f2c2ee912105fbc747def39441b541c50b87b1330a8b3b3647b2b49654" },
    TensorSpec { name: RAW_SIN, shape: &[3, 332, 128], raw_blake3: "d34eff803104785331690d7f263c4f7ce44838f6083c5f2fb5ed987de613d310" },
    TensorSpec { name: FINAL_NORM_WEIGHT, shape: &[1024], raw_blake3: "f0c43a017dbf900afe8cfa3a05012ad01263663c9715e1cba831350ec7fd833e" },
];

#[rustfmt::skip]
const METADATA: [(&str, &str); 20] = [
    ("bias", "false"),
    ("case_id", "ocr.clean_latin.0001"),
    ("decoded_text", "JUL"),
    ("device", "mps"),
    ("dtype", "bfloat16"),
    ("fixture_schema", "pvlc.decoder_stack.official.v1"),
    ("generated_tokens", "94013,898"),
    ("head_dim", "128"),
    ("hidden_size", "1024"),
    ("intermediate_size", "3072"),
    ("key_value_heads", "2"),
    ("layers", "18"),
    ("model_id", "PaddlePaddle/PaddleOCR-VL-1.6"),
    ("model_revision", MODEL_REVISION),
    ("mrope_sections", "16,24,24"),
    ("oracle", "TransformersOracle pinned remote code"),
    ("query_heads", "16"),
    ("rms_norm_epsilon", "1e-5"),
    ("tokens", "332"),
    ("trace_level", "L3"),
];

const fn policy(
    max_abs: f64,
    max_mean_abs: f64,
    max_p99_abs: f64,
    max_relative_l2: f64,
) -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs,
        max_mean_abs,
        max_p99_abs,
        max_relative_l2,
        min_cosine_similarity: -1.0,
        max_per_token_relative_l2: None,
        max_per_channel_relative_l2: None,
    }
}

// Independently measured by an optimized Rust harness that chained only the
// accepted decoder_layer_prefill_f32 primitive, loading and dropping one
// checkpoint layer's nine weights at a time. Each frozen envelope rounds
// upward from 1.25x its observed max/mean/p99/relative-L2; the minimum actual
// headroom after rounding is 25.0003%. Policies are constants, never derived
// from fixture values at test runtime.
// Observed max/mean/p99/relative-L2 by depth:
// L00 .099239/.0007929/.006638/.002406; L01 1.48456/.001304/.011183/.001059
// L02 1.39435/.001584/.013127/.001142; L03 1.39600/.001776/.014603/.001218
// L04 3.34302/.002060/.015843/.001646; L05 3.50146/.002568/.016425/.001764
// L06 3.68384/.002774/.017135/.001878; L07 4.01392/.003201/.017904/.002077
// L08 4.32104/.003550/.018807/.002336; L09 4.46674/.003794/.019683/.002610
// L10 4.84775/.004084/.020522/.002806; L11 5.07721/.004388/.022619/.002944
// L12 5.12094/.005466/.025067/.003132; L13 5.43750/.006116/.027060/.003508
// L14 5.45398/.007723/.032465/.003788; L15 6.63837/.010426/.040679/.004529
// L16 8.34808/.015420/.057706/.005677; L17 17.3330/.030682/.218224/.015934
// final 6.62232/.119962/.527341/.028823.
#[rustfmt::skip]
const LAYER_POLICIES: [ComparisonPolicy; LAYERS] = [
    policy(0.1241, 0.000_991_2, 0.008_298, 0.003_008),
    policy(1.856, 0.001_631, 0.013_98, 0.001_325),
    policy(1.743, 0.001_98, 0.016_41, 0.001_428),
    policy(1.745, 0.002_221, 0.018_26, 0.001_523),
    policy(4.179, 0.002_575, 0.019_81, 0.002_058),
    policy(4.377, 0.003_211, 0.020_54, 0.002_205),
    policy(4.605, 0.003_468, 0.021_42, 0.002_348),
    policy(5.018, 0.004_002, 0.022_39, 0.002_597),
    policy(5.402, 0.004_438, 0.023_51, 0.002_92),
    policy(5.584, 0.004_743, 0.024_61, 0.003_263),
    policy(6.06, 0.005_106, 0.025_66, 0.003_508),
    policy(6.347, 0.005_485, 0.028_28, 0.003_681),
    policy(6.402, 0.006_833, 0.031_34, 0.003_915),
    policy(6.797, 0.007_646, 0.033_83, 0.004_385),
    policy(6.818, 0.009_654, 0.040_59, 0.004_736),
    policy(8.298, 0.013_04, 0.050_85, 0.005_661),
    policy(10.44, 0.019_28, 0.072_14, 0.007_096),
    policy(21.67, 0.038_36, 0.272_8, 0.019_92),
];
const FINAL_NORM_POLICY: ComparisonPolicy = policy(8.278, 0.15, 0.659_2, 0.036_03);
// Component-only final norm observed .531746/.008308/.048550/.002340;
// unnormalized and reversed-weight relative-L2 were 1.25756 and .716027.
const FINAL_NORM_COMPONENT_POLICY: ComparisonPolicy = policy(0.665, 0.010_4, 0.060_7, 0.002_93);

// Pre-policy M5e calibration used the live trace.final_norm returned by this
// exact 18-layer gate. Observed max/mean/p99/relative-L2 were
// .6029319763/.1544219851/.3726227283/.05143819814 and cosine was
// .9994116124. Each error bound below rounds upward beyond 1.25x its observed
// value; the cosine-distance allowance has the same minimum headroom.
const INTEGRATED_LOGITS_POLICY: ComparisonPolicy = ComparisonPolicy {
    require_finite: true,
    max_abs: 0.754,
    max_mean_abs: 0.1931,
    max_p99_abs: 0.466,
    max_relative_l2: 0.0644,
    min_cosine_similarity: 0.999_26,
    max_per_token_relative_l2: Some(0.0644),
    max_per_channel_relative_l2: None,
};
const INTEGRATED_MAX_KL_REFERENCE_TO_CANDIDATE: f64 = 0.006_11;
const INTEGRATED_MAX_JENSEN_SHANNON_DIVERGENCE: f64 = 0.001_52;

const REFERENCE_TOP_32: [usize; 32] = [
    94_013, 93_992, 820, 13_476, 1_369, 5_859, 1_704, 93_965, 588, 763, 6_934, 93_969, 1_315,
    93_966, 93_971, 1_502, 93_955, 93_967, 5_083, 93_982, 2_668, 9_063, 93_970, 729, 3_448, 707,
    93_957, 93_987, 93_981, 4_373, 19_280, 36_561,
];
const INTEGRATED_TOP_32: [usize; 32] = [
    94_013, 93_992, 13_476, 820, 1_369, 5_859, 1_704, 93_965, 6_934, 763, 588, 93_966, 93_971,
    1_315, 93_969, 1_502, 93_955, 93_967, 5_083, 93_982, 9_063, 93_970, 2_668, 729, 3_448, 707,
    93_957, 36_561, 93_981, 19_280, 93_987, 5_268,
];

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/decoder-stack-official-v1.safetensors")
}

fn logits_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/prefill-lm-head-official-v1.safetensors")
}

fn model_path() -> PathBuf {
    repository()
        .join("models/snapshots")
        .join(MODEL_REVISION)
        .join("model.safetensors")
}

fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn hash_file(path: &Path) -> String {
    let mut source = File::open(path).unwrap();
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
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

fn load(catalog: &SafetensorsCatalog, name: &str) -> Vec<f32> {
    catalog.load_tensor_f32(name).unwrap()
}

fn compare_logit_vectors(reference: &[f32], candidate: &[f32]) -> ComparisonReport {
    compare_f32(
        reference,
        candidate,
        &[1, VOCAB_SIZE],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap()
}

fn layer_output_name(layer: usize) -> String {
    format!("decoder.layer.{layer:02}.output")
}

fn layer_config() -> DecoderLayerConfig {
    DecoderLayerConfig {
        tokens: TOKENS,
        hidden_size: HIDDEN_SIZE,
        intermediate_size: INTERMEDIATE_SIZE,
        query_heads: QUERY_HEADS,
        key_value_heads: KEY_VALUE_HEADS,
        head_dim: HEAD_DIM,
        rms_norm_epsilon: RMS_NORM_EPSILON,
        mrope_sections: MROPE_SECTIONS,
    }
}

fn assert_lock_identity() {
    let golden: toml::Value = toml::from_str(GOLDEN_LOCK).unwrap();
    let golden = golden.as_table().unwrap();
    assert_eq!(
        golden
            .get("format_version")
            .and_then(toml::Value::as_integer),
        Some(1)
    );
    assert_eq!(
        golden
            .get("trace_schema_version")
            .and_then(toml::Value::as_integer),
        Some(1)
    );
    assert_eq!(
        golden.get("model_revision").and_then(toml::Value::as_str),
        Some(MODEL_REVISION)
    );
    let bundles = golden
        .get("bundles")
        .and_then(toml::Value::as_array)
        .unwrap();
    let matching = bundles
        .iter()
        .filter(|bundle| {
            bundle.get("case_id").and_then(toml::Value::as_str) == Some("ocr.clean_latin.0001")
                && bundle.get("trace_level").and_then(toml::Value::as_str) == Some("L3")
        })
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 1, "golden identity must be unique");
    let bundle = matching[0].as_table().unwrap();
    assert_eq!(
        bundle.get("artifact_path").and_then(toml::Value::as_str),
        Some("artifacts/goldens/ocr.clean_latin.0001-l3")
    );
    assert_eq!(
        bundle.get("bundle_digest").and_then(toml::Value::as_str),
        Some("blake3:35572ac07da5fddee97becb46fb866f906200f433f4c5677ad20fdf36440acf9")
    );
    assert_eq!(
        bundle
            .get("semantic_fingerprint")
            .and_then(toml::Value::as_str),
        Some("blake3:632cbe2de6f47f8f84c764e3a551bf99f352d7459c98d939eb89725357f952b4")
    );
    assert_eq!(
        bundle.get("decoded_text").and_then(toml::Value::as_str),
        Some("JUL")
    );
    assert_eq!(
        bundle.get("repeat_count").and_then(toml::Value::as_integer),
        Some(2)
    );
    let generated_tokens = bundle
        .get("generated_tokens")
        .and_then(toml::Value::as_array)
        .unwrap()
        .iter()
        .map(|value| value.as_integer().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(generated_tokens, [94_013, 898]);

    let model: toml::Value = toml::from_str(MODEL_LOCK).unwrap();
    let model = model.as_table().unwrap();
    assert_eq!(
        model
            .get("format_version")
            .and_then(toml::Value::as_integer),
        Some(1)
    );
    assert_eq!(
        model
            .get("compiler_model_abi")
            .and_then(toml::Value::as_integer),
        Some(1)
    );
    assert_eq!(
        model.get("model_id").and_then(toml::Value::as_str),
        Some("PaddlePaddle/PaddleOCR-VL-1.6")
    );
    assert_eq!(
        model.get("revision").and_then(toml::Value::as_str),
        Some(MODEL_REVISION)
    );
    let model_file = model
        .get("files")
        .and_then(toml::Value::as_table)
        .and_then(|files| files.get("model.safetensors"))
        .and_then(toml::Value::as_table)
        .unwrap();
    assert_eq!(model_file.len(), 2, "model identity tuple changed");
    assert_eq!(
        model_file.get("blake3").and_then(toml::Value::as_str),
        Some(MODEL_FILE_BLAKE3)
    );
    assert_eq!(
        model_file.get("size").and_then(toml::Value::as_integer),
        Some(MODEL_FILE_BYTES as i64)
    );
}

fn assert_stage(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    comparison_policy: &ComparisonPolicy,
) {
    let report = compare_f32(
        expected,
        actual,
        &[TOKENS, HIDDEN_SIZE],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap();
    let verdict = report.assess(comparison_policy).unwrap();
    assert!(verdict.passed(), "{label}\n{report:#?}\n{verdict:#?}");
}

fn assert_rejected(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    comparison_policy: &ComparisonPolicy,
) {
    let report = compare_f32(
        expected,
        actual,
        &[TOKENS, HIDDEN_SIZE],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap();
    let verdict = report.assess(comparison_policy).unwrap();
    assert!(
        !verdict.passed(),
        "negative control {label} unexpectedly passed\n{report:#?}"
    );
}

// Keeps both future public entry points type-checked without running a second
// 18-layer official pass. The ignored hard gate below exercises the pinned ABI.
#[allow(dead_code)]
fn generic_stack_entrypoint<F>(
    input: &[f32],
    config: DecoderStackConfig,
    checkpoints: &[usize],
    final_norm_weight: &[f32],
    execute_layer: F,
) -> Result<DecoderStackPrefillTrace, CpuRefError>
where
    F: FnMut(usize, DecoderLayerConfig, &[f32]) -> Result<DecoderLayerPrefillTrace, CpuRefError>,
{
    decoder_stack_prefill_f32(input, config, checkpoints, final_norm_weight, execute_layer)
}

#[derive(Default)]
struct WeightTracker {
    live: Cell<usize>,
    peak: Cell<usize>,
    loads: Cell<usize>,
    drops: Cell<usize>,
}

impl WeightTracker {
    fn loaded(&self) {
        let live = self.live.get() + 1;
        self.live.set(live);
        self.peak.set(self.peak.get().max(live));
        self.loads.set(self.loads.get() + 1);
    }

    fn dropped(&self) {
        self.live.set(self.live.get() - 1);
        self.drops.set(self.drops.get() + 1);
    }
}

struct OwnedLayerParameters {
    input_norm_weight: Vec<f32>,
    query_weight: Vec<f32>,
    key_weight: Vec<f32>,
    value_weight: Vec<f32>,
    attention_output_weight: Vec<f32>,
    post_attention_norm_weight: Vec<f32>,
    gate_weight: Vec<f32>,
    up_weight: Vec<f32>,
    down_weight: Vec<f32>,
    tracker: Rc<WeightTracker>,
}

fn model_tensor(catalog: &SafetensorsCatalog, name: &str, shape: &[u64]) -> Vec<f32> {
    let tensor = catalog.tensor(name).unwrap();
    assert_eq!(tensor.shape, shape, "{name}");
    assert_eq!(tensor.dtype.safetensors_name(), "BF16", "{name}");
    catalog.load_tensor_f32(name).unwrap()
}

impl OwnedLayerParameters {
    fn load(catalog: &SafetensorsCatalog, layer: usize, tracker: Rc<WeightTracker>) -> Self {
        tracker.loaded();
        let name = |suffix: &str| format!("model.layers.{layer}.{suffix}");
        Self {
            input_norm_weight: model_tensor(
                catalog,
                &name("input_layernorm.weight"),
                &[HIDDEN_SIZE as u64],
            ),
            query_weight: model_tensor(
                catalog,
                &name("self_attn.q_proj.weight"),
                &[(QUERY_HEADS * HEAD_DIM) as u64, HIDDEN_SIZE as u64],
            ),
            key_weight: model_tensor(
                catalog,
                &name("self_attn.k_proj.weight"),
                &[KEY_VALUE_WIDTH as u64, HIDDEN_SIZE as u64],
            ),
            value_weight: model_tensor(
                catalog,
                &name("self_attn.v_proj.weight"),
                &[KEY_VALUE_WIDTH as u64, HIDDEN_SIZE as u64],
            ),
            attention_output_weight: model_tensor(
                catalog,
                &name("self_attn.o_proj.weight"),
                &[HIDDEN_SIZE as u64, (QUERY_HEADS * HEAD_DIM) as u64],
            ),
            post_attention_norm_weight: model_tensor(
                catalog,
                &name("post_attention_layernorm.weight"),
                &[HIDDEN_SIZE as u64],
            ),
            gate_weight: model_tensor(
                catalog,
                &name("mlp.gate_proj.weight"),
                &[INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
            ),
            up_weight: model_tensor(
                catalog,
                &name("mlp.up_proj.weight"),
                &[INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
            ),
            down_weight: model_tensor(
                catalog,
                &name("mlp.down_proj.weight"),
                &[HIDDEN_SIZE as u64, INTERMEDIATE_SIZE as u64],
            ),
            tracker,
        }
    }

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
}

impl Drop for OwnedLayerParameters {
    fn drop(&mut self) {
        self.tracker.dropped();
    }
}

#[test]
fn authenticates_official_stack_fixture_and_current_pinned_locks() {
    assert_lock_identity();
    let path = fixture_path();
    let fixture_bytes = fs::read(&path).unwrap();
    assert_eq!(fixture_bytes.len(), FIXTURE_BYTES);
    assert_eq!(hash_bytes(&fixture_bytes), FIXTURE_BLAKE3);

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
            spec.raw_blake3,
            "{}",
            spec.name
        );
    }

    // Each depth has an independent official raw anchor. This makes resets,
    // repeats, reorders, and skipped layers observable rather than checking
    // only the final state of the stack.
    let layer_hashes = TENSORS[2..2 + LAYERS]
        .iter()
        .map(|spec| spec.raw_blake3)
        .collect::<HashSet<_>>();
    assert_eq!(layer_hashes.len(), LAYERS);
    assert_ne!(TENSORS[0].raw_blake3, TENSORS[2 + LAYERS - 1].raw_blake3);
}

#[test]
fn official_final_norm_requires_the_exact_weight_and_is_not_layer_seventeen() {
    let catalog = SafetensorsCatalog::open(fixture_path()).unwrap();
    let layer_seventeen = load(&catalog, "decoder.layer.17.output");
    let expected = load(&catalog, FINAL_NORM);
    let weight = load(&catalog, FINAL_NORM_WEIGHT);
    assert_ne!(layer_seventeen, expected);

    let actual = rms_norm_f32(
        &layer_seventeen,
        TOKENS,
        HIDDEN_SIZE,
        &weight,
        RMS_NORM_EPSILON,
    )
    .unwrap();
    assert_stage(
        "official final RMSNorm component",
        &expected,
        &actual,
        &FINAL_NORM_COMPONENT_POLICY,
    );
    assert_rejected(
        "unnormalized official layer 17",
        &expected,
        &layer_seventeen,
        &FINAL_NORM_COMPONENT_POLICY,
    );

    let mut wrong_weight = weight;
    wrong_weight.reverse();
    let wrong = rms_norm_f32(
        &layer_seventeen,
        TOKENS,
        HIDDEN_SIZE,
        &wrong_weight,
        RMS_NORM_EPSILON,
    )
    .unwrap();
    assert_ne!(wrong, expected);
    assert_rejected(
        "reversed final norm weight",
        &expected,
        &wrong,
        &FINAL_NORM_COMPONENT_POLICY,
    );
}

#[test]
#[ignore = "release-only full 18-layer decoder stack with pinned 1.9 GB checkpoint"]
fn full_official_streaming_stack_matches_every_depth_caches_and_final_norm() {
    assert_lock_identity();
    let model_path = model_path();
    assert_eq!(fs::metadata(&model_path).unwrap().len(), MODEL_FILE_BYTES);
    assert_eq!(hash_file(&model_path), MODEL_FILE_BLAKE3);

    // SafetensorsCatalog::open is the bounded header-only importer; individual
    // layer payloads are materialized only inside the streaming closure.
    let model = SafetensorsCatalog::open(&model_path).unwrap();
    assert_eq!(model.tensors().len(), 620);
    let fixture = SafetensorsCatalog::open(fixture_path()).unwrap();
    let input = load(&fixture, INPUT);
    let raw_cos = load(&fixture, RAW_COS);
    let raw_sin = load(&fixture, RAW_SIN);
    let fixture_final_norm_weight = load(&fixture, FINAL_NORM_WEIGHT);
    let checkpoint_final_norm_weight =
        model_tensor(&model, FINAL_NORM_WEIGHT, &[HIDDEN_SIZE as u64]);
    assert_eq!(checkpoint_final_norm_weight, fixture_final_norm_weight);
    let preserved = (
        hash_f32(&input),
        hash_f32(&raw_cos),
        hash_f32(&raw_sin),
        hash_f32(&checkpoint_final_norm_weight),
    );

    let calls = RefCell::new(Vec::new());
    let expected_next_input = RefCell::new(Some(hash_f32(&input)));
    let tracker = Rc::new(WeightTracker::default());
    let checkpoint_layers = (0..LAYERS).collect::<Vec<_>>();
    let trace: DecoderStackPrefillTrace = pinned_decoder_stack_prefill_f32(
        &input,
        TOKENS,
        &checkpoint_layers,
        &checkpoint_final_norm_weight,
        |layer: usize,
         supplied_config: DecoderLayerConfig,
         current: &[f32]|
         -> Result<DecoderLayerPrefillTrace, CpuRefError> {
            let expected_layer = calls.borrow().len();
            assert_eq!(layer, expected_layer, "streaming layer order drifted");
            assert_eq!(
                supplied_config,
                layer_config(),
                "pinned layer topology drifted at layer {layer}"
            );
            calls.borrow_mut().push(layer);
            assert_eq!(current.len(), TOKENS * HIDDEN_SIZE);
            assert!(current.iter().all(|value: &f32| value.is_finite()));
            let expected_digest = expected_next_input.borrow_mut().take().unwrap();
            assert_eq!(
                hash_f32(current),
                expected_digest,
                "layer {layer} input drift"
            );

            let parameters = OwnedLayerParameters::load(&model, layer, Rc::clone(&tracker));
            let layer_trace = decoder_layer_prefill_f32(
                current,
                supplied_config,
                &raw_cos,
                &raw_sin,
                parameters.borrowed(),
            )?;
            *expected_next_input.borrow_mut() = Some(hash_f32(&layer_trace.output));
            Ok(layer_trace)
        },
    )
    .unwrap();

    assert_eq!(*calls.borrow(), (0..LAYERS).collect::<Vec<_>>());
    assert_eq!(tracker.loads.get(), LAYERS);
    assert_eq!(tracker.drops.get(), LAYERS);
    // This tracker is deliberately scoped to checkpoint weight sets. Stage
    // buffer liveness is enforced by the compact allocator contract.
    assert_eq!(tracker.live.get(), 0, "layer weights survived stack return");
    assert_eq!(
        tracker.peak.get(),
        1,
        "more than one layer's weights were live concurrently"
    );
    assert_eq!(trace.executed_layers, LAYERS);
    assert_eq!(trace.checkpoints.len(), LAYERS);
    assert_eq!(trace.kv_caches.len(), LAYERS);
    assert_eq!(
        trace.retained_checkpoint_elements,
        LAYERS * TOKENS * HIDDEN_SIZE
    );
    assert_eq!(
        trace.retained_kv_elements,
        LAYERS * 2 * TOKENS * KEY_VALUE_WIDTH
    );
    assert_eq!(
        trace
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.layer_index)
            .collect::<Vec<_>>(),
        checkpoint_layers
    );

    for (layer, comparison_policy) in LAYER_POLICIES.iter().enumerate() {
        let checkpoint = trace.checkpoint(layer).unwrap();
        assert_eq!(checkpoint.len(), TOKENS * HIDDEN_SIZE);
        assert!(checkpoint.iter().all(|value: &f32| value.is_finite()));
        let expected = load(&fixture, &layer_output_name(layer));
        assert_stage(
            &layer_output_name(layer),
            &expected,
            checkpoint,
            comparison_policy,
        );

        let cache = trace.kv_cache(layer).unwrap();
        assert_eq!(cache.tokens, TOKENS);
        assert_eq!(cache.key_value_heads, KEY_VALUE_HEADS);
        assert_eq!(cache.head_dim, HEAD_DIM);
        assert_eq!(cache.keys.len(), TOKENS * KEY_VALUE_WIDTH);
        assert_eq!(cache.values.len(), TOKENS * KEY_VALUE_WIDTH);
        assert!(
            cache
                .keys
                .iter()
                .chain(&cache.values)
                .all(|value: &f32| value.is_finite())
        );
        assert!(std::ptr::eq(cache, &trace.kv_caches[layer]));
    }
    assert_eq!(trace.checkpoint(LAYERS), None);
    assert_eq!(trace.checkpoint(usize::MAX), None);
    assert_eq!(trace.kv_cache(LAYERS), None);
    assert_eq!(trace.kv_cache(usize::MAX), None);
    assert_eq!(trace.final_norm.len(), TOKENS * HIDDEN_SIZE);
    assert!(trace.final_norm.iter().all(|value: &f32| value.is_finite()));
    let expected_final_norm = load(&fixture, FINAL_NORM);
    assert_stage(
        FINAL_NORM,
        &expected_final_norm,
        &trace.final_norm,
        &FINAL_NORM_POLICY,
    );

    let layer_seventeen = trace.checkpoint(LAYERS - 1).unwrap();
    assert_ne!(trace.final_norm.as_slice(), layer_seventeen);
    assert_rejected(
        "stack omitted final RMSNorm",
        &expected_final_norm,
        layer_seventeen,
        &FINAL_NORM_POLICY,
    );
    let mut wrong_weight = checkpoint_final_norm_weight.clone();
    wrong_weight.reverse();
    let wrong_final_norm = rms_norm_f32(
        layer_seventeen,
        TOKENS,
        HIDDEN_SIZE,
        &wrong_weight,
        RMS_NORM_EPSILON,
    )
    .unwrap();
    assert_ne!(trace.final_norm.as_slice(), wrong_final_norm.as_slice());
    assert_rejected(
        "stack used a reversed final norm weight",
        &expected_final_norm,
        &wrong_final_norm,
        &FINAL_NORM_POLICY,
    );

    assert_eq!(hash_f32(&input), preserved.0);
    assert_eq!(hash_f32(&raw_cos), preserved.1);
    assert_eq!(hash_f32(&raw_sin), preserved.2);
    assert_eq!(hash_f32(&checkpoint_final_norm_weight), preserved.3);
    let layer_seventeen_digest = hash_f32(layer_seventeen);
    assert_eq!(
        expected_next_input.borrow().as_deref(),
        Some(layer_seventeen_digest.as_str())
    );

    // M5e extends this live accepted stack run rather than executing a second
    // decoder stack. The integrated policy above was frozen only after a
    // policy-free run of this exact path recorded its complete metrics.
    let integrated_extension_started = Instant::now();
    let logits_fixture_path = logits_fixture_path();
    let logits_fixture_bytes = fs::read(&logits_fixture_path).unwrap();
    assert_eq!(logits_fixture_bytes.len(), LOGITS_FIXTURE_BYTES);
    assert_eq!(hash_bytes(&logits_fixture_bytes), LOGITS_FIXTURE_BLAKE3);
    let logits_fixture = SafetensorsCatalog::open(logits_fixture_path).unwrap();
    assert_eq!(
        logits_fixture
            .metadata()
            .iter()
            .find(|(key, _)| *key == "fixture_schema")
            .map(|(_, value)| value.as_str()),
        Some("pvlc.prefill_lm_head.official.v1")
    );
    assert_eq!(
        logits_fixture
            .metadata()
            .iter()
            .find(|(key, _)| *key == "model_revision")
            .map(|(_, value)| value.as_str()),
        Some(MODEL_REVISION)
    );

    let captured_final_norm_tensor = logits_fixture.tensor(FINAL_NORM).unwrap();
    assert_eq!(
        captured_final_norm_tensor.shape,
        [1, TOKENS as u64, HIDDEN_SIZE as u64]
    );
    assert_eq!(captured_final_norm_tensor.dtype.safetensors_name(), "BF16");
    assert_eq!(
        hash_bytes(&tensor_bytes(&logits_fixture, FINAL_NORM)),
        LOGITS_FINAL_NORM_RAW_BLAKE3
    );
    let expected_logits_tensor = logits_fixture.tensor(LAST_LOGITS).unwrap();
    assert_eq!(expected_logits_tensor.shape, [1, VOCAB_SIZE as u64]);
    assert_eq!(expected_logits_tensor.dtype.safetensors_name(), "BF16");
    assert_eq!(
        hash_bytes(&tensor_bytes(&logits_fixture, LAST_LOGITS)),
        LOGITS_RAW_BLAKE3
    );

    let lm_head_tensor = model.tensor(LM_HEAD_WEIGHT).unwrap();
    assert_eq!(
        lm_head_tensor.shape,
        [VOCAB_SIZE as u64, HIDDEN_SIZE as u64]
    );
    assert_eq!(lm_head_tensor.dtype.safetensors_name(), "BF16");
    assert_eq!(lm_head_tensor.byte_len(), 211_812_352);
    assert_eq!(
        hash_bytes(&tensor_bytes(&model, LM_HEAD_WEIGHT)),
        LM_HEAD_RAW_BLAKE3
    );

    let expected_logits = load(&logits_fixture, LAST_LOGITS);
    let captured_final_norm = load(&logits_fixture, FINAL_NORM);
    let lm_head_weight = load(&model, LM_HEAD_WEIGHT);
    let preserved_integrated_operands = (
        hash_f32(&trace.final_norm),
        hash_f32(layer_seventeen),
        hash_f32(&lm_head_weight),
    );

    let live_head_started = Instant::now();
    let live_logits =
        pinned_prefill_last_logits_f32(&trace.final_norm, TOKENS, &lm_head_weight).unwrap();
    let live_head_runtime = live_head_started.elapsed();
    let live_report = compare_logit_vectors(&expected_logits, &live_logits);
    let live_topology = compare_logits(&expected_logits, &live_logits, 32).unwrap();

    let captured_head_started = Instant::now();
    let captured_logits =
        pinned_prefill_last_logits_f32(&captured_final_norm, TOKENS, &lm_head_weight).unwrap();
    let captured_head_runtime = captured_head_started.elapsed();
    let live_vs_captured_report = compare_logit_vectors(&captured_logits, &live_logits);
    let live_vs_captured_topology = compare_logits(&captured_logits, &live_logits, 32).unwrap();

    let negative_head_started = Instant::now();
    let skipped_final_norm_logits =
        pinned_prefill_last_logits_f32(layer_seventeen, TOKENS, &lm_head_weight).unwrap();
    let negative_head_runtime = negative_head_started.elapsed();
    let skipped_final_norm_report =
        compare_logit_vectors(&expected_logits, &skipped_final_norm_logits);
    let skipped_final_norm_topology =
        compare_logits(&expected_logits, &skipped_final_norm_logits, 32).unwrap();

    assert_eq!(live_logits.len(), VOCAB_SIZE);
    assert!(live_logits.iter().all(|value: &f32| value.is_finite()));
    assert_eq!(hash_f32(&live_logits), INTEGRATED_LOGITS_F32_BLAKE3);
    let live_verdict = live_report.assess(&INTEGRATED_LOGITS_POLICY).unwrap();
    assert!(
        live_verdict.passed(),
        "integrated live stack logits\n{live_report:#?}\n{live_verdict:#?}"
    );
    let per_token = live_report.per_token_relative_l2.as_ref().unwrap();
    let per_channel = live_report.per_channel_relative_l2.as_ref().unwrap();
    assert_eq!(per_token.len(), 1);
    assert_eq!(per_token[0], live_report.relative_l2);
    assert_eq!(per_channel.len(), VOCAB_SIZE);

    assert_eq!(live_topology.reference_top1, 94_013);
    assert_eq!(live_topology.candidate_top1, 94_013);
    assert!(live_topology.top1_agreement);
    assert_eq!(live_topology.reference_top_k, REFERENCE_TOP_32);
    assert_eq!(live_topology.candidate_top_k, INTEGRATED_TOP_32);
    assert_eq!(live_topology.top_k_overlap, 31);
    assert_eq!(live_topology.top_k_overlap_fraction, 31.0 / 32.0);
    assert_eq!(live_topology.reference_margin, 1.25);
    assert!(
        live_topology.kl_reference_to_candidate <= INTEGRATED_MAX_KL_REFERENCE_TO_CANDIDATE,
        "{live_topology:#?}"
    );
    assert!(
        live_topology.jensen_shannon_divergence <= INTEGRATED_MAX_JENSEN_SHANNON_DIVERGENCE,
        "{live_topology:#?}"
    );
    assert!(
        live_topology.max_abs_error <= INTEGRATED_LOGITS_POLICY.max_abs,
        "{live_topology:#?}"
    );

    assert_eq!(hash_f32(&captured_logits), CAPTURED_LOGITS_F32_BLAKE3);
    assert_eq!(live_vs_captured_topology.reference_top1, 94_013);
    assert_eq!(live_vs_captured_topology.candidate_top1, 94_013);
    assert_eq!(live_vs_captured_topology.top_k_overlap, 31);
    assert!(live_vs_captured_report.relative_l2 > 0.05);
    assert!(live_vs_captured_report.relative_l2 < 0.052);

    assert_eq!(
        hash_f32(&skipped_final_norm_logits),
        SKIPPED_FINAL_NORM_LOGITS_F32_BLAKE3
    );
    let skipped_final_norm_verdict = skipped_final_norm_report
        .assess(&INTEGRATED_LOGITS_POLICY)
        .unwrap();
    assert!(
        !skipped_final_norm_verdict.passed(),
        "skipping final RMSNorm passed integrated policy\n{skipped_final_norm_report:#?}"
    );
    assert_eq!(skipped_final_norm_topology.reference_top1, 94_013);
    assert_eq!(skipped_final_norm_topology.candidate_top1, 94_013);
    assert_eq!(skipped_final_norm_topology.top_k_overlap, 14);
    assert!(skipped_final_norm_report.relative_l2 > 0.47);
    assert!(
        skipped_final_norm_topology.kl_reference_to_candidate > 0.7
            && skipped_final_norm_topology.jensen_shannon_divergence > 0.17
    );

    assert_eq!(hash_f32(&trace.final_norm), preserved_integrated_operands.0);
    assert_eq!(hash_f32(layer_seventeen), preserved_integrated_operands.1);
    assert_eq!(hash_f32(&lm_head_weight), preserved_integrated_operands.2);
    eprintln!(
        "M5e integrated live max={:.17} mean={:.17} p99={:.17} rel_l2={:.17} cosine={:.17} hash={} head_runtime={live_head_runtime:?}",
        live_report.max_abs,
        live_report.mean_abs,
        live_report.p99_abs,
        live_report.relative_l2,
        live_report.cosine_similarity,
        hash_f32(&live_logits),
    );
    eprintln!(
        "M5e integrated topology ref_top1={} live_top1={} overlap={}/32 margin={:.17} kl={:.17} js={:.17} max_selected_abs={:.17} ref_top32={:?} live_top32={:?}",
        live_topology.reference_top1,
        live_topology.candidate_top1,
        live_topology.top_k_overlap,
        live_topology.reference_margin,
        live_topology.kl_reference_to_candidate,
        live_topology.jensen_shannon_divergence,
        live_topology.max_abs_error,
        live_topology.reference_top_k,
        live_topology.candidate_top_k,
    );
    eprintln!(
        "M5e captured baseline hash={} head_runtime={captured_head_runtime:?} runtime_delta_ms={:.6} live_vs_captured max={:.17} mean={:.17} p99={:.17} rel_l2={:.17} cosine={:.17} overlap={}/32 top1={}/{}",
        hash_f32(&captured_logits),
        live_head_runtime
            .as_secs_f64()
            .mul_add(1_000.0, -captured_head_runtime.as_secs_f64() * 1_000.0),
        live_vs_captured_report.max_abs,
        live_vs_captured_report.mean_abs,
        live_vs_captured_report.p99_abs,
        live_vs_captured_report.relative_l2,
        live_vs_captured_report.cosine_similarity,
        live_vs_captured_topology.top_k_overlap,
        live_vs_captured_topology.reference_top1,
        live_vs_captured_topology.candidate_top1,
    );
    eprintln!(
        "M5e skipped_final_norm max={:.17} mean={:.17} p99={:.17} rel_l2={:.17} cosine={:.17} hash={} top1={} overlap={}/32 kl={:.17} js={:.17} head_runtime={negative_head_runtime:?}",
        skipped_final_norm_report.max_abs,
        skipped_final_norm_report.mean_abs,
        skipped_final_norm_report.p99_abs,
        skipped_final_norm_report.relative_l2,
        skipped_final_norm_report.cosine_similarity,
        hash_f32(&skipped_final_norm_logits),
        skipped_final_norm_topology.candidate_top1,
        skipped_final_norm_topology.top_k_overlap,
        skipped_final_norm_topology.kl_reference_to_candidate,
        skipped_final_norm_topology.jensen_shannon_divergence,
    );
    eprintln!(
        "M5e integrated_extension_runtime={:?}",
        integrated_extension_started.elapsed()
    );
}
