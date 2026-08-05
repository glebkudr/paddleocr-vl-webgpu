use std::{
    cell::{Cell, RefCell},
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};

use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, DecoderLayerConfig, DecoderLayerDecodeTrace,
    DecoderLayerParameters, DecoderPrefillKvCache, DecoderStackConfig, decoder_layer_decode_f32,
    rms_norm_f32,
};
use pvlc_cpu_ref::{
    DecoderStackDecodeTrace, decoder_stack_decode_f32, pinned_decoder_stack_decode_f32,
};
use pvlc_safetensors::SafetensorsCatalog;
use pvlc_testkit::{ComparisonAxes, ComparisonPolicy, ComparisonReport, compare_f32};

const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const MODEL_LOCK_BLAKE3: &str = "c2ffa46821fdda1a7e4cd6bad63cdcdc2f775b8fed250b0679cc7d5411577d10";
const MODEL_LOCK_BYTES: usize = 2_385;
const MODEL_FILE_BYTES: u64 = 1_917_255_968;
const MODEL_FILE_BLAKE3: &str = "4dc3ab13685a0c0a701f77f9c5ebafdc0004074247e00bde6b7e7b04279a41fc";
const MODEL_TENSOR_COUNT: usize = 620;
const FIXTURE_BYTES: usize = 6_438_376;
const FIXTURE_BLAKE3: &str = "386735089b35b2a1fc50ad578678689eed98b3e62f2a0c18d5e2890d0e6a8ebf";
const PREFIX_TOKENS: usize = 332;
const FULL_CACHE_TOKENS: usize = 333;
const DECODE_TOKENS: usize = 1;
const LAYERS: usize = 18;
const HIDDEN_SIZE: usize = 1_024;
const INTERMEDIATE_SIZE: usize = 3_072;
const QUERY_HEADS: usize = 16;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const QUERY_WIDTH: usize = QUERY_HEADS * HEAD_DIM;
const KEY_VALUE_WIDTH: usize = KEY_VALUE_HEADS * HEAD_DIM;
const RMS_NORM_EPSILON: f32 = 1.0e-5;
const MROPE_SECTIONS: [usize; 3] = [16, 24, 24];
const INPUT: &str = "decoder.decode.00.layer.00.input";
const RAW_COS: &str = "decoder.decode.00.rope.cos.axis_major";
const RAW_SIN: &str = "decoder.decode.00.rope.sin.axis_major";
const STACKED_CACHE_KEY: &str = "decoder.decode.00.kv.key.layer_token_major";
const STACKED_CACHE_VALUE: &str = "decoder.decode.00.kv.value.layer_token_major";
const FINAL_NORM: &str = "decoder.decode.00.final_norm";
const LOGITS: &str = "decoder.decode.00.logits";
const FINAL_NORM_WEIGHT: &str = "model.norm.weight";
const MODEL_LOCK: &str = include_str!("../../../models/paddleocr-vl-1.6.lock");

struct TensorSpec {
    name: &'static str,
    dtype: &'static str,
    shape: &'static [u64],
    raw_blake3: &'static str,
}

#[derive(Debug)]
struct OfficialDecodeStack {
    input: Vec<f32>,
    raw_cos: Vec<f32>,
    raw_sin: Vec<f32>,
    layer_outputs: Vec<Vec<f32>>,
    full_kv_caches: Vec<DecoderPrefillKvCache>,
    prefix_caches: Vec<DecoderPrefillKvCache>,
    appended_keys: Vec<Vec<f32>>,
    appended_values: Vec<Vec<f32>>,
    final_norm: Vec<f32>,
}

struct NegativeDecodeStack {
    layer_outputs: Vec<Vec<f32>>,
    full_kv_caches: Vec<DecoderPrefillKvCache>,
    final_norm: Vec<f32>,
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
                &[QUERY_WIDTH as u64, HIDDEN_SIZE as u64],
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
                &[HIDDEN_SIZE as u64, QUERY_WIDTH as u64],
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

#[rustfmt::skip]
const METADATA: [(&str, &str); 38] = [
    ("bias", "false"),
    ("cache_layout", "layer_token_major"),
    ("cache_position", "332"),
    ("cache_tokens", "333"),
    ("capture_repeat_count", "2"),
    ("capture_tool_version", "0.1.0"),
    ("case_id", "ocr.clean_latin.0001"),
    ("decode_input_token", "94013"),
    ("decode_next_token", "898"),
    ("decode_step", "1"),
    ("decode_tokens", "1"),
    ("decoded_text", "JUL"),
    ("device", "mps"),
    ("dtype", "bfloat16"),
    ("fixture_schema", "pvlc.decoder_decode.official.v1"),
    ("generated_tokens", "94013,898"),
    ("head_dim", "128"),
    ("hidden_size", "1024"),
    ("intermediate_size", "3072"),
    ("key_value_heads", "2"),
    ("layer0_weights_fixture_blake3", "blake3:30eed2f7a4d9336f8b3429d7294065c45214c4425f7559e8e2059ebd54c89522"),
    ("layers", "18"),
    ("model_id", "PaddlePaddle/PaddleOCR-VL-1.6"),
    ("model_lock_blake3", "blake3:c2ffa46821fdda1a7e4cd6bad63cdcdc2f775b8fed250b0679cc7d5411577d10"),
    ("model_revision", "66317acc4c9fc17bd154591ce650735cd2855f3e"),
    ("mrope_sections", "16,24,24"),
    ("oracle", "TransformersOracle pinned remote code"),
    ("prefix_tokens", "332"),
    ("query_heads", "16"),
    ("rms_norm_epsilon", "1e-5"),
    ("rope_delta", "-290"),
    ("source_bundle_digest", "blake3:47d07a39b638bc6ff68f1eaeab1a81af0407ab26338c745942cfd7e5c5faaa99"),
    ("source_publication_lock_blake3", "blake3:56062fbb43ee1d556cef9eee27928f6222d397bd955353b015a4f0a5f8c3edda"),
    ("source_semantic_fingerprint", "blake3:d5c7ec3c2be5cc1c3d6ed416f2ae8659ab3e5b2e851cee35afe7df10f436a9bc"),
    ("torch_version", "2.13.0"),
    ("trace_level", "L3"),
    ("transformers_version", "5.14.1"),
    ("vocab_size", "103424"),
];

#[rustfmt::skip]
const TENSORS: [TensorSpec; 44] = [
    TensorSpec { name: "decoder.decode.00.attention_mask", dtype: "I64", shape: &[1, 333], raw_blake3: "1270bc1c4d08bec42b7a1d7991dfcff27da57a571bc56e2f5203525bf5aefc51" },
    TensorSpec { name: "decoder.decode.00.cache_position", dtype: "I64", shape: &[1], raw_blake3: "ee65862e503e2ddd48498d79a774debb53ee6811c550df58c27f66573c310000" },
    TensorSpec { name: "decoder.decode.00.final_norm", dtype: "BF16", shape: &[1, 1024], raw_blake3: "16cc97ce46e4839948e6425bcc444d333cac45c0425a88297e29518ca796f6e7" },
    TensorSpec { name: "decoder.decode.00.input_token_id", dtype: "I64", shape: &[1, 1], raw_blake3: "85e7db04bbf0f35fdba9a8a7733b3ff640d47d29dd84f34796f9d6eb67e1e459" },
    TensorSpec { name: "decoder.decode.00.kv.key.layer_token_major", dtype: "BF16", shape: &[18, 333, 2, 128], raw_blake3: "6902c1a29d014177981eaa5daa12390141f0e16331b78dbac0653c45faa3692f" },
    TensorSpec { name: "decoder.decode.00.kv.value.layer_token_major", dtype: "BF16", shape: &[18, 333, 2, 128], raw_blake3: "e396c182612fc2a6e8660dbeb1a5d59d05f5267f07df13dab39c5802acf530f3" },
    TensorSpec { name: "decoder.decode.00.layer.00.attention.context.token_major", dtype: "BF16", shape: &[1, 16, 128], raw_blake3: "c595c7bc425c857e90a9b3d459fb15169aab6774731d00f599482448f32a49f3" },
    TensorSpec { name: "decoder.decode.00.layer.00.attention.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "61a05a8bbab729377d1f398bf09df0979498a9192a7f02be8812a0d1e2ccb1f5" },
    TensorSpec { name: "decoder.decode.00.layer.00.attention.residual", dtype: "BF16", shape: &[1, 1024], raw_blake3: "23442fee845311c12f1da08172f76c3e16bebed75f513fca4505ab62cfda329f" },
    TensorSpec { name: "decoder.decode.00.layer.00.input", dtype: "BF16", shape: &[1, 1024], raw_blake3: "7c0499d2c4b0d5f63a77b2c147af6088344ca84e5540569f3753d1712793502b" },
    TensorSpec { name: "decoder.decode.00.layer.00.k", dtype: "BF16", shape: &[1, 256], raw_blake3: "aad4e2ad7f6d31a152f40cf50c37fd2f648937401c6d02540804499d08c53035" },
    TensorSpec { name: "decoder.decode.00.layer.00.mlp.activation", dtype: "BF16", shape: &[1, 3072], raw_blake3: "30e156aaed6398afb8ba279aedca2a1d34cd263e674c78006a07286f054ec71c" },
    TensorSpec { name: "decoder.decode.00.layer.00.mlp.down", dtype: "BF16", shape: &[1, 1024], raw_blake3: "c1c10e1c99cd3322caa7497b79c1ab81ca652cffed1e01441b356335db6e6e0e" },
    TensorSpec { name: "decoder.decode.00.layer.00.mlp.gate", dtype: "BF16", shape: &[1, 3072], raw_blake3: "ff74206201a4c4bc22fcf992dd8022640113b43d491add15e5e21eb545feadc0" },
    TensorSpec { name: "decoder.decode.00.layer.00.mlp.up", dtype: "BF16", shape: &[1, 3072], raw_blake3: "1a46c157e31c43291fc5a368926ca3e4973f755fd225f03b276002333788dc77" },
    TensorSpec { name: "decoder.decode.00.layer.00.mrope.k.token_major", dtype: "BF16", shape: &[1, 2, 128], raw_blake3: "ca304747635805bd2292e6becc2646f43b6e9213aca057d011cc48492312df78" },
    TensorSpec { name: "decoder.decode.00.layer.00.mrope.q.token_major", dtype: "BF16", shape: &[1, 16, 128], raw_blake3: "5b323a8f75432a2f9232f82413b23c7f95fa45a4b5c342f03248d497c4df6fc6" },
    TensorSpec { name: "decoder.decode.00.layer.00.norm1", dtype: "BF16", shape: &[1, 1024], raw_blake3: "f2e2d6c08254e30bf8a1333144a4c1453e77cd9d9896f4d104cba2001433192e" },
    TensorSpec { name: "decoder.decode.00.layer.00.norm2", dtype: "BF16", shape: &[1, 1024], raw_blake3: "7dde0bbd6d78d7eac72498fc0ca6c4553993b33f361f4d2f19e9f6c0ab40b9d5" },
    TensorSpec { name: "decoder.decode.00.layer.00.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "ac0405f85d90dabab1e6c302a93ba03749590231c5e29ae0f21f960e20dbf896" },
    TensorSpec { name: "decoder.decode.00.layer.00.q", dtype: "BF16", shape: &[1, 2048], raw_blake3: "1d5e0a7592ea47bc9099726ab6a6f6f158c87d832a1bdb2370616a1063c843cf" },
    TensorSpec { name: "decoder.decode.00.layer.00.v", dtype: "BF16", shape: &[1, 256], raw_blake3: "4b92a57f32b9aec54e2ed6a8c441978e3095205e9fb5b29c72f1d14c5a86389d" },
    TensorSpec { name: "decoder.decode.00.layer.01.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "1c2c2ee257aa98140ab77bb132a32f8c24b878ffa91f6f3d67314ed784cd24be" },
    TensorSpec { name: "decoder.decode.00.layer.02.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "f555097b6dc48e3e2fa8d4b0eade89cbb3d69f2e6c5f8445280895d0b973a3d9" },
    TensorSpec { name: "decoder.decode.00.layer.03.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "59ec1ff941e5787e4a7d7b74f0be5a1a369b79623e5475a10b4dbbf9c04f8c85" },
    TensorSpec { name: "decoder.decode.00.layer.04.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "ca6c4af7dac8631f15411e48143c0b600a88f307a5ac761e6b78170e5364892c" },
    TensorSpec { name: "decoder.decode.00.layer.05.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "fa366c3264608a557261a1b091a9ac506287d3900e82ffbd3843dab97df85c8b" },
    TensorSpec { name: "decoder.decode.00.layer.06.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "ec7816055941a7b4372bdb81d3029fe48e8552cad0696db9a78a018c449a812a" },
    TensorSpec { name: "decoder.decode.00.layer.07.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "f8f4397ec91028b7b96c78af845191be2e5d4b2fffa644b80bfab0e4bc3ff923" },
    TensorSpec { name: "decoder.decode.00.layer.08.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "636206f5a07ff5b9b8a169ee179d03e752319a29945a3bdb56eb5aeaf5e0b3d9" },
    TensorSpec { name: "decoder.decode.00.layer.09.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "6506469cebd3a26eedd3cab4b78bc5d24eb46911a8d139cbdfa8804854232a5a" },
    TensorSpec { name: "decoder.decode.00.layer.10.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "7b090ae7337cf57e82f8764a042abfb4e43468a39ea35ec9c51e39935437b7d7" },
    TensorSpec { name: "decoder.decode.00.layer.11.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "73d6b4e8733961722f460ff4e01af290e6d59bf059b1eed6ebdc9e1c15e5a3f1" },
    TensorSpec { name: "decoder.decode.00.layer.12.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "725838c300f26983da0a12fd61ba227f7223e4d65a54eb5a9f366a41568a9b84" },
    TensorSpec { name: "decoder.decode.00.layer.13.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "c5e0714afb02fcda21f0b8d6d3cdd33a380a4375f3627c48d0b3f912092a7f41" },
    TensorSpec { name: "decoder.decode.00.layer.14.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "e1edac9b3f40d6427617144bd70597421006e0e8137bc6198d104e375d67e5bc" },
    TensorSpec { name: "decoder.decode.00.layer.15.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "655e17f393cdab8bbd6f561a68b6644d08a773df019ef8a7a23a8066433b03d0" },
    TensorSpec { name: "decoder.decode.00.layer.16.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "4931e766bab407f5f81de3ee07f2b95d7ac2af536f9e69bc4366e96b6620c087" },
    TensorSpec { name: "decoder.decode.00.layer.17.output", dtype: "BF16", shape: &[1, 1024], raw_blake3: "55a566b87677f3fcdeeccd8ff42e931f4a03fff38faca5cfeee19965f9885de9" },
    TensorSpec { name: "decoder.decode.00.logits", dtype: "BF16", shape: &[1, 103424], raw_blake3: "68b42601be700f15701546b3b9e5c011da5e528b87dc825e01051d56068b5dbf" },
    TensorSpec { name: "decoder.decode.00.position_ids", dtype: "I64", shape: &[3, 1, 1], raw_blake3: "eb6575dc3849f4856f89ad3c632e92be67071d2affdeb640a577a94170744b9c" },
    TensorSpec { name: "decoder.decode.00.rope.cos.axis_major", dtype: "BF16", shape: &[3, 1, 128], raw_blake3: "e7442753718c184649e35b427427c322e410782111e82a80f7b239ae53a9d786" },
    TensorSpec { name: "decoder.decode.00.rope.sin.axis_major", dtype: "BF16", shape: &[3, 1, 128], raw_blake3: "4048e08689726e22e3bcfd1932dd877b8cf744a81cbbe8b3ff9acdabd6e30abc" },
    TensorSpec { name: "decoder.mrope.delta", dtype: "I64", shape: &[1, 1], raw_blake3: "48378c7f3201de62c1fb0e040903669c4a925b3a9611aa4b1f712a8627d7787e" },
];

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/decoder-decode-official-v1.safetensors")
}

fn model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("models/snapshots")
        .join(MODEL_REVISION)
        .join("model.safetensors")
}

fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn hash_file(path: &Path) -> String {
    let bytes = fs::read(path).unwrap();
    hash_bytes(&bytes)
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

fn assert_file_identity(
    path: &Path,
    expected_bytes: usize,
    expected_blake3: &str,
) -> SafetensorsCatalog {
    let bytes = fs::read(path).unwrap();
    assert_eq!(bytes.len(), expected_bytes);
    assert_eq!(hash_bytes(&bytes), expected_blake3);
    SafetensorsCatalog::open(path).unwrap()
}

fn load(catalog: &SafetensorsCatalog, name: &str) -> Vec<f32> {
    catalog.load_tensor_f32(name).unwrap()
}

fn model_tensor(catalog: &SafetensorsCatalog, name: &str, shape: &[u64]) -> Vec<f32> {
    let tensor = catalog.tensor(name).unwrap();
    assert_eq!(tensor.shape, shape, "{name}");
    assert_eq!(tensor.dtype.safetensors_name(), "BF16", "{name}");
    catalog.load_tensor_f32(name).unwrap()
}

fn layer_output_name(layer: usize) -> String {
    format!("decoder.decode.00.layer.{layer:02}.output")
}

fn slice_layer_from_stacked_cache(values: &[f32], layer: usize, tokens: usize) -> Vec<f32> {
    let layer_span = tokens * KEY_VALUE_WIDTH;
    values[layer * layer_span..(layer + 1) * layer_span].to_vec()
}

fn write_cache(keys: &[f32], values: &[f32], tokens: usize) -> DecoderPrefillKvCache {
    DecoderPrefillKvCache {
        keys: keys.to_vec(),
        values: values.to_vec(),
        tokens,
        key_value_heads: KEY_VALUE_HEADS,
        head_dim: HEAD_DIM,
    }
}

fn pinned_layer_config() -> DecoderLayerConfig {
    DecoderLayerConfig {
        tokens: DECODE_TOKENS,
        hidden_size: HIDDEN_SIZE,
        intermediate_size: INTERMEDIATE_SIZE,
        query_heads: QUERY_HEADS,
        key_value_heads: KEY_VALUE_HEADS,
        head_dim: HEAD_DIM,
        rms_norm_epsilon: RMS_NORM_EPSILON,
        mrope_sections: MROPE_SECTIONS,
    }
}

fn pinned_stack_config() -> DecoderStackConfig {
    DecoderStackConfig {
        layer: pinned_layer_config(),
        layers: LAYERS,
    }
}

fn compare_row(label: &str, expected: &[f32], actual: &[f32], width: usize) -> ComparisonReport {
    assert_eq!(expected.len(), width, "{label} expected length");
    assert_eq!(actual.len(), width, "{label} actual length");
    compare_f32(
        expected,
        actual,
        &[DECODE_TOKENS, width],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap()
}

fn compare_hidden(label: &str, expected: &[f32], actual: &[f32]) -> ComparisonReport {
    compare_row(label, expected, actual, HIDDEN_SIZE)
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

fn print_report(label: &str, report: &ComparisonReport) {
    println!(
        "{label}: max_abs={:.12e} mean_abs={:.12e} p99_abs={:.12e} rel_l2={:.12e} cosine={:.12e}",
        report.max_abs,
        report.mean_abs,
        report.p99_abs,
        report.relative_l2,
        report.cosine_similarity
    );
}

const fn full_policy(
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

fn assert_report_passes(label: &str, report: &ComparisonReport, policy: &ComparisonPolicy) {
    let verdict = report.assess(policy).unwrap();
    assert!(
        verdict.passed(),
        "{label} violated frozen policy\npolicy={policy:#?}\nreport={report:#?}\nverdict={verdict:#?}"
    );
}

fn assert_report_rejected(label: &str, report: &ComparisonReport, policy: &ComparisonPolicy) {
    let verdict = report.assess(policy).unwrap();
    assert!(
        !verdict.passed(),
        "{label} unexpectedly passed frozen policy\npolicy={policy:#?}\nreport={report:#?}"
    );
}

// Frozen from `calibrate_independent_stack_decode_policies` on July 20, 2026.
// Every limit is `ceil_readable(1.25 * observed_metric)` from the independent
// 18-layer decode chain, with at least 25.0022% retained headroom after
// rounding. These constants are acceptance inputs, never runtime-derived.
#[rustfmt::skip]
const OUTPUT_POLICIES: [ComparisonPolicy; LAYERS] = [
    full_policy(0.00343, 0.000286, 0.00164, 0.00501),
    full_policy(0.00546, 0.000556, 0.00239, 0.00592),
    full_policy(0.00866, 0.000785, 0.00290, 0.00732),
    full_policy(0.00724, 0.00101, 0.00375, 0.00765),
    full_policy(0.00840, 0.00123, 0.00476, 0.00733),
    full_policy(0.0221, 0.00141, 0.00546, 0.00731),
    full_policy(0.0125, 0.00157, 0.00632, 0.00736),
    full_policy(0.0137, 0.00173, 0.00697, 0.00736),
    full_policy(0.0199, 0.00217, 0.00905, 0.00847),
    full_policy(0.0225, 0.00265, 0.0123, 0.00792),
    full_policy(0.0434, 0.00337, 0.0141, 0.00978),
    full_policy(0.0533, 0.00495, 0.0180, 0.00922),
    full_policy(0.0542, 0.00615, 0.0220, 0.00844),
    full_policy(0.0867, 0.00779, 0.0284, 0.00865),
    full_policy(0.0973, 0.00977, 0.0299, 0.00970),
    full_policy(0.148, 0.0123, 0.0433, 0.0110),
    full_policy(0.211, 0.0163, 0.0539, 0.0124),
    full_policy(0.107, 0.0224, 0.0733, 0.0160),
];

#[rustfmt::skip]
const APPENDED_KEY_POLICIES: [ComparisonPolicy; LAYERS] = [
    full_policy(0.0245, 0.00341, 0.0151, 0.00294),
    full_policy(0.0303, 0.00601, 0.0271, 0.00431),
    full_policy(0.0562, 0.00717, 0.0244, 0.00499),
    full_policy(0.0517, 0.00774, 0.0295, 0.00575),
    full_policy(0.0575, 0.00673, 0.0482, 0.00509),
    full_policy(0.0394, 0.00730, 0.0322, 0.00628),
    full_policy(0.0290, 0.00701, 0.0233, 0.00555),
    full_policy(0.0375, 0.00671, 0.0289, 0.00594),
    full_policy(0.0406, 0.00648, 0.0276, 0.00535),
    full_policy(0.0413, 0.00783, 0.0306, 0.00599),
    full_policy(0.0296, 0.00730, 0.0270, 0.00625),
    full_policy(0.0387, 0.00699, 0.0268, 0.00584),
    full_policy(0.0503, 0.00701, 0.0395, 0.00745),
    full_policy(0.0333, 0.00740, 0.0296, 0.00612),
    full_policy(0.0500, 0.0105, 0.0433, 0.00773),
    full_policy(0.0408, 0.00964, 0.0383, 0.00710),
    full_policy(0.0388, 0.0107, 0.0332, 0.00861),
    full_policy(0.0983, 0.0112, 0.0398, 0.00819),
];

#[rustfmt::skip]
const APPENDED_VALUE_POLICIES: [ComparisonPolicy; LAYERS] = [
    full_policy(0.00107, 0.000185, 0.000688, 0.00289),
    full_policy(0.00524, 0.00105, 0.00428, 0.00583),
    full_policy(0.00778, 0.00205, 0.00696, 0.00668),
    full_policy(0.00891, 0.00245, 0.00767, 0.00972),
    full_policy(0.00912, 0.00262, 0.00748, 0.00923),
    full_policy(0.0118, 0.00279, 0.0102, 0.0119),
    full_policy(0.0103, 0.00276, 0.00769, 0.00990),
    full_policy(0.0135, 0.00320, 0.0109, 0.0103),
    full_policy(0.0118, 0.00309, 0.0104, 0.00893),
    full_policy(0.0140, 0.00460, 0.0138, 0.00973),
    full_policy(0.0197, 0.00443, 0.0149, 0.0110),
    full_policy(0.0173, 0.00433, 0.0141, 0.0111),
    full_policy(0.0541, 0.0111, 0.0376, 0.00805),
    full_policy(0.0368, 0.00896, 0.0293, 0.00817),
    full_policy(0.0271, 0.00838, 0.0262, 0.00721),
    full_policy(0.0395, 0.00893, 0.0286, 0.0124),
    full_policy(0.0556, 0.0128, 0.0394, 0.0122),
    full_policy(0.0658, 0.0162, 0.0518, 0.0140),
];

const FINAL_NORM_POLICY: ComparisonPolicy = full_policy(0.697, 0.0887, 0.308, 0.0158);

fn assert_model_lock_identity() {
    assert_eq!(MODEL_LOCK.len(), MODEL_LOCK_BYTES);
    assert_eq!(hash_bytes(MODEL_LOCK.as_bytes()), MODEL_LOCK_BLAKE3);

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
    assert_eq!(
        model_file.get("blake3").and_then(toml::Value::as_str),
        Some(MODEL_FILE_BLAKE3)
    );
    assert_eq!(
        model_file.get("size").and_then(toml::Value::as_integer),
        Some(MODEL_FILE_BYTES as i64)
    );
}

fn assert_selected_model_tensor_headers(model: &SafetensorsCatalog) {
    assert_eq!(model.tensors().len(), MODEL_TENSOR_COUNT);
    for layer in 0..LAYERS {
        let layer_prefix = format!("model.layers.{layer}.");
        let cases = [
            ("input_layernorm.weight", vec![HIDDEN_SIZE as u64]),
            (
                "self_attn.q_proj.weight",
                vec![QUERY_WIDTH as u64, HIDDEN_SIZE as u64],
            ),
            (
                "self_attn.k_proj.weight",
                vec![KEY_VALUE_WIDTH as u64, HIDDEN_SIZE as u64],
            ),
            (
                "self_attn.v_proj.weight",
                vec![KEY_VALUE_WIDTH as u64, HIDDEN_SIZE as u64],
            ),
            (
                "self_attn.o_proj.weight",
                vec![HIDDEN_SIZE as u64, QUERY_WIDTH as u64],
            ),
            ("post_attention_layernorm.weight", vec![HIDDEN_SIZE as u64]),
            (
                "mlp.gate_proj.weight",
                vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
            ),
            (
                "mlp.up_proj.weight",
                vec![INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
            ),
            (
                "mlp.down_proj.weight",
                vec![HIDDEN_SIZE as u64, INTERMEDIATE_SIZE as u64],
            ),
        ];
        for (suffix, shape) in cases {
            let name = format!("{layer_prefix}{suffix}");
            let tensor = model.tensor(&name).unwrap();
            assert_eq!(tensor.shape, shape, "{name}");
            assert_eq!(tensor.dtype.safetensors_name(), "BF16", "{name}");
        }
    }
    let final_norm = model.tensor(FINAL_NORM_WEIGHT).unwrap();
    assert_eq!(final_norm.shape, [HIDDEN_SIZE as u64]);
    assert_eq!(final_norm.dtype.safetensors_name(), "BF16");
}

fn load_official_stack(catalog: &SafetensorsCatalog) -> OfficialDecodeStack {
    let stacked_keys = load(catalog, STACKED_CACHE_KEY);
    let stacked_values = load(catalog, STACKED_CACHE_VALUE);
    let mut full_kv_caches = Vec::with_capacity(LAYERS);
    let mut prefix_caches = Vec::with_capacity(LAYERS);
    let mut appended_keys = Vec::with_capacity(LAYERS);
    let mut appended_values = Vec::with_capacity(LAYERS);
    let mut layer_outputs = Vec::with_capacity(LAYERS);

    for layer in 0..LAYERS {
        let full_keys = slice_layer_from_stacked_cache(&stacked_keys, layer, FULL_CACHE_TOKENS);
        let full_values = slice_layer_from_stacked_cache(&stacked_values, layer, FULL_CACHE_TOKENS);
        let prefix_keys = full_keys[..PREFIX_TOKENS * KEY_VALUE_WIDTH].to_vec();
        let prefix_values = full_values[..PREFIX_TOKENS * KEY_VALUE_WIDTH].to_vec();
        let appended_key = full_keys[PREFIX_TOKENS * KEY_VALUE_WIDTH..].to_vec();
        let appended_value = full_values[PREFIX_TOKENS * KEY_VALUE_WIDTH..].to_vec();
        assert_eq!(prefix_keys.len(), PREFIX_TOKENS * KEY_VALUE_WIDTH);
        assert_eq!(prefix_values.len(), PREFIX_TOKENS * KEY_VALUE_WIDTH);
        assert_eq!(appended_key.len(), KEY_VALUE_WIDTH);
        assert_eq!(appended_value.len(), KEY_VALUE_WIDTH);

        let mut reconstructed_keys = prefix_keys.clone();
        reconstructed_keys.extend_from_slice(&appended_key);
        let mut reconstructed_values = prefix_values.clone();
        reconstructed_values.extend_from_slice(&appended_value);
        assert_eq!(reconstructed_keys, full_keys);
        assert_eq!(reconstructed_values, full_values);

        full_kv_caches.push(write_cache(&full_keys, &full_values, FULL_CACHE_TOKENS));
        prefix_caches.push(write_cache(&prefix_keys, &prefix_values, PREFIX_TOKENS));
        appended_keys.push(appended_key);
        appended_values.push(appended_value);
        layer_outputs.push(load(catalog, &layer_output_name(layer)));
    }

    OfficialDecodeStack {
        input: load(catalog, INPUT),
        raw_cos: load(catalog, RAW_COS),
        raw_sin: load(catalog, RAW_SIN),
        layer_outputs,
        full_kv_caches,
        prefix_caches,
        appended_keys,
        appended_values,
        final_norm: load(catalog, FINAL_NORM),
    }
}

fn independent_decode_stack(
    model: &SafetensorsCatalog,
    official: &OfficialDecodeStack,
    final_norm_weight: &[f32],
    tracker: Rc<WeightTracker>,
) -> NegativeDecodeStack {
    let mut current = official.input.clone();
    let mut layer_outputs = Vec::with_capacity(LAYERS);
    let mut full_kv_caches = Vec::with_capacity(LAYERS);

    for layer in 0..LAYERS {
        let parameters = OwnedLayerParameters::load(model, layer, Rc::clone(&tracker));
        let trace = decoder_layer_decode_f32(
            &current,
            pinned_layer_config(),
            &official.raw_cos,
            &official.raw_sin,
            &official.prefix_caches[layer],
            parameters.borrowed(),
        )
        .unwrap();
        current = trace.output.clone();
        layer_outputs.push(trace.output);
        full_kv_caches.push(trace.kv_cache);
    }

    let final_norm = rms_norm_f32(
        &current,
        DECODE_TOKENS,
        HIDDEN_SIZE,
        final_norm_weight,
        RMS_NORM_EPSILON,
    )
    .unwrap();

    NegativeDecodeStack {
        layer_outputs,
        full_kv_caches,
        final_norm,
    }
}

fn independent_decode_stack_with_cache_reset(
    model: &SafetensorsCatalog,
    official: &OfficialDecodeStack,
    final_norm_weight: &[f32],
) -> NegativeDecodeStack {
    let mut current = official.input.clone();
    let mut layer_outputs = Vec::with_capacity(LAYERS);
    let mut full_kv_caches = Vec::with_capacity(LAYERS);

    for layer in 0..LAYERS {
        let parameters =
            OwnedLayerParameters::load(model, layer, Rc::new(WeightTracker::default()));
        let prefix_cache = if layer == 0 {
            &official.prefix_caches[layer]
        } else {
            &official.prefix_caches[layer - 1]
        };
        let trace = decoder_layer_decode_f32(
            &current,
            pinned_layer_config(),
            &official.raw_cos,
            &official.raw_sin,
            prefix_cache,
            parameters.borrowed(),
        )
        .unwrap();
        current = trace.output.clone();
        layer_outputs.push(trace.output);
        full_kv_caches.push(trace.kv_cache);
    }

    let final_norm = rms_norm_f32(
        &current,
        DECODE_TOKENS,
        HIDDEN_SIZE,
        final_norm_weight,
        RMS_NORM_EPSILON,
    )
    .unwrap();

    NegativeDecodeStack {
        layer_outputs,
        full_kv_caches,
        final_norm,
    }
}

#[test]
fn authenticates_decode_fixture_inventory_model_lock_and_header_dependencies() {
    let fixture = assert_file_identity(&fixture_path(), FIXTURE_BYTES, FIXTURE_BLAKE3);
    let metadata = fixture
        .metadata()
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(metadata, METADATA);
    assert_eq!(fixture.metadata().len(), METADATA.len());
    assert_eq!(fixture.tensors().len(), TENSORS.len());
    for (tensor, spec) in fixture.tensors().iter().zip(TENSORS.iter()) {
        assert_eq!(tensor.name, spec.name);
        assert_eq!(tensor.shape, spec.shape);
        assert_eq!(tensor.dtype.safetensors_name(), spec.dtype);
        assert_eq!(
            hash_bytes(&tensor_bytes(&fixture, spec.name)),
            spec.raw_blake3,
            "{}",
            spec.name
        );
    }

    let official = load_official_stack(&fixture);
    let distinct_outputs = TENSORS
        .iter()
        .filter_map(|tensor| {
            let suffix = tensor
                .name
                .strip_prefix("decoder.decode.00.layer.")?
                .strip_suffix(".output")?;
            if suffix.len() == 2 && suffix.chars().all(|ch| ch.is_ascii_digit()) {
                Some(tensor.raw_blake3)
            } else {
                None
            }
        })
        .collect::<HashSet<_>>();
    assert_eq!(distinct_outputs.len(), LAYERS);
    assert_ne!(TENSORS[2].raw_blake3, TENSORS[2 + LAYERS].raw_blake3);
    assert_eq!(official.layer_outputs.len(), LAYERS);
    assert_eq!(official.prefix_caches.len(), LAYERS);
    assert_eq!(official.full_kv_caches.len(), LAYERS);
    assert_eq!(official.final_norm.len(), HIDDEN_SIZE);
    assert_eq!(load(&fixture, LOGITS).len(), 103_424);

    assert_model_lock_identity();
    let model_path = model_path();
    if !model_path.is_file() {
        eprintln!("skipping checkpoint header dependencies: model weights are not distributed");
        return;
    }
    assert_eq!(fs::metadata(&model_path).unwrap().len(), MODEL_FILE_BYTES);
    let model = SafetensorsCatalog::open(model_path).unwrap();
    assert_selected_model_tensor_headers(&model);
}

#[test]
#[ignore = "release-only real-weight independent 18-layer cached decode policy oracle"]
fn independent_stack_decode_matches_frozen_per_depth_policies_and_reset_cache_fails() {
    let started = Instant::now();
    assert_model_lock_identity();
    let fixture = SafetensorsCatalog::open(fixture_path()).unwrap();
    let official = load_official_stack(&fixture);
    let model = SafetensorsCatalog::open(model_path()).unwrap();
    let final_norm_weight = model_tensor(&model, FINAL_NORM_WEIGHT, &[HIDDEN_SIZE as u64]);

    let preserved = (
        hash_f32(&official.input),
        hash_f32(&official.raw_cos),
        hash_f32(&official.raw_sin),
        hash_f32(&final_norm_weight),
    );

    let tracker = Rc::new(WeightTracker::default());
    let independent =
        independent_decode_stack(&model, &official, &final_norm_weight, Rc::clone(&tracker));
    let negative = independent_decode_stack_with_cache_reset(&model, &official, &final_norm_weight);

    assert_eq!(tracker.loads.get(), LAYERS);
    assert_eq!(tracker.drops.get(), LAYERS);
    assert_eq!(tracker.live.get(), 0);
    assert_eq!(tracker.peak.get(), 1);

    for layer in 0..LAYERS {
        assert_eq!(
            &independent.full_kv_caches[layer].keys[..PREFIX_TOKENS * KEY_VALUE_WIDTH],
            official.prefix_caches[layer].keys.as_slice(),
            "layer {layer} key prefix drifted",
        );
        assert_eq!(
            &independent.full_kv_caches[layer].values[..PREFIX_TOKENS * KEY_VALUE_WIDTH],
            official.prefix_caches[layer].values.as_slice(),
            "layer {layer} value prefix drifted",
        );
        assert_eq!(
            independent.full_kv_caches[layer].keys.len(),
            FULL_CACHE_TOKENS * KEY_VALUE_WIDTH
        );
        assert_eq!(
            independent.full_kv_caches[layer].values.len(),
            FULL_CACHE_TOKENS * KEY_VALUE_WIDTH
        );
        let output = compare_hidden(
            &format!("layer{layer:02}.output"),
            &official.layer_outputs[layer],
            &independent.layer_outputs[layer],
        );
        let appended_key = compare_row(
            &format!("layer{layer:02}.appended_key"),
            &official.appended_keys[layer],
            &independent.full_kv_caches[layer].keys[PREFIX_TOKENS * KEY_VALUE_WIDTH..],
            KEY_VALUE_WIDTH,
        );
        let appended_value = compare_row(
            &format!("layer{layer:02}.appended_value"),
            &official.appended_values[layer],
            &independent.full_kv_caches[layer].values[PREFIX_TOKENS * KEY_VALUE_WIDTH..],
            KEY_VALUE_WIDTH,
        );
        assert_report_passes(
            &format!("layer{layer:02}.output"),
            &output,
            &OUTPUT_POLICIES[layer],
        );
        assert_report_passes(
            &format!("layer{layer:02}.appended_key"),
            &appended_key,
            &APPENDED_KEY_POLICIES[layer],
        );
        assert_report_passes(
            &format!("layer{layer:02}.appended_value"),
            &appended_value,
            &APPENDED_VALUE_POLICIES[layer],
        );
    }

    let final_norm = compare_hidden("final_norm", &official.final_norm, &independent.final_norm);
    assert_report_passes("final_norm", &final_norm, &FINAL_NORM_POLICY);

    let negative_layer17 = compare_hidden(
        "negative.layer17.output",
        &official.layer_outputs[LAYERS - 1],
        &negative.layer_outputs[LAYERS - 1],
    );
    let negative_final_norm = compare_hidden(
        "negative.final_norm",
        &official.final_norm,
        &negative.final_norm,
    );
    assert_report_rejected(
        "negative.layer17.output",
        &negative_layer17,
        &OUTPUT_POLICIES[LAYERS - 1],
    );
    assert_report_rejected(
        "negative.final_norm",
        &negative_final_norm,
        &FINAL_NORM_POLICY,
    );
    assert!(
        negative_final_norm.max_abs > final_norm.max_abs
            && negative_final_norm.mean_abs > final_norm.mean_abs
            && negative_final_norm.p99_abs > final_norm.p99_abs
            && negative_final_norm.relative_l2 > final_norm.relative_l2,
        "reset-cache negative was not materially worse\npositive={final_norm:#?}\nnegative={negative_final_norm:#?}"
    );

    assert_eq!(hash_f32(&official.input), preserved.0);
    assert_eq!(hash_f32(&official.raw_cos), preserved.1);
    assert_eq!(hash_f32(&official.raw_sin), preserved.2);
    assert_eq!(hash_f32(&final_norm_weight), preserved.3);
    println!("runtime_ms={}", started.elapsed().as_secs_f64() * 1_000.0);
}

fn generic_stack_entrypoint<F>(
    input: &[f32],
    config: DecoderStackConfig,
    prefix_caches: &[DecoderPrefillKvCache],
    checkpoint_layers: &[usize],
    final_norm_weight: &[f32],
    execute_layer: F,
) -> Result<DecoderStackDecodeTrace, CpuRefError>
where
    F: FnMut(
        usize,
        DecoderLayerConfig,
        &[f32],
        &DecoderPrefillKvCache,
    ) -> Result<DecoderLayerDecodeTrace, CpuRefError>,
{
    decoder_stack_decode_f32(
        input,
        config,
        prefix_caches,
        checkpoint_layers,
        final_norm_weight,
        execute_layer,
    )
}

#[test]
fn pinned_stack_decode_wrapper_is_behavioral_and_rejects_bad_prefix_count_before_callback() {
    let dense = |len: usize, mul: usize, add: usize, modulus: usize, divisor: f32| {
        (0..len)
            .map(|index| (((index * mul + add) % modulus) as f32 + 1.0) / divisor)
            .collect::<Vec<_>>()
    };

    let input = dense(HIDDEN_SIZE, 7, 3, 251, 23.0);
    let final_norm_weight = dense(HIDDEN_SIZE, 17, 2, 239, 37.0);
    let checkpoint_layers = vec![0, 7, 17];
    let make_prefix = |layer: usize| {
        let keys = dense(PREFIX_TOKENS * KEY_VALUE_WIDTH, 19 + layer, 7, 211, 41.0);
        let values = dense(PREFIX_TOKENS * KEY_VALUE_WIDTH, 23 + layer, 11, 227, 43.0);
        DecoderPrefillKvCache {
            keys,
            values,
            tokens: PREFIX_TOKENS,
            key_value_heads: KEY_VALUE_HEADS,
            head_dim: HEAD_DIM,
        }
    };
    let prefix_caches = (0..LAYERS).map(make_prefix).collect::<Vec<_>>();

    let malformed_error = pinned_decoder_stack_decode_f32(
        &input,
        &prefix_caches[..LAYERS - 1],
        &checkpoint_layers,
        &final_norm_weight,
        |_layer: usize,
         _config: DecoderLayerConfig,
         _current: &[f32],
         _prefix_cache: &DecoderPrefillKvCache| unreachable!(),
    )
    .unwrap_err();
    assert_eq!(malformed_error.code(), CpuRefErrorCode::DimensionMismatch);

    let generic_malformed_prefix_error = generic_stack_entrypoint(
        &input,
        pinned_stack_config(),
        &prefix_caches[..LAYERS - 1],
        &checkpoint_layers,
        &final_norm_weight,
        |_layer: usize,
         _config: DecoderLayerConfig,
         _current: &[f32],
         _prefix_cache: &DecoderPrefillKvCache| unreachable!(),
    )
    .unwrap_err();
    assert_eq!(
        generic_malformed_prefix_error.code(),
        CpuRefErrorCode::DimensionMismatch
    );

    let duplicate_checkpoint_error = pinned_decoder_stack_decode_f32(
        &input,
        &prefix_caches,
        &[0, 0],
        &final_norm_weight,
        |_layer: usize,
         _config: DecoderLayerConfig,
         _current: &[f32],
         _prefix_cache: &DecoderPrefillKvCache| unreachable!(),
    )
    .unwrap_err();
    assert_eq!(
        duplicate_checkpoint_error.code(),
        CpuRefErrorCode::InvalidCheckpointSelection
    );

    let generic_duplicate_checkpoint_error = generic_stack_entrypoint(
        &input,
        pinned_stack_config(),
        &prefix_caches,
        &[0, 0],
        &final_norm_weight,
        |_layer: usize,
         _config: DecoderLayerConfig,
         _current: &[f32],
         _prefix_cache: &DecoderPrefillKvCache| unreachable!(),
    )
    .unwrap_err();
    assert_eq!(
        generic_duplicate_checkpoint_error.code(),
        CpuRefErrorCode::InvalidCheckpointSelection
    );

    let generic_out_of_range_checkpoint_error = generic_stack_entrypoint(
        &input,
        pinned_stack_config(),
        &prefix_caches,
        &[LAYERS],
        &final_norm_weight,
        |_layer: usize,
         _config: DecoderLayerConfig,
         _current: &[f32],
         _prefix_cache: &DecoderPrefillKvCache| unreachable!(),
    )
    .unwrap_err();
    assert_eq!(
        generic_out_of_range_checkpoint_error.code(),
        CpuRefErrorCode::InvalidCheckpointSelection
    );

    let pinned_out_of_range_checkpoint_error = pinned_decoder_stack_decode_f32(
        &input,
        &prefix_caches,
        &[LAYERS],
        &final_norm_weight,
        |_layer: usize,
         _config: DecoderLayerConfig,
         _current: &[f32],
         _prefix_cache: &DecoderPrefillKvCache| unreachable!(),
    )
    .unwrap_err();
    assert_eq!(
        pinned_out_of_range_checkpoint_error.code(),
        CpuRefErrorCode::InvalidCheckpointSelection
    );

    let call_order = RefCell::new(Vec::new());
    let expected_current = RefCell::new(input.clone());
    let expected_outputs = RefCell::new(vec![Vec::new(); LAYERS]);
    let expected_caches = RefCell::new(vec![None; LAYERS]);
    let trace = pinned_decoder_stack_decode_f32(
        &input,
        &prefix_caches,
        &checkpoint_layers,
        &final_norm_weight,
        |layer: usize,
         config: DecoderLayerConfig,
         current: &[f32],
         prefix_cache: &DecoderPrefillKvCache| {
            assert_eq!(layer, call_order.borrow().len());
            assert_eq!(config, pinned_layer_config());
            assert_eq!(current, expected_current.borrow().as_slice());
            assert!(std::ptr::eq(prefix_cache, &prefix_caches[layer]));
            call_order.borrow_mut().push(layer);

            let output = dense(HIDDEN_SIZE, 29 + layer, 13, 241, 17.0 + layer as f32);
            let appended_key = dense(KEY_VALUE_WIDTH, 31 + layer, 5, 233, 19.0 + layer as f32);
            let appended_value = dense(KEY_VALUE_WIDTH, 37 + layer, 9, 229, 23.0 + layer as f32);
            let mut keys = prefix_cache.keys.clone();
            keys.extend_from_slice(&appended_key);
            let mut values = prefix_cache.values.clone();
            values.extend_from_slice(&appended_value);
            let kv_cache = DecoderPrefillKvCache {
                keys,
                values,
                tokens: FULL_CACHE_TOKENS,
                key_value_heads: KEY_VALUE_HEADS,
                head_dim: HEAD_DIM,
            };
            *expected_current.borrow_mut() = output.clone();
            expected_outputs.borrow_mut()[layer] = output.clone();
            expected_caches.borrow_mut()[layer] = Some(kv_cache.clone());
            Ok(DecoderLayerDecodeTrace {
                norm1: output.clone(),
                query: dense(QUERY_WIDTH, 41 + layer, 3, 251, 29.0),
                key: appended_key.clone(),
                value: appended_value.clone(),
                mrope_query: dense(QUERY_WIDTH, 43 + layer, 7, 239, 31.0),
                mrope_key: appended_key.clone(),
                attention_context: dense(QUERY_WIDTH, 47 + layer, 11, 227, 37.0),
                attention_output: output.clone(),
                attention_residual: output.clone(),
                norm2: output.clone(),
                mlp_gate: dense(INTERMEDIATE_SIZE, 53 + layer, 13, 257, 41.0),
                mlp_up: dense(INTERMEDIATE_SIZE, 59 + layer, 17, 263, 43.0),
                mlp_activation: dense(INTERMEDIATE_SIZE, 61 + layer, 19, 269, 47.0),
                mlp_down: output.clone(),
                output,
                kv_cache,
            })
        },
    )
    .unwrap();

    assert_eq!(*call_order.borrow(), (0..LAYERS).collect::<Vec<_>>());
    assert_eq!(trace.executed_layers, LAYERS);
    assert_eq!(trace.checkpoints.len(), checkpoint_layers.len());
    assert_eq!(trace.kv_caches.len(), LAYERS);
    assert_eq!(
        trace.retained_checkpoint_elements,
        checkpoint_layers.len() * HIDDEN_SIZE
    );
    assert_eq!(
        trace.retained_kv_elements,
        LAYERS * 2 * FULL_CACHE_TOKENS * KEY_VALUE_WIDTH
    );
    assert_eq!(
        trace
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.layer_index)
            .collect::<Vec<_>>(),
        checkpoint_layers
    );
    assert_eq!(trace.checkpoint(LAYERS), None);
    assert_eq!(trace.checkpoint(usize::MAX), None);
    assert_eq!(trace.kv_cache(LAYERS), None);
    assert_eq!(trace.kv_cache(usize::MAX), None);
    for (layer, prefix_cache) in prefix_caches.iter().enumerate() {
        let checkpoint = trace.checkpoint(layer);
        if checkpoint_layers.contains(&layer) {
            assert_f32_bits(
                &format!("checkpoint {layer}"),
                checkpoint.unwrap(),
                &expected_outputs.borrow()[layer],
            );
        } else {
            assert_eq!(checkpoint, None);
        }
        let expected_cache = expected_caches.borrow()[layer].clone().unwrap();
        assert_f32_bits(
            &format!("pinned cache keys {layer}"),
            &trace.kv_caches[layer].keys,
            &expected_cache.keys,
        );
        assert_f32_bits(
            &format!("pinned cache values {layer}"),
            &trace.kv_caches[layer].values,
            &expected_cache.values,
        );
        assert_eq!(
            &trace.kv_caches[layer].keys[..PREFIX_TOKENS * KEY_VALUE_WIDTH],
            prefix_cache.keys.as_slice()
        );
        assert_eq!(
            &trace.kv_caches[layer].values[..PREFIX_TOKENS * KEY_VALUE_WIDTH],
            prefix_cache.values.as_slice()
        );
    }
    let expected_final_norm = rms_norm_f32(
        expected_current.borrow().as_slice(),
        DECODE_TOKENS,
        HIDDEN_SIZE,
        &final_norm_weight,
        RMS_NORM_EPSILON,
    )
    .unwrap();
    assert_f32_bits("final_norm", &trace.final_norm, &expected_final_norm);

    let generic_call_order = RefCell::new(Vec::new());
    let generic_expected_current = RefCell::new(input.clone());
    let generic_expected_outputs = RefCell::new(vec![Vec::new(); LAYERS]);
    let generic_expected_caches = RefCell::new(vec![None; LAYERS]);
    let generic = generic_stack_entrypoint(
        &input,
        pinned_stack_config(),
        &prefix_caches,
        &checkpoint_layers,
        &final_norm_weight,
        |layer: usize,
         config: DecoderLayerConfig,
         current: &[f32],
         prefix_cache: &DecoderPrefillKvCache| {
            assert_eq!(layer, generic_call_order.borrow().len());
            assert_eq!(config, pinned_layer_config());
            assert_eq!(current, generic_expected_current.borrow().as_slice());
            assert!(std::ptr::eq(prefix_cache, &prefix_caches[layer]));
            generic_call_order.borrow_mut().push(layer);

            let output = dense(HIDDEN_SIZE, 71 + layer, 5, 211, 53.0);
            let appended_key = dense(KEY_VALUE_WIDTH, 73 + layer, 7, 197, 59.0);
            let appended_value = dense(KEY_VALUE_WIDTH, 79 + layer, 11, 199, 61.0);
            let mut keys = prefix_cache.keys.clone();
            keys.extend_from_slice(&appended_key);
            let mut values = prefix_cache.values.clone();
            values.extend_from_slice(&appended_value);
            let kv_cache = DecoderPrefillKvCache {
                keys,
                values,
                tokens: FULL_CACHE_TOKENS,
                key_value_heads: KEY_VALUE_HEADS,
                head_dim: HEAD_DIM,
            };
            generic_expected_current.borrow_mut().clone_from(&output);
            generic_expected_outputs.borrow_mut()[layer] = output.clone();
            generic_expected_caches.borrow_mut()[layer] = Some(kv_cache.clone());
            Ok(DecoderLayerDecodeTrace {
                norm1: output.clone(),
                query: dense(QUERY_WIDTH, 83 + layer, 13, 223, 67.0),
                key: appended_key.clone(),
                value: appended_value.clone(),
                mrope_query: dense(QUERY_WIDTH, 89 + layer, 17, 227, 71.0),
                mrope_key: appended_key,
                attention_context: dense(QUERY_WIDTH, 97 + layer, 19, 229, 73.0),
                attention_output: output.clone(),
                attention_residual: output.clone(),
                norm2: output.clone(),
                mlp_gate: dense(INTERMEDIATE_SIZE, 101 + layer, 23, 233, 79.0),
                mlp_up: dense(INTERMEDIATE_SIZE, 103 + layer, 29, 239, 83.0),
                mlp_activation: dense(INTERMEDIATE_SIZE, 107 + layer, 31, 241, 89.0),
                mlp_down: output.clone(),
                output,
                kv_cache,
            })
        },
    )
    .unwrap();
    assert_eq!(generic.executed_layers, LAYERS);
    assert_eq!(generic.checkpoints.len(), checkpoint_layers.len());
    assert_eq!(
        generic.retained_checkpoint_elements,
        checkpoint_layers.len() * HIDDEN_SIZE
    );
    assert_eq!(
        generic.retained_kv_elements,
        LAYERS * 2 * FULL_CACHE_TOKENS * KEY_VALUE_WIDTH
    );
    assert_eq!(
        generic
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.layer_index)
            .collect::<Vec<_>>(),
        checkpoint_layers
    );
    assert_eq!(generic.checkpoint(LAYERS), None);
    assert_eq!(generic.checkpoint(usize::MAX), None);
    assert_eq!(generic.kv_cache(LAYERS), None);
    assert_eq!(generic.kv_cache(usize::MAX), None);
    for (layer, prefix_cache) in prefix_caches.iter().enumerate() {
        let checkpoint = generic.checkpoint(layer);
        if checkpoint_layers.contains(&layer) {
            assert_f32_bits(
                &format!("generic checkpoint {layer}"),
                checkpoint.unwrap(),
                &generic_expected_outputs.borrow()[layer],
            );
        } else {
            assert_eq!(checkpoint, None);
        }
        let expected_cache = generic_expected_caches.borrow()[layer].clone().unwrap();
        assert_f32_bits(
            &format!("generic cache keys {layer}"),
            &generic.kv_caches[layer].keys,
            &expected_cache.keys,
        );
        assert_f32_bits(
            &format!("generic cache values {layer}"),
            &generic.kv_caches[layer].values,
            &expected_cache.values,
        );
        assert_eq!(
            &generic.kv_caches[layer].keys[..PREFIX_TOKENS * KEY_VALUE_WIDTH],
            prefix_cache.keys.as_slice()
        );
        assert_eq!(
            &generic.kv_caches[layer].values[..PREFIX_TOKENS * KEY_VALUE_WIDTH],
            prefix_cache.values.as_slice()
        );
    }
    assert_eq!(
        *generic_call_order.borrow(),
        (0..LAYERS).collect::<Vec<_>>()
    );
    let generic_final_norm = rms_norm_f32(
        generic_expected_current.borrow().as_slice(),
        DECODE_TOKENS,
        HIDDEN_SIZE,
        &final_norm_weight,
        RMS_NORM_EPSILON,
    )
    .unwrap();
    assert_f32_bits(
        "generic final_norm",
        &generic.final_norm,
        &generic_final_norm,
    );
}

#[test]
#[ignore = "release-only official 18-layer cached decode stack with pinned 1.9 GB checkpoint"]
fn full_official_cached_stack_matches_independent_chain_and_frozen_oracles() {
    let started = Instant::now();
    assert_model_lock_identity();
    let model_path = model_path();
    assert_eq!(fs::metadata(&model_path).unwrap().len(), MODEL_FILE_BYTES);
    assert_eq!(hash_file(&model_path), MODEL_FILE_BLAKE3);

    let model = SafetensorsCatalog::open(&model_path).unwrap();
    let fixture = assert_file_identity(&fixture_path(), FIXTURE_BYTES, FIXTURE_BLAKE3);
    let official = load_official_stack(&fixture);
    let final_norm_weight = model_tensor(&model, FINAL_NORM_WEIGHT, &[HIDDEN_SIZE as u64]);
    let preserved = (
        hash_f32(&official.input),
        hash_f32(&official.raw_cos),
        hash_f32(&official.raw_sin),
        hash_f32(&final_norm_weight),
        official
            .prefix_caches
            .iter()
            .map(|cache| (hash_f32(&cache.keys), hash_f32(&cache.values)))
            .collect::<Vec<_>>(),
    );

    let independent_tracker = Rc::new(WeightTracker::default());
    let independent = independent_decode_stack(
        &model,
        &official,
        &final_norm_weight,
        Rc::clone(&independent_tracker),
    );
    assert_eq!(independent_tracker.loads.get(), LAYERS);
    assert_eq!(independent_tracker.drops.get(), LAYERS);
    assert_eq!(independent_tracker.live.get(), 0);
    assert_eq!(independent_tracker.peak.get(), 1);

    let calls = RefCell::new(Vec::new());
    let expected_input = RefCell::new(official.input.clone());
    let sut_tracker = Rc::new(WeightTracker::default());
    let trace = pinned_decoder_stack_decode_f32(
        &official.input,
        &official.prefix_caches,
        &(0..LAYERS).collect::<Vec<_>>(),
        &final_norm_weight,
        |layer: usize,
         config: DecoderLayerConfig,
         current: &[f32],
         prefix_cache: &DecoderPrefillKvCache| {
            assert_eq!(layer, calls.borrow().len(), "decode layer order drifted");
            assert_eq!(config, pinned_layer_config(), "pinned topology drifted");
            assert_f32_bits(
                "layer input chaining",
                current,
                expected_input.borrow().as_slice(),
            );
            assert!(std::ptr::eq(prefix_cache, &official.prefix_caches[layer]));
            calls.borrow_mut().push(layer);

            let parameters = OwnedLayerParameters::load(&model, layer, Rc::clone(&sut_tracker));
            let layer_trace = decoder_layer_decode_f32(
                current,
                config,
                &official.raw_cos,
                &official.raw_sin,
                prefix_cache,
                parameters.borrowed(),
            )?;
            *expected_input.borrow_mut() = layer_trace.output.clone();
            Ok(layer_trace)
        },
    )
    .unwrap();

    assert_eq!(*calls.borrow(), (0..LAYERS).collect::<Vec<_>>());
    assert_eq!(sut_tracker.loads.get(), LAYERS);
    assert_eq!(sut_tracker.drops.get(), LAYERS);
    assert_eq!(sut_tracker.live.get(), 0);
    assert_eq!(sut_tracker.peak.get(), 1);
    assert_eq!(trace.executed_layers, LAYERS);
    assert_eq!(trace.checkpoints.len(), LAYERS);
    assert_eq!(trace.kv_caches.len(), LAYERS);
    assert_eq!(
        trace
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.layer_index)
            .collect::<Vec<_>>(),
        (0..LAYERS).collect::<Vec<_>>()
    );

    for layer in 0..LAYERS {
        assert_eq!(
            &trace.kv_caches[layer].keys[..PREFIX_TOKENS * KEY_VALUE_WIDTH],
            official.prefix_caches[layer].keys.as_slice(),
            "sut layer {layer} key prefix drifted",
        );
        assert_eq!(
            &trace.kv_caches[layer].values[..PREFIX_TOKENS * KEY_VALUE_WIDTH],
            official.prefix_caches[layer].values.as_slice(),
            "sut layer {layer} value prefix drifted",
        );
        assert_f32_bits(
            &format!("sut_vs_independent.layer{layer:02}.output"),
            trace.checkpoint(layer).unwrap(),
            &independent.layer_outputs[layer],
        );
        assert_f32_bits(
            &format!("sut_vs_independent.layer{layer:02}.kv.keys"),
            &trace.kv_caches[layer].keys,
            &independent.full_kv_caches[layer].keys,
        );
        assert_f32_bits(
            &format!("sut_vs_independent.layer{layer:02}.kv.values"),
            &trace.kv_caches[layer].values,
            &independent.full_kv_caches[layer].values,
        );
        assert_eq!(trace.kv_caches[layer].tokens, FULL_CACHE_TOKENS);
        assert_eq!(trace.kv_caches[layer].key_value_heads, KEY_VALUE_HEADS);
        assert_eq!(trace.kv_caches[layer].head_dim, HEAD_DIM);

        let output = compare_hidden(
            &format!("layer{layer:02}.output"),
            &official.layer_outputs[layer],
            trace.checkpoint(layer).unwrap(),
        );
        let appended_key = compare_row(
            &format!("layer{layer:02}.appended_key"),
            &official.appended_keys[layer],
            &trace.kv_caches[layer].keys[PREFIX_TOKENS * KEY_VALUE_WIDTH..],
            KEY_VALUE_WIDTH,
        );
        let appended_value = compare_row(
            &format!("layer{layer:02}.appended_value"),
            &official.appended_values[layer],
            &trace.kv_caches[layer].values[PREFIX_TOKENS * KEY_VALUE_WIDTH..],
            KEY_VALUE_WIDTH,
        );
        assert_report_passes(
            &format!("layer{layer:02}.output"),
            &output,
            &OUTPUT_POLICIES[layer],
        );
        assert_report_passes(
            &format!("layer{layer:02}.appended_key"),
            &appended_key,
            &APPENDED_KEY_POLICIES[layer],
        );
        assert_report_passes(
            &format!("layer{layer:02}.appended_value"),
            &appended_value,
            &APPENDED_VALUE_POLICIES[layer],
        );
    }

    assert_f32_bits(
        "sut_vs_independent.final_norm",
        &trace.final_norm,
        &independent.final_norm,
    );
    let final_norm = compare_hidden("final_norm", &official.final_norm, &trace.final_norm);
    assert_report_passes("final_norm", &final_norm, &FINAL_NORM_POLICY);

    assert_eq!(hash_f32(&official.input), preserved.0);
    assert_eq!(hash_f32(&official.raw_cos), preserved.1);
    assert_eq!(hash_f32(&official.raw_sin), preserved.2);
    assert_eq!(hash_f32(&final_norm_weight), preserved.3);
    for (layer, cache) in official.prefix_caches.iter().enumerate() {
        assert_eq!(hash_f32(&cache.keys), preserved.4[layer].0);
        assert_eq!(hash_f32(&cache.values), preserved.4[layer].1);
    }

    println!("runtime_ms={}", started.elapsed().as_secs_f64() * 1_000.0);
}

#[test]
#[ignore = "calibration-only helper for freezing per-depth stack decode policies"]
fn calibrate_independent_stack_decode_policies() {
    assert_model_lock_identity();
    let fixture = SafetensorsCatalog::open(fixture_path()).unwrap();
    let official = load_official_stack(&fixture);
    let model = SafetensorsCatalog::open(model_path()).unwrap();
    let final_norm_weight = model_tensor(&model, FINAL_NORM_WEIGHT, &[HIDDEN_SIZE as u64]);

    let tracker = Rc::new(WeightTracker::default());
    let independent =
        independent_decode_stack(&model, &official, &final_norm_weight, Rc::clone(&tracker));
    let negative = independent_decode_stack_with_cache_reset(&model, &official, &final_norm_weight);

    assert_eq!(tracker.loads.get(), LAYERS);
    assert_eq!(tracker.drops.get(), LAYERS);
    assert_eq!(tracker.live.get(), 0);
    assert_eq!(tracker.peak.get(), 1);

    for layer in 0..LAYERS {
        let output = compare_hidden(
            &format!("layer{layer:02}.output"),
            &official.layer_outputs[layer],
            &independent.layer_outputs[layer],
        );
        let appended_key = compare_row(
            &format!("layer{layer:02}.appended_key"),
            &official.appended_keys[layer],
            &independent.full_kv_caches[layer].keys[PREFIX_TOKENS * KEY_VALUE_WIDTH..],
            KEY_VALUE_WIDTH,
        );
        let appended_value = compare_row(
            &format!("layer{layer:02}.appended_value"),
            &official.appended_values[layer],
            &independent.full_kv_caches[layer].values[PREFIX_TOKENS * KEY_VALUE_WIDTH..],
            KEY_VALUE_WIDTH,
        );
        print_report(&format!("layer{layer:02}.output"), &output);
        print_report(&format!("layer{layer:02}.appended_key"), &appended_key);
        print_report(&format!("layer{layer:02}.appended_value"), &appended_value);

        let negative_output = compare_hidden(
            &format!("negative.layer{layer:02}.output"),
            &official.layer_outputs[layer],
            &negative.layer_outputs[layer],
        );
        print_report(
            &format!("negative.layer{layer:02}.output"),
            &negative_output,
        );
    }

    let final_norm = compare_hidden("final_norm", &official.final_norm, &independent.final_norm);
    let negative_final_norm = compare_hidden(
        "negative.final_norm",
        &official.final_norm,
        &negative.final_norm,
    );
    print_report("final_norm", &final_norm);
    print_report("negative.final_norm", &negative_final_norm);
}
