use std::{fs, path::PathBuf};

use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, DecoderLayerConfig, DecoderLayerParameters,
    DecoderLayerPrefillTrace, add_vectors_f32, apply_multimodal_rope_f32, causal_gqa_f32,
    decoder_layer_prefill_f32, linear_f32, pinned_decoder_layer_prefill_f32, rms_norm_f32, silu,
    write_pinned_decoder_prefill_kv_f32,
};
use pvlc_safetensors::SafetensorsCatalog;
use pvlc_testkit::{ComparisonAxes, ComparisonPolicy, compare_f32};

const TOKENS: usize = 332;
const HIDDEN_SIZE: usize = 1_024;
const INTERMEDIATE_SIZE: usize = 3_072;
const QUERY_HEADS: usize = 16;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const QUERY_WIDTH: usize = QUERY_HEADS * HEAD_DIM;
const KEY_VALUE_WIDTH: usize = KEY_VALUE_HEADS * HEAD_DIM;
const RMS_NORM_EPSILON: f32 = 1.0e-5;
const MROPE_SECTIONS: [usize; 3] = [16, 24, 24];
const SAMPLED_TOKENS: [usize; 5] = [0, 1, 17, 165, 331];
const FIXTURE_BLAKE3: &str = "30eed2f7a4d9336f8b3429d7294065c45214c4425f7559e8e2059ebd54c89522";

const INPUT: &str = "decoder.layer.00.input";
const NORM1: &str = "decoder.layer.00.norm1";
const QUERY: &str = "decoder.layer.00.q";
const KEY: &str = "decoder.layer.00.k";
const VALUE: &str = "decoder.layer.00.v";
const MROPE_QUERY: &str = "decoder.layer.00.mrope.q.token_major";
const MROPE_KEY: &str = "decoder.layer.00.mrope.k.token_major";
const CACHE_KEY: &str = "decoder.layer.00.kv.key.token_major";
const CACHE_VALUE: &str = "decoder.layer.00.kv.value.token_major";
const RAW_COS: &str = "decoder.rope.cos.axis_major";
const RAW_SIN: &str = "decoder.rope.sin.axis_major";
const ATTENTION_CONTEXT: &str = "decoder.layer.00.attention.context.token_major";
const ATTENTION_OUTPUT: &str = "decoder.layer.00.attention.output";
const ATTENTION_RESIDUAL: &str = "decoder.layer.00.attention.residual";
const NORM2: &str = "decoder.layer.00.norm2";
const MLP_GATE: &str = "decoder.layer.00.mlp.gate";
const MLP_UP: &str = "decoder.layer.00.mlp.up";
const MLP_ACTIVATION: &str = "decoder.layer.00.mlp.activation";
const MLP_DOWN: &str = "decoder.layer.00.mlp.down";
const OUTPUT: &str = "decoder.layer.00.output";

const INPUT_NORM_WEIGHT: &str = "model.layers.0.input_layernorm.weight";
const QUERY_WEIGHT: &str = "model.layers.0.self_attn.q_proj.weight";
const KEY_WEIGHT: &str = "model.layers.0.self_attn.k_proj.weight";
const VALUE_WEIGHT: &str = "model.layers.0.self_attn.v_proj.weight";
const ATTENTION_OUTPUT_WEIGHT: &str = "model.layers.0.self_attn.o_proj.weight";
const POST_ATTENTION_NORM_WEIGHT: &str = "model.layers.0.post_attention_layernorm.weight";
const GATE_WEIGHT: &str = "model.layers.0.mlp.gate_proj.weight";
const UP_WEIGHT: &str = "model.layers.0.mlp.up_proj.weight";
const DOWN_WEIGHT: &str = "model.layers.0.mlp.down_proj.weight";

struct TensorSpec {
    name: &'static str,
    shape: &'static [u64],
    raw_blake3: &'static str,
}

// Provenance: the oracle is the pinned remote-code snapshot
// models/snapshots/66317acc4c9fc17bd154591ce650735cd2855f3e/
// modeling_paddleocr_vl.py. The relevant implementation is RMSNorm at
// lines 451-466, the bias-free SwiGLU MLP at 211-231, decoder attention at
// 234-448, and the residual decoder layer at 472-520. Capture wiring and
// semantic stage validation live in
// tools/reference_capture/tests/test_transformers_oracle_integration.py
// around the L3 capture test and lines 647-680. The fixture was emitted by
// TransformersOracle.capture_artifacts for ocr.clean_latin.0001, not by the
// Rust implementation under test.
#[rustfmt::skip]
const TENSORS: [TensorSpec; 29] = [
    TensorSpec { name: ATTENTION_CONTEXT, shape: &[332, 16, 128], raw_blake3: "8e7a9de666991e9320c6909e84f2f9b0fcb5e5f1dac5d192a4eb80a179077b01" },
    TensorSpec { name: ATTENTION_OUTPUT, shape: &[332, 1024], raw_blake3: "4065a11aaa5fce98734cd37b6cb38291e01a826c735c6e3295db3fc29397aab8" },
    TensorSpec { name: ATTENTION_RESIDUAL, shape: &[332, 1024], raw_blake3: "fff4760d2fb144f525af4bde67ecee03a4f53fbfcfe6af6ff44ea4edb6f8eac3" },
    TensorSpec { name: INPUT, shape: &[332, 1024], raw_blake3: "8b46524fa1d413be6ee140b8af80c18547c3d505fd8f61c9781962e957f2da52" },
    TensorSpec { name: KEY, shape: &[332, 256], raw_blake3: "e729075dff364dca4699edb9c1e9e96ea856cffc2c5e091d96899a642eb0c02a" },
    TensorSpec { name: CACHE_KEY, shape: &[332, 2, 128], raw_blake3: "d0674f66af35df6bc48a6e60e3098bcaf9fa5e22ba9e20afe29bad44e82140f0" },
    TensorSpec { name: CACHE_VALUE, shape: &[332, 2, 128], raw_blake3: "fc6f30bc2fc420c6166a0380c29c349c71caf1944dab6671aca50c5eb5f27202" },
    TensorSpec { name: MLP_ACTIVATION, shape: &[332, 3072], raw_blake3: "f741f928b2e049f5063f455bfcdee515feaf6b43801aa065ea143863403bc1de" },
    TensorSpec { name: MLP_DOWN, shape: &[332, 1024], raw_blake3: "06d0dbd588511df0f72cf75092538941c19a9aedebbde59c986757182b5f3633" },
    TensorSpec { name: MLP_GATE, shape: &[332, 3072], raw_blake3: "6346f343d72cc55660073390ef3f84e138f7c7e46977806a38aba97fccafa24f" },
    TensorSpec { name: MLP_UP, shape: &[332, 3072], raw_blake3: "339cfc0afe1fd98a2b78a45da5a0d89fc3bf99b77d0799d2bb13774a9ef6aeca" },
    TensorSpec { name: MROPE_KEY, shape: &[332, 2, 128], raw_blake3: "d0674f66af35df6bc48a6e60e3098bcaf9fa5e22ba9e20afe29bad44e82140f0" },
    TensorSpec { name: MROPE_QUERY, shape: &[332, 16, 128], raw_blake3: "053c3f08e2034c513c7f8bafaa9216d6b4a29992ef7dc984b1d4b54db4527273" },
    TensorSpec { name: NORM1, shape: &[332, 1024], raw_blake3: "12ce0b7ba1b61c8edd12264be4178f9e566e78994cf64256b1c27f7cf8dcb76b" },
    TensorSpec { name: NORM2, shape: &[332, 1024], raw_blake3: "497c88807b977ca91799ec9d1e8bc87f9b33df12b9299e02d69be8da2881d9bf" },
    TensorSpec { name: OUTPUT, shape: &[332, 1024], raw_blake3: "7130fabeb187b3b9dc463fa0aea6c39775a674ae6497513f0b48b0216c9cac6e" },
    TensorSpec { name: QUERY, shape: &[332, 2048], raw_blake3: "888a4232edc8b6e404f34f494961b2e4645af3b6cebbd17f06ec66058b70b111" },
    TensorSpec { name: VALUE, shape: &[332, 256], raw_blake3: "fc6f30bc2fc420c6166a0380c29c349c71caf1944dab6671aca50c5eb5f27202" },
    TensorSpec { name: RAW_COS, shape: &[3, 332, 128], raw_blake3: "096287f2c2ee912105fbc747def39441b541c50b87b1330a8b3b3647b2b49654" },
    TensorSpec { name: RAW_SIN, shape: &[3, 332, 128], raw_blake3: "d34eff803104785331690d7f263c4f7ce44838f6083c5f2fb5ed987de613d310" },
    TensorSpec { name: INPUT_NORM_WEIGHT, shape: &[1024], raw_blake3: "06aaa3c896466a889d166becd2411bbb40ba40501a94f58673c883606005f792" },
    TensorSpec { name: DOWN_WEIGHT, shape: &[1024, 3072], raw_blake3: "b862080808c1ba24e2b7d4aea2eb63ed03eb9edccfc0fd5b63a4fc0e61b2835d" },
    TensorSpec { name: GATE_WEIGHT, shape: &[3072, 1024], raw_blake3: "4a1c6564dfff67e068261710687f78a093100b21d17628cbe20effebb2b0129e" },
    TensorSpec { name: UP_WEIGHT, shape: &[3072, 1024], raw_blake3: "efe8dfdb0a9ed2cde8ac8f68617cbb46e4dd8732e36a8e171e6162b242666af2" },
    TensorSpec { name: POST_ATTENTION_NORM_WEIGHT, shape: &[1024], raw_blake3: "bbe35de602323b40efd9c4568807d1c16c58efa70acb2d0033c42851ade095aa" },
    TensorSpec { name: KEY_WEIGHT, shape: &[256, 1024], raw_blake3: "60220e15ac1a117d74cf2897da638b3906cb3864d3e4cc61d21af1aaa44bb342" },
    TensorSpec { name: ATTENTION_OUTPUT_WEIGHT, shape: &[1024, 2048], raw_blake3: "c8fc88619213491a9cfde814c7b05f7bfe176e493228c7dd3bff4899db7645fe" },
    TensorSpec { name: QUERY_WEIGHT, shape: &[2048, 1024], raw_blake3: "fc56998413638a4540531881ef7d88be409744afed3568715f4001bd4c2bda39" },
    TensorSpec { name: VALUE_WEIGHT, shape: &[256, 1024], raw_blake3: "3318fdbf2aa80cd1de67170d674f7a8444a6ef04e6154f9114aa224826d186bd" },
];

#[rustfmt::skip]
const METADATA: [(&str, &str); 19] = [
    ("bias", "false"),
    ("case_id", "ocr.clean_latin.0001"),
    ("decoded_text", "JUL"),
    ("device", "mps"),
    ("dtype", "bfloat16"),
    ("fixture_schema", "pvlc.decoder_layer0.official.v1"),
    ("generated_tokens", "94013,898"),
    ("head_dim", "128"),
    ("hidden_size", "1024"),
    ("intermediate_size", "3072"),
    ("key_value_heads", "2"),
    ("model_id", "PaddlePaddle/PaddleOCR-VL-1.6"),
    ("model_revision", "66317acc4c9fc17bd154591ce650735cd2855f3e"),
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

// Frozen after independent CPU FP32 recomposition against the MPS BF16
// capture. These are stage-specific envelopes, not values derived at test run.
const NORM1_POLICY: ComparisonPolicy = policy(0.14, 0.000_7, 0.005, 0.002_5);
const QUERY_POLICY: ComparisonPolicy = policy(0.045, 0.001_5, 0.007, 0.003_3);
const KEY_POLICY: ComparisonPolicy = policy(0.04, 0.002_5, 0.012, 0.003_3);
const VALUE_POLICY: ComparisonPolicy = policy(0.002_1, 0.000_16, 0.000_6, 0.005_2);
const MROPE_QUERY_POLICY: ComparisonPolicy = policy(0.045, 0.001_7, 0.009, 0.003_8);
const MROPE_KEY_POLICY: ComparisonPolicy = policy(0.038, 0.002_8, 0.013, 0.003_6);
const CONTEXT_POLICY: ComparisonPolicy = policy(0.004_2, 0.000_11, 0.000_48, 0.005_3);
const ATTENTION_OUTPUT_POLICY: ComparisonPolicy = policy(0.003_4, 0.000_043, 0.000_19, 0.004_7);
const ATTENTION_RESIDUAL_POLICY: ComparisonPolicy = policy(0.035, 0.000_63, 0.005, 0.002_4);
const NORM2_POLICY: ComparisonPolicy = policy(0.056, 0.001_4, 0.006_8, 0.004_1);
const GATE_POLICY: ComparisonPolicy = policy(0.019, 0.001_1, 0.004, 0.004_1);
const UP_POLICY: ComparisonPolicy = policy(0.017, 0.000_9, 0.003_2, 0.004);
const ACTIVATION_POLICY: ComparisonPolicy = policy(0.056, 0.000_2, 0.001_35, 0.005_5);
const DOWN_POLICY: ComparisonPolicy = policy(0.075, 0.000_29, 0.001_3, 0.004_8);
const OUTPUT_POLICY: ComparisonPolicy = policy(0.11, 0.000_9, 0.007_5, 0.003_8);

#[derive(Clone, Copy)]
struct TracePolicies {
    norm1: ComparisonPolicy,
    query: ComparisonPolicy,
    key: ComparisonPolicy,
    value: ComparisonPolicy,
    mrope_query: ComparisonPolicy,
    mrope_key: ComparisonPolicy,
    context: ComparisonPolicy,
    attention_output: ComparisonPolicy,
    attention_residual: ComparisonPolicy,
    norm2: ComparisonPolicy,
    gate: ComparisonPolicy,
    up: ComparisonPolicy,
    activation: ComparisonPolicy,
    down: ComparisonPolicy,
    output: ComparisonPolicy,
}

const FULL_TRACE_POLICIES: TracePolicies = TracePolicies {
    norm1: NORM1_POLICY,
    query: QUERY_POLICY,
    key: KEY_POLICY,
    value: VALUE_POLICY,
    mrope_query: MROPE_QUERY_POLICY,
    mrope_key: MROPE_KEY_POLICY,
    context: CONTEXT_POLICY,
    attention_output: ATTENTION_OUTPUT_POLICY,
    attention_residual: ATTENTION_RESIDUAL_POLICY,
    norm2: NORM2_POLICY,
    gate: GATE_POLICY,
    up: UP_POLICY,
    activation: ACTIVATION_POLICY,
    down: DOWN_POLICY,
    output: OUTPUT_POLICY,
};

// Independently measured one-token max/mean/p99/relative-L2, in trace order:
// norm1 .015666/.00063734/.0031128/.0022052
// q     .0084252/.00099474/.0044513/.0021695
// k     .0095472/.00182963/.0084929/.0021648
// v     .00043611/.00011711/.00039993/.0056339
// mQ=q; mK=k; cache-key=mK; cache-value=v; context=v for causal token zero
// O     .00095174/.00002400/.00008129/.0036425
// res   .00070760/.00003094/.00012584/.0025557
// norm2 .013967/.00129738/.0057909/.0033534
// gate  .0083065/.00094993/.0034882/.0031799
// up    .0052168/.00079576/.0027272/.0031884
// act   .032716/.00023115/.0014637/.0044266
// down  .019469/.00038984/.0015101/.0050296
// out   .015618/.00040823/.0016825/.0045516
// Each frozen envelope below rounds upward from 1.25x the corresponding
// measurement. Full-332 envelopes remain separate and unchanged above.
const ONE_TOKEN_TRACE_POLICIES: TracePolicies = TracePolicies {
    norm1: policy(0.019_6, 0.000_797, 0.003_9, 0.002_76),
    query: policy(0.010_6, 0.001_25, 0.005_57, 0.002_72),
    key: policy(0.012, 0.002_29, 0.010_7, 0.002_71),
    value: policy(0.000_546, 0.000_147, 0.000_5, 0.007_05),
    mrope_query: policy(0.010_6, 0.001_25, 0.005_57, 0.002_72),
    mrope_key: policy(0.012, 0.002_29, 0.010_7, 0.002_71),
    context: policy(0.000_546, 0.000_147, 0.000_5, 0.007_05),
    attention_output: policy(0.001_19, 0.000_03, 0.000_102, 0.004_56),
    attention_residual: policy(0.000_885, 0.000_038_7, 0.000_158, 0.003_2),
    norm2: policy(0.017_5, 0.001_63, 0.007_24, 0.004_2),
    gate: policy(0.010_4, 0.001_19, 0.004_37, 0.003_98),
    up: policy(0.006_53, 0.000_995, 0.003_41, 0.003_99),
    activation: policy(0.040_9, 0.000_289, 0.001_83, 0.005_54),
    down: policy(0.024_4, 0.000_488, 0.001_89, 0.006_29),
    output: policy(0.019_6, 0.000_511, 0.002_11, 0.005_69),
};

#[derive(Debug)]
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

impl OwnedParameters {
    fn load(catalog: &SafetensorsCatalog) -> Self {
        Self {
            input_norm_weight: catalog.load_tensor_f32(INPUT_NORM_WEIGHT).unwrap(),
            query_weight: catalog.load_tensor_f32(QUERY_WEIGHT).unwrap(),
            key_weight: catalog.load_tensor_f32(KEY_WEIGHT).unwrap(),
            value_weight: catalog.load_tensor_f32(VALUE_WEIGHT).unwrap(),
            attention_output_weight: catalog.load_tensor_f32(ATTENTION_OUTPUT_WEIGHT).unwrap(),
            post_attention_norm_weight: catalog
                .load_tensor_f32(POST_ATTENTION_NORM_WEIGHT)
                .unwrap(),
            gate_weight: catalog.load_tensor_f32(GATE_WEIGHT).unwrap(),
            up_weight: catalog.load_tensor_f32(UP_WEIGHT).unwrap(),
            down_weight: catalog.load_tensor_f32(DOWN_WEIGHT).unwrap(),
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

    fn operand_mut(&mut self, index: usize) -> &mut Vec<f32> {
        match index {
            0 => &mut self.input_norm_weight,
            1 => &mut self.query_weight,
            2 => &mut self.key_weight,
            3 => &mut self.value_weight,
            4 => &mut self.attention_output_weight,
            5 => &mut self.post_attention_norm_weight,
            6 => &mut self.gate_weight,
            7 => &mut self.up_weight,
            8 => &mut self.down_weight,
            _ => panic!("weight operand {index} is out of range"),
        }
    }
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/decoder-layer0-official-v1.safetensors")
}

fn hash_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn tensor_bytes(catalog: &SafetensorsCatalog, name: &str) -> Vec<u8> {
    let mut bytes = Vec::new();
    catalog.copy_tensor_to(name, &mut bytes).unwrap();
    bytes
}

fn load(catalog: &SafetensorsCatalog, name: &str) -> Vec<f32> {
    catalog.load_tensor_f32(name).unwrap()
}

fn prefix_rows(catalog: &SafetensorsCatalog, name: &str, rows: usize, width: usize) -> Vec<f32> {
    let mut values = load(catalog, name);
    assert_eq!(values.len(), TOKENS * width, "{name}");
    values.truncate(rows * width);
    values
}

fn selected_rows(catalog: &SafetensorsCatalog, name: &str, width: usize) -> Vec<f32> {
    let values = load(catalog, name);
    assert_eq!(values.len(), TOKENS * width, "{name}");
    SAMPLED_TOKENS
        .iter()
        .flat_map(|&token| values[token * width..(token + 1) * width].iter().copied())
        .collect()
}

fn prefix_axis_tables(catalog: &SafetensorsCatalog, name: &str, rows: usize) -> Vec<f32> {
    let values = load(catalog, name);
    assert_eq!(values.len(), 3 * TOKENS * HEAD_DIM, "{name}");
    let mut prefix = Vec::with_capacity(3 * rows * HEAD_DIM);
    for axis in 0..3 {
        let start = axis * TOKENS * HEAD_DIM;
        prefix.extend_from_slice(&values[start..start + rows * HEAD_DIM]);
    }
    prefix
}

fn config(tokens: usize) -> DecoderLayerConfig {
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

fn independent_trace(
    input: &[f32],
    tokens: usize,
    raw_cos: &[f32],
    raw_sin: &[f32],
    parameters: &OwnedParameters,
) -> DecoderLayerPrefillTrace {
    let zero_query = vec![0.0; QUERY_WIDTH];
    let zero_key_value = vec![0.0; KEY_VALUE_WIDTH];
    let zero_hidden = vec![0.0; HIDDEN_SIZE];
    let zero_intermediate = vec![0.0; INTERMEDIATE_SIZE];
    let norm1 = rms_norm_f32(
        input,
        tokens,
        HIDDEN_SIZE,
        &parameters.input_norm_weight,
        RMS_NORM_EPSILON,
    )
    .unwrap();
    let query = linear_f32(
        &norm1,
        tokens,
        HIDDEN_SIZE,
        &parameters.query_weight,
        &zero_query,
        QUERY_WIDTH,
    )
    .unwrap();
    let key = linear_f32(
        &norm1,
        tokens,
        HIDDEN_SIZE,
        &parameters.key_weight,
        &zero_key_value,
        KEY_VALUE_WIDTH,
    )
    .unwrap();
    let value = linear_f32(
        &norm1,
        tokens,
        HIDDEN_SIZE,
        &parameters.value_weight,
        &zero_key_value,
        KEY_VALUE_WIDTH,
    )
    .unwrap();
    let (mrope_query, mrope_key) = apply_multimodal_rope_f32(
        &query,
        &key,
        tokens,
        QUERY_HEADS,
        KEY_VALUE_HEADS,
        HEAD_DIM,
        raw_cos,
        raw_sin,
        MROPE_SECTIONS,
    )
    .unwrap();
    let kv_cache = write_pinned_decoder_prefill_kv_f32(&mrope_key, &value, tokens).unwrap();
    let attention_context = causal_gqa_f32(
        &mrope_query,
        &kv_cache.keys,
        &kv_cache.values,
        tokens,
        QUERY_HEADS,
        KEY_VALUE_HEADS,
        HEAD_DIM,
    )
    .unwrap();
    let attention_output = linear_f32(
        &attention_context,
        tokens,
        QUERY_WIDTH,
        &parameters.attention_output_weight,
        &zero_hidden,
        HIDDEN_SIZE,
    )
    .unwrap();
    let attention_residual = add_vectors_f32(input, &attention_output).unwrap();
    let norm2 = rms_norm_f32(
        &attention_residual,
        tokens,
        HIDDEN_SIZE,
        &parameters.post_attention_norm_weight,
        RMS_NORM_EPSILON,
    )
    .unwrap();
    let mlp_gate = linear_f32(
        &norm2,
        tokens,
        HIDDEN_SIZE,
        &parameters.gate_weight,
        &zero_intermediate,
        INTERMEDIATE_SIZE,
    )
    .unwrap();
    let mlp_up = linear_f32(
        &norm2,
        tokens,
        HIDDEN_SIZE,
        &parameters.up_weight,
        &zero_intermediate,
        INTERMEDIATE_SIZE,
    )
    .unwrap();
    let mlp_activation = mlp_gate
        .iter()
        .zip(&mlp_up)
        .map(|(&gate, &up)| silu(gate) * up)
        .collect::<Vec<_>>();
    let mlp_down = linear_f32(
        &mlp_activation,
        tokens,
        INTERMEDIATE_SIZE,
        &parameters.down_weight,
        &zero_hidden,
        HIDDEN_SIZE,
    )
    .unwrap();
    let output = add_vectors_f32(&attention_residual, &mlp_down).unwrap();
    DecoderLayerPrefillTrace {
        norm1,
        query,
        key,
        value,
        mrope_query,
        mrope_key,
        kv_cache,
        attention_context,
        attention_output,
        attention_residual,
        norm2,
        mlp_gate,
        mlp_up,
        mlp_activation,
        mlp_down,
        output,
    }
}

fn assert_trace_exact(actual: &DecoderLayerPrefillTrace, expected: &DecoderLayerPrefillTrace) {
    assert_eq!(actual.norm1, expected.norm1);
    assert_eq!(actual.query, expected.query);
    assert_eq!(actual.key, expected.key);
    assert_eq!(actual.value, expected.value);
    assert_eq!(actual.mrope_query, expected.mrope_query);
    assert_eq!(actual.mrope_key, expected.mrope_key);
    assert_eq!(actual.kv_cache.keys, expected.kv_cache.keys);
    assert_eq!(actual.kv_cache.values, expected.kv_cache.values);
    assert_eq!(actual.kv_cache.tokens, expected.kv_cache.tokens);
    assert_eq!(
        actual.kv_cache.key_value_heads,
        expected.kv_cache.key_value_heads
    );
    assert_eq!(actual.kv_cache.head_dim, expected.kv_cache.head_dim);
    assert_eq!(actual.attention_context, expected.attention_context);
    assert_eq!(actual.attention_output, expected.attention_output);
    assert_eq!(actual.attention_residual, expected.attention_residual);
    assert_eq!(actual.norm2, expected.norm2);
    assert_eq!(actual.mlp_gate, expected.mlp_gate);
    assert_eq!(actual.mlp_up, expected.mlp_up);
    assert_eq!(actual.mlp_activation, expected.mlp_activation);
    assert_eq!(actual.mlp_down, expected.mlp_down);
    assert_eq!(actual.output, expected.output);
}

fn assert_stage(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    rows: usize,
    width: usize,
    comparison_policy: &ComparisonPolicy,
) {
    let report = compare_f32(
        expected,
        actual,
        &[rows, width],
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
    rows: usize,
    width: usize,
    comparison_policy: &ComparisonPolicy,
) {
    let report = compare_f32(
        expected,
        actual,
        &[rows, width],
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

fn assert_trace(
    catalog: &SafetensorsCatalog,
    trace: &DecoderLayerPrefillTrace,
    rows: usize,
    policies: &TracePolicies,
) {
    let check = |name: &str, actual: &[f32], width: usize, comparison_policy: &ComparisonPolicy| {
        let expected = prefix_rows(catalog, name, rows, width);
        assert_stage(name, &expected, actual, rows, width, comparison_policy);
    };
    check(NORM1, &trace.norm1, HIDDEN_SIZE, &policies.norm1);
    check(QUERY, &trace.query, QUERY_WIDTH, &policies.query);
    check(KEY, &trace.key, KEY_VALUE_WIDTH, &policies.key);
    check(VALUE, &trace.value, KEY_VALUE_WIDTH, &policies.value);
    check(
        MROPE_QUERY,
        &trace.mrope_query,
        QUERY_WIDTH,
        &policies.mrope_query,
    );
    check(
        MROPE_KEY,
        &trace.mrope_key,
        KEY_VALUE_WIDTH,
        &policies.mrope_key,
    );
    check(
        CACHE_KEY,
        &trace.kv_cache.keys,
        KEY_VALUE_WIDTH,
        &policies.mrope_key,
    );
    check(
        CACHE_VALUE,
        &trace.kv_cache.values,
        KEY_VALUE_WIDTH,
        &policies.value,
    );
    assert_eq!(trace.kv_cache.tokens, rows);
    assert_eq!(trace.kv_cache.key_value_heads, KEY_VALUE_HEADS);
    assert_eq!(trace.kv_cache.head_dim, HEAD_DIM);
    check(
        ATTENTION_CONTEXT,
        &trace.attention_context,
        QUERY_WIDTH,
        &policies.context,
    );
    check(
        ATTENTION_OUTPUT,
        &trace.attention_output,
        HIDDEN_SIZE,
        &policies.attention_output,
    );
    check(
        ATTENTION_RESIDUAL,
        &trace.attention_residual,
        HIDDEN_SIZE,
        &policies.attention_residual,
    );
    check(NORM2, &trace.norm2, HIDDEN_SIZE, &policies.norm2);
    check(MLP_GATE, &trace.mlp_gate, INTERMEDIATE_SIZE, &policies.gate);
    check(MLP_UP, &trace.mlp_up, INTERMEDIATE_SIZE, &policies.up);
    check(
        MLP_ACTIVATION,
        &trace.mlp_activation,
        INTERMEDIATE_SIZE,
        &policies.activation,
    );
    check(MLP_DOWN, &trace.mlp_down, HIDDEN_SIZE, &policies.down);
    check(OUTPUT, &trace.output, HIDDEN_SIZE, &policies.output);
}

fn assert_error<T>(result: Result<T, CpuRefError>, expected: CpuRefErrorCode) {
    match result {
        Err(error) => assert_eq!(error.code(), expected),
        Ok(_) => panic!("expected {expected:?}"),
    }
}

#[test]
fn authenticates_self_contained_official_fixture() {
    let path = fixture_path();
    let fixture = fs::read(&path).unwrap();
    assert_eq!(fixture.len(), 44_637_696);
    assert_eq!(hash_bytes(&fixture), FIXTURE_BLAKE3);

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
}

#[test]
fn sampled_official_components_match_and_negative_controls_are_rejected() {
    let catalog = SafetensorsCatalog::open(fixture_path()).unwrap();
    let parameters = OwnedParameters::load(&catalog);
    let rows = SAMPLED_TOKENS.len();
    let input = selected_rows(&catalog, INPUT, HIDDEN_SIZE);
    let zero_query = vec![0.0; QUERY_WIDTH];
    let zero_key_value = vec![0.0; KEY_VALUE_WIDTH];
    let zero_hidden = vec![0.0; HIDDEN_SIZE];
    let zero_intermediate = vec![0.0; INTERMEDIATE_SIZE];

    let norm1 = rms_norm_f32(
        &input,
        rows,
        HIDDEN_SIZE,
        &parameters.input_norm_weight,
        RMS_NORM_EPSILON,
    )
    .unwrap();
    let official_norm1 = selected_rows(&catalog, NORM1, HIDDEN_SIZE);
    assert_stage(
        NORM1,
        &official_norm1,
        &norm1,
        rows,
        HIDDEN_SIZE,
        &NORM1_POLICY,
    );

    let query = linear_f32(
        &official_norm1,
        rows,
        HIDDEN_SIZE,
        &parameters.query_weight,
        &zero_query,
        QUERY_WIDTH,
    )
    .unwrap();
    assert_stage(
        QUERY,
        &selected_rows(&catalog, QUERY, QUERY_WIDTH),
        &query,
        rows,
        QUERY_WIDTH,
        &QUERY_POLICY,
    );
    let key = linear_f32(
        &official_norm1,
        rows,
        HIDDEN_SIZE,
        &parameters.key_weight,
        &zero_key_value,
        KEY_VALUE_WIDTH,
    )
    .unwrap();
    assert_stage(
        KEY,
        &selected_rows(&catalog, KEY, KEY_VALUE_WIDTH),
        &key,
        rows,
        KEY_VALUE_WIDTH,
        &KEY_POLICY,
    );
    let value = linear_f32(
        &official_norm1,
        rows,
        HIDDEN_SIZE,
        &parameters.value_weight,
        &zero_key_value,
        KEY_VALUE_WIDTH,
    )
    .unwrap();
    assert_stage(
        VALUE,
        &selected_rows(&catalog, VALUE, KEY_VALUE_WIDTH),
        &value,
        rows,
        KEY_VALUE_WIDTH,
        &VALUE_POLICY,
    );

    // Isolate the output projection by feeding its official attention-context
    // stage input rather than a value recomputed by the layer under test.
    let official_context = selected_rows(&catalog, ATTENTION_CONTEXT, QUERY_WIDTH);
    let attention_output = linear_f32(
        &official_context,
        rows,
        QUERY_WIDTH,
        &parameters.attention_output_weight,
        &zero_hidden,
        HIDDEN_SIZE,
    )
    .unwrap();
    assert_stage(
        ATTENTION_OUTPUT,
        &selected_rows(&catalog, ATTENTION_OUTPUT, HIDDEN_SIZE),
        &attention_output,
        rows,
        HIDDEN_SIZE,
        &ATTENTION_OUTPUT_POLICY,
    );
    let official_attention_output = selected_rows(&catalog, ATTENTION_OUTPUT, HIDDEN_SIZE);
    let attention_residual = add_vectors_f32(&input, &official_attention_output).unwrap();
    let official_attention_residual = selected_rows(&catalog, ATTENTION_RESIDUAL, HIDDEN_SIZE);
    assert_stage(
        ATTENTION_RESIDUAL,
        &official_attention_residual,
        &attention_residual,
        rows,
        HIDDEN_SIZE,
        &ATTENTION_RESIDUAL_POLICY,
    );
    let norm2 = rms_norm_f32(
        &official_attention_residual,
        rows,
        HIDDEN_SIZE,
        &parameters.post_attention_norm_weight,
        RMS_NORM_EPSILON,
    )
    .unwrap();
    let official_norm2 = selected_rows(&catalog, NORM2, HIDDEN_SIZE);
    assert_stage(
        NORM2,
        &official_norm2,
        &norm2,
        rows,
        HIDDEN_SIZE,
        &NORM2_POLICY,
    );
    let gate = linear_f32(
        &official_norm2,
        rows,
        HIDDEN_SIZE,
        &parameters.gate_weight,
        &zero_intermediate,
        INTERMEDIATE_SIZE,
    )
    .unwrap();
    let official_gate = selected_rows(&catalog, MLP_GATE, INTERMEDIATE_SIZE);
    assert_stage(
        MLP_GATE,
        &official_gate,
        &gate,
        rows,
        INTERMEDIATE_SIZE,
        &GATE_POLICY,
    );
    let up = linear_f32(
        &official_norm2,
        rows,
        HIDDEN_SIZE,
        &parameters.up_weight,
        &zero_intermediate,
        INTERMEDIATE_SIZE,
    )
    .unwrap();
    let official_up = selected_rows(&catalog, MLP_UP, INTERMEDIATE_SIZE);
    assert_stage(
        MLP_UP,
        &official_up,
        &up,
        rows,
        INTERMEDIATE_SIZE,
        &UP_POLICY,
    );
    let activation = official_gate
        .iter()
        .zip(&official_up)
        .map(|(&gate, &up)| silu(gate) * up)
        .collect::<Vec<_>>();
    let official_activation = selected_rows(&catalog, MLP_ACTIVATION, INTERMEDIATE_SIZE);
    assert_stage(
        MLP_ACTIVATION,
        &official_activation,
        &activation,
        rows,
        INTERMEDIATE_SIZE,
        &ACTIVATION_POLICY,
    );
    let down = linear_f32(
        &official_activation,
        rows,
        INTERMEDIATE_SIZE,
        &parameters.down_weight,
        &zero_hidden,
        HIDDEN_SIZE,
    )
    .unwrap();
    let official_down = selected_rows(&catalog, MLP_DOWN, HIDDEN_SIZE);
    assert_stage(
        MLP_DOWN,
        &official_down,
        &down,
        rows,
        HIDDEN_SIZE,
        &DOWN_POLICY,
    );
    let output = add_vectors_f32(&official_attention_residual, &official_down).unwrap();
    let official_output = selected_rows(&catalog, OUTPUT, HIDDEN_SIZE);
    assert_stage(
        OUTPUT,
        &official_output,
        &output,
        rows,
        HIDDEN_SIZE,
        &OUTPUT_POLICY,
    );

    let swapped_activation = official_gate
        .iter()
        .zip(&official_up)
        .map(|(&gate, &up)| silu(up) * gate)
        .collect::<Vec<_>>();
    let swapped_down = linear_f32(
        &swapped_activation,
        rows,
        INTERMEDIATE_SIZE,
        &parameters.down_weight,
        &zero_hidden,
        HIDDEN_SIZE,
    )
    .unwrap();
    let swapped_output = add_vectors_f32(&official_attention_residual, &swapped_down).unwrap();
    assert_rejected(
        "swapped gate/up",
        &official_output,
        &swapped_output,
        rows,
        HIDDEN_SIZE,
        &OUTPUT_POLICY,
    );

    let skipped_gate = linear_f32(
        &official_attention_residual,
        rows,
        HIDDEN_SIZE,
        &parameters.gate_weight,
        &zero_intermediate,
        INTERMEDIATE_SIZE,
    )
    .unwrap();
    let skipped_up = linear_f32(
        &official_attention_residual,
        rows,
        HIDDEN_SIZE,
        &parameters.up_weight,
        &zero_intermediate,
        INTERMEDIATE_SIZE,
    )
    .unwrap();
    let skipped_activation = skipped_gate
        .iter()
        .zip(skipped_up)
        .map(|(&gate, up)| silu(gate) * up)
        .collect::<Vec<_>>();
    assert_rejected(
        "skipped post-attention RMSNorm",
        &official_activation,
        &skipped_activation,
        rows,
        INTERMEDIATE_SIZE,
        &ACTIVATION_POLICY,
    );
}

#[test]
fn pinned_one_token_layer_composes_every_stage_from_fixture_inputs_and_weights() {
    let catalog = SafetensorsCatalog::open(fixture_path()).unwrap();
    let parameters = OwnedParameters::load(&catalog);
    let input = prefix_rows(&catalog, INPUT, 1, HIDDEN_SIZE);
    let raw_cos = prefix_axis_tables(&catalog, RAW_COS, 1);
    let raw_sin = prefix_axis_tables(&catalog, RAW_SIN, 1);
    let preserved = (input.clone(), raw_cos.clone(), raw_sin.clone());

    let pinned =
        pinned_decoder_layer_prefill_f32(&input, 1, &raw_cos, &raw_sin, parameters.borrowed())
            .unwrap();
    assert_trace(&catalog, &pinned, 1, &ONE_TOKEN_TRACE_POLICIES);

    let generic =
        decoder_layer_prefill_f32(&input, config(1), &raw_cos, &raw_sin, parameters.borrowed())
            .unwrap();
    assert_eq!(generic, pinned);

    let zero_hidden = vec![0.0; HIDDEN_SIZE];
    let zero_intermediate = vec![0.0; INTERMEDIATE_SIZE];
    let swapped_activation = pinned
        .mlp_gate
        .iter()
        .zip(&pinned.mlp_up)
        .map(|(&gate, &up)| silu(up) * gate)
        .collect::<Vec<_>>();
    let swapped_down = linear_f32(
        &swapped_activation,
        1,
        INTERMEDIATE_SIZE,
        &parameters.down_weight,
        &zero_hidden,
        HIDDEN_SIZE,
    )
    .unwrap();
    let swapped_output = add_vectors_f32(&pinned.attention_residual, &swapped_down).unwrap();
    assert_rejected(
        "one-token swapped gate/up",
        &prefix_rows(&catalog, OUTPUT, 1, HIDDEN_SIZE),
        &swapped_output,
        1,
        HIDDEN_SIZE,
        &ONE_TOKEN_TRACE_POLICIES.output,
    );

    let skipped_gate = linear_f32(
        &pinned.attention_residual,
        1,
        HIDDEN_SIZE,
        &parameters.gate_weight,
        &zero_intermediate,
        INTERMEDIATE_SIZE,
    )
    .unwrap();
    let skipped_up = linear_f32(
        &pinned.attention_residual,
        1,
        HIDDEN_SIZE,
        &parameters.up_weight,
        &zero_intermediate,
        INTERMEDIATE_SIZE,
    )
    .unwrap();
    let skipped_activation = skipped_gate
        .iter()
        .zip(skipped_up)
        .map(|(&gate, up)| silu(gate) * up)
        .collect::<Vec<_>>();
    assert_rejected(
        "one-token skipped post-attention RMSNorm",
        &prefix_rows(&catalog, MLP_ACTIVATION, 1, INTERMEDIATE_SIZE),
        &skipped_activation,
        1,
        INTERMEDIATE_SIZE,
        &ONE_TOKEN_TRACE_POLICIES.activation,
    );
    assert_eq!(input, preserved.0);
    assert_eq!(raw_cos, preserved.1);
    assert_eq!(raw_sin, preserved.2);
}

#[test]
fn pinned_four_token_wrapper_matches_independent_accepted_primitive_composition() {
    const ROWS: usize = 4;
    let catalog = SafetensorsCatalog::open(fixture_path()).unwrap();
    let parameters = OwnedParameters::load(&catalog);
    let input = prefix_rows(&catalog, INPUT, ROWS, HIDDEN_SIZE);
    let raw_cos = prefix_axis_tables(&catalog, RAW_COS, ROWS);
    let raw_sin = prefix_axis_tables(&catalog, RAW_SIN, ROWS);
    let preserved = (input.clone(), raw_cos.clone(), raw_sin.clone());

    let expected = independent_trace(&input, ROWS, &raw_cos, &raw_sin, &parameters);
    let actual =
        pinned_decoder_layer_prefill_f32(&input, ROWS, &raw_cos, &raw_sin, parameters.borrowed())
            .unwrap();
    assert_trace_exact(&actual, &expected);
    assert_eq!(actual.kv_cache.tokens, ROWS);
    assert_eq!(actual.output.len(), ROWS * HIDDEN_SIZE);
    assert_eq!(input, preserved.0);
    assert_eq!(raw_cos, preserved.1);
    assert_eq!(raw_sin, preserved.2);
}

fn operand_mut<'a>(
    operand: usize,
    input: &'a mut Vec<f32>,
    raw_cos: &'a mut Vec<f32>,
    raw_sin: &'a mut Vec<f32>,
    parameters: &'a mut OwnedParameters,
) -> &'a mut Vec<f32> {
    match operand {
        0 => input,
        1 => raw_cos,
        2 => raw_sin,
        3..=11 => parameters.operand_mut(operand - 3),
        _ => panic!("operand {operand} is out of range"),
    }
}

#[test]
fn pinned_layer_fail_closes_for_geometry_lengths_nonfinite_and_late_malformed_weight() {
    let catalog = SafetensorsCatalog::open(fixture_path()).unwrap();
    let mut parameters = OwnedParameters::load(&catalog);
    let mut input = prefix_rows(&catalog, INPUT, 1, HIDDEN_SIZE);
    let mut raw_cos = prefix_axis_tables(&catalog, RAW_COS, 1);
    let mut raw_sin = prefix_axis_tables(&catalog, RAW_SIN, 1);

    assert_error(
        pinned_decoder_layer_prefill_f32(&[], 0, &[], &[], parameters.borrowed()),
        CpuRefErrorCode::DimensionMismatch,
    );
    assert_error(
        pinned_decoder_layer_prefill_f32(&[], usize::MAX, &[], &[], parameters.borrowed()),
        CpuRefErrorCode::DimensionMismatch,
    );

    // The three data operands and all nine weights reject both short and long
    // buffers. Restoration after each call keeps the fixture-backed baseline
    // exact and avoids substituting a separate checkpoint dependency.
    for operand in 0..12 {
        let removed = operand_mut(
            operand,
            &mut input,
            &mut raw_cos,
            &mut raw_sin,
            &mut parameters,
        )
        .pop()
        .unwrap();
        assert_error(
            pinned_decoder_layer_prefill_f32(&input, 1, &raw_cos, &raw_sin, parameters.borrowed()),
            CpuRefErrorCode::DimensionMismatch,
        );
        operand_mut(
            operand,
            &mut input,
            &mut raw_cos,
            &mut raw_sin,
            &mut parameters,
        )
        .push(removed);

        operand_mut(
            operand,
            &mut input,
            &mut raw_cos,
            &mut raw_sin,
            &mut parameters,
        )
        .push(0.0);
        assert_error(
            pinned_decoder_layer_prefill_f32(&input, 1, &raw_cos, &raw_sin, parameters.borrowed()),
            CpuRefErrorCode::DimensionMismatch,
        );
        let _ = operand_mut(
            operand,
            &mut input,
            &mut raw_cos,
            &mut raw_sin,
            &mut parameters,
        )
        .pop();
    }

    for operand in 0..12 {
        let len = {
            let values = operand_mut(
                operand,
                &mut input,
                &mut raw_cos,
                &mut raw_sin,
                &mut parameters,
            );
            values.len()
        };
        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for offset in [0, len / 2, len - 1] {
                let original = {
                    let values = operand_mut(
                        operand,
                        &mut input,
                        &mut raw_cos,
                        &mut raw_sin,
                        &mut parameters,
                    );
                    std::mem::replace(&mut values[offset], nonfinite)
                };
                assert_error(
                    pinned_decoder_layer_prefill_f32(
                        &input,
                        1,
                        &raw_cos,
                        &raw_sin,
                        parameters.borrowed(),
                    ),
                    CpuRefErrorCode::NonFiniteInput,
                );
                operand_mut(
                    operand,
                    &mut input,
                    &mut raw_cos,
                    &mut raw_sin,
                    &mut parameters,
                )[offset] = original;
            }
        }
    }

    // All shapes are prevalidated before arithmetic or finiteness scanning can
    // obscure a malformed late-stage weight.
    input[0] = f32::NAN;
    let removed = parameters.down_weight.pop().unwrap();
    assert_error(
        pinned_decoder_layer_prefill_f32(&input, 1, &raw_cos, &raw_sin, parameters.borrowed()),
        CpuRefErrorCode::DimensionMismatch,
    );
    parameters.down_weight.push(removed);
}

#[derive(Clone, Copy)]
enum WrongAttention {
    NonCausal,
    MissingScale,
    AlternatingKvHeads,
}

fn attention_index(token: usize, head: usize, dim: usize, heads: usize, head_dim: usize) -> usize {
    (token * heads + head) * head_dim + dim
}

// Deliberately incorrect test-only attention oracles. Keeping these outside
// production makes each negative control structurally independent from the
// implementation whose causal mask, scaling, and direct GQA mapping it checks.
fn wrong_attention(query: &[f32], key: &[f32], value: &[f32], mode: WrongAttention) -> Vec<f32> {
    let mut output = vec![0.0; query.len()];
    let query_heads_per_kv = QUERY_HEADS / KEY_VALUE_HEADS;
    let correct_scale = (HEAD_DIM as f32).sqrt().recip();
    for query_token in 0..TOKENS {
        for query_head in 0..QUERY_HEADS {
            let key_value_head = match mode {
                WrongAttention::AlternatingKvHeads => query_head % KEY_VALUE_HEADS,
                WrongAttention::NonCausal | WrongAttention::MissingScale => {
                    query_head / query_heads_per_kv
                }
            };
            let key_tokens = match mode {
                WrongAttention::NonCausal => TOKENS,
                WrongAttention::MissingScale | WrongAttention::AlternatingKvHeads => {
                    query_token + 1
                }
            };
            let scale = match mode {
                WrongAttention::MissingScale => 1.0,
                WrongAttention::NonCausal | WrongAttention::AlternatingKvHeads => correct_scale,
            };
            let mut probabilities = Vec::with_capacity(key_tokens);
            for key_token in 0..key_tokens {
                let mut score = 0.0_f32;
                for dim in 0..HEAD_DIM {
                    score += query
                        [attention_index(query_token, query_head, dim, QUERY_HEADS, HEAD_DIM)]
                        * key[attention_index(
                            key_token,
                            key_value_head,
                            dim,
                            KEY_VALUE_HEADS,
                            HEAD_DIM,
                        )];
                }
                probabilities.push(score * scale);
            }
            let maximum = probabilities
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0_f32;
            for probability in &mut probabilities {
                *probability = (*probability - maximum).exp();
                denominator += *probability;
            }
            for probability in &mut probabilities {
                *probability /= denominator;
            }
            for dim in 0..HEAD_DIM {
                let mut weighted = 0.0_f32;
                for (key_token, &probability) in probabilities.iter().enumerate() {
                    weighted += probability
                        * value[attention_index(
                            key_token,
                            key_value_head,
                            dim,
                            KEY_VALUE_HEADS,
                            HEAD_DIM,
                        )];
                }
                output[attention_index(query_token, query_head, dim, QUERY_HEADS, HEAD_DIM)] =
                    weighted;
            }
        }
    }
    output
}

#[test]
#[ignore = "release-only full 332-token decoder-layer oracle gate"]
fn full_official_layer_recomposes_every_value_and_rejects_attention_ablations() {
    let catalog = SafetensorsCatalog::open(fixture_path()).unwrap();
    let parameters = OwnedParameters::load(&catalog);
    let input = load(&catalog, INPUT);
    let raw_cos = load(&catalog, RAW_COS);
    let raw_sin = load(&catalog, RAW_SIN);
    let preserved = (input.clone(), raw_cos.clone(), raw_sin.clone());

    // The composed gate starts only from the captured layer input, raw
    // axis-major RoPE tables, and the nine captured model weights. No captured
    // intermediate is fed into the implementation under test.
    let trace =
        pinned_decoder_layer_prefill_f32(&input, TOKENS, &raw_cos, &raw_sin, parameters.borrowed())
            .unwrap();
    assert_trace(&catalog, &trace, TOKENS, &FULL_TRACE_POLICIES);
    assert_eq!(trace.mrope_key, trace.kv_cache.keys);
    assert_eq!(trace.value, trace.kv_cache.values);
    assert_eq!(input, preserved.0);
    assert_eq!(raw_cos, preserved.1);
    assert_eq!(raw_sin, preserved.2);

    let expected_context = load(&catalog, ATTENTION_CONTEXT);
    for (label, mode) in [
        ("noncausal attention", WrongAttention::NonCausal),
        ("missing attention scale", WrongAttention::MissingScale),
        (
            "alternating rather than grouped KV heads",
            WrongAttention::AlternatingKvHeads,
        ),
    ] {
        let wrong = wrong_attention(
            &trace.mrope_query,
            &trace.kv_cache.keys,
            &trace.kv_cache.values,
            mode,
        );
        assert_rejected(
            label,
            &expected_context,
            &wrong,
            TOKENS,
            QUERY_WIDTH,
            &CONTEXT_POLICY,
        );
    }
}
