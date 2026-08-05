use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use pvlc_cpu_ref::{
    DecoderLayerConfig, DecoderLayerDecodeTrace, DecoderLayerParameters, DecoderLayerPrefillTrace,
    DecoderPrefillKvCache, add_vectors_f32, apply_multimodal_rope_f32, decoder_layer_decode_f32,
    decoder_layer_prefill_f32, linear_f32, pinned_decoder_layer_decode_f32, rms_norm_f32, silu,
    write_pinned_decoder_prefill_kv_f32,
};
use pvlc_safetensors::SafetensorsCatalog;
use pvlc_testkit::{ComparisonAxes, ComparisonPolicy, ComparisonReport, compare_f32};

const FIXTURE_BYTES: usize = 6_438_376;
const FIXTURE_BLAKE3: &str = "386735089b35b2a1fc50ad578678689eed98b3e62f2a0c18d5e2890d0e6a8ebf";
const LAYER0_FIXTURE_BYTES: usize = 44_637_696;
const LAYER0_FIXTURE_BLAKE3: &str =
    "30eed2f7a4d9336f8b3429d7294065c45214c4425f7559e8e2059ebd54c89522";
const PREFIX_TOKENS: usize = 332;
const FULL_CACHE_TOKENS: usize = 333;
const ONE_TOKEN: usize = 1;
const HIDDEN_SIZE: usize = 1_024;
const INTERMEDIATE_SIZE: usize = 3_072;
const QUERY_HEADS: usize = 16;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIM: usize = 128;
const QUERY_WIDTH: usize = QUERY_HEADS * HEAD_DIM;
const KEY_VALUE_WIDTH: usize = KEY_VALUE_HEADS * HEAD_DIM;
const RMS_NORM_EPSILON: f32 = 1.0e-5;
const MROPE_SECTIONS: [usize; 3] = [16, 24, 24];

const DECODE_INPUT: &str = "decoder.decode.00.layer.00.input";
const DECODE_NORM1: &str = "decoder.decode.00.layer.00.norm1";
const DECODE_QUERY: &str = "decoder.decode.00.layer.00.q";
const DECODE_KEY: &str = "decoder.decode.00.layer.00.k";
const DECODE_VALUE: &str = "decoder.decode.00.layer.00.v";
const DECODE_MROPE_QUERY: &str = "decoder.decode.00.layer.00.mrope.q.token_major";
const DECODE_MROPE_KEY: &str = "decoder.decode.00.layer.00.mrope.k.token_major";
const DECODE_ATTENTION_CONTEXT: &str = "decoder.decode.00.layer.00.attention.context.token_major";
const DECODE_ATTENTION_OUTPUT: &str = "decoder.decode.00.layer.00.attention.output";
const DECODE_ATTENTION_RESIDUAL: &str = "decoder.decode.00.layer.00.attention.residual";
const DECODE_NORM2: &str = "decoder.decode.00.layer.00.norm2";
const DECODE_MLP_GATE: &str = "decoder.decode.00.layer.00.mlp.gate";
const DECODE_MLP_UP: &str = "decoder.decode.00.layer.00.mlp.up";
const DECODE_MLP_ACTIVATION: &str = "decoder.decode.00.layer.00.mlp.activation";
const DECODE_MLP_DOWN: &str = "decoder.decode.00.layer.00.mlp.down";
const DECODE_OUTPUT: &str = "decoder.decode.00.layer.00.output";
const DECODE_RAW_COS: &str = "decoder.decode.00.rope.cos.axis_major";
const DECODE_RAW_SIN: &str = "decoder.decode.00.rope.sin.axis_major";
const DECODE_STACKED_CACHE_KEY: &str = "decoder.decode.00.kv.key.layer_token_major";
const DECODE_STACKED_CACHE_VALUE: &str = "decoder.decode.00.kv.value.layer_token_major";

const LAYER0_CACHE_KEY: &str = "decoder.layer.00.kv.key.token_major";
const LAYER0_CACHE_VALUE: &str = "decoder.layer.00.kv.value.token_major";
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
    dtype: &'static str,
    shape: &'static [u64],
    raw_blake3: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
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

#[derive(Debug)]
struct OfficialDecodeLayer0Stages {
    input: Vec<f32>,
    norm1: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    mrope_query: Vec<f32>,
    mrope_key: Vec<f32>,
    attention_context: Vec<f32>,
    attention_output: Vec<f32>,
    attention_residual: Vec<f32>,
    norm2: Vec<f32>,
    mlp_gate: Vec<f32>,
    mlp_up: Vec<f32>,
    mlp_activation: Vec<f32>,
    mlp_down: Vec<f32>,
    output: Vec<f32>,
    raw_cos: Vec<f32>,
    raw_sin: Vec<f32>,
    prefix_cache: DecoderPrefillKvCache,
    appended_key: Vec<f32>,
    appended_value: Vec<f32>,
}

#[derive(Debug)]
struct IndependentDecodeLayer0Trace {
    norm1: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    mrope_query: Vec<f32>,
    mrope_key: Vec<f32>,
    kv_cache: DecoderPrefillKvCache,
    attention_context: Vec<f32>,
    attention_output: Vec<f32>,
    attention_residual: Vec<f32>,
    norm2: Vec<f32>,
    mlp_gate: Vec<f32>,
    mlp_up: Vec<f32>,
    mlp_activation: Vec<f32>,
    mlp_down: Vec<f32>,
    output: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
enum DirectGqaVariant {
    Full333,
    AppendedOnly,
    WrongGrouping,
    MissingScale,
    DroppedPrefix,
}

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

struct DecodeLayer0Observed<'a> {
    norm1: &'a [f32],
    query: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
    mrope_query: &'a [f32],
    mrope_key: &'a [f32],
    kv_cache: &'a DecoderPrefillKvCache,
    attention_context: &'a [f32],
    attention_output: &'a [f32],
    attention_residual: &'a [f32],
    norm2: &'a [f32],
    mlp_gate: &'a [f32],
    mlp_up: &'a [f32],
    mlp_activation: &'a [f32],
    mlp_down: &'a [f32],
    output: &'a [f32],
}

struct DecodeLayer0PolicyReports {
    append_key: ComparisonReport,
    append_value: ComparisonReport,
    context: ComparisonReport,
}

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
        min_cosine_similarity: 0.99999,
        max_per_token_relative_l2: None,
        max_per_channel_relative_l2: None,
    }
}

// Positive one-token decode-stage metrics measured on 2026-07-20 from the
// accepted independent probe:
// norm1      max=.007787942886353 mean=.0006714816703379 p99=.003711044788361 rel=.001918829543125 cos=.9999981792879
// query      max=.03074264526367  mean=.001373519147705  p99=.008177042007446 rel=.001832034979773 cos=.9999983862194
// key        max=.01273250579834  mean=.001900444669445  p99=.008629560470581 rel=.001605009566503 cos=.9999987231487
// value      max=.0008546411991119 mean=.0001477583641645 p99=.0005499422550201 rel=.002308513699486 cos=.9999974001850
// mrope_query max=.04473304748535 mean=.001846666142796 p99=.01077485084534 rel=.002521090501642 cos=.9999968351028
// mrope_key  max=.01958942413330  mean=.002721447799047  p99=.01202654838562 rel=.002346413718955 cos=.9999972535801
// context    max=.0006753802299500 mean=.00002504859325825 p99=.0001463145017624 rel=.001933825352616 cos=.9999981368042
// attn_out   max=.00008767843246460 mean=.00001516609364138 p99=.00006201304495335 rel=.001194022104526 cos=.9999993037920
// attn_res   max=.0002097487449646 mean=.00002408233491735 p99=.00009880959987640 rel=.001551288156831 cos=.9999987995641
// norm2      max=.02786397933960  mean=.0009668038607060 p99=.004683732986450 rel=.003286659820584 cos=.9999948898378
// gate       max=.008054256439209 mean=.0007532074232965 p99=.003108322620392 rel=.002862555632646 cos=.9999960497910
// up         max=.007636070251465 mean=.0006329178462844 p99=.002517342567444 rel=.002887672225177 cos=.9999958748154
// activation max=.003874391317368 mean=.0001411518617408 p99=.001065343618393 rel=.003934769067935 cos=.9999927330615
// down       max=.001749455928802 mean=.0001926762430742 p99=.001070827245712 rel=.003225036313649 cos=.9999951300935
// output     max=.002738833427429 mean=.0002281281110754 p99=.001305162906647 rel=.004000096238232 cos=.9999923736383
//
// Each limit below is >=1.25x the measured metric, then rounded upward to a
// readable literal. Minimum extra headroom after rounding is 0.011350832% on
// the `up.relative_l2` bound; all other bounds have at least that much slack.
const NORM1_POLICY: ComparisonPolicy = policy(0.0098, 0.00084, 0.00464, 0.00240);
const QUERY_POLICY: ComparisonPolicy = policy(0.0385, 0.00172, 0.0103, 0.00230);
const KEY_POLICY: ComparisonPolicy = policy(0.0160, 0.00238, 0.0108, 0.00201);
const VALUE_POLICY: ComparisonPolicy = policy(0.00107, 0.000185, 0.000688, 0.00289);
const MROPE_QUERY_POLICY: ComparisonPolicy = policy(0.0560, 0.00231, 0.0135, 0.00316);
const MROPE_KEY_POLICY: ComparisonPolicy = policy(0.0245, 0.00341, 0.0151, 0.00294);
const CONTEXT_POLICY: ComparisonPolicy = policy(0.000845, 0.0000314, 0.000183, 0.00242);
const ATTENTION_OUTPUT_POLICY: ComparisonPolicy = policy(0.000110, 0.0000190, 0.0000776, 0.00150);
const ATTENTION_RESIDUAL_POLICY: ComparisonPolicy = policy(0.000263, 0.0000302, 0.000124, 0.00194);
const NORM2_POLICY: ComparisonPolicy = policy(0.0349, 0.00121, 0.00586, 0.00411);
const GATE_POLICY: ComparisonPolicy = policy(0.0101, 0.000942, 0.00389, 0.00358);
const UP_POLICY: ComparisonPolicy = policy(0.00955, 0.000792, 0.00315, 0.00361);
const ACTIVATION_POLICY: ComparisonPolicy = policy(0.00485, 0.000177, 0.00134, 0.00492);
const DOWN_POLICY: ComparisonPolicy = policy(0.00219, 0.000241, 0.00134, 0.00404);
const OUTPUT_POLICY: ComparisonPolicy = policy(0.00343, 0.000286, 0.00164, 0.00501);

const TRACE_POLICIES: TracePolicies = TracePolicies {
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

// Provenance: these published artifacts are pinned to PaddleOCR-VL-1.6
// revision 66317acc4c9fc17bd154591ce650735cd2855f3e, captured by
// TransformersOracle pinned remote code and published through
// output/candidates/decoder-decode-source.golden.lock
// (BLAKE3 blake3:56062fbb43ee1d556cef9eee27928f6222d397bd955353b015a4f0a5f8c3edda).
// This contract authenticates the published bytes only and makes no git claim.
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

#[rustfmt::skip]
const LAYER0_USED_WEIGHTS: [TensorSpec; 9] = [
    TensorSpec { name: "model.layers.0.input_layernorm.weight", dtype: "BF16", shape: &[1024], raw_blake3: "06aaa3c896466a889d166becd2411bbb40ba40501a94f58673c883606005f792" },
    TensorSpec { name: "model.layers.0.mlp.down_proj.weight", dtype: "BF16", shape: &[1024, 3072], raw_blake3: "b862080808c1ba24e2b7d4aea2eb63ed03eb9edccfc0fd5b63a4fc0e61b2835d" },
    TensorSpec { name: "model.layers.0.mlp.gate_proj.weight", dtype: "BF16", shape: &[3072, 1024], raw_blake3: "4a1c6564dfff67e068261710687f78a093100b21d17628cbe20effebb2b0129e" },
    TensorSpec { name: "model.layers.0.mlp.up_proj.weight", dtype: "BF16", shape: &[3072, 1024], raw_blake3: "efe8dfdb0a9ed2cde8ac8f68617cbb46e4dd8732e36a8e171e6162b242666af2" },
    TensorSpec { name: "model.layers.0.post_attention_layernorm.weight", dtype: "BF16", shape: &[1024], raw_blake3: "bbe35de602323b40efd9c4568807d1c16c58efa70acb2d0033c42851ade095aa" },
    TensorSpec { name: "model.layers.0.self_attn.k_proj.weight", dtype: "BF16", shape: &[256, 1024], raw_blake3: "60220e15ac1a117d74cf2897da638b3906cb3864d3e4cc61d21af1aaa44bb342" },
    TensorSpec { name: "model.layers.0.self_attn.o_proj.weight", dtype: "BF16", shape: &[1024, 2048], raw_blake3: "c8fc88619213491a9cfde814c7b05f7bfe176e493228c7dd3bff4899db7645fe" },
    TensorSpec { name: "model.layers.0.self_attn.q_proj.weight", dtype: "BF16", shape: &[2048, 1024], raw_blake3: "fc56998413638a4540531881ef7d88be409744afed3568715f4001bd4c2bda39" },
    TensorSpec { name: "model.layers.0.self_attn.v_proj.weight", dtype: "BF16", shape: &[256, 1024], raw_blake3: "3318fdbf2aa80cd1de67170d674f7a8444a6ef04e6154f9114aa224826d186bd" },
];

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/decoder-decode-official-v1.safetensors")
}

fn layer0_fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
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

fn assert_full_tensor_inventory(catalog: &SafetensorsCatalog, expected: &[TensorSpec]) {
    let observed = catalog
        .tensors()
        .iter()
        .map(|tensor| {
            (
                tensor.name.as_str(),
                tensor.dtype.safetensors_name(),
                tensor.shape.as_slice(),
            )
        })
        .collect::<Vec<_>>();
    let expected_layout = expected
        .iter()
        .map(|tensor| (tensor.name, tensor.dtype, tensor.shape))
        .collect::<Vec<_>>();
    assert_eq!(observed, expected_layout);

    assert_selected_tensors(catalog, expected);
}

fn assert_selected_tensors(catalog: &SafetensorsCatalog, expected: &[TensorSpec]) {
    for tensor in expected {
        let header = catalog.tensor(tensor.name).unwrap();
        assert_eq!(header.dtype.safetensors_name(), tensor.dtype);
        assert_eq!(header.shape.as_slice(), tensor.shape);
        assert_eq!(
            hash_bytes(&tensor_bytes(catalog, tensor.name)),
            tensor.raw_blake3,
            "raw payload mismatch for {}",
            tensor.name
        );
    }
}

#[test]
fn authenticates_decoder_decode_official_fixture_inventory_and_layer0_weight_dependencies() {
    let fixture = assert_file_identity(&fixture_path(), FIXTURE_BYTES, FIXTURE_BLAKE3);
    let metadata = fixture
        .metadata()
        .iter()
        .map(|(key, value)| (key.as_str(), value.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(metadata, METADATA);
    assert_eq!(fixture.metadata().len(), 38);
    assert_eq!(fixture.tensors().len(), 44);
    assert_full_tensor_inventory(&fixture, &TENSORS);

    let layer0 = assert_file_identity(
        &layer0_fixture_path(),
        LAYER0_FIXTURE_BYTES,
        LAYER0_FIXTURE_BLAKE3,
    );
    assert_selected_tensors(&layer0, &LAYER0_USED_WEIGHTS);
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
}

fn pinned_single_token_decode_config() -> DecoderLayerConfig {
    decode_layer0_config(ONE_TOKEN)
}

fn decode_layer0_config(tokens: usize) -> DecoderLayerConfig {
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

fn load(catalog: &SafetensorsCatalog, name: &str) -> Vec<f32> {
    catalog.load_tensor_f32(name).unwrap()
}

fn slice_layer_from_stacked_cache(
    values: &[f32],
    layer: usize,
    tokens: usize,
    width: usize,
) -> Vec<f32> {
    let layer_span = tokens * width;
    values[layer * layer_span..(layer + 1) * layer_span].to_vec()
}

fn cache_index(token: usize, head: usize, dim: usize, heads: usize, head_dim: usize) -> usize {
    (token * heads + head) * head_dim + dim
}

fn load_official_decode_layer0(
    decode: &SafetensorsCatalog,
    layer0: &SafetensorsCatalog,
) -> OfficialDecodeLayer0Stages {
    let prefix_key = load(layer0, LAYER0_CACHE_KEY);
    let prefix_value = load(layer0, LAYER0_CACHE_VALUE);
    assert_eq!(prefix_key.len(), PREFIX_TOKENS * KEY_VALUE_WIDTH);
    assert_eq!(prefix_value.len(), PREFIX_TOKENS * KEY_VALUE_WIDTH);

    let stacked_keys = load(decode, DECODE_STACKED_CACHE_KEY);
    let stacked_values = load(decode, DECODE_STACKED_CACHE_VALUE);
    assert_eq!(stacked_keys.len(), 18 * FULL_CACHE_TOKENS * KEY_VALUE_WIDTH);
    assert_eq!(
        stacked_values.len(),
        18 * FULL_CACHE_TOKENS * KEY_VALUE_WIDTH
    );
    let layer0_keys =
        slice_layer_from_stacked_cache(&stacked_keys, 0, FULL_CACHE_TOKENS, KEY_VALUE_WIDTH);
    let layer0_values =
        slice_layer_from_stacked_cache(&stacked_values, 0, FULL_CACHE_TOKENS, KEY_VALUE_WIDTH);
    assert_eq!(
        &layer0_keys[..PREFIX_TOKENS * KEY_VALUE_WIDTH],
        prefix_key.as_slice()
    );
    assert_eq!(
        &layer0_values[..PREFIX_TOKENS * KEY_VALUE_WIDTH],
        prefix_value.as_slice()
    );

    let appended_key = layer0_keys[PREFIX_TOKENS * KEY_VALUE_WIDTH..].to_vec();
    let appended_value = layer0_values[PREFIX_TOKENS * KEY_VALUE_WIDTH..].to_vec();
    assert_eq!(appended_key.len(), KEY_VALUE_WIDTH);
    assert_eq!(appended_value.len(), KEY_VALUE_WIDTH);

    OfficialDecodeLayer0Stages {
        input: load(decode, DECODE_INPUT),
        norm1: load(decode, DECODE_NORM1),
        query: load(decode, DECODE_QUERY),
        key: load(decode, DECODE_KEY),
        value: load(decode, DECODE_VALUE),
        mrope_query: load(decode, DECODE_MROPE_QUERY),
        mrope_key: load(decode, DECODE_MROPE_KEY),
        attention_context: load(decode, DECODE_ATTENTION_CONTEXT),
        attention_output: load(decode, DECODE_ATTENTION_OUTPUT),
        attention_residual: load(decode, DECODE_ATTENTION_RESIDUAL),
        norm2: load(decode, DECODE_NORM2),
        mlp_gate: load(decode, DECODE_MLP_GATE),
        mlp_up: load(decode, DECODE_MLP_UP),
        mlp_activation: load(decode, DECODE_MLP_ACTIVATION),
        mlp_down: load(decode, DECODE_MLP_DOWN),
        output: load(decode, DECODE_OUTPUT),
        raw_cos: load(decode, DECODE_RAW_COS),
        raw_sin: load(decode, DECODE_RAW_SIN),
        prefix_cache: write_pinned_decoder_prefill_kv_f32(
            &prefix_key,
            &prefix_value,
            PREFIX_TOKENS,
        )
        .unwrap(),
        appended_key,
        appended_value,
    }
}

fn append_to_cache(
    prefix_cache: &DecoderPrefillKvCache,
    appended_key: &[f32],
    appended_value: &[f32],
) -> DecoderPrefillKvCache {
    let mut keys = prefix_cache.keys.clone();
    keys.extend_from_slice(appended_key);
    let mut values = prefix_cache.values.clone();
    values.extend_from_slice(appended_value);
    write_pinned_decoder_prefill_kv_f32(&keys, &values, FULL_CACHE_TOKENS).unwrap()
}

fn repeat_kv_heads_contiguously(input: &[f32], tokens: usize) -> Vec<f32> {
    let group = QUERY_HEADS / KEY_VALUE_HEADS;
    let mut repeated = Vec::with_capacity(tokens * QUERY_WIDTH);
    for token in 0..tokens {
        for kv_head in 0..KEY_VALUE_HEADS {
            let start = cache_index(token, kv_head, 0, KEY_VALUE_HEADS, HEAD_DIM);
            for _ in 0..group {
                repeated.extend_from_slice(&input[start..start + HEAD_DIM]);
            }
        }
    }
    repeated
}

fn repeat_kv_heads_with_rotated_groups(input: &[f32], tokens: usize) -> Vec<f32> {
    let group = QUERY_HEADS / KEY_VALUE_HEADS;
    let mut repeated = Vec::with_capacity(tokens * QUERY_WIDTH);
    for token in 0..tokens {
        for kv_head in 0..KEY_VALUE_HEADS {
            let wrong_head = (kv_head + 1) % KEY_VALUE_HEADS;
            let start = cache_index(token, wrong_head, 0, KEY_VALUE_HEADS, HEAD_DIM);
            for _ in 0..group {
                repeated.extend_from_slice(&input[start..start + HEAD_DIM]);
            }
        }
    }
    repeated
}

fn direct_decode_from_repeated_kv(
    query: &[f32],
    repeated_keys: &[f32],
    repeated_values: &[f32],
    tokens: usize,
    scale: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; QUERY_WIDTH];
    for head in 0..QUERY_HEADS {
        let mut logits = Vec::with_capacity(tokens);
        for token in 0..tokens {
            let mut dot = 0.0_f32;
            for dim in 0..HEAD_DIM {
                dot += query[head * HEAD_DIM + dim]
                    * repeated_keys[cache_index(token, head, dim, QUERY_HEADS, HEAD_DIM)];
            }
            logits.push(dot * scale);
        }
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities = logits
            .into_iter()
            .map(|logit| (logit - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = probabilities.iter().sum::<f32>();
        for probability in &mut probabilities {
            *probability /= denominator;
        }
        for dim in 0..HEAD_DIM {
            let mut weighted = 0.0_f32;
            for (token, probability) in probabilities.iter().copied().enumerate() {
                weighted += probability
                    * repeated_values[cache_index(token, head, dim, QUERY_HEADS, HEAD_DIM)];
            }
            output[head * HEAD_DIM + dim] = weighted;
        }
    }
    output
}

fn direct_gqa_variant(
    query: &[f32],
    prefix_cache: &DecoderPrefillKvCache,
    appended_key: &[f32],
    appended_value: &[f32],
    variant: DirectGqaVariant,
) -> Vec<f32> {
    let scale = match variant {
        DirectGqaVariant::MissingScale => 1.0,
        _ => (HEAD_DIM as f32).sqrt().recip(),
    };
    let (keys, values, tokens) = match variant {
        DirectGqaVariant::AppendedOnly => {
            (appended_key.to_vec(), appended_value.to_vec(), ONE_TOKEN)
        }
        DirectGqaVariant::DroppedPrefix => {
            let keys = prefix_cache.keys[KEY_VALUE_WIDTH..]
                .iter()
                .copied()
                .chain(appended_key.iter().copied())
                .collect::<Vec<_>>();
            let values = prefix_cache.values[KEY_VALUE_WIDTH..]
                .iter()
                .copied()
                .chain(appended_value.iter().copied())
                .collect::<Vec<_>>();
            (keys, values, PREFIX_TOKENS)
        }
        _ => {
            let cache = append_to_cache(prefix_cache, appended_key, appended_value);
            (cache.keys, cache.values, FULL_CACHE_TOKENS)
        }
    };
    let (repeated_keys, repeated_values) = match variant {
        DirectGqaVariant::WrongGrouping => (
            repeat_kv_heads_with_rotated_groups(&keys, tokens),
            repeat_kv_heads_with_rotated_groups(&values, tokens),
        ),
        _ => (
            repeat_kv_heads_contiguously(&keys, tokens),
            repeat_kv_heads_contiguously(&values, tokens),
        ),
    };
    direct_decode_from_repeated_kv(query, &repeated_keys, &repeated_values, tokens, scale)
}

fn independent_decode_layer0_trace(
    input: &[f32],
    raw_cos: &[f32],
    raw_sin: &[f32],
    prefix_cache: &DecoderPrefillKvCache,
    parameters: &OwnedParameters,
) -> IndependentDecodeLayer0Trace {
    let zero_query = vec![0.0; QUERY_WIDTH];
    let zero_key_value = vec![0.0; KEY_VALUE_WIDTH];
    let zero_hidden = vec![0.0; HIDDEN_SIZE];
    let zero_intermediate = vec![0.0; INTERMEDIATE_SIZE];

    let norm1 = rms_norm_f32(
        input,
        ONE_TOKEN,
        HIDDEN_SIZE,
        &parameters.input_norm_weight,
        RMS_NORM_EPSILON,
    )
    .unwrap();
    let query = linear_f32(
        &norm1,
        ONE_TOKEN,
        HIDDEN_SIZE,
        &parameters.query_weight,
        &zero_query,
        QUERY_WIDTH,
    )
    .unwrap();
    let key = linear_f32(
        &norm1,
        ONE_TOKEN,
        HIDDEN_SIZE,
        &parameters.key_weight,
        &zero_key_value,
        KEY_VALUE_WIDTH,
    )
    .unwrap();
    let value = linear_f32(
        &norm1,
        ONE_TOKEN,
        HIDDEN_SIZE,
        &parameters.value_weight,
        &zero_key_value,
        KEY_VALUE_WIDTH,
    )
    .unwrap();
    let (mrope_query, mrope_key) = apply_multimodal_rope_f32(
        &query,
        &key,
        ONE_TOKEN,
        QUERY_HEADS,
        KEY_VALUE_HEADS,
        HEAD_DIM,
        raw_cos,
        raw_sin,
        MROPE_SECTIONS,
    )
    .unwrap();
    let kv_cache = append_to_cache(prefix_cache, &mrope_key, &value);
    let attention_context = direct_gqa_variant(
        &mrope_query,
        prefix_cache,
        &mrope_key,
        &value,
        DirectGqaVariant::Full333,
    );
    let attention_output = linear_f32(
        &attention_context,
        ONE_TOKEN,
        QUERY_WIDTH,
        &parameters.attention_output_weight,
        &zero_hidden,
        HIDDEN_SIZE,
    )
    .unwrap();
    let attention_residual = add_vectors_f32(input, &attention_output).unwrap();
    let norm2 = rms_norm_f32(
        &attention_residual,
        ONE_TOKEN,
        HIDDEN_SIZE,
        &parameters.post_attention_norm_weight,
        RMS_NORM_EPSILON,
    )
    .unwrap();
    let mlp_gate = linear_f32(
        &norm2,
        ONE_TOKEN,
        HIDDEN_SIZE,
        &parameters.gate_weight,
        &zero_intermediate,
        INTERMEDIATE_SIZE,
    )
    .unwrap();
    let mlp_up = linear_f32(
        &norm2,
        ONE_TOKEN,
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
        ONE_TOKEN,
        INTERMEDIATE_SIZE,
        &parameters.down_weight,
        &zero_hidden,
        HIDDEN_SIZE,
    )
    .unwrap();
    let output = add_vectors_f32(&attention_residual, &mlp_down).unwrap();

    IndependentDecodeLayer0Trace {
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

fn observed_decode_layer0_trace(trace: &IndependentDecodeLayer0Trace) -> DecodeLayer0Observed<'_> {
    DecodeLayer0Observed {
        norm1: &trace.norm1,
        query: &trace.query,
        key: &trace.key,
        value: &trace.value,
        mrope_query: &trace.mrope_query,
        mrope_key: &trace.mrope_key,
        kv_cache: &trace.kv_cache,
        attention_context: &trace.attention_context,
        attention_output: &trace.attention_output,
        attention_residual: &trace.attention_residual,
        norm2: &trace.norm2,
        mlp_gate: &trace.mlp_gate,
        mlp_up: &trace.mlp_up,
        mlp_activation: &trace.mlp_activation,
        mlp_down: &trace.mlp_down,
        output: &trace.output,
    }
}

fn observed_decode_layer0_decode_trace(
    trace: &DecoderLayerDecodeTrace,
) -> DecodeLayer0Observed<'_> {
    DecodeLayer0Observed {
        norm1: &trace.norm1,
        query: &trace.query,
        key: &trace.key,
        value: &trace.value,
        mrope_query: &trace.mrope_query,
        mrope_key: &trace.mrope_key,
        kv_cache: &trace.kv_cache,
        attention_context: &trace.attention_context,
        attention_output: &trace.attention_output,
        attention_residual: &trace.attention_residual,
        norm2: &trace.norm2,
        mlp_gate: &trace.mlp_gate,
        mlp_up: &trace.mlp_up,
        mlp_activation: &trace.mlp_activation,
        mlp_down: &trace.mlp_down,
        output: &trace.output,
    }
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

fn slice_last_row(values: &[f32], tokens: usize, width: usize) -> Vec<f32> {
    values[(tokens - 1) * width..tokens * width].to_vec()
}

fn slice_last_raw_axis_major_row(values: &[f32], tokens: usize, head_dim: usize) -> Vec<f32> {
    let mut row = Vec::with_capacity(3 * head_dim);
    for axis in 0..3 {
        let start = (axis * tokens + (tokens - 1)) * head_dim;
        row.extend_from_slice(&values[start..start + head_dim]);
    }
    row
}

fn append_axis_major_row(prefix: &[f32], row: &[f32]) -> Vec<f32> {
    assert_eq!(prefix.len(), 3 * PREFIX_TOKENS * HEAD_DIM);
    assert_eq!(row.len(), 3 * HEAD_DIM);
    let mut full = Vec::with_capacity(3 * FULL_CACHE_TOKENS * HEAD_DIM);
    for axis in 0..3 {
        let prefix_start = axis * PREFIX_TOKENS * HEAD_DIM;
        let row_start = axis * HEAD_DIM;
        full.extend_from_slice(&prefix[prefix_start..prefix_start + PREFIX_TOKENS * HEAD_DIM]);
        full.extend_from_slice(&row[row_start..row_start + HEAD_DIM]);
    }
    full
}

fn assert_decode_trace_matches_last_prefill_row_exact(
    actual: &DecoderLayerDecodeTrace,
    expected: &DecoderLayerPrefillTrace,
    full_tokens: usize,
) {
    let fields: [(&str, &[f32], Vec<f32>); 15] = [
        (
            "norm1",
            &actual.norm1,
            slice_last_row(&expected.norm1, full_tokens, HIDDEN_SIZE),
        ),
        (
            "query",
            &actual.query,
            slice_last_row(&expected.query, full_tokens, QUERY_WIDTH),
        ),
        (
            "key",
            &actual.key,
            slice_last_row(&expected.key, full_tokens, KEY_VALUE_WIDTH),
        ),
        (
            "value",
            &actual.value,
            slice_last_row(&expected.value, full_tokens, KEY_VALUE_WIDTH),
        ),
        (
            "mrope_query",
            &actual.mrope_query,
            slice_last_row(&expected.mrope_query, full_tokens, QUERY_WIDTH),
        ),
        (
            "mrope_key",
            &actual.mrope_key,
            slice_last_row(&expected.mrope_key, full_tokens, KEY_VALUE_WIDTH),
        ),
        (
            "attention_context",
            &actual.attention_context,
            slice_last_row(&expected.attention_context, full_tokens, QUERY_WIDTH),
        ),
        (
            "attention_output",
            &actual.attention_output,
            slice_last_row(&expected.attention_output, full_tokens, HIDDEN_SIZE),
        ),
        (
            "attention_residual",
            &actual.attention_residual,
            slice_last_row(&expected.attention_residual, full_tokens, HIDDEN_SIZE),
        ),
        (
            "norm2",
            &actual.norm2,
            slice_last_row(&expected.norm2, full_tokens, HIDDEN_SIZE),
        ),
        (
            "mlp_gate",
            &actual.mlp_gate,
            slice_last_row(&expected.mlp_gate, full_tokens, INTERMEDIATE_SIZE),
        ),
        (
            "mlp_up",
            &actual.mlp_up,
            slice_last_row(&expected.mlp_up, full_tokens, INTERMEDIATE_SIZE),
        ),
        (
            "mlp_activation",
            &actual.mlp_activation,
            slice_last_row(&expected.mlp_activation, full_tokens, INTERMEDIATE_SIZE),
        ),
        (
            "mlp_down",
            &actual.mlp_down,
            slice_last_row(&expected.mlp_down, full_tokens, HIDDEN_SIZE),
        ),
        (
            "output",
            &actual.output,
            slice_last_row(&expected.output, full_tokens, HIDDEN_SIZE),
        ),
    ];
    for (label, actual, expected) in fields {
        assert_f32_bits(label, actual, &expected);
    }
    assert_f32_bits(
        "kv_cache.keys",
        &actual.kv_cache.keys,
        &expected.kv_cache.keys,
    );
    assert_f32_bits(
        "kv_cache.values",
        &actual.kv_cache.values,
        &expected.kv_cache.values,
    );
    assert_eq!(actual.kv_cache.tokens, expected.kv_cache.tokens);
    assert_eq!(
        actual.kv_cache.key_value_heads,
        expected.kv_cache.key_value_heads
    );
    assert_eq!(actual.kv_cache.head_dim, expected.kv_cache.head_dim);
}

fn assert_decode_trace_matches_independent_exact(
    actual: &DecoderLayerDecodeTrace,
    expected: &IndependentDecodeLayer0Trace,
) {
    assert_f32_bits("norm1", &actual.norm1, &expected.norm1);
    assert_f32_bits("query", &actual.query, &expected.query);
    assert_f32_bits("key", &actual.key, &expected.key);
    assert_f32_bits("value", &actual.value, &expected.value);
    assert_f32_bits("mrope_query", &actual.mrope_query, &expected.mrope_query);
    assert_f32_bits("mrope_key", &actual.mrope_key, &expected.mrope_key);
    assert_f32_bits(
        "attention_context",
        &actual.attention_context,
        &expected.attention_context,
    );
    assert_f32_bits(
        "attention_output",
        &actual.attention_output,
        &expected.attention_output,
    );
    assert_f32_bits(
        "attention_residual",
        &actual.attention_residual,
        &expected.attention_residual,
    );
    assert_f32_bits("norm2", &actual.norm2, &expected.norm2);
    assert_f32_bits("mlp_gate", &actual.mlp_gate, &expected.mlp_gate);
    assert_f32_bits("mlp_up", &actual.mlp_up, &expected.mlp_up);
    assert_f32_bits(
        "mlp_activation",
        &actual.mlp_activation,
        &expected.mlp_activation,
    );
    assert_f32_bits("mlp_down", &actual.mlp_down, &expected.mlp_down);
    assert_f32_bits("output", &actual.output, &expected.output);
    assert_f32_bits(
        "kv_cache.keys",
        &actual.kv_cache.keys,
        &expected.kv_cache.keys,
    );
    assert_f32_bits(
        "kv_cache.values",
        &actual.kv_cache.values,
        &expected.kv_cache.values,
    );
    assert_eq!(actual.kv_cache.tokens, expected.kv_cache.tokens);
    assert_eq!(
        actual.kv_cache.key_value_heads,
        expected.kv_cache.key_value_heads
    );
    assert_eq!(actual.kv_cache.head_dim, expected.kv_cache.head_dim);
}

fn compare_stage(label: &str, expected: &[f32], actual: &[f32], width: usize) -> ComparisonReport {
    assert_eq!(expected.len(), width, "{label} expected length");
    assert_eq!(actual.len(), width, "{label} actual length");
    assert!(
        expected.iter().chain(actual).all(|value| value.is_finite()),
        "{label} finiteness"
    );
    compare_f32(
        expected,
        actual,
        &[ONE_TOKEN, width],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap()
}

fn print_stage_report(label: &str, report: &ComparisonReport) {
    println!(
        "{label}: max_abs={:.12e} mean_abs={:.12e} p99_abs={:.12e} relative_l2={:.12e} cosine={:.12e}",
        report.max_abs,
        report.mean_abs,
        report.p99_abs,
        report.relative_l2,
        report.cosine_similarity
    );
}

fn assert_stage_policy(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    width: usize,
    comparison_policy: &ComparisonPolicy,
) -> ComparisonReport {
    let report = compare_stage(label, expected, actual, width);
    print_stage_report(label, &report);
    let verdict = report.assess(comparison_policy).unwrap();
    assert!(
        verdict.passed(),
        "{label} violated frozen policy\npolicy={comparison_policy:#?}\nreport={report:#?}\nverdict={verdict:#?}"
    );
    report
}

fn assert_stage_policy_rejected(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    width: usize,
    comparison_policy: &ComparisonPolicy,
) -> ComparisonReport {
    let report = compare_stage(label, expected, actual, width);
    print_stage_report(label, &report);
    let verdict = report.assess(comparison_policy).unwrap();
    assert!(
        !verdict.passed(),
        "{label} unexpectedly passed frozen policy\npolicy={comparison_policy:#?}\nreport={report:#?}"
    );
    report
}

fn assert_materially_worse(
    label: &str,
    baseline: &ComparisonReport,
    negative: &ComparisonReport,
    expected: &[f32],
    actual: &[f32],
) {
    assert_ne!(
        expected, actual,
        "{label} unexpectedly matched the official context exactly"
    );
    assert!(
        negative.max_abs > baseline.max_abs
            && negative.mean_abs > baseline.mean_abs
            && negative.relative_l2 > baseline.relative_l2,
        "{label} was not materially worse than the positive trace\nbaseline={baseline:#?}\nnegative={negative:#?}"
    );
}

fn assert_cache_prefix_preserved(
    full_cache: &DecoderPrefillKvCache,
    prefix_cache: &DecoderPrefillKvCache,
) {
    assert_eq!(full_cache.tokens, FULL_CACHE_TOKENS);
    assert_eq!(full_cache.key_value_heads, KEY_VALUE_HEADS);
    assert_eq!(full_cache.head_dim, HEAD_DIM);
    assert_eq!(
        &full_cache.keys[..PREFIX_TOKENS * KEY_VALUE_WIDTH],
        prefix_cache.keys.as_slice()
    );
    assert_eq!(
        &full_cache.values[..PREFIX_TOKENS * KEY_VALUE_WIDTH],
        prefix_cache.values.as_slice()
    );
}

fn assert_decode_layer0_observed_matches_official(
    official: &OfficialDecodeLayer0Stages,
    observed: DecodeLayer0Observed<'_>,
    policies: &TracePolicies,
) -> DecodeLayer0PolicyReports {
    assert_cache_prefix_preserved(observed.kv_cache, &official.prefix_cache);
    let appended_key = &observed.kv_cache.keys[PREFIX_TOKENS * KEY_VALUE_WIDTH..];
    let appended_value = &observed.kv_cache.values[PREFIX_TOKENS * KEY_VALUE_WIDTH..];

    let _ = assert_stage_policy(
        DECODE_NORM1,
        &official.norm1,
        observed.norm1,
        HIDDEN_SIZE,
        &policies.norm1,
    );
    let _ = assert_stage_policy(
        DECODE_QUERY,
        &official.query,
        observed.query,
        QUERY_WIDTH,
        &policies.query,
    );
    let _ = assert_stage_policy(
        DECODE_KEY,
        &official.key,
        observed.key,
        KEY_VALUE_WIDTH,
        &policies.key,
    );
    let _ = assert_stage_policy(
        DECODE_VALUE,
        &official.value,
        observed.value,
        KEY_VALUE_WIDTH,
        &policies.value,
    );
    let _ = assert_stage_policy(
        DECODE_MROPE_QUERY,
        &official.mrope_query,
        observed.mrope_query,
        QUERY_WIDTH,
        &policies.mrope_query,
    );
    let _ = assert_stage_policy(
        DECODE_MROPE_KEY,
        &official.mrope_key,
        observed.mrope_key,
        KEY_VALUE_WIDTH,
        &policies.mrope_key,
    );
    let append_key = assert_stage_policy(
        "decoder.decode.00.cache.appended_key",
        &official.appended_key,
        appended_key,
        KEY_VALUE_WIDTH,
        &policies.mrope_key,
    );
    let append_value = assert_stage_policy(
        "decoder.decode.00.cache.appended_value",
        &official.appended_value,
        appended_value,
        KEY_VALUE_WIDTH,
        &policies.value,
    );
    let context = assert_stage_policy(
        DECODE_ATTENTION_CONTEXT,
        &official.attention_context,
        observed.attention_context,
        QUERY_WIDTH,
        &policies.context,
    );
    let _ = assert_stage_policy(
        DECODE_ATTENTION_OUTPUT,
        &official.attention_output,
        observed.attention_output,
        HIDDEN_SIZE,
        &policies.attention_output,
    );
    let _ = assert_stage_policy(
        DECODE_ATTENTION_RESIDUAL,
        &official.attention_residual,
        observed.attention_residual,
        HIDDEN_SIZE,
        &policies.attention_residual,
    );
    let _ = assert_stage_policy(
        DECODE_NORM2,
        &official.norm2,
        observed.norm2,
        HIDDEN_SIZE,
        &policies.norm2,
    );
    let _ = assert_stage_policy(
        DECODE_MLP_GATE,
        &official.mlp_gate,
        observed.mlp_gate,
        INTERMEDIATE_SIZE,
        &policies.gate,
    );
    let _ = assert_stage_policy(
        DECODE_MLP_UP,
        &official.mlp_up,
        observed.mlp_up,
        INTERMEDIATE_SIZE,
        &policies.up,
    );
    let _ = assert_stage_policy(
        DECODE_MLP_ACTIVATION,
        &official.mlp_activation,
        observed.mlp_activation,
        INTERMEDIATE_SIZE,
        &policies.activation,
    );
    let _ = assert_stage_policy(
        DECODE_MLP_DOWN,
        &official.mlp_down,
        observed.mlp_down,
        HIDDEN_SIZE,
        &policies.down,
    );
    let _ = assert_stage_policy(
        DECODE_OUTPUT,
        &official.output,
        observed.output,
        HIDDEN_SIZE,
        &policies.output,
    );

    DecodeLayer0PolicyReports {
        append_key,
        append_value,
        context,
    }
}

#[test]
fn independent_official_trace_reports_frozen_policy_inputs() {
    let started = Instant::now();
    let decode = SafetensorsCatalog::open(fixture_path()).unwrap();
    let layer0 = SafetensorsCatalog::open(layer0_fixture_path()).unwrap();
    let official = load_official_decode_layer0(&decode, &layer0);
    let parameters = OwnedParameters::load(&layer0);

    let preserved_input = official.input.clone();
    let preserved_raw_cos = official.raw_cos.clone();
    let preserved_raw_sin = official.raw_sin.clone();
    let preserved_parameters = parameters.clone();
    let preserved_prefix_keys = official.prefix_cache.keys.clone();
    let preserved_prefix_values = official.prefix_cache.values.clone();

    let trace = independent_decode_layer0_trace(
        &official.input,
        &official.raw_cos,
        &official.raw_sin,
        &official.prefix_cache,
        &parameters,
    );

    assert_eq!(official.input, preserved_input);
    assert_eq!(official.raw_cos, preserved_raw_cos);
    assert_eq!(official.raw_sin, preserved_raw_sin);
    assert_eq!(parameters, preserved_parameters);
    assert_eq!(official.prefix_cache.keys, preserved_prefix_keys);
    assert_eq!(official.prefix_cache.values, preserved_prefix_values);
    assert_cache_prefix_preserved(&trace.kv_cache, &official.prefix_cache);
    assert_eq!(
        &trace.kv_cache.keys[PREFIX_TOKENS * KEY_VALUE_WIDTH..],
        trace.mrope_key.as_slice()
    );
    assert_eq!(
        &trace.kv_cache.values[PREFIX_TOKENS * KEY_VALUE_WIDTH..],
        trace.value.as_slice()
    );

    let policy_reports = assert_decode_layer0_observed_matches_official(
        &official,
        observed_decode_layer0_trace(&trace),
        &TRACE_POLICIES,
    );

    assert_eq!(policy_reports.append_key.non_finite_mismatches, 0);
    assert_eq!(policy_reports.append_value.non_finite_mismatches, 0);

    let appended_only = direct_gqa_variant(
        &trace.mrope_query,
        &official.prefix_cache,
        &trace.mrope_key,
        &trace.value,
        DirectGqaVariant::AppendedOnly,
    );
    let wrong_grouping = direct_gqa_variant(
        &trace.mrope_query,
        &official.prefix_cache,
        &trace.mrope_key,
        &trace.value,
        DirectGqaVariant::WrongGrouping,
    );
    let missing_scale = direct_gqa_variant(
        &trace.mrope_query,
        &official.prefix_cache,
        &trace.mrope_key,
        &trace.value,
        DirectGqaVariant::MissingScale,
    );
    let dropped_prefix = direct_gqa_variant(
        &trace.mrope_query,
        &official.prefix_cache,
        &trace.mrope_key,
        &trace.value,
        DirectGqaVariant::DroppedPrefix,
    );

    // Negative context evidence against CONTEXT_POLICY:
    // appended_only  max=.5918141156435 mean=.05838276821532 p99=.3086369335651 rel=3.762291735720 cos=.1085018068792
    // wrong_grouping max=.3845462943427 mean=.02392361247125 p99=.1026711165905 rel=1.506944873708 cos=-.03976943550238
    // missing_scale  max=.3933498859406 mean=.01457776794431 p99=.09231828898191 rel=1.118349194580 cos=.6181287745667
    // dropped_prefix max=.03930686227977 mean=.002995666120153 p99=.01569795049727 rel=.1948838342156 cos=.9866164262707
    //
    // Quantitative failure margins vs CONTEXT_POLICY:
    // appended_only  x700.37 / x1859.32 / x1686.54 / x1554.67 on max/mean/p99/relL2
    // wrong_grouping x455.08 / x761.90 / x561.04 / x622.70
    // missing_scale  x465.50 / x464.26 / x504.47 / x462.13
    // dropped_prefix x46.52 / x95.40 / x85.78 / x80.53
    let appended_only_report = assert_stage_policy_rejected(
        "negative.appended_only",
        &official.attention_context,
        &appended_only,
        QUERY_WIDTH,
        &CONTEXT_POLICY,
    );
    let wrong_grouping_report = assert_stage_policy_rejected(
        "negative.wrong_grouping",
        &official.attention_context,
        &wrong_grouping,
        QUERY_WIDTH,
        &CONTEXT_POLICY,
    );
    let missing_scale_report = assert_stage_policy_rejected(
        "negative.missing_scale",
        &official.attention_context,
        &missing_scale,
        QUERY_WIDTH,
        &CONTEXT_POLICY,
    );
    let dropped_prefix_report = assert_stage_policy_rejected(
        "negative.dropped_prefix",
        &official.attention_context,
        &dropped_prefix,
        QUERY_WIDTH,
        &CONTEXT_POLICY,
    );

    assert_materially_worse(
        "negative.appended_only",
        &policy_reports.context,
        &appended_only_report,
        &official.attention_context,
        &appended_only,
    );
    assert_materially_worse(
        "negative.wrong_grouping",
        &policy_reports.context,
        &wrong_grouping_report,
        &official.attention_context,
        &wrong_grouping,
    );
    assert_materially_worse(
        "negative.missing_scale",
        &policy_reports.context,
        &missing_scale_report,
        &official.attention_context,
        &missing_scale,
    );
    assert_materially_worse(
        "negative.dropped_prefix",
        &policy_reports.context,
        &dropped_prefix_report,
        &official.attention_context,
        &dropped_prefix,
    );

    println!("runtime_ms={}", started.elapsed().as_secs_f64() * 1_000.0);
}

// Scope: these tests cover only layer0 one-token cached decode behavior.
// Decoder stack composition, logits, and generation remain out of scope here.

#[test]
fn pinned_official_single_token_cached_layer_matches_all_stages_and_preserves_ownership() {
    let decode = SafetensorsCatalog::open(fixture_path()).unwrap();
    let layer0 = SafetensorsCatalog::open(layer0_fixture_path()).unwrap();
    let official = load_official_decode_layer0(&decode, &layer0);
    let parameters = OwnedParameters::load(&layer0);
    let independent = independent_decode_layer0_trace(
        &official.input,
        &official.raw_cos,
        &official.raw_sin,
        &official.prefix_cache,
        &parameters,
    );

    let preserved_input = official.input.clone();
    let preserved_raw_cos = official.raw_cos.clone();
    let preserved_raw_sin = official.raw_sin.clone();
    let preserved_prefix_cache = official.prefix_cache.clone();
    let preserved_parameters = parameters.clone();

    let mut pinned = pinned_decoder_layer_decode_f32(
        &official.input,
        &official.raw_cos,
        &official.raw_sin,
        &official.prefix_cache,
        parameters.borrowed(),
    )
    .unwrap();
    let generic = decoder_layer_decode_f32(
        &official.input,
        pinned_single_token_decode_config(),
        &official.raw_cos,
        &official.raw_sin,
        &official.prefix_cache,
        parameters.borrowed(),
    )
    .unwrap();

    assert_decode_trace_matches_independent_exact(&generic, &independent);
    assert_decode_trace_matches_independent_exact(&pinned, &independent);
    assert_decode_trace_matches_independent_exact(
        &pinned,
        &IndependentDecodeLayer0Trace {
            norm1: generic.norm1.clone(),
            query: generic.query.clone(),
            key: generic.key.clone(),
            value: generic.value.clone(),
            mrope_query: generic.mrope_query.clone(),
            mrope_key: generic.mrope_key.clone(),
            kv_cache: generic.kv_cache.clone(),
            attention_context: generic.attention_context.clone(),
            attention_output: generic.attention_output.clone(),
            attention_residual: generic.attention_residual.clone(),
            norm2: generic.norm2.clone(),
            mlp_gate: generic.mlp_gate.clone(),
            mlp_up: generic.mlp_up.clone(),
            mlp_activation: generic.mlp_activation.clone(),
            mlp_down: generic.mlp_down.clone(),
            output: generic.output.clone(),
        },
    );
    let _ = assert_decode_layer0_observed_matches_official(
        &official,
        observed_decode_layer0_decode_trace(&pinned),
        &TRACE_POLICIES,
    );

    assert_eq!(official.input, preserved_input);
    assert_eq!(official.raw_cos, preserved_raw_cos);
    assert_eq!(official.raw_sin, preserved_raw_sin);
    assert_eq!(official.prefix_cache, preserved_prefix_cache);
    assert_eq!(parameters, preserved_parameters);

    let preserved_key = pinned.mrope_key.clone();
    let preserved_value = pinned.value.clone();
    let append_offset = PREFIX_TOKENS * KEY_VALUE_WIDTH;
    let preserved_append_key = pinned.kv_cache.keys[append_offset];
    let preserved_append_value = pinned.kv_cache.values[append_offset + 1];
    pinned.kv_cache.keys[append_offset] = 123.5;
    pinned.kv_cache.values[append_offset + 1] = -456.25;
    assert_ne!(pinned.kv_cache.keys[append_offset], preserved_append_key);
    assert_ne!(
        pinned.kv_cache.values[append_offset + 1],
        preserved_append_value
    );
    assert_eq!(official.prefix_cache, preserved_prefix_cache);
    assert_f32_bits("detached key source", &pinned.mrope_key, &preserved_key);
    assert_f32_bits("detached value source", &pinned.value, &preserved_value);
}

#[test]
fn pinned_decode_wrapper_rejects_shape_consistent_alternate_topology() {
    const ALT_HIDDEN: usize = 7;
    const ALT_INTERMEDIATE: usize = 9;
    const ALT_QUERY_HEADS: usize = 4;
    const ALT_KEY_VALUE_HEADS: usize = 2;
    const ALT_HEAD_DIM: usize = 6;
    const ALT_PREFIX_TOKENS: usize = 3;
    const ALT_QUERY_WIDTH: usize = ALT_QUERY_HEADS * ALT_HEAD_DIM;
    const ALT_KEY_VALUE_WIDTH: usize = ALT_KEY_VALUE_HEADS * ALT_HEAD_DIM;

    let dense = |len: usize, mul: usize, add: usize, modulus: usize, divisor: f32| {
        (0..len)
            .map(|index| (((index * mul + add) % modulus) as f32 + 1.0) / divisor)
            .collect::<Vec<_>>()
    };
    let input = dense(ALT_HIDDEN, 7, 3, 23, 11.0);
    let raw_cos = dense(3 * ALT_HEAD_DIM, 5, 2, 19, 13.0);
    let raw_sin = dense(3 * ALT_HEAD_DIM, 11, 1, 29, 17.0);
    let prefix_keys = dense(ALT_PREFIX_TOKENS * ALT_KEY_VALUE_WIDTH, 13, 4, 31, 19.0);
    let prefix_values = dense(ALT_PREFIX_TOKENS * ALT_KEY_VALUE_WIDTH, 17, 6, 37, 23.0);
    let prefix_cache = DecoderPrefillKvCache {
        keys: prefix_keys,
        values: prefix_values,
        tokens: ALT_PREFIX_TOKENS,
        key_value_heads: ALT_KEY_VALUE_HEADS,
        head_dim: ALT_HEAD_DIM,
    };
    let parameters = OwnedParameters {
        input_norm_weight: dense(ALT_HIDDEN, 3, 1, 17, 7.0),
        query_weight: dense(ALT_QUERY_WIDTH * ALT_HIDDEN, 19, 5, 41, 29.0),
        key_weight: dense(ALT_KEY_VALUE_WIDTH * ALT_HIDDEN, 23, 7, 43, 31.0),
        value_weight: dense(ALT_KEY_VALUE_WIDTH * ALT_HIDDEN, 29, 9, 47, 37.0),
        attention_output_weight: dense(ALT_HIDDEN * ALT_QUERY_WIDTH, 31, 2, 53, 41.0),
        post_attention_norm_weight: dense(ALT_HIDDEN, 5, 2, 19, 11.0),
        gate_weight: dense(ALT_INTERMEDIATE * ALT_HIDDEN, 37, 3, 59, 43.0),
        up_weight: dense(ALT_INTERMEDIATE * ALT_HIDDEN, 41, 4, 61, 47.0),
        down_weight: dense(ALT_HIDDEN * ALT_INTERMEDIATE, 43, 8, 67, 53.0),
    };

    let error = pinned_decoder_layer_decode_f32(
        &input,
        &raw_cos,
        &raw_sin,
        &prefix_cache,
        parameters.borrowed(),
    )
    .unwrap_err();
    assert_eq!(
        error.code(),
        pvlc_cpu_ref::CpuRefErrorCode::DimensionMismatch
    );
}

#[test]
#[ignore = "release-only real-weight 333-token full-recompute vs cached decode"]
fn release_cached_decode_matches_last_row_of_full_333_prefill_exactly() {
    let decode = SafetensorsCatalog::open(fixture_path()).unwrap();
    let layer0 = SafetensorsCatalog::open(layer0_fixture_path()).unwrap();
    let official = load_official_decode_layer0(&decode, &layer0);
    let parameters = OwnedParameters::load(&layer0);

    let prefix_input = load(&layer0, "decoder.layer.00.input");
    let prefix_raw_cos = load(&layer0, "decoder.rope.cos.axis_major");
    let prefix_raw_sin = load(&layer0, "decoder.rope.sin.axis_major");
    let full_input = prefix_input
        .iter()
        .copied()
        .chain(official.input.iter().copied())
        .collect::<Vec<_>>();
    let full_raw_cos = append_axis_major_row(&prefix_raw_cos, &official.raw_cos);
    let full_raw_sin = append_axis_major_row(&prefix_raw_sin, &official.raw_sin);

    assert_f32_bits(
        "full_raw_cos last row",
        &slice_last_raw_axis_major_row(&full_raw_cos, FULL_CACHE_TOKENS, HEAD_DIM),
        &official.raw_cos,
    );
    assert_f32_bits(
        "full_raw_sin last row",
        &slice_last_raw_axis_major_row(&full_raw_sin, FULL_CACHE_TOKENS, HEAD_DIM),
        &official.raw_sin,
    );

    let prefix_trace = decoder_layer_prefill_f32(
        &prefix_input,
        decode_layer0_config(PREFIX_TOKENS),
        &prefix_raw_cos,
        &prefix_raw_sin,
        parameters.borrowed(),
    )
    .unwrap();
    let full_trace = decoder_layer_prefill_f32(
        &full_input,
        decode_layer0_config(FULL_CACHE_TOKENS),
        &full_raw_cos,
        &full_raw_sin,
        parameters.borrowed(),
    )
    .unwrap();
    let preserved = (
        official.input.clone(),
        official.raw_cos.clone(),
        official.raw_sin.clone(),
        prefix_trace.kv_cache.clone(),
        parameters.clone(),
    );
    let actual = decoder_layer_decode_f32(
        &official.input,
        pinned_single_token_decode_config(),
        &official.raw_cos,
        &official.raw_sin,
        &prefix_trace.kv_cache,
        parameters.borrowed(),
    )
    .unwrap();

    assert_decode_trace_matches_last_prefill_row_exact(&actual, &full_trace, FULL_CACHE_TOKENS);
    assert_f32_bits(
        "full cache keys",
        &actual.kv_cache.keys,
        &full_trace.kv_cache.keys,
    );
    assert_f32_bits(
        "full cache values",
        &actual.kv_cache.values,
        &full_trace.kv_cache.values,
    );
    assert_eq!(official.input, preserved.0);
    assert_eq!(official.raw_cos, preserved.1);
    assert_eq!(official.raw_sin, preserved.2);
    assert_eq!(prefix_trace.kv_cache, preserved.3);
    assert_eq!(parameters, preserved.4);
}
