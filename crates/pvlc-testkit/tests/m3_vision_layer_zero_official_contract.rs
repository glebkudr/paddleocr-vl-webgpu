use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use pvlc_cpu_ref::{
    KvBlockOrder, LayerNormParameters, LinearParameters, VisionEncoderLayerConfig,
    VisionEncoderLayerParameters, VisionEncoderStackConfig,
    add_interpolated_position_embedding_f32, add_vectors_f32, gelu_pytorch_tanh, layer_norm_f32,
    linear_f32, patch_projection_f32, streaming_segmented_attention_f32,
    vision_encoder_layer_identity_rope_f32, vision_encoder_stack_identity_rope_f32,
};
use pvlc_runtime_core::{
    VisionEncoderLayerInvocation as RuntimeLayerInvocation,
    VisionEncoderLayerParameters as RuntimeLayerParameters,
    VisionEncoderLayerStage as RuntimeLayerStage,
    VisionEncoderStackInvocation as RuntimeStackInvocation,
    VisionLayerNormParameters as RuntimeLayerNormParameters,
    VisionLinearParameters as RuntimeLinearParameters,
};
use pvlc_runtime_native::{BackendKind, NativeOptions, NativeRuntime, VisionLayerReadback};
use pvlc_safetensors::SafetensorsCatalog;
use pvlc_testkit::{ComparisonAxes, ComparisonPolicy, compare_f32};

const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const TOKENS: usize = 1_276;
const HIDDEN_SIZE: usize = 1_152;
const HEADS: usize = 16;
const HEAD_DIM: usize = 72;
const INTERMEDIATE_SIZE: usize = 4_304;
const LAYER_NORM_EPSILON: f32 = 1.0e-6;
const SAMPLED_TOKENS: [usize; 7] = [0, 178, 190, 237, 244, 253, 1_275];
const MODEL_PREFIX: &str = "visual.vision_model.encoder.layers.0";
const GOLDEN_LOCK: &str = include_str!("../../../goldens/golden.lock");
const MODEL_LOCK: &str = include_str!("../../../models/paddleocr-vl-1.6.lock");

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn require_model_checkpoint() -> Option<PathBuf> {
    let path = repository()
        .join("models/snapshots")
        .join(MODEL_REVISION)
        .join("model.safetensors");
    if path.is_file() {
        return Some(path);
    }
    assert_ne!(
        std::env::var("PVLC_REQUIRE_MODEL").as_deref(),
        Ok("1"),
        "PVLC_REQUIRE_MODEL=1 but pinned checkpoint is absent at {}",
        path.display()
    );
    eprintln!("skipping local vision-layer oracle: {}", path.display());
    None
}

fn assert_pinned_oracle_identity() {
    assert!(GOLDEN_LOCK.contains(&format!("model_revision = \"{MODEL_REVISION}\"")));
    assert!(MODEL_LOCK.contains(&format!("revision = \"{MODEL_REVISION}\"")));
    assert!(GOLDEN_LOCK.contains(
        "bundle_digest = \"blake3:35572ac07da5fddee97becb46fb866f906200f433f4c5677ad20fdf36440acf9\""
    ));
    assert!(GOLDEN_LOCK.contains(
        "semantic_fingerprint = \"blake3:632cbe2de6f47f8f84c764e3a551bf99f352d7459c98d939eb89725357f952b4\""
    ));
}

fn load_tensor(catalog: &SafetensorsCatalog, name: &str, shape: &[u64]) -> Vec<f32> {
    assert_eq!(catalog.tensor(name).unwrap().shape, shape);
    catalog.load_tensor_f32(name).unwrap()
}

fn model_name(suffix: &str) -> String {
    format!("{MODEL_PREFIX}.{suffix}")
}

fn layer_model_name(layer: usize, suffix: &str) -> String {
    format!("visual.vision_model.encoder.layers.{layer}.{suffix}")
}

struct OwnedLayerParameters {
    norm1_weight: Vec<f32>,
    norm1_bias: Vec<f32>,
    query_weight: Vec<f32>,
    query_bias: Vec<f32>,
    key_weight: Vec<f32>,
    key_bias: Vec<f32>,
    value_weight: Vec<f32>,
    value_bias: Vec<f32>,
    attention_output_weight: Vec<f32>,
    attention_output_bias: Vec<f32>,
    norm2_weight: Vec<f32>,
    norm2_bias: Vec<f32>,
    mlp_fc1_weight: Vec<f32>,
    mlp_fc1_bias: Vec<f32>,
    mlp_fc2_weight: Vec<f32>,
    mlp_fc2_bias: Vec<f32>,
}

impl OwnedLayerParameters {
    fn load(model: &SafetensorsCatalog) -> Self {
        Self::load_layer(model, 0)
    }

    fn load_layer(model: &SafetensorsCatalog, layer: usize) -> Self {
        let hidden = HIDDEN_SIZE as u64;
        let intermediate = INTERMEDIATE_SIZE as u64;
        let name = |suffix: &str| layer_model_name(layer, suffix);
        Self {
            norm1_weight: load_tensor(model, &name("layer_norm1.weight"), &[hidden]),
            norm1_bias: load_tensor(model, &name("layer_norm1.bias"), &[hidden]),
            query_weight: load_tensor(model, &name("self_attn.q_proj.weight"), &[hidden, hidden]),
            query_bias: load_tensor(model, &name("self_attn.q_proj.bias"), &[hidden]),
            key_weight: load_tensor(model, &name("self_attn.k_proj.weight"), &[hidden, hidden]),
            key_bias: load_tensor(model, &name("self_attn.k_proj.bias"), &[hidden]),
            value_weight: load_tensor(model, &name("self_attn.v_proj.weight"), &[hidden, hidden]),
            value_bias: load_tensor(model, &name("self_attn.v_proj.bias"), &[hidden]),
            attention_output_weight: load_tensor(
                model,
                &name("self_attn.out_proj.weight"),
                &[hidden, hidden],
            ),
            attention_output_bias: load_tensor(model, &name("self_attn.out_proj.bias"), &[hidden]),
            norm2_weight: load_tensor(model, &name("layer_norm2.weight"), &[hidden]),
            norm2_bias: load_tensor(model, &name("layer_norm2.bias"), &[hidden]),
            mlp_fc1_weight: load_tensor(model, &name("mlp.fc1.weight"), &[intermediate, hidden]),
            mlp_fc1_bias: load_tensor(model, &name("mlp.fc1.bias"), &[intermediate]),
            mlp_fc2_weight: load_tensor(model, &name("mlp.fc2.weight"), &[hidden, intermediate]),
            mlp_fc2_bias: load_tensor(model, &name("mlp.fc2.bias"), &[hidden]),
        }
    }

    fn borrowed(&self) -> VisionEncoderLayerParameters<'_> {
        VisionEncoderLayerParameters {
            norm1: LayerNormParameters {
                weight: &self.norm1_weight,
                bias: &self.norm1_bias,
            },
            query: LinearParameters {
                weight: &self.query_weight,
                bias: &self.query_bias,
            },
            key: LinearParameters {
                weight: &self.key_weight,
                bias: &self.key_bias,
            },
            value: LinearParameters {
                weight: &self.value_weight,
                bias: &self.value_bias,
            },
            attention_output: LinearParameters {
                weight: &self.attention_output_weight,
                bias: &self.attention_output_bias,
            },
            norm2: LayerNormParameters {
                weight: &self.norm2_weight,
                bias: &self.norm2_bias,
            },
            mlp_fc1: LinearParameters {
                weight: &self.mlp_fc1_weight,
                bias: &self.mlp_fc1_bias,
            },
            mlp_fc2: LinearParameters {
                weight: &self.mlp_fc2_weight,
                bias: &self.mlp_fc2_bias,
            },
        }
    }

    fn runtime_borrowed(&self) -> RuntimeLayerParameters<'_> {
        RuntimeLayerParameters {
            norm1: RuntimeLayerNormParameters {
                weight: &self.norm1_weight,
                bias: &self.norm1_bias,
            },
            query: RuntimeLinearParameters {
                weight: &self.query_weight,
                bias: &self.query_bias,
            },
            key: RuntimeLinearParameters {
                weight: &self.key_weight,
                bias: &self.key_bias,
            },
            value: RuntimeLinearParameters {
                weight: &self.value_weight,
                bias: &self.value_bias,
            },
            attention_output: RuntimeLinearParameters {
                weight: &self.attention_output_weight,
                bias: &self.attention_output_bias,
            },
            norm2: RuntimeLayerNormParameters {
                weight: &self.norm2_weight,
                bias: &self.norm2_bias,
            },
            mlp_fc1: RuntimeLinearParameters {
                weight: &self.mlp_fc1_weight,
                bias: &self.mlp_fc1_bias,
            },
            mlp_fc2: RuntimeLinearParameters {
                weight: &self.mlp_fc2_weight,
                bias: &self.mlp_fc2_bias,
            },
        }
    }
}

fn selected_rows(values: &[f32], width: usize) -> Vec<f32> {
    SAMPLED_TOKENS
        .iter()
        .flat_map(|row| values[row * width..(row + 1) * width].iter().copied())
        .collect()
}

fn policy(
    max_abs: f64,
    max_mean_abs: f64,
    max_p99_abs: f64,
    max_relative_l2: f64,
    min_cosine_similarity: f64,
    max_per_token_relative_l2: f64,
) -> ComparisonPolicy {
    ComparisonPolicy {
        require_finite: true,
        max_abs,
        max_mean_abs,
        max_p99_abs,
        max_relative_l2,
        min_cosine_similarity,
        max_per_token_relative_l2: Some(max_per_token_relative_l2),
        max_per_channel_relative_l2: None,
    }
}

fn norm_policy() -> ComparisonPolicy {
    policy(0.04, 0.003, 0.012, 0.004, 0.999_99, 0.006)
}

fn linear_policy() -> ComparisonPolicy {
    policy(0.08, 0.005, 0.02, 0.005, 0.999_98, 0.008)
}

fn attention_policy() -> ComparisonPolicy {
    policy(0.06, 0.004, 0.015, 0.005, 0.999_98, 0.008)
}

fn residual_policy() -> ComparisonPolicy {
    policy(0.032, 0.002, 0.008, 0.003, 0.999_99, 0.004)
}

fn mlp_policy() -> ComparisonPolicy {
    policy(0.15, 0.008, 0.03, 0.006, 0.999_97, 0.01)
}

fn accumulated_projection_policy() -> ComparisonPolicy {
    policy(0.04, 0.003, 0.018, 0.003, 0.999_99, 0.005)
}

fn accumulated_attention_policy() -> ComparisonPolicy {
    policy(0.15, 0.001, 0.006, 0.004, 0.999_99, 0.008)
}

fn accumulated_residual_policy() -> ComparisonPolicy {
    policy(0.35, 0.003, 0.009, 0.003, 0.999_99, 0.005)
}

// This envelope records the observed layer-0 FP32-vs-BF16 amplification only.
// It must not be reused as the acceptance budget for a composed 27-layer stack.
fn layer_zero_accumulated_norm_policy() -> ComparisonPolicy {
    policy(0.18, 0.009, 0.075, 0.038, 0.999_2, 0.052)
}

fn accumulated_mlp_policy() -> ComparisonPolicy {
    policy(0.12, 0.012, 0.045, 0.025, 0.999_6, 0.052)
}

fn accumulated_output_policy() -> ComparisonPolicy {
    policy(0.34, 0.005, 0.016, 0.0045, 0.999_98, 0.008)
}

fn layer_one_stack_policy() -> ComparisonPolicy {
    policy(0.4, 0.008, 0.03, 0.008, 0.999_97, 0.012)
}

fn layer_thirteen_stack_policy() -> ComparisonPolicy {
    policy(1.25, 0.015, 0.05, 0.015, 0.999_9, 0.04)
}

fn layer_twenty_six_stack_policy() -> ComparisonPolicy {
    policy(512.0, 0.5, 5.0, 0.04, 0.999, 0.7)
}

fn final_vision_stack_policy() -> ComparisonPolicy {
    policy(48.0, 0.012, 0.1, 0.018, 0.999_8, 0.35)
}

fn table_l2_final_vision_stack_policy() -> ComparisonPolicy {
    policy(36.0, 0.013, 0.11, 0.015, 0.999_88, 0.32)
}

fn native_cpu_layer_zero_stack_policy() -> ComparisonPolicy {
    policy(
        0.000_1,
        0.000_002,
        0.000_01,
        0.000_002,
        0.999_999_99,
        0.000_005,
    )
}

fn native_cpu_layer_one_stack_policy() -> ComparisonPolicy {
    policy(
        0.000_1,
        0.000_003,
        0.000_015,
        0.000_003,
        0.999_999_99,
        0.000_007,
    )
}

fn native_cpu_layer_thirteen_stack_policy() -> ComparisonPolicy {
    policy(
        0.000_3,
        0.000_01,
        0.000_04,
        0.000_01,
        0.999_999_99,
        0.000_03,
    )
}

fn native_cpu_layer_twenty_six_stack_policy() -> ComparisonPolicy {
    policy(0.5, 0.000_5, 0.005, 0.000_04, 0.999_999_9, 0.000_7)
}

fn native_cpu_final_stack_policy() -> ComparisonPolicy {
    policy(0.03, 0.000_012, 0.000_1, 0.000_016, 0.999_999_9, 0.000_4)
}

fn cpu_layer_output(
    input: &[f32],
    parameters: &OwnedLayerParameters,
) -> Result<Vec<f32>, pvlc_cpu_ref::CpuRefError> {
    vision_encoder_layer_identity_rope_f32(
        input,
        VisionEncoderLayerConfig {
            tokens: TOKENS,
            hidden_size: HIDDEN_SIZE,
            attention_heads: HEADS,
            head_dim: HEAD_DIM,
            intermediate_size: INTERMEDIATE_SIZE,
            layer_norm_epsilon: LAYER_NORM_EPSILON,
            attention_key_tile: 64,
            attention_order: KvBlockOrder::Forward,
        },
        &[0, TOKENS],
        parameters.borrowed(),
    )
    .map(|trace| trace.output)
}

fn assert_stage(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    rows: usize,
    width: usize,
    comparison_policy: &ComparisonPolicy,
) -> pvlc_testkit::ComparisonReport {
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
    report
}

fn assert_deep_stage(
    deep: &SafetensorsCatalog,
    name: &str,
    actual: &[f32],
    width: usize,
    comparison_policy: &ComparisonPolicy,
) {
    let expected = load_tensor(deep, name, &[1, TOKENS as u64, width as u64]);
    let report = assert_stage(name, &expected, actual, TOKENS, width, comparison_policy);
    let max_per_token = report
        .per_token_relative_l2
        .as_deref()
        .unwrap_or_default()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    eprintln!(
        "{name}: max_abs={:.9} mean_abs={:.9} p99_abs={:.9} rel_l2={:.9} cosine={:.9} max_token_rel={:.9}",
        report.max_abs,
        report.mean_abs,
        report.p99_abs,
        report.relative_l2,
        report.cosine_similarity,
        max_per_token,
    );
}

#[test]
fn pinned_layer_zero_components_match_the_official_identity_rope_trace() {
    assert_pinned_oracle_identity();
    let Some(model_path) = require_model_checkpoint() else {
        return;
    };

    let model = SafetensorsCatalog::open(model_path).unwrap();
    let deep = SafetensorsCatalog::open(
        repository()
            .join("artifacts/goldens/ocr.clean_latin.0001-l3")
            .join("deep-checkpoints.safetensors"),
    )
    .unwrap();

    let embeddings = load_tensor(
        &deep,
        "vision.embeddings.output",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let frequencies = load_tensor(&deep, "vision.rope.frequencies", &[58, 18]);
    assert_eq!(frequencies, vec![0.0; 58 * 18]);

    let norm1_weight = load_tensor(
        &model,
        &model_name("layer_norm1.weight"),
        &[HIDDEN_SIZE as u64],
    );
    let norm1_bias = load_tensor(
        &model,
        &model_name("layer_norm1.bias"),
        &[HIDDEN_SIZE as u64],
    );
    let official_norm1 = load_tensor(
        &deep,
        "vision.layer.00.norm1",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let actual_norm1 = layer_norm_f32(
        &embeddings,
        TOKENS,
        HIDDEN_SIZE,
        &norm1_weight,
        &norm1_bias,
        LAYER_NORM_EPSILON,
    )
    .unwrap();
    assert_stage(
        "layer_norm1",
        &official_norm1,
        &actual_norm1,
        TOKENS,
        HIDDEN_SIZE,
        &norm_policy(),
    );

    let sampled_norm1 = selected_rows(&official_norm1, HIDDEN_SIZE);
    for projection in ["q", "k", "v"] {
        let weight = load_tensor(
            &model,
            &model_name(&format!("self_attn.{projection}_proj.weight")),
            &[HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
        );
        let bias = load_tensor(
            &model,
            &model_name(&format!("self_attn.{projection}_proj.bias")),
            &[HIDDEN_SIZE as u64],
        );
        let official = load_tensor(
            &deep,
            &format!("vision.layer.00.{projection}"),
            &[1, TOKENS as u64, HIDDEN_SIZE as u64],
        );
        let actual = linear_f32(
            &sampled_norm1,
            SAMPLED_TOKENS.len(),
            HIDDEN_SIZE,
            &weight,
            &bias,
            HIDDEN_SIZE,
        )
        .unwrap();
        assert_stage(
            &format!("{projection}_projection"),
            &selected_rows(&official, HIDDEN_SIZE),
            &actual,
            SAMPLED_TOKENS.len(),
            HIDDEN_SIZE,
            &linear_policy(),
        );
    }

    let query = load_tensor(
        &deep,
        "vision.layer.00.q",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let key = load_tensor(
        &deep,
        "vision.layer.00.k",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let value = load_tensor(
        &deep,
        "vision.layer.00.v",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    assert_ne!(query, key);
    assert_ne!(query, value);
    assert_ne!(key, value);
    let official_context = load_tensor(
        &deep,
        "vision.layer.00.attention.context",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let actual_context = streaming_segmented_attention_f32(
        &query,
        &key,
        &value,
        TOKENS,
        HEADS,
        HEAD_DIM,
        &[0, TOKENS],
        64,
        KvBlockOrder::Forward,
    )
    .unwrap();
    assert_stage(
        "identity_rope_streaming_attention",
        &official_context,
        &actual_context,
        TOKENS,
        HIDDEN_SIZE,
        &attention_policy(),
    );

    let out_weight = load_tensor(
        &model,
        &model_name("self_attn.out_proj.weight"),
        &[HIDDEN_SIZE as u64, HIDDEN_SIZE as u64],
    );
    let out_bias = load_tensor(
        &model,
        &model_name("self_attn.out_proj.bias"),
        &[HIDDEN_SIZE as u64],
    );
    let official_attention_output = load_tensor(
        &deep,
        "vision.layer.00.attention.output",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let actual_attention_output = linear_f32(
        &selected_rows(&official_context, HIDDEN_SIZE),
        SAMPLED_TOKENS.len(),
        HIDDEN_SIZE,
        &out_weight,
        &out_bias,
        HIDDEN_SIZE,
    )
    .unwrap();
    assert_stage(
        "attention_out_projection",
        &selected_rows(&official_attention_output, HIDDEN_SIZE),
        &actual_attention_output,
        SAMPLED_TOKENS.len(),
        HIDDEN_SIZE,
        &linear_policy(),
    );

    let official_attention_residual = load_tensor(
        &deep,
        "vision.layer.00.attention.residual",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let actual_attention_residual = add_vectors_f32(
        &selected_rows(&embeddings, HIDDEN_SIZE),
        &selected_rows(&official_attention_output, HIDDEN_SIZE),
    )
    .unwrap();
    assert_stage(
        "attention_residual",
        &selected_rows(&official_attention_residual, HIDDEN_SIZE),
        &actual_attention_residual,
        SAMPLED_TOKENS.len(),
        HIDDEN_SIZE,
        &residual_policy(),
    );

    let norm2_weight = load_tensor(
        &model,
        &model_name("layer_norm2.weight"),
        &[HIDDEN_SIZE as u64],
    );
    let norm2_bias = load_tensor(
        &model,
        &model_name("layer_norm2.bias"),
        &[HIDDEN_SIZE as u64],
    );
    let official_norm2 = load_tensor(
        &deep,
        "vision.layer.00.norm2",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let actual_norm2 = layer_norm_f32(
        &selected_rows(&official_attention_residual, HIDDEN_SIZE),
        SAMPLED_TOKENS.len(),
        HIDDEN_SIZE,
        &norm2_weight,
        &norm2_bias,
        LAYER_NORM_EPSILON,
    )
    .unwrap();
    assert_stage(
        "layer_norm2",
        &selected_rows(&official_norm2, HIDDEN_SIZE),
        &actual_norm2,
        SAMPLED_TOKENS.len(),
        HIDDEN_SIZE,
        &norm_policy(),
    );

    let fc1_weight = load_tensor(
        &model,
        &model_name("mlp.fc1.weight"),
        &[INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
    );
    let fc1_bias = load_tensor(
        &model,
        &model_name("mlp.fc1.bias"),
        &[INTERMEDIATE_SIZE as u64],
    );
    let official_fc1 = load_tensor(
        &deep,
        "vision.layer.00.mlp.fc1",
        &[1, TOKENS as u64, INTERMEDIATE_SIZE as u64],
    );
    let actual_fc1 = linear_f32(
        &selected_rows(&official_norm2, HIDDEN_SIZE),
        SAMPLED_TOKENS.len(),
        HIDDEN_SIZE,
        &fc1_weight,
        &fc1_bias,
        INTERMEDIATE_SIZE,
    )
    .unwrap();
    assert_stage(
        "mlp_fc1",
        &selected_rows(&official_fc1, INTERMEDIATE_SIZE),
        &actual_fc1,
        SAMPLED_TOKENS.len(),
        INTERMEDIATE_SIZE,
        &mlp_policy(),
    );

    let official_activation = load_tensor(
        &deep,
        "vision.layer.00.mlp.activation",
        &[1, TOKENS as u64, INTERMEDIATE_SIZE as u64],
    );
    let actual_activation = selected_rows(&official_fc1, INTERMEDIATE_SIZE)
        .into_iter()
        .map(gelu_pytorch_tanh)
        .collect::<Vec<_>>();
    assert_stage(
        "mlp_gelu_tanh",
        &selected_rows(&official_activation, INTERMEDIATE_SIZE),
        &actual_activation,
        SAMPLED_TOKENS.len(),
        INTERMEDIATE_SIZE,
        &mlp_policy(),
    );

    let fc2_weight = load_tensor(
        &model,
        &model_name("mlp.fc2.weight"),
        &[HIDDEN_SIZE as u64, INTERMEDIATE_SIZE as u64],
    );
    let fc2_bias = load_tensor(&model, &model_name("mlp.fc2.bias"), &[HIDDEN_SIZE as u64]);
    let official_mlp_output = load_tensor(
        &deep,
        "vision.layer.00.mlp.output",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let actual_mlp_output = linear_f32(
        &selected_rows(&official_activation, INTERMEDIATE_SIZE),
        SAMPLED_TOKENS.len(),
        INTERMEDIATE_SIZE,
        &fc2_weight,
        &fc2_bias,
        HIDDEN_SIZE,
    )
    .unwrap();
    assert_stage(
        "mlp_fc2",
        &selected_rows(&official_mlp_output, HIDDEN_SIZE),
        &actual_mlp_output,
        SAMPLED_TOKENS.len(),
        HIDDEN_SIZE,
        &mlp_policy(),
    );

    let official_layer_output = load_tensor(
        &deep,
        "vision.layer.00.output",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let actual_layer_output = add_vectors_f32(
        &selected_rows(&official_attention_residual, HIDDEN_SIZE),
        &selected_rows(&official_mlp_output, HIDDEN_SIZE),
    )
    .unwrap();
    assert_stage(
        "mlp_residual",
        &selected_rows(&official_layer_output, HIDDEN_SIZE),
        &actual_layer_output,
        SAMPLED_TOKENS.len(),
        HIDDEN_SIZE,
        &residual_policy(),
    );
}

#[test]
#[ignore = "M3e CPU hard gate composes every operation over the full 1,276-token layer"]
fn full_composed_layer_zero_matches_the_official_identity_rope_trace() {
    assert_pinned_oracle_identity();
    let Some(model_path) = require_model_checkpoint() else {
        return;
    };
    let model = SafetensorsCatalog::open(model_path).unwrap();
    let deep = SafetensorsCatalog::open(
        repository()
            .join("artifacts/goldens/ocr.clean_latin.0001-l3")
            .join("deep-checkpoints.safetensors"),
    )
    .unwrap();
    let input = load_tensor(
        &deep,
        "vision.embeddings.output",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let frequencies = load_tensor(&deep, "vision.rope.frequencies", &[58, 18]);
    assert_eq!(frequencies, vec![0.0; 58 * 18]);
    let owned_parameters = OwnedLayerParameters::load(&model);
    let trace = vision_encoder_layer_identity_rope_f32(
        &input,
        VisionEncoderLayerConfig {
            tokens: TOKENS,
            hidden_size: HIDDEN_SIZE,
            attention_heads: HEADS,
            head_dim: HEAD_DIM,
            intermediate_size: INTERMEDIATE_SIZE,
            layer_norm_epsilon: LAYER_NORM_EPSILON,
            attention_key_tile: 64,
            attention_order: KvBlockOrder::Forward,
        },
        &[0, TOKENS],
        owned_parameters.borrowed(),
    )
    .unwrap();

    assert_deep_stage(
        &deep,
        "vision.layer.00.norm1",
        &trace.norm1,
        HIDDEN_SIZE,
        &norm_policy(),
    );
    for (name, actual) in [
        ("vision.layer.00.q", &trace.query[..]),
        ("vision.layer.00.k", &trace.key[..]),
        ("vision.layer.00.v", &trace.value[..]),
    ] {
        assert_deep_stage(
            &deep,
            name,
            actual,
            HIDDEN_SIZE,
            &accumulated_projection_policy(),
        );
    }
    for (name, actual) in [
        (
            "vision.layer.00.attention.context",
            &trace.attention_context[..],
        ),
        (
            "vision.layer.00.attention.output",
            &trace.attention_output[..],
        ),
    ] {
        assert_deep_stage(
            &deep,
            name,
            actual,
            HIDDEN_SIZE,
            &accumulated_attention_policy(),
        );
    }
    assert_deep_stage(
        &deep,
        "vision.layer.00.attention.residual",
        &trace.attention_residual,
        HIDDEN_SIZE,
        &accumulated_residual_policy(),
    );
    assert_deep_stage(
        &deep,
        "vision.layer.00.norm2",
        &trace.norm2,
        HIDDEN_SIZE,
        &layer_zero_accumulated_norm_policy(),
    );
    for (name, actual, width) in [
        (
            "vision.layer.00.mlp.fc1",
            &trace.mlp_fc1[..],
            INTERMEDIATE_SIZE,
        ),
        (
            "vision.layer.00.mlp.activation",
            &trace.mlp_activation[..],
            INTERMEDIATE_SIZE,
        ),
        (
            "vision.layer.00.mlp.output",
            &trace.mlp_output[..],
            HIDDEN_SIZE,
        ),
    ] {
        assert_deep_stage(&deep, name, actual, width, &accumulated_mlp_policy());
    }
    assert_deep_stage(
        &deep,
        "vision.layer.00.output",
        &trace.output,
        HIDDEN_SIZE,
        &accumulated_output_policy(),
    );
}

#[test]
#[ignore = "M3f native hard gate runs the full 1,276-token resident layer in one command buffer"]
fn native_resident_layer_zero_matches_every_official_identity_rope_checkpoint() {
    assert_pinned_oracle_identity();
    let Some(model_path) = require_model_checkpoint() else {
        return;
    };
    let runtime = match NativeRuntime::new(NativeOptions::default()) {
        Ok(runtime) => runtime,
        Err(error)
            if std::env::var("PVLC_REQUIRE_NATIVE_GPU").as_deref() != Ok("1")
                && std::env::var("PVLC_REQUIRE_M4_METAL").as_deref() != Ok("1") =>
        {
            eprintln!("skipping native resident layer hard gate: {error}");
            return;
        }
        Err(error) => panic!("native GPU is required: {error}"),
    };
    if std::env::var("PVLC_REQUIRE_M4_METAL").as_deref() == Ok("1") {
        assert_eq!(runtime.capabilities().backend, BackendKind::Metal);
        assert!(runtime.capabilities().adapter_name.contains("M4 Pro"));
    }

    let model = SafetensorsCatalog::open(model_path).unwrap();
    let deep = SafetensorsCatalog::open(
        repository()
            .join("artifacts/goldens/ocr.clean_latin.0001-l3")
            .join("deep-checkpoints.safetensors"),
    )
    .unwrap();
    let input = load_tensor(
        &deep,
        "vision.embeddings.output",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let frequencies = load_tensor(&deep, "vision.rope.frequencies", &[58, 18]);
    assert_eq!(frequencies, vec![0.0; 58 * 18]);
    let owned_parameters = OwnedLayerParameters::load(&model);
    let boundaries = [0, TOKENS as u32];
    let invocation = RuntimeLayerInvocation {
        tokens: TOKENS as u32,
        hidden_size: HIDDEN_SIZE as u32,
        attention_heads: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        intermediate_size: INTERMEDIATE_SIZE as u32,
        layer_norm_epsilon: LAYER_NORM_EPSILON,
        input: &input,
        cu_seqlens: &boundaries,
        parameters: owned_parameters.runtime_borrowed(),
    };

    let before = runtime.counters();
    let execution = runtime
        .run_vision_encoder_layer_identity_rope(&invocation, VisionLayerReadback::AllStages)
        .unwrap();
    let after = runtime.counters();
    assert_eq!(after.submissions - before.submissions, 1);
    assert_eq!(execution.diagnostics.submission_count, 1);
    assert_eq!(execution.diagnostics.command_buffer_count, 1);
    assert_eq!(execution.diagnostics.readback_buffer_count, 1);
    assert_eq!(
        execution.diagnostics.buffer_allocation_count,
        after.buffer_allocations - before.buffer_allocations
    );
    assert!(execution.diagnostics.buffer_allocation_count <= 32);
    assert!(execution.diagnostics.queue_wall_time_ns > 0);
    assert!(execution.diagnostics.captured_errors.is_empty());
    assert_eq!(execution.checkpoints.len(), RuntimeLayerStage::ALL.len());
    assert_eq!(
        execution.diagnostics.readback_bytes,
        ((10 * TOKENS * HIDDEN_SIZE + 2 * TOKENS * INTERMEDIATE_SIZE) * 4) as u64
    );
    if runtime.capabilities().timestamp_query {
        let timestamp = execution.diagnostics.timestamp.unwrap();
        assert!(timestamp.end_ticks > timestamp.begin_ticks);
        assert!(timestamp.duration_ns.is_finite() && timestamp.duration_ns > 0.0);
        assert_eq!(execution.diagnostics.timestamp_fresh, Some(true));
        eprintln!(
            "native layer-0 GPU duration: {:.3} ms; queue wall: {:.3} ms",
            timestamp.duration_ns / 1_000_000.0,
            execution.diagnostics.queue_wall_time_ns as f64 / 1_000_000.0
        );
    } else {
        assert!(execution.diagnostics.timestamp.is_none());
        assert!(execution.diagnostics.timestamp_fresh.is_none());
    }

    let checkpoint = |stage| execution.checkpoints.get(&stage).unwrap();
    assert_deep_stage(
        &deep,
        "vision.layer.00.norm1",
        checkpoint(RuntimeLayerStage::Norm1),
        HIDDEN_SIZE,
        &norm_policy(),
    );
    for (name, stage) in [
        ("vision.layer.00.q", RuntimeLayerStage::Query),
        ("vision.layer.00.k", RuntimeLayerStage::Key),
        ("vision.layer.00.v", RuntimeLayerStage::Value),
    ] {
        assert_deep_stage(
            &deep,
            name,
            checkpoint(stage),
            HIDDEN_SIZE,
            &accumulated_projection_policy(),
        );
    }
    for (name, stage) in [
        (
            "vision.layer.00.attention.context",
            RuntimeLayerStage::AttentionContext,
        ),
        (
            "vision.layer.00.attention.output",
            RuntimeLayerStage::AttentionOutput,
        ),
    ] {
        assert_deep_stage(
            &deep,
            name,
            checkpoint(stage),
            HIDDEN_SIZE,
            &accumulated_attention_policy(),
        );
    }
    assert_deep_stage(
        &deep,
        "vision.layer.00.attention.residual",
        checkpoint(RuntimeLayerStage::AttentionResidual),
        HIDDEN_SIZE,
        &accumulated_residual_policy(),
    );
    assert_deep_stage(
        &deep,
        "vision.layer.00.norm2",
        checkpoint(RuntimeLayerStage::Norm2),
        HIDDEN_SIZE,
        &layer_zero_accumulated_norm_policy(),
    );
    for (name, stage, width) in [
        (
            "vision.layer.00.mlp.fc1",
            RuntimeLayerStage::MlpFc1,
            INTERMEDIATE_SIZE,
        ),
        (
            "vision.layer.00.mlp.activation",
            RuntimeLayerStage::MlpActivation,
            INTERMEDIATE_SIZE,
        ),
        (
            "vision.layer.00.mlp.output",
            RuntimeLayerStage::MlpOutput,
            HIDDEN_SIZE,
        ),
    ] {
        assert_deep_stage(
            &deep,
            name,
            checkpoint(stage),
            width,
            &accumulated_mlp_policy(),
        );
    }
    assert_deep_stage(
        &deep,
        "vision.layer.00.output",
        checkpoint(RuntimeLayerStage::Output),
        HIDDEN_SIZE,
        &accumulated_output_policy(),
    );
}

fn assert_stack_stage(
    label: &str,
    reference: &[f32],
    candidate: &[f32],
    comparison_policy: &ComparisonPolicy,
) {
    let report = compare_f32(
        reference,
        candidate,
        &[TOKENS, HIDDEN_SIZE],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap();
    let verdict = report.assess(comparison_policy).unwrap();
    let max_token = report
        .per_token_relative_l2
        .as_deref()
        .unwrap()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    eprintln!(
        "{label}: max_abs={:.9} mean_abs={:.9} p99_abs={:.9} rel_l2={:.9} cosine={:.9} max_token_rel={:.9}",
        report.max_abs,
        report.mean_abs,
        report.p99_abs,
        report.relative_l2,
        report.cosine_similarity,
        max_token,
    );
    assert!(
        verdict.passed(),
        "{label}\n{report:#?}\nviolations: {:?}",
        verdict.violations()
    );
}

fn assert_stack_policy_rejects(
    label: &str,
    reference: &[f32],
    candidate: &[f32],
    comparison_policy: &ComparisonPolicy,
) {
    let report = compare_f32(
        reference,
        candidate,
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
        "{label}: structural negative unexpectedly passed\n{report:#?}"
    );
}

#[test]
#[ignore = "M3h CPU hard gate streams all 27 official vision layers and post norm"]
fn full_vision_stack_and_post_norm_match_pinned_depth_checkpoints() {
    assert_eq!(std::env::var("PVLC_REQUIRE_MODEL").as_deref(), Ok("1"));
    assert_pinned_oracle_identity();
    let model = SafetensorsCatalog::open(require_model_checkpoint().unwrap()).unwrap();
    let golden = repository().join("artifacts/goldens/ocr.clean_latin.0001-l3");
    let deep = SafetensorsCatalog::open(golden.join("deep-checkpoints.safetensors")).unwrap();
    let stage = SafetensorsCatalog::open(golden.join("stage-checkpoints.safetensors")).unwrap();
    let frequencies = load_tensor(&deep, "vision.rope.frequencies", &[58, 18]);
    assert_eq!(frequencies, vec![0.0; 58 * 18]);
    let hidden = load_tensor(
        &deep,
        "vision.embeddings.output",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let selected = [0_usize, 1, 13, 26];
    let post_weight = load_tensor(
        &model,
        "visual.vision_model.post_layernorm.weight",
        &[HIDDEN_SIZE as u64],
    );
    let post_bias = load_tensor(
        &model,
        "visual.vision_model.post_layernorm.bias",
        &[HIDDEN_SIZE as u64],
    );
    let started = std::time::Instant::now();
    let trace = vision_encoder_stack_identity_rope_f32(
        &hidden,
        VisionEncoderStackConfig {
            tokens: TOKENS,
            hidden_size: HIDDEN_SIZE,
            layers: 27,
            layer_norm_epsilon: LAYER_NORM_EPSILON,
        },
        &selected,
        LayerNormParameters {
            weight: &post_weight,
            bias: &post_bias,
        },
        |layer, input| {
            let layer_started = std::time::Instant::now();
            let parameters = OwnedLayerParameters::load_layer(&model, layer);
            let output = cpu_layer_output(input, &parameters)?;
            eprintln!(
                "vision layer {layer:02}: {:.3}s",
                layer_started.elapsed().as_secs_f64()
            );
            Ok(output)
        },
    )
    .unwrap();
    assert_eq!(trace.executed_layers, 27);
    assert_eq!(
        trace
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.layer_index)
            .collect::<Vec<_>>(),
        selected
    );
    assert_eq!(
        trace.retained_checkpoint_elements,
        selected.len() * TOKENS * HIDDEN_SIZE
    );
    let mut depth_references = BTreeMap::new();
    for (layer, comparison_policy) in [
        (0, accumulated_output_policy()),
        (1, layer_one_stack_policy()),
        (13, layer_thirteen_stack_policy()),
        (26, layer_twenty_six_stack_policy()),
    ] {
        let name = format!("vision.layer.{layer:02}.output");
        let reference = load_tensor(&deep, &name, &[1, TOKENS as u64, HIDDEN_SIZE as u64]);
        assert_stack_stage(
            &name,
            &reference,
            trace.checkpoint(layer).unwrap(),
            &comparison_policy,
        );
        depth_references.insert(layer, reference);
    }
    for (target, source, comparison_policy) in [
        (13, 1, layer_thirteen_stack_policy()),
        (26, 13, layer_twenty_six_stack_policy()),
    ] {
        assert_stack_policy_rejects(
            &format!("vision.layer.{target:02} rejects layer {source:02}"),
            depth_references.get(&target).unwrap(),
            trace.checkpoint(source).unwrap(),
            &comparison_policy,
        );
        let mut token_rotated = trace.checkpoint(target).unwrap().to_vec();
        token_rotated.rotate_left(HIDDEN_SIZE);
        assert_stack_policy_rejects(
            &format!("vision.layer.{target:02} rejects token rotation"),
            depth_references.get(&target).unwrap(),
            &token_rotated,
            &comparison_policy,
        );
    }
    let reference = load_tensor(&stage, "vision.final", &[TOKENS as u64, HIDDEN_SIZE as u64]);
    let final_policy = final_vision_stack_policy();
    assert_stack_stage("vision.final", &reference, &trace.output, &final_policy);
    assert_stack_policy_rejects(
        "vision.final rejects pre-norm layer 26",
        &reference,
        trace.checkpoint(26).unwrap(),
        &final_policy,
    );
    let mut token_rotated = trace.output.clone();
    token_rotated.rotate_left(HIDDEN_SIZE);
    assert_stack_policy_rejects(
        "vision.final rejects token rotation",
        &reference,
        &token_rotated,
        &final_policy,
    );
    eprintln!(
        "full 27-layer CPU stack hard gate: {:.3}s",
        started.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "M3i native hard gate keeps all 27 vision layers and post norm GPU-resident"]
fn native_resident_full_vision_stack_matches_cpu_and_pinned_depth_checkpoints() {
    assert_eq!(std::env::var("PVLC_REQUIRE_MODEL").as_deref(), Ok("1"));
    assert_eq!(std::env::var("PVLC_REQUIRE_NATIVE_GPU").as_deref(), Ok("1"));
    assert_eq!(std::env::var("PVLC_REQUIRE_M4_METAL").as_deref(), Ok("1"));
    assert_pinned_oracle_identity();
    let runtime = NativeRuntime::new(NativeOptions::default()).unwrap();
    assert_eq!(runtime.capabilities().backend, BackendKind::Metal);
    assert!(runtime.capabilities().adapter_name.contains("M4 Pro"));

    let model = SafetensorsCatalog::open(require_model_checkpoint().unwrap()).unwrap();
    let golden = repository().join("artifacts/goldens/ocr.clean_latin.0001-l3");
    let deep = SafetensorsCatalog::open(golden.join("deep-checkpoints.safetensors")).unwrap();
    let stage = SafetensorsCatalog::open(golden.join("stage-checkpoints.safetensors")).unwrap();
    let frequencies = load_tensor(&deep, "vision.rope.frequencies", &[58, 18]);
    assert_eq!(frequencies, vec![0.0; 58 * 18]);
    let input = load_tensor(
        &deep,
        "vision.embeddings.output",
        &[1, TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let post_weight = load_tensor(
        &model,
        "visual.vision_model.post_layernorm.weight",
        &[HIDDEN_SIZE as u64],
    );
    let post_bias = load_tensor(
        &model,
        "visual.vision_model.post_layernorm.bias",
        &[HIDDEN_SIZE as u64],
    );
    let selected = [0_usize, 1, 13, 26];
    let parameters = (0..27)
        .map(|layer| OwnedLayerParameters::load_layer(&model, layer))
        .collect::<Vec<_>>();

    let cpu_started = std::time::Instant::now();
    let cpu = vision_encoder_stack_identity_rope_f32(
        &input,
        VisionEncoderStackConfig {
            tokens: TOKENS,
            hidden_size: HIDDEN_SIZE,
            layers: parameters.len(),
            layer_norm_epsilon: LAYER_NORM_EPSILON,
        },
        &selected,
        LayerNormParameters {
            weight: &post_weight,
            bias: &post_bias,
        },
        |layer, current| cpu_layer_output(current, &parameters[layer]),
    )
    .unwrap();
    eprintln!(
        "full 27-layer CPU baseline inside native hard gate: {:.3}s",
        cpu_started.elapsed().as_secs_f64()
    );

    let runtime_parameters = parameters
        .iter()
        .map(OwnedLayerParameters::runtime_borrowed)
        .collect::<Vec<_>>();
    let boundaries = [0_u32, TOKENS as u32];
    let invocation = RuntimeStackInvocation {
        tokens: TOKENS as u32,
        hidden_size: HIDDEN_SIZE as u32,
        attention_heads: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        intermediate_size: INTERMEDIATE_SIZE as u32,
        layer_norm_epsilon: LAYER_NORM_EPSILON,
        input: &input,
        cu_seqlens: &boundaries,
        layer_parameters: &runtime_parameters,
        post_norm: RuntimeLayerNormParameters {
            weight: &post_weight,
            bias: &post_bias,
        },
    };
    let before = runtime.counters();
    let native_started = std::time::Instant::now();
    let native = runtime
        .run_vision_encoder_stack_identity_rope(&invocation, &selected)
        .unwrap();
    let after = runtime.counters();
    assert_eq!(after.submissions - before.submissions, 1);
    assert_eq!(native.diagnostics.submission_count, 1);
    assert_eq!(native.diagnostics.command_buffer_count, 1);
    assert_eq!(native.diagnostics.layer_count, 27);
    assert_eq!(native.diagnostics.dispatch_count, 27 * 12 + 1);
    assert_eq!(native.diagnostics.compute_pass_count, 28);
    assert_eq!(native.diagnostics.activation_buffer_count, 13);
    assert_eq!(native.diagnostics.weight_buffer_count, 27 * 16 + 2);
    assert_eq!(native.diagnostics.readback_buffer_count, 1);
    assert_eq!(native.diagnostics.readback_map_count, 1);
    assert_eq!(
        native.diagnostics.readback_bytes,
        ((selected.len() + 1) * TOKENS * HIDDEN_SIZE * 4) as u64
    );
    assert_eq!(
        native.diagnostics.buffer_allocation_count,
        after.buffer_allocations - before.buffer_allocations
    );
    assert_eq!(native.diagnostics.buffer_allocation_count, 450);
    assert!(native.diagnostics.captured_errors.is_empty());
    assert_eq!(
        native.checkpoints.keys().copied().collect::<Vec<_>>(),
        selected
    );
    if runtime.capabilities().timestamp_query {
        let timestamp = native.diagnostics.timestamp.unwrap();
        assert!(timestamp.end_ticks > timestamp.begin_ticks);
        assert!(timestamp.duration_ns.is_finite() && timestamp.duration_ns > 0.0);
        assert_eq!(native.diagnostics.timestamp_fresh, Some(true));
        eprintln!(
            "native sampled layer-0 GPU duration: {:.3} ms; full-stack queue wall: {:.3} ms",
            timestamp.duration_ns / 1_000_000.0,
            native.diagnostics.queue_wall_time_ns as f64 / 1_000_000.0
        );
    }
    eprintln!(
        "full 27-layer native stack hard gate: {:.3}s",
        native_started.elapsed().as_secs_f64()
    );

    for (layer, official_policy, native_cpu_policy) in [
        (
            0,
            accumulated_output_policy(),
            native_cpu_layer_zero_stack_policy(),
        ),
        (
            1,
            layer_one_stack_policy(),
            native_cpu_layer_one_stack_policy(),
        ),
        (
            13,
            layer_thirteen_stack_policy(),
            native_cpu_layer_thirteen_stack_policy(),
        ),
        (
            26,
            layer_twenty_six_stack_policy(),
            native_cpu_layer_twenty_six_stack_policy(),
        ),
    ] {
        let name = format!("vision.layer.{layer:02}.output");
        let official = load_tensor(&deep, &name, &[1, TOKENS as u64, HIDDEN_SIZE as u64]);
        assert_stack_stage(
            &format!("{name} native-vs-official"),
            &official,
            native.checkpoints.get(&layer).unwrap(),
            &official_policy,
        );
        assert_stack_stage(
            &format!("{name} native-vs-cpu"),
            cpu.checkpoint(layer).unwrap(),
            native.checkpoints.get(&layer).unwrap(),
            &native_cpu_policy,
        );
    }
    for (target, source, native_cpu_policy) in [
        (13, 1, native_cpu_layer_thirteen_stack_policy()),
        (26, 13, native_cpu_layer_twenty_six_stack_policy()),
    ] {
        assert_stack_policy_rejects(
            &format!("native layer {target:02} policy rejects layer {source:02}"),
            cpu.checkpoint(target).unwrap(),
            native.checkpoints.get(&source).unwrap(),
            &native_cpu_policy,
        );
        let mut token_rotated = native.checkpoints.get(&target).unwrap().clone();
        token_rotated.rotate_left(HIDDEN_SIZE);
        assert_stack_policy_rejects(
            &format!("native layer {target:02} policy rejects token rotation"),
            cpu.checkpoint(target).unwrap(),
            &token_rotated,
            &native_cpu_policy,
        );
    }

    let official_final = load_tensor(&stage, "vision.final", &[TOKENS as u64, HIDDEN_SIZE as u64]);
    assert_stack_stage(
        "vision.final native-vs-official",
        &official_final,
        &native.output,
        &final_vision_stack_policy(),
    );
    let final_cpu_policy = native_cpu_final_stack_policy();
    assert_stack_stage(
        "vision.final native-vs-cpu",
        &cpu.output,
        &native.output,
        &final_cpu_policy,
    );
    assert_stack_policy_rejects(
        "native final policy rejects pre-norm layer 26",
        &cpu.output,
        native.checkpoints.get(&26).unwrap(),
        &final_cpu_policy,
    );
    let mut token_rotated = native.output.clone();
    token_rotated.rotate_left(HIDDEN_SIZE);
    assert_stack_policy_rejects(
        "native final policy rejects token rotation",
        &cpu.output,
        &token_rotated,
        &final_cpu_policy,
    );
}

#[test]
#[ignore = "M3k native evidence calibrates the second official shape final policy"]
fn native_table_l2_vision_stack_matches_the_pinned_final_checkpoint() {
    assert_eq!(std::env::var("PVLC_REQUIRE_MODEL").as_deref(), Ok("1"));
    assert_eq!(std::env::var("PVLC_REQUIRE_NATIVE_GPU").as_deref(), Ok("1"));
    assert_eq!(std::env::var("PVLC_REQUIRE_M4_METAL").as_deref(), Ok("1"));
    assert!(GOLDEN_LOCK.contains(
        "bundle_digest = \"blake3:4cecc8b2abc37030920acf8860bb317084125dbbbcae5af394fccd0c8044c842\""
    ));
    assert!(GOLDEN_LOCK.contains(
        "semantic_fingerprint = \"blake3:91f72ffdf81070c2c9ca31628605064bdc02bf19417482cefce41d6a27aa5404\""
    ));
    let runtime = NativeRuntime::new(NativeOptions::default()).unwrap();
    assert_eq!(runtime.capabilities().backend, BackendKind::Metal);
    assert!(runtime.capabilities().adapter_name.contains("M4 Pro"));

    const TABLE_TOKENS: usize = 1_740;
    let model = SafetensorsCatalog::open(require_model_checkpoint().unwrap()).unwrap();
    let golden = repository().join("artifacts/goldens/table.simple.0001-l2");
    let processor = SafetensorsCatalog::open(golden.join("processor.safetensors")).unwrap();
    let stage = SafetensorsCatalog::open(golden.join("stage-checkpoints.safetensors")).unwrap();
    let pixels = load_tensor(
        &processor,
        "processor.pixel_values",
        &[TABLE_TOKENS as u64, 3, 14, 14],
    );
    let patch_weight = load_tensor(
        &model,
        "visual.vision_model.embeddings.patch_embedding.weight",
        &[HIDDEN_SIZE as u64, 3, 14, 14],
    );
    let patch_bias = load_tensor(
        &model,
        "visual.vision_model.embeddings.patch_embedding.bias",
        &[HIDDEN_SIZE as u64],
    );
    let positions = load_tensor(
        &model,
        "visual.vision_model.embeddings.position_embedding.weight",
        &[729, HIDDEN_SIZE as u64],
    );
    let patches = patch_projection_f32(
        &pixels,
        TABLE_TOKENS,
        3,
        14,
        &patch_weight,
        &patch_bias,
        HIDDEN_SIZE,
    )
    .unwrap();
    let input = add_interpolated_position_embedding_f32(
        &patches,
        HIDDEN_SIZE,
        &positions,
        27,
        27,
        &[[1, 30, 58]],
    )
    .unwrap();
    let mut input_hasher = blake3::Hasher::new();
    for value in &input {
        input_hasher.update(&value.to_le_bytes());
    }
    assert_eq!(
        input_hasher.finalize().to_hex().as_str(),
        "645e12596caffcd4b394202a1b790acbb51d242cf6f616c0bade5d012eece742"
    );

    let parameters = (0..27)
        .map(|layer| OwnedLayerParameters::load_layer(&model, layer))
        .collect::<Vec<_>>();
    let post_weight = load_tensor(
        &model,
        "visual.vision_model.post_layernorm.weight",
        &[HIDDEN_SIZE as u64],
    );
    let post_bias = load_tensor(
        &model,
        "visual.vision_model.post_layernorm.bias",
        &[HIDDEN_SIZE as u64],
    );
    let runtime_parameters = parameters
        .iter()
        .map(OwnedLayerParameters::runtime_borrowed)
        .collect::<Vec<_>>();
    let boundaries = [0_u32, TABLE_TOKENS as u32];
    let invocation = RuntimeStackInvocation {
        tokens: TABLE_TOKENS as u32,
        hidden_size: HIDDEN_SIZE as u32,
        attention_heads: HEADS as u32,
        head_dim: HEAD_DIM as u32,
        intermediate_size: INTERMEDIATE_SIZE as u32,
        layer_norm_epsilon: LAYER_NORM_EPSILON,
        input: &input,
        cu_seqlens: &boundaries,
        layer_parameters: &runtime_parameters,
        post_norm: RuntimeLayerNormParameters {
            weight: &post_weight,
            bias: &post_bias,
        },
    };
    let native = runtime
        .run_vision_encoder_stack_identity_rope(&invocation, &[])
        .unwrap();
    assert!(native.checkpoints.is_empty());
    assert_eq!(native.output.len(), TABLE_TOKENS * HIDDEN_SIZE);
    assert_eq!(native.diagnostics.readback_bytes, 8_017_920);
    let official = load_tensor(
        &stage,
        "vision.final",
        &[TABLE_TOKENS as u64, HIDDEN_SIZE as u64],
    );
    let report = compare_f32(
        &official,
        &native.output,
        &[TABLE_TOKENS, HIDDEN_SIZE],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap();
    let max_token = report
        .per_token_relative_l2
        .as_deref()
        .unwrap()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    eprintln!(
        "table L2 native-vs-official: max_abs={:.9} mean_abs={:.9} p99_abs={:.9} rel_l2={:.9} cosine={:.9} max_token_rel={:.9}",
        report.max_abs,
        report.mean_abs,
        report.p99_abs,
        report.relative_l2,
        report.cosine_similarity,
        max_token,
    );
    let table_policy = table_l2_final_vision_stack_policy();
    let verdict = report.assess(&table_policy).unwrap();
    assert!(verdict.passed(), "{report:#?}\n{verdict:#?}");
    let mut token_rotated = native.output.clone();
    token_rotated.rotate_left(HIDDEN_SIZE);
    let wrong = compare_f32(
        &official,
        &token_rotated,
        &[TABLE_TOKENS, HIDDEN_SIZE],
        ComparisonAxes {
            token_axis: Some(0),
            channel_axis: Some(1),
        },
    )
    .unwrap();
    assert!(
        !wrong.assess(&table_policy).unwrap().passed(),
        "table L2 policy accepted a one-token rotation: {wrong:#?}"
    );
}
