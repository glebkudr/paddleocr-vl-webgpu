use pvlc_cpu_ref::{
    CpuRefError, KvBlockOrder, LayerNormParameters as CpuNorm, LinearParameters as CpuLinear,
    VisionEncoderLayerConfig as CpuConfig, VisionEncoderLayerParameters as CpuParameters,
    VisionEncoderLayerTrace, vision_encoder_layer_identity_rope_f32,
};
use pvlc_runtime_core::{
    InvocationError, OwnedVisionEncoderLayerInvocation, OwnedVisionEncoderLayerParameters,
    OwnedVisionLayerNormParameters, OwnedVisionLinearParameters, VisionEncoderLayerStage,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ComparisonPolicy;

pub const M3_VISION_LAYER_TOKENS: u32 = 9;
pub const M3_VISION_LAYER_HIDDEN_SIZE: u32 = 18;
pub const M3_VISION_LAYER_ATTENTION_HEADS: u32 = 3;
pub const M3_VISION_LAYER_HEAD_DIM: u32 = 6;
pub const M3_VISION_LAYER_INTERMEDIATE_SIZE: u32 = 23;
pub const M3_VISION_LAYER_CU_SEQLENS: [u32; 4] = [0, 2, 5, 9];
pub const VISION_LAYER_FIXTURE_ALGORITHM: &str = "vision-layer-affine-mod257-binary-f32-v1";

const LAYER_NORM_EPSILON: f32 = 1.0e-6;
const FIXTURE_SEED: u32 = 401;
const POLICY: M3VisionLayerPolicy = M3VisionLayerPolicy {
    max_abs: 2.0e-4,
    max_mean_abs: 2.0e-5,
    max_p99_abs: 1.0e-4,
    max_relative_l2: 1.0e-4,
    min_cosine_similarity: 0.999_99,
    native_max_abs: 2.0e-4,
    native_max_relative_l2: 1.0e-4,
};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct M3VisionLayerPolicy {
    pub max_abs: f64,
    pub max_mean_abs: f64,
    pub max_p99_abs: f64,
    pub max_relative_l2: f64,
    pub min_cosine_similarity: f64,
    pub native_max_abs: f64,
    pub native_max_relative_l2: f64,
}

impl M3VisionLayerPolicy {
    #[must_use]
    pub const fn comparison_policy(self) -> ComparisonPolicy {
        ComparisonPolicy {
            require_finite: true,
            max_abs: self.max_abs,
            max_mean_abs: self.max_mean_abs,
            max_p99_abs: self.max_p99_abs,
            max_relative_l2: self.max_relative_l2,
            min_cosine_similarity: self.min_cosine_similarity,
            max_per_token_relative_l2: None,
            max_per_channel_relative_l2: None,
        }
    }

    #[must_use]
    pub const fn native_comparison_policy(self) -> ComparisonPolicy {
        ComparisonPolicy {
            require_finite: true,
            max_abs: self.native_max_abs,
            max_mean_abs: self.native_max_abs,
            max_p99_abs: self.native_max_abs,
            max_relative_l2: self.native_max_relative_l2,
            min_cosine_similarity: self.min_cosine_similarity,
            max_per_token_relative_l2: None,
            max_per_channel_relative_l2: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct M3VisionLayerCheckpoints {
    pub norm1: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub attention_context: Vec<f32>,
    pub attention_output: Vec<f32>,
    pub attention_residual: Vec<f32>,
    pub norm2: Vec<f32>,
    pub mlp_fc1: Vec<f32>,
    pub mlp_activation: Vec<f32>,
    pub mlp_output: Vec<f32>,
    pub output: Vec<f32>,
}

impl M3VisionLayerCheckpoints {
    #[must_use]
    pub fn stage(&self, stage: VisionEncoderLayerStage) -> &[f32] {
        match stage {
            VisionEncoderLayerStage::Norm1 => &self.norm1,
            VisionEncoderLayerStage::Query => &self.query,
            VisionEncoderLayerStage::Key => &self.key,
            VisionEncoderLayerStage::Value => &self.value,
            VisionEncoderLayerStage::AttentionContext => &self.attention_context,
            VisionEncoderLayerStage::AttentionOutput => &self.attention_output,
            VisionEncoderLayerStage::AttentionResidual => &self.attention_residual,
            VisionEncoderLayerStage::Norm2 => &self.norm2,
            VisionEncoderLayerStage::MlpFc1 => &self.mlp_fc1,
            VisionEncoderLayerStage::MlpActivation => &self.mlp_activation,
            VisionEncoderLayerStage::MlpOutput => &self.mlp_output,
            VisionEncoderLayerStage::Output => &self.output,
        }
    }
}

impl From<VisionEncoderLayerTrace> for M3VisionLayerCheckpoints {
    fn from(trace: VisionEncoderLayerTrace) -> Self {
        Self {
            norm1: trace.norm1,
            query: trace.query,
            key: trace.key,
            value: trace.value,
            attention_context: trace.attention_context,
            attention_output: trace.attention_output,
            attention_residual: trace.attention_residual,
            norm2: trace.norm2,
            mlp_fc1: trace.mlp_fc1,
            mlp_activation: trace.mlp_activation,
            mlp_output: trace.mlp_output,
            output: trace.output,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct M3VisionLayerCase {
    pub id: String,
    pub tags: Vec<String>,
    pub tokens: u32,
    pub hidden_size: u32,
    pub attention_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub layer_norm_epsilon: f32,
    pub cu_seqlens: Vec<u32>,
    pub seed: u32,
    pub poisoned_segment: Option<u32>,
    pub expected: M3VisionLayerCheckpoints,
    pub policy: M3VisionLayerPolicy,
}

impl M3VisionLayerCase {
    pub fn invocation(
        &self,
    ) -> Result<OwnedVisionEncoderLayerInvocation, M3VisionLayerCorpusError> {
        let hidden = usize::try_from(self.hidden_size)
            .map_err(|_| M3VisionLayerCorpusError::ArithmeticOverflow)?;
        let tokens = usize::try_from(self.tokens)
            .map_err(|_| M3VisionLayerCorpusError::ArithmeticOverflow)?;
        let intermediate = usize::try_from(self.intermediate_size)
            .map_err(|_| M3VisionLayerCorpusError::ArithmeticOverflow)?;
        let hidden_elements = tokens
            .checked_mul(hidden)
            .ok_or(M3VisionLayerCorpusError::ArithmeticOverflow)?;
        let mut input = affine_values(hidden_elements, self.seed, 17, 19, 3, 64.0, 0.0);
        if let Some(segment) = self.poisoned_segment {
            let segment = usize::try_from(segment)
                .map_err(|_| M3VisionLayerCorpusError::InvalidPoisonedSegment)?;
            if segment + 1 >= self.cu_seqlens.len() {
                return Err(M3VisionLayerCorpusError::InvalidPoisonedSegment);
            }
            let start = self.cu_seqlens[segment] as usize * hidden;
            let end = self.cu_seqlens[segment + 1] as usize * hidden;
            if end > input.len() {
                return Err(M3VisionLayerCorpusError::InvalidPoisonedSegment);
            }
            for value in &mut input[start..end] {
                *value = *value * -30.0 + 8.843_75;
            }
        }

        let parameters = OwnedVisionEncoderLayerParameters {
            norm1: OwnedVisionLayerNormParameters {
                weight: affine_values(hidden, self.seed, 7, 5, 11, 512.0, 1.0),
                bias: affine_values(hidden, self.seed, 11, 7, 13, 1_024.0, 0.0),
            },
            query: linear(hidden, hidden, self.seed, (17, 11, 17), (19, 13, 19)),
            key: linear(hidden, hidden, self.seed, (23, 17, 23), (29, 19, 29)),
            value: linear(hidden, hidden, self.seed, (31, 23, 31), (37, 29, 37)),
            attention_output: linear(hidden, hidden, self.seed, (41, 31, 41), (43, 37, 43)),
            norm2: OwnedVisionLayerNormParameters {
                weight: affine_values(hidden, self.seed, 47, 41, 47, 512.0, 1.0),
                bias: affine_values(hidden, self.seed, 53, 43, 53, 1_024.0, 0.0),
            },
            mlp_fc1: linear(hidden, intermediate, self.seed, (59, 47, 59), (61, 53, 61)),
            mlp_fc2: linear(intermediate, hidden, self.seed, (67, 59, 67), (71, 61, 71)),
        };
        let invocation = OwnedVisionEncoderLayerInvocation {
            tokens: self.tokens,
            hidden_size: self.hidden_size,
            attention_heads: self.attention_heads,
            head_dim: self.head_dim,
            intermediate_size: self.intermediate_size,
            layer_norm_epsilon: self.layer_norm_epsilon,
            input,
            cu_seqlens: self.cu_seqlens.clone(),
            parameters,
        };
        invocation.borrowed().plan()?;
        Ok(invocation)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct M3VisionLayerCorpus {
    pub schema_version: u32,
    pub oracle: String,
    pub fixture_algorithm: String,
    pub cases: Vec<M3VisionLayerCase>,
}

#[derive(Debug, Error)]
pub enum M3VisionLayerCorpusError {
    #[error("M3 vision-layer invocation is invalid: {0}")]
    Invocation(#[from] InvocationError),
    #[error("M3 vision-layer CPU oracle failed: {0}")]
    Cpu(#[from] CpuRefError),
    #[error("M3 vision-layer fixture arithmetic overflowed")]
    ArithmeticOverflow,
    #[error("M3 vision-layer fixture names an invalid poisoned segment")]
    InvalidPoisonedSegment,
}

pub fn m3_vision_layer_corpus() -> Result<M3VisionLayerCorpus, M3VisionLayerCorpusError> {
    let mut cases = Vec::with_capacity(4);
    cases.push(build_case(
        "vision_encoder_layer_identity_rope/baseline".to_owned(),
        vec!["baseline".to_owned(), "packed-segments".to_owned()],
        None,
    )?);
    for segment in 0..3_u32 {
        cases.push(build_case(
            format!("vision_encoder_layer_identity_rope/poison-segment-{segment}"),
            vec![
                "packed-segments".to_owned(),
                format!("poison-segment:{segment}"),
            ],
            Some(segment),
        )?);
    }
    Ok(M3VisionLayerCorpus {
        schema_version: 1,
        oracle: "pvlc-cpu-ref/vision-encoder-layer-identity-rope-f32-v1".to_owned(),
        fixture_algorithm: VISION_LAYER_FIXTURE_ALGORITHM.to_owned(),
        cases,
    })
}

fn build_case(
    id: String,
    mut tags: Vec<String>,
    poisoned_segment: Option<u32>,
) -> Result<M3VisionLayerCase, M3VisionLayerCorpusError> {
    tags.sort();
    tags.dedup();
    let mut case = M3VisionLayerCase {
        id,
        tags,
        tokens: M3_VISION_LAYER_TOKENS,
        hidden_size: M3_VISION_LAYER_HIDDEN_SIZE,
        attention_heads: M3_VISION_LAYER_ATTENTION_HEADS,
        head_dim: M3_VISION_LAYER_HEAD_DIM,
        intermediate_size: M3_VISION_LAYER_INTERMEDIATE_SIZE,
        layer_norm_epsilon: LAYER_NORM_EPSILON,
        cu_seqlens: M3_VISION_LAYER_CU_SEQLENS.to_vec(),
        seed: FIXTURE_SEED,
        poisoned_segment,
        expected: empty_checkpoints(),
        policy: POLICY,
    };
    let invocation = case.invocation()?;
    case.expected = cpu_trace(&invocation)?.into();
    Ok(case)
}

fn cpu_trace(
    invocation: &OwnedVisionEncoderLayerInvocation,
) -> Result<VisionEncoderLayerTrace, CpuRefError> {
    let parameters = &invocation.parameters;
    let boundaries = invocation
        .cu_seqlens
        .iter()
        .map(|value| *value as usize)
        .collect::<Vec<_>>();
    vision_encoder_layer_identity_rope_f32(
        &invocation.input,
        CpuConfig {
            tokens: invocation.tokens as usize,
            hidden_size: invocation.hidden_size as usize,
            attention_heads: invocation.attention_heads as usize,
            head_dim: invocation.head_dim as usize,
            intermediate_size: invocation.intermediate_size as usize,
            layer_norm_epsilon: invocation.layer_norm_epsilon,
            attention_key_tile: 4,
            attention_order: KvBlockOrder::Forward,
        },
        &boundaries,
        CpuParameters {
            norm1: CpuNorm {
                weight: &parameters.norm1.weight,
                bias: &parameters.norm1.bias,
            },
            query: cpu_linear(&parameters.query),
            key: cpu_linear(&parameters.key),
            value: cpu_linear(&parameters.value),
            attention_output: cpu_linear(&parameters.attention_output),
            norm2: CpuNorm {
                weight: &parameters.norm2.weight,
                bias: &parameters.norm2.bias,
            },
            mlp_fc1: cpu_linear(&parameters.mlp_fc1),
            mlp_fc2: cpu_linear(&parameters.mlp_fc2),
        },
    )
}

fn cpu_linear(parameters: &OwnedVisionLinearParameters) -> CpuLinear<'_> {
    CpuLinear {
        weight: &parameters.weight,
        bias: &parameters.bias,
    }
}

fn linear(
    input_width: usize,
    output_width: usize,
    seed: u32,
    weight: (u64, u64, u64),
    bias: (u64, u64, u64),
) -> OwnedVisionLinearParameters {
    OwnedVisionLinearParameters {
        weight: affine_values(
            input_width * output_width,
            seed,
            weight.0,
            weight.1,
            weight.2,
            1_024.0,
            0.0,
        ),
        bias: affine_values(output_width, seed, bias.0, bias.1, bias.2, 512.0, 0.0),
    }
}

#[allow(clippy::too_many_arguments)]
fn affine_values(
    elements: usize,
    seed: u32,
    index_multiplier: u64,
    seed_multiplier: u64,
    offset: u64,
    divisor: f32,
    shift: f32,
) -> Vec<f32> {
    let seed_term = u64::from(seed) * seed_multiplier + offset;
    (0..elements)
        .map(|index| {
            let residue = ((index as u64 * index_multiplier + seed_term) % 257) as i32;
            (residue - 128) as f32 / divisor + shift
        })
        .collect()
}

fn empty_checkpoints() -> M3VisionLayerCheckpoints {
    M3VisionLayerCheckpoints {
        norm1: Vec::new(),
        query: Vec::new(),
        key: Vec::new(),
        value: Vec::new(),
        attention_context: Vec::new(),
        attention_output: Vec::new(),
        attention_residual: Vec::new(),
        norm2: Vec::new(),
        mlp_fc1: Vec::new(),
        mlp_activation: Vec::new(),
        mlp_output: Vec::new(),
        output: Vec::new(),
    }
}
