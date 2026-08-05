//! Adapter-neutral protocol for the fixed PaddleOCR-VL FP32 primitive layer.

use std::{error::Error, fmt};

use pvlc_memory::{
    ArenaConfig, ArenaError, ArenaErrorCode, TensorLifetime, plan_static_arena,
    verify_static_arena_plan,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum KernelId {
    #[serde(rename = "gemm_f32")]
    GemmF32,
    #[serde(rename = "gemv_f32")]
    GemvF32,
    #[serde(rename = "layer_norm_f32")]
    LayerNormF32,
    #[serde(rename = "rms_norm_f32")]
    RmsNormF32,
    #[serde(rename = "silu_f32")]
    SiluF32,
    #[serde(rename = "gelu_tanh_f32")]
    GeluTanhF32,
    #[serde(rename = "gelu_erf_f32")]
    GeluErfF32,
    #[serde(rename = "rope_neox_f32")]
    RopeNeoxF32,
    #[serde(rename = "vision_attention_f32")]
    VisionAttentionF32,
    #[serde(rename = "vision_patch_projection_f32")]
    VisionPatchProjectionF32,
    #[serde(rename = "projector_merge_2x2_f32")]
    ProjectorMerge2x2F32,
    #[serde(rename = "add_f32")]
    AddF32,
    #[serde(rename = "vision_qkv_fused_f32")]
    VisionQkvFusedF32,
    #[serde(rename = "decoder_kv_append_f32")]
    DecoderKvAppendF32,
    #[serde(rename = "decoder_gqa_f32")]
    DecoderGqaF32,
    #[serde(rename = "decoder_gqa_split_partial_f32")]
    DecoderGqaSplitPartialF32,
    #[serde(rename = "decoder_gqa_split_merge_f32")]
    DecoderGqaSplitMergeF32,
    #[serde(rename = "decoder_mrope_f32")]
    DecoderMropeF32,
    #[serde(rename = "decoder_swiglu_f32")]
    DecoderSwigluF32,
    #[serde(rename = "decoder_prefill_gqa_f32")]
    DecoderPrefillGqaF32,
    #[serde(rename = "decoder_prefill_mrope_f32")]
    DecoderPrefillMropeF32,
    #[serde(rename = "decoder_kv_append_range_f32")]
    DecoderKvAppendRangeF32,
    #[serde(rename = "gemv_tiled_f32")]
    GemvTiledF32,
    #[serde(rename = "rms_norm_f16_weights")]
    RmsNormF16Weights,
    #[serde(rename = "gemv_tiled_f16_weights")]
    GemvTiledF16Weights,
    #[serde(rename = "linear_projection_f16_weights")]
    LinearProjectionF16Weights,
    #[serde(rename = "vision_rope_2d_f32")]
    VisionRope2dF32,
    #[serde(rename = "vision_qkv_fused_f16_weights")]
    VisionQkvFusedF16Weights,
    #[serde(rename = "layer_norm_f16")]
    LayerNormF16,
    #[serde(rename = "linear_projection_f16")]
    LinearProjectionF16,
    #[serde(rename = "vision_attention_f16")]
    VisionAttentionF16,
    #[serde(rename = "add_f16")]
    AddF16,
    #[serde(rename = "gelu_tanh_f16")]
    GeluTanhF16,
    #[serde(rename = "vision_rope_2d_f16")]
    VisionRope2dF16,
    #[serde(rename = "projector_merge_2x2_f16")]
    ProjectorMerge2x2F16,
    #[serde(rename = "gelu_erf_f16")]
    GeluErfF16,
}

impl KernelId {
    pub const M2_PRIMITIVES: [Self; 7] = [
        Self::GemmF32,
        Self::GemvF32,
        Self::LayerNormF32,
        Self::RmsNormF32,
        Self::SiluF32,
        Self::GeluTanhF32,
        Self::RopeNeoxF32,
    ];
    pub const ALL: [Self; 36] = [
        Self::GemmF32,
        Self::GemvF32,
        Self::LayerNormF32,
        Self::RmsNormF32,
        Self::SiluF32,
        Self::GeluTanhF32,
        Self::RopeNeoxF32,
        Self::VisionAttentionF32,
        Self::VisionPatchProjectionF32,
        Self::AddF32,
        Self::GeluErfF32,
        Self::ProjectorMerge2x2F32,
        Self::VisionQkvFusedF32,
        Self::DecoderKvAppendF32,
        Self::DecoderGqaF32,
        Self::DecoderGqaSplitPartialF32,
        Self::DecoderGqaSplitMergeF32,
        Self::DecoderMropeF32,
        Self::DecoderSwigluF32,
        Self::DecoderPrefillGqaF32,
        Self::DecoderPrefillMropeF32,
        Self::DecoderKvAppendRangeF32,
        Self::GemvTiledF32,
        Self::RmsNormF16Weights,
        Self::GemvTiledF16Weights,
        Self::LinearProjectionF16Weights,
        Self::VisionRope2dF32,
        Self::VisionQkvFusedF16Weights,
        Self::LayerNormF16,
        Self::LinearProjectionF16,
        Self::VisionAttentionF16,
        Self::AddF16,
        Self::GeluTanhF16,
        Self::VisionRope2dF16,
        Self::ProjectorMerge2x2F16,
        Self::GeluErfF16,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GemmF32 => "gemm_f32",
            Self::GemvF32 => "gemv_f32",
            Self::LayerNormF32 => "layer_norm_f32",
            Self::RmsNormF32 => "rms_norm_f32",
            Self::SiluF32 => "silu_f32",
            Self::GeluTanhF32 => "gelu_tanh_f32",
            Self::GeluErfF32 => "gelu_erf_f32",
            Self::RopeNeoxF32 => "rope_neox_f32",
            Self::VisionAttentionF32 => "vision_attention_f32",
            Self::VisionPatchProjectionF32 => "vision_patch_projection_f32",
            Self::ProjectorMerge2x2F32 => "projector_merge_2x2_f32",
            Self::AddF32 => "add_f32",
            Self::VisionQkvFusedF32 => "vision_qkv_fused_f32",
            Self::DecoderKvAppendF32 => "decoder_kv_append_f32",
            Self::DecoderGqaF32 => "decoder_gqa_f32",
            Self::DecoderGqaSplitPartialF32 => "decoder_gqa_split_partial_f32",
            Self::DecoderGqaSplitMergeF32 => "decoder_gqa_split_merge_f32",
            Self::DecoderMropeF32 => "decoder_mrope_f32",
            Self::DecoderSwigluF32 => "decoder_swiglu_f32",
            Self::DecoderPrefillGqaF32 => "decoder_prefill_gqa_f32",
            Self::DecoderPrefillMropeF32 => "decoder_prefill_mrope_f32",
            Self::DecoderKvAppendRangeF32 => "decoder_kv_append_range_f32",
            Self::GemvTiledF32 => "gemv_tiled_f32",
            Self::RmsNormF16Weights => "rms_norm_f16_weights",
            Self::GemvTiledF16Weights => "gemv_tiled_f16_weights",
            Self::LinearProjectionF16Weights => "linear_projection_f16_weights",
            Self::VisionRope2dF32 => "vision_rope_2d_f32",
            Self::VisionQkvFusedF16Weights => "vision_qkv_fused_f16_weights",
            Self::LayerNormF16 => "layer_norm_f16",
            Self::LinearProjectionF16 => "linear_projection_f16",
            Self::VisionAttentionF16 => "vision_attention_f16",
            Self::AddF16 => "add_f16",
            Self::GeluTanhF16 => "gelu_tanh_f16",
            Self::VisionRope2dF16 => "vision_rope_2d_f16",
            Self::ProjectorMerge2x2F16 => "projector_merge_2x2_f16",
            Self::GeluErfF16 => "gelu_erf_f16",
        }
    }
}

pub const MAX_VISION_HEAD_DIM: u32 = 72;
pub const MAX_DECODER_HEAD_DIM: u32 = 128;
pub const LINEAR_PROJECTION_TILE: u32 = 32;
pub const VISION_QKV_FUSED_TILE: u32 = 8;
pub const VISION_QKV_FUSED_STORAGE_BINDING_COUNT: u32 = 8;
pub const VISION_QKV_CANARY_U32: u32 = 0x7fc0_51a7;
const MAX_WEBGPU_DISPATCH_DIMENSION: u32 = 65_535;
const VISION_ATTENTION_QUERY_TILE: u32 = 128;
const VISION_ATTENTION_WORKGROUP_SIZE: u32 = 128;
const VISION_QKV_FUSED_F16_WEIGHT_ROW_TILE: u32 = 16;
const VISION_QKV_FUSED_F16_WEIGHT_COLUMN_TILE: u32 = 32;

impl fmt::Display for KernelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kernel", deny_unknown_fields)]
pub enum KernelInvocation {
    #[serde(rename = "gemm_f32")]
    GemmF32 {
        rows: u32,
        inner: u32,
        columns: u32,
        left: Vec<f32>,
        right: Vec<f32>,
    },
    #[serde(rename = "gemv_f32")]
    GemvF32 {
        rows: u32,
        columns: u32,
        matrix: Vec<f32>,
        vector: Vec<f32>,
    },
    #[serde(rename = "layer_norm_f32")]
    LayerNormF32 {
        rows: u32,
        width: u32,
        input: Vec<f32>,
        weight: Vec<f32>,
        bias: Vec<f32>,
        epsilon: f32,
    },
    #[serde(rename = "rms_norm_f32")]
    RmsNormF32 {
        rows: u32,
        width: u32,
        input: Vec<f32>,
        weight: Vec<f32>,
        epsilon: f32,
    },
    #[serde(rename = "silu_f32")]
    SiluF32 { values: Vec<f32> },
    #[serde(rename = "gelu_tanh_f32")]
    GeluTanhF32 { values: Vec<f32> },
    #[serde(rename = "gelu_erf_f32")]
    GeluErfF32 { values: Vec<f32> },
    #[serde(rename = "rope_neox_f32")]
    RopeNeoxF32 {
        rows: u32,
        width: u32,
        rotary_dim: u32,
        positions: Vec<u32>,
        base: f32,
        values: Vec<f32>,
    },
    #[serde(rename = "vision_attention_f32")]
    VisionAttentionF32 {
        tokens: u32,
        heads: u32,
        head_dim: u32,
        query: Vec<f32>,
        key: Vec<f32>,
        value: Vec<f32>,
        cu_seqlens: Vec<u32>,
    },
    #[serde(rename = "vision_patch_projection_f32")]
    VisionPatchProjectionF32 {
        patch_count: u32,
        input_width: u32,
        output_width: u32,
        input: Vec<f32>,
        weight: Vec<f32>,
        bias: Vec<f32>,
    },
    #[serde(rename = "projector_merge_2x2_f32")]
    ProjectorMerge2x2F32 {
        output_tokens: u32,
        hidden_size: u32,
        input: Vec<f32>,
        source_token_indices: Vec<u32>,
    },
    #[serde(rename = "add_f32")]
    AddF32 { left: Vec<f32>, right: Vec<f32> },
}

impl KernelInvocation {
    #[must_use]
    pub const fn kernel_id(&self) -> KernelId {
        match self {
            Self::GemmF32 { .. } => KernelId::GemmF32,
            Self::GemvF32 { .. } => KernelId::GemvF32,
            Self::LayerNormF32 { .. } => KernelId::LayerNormF32,
            Self::RmsNormF32 { .. } => KernelId::RmsNormF32,
            Self::SiluF32 { .. } => KernelId::SiluF32,
            Self::GeluTanhF32 { .. } => KernelId::GeluTanhF32,
            Self::GeluErfF32 { .. } => KernelId::GeluErfF32,
            Self::RopeNeoxF32 { .. } => KernelId::RopeNeoxF32,
            Self::VisionAttentionF32 { .. } => KernelId::VisionAttentionF32,
            Self::VisionPatchProjectionF32 { .. } => KernelId::VisionPatchProjectionF32,
            Self::ProjectorMerge2x2F32 { .. } => KernelId::ProjectorMerge2x2F32,
            Self::AddF32 { .. } => KernelId::AddF32,
        }
    }

    pub fn plan(&self) -> Result<InvocationPlan, InvocationError> {
        match self {
            Self::GemmF32 {
                rows,
                inner,
                columns,
                left,
                right,
            } => {
                require_dimensions(&[*rows, *inner, *columns])?;
                let output_elements = checked_elements(*rows, *columns)?;
                let output_bytes = checked_output_bytes(output_elements)?;
                require_len(left.len(), checked_elements(*rows, *inner)?)?;
                require_len(right.len(), checked_elements(*inner, *columns)?)?;
                require_finite(left)?;
                require_finite(right)?;
                plan(
                    KernelId::GemmF32,
                    output_elements,
                    output_bytes,
                    [8, 8, 1],
                    [ceil_div(*columns, 8), ceil_div(*rows, 8), 1],
                )
            }
            Self::GemvF32 {
                rows,
                columns,
                matrix,
                vector,
            } => {
                require_dimensions(&[*rows, *columns])?;
                let output_elements = u64::from(*rows);
                let output_bytes = checked_output_bytes(output_elements)?;
                require_len(matrix.len(), checked_elements(*rows, *columns)?)?;
                require_len(vector.len(), u64::from(*columns))?;
                require_finite(matrix)?;
                require_finite(vector)?;
                plan(
                    KernelId::GemvF32,
                    output_elements,
                    output_bytes,
                    [64, 1, 1],
                    [ceil_div(*rows, 64), 1, 1],
                )
            }
            Self::LayerNormF32 {
                rows,
                width,
                input,
                weight,
                bias,
                epsilon,
            } => {
                require_dimensions(&[*rows, *width])?;
                require_epsilon(*epsilon)?;
                let output_elements = checked_elements(*rows, *width)?;
                let output_bytes = checked_output_bytes(output_elements)?;
                require_len(input.len(), output_elements)?;
                require_len(weight.len(), u64::from(*width))?;
                require_len(bias.len(), u64::from(*width))?;
                require_finite(input)?;
                require_finite(weight)?;
                require_finite(bias)?;
                plan(
                    KernelId::LayerNormF32,
                    output_elements,
                    output_bytes,
                    [64, 1, 1],
                    [ceil_div(*rows, 64), 1, 1],
                )
            }
            Self::RmsNormF32 {
                rows,
                width,
                input,
                weight,
                epsilon,
            } => {
                require_dimensions(&[*rows, *width])?;
                require_epsilon(*epsilon)?;
                let output_elements = checked_elements(*rows, *width)?;
                let output_bytes = checked_output_bytes(output_elements)?;
                require_len(input.len(), output_elements)?;
                require_len(weight.len(), u64::from(*width))?;
                require_finite(input)?;
                require_finite(weight)?;
                plan(
                    KernelId::RmsNormF32,
                    output_elements,
                    output_bytes,
                    [64, 1, 1],
                    [ceil_div(*rows, 64), 1, 1],
                )
            }
            Self::SiluF32 { values }
            | Self::GeluTanhF32 { values }
            | Self::GeluErfF32 { values } => {
                if values.is_empty() {
                    return Err(InvocationError::new(
                        InvocationErrorCode::ZeroDimension,
                        "activation input must not be empty",
                    ));
                }
                let output_elements = u64::try_from(values.len()).map_err(|_| overflow())?;
                let output_bytes = checked_output_bytes(output_elements)?;
                let element_count = u32::try_from(values.len()).map_err(|_| overflow())?;
                require_finite(values)?;
                let kernel = self.kernel_id();
                let (dispatch, _) = bounded_linear_dispatch(element_count, 64);
                plan(kernel, output_elements, output_bytes, [64, 1, 1], dispatch)
            }
            Self::RopeNeoxF32 {
                rows,
                width,
                rotary_dim,
                positions,
                base,
                values,
            } => {
                require_dimensions(&[*rows, *width])?;
                if *rotary_dim == 0 || *rotary_dim > *width || !rotary_dim.is_multiple_of(2) {
                    return Err(InvocationError::new(
                        InvocationErrorCode::InvalidRotaryDimension,
                        "rotary_dim must be nonzero, even, and no wider than width",
                    ));
                }
                if !base.is_finite() || *base <= 0.0 {
                    return Err(InvocationError::new(
                        InvocationErrorCode::InvalidRopeBase,
                        "RoPE base must be positive and finite",
                    ));
                }
                let output_elements = checked_elements(*rows, *width)?;
                let output_bytes = checked_output_bytes(output_elements)?;
                require_len(values.len(), output_elements)?;
                require_len(positions.len(), u64::from(*rows))?;
                require_finite(values)?;
                let work_items = u64::from(*rows) * u64::from(*rotary_dim / 2);
                let work_items = u32::try_from(work_items).map_err(|_| overflow())?;
                plan(
                    KernelId::RopeNeoxF32,
                    output_elements,
                    output_bytes,
                    [64, 1, 1],
                    [ceil_div(work_items, 64), 1, 1],
                )
            }
            Self::VisionAttentionF32 {
                tokens,
                heads,
                head_dim,
                query,
                key,
                value,
                cu_seqlens,
            } => {
                require_dimensions(&[*tokens, *heads, *head_dim])?;
                if *head_dim > MAX_VISION_HEAD_DIM {
                    return Err(InvocationError::new(
                        InvocationErrorCode::UnsupportedHeadDimension,
                        format!(
                            "vision attention head_dim {head_dim} exceeds the fixed limit {MAX_VISION_HEAD_DIM}"
                        ),
                    ));
                }
                require_sequence_boundaries(cu_seqlens, *tokens)?;
                let output_elements = checked_tensor_elements(&[*tokens, *heads, *head_dim])?;
                let output_bytes = checked_output_bytes(output_elements)?;
                require_len(query.len(), output_elements)?;
                require_len(key.len(), output_elements)?;
                require_len(value.len(), output_elements)?;
                require_finite(query)?;
                require_finite(key)?;
                require_finite(value)?;
                u32::try_from(cu_seqlens.len() - 1).map_err(|_| overflow())?;
                plan(
                    KernelId::VisionAttentionF32,
                    output_elements,
                    output_bytes,
                    [VISION_ATTENTION_WORKGROUP_SIZE, 1, 1],
                    [ceil_div(*tokens, VISION_ATTENTION_QUERY_TILE), *heads, 1],
                )
            }
            Self::VisionPatchProjectionF32 {
                patch_count,
                input_width,
                output_width,
                input,
                weight,
                bias,
            } => {
                require_dimensions(&[*patch_count, *input_width, *output_width])?;
                let output_elements = checked_elements(*patch_count, *output_width)?;
                let output_bytes = checked_output_bytes(output_elements)?;
                require_len(input.len(), checked_elements(*patch_count, *input_width)?)?;
                require_len(weight.len(), checked_elements(*output_width, *input_width)?)?;
                require_len(bias.len(), u64::from(*output_width))?;
                require_finite(input)?;
                require_finite(weight)?;
                require_finite(bias)?;
                plan(
                    KernelId::VisionPatchProjectionF32,
                    output_elements,
                    output_bytes,
                    [8, 8, 1],
                    [
                        ceil_div(*output_width, LINEAR_PROJECTION_TILE),
                        ceil_div(*patch_count, LINEAR_PROJECTION_TILE),
                        1,
                    ],
                )
            }
            Self::ProjectorMerge2x2F32 {
                output_tokens,
                hidden_size,
                input,
                source_token_indices,
            } => {
                require_dimensions(&[*output_tokens, *hidden_size])?;
                let input_tokens = output_tokens.checked_mul(4).ok_or_else(overflow)?;
                let input_elements = checked_elements(input_tokens, *hidden_size)?;
                require_len(input.len(), input_elements)?;
                require_finite(input)?;
                require_projector_permutation(source_token_indices, input_tokens)?;

                let merged_width = hidden_size.checked_mul(4).ok_or_else(overflow)?;
                let output_elements = checked_elements(*output_tokens, merged_width)?;
                let output_bytes = checked_output_bytes(output_elements)?;
                let work_items = u32::try_from(output_elements).map_err(|_| overflow())?;
                let (dispatch, _) = bounded_linear_dispatch(work_items, 64);
                plan(
                    KernelId::ProjectorMerge2x2F32,
                    output_elements,
                    output_bytes,
                    [64, 1, 1],
                    dispatch,
                )
            }
            Self::AddF32 { left, right } => {
                if left.is_empty() && right.is_empty() {
                    return Err(InvocationError::new(
                        InvocationErrorCode::ZeroDimension,
                        "add operands must not be empty",
                    ));
                }
                let output_elements = u64::try_from(left.len()).map_err(|_| overflow())?;
                require_len(right.len(), output_elements)?;
                let output_bytes = checked_output_bytes(output_elements)?;
                let element_count = u32::try_from(left.len()).map_err(|_| overflow())?;
                require_finite(left)?;
                require_finite(right)?;
                plan(
                    KernelId::AddF32,
                    output_elements,
                    output_bytes,
                    [64, 1, 1],
                    [ceil_div(element_count, 64), 1, 1],
                )
            }
        }
    }

    pub fn uniform_bytes(&self) -> Result<Vec<u8>, InvocationError> {
        self.plan()?;
        let words = match self {
            Self::GemmF32 {
                rows,
                inner,
                columns,
                ..
            } => [*rows, *inner, *columns, 0],
            Self::GemvF32 { rows, columns, .. } => [*rows, *columns, 0, 0],
            Self::LayerNormF32 {
                rows,
                width,
                epsilon,
                ..
            }
            | Self::RmsNormF32 {
                rows,
                width,
                epsilon,
                ..
            } => [*rows, *width, epsilon.to_bits(), 0],
            Self::SiluF32 { values }
            | Self::GeluTanhF32 { values }
            | Self::GeluErfF32 { values } => {
                let elements = u32::try_from(values.len()).map_err(|_| overflow())?;
                let (_, row_stride) = bounded_linear_dispatch(elements, 64);
                [elements, row_stride, 0, 0]
            }
            Self::RopeNeoxF32 {
                rows,
                width,
                rotary_dim,
                base,
                ..
            } => [*rows, *width, *rotary_dim, base.to_bits()],
            Self::VisionAttentionF32 {
                tokens,
                heads,
                head_dim,
                cu_seqlens,
                ..
            } => [
                *tokens,
                *heads,
                *head_dim,
                u32::try_from(cu_seqlens.len() - 1).map_err(|_| overflow())?,
            ],
            Self::VisionPatchProjectionF32 {
                patch_count,
                input_width,
                output_width,
                ..
            } => [*patch_count, *input_width, *output_width, 0],
            Self::ProjectorMerge2x2F32 {
                output_tokens,
                hidden_size,
                ..
            } => {
                let merged_width = hidden_size.checked_mul(4).ok_or_else(overflow)?;
                let elements = u32::try_from(checked_elements(*output_tokens, merged_width)?)
                    .map_err(|_| overflow())?;
                let (_, row_stride) = bounded_linear_dispatch(elements, 64);
                [*output_tokens, *hidden_size, elements, row_stride]
            }
            Self::AddF32 { left, .. } => {
                [u32::try_from(left.len()).map_err(|_| overflow())?, 0, 0, 0]
            }
        };
        Ok(words.into_iter().flat_map(u32::to_le_bytes).collect())
    }

    #[must_use]
    pub fn inputs(&self) -> Vec<InvocationInput<'_>> {
        match self {
            Self::GemmF32 { left, right, .. } => {
                vec![InvocationInput::F32(left), InvocationInput::F32(right)]
            }
            Self::GemvF32 { matrix, vector, .. } => {
                vec![InvocationInput::F32(matrix), InvocationInput::F32(vector)]
            }
            Self::LayerNormF32 {
                input,
                weight,
                bias,
                ..
            } => vec![
                InvocationInput::F32(input),
                InvocationInput::F32(weight),
                InvocationInput::F32(bias),
            ],
            Self::RmsNormF32 { input, weight, .. } => {
                vec![InvocationInput::F32(input), InvocationInput::F32(weight)]
            }
            Self::SiluF32 { values }
            | Self::GeluTanhF32 { values }
            | Self::GeluErfF32 { values } => {
                vec![InvocationInput::F32(values)]
            }
            Self::RopeNeoxF32 {
                values, positions, ..
            } => vec![
                InvocationInput::F32(values),
                InvocationInput::U32(positions),
            ],
            Self::VisionAttentionF32 {
                query,
                key,
                value,
                cu_seqlens,
                ..
            } => vec![
                InvocationInput::F32(query),
                InvocationInput::F32(key),
                InvocationInput::F32(value),
                InvocationInput::U32(cu_seqlens),
            ],
            Self::VisionPatchProjectionF32 {
                input,
                weight,
                bias,
                ..
            } => vec![
                InvocationInput::F32(input),
                InvocationInput::F32(weight),
                InvocationInput::F32(bias),
            ],
            Self::ProjectorMerge2x2F32 {
                input,
                source_token_indices,
                ..
            } => vec![
                InvocationInput::F32(input),
                InvocationInput::U32(source_token_indices),
            ],
            Self::AddF32 { left, right } => {
                vec![InvocationInput::F32(left), InvocationInput::F32(right)]
            }
        }
    }

    #[must_use]
    pub fn output_initializer(&self) -> Option<&[f32]> {
        match self {
            Self::RopeNeoxF32 { values, .. } => Some(values),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InvocationInput<'a> {
    F32(&'a [f32]),
    U32(&'a [u32]),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InvocationPlan {
    pub kernel: KernelId,
    pub output_elements: usize,
    pub output_bytes: u64,
    pub workgroup_size: [u32; 3],
    pub dispatch: [u32; 3],
}

/// Geometry of the cooperative decoder GEMV. The packed-`vec4` shader views
/// the same f32 bytes as the scalar GEMV and is intentionally admitted only
/// for the three decoder input widths covered by its fixed shared-memory
/// lattice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GemvTiledDescriptor {
    pub rows: u32,
    pub columns: u32,
}

/// Fully determined cooperative decoder GEMV plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GemvTiledPlan {
    pub tile_rows: u32,
    pub threads_per_row: u32,
    pub vector_width: u32,
    pub shared_capacity: u32,
    pub workgroup_storage_bytes: u32,
    pub dispatch: [u32; 3],
    pub workgroup_size: [u32; 3],
    pub uniform_words: [u32; 4],
    pub output_elements: usize,
    pub output_bytes: u64,
}

const GEMV_TILED_TILE_ROWS: u32 = 8;
const GEMV_TILED_THREADS_PER_ROW: u32 = 32;
const GEMV_TILED_VECTOR_WIDTH: u32 = 4;
const GEMV_TILED_SHARED_CAPACITY: u32 = 3072;
const GEMV_TILED_WORKGROUP_STORAGE_BYTES: u32 = 13_312;
const GEMV_TILED_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

const fn gemv_tiled_columns_admitted(columns: u32) -> bool {
    matches!(columns, 1024 | 2048 | 3072)
}

impl GemvTiledDescriptor {
    pub fn plan(self) -> Result<GemvTiledPlan, InvocationError> {
        if self.rows == 0 || !gemv_tiled_columns_admitted(self.columns) {
            return Err(invalid_decoder_geometry(format!(
                "tiled decoder GEMV geometry {}x{} is outside the admitted nonzero rows and 1024/2048/3072 input widths",
                self.rows, self.columns
            )));
        }
        let dispatch_x = ceil_div(self.rows, GEMV_TILED_TILE_ROWS);
        if dispatch_x > MAX_WEBGPU_DISPATCH_DIMENSION {
            return Err(invalid_decoder_geometry(format!(
                "tiled decoder GEMV rows {} exceed the single-dispatch workgroup bound",
                self.rows
            )));
        }
        let output_elements = usize::try_from(self.rows).map_err(|_| overflow())?;
        let output_bytes = checked_output_bytes(u64::from(self.rows))?;
        Ok(GemvTiledPlan {
            tile_rows: GEMV_TILED_TILE_ROWS,
            threads_per_row: GEMV_TILED_THREADS_PER_ROW,
            vector_width: GEMV_TILED_VECTOR_WIDTH,
            shared_capacity: GEMV_TILED_SHARED_CAPACITY,
            workgroup_storage_bytes: GEMV_TILED_WORKGROUP_STORAGE_BYTES,
            dispatch: [dispatch_x, 1, 1],
            workgroup_size: GEMV_TILED_WORKGROUP_SIZE,
            uniform_words: [self.rows, self.columns, 0, 0],
            output_elements,
            output_bytes,
        })
    }
}

fn gemv_tiled_invocation(rows: u32, columns: u32) -> Result<InvocationPlan, InvocationError> {
    let tiled = GemvTiledDescriptor { rows, columns }.plan()?;
    Ok(InvocationPlan {
        kernel: KernelId::GemvTiledF32,
        output_elements: tiled.output_elements,
        output_bytes: tiled.output_bytes,
        workgroup_size: tiled.workgroup_size,
        dispatch: tiled.dispatch,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderCachedGqaStage {
    AppendKeyValue,
    DirectGqa,
    /// M7o2 amendment: split-K partial reduction over one cache chunk plane.
    SplitGqaPartial,
    /// M7o2 amendment: ascending-chunk merge of the split-K partials plane.
    SplitGqaMerge,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderCachedGqaDispatchPlan {
    pub stage: DecoderCachedGqaStage,
    pub invocation: InvocationPlan,
    pub uniform_words: [u32; 4],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderCachedGqaPlan {
    pub cache_tokens_after_append: u32,
    pub query_elements: usize,
    pub key_value_width: usize,
    pub cache_elements: usize,
    pub valid_cache_elements: usize,
    pub cache_bytes: u64,
    pub attention_bytes: u64,
    pub append: DecoderCachedGqaDispatchPlan,
    pub attention: DecoderCachedGqaDispatchPlan,
}

/// Initial compact device-side KV cache admitted by a persistent native
/// decoder session.
#[derive(Clone, Copy, Debug)]
pub struct DecoderKvSessionDescriptor<'a> {
    pub query_heads: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
    pub prefix_tokens: u32,
    pub cache_capacity: u32,
    pub key_cache: &'a [f32],
    pub value_cache: &'a [f32],
}

/// Caller-owned operands for one append followed by one direct GQA dispatch.
#[derive(Clone, Copy, Debug)]
pub struct DecoderKvSessionStep<'a> {
    pub query: &'a [f32],
    pub appended_key: &'a [f32],
    pub appended_value: &'a [f32],
}

/// Static arithmetic and dispatch authority shared by every step of one
/// persistent KV session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderKvSessionPlan {
    pub initial_cache_tokens: u32,
    pub cache_capacity: u32,
    pub query_heads: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
    pub query_elements: usize,
    pub key_value_width: usize,
    pub cache_elements: usize,
    pub cache_bytes: u64,
    pub attention_bytes: u64,
    pub append_invocation: InvocationPlan,
    pub attention_invocation: InvocationPlan,
    pub split_partials_bytes: u64,
}

/// Dynamic plan for one committed cache transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderKvSessionStepPlan {
    pub cache_tokens_before: u32,
    pub cache_tokens_after: u32,
    pub valid_cache_elements: usize,
    pub append: DecoderCachedGqaDispatchPlan,
    pub attention: DecoderCachedGqaDispatchPlan,
    pub split_gqa: DecoderGqaSplitPlan,
}

/// One pinned PaddleOCR-VL-1.6 cached-attention step.
///
/// The cache slices describe compact `[capacity, 2, 128]` physical storage.
/// Only the first `prefix_tokens` rows are semantic inputs; the next row is
/// overwritten by the append dispatch and all later rows are inert capacity.
#[derive(Clone, Copy, Debug)]
pub struct DecoderCachedGqaInvocation<'a> {
    pub query_heads: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
    pub prefix_tokens: u32,
    pub cache_capacity: u32,
    pub query: &'a [f32],
    pub appended_key: &'a [f32],
    pub appended_value: &'a [f32],
    pub key_cache: &'a [f32],
    pub value_cache: &'a [f32],
}

impl DecoderCachedGqaInvocation<'_> {
    pub fn plan(&self) -> Result<DecoderCachedGqaPlan, InvocationError> {
        let session = plan_decoder_kv_session_geometry(
            self.query_heads,
            self.key_value_heads,
            self.head_dim,
            self.prefix_tokens,
            self.cache_capacity,
        )?;
        let query_elements_u64 = u64::try_from(session.query_elements).map_err(|_| overflow())?;
        let key_value_width_u64 = u64::try_from(session.key_value_width).map_err(|_| overflow())?;
        let cache_elements_u64 = u64::try_from(session.cache_elements).map_err(|_| overflow())?;
        let prefix_cache_elements_u64 = u64::from(self.prefix_tokens)
            .checked_mul(key_value_width_u64)
            .ok_or_else(overflow)?;
        let prefix_cache_elements =
            usize::try_from(prefix_cache_elements_u64).map_err(|_| overflow())?;

        require_len(self.query.len(), query_elements_u64)?;
        require_len(self.appended_key.len(), key_value_width_u64)?;
        require_len(self.appended_value.len(), key_value_width_u64)?;
        require_len(self.key_cache.len(), cache_elements_u64)?;
        require_len(self.value_cache.len(), cache_elements_u64)?;
        require_finite(self.query)?;
        require_finite(self.appended_key)?;
        require_finite(self.appended_value)?;
        require_finite(&self.key_cache[..prefix_cache_elements])?;
        require_finite(&self.value_cache[..prefix_cache_elements])?;

        let step = session.plan_step(
            self.prefix_tokens,
            &DecoderKvSessionStep {
                query: self.query,
                appended_key: self.appended_key,
                appended_value: self.appended_value,
            },
        )?;

        Ok(DecoderCachedGqaPlan {
            cache_tokens_after_append: step.cache_tokens_after,
            query_elements: session.query_elements,
            key_value_width: session.key_value_width,
            cache_elements: session.cache_elements,
            valid_cache_elements: step.valid_cache_elements,
            cache_bytes: session.cache_bytes,
            attention_bytes: session.attention_bytes,
            append: step.append,
            attention: step.attention,
        })
    }
}

impl DecoderKvSessionDescriptor<'_> {
    pub fn plan(&self) -> Result<DecoderKvSessionPlan, InvocationError> {
        let plan = plan_decoder_kv_session_geometry(
            self.query_heads,
            self.key_value_heads,
            self.head_dim,
            self.prefix_tokens,
            self.cache_capacity,
        )?;
        let cache_elements = u64::try_from(plan.cache_elements).map_err(|_| overflow())?;
        require_len(self.key_cache.len(), cache_elements)?;
        require_len(self.value_cache.len(), cache_elements)?;
        let semantic_elements_u64 = u64::from(self.prefix_tokens)
            .checked_mul(u64::try_from(plan.key_value_width).map_err(|_| overflow())?)
            .ok_or_else(overflow)?;
        let semantic_elements = usize::try_from(semantic_elements_u64).map_err(|_| overflow())?;
        require_finite(&self.key_cache[..semantic_elements])?;
        require_finite(&self.value_cache[..semantic_elements])?;
        Ok(plan)
    }
}

impl DecoderKvSessionPlan {
    /// Plans one committed cache transition without admitting any host
    /// operands: geometry validation plus the append and direct-GQA dispatch
    /// plans with their exact uniform words.
    pub fn plan_cache_transition(
        &self,
        cache_tokens: u32,
    ) -> Result<DecoderKvSessionStepPlan, InvocationError> {
        if cache_tokens < self.initial_cache_tokens {
            return Err(invalid_decoder_geometry(
                "persistent decoder cache length cannot move before its admitted prefix",
            ));
        }
        if cache_tokens >= self.cache_capacity {
            return Err(invalid_decoder_geometry(
                "persistent decoder cache has no remaining append slot",
            ));
        }
        let cache_tokens_after = cache_tokens.checked_add(1).ok_or_else(overflow)?;
        let valid_cache_elements_u64 = u64::from(cache_tokens_after)
            .checked_mul(u64::try_from(self.key_value_width).map_err(|_| overflow())?)
            .ok_or_else(overflow)?;
        if valid_cache_elements_u64 == 0 || valid_cache_elements_u64 - 1 > u64::from(u32::MAX) {
            return Err(overflow());
        }
        let valid_cache_elements =
            usize::try_from(valid_cache_elements_u64).map_err(|_| overflow())?;

        Ok(DecoderKvSessionStepPlan {
            cache_tokens_before: cache_tokens,
            cache_tokens_after,
            valid_cache_elements,
            append: DecoderCachedGqaDispatchPlan {
                stage: DecoderCachedGqaStage::AppendKeyValue,
                invocation: self.append_invocation,
                uniform_words: [
                    cache_tokens,
                    self.key_value_heads,
                    self.head_dim,
                    self.cache_capacity,
                ],
            },
            attention: DecoderCachedGqaDispatchPlan {
                stage: DecoderCachedGqaStage::DirectGqa,
                invocation: self.attention_invocation,
                uniform_words: [
                    cache_tokens_after,
                    self.query_heads,
                    self.key_value_heads,
                    self.head_dim,
                ],
            },
            split_gqa: DecoderGqaSplitDescriptor::pinned(cache_tokens_after).plan()?,
        })
    }

    pub fn plan_step(
        &self,
        cache_tokens: u32,
        step: &DecoderKvSessionStep<'_>,
    ) -> Result<DecoderKvSessionStepPlan, InvocationError> {
        let transition = self.plan_cache_transition(cache_tokens)?;
        require_len(
            step.query.len(),
            u64::try_from(self.query_elements).map_err(|_| overflow())?,
        )?;
        require_len(
            step.appended_key.len(),
            u64::try_from(self.key_value_width).map_err(|_| overflow())?,
        )?;
        require_len(
            step.appended_value.len(),
            u64::try_from(self.key_value_width).map_err(|_| overflow())?,
        )?;
        require_finite(step.query)?;
        require_finite(step.appended_key)?;
        require_finite(step.appended_value)?;
        Ok(transition)
    }
}

fn plan_decoder_kv_session_geometry(
    query_heads: u32,
    key_value_heads: u32,
    head_dim: u32,
    prefix_tokens: u32,
    cache_capacity: u32,
) -> Result<DecoderKvSessionPlan, InvocationError> {
    require_dimensions(&[
        query_heads,
        key_value_heads,
        head_dim,
        prefix_tokens,
        cache_capacity,
    ])?;
    if head_dim > MAX_DECODER_HEAD_DIM {
        return Err(InvocationError::new(
            InvocationErrorCode::UnsupportedHeadDimension,
            format!(
                "decoder head dimension {head_dim} exceeds the supported maximum {MAX_DECODER_HEAD_DIM}"
            ),
        ));
    }

    let query_elements_u64 = checked_elements(query_heads, head_dim)?;
    let key_value_width_u64 = checked_elements(key_value_heads, head_dim)?;
    let cache_elements_u64 = u64::from(cache_capacity)
        .checked_mul(key_value_width_u64)
        .ok_or_else(overflow)?;
    let cache_tokens_after_append = prefix_tokens.checked_add(1).ok_or_else(overflow)?;
    let valid_cache_elements_u64 = u64::from(cache_tokens_after_append)
        .checked_mul(key_value_width_u64)
        .ok_or_else(overflow)?;
    for elements in [
        query_elements_u64,
        key_value_width_u64,
        cache_elements_u64,
        valid_cache_elements_u64,
    ] {
        if elements == 0 || elements - 1 > u64::from(u32::MAX) {
            return Err(overflow());
        }
    }

    if !query_heads.is_multiple_of(key_value_heads)
        || query_heads != 16
        || key_value_heads != 2
        || head_dim != MAX_DECODER_HEAD_DIM
    {
        return Err(invalid_decoder_geometry(
            "cached decoder GQA is pinned to Q16/KV2/D128 with contiguous query-head groups",
        ));
    }
    if prefix_tokens >= cache_capacity {
        return Err(invalid_decoder_geometry(format!(
            "cache capacity {cache_capacity} has no append slot after prefix length {prefix_tokens}"
        )));
    }
    // The split-K decode attention replaces the serial GQA dispatch in every
    // persistent session: the whole capacity must admit the split partial
    // dispatch and pin the exact scratch plane size.
    let split_partials_bytes = DecoderGqaSplitDescriptor::pinned(cache_capacity)
        .plan()?
        .partials_bytes;

    let query_elements = usize::try_from(query_elements_u64).map_err(|_| overflow())?;
    let key_value_width = usize::try_from(key_value_width_u64).map_err(|_| overflow())?;
    let cache_elements = usize::try_from(cache_elements_u64).map_err(|_| overflow())?;
    let cache_bytes = checked_output_bytes(cache_elements_u64)?;
    let attention_bytes = checked_output_bytes(query_elements_u64)?;
    let append_output_elements_u64 = cache_elements_u64.checked_mul(2).ok_or_else(overflow)?;
    let append_output_elements =
        usize::try_from(append_output_elements_u64).map_err(|_| overflow())?;
    let append_output_bytes = cache_bytes.checked_mul(2).ok_or_else(overflow)?;
    let key_value_width_u32 = u32::try_from(key_value_width_u64).map_err(|_| overflow())?;

    Ok(DecoderKvSessionPlan {
        initial_cache_tokens: prefix_tokens,
        cache_capacity,
        query_heads,
        key_value_heads,
        head_dim,
        query_elements,
        key_value_width,
        cache_elements,
        cache_bytes,
        attention_bytes,
        append_invocation: InvocationPlan {
            kernel: KernelId::DecoderKvAppendF32,
            output_elements: append_output_elements,
            output_bytes: append_output_bytes,
            workgroup_size: [64, 1, 1],
            dispatch: [ceil_div(key_value_width_u32, 64), 1, 1],
        },
        attention_invocation: InvocationPlan {
            kernel: KernelId::DecoderGqaF32,
            output_elements: query_elements,
            output_bytes: attention_bytes,
            workgroup_size: [64, 1, 1],
            dispatch: [ceil_div(query_heads, 64), 1, 1],
        },
        split_partials_bytes,
    })
}

/// Pinned PaddleOCR-VL-1.6 decoder attention-block topology constants.
pub const PINNED_DECODER_HIDDEN_SIZE: u32 = 1024;
pub const PINNED_DECODER_INTERMEDIATE_SIZE: u32 = 3072;
pub const PINNED_DECODER_LAYERS: u32 = 18;
pub const PINNED_DECODER_QUERY_HEADS: u32 = 16;
pub const PINNED_DECODER_KEY_VALUE_HEADS: u32 = 2;
pub const PINNED_DECODER_RMS_NORM_EPSILON: f32 = 1.0e-5;
pub const PINNED_DECODER_MROPE_SECTIONS: [usize; 3] = [16, 24, 24];
/// Pinned PaddleOCR-VL-1.6 vocabulary width of the output-major LM head.
pub const PINNED_DECODER_VOCAB_SIZE: u32 = 103_424;

/// Attention weight and rotary-table operands admitted by one persistent
/// decoder attention-block session.
#[derive(Clone, Copy, Debug)]
pub struct DecoderAttentionBlockDescriptor<'a> {
    pub hidden_size: u32,
    pub query_heads: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
    pub rms_norm_epsilon: f32,
    pub norm1_weight: &'a [f32],
    pub q_weight: &'a [f32],
    pub k_weight: &'a [f32],
    pub v_weight: &'a [f32],
    pub o_weight: &'a [f32],
    pub mrope_cos: &'a [f32],
    pub mrope_sin: &'a [f32],
    pub cache_capacity: u32,
}

/// Exact static geometry and per-stage dispatch authority shared by every
/// step of one persistent decoder attention-block session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoderAttentionBlockPlan {
    pub hidden_size: u32,
    pub query_heads: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
    pub query_width: usize,
    pub key_value_width: usize,
    pub rope_elements: usize,
    pub cache_capacity: u32,
    pub rms_norm_epsilon: f32,
    pub mrope_sections: [usize; 3],
    pub rms_norm_invocation: InvocationPlan,
    pub query_invocation: InvocationPlan,
    pub key_invocation: InvocationPlan,
    pub value_invocation: InvocationPlan,
    pub output_invocation: InvocationPlan,
    pub mrope_invocation: InvocationPlan,
    pub residual_invocation: InvocationPlan,
}

impl DecoderAttentionBlockDescriptor<'_> {
    pub fn plan(&self) -> Result<DecoderAttentionBlockPlan, InvocationError> {
        if self.cache_capacity == 0 {
            return Err(invalid_decoder_geometry(
                "decoder attention block requires a nonzero cache capacity",
            ));
        }
        require_dimensions(&[
            self.hidden_size,
            self.query_heads,
            self.key_value_heads,
            self.head_dim,
            self.cache_capacity,
        ])?;
        if self.head_dim > MAX_DECODER_HEAD_DIM {
            return Err(InvocationError::new(
                InvocationErrorCode::UnsupportedHeadDimension,
                format!(
                    "decoder head dimension {} exceeds the supported maximum {MAX_DECODER_HEAD_DIM}",
                    self.head_dim
                ),
            ));
        }
        if self.hidden_size != PINNED_DECODER_HIDDEN_SIZE
            || self.query_heads != PINNED_DECODER_QUERY_HEADS
            || self.key_value_heads != PINNED_DECODER_KEY_VALUE_HEADS
            || self.head_dim != MAX_DECODER_HEAD_DIM
            || !self.query_heads.is_multiple_of(self.key_value_heads)
        {
            return Err(invalid_decoder_geometry(
                "decoder attention block is pinned to H1024 / Q16 / KV2 / D128 with contiguous query-head groups",
            ));
        }
        if self.rms_norm_epsilon != PINNED_DECODER_RMS_NORM_EPSILON {
            return Err(invalid_decoder_geometry(format!(
                "decoder rms-norm epsilon {} differs from the pinned {PINNED_DECODER_RMS_NORM_EPSILON}",
                self.rms_norm_epsilon
            )));
        }

        let query_elements_u64 = checked_elements(self.query_heads, self.head_dim)?;
        let key_value_width_u64 = checked_elements(self.key_value_heads, self.head_dim)?;
        let rope_elements_u64 = u64::from(3u32)
            .checked_mul(u64::from(self.cache_capacity))
            .and_then(|value| value.checked_mul(u64::from(self.head_dim)))
            .ok_or_else(overflow)?;
        for elements in [query_elements_u64, key_value_width_u64, rope_elements_u64] {
            if elements == 0 || elements - 1 > u64::from(u32::MAX) {
                return Err(overflow());
            }
        }
        let query_width = usize::try_from(query_elements_u64).map_err(|_| overflow())?;
        let key_value_width = usize::try_from(key_value_width_u64).map_err(|_| overflow())?;
        let rope_elements = usize::try_from(rope_elements_u64).map_err(|_| overflow())?;

        require_len(self.norm1_weight.len(), u64::from(self.hidden_size))?;
        require_len(
            self.q_weight.len(),
            query_elements_u64
                .checked_mul(u64::from(self.hidden_size))
                .ok_or_else(overflow)?,
        )?;
        require_len(
            self.k_weight.len(),
            key_value_width_u64
                .checked_mul(u64::from(self.hidden_size))
                .ok_or_else(overflow)?,
        )?;
        require_len(
            self.v_weight.len(),
            key_value_width_u64
                .checked_mul(u64::from(self.hidden_size))
                .ok_or_else(overflow)?,
        )?;
        require_len(
            self.o_weight.len(),
            u64::from(self.hidden_size)
                .checked_mul(query_elements_u64)
                .ok_or_else(overflow)?,
        )?;
        require_len(self.mrope_cos.len(), rope_elements_u64)?;
        require_len(self.mrope_sin.len(), rope_elements_u64)?;

        require_finite(self.norm1_weight)?;
        require_finite(self.q_weight)?;
        require_finite(self.k_weight)?;
        require_finite(self.v_weight)?;
        require_finite(self.o_weight)?;
        require_finite(self.mrope_cos)?;
        require_finite(self.mrope_sin)?;

        let hidden = self.hidden_size;
        Ok(DecoderAttentionBlockPlan {
            hidden_size: self.hidden_size,
            query_heads: self.query_heads,
            key_value_heads: self.key_value_heads,
            head_dim: self.head_dim,
            query_width,
            key_value_width,
            rope_elements,
            cache_capacity: self.cache_capacity,
            rms_norm_epsilon: self.rms_norm_epsilon,
            mrope_sections: PINNED_DECODER_MROPE_SECTIONS,
            rms_norm_invocation: InvocationPlan {
                kernel: KernelId::RmsNormF32,
                output_elements: hidden as usize,
                output_bytes: checked_output_bytes(u64::from(hidden))?,
                workgroup_size: [64, 1, 1],
                dispatch: [1, 1, 1],
            },
            query_invocation: gemv_tiled_invocation(
                self.query_heads * self.head_dim,
                self.hidden_size,
            )?,
            key_invocation: gemv_tiled_invocation(
                self.key_value_heads * self.head_dim,
                self.hidden_size,
            )?,
            value_invocation: gemv_tiled_invocation(
                self.key_value_heads * self.head_dim,
                self.hidden_size,
            )?,
            output_invocation: gemv_tiled_invocation(
                self.hidden_size,
                self.query_heads * self.head_dim,
            )?,
            mrope_invocation: InvocationPlan {
                kernel: KernelId::DecoderMropeF32,
                output_elements: ((self.query_heads + self.key_value_heads) * self.head_dim)
                    as usize,
                output_bytes: checked_output_bytes(u64::from(
                    (self.query_heads + self.key_value_heads) * self.head_dim,
                ))?,
                workgroup_size: [64, 1, 1],
                dispatch: [
                    ceil_div(
                        (self.query_heads + self.key_value_heads) * self.head_dim,
                        64,
                    ),
                    1,
                    1,
                ],
            },
            residual_invocation: InvocationPlan {
                kernel: KernelId::AddF32,
                output_elements: hidden as usize,
                output_bytes: checked_output_bytes(u64::from(hidden))?,
                workgroup_size: [64, 1, 1],
                dispatch: [ceil_div(hidden, 64), 1, 1],
            },
        })
    }
}

/// Caller-owned operand for one decoder attention-block step.
#[derive(Clone, Copy, Debug)]
pub struct DecoderAttentionBlockStep<'a> {
    pub hidden_row: &'a [f32],
}

/// Dynamic plan for one admitted decoder attention-block step: the seven
/// stage-uniform word sets in chain order (`rmsnorm`, `linear q`, `linear k`,
/// `linear v`, `mrope q/k`, `linear o`, `residual add`) with the step position
/// applied to the M-RoPE stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderAttentionBlockStepPlan {
    pub position: u32,
    pub stage_uniform_words: [[u32; 4]; 7],
}

impl DecoderAttentionBlockPlan {
    pub fn plan_step(
        &self,
        cache_tokens: u32,
        step: &DecoderAttentionBlockStep<'_>,
    ) -> Result<DecoderAttentionBlockStepPlan, InvocationError> {
        if cache_tokens >= self.cache_capacity {
            return Err(invalid_decoder_geometry(
                "decoder attention block step position has no rope-table row",
            ));
        }
        require_len(step.hidden_row.len(), u64::from(self.hidden_size))?;
        require_finite(step.hidden_row)?;

        let query_width = u32::try_from(self.query_width).map_err(|_| overflow())?;
        let key_value_width = u32::try_from(self.key_value_width).map_err(|_| overflow())?;
        Ok(DecoderAttentionBlockStepPlan {
            position: cache_tokens,
            stage_uniform_words: [
                [1, self.hidden_size, self.rms_norm_epsilon.to_bits(), 0],
                [query_width, self.hidden_size, 0, 0],
                [key_value_width, self.hidden_size, 0, 0],
                [key_value_width, self.hidden_size, 0, 0],
                [cache_tokens, self.cache_capacity, 0, 0],
                [self.hidden_size, query_width, 0, 0],
                [self.hidden_size, 0, 0, 0],
            ],
        })
    }
}

/// Full-layer weight operands admitted by one persistent decoder layer
/// session: the attention-block operands plus the SwiGLU MLP weights.
#[derive(Clone, Copy, Debug)]
pub struct DecoderLayerDescriptor<'a> {
    pub attention: DecoderAttentionBlockDescriptor<'a>,
    pub intermediate_size: u32,
    pub norm2_weight: &'a [f32],
    pub gate_weight: &'a [f32],
    pub up_weight: &'a [f32],
    pub down_weight: &'a [f32],
}

/// Exact static geometry and per-stage dispatch authority shared by every
/// step of one persistent full decoder layer session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoderLayerPlan {
    pub attention_block: DecoderAttentionBlockPlan,
    pub intermediate_size: u32,
    pub norm2_invocation: InvocationPlan,
    pub gate_invocation: InvocationPlan,
    pub up_invocation: InvocationPlan,
    pub swiglu_invocation: InvocationPlan,
    pub down_invocation: InvocationPlan,
    pub second_residual_invocation: InvocationPlan,
}

/// Caller-owned operand for one decoder layer step.
#[derive(Clone, Copy, Debug)]
pub struct DecoderLayerStep<'a> {
    pub hidden_row: &'a [f32],
}

/// Dynamic plan for one admitted decoder layer step: the thirteen
/// stage-uniform word sets in chain order (`rmsnorm`, `linear q`, `linear k`,
/// `linear v`, `mrope q/k`, `linear o`, `residual add`, `post-attention
/// rmsnorm`, `linear gate`, `linear up`, `swiglu`, `linear down`, `residual
/// add`) with the step position applied to the M-RoPE stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderLayerStepPlan {
    pub position: u32,
    pub stage_uniform_words: [[u32; 4]; 13],
}

impl DecoderLayerDescriptor<'_> {
    pub fn plan(&self) -> Result<DecoderLayerPlan, InvocationError> {
        let attention_block = self.attention.plan()?;
        if self.intermediate_size != PINNED_DECODER_INTERMEDIATE_SIZE {
            return Err(invalid_decoder_geometry(format!(
                "decoder intermediate size {} differs from the pinned {PINNED_DECODER_INTERMEDIATE_SIZE}",
                self.intermediate_size
            )));
        }
        let hidden = attention_block.hidden_size;
        let hidden_size = hidden as usize;
        let intermediate = self.intermediate_size;
        let intermediate_size = intermediate as usize;

        let intermediate_weight_elements_u64 = u64::from(intermediate)
            .checked_mul(u64::from(hidden))
            .ok_or_else(overflow)?;
        require_len(self.norm2_weight.len(), u64::from(hidden))?;
        require_len(self.gate_weight.len(), intermediate_weight_elements_u64)?;
        require_len(self.up_weight.len(), intermediate_weight_elements_u64)?;
        require_len(self.down_weight.len(), intermediate_weight_elements_u64)?;

        require_finite(self.norm2_weight)?;
        require_finite(self.gate_weight)?;
        require_finite(self.up_weight)?;
        require_finite(self.down_weight)?;

        Ok(DecoderLayerPlan {
            attention_block,
            intermediate_size: intermediate,
            norm2_invocation: InvocationPlan {
                kernel: KernelId::RmsNormF32,
                output_elements: hidden_size,
                output_bytes: checked_output_bytes(u64::from(hidden))?,
                workgroup_size: [64, 1, 1],
                dispatch: [1, 1, 1],
            },
            gate_invocation: gemv_tiled_invocation(intermediate, hidden)?,
            up_invocation: gemv_tiled_invocation(intermediate, hidden)?,
            swiglu_invocation: InvocationPlan {
                kernel: KernelId::DecoderSwigluF32,
                output_elements: intermediate_size,
                output_bytes: checked_output_bytes(u64::from(intermediate))?,
                workgroup_size: [64, 1, 1],
                dispatch: [ceil_div(intermediate, 64), 1, 1],
            },
            down_invocation: gemv_tiled_invocation(hidden, intermediate)?,
            second_residual_invocation: InvocationPlan {
                kernel: KernelId::AddF32,
                output_elements: hidden_size,
                output_bytes: checked_output_bytes(u64::from(hidden))?,
                workgroup_size: [64, 1, 1],
                dispatch: [ceil_div(hidden, 64), 1, 1],
            },
        })
    }
}

impl DecoderLayerPlan {
    pub fn plan_step(
        &self,
        cache_tokens: u32,
        step: &DecoderLayerStep<'_>,
    ) -> Result<DecoderLayerStepPlan, InvocationError> {
        let attention_step = self.attention_block.plan_step(
            cache_tokens,
            &DecoderAttentionBlockStep {
                hidden_row: step.hidden_row,
            },
        )?;
        let hidden = self.attention_block.hidden_size;
        let intermediate = self.intermediate_size;
        let mut stage_uniform_words = [[0u32; 4]; 13];
        stage_uniform_words[..7].copy_from_slice(&attention_step.stage_uniform_words);
        stage_uniform_words[7] = [
            1,
            hidden,
            self.attention_block.rms_norm_epsilon.to_bits(),
            0,
        ];
        stage_uniform_words[8] = [intermediate, hidden, 0, 0];
        stage_uniform_words[9] = [intermediate, hidden, 0, 0];
        stage_uniform_words[10] = [intermediate, 0, 0, 0];
        stage_uniform_words[11] = [hidden, intermediate, 0, 0];
        stage_uniform_words[12] = [hidden, 0, 0, 0];
        Ok(DecoderLayerStepPlan {
            position: attention_step.position,
            stage_uniform_words,
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoderWeightStorage {
    F32,
    F16,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearWeightLayout {
    #[default]
    OutputMajor,
    InputMajor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionEncoderPrecision {
    pub matrix_weight_storage: DecoderWeightStorage,
    pub matrix_weight_layout: LinearWeightLayout,
    pub vector_weight_storage: DecoderWeightStorage,
    pub activation_storage: DecoderWeightStorage,
}

impl VisionEncoderPrecision {
    #[must_use]
    pub const fn legacy(
        matrix_weight_storage: DecoderWeightStorage,
        matrix_weight_layout: LinearWeightLayout,
    ) -> Self {
        Self {
            matrix_weight_storage,
            matrix_weight_layout,
            vector_weight_storage: DecoderWeightStorage::F32,
            activation_storage: DecoderWeightStorage::F32,
        }
    }

    #[must_use]
    pub fn is_full_f16(self) -> bool {
        self.matrix_weight_storage == DecoderWeightStorage::F16
            && self.matrix_weight_layout == LinearWeightLayout::InputMajor
            && self.vector_weight_storage == DecoderWeightStorage::F16
            && self.activation_storage == DecoderWeightStorage::F16
    }
}

impl LinearWeightLayout {
    #[must_use]
    pub const fn uniform_word(self) -> u32 {
        match self {
            Self::OutputMajor => 0,
            Self::InputMajor => 1,
        }
    }
}

impl DecoderWeightStorage {
    #[must_use]
    pub const fn bytes_per_element(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }

    #[must_use]
    pub const fn storage_bytes(self, elements: u64) -> Option<u64> {
        elements.checked_mul(self.bytes_per_element())
    }

    #[must_use]
    pub const fn from_f32_byte_offset(self, offset: u64) -> Option<u64> {
        if !offset.is_multiple_of(4) {
            return None;
        }
        match self {
            Self::F32 => Some(offset),
            Self::F16 => Some(offset / 2),
        }
    }

    #[must_use]
    pub const fn requires_shader_f16(self) -> bool {
        matches!(self, Self::F16)
    }

    #[must_use]
    pub const fn linear_projection_output_columns_per_workgroup(self) -> u32 {
        LINEAR_PROJECTION_TILE
    }

    pub fn validate_finite_bytes(self, bytes: &[u8]) -> Result<u64, DecoderWeightStorageError> {
        let element_width = self.bytes_per_element() as usize;
        if !bytes.len().is_multiple_of(element_width) {
            return Err(DecoderWeightStorageError::new(
                DecoderWeightStorageErrorCode::ByteLengthNotAligned,
                None,
                "weight payload is not aligned to its storage element width",
            ));
        }
        match self {
            Self::F16 => {
                for (index, value) in bytes.chunks_exact(2).enumerate() {
                    let bits = u16::from_le_bytes([value[0], value[1]]);
                    if bits & 0x7c00 == 0x7c00 {
                        return Err(DecoderWeightStorageError::new(
                            DecoderWeightStorageErrorCode::NonFinite,
                            Some(index as u64),
                            "F16 weight payload contains NaN or infinity",
                        ));
                    }
                }
            }
            Self::F32 => {
                for (index, value) in bytes.chunks_exact(4).enumerate() {
                    let bits = u32::from_le_bytes([value[0], value[1], value[2], value[3]]);
                    if !f32::from_bits(bits).is_finite() {
                        return Err(DecoderWeightStorageError::new(
                            DecoderWeightStorageErrorCode::NonFinite,
                            Some(index as u64),
                            "F32 weight payload contains NaN or infinity",
                        ));
                    }
                }
            }
        }
        Ok((bytes.len() / element_width) as u64)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderWeightStorageErrorCode {
    ByteLengthNotAligned,
    NonFinite,
    IncompleteLogitsWeights,
    InvalidGeometry,
    Overflow,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderWeightStorageError {
    code: DecoderWeightStorageErrorCode,
    element_index: Option<u64>,
    message: String,
}

impl DecoderWeightStorageError {
    fn new(
        code: DecoderWeightStorageErrorCode,
        element_index: Option<u64>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            element_index,
            message: message.into(),
        }
    }

    #[must_use]
    pub const fn code(&self) -> DecoderWeightStorageErrorCode {
        self.code
    }

    #[must_use]
    pub const fn element_index(&self) -> Option<u64> {
        self.element_index
    }
}

impl fmt::Display for DecoderWeightStorageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "decoder weight-storage error {:?}: {}",
            self.code, self.message
        )
    }
}

impl Error for DecoderWeightStorageError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionPatchProjectionBytesDescriptor {
    pub patch_count: u32,
    pub input_width: u32,
    pub output_width: u32,
    pub weight_storage: DecoderWeightStorage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionPatchProjectionBytesPlan {
    pub kernel: KernelId,
    pub weight_storage: DecoderWeightStorage,
    pub input_bytes: u64,
    pub weight_bytes: u64,
    pub bias_bytes: u64,
    pub output_bytes: u64,
    pub dispatch: [u32; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionPatchProjectionBytesErrorCode {
    InvalidGeometry,
    Overflow,
    LengthMismatch,
    NonFinite,
    MissingShaderF16,
    DispatchLimitExceeded,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionPatchProjectionBytesError {
    code: VisionPatchProjectionBytesErrorCode,
    message: String,
}

impl VisionPatchProjectionBytesError {
    fn new(code: VisionPatchProjectionBytesErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    #[must_use]
    pub const fn code(&self) -> VisionPatchProjectionBytesErrorCode {
        self.code
    }
}

impl fmt::Display for VisionPatchProjectionBytesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "vision patch-projection bytes error {:?}: {}",
            self.code, self.message
        )
    }
}

impl Error for VisionPatchProjectionBytesError {}

impl VisionPatchProjectionBytesDescriptor {
    pub fn plan(self) -> Result<VisionPatchProjectionBytesPlan, VisionPatchProjectionBytesError> {
        if self.patch_count == 0 || self.input_width == 0 || self.output_width == 0 {
            return Err(VisionPatchProjectionBytesError::new(
                VisionPatchProjectionBytesErrorCode::InvalidGeometry,
                "patch_count, input_width, and output_width must be positive",
            ));
        }
        if self.weight_storage == DecoderWeightStorage::F16
            && (self.input_width % 4 != 0 || self.output_width % 4 != 0)
        {
            return Err(VisionPatchProjectionBytesError::new(
                VisionPatchProjectionBytesErrorCode::InvalidGeometry,
                "FP16 projection input_width and output_width must be a multiple of 4 for packed vec4 IO",
            ));
        }
        let patch_count = u64::from(self.patch_count);
        let input_width = u64::from(self.input_width);
        let output_width = u64::from(self.output_width);
        let input_elements = patch_count
            .checked_mul(input_width)
            .ok_or_else(vision_patch_projection_bytes_overflow)?;
        let weight_elements = output_width
            .checked_mul(input_width)
            .ok_or_else(vision_patch_projection_bytes_overflow)?;
        let output_elements = patch_count
            .checked_mul(output_width)
            .ok_or_else(vision_patch_projection_bytes_overflow)?;
        let input_bytes = input_elements
            .checked_mul(4)
            .ok_or_else(vision_patch_projection_bytes_overflow)?;
        let weight_bytes = self
            .weight_storage
            .storage_bytes(weight_elements)
            .ok_or_else(vision_patch_projection_bytes_overflow)?;
        let bias_bytes = output_width
            .checked_mul(4)
            .ok_or_else(vision_patch_projection_bytes_overflow)?;
        let output_bytes = output_elements
            .checked_mul(4)
            .ok_or_else(vision_patch_projection_bytes_overflow)?;
        let kernel = match self.weight_storage {
            DecoderWeightStorage::F32 => KernelId::VisionPatchProjectionF32,
            DecoderWeightStorage::F16 => KernelId::LinearProjectionF16Weights,
        };
        Ok(VisionPatchProjectionBytesPlan {
            kernel,
            weight_storage: self.weight_storage,
            input_bytes,
            weight_bytes,
            bias_bytes,
            output_bytes,
            dispatch: [
                ceil_div(
                    self.output_width,
                    self.weight_storage
                        .linear_projection_output_columns_per_workgroup(),
                ),
                ceil_div(self.patch_count, LINEAR_PROJECTION_TILE),
                1,
            ],
        })
    }
}

impl VisionPatchProjectionBytesPlan {
    #[must_use]
    pub const fn requires_shader_f16(self) -> bool {
        self.weight_storage.requires_shader_f16()
    }

    pub fn validate_capabilities(
        self,
        shader_f16_available: bool,
        max_compute_workgroups_per_dimension: u32,
    ) -> Result<(), VisionPatchProjectionBytesError> {
        if self.requires_shader_f16() && !shader_f16_available {
            return Err(VisionPatchProjectionBytesError::new(
                VisionPatchProjectionBytesErrorCode::MissingShaderF16,
                "F16 patch-projection weights require shader-f16",
            ));
        }
        if max_compute_workgroups_per_dimension == 0
            || self.dispatch[..2]
                .iter()
                .any(|&count| count > max_compute_workgroups_per_dimension)
        {
            return Err(VisionPatchProjectionBytesError::new(
                VisionPatchProjectionBytesErrorCode::DispatchLimitExceeded,
                "patch-projection dispatch exceeds the device limit",
            ));
        }
        Ok(())
    }

    pub fn validate_operands(
        self,
        input: &[u8],
        weight: &[u8],
        bias: &[u8],
    ) -> Result<(), VisionPatchProjectionBytesError> {
        require_vision_patch_projection_byte_length(input, self.input_bytes, "input")?;
        require_vision_patch_projection_byte_length(weight, self.weight_bytes, "weight")?;
        require_vision_patch_projection_byte_length(bias, self.bias_bytes, "bias")?;
        require_finite_f32_bytes(input, "input")?;
        self.weight_storage
            .validate_finite_bytes(weight)
            .map_err(|error| {
                let code = match error.code() {
                    DecoderWeightStorageErrorCode::NonFinite => {
                        VisionPatchProjectionBytesErrorCode::NonFinite
                    }
                    _ => VisionPatchProjectionBytesErrorCode::LengthMismatch,
                };
                VisionPatchProjectionBytesError::new(
                    code,
                    "patch-projection weight payload is malformed",
                )
            })?;
        require_finite_f32_bytes(bias, "bias")?;
        Ok(())
    }
}

fn vision_patch_projection_bytes_overflow() -> VisionPatchProjectionBytesError {
    VisionPatchProjectionBytesError::new(
        VisionPatchProjectionBytesErrorCode::Overflow,
        "patch-projection byte geometry overflowed",
    )
}

fn require_vision_patch_projection_byte_length(
    bytes: &[u8],
    expected: u64,
    label: &str,
) -> Result<(), VisionPatchProjectionBytesError> {
    if bytes.len() as u64 != expected {
        return Err(VisionPatchProjectionBytesError::new(
            VisionPatchProjectionBytesErrorCode::LengthMismatch,
            format!(
                "patch-projection {label} has {} bytes, expected {expected}",
                bytes.len()
            ),
        ));
    }
    Ok(())
}

fn require_finite_f32_bytes(
    bytes: &[u8],
    label: &str,
) -> Result<(), VisionPatchProjectionBytesError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(VisionPatchProjectionBytesError::new(
            VisionPatchProjectionBytesErrorCode::LengthMismatch,
            format!("patch-projection {label} is not F32 aligned"),
        ));
    }
    if bytes.chunks_exact(4).any(|value| {
        !f32::from_le_bytes(value.try_into().expect("F32 chunk has four bytes")).is_finite()
    }) {
        return Err(VisionPatchProjectionBytesError::new(
            VisionPatchProjectionBytesErrorCode::NonFinite,
            format!("patch-projection {label} contains NaN or infinity"),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderWeightResourceDescriptor {
    pub layers: u32,
    pub f32_layer_weight_stride_bytes: [u64; 9],
    pub f32_rope_table_bytes: u64,
    pub f32_final_norm_weight_bytes: Option<u64>,
    pub f32_lm_head_weight_bytes: Option<u64>,
    pub storage: DecoderWeightStorage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderWeightResourcePlan {
    pub layers: u32,
    pub storage: DecoderWeightStorage,
    pub layer_weight_stride_bytes: [u64; 9],
    pub layer_weight_bulk_bytes: [u64; 9],
    pub rope_table_bytes: u64,
    pub final_norm_weight_bytes: Option<u64>,
    pub lm_head_weight_bytes: Option<u64>,
    pub checkpoint_shard_count: u32,
    pub f16_checkpoint_shard_count: u32,
    pub f32_checkpoint_shard_count: u32,
    pub f32_table_shard_count: u32,
}

impl DecoderWeightResourceDescriptor {
    pub fn plan(self) -> Result<DecoderWeightResourcePlan, DecoderWeightStorageError> {
        if self.layers == 0 || !self.f32_rope_table_bytes.is_multiple_of(4) {
            return Err(DecoderWeightStorageError::new(
                DecoderWeightStorageErrorCode::InvalidGeometry,
                None,
                "decoder weight-resource geometry is invalid",
            ));
        }
        let logits = match (
            self.f32_final_norm_weight_bytes,
            self.f32_lm_head_weight_bytes,
        ) {
            (Some(final_norm), Some(lm_head)) => Some((final_norm, lm_head)),
            (None, None) => None,
            _ => {
                return Err(DecoderWeightStorageError::new(
                    DecoderWeightStorageErrorCode::IncompleteLogitsWeights,
                    None,
                    "final norm and LM head must either both be present or both be absent",
                ));
            }
        };

        let mut layer_weight_stride_bytes = [0_u64; 9];
        let mut layer_weight_bulk_bytes = [0_u64; 9];
        for (slot, f32_stride) in self.f32_layer_weight_stride_bytes.into_iter().enumerate() {
            let stride = self
                .storage
                .from_f32_byte_offset(f32_stride)
                .ok_or_else(|| {
                    DecoderWeightStorageError::new(
                        DecoderWeightStorageErrorCode::InvalidGeometry,
                        None,
                        "F32 layer-weight stride is not element aligned",
                    )
                })?;
            layer_weight_stride_bytes[slot] = stride;
            layer_weight_bulk_bytes[slot] = stride
                .checked_mul(u64::from(self.layers))
                .ok_or_else(weight_storage_overflow)?;
        }

        let (final_norm_weight_bytes, lm_head_weight_bytes) =
            logits.map_or(Ok((None, None)), |(final_norm, lm_head)| {
                Ok((
                    Some(
                        self.storage
                            .from_f32_byte_offset(final_norm)
                            .ok_or_else(weight_storage_invalid_geometry)?,
                    ),
                    Some(
                        self.storage
                            .from_f32_byte_offset(lm_head)
                            .ok_or_else(weight_storage_invalid_geometry)?,
                    ),
                ))
            })?;
        let checkpoint_shard_count = if logits.is_some() { 11 } else { 9 };
        let (f16_checkpoint_shard_count, f32_checkpoint_shard_count) = match self.storage {
            DecoderWeightStorage::F16 => (checkpoint_shard_count, 0),
            DecoderWeightStorage::F32 => (0, checkpoint_shard_count),
        };

        Ok(DecoderWeightResourcePlan {
            layers: self.layers,
            storage: self.storage,
            layer_weight_stride_bytes,
            layer_weight_bulk_bytes,
            rope_table_bytes: self.f32_rope_table_bytes,
            final_norm_weight_bytes,
            lm_head_weight_bytes,
            checkpoint_shard_count,
            f16_checkpoint_shard_count,
            f32_checkpoint_shard_count,
            f32_table_shard_count: 2,
        })
    }
}

impl DecoderWeightResourcePlan {
    #[must_use]
    pub const fn requires_shader_f16(self) -> bool {
        self.storage.requires_shader_f16()
    }

    #[must_use]
    pub fn layer_weight_offsets(self, layer: u32) -> Option<[u64; 9]> {
        if layer >= self.layers {
            return None;
        }
        Some(
            self.layer_weight_stride_bytes
                .map(|stride| u64::from(layer) * stride),
        )
    }

    #[must_use]
    pub fn layer_weight_offset(self, layer: u32, slot: usize) -> Option<u64> {
        self.layer_weight_offsets(layer)
            .and_then(|offsets| offsets.get(slot).copied())
    }

    #[must_use]
    pub fn layer_weight_range(self, layer: u32, slot: usize) -> Option<(u64, u64)> {
        Some((
            self.layer_weight_offset(layer, slot)?,
            *self.layer_weight_stride_bytes.get(slot)?,
        ))
    }

    #[must_use]
    pub const fn final_norm_weight_range(self) -> Option<(u64, u64)> {
        match self.final_norm_weight_bytes {
            Some(bytes) => Some((0, bytes)),
            None => None,
        }
    }

    #[must_use]
    pub const fn lm_head_weight_range(self) -> Option<(u64, u64)> {
        match self.lm_head_weight_bytes {
            Some(bytes) => Some((0, bytes)),
            None => None,
        }
    }
}

fn weight_storage_overflow() -> DecoderWeightStorageError {
    DecoderWeightStorageError::new(
        DecoderWeightStorageErrorCode::Overflow,
        None,
        "decoder weight-resource size overflowed",
    )
}

fn weight_storage_invalid_geometry() -> DecoderWeightStorageError {
    DecoderWeightStorageError::new(
        DecoderWeightStorageErrorCode::InvalidGeometry,
        None,
        "F32 checkpoint weight size is not element aligned",
    )
}

/// Payload-free geometry admitted by a persistent pinned decoder stack.
///
/// This descriptor is the planning authority for runtimes that keep
/// checkpoint weights in a physical format other than F32. It deliberately
/// contains no model operands: authenticated pack/storage validation remains
/// a separate admission step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoderStackGeometryDescriptor {
    pub layers: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub query_heads: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
    pub rms_norm_epsilon: f32,
    pub cache_capacity: u32,
}

impl DecoderStackGeometryDescriptor {
    /// Plans the exact pinned stack without materializing checkpoint weights.
    pub fn plan(&self) -> Result<DecoderStackPlan, InvocationError> {
        if self.layers != PINNED_DECODER_LAYERS {
            return Err(invalid_decoder_geometry(format!(
                "decoder stack layer count {} differs from the pinned {PINNED_DECODER_LAYERS}",
                self.layers
            )));
        }
        if self.cache_capacity == 0 {
            return Err(invalid_decoder_geometry(
                "decoder stack requires a nonzero cache capacity",
            ));
        }
        if self.hidden_size != PINNED_DECODER_HIDDEN_SIZE
            || self.intermediate_size != PINNED_DECODER_INTERMEDIATE_SIZE
            || self.query_heads != PINNED_DECODER_QUERY_HEADS
            || self.key_value_heads != PINNED_DECODER_KEY_VALUE_HEADS
            || self.head_dim != MAX_DECODER_HEAD_DIM
            || !self.query_heads.is_multiple_of(self.key_value_heads)
        {
            return Err(invalid_decoder_geometry(
                "decoder stack is pinned to H1024 / I3072 / Q16 / KV2 / D128 with contiguous query-head groups",
            ));
        }
        if self.rms_norm_epsilon != PINNED_DECODER_RMS_NORM_EPSILON {
            return Err(invalid_decoder_geometry(format!(
                "decoder rms-norm epsilon {} differs from the pinned {PINNED_DECODER_RMS_NORM_EPSILON}",
                self.rms_norm_epsilon
            )));
        }

        let hidden = u64::from(self.hidden_size);
        let intermediate = u64::from(self.intermediate_size);
        let query_width_u32 = self
            .query_heads
            .checked_mul(self.head_dim)
            .ok_or_else(overflow)?;
        let key_value_width_u32 = self
            .key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(overflow)?;
        let query_width = u64::from(query_width_u32);
        let key_value_width = u64::from(key_value_width_u32);
        let rope_elements_u64 = 3_u64
            .checked_mul(u64::from(self.cache_capacity))
            .and_then(|value| value.checked_mul(u64::from(self.head_dim)))
            .ok_or_else(overflow)?;
        let rope_elements = usize::try_from(rope_elements_u64).map_err(|_| overflow())?;
        let mrope_width = self
            .query_heads
            .checked_add(self.key_value_heads)
            .and_then(|heads| heads.checked_mul(self.head_dim))
            .ok_or_else(overflow)?;
        let hidden_output_bytes = checked_output_bytes(hidden)?;
        let intermediate_output_bytes = checked_output_bytes(intermediate)?;
        let mrope_output_bytes = checked_output_bytes(u64::from(mrope_width))?;

        let attention_block = DecoderAttentionBlockPlan {
            hidden_size: self.hidden_size,
            query_heads: self.query_heads,
            key_value_heads: self.key_value_heads,
            head_dim: self.head_dim,
            query_width: usize::try_from(query_width).map_err(|_| overflow())?,
            key_value_width: usize::try_from(key_value_width).map_err(|_| overflow())?,
            rope_elements,
            cache_capacity: self.cache_capacity,
            rms_norm_epsilon: self.rms_norm_epsilon,
            mrope_sections: PINNED_DECODER_MROPE_SECTIONS,
            rms_norm_invocation: InvocationPlan {
                kernel: KernelId::RmsNormF32,
                output_elements: self.hidden_size as usize,
                output_bytes: hidden_output_bytes,
                workgroup_size: [64, 1, 1],
                dispatch: [1, 1, 1],
            },
            query_invocation: gemv_tiled_invocation(query_width_u32, self.hidden_size)?,
            key_invocation: gemv_tiled_invocation(key_value_width_u32, self.hidden_size)?,
            value_invocation: gemv_tiled_invocation(key_value_width_u32, self.hidden_size)?,
            output_invocation: gemv_tiled_invocation(self.hidden_size, query_width_u32)?,
            mrope_invocation: InvocationPlan {
                kernel: KernelId::DecoderMropeF32,
                output_elements: mrope_width as usize,
                output_bytes: mrope_output_bytes,
                workgroup_size: [64, 1, 1],
                dispatch: [ceil_div(mrope_width, 64), 1, 1],
            },
            residual_invocation: InvocationPlan {
                kernel: KernelId::AddF32,
                output_elements: self.hidden_size as usize,
                output_bytes: hidden_output_bytes,
                workgroup_size: [64, 1, 1],
                dispatch: [ceil_div(self.hidden_size, 64), 1, 1],
            },
        };
        let layer_plan = DecoderLayerPlan {
            attention_block,
            intermediate_size: self.intermediate_size,
            norm2_invocation: InvocationPlan {
                kernel: KernelId::RmsNormF32,
                output_elements: self.hidden_size as usize,
                output_bytes: hidden_output_bytes,
                workgroup_size: [64, 1, 1],
                dispatch: [1, 1, 1],
            },
            gate_invocation: gemv_tiled_invocation(self.intermediate_size, self.hidden_size)?,
            up_invocation: gemv_tiled_invocation(self.intermediate_size, self.hidden_size)?,
            swiglu_invocation: InvocationPlan {
                kernel: KernelId::DecoderSwigluF32,
                output_elements: self.intermediate_size as usize,
                output_bytes: intermediate_output_bytes,
                workgroup_size: [64, 1, 1],
                dispatch: [ceil_div(self.intermediate_size, 64), 1, 1],
            },
            down_invocation: gemv_tiled_invocation(self.hidden_size, self.intermediate_size)?,
            second_residual_invocation: InvocationPlan {
                kernel: KernelId::AddF32,
                output_elements: self.hidden_size as usize,
                output_bytes: hidden_output_bytes,
                workgroup_size: [64, 1, 1],
                dispatch: [ceil_div(self.hidden_size, 64), 1, 1],
            },
        };

        Ok(DecoderStackPlan {
            layers: self.layers,
            layer_plan,
            weight_stride_bytes: [
                hidden * 4,
                query_width * hidden * 4,
                key_value_width * hidden * 4,
                key_value_width * hidden * 4,
                hidden * query_width * 4,
                hidden * 4,
                intermediate * hidden * 4,
                intermediate * hidden * 4,
                hidden * intermediate * 4,
            ],
            cache_stride_bytes: u64::from(self.cache_capacity) * key_value_width * 4,
            hidden_stride_bytes: hidden_output_bytes,
        })
    }

    /// Plans the exact multi-token prefill lattice without model operands.
    pub fn plan_prefill(&self, tokens: u32) -> Result<DecoderStackPrefillPlan, InvocationError> {
        require_decoder_prefill_tokens(tokens, self.cache_capacity)?;
        let stack = self.plan()?;
        plan_decoder_stack_prefill_geometry(self, stack, tokens)
    }
}

/// Full-stack layer-major weight operands admitted by one persistent decoder
/// stack session: nine bulk operands (exactly `layers ×` the per-layer
/// shape) plus the shared axis-major M-RoPE tables used by every layer.
#[derive(Clone, Copy, Debug)]
pub struct DecoderStackDescriptor<'a> {
    pub layers: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub query_heads: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
    pub rms_norm_epsilon: f32,
    pub norm1_weight: &'a [f32],
    pub q_weight: &'a [f32],
    pub k_weight: &'a [f32],
    pub v_weight: &'a [f32],
    pub o_weight: &'a [f32],
    pub mrope_cos: &'a [f32],
    pub mrope_sin: &'a [f32],
    pub norm2_weight: &'a [f32],
    pub gate_weight: &'a [f32],
    pub up_weight: &'a [f32],
    pub down_weight: &'a [f32],
    pub cache_capacity: u32,
}

/// Exact static stack geometry: the shared per-layer plan, the pinned layer
/// count, and the per-layer byte strides of every bulk resource addressed
/// through 256-aligned dynamic storage-buffer offsets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoderStackPlan {
    pub layers: u32,
    pub layer_plan: DecoderLayerPlan,
    pub weight_stride_bytes: [u64; 9],
    pub cache_stride_bytes: u64,
    pub hidden_stride_bytes: u64,
}

/// Caller-owned operand for one decoder stack step.
#[derive(Clone, Copy, Debug)]
pub struct DecoderStackStep<'a> {
    pub hidden_row: &'a [f32],
}

impl DecoderStackDescriptor<'_> {
    pub fn plan(&self) -> Result<DecoderStackPlan, InvocationError> {
        // Geometry first (before any bulk operand check) so geometry drift
        // outranks bulk length and finiteness violations. The payload-free
        // descriptor is the shared plan authority for F32 and compact weight
        // storage.
        let geometry_plan = DecoderStackGeometryDescriptor {
            layers: self.layers,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            query_heads: self.query_heads,
            key_value_heads: self.key_value_heads,
            head_dim: self.head_dim,
            rms_norm_epsilon: self.rms_norm_epsilon,
            cache_capacity: self.cache_capacity,
        }
        .plan()?;

        let layers = u64::from(self.layers);
        let hidden = u64::from(self.hidden_size);
        let query_elements_u64 = u64::from(self.query_heads) * u64::from(self.head_dim);
        let key_value_width_u64 = u64::from(self.key_value_heads) * u64::from(self.head_dim);
        let intermediate = u64::from(self.intermediate_size);
        let rope_elements_u64 = 3_u64
            .checked_mul(u64::from(self.cache_capacity))
            .and_then(|value| value.checked_mul(u64::from(self.head_dim)))
            .ok_or_else(overflow)?;
        let bulk_expectations: [(usize, u64); 11] = [
            (self.norm1_weight.len(), layers * hidden),
            (self.q_weight.len(), layers * query_elements_u64 * hidden),
            (self.k_weight.len(), layers * key_value_width_u64 * hidden),
            (self.v_weight.len(), layers * key_value_width_u64 * hidden),
            (self.o_weight.len(), layers * hidden * query_elements_u64),
            (self.mrope_cos.len(), rope_elements_u64),
            (self.mrope_sin.len(), rope_elements_u64),
            (self.norm2_weight.len(), layers * hidden),
            (self.gate_weight.len(), layers * intermediate * hidden),
            (self.up_weight.len(), layers * intermediate * hidden),
            (self.down_weight.len(), layers * hidden * intermediate),
        ];
        for (actual, expected) in bulk_expectations {
            require_len(actual, expected)?;
        }

        let query_width = usize::try_from(query_elements_u64).map_err(|_| overflow())?;
        let key_value_width = usize::try_from(key_value_width_u64).map_err(|_| overflow())?;
        let norm_elements = self.hidden_size as usize;
        let validated_layer = DecoderLayerDescriptor {
            attention: DecoderAttentionBlockDescriptor {
                hidden_size: self.hidden_size,
                query_heads: self.query_heads,
                key_value_heads: self.key_value_heads,
                head_dim: self.head_dim,
                rms_norm_epsilon: self.rms_norm_epsilon,
                norm1_weight: &self.norm1_weight[..norm_elements],
                q_weight: &self.q_weight[..query_width * norm_elements],
                k_weight: &self.k_weight[..key_value_width * norm_elements],
                v_weight: &self.v_weight[..key_value_width * norm_elements],
                o_weight: &self.o_weight[..norm_elements * query_width],
                mrope_cos: self.mrope_cos,
                mrope_sin: self.mrope_sin,
                cache_capacity: self.cache_capacity,
            },
            intermediate_size: self.intermediate_size,
            norm2_weight: &self.norm2_weight[..norm_elements],
            gate_weight: &self.gate_weight[..self.intermediate_size as usize * norm_elements],
            up_weight: &self.up_weight[..self.intermediate_size as usize * norm_elements],
            down_weight: &self.down_weight[..norm_elements * self.intermediate_size as usize],
        }
        .plan()?;
        debug_assert_eq!(geometry_plan.layer_plan, validated_layer);

        require_finite(self.norm1_weight)?;
        require_finite(self.q_weight)?;
        require_finite(self.k_weight)?;
        require_finite(self.v_weight)?;
        require_finite(self.o_weight)?;
        require_finite(self.norm2_weight)?;
        require_finite(self.gate_weight)?;
        require_finite(self.up_weight)?;
        require_finite(self.down_weight)?;

        Ok(geometry_plan)
    }
}

impl DecoderStackPlan {
    pub fn plan_step(
        &self,
        cache_tokens: u32,
        step: &DecoderStackStep<'_>,
    ) -> Result<DecoderLayerStepPlan, InvocationError> {
        self.layer_plan.plan_step(
            cache_tokens,
            &DecoderLayerStep {
                hidden_row: step.hidden_row,
            },
        )
    }
}

/// Full-stack prefill admission: the accepted `DecoderStackDescriptor` bulk
/// operands plus the prompt token count, admitted with the descriptor so one
/// validated plan covers the whole prompt (`1 <= tokens <= cache_capacity`).
#[derive(Clone, Copy, Debug)]
pub struct DecoderStackPrefillDescriptor<'a> {
    pub layers: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub query_heads: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
    pub rms_norm_epsilon: f32,
    pub norm1_weight: &'a [f32],
    pub q_weight: &'a [f32],
    pub k_weight: &'a [f32],
    pub v_weight: &'a [f32],
    pub o_weight: &'a [f32],
    pub mrope_cos: &'a [f32],
    pub mrope_sin: &'a [f32],
    pub norm2_weight: &'a [f32],
    pub gate_weight: &'a [f32],
    pub up_weight: &'a [f32],
    pub down_weight: &'a [f32],
    pub cache_capacity: u32,
    pub tokens: u32,
}

/// Exact static prefill plan: the pinned layer count and admitted prompt
/// token count, the per-layer byte strides shared with the accepted stack
/// plan, and the fifteen stage invocations and stage-uniform word sets of
/// one prefill layer in chain order (`rmsnorm`, `linear q`, `linear k`,
/// `linear v`, `prefill mrope q/k`, `kv range append`, `causal prefill gqa`,
/// `linear o`, `residual add`, `post-attention rmsnorm`, `linear gate`,
/// `linear up`, `swiglu`, `linear down`, `residual add`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoderStackPrefillPlan {
    pub layers: u32,
    pub tokens: u32,
    pub cache_capacity: u32,
    pub weight_stride_bytes: [u64; 9],
    pub cache_stride_bytes: u64,
    pub hidden_stride_bytes: u64,
    pub stage_invocations: [InvocationPlan; 15],
    pub stage_uniform_words: [[u32; 4]; 15],
}

fn require_decoder_prefill_tokens(tokens: u32, cache_capacity: u32) -> Result<(), InvocationError> {
    if tokens == 0 || tokens > cache_capacity {
        return Err(invalid_decoder_geometry(format!(
            "decoder stack prefill token count {tokens} is outside 1..={cache_capacity}",
        )));
    }
    Ok(())
}

fn plan_decoder_stack_prefill_geometry(
    geometry: &DecoderStackGeometryDescriptor,
    stack: DecoderStackPlan,
    tokens: u32,
) -> Result<DecoderStackPrefillPlan, InvocationError> {
    let hidden = geometry.hidden_size;
    let intermediate = geometry.intermediate_size;
    let query_width = geometry
        .query_heads
        .checked_mul(geometry.head_dim)
        .ok_or_else(overflow)?;
    let key_value_width = geometry
        .key_value_heads
        .checked_mul(geometry.head_dim)
        .ok_or_else(overflow)?;
    let mrope_width = geometry
        .query_heads
        .checked_add(geometry.key_value_heads)
        .and_then(|heads| heads.checked_mul(geometry.head_dim))
        .ok_or_else(overflow)?;
    let epsilon_bits = geometry.rms_norm_epsilon.to_bits();

    let token_elements = |width: u32| {
        u64::from(tokens)
            .checked_mul(u64::from(width))
            .ok_or_else(overflow)
    };
    // One 64-wide workgroup axis over the given work-item count; every
    // non-projection prefill stage uses the shared [64, 1, 1] shape.
    let row_stage = |kernel: KernelId,
                     output_elements: u64,
                     work_items: u64|
     -> Result<InvocationPlan, InvocationError> {
        Ok(InvocationPlan {
            kernel,
            output_elements: usize::try_from(output_elements).map_err(|_| overflow())?,
            output_bytes: checked_output_bytes(output_elements)?,
            workgroup_size: [64, 1, 1],
            dispatch: [
                u32::try_from(work_items.div_ceil(64)).map_err(|_| overflow())?,
                1,
                1,
            ],
        })
    };
    // The accepted multi-token linear ABI: one 32x32 output tile is
    // cooperatively computed by each 8x8 vision projection workgroup.
    let projection_stage = |output_width: u32| -> Result<InvocationPlan, InvocationError> {
        let output_elements = token_elements(output_width)?;
        Ok(InvocationPlan {
            kernel: KernelId::VisionPatchProjectionF32,
            output_elements: usize::try_from(output_elements).map_err(|_| overflow())?,
            output_bytes: checked_output_bytes(output_elements)?,
            workgroup_size: [8, 8, 1],
            dispatch: [
                ceil_div(output_width, LINEAR_PROJECTION_TILE),
                ceil_div(tokens, LINEAR_PROJECTION_TILE),
                1,
            ],
        })
    };

    let hidden_elements = token_elements(hidden)?;
    let query_elements = token_elements(query_width)?;
    let key_value_elements = token_elements(key_value_width)?;
    let mrope_elements = token_elements(mrope_width)?;
    let intermediate_elements = token_elements(intermediate)?;
    let gqa_work_items = u64::from(tokens)
        .checked_mul(u64::from(geometry.query_heads))
        .ok_or_else(overflow)?;
    // Both physical cache planes are the range-append output resource,
    // exactly as the accepted single-token append plan reports them.
    let cache_plane_elements = u64::from(geometry.cache_capacity)
        .checked_mul(u64::from(key_value_width))
        .and_then(|plane| plane.checked_mul(2))
        .ok_or_else(overflow)?;
    let hidden_length = tokens.checked_mul(hidden).ok_or_else(overflow)?;
    let intermediate_length = tokens.checked_mul(intermediate).ok_or_else(overflow)?;

    let rms_norm = row_stage(KernelId::RmsNormF32, hidden_elements, u64::from(tokens))?;
    let query_projection = projection_stage(query_width)?;
    let key_value_projection = projection_stage(key_value_width)?;
    let mrope = row_stage(
        KernelId::DecoderPrefillMropeF32,
        mrope_elements,
        mrope_elements,
    )?;
    let kv_append = row_stage(
        KernelId::DecoderKvAppendRangeF32,
        cache_plane_elements,
        key_value_elements,
    )?;
    let gqa = row_stage(
        KernelId::DecoderPrefillGqaF32,
        query_elements,
        gqa_work_items,
    )?;
    let hidden_projection = projection_stage(hidden)?;
    let residual = row_stage(KernelId::AddF32, hidden_elements, hidden_elements)?;
    let intermediate_projection = projection_stage(intermediate)?;
    let swiglu = row_stage(
        KernelId::DecoderSwigluF32,
        intermediate_elements,
        intermediate_elements,
    )?;

    Ok(DecoderStackPrefillPlan {
        layers: stack.layers,
        tokens,
        cache_capacity: geometry.cache_capacity,
        weight_stride_bytes: stack.weight_stride_bytes,
        cache_stride_bytes: stack.cache_stride_bytes,
        hidden_stride_bytes: stack.hidden_stride_bytes,
        stage_invocations: [
            rms_norm,
            query_projection,
            key_value_projection,
            key_value_projection,
            mrope,
            kv_append,
            gqa,
            hidden_projection,
            residual,
            rms_norm,
            intermediate_projection,
            intermediate_projection,
            swiglu,
            hidden_projection,
            residual,
        ],
        stage_uniform_words: [
            [tokens, hidden, epsilon_bits, 0],
            [tokens, hidden, query_width, 0],
            [tokens, hidden, key_value_width, 0],
            [tokens, hidden, key_value_width, 0],
            [tokens, geometry.cache_capacity, 0, 0],
            [tokens, geometry.cache_capacity, 0, 0],
            [
                tokens,
                geometry.query_heads,
                geometry.key_value_heads,
                geometry.head_dim,
            ],
            [tokens, query_width, hidden, 0],
            [hidden_length, 0, 0, 0],
            [tokens, hidden, epsilon_bits, 0],
            [tokens, hidden, intermediate, 0],
            [tokens, hidden, intermediate, 0],
            [intermediate_length, 0, 0, 0],
            [tokens, intermediate, hidden, 0],
            [hidden_length, 0, 0, 0],
        ],
    })
}

impl DecoderStackPrefillDescriptor<'_> {
    pub fn plan(&self) -> Result<DecoderStackPrefillPlan, InvocationError> {
        // Token bounds stay ahead of all payload checks, preserving the
        // accepted descriptor's deterministic error precedence.
        require_decoder_prefill_tokens(self.tokens, self.cache_capacity)?;
        let geometry = DecoderStackGeometryDescriptor {
            layers: self.layers,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            query_heads: self.query_heads,
            key_value_heads: self.key_value_heads,
            head_dim: self.head_dim,
            rms_norm_epsilon: self.rms_norm_epsilon,
            cache_capacity: self.cache_capacity,
        };
        // The legacy descriptor still validates all eleven bulk operands;
        // only static dispatch construction is shared with compact storage.
        let stack = DecoderStackDescriptor {
            layers: self.layers,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            query_heads: self.query_heads,
            key_value_heads: self.key_value_heads,
            head_dim: self.head_dim,
            rms_norm_epsilon: self.rms_norm_epsilon,
            norm1_weight: self.norm1_weight,
            q_weight: self.q_weight,
            k_weight: self.k_weight,
            v_weight: self.v_weight,
            o_weight: self.o_weight,
            mrope_cos: self.mrope_cos,
            mrope_sin: self.mrope_sin,
            norm2_weight: self.norm2_weight,
            gate_weight: self.gate_weight,
            up_weight: self.up_weight,
            down_weight: self.down_weight,
            cache_capacity: self.cache_capacity,
        }
        .plan()?;
        plan_decoder_stack_prefill_geometry(&geometry, stack, self.tokens)
    }
}

/// Payload-free final-norm and LM-head geometry.
///
/// Compact-weight runtimes use this descriptor to admit the logits dispatch
/// lattice while validating the authenticated checkpoint bytes separately.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoderLmHeadGeometryDescriptor {
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub rms_norm_epsilon: f32,
}

impl DecoderLmHeadGeometryDescriptor {
    /// Supplies the literal pinned PaddleOCR-VL-1.6 logits geometry.
    #[must_use]
    pub const fn pinned() -> Self {
        Self {
            hidden_size: PINNED_DECODER_HIDDEN_SIZE,
            vocab_size: PINNED_DECODER_VOCAB_SIZE,
            rms_norm_epsilon: PINNED_DECODER_RMS_NORM_EPSILON,
        }
    }

    /// Plans logits without materializing final-norm or LM-head weights.
    pub fn plan(&self) -> Result<DecoderLmHeadPlan, InvocationError> {
        if self.hidden_size == 0 || self.vocab_size == 0 {
            return Err(invalid_decoder_geometry(format!(
                "decoder LM head geometry H{} / V{} contains a zero dimension",
                self.hidden_size, self.vocab_size
            )));
        }
        let tiled_lm_head = gemv_tiled_columns_admitted(self.hidden_size);
        let rows_per_workgroup = if tiled_lm_head {
            GEMV_TILED_TILE_ROWS
        } else {
            64
        };
        if ceil_div(self.vocab_size, rows_per_workgroup) > MAX_WEBGPU_DISPATCH_DIMENSION {
            return Err(invalid_decoder_geometry(format!(
                "decoder LM head vocabulary {} exceeds the single-dispatch workgroup bound",
                self.vocab_size
            )));
        }
        if !self.rms_norm_epsilon.is_finite() || self.rms_norm_epsilon <= 0.0 {
            return Err(invalid_decoder_geometry(format!(
                "decoder rms-norm epsilon {} is not positive and finite",
                self.rms_norm_epsilon
            )));
        }

        let hidden = self.hidden_size;
        let vocab = self.vocab_size;
        let lm_head_elements = u64::from(vocab)
            .checked_mul(u64::from(hidden))
            .ok_or_else(overflow)?;
        let normed_row_bytes = checked_output_bytes(u64::from(hidden))?;
        let logits_bytes = checked_output_bytes(u64::from(vocab))?;
        Ok(DecoderLmHeadPlan {
            hidden_size: hidden,
            vocab_size: vocab,
            final_norm_weight_bytes: normed_row_bytes,
            lm_head_weight_bytes: checked_output_bytes(lm_head_elements)?,
            normed_row_bytes,
            logits_bytes,
            stage_invocations: [
                InvocationPlan {
                    kernel: KernelId::RmsNormF32,
                    output_elements: hidden as usize,
                    output_bytes: normed_row_bytes,
                    workgroup_size: [64, 1, 1],
                    dispatch: [1, 1, 1],
                },
                if tiled_lm_head {
                    gemv_tiled_invocation(vocab, hidden)?
                } else {
                    InvocationPlan {
                        kernel: KernelId::GemvF32,
                        output_elements: vocab as usize,
                        output_bytes: logits_bytes,
                        workgroup_size: [64, 1, 1],
                        dispatch: [ceil_div(vocab, 64), 1, 1],
                    }
                },
            ],
            stage_uniform_words: [
                [1, hidden, self.rms_norm_epsilon.to_bits(), 0],
                [vocab, hidden, 0, 0],
            ],
        })
    }
}

/// Final-norm and LM-head weight operands admitted by one logits call of a
/// persistent decoder stack session.
#[derive(Clone, Copy, Debug)]
pub struct DecoderLmHeadDescriptor<'a> {
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub rms_norm_epsilon: f32,
    pub final_norm_weight: &'a [f32],
    pub lm_head_weight: &'a [f32],
}

impl DecoderLmHeadDescriptor<'_> {
    /// Supplies the literal pinned PaddleOCR-VL-1.6 geometry; the pinned
    /// descriptor never infers its geometry from operand lengths.
    pub fn pinned<'a>(
        final_norm_weight: &'a [f32],
        lm_head_weight: &'a [f32],
    ) -> DecoderLmHeadDescriptor<'a> {
        DecoderLmHeadDescriptor {
            hidden_size: PINNED_DECODER_HIDDEN_SIZE,
            vocab_size: PINNED_DECODER_VOCAB_SIZE,
            rms_norm_epsilon: PINNED_DECODER_RMS_NORM_EPSILON,
            final_norm_weight,
            lm_head_weight,
        }
    }
}

/// The fully static two-stage logits plan: final rmsnorm of the single
/// current hidden row, then the bias-free output-major LM-head GEMV, with
/// the exact buffer sizes of the persistent logits topology.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoderLmHeadPlan {
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub final_norm_weight_bytes: u64,
    pub lm_head_weight_bytes: u64,
    pub normed_row_bytes: u64,
    pub logits_bytes: u64,
    pub stage_invocations: [InvocationPlan; 2],
    pub stage_uniform_words: [[u32; 4]; 2],
}

impl DecoderLmHeadDescriptor<'_> {
    pub fn plan(&self) -> Result<DecoderLmHeadPlan, InvocationError> {
        // Geometry is validated before any bulk operand check so geometry
        // drift outranks operand length and finiteness violations, exactly
        // like the accepted decoder descriptors.
        let plan = DecoderLmHeadGeometryDescriptor {
            hidden_size: self.hidden_size,
            vocab_size: self.vocab_size,
            rms_norm_epsilon: self.rms_norm_epsilon,
        }
        .plan()?;
        let lm_head_elements = u64::from(plan.vocab_size)
            .checked_mul(u64::from(plan.hidden_size))
            .ok_or_else(overflow)?;

        require_len(self.final_norm_weight.len(), u64::from(self.hidden_size))?;
        require_len(self.lm_head_weight.len(), lm_head_elements)?;

        require_finite(self.final_norm_weight)?;
        require_finite(self.lm_head_weight)?;
        Ok(plan)
    }
}

/// Geometry of one split-K decode GQA attention over the current cache span:
/// the accepted serial per-key loop is split into fixed chunks of the pinned
/// chunk size, each reduced by the split partial kernel and merged in
/// ascending chunk order.
#[derive(Clone, Copy, Debug)]
pub struct DecoderGqaSplitDescriptor {
    pub cache_tokens: u32,
    pub query_heads: u32,
    pub key_value_heads: u32,
    pub head_dim: u32,
}

/// The fully determined split-K plan: the pinned chunk geometry, the exact
/// partials scratch plane, the partial and merge invocations, and the two
/// identical position-dependent uniform word sets written per step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecoderGqaSplitPlan {
    pub chunk_size: u32,
    pub chunk_count: u32,
    pub partials_elements: usize,
    pub partials_bytes: u64,
    pub partial_stride_f32: u32,
    pub partial_invocation: InvocationPlan,
    pub merge_invocation: InvocationPlan,
    pub uniform_words: [[u32; 4]; 2],
}

/// Pinned split-K chunk size of the decode GQA kernels.
pub const PINNED_DECODER_GQA_SPLIT_CHUNK_SIZE: u32 = 32;
/// Pinned f32 stride of one (query_head, chunk) partials record: the 128
/// weighted-V elements plus the chunk max and the chunk sum, padded.
pub const PINNED_DECODER_GQA_SPLIT_PARTIAL_STRIDE_F32: u32 = 192;

impl DecoderGqaSplitDescriptor {
    /// Supplies the literal pinned PaddleOCR-VL-1.6 attention topology
    /// (Q16/KV2/D128); the pinned descriptor never infers its geometry.
    pub fn pinned(cache_tokens: u32) -> DecoderGqaSplitDescriptor {
        DecoderGqaSplitDescriptor {
            cache_tokens,
            query_heads: PINNED_DECODER_QUERY_HEADS,
            key_value_heads: PINNED_DECODER_KEY_VALUE_HEADS,
            head_dim: MAX_DECODER_HEAD_DIM,
        }
    }

    pub fn plan(&self) -> Result<DecoderGqaSplitPlan, InvocationError> {
        if self.cache_tokens == 0
            || self.query_heads == 0
            || self.key_value_heads == 0
            || self.head_dim == 0
        {
            return Err(invalid_decoder_geometry(format!(
                "decoder GQA split geometry T{} / Q{} / KV{} / D{} contains a zero dimension",
                self.cache_tokens, self.query_heads, self.key_value_heads, self.head_dim
            )));
        }
        if self.head_dim != MAX_DECODER_HEAD_DIM {
            return Err(invalid_decoder_geometry(format!(
                "decoder GQA split head dimension {} drifted from the pinned {MAX_DECODER_HEAD_DIM}",
                self.head_dim
            )));
        }
        if !self.query_heads.is_multiple_of(self.key_value_heads) {
            return Err(invalid_decoder_geometry(format!(
                "decoder GQA split query heads {} are not contiguous groups of key-value heads {}",
                self.query_heads, self.key_value_heads
            )));
        }
        if self.query_heads != PINNED_DECODER_QUERY_HEADS
            || self.key_value_heads != PINNED_DECODER_KEY_VALUE_HEADS
        {
            return Err(invalid_decoder_geometry(format!(
                "decoder GQA split head counts Q{} / KV{} drifted from the pinned Q{PINNED_DECODER_QUERY_HEADS} / KV{PINNED_DECODER_KEY_VALUE_HEADS}",
                self.query_heads, self.key_value_heads
            )));
        }
        let chunk_count = self
            .cache_tokens
            .div_ceil(PINNED_DECODER_GQA_SPLIT_CHUNK_SIZE);
        let partial_workgroups = u64::from(self.query_heads)
            .checked_mul(u64::from(chunk_count))
            .ok_or_else(overflow)?;
        if partial_workgroups > u64::from(MAX_WEBGPU_DISPATCH_DIMENSION) {
            return Err(invalid_decoder_geometry(format!(
                "decoder GQA split partial dispatch {partial_workgroups} exceeds the single-dispatch workgroup bound"
            )));
        }
        let partial_record = u64::from(PINNED_DECODER_GQA_SPLIT_PARTIAL_STRIDE_F32);
        let partials_elements_u64 = partial_workgroups
            .checked_mul(partial_record)
            .ok_or_else(overflow)?;
        let partials_elements = usize::try_from(partials_elements_u64).map_err(|_| overflow())?;
        let partials_bytes = partials_elements_u64.checked_mul(4).ok_or_else(overflow)?;
        let query_elements_u64 = checked_elements(self.query_heads, self.head_dim)?;
        let query_elements = usize::try_from(query_elements_u64).map_err(|_| overflow())?;
        let uniform_words = [[self.cache_tokens, chunk_count, 0, 0]; 2];
        Ok(DecoderGqaSplitPlan {
            chunk_size: PINNED_DECODER_GQA_SPLIT_CHUNK_SIZE,
            chunk_count,
            partials_elements,
            partials_bytes,
            partial_stride_f32: PINNED_DECODER_GQA_SPLIT_PARTIAL_STRIDE_F32,
            partial_invocation: InvocationPlan {
                kernel: KernelId::DecoderGqaSplitPartialF32,
                output_elements: partials_elements,
                output_bytes: partials_bytes,
                workgroup_size: [64, 1, 1],
                dispatch: [
                    u32::try_from(partial_workgroups).map_err(|_| overflow())?,
                    1,
                    1,
                ],
            },
            merge_invocation: InvocationPlan {
                kernel: KernelId::DecoderGqaSplitMergeF32,
                output_elements: query_elements,
                output_bytes: checked_output_bytes(query_elements_u64)?,
                workgroup_size: [64, 1, 1],
                dispatch: [ceil_div(self.query_heads * self.head_dim, 64), 1, 1],
            },
            uniform_words,
        })
    }
}

/// Adapter-neutral compute limits checked before an invocation can reach a GPU adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComputeDispatchLimits {
    pub max_workgroup_size: [u32; 3],
    pub max_invocations_per_workgroup: u32,
    pub max_workgroups_per_dimension: u32,
}

impl ComputeDispatchLimits {
    pub fn validate(&self, invocation: &InvocationPlan) -> Result<(), InvocationError> {
        for axis in 0..3 {
            let workgroup_size = invocation.workgroup_size[axis];
            if workgroup_size == 0 || workgroup_size > self.max_workgroup_size[axis] {
                return Err(invalid_fusion_target(format!(
                    "compute workgroup axis {axis} has size {workgroup_size}, outside the adapter limit 1..={}",
                    self.max_workgroup_size[axis]
                )));
            }

            let workgroups = invocation.dispatch[axis];
            if workgroups == 0 || workgroups > self.max_workgroups_per_dimension {
                return Err(invalid_fusion_target(format!(
                    "compute dispatch axis {axis} has {workgroups} workgroups, outside the adapter limit 1..={}",
                    self.max_workgroups_per_dimension
                )));
            }
        }

        let invocations_per_workgroup = invocation
            .workgroup_size
            .into_iter()
            .try_fold(1_u32, u32::checked_mul)
            .ok_or_else(overflow)?;
        if invocations_per_workgroup > self.max_invocations_per_workgroup {
            return Err(invalid_fusion_target(format!(
                "compute workgroup requires {invocations_per_workgroup} invocations but the adapter limit is {}",
                self.max_invocations_per_workgroup
            )));
        }

        Ok(())
    }
}

/// Inputs to the common optimized vision-stack readback planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionQkvReadbackRequirements {
    pub semantic_readback_bytes: u64,
    pub scratch_canary_readback_bytes: u64,
    pub qkv_canary_readback_bytes: u64,
    pub workspace_allocation_bytes: u64,
    pub max_buffer_size: u64,
    pub max_host_elements: u64,
}

/// Validated byte and host-element layout shared by every Q/K/V execution adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionQkvReadbackLayout {
    semantic_offset: u64,
    semantic_readback_bytes: u64,
    scratch_canary_offset: u64,
    scratch_canary_readback_bytes: u64,
    qkv_canary_offset: u64,
    qkv_canary_readback_bytes: u64,
    total_readback_bytes: u64,
    workspace_allocation_bytes: u64,
    readback_f32_elements: usize,
    workspace_u32_words: usize,
}

impl VisionQkvReadbackLayout {
    #[must_use]
    pub const fn semantic_offset(&self) -> u64 {
        self.semantic_offset
    }

    #[must_use]
    pub const fn semantic_readback_bytes(&self) -> u64 {
        self.semantic_readback_bytes
    }

    #[must_use]
    pub const fn scratch_canary_offset(&self) -> u64 {
        self.scratch_canary_offset
    }

    #[must_use]
    pub const fn scratch_canary_readback_bytes(&self) -> u64 {
        self.scratch_canary_readback_bytes
    }

    #[must_use]
    pub const fn qkv_canary_offset(&self) -> u64 {
        self.qkv_canary_offset
    }

    #[must_use]
    pub const fn qkv_canary_readback_bytes(&self) -> u64 {
        self.qkv_canary_readback_bytes
    }

    #[must_use]
    pub const fn total_readback_bytes(&self) -> u64 {
        self.total_readback_bytes
    }

    #[must_use]
    pub const fn workspace_allocation_bytes(&self) -> u64 {
        self.workspace_allocation_bytes
    }

    #[must_use]
    pub const fn readback_f32_elements(&self) -> usize {
        self.readback_f32_elements
    }

    #[must_use]
    pub const fn workspace_u32_words(&self) -> usize {
        self.workspace_u32_words
    }
}

pub fn plan_vision_qkv_readback_layout(
    requirements: VisionQkvReadbackRequirements,
) -> Result<VisionQkvReadbackLayout, InvocationError> {
    let scratch_canary_offset = requirements.semantic_readback_bytes;
    let qkv_canary_offset = scratch_canary_offset
        .checked_add(requirements.scratch_canary_readback_bytes)
        .ok_or_else(overflow)?;
    let total_readback_bytes = qkv_canary_offset
        .checked_add(requirements.qkv_canary_readback_bytes)
        .ok_or_else(overflow)?;

    for (label, bytes) in [
        ("semantic readback", requirements.semantic_readback_bytes),
        (
            "scratch-canary readback",
            requirements.scratch_canary_readback_bytes,
        ),
        (
            "Q/K/V-canary readback",
            requirements.qkv_canary_readback_bytes,
        ),
        (
            "workspace allocation",
            requirements.workspace_allocation_bytes,
        ),
    ] {
        if !bytes.is_multiple_of(4) {
            return Err(invalid_fusion_target(format!(
                "optimized vision-stack {label} size {bytes} is not exactly word-aligned"
            )));
        }
    }

    if total_readback_bytes > requirements.max_buffer_size {
        return Err(invalid_fusion_target(format!(
            "optimized vision-stack readback requires {total_readback_bytes} bytes but the adapter limit is {}",
            requirements.max_buffer_size
        )));
    }
    if requirements.workspace_allocation_bytes > requirements.max_buffer_size {
        return Err(invalid_fusion_target(format!(
            "optimized vision-stack workspace requires {} bytes but the adapter limit is {}",
            requirements.workspace_allocation_bytes, requirements.max_buffer_size
        )));
    }

    let readback_element_count = total_readback_bytes / 4;
    let workspace_word_count = requirements.workspace_allocation_bytes / 4;
    if readback_element_count > requirements.max_host_elements {
        return Err(invalid_fusion_target(format!(
            "optimized vision-stack readback requires {readback_element_count} host elements but the host limit is {}",
            requirements.max_host_elements
        )));
    }
    if workspace_word_count > requirements.max_host_elements {
        return Err(invalid_fusion_target(format!(
            "optimized vision-stack workspace requires {workspace_word_count} host words but the host limit is {}",
            requirements.max_host_elements
        )));
    }

    let readback_f32_elements = usize::try_from(readback_element_count).map_err(|_| {
        invalid_fusion_target("optimized vision-stack readback does not fit host usize")
    })?;
    let workspace_u32_words = usize::try_from(workspace_word_count).map_err(|_| {
        invalid_fusion_target("optimized vision-stack workspace does not fit host usize")
    })?;

    Ok(VisionQkvReadbackLayout {
        semantic_offset: 0,
        semantic_readback_bytes: requirements.semantic_readback_bytes,
        scratch_canary_offset,
        scratch_canary_readback_bytes: requirements.scratch_canary_readback_bytes,
        qkv_canary_offset,
        qkv_canary_readback_bytes: requirements.qkv_canary_readback_bytes,
        total_readback_bytes,
        workspace_allocation_bytes: requirements.workspace_allocation_bytes,
        readback_f32_elements,
        workspace_u32_words,
    })
}

/// Adapter limits which affect the fused vision Q/K/V binding ABI and output layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionQkvFusedTargetLimits {
    pub min_storage_buffer_offset_alignment: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub max_compute_workgroups_per_dimension: u32,
}

/// Whole-stack Q/K/V execution policy selected before any GPU side effect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionQkvExecutionPolicy {
    Disabled,
    Preferred,
    Required,
}

/// Atomic whole-stack topology selected for one optimized invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionQkvSelectionOutcome {
    Disabled,
    Fused,
    FallbackUnsupportedTarget,
}

/// Semantic stage attached to an actually encoded vision-stack dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionQkvStackStage {
    Norm1,
    Query,
    Key,
    Value,
    QkvFused,
    AttentionContext,
    AttentionOutput,
    AttentionResidual,
    Norm2,
    MlpFc1,
    MlpActivation,
    MlpOutput,
    Output,
    PostNorm,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionQkvCopyPurpose {
    Checkpoint,
    SemanticOutput,
    CanaryEvidence,
    TimestampQuery,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionQkvMapPurpose {
    SemanticOutput,
    TimestampQuery,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionQkvCanaryKind {
    Prefix,
    InternalPadding { plane: usize },
    Suffix,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvBufferBindingEvidence {
    pub binding: u32,
    pub buffer_identity: u64,
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvPipelineCreationEvidence {
    pub kernel: KernelId,
    pub shader_blake3: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvBindGroupCreationEvidence {
    pub layer: Option<usize>,
    pub stage: VisionQkvStackStage,
    pub bindings: Vec<VisionQkvBufferBindingEvidence>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvCommandEncoderCreationEvidence {
    pub label: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvDispatchEvidence {
    pub ordinal: usize,
    pub layer: Option<usize>,
    pub stage: VisionQkvStackStage,
    pub kernel: KernelId,
    pub workgroups: [u32; 3],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvCopyEvidence {
    pub ordinal: usize,
    pub source_buffer_identity: u64,
    pub source_offset: u64,
    pub destination_buffer_identity: u64,
    pub destination_offset: u64,
    pub byte_length: u64,
    pub purpose: VisionQkvCopyPurpose,
    pub after_dispatch_ordinal: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvMapEvidence {
    pub purpose: VisionQkvMapPurpose,
    pub buffer_identity: u64,
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvWorkspaceEvidence {
    pub logical_buffer_id: String,
    pub buffer_identity: u64,
    pub allocation_bytes: u64,
    pub semantic_base: u64,
    pub semantic_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvAttentionBindingEvidence {
    pub layer: usize,
    pub binding: u32,
    pub buffer_identity: u64,
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvCanaryEvidence {
    pub kind: VisionQkvCanaryKind,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub expected_bits: u32,
    pub passed: bool,
}

/// Encoding-time evidence for the additive optimized native stack entry point.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvStackExecutionEvidence {
    pub policy: VisionQkvExecutionPolicy,
    pub outcome: VisionQkvSelectionOutcome,
    pub canonical_layer_plan_blake3: Vec<String>,
    pub pipeline_creations: Vec<VisionQkvPipelineCreationEvidence>,
    pub bind_group_creations: Vec<VisionQkvBindGroupCreationEvidence>,
    pub command_encoder_creations: Vec<VisionQkvCommandEncoderCreationEvidence>,
    pub encoded_dispatches: Vec<VisionQkvDispatchEvidence>,
    pub encoded_copies: Vec<VisionQkvCopyEvidence>,
    pub map_requests: Vec<VisionQkvMapEvidence>,
    pub dispatch_count: usize,
    pub compute_pass_count: usize,
    pub command_buffer_count: usize,
    pub submission_count: usize,
    pub map_count: usize,
    pub workspace: Option<VisionQkvWorkspaceEvidence>,
    pub attention_bindings: Vec<VisionQkvAttentionBindingEvidence>,
    pub canaries: Vec<VisionQkvCanaryEvidence>,
}

/// One semantic tensor slice inside the padded fused Q/K/V output buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionQkvFusedOutputSlice {
    pub offset: u64,
    pub size: u64,
}

/// Physical output-buffer layout for the three semantic Q/K/V planes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionQkvFusedOutputLayout {
    pub plane_elements: u64,
    pub plane_bytes: u64,
    pub plane_stride_bytes: u64,
    pub physical_bytes: u64,
    pub query: VisionQkvFusedOutputSlice,
    pub key: VisionQkvFusedOutputSlice,
    pub value: VisionQkvFusedOutputSlice,
}

/// Complete adapter-neutral dispatch and physical-layout plan for fused Q/K/V.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionQkvFusedPlan {
    pub invocation: InvocationPlan,
    pub uniform_words: [u32; 4],
    pub output_layout: VisionQkvFusedOutputLayout,
}

/// Borrowed seven-input fused Q/K/V invocation in binding order.
#[derive(Clone, Copy, Debug)]
pub struct VisionQkvFusedInvocation<'a> {
    pub tokens: u32,
    pub input_width: u32,
    pub output_width: u32,
    pub input: &'a [f32],
    pub query_weight: &'a [f32],
    pub query_bias: &'a [f32],
    pub key_weight: &'a [f32],
    pub key_bias: &'a [f32],
    pub value_weight: &'a [f32],
    pub value_bias: &'a [f32],
}

impl<'a> VisionQkvFusedInvocation<'a> {
    /// Returns the seven read-only storage inputs in the frozen shader ABI order.
    #[must_use]
    pub const fn inputs(&self) -> [&'a [f32]; 7] {
        [
            self.input,
            self.query_weight,
            self.query_bias,
            self.key_weight,
            self.key_bias,
            self.value_weight,
            self.value_bias,
        ]
    }

    pub fn plan(
        &self,
        target: VisionQkvFusedTargetLimits,
    ) -> Result<VisionQkvFusedPlan, InvocationError> {
        let plan = plan_vision_qkv_fused_geometry(
            self.tokens,
            self.input_width,
            self.output_width,
            target,
        )?;
        let input_elements = checked_elements(self.tokens, self.input_width)?;
        let weight_elements = checked_elements(self.output_width, self.input_width)?;
        let bias_elements = u64::from(self.output_width);
        let expected = [
            input_elements,
            weight_elements,
            bias_elements,
            weight_elements,
            bias_elements,
            weight_elements,
            bias_elements,
        ];
        let inputs = self.inputs();
        for (input, expected_elements) in inputs.into_iter().zip(expected) {
            require_len(input.len(), expected_elements)?;
        }
        for input in inputs {
            require_finite(input)?;
        }
        Ok(plan)
    }
}

/// Plans fused vision Q/K/V geometry without reading or allocating operand data.
pub fn plan_vision_qkv_fused_geometry(
    tokens: u32,
    input_width: u32,
    output_width: u32,
    target: VisionQkvFusedTargetLimits,
) -> Result<VisionQkvFusedPlan, InvocationError> {
    plan_vision_qkv_fused_geometry_with_storage(
        tokens,
        input_width,
        output_width,
        target,
        KernelId::VisionQkvFusedF32,
        DecoderWeightStorage::F32,
        VISION_QKV_FUSED_TILE,
        VISION_QKV_FUSED_TILE,
        3,
    )
}

/// Plans the packed input-major FP16-weight browser Q/K/V kernel.
///
/// The kernel computes all three projections inside one 16-row by 32-column
/// workgroup tile, so the dispatch Z dimension is one. Activations, biases,
/// accumulators, and the three output planes remain F32.
pub fn plan_vision_qkv_fused_f16_weight_geometry(
    tokens: u32,
    input_width: u32,
    output_width: u32,
    target: VisionQkvFusedTargetLimits,
) -> Result<VisionQkvFusedPlan, InvocationError> {
    if input_width % 4 != 0 || output_width % 4 != 0 {
        return Err(invalid_fusion_target(
            "packed FP16-weight Q/K/V requires input and output widths divisible by four",
        ));
    }
    plan_vision_qkv_fused_geometry_with_storage(
        tokens,
        input_width,
        output_width,
        target,
        KernelId::VisionQkvFusedF16Weights,
        DecoderWeightStorage::F16,
        VISION_QKV_FUSED_F16_WEIGHT_ROW_TILE,
        VISION_QKV_FUSED_F16_WEIGHT_COLUMN_TILE,
        1,
    )
}

fn plan_vision_qkv_fused_geometry_with_storage(
    tokens: u32,
    input_width: u32,
    output_width: u32,
    target: VisionQkvFusedTargetLimits,
    kernel: KernelId,
    weight_storage: DecoderWeightStorage,
    row_tile: u32,
    column_tile: u32,
    dispatch_planes: u32,
) -> Result<VisionQkvFusedPlan, InvocationError> {
    require_dimensions(&[tokens, input_width, output_width])?;

    let input_elements = checked_elements(tokens, input_width)?;
    let weight_elements = checked_elements(output_width, input_width)?;
    let bias_elements = u64::from(output_width);
    let plane_elements = checked_elements(tokens, output_width)?;
    let input_bytes = checked_output_bytes(input_elements)?;
    let weight_bytes = weight_elements
        .checked_mul(weight_storage.bytes_per_element())
        .ok_or_else(overflow)?;
    let bias_bytes = checked_output_bytes(bias_elements)?;
    let plane_bytes = checked_output_bytes(plane_elements)?;
    if input_elements - 1 > u64::from(u32::MAX) || weight_elements - 1 > u64::from(u32::MAX) {
        return Err(overflow());
    }

    let alignment = u64::from(target.min_storage_buffer_offset_alignment);
    if alignment < 4 || !alignment.is_power_of_two() {
        return Err(invalid_fusion_target(
            "minimum storage-buffer offset alignment must be a power of two and at least four bytes",
        ));
    }
    let plane_stride_bytes = checked_align_up(plane_bytes, alignment)?;
    let plane_stride_elements = u32::try_from(plane_stride_bytes / 4).map_err(|_| overflow())?;
    let maximum_output_index = u64::from(plane_stride_elements)
        .checked_mul(2)
        .and_then(|offset| offset.checked_add(plane_elements - 1))
        .ok_or_else(overflow)?;
    if maximum_output_index > u64::from(u32::MAX) {
        return Err(overflow());
    }
    let physical_bytes = plane_stride_bytes.checked_mul(3).ok_or_else(overflow)?;
    let output_elements = usize::try_from(physical_bytes / 4).map_err(|_| overflow())?;
    let dispatch = [
        ceil_div(output_width, column_tile),
        ceil_div(tokens, row_tile),
        dispatch_planes,
    ];

    if dispatch
        .iter()
        .any(|dimension| *dimension > MAX_WEBGPU_DISPATCH_DIMENSION)
    {
        return Err(overflow());
    }
    if target.max_storage_buffers_per_shader_stage < VISION_QKV_FUSED_STORAGE_BINDING_COUNT {
        return Err(invalid_fusion_target(format!(
            "fused Q/K/V requires {VISION_QKV_FUSED_STORAGE_BINDING_COUNT} storage bindings, but the target exposes {}",
            target.max_storage_buffers_per_shader_stage
        )));
    }
    if dispatch
        .iter()
        .any(|dimension| *dimension > target.max_compute_workgroups_per_dimension)
    {
        return Err(invalid_fusion_target(format!(
            "fused Q/K/V dispatch {dispatch:?} exceeds the target per-dimension workgroup limit {}",
            target.max_compute_workgroups_per_dimension
        )));
    }
    for (label, bytes) in [
        ("input", input_bytes),
        ("projection weight", weight_bytes),
        ("projection bias", bias_bytes),
        ("physical output", physical_bytes),
    ] {
        if bytes > target.max_storage_buffer_binding_size {
            return Err(invalid_fusion_target(format!(
                "fused Q/K/V {label} buffer of {bytes} bytes exceeds max_storage_buffer_binding_size {}",
                target.max_storage_buffer_binding_size
            )));
        }
        if bytes > target.max_buffer_size {
            return Err(invalid_fusion_target(format!(
                "fused Q/K/V {label} buffer of {bytes} bytes exceeds max_buffer_size {}",
                target.max_buffer_size
            )));
        }
    }

    let query = VisionQkvFusedOutputSlice {
        offset: 0,
        size: plane_bytes,
    };
    let key = VisionQkvFusedOutputSlice {
        offset: plane_stride_bytes,
        size: plane_bytes,
    };
    let value = VisionQkvFusedOutputSlice {
        offset: plane_stride_bytes.checked_mul(2).ok_or_else(overflow)?,
        size: plane_bytes,
    };
    Ok(VisionQkvFusedPlan {
        invocation: InvocationPlan {
            kernel,
            output_elements,
            output_bytes: physical_bytes,
            workgroup_size: [8, 8, 1],
            dispatch,
        },
        uniform_words: [tokens, input_width, output_width, plane_stride_elements],
        output_layout: VisionQkvFusedOutputLayout {
            plane_elements,
            plane_bytes,
            plane_stride_bytes,
            physical_bytes,
            query,
            key,
            value,
        },
    })
}

#[derive(Clone, Copy, Debug)]
pub struct VisionLinearParameters<'a> {
    pub weight: &'a [f32],
    pub bias: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
pub struct VisionLayerNormParameters<'a> {
    pub weight: &'a [f32],
    pub bias: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
pub struct VisionEncoderLayerParameters<'a> {
    pub norm1: VisionLayerNormParameters<'a>,
    pub query: VisionLinearParameters<'a>,
    pub key: VisionLinearParameters<'a>,
    pub value: VisionLinearParameters<'a>,
    pub attention_output: VisionLinearParameters<'a>,
    pub norm2: VisionLayerNormParameters<'a>,
    pub mlp_fc1: VisionLinearParameters<'a>,
    pub mlp_fc2: VisionLinearParameters<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct VisionEncoderLayerInvocation<'a> {
    pub tokens: u32,
    pub hidden_size: u32,
    pub attention_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub layer_norm_epsilon: f32,
    pub input: &'a [f32],
    pub cu_seqlens: &'a [u32],
    pub parameters: VisionEncoderLayerParameters<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct VisionEncoderLayerGeometry<'a> {
    pub tokens: u32,
    pub hidden_size: u32,
    pub attention_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub layer_norm_epsilon: f32,
    pub cu_seqlens: &'a [u32],
}

#[derive(Clone, Copy, Debug)]
pub struct VisionEncoderStackInvocation<'a> {
    pub tokens: u32,
    pub hidden_size: u32,
    pub attention_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub layer_norm_epsilon: f32,
    pub input: &'a [f32],
    pub cu_seqlens: &'a [u32],
    pub layer_parameters: &'a [VisionEncoderLayerParameters<'a>],
    pub post_norm: VisionLayerNormParameters<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectorParameters<'a> {
    pub pre_norm: VisionLayerNormParameters<'a>,
    pub linear1: VisionLinearParameters<'a>,
    pub linear2: VisionLinearParameters<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectorInvocation<'a> {
    pub hidden_size: u32,
    pub output_size: u32,
    pub layer_norm_epsilon: f32,
    pub input: &'a [f32],
    pub image_grid_thw: &'a [[u32; 3]],
    pub parameters: ProjectorParameters<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectorGeometry<'a> {
    pub hidden_size: u32,
    pub output_size: u32,
    pub layer_norm_epsilon: f32,
    pub image_grid_thw: &'a [[u32; 3]],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnedVisionLinearParameters {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl OwnedVisionLinearParameters {
    #[must_use]
    pub fn borrowed(&self) -> VisionLinearParameters<'_> {
        VisionLinearParameters {
            weight: &self.weight,
            bias: &self.bias,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnedVisionLayerNormParameters {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl OwnedVisionLayerNormParameters {
    #[must_use]
    pub fn borrowed(&self) -> VisionLayerNormParameters<'_> {
        VisionLayerNormParameters {
            weight: &self.weight,
            bias: &self.bias,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnedProjectorParameters {
    pub pre_norm: OwnedVisionLayerNormParameters,
    pub linear1: OwnedVisionLinearParameters,
    pub linear2: OwnedVisionLinearParameters,
}

impl OwnedProjectorParameters {
    #[must_use]
    pub fn borrowed(&self) -> ProjectorParameters<'_> {
        ProjectorParameters {
            pre_norm: self.pre_norm.borrowed(),
            linear1: self.linear1.borrowed(),
            linear2: self.linear2.borrowed(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnedProjectorInvocation {
    pub hidden_size: u32,
    pub output_size: u32,
    pub layer_norm_epsilon: f32,
    pub input: Vec<f32>,
    pub image_grid_thw: Vec<[u32; 3]>,
    pub parameters: OwnedProjectorParameters,
}

impl OwnedProjectorInvocation {
    #[must_use]
    pub fn borrowed(&self) -> ProjectorInvocation<'_> {
        ProjectorInvocation {
            hidden_size: self.hidden_size,
            output_size: self.output_size,
            layer_norm_epsilon: self.layer_norm_epsilon,
            input: &self.input,
            image_grid_thw: &self.image_grid_thw,
            parameters: self.parameters.borrowed(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnedVisionEncoderLayerParameters {
    pub norm1: OwnedVisionLayerNormParameters,
    pub query: OwnedVisionLinearParameters,
    pub key: OwnedVisionLinearParameters,
    pub value: OwnedVisionLinearParameters,
    pub attention_output: OwnedVisionLinearParameters,
    pub norm2: OwnedVisionLayerNormParameters,
    pub mlp_fc1: OwnedVisionLinearParameters,
    pub mlp_fc2: OwnedVisionLinearParameters,
}

impl OwnedVisionEncoderLayerParameters {
    #[must_use]
    pub fn borrowed(&self) -> VisionEncoderLayerParameters<'_> {
        VisionEncoderLayerParameters {
            norm1: self.norm1.borrowed(),
            query: self.query.borrowed(),
            key: self.key.borrowed(),
            value: self.value.borrowed(),
            attention_output: self.attention_output.borrowed(),
            norm2: self.norm2.borrowed(),
            mlp_fc1: self.mlp_fc1.borrowed(),
            mlp_fc2: self.mlp_fc2.borrowed(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnedVisionEncoderLayerInvocation {
    pub tokens: u32,
    pub hidden_size: u32,
    pub attention_heads: u32,
    pub head_dim: u32,
    pub intermediate_size: u32,
    pub layer_norm_epsilon: f32,
    pub input: Vec<f32>,
    pub cu_seqlens: Vec<u32>,
    pub parameters: OwnedVisionEncoderLayerParameters,
}

impl OwnedVisionEncoderLayerInvocation {
    #[must_use]
    pub fn borrowed(&self) -> VisionEncoderLayerInvocation<'_> {
        VisionEncoderLayerInvocation {
            tokens: self.tokens,
            hidden_size: self.hidden_size,
            attention_heads: self.attention_heads,
            head_dim: self.head_dim,
            intermediate_size: self.intermediate_size,
            layer_norm_epsilon: self.layer_norm_epsilon,
            input: &self.input,
            cu_seqlens: &self.cu_seqlens,
            parameters: self.parameters.borrowed(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionEncoderLayerStage {
    Norm1,
    Query,
    Key,
    Value,
    AttentionContext,
    AttentionOutput,
    AttentionResidual,
    Norm2,
    MlpFc1,
    MlpActivation,
    MlpOutput,
    Output,
}

impl VisionEncoderLayerStage {
    pub const ALL: [Self; 12] = [
        Self::Norm1,
        Self::Query,
        Self::Key,
        Self::Value,
        Self::AttentionContext,
        Self::AttentionOutput,
        Self::AttentionResidual,
        Self::Norm2,
        Self::MlpFc1,
        Self::MlpActivation,
        Self::MlpOutput,
        Self::Output,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Norm1 => "norm1",
            Self::Query => "query",
            Self::Key => "key",
            Self::Value => "value",
            Self::AttentionContext => "attention-context",
            Self::AttentionOutput => "attention-output",
            Self::AttentionResidual => "attention-residual",
            Self::Norm2 => "norm2",
            Self::MlpFc1 => "mlp-fc1",
            Self::MlpActivation => "mlp-activation",
            Self::MlpOutput => "mlp-output",
            Self::Output => "output",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionRopeSpecialization {
    Identity,
    Spatial2d,
}

#[derive(Clone, Copy, Debug)]
pub struct VisionRope2dDescriptor<'a> {
    pub tokens: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub cos: &'a [f32],
    pub sin: &'a [f32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionRope2dPlan {
    pub specialization: VisionRopeSpecialization,
    pub invocation: InvocationPlan,
    pub table_elements: usize,
    pub table_bytes: u64,
    pub uniform_words: [u32; 4],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionEncoderLayerExecutionStage {
    Norm1,
    Query,
    Key,
    Value,
    SpatialRope,
    AttentionContext,
    AttentionOutput,
    AttentionResidual,
    Norm2,
    MlpFc1,
    MlpActivation,
    MlpOutput,
    Output,
}

impl VisionEncoderLayerExecutionStage {
    pub const SPATIAL_2D: [Self; 13] = [
        Self::Norm1,
        Self::Query,
        Self::Key,
        Self::Value,
        Self::SpatialRope,
        Self::AttentionContext,
        Self::AttentionOutput,
        Self::AttentionResidual,
        Self::Norm2,
        Self::MlpFc1,
        Self::MlpActivation,
        Self::MlpOutput,
        Self::Output,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionEncoderLayerSpatial2dPlan {
    pub base: VisionEncoderLayerPlan,
    pub rope_specialization: VisionRopeSpecialization,
    pub rope: VisionRope2dPlan,
    pub execution_stages: [VisionEncoderLayerExecutionStage; 13],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionEncoderStackSpatial2dPlan {
    pub layer_count: usize,
    pub layer_dispatch_count: usize,
    pub rope_dispatch_count: usize,
    pub post_norm_dispatch_count: usize,
    pub dispatch_count: usize,
    pub rope_table_buffer_count: usize,
    pub rope_table_upload_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionLayerReadback {
    OutputOnly,
    AllStages,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectorStage {
    PreNorm,
    Merge,
    Linear1,
    Activation,
    Linear2,
}

impl ProjectorStage {
    pub const ALL: [Self; 5] = [
        Self::PreNorm,
        Self::Merge,
        Self::Linear1,
        Self::Activation,
        Self::Linear2,
    ];

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PreNorm => "pre-norm",
            Self::Merge => "merge",
            Self::Linear1 => "linear1",
            Self::Activation => "activation",
            Self::Linear2 => "linear2",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProjectorReadback {
    OutputOnly,
    AllStages,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectorDispatch {
    pub stage: ProjectorStage,
    pub invocation: InvocationPlan,
    pub uniform_words: [u32; 4],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProjectorPlan {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub merged_width: u32,
    pub source_token_indices: Vec<u32>,
    pub dispatches: [ProjectorDispatch; 5],
    pub resident_intermediate_bytes: u64,
    pub resident_weight_bytes: u64,
}

impl ProjectorPlan {
    #[must_use]
    pub const fn readback_bytes(&self, readback: ProjectorReadback) -> u64 {
        match readback {
            ProjectorReadback::OutputOnly => self.dispatches[4].invocation.output_bytes,
            ProjectorReadback::AllStages => self.resident_intermediate_bytes,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionEncoderLayerDispatch {
    pub stage: VisionEncoderLayerStage,
    pub invocation: InvocationPlan,
    pub uniform_words: [u32; 4],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionEncoderLayerPlan {
    pub rope_specialization: VisionRopeSpecialization,
    pub dispatches: [VisionEncoderLayerDispatch; 12],
    pub resident_intermediate_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionEncoderStackPlan {
    pub rope_specialization: VisionRopeSpecialization,
    pub layer_count: usize,
    pub layer_dispatches: [VisionEncoderLayerDispatch; 12],
    pub post_norm_dispatch: InvocationPlan,
    pub post_norm_uniform_words: [u32; 4],
    pub checkpoint_layers: Vec<usize>,
    pub dispatch_count: usize,
    pub compute_pass_count: usize,
    pub activation_buffer_count: usize,
    pub activation_arena_bytes: u64,
    pub resident_weight_bytes: u64,
    pub readback_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionStackActivationStrategy {
    SeparateBuffers,
    StaticArenaNoAlias,
    StaticArenaAlias,
}

/// Configuration for lowering a vision-stack plan into static activation
/// layout metadata. This does not allocate buffers or execute any dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionStackActivationLayoutConfig {
    pub allow_aliasing: bool,
    pub storage_buffer_offset_alignment: u64,
    pub arena_alignment: u64,
}

/// One verified scratch slice in a lowered vision-stack activation layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionStackScratchAllocation {
    pub stage: VisionEncoderLayerStage,
    pub offset: u64,
    pub size: u64,
    pub alignment: u64,
    pub first_write: u32,
    pub last_use: u32,
}

/// Verified activation-layout metadata for a vision stack. This describes
/// future buffer allocation and binding; it performs neither at runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionStackActivationLayout {
    pub scratch_allocations: Vec<VisionStackScratchAllocation>,
    pub scratch_arena_bytes: u64,
    pub main_buffers_bytes: u64,
    pub total_activation_bytes: u64,
    pub physical_buffer_count: usize,
}

const VISION_STACK_SCRATCH_SCHEDULE: [(VisionEncoderLayerStage, u32, u32); 11] = [
    (VisionEncoderLayerStage::Norm1, 0, 3),
    (VisionEncoderLayerStage::Query, 1, 4),
    (VisionEncoderLayerStage::Key, 2, 4),
    (VisionEncoderLayerStage::Value, 3, 4),
    (VisionEncoderLayerStage::AttentionContext, 4, 5),
    (VisionEncoderLayerStage::AttentionOutput, 5, 6),
    (VisionEncoderLayerStage::AttentionResidual, 6, 11),
    (VisionEncoderLayerStage::Norm2, 7, 8),
    (VisionEncoderLayerStage::MlpFc1, 8, 9),
    (VisionEncoderLayerStage::MlpActivation, 9, 10),
    (VisionEncoderLayerStage::MlpOutput, 10, 11),
];

impl VisionRope2dDescriptor<'_> {
    pub fn plan(&self) -> Result<VisionRope2dPlan, InvocationError> {
        require_dimensions(&[self.tokens, self.heads])?;
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(4) {
            return Err(InvocationError::new(
                InvocationErrorCode::InvalidRotaryDimension,
                "vision RoPE head_dim must be positive and divisible by four",
            ));
        }
        if self.head_dim > MAX_VISION_HEAD_DIM {
            return Err(InvocationError::new(
                InvocationErrorCode::UnsupportedHeadDimension,
                format!(
                    "vision RoPE head_dim {} exceeds the fixed limit {MAX_VISION_HEAD_DIM}",
                    self.head_dim
                ),
            ));
        }
        let pair_count = self.head_dim / 2;
        let table_elements_u64 = checked_elements(self.tokens, pair_count)?;
        require_len(self.cos.len(), table_elements_u64)?;
        require_len(self.sin.len(), table_elements_u64)?;
        require_finite(self.cos)?;
        require_finite(self.sin)?;

        let tensor_elements = checked_tensor_elements(&[self.tokens, self.heads, self.head_dim])?;
        let output_elements_u64 = tensor_elements.checked_mul(2).ok_or_else(overflow)?;
        let work_items_u64 = checked_tensor_elements(&[self.tokens, self.heads, pair_count])?;
        let work_items = u32::try_from(work_items_u64).map_err(|_| overflow())?;
        let dispatch_x = ceil_div(work_items, 64);
        if dispatch_x > MAX_WEBGPU_DISPATCH_DIMENSION {
            return Err(InvocationError::new(
                InvocationErrorCode::InvalidVisionGeometry,
                "vision RoPE dispatch exceeds the WebGPU per-dimension limit",
            ));
        }

        Ok(VisionRope2dPlan {
            specialization: VisionRopeSpecialization::Spatial2d,
            invocation: InvocationPlan {
                kernel: KernelId::VisionRope2dF32,
                output_elements: usize::try_from(output_elements_u64).map_err(|_| overflow())?,
                output_bytes: checked_output_bytes(output_elements_u64)?,
                workgroup_size: [64, 1, 1],
                dispatch: [dispatch_x, 1, 1],
            },
            table_elements: usize::try_from(table_elements_u64).map_err(|_| overflow())?,
            table_bytes: checked_output_bytes(table_elements_u64)?,
            uniform_words: [self.tokens, self.heads, self.head_dim, 0],
        })
    }
}

impl VisionEncoderLayerPlan {
    /// Lowers this layer's dispatch plan into the same verified static-arena
    /// metadata used by a full vision stack. The returned value does not
    /// allocate, bind, or execute runtime buffers.
    pub fn stack_activation_layout(
        &self,
        config: VisionStackActivationLayoutConfig,
    ) -> Result<VisionStackActivationLayout, InvocationError> {
        vision_stack_activation_layout(&self.dispatches, config)
    }

    pub fn with_spatial_rope(
        self,
        rope: VisionRope2dPlan,
    ) -> Result<VisionEncoderLayerSpatial2dPlan, InvocationError> {
        let [tokens, hidden_size, _, _] = self.dispatches[0].uniform_words;
        let [attention_tokens, heads, head_dim, _] = self.dispatches[4].uniform_words;
        if rope.specialization != VisionRopeSpecialization::Spatial2d
            || rope.uniform_words[..3] != [tokens, heads, head_dim]
            || attention_tokens != tokens
            || hidden_size != heads.checked_mul(head_dim).ok_or_else(overflow)?
        {
            return Err(InvocationError::new(
                InvocationErrorCode::InvalidVisionGeometry,
                "vision RoPE geometry does not match the encoder layer",
            ));
        }
        Ok(VisionEncoderLayerSpatial2dPlan {
            base: self,
            rope_specialization: VisionRopeSpecialization::Spatial2d,
            rope,
            execution_stages: VisionEncoderLayerExecutionStage::SPATIAL_2D,
        })
    }
}

impl VisionEncoderLayerSpatial2dPlan {
    pub fn stack_plan(
        &self,
        layer_count: usize,
    ) -> Result<VisionEncoderStackSpatial2dPlan, InvocationError> {
        if layer_count == 0 {
            return Err(InvocationError::new(
                InvocationErrorCode::ZeroDimension,
                "vision encoder stack must contain at least one layer",
            ));
        }
        let layer_dispatch_count = layer_count
            .checked_mul(self.execution_stages.len())
            .ok_or_else(overflow)?;
        let post_norm_dispatch_count = 1;
        let dispatch_count = layer_dispatch_count
            .checked_add(post_norm_dispatch_count)
            .ok_or_else(overflow)?;
        Ok(VisionEncoderStackSpatial2dPlan {
            layer_count,
            layer_dispatch_count,
            rope_dispatch_count: layer_count,
            post_norm_dispatch_count,
            dispatch_count,
            rope_table_buffer_count: 2,
            rope_table_upload_count: 2,
        })
    }
}

impl VisionEncoderStackPlan {
    /// Lowers the existing dispatch plan into verified static-arena metadata.
    /// The returned value does not allocate, bind, or execute runtime buffers.
    pub fn activation_layout(
        &self,
        config: VisionStackActivationLayoutConfig,
    ) -> Result<VisionStackActivationLayout, InvocationError> {
        vision_stack_activation_layout(&self.layer_dispatches, config)
    }
}

fn vision_stack_activation_layout(
    dispatches: &[VisionEncoderLayerDispatch; 12],
    config: VisionStackActivationLayoutConfig,
) -> Result<VisionStackActivationLayout, InvocationError> {
    validate_activation_layout_stage_order(dispatches)?;

    let lifetimes = VISION_STACK_SCRATCH_SCHEDULE
        .iter()
        .enumerate()
        .map(|(index, &(stage, first_write, last_use))| TensorLifetime {
            id: stage.as_str().to_owned(),
            byte_size: dispatches[index].invocation.output_bytes,
            alignment: 4,
            first_write,
            last_use,
            stage_label: Some(stage.as_str().to_owned()),
        })
        .collect::<Vec<_>>();
    let arena_config = ArenaConfig {
        allow_aliasing: config.allow_aliasing,
        arena_alignment: config.arena_alignment,
        base_alignment: config.storage_buffer_offset_alignment,
    };
    let arena_plan =
        plan_static_arena(&lifetimes, arena_config).map_err(map_activation_layout_arena_error)?;
    verify_static_arena_plan(&lifetimes, &arena_plan, arena_config)
        .map_err(map_activation_layout_arena_error)?;

    if arena_plan.allocations.len() != VISION_STACK_SCRATCH_SCHEDULE.len() {
        return Err(invalid_activation_layout(format!(
            "verified scratch allocation count {} does not match expected stage count {}",
            arena_plan.allocations.len(),
            VISION_STACK_SCRATCH_SCHEDULE.len()
        )));
    }

    let mut scratch_allocations = Vec::with_capacity(arena_plan.allocations.len());
    for (index, (allocation, &(stage, first_write, last_use))) in arena_plan
        .allocations
        .iter()
        .zip(VISION_STACK_SCRATCH_SCHEDULE.iter())
        .enumerate()
    {
        let expected_id = stage.as_str();
        let expected_size = dispatches[index].invocation.output_bytes;
        if allocation.id != expected_id
            || allocation.stage_label.as_deref() != Some(expected_id)
            || allocation.size != expected_size
            || allocation.first_write != first_write
            || allocation.last_use != last_use
        {
            return Err(invalid_activation_layout(format!(
                "verified scratch allocation at index {index} does not match stage {expected_id}"
            )));
        }
        scratch_allocations.push(VisionStackScratchAllocation {
            stage,
            offset: allocation.offset,
            size: allocation.size,
            alignment: allocation.alignment,
            first_write: allocation.first_write,
            last_use: allocation.last_use,
        });
    }

    let hidden_output_bytes = dispatches[11].invocation.output_bytes;
    let main_buffers_bytes = hidden_output_bytes.checked_mul(2).ok_or_else(|| {
        activation_layout_overflow("two main activation buffers overflow u64 byte bounds")
    })?;
    let total_activation_bytes = arena_plan
        .arena_bytes
        .checked_add(main_buffers_bytes)
        .ok_or_else(|| {
            activation_layout_overflow(
                "scratch arena plus main activation buffers overflow u64 byte bounds",
            )
        })?;

    Ok(VisionStackActivationLayout {
        scratch_allocations,
        scratch_arena_bytes: arena_plan.arena_bytes,
        main_buffers_bytes,
        total_activation_bytes,
        physical_buffer_count: 3,
    })
}

fn validate_activation_layout_stage_order(
    dispatches: &[VisionEncoderLayerDispatch; 12],
) -> Result<(), InvocationError> {
    for (index, (&(expected, _, _), dispatch)) in VISION_STACK_SCRATCH_SCHEDULE
        .iter()
        .zip(dispatches.iter())
        .enumerate()
    {
        if dispatch.stage != expected {
            return Err(invalid_activation_layout(format!(
                "vision layer dispatch {index} must be stage {}, found {}",
                expected.as_str(),
                dispatch.stage.as_str()
            )));
        }
    }
    let output_index = VISION_STACK_SCRATCH_SCHEDULE.len();
    let output_stage = dispatches[output_index].stage;
    if output_stage != VisionEncoderLayerStage::Output {
        return Err(invalid_activation_layout(format!(
            "vision layer dispatch {output_index} must be stage {}, found {}",
            VisionEncoderLayerStage::Output.as_str(),
            output_stage.as_str()
        )));
    }
    Ok(())
}

impl ProjectorInvocation<'_> {
    pub fn plan(&self) -> Result<ProjectorPlan, InvocationError> {
        require_dimensions(&[self.hidden_size, self.output_size])?;
        let merged_width = self.hidden_size.checked_mul(4).ok_or_else(overflow)?;
        let (input_tokens, _) = projector_token_counts(self.image_grid_thw)?;

        require_len(
            self.input.len(),
            checked_elements(input_tokens, self.hidden_size)?,
        )?;
        require_finite(self.input)?;
        validate_vision_norm(self.parameters.pre_norm, self.hidden_size)?;
        validate_vision_linear(self.parameters.linear1, merged_width, merged_width)?;
        validate_vision_linear(self.parameters.linear2, merged_width, self.output_size)?;

        ProjectorGeometry {
            hidden_size: self.hidden_size,
            output_size: self.output_size,
            layer_norm_epsilon: self.layer_norm_epsilon,
            image_grid_thw: self.image_grid_thw,
        }
        .plan_with_storage(DecoderWeightStorage::F32)
    }
}

impl ProjectorGeometry<'_> {
    pub fn plan_full_f16(&self) -> Result<ProjectorPlan, InvocationError> {
        self.plan_with_storage(DecoderWeightStorage::F16)
    }

    fn plan_with_storage(
        &self,
        storage: DecoderWeightStorage,
    ) -> Result<ProjectorPlan, InvocationError> {
        require_dimensions(&[self.hidden_size, self.output_size])?;
        require_epsilon(self.layer_norm_epsilon)?;
        if storage == DecoderWeightStorage::F16
            && (self.hidden_size % 4 != 0 || self.output_size % 4 != 0)
        {
            return Err(InvocationError::new(
                InvocationErrorCode::InvalidProjectorGeometry,
                "FP16 projector hidden_size and output_size must be multiples of 4",
            ));
        }
        let merged_width = self.hidden_size.checked_mul(4).ok_or_else(overflow)?;
        let (input_tokens, output_tokens) = projector_token_counts(self.image_grid_thw)?;
        let source_token_indices =
            projector_source_token_indices(self.image_grid_thw, input_tokens, output_tokens)?;
        let hidden_elements = checked_elements(input_tokens, self.hidden_size)?;
        let hidden_bytes = storage
            .storage_bytes(hidden_elements)
            .ok_or_else(overflow)?;
        let merged_elements = checked_elements(output_tokens, merged_width)?;
        let merged_bytes = storage
            .storage_bytes(merged_elements)
            .ok_or_else(overflow)?;
        let output_elements = checked_elements(output_tokens, self.output_size)?;
        let output_bytes = storage
            .storage_bytes(output_elements)
            .ok_or_else(overflow)?;
        let merged_element_count = u32::try_from(merged_elements).map_err(|_| overflow())?;
        let packed_divisor = if storage == DecoderWeightStorage::F16 {
            4
        } else {
            1
        };
        let (merged_dispatch, merged_row_stride) = bounded_linear_dispatch(
            ceil_div(merged_element_count, packed_divisor),
            64,
        );
        let (norm_kernel, merge_kernel, projection_kernel, activation_kernel) =
            match storage {
                DecoderWeightStorage::F32 => (
                    KernelId::LayerNormF32,
                    KernelId::ProjectorMerge2x2F32,
                    KernelId::VisionPatchProjectionF32,
                    KernelId::GeluErfF32,
                ),
                DecoderWeightStorage::F16 => (
                    KernelId::LayerNormF16,
                    KernelId::ProjectorMerge2x2F16,
                    KernelId::LinearProjectionF16,
                    KernelId::GeluErfF16,
                ),
            };

        let dispatches = [
            projector_dispatch(
                ProjectorStage::PreNorm,
                norm_kernel,
                hidden_elements,
                hidden_bytes,
                [64, 1, 1],
                [ceil_div(input_tokens, 64), 1, 1],
                [
                    input_tokens,
                    self.hidden_size,
                    self.layer_norm_epsilon.to_bits(),
                    0,
                ],
            )?,
            projector_dispatch(
                ProjectorStage::Merge,
                merge_kernel,
                merged_elements,
                merged_bytes,
                [64, 1, 1],
                merged_dispatch,
                [
                    output_tokens,
                    self.hidden_size,
                    merged_element_count,
                    merged_row_stride,
                ],
            )?,
            projector_dispatch(
                ProjectorStage::Linear1,
                projection_kernel,
                merged_elements,
                merged_bytes,
                [8, 8, 1],
                [
                    ceil_div(merged_width, LINEAR_PROJECTION_TILE),
                    ceil_div(output_tokens, LINEAR_PROJECTION_TILE),
                    1,
                ],
                [output_tokens, merged_width, merged_width, 0],
            )?,
            projector_dispatch(
                ProjectorStage::Activation,
                activation_kernel,
                merged_elements,
                merged_bytes,
                [64, 1, 1],
                merged_dispatch,
                [merged_element_count, merged_row_stride, 0, 0],
            )?,
            projector_dispatch(
                ProjectorStage::Linear2,
                projection_kernel,
                output_elements,
                output_bytes,
                [8, 8, 1],
                [
                    ceil_div(self.output_size, LINEAR_PROJECTION_TILE),
                    ceil_div(output_tokens, LINEAR_PROJECTION_TILE),
                    1,
                ],
                [output_tokens, merged_width, self.output_size, 0],
            )?,
        ];
        let resident_intermediate_bytes =
            dispatches.iter().try_fold(0_u64, |bytes, dispatch| {
                bytes
                    .checked_add(dispatch.invocation.output_bytes)
                    .ok_or_else(overflow)
            })?;
        let resident_weight_elements = u64::from(self.hidden_size)
            .checked_mul(2)
            .and_then(|elements| {
                u64::from(merged_width)
                    .checked_mul(u64::from(merged_width))
                    .and_then(|matrix| elements.checked_add(matrix))
            })
            .and_then(|elements| elements.checked_add(u64::from(merged_width)))
            .and_then(|elements| {
                u64::from(self.output_size)
                    .checked_mul(u64::from(merged_width))
                    .and_then(|matrix| elements.checked_add(matrix))
            })
            .and_then(|elements| elements.checked_add(u64::from(self.output_size)))
            .ok_or_else(overflow)?;
        let resident_weight_bytes = storage
            .storage_bytes(resident_weight_elements)
            .ok_or_else(overflow)?;

        Ok(ProjectorPlan {
            input_tokens,
            output_tokens,
            merged_width,
            source_token_indices,
            dispatches,
            resident_intermediate_bytes,
            resident_weight_bytes,
        })
    }
}

impl VisionEncoderLayerInvocation<'_> {
    pub fn plan(&self) -> Result<VisionEncoderLayerPlan, InvocationError> {
        let plan = VisionEncoderLayerGeometry {
            tokens: self.tokens,
            hidden_size: self.hidden_size,
            attention_heads: self.attention_heads,
            head_dim: self.head_dim,
            intermediate_size: self.intermediate_size,
            layer_norm_epsilon: self.layer_norm_epsilon,
            cu_seqlens: self.cu_seqlens,
        }
        .plan()?;
        let hidden_elements =
            u64::try_from(plan.dispatches[0].invocation.output_elements).map_err(|_| overflow())?;
        require_len(self.input.len(), hidden_elements)?;
        require_finite(self.input)?;
        validate_vision_norm(self.parameters.norm1, self.hidden_size)?;
        validate_vision_linear(self.parameters.query, self.hidden_size, self.hidden_size)?;
        validate_vision_linear(self.parameters.key, self.hidden_size, self.hidden_size)?;
        validate_vision_linear(self.parameters.value, self.hidden_size, self.hidden_size)?;
        validate_vision_linear(
            self.parameters.attention_output,
            self.hidden_size,
            self.hidden_size,
        )?;
        validate_vision_norm(self.parameters.norm2, self.hidden_size)?;
        validate_vision_linear(
            self.parameters.mlp_fc1,
            self.hidden_size,
            self.intermediate_size,
        )?;
        validate_vision_linear(
            self.parameters.mlp_fc2,
            self.intermediate_size,
            self.hidden_size,
        )?;
        Ok(plan)
    }
}

impl VisionEncoderLayerGeometry<'_> {
    pub fn plan(&self) -> Result<VisionEncoderLayerPlan, InvocationError> {
        self.plan_with_matrix_weight_storage(DecoderWeightStorage::F32)
    }

    pub fn plan_with_matrix_weight_storage(
        &self,
        matrix_weight_storage: DecoderWeightStorage,
    ) -> Result<VisionEncoderLayerPlan, InvocationError> {
        self.plan_with_matrix_weight_storage_and_layout(
            matrix_weight_storage,
            LinearWeightLayout::OutputMajor,
        )
    }

    pub fn plan_with_matrix_weight_storage_and_layout(
        &self,
        matrix_weight_storage: DecoderWeightStorage,
        matrix_weight_layout: LinearWeightLayout,
    ) -> Result<VisionEncoderLayerPlan, InvocationError> {
        self.plan_with_precision(VisionEncoderPrecision::legacy(
            matrix_weight_storage,
            matrix_weight_layout,
        ))
    }

    pub fn plan_with_precision(
        &self,
        precision: VisionEncoderPrecision,
    ) -> Result<VisionEncoderLayerPlan, InvocationError> {
        let matrix_weight_storage = precision.matrix_weight_storage;
        let matrix_weight_layout = precision.matrix_weight_layout;
        let full_f16 = precision.is_full_f16();
        let legacy_f32_activations = precision.vector_weight_storage == DecoderWeightStorage::F32
            && precision.activation_storage == DecoderWeightStorage::F32;
        if !full_f16 && !legacy_f32_activations {
            return Err(InvocationError::new(
                InvocationErrorCode::InvalidVisionGeometry,
                "vision execution must use either legacy F32 activations/vectors or one coherent full FP16 matrix/vector/activation profile",
            ));
        }
        if matrix_weight_layout == LinearWeightLayout::InputMajor
            && matrix_weight_storage != DecoderWeightStorage::F16
        {
            return Err(InvocationError::new(
                InvocationErrorCode::InvalidVisionGeometry,
                "input-major linear weights require F16 matrix storage",
            ));
        }
        require_dimensions(&[
            self.tokens,
            self.hidden_size,
            self.attention_heads,
            self.head_dim,
            self.intermediate_size,
        ])?;
        if matrix_weight_storage == DecoderWeightStorage::F16
            && (self.hidden_size % 4 != 0 || self.intermediate_size % 4 != 0)
        {
            return Err(InvocationError::new(
                InvocationErrorCode::InvalidVisionGeometry,
                "FP16 vision hidden_size and intermediate_size must be a multiple of 4 for packed vec4 IO",
            ));
        }
        let projected_hidden = checked_elements(self.attention_heads, self.head_dim)?;
        if projected_hidden != u64::from(self.hidden_size) {
            return Err(InvocationError::new(
                InvocationErrorCode::InvalidVisionGeometry,
                "attention_heads * head_dim must equal hidden_size",
            ));
        }
        if self.head_dim > MAX_VISION_HEAD_DIM {
            return Err(InvocationError::new(
                InvocationErrorCode::UnsupportedHeadDimension,
                format!(
                    "vision attention head_dim {} exceeds the fixed limit {MAX_VISION_HEAD_DIM}",
                    self.head_dim
                ),
            ));
        }
        require_epsilon(self.layer_norm_epsilon)?;

        let hidden_elements = checked_elements(self.tokens, self.hidden_size)?;
        let hidden_bytes = precision
            .activation_storage
            .storage_bytes(hidden_elements)
            .ok_or_else(overflow)?;
        let intermediate_elements = checked_elements(self.tokens, self.intermediate_size)?;
        let intermediate_bytes = precision
            .activation_storage
            .storage_bytes(intermediate_elements)
            .ok_or_else(overflow)?;
        require_sequence_boundaries(self.cu_seqlens, self.tokens)?;
        let hidden_element_count = u32::try_from(hidden_elements).map_err(|_| overflow())?;
        let intermediate_element_count =
            u32::try_from(intermediate_elements).map_err(|_| overflow())?;
        let packed_element_divisor = if full_f16 { 4 } else { 1 };
        let (add_dispatch, add_row_stride) =
            bounded_linear_dispatch(ceil_div(hidden_element_count, packed_element_divisor), 64);
        let (activation_dispatch, activation_row_stride) = bounded_linear_dispatch(
            ceil_div(intermediate_element_count, packed_element_divisor),
            64,
        );
        let segment_count = u32::try_from(self.cu_seqlens.len() - 1).map_err(|_| overflow())?;
        let projection_kernel = match (full_f16, matrix_weight_storage) {
            (true, _) => KernelId::LinearProjectionF16,
            (false, DecoderWeightStorage::F32) => KernelId::VisionPatchProjectionF32,
            (false, DecoderWeightStorage::F16) => KernelId::LinearProjectionF16Weights,
        };
        let norm_kernel = if full_f16 {
            KernelId::LayerNormF16
        } else {
            KernelId::LayerNormF32
        };
        let attention_kernel = if full_f16 {
            KernelId::VisionAttentionF16
        } else {
            KernelId::VisionAttentionF32
        };
        let add_kernel = if full_f16 {
            KernelId::AddF16
        } else {
            KernelId::AddF32
        };
        let activation_kernel = if full_f16 {
            KernelId::GeluTanhF16
        } else {
            KernelId::GeluTanhF32
        };

        let norm = |stage| {
            vision_layer_dispatch(
                stage,
                norm_kernel,
                hidden_elements,
                hidden_bytes,
                [64, 1, 1],
                [ceil_div(self.tokens, 64), 1, 1],
                [
                    self.tokens,
                    self.hidden_size,
                    self.layer_norm_epsilon.to_bits(),
                    0,
                ],
            )
        };
        let hidden_linear = |stage, input_width, output_width, output_elements, output_bytes| {
            vision_layer_dispatch(
                stage,
                projection_kernel,
                output_elements,
                output_bytes,
                [8, 8, 1],
                [
                    ceil_div(
                        output_width,
                        matrix_weight_storage.linear_projection_output_columns_per_workgroup(),
                    ),
                    ceil_div(self.tokens, LINEAR_PROJECTION_TILE),
                    1,
                ],
                [
                    self.tokens,
                    input_width,
                    output_width,
                    matrix_weight_layout.uniform_word(),
                ],
            )
        };
        let add = |stage| {
            vision_layer_dispatch(
                stage,
                add_kernel,
                hidden_elements,
                hidden_bytes,
                [64, 1, 1],
                add_dispatch,
                [hidden_element_count, add_row_stride, 0, 0],
            )
        };

        let dispatches = [
            norm(VisionEncoderLayerStage::Norm1)?,
            hidden_linear(
                VisionEncoderLayerStage::Query,
                self.hidden_size,
                self.hidden_size,
                hidden_elements,
                hidden_bytes,
            )?,
            hidden_linear(
                VisionEncoderLayerStage::Key,
                self.hidden_size,
                self.hidden_size,
                hidden_elements,
                hidden_bytes,
            )?,
            hidden_linear(
                VisionEncoderLayerStage::Value,
                self.hidden_size,
                self.hidden_size,
                hidden_elements,
                hidden_bytes,
            )?,
            vision_layer_dispatch(
                VisionEncoderLayerStage::AttentionContext,
                attention_kernel,
                hidden_elements,
                hidden_bytes,
                [VISION_ATTENTION_WORKGROUP_SIZE, 1, 1],
                [
                    ceil_div(self.tokens, VISION_ATTENTION_QUERY_TILE),
                    self.attention_heads,
                    1,
                ],
                [
                    self.tokens,
                    self.attention_heads,
                    self.head_dim,
                    segment_count,
                ],
            )?,
            hidden_linear(
                VisionEncoderLayerStage::AttentionOutput,
                self.hidden_size,
                self.hidden_size,
                hidden_elements,
                hidden_bytes,
            )?,
            add(VisionEncoderLayerStage::AttentionResidual)?,
            norm(VisionEncoderLayerStage::Norm2)?,
            hidden_linear(
                VisionEncoderLayerStage::MlpFc1,
                self.hidden_size,
                self.intermediate_size,
                intermediate_elements,
                intermediate_bytes,
            )?,
            vision_layer_dispatch(
                VisionEncoderLayerStage::MlpActivation,
                activation_kernel,
                intermediate_elements,
                intermediate_bytes,
                [64, 1, 1],
                activation_dispatch,
                [intermediate_element_count, activation_row_stride, 0, 0],
            )?,
            hidden_linear(
                VisionEncoderLayerStage::MlpOutput,
                self.intermediate_size,
                self.hidden_size,
                hidden_elements,
                hidden_bytes,
            )?,
            add(VisionEncoderLayerStage::Output)?,
        ];
        let resident_intermediate_bytes = hidden_bytes
            .checked_mul(10)
            .and_then(|bytes| {
                intermediate_bytes
                    .checked_mul(2)
                    .and_then(|mlp| bytes.checked_add(mlp))
            })
            .ok_or_else(overflow)?;

        Ok(VisionEncoderLayerPlan {
            rope_specialization: VisionRopeSpecialization::Identity,
            dispatches,
            resident_intermediate_bytes,
        })
    }
}

impl VisionEncoderStackInvocation<'_> {
    pub fn plan(
        &self,
        checkpoint_layers: &[usize],
    ) -> Result<VisionEncoderStackPlan, InvocationError> {
        if self.layer_parameters.is_empty() {
            return Err(InvocationError::new(
                InvocationErrorCode::ZeroDimension,
                "vision encoder stack must contain at least one layer",
            ));
        }
        validate_checkpoint_selection(checkpoint_layers, self.layer_parameters.len())?;

        let layer_invocation = |parameters| VisionEncoderLayerInvocation {
            tokens: self.tokens,
            hidden_size: self.hidden_size,
            attention_heads: self.attention_heads,
            head_dim: self.head_dim,
            intermediate_size: self.intermediate_size,
            layer_norm_epsilon: self.layer_norm_epsilon,
            input: self.input,
            cu_seqlens: self.cu_seqlens,
            parameters,
        };
        let first_layer_plan = layer_invocation(self.layer_parameters[0]).plan()?;
        for parameters in &self.layer_parameters[1..] {
            let layer_plan = layer_invocation(*parameters).plan()?;
            debug_assert_eq!(layer_plan, first_layer_plan);
        }
        validate_vision_norm(self.post_norm, self.hidden_size)?;

        let hidden_bytes = first_layer_plan.dispatches[0].invocation.output_bytes;
        let intermediate_bytes = first_layer_plan.dispatches[8].invocation.output_bytes;
        let activation_arena_bytes = hidden_bytes
            .checked_mul(11)
            .and_then(|hidden| {
                intermediate_bytes
                    .checked_mul(2)
                    .and_then(|intermediate| hidden.checked_add(intermediate))
            })
            .ok_or_else(overflow)?;
        let resident_weight_bytes =
            self.layer_parameters
                .iter()
                .try_fold(0_u64, |bytes, parameters| {
                    vision_layer_parameter_slices(*parameters)
                        .into_iter()
                        .try_fold(bytes, checked_add_f32_slice_bytes)
                })?;
        let resident_weight_bytes = [self.post_norm.weight, self.post_norm.bias]
            .into_iter()
            .try_fold(resident_weight_bytes, checked_add_f32_slice_bytes)?;
        let readback_copies = u64::try_from(checkpoint_layers.len())
            .map_err(|_| overflow())?
            .checked_add(1)
            .ok_or_else(overflow)?;
        let readback_bytes = hidden_bytes
            .checked_mul(readback_copies)
            .ok_or_else(overflow)?;
        let dispatch_count = self
            .layer_parameters
            .len()
            .checked_mul(first_layer_plan.dispatches.len())
            .and_then(|dispatches| dispatches.checked_add(1))
            .ok_or_else(overflow)?;
        let compute_pass_count = self
            .layer_parameters
            .len()
            .checked_add(1)
            .ok_or_else(overflow)?;
        let post_norm_dispatch = first_layer_plan.dispatches[0].invocation;
        let post_norm_uniform_words = [
            self.tokens,
            self.hidden_size,
            self.layer_norm_epsilon.to_bits(),
            0,
        ];

        Ok(VisionEncoderStackPlan {
            rope_specialization: first_layer_plan.rope_specialization,
            layer_count: self.layer_parameters.len(),
            layer_dispatches: first_layer_plan.dispatches,
            post_norm_dispatch,
            post_norm_uniform_words,
            checkpoint_layers: checkpoint_layers.to_vec(),
            dispatch_count,
            compute_pass_count,
            activation_buffer_count: 13,
            activation_arena_bytes,
            resident_weight_bytes,
            readback_bytes,
        })
    }
}

fn vision_layer_parameter_slices(parameters: VisionEncoderLayerParameters<'_>) -> [&[f32]; 16] {
    [
        parameters.norm1.weight,
        parameters.norm1.bias,
        parameters.query.weight,
        parameters.query.bias,
        parameters.key.weight,
        parameters.key.bias,
        parameters.value.weight,
        parameters.value.bias,
        parameters.attention_output.weight,
        parameters.attention_output.bias,
        parameters.norm2.weight,
        parameters.norm2.bias,
        parameters.mlp_fc1.weight,
        parameters.mlp_fc1.bias,
        parameters.mlp_fc2.weight,
        parameters.mlp_fc2.bias,
    ]
}

fn projector_token_counts(grids: &[[u32; 3]]) -> Result<(u32, u32), InvocationError> {
    if grids.is_empty() {
        return Err(invalid_projector_geometry(
            "image_grid_thw must contain at least one image",
        ));
    }
    let mut input_tokens = 0_u32;
    let mut output_tokens = 0_u32;
    for &[temporal, height, width] in grids {
        if temporal == 0 || height == 0 || width == 0 {
            return Err(invalid_projector_geometry(
                "every projector grid dimension must be nonzero",
            ));
        }
        if !height.is_multiple_of(2) || !width.is_multiple_of(2) {
            return Err(invalid_projector_geometry(
                "projector grid height and width must be even",
            ));
        }
        let grid_input = u32::try_from(checked_tensor_elements(&[temporal, height, width])?)
            .map_err(|_| overflow())?;
        let grid_output =
            u32::try_from(checked_tensor_elements(&[temporal, height / 2, width / 2])?)
                .map_err(|_| overflow())?;
        input_tokens = input_tokens.checked_add(grid_input).ok_or_else(overflow)?;
        output_tokens = output_tokens
            .checked_add(grid_output)
            .ok_or_else(overflow)?;
    }
    Ok((input_tokens, output_tokens))
}

fn projector_source_token_indices(
    grids: &[[u32; 3]],
    input_tokens: u32,
    output_tokens: u32,
) -> Result<Vec<u32>, InvocationError> {
    let mapping_length = output_tokens.checked_mul(4).ok_or_else(overflow)?;
    if mapping_length != input_tokens {
        return Err(invalid_projector_geometry(
            "2x2 projector merge must consume every input token exactly once",
        ));
    }
    let mut mapping = Vec::with_capacity(usize::try_from(mapping_length).map_err(|_| overflow())?);
    let mut grid_offset = 0_u64;
    for &[temporal, height, width] in grids {
        let height = u64::from(height);
        let width = u64::from(width);
        for temporal_index in 0..u64::from(temporal) {
            let temporal_offset = temporal_index
                .checked_mul(height)
                .and_then(|value| value.checked_mul(width))
                .and_then(|value| value.checked_add(grid_offset))
                .ok_or_else(overflow)?;
            for block_y in 0..height / 2 {
                let top = block_y.checked_mul(2).ok_or_else(overflow)?;
                for block_x in 0..width / 2 {
                    let left = block_x.checked_mul(2).ok_or_else(overflow)?;
                    let top_left = temporal_offset
                        .checked_add(top.checked_mul(width).ok_or_else(overflow)?)
                        .and_then(|value| value.checked_add(left))
                        .ok_or_else(overflow)?;
                    for source in [
                        top_left,
                        top_left + 1,
                        top_left + width,
                        top_left + width + 1,
                    ] {
                        mapping.push(u32::try_from(source).map_err(|_| overflow())?);
                    }
                }
            }
        }
        grid_offset = grid_offset
            .checked_add(checked_tensor_elements(&[
                temporal,
                u32::try_from(height).map_err(|_| overflow())?,
                u32::try_from(width).map_err(|_| overflow())?,
            ])?)
            .ok_or_else(overflow)?;
    }
    debug_assert_eq!(mapping.len(), mapping_length as usize);
    Ok(mapping)
}

fn require_projector_permutation(
    source_token_indices: &[u32],
    input_tokens: u32,
) -> Result<(), InvocationError> {
    if source_token_indices.len() != usize::try_from(input_tokens).map_err(|_| overflow())? {
        return Err(invalid_projector_geometry(
            "projector source map must contain one entry per input token",
        ));
    }
    let mut seen = vec![false; source_token_indices.len()];
    for &source in source_token_indices {
        let index = usize::try_from(source).map_err(|_| overflow())?;
        let Some(slot) = seen.get_mut(index) else {
            return Err(invalid_projector_geometry(
                "projector source map contains an out-of-bounds token",
            ));
        };
        if *slot {
            return Err(invalid_projector_geometry(
                "projector source map contains a duplicate token",
            ));
        }
        *slot = true;
    }
    if seen.contains(&false) {
        return Err(invalid_projector_geometry(
            "projector source map does not cover every input token",
        ));
    }
    Ok(())
}

fn checked_add_f32_slice_bytes(bytes: u64, values: &[f32]) -> Result<u64, InvocationError> {
    let slice_bytes = u64::try_from(values.len())
        .map_err(|_| overflow())?
        .checked_mul(4)
        .ok_or_else(overflow)?;
    bytes.checked_add(slice_bytes).ok_or_else(overflow)
}

fn validate_checkpoint_selection(
    checkpoint_layers: &[usize],
    layer_count: usize,
) -> Result<(), InvocationError> {
    if checkpoint_layers.iter().any(|layer| *layer >= layer_count)
        || checkpoint_layers.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(InvocationError::new(
            InvocationErrorCode::InvalidCheckpointSelection,
            "checkpoint layers must be unique, strictly increasing, and within the stack",
        ));
    }
    Ok(())
}

fn validate_vision_norm(
    parameters: VisionLayerNormParameters<'_>,
    width: u32,
) -> Result<(), InvocationError> {
    require_len(parameters.weight.len(), u64::from(width))?;
    require_len(parameters.bias.len(), u64::from(width))?;
    require_finite(parameters.weight)?;
    require_finite(parameters.bias)
}

fn validate_vision_linear(
    parameters: VisionLinearParameters<'_>,
    input_width: u32,
    output_width: u32,
) -> Result<(), InvocationError> {
    require_len(
        parameters.weight.len(),
        checked_elements(output_width, input_width)?,
    )?;
    require_len(parameters.bias.len(), u64::from(output_width))?;
    require_finite(parameters.weight)?;
    require_finite(parameters.bias)
}

fn vision_layer_dispatch(
    stage: VisionEncoderLayerStage,
    kernel: KernelId,
    output_elements: u64,
    output_bytes: u64,
    workgroup_size: [u32; 3],
    dispatch: [u32; 3],
    uniform_words: [u32; 4],
) -> Result<VisionEncoderLayerDispatch, InvocationError> {
    Ok(VisionEncoderLayerDispatch {
        stage,
        invocation: plan(
            kernel,
            output_elements,
            output_bytes,
            workgroup_size,
            dispatch,
        )?,
        uniform_words,
    })
}

fn projector_dispatch(
    stage: ProjectorStage,
    kernel: KernelId,
    output_elements: u64,
    output_bytes: u64,
    workgroup_size: [u32; 3],
    dispatch: [u32; 3],
    uniform_words: [u32; 4],
) -> Result<ProjectorDispatch, InvocationError> {
    Ok(ProjectorDispatch {
        stage,
        invocation: plan(
            kernel,
            output_elements,
            output_bytes,
            workgroup_size,
            dispatch,
        )?,
        uniform_words,
    })
}

fn plan(
    kernel: KernelId,
    output_elements: u64,
    output_bytes: u64,
    workgroup_size: [u32; 3],
    dispatch: [u32; 3],
) -> Result<InvocationPlan, InvocationError> {
    Ok(InvocationPlan {
        kernel,
        output_elements: usize::try_from(output_elements).map_err(|_| overflow())?,
        output_bytes,
        workgroup_size,
        dispatch,
    })
}

fn require_dimensions(dimensions: &[u32]) -> Result<(), InvocationError> {
    if dimensions.contains(&0) {
        return Err(InvocationError::new(
            InvocationErrorCode::ZeroDimension,
            "all invocation dimensions must be nonzero",
        ));
    }
    Ok(())
}

fn checked_elements(left: u32, right: u32) -> Result<u64, InvocationError> {
    u64::from(left)
        .checked_mul(u64::from(right))
        .ok_or_else(overflow)
}

fn checked_tensor_elements(dimensions: &[u32]) -> Result<u64, InvocationError> {
    dimensions.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(u64::from(*dimension))
            .ok_or_else(overflow)
    })
}

fn checked_output_bytes(elements: u64) -> Result<u64, InvocationError> {
    elements.checked_mul(4).ok_or_else(overflow)
}

fn checked_align_up(value: u64, alignment: u64) -> Result<u64, InvocationError> {
    value
        .checked_add(alignment - 1)
        .map(|upper| upper & !(alignment - 1))
        .ok_or_else(overflow)
}

fn require_len(actual: usize, expected: u64) -> Result<(), InvocationError> {
    if u64::try_from(actual).map_err(|_| overflow())? != expected {
        return Err(InvocationError::new(
            InvocationErrorCode::LengthMismatch,
            format!("buffer has {actual} elements but the invocation requires {expected}"),
        ));
    }
    Ok(())
}

fn require_finite(values: &[f32]) -> Result<(), InvocationError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(InvocationError::new(
            InvocationErrorCode::NonFiniteInput,
            "floating input contains NaN or infinity",
        ));
    }
    Ok(())
}

fn require_epsilon(epsilon: f32) -> Result<(), InvocationError> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(InvocationError::new(
            InvocationErrorCode::InvalidEpsilon,
            "normalization epsilon must be positive and finite",
        ));
    }
    Ok(())
}

fn require_sequence_boundaries(cu_seqlens: &[u32], tokens: u32) -> Result<(), InvocationError> {
    if cu_seqlens.len() < 2
        || cu_seqlens.first() != Some(&0)
        || cu_seqlens.last() != Some(&tokens)
        || cu_seqlens.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(InvocationError::new(
            InvocationErrorCode::InvalidSequenceBoundaries,
            "cu_seqlens must start at zero, end at tokens, and be strictly increasing",
        ));
    }
    Ok(())
}

fn ceil_div(value: u32, divisor: u32) -> u32 {
    value / divisor + u32::from(!value.is_multiple_of(divisor))
}

fn bounded_linear_dispatch(work_items: u32, workgroup_width: u32) -> ([u32; 3], u32) {
    let workgroups = ceil_div(work_items, workgroup_width);
    let dispatch_y = ceil_div(workgroups, MAX_WEBGPU_DISPATCH_DIMENSION);
    let dispatch_x = ceil_div(workgroups, dispatch_y);
    let row_stride = if dispatch_y == 1 {
        0
    } else {
        dispatch_x * workgroup_width
    };
    ([dispatch_x, dispatch_y, 1], row_stride)
}

fn overflow() -> InvocationError {
    InvocationError::new(
        InvocationErrorCode::ArithmeticOverflow,
        "invocation size arithmetic overflowed",
    )
}

fn activation_layout_overflow(message: impl Into<String>) -> InvocationError {
    InvocationError::new(InvocationErrorCode::ArithmeticOverflow, message)
}

fn invalid_activation_layout(message: impl Into<String>) -> InvocationError {
    InvocationError::new(InvocationErrorCode::InvalidActivationLayout, message)
}

fn map_activation_layout_arena_error(error: ArenaError) -> InvocationError {
    let message = format!("vision stack activation-layout lowering failed: {error}");
    match error.code() {
        ArenaErrorCode::ArithmeticOverflow => activation_layout_overflow(message),
        _ => invalid_activation_layout(message),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvocationErrorCode {
    ZeroDimension,
    LengthMismatch,
    ArithmeticOverflow,
    InvalidEpsilon,
    InvalidRotaryDimension,
    InvalidRopeBase,
    UnsupportedHeadDimension,
    InvalidDecoderGeometry,
    InvalidVisionGeometry,
    InvalidProjectorGeometry,
    InvalidSequenceBoundaries,
    InvalidCheckpointSelection,
    InvalidActivationLayout,
    InvalidFusionTarget,
    NonFiniteInput,
}

fn invalid_fusion_target(message: impl Into<String>) -> InvocationError {
    InvocationError::new(InvocationErrorCode::InvalidFusionTarget, message)
}

fn invalid_decoder_geometry(message: impl Into<String>) -> InvocationError {
    InvocationError::new(InvocationErrorCode::InvalidDecoderGeometry, message)
}

fn invalid_projector_geometry(message: impl Into<String>) -> InvocationError {
    InvocationError::new(InvocationErrorCode::InvalidProjectorGeometry, message)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvocationError {
    code: InvocationErrorCode,
    message: String,
}

impl InvocationError {
    fn new(code: InvocationErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    #[must_use]
    pub const fn code(&self) -> InvocationErrorCode {
        self.code
    }
}

impl fmt::Display for InvocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "runtime invocation error {:?}: {}",
            self.code, self.message
        )
    }
}

impl Error for InvocationError {}
