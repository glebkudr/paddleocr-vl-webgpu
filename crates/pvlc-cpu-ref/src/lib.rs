//! Slow and explicit f32 arithmetic used as the kernel oracle.

mod multimodal;

use std::cmp::Ordering;
use std::collections::HashSet;
use std::error::Error;
use std::fmt;

pub use multimodal::{
    MultimodalRopePositions, assemble_multimodal_embeddings_f32, decode_mrope_position_ids,
    image_placeholder_count, mrope_position_ids,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuRefErrorCode {
    DimensionMismatch,
    NonPositiveEpsilon,
    AllMasked,
    InvalidRotaryDimension,
    InvalidRopeBase,
    InvalidK,
    NonFiniteInput,
    InvalidImageGeometry,
    InvalidPreprocessConfig,
    InvalidProjectorGeometry,
    InvalidSequenceBoundaries,
    InvalidTileSize,
    InvalidCheckpointSelection,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuRefError {
    code: CpuRefErrorCode,
    message: &'static str,
}

impl CpuRefError {
    #[must_use]
    pub const fn code(&self) -> CpuRefErrorCode {
        self.code
    }

    const fn new(code: CpuRefErrorCode, message: &'static str) -> Self {
        Self { code, message }
    }
}

impl fmt::Display for CpuRefError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "CPU reference error {:?}: {}",
            self.code, self.message
        )
    }
}

impl Error for CpuRefError {}

pub fn gemm_f32(
    left: &[f32],
    rows: usize,
    inner: usize,
    right: &[f32],
    columns: usize,
) -> Result<Vec<f32>, CpuRefError> {
    require_len(left, rows, inner)?;
    require_len(right, inner, columns)?;
    require_finite(left)?;
    require_finite(right)?;
    let output_len = rows.checked_mul(columns).ok_or_else(dimension_error)?;
    let mut output = vec![0.0_f32; output_len];
    for row in 0..rows {
        for column in 0..columns {
            let mut accumulator = 0.0_f32;
            for depth in 0..inner {
                let product = left[row * inner + depth] * right[depth * columns + column];
                accumulator += product;
            }
            output[row * columns + column] = accumulator;
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn patch_projection_f32(
    patches: &[f32],
    patch_count: usize,
    channels: usize,
    patch_size: usize,
    weights: &[f32],
    bias: &[f32],
    output_width: usize,
) -> Result<Vec<f32>, CpuRefError> {
    if patch_count == 0 || channels == 0 || patch_size == 0 || output_width == 0 {
        return Err(dimension_error());
    }
    let input_width = channels
        .checked_mul(patch_size)
        .and_then(|elements| elements.checked_mul(patch_size))
        .ok_or_else(dimension_error)?;
    require_len(patches, patch_count, input_width)?;
    require_len(weights, output_width, input_width)?;
    if bias.len() != output_width {
        return Err(dimension_error());
    }
    require_finite(patches)?;
    require_finite(weights)?;
    require_finite(bias)?;

    let output_len = patch_count
        .checked_mul(output_width)
        .ok_or_else(dimension_error)?;
    let mut output = vec![0.0_f32; output_len];
    for (patch_index, patch) in patches.chunks_exact(input_width).enumerate() {
        for output_channel in 0..output_width {
            let weight_start = output_channel * input_width;
            let weight = &weights[weight_start..weight_start + input_width];
            let mut accumulator = bias[output_channel];
            for (&input, &coefficient) in patch.iter().zip(weight) {
                accumulator += input * coefficient;
            }
            output[patch_index * output_width + output_channel] = accumulator;
        }
    }
    Ok(output)
}

pub fn linear_f32(
    input: &[f32],
    rows: usize,
    input_width: usize,
    weights: &[f32],
    bias: &[f32],
    output_width: usize,
) -> Result<Vec<f32>, CpuRefError> {
    patch_projection_f32(input, rows, input_width, 1, weights, bias, output_width)
}

pub fn add_vectors_f32(left: &[f32], right: &[f32]) -> Result<Vec<f32>, CpuRefError> {
    if left.is_empty() || left.len() != right.len() {
        return Err(dimension_error());
    }
    require_finite(left)?;
    require_finite(right)?;

    Ok(left
        .iter()
        .zip(right)
        .map(|(&left_value, &right_value)| left_value + right_value)
        .collect())
}

pub fn add_interpolated_position_embedding_f32(
    patch_embeddings: &[f32],
    hidden_size: usize,
    position_embedding: &[f32],
    source_height: usize,
    source_width: usize,
    image_grid_thw: &[[usize; 3]],
) -> Result<Vec<f32>, CpuRefError> {
    if hidden_size == 0 || source_height == 0 || source_width == 0 || image_grid_thw.is_empty() {
        return Err(dimension_error());
    }
    let source_positions = source_height
        .checked_mul(source_width)
        .ok_or_else(dimension_error)?;
    require_len(position_embedding, source_positions, hidden_size)?;
    let total_tokens = image_grid_thw.iter().try_fold(0_usize, |total, grid| {
        let [temporal, height, width] = *grid;
        if temporal == 0 || height == 0 || width == 0 {
            return Err(dimension_error());
        }
        let tokens = temporal
            .checked_mul(height)
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(dimension_error)?;
        total.checked_add(tokens).ok_or_else(dimension_error)
    })?;
    require_len(patch_embeddings, total_tokens, hidden_size)?;
    require_finite(patch_embeddings)?;
    require_finite(position_embedding)?;

    let mut output = patch_embeddings.to_vec();
    let mut token_offset = 0_usize;
    for &[temporal, height, width] in image_grid_thw {
        let spatial_tokens = height.checked_mul(width).ok_or_else(dimension_error)?;
        for time in 0..temporal {
            for y in 0..height {
                let (source_y0, source_y1, y_fraction) = bilinear_axis(source_height, height, y);
                for x in 0..width {
                    let (source_x0, source_x1, x_fraction) = bilinear_axis(source_width, width, x);
                    let target_token = token_offset + time * spatial_tokens + y * width + x;
                    let target_start = target_token * hidden_size;
                    for channel in 0..hidden_size {
                        let top_left = position_embedding
                            [(source_y0 * source_width + source_x0) * hidden_size + channel];
                        let top_right = position_embedding
                            [(source_y0 * source_width + source_x1) * hidden_size + channel];
                        let bottom_left = position_embedding
                            [(source_y1 * source_width + source_x0) * hidden_size + channel];
                        let bottom_right = position_embedding
                            [(source_y1 * source_width + source_x1) * hidden_size + channel];
                        let top = top_left + (top_right - top_left) * x_fraction;
                        let bottom = bottom_left + (bottom_right - bottom_left) * x_fraction;
                        let position = top + (bottom - top) * y_fraction;
                        output[target_start + channel] += position;
                    }
                }
            }
        }
        token_offset += temporal * spatial_tokens;
    }
    Ok(output)
}

pub fn projector_merge_2x2_f32(
    image_features: &[f32],
    hidden_size: usize,
    image_grid_thw: &[[usize; 3]],
) -> Result<Vec<f32>, CpuRefError> {
    let layout = projector_layout(hidden_size, image_grid_thw)?;
    require_len(image_features, layout.input_tokens, hidden_size)?;
    require_finite(image_features)?;
    let output_len = layout
        .output_tokens
        .checked_mul(layout.merged_width)
        .ok_or_else(dimension_error)?;
    if output_len != image_features.len() {
        return Err(dimension_error());
    }

    let mut output = Vec::with_capacity(output_len);
    let mut image_token_offset = 0_usize;
    for &[temporal, height, width] in image_grid_thw {
        let spatial_tokens = height * width;
        for time in 0..temporal {
            for merged_y in 0..height / 2 {
                for merged_x in 0..width / 2 {
                    for patch_y in 0..2 {
                        for patch_x in 0..2 {
                            let source_y = merged_y * 2 + patch_y;
                            let source_x = merged_x * 2 + patch_x;
                            let source_token = image_token_offset
                                + time * spatial_tokens
                                + source_y * width
                                + source_x;
                            let source_start = source_token * hidden_size;
                            output.extend_from_slice(
                                &image_features[source_start..source_start + hidden_size],
                            );
                        }
                    }
                }
            }
        }
        image_token_offset += temporal * spatial_tokens;
    }
    debug_assert_eq!(output.len(), output_len);
    Ok(output)
}

fn bilinear_axis(
    source_size: usize,
    target_size: usize,
    target_index: usize,
) -> (usize, usize, f32) {
    let source_coordinate = if target_size == 1 {
        0.0
    } else {
        target_index as f32 * (source_size - 1) as f32 / (target_size - 1) as f32
    };
    let lower = (source_coordinate.floor() as usize).min(source_size - 1);
    let upper = (lower + 1).min(source_size - 1);
    (lower, upper, source_coordinate - lower as f32)
}

pub fn layer_norm_f32(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, CpuRefError> {
    require_positive_finite_epsilon(epsilon)?;
    require_nonzero_width(width)?;
    require_len(input, rows, width)?;
    if weight.len() != width || bias.len() != width {
        return Err(dimension_error());
    }
    require_finite(input)?;
    require_finite(weight)?;
    require_finite(bias)?;

    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let source = &input[row * width..(row + 1) * width];
        let mut mean = 0.0_f32;
        for value in source {
            mean += *value;
        }
        mean /= width as f32;
        let mut variance = 0.0_f32;
        for value in source {
            let centered = *value - mean;
            variance += centered * centered;
        }
        variance /= width as f32;
        let inverse_stddev = (variance + epsilon).sqrt().recip();
        for column in 0..width {
            output[row * width + column] =
                (source[column] - mean) * inverse_stddev * weight[column] + bias[column];
        }
    }
    Ok(output)
}

pub fn rms_norm_f32(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, CpuRefError> {
    require_positive_finite_epsilon(epsilon)?;
    require_nonzero_width(width)?;
    require_len(input, rows, width)?;
    if weight.len() != width {
        return Err(dimension_error());
    }
    require_finite(input)?;
    require_finite(weight)?;

    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let source = &input[row * width..(row + 1) * width];
        let mut mean_square = 0.0_f32;
        for value in source {
            mean_square += *value * *value;
        }
        mean_square /= width as f32;
        let inverse_rms = (mean_square + epsilon).sqrt().recip();
        for column in 0..width {
            output[row * width + column] = source[column] * inverse_rms * weight[column];
        }
    }
    Ok(output)
}

#[must_use]
pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[must_use]
pub fn gelu_pytorch_tanh(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.797_884_6 * (value + 0.044_715 * value.powi(3))).tanh())
}

/// Exact-mode GELU used by Transformers' `GELUActivation`.
#[must_use]
pub fn gelu_erf_f32(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erff(value * std::f32::consts::FRAC_1_SQRT_2))
}

pub fn softmax_rows_f32(
    logits: &[f32],
    rows: usize,
    columns: usize,
    mask: Option<&[bool]>,
) -> Result<Vec<f32>, CpuRefError> {
    require_nonzero_width(columns)?;
    require_len(logits, rows, columns)?;
    if let Some(mask) = mask
        && mask.len() != logits.len()
    {
        return Err(dimension_error());
    }
    require_finite(logits)?;

    let mut output = vec![0.0_f32; logits.len()];
    for row in 0..rows {
        let start = row * columns;
        let end = start + columns;
        let active = |index: usize| mask.is_none_or(|values| values[index]);
        let mut maximum = f32::NEG_INFINITY;
        for (relative_index, &logit) in logits[start..end].iter().enumerate() {
            let index = start + relative_index;
            if active(index) {
                maximum = maximum.max(logit);
            }
        }
        if maximum == f32::NEG_INFINITY {
            return Err(CpuRefError::new(
                CpuRefErrorCode::AllMasked,
                "softmax row has no active values",
            ));
        }
        let mut denominator = 0.0_f32;
        for (relative_index, (&logit, result)) in logits[start..end]
            .iter()
            .zip(&mut output[start..end])
            .enumerate()
        {
            let index = start + relative_index;
            if active(index) {
                let exponential = (logit - maximum).exp();
                *result = exponential;
                denominator += exponential;
            }
        }
        for (relative_index, result) in output[start..end].iter_mut().enumerate() {
            let index = start + relative_index;
            if active(index) {
                *result /= denominator;
            }
        }
    }
    Ok(output)
}

pub fn apply_rope_neox(
    values: &mut [f32],
    rows: usize,
    width: usize,
    rotary_dim: usize,
    positions: &[u32],
    base: f32,
) -> Result<(), CpuRefError> {
    require_len(values, rows, width)?;
    if positions.len() != rows {
        return Err(dimension_error());
    }
    if rotary_dim == 0 || rotary_dim > width || !rotary_dim.is_multiple_of(2) {
        return Err(CpuRefError::new(
            CpuRefErrorCode::InvalidRotaryDimension,
            "rotary dimension must be nonzero, even, and no wider than a row",
        ));
    }
    if !base.is_finite() || base <= 0.0 {
        return Err(CpuRefError::new(
            CpuRefErrorCode::InvalidRopeBase,
            "RoPE base must be positive and finite",
        ));
    }
    require_finite(values)?;

    let half = rotary_dim / 2;
    for (row, position) in positions.iter().copied().enumerate() {
        let row_start = row * width;
        for pair in 0..half {
            let first_index = row_start + pair;
            let second_index = row_start + half + pair;
            let first = values[first_index];
            let second = values[second_index];
            let exponent = -(2.0 * pair as f32 / rotary_dim as f32);
            let angle = position as f32 * base.powf(exponent);
            let (sine, cosine) = angle.sin_cos();
            values[first_index] = first * cosine - second * sine;
            values[second_index] = second * cosine + first * sine;
        }
    }
    Ok(())
}

/// Applies the current Transformers PaddleOCR-VL vision encoder's 2-D
/// height/width rotary embedding to token-major `[tokens, heads, head_dim]`
/// query and key tensors.
pub fn vision_rope_2d_f32(
    query: &[f32],
    key: &[f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    position_ids: &[[usize; 2]],
    base: f32,
) -> Result<(Vec<f32>, Vec<f32>), CpuRefError> {
    if tokens == 0 || heads == 0 {
        return Err(dimension_error());
    }
    if head_dim == 0 || !head_dim.is_multiple_of(4) {
        return Err(CpuRefError::new(
            CpuRefErrorCode::InvalidRotaryDimension,
            "vision RoPE head dimension must be positive and divisible by four",
        ));
    }
    if !base.is_finite() || base <= 0.0 {
        return Err(CpuRefError::new(
            CpuRefErrorCode::InvalidRopeBase,
            "RoPE base must be positive and finite",
        ));
    }
    if position_ids.len() != tokens {
        return Err(dimension_error());
    }
    let tensor_elements = tokens
        .checked_mul(heads)
        .and_then(|elements| elements.checked_mul(head_dim))
        .ok_or_else(dimension_error)?;
    if query.len() != tensor_elements || key.len() != tensor_elements {
        return Err(dimension_error());
    }
    require_finite(query)?;
    require_finite(key)?;

    let pair_count = head_dim / 2;
    let frequency_count = pair_count / 2;
    let mut rotated_query = query.to_vec();
    let mut rotated_key = key.to_vec();
    for (token, &[height, width]) in position_ids.iter().enumerate() {
        for head in 0..heads {
            let head_start = (token * heads + head) * head_dim;
            for pair in 0..pair_count {
                let axis_position = if pair < frequency_count {
                    height
                } else {
                    width
                };
                let frequency_index = pair % frequency_count;
                let exponent = -(2.0 * frequency_index as f32 / pair_count as f32);
                let angle = axis_position as f32 * base.powf(exponent);
                let (sine, cosine) = angle.sin_cos();
                let first_index = head_start + pair;
                let second_index = first_index + pair_count;

                let query_first = query[first_index];
                let query_second = query[second_index];
                rotated_query[first_index] = query_first * cosine - query_second * sine;
                rotated_query[second_index] = query_second * cosine + query_first * sine;

                let key_first = key[first_index];
                let key_second = key[second_index];
                rotated_key[first_index] = key_first * cosine - key_second * sine;
                rotated_key[second_index] = key_second * cosine + key_first * sine;
            }
        }
    }
    Ok((rotated_query, rotated_key))
}

const PINNED_DECODER_QUERY_HEADS: usize = 16;
const PINNED_DECODER_KEY_VALUE_HEADS: usize = 2;
const PINNED_DECODER_HEAD_DIM: usize = 128;
const PINNED_DECODER_MROPE_SECTIONS: [usize; 3] = [16, 24, 24];
const PINNED_DECODER_HIDDEN_SIZE: usize = 1_024;
const PINNED_DECODER_INTERMEDIATE_SIZE: usize = 3_072;
const PINNED_DECODER_RMS_NORM_EPSILON: f32 = 1.0e-5;
const PINNED_DECODER_LAYERS: usize = 18;
const PINNED_DECODER_EOS_TOKEN_ID: usize = 2;
pub const PINNED_DECODER_VOCAB_SIZE: usize = 103_424;

/// Parameterized multimodal RoPE arithmetic used by internal CPU differential
/// and boundary tests.
///
/// This is not the milestone-facing pinned decoder ABI. Call
/// [`apply_pinned_decoder_multimodal_rope_f32`] for the fixed
/// PaddleOCR-VL-1.6 decoder topology.
#[allow(clippy::too_many_arguments)]
pub fn apply_multimodal_rope_f32(
    query: &[f32],
    key: &[f32],
    tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    raw_cos: &[f32],
    raw_sin: &[f32],
    mrope_sections: [usize; 3],
) -> Result<(Vec<f32>, Vec<f32>), CpuRefError> {
    if tokens == 0 || query_heads == 0 || key_value_heads == 0 || head_dim == 0 {
        return Err(dimension_error());
    }
    if !query_heads.is_multiple_of(key_value_heads) {
        return Err(dimension_error());
    }
    if mrope_sections.into_iter().any(|section| section == 0) {
        return Err(dimension_error());
    }
    let section_sum = mrope_sections
        .into_iter()
        .try_fold(0usize, |sum, section| sum.checked_add(section))
        .ok_or_else(dimension_error)?;
    let repeated_dim = section_sum.checked_mul(2).ok_or_else(dimension_error)?;
    if repeated_dim != head_dim {
        return Err(dimension_error());
    }

    let query_width = query_heads
        .checked_mul(head_dim)
        .ok_or_else(dimension_error)?;
    let key_width = key_value_heads
        .checked_mul(head_dim)
        .ok_or_else(dimension_error)?;
    let raw_width = tokens
        .checked_mul(head_dim)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(dimension_error)?;
    require_len(query, tokens, query_width)?;
    require_len(key, tokens, key_width)?;
    if raw_cos.len() != raw_width || raw_sin.len() != raw_width {
        return Err(dimension_error());
    }
    require_finite(query)?;
    require_finite(key)?;
    require_finite(raw_cos)?;
    require_finite(raw_sin)?;

    let mut rotated_query = vec![0.0_f32; query.len()];
    let mut rotated_key = vec![0.0_f32; key.len()];
    let repeated_sections = [
        mrope_sections[0],
        mrope_sections[1],
        mrope_sections[2],
        mrope_sections[0],
        mrope_sections[1],
        mrope_sections[2],
    ];
    let half_dim = head_dim / 2;

    for token in 0..tokens {
        let mut selected_cos = vec![0.0_f32; head_dim];
        let mut selected_sin = vec![0.0_f32; head_dim];
        let mut selected_offset = 0usize;
        for (chunk_index, chunk_size) in repeated_sections.into_iter().enumerate() {
            let axis = chunk_index % 3;
            let axis_base = axis
                .checked_mul(tokens)
                .and_then(|value| value.checked_mul(head_dim))
                .and_then(|value| value.checked_add(token * head_dim))
                .and_then(|value| value.checked_add(selected_offset))
                .ok_or_else(dimension_error)?;
            let destination_end = selected_offset
                .checked_add(chunk_size)
                .ok_or_else(dimension_error)?;
            selected_cos[selected_offset..destination_end]
                .copy_from_slice(&raw_cos[axis_base..axis_base + chunk_size]);
            selected_sin[selected_offset..destination_end]
                .copy_from_slice(&raw_sin[axis_base..axis_base + chunk_size]);
            selected_offset = destination_end;
        }

        for head in 0..query_heads {
            let row_start = attention_index(token, head, 0, query_heads, head_dim);
            for dim in 0..head_dim {
                let source = if dim < half_dim {
                    -query[row_start + dim + half_dim]
                } else {
                    query[row_start + dim - half_dim]
                };
                rotated_query[row_start + dim] =
                    query[row_start + dim] * selected_cos[dim] + source * selected_sin[dim];
            }
        }
        for head in 0..key_value_heads {
            let row_start = attention_index(token, head, 0, key_value_heads, head_dim);
            for dim in 0..head_dim {
                let source = if dim < half_dim {
                    -key[row_start + dim + half_dim]
                } else {
                    key[row_start + dim - half_dim]
                };
                rotated_key[row_start + dim] =
                    key[row_start + dim] * selected_cos[dim] + source * selected_sin[dim];
            }
        }
    }

    Ok((rotated_query, rotated_key))
}

/// Applies the pinned PaddleOCR-VL-1.6 decoder's multimodal RoPE to token-major
/// bias-free query and key rows.
pub fn apply_pinned_decoder_multimodal_rope_f32(
    query: &[f32],
    key: &[f32],
    tokens: usize,
    raw_cos: &[f32],
    raw_sin: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), CpuRefError> {
    apply_multimodal_rope_f32(
        query,
        key,
        tokens,
        PINNED_DECODER_QUERY_HEADS,
        PINNED_DECODER_KEY_VALUE_HEADS,
        PINNED_DECODER_HEAD_DIM,
        raw_cos,
        raw_sin,
        PINNED_DECODER_MROPE_SECTIONS,
    )
}

fn validate_gqa_geometry(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Result<(usize, usize), CpuRefError> {
    if tokens == 0 || query_heads == 0 || key_value_heads == 0 || head_dim == 0 {
        return Err(dimension_error());
    }
    if !query_heads.is_multiple_of(key_value_heads) {
        return Err(dimension_error());
    }
    let query_width = query_heads
        .checked_mul(head_dim)
        .ok_or_else(dimension_error)?;
    let key_value_width = key_value_heads
        .checked_mul(head_dim)
        .ok_or_else(dimension_error)?;
    require_len(query, tokens, query_width)?;
    require_len(key, tokens, key_value_width)?;
    require_len(value, tokens, key_value_width)?;
    require_finite(query)?;
    require_finite(key)?;
    require_finite(value)?;
    Ok((query_width, key_value_width))
}

fn validate_decode_gqa_geometry(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    cache_tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Result<(usize, usize), CpuRefError> {
    if cache_tokens == 0 || query_heads == 0 || key_value_heads == 0 || head_dim == 0 {
        return Err(dimension_error());
    }
    if !query_heads.is_multiple_of(key_value_heads) {
        return Err(dimension_error());
    }
    let query_width = query_heads
        .checked_mul(head_dim)
        .ok_or_else(dimension_error)?;
    let key_value_width = key_value_heads
        .checked_mul(head_dim)
        .ok_or_else(dimension_error)?;
    require_len(query, 1, query_width)?;
    require_len(key, cache_tokens, key_value_width)?;
    require_len(value, cache_tokens, key_value_width)?;
    require_finite(query)?;
    require_finite(key)?;
    require_finite(value)?;
    Ok((query_width, key_value_width))
}

fn validate_kv_geometry(
    key: &[f32],
    value: &[f32],
    tokens: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Result<usize, CpuRefError> {
    if tokens == 0 || key_value_heads == 0 || head_dim == 0 {
        return Err(dimension_error());
    }
    let key_value_width = key_value_heads
        .checked_mul(head_dim)
        .ok_or_else(dimension_error)?;
    require_len(key, tokens, key_value_width)?;
    require_len(value, tokens, key_value_width)?;
    require_finite(key)?;
    require_finite(value)?;
    Ok(key_value_width)
}

/// Parameterized causal grouped-query attention arithmetic used by internal CPU
/// differential and boundary tests.
///
/// This helper keeps direct KV grouping without physically repeating KV heads
/// and allocates only linear scratch per `(query_token, query_head)`.
#[allow(clippy::too_many_arguments)]
pub fn causal_gqa_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, CpuRefError> {
    let (query_width, _) = validate_gqa_geometry(
        query,
        key,
        value,
        tokens,
        query_heads,
        key_value_heads,
        head_dim,
    )?;
    let mut output = vec![0.0_f32; query.len()];
    let query_heads_per_kv = query_heads / key_value_heads;
    let scale = (head_dim as f32).sqrt().recip();

    for query_token in 0..tokens {
        for query_head in 0..query_heads {
            let key_value_head = query_head / query_heads_per_kv;
            let mut probabilities = Vec::with_capacity(query_token + 1);
            for key_token in 0..=query_token {
                let mut score = 0.0_f32;
                for dim in 0..head_dim {
                    score += query
                        [attention_index(query_token, query_head, dim, query_heads, head_dim)]
                        * key[attention_index(
                            key_token,
                            key_value_head,
                            dim,
                            key_value_heads,
                            head_dim,
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
            let output_start = query_token * query_width + query_head * head_dim;
            for dim in 0..head_dim {
                let mut weighted = 0.0_f32;
                for (key_token, probability) in probabilities.iter().copied().enumerate() {
                    weighted += probability
                        * value[attention_index(
                            key_token,
                            key_value_head,
                            dim,
                            key_value_heads,
                            head_dim,
                        )];
                }
                output[output_start + dim] = weighted;
            }
        }
    }
    Ok(output)
}

/// Applies grouped-query attention to exactly one query token against a full
/// token-major KV cache, keeping direct KV grouping without physical repeats.
#[allow(clippy::too_many_arguments)]
pub fn decode_gqa_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    cache_tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, CpuRefError> {
    let (query_width, _) = validate_decode_gqa_geometry(
        query,
        key,
        value,
        cache_tokens,
        query_heads,
        key_value_heads,
        head_dim,
    )?;
    let mut output = vec![0.0_f32; query_width];
    let query_heads_per_kv = query_heads / key_value_heads;
    let scale = (head_dim as f32).sqrt().recip();

    for query_head in 0..query_heads {
        let key_value_head = query_head / query_heads_per_kv;
        let mut probabilities = Vec::with_capacity(cache_tokens);
        for key_token in 0..cache_tokens {
            let mut score = 0.0_f32;
            for dim in 0..head_dim {
                score += query[attention_index(0, query_head, dim, query_heads, head_dim)]
                    * key[attention_index(
                        key_token,
                        key_value_head,
                        dim,
                        key_value_heads,
                        head_dim,
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

        let output_start = query_head * head_dim;
        for dim in 0..head_dim {
            let mut weighted = 0.0_f32;
            for (key_token, probability) in probabilities.iter().copied().enumerate() {
                weighted += probability
                    * value[attention_index(
                        key_token,
                        key_value_head,
                        dim,
                        key_value_heads,
                        head_dim,
                    )];
            }
            output[output_start + dim] = weighted;
        }
    }
    Ok(output)
}

/// Applies the pinned PaddleOCR-VL-1.6 decoder's causal grouped-query
/// attention with direct KV grouping and linear scratch.
pub fn pinned_decoder_causal_gqa_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    tokens: usize,
) -> Result<Vec<f32>, CpuRefError> {
    causal_gqa_f32(
        query,
        key,
        value,
        tokens,
        PINNED_DECODER_QUERY_HEADS,
        PINNED_DECODER_KEY_VALUE_HEADS,
        PINNED_DECODER_HEAD_DIM,
    )
}

/// Applies the pinned PaddleOCR-VL-1.6 decoder's single-token grouped-query
/// attention against a full detached KV cache.
pub fn pinned_decoder_decode_gqa_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    cache_tokens: usize,
) -> Result<Vec<f32>, CpuRefError> {
    decode_gqa_f32(
        query,
        key,
        value,
        cache_tokens,
        PINNED_DECODER_QUERY_HEADS,
        PINNED_DECODER_KEY_VALUE_HEADS,
        PINNED_DECODER_HEAD_DIM,
    )
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderPrefillKvCache {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub tokens: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
}

fn write_decoder_prefill_kv_f32(
    rotated_key: &[f32],
    value: &[f32],
    tokens: usize,
    key_value_heads: usize,
    head_dim: usize,
) -> Result<DecoderPrefillKvCache, CpuRefError> {
    validate_kv_geometry(rotated_key, value, tokens, key_value_heads, head_dim)?;
    Ok(DecoderPrefillKvCache {
        keys: rotated_key.to_vec(),
        values: value.to_vec(),
        tokens,
        key_value_heads,
        head_dim,
    })
}

/// Clones the pinned decoder's prefill KV tensors into a detached CPU cache.
pub fn write_pinned_decoder_prefill_kv_f32(
    rotated_key: &[f32],
    value: &[f32],
    tokens: usize,
) -> Result<DecoderPrefillKvCache, CpuRefError> {
    write_decoder_prefill_kv_f32(
        rotated_key,
        value,
        tokens,
        PINNED_DECODER_KEY_VALUE_HEADS,
        PINNED_DECODER_HEAD_DIM,
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoderLayerConfig {
    pub tokens: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_epsilon: f32,
    pub mrope_sections: [usize; 3],
}

const fn pinned_decoder_layer_config(tokens: usize) -> DecoderLayerConfig {
    DecoderLayerConfig {
        tokens,
        hidden_size: PINNED_DECODER_HIDDEN_SIZE,
        intermediate_size: PINNED_DECODER_INTERMEDIATE_SIZE,
        query_heads: PINNED_DECODER_QUERY_HEADS,
        key_value_heads: PINNED_DECODER_KEY_VALUE_HEADS,
        head_dim: PINNED_DECODER_HEAD_DIM,
        rms_norm_epsilon: PINNED_DECODER_RMS_NORM_EPSILON,
        mrope_sections: PINNED_DECODER_MROPE_SECTIONS,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoderLayerParameters<'a> {
    pub input_norm_weight: &'a [f32],
    pub query_weight: &'a [f32],
    pub key_weight: &'a [f32],
    pub value_weight: &'a [f32],
    pub attention_output_weight: &'a [f32],
    pub post_attention_norm_weight: &'a [f32],
    pub gate_weight: &'a [f32],
    pub up_weight: &'a [f32],
    pub down_weight: &'a [f32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderLayerPrefillTrace {
    pub norm1: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub mrope_query: Vec<f32>,
    pub mrope_key: Vec<f32>,
    pub kv_cache: DecoderPrefillKvCache,
    pub attention_context: Vec<f32>,
    pub attention_output: Vec<f32>,
    pub attention_residual: Vec<f32>,
    pub norm2: Vec<f32>,
    pub mlp_gate: Vec<f32>,
    pub mlp_up: Vec<f32>,
    pub mlp_activation: Vec<f32>,
    pub mlp_down: Vec<f32>,
    pub output: Vec<f32>,
}

pub type DecoderLayerDecodeTrace = DecoderLayerPrefillTrace;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoderStackConfig {
    pub layer: DecoderLayerConfig,
    pub layers: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderStackCheckpoint {
    pub layer_index: usize,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderStackPrefillTrace {
    pub checkpoints: Vec<DecoderStackCheckpoint>,
    pub kv_caches: Vec<DecoderPrefillKvCache>,
    pub final_norm: Vec<f32>,
    pub executed_layers: usize,
    pub retained_checkpoint_elements: usize,
    pub retained_kv_elements: usize,
}

impl DecoderStackPrefillTrace {
    #[must_use]
    pub fn checkpoint(&self, layer_index: usize) -> Option<&[f32]> {
        self.checkpoints
            .binary_search_by_key(&layer_index, |checkpoint| checkpoint.layer_index)
            .ok()
            .map(|index| self.checkpoints[index].values.as_slice())
    }

    #[must_use]
    pub fn kv_cache(&self, layer_index: usize) -> Option<&DecoderPrefillKvCache> {
        self.kv_caches.get(layer_index)
    }
}

pub type DecoderStackDecodeTrace = DecoderStackPrefillTrace;

#[derive(Clone, Copy)]
struct DecoderLayerGeometry {
    query_width: usize,
    key_value_width: usize,
    layer_elements: usize,
    key_value_elements: usize,
    raw_table_elements: usize,
    query_weight_elements: usize,
    key_value_weight_elements: usize,
    attention_output_weight_elements: usize,
    intermediate_weight_elements: usize,
    down_weight_elements: usize,
}

fn validate_decoder_layer_config(
    config: DecoderLayerConfig,
) -> Result<DecoderLayerGeometry, CpuRefError> {
    if config.tokens == 0
        || config.hidden_size == 0
        || config.intermediate_size == 0
        || config.query_heads == 0
        || config.key_value_heads == 0
        || config.head_dim == 0
        || !config.query_heads.is_multiple_of(config.key_value_heads)
        || config
            .mrope_sections
            .into_iter()
            .any(|section| section == 0)
    {
        return Err(dimension_error());
    }
    let section_sum = config
        .mrope_sections
        .into_iter()
        .try_fold(0usize, |sum, section| sum.checked_add(section))
        .ok_or_else(dimension_error)?;
    if section_sum.checked_mul(2).ok_or_else(dimension_error)? != config.head_dim {
        return Err(dimension_error());
    }
    require_positive_finite_epsilon(config.rms_norm_epsilon)?;

    let query_width = config
        .query_heads
        .checked_mul(config.head_dim)
        .ok_or_else(dimension_error)?;
    let key_value_width = config
        .key_value_heads
        .checked_mul(config.head_dim)
        .ok_or_else(dimension_error)?;
    let layer_elements = config
        .tokens
        .checked_mul(config.hidden_size)
        .ok_or_else(dimension_error)?;
    config
        .tokens
        .checked_mul(query_width)
        .ok_or_else(dimension_error)?;
    let key_value_elements = config
        .tokens
        .checked_mul(key_value_width)
        .ok_or_else(dimension_error)?;
    config
        .tokens
        .checked_mul(config.intermediate_size)
        .ok_or_else(dimension_error)?;
    let raw_table_elements = config
        .tokens
        .checked_mul(config.head_dim)
        .and_then(|length| length.checked_mul(3))
        .ok_or_else(dimension_error)?;
    let query_weight_elements = query_width
        .checked_mul(config.hidden_size)
        .ok_or_else(dimension_error)?;
    let key_value_weight_elements = key_value_width
        .checked_mul(config.hidden_size)
        .ok_or_else(dimension_error)?;
    let attention_output_weight_elements = config
        .hidden_size
        .checked_mul(query_width)
        .ok_or_else(dimension_error)?;
    let intermediate_weight_elements = config
        .intermediate_size
        .checked_mul(config.hidden_size)
        .ok_or_else(dimension_error)?;
    let down_weight_elements = config
        .hidden_size
        .checked_mul(config.intermediate_size)
        .ok_or_else(dimension_error)?;

    Ok(DecoderLayerGeometry {
        query_width,
        key_value_width,
        layer_elements,
        key_value_elements,
        raw_table_elements,
        query_weight_elements,
        key_value_weight_elements,
        attention_output_weight_elements,
        intermediate_weight_elements,
        down_weight_elements,
    })
}

fn validate_decoder_layer_prefill(
    input: &[f32],
    config: DecoderLayerConfig,
    raw_cos: &[f32],
    raw_sin: &[f32],
    parameters: &DecoderLayerParameters<'_>,
) -> Result<DecoderLayerGeometry, CpuRefError> {
    let geometry = validate_decoder_layer_config(config)?;

    if input.len() != geometry.layer_elements
        || raw_cos.len() != geometry.raw_table_elements
        || raw_sin.len() != geometry.raw_table_elements
        || parameters.input_norm_weight.len() != config.hidden_size
        || parameters.query_weight.len() != geometry.query_weight_elements
        || parameters.key_weight.len() != geometry.key_value_weight_elements
        || parameters.value_weight.len() != geometry.key_value_weight_elements
        || parameters.attention_output_weight.len() != geometry.attention_output_weight_elements
        || parameters.post_attention_norm_weight.len() != config.hidden_size
        || parameters.gate_weight.len() != geometry.intermediate_weight_elements
        || parameters.up_weight.len() != geometry.intermediate_weight_elements
        || parameters.down_weight.len() != geometry.down_weight_elements
    {
        return Err(dimension_error());
    }

    for operand in [
        input,
        raw_cos,
        raw_sin,
        parameters.input_norm_weight,
        parameters.query_weight,
        parameters.key_weight,
        parameters.value_weight,
        parameters.attention_output_weight,
        parameters.post_attention_norm_weight,
        parameters.gate_weight,
        parameters.up_weight,
        parameters.down_weight,
    ] {
        require_finite(operand)?;
    }

    Ok(geometry)
}

struct DecoderLayerDecodeValidation {
    geometry: DecoderLayerGeometry,
    cache_tokens: usize,
}

fn validate_decoder_layer_decode(
    input: &[f32],
    config: DecoderLayerConfig,
    raw_cos: &[f32],
    raw_sin: &[f32],
    prefix_cache: &DecoderPrefillKvCache,
    parameters: &DecoderLayerParameters<'_>,
) -> Result<DecoderLayerDecodeValidation, CpuRefError> {
    if config.tokens != 1 {
        return Err(dimension_error());
    }
    let geometry = validate_decoder_layer_prefill(input, config, raw_cos, raw_sin, parameters)?;
    if prefix_cache.key_value_heads != config.key_value_heads
        || prefix_cache.head_dim != config.head_dim
    {
        return Err(dimension_error());
    }
    validate_kv_geometry(
        &prefix_cache.keys,
        &prefix_cache.values,
        prefix_cache.tokens,
        config.key_value_heads,
        config.head_dim,
    )?;
    let cache_tokens = prefix_cache
        .tokens
        .checked_add(1)
        .ok_or_else(dimension_error)?;
    Ok(DecoderLayerDecodeValidation {
        geometry,
        cache_tokens,
    })
}

fn linear_no_bias_f32(
    input: &[f32],
    rows: usize,
    input_width: usize,
    weights: &[f32],
    output_width: usize,
) -> Result<Vec<f32>, CpuRefError> {
    let zero_bias = vec![0.0_f32; output_width];
    linear_f32(input, rows, input_width, weights, &zero_bias, output_width)
}

fn decoder_layer_trace_from_validated_operands<F>(
    input: &[f32],
    config: DecoderLayerConfig,
    raw_cos: &[f32],
    raw_sin: &[f32],
    parameters: DecoderLayerParameters<'_>,
    geometry: DecoderLayerGeometry,
    attention: F,
) -> Result<DecoderLayerPrefillTrace, CpuRefError>
where
    F: FnOnce(&[f32], &[f32], &[f32]) -> Result<(DecoderPrefillKvCache, Vec<f32>), CpuRefError>,
{
    let norm1 = rms_norm_f32(
        input,
        config.tokens,
        config.hidden_size,
        parameters.input_norm_weight,
        config.rms_norm_epsilon,
    )?;
    let query = linear_no_bias_f32(
        &norm1,
        config.tokens,
        config.hidden_size,
        parameters.query_weight,
        geometry.query_width,
    )?;
    let key = linear_no_bias_f32(
        &norm1,
        config.tokens,
        config.hidden_size,
        parameters.key_weight,
        geometry.key_value_width,
    )?;
    let value = linear_no_bias_f32(
        &norm1,
        config.tokens,
        config.hidden_size,
        parameters.value_weight,
        geometry.key_value_width,
    )?;
    let (mrope_query, mrope_key) = apply_multimodal_rope_f32(
        &query,
        &key,
        config.tokens,
        config.query_heads,
        config.key_value_heads,
        config.head_dim,
        raw_cos,
        raw_sin,
        config.mrope_sections,
    )?;
    let (kv_cache, attention_context) = attention(&mrope_query, &mrope_key, &value)?;
    let attention_output = linear_no_bias_f32(
        &attention_context,
        config.tokens,
        geometry.query_width,
        parameters.attention_output_weight,
        config.hidden_size,
    )?;
    let attention_residual = add_vectors_f32(input, &attention_output)?;
    let norm2 = rms_norm_f32(
        &attention_residual,
        config.tokens,
        config.hidden_size,
        parameters.post_attention_norm_weight,
        config.rms_norm_epsilon,
    )?;
    let mlp_gate = linear_no_bias_f32(
        &norm2,
        config.tokens,
        config.hidden_size,
        parameters.gate_weight,
        config.intermediate_size,
    )?;
    let mlp_up = linear_no_bias_f32(
        &norm2,
        config.tokens,
        config.hidden_size,
        parameters.up_weight,
        config.intermediate_size,
    )?;
    let mlp_activation = mlp_gate
        .iter()
        .zip(&mlp_up)
        .map(|(&gate, &up)| silu(gate) * up)
        .collect::<Vec<_>>();
    let mlp_down = linear_no_bias_f32(
        &mlp_activation,
        config.tokens,
        config.intermediate_size,
        parameters.down_weight,
        config.hidden_size,
    )?;
    let output = add_vectors_f32(&attention_residual, &mlp_down)?;

    Ok(DecoderLayerPrefillTrace {
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
    })
}

fn append_decoder_kv_cache(
    prefix_cache: &DecoderPrefillKvCache,
    rotated_key: &[f32],
    value: &[f32],
    cache_tokens: usize,
) -> Result<DecoderPrefillKvCache, CpuRefError> {
    validate_kv_geometry(
        rotated_key,
        value,
        1,
        prefix_cache.key_value_heads,
        prefix_cache.head_dim,
    )?;
    let key_value_elements = cache_tokens
        .checked_mul(prefix_cache.key_value_heads)
        .and_then(|elements| elements.checked_mul(prefix_cache.head_dim))
        .ok_or_else(dimension_error)?;
    let mut keys = Vec::with_capacity(key_value_elements);
    keys.extend_from_slice(&prefix_cache.keys);
    keys.extend_from_slice(rotated_key);
    let mut values = Vec::with_capacity(key_value_elements);
    values.extend_from_slice(&prefix_cache.values);
    values.extend_from_slice(value);
    Ok(DecoderPrefillKvCache {
        keys,
        values,
        tokens: cache_tokens,
        key_value_heads: prefix_cache.key_value_heads,
        head_dim: prefix_cache.head_dim,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillLmHeadConfig {
    pub tokens: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LmHeadTopKConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub k: usize,
    pub chunk_size: usize,
}

/// Projects only the last token of a token-major decoder prefill through a
/// bias-free, output-major language-model head.
pub fn prefill_last_logits_f32(
    final_norm: &[f32],
    config: PrefillLmHeadConfig,
    lm_head_weight: &[f32],
) -> Result<Vec<f32>, CpuRefError> {
    if config.tokens == 0 || config.hidden_size == 0 || config.vocab_size == 0 {
        return Err(dimension_error());
    }
    require_len(final_norm, config.tokens, config.hidden_size)?;
    require_len(lm_head_weight, config.vocab_size, config.hidden_size)?;
    require_finite(final_norm)?;
    require_finite(lm_head_weight)?;

    let last_row_start = final_norm.len() - config.hidden_size;
    linear_no_bias_f32(
        &final_norm[last_row_start..],
        1,
        config.hidden_size,
        lm_head_weight,
        config.vocab_size,
    )
}

/// Applies the pinned PaddleOCR-VL-1.6 prefill LM head to the last token.
pub fn pinned_prefill_last_logits_f32(
    final_norm: &[f32],
    tokens: usize,
    lm_head_weight: &[f32],
) -> Result<Vec<f32>, CpuRefError> {
    prefill_last_logits_f32(
        final_norm,
        PrefillLmHeadConfig {
            tokens,
            hidden_size: PINNED_DECODER_HIDDEN_SIZE,
            vocab_size: PINNED_DECODER_VOCAB_SIZE,
        },
        lm_head_weight,
    )
}

/// Projects one final-norm row through a bias-free, output-major LM head
/// while retaining only the globally best `k` vocabulary entries.
pub fn chunked_lm_head_top_k_f32(
    final_norm_one_row: &[f32],
    config: LmHeadTopKConfig,
    lm_head_weight: &[f32],
) -> Result<Vec<TopKEntry>, CpuRefError> {
    if config.hidden_size == 0 || config.vocab_size == 0 {
        return Err(dimension_error());
    }
    if config.chunk_size == 0 {
        return Err(CpuRefError::new(
            CpuRefErrorCode::InvalidTileSize,
            "LM-head chunk size must be nonzero",
        ));
    }
    if config.k > config.vocab_size {
        return Err(invalid_top_k_error());
    }
    require_len(final_norm_one_row, 1, config.hidden_size)?;
    require_len(lm_head_weight, config.vocab_size, config.hidden_size)?;
    require_finite(final_norm_one_row)?;
    require_finite(lm_head_weight)?;
    if config.k == 0 {
        return Ok(Vec::new());
    }

    let mut best = Vec::with_capacity(config.k);
    let mut chunk_start = 0_usize;
    while chunk_start < config.vocab_size {
        let remaining = config.vocab_size - chunk_start;
        let chunk_len = remaining.min(config.chunk_size);
        let chunk_end = chunk_start
            .checked_add(chunk_len)
            .ok_or_else(dimension_error)?;
        let weight_start = chunk_start
            .checked_mul(config.hidden_size)
            .ok_or_else(dimension_error)?;
        let weight_end = chunk_end
            .checked_mul(config.hidden_size)
            .ok_or_else(dimension_error)?;
        let chunk_logits = linear_no_bias_f32(
            final_norm_one_row,
            1,
            config.hidden_size,
            &lm_head_weight[weight_start..weight_end],
            chunk_len,
        )?;
        for (local_index, value) in chunk_logits.into_iter().enumerate() {
            let index = chunk_start
                .checked_add(local_index)
                .ok_or_else(dimension_error)?;
            best.push(TopKEntry { index, value });
        }
        best.sort_unstable_by(compare_top_k_entries);
        best.truncate(config.k);
        chunk_start = chunk_end;
    }
    Ok(best)
}

/// Applies the pinned PaddleOCR-VL-1.6 LM head to one final-norm row while
/// retaining only the globally best `k` vocabulary entries.
pub fn pinned_chunked_lm_head_top_k_f32(
    final_norm_one_row: &[f32],
    k: usize,
    chunk_size: usize,
    lm_head_weight: &[f32],
) -> Result<Vec<TopKEntry>, CpuRefError> {
    chunked_lm_head_top_k_f32(
        final_norm_one_row,
        LmHeadTopKConfig {
            hidden_size: PINNED_DECODER_HIDDEN_SIZE,
            vocab_size: PINNED_DECODER_VOCAB_SIZE,
            k,
            chunk_size,
        },
        lm_head_weight,
    )
}

/// Composes one token-major, bias-free decoder prefill layer from the accepted
/// CPU reference primitives.
pub fn decoder_layer_prefill_f32(
    input: &[f32],
    config: DecoderLayerConfig,
    raw_cos: &[f32],
    raw_sin: &[f32],
    parameters: DecoderLayerParameters<'_>,
) -> Result<DecoderLayerPrefillTrace, CpuRefError> {
    let geometry = validate_decoder_layer_prefill(input, config, raw_cos, raw_sin, &parameters)?;
    decoder_layer_trace_from_validated_operands(
        input,
        config,
        raw_cos,
        raw_sin,
        parameters,
        geometry,
        |mrope_query, mrope_key, value| {
            let kv_cache = write_decoder_prefill_kv_f32(
                mrope_key,
                value,
                config.tokens,
                config.key_value_heads,
                config.head_dim,
            )?;
            let attention_context = causal_gqa_f32(
                mrope_query,
                &kv_cache.keys,
                &kv_cache.values,
                config.tokens,
                config.query_heads,
                config.key_value_heads,
                config.head_dim,
            )?;
            Ok((kv_cache, attention_context))
        },
    )
}

/// Composes the pinned PaddleOCR-VL-1.6 decoder prefill layer topology.
pub fn pinned_decoder_layer_prefill_f32(
    input: &[f32],
    tokens: usize,
    raw_cos: &[f32],
    raw_sin: &[f32],
    parameters: DecoderLayerParameters<'_>,
) -> Result<DecoderLayerPrefillTrace, CpuRefError> {
    decoder_layer_prefill_f32(
        input,
        pinned_decoder_layer_config(tokens),
        raw_cos,
        raw_sin,
        parameters,
    )
}

/// Composes one cached single-token decoder layer against a detached prefix KV
/// cache without mutating the provided operands or cache.
pub fn decoder_layer_decode_f32(
    input: &[f32],
    config: DecoderLayerConfig,
    raw_cos: &[f32],
    raw_sin: &[f32],
    prefix_cache: &DecoderPrefillKvCache,
    parameters: DecoderLayerParameters<'_>,
) -> Result<DecoderLayerDecodeTrace, CpuRefError> {
    let validation =
        validate_decoder_layer_decode(input, config, raw_cos, raw_sin, prefix_cache, &parameters)?;
    decoder_layer_trace_from_validated_operands(
        input,
        config,
        raw_cos,
        raw_sin,
        parameters,
        validation.geometry,
        |mrope_query, mrope_key, value| {
            let kv_cache =
                append_decoder_kv_cache(prefix_cache, mrope_key, value, validation.cache_tokens)?;
            let attention_context = decode_gqa_f32(
                mrope_query,
                &kv_cache.keys,
                &kv_cache.values,
                kv_cache.tokens,
                config.query_heads,
                config.key_value_heads,
                config.head_dim,
            )?;
            Ok((kv_cache, attention_context))
        },
    )
}

/// Composes the pinned PaddleOCR-VL-1.6 single-token cached decoder layer.
pub fn pinned_decoder_layer_decode_f32(
    input: &[f32],
    raw_cos: &[f32],
    raw_sin: &[f32],
    prefix_cache: &DecoderPrefillKvCache,
    parameters: DecoderLayerParameters<'_>,
) -> Result<DecoderLayerDecodeTrace, CpuRefError> {
    decoder_layer_decode_f32(
        input,
        pinned_decoder_layer_config(1),
        raw_cos,
        raw_sin,
        prefix_cache,
        parameters,
    )
}

#[derive(Clone, Copy)]
struct DecoderStackValidation {
    geometry: DecoderLayerGeometry,
    expected_cache_tokens: usize,
    retained_checkpoint_elements: usize,
    retained_kv_elements: usize,
}

fn validate_decoder_stack_prefill(
    input: &[f32],
    config: DecoderStackConfig,
    checkpoint_layers: &[usize],
    final_norm_weight: &[f32],
) -> Result<DecoderStackValidation, CpuRefError> {
    if config.layers == 0 {
        return Err(dimension_error());
    }
    let geometry = validate_decoder_layer_config(config.layer)?;
    if input.len() != geometry.layer_elements || final_norm_weight.len() != config.layer.hidden_size
    {
        return Err(dimension_error());
    }
    require_finite(input)?;
    require_finite(final_norm_weight)?;
    validate_checkpoint_selection(checkpoint_layers, config.layers)?;

    let retained_checkpoint_elements = geometry
        .layer_elements
        .checked_mul(checkpoint_layers.len())
        .ok_or_else(dimension_error)?;
    let retained_kv_elements = geometry
        .key_value_elements
        .checked_mul(2)
        .and_then(|elements| elements.checked_mul(config.layers))
        .ok_or_else(dimension_error)?;

    Ok(DecoderStackValidation {
        geometry,
        expected_cache_tokens: config.layer.tokens,
        retained_checkpoint_elements,
        retained_kv_elements,
    })
}

fn validate_decoder_stack_decode(
    input: &[f32],
    config: DecoderStackConfig,
    prefix_caches: &[DecoderPrefillKvCache],
    checkpoint_layers: &[usize],
    final_norm_weight: &[f32],
) -> Result<DecoderStackValidation, CpuRefError> {
    if config.layers == 0 || config.layer.tokens != 1 {
        return Err(dimension_error());
    }
    let geometry = validate_decoder_layer_config(config.layer)?;
    if input.len() != geometry.layer_elements || final_norm_weight.len() != config.layer.hidden_size
    {
        return Err(dimension_error());
    }
    require_finite(input)?;
    require_finite(final_norm_weight)?;
    validate_checkpoint_selection(checkpoint_layers, config.layers)?;
    if prefix_caches.len() != config.layers {
        return Err(dimension_error());
    }

    let mut prefix_tokens = None;
    for cache in prefix_caches {
        if cache.key_value_heads != config.layer.key_value_heads
            || cache.head_dim != config.layer.head_dim
        {
            return Err(dimension_error());
        }
        validate_kv_geometry(
            &cache.keys,
            &cache.values,
            cache.tokens,
            cache.key_value_heads,
            cache.head_dim,
        )?;
        match prefix_tokens {
            Some(expected) if cache.tokens != expected => return Err(dimension_error()),
            Some(_) => {}
            None => prefix_tokens = Some(cache.tokens),
        }
    }

    let expected_cache_tokens = prefix_tokens
        .ok_or_else(dimension_error)?
        .checked_add(1)
        .ok_or_else(dimension_error)?;
    let key_value_elements = expected_cache_tokens
        .checked_mul(geometry.key_value_width)
        .ok_or_else(dimension_error)?;
    let retained_checkpoint_elements = geometry
        .layer_elements
        .checked_mul(checkpoint_layers.len())
        .ok_or_else(dimension_error)?;
    let retained_kv_elements = key_value_elements
        .checked_mul(2)
        .and_then(|elements| elements.checked_mul(config.layers))
        .ok_or_else(dimension_error)?;

    Ok(DecoderStackValidation {
        geometry,
        expected_cache_tokens,
        retained_checkpoint_elements,
        retained_kv_elements,
    })
}

fn try_reserve_exact<T>(values: &mut Vec<T>, additional: usize) -> Result<(), CpuRefError> {
    values
        .try_reserve_exact(additional)
        .map_err(|_| dimension_error())
}

fn try_clone_f32(values: &[f32]) -> Result<Vec<f32>, CpuRefError> {
    let mut clone = Vec::new();
    try_reserve_exact(&mut clone, values.len())?;
    clone.extend_from_slice(values);
    Ok(clone)
}

fn validate_decoder_stack_layer_buffers(
    output: &[f32],
    kv_cache: &DecoderPrefillKvCache,
    geometry: DecoderLayerGeometry,
    key_value_heads: usize,
    head_dim: usize,
    expected_cache_tokens: usize,
) -> Result<(), CpuRefError> {
    if output.len() != geometry.layer_elements
        || kv_cache.tokens != expected_cache_tokens
        || kv_cache.key_value_heads != key_value_heads
        || kv_cache.head_dim != head_dim
    {
        return Err(dimension_error());
    }
    require_finite(output)?;
    validate_kv_geometry(
        &kv_cache.keys,
        &kv_cache.values,
        expected_cache_tokens,
        key_value_heads,
        head_dim,
    )?;
    Ok(())
}

fn decoder_stack_trace_from_validated_layers<F>(
    input: &[f32],
    config: DecoderStackConfig,
    final_norm_weight: &[f32],
    checkpoint_layers: &[usize],
    validation: DecoderStackValidation,
    mut execute_layer: F,
) -> Result<DecoderStackPrefillTrace, CpuRefError>
where
    F: FnMut(
        usize,
        DecoderLayerConfig,
        &[f32],
    ) -> Result<(Vec<f32>, DecoderPrefillKvCache), CpuRefError>,
{
    let mut checkpoints = Vec::new();
    try_reserve_exact(&mut checkpoints, checkpoint_layers.len())?;
    let mut kv_caches = Vec::new();
    try_reserve_exact(&mut kv_caches, config.layers)?;

    let mut checkpoint_cursor = 0_usize;
    let mut current: Option<Vec<f32>> = None;
    for layer_index in 0..config.layers {
        let current_values = current.as_deref().unwrap_or(input);
        let (output, kv_cache) = execute_layer(layer_index, config.layer, current_values)?;
        validate_decoder_stack_layer_buffers(
            &output,
            &kv_cache,
            validation.geometry,
            config.layer.key_value_heads,
            config.layer.head_dim,
            validation.expected_cache_tokens,
        )?;

        if checkpoint_layers.get(checkpoint_cursor) == Some(&layer_index) {
            checkpoints.push(DecoderStackCheckpoint {
                layer_index,
                values: try_clone_f32(&output)?,
            });
            checkpoint_cursor += 1;
        }
        kv_caches.push(kv_cache);
        current = Some(output);
    }

    let final_norm = rms_norm_f32(
        current
            .as_deref()
            .expect("nonzero decoder layer count was validated"),
        config.layer.tokens,
        config.layer.hidden_size,
        final_norm_weight,
        config.layer.rms_norm_epsilon,
    )?;
    drop(current);

    Ok(DecoderStackPrefillTrace {
        checkpoints,
        kv_caches,
        final_norm,
        executed_layers: config.layers,
        retained_checkpoint_elements: validation.retained_checkpoint_elements,
        retained_kv_elements: validation.retained_kv_elements,
    })
}

/// Streams a decoder prefill stack while retaining only selected layer outputs
/// and every layer's detached KV cache.
pub fn decoder_stack_prefill_f32<F>(
    input: &[f32],
    config: DecoderStackConfig,
    checkpoint_layers: &[usize],
    final_norm_weight: &[f32],
    mut execute_layer: F,
) -> Result<DecoderStackPrefillTrace, CpuRefError>
where
    F: FnMut(usize, DecoderLayerConfig, &[f32]) -> Result<DecoderLayerPrefillTrace, CpuRefError>,
{
    let validation =
        validate_decoder_stack_prefill(input, config, checkpoint_layers, final_norm_weight)?;
    decoder_stack_trace_from_validated_layers(
        input,
        config,
        final_norm_weight,
        checkpoint_layers,
        validation,
        |layer_index, layer_config, current_values| {
            let trace = execute_layer(layer_index, layer_config, current_values)?;
            let DecoderLayerPrefillTrace {
                output, kv_cache, ..
            } = trace;
            Ok((output, kv_cache))
        },
    )
}

/// Streams the pinned 18-layer PaddleOCR-VL-1.6 decoder prefill stack.
pub fn pinned_decoder_stack_prefill_f32<F>(
    input: &[f32],
    tokens: usize,
    checkpoint_layers: &[usize],
    final_norm_weight: &[f32],
    execute_layer: F,
) -> Result<DecoderStackPrefillTrace, CpuRefError>
where
    F: FnMut(usize, DecoderLayerConfig, &[f32]) -> Result<DecoderLayerPrefillTrace, CpuRefError>,
{
    decoder_stack_prefill_f32(
        input,
        DecoderStackConfig {
            layer: pinned_decoder_layer_config(tokens),
            layers: PINNED_DECODER_LAYERS,
        },
        checkpoint_layers,
        final_norm_weight,
        execute_layer,
    )
}

/// Streams a cached single-token decoder stack while retaining only selected
/// layer outputs and every layer's owned appended KV cache.
pub fn decoder_stack_decode_f32<F>(
    input: &[f32],
    config: DecoderStackConfig,
    prefix_caches: &[DecoderPrefillKvCache],
    checkpoint_layers: &[usize],
    final_norm_weight: &[f32],
    mut execute_layer: F,
) -> Result<DecoderStackDecodeTrace, CpuRefError>
where
    F: FnMut(
        usize,
        DecoderLayerConfig,
        &[f32],
        &DecoderPrefillKvCache,
    ) -> Result<DecoderLayerDecodeTrace, CpuRefError>,
{
    let validation = validate_decoder_stack_decode(
        input,
        config,
        prefix_caches,
        checkpoint_layers,
        final_norm_weight,
    )?;
    decoder_stack_trace_from_validated_layers(
        input,
        config,
        final_norm_weight,
        checkpoint_layers,
        validation,
        |layer_index, layer_config, current_values| {
            let trace = execute_layer(
                layer_index,
                layer_config,
                current_values,
                &prefix_caches[layer_index],
            )?;
            let DecoderLayerPrefillTrace {
                output, kv_cache, ..
            } = trace;
            Ok((output, kv_cache))
        },
    )
}

/// Streams the pinned 18-layer PaddleOCR-VL-1.6 cached single-token decoder
/// stack.
pub fn pinned_decoder_stack_decode_f32<F>(
    input: &[f32],
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
        DecoderStackConfig {
            layer: pinned_decoder_layer_config(1),
            layers: PINNED_DECODER_LAYERS,
        },
        prefix_caches,
        checkpoint_layers,
        final_norm_weight,
        execute_layer,
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TopKEntry {
    pub index: usize,
    pub value: f32,
}

fn compare_top_k_entries(left: &TopKEntry, right: &TopKEntry) -> Ordering {
    right
        .value
        .total_cmp(&left.value)
        .then_with(|| left.index.cmp(&right.index))
}

const fn invalid_top_k_error() -> CpuRefError {
    CpuRefError::new(
        CpuRefErrorCode::InvalidK,
        "top-k exceeds the number of logits",
    )
}

pub fn top_k(values: &[f32], k: usize) -> Result<Vec<TopKEntry>, CpuRefError> {
    if k > values.len() {
        return Err(invalid_top_k_error());
    }
    require_finite(values)?;
    let mut entries: Vec<_> = values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| TopKEntry { index, value })
        .collect();
    entries.sort_unstable_by(compare_top_k_entries);
    entries.truncate(k);
    Ok(entries)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GreedyGenerationConfig {
    pub layers: usize,
    pub vocab_size: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub max_new_tokens: usize,
    pub eos_token_id: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GreedyDecodeStep {
    pub top_k: Vec<TopKEntry>,
    pub kv_caches: Vec<DecoderPrefillKvCache>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GreedyStopReason {
    MaxNewTokens,
    EosToken,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GreedyGenerationTrace {
    pub generated_tokens: Vec<usize>,
    pub kv_caches: Vec<DecoderPrefillKvCache>,
    pub decode_steps: usize,
    pub stop_reason: GreedyStopReason,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GreedyChunkConfig {
    pub generation: GreedyGenerationConfig,
    pub decode_chunk_size: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GreedyDecodeChunk {
    pub steps: Vec<GreedyDecodeStep>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GreedyChunkedGenerationTrace {
    pub generation: GreedyGenerationTrace,
    pub decode_chunks: usize,
}

fn validate_greedy_generation_config(config: GreedyGenerationConfig) -> Result<(), CpuRefError> {
    if config.layers == 0
        || config.vocab_size == 0
        || config.key_value_heads == 0
        || config.head_dim == 0
        || config.max_new_tokens == 0
        || config.eos_token_id >= config.vocab_size
    {
        return Err(dimension_error());
    }
    Ok(())
}

fn validate_generation_candidates(
    candidates: &[TopKEntry],
    vocab_size: usize,
) -> Result<(), CpuRefError> {
    if candidates.is_empty() || candidates.len() > vocab_size {
        return Err(dimension_error());
    }
    for candidate in candidates {
        require_finite(std::slice::from_ref(&candidate.value))?;
    }

    let mut token_ids = HashSet::new();
    token_ids
        .try_reserve(candidates.len())
        .map_err(|_| dimension_error())?;
    for candidate in candidates {
        if candidate.index >= vocab_size || !token_ids.insert(candidate.index) {
            return Err(dimension_error());
        }
    }
    if candidates
        .windows(2)
        .any(|pair| compare_top_k_entries(&pair[0], &pair[1]) != Ordering::Less)
    {
        return Err(dimension_error());
    }
    Ok(())
}

fn validate_generation_caches(
    caches: &[DecoderPrefillKvCache],
    config: GreedyGenerationConfig,
    expected_tokens: Option<usize>,
) -> Result<usize, CpuRefError> {
    if caches.len() != config.layers {
        return Err(dimension_error());
    }
    let tokens = caches[0].tokens;
    if tokens == 0 || expected_tokens.is_some_and(|expected| tokens != expected) {
        return Err(dimension_error());
    }
    for cache in caches {
        if cache.tokens != tokens
            || cache.key_value_heads != config.key_value_heads
            || cache.head_dim != config.head_dim
        {
            return Err(dimension_error());
        }
        validate_kv_geometry(
            &cache.keys,
            &cache.values,
            tokens,
            config.key_value_heads,
            config.head_dim,
        )?;
    }
    Ok(tokens)
}

struct GreedyGenerationState {
    config: GreedyGenerationConfig,
    generated_tokens: Vec<usize>,
    current_token: usize,
    current_caches: Vec<DecoderPrefillKvCache>,
    cache_tokens: usize,
    decode_steps: usize,
}

impl GreedyGenerationState {
    fn initialize(
        prefill_top_k: &[TopKEntry],
        initial_kv_caches: Vec<DecoderPrefillKvCache>,
        config: GreedyGenerationConfig,
    ) -> Result<Self, CpuRefError> {
        validate_greedy_generation_config(config)?;
        validate_generation_candidates(prefill_top_k, config.vocab_size)?;
        let cache_tokens = validate_generation_caches(&initial_kv_caches, config, None)?;
        let current_token = prefill_top_k[0].index;
        let generated_tokens = vec![current_token];
        Ok(Self {
            config,
            generated_tokens,
            current_token,
            current_caches: initial_kv_caches,
            cache_tokens,
            decode_steps: 0,
        })
    }

    fn terminal_reason(&self) -> Option<GreedyStopReason> {
        if self.current_token == self.config.eos_token_id {
            Some(GreedyStopReason::EosToken)
        } else if self.generated_tokens.len() == self.config.max_new_tokens {
            Some(GreedyStopReason::MaxNewTokens)
        } else {
            None
        }
    }

    fn remaining_decode_steps(&self) -> Result<usize, CpuRefError> {
        self.config
            .max_new_tokens
            .checked_sub(self.generated_tokens.len())
            .ok_or_else(dimension_error)
    }

    fn apply_decode_step(
        &mut self,
        decoded: GreedyDecodeStep,
    ) -> Result<Option<GreedyStopReason>, CpuRefError> {
        validate_generation_candidates(&decoded.top_k, self.config.vocab_size)?;
        let next_cache_tokens = self
            .cache_tokens
            .checked_add(1)
            .ok_or_else(dimension_error)?;
        validate_generation_caches(&decoded.kv_caches, self.config, Some(next_cache_tokens))?;
        let next_decode_steps = self
            .decode_steps
            .checked_add(1)
            .ok_or_else(dimension_error)?;

        self.current_token = decoded.top_k[0].index;
        self.current_caches = decoded.kv_caches;
        self.cache_tokens = next_cache_tokens;
        self.generated_tokens.push(self.current_token);
        self.decode_steps = next_decode_steps;
        Ok(self.terminal_reason())
    }

    fn into_trace(self, stop_reason: GreedyStopReason) -> GreedyGenerationTrace {
        GreedyGenerationTrace {
            generated_tokens: self.generated_tokens,
            kv_caches: self.current_caches,
            decode_steps: self.decode_steps,
            stop_reason,
        }
    }
}

/// Runs deterministic top-1 generation over caller-supplied single-token
/// decode steps while retaining only the current layer KV caches.
pub fn greedy_generate_f32<F>(
    prefill_top_k: &[TopKEntry],
    initial_kv_caches: Vec<DecoderPrefillKvCache>,
    config: GreedyGenerationConfig,
    mut decode_step: F,
) -> Result<GreedyGenerationTrace, CpuRefError>
where
    F: FnMut(usize, usize, &[DecoderPrefillKvCache]) -> Result<GreedyDecodeStep, CpuRefError>,
{
    let mut state = GreedyGenerationState::initialize(prefill_top_k, initial_kv_caches, config)?;
    if let Some(stop_reason) = state.terminal_reason() {
        return Ok(state.into_trace(stop_reason));
    }
    loop {
        let decoded = decode_step(
            state.decode_steps,
            state.current_token,
            &state.current_caches,
        )?;
        if let Some(stop_reason) = state.apply_decode_step(decoded)? {
            return Ok(state.into_trace(stop_reason));
        }
    }
}

/// Runs deterministic greedy generation using caller-supplied batches of
/// sequential single-token decode results.
pub fn greedy_generate_chunked_f32<F>(
    prefill_top_k: &[TopKEntry],
    initial_kv_caches: Vec<DecoderPrefillKvCache>,
    config: GreedyChunkConfig,
    mut decode_chunk: F,
) -> Result<GreedyChunkedGenerationTrace, CpuRefError>
where
    F: FnMut(
        usize,
        usize,
        &[DecoderPrefillKvCache],
        usize,
    ) -> Result<GreedyDecodeChunk, CpuRefError>,
{
    if config.decode_chunk_size == 0 {
        return Err(dimension_error());
    }
    let mut state =
        GreedyGenerationState::initialize(prefill_top_k, initial_kv_caches, config.generation)?;
    if let Some(stop_reason) = state.terminal_reason() {
        return Ok(GreedyChunkedGenerationTrace {
            generation: state.into_trace(stop_reason),
            decode_chunks: 0,
        });
    }

    let mut decode_chunks = 0_usize;
    loop {
        let requested_steps = config
            .decode_chunk_size
            .min(state.remaining_decode_steps()?);
        let decoded_chunk = decode_chunk(
            decode_chunks,
            state.current_token,
            &state.current_caches,
            requested_steps,
        )?;
        let returned_steps = decoded_chunk.steps.len();
        if returned_steps == 0 || returned_steps > requested_steps {
            return Err(dimension_error());
        }

        let mut terminal_reason = None;
        for (step_index, decoded) in decoded_chunk.steps.into_iter().enumerate() {
            let step_terminal_reason = state.apply_decode_step(decoded)?;
            if step_terminal_reason.is_some() && step_index + 1 != returned_steps {
                return Err(dimension_error());
            }
            terminal_reason = step_terminal_reason;
        }
        decode_chunks = decode_chunks.checked_add(1).ok_or_else(dimension_error)?;
        if let Some(stop_reason) = terminal_reason {
            return Ok(GreedyChunkedGenerationTrace {
                generation: state.into_trace(stop_reason),
                decode_chunks,
            });
        }
    }
}

const fn pinned_greedy_generation_config(max_new_tokens: usize) -> GreedyGenerationConfig {
    GreedyGenerationConfig {
        layers: PINNED_DECODER_LAYERS,
        vocab_size: PINNED_DECODER_VOCAB_SIZE,
        key_value_heads: PINNED_DECODER_KEY_VALUE_HEADS,
        head_dim: PINNED_DECODER_HEAD_DIM,
        max_new_tokens,
        eos_token_id: PINNED_DECODER_EOS_TOKEN_ID,
    }
}

/// Runs greedy generation with the pinned PaddleOCR-VL-1.6 decoder topology.
pub fn pinned_greedy_generate_f32<F>(
    prefill_top_k: &[TopKEntry],
    initial_kv_caches: Vec<DecoderPrefillKvCache>,
    max_new_tokens: usize,
    decode_step: F,
) -> Result<GreedyGenerationTrace, CpuRefError>
where
    F: FnMut(usize, usize, &[DecoderPrefillKvCache]) -> Result<GreedyDecodeStep, CpuRefError>,
{
    greedy_generate_f32(
        prefill_top_k,
        initial_kv_caches,
        pinned_greedy_generation_config(max_new_tokens),
        decode_step,
    )
}

/// Runs chunked greedy generation with the pinned PaddleOCR-VL-1.6 decoder
/// topology.
pub fn pinned_greedy_generate_chunked_f32<F>(
    prefill_top_k: &[TopKEntry],
    initial_kv_caches: Vec<DecoderPrefillKvCache>,
    max_new_tokens: usize,
    decode_chunk_size: usize,
    decode_chunk: F,
) -> Result<GreedyChunkedGenerationTrace, CpuRefError>
where
    F: FnMut(
        usize,
        usize,
        &[DecoderPrefillKvCache],
        usize,
    ) -> Result<GreedyDecodeChunk, CpuRefError>,
{
    greedy_generate_chunked_f32(
        prefill_top_k,
        initial_kv_caches,
        GreedyChunkConfig {
            generation: pinned_greedy_generation_config(max_new_tokens),
            decode_chunk_size,
        },
        decode_chunk,
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionPreprocessConfig {
    pub patch_size: usize,
    pub merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessedVisionInput {
    pub resized_height: usize,
    pub resized_width: usize,
    pub image_grid_thw: [usize; 3],
    pub values: Vec<f32>,
}

pub fn smart_resize_paddleocr_vl(
    height: usize,
    width: usize,
    config: VisionPreprocessConfig,
) -> Result<(usize, usize), CpuRefError> {
    validate_preprocess_config(config)?;
    if height == 0 || width == 0 {
        return Err(invalid_image_geometry());
    }

    let factor = config
        .patch_size
        .checked_mul(config.merge_size)
        .ok_or_else(invalid_preprocess_config)?;
    let mut adjusted_height = height;
    let mut adjusted_width = width;
    if adjusted_height < factor {
        adjusted_width = python_round_nonnegative(
            adjusted_width as f64 * factor as f64 / adjusted_height as f64,
        )?;
        adjusted_height = factor;
    }
    if adjusted_width < factor {
        adjusted_height = python_round_nonnegative(
            adjusted_height as f64 * factor as f64 / adjusted_width as f64,
        )?;
        adjusted_width = factor;
    }

    let longer = adjusted_height.max(adjusted_width) as f64;
    let shorter = adjusted_height.min(adjusted_width) as f64;
    if longer / shorter > 200.0 {
        return Err(invalid_image_geometry());
    }

    let factor_f64 = factor as f64;
    let mut resized_height = python_round_nonnegative(adjusted_height as f64 / factor_f64)?
        .checked_mul(factor)
        .ok_or_else(invalid_image_geometry)?;
    let mut resized_width = python_round_nonnegative(adjusted_width as f64 / factor_f64)?
        .checked_mul(factor)
        .ok_or_else(invalid_image_geometry)?;
    let rounded_pixels = resized_height
        .checked_mul(resized_width)
        .ok_or_else(invalid_image_geometry)?;
    let source_pixels = adjusted_height
        .checked_mul(adjusted_width)
        .ok_or_else(invalid_image_geometry)?;

    if rounded_pixels > config.max_pixels {
        let beta = (source_pixels as f64 / config.max_pixels as f64).sqrt();
        resized_height = ((adjusted_height as f64 / beta / factor_f64).floor() as usize)
            .checked_mul(factor)
            .ok_or_else(invalid_image_geometry)?;
        resized_width = ((adjusted_width as f64 / beta / factor_f64).floor() as usize)
            .checked_mul(factor)
            .ok_or_else(invalid_image_geometry)?;
    } else if rounded_pixels < config.min_pixels {
        let beta = (config.min_pixels as f64 / source_pixels as f64).sqrt();
        resized_height = ((adjusted_height as f64 * beta / factor_f64).ceil() as usize)
            .checked_mul(factor)
            .ok_or_else(invalid_image_geometry)?;
        resized_width = ((adjusted_width as f64 * beta / factor_f64).ceil() as usize)
            .checked_mul(factor)
            .ok_or_else(invalid_image_geometry)?;
    }

    if resized_height == 0 || resized_width == 0 {
        return Err(invalid_image_geometry());
    }
    Ok((resized_height, resized_width))
}

pub fn canonical_rgb8_to_patches(
    rgb: &[u8],
    height: usize,
    width: usize,
    config: VisionPreprocessConfig,
) -> Result<PreprocessedVisionInput, CpuRefError> {
    validate_preprocess_config(config)?;
    let source_elements = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(dimension_error)?;
    if rgb.len() != source_elements {
        return Err(dimension_error());
    }
    let (resized_height, resized_width) = smart_resize_paddleocr_vl(height, width, config)?;
    let resized = pillow_bicubic_rgb8(rgb, height, width, resized_height, resized_width)?;
    let grid_height = resized_height / config.patch_size;
    let grid_width = resized_width / config.patch_size;
    let output_elements = grid_height
        .checked_mul(grid_width)
        .and_then(|patches| patches.checked_mul(3))
        .and_then(|values| values.checked_mul(config.patch_size))
        .and_then(|values| values.checked_mul(config.patch_size))
        .ok_or_else(dimension_error)?;
    let mut values = Vec::with_capacity(output_elements);
    for patch_y in 0..grid_height {
        for patch_x in 0..grid_width {
            for channel in 0..3 {
                for local_y in 0..config.patch_size {
                    let source_y = patch_y * config.patch_size + local_y;
                    for local_x in 0..config.patch_size {
                        let source_x = patch_x * config.patch_size + local_x;
                        let byte = resized[(source_y * resized_width + source_x) * 3 + channel];
                        let rescaled = (f64::from(byte) / 255.0) as f32;
                        values.push((rescaled - 0.5) / 0.5);
                    }
                }
            }
        }
    }
    debug_assert_eq!(values.len(), output_elements);
    Ok(PreprocessedVisionInput {
        resized_height,
        resized_width,
        image_grid_thw: [1, grid_height, grid_width],
        values,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvBlockOrder {
    Forward,
    Reverse,
}

#[derive(Clone, Copy, Debug)]
pub struct LinearParameters<'a> {
    pub weight: &'a [f32],
    pub bias: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
pub struct LayerNormParameters<'a> {
    pub weight: &'a [f32],
    pub bias: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
pub struct ProjectorParameters<'a> {
    pub pre_norm: LayerNormParameters<'a>,
    pub linear1: LinearParameters<'a>,
    pub linear2: LinearParameters<'a>,
}

#[derive(Debug, PartialEq)]
pub struct ProjectorTrace {
    pub pre_norm: Vec<f32>,
    pub merged: Vec<f32>,
    pub linear1: Vec<f32>,
    pub activation: Vec<f32>,
    pub output: Vec<f32>,
}

pub fn projector_f32(
    image_features: &[f32],
    hidden_size: usize,
    image_grid_thw: &[[usize; 3]],
    parameters: ProjectorParameters<'_>,
    layer_norm_epsilon: f32,
) -> Result<ProjectorTrace, CpuRefError> {
    require_positive_finite_epsilon(layer_norm_epsilon)?;
    let layout = projector_layout(hidden_size, image_grid_thw)?;

    // Validate every shape and operand before allocating or executing a stage,
    // so malformed late-stage parameters cannot yield a partial trace.
    require_len(image_features, layout.input_tokens, hidden_size)?;
    if parameters.pre_norm.weight.len() != hidden_size
        || parameters.pre_norm.bias.len() != hidden_size
        || parameters.linear1.bias.len() != layout.merged_width
        || parameters.linear2.bias.is_empty()
    {
        return Err(dimension_error());
    }
    require_len(
        parameters.linear1.weight,
        layout.merged_width,
        layout.merged_width,
    )?;
    require_len(
        parameters.linear2.weight,
        parameters.linear2.bias.len(),
        layout.merged_width,
    )?;
    require_finite(image_features)?;
    require_finite(parameters.pre_norm.weight)?;
    require_finite(parameters.pre_norm.bias)?;
    require_finite(parameters.linear1.weight)?;
    require_finite(parameters.linear1.bias)?;
    require_finite(parameters.linear2.weight)?;
    require_finite(parameters.linear2.bias)?;

    let pre_norm = layer_norm_f32(
        image_features,
        layout.input_tokens,
        hidden_size,
        parameters.pre_norm.weight,
        parameters.pre_norm.bias,
        layer_norm_epsilon,
    )?;
    require_finite(&pre_norm)?;
    let merged = projector_merge_2x2_f32(&pre_norm, hidden_size, image_grid_thw)?;
    let linear1 = linear_f32(
        &merged,
        layout.output_tokens,
        layout.merged_width,
        parameters.linear1.weight,
        parameters.linear1.bias,
        layout.merged_width,
    )?;
    require_finite(&linear1)?;
    let activation = linear1
        .iter()
        .copied()
        .map(gelu_erf_f32)
        .collect::<Vec<_>>();
    require_finite(&activation)?;
    let output = linear_f32(
        &activation,
        layout.output_tokens,
        layout.merged_width,
        parameters.linear2.weight,
        parameters.linear2.bias,
        parameters.linear2.bias.len(),
    )?;
    require_finite(&output)?;

    Ok(ProjectorTrace {
        pre_norm,
        merged,
        linear1,
        activation,
        output,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionEncoderLayerConfig {
    pub tokens: usize,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub layer_norm_epsilon: f32,
    pub attention_key_tile: usize,
    pub attention_order: KvBlockOrder,
}

#[derive(Clone, Copy, Debug)]
pub struct VisionEncoderLayerParameters<'a> {
    pub norm1: LayerNormParameters<'a>,
    pub query: LinearParameters<'a>,
    pub key: LinearParameters<'a>,
    pub value: LinearParameters<'a>,
    pub attention_output: LinearParameters<'a>,
    pub norm2: LayerNormParameters<'a>,
    pub mlp_fc1: LinearParameters<'a>,
    pub mlp_fc2: LinearParameters<'a>,
}

#[derive(Debug, PartialEq)]
pub struct VisionEncoderLayerTrace {
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionEncoderStackConfig {
    pub tokens: usize,
    pub hidden_size: usize,
    pub layers: usize,
    pub layer_norm_epsilon: f32,
}

#[derive(Debug, PartialEq)]
pub struct VisionEncoderStackCheckpoint {
    pub layer_index: usize,
    pub values: Vec<f32>,
}

#[derive(Debug, PartialEq)]
pub struct VisionEncoderStackTrace {
    pub checkpoints: Vec<VisionEncoderStackCheckpoint>,
    pub output: Vec<f32>,
    pub executed_layers: usize,
    pub retained_checkpoint_elements: usize,
}

impl VisionEncoderStackTrace {
    #[must_use]
    pub fn checkpoint(&self, layer_index: usize) -> Option<&[f32]> {
        self.checkpoints
            .iter()
            .find(|checkpoint| checkpoint.layer_index == layer_index)
            .map(|checkpoint| checkpoint.values.as_slice())
    }
}

pub fn materialized_segmented_attention_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    cu_seqlens: &[usize],
) -> Result<Vec<f32>, CpuRefError> {
    let tensor_elements =
        validate_attention_inputs(query, key, value, tokens, heads, head_dim, cu_seqlens)?;
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0_f32; tensor_elements];

    for segment in cu_seqlens.windows(2) {
        let start = segment[0];
        let end = segment[1];
        let segment_tokens = end - start;
        for head in 0..heads {
            let score_elements = segment_tokens
                .checked_mul(segment_tokens)
                .ok_or_else(dimension_error)?;
            let mut scores = vec![0.0_f32; score_elements];
            for query_offset in 0..segment_tokens {
                let query_token = start + query_offset;
                for key_offset in 0..segment_tokens {
                    let key_token = start + key_offset;
                    scores[query_offset * segment_tokens + key_offset] =
                        attention_dot(query, key, query_token, key_token, head, heads, head_dim)
                            * scale;
                }
            }
            let probabilities = softmax_rows_f32(&scores, segment_tokens, segment_tokens, None)?;
            for query_offset in 0..segment_tokens {
                let output_token = start + query_offset;
                for dimension in 0..head_dim {
                    let mut accumulator = 0.0_f32;
                    for key_offset in 0..segment_tokens {
                        let key_token = start + key_offset;
                        let probability = probabilities[query_offset * segment_tokens + key_offset];
                        accumulator += probability
                            * value[attention_index(key_token, head, dimension, heads, head_dim)];
                    }
                    output[attention_index(output_token, head, dimension, heads, head_dim)] =
                        accumulator;
                }
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn streaming_segmented_attention_f32(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    cu_seqlens: &[usize],
    key_tile: usize,
    order: KvBlockOrder,
) -> Result<Vec<f32>, CpuRefError> {
    require_nonzero_attention_tile(key_tile)?;
    let tensor_elements =
        validate_attention_inputs(query, key, value, tokens, heads, head_dim, cu_seqlens)?;
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0_f32; tensor_elements];
    let mut weighted_values = vec![0.0_f32; head_dim];

    for segment in cu_seqlens.windows(2) {
        let segment_start = segment[0];
        let segment_end = segment[1];
        for query_token in segment_start..segment_end {
            for head in 0..heads {
                weighted_values.fill(0.0);
                let mut maximum = f32::NEG_INFINITY;
                let mut denominator = 0.0_f32;
                let mut process_key = |key_token: usize| {
                    let score =
                        attention_dot(query, key, query_token, key_token, head, heads, head_dim)
                            * scale;
                    let next_maximum = maximum.max(score);
                    let previous_scale = (maximum - next_maximum).exp();
                    let score_scale = (score - next_maximum).exp();
                    denominator = denominator * previous_scale + score_scale;
                    for (dimension, weighted_value) in weighted_values.iter_mut().enumerate() {
                        *weighted_value = *weighted_value * previous_scale
                            + value[attention_index(key_token, head, dimension, heads, head_dim)]
                                * score_scale;
                    }
                    maximum = next_maximum;
                };

                match order {
                    KvBlockOrder::Forward => {
                        let mut block_start = segment_start;
                        while block_start < segment_end {
                            let block_end = block_start.saturating_add(key_tile).min(segment_end);
                            for key_token in block_start..block_end {
                                process_key(key_token);
                            }
                            block_start = block_end;
                        }
                    }
                    KvBlockOrder::Reverse => {
                        let mut block_end = segment_end;
                        while block_end > segment_start {
                            let block_start = block_end.saturating_sub(key_tile).max(segment_start);
                            for key_token in block_start..block_end {
                                process_key(key_token);
                            }
                            block_end = block_start;
                        }
                    }
                }

                for (dimension, weighted_value) in weighted_values.iter().copied().enumerate() {
                    output[attention_index(query_token, head, dimension, heads, head_dim)] =
                        weighted_value / denominator;
                }
            }
        }
    }
    Ok(output)
}

pub fn vision_encoder_layer_identity_rope_f32(
    input: &[f32],
    config: VisionEncoderLayerConfig,
    cu_seqlens: &[usize],
    parameters: VisionEncoderLayerParameters<'_>,
) -> Result<VisionEncoderLayerTrace, CpuRefError> {
    validate_vision_encoder_layer_config(config)?;
    require_nonzero_attention_tile(config.attention_key_tile)?;
    validate_sequence_boundaries(config.tokens, cu_seqlens)?;

    let norm1 = layer_norm_f32(
        input,
        config.tokens,
        config.hidden_size,
        parameters.norm1.weight,
        parameters.norm1.bias,
        config.layer_norm_epsilon,
    )?;
    let query = linear_f32(
        &norm1,
        config.tokens,
        config.hidden_size,
        parameters.query.weight,
        parameters.query.bias,
        config.hidden_size,
    )?;
    let key = linear_f32(
        &norm1,
        config.tokens,
        config.hidden_size,
        parameters.key.weight,
        parameters.key.bias,
        config.hidden_size,
    )?;
    let value = linear_f32(
        &norm1,
        config.tokens,
        config.hidden_size,
        parameters.value.weight,
        parameters.value.bias,
        config.hidden_size,
    )?;
    let attention_context = streaming_segmented_attention_f32(
        &query,
        &key,
        &value,
        config.tokens,
        config.attention_heads,
        config.head_dim,
        cu_seqlens,
        config.attention_key_tile,
        config.attention_order,
    )?;
    let attention_output = linear_f32(
        &attention_context,
        config.tokens,
        config.hidden_size,
        parameters.attention_output.weight,
        parameters.attention_output.bias,
        config.hidden_size,
    )?;
    let attention_residual = add_vectors_f32(input, &attention_output)?;
    let norm2 = layer_norm_f32(
        &attention_residual,
        config.tokens,
        config.hidden_size,
        parameters.norm2.weight,
        parameters.norm2.bias,
        config.layer_norm_epsilon,
    )?;
    let mlp_fc1 = linear_f32(
        &norm2,
        config.tokens,
        config.hidden_size,
        parameters.mlp_fc1.weight,
        parameters.mlp_fc1.bias,
        config.intermediate_size,
    )?;
    let mlp_activation = mlp_fc1
        .iter()
        .copied()
        .map(gelu_pytorch_tanh)
        .collect::<Vec<_>>();
    let mlp_output = linear_f32(
        &mlp_activation,
        config.tokens,
        config.intermediate_size,
        parameters.mlp_fc2.weight,
        parameters.mlp_fc2.bias,
        config.hidden_size,
    )?;
    let output = add_vectors_f32(&attention_residual, &mlp_output)?;

    Ok(VisionEncoderLayerTrace {
        norm1,
        query,
        key,
        value,
        attention_context,
        attention_output,
        attention_residual,
        norm2,
        mlp_fc1,
        mlp_activation,
        mlp_output,
        output,
    })
}

pub fn vision_encoder_stack_identity_rope_f32<F, O>(
    input: &[f32],
    config: VisionEncoderStackConfig,
    checkpoint_layers: &[usize],
    post_norm: LayerNormParameters<'_>,
    mut execute_layer: F,
) -> Result<VisionEncoderStackTrace, CpuRefError>
where
    F: FnMut(usize, &[f32]) -> Result<O, CpuRefError>,
    O: AsRef<[f32]>,
{
    if config.tokens == 0 || config.hidden_size == 0 || config.layers == 0 {
        return Err(dimension_error());
    }
    let layer_elements = config
        .tokens
        .checked_mul(config.hidden_size)
        .ok_or_else(dimension_error)?;
    if input.len() != layer_elements
        || post_norm.weight.len() != config.hidden_size
        || post_norm.bias.len() != config.hidden_size
    {
        return Err(dimension_error());
    }
    require_positive_finite_epsilon(config.layer_norm_epsilon)?;
    require_finite(input)?;
    require_finite(post_norm.weight)?;
    require_finite(post_norm.bias)?;
    validate_checkpoint_selection(checkpoint_layers, config.layers)?;

    let mut checkpoints = Vec::with_capacity(checkpoint_layers.len());
    let mut retained_checkpoint_elements = 0_usize;
    let mut checkpoint_cursor = 0_usize;
    let mut current: Option<O> = None;

    for layer_index in 0..config.layers {
        let current_values = current.as_ref().map_or(input, |output| output.as_ref());
        let next = execute_layer(layer_index, current_values)?;
        let next_values = next.as_ref();
        if next_values.len() != layer_elements {
            return Err(dimension_error());
        }
        require_finite(next_values)?;

        if checkpoint_layers.get(checkpoint_cursor) == Some(&layer_index) {
            checkpoints.push(VisionEncoderStackCheckpoint {
                layer_index,
                values: next_values.to_vec(),
            });
            retained_checkpoint_elements = retained_checkpoint_elements
                .checked_add(layer_elements)
                .ok_or_else(dimension_error)?;
            checkpoint_cursor += 1;
        }
        current = Some(next);
    }

    let output = layer_norm_f32(
        current
            .as_ref()
            .expect("nonzero layer count was validated")
            .as_ref(),
        config.tokens,
        config.hidden_size,
        post_norm.weight,
        post_norm.bias,
        config.layer_norm_epsilon,
    )?;

    Ok(VisionEncoderStackTrace {
        checkpoints,
        output,
        executed_layers: config.layers,
        retained_checkpoint_elements,
    })
}

fn validate_preprocess_config(config: VisionPreprocessConfig) -> Result<(), CpuRefError> {
    if config.patch_size == 0
        || config.merge_size == 0
        || config.min_pixels == 0
        || config.max_pixels < config.min_pixels
        || config.patch_size.checked_mul(config.merge_size).is_none()
    {
        return Err(invalid_preprocess_config());
    }
    Ok(())
}

fn python_round_nonnegative(value: f64) -> Result<usize, CpuRefError> {
    if !value.is_finite() || value < 0.0 || value > usize::MAX as f64 {
        return Err(invalid_image_geometry());
    }
    Ok(value.round_ties_even() as usize)
}

#[derive(Debug)]
struct PillowCoefficients {
    kernel_size: usize,
    bounds: Vec<(usize, usize)>,
    weights: Vec<i32>,
}

fn pillow_bicubic_rgb8(
    input: &[u8],
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
) -> Result<Vec<u8>, CpuRefError> {
    if input_height == output_height && input_width == output_width {
        return Ok(input.to_vec());
    }
    let horizontal = if input_width != output_width {
        Some(pillow_bicubic_coefficients(input_width, output_width)?)
    } else {
        None
    };
    let vertical = if input_height != output_height {
        Some(pillow_bicubic_coefficients(input_height, output_height)?)
    } else {
        None
    };

    let horizontally_resized;
    let (vertical_input, vertical_input_width, vertical_input_height) =
        if let Some(coefficients) = horizontal.as_ref() {
            horizontally_resized = pillow_resample_horizontal_rgb8(
                input,
                input_height,
                input_width,
                output_width,
                coefficients,
            )?;
            (&horizontally_resized[..], output_width, input_height)
        } else {
            (input, input_width, input_height)
        };

    if let Some(coefficients) = vertical.as_ref() {
        pillow_resample_vertical_rgb8(
            vertical_input,
            vertical_input_height,
            vertical_input_width,
            output_height,
            coefficients,
        )
    } else {
        Ok(vertical_input.to_vec())
    }
}

fn pillow_bicubic_coefficients(
    input_size: usize,
    output_size: usize,
) -> Result<PillowCoefficients, CpuRefError> {
    let scale = input_size as f64 / output_size as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let kernel_size = (support.ceil() as usize)
        .checked_mul(2)
        .and_then(|size| size.checked_add(1))
        .ok_or_else(dimension_error)?;
    let weight_elements = output_size
        .checked_mul(kernel_size)
        .ok_or_else(dimension_error)?;
    let mut bounds = Vec::with_capacity(output_size);
    let mut floating_weights = vec![0.0_f64; weight_elements];
    let inverse_filter_scale = filter_scale.recip();

    for output_index in 0..output_size {
        let center = (output_index as f64 + 0.5) * scale;
        let minimum = ((center - support + 0.5) as isize).max(0) as usize;
        let maximum = ((center + support + 0.5) as usize).min(input_size);
        let count = maximum.saturating_sub(minimum);
        let row =
            &mut floating_weights[output_index * kernel_size..(output_index + 1) * kernel_size];
        let mut sum = 0.0_f64;
        for (offset, coefficient) in row.iter_mut().take(count).enumerate() {
            let distance = (offset as f64 + minimum as f64 - center + 0.5) * inverse_filter_scale;
            *coefficient = pillow_bicubic_filter(distance);
            sum += *coefficient;
        }
        if sum != 0.0 {
            for coefficient in row.iter_mut().take(count) {
                *coefficient /= sum;
            }
        }
        bounds.push((minimum, count));
    }

    const COEFFICIENT_SCALE: f64 = (1_u32 << 22) as f64;
    let weights = floating_weights
        .into_iter()
        .map(|coefficient| {
            if coefficient < 0.0 {
                (-0.5 + coefficient * COEFFICIENT_SCALE) as i32
            } else {
                (0.5 + coefficient * COEFFICIENT_SCALE) as i32
            }
        })
        .collect();
    Ok(PillowCoefficients {
        kernel_size,
        bounds,
        weights,
    })
}

fn pillow_bicubic_filter(mut value: f64) -> f64 {
    const A: f64 = -0.5;
    value = value.abs();
    if value < 1.0 {
        return ((A + 2.0) * value - (A + 3.0)) * value * value + 1.0;
    }
    if value < 2.0 {
        return (((value - 5.0) * value + 8.0) * value - 4.0) * A;
    }
    0.0
}

fn pillow_resample_horizontal_rgb8(
    input: &[u8],
    input_height: usize,
    input_width: usize,
    output_width: usize,
    coefficients: &PillowCoefficients,
) -> Result<Vec<u8>, CpuRefError> {
    let output_elements = input_height
        .checked_mul(output_width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(dimension_error)?;
    let mut output = vec![0_u8; output_elements];
    for y in 0..input_height {
        for output_x in 0..output_width {
            let (minimum, count) = coefficients.bounds[output_x];
            let weights = &coefficients.weights
                [output_x * coefficients.kernel_size..output_x * coefficients.kernel_size + count];
            for channel in 0..3 {
                let mut accumulator = 1_i64 << 21;
                for (offset, coefficient) in weights.iter().copied().enumerate() {
                    let byte = input[(y * input_width + minimum + offset) * 3 + channel];
                    accumulator += i64::from(byte) * i64::from(coefficient);
                }
                output[(y * output_width + output_x) * 3 + channel] = pillow_clip_u8(accumulator);
            }
        }
    }
    Ok(output)
}

fn pillow_resample_vertical_rgb8(
    input: &[u8],
    input_height: usize,
    width: usize,
    output_height: usize,
    coefficients: &PillowCoefficients,
) -> Result<Vec<u8>, CpuRefError> {
    let output_elements = output_height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(dimension_error)?;
    let mut output = vec![0_u8; output_elements];
    for output_y in 0..output_height {
        let (minimum, count) = coefficients.bounds[output_y];
        debug_assert!(minimum + count <= input_height);
        let weights = &coefficients.weights
            [output_y * coefficients.kernel_size..output_y * coefficients.kernel_size + count];
        for x in 0..width {
            for channel in 0..3 {
                let mut accumulator = 1_i64 << 21;
                for (offset, coefficient) in weights.iter().copied().enumerate() {
                    let byte = input[((minimum + offset) * width + x) * 3 + channel];
                    accumulator += i64::from(byte) * i64::from(coefficient);
                }
                output[(output_y * width + x) * 3 + channel] = pillow_clip_u8(accumulator);
            }
        }
    }
    Ok(output)
}

fn pillow_clip_u8(accumulator: i64) -> u8 {
    (accumulator >> 22).clamp(0, 255) as u8
}

fn validate_attention_inputs(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    cu_seqlens: &[usize],
) -> Result<usize, CpuRefError> {
    if tokens == 0 || heads == 0 || head_dim == 0 {
        return Err(dimension_error());
    }
    let tensor_elements = tokens
        .checked_mul(heads)
        .and_then(|elements| elements.checked_mul(head_dim))
        .ok_or_else(dimension_error)?;
    if query.len() != tensor_elements
        || key.len() != tensor_elements
        || value.len() != tensor_elements
    {
        return Err(dimension_error());
    }
    require_finite(query)?;
    require_finite(key)?;
    require_finite(value)?;
    validate_sequence_boundaries(tokens, cu_seqlens)?;
    Ok(tensor_elements)
}

#[derive(Clone, Copy, Debug)]
struct ProjectorLayout {
    input_tokens: usize,
    output_tokens: usize,
    merged_width: usize,
}

fn projector_layout(
    hidden_size: usize,
    image_grid_thw: &[[usize; 3]],
) -> Result<ProjectorLayout, CpuRefError> {
    if hidden_size == 0 {
        return Err(dimension_error());
    }
    if image_grid_thw.is_empty() {
        return Err(invalid_projector_geometry());
    }
    let merged_width = hidden_size.checked_mul(4).ok_or_else(dimension_error)?;
    let mut input_tokens = 0_usize;
    let mut output_tokens = 0_usize;
    for &[temporal, height, width] in image_grid_thw {
        if temporal == 0 || height == 0 || width == 0 || height % 2 != 0 || width % 2 != 0 {
            return Err(invalid_projector_geometry());
        }
        let tokens = temporal
            .checked_mul(height)
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(invalid_projector_geometry)?;
        let merged_tokens = temporal
            .checked_mul(height / 2)
            .and_then(|value| value.checked_mul(width / 2))
            .ok_or_else(invalid_projector_geometry)?;
        input_tokens = input_tokens
            .checked_add(tokens)
            .ok_or_else(invalid_projector_geometry)?;
        output_tokens = output_tokens
            .checked_add(merged_tokens)
            .ok_or_else(invalid_projector_geometry)?;
    }
    Ok(ProjectorLayout {
        input_tokens,
        output_tokens,
        merged_width,
    })
}

fn validate_vision_encoder_layer_config(
    config: VisionEncoderLayerConfig,
) -> Result<(), CpuRefError> {
    if config.tokens == 0
        || config.hidden_size == 0
        || config.intermediate_size == 0
        || config
            .attention_heads
            .checked_mul(config.head_dim)
            .is_none_or(|width| width != config.hidden_size)
    {
        return Err(dimension_error());
    }
    Ok(())
}

fn validate_checkpoint_selection(
    checkpoint_layers: &[usize],
    layers: usize,
) -> Result<(), CpuRefError> {
    if checkpoint_layers
        .iter()
        .any(|layer_index| *layer_index >= layers)
        || checkpoint_layers.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(CpuRefError::new(
            CpuRefErrorCode::InvalidCheckpointSelection,
            "checkpoint layers must be unique, strictly increasing, and within the stack",
        ));
    }
    Ok(())
}

fn require_nonzero_attention_tile(key_tile: usize) -> Result<(), CpuRefError> {
    if key_tile == 0 {
        return Err(CpuRefError::new(
            CpuRefErrorCode::InvalidTileSize,
            "attention tile size must be nonzero",
        ));
    }
    Ok(())
}

fn validate_sequence_boundaries(tokens: usize, cu_seqlens: &[usize]) -> Result<(), CpuRefError> {
    if cu_seqlens.len() < 2
        || cu_seqlens.first() != Some(&0)
        || cu_seqlens.last() != Some(&tokens)
        || cu_seqlens.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(CpuRefError::new(
            CpuRefErrorCode::InvalidSequenceBoundaries,
            "cu_seqlens must start at zero, end at tokens, and increase strictly",
        ));
    }
    Ok(())
}

fn attention_dot(
    query: &[f32],
    key: &[f32],
    query_token: usize,
    key_token: usize,
    head: usize,
    heads: usize,
    head_dim: usize,
) -> f32 {
    let mut score = 0.0_f32;
    for dimension in 0..head_dim {
        score += query[attention_index(query_token, head, dimension, heads, head_dim)]
            * key[attention_index(key_token, head, dimension, heads, head_dim)];
    }
    score
}

const fn attention_index(
    token: usize,
    head: usize,
    dimension: usize,
    heads: usize,
    head_dim: usize,
) -> usize {
    (token * heads + head) * head_dim + dimension
}

fn require_len(values: &[f32], rows: usize, columns: usize) -> Result<(), CpuRefError> {
    let expected = rows.checked_mul(columns).ok_or_else(dimension_error)?;
    if values.len() != expected {
        return Err(dimension_error());
    }
    Ok(())
}

fn require_nonzero_width(width: usize) -> Result<(), CpuRefError> {
    if width == 0 {
        return Err(dimension_error());
    }
    Ok(())
}

fn require_positive_finite_epsilon(epsilon: f32) -> Result<(), CpuRefError> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(CpuRefError::new(
            CpuRefErrorCode::NonPositiveEpsilon,
            "epsilon must be positive and finite",
        ));
    }
    Ok(())
}

fn require_finite(values: &[f32]) -> Result<(), CpuRefError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CpuRefError::new(
            CpuRefErrorCode::NonFiniteInput,
            "input contains NaN or infinity",
        ));
    }
    Ok(())
}

const fn dimension_error() -> CpuRefError {
    CpuRefError::new(
        CpuRefErrorCode::DimensionMismatch,
        "tensor dimensions do not match the supplied shape",
    )
}

const fn invalid_image_geometry() -> CpuRefError {
    CpuRefError::new(
        CpuRefErrorCode::InvalidImageGeometry,
        "image geometry must be nonzero, finite, and within the model aspect-ratio contract",
    )
}

const fn invalid_preprocess_config() -> CpuRefError {
    CpuRefError::new(
        CpuRefErrorCode::InvalidPreprocessConfig,
        "preprocessing pixel bounds and patch geometry are inconsistent",
    )
}

const fn invalid_projector_geometry() -> CpuRefError {
    CpuRefError::new(
        CpuRefErrorCode::InvalidProjectorGeometry,
        "projector grids must be nonempty, nonzero, overflow-free, and divisible by the 2x2 merge",
    )
}
