use pvlc_cpu_ref::{CpuRefError, materialized_segmented_attention_f32};
use pvlc_runtime_core::{InvocationError, KernelInvocation};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ComparisonPolicy;

pub const M3_VISION_ATTENTION_SEQUENCE_LENGTHS: [u32; 6] = [8, 16, 31, 64, 127, 256];
pub const PADDLEOCR_VL_VISION_HEADS: u32 = 16;
pub const PADDLEOCR_VL_VISION_HEAD_DIM: u32 = 72;
pub const VISION_ATTENTION_FIXTURE_ALGORITHM: &str = "affine-mod257-binary-f32-v1";

const POLICY: M3VisionAttentionPolicy = M3VisionAttentionPolicy {
    max_abs: 1.0e-3,
    max_mean_abs: 2.0e-4,
    max_p99_abs: 6.0e-4,
    max_relative_l2: 3.0e-4,
    min_cosine_similarity: 0.999_99,
    native_max_abs: 3.0e-4,
    native_max_relative_l2: 1.0e-4,
};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct M3VisionAttentionPolicy {
    pub max_abs: f64,
    pub max_mean_abs: f64,
    pub max_p99_abs: f64,
    pub max_relative_l2: f64,
    pub min_cosine_similarity: f64,
    pub native_max_abs: f64,
    pub native_max_relative_l2: f64,
}

impl M3VisionAttentionPolicy {
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
pub struct M3VisionAttentionCase {
    pub id: String,
    pub tags: Vec<String>,
    pub tokens: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub cu_seqlens: Vec<u32>,
    pub seed: u32,
    pub poisoned_segment: Option<u32>,
    pub expected: Vec<f32>,
    pub shape: Vec<usize>,
    pub policy: M3VisionAttentionPolicy,
}

impl M3VisionAttentionCase {
    pub fn invocation(&self) -> Result<KernelInvocation, M3VisionAttentionCorpusError> {
        let elements = usize::try_from(
            u64::from(self.tokens)
                .checked_mul(u64::from(self.heads))
                .and_then(|value| value.checked_mul(u64::from(self.head_dim)))
                .ok_or(M3VisionAttentionCorpusError::ArithmeticOverflow)?,
        )
        .map_err(|_| M3VisionAttentionCorpusError::ArithmeticOverflow)?;
        let mut query = fixture_values(elements, self.seed, 17, 19, 3);
        let mut key = fixture_values(elements, self.seed, 29, 23, 11);
        let mut value = fixture_values(elements, self.seed, 13, 31, 5);

        if let Some(segment) = self.poisoned_segment {
            let segment = usize::try_from(segment)
                .map_err(|_| M3VisionAttentionCorpusError::InvalidPoisonedSegment)?;
            if segment + 1 >= self.cu_seqlens.len() {
                return Err(M3VisionAttentionCorpusError::InvalidPoisonedSegment);
            }
            let row_width = usize::try_from(
                u64::from(self.heads)
                    .checked_mul(u64::from(self.head_dim))
                    .ok_or(M3VisionAttentionCorpusError::ArithmeticOverflow)?,
            )
            .map_err(|_| M3VisionAttentionCorpusError::ArithmeticOverflow)?;
            let start = self.cu_seqlens[segment] as usize * row_width;
            let end = self.cu_seqlens[segment + 1] as usize * row_width;
            if end > elements {
                return Err(M3VisionAttentionCorpusError::InvalidPoisonedSegment);
            }
            for index in start..end {
                key[index] = key[index] * -31.0 + 7.0;
                value[index] = value[index] * 47.0 - 11.0;
            }
        }

        let invocation = KernelInvocation::VisionAttentionF32 {
            tokens: self.tokens,
            heads: self.heads,
            head_dim: self.head_dim,
            query: std::mem::take(&mut query),
            key,
            value,
            cu_seqlens: self.cu_seqlens.clone(),
        };
        invocation.plan()?;
        Ok(invocation)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct M3VisionAttentionCorpus {
    pub schema_version: u32,
    pub oracle: String,
    pub fixture_algorithm: String,
    pub cases: Vec<M3VisionAttentionCase>,
}

#[derive(Debug, Error)]
pub enum M3VisionAttentionCorpusError {
    #[error("M3 vision attention invocation is invalid: {0}")]
    Invocation(#[from] InvocationError),
    #[error("M3 materialized attention oracle failed: {0}")]
    Cpu(#[from] CpuRefError),
    #[error("M3 vision attention fixture arithmetic overflowed")]
    ArithmeticOverflow,
    #[error("M3 vision attention fixture names an invalid poisoned segment")]
    InvalidPoisonedSegment,
}

pub fn m3_vision_attention_corpus() -> Result<M3VisionAttentionCorpus, M3VisionAttentionCorpusError>
{
    let mut cases = Vec::with_capacity(11);
    for (index, tokens) in M3_VISION_ATTENTION_SEQUENCE_LENGTHS.into_iter().enumerate() {
        cases.push(build_case(
            format!("vision_attention_f32/single-s{tokens:04}"),
            vec!["required-sequence".to_owned(), "single-segment".to_owned()],
            tokens,
            vec![0, tokens],
            101 + index as u32,
            None,
        )?);
    }
    cases.push(build_case(
        "vision_attention_f32/packed-0003-0011-0031".to_owned(),
        vec!["packed-segments".to_owned()],
        31,
        vec![0, 3, 11, 31],
        201,
        None,
    )?);
    cases.push(build_case(
        "vision_attention_f32/isolation-baseline".to_owned(),
        vec!["baseline".to_owned(), "isolation".to_owned()],
        17,
        vec![0, 3, 9, 17],
        301,
        None,
    )?);
    for segment in 0..3 {
        cases.push(build_case(
            format!("vision_attention_f32/isolation-poison-{segment}"),
            vec!["isolation".to_owned(), format!("poison-segment:{segment}")],
            17,
            vec![0, 3, 9, 17],
            301,
            Some(segment),
        )?);
    }
    Ok(M3VisionAttentionCorpus {
        schema_version: 1,
        oracle: "pvlc-cpu-ref/materialized-segmented-attention-f32-v1".to_owned(),
        fixture_algorithm: VISION_ATTENTION_FIXTURE_ALGORITHM.to_owned(),
        cases,
    })
}

fn build_case(
    id: String,
    mut tags: Vec<String>,
    tokens: u32,
    cu_seqlens: Vec<u32>,
    seed: u32,
    poisoned_segment: Option<u32>,
) -> Result<M3VisionAttentionCase, M3VisionAttentionCorpusError> {
    tags.sort();
    tags.dedup();
    let mut case = M3VisionAttentionCase {
        id,
        tags,
        tokens,
        heads: PADDLEOCR_VL_VISION_HEADS,
        head_dim: PADDLEOCR_VL_VISION_HEAD_DIM,
        cu_seqlens,
        seed,
        poisoned_segment,
        expected: Vec::new(),
        shape: vec![
            tokens as usize,
            PADDLEOCR_VL_VISION_HEADS as usize,
            PADDLEOCR_VL_VISION_HEAD_DIM as usize,
        ],
        policy: POLICY,
    };
    let invocation = case.invocation()?;
    let KernelInvocation::VisionAttentionF32 {
        query,
        key,
        value,
        cu_seqlens,
        ..
    } = invocation
    else {
        unreachable!()
    };
    let boundaries = cu_seqlens
        .iter()
        .map(|boundary| *boundary as usize)
        .collect::<Vec<_>>();
    case.expected = materialized_segmented_attention_f32(
        &query,
        &key,
        &value,
        tokens as usize,
        PADDLEOCR_VL_VISION_HEADS as usize,
        PADDLEOCR_VL_VISION_HEAD_DIM as usize,
        &boundaries,
    )?;
    Ok(case)
}

fn fixture_values(
    elements: usize,
    seed: u32,
    index_multiplier: u64,
    seed_multiplier: u64,
    offset: u64,
) -> Vec<f32> {
    let seed_term = u64::from(seed) * seed_multiplier + offset;
    (0..elements)
        .map(|index| {
            let residue = ((index as u64 * index_multiplier + seed_term) % 257) as i32;
            (residue - 128) as f32 / 64.0
        })
        .collect()
}
