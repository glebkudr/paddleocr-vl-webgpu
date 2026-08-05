//! Numerical comparison reports shared by native and browser correctness tests.

use std::{error::Error, fmt};

mod m2_corpus;
mod m3_vision_attention;
mod m3_vision_layer;

pub use m2_corpus::{
    M2_BOUNDARIES, M2_INPUT_FAMILIES, M2CasePolicy, M2CorpusError, M2PrimitiveCase,
    M2PrimitiveCorpus, m2_primitive_corpus,
};
pub use m3_vision_attention::{
    M3_VISION_ATTENTION_SEQUENCE_LENGTHS, M3VisionAttentionCase, M3VisionAttentionCorpus,
    M3VisionAttentionCorpusError, M3VisionAttentionPolicy, PADDLEOCR_VL_VISION_HEAD_DIM,
    PADDLEOCR_VL_VISION_HEADS, VISION_ATTENTION_FIXTURE_ALGORITHM, m3_vision_attention_corpus,
};
pub use m3_vision_layer::{
    M3_VISION_LAYER_ATTENTION_HEADS, M3_VISION_LAYER_CU_SEQLENS, M3_VISION_LAYER_HEAD_DIM,
    M3_VISION_LAYER_HIDDEN_SIZE, M3_VISION_LAYER_INTERMEDIATE_SIZE, M3_VISION_LAYER_TOKENS,
    M3VisionLayerCase, M3VisionLayerCheckpoints, M3VisionLayerCorpus, M3VisionLayerCorpusError,
    M3VisionLayerPolicy, VISION_LAYER_FIXTURE_ALGORITHM, m3_vision_layer_corpus,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ComparisonAxes {
    pub token_axis: Option<usize>,
    pub channel_axis: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComparisonPolicy {
    pub require_finite: bool,
    pub max_abs: f64,
    pub max_mean_abs: f64,
    pub max_p99_abs: f64,
    pub max_relative_l2: f64,
    pub min_cosine_similarity: f64,
    pub max_per_token_relative_l2: Option<f64>,
    pub max_per_channel_relative_l2: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NonFiniteCounts {
    pub nan: usize,
    pub positive_infinity: usize,
    pub negative_infinity: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonReport {
    pub element_count: usize,
    pub finite_pair_count: usize,
    pub max_abs: f64,
    pub mean_abs: f64,
    pub p50_abs: f64,
    pub p90_abs: f64,
    pub p99_abs: f64,
    pub relative_l2: f64,
    pub cosine_similarity: f64,
    pub per_token_relative_l2: Option<Vec<f64>>,
    pub per_channel_relative_l2: Option<Vec<f64>>,
    pub reference_non_finite: NonFiniteCounts,
    pub candidate_non_finite: NonFiniteCounts,
    pub non_finite_mismatches: usize,
}

pub fn compare_f32(
    reference: &[f32],
    candidate: &[f32],
    shape: &[usize],
    axes: ComparisonAxes,
) -> Result<ComparisonReport, ComparisonError> {
    if reference.is_empty() && candidate.is_empty() {
        return Err(ComparisonError::new(
            ComparisonErrorCode::EmptyTensor,
            "cannot compare an empty tensor",
        ));
    }
    if reference.len() != candidate.len() {
        return Err(ComparisonError::new(
            ComparisonErrorCode::LengthMismatch,
            "reference and candidate lengths differ",
        ));
    }
    let shape_elements = shape.iter().try_fold(1_usize, |elements, dimension| {
        elements.checked_mul(*dimension).ok_or_else(|| {
            ComparisonError::new(
                ComparisonErrorCode::ShapeOverflow,
                "tensor shape product overflowed",
            )
        })
    })?;
    if shape_elements != reference.len() {
        return Err(ComparisonError::new(
            ComparisonErrorCode::ShapeMismatch,
            "tensor shape does not match the flat buffer length",
        ));
    }
    validate_axes(shape, axes)?;

    let reference_non_finite = count_non_finite(reference);
    let candidate_non_finite = count_non_finite(candidate);
    let mut non_finite_mismatches = 0;
    let mut absolute_errors = Vec::with_capacity(reference.len());
    let mut error_l2 = 0.0_f64;
    let mut reference_l2 = 0.0_f64;
    let mut candidate_l2 = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut all_finite_equal = true;
    for (&expected, &actual) in reference.iter().zip(candidate) {
        match (non_finite_kind(expected), non_finite_kind(actual)) {
            (None, None) => {
                let expected = f64::from(expected);
                let actual = f64::from(actual);
                let error = actual - expected;
                let absolute = error.abs();
                absolute_errors.push(absolute);
                error_l2 += error * error;
                reference_l2 += expected * expected;
                candidate_l2 += actual * actual;
                dot += expected * actual;
                all_finite_equal &= expected == actual;
            }
            (left, right) => non_finite_mismatches += usize::from(left != right),
        }
    }
    absolute_errors.sort_by(f64::total_cmp);
    let finite_pair_count = absolute_errors.len();
    let (max_abs, mean_abs, p50_abs, p90_abs, p99_abs) = if finite_pair_count == 0 {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    } else {
        (
            *absolute_errors.last().expect("nonempty"),
            absolute_errors.iter().sum::<f64>() / finite_pair_count as f64,
            nearest_rank(&absolute_errors, 0.50),
            nearest_rank(&absolute_errors, 0.90),
            nearest_rank(&absolute_errors, 0.99),
        )
    };
    let relative_l2 = relative_l2(error_l2, reference_l2);
    let cosine_similarity = if all_finite_equal || (reference_l2 == 0.0 && candidate_l2 == 0.0) {
        1.0
    } else if reference_l2 == 0.0 || candidate_l2 == 0.0 {
        0.0
    } else {
        (dot / (reference_l2.sqrt() * candidate_l2.sqrt())).clamp(-1.0, 1.0)
    };

    Ok(ComparisonReport {
        element_count: reference.len(),
        finite_pair_count,
        max_abs,
        mean_abs,
        p50_abs,
        p90_abs,
        p99_abs,
        relative_l2,
        cosine_similarity,
        per_token_relative_l2: axes
            .token_axis
            .map(|axis| relative_l2_by_axis(reference, candidate, shape, axis)),
        per_channel_relative_l2: axes
            .channel_axis
            .map(|axis| relative_l2_by_axis(reference, candidate, shape, axis)),
        reference_non_finite,
        candidate_non_finite,
        non_finite_mismatches,
    })
}

impl ComparisonReport {
    pub fn assess(&self, policy: &ComparisonPolicy) -> Result<ComparisonVerdict, ComparisonError> {
        validate_policy(policy)?;
        let mut violations = Vec::new();
        let has_non_finite = self.reference_non_finite != NonFiniteCounts::default()
            || self.candidate_non_finite != NonFiniteCounts::default();
        if policy.require_finite && has_non_finite {
            violations.push(MetricViolation::NonFinite);
        } else if self.non_finite_mismatches != 0 {
            violations.push(MetricViolation::NonFiniteMismatch);
        }
        if self.max_abs > policy.max_abs {
            violations.push(MetricViolation::MaxAbs);
        }
        if self.mean_abs > policy.max_mean_abs {
            violations.push(MetricViolation::MeanAbs);
        }
        if self.p99_abs > policy.max_p99_abs {
            violations.push(MetricViolation::P99Abs);
        }
        if self.relative_l2 > policy.max_relative_l2 {
            violations.push(MetricViolation::RelativeL2);
        }
        if self.cosine_similarity < policy.min_cosine_similarity {
            violations.push(MetricViolation::CosineSimilarity);
        }
        if let (Some(limit), Some(values)) = (
            policy.max_per_token_relative_l2,
            &self.per_token_relative_l2,
        ) && values.iter().any(|value| *value > limit)
        {
            violations.push(MetricViolation::PerTokenRelativeL2);
        }
        if let (Some(limit), Some(values)) = (
            policy.max_per_channel_relative_l2,
            &self.per_channel_relative_l2,
        ) && values.iter().any(|value| *value > limit)
        {
            violations.push(MetricViolation::PerChannelRelativeL2);
        }
        Ok(ComparisonVerdict { violations })
    }
}

fn validate_axes(shape: &[usize], axes: ComparisonAxes) -> Result<(), ComparisonError> {
    for axis in [axes.token_axis, axes.channel_axis].into_iter().flatten() {
        if axis >= shape.len() {
            return Err(ComparisonError::new(
                ComparisonErrorCode::AxisOutOfRange,
                "comparison axis is outside the tensor rank",
            ));
        }
    }
    if axes.token_axis.is_some() && axes.token_axis == axes.channel_axis {
        return Err(ComparisonError::new(
            ComparisonErrorCode::DuplicateAxis,
            "token and channel axes must differ",
        ));
    }
    Ok(())
}

fn validate_policy(policy: &ComparisonPolicy) -> Result<(), ComparisonError> {
    let upper_limits = [
        policy.max_abs,
        policy.max_mean_abs,
        policy.max_p99_abs,
        policy.max_relative_l2,
    ];
    if upper_limits
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || policy
            .max_per_token_relative_l2
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        || policy
            .max_per_channel_relative_l2
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        || !policy.min_cosine_similarity.is_finite()
        || !(-1.0..=1.0).contains(&policy.min_cosine_similarity)
    {
        return Err(ComparisonError::new(
            ComparisonErrorCode::InvalidPolicy,
            "comparison policy contains an invalid threshold",
        ));
    }
    Ok(())
}

fn nearest_rank(sorted: &[f64], quantile: f64) -> f64 {
    let rank = (quantile * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1)]
}

fn relative_l2(error_squared: f64, reference_squared: f64) -> f64 {
    if reference_squared == 0.0 {
        if error_squared == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (error_squared / reference_squared).sqrt()
    }
}

fn relative_l2_by_axis(
    reference: &[f32],
    candidate: &[f32],
    shape: &[usize],
    axis: usize,
) -> Vec<f64> {
    let mut error_squared = vec![0.0_f64; shape[axis]];
    let mut reference_squared = vec![0.0_f64; shape[axis]];
    let stride = shape[axis + 1..].iter().product::<usize>();
    for (flat_index, (&expected, &actual)) in reference.iter().zip(candidate).enumerate() {
        if !expected.is_finite() || !actual.is_finite() {
            continue;
        }
        let coordinate = (flat_index / stride) % shape[axis];
        let expected = f64::from(expected);
        let error = f64::from(actual) - expected;
        error_squared[coordinate] += error * error;
        reference_squared[coordinate] += expected * expected;
    }
    error_squared
        .into_iter()
        .zip(reference_squared)
        .map(|(error, reference)| relative_l2(error, reference))
        .collect()
}

fn count_non_finite(values: &[f32]) -> NonFiniteCounts {
    let mut counts = NonFiniteCounts::default();
    for value in values {
        match non_finite_kind(*value) {
            Some(NonFiniteKind::Nan) => counts.nan += 1,
            Some(NonFiniteKind::PositiveInfinity) => counts.positive_infinity += 1,
            Some(NonFiniteKind::NegativeInfinity) => counts.negative_infinity += 1,
            None => {}
        }
    }
    counts
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NonFiniteKind {
    Nan,
    PositiveInfinity,
    NegativeInfinity,
}

fn non_finite_kind(value: f32) -> Option<NonFiniteKind> {
    if value.is_nan() {
        Some(NonFiniteKind::Nan)
    } else if value == f32::INFINITY {
        Some(NonFiniteKind::PositiveInfinity)
    } else if value == f32::NEG_INFINITY {
        Some(NonFiniteKind::NegativeInfinity)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetricViolation {
    NonFinite,
    NonFiniteMismatch,
    MaxAbs,
    MeanAbs,
    P99Abs,
    RelativeL2,
    CosineSimilarity,
    PerTokenRelativeL2,
    PerChannelRelativeL2,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComparisonVerdict {
    violations: Vec<MetricViolation>,
}

impl ComparisonVerdict {
    #[must_use]
    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }

    #[must_use]
    pub fn violations(&self) -> &[MetricViolation] {
        &self.violations
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TopCandidateError {
    pub index: usize,
    pub reference_logit: f32,
    pub candidate_logit: f32,
    pub absolute_error: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LogitComparison {
    pub reference_top1: usize,
    pub candidate_top1: usize,
    pub top1_agreement: bool,
    pub reference_top_k: Vec<usize>,
    pub candidate_top_k: Vec<usize>,
    pub top_k_overlap: usize,
    pub top_k_overlap_fraction: f64,
    pub reference_margin: f64,
    pub selected_indices: Vec<usize>,
    pub top_candidate_errors: Vec<TopCandidateError>,
    pub kl_reference_to_candidate: f64,
    pub jensen_shannon_divergence: f64,
    pub max_abs_error: f64,
}

pub fn compare_logits(
    reference: &[f32],
    candidate: &[f32],
    top_k: usize,
) -> Result<LogitComparison, ComparisonError> {
    if reference.len() != candidate.len() {
        return Err(ComparisonError::new(
            ComparisonErrorCode::LengthMismatch,
            "reference and candidate logits lengths differ",
        ));
    }
    if reference.is_empty() {
        return Err(ComparisonError::new(
            ComparisonErrorCode::EmptyTensor,
            "cannot compare empty logits",
        ));
    }
    if top_k == 0 || top_k > reference.len() {
        return Err(ComparisonError::new(
            ComparisonErrorCode::InvalidTopK,
            "top-k must be between one and the logits length",
        ));
    }
    if reference
        .iter()
        .chain(candidate)
        .any(|value| !value.is_finite())
    {
        return Err(ComparisonError::new(
            ComparisonErrorCode::NonFiniteInput,
            "logits contain NaN or infinity",
        ));
    }

    let reference_top_k = top_indices(reference, top_k);
    let candidate_top_k = top_indices(candidate, top_k);
    let reference_top1 = reference_top_k[0];
    let candidate_top1 = candidate_top_k[0];
    let top_k_overlap = reference_top_k
        .iter()
        .filter(|index| candidate_top_k.contains(index))
        .count();
    let reference_margin = if reference.len() == 1 {
        f64::INFINITY
    } else {
        f64::from(reference[reference_top_k[0]] - reference[top_indices(reference, 2)[1]])
    };
    let mut selected_indices = reference_top_k.clone();
    selected_indices.extend_from_slice(&candidate_top_k);
    selected_indices.sort_unstable();
    selected_indices.dedup();
    let top_candidate_errors = selected_indices
        .iter()
        .map(|&index| TopCandidateError {
            index,
            reference_logit: reference[index],
            candidate_logit: candidate[index],
            absolute_error: (f64::from(reference[index]) - f64::from(candidate[index])).abs(),
        })
        .collect();
    let reference_distribution = selected_softmax(reference, &selected_indices);
    let candidate_distribution = selected_softmax(candidate, &selected_indices);
    let midpoint: Vec<_> = reference_distribution
        .iter()
        .zip(&candidate_distribution)
        .map(|(left, right)| (left + right) * 0.5)
        .collect();
    let kl_reference_to_candidate = kl_divergence(&reference_distribution, &candidate_distribution);
    let jensen_shannon_divergence = 0.5 * kl_divergence(&reference_distribution, &midpoint)
        + 0.5 * kl_divergence(&candidate_distribution, &midpoint);
    let max_abs_error = reference
        .iter()
        .zip(candidate)
        .map(|(left, right)| (f64::from(*left) - f64::from(*right)).abs())
        .fold(0.0, f64::max);

    Ok(LogitComparison {
        reference_top1,
        candidate_top1,
        top1_agreement: reference_top1 == candidate_top1,
        reference_top_k,
        candidate_top_k,
        top_k_overlap,
        top_k_overlap_fraction: top_k_overlap as f64 / top_k as f64,
        reference_margin,
        selected_indices,
        top_candidate_errors,
        kl_reference_to_candidate,
        jensen_shannon_divergence,
        max_abs_error,
    })
}

impl LogitComparison {
    pub fn stable_token_verdict(
        &self,
        error_envelope: f64,
    ) -> Result<StableTokenVerdict, ComparisonError> {
        if !error_envelope.is_finite() || error_envelope < 0.0 {
            return Err(ComparisonError::new(
                ComparisonErrorCode::InvalidPolicy,
                "stable-token error envelope must be nonnegative and finite",
            ));
        }
        if self.reference_margin > 2.0 * error_envelope {
            if self.top1_agreement {
                Ok(StableTokenVerdict::RequiredAndMatched)
            } else {
                Ok(StableTokenVerdict::RequiredButChanged)
            }
        } else {
            Ok(StableTokenVerdict::Ambiguous)
        }
    }
}

fn top_indices(values: &[f32], count: usize) -> Vec<usize> {
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    indices.truncate(count);
    indices
}

fn selected_softmax(values: &[f32], indices: &[usize]) -> Vec<f64> {
    let maximum = indices
        .iter()
        .map(|&index| f64::from(values[index]))
        .fold(f64::NEG_INFINITY, f64::max);
    let mut distribution: Vec<_> = indices
        .iter()
        .map(|&index| (f64::from(values[index]) - maximum).exp())
        .collect();
    let denominator: f64 = distribution.iter().sum();
    for value in &mut distribution {
        *value /= denominator;
    }
    distribution
}

fn kl_divergence(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * (left / right).ln())
        .sum()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StableTokenVerdict {
    RequiredAndMatched,
    RequiredButChanged,
    Ambiguous,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComparisonErrorCode {
    EmptyTensor,
    LengthMismatch,
    ShapeMismatch,
    ShapeOverflow,
    AxisOutOfRange,
    DuplicateAxis,
    InvalidPolicy,
    InvalidTopK,
    NonFiniteInput,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComparisonError {
    code: ComparisonErrorCode,
    message: String,
}

impl ComparisonError {
    fn new(code: ComparisonErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    #[must_use]
    pub const fn code(&self) -> ComparisonErrorCode {
        self.code
    }
}

impl fmt::Display for ComparisonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "tensor comparison error {:?}: {}",
            self.code, self.message
        )
    }
}

impl Error for ComparisonError {}
