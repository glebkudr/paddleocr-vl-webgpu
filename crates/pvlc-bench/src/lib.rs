#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

mod assembly;
mod browser_cohort;

pub use assembly::{
    AssembledBenchmarkEvidenceV1, BenchmarkCohortV1, BenchmarkEvidenceAssemblyInputV1,
    BenchmarkSampleAttemptV1, LoadOrCompileObservationV1,
    canonical_benchmark_evidence_assembly_bytes_v1, validate_load_or_compile_observation_v1,
};
pub use browser_cohort::{
    assemble_browser_benchmark_cohort_v1, validate_browser_benchmark_cohort_plan_v1,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum ObservationV1 {
    Available { value: String, method: String },
    Unavailable { reason: String, method: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKindV1 {
    NativeWgpu,
    ChromeWebgpu,
    WebkitWebgpu,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendIdentityV1 {
    pub kind: BackendKindV1,
    pub browser_version: Option<String>,
    pub user_agent: Option<String>,
    pub adapter_backend: String,
    pub features: Vec<String>,
    pub limits: BTreeMap<String, u64>,
    pub timestamp_query: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentityV1 {
    pub revision: String,
    pub model_lock_blake3: String,
    pub pack_blake3: String,
    pub manifest_sha256: String,
    pub profile: String,
    pub case_id: String,
    pub input_blake3: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkPassportV1 {
    pub machine: String,
    pub soc: String,
    pub adapter_name: String,
    pub physical_memory_bytes: u64,
    pub os_version: String,
    pub os_build: String,
    pub power_source: ObservationV1,
    pub power_profile: ObservationV1,
    pub low_power_mode: ObservationV1,
    pub thermal_state: ObservationV1,
    pub display_attached: ObservationV1,
    pub source_tree_blake3: String,
    pub compiler_runtime_blake3: String,
    pub wgsl_runtime_blake3: String,
    pub collector_blake3: String,
    pub rustc_version: String,
    pub cargo_version: String,
    pub wgpu_version: String,
    pub build_profile: String,
    pub backend: BackendIdentityV1,
    pub model: ModelIdentityV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionBoundaryV1 {
    ApiWall,
    QueueWall,
    GpuTimestamp,
    LoadOrCompile,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedTopologyV1 {
    pub dispatch_count: u64,
    pub compute_pass_count: u64,
    pub command_buffer_count: u64,
    pub submission_count: u64,
    pub map_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelVariantIdentityV1 {
    pub id: String,
    pub source_set_blake3: String,
    pub abi_blake3: String,
    pub expected_topology: ExpectedTopologyV1,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidencyPlanIdentityV1 {
    pub id: String,
    pub activation_strategy: String,
    pub activation_buffer_count: u64,
    pub activation_arena_bytes: u64,
    pub scratch_arena_bytes: u64,
    pub main_buffers_bytes: u64,
    pub logical_gpu_bytes: u64,
    pub allocated_gpu_bytes: u64,
    pub max_resident_shard_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisionStackWorkloadV1 {
    pub tokens: u32,
    pub hidden_size: u32,
    pub layer_count: u32,
    pub checkpoint_policy: String,
    pub checkpoint_sha256: String,
    #[serde(deserialize_with = "deserialize_required_nullable_string")]
    pub semantic_graph_blake3: Option<String>,
    pub manifest_sha256: String,
    pub ordered_layer_plans_blake3: Vec<String>,
    pub qkv_policy: String,
    pub qkv_outcome: String,
    pub kernel_variant: KernelVariantIdentityV1,
    pub residency_plan: ResidencyPlanIdentityV1,
    pub readback_policy: String,
    pub execution_boundary: ExecutionBoundaryV1,
}

fn deserialize_required_nullable_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorrectnessAnchorV1 {
    pub validator_blake3: String,
    pub policy_id: String,
    pub expected_checkpoint_sha256: String,
    pub causal_validator_blake3: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkClassV1 {
    Micro,
    StageMacro,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkProtocolV1 {
    pub class: BenchmarkClassV1,
    pub build_profile: String,
    pub warmup_count: u32,
    pub measured_count: u32,
    pub synchronization: String,
    pub clock_source: String,
    pub clock_resolution_ns: u64,
    pub schedule: String,
    pub output_validation_policy: String,
    pub isolation_policy: String,
    pub interruption_policy: String,
    pub background_load_policy: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum DurationObservationV1 {
    Available { duration_ns: u64 },
    Unavailable { reason: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum GpuTimestampObservationV1 {
    Available {
        begin_ticks: u64,
        end_ticks: u64,
        period_ns: String,
        duration_ns: u64,
    },
    Unavailable {
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum SampleStatusV1 {
    Passed,
    Failed { code: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkSampleV1 {
    pub index: u32,
    pub schedule_slot: u32,
    pub kernel_variant_id: String,
    pub residency_plan_id: String,
    pub api_wall_ns: u64,
    pub queue_wall: DurationObservationV1,
    pub gpu_timestamp: GpuTimestampObservationV1,
    pub topology: ExpectedTopologyV1,
    pub output_sha256: String,
    pub correctness_report_blake3: String,
    pub causal_evidence_blake3: String,
    pub logical_gpu_bytes: u64,
    pub allocated_gpu_bytes: u64,
    pub thermal_before: ObservationV1,
    pub thermal_after: ObservationV1,
    pub status: SampleStatusV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkEvidenceInputV1 {
    pub passport: BenchmarkPassportV1,
    pub workload: VisionStackWorkloadV1,
    pub correctness_anchor: CorrectnessAnchorV1,
    pub protocol: BenchmarkProtocolV1,
    pub cold_sample: BenchmarkSampleV1,
    pub warmup_samples: Vec<BenchmarkSampleV1>,
    pub measured_samples: Vec<BenchmarkSampleV1>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimClassV1 {
    BaselineOnly,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExactRationalV1 {
    pub numerator: String,
    pub denominator: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkSummaryV1 {
    pub count: u32,
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: ExactRationalV1,
    pub median_ns: ExactRationalV1,
    pub p90_ns: u64,
    pub p95_ns: u64,
    pub median_absolute_deviation_ns: ExactRationalV1,
    pub raw_order_blake3: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkErrorCodeV1 {
    NotImplemented,
    SchemaMismatch,
    UnsupportedSchema,
    UnsupportedClaim,
    NonCanonical,
    SelfHashMismatch,
    SummaryMismatch,
    InvalidIdentity,
    InvalidEnvironment,
    InvalidProtocol,
    InvalidAttemptJournal,
    InvalidPreparation,
    InvalidInteger,
    InvalidDuration,
    InvalidIndex,
    InvalidSchedule,
    InvalidQueueObservation,
    InvalidTimestamp,
    TimestampOverflow,
    StaleTimestamp,
    CrossLinkMismatch,
    TopologyMismatch,
    ResourceMismatch,
    FailedSample,
}

impl BenchmarkErrorCodeV1 {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotImplemented => "not_implemented",
            Self::SchemaMismatch => "schema_mismatch",
            Self::UnsupportedSchema => "unsupported_schema",
            Self::UnsupportedClaim => "unsupported_claim",
            Self::NonCanonical => "non_canonical",
            Self::SelfHashMismatch => "self_hash_mismatch",
            Self::SummaryMismatch => "summary_mismatch",
            Self::InvalidIdentity => "invalid_identity",
            Self::InvalidEnvironment => "invalid_environment",
            Self::InvalidProtocol => "invalid_protocol",
            Self::InvalidAttemptJournal => "invalid_attempt_journal",
            Self::InvalidPreparation => "invalid_preparation",
            Self::InvalidInteger => "invalid_integer",
            Self::InvalidDuration => "invalid_duration",
            Self::InvalidIndex => "invalid_index",
            Self::InvalidSchedule => "invalid_schedule",
            Self::InvalidQueueObservation => "invalid_queue_observation",
            Self::InvalidTimestamp => "invalid_timestamp",
            Self::TimestampOverflow => "timestamp_overflow",
            Self::StaleTimestamp => "stale_timestamp",
            Self::CrossLinkMismatch => "cross_link_mismatch",
            Self::TopologyMismatch => "topology_mismatch",
            Self::ResourceMismatch => "resource_mismatch",
            Self::FailedSample => "failed_sample",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BenchmarkError {
    #[error("M7d1 benchmark evidence implementation is not present")]
    NotImplemented,
    #[error("benchmark evidence does not match the closed V1 schema")]
    SchemaMismatch,
    #[error("benchmark evidence schema version is unsupported")]
    UnsupportedSchema,
    #[error("benchmark evidence claim class is unsupported")]
    UnsupportedClaim,
    #[error("benchmark evidence is not canonical JSON with one terminal LF")]
    NonCanonical,
    #[error("benchmark evidence self-hash does not match its canonical preimage")]
    SelfHashMismatch,
    #[error("benchmark summary does not match the measured raw samples")]
    SummaryMismatch,
    #[error("benchmark identity is incomplete or invalid")]
    InvalidIdentity,
    #[error("benchmark environment observation is incomplete or invalid")]
    InvalidEnvironment,
    #[error("benchmark protocol is unsupported or internally inconsistent")]
    InvalidProtocol,
    #[error("benchmark attempt journal is incomplete, reordered, or internally inconsistent")]
    InvalidAttemptJournal,
    #[error("benchmark load-or-compile preparation is invalid or belongs to another leaf")]
    InvalidPreparation,
    #[error("benchmark evidence contains a non-integer integer field")]
    InvalidInteger,
    #[error("benchmark duration is zero, saturated, empty, or otherwise invalid")]
    InvalidDuration,
    #[error("benchmark sample index is not fresh and contiguous")]
    InvalidIndex,
    #[error("benchmark schedule or schedule slot is invalid")]
    InvalidSchedule,
    #[error("benchmark queue-wall observation is invalid")]
    InvalidQueueObservation,
    #[error("benchmark GPU timestamp observation is invalid")]
    InvalidTimestamp,
    #[error("benchmark GPU timestamp arithmetic overflowed V1 nanoseconds")]
    TimestampOverflow,
    #[error("benchmark GPU timestamp pair was reused")]
    StaleTimestamp,
    #[error("benchmark evidence identities are not cross-linked")]
    CrossLinkMismatch,
    #[error("benchmark sample topology does not match the workload")]
    TopologyMismatch,
    #[error("benchmark sample resources do not match the residency plan")]
    ResourceMismatch,
    #[error("failed benchmark samples are not admissible evidence")]
    FailedSample,
}

impl BenchmarkError {
    pub fn code(&self) -> BenchmarkErrorCodeV1 {
        match self {
            Self::NotImplemented => BenchmarkErrorCodeV1::NotImplemented,
            Self::SchemaMismatch => BenchmarkErrorCodeV1::SchemaMismatch,
            Self::UnsupportedSchema => BenchmarkErrorCodeV1::UnsupportedSchema,
            Self::UnsupportedClaim => BenchmarkErrorCodeV1::UnsupportedClaim,
            Self::NonCanonical => BenchmarkErrorCodeV1::NonCanonical,
            Self::SelfHashMismatch => BenchmarkErrorCodeV1::SelfHashMismatch,
            Self::SummaryMismatch => BenchmarkErrorCodeV1::SummaryMismatch,
            Self::InvalidIdentity => BenchmarkErrorCodeV1::InvalidIdentity,
            Self::InvalidEnvironment => BenchmarkErrorCodeV1::InvalidEnvironment,
            Self::InvalidProtocol => BenchmarkErrorCodeV1::InvalidProtocol,
            Self::InvalidAttemptJournal => BenchmarkErrorCodeV1::InvalidAttemptJournal,
            Self::InvalidPreparation => BenchmarkErrorCodeV1::InvalidPreparation,
            Self::InvalidInteger => BenchmarkErrorCodeV1::InvalidInteger,
            Self::InvalidDuration => BenchmarkErrorCodeV1::InvalidDuration,
            Self::InvalidIndex => BenchmarkErrorCodeV1::InvalidIndex,
            Self::InvalidSchedule => BenchmarkErrorCodeV1::InvalidSchedule,
            Self::InvalidQueueObservation => BenchmarkErrorCodeV1::InvalidQueueObservation,
            Self::InvalidTimestamp => BenchmarkErrorCodeV1::InvalidTimestamp,
            Self::TimestampOverflow => BenchmarkErrorCodeV1::TimestampOverflow,
            Self::StaleTimestamp => BenchmarkErrorCodeV1::StaleTimestamp,
            Self::CrossLinkMismatch => BenchmarkErrorCodeV1::CrossLinkMismatch,
            Self::TopologyMismatch => BenchmarkErrorCodeV1::TopologyMismatch,
            Self::ResourceMismatch => BenchmarkErrorCodeV1::ResourceMismatch,
            Self::FailedSample => BenchmarkErrorCodeV1::FailedSample,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkEvidenceDataV1 {
    schema_version: u32,
    claim_class: ClaimClassV1,
    passport: BenchmarkPassportV1,
    workload: VisionStackWorkloadV1,
    correctness_anchor: CorrectnessAnchorV1,
    protocol: BenchmarkProtocolV1,
    cold_sample: BenchmarkSampleV1,
    warmup_samples: Vec<BenchmarkSampleV1>,
    measured_samples: Vec<BenchmarkSampleV1>,
    summary: BenchmarkSummaryV1,
    evidence_blake3: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkEvidenceV1 {
    data: BenchmarkEvidenceDataV1,
}

impl Serialize for BenchmarkEvidenceV1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.data.serialize(serializer)
    }
}

pub fn summarize_api_wall_ns(durations_ns: &[u64]) -> Result<BenchmarkSummaryV1, BenchmarkError> {
    if durations_ns.is_empty()
        || durations_ns
            .iter()
            .any(|duration| *duration == 0 || *duration == u64::MAX)
    {
        return Err(BenchmarkError::InvalidDuration);
    }

    let count = u32::try_from(durations_ns.len()).map_err(|_| BenchmarkError::InvalidDuration)?;
    let sum = durations_ns
        .iter()
        .try_fold(0_u128, |accumulator, duration| {
            accumulator
                .checked_add(u128::from(*duration))
                .ok_or(BenchmarkError::InvalidDuration)
        })?;
    let mut sorted = durations_ns.to_vec();
    sorted.sort_unstable();

    let mean_ns = reduced_rational(sum, u128::from(count));
    let median_ns = median_of_sorted_u64(&sorted);
    let p90_ns = nearest_rank(&sorted, 90);
    let p95_ns = nearest_rank(&sorted, 95);
    let median_absolute_deviation_ns = median_absolute_deviation(&sorted, &median_ns);

    let mut raw_order_preimage = Vec::with_capacity(durations_ns.len().saturating_mul(8));
    for duration in durations_ns {
        raw_order_preimage.extend_from_slice(&duration.to_le_bytes());
    }

    Ok(BenchmarkSummaryV1 {
        count,
        min_ns: sorted[0],
        max_ns: sorted[sorted.len() - 1],
        mean_ns,
        median_ns,
        p90_ns,
        p95_ns,
        median_absolute_deviation_ns,
        raw_order_blake3: blake3::hash(&raw_order_preimage).to_hex().to_string(),
    })
}

impl BenchmarkEvidenceV1 {
    pub fn build(input: BenchmarkEvidenceInputV1) -> Result<Self, BenchmarkError> {
        validate_input(&input)?;
        let summary = summarize_api_wall_ns(
            &input
                .measured_samples
                .iter()
                .map(|sample| sample.api_wall_ns)
                .collect::<Vec<_>>(),
        )?;

        let BenchmarkEvidenceInputV1 {
            passport,
            workload,
            correctness_anchor,
            protocol,
            cold_sample,
            warmup_samples,
            measured_samples,
        } = input;
        let data = BenchmarkEvidenceDataV1 {
            schema_version: 1,
            claim_class: ClaimClassV1::BaselineOnly,
            passport,
            workload,
            correctness_anchor,
            protocol,
            cold_sample,
            warmup_samples,
            measured_samples,
            summary,
            evidence_blake3: String::new(),
        };
        let mut evidence = Self { data };
        evidence.data.evidence_blake3 = evidence.computed_evidence_blake3();
        Ok(evidence)
    }

    pub fn parse_canonical(bytes: &[u8]) -> Result<Self, BenchmarkError> {
        let unique = parse_unique_json(bytes)?;
        if canonical_value_bytes(&unique) != bytes {
            return Err(BenchmarkError::NonCanonical);
        }

        let object = unique.as_object().ok_or(BenchmarkError::SchemaMismatch)?;
        match object
            .get("schema_version")
            .and_then(serde_json::Value::as_u64)
        {
            Some(1) => {}
            Some(_) => return Err(BenchmarkError::UnsupportedSchema),
            None if object.contains_key("schema_version") => {
                return Err(BenchmarkError::InvalidInteger);
            }
            None => return Err(BenchmarkError::SchemaMismatch),
        }
        match object
            .get("claim_class")
            .and_then(serde_json::Value::as_str)
        {
            Some("baseline_only") => {}
            Some(_) => return Err(BenchmarkError::UnsupportedClaim),
            None => return Err(BenchmarkError::SchemaMismatch),
        }

        let supplied_hash = object
            .get("evidence_blake3")
            .and_then(serde_json::Value::as_str)
            .ok_or(BenchmarkError::SchemaMismatch)?;
        let mut unsigned = unique.clone();
        unsigned
            .as_object_mut()
            .expect("top-level object checked above")
            .remove("evidence_blake3");
        let expected_hash = blake3::hash(&canonical_value_bytes(&unsigned))
            .to_hex()
            .to_string();
        if supplied_hash != expected_hash {
            return Err(BenchmarkError::SelfHashMismatch);
        }

        let data: BenchmarkEvidenceDataV1 =
            serde_json::from_value(unique).map_err(classify_typed_json_error)?;
        let evidence = Self { data };
        if evidence.data.schema_version != 1 {
            return Err(BenchmarkError::UnsupportedSchema);
        }
        if evidence.data.claim_class != ClaimClassV1::BaselineOnly {
            return Err(BenchmarkError::UnsupportedClaim);
        }

        let input = evidence.input_clone();
        validate_input(&input)?;
        let expected_summary = summarize_api_wall_ns(
            &evidence
                .data
                .measured_samples
                .iter()
                .map(|sample| sample.api_wall_ns)
                .collect::<Vec<_>>(),
        )?;
        if evidence.data.summary != expected_summary {
            return Err(BenchmarkError::SummaryMismatch);
        }
        Ok(evidence)
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        canonical_serializable_bytes(self)
    }

    pub fn evidence_blake3(&self) -> &str {
        &self.data.evidence_blake3
    }

    pub fn claim_class(&self) -> ClaimClassV1 {
        self.data.claim_class
    }

    pub fn summary(&self) -> &BenchmarkSummaryV1 {
        &self.data.summary
    }

    fn computed_evidence_blake3(&self) -> String {
        let mut value = serde_json::to_value(self)
            .expect("the closed benchmark evidence schema is always serializable");
        value
            .as_object_mut()
            .expect("benchmark evidence serializes as an object")
            .remove("evidence_blake3");
        blake3::hash(&canonical_value_bytes(&value))
            .to_hex()
            .to_string()
    }

    fn input_clone(&self) -> BenchmarkEvidenceInputV1 {
        BenchmarkEvidenceInputV1 {
            passport: self.data.passport.clone(),
            workload: self.data.workload.clone(),
            correctness_anchor: self.data.correctness_anchor.clone(),
            protocol: self.data.protocol.clone(),
            cold_sample: self.data.cold_sample.clone(),
            warmup_samples: self.data.warmup_samples.clone(),
            measured_samples: self.data.measured_samples.clone(),
        }
    }
}

fn reduced_rational(numerator: u128, denominator: u128) -> ExactRationalV1 {
    let divisor = greatest_common_divisor(numerator, denominator);
    let denominator = denominator / divisor;
    ExactRationalV1 {
        numerator: (numerator / divisor).to_string(),
        denominator: u64::try_from(denominator)
            .expect("V1 statistic denominators are bounded by the u32 sample count"),
    }
}

fn greatest_common_divisor(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn median_of_sorted_u64(sorted: &[u64]) -> ExactRationalV1 {
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        reduced_rational(u128::from(sorted[middle]), 1)
    } else {
        reduced_rational(
            u128::from(sorted[middle - 1]) + u128::from(sorted[middle]),
            2,
        )
    }
}

fn median_absolute_deviation(sorted: &[u64], median: &ExactRationalV1) -> ExactRationalV1 {
    let median_numerator = median
        .numerator
        .parse::<u128>()
        .expect("statistics create canonical unsigned rational numerators");
    let median_denominator = u128::from(median.denominator);
    let mut deviations = sorted
        .iter()
        .map(|value| {
            u128::from(*value)
                .checked_mul(median_denominator)
                .expect("a V1 median denominator is at most two")
                .abs_diff(median_numerator)
        })
        .collect::<Vec<_>>();
    deviations.sort_unstable();
    let middle = deviations.len() / 2;
    if deviations.len() % 2 == 1 {
        reduced_rational(deviations[middle], median_denominator)
    } else {
        reduced_rational(
            deviations[middle - 1] + deviations[middle],
            median_denominator * 2,
        )
    }
}

fn nearest_rank(sorted: &[u64], percentile: u64) -> u64 {
    let sample_count =
        u64::try_from(sorted.len()).expect("V1 statistics reject sample counts above u32::MAX");
    sorted[nearest_rank_index(sample_count, percentile)]
}

fn nearest_rank_index(sample_count: u64, percentile: u64) -> usize {
    debug_assert!(sample_count > 0);
    debug_assert!((1..=100).contains(&percentile));
    let one_based_rank = percentile
        .checked_mul(sample_count)
        .expect("V1 percentile times a u32 sample count fits u64")
        .div_ceil(100);
    usize::try_from(one_based_rank - 1)
        .expect("a zero-based rank below a u32 sample count fits every supported target")
}

/// Validates the immutable, sample-independent identity of one benchmark leaf.
///
/// This is the reusable static-admission boundary for physical cohort runners. It deliberately
/// does not validate a supplied attempt journal or claim that any attempt occurred.
pub fn validate_benchmark_leaf_v1(
    passport: &BenchmarkPassportV1,
    workload: &VisionStackWorkloadV1,
    correctness_anchor: &CorrectnessAnchorV1,
    protocol: &BenchmarkProtocolV1,
) -> Result<(), BenchmarkError> {
    validate_leaf_identity_and_protocol(passport, workload, correctness_anchor, protocol)?;
    validate_cross_links(passport, workload, correctness_anchor)?;
    validate_passport_environment(passport)
}

fn validate_input(input: &BenchmarkEvidenceInputV1) -> Result<(), BenchmarkError> {
    validate_leaf_identity_and_protocol(
        &input.passport,
        &input.workload,
        &input.correctness_anchor,
        &input.protocol,
    )?;
    validate_protocol_sample_counts(input)?;
    validate_cross_links(&input.passport, &input.workload, &input.correctness_anchor)?;
    validate_passport_environment(&input.passport)?;

    let mut timestamp_pairs = BTreeSet::new();
    validate_sample(&input.cold_sample, 0, input, &mut timestamp_pairs)?;
    validate_cohort(&input.warmup_samples, input, &mut timestamp_pairs)?;
    validate_cohort(&input.measured_samples, input, &mut timestamp_pairs)?;
    Ok(())
}

fn validate_leaf_identity_and_protocol(
    passport: &BenchmarkPassportV1,
    workload: &VisionStackWorkloadV1,
    correctness_anchor: &CorrectnessAnchorV1,
    protocol: &BenchmarkProtocolV1,
) -> Result<(), BenchmarkError> {
    validate_passport_identity(passport)?;
    validate_workload_identity(workload)?;
    validate_correctness_anchor(correctness_anchor)?;
    validate_protocol_identity(passport, workload, protocol)
}

fn validate_passport_identity(passport: &BenchmarkPassportV1) -> Result<(), BenchmarkError> {
    for identity in [
        &passport.machine,
        &passport.soc,
        &passport.adapter_name,
        &passport.os_version,
        &passport.os_build,
        &passport.rustc_version,
        &passport.cargo_version,
        &passport.wgpu_version,
    ] {
        require_nonempty(identity, BenchmarkError::InvalidIdentity)?;
    }
    if passport.physical_memory_bytes == 0 {
        return Err(BenchmarkError::InvalidIdentity);
    }
    for hash in [
        &passport.source_tree_blake3,
        &passport.compiler_runtime_blake3,
        &passport.wgsl_runtime_blake3,
        &passport.collector_blake3,
        &passport.model.model_lock_blake3,
        &passport.model.pack_blake3,
        &passport.model.manifest_sha256,
        &passport.model.input_blake3,
    ] {
        require_hash(hash)?;
    }
    for identity in [
        &passport.model.revision,
        &passport.model.profile,
        &passport.model.case_id,
        &passport.backend.adapter_backend,
    ] {
        require_nonempty(identity, BenchmarkError::InvalidIdentity)?;
    }

    if passport
        .backend
        .features
        .iter()
        .any(|feature| feature.trim().is_empty())
        || passport
            .backend
            .features
            .windows(2)
            .any(|window| window[0] >= window[1])
        || passport
            .backend
            .limits
            .iter()
            .any(|(name, value)| name.trim().is_empty() || *value == 0)
    {
        return Err(BenchmarkError::InvalidIdentity);
    }
    let advertises_timestamp = passport
        .backend
        .features
        .iter()
        .any(|feature| feature == "timestamp_query");
    if advertises_timestamp != passport.backend.timestamp_query {
        return Err(BenchmarkError::InvalidIdentity);
    }

    match passport.backend.kind {
        BackendKindV1::NativeWgpu => {
            if passport.backend.browser_version.is_some() || passport.backend.user_agent.is_some() {
                return Err(BenchmarkError::InvalidIdentity);
            }
        }
        BackendKindV1::ChromeWebgpu | BackendKindV1::WebkitWebgpu => {
            require_optional_nonempty(
                passport.backend.browser_version.as_deref(),
                BenchmarkError::InvalidIdentity,
            )?;
            require_optional_nonempty(
                passport.backend.user_agent.as_deref(),
                BenchmarkError::InvalidIdentity,
            )?;
        }
    }
    Ok(())
}

fn validate_workload_identity(workload: &VisionStackWorkloadV1) -> Result<(), BenchmarkError> {
    if workload.tokens == 0 || workload.hidden_size == 0 || workload.layer_count == 0 {
        return Err(BenchmarkError::InvalidIdentity);
    }
    for identity in [
        &workload.checkpoint_policy,
        &workload.qkv_policy,
        &workload.qkv_outcome,
        &workload.kernel_variant.id,
        &workload.residency_plan.id,
        &workload.residency_plan.activation_strategy,
        &workload.readback_policy,
    ] {
        require_nonempty(identity, BenchmarkError::InvalidIdentity)?;
    }
    for hash in [
        &workload.checkpoint_sha256,
        &workload.manifest_sha256,
        &workload.kernel_variant.source_set_blake3,
        &workload.kernel_variant.abi_blake3,
    ] {
        require_hash(hash)?;
    }
    match (workload.qkv_policy.as_str(), workload.qkv_outcome.as_str()) {
        ("required", "fused") => {
            let semantic_graph_blake3 = workload
                .semantic_graph_blake3
                .as_deref()
                .ok_or(BenchmarkError::InvalidIdentity)?;
            require_hash(semantic_graph_blake3)?;
            if workload.ordered_layer_plans_blake3.len() != workload.layer_count as usize
                || workload
                    .ordered_layer_plans_blake3
                    .iter()
                    .any(|hash| !is_lower_hex_digest_v1(hash))
            {
                return Err(BenchmarkError::InvalidIdentity);
            }
        }
        ("disabled", "disabled") => {
            if workload.semantic_graph_blake3.is_some()
                || !workload.ordered_layer_plans_blake3.is_empty()
            {
                return Err(BenchmarkError::InvalidIdentity);
            }
        }
        _ => return Err(BenchmarkError::InvalidIdentity),
    }

    let topology = &workload.kernel_variant.expected_topology;
    if topology.dispatch_count == 0
        || topology.compute_pass_count == 0
        || topology.command_buffer_count == 0
        || topology.submission_count == 0
        || topology.map_count == 0
    {
        return Err(BenchmarkError::InvalidIdentity);
    }
    let residency = &workload.residency_plan;
    if residency.activation_buffer_count == 0
        || residency.activation_arena_bytes == 0
        || residency.scratch_arena_bytes == 0
        || residency.main_buffers_bytes == 0
        || residency.logical_gpu_bytes == 0
        || residency.allocated_gpu_bytes < residency.logical_gpu_bytes
        || residency.max_resident_shard_bytes == 0
        || residency.max_resident_shard_bytes > residency.allocated_gpu_bytes
    {
        return Err(BenchmarkError::InvalidIdentity);
    }
    Ok(())
}

fn validate_correctness_anchor(anchor: &CorrectnessAnchorV1) -> Result<(), BenchmarkError> {
    require_hash(&anchor.validator_blake3)?;
    require_hash(&anchor.expected_checkpoint_sha256)?;
    require_hash(&anchor.causal_validator_blake3)?;
    require_nonempty(&anchor.policy_id, BenchmarkError::InvalidIdentity)
}

fn validate_protocol_identity(
    passport: &BenchmarkPassportV1,
    workload: &VisionStackWorkloadV1,
    protocol: &BenchmarkProtocolV1,
) -> Result<(), BenchmarkError> {
    if passport.build_profile != "release"
        || protocol.build_profile != "release"
        || protocol.build_profile != passport.build_profile
        || protocol.clock_resolution_ns == 0
    {
        return Err(BenchmarkError::InvalidProtocol);
    }
    let minimums_satisfied = match protocol.class {
        BenchmarkClassV1::Micro => protocol.warmup_count >= 10 && protocol.measured_count >= 30,
        BenchmarkClassV1::StageMacro => protocol.warmup_count >= 3 && protocol.measured_count >= 10,
    };
    if !minimums_satisfied {
        return Err(BenchmarkError::InvalidProtocol);
    }
    if protocol.schedule != "single-stable-variant-v1" {
        return Err(BenchmarkError::InvalidSchedule);
    }
    if protocol.synchronization != "await-complete-map-validate"
        || protocol.output_validation_policy != "validate-every-sample"
        || protocol.isolation_policy != "dedicated-process-no-background-load"
        || protocol.interruption_policy != "reject-any-interruption"
        || protocol.background_load_policy != "reject-observed-heavy-load"
        || workload.execution_boundary != ExecutionBoundaryV1::ApiWall
    {
        return Err(BenchmarkError::InvalidProtocol);
    }
    let expected_clock = match passport.backend.kind {
        BackendKindV1::NativeWgpu => "std-instant-monotonic",
        BackendKindV1::ChromeWebgpu | BackendKindV1::WebkitWebgpu => "performance-now",
    };
    if protocol.clock_source != expected_clock {
        return Err(BenchmarkError::InvalidProtocol);
    }
    Ok(())
}

fn validate_protocol_sample_counts(input: &BenchmarkEvidenceInputV1) -> Result<(), BenchmarkError> {
    if input.protocol.warmup_count as usize != input.warmup_samples.len()
        || input.protocol.measured_count as usize != input.measured_samples.len()
    {
        return Err(BenchmarkError::InvalidProtocol);
    }
    Ok(())
}

fn validate_cross_links(
    passport: &BenchmarkPassportV1,
    workload: &VisionStackWorkloadV1,
    correctness_anchor: &CorrectnessAnchorV1,
) -> Result<(), BenchmarkError> {
    if passport.model.manifest_sha256 != workload.manifest_sha256
        || workload.checkpoint_sha256 != correctness_anchor.expected_checkpoint_sha256
    {
        return Err(BenchmarkError::CrossLinkMismatch);
    }
    Ok(())
}

fn validate_passport_environment(passport: &BenchmarkPassportV1) -> Result<(), BenchmarkError> {
    for observation in [
        &passport.power_source,
        &passport.power_profile,
        &passport.low_power_mode,
        &passport.thermal_state,
        &passport.display_attached,
    ] {
        validate_observation(observation)?;
    }
    Ok(())
}

fn validate_cohort(
    samples: &[BenchmarkSampleV1],
    input: &BenchmarkEvidenceInputV1,
    timestamp_pairs: &mut BTreeSet<(u64, u64)>,
) -> Result<(), BenchmarkError> {
    for (expected_index, sample) in samples.iter().enumerate() {
        validate_sample(
            sample,
            u32::try_from(expected_index).map_err(|_| BenchmarkError::InvalidIndex)?,
            input,
            timestamp_pairs,
        )?;
    }
    Ok(())
}

fn validate_sample(
    sample: &BenchmarkSampleV1,
    expected_index: u32,
    input: &BenchmarkEvidenceInputV1,
    timestamp_pairs: &mut BTreeSet<(u64, u64)>,
) -> Result<(), BenchmarkError> {
    if sample.index != expected_index {
        return Err(BenchmarkError::InvalidIndex);
    }
    if sample.schedule_slot != expected_index {
        return Err(BenchmarkError::InvalidSchedule);
    }
    if sample.api_wall_ns == 0 || sample.api_wall_ns == u64::MAX {
        return Err(BenchmarkError::InvalidDuration);
    }
    validate_queue_observation(&sample.queue_wall, sample.api_wall_ns)?;
    validate_timestamp_observation(
        &sample.gpu_timestamp,
        input.passport.backend.timestamp_query,
        sample.api_wall_ns,
        timestamp_pairs,
    )?;

    if sample.kernel_variant_id != input.workload.kernel_variant.id
        || sample.residency_plan_id != input.workload.residency_plan.id
        || sample.output_sha256 != input.workload.checkpoint_sha256
    {
        return Err(BenchmarkError::CrossLinkMismatch);
    }
    if sample.topology != input.workload.kernel_variant.expected_topology {
        return Err(BenchmarkError::TopologyMismatch);
    }
    if !is_lower_hex_digest_v1(&sample.correctness_report_blake3)
        || !is_lower_hex_digest_v1(&sample.causal_evidence_blake3)
    {
        return Err(BenchmarkError::InvalidIdentity);
    }
    if sample.logical_gpu_bytes != input.workload.residency_plan.logical_gpu_bytes
        || sample.allocated_gpu_bytes != input.workload.residency_plan.allocated_gpu_bytes
    {
        return Err(BenchmarkError::ResourceMismatch);
    }
    validate_observation(&sample.thermal_before)?;
    validate_observation(&sample.thermal_after)?;
    match &sample.status {
        SampleStatusV1::Passed => {}
        SampleStatusV1::Failed { .. } => return Err(BenchmarkError::FailedSample),
    }
    Ok(())
}

fn validate_queue_observation(
    observation: &DurationObservationV1,
    api_wall_ns: u64,
) -> Result<(), BenchmarkError> {
    match observation {
        DurationObservationV1::Available { duration_ns }
            if *duration_ns > 0 && *duration_ns <= api_wall_ns =>
        {
            Ok(())
        }
        DurationObservationV1::Unavailable { reason } if !reason.trim().is_empty() => Ok(()),
        DurationObservationV1::Available { .. } | DurationObservationV1::Unavailable { .. } => {
            Err(BenchmarkError::InvalidQueueObservation)
        }
    }
}

fn validate_timestamp_observation(
    observation: &GpuTimestampObservationV1,
    timestamp_query: bool,
    api_wall_ns: u64,
    timestamp_pairs: &mut BTreeSet<(u64, u64)>,
) -> Result<(), BenchmarkError> {
    match observation {
        GpuTimestampObservationV1::Unavailable { reason } => {
            if timestamp_query || reason.trim().is_empty() {
                return Err(BenchmarkError::InvalidTimestamp);
            }
            Ok(())
        }
        GpuTimestampObservationV1::Available {
            begin_ticks,
            end_ticks,
            period_ns,
            duration_ns,
        } => {
            if !timestamp_query {
                return Err(BenchmarkError::InvalidTimestamp);
            }
            let exact_duration =
                exact_gpu_timestamp_duration_ns_v1(*begin_ticks, *end_ticks, period_ns)?;
            if exact_duration != *duration_ns || *duration_ns > api_wall_ns {
                return Err(BenchmarkError::InvalidTimestamp);
            }
            if !timestamp_pairs.insert((*begin_ticks, *end_ticks)) {
                return Err(BenchmarkError::StaleTimestamp);
            }
            Ok(())
        }
    }
}

pub fn exact_gpu_timestamp_duration_ns_v1(
    begin_ticks: u64,
    end_ticks: u64,
    period_ns: &str,
) -> Result<u64, BenchmarkError> {
    if begin_ticks == 0 || end_ticks <= begin_ticks {
        return Err(BenchmarkError::InvalidTimestamp);
    }
    let (period_numerator, period_denominator) = parse_exact_decimal_v1(period_ns)?;
    let ticks = u128::from(end_ticks - begin_ticks);
    let scaled = ticks
        .checked_mul(period_numerator)
        .ok_or(BenchmarkError::TimestampOverflow)?;
    if scaled % period_denominator != 0 {
        return Err(BenchmarkError::InvalidTimestamp);
    }
    let exact_duration = scaled / period_denominator;
    if exact_duration == 0 {
        return Err(BenchmarkError::InvalidTimestamp);
    }
    u64::try_from(exact_duration).map_err(|_| BenchmarkError::TimestampOverflow)
}

pub fn parse_exact_decimal_v1(value: &str) -> Result<(u128, u128), BenchmarkError> {
    let (integer, fractional) = match value.split_once('.') {
        Some((integer, fractional)) => (integer, Some(fractional)),
        None => (value, None),
    };
    if integer.is_empty()
        || !integer.bytes().all(|byte| byte.is_ascii_digit())
        || (integer.len() > 1 && integer.starts_with('0'))
        || integer == "0" && fractional.is_none()
        || fractional.is_some_and(|fractional| {
            fractional.is_empty()
                || !fractional.bytes().all(|byte| byte.is_ascii_digit())
                || fractional.ends_with('0')
        })
    {
        return Err(BenchmarkError::InvalidTimestamp);
    }
    let integer = integer
        .parse::<u128>()
        .map_err(|_| BenchmarkError::TimestampOverflow)?;
    let Some(fractional) = fractional else {
        return Ok((integer, 1));
    };
    let denominator = 10_u128
        .checked_pow(
            u32::try_from(fractional.len()).map_err(|_| BenchmarkError::TimestampOverflow)?,
        )
        .ok_or(BenchmarkError::TimestampOverflow)?;
    let fractional = fractional
        .parse::<u128>()
        .map_err(|_| BenchmarkError::TimestampOverflow)?;
    let numerator = integer
        .checked_mul(denominator)
        .and_then(|integer| integer.checked_add(fractional))
        .ok_or(BenchmarkError::TimestampOverflow)?;
    Ok((numerator, denominator))
}

fn validate_observation(observation: &ObservationV1) -> Result<(), BenchmarkError> {
    let valid = match observation {
        ObservationV1::Available { value, method } => {
            !value.trim().is_empty() && !method.trim().is_empty()
        }
        ObservationV1::Unavailable { reason, method } => {
            !reason.trim().is_empty() && !method.trim().is_empty()
        }
    };
    if valid {
        Ok(())
    } else {
        Err(BenchmarkError::InvalidEnvironment)
    }
}

fn require_nonempty(value: &str, error: BenchmarkError) -> Result<(), BenchmarkError> {
    if value.trim().is_empty() {
        Err(error)
    } else {
        Ok(())
    }
}

fn require_optional_nonempty(
    value: Option<&str>,
    error: BenchmarkError,
) -> Result<(), BenchmarkError> {
    match value {
        Some(value) => require_nonempty(value, error),
        None => Err(error),
    }
}

fn require_hash(value: &str) -> Result<(), BenchmarkError> {
    if is_lower_hex_digest_v1(value) {
        Ok(())
    } else {
        Err(BenchmarkError::InvalidIdentity)
    }
}

pub fn is_lower_hex_digest_v1(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical_serializable_bytes(value: &impl Serialize) -> Vec<u8> {
    let value = serde_json::to_value(value)
        .expect("the closed benchmark evidence schema is always serializable");
    canonical_value_bytes(&value)
}

fn canonical_value_bytes(value: &serde_json::Value) -> Vec<u8> {
    let mut bytes =
        serde_json::to_vec(value).expect("an already materialized JSON value is serializable");
    bytes.push(b'\n');
    bytes
}

fn classify_typed_json_error(error: serde_json::Error) -> BenchmarkError {
    let message = error.to_string();
    if message.contains("invalid type: floating point")
        || message.contains("invalid value: integer")
        || message.contains("number out of range")
    {
        BenchmarkError::InvalidInteger
    } else {
        BenchmarkError::SchemaMismatch
    }
}

struct UniqueJson(serde_json::Value);

impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueJsonVisitor)
    }
}

struct UniqueJsonVisitor;

impl<'de> serde::de::Visitor<'de> for UniqueJsonVisitor {
    type Value = UniqueJson;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON value with unique object keys")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(UniqueJson(serde_json::Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(UniqueJson(serde_json::Value::Number(value.into())))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(UniqueJson(serde_json::Value::Number(value.into())))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        let number = serde_json::Number::from_f64(value)
            .ok_or_else(|| E::custom("non-finite JSON number"))?;
        Ok(UniqueJson(serde_json::Value::Number(number)))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.visit_string(value.to_owned())
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(UniqueJson(serde_json::Value::String(value)))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJson(serde_json::Value::Null))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(UniqueJson(serde_json::Value::Null))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(UniqueJson(value)) = sequence.next_element()? {
            values.push(value);
        }
        Ok(UniqueJson(serde_json::Value::Array(values)))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut values = serde_json::Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(serde::de::Error::custom(format_args!(
                    "duplicate object key `{key}`"
                )));
            }
            let UniqueJson(value) = map.next_value()?;
            values.insert(key, value);
        }
        Ok(UniqueJson(serde_json::Value::Object(values)))
    }
}

fn parse_unique_json(bytes: &[u8]) -> Result<serde_json::Value, BenchmarkError> {
    serde_json::from_slice::<UniqueJson>(bytes)
        .map(|value| value.0)
        .map_err(|_| BenchmarkError::SchemaMismatch)
}

#[cfg(test)]
mod tests {
    use super::nearest_rank_index;

    #[test]
    fn nearest_rank_index_is_exact_beyond_the_wasm32_multiplication_boundary() {
        let widened_rank: fn(u64, u64) -> usize = nearest_rank_index;
        assert_eq!(widened_rank(45_210_183, 95), 42_949_673);
        assert_eq!(widened_rank(20, 95), 18);
        assert_eq!(widened_rank(1, 90), 0);
    }
}
