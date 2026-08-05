use std::collections::BTreeMap;

use pvlc_bench::{
    AssembledBenchmarkEvidenceV1, BenchmarkPassportV1, BenchmarkProtocolV1,
    BenchmarkSampleAttemptV1, CorrectnessAnchorV1, LoadOrCompileObservationV1, ObservationV1,
    VisionStackWorkloadV1, is_lower_hex_digest_v1,
};
use pvlc_runtime_native::{
    VisionQkvStackExecution, VisionStackExecution, native_system_thermal_state_v1,
};
use sha2::{Digest, Sha256};

use crate::{AcceptedVisionStackValidationV1, CollectorError, VisionStackSampleDescriptorV1};

/// Immutable numerical and causal oracle for one native benchmark leaf.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeBenchmarkVisionStackValidationReferenceV1 {
    pub expected_checkpoints: BTreeMap<usize, Vec<f32>>,
    pub expected_output: Vec<f32>,
    pub max_abs_error: f32,
    pub accepted: AcceptedVisionStackValidationV1,
}

/// Complete immutable plan admitted by the sealed native cohort runner.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeBenchmarkLeafPlanV1 {
    pub run_id: String,
    pub passport: BenchmarkPassportV1,
    pub workload: VisionStackWorkloadV1,
    pub correctness_anchor: CorrectnessAnchorV1,
    pub validation_reference: NativeBenchmarkVisionStackValidationReferenceV1,
    pub protocol: BenchmarkProtocolV1,
    pub load_or_compile: LoadOrCompileObservationV1,
    pub base_descriptor: VisionStackSampleDescriptorV1,
}

/// Closed authority for reading the native host thermal state.
#[derive(Clone, Debug)]
pub struct NativeBenchmarkEnvironmentProbeV1 {
    authority: NativeEnvironmentAuthorityV1,
}

#[derive(Clone, Copy, Debug)]
enum NativeEnvironmentAuthorityV1 {
    ProcessInfoThermalState,
}

impl NativeBenchmarkEnvironmentProbeV1 {
    #[must_use]
    pub const fn system() -> Self {
        Self {
            authority: NativeEnvironmentAuthorityV1::ProcessInfoThermalState,
        }
    }

    pub(crate) fn observe(&mut self) -> Result<ObservationV1, String> {
        match self.authority {
            NativeEnvironmentAuthorityV1::ProcessInfoThermalState => {
                Ok(match native_system_thermal_state_v1() {
                    Ok(value) => ObservationV1::Available {
                        value: value.to_owned(),
                        method: "ProcessInfo.thermalState".to_owned(),
                    },
                    Err(reason) => ObservationV1::Unavailable {
                        reason,
                        method: "ProcessInfo.thermalState".to_owned(),
                    },
                })
            }
        }
    }
}

/// Closed numerical and causal authority bound to one exact leaf at construction time.
#[derive(Clone, Debug)]
pub struct NativeBenchmarkVisionStackValidatorV1 {
    bound_leaf: NativeBenchmarkLeafPlanV1,
}

impl NativeBenchmarkVisionStackValidatorV1 {
    pub fn from_leaf(leaf: &NativeBenchmarkLeafPlanV1) -> Result<Self, CollectorError> {
        validate_authored_reference(leaf)?;
        Ok(Self {
            bound_leaf: leaf.clone(),
        })
    }

    pub(crate) fn is_bound_to(&self, leaf: &NativeBenchmarkLeafPlanV1) -> bool {
        self.bound_leaf == *leaf
    }

    pub(crate) fn validate_legacy(
        &mut self,
        execution: &VisionStackExecution,
    ) -> Result<AcceptedVisionStackValidationV1, String> {
        validate_numerical_execution(
            &self.bound_leaf.validation_reference,
            &execution.checkpoints,
            &execution.output,
        )?;
        Ok(self.bound_leaf.validation_reference.accepted.clone())
    }

    pub(crate) fn validate_qkv(
        &mut self,
        execution: &VisionQkvStackExecution,
    ) -> Result<AcceptedVisionStackValidationV1, String> {
        validate_numerical_execution(
            &self.bound_leaf.validation_reference,
            &execution.checkpoints,
            &execution.output,
        )?;
        if execution.evidence.policy != pvlc_runtime_core::VisionQkvExecutionPolicy::Required
            || execution.evidence.outcome != pvlc_runtime_core::VisionQkvSelectionOutcome::Fused
            || execution.evidence.canonical_layer_plan_blake3
                != self.bound_leaf.workload.ordered_layer_plans_blake3
            || u64::try_from(execution.evidence.dispatch_count).ok()
                != Some(
                    self.bound_leaf
                        .workload
                        .kernel_variant
                        .expected_topology
                        .dispatch_count,
                )
            || u64::try_from(execution.evidence.compute_pass_count).ok()
                != Some(
                    self.bound_leaf
                        .workload
                        .kernel_variant
                        .expected_topology
                        .compute_pass_count,
                )
            || u64::try_from(execution.evidence.command_buffer_count).ok()
                != Some(
                    self.bound_leaf
                        .workload
                        .kernel_variant
                        .expected_topology
                        .command_buffer_count,
                )
            || u64::try_from(execution.evidence.submission_count).ok()
                != Some(
                    self.bound_leaf
                        .workload
                        .kernel_variant
                        .expected_topology
                        .submission_count,
                )
            || u64::try_from(execution.evidence.map_count).ok()
                != Some(
                    self.bound_leaf
                        .workload
                        .kernel_variant
                        .expected_topology
                        .map_count,
                )
        {
            return Err("fused Q/K/V causal evidence differs from the bound leaf".to_owned());
        }
        Ok(self.bound_leaf.validation_reference.accepted.clone())
    }
}

pub(crate) fn validate_authored_reference(
    leaf: &NativeBenchmarkLeafPlanV1,
) -> Result<(), CollectorError> {
    let reference = &leaf.validation_reference;
    let expected_elements = u64::from(leaf.workload.tokens)
        .checked_mul(u64::from(leaf.workload.hidden_size))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(CollectorError::InvalidDescriptor)?;
    if !reference.max_abs_error.is_finite()
        || reference.max_abs_error < 0.0
        || expected_elements == 0
        || reference.expected_output.len() != expected_elements
        || reference
            .expected_output
            .iter()
            .any(|value| !value.is_finite())
        || reference.expected_checkpoints.is_empty()
        || reference.expected_checkpoints.values().any(|tensor| {
            tensor.len() != expected_elements || tensor.iter().any(|value| !value.is_finite())
        })
        || reference.accepted.output_sha256 != leaf.base_descriptor.expected_output_sha256
        || reference.accepted.output_sha256 != leaf.workload.checkpoint_sha256
        || reference.accepted.output_sha256 != leaf.correctness_anchor.expected_checkpoint_sha256
        || !is_lower_hex_digest_v1(&reference.accepted.output_sha256)
        || !is_lower_hex_digest_v1(&reference.accepted.correctness_report_blake3)
        || !is_lower_hex_digest_v1(&reference.accepted.causal_evidence_blake3)
    {
        return Err(CollectorError::InvalidDescriptor);
    }
    if semantic_readback_sha256(reference) != reference.accepted.output_sha256 {
        return Err(CollectorError::InvalidDescriptor);
    }
    Ok(())
}

fn semantic_readback_sha256(reference: &NativeBenchmarkVisionStackValidationReferenceV1) -> String {
    let mut hasher = Sha256::new();
    for tensor in reference.expected_checkpoints.values() {
        for value in tensor {
            hasher.update(value.to_le_bytes());
        }
    }
    for value in &reference.expected_output {
        hasher.update(value.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(64);
    const LOWER_HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        encoded.push(char::from(LOWER_HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(LOWER_HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn validate_numerical_execution(
    reference: &NativeBenchmarkVisionStackValidationReferenceV1,
    checkpoints: &BTreeMap<usize, Vec<f32>>,
    output: &[f32],
) -> Result<(), String> {
    if checkpoints.len() != reference.expected_checkpoints.len()
        || checkpoints.keys().ne(reference.expected_checkpoints.keys())
    {
        return Err("checkpoint key set differs from the bound reference".to_owned());
    }
    validate_tensor(output, &reference.expected_output, reference.max_abs_error)?;
    for (layer, expected) in &reference.expected_checkpoints {
        let actual = checkpoints
            .get(layer)
            .ok_or_else(|| "checkpoint key set differs from the bound reference".to_owned())?;
        validate_tensor(actual, expected, reference.max_abs_error)?;
    }
    Ok(())
}

fn validate_tensor(actual: &[f32], expected: &[f32], tolerance: f32) -> Result<(), String> {
    if actual.len() != expected.len()
        || actual.iter().zip(expected).any(|(actual, expected)| {
            !actual.is_finite() || (*actual - *expected).abs() > tolerance
        })
    {
        return Err("tensor differs from the bound numerical reference".to_owned());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeBenchmarkCohortFailurePhaseV1 {
    StaticAdmission,
    Attempt,
}

impl NativeBenchmarkCohortFailurePhaseV1 {
    const fn as_str(self) -> &'static str {
        match self {
            Self::StaticAdmission => "static_admission",
            Self::Attempt => "attempt",
        }
    }
}

/// Opaque successful completion of one exact physical cohort.
#[derive(Debug, PartialEq, Eq)]
pub struct NativeBenchmarkCohortSuccessV1 {
    run_id: String,
    attempt_count: usize,
    assembled: AssembledBenchmarkEvidenceV1,
}

impl NativeBenchmarkCohortSuccessV1 {
    pub(crate) fn new(
        run_id: String,
        attempt_count: usize,
        assembled: AssembledBenchmarkEvidenceV1,
    ) -> Self {
        Self {
            run_id,
            attempt_count,
            assembled,
        }
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub const fn attempt_count(&self) -> usize {
        self.attempt_count
    }

    pub const fn assembled(&self) -> &AssembledBenchmarkEvidenceV1 {
        &self.assembled
    }
}

/// Opaque, canonical, self-hashed terminal failure of one physical cohort.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeBenchmarkCohortFailureV1 {
    run_id: String,
    phase: NativeBenchmarkCohortFailurePhaseV1,
    failure_code: String,
    expected_attempt_count: u64,
    attempt_log: Vec<BenchmarkSampleAttemptV1>,
    canonical_bytes: Vec<u8>,
}

impl NativeBenchmarkCohortFailureV1 {
    pub(crate) fn new(
        run_id: String,
        phase: NativeBenchmarkCohortFailurePhaseV1,
        failure_code: impl Into<String>,
        expected_attempt_count: u64,
        attempt_log: Vec<BenchmarkSampleAttemptV1>,
    ) -> Self {
        let failure_code = failure_code.into();
        let mut unsigned = serde_json::Map::new();
        unsigned.insert(
            "attempt_log".to_owned(),
            serde_json::to_value(&attempt_log)
                .expect("the closed benchmark attempt schema is serializable"),
        );
        unsigned.insert(
            "expected_attempt_count".to_owned(),
            serde_json::Value::from(expected_attempt_count),
        );
        unsigned.insert(
            "failure_code".to_owned(),
            serde_json::Value::String(failure_code.clone()),
        );
        unsigned.insert(
            "phase".to_owned(),
            serde_json::Value::String(phase.as_str().to_owned()),
        );
        unsigned.insert(
            "run_id".to_owned(),
            serde_json::Value::String(run_id.clone()),
        );
        unsigned.insert("schema_version".to_owned(), serde_json::Value::from(1));
        unsigned.insert(
            "status".to_owned(),
            serde_json::Value::String("failed".to_owned()),
        );

        let mut unsigned_bytes = serde_json::to_vec(&unsigned)
            .expect("the closed benchmark failure schema is serializable");
        unsigned_bytes.push(b'\n');
        let failure_blake3 = blake3::hash(&unsigned_bytes).to_hex().to_string();
        unsigned.insert(
            "failure_blake3".to_owned(),
            serde_json::Value::String(failure_blake3),
        );
        let mut canonical_bytes = serde_json::to_vec(&unsigned)
            .expect("the closed benchmark failure schema is serializable");
        canonical_bytes.push(b'\n');

        Self {
            run_id,
            phase,
            failure_code,
            expected_attempt_count,
            attempt_log,
            canonical_bytes,
        }
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub const fn phase(&self) -> NativeBenchmarkCohortFailurePhaseV1 {
        self.phase
    }

    pub fn code(&self) -> &str {
        &self.failure_code
    }

    pub const fn expected_attempt_count(&self) -> u64 {
        self.expected_attempt_count
    }

    pub fn attempt_log(&self) -> &Vec<BenchmarkSampleAttemptV1> {
        &self.attempt_log
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        self.canonical_bytes.clone()
    }
}
