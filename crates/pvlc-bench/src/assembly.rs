use serde::{Deserialize, Serialize};

use super::{
    BenchmarkError, BenchmarkEvidenceInputV1, BenchmarkEvidenceV1, BenchmarkPassportV1,
    BenchmarkProtocolV1, BenchmarkSampleV1, CorrectnessAnchorV1, ExecutionBoundaryV1,
    ObservationV1, SampleStatusV1, VisionStackWorkloadV1, canonical_serializable_bytes,
    canonical_value_bytes, classify_typed_json_error, parse_unique_json, validate_observation,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkCohortV1 {
    Cold,
    Warmup,
    Measured,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
// This offline journal intentionally owns each complete sample snapshot by value. Boxing would
// weaken the frozen construction API without affecting the canonical wire or any GPU hot path.
#[allow(clippy::large_enum_variant)]
pub enum BenchmarkSampleAttemptV1 {
    Passed {
        sequence: u32,
        cohort: BenchmarkCohortV1,
        planned_slot: u32,
        sample: BenchmarkSampleV1,
    },
    Failed {
        sequence: u32,
        cohort: BenchmarkCohortV1,
        planned_slot: u32,
        code: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadOrCompileObservationV1 {
    pub execution_boundary: ExecutionBoundaryV1,
    pub duration_ns: u64,
    pub clock_source: String,
    pub clock_resolution_ns: u64,
    pub passport_blake3: String,
    pub workload_blake3: String,
    pub protocol_blake3: String,
    pub thermal_before: ObservationV1,
    pub thermal_after: ObservationV1,
    pub status: SampleStatusV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkEvidenceAssemblyInputV1 {
    pub passport: BenchmarkPassportV1,
    pub workload: VisionStackWorkloadV1,
    pub correctness_anchor: CorrectnessAnchorV1,
    pub protocol: BenchmarkProtocolV1,
    pub load_or_compile: LoadOrCompileObservationV1,
    pub attempt_log: Vec<BenchmarkSampleAttemptV1>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkEvidenceAssemblyDataV1 {
    schema_version: u32,
    passport: BenchmarkPassportV1,
    workload: VisionStackWorkloadV1,
    correctness_anchor: CorrectnessAnchorV1,
    protocol: BenchmarkProtocolV1,
    load_or_compile: LoadOrCompileObservationV1,
    attempt_log: Vec<BenchmarkSampleAttemptV1>,
    assembly_blake3: String,
}

impl BenchmarkEvidenceAssemblyDataV1 {
    fn from_input(input: &BenchmarkEvidenceAssemblyInputV1) -> Self {
        let mut data = Self {
            schema_version: 1,
            passport: input.passport.clone(),
            workload: input.workload.clone(),
            correctness_anchor: input.correctness_anchor.clone(),
            protocol: input.protocol.clone(),
            load_or_compile: input.load_or_compile.clone(),
            attempt_log: input.attempt_log.clone(),
            assembly_blake3: String::new(),
        };
        data.assembly_blake3 = data.computed_blake3();
        data
    }

    fn computed_blake3(&self) -> String {
        let mut value = serde_json::to_value(self)
            .expect("the closed benchmark assembly schema is always serializable");
        value
            .as_object_mut()
            .expect("benchmark assembly serializes as an object")
            .remove("assembly_blake3");
        blake3::hash(&canonical_value_bytes(&value))
            .to_hex()
            .to_string()
    }

    fn into_input(self) -> BenchmarkEvidenceAssemblyInputV1 {
        BenchmarkEvidenceAssemblyInputV1 {
            passport: self.passport,
            workload: self.workload,
            correctness_anchor: self.correctness_anchor,
            protocol: self.protocol,
            load_or_compile: self.load_or_compile,
            attempt_log: self.attempt_log,
        }
    }
}

/// Canonical, self-hashed V1 bytes for one supplied offline assembly journal.
///
/// The digest makes an already-authored journal immutable by identity. It does not prove that a
/// physical attempt was omitted before authorship. The sealed M7d1a3 cohort runner is responsible
/// for producing the complete journal that this pure boundary validates and preserves.
pub fn canonical_benchmark_evidence_assembly_bytes_v1(
    input: &BenchmarkEvidenceAssemblyInputV1,
) -> Vec<u8> {
    canonical_serializable_bytes(&BenchmarkEvidenceAssemblyDataV1::from_input(input))
}

/// Opaque result of validating one content-addressed preparation and supplied attempt journal.
///
/// This pure offline type attests only to the exact supplied journal. Physical run completeness is
/// intentionally outside its authority and belongs to the sealed M7d1a3 cohort runner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssembledBenchmarkEvidenceV1 {
    canonical_assembly: Vec<u8>,
    assembly_blake3: String,
    load_or_compile: LoadOrCompileObservationV1,
    evidence: BenchmarkEvidenceV1,
}

impl AssembledBenchmarkEvidenceV1 {
    pub fn assemble(input: BenchmarkEvidenceAssemblyInputV1) -> Result<Self, BenchmarkError> {
        validate_preparation(&input)?;
        let evidence_input = validated_evidence_input(&input)?;
        let evidence = BenchmarkEvidenceV1::build(evidence_input)?;
        let data = BenchmarkEvidenceAssemblyDataV1::from_input(&input);
        let canonical_assembly = canonical_serializable_bytes(&data);
        Ok(Self {
            canonical_assembly,
            assembly_blake3: data.assembly_blake3,
            load_or_compile: input.load_or_compile,
            evidence,
        })
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

        let supplied_hash = object
            .get("assembly_blake3")
            .and_then(serde_json::Value::as_str)
            .ok_or(BenchmarkError::SchemaMismatch)?;
        let mut unsigned = unique.clone();
        unsigned
            .as_object_mut()
            .expect("top-level object checked above")
            .remove("assembly_blake3");
        let expected_hash = blake3::hash(&canonical_value_bytes(&unsigned))
            .to_hex()
            .to_string();
        if supplied_hash != expected_hash {
            return Err(BenchmarkError::SelfHashMismatch);
        }

        let data: BenchmarkEvidenceAssemblyDataV1 =
            serde_json::from_value(unique).map_err(classify_typed_json_error)?;
        if data.schema_version != 1 {
            return Err(BenchmarkError::UnsupportedSchema);
        }
        let assembled = Self::assemble(data.into_input())?;
        if assembled.canonical_assembly != bytes {
            return Err(BenchmarkError::NonCanonical);
        }
        Ok(assembled)
    }

    pub fn canonical_assembly_bytes(&self) -> Vec<u8> {
        self.canonical_assembly.clone()
    }

    pub fn assembly_blake3(&self) -> &str {
        &self.assembly_blake3
    }

    pub fn load_or_compile(&self) -> &LoadOrCompileObservationV1 {
        &self.load_or_compile
    }

    pub fn evidence(&self) -> &BenchmarkEvidenceV1 {
        &self.evidence
    }
}

fn validate_preparation(input: &BenchmarkEvidenceAssemblyInputV1) -> Result<(), BenchmarkError> {
    validate_load_or_compile_observation_v1(
        &input.load_or_compile,
        &input.passport,
        &input.workload,
        &input.protocol,
    )
}

/// Validates the immutable preparation observation for a benchmark leaf.
///
/// This reusable static-admission boundary checks content links and preparation status without
/// accepting or constructing an attempt journal.
pub fn validate_load_or_compile_observation_v1(
    preparation: &LoadOrCompileObservationV1,
    passport: &BenchmarkPassportV1,
    workload: &VisionStackWorkloadV1,
    protocol: &BenchmarkProtocolV1,
) -> Result<(), BenchmarkError> {
    if preparation.execution_boundary != ExecutionBoundaryV1::LoadOrCompile
        || preparation.duration_ns == 0
        || preparation.duration_ns == u64::MAX
        || preparation.clock_source != protocol.clock_source
        || preparation.clock_resolution_ns != protocol.clock_resolution_ns
        || preparation.passport_blake3 != component_blake3(passport)
        || preparation.workload_blake3 != component_blake3(workload)
        || preparation.protocol_blake3 != component_blake3(protocol)
        || preparation.thermal_before != passport.thermal_state
    {
        return Err(BenchmarkError::InvalidPreparation);
    }
    validate_observation(&preparation.thermal_before)
        .map_err(|_| BenchmarkError::InvalidPreparation)?;
    validate_observation(&preparation.thermal_after)
        .map_err(|_| BenchmarkError::InvalidPreparation)?;
    match &preparation.status {
        SampleStatusV1::Passed => Ok(()),
        SampleStatusV1::Failed { code } if code.trim().is_empty() => {
            Err(BenchmarkError::InvalidPreparation)
        }
        SampleStatusV1::Failed { .. } => Err(BenchmarkError::FailedSample),
    }
}

fn component_blake3(value: &impl Serialize) -> String {
    blake3::hash(&canonical_serializable_bytes(value))
        .to_hex()
        .to_string()
}

fn validated_evidence_input(
    input: &BenchmarkEvidenceAssemblyInputV1,
) -> Result<BenchmarkEvidenceInputV1, BenchmarkError> {
    for attempt in &input.attempt_log {
        match attempt {
            BenchmarkSampleAttemptV1::Failed { code, .. } if code.trim().is_empty() => {
                return Err(BenchmarkError::InvalidAttemptJournal);
            }
            BenchmarkSampleAttemptV1::Failed { .. } => {
                return Err(BenchmarkError::FailedSample);
            }
            BenchmarkSampleAttemptV1::Passed { sample, .. }
                if matches!(sample.status, SampleStatusV1::Failed { .. }) =>
            {
                return Err(BenchmarkError::FailedSample);
            }
            BenchmarkSampleAttemptV1::Passed { .. } => {}
        }
    }

    let warmup_count = usize::try_from(input.protocol.warmup_count)
        .map_err(|_| BenchmarkError::InvalidAttemptJournal)?;
    let measured_count = usize::try_from(input.protocol.measured_count)
        .map_err(|_| BenchmarkError::InvalidAttemptJournal)?;
    let expected_count = 1_usize
        .checked_add(warmup_count)
        .and_then(|count| count.checked_add(measured_count))
        .ok_or(BenchmarkError::InvalidAttemptJournal)?;
    if input.attempt_log.len() != expected_count {
        return Err(BenchmarkError::InvalidAttemptJournal);
    }

    let mut cold_sample = None;
    let mut warmup_samples = Vec::with_capacity(warmup_count);
    let mut measured_samples = Vec::with_capacity(measured_count);
    for (position, attempt) in input.attempt_log.iter().enumerate() {
        let sequence =
            u32::try_from(position).map_err(|_| BenchmarkError::InvalidAttemptJournal)?;
        let (expected_cohort, expected_slot) = if position == 0 {
            (BenchmarkCohortV1::Cold, 0)
        } else if position <= warmup_count {
            (
                BenchmarkCohortV1::Warmup,
                u32::try_from(position - 1).map_err(|_| BenchmarkError::InvalidAttemptJournal)?,
            )
        } else {
            (
                BenchmarkCohortV1::Measured,
                u32::try_from(position - 1 - warmup_count)
                    .map_err(|_| BenchmarkError::InvalidAttemptJournal)?,
            )
        };

        let BenchmarkSampleAttemptV1::Passed {
            sequence: actual_sequence,
            cohort,
            planned_slot,
            sample,
        } = attempt
        else {
            unreachable!("failed attempts returned before journal layout validation");
        };
        if *actual_sequence != sequence
            || *cohort != expected_cohort
            || *planned_slot != expected_slot
            || sample.index != expected_slot
            || sample.schedule_slot != expected_slot
        {
            return Err(BenchmarkError::InvalidAttemptJournal);
        }
        match expected_cohort {
            BenchmarkCohortV1::Cold => cold_sample = Some(sample.clone()),
            BenchmarkCohortV1::Warmup => warmup_samples.push(sample.clone()),
            BenchmarkCohortV1::Measured => measured_samples.push(sample.clone()),
        }
    }

    Ok(BenchmarkEvidenceInputV1 {
        passport: input.passport.clone(),
        workload: input.workload.clone(),
        correctness_anchor: input.correctness_anchor.clone(),
        protocol: input.protocol.clone(),
        cold_sample: cold_sample.ok_or(BenchmarkError::InvalidAttemptJournal)?,
        warmup_samples,
        measured_samples,
    })
}
