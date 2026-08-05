use serde::{Deserialize, de::DeserializeOwned};

use super::{
    AssembledBenchmarkEvidenceV1, BackendKindV1, BenchmarkError, BenchmarkEvidenceAssemblyInputV1,
    BenchmarkPassportV1, BenchmarkProtocolV1, BenchmarkSampleAttemptV1, CorrectnessAnchorV1,
    LoadOrCompileObservationV1, VisionStackWorkloadV1, canonical_value_bytes,
    classify_typed_json_error, parse_unique_json, validate_benchmark_leaf_v1,
    validate_load_or_compile_observation_v1,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BrowserBenchmarkCohortPlanV1 {
    schema_version: u32,
    passport: BenchmarkPassportV1,
    workload: VisionStackWorkloadV1,
    correctness_anchor: CorrectnessAnchorV1,
    protocol: BenchmarkProtocolV1,
    load_or_compile: LoadOrCompileObservationV1,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BrowserBenchmarkCohortAssemblyRequestV1 {
    schema_version: u32,
    passport: BenchmarkPassportV1,
    workload: VisionStackWorkloadV1,
    correctness_anchor: CorrectnessAnchorV1,
    protocol: BenchmarkProtocolV1,
    load_or_compile: LoadOrCompileObservationV1,
    attempt_log: Vec<BenchmarkSampleAttemptV1>,
}

impl BrowserBenchmarkCohortPlanV1 {
    fn validate(&self) -> Result<(), BenchmarkError> {
        if self.schema_version != 1 {
            return Err(BenchmarkError::UnsupportedSchema);
        }
        if !matches!(
            self.passport.backend.kind,
            BackendKindV1::ChromeWebgpu | BackendKindV1::WebkitWebgpu
        ) {
            return Err(BenchmarkError::InvalidEnvironment);
        }
        1_u32
            .checked_add(self.protocol.warmup_count)
            .and_then(|count| count.checked_add(self.protocol.measured_count))
            .ok_or(BenchmarkError::InvalidProtocol)?;
        validate_benchmark_leaf_v1(
            &self.passport,
            &self.workload,
            &self.correctness_anchor,
            &self.protocol,
        )?;
        validate_load_or_compile_observation_v1(
            &self.load_or_compile,
            &self.passport,
            &self.workload,
            &self.protocol,
        )
    }
}

impl From<BrowserBenchmarkCohortAssemblyRequestV1> for BenchmarkEvidenceAssemblyInputV1 {
    fn from(value: BrowserBenchmarkCohortAssemblyRequestV1) -> Self {
        Self {
            passport: value.passport,
            workload: value.workload,
            correctness_anchor: value.correctness_anchor,
            protocol: value.protocol,
            load_or_compile: value.load_or_compile,
            attempt_log: value.attempt_log,
        }
    }
}

fn parse_canonical_wire<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, BenchmarkError> {
    let value = parse_unique_json(bytes)?;
    if canonical_value_bytes(&value) != bytes {
        return Err(BenchmarkError::NonCanonical);
    }
    let object = value.as_object().ok_or(BenchmarkError::SchemaMismatch)?;
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
    serde_json::from_value(value).map_err(classify_typed_json_error)
}

/// Validates one complete browser cohort plan before any physical benchmark effect.
pub fn validate_browser_benchmark_cohort_plan_v1(
    canonical_plan: &[u8],
) -> Result<(), BenchmarkError> {
    let plan: BrowserBenchmarkCohortPlanV1 = parse_canonical_wire(canonical_plan)?;
    plan.validate()
}

/// Delegates one complete canonical browser-authored journal to the accepted offline assembler.
pub fn assemble_browser_benchmark_cohort_v1(
    canonical_input: &[u8],
) -> Result<Vec<u8>, BenchmarkError> {
    let request: BrowserBenchmarkCohortAssemblyRequestV1 = parse_canonical_wire(canonical_input)?;
    let plan = BrowserBenchmarkCohortPlanV1 {
        schema_version: request.schema_version,
        passport: request.passport.clone(),
        workload: request.workload.clone(),
        correctness_anchor: request.correctness_anchor.clone(),
        protocol: request.protocol.clone(),
        load_or_compile: request.load_or_compile.clone(),
    };
    plan.validate()?;
    AssembledBenchmarkEvidenceV1::assemble(request.into())
        .map(|assembled| assembled.canonical_assembly_bytes())
}
