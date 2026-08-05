#![forbid(unsafe_code)]

use std::time::Instant;

use pvlc_bench::{
    BenchmarkSampleV1, DurationObservationV1, ExpectedTopologyV1, GpuTimestampObservationV1,
    ObservationV1, SampleStatusV1, exact_gpu_timestamp_duration_ns_v1, is_lower_hex_digest_v1,
};
use pvlc_passes::VisionQkvStackSelection;
use pvlc_runtime_core::{
    VisionEncoderStackInvocation, VisionQkvExecutionPolicy, VisionQkvSelectionOutcome,
    VisionStackActivationStrategy,
};
use pvlc_runtime_native::{
    ErrorScopeKind, NativeRuntime, RuntimeError, VisionQkvStackExecution, VisionStackDiagnostics,
    VisionStackExecution,
};

mod cohort;
mod cohort_types;

#[cfg(not(test))]
pub use cohort::{
    run_native_public_legacy_benchmark_cohort_v1, run_native_public_qkv_benchmark_cohort_v1,
};
#[cfg(test)]
pub(crate) use cohort::{
    run_native_public_legacy_benchmark_cohort_v1, run_native_public_qkv_benchmark_cohort_v1,
};
pub use cohort_types::{
    NativeBenchmarkCohortFailurePhaseV1, NativeBenchmarkCohortFailureV1,
    NativeBenchmarkCohortSuccessV1, NativeBenchmarkEnvironmentProbeV1, NativeBenchmarkLeafPlanV1,
    NativeBenchmarkVisionStackValidationReferenceV1, NativeBenchmarkVisionStackValidatorV1,
};

const CHECKED_SCOPE_ORDER: [ErrorScopeKind; 3] = [
    ErrorScopeKind::Validation,
    ErrorScopeKind::OutOfMemory,
    ErrorScopeKind::Internal,
];
const NATIVE_QUEUE_UNAVAILABLE_REASON: &str = "native queue-wall observation unavailable";
const NATIVE_TIMESTAMP_UNAVAILABLE_REASON: &str =
    "native runtime timestamp-query feature unavailable";

pub const LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1: &str = "vision-stack-legacy-f32-v1";
pub const FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1: &str = "vision-qkv-fused-f32-v1";

#[derive(Clone, Debug, PartialEq)]
struct NativeTimestampFactsV1 {
    begin_ticks: u64,
    end_ticks: u64,
    period_ns: f64,
    reported_duration_ns: f64,
    fresh: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
struct NativeVisionStackFactsV1 {
    checked_error_scopes: [ErrorScopeKind; 3],
    captured_errors: Vec<String>,
    queue_wall_time_ns: Option<u64>,
    timestamp: Option<NativeTimestampFactsV1>,
    topology: ExpectedTopologyV1,
    activation_strategy: VisionStackActivationStrategy,
    activation_buffer_count: u64,
    activation_arena_bytes: u64,
    scratch_arena_bytes: u64,
    main_buffers_bytes: u64,
}

trait CollectorClockV1 {
    fn now_ns(&mut self) -> Result<u64, String>;
}

mod sealed {
    pub trait Sealed {}

    impl Sealed for pvlc_runtime_native::NativeRuntime {}
}

#[doc(hidden)]
pub trait NativeCollectorRuntimeV1: sealed::Sealed {
    type MonotonicReading;

    fn timestamp_query(&self) -> bool;
    fn collector_monotonic_now(&self) -> Result<Self::MonotonicReading, String>;
    fn collector_elapsed_ns(
        &self,
        started: &Self::MonotonicReading,
        ended: &Self::MonotonicReading,
    ) -> Result<u64, String>;
}

impl NativeCollectorRuntimeV1 for NativeRuntime {
    type MonotonicReading = Instant;

    fn timestamp_query(&self) -> bool {
        self.capabilities().timestamp_query
    }

    fn collector_monotonic_now(&self) -> Result<Self::MonotonicReading, String> {
        Ok(Instant::now())
    }

    fn collector_elapsed_ns(
        &self,
        started: &Self::MonotonicReading,
        ended: &Self::MonotonicReading,
    ) -> Result<u64, String> {
        let duration = ended
            .checked_duration_since(*started)
            .ok_or_else(|| "native monotonic clock reversed".to_owned())?;
        u64::try_from(duration.as_nanos())
            .map_err(|_| "native monotonic clock exceeded u64 nanoseconds".to_owned())
    }
}

pub type NativeLegacyPublicOperationV1<R> = fn(
    &R,
    &VisionEncoderStackInvocation<'_>,
    &[usize],
    VisionStackActivationStrategy,
) -> Result<VisionStackExecution, RuntimeError>;

pub type NativeQkvPublicOperationV1<R> = fn(
    &R,
    &VisionEncoderStackInvocation<'_>,
    &[usize],
    VisionStackActivationStrategy,
    &VisionQkvStackSelection,
) -> Result<VisionQkvStackExecution, RuntimeError>;

#[doc(hidden)]
pub trait NativePublicVisionStackRuntimeV1: NativeCollectorRuntimeV1 + Sized {
    const LEGACY_PUBLIC_OPERATION_V1: NativeLegacyPublicOperationV1<Self>;
    const QKV_PUBLIC_OPERATION_V1: NativeQkvPublicOperationV1<Self>;
}

impl NativePublicVisionStackRuntimeV1 for NativeRuntime {
    const LEGACY_PUBLIC_OPERATION_V1: NativeLegacyPublicOperationV1<Self> =
        NativeRuntime::run_vision_encoder_stack_identity_rope_with_activation_strategy;
    const QKV_PUBLIC_OPERATION_V1: NativeQkvPublicOperationV1<Self> =
        NativeRuntime::run_vision_encoder_stack_identity_rope_with_qkv_selection;
}

struct NativeRuntimeClockV1<'a, R: NativeCollectorRuntimeV1> {
    runtime: &'a R,
    started: Option<R::MonotonicReading>,
    closed: bool,
}

impl<'a, R: NativeCollectorRuntimeV1> NativeRuntimeClockV1<'a, R> {
    fn new(runtime: &'a R) -> Self {
        Self {
            runtime,
            started: None,
            closed: false,
        }
    }
}

impl<R: NativeCollectorRuntimeV1> CollectorClockV1 for NativeRuntimeClockV1<'_, R> {
    fn now_ns(&mut self) -> Result<u64, String> {
        if self.closed {
            return Err("native collector clock was read more than twice".to_owned());
        }
        let reading = self.runtime.collector_monotonic_now()?;
        let Some(started) = self.started.as_ref() else {
            self.started = Some(reading);
            return Ok(0);
        };
        self.closed = true;
        self.runtime.collector_elapsed_ns(started, &reading)
    }
}

trait NativeCollectorExecutionV1 {
    fn collector_facts(&self) -> Result<NativeVisionStackFactsV1, CollectorError>;
}

impl NativeCollectorExecutionV1 for VisionStackExecution {
    fn collector_facts(&self) -> Result<NativeVisionStackFactsV1, CollectorError> {
        native_facts(&self.diagnostics)
    }
}

impl NativeCollectorExecutionV1 for VisionQkvStackExecution {
    fn collector_facts(&self) -> Result<NativeVisionStackFactsV1, CollectorError> {
        native_facts(&self.diagnostics)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionStackSampleDescriptorV1 {
    pub index: u32,
    pub schedule_slot: u32,
    pub kernel_variant_id: String,
    pub residency_plan_id: String,
    pub expected_topology: ExpectedTopologyV1,
    pub expected_output_sha256: String,
    pub logical_gpu_bytes: u64,
    pub allocated_gpu_bytes: u64,
    pub activation_strategy: String,
    pub activation_buffer_count: u64,
    pub activation_arena_bytes: u64,
    pub scratch_arena_bytes: u64,
    pub main_buffers_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcceptedVisionStackValidationV1 {
    pub output_sha256: String,
    pub correctness_report_blake3: String,
    pub causal_evidence_blake3: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectorErrorCodeV1 {
    NotImplemented,
    InvalidDescriptor,
    EnvironmentProbeFailed,
    ClockFailed,
    NonMonotonicClock,
    ExecutionFailed,
    ValidationFailed,
    InvalidDiagnostics,
    InvalidQueueObservation,
    InvalidTimestamp,
    TopologyMismatch,
    ResourceMismatch,
    CrossLinkMismatch,
    OperationBindingMismatch,
}

#[derive(Debug, thiserror::Error)]
pub enum CollectorError {
    #[error("M7d1a2 collector implementation is not present")]
    NotImplemented,
    #[error("collector sample descriptor is invalid")]
    InvalidDescriptor,
    #[error("collector environment probe failed")]
    EnvironmentProbeFailed,
    #[error("collector monotonic clock failed")]
    ClockFailed,
    #[error("collector monotonic clock is zero, reversed, or saturated")]
    NonMonotonicClock,
    #[error("collector execution failed")]
    ExecutionFailed,
    #[error("collector correctness or causal validation failed")]
    ValidationFailed,
    #[error("collector runtime diagnostics are invalid")]
    InvalidDiagnostics,
    #[error("collector queue-wall observation is invalid")]
    InvalidQueueObservation,
    #[error("collector GPU timestamp is invalid")]
    InvalidTimestamp,
    #[error("collector topology differs from the immutable descriptor")]
    TopologyMismatch,
    #[error("collector resource facts differ from the immutable descriptor")]
    ResourceMismatch,
    #[error("collector output or validator identity differs from the descriptor")]
    CrossLinkMismatch,
    #[error("collector operation differs from the immutable public-operation binding")]
    OperationBindingMismatch,
}

impl CollectorError {
    pub fn code(&self) -> CollectorErrorCodeV1 {
        match self {
            Self::NotImplemented => CollectorErrorCodeV1::NotImplemented,
            Self::InvalidDescriptor => CollectorErrorCodeV1::InvalidDescriptor,
            Self::EnvironmentProbeFailed => CollectorErrorCodeV1::EnvironmentProbeFailed,
            Self::ClockFailed => CollectorErrorCodeV1::ClockFailed,
            Self::NonMonotonicClock => CollectorErrorCodeV1::NonMonotonicClock,
            Self::ExecutionFailed => CollectorErrorCodeV1::ExecutionFailed,
            Self::ValidationFailed => CollectorErrorCodeV1::ValidationFailed,
            Self::InvalidDiagnostics => CollectorErrorCodeV1::InvalidDiagnostics,
            Self::InvalidQueueObservation => CollectorErrorCodeV1::InvalidQueueObservation,
            Self::InvalidTimestamp => CollectorErrorCodeV1::InvalidTimestamp,
            Self::TopologyMismatch => CollectorErrorCodeV1::TopologyMismatch,
            Self::ResourceMismatch => CollectorErrorCodeV1::ResourceMismatch,
            Self::CrossLinkMismatch => CollectorErrorCodeV1::CrossLinkMismatch,
            Self::OperationBindingMismatch => CollectorErrorCodeV1::OperationBindingMismatch,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_native_sample_with_clock<R, T, C, P, E, V>(
    runtime: &R,
    descriptor: &VisionStackSampleDescriptorV1,
    clock: &mut C,
    mut environment_probe: P,
    execute: E,
    validate: V,
) -> Result<BenchmarkSampleV1, CollectorError>
where
    R: NativeCollectorRuntimeV1,
    T: NativeCollectorExecutionV1,
    C: CollectorClockV1,
    P: FnMut() -> Result<ObservationV1, String>,
    E: FnOnce(&R) -> Result<T, String>,
    V: FnOnce(&T) -> Result<AcceptedVisionStackValidationV1, String>,
{
    validate_descriptor(descriptor)?;

    let thermal_before = environment_probe()
        .map_err(|_| CollectorError::EnvironmentProbeFailed)
        .and_then(validate_environment_observation)?;
    let started_ns = clock.now_ns().map_err(|_| CollectorError::ClockFailed)?;

    let execution = match execute(runtime) {
        Ok(execution) => execution,
        Err(_) => {
            close_failed_timing(clock)?;
            return Err(CollectorError::ExecutionFailed);
        }
    };
    let accepted = match validate(&execution) {
        Ok(accepted) => accepted,
        Err(_) => {
            close_failed_timing(clock)?;
            return Err(CollectorError::ValidationFailed);
        }
    };

    let ended_ns = clock.now_ns().map_err(|_| CollectorError::ClockFailed)?;
    let api_wall_ns = ended_ns
        .checked_sub(started_ns)
        .filter(|duration| *duration > 0 && *duration != u64::MAX)
        .ok_or(CollectorError::NonMonotonicClock)?;
    let thermal_after = environment_probe()
        .map_err(|_| CollectorError::EnvironmentProbeFailed)
        .and_then(validate_environment_observation)?;
    let facts = execution.collector_facts()?;

    admit_sample(
        runtime.timestamp_query(),
        descriptor,
        accepted,
        facts,
        api_wall_ns,
        thermal_before,
        thermal_after,
    )
}

pub fn collect_native_vision_stack_sample<R, P, E, V>(
    runtime: &R,
    descriptor: &VisionStackSampleDescriptorV1,
    environment_probe: P,
    execute: E,
    validate: V,
) -> Result<BenchmarkSampleV1, CollectorError>
where
    R: NativeCollectorRuntimeV1,
    P: FnMut() -> Result<ObservationV1, String>,
    E: FnOnce(&R) -> Result<VisionStackExecution, String>,
    V: FnOnce(&VisionStackExecution) -> Result<AcceptedVisionStackValidationV1, String>,
{
    let mut clock = NativeRuntimeClockV1::new(runtime);
    collect_native_sample_with_clock(
        runtime,
        descriptor,
        &mut clock,
        environment_probe,
        execute,
        validate,
    )
}

pub fn collect_native_qkv_vision_stack_sample<R, P, E, V>(
    runtime: &R,
    descriptor: &VisionStackSampleDescriptorV1,
    environment_probe: P,
    execute: E,
    validate: V,
) -> Result<BenchmarkSampleV1, CollectorError>
where
    R: NativeCollectorRuntimeV1,
    P: FnMut() -> Result<ObservationV1, String>,
    E: FnOnce(&R) -> Result<VisionQkvStackExecution, String>,
    V: FnOnce(&VisionQkvStackExecution) -> Result<AcceptedVisionStackValidationV1, String>,
{
    let mut clock = NativeRuntimeClockV1::new(runtime);
    collect_native_sample_with_clock(
        runtime,
        descriptor,
        &mut clock,
        environment_probe,
        execute,
        validate,
    )
}

pub fn collect_native_public_legacy_vision_stack_sample<R, P, V>(
    runtime: &R,
    descriptor: &VisionStackSampleDescriptorV1,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
    environment_probe: P,
    validate: V,
) -> Result<BenchmarkSampleV1, CollectorError>
where
    R: NativePublicVisionStackRuntimeV1,
    P: FnMut() -> Result<ObservationV1, String>,
    V: FnOnce(&VisionStackExecution) -> Result<AcceptedVisionStackValidationV1, String>,
{
    validate_descriptor(descriptor)?;
    validate_public_operation_binding(
        descriptor,
        activation_strategy,
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
    )?;
    collect_native_vision_stack_sample(
        runtime,
        descriptor,
        environment_probe,
        |runtime| {
            (R::LEGACY_PUBLIC_OPERATION_V1)(
                runtime,
                invocation,
                checkpoint_layers,
                activation_strategy,
            )
            .map_err(|error| error.to_string())
        },
        validate,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn collect_native_public_qkv_vision_stack_sample<R, P, V>(
    runtime: &R,
    descriptor: &VisionStackSampleDescriptorV1,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
    selection: &VisionQkvStackSelection,
    environment_probe: P,
    validate: V,
) -> Result<BenchmarkSampleV1, CollectorError>
where
    R: NativePublicVisionStackRuntimeV1,
    P: FnMut() -> Result<ObservationV1, String>,
    V: FnOnce(&VisionQkvStackExecution) -> Result<AcceptedVisionStackValidationV1, String>,
{
    validate_descriptor(descriptor)?;
    validate_public_operation_binding(
        descriptor,
        activation_strategy,
        FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1,
    )?;
    if selection.policy() != VisionQkvExecutionPolicy::Required
        || selection.outcome() != VisionQkvSelectionOutcome::Fused
    {
        return Err(CollectorError::OperationBindingMismatch);
    }
    collect_native_qkv_vision_stack_sample(
        runtime,
        descriptor,
        environment_probe,
        |runtime| {
            (R::QKV_PUBLIC_OPERATION_V1)(
                runtime,
                invocation,
                checkpoint_layers,
                activation_strategy,
                selection,
            )
            .map_err(|error| error.to_string())
        },
        validate,
    )
}

fn validate_public_operation_binding(
    descriptor: &VisionStackSampleDescriptorV1,
    activation_strategy: VisionStackActivationStrategy,
    expected_kernel_variant_id: &str,
) -> Result<(), CollectorError> {
    if descriptor.kernel_variant_id != expected_kernel_variant_id
        || descriptor.activation_strategy != activation_strategy_name(activation_strategy)
    {
        return Err(CollectorError::OperationBindingMismatch);
    }
    Ok(())
}

fn native_facts(
    diagnostics: &VisionStackDiagnostics,
) -> Result<NativeVisionStackFactsV1, CollectorError> {
    Ok(NativeVisionStackFactsV1 {
        checked_error_scopes: diagnostics.checked_error_scopes,
        captured_errors: diagnostics.captured_errors.clone(),
        queue_wall_time_ns: Some(diagnostics.queue_wall_time_ns),
        timestamp: diagnostics
            .timestamp
            .map(|timestamp| NativeTimestampFactsV1 {
                begin_ticks: timestamp.begin_ticks,
                end_ticks: timestamp.end_ticks,
                period_ns: timestamp.period_ns,
                reported_duration_ns: timestamp.duration_ns,
                fresh: diagnostics.timestamp_fresh,
            }),
        topology: ExpectedTopologyV1 {
            dispatch_count: u64::try_from(diagnostics.dispatch_count)
                .map_err(|_| CollectorError::InvalidDiagnostics)?,
            compute_pass_count: u64::try_from(diagnostics.compute_pass_count)
                .map_err(|_| CollectorError::InvalidDiagnostics)?,
            command_buffer_count: u64::from(diagnostics.command_buffer_count),
            submission_count: diagnostics.submission_count,
            map_count: u64::from(diagnostics.readback_map_count),
        },
        activation_strategy: diagnostics.activation_strategy,
        activation_buffer_count: u64::try_from(diagnostics.activation_buffer_count)
            .map_err(|_| CollectorError::InvalidDiagnostics)?,
        activation_arena_bytes: diagnostics.activation_arena_bytes,
        scratch_arena_bytes: diagnostics.scratch_arena_bytes,
        main_buffers_bytes: diagnostics.main_buffers_bytes,
    })
}

fn close_failed_timing(clock: &mut impl CollectorClockV1) -> Result<(), CollectorError> {
    clock
        .now_ns()
        .map(|_| ())
        .map_err(|_| CollectorError::ClockFailed)
}

fn validate_environment_observation(
    observation: ObservationV1,
) -> Result<ObservationV1, CollectorError> {
    let valid = match &observation {
        ObservationV1::Available { value, method } => {
            !value.trim().is_empty() && !method.trim().is_empty()
        }
        ObservationV1::Unavailable { reason, method } => {
            !reason.trim().is_empty() && !method.trim().is_empty()
        }
    };
    valid
        .then_some(observation)
        .ok_or(CollectorError::EnvironmentProbeFailed)
}

fn validate_descriptor(descriptor: &VisionStackSampleDescriptorV1) -> Result<(), CollectorError> {
    let topology = &descriptor.expected_topology;
    let valid = descriptor.schedule_slot == descriptor.index
        && !descriptor.kernel_variant_id.trim().is_empty()
        && !descriptor.residency_plan_id.trim().is_empty()
        && is_lower_hex_digest_v1(&descriptor.expected_output_sha256)
        && topology.dispatch_count > 0
        && topology.compute_pass_count > 0
        && topology.command_buffer_count > 0
        && topology.submission_count > 0
        && topology.map_count > 0
        && descriptor.logical_gpu_bytes > 0
        && descriptor.allocated_gpu_bytes >= descriptor.logical_gpu_bytes
        && matches!(
            descriptor.activation_strategy.as_str(),
            "separate_buffers" | "static_arena_no_alias" | "static_arena_alias"
        )
        && descriptor.activation_buffer_count > 0
        && descriptor.activation_arena_bytes > 0
        && descriptor.scratch_arena_bytes > 0
        && descriptor.main_buffers_bytes > 0;
    valid.then_some(()).ok_or(CollectorError::InvalidDescriptor)
}

#[allow(clippy::too_many_arguments)]
fn admit_sample(
    timestamp_query: bool,
    descriptor: &VisionStackSampleDescriptorV1,
    accepted: AcceptedVisionStackValidationV1,
    facts: NativeVisionStackFactsV1,
    api_wall_ns: u64,
    thermal_before: ObservationV1,
    thermal_after: ObservationV1,
) -> Result<BenchmarkSampleV1, CollectorError> {
    if facts.checked_error_scopes != CHECKED_SCOPE_ORDER || !facts.captured_errors.is_empty() {
        return Err(CollectorError::InvalidDiagnostics);
    }
    let queue_wall = match facts.queue_wall_time_ns {
        Some(duration_ns) if duration_ns > 0 && duration_ns <= api_wall_ns => {
            DurationObservationV1::Available { duration_ns }
        }
        None => DurationObservationV1::Unavailable {
            reason: NATIVE_QUEUE_UNAVAILABLE_REASON.to_owned(),
        },
        Some(_) => return Err(CollectorError::InvalidQueueObservation),
    };
    let gpu_timestamp = map_timestamp(timestamp_query, facts.timestamp, api_wall_ns)?;
    if facts.topology != descriptor.expected_topology {
        return Err(CollectorError::TopologyMismatch);
    }
    if activation_strategy_name(facts.activation_strategy) != descriptor.activation_strategy
        || facts.activation_buffer_count != descriptor.activation_buffer_count
        || facts.activation_arena_bytes != descriptor.activation_arena_bytes
        || facts.scratch_arena_bytes != descriptor.scratch_arena_bytes
        || facts.main_buffers_bytes != descriptor.main_buffers_bytes
    {
        return Err(CollectorError::ResourceMismatch);
    }
    if accepted.output_sha256 != descriptor.expected_output_sha256
        || !is_lower_hex_digest_v1(&accepted.output_sha256)
        || !is_lower_hex_digest_v1(&accepted.correctness_report_blake3)
        || !is_lower_hex_digest_v1(&accepted.causal_evidence_blake3)
    {
        return Err(CollectorError::CrossLinkMismatch);
    }

    Ok(BenchmarkSampleV1 {
        index: descriptor.index,
        schedule_slot: descriptor.schedule_slot,
        kernel_variant_id: descriptor.kernel_variant_id.clone(),
        residency_plan_id: descriptor.residency_plan_id.clone(),
        api_wall_ns,
        queue_wall,
        gpu_timestamp,
        topology: facts.topology,
        output_sha256: accepted.output_sha256,
        correctness_report_blake3: accepted.correctness_report_blake3,
        causal_evidence_blake3: accepted.causal_evidence_blake3,
        logical_gpu_bytes: descriptor.logical_gpu_bytes,
        allocated_gpu_bytes: descriptor.allocated_gpu_bytes,
        thermal_before,
        thermal_after,
        status: SampleStatusV1::Passed,
    })
}

fn map_timestamp(
    timestamp_query: bool,
    timestamp: Option<NativeTimestampFactsV1>,
    api_wall_ns: u64,
) -> Result<GpuTimestampObservationV1, CollectorError> {
    if !timestamp_query {
        return match timestamp {
            None => Ok(GpuTimestampObservationV1::Unavailable {
                reason: NATIVE_TIMESTAMP_UNAVAILABLE_REASON.to_owned(),
            }),
            Some(_) => Err(CollectorError::InvalidTimestamp),
        };
    }
    let timestamp = timestamp.ok_or(CollectorError::InvalidTimestamp)?;
    if timestamp.fresh != Some(true)
        || !timestamp.period_ns.is_finite()
        || timestamp.period_ns <= 0.0
    {
        return Err(CollectorError::InvalidTimestamp);
    }
    let period_ns = timestamp.period_ns.to_string();
    let duration_ns =
        exact_gpu_timestamp_duration_ns_v1(timestamp.begin_ticks, timestamp.end_ticks, &period_ns)
            .map_err(|_| CollectorError::InvalidTimestamp)?;
    if duration_ns > api_wall_ns {
        return Err(CollectorError::InvalidTimestamp);
    }
    let _reported_duration_ns_is_not_an_oracle = timestamp.reported_duration_ns;
    Ok(GpuTimestampObservationV1::Available {
        begin_ticks: timestamp.begin_ticks,
        end_ticks: timestamp.end_ticks,
        period_ns,
        duration_ns,
    })
}

const fn activation_strategy_name(strategy: VisionStackActivationStrategy) -> &'static str {
    match strategy {
        VisionStackActivationStrategy::SeparateBuffers => "separate_buffers",
        VisionStackActivationStrategy::StaticArenaNoAlias => "static_arena_no_alias",
        VisionStackActivationStrategy::StaticArenaAlias => "static_arena_alias",
    }
}

#[cfg(test)]
#[allow(dead_code)]
mod cohort_tests;
#[cfg(test)]
mod contract_tests;
