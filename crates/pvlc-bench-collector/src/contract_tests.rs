use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    rc::Rc,
    thread,
    time::{Duration, Instant},
};

use pvlc_bench::{DurationObservationV1, GpuTimestampObservationV1, SampleStatusV1};
use pvlc_ir::SemanticGraph;
use pvlc_passes::{
    build_verified_vision_qkv_stack_overlay, canonical_synthetic_vision_qkv_tensor_catalog,
    select_vision_qkv_stack_overlay,
};
use pvlc_runtime_core::{
    VisionEncoderLayerGeometry, VisionEncoderStackInvocation, VisionLayerNormParameters,
    VisionQkvExecutionPolicy, VisionQkvFusedTargetLimits, VisionQkvSelectionOutcome,
    VisionQkvStackExecutionEvidence, VisionRopeSpecialization,
};
use pvlc_runtime_native::{GpuTimestamp, RuntimeError, VisionStackDiagnostics};

use super::*;

const OUTPUT_SHA256: &str = "3d36c0e0c95be0bc87ddf5298d45b96b83d1756c46a4c24aa493e5eb405ca567";
const CHECKED_SCOPES: [ErrorScopeKind; 3] = [
    ErrorScopeKind::Validation,
    ErrorScopeKind::OutOfMemory,
    ErrorScopeKind::Internal,
];
const CLOSED_TIMED_EVENTS: [&str; 6] = ["probe", "clock", "execute", "validate", "clock", "probe"];

type DescriptorMutation = Box<dyn Fn(&mut VisionStackSampleDescriptorV1)>;
type AttemptMutation = Box<dyn Fn(&mut Attempt)>;
type ValidationMutation = Box<dyn Fn(&mut AcceptedVisionStackValidationV1)>;
type PublicBindingCase = (
    &'static str,
    fn(&mut VisionStackSampleDescriptorV1),
    VisionStackActivationStrategy,
);

fn hash(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn observation(value: &str) -> ObservationV1 {
    ObservationV1::Available {
        value: value.to_owned(),
        method: "ProcessInfo.thermalState".to_owned(),
    }
}

fn topology() -> ExpectedTopologyV1 {
    ExpectedTopologyV1 {
        dispatch_count: 271,
        compute_pass_count: 28,
        command_buffer_count: 1,
        submission_count: 1,
        map_count: 1,
    }
}

fn descriptor() -> VisionStackSampleDescriptorV1 {
    VisionStackSampleDescriptorV1 {
        index: 7,
        schedule_slot: 7,
        kernel_variant_id: "vision-qkv-fused-f32-v1".to_owned(),
        residency_plan_id: "bounded-shard-static-alias-v1".to_owned(),
        expected_topology: topology(),
        expected_output_sha256: OUTPUT_SHA256.to_owned(),
        logical_gpu_bytes: 151_931_712,
        allocated_gpu_bytes: 151_932_736,
        activation_strategy: "static_arena_alias".to_owned(),
        activation_buffer_count: 3,
        activation_arena_bytes: 61_574_656,
        scratch_arena_bytes: 49_815_040,
        main_buffers_bytes: 11_759_616,
    }
}

fn variant_descriptor(index: u32, activation_strategy: &str) -> VisionStackSampleDescriptorV1 {
    VisionStackSampleDescriptorV1 {
        index,
        schedule_slot: index,
        kernel_variant_id: format!("variant-{index}"),
        residency_plan_id: format!("residency-{index}"),
        expected_topology: ExpectedTopologyV1 {
            dispatch_count: u64::from(index) + 300,
            compute_pass_count: u64::from(index) + 30,
            command_buffer_count: 2,
            submission_count: 2,
            map_count: 2,
        },
        expected_output_sha256: if index.is_multiple_of(2) {
            hash('e')
        } else {
            hash('f')
        },
        logical_gpu_bytes: 200_000_000 + u64::from(index),
        allocated_gpu_bytes: 200_004_096 + u64::from(index),
        activation_strategy: activation_strategy.to_owned(),
        activation_buffer_count: u64::from(index) + 4,
        activation_arena_bytes: 70_000_000 + u64::from(index),
        scratch_arena_bytes: 50_000_000 + u64::from(index),
        main_buffers_bytes: 12_000_000 + u64::from(index),
    }
}

fn facts(timestamp_query: bool) -> NativeVisionStackFactsV1 {
    NativeVisionStackFactsV1 {
        checked_error_scopes: CHECKED_SCOPES,
        captured_errors: Vec::new(),
        queue_wall_time_ns: Some(800),
        timestamp: timestamp_query.then_some(NativeTimestampFactsV1 {
            begin_ticks: 10,
            end_ticks: 1_010,
            period_ns: 1.0,
            reported_duration_ns: 1_000.0,
            fresh: Some(true),
        }),
        topology: topology(),
        activation_strategy: VisionStackActivationStrategy::StaticArenaAlias,
        activation_buffer_count: 3,
        activation_arena_bytes: 61_574_656,
        scratch_arena_bytes: 49_815_040,
        main_buffers_bytes: 11_759_616,
    }
}

fn validation_for(descriptor: &VisionStackSampleDescriptorV1) -> AcceptedVisionStackValidationV1 {
    validation_with_hashes(descriptor, 'c', 'd')
}

fn validation_with_hashes(
    descriptor: &VisionStackSampleDescriptorV1,
    correctness: char,
    causal: char,
) -> AcceptedVisionStackValidationV1 {
    AcceptedVisionStackValidationV1 {
        output_sha256: descriptor.expected_output_sha256.clone(),
        correctness_report_blake3: hash(correctness),
        causal_evidence_blake3: hash(causal),
    }
}

fn validation() -> AcceptedVisionStackValidationV1 {
    validation_for(&descriptor())
}

#[derive(Clone)]
struct FakeRuntime {
    timestamp_query: bool,
}

impl super::sealed::Sealed for FakeRuntime {}

impl NativeCollectorRuntimeV1 for FakeRuntime {
    type MonotonicReading = Instant;

    fn timestamp_query(&self) -> bool {
        self.timestamp_query
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
            .ok_or_else(|| "test monotonic clock reversed".to_owned())?;
        u64::try_from(duration.as_nanos()).map_err(|_| "test monotonic clock overflowed".to_owned())
    }
}

struct InstrumentedRuntime {
    timestamp_query: bool,
    events: Rc<RefCell<Vec<&'static str>>>,
    readings: RefCell<VecDeque<Result<u64, String>>>,
}

impl InstrumentedRuntime {
    fn new(events: Rc<RefCell<Vec<&'static str>>>, readings: [u64; 2]) -> Self {
        Self {
            timestamp_query: false,
            events,
            readings: RefCell::new(readings.map(Ok).into()),
        }
    }
}

impl super::sealed::Sealed for InstrumentedRuntime {}

impl NativeCollectorRuntimeV1 for InstrumentedRuntime {
    type MonotonicReading = u64;

    fn timestamp_query(&self) -> bool {
        self.timestamp_query
    }

    fn collector_monotonic_now(&self) -> Result<Self::MonotonicReading, String> {
        self.events.borrow_mut().push("clock");
        self.readings
            .borrow_mut()
            .pop_front()
            .expect("the public collector performs at most two clock reads")
    }

    fn collector_elapsed_ns(
        &self,
        started: &Self::MonotonicReading,
        ended: &Self::MonotonicReading,
    ) -> Result<u64, String> {
        ended
            .checked_sub(*started)
            .filter(|duration| *duration > 0 && *duration != u64::MAX)
            .ok_or_else(|| "test monotonic clock is invalid".to_owned())
    }
}

#[derive(Clone)]
struct FakeExecution {
    facts: NativeVisionStackFactsV1,
    events: Rc<RefCell<Vec<&'static str>>>,
}

impl NativeCollectorExecutionV1 for FakeExecution {
    fn collector_facts(&self) -> Result<NativeVisionStackFactsV1, CollectorError> {
        self.events.borrow_mut().push("facts");
        Ok(self.facts.clone())
    }
}

struct FakeClock {
    events: Rc<RefCell<Vec<&'static str>>>,
    readings: VecDeque<Result<u64, String>>,
}

impl CollectorClockV1 for FakeClock {
    fn now_ns(&mut self) -> Result<u64, String> {
        self.events.borrow_mut().push("clock");
        self.readings
            .pop_front()
            .expect("test clock reading is present")
    }
}

struct Attempt {
    descriptor: VisionStackSampleDescriptorV1,
    runtime: FakeRuntime,
    facts: NativeVisionStackFactsV1,
    validation: AcceptedVisionStackValidationV1,
    clock: VecDeque<Result<u64, String>>,
    probes: VecDeque<Result<ObservationV1, String>>,
    execution_error: Option<String>,
    validation_error: Option<String>,
}

impl Attempt {
    fn valid(timestamp_query: bool) -> Self {
        Self {
            descriptor: descriptor(),
            runtime: FakeRuntime { timestamp_query },
            facts: facts(timestamp_query),
            validation: validation(),
            clock: VecDeque::from([Ok(10_000), Ok(11_000)]),
            probes: VecDeque::from([Ok(observation("nominal")), Ok(observation("fair"))]),
            execution_error: None,
            validation_error: None,
        }
    }

    fn run(self) -> (Result<BenchmarkSampleV1, CollectorError>, Vec<&'static str>) {
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut clock = FakeClock {
            events: Rc::clone(&events),
            readings: self.clock,
        };
        let probe_events = Rc::clone(&events);
        let mut probes = self.probes;
        let probe = move || {
            probe_events.borrow_mut().push("probe");
            probes
                .pop_front()
                .expect("test environment probe is present")
        };
        let execution_events = Rc::clone(&events);
        let facts_events = Rc::clone(&events);
        let execution_error = self.execution_error;
        let facts = self.facts;
        let execute = move |_runtime: &FakeRuntime| {
            execution_events.borrow_mut().push("execute");
            match execution_error {
                Some(error) => Err(error),
                None => Ok(FakeExecution {
                    facts,
                    events: facts_events,
                }),
            }
        };
        let validation_events = Rc::clone(&events);
        let validation_error = self.validation_error;
        let accepted = self.validation;
        let validate = move |_execution: &FakeExecution| {
            validation_events.borrow_mut().push("validate");
            match validation_error {
                Some(error) => Err(error),
                None => Ok(accepted),
            }
        };
        let result = collect_native_sample_with_clock(
            &self.runtime,
            &self.descriptor,
            &mut clock,
            probe,
            execute,
            validate,
        );
        let events = events.borrow().clone();
        (result, events)
    }
}

fn activation_strategy(value: &str) -> VisionStackActivationStrategy {
    match value {
        "separate_buffers" => VisionStackActivationStrategy::SeparateBuffers,
        "static_arena_no_alias" => VisionStackActivationStrategy::StaticArenaNoAlias,
        "static_arena_alias" => VisionStackActivationStrategy::StaticArenaAlias,
        _ => panic!("test descriptor has an accepted activation strategy"),
    }
}

fn real_diagnostics(
    descriptor: &VisionStackSampleDescriptorV1,
    timestamp_query: bool,
    queue_wall_time_ns: u64,
) -> VisionStackDiagnostics {
    VisionStackDiagnostics {
        checked_error_scopes: CHECKED_SCOPES,
        captured_errors: Vec::new(),
        queue_wall_time_ns,
        timestamp: timestamp_query.then_some(GpuTimestamp {
            begin_ticks: 20,
            end_ticks: 24,
            period_ns: 0.5,
            duration_ns: 2.0,
        }),
        timestamp_fresh: timestamp_query.then_some(true),
        shader_blake3: BTreeMap::new(),
        rope_specialization: VisionRopeSpecialization::Identity,
        layer_count: 2,
        checkpoint_layers: vec![0, 1],
        dispatch_count: usize::try_from(descriptor.expected_topology.dispatch_count).unwrap(),
        compute_pass_count: usize::try_from(descriptor.expected_topology.compute_pass_count)
            .unwrap(),
        submission_count: descriptor.expected_topology.submission_count,
        command_buffer_count: u32::try_from(descriptor.expected_topology.command_buffer_count)
            .unwrap(),
        buffer_allocation_count: 9,
        weight_buffer_count: 2,
        activation_strategy: activation_strategy(&descriptor.activation_strategy),
        activation_buffer_count: usize::try_from(descriptor.activation_buffer_count).unwrap(),
        activation_arena_bytes: descriptor.activation_arena_bytes,
        scratch_arena_bytes: descriptor.scratch_arena_bytes,
        main_buffers_bytes: descriptor.main_buffers_bytes,
        scratch_allocations: Vec::new(),
        readback_buffer_count: 1,
        readback_map_count: u32::try_from(descriptor.expected_topology.map_count).unwrap(),
        readback_bytes: 4,
    }
}

fn legacy_execution(
    descriptor: &VisionStackSampleDescriptorV1,
    timestamp_query: bool,
    queue_wall_time_ns: u64,
) -> VisionStackExecution {
    VisionStackExecution {
        checkpoints: BTreeMap::new(),
        output: Vec::new(),
        diagnostics: real_diagnostics(descriptor, timestamp_query, queue_wall_time_ns),
    }
}

fn qkv_execution(
    descriptor: &VisionStackSampleDescriptorV1,
    timestamp_query: bool,
    queue_wall_time_ns: u64,
) -> VisionQkvStackExecution {
    VisionQkvStackExecution {
        checkpoints: BTreeMap::new(),
        output: Vec::new(),
        diagnostics: real_diagnostics(descriptor, timestamp_query, queue_wall_time_ns),
        evidence: VisionQkvStackExecutionEvidence {
            policy: VisionQkvExecutionPolicy::Required,
            outcome: VisionQkvSelectionOutcome::Fused,
            canonical_layer_plan_blake3: Vec::new(),
            pipeline_creations: Vec::new(),
            bind_group_creations: Vec::new(),
            command_encoder_creations: Vec::new(),
            encoded_dispatches: Vec::new(),
            encoded_copies: Vec::new(),
            map_requests: Vec::new(),
            dispatch_count: usize::try_from(descriptor.expected_topology.dispatch_count).unwrap(),
            compute_pass_count: usize::try_from(descriptor.expected_topology.compute_pass_count)
                .unwrap(),
            command_buffer_count: usize::try_from(
                descriptor.expected_topology.command_buffer_count,
            )
            .unwrap(),
            submission_count: usize::try_from(descriptor.expected_topology.submission_count)
                .unwrap(),
            map_count: usize::try_from(descriptor.expected_topology.map_count).unwrap(),
            workspace: None,
            attention_bindings: Vec::new(),
            canaries: Vec::new(),
        },
    }
}

fn run_real_with_clock<T>(
    descriptor: VisionStackSampleDescriptorV1,
    timestamp_query: bool,
    execution: T,
) -> (Result<BenchmarkSampleV1, CollectorError>, Vec<&'static str>)
where
    T: NativeCollectorExecutionV1,
{
    let events = Rc::new(RefCell::new(Vec::new()));
    let runtime = FakeRuntime { timestamp_query };
    let mut clock = FakeClock {
        events: Rc::clone(&events),
        readings: VecDeque::from([Ok(10_000), Ok(11_000)]),
    };
    let probe_events = Rc::clone(&events);
    let mut probes = VecDeque::from([Ok(observation("nominal")), Ok(observation("fair"))]);
    let execute_events = Rc::clone(&events);
    let validate_events = Rc::clone(&events);
    let accepted = validation_for(&descriptor);
    let result = collect_native_sample_with_clock(
        &runtime,
        &descriptor,
        &mut clock,
        move || {
            probe_events.borrow_mut().push("probe");
            probes.pop_front().unwrap()
        },
        move |_runtime| {
            execute_events.borrow_mut().push("execute");
            Ok(execution)
        },
        move |_execution| {
            validate_events.borrow_mut().push("validate");
            Ok(accepted)
        },
    );
    let events = events.borrow().clone();
    (result, events)
}

fn assert_closed_admission(events: &[&'static str]) {
    assert!(
        events == CLOSED_TIMED_EVENTS
            || events == [CLOSED_TIMED_EVENTS.as_slice(), &["facts"]].concat(),
        "admission failure escaped the closed timing boundary: {events:?}"
    );
}

#[test]
fn native_collector_wraps_execution_and_validation_with_exactly_two_clock_reads() {
    let (sample, events) = Attempt::valid(true).run();
    assert_eq!(
        events,
        [CLOSED_TIMED_EVENTS.as_slice(), &["facts"]].concat()
    );
    assert_eq!(
        sample.unwrap(),
        BenchmarkSampleV1 {
            index: 7,
            schedule_slot: 7,
            kernel_variant_id: "vision-qkv-fused-f32-v1".to_owned(),
            residency_plan_id: "bounded-shard-static-alias-v1".to_owned(),
            api_wall_ns: 1_000,
            queue_wall: DurationObservationV1::Available { duration_ns: 800 },
            gpu_timestamp: GpuTimestampObservationV1::Available {
                begin_ticks: 10,
                end_ticks: 1_010,
                period_ns: "1".to_owned(),
                duration_ns: 1_000,
            },
            topology: topology(),
            output_sha256: OUTPUT_SHA256.to_owned(),
            correctness_report_blake3: hash('c'),
            causal_evidence_blake3: hash('d'),
            logical_gpu_bytes: 151_931_712,
            allocated_gpu_bytes: 151_932_736,
            thermal_before: observation("nominal"),
            thermal_after: observation("fair"),
            status: SampleStatusV1::Passed,
        }
    );
}

#[test]
fn unsupported_native_timestamp_and_queue_are_explicit_without_weakening_api_wall() {
    let (sample, events) = Attempt::valid(false).run();
    assert_eq!(
        events,
        [CLOSED_TIMED_EVENTS.as_slice(), &["facts"]].concat()
    );
    assert_eq!(
        sample.unwrap().gpu_timestamp,
        GpuTimestampObservationV1::Unavailable {
            reason: "native runtime timestamp-query feature unavailable".to_owned(),
        }
    );

    let mut unavailable = Attempt::valid(false);
    unavailable.facts.queue_wall_time_ns = None;
    let (sample, events) = unavailable.run();
    assert_eq!(
        events,
        [CLOSED_TIMED_EVENTS.as_slice(), &["facts"]].concat()
    );
    assert_eq!(
        sample.unwrap().queue_wall,
        DurationObservationV1::Unavailable {
            reason: "native queue-wall observation unavailable".to_owned(),
        }
    );
}

#[test]
fn descriptor_validation_precedes_probe_clock_execution_and_validator_effects() {
    let mutations: Vec<DescriptorMutation> = vec![
        Box::new(|value| value.schedule_slot += 1),
        Box::new(|value| value.kernel_variant_id.clear()),
        Box::new(|value| value.kernel_variant_id = "   ".to_owned()),
        Box::new(|value| value.residency_plan_id.clear()),
        Box::new(|value| value.expected_output_sha256 = "bad".to_owned()),
        Box::new(|value| value.expected_output_sha256 = hash('A')),
        Box::new(|value| value.expected_output_sha256 = hash('g')),
        Box::new(|value| value.expected_topology.dispatch_count = 0),
        Box::new(|value| value.expected_topology.compute_pass_count = 0),
        Box::new(|value| value.expected_topology.command_buffer_count = 0),
        Box::new(|value| value.expected_topology.submission_count = 0),
        Box::new(|value| value.expected_topology.map_count = 0),
        Box::new(|value| value.logical_gpu_bytes = 0),
        Box::new(|value| value.allocated_gpu_bytes = value.logical_gpu_bytes - 1),
        Box::new(|value| value.activation_strategy = "unknown".to_owned()),
        Box::new(|value| value.activation_buffer_count = 0),
        Box::new(|value| value.activation_arena_bytes = 0),
        Box::new(|value| value.scratch_arena_bytes = 0),
        Box::new(|value| value.main_buffers_bytes = 0),
    ];
    for mutate in mutations {
        let mut attempt = Attempt::valid(false);
        mutate(&mut attempt.descriptor);
        let (error, events) = attempt.run();
        assert_eq!(
            error.unwrap_err().code(),
            CollectorErrorCodeV1::InvalidDescriptor
        );
        assert!(
            events.is_empty(),
            "invalid descriptor produced effects: {events:?}"
        );
    }
}

#[test]
fn operation_and_validator_failures_close_the_clock_without_probe_after_or_retry() {
    let mut execution = Attempt::valid(false);
    execution.execution_error = Some("device lost".to_owned());
    let (error, events) = execution.run();
    assert_eq!(
        error.unwrap_err().code(),
        CollectorErrorCodeV1::ExecutionFailed
    );
    assert_eq!(events, ["probe", "clock", "execute", "clock"]);

    let mut validation = Attempt::valid(false);
    validation.validation_error = Some("checkpoint mismatch".to_owned());
    let (error, events) = validation.run();
    assert_eq!(
        error.unwrap_err().code(),
        CollectorErrorCodeV1::ValidationFailed
    );
    assert_eq!(events, ["probe", "clock", "execute", "validate", "clock"]);
}

#[test]
fn probe_and_clock_failures_are_stage_exact_and_never_return_a_sample() {
    let mut before_probe = Attempt::valid(false);
    before_probe.probes[0] = Err("thermal unavailable".to_owned());
    let (error, events) = before_probe.run();
    assert_eq!(
        error.unwrap_err().code(),
        CollectorErrorCodeV1::EnvironmentProbeFailed
    );
    assert_eq!(events, ["probe"]);

    let mut start_clock = Attempt::valid(false);
    start_clock.clock[0] = Err("clock unavailable".to_owned());
    let (error, events) = start_clock.run();
    assert_eq!(error.unwrap_err().code(), CollectorErrorCodeV1::ClockFailed);
    assert_eq!(events, ["probe", "clock"]);

    let mut end_clock = Attempt::valid(false);
    end_clock.clock[1] = Err("clock unavailable".to_owned());
    let (error, events) = end_clock.run();
    assert_eq!(error.unwrap_err().code(), CollectorErrorCodeV1::ClockFailed);
    assert_eq!(events, ["probe", "clock", "execute", "validate", "clock"]);

    let mut after_probe = Attempt::valid(false);
    after_probe.probes[1] = Err("thermal unavailable".to_owned());
    let (error, events) = after_probe.run();
    assert_eq!(
        error.unwrap_err().code(),
        CollectorErrorCodeV1::EnvironmentProbeFailed
    );
    assert_eq!(events, CLOSED_TIMED_EVENTS);
}

#[test]
fn non_monotonic_zero_and_saturated_outer_durations_fail_closed() {
    for readings in [
        [Ok(10_000), Ok(9_999)],
        [Ok(10_000), Ok(10_000)],
        [Ok(0), Ok(u64::MAX)],
    ] {
        let mut attempt = Attempt::valid(false);
        attempt.clock = VecDeque::from(readings);
        let (error, events) = attempt.run();
        assert_eq!(
            error.unwrap_err().code(),
            CollectorErrorCodeV1::NonMonotonicClock
        );
        assert_eq!(events, ["probe", "clock", "execute", "validate", "clock"]);
    }
}

#[test]
fn queue_observation_accepts_equality_and_rejects_zero_or_outer_overrun() {
    let mut equal = Attempt::valid(false);
    equal.facts.queue_wall_time_ns = Some(1_000);
    assert_eq!(
        equal.run().0.unwrap().queue_wall,
        DurationObservationV1::Available { duration_ns: 1_000 }
    );

    for duration in [0, 1_001] {
        let mut invalid = Attempt::valid(false);
        invalid.facts.queue_wall_time_ns = Some(duration);
        let (error, events) = invalid.run();
        assert_eq!(
            error.unwrap_err().code(),
            CollectorErrorCodeV1::InvalidQueueObservation
        );
        assert_eq!(
            events,
            [CLOSED_TIMED_EVENTS.as_slice(), &["facts"]].concat()
        );
    }
}

type DiagnosticsMutation = (
    CollectorErrorCodeV1,
    Box<dyn Fn(&mut VisionStackDiagnostics)>,
);

fn diagnostics_mutations() -> Vec<DiagnosticsMutation> {
    vec![
        (
            CollectorErrorCodeV1::InvalidDiagnostics,
            Box::new(|value| value.checked_error_scopes.swap(0, 2)),
        ),
        (
            CollectorErrorCodeV1::InvalidDiagnostics,
            Box::new(|value| value.captured_errors.push("validation".to_owned())),
        ),
        (
            CollectorErrorCodeV1::InvalidQueueObservation,
            Box::new(|value| value.queue_wall_time_ns = 0),
        ),
        (
            CollectorErrorCodeV1::InvalidQueueObservation,
            Box::new(|value| value.queue_wall_time_ns = 1_001),
        ),
        (
            CollectorErrorCodeV1::TopologyMismatch,
            Box::new(|value| value.dispatch_count -= 1),
        ),
        (
            CollectorErrorCodeV1::TopologyMismatch,
            Box::new(|value| value.compute_pass_count -= 1),
        ),
        (
            CollectorErrorCodeV1::TopologyMismatch,
            Box::new(|value| value.command_buffer_count -= 1),
        ),
        (
            CollectorErrorCodeV1::TopologyMismatch,
            Box::new(|value| value.submission_count -= 1),
        ),
        (
            CollectorErrorCodeV1::TopologyMismatch,
            Box::new(|value| value.readback_map_count -= 1),
        ),
        (
            CollectorErrorCodeV1::ResourceMismatch,
            Box::new(|value| {
                value.activation_strategy = VisionStackActivationStrategy::StaticArenaNoAlias;
            }),
        ),
        (
            CollectorErrorCodeV1::ResourceMismatch,
            Box::new(|value| value.activation_buffer_count -= 1),
        ),
        (
            CollectorErrorCodeV1::ResourceMismatch,
            Box::new(|value| value.activation_arena_bytes -= 4),
        ),
        (
            CollectorErrorCodeV1::ResourceMismatch,
            Box::new(|value| value.scratch_arena_bytes -= 4),
        ),
        (
            CollectorErrorCodeV1::ResourceMismatch,
            Box::new(|value| value.main_buffers_bytes -= 4),
        ),
    ]
}

#[test]
fn both_real_native_adapters_map_every_topology_scope_and_resource_field() {
    let descriptor = descriptor();
    let (sample, events) = run_real_with_clock(
        descriptor.clone(),
        false,
        legacy_execution(&descriptor, false, 800),
    );
    assert_eq!(events, CLOSED_TIMED_EVENTS);
    assert_eq!(sample.unwrap().topology, descriptor.expected_topology);

    for (expected, mutate) in diagnostics_mutations() {
        let mut execution = legacy_execution(&descriptor, false, 800);
        mutate(&mut execution.diagnostics);
        let (error, events) = run_real_with_clock(descriptor.clone(), false, execution);
        assert_eq!(error.unwrap_err().code(), expected);
        assert_eq!(events, CLOSED_TIMED_EVENTS);
    }

    for (expected, mutate) in diagnostics_mutations() {
        let mut execution = qkv_execution(&descriptor, false, 800);
        mutate(&mut execution.diagnostics);
        let (error, events) = run_real_with_clock(descriptor.clone(), false, execution);
        assert_eq!(error.unwrap_err().code(), expected);
        assert_eq!(events, CLOSED_TIMED_EVENTS);
    }
}

#[test]
fn native_timestamp_uses_exact_decimal_rational_and_ignores_float_duration_oracle() {
    let mut decimal = Attempt::valid(true);
    decimal.facts.timestamp = Some(NativeTimestampFactsV1 {
        begin_ticks: 10,
        end_ticks: 110,
        period_ns: 0.29,
        reported_duration_ns: f64::NAN,
        fresh: Some(true),
    });
    let (sample, events) = decimal.run();
    assert_eq!(
        events,
        [CLOSED_TIMED_EVENTS.as_slice(), &["facts"]].concat()
    );
    assert_eq!(
        sample.unwrap().gpu_timestamp,
        GpuTimestampObservationV1::Available {
            begin_ticks: 10,
            end_ticks: 110,
            period_ns: "0.29".to_owned(),
            duration_ns: 29,
        }
    );

    let mut fractional = Attempt::valid(true);
    fractional.facts.timestamp = Some(NativeTimestampFactsV1 {
        begin_ticks: 10,
        end_ticks: 14,
        period_ns: 0.5,
        reported_duration_ns: 999.0,
        fresh: Some(true),
    });
    assert_eq!(
        fractional.run().0.unwrap().gpu_timestamp,
        GpuTimestampObservationV1::Available {
            begin_ticks: 10,
            end_ticks: 14,
            period_ns: "0.5".to_owned(),
            duration_ns: 2,
        }
    );
}

#[test]
fn both_real_native_execution_adapters_preserve_raw_timestamp_authority_and_freshness() {
    let descriptor = descriptor();
    let mut legacy = legacy_execution(&descriptor, true, 800);
    legacy.diagnostics.timestamp = Some(GpuTimestamp {
        begin_ticks: 100,
        end_ticks: 200,
        period_ns: 0.29,
        duration_ns: f64::NAN,
    });
    legacy.diagnostics.timestamp_fresh = Some(true);
    let (sample, events) = run_real_with_clock(descriptor.clone(), true, legacy);
    assert_eq!(events, CLOSED_TIMED_EVENTS);
    assert_eq!(
        sample.unwrap().gpu_timestamp,
        GpuTimestampObservationV1::Available {
            begin_ticks: 100,
            end_ticks: 200,
            period_ns: "0.29".to_owned(),
            duration_ns: 29,
        }
    );

    let mut qkv = qkv_execution(&descriptor, true, 800);
    qkv.diagnostics.timestamp = Some(GpuTimestamp {
        begin_ticks: 300,
        end_ticks: 400,
        period_ns: 0.31,
        duration_ns: 30.999_999_999,
    });
    qkv.diagnostics.timestamp_fresh = Some(true);
    let (sample, events) = run_real_with_clock(descriptor.clone(), true, qkv);
    assert_eq!(events, CLOSED_TIMED_EVENTS);
    assert_eq!(
        sample.unwrap().gpu_timestamp,
        GpuTimestampObservationV1::Available {
            begin_ticks: 300,
            end_ticks: 400,
            period_ns: "0.31".to_owned(),
            duration_ns: 31,
        }
    );

    for mutate in [
        |diagnostics: &mut VisionStackDiagnostics| diagnostics.timestamp = None,
        |diagnostics: &mut VisionStackDiagnostics| diagnostics.timestamp_fresh = None,
        |diagnostics: &mut VisionStackDiagnostics| diagnostics.timestamp_fresh = Some(false),
    ] {
        let mut qkv = qkv_execution(&descriptor, true, 800);
        mutate(&mut qkv.diagnostics);
        let (error, events) = run_real_with_clock(descriptor.clone(), true, qkv);
        assert_eq!(
            error.unwrap_err().code(),
            CollectorErrorCodeV1::InvalidTimestamp
        );
        assert_eq!(events, CLOSED_TIMED_EVENTS);
    }
}

#[test]
fn native_timestamp_rejects_capability_freshness_pair_period_product_and_overflow_mutants() {
    let mutations: Vec<AttemptMutation> = vec![
        Box::new(|value| value.facts.timestamp = None),
        Box::new(|value| value.runtime.timestamp_query = false),
        Box::new(|value| value.facts.timestamp.as_mut().unwrap().fresh = None),
        Box::new(|value| value.facts.timestamp.as_mut().unwrap().fresh = Some(false)),
        Box::new(|value| value.facts.timestamp.as_mut().unwrap().begin_ticks = 0),
        Box::new(|value| value.facts.timestamp.as_mut().unwrap().end_ticks = 10),
        Box::new(|value| {
            let timestamp = value.facts.timestamp.as_mut().unwrap();
            timestamp.begin_ticks = 11;
            timestamp.end_ticks = 10;
        }),
        Box::new(|value| value.facts.timestamp.as_mut().unwrap().period_ns = f64::NAN),
        Box::new(|value| value.facts.timestamp.as_mut().unwrap().period_ns = f64::INFINITY),
        Box::new(|value| value.facts.timestamp.as_mut().unwrap().period_ns = -1.0),
        Box::new(|value| value.facts.timestamp.as_mut().unwrap().period_ns = 0.0),
        Box::new(|value| {
            let timestamp = value.facts.timestamp.as_mut().unwrap();
            timestamp.end_ticks = 11;
            timestamp.period_ns = 0.5;
        }),
        Box::new(|value| {
            let timestamp = value.facts.timestamp.as_mut().unwrap();
            timestamp.begin_ticks = 1;
            timestamp.end_ticks = u64::MAX;
            timestamp.period_ns = 2.0;
        }),
        Box::new(|value| value.facts.timestamp.as_mut().unwrap().period_ns = 1e-30),
    ];
    for mutate in mutations {
        let mut attempt = Attempt::valid(true);
        mutate(&mut attempt);
        let (error, events) = attempt.run();
        assert_eq!(
            error.unwrap_err().code(),
            CollectorErrorCodeV1::InvalidTimestamp
        );
        assert_eq!(
            events,
            [CLOSED_TIMED_EVENTS.as_slice(), &["facts"]].concat()
        );
    }
}

#[test]
fn validator_digests_are_lowercase_closed_and_output_is_cross_linked_after_timing() {
    let mutations: Vec<ValidationMutation> = vec![
        Box::new(|value| value.output_sha256 = hash('0')),
        Box::new(|value| value.output_sha256 = hash('A')),
        Box::new(|value| value.output_sha256 = hash('g')),
        Box::new(|value| value.correctness_report_blake3 = "bad".to_owned()),
        Box::new(|value| value.correctness_report_blake3 = hash('A')),
        Box::new(|value| value.correctness_report_blake3 = hash('g')),
        Box::new(|value| value.causal_evidence_blake3 = "bad".to_owned()),
        Box::new(|value| value.causal_evidence_blake3 = hash('A')),
        Box::new(|value| value.causal_evidence_blake3 = hash('g')),
    ];
    for mutate in mutations {
        let mut attempt = Attempt::valid(false);
        mutate(&mut attempt.validation);
        let (error, events) = attempt.run();
        assert_eq!(
            error.unwrap_err().code(),
            CollectorErrorCodeV1::CrossLinkMismatch
        );
        assert_closed_admission(&events);
    }
}

fn public_probes(
    events: Rc<RefCell<Vec<&'static str>>>,
    before: &'static str,
    after: &'static str,
) -> impl FnMut() -> Result<ObservationV1, String> {
    let mut probes = VecDeque::from([Ok(observation(before)), Ok(observation(after))]);
    move || {
        events.borrow_mut().push("probe");
        probes.pop_front().unwrap()
    }
}

#[test]
fn both_public_native_collectors_reject_invalid_descriptors_before_any_clock_or_other_effect() {
    let mut invalid = descriptor();
    invalid.schedule_slot += 1;

    let legacy_events = Rc::new(RefCell::new(Vec::new()));
    let legacy_runtime = InstrumentedRuntime::new(Rc::clone(&legacy_events), [10, 11]);
    let legacy = collect_native_vision_stack_sample(
        &legacy_runtime,
        &invalid,
        public_probes(Rc::clone(&legacy_events), "unused", "unused"),
        |_runtime| -> Result<VisionStackExecution, String> {
            panic!("invalid descriptor reached the legacy operation")
        },
        |_execution| -> Result<AcceptedVisionStackValidationV1, String> {
            panic!("invalid descriptor reached the legacy validator")
        },
    );
    assert_eq!(
        legacy.unwrap_err().code(),
        CollectorErrorCodeV1::InvalidDescriptor
    );
    assert!(legacy_events.borrow().is_empty());

    let qkv_events = Rc::new(RefCell::new(Vec::new()));
    let qkv_runtime = InstrumentedRuntime::new(Rc::clone(&qkv_events), [10, 11]);
    let qkv = collect_native_qkv_vision_stack_sample(
        &qkv_runtime,
        &invalid,
        public_probes(Rc::clone(&qkv_events), "unused", "unused"),
        |_runtime| -> Result<VisionQkvStackExecution, String> {
            panic!("invalid descriptor reached the QKV operation")
        },
        |_execution| -> Result<AcceptedVisionStackValidationV1, String> {
            panic!("invalid descriptor reached the QKV validator")
        },
    );
    assert_eq!(
        qkv.unwrap_err().code(),
        CollectorErrorCodeV1::InvalidDescriptor
    );
    assert!(qkv_events.borrow().is_empty());
}

#[test]
fn both_public_native_collectors_delegate_to_real_execution_adapters_without_fixture_constants() {
    let legacy_descriptor = variant_descriptor(11, "static_arena_no_alias");
    let legacy_execution = legacy_execution(&legacy_descriptor, false, 1);
    let legacy_validation = validation_with_hashes(&legacy_descriptor, '1', '2');
    let legacy_events = Rc::new(RefCell::new(Vec::new()));
    let legacy_runtime = InstrumentedRuntime::new(Rc::clone(&legacy_events), [10_000, 11_000]);
    let execute_events = Rc::clone(&legacy_events);
    let validate_events = Rc::clone(&legacy_events);
    let legacy = collect_native_vision_stack_sample(
        &legacy_runtime,
        &legacy_descriptor,
        public_probes(Rc::clone(&legacy_events), "legacy-before", "legacy-after"),
        move |_runtime| {
            execute_events.borrow_mut().push("execute");
            thread::sleep(Duration::from_millis(2));
            Ok(legacy_execution)
        },
        move |_execution| {
            validate_events.borrow_mut().push("validate");
            thread::sleep(Duration::from_millis(2));
            Ok(legacy_validation)
        },
    )
    .unwrap();
    assert_eq!(
        *legacy_events.borrow(),
        ["probe", "clock", "execute", "validate", "clock", "probe"]
    );
    let legacy_api_wall_ns = legacy.api_wall_ns;
    assert_eq!(legacy_api_wall_ns, 1_000);
    assert_eq!(
        legacy,
        BenchmarkSampleV1 {
            index: legacy_descriptor.index,
            schedule_slot: legacy_descriptor.schedule_slot,
            kernel_variant_id: legacy_descriptor.kernel_variant_id.clone(),
            residency_plan_id: legacy_descriptor.residency_plan_id.clone(),
            api_wall_ns: legacy_api_wall_ns,
            queue_wall: DurationObservationV1::Available { duration_ns: 1 },
            gpu_timestamp: GpuTimestampObservationV1::Unavailable {
                reason: "native runtime timestamp-query feature unavailable".to_owned(),
            },
            topology: legacy_descriptor.expected_topology.clone(),
            output_sha256: legacy_descriptor.expected_output_sha256.clone(),
            correctness_report_blake3: hash('1'),
            causal_evidence_blake3: hash('2'),
            logical_gpu_bytes: legacy_descriptor.logical_gpu_bytes,
            allocated_gpu_bytes: legacy_descriptor.allocated_gpu_bytes,
            thermal_before: observation("legacy-before"),
            thermal_after: observation("legacy-after"),
            status: SampleStatusV1::Passed,
        }
    );

    let qkv_descriptor = variant_descriptor(12, "separate_buffers");
    let qkv_execution = qkv_execution(&qkv_descriptor, false, 1);
    let qkv_validation = validation_with_hashes(&qkv_descriptor, 'a', 'b');
    let qkv_events = Rc::new(RefCell::new(Vec::new()));
    let qkv_runtime = InstrumentedRuntime::new(Rc::clone(&qkv_events), [20_000, 22_000]);
    let execute_events = Rc::clone(&qkv_events);
    let validate_events = Rc::clone(&qkv_events);
    let qkv = collect_native_qkv_vision_stack_sample(
        &qkv_runtime,
        &qkv_descriptor,
        public_probes(Rc::clone(&qkv_events), "qkv-before", "qkv-after"),
        move |_runtime| {
            execute_events.borrow_mut().push("execute");
            thread::sleep(Duration::from_millis(3));
            Ok(qkv_execution)
        },
        move |_execution| {
            validate_events.borrow_mut().push("validate");
            thread::sleep(Duration::from_millis(2));
            Ok(qkv_validation)
        },
    )
    .unwrap();
    assert_eq!(
        *qkv_events.borrow(),
        ["probe", "clock", "execute", "validate", "clock", "probe"]
    );
    let qkv_api_wall_ns = qkv.api_wall_ns;
    assert_eq!(qkv_api_wall_ns, 2_000);
    assert_eq!(
        qkv,
        BenchmarkSampleV1 {
            index: qkv_descriptor.index,
            schedule_slot: qkv_descriptor.schedule_slot,
            kernel_variant_id: qkv_descriptor.kernel_variant_id.clone(),
            residency_plan_id: qkv_descriptor.residency_plan_id.clone(),
            api_wall_ns: qkv_api_wall_ns,
            queue_wall: DurationObservationV1::Available { duration_ns: 1 },
            gpu_timestamp: GpuTimestampObservationV1::Unavailable {
                reason: "native runtime timestamp-query feature unavailable".to_owned(),
            },
            topology: qkv_descriptor.expected_topology.clone(),
            output_sha256: qkv_descriptor.expected_output_sha256.clone(),
            correctness_report_blake3: hash('a'),
            causal_evidence_blake3: hash('b'),
            logical_gpu_bytes: qkv_descriptor.logical_gpu_bytes,
            allocated_gpu_bytes: qkv_descriptor.allocated_gpu_bytes,
            thermal_before: observation("qkv-before"),
            thermal_after: observation("qkv-after"),
            status: SampleStatusV1::Passed,
        }
    );
}

#[test]
fn both_public_native_collectors_propagate_operation_and_validator_failures() {
    let descriptor = descriptor();

    let legacy_events = Rc::new(RefCell::new(Vec::new()));
    let legacy_runtime = InstrumentedRuntime::new(Rc::clone(&legacy_events), [10, 11]);
    let execute_events = Rc::clone(&legacy_events);
    let error = collect_native_vision_stack_sample(
        &legacy_runtime,
        &descriptor,
        public_probes(Rc::clone(&legacy_events), "before", "unused"),
        move |_runtime| {
            execute_events.borrow_mut().push("execute");
            Err("device lost".to_owned())
        },
        |_execution: &VisionStackExecution| Ok(validation()),
    )
    .unwrap_err();
    assert_eq!(error.code(), CollectorErrorCodeV1::ExecutionFailed);
    assert_eq!(
        *legacy_events.borrow(),
        ["probe", "clock", "execute", "clock"]
    );

    let qkv_events = Rc::new(RefCell::new(Vec::new()));
    let qkv_runtime = InstrumentedRuntime::new(Rc::clone(&qkv_events), [20, 21]);
    let execute_events = Rc::clone(&qkv_events);
    let validate_events = Rc::clone(&qkv_events);
    let execution = qkv_execution(&descriptor, false, 1);
    let error = collect_native_qkv_vision_stack_sample(
        &qkv_runtime,
        &descriptor,
        public_probes(Rc::clone(&qkv_events), "before", "unused"),
        move |_runtime| {
            execute_events.borrow_mut().push("execute");
            Ok(execution)
        },
        move |_execution| {
            validate_events.borrow_mut().push("validate");
            Err("checkpoint mismatch".to_owned())
        },
    )
    .unwrap_err();
    assert_eq!(error.code(), CollectorErrorCodeV1::ValidationFailed);
    assert_eq!(
        *qkv_events.borrow(),
        ["probe", "clock", "execute", "validate", "clock"]
    );
}

#[test]
fn admitted_sample_schema_contains_no_summary_claim_self_hash_or_measured_residency() {
    let (sample, _) = Attempt::valid(false).run();
    let value = serde_json::to_value(sample.unwrap()).unwrap();
    let object = value.as_object().unwrap();
    let actual: BTreeSet<_> = object.keys().map(String::as_str).collect();
    let expected = BTreeSet::from([
        "allocated_gpu_bytes",
        "api_wall_ns",
        "causal_evidence_blake3",
        "correctness_report_blake3",
        "gpu_timestamp",
        "index",
        "kernel_variant_id",
        "logical_gpu_bytes",
        "output_sha256",
        "queue_wall",
        "residency_plan_id",
        "schedule_slot",
        "status",
        "thermal_after",
        "thermal_before",
        "topology",
    ]);
    assert_eq!(actual, expected);
    assert!(!object.contains_key("evidence_blake3"));
    let text = serde_json::to_string(&value).unwrap();
    for forbidden in [
        "summary",
        "speedup",
        "winner",
        "throughput",
        "resident_memory",
    ] {
        assert!(
            !text.contains(forbidden),
            "sample leaked forbidden field {forbidden}"
        );
    }
}

#[test]
fn independent_fixture_constants_are_not_runtime_authored() {
    assert_eq!(topology().dispatch_count, 271);
    assert_eq!(topology().submission_count, 1);
    assert_eq!(descriptor().logical_gpu_bytes, 151_931_712);
    assert_eq!(descriptor().allocated_gpu_bytes, 151_932_736);
    assert_eq!(OUTPUT_SHA256.len(), 64);
}

#[test]
fn shared_browser_sample_fixture_is_canonical_and_matches_the_rust_wire_schema() {
    const FIXTURE: &[u8] =
        include_bytes!("../../../web/tests/fixtures/m7d1a2_browser_sample_v1.json");
    assert_eq!(FIXTURE.last(), Some(&b'\n'));
    assert_eq!(FIXTURE.iter().filter(|byte| **byte == b'\n').count(), 1);
    let sample: BenchmarkSampleV1 = serde_json::from_slice(FIXTURE).unwrap();
    assert_eq!(sample.index, 7);
    assert_eq!(sample.schedule_slot, 7);
    assert_eq!(sample.api_wall_ns, 1_000);
    assert_eq!(sample.topology.dispatch_count, 271);
    assert_eq!(sample.topology.submission_count, 28);
    assert_eq!(sample.output_sha256, OUTPUT_SHA256);
    let value: serde_json::Value = serde_json::from_slice(FIXTURE).unwrap();
    let mut canonical = serde_json::to_vec(&value).unwrap();
    canonical.push(b'\n');
    assert_eq!(canonical, FIXTURE);
}

#[test]
fn frozen_native_sample_fixture_is_the_exact_sealed_collector_wire() {
    const FIXTURE: &[u8] =
        include_bytes!("../../../web/tests/fixtures/m7d1a2_native_sample_v1.json");
    assert_eq!(FIXTURE.last(), Some(&b'\n'));
    assert_eq!(FIXTURE.iter().filter(|byte| **byte == b'\n').count(), 1);

    let (sample, events) = Attempt::valid(true).run();
    assert_eq!(
        events,
        [CLOSED_TIMED_EVENTS.as_slice(), &["facts"]].concat()
    );
    let value = serde_json::to_value(sample.unwrap()).unwrap();
    let mut canonical = serde_json::to_vec(&value).unwrap();
    canonical.push(b'\n');
    assert_eq!(canonical, FIXTURE);

    let frozen: BenchmarkSampleV1 = serde_json::from_slice(FIXTURE).unwrap();
    assert_eq!((frozen.index, frozen.schedule_slot), (7, 7));
    assert_eq!(
        frozen.gpu_timestamp,
        GpuTimestampObservationV1::Available {
            begin_ticks: 10,
            end_ticks: 1_010,
            period_ns: "1".to_owned(),
            duration_ns: 1_000,
        }
    );
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PublicOperationCall {
    kind: &'static str,
    invocation_address: usize,
    checkpoint_layers: Vec<usize>,
    activation_strategy: VisionStackActivationStrategy,
    selection_address: Option<usize>,
}

struct PublicOperationRuntime {
    events: Rc<RefCell<Vec<&'static str>>>,
    readings: RefCell<VecDeque<Result<u64, String>>>,
    descriptor: VisionStackSampleDescriptorV1,
    legacy_error: bool,
    qkv_error: bool,
    calls: RefCell<Vec<PublicOperationCall>>,
}

impl PublicOperationRuntime {
    fn new(
        descriptor: VisionStackSampleDescriptorV1,
        events: Rc<RefCell<Vec<&'static str>>>,
    ) -> Self {
        Self {
            events,
            readings: RefCell::new(VecDeque::from([Ok(10_000), Ok(11_000)])),
            descriptor,
            legacy_error: false,
            qkv_error: false,
            calls: RefCell::new(Vec::new()),
        }
    }
}

impl super::sealed::Sealed for PublicOperationRuntime {}

impl NativeCollectorRuntimeV1 for PublicOperationRuntime {
    type MonotonicReading = u64;

    fn timestamp_query(&self) -> bool {
        false
    }

    fn collector_monotonic_now(&self) -> Result<Self::MonotonicReading, String> {
        self.events.borrow_mut().push("clock");
        self.readings
            .borrow_mut()
            .pop_front()
            .expect("public-operation collector performs exactly two clock reads")
    }

    fn collector_elapsed_ns(
        &self,
        started: &Self::MonotonicReading,
        ended: &Self::MonotonicReading,
    ) -> Result<u64, String> {
        ended
            .checked_sub(*started)
            .filter(|duration| *duration > 0 && *duration != u64::MAX)
            .ok_or_else(|| "test public-operation clock is invalid".to_owned())
    }
}

fn fake_public_legacy_operation(
    runtime: &PublicOperationRuntime,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
) -> Result<VisionStackExecution, RuntimeError> {
    runtime.events.borrow_mut().push("public_legacy");
    runtime.calls.borrow_mut().push(PublicOperationCall {
        kind: "legacy",
        invocation_address: std::ptr::from_ref(invocation).addr(),
        checkpoint_layers: checkpoint_layers.to_vec(),
        activation_strategy,
        selection_address: None,
    });
    if runtime.legacy_error {
        Err(RuntimeError::operation(
            "injected exact legacy public-operation failure",
        ))
    } else {
        Ok(legacy_execution(&runtime.descriptor, false, 800))
    }
}

fn fake_public_qkv_operation(
    runtime: &PublicOperationRuntime,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
    selection: &pvlc_passes::VisionQkvStackSelection,
) -> Result<VisionQkvStackExecution, RuntimeError> {
    runtime.events.borrow_mut().push("public_qkv");
    runtime.calls.borrow_mut().push(PublicOperationCall {
        kind: "qkv",
        invocation_address: std::ptr::from_ref(invocation).addr(),
        checkpoint_layers: checkpoint_layers.to_vec(),
        activation_strategy,
        selection_address: Some(std::ptr::from_ref(selection).addr()),
    });
    if runtime.qkv_error {
        Err(RuntimeError::operation(
            "injected exact QKV public-operation failure",
        ))
    } else {
        Ok(qkv_execution(&runtime.descriptor, false, 800))
    }
}

impl NativePublicVisionStackRuntimeV1 for PublicOperationRuntime {
    const LEGACY_PUBLIC_OPERATION_V1: NativeLegacyPublicOperationV1<Self> =
        fake_public_legacy_operation;
    const QKV_PUBLIC_OPERATION_V1: NativeQkvPublicOperationV1<Self> = fake_public_qkv_operation;
}

fn dummy_stack_invocation() -> VisionEncoderStackInvocation<'static> {
    VisionEncoderStackInvocation {
        tokens: 1,
        hidden_size: 1,
        attention_heads: 1,
        head_dim: 1,
        intermediate_size: 1,
        layer_norm_epsilon: 0.000_01,
        input: &[0.25],
        cu_seqlens: &[0, 1],
        layer_parameters: &[],
        post_norm: VisionLayerNormParameters {
            weight: &[],
            bias: &[],
        },
    }
}

fn genuine_fused_selection(
    policy: VisionQkvExecutionPolicy,
) -> pvlc_passes::VisionQkvStackSelection {
    assert!(matches!(
        policy,
        VisionQkvExecutionPolicy::Preferred | VisionQkvExecutionPolicy::Required
    ));
    let geometry = VisionEncoderLayerGeometry {
        tokens: 3,
        hidden_size: 4,
        attention_heads: 2,
        head_dim: 2,
        intermediate_size: 7,
        layer_norm_epsilon: 0.000_01,
        cu_seqlens: &[0, 1, 3],
    }
    .plan()
    .expect("valid compact QKV geometry");
    let catalog =
        canonical_synthetic_vision_qkv_tensor_catalog(1, 4).expect("valid compact QKV catalog");
    let target = VisionQkvFusedTargetLimits {
        min_storage_buffer_offset_alignment: 32,
        max_storage_buffers_per_shader_stage: 8,
        max_storage_buffer_binding_size: 1_u64 << 34,
        max_buffer_size: 1_u64 << 34,
        max_compute_workgroups_per_dimension: 65_535,
    };
    let selection = select_vision_qkv_stack_overlay(policy, || {
        build_verified_vision_qkv_stack_overlay(
            &SemanticGraph::paddleocr_vl_16(),
            1,
            &geometry,
            &catalog,
            target,
        )
    })
    .expect("canonical compact selected policy must fuse");
    assert_eq!(selection.policy(), policy);
    assert_eq!(selection.outcome(), VisionQkvSelectionOutcome::Fused);
    selection
}

fn genuine_required_fused_selection() -> pvlc_passes::VisionQkvStackSelection {
    genuine_fused_selection(VisionQkvExecutionPolicy::Required)
}

fn public_probe(
    events: Rc<RefCell<Vec<&'static str>>>,
) -> impl FnMut() -> Result<ObservationV1, String> {
    let mut values = VecDeque::from([observation("nominal"), observation("nominal")]);
    move || {
        events.borrow_mut().push("probe");
        Ok(values
            .pop_front()
            .expect("public-operation probe is called exactly twice"))
    }
}

#[test]
fn native_public_legacy_wiring_invokes_the_exact_sealed_operation_inside_the_outer_boundary() {
    let mut descriptor = descriptor();
    descriptor.kernel_variant_id = LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1.to_owned();
    assert_eq!(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "vision-stack-legacy-f32-v1"
    );
    let events = Rc::new(RefCell::new(Vec::new()));
    let runtime = PublicOperationRuntime::new(descriptor.clone(), Rc::clone(&events));
    let invocation = dummy_stack_invocation();
    let checkpoints = [0_usize, 2];
    let validate_events = Rc::clone(&events);
    let validation_descriptor = descriptor.clone();

    let sample = collect_native_public_legacy_vision_stack_sample(
        &runtime,
        &descriptor,
        &invocation,
        &checkpoints,
        VisionStackActivationStrategy::StaticArenaAlias,
        public_probe(Rc::clone(&events)),
        move |execution| {
            validate_events.borrow_mut().push("validate");
            assert_eq!(execution.output, Vec::<f32>::new());
            Ok(validation_for(&validation_descriptor))
        },
    )
    .expect("the exact legacy public operation must produce one admitted sample");

    assert_eq!(
        *events.borrow(),
        [
            "probe",
            "clock",
            "public_legacy",
            "validate",
            "clock",
            "probe"
        ]
    );
    assert_eq!(
        sample.kernel_variant_id,
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1
    );
    assert_eq!(sample.api_wall_ns, 1_000);
    assert_eq!(
        *runtime.calls.borrow(),
        [PublicOperationCall {
            kind: "legacy",
            invocation_address: std::ptr::from_ref(&invocation).addr(),
            checkpoint_layers: checkpoints.to_vec(),
            activation_strategy: VisionStackActivationStrategy::StaticArenaAlias,
            selection_address: None,
        }]
    );
}

#[test]
fn concrete_native_runtime_authority_is_function_pointer_bound_to_the_two_exact_public_methods() {
    let actual_legacy =
        <NativeRuntime as NativePublicVisionStackRuntimeV1>::LEGACY_PUBLIC_OPERATION_V1;
    let expected_legacy: NativeLegacyPublicOperationV1<NativeRuntime> =
        NativeRuntime::run_vision_encoder_stack_identity_rope_with_activation_strategy;
    assert!(std::ptr::fn_addr_eq(actual_legacy, expected_legacy));

    let actual_qkv = <NativeRuntime as NativePublicVisionStackRuntimeV1>::QKV_PUBLIC_OPERATION_V1;
    let expected_qkv: NativeQkvPublicOperationV1<NativeRuntime> =
        NativeRuntime::run_vision_encoder_stack_identity_rope_with_qkv_selection;
    assert!(std::ptr::fn_addr_eq(actual_qkv, expected_qkv));

    assert_eq!(
        FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "vision-qkv-fused-f32-v1"
    );
}

#[test]
fn native_public_qkv_wiring_requires_and_invokes_one_exact_required_fused_selection() {
    let descriptor = descriptor();
    assert_eq!(
        descriptor.kernel_variant_id,
        FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1
    );
    let events = Rc::new(RefCell::new(Vec::new()));
    let runtime = PublicOperationRuntime::new(descriptor.clone(), Rc::clone(&events));
    let invocation = dummy_stack_invocation();
    let checkpoints = [1_usize];
    let selection = genuine_required_fused_selection();
    let validate_events = Rc::clone(&events);
    let validation_descriptor = descriptor.clone();

    let sample = collect_native_public_qkv_vision_stack_sample(
        &runtime,
        &descriptor,
        &invocation,
        &checkpoints,
        VisionStackActivationStrategy::StaticArenaAlias,
        &selection,
        public_probe(Rc::clone(&events)),
        move |execution| {
            validate_events.borrow_mut().push("validate");
            assert_eq!(
                execution.evidence.policy,
                VisionQkvExecutionPolicy::Required
            );
            assert_eq!(execution.evidence.outcome, VisionQkvSelectionOutcome::Fused);
            Ok(validation_for(&validation_descriptor))
        },
    )
    .expect("the sealed selection authority marks this test handle Required/Fused");

    assert_eq!(
        *events.borrow(),
        ["probe", "clock", "public_qkv", "validate", "clock", "probe",]
    );
    assert_eq!(
        sample.kernel_variant_id,
        FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1
    );
    assert_eq!(
        *runtime.calls.borrow(),
        [PublicOperationCall {
            kind: "qkv",
            invocation_address: std::ptr::from_ref(&invocation).addr(),
            checkpoint_layers: checkpoints.to_vec(),
            activation_strategy: VisionStackActivationStrategy::StaticArenaAlias,
            selection_address: Some(std::ptr::from_ref(&selection).addr()),
        }]
    );
}

#[test]
fn native_public_wiring_rejects_descriptor_strategy_variant_and_selection_drift_before_gpu_or_clock()
 {
    let invocation = dummy_stack_invocation();
    let checkpoints = [0_usize];

    let legacy_binding_cases: [PublicBindingCase; 3] = [
        (
            "invalid descriptor",
            |descriptor: &mut VisionStackSampleDescriptorV1| descriptor.schedule_slot += 1,
            VisionStackActivationStrategy::StaticArenaAlias,
        ),
        (
            "wrong legacy variant",
            |descriptor: &mut VisionStackSampleDescriptorV1| {
                descriptor.kernel_variant_id =
                    FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1.to_owned();
            },
            VisionStackActivationStrategy::StaticArenaAlias,
        ),
        (
            "wrong activation strategy",
            |_descriptor: &mut VisionStackSampleDescriptorV1| {},
            VisionStackActivationStrategy::StaticArenaNoAlias,
        ),
    ];
    for (label, mutate, strategy) in legacy_binding_cases {
        let mut descriptor = descriptor();
        descriptor.kernel_variant_id = LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1.to_owned();
        mutate(&mut descriptor);
        let events = Rc::new(RefCell::new(Vec::new()));
        let runtime = PublicOperationRuntime::new(descriptor.clone(), Rc::clone(&events));
        let error = collect_native_public_legacy_vision_stack_sample(
            &runtime,
            &descriptor,
            &invocation,
            &checkpoints,
            strategy,
            public_probe(Rc::clone(&events)),
            |_| panic!("{label}: rejected binding reached validation"),
        )
        .expect_err(label);
        assert_eq!(
            error.code(),
            if label == "invalid descriptor" {
                CollectorErrorCodeV1::InvalidDescriptor
            } else {
                CollectorErrorCodeV1::OperationBindingMismatch
            },
            "{label}"
        );
        assert!(
            events.borrow().is_empty(),
            "{label}: effect escaped preflight"
        );
        assert!(
            runtime.calls.borrow().is_empty(),
            "{label}: public operation ran"
        );
    }

    for (label, selection) in [
        (
            "Disabled selection",
            select_vision_qkv_stack_overlay(VisionQkvExecutionPolicy::Disabled, || {
                unreachable!("Disabled selection construction must not request an overlay")
            })
            .unwrap(),
        ),
        (
            "Preferred/Fused selection",
            genuine_fused_selection(VisionQkvExecutionPolicy::Preferred),
        ),
    ] {
        let descriptor = descriptor();
        let events = Rc::new(RefCell::new(Vec::new()));
        let runtime = PublicOperationRuntime::new(descriptor.clone(), Rc::clone(&events));
        let error = collect_native_public_qkv_vision_stack_sample(
            &runtime,
            &descriptor,
            &invocation,
            &checkpoints,
            VisionStackActivationStrategy::StaticArenaAlias,
            &selection,
            public_probe(Rc::clone(&events)),
            |_| panic!("{label} reached validation"),
        )
        .expect_err(label);
        assert_eq!(error.code(), CollectorErrorCodeV1::OperationBindingMismatch);
        assert!(events.borrow().is_empty(), "{label}: effect escaped");
        assert!(runtime.calls.borrow().is_empty(), "{label}: operation ran");
    }
}

#[test]
fn native_public_operation_failure_closes_the_clock_without_validation_or_retry() {
    let mut descriptor = descriptor();
    descriptor.kernel_variant_id = LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1.to_owned();
    let events = Rc::new(RefCell::new(Vec::new()));
    let mut runtime = PublicOperationRuntime::new(descriptor.clone(), Rc::clone(&events));
    runtime.legacy_error = true;
    let invocation = dummy_stack_invocation();

    let error = collect_native_public_legacy_vision_stack_sample(
        &runtime,
        &descriptor,
        &invocation,
        &[0],
        VisionStackActivationStrategy::StaticArenaAlias,
        public_probe(Rc::clone(&events)),
        |_| panic!("failed public operation reached validator"),
    )
    .expect_err("public operation failure must not admit a sample");
    assert_eq!(error.code(), CollectorErrorCodeV1::ExecutionFailed);
    assert_eq!(
        *events.borrow(),
        ["probe", "clock", "public_legacy", "clock"]
    );
    assert_eq!(runtime.calls.borrow().len(), 1);
}

#[test]
fn native_public_qkv_failure_has_no_fallback_retry_and_closes_the_clock_once() {
    let descriptor = descriptor();
    let events = Rc::new(RefCell::new(Vec::new()));
    let mut runtime = PublicOperationRuntime::new(descriptor.clone(), Rc::clone(&events));
    runtime.qkv_error = true;
    let invocation = dummy_stack_invocation();
    let selection = genuine_required_fused_selection();

    let error = collect_native_public_qkv_vision_stack_sample(
        &runtime,
        &descriptor,
        &invocation,
        &[0],
        VisionStackActivationStrategy::StaticArenaAlias,
        &selection,
        public_probe(Rc::clone(&events)),
        |_| panic!("failed QKV public operation reached validator"),
    )
    .expect_err("QKV public operation failure must not fall back or admit a sample");
    assert_eq!(error.code(), CollectorErrorCodeV1::ExecutionFailed);
    assert_eq!(*events.borrow(), ["probe", "clock", "public_qkv", "clock"]);
    assert_eq!(runtime.calls.borrow().len(), 1);
    assert_eq!(runtime.calls.borrow()[0].kind, "qkv");
}

#[test]
fn native_public_qkv_binding_rejects_wrong_variant_and_strategy_before_clock_or_operation() {
    let invocation = dummy_stack_invocation();
    let selection = genuine_required_fused_selection();
    let qkv_binding_cases: [PublicBindingCase; 2] = [
        (
            "wrong QKV variant",
            |descriptor: &mut VisionStackSampleDescriptorV1| {
                descriptor.kernel_variant_id = LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1.to_owned();
            },
            VisionStackActivationStrategy::StaticArenaAlias,
        ),
        (
            "wrong QKV activation strategy",
            |_descriptor: &mut VisionStackSampleDescriptorV1| {},
            VisionStackActivationStrategy::StaticArenaNoAlias,
        ),
    ];
    for (label, mutate, strategy) in qkv_binding_cases {
        let mut descriptor = descriptor();
        mutate(&mut descriptor);
        let events = Rc::new(RefCell::new(Vec::new()));
        let runtime = PublicOperationRuntime::new(descriptor.clone(), Rc::clone(&events));
        let error = collect_native_public_qkv_vision_stack_sample(
            &runtime,
            &descriptor,
            &invocation,
            &[0],
            strategy,
            &selection,
            public_probe(Rc::clone(&events)),
            |_| panic!("{label}: rejected binding reached validation"),
        )
        .expect_err(label);
        assert_eq!(error.code(), CollectorErrorCodeV1::OperationBindingMismatch);
        assert!(
            events.borrow().is_empty(),
            "{label}: effect escaped preflight"
        );
        assert!(runtime.calls.borrow().is_empty(), "{label}: operation ran");
    }
}

#[test]
fn native_public_operation_failure_yields_to_the_required_closing_clock_failure() {
    let mut descriptor = descriptor();
    descriptor.kernel_variant_id = LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1.to_owned();
    let events = Rc::new(RefCell::new(Vec::new()));
    let mut runtime = PublicOperationRuntime::new(descriptor.clone(), Rc::clone(&events));
    runtime.legacy_error = true;
    runtime.readings = RefCell::new(VecDeque::from([
        Ok(10_000),
        Err("injected closing clock failure".to_owned()),
    ]));

    let error = collect_native_public_legacy_vision_stack_sample(
        &runtime,
        &descriptor,
        &dummy_stack_invocation(),
        &[0],
        VisionStackActivationStrategy::StaticArenaAlias,
        public_probe(Rc::clone(&events)),
        |_| panic!("failed operation reached validation"),
    )
    .expect_err("closing clock failure must not be hidden by the operation failure");
    assert_eq!(error.code(), CollectorErrorCodeV1::ClockFailed);
    assert_eq!(
        *events.borrow(),
        ["probe", "clock", "public_legacy", "clock"]
    );
    assert_eq!(runtime.calls.borrow().len(), 1);
}
