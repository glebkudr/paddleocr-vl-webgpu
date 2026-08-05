use std::collections::BTreeMap;

use pvlc_bench::{
    AssembledBenchmarkEvidenceV1, BackendIdentityV1, BackendKindV1, BenchmarkClassV1,
    BenchmarkCohortV1, BenchmarkErrorCodeV1, BenchmarkEvidenceAssemblyInputV1,
    BenchmarkEvidenceInputV1, BenchmarkEvidenceV1, BenchmarkPassportV1, BenchmarkProtocolV1,
    BenchmarkSampleAttemptV1, BenchmarkSampleV1, BenchmarkSummaryV1, ClaimClassV1,
    CorrectnessAnchorV1, DurationObservationV1, ExactRationalV1, ExecutionBoundaryV1,
    ExpectedTopologyV1, GpuTimestampObservationV1, KernelVariantIdentityV1,
    LoadOrCompileObservationV1, ModelIdentityV1, ObservationV1, ResidencyPlanIdentityV1,
    SampleStatusV1, VisionStackWorkloadV1, assemble_browser_benchmark_cohort_v1,
    canonical_benchmark_evidence_assembly_bytes_v1, validate_browser_benchmark_cohort_plan_v1,
};
use serde_json::{Value, json};

type InputMutation = (
    &'static str,
    BenchmarkErrorCodeV1,
    Box<dyn Fn(&mut BenchmarkEvidenceInputV1)>,
);
type SampleMutation = (
    &'static str,
    BenchmarkErrorCodeV1,
    Box<dyn Fn(&mut BenchmarkSampleV1)>,
);
type JsonMutation = Box<dyn Fn(&mut Value)>;
type AssemblyMutation = (
    &'static str,
    Box<dyn Fn(&mut BenchmarkEvidenceAssemblyInputV1)>,
);
type AssemblyJsonMutation = (&'static str, BenchmarkErrorCodeV1, JsonMutation);

trait AmbiguousIfDeserialize<Marker> {
    fn marker() {}
}

impl<T: ?Sized> AmbiguousIfDeserialize<()> for T {}
impl<T: serde::de::DeserializeOwned> AmbiguousIfDeserialize<u8> for T {}

const MEASURED_DURATIONS: [u64; 10] = [
    152_000_152,
    1_000_001,
    185_000_185,
    59_000_059,
    197_000_197,
    82_000_082,
    27_000_027,
    179_000_179,
    69_000_069,
    115_000_115,
];

const EXPECTED_RAW_ORDER_BLAKE3: &str =
    "05110d33cefa9c98cc26ab41f5efb677e7c7f95923a5fbcb4dd4b849c7396772";
const EXPECTED_UNSIGNED_LENGTH: usize = 19_536;
const EXPECTED_EVIDENCE_BLAKE3: &str =
    "3ec11aae5bc109b0efa6c2b495dc07ef6628317f74bcd0e88959a21b6055ba11";
const EXPECTED_SIGNED_LENGTH: usize = 19_621;
const EXPECTED_SIGNED_BLAKE3: &str =
    "2dbda8725319ae7c3575e8c78012f4c712fe6b23e64857ed97391d59165d91cd";
const OFFICIAL_MANIFEST_SHA256: &str =
    "484f982080f3114c285b9db368396859815f768f713ceb960e7fd409f8d6c48b";
const OFFICIAL_CHECKPOINT_SHA256: &str =
    "3d36c0e0c95be0bc87ddf5298d45b96b83d1756c46a4c24aa493e5eb405ca567";
const CANONICAL_SEMANTIC_GRAPH_BLAKE3: &str =
    "2b2556c363545dcef569e3e6d0db01967973a081706c8483e1c5af3c7dc5bf73";
const PUBLIC_OPERATION_ABI_BLAKE3: &str =
    "381449b069ab77f09eaf174cad031d1dfb26cde381b91f5195053af08ae0d2e1";

fn hash(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn available(value: &str, method: &str) -> ObservationV1 {
    ObservationV1::Available {
        value: value.to_owned(),
        method: method.to_owned(),
    }
}

fn topology() -> ExpectedTopologyV1 {
    ExpectedTopologyV1 {
        dispatch_count: 271,
        compute_pass_count: 28,
        command_buffer_count: 28,
        submission_count: 28,
        map_count: 1,
    }
}

fn sample(index: u32, duration_ns: u64, tick_seed: u64) -> BenchmarkSampleV1 {
    BenchmarkSampleV1 {
        index,
        schedule_slot: index,
        kernel_variant_id: "vision-qkv-fused-f32-v1".to_owned(),
        residency_plan_id: "bounded-shard-static-alias-v1".to_owned(),
        api_wall_ns: duration_ns,
        queue_wall: DurationObservationV1::Available {
            duration_ns: duration_ns - 1,
        },
        gpu_timestamp: GpuTimestampObservationV1::Available {
            begin_ticks: tick_seed,
            end_ticks: tick_seed + duration_ns,
            period_ns: "1".to_owned(),
            duration_ns,
        },
        topology: topology(),
        output_sha256: OFFICIAL_CHECKPOINT_SHA256.to_owned(),
        correctness_report_blake3: hash('e'),
        causal_evidence_blake3: hash('f'),
        logical_gpu_bytes: 151_931_712,
        allocated_gpu_bytes: 151_932_736,
        thermal_before: available("nominal", "ProcessInfo.thermalState"),
        thermal_after: available("nominal", "ProcessInfo.thermalState"),
        status: SampleStatusV1::Passed,
    }
}

fn browser_sample(index: u32, duration_ns: u64, tick_seed: u64) -> BenchmarkSampleV1 {
    let mut sample = sample(index, duration_ns, tick_seed);
    sample.gpu_timestamp = GpuTimestampObservationV1::Unavailable {
        reason: "adapter feature unavailable".to_owned(),
    };
    sample
}

fn fixture() -> BenchmarkEvidenceInputV1 {
    let mut limits = BTreeMap::new();
    limits.insert("max_buffer_size".to_owned(), 4_294_967_292);
    limits.insert("max_storage_buffer_binding_size".to_owned(), 4_294_967_292);
    limits.insert("min_storage_buffer_offset_alignment".to_owned(), 256);

    let measured_samples = MEASURED_DURATIONS
        .iter()
        .enumerate()
        .map(|(index, duration)| sample(index as u32, *duration, 10_000_000_000 + index as u64))
        .collect();

    let mut input = BenchmarkEvidenceInputV1 {
        passport: BenchmarkPassportV1 {
            machine: "MacBook Pro".to_owned(),
            soc: "Apple M4 Pro".to_owned(),
            adapter_name: "Apple M4 Pro".to_owned(),
            physical_memory_bytes: 51_539_607_552,
            os_version: "macOS 26.5".to_owned(),
            os_build: "25F90".to_owned(),
            power_source: available("ac", "pmset -g batt"),
            power_profile: available("automatic", "pmset -g custom"),
            low_power_mode: available("false", "pmset -g custom"),
            thermal_state: available("nominal", "ProcessInfo.thermalState"),
            display_attached: available("true", "CGGetOnlineDisplayList"),
            source_tree_blake3: hash('1'),
            compiler_runtime_blake3: hash('2'),
            wgsl_runtime_blake3: hash('3'),
            collector_blake3: hash('4'),
            rustc_version: "rustc 1.92.0".to_owned(),
            cargo_version: "cargo 1.92.0".to_owned(),
            wgpu_version: "30.0.0".to_owned(),
            build_profile: "release".to_owned(),
            backend: BackendIdentityV1 {
                kind: BackendKindV1::ChromeWebgpu,
                browser_version: Some("150.0.0.0".to_owned()),
                user_agent: Some("Chrome/150.0.0.0".to_owned()),
                adapter_backend: "browser_webgpu".to_owned(),
                features: Vec::new(),
                limits,
                timestamp_query: false,
            },
            model: ModelIdentityV1 {
                revision: "PaddleOCR-VL-1.6@pinned".to_owned(),
                model_lock_blake3: hash('5'),
                pack_blake3: hash('6'),
                manifest_sha256: OFFICIAL_MANIFEST_SHA256.to_owned(),
                profile: "ocr-clean-latin-l3".to_owned(),
                case_id: "ocr.clean_latin.0001/vision.stack.27".to_owned(),
                input_blake3: hash('8'),
            },
        },
        workload: VisionStackWorkloadV1 {
            tokens: 1276,
            hidden_size: 1152,
            layer_count: 27,
            checkpoint_policy: "depths-0-1-13-26-final".to_owned(),
            checkpoint_sha256: OFFICIAL_CHECKPOINT_SHA256.to_owned(),
            semantic_graph_blake3: Some(CANONICAL_SEMANTIC_GRAPH_BLAKE3.to_owned()),
            manifest_sha256: OFFICIAL_MANIFEST_SHA256.to_owned(),
            ordered_layer_plans_blake3: (0..27).map(|index| format!("{index:064x}")).collect(),
            qkv_policy: "required".to_owned(),
            qkv_outcome: "fused".to_owned(),
            kernel_variant: KernelVariantIdentityV1 {
                id: "vision-qkv-fused-f32-v1".to_owned(),
                source_set_blake3: hash('a'),
                abi_blake3: hash('b'),
                expected_topology: topology(),
            },
            residency_plan: ResidencyPlanIdentityV1 {
                id: "bounded-shard-static-alias-v1".to_owned(),
                activation_strategy: "static_arena_alias".to_owned(),
                activation_buffer_count: 3,
                activation_arena_bytes: 61_574_656,
                scratch_arena_bytes: 49_815_040,
                main_buffers_bytes: 11_759_616,
                logical_gpu_bytes: 151_931_712,
                allocated_gpu_bytes: 151_932_736,
                max_resident_shard_bytes: 60_958_016,
            },
            readback_policy: "depths-0-1-13-26-final-plus-qkv-canaries".to_owned(),
            execution_boundary: ExecutionBoundaryV1::ApiWall,
        },
        correctness_anchor: CorrectnessAnchorV1 {
            validator_blake3: hash('c'),
            policy_id: "official-l3-existing-envelope-v1".to_owned(),
            expected_checkpoint_sha256: OFFICIAL_CHECKPOINT_SHA256.to_owned(),
            causal_validator_blake3: hash('e'),
        },
        protocol: BenchmarkProtocolV1 {
            class: BenchmarkClassV1::StageMacro,
            build_profile: "release".to_owned(),
            warmup_count: 3,
            measured_count: 10,
            synchronization: "await-complete-map-validate".to_owned(),
            clock_source: "performance-now".to_owned(),
            clock_resolution_ns: 1_000,
            schedule: "single-stable-variant-v1".to_owned(),
            output_validation_policy: "validate-every-sample".to_owned(),
            isolation_policy: "dedicated-process-no-background-load".to_owned(),
            interruption_policy: "reject-any-interruption".to_owned(),
            background_load_policy: "reject-observed-heavy-load".to_owned(),
        },
        cold_sample: sample(0, 200_000_000, 1_000_000_000),
        warmup_samples: vec![
            sample(0, 120_000_000, 2_000_000_000),
            sample(1, 110_000_000, 3_000_000_000),
            sample(2, 105_000_000, 4_000_000_000),
        ],
        measured_samples,
    };
    for sample in std::iter::once(&mut input.cold_sample)
        .chain(input.warmup_samples.iter_mut())
        .chain(input.measured_samples.iter_mut())
    {
        sample.gpu_timestamp = GpuTimestampObservationV1::Unavailable {
            reason: "adapter feature unavailable".to_owned(),
        };
    }
    input
}

fn supported_timestamp_fixture() -> BenchmarkEvidenceInputV1 {
    let mut input = fixture();
    input.passport.backend.kind = BackendKindV1::NativeWgpu;
    input.passport.backend.browser_version = None;
    input.passport.backend.user_agent = None;
    input.passport.backend.adapter_backend = "metal".to_owned();
    input.passport.backend.features = vec!["timestamp_query".to_owned()];
    input.passport.backend.timestamp_query = true;
    input.protocol.clock_source = "std-instant-monotonic".to_owned();
    input.protocol.clock_resolution_ns = 1;
    input
        .workload
        .kernel_variant
        .expected_topology
        .command_buffer_count = 1;
    input
        .workload
        .kernel_variant
        .expected_topology
        .submission_count = 1;
    let expected_topology = input.workload.kernel_variant.expected_topology.clone();
    let mut tick = 1_000_000_000_u64;
    for sample in std::iter::once(&mut input.cold_sample)
        .chain(input.warmup_samples.iter_mut())
        .chain(input.measured_samples.iter_mut())
    {
        sample.topology = expected_topology.clone();
        sample.gpu_timestamp = GpuTimestampObservationV1::Available {
            begin_ticks: tick,
            end_ticks: tick + sample.api_wall_ns,
            period_ns: "1".to_owned(),
            duration_ns: sample.api_wall_ns,
        };
        tick += sample.api_wall_ns + 1_000;
    }
    input
}

fn browser_timestamp_fixture() -> BenchmarkEvidenceInputV1 {
    let mut input = fixture();
    input.passport.backend.features = vec!["timestamp_query".to_owned()];
    input.passport.backend.timestamp_query = true;
    let mut tick = 1_000_000_000_u64;
    for sample in std::iter::once(&mut input.cold_sample)
        .chain(input.warmup_samples.iter_mut())
        .chain(input.measured_samples.iter_mut())
    {
        sample.gpu_timestamp = GpuTimestampObservationV1::Available {
            begin_ticks: tick,
            end_ticks: tick + sample.api_wall_ns,
            period_ns: "1".to_owned(),
            duration_ns: sample.api_wall_ns,
        };
        tick += sample.api_wall_ns + 1_000;
    }
    input
}

fn webkit_fixture() -> BenchmarkEvidenceInputV1 {
    let mut input = fixture();
    input.passport.backend.kind = BackendKindV1::WebkitWebgpu;
    input.passport.backend.browser_version = Some("26.5".to_owned());
    input.passport.backend.user_agent = Some("Version/26.5 Safari/605.1.15".to_owned());
    input.passport.backend.adapter_backend = "browser_webgpu".to_owned();
    input
}

fn observation_json(observation: &ObservationV1) -> Value {
    match observation {
        ObservationV1::Available { value, method } => json!({
            "method": method,
            "status": "available",
            "value": value,
        }),
        ObservationV1::Unavailable { reason, method } => json!({
            "method": method,
            "reason": reason,
            "status": "unavailable",
        }),
    }
}

fn topology_json(value: &ExpectedTopologyV1) -> Value {
    json!({
        "command_buffer_count": value.command_buffer_count,
        "compute_pass_count": value.compute_pass_count,
        "dispatch_count": value.dispatch_count,
        "map_count": value.map_count,
        "submission_count": value.submission_count,
    })
}

fn duration_observation_json(value: &DurationObservationV1) -> Value {
    match value {
        DurationObservationV1::Available { duration_ns } => json!({
            "duration_ns": duration_ns,
            "status": "available",
        }),
        DurationObservationV1::Unavailable { reason } => json!({
            "reason": reason,
            "status": "unavailable",
        }),
    }
}

fn timestamp_observation_json(value: &GpuTimestampObservationV1) -> Value {
    match value {
        GpuTimestampObservationV1::Available {
            begin_ticks,
            end_ticks,
            period_ns,
            duration_ns,
        } => json!({
            "begin_ticks": begin_ticks,
            "duration_ns": duration_ns,
            "end_ticks": end_ticks,
            "period_ns": period_ns,
            "status": "available",
        }),
        GpuTimestampObservationV1::Unavailable { reason } => json!({
            "reason": reason,
            "status": "unavailable",
        }),
    }
}

fn sample_json(value: &BenchmarkSampleV1) -> Value {
    let status = match &value.status {
        SampleStatusV1::Passed => json!({ "status": "passed" }),
        SampleStatusV1::Failed { code } => json!({ "code": code, "status": "failed" }),
    };
    json!({
        "allocated_gpu_bytes": value.allocated_gpu_bytes,
        "api_wall_ns": value.api_wall_ns,
        "causal_evidence_blake3": value.causal_evidence_blake3,
        "correctness_report_blake3": value.correctness_report_blake3,
        "gpu_timestamp": timestamp_observation_json(&value.gpu_timestamp),
        "index": value.index,
        "kernel_variant_id": value.kernel_variant_id,
        "logical_gpu_bytes": value.logical_gpu_bytes,
        "output_sha256": value.output_sha256,
        "queue_wall": duration_observation_json(&value.queue_wall),
        "residency_plan_id": value.residency_plan_id,
        "schedule_slot": value.schedule_slot,
        "status": status,
        "thermal_after": observation_json(&value.thermal_after),
        "thermal_before": observation_json(&value.thermal_before),
        "topology": topology_json(&value.topology),
    })
}

fn rational_json(value: &ExactRationalV1) -> Value {
    json!({
        "denominator": value.denominator,
        "numerator": value.numerator,
    })
}

fn summary_json(value: &BenchmarkSummaryV1) -> Value {
    json!({
        "count": value.count,
        "max_ns": value.max_ns,
        "mean_ns": rational_json(&value.mean_ns),
        "median_absolute_deviation_ns": rational_json(&value.median_absolute_deviation_ns),
        "median_ns": rational_json(&value.median_ns),
        "min_ns": value.min_ns,
        "p90_ns": value.p90_ns,
        "p95_ns": value.p95_ns,
        "raw_order_blake3": value.raw_order_blake3,
    })
}

fn raw_order_hash(durations: impl IntoIterator<Item = u64>) -> String {
    let mut preimage = Vec::new();
    for duration in durations {
        preimage.extend_from_slice(&duration.to_le_bytes());
    }
    blake3::hash(&preimage).to_hex().to_string()
}

fn expected_summary() -> BenchmarkSummaryV1 {
    BenchmarkSummaryV1 {
        count: 10,
        min_ns: 1_000_001,
        max_ns: 197_000_197,
        mean_ns: ExactRationalV1 {
            numerator: "533000533".to_owned(),
            denominator: 5,
        },
        median_ns: ExactRationalV1 {
            numerator: "197000197".to_owned(),
            denominator: 2,
        },
        p90_ns: 185_000_185,
        p95_ns: 197_000_197,
        median_absolute_deviation_ns: ExactRationalV1 {
            numerator: "125000125".to_owned(),
            denominator: 2,
        },
        raw_order_blake3: EXPECTED_RAW_ORDER_BLAKE3.to_owned(),
    }
}

fn reference_unsigned_value(
    input: &BenchmarkEvidenceInputV1,
    summary: &BenchmarkSummaryV1,
) -> Value {
    let backend_kind = match input.passport.backend.kind {
        BackendKindV1::NativeWgpu => "native_wgpu",
        BackendKindV1::ChromeWebgpu => "chrome_webgpu",
        BackendKindV1::WebkitWebgpu => "webkit_webgpu",
    };
    let benchmark_class = match input.protocol.class {
        BenchmarkClassV1::Micro => "micro",
        BenchmarkClassV1::StageMacro => "stage_macro",
    };
    let execution_boundary = match input.workload.execution_boundary {
        ExecutionBoundaryV1::ApiWall => "api_wall",
        ExecutionBoundaryV1::QueueWall => "queue_wall",
        ExecutionBoundaryV1::GpuTimestamp => "gpu_timestamp",
        ExecutionBoundaryV1::LoadOrCompile => "load_or_compile",
    };
    json!({
        "claim_class": "baseline_only",
        "cold_sample": sample_json(&input.cold_sample),
        "correctness_anchor": {
            "causal_validator_blake3": input.correctness_anchor.causal_validator_blake3,
            "expected_checkpoint_sha256": input.correctness_anchor.expected_checkpoint_sha256,
            "policy_id": input.correctness_anchor.policy_id,
            "validator_blake3": input.correctness_anchor.validator_blake3,
        },
        "measured_samples": input.measured_samples.iter().map(sample_json).collect::<Vec<_>>(),
        "passport": {
            "adapter_name": input.passport.adapter_name,
            "backend": {
                "adapter_backend": input.passport.backend.adapter_backend,
                "browser_version": input.passport.backend.browser_version,
                "features": input.passport.backend.features,
                "kind": backend_kind,
                "limits": input.passport.backend.limits,
                "timestamp_query": input.passport.backend.timestamp_query,
                "user_agent": input.passport.backend.user_agent,
            },
            "build_profile": input.passport.build_profile,
            "cargo_version": input.passport.cargo_version,
            "collector_blake3": input.passport.collector_blake3,
            "compiler_runtime_blake3": input.passport.compiler_runtime_blake3,
            "display_attached": observation_json(&input.passport.display_attached),
            "low_power_mode": observation_json(&input.passport.low_power_mode),
            "machine": input.passport.machine,
            "model": {
                "case_id": input.passport.model.case_id,
                "input_blake3": input.passport.model.input_blake3,
                "manifest_sha256": input.passport.model.manifest_sha256,
                "model_lock_blake3": input.passport.model.model_lock_blake3,
                "pack_blake3": input.passport.model.pack_blake3,
                "profile": input.passport.model.profile,
                "revision": input.passport.model.revision,
            },
            "os_build": input.passport.os_build,
            "os_version": input.passport.os_version,
            "physical_memory_bytes": input.passport.physical_memory_bytes,
            "power_profile": observation_json(&input.passport.power_profile),
            "power_source": observation_json(&input.passport.power_source),
            "rustc_version": input.passport.rustc_version,
            "soc": input.passport.soc,
            "source_tree_blake3": input.passport.source_tree_blake3,
            "thermal_state": observation_json(&input.passport.thermal_state),
            "wgpu_version": input.passport.wgpu_version,
            "wgsl_runtime_blake3": input.passport.wgsl_runtime_blake3,
        },
        "protocol": {
            "background_load_policy": input.protocol.background_load_policy,
            "build_profile": input.protocol.build_profile,
            "class": benchmark_class,
            "clock_resolution_ns": input.protocol.clock_resolution_ns,
            "clock_source": input.protocol.clock_source,
            "interruption_policy": input.protocol.interruption_policy,
            "isolation_policy": input.protocol.isolation_policy,
            "measured_count": input.protocol.measured_count,
            "output_validation_policy": input.protocol.output_validation_policy,
            "schedule": input.protocol.schedule,
            "synchronization": input.protocol.synchronization,
            "warmup_count": input.protocol.warmup_count,
        },
        "schema_version": 1,
        "summary": summary_json(summary),
        "warmup_samples": input.warmup_samples.iter().map(sample_json).collect::<Vec<_>>(),
        "workload": {
            "checkpoint_sha256": input.workload.checkpoint_sha256,
            "checkpoint_policy": input.workload.checkpoint_policy,
            "execution_boundary": execution_boundary,
            "hidden_size": input.workload.hidden_size,
            "kernel_variant": {
                "abi_blake3": input.workload.kernel_variant.abi_blake3,
                "expected_topology": topology_json(&input.workload.kernel_variant.expected_topology),
                "id": input.workload.kernel_variant.id,
                "source_set_blake3": input.workload.kernel_variant.source_set_blake3,
            },
            "layer_count": input.workload.layer_count,
            "manifest_sha256": input.workload.manifest_sha256,
            "ordered_layer_plans_blake3": input.workload.ordered_layer_plans_blake3,
            "qkv_outcome": input.workload.qkv_outcome,
            "qkv_policy": input.workload.qkv_policy,
            "readback_policy": input.workload.readback_policy,
            "residency_plan": {
                "activation_arena_bytes": input.workload.residency_plan.activation_arena_bytes,
                "activation_buffer_count": input.workload.residency_plan.activation_buffer_count,
                "activation_strategy": input.workload.residency_plan.activation_strategy,
                "allocated_gpu_bytes": input.workload.residency_plan.allocated_gpu_bytes,
                "id": input.workload.residency_plan.id,
                "logical_gpu_bytes": input.workload.residency_plan.logical_gpu_bytes,
                "main_buffers_bytes": input.workload.residency_plan.main_buffers_bytes,
                "max_resident_shard_bytes": input.workload.residency_plan.max_resident_shard_bytes,
                "scratch_arena_bytes": input.workload.residency_plan.scratch_arena_bytes,
            },
            "semantic_graph_blake3": input.workload.semantic_graph_blake3,
            "tokens": input.workload.tokens,
        },
    })
}

fn canonical_value_bytes(value: &Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(value).unwrap();
    bytes.push(b'\n');
    bytes
}

fn canonical_value_bytes_with_top_level_order(value: &Value, order: &[&str]) -> Vec<u8> {
    let object = value.as_object().unwrap();
    assert_eq!(object.len(), order.len());
    let mut bytes = vec![b'{'];
    for (index, key) in order.iter().enumerate() {
        if index != 0 {
            bytes.push(b',');
        }
        bytes.extend_from_slice(&serde_json::to_vec(key).unwrap());
        bytes.push(b':');
        bytes.extend_from_slice(&serde_json::to_vec(&object[*key]).unwrap());
    }
    bytes.extend_from_slice(b"}\n");
    bytes
}

fn reference_signed_bytes(input: &BenchmarkEvidenceInputV1) -> (Vec<u8>, String, Vec<u8>) {
    let mut value = reference_unsigned_value(input, &expected_summary());
    let unsigned = canonical_value_bytes(&value);
    let evidence_blake3 = blake3::hash(&unsigned).to_hex().to_string();
    value["evidence_blake3"] = Value::String(evidence_blake3.clone());
    (canonical_value_bytes(&value), evidence_blake3, unsigned)
}

fn resign(value: &mut Value) -> Vec<u8> {
    value.as_object_mut().unwrap().remove("evidence_blake3");
    let unsigned = canonical_value_bytes(value);
    value["evidence_blake3"] = Value::String(blake3::hash(&unsigned).to_hex().to_string());
    canonical_value_bytes(value)
}

fn assert_rejected(input: BenchmarkEvidenceInputV1, code: BenchmarkErrorCodeV1) {
    let error = BenchmarkEvidenceV1::build(input).expect_err("hostile input must fail closed");
    assert_eq!(error.code(), code, "unexpected rejection: {error}");
}

fn assert_parse_rejected(bytes: &[u8], code: BenchmarkErrorCodeV1) {
    let error = BenchmarkEvidenceV1::parse_canonical(bytes)
        .expect_err("hostile serialized evidence must fail closed");
    assert_eq!(error.code(), code, "unexpected rejection: {error}");
}

#[test]
fn independent_statistics_oracle_pins_nearest_rank_and_exact_even_rationals() {
    let mut sorted = MEASURED_DURATIONS;
    sorted.sort_unstable();
    let sum = sorted.iter().map(|value| u128::from(*value)).sum::<u128>();
    assert_eq!(sum, 1_066_001_066);
    assert_eq!((sum / 2, 5), (533_000_533, 5));
    assert_eq!((sorted[4] + sorted[5], 2), (197_000_197, 2));
    assert_eq!(sorted[8], 185_000_185);
    assert_eq!(sorted[9], 197_000_197);

    let median_twice = 197_000_197_i128;
    let mut deviations_twice = sorted
        .iter()
        .map(|value| (i128::from(*value) * 2 - median_twice).unsigned_abs())
        .collect::<Vec<_>>();
    deviations_twice.sort_unstable();
    assert_eq!(
        (deviations_twice[4] + deviations_twice[5], 4),
        (250_000_250, 4)
    );
    assert_eq!(
        raw_order_hash(MEASURED_DURATIONS),
        EXPECTED_RAW_ORDER_BLAKE3
    );
}

#[test]
fn validated_evidence_cannot_be_deserialized_around_the_canonical_parser() {
    // If `BenchmarkEvidenceV1: DeserializeOwned`, both marker impls apply and
    // type inference intentionally fails with E0283 at compile time.
    let _ = <BenchmarkEvidenceV1 as AmbiguousIfDeserialize<_>>::marker;
}

#[test]
fn independently_authored_complete_wire_record_has_frozen_preimage_and_identity() {
    let (signed, evidence_blake3, unsigned) = reference_signed_bytes(&fixture());
    assert_eq!(
        raw_order_hash(MEASURED_DURATIONS),
        EXPECTED_RAW_ORDER_BLAKE3
    );
    assert_eq!(unsigned.len(), EXPECTED_UNSIGNED_LENGTH);
    assert_eq!(
        blake3::hash(&unsigned).to_hex().as_str(),
        EXPECTED_EVIDENCE_BLAKE3
    );
    assert_eq!(evidence_blake3, EXPECTED_EVIDENCE_BLAKE3);
    assert_eq!(signed.len(), EXPECTED_SIGNED_LENGTH);
    assert_eq!(
        blake3::hash(&signed).to_hex().as_str(),
        EXPECTED_SIGNED_BLAKE3
    );
    assert_eq!(unsigned.last(), Some(&b'\n'));
    assert_eq!(signed.last(), Some(&b'\n'));
    assert_eq!(unsigned.iter().filter(|byte| **byte == b'\n').count(), 1);
    assert_eq!(signed.iter().filter(|byte| **byte == b'\n').count(), 1);
    let value: Value = serde_json::from_slice(&signed).unwrap();
    assert_eq!(
        value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec![
            "claim_class",
            "cold_sample",
            "correctness_anchor",
            "evidence_blake3",
            "measured_samples",
            "passport",
            "protocol",
            "schema_version",
            "summary",
            "warmup_samples",
            "workload",
        ]
    );
}

#[test]
fn statistics_api_preserves_raw_order_and_handles_fractional_and_high_values_exactly() {
    let summary = pvlc_bench::summarize_api_wall_ns(&MEASURED_DURATIONS).unwrap();
    assert_eq!(summary, expected_summary());

    let mut permutation = MEASURED_DURATIONS;
    permutation.rotate_left(3);
    let permuted = pvlc_bench::summarize_api_wall_ns(&permutation).unwrap();
    assert_eq!(permuted.count, summary.count);
    assert_eq!(permuted.min_ns, summary.min_ns);
    assert_eq!(permuted.max_ns, summary.max_ns);
    assert_eq!(permuted.mean_ns, summary.mean_ns);
    assert_eq!(permuted.median_ns, summary.median_ns);
    assert_eq!(permuted.p90_ns, summary.p90_ns);
    assert_eq!(permuted.p95_ns, summary.p95_ns);
    assert_eq!(
        permuted.median_absolute_deviation_ns,
        summary.median_absolute_deviation_ns
    );
    assert_ne!(permuted.raw_order_blake3, summary.raw_order_blake3);
    assert_eq!(permuted.raw_order_blake3, raw_order_hash(permutation));

    let high = [u64::MAX - 101, u64::MAX - 51, u64::MAX - 2];
    let high_summary = pvlc_bench::summarize_api_wall_ns(&high).unwrap();
    let high_sum = high.iter().map(|value| u128::from(*value)).sum::<u128>();
    assert_eq!(high_summary.mean_ns.numerator, high_sum.to_string());
    assert_eq!(high_summary.mean_ns.denominator, 3);
    assert_eq!(high_summary.median_ns.numerator, high[1].to_string());
    assert_eq!(high_summary.median_ns.denominator, 1);
    assert_eq!(high_summary.p90_ns, high[2]);
    assert_eq!(high_summary.p95_ns, high[2]);

    let near_limit_even = [u64::MAX - 105, u64::MAX - 100, u64::MAX - 3, u64::MAX - 1];
    let even_summary = pvlc_bench::summarize_api_wall_ns(&near_limit_even).unwrap();
    assert_eq!(
        even_summary.median_ns,
        ExactRationalV1 {
            numerator: "36893488147419103127".to_owned(),
            denominator: 2,
        }
    );
    assert_eq!(
        even_summary.median_absolute_deviation_ns,
        ExactRationalV1 {
            numerator: "99".to_owned(),
            denominator: 2,
        }
    );

    let twenty = (1_u64..=20).collect::<Vec<_>>();
    let twenty_summary = pvlc_bench::summarize_api_wall_ns(&twenty).unwrap();
    assert_eq!(twenty_summary.p90_ns, 18);
    assert_eq!(twenty_summary.p95_ns, 19);
    assert_eq!(twenty_summary.max_ns, 20);
}

#[test]
fn statistics_percentile_rank_boundaries_and_invalid_inputs_are_explicit() {
    let cases = [
        (vec![7], "7", 1, 7, 7),
        (vec![9, 1], "5", 1, 9, 9),
        (vec![9, 1, 5], "5", 1, 9, 9),
        (vec![4, 1, 3, 2], "5", 2, 4, 4),
    ];
    for (durations, median_numerator, median_denominator, p90, p95) in cases {
        let summary = pvlc_bench::summarize_api_wall_ns(&durations).unwrap();
        assert_eq!(summary.median_ns.numerator, median_numerator);
        assert_eq!(summary.median_ns.denominator, median_denominator);
        assert_eq!(summary.p90_ns, p90);
        assert_eq!(summary.p95_ns, p95);
        assert_eq!(summary.raw_order_blake3, raw_order_hash(durations));
    }

    for invalid in [Vec::new(), vec![0], vec![u64::MAX]] {
        let error = pvlc_bench::summarize_api_wall_ns(&invalid).unwrap_err();
        assert_eq!(error.code(), BenchmarkErrorCodeV1::InvalidDuration);
    }
}

#[test]
fn build_derives_exact_summary_self_hash_and_canonical_roundtrip() {
    let input = fixture();
    let (expected_bytes, expected_hash, _) = reference_signed_bytes(&input);
    let evidence = BenchmarkEvidenceV1::build(input).expect("valid baseline evidence");
    assert_eq!(evidence.claim_class(), ClaimClassV1::BaselineOnly);
    assert_eq!(evidence.summary(), &expected_summary());

    let bytes = evidence.canonical_bytes();
    assert_eq!(bytes, expected_bytes);
    assert_eq!(bytes.last(), Some(&b'\n'));
    assert_eq!(bytes.iter().filter(|byte| **byte == b'\n').count(), 1);
    assert_eq!(evidence.evidence_blake3(), expected_hash);
    assert_eq!(
        BenchmarkEvidenceV1::parse_canonical(&bytes).unwrap(),
        evidence
    );
}

#[test]
fn canonical_strings_use_json_escaping_without_losing_identity() {
    let mut input = fixture();
    input.passport.model.case_id = "ocr.\"quoted\"\nline/vision.stack.27".to_owned();
    let evidence = BenchmarkEvidenceV1::build(input).unwrap();
    let bytes = evidence.canonical_bytes();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.contains("ocr.\\\"quoted\\\"\\nline/vision.stack.27"));
    assert!(!text.contains("\nline/vision.stack.27"));
    assert_eq!(
        BenchmarkEvidenceV1::parse_canonical(&bytes).unwrap(),
        evidence
    );
}

#[test]
fn cold_and_warmup_observations_never_enter_measured_statistics() {
    let mut input = fixture();
    input.cold_sample.api_wall_ns = 9_000_000_000;
    input.warmup_samples[0].api_wall_ns = 8_000_000_000;
    input.warmup_samples[1].api_wall_ns = 7_000_000_000;
    input.warmup_samples[2].api_wall_ns = 6_000_000_000;
    let expected_cold = sample_json(&input.cold_sample);
    let expected_warmups = input
        .warmup_samples
        .iter()
        .map(sample_json)
        .collect::<Vec<_>>();
    let evidence = BenchmarkEvidenceV1::build(input).unwrap();
    assert_eq!(evidence.summary(), &expected_summary());
    let serialized: Value = serde_json::from_slice(&evidence.canonical_bytes()).unwrap();
    assert_eq!(serialized["cold_sample"], expected_cold);
    assert_eq!(serialized["warmup_samples"], json!(expected_warmups));
}

#[test]
fn identity_environment_and_protocol_validation_fail_closed() {
    let mutations: Vec<InputMutation> = vec![
        (
            "empty machine",
            BenchmarkErrorCodeV1::InvalidIdentity,
            Box::new(|v| v.passport.machine.clear()),
        ),
        (
            "zero memory",
            BenchmarkErrorCodeV1::InvalidIdentity,
            Box::new(|v| v.passport.physical_memory_bytes = 0),
        ),
        (
            "bad content hash",
            BenchmarkErrorCodeV1::InvalidIdentity,
            Box::new(|v| v.passport.source_tree_blake3 = "no".into()),
        ),
        (
            "manifest disagreement",
            BenchmarkErrorCodeV1::CrossLinkMismatch,
            Box::new(|v| v.workload.manifest_sha256 = hash('0')),
        ),
        (
            "unsorted features",
            BenchmarkErrorCodeV1::InvalidIdentity,
            Box::new(|v| v.passport.backend.features = vec!["z".into(), "a".into()]),
        ),
        (
            "duplicate feature",
            BenchmarkErrorCodeV1::InvalidIdentity,
            Box::new(|v| {
                v.passport.backend.features = vec!["shader_f16".into(), "shader_f16".into()]
            }),
        ),
        (
            "profile mismatch",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| v.protocol.build_profile = "debug".into()),
        ),
        (
            "matching debug profile remains ineligible",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| {
                v.passport.build_profile = "debug".into();
                v.protocol.build_profile = "debug".into();
            }),
        ),
        (
            "insufficient warmups",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| {
                v.protocol.warmup_count = 2;
                v.warmup_samples.pop();
            }),
        ),
        (
            "insufficient samples",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| {
                v.protocol.measured_count = 9;
                v.measured_samples.pop();
            }),
        ),
        (
            "count disagreement",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| v.protocol.measured_count = 11),
        ),
        (
            "zero clock resolution",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| v.protocol.clock_resolution_ns = 0),
        ),
        (
            "unknown schedule",
            BenchmarkErrorCodeV1::InvalidSchedule,
            Box::new(|v| v.protocol.schedule = "best-of-one".into()),
        ),
        (
            "unknown synchronization",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| v.protocol.synchronization = "enqueue-only".into()),
        ),
        (
            "wrong browser clock",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| v.protocol.clock_source = "date-now".into()),
        ),
        (
            "unknown validation policy",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| v.protocol.output_validation_policy = "validate-first-only".into()),
        ),
        (
            "unknown isolation policy",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| v.protocol.isolation_policy = "shared-process".into()),
        ),
        (
            "wrong execution boundary",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| v.workload.execution_boundary = ExecutionBoundaryV1::QueueWall),
        ),
        (
            "bad qkv outcome",
            BenchmarkErrorCodeV1::InvalidIdentity,
            Box::new(|v| v.workload.qkv_outcome = "fallback".into()),
        ),
        (
            "wrong plan depth",
            BenchmarkErrorCodeV1::InvalidIdentity,
            Box::new(|v| {
                v.workload.ordered_layer_plans_blake3.pop();
            }),
        ),
        (
            "empty interruption policy",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| v.protocol.interruption_policy.clear()),
        ),
        (
            "dishonest interruption policy",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| v.protocol.interruption_policy = "drop-interrupted-samples".into()),
        ),
        (
            "empty background policy",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| v.protocol.background_load_policy.clear()),
        ),
        (
            "dishonest background policy",
            BenchmarkErrorCodeV1::InvalidProtocol,
            Box::new(|v| v.protocol.background_load_policy = "ignore-heavy-load".into()),
        ),
    ];

    for (name, code, mutate) in mutations {
        let mut input = fixture();
        mutate(&mut input);
        let error = BenchmarkEvidenceV1::build(input).unwrap_err();
        assert_eq!(error.code(), code, "{name}: unexpected rejection: {error}");
    }
}

#[test]
fn every_sample_is_linked_to_workload_correctness_topology_and_resource_plan() {
    let mutations: Vec<SampleMutation> = vec![
        (
            "zero wall",
            BenchmarkErrorCodeV1::InvalidDuration,
            Box::new(|s| s.api_wall_ns = 0),
        ),
        (
            "saturated wall",
            BenchmarkErrorCodeV1::InvalidDuration,
            Box::new(|s| s.api_wall_ns = u64::MAX),
        ),
        (
            "queue exceeds API",
            BenchmarkErrorCodeV1::InvalidQueueObservation,
            Box::new(|s| {
                s.queue_wall = DurationObservationV1::Available {
                    duration_ns: s.api_wall_ns + 1,
                }
            }),
        ),
        (
            "variant drift",
            BenchmarkErrorCodeV1::CrossLinkMismatch,
            Box::new(|s| s.kernel_variant_id = "other".into()),
        ),
        (
            "residency drift",
            BenchmarkErrorCodeV1::CrossLinkMismatch,
            Box::new(|s| s.residency_plan_id = "other".into()),
        ),
        (
            "topology drift",
            BenchmarkErrorCodeV1::TopologyMismatch,
            Box::new(|s| s.topology.dispatch_count -= 1),
        ),
        (
            "output drift",
            BenchmarkErrorCodeV1::CrossLinkMismatch,
            Box::new(|s| s.output_sha256 = hash('0')),
        ),
        (
            "bad correctness hash",
            BenchmarkErrorCodeV1::InvalidIdentity,
            Box::new(|s| s.correctness_report_blake3 = "bad".into()),
        ),
        (
            "bad causal hash",
            BenchmarkErrorCodeV1::InvalidIdentity,
            Box::new(|s| s.causal_evidence_blake3 = "bad".into()),
        ),
        (
            "logical-byte drift",
            BenchmarkErrorCodeV1::ResourceMismatch,
            Box::new(|s| s.logical_gpu_bytes -= 4),
        ),
        (
            "allocated-byte drift",
            BenchmarkErrorCodeV1::ResourceMismatch,
            Box::new(|s| s.allocated_gpu_bytes -= 4),
        ),
        (
            "failed sample",
            BenchmarkErrorCodeV1::FailedSample,
            Box::new(|s| {
                s.status = SampleStatusV1::Failed {
                    code: "device-lost".into(),
                }
            }),
        ),
        (
            "empty thermal method",
            BenchmarkErrorCodeV1::InvalidEnvironment,
            Box::new(|s| s.thermal_after = available("nominal", "")),
        ),
    ];

    for (name, code, mutate) in mutations {
        let mut input = fixture();
        mutate(&mut input.measured_samples[5]);
        let error = BenchmarkEvidenceV1::build(input).unwrap_err();
        assert_eq!(error.code(), code, "{name}: unexpected rejection: {error}");
    }
}

#[test]
fn sample_indices_schedule_slots_and_timestamp_pairs_are_fresh_and_exact() {
    let mut gapped = supported_timestamp_fixture();
    for sample in &mut gapped.measured_samples[4..] {
        sample.index += 1;
        sample.schedule_slot += 1;
    }
    assert_rejected(gapped, BenchmarkErrorCodeV1::InvalidIndex);

    let mut duplicate = supported_timestamp_fixture();
    duplicate.measured_samples[4].index = 3;
    duplicate.measured_samples[4].schedule_slot = 3;
    assert_rejected(duplicate, BenchmarkErrorCodeV1::InvalidIndex);

    let mut slot_drift = supported_timestamp_fixture();
    slot_drift.measured_samples[4].schedule_slot = 7;
    assert_rejected(slot_drift, BenchmarkErrorCodeV1::InvalidSchedule);

    let mut reversed = supported_timestamp_fixture();
    reversed.measured_samples[4].gpu_timestamp = GpuTimestampObservationV1::Available {
        begin_ticks: 99,
        end_ticks: 98,
        period_ns: "1".to_owned(),
        duration_ns: 1,
    };
    assert_rejected(reversed, BenchmarkErrorCodeV1::InvalidTimestamp);

    let mut wrong_duration = supported_timestamp_fixture();
    if let GpuTimestampObservationV1::Available { duration_ns, .. } =
        &mut wrong_duration.measured_samples[4].gpu_timestamp
    {
        *duration_ns -= 1;
    }
    assert_rejected(wrong_duration, BenchmarkErrorCodeV1::InvalidTimestamp);

    let mut reused = supported_timestamp_fixture();
    reused.measured_samples[5].gpu_timestamp = reused.measured_samples[1].gpu_timestamp.clone();
    assert_rejected(reused, BenchmarkErrorCodeV1::StaleTimestamp);

    let mut malformed_period = supported_timestamp_fixture();
    if let GpuTimestampObservationV1::Available { period_ns, .. } =
        &mut malformed_period.measured_samples[4].gpu_timestamp
    {
        *period_ns = "NaN".to_owned();
    }
    assert_rejected(malformed_period, BenchmarkErrorCodeV1::InvalidTimestamp);

    for period in ["0", "-1", "1.0", "+1", "01", "1e0", "0.50"] {
        let mut malformed = supported_timestamp_fixture();
        if let GpuTimestampObservationV1::Available { period_ns, .. } =
            &mut malformed.measured_samples[4].gpu_timestamp
        {
            *period_ns = period.to_owned();
        }
        assert_rejected(malformed, BenchmarkErrorCodeV1::InvalidTimestamp);
    }

    let mut zero_tick = supported_timestamp_fixture();
    if let GpuTimestampObservationV1::Available {
        begin_ticks,
        end_ticks,
        duration_ns,
        ..
    } = &mut zero_tick.measured_samples[4].gpu_timestamp
    {
        *begin_ticks = 0;
        *end_ticks = 1;
        *duration_ns = 1;
    }
    assert_rejected(zero_tick, BenchmarkErrorCodeV1::InvalidTimestamp);

    let mut overflow = supported_timestamp_fixture();
    overflow.measured_samples[4].api_wall_ns = u64::MAX - 1;
    overflow.measured_samples[4].gpu_timestamp = GpuTimestampObservationV1::Available {
        begin_ticks: 1,
        end_ticks: u64::MAX,
        period_ns: "2".to_owned(),
        duration_ns: u64::MAX - 1,
    };
    assert_rejected(overflow, BenchmarkErrorCodeV1::TimestampOverflow);

    let mut fractional = supported_timestamp_fixture();
    let mut tick = 10_u64;
    for sample in std::iter::once(&mut fractional.cold_sample)
        .chain(fractional.warmup_samples.iter_mut())
        .chain(fractional.measured_samples.iter_mut())
    {
        sample.gpu_timestamp = GpuTimestampObservationV1::Available {
            begin_ticks: tick,
            end_ticks: tick + 4,
            period_ns: "0.5".to_owned(),
            duration_ns: 2,
        };
        tick += 10;
    }
    BenchmarkEvidenceV1::build(fractional).expect("exact fractional timestamp periods are valid");

    let mut decimal_exact = supported_timestamp_fixture();
    let mut tick = 1_000_u64;
    for sample in std::iter::once(&mut decimal_exact.cold_sample)
        .chain(decimal_exact.warmup_samples.iter_mut())
        .chain(decimal_exact.measured_samples.iter_mut())
    {
        sample.gpu_timestamp = GpuTimestampObservationV1::Available {
            begin_ticks: tick,
            end_ticks: tick + 100,
            period_ns: "0.29".to_owned(),
            duration_ns: 29,
        };
        tick += 1_000;
    }
    BenchmarkEvidenceV1::build(decimal_exact)
        .expect("decimal timestamp arithmetic must not truncate binary floating point");
}

#[test]
fn timestamp_availability_tracks_the_actual_backend_capability() {
    BenchmarkEvidenceV1::build(browser_timestamp_fixture())
        .expect("timestamp capability validation is not native-only");

    let mut missing = supported_timestamp_fixture();
    missing.measured_samples[0].gpu_timestamp = GpuTimestampObservationV1::Unavailable {
        reason: "not supported".to_owned(),
    };
    assert_rejected(missing, BenchmarkErrorCodeV1::InvalidTimestamp);

    let unsupported = fixture();
    BenchmarkEvidenceV1::build(unsupported).expect("explicit unsupported timestamps are honest");

    let mut dishonest = fixture();
    dishonest.measured_samples[0].gpu_timestamp = GpuTimestampObservationV1::Available {
        begin_ticks: 1,
        end_ticks: 2,
        period_ns: "1".to_owned(),
        duration_ns: 1,
    };
    assert_rejected(dishonest, BenchmarkErrorCodeV1::InvalidTimestamp);

    let mut feature_without_capability = fixture();
    feature_without_capability.passport.backend.features = vec!["timestamp_query".to_owned()];
    assert_rejected(
        feature_without_capability,
        BenchmarkErrorCodeV1::InvalidIdentity,
    );

    let mut capability_without_feature = supported_timestamp_fixture();
    capability_without_feature.passport.backend.features.clear();
    assert_rejected(
        capability_without_feature,
        BenchmarkErrorCodeV1::InvalidIdentity,
    );

    let mut duplicate_feature = browser_timestamp_fixture();
    duplicate_feature
        .passport
        .backend
        .features
        .push("timestamp_query".to_owned());
    assert_rejected(duplicate_feature, BenchmarkErrorCodeV1::InvalidIdentity);

    let mut empty_reason = fixture();
    empty_reason.measured_samples[0].gpu_timestamp = GpuTimestampObservationV1::Unavailable {
        reason: String::new(),
    };
    assert_rejected(empty_reason, BenchmarkErrorCodeV1::InvalidTimestamp);
}

#[test]
fn unavailable_environment_is_baseline_only_but_must_preserve_method_and_reason() {
    let mut input = fixture();
    input.passport.power_profile = ObservationV1::Unavailable {
        reason: "platform API unavailable".to_owned(),
        method: "collector-v1 probe".to_owned(),
    };
    input.passport.thermal_state = ObservationV1::Unavailable {
        reason: "platform API unavailable".to_owned(),
        method: "collector-v1 probe".to_owned(),
    };
    input.measured_samples[0].thermal_before = ObservationV1::Unavailable {
        reason: "per-sample probe unavailable".to_owned(),
        method: "collector-v1 sample probe".to_owned(),
    };
    let evidence = BenchmarkEvidenceV1::build(input).unwrap();
    assert_eq!(evidence.claim_class(), ClaimClassV1::BaselineOnly);
    let value: Value = serde_json::from_slice(&evidence.canonical_bytes()).unwrap();
    assert_eq!(value["passport"]["power_profile"]["status"], "unavailable");
    assert_eq!(
        value["passport"]["power_profile"]["reason"],
        "platform API unavailable"
    );
    assert_eq!(
        value["measured_samples"][0]["thermal_before"]["method"],
        "collector-v1 sample probe"
    );

    let invalid_observations = [
        ObservationV1::Available {
            value: String::new(),
            method: "probe".to_owned(),
        },
        ObservationV1::Available {
            value: "nominal".to_owned(),
            method: String::new(),
        },
        ObservationV1::Unavailable {
            reason: String::new(),
            method: "probe".to_owned(),
        },
        ObservationV1::Unavailable {
            reason: "unavailable".to_owned(),
            method: String::new(),
        },
    ];
    for observation in invalid_observations {
        let mut passport = fixture();
        passport.passport.thermal_state = observation.clone();
        assert_rejected(passport, BenchmarkErrorCodeV1::InvalidEnvironment);

        let mut sample = fixture();
        sample.measured_samples[0].thermal_after = observation;
        assert_rejected(sample, BenchmarkErrorCodeV1::InvalidEnvironment);
    }
}

#[test]
fn queue_unavailability_is_explicit_and_empty_reasons_fail_closed() {
    let mut honest = fixture();
    honest.measured_samples[0].queue_wall = DurationObservationV1::Unavailable {
        reason: "collector channel unavailable".to_owned(),
    };
    let evidence = BenchmarkEvidenceV1::build(honest).unwrap();
    let value: Value = serde_json::from_slice(&evidence.canonical_bytes()).unwrap();
    assert_eq!(
        value["measured_samples"][0]["queue_wall"],
        json!({"reason":"collector channel unavailable","status":"unavailable"})
    );

    let mut empty = fixture();
    empty.measured_samples[0].queue_wall = DurationObservationV1::Unavailable {
        reason: String::new(),
    };
    assert_rejected(empty, BenchmarkErrorCodeV1::InvalidQueueObservation);

    let mut zero = fixture();
    zero.measured_samples[0].queue_wall = DurationObservationV1::Available { duration_ns: 0 };
    assert_rejected(zero, BenchmarkErrorCodeV1::InvalidQueueObservation);
}

#[test]
fn checkpoint_identity_is_cross_linked_in_workload_anchor_and_every_cohort() {
    let mut workload = fixture();
    workload.workload.checkpoint_sha256 = hash('0');
    assert_rejected(workload, BenchmarkErrorCodeV1::CrossLinkMismatch);

    let mut anchor = fixture();
    anchor.correctness_anchor.expected_checkpoint_sha256 = hash('0');
    assert_rejected(anchor, BenchmarkErrorCodeV1::CrossLinkMismatch);

    for cohort in ["cold", "warmup", "measured"] {
        let mut input = fixture();
        match cohort {
            "cold" => input.cold_sample.output_sha256 = hash('0'),
            "warmup" => input.warmup_samples[1].output_sha256 = hash('0'),
            "measured" => input.measured_samples[1].output_sha256 = hash('0'),
            _ => unreachable!(),
        }
        assert_rejected(input, BenchmarkErrorCodeV1::CrossLinkMismatch);
    }
}

#[test]
fn malformed_cold_and_warmup_samples_are_validated_not_merely_preserved() {
    let mut cold = fixture();
    cold.cold_sample.topology.map_count = 0;
    assert_rejected(cold, BenchmarkErrorCodeV1::TopologyMismatch);

    let mut warmup = fixture();
    warmup.warmup_samples[1].logical_gpu_bytes -= 4;
    assert_rejected(warmup, BenchmarkErrorCodeV1::ResourceMismatch);
}

#[test]
fn micro_protocol_requires_ten_warmups_and_thirty_measured_samples() {
    let mut input = fixture();
    input.protocol.class = BenchmarkClassV1::Micro;
    input.protocol.warmup_count = 10;
    input.protocol.measured_count = 30;
    input.warmup_samples = (0..10)
        .map(|index| {
            browser_sample(
                index,
                10_000_000 + u64::from(index),
                20_000_000_000 + u64::from(index),
            )
        })
        .collect();
    input.measured_samples = (0..30)
        .map(|index| {
            browser_sample(
                index,
                20_000_000 + u64::from(index),
                30_000_000_000 + u64::from(index),
            )
        })
        .collect();
    BenchmarkEvidenceV1::build(input.clone()).expect("complete micro protocol");

    input.protocol.measured_count = 29;
    input.measured_samples.pop();
    assert_rejected(input, BenchmarkErrorCodeV1::InvalidProtocol);

    let mut nine_warmups = fixture();
    nine_warmups.protocol.class = BenchmarkClassV1::Micro;
    nine_warmups.protocol.warmup_count = 9;
    nine_warmups.protocol.measured_count = 30;
    nine_warmups.warmup_samples = (0..9)
        .map(|index| {
            browser_sample(
                index,
                10_000_000 + u64::from(index),
                60_000_000_000 + u64::from(index),
            )
        })
        .collect();
    nine_warmups.measured_samples = (0..30)
        .map(|index| {
            browser_sample(
                index,
                20_000_000 + u64::from(index),
                70_000_000_000 + u64::from(index),
            )
        })
        .collect();
    assert_rejected(nine_warmups, BenchmarkErrorCodeV1::InvalidProtocol);

    let mut above_minimum = fixture();
    above_minimum.protocol.warmup_count = 4;
    above_minimum.protocol.measured_count = 11;
    above_minimum
        .warmup_samples
        .push(browser_sample(3, 90_000_009, 40_000_000_003));
    above_minimum
        .measured_samples
        .push(browser_sample(10, 130_000_013, 50_000_000_010));
    BenchmarkEvidenceV1::build(above_minimum).expect("minimum counts are not exact maxima");

    let mut mismatch = fixture();
    mismatch.protocol.warmup_count = 4;
    assert_rejected(mismatch, BenchmarkErrorCodeV1::InvalidProtocol);
}

#[test]
fn canonical_parser_rejects_noncanonical_or_forged_evidence_and_performance_claims() {
    let evidence = BenchmarkEvidenceV1::build(fixture()).unwrap();
    let canonical = evidence.canonical_bytes();

    let value: Value = serde_json::from_slice(&canonical).unwrap();
    let mut pretty = serde_json::to_vec_pretty(&value).unwrap();
    pretty.push(b'\n');
    assert_parse_rejected(&pretty, BenchmarkErrorCodeV1::NonCanonical);

    let mut no_lf = canonical.clone();
    no_lf.pop();
    assert_parse_rejected(&no_lf, BenchmarkErrorCodeV1::NonCanonical);

    let mut two_lf = canonical.clone();
    two_lf.push(b'\n');
    assert_parse_rejected(&two_lf, BenchmarkErrorCodeV1::NonCanonical);

    let reordered = canonical_value_bytes_with_top_level_order(
        &value,
        &[
            "cold_sample",
            "claim_class",
            "correctness_anchor",
            "evidence_blake3",
            "measured_samples",
            "passport",
            "protocol",
            "schema_version",
            "summary",
            "warmup_samples",
            "workload",
        ],
    );
    assert_parse_rejected(&reordered, BenchmarkErrorCodeV1::NonCanonical);

    let exponent = String::from_utf8(canonical.clone()).unwrap().replacen(
        "\"clock_resolution_ns\":1000",
        "\"clock_resolution_ns\":1e3",
        1,
    );
    assert_parse_rejected(exponent.as_bytes(), BenchmarkErrorCodeV1::NonCanonical);

    let mut duplicate = String::from_utf8(canonical.clone()).unwrap();
    duplicate.insert_str(1, "\"claim_class\":\"baseline_only\",");
    assert_parse_rejected(duplicate.as_bytes(), BenchmarkErrorCodeV1::SchemaMismatch);

    let mut value: Value = serde_json::from_slice(&canonical).unwrap();
    value["evidence_blake3"] = Value::String(hash('0'));
    assert_parse_rejected(
        &canonical_value_bytes(&value),
        BenchmarkErrorCodeV1::SelfHashMismatch,
    );

    let mut missing = serde_json::from_slice::<Value>(&canonical).unwrap();
    missing.as_object_mut().unwrap().remove("protocol");
    assert_parse_rejected(&resign(&mut missing), BenchmarkErrorCodeV1::SchemaMismatch);

    let mut missing_nested = serde_json::from_slice::<Value>(&canonical).unwrap();
    missing_nested["protocol"]
        .as_object_mut()
        .unwrap()
        .remove("clock_source");
    assert_parse_rejected(
        &resign(&mut missing_nested),
        BenchmarkErrorCodeV1::SchemaMismatch,
    );

    let duplicate_nested = String::from_utf8(canonical.clone()).unwrap().replacen(
        "\"adapter_backend\":\"browser_webgpu\",",
        "\"adapter_backend\":\"browser_webgpu\",\"adapter_backend\":\"browser_webgpu\",",
        1,
    );
    assert_parse_rejected(
        duplicate_nested.as_bytes(),
        BenchmarkErrorCodeV1::SchemaMismatch,
    );

    let mut unknown_top_level = serde_json::from_slice::<Value>(&canonical).unwrap();
    unknown_top_level["arbitrary_unknown"] = Value::from(1);
    assert_parse_rejected(
        &resign(&mut unknown_top_level),
        BenchmarkErrorCodeV1::SchemaMismatch,
    );

    let nested_unknown_mutations: Vec<JsonMutation> = vec![
        Box::new(|v| v["passport"]["unknown"] = Value::from(1)),
        Box::new(|v| v["passport"]["backend"]["unknown"] = Value::from(1)),
        Box::new(|v| v["workload"]["unknown"] = Value::from(1)),
        Box::new(|v| v["protocol"]["unknown"] = Value::from(1)),
        Box::new(|v| v["passport"]["thermal_state"]["unknown"] = Value::from(1)),
        Box::new(|v| v["summary"]["unknown"] = Value::from(1)),
    ];
    for mutate in nested_unknown_mutations {
        let mut value = serde_json::from_slice::<Value>(&canonical).unwrap();
        mutate(&mut value);
        assert_parse_rejected(&resign(&mut value), BenchmarkErrorCodeV1::SchemaMismatch);
    }

    let mut renamed = serde_json::from_slice::<Value>(&canonical).unwrap();
    let clock = renamed["protocol"]
        .as_object_mut()
        .unwrap()
        .remove("clock_source")
        .unwrap();
    renamed["protocol"]["timer_source"] = clock;
    assert_parse_rejected(&resign(&mut renamed), BenchmarkErrorCodeV1::SchemaMismatch);

    let mut wrong_schema = serde_json::from_slice::<Value>(&canonical).unwrap();
    wrong_schema["schema_version"] = Value::from(2);
    assert_parse_rejected(
        &resign(&mut wrong_schema),
        BenchmarkErrorCodeV1::UnsupportedSchema,
    );

    let mut wrong_claim = serde_json::from_slice::<Value>(&canonical).unwrap();
    wrong_claim["claim_class"] = Value::String("accepted_improvement".to_owned());
    assert_parse_rejected(
        &resign(&mut wrong_claim),
        BenchmarkErrorCodeV1::UnsupportedClaim,
    );

    let mut swapped_records = serde_json::from_slice::<Value>(&canonical).unwrap();
    swapped_records["measured_samples"]
        .as_array_mut()
        .unwrap()
        .swap(0, 1);
    let swapped_durations = swapped_records["measured_samples"]
        .as_array()
        .unwrap()
        .iter()
        .map(|sample| sample["api_wall_ns"].as_u64().unwrap())
        .collect::<Vec<_>>();
    swapped_records["summary"]["raw_order_blake3"] =
        Value::String(raw_order_hash(swapped_durations));
    assert_parse_rejected(
        &resign(&mut swapped_records),
        BenchmarkErrorCodeV1::InvalidIndex,
    );

    for forbidden in [
        "arbitrary_unknown",
        "speedup_pct",
        "winner",
        "resident_memory_bytes",
    ] {
        let mut value: Value = serde_json::from_slice(&canonical).unwrap();
        value["measured_samples"][0][forbidden] = Value::from(1);
        assert_parse_rejected(&resign(&mut value), BenchmarkErrorCodeV1::SchemaMismatch);
    }

    let mut value: Value = serde_json::from_slice(&canonical).unwrap();
    value["measured_samples"][0]["api_wall_ns"] = Value::from(1.5);
    assert_parse_rejected(&resign(&mut value), BenchmarkErrorCodeV1::InvalidInteger);

    let mut value: Value = serde_json::from_slice(&canonical).unwrap();
    value["measured_samples"][0]["api_wall_ns"] = Value::from(-1);
    assert_parse_rejected(&resign(&mut value), BenchmarkErrorCodeV1::InvalidInteger);
}

#[test]
fn every_authored_summary_component_is_recomputed_after_a_valid_self_resign() {
    let canonical = BenchmarkEvidenceV1::build(fixture())
        .unwrap()
        .canonical_bytes();
    let mutations: Vec<JsonMutation> = vec![
        Box::new(|v| v["summary"]["count"] = Value::from(9)),
        Box::new(|v| v["summary"]["min_ns"] = Value::from(2)),
        Box::new(|v| v["summary"]["max_ns"] = Value::from(2)),
        Box::new(|v| v["summary"]["mean_ns"]["numerator"] = Value::String("1".into())),
        Box::new(|v| v["summary"]["mean_ns"]["denominator"] = Value::from(1)),
        Box::new(|v| v["summary"]["median_ns"]["numerator"] = Value::String("1".into())),
        Box::new(|v| v["summary"]["median_ns"]["denominator"] = Value::from(1)),
        Box::new(|v| v["summary"]["p90_ns"] = Value::from(1)),
        Box::new(|v| v["summary"]["p95_ns"] = Value::from(1)),
        Box::new(|v| {
            v["summary"]["median_absolute_deviation_ns"]["numerator"] = Value::String("1".into())
        }),
        Box::new(|v| v["summary"]["median_absolute_deviation_ns"]["denominator"] = Value::from(1)),
        Box::new(|v| v["summary"]["raw_order_blake3"] = Value::String(hash('0'))),
    ];
    for mutate in mutations {
        let mut value: Value = serde_json::from_slice(&canonical).unwrap();
        mutate(&mut value);
        let resigned = resign(&mut value);
        assert_parse_rejected(&resigned, BenchmarkErrorCodeV1::SummaryMismatch);
    }
}

#[test]
fn serialized_resource_accounting_is_explicitly_not_measured_residency() {
    let evidence = BenchmarkEvidenceV1::build(fixture()).unwrap();
    let value: Value = serde_json::from_slice(&evidence.canonical_bytes()).unwrap();
    let text = serde_json::to_string(&value).unwrap();
    assert!(text.contains("logical_gpu_bytes"));
    assert!(text.contains("allocated_gpu_bytes"));
    assert!(!text.contains("resident_memory"));
    assert!(!text.contains("speedup"));
    assert!(!text.contains("throughput"));
    assert!(!text.contains("winner"));
}

#[test]
fn browser_identity_requires_version_and_user_agent_while_native_forbids_them() {
    let chrome = fixture();
    BenchmarkEvidenceV1::build(chrome.clone()).expect("closed Chrome identity");

    let mut missing_user_agent = chrome.clone();
    missing_user_agent.passport.backend.user_agent = None;
    assert_rejected(missing_user_agent, BenchmarkErrorCodeV1::InvalidIdentity);

    let mut missing_version = chrome;
    missing_version.passport.backend.browser_version = None;
    assert_rejected(missing_version, BenchmarkErrorCodeV1::InvalidIdentity);

    BenchmarkEvidenceV1::build(webkit_fixture()).expect("closed WebKit identity");

    let native = supported_timestamp_fixture();
    BenchmarkEvidenceV1::build(native.clone()).expect("closed native identity");

    let mut native_version = native.clone();
    native_version.passport.backend.browser_version = Some("not-applicable".to_owned());
    assert_rejected(native_version, BenchmarkErrorCodeV1::InvalidIdentity);

    let mut native_user_agent = native;
    native_user_agent.passport.backend.user_agent = Some("not-applicable".to_owned());
    assert_rejected(native_user_agent, BenchmarkErrorCodeV1::InvalidIdentity);
}

fn falsely_fused_legacy_fixture() -> BenchmarkEvidenceInputV1 {
    let mut legacy = fixture();
    legacy.workload.qkv_policy = "disabled".to_owned();
    legacy.workload.qkv_outcome = "disabled".to_owned();
    legacy.workload.kernel_variant.id = "vision-stack-legacy-f32-v1".to_owned();
    legacy.workload.kernel_variant.source_set_blake3 = hash('0');
    legacy
        .workload
        .kernel_variant
        .expected_topology
        .dispatch_count = 325;
    let variant_id = legacy.workload.kernel_variant.id.clone();
    let topology = legacy.workload.kernel_variant.expected_topology.clone();
    for sample in std::iter::once(&mut legacy.cold_sample)
        .chain(legacy.warmup_samples.iter_mut())
        .chain(legacy.measured_samples.iter_mut())
    {
        sample.kernel_variant_id.clone_from(&variant_id);
        sample.topology = topology.clone();
    }
    legacy
}

fn set_actual_legacy_compiler_identity(value: &mut Value) {
    value["workload"]["semantic_graph_blake3"] = Value::Null;
    value["workload"]["ordered_layer_plans_blake3"] = Value::Array(Vec::new());
    value["load_or_compile"]["workload_blake3"] =
        Value::String(component_blake3(&value["workload"]));
}

fn actual_legacy_browser_plan_value() -> Value {
    let mut value = browser_cohort_plan_value(&assembly_fixture(falsely_fused_legacy_fixture()));
    set_actual_legacy_compiler_identity(&mut value);
    value
}

fn actual_legacy_browser_assembly_bytes() -> Vec<u8> {
    let mut value =
        browser_cohort_assembly_value(&assembly_fixture(falsely_fused_legacy_fixture()));
    set_actual_legacy_compiler_identity(&mut value);
    resign_assembly(&mut value)
}

#[test]
fn legacy_unfused_qkv_is_a_first_class_baseline_not_only_fused_required() {
    let assembled =
        AssembledBenchmarkEvidenceV1::parse_canonical(&actual_legacy_browser_assembly_bytes())
            .expect("legacy baseline with actual null/empty compiler identity is required");
    assert_eq!(
        assembled.evidence().claim_class(),
        ClaimClassV1::BaselineOnly,
    );
}

#[test]
fn disabled_workload_accepts_only_the_compiler_observed_absent_semantic_graph_and_empty_planir() {
    let accepted = actual_legacy_browser_assembly_bytes();
    let parsed = AssembledBenchmarkEvidenceV1::parse_canonical(&accepted)
        .expect("Disabled must serialize the compiler's actual null/empty identity evidence");
    let evidence: Value = serde_json::from_slice(&parsed.evidence().canonical_bytes()).unwrap();
    assert_eq!(evidence["workload"]["semantic_graph_blake3"], Value::Null);
    assert_eq!(
        evidence["workload"]["ordered_layer_plans_blake3"],
        Value::Array(Vec::new())
    );

    let mut required_null_semantic = browser_cohort_assembly_value(&assembly_fixture(fixture()));
    required_null_semantic["workload"]["semantic_graph_blake3"] = Value::Null;
    assert_eq!(
        required_null_semantic["workload"]["ordered_layer_plans_blake3"]
            .as_array()
            .unwrap()
            .len(),
        27,
        "the Required semantic-null mutant must retain valid full-cardinality PlanIR"
    );
    required_null_semantic["load_or_compile"]["workload_blake3"] =
        Value::String(component_blake3(&required_null_semantic["workload"]));
    let required_null_semantic = resign_assembly(&mut required_null_semantic);
    assert_assembly_parse_rejected(
        &required_null_semantic,
        BenchmarkErrorCodeV1::InvalidIdentity,
    );

    let mut required_empty_planir = browser_cohort_assembly_value(&assembly_fixture(fixture()));
    required_empty_planir["workload"]["ordered_layer_plans_blake3"] = Value::Array(Vec::new());
    assert_eq!(
        required_empty_planir["workload"]["semantic_graph_blake3"], CANONICAL_SEMANTIC_GRAPH_BLAKE3,
        "the Required empty-PlanIR mutant must retain a valid semantic identity"
    );
    required_empty_planir["load_or_compile"]["workload_blake3"] =
        Value::String(component_blake3(&required_empty_planir["workload"]));
    let required_empty_planir = resign_assembly(&mut required_empty_planir);
    assert_assembly_parse_rejected(
        &required_empty_planir,
        BenchmarkErrorCodeV1::InvalidIdentity,
    );

    let mut falsely_fused =
        browser_cohort_assembly_value(&assembly_fixture(falsely_fused_legacy_fixture()));
    assert_ne!(
        falsely_fused["workload"]["semantic_graph_blake3"],
        Value::Null,
        "the falsely fused legacy mutant must retain a semantic identity",
    );
    assert_eq!(
        falsely_fused["workload"]["ordered_layer_plans_blake3"]
            .as_array()
            .unwrap()
            .len(),
        27,
        "the falsely fused legacy mutant must retain full PlanIR",
    );
    falsely_fused["load_or_compile"]["workload_blake3"] =
        Value::String(component_blake3(&falsely_fused["workload"]));
    let falsely_fused = resign_assembly(&mut falsely_fused);
    assert_assembly_parse_rejected(&falsely_fused, BenchmarkErrorCodeV1::InvalidIdentity);
}

#[test]
fn disabled_semantic_identity_is_required_even_when_its_value_is_nullable() {
    let mut missing_semantic = actual_legacy_browser_plan_value();
    assert_eq!(
        missing_semantic["workload"]["semantic_graph_blake3"],
        Value::Null,
        "the missing-field mutant must start from an explicit semantic null",
    );
    let removed = missing_semantic["workload"]
        .as_object_mut()
        .unwrap()
        .remove("semantic_graph_blake3");
    assert_eq!(removed, Some(Value::Null));
    missing_semantic["load_or_compile"]["workload_blake3"] =
        Value::String(component_blake3(&missing_semantic["workload"]));
    assert_browser_plan_rejected(
        &canonical_value_bytes(&missing_semantic),
        BenchmarkErrorCodeV1::SchemaMismatch,
    );
}

const EXPECTED_ASSEMBLY_CANONICAL_LENGTH: usize = 20_906;
const EXPECTED_ASSEMBLY_BLAKE3: &str =
    "d8dd3f6ccd6267a398085bd6ced2af148f2e7d88d62a12937b3b3d6c6e400bfb";
const EXPECTED_ASSEMBLY_CANONICAL_BLAKE3: &str =
    "dd6e8d8779d10ab34cfefe4dfaa6b07507ab160297ee76d5a1d97b1cb9b319e8";

fn component_blake3(value: &Value) -> String {
    blake3::hash(&canonical_value_bytes(value))
        .to_hex()
        .to_string()
}

fn cohort_json(cohort: BenchmarkCohortV1) -> &'static str {
    match cohort {
        BenchmarkCohortV1::Cold => "cold",
        BenchmarkCohortV1::Warmup => "warmup",
        BenchmarkCohortV1::Measured => "measured",
    }
}

fn sample_status_json(status: &SampleStatusV1) -> Value {
    match status {
        SampleStatusV1::Passed => json!({ "status": "passed" }),
        SampleStatusV1::Failed { code } => json!({ "code": code, "status": "failed" }),
    }
}

fn attempt_json(attempt: &BenchmarkSampleAttemptV1) -> Value {
    match attempt {
        BenchmarkSampleAttemptV1::Passed {
            sequence,
            cohort,
            planned_slot,
            sample,
        } => json!({
            "cohort": cohort_json(*cohort),
            "planned_slot": planned_slot,
            "sample": sample_json(sample),
            "sequence": sequence,
            "status": "passed",
        }),
        BenchmarkSampleAttemptV1::Failed {
            sequence,
            cohort,
            planned_slot,
            code,
        } => json!({
            "code": code,
            "cohort": cohort_json(*cohort),
            "planned_slot": planned_slot,
            "sequence": sequence,
            "status": "failed",
        }),
    }
}

fn load_or_compile_json(observation: &LoadOrCompileObservationV1) -> Value {
    let boundary = match observation.execution_boundary {
        ExecutionBoundaryV1::LoadOrCompile => "load_or_compile",
        ExecutionBoundaryV1::ApiWall => "api_wall",
        ExecutionBoundaryV1::QueueWall => "queue_wall",
        ExecutionBoundaryV1::GpuTimestamp => "gpu_timestamp",
    };
    json!({
        "clock_resolution_ns": observation.clock_resolution_ns,
        "clock_source": observation.clock_source,
        "duration_ns": observation.duration_ns,
        "execution_boundary": boundary,
        "passport_blake3": observation.passport_blake3,
        "protocol_blake3": observation.protocol_blake3,
        "status": sample_status_json(&observation.status),
        "thermal_after": observation_json(&observation.thermal_after),
        "thermal_before": observation_json(&observation.thermal_before),
        "workload_blake3": observation.workload_blake3,
    })
}

fn passed_attempt(
    sequence: u32,
    cohort: BenchmarkCohortV1,
    planned_slot: u32,
    sample: BenchmarkSampleV1,
) -> BenchmarkSampleAttemptV1 {
    BenchmarkSampleAttemptV1::Passed {
        sequence,
        cohort,
        planned_slot,
        sample,
    }
}

fn failed_attempt(
    sequence: u32,
    cohort: BenchmarkCohortV1,
    planned_slot: u32,
    code: &str,
) -> BenchmarkSampleAttemptV1 {
    BenchmarkSampleAttemptV1::Failed {
        sequence,
        cohort,
        planned_slot,
        code: code.to_owned(),
    }
}

fn assembly_fixture(input: BenchmarkEvidenceInputV1) -> BenchmarkEvidenceAssemblyInputV1 {
    let reference = reference_unsigned_value(&input, &expected_summary());
    let load_or_compile = LoadOrCompileObservationV1 {
        execution_boundary: ExecutionBoundaryV1::LoadOrCompile,
        duration_ns: 9_876_543_210,
        clock_source: input.protocol.clock_source.clone(),
        clock_resolution_ns: input.protocol.clock_resolution_ns,
        passport_blake3: component_blake3(&reference["passport"]),
        workload_blake3: component_blake3(&reference["workload"]),
        protocol_blake3: component_blake3(&reference["protocol"]),
        thermal_before: input.passport.thermal_state.clone(),
        thermal_after: input.passport.thermal_state.clone(),
        status: SampleStatusV1::Passed,
    };
    let BenchmarkEvidenceInputV1 {
        passport,
        workload,
        correctness_anchor,
        protocol,
        cold_sample,
        warmup_samples,
        measured_samples,
    } = input;
    let mut attempt_log = Vec::with_capacity(1 + warmup_samples.len() + measured_samples.len());
    attempt_log.push(passed_attempt(0, BenchmarkCohortV1::Cold, 0, cold_sample));
    for (slot, sample) in warmup_samples.into_iter().enumerate() {
        attempt_log.push(passed_attempt(
            u32::try_from(attempt_log.len()).unwrap(),
            BenchmarkCohortV1::Warmup,
            u32::try_from(slot).unwrap(),
            sample,
        ));
    }
    for (slot, sample) in measured_samples.into_iter().enumerate() {
        attempt_log.push(passed_attempt(
            u32::try_from(attempt_log.len()).unwrap(),
            BenchmarkCohortV1::Measured,
            u32::try_from(slot).unwrap(),
            sample,
        ));
    }
    BenchmarkEvidenceAssemblyInputV1 {
        passport,
        workload,
        correctness_anchor,
        protocol,
        load_or_compile,
        attempt_log,
    }
}

fn browser_cohort_plan_value(input: &BenchmarkEvidenceAssemblyInputV1) -> Value {
    json!({
        "correctness_anchor": input.correctness_anchor,
        "load_or_compile": load_or_compile_json(&input.load_or_compile),
        "passport": input.passport,
        "protocol": input.protocol,
        "schema_version": 1,
        "workload": input.workload,
    })
}

fn browser_cohort_assembly_value(input: &BenchmarkEvidenceAssemblyInputV1) -> Value {
    let mut value = browser_cohort_plan_value(input);
    value["attempt_log"] = Value::Array(input.attempt_log.iter().map(attempt_json).collect());
    value
}

fn reference_signed_assembly_bytes(
    input: &BenchmarkEvidenceAssemblyInputV1,
    evidence_source: &BenchmarkEvidenceInputV1,
) -> (Vec<u8>, String, Vec<u8>) {
    let evidence = reference_unsigned_value(evidence_source, &expected_summary());
    let mut value = json!({
        "attempt_log": input.attempt_log.iter().map(attempt_json).collect::<Vec<_>>(),
        "correctness_anchor": evidence["correctness_anchor"],
        "load_or_compile": load_or_compile_json(&input.load_or_compile),
        "passport": evidence["passport"],
        "protocol": evidence["protocol"],
        "schema_version": 1,
        "workload": evidence["workload"],
    });
    let unsigned = canonical_value_bytes(&value);
    let assembly_blake3 = blake3::hash(&unsigned).to_hex().to_string();
    value["assembly_blake3"] = Value::String(assembly_blake3.clone());
    (canonical_value_bytes(&value), assembly_blake3, unsigned)
}

fn resign_assembly(value: &mut Value) -> Vec<u8> {
    value.as_object_mut().unwrap().remove("assembly_blake3");
    let unsigned = canonical_value_bytes(value);
    value["assembly_blake3"] = Value::String(blake3::hash(&unsigned).to_hex().to_string());
    canonical_value_bytes(value)
}

fn assert_assembly_rejected(
    input: BenchmarkEvidenceAssemblyInputV1,
    expected: BenchmarkErrorCodeV1,
) {
    let canonical = canonical_benchmark_evidence_assembly_bytes_v1(&input);
    let parse_error = AssembledBenchmarkEvidenceV1::parse_canonical(&canonical)
        .expect_err("self-consistent hostile canonical assembly must fail closed");
    assert_eq!(
        parse_error.code(),
        expected,
        "unexpected canonical rejection: {parse_error}",
    );
    let error = AssembledBenchmarkEvidenceV1::assemble(input)
        .expect_err("hostile assembly input must fail closed");
    assert_eq!(error.code(), expected, "unexpected rejection: {error}");
}

fn assert_assembly_roundtrip_accepted(
    input: BenchmarkEvidenceAssemblyInputV1,
    context: &str,
) -> AssembledBenchmarkEvidenceV1 {
    let canonical = canonical_benchmark_evidence_assembly_bytes_v1(&input);
    let parsed = AssembledBenchmarkEvidenceV1::parse_canonical(&canonical)
        .unwrap_or_else(|error| panic!("{context}: canonical parse failed: {error}"));
    let direct = AssembledBenchmarkEvidenceV1::assemble(input)
        .unwrap_or_else(|error| panic!("{context}: direct assembly failed: {error}"));
    assert_eq!(parsed.canonical_assembly_bytes(), canonical, "{context}");
    assert_eq!(
        parsed.evidence().canonical_bytes(),
        direct.evidence().canonical_bytes(),
        "{context}",
    );
    assert_eq!(
        parsed.assembly_blake3(),
        direct.assembly_blake3(),
        "{context}"
    );
    parsed
}

fn assert_assembly_parse_rejected(bytes: &[u8], expected: BenchmarkErrorCodeV1) {
    let error = AssembledBenchmarkEvidenceV1::parse_canonical(bytes)
        .expect_err("hostile canonical assembly must fail closed");
    assert_eq!(error.code(), expected, "unexpected rejection: {error}");
}

fn attempt_sample(attempt: &BenchmarkSampleAttemptV1) -> &BenchmarkSampleV1 {
    match attempt {
        BenchmarkSampleAttemptV1::Passed { sample, .. } => sample,
        BenchmarkSampleAttemptV1::Failed { .. } => panic!("test expected a passed attempt"),
    }
}

fn attempt_sample_mut(attempt: &mut BenchmarkSampleAttemptV1) -> &mut BenchmarkSampleV1 {
    match attempt {
        BenchmarkSampleAttemptV1::Passed { sample, .. } => sample,
        BenchmarkSampleAttemptV1::Failed { .. } => panic!("test expected a passed attempt"),
    }
}

fn set_attempt_sequence(attempt: &mut BenchmarkSampleAttemptV1, value: u32) {
    match attempt {
        BenchmarkSampleAttemptV1::Passed { sequence, .. }
        | BenchmarkSampleAttemptV1::Failed { sequence, .. } => *sequence = value,
    }
}

fn renumber_attempt_sequences(attempts: &mut [BenchmarkSampleAttemptV1]) {
    for (sequence, attempt) in attempts.iter_mut().enumerate() {
        set_attempt_sequence(attempt, u32::try_from(sequence).unwrap());
    }
}

fn indexed_sample(template: &BenchmarkSampleV1, index: u32, duration_ns: u64) -> BenchmarkSampleV1 {
    let mut sample = template.clone();
    sample.index = index;
    sample.schedule_slot = index;
    sample.api_wall_ns = duration_ns;
    sample.queue_wall = DurationObservationV1::Available {
        duration_ns: duration_ns - 1,
    };
    sample
}

fn cohort_fixture(
    class: BenchmarkClassV1,
    warmup_count: u32,
    measured_count: u32,
) -> BenchmarkEvidenceInputV1 {
    let mut input = fixture();
    input.protocol.class = class;
    input.protocol.warmup_count = warmup_count;
    input.protocol.measured_count = measured_count;
    let template = input.cold_sample.clone();
    input.warmup_samples = (0..warmup_count)
        .map(|index| indexed_sample(&template, index, 10_000_000 + u64::from(index)))
        .collect();
    input.measured_samples = (0..measured_count)
        .map(|index| indexed_sample(&template, index, 20_000_000 + u64::from(index)))
        .collect();
    input
}

fn micro_fixture(warmup_count: u32, measured_count: u32) -> BenchmarkEvidenceInputV1 {
    cohort_fixture(BenchmarkClassV1::Micro, warmup_count, measured_count)
}

fn stage_fixture(warmup_count: u32, measured_count: u32) -> BenchmarkEvidenceInputV1 {
    cohort_fixture(BenchmarkClassV1::StageMacro, warmup_count, measured_count)
}

fn contains_key_recursive(value: &Value, forbidden: &str) -> bool {
    match value {
        Value::Object(object) => {
            object.contains_key(forbidden)
                || object
                    .values()
                    .any(|value| contains_key_recursive(value, forbidden))
        }
        Value::Array(array) => array
            .iter()
            .any(|value| contains_key_recursive(value, forbidden)),
        _ => false,
    }
}

#[test]
fn offline_assembly_has_an_independent_versioned_self_hash_oracle_and_exact_evidence() {
    let source = fixture();
    let expected_evidence = BenchmarkEvidenceV1::build(source.clone()).unwrap();
    let input = assembly_fixture(source.clone());
    let (reference_bytes, reference_hash, reference_unsigned) =
        reference_signed_assembly_bytes(&input, &source);
    let canonical = canonical_benchmark_evidence_assembly_bytes_v1(&input);

    assert_eq!(canonical, reference_bytes);
    assert_eq!(canonical.len(), EXPECTED_ASSEMBLY_CANONICAL_LENGTH);
    assert_eq!(reference_hash, EXPECTED_ASSEMBLY_BLAKE3);
    assert_eq!(
        blake3::hash(&canonical).to_hex().as_str(),
        EXPECTED_ASSEMBLY_CANONICAL_BLAKE3,
    );
    assert_eq!(
        blake3::hash(&reference_unsigned).to_hex().as_str(),
        EXPECTED_ASSEMBLY_BLAKE3,
    );
    assert_eq!(canonical.last(), Some(&b'\n'));
    assert_eq!(canonical.iter().filter(|byte| **byte == b'\n').count(), 1);

    let assembled = AssembledBenchmarkEvidenceV1::parse_canonical(&canonical).unwrap();
    assert_eq!(assembled.canonical_assembly_bytes(), canonical);
    assert_eq!(assembled.assembly_blake3(), EXPECTED_ASSEMBLY_BLAKE3);
    assert_eq!(assembled.load_or_compile().duration_ns, 9_876_543_210);
    assert_eq!(
        assembled.evidence().canonical_bytes(),
        expected_evidence.canonical_bytes(),
    );
    assert_eq!(assembled.evidence().summary(), expected_evidence.summary());
    BenchmarkEvidenceV1::parse_canonical(&assembled.evidence().canonical_bytes()).unwrap();

    let value: Value = serde_json::from_slice(&canonical).unwrap();
    assert_eq!(
        value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [
            "assembly_blake3",
            "attempt_log",
            "correctness_anchor",
            "load_or_compile",
            "passport",
            "protocol",
            "schema_version",
            "workload",
        ],
    );
}

#[test]
fn preparation_is_a_separate_content_addressed_companion_not_api_wall_time() {
    let source = fixture();
    let first = AssembledBenchmarkEvidenceV1::assemble(assembly_fixture(source.clone())).unwrap();

    let mut duration_changed = assembly_fixture(source.clone());
    duration_changed.load_or_compile.duration_ns += 1;
    let duration_changed = AssembledBenchmarkEvidenceV1::assemble(duration_changed).unwrap();

    let mut valid_thermal_drift = assembly_fixture(source);
    valid_thermal_drift.load_or_compile.thermal_after =
        available("fair", "ProcessInfo.thermalState");
    let valid_thermal_drift = AssembledBenchmarkEvidenceV1::assemble(valid_thermal_drift).unwrap();

    for changed in [&duration_changed, &valid_thermal_drift] {
        assert_eq!(
            changed.evidence().canonical_bytes(),
            first.evidence().canonical_bytes(),
        );
        assert_eq!(
            changed.evidence().evidence_blake3(),
            first.evidence().evidence_blake3(),
        );
        assert_ne!(changed.assembly_blake3(), first.assembly_blake3());
        assert_ne!(
            changed.canonical_assembly_bytes(),
            first.canonical_assembly_bytes(),
        );
    }
    assert_eq!(
        duration_changed.evidence().summary(),
        first.evidence().summary(),
    );
    let evidence: Value = serde_json::from_slice(&first.evidence().canonical_bytes()).unwrap();
    assert!(!contains_key_recursive(&evidence, "load_or_compile"));
    assert!(!contains_key_recursive(&evidence, "attempt_log"));
}

#[test]
fn frozen_browser_and_native_collector_wires_enter_measured_slot_seven_unchanged() {
    let browser_wire = include_bytes!("../../../web/tests/fixtures/m7d1a2_browser_sample_v1.json");
    let browser_sample: BenchmarkSampleV1 = serde_json::from_slice(browser_wire).unwrap();
    assert_eq!((browser_sample.index, browser_sample.schedule_slot), (7, 7));
    let browser_wire_value: Value = serde_json::from_slice(browser_wire).unwrap();
    let mut browser = fixture();
    browser.measured_samples[7] = browser_sample;
    let browser_input = assembly_fixture(browser);
    let browser_canonical = canonical_benchmark_evidence_assembly_bytes_v1(&browser_input);
    let browser_value: Value = serde_json::from_slice(&browser_canonical).unwrap();
    assert_eq!(
        browser_value["attempt_log"][11]["sample"],
        browser_wire_value
    );
    let browser_assembled =
        AssembledBenchmarkEvidenceV1::parse_canonical(&browser_canonical).unwrap();
    assert_eq!(
        browser_assembled.canonical_assembly_bytes(),
        browser_canonical
    );

    let native_wire = include_bytes!("../../../web/tests/fixtures/m7d1a2_native_sample_v1.json");
    let native_sample: BenchmarkSampleV1 = serde_json::from_slice(native_wire).unwrap();
    assert_eq!((native_sample.index, native_sample.schedule_slot), (7, 7));
    let native_wire_value: Value = serde_json::from_slice(native_wire).unwrap();
    let mut native = supported_timestamp_fixture();
    native.measured_samples[7] = native_sample;
    let native_input = assembly_fixture(native);
    let native_canonical = canonical_benchmark_evidence_assembly_bytes_v1(&native_input);
    let native_value: Value = serde_json::from_slice(&native_canonical).unwrap();
    assert_eq!(native_value["attempt_log"][11]["sample"], native_wire_value);
    let native_assembled =
        AssembledBenchmarkEvidenceV1::parse_canonical(&native_canonical).unwrap();
    assert_eq!(
        native_assembled.canonical_assembly_bytes(),
        native_canonical
    );
    assert_eq!(
        native_assembled.load_or_compile().clock_source,
        "std-instant-monotonic",
    );
}

#[test]
fn webkit_and_legacy_unfused_baselines_pass_through_the_same_assembly_boundary() {
    let webkit = assert_assembly_roundtrip_accepted(
        assembly_fixture(webkit_fixture()),
        "WebKit baseline assembly is supported",
    );
    assert_eq!(webkit.load_or_compile().clock_source, "performance-now");
    assert_eq!(webkit.evidence().claim_class(), ClaimClassV1::BaselineOnly);

    let legacy =
        AssembledBenchmarkEvidenceV1::parse_canonical(&actual_legacy_browser_assembly_bytes())
            .expect("legacy unfused baseline assembly is supported");
    let evidence: Value = serde_json::from_slice(&legacy.evidence().canonical_bytes()).unwrap();
    assert_eq!(evidence["workload"]["qkv_policy"], "disabled");
    assert_eq!(evidence["workload"]["qkv_outcome"], "disabled");
    assert_eq!(
        evidence["workload"]["kernel_variant"]["id"],
        "vision-stack-legacy-f32-v1",
    );
}

#[test]
fn supplied_attempt_journal_rejects_recorded_failures_and_invalid_failure_codes() {
    for (position, cohort, slot, code) in [
        (0, BenchmarkCohortV1::Cold, 0, "cold-failed"),
        (2, BenchmarkCohortV1::Warmup, 1, "thermal-interruption"),
        (9, BenchmarkCohortV1::Measured, 5, "validator-failed"),
    ] {
        let mut input = assembly_fixture(fixture());
        input.attempt_log[position] = failed_attempt(position as u32, cohort, slot, code);
        assert_assembly_rejected(input, BenchmarkErrorCodeV1::FailedSample);
    }

    let mut failed_then_retry = assembly_fixture(fixture());
    let retry = attempt_sample(&failed_then_retry.attempt_log[2]).clone();
    failed_then_retry.attempt_log[2] =
        failed_attempt(2, BenchmarkCohortV1::Warmup, 1, "background-load");
    failed_then_retry
        .attempt_log
        .insert(3, passed_attempt(3, BenchmarkCohortV1::Warmup, 1, retry));
    renumber_attempt_sequences(&mut failed_then_retry.attempt_log);
    assert_assembly_rejected(failed_then_retry, BenchmarkErrorCodeV1::FailedSample);

    let mut empty_code = assembly_fixture(fixture());
    empty_code.attempt_log[0] = failed_attempt(0, BenchmarkCohortV1::Cold, 0, "   ");
    assert_assembly_rejected(empty_code, BenchmarkErrorCodeV1::InvalidAttemptJournal);

    let mut nested_failure = assembly_fixture(fixture());
    attempt_sample_mut(&mut nested_failure.attempt_log[9]).status = SampleStatusV1::Failed {
        code: "validator-failed".to_owned(),
    };
    assert_assembly_rejected(nested_failure, BenchmarkErrorCodeV1::FailedSample);
}

#[test]
fn supplied_attempt_journal_rejects_zero_extra_omitted_transferred_and_misidentified_entries() {
    let cases: Vec<AssemblyMutation> = vec![
        (
            "zero cold",
            Box::new(|input| {
                input.attempt_log.remove(0);
                renumber_attempt_sequences(&mut input.attempt_log);
            }),
        ),
        (
            "extra distinct cold",
            Box::new(|input| {
                let extra = indexed_sample(attempt_sample(&input.attempt_log[0]), 1, 200_000_001);
                input
                    .attempt_log
                    .insert(1, passed_attempt(1, BenchmarkCohortV1::Cold, 1, extra));
                renumber_attempt_sequences(&mut input.attempt_log);
            }),
        ),
        (
            "warmup omission",
            Box::new(|input| {
                input.attempt_log.remove(2);
                renumber_attempt_sequences(&mut input.attempt_log);
            }),
        ),
        (
            "extra independently indexed warmup",
            Box::new(|input| {
                let extra = indexed_sample(attempt_sample(&input.attempt_log[3]), 3, 104_000_004);
                input
                    .attempt_log
                    .insert(4, passed_attempt(4, BenchmarkCohortV1::Warmup, 3, extra));
                renumber_attempt_sequences(&mut input.attempt_log);
            }),
        ),
        (
            "measured omission",
            Box::new(|input| {
                input.attempt_log.pop();
            }),
        ),
        (
            "extra independently indexed measured sample",
            Box::new(|input| {
                let extra = indexed_sample(
                    attempt_sample(input.attempt_log.last().unwrap()),
                    10,
                    116_000_116,
                );
                input
                    .attempt_log
                    .push(passed_attempt(14, BenchmarkCohortV1::Measured, 10, extra));
            }),
        ),
        (
            "cohort transfer",
            Box::new(|input| match &mut input.attempt_log[4] {
                BenchmarkSampleAttemptV1::Passed { cohort, .. } => {
                    *cohort = BenchmarkCohortV1::Warmup;
                }
                BenchmarkSampleAttemptV1::Failed { .. } => unreachable!(),
            }),
        ),
        (
            "warmup and measured attempts interleaved",
            Box::new(|input| {
                input.attempt_log.swap(3, 4);
                renumber_attempt_sequences(&mut input.attempt_log);
            }),
        ),
        (
            "measured attempts reordered with global sequence relabelled",
            Box::new(|input| {
                input.attempt_log.swap(5, 9);
                renumber_attempt_sequences(&mut input.attempt_log);
            }),
        ),
        (
            "sequence gap",
            Box::new(|input| set_attempt_sequence(&mut input.attempt_log[8], 80)),
        ),
        (
            "planned slot differs from sample",
            Box::new(|input| match &mut input.attempt_log[8] {
                BenchmarkSampleAttemptV1::Passed { planned_slot, .. } => *planned_slot = 80,
                BenchmarkSampleAttemptV1::Failed { .. } => unreachable!(),
            }),
        ),
    ];
    for (name, mutate) in cases {
        let mut input = assembly_fixture(fixture());
        mutate(&mut input);
        let canonical = canonical_benchmark_evidence_assembly_bytes_v1(&input);
        let parse_error = AssembledBenchmarkEvidenceV1::parse_canonical(&canonical)
            .expect_err("invalid self-consistent journal must fail canonical parsing");
        assert_eq!(
            parse_error.code(),
            BenchmarkErrorCodeV1::InvalidAttemptJournal,
            "canonical case {name}: {parse_error}",
        );
        let error = AssembledBenchmarkEvidenceV1::assemble(input)
            .expect_err("invalid journal must fail closed");
        assert_eq!(
            error.code(),
            BenchmarkErrorCodeV1::InvalidAttemptJournal,
            "case {name}: {error}",
        );
    }
}

#[test]
fn content_address_detects_post_freeze_retry_replacement_and_reorder_with_relabelling() {
    let canonical = canonical_benchmark_evidence_assembly_bytes_v1(&assembly_fixture(fixture()));

    let mut retry_replacement: Value = serde_json::from_slice(&canonical).unwrap();
    retry_replacement["attempt_log"][2]["sample"]["api_wall_ns"] = Value::from(109_000_109);
    retry_replacement["attempt_log"][2]["sample"]["queue_wall"]["duration_ns"] =
        Value::from(109_000_108);
    assert_assembly_parse_rejected(
        &canonical_value_bytes(&retry_replacement),
        BenchmarkErrorCodeV1::SelfHashMismatch,
    );

    let mut reordered: Value = serde_json::from_slice(&canonical).unwrap();
    reordered["attempt_log"].as_array_mut().unwrap().swap(1, 2);
    for (position, slot) in [(1, 0), (2, 1)] {
        reordered["attempt_log"][position]["sequence"] = Value::from(position as u64);
        reordered["attempt_log"][position]["planned_slot"] = Value::from(slot);
        reordered["attempt_log"][position]["sample"]["index"] = Value::from(slot);
        reordered["attempt_log"][position]["sample"]["schedule_slot"] = Value::from(slot);
    }
    assert_assembly_parse_rejected(
        &canonical_value_bytes(&reordered),
        BenchmarkErrorCodeV1::SelfHashMismatch,
    );

    // This pure boundary proves integrity of the supplied frozen journal. It cannot observe an
    // event removed before authorship; M7d1a3's sealed cohort runner must own that completeness.
}

#[test]
fn stage_macro_and_micro_minimum_counts_are_enforced_through_assembly() {
    for (context, input) in [
        ("stage minimum 3/10", stage_fixture(3, 10)),
        ("stage measured above minimum 3/11", stage_fixture(3, 11)),
        ("stage warmup above minimum 4/10", stage_fixture(4, 10)),
        ("stage both above minimum 4/11", stage_fixture(4, 11)),
        ("micro minimum 10/30", micro_fixture(10, 30)),
        ("micro measured above minimum 10/31", micro_fixture(10, 31)),
        ("micro warmup above minimum 11/30", micro_fixture(11, 30)),
        ("micro both above minimum 11/31", micro_fixture(11, 31)),
    ] {
        let _ = assert_assembly_roundtrip_accepted(assembly_fixture(input), context);
    }

    let mut short_stage = fixture();
    short_stage.protocol.warmup_count = 2;
    short_stage.warmup_samples.truncate(2);
    assert_assembly_rejected(
        assembly_fixture(short_stage),
        BenchmarkErrorCodeV1::InvalidProtocol,
    );

    assert_assembly_rejected(
        assembly_fixture(micro_fixture(9, 30)),
        BenchmarkErrorCodeV1::InvalidProtocol,
    );
    assert_assembly_rejected(
        assembly_fixture(micro_fixture(10, 29)),
        BenchmarkErrorCodeV1::InvalidProtocol,
    );
}

#[test]
fn preparation_requires_exact_clock_environment_and_content_addressed_leaf_links() {
    let mutations: Vec<AssemblyMutation> = vec![
        (
            "wrong boundary",
            Box::new(|input| {
                input.load_or_compile.execution_boundary = ExecutionBoundaryV1::ApiWall;
            }),
        ),
        (
            "zero duration",
            Box::new(|input| input.load_or_compile.duration_ns = 0),
        ),
        (
            "saturated duration",
            Box::new(|input| input.load_or_compile.duration_ns = u64::MAX),
        ),
        (
            "valid native clock on browser leaf",
            Box::new(|input| {
                input.load_or_compile.clock_source = "std-instant-monotonic".to_owned();
            }),
        ),
        (
            "different positive resolution",
            Box::new(|input| input.load_or_compile.clock_resolution_ns += 1),
        ),
        (
            "passport companion hash",
            Box::new(|input| input.load_or_compile.passport_blake3 = hash('0')),
        ),
        (
            "workload companion hash",
            Box::new(|input| input.load_or_compile.workload_blake3 = hash('0')),
        ),
        (
            "protocol companion hash",
            Box::new(|input| input.load_or_compile.protocol_blake3 = hash('0')),
        ),
        (
            "source tree leaf",
            Box::new(|input| input.passport.source_tree_blake3 = hash('0')),
        ),
        (
            "compiler runtime leaf",
            Box::new(|input| input.passport.compiler_runtime_blake3 = hash('0')),
        ),
        (
            "adapter name leaf",
            Box::new(|input| input.passport.adapter_name = "different adapter".to_owned()),
        ),
        (
            "backend leaf",
            Box::new(|input| input.passport.backend.kind = BackendKindV1::WebkitWebgpu),
        ),
        (
            "model lock leaf",
            Box::new(|input| input.passport.model.model_lock_blake3 = hash('0')),
        ),
        (
            "model pack leaf",
            Box::new(|input| input.passport.model.pack_blake3 = hash('0')),
        ),
        (
            "kernel variant leaf",
            Box::new(|input| input.workload.kernel_variant.id.push_str("-other")),
        ),
        (
            "kernel source-set leaf",
            Box::new(|input| input.workload.kernel_variant.source_set_blake3 = hash('0')),
        ),
        (
            "kernel ABI leaf",
            Box::new(|input| input.workload.kernel_variant.abi_blake3 = hash('0')),
        ),
        (
            "residency-plan leaf",
            Box::new(|input| input.workload.residency_plan.id.push_str("-other")),
        ),
        (
            "valid but mismatched thermal before",
            Box::new(|input| {
                input.load_or_compile.thermal_before =
                    available("fair", "ProcessInfo.thermalState");
            }),
        ),
        (
            "malformed thermal before",
            Box::new(|input| {
                input.load_or_compile.thermal_before = ObservationV1::Unavailable {
                    reason: String::new(),
                    method: "probe".to_owned(),
                };
            }),
        ),
        (
            "malformed thermal after",
            Box::new(|input| {
                input.load_or_compile.thermal_after = ObservationV1::Available {
                    value: "nominal".to_owned(),
                    method: String::new(),
                };
            }),
        ),
    ];
    for (name, mutate) in mutations {
        let mut input = assembly_fixture(fixture());
        mutate(&mut input);
        let canonical = canonical_benchmark_evidence_assembly_bytes_v1(&input);
        let parse_error = AssembledBenchmarkEvidenceV1::parse_canonical(&canonical)
            .expect_err("invalid self-consistent preparation must fail canonical parsing");
        assert_eq!(
            parse_error.code(),
            BenchmarkErrorCodeV1::InvalidPreparation,
            "canonical case {name}: {parse_error}",
        );
        let error = AssembledBenchmarkEvidenceV1::assemble(input)
            .expect_err("invalid preparation must fail closed");
        assert_eq!(
            error.code(),
            BenchmarkErrorCodeV1::InvalidPreparation,
            "case {name}: {error}",
        );
    }

    let mut native_wrong_clock = assembly_fixture(supported_timestamp_fixture());
    native_wrong_clock.load_or_compile.clock_source = "performance-now".to_owned();
    assert_assembly_rejected(native_wrong_clock, BenchmarkErrorCodeV1::InvalidPreparation);

    let mut failed = assembly_fixture(fixture());
    failed.load_or_compile.status = SampleStatusV1::Failed {
        code: "shader-compilation-failed".to_owned(),
    };
    assert_assembly_rejected(failed, BenchmarkErrorCodeV1::FailedSample);

    let mut empty_failure = assembly_fixture(fixture());
    empty_failure.load_or_compile.status = SampleStatusV1::Failed {
        code: " ".to_owned(),
    };
    assert_assembly_rejected(empty_failure, BenchmarkErrorCodeV1::InvalidPreparation);
}

fn insert_before(bytes: &[u8], needle: &[u8], insertion: &[u8]) -> Vec<u8> {
    let position = bytes
        .windows(needle.len())
        .position(|window| window == needle)
        .expect("test needle is present");
    let mut mutated = bytes.to_vec();
    mutated.splice(position..position, insertion.iter().copied());
    mutated
}

#[test]
fn canonical_assembly_parser_is_recursive_closed_and_integer_exact() {
    let canonical = canonical_benchmark_evidence_assembly_bytes_v1(&assembly_fixture(fixture()));

    let mutations: Vec<AssemblyJsonMutation> = vec![
        (
            "top unknown",
            BenchmarkErrorCodeV1::SchemaMismatch,
            Box::new(|value| value["unknown"] = Value::from(1)),
        ),
        (
            "top missing",
            BenchmarkErrorCodeV1::SchemaMismatch,
            Box::new(|value| {
                value.as_object_mut().unwrap().remove("load_or_compile");
            }),
        ),
        (
            "load unknown",
            BenchmarkErrorCodeV1::SchemaMismatch,
            Box::new(|value| value["load_or_compile"]["unknown"] = Value::from(1)),
        ),
        (
            "load missing",
            BenchmarkErrorCodeV1::SchemaMismatch,
            Box::new(|value| {
                value["load_or_compile"]
                    .as_object_mut()
                    .unwrap()
                    .remove("thermal_after");
            }),
        ),
        (
            "attempt unknown",
            BenchmarkErrorCodeV1::SchemaMismatch,
            Box::new(|value| value["attempt_log"][0]["unknown"] = Value::from(1)),
        ),
        (
            "attempt missing",
            BenchmarkErrorCodeV1::SchemaMismatch,
            Box::new(|value| {
                value["attempt_log"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("planned_slot");
            }),
        ),
        (
            "malformed attempt status",
            BenchmarkErrorCodeV1::SchemaMismatch,
            Box::new(|value| value["attempt_log"][0]["status"] = Value::from("failed")),
        ),
        (
            "malformed preparation status",
            BenchmarkErrorCodeV1::SchemaMismatch,
            Box::new(|value| {
                value["load_or_compile"]["status"]["status"] = Value::from("unknown");
            }),
        ),
        (
            "malformed nested observation",
            BenchmarkErrorCodeV1::SchemaMismatch,
            Box::new(|value| {
                value["load_or_compile"]["thermal_after"]
                    .as_object_mut()
                    .unwrap()
                    .remove("value");
            }),
        ),
        (
            "float preparation duration",
            BenchmarkErrorCodeV1::InvalidInteger,
            Box::new(|value| value["load_or_compile"]["duration_ns"] = Value::from(1.5)),
        ),
        (
            "negative attempt sequence",
            BenchmarkErrorCodeV1::InvalidInteger,
            Box::new(|value| value["attempt_log"][0]["sequence"] = Value::from(-1)),
        ),
    ];
    for (name, expected, mutate) in mutations {
        let mut value: Value = serde_json::from_slice(&canonical).unwrap();
        mutate(&mut value);
        let bytes = resign_assembly(&mut value);
        let error = AssembledBenchmarkEvidenceV1::parse_canonical(&bytes)
            .expect_err("hostile canonical assembly must fail closed");
        assert_eq!(error.code(), expected, "case {name}: {error}");
    }

    let top_duplicate = insert_before(&canonical, b"\"attempt_log\":", b"\"schema_version\":1,");
    assert_assembly_parse_rejected(&top_duplicate, BenchmarkErrorCodeV1::SchemaMismatch);
    let load_duplicate = insert_before(
        &canonical,
        b"\"duration_ns\":9876543210",
        b"\"duration_ns\":1,",
    );
    assert_assembly_parse_rejected(&load_duplicate, BenchmarkErrorCodeV1::SchemaMismatch);
    let attempt_duplicate = insert_before(
        &canonical,
        b"\"cohort\":\"cold\"",
        b"\"cohort\":\"measured\",",
    );
    assert_assembly_parse_rejected(&attempt_duplicate, BenchmarkErrorCodeV1::SchemaMismatch);

    let mut wrong_schema: Value = serde_json::from_slice(&canonical).unwrap();
    wrong_schema["schema_version"] = Value::from(2);
    assert_assembly_parse_rejected(
        &resign_assembly(&mut wrong_schema),
        BenchmarkErrorCodeV1::UnsupportedSchema,
    );
    let mut float_schema: Value = serde_json::from_slice(&canonical).unwrap();
    float_schema["schema_version"] = Value::from(1.5);
    assert_assembly_parse_rejected(
        &resign_assembly(&mut float_schema),
        BenchmarkErrorCodeV1::InvalidInteger,
    );

    let mut wrong_hash: Value = serde_json::from_slice(&canonical).unwrap();
    wrong_hash["assembly_blake3"] = Value::String(hash('0'));
    assert_assembly_parse_rejected(
        &canonical_value_bytes(&wrong_hash),
        BenchmarkErrorCodeV1::SelfHashMismatch,
    );

    let mut leading_space = canonical.clone();
    leading_space.insert(0, b' ');
    assert_assembly_parse_rejected(&leading_space, BenchmarkErrorCodeV1::NonCanonical);
    let mut missing_lf = canonical.clone();
    missing_lf.pop();
    assert_assembly_parse_rejected(&missing_lf, BenchmarkErrorCodeV1::NonCanonical);
    let mut extra_lf = canonical;
    extra_lf.push(b'\n');
    assert_assembly_parse_rejected(&extra_lf, BenchmarkErrorCodeV1::NonCanonical);
}

#[test]
fn assembled_result_is_opaque_baseline_only_and_has_no_performance_claim_surface() {
    let _ = <AssembledBenchmarkEvidenceV1 as AmbiguousIfDeserialize<_>>::marker;
    let assembled = AssembledBenchmarkEvidenceV1::assemble(assembly_fixture(fixture())).unwrap();
    assert_eq!(
        assembled.evidence().claim_class(),
        ClaimClassV1::BaselineOnly
    );

    let assembly: Value = serde_json::from_slice(&assembled.canonical_assembly_bytes()).unwrap();
    let evidence: Value = serde_json::from_slice(&assembled.evidence().canonical_bytes()).unwrap();
    assert_eq!(
        evidence
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [
            "claim_class",
            "cold_sample",
            "correctness_anchor",
            "evidence_blake3",
            "measured_samples",
            "passport",
            "protocol",
            "schema_version",
            "summary",
            "warmup_samples",
            "workload",
        ],
    );
    for forbidden in [
        "comparable_pair",
        "accepted_improvement",
        "speedup",
        "winner",
        "throughput",
        "resident_memory",
        "physical_attempts_complete",
    ] {
        assert!(!contains_key_recursive(&assembly, forbidden));
        assert!(!contains_key_recursive(&evidence, forbidden));
    }
}

fn assert_browser_plan_rejected(bytes: &[u8], expected: BenchmarkErrorCodeV1) {
    let error = validate_browser_benchmark_cohort_plan_v1(bytes)
        .expect_err("hostile browser cohort plan must fail before execution");
    assert_eq!(error.code(), expected, "{error}");
}

fn assert_browser_assembly_rejected(bytes: &[u8], expected: BenchmarkErrorCodeV1) {
    let error = assemble_browser_benchmark_cohort_v1(bytes)
        .expect_err("hostile browser cohort journal must fail before evidence admission");
    assert_eq!(error.code(), expected, "{error}");
}

#[test]
fn browser_cohort_adapter_validates_the_static_plan_and_delegates_exactly_to_assembly() {
    for source in [fixture(), webkit_fixture(), browser_timestamp_fixture()] {
        let input = assembly_fixture(source);
        let plan_bytes = canonical_value_bytes(&browser_cohort_plan_value(&input));
        validate_browser_benchmark_cohort_plan_v1(&plan_bytes)
            .expect("the exact static browser leaf and preparation links must be admitted");

        let request_bytes = canonical_value_bytes(&browser_cohort_assembly_value(&input));
        let bridged = assemble_browser_benchmark_cohort_v1(&request_bytes)
            .expect("the browser journal adapter must call the accepted assembler");
        let direct = AssembledBenchmarkEvidenceV1::assemble(input)
            .expect("the same typed journal must assemble directly");
        assert_eq!(bridged, direct.canonical_assembly_bytes());

        let reparsed = AssembledBenchmarkEvidenceV1::parse_canonical(&bridged)
            .expect("the bridge result must pass the canonical parser");
        assert_eq!(
            reparsed.assembly_blake3(),
            direct.assembly_blake3(),
            "direct and canonical paths must preserve one assembly identity",
        );
        assert_eq!(
            reparsed.evidence().canonical_bytes(),
            direct.evidence().canonical_bytes(),
            "the bridge cannot substitute a different admitted evidence record",
        );
    }

    let legacy_plan_bytes = canonical_value_bytes(&actual_legacy_browser_plan_value());
    validate_browser_benchmark_cohort_plan_v1(&legacy_plan_bytes)
        .expect("the actual null/empty legacy static plan must be admitted");
    let legacy_request = actual_legacy_browser_assembly_bytes();
    let mut legacy_request_value: Value = serde_json::from_slice(&legacy_request).unwrap();
    legacy_request_value
        .as_object_mut()
        .unwrap()
        .remove("assembly_blake3");
    let legacy_bridged =
        assemble_browser_benchmark_cohort_v1(&canonical_value_bytes(&legacy_request_value))
            .expect("the actual null/empty legacy journal must cross the browser adapter");
    assert_eq!(
        legacy_bridged, legacy_request,
        "the browser adapter changed the canonical legacy assembly",
    );
    AssembledBenchmarkEvidenceV1::parse_canonical(&legacy_bridged)
        .expect("the bridged legacy assembly must remain canonically admissible");
}

#[test]
fn browser_cohort_plan_adapter_is_closed_canonical_and_fail_closed_before_effects() {
    let input = assembly_fixture(fixture());
    let accepted = browser_cohort_plan_value(&input);

    let mut wrong_link = accepted.clone();
    wrong_link["load_or_compile"]["protocol_blake3"] = Value::String(hash('0'));
    assert_browser_plan_rejected(
        &canonical_value_bytes(&wrong_link),
        BenchmarkErrorCodeV1::InvalidPreparation,
    );

    let mut failed_preparation = accepted.clone();
    failed_preparation["load_or_compile"]["status"] =
        json!({ "code": "preparation_failed", "status": "failed" });
    assert_browser_plan_rejected(
        &canonical_value_bytes(&failed_preparation),
        BenchmarkErrorCodeV1::FailedSample,
    );

    let mut invalid_leaf = accepted.clone();
    invalid_leaf["workload"]["checkpoint_sha256"] = Value::String(hash('0'));
    assert_browser_plan_rejected(
        &canonical_value_bytes(&invalid_leaf),
        BenchmarkErrorCodeV1::CrossLinkMismatch,
    );

    let mut invalid_protocol = accepted.clone();
    invalid_protocol["protocol"]["synchronization"] =
        Value::String("return-before-validation".to_owned());
    invalid_protocol["load_or_compile"]["protocol_blake3"] =
        Value::String(component_blake3(&invalid_protocol["protocol"]));
    assert_browser_plan_rejected(
        &canonical_value_bytes(&invalid_protocol),
        BenchmarkErrorCodeV1::InvalidProtocol,
    );

    let mut schedule_cardinality_overflow = accepted.clone();
    schedule_cardinality_overflow["protocol"]["warmup_count"] = Value::from(u64::from(u32::MAX));
    schedule_cardinality_overflow["protocol"]["measured_count"] = Value::from(10);
    schedule_cardinality_overflow["load_or_compile"]["protocol_blake3"] =
        Value::String(component_blake3(&schedule_cardinality_overflow["protocol"]));
    assert_browser_plan_rejected(
        &canonical_value_bytes(&schedule_cardinality_overflow),
        BenchmarkErrorCodeV1::InvalidProtocol,
    );

    let mut invalid_schedule = accepted.clone();
    invalid_schedule["protocol"]["schedule"] = Value::String("cold-then-retry-outliers".to_owned());
    invalid_schedule["load_or_compile"]["protocol_blake3"] =
        Value::String(component_blake3(&invalid_schedule["protocol"]));
    assert_browser_plan_rejected(
        &canonical_value_bytes(&invalid_schedule),
        BenchmarkErrorCodeV1::InvalidSchedule,
    );

    let mut invalid_clock = accepted.clone();
    invalid_clock["protocol"]["clock_source"] = Value::String("std-instant-monotonic".to_owned());
    invalid_clock["load_or_compile"]["clock_source"] =
        Value::String("std-instant-monotonic".to_owned());
    invalid_clock["load_or_compile"]["protocol_blake3"] =
        Value::String(component_blake3(&invalid_clock["protocol"]));
    assert_browser_plan_rejected(
        &canonical_value_bytes(&invalid_clock),
        BenchmarkErrorCodeV1::InvalidProtocol,
    );

    let mut native_backend = accepted.clone();
    native_backend["passport"]["backend"]["kind"] = Value::String("native_wgpu".to_owned());
    native_backend["passport"]["backend"]["browser_version"] = Value::Null;
    native_backend["passport"]["backend"]["user_agent"] = Value::Null;
    native_backend["passport"]["backend"]["adapter_backend"] = Value::String("metal".to_owned());
    native_backend["protocol"]["clock_source"] = Value::String("std-instant-monotonic".to_owned());
    native_backend["load_or_compile"]["clock_source"] =
        Value::String("std-instant-monotonic".to_owned());
    native_backend["load_or_compile"]["passport_blake3"] =
        Value::String(component_blake3(&native_backend["passport"]));
    native_backend["load_or_compile"]["protocol_blake3"] =
        Value::String(component_blake3(&native_backend["protocol"]));
    assert_browser_plan_rejected(
        &canonical_value_bytes(&native_backend),
        BenchmarkErrorCodeV1::InvalidEnvironment,
    );

    let mut unknown = accepted.clone();
    unknown["attempt_log"] = Value::Array(Vec::new());
    assert_browser_plan_rejected(
        &canonical_value_bytes(&unknown),
        BenchmarkErrorCodeV1::SchemaMismatch,
    );

    let mut missing_protocol_field = accepted.clone();
    missing_protocol_field["protocol"]
        .as_object_mut()
        .unwrap()
        .remove("schedule");
    assert_browser_plan_rejected(
        &canonical_value_bytes(&missing_protocol_field),
        BenchmarkErrorCodeV1::SchemaMismatch,
    );

    let mut unsupported = accepted.clone();
    unsupported["schema_version"] = Value::from(2);
    assert_browser_plan_rejected(
        &canonical_value_bytes(&unsupported),
        BenchmarkErrorCodeV1::UnsupportedSchema,
    );

    let canonical = canonical_value_bytes(&accepted);
    let duplicate = insert_before(&canonical, b"\"passport\":", b"\"schema_version\":1,");
    assert_browser_plan_rejected(&duplicate, BenchmarkErrorCodeV1::SchemaMismatch);
    let nested_duplicate = insert_before(
        &canonical,
        b"\"synchronization\":",
        b"\"schedule\":\"single-stable-variant-v1\",",
    );
    assert_browser_plan_rejected(&nested_duplicate, BenchmarkErrorCodeV1::SchemaMismatch);

    let mut leading_space = canonical.clone();
    leading_space.insert(0, b' ');
    assert_browser_plan_rejected(&leading_space, BenchmarkErrorCodeV1::NonCanonical);
    let mut missing_lf = canonical;
    missing_lf.pop();
    assert_browser_plan_rejected(&missing_lf, BenchmarkErrorCodeV1::NonCanonical);
}

#[test]
fn browser_cohort_assembly_adapter_cannot_bypass_the_closed_journal_or_parser() {
    let input = assembly_fixture(fixture());
    let accepted = browser_cohort_assembly_value(&input);

    let mut failed = accepted.clone();
    let last = failed["attempt_log"].as_array().unwrap().len() - 1;
    failed["attempt_log"][last] = json!({
        "code": "collection_failed",
        "cohort": "measured",
        "planned_slot": 9,
        "sequence": 13,
        "status": "failed",
    });
    assert_browser_assembly_rejected(
        &canonical_value_bytes(&failed),
        BenchmarkErrorCodeV1::FailedSample,
    );

    let mut reordered = accepted.clone();
    reordered["attempt_log"].as_array_mut().unwrap().swap(1, 2);
    assert_browser_assembly_rejected(
        &canonical_value_bytes(&reordered),
        BenchmarkErrorCodeV1::InvalidAttemptJournal,
    );

    let mut missing = accepted.clone();
    missing
        .as_object_mut()
        .unwrap()
        .remove("correctness_anchor");
    assert_browser_assembly_rejected(
        &canonical_value_bytes(&missing),
        BenchmarkErrorCodeV1::SchemaMismatch,
    );

    let mut signed_input = accepted.clone();
    signed_input["assembly_blake3"] = Value::String(hash('a'));
    assert_browser_assembly_rejected(
        &canonical_value_bytes(&signed_input),
        BenchmarkErrorCodeV1::SchemaMismatch,
    );

    let mut missing_sample_field = accepted.clone();
    missing_sample_field["attempt_log"][0]["sample"]
        .as_object_mut()
        .unwrap()
        .remove("topology");
    assert_browser_assembly_rejected(
        &canonical_value_bytes(&missing_sample_field),
        BenchmarkErrorCodeV1::SchemaMismatch,
    );

    let canonical = canonical_value_bytes(&accepted);
    let duplicate = insert_before(&canonical, b"\"attempt_log\":", b"\"attempt_log\":[],");
    assert_browser_assembly_rejected(&duplicate, BenchmarkErrorCodeV1::SchemaMismatch);
    let nested_duplicate = insert_before(&canonical, b"\"queue_wall\":", b"\"api_wall_ns\":10000,");
    assert_browser_assembly_rejected(&nested_duplicate, BenchmarkErrorCodeV1::SchemaMismatch);

    let mut extra_lf = canonical;
    extra_lf.push(b'\n');
    assert_browser_assembly_rejected(&extra_lf, BenchmarkErrorCodeV1::NonCanonical);
}

#[test]
fn shared_m7d1a3_browser_cohort_fixture_is_a_typed_above_minimum_rust_oracle() {
    let mut source = fixture();
    source.protocol.warmup_count = 4;
    source.protocol.measured_count = 11;
    source.protocol.clock_resolution_ns = 1;
    source
        .passport
        .backend
        .limits
        .insert("max_compute_workgroups_per_dimension".to_owned(), 65_535);
    source
        .passport
        .backend
        .limits
        .insert("max_storage_buffers_per_shader_stage".to_owned(), 10);
    source.workload.kernel_variant.source_set_blake3 =
        "f20552fafb9b770ce116554732aa89631e1f7808674e68655ffdb1ec99da71c4".to_owned();
    source.workload.kernel_variant.abi_blake3 = PUBLIC_OPERATION_ABI_BLAKE3.to_owned();
    source.workload.ordered_layer_plans_blake3 = [
        "47a19d500f711826ec67aae6cd0a8f3e17b77ab971de4c4def025fc98d460d0f",
        "64514e927531e6088e6c75315fae5fb2c57ec514cd7628cf12173ea88189334f",
        "0fc996729fb058ecc597bafbb61fc532af728b0ebec47c4641eaf74dc8b5eb01",
        "5b0881b5b7d8f281119bd65e5f670d5f7f2c33973d6423914cbb9728afa4955a",
        "f66b7f891aa1e1770eb4139dfbdff0cbd0d2e0bba4852f0a294ff3910cfd9397",
        "5b2d2fa97c7a5ac0314e6383f40a7ac5e12f5b092de9f430ba8d985042787403",
        "f1fb31b12a09981e98d4724ac7467fb66d1acb4d0005cce9c53ab2579db5f40b",
        "ff0cd0c033ff85b2680636e380bb3e00ce6ace5b70e2651ee9f12179421d7d98",
        "52ef51a564146e85dc4c6fd6418dc81e6fc008d7dcbce230b0ef2085a4945967",
        "b271a656d0302471f8cb39f48f0d8aeef0ce1fb4d9c021bf24601d2b6de91746",
        "b2e41a352997382089c3b2ec3ef17090722f7bbb45c177daf0f1a57a3210b89c",
        "3420bfcbe28bd8c3fb010510cb9469fc7db90b2495a46865c4eb3e4cdedd7298",
        "3047558db0402807781ca92a45d6ee27d391c0eabb6ff6bd4c7adc58fc5563b0",
        "f1f5b5678c6d6b0c55bce58a5451ae3e73fbebe713ad97e74139a1ddfa5a9359",
        "6059bd210ce17f6e70746f8d90dd0c5d4bfef7da2dfc2d81576ca3e273abf5bd",
        "1a87f9eb68505315f95faaf496b991c4d13bb2775d64a911ce1fba1ae7bd90ef",
        "746afc29839766bb2b35e450a4ecc7912db1ee676a206097a1ca9472db9e0a59",
        "823f75975c4aae93b889b33b0d6103af387ebfea5faa5bf5cc789234068bbcb5",
        "05d36cdf29a1cbb9962f80e0b488d1883c919053b25c4b06ac2776e15fd324da",
        "4fc89cd6cfa02e9d134db75c74b25433df15ebc1fabb2b33751ef26bf94991ec",
        "ede29ab17d6f2db505d660c574654f45a7556ee3e9fcf8fadde3360a7410e41b",
        "cb40d31a6fb5c52406a53be6ba0e527885bc69faa8d665a94317eab27edd8232",
        "662b4a6c1135f51a88d616672182ed018d6bb16f5ba84c2a78aff7d012c6fe60",
        "34d6775f918898a3d4a5fc8e495c2b00b4d93812453494f9decb6959fabe484f",
        "0c554b2fa510ceb4e0628dc62b85740181d80b875b8f8a1e039baa11de25f195",
        "576e54363fc772478b1ca8f765c625ef3f4971d75cce6487692552c6f9b6eb85",
        "35957c10d79549fc8d982982d10021c702401abf091a11c20d6a6cceac3cb191",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    source.workload.residency_plan.id = "browser-bounded-shard-static_arena_alias-v1".to_owned();
    let browser_residency_plan_id = source.workload.residency_plan.id.clone();

    let mut samples = (0_u32..16)
        .map(|sequence| {
            let slot = match sequence {
                0 => 0,
                1..=4 => sequence - 1,
                _ => sequence - 5,
            };
            let mut sample = browser_sample(
                slot,
                10_000 + u64::from(sequence),
                1_000_000 + u64::from(sequence),
            );
            sample.residency_plan_id = browser_residency_plan_id.clone();
            sample.queue_wall = DurationObservationV1::Available {
                duration_ns: 5_000 + u64::from(sequence),
            };
            sample
        })
        .collect::<Vec<_>>();
    source.cold_sample = samples.remove(0);
    source.warmup_samples = samples.drain(..4).collect();
    source.measured_samples = samples;

    let mut assembly_input = assembly_fixture(source);
    assembly_input.load_or_compile.duration_ns = 40;
    assembly_input.load_or_compile.thermal_after = available("fair", "ProcessInfo.thermalState");

    let mut plan = browser_cohort_plan_value(&assembly_input);
    plan["load_or_compile"]["duration_ns"] = Value::from(1);
    plan["load_or_compile"]["thermal_after"] = plan["load_or_compile"]["thermal_before"].clone();
    let assembly_request = browser_cohort_assembly_value(&assembly_input);
    let assembled = AssembledBenchmarkEvidenceV1::assemble(assembly_input.clone()).unwrap();
    let evidence: Value = serde_json::from_slice(&assembled.evidence().canonical_bytes()).unwrap();
    let workload = &assembly_request["workload"];
    let mut legacy_input = assembly_input.clone();
    legacy_input.workload.qkv_policy = "disabled".to_owned();
    legacy_input.workload.qkv_outcome = "disabled".to_owned();
    legacy_input.workload.kernel_variant.id = "vision-stack-legacy-f32-v1".to_owned();
    legacy_input.workload.kernel_variant.source_set_blake3 =
        "70d25bb7bb9f04ac0faaf57514f454b5f70144d8501418a7c4cb29eed35138aa".to_owned();
    legacy_input
        .workload
        .kernel_variant
        .expected_topology
        .dispatch_count = 325;
    let legacy_variant_id = legacy_input.workload.kernel_variant.id.clone();
    let legacy_topology = legacy_input
        .workload
        .kernel_variant
        .expected_topology
        .clone();
    for attempt in &mut legacy_input.attempt_log {
        let sample = attempt_sample_mut(attempt);
        sample.kernel_variant_id.clone_from(&legacy_variant_id);
        sample.topology = legacy_topology.clone();
    }
    let mut legacy_request = browser_cohort_assembly_value(&legacy_input);
    legacy_request["workload"]["semantic_graph_blake3"] = Value::Null;
    legacy_request["workload"]["ordered_layer_plans_blake3"] = Value::Array(Vec::new());
    let legacy_workload_blake3 = component_blake3(&legacy_request["workload"]);
    legacy_request["load_or_compile"]["workload_blake3"] =
        Value::String(legacy_workload_blake3.clone());
    let mut legacy_signed = legacy_request;
    let legacy_canonical = resign_assembly(&mut legacy_signed);
    let legacy_assembled = AssembledBenchmarkEvidenceV1::parse_canonical(&legacy_canonical)
        .expect("the shared legacy variant must admit actual null/empty compiler identities");
    let mut legacy_plan = browser_cohort_plan_value(&legacy_input);
    legacy_plan["workload"]["semantic_graph_blake3"] = Value::Null;
    legacy_plan["workload"]["ordered_layer_plans_blake3"] = Value::Array(Vec::new());
    legacy_plan["load_or_compile"]["workload_blake3"] = Value::String(legacy_workload_blake3);
    let mut webkit_input = assembly_input.clone();
    webkit_input.passport.backend.kind = BackendKindV1::WebkitWebgpu;
    webkit_input.passport.backend.browser_version = Some("26.5".to_owned());
    webkit_input.passport.backend.user_agent = Some("Version/26.5 Safari/605.1.15".to_owned());
    webkit_input.load_or_compile.passport_blake3 = component_blake3(
        &serde_json::to_value(&webkit_input.passport).expect("WebKit passport is serializable"),
    );
    let webkit_plan = browser_cohort_plan_value(&webkit_input);
    let webkit_assembled = AssembledBenchmarkEvidenceV1::assemble(webkit_input)
        .expect("the shared WebKit variant must be a valid signed assembly");
    let mut timestamp_input = assembly_input.clone();
    timestamp_input.passport.backend.features = vec!["timestamp_query".to_owned()];
    timestamp_input.passport.backend.timestamp_query = true;
    timestamp_input.load_or_compile.passport_blake3 = component_blake3(
        &serde_json::to_value(&timestamp_input.passport)
            .expect("timestamp-enabled passport is serializable"),
    );
    let timestamp_plan = browser_cohort_plan_value(&timestamp_input);
    let variant_links = json!({
        "legacy_assembly_blake3": legacy_assembled.assembly_blake3(),
        "legacy_workload_blake3": legacy_plan["load_or_compile"]["workload_blake3"],
        "timestamp_passport_blake3": timestamp_plan["load_or_compile"]["passport_blake3"],
        "webkit_assembly_blake3": webkit_assembled.assembly_blake3(),
        "webkit_passport_blake3": webkit_plan["load_or_compile"]["passport_blake3"],
    });
    let fixture = json!({
        "assembly_request": assembly_request,
        "clock_duration_ns": [1.4, 2.4, 3.4, 4.4, 1.4, 2.4, 3.4, 4.4, 1.4, 2.4, 3.4, 4.4, 1.4, 2.4, 3.4, 4.4],
        "expected_assembly_blake3": assembled.assembly_blake3(),
        "expected_attempt_count": 16,
        "expected_evidence_blake3": evidence["evidence_blake3"],
        "operation_binding": {
            "descriptor": {
                "activation_arena_bytes": workload["residency_plan"]["activation_arena_bytes"],
                "activation_buffer_count": workload["residency_plan"]["activation_buffer_count"],
                "activation_strategy": workload["residency_plan"]["activation_strategy"],
                "allocated_gpu_bytes": workload["residency_plan"]["allocated_gpu_bytes"],
                "expected_output_sha256": workload["checkpoint_sha256"],
                "expected_topology": workload["kernel_variant"]["expected_topology"],
                "index": 777,
                "kernel_variant_id": workload["kernel_variant"]["id"],
                "logical_gpu_bytes": workload["residency_plan"]["logical_gpu_bytes"],
                "main_buffers_bytes": workload["residency_plan"]["main_buffers_bytes"],
                "residency_plan_id": workload["residency_plan"]["id"],
                "schedule_slot": 777,
                "scratch_arena_bytes": workload["residency_plan"]["scratch_arena_bytes"],
                "timestamp_query": false
            },
            "qkv_policy": "required"
        },
        "plan": plan,
        "run_id": "m7d1a3-browser-fused-above-minimum",
        "schema_version": 1,
        "variant_links": variant_links
    });
    let committed_bytes =
        include_bytes!("../../../web/tests/fixtures/m7d1a3_browser_cohort_v1.json");
    let committed: Value = serde_json::from_slice(committed_bytes).unwrap();
    assert_eq!(committed["variant_links"], fixture["variant_links"]);
    assert_eq!(committed, fixture);

    validate_browser_benchmark_cohort_plan_v1(&canonical_value_bytes(&fixture["plan"]))
        .expect("the exact Node fixture plan must cross the real Rust admission boundary");
    let bridged =
        assemble_browser_benchmark_cohort_v1(&canonical_value_bytes(&fixture["assembly_request"]))
            .expect("the exact Node fixture journal must cross the real Rust assembly boundary");
    assert_eq!(bridged, assembled.canonical_assembly_bytes());
}
