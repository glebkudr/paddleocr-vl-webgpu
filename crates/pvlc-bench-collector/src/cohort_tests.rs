use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, VecDeque},
    rc::Rc,
    sync::Arc,
};

use pvlc_bench::{
    AssembledBenchmarkEvidenceV1, BackendIdentityV1, BackendKindV1, BenchmarkClassV1,
    BenchmarkCohortV1, BenchmarkEvidenceAssemblyInputV1, BenchmarkPassportV1, BenchmarkProtocolV1,
    BenchmarkSampleAttemptV1, BenchmarkSampleV1, CorrectnessAnchorV1, DurationObservationV1,
    ExecutionBoundaryV1, ExpectedTopologyV1, GpuTimestampObservationV1, KernelVariantIdentityV1,
    LoadOrCompileObservationV1, ModelIdentityV1, ObservationV1, ResidencyPlanIdentityV1,
    SampleStatusV1, VisionStackWorkloadV1,
};
use pvlc_cpu_ref::{
    KvBlockOrder, LayerNormParameters as CpuLayerNormParameters,
    LinearParameters as CpuLinearParameters, VisionEncoderLayerConfig,
    VisionEncoderLayerParameters as CpuVisionEncoderLayerParameters, VisionEncoderStackConfig,
    vision_encoder_layer_identity_rope_f32, vision_encoder_stack_identity_rope_f32,
};
use pvlc_ir::SemanticGraph;
use pvlc_passes::{
    VisionQkvStackSelection, build_verified_vision_qkv_stack_overlay,
    canonical_synthetic_vision_qkv_tensor_catalog, select_vision_qkv_stack_overlay,
};
use pvlc_runtime_core::{
    VisionEncoderLayerGeometry, VisionEncoderLayerParameters, VisionEncoderStackInvocation,
    VisionLayerNormParameters, VisionLinearParameters, VisionQkvExecutionPolicy,
    VisionQkvFusedTargetLimits, VisionQkvSelectionOutcome, VisionQkvStackExecutionEvidence,
    VisionRopeSpecialization, VisionStackActivationLayoutConfig, VisionStackActivationStrategy,
};
use pvlc_runtime_native::{
    BackendKind, ErrorScopeKind, GpuTimestamp, NativeCapabilities, NativeOptions, NativeRuntime,
    RuntimeError, RuntimeEvent, RuntimeObserver, VisionQkvStackExecution, VisionStackDiagnostics,
    VisionStackExecution,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::*;

const MANIFEST_SHA256: &str = "484f982080f3114c285b9db368396859815f768f713ceb960e7fd409f8d6c48b";
// Independent Node/WebCrypto-compatible SHA-256 of the complete semantic readback:
// checkpoint layer 0, checkpoint layer 1, then final output, all encoded as little-endian f32.
const OUTPUT_SHA256: &str = "c60ed82989381323c2fe6e7249d3358f8a262f0c23a5043058bc66f4e5c47dad";
const FINAL_ONLY_SHA256: &str = "2b967ea9d39a2c3953906e1e4407abb2d4cd0534a421501c180c81cc57ed9da4";
const SEMANTIC_GRAPH_BLAKE3: &str =
    "2b2556c363545dcef569e3e6d0db01967973a081706c8483e1c5af3c7dc5bf73";
const RUN_ID: &str = "m7d1a3a1-native-test-run-0001";
const FIXTURE_LAYER_COUNT: usize = 2;
const CHECKPOINT_LAYERS: &[usize] = &[0, 1];
const FIXTURE_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT: u32 = 256;
const FIXTURE_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE: u32 = 8;
const FIXTURE_MAX_STORAGE_BUFFER_BINDING_SIZE: u64 = 4_294_967_292;
const FIXTURE_MAX_BUFFER_SIZE: u64 = 4_294_967_292;
const FIXTURE_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: u32 = 65_535;
const FIXTURE_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP: u32 = 256;
const FIXTURE_MAX_COMPUTE_WORKGROUP_SIZE_X: u32 = 256;
const FIXTURE_MAX_COMPUTE_WORKGROUP_SIZE_Y: u32 = 256;
const FIXTURE_MAX_COMPUTE_WORKGROUP_SIZE_Z: u32 = 64;
const FIXTURE_MAX_COMPUTE_WORKGROUP_STORAGE_SIZE: u32 = 32_768;
const EXPECTED_CHECKPOINT_0: [f32; 12] = [
    0.125, 0.25, 0.375, 0.5, 0.125, 0.25, 0.375, 0.5, 0.125, 0.25, 0.375, 0.5,
];
const EXPECTED_CHECKPOINT_1: [f32; 12] = [
    0.1875, 0.125, 0.625, 0.125, 0.1875, 0.125, 0.625, 0.125, 0.1875, 0.125, 0.625, 0.125,
];
const EXPECTED_OUTPUT: [f32; 12] = [
    -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75, -0.75, -0.25, 0.25, 0.75,
];
const ATTEMPT_DURATIONS_NS: [u64; 16] = [
    1_607, 1_103, 2_509, 1_307, 2_113, 1_409, 2_719, 1_211, 2_317, 1_013, 2_903, 1_507, 2_411,
    1_709, 2_603, 1_801,
];

#[derive(Clone, Debug, PartialEq, Eq)]
enum CohortEvent {
    ProbeBefore {
        sequence: usize,
    },
    Clock {
        sequence: usize,
        boundary: &'static str,
        value: Option<u64>,
    },
    LegacyOperation {
        sequence: usize,
    },
    QkvOperation {
        sequence: usize,
    },
    RuntimeObserver {
        sequence: usize,
    },
    Validate {
        sequence: usize,
    },
    ProbeAfter {
        sequence: usize,
    },
}

fn hash(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn semantic_readback_sha256(checkpoints: &BTreeMap<usize, Vec<f32>>, output: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for tensor in checkpoints.values() {
        for value in tensor {
            hasher.update(value.to_le_bytes());
        }
    }
    for value in output {
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

fn validation_reference() -> NativeBenchmarkVisionStackValidationReferenceV1 {
    NativeBenchmarkVisionStackValidationReferenceV1 {
        expected_checkpoints: reference_checkpoints(),
        expected_output: EXPECTED_OUTPUT.to_vec(),
        max_abs_error: 0.0,
        accepted: AcceptedVisionStackValidationV1 {
            output_sha256: OUTPUT_SHA256.to_owned(),
            correctness_report_blake3: hash('e'),
            causal_evidence_blake3: hash('f'),
        },
    }
}

fn reference_checkpoints() -> BTreeMap<usize, Vec<f32>> {
    BTreeMap::from([
        (0, EXPECTED_CHECKPOINT_0.to_vec()),
        (1, EXPECTED_CHECKPOINT_1.to_vec()),
    ])
}

fn alternate_validation_reference() -> NativeBenchmarkVisionStackValidationReferenceV1 {
    let (expected_checkpoints, expected_output) =
        independently_computed_reference_for(&alternate_invocation());
    let output_sha256 = semantic_readback_sha256(&expected_checkpoints, &expected_output);
    NativeBenchmarkVisionStackValidationReferenceV1 {
        expected_checkpoints,
        expected_output,
        max_abs_error: 0.0,
        accepted: AcceptedVisionStackValidationV1 {
            output_sha256,
            correctness_report_blake3: hash('7'),
            causal_evidence_blake3: hash('8'),
        },
    }
}

fn observation(value: &str) -> ObservationV1 {
    ObservationV1::Available {
        value: value.to_owned(),
        method: "ProcessInfo.thermalState".to_owned(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FixtureExecutionShape {
    topology: ExpectedTopologyV1,
    activation_buffer_count: u64,
    activation_arena_bytes: u64,
    scratch_arena_bytes: u64,
    main_buffers_bytes: u64,
    logical_gpu_bytes: u64,
    allocated_gpu_bytes: u64,
    max_resident_shard_bytes: u64,
    readback_bytes: u64,
}

fn fixture_execution_shape(variant: &str) -> FixtureExecutionShape {
    let stack = invocation()
        .plan(CHECKPOINT_LAYERS)
        .expect("the cohort fixture invocation must remain accepted by the real planner");
    let activation = stack
        .activation_layout(VisionStackActivationLayoutConfig {
            allow_aliasing: true,
            storage_buffer_offset_alignment: u64::from(FIXTURE_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT),
            arena_alignment: u64::from(FIXTURE_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT),
        })
        .expect("the cohort fixture activation layout must remain accepted");
    let dispatch_count = match variant {
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1 => stack.dispatch_count,
        FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1 => stack
            .dispatch_count
            .checked_sub(2 * FIXTURE_LAYER_COUNT)
            .expect("one fused QKV dispatch replaces three legacy projections per layer"),
        _ => panic!("unsupported cohort fixture variant {variant}"),
    };
    let logical_gpu_bytes = stack
        .resident_weight_bytes
        .checked_add(activation.total_activation_bytes)
        .and_then(|bytes| bytes.checked_add(stack.readback_bytes))
        .expect("the tiny fixture resource total must fit u64");
    FixtureExecutionShape {
        topology: ExpectedTopologyV1 {
            dispatch_count: u64::try_from(dispatch_count).unwrap(),
            compute_pass_count: u64::try_from(stack.compute_pass_count).unwrap(),
            command_buffer_count: 1,
            submission_count: 1,
            map_count: 1,
        },
        activation_buffer_count: u64::try_from(activation.physical_buffer_count).unwrap(),
        activation_arena_bytes: activation.total_activation_bytes,
        scratch_arena_bytes: activation.scratch_arena_bytes,
        main_buffers_bytes: activation.main_buffers_bytes,
        logical_gpu_bytes,
        allocated_gpu_bytes: logical_gpu_bytes,
        max_resident_shard_bytes: stack.resident_weight_bytes,
        readback_bytes: stack.readback_bytes,
    }
}

fn descriptor(variant: &str) -> VisionStackSampleDescriptorV1 {
    let shape = fixture_execution_shape(variant);
    VisionStackSampleDescriptorV1 {
        // These caller-authored values are deliberately hostile. The cohort runner must replace
        // them with its own cohort-local slot before the descriptor reaches the collector.
        index: 777,
        schedule_slot: 777,
        kernel_variant_id: variant.to_owned(),
        residency_plan_id: "bounded-shard-static-alias-v1".to_owned(),
        expected_topology: shape.topology,
        expected_output_sha256: OUTPUT_SHA256.to_owned(),
        logical_gpu_bytes: shape.logical_gpu_bytes,
        allocated_gpu_bytes: shape.allocated_gpu_bytes,
        activation_strategy: "static_arena_alias".to_owned(),
        activation_buffer_count: shape.activation_buffer_count,
        activation_arena_bytes: shape.activation_arena_bytes,
        scratch_arena_bytes: shape.scratch_arena_bytes,
        main_buffers_bytes: shape.main_buffers_bytes,
    }
}

fn canonical_component_blake3(value: &impl Serialize) -> String {
    // serde_json's map is key-sorted without preserve_order, recursively matching the frozen V1
    // object ordering. This is an independent test oracle, not a call back into the assembler.
    let value = serde_json::to_value(value).unwrap();
    let mut bytes = serde_json::to_vec(&value).unwrap();
    bytes.push(b'\n');
    blake3::hash(&bytes).to_hex().to_string()
}

fn native_capability_limits(capabilities: &NativeCapabilities) -> BTreeMap<String, u64> {
    BTreeMap::from([
        ("max_buffer_size".to_owned(), capabilities.max_buffer_size),
        (
            "max_compute_invocations_per_workgroup".to_owned(),
            u64::from(capabilities.max_compute_invocations_per_workgroup),
        ),
        (
            "max_compute_workgroup_size_x".to_owned(),
            u64::from(capabilities.max_compute_workgroup_size_x),
        ),
        (
            "max_compute_workgroup_size_y".to_owned(),
            u64::from(capabilities.max_compute_workgroup_size_y),
        ),
        (
            "max_compute_workgroup_size_z".to_owned(),
            u64::from(capabilities.max_compute_workgroup_size_z),
        ),
        (
            "max_compute_workgroup_storage_size".to_owned(),
            u64::from(capabilities.max_compute_workgroup_storage_size),
        ),
        (
            "max_compute_workgroups_per_dimension".to_owned(),
            u64::from(capabilities.max_compute_workgroups_per_dimension),
        ),
        (
            "max_storage_buffer_binding_size".to_owned(),
            capabilities.max_storage_buffer_binding_size,
        ),
        (
            "max_storage_buffers_per_shader_stage".to_owned(),
            u64::from(capabilities.max_storage_buffers_per_shader_stage),
        ),
        (
            "min_storage_buffer_offset_alignment".to_owned(),
            u64::from(capabilities.min_storage_buffer_offset_alignment),
        ),
    ])
}

fn leaf_plan(
    variant: &str,
    qkv_policy: &str,
    qkv_outcome: &str,
    warmup_count: u32,
    measured_count: u32,
) -> NativeBenchmarkLeafPlanV1 {
    let shape = fixture_execution_shape(variant);
    let limits = native_capability_limits(&fixture_native_capabilities());

    let passport = BenchmarkPassportV1 {
        machine: "MacBook Pro".to_owned(),
        soc: "Apple M4 Pro".to_owned(),
        adapter_name: "Apple M4 Pro".to_owned(),
        physical_memory_bytes: 51_539_607_552,
        os_version: "macOS 26.5".to_owned(),
        os_build: "25F90".to_owned(),
        power_source: observation("ac"),
        power_profile: observation("automatic"),
        low_power_mode: observation("false"),
        thermal_state: observation("nominal"),
        display_attached: observation("true"),
        source_tree_blake3: hash('1'),
        compiler_runtime_blake3: hash('2'),
        wgsl_runtime_blake3: hash('3'),
        collector_blake3: hash('4'),
        rustc_version: "rustc 1.92.0".to_owned(),
        cargo_version: "cargo 1.92.0".to_owned(),
        wgpu_version: "30.0.0".to_owned(),
        build_profile: "release".to_owned(),
        backend: BackendIdentityV1 {
            kind: BackendKindV1::NativeWgpu,
            browser_version: None,
            user_agent: None,
            adapter_backend: "metal".to_owned(),
            features: Vec::new(),
            limits,
            timestamp_query: false,
        },
        model: ModelIdentityV1 {
            revision: "PaddleOCR-VL-1.6@pinned".to_owned(),
            model_lock_blake3: hash('5'),
            pack_blake3: hash('6'),
            manifest_sha256: MANIFEST_SHA256.to_owned(),
            profile: "ocr-clean-latin-l3".to_owned(),
            case_id: "ocr.clean_latin.0001/vision.stack.2".to_owned(),
            input_blake3: hash('8'),
        },
    };
    let workload = VisionStackWorkloadV1 {
        tokens: 3,
        hidden_size: 4,
        layer_count: u32::try_from(FIXTURE_LAYER_COUNT).unwrap(),
        checkpoint_policy: "depth-0-1-final".to_owned(),
        checkpoint_sha256: OUTPUT_SHA256.to_owned(),
        semantic_graph_blake3: (qkv_policy == "required").then(|| SEMANTIC_GRAPH_BLAKE3.to_owned()),
        manifest_sha256: MANIFEST_SHA256.to_owned(),
        ordered_layer_plans_blake3: if qkv_policy == "required" {
            fixture_layer_plan_blake3()
        } else {
            Vec::new()
        },
        qkv_policy: qkv_policy.to_owned(),
        qkv_outcome: qkv_outcome.to_owned(),
        kernel_variant: KernelVariantIdentityV1 {
            id: variant.to_owned(),
            source_set_blake3: hash('a'),
            abi_blake3: hash('b'),
            expected_topology: shape.topology.clone(),
        },
        residency_plan: ResidencyPlanIdentityV1 {
            id: "bounded-shard-static-alias-v1".to_owned(),
            activation_strategy: "static_arena_alias".to_owned(),
            activation_buffer_count: shape.activation_buffer_count,
            activation_arena_bytes: shape.activation_arena_bytes,
            scratch_arena_bytes: shape.scratch_arena_bytes,
            main_buffers_bytes: shape.main_buffers_bytes,
            logical_gpu_bytes: shape.logical_gpu_bytes,
            allocated_gpu_bytes: shape.allocated_gpu_bytes,
            max_resident_shard_bytes: shape.max_resident_shard_bytes,
        },
        readback_policy: match variant {
            LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1 => "depth-0-1-final".to_owned(),
            FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1 => {
                "depth-0-1-final-plus-qkv-canaries".to_owned()
            }
            _ => unreachable!(),
        },
        execution_boundary: ExecutionBoundaryV1::ApiWall,
    };
    let correctness_anchor = CorrectnessAnchorV1 {
        validator_blake3: hash('c'),
        policy_id: "official-l3-existing-envelope-v1".to_owned(),
        expected_checkpoint_sha256: OUTPUT_SHA256.to_owned(),
        causal_validator_blake3: hash('d'),
    };
    let protocol = BenchmarkProtocolV1 {
        class: BenchmarkClassV1::StageMacro,
        build_profile: "release".to_owned(),
        warmup_count,
        measured_count,
        synchronization: "await-complete-map-validate".to_owned(),
        clock_source: "std-instant-monotonic".to_owned(),
        clock_resolution_ns: 1,
        schedule: "single-stable-variant-v1".to_owned(),
        output_validation_policy: "validate-every-sample".to_owned(),
        isolation_policy: "dedicated-process-no-background-load".to_owned(),
        interruption_policy: "reject-any-interruption".to_owned(),
        background_load_policy: "reject-observed-heavy-load".to_owned(),
    };
    let load_or_compile = LoadOrCompileObservationV1 {
        execution_boundary: ExecutionBoundaryV1::LoadOrCompile,
        duration_ns: 9_876_543_210,
        clock_source: protocol.clock_source.clone(),
        clock_resolution_ns: protocol.clock_resolution_ns,
        passport_blake3: canonical_component_blake3(&passport),
        workload_blake3: canonical_component_blake3(&workload),
        protocol_blake3: canonical_component_blake3(&protocol),
        thermal_before: passport.thermal_state.clone(),
        thermal_after: observation("nominal"),
        status: SampleStatusV1::Passed,
    };

    NativeBenchmarkLeafPlanV1 {
        run_id: RUN_ID.to_owned(),
        passport,
        workload,
        correctness_anchor,
        validation_reference: validation_reference(),
        protocol,
        load_or_compile,
        base_descriptor: descriptor(variant),
    }
}

fn refresh_preparation_links(plan: &mut NativeBenchmarkLeafPlanV1) {
    plan.load_or_compile.passport_blake3 = canonical_component_blake3(&plan.passport);
    plan.load_or_compile.workload_blake3 = canonical_component_blake3(&plan.workload);
    plan.load_or_compile.protocol_blake3 = canonical_component_blake3(&plan.protocol);
    plan.load_or_compile.thermal_before = plan.passport.thermal_state.clone();
}

fn linked_legacy_plan(
    mutate: impl FnOnce(&mut NativeBenchmarkLeafPlanV1),
) -> NativeBenchmarkLeafPlanV1 {
    let mut plan = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    mutate(&mut plan);
    refresh_preparation_links(&mut plan);
    plan
}

fn qkv_plan_with_equivalent_static_mutation(
    name: &str,
    legacy_plan: NativeBenchmarkLeafPlanV1,
) -> NativeBenchmarkLeafPlanV1 {
    let mut qkv = leaf_plan(
        FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "required",
        "fused",
        legacy_plan.protocol.warmup_count,
        legacy_plan.protocol.measured_count,
    );
    let legacy_descriptor = legacy_plan.base_descriptor.clone();
    let legacy_residency_strategy = legacy_plan
        .workload
        .residency_plan
        .activation_strategy
        .clone();

    qkv.run_id = legacy_plan.run_id;
    qkv.passport = legacy_plan.passport;
    qkv.workload.checkpoint_policy = legacy_plan.workload.checkpoint_policy;
    qkv.workload.checkpoint_sha256 = legacy_plan.workload.checkpoint_sha256;
    qkv.workload.manifest_sha256 = legacy_plan.workload.manifest_sha256;
    qkv.workload.execution_boundary = legacy_plan.workload.execution_boundary;
    qkv.correctness_anchor = legacy_plan.correctness_anchor;
    qkv.validation_reference = legacy_plan.validation_reference;
    qkv.protocol = legacy_plan.protocol;
    qkv.load_or_compile = legacy_plan.load_or_compile;

    match name {
        "descriptor activation" => {
            qkv.base_descriptor.activation_strategy = legacy_descriptor.activation_strategy;
        }
        "descriptor topology" => qkv.base_descriptor.expected_topology.dispatch_count += 1,
        "descriptor resources" => qkv.base_descriptor.logical_gpu_bytes += 1,
        "descriptor allocated bytes" => qkv.base_descriptor.allocated_gpu_bytes += 4,
        "descriptor activation count" => qkv.base_descriptor.activation_buffer_count += 1,
        "descriptor activation arena" => qkv.base_descriptor.activation_arena_bytes += 4,
        "descriptor scratch bytes" => qkv.base_descriptor.scratch_arena_bytes += 4,
        "descriptor main bytes" => qkv.base_descriptor.main_buffers_bytes += 4,
        "workload activation strategy" => {
            qkv.workload.residency_plan.activation_strategy = legacy_residency_strategy;
        }
        "descriptor output" => {
            qkv.base_descriptor.expected_output_sha256 = legacy_descriptor.expected_output_sha256;
        }
        "descriptor residency" => {
            qkv.base_descriptor.residency_plan_id = legacy_descriptor.residency_plan_id;
        }
        _ => {}
    }

    refresh_preparation_links(&mut qkv);
    if name == "preparation link" {
        qkv.load_or_compile.workload_blake3 = hash('0');
    }
    qkv
}

fn enable_timestamp_query(plan: &mut NativeBenchmarkLeafPlanV1) {
    plan.passport.backend.timestamp_query = true;
    plan.passport.backend.features = vec!["timestamp_query".to_owned()];
    refresh_preparation_links(plan);
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OperationCall {
    kind: &'static str,
    invocation_address: usize,
    checkpoint_layers: Vec<usize>,
    activation_strategy: VisionStackActivationStrategy,
    selection_address: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumericalTensor {
    Output,
    Checkpoint(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumericalValueMutation {
    Drift,
    PositiveInterior,
    NegativeInterior,
    InclusiveBoundary,
    ImmediatelyOutside,
    NonFinite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumericalLengthMutation {
    Short,
    Long,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuntimeFault {
    Operation(usize),
    NumericalValue {
        sequence: usize,
        tensor: NumericalTensor,
        index: usize,
        mutation: NumericalValueMutation,
    },
    NumericalLength {
        sequence: usize,
        tensor: NumericalTensor,
        mutation: NumericalLengthMutation,
    },
    MissingCheckpoint(usize),
    ExtraCheckpoint(usize),
    QkvEvidenceFirst(usize),
    QkvEvidenceSecond(usize),
    QkvEvidenceOrder(usize),
    QkvEvidenceDuplicate(usize),
    QkvEvidenceMissing(usize),
    QkvEvidenceExtra(usize),
    QkvPolicy(usize),
    QkvOutcome(usize),
    ClockRead(usize),
    NonMonotonicClock(usize),
    InvalidDiagnostics(usize),
    InvalidQueueObservation(usize),
    InvalidTimestamp(usize),
    StaleTimestamp(usize),
    Topology(usize),
    Resource(usize),
}

const fn value_fault(
    sequence: usize,
    tensor: NumericalTensor,
    index: usize,
    mutation: NumericalValueMutation,
) -> RuntimeFault {
    RuntimeFault::NumericalValue {
        sequence,
        tensor,
        index,
        mutation,
    }
}

const fn length_fault(
    sequence: usize,
    tensor: NumericalTensor,
    mutation: NumericalLengthMutation,
) -> RuntimeFault {
    RuntimeFault::NumericalLength {
        sequence,
        tensor,
        mutation,
    }
}

fn next_f32_above(value: f32) -> f32 {
    assert!(value.is_finite() && value >= 0.0);
    f32::from_bits(value.to_bits() + 1)
}

fn numerical_execution(
    faults: &[RuntimeFault],
    sequence: usize,
    invocation: &VisionEncoderStackInvocation<'_>,
) -> (BTreeMap<usize, Vec<f32>>, Vec<f32>) {
    let (mut checkpoints, mut output) = independently_computed_reference_for(invocation);
    for fault in faults {
        match *fault {
            RuntimeFault::NumericalValue {
                sequence: fault_sequence,
                tensor,
                index,
                mutation,
            } if fault_sequence == sequence => {
                let values = match tensor {
                    NumericalTensor::Output => &mut output,
                    NumericalTensor::Checkpoint(layer) => checkpoints
                        .get_mut(&layer)
                        .expect("a numerical fault names an authored checkpoint"),
                };
                let value = values
                    .get_mut(index)
                    .expect("a numerical fault names an authored tensor element");
                match mutation {
                    NumericalValueMutation::Drift => *value += 9.0,
                    NumericalValueMutation::PositiveInterior => *value += 0.125,
                    NumericalValueMutation::NegativeInterior => *value -= 0.125,
                    NumericalValueMutation::InclusiveBoundary => *value += 0.25,
                    NumericalValueMutation::ImmediatelyOutside => {
                        *value = next_f32_above(*value + 0.25);
                    }
                    NumericalValueMutation::NonFinite => *value = f32::NAN,
                }
            }
            RuntimeFault::NumericalLength {
                sequence: fault_sequence,
                tensor,
                mutation,
            } if fault_sequence == sequence => {
                let values = match tensor {
                    NumericalTensor::Output => &mut output,
                    NumericalTensor::Checkpoint(layer) => checkpoints
                        .get_mut(&layer)
                        .expect("a length fault names an authored checkpoint"),
                };
                match mutation {
                    NumericalLengthMutation::Short => {
                        values.pop();
                    }
                    NumericalLengthMutation::Long => values.push(123.0),
                }
            }
            _ => {}
        }
    }
    if faults.contains(&RuntimeFault::MissingCheckpoint(sequence)) {
        checkpoints.remove(&1);
    }
    if faults.contains(&RuntimeFault::ExtraCheckpoint(sequence)) {
        checkpoints.insert(2, vec![42.0; output.len()]);
    }

    (checkpoints, output)
}

fn fixture_native_capabilities() -> NativeCapabilities {
    NativeCapabilities {
        adapter_name: "Apple M4 Pro".to_owned(),
        backend: BackendKind::Metal,
        timestamp_query: false,
        min_storage_buffer_offset_alignment: FIXTURE_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT,
        max_storage_buffer_binding_size: FIXTURE_MAX_STORAGE_BUFFER_BINDING_SIZE,
        max_compute_workgroups_per_dimension: FIXTURE_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        max_compute_invocations_per_workgroup: FIXTURE_MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP,
        max_compute_workgroup_size_x: FIXTURE_MAX_COMPUTE_WORKGROUP_SIZE_X,
        max_compute_workgroup_size_y: FIXTURE_MAX_COMPUTE_WORKGROUP_SIZE_Y,
        max_compute_workgroup_size_z: FIXTURE_MAX_COMPUTE_WORKGROUP_SIZE_Z,
        max_compute_workgroup_storage_size: FIXTURE_MAX_COMPUTE_WORKGROUP_STORAGE_SIZE,
        max_storage_buffers_per_shader_stage: FIXTURE_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        max_buffer_size: FIXTURE_MAX_BUFFER_SIZE,
    }
}

fn native_backend_name(backend: BackendKind) -> &'static str {
    match backend {
        BackendKind::Noop => "noop",
        BackendKind::Vulkan => "vulkan",
        BackendKind::Metal => "metal",
        BackendKind::Dx12 => "dx12",
        BackendKind::Gl => "gl",
        BackendKind::BrowserWebGpu => "browser_webgpu",
    }
}

#[derive(Debug)]
struct NoopRuntimeObserver;

impl RuntimeObserver for NoopRuntimeObserver {
    fn on_event(&self, _event: RuntimeEvent) {}
}

struct CohortRuntime {
    events: Rc<RefCell<Vec<CohortEvent>>>,
    clock_readings: RefCell<VecDeque<u64>>,
    clock_read_index: Cell<usize>,
    elapsed_index: Cell<usize>,
    operation_index: Cell<usize>,
    capabilities: NativeCapabilities,
    observer_attached: bool,
    faults: Vec<RuntimeFault>,
    descriptor: VisionStackSampleDescriptorV1,
    calls: RefCell<Vec<OperationCall>>,
}

impl CohortRuntime {
    fn new(
        descriptor: VisionStackSampleDescriptorV1,
        events: Rc<RefCell<Vec<CohortEvent>>>,
    ) -> Self {
        let mut clock_readings = VecDeque::new();
        let mut next_start = 10_000_u64;
        for duration in ATTEMPT_DURATIONS_NS {
            clock_readings.push_back(next_start);
            let end = next_start + duration;
            clock_readings.push_back(end);
            next_start = end + 97;
        }
        Self {
            events,
            clock_readings: RefCell::new(clock_readings),
            clock_read_index: Cell::new(0),
            elapsed_index: Cell::new(0),
            operation_index: Cell::new(0),
            capabilities: fixture_native_capabilities(),
            observer_attached: false,
            faults: Vec::new(),
            descriptor,
            calls: RefCell::new(Vec::new()),
        }
    }

    fn with_fault(
        descriptor: VisionStackSampleDescriptorV1,
        events: Rc<RefCell<Vec<CohortEvent>>>,
        fault: RuntimeFault,
    ) -> Self {
        let mut runtime = Self::new(descriptor, events);
        runtime.faults.push(fault);
        runtime
    }

    fn with_faults(
        descriptor: VisionStackSampleDescriptorV1,
        events: Rc<RefCell<Vec<CohortEvent>>>,
        faults: impl IntoIterator<Item = RuntimeFault>,
    ) -> Self {
        let mut runtime = Self::new(descriptor, events);
        runtime.faults.extend(faults);
        runtime
    }

    fn has_fault(&self, fault: RuntimeFault) -> bool {
        self.faults.contains(&fault)
    }

    fn with_timestamp_query(mut self) -> Self {
        self.capabilities.timestamp_query = true;
        self
    }

    fn with_observer(mut self) -> Self {
        self.observer_attached = true;
        self
    }

    fn with_capability_mutation(mut self, mutate: impl FnOnce(&mut NativeCapabilities)) -> Self {
        mutate(&mut self.capabilities);
        self
    }

    fn next_operation_sequence(&self) -> usize {
        let index = self.operation_index.get();
        self.operation_index.set(index + 1);
        index
    }
}

impl super::sealed::Sealed for CohortRuntime {}

impl NativeCollectorRuntimeV1 for CohortRuntime {
    type MonotonicReading = u64;

    fn timestamp_query(&self) -> bool {
        self.capabilities.timestamp_query
    }

    fn collector_monotonic_now(&self) -> Result<Self::MonotonicReading, String> {
        let read_index = self.clock_read_index.get();
        self.clock_read_index.set(read_index + 1);
        let sequence = read_index / 2;
        let boundary = if read_index.is_multiple_of(2) {
            "start"
        } else {
            "end"
        };
        if self.has_fault(RuntimeFault::ClockRead(read_index)) {
            self.events.borrow_mut().push(CohortEvent::Clock {
                sequence,
                boundary,
                value: None,
            });
            return Err("injected cohort clock failure".to_owned());
        }
        let value = self
            .clock_readings
            .borrow_mut()
            .pop_front()
            .expect("the cohort runner cannot read beyond its fixed schedule");
        self.events.borrow_mut().push(CohortEvent::Clock {
            sequence,
            boundary,
            value: Some(value),
        });
        Ok(value)
    }

    fn collector_elapsed_ns(
        &self,
        started: &Self::MonotonicReading,
        ended: &Self::MonotonicReading,
    ) -> Result<u64, String> {
        let sequence = self.elapsed_index.get();
        self.elapsed_index.set(sequence + 1);
        if self.has_fault(RuntimeFault::NonMonotonicClock(sequence)) {
            return Ok(0);
        }
        ended
            .checked_sub(*started)
            .filter(|duration| *duration > 0 && *duration != u64::MAX)
            .ok_or_else(|| "test cohort clock is invalid".to_owned())
    }
}

impl super::cohort::NativeCohortRuntimeAuthorityV1 for CohortRuntime {
    fn cohort_capabilities(&self) -> &NativeCapabilities {
        &self.capabilities
    }

    fn cohort_has_observer(&self) -> bool {
        self.observer_attached
    }
}

fn diagnostics(
    descriptor: &VisionStackSampleDescriptorV1,
    sequence: usize,
    timestamp_query: bool,
    faults: &[RuntimeFault],
) -> VisionStackDiagnostics {
    let timestamp_sequence = if faults.contains(&RuntimeFault::StaleTimestamp(sequence)) {
        sequence.saturating_sub(1)
    } else {
        sequence
    };
    let timestamp = timestamp_query
        .then_some(GpuTimestamp {
            begin_ticks: 50_000 + u64::try_from(timestamp_sequence).unwrap() * 100,
            end_ticks: 50_010 + u64::try_from(timestamp_sequence).unwrap() * 100,
            period_ns: 1.0,
            duration_ns: 10.0,
        })
        .filter(|_| !faults.contains(&RuntimeFault::InvalidTimestamp(sequence)));
    let mut diagnostics = VisionStackDiagnostics {
        checked_error_scopes: [
            ErrorScopeKind::Validation,
            ErrorScopeKind::OutOfMemory,
            ErrorScopeKind::Internal,
        ],
        captured_errors: Vec::new(),
        queue_wall_time_ns: ATTEMPT_DURATIONS_NS[sequence] / 2,
        timestamp,
        timestamp_fresh: timestamp_query.then_some(true),
        shader_blake3: BTreeMap::new(),
        rope_specialization: VisionRopeSpecialization::Identity,
        layer_count: FIXTURE_LAYER_COUNT,
        checkpoint_layers: CHECKPOINT_LAYERS.to_vec(),
        dispatch_count: usize::try_from(descriptor.expected_topology.dispatch_count).unwrap(),
        compute_pass_count: usize::try_from(descriptor.expected_topology.compute_pass_count)
            .unwrap(),
        submission_count: descriptor.expected_topology.submission_count,
        command_buffer_count: u32::try_from(descriptor.expected_topology.command_buffer_count)
            .unwrap(),
        buffer_allocation_count: 9,
        weight_buffer_count: 1,
        activation_strategy: VisionStackActivationStrategy::StaticArenaAlias,
        activation_buffer_count: 3,
        activation_arena_bytes: descriptor.activation_arena_bytes,
        scratch_arena_bytes: descriptor.scratch_arena_bytes,
        main_buffers_bytes: descriptor.main_buffers_bytes,
        scratch_allocations: Vec::new(),
        readback_buffer_count: 1,
        readback_map_count: u32::try_from(descriptor.expected_topology.map_count).unwrap(),
        readback_bytes: fixture_execution_shape(&descriptor.kernel_variant_id).readback_bytes,
    };
    if faults.contains(&RuntimeFault::InvalidDiagnostics(sequence)) {
        diagnostics.captured_errors.push("injected".to_owned());
    }
    if faults.contains(&RuntimeFault::InvalidQueueObservation(sequence)) {
        diagnostics.queue_wall_time_ns = ATTEMPT_DURATIONS_NS[sequence] + 1;
    }
    if faults.contains(&RuntimeFault::Topology(sequence)) {
        diagnostics.dispatch_count += 1;
    }
    if faults.contains(&RuntimeFault::Resource(sequence)) {
        diagnostics.scratch_arena_bytes += 4;
    }
    diagnostics
}

fn fake_legacy_operation(
    runtime: &CohortRuntime,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
) -> Result<VisionStackExecution, RuntimeError> {
    let sequence = runtime.next_operation_sequence();
    runtime
        .events
        .borrow_mut()
        .push(CohortEvent::LegacyOperation { sequence });
    if runtime.observer_attached {
        runtime
            .events
            .borrow_mut()
            .push(CohortEvent::RuntimeObserver { sequence });
    }
    runtime.calls.borrow_mut().push(OperationCall {
        kind: "legacy",
        invocation_address: std::ptr::from_ref(invocation).addr(),
        checkpoint_layers: checkpoint_layers.to_vec(),
        activation_strategy,
        selection_address: None,
    });
    if runtime.has_fault(RuntimeFault::Operation(sequence)) {
        return Err(RuntimeError::operation("injected cohort operation failure"));
    }
    let (checkpoints, output) = numerical_execution(&runtime.faults, sequence, invocation);
    Ok(VisionStackExecution {
        checkpoints,
        output,
        diagnostics: diagnostics(
            &runtime.descriptor,
            sequence,
            runtime.capabilities.timestamp_query,
            &runtime.faults,
        ),
    })
}

fn fake_qkv_operation(
    runtime: &CohortRuntime,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
    selection: &VisionQkvStackSelection,
) -> Result<VisionQkvStackExecution, RuntimeError> {
    let sequence = runtime.next_operation_sequence();
    runtime
        .events
        .borrow_mut()
        .push(CohortEvent::QkvOperation { sequence });
    if runtime.observer_attached {
        runtime
            .events
            .borrow_mut()
            .push(CohortEvent::RuntimeObserver { sequence });
    }
    runtime.calls.borrow_mut().push(OperationCall {
        kind: "qkv",
        invocation_address: std::ptr::from_ref(invocation).addr(),
        checkpoint_layers: checkpoint_layers.to_vec(),
        activation_strategy,
        selection_address: Some(std::ptr::from_ref(selection).addr()),
    });
    if runtime.has_fault(RuntimeFault::Operation(sequence)) {
        return Err(RuntimeError::operation("injected cohort QKV failure"));
    }
    let (checkpoints, output) = numerical_execution(&runtime.faults, sequence, invocation);
    let mut canonical_layer_plan_blake3: Vec<String> = selection
        .overlay()
        .expect("the test kernel admits only fused selections")
        .layers()
        .iter()
        .map(|layer| layer.canonical_plan_blake3_hex().to_owned())
        .collect();
    if runtime.has_fault(RuntimeFault::QkvEvidenceFirst(sequence)) {
        canonical_layer_plan_blake3[0] = hash('0');
    }
    if runtime.has_fault(RuntimeFault::QkvEvidenceSecond(sequence)) {
        canonical_layer_plan_blake3[1] = hash('0');
    }
    if runtime.has_fault(RuntimeFault::QkvEvidenceOrder(sequence)) {
        canonical_layer_plan_blake3.swap(0, 1);
    }
    if runtime.has_fault(RuntimeFault::QkvEvidenceDuplicate(sequence)) {
        let first = canonical_layer_plan_blake3[0].clone();
        canonical_layer_plan_blake3[1] = first;
    }
    if runtime.has_fault(RuntimeFault::QkvEvidenceMissing(sequence)) {
        canonical_layer_plan_blake3.pop();
    }
    if runtime.has_fault(RuntimeFault::QkvEvidenceExtra(sequence)) {
        canonical_layer_plan_blake3.push(hash('0'));
    }
    let policy = if runtime.has_fault(RuntimeFault::QkvPolicy(sequence)) {
        VisionQkvExecutionPolicy::Preferred
    } else {
        VisionQkvExecutionPolicy::Required
    };
    let outcome = if runtime.has_fault(RuntimeFault::QkvOutcome(sequence)) {
        VisionQkvSelectionOutcome::FallbackUnsupportedTarget
    } else {
        VisionQkvSelectionOutcome::Fused
    };
    Ok(VisionQkvStackExecution {
        checkpoints,
        output,
        diagnostics: diagnostics(
            &runtime.descriptor,
            sequence,
            runtime.capabilities.timestamp_query,
            &runtime.faults,
        ),
        evidence: VisionQkvStackExecutionEvidence {
            policy,
            outcome,
            canonical_layer_plan_blake3,
            pipeline_creations: Vec::new(),
            bind_group_creations: Vec::new(),
            command_encoder_creations: Vec::new(),
            encoded_dispatches: Vec::new(),
            encoded_copies: Vec::new(),
            map_requests: Vec::new(),
            dispatch_count: usize::try_from(runtime.descriptor.expected_topology.dispatch_count)
                .unwrap(),
            compute_pass_count: usize::try_from(
                runtime.descriptor.expected_topology.compute_pass_count,
            )
            .unwrap(),
            command_buffer_count: usize::try_from(
                runtime.descriptor.expected_topology.command_buffer_count,
            )
            .unwrap(),
            submission_count: usize::try_from(
                runtime.descriptor.expected_topology.submission_count,
            )
            .unwrap(),
            map_count: usize::try_from(runtime.descriptor.expected_topology.map_count).unwrap(),
            workspace: None,
            attention_bindings: Vec::new(),
            canaries: Vec::new(),
        },
    })
}

impl NativePublicVisionStackRuntimeV1 for CohortRuntime {
    const LEGACY_PUBLIC_OPERATION_V1: NativeLegacyPublicOperationV1<Self> = fake_legacy_operation;
    const QKV_PUBLIC_OPERATION_V1: NativeQkvPublicOperationV1<Self> = fake_qkv_operation;
}

static ZERO_4: [f32; 4] = [0.0; 4];
static ZERO_7: [f32; 7] = [0.0; 7];
static ZERO_12: [f32; 12] = [0.0; 12];
static ZERO_16: [f32; 16] = [0.0; 16];
static ZERO_28: [f32; 28] = [0.0; 28];
static CU_SEQLENS: [u32; 3] = [0, 1, 3];
static LAYER_0_ATTENTION_OUTPUT_BIAS: [f32; 4] = [0.125, 0.25, 0.375, 0.5];
static LAYER_1_ATTENTION_OUTPUT_BIAS: [f32; 4] = [0.0625, -0.125, 0.25, -0.375];
static POST_NORM_BIAS: [f32; 4] = [-0.75, -0.25, 0.25, 0.75];
static ALTERNATE_LAYER_0_ATTENTION_OUTPUT_BIAS: [f32; 4] = [0.25, -0.125, 0.5, 0.0625];
static ALTERNATE_LAYER_1_ATTENTION_OUTPUT_BIAS: [f32; 4] = [0.125, 0.25, -0.0625, 0.375];
static ALTERNATE_POST_NORM_BIAS: [f32; 4] = [0.5, 0.25, -0.25, -0.5];

const fn fixture_layer_parameters(
    attention_output_bias: &'static [f32],
) -> VisionEncoderLayerParameters<'static> {
    VisionEncoderLayerParameters {
        norm1: VisionLayerNormParameters {
            weight: &ZERO_4,
            bias: &ZERO_4,
        },
        query: VisionLinearParameters {
            weight: &ZERO_16,
            bias: &ZERO_4,
        },
        key: VisionLinearParameters {
            weight: &ZERO_16,
            bias: &ZERO_4,
        },
        value: VisionLinearParameters {
            weight: &ZERO_16,
            bias: &ZERO_4,
        },
        attention_output: VisionLinearParameters {
            weight: &ZERO_16,
            bias: attention_output_bias,
        },
        norm2: VisionLayerNormParameters {
            weight: &ZERO_4,
            bias: &ZERO_4,
        },
        mlp_fc1: VisionLinearParameters {
            weight: &ZERO_28,
            bias: &ZERO_7,
        },
        mlp_fc2: VisionLinearParameters {
            weight: &ZERO_28,
            bias: &ZERO_4,
        },
    }
}

static LAYERS: [VisionEncoderLayerParameters<'static>; FIXTURE_LAYER_COUNT] = [
    fixture_layer_parameters(&LAYER_0_ATTENTION_OUTPUT_BIAS),
    fixture_layer_parameters(&LAYER_1_ATTENTION_OUTPUT_BIAS),
];
static ALTERNATE_LAYERS: [VisionEncoderLayerParameters<'static>; FIXTURE_LAYER_COUNT] = [
    fixture_layer_parameters(&ALTERNATE_LAYER_0_ATTENTION_OUTPUT_BIAS),
    fixture_layer_parameters(&ALTERNATE_LAYER_1_ATTENTION_OUTPUT_BIAS),
];

fn invocation_with(
    layers: &'static [VisionEncoderLayerParameters<'static>],
    post_norm_bias: &'static [f32],
) -> VisionEncoderStackInvocation<'static> {
    VisionEncoderStackInvocation {
        tokens: 3,
        hidden_size: 4,
        attention_heads: 2,
        head_dim: 2,
        intermediate_size: 7,
        layer_norm_epsilon: 0.000_01,
        input: &ZERO_12,
        cu_seqlens: &CU_SEQLENS,
        layer_parameters: layers,
        post_norm: VisionLayerNormParameters {
            weight: &ZERO_4,
            bias: post_norm_bias,
        },
    }
}

fn invocation() -> VisionEncoderStackInvocation<'static> {
    invocation_with(&LAYERS, &POST_NORM_BIAS)
}

fn alternate_invocation() -> VisionEncoderStackInvocation<'static> {
    invocation_with(&ALTERNATE_LAYERS, &ALTERNATE_POST_NORM_BIAS)
}

fn cpu_linear(parameters: VisionLinearParameters<'_>) -> CpuLinearParameters<'_> {
    CpuLinearParameters {
        weight: parameters.weight,
        bias: parameters.bias,
    }
}

fn cpu_norm(parameters: VisionLayerNormParameters<'_>) -> CpuLayerNormParameters<'_> {
    CpuLayerNormParameters {
        weight: parameters.weight,
        bias: parameters.bias,
    }
}

fn cpu_layer(parameters: VisionEncoderLayerParameters<'_>) -> CpuVisionEncoderLayerParameters<'_> {
    CpuVisionEncoderLayerParameters {
        norm1: cpu_norm(parameters.norm1),
        query: cpu_linear(parameters.query),
        key: cpu_linear(parameters.key),
        value: cpu_linear(parameters.value),
        attention_output: cpu_linear(parameters.attention_output),
        norm2: cpu_norm(parameters.norm2),
        mlp_fc1: cpu_linear(parameters.mlp_fc1),
        mlp_fc2: cpu_linear(parameters.mlp_fc2),
    }
}

fn independently_computed_reference_for(
    invocation: &VisionEncoderStackInvocation<'_>,
) -> (BTreeMap<usize, Vec<f32>>, Vec<f32>) {
    let boundaries = invocation
        .cu_seqlens
        .iter()
        .map(|value| usize::try_from(*value).unwrap())
        .collect::<Vec<_>>();
    let layers = invocation
        .layer_parameters
        .iter()
        .copied()
        .map(cpu_layer)
        .collect::<Vec<_>>();
    let layer_config = VisionEncoderLayerConfig {
        tokens: usize::try_from(invocation.tokens).unwrap(),
        hidden_size: usize::try_from(invocation.hidden_size).unwrap(),
        attention_heads: usize::try_from(invocation.attention_heads).unwrap(),
        head_dim: usize::try_from(invocation.head_dim).unwrap(),
        intermediate_size: usize::try_from(invocation.intermediate_size).unwrap(),
        layer_norm_epsilon: invocation.layer_norm_epsilon,
        attention_key_tile: 4,
        attention_order: KvBlockOrder::Forward,
    };
    let trace = vision_encoder_stack_identity_rope_f32(
        invocation.input,
        VisionEncoderStackConfig {
            tokens: layer_config.tokens,
            hidden_size: layer_config.hidden_size,
            layers: layers.len(),
            layer_norm_epsilon: layer_config.layer_norm_epsilon,
        },
        CHECKPOINT_LAYERS,
        cpu_norm(invocation.post_norm),
        |layer, input| {
            vision_encoder_layer_identity_rope_f32(input, layer_config, &boundaries, layers[layer])
                .map(|layer| layer.output)
        },
    )
    .expect("the independent CPU oracle must accept the cohort invocation");
    let checkpoints = trace
        .checkpoints
        .into_iter()
        .map(|checkpoint| (checkpoint.layer_index, checkpoint.values))
        .collect();
    (checkpoints, trace.output)
}

fn independently_computed_reference() -> (BTreeMap<usize, Vec<f32>>, Vec<f32>) {
    independently_computed_reference_for(&invocation())
}

fn fixture_target_limits(min_storage_buffer_offset_alignment: u32) -> VisionQkvFusedTargetLimits {
    VisionQkvFusedTargetLimits {
        min_storage_buffer_offset_alignment,
        max_storage_buffers_per_shader_stage: FIXTURE_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE,
        max_storage_buffer_binding_size: FIXTURE_MAX_STORAGE_BUFFER_BINDING_SIZE,
        max_buffer_size: FIXTURE_MAX_BUFFER_SIZE,
        max_compute_workgroups_per_dimension: FIXTURE_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
    }
}

fn fused_selection_for_geometry(
    policy: VisionQkvExecutionPolicy,
    tokens: u32,
    cu_seqlens: &[u32],
    target: VisionQkvFusedTargetLimits,
) -> VisionQkvStackSelection {
    let geometry = VisionEncoderLayerGeometry {
        tokens,
        hidden_size: 4,
        attention_heads: 2,
        head_dim: 2,
        intermediate_size: 7,
        layer_norm_epsilon: 0.000_01,
        cu_seqlens,
    }
    .plan()
    .unwrap();
    let catalog = canonical_synthetic_vision_qkv_tensor_catalog(FIXTURE_LAYER_COUNT, 4).unwrap();
    select_vision_qkv_stack_overlay(policy, || {
        build_verified_vision_qkv_stack_overlay(
            &SemanticGraph::paddleocr_vl_16(),
            FIXTURE_LAYER_COUNT,
            &geometry,
            &catalog,
            target,
        )
    })
    .unwrap()
}

fn fused_selection(policy: VisionQkvExecutionPolicy) -> VisionQkvStackSelection {
    fused_selection_for_geometry(
        policy,
        3,
        &[0, 1, 3],
        fixture_target_limits(FIXTURE_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT),
    )
}

fn required_fused_selection() -> VisionQkvStackSelection {
    fused_selection(VisionQkvExecutionPolicy::Required)
}

fn alternate_required_fused_selection() -> VisionQkvStackSelection {
    fused_selection_for_geometry(
        VisionQkvExecutionPolicy::Required,
        4,
        &[0, 2, 4],
        fixture_target_limits(FIXTURE_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT),
    )
}

fn wrong_target_required_fused_selection() -> VisionQkvStackSelection {
    let mut target = fixture_target_limits(FIXTURE_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT);
    target.max_storage_buffers_per_shader_stage += 1;
    fused_selection_for_geometry(VisionQkvExecutionPolicy::Required, 3, &[0, 1, 3], target)
}

fn selection_layer_plan_blake3(selection: &VisionQkvStackSelection) -> Vec<String> {
    selection
        .overlay()
        .expect("the fixture selection must be fused")
        .layers()
        .iter()
        .map(|layer| layer.canonical_plan_blake3_hex().to_owned())
        .collect()
}

fn fixture_layer_plan_blake3() -> Vec<String> {
    let selection = required_fused_selection();
    let digests = selection_layer_plan_blake3(&selection);
    assert_eq!(digests.len(), FIXTURE_LAYER_COUNT);
    digests
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProbeFault {
    Before(usize),
    After(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValidationFault {
    Reject(usize),
    CrossLink(usize),
}

fn attempt_observation(sequence: usize, boundary: &str) -> ObservationV1 {
    observation(&format!("attempt-{sequence}-{boundary}"))
}

fn scripted_probe(
    events: Rc<RefCell<Vec<CohortEvent>>>,
    fault: Option<ProbeFault>,
) -> impl FnMut() -> Result<ObservationV1, String> {
    let call_index = Cell::new(0_usize);
    move || {
        let call = call_index.get();
        call_index.set(call + 1);
        let sequence = call / 2;
        if call.is_multiple_of(2) {
            events
                .borrow_mut()
                .push(CohortEvent::ProbeBefore { sequence });
            if fault == Some(ProbeFault::Before(sequence)) {
                return Err("injected before-probe failure".to_owned());
            }
            Ok(attempt_observation(sequence, "before"))
        } else {
            events
                .borrow_mut()
                .push(CohortEvent::ProbeAfter { sequence });
            if fault == Some(ProbeFault::After(sequence)) {
                return Err("injected after-probe failure".to_owned());
            }
            Ok(attempt_observation(sequence, "after"))
        }
    }
}

fn digest_for(sequence: usize, domain: u64) -> String {
    format!("{:064x}", domain + u64::try_from(sequence).unwrap())
}

fn validation_for(sequence: usize) -> AcceptedVisionStackValidationV1 {
    AcceptedVisionStackValidationV1 {
        output_sha256: OUTPUT_SHA256.to_owned(),
        correctness_report_blake3: digest_for(sequence, 0x1_000),
        causal_evidence_blake3: digest_for(sequence, 0x2_000),
    }
}

fn validate_sequence(
    output: &[f32],
    sequence: usize,
    events: &Rc<RefCell<Vec<CohortEvent>>>,
    fault: Option<ValidationFault>,
) -> Result<AcceptedVisionStackValidationV1, String> {
    events.borrow_mut().push(CohortEvent::Validate { sequence });
    assert_eq!(output, EXPECTED_OUTPUT);
    if fault == Some(ValidationFault::Reject(sequence)) {
        return Err("injected checkpoint mismatch".to_owned());
    }
    let mut accepted = validation_for(sequence);
    if fault == Some(ValidationFault::CrossLink(sequence)) {
        accepted.output_sha256 = hash('0');
    }
    Ok(accepted)
}

fn legacy_validator(
    events: Rc<RefCell<Vec<CohortEvent>>>,
    fault: Option<ValidationFault>,
) -> impl FnMut(&VisionStackExecution) -> Result<AcceptedVisionStackValidationV1, String> {
    let sequence = Cell::new(0_usize);
    move |execution| {
        let index = sequence.get();
        sequence.set(index + 1);
        validate_sequence(&execution.output, index, &events, fault)
    }
}

fn qkv_validator(
    events: Rc<RefCell<Vec<CohortEvent>>>,
    fault: Option<ValidationFault>,
) -> impl FnMut(&VisionQkvStackExecution) -> Result<AcceptedVisionStackValidationV1, String> {
    let sequence = Cell::new(0_usize);
    move |execution| {
        let index = sequence.get();
        sequence.set(index + 1);
        assert_eq!(
            execution.evidence.policy,
            VisionQkvExecutionPolicy::Required
        );
        assert_eq!(execution.evidence.outcome, VisionQkvSelectionOutcome::Fused);
        assert_eq!(
            execution.evidence.canonical_layer_plan_blake3,
            fixture_layer_plan_blake3()
        );
        validate_sequence(&execution.output, index, &events, fault)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CohortOperation {
    Legacy,
    Qkv,
}

impl CohortOperation {
    const ALL: [Self; 2] = [Self::Legacy, Self::Qkv];

    fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Qkv => "qkv",
        }
    }

    fn variant(self) -> &'static str {
        match self {
            Self::Legacy => LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
            Self::Qkv => FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1,
        }
    }

    fn plan(self, warmup_count: u32, measured_count: u32) -> NativeBenchmarkLeafPlanV1 {
        match self {
            Self::Legacy => leaf_plan(
                self.variant(),
                "disabled",
                "disabled",
                warmup_count,
                measured_count,
            ),
            Self::Qkv => leaf_plan(
                self.variant(),
                "required",
                "fused",
                warmup_count,
                measured_count,
            ),
        }
    }
}

fn alternate_leaf_plan(operation: CohortOperation) -> NativeBenchmarkLeafPlanV1 {
    let mut plan = operation.plan(3, 10);
    let validation_reference = alternate_validation_reference();
    let output_sha256 = validation_reference.accepted.output_sha256.clone();
    plan.passport.model.revision = "PaddleOCR-VL-1.6@alternate-test-fixture".to_owned();
    plan.passport.model.model_lock_blake3 = hash('7');
    plan.passport.model.pack_blake3 = hash('8');
    plan.passport.model.manifest_sha256 = hash('9');
    plan.passport.model.case_id = "ocr.clean_latin.0002/vision.stack.2".to_owned();
    plan.workload.manifest_sha256 = plan.passport.model.manifest_sha256.clone();
    plan.workload.checkpoint_sha256 = output_sha256.clone();
    plan.correctness_anchor.expected_checkpoint_sha256 = output_sha256.clone();
    plan.validation_reference = validation_reference;
    plan.base_descriptor.expected_output_sha256 = output_sha256;
    refresh_preparation_links(&mut plan);
    plan
}

fn run_test_cohort(
    operation: CohortOperation,
    runtime: &CohortRuntime,
    plan: NativeBenchmarkLeafPlanV1,
    events: Rc<RefCell<Vec<CohortEvent>>>,
    probe_fault: Option<ProbeFault>,
    validation_fault: Option<ValidationFault>,
) -> Result<NativeBenchmarkCohortSuccessV1, NativeBenchmarkCohortFailureV1> {
    match operation {
        CohortOperation::Legacy => run_native_public_legacy_benchmark_cohort_v1(
            runtime,
            plan,
            &invocation(),
            CHECKPOINT_LAYERS,
            VisionStackActivationStrategy::StaticArenaAlias,
            scripted_probe(Rc::clone(&events), probe_fault),
            legacy_validator(events, validation_fault),
        ),
        CohortOperation::Qkv => {
            let selection = required_fused_selection();
            run_native_public_qkv_benchmark_cohort_v1(
                runtime,
                plan,
                &invocation(),
                CHECKPOINT_LAYERS,
                VisionStackActivationStrategy::StaticArenaAlias,
                &selection,
                scripted_probe(Rc::clone(&events), probe_fault),
                qkv_validator(events, validation_fault),
            )
        }
    }
}

fn run_concrete_validator_cohort(
    operation: CohortOperation,
    runtime: &CohortRuntime,
    plan: NativeBenchmarkLeafPlanV1,
    events: Rc<RefCell<Vec<CohortEvent>>>,
    validator: &mut NativeBenchmarkVisionStackValidatorV1,
) -> Result<NativeBenchmarkCohortSuccessV1, NativeBenchmarkCohortFailureV1> {
    run_concrete_validator_cohort_with_invocation(
        operation,
        runtime,
        plan,
        &invocation(),
        events,
        validator,
    )
}

fn run_concrete_validator_cohort_with_invocation(
    operation: CohortOperation,
    runtime: &CohortRuntime,
    plan: NativeBenchmarkLeafPlanV1,
    invocation: &VisionEncoderStackInvocation<'_>,
    events: Rc<RefCell<Vec<CohortEvent>>>,
    validator: &mut NativeBenchmarkVisionStackValidatorV1,
) -> Result<NativeBenchmarkCohortSuccessV1, NativeBenchmarkCohortFailureV1> {
    match operation {
        CohortOperation::Legacy => run_native_public_legacy_benchmark_cohort_v1(
            runtime,
            plan,
            invocation,
            CHECKPOINT_LAYERS,
            VisionStackActivationStrategy::StaticArenaAlias,
            scripted_probe(events, None),
            validator,
        ),
        CohortOperation::Qkv => {
            let selection = required_fused_selection();
            run_native_public_qkv_benchmark_cohort_v1(
                runtime,
                plan,
                invocation,
                CHECKPOINT_LAYERS,
                VisionStackActivationStrategy::StaticArenaAlias,
                &selection,
                scripted_probe(events, None),
                validator,
            )
        }
    }
}

fn schedule_identity(position: usize, warmup_count: usize) -> (BenchmarkCohortV1, u32) {
    if position == 0 {
        (BenchmarkCohortV1::Cold, 0)
    } else if position <= warmup_count {
        (BenchmarkCohortV1::Warmup, (position - 1) as u32)
    } else {
        (
            BenchmarkCohortV1::Measured,
            (position - warmup_count - 1) as u32,
        )
    }
}

fn expected_sample(variant: &str, sequence: usize, slot: u32) -> BenchmarkSampleV1 {
    let duration_ns = ATTEMPT_DURATIONS_NS[sequence];
    let shape = fixture_execution_shape(variant);
    BenchmarkSampleV1 {
        index: slot,
        schedule_slot: slot,
        kernel_variant_id: variant.to_owned(),
        residency_plan_id: "bounded-shard-static-alias-v1".to_owned(),
        api_wall_ns: duration_ns,
        queue_wall: DurationObservationV1::Available {
            duration_ns: duration_ns / 2,
        },
        gpu_timestamp: GpuTimestampObservationV1::Unavailable {
            reason: "native runtime timestamp-query feature unavailable".to_owned(),
        },
        topology: shape.topology,
        output_sha256: OUTPUT_SHA256.to_owned(),
        correctness_report_blake3: digest_for(sequence, 0x1_000),
        causal_evidence_blake3: digest_for(sequence, 0x2_000),
        logical_gpu_bytes: shape.logical_gpu_bytes,
        allocated_gpu_bytes: shape.allocated_gpu_bytes,
        thermal_before: attempt_observation(sequence, "before"),
        thermal_after: attempt_observation(sequence, "after"),
        status: SampleStatusV1::Passed,
    }
}

fn expected_concrete_validator_sample(
    variant: &str,
    sequence: usize,
    slot: u32,
) -> BenchmarkSampleV1 {
    expected_concrete_validator_sample_for_reference(
        variant,
        sequence,
        slot,
        &validation_reference(),
    )
}

fn expected_concrete_validator_sample_for_reference(
    variant: &str,
    sequence: usize,
    slot: u32,
    reference: &NativeBenchmarkVisionStackValidationReferenceV1,
) -> BenchmarkSampleV1 {
    let mut sample = expected_sample(variant, sequence, slot);
    sample.output_sha256 = reference.accepted.output_sha256.clone();
    sample.correctness_report_blake3 = reference.accepted.correctness_report_blake3.clone();
    sample.causal_evidence_blake3 = reference.accepted.causal_evidence_blake3.clone();
    sample
}

fn expected_timestamp_sample(variant: &str, sequence: usize, slot: u32) -> BenchmarkSampleV1 {
    let mut sample = expected_sample(variant, sequence, slot);
    sample.gpu_timestamp = GpuTimestampObservationV1::Available {
        begin_ticks: 50_000 + u64::try_from(sequence).unwrap() * 100,
        end_ticks: 50_010 + u64::try_from(sequence).unwrap() * 100,
        period_ns: "1".to_owned(),
        duration_ns: 10,
    };
    sample
}

fn expected_passed_attempts(
    variant: &str,
    warmup_count: usize,
    attempt_count: usize,
) -> Vec<BenchmarkSampleAttemptV1> {
    (0..attempt_count)
        .map(|sequence| {
            let (cohort, slot) = schedule_identity(sequence, warmup_count);
            BenchmarkSampleAttemptV1::Passed {
                sequence: sequence as u32,
                cohort,
                planned_slot: slot,
                sample: expected_sample(variant, sequence, slot),
            }
        })
        .collect()
}

fn expected_concrete_validator_attempts(
    variant: &str,
    warmup_count: usize,
    attempt_count: usize,
) -> Vec<BenchmarkSampleAttemptV1> {
    expected_concrete_validator_attempts_for_reference(
        variant,
        warmup_count,
        attempt_count,
        &validation_reference(),
    )
}

fn expected_concrete_validator_attempts_for_reference(
    variant: &str,
    warmup_count: usize,
    attempt_count: usize,
    reference: &NativeBenchmarkVisionStackValidationReferenceV1,
) -> Vec<BenchmarkSampleAttemptV1> {
    (0..attempt_count)
        .map(|sequence| {
            let (cohort, slot) = schedule_identity(sequence, warmup_count);
            BenchmarkSampleAttemptV1::Passed {
                sequence: sequence as u32,
                cohort,
                planned_slot: slot,
                sample: expected_concrete_validator_sample_for_reference(
                    variant, sequence, slot, reference,
                ),
            }
        })
        .collect()
}

fn expected_event_prefix(operation: &'static str, complete_attempts: usize) -> Vec<CohortEvent> {
    let mut events = Vec::new();
    let mut next_start = 10_000_u64;
    for (sequence, duration) in ATTEMPT_DURATIONS_NS
        .iter()
        .copied()
        .take(complete_attempts)
        .enumerate()
    {
        events.push(CohortEvent::ProbeBefore { sequence });
        events.push(CohortEvent::Clock {
            sequence,
            boundary: "start",
            value: Some(next_start),
        });
        events.push(match operation {
            "legacy" => CohortEvent::LegacyOperation { sequence },
            "qkv" => CohortEvent::QkvOperation { sequence },
            _ => unreachable!(),
        });
        events.push(CohortEvent::Validate { sequence });
        let end = next_start + duration;
        events.push(CohortEvent::Clock {
            sequence,
            boundary: "end",
            value: Some(end),
        });
        events.push(CohortEvent::ProbeAfter { sequence });
        next_start = end + 97;
    }
    events
}

fn expected_concrete_validator_event_prefix(
    operation: &'static str,
    complete_attempts: usize,
) -> Vec<CohortEvent> {
    expected_event_prefix(operation, complete_attempts)
        .into_iter()
        .filter(|event| !matches!(event, CohortEvent::Validate { .. }))
        .collect()
}

fn expected_concrete_validation_failure_events(
    operation: &'static str,
    sequence: usize,
) -> Vec<CohortEvent> {
    let mut events = expected_concrete_validator_event_prefix(operation, sequence);
    let start = clock_start_for(sequence);
    events.extend([
        CohortEvent::ProbeBefore { sequence },
        CohortEvent::Clock {
            sequence,
            boundary: "start",
            value: Some(start),
        },
        operation_event(operation, sequence),
        CohortEvent::Clock {
            sequence,
            boundary: "end",
            value: Some(start + ATTEMPT_DURATIONS_NS[sequence]),
        },
    ]);
    events
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TerminalEffect {
    ProbeBefore,
    OpeningClock,
    Execution,
    Validation,
    ClosingClock,
    NonMonotonicClock,
    ProbeAfter,
    PostCollection,
    ExecutionAndClosingClock,
}

fn operation_event(operation: &'static str, sequence: usize) -> CohortEvent {
    match operation {
        "legacy" => CohortEvent::LegacyOperation { sequence },
        "qkv" => CohortEvent::QkvOperation { sequence },
        _ => unreachable!(),
    }
}

fn expected_terminal_events(
    operation: &'static str,
    sequence: usize,
    terminal: TerminalEffect,
) -> Vec<CohortEvent> {
    if terminal == TerminalEffect::PostCollection {
        return expected_event_prefix(operation, sequence + 1);
    }

    let mut events = expected_event_prefix(operation, sequence);
    events.push(CohortEvent::ProbeBefore { sequence });
    if terminal == TerminalEffect::ProbeBefore {
        return events;
    }

    let start = clock_start_for(sequence);
    events.push(CohortEvent::Clock {
        sequence,
        boundary: "start",
        value: (terminal != TerminalEffect::OpeningClock).then_some(start),
    });
    if terminal == TerminalEffect::OpeningClock {
        return events;
    }

    events.push(operation_event(operation, sequence));
    if terminal == TerminalEffect::Execution || terminal == TerminalEffect::ExecutionAndClosingClock
    {
        events.push(CohortEvent::Clock {
            sequence,
            boundary: "end",
            value: (terminal == TerminalEffect::Execution)
                .then_some(start + ATTEMPT_DURATIONS_NS[sequence]),
        });
        return events;
    }

    events.push(CohortEvent::Validate { sequence });
    if matches!(
        terminal,
        TerminalEffect::Validation
            | TerminalEffect::ClosingClock
            | TerminalEffect::NonMonotonicClock
    ) {
        events.push(CohortEvent::Clock {
            sequence,
            boundary: "end",
            value: (terminal != TerminalEffect::ClosingClock)
                .then_some(start + ATTEMPT_DURATIONS_NS[sequence]),
        });
        return events;
    }

    events.push(CohortEvent::Clock {
        sequence,
        boundary: "end",
        value: Some(start + ATTEMPT_DURATIONS_NS[sequence]),
    });
    events.push(CohortEvent::ProbeAfter { sequence });
    assert_eq!(terminal, TerminalEffect::ProbeAfter);
    events
}

fn clock_start_for(sequence: usize) -> u64 {
    10_000
        + ATTEMPT_DURATIONS_NS
            .iter()
            .take(sequence)
            .map(|duration| duration + 97)
            .sum::<u64>()
}

fn failed_attempt(sequence: usize, warmup_count: usize, code: &str) -> BenchmarkSampleAttemptV1 {
    let (cohort, planned_slot) = schedule_identity(sequence, warmup_count);
    BenchmarkSampleAttemptV1::Failed {
        sequence: sequence as u32,
        cohort,
        planned_slot,
        code: code.to_owned(),
    }
}

fn assert_failure_artifact(
    failure: &NativeBenchmarkCohortFailureV1,
    expected_run_id: &str,
    phase: &str,
    code: &str,
    expected_attempt_count: u64,
    expected_attempts: &[BenchmarkSampleAttemptV1],
) {
    assert_eq!(failure.run_id(), expected_run_id);
    assert_eq!(
        failure.phase(),
        match phase {
            "static_admission" => NativeBenchmarkCohortFailurePhaseV1::StaticAdmission,
            "attempt" => NativeBenchmarkCohortFailurePhaseV1::Attempt,
            _ => panic!("unsupported test failure phase {phase}"),
        }
    );
    assert_eq!(failure.code(), code);
    assert_eq!(failure.expected_attempt_count(), expected_attempt_count);
    assert_eq!(failure.attempt_log(), expected_attempts);
    let bytes = failure.canonical_bytes();
    assert_eq!(bytes.last(), Some(&b'\n'));
    assert_eq!(bytes.iter().filter(|byte| **byte == b'\n').count(), 1);
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let mut independently_canonical = serde_json::to_vec(&value).unwrap();
    independently_canonical.push(b'\n');
    assert_eq!(
        bytes, independently_canonical,
        "failure bytes must be compact, key-sorted, duplicate-free canonical JSON"
    );
    assert_eq!(
        value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [
            "attempt_log",
            "expected_attempt_count",
            "failure_blake3",
            "failure_code",
            "phase",
            "run_id",
            "schema_version",
            "status",
        ]
    );
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["status"], "failed");
    assert_eq!(value["phase"], phase);
    assert_eq!(value["failure_code"], code);
    assert_eq!(value["run_id"], expected_run_id);
    assert_eq!(value["expected_attempt_count"], expected_attempt_count);
    assert_eq!(
        serde_json::from_value::<Vec<BenchmarkSampleAttemptV1>>(value["attempt_log"].clone())
            .unwrap(),
        expected_attempts
    );
    let supplied_hash = value["failure_blake3"].as_str().unwrap();
    let mut unsigned = value.clone();
    unsigned.as_object_mut().unwrap().remove("failure_blake3");
    assert_eq!(supplied_hash, canonical_component_blake3(&unsigned));
    let object = value.as_object().unwrap();
    for forbidden in ["evidence", "evidence_blake3", "summary"] {
        assert!(
            !object.contains_key(forbidden),
            "failure artifact leaked top-level {forbidden}"
        );
    }
    let serialized = serde_json::to_string(&value).unwrap();
    for forbidden in ["speedup", "winner", "throughput", "resident_memory"] {
        assert!(
            !serialized.contains(forbidden),
            "failure artifact leaked {forbidden}"
        );
    }
    assert!(AssembledBenchmarkEvidenceV1::parse_canonical(&bytes).is_err());
}

fn assert_attempt_identity(
    attempt: &BenchmarkSampleAttemptV1,
    sequence: u32,
    cohort: BenchmarkCohortV1,
    slot: u32,
) {
    match attempt {
        BenchmarkSampleAttemptV1::Passed {
            sequence: actual_sequence,
            cohort: actual_cohort,
            planned_slot,
            sample,
        } => {
            assert_eq!(
                (*actual_sequence, *actual_cohort, *planned_slot),
                (sequence, cohort, slot)
            );
            assert_eq!((sample.index, sample.schedule_slot), (slot, slot));
        }
        BenchmarkSampleAttemptV1::Failed { .. } => panic!("expected a passed attempt"),
    }
}

fn assert_completed_matches_plan(
    completed: &NativeBenchmarkCohortSuccessV1,
    expected_plan: &NativeBenchmarkLeafPlanV1,
    expected_attempts: &[BenchmarkSampleAttemptV1],
) {
    assert_eq!(completed.run_id(), expected_plan.run_id);
    assert_eq!(completed.attempt_count(), expected_attempts.len());
    let expected_assembly =
        AssembledBenchmarkEvidenceV1::assemble(BenchmarkEvidenceAssemblyInputV1 {
            passport: expected_plan.passport.clone(),
            workload: expected_plan.workload.clone(),
            correctness_anchor: expected_plan.correctness_anchor.clone(),
            protocol: expected_plan.protocol.clone(),
            load_or_compile: expected_plan.load_or_compile.clone(),
            attempt_log: expected_attempts.to_vec(),
        })
        .unwrap();
    assert_eq!(
        completed.assembled().canonical_assembly_bytes(),
        expected_assembly.canonical_assembly_bytes()
    );
    let reparsed = AssembledBenchmarkEvidenceV1::parse_canonical(
        &completed.assembled().canonical_assembly_bytes(),
    )
    .unwrap();
    assert_eq!(reparsed, *completed.assembled());
}

#[test]
fn legacy_runner_authors_the_complete_above_minimum_schedule_and_directly_assembles_it() {
    let plan = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        4,
        11,
    );
    let expected_plan = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        4,
        11,
    );
    let events = Rc::new(RefCell::new(Vec::new()));
    let runtime = CohortRuntime::new(plan.base_descriptor.clone(), Rc::clone(&events));
    let invocation = invocation();
    let checkpoints = CHECKPOINT_LAYERS.to_vec();

    let completed = run_native_public_legacy_benchmark_cohort_v1(
        &runtime,
        plan,
        &invocation,
        &checkpoints,
        VisionStackActivationStrategy::StaticArenaAlias,
        scripted_probe(Rc::clone(&events), None),
        legacy_validator(Rc::clone(&events), None),
    )
    .expect("the exact complete legacy cohort must assemble");

    assert_eq!(runtime.calls.borrow().len(), 16);
    assert_eq!(*events.borrow(), expected_event_prefix("legacy", 16));
    assert!(runtime.calls.borrow().iter().all(|call| {
        call.kind == "legacy"
            && call.invocation_address == std::ptr::from_ref(&invocation).addr()
            && call.checkpoint_layers == checkpoints
            && call.activation_strategy == VisionStackActivationStrategy::StaticArenaAlias
            && call.selection_address.is_none()
    }));

    let canonical = completed.assembled().canonical_assembly_bytes();
    let value: serde_json::Value = serde_json::from_slice(&canonical).unwrap();
    let attempts: Vec<BenchmarkSampleAttemptV1> =
        serde_json::from_value(value["attempt_log"].clone()).unwrap();
    let expected_attempts =
        expected_passed_attempts(LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1, 4, 16);
    assert_eq!(attempts, expected_attempts);
    assert_attempt_identity(&attempts[0], 0, BenchmarkCohortV1::Cold, 0);
    assert_attempt_identity(&attempts[1], 1, BenchmarkCohortV1::Warmup, 0);
    assert_attempt_identity(&attempts[4], 4, BenchmarkCohortV1::Warmup, 3);
    assert_attempt_identity(&attempts[5], 5, BenchmarkCohortV1::Measured, 0);
    assert_attempt_identity(&attempts[15], 15, BenchmarkCohortV1::Measured, 10);
    assert_eq!(completed.assembled().evidence().summary().count, 11);
    assert_completed_matches_plan(&completed, &expected_plan, &expected_attempts);
}

#[test]
fn fused_runner_reuses_one_exact_required_selection_for_every_authored_attempt() {
    let plan = leaf_plan(
        FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "required",
        "fused",
        4,
        11,
    );
    let expected_plan = leaf_plan(
        FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "required",
        "fused",
        4,
        11,
    );
    let events = Rc::new(RefCell::new(Vec::new()));
    let runtime = CohortRuntime::new(plan.base_descriptor.clone(), Rc::clone(&events));
    let invocation = invocation();
    let checkpoints = CHECKPOINT_LAYERS.to_vec();
    let selection = required_fused_selection();
    let selection_address = std::ptr::from_ref(&selection).addr();
    let overlay = selection.overlay().expect("Required must select fused");
    assert_eq!(overlay.layers().len(), FIXTURE_LAYER_COUNT);
    assert_ne!(
        overlay.layers()[0].canonical_plan_blake3_hex(),
        overlay.layers()[1].canonical_plan_blake3_hex(),
        "the ordered-vector fixture must contain two independently identifiable layer plans",
    );
    assert_eq!(
        overlay.target_limits(),
        fixture_target_limits(FIXTURE_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT)
    );
    assert_eq!(
        expected_plan.workload.ordered_layer_plans_blake3,
        overlay
            .layers()
            .iter()
            .map(|layer| layer.canonical_plan_blake3_hex().to_owned())
            .collect::<Vec<_>>()
    );

    let completed = run_native_public_qkv_benchmark_cohort_v1(
        &runtime,
        plan,
        &invocation,
        &checkpoints,
        VisionStackActivationStrategy::StaticArenaAlias,
        &selection,
        scripted_probe(Rc::clone(&events), None),
        qkv_validator(Rc::clone(&events), None),
    )
    .expect("the exact fused cohort must assemble");

    assert_eq!(runtime.calls.borrow().len(), 16);
    assert_eq!(*events.borrow(), expected_event_prefix("qkv", 16));
    assert!(runtime.calls.borrow().iter().all(|call| {
        call.kind == "qkv"
            && call.invocation_address == std::ptr::from_ref(&invocation).addr()
            && call.checkpoint_layers == checkpoints
            && call.activation_strategy == VisionStackActivationStrategy::StaticArenaAlias
            && call.selection_address == Some(selection_address)
    }));
    let value: serde_json::Value =
        serde_json::from_slice(&completed.assembled().canonical_assembly_bytes()).unwrap();
    let attempts: Vec<BenchmarkSampleAttemptV1> =
        serde_json::from_value(value["attempt_log"].clone()).unwrap();
    let expected_attempts =
        expected_passed_attempts(FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1, 4, 16);
    assert_eq!(attempts, expected_attempts);
    assert_eq!(completed.assembled().evidence().summary().count, 11);
    assert_completed_matches_plan(&completed, &expected_plan, &expected_attempts);
}

#[test]
fn minimum_schedule_stops_after_exactly_fourteen_successful_attempts() {
    let plan = CohortOperation::Legacy.plan(3, 10);
    let expected_plan = CohortOperation::Legacy.plan(3, 10);
    let events = Rc::new(RefCell::new(Vec::new()));
    let runtime = CohortRuntime::new(plan.base_descriptor.clone(), Rc::clone(&events));

    let completed = run_test_cohort(
        CohortOperation::Legacy,
        &runtime,
        plan,
        Rc::clone(&events),
        None,
        None,
    )
    .expect("the exact minimum stage-macro schedule must complete");

    assert_eq!(runtime.calls.borrow().len(), 14);
    assert_eq!(runtime.operation_index.get(), 14);
    assert_eq!(runtime.clock_read_index.get(), 28);
    assert_eq!(*events.borrow(), expected_event_prefix("legacy", 14));
    let expected_attempts =
        expected_passed_attempts(LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1, 3, 14);
    assert_completed_matches_plan(&completed, &expected_plan, &expected_attempts);
}

#[derive(Clone, Copy, Debug)]
enum RuntimeCapabilityMutation {
    AdapterName,
    Backend,
    TimestampQuery,
    MinStorageBufferOffsetAlignment,
    MaxStorageBufferBindingSize,
    MaxComputeWorkgroupsPerDimension,
    MaxComputeInvocationsPerWorkgroup,
    MaxComputeWorkgroupSizeX,
    MaxComputeWorkgroupSizeY,
    MaxComputeWorkgroupSizeZ,
    MaxComputeWorkgroupStorageSize,
    MaxStorageBuffersPerShaderStage,
    MaxBufferSize,
}

impl RuntimeCapabilityMutation {
    const ALL: [Self; 13] = [
        Self::AdapterName,
        Self::Backend,
        Self::TimestampQuery,
        Self::MinStorageBufferOffsetAlignment,
        Self::MaxStorageBufferBindingSize,
        Self::MaxComputeWorkgroupsPerDimension,
        Self::MaxComputeInvocationsPerWorkgroup,
        Self::MaxComputeWorkgroupSizeX,
        Self::MaxComputeWorkgroupSizeY,
        Self::MaxComputeWorkgroupSizeZ,
        Self::MaxComputeWorkgroupStorageSize,
        Self::MaxStorageBuffersPerShaderStage,
        Self::MaxBufferSize,
    ];

    fn apply(self, capabilities: &mut NativeCapabilities) {
        match self {
            Self::AdapterName => capabilities.adapter_name.push_str(" forged"),
            Self::Backend => capabilities.backend = BackendKind::Vulkan,
            Self::TimestampQuery => capabilities.timestamp_query = !capabilities.timestamp_query,
            Self::MinStorageBufferOffsetAlignment => {
                capabilities.min_storage_buffer_offset_alignment *= 2;
            }
            Self::MaxStorageBufferBindingSize => {
                capabilities.max_storage_buffer_binding_size += 4;
            }
            Self::MaxComputeWorkgroupsPerDimension => {
                capabilities.max_compute_workgroups_per_dimension += 1;
            }
            Self::MaxComputeInvocationsPerWorkgroup => {
                capabilities.max_compute_invocations_per_workgroup += 1;
            }
            Self::MaxComputeWorkgroupSizeX => capabilities.max_compute_workgroup_size_x += 1,
            Self::MaxComputeWorkgroupSizeY => capabilities.max_compute_workgroup_size_y += 1,
            Self::MaxComputeWorkgroupSizeZ => capabilities.max_compute_workgroup_size_z += 1,
            Self::MaxComputeWorkgroupStorageSize => {
                capabilities.max_compute_workgroup_storage_size += 1;
            }
            Self::MaxStorageBuffersPerShaderStage => {
                capabilities.max_storage_buffers_per_shader_stage += 1;
            }
            Self::MaxBufferSize => capabilities.max_buffer_size += 4,
        }
    }
}

fn assert_static_runtime_binding_failure(
    operation: CohortOperation,
    plan: NativeBenchmarkLeafPlanV1,
    runtime: &CohortRuntime,
    events: &Rc<RefCell<Vec<CohortEvent>>>,
    case: &str,
) {
    let failure = run_test_cohort(operation, runtime, plan, Rc::clone(events), None, None)
        .expect_err("runtime authority drift must reject the leaf before one physical attempt");
    assert_eq!(
        failure.phase(),
        NativeBenchmarkCohortFailurePhaseV1::StaticAdmission,
        "{}/{case}",
        operation.name(),
    );
    assert_eq!(
        failure.code(),
        "runtime_binding_mismatch",
        "{}/{case}",
        operation.name(),
    );
    assert!(events.borrow().is_empty(), "{}/{case}", operation.name());
    assert!(
        runtime.calls.borrow().is_empty(),
        "{}/{case}",
        operation.name()
    );
    assert_eq!(
        runtime.operation_index.get(),
        0,
        "{}/{case}",
        operation.name()
    );
    assert_eq!(
        runtime.clock_read_index.get(),
        0,
        "{}/{case}",
        operation.name()
    );
    assert_failure_artifact(
        &failure,
        RUN_ID,
        "static_admission",
        "runtime_binding_mismatch",
        14,
        &[],
    );
}

fn bind_passport_to_native_capabilities(
    passport: &mut BenchmarkPassportV1,
    capabilities: &NativeCapabilities,
) {
    passport.adapter_name = capabilities.adapter_name.clone();
    passport.backend.adapter_backend = native_backend_name(capabilities.backend).to_owned();
    passport.backend.timestamp_query = capabilities.timestamp_query;
    passport.backend.features = if capabilities.timestamp_query {
        vec!["timestamp_query".to_owned()]
    } else {
        Vec::new()
    };
    passport.backend.limits = native_capability_limits(capabilities);
}

#[test]
fn runtime_observer_callback_is_rejected_before_probe_clock_or_operation() {
    for operation in CohortOperation::ALL {
        let plan = operation.plan(3, 10);
        let events = Rc::new(RefCell::new(Vec::new()));
        let runtime =
            CohortRuntime::new(plan.base_descriptor.clone(), Rc::clone(&events)).with_observer();
        assert_static_runtime_binding_failure(
            operation,
            plan,
            &runtime,
            &events,
            "embedded runtime observer",
        );
    }
}

#[test]
fn concrete_native_runtime_bridge_reads_real_observer_and_capabilities() {
    let Ok(clean_runtime) = NativeRuntime::new(NativeOptions::default()) else {
        return;
    };
    assert!(
        !clean_runtime.has_observer(),
        "default native runtime must expose its observer-free construction state"
    );
    let mut exact_passport = CohortOperation::Legacy.plan(3, 10).passport;
    bind_passport_to_native_capabilities(&mut exact_passport, clean_runtime.capabilities());
    assert_eq!(
        super::cohort::validate_native_runtime_authority_v1(&clean_runtime, &exact_passport),
        Ok(()),
        "the private cohort bridge must consume the real concrete runtime identity"
    );

    for mutation in RuntimeCapabilityMutation::ALL {
        let mut forged_capabilities = clean_runtime.capabilities().clone();
        mutation.apply(&mut forged_capabilities);
        let mut forged_passport = exact_passport.clone();
        bind_passport_to_native_capabilities(&mut forged_passport, &forged_capabilities);
        assert_eq!(
            super::cohort::validate_native_runtime_authority_v1(&clean_runtime, &forged_passport,),
            Err("runtime_binding_mismatch"),
            "the concrete bridge accepted a forged {mutation:?} passport"
        );
    }

    let Ok(observed_runtime) = NativeRuntime::new(NativeOptions {
        observer: Some(Arc::new(NoopRuntimeObserver)),
    }) else {
        return;
    };
    assert!(
        observed_runtime.has_observer(),
        "the concrete runtime must expose an installed callback before cohort execution"
    );
    let mut observed_passport = CohortOperation::Legacy.plan(3, 10).passport;
    bind_passport_to_native_capabilities(&mut observed_passport, observed_runtime.capabilities());
    assert_eq!(
        super::cohort::validate_native_runtime_authority_v1(&observed_runtime, &observed_passport,),
        Err("runtime_binding_mismatch"),
        "a concrete runtime with caller code must fail before the cohort can invoke it"
    );
}

#[test]
fn every_concrete_runtime_capability_is_bound_to_the_exact_passport_before_effects() {
    for operation in CohortOperation::ALL {
        for mutation in RuntimeCapabilityMutation::ALL {
            let plan = operation.plan(3, 10);
            let events = Rc::new(RefCell::new(Vec::new()));
            let runtime = CohortRuntime::new(plan.base_descriptor.clone(), Rc::clone(&events))
                .with_capability_mutation(|capabilities| mutation.apply(capabilities));
            assert_static_runtime_binding_failure(
                operation,
                plan,
                &runtime,
                &events,
                &format!("runtime {mutation:?}"),
            );
        }
    }
}

#[test]
fn missing_extra_or_false_passport_capabilities_fail_before_effects() {
    for operation in CohortOperation::ALL {
        let mut missing = operation.plan(3, 10);
        missing
            .passport
            .backend
            .limits
            .remove("max_compute_invocations_per_workgroup");
        refresh_preparation_links(&mut missing);

        let mut extra = operation.plan(3, 10);
        extra
            .passport
            .backend
            .limits
            .insert("forged_extra_limit".to_owned(), 1);
        refresh_preparation_links(&mut extra);

        let mut false_timestamp = operation.plan(3, 10);
        enable_timestamp_query(&mut false_timestamp);

        for (case, plan) in [
            ("missing passport limit", missing),
            ("extra passport limit", extra),
            ("false passport timestamp feature", false_timestamp),
        ] {
            let events = Rc::new(RefCell::new(Vec::new()));
            let runtime = CohortRuntime::new(plan.base_descriptor.clone(), Rc::clone(&events));
            assert_static_runtime_binding_failure(operation, plan, &runtime, &events, case);
        }
    }
}

#[test]
fn numerical_reference_is_non_degenerate_and_matches_the_independent_cpu_semantics() {
    let reference = validation_reference();
    let (cpu_checkpoints, cpu_output) = independently_computed_reference();

    assert_eq!(reference.expected_checkpoints, cpu_checkpoints);
    assert_eq!(reference.expected_output, cpu_output);
    assert_eq!(reference.expected_checkpoints.len(), 2);
    assert_ne!(
        reference.expected_checkpoints[&0], reference.expected_checkpoints[&1],
        "the validator must distinguish checkpoint depth"
    );
    assert_ne!(
        reference.expected_checkpoints[&0], reference.expected_output,
        "the validator must distinguish the first checkpoint from final output"
    );
    assert_ne!(
        reference.expected_checkpoints[&1], reference.expected_output,
        "the validator must distinguish the second checkpoint from final output"
    );
    for tensor in reference
        .expected_checkpoints
        .values()
        .chain(std::iter::once(&reference.expected_output))
    {
        assert!(tensor.iter().all(|value| value.is_finite()));
        assert!(tensor.iter().any(|value| *value != 0.0));
    }

    let alternate = alternate_validation_reference();
    let (alternate_cpu_checkpoints, alternate_cpu_output) =
        independently_computed_reference_for(&alternate_invocation());
    assert_eq!(alternate.expected_checkpoints, alternate_cpu_checkpoints);
    assert_eq!(alternate.expected_output, alternate_cpu_output);
    assert_eq!(
        alternate.expected_output,
        [
            0.5, 0.25, -0.25, -0.5, 0.5, 0.25, -0.25, -0.5, 0.5, 0.25, -0.25, -0.5
        ]
    );
    assert_ne!(
        alternate.expected_checkpoints,
        reference.expected_checkpoints
    );
    assert_ne!(alternate.expected_output, reference.expected_output);
    assert_ne!(alternate.accepted, reference.accepted);
}

#[test]
fn checkpoint_identity_hashes_the_complete_ordered_semantic_readback() {
    let reference = validation_reference();
    let digest =
        semantic_readback_sha256(&reference.expected_checkpoints, &reference.expected_output);
    assert_eq!(digest, OUTPUT_SHA256);
    assert_eq!(reference.accepted.output_sha256, OUTPUT_SHA256);

    let final_only = semantic_readback_sha256(&BTreeMap::new(), &reference.expected_output);
    assert_eq!(final_only, FINAL_ONLY_SHA256);
    assert_ne!(
        digest, final_only,
        "the native checkpoint identity must include both ordered checkpoints before final output"
    );
    let element_count = reference
        .expected_checkpoints
        .values()
        .map(Vec::len)
        .sum::<usize>()
        + reference.expected_output.len();
    assert_eq!(element_count, 36);
    assert_eq!(element_count * std::mem::size_of::<f32>(), 144);

    let mut reordered = reference.expected_checkpoints.clone();
    let first = reordered.remove(&0).unwrap();
    let second = reordered.remove(&1).unwrap();
    let reordered = BTreeMap::from([(0, second), (1, first)]);
    assert_ne!(
        semantic_readback_sha256(&reordered, &reference.expected_output),
        OUTPUT_SHA256,
        "checkpoint layer order is part of the authenticated byte domain"
    );
}

#[test]
fn validator_constructor_and_static_admission_authenticate_the_reference_byte_domain() {
    for operation in CohortOperation::ALL {
        let mut tensor_drift = operation.plan(3, 10);
        tensor_drift
            .validation_reference
            .expected_checkpoints
            .get_mut(&1)
            .unwrap()[7] += 0.5;
        let tensor_error = NativeBenchmarkVisionStackValidatorV1::from_leaf(&tensor_drift)
            .expect_err("checkpoint reference drift without a new digest must be rejected");
        assert_eq!(tensor_error.code(), CollectorErrorCodeV1::InvalidDescriptor);

        let mut output_drift = operation.plan(3, 10);
        output_drift.validation_reference.expected_output[9] -= 0.5;
        let output_error = NativeBenchmarkVisionStackValidatorV1::from_leaf(&output_drift)
            .expect_err("final reference drift without a new digest must be rejected");
        assert_eq!(output_error.code(), CollectorErrorCodeV1::InvalidDescriptor);

        let mut forged_hash = operation.plan(3, 10);
        let forged = hash('0');
        forged_hash.validation_reference.accepted.output_sha256 = forged.clone();
        forged_hash.workload.checkpoint_sha256 = forged.clone();
        forged_hash.correctness_anchor.expected_checkpoint_sha256 = forged.clone();
        forged_hash.base_descriptor.expected_output_sha256 = forged;
        refresh_preparation_links(&mut forged_hash);
        let hash_error = NativeBenchmarkVisionStackValidatorV1::from_leaf(&forged_hash)
            .expect_err("coherently linked arbitrary hex is not a semantic readback digest");
        assert_eq!(hash_error.code(), CollectorErrorCodeV1::InvalidDescriptor);

        for malformed in [tensor_drift, output_drift, forged_hash] {
            let events = Rc::new(RefCell::new(Vec::new()));
            let runtime = CohortRuntime::new(malformed.base_descriptor.clone(), Rc::clone(&events));
            let failure = run_test_cohort(
                operation,
                &runtime,
                malformed,
                Rc::clone(&events),
                None,
                None,
            )
            .expect_err("an unauthenticated reference must fail before one physical attempt");
            assert_eq!(
                failure.phase(),
                NativeBenchmarkCohortFailurePhaseV1::StaticAdmission
            );
            assert_eq!(failure.code(), "invalid_descriptor");
            assert!(events.borrow().is_empty());
            assert!(runtime.calls.borrow().is_empty());
            assert_eq!(runtime.clock_read_index.get(), 0);
            assert_failure_artifact(
                &failure,
                RUN_ID,
                "static_admission",
                "invalid_descriptor",
                14,
                &[],
            );
        }
    }
}

#[test]
fn concrete_validator_accepts_both_public_operations_through_the_shared_runner() {
    for operation in CohortOperation::ALL {
        let plan = operation.plan(3, 10);
        let expected_plan = plan.clone();
        let mut validator = NativeBenchmarkVisionStackValidatorV1::from_leaf(&plan)
            .expect("the exact leaf reference must construct the concrete authority");
        let events = Rc::new(RefCell::new(Vec::new()));
        let runtime = CohortRuntime::new(plan.base_descriptor.clone(), Rc::clone(&events));

        let completed = run_concrete_validator_cohort(
            operation,
            &runtime,
            plan,
            Rc::clone(&events),
            &mut validator,
        )
        .expect("the concrete authority must accept the exact output, checkpoints, and evidence");

        assert_eq!(runtime.calls.borrow().len(), 14, "{}", operation.name());
        assert_eq!(
            *events.borrow(),
            expected_concrete_validator_event_prefix(operation.name(), 14),
            "{}",
            operation.name(),
        );
        let expected_attempts = expected_concrete_validator_attempts(operation.variant(), 3, 14);
        assert_completed_matches_plan(&completed, &expected_plan, &expected_attempts);
    }
}

#[test]
fn concrete_validator_is_data_driven_for_a_second_valid_leaf_and_matching_execution() {
    for operation in CohortOperation::ALL {
        let plan = alternate_leaf_plan(operation);
        let expected_plan = plan.clone();
        let expected_reference = plan.validation_reference.clone();
        assert_ne!(expected_reference, validation_reference());
        let mut validator = NativeBenchmarkVisionStackValidatorV1::from_leaf(&plan)
            .expect("the alternate data-only leaf must construct a concrete authority");
        let events = Rc::new(RefCell::new(Vec::new()));
        let runtime = CohortRuntime::new(plan.base_descriptor.clone(), Rc::clone(&events));
        let invocation = alternate_invocation();
        let invocation_address = std::ptr::from_ref(&invocation).addr();

        let completed = run_concrete_validator_cohort_with_invocation(
            operation,
            &runtime,
            plan,
            &invocation,
            Rc::clone(&events),
            &mut validator,
        )
        .expect("the alternate reference must validate its matching execution");

        assert_eq!(runtime.calls.borrow().len(), 14, "{}", operation.name());
        assert!(
            runtime
                .calls
                .borrow()
                .iter()
                .all(|call| call.invocation_address == invocation_address),
            "{}",
            operation.name(),
        );
        assert_eq!(
            *events.borrow(),
            expected_concrete_validator_event_prefix(operation.name(), 14),
            "{}",
            operation.name(),
        );
        let expected_attempts = expected_concrete_validator_attempts_for_reference(
            operation.variant(),
            3,
            14,
            &expected_reference,
        );
        assert_completed_matches_plan(&completed, &expected_plan, &expected_attempts);
    }
}

#[test]
fn concrete_validator_is_bound_to_one_leaf_before_any_external_effect() {
    let authority_leaf = CohortOperation::Legacy.plan(3, 10);
    let mut validator = NativeBenchmarkVisionStackValidatorV1::from_leaf(&authority_leaf)
        .expect("the source leaf must construct a validator authority");
    let mut foreign_leaf = CohortOperation::Legacy.plan(3, 10);
    foreign_leaf.passport.model.case_id = "ocr.clean_latin.0002/vision.stack.2".to_owned();
    foreign_leaf.passport.model.input_blake3 = hash('9');
    refresh_preparation_links(&mut foreign_leaf);
    let events = Rc::new(RefCell::new(Vec::new()));
    let runtime = CohortRuntime::new(foreign_leaf.base_descriptor.clone(), Rc::clone(&events));

    let failure = run_concrete_validator_cohort(
        CohortOperation::Legacy,
        &runtime,
        foreign_leaf,
        Rc::clone(&events),
        &mut validator,
    )
    .expect_err("a validator created for another leaf must not validate one attempt");

    assert_eq!(
        failure.phase(),
        NativeBenchmarkCohortFailurePhaseV1::StaticAdmission
    );
    assert_eq!(failure.code(), "validator_binding_mismatch");
    assert!(events.borrow().is_empty());
    assert!(runtime.calls.borrow().is_empty());
    assert_eq!(runtime.clock_read_index.get(), 0);
    assert_failure_artifact(
        &failure,
        RUN_ID,
        "static_admission",
        "validator_binding_mismatch",
        14,
        &[],
    );
}

#[test]
fn concrete_validator_constructor_rejects_invalid_or_non_finite_tolerance() {
    for tolerance in [-1.0_f32, f32::NAN, f32::INFINITY] {
        let mut plan = CohortOperation::Legacy.plan(3, 10);
        plan.validation_reference.max_abs_error = tolerance;
        let error = NativeBenchmarkVisionStackValidatorV1::from_leaf(&plan)
            .expect_err("the numerical envelope must be finite and non-negative");
        assert_eq!(error.code(), CollectorErrorCodeV1::InvalidDescriptor);
    }
}

#[test]
fn concrete_validator_constructor_rejects_non_finite_authored_reference_values() {
    for operation in CohortOperation::ALL {
        let mut output_plan = operation.plan(3, 10);
        output_plan.validation_reference.expected_output[5] = f32::NAN;
        let output_error = NativeBenchmarkVisionStackValidatorV1::from_leaf(&output_plan)
            .expect_err("a non-finite authored final reference must fail before execution");
        assert_eq!(output_error.code(), CollectorErrorCodeV1::InvalidDescriptor);

        let mut checkpoint_plan = operation.plan(3, 10);
        checkpoint_plan
            .validation_reference
            .expected_checkpoints
            .get_mut(&1)
            .unwrap()[6] = f32::INFINITY;
        let checkpoint_error = NativeBenchmarkVisionStackValidatorV1::from_leaf(&checkpoint_plan)
            .expect_err("a non-finite authored checkpoint reference must fail before execution");
        assert_eq!(
            checkpoint_error.code(),
            CollectorErrorCodeV1::InvalidDescriptor
        );
    }
}

#[test]
fn concrete_validator_enforces_the_positive_tolerance_boundary_for_outputs_and_checkpoints() {
    for expected in [EXPECTED_OUTPUT[11], EXPECTED_CHECKPOINT_1[11]] {
        let boundary = expected + 0.25;
        let outside = next_f32_above(boundary);
        assert_eq!((boundary - expected).abs(), 0.25);
        assert!(
            (outside - expected).abs() > 0.25,
            "the hostile value must be observably outside after f32 rounding"
        );
        assert_eq!(outside.to_bits(), boundary.to_bits() + 1);
    }

    for operation in CohortOperation::ALL {
        for (name, accepted_faults, outside) in [
            (
                "final output",
                [
                    (
                        "positive interior",
                        value_fault(
                            1,
                            NumericalTensor::Output,
                            0,
                            NumericalValueMutation::PositiveInterior,
                        ),
                    ),
                    (
                        "negative interior",
                        value_fault(
                            1,
                            NumericalTensor::Output,
                            5,
                            NumericalValueMutation::NegativeInterior,
                        ),
                    ),
                    (
                        "inclusive boundary",
                        value_fault(
                            1,
                            NumericalTensor::Output,
                            11,
                            NumericalValueMutation::InclusiveBoundary,
                        ),
                    ),
                ],
                value_fault(
                    2,
                    NumericalTensor::Output,
                    11,
                    NumericalValueMutation::ImmediatelyOutside,
                ),
            ),
            (
                "checkpoint",
                [
                    (
                        "positive interior",
                        value_fault(
                            1,
                            NumericalTensor::Checkpoint(0),
                            0,
                            NumericalValueMutation::PositiveInterior,
                        ),
                    ),
                    (
                        "negative interior",
                        value_fault(
                            1,
                            NumericalTensor::Checkpoint(0),
                            5,
                            NumericalValueMutation::NegativeInterior,
                        ),
                    ),
                    (
                        "inclusive boundary",
                        value_fault(
                            1,
                            NumericalTensor::Checkpoint(1),
                            11,
                            NumericalValueMutation::InclusiveBoundary,
                        ),
                    ),
                ],
                value_fault(
                    2,
                    NumericalTensor::Checkpoint(1),
                    11,
                    NumericalValueMutation::ImmediatelyOutside,
                ),
            ),
        ] {
            for (position, accepted_fault) in accepted_faults {
                let mut accepted_plan = operation.plan(3, 10);
                accepted_plan.validation_reference.max_abs_error = 0.25;
                let expected_plan = accepted_plan.clone();
                let mut accepted_validator =
                    NativeBenchmarkVisionStackValidatorV1::from_leaf(&accepted_plan)
                        .expect("a finite positive envelope must construct");
                let accepted_events = Rc::new(RefCell::new(Vec::new()));
                let accepted_runtime = CohortRuntime::with_fault(
                    accepted_plan.base_descriptor.clone(),
                    Rc::clone(&accepted_events),
                    accepted_fault,
                );
                let completed = run_concrete_validator_cohort(
                    operation,
                    &accepted_runtime,
                    accepted_plan,
                    Rc::clone(&accepted_events),
                    &mut accepted_validator,
                )
                .expect("a value on or inside the authored envelope must be accepted");
                let expected_attempts =
                    expected_concrete_validator_attempts(operation.variant(), 3, 14);
                assert_completed_matches_plan(&completed, &expected_plan, &expected_attempts);
                assert_eq!(
                    *accepted_events.borrow(),
                    expected_concrete_validator_event_prefix(operation.name(), 14),
                    "{}/{name}/{position}",
                    operation.name(),
                );
            }

            let mut rejected_plan = operation.plan(3, 10);
            rejected_plan.validation_reference.max_abs_error = 0.25;
            let mut rejected_validator =
                NativeBenchmarkVisionStackValidatorV1::from_leaf(&rejected_plan)
                    .expect("a finite positive envelope must construct");
            let rejected_events = Rc::new(RefCell::new(Vec::new()));
            let rejected_runtime = CohortRuntime::with_fault(
                rejected_plan.base_descriptor.clone(),
                Rc::clone(&rejected_events),
                outside,
            );
            let failure = run_concrete_validator_cohort(
                operation,
                &rejected_runtime,
                rejected_plan,
                Rc::clone(&rejected_events),
                &mut rejected_validator,
            )
            .expect_err("a value outside the authored envelope must terminate the leaf");
            assert_eq!(
                failure.code(),
                "validation_failed",
                "{}/{name}",
                operation.name()
            );
            assert_eq!(
                *rejected_events.borrow(),
                expected_concrete_validation_failure_events(operation.name(), 2),
                "{}/{name}",
                operation.name(),
            );
            let mut rejected_attempts =
                expected_concrete_validator_attempts(operation.variant(), 3, 2);
            rejected_attempts.push(failed_attempt(2, 3, "validation_failed"));
            assert_failure_artifact(
                &failure,
                RUN_ID,
                "attempt",
                "validation_failed",
                14,
                &rejected_attempts,
            );
        }
    }
}

fn assert_concrete_validation_failure(
    operation: CohortOperation,
    name: &str,
    fault: RuntimeFault,
    sequence: usize,
) {
    let plan = operation.plan(3, 10);
    let mut validator = NativeBenchmarkVisionStackValidatorV1::from_leaf(&plan)
        .expect("the exact leaf must construct a validator authority");
    let events = Rc::new(RefCell::new(Vec::new()));
    let runtime =
        CohortRuntime::with_fault(plan.base_descriptor.clone(), Rc::clone(&events), fault);

    let failure = run_concrete_validator_cohort(
        operation,
        &runtime,
        plan,
        Rc::clone(&events),
        &mut validator,
    )
    .expect_err("numerical corruption must terminate the authored leaf");

    assert_eq!(
        failure.code(),
        "validation_failed",
        "{}/{name}",
        operation.name(),
    );
    assert_eq!(
        runtime.calls.borrow().len(),
        sequence + 1,
        "{}/{name}",
        operation.name(),
    );
    assert_eq!(
        *events.borrow(),
        expected_concrete_validation_failure_events(operation.name(), sequence),
        "{}/{name}",
        operation.name(),
    );
    let mut expected_attempts =
        expected_concrete_validator_attempts(operation.variant(), 3, sequence);
    expected_attempts.push(failed_attempt(sequence, 3, "validation_failed"));
    assert_failure_artifact(
        &failure,
        RUN_ID,
        "attempt",
        "validation_failed",
        14,
        &expected_attempts,
    );
}

#[test]
fn concrete_validator_checks_every_tensor_element_both_depths_and_late_attempts() {
    let failure_sequences = [0_usize, 4, 13];
    for operation in CohortOperation::ALL {
        for tensor in [
            NumericalTensor::Output,
            NumericalTensor::Checkpoint(0),
            NumericalTensor::Checkpoint(1),
        ] {
            for index in 0..EXPECTED_OUTPUT.len() {
                let sequence = failure_sequences[index % failure_sequences.len()];
                assert_concrete_validation_failure(
                    operation,
                    &format!("{tensor:?} element {index} at attempt {sequence}"),
                    value_fault(sequence, tensor, index, NumericalValueMutation::Drift),
                    sequence,
                );
            }

            assert_concrete_validation_failure(
                operation,
                &format!("non-finite {tensor:?} in final measured attempt"),
                value_fault(
                    13,
                    tensor,
                    EXPECTED_OUTPUT.len() / 2,
                    NumericalValueMutation::NonFinite,
                ),
                13,
            );
            for mutation in [
                NumericalLengthMutation::Short,
                NumericalLengthMutation::Long,
            ] {
                assert_concrete_validation_failure(
                    operation,
                    &format!("{mutation:?} {tensor:?} in final measured attempt"),
                    length_fault(13, tensor, mutation),
                    13,
                );
            }
        }

        for (name, fault) in [
            (
                "missing checkpoint key",
                RuntimeFault::MissingCheckpoint(13),
            ),
            ("extra checkpoint key", RuntimeFault::ExtraCheckpoint(13)),
        ] {
            assert_concrete_validation_failure(operation, name, fault, 13);
        }
    }
}

#[test]
fn concrete_validator_rejects_qkv_causal_evidence_before_closing_the_leaf() {
    for (name, fault, sequence) in [
        ("first plan digest", RuntimeFault::QkvEvidenceFirst(1), 1),
        ("second plan digest", RuntimeFault::QkvEvidenceSecond(1), 1),
        ("plan digest order", RuntimeFault::QkvEvidenceOrder(1), 1),
        (
            "duplicate plan digest",
            RuntimeFault::QkvEvidenceDuplicate(1),
            1,
        ),
        (
            "missing plan digest",
            RuntimeFault::QkvEvidenceMissing(1),
            1,
        ),
        (
            "extra plan digest in final measured attempt",
            RuntimeFault::QkvEvidenceExtra(13),
            13,
        ),
        ("policy", RuntimeFault::QkvPolicy(1), 1),
        ("outcome", RuntimeFault::QkvOutcome(1), 1),
    ] {
        let plan = CohortOperation::Qkv.plan(3, 10);
        let mut validator = NativeBenchmarkVisionStackValidatorV1::from_leaf(&plan)
            .expect("the exact fused leaf must construct a validator authority");
        let events = Rc::new(RefCell::new(Vec::new()));
        let runtime =
            CohortRuntime::with_fault(plan.base_descriptor.clone(), Rc::clone(&events), fault);

        let failure = run_concrete_validator_cohort(
            CohortOperation::Qkv,
            &runtime,
            plan,
            Rc::clone(&events),
            &mut validator,
        )
        .expect_err("wrong fused evidence must terminate the authored leaf");

        assert_eq!(failure.code(), "validation_failed", "{name}");
        assert_eq!(runtime.calls.borrow().len(), sequence + 1, "{name}");
        assert_eq!(
            *events.borrow(),
            expected_concrete_validation_failure_events("qkv", sequence),
            "{name}",
        );
        let mut expected_attempts = expected_concrete_validator_attempts(
            FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1,
            3,
            sequence,
        );
        expected_attempts.push(failed_attempt(sequence, 3, "validation_failed"));
        assert_failure_artifact(
            &failure,
            RUN_ID,
            "attempt",
            "validation_failed",
            14,
            &expected_attempts,
        );
    }
}

#[test]
fn first_operation_failure_is_terminal_recorded_once_and_never_retried_or_assembled() {
    let plan = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    // sequence 6 is measured slot 2 after cold plus three warmups.
    let events = Rc::new(RefCell::new(Vec::new()));
    let runtime = CohortRuntime::with_fault(
        plan.base_descriptor.clone(),
        Rc::clone(&events),
        RuntimeFault::Operation(6),
    );

    let failure = run_native_public_legacy_benchmark_cohort_v1(
        &runtime,
        plan,
        &invocation(),
        CHECKPOINT_LAYERS,
        VisionStackActivationStrategy::StaticArenaAlias,
        scripted_probe(Rc::clone(&events), None),
        legacy_validator(Rc::clone(&events), None),
    )
    .expect_err("one failed operation must make the complete leaf ineligible");

    assert_eq!(
        failure.phase(),
        NativeBenchmarkCohortFailurePhaseV1::Attempt
    );
    assert_eq!(failure.code(), "execution_failed");
    assert_eq!(failure.expected_attempt_count(), 14);
    assert_eq!(runtime.calls.borrow().len(), 7);
    let mut expected_events = expected_event_prefix("legacy", 6);
    let start = clock_start_for(6);
    expected_events.extend([
        CohortEvent::ProbeBefore { sequence: 6 },
        CohortEvent::Clock {
            sequence: 6,
            boundary: "start",
            value: Some(start),
        },
        CohortEvent::LegacyOperation { sequence: 6 },
        CohortEvent::Clock {
            sequence: 6,
            boundary: "end",
            value: Some(start + ATTEMPT_DURATIONS_NS[6]),
        },
    ]);
    assert_eq!(*events.borrow(), expected_events);

    let mut expected_attempts =
        expected_passed_attempts(LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1, 3, 6);
    expected_attempts.push(failed_attempt(6, 3, "execution_failed"));
    assert_failure_artifact(
        &failure,
        RUN_ID,
        "attempt",
        "execution_failed",
        14,
        &expected_attempts,
    );
}

#[test]
fn validator_failure_closes_the_current_clock_then_stops_without_a_later_attempt() {
    let plan = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    let events = Rc::new(RefCell::new(Vec::new()));
    let runtime = CohortRuntime::new(plan.base_descriptor.clone(), Rc::clone(&events));

    let failure = run_native_public_legacy_benchmark_cohort_v1(
        &runtime,
        plan,
        &invocation(),
        CHECKPOINT_LAYERS,
        VisionStackActivationStrategy::StaticArenaAlias,
        scripted_probe(Rc::clone(&events), None),
        legacy_validator(Rc::clone(&events), Some(ValidationFault::Reject(5))),
    )
    .expect_err("validator failure must terminate the physical leaf");

    assert_eq!(failure.code(), "validation_failed");
    assert_eq!(runtime.calls.borrow().len(), 6);
    let mut expected_events = expected_event_prefix("legacy", 5);
    let start = clock_start_for(5);
    expected_events.extend([
        CohortEvent::ProbeBefore { sequence: 5 },
        CohortEvent::Clock {
            sequence: 5,
            boundary: "start",
            value: Some(start),
        },
        CohortEvent::LegacyOperation { sequence: 5 },
        CohortEvent::Validate { sequence: 5 },
        CohortEvent::Clock {
            sequence: 5,
            boundary: "end",
            value: Some(start + ATTEMPT_DURATIONS_NS[5]),
        },
    ]);
    assert_eq!(*events.borrow(), expected_events);
    let mut expected_attempts =
        expected_passed_attempts(LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1, 3, 5);
    expected_attempts.push(failed_attempt(5, 3, "validation_failed"));
    assert_failure_artifact(
        &failure,
        RUN_ID,
        "attempt",
        "validation_failed",
        14,
        &expected_attempts,
    );
}

#[test]
fn every_attempt_failure_keeps_the_exact_prefix_and_never_retries() {
    let sequence = 4_usize;
    let cases = vec![
        (
            "before probe",
            Vec::new(),
            Some(ProbeFault::Before(sequence)),
            None,
            "environment_probe_failed",
            TerminalEffect::ProbeBefore,
        ),
        (
            "opening clock",
            vec![RuntimeFault::ClockRead(sequence * 2)],
            None,
            None,
            "clock_failed",
            TerminalEffect::OpeningClock,
        ),
        (
            "closing clock",
            vec![RuntimeFault::ClockRead(sequence * 2 + 1)],
            None,
            None,
            "clock_failed",
            TerminalEffect::ClosingClock,
        ),
        (
            "non-monotonic clock",
            vec![RuntimeFault::NonMonotonicClock(sequence)],
            None,
            None,
            "non_monotonic_clock",
            TerminalEffect::NonMonotonicClock,
        ),
        (
            "after probe",
            Vec::new(),
            Some(ProbeFault::After(sequence)),
            None,
            "environment_probe_failed",
            TerminalEffect::ProbeAfter,
        ),
        (
            "diagnostics",
            vec![RuntimeFault::InvalidDiagnostics(sequence)],
            None,
            None,
            "invalid_diagnostics",
            TerminalEffect::PostCollection,
        ),
        (
            "queue observation",
            vec![RuntimeFault::InvalidQueueObservation(sequence)],
            None,
            None,
            "invalid_queue_observation",
            TerminalEffect::PostCollection,
        ),
        (
            "topology",
            vec![RuntimeFault::Topology(sequence)],
            None,
            None,
            "topology_mismatch",
            TerminalEffect::PostCollection,
        ),
        (
            "resources",
            vec![RuntimeFault::Resource(sequence)],
            None,
            None,
            "resource_mismatch",
            TerminalEffect::PostCollection,
        ),
        (
            "validator cross-link",
            Vec::new(),
            None,
            Some(ValidationFault::CrossLink(sequence)),
            "cross_link_mismatch",
            TerminalEffect::PostCollection,
        ),
        (
            "operation plus failed closing clock",
            vec![
                RuntimeFault::Operation(sequence),
                RuntimeFault::ClockRead(sequence * 2 + 1),
            ],
            None,
            None,
            "clock_failed",
            TerminalEffect::ExecutionAndClosingClock,
        ),
    ];

    for (name, faults, probe_fault, validation_fault, code, terminal) in cases {
        for operation in CohortOperation::ALL {
            let plan = operation.plan(3, 10);
            let events = Rc::new(RefCell::new(Vec::new()));
            let runtime = CohortRuntime::with_faults(
                plan.base_descriptor.clone(),
                Rc::clone(&events),
                faults.iter().copied(),
            );
            let failure = run_test_cohort(
                operation,
                &runtime,
                plan,
                Rc::clone(&events),
                probe_fault,
                validation_fault,
            )
            .expect_err("the first failed effect must terminate the exact leaf");

            assert_eq!(
                failure.phase(),
                NativeBenchmarkCohortFailurePhaseV1::Attempt,
                "{}/{name}",
                operation.name(),
            );
            assert_eq!(failure.code(), code, "{}/{name}", operation.name());
            assert_eq!(
                failure.expected_attempt_count(),
                14,
                "{}/{name}",
                operation.name(),
            );
            let expected_calls = match terminal {
                TerminalEffect::ProbeBefore | TerminalEffect::OpeningClock => sequence,
                _ => sequence + 1,
            };
            assert_eq!(
                runtime.calls.borrow().len(),
                expected_calls,
                "{}/{name}",
                operation.name(),
            );
            assert_eq!(
                *events.borrow(),
                expected_terminal_events(operation.name(), sequence, terminal),
                "{}/{name}",
                operation.name(),
            );
            let mut expected_attempts = expected_passed_attempts(operation.variant(), 3, sequence);
            expected_attempts.push(failed_attempt(sequence, 3, code));
            assert_failure_artifact(&failure, RUN_ID, "attempt", code, 14, &expected_attempts);
        }
    }
}

#[test]
fn timestamp_failures_are_terminal_before_late_assembly_or_panic() {
    for operation in CohortOperation::ALL {
        let mut invalid_plan = operation.plan(3, 10);
        enable_timestamp_query(&mut invalid_plan);
        let invalid_events = Rc::new(RefCell::new(Vec::new()));
        let invalid_runtime = CohortRuntime::with_fault(
            invalid_plan.base_descriptor.clone(),
            Rc::clone(&invalid_events),
            RuntimeFault::InvalidTimestamp(0),
        )
        .with_timestamp_query();
        let invalid_failure = run_test_cohort(
            operation,
            &invalid_runtime,
            invalid_plan,
            Rc::clone(&invalid_events),
            None,
            None,
        )
        .expect_err("an invalid timestamp must fail the assigned cold attempt");
        assert_eq!(invalid_failure.code(), "invalid_timestamp");
        assert_eq!(invalid_runtime.calls.borrow().len(), 1);
        assert_eq!(
            *invalid_events.borrow(),
            expected_terminal_events(operation.name(), 0, TerminalEffect::PostCollection)
        );
        assert_failure_artifact(
            &invalid_failure,
            RUN_ID,
            "attempt",
            "invalid_timestamp",
            14,
            &[failed_attempt(0, 3, "invalid_timestamp")],
        );

        let mut stale_plan = operation.plan(3, 10);
        enable_timestamp_query(&mut stale_plan);
        let stale_events = Rc::new(RefCell::new(Vec::new()));
        let stale_runtime = CohortRuntime::with_fault(
            stale_plan.base_descriptor.clone(),
            Rc::clone(&stale_events),
            RuntimeFault::StaleTimestamp(1),
        )
        .with_timestamp_query();
        let stale_failure = run_test_cohort(
            operation,
            &stale_runtime,
            stale_plan,
            Rc::clone(&stale_events),
            None,
            None,
        )
        .expect_err(
            "a repeated timestamp pair must become a failed attempt, not an assembly panic",
        );
        assert_eq!(stale_failure.code(), "stale_timestamp");
        assert_eq!(stale_runtime.calls.borrow().len(), 2);
        assert_eq!(
            *stale_events.borrow(),
            expected_event_prefix(operation.name(), 2)
        );
        let first_sample = expected_timestamp_sample(operation.variant(), 0, 0);
        let expected_attempts = vec![
            BenchmarkSampleAttemptV1::Passed {
                sequence: 0,
                cohort: BenchmarkCohortV1::Cold,
                planned_slot: 0,
                sample: first_sample,
            },
            failed_attempt(1, 3, "stale_timestamp"),
        ];
        assert_failure_artifact(
            &stale_failure,
            RUN_ID,
            "attempt",
            "stale_timestamp",
            14,
            &expected_attempts,
        );
    }
}

#[test]
fn cold_last_warmup_and_final_measured_operation_failures_are_terminal_for_both_operations() {
    for operation in CohortOperation::ALL {
        for sequence in [0_usize, 3, 13] {
            let plan = operation.plan(3, 10);
            let events = Rc::new(RefCell::new(Vec::new()));
            let runtime = CohortRuntime::with_fault(
                plan.base_descriptor.clone(),
                Rc::clone(&events),
                RuntimeFault::Operation(sequence),
            );
            let failure =
                run_test_cohort(operation, &runtime, plan, Rc::clone(&events), None, None)
                    .expect_err("operation failure must terminate");

            assert_eq!(
                failure.code(),
                "execution_failed",
                "{}/{sequence}",
                operation.name(),
            );
            assert_eq!(runtime.calls.borrow().len(), sequence + 1);
            assert_eq!(
                *events.borrow(),
                expected_terminal_events(operation.name(), sequence, TerminalEffect::Execution)
            );
            let mut expected_attempts = expected_passed_attempts(operation.variant(), 3, sequence);
            expected_attempts.push(failed_attempt(sequence, 3, "execution_failed"));
            assert_failure_artifact(
                &failure,
                RUN_ID,
                "attempt",
                "execution_failed",
                14,
                &expected_attempts,
            );
        }
    }
}

#[test]
fn qkv_policy_mismatch_is_rejected_before_any_external_effect() {
    let plan = leaf_plan(
        FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "required",
        "fused",
        3,
        10,
    );
    let events = Rc::new(RefCell::new(Vec::new()));
    let runtime = CohortRuntime::new(plan.base_descriptor.clone(), Rc::clone(&events));
    let preferred_selection = fused_selection(VisionQkvExecutionPolicy::Preferred);

    let failure = run_native_public_qkv_benchmark_cohort_v1(
        &runtime,
        plan,
        &invocation(),
        CHECKPOINT_LAYERS,
        VisionStackActivationStrategy::StaticArenaAlias,
        &preferred_selection,
        scripted_probe(Rc::clone(&events), None),
        qkv_validator(Rc::clone(&events), None),
    )
    .expect_err("only one exact Required -> Fused selection may enter the cohort");

    assert_eq!(
        failure.phase(),
        NativeBenchmarkCohortFailurePhaseV1::StaticAdmission
    );
    assert_eq!(failure.code(), "operation_binding_mismatch");
    assert!(events.borrow().is_empty());
    assert!(runtime.calls.borrow().is_empty());
    assert_eq!(runtime.clock_read_index.get(), 0);
    assert_failure_artifact(
        &failure,
        RUN_ID,
        "static_admission",
        "operation_binding_mismatch",
        14,
        &[],
    );
}

fn assert_qkv_static_binding_failure(
    plan: NativeBenchmarkLeafPlanV1,
    selection: &VisionQkvStackSelection,
    name: &str,
) {
    let events = Rc::new(RefCell::new(Vec::new()));
    let runtime = CohortRuntime::new(plan.base_descriptor.clone(), Rc::clone(&events));
    let failure = run_native_public_qkv_benchmark_cohort_v1(
        &runtime,
        plan,
        &invocation(),
        CHECKPOINT_LAYERS,
        VisionStackActivationStrategy::StaticArenaAlias,
        selection,
        scripted_probe(Rc::clone(&events), None),
        qkv_validator(Rc::clone(&events), None),
    )
    .expect_err("a QKV authority mismatch must be rejected before one attempt");

    assert_eq!(
        failure.phase(),
        NativeBenchmarkCohortFailurePhaseV1::StaticAdmission,
        "{name}",
    );
    assert_eq!(failure.code(), "operation_binding_mismatch", "{name}");
    assert!(events.borrow().is_empty(), "{name}");
    assert!(runtime.calls.borrow().is_empty(), "{name}");
    assert_eq!(runtime.clock_read_index.get(), 0, "{name}");
    assert_failure_artifact(
        &failure,
        RUN_ID,
        "static_admission",
        "operation_binding_mismatch",
        14,
        &[],
    );
}

#[test]
fn every_qkv_selection_target_limit_is_bound_to_the_passport_before_effects() {
    let expected_target = fixture_target_limits(FIXTURE_MIN_STORAGE_BUFFER_OFFSET_ALIGNMENT);
    let expected_digests = fixture_layer_plan_blake3();
    let mut cases = Vec::new();

    let mut alignment = expected_target;
    alignment.min_storage_buffer_offset_alignment *= 2;
    cases.push(("alignment", alignment, false));

    let mut storage_bindings = expected_target;
    storage_bindings.max_storage_buffers_per_shader_stage += 1;
    cases.push(("storage binding count", storage_bindings, true));

    let mut binding_size = expected_target;
    binding_size.max_storage_buffer_binding_size += 4;
    cases.push(("storage binding size", binding_size, true));

    let mut buffer_size = expected_target;
    buffer_size.max_buffer_size += 4;
    cases.push(("buffer size", buffer_size, true));

    let mut workgroups = expected_target;
    workgroups.max_compute_workgroups_per_dimension += 1;
    cases.push(("workgroup count", workgroups, true));

    for (name, target, digest_must_be_unchanged) in cases {
        let selection =
            fused_selection_for_geometry(VisionQkvExecutionPolicy::Required, 3, &[0, 1, 3], target);
        let selection_digests = selection_layer_plan_blake3(&selection);
        if digest_must_be_unchanged {
            assert_eq!(
                selection_digests, expected_digests,
                "{name} must isolate target-limit validation from layer-plan digest validation",
            );
        }
        let mut plan = CohortOperation::Qkv.plan(3, 10);
        plan.workload.ordered_layer_plans_blake3 = selection_digests;
        refresh_preparation_links(&mut plan);
        assert_qkv_static_binding_failure(plan, &selection, name);
    }
}

#[test]
fn qkv_passport_must_record_every_limit_used_by_the_selection() {
    let selection = required_fused_selection();
    for key in [
        "min_storage_buffer_offset_alignment",
        "max_storage_buffers_per_shader_stage",
        "max_storage_buffer_binding_size",
        "max_buffer_size",
        "max_compute_workgroups_per_dimension",
    ] {
        let mut plan = CohortOperation::Qkv.plan(3, 10);
        assert!(plan.passport.backend.limits.remove(key).is_some());
        refresh_preparation_links(&mut plan);
        assert_qkv_static_binding_failure(plan, &selection, key);
    }
}

#[test]
fn operation_policy_kernel_and_activation_are_one_static_binding() {
    let legacy_mislabeled = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "required",
        "fused",
        3,
        10,
    );
    let legacy_events = Rc::new(RefCell::new(Vec::new()));
    let legacy_runtime = CohortRuntime::new(
        legacy_mislabeled.base_descriptor.clone(),
        Rc::clone(&legacy_events),
    );
    let legacy_failure = run_native_public_legacy_benchmark_cohort_v1(
        &legacy_runtime,
        legacy_mislabeled,
        &invocation(),
        CHECKPOINT_LAYERS,
        VisionStackActivationStrategy::StaticArenaAlias,
        scripted_probe(Rc::clone(&legacy_events), None),
        legacy_validator(Rc::clone(&legacy_events), None),
    )
    .expect_err("legacy execution cannot claim a fused-required workload");
    assert_eq!(
        legacy_failure.phase(),
        NativeBenchmarkCohortFailurePhaseV1::StaticAdmission
    );
    assert_eq!(legacy_failure.code(), "operation_binding_mismatch");
    assert!(legacy_events.borrow().is_empty());
    assert!(legacy_runtime.calls.borrow().is_empty());
    assert_eq!(legacy_runtime.clock_read_index.get(), 0);
    assert_failure_artifact(
        &legacy_failure,
        RUN_ID,
        "static_admission",
        "operation_binding_mismatch",
        14,
        &[],
    );

    let mut qkv_cases = Vec::new();
    qkv_cases.push((
        "disabled workload",
        leaf_plan(
            FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1,
            "disabled",
            "disabled",
            3,
            10,
        ),
    ));
    let mut mislabeled_kernel = leaf_plan(
        FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "required",
        "fused",
        3,
        10,
    );
    mislabeled_kernel.workload.kernel_variant.id =
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1.to_owned();
    refresh_preparation_links(&mut mislabeled_kernel);
    qkv_cases.push(("workload kernel", mislabeled_kernel));
    let mut mismatched_activation = leaf_plan(
        FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "required",
        "fused",
        3,
        10,
    );
    mismatched_activation.base_descriptor.activation_strategy = "separate_buffers".to_owned();
    qkv_cases.push(("descriptor activation", mismatched_activation));

    for (name, plan) in qkv_cases {
        let events = Rc::new(RefCell::new(Vec::new()));
        let runtime = CohortRuntime::new(plan.base_descriptor.clone(), Rc::clone(&events));
        let selection = required_fused_selection();
        let failure = run_native_public_qkv_benchmark_cohort_v1(
            &runtime,
            plan,
            &invocation(),
            CHECKPOINT_LAYERS,
            VisionStackActivationStrategy::StaticArenaAlias,
            &selection,
            scripted_probe(Rc::clone(&events), None),
            qkv_validator(Rc::clone(&events), None),
        )
        .expect_err("the QKV public operation has one exact static binding");
        assert_eq!(
            failure.phase(),
            NativeBenchmarkCohortFailurePhaseV1::StaticAdmission,
            "{name}"
        );
        assert_eq!(failure.code(), "operation_binding_mismatch", "{name}");
        assert!(events.borrow().is_empty(), "{name}");
        assert!(runtime.calls.borrow().is_empty(), "{name}");
        assert_eq!(runtime.clock_read_index.get(), 0, "{name}");
        assert_failure_artifact(
            &failure,
            RUN_ID,
            "static_admission",
            "operation_binding_mismatch",
            14,
            &[],
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OperationInputMutation {
    Tokens,
    HiddenSize,
    LayerCount,
    InputLength,
    CuSeqlens,
    ParameterLength,
    Checkpoints,
    Activation,
    QkvSelectionGeometry,
    QkvSelectionTarget,
    QkvPlanDigestSecond,
    QkvPlanDigestOrder,
    QkvPlanDigestDuplicate,
    QkvPlanDigestMissing,
}

#[test]
fn invocation_checkpoints_activation_and_selection_are_admitted_before_effects() {
    for operation in CohortOperation::ALL {
        let mut mutations = vec![
            OperationInputMutation::Tokens,
            OperationInputMutation::HiddenSize,
            OperationInputMutation::LayerCount,
            OperationInputMutation::InputLength,
            OperationInputMutation::CuSeqlens,
            OperationInputMutation::ParameterLength,
            OperationInputMutation::Checkpoints,
            OperationInputMutation::Activation,
        ];
        if operation == CohortOperation::Qkv {
            mutations.extend([
                OperationInputMutation::QkvSelectionGeometry,
                OperationInputMutation::QkvSelectionTarget,
                OperationInputMutation::QkvPlanDigestSecond,
                OperationInputMutation::QkvPlanDigestOrder,
                OperationInputMutation::QkvPlanDigestDuplicate,
                OperationInputMutation::QkvPlanDigestMissing,
            ]);
        }

        for mutation in mutations {
            let mut plan = operation.plan(3, 10);
            match mutation {
                OperationInputMutation::QkvSelectionGeometry => {
                    plan.workload.ordered_layer_plans_blake3 =
                        selection_layer_plan_blake3(&alternate_required_fused_selection());
                }
                OperationInputMutation::QkvPlanDigestSecond => {
                    plan.workload.ordered_layer_plans_blake3[1] = hash('0');
                }
                OperationInputMutation::QkvPlanDigestOrder => {
                    plan.workload.ordered_layer_plans_blake3.swap(0, 1);
                }
                OperationInputMutation::QkvPlanDigestDuplicate => {
                    let first = plan.workload.ordered_layer_plans_blake3[0].clone();
                    plan.workload.ordered_layer_plans_blake3[1] = first;
                }
                OperationInputMutation::QkvPlanDigestMissing => {
                    plan.workload.ordered_layer_plans_blake3.pop();
                }
                _ => {}
            }
            if matches!(
                mutation,
                OperationInputMutation::QkvSelectionGeometry
                    | OperationInputMutation::QkvPlanDigestSecond
                    | OperationInputMutation::QkvPlanDigestOrder
                    | OperationInputMutation::QkvPlanDigestDuplicate
                    | OperationInputMutation::QkvPlanDigestMissing
            ) {
                refresh_preparation_links(&mut plan);
            }
            let events = Rc::new(RefCell::new(Vec::new()));
            let runtime = CohortRuntime::new(plan.base_descriptor.clone(), Rc::clone(&events));
            let mut invocation = invocation();
            let bad_input = [0.0_f32; 11];
            let bad_cu_seqlens = [0_u32, 2, 2];
            let bad_query_weight = [0.0_f32; 15];
            let mut bad_layer = invocation.layer_parameters[0];
            bad_layer.query.weight = &bad_query_weight;
            let bad_layers = [bad_layer, invocation.layer_parameters[1]];
            let mut checkpoints = CHECKPOINT_LAYERS;
            let mut activation = VisionStackActivationStrategy::StaticArenaAlias;
            match mutation {
                OperationInputMutation::Tokens => invocation.tokens += 1,
                OperationInputMutation::HiddenSize => invocation.hidden_size += 1,
                OperationInputMutation::LayerCount => invocation.layer_parameters = &[],
                OperationInputMutation::InputLength => invocation.input = &bad_input,
                OperationInputMutation::CuSeqlens => invocation.cu_seqlens = &bad_cu_seqlens,
                OperationInputMutation::ParameterLength => {
                    invocation.layer_parameters = &bad_layers
                }
                OperationInputMutation::Checkpoints => checkpoints = &[0],
                OperationInputMutation::Activation => {
                    activation = VisionStackActivationStrategy::SeparateBuffers;
                }
                OperationInputMutation::QkvSelectionGeometry
                | OperationInputMutation::QkvSelectionTarget
                | OperationInputMutation::QkvPlanDigestSecond
                | OperationInputMutation::QkvPlanDigestOrder
                | OperationInputMutation::QkvPlanDigestDuplicate
                | OperationInputMutation::QkvPlanDigestMissing => {}
            }

            let failure = match operation {
                CohortOperation::Legacy => run_native_public_legacy_benchmark_cohort_v1(
                    &runtime,
                    plan,
                    &invocation,
                    checkpoints,
                    activation,
                    scripted_probe(Rc::clone(&events), None),
                    legacy_validator(Rc::clone(&events), None),
                ),
                CohortOperation::Qkv => {
                    let selection = match mutation {
                        OperationInputMutation::QkvSelectionGeometry => {
                            alternate_required_fused_selection()
                        }
                        OperationInputMutation::QkvSelectionTarget => {
                            wrong_target_required_fused_selection()
                        }
                        _ => required_fused_selection(),
                    };
                    run_native_public_qkv_benchmark_cohort_v1(
                        &runtime,
                        plan,
                        &invocation,
                        checkpoints,
                        activation,
                        &selection,
                        scripted_probe(Rc::clone(&events), None),
                        qkv_validator(Rc::clone(&events), None),
                    )
                }
            }
            .expect_err("operation inputs must be bound before the first external effect");

            assert_eq!(
                failure.phase(),
                NativeBenchmarkCohortFailurePhaseV1::StaticAdmission,
                "{}/{mutation:?}",
                operation.name(),
            );
            let expected_code = match mutation {
                OperationInputMutation::InputLength
                | OperationInputMutation::CuSeqlens
                | OperationInputMutation::ParameterLength => "invalid_invocation",
                OperationInputMutation::QkvPlanDigestMissing => "invalid_identity",
                _ => "operation_binding_mismatch",
            };
            assert_eq!(
                failure.code(),
                expected_code,
                "{}/{mutation:?}",
                operation.name(),
            );
            assert!(events.borrow().is_empty());
            assert!(runtime.calls.borrow().is_empty());
            assert_eq!(runtime.clock_read_index.get(), 0);
            assert_failure_artifact(&failure, RUN_ID, "static_admission", expected_code, 14, &[]);
        }
    }
}

#[test]
fn invalid_static_leaf_fails_before_probe_clock_operation_or_validation() {
    let mut cases: Vec<(&str, NativeBenchmarkLeafPlanV1, &str)> = Vec::new();

    let mut empty_run_id = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    empty_run_id.run_id.clear();
    cases.push(("empty run id", empty_run_id, "invalid_identity"));

    let mut bad_schedule = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_schedule.protocol.schedule = "caller-selected-order".to_owned();
    refresh_preparation_links(&mut bad_schedule);
    cases.push(("schedule", bad_schedule, "invalid_schedule"));

    cases.push((
        "synchronization",
        linked_legacy_plan(|plan| {
            plan.protocol.synchronization = "submit-without-map".to_owned();
        }),
        "invalid_protocol",
    ));
    cases.push((
        "native clock source",
        linked_legacy_plan(|plan| {
            plan.protocol.clock_source = "performance-now".to_owned();
            plan.load_or_compile.clock_source = "performance-now".to_owned();
        }),
        "invalid_protocol",
    ));
    cases.push((
        "zero clock resolution",
        linked_legacy_plan(|plan| {
            plan.protocol.clock_resolution_ns = 0;
            plan.load_or_compile.clock_resolution_ns = 0;
        }),
        "invalid_protocol",
    ));
    cases.push((
        "validation policy",
        linked_legacy_plan(|plan| {
            plan.protocol.output_validation_policy = "validate-measured-only".to_owned();
        }),
        "invalid_protocol",
    ));
    cases.push((
        "isolation policy",
        linked_legacy_plan(|plan| {
            plan.protocol.isolation_policy = "shared-process".to_owned();
        }),
        "invalid_protocol",
    ));
    cases.push((
        "interruption policy",
        linked_legacy_plan(|plan| {
            plan.protocol.interruption_policy = "retry".to_owned();
        }),
        "invalid_protocol",
    ));
    cases.push((
        "background-load policy",
        linked_legacy_plan(|plan| {
            plan.protocol.background_load_policy = "ignore".to_owned();
        }),
        "invalid_protocol",
    ));
    cases.push((
        "sample execution boundary",
        linked_legacy_plan(|plan| {
            plan.workload.execution_boundary = ExecutionBoundaryV1::LoadOrCompile;
        }),
        "invalid_protocol",
    ));

    let mut too_few_warmups = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        2,
        10,
    );
    refresh_preparation_links(&mut too_few_warmups);
    cases.push(("warmup minimum", too_few_warmups, "invalid_protocol"));

    let mut too_few_measurements = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        9,
    );
    refresh_preparation_links(&mut too_few_measurements);
    cases.push((
        "measurement minimum",
        too_few_measurements,
        "invalid_protocol",
    ));

    let mut sequence_overflow = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        u32::MAX,
        u32::MAX,
    );
    refresh_preparation_links(&mut sequence_overflow);
    cases.push(("sequence overflow", sequence_overflow, "invalid_protocol"));

    let mut bad_build = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_build.passport.build_profile = "debug".to_owned();
    refresh_preparation_links(&mut bad_build);
    cases.push(("build profile", bad_build, "invalid_protocol"));

    let mut bad_passport = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_passport.passport.machine.clear();
    refresh_preparation_links(&mut bad_passport);
    cases.push(("passport identity", bad_passport, "invalid_identity"));

    let mut foreign_backend = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    foreign_backend.passport.backend.kind = BackendKindV1::ChromeWebgpu;
    foreign_backend.passport.backend.browser_version = Some("Chrome 140".to_owned());
    foreign_backend.passport.backend.user_agent = Some("test browser".to_owned());
    foreign_backend.passport.backend.adapter_backend = "webgpu".to_owned();
    foreign_backend.protocol.clock_source = "performance-now".to_owned();
    foreign_backend.load_or_compile.clock_source = "performance-now".to_owned();
    refresh_preparation_links(&mut foreign_backend);
    cases.push(("foreign backend", foreign_backend, "backend_mismatch"));

    let mut bad_manifest_link = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_manifest_link.workload.manifest_sha256 = hash('e');
    refresh_preparation_links(&mut bad_manifest_link);
    cases.push((
        "manifest cross-link",
        bad_manifest_link,
        "cross_link_mismatch",
    ));

    let mut bad_correctness_link = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_correctness_link
        .correctness_anchor
        .expected_checkpoint_sha256 = hash('e');
    cases.push((
        "correctness cross-link",
        bad_correctness_link,
        "cross_link_mismatch",
    ));

    let mut bad_preparation = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_preparation.load_or_compile.workload_blake3 = hash('0');
    cases.push(("preparation link", bad_preparation, "invalid_preparation"));

    let mut zero_preparation = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    zero_preparation.load_or_compile.duration_ns = 0;
    cases.push((
        "zero preparation duration",
        zero_preparation,
        "invalid_preparation",
    ));

    let mut saturated_preparation = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    saturated_preparation.load_or_compile.duration_ns = u64::MAX;
    cases.push((
        "saturated preparation duration",
        saturated_preparation,
        "invalid_preparation",
    ));

    let mut bad_preparation_boundary = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_preparation_boundary.load_or_compile.execution_boundary = ExecutionBoundaryV1::ApiWall;
    cases.push((
        "preparation boundary",
        bad_preparation_boundary,
        "invalid_preparation",
    ));

    let mut failed_preparation = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    failed_preparation.load_or_compile.status = SampleStatusV1::Failed {
        code: "compile_failed".to_owned(),
    };
    cases.push(("failed preparation", failed_preparation, "failed_sample"));

    let mut bad_variant = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_variant.base_descriptor.kernel_variant_id =
        FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1.to_owned();
    cases.push((
        "descriptor variant",
        bad_variant,
        "operation_binding_mismatch",
    ));

    let mut bad_activation = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_activation.base_descriptor.activation_strategy = "separate_buffers".to_owned();
    cases.push((
        "descriptor activation",
        bad_activation,
        "operation_binding_mismatch",
    ));

    let mut bad_topology = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_topology
        .base_descriptor
        .expected_topology
        .dispatch_count += 1;
    cases.push(("descriptor topology", bad_topology, "cross_link_mismatch"));

    let mut bad_resources = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_resources.base_descriptor.logical_gpu_bytes += 1;
    cases.push(("descriptor resources", bad_resources, "cross_link_mismatch"));

    let mut bad_allocated_bytes = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_allocated_bytes.base_descriptor.allocated_gpu_bytes += 4;
    cases.push((
        "descriptor allocated bytes",
        bad_allocated_bytes,
        "cross_link_mismatch",
    ));

    let mut bad_activation_count = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_activation_count.base_descriptor.activation_buffer_count += 1;
    cases.push((
        "descriptor activation count",
        bad_activation_count,
        "cross_link_mismatch",
    ));

    let mut bad_activation_arena = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_activation_arena.base_descriptor.activation_arena_bytes += 4;
    cases.push((
        "descriptor activation arena",
        bad_activation_arena,
        "cross_link_mismatch",
    ));

    let mut bad_scratch_link = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_scratch_link.base_descriptor.scratch_arena_bytes += 4;
    cases.push((
        "descriptor scratch bytes",
        bad_scratch_link,
        "cross_link_mismatch",
    ));

    let mut bad_main_link = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_main_link.base_descriptor.main_buffers_bytes += 4;
    cases.push((
        "descriptor main bytes",
        bad_main_link,
        "cross_link_mismatch",
    ));

    let bad_workload_activation = linked_legacy_plan(|plan| {
        plan.workload.residency_plan.activation_strategy = "separate_buffers".to_owned();
    });
    cases.push((
        "workload activation strategy",
        bad_workload_activation,
        "cross_link_mismatch",
    ));

    let mut bad_output_link = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_output_link.base_descriptor.expected_output_sha256 = hash('e');
    cases.push(("descriptor output", bad_output_link, "cross_link_mismatch"));

    let mut bad_residency = leaf_plan(
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        3,
        10,
    );
    bad_residency.base_descriptor.residency_plan_id = "another-plan".to_owned();
    cases.push(("descriptor residency", bad_residency, "cross_link_mismatch"));

    for (name, plan, expected_code) in cases {
        let mut operation_plans = vec![(CohortOperation::Legacy, plan.clone())];
        if name != "descriptor variant" {
            let qkv_plan = qkv_plan_with_equivalent_static_mutation(name, plan);
            operation_plans.push((CohortOperation::Qkv, qkv_plan));
        }

        for (operation, plan) in operation_plans {
            let expected_run_id = plan.run_id.clone();
            let expected_count = 1_u64
                + u64::from(plan.protocol.warmup_count)
                + u64::from(plan.protocol.measured_count);
            let events = Rc::new(RefCell::new(Vec::new()));
            let runtime = CohortRuntime::new(plan.base_descriptor.clone(), Rc::clone(&events));
            let failure =
                run_test_cohort(operation, &runtime, plan, Rc::clone(&events), None, None)
                    .expect_err("invalid static leaf must fail closed");

            assert_eq!(
                failure.phase(),
                NativeBenchmarkCohortFailurePhaseV1::StaticAdmission,
                "{}/{name}",
                operation.name(),
            );
            assert_eq!(failure.code(), expected_code, "{}/{name}", operation.name(),);
            assert!(
                failure.attempt_log().is_empty(),
                "{}/{name}",
                operation.name(),
            );
            assert_eq!(
                runtime.calls.borrow().len(),
                0,
                "{}/{name}",
                operation.name(),
            );
            assert_eq!(runtime.operation_index.get(), 0, "{name}");
            assert_eq!(runtime.clock_read_index.get(), 0, "{name}");
            assert!(events.borrow().is_empty(), "{name}");
            assert_failure_artifact(
                &failure,
                &expected_run_id,
                "static_admission",
                expected_code,
                expected_count,
                &[],
            );
        }
    }
}
