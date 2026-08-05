use std::collections::BTreeSet;

use pvlc_bench::{
    AssembledBenchmarkEvidenceV1, BackendKindV1, BenchmarkCohortV1, BenchmarkErrorCodeV1,
    BenchmarkEvidenceAssemblyInputV1, BenchmarkPassportV1, BenchmarkSampleAttemptV1,
    BenchmarkSampleV1, ExpectedTopologyV1, GpuTimestampObservationV1, ObservationV1,
    validate_benchmark_leaf_v1, validate_load_or_compile_observation_v1,
};
use pvlc_passes::{VisionQkvStackSelection, prepare_vision_qkv_stack_execution};
use pvlc_runtime_core::{
    VisionEncoderLayerGeometry, VisionEncoderStackInvocation, VisionQkvExecutionPolicy,
    VisionQkvFusedTargetLimits, VisionQkvSelectionOutcome, VisionStackActivationLayoutConfig,
    VisionStackActivationStrategy,
};
use pvlc_runtime_native::{
    BackendKind as NativeBackendKind, NativeCapabilities, NativeRuntime, VisionQkvStackExecution,
    VisionStackExecution,
};

use crate::cohort_types::validate_authored_reference;
use crate::{
    AcceptedVisionStackValidationV1, CollectorError, CollectorErrorCodeV1,
    FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1, LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
    NativeBenchmarkCohortFailurePhaseV1, NativeBenchmarkCohortFailureV1,
    NativeBenchmarkCohortSuccessV1, NativeBenchmarkEnvironmentProbeV1, NativeBenchmarkLeafPlanV1,
    NativeBenchmarkVisionStackValidatorV1, NativePublicVisionStackRuntimeV1,
    VisionStackSampleDescriptorV1, activation_strategy_name,
    collect_native_public_legacy_vision_stack_sample,
    collect_native_public_qkv_vision_stack_sample, validate_descriptor,
};

trait NativeEnvironmentAuthorityV1 {
    fn observe(&mut self) -> Result<ObservationV1, String>;
}

pub(crate) trait NativeCohortRuntimeAuthorityV1: NativePublicVisionStackRuntimeV1 {
    fn cohort_capabilities(&self) -> &NativeCapabilities;
    fn cohort_has_observer(&self) -> bool;
}

impl NativeCohortRuntimeAuthorityV1 for NativeRuntime {
    fn cohort_capabilities(&self) -> &NativeCapabilities {
        self.capabilities()
    }

    fn cohort_has_observer(&self) -> bool {
        self.has_observer()
    }
}

impl NativeEnvironmentAuthorityV1 for &mut NativeBenchmarkEnvironmentProbeV1 {
    fn observe(&mut self) -> Result<ObservationV1, String> {
        NativeBenchmarkEnvironmentProbeV1::observe(self)
    }
}

#[cfg(test)]
impl<F> NativeEnvironmentAuthorityV1 for F
where
    F: FnMut() -> Result<ObservationV1, String>,
{
    fn observe(&mut self) -> Result<ObservationV1, String> {
        self()
    }
}

trait LegacyValidationAuthorityV1 {
    fn is_bound_to(&self, leaf: &NativeBenchmarkLeafPlanV1) -> bool;
    fn validate(
        &mut self,
        execution: &VisionStackExecution,
    ) -> Result<AcceptedVisionStackValidationV1, String>;
}

impl LegacyValidationAuthorityV1 for &mut NativeBenchmarkVisionStackValidatorV1 {
    fn is_bound_to(&self, leaf: &NativeBenchmarkLeafPlanV1) -> bool {
        NativeBenchmarkVisionStackValidatorV1::is_bound_to(self, leaf)
    }

    fn validate(
        &mut self,
        execution: &VisionStackExecution,
    ) -> Result<AcceptedVisionStackValidationV1, String> {
        NativeBenchmarkVisionStackValidatorV1::validate_legacy(self, execution)
    }
}

#[cfg(test)]
impl<F> LegacyValidationAuthorityV1 for F
where
    F: FnMut(&VisionStackExecution) -> Result<AcceptedVisionStackValidationV1, String>,
{
    fn is_bound_to(&self, _leaf: &NativeBenchmarkLeafPlanV1) -> bool {
        true
    }

    fn validate(
        &mut self,
        execution: &VisionStackExecution,
    ) -> Result<AcceptedVisionStackValidationV1, String> {
        self(execution)
    }
}

trait QkvValidationAuthorityV1 {
    fn is_bound_to(&self, leaf: &NativeBenchmarkLeafPlanV1) -> bool;
    fn validate(
        &mut self,
        execution: &VisionQkvStackExecution,
    ) -> Result<AcceptedVisionStackValidationV1, String>;
}

impl QkvValidationAuthorityV1 for &mut NativeBenchmarkVisionStackValidatorV1 {
    fn is_bound_to(&self, leaf: &NativeBenchmarkLeafPlanV1) -> bool {
        NativeBenchmarkVisionStackValidatorV1::is_bound_to(self, leaf)
    }

    fn validate(
        &mut self,
        execution: &VisionQkvStackExecution,
    ) -> Result<AcceptedVisionStackValidationV1, String> {
        NativeBenchmarkVisionStackValidatorV1::validate_qkv(self, execution)
    }
}

#[cfg(test)]
impl<F> QkvValidationAuthorityV1 for F
where
    F: FnMut(&VisionQkvStackExecution) -> Result<AcceptedVisionStackValidationV1, String>,
{
    fn is_bound_to(&self, _leaf: &NativeBenchmarkLeafPlanV1) -> bool {
        true
    }

    fn validate(
        &mut self,
        execution: &VisionQkvStackExecution,
    ) -> Result<AcceptedVisionStackValidationV1, String> {
        self(execution)
    }
}

trait NativeCohortOperationV1<R, P>
where
    R: NativeCohortRuntimeAuthorityV1,
    P: NativeEnvironmentAuthorityV1,
{
    fn static_admit(&self, leaf: &NativeBenchmarkLeafPlanV1) -> Result<(), &'static str>;

    fn collect(
        &mut self,
        runtime: &R,
        descriptor: &VisionStackSampleDescriptorV1,
        probe: &mut P,
    ) -> Result<BenchmarkSampleV1, CollectorError>;
}

struct LegacyCohortOperationV1<'invocation, 'data, V> {
    invocation: &'invocation VisionEncoderStackInvocation<'data>,
    checkpoint_layers: &'invocation [usize],
    activation_strategy: VisionStackActivationStrategy,
    validator: V,
}

impl<R, P, V> NativeCohortOperationV1<R, P> for LegacyCohortOperationV1<'_, '_, V>
where
    R: NativeCohortRuntimeAuthorityV1,
    P: NativeEnvironmentAuthorityV1,
    V: LegacyValidationAuthorityV1,
{
    fn static_admit(&self, leaf: &NativeBenchmarkLeafPlanV1) -> Result<(), &'static str> {
        validate_legacy_operation_binding(
            leaf,
            self.invocation,
            self.checkpoint_layers,
            self.activation_strategy,
        )?;
        self.validator
            .is_bound_to(leaf)
            .then_some(())
            .ok_or("validator_binding_mismatch")
    }

    fn collect(
        &mut self,
        runtime: &R,
        descriptor: &VisionStackSampleDescriptorV1,
        probe: &mut P,
    ) -> Result<BenchmarkSampleV1, CollectorError> {
        let invocation = self.invocation;
        let checkpoint_layers = self.checkpoint_layers;
        let activation_strategy = self.activation_strategy;
        let validator = &mut self.validator;
        collect_native_public_legacy_vision_stack_sample(
            runtime,
            descriptor,
            invocation,
            checkpoint_layers,
            activation_strategy,
            || probe.observe(),
            |execution| validator.validate(execution),
        )
    }
}

struct QkvCohortOperationV1<'invocation, 'data, V> {
    invocation: &'invocation VisionEncoderStackInvocation<'data>,
    checkpoint_layers: &'invocation [usize],
    activation_strategy: VisionStackActivationStrategy,
    selection: &'invocation VisionQkvStackSelection,
    validator: V,
}

impl<R, P, V> NativeCohortOperationV1<R, P> for QkvCohortOperationV1<'_, '_, V>
where
    R: NativeCohortRuntimeAuthorityV1,
    P: NativeEnvironmentAuthorityV1,
    V: QkvValidationAuthorityV1,
{
    fn static_admit(&self, leaf: &NativeBenchmarkLeafPlanV1) -> Result<(), &'static str> {
        validate_qkv_operation_binding(
            leaf,
            self.invocation,
            self.checkpoint_layers,
            self.activation_strategy,
            self.selection,
        )?;
        self.validator
            .is_bound_to(leaf)
            .then_some(())
            .ok_or("validator_binding_mismatch")
    }

    fn collect(
        &mut self,
        runtime: &R,
        descriptor: &VisionStackSampleDescriptorV1,
        probe: &mut P,
    ) -> Result<BenchmarkSampleV1, CollectorError> {
        let invocation = self.invocation;
        let checkpoint_layers = self.checkpoint_layers;
        let activation_strategy = self.activation_strategy;
        let selection = self.selection;
        let validator = &mut self.validator;
        collect_native_public_qkv_vision_stack_sample(
            runtime,
            descriptor,
            invocation,
            checkpoint_layers,
            activation_strategy,
            selection,
            || probe.observe(),
            |execution| validator.validate(execution),
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_legacy_cohort<R, P, V>(
    runtime: &R,
    plan: NativeBenchmarkLeafPlanV1,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
    probe: P,
    validator: V,
) -> Result<NativeBenchmarkCohortSuccessV1, NativeBenchmarkCohortFailureV1>
where
    R: NativeCohortRuntimeAuthorityV1,
    P: NativeEnvironmentAuthorityV1,
    V: LegacyValidationAuthorityV1,
{
    run_native_benchmark_cohort_engine_v1(
        runtime,
        plan,
        probe,
        LegacyCohortOperationV1 {
            invocation,
            checkpoint_layers,
            activation_strategy,
            validator,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn run_qkv_cohort<R, P, V>(
    runtime: &R,
    plan: NativeBenchmarkLeafPlanV1,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
    selection: &VisionQkvStackSelection,
    probe: P,
    validator: V,
) -> Result<NativeBenchmarkCohortSuccessV1, NativeBenchmarkCohortFailureV1>
where
    R: NativeCohortRuntimeAuthorityV1,
    P: NativeEnvironmentAuthorityV1,
    V: QkvValidationAuthorityV1,
{
    run_native_benchmark_cohort_engine_v1(
        runtime,
        plan,
        probe,
        QkvCohortOperationV1 {
            invocation,
            checkpoint_layers,
            activation_strategy,
            selection,
            validator,
        },
    )
}

#[cfg(not(test))]
#[allow(clippy::too_many_arguments)]
pub fn run_native_public_legacy_benchmark_cohort_v1(
    runtime: &NativeRuntime,
    plan: NativeBenchmarkLeafPlanV1,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
    probe: &mut NativeBenchmarkEnvironmentProbeV1,
    validator: &mut NativeBenchmarkVisionStackValidatorV1,
) -> Result<NativeBenchmarkCohortSuccessV1, NativeBenchmarkCohortFailureV1> {
    run_legacy_cohort(
        runtime,
        plan,
        invocation,
        checkpoint_layers,
        activation_strategy,
        probe,
        validator,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
#[allow(private_bounds)]
pub(crate) fn run_native_public_legacy_benchmark_cohort_v1<R, P, V>(
    runtime: &R,
    plan: NativeBenchmarkLeafPlanV1,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
    probe: P,
    validator: V,
) -> Result<NativeBenchmarkCohortSuccessV1, NativeBenchmarkCohortFailureV1>
where
    R: NativeCohortRuntimeAuthorityV1,
    P: NativeEnvironmentAuthorityV1,
    V: LegacyValidationAuthorityV1,
{
    run_legacy_cohort(
        runtime,
        plan,
        invocation,
        checkpoint_layers,
        activation_strategy,
        probe,
        validator,
    )
}

#[cfg(not(test))]
#[allow(clippy::too_many_arguments)]
pub fn run_native_public_qkv_benchmark_cohort_v1(
    runtime: &NativeRuntime,
    plan: NativeBenchmarkLeafPlanV1,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
    selection: &VisionQkvStackSelection,
    probe: &mut NativeBenchmarkEnvironmentProbeV1,
    validator: &mut NativeBenchmarkVisionStackValidatorV1,
) -> Result<NativeBenchmarkCohortSuccessV1, NativeBenchmarkCohortFailureV1> {
    run_qkv_cohort(
        runtime,
        plan,
        invocation,
        checkpoint_layers,
        activation_strategy,
        selection,
        probe,
        validator,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
#[allow(private_bounds)]
pub(crate) fn run_native_public_qkv_benchmark_cohort_v1<R, P, V>(
    runtime: &R,
    plan: NativeBenchmarkLeafPlanV1,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
    selection: &VisionQkvStackSelection,
    probe: P,
    validator: V,
) -> Result<NativeBenchmarkCohortSuccessV1, NativeBenchmarkCohortFailureV1>
where
    R: NativeCohortRuntimeAuthorityV1,
    P: NativeEnvironmentAuthorityV1,
    V: QkvValidationAuthorityV1,
{
    run_qkv_cohort(
        runtime,
        plan,
        invocation,
        checkpoint_layers,
        activation_strategy,
        selection,
        probe,
        validator,
    )
}

fn run_native_benchmark_cohort_engine_v1<R, P, O>(
    runtime: &R,
    plan: NativeBenchmarkLeafPlanV1,
    mut probe: P,
    mut operation: O,
) -> Result<NativeBenchmarkCohortSuccessV1, NativeBenchmarkCohortFailureV1>
where
    R: NativeCohortRuntimeAuthorityV1,
    P: NativeEnvironmentAuthorityV1,
    O: NativeCohortOperationV1<R, P>,
{
    let expected_attempt_count = 1_u64
        .checked_add(u64::from(plan.protocol.warmup_count))
        .and_then(|value| value.checked_add(u64::from(plan.protocol.measured_count)))
        .expect("two u32 cohort counts plus one always fit u64");
    let run_id = plan.run_id.clone();
    if let Err(code) = validate_static_leaf(&plan, expected_attempt_count)
        .and_then(|()| operation.static_admit(&plan))
        .and_then(|()| validate_native_runtime_authority_v1(runtime, &plan.passport))
        .and_then(|()| validate_static_descriptor_and_reference(&plan))
    {
        return Err(NativeBenchmarkCohortFailureV1::new(
            run_id,
            NativeBenchmarkCohortFailurePhaseV1::StaticAdmission,
            code,
            expected_attempt_count,
            Vec::new(),
        ));
    }

    let attempt_capacity = usize::try_from(expected_attempt_count)
        .expect("static admission bounds the cohort count to u32::MAX");
    let warmup_count = usize::try_from(plan.protocol.warmup_count)
        .expect("u32 cohort counts fit every supported native target");
    let mut attempt_log = Vec::with_capacity(attempt_capacity.min(4_096));
    let mut timestamp_pairs = BTreeSet::new();
    for sequence in 0..attempt_capacity {
        let (cohort, planned_slot) = schedule_identity(sequence, warmup_count);
        let mut descriptor = plan.base_descriptor.clone();
        descriptor.index = planned_slot;
        descriptor.schedule_slot = planned_slot;
        let sample = match operation.collect(runtime, &descriptor, &mut probe) {
            Ok(sample) => sample,
            Err(error) => {
                let code = collector_error_code(error.code());
                attempt_log.push(failed_attempt(sequence, cohort, planned_slot, code));
                return Err(NativeBenchmarkCohortFailureV1::new(
                    run_id,
                    NativeBenchmarkCohortFailurePhaseV1::Attempt,
                    code,
                    expected_attempt_count,
                    attempt_log,
                ));
            }
        };
        if let GpuTimestampObservationV1::Available {
            begin_ticks,
            end_ticks,
            ..
        } = sample.gpu_timestamp
            && !timestamp_pairs.insert((begin_ticks, end_ticks))
        {
            let code = "stale_timestamp";
            attempt_log.push(failed_attempt(sequence, cohort, planned_slot, code));
            return Err(NativeBenchmarkCohortFailureV1::new(
                run_id,
                NativeBenchmarkCohortFailurePhaseV1::Attempt,
                code,
                expected_attempt_count,
                attempt_log,
            ));
        }
        attempt_log.push(BenchmarkSampleAttemptV1::Passed {
            sequence: u32::try_from(sequence).expect("static admission bounds the sequence to u32"),
            cohort,
            planned_slot,
            sample,
        });
    }

    let attempt_count = attempt_log.len();
    let failed_assembly_journal = attempt_log.clone();
    let assembled = match AssembledBenchmarkEvidenceV1::assemble(BenchmarkEvidenceAssemblyInputV1 {
        passport: plan.passport,
        workload: plan.workload,
        correctness_anchor: plan.correctness_anchor,
        protocol: plan.protocol,
        load_or_compile: plan.load_or_compile,
        attempt_log,
    }) {
        Ok(assembled) => assembled,
        Err(error) => {
            return Err(NativeBenchmarkCohortFailureV1::new(
                run_id,
                NativeBenchmarkCohortFailurePhaseV1::Attempt,
                benchmark_error_code(error.code()),
                expected_attempt_count,
                failed_assembly_journal,
            ));
        }
    };
    Ok(NativeBenchmarkCohortSuccessV1::new(
        run_id,
        attempt_count,
        assembled,
    ))
}

pub(crate) fn validate_native_runtime_authority_v1<R>(
    runtime: &R,
    passport: &BenchmarkPassportV1,
) -> Result<(), &'static str>
where
    R: NativeCohortRuntimeAuthorityV1,
{
    let capabilities = runtime.cohort_capabilities();
    let adapter_backend = match capabilities.backend {
        NativeBackendKind::Noop => "noop",
        NativeBackendKind::Vulkan => "vulkan",
        NativeBackendKind::Metal => "metal",
        NativeBackendKind::Dx12 => "dx12",
        NativeBackendKind::Gl => "gl",
        NativeBackendKind::BrowserWebGpu => "browser_webgpu",
    };
    let exact_limits = [
        ("max_buffer_size", capabilities.max_buffer_size),
        (
            "max_compute_invocations_per_workgroup",
            u64::from(capabilities.max_compute_invocations_per_workgroup),
        ),
        (
            "max_compute_workgroup_size_x",
            u64::from(capabilities.max_compute_workgroup_size_x),
        ),
        (
            "max_compute_workgroup_size_y",
            u64::from(capabilities.max_compute_workgroup_size_y),
        ),
        (
            "max_compute_workgroup_size_z",
            u64::from(capabilities.max_compute_workgroup_size_z),
        ),
        (
            "max_compute_workgroup_storage_size",
            u64::from(capabilities.max_compute_workgroup_storage_size),
        ),
        (
            "max_compute_workgroups_per_dimension",
            u64::from(capabilities.max_compute_workgroups_per_dimension),
        ),
        (
            "max_storage_buffer_binding_size",
            capabilities.max_storage_buffer_binding_size,
        ),
        (
            "max_storage_buffers_per_shader_stage",
            u64::from(capabilities.max_storage_buffers_per_shader_stage),
        ),
        (
            "min_storage_buffer_offset_alignment",
            u64::from(capabilities.min_storage_buffer_offset_alignment),
        ),
    ];
    if runtime.cohort_has_observer()
        || capabilities.adapter_name != passport.adapter_name
        || passport.backend.adapter_backend != adapter_backend
        || capabilities.timestamp_query != passport.backend.timestamp_query
        || passport.backend.limits.len() != exact_limits.len()
        || exact_limits
            .iter()
            .any(|(name, value)| passport.backend.limits.get(*name) != Some(value))
    {
        return Err("runtime_binding_mismatch");
    }
    Ok(())
}

fn validate_static_leaf(
    plan: &NativeBenchmarkLeafPlanV1,
    expected_attempt_count: u64,
) -> Result<(), &'static str> {
    if plan.run_id.trim().is_empty() {
        return Err("invalid_identity");
    }
    validate_benchmark_leaf_v1(
        &plan.passport,
        &plan.workload,
        &plan.correctness_anchor,
        &plan.protocol,
    )
    .map_err(|error| benchmark_error_code(error.code()))?;
    validate_load_or_compile_observation_v1(
        &plan.load_or_compile,
        &plan.passport,
        &plan.workload,
        &plan.protocol,
    )
    .map_err(|error| benchmark_error_code(error.code()))?;
    if plan.passport.backend.kind != BackendKindV1::NativeWgpu {
        return Err("backend_mismatch");
    }
    if expected_attempt_count > u64::from(u32::MAX) {
        return Err("invalid_protocol");
    }
    Ok(())
}

fn validate_static_descriptor_and_reference(
    plan: &NativeBenchmarkLeafPlanV1,
) -> Result<(), &'static str> {
    validate_descriptor(&plan.base_descriptor)
        .map_err(|error| collector_error_code(error.code()))?;
    validate_authored_reference(plan).map_err(|error| collector_error_code(error.code()))
}

fn validate_legacy_operation_binding(
    leaf: &NativeBenchmarkLeafPlanV1,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
) -> Result<(), &'static str> {
    validate_exact_operation_identity(
        leaf,
        LEGACY_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "disabled",
        "disabled",
        activation_strategy,
    )?;
    validate_descriptor_workload_cross_links(leaf)?;
    validate_invocation_binding(
        leaf,
        invocation,
        checkpoint_layers,
        activation_strategy,
        false,
    )
}

fn validate_qkv_operation_binding(
    leaf: &NativeBenchmarkLeafPlanV1,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
    selection: &VisionQkvStackSelection,
) -> Result<(), &'static str> {
    validate_exact_operation_identity(
        leaf,
        FUSED_QKV_VISION_STACK_KERNEL_VARIANT_ID_V1,
        "required",
        "fused",
        activation_strategy,
    )?;
    validate_descriptor_workload_cross_links(leaf)?;
    validate_invocation_binding(
        leaf,
        invocation,
        checkpoint_layers,
        activation_strategy,
        true,
    )?;
    if selection.policy() != VisionQkvExecutionPolicy::Required
        || selection.outcome() != VisionQkvSelectionOutcome::Fused
    {
        return Err("operation_binding_mismatch");
    }
    let overlay = selection.overlay().ok_or("operation_binding_mismatch")?;
    let target = target_limits_from_passport(leaf)?;
    if overlay.target_limits() != target
        || overlay
            .layers()
            .iter()
            .map(|layer| layer.canonical_plan_blake3_hex())
            .ne(leaf
                .workload
                .ordered_layer_plans_blake3
                .iter()
                .map(String::as_str))
    {
        return Err("operation_binding_mismatch");
    }
    let geometry = VisionEncoderLayerGeometry {
        tokens: invocation.tokens,
        hidden_size: invocation.hidden_size,
        attention_heads: invocation.attention_heads,
        head_dim: invocation.head_dim,
        intermediate_size: invocation.intermediate_size,
        layer_norm_epsilon: invocation.layer_norm_epsilon,
        cu_seqlens: invocation.cu_seqlens,
    }
    .plan()
    .map_err(|_| "invalid_invocation")?;
    prepare_vision_qkv_stack_execution(
        overlay,
        invocation.layer_parameters.len(),
        &geometry,
        target,
    )
    .map(|_| ())
    .map_err(|_| "operation_binding_mismatch")
}

fn validate_exact_operation_identity(
    leaf: &NativeBenchmarkLeafPlanV1,
    variant: &str,
    qkv_policy: &str,
    qkv_outcome: &str,
    activation_strategy: VisionStackActivationStrategy,
) -> Result<(), &'static str> {
    if leaf.base_descriptor.kernel_variant_id != variant
        || leaf.workload.kernel_variant.id != variant
        || leaf.workload.qkv_policy != qkv_policy
        || leaf.workload.qkv_outcome != qkv_outcome
        || leaf.base_descriptor.activation_strategy != activation_strategy_name(activation_strategy)
    {
        return Err("operation_binding_mismatch");
    }
    Ok(())
}

fn validate_descriptor_workload_cross_links(
    leaf: &NativeBenchmarkLeafPlanV1,
) -> Result<(), &'static str> {
    let descriptor = &leaf.base_descriptor;
    let workload = &leaf.workload;
    let residency = &workload.residency_plan;
    if descriptor.kernel_variant_id != workload.kernel_variant.id
        || descriptor.expected_topology != workload.kernel_variant.expected_topology
        || descriptor.residency_plan_id != residency.id
        || descriptor.expected_output_sha256 != workload.checkpoint_sha256
        || descriptor.expected_output_sha256 != leaf.validation_reference.accepted.output_sha256
        || descriptor.logical_gpu_bytes != residency.logical_gpu_bytes
        || descriptor.allocated_gpu_bytes != residency.allocated_gpu_bytes
        || descriptor.activation_strategy != residency.activation_strategy
        || descriptor.activation_buffer_count != residency.activation_buffer_count
        || descriptor.activation_arena_bytes != residency.activation_arena_bytes
        || descriptor.scratch_arena_bytes != residency.scratch_arena_bytes
        || descriptor.main_buffers_bytes != residency.main_buffers_bytes
    {
        return Err("cross_link_mismatch");
    }
    Ok(())
}

fn validate_invocation_binding(
    leaf: &NativeBenchmarkLeafPlanV1,
    invocation: &VisionEncoderStackInvocation<'_>,
    checkpoint_layers: &[usize],
    activation_strategy: VisionStackActivationStrategy,
    fused_qkv: bool,
) -> Result<(), &'static str> {
    if invocation.tokens != leaf.workload.tokens
        || invocation.hidden_size != leaf.workload.hidden_size
        || invocation.layer_parameters.len()
            != usize::try_from(leaf.workload.layer_count)
                .map_err(|_| "operation_binding_mismatch")?
    {
        return Err("operation_binding_mismatch");
    }
    let stack = invocation
        .plan(checkpoint_layers)
        .map_err(|_| "invalid_invocation")?;
    if leaf
        .validation_reference
        .expected_checkpoints
        .keys()
        .copied()
        .ne(checkpoint_layers.iter().copied())
    {
        return Err("operation_binding_mismatch");
    }
    let checkpoint_policy = checkpoint_policy(checkpoint_layers);
    let mut readback_policy = checkpoint_policy.clone();
    if fused_qkv {
        readback_policy.push_str("-plus-qkv-canaries");
    }
    if leaf.workload.checkpoint_policy != checkpoint_policy
        || leaf.workload.readback_policy != readback_policy
    {
        return Err("operation_binding_mismatch");
    }

    let alignment = leaf
        .passport
        .backend
        .limits
        .get("min_storage_buffer_offset_alignment")
        .copied()
        .ok_or("operation_binding_mismatch")?;
    let activation = stack
        .activation_layout(VisionStackActivationLayoutConfig {
            allow_aliasing: activation_strategy == VisionStackActivationStrategy::StaticArenaAlias,
            storage_buffer_offset_alignment: alignment,
            arena_alignment: alignment,
        })
        .map_err(|_| "invalid_invocation")?;
    let dispatch_count = if fused_qkv {
        stack
            .dispatch_count
            .checked_sub(2_usize.saturating_mul(stack.layer_count))
            .ok_or("operation_binding_mismatch")?
    } else {
        stack.dispatch_count
    };
    let topology = ExpectedTopologyV1 {
        dispatch_count: u64::try_from(dispatch_count).map_err(|_| "operation_binding_mismatch")?,
        compute_pass_count: u64::try_from(stack.compute_pass_count)
            .map_err(|_| "operation_binding_mismatch")?,
        command_buffer_count: 1,
        submission_count: 1,
        map_count: 1,
    };
    let logical_gpu_bytes = stack
        .resident_weight_bytes
        .checked_add(activation.total_activation_bytes)
        .and_then(|value| value.checked_add(stack.readback_bytes))
        .ok_or("operation_binding_mismatch")?;
    let residency = &leaf.workload.residency_plan;
    if topology != leaf.workload.kernel_variant.expected_topology
        || topology != leaf.base_descriptor.expected_topology
        || u64::try_from(activation.physical_buffer_count)
            .map_err(|_| "operation_binding_mismatch")?
            != residency.activation_buffer_count
        || activation.total_activation_bytes != residency.activation_arena_bytes
        || activation.scratch_arena_bytes != residency.scratch_arena_bytes
        || activation.main_buffers_bytes != residency.main_buffers_bytes
        || logical_gpu_bytes != residency.logical_gpu_bytes
        || stack.resident_weight_bytes != residency.max_resident_shard_bytes
    {
        return Err("operation_binding_mismatch");
    }
    Ok(())
}

fn checkpoint_policy(checkpoint_layers: &[usize]) -> String {
    let depths = checkpoint_layers
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("-");
    let mut policy = String::from("depth-");
    policy.push_str(&depths);
    policy.push_str("-final");
    policy
}

fn target_limits_from_passport(
    leaf: &NativeBenchmarkLeafPlanV1,
) -> Result<VisionQkvFusedTargetLimits, &'static str> {
    let limit = |name| {
        leaf.passport
            .backend
            .limits
            .get(name)
            .copied()
            .ok_or("operation_binding_mismatch")
    };
    Ok(VisionQkvFusedTargetLimits {
        min_storage_buffer_offset_alignment: u32::try_from(limit(
            "min_storage_buffer_offset_alignment",
        )?)
        .map_err(|_| "operation_binding_mismatch")?,
        max_storage_buffers_per_shader_stage: u32::try_from(limit(
            "max_storage_buffers_per_shader_stage",
        )?)
        .map_err(|_| "operation_binding_mismatch")?,
        max_storage_buffer_binding_size: limit("max_storage_buffer_binding_size")?,
        max_buffer_size: limit("max_buffer_size")?,
        max_compute_workgroups_per_dimension: u32::try_from(limit(
            "max_compute_workgroups_per_dimension",
        )?)
        .map_err(|_| "operation_binding_mismatch")?,
    })
}

fn schedule_identity(sequence: usize, warmup_count: usize) -> (BenchmarkCohortV1, u32) {
    if sequence == 0 {
        (BenchmarkCohortV1::Cold, 0)
    } else if sequence <= warmup_count {
        (
            BenchmarkCohortV1::Warmup,
            u32::try_from(sequence - 1).expect("admitted sequence fits u32"),
        )
    } else {
        (
            BenchmarkCohortV1::Measured,
            u32::try_from(sequence - warmup_count - 1).expect("admitted sequence fits u32"),
        )
    }
}

fn failed_attempt(
    sequence: usize,
    cohort: BenchmarkCohortV1,
    planned_slot: u32,
    code: &str,
) -> BenchmarkSampleAttemptV1 {
    BenchmarkSampleAttemptV1::Failed {
        sequence: u32::try_from(sequence).expect("admitted sequence fits u32"),
        cohort,
        planned_slot,
        code: code.to_owned(),
    }
}

const fn collector_error_code(code: CollectorErrorCodeV1) -> &'static str {
    match code {
        CollectorErrorCodeV1::NotImplemented => "not_implemented",
        CollectorErrorCodeV1::InvalidDescriptor => "invalid_descriptor",
        CollectorErrorCodeV1::EnvironmentProbeFailed => "environment_probe_failed",
        CollectorErrorCodeV1::ClockFailed => "clock_failed",
        CollectorErrorCodeV1::NonMonotonicClock => "non_monotonic_clock",
        CollectorErrorCodeV1::ExecutionFailed => "execution_failed",
        CollectorErrorCodeV1::ValidationFailed => "validation_failed",
        CollectorErrorCodeV1::InvalidDiagnostics => "invalid_diagnostics",
        CollectorErrorCodeV1::InvalidQueueObservation => "invalid_queue_observation",
        CollectorErrorCodeV1::InvalidTimestamp => "invalid_timestamp",
        CollectorErrorCodeV1::TopologyMismatch => "topology_mismatch",
        CollectorErrorCodeV1::ResourceMismatch => "resource_mismatch",
        CollectorErrorCodeV1::CrossLinkMismatch => "cross_link_mismatch",
        CollectorErrorCodeV1::OperationBindingMismatch => "operation_binding_mismatch",
    }
}

const fn benchmark_error_code(code: BenchmarkErrorCodeV1) -> &'static str {
    match code {
        BenchmarkErrorCodeV1::NotImplemented => "not_implemented",
        BenchmarkErrorCodeV1::SchemaMismatch => "schema_mismatch",
        BenchmarkErrorCodeV1::UnsupportedSchema => "unsupported_schema",
        BenchmarkErrorCodeV1::UnsupportedClaim => "unsupported_claim",
        BenchmarkErrorCodeV1::NonCanonical => "non_canonical",
        BenchmarkErrorCodeV1::SelfHashMismatch => "self_hash_mismatch",
        BenchmarkErrorCodeV1::SummaryMismatch => "summary_mismatch",
        BenchmarkErrorCodeV1::InvalidIdentity => "invalid_identity",
        BenchmarkErrorCodeV1::InvalidEnvironment => "invalid_environment",
        BenchmarkErrorCodeV1::InvalidProtocol => "invalid_protocol",
        BenchmarkErrorCodeV1::InvalidAttemptJournal => "invalid_attempt_journal",
        BenchmarkErrorCodeV1::InvalidPreparation => "invalid_preparation",
        BenchmarkErrorCodeV1::InvalidInteger => "invalid_integer",
        BenchmarkErrorCodeV1::InvalidDuration => "invalid_duration",
        BenchmarkErrorCodeV1::InvalidIndex => "invalid_index",
        BenchmarkErrorCodeV1::InvalidSchedule => "invalid_schedule",
        BenchmarkErrorCodeV1::InvalidQueueObservation => "invalid_queue_observation",
        BenchmarkErrorCodeV1::InvalidTimestamp => "invalid_timestamp",
        BenchmarkErrorCodeV1::TimestampOverflow => "timestamp_overflow",
        BenchmarkErrorCodeV1::StaleTimestamp => "stale_timestamp",
        BenchmarkErrorCodeV1::CrossLinkMismatch => "cross_link_mismatch",
        BenchmarkErrorCodeV1::TopologyMismatch => "topology_mismatch",
        BenchmarkErrorCodeV1::ResourceMismatch => "resource_mismatch",
        BenchmarkErrorCodeV1::FailedSample => "failed_sample",
    }
}
