use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    fmt,
    future::Future,
    num::NonZeroU64,
    rc::Rc,
    sync::{Arc, Mutex},
};

use futures_channel::oneshot;
use pvlc_pack::{
    VisionStackShardKind, VisionStackShardManifest, VisionStackShardObservation,
    VisionStackShardPlan, VisionStackShardProtocol, parse_vision_stack_shard_manifest,
};
use pvlc_passes::{VisionQkvPhysicalExecutionSpec, VisionQkvStackSelection};
use pvlc_runtime_core::{
    DecoderWeightStorage, InvocationInput, InvocationPlan, KernelId, KernelInvocation,
    LinearWeightLayout, OwnedProjectorInvocation, OwnedVisionEncoderLayerInvocation,
    ProjectorGeometry, ProjectorInvocation, ProjectorPlan, ProjectorReadback, ProjectorStage,
    VISION_QKV_CANARY_U32,
    VisionEncoderLayerInvocation, VisionEncoderLayerPlan, VisionEncoderLayerSpatial2dPlan,
    VisionEncoderLayerStage, VisionEncoderStackSpatial2dPlan, VisionLayerReadback,
    VisionPatchProjectionBytesDescriptor, VisionPatchProjectionBytesPlan, VisionQkvExecutionPolicy,
    VisionQkvFusedPlan, VisionQkvFusedTargetLimits, VisionQkvSelectionOutcome,
    VisionRope2dDescriptor, VisionRopeSpecialization, VisionStackActivationLayout,
    VisionStackActivationLayoutConfig, VisionStackActivationStrategy,
    plan_vision_qkv_fused_f16_weight_geometry,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::{JsCast, prelude::*};
use wasm_bindgen_futures::JsFuture;
use wgpu::util::DeviceExt;

mod decoder_full_layer_session;
mod decoder_kv_session;
mod decoder_layer_session;
mod decoder_stack_session;
mod vision_stack_first_effect;

use vision_stack_first_effect::run_first_webgpu_effect;

use crate::{
    AbortDisposition, AsyncSessionOwner, BrowserVisionQkvBeginExecutionEvidence,
    BrowserVisionQkvExecutionEvidencePlan, BrowserVisionQkvFinalExecutionEvidence,
    BrowserVisionStackLayerWeightPlan, CompletionOutcome, VISION_STACK_PREFIX_CANARY_U32,
    VISION_STACK_SCRATCH_POISON_U32, VISION_STACK_SUFFIX_CANARY_U32, VisionQkvCompilerCapabilities,
    VisionQkvCompilerHandoff, VisionQkvCompilerReadbackRequest,
    VisionQkvSelectionEvidencePropagation, VisionQkvWebBindGroupEntry, VisionQkvWebBindGroupKind,
    VisionQkvWebBindingResource, VisionQkvWebPhysicalBuffer, VisionQkvWebPhysicalCommand,
    VisionQkvWebPhysicalCommandEffectSink, VisionQkvWebPhysicalCommandExecutionError,
    VisionQkvWebPhysicalCommandPhase, VisionQkvWebPhysicalCommandPlan, VisionStackEvidenceError,
    VisionStackLegacyDiagnosticsRecord, VisionStackMemoryHardening, VisionStackMemoryHardeningPlan,
    build_vision_qkv_selection_evidence_propagation, compile_vision_qkv_stack_handoff,
    execute_vision_qkv_web_physical_commands, plan_vision_qkv_web_physical_commands,
    prepare_browser_vision_stack_execution, prepare_vision_qkv_stack_handoff_execution,
    serialize_vision_stack_qkv_begin_status_json, validate_vision_qkv_stack_handoff_binding,
    vision_stack_causal::{
        VisionStackAsyncOperation, VisionStackErrorScopeAuthority, VisionStackErrorScopePopAttempt,
        VisionStackErrorScopePushFailure, VisionStackGpuEffectBoundary, VisionStackPostEffectToken,
        VisionStackResidentCacheDisposition, VisionStackResidentFailure,
        VisionStackResidentWeightCache, VisionStackStreamingFailure,
        VisionStackStreamingLayerSchedule, VisionStackStreamingWeightCache,
        VisionStackStreamingWeightRange, collect_vision_stack_session_resources,
        complete_vision_stack_async_operation, coordinate_vision_stack_completion_busy,
        observe_vision_stack_error_scope_pop, push_vision_stack_error_scope_or_drain,
        require_vision_stack_error_scope_admission_available,
        run_vision_stack_error_scoped_operation, run_vision_stack_operation_transaction,
        run_vision_stack_resident_cold_layer,
        run_vision_stack_streaming_session_layer as run_causal_vision_stack_streaming_session_layer,
    },
    vision_stack_resident_weight_key,
};

const CHECKED_SCOPES: [&str; 3] = ["validation", "out_of_memory", "internal"];
const VISION_LAYER_KERNELS: [KernelId; 5] = [
    KernelId::LayerNormF32,
    KernelId::VisionPatchProjectionF32,
    KernelId::VisionAttentionF32,
    KernelId::AddF32,
    KernelId::GeluTanhF32,
];
const VISION_QKV_STACK_KERNELS: [KernelId; 6] = [
    KernelId::LayerNormF32,
    KernelId::VisionPatchProjectionF32,
    KernelId::VisionAttentionF32,
    KernelId::AddF32,
    KernelId::GeluTanhF32,
    KernelId::VisionQkvFusedF32,
];
const PROJECTOR_KERNELS: [KernelId; 4] = [
    KernelId::LayerNormF32,
    KernelId::GeluErfF32,
    KernelId::VisionPatchProjectionF32,
    KernelId::ProjectorMerge2x2F32,
];
const PROJECTOR_F16_KERNELS: [KernelId; 4] = [
    KernelId::LayerNormF16,
    KernelId::ProjectorMerge2x2F16,
    KernelId::LinearProjectionF16,
    KernelId::GeluErfF16,
];
const VISION_LAYER_UNIFORM_BYTES: u64 = 16;
const JS_BRIDGE_CHUNK_BYTES: u32 = 8 * 1024 * 1024;
const ENABLE_TILED_FP16_QKV: bool = false;

#[derive(Debug)]
struct BrowserVisionStackError(String);

impl fmt::Display for BrowserVisionStackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl From<String> for BrowserVisionStackError {
    fn from(error: String) -> Self {
        Self(error)
    }
}

impl From<BrowserVisionStackError> for String {
    fn from(error: BrowserVisionStackError) -> Self {
        error.0
    }
}

impl From<VisionStackEvidenceError> for BrowserVisionStackError {
    fn from(error: VisionStackEvidenceError) -> Self {
        Self(error.to_string())
    }
}

impl From<BrowserVisionStackError> for JsValue {
    fn from(error: BrowserVisionStackError) -> Self {
        js_sys::Error::new(&error.0).into()
    }
}

impl From<VisionStackStreamingFailure<BrowserVisionStackError>> for String {
    fn from(error: VisionStackStreamingFailure<BrowserVisionStackError>) -> Self {
        match error {
            VisionStackStreamingFailure::Unavailable(error) => {
                format!("vision-stack session is unavailable: {error:?}")
            }
            VisionStackStreamingFailure::Admission(error) => error.to_string(),
            VisionStackStreamingFailure::CacheLengthMismatch {
                slot,
                expected_bytes,
                actual_bytes,
            } => format!(
                "vision-stack streaming weight slot {slot} has {actual_bytes} bytes; cached geometry requires {expected_bytes}"
            ),
            VisionStackStreamingFailure::Effect { error, boundary } => {
                format!("vision-stack streaming GPU effect failed at {boundary:?}: {error}")
            }
            VisionStackStreamingFailure::Completion(outcome) => {
                format!("vision-stack streaming session completed unexpectedly: {outcome:?}")
            }
        }
    }
}

struct PreparedVisionStackFirstErrorScope<'a> {
    authority: VisionStackErrorScopeAuthority<'a, ScopeKind>,
    raw_device: &'a JsValue,
    push: js_sys::Function,
    pop: js_sys::Function,
}

struct BrowserVisionStackErrorScopes<'a> {
    authority: VisionStackErrorScopeAuthority<'a, ScopeKind>,
    raw_device: &'a JsValue,
    pop: js_sys::Function,
}

enum PreparedFirstWebGpuEffect<'a> {
    PushErrorScope {
        raw_device: &'a JsValue,
        push: &'a js_sys::Function,
        filter: &'static str,
    },
    CreateShaderModule {
        label: &'a str,
        source: &'a str,
    },
    CreateComputePipeline {
        label: &'a str,
        module: &'a wgpu::ShaderModule,
        entry_point: &'a str,
    },
}

enum FirstWebGpuEffectOutput {
    ErrorScope,
    ShaderModule(wgpu::ShaderModule),
    ComputePipeline(wgpu::ComputePipeline),
}

fn benchmark_error_js(error: pvlc_bench::BenchmarkError) -> JsValue {
    js_sys::Error::new(error.code().as_str()).into()
}

#[wasm_bindgen(inline_js = r#"
const PVLC_UINT8_ARRAY = Uint8Array;
const PVLC_UINT8_ARRAY_PROTOTYPE = PVLC_UINT8_ARRAY.prototype;
const PVLC_TYPED_ARRAY_CONSTRUCTOR = Object.getPrototypeOf(PVLC_UINT8_ARRAY);
const PVLC_TYPED_ARRAY_PROTOTYPE = Object.getPrototypeOf(PVLC_UINT8_ARRAY_PROTOTYPE);
const PVLC_GET_PROTOTYPE_OF = Object.getPrototypeOf;
const PVLC_GET_OWN_PROPERTY_DESCRIPTOR = Object.getOwnPropertyDescriptor;
const PVLC_REFLECT_APPLY = Reflect.apply;
const PVLC_ERROR = Error;
const PVLC_GLOBAL = globalThis;
const PVLC_SYMBOL_SPECIES = Symbol.species;
const PVLC_UINT8_ARRAY_GLOBAL_DESCRIPTOR =
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_GLOBAL, "Uint8Array");
const PVLC_UINT8_ARRAY_CONSTRUCTOR_DESCRIPTOR =
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY_PROTOTYPE, "constructor");
const PVLC_UINT8_ARRAY_SPECIES_DESCRIPTOR =
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY, PVLC_SYMBOL_SPECIES);
const PVLC_TYPED_ARRAY_SPECIES_DESCRIPTOR =
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(
        PVLC_TYPED_ARRAY_CONSTRUCTOR,
        PVLC_SYMBOL_SPECIES,
    );
const PVLC_BRIDGE_PROPERTY_NAMES = [
    "byteLength",
    "length",
    "set",
    "subarray",
    "slice",
];
const PVLC_TYPED_ARRAY_PROPERTY_DESCRIPTORS = [
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_TYPED_ARRAY_PROTOTYPE, "byteLength"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_TYPED_ARRAY_PROTOTYPE, "length"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_TYPED_ARRAY_PROTOTYPE, "set"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_TYPED_ARRAY_PROTOTYPE, "subarray"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_TYPED_ARRAY_PROTOTYPE, "slice"),
];
const PVLC_UINT8_ARRAY_PROPERTY_DESCRIPTORS = [
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY_PROTOTYPE, "byteLength"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY_PROTOTYPE, "length"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY_PROTOTYPE, "set"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY_PROTOTYPE, "subarray"),
    PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY_PROTOTYPE, "slice"),
];
const PVLC_TYPED_ARRAY_BYTE_LENGTH_GETTER =
    PVLC_TYPED_ARRAY_PROPERTY_DESCRIPTORS[0]?.get;
const PVLC_TYPED_ARRAY_SET = PVLC_TYPED_ARRAY_PROPERTY_DESCRIPTORS[2]?.value;

function pvlcOwnDescriptorField(descriptor, name) {
    const field = PVLC_GET_OWN_PROPERTY_DESCRIPTOR(descriptor, name);
    return field === undefined ? undefined : field.value;
}

function pvlcDescriptorMatches(expected, observed) {
    if (expected === undefined || observed === undefined) {
        return expected === observed;
    }
    return (
        pvlcOwnDescriptorField(expected, "value") ===
            pvlcOwnDescriptorField(observed, "value") &&
        pvlcOwnDescriptorField(expected, "get") ===
            pvlcOwnDescriptorField(observed, "get") &&
        pvlcOwnDescriptorField(expected, "set") ===
            pvlcOwnDescriptorField(observed, "set") &&
        pvlcOwnDescriptorField(expected, "configurable") ===
            pvlcOwnDescriptorField(observed, "configurable") &&
        pvlcOwnDescriptorField(expected, "enumerable") ===
            pvlcOwnDescriptorField(observed, "enumerable") &&
        pvlcOwnDescriptorField(expected, "writable") ===
            pvlcOwnDescriptorField(observed, "writable")
    );
}

function pvlcAssertUint8ArrayBridgeIntrinsics() {
    if (!pvlcDescriptorMatches(
            PVLC_UINT8_ARRAY_GLOBAL_DESCRIPTOR,
            PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_GLOBAL, "Uint8Array"),
        ) ||
        PVLC_UINT8_ARRAY.prototype !== PVLC_UINT8_ARRAY_PROTOTYPE ||
        PVLC_GET_PROTOTYPE_OF(PVLC_UINT8_ARRAY) !==
            PVLC_TYPED_ARRAY_CONSTRUCTOR ||
        PVLC_GET_PROTOTYPE_OF(PVLC_UINT8_ARRAY_PROTOTYPE) !==
            PVLC_TYPED_ARRAY_PROTOTYPE ||
        !pvlcDescriptorMatches(
            PVLC_UINT8_ARRAY_CONSTRUCTOR_DESCRIPTOR,
            PVLC_GET_OWN_PROPERTY_DESCRIPTOR(
                PVLC_UINT8_ARRAY_PROTOTYPE,
                "constructor",
            ),
        ) ||
        !pvlcDescriptorMatches(
            PVLC_UINT8_ARRAY_SPECIES_DESCRIPTOR,
            PVLC_GET_OWN_PROPERTY_DESCRIPTOR(
                PVLC_UINT8_ARRAY,
                PVLC_SYMBOL_SPECIES,
            ),
        ) ||
        !pvlcDescriptorMatches(
            PVLC_TYPED_ARRAY_SPECIES_DESCRIPTOR,
            PVLC_GET_OWN_PROPERTY_DESCRIPTOR(
                PVLC_TYPED_ARRAY_CONSTRUCTOR,
                PVLC_SYMBOL_SPECIES,
            ),
        )) {
        throw new PVLC_ERROR("pvlc Uint8Array bridge intrinsic boundary drifted");
    }
    for (let index = 0; index < PVLC_BRIDGE_PROPERTY_NAMES.length; index += 1) {
        const name = PVLC_BRIDGE_PROPERTY_NAMES[index];
        const observedTypedArray =
            PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_TYPED_ARRAY_PROTOTYPE, name);
        const observedUint8Array =
            PVLC_GET_OWN_PROPERTY_DESCRIPTOR(PVLC_UINT8_ARRAY_PROTOTYPE, name);
        if (!pvlcDescriptorMatches(
            PVLC_TYPED_ARRAY_PROPERTY_DESCRIPTORS[index],
            observedTypedArray,
        ) || !pvlcDescriptorMatches(
            PVLC_UINT8_ARRAY_PROPERTY_DESCRIPTORS[index],
            observedUint8Array,
        )) {
            throw new PVLC_ERROR(
                `pvlc Uint8Array bridge intrinsic boundary drifted: ${name}`,
            );
        }
    }
}

export function own_pvlc_uint8array_bridge_input(value) {
    pvlcAssertUint8ArrayBridgeIntrinsics();
    try {
        if (PVLC_GET_PROTOTYPE_OF(value) !== PVLC_UINT8_ARRAY_PROTOTYPE ||
            typeof PVLC_TYPED_ARRAY_BYTE_LENGTH_GETTER !== "function" ||
            typeof PVLC_TYPED_ARRAY_SET !== "function") {
            throw new PVLC_ERROR("pvlc Uint8Array bridge input is invalid");
        }
        const byteLength = PVLC_REFLECT_APPLY(
            PVLC_TYPED_ARRAY_BYTE_LENGTH_GETTER,
            value,
            [],
        );
        const owned = new PVLC_UINT8_ARRAY(byteLength);
        PVLC_REFLECT_APPLY(PVLC_TYPED_ARRAY_SET, owned, [value, 0]);
        return owned;
    } catch {
        throw new PVLC_ERROR("pvlc Uint8Array bridge input is invalid");
    }
}
"#)]
extern "C" {
    #[wasm_bindgen(catch)]
    fn own_pvlc_uint8array_bridge_input(
        value: &js_sys::Uint8Array,
    ) -> Result<js_sys::Uint8Array, JsValue>;
}

#[wasm_bindgen]
pub fn validate_browser_benchmark_cohort_plan_v1(
    canonical_plan: &js_sys::Uint8Array,
) -> Result<(), JsValue> {
    let canonical_plan = own_pvlc_uint8array_bridge_input(canonical_plan)?.to_vec();
    pvlc_bench::validate_browser_benchmark_cohort_plan_v1(&canonical_plan)
        .map_err(benchmark_error_js)
}

#[wasm_bindgen]
pub fn assemble_browser_benchmark_cohort_v1(
    canonical_input: &js_sys::Uint8Array,
) -> Result<js_sys::Uint8Array, JsValue> {
    let canonical_input = own_pvlc_uint8array_bridge_input(canonical_input)?.to_vec();
    pvlc_bench::assemble_browser_benchmark_cohort_v1(&canonical_input)
        .map(|canonical_assembly| js_sys::Uint8Array::from(canonical_assembly.as_slice()))
        .map_err(benchmark_error_js)
}

#[wasm_bindgen]
pub fn canonical_vision_encoder_stack_shader_sources_json(
    activation_strategy: &str,
) -> Result<String, JsValue> {
    let activation_strategy =
        parse_vision_stack_activation_strategy(activation_strategy).map_err(js_error)?;
    match activation_strategy {
        VisionStackActivationStrategy::StaticArenaNoAlias
        | VisionStackActivationStrategy::StaticArenaAlias => to_json(
            &vision_stack_shader_sources(
                activation_strategy,
                KernelId::VisionPatchProjectionF32,
                DecoderWeightStorage::F32,
                LinearWeightLayout::OutputMajor,
                DecoderWeightStorage::F32,
                KernelId::VisionRope2dF32,
            )
            .map_err(js_error)?,
        ),
        VisionStackActivationStrategy::SeparateBuffers => Err(js_error(
            "canonical vision-stack source export only supports reviewed static activation strategies",
        )),
    }
}

#[derive(Clone, Copy)]
enum ScopeKind {
    Internal,
    OutOfMemory,
    Validation,
}

impl ScopeKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Internal => "internal",
            Self::OutOfMemory => "out_of_memory",
            Self::Validation => "validation",
        }
    }

    const fn filter_str(self) -> &'static str {
        match self {
            Self::Internal => "internal",
            Self::OutOfMemory => "out-of-memory",
            Self::Validation => "validation",
        }
    }
}

type BrowserVisionStackScopeCapture = Option<(&'static str, String)>;

async fn pop_browser_vision_stack_error_scope(
    raw_device: &JsValue,
    pop: &js_sys::Function,
    scope: ScopeKind,
) -> VisionStackErrorScopePopAttempt<BrowserVisionStackScopeCapture, BrowserVisionStackError> {
    let invocation = pop
        .call0(raw_device)
        .map_err(|error| {
            BrowserVisionStackError(format!(
                "cannot invoke popErrorScope for {} scope: {error:?}",
                scope.as_str()
            ))
        })
        .and_then(|pending| {
            pending.dyn_into::<js_sys::Promise>().map_err(|_| {
                BrowserVisionStackError(format!(
                    "popErrorScope for {} scope did not return a Promise",
                    scope.as_str()
                ))
            })
        });
    observe_vision_stack_error_scope_pop(
        invocation,
        |promise| async move {
            JsFuture::from(promise).await.map_err(|error| {
                BrowserVisionStackError(format!(
                    "popErrorScope Promise for {} scope rejected before pop was confirmed: {error:?}",
                    scope.as_str()
                ))
            })
        },
        |error| {
            let captured = if error.is_null() || error.is_undefined() {
                None
            } else {
                let message = js_sys::Reflect::get(&error, &JsValue::from_str("message"))
                    .map_err(|failure| {
                        BrowserVisionStackError(format!(
                            "cannot normalize confirmed {} WebGPU error-scope pop: {failure:?}",
                            scope.as_str()
                        ))
                    })?
                    .as_string()
                    .unwrap_or_else(|| format!("{error:?}"));
                Some((scope.as_str(), message))
            };
            Ok(captured)
        },
    )
    .await
}

fn browser_vision_stack_scope_push_failure(
    failed_scope: ScopeKind,
    failure: VisionStackErrorScopePushFailure<
        BrowserVisionStackScopeCapture,
        BrowserVisionStackError,
    >,
) -> BrowserVisionStackError {
    let (push_error, cleanup) = failure.into_parts();
    let (captures, cleanup_failures, remaining) = cleanup.into_parts();
    let mut message = push_error.0;
    for (scope, captured) in captures.into_iter().flatten() {
        message.push_str("; cleanup captured ");
        message.push_str(scope);
        message.push_str(": ");
        message.push_str(&captured);
    }
    for cleanup_failure in cleanup_failures {
        message.push_str("; cleanup failed: ");
        message.push_str(&cleanup_failure.0);
    }
    if remaining > 0 {
        message.push_str("; persistent scope authority poisoned with ");
        message.push_str(&remaining.to_string());
        message.push_str(" unconfirmed scope(s)");
    }
    message.push_str("; failed scope: ");
    message.push_str(failed_scope.as_str());
    BrowserVisionStackError(message)
}

#[derive(Clone, Serialize)]
struct BrowserLimits {
    max_storage_buffer_binding_size: u64,
    max_compute_workgroups_per_dimension: u32,
    max_compute_invocations_per_workgroup: u32,
    max_compute_workgroup_size_x: u32,
    max_compute_workgroup_size_y: u32,
    max_compute_workgroup_size_z: u32,
    max_compute_workgroup_storage_size: u32,
    max_storage_buffers_per_shader_stage: u32,
    max_buffer_size: u64,
    min_uniform_buffer_offset_alignment: u32,
    min_storage_buffer_offset_alignment: u32,
}

impl From<&wgpu::Limits> for BrowserLimits {
    fn from(limits: &wgpu::Limits) -> Self {
        Self {
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: limits.max_compute_workgroup_size_x,
            max_compute_workgroup_size_y: limits.max_compute_workgroup_size_y,
            max_compute_workgroup_size_z: limits.max_compute_workgroup_size_z,
            max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
            max_buffer_size: limits.max_buffer_size,
            min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment,
            min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
        }
    }
}

#[derive(Clone, Serialize)]
struct BrowserCapabilities {
    adapter_name: String,
    adapter_vendor: u32,
    adapter_device: u32,
    adapter_device_type: String,
    backend: &'static str,
    shader_f16: bool,
    timestamp_query: bool,
    limits: BrowserLimits,
}

#[derive(Serialize)]
struct BrowserDiagnostics {
    kernel: KernelId,
    checked_error_scopes: [&'static str; 3],
    captured_errors: Vec<String>,
    queue_wall_time_ns: u64,
    shader_blake3: String,
}

#[derive(Serialize)]
struct BrowserExecution {
    values: Vec<f32>,
    diagnostics: BrowserDiagnostics,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BrowserVisionPatchProjectionBytesDescriptor {
    schema_version: u32,
    patch_count: u32,
    input_width: u32,
    output_width: u32,
    weight_storage: DecoderWeightStorage,
}

#[derive(Serialize)]
struct BrowserVisionLayerDiagnostics {
    checked_error_scopes: [&'static str; 3],
    captured_errors: Vec<String>,
    queue_wall_time_ns: u64,
    shader_blake3: BTreeMap<KernelId, String>,
    dispatch_stages: [VisionEncoderLayerStage; 12],
    rope_specialization: VisionRopeSpecialization,
    submission_count: u64,
    command_buffer_count: u32,
    compute_pass_count: u32,
    dispatch_count: u32,
    buffer_allocation_count: u64,
    readback_buffer_count: u32,
    readback_bytes: u64,
}

struct BrowserVisionLayerExecution {
    checkpoint_values: Vec<f32>,
    checkpoint_spans: Vec<BrowserVisionLayerCheckpoint>,
    diagnostics: BrowserVisionLayerDiagnostics,
}

#[derive(Serialize)]
struct BrowserProjectorDiagnostics {
    checked_error_scopes: [&'static str; 3],
    captured_errors: Vec<String>,
    queue_wall_time_ns: u64,
    shader_blake3: BTreeMap<KernelId, String>,
    dispatch_stages: [ProjectorStage; 5],
    submission_count: u64,
    command_buffer_count: u32,
    compute_pass_count: u32,
    dispatch_count: u32,
    buffer_allocation_count: u64,
    readback_buffer_count: u32,
    readback_map_count: u32,
    readback_bytes: u64,
    resident_intermediate_bytes: u64,
    resident_weight_bytes: u64,
}

struct BrowserProjectorExecution {
    checkpoint_values: Vec<f32>,
    checkpoint_spans: Vec<BrowserProjectorCheckpoint>,
    diagnostics: BrowserProjectorDiagnostics,
}

type SharedVisionStackResidentWeightCache =
    Rc<RefCell<VisionStackResidentWeightCache<String, wgpu::Buffer>>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BrowserVisionStackWeightResidency {
    Cold,
    Ready,
}

#[derive(Clone)]
struct BrowserVisionStackResidentWeights {
    key: String,
    disposition: BrowserVisionStackWeightResidency,
    cache: SharedVisionStackResidentWeightCache,
}

#[derive(Clone, Debug, Deserialize)]
struct BrowserProjectorF16Descriptor {
    schema_version: u32,
    weights_blake3: String,
    weights_bytes: u64,
    weight_storage: String,
    matrix_weight_layout: String,
    activation_storage: String,
    hidden_size: u32,
    output_size: u32,
    layer_norm_epsilon: f32,
}

impl BrowserProjectorF16Descriptor {
    fn cache_key(&self) -> String {
        format!(
            "projector-f16:{}:{}:{}:{}:{:08x}",
            self.weights_blake3,
            self.weights_bytes,
            self.hidden_size,
            self.output_size,
            self.layer_norm_epsilon.to_bits(),
        )
    }
}

#[derive(Clone)]
struct BrowserProjectorF16ResidentWeights {
    key: String,
    buffers: Vec<wgpu::Buffer>,
}

type SharedProjectorF16ResidentWeightCache =
    Rc<RefCell<Option<BrowserProjectorF16ResidentWeights>>>;

#[derive(Serialize)]
struct BrowserProjectorF16Diagnostics {
    queue_wall_time_ns: u64,
    output_tokens: u32,
    output_bytes: u64,
    submission_count: u64,
    dispatch_count: u32,
    resident_weight_bytes: u64,
    cpu_bridge_elided: bool,
}

#[derive(Clone, Eq, PartialEq, Serialize)]
struct BrowserVisionStackStaticPlan {
    layer_count: u32,
    shard_count: usize,
    input_bytes: u64,
    hidden_bytes: u64,
    intermediate_bytes: u64,
    layer_weight_bytes: u64,
    post_norm_bytes: u64,
    transport_bytes: u64,
    activation_buffer_count: u32,
    activation_arena_bytes: u64,
    scratch_arena_bytes: u64,
    main_buffers_bytes: u64,
    activation_strategy: VisionStackActivationStrategy,
    min_storage_buffer_offset_alignment: u32,
    readback_bytes: u64,
    peak_gpu_data_bytes: u64,
    submission_count: u32,
    compute_pass_count: u32,
    dispatch_count: u32,
}

#[derive(Clone)]
struct BrowserVisionStackSession {
    protocol: VisionStackShardProtocol,
    plan: VisionStackShardPlan,
    layer_plan: VisionEncoderLayerPlan,
    weight_plan: BrowserVisionStackLayerWeightPlan,
    fp16_qkv_plan: Option<VisionQkvFusedPlan>,
    qkv_selection: VisionQkvStackSelection,
    qkv_physical_execution: Option<VisionQkvPhysicalExecutionSpec>,
    qkv_physical_commands: Option<VisionQkvWebPhysicalCommandPlan>,
    qkv_selection_evidence: Option<VisionQkvSelectionEvidencePropagation>,
    qkv_execution_evidence_plan: Option<BrowserVisionQkvExecutionEvidencePlan>,
    activation_strategy: VisionStackActivationStrategy,
    activation_layout: Option<VisionStackActivationLayout>,
    static_plan: Option<BrowserVisionStackStaticPlan>,
    memory_hardening: Option<VisionStackMemoryHardeningPlan>,
    storage_alignment: u32,
    shader_sources: BTreeMap<KernelId, String>,
    spatial_rope: Option<BrowserVisionSpatialRope>,
    resident_weights: Option<BrowserVisionStackResidentWeights>,
    before_buffer_allocations: u64,
    before_submissions: u64,
    gpu: Option<BrowserVisionStackGpuState>,
}

struct BrowserVisionStackPreparedSession {
    protocol: VisionStackShardProtocol,
    plan: VisionStackShardPlan,
    layer_plan: VisionEncoderLayerPlan,
    weight_plan: BrowserVisionStackLayerWeightPlan,
    fp16_qkv_plan: Option<VisionQkvFusedPlan>,
    activation_strategy: VisionStackActivationStrategy,
    activation_layout: Option<VisionStackActivationLayout>,
    static_plan: Option<BrowserVisionStackStaticPlan>,
    memory_hardening: Option<VisionStackMemoryHardeningPlan>,
    storage_alignment: u32,
    shader_sources: BTreeMap<KernelId, String>,
}

#[derive(Clone)]
struct BrowserVisionSpatialRope {
    cos: Vec<f32>,
    sin: Vec<f32>,
    layer_plan: VisionEncoderLayerSpatial2dPlan,
    stack_plan: VisionEncoderStackSpatial2dPlan,
}

#[derive(Clone)]
struct BrowserVisionSpatialRopeGpuState {
    cos_buffer: wgpu::Buffer,
    sin_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
}

#[derive(Clone)]
struct BrowserVisionStackGpuState {
    pipelines: BTreeMap<KernelId, wgpu::ComputePipeline>,
    shader_blake3: BTreeMap<KernelId, String>,
    main_buffers: [wgpu::Buffer; 2],
    scratch: BrowserVisionStackScratch,
    boundary_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    qkv_physical_storage: BrowserVisionQkvPhysicalStorage,
    fp16_qkv_workspace: Option<wgpu::Buffer>,
    spatial_rope: Option<BrowserVisionSpatialRopeGpuState>,
    uniform_stride: u64,
    current_main: usize,
    started_ms: f64,
}

#[derive(Clone)]
enum BrowserVisionStackScratch {
    Separate(Vec<wgpu::Buffer>),
    Static {
        arena: wgpu::Buffer,
        ranges: Vec<VisionStackTensorRange>,
    },
}

#[derive(Clone, Copy)]
struct VisionStackTensorRange {
    offset: u64,
    bytes: u64,
}

#[derive(Clone, Copy)]
struct VisionStackBufferBinding<'a> {
    buffer: &'a wgpu::Buffer,
    offset: u64,
    size: Option<NonZeroU64>,
    bytes: u64,
}

#[derive(Clone)]
struct BrowserVisionQkvPhysicalStorage {
    buffers: BTreeMap<VisionQkvWebPhysicalBuffer, wgpu::Buffer>,
}

struct BrowserVisionQkvLayerBindGroups {
    bind_groups: BTreeMap<(u32, VisionQkvWebBindGroupKind), wgpu::BindGroup>,
}

struct BrowserVisionQkvCreatedBuffer {
    logical_buffer: VisionQkvWebPhysicalBuffer,
    gpu_buffer: wgpu::Buffer,
}

struct BrowserVisionQkvCreatedBindGroup {
    layer_index: u32,
    kind: VisionQkvWebBindGroupKind,
    bind_group: wgpu::BindGroup,
}

struct BrowserVisionQkvLayerResolutionContext<'a> {
    device: &'a wgpu::Device,
    pipelines: &'a BTreeMap<KernelId, wgpu::ComputePipeline>,
    buffers: BTreeMap<VisionQkvWebPhysicalBuffer, wgpu::Buffer>,
    encoder: &'a RefCell<wgpu::CommandEncoder>,
    norm1_output: VisionStackBufferBinding<'a>,
    query_weight: VisionStackBufferBinding<'a>,
    query_bias: VisionStackBufferBinding<'a>,
    key_weight: VisionStackBufferBinding<'a>,
    key_bias: VisionStackBufferBinding<'a>,
    value_weight: VisionStackBufferBinding<'a>,
    value_bias: VisionStackBufferBinding<'a>,
    cu_seqlens: VisionStackBufferBinding<'a>,
    attention_output: VisionStackBufferBinding<'a>,
    uniform_buffer: &'a wgpu::Buffer,
    uniform_stride: u64,
    mapped_range: &'a RefCell<Option<std::ops::Range<u64>>>,
}

struct BrowserVisionQkvAllocationAuthority<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    buffer_allocations: &'a Cell<u64>,
}

struct BrowserVisionQkvPhysicalCommandEffectSink<'a, 'b> {
    allocation: BrowserVisionQkvAllocationAuthority<'a>,
    context: &'a BrowserVisionQkvLayerResolutionContext<'a>,
    storage: &'b mut BrowserVisionQkvPhysicalStorage,
    bind_groups: &'b mut BrowserVisionQkvLayerBindGroups,
}

impl BrowserVisionStackStaticPlan {
    fn new(
        plan: &VisionStackShardPlan,
        layout: &VisionStackActivationLayout,
        activation_strategy: VisionStackActivationStrategy,
        min_storage_buffer_offset_alignment: u32,
    ) -> Result<Self, String> {
        let activation_buffer_count = u32::try_from(layout.physical_buffer_count)
            .map_err(|_| "vision-stack activation buffer count overflowed".to_owned())?;
        let peak_resident_shard = plan
            .hidden_bytes
            .max(plan.layer_weight_bytes)
            .max(plan.post_norm_bytes);
        let peak_gpu_data_bytes = layout
            .total_activation_bytes
            .checked_add(plan.readback_bytes)
            .and_then(|bytes| bytes.checked_add(peak_resident_shard))
            .ok_or_else(|| "vision-stack static peak GPU bytes overflowed".to_owned())?;
        Ok(Self {
            layer_count: plan.layer_count,
            shard_count: plan.shard_count,
            input_bytes: plan.input_bytes,
            hidden_bytes: plan.hidden_bytes,
            intermediate_bytes: plan.intermediate_bytes,
            layer_weight_bytes: plan.layer_weight_bytes,
            post_norm_bytes: plan.post_norm_bytes,
            transport_bytes: plan.transport_bytes,
            activation_buffer_count,
            activation_arena_bytes: layout.total_activation_bytes,
            scratch_arena_bytes: layout.scratch_arena_bytes,
            main_buffers_bytes: layout.main_buffers_bytes,
            activation_strategy,
            min_storage_buffer_offset_alignment,
            readback_bytes: plan.readback_bytes,
            peak_gpu_data_bytes,
            submission_count: plan.submission_count,
            compute_pass_count: plan.compute_pass_count,
            dispatch_count: plan.dispatch_count,
        })
    }
}

impl BrowserVisionStackScratch {
    fn binding(&self, index: usize) -> Result<VisionStackBufferBinding<'_>, String> {
        match self {
            Self::Separate(buffers) => buffers
                .get(index)
                .map(VisionStackBufferBinding::whole)
                .ok_or_else(|| format!("vision-stack scratch buffer {index} is missing")),
            Self::Static { arena, ranges } => {
                let range = ranges
                    .get(index)
                    .ok_or_else(|| format!("vision-stack scratch slice {index} is missing"))?;
                let size = NonZeroU64::new(range.bytes)
                    .ok_or_else(|| format!("vision-stack scratch slice {index} is empty"))?;
                Ok(VisionStackBufferBinding {
                    buffer: arena,
                    offset: range.offset,
                    size: Some(size),
                    bytes: range.bytes,
                })
            }
        }
    }
}

impl<'a> VisionStackBufferBinding<'a> {
    fn whole(buffer: &'a wgpu::Buffer) -> Self {
        Self {
            buffer,
            offset: 0,
            size: None,
            bytes: buffer.size(),
        }
    }

    fn resource(self) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.buffer,
            offset: self.offset,
            size: self.size,
        })
    }
}

fn vision_stack_fp16_qkv_workspace_bindings<'a>(
    workspace: &'a wgpu::Buffer,
    plan: &VisionQkvFusedPlan,
) -> Result<[VisionStackBufferBinding<'a>; 3], String> {
    if workspace.size() != plan.output_layout.physical_bytes {
        return Err(format!(
            "tiled FP16 Q/K/V workspace has {} bytes, expected {}",
            workspace.size(),
            plan.output_layout.physical_bytes,
        ));
    }
    let bind = |offset: u64, bytes: u64| -> Result<VisionStackBufferBinding<'a>, String> {
        Ok(VisionStackBufferBinding {
            buffer: workspace,
            offset,
            size: Some(
                NonZeroU64::new(bytes)
                    .ok_or_else(|| "tiled FP16 Q/K/V plane is empty".to_owned())?,
            ),
            bytes,
        })
    };
    Ok([
        bind(
            plan.output_layout.query.offset,
            plan.output_layout.query.size,
        )?,
        bind(plan.output_layout.key.offset, plan.output_layout.key.size)?,
        bind(
            plan.output_layout.value.offset,
            plan.output_layout.value.size,
        )?,
    ])
}

impl BrowserVisionQkvPhysicalStorage {
    fn new() -> Self {
        Self {
            buffers: BTreeMap::new(),
        }
    }

    fn store_vision_qkv_web_created_buffer(&mut self, created: BrowserVisionQkvCreatedBuffer) {
        self.buffers
            .insert(created.logical_buffer, created.gpu_buffer);
    }
}

impl BrowserVisionQkvLayerBindGroups {
    fn new() -> Self {
        Self {
            bind_groups: BTreeMap::new(),
        }
    }

    fn store_vision_qkv_web_created_bind_group(
        &mut self,
        created: BrowserVisionQkvCreatedBindGroup,
    ) {
        self.bind_groups
            .insert((created.layer_index, created.kind), created.bind_group);
    }
}

#[expect(
    clippy::needless_lifetimes,
    reason = "explicit borrow provenance is part of the sealed authority proof"
)]
fn get_vision_qkv_web_bind_group<'a>(
    groups: &'a BrowserVisionQkvLayerBindGroups,
    layer_index: u32,
    kind: VisionQkvWebBindGroupKind,
) -> &'a wgpu::BindGroup {
    groups
        .bind_groups
        .get(&(layer_index, kind))
        .expect("sealed layer bind group was created")
}

fn vision_qkv_web_attention_workspace_ranges(
    plan: &VisionQkvWebPhysicalCommandPlan,
    layer_index: u32,
) -> Result<[VisionStackTensorRange; 2], String> {
    let mut matches = plan.commands().iter().filter_map(|command| match command {
        VisionQkvWebPhysicalCommand::CreateBindGroup {
            layer_index: command_layer,
            kind: VisionQkvWebBindGroupKind::Attention,
            entries,
            ..
        } if *command_layer == layer_index => Some(entries.as_slice()),
        _ => None,
    });
    let entries = matches.next().ok_or_else(|| {
        format!("typed Q/K/V attention command is missing for layer {layer_index}")
    })?;
    if matches.next().is_some() {
        return Err(format!(
            "typed Q/K/V attention command is duplicated for layer {layer_index}"
        ));
    }
    let range = |binding: u32| {
        let entry = entries
            .iter()
            .find(|entry| entry.binding() == binding)
            .ok_or_else(|| {
                format!(
                    "typed Q/K/V attention workspace binding {binding} is missing for layer {layer_index}"
                )
            })?;
        match entry.resource() {
            VisionQkvWebBindingResource::WorkspaceRange {
                byte_offset,
                byte_length,
            } => Ok(VisionStackTensorRange {
                offset: *byte_offset,
                bytes: *byte_length,
            }),
            _ => Err(format!(
                "typed Q/K/V attention binding {binding} is not a workspace range for layer {layer_index}"
            )),
        }
    };
    Ok([range(0)?, range(1)?])
}

fn validate_vision_qkv_web_uniform_slot(
    kind: &VisionQkvWebBindGroupKind,
    uniform_slot: &u32,
    entries: &[VisionQkvWebBindGroupEntry],
) {
    let expected = match kind {
        VisionQkvWebBindGroupKind::FusedQkv => 1,
        VisionQkvWebBindGroupKind::Attention => 4,
    };
    let entry_slot = entries
        .iter()
        .find_map(|entry| match entry.resource() {
            VisionQkvWebBindingResource::Uniform {
                slot,
                byte_length: _,
            } => Some(slot),
            _ => None,
        })
        .expect("sealed bind group omitted Uniform");
    assert_eq!(uniform_slot, &expected);
    assert_eq!(uniform_slot, entry_slot);
}

impl BrowserVisionQkvLayerResolutionContext<'_> {
    fn resolve_buffer(&self, buffer: &VisionQkvWebPhysicalBuffer) -> &wgpu::Buffer {
        self.buffers
            .get(buffer)
            .expect("sealed physical buffer was stored")
    }

    fn resolve_vision_qkv_web_uniform_offset(&self, slot: u32, uniform_stride: u64) -> u64 {
        u64::from(slot)
            .checked_mul(uniform_stride)
            .expect("sealed uniform offset overflowed")
    }

    fn resolve_vision_qkv_web_context_binding<'a>(
        &self,
        binding: &VisionStackBufferBinding<'a>,
        byte_length: u64,
    ) -> wgpu::BindingResource<'a> {
        assert!(binding.bytes == byte_length);
        binding.resource()
    }

    fn resolve_vision_qkv_web_binding_resource(
        &self,
        resource: &VisionQkvWebBindingResource,
    ) -> wgpu::BindingResource<'_> {
        match resource {
            VisionQkvWebBindingResource::Norm1Output { byte_length } => {
                self.resolve_vision_qkv_web_context_binding(&self.norm1_output, *byte_length)
            }
            VisionQkvWebBindingResource::QueryWeight { byte_length } => {
                self.resolve_vision_qkv_web_context_binding(&self.query_weight, *byte_length)
            }
            VisionQkvWebBindingResource::QueryBias { byte_length } => {
                self.resolve_vision_qkv_web_context_binding(&self.query_bias, *byte_length)
            }
            VisionQkvWebBindingResource::KeyWeight { byte_length } => {
                self.resolve_vision_qkv_web_context_binding(&self.key_weight, *byte_length)
            }
            VisionQkvWebBindingResource::KeyBias { byte_length } => {
                self.resolve_vision_qkv_web_context_binding(&self.key_bias, *byte_length)
            }
            VisionQkvWebBindingResource::ValueWeight { byte_length } => {
                self.resolve_vision_qkv_web_context_binding(&self.value_weight, *byte_length)
            }
            VisionQkvWebBindingResource::ValueBias { byte_length } => {
                self.resolve_vision_qkv_web_context_binding(&self.value_bias, *byte_length)
            }
            VisionQkvWebBindingResource::WorkspaceRange {
                byte_offset,
                byte_length,
            } => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: self.resolve_buffer(&VisionQkvWebPhysicalBuffer::Workspace),
                offset: *byte_offset,
                size: wgpu::BufferSize::new(*byte_length),
            }),
            VisionQkvWebBindingResource::Uniform { slot, byte_length } => {
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: self.uniform_buffer,
                    offset: self.resolve_vision_qkv_web_uniform_offset(*slot, self.uniform_stride),
                    size: wgpu::BufferSize::new(*byte_length),
                })
            }
            VisionQkvWebBindingResource::CuSeqlens { byte_length } => {
                self.resolve_vision_qkv_web_context_binding(&self.cu_seqlens, *byte_length)
            }
            VisionQkvWebBindingResource::AttentionOutput { byte_length } => {
                self.resolve_vision_qkv_web_context_binding(&self.attention_output, *byte_length)
            }
        }
    }

    fn resolve_vision_qkv_web_bind_group_entries(
        &self,
        layer_index: &u32,
        kind: &VisionQkvWebBindGroupKind,
        uniform_slot: &u32,
        entries: &[VisionQkvWebBindGroupEntry],
    ) -> Vec<wgpu::BindGroupEntry<'_>> {
        let _ = layer_index;
        validate_vision_qkv_web_uniform_slot(kind, uniform_slot, entries);
        entries
            .iter()
            .map(|entry| {
                let resolved_resource =
                    self.resolve_vision_qkv_web_binding_resource(entry.resource());
                wgpu::BindGroupEntry {
                    binding: entry.binding(),
                    resource: resolved_resource,
                }
            })
            .collect()
    }

    fn apply_vision_qkv_web_create_bind_group_command(
        &self,
        command: &VisionQkvWebPhysicalCommand,
    ) -> BrowserVisionQkvCreatedBindGroup {
        let VisionQkvWebPhysicalCommand::CreateBindGroup {
            layer_index,
            kind,
            label,
            uniform_slot,
            entries,
        } = command
        else {
            unreachable!("typed bind-group adapter received another command variant")
        };
        let gpu_entries = self.resolve_vision_qkv_web_bind_group_entries(
            layer_index,
            kind,
            uniform_slot,
            entries,
        );
        let pipeline = match kind {
            VisionQkvWebBindGroupKind::FusedQkv => &self.pipelines[&KernelId::VisionQkvFusedF32],
            VisionQkvWebBindGroupKind::Attention => &self.pipelines[&KernelId::VisionAttentionF32],
        };
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &layout,
            entries: &gpu_entries,
        });
        BrowserVisionQkvCreatedBindGroup {
            layer_index: *layer_index,
            kind: *kind,
            bind_group,
        }
    }

    fn apply_vision_qkv_web_copy_buffer_command(&self, command: &VisionQkvWebPhysicalCommand) {
        let VisionQkvWebPhysicalCommand::CopyBuffer {
            source,
            source_offset,
            destination,
            destination_offset,
            byte_length,
            ..
        } = command
        else {
            unreachable!("typed copy adapter received another command variant")
        };
        let source_buffer = self.resolve_buffer(source);
        let destination_buffer = self.resolve_buffer(destination);
        self.encoder.borrow_mut().copy_buffer_to_buffer(
            source_buffer,
            *source_offset,
            destination_buffer,
            *destination_offset,
            *byte_length,
        );
    }

    fn apply_vision_qkv_web_map_range_command(&self, command: &VisionQkvWebPhysicalCommand) {
        let VisionQkvWebPhysicalCommand::MapRange {
            buffer, byte_range, ..
        } = command
        else {
            unreachable!("typed map adapter received another command variant")
        };
        let mapped_buffer = self.resolve_buffer(buffer);
        self.mapped_range.replace(Some(byte_range.clone()));
        let _mapped_access = || mapped_buffer.slice(byte_range.clone()).get_mapped_range();
    }
}

impl BrowserVisionQkvAllocationAuthority<'_> {
    fn create_buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        self.buffer_allocations
            .set(self.buffer_allocations.get().saturating_add(1));
        buffer
    }

    fn apply_vision_qkv_web_create_buffer_command(
        &self,
        command: &VisionQkvWebPhysicalCommand,
    ) -> BrowserVisionQkvCreatedBuffer {
        let VisionQkvWebPhysicalCommand::CreateBuffer {
            buffer,
            label,
            byte_length,
        } = command
        else {
            unreachable!("typed buffer adapter received another command variant")
        };
        let usage = match buffer {
            VisionQkvWebPhysicalBuffer::Workspace => {
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST
            }
            VisionQkvWebPhysicalBuffer::Readback => {
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
            }
        };
        let gpu_buffer = self.create_buffer(label, *byte_length, usage);
        match buffer {
            VisionQkvWebPhysicalBuffer::Workspace => {
                self.initialize_vision_qkv_web_workspace_buffer(&gpu_buffer, *byte_length)
            }
            VisionQkvWebPhysicalBuffer::Readback => {}
        }
        BrowserVisionQkvCreatedBuffer {
            logical_buffer: *buffer,
            gpu_buffer,
        }
    }

    fn initialize_vision_qkv_web_workspace_buffer(&self, buffer: &wgpu::Buffer, byte_length: u64) {
        let canary_words = vec![VISION_QKV_CANARY_U32; 16 * 1024];
        let canary_bytes = bytemuck::cast_slice(&canary_words);
        let mut offset = 0_u64;
        while offset < byte_length {
            let write_length = (byte_length - offset).min(canary_bytes.len() as u64);
            self.queue
                .write_buffer(buffer, offset, &canary_bytes[..write_length as usize]);
            offset += write_length;
        }
    }
}

impl VisionQkvWebPhysicalCommandEffectSink for BrowserVisionQkvPhysicalCommandEffectSink<'_, '_> {
    type CreatedBuffer = BrowserVisionQkvCreatedBuffer;
    type CreatedBindGroup = BrowserVisionQkvCreatedBindGroup;
    type Error = String;

    fn apply_create_buffer(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<Self::CreatedBuffer, Self::Error> {
        let _ = command_index;
        let created = self
            .allocation
            .apply_vision_qkv_web_create_buffer_command(command);
        Ok(created)
    }

    fn store_created_buffer(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
        created: Self::CreatedBuffer,
    ) -> Result<(), Self::Error> {
        let _ = (command_index, command);
        self.storage.store_vision_qkv_web_created_buffer(created);
        Ok(())
    }

    fn apply_create_bind_group(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<Self::CreatedBindGroup, Self::Error> {
        let _ = command_index;
        let created = self
            .context
            .apply_vision_qkv_web_create_bind_group_command(command);
        Ok(created)
    }

    fn store_created_bind_group(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
        created: Self::CreatedBindGroup,
    ) -> Result<(), Self::Error> {
        let _ = (command_index, command);
        self.bind_groups
            .store_vision_qkv_web_created_bind_group(created);
        Ok(())
    }

    fn apply_copy_buffer(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<(), Self::Error> {
        let _ = command_index;
        self.context
            .apply_vision_qkv_web_copy_buffer_command(command);
        Ok(())
    }

    fn apply_map_range(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<(), Self::Error> {
        let _ = command_index;
        self.context.apply_vision_qkv_web_map_range_command(command);
        Ok(())
    }
}

impl WebRuntime {
    fn apply_vision_qkv_web_start_commands(
        &self,
        context: &BrowserVisionQkvLayerResolutionContext<'_>,
        storage: &mut BrowserVisionQkvPhysicalStorage,
        bind_groups: &mut BrowserVisionQkvLayerBindGroups,
        plan: &VisionQkvWebPhysicalCommandPlan,
    ) -> Result<(), VisionQkvWebPhysicalCommandExecutionError<String>> {
        let mut sink = BrowserVisionQkvPhysicalCommandEffectSink {
            allocation: BrowserVisionQkvAllocationAuthority {
                device: &self.device,
                queue: &self.queue,
                buffer_allocations: &self.buffer_allocations,
            },
            context,
            storage,
            bind_groups,
        };
        execute_vision_qkv_web_physical_commands(
            plan,
            VisionQkvWebPhysicalCommandPhase::Start,
            &mut sink,
        )
    }
}

fn apply_vision_qkv_web_layer_commands(
    context: &BrowserVisionQkvLayerResolutionContext<'_>,
    allocation: BrowserVisionQkvAllocationAuthority<'_>,
    storage: &mut BrowserVisionQkvPhysicalStorage,
    bind_groups: &mut BrowserVisionQkvLayerBindGroups,
    plan: &VisionQkvWebPhysicalCommandPlan,
    layer_index: u32,
) -> Result<(), VisionQkvWebPhysicalCommandExecutionError<String>> {
    let mut sink = BrowserVisionQkvPhysicalCommandEffectSink {
        allocation,
        context,
        storage,
        bind_groups,
    };
    execute_vision_qkv_web_physical_commands(
        plan,
        VisionQkvWebPhysicalCommandPhase::Layer { layer_index },
        &mut sink,
    )
}

fn apply_vision_qkv_web_finish_commands(
    allocation: BrowserVisionQkvAllocationAuthority<'_>,
    context: &BrowserVisionQkvLayerResolutionContext<'_>,
    storage: &mut BrowserVisionQkvPhysicalStorage,
    bind_groups: &mut BrowserVisionQkvLayerBindGroups,
    plan: &VisionQkvWebPhysicalCommandPlan,
) -> Result<(), VisionQkvWebPhysicalCommandExecutionError<String>> {
    let mut sink = BrowserVisionQkvPhysicalCommandEffectSink {
        allocation,
        context,
        storage,
        bind_groups,
    };
    execute_vision_qkv_web_physical_commands(
        plan,
        VisionQkvWebPhysicalCommandPhase::Finish,
        &mut sink,
    )
}

#[derive(Clone, Copy)]
struct BrowserVisionLayerCheckpoint {
    stage: VisionEncoderLayerStage,
    element_offset: usize,
    elements: usize,
}

#[derive(Clone, Copy)]
struct BrowserProjectorCheckpoint {
    stage: ProjectorStage,
    element_offset: usize,
    elements: usize,
}

#[derive(Serialize)]
struct BrowserVisionLayerJsonExecution<'a> {
    checkpoints: BTreeMap<VisionEncoderLayerStage, &'a [f32]>,
    diagnostics: &'a BrowserVisionLayerDiagnostics,
}

#[derive(Serialize)]
struct BrowserProjectorJsonExecution<'a> {
    checkpoints: BTreeMap<ProjectorStage, &'a [f32]>,
    diagnostics: &'a BrowserProjectorDiagnostics,
}

impl BrowserVisionLayerExecution {
    fn json_view(&self) -> BrowserVisionLayerJsonExecution<'_> {
        let checkpoints = self
            .checkpoint_spans
            .iter()
            .map(|checkpoint| {
                let end = checkpoint.element_offset + checkpoint.elements;
                (
                    checkpoint.stage,
                    &self.checkpoint_values[checkpoint.element_offset..end],
                )
            })
            .collect();
        BrowserVisionLayerJsonExecution {
            checkpoints,
            diagnostics: &self.diagnostics,
        }
    }

    fn checkpoint_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.checkpoint_values)
    }
}

impl BrowserProjectorExecution {
    fn json_view(&self) -> BrowserProjectorJsonExecution<'_> {
        let checkpoints = self
            .checkpoint_spans
            .iter()
            .map(|checkpoint| {
                let end = checkpoint.element_offset + checkpoint.elements;
                (
                    checkpoint.stage,
                    &self.checkpoint_values[checkpoint.element_offset..end],
                )
            })
            .collect();
        BrowserProjectorJsonExecution {
            checkpoints,
            diagnostics: &self.diagnostics,
        }
    }

    fn checkpoint_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.checkpoint_values)
    }
}

#[derive(Serialize)]
struct VisionLayerShaderSourcesReport {
    sources: BTreeMap<KernelId, &'static str>,
    shader_blake3: BTreeMap<KernelId, String>,
}

#[derive(Serialize)]
struct ProjectorShaderSourcesReport {
    sources: BTreeMap<KernelId, &'static str>,
    shader_blake3: BTreeMap<KernelId, String>,
}

#[derive(Serialize)]
struct PipelineValidationReport {
    validated_kernels: Vec<KernelId>,
    checked_error_scopes: [&'static str; 3],
    captured_errors: Vec<String>,
}

#[derive(Serialize)]
struct ValidationProbeReport<'a> {
    checked_error_scopes: [&'static str; 3],
    captured_scope: &'static str,
    captured_error_count: usize,
    message: String,
    attempted_label: &'a str,
    attempted_entry_point: &'a str,
    shader_blake3: String,
}

#[wasm_bindgen]
pub struct WebVisionQkvStackSelection {
    handoff: VisionQkvCompilerHandoff,
    evidence: VisionQkvSelectionEvidencePropagation,
}

#[wasm_bindgen]
impl WebVisionQkvStackSelection {
    pub fn evidence_json(&self) -> Result<String, JsValue> {
        self.evidence.evidence_json().map_err(evidence_js_error)
    }
}

fn evidence_js_error(error: VisionStackEvidenceError) -> JsValue {
    js_error(error.to_string())
}

#[wasm_bindgen]
pub struct WebRuntime {
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    capabilities: BrowserCapabilities,
    pipelines: RefCell<BTreeMap<KernelId, wgpu::ComputePipeline>>,
    uncaptured_errors: Arc<Mutex<Vec<String>>>,
    buffer_allocations: Cell<u64>,
    submissions: Cell<u64>,
    execution_busy: Cell<bool>,
    vision_stack_error_scopes_healthy: Cell<bool>,
    vision_stack_error_scopes_occupied: Cell<bool>,
    vision_stack_session: RefCell<AsyncSessionOwner<BrowserVisionStackSession>>,
    vision_stack_streaming_weight_cache: RefCell<VisionStackStreamingWeightCache<wgpu::Buffer>>,
    vision_stack_resident_weight_cache: SharedVisionStackResidentWeightCache,
    projector_f16_resident_weight_cache: SharedProjectorF16ResidentWeightCache,
    decoder_kv_session: decoder_kv_session::DecoderKvSessionAuthority,
    decoder_layer_session: decoder_layer_session::DecoderLayerSessionAuthority,
    decoder_full_layer_session: decoder_full_layer_session::DecoderFullLayerSessionAuthority,
    decoder_stack_session: decoder_stack_session::DecoderStackSessionAuthority,
}

type BrowserDeviceCache = (
    wgpu::Device,
    wgpu::Queue,
    BrowserCapabilities,
    Arc<Mutex<Vec<String>>>,
    SharedVisionStackResidentWeightCache,
    SharedProjectorF16ResidentWeightCache,
    decoder_stack_session::SharedDecoderStackResidentWeightCache,
);

thread_local! {
    static BROWSER_DEVICE: RefCell<Option<BrowserDeviceCache>> = const { RefCell::new(None) };
}

#[wasm_bindgen]
impl WebRuntime {
    #[wasm_bindgen(js_name = create)]
    pub async fn create() -> Result<WebRuntime, JsValue> {
        let (
            device,
            queue,
            capabilities,
            uncaptured_errors,
            resident_weight_cache,
            projector_f16_resident_weight_cache,
            decoder_stack_resident_weight_cache,
        ) =
            match BROWSER_DEVICE.with(|slot| slot.borrow().clone()) {
                Some((
                    device,
                    queue,
                    capabilities,
                    uncaptured_errors,
                    resident_weight_cache,
                    projector_f16_resident_weight_cache,
                    decoder_stack_resident_weight_cache,
                )) => (
                    device,
                    queue,
                    capabilities,
                    uncaptured_errors,
                    resident_weight_cache,
                    projector_f16_resident_weight_cache,
                    decoder_stack_resident_weight_cache,
                ),
                None => {
                    let mut instance_descriptor =
                        wgpu::InstanceDescriptor::new_without_display_handle();
                    instance_descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
                    let instance = wgpu::Instance::new(instance_descriptor);
                    let adapter = instance
                        .request_adapter(&wgpu::RequestAdapterOptions {
                            power_preference: wgpu::PowerPreference::HighPerformance,
                            ..Default::default()
                        })
                        .await
                        .map_err(|error| {
                            js_error(format!("failed to acquire browser WebGPU adapter: {error}"))
                        })?;
                    let adapter_info = adapter.get_info();
                    let adapter_limits = adapter.limits();
                    let adapter_features = adapter.features();
                    let mut required_features = wgpu::Features::empty();
                    if adapter_features.contains(wgpu::Features::SHADER_F16) {
                        required_features |= wgpu::Features::SHADER_F16;
                    }
                    if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
                        required_features |= wgpu::Features::TIMESTAMP_QUERY;
                    }
                    let descriptor = wgpu::DeviceDescriptor {
                        label: Some("pvlc-browser-device"),
                        required_features,
                        required_limits: adapter_limits.clone(),
                        ..Default::default()
                    };
                    let (device, queue) =
                        adapter.request_device(&descriptor).await.map_err(|error| {
                            js_error(format!("failed to create browser WebGPU device: {error}"))
                        })?;
                    let capabilities = BrowserCapabilities {
                        adapter_name: if adapter_info.name.is_empty() {
                            "Browser WebGPU adapter".to_owned()
                        } else {
                            adapter_info.name
                        },
                        adapter_vendor: adapter_info.vendor,
                        adapter_device: adapter_info.device,
                        adapter_device_type: format!("{:?}", adapter_info.device_type)
                            .to_ascii_lowercase(),
                        backend: "browser_webgpu",
                        shader_f16: required_features.contains(wgpu::Features::SHADER_F16),
                        timestamp_query: required_features.contains(wgpu::Features::TIMESTAMP_QUERY),
                        limits: BrowserLimits::from(&adapter_limits),
                    };
                    let uncaptured_errors = Arc::new(Mutex::new(Vec::new()));
                    let uncaptured_sink = Arc::clone(&uncaptured_errors);
                    device.on_uncaptured_error(Arc::new(move |error| {
                        uncaptured_sink
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .push(error.to_string());
                    }));
                    let resident_weight_cache =
                        Rc::new(RefCell::new(VisionStackResidentWeightCache::new()));
                    let projector_f16_resident_weight_cache = Rc::new(RefCell::new(None));
                    let decoder_stack_resident_weight_cache =
                        decoder_stack_session::shared_resident_weight_cache();
                    BROWSER_DEVICE.with(|slot| {
                        let mut slot = slot.borrow_mut();
                        if slot.is_none() {
                            *slot = Some((
                                device.clone(),
                                queue.clone(),
                                capabilities.clone(),
                                Arc::clone(&uncaptured_errors),
                                Rc::clone(&resident_weight_cache),
                                Rc::clone(&projector_f16_resident_weight_cache),
                                decoder_stack_resident_weight_cache.clone(),
                            ));
                        }
                    });
                    let (
                        device,
                        queue,
                        capabilities,
                        uncaptured_errors,
                        resident_weight_cache,
                        projector_f16_resident_weight_cache,
                        decoder_stack_resident_weight_cache,
                    ) = BROWSER_DEVICE
                        .with(|slot| slot.borrow().clone())
                        .expect("browser device cache was just populated");
                    (
                        device,
                        queue,
                        capabilities,
                        uncaptured_errors,
                        resident_weight_cache,
                        projector_f16_resident_weight_cache,
                        decoder_stack_resident_weight_cache,
                    )
                }
            };
        let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(instance_descriptor);

        Ok(Self {
            _instance: instance,
            decoder_kv_session: decoder_kv_session::DecoderKvSessionAuthority::new(
                device.clone(),
                queue.clone(),
            ),
            decoder_layer_session: decoder_layer_session::DecoderLayerSessionAuthority::new(
                device.clone(),
                queue.clone(),
            ),
            decoder_full_layer_session:
                decoder_full_layer_session::DecoderFullLayerSessionAuthority::new(
                    device.clone(),
                    queue.clone(),
                ),
            decoder_stack_session: decoder_stack_session::DecoderStackSessionAuthority::new(
                device.clone(),
                queue.clone(),
                decoder_stack_resident_weight_cache,
            ),
            device,
            queue,
            capabilities,
            pipelines: RefCell::new(BTreeMap::new()),
            uncaptured_errors,
            buffer_allocations: Cell::new(0),
            submissions: Cell::new(0),
            execution_busy: Cell::new(false),
            vision_stack_error_scopes_healthy: Cell::new(true),
            vision_stack_error_scopes_occupied: Cell::new(false),
            vision_stack_session: RefCell::new(AsyncSessionOwner::new()),
            vision_stack_streaming_weight_cache: RefCell::new(
                VisionStackStreamingWeightCache::new(),
            ),
            vision_stack_resident_weight_cache: resident_weight_cache,
            projector_f16_resident_weight_cache,
        })
    }

    pub fn capabilities_json(&self) -> Result<String, JsValue> {
        to_json(&self.capabilities)
    }

    pub fn probe_m7q1_precision_admission_json(
        &self,
        precision_profile: &str,
        shader_f16_available: bool,
    ) -> Result<String, JsValue> {
        self.decoder_stack_session
            .probe_m7q1_precision_admission_json(precision_profile, shader_f16_available)
    }

    pub async fn run_m7q1_fp16_weight_probe_json(
        &self,
        fixture_json: &str,
    ) -> Result<String, JsValue> {
        self.decoder_stack_session
            .run_m7q1_fp16_weight_probe_json(fixture_json)
            .await
    }

    pub fn compile_vision_encoder_stack_qkv_selection(
        &self,
        manifest_json: &str,
        policy: &str,
    ) -> Result<WebVisionQkvStackSelection, JsValue> {
        let policy = parse_vision_qkv_execution_policy(policy).map_err(js_error)?;
        let handoff = compile_vision_qkv_stack_handoff(
            manifest_json.as_bytes(),
            policy,
            self.vision_qkv_compiler_capabilities(),
        )
        .map_err(|error| js_error(error.to_string()))?;
        let evidence = build_vision_qkv_selection_evidence_propagation(&handoff);
        Ok(WebVisionQkvStackSelection { handoff, evidence })
    }

    pub fn vision_encoder_stack_qkv_shader_sources_json(
        &self,
        activation_strategy: &str,
    ) -> Result<String, JsValue> {
        let activation_strategy =
            parse_vision_stack_activation_strategy(activation_strategy).map_err(js_error)?;
        let sources = vision_qkv_stack_shader_sources(activation_strategy).map_err(js_error)?;
        to_json(&sources)
    }

    pub async fn validate_all_pipelines_json(&self) -> Result<String, JsValue> {
        let mut validated_kernels = Vec::with_capacity(KernelId::ALL.len());
        for module in pvlc_wgsl::catalog() {
            self.validate_pipeline_source(
                module.spec.kernel.as_str(),
                module.source,
                module.spec.entry_point,
            )
            .await
            .map_err(|error| js_error(error.0))?;
            validated_kernels.push(module.spec.kernel);
        }
        to_json(&PipelineValidationReport {
            validated_kernels,
            checked_error_scopes: CHECKED_SCOPES,
            captured_errors: Vec::new(),
        })
    }

    pub async fn validate_vision_attention_pipeline_json(&self) -> Result<String, JsValue> {
        let module = pvlc_wgsl::module(KernelId::VisionAttentionF32)
            .expect("the fixed vision attention pipeline must exist");
        self.validate_pipeline_source(
            module.spec.kernel.as_str(),
            module.source,
            module.spec.entry_point,
        )
        .await
        .map_err(|error| js_error(error.0))?;
        to_json(&PipelineValidationReport {
            validated_kernels: vec![module.spec.kernel],
            checked_error_scopes: CHECKED_SCOPES,
            captured_errors: Vec::new(),
        })
    }

    pub async fn validate_vision_encoder_layer_pipelines_json(&self) -> Result<String, JsValue> {
        for kernel in VISION_LAYER_KERNELS {
            let module = pvlc_wgsl::module(kernel)
                .expect("every resident vision-layer kernel must have fixed WGSL");
            self.validate_pipeline_source(kernel.as_str(), module.source, module.spec.entry_point)
                .await
                .map_err(|error| js_error(error.0))?;
        }
        to_json(&PipelineValidationReport {
            validated_kernels: VISION_LAYER_KERNELS.to_vec(),
            checked_error_scopes: CHECKED_SCOPES,
            captured_errors: Vec::new(),
        })
    }

    pub async fn validate_projector_pipelines_json(&self) -> Result<String, JsValue> {
        for kernel in PROJECTOR_KERNELS {
            let module = pvlc_wgsl::module(kernel)
                .expect("every resident projector kernel must have fixed WGSL");
            self.validate_pipeline_source(kernel.as_str(), module.source, module.spec.entry_point)
                .await
                .map_err(|error| js_error(error.0))?;
        }
        to_json(&PipelineValidationReport {
            validated_kernels: PROJECTOR_KERNELS.to_vec(),
            checked_error_scopes: CHECKED_SCOPES,
            captured_errors: Vec::new(),
        })
    }

    pub fn vision_encoder_layer_shader_sources_json(&self) -> Result<String, JsValue> {
        let mut sources = BTreeMap::new();
        let mut shader_blake3 = BTreeMap::new();
        for kernel in VISION_LAYER_KERNELS {
            let source = pvlc_wgsl::module(kernel)
                .expect("every resident vision-layer kernel must have fixed WGSL")
                .source;
            sources.insert(kernel, source);
            shader_blake3.insert(kernel, blake3_hex(source));
        }
        to_json(&VisionLayerShaderSourcesReport {
            sources,
            shader_blake3,
        })
    }

    pub fn vision_encoder_stack_shader_sources_json(
        &self,
        activation_strategy: &str,
    ) -> Result<String, JsValue> {
        let activation_strategy =
            parse_vision_stack_activation_strategy(activation_strategy).map_err(js_error)?;
        let sources = vision_stack_shader_sources(
            activation_strategy,
            KernelId::VisionPatchProjectionF32,
            DecoderWeightStorage::F32,
            LinearWeightLayout::OutputMajor,
            DecoderWeightStorage::F32,
            KernelId::VisionRope2dF32,
        )
        .map_err(js_error)?;
        to_json(&sources)
    }

    pub fn projector_shader_sources_json(&self) -> Result<String, JsValue> {
        let mut sources = BTreeMap::new();
        let mut shader_blake3 = BTreeMap::new();
        for kernel in PROJECTOR_KERNELS {
            let source = pvlc_wgsl::module(kernel)
                .expect("every resident projector kernel must have fixed WGSL")
                .source;
            sources.insert(kernel, source);
            shader_blake3.insert(kernel, blake3_hex(source));
        }
        to_json(&ProjectorShaderSourcesReport {
            sources,
            shader_blake3,
        })
    }

    #[must_use]
    pub fn blake3_hex(&self, source: &str) -> String {
        blake3_hex(source)
    }

    #[must_use]
    pub fn blake3_bytes_hex(&self, bytes: &js_sys::Uint8Array) -> String {
        blake3_js_bytes(bytes)
    }

    pub async fn run_vision_patch_projection_bytes(
        &self,
        descriptor_json: &str,
        input: &js_sys::Uint8Array,
        weight: &js_sys::Uint8Array,
        bias: &js_sys::Uint8Array,
    ) -> Result<JsValue, JsValue> {
        let wire: BrowserVisionPatchProjectionBytesDescriptor =
            serde_json::from_str(descriptor_json).map_err(|error| {
                js_error(format!(
                    "invalid vision patch-projection bytes descriptor: {error}"
                ))
            })?;
        if wire.schema_version != 1 {
            return Err(js_error(
                "invalid vision patch-projection bytes descriptor: unsupported schema_version",
            ));
        }
        let descriptor = VisionPatchProjectionBytesDescriptor {
            patch_count: wire.patch_count,
            input_width: wire.input_width,
            output_width: wire.output_width,
            weight_storage: wire.weight_storage,
        };
        let plan = descriptor
            .plan()
            .map_err(|error| js_error(format!("invalid vision patch-projection plan: {error}")))?;
        plan.validate_capabilities(
            self.device.features().contains(wgpu::Features::SHADER_F16),
            self.capabilities
                .limits
                .max_compute_workgroups_per_dimension,
        )
        .map_err(|error| {
            js_error(format!(
                "unsupported vision patch-projection capabilities: {error}"
            ))
        })?;
        self.validate_vision_patch_projection_buffer_limits(plan)
            .map_err(js_error)?;

        let input_bytes = input.to_vec();
        let weight_bytes = weight.to_vec();
        let bias_bytes = bias.to_vec();
        plan.validate_operands(&input_bytes, &weight_bytes, &bias_bytes)
            .map_err(|error| {
                js_error(format!("invalid vision patch-projection operands: {error}"))
            })?;
        let execution = self
            .run_vision_patch_projection_bytes_source(
                descriptor,
                plan,
                &input_bytes,
                &weight_bytes,
                &bias_bytes,
            )
            .await
            .map_err(JsValue::from)?;

        let result = js_sys::Object::new();
        let checkpoint_bytes = js_sys::Uint8Array::from(execution.0.as_slice());
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("checkpoint_bytes"),
            &checkpoint_bytes,
        )?;
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("diagnostics_json"),
            &JsValue::from_str(&to_json(&execution.1)?),
        )?;
        Ok(result.into())
    }

    pub async fn run_vision_encoder_layer_identity_rope_json(
        &self,
        invocation_json: &str,
        readback: &str,
    ) -> Result<String, JsValue> {
        let invocation = parse_vision_layer_invocation(invocation_json).map_err(js_error)?;
        let readback = parse_vision_layer_readback(readback).map_err(js_error)?;
        let execution = self
            .run_vision_layer_source(&invocation, readback, None)
            .await
            .map_err(js_error)?;
        to_json(&execution.json_view())
    }

    pub async fn run_projector_json(
        &self,
        invocation_json: &str,
        readback: &str,
    ) -> Result<String, JsValue> {
        let invocation = parse_projector_invocation(invocation_json).map_err(js_error)?;
        let readback = parse_projector_readback(readback).map_err(js_error)?;
        let execution = self
            .run_projector_source(&invocation, readback, None)
            .await
            .map_err(js_error)?;
        to_json(&execution.json_view())
    }

    pub async fn run_projector_bytes(
        &self,
        descriptor_json: &str,
        profile: &str,
        input: &js_sys::Uint8Array,
        weights: &js_sys::Uint8Array,
    ) -> Result<JsValue, JsValue> {
        let descriptor =
            pvlc_pack::parse_projector_self_test_descriptor(descriptor_json.as_bytes())
                .map_err(|error| js_error(format!("invalid projector descriptor: {error}")))?;
        let mut matching_cases = descriptor
            .cases
            .iter()
            .filter(|candidate| candidate.profile == profile);
        let readback = matching_cases
            .next()
            .map(|case| case.readback)
            .ok_or_else(|| js_error(format!("projector profile {profile:?} is missing")))?;
        if matching_cases.next().is_some() {
            return Err(js_error(format!(
                "projector profile {profile:?} is duplicated"
            )));
        }
        let input_bytes = input.to_vec();
        let weight_bytes = weights.to_vec();
        let invocation = pvlc_pack::decode_projector_self_test_invocation(
            &descriptor,
            profile,
            &input_bytes,
            &weight_bytes,
        )
        .map_err(|error| js_error(format!("invalid projector payloads: {error}")))?;
        drop(input_bytes);
        drop(weight_bytes);
        drop(descriptor);
        let execution = self
            .run_projector_source(&invocation, readback, None)
            .await
            .map_err(js_error)?;

        let result = js_sys::Object::new();
        let checkpoint_bytes = js_sys::Uint8Array::from(execution.checkpoint_bytes());
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("checkpoint_bytes"),
            &checkpoint_bytes,
        )?;
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("diagnostics_json"),
            &JsValue::from_str(&to_json(&execution.diagnostics)?),
        )?;
        Ok(result.into())
    }

    pub async fn run_projector_f16_resident_bytes(
        &self,
        descriptor_json: &str,
        image_grid_thw_json: &str,
        input: &js_sys::Uint8Array,
    ) -> Result<JsValue, JsValue> {
        self.run_projector_f16_resident_input(
            descriptor_json,
            image_grid_thw_json,
            input,
        )
        .await
        .map_err(js_error)
    }

    pub async fn run_vision_encoder_layer_identity_rope_bytes(
        &self,
        descriptor_json: &str,
        weights: &js_sys::Uint8Array,
        readback: &str,
    ) -> Result<JsValue, JsValue> {
        let descriptor =
            pvlc_pack::parse_vision_layer_self_test_descriptor(descriptor_json.as_bytes())
                .map_err(|error| js_error(format!("invalid vision-layer descriptor: {error}")))?;
        let weight_bytes = weights.to_vec();
        let invocation =
            pvlc_pack::decode_vision_layer_self_test_invocation(&descriptor, &weight_bytes)
                .map_err(|error| js_error(format!("invalid vision-layer weights: {error}")))?;
        drop(weight_bytes);
        let readback = parse_vision_layer_readback(readback).map_err(js_error)?;
        let execution = self
            .run_vision_layer_source(&invocation, readback, None)
            .await
            .map_err(js_error)?;

        let result = js_sys::Object::new();
        let checkpoint_bytes = js_sys::Uint8Array::from(execution.checkpoint_bytes());
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("checkpoint_bytes"),
            &checkpoint_bytes,
        )?;
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("diagnostics_json"),
            &JsValue::from_str(&to_json(&execution.diagnostics)?),
        )?;
        Ok(result.into())
    }

    pub fn begin_vision_encoder_stack_sharded_json(
        &self,
        manifest_json: &str,
    ) -> Result<String, JsValue> {
        self.begin_vision_stack_sharded(
            manifest_json,
            VisionStackActivationStrategy::SeparateBuffers,
            None,
        )
        .map_err(js_error)
    }

    pub fn begin_vision_encoder_stack_sharded_with_activation_strategy_json(
        &self,
        manifest_json: &str,
        activation_strategy: &str,
    ) -> Result<String, JsValue> {
        let activation_strategy =
            parse_vision_stack_activation_strategy(activation_strategy).map_err(js_error)?;
        self.begin_vision_stack_sharded(manifest_json, activation_strategy, None)
            .map_err(js_error)
    }

    pub fn begin_vision_encoder_stack_sharded_with_activation_strategy_and_memory_hardening_json(
        &self,
        manifest_json: &str,
        activation_strategy: &str,
        memory_hardening: &str,
    ) -> Result<String, JsValue> {
        let activation_strategy =
            parse_vision_stack_activation_strategy(activation_strategy).map_err(js_error)?;
        let memory_hardening = memory_hardening
            .parse::<VisionStackMemoryHardening>()
            .map_err(js_error)?;
        if !matches!(
            activation_strategy,
            VisionStackActivationStrategy::StaticArenaNoAlias
                | VisionStackActivationStrategy::StaticArenaAlias
        ) {
            return Err(js_error(
                "vision-stack memory hardening requires static_arena_no_alias or static_arena_alias",
            ));
        }
        self.begin_vision_stack_sharded(manifest_json, activation_strategy, Some(memory_hardening))
            .map_err(js_error)
    }

    pub fn begin_vision_encoder_stack_sharded_with_activation_strategy_and_qkv_selection_json(
        &self,
        manifest_json: &str,
        activation_strategy: &str,
        qkv_selection: &WebVisionQkvStackSelection,
    ) -> Result<String, JsValue> {
        let activation_strategy =
            parse_vision_stack_activation_strategy(activation_strategy).map_err(js_error)?;
        self.begin_vision_stack_sharded_with_qkv_selection(
            manifest_json,
            activation_strategy,
            None,
            qkv_selection,
        )
        .map_err(js_error)
    }

    pub fn has_vision_encoder_stack_resident_weights(
        &self,
        manifest_json: &str,
    ) -> Result<bool, JsValue> {
        let manifest = parse_vision_stack_shard_manifest(manifest_json.as_bytes())
            .map_err(|error| js_error(format!("invalid vision-stack shard manifest: {error}")))?;
        require_resident_vision_stack_manifest(&manifest).map_err(js_error)?;
        let key = vision_stack_resident_weight_key(&manifest).map_err(|error| {
            js_error(format!(
                "cannot derive resident vision-weight identity: {error}"
            ))
        })?;
        let layer_count = usize::try_from(manifest.layer_count)
            .map_err(|_| js_error("vision-stack layer count does not fit usize"))?;
        Ok(self
            .vision_stack_resident_weight_cache
            .borrow()
            .is_ready_for(&key, layer_count))
    }

    pub fn has_projector_f16_resident_weights(
        &self,
        descriptor_json: &str,
    ) -> Result<bool, JsValue> {
        let descriptor =
            parse_projector_f16_descriptor(descriptor_json).map_err(js_error)?;
        let key = descriptor.cache_key();
        Ok(self
            .projector_f16_resident_weight_cache
            .borrow()
            .as_ref()
            .is_some_and(|resident| resident.key == key))
    }

    pub fn prepare_projector_f16_resident_weights(
        &self,
        descriptor_json: &str,
        weights: &js_sys::Uint8Array,
    ) -> Result<String, JsValue> {
        self.prepare_projector_f16_weights(descriptor_json, weights)
            .map_err(js_error)
    }

    pub fn begin_vision_encoder_stack_sharded_resident_with_activation_strategy_and_qkv_selection_json(
        &self,
        manifest_json: &str,
        activation_strategy: &str,
        qkv_selection: &WebVisionQkvStackSelection,
    ) -> Result<String, JsValue> {
        let activation_strategy =
            parse_vision_stack_activation_strategy(activation_strategy).map_err(js_error)?;
        self.begin_vision_stack_sharded_resident_with_qkv_selection(
            manifest_json,
            activation_strategy,
            qkv_selection,
        )
        .map_err(js_error)
    }

    pub fn begin_vision_encoder_stack_sharded_with_activation_strategy_and_memory_hardening_and_qkv_selection_json(
        &self,
        manifest_json: &str,
        activation_strategy: &str,
        memory_hardening: &str,
        qkv_selection: &WebVisionQkvStackSelection,
    ) -> Result<String, JsValue> {
        let activation_strategy =
            parse_vision_stack_activation_strategy(activation_strategy).map_err(js_error)?;
        let memory_hardening = memory_hardening
            .parse::<VisionStackMemoryHardening>()
            .map_err(js_error)?;
        if !matches!(
            activation_strategy,
            VisionStackActivationStrategy::StaticArenaNoAlias
                | VisionStackActivationStrategy::StaticArenaAlias
        ) {
            return Err(js_error(
                "vision-stack memory hardening requires static_arena_no_alias or static_arena_alias",
            ));
        }
        self.begin_vision_stack_sharded_with_qkv_selection(
            manifest_json,
            activation_strategy,
            Some(memory_hardening),
            qkv_selection,
        )
        .map_err(js_error)
    }

    pub fn configure_vision_encoder_stack_spatial_rope_f32(
        &self,
        cos: &js_sys::Float32Array,
        sin: &js_sys::Float32Array,
    ) -> Result<String, JsValue> {
        self.configure_vision_stack_spatial_rope(cos, sin)
            .map_err(js_error)
    }

    pub fn preflight_vision_encoder_stack_shard_json(
        &self,
        shard_id: &str,
        bytes: &js_sys::Uint8Array,
    ) -> Result<String, JsValue> {
        self.preflight_vision_stack_shard(shard_id, bytes)
            .map_err(js_error)
    }

    pub fn preflight_vision_encoder_stack_manifest_shard_json(
        &self,
        shard_id: &str,
    ) -> Result<String, JsValue> {
        self.preflight_vision_stack_manifest_shard(shard_id)
            .map_err(js_error)
    }

    pub async fn start_vision_encoder_stack_sharded_json(
        &self,
        shard_id: &str,
        bytes: &js_sys::Uint8Array,
    ) -> Result<String, JsValue> {
        self.start_vision_stack_sharded(shard_id, bytes)
            .await
            .map_err(js_error)
    }

    pub async fn run_vision_encoder_stack_sharded_layer_json(
        &self,
        shard_id: &str,
        bytes: &js_sys::Uint8Array,
    ) -> Result<String, JsValue> {
        self.run_vision_stack_sharded_layer(shard_id, bytes)
            .await
            .map_err(js_error)
    }

    pub fn enqueue_vision_encoder_stack_sharded_layer_json(
        &self,
        shard_id: &str,
        bytes: &js_sys::Uint8Array,
    ) -> Result<String, JsValue> {
        self.enqueue_vision_stack_sharded_layer(shard_id, bytes)
            .map_err(js_error)
    }

    pub fn enqueue_vision_encoder_stack_sharded_resident_layer_json(
        &self,
        shard_id: &str,
    ) -> Result<String, JsValue> {
        self.enqueue_vision_stack_sharded_resident_layer(shard_id)
            .map_err(js_error)
    }

    pub async fn finish_vision_encoder_stack_sharded(
        &self,
        shard_id: &str,
        bytes: &js_sys::Uint8Array,
    ) -> Result<JsValue, JsValue> {
        self.finish_vision_stack_sharded(shard_id, bytes)
            .await
            .map_err(js_error)
    }

    pub async fn finish_vision_encoder_stack_sharded_resident(
        &self,
        shard_id: &str,
    ) -> Result<JsValue, JsValue> {
        self.finish_vision_stack_sharded_resident(shard_id)
            .await
            .map_err(js_error)
    }

    pub async fn finish_vision_encoder_stack_sharded_resident_with_projector_f16(
        &self,
        shard_id: &str,
        projector_descriptor_json: &str,
        image_grid_thw_json: &str,
    ) -> Result<JsValue, JsValue> {
        self.finish_vision_stack_sharded_resident_with_projector_f16(
            shard_id,
            projector_descriptor_json,
            image_grid_thw_json,
        )
        .await
        .map_err(js_error)
    }

    pub fn abort_vision_encoder_stack_sharded(&self) {
        match self.vision_stack_session.borrow_mut().abort() {
            AbortDisposition::Released => self.execution_busy.set(false),
            AbortDisposition::Deferred | AbortDisposition::AlreadyIdle => {}
        }
    }

    pub fn begin_decoder_kv_session(
        &self,
        descriptor_json: &str,
        key_cache: &js_sys::Uint8Array,
        value_cache: &js_sys::Uint8Array,
    ) -> js_sys::Promise {
        self.decoder_kv_session
            .begin(descriptor_json, key_cache, value_cache)
    }

    pub fn begin_decoder_kv_session_with_shader_override(
        &self,
        descriptor_json: &str,
        key_cache: &js_sys::Uint8Array,
        value_cache: &js_sys::Uint8Array,
        kernel: &str,
        source: &str,
    ) -> js_sys::Promise {
        self.decoder_kv_session.begin_with_shader_override(
            descriptor_json,
            key_cache,
            value_cache,
            kernel,
            source,
        )
    }

    pub fn step_decoder_kv_session(
        &self,
        query: &js_sys::Uint8Array,
        appended_key: &js_sys::Uint8Array,
        appended_value: &js_sys::Uint8Array,
    ) -> js_sys::Promise {
        self.decoder_kv_session
            .step(query, appended_key, appended_value)
    }

    pub fn finish_decoder_kv_session(&self) -> js_sys::Promise {
        self.decoder_kv_session.finish()
    }

    pub fn abort_decoder_kv_session(&self) {
        self.decoder_kv_session.abort()
    }

    pub fn decoder_kv_session_shader_sources_json(&self) -> Result<String, JsValue> {
        self.decoder_kv_session.shader_sources_json()
    }

    pub fn begin_decoder_layer_session(
        &self,
        descriptor_json: &str,
        pack: &js_sys::Uint8Array,
        key_cache: &js_sys::Uint8Array,
        value_cache: &js_sys::Uint8Array,
    ) -> js_sys::Promise {
        self.decoder_layer_session
            .begin(descriptor_json, pack, key_cache, value_cache)
    }

    pub fn begin_decoder_layer_session_with_shader_override(
        &self,
        descriptor_json: &str,
        pack: &js_sys::Uint8Array,
        key_cache: &js_sys::Uint8Array,
        value_cache: &js_sys::Uint8Array,
        kernel: &str,
        source: &str,
    ) -> js_sys::Promise {
        self.decoder_layer_session.begin_with_shader_override(
            descriptor_json,
            pack,
            key_cache,
            value_cache,
            kernel,
            source,
        )
    }

    pub fn step_decoder_layer_session(&self, hidden_token: &js_sys::Uint8Array) -> js_sys::Promise {
        self.decoder_layer_session.step(hidden_token)
    }

    pub fn finish_decoder_layer_session(&self) -> js_sys::Promise {
        self.decoder_layer_session.finish()
    }

    pub fn abort_decoder_layer_session(&self) {
        self.decoder_layer_session.abort()
    }

    pub fn decoder_layer_session_shader_sources_json(&self) -> Result<String, JsValue> {
        self.decoder_layer_session.shader_sources_json()
    }

    pub fn begin_decoder_full_layer_session(
        &self,
        descriptor_json: &str,
        pack: &js_sys::Uint8Array,
        key_cache: &js_sys::Uint8Array,
        value_cache: &js_sys::Uint8Array,
    ) -> js_sys::Promise {
        self.decoder_full_layer_session
            .begin(descriptor_json, pack, key_cache, value_cache)
    }

    pub fn begin_decoder_full_layer_session_with_shader_override(
        &self,
        descriptor_json: &str,
        pack: &js_sys::Uint8Array,
        key_cache: &js_sys::Uint8Array,
        value_cache: &js_sys::Uint8Array,
        kernel: &str,
        source: &str,
    ) -> js_sys::Promise {
        self.decoder_full_layer_session.begin_with_shader_override(
            descriptor_json,
            pack,
            key_cache,
            value_cache,
            kernel,
            source,
        )
    }

    pub fn step_decoder_full_layer_session(
        &self,
        hidden_token: &js_sys::Uint8Array,
    ) -> js_sys::Promise {
        self.decoder_full_layer_session.step(hidden_token)
    }

    pub fn finish_decoder_full_layer_session(&self) -> js_sys::Promise {
        self.decoder_full_layer_session.finish()
    }

    pub fn abort_decoder_full_layer_session(&self) {
        self.decoder_full_layer_session.abort()
    }

    pub fn decoder_full_layer_session_shader_sources_json(&self) -> Result<String, JsValue> {
        self.decoder_full_layer_session.shader_sources_json()
    }

    pub fn begin_decoder_stack_session(
        &self,
        descriptor_json: &str,
        pack: &js_sys::Uint8Array,
        key_cache: &js_sys::Uint8Array,
        value_cache: &js_sys::Uint8Array,
    ) -> js_sys::Promise {
        self.decoder_stack_session
            .begin(descriptor_json, pack, key_cache, value_cache)
    }

    pub fn begin_decoder_stack_session_resident(
        &self,
        descriptor_json: &str,
        rope_cos: &js_sys::Uint8Array,
        rope_sin: &js_sys::Uint8Array,
    ) -> js_sys::Promise {
        self.decoder_stack_session
            .begin_resident(descriptor_json, rope_cos, rope_sin)
    }

    pub fn begin_decoder_stack_session_with_shader_override(
        &self,
        descriptor_json: &str,
        pack: &js_sys::Uint8Array,
        key_cache: &js_sys::Uint8Array,
        value_cache: &js_sys::Uint8Array,
        kernel: &str,
        source: &str,
    ) -> js_sys::Promise {
        self.decoder_stack_session.begin_with_shader_override(
            descriptor_json,
            pack,
            key_cache,
            value_cache,
            kernel,
            source,
        )
    }

    pub fn step_decoder_stack_session(&self, hidden_token: &js_sys::Uint8Array) -> js_sys::Promise {
        self.decoder_stack_session.step(hidden_token)
    }

    pub fn prefill_decoder_stack_session(
        &self,
        hidden_states: &js_sys::Uint8Array,
    ) -> js_sys::Promise {
        self.decoder_stack_session.prefill(hidden_states)
    }

    pub fn logits_decoder_stack_session(&self) -> js_sys::Promise {
        self.decoder_stack_session.logits()
    }

    pub fn top1_decoder_stack_session(&self) -> js_sys::Promise {
        self.decoder_stack_session.top1()
    }

    pub fn decoder_stack_resident_weights_json(&self) -> Result<String, JsValue> {
        self.decoder_stack_session.resident_weights_json()
    }

    pub fn finish_decoder_stack_session(&self) -> js_sys::Promise {
        self.decoder_stack_session.finish()
    }

    pub fn abort_decoder_stack_session(&self) {
        self.decoder_stack_session.abort()
    }

    pub fn decoder_stack_session_shader_sources_json(&self) -> Result<String, JsValue> {
        self.decoder_stack_session.shader_sources_json()
    }

    pub async fn run_vision_encoder_layer_identity_rope_with_shader_override_json(
        &self,
        invocation_json: &str,
        readback: &str,
        kernel: &str,
        source: &str,
    ) -> Result<String, JsValue> {
        let invocation = parse_vision_layer_invocation(invocation_json).map_err(js_error)?;
        let readback = parse_vision_layer_readback(readback).map_err(js_error)?;
        let kernel = parse_vision_layer_kernel(kernel).map_err(js_error)?;
        require_nonempty(source, "vision-layer shader override").map_err(js_error)?;
        let execution = self
            .run_vision_layer_source(&invocation, readback, Some((kernel, source)))
            .await
            .map_err(js_error)?;
        to_json(&execution.json_view())
    }

    pub async fn run_projector_with_shader_override_json(
        &self,
        invocation_json: &str,
        readback: &str,
        kernel: &str,
        source: &str,
    ) -> Result<String, JsValue> {
        let invocation = parse_projector_invocation(invocation_json).map_err(js_error)?;
        let readback = parse_projector_readback(readback).map_err(js_error)?;
        let kernel = parse_projector_kernel(kernel).map_err(js_error)?;
        require_nonempty(source, "projector shader override").map_err(js_error)?;
        let execution = self
            .run_projector_source(&invocation, readback, Some((kernel, source)))
            .await
            .map_err(js_error)?;
        to_json(&execution.json_view())
    }

    pub async fn run_json(&self, invocation_json: &str) -> Result<String, JsValue> {
        let invocation: KernelInvocation = serde_json::from_str(invocation_json)
            .map_err(|error| js_error(format!("invalid kernel invocation JSON: {error}")))?;
        let module = pvlc_wgsl::module(invocation.kernel_id()).ok_or_else(|| {
            js_error(format!(
                "WGSL catalog has no {} module",
                invocation.kernel_id()
            ))
        })?;
        let execution = self
            .run_source(
                &invocation,
                module.spec.kernel.as_str(),
                module.source,
                module.spec.entry_point,
                Some(module.spec.kernel),
            )
            .await
            .map_err(|error| js_error(error.0))?;
        to_json(&execution)
    }

    pub async fn run_with_shader_json(
        &self,
        invocation_json: &str,
        label: &str,
        source: &str,
        entry_point: &str,
    ) -> Result<String, JsValue> {
        require_nonempty(label, "custom shader label").map_err(js_error)?;
        require_nonempty(source, "custom shader source").map_err(js_error)?;
        require_nonempty(entry_point, "custom shader entry point").map_err(js_error)?;
        let invocation: KernelInvocation = serde_json::from_str(invocation_json)
            .map_err(|error| js_error(format!("invalid custom kernel invocation JSON: {error}")))?;
        let execution = self
            .run_source(&invocation, label, source, entry_point, None)
            .await
            .map_err(|error| js_error(error.0))?;
        to_json(&execution)
    }

    pub async fn probe_validation_error_json(
        &self,
        label: &str,
        source: &str,
        missing_entry_point: &str,
    ) -> Result<String, JsValue> {
        require_nonempty(label, "validation probe label").map_err(js_error)?;
        require_nonempty(source, "validation probe source").map_err(js_error)?;
        require_nonempty(missing_entry_point, "validation probe entry point").map_err(js_error)?;
        let guards = self.push_browser_error_scopes().await?;
        let completed = self
            .complete_browser_error_scoped_operation(guards, async {
                let shader = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(label),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    });
                let _pipeline =
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(label),
                            layout: None,
                            module: &shader,
                            entry_point: Some(missing_entry_point),
                            compilation_options: Default::default(),
                            cache: None,
                        });
                Ok::<(), BrowserVisionStackError>(())
            })
            .await?;
        let (operation, captured) = completed;
        operation?;
        let uncaptured = self.take_uncaptured_errors();
        if !uncaptured.is_empty() {
            return Err(js_error(format!(
                "validation probe emitted uncaptured WebGPU errors: {}",
                uncaptured.join("; ")
            )));
        }
        if captured.len() != 1 || captured[0].0 != "validation" {
            return Err(js_error(format!(
                "validation probe expected one validation error, got {captured:?}"
            )));
        }
        to_json(&ValidationProbeReport {
            checked_error_scopes: CHECKED_SCOPES,
            captured_scope: "validation",
            captured_error_count: 1,
            message: captured[0].1.clone(),
            attempted_label: label,
            attempted_entry_point: missing_entry_point,
            shader_blake3: blake3::hash(source.as_bytes()).to_hex().to_string(),
        })
    }
}

impl WebRuntime {
    fn vision_qkv_compiler_capabilities(&self) -> VisionQkvCompilerCapabilities {
        VisionQkvCompilerCapabilities {
            min_storage_buffer_offset_alignment: self
                .capabilities
                .limits
                .min_storage_buffer_offset_alignment,
            max_storage_buffers_per_shader_stage: self
                .capabilities
                .limits
                .max_storage_buffers_per_shader_stage,
            max_storage_buffer_binding_size: self
                .capabilities
                .limits
                .max_storage_buffer_binding_size,
            max_buffer_size: self.capabilities.limits.max_buffer_size,
            max_compute_workgroup_size: [
                self.capabilities.limits.max_compute_workgroup_size_x,
                self.capabilities.limits.max_compute_workgroup_size_y,
                self.capabilities.limits.max_compute_workgroup_size_z,
            ],
            max_compute_invocations_per_workgroup: self
                .capabilities
                .limits
                .max_compute_invocations_per_workgroup,
            max_compute_workgroups_per_dimension: self
                .capabilities
                .limits
                .max_compute_workgroups_per_dimension,
            max_host_elements: u64::from(u32::MAX),
        }
    }

    fn begin_vision_stack_sharded(
        &self,
        manifest_json: &str,
        activation_strategy: VisionStackActivationStrategy,
        memory_hardening: Option<VisionStackMemoryHardening>,
    ) -> Result<String, String> {
        require_vision_stack_error_scope_admission_available(
            &self.vision_stack_error_scopes_healthy,
            &self.vision_stack_error_scopes_occupied,
            "persistent WebGPU error-scope authority is poisoned".to_owned(),
            "another browser WebGPU error-scope operation is already in progress".to_owned(),
        )?;
        if self.execution_busy.get() || self.vision_stack_session.borrow().is_busy() {
            return Err("another browser WebGPU execution is already in progress".to_owned());
        }
        let prepared = self.prepare_browser_stack(
            manifest_json,
            activation_strategy,
            memory_hardening,
            VisionQkvSelectionOutcome::Disabled,
        )?;
        let disabled_handoff = compile_vision_qkv_stack_handoff(
            manifest_json.as_bytes(),
            VisionQkvExecutionPolicy::Disabled,
            self.vision_qkv_compiler_capabilities(),
        )
        .map_err(|error| error.to_string())?;
        let qkv_selection = disabled_handoff.selection().clone();
        let BrowserVisionStackPreparedSession {
            protocol,
            plan,
            layer_plan,
            weight_plan,
            fp16_qkv_plan,
            activation_strategy,
            activation_layout,
            static_plan,
            memory_hardening,
            storage_alignment,
            shader_sources,
        } = prepared;
        let session = BrowserVisionStackSession {
            protocol,
            plan,
            layer_plan,
            weight_plan,
            fp16_qkv_plan,
            qkv_selection,
            qkv_physical_execution: None,
            qkv_physical_commands: None,
            qkv_selection_evidence: None,
            qkv_execution_evidence_plan: None,
            activation_strategy,
            activation_layout,
            static_plan,
            memory_hardening,
            storage_alignment,
            shader_sources,
            spatial_rope: None,
            resident_weights: None,
            before_buffer_allocations: self.buffer_allocations.get(),
            before_submissions: self.submissions.get(),
            gpu: None,
        };
        let status = vision_stack_status_json(&session, true).map_err(|error| error.to_string())?;
        self.vision_stack_session
            .borrow_mut()
            .begin(session)
            .map_err(|error| format!("cannot begin vision-stack session: {error:?}"))?;
        self.execution_busy.set(true);
        Ok(status)
    }

    fn begin_vision_stack_sharded_with_qkv_selection(
        &self,
        manifest_json: &str,
        activation_strategy: VisionStackActivationStrategy,
        memory_hardening: Option<VisionStackMemoryHardening>,
        qkv_selection: &WebVisionQkvStackSelection,
    ) -> Result<String, String> {
        require_vision_stack_error_scope_admission_available(
            &self.vision_stack_error_scopes_healthy,
            &self.vision_stack_error_scopes_occupied,
            "persistent WebGPU error-scope authority is poisoned".to_owned(),
            "another browser WebGPU error-scope operation is already in progress".to_owned(),
        )?;
        if self.execution_busy.get() || self.vision_stack_session.borrow().is_busy() {
            return Err("another browser WebGPU execution is already in progress".to_owned());
        }
        let manifest_bytes = manifest_json.as_bytes();
        let handoff = &qkv_selection.handoff;
        let capabilities = self.vision_qkv_compiler_capabilities();
        let validated_handoff_binding =
            validate_vision_qkv_stack_handoff_binding(handoff, manifest_bytes, capabilities)
                .map_err(|error| error.to_string())?;
        let _ = validated_handoff_binding;
        let qkv_outcome = handoff.selection().outcome();
        let prepared = self.prepare_browser_stack(
            manifest_json,
            activation_strategy,
            memory_hardening,
            qkv_outcome,
        )?;
        let scratch_canary_readback_bytes = prepared
            .memory_hardening
            .as_ref()
            .map_or(0, |plan| plan.guard_bytes() * 2);
        let readback = VisionQkvCompilerReadbackRequest {
            semantic_readback_bytes: prepared.plan.readback_bytes,
            scratch_canary_readback_bytes,
        };
        let session_evidence = qkv_selection.evidence.clone();
        let qkv_selection_value = handoff.selection().clone();
        let session = if qkv_outcome == VisionQkvSelectionOutcome::Fused {
            let sealed_qkv_physical_execution = prepare_vision_qkv_stack_handoff_execution(
                handoff,
                manifest_bytes,
                capabilities,
                readback,
            )
            .map_err(|error| error.to_string())?;
            let qkv_physical_commands =
                plan_vision_qkv_web_physical_commands(&sealed_qkv_physical_execution);
            let mut qkv_physical_execution = Some(sealed_qkv_physical_execution);
            let qkv_execution_evidence_plan = BrowserVisionQkvExecutionEvidencePlan::from_prepared(
                qkv_physical_execution.as_ref(),
            )
            .map_err(|error| error.to_string())?;
            let sealed_qkv_physical_execution = qkv_physical_execution
                .take()
                .expect("fused Q/K/V physical execution was just prepared");
            let BrowserVisionStackPreparedSession {
                protocol,
                plan,
                layer_plan,
                weight_plan,
                fp16_qkv_plan,
                activation_strategy,
                activation_layout,
                static_plan,
                memory_hardening,
                storage_alignment,
                shader_sources,
            } = prepared;
            BrowserVisionStackSession {
                protocol,
                plan,
                layer_plan,
                weight_plan,
                fp16_qkv_plan,
                qkv_selection: qkv_selection_value,
                qkv_physical_execution: Some(sealed_qkv_physical_execution),
                qkv_physical_commands: Some(qkv_physical_commands),
                qkv_selection_evidence: Some(session_evidence),
                qkv_execution_evidence_plan,
                activation_strategy,
                activation_layout,
                static_plan,
                memory_hardening,
                storage_alignment,
                shader_sources,
                spatial_rope: None,
                resident_weights: None,
                before_buffer_allocations: self.buffer_allocations.get(),
                before_submissions: self.submissions.get(),
                gpu: None,
            }
        } else {
            build_legacy_qkv_browser_session(
                prepared,
                qkv_selection_value,
                session_evidence,
                self.buffer_allocations.get(),
                self.submissions.get(),
            )
        };
        let status =
            vision_stack_qkv_status_json(&session, true).map_err(|error| error.to_string())?;
        self.vision_stack_session
            .borrow_mut()
            .begin(session)
            .map_err(|error| format!("cannot begin vision-stack session: {error:?}"))?;
        self.execution_busy.set(true);
        Ok(status)
    }

    fn begin_vision_stack_sharded_resident_with_qkv_selection(
        &self,
        manifest_json: &str,
        activation_strategy: VisionStackActivationStrategy,
        qkv_selection: &WebVisionQkvStackSelection,
    ) -> Result<String, String> {
        let manifest = parse_vision_stack_shard_manifest(manifest_json.as_bytes())
            .map_err(|error| format!("invalid vision-stack shard manifest: {error}"))?;
        require_resident_vision_stack_manifest(&manifest)?;
        let key = vision_stack_resident_weight_key(&manifest)
            .map_err(|error| format!("cannot derive resident vision-weight identity: {error}"))?;
        let layer_count = usize::try_from(manifest.layer_count)
            .map_err(|_| "vision-stack layer count does not fit usize".to_owned())?;
        let disposition = self
            .vision_stack_resident_weight_cache
            .borrow_mut()
            .prepare(key.clone(), layer_count)
            .map_err(|error| format!("cannot prepare resident vision weights: {error}"))?;
        let status = self.begin_vision_stack_sharded_with_qkv_selection(
            manifest_json,
            activation_strategy,
            None,
            qkv_selection,
        )?;
        let owner = self.vision_stack_session.borrow();
        let mut session = owner.stored_mut().ok_or_else(|| {
            "resident vision-stack session was not stored after a successful begin".to_owned()
        })?;
        session.resident_weights = Some(BrowserVisionStackResidentWeights {
            key,
            disposition: match disposition {
                VisionStackResidentCacheDisposition::Cold => {
                    BrowserVisionStackWeightResidency::Cold
                }
                VisionStackResidentCacheDisposition::Ready => {
                    BrowserVisionStackWeightResidency::Ready
                }
            },
            cache: Rc::clone(&self.vision_stack_resident_weight_cache),
        });
        Ok(status)
    }

    fn prepare_projector_f16_weights(
        &self,
        descriptor_json: &str,
        weights: &js_sys::Uint8Array,
    ) -> Result<String, String> {
        if self.execution_busy.get() || self.vision_stack_session.borrow().is_busy() {
            return Err(
                "projector FP16 weights must be prepared before vision execution begins"
                    .to_owned(),
            );
        }
        let descriptor = parse_projector_f16_descriptor(descriptor_json)?;
        let key = descriptor.cache_key();
        if self
            .projector_f16_resident_weight_cache
            .borrow()
            .as_ref()
            .is_some_and(|resident| resident.key == key)
        {
            return Ok(serde_json::json!({
                "status": "ready",
                "resident": true,
                "weights_bytes": descriptor.weights_bytes,
            })
            .to_string());
        }
        if u64::from(weights.length()) != descriptor.weights_bytes {
            return Err(format!(
                "projector FP16 payload has {} bytes, expected {}",
                weights.length(),
                descriptor.weights_bytes,
            ));
        }
        let observed_blake3 = blake3_js_bytes(weights);
        if observed_blake3 != descriptor.weights_blake3 {
            return Err(format!(
                "projector FP16 payload digest {observed_blake3} does not match {}",
                descriptor.weights_blake3,
            ));
        }
        let ranges = projector_f16_weight_ranges(&descriptor)?;
        let mut buffers = Vec::with_capacity(ranges.len());
        for (index, range) in ranges.into_iter().enumerate() {
            self.validate_storage_buffer_bytes(
                &format!("projector-f16-weight-{index}"),
                range.bytes,
            )?;
            buffers.push(self.create_uploaded_js_buffer(
                &format!("projector-f16-resident-weight-{index}"),
                weights,
                range,
            )?);
        }
        self.projector_f16_resident_weight_cache.replace(Some(
            BrowserProjectorF16ResidentWeights {
                key,
                buffers,
            },
        ));
        Ok(serde_json::json!({
            "status": "uploaded",
            "resident": true,
            "weights_bytes": descriptor.weights_bytes,
        })
        .to_string())
    }

    fn configure_vision_stack_spatial_rope(
        &self,
        cos: &js_sys::Float32Array,
        sin: &js_sys::Float32Array,
    ) -> Result<String, String> {
        let owner = self.vision_stack_session.borrow();
        let mut session = owner
            .stored_mut()
            .ok_or_else(|| "vision-stack spatial RoPE requires an active session".to_owned())?;
        if session.gpu.is_some() {
            return Err(
                "vision-stack spatial RoPE must be configured before GPU allocation".to_owned(),
            );
        }
        if session.spatial_rope.is_some() {
            return Err("vision-stack spatial RoPE is already configured".to_owned());
        }

        let cos = cos.to_vec();
        let sin = sin.to_vec();
        let manifest = session.protocol.manifest();
        let rope = VisionRope2dDescriptor {
            tokens: manifest.tokens,
            heads: manifest.attention_heads,
            head_dim: manifest.head_dim,
            cos: &cos,
            sin: &sin,
        }
        .plan()
        .map_err(|error| format!("invalid vision-stack spatial RoPE: {error}"))?;
        self.validate_storage_buffer_bytes("vision RoPE cosine table", rope.table_bytes)?;
        self.validate_storage_buffer_bytes("vision RoPE sine table", rope.table_bytes)?;
        let layer_plan = session
            .layer_plan
            .with_spatial_rope(rope)
            .map_err(|error| format!("invalid vision-stack spatial RoPE plan: {error}"))?;
        let layer_count = usize::try_from(session.plan.layer_count)
            .map_err(|_| "vision-stack layer count does not fit usize".to_owned())?;
        let stack_plan = layer_plan
            .stack_plan(layer_count)
            .map_err(|error| format!("invalid vision-stack spatial RoPE stack plan: {error}"))?;

        let rope_kernel = session.weight_plan.rope_kernel;
        let module = pvlc_wgsl::module(rope_kernel)
            .expect("the spatial vision RoPE kernel must have fixed WGSL");
        pvlc_wgsl::validate_source_contract(&module.spec, module.source)
            .map_err(|error| format!("resident vision RoPE shader is invalid: {error}"))?;
        session
            .shader_sources
            .insert(rope_kernel, module.source.to_owned());
        session.spatial_rope = Some(BrowserVisionSpatialRope {
            cos,
            sin,
            layer_plan,
            stack_plan,
        });
        vision_stack_status_json(&session, false).map_err(|error| error.to_string())
    }

    fn prepare_browser_stack(
        &self,
        manifest_json: &str,
        activation_strategy: VisionStackActivationStrategy,
        memory_hardening: Option<VisionStackMemoryHardening>,
        qkv_outcome: VisionQkvSelectionOutcome,
    ) -> Result<BrowserVisionStackPreparedSession, String> {
        let manifest = parse_vision_stack_shard_manifest(manifest_json.as_bytes())
            .map_err(|error| format!("invalid vision-stack shard manifest: {error}"))?;
        let plan = manifest
            .plan()
            .map_err(|error| format!("invalid vision-stack shard plan: {error}"))?;
        let execution_preparation = prepare_browser_vision_stack_execution(
            &manifest,
            qkv_outcome,
            self.device.features().contains(wgpu::Features::SHADER_F16),
        )
        .map_err(|error| error.to_string())?;
        let layer_plan = execution_preparation.layer_plan;
        let weight_plan = execution_preparation.weights;
        let fp16_qkv_plan = weight_plan
            .tiled_fp16_qkv_kernel
            .filter(|_| ENABLE_TILED_FP16_QKV)
            .map(|kernel| {
                if kernel != KernelId::VisionQkvFusedF16Weights {
                    return Err(format!("unsupported tiled FP16 Q/K/V kernel {kernel}"));
                }
                plan_vision_qkv_fused_f16_weight_geometry(
                    manifest.tokens,
                    manifest.hidden_size,
                    manifest.hidden_size,
                    VisionQkvFusedTargetLimits {
                        min_storage_buffer_offset_alignment: self
                            .capabilities
                            .limits
                            .min_storage_buffer_offset_alignment,
                        max_storage_buffers_per_shader_stage: self
                            .capabilities
                            .limits
                            .max_storage_buffers_per_shader_stage,
                        max_storage_buffer_binding_size: self
                            .capabilities
                            .limits
                            .max_storage_buffer_binding_size,
                        max_buffer_size: self.capabilities.limits.max_buffer_size,
                        max_compute_workgroups_per_dimension: self
                            .capabilities
                            .limits
                            .max_compute_workgroups_per_dimension,
                    },
                )
                .map_err(|error| format!("invalid tiled FP16 Q/K/V plan: {error}"))
            })
            .transpose()?;
        let shader_sources = if qkv_outcome == VisionQkvSelectionOutcome::Fused {
            vision_qkv_stack_shader_sources(activation_strategy)?
        } else {
            vision_stack_shader_sources(
                activation_strategy,
                weight_plan.projection_kernel,
                weight_plan.matrix_weight_storage,
                weight_plan.matrix_weight_layout,
                weight_plan.activation_storage,
                weight_plan.rope_kernel,
            )?
        };
        let storage_alignment =
            u64::from(self.capabilities.limits.min_storage_buffer_offset_alignment).max(1);
        let activation_layout = match activation_strategy {
            VisionStackActivationStrategy::SeparateBuffers => None,
            VisionStackActivationStrategy::StaticArenaNoAlias
            | VisionStackActivationStrategy::StaticArenaAlias => Some(
                layer_plan
                    .stack_activation_layout(VisionStackActivationLayoutConfig {
                        allow_aliasing: matches!(
                            activation_strategy,
                            VisionStackActivationStrategy::StaticArenaAlias
                        ),
                        storage_buffer_offset_alignment: storage_alignment,
                        arena_alignment: storage_alignment,
                    })
                    .map_err(|error| format!("invalid vision-stack activation layout: {error}"))?,
            ),
        };
        self.validate_vision_stack_capabilities(&manifest, &plan, &layer_plan, &weight_plan)?;
        if let Some(layout) = activation_layout.as_ref() {
            self.validate_vision_stack_activation_layout(
                &plan,
                &layer_plan,
                layout,
                storage_alignment,
            )?;
        }
        let static_plan = activation_layout
            .as_ref()
            .map(|layout| {
                BrowserVisionStackStaticPlan::new(
                    &plan,
                    layout,
                    activation_strategy,
                    self.capabilities.limits.min_storage_buffer_offset_alignment,
                )
            })
            .transpose()?;
        let memory_hardening = match (
            memory_hardening,
            activation_layout.as_ref(),
            static_plan.as_ref(),
        ) {
            (None, _, _) => None,
            (Some(mode), Some(layout), Some(static_plan)) => Some(
                VisionStackMemoryHardeningPlan::new(
                    mode,
                    storage_alignment,
                    layout.scratch_arena_bytes,
                    plan.readback_bytes,
                    static_plan.peak_gpu_data_bytes,
                )
                .map_err(|error| format!("invalid vision-stack memory-hardening plan: {error}"))?,
            ),
            (Some(_), _, _) => {
                return Err(
                    "vision-stack memory hardening requires a static activation layout".to_owned(),
                );
            }
        };
        if let Some(hardening) = memory_hardening.as_ref() {
            self.validate_vision_stack_memory_hardening_capabilities(hardening)?;
        }
        let protocol = VisionStackShardProtocol::new(manifest)
            .map_err(|error| format!("invalid vision-stack shard protocol: {error}"))?;
        Ok(BrowserVisionStackPreparedSession {
            protocol,
            plan,
            layer_plan,
            weight_plan,
            fp16_qkv_plan,
            activation_strategy,
            activation_layout,
            static_plan,
            memory_hardening,
            storage_alignment: self.capabilities.limits.min_storage_buffer_offset_alignment,
            shader_sources,
        })
    }

    fn preflight_vision_stack_shard(
        &self,
        shard_id: &str,
        bytes: &js_sys::Uint8Array,
    ) -> Result<String, String> {
        let owner = self.vision_stack_session.borrow_mut();
        let mut session = owner
            .stored_mut()
            .ok_or_else(|| "vision-stack preflight has not been started".to_owned())?;
        if session.gpu.is_some() {
            return Err("vision-stack preflight cannot continue after GPU allocation".to_owned());
        }
        let observation = inspect_js_vision_stack_shard(
            session.protocol.manifest(),
            &session.weight_plan,
            shard_id,
            bytes,
        )?;
        session
            .protocol
            .accept_preflight(&observation)
            .map_err(|error| format!("vision-stack preflight rejected {shard_id}: {error}"))?;
        vision_stack_status_json(&session, false).map_err(|error| error.to_string())
    }

    fn preflight_vision_stack_manifest_shard(&self, shard_id: &str) -> Result<String, String> {
        let owner = self.vision_stack_session.borrow();
        let mut session = owner
            .stored_mut()
            .ok_or_else(|| "vision-stack preflight has not been started".to_owned())?;
        session
            .protocol
            .accept_deferred_preflight(shard_id)
            .map_err(|error| format!("vision-stack preflight rejected {shard_id}: {error}"))?;
        vision_stack_status_json(&session, false).map_err(|error| error.to_string())
    }

    async fn start_vision_stack_sharded(
        &self,
        shard_id: &str,
        bytes: &js_sys::Uint8Array,
    ) -> Result<String, String> {
        let (lease, session) = self
            .vision_stack_session
            .borrow_mut()
            .acquire()
            .map_err(|_error| "vision-stack session is unavailable".to_owned())?;
        let transaction = run_vision_stack_operation_transaction(
            session,
            |shadow| {
                let validated = self.validate_vision_stack_start(shadow, shard_id, bytes)?;
                let prepared_first_effect = self.prepare_vision_stack_first_error_scope()?;
                Ok((validated, prepared_first_effect))
            },
            async move |shadow, (_validated, prepared_first_effect), effect_boundary| {
                effect_boundary
                    .run_webgpu_effect(async move |post_effect| {
                        self.start_vision_stack_sharded_once(
                            shadow,
                            bytes,
                            prepared_first_effect,
                            post_effect,
                        )
                        .await
                    })
                    .await
            },
        )
        .await;
        let (outcome, result) = {
            let mut owner = self.vision_stack_session.borrow_mut();
            complete_vision_stack_async_operation(
                &mut owner,
                lease,
                VisionStackAsyncOperation::Start,
                transaction,
            )
        };
        self.finish_vision_stack_transaction(outcome, result, "starting")
    }

    fn validate_vision_stack_start(
        &self,
        session: &mut BrowserVisionStackSession,
        shard_id: &str,
        bytes: &js_sys::Uint8Array,
    ) -> Result<VisionStackShardKind, BrowserVisionStackError> {
        if session.gpu.is_some() {
            return Err("vision-stack GPU execution was already started"
                .to_owned()
                .into());
        }
        self.validate_vision_stack_session_authority(session)?;
        let observation = inspect_js_vision_stack_shard(
            session.protocol.manifest(),
            &session.weight_plan,
            shard_id,
            bytes,
        )?;
        let accepted = session
            .protocol
            .accept_execution(&observation)
            .map_err(|error| {
                let mut message = "vision-stack execution rejected ".to_owned();
                message.push_str(shard_id);
                message.push_str(": ");
                message.push_str(&error.to_string());
                message
            })?;
        if accepted.kind != VisionStackShardKind::Input {
            let mut message = "vision-stack execution expected input, got ".to_owned();
            message.push_str(shard_id);
            return Err(message.into());
        }
        Ok(accepted.kind)
    }

    fn validate_vision_stack_session_authority(
        &self,
        session: &BrowserVisionStackSession,
    ) -> Result<(), BrowserVisionStackError> {
        match session.activation_strategy {
            VisionStackActivationStrategy::SeparateBuffers => {
                if session.activation_layout.is_some() || session.static_plan.is_some() {
                    return Err(
                        "separate-buffer session retained a static activation authority"
                            .to_owned()
                            .into(),
                    );
                }
            }
            VisionStackActivationStrategy::StaticArenaNoAlias
            | VisionStackActivationStrategy::StaticArenaAlias => {
                let layout = session.activation_layout.as_ref().ok_or_else(|| {
                    BrowserVisionStackError(
                        "static vision-stack session omitted its activation layout".to_owned(),
                    )
                })?;
                let expected = BrowserVisionStackStaticPlan::new(
                    &session.plan,
                    layout,
                    session.activation_strategy,
                    session.storage_alignment,
                )?;
                if session.static_plan.as_ref() != Some(&expected) {
                    return Err(
                        "static vision-stack session authority drifted after preflight"
                            .to_owned()
                            .into(),
                    );
                }
            }
        }

        match session.qkv_selection.outcome() {
            VisionQkvSelectionOutcome::Fused => {
                session.qkv_physical_execution.as_ref().ok_or_else(|| {
                    BrowserVisionStackError(
                        "fused Q/K/V session omitted its sealed physical authority".to_owned(),
                    )
                })?;
                if session.qkv_physical_commands.is_none()
                    || session.qkv_execution_evidence_plan.is_none()
                {
                    return Err("fused Q/K/V session authority is incomplete or mismatched"
                        .to_owned()
                        .into());
                }
            }
            VisionQkvSelectionOutcome::Disabled
            | VisionQkvSelectionOutcome::FallbackUnsupportedTarget => {
                if session.qkv_physical_execution.is_some()
                    || session.qkv_physical_commands.is_some()
                    || session.qkv_execution_evidence_plan.is_some()
                {
                    return Err("legacy vision-stack session retained fused Q/K/V authority"
                        .to_owned()
                        .into());
                }
            }
        }
        Ok(())
    }

    async fn start_vision_stack_sharded_once(
        &self,
        session: &mut BrowserVisionStackSession,
        bytes: &js_sys::Uint8Array,
        prepared_first_effect: PreparedVisionStackFirstErrorScope<'_>,
        post_effect: &VisionStackPostEffectToken,
    ) -> Result<String, BrowserVisionStackError> {
        let guards = self
            .push_vision_stack_error_scopes(prepared_first_effect, post_effect)
            .await?;
        let completed = self
            .complete_browser_error_scoped_operation(guards, async {
                self.allocate_vision_stack_gpu(session, bytes, post_effect)
                    .map_err(BrowserVisionStackError::from)
            })
            .await?;
        let (operation, captured) = completed;
        if !captured.is_empty() {
            let operation_context = operation
                .as_ref()
                .err()
                .map(|error| format!("; operation also failed: {error}"))
                .unwrap_or_default();
            return Err(format!(
                "vision-stack start captured WebGPU errors: {captured:?}{operation_context}"
            )
            .into());
        }
        session.gpu = Some(operation?);
        self.require_no_uncaptured_errors("vision-stack start")?;
        Ok(vision_stack_status_json(session, false).map_err(|error| error.to_string())?)
    }

    fn allocate_vision_stack_gpu(
        &self,
        session: &BrowserVisionStackSession,
        input: &js_sys::Uint8Array,
        post_effect: &VisionStackPostEffectToken,
    ) -> Result<BrowserVisionStackGpuState, String> {
        let mut pipeline_specs = Vec::new();
        let mut shader_blake3 = BTreeMap::new();
        for (&kernel, source) in &session.shader_sources {
            let module = pvlc_wgsl::module(kernel)
                .expect("every resident vision-stack kernel must have fixed WGSL");
            pipeline_specs.push((
                kernel,
                (kernel.as_str(), source.as_str(), module.spec.entry_point),
            ));
            shader_blake3.insert(kernel, blake3_hex(source));
        }
        #[rustfmt::skip]
        let pipelines = collect_vision_stack_session_resources(pipeline_specs, |(label, source, entry)| self.vision_stack_pipeline(post_effect, label, source, entry))?;

        let hidden_bytes = session.plan.hidden_bytes;
        let main_labels = match session.activation_strategy {
            VisionStackActivationStrategy::SeparateBuffers => [
                "vision-stack-activation-main-0",
                "vision-stack-activation-main-1",
            ],
            VisionStackActivationStrategy::StaticArenaNoAlias
            | VisionStackActivationStrategy::StaticArenaAlias => [
                "vision-stack-activation-main-a",
                "vision-stack-activation-main-b",
            ],
        };
        let main_buffers = [
            self.create_runtime_buffer(
                main_labels[0],
                hidden_bytes,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            ),
            self.create_runtime_buffer(
                main_labels[1],
                hidden_bytes,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            ),
        ];
        let scratch = match session.activation_strategy {
            VisionStackActivationStrategy::SeparateBuffers => {
                let buffers = session.layer_plan.dispatches[..11]
                    .iter()
                    .enumerate()
                    .map(|(index, dispatch)| {
                        let mut label = "vision-stack-activation-scratch-".to_owned();
                        label.push_str(&index.to_string());
                        self.create_runtime_buffer(
                            &label,
                            dispatch.invocation.output_bytes,
                            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        )
                    })
                    .collect::<Vec<_>>();
                if main_buffers.len() + buffers.len() != 13 {
                    return Err("vision-stack separate activation buffer count drifted".to_owned());
                }
                BrowserVisionStackScratch::Separate(buffers)
            }
            VisionStackActivationStrategy::StaticArenaNoAlias
            | VisionStackActivationStrategy::StaticArenaAlias => {
                let layout = session
                    .activation_layout
                    .as_ref()
                    .ok_or_else(|| "vision-stack static activation layout is missing".to_owned())?;
                let arena_bytes = session
                    .memory_hardening
                    .as_ref()
                    .map_or(layout.scratch_arena_bytes, |plan| {
                        plan.physical_scratch_bytes()
                    });
                let mut usages = wgpu::BufferUsages::STORAGE;
                if session.memory_hardening.is_some() {
                    usages |= wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
                }
                let arena = self.create_runtime_buffer(
                    "vision-stack-activation-scratch-arena",
                    arena_bytes,
                    usages,
                );
                let ranges = layout
                    .scratch_allocations
                    .iter()
                    .map(|allocation| {
                        let offset = match session.memory_hardening.as_ref() {
                            Some(plan) => plan
                                .shift_scratch_binding(allocation.offset, allocation.size)?
                                .physical_offset(),
                            None => allocation.offset,
                        };
                        Ok(VisionStackTensorRange {
                            offset,
                            bytes: allocation.size,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                if let Some(plan) = session.memory_hardening.as_ref() {
                    write_vision_stack_hardening_patterns(&self.queue, &arena, plan)?;
                }
                BrowserVisionStackScratch::Static { arena, ranges }
            }
        };

        let manifest = session.protocol.manifest();
        let boundary_bytes = u64::try_from(manifest.cu_seqlens.len())
            .map_err(|_| "vision-stack boundary length overflowed".to_owned())?
            .checked_mul(4)
            .ok_or_else(|| "vision-stack boundary byte length overflowed".to_owned())?;
        let boundary_buffer = self.create_runtime_buffer(
            "vision-stack-cu-seqlens",
            boundary_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.queue.write_buffer(
            &boundary_buffer,
            0,
            bytemuck::cast_slice(&manifest.cu_seqlens),
        );

        let uniform_stride =
            vision_layer_uniform_stride(self.device.limits().min_uniform_buffer_offset_alignment);
        let uniform_arena_bytes = uniform_stride
            .checked_mul(13)
            .ok_or_else(|| "vision-stack uniform arena overflowed".to_owned())?;
        let uniform_len = usize::try_from(uniform_arena_bytes)
            .map_err(|_| "vision-stack uniform arena is too large".to_owned())?;
        let mut uniform_contents = std::iter::repeat_n(0_u8, uniform_len).collect::<Vec<_>>();
        for (index, dispatch) in session.layer_plan.dispatches.iter().enumerate() {
            write_uniform_slot(
                &mut uniform_contents,
                uniform_stride,
                index,
                dispatch.uniform_words,
            )?;
        }
        if let Some(qkv_plan) = session.fp16_qkv_plan.as_ref() {
            write_uniform_slot(
                &mut uniform_contents,
                uniform_stride,
                1,
                qkv_plan.uniform_words,
            )?;
        }
        write_uniform_slot(
            &mut uniform_contents,
            uniform_stride,
            12,
            session.layer_plan.dispatches[0].uniform_words,
        )?;
        let mut uniform_usage = wgpu::BufferUsages::UNIFORM;
        if session.qkv_physical_commands.is_some() {
            uniform_usage |= wgpu::BufferUsages::COPY_DST;
        }
        let uniform_buffer = self.create_initialized_buffer(
            "vision-stack-uniform-arena",
            &uniform_contents,
            uniform_usage,
        );
        let mut readback_buffer = if session.qkv_physical_commands.is_none() {
            let buffer = self.create_runtime_buffer(
                "vision-stack-readback",
                session
                    .memory_hardening
                    .as_ref()
                    .map_or(session.plan.readback_bytes, |plan| {
                        plan.physical_readback_bytes()
                    }),
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            Some(buffer)
        } else {
            None
        };
        upload_js_range(
            &self.queue,
            &main_buffers[0],
            input,
            VisionStackTensorRange {
                offset: 0,
                bytes: hidden_bytes,
            },
        )?;

        let mut qkv_physical_storage = BrowserVisionQkvPhysicalStorage::new();
        if let Some(qkv_physical_commands) = session.qkv_physical_commands.as_ref() {
            let fused_uniform_words = qkv_physical_commands
                .fused_uniform_words(0)
                .ok_or_else(|| "fused Q/K/V uniform authority is missing".to_owned())?;
            self.queue.write_buffer(
                &uniform_buffer,
                uniform_stride,
                bytemuck::cast_slice(&fused_uniform_words),
            );
            let placeholder = scratch.binding(0)?;
            let mapped_range = RefCell::new(None);
            let encoder = RefCell::new(self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("vision-stack-qkv-start-command-adapter"),
                },
            ));
            let context = BrowserVisionQkvLayerResolutionContext {
                device: &self.device,
                pipelines: &pipelines,
                buffers: qkv_physical_storage.buffers.clone(),
                encoder: &encoder,
                norm1_output: placeholder,
                query_weight: placeholder,
                query_bias: placeholder,
                key_weight: placeholder,
                key_bias: placeholder,
                value_weight: placeholder,
                value_bias: placeholder,
                cu_seqlens: VisionStackBufferBinding::whole(&boundary_buffer),
                attention_output: placeholder,
                uniform_buffer: &uniform_buffer,
                uniform_stride,
                mapped_range: &mapped_range,
            };
            let mut qkv_bind_groups = BrowserVisionQkvLayerBindGroups::new();
            self.apply_vision_qkv_web_start_commands(
                &context,
                &mut qkv_physical_storage,
                &mut qkv_bind_groups,
                qkv_physical_commands,
            )
            .map_err(|_| "cannot apply typed Q/K/V start commands".to_owned())?;
            let typed_readback = qkv_physical_storage
                .buffers
                .get(&VisionQkvWebPhysicalBuffer::Readback)
                .expect("typed Q/K/V start commands create Readback")
                .clone();
            readback_buffer = Some(typed_readback);
        }
        let fp16_qkv_workspace = session.fp16_qkv_plan.as_ref().map(|qkv_plan| {
            self.create_runtime_buffer(
                "vision-stack-fp16-qkv-workspace",
                qkv_plan.output_layout.physical_bytes,
                wgpu::BufferUsages::STORAGE,
            )
        });

        let readback_buffer = readback_buffer
            .ok_or_else(|| "vision-stack readback authority was not created".to_owned())?;
        let spatial_rope = if let Some(rope) = session.spatial_rope.as_ref() {
            if rope.stack_plan.rope_table_buffer_count != 2
                || rope.stack_plan.rope_table_upload_count != 2
            {
                return Err(
                    "vision-stack spatial RoPE must own exactly two shared table buffers and uploads"
                        .to_owned(),
                );
            }
            let cos_buffer = self.create_initialized_buffer(
                "vision-stack-rope-cos",
                bytemuck::cast_slice(&rope.cos),
                wgpu::BufferUsages::STORAGE,
            );
            let sin_buffer = self.create_initialized_buffer(
                "vision-stack-rope-sin",
                bytemuck::cast_slice(&rope.sin),
                wgpu::BufferUsages::STORAGE,
            );
            let uniform_buffer = self.create_initialized_buffer(
                "vision-stack-rope-uniform",
                bytemuck::cast_slice(&rope.layer_plan.rope.uniform_words),
                wgpu::BufferUsages::UNIFORM,
            );
            let gpu_state = BrowserVisionSpatialRopeGpuState {
                cos_buffer,
                sin_buffer,
                uniform_buffer,
            };
            Some(gpu_state)
        } else {
            None
        };

        Ok(BrowserVisionStackGpuState {
            pipelines,
            shader_blake3,
            main_buffers,
            scratch,
            boundary_buffer,
            uniform_buffer,
            readback_buffer,
            qkv_physical_storage,
            fp16_qkv_workspace,
            spatial_rope,
            uniform_stride,
            current_main: 0,
            started_ms: js_sys::Date::now(),
        })
    }

    fn enqueue_vision_stack_sharded_layer(
        &self,
        shard_id: &str,
        bytes: &js_sys::Uint8Array,
    ) -> Result<String, String> {
        #[rustfmt::skip]
        run_vision_stack_streaming_session_layer(
            &*self.vision_stack_session.borrow(),
            &self.execution_busy,
            &mut *self.vision_stack_streaming_weight_cache.borrow_mut(),
            |session| self.validate_vision_stack_streaming_layer(session, shard_id, bytes),
            |slot, range| self.create_vision_stack_streaming_weight_buffer(slot, range),
            |slot, range, resource| self.upload_vision_stack_streaming_weight(bytes, slot, range, resource),
            |session, layer_index, checkpoint_slot, resources| self.encode_and_submit_vision_stack_layer(
                session,
                layer_index,
                checkpoint_slot,
                resources,
            ),
        )?;
        let owner = self.vision_stack_session.borrow();
        let session = owner
            .stored()
            .ok_or_else(|| "vision-stack streaming session was not restored".to_owned())?;
        vision_stack_status_json(&session, false).map_err(|error| error.to_string())
    }

    fn enqueue_vision_stack_sharded_resident_layer(
        &self,
        shard_id: &str,
    ) -> Result<String, String> {
        let owner = self.vision_stack_session.borrow();
        let (lease, session) = owner
            .acquire()
            .map_err(|error| format!("vision-stack session is unavailable: {error:?}"))?;
        let mut shadow = session.clone();
        let (layer_index, checkpoint_slot) = match self
            .validate_vision_stack_resident_layer(&mut shadow, shard_id)
        {
            Ok(validated) => validated,
            Err(error) => {
                let completion = owner.complete(lease, session, crate::CompletionAction::Restore);
                let completion =
                    coordinate_vision_stack_completion_busy(&self.execution_busy, completion);
                return match completion {
                    crate::CompletionOutcome::Restored => Err(error.to_string()),
                    _ => Err(format!(
                        "resident vision layer admission completed unexpectedly: {completion:?}"
                    )),
                };
            }
        };
        let resources = (|| {
            let resident = shadow.resident_weights.as_ref().ok_or_else(|| {
                "resident vision layer API requires a resident-weight session".to_owned()
            })?;
            if resident.disposition != BrowserVisionStackWeightResidency::Ready {
                return Err(
                    "resident vision layer API cannot read an incomplete weight cache".to_owned(),
                );
            }
            let layer = usize::try_from(layer_index)
                .map_err(|_| "resident vision layer index does not fit usize".to_owned())?;
            let layer_count = usize::try_from(shadow.plan.layer_count)
                .map_err(|_| "resident vision layer count does not fit usize".to_owned())?;
            let cache = resident.cache.borrow();
            if !cache.is_ready_for(&resident.key, layer_count) {
                return Err(
                    "resident vision-weight cache identity changed during execution".to_owned(),
                );
            }
            let resources = cache
                .clone_layer(layer)
                .map_err(|error| format!("cannot resolve resident vision layer: {error}"))?;
            Ok::<Vec<wgpu::Buffer>, String>(resources)
        })();
        let resources = match resources {
            Ok(resources) => resources,
            Err(error) => {
                let completion = owner.complete(lease, session, crate::CompletionAction::Restore);
                let completion =
                    coordinate_vision_stack_completion_busy(&self.execution_busy, completion);
                return match completion {
                    crate::CompletionOutcome::Restored => Err(error),
                    _ => Err(format!(
                        "resident vision layer cache check completed unexpectedly: {completion:?}"
                    )),
                };
            }
        };
        if let Err(error) = self.encode_and_submit_vision_stack_layer(
            &mut shadow,
            layer_index,
            checkpoint_slot,
            &resources,
        ) {
            let completion = owner.complete(lease, session, crate::CompletionAction::Finish);
            let completion =
                coordinate_vision_stack_completion_busy(&self.execution_busy, completion);
            return match completion {
                crate::CompletionOutcome::Finished => Err(error.to_string()),
                _ => Err(format!(
                    "resident vision layer GPU failure completed unexpectedly: {completion:?}"
                )),
            };
        }
        let completion = owner.complete(lease, shadow, crate::CompletionAction::Restore);
        let completion = coordinate_vision_stack_completion_busy(&self.execution_busy, completion);
        if completion != crate::CompletionOutcome::Restored {
            return Err(format!(
                "resident vision layer completed unexpectedly: {completion:?}"
            ));
        }
        let session = owner
            .stored()
            .ok_or_else(|| "resident vision-stack session was not restored".to_owned())?;
        vision_stack_status_json(&session, false).map_err(|error| error.to_string())
    }

    fn validate_vision_stack_streaming_layer(
        &self,
        session: &mut BrowserVisionStackSession,
        shard_id: &str,
        bytes: &js_sys::Uint8Array,
    ) -> Result<VisionStackStreamingLayerSchedule, BrowserVisionStackError> {
        let (layer_index, checkpoint_slot) =
            self.validate_vision_stack_layer(session, shard_id, bytes)?;
        let ranges = session
            .weight_plan
            .ranges
            .map(|range| VisionStackStreamingWeightRange::new(range.offset, range.bytes));
        Ok(VisionStackStreamingLayerSchedule::new(
            layer_index,
            checkpoint_slot,
            ranges,
        ))
    }

    fn validate_vision_stack_resident_layer(
        &self,
        session: &mut BrowserVisionStackSession,
        shard_id: &str,
    ) -> Result<(u32, Option<usize>), BrowserVisionStackError> {
        let descriptor = session
            .protocol
            .manifest()
            .shards
            .iter()
            .find(|descriptor| descriptor.id == shard_id)
            .cloned()
            .ok_or_else(|| format!("resident vision shard {shard_id:?} is undeclared"))?;
        let observation = VisionStackShardObservation {
            id: descriptor.id,
            bytes: descriptor.bytes,
            blake3: descriptor.blake3,
            all_finite: true,
        };
        let accepted = session
            .protocol
            .accept_execution(&observation)
            .map_err(|error| {
                format!("resident vision-stack execution rejected {shard_id}: {error}")
            })?;
        let layer = accepted
            .layer_index
            .filter(|_| accepted.kind == VisionStackShardKind::Layer)
            .ok_or_else(|| format!("resident vision-stack expected a layer shard: {shard_id}"))?;
        Ok((layer, accepted.checkpoint_slot))
    }

    fn create_vision_stack_streaming_weight_buffer(
        &self,
        slot: usize,
        range: VisionStackStreamingWeightRange,
    ) -> Result<wgpu::Buffer, BrowserVisionStackError> {
        Ok(self.create_runtime_buffer(
            &format!("vision-stack-streaming-weight-{slot:02}"),
            range.length_bytes(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ))
    }

    fn upload_vision_stack_streaming_weight(
        &self,
        bytes: &js_sys::Uint8Array,
        _slot: usize,
        range: VisionStackStreamingWeightRange,
        resource: &wgpu::Buffer,
    ) -> Result<(), BrowserVisionStackError> {
        upload_js_range(
            &self.queue,
            resource,
            bytes,
            vision_stack_streaming_tensor_range(range),
        )
        .map_err(BrowserVisionStackError::from)
    }

    async fn run_vision_stack_sharded_layer(
        &self,
        shard_id: &str,
        bytes: &js_sys::Uint8Array,
    ) -> Result<String, String> {
        let (lease, session) = self
            .vision_stack_session
            .borrow_mut()
            .acquire()
            .map_err(|_error| "vision-stack session is unavailable".to_owned())?;
        let transaction = run_vision_stack_operation_transaction(
            session,
            |shadow| {
                let validated = self.validate_vision_stack_layer(shadow, shard_id, bytes)?;
                let prepared_first_effect = self.prepare_vision_stack_first_error_scope()?;
                Ok((validated, prepared_first_effect))
            },
            async move |shadow,
                        ((layer, checkpoint_slot), prepared_first_effect),
                        effect_boundary| {
                effect_boundary
                    .run_webgpu_effect(async move |post_effect| {
                        self.run_vision_stack_sharded_layer_once(
                            shadow,
                            layer,
                            checkpoint_slot,
                            bytes,
                            prepared_first_effect,
                            post_effect,
                        )
                        .await
                    })
                    .await
            },
        )
        .await;
        let (outcome, result) = {
            let mut owner = self.vision_stack_session.borrow_mut();
            complete_vision_stack_async_operation(
                &mut owner,
                lease,
                VisionStackAsyncOperation::Layer,
                transaction,
            )
        };
        self.finish_vision_stack_transaction(outcome, result, "running a layer")
    }

    fn validate_vision_stack_layer(
        &self,
        session: &mut BrowserVisionStackSession,
        shard_id: &str,
        bytes: &js_sys::Uint8Array,
    ) -> Result<(u32, Option<usize>), BrowserVisionStackError> {
        let observation = inspect_js_vision_stack_shard(
            session.protocol.manifest(),
            &session.weight_plan,
            shard_id,
            bytes,
        )?;
        let accepted = session
            .protocol
            .accept_execution(&observation)
            .map_err(|error| {
                let mut message = "vision-stack execution rejected ".to_owned();
                message.push_str(shard_id);
                message.push_str(": ");
                message.push_str(&error.to_string());
                message
            })?;
        let layer = accepted
            .layer_index
            .filter(|_| accepted.kind == VisionStackShardKind::Layer)
            .ok_or_else(|| {
                let mut message = "vision-stack expected a layer shard, got ".to_owned();
                message.push_str(shard_id);
                message
            })?;
        Ok((layer, accepted.checkpoint_slot))
    }

    async fn run_vision_stack_sharded_layer_once(
        &self,
        session: &mut BrowserVisionStackSession,
        layer: u32,
        checkpoint_slot: Option<usize>,
        bytes: &js_sys::Uint8Array,
        prepared_first_effect: PreparedVisionStackFirstErrorScope<'_>,
        post_effect: &VisionStackPostEffectToken,
    ) -> Result<String, BrowserVisionStackError> {
        let guards = self
            .push_vision_stack_error_scopes(prepared_first_effect, post_effect)
            .await?;
        let completed = self
            .complete_browser_error_scoped_operation(guards, async {
                self.execute_vision_stack_layer(session, layer, checkpoint_slot, bytes)
                    .await
                    .map_err(BrowserVisionStackError::from)
            })
            .await?;
        let (operation, captured) = completed;
        if !captured.is_empty() {
            return Err(
                format!("vision-stack layer {layer} captured WebGPU errors: {captured:?}").into(),
            );
        }
        operation?;
        self.require_no_uncaptured_errors(&format!("vision-stack layer {layer}"))?;
        Ok(vision_stack_status_json(session, false).map_err(|error| error.to_string())?)
    }

    async fn execute_vision_stack_layer(
        &self,
        session: &mut BrowserVisionStackSession,
        layer: u32,
        checkpoint_slot: Option<usize>,
        bytes: &js_sys::Uint8Array,
    ) -> Result<(), String> {
        let ranges = session
            .weight_plan
            .ranges
            .map(|range| VisionStackTensorRange {
                offset: range.offset,
                bytes: range.bytes,
            });
        let weights = ranges
            .iter()
            .enumerate()
            .map(|(index, range)| {
                self.create_uploaded_js_buffer(
                    &format!("vision-stack-layer-{layer:02}-weight-{index}"),
                    bytes,
                    *range,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.encode_and_submit_vision_stack_layer(session, layer, checkpoint_slot, &weights)?;
        await_queue_completion(&self.queue).await?;
        destroy_vision_qkv_web_layer_weights(&weights);
        Ok(())
    }

    fn encode_and_submit_vision_stack_layer(
        &self,
        session: &mut BrowserVisionStackSession,
        layer: u32,
        checkpoint_slot: Option<usize>,
        weights: &[wgpu::Buffer],
    ) -> Result<(), BrowserVisionStackError> {
        if session.fp16_qkv_plan.is_some() {
            return self.encode_and_submit_vision_stack_fp16_qkv_layer(
                session,
                layer,
                checkpoint_slot,
                weights,
            );
        }
        let layer_plan = session.layer_plan;
        let rope_kernel = session.weight_plan.rope_kernel;
        let spatial_rope_layer_plan = session
            .spatial_rope
            .as_ref()
            .map(|spatial_rope| spatial_rope.layer_plan);
        let fused_spatial_rope_ranges = if spatial_rope_layer_plan.is_some()
            && session.qkv_selection.outcome() == VisionQkvSelectionOutcome::Fused
        {
            let physical_commands = session
                .qkv_physical_commands
                .as_ref()
                .ok_or_else(|| "fused Q/K/V physical command plan is missing".to_owned())?;
            Some(vision_qkv_web_attention_workspace_ranges(
                physical_commands,
                layer,
            )?)
        } else {
            None
        };
        let gpu = session
            .gpu
            .as_mut()
            .ok_or_else(|| "vision-stack GPU execution has not been started".to_owned())?;
        let current_buffer = &gpu.main_buffers[gpu.current_main];
        let next_index = 1 - gpu.current_main;
        let next_buffer = &gpu.main_buffers[next_index];
        let current = VisionStackBufferBinding::whole(current_buffer);
        let next = VisionStackBufferBinding::whole(next_buffer);
        let weight_bindings = weights
            .iter()
            .map(VisionStackBufferBinding::whole)
            .collect::<Vec<_>>();
        let boundary = VisionStackBufferBinding::whole(&gpu.boundary_buffer);
        let scratch = (0..11)
            .map(|index| gpu.scratch.binding(index))
            .collect::<Result<Vec<_>, _>>()?;
        let encoder = RefCell::new(self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("vision-stack-layer-encoder"),
            },
        ));

        match session.qkv_selection.outcome() {
            VisionQkvSelectionOutcome::Fused => {
                let norm1_bind_group = self.create_vision_stack_bind_group(
                    &layer_plan,
                    gpu,
                    0,
                    &[current, weight_bindings[0], weight_bindings[1]],
                    scratch[0],
                    0,
                )?;
                let post_bind_groups = [
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        5,
                        &[scratch[4], weight_bindings[8], weight_bindings[9]],
                        scratch[5],
                        5,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        6,
                        &[current, scratch[5]],
                        scratch[6],
                        6,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        7,
                        &[scratch[6], weight_bindings[10], weight_bindings[11]],
                        scratch[7],
                        7,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        8,
                        &[scratch[7], weight_bindings[12], weight_bindings[13]],
                        scratch[8],
                        8,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        9,
                        &[scratch[8]],
                        scratch[9],
                        9,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        10,
                        &[scratch[9], weight_bindings[14], weight_bindings[15]],
                        scratch[10],
                        10,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        11,
                        &[scratch[6], scratch[10]],
                        next,
                        11,
                    )?,
                ];
                let mut qkv_bind_groups = BrowserVisionQkvLayerBindGroups::new();
                let mapped_range = RefCell::new(None);
                let context = BrowserVisionQkvLayerResolutionContext {
                    device: &self.device,
                    pipelines: &gpu.pipelines,
                    buffers: gpu.qkv_physical_storage.buffers.clone(),
                    encoder: &encoder,
                    norm1_output: scratch[0],
                    query_weight: weight_bindings[2],
                    query_bias: weight_bindings[3],
                    key_weight: weight_bindings[4],
                    key_bias: weight_bindings[5],
                    value_weight: weight_bindings[6],
                    value_bias: weight_bindings[7],
                    cu_seqlens: boundary,
                    attention_output: scratch[4],
                    uniform_buffer: &gpu.uniform_buffer,
                    uniform_stride: gpu.uniform_stride,
                    mapped_range: &mapped_range,
                };
                let qkv_physical_commands = session
                    .qkv_physical_commands
                    .as_ref()
                    .ok_or_else(|| "fused Q/K/V physical command plan is missing".to_owned())?;
                let fused_dispatch_workgroups = qkv_physical_commands
                    .fused_dispatch_workgroups(layer)
                    .ok_or_else(|| {
                        format!("fused Q/K/V dispatch authority is missing for layer {layer}")
                    })?;
                apply_vision_qkv_web_layer_commands(
                    &context,
                    BrowserVisionQkvAllocationAuthority {
                        device: &self.device,
                        queue: &self.queue,
                        buffer_allocations: &self.buffer_allocations,
                    },
                    &mut gpu.qkv_physical_storage,
                    &mut qkv_bind_groups,
                    qkv_physical_commands,
                    layer,
                )
                .map_err(|error| format!("cannot apply typed Q/K/V layer commands: {error:?}"))?;
                let spatial_rope_bind_group =
                    match (spatial_rope_layer_plan.as_ref(), fused_spatial_rope_ranges) {
                        (Some(spatial_plan), Some([query_range, key_range])) => {
                            let workspace = gpu
                                .qkv_physical_storage
                                .buffers
                                .get(&VisionQkvWebPhysicalBuffer::Workspace)
                                .ok_or_else(|| {
                                    "fused Q/K/V workspace buffer is missing".to_owned()
                                })?;
                            let query = VisionStackBufferBinding {
                                buffer: workspace,
                                offset: query_range.offset,
                                size: Some(NonZeroU64::new(query_range.bytes).ok_or_else(
                                    || "fused Q/K/V query range is empty".to_owned(),
                                )?),
                                bytes: query_range.bytes,
                            };
                            let key =
                                VisionStackBufferBinding {
                                    buffer: workspace,
                                    offset: key_range.offset,
                                    size: Some(NonZeroU64::new(key_range.bytes).ok_or_else(
                                        || "fused Q/K/V key range is empty".to_owned(),
                                    )?),
                                    bytes: key_range.bytes,
                                };
                            Some(self.create_vision_stack_rope_bind_group(
                                spatial_plan,
                                gpu,
                                rope_kernel,
                                query,
                                key,
                            )?)
                        }
                        (None, None) => None,
                        _ => {
                            return Err("fused vision-stack spatial RoPE authority drifted"
                                .to_owned()
                                .into());
                        }
                    };
                {
                    let mut encoded = encoder.borrow_mut();
                    let mut pass = encoded.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("vision-stack-fused-layer-pass"),
                        timestamp_writes: None,
                    });
                    let norm1_dispatch = layer_plan.dispatches[0];
                    pass.set_pipeline(&gpu.pipelines[&norm1_dispatch.invocation.kernel]);
                    pass.set_bind_group(0, &norm1_bind_group, &[]);
                    pass.dispatch_workgroups(
                        norm1_dispatch.invocation.dispatch[0],
                        norm1_dispatch.invocation.dispatch[1],
                        norm1_dispatch.invocation.dispatch[2],
                    );
                    pass.set_pipeline(&gpu.pipelines[&KernelId::VisionQkvFusedF32]);
                    #[rustfmt::skip]
                    let fused_qkv_bind_group = get_vision_qkv_web_bind_group(&qkv_bind_groups, layer, VisionQkvWebBindGroupKind::FusedQkv);
                    pass.set_bind_group(0, fused_qkv_bind_group, &[]);
                    pass.dispatch_workgroups(
                        fused_dispatch_workgroups[0],
                        fused_dispatch_workgroups[1],
                        fused_dispatch_workgroups[2],
                    );
                    if let (Some(spatial_plan), Some(bind_group)) = (
                        spatial_rope_layer_plan.as_ref(),
                        spatial_rope_bind_group.as_ref(),
                    ) {
                        pass.set_pipeline(&gpu.pipelines[&rope_kernel]);
                        pass.set_bind_group(0, bind_group, &[]);
                        pass.dispatch_workgroups(
                            spatial_plan.rope.invocation.dispatch[0],
                            spatial_plan.rope.invocation.dispatch[1],
                            spatial_plan.rope.invocation.dispatch[2],
                        );
                    }
                    let attention_dispatch = layer_plan.dispatches[4];
                    pass.set_pipeline(&gpu.pipelines[&KernelId::VisionAttentionF32]);
                    #[rustfmt::skip]
                    let attention_bind_group = get_vision_qkv_web_bind_group(&qkv_bind_groups, layer, VisionQkvWebBindGroupKind::Attention);
                    pass.set_bind_group(0, attention_bind_group, &[]);
                    pass.dispatch_workgroups(
                        attention_dispatch.invocation.dispatch[0],
                        attention_dispatch.invocation.dispatch[1],
                        attention_dispatch.invocation.dispatch[2],
                    );
                    for (dispatch_index, bind_group) in (5..12).zip(&post_bind_groups) {
                        let dispatch = layer_plan.dispatches[dispatch_index];
                        pass.set_pipeline(&gpu.pipelines[&dispatch.invocation.kernel]);
                        pass.set_bind_group(0, bind_group, &[]);
                        pass.dispatch_workgroups(
                            dispatch.invocation.dispatch[0],
                            dispatch.invocation.dispatch[1],
                            dispatch.invocation.dispatch[2],
                        );
                    }
                }
                let mut encoded = encoder.into_inner();
                if let Some(slot) = checkpoint_slot {
                    let offset = u64::try_from(slot)
                        .map_err(|_| "vision-stack checkpoint slot overflowed".to_owned())?
                        .checked_mul(session.plan.hidden_bytes)
                        .ok_or_else(|| "vision-stack checkpoint offset overflowed".to_owned())?;
                    encoded.copy_buffer_to_buffer(
                        next_buffer,
                        0,
                        &gpu.readback_buffer,
                        offset,
                        session.plan.hidden_bytes,
                    );
                }
                self.submit_command_buffers([encoded.finish()]);
                drop(qkv_bind_groups);
            }
            VisionQkvSelectionOutcome::Disabled
            | VisionQkvSelectionOutcome::FallbackUnsupportedTarget => {
                let bind_groups = [
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        0,
                        &[current, weight_bindings[0], weight_bindings[1]],
                        scratch[0],
                        0,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        1,
                        &[scratch[0], weight_bindings[2], weight_bindings[3]],
                        scratch[1],
                        1,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        2,
                        &[scratch[0], weight_bindings[4], weight_bindings[5]],
                        scratch[2],
                        2,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        3,
                        &[scratch[0], weight_bindings[6], weight_bindings[7]],
                        scratch[3],
                        3,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        4,
                        &[scratch[1], scratch[2], scratch[3], boundary],
                        scratch[4],
                        4,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        5,
                        &[scratch[4], weight_bindings[8], weight_bindings[9]],
                        scratch[5],
                        5,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        6,
                        &[current, scratch[5]],
                        scratch[6],
                        6,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        7,
                        &[scratch[6], weight_bindings[10], weight_bindings[11]],
                        scratch[7],
                        7,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        8,
                        &[scratch[7], weight_bindings[12], weight_bindings[13]],
                        scratch[8],
                        8,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        9,
                        &[scratch[8]],
                        scratch[9],
                        9,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        10,
                        &[scratch[9], weight_bindings[14], weight_bindings[15]],
                        scratch[10],
                        10,
                    )?,
                    self.create_vision_stack_bind_group(
                        &layer_plan,
                        gpu,
                        11,
                        &[scratch[6], scratch[10]],
                        next,
                        11,
                    )?,
                ];
                let spatial_rope_bind_group = spatial_rope_layer_plan
                    .as_ref()
                    .map(|spatial_plan| {
                        self.create_vision_stack_rope_bind_group(
                            spatial_plan,
                            gpu,
                            rope_kernel,
                            scratch[1],
                            scratch[2],
                        )
                    })
                    .transpose()?;
                let mut encoded = encoder.into_inner();
                {
                    let mut pass = encoded.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("vision-stack-layer-pass"),
                        timestamp_writes: None,
                    });
                    for index in 0..4 {
                        let dispatch = layer_plan.dispatches[index];
                        pass.set_pipeline(&gpu.pipelines[&dispatch.invocation.kernel]);
                        pass.set_bind_group(0, &bind_groups[index], &[]);
                        pass.dispatch_workgroups(
                            dispatch.invocation.dispatch[0],
                            dispatch.invocation.dispatch[1],
                            dispatch.invocation.dispatch[2],
                        );
                    }
                    if let (Some(spatial_plan), Some(bind_group)) = (
                        spatial_rope_layer_plan.as_ref(),
                        spatial_rope_bind_group.as_ref(),
                    ) {
                        pass.set_pipeline(&gpu.pipelines[&rope_kernel]);
                        pass.set_bind_group(0, bind_group, &[]);
                        pass.dispatch_workgroups(
                            spatial_plan.rope.invocation.dispatch[0],
                            spatial_plan.rope.invocation.dispatch[1],
                            spatial_plan.rope.invocation.dispatch[2],
                        );
                    }
                    for index in 4..layer_plan.dispatches.len() {
                        let dispatch = layer_plan.dispatches[index];
                        pass.set_pipeline(&gpu.pipelines[&dispatch.invocation.kernel]);
                        pass.set_bind_group(0, &bind_groups[index], &[]);
                        pass.dispatch_workgroups(
                            dispatch.invocation.dispatch[0],
                            dispatch.invocation.dispatch[1],
                            dispatch.invocation.dispatch[2],
                        );
                    }
                }
                if let Some(slot) = checkpoint_slot {
                    let offset = u64::try_from(slot)
                        .map_err(|_| "vision-stack checkpoint slot overflowed".to_owned())?
                        .checked_mul(session.plan.hidden_bytes)
                        .ok_or_else(|| "vision-stack checkpoint offset overflowed".to_owned())?;
                    encoded.copy_buffer_to_buffer(
                        next_buffer,
                        0,
                        &gpu.readback_buffer,
                        offset,
                        session.plan.hidden_bytes,
                    );
                }
                self.submit_command_buffers([encoded.finish()]);
            }
        }
        gpu.current_main = next_index;
        Ok(())
    }

    fn encode_and_submit_vision_stack_fp16_qkv_layer(
        &self,
        session: &mut BrowserVisionStackSession,
        layer: u32,
        checkpoint_slot: Option<usize>,
        weights: &[wgpu::Buffer],
    ) -> Result<(), BrowserVisionStackError> {
        let layer_plan = session.layer_plan;
        let rope_kernel = session.weight_plan.rope_kernel;
        let qkv_plan = session
            .fp16_qkv_plan
            .ok_or_else(|| "tiled FP16 Q/K/V plan is missing".to_owned())?;
        let spatial_rope_layer_plan = session
            .spatial_rope
            .as_ref()
            .map(|spatial_rope| spatial_rope.layer_plan);
        let gpu = session
            .gpu
            .as_mut()
            .ok_or_else(|| "vision-stack GPU execution has not been started".to_owned())?;
        let current_buffer = &gpu.main_buffers[gpu.current_main];
        let next_index = 1 - gpu.current_main;
        let next_buffer = &gpu.main_buffers[next_index];
        let current = VisionStackBufferBinding::whole(current_buffer);
        let next = VisionStackBufferBinding::whole(next_buffer);
        let weight_bindings = weights
            .iter()
            .map(VisionStackBufferBinding::whole)
            .collect::<Vec<_>>();
        if weight_bindings.len() != 16 {
            return Err(format!(
                "tiled FP16 Q/K/V layer {layer} expected 16 weight tensors, got {}",
                weight_bindings.len()
            )
            .into());
        }
        let boundary = VisionStackBufferBinding::whole(&gpu.boundary_buffer);
        let scratch = (0..11)
            .map(|index| gpu.scratch.binding(index))
            .collect::<Result<Vec<_>, _>>()?;
        let fp16_qkv_workspace = gpu
            .fp16_qkv_workspace
            .as_ref()
            .ok_or_else(|| "tiled FP16 Q/K/V workspace is missing".to_owned())?;
        let [query, key, value] =
            vision_stack_fp16_qkv_workspace_bindings(fp16_qkv_workspace, &qkv_plan)?;
        let norm1_bind_group = self.create_vision_stack_bind_group(
            &layer_plan,
            gpu,
            0,
            &[current, weight_bindings[0], weight_bindings[1]],
            scratch[0],
            0,
        )?;
        let fused_qkv_bind_group = self.create_vision_stack_fp16_qkv_bind_group(
            gpu,
            scratch[0],
            weight_bindings[2],
            weight_bindings[3],
            weight_bindings[4],
            weight_bindings[5],
            weight_bindings[6],
            weight_bindings[7],
            VisionStackBufferBinding::whole(fp16_qkv_workspace),
        )?;
        let attention_bind_group = self.create_vision_stack_bind_group(
            &layer_plan,
            gpu,
            4,
            &[query, key, value, boundary],
            scratch[4],
            4,
        )?;
        let post_bind_groups = [
            self.create_vision_stack_bind_group(
                &layer_plan,
                gpu,
                5,
                &[scratch[4], weight_bindings[8], weight_bindings[9]],
                scratch[5],
                5,
            )?,
            self.create_vision_stack_bind_group(
                &layer_plan,
                gpu,
                6,
                &[current, scratch[5]],
                scratch[6],
                6,
            )?,
            self.create_vision_stack_bind_group(
                &layer_plan,
                gpu,
                7,
                &[scratch[6], weight_bindings[10], weight_bindings[11]],
                scratch[7],
                7,
            )?,
            self.create_vision_stack_bind_group(
                &layer_plan,
                gpu,
                8,
                &[scratch[7], weight_bindings[12], weight_bindings[13]],
                scratch[8],
                8,
            )?,
            self.create_vision_stack_bind_group(&layer_plan, gpu, 9, &[scratch[8]], scratch[9], 9)?,
            self.create_vision_stack_bind_group(
                &layer_plan,
                gpu,
                10,
                &[scratch[9], weight_bindings[14], weight_bindings[15]],
                scratch[10],
                10,
            )?,
            self.create_vision_stack_bind_group(
                &layer_plan,
                gpu,
                11,
                &[scratch[6], scratch[10]],
                next,
                11,
            )?,
        ];
        let spatial_rope_bind_group = spatial_rope_layer_plan
            .as_ref()
            .map(|spatial_plan| {
                self.create_vision_stack_rope_bind_group(spatial_plan, gpu, rope_kernel, query, key)
            })
            .transpose()?;
        let mut encoded = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vision-stack-fp16-qkv-layer-encoder"),
            });
        {
            let mut pass = encoded.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vision-stack-fp16-qkv-layer-pass"),
                timestamp_writes: None,
            });
            let norm1_dispatch = layer_plan.dispatches[0];
            pass.set_pipeline(&gpu.pipelines[&norm1_dispatch.invocation.kernel]);
            pass.set_bind_group(0, &norm1_bind_group, &[]);
            pass.dispatch_workgroups(
                norm1_dispatch.invocation.dispatch[0],
                norm1_dispatch.invocation.dispatch[1],
                norm1_dispatch.invocation.dispatch[2],
            );
            pass.set_pipeline(&gpu.pipelines[&KernelId::VisionQkvFusedF16Weights]);
            pass.set_bind_group(0, &fused_qkv_bind_group, &[]);
            pass.dispatch_workgroups(
                qkv_plan.invocation.dispatch[0],
                qkv_plan.invocation.dispatch[1],
                qkv_plan.invocation.dispatch[2],
            );
            if let (Some(spatial_plan), Some(bind_group)) = (
                spatial_rope_layer_plan.as_ref(),
                spatial_rope_bind_group.as_ref(),
            ) {
                pass.set_pipeline(&gpu.pipelines[&rope_kernel]);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(
                    spatial_plan.rope.invocation.dispatch[0],
                    spatial_plan.rope.invocation.dispatch[1],
                    spatial_plan.rope.invocation.dispatch[2],
                );
            }
            let attention_dispatch = layer_plan.dispatches[4];
            pass.set_pipeline(&gpu.pipelines[&attention_dispatch.invocation.kernel]);
            pass.set_bind_group(0, &attention_bind_group, &[]);
            pass.dispatch_workgroups(
                attention_dispatch.invocation.dispatch[0],
                attention_dispatch.invocation.dispatch[1],
                attention_dispatch.invocation.dispatch[2],
            );
            for (dispatch_index, bind_group) in (5..12).zip(&post_bind_groups) {
                let dispatch = layer_plan.dispatches[dispatch_index];
                pass.set_pipeline(&gpu.pipelines[&dispatch.invocation.kernel]);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(
                    dispatch.invocation.dispatch[0],
                    dispatch.invocation.dispatch[1],
                    dispatch.invocation.dispatch[2],
                );
            }
        }
        if let Some(slot) = checkpoint_slot {
            let offset = u64::try_from(slot)
                .map_err(|_| "vision-stack checkpoint slot overflowed".to_owned())?
                .checked_mul(session.plan.hidden_bytes)
                .ok_or_else(|| "vision-stack checkpoint offset overflowed".to_owned())?;
            encoded.copy_buffer_to_buffer(
                next_buffer,
                0,
                &gpu.readback_buffer,
                offset,
                session.plan.hidden_bytes,
            );
        }
        self.submit_command_buffers([encoded.finish()]);
        gpu.current_main = next_index;
        Ok(())
    }

    async fn finish_vision_stack_sharded(
        &self,
        shard_id: &str,
        bytes: &js_sys::Uint8Array,
    ) -> Result<JsValue, String> {
        let (lease, session) = self
            .vision_stack_session
            .borrow_mut()
            .acquire()
            .map_err(|_error| "vision-stack session is unavailable".to_owned())?;
        let transaction = run_vision_stack_operation_transaction(
            session,
            |shadow| {
                let validated = self.validate_vision_stack_finish(shadow, shard_id, bytes)?;
                let prepared_first_effect = self.prepare_vision_stack_first_error_scope()?;
                Ok((validated, prepared_first_effect))
            },
            async move |shadow, (_validated, prepared_first_effect), effect_boundary| {
                effect_boundary
                    .run_webgpu_effect(async move |post_effect| {
                        self.finish_vision_stack_sharded_once(
                            shadow,
                            bytes,
                            prepared_first_effect,
                            post_effect,
                        )
                        .await
                    })
                    .await
            },
        )
        .await;
        let (outcome, result) = {
            let mut owner = self.vision_stack_session.borrow_mut();
            complete_vision_stack_async_operation(
                &mut owner,
                lease,
                VisionStackAsyncOperation::Finish,
                transaction,
            )
        };
        self.finish_vision_stack_transaction(outcome, result, "finishing")
    }

    async fn finish_vision_stack_sharded_resident(
        &self,
        shard_id: &str,
    ) -> Result<JsValue, String> {
        let (lease, session) = self
            .vision_stack_session
            .borrow_mut()
            .acquire()
            .map_err(|_error| "vision-stack session is unavailable".to_owned())?;
        let transaction = run_vision_stack_operation_transaction(
            session,
            |shadow| {
                let post_weights =
                    self.resolve_vision_stack_resident_post_norm(shadow, shard_id)?;
                let prepared_first_effect = self.prepare_vision_stack_first_error_scope()?;
                Ok((post_weights, prepared_first_effect))
            },
            async move |shadow, (post_weights, prepared_first_effect), effect_boundary| {
                effect_boundary
                    .run_webgpu_effect(async move |post_effect| {
                        self.finish_vision_stack_sharded_resident_once(
                            shadow,
                            post_weights,
                            prepared_first_effect,
                            post_effect,
                        )
                        .await
                    })
                    .await
            },
        )
        .await;
        let (outcome, result) = {
            let mut owner = self.vision_stack_session.borrow_mut();
            complete_vision_stack_async_operation(
                &mut owner,
                lease,
                VisionStackAsyncOperation::Finish,
                transaction,
            )
        };
        self.finish_vision_stack_transaction(outcome, result, "finishing resident")
    }

    async fn finish_vision_stack_sharded_resident_with_projector_f16(
        &self,
        shard_id: &str,
        projector_descriptor_json: &str,
        image_grid_thw_json: &str,
    ) -> Result<JsValue, String> {
        let (descriptor, plan) = projector_f16_execution_plan(
            projector_descriptor_json,
            image_grid_thw_json,
        )?;
        let resident_projector = self
            .projector_f16_resident_weight_cache
            .borrow()
            .as_ref()
            .filter(|resident| resident.key == descriptor.cache_key())
            .cloned()
            .ok_or_else(|| {
                "FP16 projector weights are not resident; prepare them before vision execution"
                    .to_owned()
            })?;
        let (lease, session) = self
            .vision_stack_session
            .borrow_mut()
            .acquire()
            .map_err(|_error| "vision-stack session is unavailable".to_owned())?;
        let transaction = run_vision_stack_operation_transaction(
            session,
            |shadow| {
                let manifest = shadow.protocol.manifest();
                if manifest.activation_storage != DecoderWeightStorage::F16
                    || manifest.hidden_size != descriptor.hidden_size
                    || manifest.tokens != plan.input_tokens
                {
                    return Err(BrowserVisionStackError(format!(
                        "vision/projector FP16 geometry mismatch: vision {}x{} {:?}, projector {}x{}",
                        manifest.tokens,
                        manifest.hidden_size,
                        manifest.activation_storage,
                        plan.input_tokens,
                        descriptor.hidden_size,
                    )));
                }
                let post_weights =
                    self.resolve_vision_stack_resident_post_norm(shadow, shard_id)?;
                let prepared_first_effect = self.prepare_vision_stack_first_error_scope()?;
                Ok((
                    post_weights,
                    plan.clone(),
                    resident_projector.clone(),
                    prepared_first_effect,
                ))
            },
            async move |
                shadow,
                (post_weights, plan, resident_projector, prepared_first_effect),
                effect_boundary,
            | {
                effect_boundary
                    .run_webgpu_effect(async move |post_effect| {
                        self.finish_vision_stack_projector_f16_once(
                            shadow,
                            post_weights,
                            plan,
                            resident_projector,
                            prepared_first_effect,
                            post_effect,
                        )
                        .await
                    })
                    .await
            },
        )
        .await;
        let (outcome, result) = {
            let mut owner = self.vision_stack_session.borrow_mut();
            complete_vision_stack_async_operation(
                &mut owner,
                lease,
                VisionStackAsyncOperation::Finish,
                transaction,
            )
        };
        self.finish_vision_stack_transaction(outcome, result, "finishing resident projector")
    }

    async fn finish_vision_stack_projector_f16_once(
        &self,
        session: &mut BrowserVisionStackSession,
        post_weights: Vec<wgpu::Buffer>,
        projector_plan: ProjectorPlan,
        projector_weights: BrowserProjectorF16ResidentWeights,
        prepared_first_effect: PreparedVisionStackFirstErrorScope<'_>,
        post_effect: &VisionStackPostEffectToken,
    ) -> Result<JsValue, BrowserVisionStackError> {
        let guards = self
            .push_vision_stack_error_scopes(prepared_first_effect, post_effect)
            .await?;
        let before_submissions = self.submissions.get();
        let started_ms = js_sys::Date::now();
        let completed = self
            .complete_browser_error_scoped_operation(guards, async {
                let (output_index, mapped_range) = self
                    .execute_vision_stack_post_norm(session, &post_weights, false)
                    .await
                    .map_err(BrowserVisionStackError::from)?;
                if mapped_range.is_some() {
                    return Err(BrowserVisionStackError(
                        "GPU-only vision post-norm unexpectedly requested host readback"
                            .to_owned(),
                    ));
                }
                let vision_output = {
                    let gpu = session.gpu.as_mut().ok_or_else(|| {
                        BrowserVisionStackError(
                            "vision-stack GPU execution has not been started".to_owned(),
                        )
                    })?;
                    gpu.current_main = output_index;
                    gpu.main_buffers[output_index].clone()
                };
                self.execute_projector_f16_from_buffer(
                    &vision_output,
                    &projector_plan,
                    &projector_weights,
                    started_ms,
                    before_submissions,
                )
                .await
                .map_err(BrowserVisionStackError::from)
            })
            .await?;
        let (operation, captured) = completed;
        if !captured.is_empty() {
            let operation_context = operation
                .as_ref()
                .err()
                .map(|error| format!("; operation also failed: {error}"))
                .unwrap_or_default();
            return Err(format!(
                "resident vision/projector FP16 chain captured WebGPU errors: {captured:?}{operation_context}"
            )
            .into());
        }
        let (checkpoint_bytes, diagnostics) = operation?;
        self.require_no_uncaptured_errors("resident vision/projector FP16 chain")?;
        self.build_projector_f16_result(checkpoint_bytes, &diagnostics)
    }

    fn build_projector_f16_result(
        &self,
        checkpoint_bytes: js_sys::Uint8Array,
        diagnostics: &BrowserProjectorF16Diagnostics,
    ) -> Result<JsValue, BrowserVisionStackError> {
        let result = js_sys::Object::new();
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("checkpoint_bytes"),
            &checkpoint_bytes,
        )
        .map_err(|error| format!("cannot set projector FP16 checkpoint bytes: {error:?}"))?;
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("diagnostics_json"),
            &JsValue::from_str(
                &serde_json::to_string(&diagnostics).map_err(|error| {
                    BrowserVisionStackError(format!(
                        "cannot serialize projector FP16 diagnostics: {error}"
                    ))
                })?,
            ),
        )
        .map_err(|error| format!("cannot set projector FP16 diagnostics: {error:?}"))?;
        Ok(result.into())
    }

    fn resolve_vision_stack_resident_post_norm(
        &self,
        session: &mut BrowserVisionStackSession,
        shard_id: &str,
    ) -> Result<Vec<wgpu::Buffer>, BrowserVisionStackError> {
        let descriptor = session
            .protocol
            .manifest()
            .shards
            .iter()
            .find(|descriptor| descriptor.id == shard_id)
            .cloned()
            .ok_or_else(|| format!("resident vision shard {shard_id:?} is undeclared"))?;
        let observation = VisionStackShardObservation {
            id: descriptor.id,
            bytes: descriptor.bytes,
            blake3: descriptor.blake3,
            all_finite: true,
        };
        let accepted = session
            .protocol
            .accept_execution(&observation)
            .map_err(|error| {
                format!("resident vision-stack execution rejected {shard_id}: {error}")
            })?;
        if accepted.kind != VisionStackShardKind::PostNorm {
            return Err(format!("resident vision-stack expected post-norm, got {shard_id}").into());
        }
        let resident = session.resident_weights.as_ref().ok_or_else(|| {
            BrowserVisionStackError(
                "resident post-norm API requires a resident-weight session".to_owned(),
            )
        })?;
        if resident.disposition != BrowserVisionStackWeightResidency::Ready {
            return Err(
                "resident post-norm API cannot read an incomplete weight cache"
                    .to_owned()
                    .into(),
            );
        }
        let layer_count = usize::try_from(session.plan.layer_count)
            .map_err(|_| "resident vision layer count does not fit usize".to_owned())?;
        let cache = resident.cache.borrow();
        if !cache.is_ready_for(&resident.key, layer_count) {
            return Err(
                "resident vision-weight cache identity changed during execution"
                    .to_owned()
                    .into(),
            );
        }
        cache
            .clone_post_norm()
            .map_err(|error| format!("cannot resolve resident post-norm: {error}").into())
    }

    fn validate_vision_stack_finish(
        &self,
        session: &mut BrowserVisionStackSession,
        shard_id: &str,
        bytes: &js_sys::Uint8Array,
    ) -> Result<VisionStackShardKind, BrowserVisionStackError> {
        if session.resident_weights.as_ref().is_some_and(|resident| {
            resident.disposition == BrowserVisionStackWeightResidency::Ready
        }) {
            return Err(
                "resident vision weights are already complete; use the payload-free resident finish API"
                    .to_owned()
                    .into(),
            );
        }
        let observation = inspect_js_vision_stack_shard(
            session.protocol.manifest(),
            &session.weight_plan,
            shard_id,
            bytes,
        )?;
        let accepted = session
            .protocol
            .accept_execution(&observation)
            .map_err(|error| {
                let mut message = "vision-stack execution rejected ".to_owned();
                message.push_str(shard_id);
                message.push_str(": ");
                message.push_str(&error.to_string());
                message
            })?;
        if accepted.kind != VisionStackShardKind::PostNorm {
            let mut message = "vision-stack expected post-norm, got ".to_owned();
            message.push_str(shard_id);
            return Err(message.into());
        }
        Ok(accepted.kind)
    }

    async fn finish_vision_stack_sharded_once(
        &self,
        session: &mut BrowserVisionStackSession,
        bytes: &js_sys::Uint8Array,
        prepared_first_effect: PreparedVisionStackFirstErrorScope<'_>,
        post_effect: &VisionStackPostEffectToken,
    ) -> Result<JsValue, BrowserVisionStackError> {
        let guards = self
            .push_vision_stack_error_scopes(prepared_first_effect, post_effect)
            .await?;
        let completed = self
            .complete_browser_error_scoped_operation(guards, async {
                let post_weights = self
                    .create_vision_stack_post_norm_weights(session, bytes)
                    .map_err(BrowserVisionStackError::from)?;
                let (output_index, mapped_range) = self
                    .execute_vision_stack_post_norm(session, &post_weights, true)
                    .await
                    .map_err(BrowserVisionStackError::from)?;
                let mapped_range = mapped_range.ok_or_else(|| {
                    BrowserVisionStackError(
                        "vision-stack post-norm omitted its requested readback".to_owned(),
                    )
                })?;
                let legacy_readback_bytes = session
                    .memory_hardening
                    .as_ref()
                    .map_or(session.plan.readback_bytes, |plan| {
                        plan.physical_readback_bytes()
                    });
                let legacy_readback_len = usize::try_from(legacy_readback_bytes).map_err(|_| {
                    BrowserVisionStackError(
                        "vision-stack legacy readback length is too large".to_owned(),
                    )
                })?;
                let qkv_evidence_plan = session.qkv_execution_evidence_plan.as_ref();
                let gpu = session.gpu.as_mut().ok_or_else(|| {
                    BrowserVisionStackError(
                        "vision-stack GPU execution has not been started".to_owned(),
                    )
                })?;
                let map_result = map_read(&gpu.readback_buffer, mapped_range.clone()).await;
                let readback = crate::with_vision_stack_mapped_readback(
                    map_result,
                    || gpu.readback_buffer.unmap(),
                    || {
                        let mapped = gpu.readback_buffer.get_mapped_range(mapped_range).map_err(
                            |error| format!("cannot view mapped vision-stack output: {error}"),
                        )?;
                        let legacy_mapped = mapped.get(..legacy_readback_len).ok_or_else(|| {
                            "mapped vision-stack output is shorter than its legacy region"
                                .to_owned()
                        })?;
                        let checkpoint_bytes = match session.memory_hardening.as_ref() {
                            Some(plan) => plan
                                .verify_and_split_readback(legacy_mapped)
                                .map(js_sys::Uint8Array::from)?,
                            None => js_sys::Uint8Array::from(legacy_mapped),
                        };
                        let canary_results = verify_mapped_qkv_canaries(
                            qkv_evidence_plan,
                            legacy_readback_len,
                            &mapped,
                        )
                        .map_err(|error| error.to_string());
                        drop(mapped);
                        canary_results.map(|results| (checkpoint_bytes, results))
                    },
                );
                readback
                    .inspect(|_| {
                        gpu.current_main = output_index;
                    })
                    .map(|(checkpoint_bytes, canary_results)| {
                        (checkpoint_bytes, canary_results, post_weights)
                    })
                    .map_err(BrowserVisionStackError::from)
            })
            .await?;
        let (readback, captured) = completed;
        if !captured.is_empty() {
            let operation_context = readback
                .as_ref()
                .err()
                .map(|error| format!("; operation also failed: {error}"))
                .unwrap_or_default();
            return Err(format!(
                "vision-stack post-norm captured WebGPU errors: {captured:?}{operation_context}"
            )
            .into());
        }
        let (checkpoint_bytes, canary_results, post_weights) = readback?;
        self.require_no_uncaptured_errors("vision-stack post-norm")?;
        self.commit_vision_stack_resident_post_norm(session, &post_weights)?;
        let gpu = session
            .gpu
            .as_ref()
            .ok_or_else(|| "vision-stack GPU execution has not been started".to_owned())?;
        let queue_wall_time_ns =
            (((js_sys::Date::now() - gpu.started_ms).max(0.000_001)) * 1_000_000.0).round() as u64;
        let weight_buffer_count = session
            .plan
            .layer_count
            .checked_mul(16)
            .and_then(|count| count.checked_add(2))
            .ok_or_else(|| "vision-stack weight count overflowed".to_owned())?;
        let submission_count = self
            .submissions
            .get()
            .checked_sub(session.before_submissions)
            .ok_or_else(|| "vision-stack submission counter regressed".to_owned())?;
        let planned_submission_count = u64::from(session.plan.submission_count);
        if submission_count != planned_submission_count {
            return Err(format!(
                "vision-stack observed {submission_count} submissions, expected {planned_submission_count}"
            )
            .into());
        }
        let buffer_allocation_count = self
            .buffer_allocations
            .get()
            .checked_sub(session.before_buffer_allocations)
            .ok_or_else(|| "vision-stack buffer-allocation counter regressed".to_owned())?;
        let legacy_diagnostics = crate::build_vision_stack_legacy_diagnostics_record(
            &session.plan,
            session.activation_strategy,
            session.activation_layout.as_ref(),
            session.memory_hardening.as_ref(),
            session.storage_alignment,
            &gpu.shader_blake3,
            queue_wall_time_ns.max(1),
            buffer_allocation_count,
            weight_buffer_count,
        )?;
        let diagnostics_json =
            vision_stack_qkv_diagnostics_json(&legacy_diagnostics, session, &canary_results)?;
        let result = js_sys::Object::new();
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("checkpoint_bytes"),
            &checkpoint_bytes,
        )
        .map_err(|error| format!("cannot set vision-stack checkpoint bytes: {error:?}"))?;
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("diagnostics_json"),
            &JsValue::from_str(&diagnostics_json),
        )
        .map_err(|error| format!("cannot set vision-stack diagnostics: {error:?}"))?;
        Ok(result.into())
    }

    async fn finish_vision_stack_sharded_resident_once(
        &self,
        session: &mut BrowserVisionStackSession,
        post_weights: Vec<wgpu::Buffer>,
        prepared_first_effect: PreparedVisionStackFirstErrorScope<'_>,
        post_effect: &VisionStackPostEffectToken,
    ) -> Result<JsValue, BrowserVisionStackError> {
        let guards = self
            .push_vision_stack_error_scopes(prepared_first_effect, post_effect)
            .await?;
        let completed = self
            .complete_browser_error_scoped_operation(guards, async {
                self.execute_and_read_vision_stack_post_norm(session, &post_weights)
                    .await
            })
            .await?;
        let (readback, captured) = completed;
        if !captured.is_empty() {
            let operation_context = readback
                .as_ref()
                .err()
                .map(|error| format!("; operation also failed: {error}"))
                .unwrap_or_default();
            return Err(format!(
                "resident vision-stack post-norm captured WebGPU errors: {captured:?}{operation_context}"
            )
            .into());
        }
        let (checkpoint_bytes, canary_results) = readback?;
        self.require_no_uncaptured_errors("resident vision-stack post-norm")?;
        self.build_vision_stack_finish_result(session, checkpoint_bytes, canary_results)
    }

    fn create_vision_stack_post_norm_weights(
        &self,
        session: &BrowserVisionStackSession,
        bytes: &js_sys::Uint8Array,
    ) -> Result<Vec<wgpu::Buffer>, String> {
        let hidden_bytes = session.plan.hidden_bytes;
        let tensor_bytes = hidden_bytes / u64::from(session.protocol.manifest().tokens);
        [
            VisionStackTensorRange {
                offset: 0,
                bytes: tensor_bytes,
            },
            VisionStackTensorRange {
                offset: tensor_bytes,
                bytes: tensor_bytes,
            },
        ]
        .iter()
        .enumerate()
        .map(|(index, range)| {
            self.create_uploaded_js_buffer(
                &format!("vision-stack-post-norm-weight-{index}"),
                bytes,
                *range,
            )
        })
        .collect()
    }

    fn commit_vision_stack_resident_post_norm(
        &self,
        session: &mut BrowserVisionStackSession,
        post_weights: &[wgpu::Buffer],
    ) -> Result<(), BrowserVisionStackError> {
        let Some(resident) = session.resident_weights.as_mut() else {
            return Ok(());
        };
        if resident.disposition != BrowserVisionStackWeightResidency::Cold {
            return Err(
                "authenticated post-norm payload cannot overwrite ready resident weights"
                    .to_owned()
                    .into(),
            );
        }
        let layer_count = usize::try_from(session.plan.layer_count)
            .map_err(|_| "resident vision layer count does not fit usize".to_owned())?;
        if !resident
            .cache
            .borrow()
            .is_prepared_for(&resident.key, layer_count)
        {
            return Err(
                "resident vision-weight cache identity changed before post-norm commit"
                    .to_owned()
                    .into(),
            );
        }
        resident
            .cache
            .borrow_mut()
            .store_post_norm(post_weights.to_vec())
            .map_err(|error| format!("cannot commit resident post-norm: {error}"))?;
        resident.disposition = BrowserVisionStackWeightResidency::Ready;
        Ok(())
    }

    async fn execute_and_read_vision_stack_post_norm(
        &self,
        session: &mut BrowserVisionStackSession,
        post_weights: &[wgpu::Buffer],
    ) -> Result<(js_sys::Uint8Array, Vec<bool>), BrowserVisionStackError> {
        let (output_index, mapped_range) = self
            .execute_vision_stack_post_norm(session, post_weights, true)
            .await
            .map_err(BrowserVisionStackError::from)?;
        let mapped_range = mapped_range.ok_or_else(|| {
            BrowserVisionStackError(
                "vision-stack post-norm omitted its requested readback".to_owned(),
            )
        })?;
        let legacy_readback_bytes = session
            .memory_hardening
            .as_ref()
            .map_or(session.plan.readback_bytes, |plan| {
                plan.physical_readback_bytes()
            });
        let legacy_readback_len = usize::try_from(legacy_readback_bytes).map_err(|_| {
            BrowserVisionStackError("vision-stack legacy readback length is too large".to_owned())
        })?;
        let qkv_evidence_plan = session.qkv_execution_evidence_plan.as_ref();
        let gpu = session.gpu.as_mut().ok_or_else(|| {
            BrowserVisionStackError("vision-stack GPU execution has not been started".to_owned())
        })?;
        let map_result = map_read(&gpu.readback_buffer, mapped_range.clone()).await;
        let readback = crate::with_vision_stack_mapped_readback(
            map_result,
            || gpu.readback_buffer.unmap(),
            || {
                let mapped = gpu
                    .readback_buffer
                    .get_mapped_range(mapped_range)
                    .map_err(|error| format!("cannot view mapped vision-stack output: {error}"))?;
                let legacy_mapped = mapped.get(..legacy_readback_len).ok_or_else(|| {
                    "mapped vision-stack output is shorter than its legacy region".to_owned()
                })?;
                let checkpoint_bytes = match session.memory_hardening.as_ref() {
                    Some(plan) => plan
                        .verify_and_split_readback(legacy_mapped)
                        .map(js_sys::Uint8Array::from)?,
                    None => js_sys::Uint8Array::from(legacy_mapped),
                };
                let canary_results =
                    verify_mapped_qkv_canaries(qkv_evidence_plan, legacy_readback_len, &mapped)
                        .map_err(|error| error.to_string());
                drop(mapped);
                canary_results.map(|results| (checkpoint_bytes, results))
            },
        );
        readback
            .inspect(|_| {
                gpu.current_main = output_index;
            })
            .map_err(BrowserVisionStackError::from)
    }

    fn build_vision_stack_finish_result(
        &self,
        session: &BrowserVisionStackSession,
        checkpoint_bytes: js_sys::Uint8Array,
        canary_results: Vec<bool>,
    ) -> Result<JsValue, BrowserVisionStackError> {
        let gpu = session
            .gpu
            .as_ref()
            .ok_or_else(|| "vision-stack GPU execution has not been started".to_owned())?;
        let queue_wall_time_ns =
            (((js_sys::Date::now() - gpu.started_ms).max(0.000_001)) * 1_000_000.0).round() as u64;
        let weight_buffer_count = session
            .plan
            .layer_count
            .checked_mul(16)
            .and_then(|count| count.checked_add(2))
            .ok_or_else(|| "vision-stack weight count overflowed".to_owned())?;
        let submission_count = self
            .submissions
            .get()
            .checked_sub(session.before_submissions)
            .ok_or_else(|| "vision-stack submission counter regressed".to_owned())?;
        let planned_submission_count = u64::from(session.plan.submission_count);
        if submission_count != planned_submission_count {
            return Err(format!(
                "vision-stack observed {submission_count} submissions, expected {planned_submission_count}"
            )
            .into());
        }
        let buffer_allocation_count = self
            .buffer_allocations
            .get()
            .checked_sub(session.before_buffer_allocations)
            .ok_or_else(|| "vision-stack buffer-allocation counter regressed".to_owned())?;
        let legacy_diagnostics = crate::build_vision_stack_legacy_diagnostics_record(
            &session.plan,
            session.activation_strategy,
            session.activation_layout.as_ref(),
            session.memory_hardening.as_ref(),
            session.storage_alignment,
            &gpu.shader_blake3,
            queue_wall_time_ns.max(1),
            buffer_allocation_count,
            weight_buffer_count,
        )?;
        let diagnostics_json =
            vision_stack_qkv_diagnostics_json(&legacy_diagnostics, session, &canary_results)?;
        let result = js_sys::Object::new();
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("checkpoint_bytes"),
            &checkpoint_bytes,
        )
        .map_err(|error| format!("cannot set vision-stack checkpoint bytes: {error:?}"))?;
        js_sys::Reflect::set(
            &result,
            &JsValue::from_str("diagnostics_json"),
            &JsValue::from_str(&diagnostics_json),
        )
        .map_err(|error| format!("cannot set vision-stack diagnostics: {error:?}"))?;
        Ok(result.into())
    }

    async fn execute_vision_stack_post_norm(
        &self,
        session: &mut BrowserVisionStackSession,
        post_weights: &[wgpu::Buffer],
        copy_readback: bool,
    ) -> Result<(usize, Option<std::ops::Range<u64>>), String> {
        if post_weights.len() != 2 {
            return Err(format!(
                "vision-stack post-norm requires two weight buffers, got {}",
                post_weights.len()
            ));
        }
        let hidden_bytes = session.plan.hidden_bytes;
        let layer_plan = session.layer_plan;
        let gpu = session
            .gpu
            .as_mut()
            .ok_or_else(|| "vision-stack GPU execution has not been started".to_owned())?;
        let current = &gpu.main_buffers[gpu.current_main];
        let output_index = 1 - gpu.current_main;
        let output = &gpu.main_buffers[output_index];
        let post_weight_bindings = post_weights
            .iter()
            .map(VisionStackBufferBinding::whole)
            .collect::<Vec<_>>();
        let bind_group = self.create_vision_stack_bind_group(
            &layer_plan,
            gpu,
            0,
            &[
                VisionStackBufferBinding::whole(current),
                post_weight_bindings[0],
                post_weight_bindings[1],
            ],
            VisionStackBufferBinding::whole(output),
            12,
        )?;
        let dispatch = layer_plan.dispatches[0];
        let encoder = RefCell::new(self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("vision-stack-post-norm-encoder"),
            },
        ));
        {
            let mut encoded = encoder.borrow_mut();
            let mut pass = encoded.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vision-stack-post-norm-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.pipelines[&dispatch.invocation.kernel]);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                dispatch.invocation.dispatch[0],
                dispatch.invocation.dispatch[1],
                dispatch.invocation.dispatch[2],
            );
        }
        if copy_readback {
            let final_slot = u64::try_from(session.protocol.manifest().checkpoint_layers.len())
                .map_err(|_| "vision-stack final checkpoint slot overflowed".to_owned())?;
            let final_offset = final_slot
                .checked_mul(hidden_bytes)
                .ok_or_else(|| "vision-stack final checkpoint offset overflowed".to_owned())?;
            encoder.borrow_mut().copy_buffer_to_buffer(
                output,
                0,
                &gpu.readback_buffer,
                final_offset,
                hidden_bytes,
            );
            if let Some(plan) = session.memory_hardening.as_ref() {
                let scratch_arena = match &gpu.scratch {
                    BrowserVisionStackScratch::Static { arena, .. } => arena,
                    BrowserVisionStackScratch::Separate(_) => {
                        return Err(
                            "vision-stack memory hardening requires the static scratch arena"
                                .to_owned(),
                        );
                    }
                };
                encoder.borrow_mut().copy_buffer_to_buffer(
                    scratch_arena,
                    0,
                    &gpu.readback_buffer,
                    plan.readback_prefix_canary_offset(),
                    plan.guard_bytes(),
                );
                encoder.borrow_mut().copy_buffer_to_buffer(
                    scratch_arena,
                    plan.scratch_suffix_offset(),
                    &gpu.readback_buffer,
                    plan.readback_suffix_canary_offset(),
                    plan.guard_bytes(),
                );
            }
        } else if session.qkv_physical_commands.is_some() {
            return Err(
                "GPU-only vision/projector chaining is unavailable with Q/K/V evidence readback"
                    .to_owned(),
            );
        }
        let mapped_range = RefCell::new(copy_readback.then(|| 0..gpu.readback_buffer.size()));
        if let Some(qkv_physical_commands) = session.qkv_physical_commands.as_ref() {
            mapped_range.replace(None);
            let placeholder = VisionStackBufferBinding::whole(current);
            let context = BrowserVisionQkvLayerResolutionContext {
                device: &self.device,
                pipelines: &gpu.pipelines,
                buffers: gpu.qkv_physical_storage.buffers.clone(),
                encoder: &encoder,
                norm1_output: placeholder,
                query_weight: post_weight_bindings[0],
                query_bias: post_weight_bindings[1],
                key_weight: placeholder,
                key_bias: placeholder,
                value_weight: placeholder,
                value_bias: placeholder,
                cu_seqlens: VisionStackBufferBinding::whole(&gpu.boundary_buffer),
                attention_output: VisionStackBufferBinding::whole(output),
                uniform_buffer: &gpu.uniform_buffer,
                uniform_stride: gpu.uniform_stride,
                mapped_range: &mapped_range,
            };
            let mut qkv_bind_groups = BrowserVisionQkvLayerBindGroups::new();
            apply_vision_qkv_web_finish_commands(
                BrowserVisionQkvAllocationAuthority {
                    device: &self.device,
                    queue: &self.queue,
                    buffer_allocations: &self.buffer_allocations,
                },
                &context,
                &mut gpu.qkv_physical_storage,
                &mut qkv_bind_groups,
                qkv_physical_commands,
            )
            .map_err(|error| format!("cannot apply typed Q/K/V finish commands: {error:?}"))?;
            drop(qkv_bind_groups);
        }
        let mapped_range = mapped_range.into_inner();
        if copy_readback && mapped_range.is_none() {
            return Err("typed Q/K/V finish commands omitted MapRange".to_owned());
        }
        self.submit_command_buffers([encoder.into_inner().finish()]);
        Ok((output_index, mapped_range))
    }

    fn create_vision_stack_bind_group(
        &self,
        layer_plan: &VisionEncoderLayerPlan,
        gpu: &BrowserVisionStackGpuState,
        dispatch_index: usize,
        inputs: &[VisionStackBufferBinding<'_>],
        output: VisionStackBufferBinding<'_>,
        uniform_slot: usize,
    ) -> Result<wgpu::BindGroup, String> {
        let dispatch = layer_plan
            .dispatches
            .get(dispatch_index)
            .ok_or_else(|| format!("vision-stack dispatch {dispatch_index} is missing"))?;
        let pipeline = &gpu.pipelines[&dispatch.invocation.kernel];
        let layout = pipeline.get_bind_group_layout(0);
        let mut entries = Vec::with_capacity(inputs.len() + 2);
        for (binding, input) in inputs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: input.resource(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: output.resource(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32 + 1,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &gpu.uniform_buffer,
                offset: uniform_slot as u64 * gpu.uniform_stride,
                size: wgpu::BufferSize::new(VISION_LAYER_UNIFORM_BYTES),
            }),
        });
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vision-stack-bind-group"),
            layout: &layout,
            entries: &entries,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn create_vision_stack_fp16_qkv_bind_group(
        &self,
        gpu: &BrowserVisionStackGpuState,
        input: VisionStackBufferBinding<'_>,
        query_weight: VisionStackBufferBinding<'_>,
        query_bias: VisionStackBufferBinding<'_>,
        key_weight: VisionStackBufferBinding<'_>,
        key_bias: VisionStackBufferBinding<'_>,
        value_weight: VisionStackBufferBinding<'_>,
        value_bias: VisionStackBufferBinding<'_>,
        output: VisionStackBufferBinding<'_>,
    ) -> Result<wgpu::BindGroup, String> {
        let pipeline = gpu
            .pipelines
            .get(&KernelId::VisionQkvFusedF16Weights)
            .ok_or_else(|| "tiled FP16 Q/K/V pipeline is missing".to_owned())?;
        let layout = pipeline.get_bind_group_layout(0);
        let resources = [
            input,
            query_weight,
            query_bias,
            key_weight,
            key_bias,
            value_weight,
            value_bias,
            output,
        ];
        let mut entries = resources
            .into_iter()
            .enumerate()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: resource.resource(),
            })
            .collect::<Vec<_>>();
        entries.push(wgpu::BindGroupEntry {
            binding: 8,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &gpu.uniform_buffer,
                offset: gpu.uniform_stride,
                size: wgpu::BufferSize::new(VISION_LAYER_UNIFORM_BYTES),
            }),
        });
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vision-stack-fp16-qkv-bind-group"),
            layout: &layout,
            entries: &entries,
        }))
    }

    fn create_vision_stack_rope_bind_group(
        &self,
        spatial_plan: &VisionEncoderLayerSpatial2dPlan,
        gpu: &BrowserVisionStackGpuState,
        rope_kernel: KernelId,
        query: VisionStackBufferBinding<'_>,
        key: VisionStackBufferBinding<'_>,
    ) -> Result<wgpu::BindGroup, String> {
        let query_bytes = spatial_plan.base.dispatches[1].invocation.output_bytes;
        let key_bytes = spatial_plan.base.dispatches[2].invocation.output_bytes;
        if query.bytes != query_bytes || key.bytes != key_bytes {
            return Err(format!(
                "vision-stack spatial RoPE Q/K bytes drifted: expected {query_bytes}/{key_bytes}, got {}/{}",
                query.bytes, key.bytes
            ));
        }
        let rope_gpu = gpu
            .spatial_rope
            .as_ref()
            .ok_or_else(|| "vision-stack spatial RoPE GPU buffers are missing".to_owned())?;
        if rope_gpu.cos_buffer.size() != spatial_plan.rope.table_bytes
            || rope_gpu.sin_buffer.size() != spatial_plan.rope.table_bytes
        {
            return Err("vision-stack spatial RoPE table buffer size drifted".to_owned());
        }
        let pipeline = gpu
            .pipelines
            .get(&rope_kernel)
            .ok_or_else(|| "vision-stack spatial RoPE pipeline is missing".to_owned())?;
        let layout = pipeline.get_bind_group_layout(0);
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vision-stack-spatial-rope-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query.resource(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: key.resource(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rope_gpu.cos_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: rope_gpu.sin_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &rope_gpu.uniform_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(VISION_LAYER_UNIFORM_BYTES),
                    }),
                },
            ],
        }))
    }

    fn create_uploaded_js_buffer(
        &self,
        label: &str,
        bytes: &js_sys::Uint8Array,
        range: VisionStackTensorRange,
    ) -> Result<wgpu::Buffer, String> {
        let buffer = self.create_runtime_buffer(
            label,
            range.bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        upload_js_range(&self.queue, &buffer, bytes, range)?;
        Ok(buffer)
    }

    fn validate_vision_stack_activation_layout(
        &self,
        plan: &VisionStackShardPlan,
        layer_plan: &VisionEncoderLayerPlan,
        layout: &VisionStackActivationLayout,
        storage_alignment: u64,
    ) -> Result<(), String> {
        if storage_alignment == 0 {
            return Err("vision-stack storage offset alignment is zero".to_owned());
        }
        if layout.physical_buffer_count != 3 {
            return Err(format!(
                "vision-stack static layout has {} physical buffers, expected 3",
                layout.physical_buffer_count
            ));
        }
        if layout.scratch_arena_bytes == 0 {
            return Err("vision-stack static scratch arena is empty".to_owned());
        }
        if layout.scratch_arena_bytes > self.capabilities.limits.max_buffer_size {
            return Err(format!(
                "vision-stack scratch arena {} exceeds browser max_buffer_size {}",
                layout.scratch_arena_bytes, self.capabilities.limits.max_buffer_size
            ));
        }
        if !layout.scratch_arena_bytes.is_multiple_of(storage_alignment) {
            return Err(format!(
                "vision-stack scratch arena {} is not aligned to {storage_alignment}",
                layout.scratch_arena_bytes
            ));
        }
        let expected_main_bytes = plan
            .hidden_bytes
            .checked_mul(2)
            .ok_or_else(|| "vision-stack two main activation buffers overflowed".to_owned())?;
        if layout.main_buffers_bytes != expected_main_bytes {
            return Err(format!(
                "vision-stack main activation bytes {} do not match expected {expected_main_bytes}",
                layout.main_buffers_bytes
            ));
        }
        let expected_total = layout
            .scratch_arena_bytes
            .checked_add(layout.main_buffers_bytes)
            .ok_or_else(|| "vision-stack total activation bytes overflowed".to_owned())?;
        if layout.total_activation_bytes != expected_total {
            return Err(format!(
                "vision-stack total activation bytes {} do not match expected {expected_total}",
                layout.total_activation_bytes
            ));
        }
        if layout.scratch_allocations.len() != 11 {
            return Err(format!(
                "vision-stack static layout has {} scratch slices, expected 11",
                layout.scratch_allocations.len()
            ));
        }
        for (index, (allocation, dispatch)) in layout
            .scratch_allocations
            .iter()
            .zip(layer_plan.dispatches[..11].iter())
            .enumerate()
        {
            if allocation.size == 0 {
                return Err(format!("vision-stack scratch slice {index} is empty"));
            }
            if allocation.stage != dispatch.stage
                || allocation.size != dispatch.invocation.output_bytes
            {
                return Err(format!(
                    "vision-stack scratch slice {index} does not match dispatch stage {}",
                    dispatch.stage.as_str()
                ));
            }
            let expected_alignment = storage_alignment.max(4);
            if allocation.alignment != expected_alignment
                || allocation.offset % storage_alignment != 0
                || allocation.offset % allocation.alignment != 0
            {
                return Err(format!(
                    "vision-stack scratch slice {index} offset/alignment {}/{} does not respect adapter alignment {storage_alignment}",
                    allocation.offset, allocation.alignment
                ));
            }
            let end = allocation
                .offset
                .checked_add(allocation.size)
                .ok_or_else(|| format!("vision-stack scratch slice {index} end overflowed"))?;
            if end > layout.scratch_arena_bytes {
                return Err(format!(
                    "vision-stack scratch slice {index} end {end} exceeds arena {}",
                    layout.scratch_arena_bytes
                ));
            }
            if allocation.size > self.capabilities.limits.max_storage_buffer_binding_size {
                return Err(format!(
                    "vision-stack scratch slice {index} size {} exceeds browser max_storage_buffer_binding_size {}",
                    allocation.size, self.capabilities.limits.max_storage_buffer_binding_size
                ));
            }
        }
        Ok(())
    }

    fn validate_vision_stack_memory_hardening_capabilities(
        &self,
        plan: &VisionStackMemoryHardeningPlan,
    ) -> Result<(), String> {
        let maximum = self.capabilities.limits.max_buffer_size;
        if plan.physical_scratch_bytes() > maximum {
            return Err(format!(
                "vision-stack physical hardened scratch {} exceeds browser max_buffer_size {maximum}",
                plan.physical_scratch_bytes()
            ));
        }
        if plan.physical_readback_bytes() > maximum {
            return Err(format!(
                "vision-stack physical hardened readback {} exceeds browser max_buffer_size {maximum}",
                plan.physical_readback_bytes()
            ));
        }
        Ok(())
    }

    fn validate_vision_stack_capabilities(
        &self,
        manifest: &VisionStackShardManifest,
        plan: &VisionStackShardPlan,
        layer_plan: &VisionEncoderLayerPlan,
        weight_plan: &BrowserVisionStackLayerWeightPlan,
    ) -> Result<(), String> {
        let maximum_dispatch = self
            .capabilities
            .limits
            .max_compute_workgroups_per_dimension;
        for dispatch in layer_plan.dispatches {
            if dispatch
                .invocation
                .dispatch
                .iter()
                .any(|dimension| *dimension > maximum_dispatch)
            {
                return Err(format!(
                    "{} dispatch {:?} exceeds browser adapter limit {maximum_dispatch}",
                    dispatch.stage.as_str(),
                    dispatch.invocation.dispatch
                ));
            }
            self.validate_storage_buffer_bytes(
                dispatch.stage.as_str(),
                dispatch.invocation.output_bytes,
            )?;
        }
        for (index, range) in weight_plan.ranges.iter().enumerate() {
            self.validate_storage_buffer_bytes(
                &format!("vision-stack-weight-{index}"),
                range.bytes,
            )?;
        }
        self.validate_storage_buffer_bytes("vision-stack input", plan.hidden_bytes)?;
        let boundary_bytes = u64::try_from(manifest.cu_seqlens.len())
            .map_err(|_| "vision-stack boundary length overflowed".to_owned())?
            .checked_mul(4)
            .ok_or_else(|| "vision-stack boundary byte length overflowed".to_owned())?;
        self.validate_storage_buffer_bytes("vision-stack boundaries", boundary_bytes)?;
        if plan.readback_bytes > self.capabilities.limits.max_buffer_size {
            return Err(format!(
                "vision-stack readback {} exceeds browser max_buffer_size {}",
                plan.readback_bytes, self.capabilities.limits.max_buffer_size
            ));
        }
        Ok(())
    }

    fn finish_vision_stack_transaction<T>(
        &self,
        outcome: CompletionOutcome,
        result: Result<T, BrowserVisionStackError>,
        activity: &str,
    ) -> Result<T, String> {
        let _ = activity;
        let outcome = coordinate_vision_stack_completion_busy(&self.execution_busy, outcome);
        match outcome {
            CompletionOutcome::Restored | CompletionOutcome::Finished => {
                result.map_err(|error| error.to_string())
            }
            CompletionOutcome::Cancelled => Err("vision-stack execution was aborted".to_owned()),
            CompletionOutcome::Stale => {
                Err("stale vision-stack async completion was ignored".to_owned())
            }
        }
    }

    async fn run_projector_f16_resident_input(
        &self,
        descriptor_json: &str,
        image_grid_thw_json: &str,
        input: &js_sys::Uint8Array,
    ) -> Result<JsValue, String> {
        let (descriptor, plan) =
            projector_f16_execution_plan(descriptor_json, image_grid_thw_json)?;
        let expected_input_bytes = plan.dispatches[0].invocation.output_bytes;
        if u64::from(input.length()) != expected_input_bytes {
            return Err(format!(
                "FP16 projector input has {} bytes, expected {expected_input_bytes}",
                input.length(),
            ));
        }
        let resident_projector = self
            .projector_f16_resident_weight_cache
            .borrow()
            .as_ref()
            .filter(|resident| resident.key == descriptor.cache_key())
            .cloned()
            .ok_or_else(|| {
                "FP16 projector weights are not resident; prepare them before execution"
                    .to_owned()
            })?;
        let guards = self
            .push_browser_error_scopes()
            .await
            .map_err(|error| error.0)?;
        let before_submissions = self.submissions.get();
        let completed = self
            .complete_browser_error_scoped_operation(guards, async {
                let input_buffer = self
                    .create_uploaded_js_buffer(
                        "projector-f16-host-input",
                        input,
                        VisionStackTensorRange {
                            offset: 0,
                            bytes: expected_input_bytes,
                        },
                    )
                    .map_err(BrowserVisionStackError::from)?;
                self.execute_projector_f16_from_buffer(
                    &input_buffer,
                    &plan,
                    &resident_projector,
                    js_sys::Date::now(),
                    before_submissions,
                )
                .await
                .map_err(BrowserVisionStackError::from)
            })
            .await
            .map_err(|error| error.0)?;
        let (operation, captured) = completed;
        if !captured.is_empty() {
            let operation_context = operation
                .as_ref()
                .err()
                .map(|error| format!("; operation also failed: {error}"))
                .unwrap_or_default();
            return Err(format!(
                "resident FP16 projector captured WebGPU errors: {captured:?}{operation_context}"
            ));
        }
        let (checkpoint_bytes, diagnostics) =
            operation.map_err(|error| error.0)?;
        self.require_no_uncaptured_errors("resident FP16 projector")?;
        self.build_projector_f16_result(checkpoint_bytes, &diagnostics)
            .map_err(|error| error.0)
    }

    async fn run_projector_source(
        &self,
        invocation: &OwnedProjectorInvocation,
        readback: ProjectorReadback,
        shader_override: Option<(KernelId, &str)>,
    ) -> Result<BrowserProjectorExecution, String> {
        let borrowed = invocation.borrowed();
        let plan = borrowed
            .plan()
            .map_err(|error| format!("invalid projector invocation: {error}"))?;
        let mut sources = BTreeMap::new();
        for kernel in PROJECTOR_KERNELS {
            let module = pvlc_wgsl::module(kernel)
                .expect("every resident projector kernel must have fixed WGSL");
            let source = shader_override
                .filter(|(overridden, _)| *overridden == kernel)
                .map_or(module.source, |(_, source)| source);
            pvlc_wgsl::validate_source_contract(&module.spec, source).map_err(|error| {
                format!("resident projector shader {kernel} is invalid: {error}")
            })?;
            sources.insert(kernel, source.to_owned());
        }
        self.validate_projector_capabilities(&borrowed, &plan, readback)?;
        self.run_projector_scoped(
            &borrowed,
            plan,
            readback,
            &sources,
            shader_override.map(|(kernel, _)| kernel),
        )
        .await
        .map_err(|error| error.0)
    }

    async fn run_projector_scoped(
        &self,
        invocation: &ProjectorInvocation<'_>,
        plan: ProjectorPlan,
        readback: ProjectorReadback,
        sources: &BTreeMap<KernelId, String>,
        overridden_kernel: Option<KernelId>,
    ) -> Result<BrowserProjectorExecution, BrowserVisionStackError> {
        let guards = self.push_browser_error_scopes().await?;
        let completed = self
            .complete_browser_error_scoped_operation(guards, async {
                self.execute_projector_once(invocation, plan, readback, sources, overridden_kernel)
                    .await
                    .map_err(BrowserVisionStackError::from)
            })
            .await?;
        let (operation, captured) = completed;
        if !captured.is_empty() {
            let operation_context = operation
                .as_ref()
                .err()
                .map(|error| format!("; operation also failed: {error}"))
                .unwrap_or_default();
            return Err(format!(
                "resident projector captured WebGPU errors: {captured:?}{operation_context}"
            )
            .into());
        }
        let execution = operation?;
        self.require_no_uncaptured_errors("resident projector")?;
        Ok(execution)
    }

    fn validate_projector_capabilities(
        &self,
        invocation: &ProjectorInvocation<'_>,
        plan: &ProjectorPlan,
        readback: ProjectorReadback,
    ) -> Result<(), String> {
        let maximum_dispatch = self
            .capabilities
            .limits
            .max_compute_workgroups_per_dimension;
        for dispatch in plan.dispatches {
            if dispatch
                .invocation
                .dispatch
                .iter()
                .any(|dimension| *dimension > maximum_dispatch)
            {
                return Err(format!(
                    "{} dispatch {:?} exceeds browser adapter limit {maximum_dispatch}",
                    dispatch.stage.as_str(),
                    dispatch.invocation.dispatch
                ));
            }
            self.validate_storage_buffer_bytes(
                dispatch.stage.as_str(),
                dispatch.invocation.output_bytes,
            )?;
        }

        let parameters = invocation.parameters;
        for (label, elements) in [
            ("projector-input", invocation.input.len()),
            (
                "projector-source-token-indices",
                plan.source_token_indices.len(),
            ),
            (
                "projector-pre-norm-weight",
                parameters.pre_norm.weight.len(),
            ),
            ("projector-pre-norm-bias", parameters.pre_norm.bias.len()),
            ("projector-linear1-weight", parameters.linear1.weight.len()),
            ("projector-linear1-bias", parameters.linear1.bias.len()),
            ("projector-linear2-weight", parameters.linear2.weight.len()),
            ("projector-linear2-bias", parameters.linear2.bias.len()),
        ] {
            let bytes = u64::try_from(elements)
                .ok()
                .and_then(|elements| elements.checked_mul(4))
                .ok_or_else(|| format!("{label} byte size overflowed"))?;
            self.validate_storage_buffer_bytes(label, bytes)?;
        }
        let uniform_stride =
            vision_layer_uniform_stride(self.device.limits().min_uniform_buffer_offset_alignment);
        let uniform_bytes = uniform_stride
            .checked_mul(plan.dispatches.len() as u64)
            .ok_or_else(|| "projector uniform arena byte size overflowed".to_owned())?;
        if uniform_bytes > self.capabilities.limits.max_buffer_size {
            return Err(format!(
                "projector uniform arena requires {uniform_bytes} bytes but browser max_buffer_size is {}",
                self.capabilities.limits.max_buffer_size
            ));
        }
        let readback_bytes = plan.readback_bytes(readback);
        if readback_bytes > self.capabilities.limits.max_buffer_size {
            return Err(format!(
                "projector readback requires {readback_bytes} bytes but browser max_buffer_size is {}",
                self.capabilities.limits.max_buffer_size
            ));
        }
        Ok(())
    }

    async fn run_vision_layer_source(
        &self,
        invocation: &OwnedVisionEncoderLayerInvocation,
        readback: VisionLayerReadback,
        shader_override: Option<(KernelId, &str)>,
    ) -> Result<BrowserVisionLayerExecution, String> {
        let borrowed = invocation.borrowed();
        let plan = borrowed
            .plan()
            .map_err(|error| format!("invalid vision-layer invocation: {error}"))?;
        let mut sources = BTreeMap::new();
        for kernel in VISION_LAYER_KERNELS {
            let module = pvlc_wgsl::module(kernel)
                .expect("every resident vision-layer kernel must have fixed WGSL");
            let source = shader_override
                .filter(|(overridden, _)| *overridden == kernel)
                .map_or(module.source, |(_, source)| source);
            pvlc_wgsl::validate_source_contract(&module.spec, source).map_err(|error| {
                format!("resident vision-layer shader {kernel} is invalid: {error}")
            })?;
            sources.insert(kernel, source.to_owned());
        }
        self.validate_vision_layer_capabilities(&borrowed, &plan, readback)?;
        self.run_vision_layer_scoped(
            &borrowed,
            plan,
            readback,
            &sources,
            shader_override.map(|(kernel, _)| kernel),
        )
        .await
        .map_err(|error| error.0)
    }

    async fn run_vision_layer_scoped(
        &self,
        invocation: &VisionEncoderLayerInvocation<'_>,
        plan: VisionEncoderLayerPlan,
        readback: VisionLayerReadback,
        sources: &BTreeMap<KernelId, String>,
        overridden_kernel: Option<KernelId>,
    ) -> Result<BrowserVisionLayerExecution, BrowserVisionStackError> {
        let guards = self.push_browser_error_scopes().await?;
        let completed = self
            .complete_browser_error_scoped_operation(guards, async {
                self.execute_vision_layer_once(
                    invocation,
                    plan,
                    readback,
                    sources,
                    overridden_kernel,
                )
                .await
                .map_err(BrowserVisionStackError::from)
            })
            .await?;
        let (operation, captured) = completed;
        if !captured.is_empty() {
            let operation_context = operation
                .as_ref()
                .err()
                .map(|error| format!("; operation also failed: {error}"))
                .unwrap_or_default();
            return Err(format!(
                "resident vision layer captured WebGPU errors: {captured:?}{operation_context}"
            )
            .into());
        }
        let execution = operation?;
        self.require_no_uncaptured_errors("resident vision layer")?;
        Ok(execution)
    }

    fn validate_vision_layer_capabilities(
        &self,
        invocation: &VisionEncoderLayerInvocation<'_>,
        plan: &VisionEncoderLayerPlan,
        readback: VisionLayerReadback,
    ) -> Result<(), String> {
        let maximum_dispatch = self
            .capabilities
            .limits
            .max_compute_workgroups_per_dimension;
        for dispatch in plan.dispatches {
            if dispatch
                .invocation
                .dispatch
                .iter()
                .any(|dimension| *dimension > maximum_dispatch)
            {
                return Err(format!(
                    "{} dispatch {:?} exceeds browser adapter limit {maximum_dispatch}",
                    dispatch.stage.as_str(),
                    dispatch.invocation.dispatch
                ));
            }
            self.validate_storage_buffer_bytes(
                dispatch.stage.as_str(),
                dispatch.invocation.output_bytes,
            )?;
        }

        let parameters = invocation.parameters;
        for (label, elements) in [
            ("input", invocation.input.len()),
            ("norm1-weight", parameters.norm1.weight.len()),
            ("norm1-bias", parameters.norm1.bias.len()),
            ("query-weight", parameters.query.weight.len()),
            ("query-bias", parameters.query.bias.len()),
            ("key-weight", parameters.key.weight.len()),
            ("key-bias", parameters.key.bias.len()),
            ("value-weight", parameters.value.weight.len()),
            ("value-bias", parameters.value.bias.len()),
            (
                "attention-output-weight",
                parameters.attention_output.weight.len(),
            ),
            (
                "attention-output-bias",
                parameters.attention_output.bias.len(),
            ),
            ("norm2-weight", parameters.norm2.weight.len()),
            ("norm2-bias", parameters.norm2.bias.len()),
            ("mlp-fc1-weight", parameters.mlp_fc1.weight.len()),
            ("mlp-fc1-bias", parameters.mlp_fc1.bias.len()),
            ("mlp-fc2-weight", parameters.mlp_fc2.weight.len()),
            ("mlp-fc2-bias", parameters.mlp_fc2.bias.len()),
        ] {
            let bytes = u64::try_from(elements)
                .ok()
                .and_then(|elements| elements.checked_mul(4))
                .ok_or_else(|| format!("{label} byte size overflowed"))?;
            self.validate_storage_buffer_bytes(label, bytes)?;
        }
        self.validate_storage_buffer_bytes(
            "cu-seqlens",
            u64::try_from(invocation.cu_seqlens.len())
                .ok()
                .and_then(|elements| elements.checked_mul(4))
                .ok_or_else(|| "cu-seqlens byte size overflowed".to_owned())?,
        )?;
        let readback_bytes = vision_layer_readback_indices(readback)
            .iter()
            .try_fold(0_u64, |bytes, index| {
                bytes.checked_add(plan.dispatches[*index].invocation.output_bytes)
            })
            .ok_or_else(|| "vision-layer readback byte size overflowed".to_owned())?;
        if readback_bytes > self.capabilities.limits.max_buffer_size {
            return Err(format!(
                "vision-layer readback requires {readback_bytes} bytes but browser max_buffer_size is {}",
                self.capabilities.limits.max_buffer_size
            ));
        }
        Ok(())
    }

    fn validate_storage_buffer_bytes(&self, label: &str, bytes: u64) -> Result<(), String> {
        let limit = self
            .capabilities
            .limits
            .max_storage_buffer_binding_size
            .min(self.capabilities.limits.max_buffer_size);
        if bytes > limit {
            Err(format!(
                "{label} requires {bytes} bytes but browser storage-buffer limit is {limit}"
            ))
        } else {
            Ok(())
        }
    }

    #[allow(clippy::too_many_lines)]
    async fn execute_projector_once(
        &self,
        invocation: &ProjectorInvocation<'_>,
        plan: ProjectorPlan,
        readback: ProjectorReadback,
        sources: &BTreeMap<KernelId, String>,
        overridden_kernel: Option<KernelId>,
    ) -> Result<BrowserProjectorExecution, String> {
        let before_buffers = self.buffer_allocations.get();
        let before_submissions = self.submissions.get();
        let mut pipelines = BTreeMap::new();
        let mut shader_blake3 = BTreeMap::new();
        for kernel in PROJECTOR_KERNELS {
            let source = sources[&kernel].as_str();
            let override_active = overridden_kernel == Some(kernel);
            let label = if override_active {
                format!("pvlc-projector-nonce-{kernel}")
            } else {
                kernel.as_str().to_owned()
            };
            let pipeline =
                self.pipeline(&label, source, "main", (!override_active).then_some(kernel));
            pipelines.insert(kernel, pipeline);
            shader_blake3.insert(kernel, blake3_hex(source));
        }

        let parameters = invocation.parameters;
        let immutable_buffers = [
            self.create_initialized_buffer(
                "projector-input",
                bytemuck::cast_slice(invocation.input),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "projector-source-token-indices",
                bytemuck::cast_slice(&plan.source_token_indices),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "projector-pre-norm-weight",
                bytemuck::cast_slice(parameters.pre_norm.weight),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "projector-pre-norm-bias",
                bytemuck::cast_slice(parameters.pre_norm.bias),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "projector-linear1-weight",
                bytemuck::cast_slice(parameters.linear1.weight),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "projector-linear1-bias",
                bytemuck::cast_slice(parameters.linear1.bias),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "projector-linear2-weight",
                bytemuck::cast_slice(parameters.linear2.weight),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "projector-linear2-bias",
                bytemuck::cast_slice(parameters.linear2.bias),
                wgpu::BufferUsages::STORAGE,
            ),
        ];
        let output_buffers = plan
            .dispatches
            .iter()
            .map(|dispatch| {
                self.create_runtime_buffer(
                    &format!("projector-{}", dispatch.stage.as_str()),
                    dispatch.invocation.output_bytes,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                )
            })
            .collect::<Vec<_>>();

        let uniform_stride =
            vision_layer_uniform_stride(self.device.limits().min_uniform_buffer_offset_alignment);
        let uniform_arena_bytes = uniform_stride
            .checked_mul(plan.dispatches.len() as u64)
            .ok_or_else(|| "projector uniform arena overflowed".to_owned())?;
        let mut uniform_contents = vec![0_u8; uniform_arena_bytes as usize];
        for (index, dispatch) in plan.dispatches.iter().enumerate() {
            let offset = index * uniform_stride as usize;
            let bytes = dispatch
                .uniform_words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            uniform_contents[offset..offset + VISION_LAYER_UNIFORM_BYTES as usize]
                .copy_from_slice(&bytes);
        }
        let uniform_buffer = self.create_initialized_buffer(
            "projector-uniform-arena",
            &uniform_contents,
            wgpu::BufferUsages::UNIFORM,
        );
        let readback_indices = projector_readback_indices(readback);
        let readback_bytes = plan.readback_bytes(readback);
        let readback_buffer = self.create_runtime_buffer(
            "projector-readback",
            readback_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let bind_groups = [
            self.create_staged_bind_group(
                "projector-pre-norm-bind-group",
                &pipelines[&KernelId::LayerNormF32],
                &[
                    &immutable_buffers[0],
                    &immutable_buffers[2],
                    &immutable_buffers[3],
                ],
                &output_buffers[0],
                &uniform_buffer,
                0,
            ),
            self.create_staged_bind_group(
                "projector-merge-bind-group",
                &pipelines[&KernelId::ProjectorMerge2x2F32],
                &[&output_buffers[0], &immutable_buffers[1]],
                &output_buffers[1],
                &uniform_buffer,
                uniform_stride,
            ),
            self.create_staged_bind_group(
                "projector-linear1-bind-group",
                &pipelines[&KernelId::VisionPatchProjectionF32],
                &[
                    &output_buffers[1],
                    &immutable_buffers[4],
                    &immutable_buffers[5],
                ],
                &output_buffers[2],
                &uniform_buffer,
                uniform_stride * 2,
            ),
            self.create_staged_bind_group(
                "projector-activation-bind-group",
                &pipelines[&KernelId::GeluErfF32],
                &[&output_buffers[2]],
                &output_buffers[3],
                &uniform_buffer,
                uniform_stride * 3,
            ),
            self.create_staged_bind_group(
                "projector-linear2-bind-group",
                &pipelines[&KernelId::VisionPatchProjectionF32],
                &[
                    &output_buffers[3],
                    &immutable_buffers[6],
                    &immutable_buffers[7],
                ],
                &output_buffers[4],
                &uniform_buffer,
                uniform_stride * 4,
            ),
        ];

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("projector-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("projector-pass"),
                timestamp_writes: None,
            });
            for (index, dispatch) in plan.dispatches.iter().enumerate() {
                pass.set_pipeline(&pipelines[&dispatch.invocation.kernel]);
                pass.set_bind_group(0, &bind_groups[index], &[]);
                pass.dispatch_workgroups(
                    dispatch.invocation.dispatch[0],
                    dispatch.invocation.dispatch[1],
                    dispatch.invocation.dispatch[2],
                );
            }
        }
        let mut readback_offset = 0_u64;
        for &index in readback_indices {
            let bytes = plan.dispatches[index].invocation.output_bytes;
            encoder.copy_buffer_to_buffer(
                &output_buffers[index],
                0,
                &readback_buffer,
                readback_offset,
                bytes,
            );
            readback_offset += bytes;
        }
        debug_assert_eq!(readback_offset, readback_bytes);
        let started = js_sys::Date::now();
        self.submit_command_buffers([encoder.finish()]);
        map_read(&readback_buffer, 0..readback_buffer.size()).await?;
        let queue_wall_time_ns =
            (((js_sys::Date::now() - started).max(0.000_001)) * 1_000_000.0).round() as u64;
        let readback_elements = usize::try_from(readback_bytes / 4)
            .map_err(|_| "projector readback is too large".to_owned())?;
        let flat_values = read_f32_buffer(&readback_buffer, readback_elements)?;
        let mut checkpoint_spans = Vec::with_capacity(readback_indices.len());
        let mut element_offset = 0_usize;
        for &index in readback_indices {
            let dispatch = plan.dispatches[index];
            checkpoint_spans.push(BrowserProjectorCheckpoint {
                stage: dispatch.stage,
                element_offset,
                elements: dispatch.invocation.output_elements,
            });
            element_offset += dispatch.invocation.output_elements;
        }
        debug_assert_eq!(element_offset, flat_values.len());

        Ok(BrowserProjectorExecution {
            checkpoint_values: flat_values,
            checkpoint_spans,
            diagnostics: BrowserProjectorDiagnostics {
                checked_error_scopes: CHECKED_SCOPES,
                captured_errors: Vec::new(),
                queue_wall_time_ns: queue_wall_time_ns.max(1),
                shader_blake3,
                dispatch_stages: plan.dispatches.map(|dispatch| dispatch.stage),
                submission_count: self.submissions.get() - before_submissions,
                command_buffer_count: 1,
                compute_pass_count: 1,
                dispatch_count: plan.dispatches.len() as u32,
                buffer_allocation_count: self.buffer_allocations.get() - before_buffers,
                readback_buffer_count: 1,
                readback_map_count: 1,
                readback_bytes,
                resident_intermediate_bytes: plan.resident_intermediate_bytes,
                resident_weight_bytes: plan.resident_weight_bytes,
            },
        })
    }

    #[allow(clippy::too_many_lines)]
    async fn execute_projector_f16_from_buffer(
        &self,
        input_buffer: &wgpu::Buffer,
        plan: &ProjectorPlan,
        resident_weights: &BrowserProjectorF16ResidentWeights,
        started_ms: f64,
        before_submissions: u64,
    ) -> Result<
        (js_sys::Uint8Array, BrowserProjectorF16Diagnostics),
        String,
    > {
        if resident_weights.buffers.len() != 6 {
            return Err(format!(
                "resident FP16 projector requires six weight buffers, got {}",
                resident_weights.buffers.len(),
            ));
        }
        let expected_input_bytes = plan.dispatches[0].invocation.output_bytes;
        if input_buffer.size() < expected_input_bytes {
            return Err(format!(
                "vision output buffer has {} bytes, FP16 projector requires {expected_input_bytes}",
                input_buffer.size(),
            ));
        }
        let mut pipelines = BTreeMap::new();
        for kernel in PROJECTOR_F16_KERNELS {
            let module = pvlc_wgsl::module(kernel)
                .expect("every FP16 projector kernel must have fixed WGSL");
            pvlc_wgsl::validate_source_contract(&module.spec, module.source)
                .map_err(|error| format!("FP16 projector shader {kernel} is invalid: {error}"))?;
            pipelines.insert(
                kernel,
                self.pipeline(kernel.as_str(), module.source, "main", Some(kernel)),
            );
        }

        let source_indices = self.create_initialized_buffer(
            "projector-f16-source-token-indices",
            bytemuck::cast_slice(&plan.source_token_indices),
            wgpu::BufferUsages::STORAGE,
        );
        let output_buffers = plan
            .dispatches
            .iter()
            .map(|dispatch| {
                self.create_runtime_buffer(
                    &format!("projector-f16-{}", dispatch.stage.as_str()),
                    dispatch.invocation.output_bytes,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                )
            })
            .collect::<Vec<_>>();

        let uniform_stride =
            vision_layer_uniform_stride(self.device.limits().min_uniform_buffer_offset_alignment);
        let uniform_arena_bytes = uniform_stride
            .checked_mul(plan.dispatches.len() as u64)
            .ok_or_else(|| "FP16 projector uniform arena overflowed".to_owned())?;
        let mut uniform_contents = vec![0_u8; uniform_arena_bytes as usize];
        for (index, dispatch) in plan.dispatches.iter().enumerate() {
            let offset = index * uniform_stride as usize;
            let bytes = dispatch
                .uniform_words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            uniform_contents[offset..offset + VISION_LAYER_UNIFORM_BYTES as usize]
                .copy_from_slice(&bytes);
        }
        let uniform_buffer = self.create_initialized_buffer(
            "projector-f16-uniform-arena",
            &uniform_contents,
            wgpu::BufferUsages::UNIFORM,
        );
        let final_output_bytes = plan.dispatches[4].invocation.output_bytes;
        let readback_buffer = self.create_runtime_buffer(
            "projector-f16-readback",
            final_output_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let weights = &resident_weights.buffers;
        let bind_groups = [
            self.create_staged_bind_group(
                "projector-f16-pre-norm-bind-group",
                &pipelines[&KernelId::LayerNormF16],
                &[input_buffer, &weights[0], &weights[1]],
                &output_buffers[0],
                &uniform_buffer,
                0,
            ),
            self.create_staged_bind_group(
                "projector-f16-merge-bind-group",
                &pipelines[&KernelId::ProjectorMerge2x2F16],
                &[&output_buffers[0], &source_indices],
                &output_buffers[1],
                &uniform_buffer,
                uniform_stride,
            ),
            self.create_staged_bind_group(
                "projector-f16-linear1-bind-group",
                &pipelines[&KernelId::LinearProjectionF16],
                &[&output_buffers[1], &weights[2], &weights[3]],
                &output_buffers[2],
                &uniform_buffer,
                uniform_stride * 2,
            ),
            self.create_staged_bind_group(
                "projector-f16-activation-bind-group",
                &pipelines[&KernelId::GeluErfF16],
                &[&output_buffers[2]],
                &output_buffers[3],
                &uniform_buffer,
                uniform_stride * 3,
            ),
            self.create_staged_bind_group(
                "projector-f16-linear2-bind-group",
                &pipelines[&KernelId::LinearProjectionF16],
                &[&output_buffers[3], &weights[4], &weights[5]],
                &output_buffers[4],
                &uniform_buffer,
                uniform_stride * 4,
            ),
        ];

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("projector-f16-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("projector-f16-pass"),
                timestamp_writes: None,
            });
            for (index, dispatch) in plan.dispatches.iter().enumerate() {
                pass.set_pipeline(&pipelines[&dispatch.invocation.kernel]);
                pass.set_bind_group(0, &bind_groups[index], &[]);
                pass.dispatch_workgroups(
                    dispatch.invocation.dispatch[0],
                    dispatch.invocation.dispatch[1],
                    dispatch.invocation.dispatch[2],
                );
            }
        }
        encoder.copy_buffer_to_buffer(
            &output_buffers[4],
            0,
            &readback_buffer,
            0,
            final_output_bytes,
        );
        self.submit_command_buffers([encoder.finish()]);
        map_read(&readback_buffer, 0..final_output_bytes).await?;
        let mapped = readback_buffer
            .get_mapped_range(0..final_output_bytes)
            .map_err(|error| format!("cannot view mapped FP16 projector output: {error}"))?;
        let checkpoint_bytes = js_sys::Uint8Array::from(&mapped[..]);
        drop(mapped);
        readback_buffer.unmap();
        let queue_wall_time_ns =
            (((js_sys::Date::now() - started_ms).max(0.000_001)) * 1_000_000.0).round() as u64;
        Ok((
            checkpoint_bytes,
            BrowserProjectorF16Diagnostics {
                queue_wall_time_ns: queue_wall_time_ns.max(1),
                output_tokens: plan.output_tokens,
                output_bytes: final_output_bytes,
                submission_count: self.submissions.get().saturating_sub(before_submissions),
                dispatch_count: plan.dispatches.len() as u32 + 1,
                resident_weight_bytes: plan.resident_weight_bytes,
                cpu_bridge_elided: true,
            },
        ))
    }

    #[allow(clippy::too_many_lines)]
    async fn execute_vision_layer_once(
        &self,
        invocation: &VisionEncoderLayerInvocation<'_>,
        plan: VisionEncoderLayerPlan,
        readback: VisionLayerReadback,
        sources: &BTreeMap<KernelId, String>,
        overridden_kernel: Option<KernelId>,
    ) -> Result<BrowserVisionLayerExecution, String> {
        let before_buffers = self.buffer_allocations.get();
        let before_submissions = self.submissions.get();
        let mut pipelines = BTreeMap::new();
        let mut shader_blake3 = BTreeMap::new();
        for kernel in VISION_LAYER_KERNELS {
            let source = sources[&kernel].as_str();
            let override_active = overridden_kernel == Some(kernel);
            let label = if override_active {
                format!("pvlc-vision-layer-nonce-{kernel}")
            } else {
                kernel.as_str().to_owned()
            };
            let pipeline =
                self.pipeline(&label, source, "main", (!override_active).then_some(kernel));
            pipelines.insert(kernel, pipeline);
            shader_blake3.insert(kernel, blake3_hex(source));
        }

        let parameters = invocation.parameters;
        let immutable_buffers = [
            self.create_initialized_buffer(
                "vision-layer-input",
                bytemuck::cast_slice(invocation.input),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-cu-seqlens",
                bytemuck::cast_slice(invocation.cu_seqlens),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-norm1-weight",
                bytemuck::cast_slice(parameters.norm1.weight),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-norm1-bias",
                bytemuck::cast_slice(parameters.norm1.bias),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-query-weight",
                bytemuck::cast_slice(parameters.query.weight),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-query-bias",
                bytemuck::cast_slice(parameters.query.bias),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-key-weight",
                bytemuck::cast_slice(parameters.key.weight),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-key-bias",
                bytemuck::cast_slice(parameters.key.bias),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-value-weight",
                bytemuck::cast_slice(parameters.value.weight),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-value-bias",
                bytemuck::cast_slice(parameters.value.bias),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-attention-output-weight",
                bytemuck::cast_slice(parameters.attention_output.weight),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-attention-output-bias",
                bytemuck::cast_slice(parameters.attention_output.bias),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-norm2-weight",
                bytemuck::cast_slice(parameters.norm2.weight),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-norm2-bias",
                bytemuck::cast_slice(parameters.norm2.bias),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-mlp-fc1-weight",
                bytemuck::cast_slice(parameters.mlp_fc1.weight),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-mlp-fc1-bias",
                bytemuck::cast_slice(parameters.mlp_fc1.bias),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-mlp-fc2-weight",
                bytemuck::cast_slice(parameters.mlp_fc2.weight),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-layer-mlp-fc2-bias",
                bytemuck::cast_slice(parameters.mlp_fc2.bias),
                wgpu::BufferUsages::STORAGE,
            ),
        ];
        let output_buffers = plan
            .dispatches
            .iter()
            .map(|dispatch| {
                self.create_runtime_buffer(
                    &format!("vision-layer-{}", dispatch.stage.as_str()),
                    dispatch.invocation.output_bytes,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                )
            })
            .collect::<Vec<_>>();

        let uniform_stride =
            vision_layer_uniform_stride(self.device.limits().min_uniform_buffer_offset_alignment);
        let uniform_arena_bytes = uniform_stride
            .checked_mul(plan.dispatches.len() as u64)
            .ok_or_else(|| "vision-layer uniform arena overflowed".to_owned())?;
        let mut uniform_contents = vec![0_u8; uniform_arena_bytes as usize];
        for (index, dispatch) in plan.dispatches.iter().enumerate() {
            let offset = index * uniform_stride as usize;
            let bytes = dispatch
                .uniform_words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            uniform_contents[offset..offset + VISION_LAYER_UNIFORM_BYTES as usize]
                .copy_from_slice(&bytes);
        }
        let uniform_buffer = self.create_initialized_buffer(
            "vision-layer-uniform-arena",
            &uniform_contents,
            wgpu::BufferUsages::UNIFORM,
        );
        let readback_indices = vision_layer_readback_indices(readback);
        let readback_bytes = readback_indices.iter().fold(0_u64, |bytes, index| {
            bytes + plan.dispatches[*index].invocation.output_bytes
        });
        let readback_buffer = self.create_runtime_buffer(
            "vision-layer-readback",
            readback_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let mut bind_groups = Vec::with_capacity(plan.dispatches.len());
        bind_groups.push(self.create_vision_layer_bind_group(
            &plan,
            0,
            &pipelines,
            &[
                &immutable_buffers[0],
                &immutable_buffers[2],
                &immutable_buffers[3],
            ],
            &output_buffers[0],
            &uniform_buffer,
            uniform_stride,
        ));
        bind_groups.push(self.create_vision_layer_bind_group(
            &plan,
            1,
            &pipelines,
            &[
                &output_buffers[0],
                &immutable_buffers[4],
                &immutable_buffers[5],
            ],
            &output_buffers[1],
            &uniform_buffer,
            uniform_stride,
        ));
        bind_groups.push(self.create_vision_layer_bind_group(
            &plan,
            2,
            &pipelines,
            &[
                &output_buffers[0],
                &immutable_buffers[6],
                &immutable_buffers[7],
            ],
            &output_buffers[2],
            &uniform_buffer,
            uniform_stride,
        ));
        bind_groups.push(self.create_vision_layer_bind_group(
            &plan,
            3,
            &pipelines,
            &[
                &output_buffers[0],
                &immutable_buffers[8],
                &immutable_buffers[9],
            ],
            &output_buffers[3],
            &uniform_buffer,
            uniform_stride,
        ));
        bind_groups.push(self.create_vision_layer_bind_group(
            &plan,
            4,
            &pipelines,
            &[
                &output_buffers[1],
                &output_buffers[2],
                &output_buffers[3],
                &immutable_buffers[1],
            ],
            &output_buffers[4],
            &uniform_buffer,
            uniform_stride,
        ));
        bind_groups.push(self.create_vision_layer_bind_group(
            &plan,
            5,
            &pipelines,
            &[
                &output_buffers[4],
                &immutable_buffers[10],
                &immutable_buffers[11],
            ],
            &output_buffers[5],
            &uniform_buffer,
            uniform_stride,
        ));
        bind_groups.push(self.create_vision_layer_bind_group(
            &plan,
            6,
            &pipelines,
            &[&immutable_buffers[0], &output_buffers[5]],
            &output_buffers[6],
            &uniform_buffer,
            uniform_stride,
        ));
        bind_groups.push(self.create_vision_layer_bind_group(
            &plan,
            7,
            &pipelines,
            &[
                &output_buffers[6],
                &immutable_buffers[12],
                &immutable_buffers[13],
            ],
            &output_buffers[7],
            &uniform_buffer,
            uniform_stride,
        ));
        bind_groups.push(self.create_vision_layer_bind_group(
            &plan,
            8,
            &pipelines,
            &[
                &output_buffers[7],
                &immutable_buffers[14],
                &immutable_buffers[15],
            ],
            &output_buffers[8],
            &uniform_buffer,
            uniform_stride,
        ));
        bind_groups.push(self.create_vision_layer_bind_group(
            &plan,
            9,
            &pipelines,
            &[&output_buffers[8]],
            &output_buffers[9],
            &uniform_buffer,
            uniform_stride,
        ));
        bind_groups.push(self.create_vision_layer_bind_group(
            &plan,
            10,
            &pipelines,
            &[
                &output_buffers[9],
                &immutable_buffers[16],
                &immutable_buffers[17],
            ],
            &output_buffers[10],
            &uniform_buffer,
            uniform_stride,
        ));
        bind_groups.push(self.create_vision_layer_bind_group(
            &plan,
            11,
            &pipelines,
            &[&output_buffers[6], &output_buffers[10]],
            &output_buffers[11],
            &uniform_buffer,
            uniform_stride,
        ));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vision-layer-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vision-layer-pass"),
                timestamp_writes: None,
            });
            for (index, dispatch) in plan.dispatches.iter().enumerate() {
                pass.set_pipeline(&pipelines[&dispatch.invocation.kernel]);
                pass.set_bind_group(0, &bind_groups[index], &[]);
                pass.dispatch_workgroups(
                    dispatch.invocation.dispatch[0],
                    dispatch.invocation.dispatch[1],
                    dispatch.invocation.dispatch[2],
                );
            }
        }
        let mut readback_offset = 0_u64;
        for &index in readback_indices {
            let bytes = plan.dispatches[index].invocation.output_bytes;
            encoder.copy_buffer_to_buffer(
                &output_buffers[index],
                0,
                &readback_buffer,
                readback_offset,
                bytes,
            );
            readback_offset += bytes;
        }
        let started = js_sys::Date::now();
        self.submit_command_buffers([encoder.finish()]);
        map_read(&readback_buffer, 0..readback_buffer.size()).await?;
        let queue_wall_time_ns =
            (((js_sys::Date::now() - started).max(0.000_001)) * 1_000_000.0).round() as u64;
        let readback_elements = usize::try_from(readback_bytes / 4)
            .map_err(|_| "vision-layer readback is too large".to_owned())?;
        let flat_values = read_f32_buffer(&readback_buffer, readback_elements)?;
        let mut checkpoint_spans = Vec::with_capacity(readback_indices.len());
        let mut element_offset = 0_usize;
        for &index in readback_indices {
            let dispatch = plan.dispatches[index];
            checkpoint_spans.push(BrowserVisionLayerCheckpoint {
                stage: dispatch.stage,
                element_offset,
                elements: dispatch.invocation.output_elements,
            });
            element_offset += dispatch.invocation.output_elements;
        }
        debug_assert_eq!(element_offset, flat_values.len());

        Ok(BrowserVisionLayerExecution {
            checkpoint_values: flat_values,
            checkpoint_spans,
            diagnostics: BrowserVisionLayerDiagnostics {
                checked_error_scopes: CHECKED_SCOPES,
                captured_errors: Vec::new(),
                queue_wall_time_ns: queue_wall_time_ns.max(1),
                shader_blake3,
                dispatch_stages: plan.dispatches.map(|dispatch| dispatch.stage),
                rope_specialization: plan.rope_specialization,
                submission_count: self.submissions.get() - before_submissions,
                command_buffer_count: 1,
                compute_pass_count: 1,
                dispatch_count: plan.dispatches.len() as u32,
                buffer_allocation_count: self.buffer_allocations.get() - before_buffers,
                readback_buffer_count: 1,
                readback_bytes,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create_vision_layer_bind_group(
        &self,
        plan: &VisionEncoderLayerPlan,
        index: usize,
        pipelines: &BTreeMap<KernelId, wgpu::ComputePipeline>,
        inputs: &[&wgpu::Buffer],
        output: &wgpu::Buffer,
        uniform_buffer: &wgpu::Buffer,
        uniform_stride: u64,
    ) -> wgpu::BindGroup {
        let dispatch = plan.dispatches[index];
        let pipeline = &pipelines[&dispatch.invocation.kernel];
        self.create_staged_bind_group(
            &format!("vision-layer-{}-bind-group", dispatch.stage.as_str()),
            pipeline,
            inputs,
            output,
            uniform_buffer,
            index as u64 * uniform_stride,
        )
    }

    fn create_staged_bind_group(
        &self,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        inputs: &[&wgpu::Buffer],
        output: &wgpu::Buffer,
        uniform_buffer: &wgpu::Buffer,
        uniform_offset: u64,
    ) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        let mut entries = Vec::with_capacity(inputs.len() + 2);
        for (binding, input) in inputs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: input.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: output.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32 + 1,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: uniform_buffer,
                offset: uniform_offset,
                size: wgpu::BufferSize::new(VISION_LAYER_UNIFORM_BYTES),
            }),
        });
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &layout,
            entries: &entries,
        })
    }

    fn create_initialized_buffer(
        &self,
        label: &str,
        contents: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            });
        self.buffer_allocations
            .set(self.buffer_allocations.get().saturating_add(1));
        buffer
    }

    fn create_runtime_buffer(
        &self,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        BrowserVisionQkvAllocationAuthority {
            device: &self.device,
            queue: &self.queue,
            buffer_allocations: &self.buffer_allocations,
        }
        .create_buffer(label, size, usage)
    }

    fn submit_command_buffers<const COUNT: usize>(
        &self,
        command_buffers: [wgpu::CommandBuffer; COUNT],
    ) {
        self.queue.submit(command_buffers);
        self.submissions
            .set(self.submissions.get().saturating_add(1));
    }

    async fn validate_pipeline_source(
        &self,
        label: &str,
        source: &str,
        entry_point: &str,
    ) -> Result<(), BrowserVisionStackError> {
        let guards = self.push_browser_error_scopes().await?;
        let completed = self
            .complete_browser_error_scoped_operation(guards, async {
                let shader = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(label),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    });
                let _pipeline =
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(label),
                            layout: None,
                            module: &shader,
                            entry_point: Some(entry_point),
                            compilation_options: Default::default(),
                            cache: None,
                        });
                Ok::<(), BrowserVisionStackError>(())
            })
            .await?;
        let (operation, captured) = completed;
        if !captured.is_empty() {
            return Err(format!("pipeline {label} failed WebGPU validation: {captured:?}").into());
        }
        operation?;
        self.require_no_uncaptured_errors(&format!("pipeline {label}"))
            .map_err(BrowserVisionStackError::from)
    }

    fn validate_vision_patch_projection_buffer_limits(
        &self,
        plan: VisionPatchProjectionBytesPlan,
    ) -> Result<(), String> {
        let limits = &self.capabilities.limits;
        for (label, bytes) in [
            ("input", plan.input_bytes),
            ("weight", plan.weight_bytes),
            ("bias", plan.bias_bytes),
            ("output", plan.output_bytes),
        ] {
            if bytes > limits.max_buffer_size {
                return Err(format!(
                    "vision patch-projection {label} buffer {bytes} exceeds max_buffer_size {}",
                    limits.max_buffer_size
                ));
            }
            if bytes > limits.max_storage_buffer_binding_size {
                return Err(format!(
                    "vision patch-projection {label} binding {bytes} exceeds max_storage_buffer_binding_size {}",
                    limits.max_storage_buffer_binding_size
                ));
            }
        }
        if limits.max_storage_buffers_per_shader_stage < 4
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 8
            || limits.max_compute_workgroup_size_y < 8
        {
            return Err(
                "vision patch-projection device limits do not admit the fixed kernel ABI"
                    .to_owned(),
            );
        }
        Ok(())
    }

    async fn run_vision_patch_projection_bytes_source(
        &self,
        descriptor: VisionPatchProjectionBytesDescriptor,
        plan: VisionPatchProjectionBytesPlan,
        input: &[u8],
        weight: &[u8],
        bias: &[u8],
    ) -> Result<(Vec<u8>, BrowserDiagnostics), BrowserVisionStackError> {
        let module = pvlc_wgsl::module(plan.kernel).ok_or_else(|| {
            BrowserVisionStackError(format!(
                "WGSL catalog has no {} module",
                plan.kernel.as_str()
            ))
        })?;
        let uniform_bytes = [
            descriptor.patch_count,
            descriptor.input_width,
            descriptor.output_width,
            0,
        ]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
        let guards = self.push_browser_error_scopes().await?;
        let completed = self
            .complete_browser_error_scoped_operation(guards, async {
                self.execute_vision_patch_projection_bytes_once(
                    plan,
                    input,
                    weight,
                    bias,
                    &uniform_bytes,
                    module.source,
                    module.spec.entry_point,
                )
                .await
                .map_err(BrowserVisionStackError::from)
            })
            .await?;
        let (operation, captured) = completed;
        if !captured.is_empty() {
            let operation_context = operation
                .as_ref()
                .err()
                .map(|error| format!("; operation also failed: {error}"))
                .unwrap_or_default();
            return Err(format!(
                "vision patch-projection captured WebGPU errors: {captured:?}{operation_context}"
            )
            .into());
        }
        let execution = operation?;
        self.require_no_uncaptured_errors("vision patch-projection")?;
        Ok(execution)
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_vision_patch_projection_bytes_once(
        &self,
        plan: VisionPatchProjectionBytesPlan,
        input: &[u8],
        weight: &[u8],
        bias: &[u8],
        uniform_bytes: &[u8],
        source: &str,
        entry_point: &str,
    ) -> Result<(Vec<u8>, BrowserDiagnostics), String> {
        let label = plan.kernel.as_str();
        let pipeline = self.pipeline(label, source, entry_point, Some(plan.kernel));
        let input_buffers = [
            self.create_initialized_buffer(
                "vision-patch-projection-input",
                input,
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-patch-projection-weight",
                weight,
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-patch-projection-bias",
                bias,
                wgpu::BufferUsages::STORAGE,
            ),
        ];
        let output_buffer = self.create_runtime_buffer(
            "vision-patch-projection-output",
            plan.output_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let uniform_buffer = self.create_initialized_buffer(
            "vision-patch-projection-uniform",
            uniform_bytes,
            wgpu::BufferUsages::UNIFORM,
        );
        let readback_buffer = self.create_runtime_buffer(
            "vision-patch-projection-readback",
            plan.output_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vision-patch-projection-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffers[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffers[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_buffers[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vision-patch-projection-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vision-patch-projection-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(plan.dispatch[0], plan.dispatch[1], plan.dispatch[2]);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, plan.output_bytes);
        let started = js_sys::Date::now();
        self.submit_command_buffers([encoder.finish()]);
        map_read(&readback_buffer, 0..plan.output_bytes).await?;
        let queue_wall_time_ns =
            (((js_sys::Date::now() - started).max(0.000_001)) * 1_000_000.0).round() as u64;
        let elements = usize::try_from(plan.output_bytes / 4)
            .map_err(|_| "vision patch-projection output is too large".to_owned())?;
        let values = read_f32_buffer(&readback_buffer, elements)?;
        let checkpoint_bytes = bytemuck::cast_slice(&values).to_vec();
        Ok((
            checkpoint_bytes,
            BrowserDiagnostics {
                kernel: plan.kernel,
                checked_error_scopes: CHECKED_SCOPES,
                captured_errors: Vec::new(),
                queue_wall_time_ns: queue_wall_time_ns.max(1),
                shader_blake3: blake3::hash(source.as_bytes()).to_hex().to_string(),
            },
        ))
    }

    async fn run_source(
        &self,
        invocation: &KernelInvocation,
        label: &str,
        source: &str,
        entry_point: &str,
        cached_kernel: Option<KernelId>,
    ) -> Result<BrowserExecution, BrowserVisionStackError> {
        let plan = invocation
            .plan()
            .map_err(|error| format!("invalid kernel invocation: {error}"))?;
        let uniform_bytes = invocation
            .uniform_bytes()
            .map_err(|error| format!("cannot encode invocation uniform: {error}"))?;
        let guards = self.push_browser_error_scopes().await?;
        let completed = self
            .complete_browser_error_scoped_operation(guards, async {
                self.execute_once(
                    invocation,
                    plan,
                    &uniform_bytes,
                    label,
                    source,
                    entry_point,
                    cached_kernel,
                )
                .await
                .map_err(BrowserVisionStackError::from)
            })
            .await?;
        let (operation, captured) = completed;
        if !captured.is_empty() {
            let operation_context = operation
                .as_ref()
                .err()
                .map(|error| format!("; operation also failed: {error}"))
                .unwrap_or_default();
            return Err(
                format!("{label} captured WebGPU errors: {captured:?}{operation_context}").into(),
            );
        }
        let execution = operation?;
        self.require_no_uncaptured_errors(label)?;
        Ok(execution)
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_once(
        &self,
        invocation: &KernelInvocation,
        plan: InvocationPlan,
        uniform_bytes: &[u8],
        label: &str,
        source: &str,
        entry_point: &str,
        cached_kernel: Option<KernelId>,
    ) -> Result<BrowserExecution, String> {
        let pipeline = self.pipeline(label, source, entry_point, cached_kernel);
        let input_data = invocation.inputs();
        let input_buffers = input_data
            .iter()
            .enumerate()
            .map(|(index, input)| {
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("{label}-input-{index}")),
                        contents: invocation_input_bytes(input),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            })
            .collect::<Vec<_>>();
        let output_contents = invocation.output_initializer().map_or_else(
            || vec![0_u8; plan.output_bytes as usize],
            |values| bytemuck::cast_slice(values).to_vec(),
        );
        let output_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label}-output")),
                contents: &output_contents,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label}-uniform")),
                contents: uniform_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let readback_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-readback")),
            size: plan.output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let mut entries = Vec::with_capacity(input_buffers.len() + 2);
        for (binding, buffer) in input_buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: input_buffers.len() as u32,
            resource: output_buffer.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: input_buffers.len() as u32 + 1,
            resource: uniform_buffer.as_entire_binding(),
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label}-bind-group")),
            layout: &bind_group_layout,
            entries: &entries,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{label}-encoder")),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label}-pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(plan.dispatch[0], plan.dispatch[1], plan.dispatch[2]);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, plan.output_bytes);

        let started = js_sys::Date::now();
        self.queue.submit([encoder.finish()]);
        map_read(&readback_buffer, 0..readback_buffer.size()).await?;
        let queue_wall_time_ns =
            (((js_sys::Date::now() - started).max(0.000_001)) * 1_000_000.0).round() as u64;
        let values = read_f32_buffer(&readback_buffer, plan.output_elements)?;
        Ok(BrowserExecution {
            values,
            diagnostics: BrowserDiagnostics {
                kernel: invocation.kernel_id(),
                checked_error_scopes: CHECKED_SCOPES,
                captured_errors: Vec::new(),
                queue_wall_time_ns: queue_wall_time_ns.max(1),
                shader_blake3: blake3::hash(source.as_bytes()).to_hex().to_string(),
            },
        })
    }

    fn pipeline(
        &self,
        label: &str,
        source: &str,
        entry_point: &str,
        cached_kernel: Option<KernelId>,
    ) -> wgpu::ComputePipeline {
        if let Some(kernel) = cached_kernel
            && let Some(pipeline) = self.pipelines.borrow().get(&kernel).cloned()
        {
            return pipeline;
        }
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(kernel) = cached_kernel {
            self.pipelines.borrow_mut().insert(kernel, pipeline.clone());
        }
        pipeline
    }

    fn vision_stack_pipeline(
        &self,
        post_effect: &VisionStackPostEffectToken,
        label: &str,
        source: &str,
        entry_point: &str,
    ) -> Result<wgpu::ComputePipeline, String> {
        let shader = match run_first_webgpu_effect(
            &self.device,
            post_effect,
            PreparedFirstWebGpuEffect::CreateShaderModule { label, source },
        )
        .map_err(|error| error.to_string())?
        {
            FirstWebGpuEffectOutput::ShaderModule(shader) => shader,
            _ => return Err("sealed shader creation returned the wrong output".to_owned()),
        };
        let pipeline = match run_first_webgpu_effect(
            &self.device,
            post_effect,
            PreparedFirstWebGpuEffect::CreateComputePipeline {
                label,
                module: &shader,
                entry_point,
            },
        )
        .map_err(|error| error.to_string())?
        {
            FirstWebGpuEffectOutput::ComputePipeline(pipeline) => pipeline,
            _ => return Err("sealed pipeline creation returned the wrong output".to_owned()),
        };
        Ok(pipeline)
    }

    fn prepare_vision_stack_first_error_scope(
        &self,
    ) -> Result<PreparedVisionStackFirstErrorScope<'_>, BrowserVisionStackError> {
        let authority = VisionStackErrorScopeAuthority::acquire(
            &self.vision_stack_error_scopes_healthy,
            &self.vision_stack_error_scopes_occupied,
            "persistent WebGPU error-scope authority is poisoned".to_owned(),
            "another browser WebGPU error-scope operation is already in progress".to_owned(),
        )?;
        self.clear_uncaptured_errors();
        let (raw_device, push) = self.raw_device_method("pushErrorScope")?;
        let (_, pop) = self.raw_device_method("popErrorScope")?;
        Ok(PreparedVisionStackFirstErrorScope {
            authority,
            raw_device,
            push,
            pop,
        })
    }

    async fn push_browser_error_scopes<'a>(
        &'a self,
    ) -> Result<BrowserVisionStackErrorScopes<'a>, BrowserVisionStackError> {
        if self.execution_busy.get() || self.vision_stack_session.borrow().is_busy() {
            return Err(BrowserVisionStackError(
                "another browser WebGPU execution is already in progress".to_owned(),
            ));
        }
        let PreparedVisionStackFirstErrorScope {
            mut authority,
            raw_device,
            push,
            pop,
        } = self.prepare_vision_stack_first_error_scope()?;
        let internal = push_vision_stack_error_scope_or_drain(
            &mut authority,
            ScopeKind::Internal,
            |scope| {
                push.call1(raw_device, &JsValue::from_str(scope.filter_str()))
                    .map(|_| ())
                    .map_err(|error| {
                        BrowserVisionStackError(format!(
                            "cannot push {} WebGPU error scope: {error:?}",
                            scope.as_str()
                        ))
                    })
            },
            |scope| pop_browser_vision_stack_error_scope(raw_device, &pop, scope),
        )
        .await;
        if let Err(failure) = internal {
            return Err(browser_vision_stack_scope_push_failure(
                ScopeKind::Internal,
                failure,
            ));
        }
        let out_of_memory = push_vision_stack_error_scope_or_drain(
            &mut authority,
            ScopeKind::OutOfMemory,
            |scope| {
                push.call1(raw_device, &JsValue::from_str(scope.filter_str()))
                    .map(|_| ())
                    .map_err(|error| {
                        BrowserVisionStackError(format!(
                            "cannot push {} WebGPU error scope: {error:?}",
                            scope.as_str()
                        ))
                    })
            },
            |scope| pop_browser_vision_stack_error_scope(raw_device, &pop, scope),
        )
        .await;
        if let Err(failure) = out_of_memory {
            return Err(browser_vision_stack_scope_push_failure(
                ScopeKind::OutOfMemory,
                failure,
            ));
        }
        let validation = push_vision_stack_error_scope_or_drain(
            &mut authority,
            ScopeKind::Validation,
            |scope| {
                push.call1(raw_device, &JsValue::from_str(scope.filter_str()))
                    .map(|_| ())
                    .map_err(|error| {
                        BrowserVisionStackError(format!(
                            "cannot push {} WebGPU error scope: {error:?}",
                            scope.as_str()
                        ))
                    })
            },
            |scope| pop_browser_vision_stack_error_scope(raw_device, &pop, scope),
        )
        .await;
        if let Err(failure) = validation {
            return Err(browser_vision_stack_scope_push_failure(
                ScopeKind::Validation,
                failure,
            ));
        }
        Ok(BrowserVisionStackErrorScopes {
            authority,
            raw_device,
            pop,
        })
    }

    async fn push_vision_stack_error_scopes<'a>(
        &self,
        prepared_first_effect: PreparedVisionStackFirstErrorScope<'a>,
        post_effect: &VisionStackPostEffectToken,
    ) -> Result<BrowserVisionStackErrorScopes<'a>, BrowserVisionStackError> {
        if post_effect.effect_tracker_id() == 0
            || post_effect.effect_boundary() != VisionStackGpuEffectBoundary::PostEffect
        {
            return Err(
                "sealed first WebGPU effect received invalid causal authority"
                    .to_owned()
                    .into(),
            );
        }
        let PreparedVisionStackFirstErrorScope {
            mut authority,
            raw_device,
            push,
            pop,
        } = prepared_first_effect;
        let first = run_first_webgpu_effect(
            &self.device,
            post_effect,
            PreparedFirstWebGpuEffect::PushErrorScope {
                raw_device,
                push: &push,
                filter: ScopeKind::Internal.filter_str(),
            },
        )?;
        VisionStackErrorScopeAuthority::after_first_push(&mut authority, ScopeKind::Internal);
        let FirstWebGpuEffectOutput::ErrorScope = first else {
            return Err("sealed first WebGPU effect returned the wrong output"
                .to_owned()
                .into());
        };
        let out_of_memory = push_vision_stack_error_scope_or_drain(
            &mut authority,
            ScopeKind::OutOfMemory,
            |scope| {
                push.call1(raw_device, &JsValue::from_str(scope.filter_str()))
                    .map(|_| ())
                    .map_err(|_| {
                        "cannot push out-of-memory WebGPU error scope"
                            .to_owned()
                            .into()
                    })
            },
            |scope| pop_browser_vision_stack_error_scope(raw_device, &pop, scope),
        )
        .await;
        if let Err(failure) = out_of_memory {
            return Err(browser_vision_stack_scope_push_failure(
                ScopeKind::OutOfMemory,
                failure,
            ));
        }
        let validation = push_vision_stack_error_scope_or_drain(
            &mut authority,
            ScopeKind::Validation,
            |scope| {
                push.call1(raw_device, &JsValue::from_str(scope.filter_str()))
                    .map(|_| ())
                    .map_err(|_| {
                        "cannot push validation WebGPU error scope"
                            .to_owned()
                            .into()
                    })
            },
            |scope| pop_browser_vision_stack_error_scope(raw_device, &pop, scope),
        )
        .await;
        if let Err(failure) = validation {
            return Err(browser_vision_stack_scope_push_failure(
                ScopeKind::Validation,
                failure,
            ));
        }
        Ok(BrowserVisionStackErrorScopes {
            authority,
            raw_device,
            pop,
        })
    }

    async fn complete_browser_error_scoped_operation<'a, T, Operation>(
        &self,
        guards: BrowserVisionStackErrorScopes<'a>,
        operation: Operation,
    ) -> Result<
        (
            Result<T, BrowserVisionStackError>,
            Vec<(&'static str, String)>,
        ),
        BrowserVisionStackError,
    >
    where
        Operation: Future<Output = Result<T, BrowserVisionStackError>>,
    {
        let completed =
            run_vision_stack_error_scoped_operation(guards.authority, operation, |scope| {
                pop_browser_vision_stack_error_scope(guards.raw_device, &guards.pop, scope)
            })
            .await;
        let (operation, cleanup) = completed.into_parts();
        let (captures, failures, remaining) = cleanup.into_parts();
        if !failures.is_empty() || remaining > 0 {
            let mut message = "cannot close browser WebGPU error scopes".to_owned();
            for failure in failures {
                message.push_str("; ");
                message.push_str(&failure.0);
            }
            if remaining > 0 {
                message.push_str("; persistent scope authority poisoned with ");
                message.push_str(&remaining.to_string());
                message.push_str(" unconfirmed scope(s)");
            }
            if let Err(operation_error) = &operation {
                message.push_str("; operation also failed: ");
                message.push_str(&operation_error.0);
            }
            return Err(BrowserVisionStackError(message));
        }
        Ok((operation, captures.into_iter().flatten().collect()))
    }

    fn raw_device_method(
        &self,
        name: &'static str,
    ) -> Result<(&JsValue, js_sys::Function), String> {
        let device = self.raw_device()?;
        let function = js_sys::Reflect::get(device, &JsValue::from_str(name))
            .map_err(|error| format!("cannot access GPUDevice.{name}: {error:?}"))?
            .dyn_into::<js_sys::Function>()
            .map_err(|_| format!("GPUDevice.{name} is not callable"))?;
        Ok((device, function))
    }

    fn raw_device(&self) -> Result<&JsValue, String> {
        self.device
            .as_webgpu()
            .map(AsRef::as_ref)
            .ok_or_else(|| "wgpu device has no BrowserWebGPU handle".to_owned())
    }

    fn clear_uncaptured_errors(&self) {
        self.uncaptured_errors
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    fn take_uncaptured_errors(&self) -> Vec<String> {
        std::mem::take(
            &mut *self
                .uncaptured_errors
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }

    fn require_no_uncaptured_errors(&self, context: &str) -> Result<(), String> {
        let errors = self.take_uncaptured_errors();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{context} emitted uncaptured WebGPU errors: {}",
                errors.join("; ")
            ))
        }
    }
}

fn run_vision_stack_streaming_session_layer<Validate, Allocate, Upload, Submit>(
    owner: &AsyncSessionOwner<BrowserVisionStackSession>,
    execution_busy: &Cell<bool>,
    streaming_cache: &mut VisionStackStreamingWeightCache<wgpu::Buffer>,
    validate: Validate,
    allocate: Allocate,
    upload: Upload,
    submit: Submit,
) -> Result<(), VisionStackStreamingFailure<BrowserVisionStackError>>
where
    Validate: FnOnce(
        &mut BrowserVisionStackSession,
    ) -> Result<VisionStackStreamingLayerSchedule, BrowserVisionStackError>,
    Allocate: FnMut(
        usize,
        VisionStackStreamingWeightRange,
    ) -> Result<wgpu::Buffer, BrowserVisionStackError>,
    Upload: FnMut(
        usize,
        VisionStackStreamingWeightRange,
        &wgpu::Buffer,
    ) -> Result<(), BrowserVisionStackError>,
    Submit: FnOnce(
        &mut BrowserVisionStackSession,
        u32,
        Option<usize>,
        &[wgpu::Buffer],
    ) -> Result<(), BrowserVisionStackError>,
{
    let resident_disposition = owner.stored().and_then(|session| {
        session
            .resident_weights
            .as_ref()
            .map(|resident| resident.disposition)
    });
    match resident_disposition {
        None => run_causal_vision_stack_streaming_session_layer(
            owner,
            execution_busy,
            streaming_cache,
            validate,
            allocate,
            upload,
            submit,
        ),
        Some(BrowserVisionStackWeightResidency::Ready) => {
            Err(VisionStackStreamingFailure::Admission(
                BrowserVisionStackError(
                    "resident vision weights are already complete; use the payload-free resident layer API"
                        .to_owned(),
                ),
            ))
        }
        Some(BrowserVisionStackWeightResidency::Cold) => {
            let (lease, mut session) = owner
                .acquire()
                .map_err(VisionStackStreamingFailure::Unavailable)?;
            let (resident_cache, resident_key) = session
                .resident_weights
                .as_ref()
                .map(|resident| (Rc::clone(&resident.cache), resident.key.clone()))
                .ok_or_else(|| {
                    VisionStackStreamingFailure::Admission(BrowserVisionStackError(
                        "resident vision-weight cache authority is missing".to_owned(),
                    ))
            })?;
            let layer_count = usize::try_from(session.plan.layer_count)
                .expect("u32 vision layer count must fit the wasm32 usize ABI");
            let identity_matches = resident_cache
                .borrow()
                .is_prepared_for(&resident_key, layer_count);
            let resident_outcome = if identity_matches {
                run_vision_stack_resident_cold_layer(
                    &mut *resident_cache.borrow_mut(),
                    &mut session,
                    validate,
                    allocate,
                    upload,
                    submit,
                )
            } else {
                Err(VisionStackResidentFailure::Admission(
                    BrowserVisionStackError(
                        "resident vision-weight cache identity changed during execution".to_owned(),
                    ),
                ))
            };
            let outcome = resident_outcome.map_err(|error| match error {
                VisionStackResidentFailure::Admission(error) => {
                    VisionStackStreamingFailure::Admission(error)
                }
                VisionStackResidentFailure::Effect { error, boundary } => {
                    VisionStackStreamingFailure::Effect { error, boundary }
                }
                VisionStackResidentFailure::Cache(error) => {
                    VisionStackStreamingFailure::Effect {
                        error: BrowserVisionStackError(format!(
                            "cannot commit resident vision layer: {error}"
                        )),
                        boundary: VisionStackGpuEffectBoundary::PostEffect,
                    }
                }
            });
            let action = match &outcome {
                Ok(()) | Err(VisionStackStreamingFailure::Admission(_)) => {
                    crate::CompletionAction::Restore
                }
                Err(VisionStackStreamingFailure::Effect { .. })
                | Err(VisionStackStreamingFailure::Unavailable(_))
                | Err(VisionStackStreamingFailure::CacheLengthMismatch { .. })
                | Err(VisionStackStreamingFailure::Completion(_)) => {
                    crate::CompletionAction::Finish
                }
            };
            let completion = owner.complete(lease, session, action);
            let _ = coordinate_vision_stack_completion_busy(execution_busy, completion);
            match (&outcome, completion) {
                (Ok(()), crate::CompletionOutcome::Restored)
                | (
                    Err(VisionStackStreamingFailure::Admission(_)),
                    crate::CompletionOutcome::Restored,
                )
                | (
                    Err(VisionStackStreamingFailure::Effect { .. }),
                    crate::CompletionOutcome::Finished,
                ) => outcome,
                _ => Err(VisionStackStreamingFailure::Completion(completion)),
            }
        }
    }
}

fn parse_vision_stack_activation_strategy(
    value: &str,
) -> Result<VisionStackActivationStrategy, String> {
    match value {
        "separate_buffers" => Ok(VisionStackActivationStrategy::SeparateBuffers),
        "static_arena_no_alias" => Ok(VisionStackActivationStrategy::StaticArenaNoAlias),
        "static_arena_alias" => Ok(VisionStackActivationStrategy::StaticArenaAlias),
        _ => Err(format!(
            "unknown vision-stack activation strategy {value:?}; expected separate_buffers, static_arena_no_alias, or static_arena_alias"
        )),
    }
}

fn require_resident_vision_stack_manifest(
    manifest: &VisionStackShardManifest,
) -> Result<(), String> {
    if manifest.matrix_weight_storage != DecoderWeightStorage::F16
        || manifest.matrix_weight_layout != LinearWeightLayout::InputMajor
    {
        return Err("resident vision weights require FP16 input-major matrix storage".to_owned());
    }
    Ok(())
}

fn parse_vision_qkv_execution_policy(value: &str) -> Result<VisionQkvExecutionPolicy, String> {
    match value {
        "disabled" => Ok(VisionQkvExecutionPolicy::Disabled),
        "preferred" => Ok(VisionQkvExecutionPolicy::Preferred),
        "required" => Ok(VisionQkvExecutionPolicy::Required),
        _ => Err(format!(
            "unknown vision Q/K/V policy {value:?}; expected disabled, preferred, or required"
        )),
    }
}

fn vision_stack_shader_sources(
    activation_strategy: VisionStackActivationStrategy,
    projection_kernel: KernelId,
    matrix_weight_storage: DecoderWeightStorage,
    matrix_weight_layout: LinearWeightLayout,
    activation_storage: DecoderWeightStorage,
    rope_kernel: KernelId,
) -> Result<BTreeMap<KernelId, String>, String> {
    if !matches!(
        projection_kernel,
        KernelId::VisionPatchProjectionF32
            | KernelId::LinearProjectionF16Weights
            | KernelId::LinearProjectionF16
    ) {
        return Err(format!(
            "unsupported vision-stack projection kernel {projection_kernel}"
        ));
    }
    let tiled_fp16_qkv_kernel = (matrix_weight_storage == DecoderWeightStorage::F16
        && matrix_weight_layout == LinearWeightLayout::InputMajor
        && activation_storage == DecoderWeightStorage::F32)
        .then_some(KernelId::VisionQkvFusedF16Weights);
    let kernels = match activation_storage {
        DecoderWeightStorage::F16 => vec![
            KernelId::LayerNormF16,
            KernelId::LinearProjectionF16,
            KernelId::VisionAttentionF16,
            KernelId::AddF16,
            KernelId::GeluTanhF16,
            KernelId::VisionRope2dF16,
        ],
        DecoderWeightStorage::F32 => vec![
            KernelId::LayerNormF32,
            projection_kernel,
            KernelId::VisionAttentionF32,
            KernelId::AddF32,
            KernelId::GeluTanhF32,
            KernelId::VisionRope2dF32,
        ],
    };
    kernels
        .into_iter()
        .chain(tiled_fp16_qkv_kernel)
        .map(|kernel| {
            if matches!(
                kernel,
                KernelId::VisionRope2dF32 | KernelId::VisionRope2dF16
            ) && kernel != rope_kernel
            {
                return Err(format!(
                    "vision-stack precision selected {rope_kernel}, but shader family contains {kernel}",
                ));
            }
            let module = pvlc_wgsl::module(kernel)
            .expect("every resident vision-stack kernel must have fixed WGSL");
        let source = match (activation_strategy, kernel) {
            (
                VisionStackActivationStrategy::StaticArenaNoAlias
                | VisionStackActivationStrategy::StaticArenaAlias,
                KernelId::VisionQkvFusedF16Weights,
            ) => {
                pvlc_wgsl::validate_source_contract(&module.spec, module.source).map_err(
                    |error| format!("resident vision-stack shader {kernel} is invalid: {error}"),
                )?;
                module.source.to_owned()
            }
            (VisionStackActivationStrategy::SeparateBuffers, _) => {
                pvlc_wgsl::validate_source_contract(&module.spec, module.source).map_err(
                    |error| format!("resident vision-stack shader {kernel} is invalid: {error}"),
                )?;
                module.source.to_owned()
            }
            (
                VisionStackActivationStrategy::StaticArenaNoAlias
                | VisionStackActivationStrategy::StaticArenaAlias,
                _,
            ) => pvlc_wgsl::storage_read_write_variant(&module.spec, module.source).map_err(
                |error| format!("resident vision-stack static shader {kernel} is invalid: {error}"),
            )?,
        };
        Ok((kernel, source))
    })
    .collect()
}

fn vision_qkv_stack_shader_sources(
    activation_strategy: VisionStackActivationStrategy,
) -> Result<BTreeMap<KernelId, String>, String> {
    VISION_QKV_STACK_KERNELS
        .into_iter()
        .map(|kernel| {
            let module = pvlc_wgsl::module(kernel)
                .expect("every optimized vision-stack kernel must have fixed WGSL");
            let source = match activation_strategy {
                VisionStackActivationStrategy::SeparateBuffers => {
                    pvlc_wgsl::validate_source_contract(&module.spec, module.source).map_err(
                        |error| {
                            format!("optimized vision-stack shader {kernel} is invalid: {error}")
                        },
                    )?;
                    module.source.to_owned()
                }
                VisionStackActivationStrategy::StaticArenaNoAlias
                | VisionStackActivationStrategy::StaticArenaAlias => {
                    pvlc_wgsl::storage_read_write_variant(&module.spec, module.source).map_err(
                        |error| {
                            format!(
                                "optimized vision-stack static shader {kernel} is invalid: {error}"
                            )
                        },
                    )?
                }
            };
            Ok((kernel, source))
        })
        .collect()
}

fn parse_projector_invocation(invocation_json: &str) -> Result<OwnedProjectorInvocation, String> {
    serde_json::from_str(invocation_json)
        .map_err(|error| format!("invalid projector invocation JSON: {error}"))
}

fn parse_projector_f16_descriptor(
    descriptor_json: &str,
) -> Result<BrowserProjectorF16Descriptor, String> {
    let descriptor: BrowserProjectorF16Descriptor = serde_json::from_str(descriptor_json)
        .map_err(|error| format!("invalid FP16 projector descriptor JSON: {error}"))?;
    if descriptor.schema_version != 1 {
        return Err(format!(
            "unsupported FP16 projector schema version {}",
            descriptor.schema_version,
        ));
    }
    if descriptor.weights_blake3.len() != 64
        || !descriptor
            .weights_blake3
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("FP16 projector weights_blake3 must be a 64-digit hex digest".to_owned());
    }
    if descriptor.weight_storage != "f16"
        || descriptor.matrix_weight_layout != "input_major"
        || descriptor.activation_storage != "f16"
    {
        return Err(
            "projector GPU chain requires F16 weights/activations and input-major matrices"
                .to_owned(),
        );
    }
    if descriptor.hidden_size == 0
        || descriptor.output_size == 0
        || !descriptor.hidden_size.is_multiple_of(4)
        || !descriptor.output_size.is_multiple_of(4)
        || !descriptor.layer_norm_epsilon.is_finite()
        || descriptor.layer_norm_epsilon <= 0.0
    {
        return Err("FP16 projector dimensions or layer-norm epsilon are invalid".to_owned());
    }
    let ranges = projector_f16_weight_ranges(&descriptor)?;
    let planned_bytes = ranges
        .last()
        .and_then(|range| range.offset.checked_add(range.bytes))
        .ok_or_else(|| "FP16 projector weight layout is empty or overflowed".to_owned())?;
    if planned_bytes != descriptor.weights_bytes {
        return Err(format!(
            "FP16 projector descriptor declares {} bytes, geometry requires {planned_bytes}",
            descriptor.weights_bytes,
        ));
    }
    Ok(descriptor)
}

fn projector_f16_weight_ranges(
    descriptor: &BrowserProjectorF16Descriptor,
) -> Result<Vec<VisionStackTensorRange>, String> {
    let hidden = u64::from(descriptor.hidden_size);
    let merged = hidden
        .checked_mul(4)
        .ok_or_else(|| "FP16 projector merged width overflowed".to_owned())?;
    let output = u64::from(descriptor.output_size);
    let element_counts = [
        hidden,
        hidden,
        merged
            .checked_mul(merged)
            .ok_or_else(|| "FP16 projector linear1 matrix overflowed".to_owned())?,
        merged,
        output
            .checked_mul(merged)
            .ok_or_else(|| "FP16 projector linear2 matrix overflowed".to_owned())?,
        output,
    ];
    let mut offset = 0_u64;
    let mut ranges = Vec::with_capacity(element_counts.len());
    for elements in element_counts {
        let bytes = elements
            .checked_mul(2)
            .ok_or_else(|| "FP16 projector tensor byte size overflowed".to_owned())?;
        ranges.push(VisionStackTensorRange { offset, bytes });
        offset = offset
            .checked_add(bytes)
            .ok_or_else(|| "FP16 projector payload offset overflowed".to_owned())?;
    }
    Ok(ranges)
}

fn projector_f16_execution_plan(
    descriptor_json: &str,
    image_grid_thw_json: &str,
) -> Result<(BrowserProjectorF16Descriptor, ProjectorPlan), String> {
    let descriptor = parse_projector_f16_descriptor(descriptor_json)?;
    let image_grid_thw: Vec<[u32; 3]> = serde_json::from_str(image_grid_thw_json)
        .map_err(|error| format!("invalid projector image_grid_thw: {error}"))?;
    let plan = ProjectorGeometry {
        hidden_size: descriptor.hidden_size,
        output_size: descriptor.output_size,
        layer_norm_epsilon: descriptor.layer_norm_epsilon,
        image_grid_thw: &image_grid_thw,
    }
    .plan_full_f16()
    .map_err(|error| format!("invalid FP16 projector geometry: {error}"))?;
    if plan.resident_weight_bytes != descriptor.weights_bytes {
        return Err(format!(
            "FP16 projector plan requires {} weight bytes, descriptor declares {}",
            plan.resident_weight_bytes, descriptor.weights_bytes,
        ));
    }
    Ok((descriptor, plan))
}

fn parse_projector_readback(value: &str) -> Result<ProjectorReadback, String> {
    match value {
        "all_stages" => Ok(ProjectorReadback::AllStages),
        "output_only" => Ok(ProjectorReadback::OutputOnly),
        _ => Err(format!(
            "invalid projector readback {value:?}; expected all_stages or output_only"
        )),
    }
}

fn parse_projector_kernel(value: &str) -> Result<KernelId, String> {
    PROJECTOR_KERNELS
        .into_iter()
        .find(|kernel| kernel.as_str() == value)
        .ok_or_else(|| format!("{value:?} is not a physical resident projector kernel"))
}

fn parse_vision_layer_invocation(
    invocation_json: &str,
) -> Result<OwnedVisionEncoderLayerInvocation, String> {
    serde_json::from_str(invocation_json)
        .map_err(|error| format!("invalid vision-layer invocation JSON: {error}"))
}

fn parse_vision_layer_readback(value: &str) -> Result<VisionLayerReadback, String> {
    match value {
        "all_stages" => Ok(VisionLayerReadback::AllStages),
        "output_only" => Ok(VisionLayerReadback::OutputOnly),
        _ => Err(format!(
            "invalid vision-layer readback {value:?}; expected all_stages or output_only"
        )),
    }
}

fn parse_vision_layer_kernel(value: &str) -> Result<KernelId, String> {
    VISION_LAYER_KERNELS
        .into_iter()
        .find(|kernel| kernel.as_str() == value)
        .ok_or_else(|| format!("{value:?} is not a physical resident vision-layer kernel"))
}

fn vision_layer_readback_indices(readback: VisionLayerReadback) -> &'static [usize] {
    const OUTPUT_ONLY: [usize; 1] = [11];
    const ALL_STAGES: [usize; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    match readback {
        VisionLayerReadback::OutputOnly => &OUTPUT_ONLY,
        VisionLayerReadback::AllStages => &ALL_STAGES,
    }
}

fn projector_readback_indices(readback: ProjectorReadback) -> &'static [usize] {
    const OUTPUT_ONLY: [usize; 1] = [4];
    const ALL_STAGES: [usize; 5] = [0, 1, 2, 3, 4];
    match readback {
        ProjectorReadback::OutputOnly => &OUTPUT_ONLY,
        ProjectorReadback::AllStages => &ALL_STAGES,
    }
}

fn vision_layer_uniform_stride(minimum_alignment: u32) -> u64 {
    let alignment = u64::from(minimum_alignment.max(1));
    VISION_LAYER_UNIFORM_BYTES.div_ceil(alignment) * alignment
}

fn blake3_hex(source: &str) -> String {
    blake3::hash(source.as_bytes()).to_hex().to_string()
}

fn blake3_js_bytes(bytes: &js_sys::Uint8Array) -> String {
    let mut hasher = blake3::Hasher::new();
    let mut scratch = vec![0_u8; JS_BRIDGE_CHUNK_BYTES as usize];
    let mut offset = 0_u32;
    while offset < bytes.length() {
        let end = offset
            .saturating_add(JS_BRIDGE_CHUNK_BYTES)
            .min(bytes.length());
        let length = (end - offset) as usize;
        bytes.subarray(offset, end).copy_to(&mut scratch[..length]);
        hasher.update(&scratch[..length]);
        offset = end;
    }
    hasher.finalize().to_hex().to_string()
}

fn inspect_js_vision_stack_shard(
    manifest: &VisionStackShardManifest,
    weight_plan: &BrowserVisionStackLayerWeightPlan,
    id: &str,
    bytes: &js_sys::Uint8Array,
) -> Result<VisionStackShardObservation, String> {
    let descriptor = manifest
        .shards
        .iter()
        .find(|descriptor| descriptor.id == id)
        .ok_or_else(|| format!("vision-stack shard {id} is not declared"))?;
    let mut hasher = blake3::Hasher::new();
    let mut scratch = vec![0_u8; JS_BRIDGE_CHUNK_BYTES as usize];
    let length_matches = u64::from(bytes.length()) == descriptor.bytes;
    let ranges = match descriptor.kind {
        VisionStackShardKind::Layer if length_matches => weight_plan
            .ranges
            .iter()
            .map(|range| (range.offset, range.bytes, range.storage))
            .collect::<Vec<_>>(),
        _ => vec![(0, u64::from(bytes.length()), DecoderWeightStorage::F32)],
    };
    let mut all_finite = length_matches;
    let mut hashed_bytes = 0_u64;
    for (range_offset, range_bytes, storage) in ranges {
        let range_end = range_offset
            .checked_add(range_bytes)
            .ok_or_else(|| "vision-stack inspection range overflowed".to_owned())?;
        if range_end > u64::from(bytes.length()) {
            all_finite = false;
            continue;
        }
        let mut offset = range_offset;
        while offset < range_end {
            let end = offset
                .saturating_add(u64::from(JS_BRIDGE_CHUNK_BYTES))
                .min(range_end);
            let js_offset = u32::try_from(offset)
                .map_err(|_| "vision-stack inspection offset exceeds Uint8Array".to_owned())?;
            let js_end = u32::try_from(end)
                .map_err(|_| "vision-stack inspection end exceeds Uint8Array".to_owned())?;
            let length = usize::try_from(end - offset)
                .map_err(|_| "vision-stack inspection chunk is too large".to_owned())?;
            bytes
                .subarray(js_offset, js_end)
                .copy_to(&mut scratch[..length]);
            hasher.update(&scratch[..length]);
            hashed_bytes = hashed_bytes
                .checked_add(end - offset)
                .ok_or_else(|| "vision-stack inspected-byte count overflowed".to_owned())?;
            if all_finite && storage.validate_finite_bytes(&scratch[..length]).is_err() {
                all_finite = false;
            }
            offset = end;
        }
    }
    if hashed_bytes != u64::from(bytes.length()) {
        // Length mismatches are rejected by the protocol before finiteness,
        // but their observation must still authenticate the exact supplied
        // bytes rather than a partial declared layout.
        hasher = blake3::Hasher::new();
        let mut offset = 0_u32;
        while offset < bytes.length() {
            let end = offset
                .saturating_add(JS_BRIDGE_CHUNK_BYTES)
                .min(bytes.length());
            let length = (end - offset) as usize;
            bytes.subarray(offset, end).copy_to(&mut scratch[..length]);
            hasher.update(&scratch[..length]);
            offset = end;
        }
        all_finite = false;
    }
    Ok(VisionStackShardObservation {
        id: id.to_owned(),
        bytes: u64::from(bytes.length()),
        blake3: hasher.finalize().to_hex().to_string(),
        all_finite,
    })
}

fn destroy_vision_qkv_web_layer_weights(weights: &[wgpu::Buffer]) {
    for weight in weights {
        weight.destroy();
    }
}

fn vision_stack_streaming_tensor_range(
    range: VisionStackStreamingWeightRange,
) -> VisionStackTensorRange {
    VisionStackTensorRange {
        offset: range.offset_bytes(),
        bytes: range.length_bytes(),
    }
}

fn upload_js_range(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    source: &js_sys::Uint8Array,
    range: VisionStackTensorRange,
) -> Result<(), String> {
    let chunk_bytes = u64::from(JS_BRIDGE_CHUNK_BYTES);
    let end = range
        .offset
        .checked_add(range.bytes)
        .ok_or_else(|| "vision-stack upload range overflowed".to_owned())?;
    if end > u64::from(source.length()) || range.bytes != buffer.size() {
        return Err(format!(
            "vision-stack upload range {}..{end} does not match source {} or buffer {}",
            range.offset,
            source.length(),
            buffer.size()
        ));
    }
    let scratch_len = usize::try_from(range.bytes.min(chunk_bytes))
        .map_err(|_| "vision-stack upload scratch is too large".to_owned())?;
    let mut scratch = vec![0_u8; scratch_len];
    let mut copied = 0_u64;
    while copied < range.bytes {
        let length = (range.bytes - copied).min(chunk_bytes);
        let source_start = range
            .offset
            .checked_add(copied)
            .ok_or_else(|| "vision-stack upload source offset overflowed".to_owned())?;
        let source_end = source_start
            .checked_add(length)
            .ok_or_else(|| "vision-stack upload source end overflowed".to_owned())?;
        let source_start = u32::try_from(source_start)
            .map_err(|_| "vision-stack upload source offset exceeds Uint8Array".to_owned())?;
        let source_end = u32::try_from(source_end)
            .map_err(|_| "vision-stack upload source end exceeds Uint8Array".to_owned())?;
        let length = usize::try_from(length)
            .map_err(|_| "vision-stack upload chunk is too large".to_owned())?;
        source
            .subarray(source_start, source_end)
            .copy_to(&mut scratch[..length]);
        queue.write_buffer(buffer, copied, &scratch[..length]);
        copied = copied
            .checked_add(length as u64)
            .ok_or_else(|| "vision-stack upload cursor overflowed".to_owned())?;
    }
    Ok(())
}

fn write_vision_stack_hardening_patterns(
    queue: &wgpu::Queue,
    arena: &wgpu::Buffer,
    plan: &VisionStackMemoryHardeningPlan,
) -> Result<(), String> {
    const MAX_CHUNK_BYTES: u64 = 1024 * 1024;

    write_vision_stack_u32_pattern(
        queue,
        arena,
        0,
        plan.guard_bytes(),
        VISION_STACK_PREFIX_CANARY_U32,
    )?;

    let chunk_bytes = plan.logical_scratch_bytes().min(MAX_CHUNK_BYTES);
    let chunk_words = usize::try_from(chunk_bytes / 4)
        .map_err(|_| "vision-stack poison upload chunk is too large".to_owned())?;
    let poison = vec![VISION_STACK_SCRATCH_POISON_U32; chunk_words];
    let poison_bytes = bytemuck::cast_slice(&poison);
    let mut copied = 0_u64;
    while copied < plan.logical_scratch_bytes() {
        let bytes = (plan.logical_scratch_bytes() - copied).min(MAX_CHUNK_BYTES);
        let length = usize::try_from(bytes)
            .map_err(|_| "vision-stack poison upload length is too large".to_owned())?;
        let offset = plan
            .scratch_logical_offset()
            .checked_add(copied)
            .ok_or_else(|| "vision-stack poison upload offset overflowed".to_owned())?;
        queue.write_buffer(arena, offset, &poison_bytes[..length]);
        copied = copied
            .checked_add(bytes)
            .ok_or_else(|| "vision-stack poison upload cursor overflowed".to_owned())?;
    }

    write_vision_stack_u32_pattern(
        queue,
        arena,
        plan.scratch_suffix_offset(),
        plan.guard_bytes(),
        VISION_STACK_SUFFIX_CANARY_U32,
    )
}

fn write_vision_stack_u32_pattern(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: u64,
    bytes: u64,
    pattern: u32,
) -> Result<(), String> {
    let words = usize::try_from(bytes / 4)
        .map_err(|_| "vision-stack canary upload is too large".to_owned())?;
    let contents = vec![pattern; words];
    queue.write_buffer(buffer, offset, bytemuck::cast_slice(&contents));
    Ok(())
}

fn write_uniform_slot(
    contents: &mut [u8],
    stride: u64,
    slot: usize,
    words: [u32; 4],
) -> Result<(), String> {
    let offset = u64::try_from(slot)
        .map_err(|_| "vision-stack uniform slot overflowed".to_owned())?
        .checked_mul(stride)
        .ok_or_else(|| "vision-stack uniform offset overflowed".to_owned())?;
    let offset = usize::try_from(offset)
        .map_err(|_| "vision-stack uniform offset is too large".to_owned())?;
    let end = offset
        .checked_add(VISION_LAYER_UNIFORM_BYTES as usize)
        .ok_or_else(|| "vision-stack uniform end overflowed".to_owned())?;
    if end > contents.len() {
        return Err("vision-stack uniform slot exceeds its arena".to_owned());
    }
    for (index, word) in words.into_iter().enumerate() {
        let start = offset + index * 4;
        contents[start..start + 4].copy_from_slice(&word.to_le_bytes());
    }
    Ok(())
}

async fn await_queue_completion(queue: &wgpu::Queue) -> Result<(), String> {
    let (sender, receiver) = oneshot::channel();
    queue.on_submitted_work_done(move || {
        let _ = sender.send(());
    });
    receiver
        .await
        .map_err(|error| format!("GPU queue completion callback was canceled: {error}"))
}

fn build_legacy_qkv_browser_session(
    prepared: BrowserVisionStackPreparedSession,
    qkv_selection: VisionQkvStackSelection,
    qkv_selection_evidence: VisionQkvSelectionEvidencePropagation,
    before_buffer_allocations: u64,
    before_submissions: u64,
) -> BrowserVisionStackSession {
    let BrowserVisionStackPreparedSession {
        protocol,
        plan,
        layer_plan,
        weight_plan,
        fp16_qkv_plan,
        activation_strategy,
        activation_layout,
        static_plan,
        memory_hardening,
        storage_alignment,
        shader_sources,
    } = prepared;
    BrowserVisionStackSession {
        protocol,
        plan,
        layer_plan,
        weight_plan,
        fp16_qkv_plan,
        qkv_selection,
        qkv_physical_execution: None,
        qkv_physical_commands: None,
        qkv_selection_evidence: Some(qkv_selection_evidence),
        qkv_execution_evidence_plan: None,
        activation_strategy,
        activation_layout,
        static_plan,
        memory_hardening,
        storage_alignment,
        shader_sources,
        spatial_rope: None,
        resident_weights: None,
        before_buffer_allocations,
        before_submissions,
        gpu: None,
    }
}

fn vision_stack_status_json(
    session: &BrowserVisionStackSession,
    include_plan: bool,
) -> Result<String, crate::VisionStackEvidenceError> {
    let record = crate::build_vision_stack_legacy_status_record(
        session.protocol.phase(),
        session.protocol.next_shard_id(),
        &session.plan,
        session.activation_strategy,
        session.activation_layout.as_ref(),
        session.memory_hardening.as_ref(),
        session.storage_alignment,
        include_plan,
    )?;
    crate::serialize_vision_stack_legacy_status_json(&record)
}

fn vision_stack_qkv_status_json(
    session: &BrowserVisionStackSession,
    include_plan: bool,
) -> Result<String, crate::VisionStackEvidenceError> {
    let legacy_status = crate::build_vision_stack_legacy_status_record(
        session.protocol.phase(),
        session.protocol.next_shard_id(),
        &session.plan,
        session.activation_strategy,
        session.activation_layout.as_ref(),
        session.memory_hardening.as_ref(),
        session.storage_alignment,
        include_plan,
    )?;
    let qkv_execution = BrowserVisionQkvBeginExecutionEvidence::from_plan(
        session.qkv_execution_evidence_plan.as_ref(),
    );
    let selection_evidence = session.qkv_selection_evidence.as_ref().unwrap();
    let evidence = selection_evidence.additive_begin_evidence(qkv_execution.as_ref());
    let serialized_json = serialize_vision_stack_qkv_begin_status_json(&legacy_status, evidence)?;
    Ok(serialized_json)
}

fn vision_stack_qkv_diagnostics_json(
    legacy_diagnostics: &VisionStackLegacyDiagnosticsRecord,
    session: &BrowserVisionStackSession,
    canary_results: &[bool],
) -> Result<String, crate::VisionStackEvidenceError> {
    let selection_option = session.qkv_selection_evidence.as_ref();
    match selection_option {
        Some(selection_evidence) => {
            let qkv_execution = BrowserVisionQkvFinalExecutionEvidence::from_verified_plan(
                session.qkv_execution_evidence_plan.as_ref(),
                canary_results,
            )?;
            let evidence = selection_evidence.final_diagnostics_evidence(qkv_execution.as_ref());
            crate::serialize_vision_stack_qkv_final_diagnostics_json(legacy_diagnostics, evidence)
        }
        None => crate::serialize_vision_stack_legacy_diagnostics_json(legacy_diagnostics),
    }
}

fn verify_mapped_qkv_canaries(
    plan: Option<&BrowserVisionQkvExecutionEvidencePlan>,
    qkv_start: usize,
    mapped: &[u8],
) -> Result<Vec<bool>, BrowserVisionStackError> {
    let Some(plan) = plan else {
        return Ok(Vec::new());
    };
    let end = plan.canaries.iter().try_fold(qkv_start, |end, canary| {
        let byte_length = usize::try_from(canary.byte_length)
            .map_err(|_| BrowserVisionStackError("Q/K/V canary length is too large".to_owned()))?;
        end.checked_add(byte_length)
            .ok_or_else(|| BrowserVisionStackError("Q/K/V canary range overflowed".to_owned()))
    })?;
    if mapped.len() != end {
        return Err(BrowserVisionStackError(format!(
            "mapped Q/K/V readback length {} differs from expected {end}",
            mapped.len()
        )));
    }
    let mut cursor = qkv_start;
    let mut results = Vec::new();
    for canary in &plan.canaries {
        let byte_length = usize::try_from(canary.byte_length)
            .map_err(|_| BrowserVisionStackError("Q/K/V canary length is too large".to_owned()))?;
        let canary_end = cursor
            .checked_add(byte_length)
            .ok_or_else(|| BrowserVisionStackError("Q/K/V canary range overflowed".to_owned()))?;
        let bytes = mapped.get(cursor..canary_end).ok_or_else(|| {
            BrowserVisionStackError("Q/K/V canary range exceeds mapped readback".to_owned())
        })?;
        if !bytes.len().is_multiple_of(4) {
            return Err(BrowserVisionStackError(
                "Q/K/V canary length is not a whole u32 sequence".to_owned(),
            ));
        }
        results.push(bytes.chunks_exact(4).all(|word| {
            u32::from_le_bytes(word.try_into().expect("Q/K/V canary word has four bytes"))
                == VISION_QKV_CANARY_U32
        }));
        cursor = canary_end;
    }
    if cursor != end {
        return Err(BrowserVisionStackError(
            "Q/K/V canary ranges do not consume their sealed readback region".to_owned(),
        ));
    }
    Ok(results)
}

async fn map_read(buffer: &wgpu::Buffer, range: std::ops::Range<u64>) -> Result<(), String> {
    let (sender, receiver) = oneshot::channel();
    buffer.map_async(wgpu::MapMode::Read, range, move |result| {
        let _ = sender.send(result);
    });
    receiver
        .await
        .map_err(|error| format!("GPU mapping callback was canceled: {error}"))?
        .map_err(|error| format!("GPU output mapping failed: {error}"))
}

fn read_f32_buffer(buffer: &wgpu::Buffer, elements: usize) -> Result<Vec<f32>, String> {
    let mapped = buffer
        .get_mapped_range(..)
        .map_err(|error| format!("cannot view mapped GPU output: {error}"))?;
    let values = mapped
        .chunks_exact(4)
        .take(elements)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 occupies four bytes")))
        .collect();
    drop(mapped);
    buffer.unmap();
    Ok(values)
}

fn invocation_input_bytes<'a>(input: &InvocationInput<'a>) -> &'a [u8] {
    match input {
        InvocationInput::F32(values) => bytemuck::cast_slice(values),
        InvocationInput::U32(values) => bytemuck::cast_slice(values),
    }
}

fn require_nonempty(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() {
        Err(format!("{label} must not be empty"))
    } else {
        Ok(())
    }
}

fn to_json<T: Serialize>(value: &T) -> Result<String, JsValue> {
    serde_json::to_string(value)
        .map_err(|error| js_error(format!("cannot serialize browser runtime report: {error}")))
}

fn js_error(message: impl AsRef<str>) -> JsValue {
    js_sys::Error::new(message.as_ref()).into()
}
