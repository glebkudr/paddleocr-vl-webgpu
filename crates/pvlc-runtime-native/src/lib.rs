//! Native `wgpu` execution for the fixed PaddleOCR-VL FP32 primitive layer.

use std::{
    collections::BTreeMap,
    env, fmt, fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use pvlc_passes::{
    VisionQkvPhysicalExecutionSpec, VisionQkvStackSelection, bind_vision_qkv_physical_execution,
    prepare_vision_qkv_stack_execution,
};
use pvlc_runtime_core::{
    ComputeDispatchLimits, DecoderCachedGqaInvocation, DecoderCachedGqaPlan, DecoderCachedGqaStage,
    InvocationInput, InvocationPlan, KernelId, KernelInvocation, ProjectorInvocation,
    ProjectorPlan, ProjectorReadback, ProjectorStage, VISION_QKV_CANARY_U32,
    VisionEncoderLayerInvocation, VisionEncoderLayerParameters, VisionEncoderLayerPlan,
    VisionEncoderLayerStage, VisionEncoderStackInvocation, VisionEncoderStackPlan,
    VisionQkvAttentionBindingEvidence, VisionQkvBindGroupCreationEvidence,
    VisionQkvBufferBindingEvidence, VisionQkvCanaryEvidence,
    VisionQkvCommandEncoderCreationEvidence, VisionQkvCopyEvidence, VisionQkvCopyPurpose,
    VisionQkvDispatchEvidence, VisionQkvExecutionPolicy, VisionQkvFusedInvocation,
    VisionQkvFusedTargetLimits, VisionQkvMapEvidence, VisionQkvMapPurpose,
    VisionQkvPipelineCreationEvidence, VisionQkvReadbackLayout, VisionQkvReadbackRequirements,
    VisionQkvSelectionOutcome, VisionQkvStackExecutionEvidence, VisionQkvStackStage,
    VisionQkvWorkspaceEvidence, VisionRopeSpecialization, VisionStackActivationLayoutConfig,
    VisionStackActivationStrategy, VisionStackScratchAllocation, plan_vision_qkv_readback_layout,
};
pub use pvlc_runtime_core::{
    DecoderKvSessionDescriptor, DecoderKvSessionStep, VisionLayerReadback,
};
use wgpu::util::DeviceExt;

mod decoder_kv_session;
pub use decoder_kv_session::NativeDecoderKvSession;

#[cfg(test)]
mod m7c2a_tests;

const ERROR_SCOPE_ORDER: [ErrorScopeKind; 3] = [
    ErrorScopeKind::Internal,
    ErrorScopeKind::OutOfMemory,
    ErrorScopeKind::Validation,
];
const CHECKED_SCOPE_ORDER: [ErrorScopeKind; 3] = [
    ErrorScopeKind::Validation,
    ErrorScopeKind::OutOfMemory,
    ErrorScopeKind::Internal,
];
const GPU_WAIT_TIMEOUT: Duration = Duration::from_secs(30);
const TIMESTAMP_ATTEMPTS: usize = 8;
const CAPTURE_ARTIFACT_TIMEOUT: Duration = Duration::from_secs(10);

/// Reads the process thermal state through the native platform authority.
#[cfg(target_os = "macos")]
pub fn native_system_thermal_state_v1() -> Result<&'static str, String> {
    use objc2_foundation::{NSProcessInfo, NSProcessInfoThermalState};

    match NSProcessInfo::processInfo().thermalState() {
        NSProcessInfoThermalState::Nominal => Ok("nominal"),
        NSProcessInfoThermalState::Fair => Ok("fair"),
        NSProcessInfoThermalState::Serious => Ok("serious"),
        NSProcessInfoThermalState::Critical => Ok("critical"),
        state => Err(format!("unsupported ProcessInfo thermal state {}", state.0)),
    }
}

/// Reports that the macOS process thermal-state authority is unavailable on this target.
#[cfg(not(target_os = "macos"))]
pub fn native_system_thermal_state_v1() -> Result<&'static str, String> {
    Err("ProcessInfo.thermalState is unavailable on this platform".to_owned())
}
const VISION_LAYER_KERNELS: [KernelId; 5] = [
    KernelId::LayerNormF32,
    KernelId::VisionPatchProjectionF32,
    KernelId::VisionAttentionF32,
    KernelId::AddF32,
    KernelId::GeluTanhF32,
];
const VISION_LAYER_UNIFORM_BYTES: u64 = 16;
const PROJECTOR_KERNELS: [KernelId; 4] = [
    KernelId::LayerNormF32,
    KernelId::GeluErfF32,
    KernelId::VisionPatchProjectionF32,
    KernelId::ProjectorMerge2x2F32,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendKind {
    Noop,
    Vulkan,
    Metal,
    Dx12,
    Gl,
    BrowserWebGpu,
}

impl From<wgpu::Backend> for BackendKind {
    fn from(backend: wgpu::Backend) -> Self {
        match backend {
            wgpu::Backend::Noop => Self::Noop,
            wgpu::Backend::Vulkan => Self::Vulkan,
            wgpu::Backend::Metal => Self::Metal,
            wgpu::Backend::Dx12 => Self::Dx12,
            wgpu::Backend::Gl => Self::Gl,
            wgpu::Backend::BrowserWebGpu => Self::BrowserWebGpu,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeCapabilities {
    pub adapter_name: String,
    pub backend: BackendKind,
    pub timestamp_query: bool,
    pub min_storage_buffer_offset_alignment: u32,
    pub max_storage_buffer_binding_size: u64,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_size_y: u32,
    pub max_compute_workgroup_size_z: u32,
    pub max_compute_workgroup_storage_size: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_buffer_size: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeCounters {
    pub buffer_allocations: u64,
    pub submissions: u64,
    pub pipeline_creations: u64,
    pub bind_group_creations: u64,
    pub command_encoder_creations: u64,
    pub dispatch_encodings: u64,
    pub buffer_copy_encodings: u64,
    pub map_requests: u64,
    pub queue_writes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorScopeKind {
    Validation,
    OutOfMemory,
    Internal,
}

impl ErrorScopeKind {
    const fn error_filter(self) -> wgpu::ErrorFilter {
        match self {
            Self::Validation => wgpu::ErrorFilter::Validation,
            Self::OutOfMemory => wgpu::ErrorFilter::OutOfMemory,
            Self::Internal => wgpu::ErrorFilter::Internal,
        }
    }

    const fn error_code(self) -> RuntimeErrorCode {
        match self {
            Self::Validation => RuntimeErrorCode::Validation,
            Self::OutOfMemory => RuntimeErrorCode::OutOfMemory,
            Self::Internal => RuntimeErrorCode::Internal,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeEvent {
    ScopePushed(ErrorScopeKind),
    ScopePopped {
        scope: ErrorScopeKind,
        captured_error: bool,
    },
    BufferAllocated {
        label: String,
        bytes: u64,
    },
    SubmissionQueued {
        submission: u64,
        command_buffers: u32,
    },
    ReadbackMapRequested {
        label: String,
        bytes: u64,
    },
    PipelineCreated {
        kernel: KernelId,
        shader_blake3: [u8; 32],
    },
    BindGroupCreated {
        layer: Option<usize>,
        stage: VisionQkvStackStage,
        bindings: Vec<VisionQkvBufferBindingEvidence>,
    },
    CommandEncoderCreated {
        label: String,
    },
    DispatchEncoded {
        ordinal: usize,
        layer: Option<usize>,
        stage: VisionQkvStackStage,
        kernel: KernelId,
        workgroups: [u32; 3],
    },
    BufferCopyEncoded {
        ordinal: usize,
        source_buffer_identity: u64,
        source_offset: u64,
        destination_buffer_identity: u64,
        destination_offset: u64,
        byte_length: u64,
        purpose: VisionQkvCopyPurpose,
        after_dispatch_ordinal: usize,
    },
    MapRequested {
        purpose: VisionQkvMapPurpose,
        buffer_identity: u64,
        byte_offset: u64,
        byte_length: u64,
    },
    QueueBufferWritten {
        label: String,
        buffer_identity: u64,
        byte_offset: u64,
        byte_length: u64,
    },
    DecoderCommandEncoderCreated {
        label: String,
    },
    DecoderComputePassEncoded {
        pass_index: usize,
        stage: DecoderCachedGqaStage,
    },
    DecoderDispatchEncoded {
        ordinal: usize,
        stage: DecoderCachedGqaStage,
        kernel: KernelId,
        workgroups: [u32; 3],
    },
    DecoderBufferCopyEncoded {
        ordinal: usize,
        source_buffer_identity: u64,
        source_offset: u64,
        destination_buffer_identity: u64,
        destination_offset: u64,
        byte_length: u64,
    },
    DecoderMapRequested {
        buffer_identity: u64,
        byte_offset: u64,
        byte_length: u64,
    },
    CanaryChecked {
        buffer_identity: u64,
        canaries: Vec<VisionQkvCanaryEvidence>,
    },
}

pub trait RuntimeObserver: Send + Sync {
    fn on_event(&self, event: RuntimeEvent);
}

#[derive(Default)]
pub struct NativeOptions {
    pub observer: Option<Arc<dyn RuntimeObserver>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeErrorCode {
    AdapterUnavailable,
    DeviceRequest,
    InvalidInvocation,
    Validation,
    OutOfMemory,
    Internal,
    Operation,
    Mapping,
    Capture,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeError {
    code: RuntimeErrorCode,
    scope: Option<ErrorScopeKind>,
    message: String,
}

impl RuntimeError {
    fn new(
        code: RuntimeErrorCode,
        scope: Option<ErrorScopeKind>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            scope,
            message: message.into(),
        }
    }

    #[must_use]
    pub fn operation(message: impl Into<String>) -> Self {
        Self::new(RuntimeErrorCode::Operation, None, message)
    }

    fn mapping(message: impl Into<String>) -> Self {
        Self::new(RuntimeErrorCode::Mapping, None, message)
    }

    fn capture(message: impl Into<String>) -> Self {
        Self::new(RuntimeErrorCode::Capture, None, message)
    }

    fn scoped(scope: ErrorScopeKind, message: impl Into<String>) -> Self {
        Self::new(scope.error_code(), Some(scope), message)
    }

    fn with_context(mut self, context: impl AsRef<str>) -> Self {
        self.message = format!("{}: {}", context.as_ref(), self.message);
        self
    }

    #[must_use]
    pub const fn code(&self) -> RuntimeErrorCode {
        self.code
    }

    #[must_use]
    pub const fn scope(&self) -> Option<ErrorScopeKind> {
        self.scope
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "native runtime error {:?}: {}",
            self.code, self.message
        )
    }
}

impl std::error::Error for RuntimeError {}

pub trait ErrorScopeDriver {
    fn push_scope(&mut self, scope: ErrorScopeKind);
    fn pop_scope(&mut self, scope: ErrorScopeKind) -> Option<String>;
}

/// Runs an operation under all three WebGPU error scopes and always unwinds them.
///
/// WebGPU validation errors have priority over OOM and internal errors because
/// the innermost scope is inspected first. If both the operation and a scope
/// fail, the operation failure remains attached as diagnostic context.
pub fn drive_error_scopes<D, F, T>(driver: &mut D, operation: F) -> Result<T, RuntimeError>
where
    D: ErrorScopeDriver,
    F: FnOnce() -> Result<T, RuntimeError>,
{
    for scope in ERROR_SCOPE_ORDER {
        driver.push_scope(scope);
    }

    let operation_result = operation();
    let mut captured = None;
    for scope in CHECKED_SCOPE_ORDER {
        if let Some(message) = driver.pop_scope(scope)
            && captured.is_none()
        {
            captured = Some((scope, message));
        }
    }

    if let Some((scope, message)) = captured {
        let message = match operation_result.as_ref().err() {
            Some(operation_error) => format!("{message}; operation also failed: {operation_error}"),
            None => message,
        };
        return Err(RuntimeError::scoped(scope, message));
    }
    operation_result
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuTimestamp {
    pub begin_ticks: u64,
    pub end_ticks: u64,
    pub period_ns: f64,
    pub duration_ns: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelDiagnostics {
    pub kernel: KernelId,
    pub checked_error_scopes: [ErrorScopeKind; 3],
    pub captured_errors: Vec<String>,
    pub queue_wall_time_ns: u64,
    pub timestamp: Option<GpuTimestamp>,
    pub shader_blake3: [u8; 32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelExecution {
    pub values: Vec<f32>,
    pub diagnostics: KernelDiagnostics,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DecoderCachedGqaBufferRole {
    Query,
    AppendedKey,
    AppendedValue,
    KeyCache,
    ValueCache,
    AttentionOutput,
    AppendUniform,
    AttentionUniform,
    Readback,
    /// M7o2 amendment: split-K partials scratch plane of the persistent
    /// decoder KV session.
    SplitPartials,
    /// M7o2 amendment: position-dependent uniform of the split partial
    /// dispatch.
    SplitPartialUniform,
    /// M7o2 amendment: position-dependent uniform of the split merge
    /// dispatch.
    SplitMergeUniform,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderCachedGqaBufferEvidence {
    pub role: DecoderCachedGqaBufferRole,
    pub buffer_identity: u64,
    pub allocation_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderCachedGqaBindingEvidence {
    pub binding: u32,
    pub buffer_identity: u64,
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderCachedGqaBindGroupEvidence {
    pub stage: DecoderCachedGqaStage,
    pub bindings: Vec<DecoderCachedGqaBindingEvidence>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderCachedGqaDispatchEvidence {
    pub ordinal: usize,
    pub stage: DecoderCachedGqaStage,
    pub kernel: KernelId,
    pub workgroups: [u32; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderCachedGqaCopyPurpose {
    Attention,
    KeyCache,
    ValueCache,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderCachedGqaCopyEvidence {
    pub ordinal: usize,
    pub source_buffer_identity: u64,
    pub source_offset: u64,
    pub destination_buffer_identity: u64,
    pub destination_offset: u64,
    pub byte_length: u64,
    pub purpose: DecoderCachedGqaCopyPurpose,
    pub after_dispatch_ordinal: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderCachedGqaMapEvidence {
    pub buffer_identity: u64,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub after_copy_ordinal: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecoderCachedGqaOperationEvidence {
    pub buffers: Vec<DecoderCachedGqaBufferEvidence>,
    pub bind_groups: Vec<DecoderCachedGqaBindGroupEvidence>,
    pub dispatches: Vec<DecoderCachedGqaDispatchEvidence>,
    pub copies: Vec<DecoderCachedGqaCopyEvidence>,
    pub maps: Vec<DecoderCachedGqaMapEvidence>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderCachedGqaDiagnostics {
    pub checked_error_scopes: [ErrorScopeKind; 3],
    pub captured_errors: Vec<String>,
    pub queue_wall_time_ns: u64,
    pub shader_blake3: BTreeMap<KernelId, [u8; 32]>,
    pub dispatch_stages: [DecoderCachedGqaStage; 2],
    pub dispatch_count: usize,
    pub compute_pass_count: usize,
    pub command_buffer_count: u32,
    pub submission_count: u64,
    pub readback_buffer_count: u32,
    pub readback_map_count: u32,
    pub readback_bytes: u64,
    pub operation_evidence: DecoderCachedGqaOperationEvidence,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderCachedGqaExecution {
    pub attention: Vec<f32>,
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
    pub cache_tokens: u32,
    pub cache_capacity: u32,
    pub diagnostics: DecoderCachedGqaDiagnostics,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderKvSessionCreationDiagnostics {
    pub initial_cache_tokens: u32,
    pub cache_capacity: u32,
    pub checked_error_scopes: [ErrorScopeKind; 3],
    pub captured_errors: Vec<String>,
    pub shader_blake3: BTreeMap<KernelId, [u8; 32]>,
    pub buffers: Vec<DecoderCachedGqaBufferEvidence>,
    pub bind_groups: Vec<DecoderCachedGqaBindGroupEvidence>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecoderKvSessionEffect {
    QueueWrite {
        ordinal: usize,
        role: DecoderCachedGqaBufferRole,
        buffer_identity: u64,
        byte_offset: u64,
        byte_length: u64,
    },
    Dispatch {
        ordinal: usize,
        stage: DecoderCachedGqaStage,
        kernel: KernelId,
        workgroups: [u32; 3],
    },
    CopyAttention {
        ordinal: usize,
        source_buffer_identity: u64,
        destination_buffer_identity: u64,
        byte_length: u64,
    },
    Submit {
        ordinal: usize,
        command_buffer_count: u32,
    },
    MapAttention {
        ordinal: usize,
        buffer_identity: u64,
        byte_length: u64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderKvSessionStepDiagnostics {
    pub cache_tokens_before: u32,
    pub cache_tokens_after: u32,
    pub checked_error_scopes: [ErrorScopeKind; 3],
    pub captured_errors: Vec<String>,
    pub queue_wall_time_ns: u64,
    pub shader_blake3: BTreeMap<KernelId, [u8; 32]>,
    pub dispatch_count: usize,
    pub compute_pass_count: usize,
    pub command_buffer_count: u32,
    pub copy_count: usize,
    pub submission_count: u64,
    pub map_count: usize,
    pub readback_bytes: u64,
    pub effects: Vec<DecoderKvSessionEffect>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderKvSessionStepExecution {
    pub attention: Vec<f32>,
    pub cache_tokens: u32,
    pub diagnostics: DecoderKvSessionStepDiagnostics,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderKvSessionSnapshotDiagnostics {
    pub checked_error_scopes: [ErrorScopeKind; 3],
    pub captured_errors: Vec<String>,
    pub queue_wall_time_ns: u64,
    pub readback_buffer_identity: u64,
    pub readback_bytes: u64,
    pub copy_count: usize,
    pub command_buffer_count: u32,
    pub submission_count: u64,
    pub map_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecoderKvSessionSnapshot {
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
    pub cache_tokens: u32,
    pub cache_capacity: u32,
    pub diagnostics: DecoderKvSessionSnapshotDiagnostics,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisionLayerDiagnostics {
    pub checked_error_scopes: [ErrorScopeKind; 3],
    pub captured_errors: Vec<String>,
    pub queue_wall_time_ns: u64,
    pub timestamp: Option<GpuTimestamp>,
    pub timestamp_fresh: Option<bool>,
    pub shader_blake3: BTreeMap<KernelId, [u8; 32]>,
    pub dispatch_stages: [VisionEncoderLayerStage; 12],
    pub rope_specialization: VisionRopeSpecialization,
    pub submission_count: u64,
    pub command_buffer_count: u32,
    pub buffer_allocation_count: u64,
    pub readback_buffer_count: u32,
    pub readback_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisionLayerExecution {
    pub checkpoints: BTreeMap<VisionEncoderLayerStage, Vec<f32>>,
    pub diagnostics: VisionLayerDiagnostics,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisionStackDiagnostics {
    pub checked_error_scopes: [ErrorScopeKind; 3],
    pub captured_errors: Vec<String>,
    pub queue_wall_time_ns: u64,
    pub timestamp: Option<GpuTimestamp>,
    pub timestamp_fresh: Option<bool>,
    pub shader_blake3: BTreeMap<KernelId, [u8; 32]>,
    pub rope_specialization: VisionRopeSpecialization,
    pub layer_count: usize,
    pub checkpoint_layers: Vec<usize>,
    pub dispatch_count: usize,
    pub compute_pass_count: usize,
    pub submission_count: u64,
    pub command_buffer_count: u32,
    pub buffer_allocation_count: u64,
    pub weight_buffer_count: usize,
    pub activation_strategy: VisionStackActivationStrategy,
    pub activation_buffer_count: usize,
    pub activation_arena_bytes: u64,
    pub scratch_arena_bytes: u64,
    pub main_buffers_bytes: u64,
    pub scratch_allocations: Vec<VisionStackScratchAllocation>,
    pub readback_buffer_count: u32,
    pub readback_map_count: u32,
    pub readback_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisionStackExecution {
    pub checkpoints: BTreeMap<usize, Vec<f32>>,
    pub output: Vec<f32>,
    pub diagnostics: VisionStackDiagnostics,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisionQkvStackExecution {
    pub checkpoints: BTreeMap<usize, Vec<f32>>,
    pub output: Vec<f32>,
    pub diagnostics: VisionStackDiagnostics,
    pub evidence: VisionQkvStackExecutionEvidence,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProjectorDiagnostics {
    pub checked_error_scopes: [ErrorScopeKind; 3],
    pub captured_errors: Vec<String>,
    pub queue_wall_time_ns: u64,
    pub timestamp: Option<GpuTimestamp>,
    pub timestamp_fresh: Option<bool>,
    pub shader_blake3: BTreeMap<KernelId, [u8; 32]>,
    pub dispatch_stages: [ProjectorStage; 5],
    pub submission_count: u64,
    pub command_buffer_count: u32,
    pub compute_pass_count: usize,
    pub dispatch_count: usize,
    pub buffer_allocation_count: u64,
    pub readback_buffer_count: u32,
    pub readback_map_count: u32,
    pub readback_bytes: u64,
    pub resident_intermediate_bytes: u64,
    pub resident_weight_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProjectorExecution {
    pub checkpoints: BTreeMap<ProjectorStage, Vec<f32>>,
    pub diagnostics: ProjectorDiagnostics,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineValidationReport {
    pub validated_kernels: Vec<KernelId>,
    pub captured_errors: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalCaptureArtifact {
    pub path: PathBuf,
    pub file_count: u64,
    pub byte_count: u64,
}

pub struct NativeRuntime {
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    capabilities: NativeCapabilities,
    observer: Option<Arc<dyn RuntimeObserver>>,
    buffer_allocations: AtomicU64,
    submissions: AtomicU64,
    pipeline_creations: AtomicU64,
    bind_group_creations: AtomicU64,
    command_encoder_creations: AtomicU64,
    dispatch_encodings: AtomicU64,
    buffer_copy_encodings: AtomicU64,
    map_requests: AtomicU64,
    queue_writes: AtomicU64,
    execution_lock: Mutex<()>,
    capture_lock: Mutex<()>,
    pipelines: Mutex<BTreeMap<KernelId, wgpu::ComputePipeline>>,
    timestamp_resources: Option<TimestampResources>,
    last_timestamp: Mutex<Option<(u64, u64)>>,
    _uncaptured_errors: Arc<Mutex<Vec<String>>>,
}

struct PreparedVisionStackActivation {
    strategy: VisionStackActivationStrategy,
    activation_buffer_count: usize,
    scratch_arena_bytes: u64,
    main_buffers_bytes: u64,
    total_activation_bytes: u64,
    scratch_allocations: Vec<VisionStackScratchAllocation>,
}

#[derive(Clone, Copy)]
pub(crate) struct VisionQkvExecutionAllocationPreflight {
    pub(crate) semantic_readback_bytes: u64,
    pub(crate) canary_readback_bytes: u64,
    pub(crate) workspace_allocation_bytes: u64,
    pub(crate) max_buffer_size: u64,
    pub(crate) max_host_elements: u64,
}

fn vision_qkv_preflight_validation(message: impl Into<String>) -> RuntimeError {
    RuntimeError::new(RuntimeErrorCode::Validation, None, message)
}

pub(crate) fn preflight_vision_qkv_execution_allocations(
    input: VisionQkvExecutionAllocationPreflight,
) -> Result<PreparedVisionQkvExecutionAllocations, RuntimeError> {
    let layout = plan_vision_qkv_readback_layout(VisionQkvReadbackRequirements {
        semantic_readback_bytes: input.semantic_readback_bytes,
        scratch_canary_readback_bytes: 0,
        qkv_canary_readback_bytes: input.canary_readback_bytes,
        workspace_allocation_bytes: input.workspace_allocation_bytes,
        max_buffer_size: input.max_buffer_size,
        max_host_elements: input.max_host_elements,
    })
    .map_err(|error| vision_qkv_preflight_validation(error.to_string()))?;
    Ok(PreparedVisionQkvExecutionAllocations {
        layout,
        total_readback_bytes: layout.total_readback_bytes(),
        readback_f32_elements: layout.readback_f32_elements(),
        workspace_u32_words: layout.workspace_u32_words(),
    })
}

struct PreparedVisionQkvExecution {
    policy: VisionQkvExecutionPolicy,
    outcome: VisionQkvSelectionOutcome,
    qkv_physical_execution: Option<VisionQkvPhysicalExecutionSpec>,
    total_readback_bytes: u64,
    readback_f32_elements: usize,
    workspace_u32_words: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct PreparedVisionQkvExecutionAllocations {
    pub(crate) layout: VisionQkvReadbackLayout,
    pub(crate) total_readback_bytes: u64,
    pub(crate) readback_f32_elements: usize,
    pub(crate) workspace_u32_words: usize,
}

enum VisionStackScratchBuffers {
    Separate(Vec<wgpu::Buffer>),
    Static(wgpu::Buffer),
}

#[derive(Clone, Copy)]
struct StorageBufferBinding<'a> {
    buffer: &'a wgpu::Buffer,
    offset: u64,
    size: Option<wgpu::BufferSize>,
}

#[derive(Clone, Copy)]
enum SingleKernelOutputInitializer<'a> {
    Zero,
    F32(&'a [f32]),
    FillBits(u32),
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum SingleKernelTimestampPolicy {
    RequireFresh,
    SingleSubmission,
}

impl<'a> StorageBufferBinding<'a> {
    const fn entire(buffer: &'a wgpu::Buffer) -> Self {
        Self {
            buffer,
            offset: 0,
            size: None,
        }
    }

    fn slice(buffer: &'a wgpu::Buffer, offset: u64, size: u64) -> Result<Self, RuntimeError> {
        let size = wgpu::BufferSize::new(size).ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                "vision-stack scratch binding size must be nonzero",
            )
        })?;
        Ok(Self {
            buffer,
            offset,
            size: Some(size),
        })
    }

    const fn as_wgpu(self) -> wgpu::BufferBinding<'a> {
        wgpu::BufferBinding {
            buffer: self.buffer,
            offset: self.offset,
            size: self.size,
        }
    }
}

impl VisionStackScratchBuffers {
    fn binding<'a>(
        &'a self,
        allocations: &[VisionStackScratchAllocation],
        index: usize,
    ) -> Result<StorageBufferBinding<'a>, RuntimeError> {
        match self {
            Self::Separate(buffers) => buffers
                .get(index)
                .map(StorageBufferBinding::entire)
                .ok_or_else(|| {
                    RuntimeError::operation(format!("missing separate scratch buffer {index}"))
                }),
            Self::Static(arena) => {
                let allocation = allocations.get(index).ok_or_else(|| {
                    RuntimeError::operation(format!("missing static scratch allocation {index}"))
                })?;
                StorageBufferBinding::slice(arena, allocation.offset, allocation.size)
            }
        }
    }

    fn copy_source<'a>(
        &'a self,
        allocations: &[VisionStackScratchAllocation],
        index: usize,
    ) -> Result<(&'a wgpu::Buffer, u64), RuntimeError> {
        match self {
            Self::Separate(buffers) => buffers
                .get(index)
                .map(|buffer| (buffer, 0))
                .ok_or_else(|| RuntimeError::operation(format!("missing scratch buffer {index}"))),
            Self::Static(arena) => allocations
                .get(index)
                .map(|allocation| (arena, allocation.offset))
                .ok_or_else(|| {
                    RuntimeError::operation(format!("missing static scratch allocation {index}"))
                }),
        }
    }
}

struct TimestampResources {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl NativeRuntime {
    pub fn new(options: NativeOptions) -> Result<Self, RuntimeError> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|error| {
            RuntimeError::new(
                RuntimeErrorCode::AdapterUnavailable,
                None,
                format!("failed to acquire a native WebGPU adapter: {error}"),
            )
        })?;

        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();
        let required_features =
            adapter.features() & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SHADER_F16);
        let timestamp_query = required_features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("pvlc-native-device"),
            required_features,
            required_limits: adapter_limits.clone(),
            ..Default::default()
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&descriptor)).map_err(|error| {
                RuntimeError::new(
                    RuntimeErrorCode::DeviceRequest,
                    None,
                    format!("failed to create native WebGPU device: {error}"),
                )
            })?;

        let uncaptured_errors = Arc::new(Mutex::new(Vec::new()));
        let uncaptured_sink = Arc::clone(&uncaptured_errors);
        device.on_uncaptured_error(Arc::new(move |error| {
            let mut errors = uncaptured_sink
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            errors.push(error.to_string());
        }));
        let timestamp_resources = timestamp_query.then(|| TimestampResources {
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pvlc-native-timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pvlc-native-timestamp-resolve"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pvlc-native-timestamp-readback"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        });

        let runtime = Self {
            _instance: instance,
            device,
            queue,
            capabilities: NativeCapabilities {
                adapter_name: adapter_info.name,
                backend: adapter_info.backend.into(),
                timestamp_query,
                min_storage_buffer_offset_alignment: adapter_limits
                    .min_storage_buffer_offset_alignment,
                max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                max_compute_workgroups_per_dimension: adapter_limits
                    .max_compute_workgroups_per_dimension,
                max_compute_invocations_per_workgroup: adapter_limits
                    .max_compute_invocations_per_workgroup,
                max_compute_workgroup_size_x: adapter_limits.max_compute_workgroup_size_x,
                max_compute_workgroup_size_y: adapter_limits.max_compute_workgroup_size_y,
                max_compute_workgroup_size_z: adapter_limits.max_compute_workgroup_size_z,
                max_compute_workgroup_storage_size: adapter_limits
                    .max_compute_workgroup_storage_size,
                max_storage_buffers_per_shader_stage: adapter_limits
                    .max_storage_buffers_per_shader_stage,
                max_buffer_size: adapter_limits.max_buffer_size,
            },
            observer: options.observer,
            buffer_allocations: AtomicU64::new(u64::from(timestamp_query) * 2),
            submissions: AtomicU64::new(0),
            pipeline_creations: AtomicU64::new(0),
            bind_group_creations: AtomicU64::new(0),
            command_encoder_creations: AtomicU64::new(0),
            dispatch_encodings: AtomicU64::new(0),
            buffer_copy_encodings: AtomicU64::new(0),
            map_requests: AtomicU64::new(0),
            queue_writes: AtomicU64::new(0),
            execution_lock: Mutex::new(()),
            capture_lock: Mutex::new(()),
            pipelines: Mutex::new(BTreeMap::new()),
            timestamp_resources,
            last_timestamp: Mutex::new(None),
            _uncaptured_errors: uncaptured_errors,
        };
        runtime.prime_timestamp_query()?;
        Ok(runtime)
    }

    #[must_use]
    pub const fn capabilities(&self) -> &NativeCapabilities {
        &self.capabilities
    }

    /// Reports whether caller-owned observer code is installed in this runtime.
    #[must_use]
    pub const fn has_observer(&self) -> bool {
        self.observer.is_some()
    }

    #[must_use]
    pub fn counters(&self) -> RuntimeCounters {
        RuntimeCounters {
            buffer_allocations: self.buffer_allocations.load(Ordering::Relaxed),
            submissions: self.submissions.load(Ordering::Relaxed),
            pipeline_creations: self.pipeline_creations.load(Ordering::Relaxed),
            bind_group_creations: self.bind_group_creations.load(Ordering::Relaxed),
            command_encoder_creations: self.command_encoder_creations.load(Ordering::Relaxed),
            dispatch_encodings: self.dispatch_encodings.load(Ordering::Relaxed),
            buffer_copy_encodings: self.buffer_copy_encodings.load(Ordering::Relaxed),
            map_requests: self.map_requests.load(Ordering::Relaxed),
            queue_writes: self.queue_writes.load(Ordering::Relaxed),
        }
    }

    fn prime_timestamp_query(&self) -> Result<(), RuntimeError> {
        if self.timestamp_resources.is_none() {
            return Ok(());
        }
        let execution = self.run(&KernelInvocation::SiluF32 { values: vec![0.0] })?;
        debug_assert!(execution.diagnostics.timestamp.is_some());
        Ok(())
    }

    pub fn validate_all_pipelines(&self) -> Result<PipelineValidationReport, RuntimeError> {
        let mut validated_kernels = Vec::with_capacity(KernelId::ALL.len());
        for module in pvlc_wgsl::full_catalog() {
            self.validate_pipeline_source(
                module.spec.kernel.as_str(),
                module.source,
                module.spec.entry_point,
            )?;
            validated_kernels.push(module.spec.kernel);
        }
        Ok(PipelineValidationReport {
            validated_kernels,
            captured_errors: Vec::new(),
        })
    }

    pub fn capture_to_gputrace<T, F>(
        &self,
        path: &Path,
        operation: F,
    ) -> Result<(T, MetalCaptureArtifact), RuntimeError>
    where
        F: FnOnce(&Self) -> Result<T, RuntimeError>,
    {
        validate_capture_target(path)?;
        let _capture = self
            .capture_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.capabilities.backend != BackendKind::Metal {
            return Err(RuntimeError::capture(format!(
                "GPU trace capture requires Metal, but the selected backend is {:?}",
                self.capabilities.backend
            )));
        }
        if env::var("MTL_CAPTURE_ENABLED").as_deref() != Ok("1") {
            return Err(RuntimeError::capture(
                "Metal capture is disabled; launch the process with MTL_CAPTURE_ENABLED=1",
            ));
        }

        #[cfg(target_os = "macos")]
        {
            self.capture_to_gputrace_macos(path, operation)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = operation;
            Err(RuntimeError::capture(
                "Metal GPU trace capture is available only on macOS",
            ))
        }
    }

    pub fn is_metal_capture_active(&self) -> Result<bool, RuntimeError> {
        #[cfg(target_os = "macos")]
        {
            use objc2_metal::MTLCaptureManager;

            // SAFETY: Apple exposes a process-global retained capture manager.
            let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
            Ok(manager.isCapturing())
        }
        #[cfg(not(target_os = "macos"))]
        {
            Ok(false)
        }
    }

    #[cfg(target_os = "macos")]
    fn capture_to_gputrace_macos<T, F>(
        &self,
        path: &Path,
        operation: F,
    ) -> Result<(T, MetalCaptureArtifact), RuntimeError>
    where
        F: FnOnce(&Self) -> Result<T, RuntimeError>,
    {
        use objc2_foundation::{NSString, NSURL};
        use objc2_metal::{MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager};

        // SAFETY: Apple exposes a process-global retained capture manager.
        let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
        if manager.isCapturing() {
            return Err(RuntimeError::capture(
                "another Metal capture is already active in this process",
            ));
        }
        if !manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
            return Err(RuntimeError::capture(
                "this Metal device cannot write GPU trace documents",
            ));
        }

        let descriptor = MTLCaptureDescriptor::new();
        descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
        let path_text = path
            .to_str()
            .ok_or_else(|| RuntimeError::capture("Metal capture target is not valid UTF-8"))?;
        let path_string = NSString::from_str(path_text);
        let output_url = NSURL::fileURLWithPath(&path_string);
        descriptor.setOutputURL(Some(&output_url));
        let hal_device = unsafe { self.device.as_hal::<wgpu::hal::api::Metal>() }
            .ok_or_else(|| RuntimeError::capture("wgpu did not expose a Metal HAL device"))?;
        descriptor.set_capture_device(hal_device.raw_device());
        manager
            .startCaptureWithDescriptor_error(&descriptor)
            .map_err(|error| {
                RuntimeError::capture(format!("Metal refused to start GPU capture: {error}"))
            })?;
        drop(hal_device);

        struct StopCaptureOnDrop<'a>(&'a MTLCaptureManager);
        impl Drop for StopCaptureOnDrop<'_> {
            fn drop(&mut self) {
                if self.0.isCapturing() {
                    self.0.stopCapture();
                }
            }
        }
        let stop_guard = StopCaptureOnDrop(&manager);
        let operation_result = operation(self);
        drop(stop_guard);

        match operation_result {
            Ok(value) => {
                let (file_count, byte_count) = wait_for_capture_artifact(path)?;
                if file_count == 0 || byte_count == 0 {
                    remove_capture_artifact(path).ok();
                    return Err(RuntimeError::capture(
                        "Metal produced an empty GPU trace document",
                    ));
                }
                Ok((
                    value,
                    MetalCaptureArtifact {
                        path: path.to_path_buf(),
                        file_count,
                        byte_count,
                    },
                ))
            }
            Err(error) => {
                remove_capture_artifact(path).map_err(|cleanup_error| {
                    error.clone().with_context(format!(
                        "failed to remove partial capture {}: {cleanup_error}",
                        path.display()
                    ))
                })?;
                Err(error)
            }
        }
    }

    pub fn validate_pipeline_source(
        &self,
        label: &str,
        source: &str,
        entry_point: &str,
    ) -> Result<(), RuntimeError> {
        let _execution = self
            .execution_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut scopes = WgpuScopeDriver::new(self.device.clone(), self.observer.clone());
        drive_error_scopes(&mut scopes, || {
            let shader = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            let _pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: None,
                    module: &shader,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });
            Ok(())
        })
        .map_err(|error| error.with_context(format!("pipeline {label}")))
    }

    pub fn run(&self, invocation: &KernelInvocation) -> Result<KernelExecution, RuntimeError> {
        let module = pvlc_wgsl::module(invocation.kernel_id()).ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                format!("no WGSL module exists for {}", invocation.kernel_id()),
            )
        })?;
        self.run_internal(
            invocation,
            invocation.kernel_id().as_str(),
            module.source,
            module.spec.entry_point,
            Some(invocation.kernel_id()),
        )
    }

    pub fn run_decoder_cached_gqa(
        &self,
        invocation: &DecoderCachedGqaInvocation<'_>,
    ) -> Result<DecoderCachedGqaExecution, RuntimeError> {
        self.run_decoder_cached_gqa_with_shader_overrides(invocation, &BTreeMap::new())
    }

    pub fn begin_decoder_kv_session<'runtime>(
        &'runtime self,
        descriptor: &DecoderKvSessionDescriptor<'_>,
    ) -> Result<NativeDecoderKvSession<'runtime>, RuntimeError> {
        decoder_kv_session::begin(self, descriptor, &BTreeMap::new())
    }

    #[doc(hidden)]
    pub fn begin_decoder_kv_session_with_shader_overrides<'runtime>(
        &'runtime self,
        descriptor: &DecoderKvSessionDescriptor<'_>,
        shader_overrides: &BTreeMap<KernelId, String>,
    ) -> Result<NativeDecoderKvSession<'runtime>, RuntimeError> {
        decoder_kv_session::begin(self, descriptor, shader_overrides)
    }

    #[doc(hidden)]
    pub fn run_decoder_cached_gqa_with_shader_overrides(
        &self,
        invocation: &DecoderCachedGqaInvocation<'_>,
        shader_overrides: &BTreeMap<KernelId, String>,
    ) -> Result<DecoderCachedGqaExecution, RuntimeError> {
        let plan = invocation.plan().map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidInvocation, None, error.to_string())
        })?;
        let sources = self.validated_decoder_cached_gqa_sources(shader_overrides)?;
        self.validate_decoder_cached_gqa_capabilities(&plan)?;

        let _execution = self
            .execution_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut scopes = WgpuScopeDriver::new(self.device.clone(), self.observer.clone());
        drive_error_scopes(&mut scopes, || {
            self.execute_decoder_cached_gqa_once(invocation, plan, &sources, shader_overrides)
        })
        .map_err(|error| error.with_context("decoder cached GQA"))
    }

    fn validated_decoder_cached_gqa_sources(
        &self,
        shader_overrides: &BTreeMap<KernelId, String>,
    ) -> Result<BTreeMap<KernelId, String>, RuntimeError> {
        const KERNELS: [KernelId; 2] = [KernelId::DecoderKvAppendF32, KernelId::DecoderGqaF32];
        self.validated_decoder_kernel_sources(
            "decoder cached GQA",
            "decoder cached-GQA",
            &KERNELS,
            shader_overrides,
        )
    }

    /// M7o2 amendment: the persistent decoder KV session executes the split-K
    /// GQA pair, so its accepted override set is the append kernel plus the
    /// split partial and split merge kernels; the serial `decoder_gqa_f32`
    /// kernel is no longer part of the session authority.
    fn validated_decoder_kv_session_sources(
        &self,
        shader_overrides: &BTreeMap<KernelId, String>,
    ) -> Result<BTreeMap<KernelId, String>, RuntimeError> {
        const KERNELS: [KernelId; 3] = [
            KernelId::DecoderKvAppendF32,
            KernelId::DecoderGqaSplitPartialF32,
            KernelId::DecoderGqaSplitMergeF32,
        ];
        self.validated_decoder_kernel_sources(
            "decoder KV session",
            "decoder KV session",
            &KERNELS,
            shader_overrides,
        )
    }

    fn validated_decoder_kernel_sources(
        &self,
        usage_label: &str,
        shader_label: &str,
        kernels: &[KernelId],
        shader_overrides: &BTreeMap<KernelId, String>,
    ) -> Result<BTreeMap<KernelId, String>, RuntimeError> {
        if let Some(kernel) = shader_overrides
            .keys()
            .find(|kernel| !kernels.contains(kernel))
        {
            return Err(RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                format!("{kernel} is not used by {usage_label}"),
            ));
        }

        let mut sources = BTreeMap::new();
        for kernel in kernels {
            let module = pvlc_wgsl::module(*kernel)
                .expect("every decoder cached-GQA kernel has a fixed WGSL module");
            let source = shader_overrides
                .get(kernel)
                .map_or(module.source, String::as_str);
            pvlc_wgsl::validate_source_contract(&module.spec, source).map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::Validation, None, error.to_string())
                    .with_context(format!("{shader_label} shader {kernel}"))
            })?;
            sources.insert(*kernel, source.to_owned());
        }
        Ok(sources)
    }

    fn validate_decoder_cached_gqa_capabilities(
        &self,
        plan: &DecoderCachedGqaPlan,
    ) -> Result<(), RuntimeError> {
        if self.capabilities.max_storage_buffers_per_shader_stage < 4 {
            return Err(RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                "decoder cached GQA requires four storage buffers per shader stage",
            ));
        }
        let limits = ComputeDispatchLimits {
            max_workgroup_size: [
                self.capabilities.max_compute_workgroup_size_x,
                self.capabilities.max_compute_workgroup_size_y,
                self.capabilities.max_compute_workgroup_size_z,
            ],
            max_invocations_per_workgroup: self.capabilities.max_compute_invocations_per_workgroup,
            max_workgroups_per_dimension: self.capabilities.max_compute_workgroups_per_dimension,
        };
        for dispatch in [plan.append.invocation, plan.attention.invocation] {
            limits.validate(&dispatch).map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::Validation, None, error.to_string())
            })?;
        }
        let key_value_row_bytes = u64::try_from(plan.key_value_width)
            .ok()
            .and_then(|elements| elements.checked_mul(4))
            .ok_or_else(|| {
                RuntimeError::new(
                    RuntimeErrorCode::InvalidInvocation,
                    None,
                    "decoder key/value row byte size overflowed",
                )
            })?;
        for (label, bytes) in [
            ("decoder query", plan.attention_bytes),
            ("decoder appended key/value row", key_value_row_bytes),
            ("decoder compact key/value cache", plan.cache_bytes),
            ("decoder attention output", plan.attention_bytes),
        ] {
            self.validate_storage_buffer_bytes(label, bytes)?;
        }
        let readback_bytes = plan
            .attention_bytes
            .checked_add(
                plan.cache_bytes
                    .checked_mul(2)
                    .ok_or_else(|| RuntimeError::operation("decoder readback size overflowed"))?,
            )
            .ok_or_else(|| RuntimeError::operation("decoder readback size overflowed"))?;
        if readback_bytes > self.capabilities.max_buffer_size {
            return Err(RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                format!(
                    "decoder cached-GQA readback requires {readback_bytes} bytes but max_buffer_size is {}",
                    self.capabilities.max_buffer_size
                ),
            ));
        }
        Ok(())
    }

    pub fn run_projector(
        &self,
        invocation: &ProjectorInvocation<'_>,
        readback: ProjectorReadback,
    ) -> Result<ProjectorExecution, RuntimeError> {
        self.run_projector_with_shader_overrides(invocation, readback, &BTreeMap::new())
    }

    #[doc(hidden)]
    pub fn run_projector_with_shader_overrides(
        &self,
        invocation: &ProjectorInvocation<'_>,
        readback: ProjectorReadback,
        shader_overrides: &BTreeMap<KernelId, String>,
    ) -> Result<ProjectorExecution, RuntimeError> {
        let plan = invocation.plan().map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidInvocation, None, error.to_string())
        })?;
        let sources = self.validated_projector_sources(shader_overrides)?;
        self.validate_projector_capabilities(invocation, &plan, readback)?;

        let _execution = self
            .execution_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = self.counters();
        let mut scopes = WgpuScopeDriver::new(self.device.clone(), self.observer.clone());
        drive_error_scopes(&mut scopes, || {
            self.execute_projector_once(
                invocation,
                plan,
                readback,
                &sources,
                shader_overrides,
                before,
            )
        })
        .map_err(|error| error.with_context("resident projector"))
    }

    pub fn run_vision_encoder_layer_identity_rope(
        &self,
        invocation: &VisionEncoderLayerInvocation<'_>,
        readback: VisionLayerReadback,
    ) -> Result<VisionLayerExecution, RuntimeError> {
        self.run_vision_encoder_layer_identity_rope_with_shader_overrides(
            invocation,
            readback,
            &BTreeMap::new(),
        )
    }

    #[doc(hidden)]
    pub fn run_vision_encoder_layer_identity_rope_with_shader_overrides(
        &self,
        invocation: &VisionEncoderLayerInvocation<'_>,
        readback: VisionLayerReadback,
        shader_overrides: &BTreeMap<KernelId, String>,
    ) -> Result<VisionLayerExecution, RuntimeError> {
        let plan = invocation.plan().map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidInvocation, None, error.to_string())
        })?;
        let sources = self.validated_vision_layer_sources(shader_overrides)?;
        self.validate_vision_layer_capabilities(invocation, &plan, readback)?;

        let _execution = self
            .execution_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = self.counters();
        let mut scopes = WgpuScopeDriver::new(self.device.clone(), self.observer.clone());
        drive_error_scopes(&mut scopes, || {
            self.execute_vision_layer_once(
                invocation,
                plan,
                readback,
                &sources,
                shader_overrides,
                before,
            )
        })
        .map_err(|error| error.with_context("resident identity-RoPE vision encoder layer"))
    }

    pub fn run_vision_encoder_stack_identity_rope(
        &self,
        invocation: &VisionEncoderStackInvocation<'_>,
        checkpoint_layers: &[usize],
    ) -> Result<VisionStackExecution, RuntimeError> {
        self.run_vision_encoder_stack_identity_rope_with_activation_strategy(
            invocation,
            checkpoint_layers,
            VisionStackActivationStrategy::SeparateBuffers,
        )
    }

    pub fn run_vision_encoder_stack_identity_rope_with_activation_strategy(
        &self,
        invocation: &VisionEncoderStackInvocation<'_>,
        checkpoint_layers: &[usize],
        activation_strategy: VisionStackActivationStrategy,
    ) -> Result<VisionStackExecution, RuntimeError> {
        let plan = invocation.plan(checkpoint_layers).map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidInvocation, None, error.to_string())
        })?;
        let sources = self.validated_vision_stack_sources(activation_strategy)?;
        self.validate_vision_stack_capabilities(invocation, &plan)?;
        let activation = self.prepare_vision_stack_activation(&plan, activation_strategy)?;

        let _execution = self
            .execution_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = self.counters();
        let mut scopes = WgpuScopeDriver::new(self.device.clone(), self.observer.clone());
        drive_error_scopes(&mut scopes, || {
            self.execute_vision_stack_once(invocation, plan, activation, &sources, before)
        })
        .map_err(|error| error.with_context("resident identity-RoPE vision encoder stack"))
    }

    pub fn run_vision_encoder_stack_identity_rope_with_qkv_selection(
        &self,
        invocation: &VisionEncoderStackInvocation<'_>,
        checkpoint_layers: &[usize],
        activation_strategy: VisionStackActivationStrategy,
        selection: &VisionQkvStackSelection,
    ) -> Result<VisionQkvStackExecution, RuntimeError> {
        self.run_vision_encoder_stack_identity_rope_with_qkv_selection_and_shader_overrides(
            invocation,
            checkpoint_layers,
            activation_strategy,
            selection,
            &BTreeMap::new(),
        )
    }

    #[doc(hidden)]
    pub fn run_vision_encoder_stack_identity_rope_with_qkv_selection_and_shader_overrides(
        &self,
        invocation: &VisionEncoderStackInvocation<'_>,
        checkpoint_layers: &[usize],
        activation_strategy: VisionStackActivationStrategy,
        selection: &VisionQkvStackSelection,
        shader_overrides: &BTreeMap<KernelId, String>,
    ) -> Result<VisionQkvStackExecution, RuntimeError> {
        // Stack data and checkpoint validation intentionally precede any
        // selection inspection, shader work, lock acquisition, or GPU effect.
        let plan = invocation.plan(checkpoint_layers).map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidInvocation, None, error.to_string())
        })?;
        self.validate_vision_stack_capabilities(invocation, &plan)?;
        let activation = self.prepare_vision_stack_activation(&plan, activation_strategy)?;
        let prepared = self.prepare_vision_qkv_execution(selection, invocation, &plan)?;
        let sources = self.validated_vision_qkv_stack_sources(
            activation_strategy,
            prepared.qkv_physical_execution.is_some(),
            shader_overrides,
        )?;
        let _execution = self
            .execution_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = self.counters();
        let mut scopes = WgpuScopeDriver::new(self.device.clone(), self.observer.clone());
        drive_error_scopes(&mut scopes, || {
            self.execute_vision_stack_once_optimized(
                invocation,
                plan,
                activation,
                &sources,
                shader_overrides,
                prepared,
                before,
            )
        })
        .map_err(|error| error.with_context("resident verified-Q/K/V vision encoder stack"))
    }

    fn prepare_vision_qkv_execution(
        &self,
        selection: &VisionQkvStackSelection,
        invocation: &VisionEncoderStackInvocation<'_>,
        plan: &VisionEncoderStackPlan,
    ) -> Result<PreparedVisionQkvExecution, RuntimeError> {
        let policy = selection.policy();
        let outcome = selection.outcome();
        let overlay = selection.overlay();
        let consistent = matches!(
            (policy, outcome, overlay),
            (
                VisionQkvExecutionPolicy::Disabled,
                VisionQkvSelectionOutcome::Disabled,
                None,
            ) | (
                VisionQkvExecutionPolicy::Preferred | VisionQkvExecutionPolicy::Required,
                VisionQkvSelectionOutcome::Fused,
                Some(_),
            ) | (
                VisionQkvExecutionPolicy::Preferred,
                VisionQkvSelectionOutcome::FallbackUnsupportedTarget,
                None,
            )
        );
        if !consistent {
            return Err(invalid_optimized_invocation(
                "Q/K/V selection policy, outcome, and overlay are inconsistent",
            ));
        }
        if plan.layer_count != invocation.layer_parameters.len() {
            return Err(invalid_optimized_invocation(
                "vision-stack plan depth differs from the invocation parameters",
            ));
        }

        let target = VisionQkvFusedTargetLimits {
            min_storage_buffer_offset_alignment: self
                .capabilities
                .min_storage_buffer_offset_alignment,
            max_storage_buffers_per_shader_stage: self
                .capabilities
                .max_storage_buffers_per_shader_stage,
            max_storage_buffer_binding_size: self.capabilities.max_storage_buffer_binding_size,
            max_buffer_size: self.capabilities.max_buffer_size,
            max_compute_workgroups_per_dimension: self
                .capabilities
                .max_compute_workgroups_per_dimension,
        };
        let geometry = VisionEncoderLayerPlan {
            rope_specialization: plan.rope_specialization,
            dispatches: plan.layer_dispatches,
            resident_intermediate_bytes: plan.activation_arena_bytes,
        };
        let prepared_execution = if let Some(overlay) = overlay {
            let prepared_execution =
                prepare_vision_qkv_stack_execution(overlay, plan.layer_count, &geometry, target)
                    .map_err(map_vision_qkv_prepared_execution_error)?;
            Some(prepared_execution)
        } else {
            None
        };

        if let Some(prepared_execution) = prepared_execution.as_ref() {
            let executor_invocation = prepared_execution
                .layers()
                .first()
                .ok_or_else(|| {
                    invalid_optimized_invocation(
                        "prepared Q/K/V execution has no layer descriptors",
                    )
                })?
                .invocation();
            let compute_limits = ComputeDispatchLimits {
                max_workgroup_size: [
                    self.capabilities.max_compute_workgroup_size_x,
                    self.capabilities.max_compute_workgroup_size_y,
                    self.capabilities.max_compute_workgroup_size_z,
                ],
                max_invocations_per_workgroup: self
                    .capabilities
                    .max_compute_invocations_per_workgroup,
                max_workgroups_per_dimension: self
                    .capabilities
                    .max_compute_workgroups_per_dimension,
            };
            let validation = compute_limits.validate(&executor_invocation);
            validation.map_err(|error| vision_qkv_preflight_validation(error.to_string()))?;
        }

        let canary_readback_bytes = prepared_execution
            .as_ref()
            .map_or(0, |prepared| prepared.workspace().canary_readback_bytes());
        let workspace_allocation_bytes = prepared_execution
            .as_ref()
            .map_or(0, |prepared| prepared.workspace().allocation_bytes());
        let max_host_elements = u64::try_from(usize::MAX).unwrap_or(u64::MAX);
        let preflight =
            preflight_vision_qkv_execution_allocations(VisionQkvExecutionAllocationPreflight {
                semantic_readback_bytes: plan.readback_bytes,
                canary_readback_bytes,
                workspace_allocation_bytes,
                max_buffer_size: self.capabilities.max_buffer_size,
                max_host_elements,
            })?;

        match prepared_execution {
            Some(prepared_execution) => {
                let physical_execution =
                    bind_vision_qkv_physical_execution(prepared_execution, preflight.layout)
                        .map_err(|error| invalid_optimized_invocation(error.to_string()))?;
                Ok(PreparedVisionQkvExecution {
                    policy,
                    outcome,
                    qkv_physical_execution: Some(physical_execution),
                    total_readback_bytes: preflight.total_readback_bytes,
                    readback_f32_elements: preflight.readback_f32_elements,
                    workspace_u32_words: preflight.workspace_u32_words,
                })
            }
            None => Ok(PreparedVisionQkvExecution {
                policy,
                outcome,
                qkv_physical_execution: None,
                total_readback_bytes: preflight.total_readback_bytes,
                readback_f32_elements: preflight.readback_f32_elements,
                workspace_u32_words: preflight.workspace_u32_words,
            }),
        }
    }
    fn validated_vision_qkv_stack_sources(
        &self,
        activation_strategy: VisionStackActivationStrategy,
        fused: bool,
        shader_overrides: &BTreeMap<KernelId, String>,
    ) -> Result<BTreeMap<KernelId, String>, RuntimeError> {
        if !fused && !shader_overrides.is_empty() {
            return Err(invalid_optimized_invocation(
                "shader overrides require a selected fused Q/K/V overlay",
            ));
        }
        if let Some(kernel) = shader_overrides
            .keys()
            .find(|kernel| **kernel != KernelId::VisionQkvFusedF32)
        {
            return Err(RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                format!("{kernel} is not an overridable optimized-stack kernel"),
            ));
        }
        let mut sources = self.validated_vision_stack_sources(activation_strategy)?;
        if fused {
            let module = pvlc_wgsl::module(KernelId::VisionQkvFusedF32)
                .expect("the fused vision Q/K/V kernel has a fixed WGSL module");
            let source = shader_overrides
                .get(&KernelId::VisionQkvFusedF32)
                .map_or(module.source, String::as_str);
            pvlc_wgsl::validate_source_contract(&module.spec, source).map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::Validation, None, error.to_string())
                    .with_context("optimized stack fused Q/K/V shader")
            })?;
            sources.insert(KernelId::VisionQkvFusedF32, source.to_owned());
        }
        Ok(sources)
    }

    fn validated_vision_layer_sources<'a>(
        &self,
        shader_overrides: &'a BTreeMap<KernelId, String>,
    ) -> Result<BTreeMap<KernelId, &'a str>, RuntimeError> {
        if let Some(kernel) = shader_overrides
            .keys()
            .find(|kernel| !VISION_LAYER_KERNELS.contains(kernel))
        {
            return Err(RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                format!("{kernel} is not used by the resident vision layer"),
            ));
        }

        let mut sources = BTreeMap::new();
        for kernel in VISION_LAYER_KERNELS {
            let module = pvlc_wgsl::module(kernel)
                .expect("every resident vision-layer kernel has a fixed WGSL module");
            let source = shader_overrides
                .get(&kernel)
                .map_or(module.source, String::as_str);
            pvlc_wgsl::validate_source_contract(&module.spec, source).map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::Validation, None, error.to_string())
                    .with_context(format!("resident vision-layer shader {kernel}"))
            })?;
            sources.insert(kernel, source);
        }
        Ok(sources)
    }

    fn validated_vision_stack_sources(
        &self,
        activation_strategy: VisionStackActivationStrategy,
    ) -> Result<BTreeMap<KernelId, String>, RuntimeError> {
        let shader_overrides = BTreeMap::new();
        let sources = self.validated_vision_layer_sources(&shader_overrides)?;
        sources
            .into_iter()
            .map(|(kernel, source)| {
                let source = if activation_strategy
                    == VisionStackActivationStrategy::SeparateBuffers
                {
                    source.to_owned()
                } else {
                    let module = pvlc_wgsl::module(kernel)
                        .expect("every resident vision-layer kernel has a fixed WGSL module");
                    pvlc_wgsl::storage_read_write_variant(&module.spec, source).map_err(
                        |error| {
                            RuntimeError::new(RuntimeErrorCode::Validation, None, error.to_string())
                                .with_context(format!("static-arena vision-layer shader {kernel}"))
                        },
                    )?
                };
                Ok((kernel, source))
            })
            .collect()
    }

    fn validated_projector_sources<'a>(
        &self,
        shader_overrides: &'a BTreeMap<KernelId, String>,
    ) -> Result<BTreeMap<KernelId, &'a str>, RuntimeError> {
        if let Some(kernel) = shader_overrides
            .keys()
            .find(|kernel| !PROJECTOR_KERNELS.contains(kernel))
        {
            return Err(RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                format!("{kernel} is not used by the resident projector"),
            ));
        }

        let mut sources = BTreeMap::new();
        for kernel in PROJECTOR_KERNELS {
            let module = pvlc_wgsl::module(kernel)
                .expect("every resident projector kernel has a fixed WGSL module");
            let source = shader_overrides
                .get(&kernel)
                .map_or(module.source, String::as_str);
            pvlc_wgsl::validate_source_contract(&module.spec, source).map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::Validation, None, error.to_string())
                    .with_context(format!("resident projector shader {kernel}"))
            })?;
            sources.insert(kernel, source);
        }
        Ok(sources)
    }

    pub fn run_vision_qkv_fused(
        &self,
        invocation: &VisionQkvFusedInvocation<'_>,
    ) -> Result<KernelExecution, RuntimeError> {
        let module = pvlc_wgsl::module(KernelId::VisionQkvFusedF32)
            .expect("the fused vision Q/K/V kernel has a fixed WGSL module");
        self.run_vision_qkv_fused_internal(
            invocation,
            KernelId::VisionQkvFusedF32.as_str(),
            module.source,
            module.spec.entry_point,
            0.0_f32.to_bits(),
            Some(KernelId::VisionQkvFusedF32),
        )
    }

    #[doc(hidden)]
    pub fn run_vision_qkv_fused_with_shader(
        &self,
        invocation: &VisionQkvFusedInvocation<'_>,
        label: &str,
        source: &str,
        entry_point: &str,
        output_fill_bits: u32,
    ) -> Result<KernelExecution, RuntimeError> {
        self.run_vision_qkv_fused_internal(
            invocation,
            label,
            source,
            entry_point,
            output_fill_bits,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_vision_qkv_fused_internal(
        &self,
        invocation: &VisionQkvFusedInvocation<'_>,
        label: &str,
        source: &str,
        entry_point: &str,
        output_fill_bits: u32,
        cached_kernel: Option<KernelId>,
    ) -> Result<KernelExecution, RuntimeError> {
        let target = VisionQkvFusedTargetLimits {
            min_storage_buffer_offset_alignment: self
                .capabilities
                .min_storage_buffer_offset_alignment,
            max_storage_buffers_per_shader_stage: self
                .capabilities
                .max_storage_buffers_per_shader_stage,
            max_storage_buffer_binding_size: self.capabilities.max_storage_buffer_binding_size,
            max_buffer_size: self.capabilities.max_buffer_size,
            max_compute_workgroups_per_dimension: self
                .capabilities
                .max_compute_workgroups_per_dimension,
        };
        let plan = invocation.plan(target).map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidInvocation, None, error.to_string())
        })?;
        let spec = pvlc_wgsl::module(KernelId::VisionQkvFusedF32)
            .expect("the fused vision Q/K/V kernel has a fixed WGSL specification")
            .spec;
        pvlc_wgsl::validate_source_contract(&spec, source).map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::Validation, None, error.to_string())
                .with_context(format!("shader {label}"))
        })?;
        if entry_point != spec.entry_point {
            return Err(RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                format!(
                    "shader {label} selects entry point {entry_point:?}, but the ABI requires {:?}",
                    spec.entry_point
                ),
            ));
        }

        let input_data = invocation.inputs().map(InvocationInput::F32);
        let uniform_bytes = plan
            .uniform_words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        self.run_single_kernel_validated(
            KernelId::VisionQkvFusedF32,
            &input_data,
            plan.invocation,
            &uniform_bytes,
            SingleKernelOutputInitializer::FillBits(output_fill_bits),
            SingleKernelTimestampPolicy::SingleSubmission,
            label,
            source,
            entry_point,
            cached_kernel,
        )
    }

    pub fn run_with_shader(
        &self,
        invocation: &KernelInvocation,
        label: &str,
        source: &str,
        entry_point: &str,
    ) -> Result<KernelExecution, RuntimeError> {
        self.run_internal(invocation, label, source, entry_point, None)
    }

    fn run_internal(
        &self,
        invocation: &KernelInvocation,
        label: &str,
        source: &str,
        entry_point: &str,
        cached_kernel: Option<KernelId>,
    ) -> Result<KernelExecution, RuntimeError> {
        let plan = invocation.plan().map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidInvocation, None, error.to_string())
        })?;
        let uniform_bytes = invocation.uniform_bytes().map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidInvocation, None, error.to_string())
        })?;
        let spec = pvlc_wgsl::module(invocation.kernel_id())
            .expect("every invocation kernel has a fixed WGSL specification")
            .spec;
        pvlc_wgsl::validate_source_contract(&spec, source).map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::Validation, None, error.to_string())
                .with_context(format!("shader {label}"))
        })?;
        if entry_point != spec.entry_point {
            return Err(RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                format!(
                    "shader {label} selects entry point {entry_point:?}, but the ABI requires {:?}",
                    spec.entry_point
                ),
            ));
        }

        let input_data = invocation.inputs();
        let output_initializer = invocation.output_initializer().map_or(
            SingleKernelOutputInitializer::Zero,
            SingleKernelOutputInitializer::F32,
        );
        self.run_single_kernel_validated(
            invocation.kernel_id(),
            &input_data,
            plan,
            &uniform_bytes,
            output_initializer,
            SingleKernelTimestampPolicy::RequireFresh,
            label,
            source,
            entry_point,
            cached_kernel,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_single_kernel_validated(
        &self,
        kernel: KernelId,
        input_data: &[InvocationInput<'_>],
        plan: InvocationPlan,
        uniform_bytes: &[u8],
        output_initializer: SingleKernelOutputInitializer<'_>,
        timestamp_policy: SingleKernelTimestampPolicy,
        label: &str,
        source: &str,
        entry_point: &str,
        cached_kernel: Option<KernelId>,
    ) -> Result<KernelExecution, RuntimeError> {
        let _execution = self
            .execution_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut scopes = WgpuScopeDriver::new(self.device.clone(), self.observer.clone());
        drive_error_scopes(&mut scopes, || {
            self.execute_single_kernel_scoped(
                kernel,
                input_data,
                plan,
                uniform_bytes,
                output_initializer,
                timestamp_policy,
                label,
                source,
                entry_point,
                cached_kernel,
            )
        })
        .map_err(|error| error.with_context(format!("kernel {label}")))
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_single_kernel_scoped(
        &self,
        kernel: KernelId,
        input_data: &[InvocationInput<'_>],
        plan: InvocationPlan,
        uniform_bytes: &[u8],
        output_initializer: SingleKernelOutputInitializer<'_>,
        timestamp_policy: SingleKernelTimestampPolicy,
        label: &str,
        source: &str,
        entry_point: &str,
        cached_kernel: Option<KernelId>,
    ) -> Result<KernelExecution, RuntimeError> {
        for attempt in 0..TIMESTAMP_ATTEMPTS {
            let execution = self.execute_single_kernel_once(
                kernel,
                input_data,
                plan,
                uniform_bytes,
                output_initializer,
                timestamp_policy,
                label,
                source,
                entry_point,
                cached_kernel,
            )?;
            if timestamp_policy == SingleKernelTimestampPolicy::SingleSubmission {
                return Ok(execution);
            }
            let Some(timestamp) = execution.diagnostics.timestamp else {
                return Ok(execution);
            };
            if self.accept_fresh_timestamp(timestamp) {
                return Ok(execution);
            }
            if attempt + 1 == TIMESTAMP_ATTEMPTS {
                return Err(RuntimeError::operation(format!(
                    "timestamp query for {label} remained zero or stale after {TIMESTAMP_ATTEMPTS} GPU submissions"
                )));
            }
        }
        unreachable!("the timestamp retry loop always returns")
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_single_kernel_once(
        &self,
        kernel: KernelId,
        input_data: &[InvocationInput<'_>],
        plan: InvocationPlan,
        uniform_bytes: &[u8],
        output_initializer: SingleKernelOutputInitializer<'_>,
        timestamp_policy: SingleKernelTimestampPolicy,
        label: &str,
        source: &str,
        entry_point: &str,
        cached_kernel: Option<KernelId>,
    ) -> Result<KernelExecution, RuntimeError> {
        let timestamp_resources = match timestamp_policy {
            SingleKernelTimestampPolicy::RequireFresh => self.timestamp_resources.as_ref(),
            SingleKernelTimestampPolicy::SingleSubmission => None,
        };
        let pipeline = self.pipeline(label, source, entry_point, cached_kernel);
        let mut input_buffers = Vec::with_capacity(input_data.len());
        for (index, data) in input_data.iter().enumerate() {
            input_buffers.push(self.create_initialized_buffer(
                &format!("{label}-input-{index}"),
                invocation_input_bytes(data),
                wgpu::BufferUsages::STORAGE,
            ));
        }

        let output_contents = single_kernel_output_contents(output_initializer, plan)?;
        let output_buffer = self.create_initialized_buffer(
            &format!("{label}-output"),
            &output_contents,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let uniform_buffer = self.create_initialized_buffer(
            &format!("{label}-uniform"),
            uniform_bytes,
            wgpu::BufferUsages::UNIFORM,
        );
        let readback_buffer = self.create_buffer(
            &format!("{label}-readback"),
            plan.output_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

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
            let timestamp_writes =
                timestamp_resources.map(|resources| wgpu::ComputePassTimestampWrites {
                    query_set: &resources.query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{label}-pass")),
                timestamp_writes,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(plan.dispatch[0], plan.dispatch[1], plan.dispatch[2]);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, plan.output_bytes);
        if let Some(resources) = timestamp_resources {
            encoder.resolve_query_set(&resources.query_set, 0..2, &resources.resolve, 0);
            encoder.copy_buffer_to_buffer(&resources.resolve, 0, &resources.readback, 0, 16);
        }

        let started = Instant::now();
        let submission_index = self.submit_command_buffers([encoder.finish()]);

        let output_receiver = map_read(&readback_buffer);
        let timestamp_receiver = timestamp_resources.map(|resources| map_read(&resources.readback));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: Some(GPU_WAIT_TIMEOUT),
            })
            .map_err(|error| RuntimeError::operation(format!("GPU completion failed: {error}")))?;
        await_mapping(output_receiver, "kernel output")?;
        if let Some(receiver) = timestamp_receiver {
            await_mapping(receiver, "timestamp query")?;
        }
        let queue_wall_time_ns = elapsed_ns(started.elapsed());

        let values = read_f32_buffer(&readback_buffer, plan.output_elements)?;
        let timestamp = timestamp_resources
            .map(|resources| {
                let ticks = read_u64_buffer(&resources.readback)?;
                let period_ns = f64::from(self.queue.get_timestamp_period());
                Ok(GpuTimestamp {
                    begin_ticks: ticks[0],
                    end_ticks: ticks[1],
                    period_ns,
                    duration_ns: ticks[1].saturating_sub(ticks[0]) as f64 * period_ns,
                })
            })
            .transpose()?;

        Ok(KernelExecution {
            values,
            diagnostics: KernelDiagnostics {
                kernel,
                checked_error_scopes: CHECKED_SCOPE_ORDER,
                captured_errors: Vec::new(),
                queue_wall_time_ns,
                timestamp,
                shader_blake3: *blake3::hash(source.as_bytes()).as_bytes(),
            },
        })
    }

    fn validate_projector_capabilities(
        &self,
        invocation: &ProjectorInvocation<'_>,
        plan: &ProjectorPlan,
        readback: ProjectorReadback,
    ) -> Result<(), RuntimeError> {
        let maximum_dispatch = self.capabilities.max_compute_workgroups_per_dimension;
        for dispatch in plan.dispatches {
            if dispatch
                .invocation
                .dispatch
                .iter()
                .any(|dimension| *dimension > maximum_dispatch)
            {
                return Err(RuntimeError::new(
                    RuntimeErrorCode::Validation,
                    None,
                    format!(
                        "{} dispatch {:?} exceeds adapter limit {maximum_dispatch}",
                        dispatch.stage.as_str(),
                        dispatch.invocation.dispatch
                    ),
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
                .ok_or_else(|| {
                    RuntimeError::new(
                        RuntimeErrorCode::InvalidInvocation,
                        None,
                        format!("{label} byte size overflowed"),
                    )
                })?;
            self.validate_storage_buffer_bytes(label, bytes)?;
        }

        let readback_bytes = plan.readback_bytes(readback);
        if readback_bytes > self.capabilities.max_buffer_size {
            return Err(RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                format!(
                    "projector readback requires {readback_bytes} bytes but the adapter limit is {}",
                    self.capabilities.max_buffer_size
                ),
            ));
        }
        Ok(())
    }

    fn validate_vision_layer_capabilities(
        &self,
        invocation: &VisionEncoderLayerInvocation<'_>,
        plan: &VisionEncoderLayerPlan,
        readback: VisionLayerReadback,
    ) -> Result<(), RuntimeError> {
        let maximum_dispatch = self.capabilities.max_compute_workgroups_per_dimension;
        for dispatch in plan.dispatches {
            if dispatch
                .invocation
                .dispatch
                .iter()
                .any(|dimension| *dimension > maximum_dispatch)
            {
                return Err(RuntimeError::new(
                    RuntimeErrorCode::Validation,
                    None,
                    format!(
                        "{} dispatch {:?} exceeds adapter limit {maximum_dispatch}",
                        dispatch.stage.as_str(),
                        dispatch.invocation.dispatch
                    ),
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
                .ok_or_else(|| {
                    RuntimeError::new(
                        RuntimeErrorCode::InvalidInvocation,
                        None,
                        format!("{label} byte size overflowed"),
                    )
                })?;
            self.validate_storage_buffer_bytes(label, bytes)?;
        }
        self.validate_storage_buffer_bytes(
            "cu-seqlens",
            u64::try_from(invocation.cu_seqlens.len())
                .ok()
                .and_then(|elements| elements.checked_mul(4))
                .ok_or_else(|| {
                    RuntimeError::new(
                        RuntimeErrorCode::InvalidInvocation,
                        None,
                        "cu-seqlens byte size overflowed",
                    )
                })?,
        )?;

        let readback_bytes = vision_layer_readback_indices(readback)
            .iter()
            .try_fold(0_u64, |bytes, index| {
                bytes.checked_add(plan.dispatches[*index].invocation.output_bytes)
            })
            .ok_or_else(|| {
                RuntimeError::new(
                    RuntimeErrorCode::InvalidInvocation,
                    None,
                    "vision-layer readback byte size overflowed",
                )
            })?;
        if readback_bytes > self.capabilities.max_buffer_size {
            return Err(RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                format!(
                    "vision-layer readback requires {readback_bytes} bytes but the adapter limit is {}",
                    self.capabilities.max_buffer_size
                ),
            ));
        }
        Ok(())
    }

    fn validate_vision_stack_capabilities(
        &self,
        invocation: &VisionEncoderStackInvocation<'_>,
        plan: &VisionEncoderStackPlan,
    ) -> Result<(), RuntimeError> {
        let first_layer = VisionEncoderLayerInvocation {
            tokens: invocation.tokens,
            hidden_size: invocation.hidden_size,
            attention_heads: invocation.attention_heads,
            head_dim: invocation.head_dim,
            intermediate_size: invocation.intermediate_size,
            layer_norm_epsilon: invocation.layer_norm_epsilon,
            input: invocation.input,
            cu_seqlens: invocation.cu_seqlens,
            parameters: invocation.layer_parameters[0],
        };
        let first_layer_plan = first_layer.plan().map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidInvocation, None, error.to_string())
        })?;
        self.validate_vision_layer_capabilities(
            &first_layer,
            &first_layer_plan,
            VisionLayerReadback::OutputOnly,
        )?;
        for (label, values) in [
            ("post-norm-weight", invocation.post_norm.weight),
            ("post-norm-bias", invocation.post_norm.bias),
        ] {
            let bytes = u64::try_from(values.len())
                .ok()
                .and_then(|elements| elements.checked_mul(4))
                .ok_or_else(|| {
                    RuntimeError::new(
                        RuntimeErrorCode::InvalidInvocation,
                        None,
                        format!("{label} byte size overflowed"),
                    )
                })?;
            self.validate_storage_buffer_bytes(label, bytes)?;
        }
        if plan.readback_bytes > self.capabilities.max_buffer_size {
            return Err(RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                format!(
                    "vision-stack readback requires {} bytes but the adapter limit is {}",
                    plan.readback_bytes, self.capabilities.max_buffer_size
                ),
            ));
        }
        Ok(())
    }

    fn prepare_vision_stack_activation(
        &self,
        plan: &VisionEncoderStackPlan,
        strategy: VisionStackActivationStrategy,
    ) -> Result<PreparedVisionStackActivation, RuntimeError> {
        let hidden_output_bytes = plan.layer_dispatches[11].invocation.output_bytes;
        let main_buffers_bytes = hidden_output_bytes.checked_mul(2).ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                "vision-stack main activation buffer bytes overflowed",
            )
        })?;

        if strategy == VisionStackActivationStrategy::SeparateBuffers {
            let scratch_arena_bytes =
                plan.layer_dispatches[..11]
                    .iter()
                    .try_fold(0_u64, |bytes, dispatch| {
                        bytes
                            .checked_add(dispatch.invocation.output_bytes)
                            .ok_or_else(|| {
                                RuntimeError::new(
                                    RuntimeErrorCode::Validation,
                                    None,
                                    "vision-stack separate scratch bytes overflowed",
                                )
                            })
                    })?;
            let total_activation_bytes = scratch_arena_bytes
                .checked_add(main_buffers_bytes)
                .ok_or_else(|| {
                    RuntimeError::new(
                        RuntimeErrorCode::Validation,
                        None,
                        "vision-stack separate activation bytes overflowed",
                    )
                })?;
            if total_activation_bytes != plan.activation_arena_bytes {
                return Err(RuntimeError::new(
                    RuntimeErrorCode::Validation,
                    None,
                    format!(
                        "vision-stack separate activation total {total_activation_bytes} differs from plan diagnostic {}",
                        plan.activation_arena_bytes
                    ),
                ));
            }
            return Ok(PreparedVisionStackActivation {
                strategy,
                activation_buffer_count: plan.activation_buffer_count,
                scratch_arena_bytes,
                main_buffers_bytes,
                total_activation_bytes,
                scratch_allocations: Vec::new(),
            });
        }

        let alignment = u64::from(self.capabilities.min_storage_buffer_offset_alignment).max(1);
        let layout = plan
            .activation_layout(VisionStackActivationLayoutConfig {
                allow_aliasing: strategy == VisionStackActivationStrategy::StaticArenaAlias,
                storage_buffer_offset_alignment: alignment,
                arena_alignment: alignment,
            })
            .map_err(|error| {
                RuntimeError::new(
                    RuntimeErrorCode::Validation,
                    None,
                    format!("vision-stack static activation layout failed: {error}"),
                )
            })?;
        if layout.scratch_arena_bytes > self.capabilities.max_buffer_size {
            return Err(RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                format!(
                    "vision-stack scratch arena requires {} bytes but max_buffer_size is {}",
                    layout.scratch_arena_bytes, self.capabilities.max_buffer_size
                ),
            ));
        }
        if layout.main_buffers_bytes != main_buffers_bytes || layout.physical_buffer_count != 3 {
            return Err(RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                "vision-stack static activation layout has inconsistent main-buffer metadata",
            ));
        }
        for allocation in &layout.scratch_allocations {
            if allocation.size > self.capabilities.max_storage_buffer_binding_size {
                return Err(RuntimeError::new(
                    RuntimeErrorCode::Validation,
                    None,
                    format!(
                        "vision-stack {} binding requires {} bytes but max_storage_buffer_binding_size is {}",
                        allocation.stage.as_str(),
                        allocation.size,
                        self.capabilities.max_storage_buffer_binding_size
                    ),
                ));
            }
            let end = allocation
                .offset
                .checked_add(allocation.size)
                .ok_or_else(|| {
                    RuntimeError::new(
                        RuntimeErrorCode::Validation,
                        None,
                        format!(
                            "vision-stack {} scratch binding end overflowed",
                            allocation.stage.as_str()
                        ),
                    )
                })?;
            if allocation.offset % alignment != 0 || end > layout.scratch_arena_bytes {
                return Err(RuntimeError::new(
                    RuntimeErrorCode::Validation,
                    None,
                    format!(
                        "vision-stack {} scratch binding is misaligned or outside the arena",
                        allocation.stage.as_str()
                    ),
                ));
            }
            if wgpu::BufferSize::new(allocation.size).is_none() {
                return Err(RuntimeError::new(
                    RuntimeErrorCode::Validation,
                    None,
                    format!(
                        "vision-stack {} scratch binding size must be nonzero",
                        allocation.stage.as_str()
                    ),
                ));
            }
        }

        Ok(PreparedVisionStackActivation {
            strategy,
            activation_buffer_count: layout.physical_buffer_count,
            scratch_arena_bytes: layout.scratch_arena_bytes,
            main_buffers_bytes: layout.main_buffers_bytes,
            total_activation_bytes: layout.total_activation_bytes,
            scratch_allocations: layout.scratch_allocations,
        })
    }

    fn validate_storage_buffer_bytes(&self, label: &str, bytes: u64) -> Result<(), RuntimeError> {
        let limit = self
            .capabilities
            .max_storage_buffer_binding_size
            .min(self.capabilities.max_buffer_size);
        if bytes > limit {
            return Err(RuntimeError::new(
                RuntimeErrorCode::Validation,
                None,
                format!("{label} requires {bytes} bytes but the storage-buffer limit is {limit}"),
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn execute_decoder_cached_gqa_once(
        &self,
        invocation: &DecoderCachedGqaInvocation<'_>,
        plan: DecoderCachedGqaPlan,
        sources: &BTreeMap<KernelId, String>,
        shader_overrides: &BTreeMap<KernelId, String>,
    ) -> Result<DecoderCachedGqaExecution, RuntimeError> {
        const KERNELS: [KernelId; 2] = [KernelId::DecoderKvAppendF32, KernelId::DecoderGqaF32];
        let mut pipelines = BTreeMap::new();
        let mut shader_blake3 = BTreeMap::new();
        for kernel in KERNELS {
            let source = &sources[&kernel];
            let cached_kernel = (!shader_overrides.contains_key(&kernel)).then_some(kernel);
            let (pipeline, created) =
                self.pipeline_with_creation_status(kernel.as_str(), source, "main", cached_kernel);
            if created {
                self.pipeline_creations.fetch_add(1, Ordering::Relaxed);
            }
            pipelines.insert(kernel, pipeline);
            shader_blake3.insert(kernel, *blake3::hash(source.as_bytes()).as_bytes());
        }

        let query_buffer = self.create_initialized_buffer(
            "decoder-cached-gqa-query",
            bytemuck::cast_slice(invocation.query),
            wgpu::BufferUsages::STORAGE,
        );
        let appended_key_buffer = self.create_initialized_buffer(
            "decoder-cached-gqa-appended-key",
            bytemuck::cast_slice(invocation.appended_key),
            wgpu::BufferUsages::STORAGE,
        );
        let appended_value_buffer = self.create_initialized_buffer(
            "decoder-cached-gqa-appended-value",
            bytemuck::cast_slice(invocation.appended_value),
            wgpu::BufferUsages::STORAGE,
        );
        let key_cache_buffer = self.create_initialized_buffer(
            "decoder-cached-gqa-key-cache",
            bytemuck::cast_slice(invocation.key_cache),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let value_cache_buffer = self.create_initialized_buffer(
            "decoder-cached-gqa-value-cache",
            bytemuck::cast_slice(invocation.value_cache),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let attention_initializer = vec![0.0_f32; plan.query_elements];
        let attention_buffer = self.create_initialized_buffer(
            "decoder-cached-gqa-attention",
            bytemuck::cast_slice(&attention_initializer),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let append_uniform_bytes = plan
            .append
            .uniform_words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let append_uniform_buffer = self.create_initialized_buffer(
            "decoder-cached-gqa-append-uniform",
            &append_uniform_bytes,
            wgpu::BufferUsages::UNIFORM,
        );
        let attention_uniform_bytes = plan
            .attention
            .uniform_words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let attention_uniform_buffer = self.create_initialized_buffer(
            "decoder-cached-gqa-attention-uniform",
            &attention_uniform_bytes,
            wgpu::BufferUsages::UNIFORM,
        );
        let readback_bytes = plan
            .attention_bytes
            .checked_add(
                plan.cache_bytes
                    .checked_mul(2)
                    .ok_or_else(|| RuntimeError::operation("decoder readback size overflowed"))?,
            )
            .ok_or_else(|| RuntimeError::operation("decoder readback size overflowed"))?;
        let readback_elements = plan
            .query_elements
            .checked_add(
                plan.cache_elements
                    .checked_mul(2)
                    .ok_or_else(|| RuntimeError::operation("decoder readback size overflowed"))?,
            )
            .ok_or_else(|| RuntimeError::operation("decoder readback size overflowed"))?;
        let readback_buffer = self.create_buffer(
            "decoder-cached-gqa-readback",
            readback_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        use DecoderCachedGqaBufferRole::{
            AppendUniform, AppendedKey, AppendedValue, AttentionOutput, AttentionUniform, KeyCache,
            Query, Readback, ValueCache,
        };
        let buffers = vec![
            decoder_buffer_evidence(Query, &query_buffer),
            decoder_buffer_evidence(AppendedKey, &appended_key_buffer),
            decoder_buffer_evidence(AppendedValue, &appended_value_buffer),
            decoder_buffer_evidence(KeyCache, &key_cache_buffer),
            decoder_buffer_evidence(ValueCache, &value_cache_buffer),
            decoder_buffer_evidence(AttentionOutput, &attention_buffer),
            decoder_buffer_evidence(AppendUniform, &append_uniform_buffer),
            decoder_buffer_evidence(AttentionUniform, &attention_uniform_buffer),
            decoder_buffer_evidence(Readback, &readback_buffer),
        ];

        let append_pipeline = &pipelines[&KernelId::DecoderKvAppendF32];
        let append_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decoder-cached-gqa-append-bind-group"),
            layout: &append_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: appended_key_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: appended_value_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: key_cache_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: value_cache_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: append_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let attention_pipeline = &pipelines[&KernelId::DecoderGqaF32];
        let attention_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decoder-cached-gqa-attention-bind-group"),
            layout: &attention_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: key_cache_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: value_cache_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: attention_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: attention_uniform_buffer.as_entire_binding(),
                },
            ],
        });
        self.bind_group_creations.fetch_add(2, Ordering::Relaxed);
        let bind_groups = vec![
            DecoderCachedGqaBindGroupEvidence {
                stage: DecoderCachedGqaStage::AppendKeyValue,
                bindings: vec![
                    decoder_binding_evidence(0, &appended_key_buffer),
                    decoder_binding_evidence(1, &appended_value_buffer),
                    decoder_binding_evidence(2, &key_cache_buffer),
                    decoder_binding_evidence(3, &value_cache_buffer),
                    decoder_binding_evidence(4, &append_uniform_buffer),
                ],
            },
            DecoderCachedGqaBindGroupEvidence {
                stage: DecoderCachedGqaStage::DirectGqa,
                bindings: vec![
                    decoder_binding_evidence(0, &query_buffer),
                    decoder_binding_evidence(1, &key_cache_buffer),
                    decoder_binding_evidence(2, &value_cache_buffer),
                    decoder_binding_evidence(3, &attention_buffer),
                    decoder_binding_evidence(4, &attention_uniform_buffer),
                ],
            },
        ];

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("decoder-cached-gqa-encoder"),
            });
        self.command_encoder_creations
            .fetch_add(1, Ordering::Relaxed);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("decoder-cached-gqa-append-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(append_pipeline);
            pass.set_bind_group(0, &append_bind_group, &[]);
            pass.dispatch_workgroups(
                plan.append.invocation.dispatch[0],
                plan.append.invocation.dispatch[1],
                plan.append.invocation.dispatch[2],
            );
        }
        self.dispatch_encodings.fetch_add(1, Ordering::Relaxed);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("decoder-cached-gqa-attention-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(attention_pipeline);
            pass.set_bind_group(0, &attention_bind_group, &[]);
            pass.dispatch_workgroups(
                plan.attention.invocation.dispatch[0],
                plan.attention.invocation.dispatch[1],
                plan.attention.invocation.dispatch[2],
            );
        }
        self.dispatch_encodings.fetch_add(1, Ordering::Relaxed);
        let dispatches = vec![
            DecoderCachedGqaDispatchEvidence {
                ordinal: 1,
                stage: DecoderCachedGqaStage::AppendKeyValue,
                kernel: KernelId::DecoderKvAppendF32,
                workgroups: plan.append.invocation.dispatch,
            },
            DecoderCachedGqaDispatchEvidence {
                ordinal: 2,
                stage: DecoderCachedGqaStage::DirectGqa,
                kernel: KernelId::DecoderGqaF32,
                workgroups: plan.attention.invocation.dispatch,
            },
        ];

        let attention_offset = 0;
        let key_cache_offset = plan.attention_bytes;
        let value_cache_offset = key_cache_offset
            .checked_add(plan.cache_bytes)
            .ok_or_else(|| RuntimeError::operation("decoder readback offset overflowed"))?;
        encoder.copy_buffer_to_buffer(
            &attention_buffer,
            0,
            &readback_buffer,
            attention_offset,
            plan.attention_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &key_cache_buffer,
            0,
            &readback_buffer,
            key_cache_offset,
            plan.cache_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &value_cache_buffer,
            0,
            &readback_buffer,
            value_cache_offset,
            plan.cache_bytes,
        );
        self.buffer_copy_encodings.fetch_add(3, Ordering::Relaxed);
        let copies = vec![
            decoder_copy_evidence(
                1,
                &attention_buffer,
                &readback_buffer,
                attention_offset,
                plan.attention_bytes,
                DecoderCachedGqaCopyPurpose::Attention,
            ),
            decoder_copy_evidence(
                2,
                &key_cache_buffer,
                &readback_buffer,
                key_cache_offset,
                plan.cache_bytes,
                DecoderCachedGqaCopyPurpose::KeyCache,
            ),
            decoder_copy_evidence(
                3,
                &value_cache_buffer,
                &readback_buffer,
                value_cache_offset,
                plan.cache_bytes,
                DecoderCachedGqaCopyPurpose::ValueCache,
            ),
        ];

        let started = Instant::now();
        let submission_index = self.submit_command_buffers([encoder.finish()]);
        self.map_requests.fetch_add(1, Ordering::Relaxed);
        self.observe(RuntimeEvent::ReadbackMapRequested {
            label: "decoder-cached-gqa-readback".to_owned(),
            bytes: readback_bytes,
        });
        let maps = vec![DecoderCachedGqaMapEvidence {
            buffer_identity: buffer_identity(&readback_buffer),
            byte_offset: 0,
            byte_length: readback_bytes,
            after_copy_ordinal: 3,
        }];
        let receiver = map_read(&readback_buffer);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: Some(GPU_WAIT_TIMEOUT),
            })
            .map_err(|error| RuntimeError::operation(format!("GPU completion failed: {error}")))?;
        await_mapping(receiver, "decoder cached-GQA readback")?;
        let queue_wall_time_ns = elapsed_ns(started.elapsed());
        let values = read_f32_buffer(&readback_buffer, readback_elements)?;
        let attention_end = plan.query_elements;
        let key_cache_end = attention_end
            .checked_add(plan.cache_elements)
            .ok_or_else(|| RuntimeError::operation("decoder readback split overflowed"))?;
        let value_cache_end = key_cache_end
            .checked_add(plan.cache_elements)
            .ok_or_else(|| RuntimeError::operation("decoder readback split overflowed"))?;
        if values.len() != value_cache_end {
            return Err(RuntimeError::mapping(format!(
                "decoder readback returned {} elements but {value_cache_end} were required",
                values.len()
            )));
        }

        Ok(DecoderCachedGqaExecution {
            attention: values[..attention_end].to_vec(),
            key_cache: values[attention_end..key_cache_end].to_vec(),
            value_cache: values[key_cache_end..value_cache_end].to_vec(),
            cache_tokens: plan.cache_tokens_after_append,
            cache_capacity: invocation.cache_capacity,
            diagnostics: DecoderCachedGqaDiagnostics {
                checked_error_scopes: CHECKED_SCOPE_ORDER,
                captured_errors: Vec::new(),
                queue_wall_time_ns,
                shader_blake3,
                dispatch_stages: [
                    DecoderCachedGqaStage::AppendKeyValue,
                    DecoderCachedGqaStage::DirectGqa,
                ],
                dispatch_count: 2,
                compute_pass_count: 2,
                command_buffer_count: 1,
                submission_count: 1,
                readback_buffer_count: 1,
                readback_map_count: 1,
                readback_bytes,
                operation_evidence: DecoderCachedGqaOperationEvidence {
                    buffers,
                    bind_groups,
                    dispatches,
                    copies,
                    maps,
                },
            },
        })
    }

    #[allow(clippy::too_many_lines)]
    fn execute_projector_once(
        &self,
        invocation: &ProjectorInvocation<'_>,
        plan: ProjectorPlan,
        readback: ProjectorReadback,
        sources: &BTreeMap<KernelId, &str>,
        shader_overrides: &BTreeMap<KernelId, String>,
        before: RuntimeCounters,
    ) -> Result<ProjectorExecution, RuntimeError> {
        let mut pipelines = BTreeMap::new();
        let mut shader_blake3 = BTreeMap::new();
        for kernel in PROJECTOR_KERNELS {
            let source = sources[&kernel];
            pipelines.insert(
                kernel,
                self.pipeline(
                    kernel.as_str(),
                    source,
                    "main",
                    (!shader_overrides.contains_key(&kernel)).then_some(kernel),
                ),
            );
            shader_blake3.insert(kernel, *blake3::hash(source.as_bytes()).as_bytes());
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
                self.create_buffer(
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
            .ok_or_else(|| RuntimeError::operation("projector uniform arena overflowed"))?;
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
        let readback_buffer = self.create_buffer(
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
            let timestamp_writes = self.timestamp_resources.as_ref().map(|resources| {
                wgpu::ComputePassTimestampWrites {
                    query_set: &resources.query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("projector-pass"),
                timestamp_writes,
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
        if let Some(resources) = self.timestamp_resources.as_ref() {
            encoder.resolve_query_set(&resources.query_set, 0..2, &resources.resolve, 0);
            encoder.copy_buffer_to_buffer(&resources.resolve, 0, &resources.readback, 0, 16);
        }

        let started = Instant::now();
        let submission_index = self.submit_command_buffers([encoder.finish()]);
        self.observe(RuntimeEvent::ReadbackMapRequested {
            label: "projector-readback".to_owned(),
            bytes: readback_bytes,
        });
        let output_receiver = map_read(&readback_buffer);
        let timestamp_receiver = self
            .timestamp_resources
            .as_ref()
            .map(|resources| map_read(&resources.readback));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: Some(GPU_WAIT_TIMEOUT),
            })
            .map_err(|error| RuntimeError::operation(format!("GPU completion failed: {error}")))?;
        await_mapping(output_receiver, "projector readback")?;
        if let Some(receiver) = timestamp_receiver {
            await_mapping(receiver, "projector timestamp query")?;
        }
        let queue_wall_time_ns = elapsed_ns(started.elapsed());

        let readback_elements = usize::try_from(readback_bytes / 4)
            .map_err(|_| RuntimeError::mapping("projector readback is too large"))?;
        let flat_values = read_f32_buffer(&readback_buffer, readback_elements)?;
        let mut checkpoints = BTreeMap::new();
        let mut element_offset = 0_usize;
        for &index in readback_indices {
            let dispatch = plan.dispatches[index];
            let end = element_offset + dispatch.invocation.output_elements;
            checkpoints.insert(dispatch.stage, flat_values[element_offset..end].to_vec());
            element_offset = end;
        }
        debug_assert_eq!(element_offset, flat_values.len());

        let timestamp = self
            .timestamp_resources
            .as_ref()
            .map(|resources| {
                let ticks = read_u64_buffer(&resources.readback)?;
                let period_ns = f64::from(self.queue.get_timestamp_period());
                Ok(GpuTimestamp {
                    begin_ticks: ticks[0],
                    end_ticks: ticks[1],
                    period_ns,
                    duration_ns: ticks[1].saturating_sub(ticks[0]) as f64 * period_ns,
                })
            })
            .transpose()?;
        if let Some(timestamp) = timestamp
            && (timestamp.begin_ticks == 0 || timestamp.end_ticks <= timestamp.begin_ticks)
        {
            return Err(RuntimeError::operation(format!(
                "resident projector timestamp query was zero or reversed: begin={}, end={}",
                timestamp.begin_ticks, timestamp.end_ticks
            )));
        }
        let timestamp_fresh = timestamp.map(|timestamp| self.accept_fresh_timestamp(timestamp));
        let after = self.counters();

        Ok(ProjectorExecution {
            checkpoints,
            diagnostics: ProjectorDiagnostics {
                checked_error_scopes: CHECKED_SCOPE_ORDER,
                captured_errors: Vec::new(),
                queue_wall_time_ns,
                timestamp,
                timestamp_fresh,
                shader_blake3,
                dispatch_stages: plan.dispatches.map(|dispatch| dispatch.stage),
                submission_count: after.submissions - before.submissions,
                command_buffer_count: 1,
                compute_pass_count: 1,
                dispatch_count: plan.dispatches.len(),
                buffer_allocation_count: after.buffer_allocations - before.buffer_allocations,
                readback_buffer_count: 1,
                readback_map_count: 1,
                readback_bytes,
                resident_intermediate_bytes: plan.resident_intermediate_bytes,
                resident_weight_bytes: plan.resident_weight_bytes,
            },
        })
    }

    #[allow(clippy::too_many_lines)]
    fn execute_vision_layer_once(
        &self,
        invocation: &VisionEncoderLayerInvocation<'_>,
        plan: VisionEncoderLayerPlan,
        readback: VisionLayerReadback,
        sources: &BTreeMap<KernelId, &str>,
        shader_overrides: &BTreeMap<KernelId, String>,
        before: RuntimeCounters,
    ) -> Result<VisionLayerExecution, RuntimeError> {
        let mut pipelines = BTreeMap::new();
        let mut shader_blake3 = BTreeMap::new();
        for kernel in VISION_LAYER_KERNELS {
            let source = sources[&kernel];
            let pipeline = self.pipeline(
                kernel.as_str(),
                source,
                "main",
                (!shader_overrides.contains_key(&kernel)).then_some(kernel),
            );
            pipelines.insert(kernel, pipeline);
            shader_blake3.insert(kernel, *blake3::hash(source.as_bytes()).as_bytes());
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
                self.create_buffer(
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
            .ok_or_else(|| RuntimeError::operation("vision-layer uniform arena overflowed"))?;
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
        let readback_buffer = self.create_buffer(
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
            let timestamp_writes = self.timestamp_resources.as_ref().map(|resources| {
                wgpu::ComputePassTimestampWrites {
                    query_set: &resources.query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vision-layer-pass"),
                timestamp_writes,
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
        if let Some(resources) = self.timestamp_resources.as_ref() {
            encoder.resolve_query_set(&resources.query_set, 0..2, &resources.resolve, 0);
            encoder.copy_buffer_to_buffer(&resources.resolve, 0, &resources.readback, 0, 16);
        }

        let started = Instant::now();
        let submission_index = self.submit_command_buffers([encoder.finish()]);
        let output_receiver = map_read(&readback_buffer);
        let timestamp_receiver = self
            .timestamp_resources
            .as_ref()
            .map(|resources| map_read(&resources.readback));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: Some(GPU_WAIT_TIMEOUT),
            })
            .map_err(|error| RuntimeError::operation(format!("GPU completion failed: {error}")))?;
        await_mapping(output_receiver, "vision-layer readback")?;
        if let Some(receiver) = timestamp_receiver {
            await_mapping(receiver, "vision-layer timestamp query")?;
        }
        let queue_wall_time_ns = elapsed_ns(started.elapsed());

        let readback_elements = usize::try_from(readback_bytes / 4)
            .map_err(|_| RuntimeError::mapping("vision-layer readback is too large"))?;
        let flat_values = read_f32_buffer(&readback_buffer, readback_elements)?;
        let mut checkpoints = BTreeMap::new();
        let mut element_offset = 0_usize;
        for &index in readback_indices {
            let dispatch = plan.dispatches[index];
            let end = element_offset + dispatch.invocation.output_elements;
            checkpoints.insert(dispatch.stage, flat_values[element_offset..end].to_vec());
            element_offset = end;
        }
        debug_assert_eq!(element_offset, flat_values.len());

        let timestamp = self
            .timestamp_resources
            .as_ref()
            .map(|resources| {
                let ticks = read_u64_buffer(&resources.readback)?;
                let period_ns = f64::from(self.queue.get_timestamp_period());
                Ok(GpuTimestamp {
                    begin_ticks: ticks[0],
                    end_ticks: ticks[1],
                    period_ns,
                    duration_ns: ticks[1].saturating_sub(ticks[0]) as f64 * period_ns,
                })
            })
            .transpose()?;
        if let Some(timestamp) = timestamp
            && (timestamp.begin_ticks == 0 || timestamp.end_ticks <= timestamp.begin_ticks)
        {
            return Err(RuntimeError::operation(format!(
                "resident vision-layer timestamp query was zero or reversed: begin={}, end={}",
                timestamp.begin_ticks, timestamp.end_ticks
            )));
        }
        let timestamp_fresh = timestamp.map(|timestamp| self.accept_fresh_timestamp(timestamp));

        let after = self.counters();
        Ok(VisionLayerExecution {
            checkpoints,
            diagnostics: VisionLayerDiagnostics {
                checked_error_scopes: CHECKED_SCOPE_ORDER,
                captured_errors: Vec::new(),
                queue_wall_time_ns,
                timestamp,
                timestamp_fresh,
                shader_blake3,
                dispatch_stages: plan.dispatches.map(|dispatch| dispatch.stage),
                rope_specialization: plan.rope_specialization,
                submission_count: after.submissions - before.submissions,
                command_buffer_count: 1,
                buffer_allocation_count: after.buffer_allocations - before.buffer_allocations,
                readback_buffer_count: 1,
                readback_bytes,
            },
        })
    }

    fn execute_vision_stack_once(
        &self,
        invocation: &VisionEncoderStackInvocation<'_>,
        plan: VisionEncoderStackPlan,
        activation: PreparedVisionStackActivation,
        sources: &BTreeMap<KernelId, String>,
        before: RuntimeCounters,
    ) -> Result<VisionStackExecution, RuntimeError> {
        self.execute_vision_stack_once_common(
            invocation,
            plan,
            activation,
            sources,
            &BTreeMap::new(),
            None,
            before,
        )
        .map(|(execution, evidence)| {
            debug_assert!(evidence.is_none());
            execution
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_vision_stack_once_optimized(
        &self,
        invocation: &VisionEncoderStackInvocation<'_>,
        plan: VisionEncoderStackPlan,
        activation: PreparedVisionStackActivation,
        sources: &BTreeMap<KernelId, String>,
        shader_overrides: &BTreeMap<KernelId, String>,
        prepared: PreparedVisionQkvExecution,
        before: RuntimeCounters,
    ) -> Result<VisionQkvStackExecution, RuntimeError> {
        let (execution, evidence) = self.execute_vision_stack_once_common(
            invocation,
            plan,
            activation,
            sources,
            shader_overrides,
            Some(prepared),
            before,
        )?;
        let evidence = evidence.ok_or_else(|| {
            RuntimeError::operation("optimized stack execution produced no encoding evidence")
        })?;
        Ok(VisionQkvStackExecution {
            checkpoints: execution.checkpoints,
            output: execution.output,
            diagnostics: execution.diagnostics,
            evidence,
        })
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn execute_vision_stack_once_common(
        &self,
        invocation: &VisionEncoderStackInvocation<'_>,
        plan: VisionEncoderStackPlan,
        activation: PreparedVisionStackActivation,
        sources: &BTreeMap<KernelId, String>,
        shader_overrides: &BTreeMap<KernelId, String>,
        optimized: Option<PreparedVisionQkvExecution>,
        before: RuntimeCounters,
    ) -> Result<
        (
            VisionStackExecution,
            Option<VisionQkvStackExecutionEvidence>,
        ),
        RuntimeError,
    > {
        let fused = optimized
            .as_ref()
            .is_some_and(|prepared| prepared.qkv_physical_execution.is_some());
        let qkv_physical_execution = optimized
            .as_ref()
            .and_then(|prepared| prepared.qkv_physical_execution.as_ref());
        let mut trace = optimized
            .as_ref()
            .map(|prepared| VisionQkvStackExecutionEvidence {
                policy: prepared.policy,
                outcome: prepared.outcome,
                canonical_layer_plan_blake3: prepared.qkv_physical_execution.as_ref().map_or_else(
                    Vec::new,
                    |physical| {
                        physical
                            .prepared_execution()
                            .layers()
                            .iter()
                            .map(|layer| layer.canonical_plan_blake3_hex().to_owned())
                            .collect()
                    },
                ),
                pipeline_creations: Vec::new(),
                bind_group_creations: Vec::new(),
                command_encoder_creations: Vec::new(),
                encoded_dispatches: Vec::new(),
                encoded_copies: Vec::new(),
                map_requests: Vec::new(),
                dispatch_count: 0,
                compute_pass_count: plan.compute_pass_count,
                command_buffer_count: 1,
                submission_count: 1,
                map_count: 0,
                workspace: None,
                attention_bindings: Vec::new(),
                canaries: Vec::new(),
            });
        let mut pipelines = BTreeMap::new();
        let mut shader_blake3 = BTreeMap::new();
        let kernels = VISION_LAYER_KERNELS
            .into_iter()
            .chain(fused.then_some(KernelId::VisionQkvFusedF32));
        for kernel in kernels {
            let source = sources[&kernel].as_str();
            let shader_hash = *blake3::hash(source.as_bytes()).as_bytes();
            let (pipeline, created) = self.pipeline_with_creation_status(
                kernel.as_str(),
                source,
                "main",
                ((activation.strategy == VisionStackActivationStrategy::SeparateBuffers
                    || kernel == KernelId::VisionQkvFusedF32)
                    && !shader_overrides.contains_key(&kernel))
                .then_some(kernel),
            );
            pipelines.insert(kernel, pipeline);
            shader_blake3.insert(kernel, shader_hash);
            if created {
                self.record_optimized_pipeline_creation(&mut trace, kernel, shader_hash);
            }
        }

        let layer_weight_buffers = invocation
            .layer_parameters
            .iter()
            .copied()
            .enumerate()
            .map(|(layer, parameters)| {
                self.create_vision_stack_layer_weight_buffers(layer, parameters)
            })
            .collect::<Vec<_>>();
        let post_norm_buffers = [
            self.create_initialized_buffer(
                "vision-stack-post-norm-weight",
                bytemuck::cast_slice(invocation.post_norm.weight),
                wgpu::BufferUsages::STORAGE,
            ),
            self.create_initialized_buffer(
                "vision-stack-post-norm-bias",
                bytemuck::cast_slice(invocation.post_norm.bias),
                wgpu::BufferUsages::STORAGE,
            ),
        ];
        let boundary_buffer = self.create_initialized_buffer(
            "vision-stack-cu-seqlens",
            bytemuck::cast_slice(invocation.cu_seqlens),
            wgpu::BufferUsages::STORAGE,
        );

        let hidden_bytes = plan.layer_dispatches[0].invocation.output_bytes;
        let main_buffers = [
            self.create_initialized_buffer(
                "vision-stack-activation-main-a",
                bytemuck::cast_slice(invocation.input),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            self.create_buffer(
                "vision-stack-activation-main-b",
                hidden_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
        ];
        let scratch_buffers =
            if activation.strategy == VisionStackActivationStrategy::SeparateBuffers {
                VisionStackScratchBuffers::Separate(
                    plan.layer_dispatches[..11]
                        .iter()
                        .map(|dispatch| {
                            self.create_buffer(
                                &format!(
                                    "vision-stack-activation-scratch-{}",
                                    dispatch.stage.as_str()
                                ),
                                dispatch.invocation.output_bytes,
                                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                            )
                        })
                        .collect(),
                )
            } else {
                VisionStackScratchBuffers::Static(self.create_buffer(
                    "vision-stack-activation-scratch-arena",
                    activation.scratch_arena_bytes,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                ))
            };

        let prepared_workspace =
            qkv_physical_execution.map(|physical| physical.prepared_execution().workspace());
        let qkv_workspace = match (qkv_physical_execution, optimized.as_ref()) {
            (Some(physical), Some(prepared)) => {
                Some(self.create_vision_qkv_physical_workspace_buffer(
                    physical,
                    prepared.workspace_u32_words,
                )?)
            }
            (None, _) => None,
            (Some(_), None) => {
                return Err(RuntimeError::operation(
                    "optimized Q/K/V workspace has no prepared allocation preflight",
                ));
            }
        };
        if let (Some(trace), Some(workspace), Some(buffer)) =
            (trace.as_mut(), prepared_workspace, qkv_workspace.as_ref())
        {
            trace.workspace = Some(VisionQkvWorkspaceEvidence {
                logical_buffer_id: "vision-stack-qkv-workspace".to_owned(),
                buffer_identity: buffer_identity(buffer),
                allocation_bytes: workspace.allocation_bytes(),
                semantic_base: workspace.semantic_base(),
                semantic_bytes: workspace.semantic_bytes(),
            });
        }

        let uniform_stride =
            vision_layer_uniform_stride(self.device.limits().min_uniform_buffer_offset_alignment);
        let uniform_arena_bytes = uniform_stride
            .checked_mul(plan.layer_dispatches.len() as u64)
            .ok_or_else(|| RuntimeError::operation("vision-stack uniform arena overflowed"))?;
        let mut uniform_contents = vec![0_u8; uniform_arena_bytes as usize];
        for (index, dispatch) in plan.layer_dispatches.iter().enumerate() {
            let offset = index * uniform_stride as usize;
            let bytes = dispatch
                .uniform_words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            uniform_contents[offset..offset + VISION_LAYER_UNIFORM_BYTES as usize]
                .copy_from_slice(&bytes);
        }
        if let Some(prepared_execution) = optimized
            .as_ref()
            .and_then(|prepared| prepared.qkv_physical_execution.as_ref())
            .map(VisionQkvPhysicalExecutionSpec::prepared_execution)
        {
            let offset = uniform_stride as usize;
            let bytes = prepared_execution.layers()[0]
                .uniform_words()
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>();
            uniform_contents[offset..offset + VISION_LAYER_UNIFORM_BYTES as usize]
                .copy_from_slice(&bytes);
        }
        let uniform_buffer = self.create_initialized_buffer(
            "vision-stack-uniform-arena",
            &uniform_contents,
            wgpu::BufferUsages::UNIFORM,
        );
        let (total_readback_bytes, readback_elements) = if let Some(prepared) = optimized.as_ref() {
            (
                prepared.total_readback_bytes,
                prepared.readback_f32_elements,
            )
        } else {
            if !plan.readback_bytes.is_multiple_of(4) {
                return Err(RuntimeError::mapping(
                    "legacy vision-stack readback is not F32-aligned",
                ));
            }
            let elements = usize::try_from(plan.readback_bytes / 4)
                .map_err(|_| RuntimeError::mapping("vision-stack readback is too large"))?;
            (plan.readback_bytes, elements)
        };
        let readback_buffer = match qkv_physical_execution {
            Some(physical) => self.create_vision_qkv_physical_readback_buffer(physical),
            None => self.create_buffer(
                "vision-stack-readback",
                total_readback_bytes,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
        };

        let resident_intermediate_bytes =
            plan.layer_dispatches
                .iter()
                .try_fold(0_u64, |bytes, dispatch| {
                    bytes
                        .checked_add(dispatch.invocation.output_bytes)
                        .ok_or_else(|| RuntimeError::operation("vision-layer arena overflowed"))
                })?;
        let layer_plan = VisionEncoderLayerPlan {
            rope_specialization: plan.rope_specialization,
            dispatches: plan.layer_dispatches,
            resident_intermediate_bytes,
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vision-stack-encoder"),
            });
        self.record_optimized_command_encoder(&mut trace, "vision-stack-encoder");
        let mut checkpoint_cursor = 0_usize;
        let mut readback_offset = 0_u64;

        for layer in 0..plan.layer_count {
            let current = &main_buffers[layer % 2];
            let next = &main_buffers[(layer + 1) % 2];
            let weights = &layer_weight_buffers[layer];
            let whole = StorageBufferBinding::entire;
            let scratch = |index| scratch_buffers.binding(&activation.scratch_allocations, index);
            macro_rules! layer_group {
                ($index:expr, $inputs:expr, $output:expr) => {
                    self.create_optimized_vision_layer_bind_group(
                        &layer_plan,
                        $index,
                        &pipelines,
                        $inputs,
                        $output,
                        &uniform_buffer,
                        uniform_stride,
                        layer,
                        &mut trace,
                    )
                };
            }
            let (bind_groups, dispatches) = if fused {
                let physical_execution = qkv_physical_execution.ok_or_else(|| {
                    RuntimeError::operation("missing fused Q/K/V physical execution")
                })?;
                let prepared_execution = physical_execution.prepared_execution();
                let workspace = prepared_workspace
                    .ok_or_else(|| RuntimeError::operation("missing fused Q/K/V workspace plan"))?;
                let workspace_buffer = qkv_workspace
                    .as_ref()
                    .ok_or_else(|| RuntimeError::operation("missing fused Q/K/V workspace"))?;
                let descriptor = &prepared_execution.layers()[layer];
                let semantic_output = StorageBufferBinding::slice(
                    workspace_buffer,
                    workspace.semantic_base(),
                    workspace.semantic_bytes(),
                )?;
                let groups = vec![
                    layer_group!(
                        0,
                        &[whole(current), whole(&weights[0]), whole(&weights[1])],
                        scratch(0)?
                    ),
                    self.create_optimized_stack_bind_group(
                        "vision-layer-qkv-fused-bind-group",
                        &pipelines[&KernelId::VisionQkvFusedF32],
                        &[
                            scratch(0)?,
                            whole(&weights[2]),
                            whole(&weights[3]),
                            whole(&weights[4]),
                            whole(&weights[5]),
                            whole(&weights[6]),
                            whole(&weights[7]),
                        ],
                        semantic_output,
                        &uniform_buffer,
                        uniform_stride,
                        Some(layer),
                        VisionQkvStackStage::QkvFused,
                        &mut trace,
                    ),
                    self.create_vision_qkv_physical_attention_bind_group(
                        physical_execution,
                        layer,
                        &pipelines[&KernelId::VisionAttentionF32],
                        workspace_buffer,
                        &boundary_buffer,
                        scratch(4)?,
                        &uniform_buffer,
                        uniform_stride,
                        &mut trace,
                    )?,
                    layer_group!(
                        5,
                        &[scratch(4)?, whole(&weights[8]), whole(&weights[9])],
                        scratch(5)?
                    ),
                    layer_group!(6, &[whole(current), scratch(5)?], scratch(6)?),
                    layer_group!(
                        7,
                        &[scratch(6)?, whole(&weights[10]), whole(&weights[11])],
                        scratch(7)?
                    ),
                    layer_group!(
                        8,
                        &[scratch(7)?, whole(&weights[12]), whole(&weights[13])],
                        scratch(8)?
                    ),
                    layer_group!(9, &[scratch(8)?], scratch(9)?),
                    layer_group!(
                        10,
                        &[scratch(9)?, whole(&weights[14]), whole(&weights[15])],
                        scratch(10)?
                    ),
                    layer_group!(11, &[scratch(6)?, scratch(10)?], whole(next)),
                ];
                if let Some(trace) = trace.as_mut() {
                    trace.attention_bindings.extend(
                        descriptor
                            .attention_bridge()
                            .bindings()
                            .iter()
                            .map(|binding| VisionQkvAttentionBindingEvidence {
                                layer,
                                binding: binding.binding(),
                                buffer_identity: buffer_identity(workspace_buffer),
                                byte_offset: workspace.semantic_base() + binding.byte_offset(),
                                byte_length: binding.byte_length(),
                            }),
                    );
                }
                let mut dispatches = Vec::with_capacity(10);
                dispatches.push((
                    optimized_stage(layer_plan.dispatches[0].stage),
                    layer_plan.dispatches[0].invocation,
                ));
                dispatches.push((VisionQkvStackStage::QkvFused, descriptor.invocation()));
                dispatches.extend(
                    layer_plan.dispatches[4..]
                        .iter()
                        .map(|dispatch| (optimized_stage(dispatch.stage), dispatch.invocation)),
                );
                (groups, dispatches)
            } else {
                let groups = vec![
                    layer_group!(
                        0,
                        &[whole(current), whole(&weights[0]), whole(&weights[1])],
                        scratch(0)?
                    ),
                    layer_group!(
                        1,
                        &[scratch(0)?, whole(&weights[2]), whole(&weights[3])],
                        scratch(1)?
                    ),
                    layer_group!(
                        2,
                        &[scratch(0)?, whole(&weights[4]), whole(&weights[5])],
                        scratch(2)?
                    ),
                    layer_group!(
                        3,
                        &[scratch(0)?, whole(&weights[6]), whole(&weights[7])],
                        scratch(3)?
                    ),
                    layer_group!(
                        4,
                        &[
                            scratch(1)?,
                            scratch(2)?,
                            scratch(3)?,
                            whole(&boundary_buffer),
                        ],
                        scratch(4)?
                    ),
                    layer_group!(
                        5,
                        &[scratch(4)?, whole(&weights[8]), whole(&weights[9])],
                        scratch(5)?
                    ),
                    layer_group!(6, &[whole(current), scratch(5)?], scratch(6)?),
                    layer_group!(
                        7,
                        &[scratch(6)?, whole(&weights[10]), whole(&weights[11])],
                        scratch(7)?
                    ),
                    layer_group!(
                        8,
                        &[scratch(7)?, whole(&weights[12]), whole(&weights[13])],
                        scratch(8)?
                    ),
                    layer_group!(9, &[scratch(8)?], scratch(9)?),
                    layer_group!(
                        10,
                        &[scratch(9)?, whole(&weights[14]), whole(&weights[15])],
                        scratch(10)?
                    ),
                    layer_group!(11, &[scratch(6)?, scratch(10)?], whole(next)),
                ];
                let dispatches = layer_plan
                    .dispatches
                    .iter()
                    .map(|dispatch| (optimized_stage(dispatch.stage), dispatch.invocation))
                    .collect();
                (groups, dispatches)
            };
            let timestamp_writes = (layer == 0)
                .then(|| {
                    self.timestamp_resources.as_ref().map(|resources| {
                        wgpu::ComputePassTimestampWrites {
                            query_set: &resources.query_set,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        }
                    })
                })
                .flatten();
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vision-stack-layer-pass"),
                    timestamp_writes,
                });
                for ((stage, invocation), bind_group) in dispatches.iter().zip(&bind_groups) {
                    pass.set_pipeline(&pipelines[&invocation.kernel]);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups(
                        invocation.dispatch[0],
                        invocation.dispatch[1],
                        invocation.dispatch[2],
                    );
                    self.record_optimized_dispatch(&mut trace, Some(layer), *stage, *invocation);
                }
            }
            if plan.checkpoint_layers.get(checkpoint_cursor) == Some(&layer) {
                encoder.copy_buffer_to_buffer(
                    next,
                    0,
                    &readback_buffer,
                    readback_offset,
                    hidden_bytes,
                );
                self.record_optimized_copy(
                    &mut trace,
                    next,
                    0,
                    &readback_buffer,
                    readback_offset,
                    hidden_bytes,
                    VisionQkvCopyPurpose::Checkpoint,
                );
                readback_offset += hidden_bytes;
                checkpoint_cursor += 1;
            }
        }

        let final_hidden = &main_buffers[plan.layer_count % 2];
        let post_norm_output = scratch_buffers.binding(&activation.scratch_allocations, 0)?;
        let post_norm_bind_group = self.create_optimized_stack_bind_group(
            "vision-stack-post-norm-bind-group",
            &pipelines[&plan.post_norm_dispatch.kernel],
            &[
                StorageBufferBinding::entire(final_hidden),
                StorageBufferBinding::entire(&post_norm_buffers[0]),
                StorageBufferBinding::entire(&post_norm_buffers[1]),
            ],
            post_norm_output,
            &uniform_buffer,
            0,
            None,
            VisionQkvStackStage::PostNorm,
            &mut trace,
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vision-stack-post-norm-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipelines[&plan.post_norm_dispatch.kernel]);
            pass.set_bind_group(0, &post_norm_bind_group, &[]);
            pass.dispatch_workgroups(
                plan.post_norm_dispatch.dispatch[0],
                plan.post_norm_dispatch.dispatch[1],
                plan.post_norm_dispatch.dispatch[2],
            );
            self.record_optimized_dispatch(
                &mut trace,
                None,
                VisionQkvStackStage::PostNorm,
                plan.post_norm_dispatch,
            );
        }
        let (post_norm_source, post_norm_source_offset) =
            scratch_buffers.copy_source(&activation.scratch_allocations, 0)?;
        encoder.copy_buffer_to_buffer(
            post_norm_source,
            post_norm_source_offset,
            &readback_buffer,
            readback_offset,
            hidden_bytes,
        );
        self.record_optimized_copy(
            &mut trace,
            post_norm_source,
            post_norm_source_offset,
            &readback_buffer,
            readback_offset,
            hidden_bytes,
            VisionQkvCopyPurpose::SemanticOutput,
        );
        readback_offset += hidden_bytes;
        debug_assert_eq!(readback_offset, plan.readback_bytes);
        if let (Some(physical_execution), Some(workspace_buffer)) =
            (qkv_physical_execution, qkv_workspace.as_ref())
        {
            readback_offset = self.copy_vision_qkv_physical_canaries(
                physical_execution,
                &mut encoder,
                workspace_buffer,
                &readback_buffer,
                readback_offset,
                &mut trace,
            );
        }
        debug_assert_eq!(readback_offset, total_readback_bytes);
        if let Some(resources) = self.timestamp_resources.as_ref() {
            encoder.resolve_query_set(&resources.query_set, 0..2, &resources.resolve, 0);
            encoder.copy_buffer_to_buffer(&resources.resolve, 0, &resources.readback, 0, 16);
            self.record_optimized_copy(
                &mut trace,
                &resources.resolve,
                0,
                &resources.readback,
                0,
                16,
                VisionQkvCopyPurpose::TimestampQuery,
            );
        }

        let started = Instant::now();
        let submission_index = self.submit_command_buffers([encoder.finish()]);
        self.observe(RuntimeEvent::ReadbackMapRequested {
            label: "vision-stack-readback".to_owned(),
            bytes: total_readback_bytes,
        });
        let output_receiver = map_read(&readback_buffer);
        self.record_optimized_map(
            &mut trace,
            VisionQkvMapPurpose::SemanticOutput,
            &readback_buffer,
            0,
            total_readback_bytes,
        );
        let timestamp_receiver = self.timestamp_resources.as_ref().map(|resources| {
            let receiver = map_read(&resources.readback);
            self.record_optimized_map(
                &mut trace,
                VisionQkvMapPurpose::TimestampQuery,
                &resources.readback,
                0,
                16,
            );
            receiver
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: Some(GPU_WAIT_TIMEOUT),
            })
            .map_err(|error| RuntimeError::operation(format!("GPU completion failed: {error}")))?;
        await_mapping(output_receiver, "vision-stack readback")?;
        if let Some(receiver) = timestamp_receiver {
            await_mapping(receiver, "vision-stack timestamp query")?;
        }
        let queue_wall_time_ns = elapsed_ns(started.elapsed());

        let flat_values = match qkv_physical_execution {
            Some(physical_execution) => {
                self.read_vision_qkv_physical_readback(physical_execution, &readback_buffer)?
            }
            None => read_f32_buffer(&readback_buffer, readback_elements)?,
        };
        // Consume and unmap the optional timestamp readback before inspecting
        // canaries. A canary failure is terminal for this invocation, but must
        // not leave a reusable runtime buffer mapped for the next invocation.
        let timestamp = self
            .timestamp_resources
            .as_ref()
            .map(|resources| {
                let ticks = read_u64_buffer(&resources.readback)?;
                let period_ns = f64::from(self.queue.get_timestamp_period());
                Ok(GpuTimestamp {
                    begin_ticks: ticks[0],
                    end_ticks: ticks[1],
                    period_ns,
                    duration_ns: ticks[1].saturating_sub(ticks[0]) as f64 * period_ns,
                })
            })
            .transpose()?;
        let hidden_elements = plan.post_norm_dispatch.output_elements;
        let mut checkpoints = BTreeMap::new();
        let mut element_offset = 0_usize;
        for &layer in &plan.checkpoint_layers {
            let end = element_offset + hidden_elements;
            checkpoints.insert(layer, flat_values[element_offset..end].to_vec());
            element_offset = end;
        }
        let output_end = element_offset + hidden_elements;
        let output = flat_values[element_offset..output_end].to_vec();
        debug_assert_eq!(
            u64::try_from(output_end).unwrap_or(u64::MAX) * 4,
            plan.readback_bytes
        );

        if let (Some(workspace), Some(workspace_buffer)) =
            (prepared_workspace, qkv_workspace.as_ref())
        {
            let mut canary_element_offset = output_end;
            let canaries = workspace
                .canaries()
                .iter()
                .map(|canary| {
                    let element_count = usize::try_from(canary.byte_length() / 4)
                        .map_err(|_| RuntimeError::mapping("Q/K/V canary is too large"))?;
                    let end = canary_element_offset
                        .checked_add(element_count)
                        .ok_or_else(|| {
                            RuntimeError::mapping("Q/K/V canary readback range overflowed")
                        })?;
                    let values = flat_values.get(canary_element_offset..end).ok_or_else(|| {
                        RuntimeError::mapping("Q/K/V canary readback range was truncated")
                    })?;
                    canary_element_offset = end;
                    Ok(VisionQkvCanaryEvidence {
                        kind: canary.kind(),
                        byte_offset: canary.byte_offset(),
                        byte_length: canary.byte_length(),
                        expected_bits: VISION_QKV_CANARY_U32,
                        passed: values
                            .iter()
                            .all(|value| value.to_bits() == VISION_QKV_CANARY_U32),
                    })
                })
                .collect::<Result<Vec<_>, RuntimeError>>()?;
            if canary_element_offset != flat_values.len() {
                return Err(RuntimeError::mapping(
                    "Q/K/V canary readback did not cover the guarded evidence tail",
                ));
            }
            if let Some(evidence) = trace.as_mut() {
                evidence.canaries = canaries.clone();
            }
            self.observe(RuntimeEvent::CanaryChecked {
                buffer_identity: buffer_identity(workspace_buffer),
                canaries: canaries.clone(),
            });
            if canaries.iter().any(|canary| !canary.passed) {
                return Err(RuntimeError::operation(
                    "fused Q/K/V workspace canary was modified",
                ));
            }
        } else {
            debug_assert_eq!(output_end, flat_values.len());
        }

        if let Some(timestamp) = timestamp
            && (timestamp.begin_ticks == 0 || timestamp.end_ticks <= timestamp.begin_ticks)
        {
            return Err(RuntimeError::operation(format!(
                "resident vision-stack timestamp query was zero or reversed: begin={}, end={}",
                timestamp.begin_ticks, timestamp.end_ticks
            )));
        }
        let timestamp_fresh = timestamp.map(|timestamp| self.accept_fresh_timestamp(timestamp));
        let after = self.counters();
        let weight_buffer_count = plan
            .layer_count
            .checked_mul(16)
            .and_then(|count| count.checked_add(2))
            .ok_or_else(|| RuntimeError::operation("vision-stack weight count overflowed"))?;

        let actual_dispatch_count = trace.as_ref().map_or(plan.dispatch_count, |evidence| {
            evidence.encoded_dispatches.len()
        });
        if let Some(evidence) = trace.as_mut() {
            evidence.dispatch_count = actual_dispatch_count;
            evidence.map_count = evidence.map_requests.len();
        }
        let execution = VisionStackExecution {
            checkpoints,
            output,
            diagnostics: VisionStackDiagnostics {
                checked_error_scopes: CHECKED_SCOPE_ORDER,
                captured_errors: Vec::new(),
                queue_wall_time_ns,
                timestamp,
                timestamp_fresh,
                shader_blake3,
                rope_specialization: plan.rope_specialization,
                layer_count: plan.layer_count,
                checkpoint_layers: plan.checkpoint_layers,
                dispatch_count: actual_dispatch_count,
                compute_pass_count: plan.compute_pass_count,
                submission_count: after.submissions - before.submissions,
                command_buffer_count: 1,
                buffer_allocation_count: after.buffer_allocations - before.buffer_allocations,
                weight_buffer_count,
                activation_strategy: activation.strategy,
                activation_buffer_count: activation.activation_buffer_count,
                activation_arena_bytes: activation.total_activation_bytes,
                scratch_arena_bytes: activation.scratch_arena_bytes,
                main_buffers_bytes: activation.main_buffers_bytes,
                scratch_allocations: activation.scratch_allocations,
                readback_buffer_count: 1,
                readback_map_count: 1,
                readback_bytes: total_readback_bytes,
            },
        };
        Ok((execution, trace))
    }

    fn create_vision_stack_layer_weight_buffers(
        &self,
        layer: usize,
        parameters: VisionEncoderLayerParameters<'_>,
    ) -> [wgpu::Buffer; 16] {
        let entries = [
            ("norm1-weight", parameters.norm1.weight),
            ("norm1-bias", parameters.norm1.bias),
            ("query-weight", parameters.query.weight),
            ("query-bias", parameters.query.bias),
            ("key-weight", parameters.key.weight),
            ("key-bias", parameters.key.bias),
            ("value-weight", parameters.value.weight),
            ("value-bias", parameters.value.bias),
            (
                "attention-output-weight",
                parameters.attention_output.weight,
            ),
            ("attention-output-bias", parameters.attention_output.bias),
            ("norm2-weight", parameters.norm2.weight),
            ("norm2-bias", parameters.norm2.bias),
            ("mlp-fc1-weight", parameters.mlp_fc1.weight),
            ("mlp-fc1-bias", parameters.mlp_fc1.bias),
            ("mlp-fc2-weight", parameters.mlp_fc2.weight),
            ("mlp-fc2-bias", parameters.mlp_fc2.bias),
        ];
        std::array::from_fn(|index| {
            let (name, values) = entries[index];
            self.create_initialized_buffer(
                &format!("vision-stack-layer-{layer:02}-{name}"),
                bytemuck::cast_slice(values),
                wgpu::BufferUsages::STORAGE,
            )
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
        let inputs = inputs
            .iter()
            .map(|input| StorageBufferBinding::entire(input))
            .collect::<Vec<_>>();
        self.create_vision_layer_bind_group_with_bindings(
            plan,
            index,
            pipelines,
            &inputs,
            StorageBufferBinding::entire(output),
            uniform_buffer,
            uniform_stride,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_vision_layer_bind_group_with_bindings(
        &self,
        plan: &VisionEncoderLayerPlan,
        index: usize,
        pipelines: &BTreeMap<KernelId, wgpu::ComputePipeline>,
        inputs: &[StorageBufferBinding<'_>],
        output: StorageBufferBinding<'_>,
        uniform_buffer: &wgpu::Buffer,
        uniform_stride: u64,
    ) -> wgpu::BindGroup {
        let dispatch = plan.dispatches[index];
        let pipeline = &pipelines[&dispatch.invocation.kernel];
        self.create_staged_bind_group_with_bindings(
            &format!("vision-layer-{}-bind-group", dispatch.stage.as_str()),
            pipeline,
            inputs,
            output,
            uniform_buffer,
            index as u64 * uniform_stride,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_optimized_vision_layer_bind_group(
        &self,
        plan: &VisionEncoderLayerPlan,
        index: usize,
        pipelines: &BTreeMap<KernelId, wgpu::ComputePipeline>,
        inputs: &[StorageBufferBinding<'_>],
        output: StorageBufferBinding<'_>,
        uniform_buffer: &wgpu::Buffer,
        uniform_stride: u64,
        layer: usize,
        trace: &mut Option<VisionQkvStackExecutionEvidence>,
    ) -> wgpu::BindGroup {
        let dispatch = plan.dispatches[index];
        self.create_optimized_stack_bind_group(
            &format!("vision-layer-{}-bind-group", dispatch.stage.as_str()),
            &pipelines[&dispatch.invocation.kernel],
            inputs,
            output,
            uniform_buffer,
            index as u64 * uniform_stride,
            Some(layer),
            optimized_stage(dispatch.stage),
            trace,
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
        let inputs = inputs
            .iter()
            .map(|input| StorageBufferBinding::entire(input))
            .collect::<Vec<_>>();
        self.create_staged_bind_group_with_bindings(
            label,
            pipeline,
            &inputs,
            StorageBufferBinding::entire(output),
            uniform_buffer,
            uniform_offset,
        )
    }

    fn create_staged_bind_group_with_bindings(
        &self,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        inputs: &[StorageBufferBinding<'_>],
        output: StorageBufferBinding<'_>,
        uniform_buffer: &wgpu::Buffer,
        uniform_offset: u64,
    ) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        let mut entries = Vec::with_capacity(inputs.len() + 2);
        for (binding, input) in inputs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: wgpu::BindingResource::Buffer(input.as_wgpu()),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: wgpu::BindingResource::Buffer(output.as_wgpu()),
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

    fn accept_fresh_timestamp(&self, timestamp: GpuTimestamp) -> bool {
        if timestamp.begin_ticks == 0 || timestamp.end_ticks <= timestamp.begin_ticks {
            return false;
        }
        let pair = (timestamp.begin_ticks, timestamp.end_ticks);
        let mut previous = self
            .last_timestamp
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if previous.as_ref() == Some(&pair) {
            return false;
        }
        *previous = Some(pair);
        true
    }

    fn pipeline(
        &self,
        label: &str,
        source: &str,
        entry_point: &str,
        cached_kernel: Option<KernelId>,
    ) -> wgpu::ComputePipeline {
        self.pipeline_with_creation_status(label, source, entry_point, cached_kernel)
            .0
    }

    fn pipeline_with_creation_status(
        &self,
        label: &str,
        source: &str,
        entry_point: &str,
        cached_kernel: Option<KernelId>,
    ) -> (wgpu::ComputePipeline, bool) {
        if let Some(kernel) = cached_kernel
            && let Some(pipeline) = self
                .pipelines
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get(&kernel)
                .cloned()
        {
            return (pipeline, false);
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
            self.pipelines
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(kernel, pipeline.clone());
        }
        (pipeline, true)
    }

    fn create_vision_qkv_physical_workspace_buffer(
        &self,
        physical: &VisionQkvPhysicalExecutionSpec,
        workspace_u32_words: usize,
    ) -> Result<wgpu::Buffer, RuntimeError> {
        let prepared_execution = physical.prepared_execution();
        let workspace = prepared_execution.workspace();
        let allocation_bytes = workspace.allocation_bytes();
        let buffer = self.create_buffer(
            "vision-stack-qkv-workspace",
            allocation_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let canary_words = vec![VISION_QKV_CANARY_U32; workspace_u32_words];
        let canary_bytes = bytemuck::cast_slice(&canary_words);
        if canary_bytes.len() as u64 != allocation_bytes {
            return Err(RuntimeError::operation(
                "prepared Q/K/V workspace initialization length differs from the sealed allocation",
            ));
        }
        self.write_buffer(
            "vision-qkv-physical-workspace-initialization",
            &buffer,
            0,
            canary_bytes,
        );
        Ok(buffer)
    }

    fn create_vision_qkv_physical_readback_buffer(
        &self,
        physical: &VisionQkvPhysicalExecutionSpec,
    ) -> wgpu::Buffer {
        let readback_layout = physical.readback_layout();
        let total_readback_bytes = readback_layout.total_readback_bytes();
        self.create_buffer(
            "vision-stack-readback",
            total_readback_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_vision_qkv_physical_attention_bind_group(
        &self,
        physical: &VisionQkvPhysicalExecutionSpec,
        layer: usize,
        pipeline: &wgpu::ComputePipeline,
        workspace_buffer: &wgpu::Buffer,
        boundary_buffer: &wgpu::Buffer,
        output: StorageBufferBinding<'_>,
        uniform_buffer: &wgpu::Buffer,
        uniform_stride: u64,
        trace: &mut Option<VisionQkvStackExecutionEvidence>,
    ) -> Result<wgpu::BindGroup, RuntimeError> {
        let prepared_execution = physical.prepared_execution();
        let descriptor = prepared_execution
            .layers()
            .get(layer)
            .ok_or_else(|| RuntimeError::operation("missing physical Q/K/V layer descriptor"))?;
        let attention_bridge = descriptor.attention_bridge();
        let bindings = attention_bridge.bindings();
        let workspace = prepared_execution.workspace();
        let mut inputs = bindings
            .iter()
            .map(|binding| {
                StorageBufferBinding::slice(
                    workspace_buffer,
                    workspace.semantic_base() + binding.byte_offset(),
                    binding.byte_length(),
                )
            })
            .collect::<Result<Vec<_>, RuntimeError>>()?;
        inputs.push(StorageBufferBinding::entire(boundary_buffer));
        let uniform_offset = 4 * uniform_stride;
        let layout = pipeline.get_bind_group_layout(0);
        let mut entries = Vec::with_capacity(inputs.len() + 2);
        for (binding, input) in inputs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: wgpu::BindingResource::Buffer(input.as_wgpu()),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32,
            resource: wgpu::BindingResource::Buffer(output.as_wgpu()),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: inputs.len() as u32 + 1,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: uniform_buffer,
                offset: uniform_offset,
                size: wgpu::BufferSize::new(VISION_LAYER_UNIFORM_BYTES),
            }),
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vision-layer-attention-bind-group"),
            layout: &layout,
            entries: &entries,
        });
        self.record_optimized_bind_group_creation(
            &inputs,
            output,
            uniform_buffer,
            uniform_offset,
            Some(layer),
            VisionQkvStackStage::AttentionContext,
            trace,
        );
        Ok(bind_group)
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_vision_qkv_physical_canaries(
        &self,
        physical: &VisionQkvPhysicalExecutionSpec,
        encoder: &mut wgpu::CommandEncoder,
        workspace_buffer: &wgpu::Buffer,
        readback_buffer: &wgpu::Buffer,
        semantic_end: u64,
        trace: &mut Option<VisionQkvStackExecutionEvidence>,
    ) -> u64 {
        let readback_layout = physical.readback_layout();
        let qkv_canary_offset = readback_layout.qkv_canary_offset();
        debug_assert_eq!(semantic_end, qkv_canary_offset);
        let mut destination_offset = qkv_canary_offset;
        let prepared_execution = physical.prepared_execution();
        let workspace = prepared_execution.workspace();
        let canaries = workspace.canaries();
        for canary in canaries {
            encoder.copy_buffer_to_buffer(
                workspace_buffer,
                canary.byte_offset(),
                readback_buffer,
                destination_offset,
                canary.byte_length(),
            );
            self.record_optimized_copy(
                trace,
                workspace_buffer,
                canary.byte_offset(),
                readback_buffer,
                destination_offset,
                canary.byte_length(),
                VisionQkvCopyPurpose::CanaryEvidence,
            );
            destination_offset += canary.byte_length();
        }
        destination_offset
    }

    fn read_vision_qkv_physical_readback(
        &self,
        physical: &VisionQkvPhysicalExecutionSpec,
        buffer: &wgpu::Buffer,
    ) -> Result<Vec<f32>, RuntimeError> {
        let readback_layout = physical.readback_layout();
        let total_readback_bytes = readback_layout.total_readback_bytes();
        let readback_elements = readback_layout.readback_f32_elements();
        let mapped = match buffer.slice(0..total_readback_bytes).get_mapped_range() {
            Ok(mapped) => mapped,
            Err(error) => {
                buffer.unmap();
                return Err(RuntimeError::mapping(format!(
                    "cannot view mapped physical Q/K/V output: {error}"
                )));
            }
        };
        let values = mapped
            .chunks_exact(4)
            .take(readback_elements)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 occupies four bytes")))
            .collect();
        drop(mapped);
        buffer.unmap();
        Ok(values)
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
        self.record_buffer(label, contents.len() as u64);
        buffer
    }

    fn write_buffer(&self, label: &str, buffer: &wgpu::Buffer, byte_offset: u64, bytes: &[u8]) {
        self.queue.write_buffer(buffer, byte_offset, bytes);
        self.queue_writes.fetch_add(1, Ordering::Relaxed);
        self.observe(RuntimeEvent::QueueBufferWritten {
            label: label.to_owned(),
            buffer_identity: buffer_identity(buffer),
            byte_offset,
            byte_length: bytes.len() as u64,
        });
    }

    fn submit_command_buffers<const COUNT: usize>(
        &self,
        command_buffers: [wgpu::CommandBuffer; COUNT],
    ) -> wgpu::SubmissionIndex {
        let submission_index = self.queue.submit(command_buffers);
        let submission = self.submissions.fetch_add(1, Ordering::Relaxed) + 1;
        self.observe(RuntimeEvent::SubmissionQueued {
            submission,
            command_buffers: u32::try_from(COUNT).unwrap_or(u32::MAX),
        });
        submission_index
    }

    fn create_buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        self.record_buffer(label, size);
        buffer
    }

    fn record_buffer(&self, label: &str, bytes: u64) {
        self.buffer_allocations.fetch_add(1, Ordering::Relaxed);
        self.observe(RuntimeEvent::BufferAllocated {
            label: label.to_owned(),
            bytes,
        });
    }

    fn record_optimized_pipeline_creation(
        &self,
        trace: &mut Option<VisionQkvStackExecutionEvidence>,
        kernel: KernelId,
        shader_blake3: [u8; 32],
    ) {
        let Some(trace) = trace.as_mut() else {
            return;
        };
        self.pipeline_creations.fetch_add(1, Ordering::Relaxed);
        let evidence = VisionQkvPipelineCreationEvidence {
            kernel,
            shader_blake3,
        };
        trace.pipeline_creations.push(evidence);
        self.observe(RuntimeEvent::PipelineCreated {
            kernel,
            shader_blake3,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn create_optimized_stack_bind_group(
        &self,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        inputs: &[StorageBufferBinding<'_>],
        output: StorageBufferBinding<'_>,
        uniform_buffer: &wgpu::Buffer,
        uniform_offset: u64,
        layer: Option<usize>,
        stage: VisionQkvStackStage,
        trace: &mut Option<VisionQkvStackExecutionEvidence>,
    ) -> wgpu::BindGroup {
        let bind_group = self.create_staged_bind_group_with_bindings(
            label,
            pipeline,
            inputs,
            output,
            uniform_buffer,
            uniform_offset,
        );
        self.record_optimized_bind_group_creation(
            inputs,
            output,
            uniform_buffer,
            uniform_offset,
            layer,
            stage,
            trace,
        );
        bind_group
    }

    #[allow(clippy::too_many_arguments)]
    fn record_optimized_bind_group_creation(
        &self,
        inputs: &[StorageBufferBinding<'_>],
        output: StorageBufferBinding<'_>,
        uniform_buffer: &wgpu::Buffer,
        uniform_offset: u64,
        layer: Option<usize>,
        stage: VisionQkvStackStage,
        trace: &mut Option<VisionQkvStackExecutionEvidence>,
    ) {
        let Some(trace) = trace.as_mut() else {
            return;
        };
        let mut bindings = inputs
            .iter()
            .enumerate()
            .map(|(binding, input)| optimized_binding_evidence(binding, *input))
            .collect::<Vec<_>>();
        bindings.push(optimized_binding_evidence(inputs.len(), output));
        bindings.push(VisionQkvBufferBindingEvidence {
            binding: u32::try_from(inputs.len() + 1).unwrap_or(u32::MAX),
            buffer_identity: buffer_identity(uniform_buffer),
            byte_offset: uniform_offset,
            byte_length: VISION_LAYER_UNIFORM_BYTES,
        });
        self.bind_group_creations.fetch_add(1, Ordering::Relaxed);
        trace
            .bind_group_creations
            .push(VisionQkvBindGroupCreationEvidence {
                layer,
                stage,
                bindings: bindings.clone(),
            });
        self.observe(RuntimeEvent::BindGroupCreated {
            layer,
            stage,
            bindings,
        });
    }

    fn record_optimized_command_encoder(
        &self,
        trace: &mut Option<VisionQkvStackExecutionEvidence>,
        label: &str,
    ) {
        let Some(trace) = trace.as_mut() else {
            return;
        };
        self.command_encoder_creations
            .fetch_add(1, Ordering::Relaxed);
        trace
            .command_encoder_creations
            .push(VisionQkvCommandEncoderCreationEvidence {
                label: label.to_owned(),
            });
        self.observe(RuntimeEvent::CommandEncoderCreated {
            label: label.to_owned(),
        });
    }

    fn record_optimized_dispatch(
        &self,
        trace: &mut Option<VisionQkvStackExecutionEvidence>,
        layer: Option<usize>,
        stage: VisionQkvStackStage,
        invocation: InvocationPlan,
    ) {
        let Some(trace) = trace.as_mut() else {
            return;
        };
        let ordinal = trace.encoded_dispatches.len();
        self.dispatch_encodings.fetch_add(1, Ordering::Relaxed);
        trace.encoded_dispatches.push(VisionQkvDispatchEvidence {
            ordinal,
            layer,
            stage,
            kernel: invocation.kernel,
            workgroups: invocation.dispatch,
        });
        self.observe(RuntimeEvent::DispatchEncoded {
            ordinal,
            layer,
            stage,
            kernel: invocation.kernel,
            workgroups: invocation.dispatch,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn record_optimized_copy(
        &self,
        trace: &mut Option<VisionQkvStackExecutionEvidence>,
        source: &wgpu::Buffer,
        source_offset: u64,
        destination: &wgpu::Buffer,
        destination_offset: u64,
        byte_length: u64,
        purpose: VisionQkvCopyPurpose,
    ) {
        let Some(trace) = trace.as_mut() else {
            return;
        };
        let ordinal = trace.encoded_copies.len();
        let after_dispatch_ordinal = trace
            .encoded_dispatches
            .last()
            .map_or(0, |dispatch| dispatch.ordinal);
        let evidence = VisionQkvCopyEvidence {
            ordinal,
            source_buffer_identity: buffer_identity(source),
            source_offset,
            destination_buffer_identity: buffer_identity(destination),
            destination_offset,
            byte_length,
            purpose,
            after_dispatch_ordinal,
        };
        self.buffer_copy_encodings.fetch_add(1, Ordering::Relaxed);
        trace.encoded_copies.push(evidence.clone());
        self.observe(RuntimeEvent::BufferCopyEncoded {
            ordinal,
            source_buffer_identity: evidence.source_buffer_identity,
            source_offset,
            destination_buffer_identity: evidence.destination_buffer_identity,
            destination_offset,
            byte_length,
            purpose,
            after_dispatch_ordinal,
        });
    }

    fn record_optimized_map(
        &self,
        trace: &mut Option<VisionQkvStackExecutionEvidence>,
        purpose: VisionQkvMapPurpose,
        buffer: &wgpu::Buffer,
        byte_offset: u64,
        byte_length: u64,
    ) {
        let Some(trace) = trace.as_mut() else {
            return;
        };
        let evidence = VisionQkvMapEvidence {
            purpose,
            buffer_identity: buffer_identity(buffer),
            byte_offset,
            byte_length,
        };
        self.map_requests.fetch_add(1, Ordering::Relaxed);
        trace.map_requests.push(evidence.clone());
        self.observe(RuntimeEvent::MapRequested {
            purpose,
            buffer_identity: evidence.buffer_identity,
            byte_offset,
            byte_length,
        });
    }

    fn observe(&self, event: RuntimeEvent) {
        if let Some(observer) = &self.observer {
            observer.on_event(event);
        }
    }
}

struct WgpuScopeDriver {
    device: wgpu::Device,
    observer: Option<Arc<dyn RuntimeObserver>>,
    guards: Vec<(ErrorScopeKind, wgpu::ErrorScopeGuard)>,
}

impl WgpuScopeDriver {
    fn new(device: wgpu::Device, observer: Option<Arc<dyn RuntimeObserver>>) -> Self {
        Self {
            device,
            observer,
            guards: Vec::with_capacity(ERROR_SCOPE_ORDER.len()),
        }
    }

    fn observe(&self, event: RuntimeEvent) {
        if let Some(observer) = &self.observer {
            observer.on_event(event);
        }
    }
}

impl ErrorScopeDriver for WgpuScopeDriver {
    fn push_scope(&mut self, scope: ErrorScopeKind) {
        let guard = self.device.push_error_scope(scope.error_filter());
        self.guards.push((scope, guard));
        self.observe(RuntimeEvent::ScopePushed(scope));
    }

    fn pop_scope(&mut self, scope: ErrorScopeKind) -> Option<String> {
        let (actual_scope, guard) = self
            .guards
            .pop()
            .expect("error scopes must be popped after they are pushed");
        assert_eq!(
            actual_scope, scope,
            "error scopes must unwind in LIFO order"
        );
        let error = pollster::block_on(guard.pop()).map(|error| error.to_string());
        self.observe(RuntimeEvent::ScopePopped {
            scope,
            captured_error: error.is_some(),
        });
        error
    }
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

fn invocation_input_bytes<'a>(input: &InvocationInput<'a>) -> &'a [u8] {
    match input {
        InvocationInput::F32(values) => bytemuck::cast_slice(values),
        InvocationInput::U32(values) => bytemuck::cast_slice(values),
    }
}

fn single_kernel_output_contents(
    initializer: SingleKernelOutputInitializer<'_>,
    plan: InvocationPlan,
) -> Result<Vec<u8>, RuntimeError> {
    let output_bytes = usize::try_from(plan.output_bytes).map_err(|_| {
        RuntimeError::operation(format!(
            "{} output size {} does not fit the native host address space",
            plan.kernel, plan.output_bytes
        ))
    })?;
    match initializer {
        SingleKernelOutputInitializer::Zero => Ok(vec![0_u8; output_bytes]),
        SingleKernelOutputInitializer::F32(values) => {
            let bytes = bytemuck::cast_slice(values);
            if bytes.len() != output_bytes {
                return Err(RuntimeError::operation(format!(
                    "{} output initializer has {} bytes but the plan requires {output_bytes}",
                    plan.kernel,
                    bytes.len()
                )));
            }
            Ok(bytes.to_vec())
        }
        SingleKernelOutputInitializer::FillBits(bits) => {
            if !output_bytes.is_multiple_of(4) {
                return Err(RuntimeError::operation(format!(
                    "{} output size {output_bytes} is not f32-aligned",
                    plan.kernel
                )));
            }
            let mut bytes = vec![0_u8; output_bytes];
            let word = bits.to_le_bytes();
            for chunk in bytes.chunks_exact_mut(4) {
                chunk.copy_from_slice(&word);
            }
            Ok(bytes)
        }
    }
}

fn invalid_optimized_invocation(message: impl Into<String>) -> RuntimeError {
    RuntimeError::new(RuntimeErrorCode::InvalidInvocation, None, message)
}

fn map_vision_qkv_prepared_execution_error(
    error: pvlc_passes::VisionQkvPreparedExecutionError,
) -> RuntimeError {
    invalid_optimized_invocation(error.to_string())
}

fn buffer_identity(buffer: &wgpu::Buffer) -> u64 {
    buffer as *const wgpu::Buffer as usize as u64
}

fn decoder_buffer_evidence(
    role: DecoderCachedGqaBufferRole,
    buffer: &wgpu::Buffer,
) -> DecoderCachedGqaBufferEvidence {
    DecoderCachedGqaBufferEvidence {
        role,
        buffer_identity: buffer_identity(buffer),
        allocation_bytes: buffer.size(),
    }
}

fn decoder_binding_evidence(
    binding: u32,
    buffer: &wgpu::Buffer,
) -> DecoderCachedGqaBindingEvidence {
    DecoderCachedGqaBindingEvidence {
        binding,
        buffer_identity: buffer_identity(buffer),
        byte_offset: 0,
        byte_length: buffer.size(),
    }
}

fn decoder_copy_evidence(
    ordinal: usize,
    source: &wgpu::Buffer,
    destination: &wgpu::Buffer,
    destination_offset: u64,
    byte_length: u64,
    purpose: DecoderCachedGqaCopyPurpose,
) -> DecoderCachedGqaCopyEvidence {
    DecoderCachedGqaCopyEvidence {
        ordinal,
        source_buffer_identity: buffer_identity(source),
        source_offset: 0,
        destination_buffer_identity: buffer_identity(destination),
        destination_offset,
        byte_length,
        purpose,
        after_dispatch_ordinal: 2,
    }
}

fn optimized_binding_evidence(
    binding: usize,
    resource: StorageBufferBinding<'_>,
) -> VisionQkvBufferBindingEvidence {
    VisionQkvBufferBindingEvidence {
        binding: u32::try_from(binding).unwrap_or(u32::MAX),
        buffer_identity: buffer_identity(resource.buffer),
        byte_offset: resource.offset,
        byte_length: resource.size.map_or_else(
            || resource.buffer.size().saturating_sub(resource.offset),
            wgpu::BufferSize::get,
        ),
    }
}

const fn optimized_stage(stage: VisionEncoderLayerStage) -> VisionQkvStackStage {
    match stage {
        VisionEncoderLayerStage::Norm1 => VisionQkvStackStage::Norm1,
        VisionEncoderLayerStage::Query => VisionQkvStackStage::Query,
        VisionEncoderLayerStage::Key => VisionQkvStackStage::Key,
        VisionEncoderLayerStage::Value => VisionQkvStackStage::Value,
        VisionEncoderLayerStage::AttentionContext => VisionQkvStackStage::AttentionContext,
        VisionEncoderLayerStage::AttentionOutput => VisionQkvStackStage::AttentionOutput,
        VisionEncoderLayerStage::AttentionResidual => VisionQkvStackStage::AttentionResidual,
        VisionEncoderLayerStage::Norm2 => VisionQkvStackStage::Norm2,
        VisionEncoderLayerStage::MlpFc1 => VisionQkvStackStage::MlpFc1,
        VisionEncoderLayerStage::MlpActivation => VisionQkvStackStage::MlpActivation,
        VisionEncoderLayerStage::MlpOutput => VisionQkvStackStage::MlpOutput,
        VisionEncoderLayerStage::Output => VisionQkvStackStage::Output,
    }
}

fn map_read(buffer: &wgpu::Buffer) -> mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
    let (sender, receiver) = mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    receiver
}

fn await_mapping(
    receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    label: &str,
) -> Result<(), RuntimeError> {
    receiver
        .recv_timeout(GPU_WAIT_TIMEOUT)
        .map_err(|error| {
            RuntimeError::mapping(format!("{label} mapping callback failed: {error}"))
        })?
        .map_err(|error| RuntimeError::mapping(format!("{label} mapping failed: {error}")))
}

fn read_f32_buffer(buffer: &wgpu::Buffer, elements: usize) -> Result<Vec<f32>, RuntimeError> {
    let mapped = buffer
        .slice(..)
        .get_mapped_range()
        .map_err(|error| RuntimeError::mapping(format!("cannot view mapped output: {error}")))?;
    let values = mapped
        .chunks_exact(4)
        .take(elements)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 occupies four bytes")))
        .collect();
    drop(mapped);
    buffer.unmap();
    Ok(values)
}

fn read_u64_buffer(buffer: &wgpu::Buffer) -> Result<[u64; 2], RuntimeError> {
    let mapped = buffer.slice(..).get_mapped_range().map_err(|error| {
        RuntimeError::mapping(format!("cannot view mapped timestamps: {error}"))
    })?;
    let mut chunks = mapped.chunks_exact(8);
    let values = [
        u64::from_le_bytes(
            chunks
                .next()
                .expect("timestamp buffer contains a begin tick")
                .try_into()
                .expect("u64 occupies eight bytes"),
        ),
        u64::from_le_bytes(
            chunks
                .next()
                .expect("timestamp buffer contains an end tick")
                .try_into()
                .expect("u64 occupies eight bytes"),
        ),
    ];
    drop(mapped);
    buffer.unmap();
    Ok(values)
}

fn elapsed_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos())
        .unwrap_or(u64::MAX)
        .max(1)
}

fn validate_capture_target(path: &Path) -> Result<(), RuntimeError> {
    if !path.is_absolute() {
        return Err(RuntimeError::capture(
            "Metal capture target must be an absolute path",
        ));
    }
    if path.extension().and_then(|extension| extension.to_str()) != Some("gputrace") {
        return Err(RuntimeError::capture(
            "Metal capture target must use the .gputrace extension",
        ));
    }
    let parent = path.parent().ok_or_else(|| {
        RuntimeError::capture("Metal capture target must have a parent directory")
    })?;
    let parent_metadata = fs::metadata(parent).map_err(|error| {
        RuntimeError::capture(format!(
            "Metal capture parent {} is unavailable: {error}",
            parent.display()
        ))
    })?;
    if !parent_metadata.is_dir() {
        return Err(RuntimeError::capture(format!(
            "Metal capture parent {} is not a directory",
            parent.display()
        )));
    }
    match fs::symlink_metadata(path) {
        Ok(_) => Err(RuntimeError::capture(format!(
            "Metal capture target {} already exists",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(RuntimeError::capture(format!(
            "cannot inspect Metal capture target {}: {error}",
            path.display()
        ))),
    }
}

fn wait_for_capture_artifact(path: &Path) -> Result<(u64, u64), RuntimeError> {
    let started = Instant::now();
    loop {
        match fs::symlink_metadata(path) {
            Ok(_) => return capture_artifact_inventory(path),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if started.elapsed() >= CAPTURE_ARTIFACT_TIMEOUT {
                    return Err(RuntimeError::capture(format!(
                        "Metal did not materialize {} within {:?}",
                        path.display(),
                        CAPTURE_ARTIFACT_TIMEOUT
                    )));
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(error) => {
                return Err(RuntimeError::capture(format!(
                    "cannot inspect Metal capture artifact {}: {error}",
                    path.display()
                )));
            }
        }
    }
}

fn capture_artifact_inventory(path: &Path) -> Result<(u64, u64), RuntimeError> {
    let root = fs::symlink_metadata(path).map_err(|error| {
        RuntimeError::capture(format!(
            "cannot inspect Metal capture artifact {}: {error}",
            path.display()
        ))
    })?;
    if root.file_type().is_symlink() {
        return Err(RuntimeError::capture(
            "Metal capture artifact may not be a symbolic link",
        ));
    }
    if root.is_file() {
        return Ok((1, root.len()));
    }
    if !root.is_dir() {
        return Err(RuntimeError::capture(
            "Metal capture artifact is neither a file nor a package directory",
        ));
    }

    let mut file_count = 0_u64;
    let mut byte_count = 0_u64;
    let mut pending = vec![path.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory).map_err(|error| {
            RuntimeError::capture(format!(
                "cannot enumerate Metal capture package {}: {error}",
                directory.display()
            ))
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                RuntimeError::capture(format!("cannot read Metal capture entry: {error}"))
            })?;
            let metadata = fs::symlink_metadata(entry.path()).map_err(|error| {
                RuntimeError::capture(format!(
                    "cannot inspect Metal capture entry {}: {error}",
                    entry.path().display()
                ))
            })?;
            if metadata.file_type().is_symlink() {
                return Err(RuntimeError::capture(format!(
                    "Metal capture package contains symbolic link {}",
                    entry.path().display()
                )));
            }
            if metadata.is_dir() {
                pending.push(entry.path());
            } else if metadata.is_file() {
                file_count = file_count
                    .checked_add(1)
                    .ok_or_else(|| RuntimeError::capture("Metal capture file count overflowed"))?;
                byte_count = byte_count
                    .checked_add(metadata.len())
                    .ok_or_else(|| RuntimeError::capture("Metal capture byte count overflowed"))?;
            } else {
                return Err(RuntimeError::capture(format!(
                    "Metal capture package contains unsupported entry {}",
                    entry.path().display()
                )));
            }
        }
    }
    Ok((file_count, byte_count))
}

fn remove_capture_artifact(path: &Path) -> std::io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            fs::remove_dir_all(path)
        }
        Ok(_) => fs::remove_file(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}
