#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    collections::BTreeMap,
    error::Error,
    fmt,
    rc::Rc,
    str::FromStr,
};

use pvlc_ir::SemanticGraph;
use pvlc_model_schema::PaddleOcrVl16Schema;
use pvlc_pack::{
    VisionStackLayerTensorRange, VisionStackShardError, VisionStackShardErrorCode,
    VisionStackShardManifest, VisionStackShardOracle, VisionStackShardPlan,
    VisionStackShardProtocolPhase, parse_vision_stack_shard_manifest,
    vision_stack_layer_weight_ranges_with_vector_storage,
};
use pvlc_passes::{
    VisionQkvPhysicalExecutionSpec, VisionQkvPreparedExecutionError,
    VisionQkvPreparedExecutionErrorCode, VisionQkvStackOverlayError,
    VisionQkvStackOverlayErrorCode, VisionQkvStackSelection, bind_vision_qkv_physical_execution,
    build_verified_vision_qkv_stack_overlay, canonical_synthetic_vision_qkv_tensor_catalog,
    prepare_vision_qkv_stack_execution, select_vision_qkv_stack_overlay,
};
use pvlc_runtime_core::{
    ComputeDispatchLimits, DecoderWeightStorage, KernelId, LinearWeightLayout,
    VisionEncoderLayerGeometry, VisionEncoderLayerPlan, VisionEncoderPrecision,
    VisionQkvCanaryKind, VisionQkvExecutionPolicy, VisionQkvFusedTargetLimits,
    VisionQkvReadbackRequirements, VisionQkvSelectionOutcome, VisionStackActivationLayout,
    VisionStackActivationStrategy, plan_vision_qkv_readback_layout,
};
use serde::Serialize;

#[cfg(target_arch = "wasm32")]
mod web;

mod vision_stack_causal;

#[cfg(target_arch = "wasm32")]
pub use web::{WebRuntime, WebVisionQkvStackSelection};

#[cfg(test)]
mod vision_stack_causal_tests;

pub const VISION_STACK_SCRATCH_POISON_U32: u32 = 0x7fc0_a5a5;
pub const VISION_STACK_PREFIX_CANARY_U32: u32 = 0x51c0_ffee;
pub const VISION_STACK_SUFFIX_CANARY_U32: u32 = 0xa11a_5eed;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BrowserVisionStackWeightPlanErrorCode {
    InvalidManifest,
    InvalidGeometry,
    MissingShaderF16,
    UnsupportedFusedQkv,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BrowserVisionStackWeightPlanError {
    code: BrowserVisionStackWeightPlanErrorCode,
    message: String,
}

impl BrowserVisionStackWeightPlanError {
    fn new(code: BrowserVisionStackWeightPlanErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    #[must_use]
    pub const fn code(&self) -> BrowserVisionStackWeightPlanErrorCode {
        self.code
    }
}

impl fmt::Display for BrowserVisionStackWeightPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "browser vision-stack weight plan {:?}: {}",
            self.code, self.message
        )
    }
}

impl Error for BrowserVisionStackWeightPlanError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BrowserVisionStackLayerWeightPlan {
    pub matrix_weight_storage: DecoderWeightStorage,
    pub matrix_weight_layout: LinearWeightLayout,
    pub vector_weight_storage: DecoderWeightStorage,
    pub activation_storage: DecoderWeightStorage,
    pub projection_kernel: KernelId,
    pub rope_kernel: KernelId,
    pub requires_shader_f16: bool,
    pub fused_qkv_supported: bool,
    pub tiled_fp16_qkv_kernel: Option<KernelId>,
    pub ranges: [VisionStackLayerTensorRange; 16],
}

impl BrowserVisionStackLayerWeightPlan {
    pub fn validate_capabilities(
        self,
        shader_f16_available: bool,
    ) -> Result<(), BrowserVisionStackWeightPlanError> {
        if self.requires_shader_f16 && !shader_f16_available {
            return Err(BrowserVisionStackWeightPlanError::new(
                BrowserVisionStackWeightPlanErrorCode::MissingShaderF16,
                "FP16 vision tensors require the WebGPU shader-f16 feature",
            ));
        }
        Ok(())
    }

    pub fn validate_qkv_outcome(
        self,
        outcome: VisionQkvSelectionOutcome,
    ) -> Result<(), BrowserVisionStackWeightPlanError> {
        if !self.fused_qkv_supported && outcome == VisionQkvSelectionOutcome::Fused {
            return Err(BrowserVisionStackWeightPlanError::new(
                BrowserVisionStackWeightPlanErrorCode::UnsupportedFusedQkv,
                "the available fused vision Q/K/V kernel accepts only F32 matrix weights",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrowserVisionStackExecutionPreparation {
    pub weights: BrowserVisionStackLayerWeightPlan,
    pub layer_plan: VisionEncoderLayerPlan,
}

pub fn plan_browser_vision_stack_layer_weights(
    manifest: &VisionStackShardManifest,
) -> Result<BrowserVisionStackLayerWeightPlan, BrowserVisionStackWeightPlanError> {
    manifest.plan().map_err(|error| {
        BrowserVisionStackWeightPlanError::new(
            BrowserVisionStackWeightPlanErrorCode::InvalidManifest,
            error.to_string(),
        )
    })?;
    let ranges = vision_stack_layer_weight_ranges_with_vector_storage(
        manifest.hidden_size,
        manifest.intermediate_size,
        manifest.matrix_weight_storage,
        manifest.vector_weight_storage,
    )
    .map_err(|error| {
        BrowserVisionStackWeightPlanError::new(
            BrowserVisionStackWeightPlanErrorCode::InvalidManifest,
            error.to_string(),
        )
    })?;
    let projection_kernel = match (manifest.activation_storage, manifest.matrix_weight_storage) {
        (DecoderWeightStorage::F16, _) => KernelId::LinearProjectionF16,
        (DecoderWeightStorage::F32, DecoderWeightStorage::F32) => {
            KernelId::VisionPatchProjectionF32
        }
        (DecoderWeightStorage::F32, DecoderWeightStorage::F16) => {
            KernelId::LinearProjectionF16Weights
        }
    };
    let rope_kernel = match manifest.activation_storage {
        DecoderWeightStorage::F32 => KernelId::VisionRope2dF32,
        DecoderWeightStorage::F16 => KernelId::VisionRope2dF16,
    };
    Ok(BrowserVisionStackLayerWeightPlan {
        matrix_weight_storage: manifest.matrix_weight_storage,
        matrix_weight_layout: manifest.matrix_weight_layout,
        vector_weight_storage: manifest.vector_weight_storage,
        activation_storage: manifest.activation_storage,
        projection_kernel,
        rope_kernel,
        requires_shader_f16: [
            manifest.matrix_weight_storage,
            manifest.vector_weight_storage,
            manifest.activation_storage,
        ]
        .into_iter()
        .any(DecoderWeightStorage::requires_shader_f16),
        fused_qkv_supported: manifest.matrix_weight_storage == DecoderWeightStorage::F32
            && manifest.vector_weight_storage == DecoderWeightStorage::F32
            && manifest.activation_storage == DecoderWeightStorage::F32,
        tiled_fp16_qkv_kernel: (manifest.matrix_weight_storage == DecoderWeightStorage::F16
            && manifest.matrix_weight_layout == LinearWeightLayout::InputMajor
            && manifest.activation_storage == DecoderWeightStorage::F32)
            .then_some(KernelId::VisionQkvFusedF16Weights),
        ranges,
    })
}

pub fn vision_stack_resident_weight_key(
    manifest: &VisionStackShardManifest,
) -> Result<String, BrowserVisionStackWeightPlanError> {
    #[derive(Serialize)]
    struct ResidentWeightIdentity<'a> {
        model_id: &'a str,
        model_revision: &'a str,
        compiler_model_abi: u32,
        compiler_build: &'a str,
        matrix_weight_storage: DecoderWeightStorage,
        matrix_weight_layout: LinearWeightLayout,
        vector_weight_storage: DecoderWeightStorage,
        activation_storage: DecoderWeightStorage,
        hidden_size: u32,
        attention_heads: u32,
        head_dim: u32,
        intermediate_size: u32,
        layer_norm_epsilon_bits: u32,
        layer_count: u32,
        weight_shards: &'a [pvlc_pack::VisionStackShardDescriptor],
    }

    manifest.plan().map_err(|error| {
        BrowserVisionStackWeightPlanError::new(
            BrowserVisionStackWeightPlanErrorCode::InvalidManifest,
            error.to_string(),
        )
    })?;
    let weight_shards = manifest.shards.get(1..).ok_or_else(|| {
        BrowserVisionStackWeightPlanError::new(
            BrowserVisionStackWeightPlanErrorCode::InvalidManifest,
            "vision-stack manifest has no weight shard directory",
        )
    })?;
    let identity = ResidentWeightIdentity {
        model_id: &manifest.model_id,
        model_revision: &manifest.model_revision,
        compiler_model_abi: manifest.compiler_model_abi,
        compiler_build: &manifest.compiler_build,
        matrix_weight_storage: manifest.matrix_weight_storage,
        matrix_weight_layout: manifest.matrix_weight_layout,
        vector_weight_storage: manifest.vector_weight_storage,
        activation_storage: manifest.activation_storage,
        hidden_size: manifest.hidden_size,
        attention_heads: manifest.attention_heads,
        head_dim: manifest.head_dim,
        intermediate_size: manifest.intermediate_size,
        layer_norm_epsilon_bits: manifest.layer_norm_epsilon.to_bits(),
        layer_count: manifest.layer_count,
        weight_shards,
    };
    let serialized = serde_json::to_vec(&identity).map_err(|error| {
        BrowserVisionStackWeightPlanError::new(
            BrowserVisionStackWeightPlanErrorCode::InvalidManifest,
            format!("cannot serialize resident vision-weight identity: {error}"),
        )
    })?;
    Ok(blake3::hash(&serialized).to_hex().to_string())
}

pub fn prepare_browser_vision_stack_execution(
    manifest: &VisionStackShardManifest,
    qkv_outcome: VisionQkvSelectionOutcome,
    shader_f16_available: bool,
) -> Result<BrowserVisionStackExecutionPreparation, BrowserVisionStackWeightPlanError> {
    let weights = plan_browser_vision_stack_layer_weights(manifest)?;
    weights.validate_capabilities(shader_f16_available)?;
    weights.validate_qkv_outcome(qkv_outcome)?;
    let layer_plan = VisionEncoderLayerGeometry {
        tokens: manifest.tokens,
        hidden_size: manifest.hidden_size,
        attention_heads: manifest.attention_heads,
        head_dim: manifest.head_dim,
        intermediate_size: manifest.intermediate_size,
        layer_norm_epsilon: manifest.layer_norm_epsilon,
        cu_seqlens: &manifest.cu_seqlens,
    }
    .plan_with_precision(VisionEncoderPrecision {
        matrix_weight_storage: manifest.matrix_weight_storage,
        matrix_weight_layout: manifest.matrix_weight_layout,
        vector_weight_storage: manifest.vector_weight_storage,
        activation_storage: manifest.activation_storage,
    })
    .map_err(|error| {
        BrowserVisionStackWeightPlanError::new(
            BrowserVisionStackWeightPlanErrorCode::InvalidGeometry,
            error.to_string(),
        )
    })?;
    Ok(BrowserVisionStackExecutionPreparation {
        weights,
        layer_plan,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionStackMemoryHardening {
    PoisonCanary,
}

impl VisionStackMemoryHardening {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PoisonCanary => "poison_canary",
        }
    }
}

impl FromStr for VisionStackMemoryHardening {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "poison_canary" => Ok(Self::PoisonCanary),
            _ => Err(format!(
                "unknown vision-stack memory hardening mode {value:?}; expected poison_canary"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionStackScratchBinding {
    logical_offset: u64,
    physical_offset: u64,
    bytes: u64,
}

impl VisionStackScratchBinding {
    #[must_use]
    pub const fn logical_offset(self) -> u64 {
        self.logical_offset
    }

    #[must_use]
    pub const fn physical_offset(self) -> u64 {
        self.physical_offset
    }

    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionStackMemoryHardeningPlan {
    mode: VisionStackMemoryHardening,
    storage_alignment: u64,
    guard_bytes: u64,
    logical_scratch_bytes: u64,
    scratch_logical_offset: u64,
    scratch_suffix_offset: u64,
    physical_scratch_bytes: u64,
    semantic_checkpoint_bytes: u64,
    readback_prefix_canary_offset: u64,
    readback_suffix_canary_offset: u64,
    physical_readback_bytes: u64,
    logical_peak_gpu_data_bytes: u64,
    physical_peak_gpu_data_bytes: u64,
}

impl VisionStackMemoryHardeningPlan {
    pub fn new(
        mode: VisionStackMemoryHardening,
        storage_alignment: u64,
        logical_scratch_bytes: u64,
        semantic_checkpoint_bytes: u64,
        logical_peak_gpu_data_bytes: u64,
    ) -> Result<Self, String> {
        if storage_alignment < 4 || !storage_alignment.is_power_of_two() {
            return Err(format!(
                "vision-stack storage alignment {storage_alignment} must be a power of two at least four"
            ));
        }
        for (label, bytes) in [
            ("logical scratch", logical_scratch_bytes),
            ("semantic checkpoint", semantic_checkpoint_bytes),
            ("logical peak GPU data", logical_peak_gpu_data_bytes),
        ] {
            if bytes == 0 || !bytes.is_multiple_of(4) {
                return Err(format!(
                    "vision-stack {label} bytes {bytes} must be a nonzero multiple of four"
                ));
            }
        }
        if logical_peak_gpu_data_bytes < logical_scratch_bytes {
            return Err(format!(
                "vision-stack logical peak GPU data bytes {logical_peak_gpu_data_bytes} are smaller than scratch bytes {logical_scratch_bytes}"
            ));
        }

        let guard_bytes = storage_alignment;
        let scratch_logical_offset = guard_bytes;
        let scratch_suffix_offset = scratch_logical_offset
            .checked_add(logical_scratch_bytes)
            .ok_or_else(|| "vision-stack scratch suffix offset overflowed".to_owned())?;
        let physical_scratch_bytes = scratch_suffix_offset
            .checked_add(guard_bytes)
            .ok_or_else(|| "vision-stack physical scratch bytes overflowed".to_owned())?;
        let readback_prefix_canary_offset = semantic_checkpoint_bytes;
        let readback_suffix_canary_offset = readback_prefix_canary_offset
            .checked_add(guard_bytes)
            .ok_or_else(|| "vision-stack readback suffix offset overflowed".to_owned())?;
        let physical_readback_bytes = readback_suffix_canary_offset
            .checked_add(guard_bytes)
            .ok_or_else(|| "vision-stack physical readback bytes overflowed".to_owned())?;
        let hardening_overhead = guard_bytes
            .checked_mul(4)
            .ok_or_else(|| "vision-stack hardening overhead overflowed".to_owned())?;
        let physical_peak_gpu_data_bytes = logical_peak_gpu_data_bytes
            .checked_add(hardening_overhead)
            .ok_or_else(|| "vision-stack physical peak GPU data bytes overflowed".to_owned())?;

        Ok(Self {
            mode,
            storage_alignment,
            guard_bytes,
            logical_scratch_bytes,
            scratch_logical_offset,
            scratch_suffix_offset,
            physical_scratch_bytes,
            semantic_checkpoint_bytes,
            readback_prefix_canary_offset,
            readback_suffix_canary_offset,
            physical_readback_bytes,
            logical_peak_gpu_data_bytes,
            physical_peak_gpu_data_bytes,
        })
    }

    #[must_use]
    pub const fn mode(&self) -> VisionStackMemoryHardening {
        self.mode
    }

    #[must_use]
    pub const fn storage_alignment(&self) -> u64 {
        self.storage_alignment
    }

    #[must_use]
    pub const fn guard_bytes(&self) -> u64 {
        self.guard_bytes
    }

    #[must_use]
    pub const fn logical_scratch_bytes(&self) -> u64 {
        self.logical_scratch_bytes
    }

    #[must_use]
    pub const fn scratch_logical_offset(&self) -> u64 {
        self.scratch_logical_offset
    }

    #[must_use]
    pub const fn scratch_suffix_offset(&self) -> u64 {
        self.scratch_suffix_offset
    }

    #[must_use]
    pub const fn physical_scratch_bytes(&self) -> u64 {
        self.physical_scratch_bytes
    }

    #[must_use]
    pub const fn semantic_checkpoint_bytes(&self) -> u64 {
        self.semantic_checkpoint_bytes
    }

    #[must_use]
    pub const fn readback_prefix_canary_offset(&self) -> u64 {
        self.readback_prefix_canary_offset
    }

    #[must_use]
    pub const fn readback_suffix_canary_offset(&self) -> u64 {
        self.readback_suffix_canary_offset
    }

    #[must_use]
    pub const fn physical_readback_bytes(&self) -> u64 {
        self.physical_readback_bytes
    }

    #[must_use]
    pub const fn logical_peak_gpu_data_bytes(&self) -> u64 {
        self.logical_peak_gpu_data_bytes
    }

    #[must_use]
    pub const fn physical_peak_gpu_data_bytes(&self) -> u64 {
        self.physical_peak_gpu_data_bytes
    }

    pub fn shift_scratch_binding(
        &self,
        logical_offset: u64,
        bytes: u64,
    ) -> Result<VisionStackScratchBinding, String> {
        if bytes == 0 || !bytes.is_multiple_of(4) {
            return Err(format!(
                "vision-stack scratch binding bytes {bytes} must be a nonzero multiple of four"
            ));
        }
        if !logical_offset.is_multiple_of(self.storage_alignment) {
            return Err(format!(
                "vision-stack logical scratch offset {logical_offset} is not aligned to {}",
                self.storage_alignment
            ));
        }
        let logical_end = logical_offset
            .checked_add(bytes)
            .ok_or_else(|| "vision-stack logical scratch binding end overflowed".to_owned())?;
        if logical_end > self.logical_scratch_bytes {
            return Err(format!(
                "vision-stack logical scratch binding {logical_offset}..{logical_end} exceeds {}",
                self.logical_scratch_bytes
            ));
        }
        let physical_offset = self
            .scratch_logical_offset
            .checked_add(logical_offset)
            .ok_or_else(|| "vision-stack physical scratch binding offset overflowed".to_owned())?;
        Ok(VisionStackScratchBinding {
            logical_offset,
            physical_offset,
            bytes,
        })
    }

    pub fn verify_and_split_readback<'a>(&self, mapped: &'a [u8]) -> Result<&'a [u8], String> {
        let physical_len = usize::try_from(self.physical_readback_bytes)
            .map_err(|_| "vision-stack physical readback length is too large".to_owned())?;
        if mapped.len() != physical_len {
            return Err(format!(
                "vision-stack mapped readback length {} differs from expected {physical_len}",
                mapped.len()
            ));
        }
        let semantic_end = usize::try_from(self.semantic_checkpoint_bytes)
            .map_err(|_| "vision-stack semantic readback length is too large".to_owned())?;
        let prefix_end = usize::try_from(self.readback_suffix_canary_offset)
            .map_err(|_| "vision-stack prefix guard end is too large".to_owned())?;
        verify_u32_pattern(
            &mapped[semantic_end..prefix_end],
            VISION_STACK_PREFIX_CANARY_U32,
            "prefix",
        )?;
        verify_u32_pattern(
            &mapped[prefix_end..physical_len],
            VISION_STACK_SUFFIX_CANARY_U32,
            "suffix",
        )?;
        Ok(&mapped[..semantic_end])
    }
}

fn verify_u32_pattern(bytes: &[u8], expected: u32, label: &str) -> Result<(), String> {
    for (index, word) in bytes.chunks_exact(4).enumerate() {
        let observed = u32::from_le_bytes(word.try_into().expect("u32 chunk has four bytes"));
        if observed != expected {
            return Err(format!(
                "vision-stack {label} canary word {index} is {observed:#010x}, expected {expected:#010x}"
            ));
        }
    }
    Ok(())
}

/// Owns a session while it moves between synchronous code and one async operation.
///
/// The payload always has exactly one owner: either this value while it is stored, or
/// the caller between [`Self::acquire`] and [`Self::complete`]. All transitions are
/// available through shared references so a sealed authority can own this value
/// directly next to its execution handles.
pub struct AsyncSessionOwner<T> {
    inner: Rc<AsyncSessionOwnerInner<T>>,
}

struct AsyncSessionOwnerInner<T> {
    stored: RefCell<Option<T>>,
    generation: Cell<Option<u64>>,
    in_flight_lease: Cell<Option<u64>>,
    cancellation_requested: Cell<bool>,
    last_generation: Cell<u64>,
    last_lease: Cell<u64>,
}

impl<T> Clone for AsyncSessionOwner<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
        }
    }
}

impl<T> AsyncSessionOwner<T> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Rc::new(AsyncSessionOwnerInner {
                stored: RefCell::new(None),
                generation: Cell::new(None),
                in_flight_lease: Cell::new(None),
                cancellation_requested: Cell::new(false),
                last_generation: Cell::new(0),
                last_lease: Cell::new(0),
            }),
        }
    }

    /// Begins a new generation and stores its payload.
    ///
    /// A rejected payload is consumed and dropped by this call.
    pub fn begin(&self, session: T) -> Result<u64, SessionOwnerError> {
        if self.is_busy() {
            return Err(SessionOwnerError::Busy);
        }
        let generation = self
            .inner
            .last_generation
            .get()
            .checked_add(1)
            .ok_or(SessionOwnerError::GenerationOverflow)?;
        self.inner.last_generation.set(generation);
        self.inner.generation.set(Some(generation));
        self.inner.stored.replace(Some(session));
        Ok(generation)
    }

    #[must_use]
    pub fn is_busy(&self) -> bool {
        self.inner.generation.get().is_some()
    }

    #[must_use]
    pub fn is_in_flight(&self) -> bool {
        self.inner.in_flight_lease.get().is_some()
    }

    #[must_use]
    pub fn generation(&self) -> Option<u64> {
        self.inner.generation.get()
    }

    #[must_use]
    pub fn stored(&self) -> Option<Ref<'_, T>> {
        Ref::filter_map(self.inner.stored.borrow(), Option::as_ref).ok()
    }

    #[must_use]
    pub fn stored_mut(&self) -> Option<RefMut<'_, T>> {
        RefMut::filter_map(self.inner.stored.borrow_mut(), Option::as_mut).ok()
    }

    /// Moves the stored payload to an async caller and returns its completion lease.
    pub fn acquire(&self) -> Result<(AsyncSessionLease, T), SessionOwnerError> {
        let lease_id = self
            .inner
            .last_lease
            .get()
            .checked_add(1)
            .ok_or(SessionOwnerError::LeaseOverflow)?;
        let generation = self
            .inner
            .generation
            .get()
            .ok_or(SessionOwnerError::NoStoredSession)?;
        let session = self
            .inner
            .stored
            .take()
            .ok_or(SessionOwnerError::NoStoredSession)?;
        self.inner.last_lease.set(lease_id);
        self.inner.in_flight_lease.set(Some(lease_id));
        self.inner.cancellation_requested.set(false);
        Ok((
            AsyncSessionLease {
                generation,
                lease_id,
            },
            session,
        ))
    }

    /// Cancels the current generation without ever taking ownership away from an
    /// already-running async caller.
    pub fn abort(&self) -> AbortDisposition {
        if self.is_in_flight() {
            self.inner.cancellation_requested.set(true);
            return AbortDisposition::Deferred;
        }
        if self.inner.stored.take().is_some() {
            self.inner.generation.set(None);
            self.inner.cancellation_requested.set(false);
            return AbortDisposition::Released;
        }
        AbortDisposition::AlreadyIdle
    }

    /// Cancels the current generation and releases it for a new one immediately.
    ///
    /// Unlike [`Self::abort`], an in-flight caller keeps its payload and detects the
    /// cleared generation as staleness: its completion can only drop that payload and
    /// cannot change a stored or in-flight newer operation.
    pub fn cancel_and_release(&self) -> AbortDisposition {
        if self.is_in_flight() {
            self.inner.cancellation_requested.set(true);
            self.inner.generation.set(None);
            self.inner.in_flight_lease.set(None);
            return AbortDisposition::Deferred;
        }
        if self.inner.stored.take().is_some() {
            self.inner.generation.set(None);
            self.inner.cancellation_requested.set(false);
            return AbortDisposition::Released;
        }
        AbortDisposition::AlreadyIdle
    }

    /// Returns an async payload to the owner or finishes its generation.
    ///
    /// A stale lease can only drop the payload supplied with that stale completion;
    /// it cannot change a stored or in-flight newer operation.
    pub fn complete(
        &self,
        lease: AsyncSessionLease,
        session: T,
        action: CompletionAction,
    ) -> CompletionOutcome {
        if self.inner.generation.get() != Some(lease.generation)
            || self.inner.in_flight_lease.get() != Some(lease.lease_id)
        {
            return CompletionOutcome::Stale;
        }

        self.inner.in_flight_lease.set(None);
        if self.inner.cancellation_requested.get() {
            self.inner.cancellation_requested.set(false);
            self.inner.generation.set(None);
            return CompletionOutcome::Cancelled;
        }

        match action {
            CompletionAction::Restore => {
                self.inner.stored.replace(Some(session));
                CompletionOutcome::Restored
            }
            CompletionAction::Finish => {
                self.inner.generation.set(None);
                CompletionOutcome::Finished
            }
        }
    }
}

impl<T> Default for AsyncSessionOwner<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsyncSessionLease {
    generation: u64,
    lease_id: u64,
}

impl AsyncSessionLease {
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionOwnerError {
    Busy,
    NoStoredSession,
    GenerationOverflow,
    LeaseOverflow,
}

pub type AsyncSessionOwnerError = SessionOwnerError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbortDisposition {
    Released,
    Deferred,
    AlreadyIdle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionAction {
    Restore,
    Finish,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompletionOutcome {
    Restored,
    Finished,
    Cancelled,
    Stale,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionQkvCompilerCapabilities {
    pub min_storage_buffer_offset_alignment: u32,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub max_compute_workgroup_size: [u32; 3],
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_host_elements: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VisionQkvCompilerReadbackRequest {
    pub semantic_readback_bytes: u64,
    pub scratch_canary_readback_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct VisionQkvCompilerManifestGeometry {
    tokens: u32,
    hidden_size: u32,
    attention_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    layer_count: u32,
}

impl VisionQkvCompilerManifestGeometry {
    #[must_use]
    pub const fn tokens(&self) -> u32 {
        self.tokens
    }

    #[must_use]
    pub const fn hidden_size(&self) -> u32 {
        self.hidden_size
    }

    #[must_use]
    pub const fn attention_heads(&self) -> u32 {
        self.attention_heads
    }

    #[must_use]
    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }

    #[must_use]
    pub const fn intermediate_size(&self) -> u32 {
        self.intermediate_size
    }

    #[must_use]
    pub const fn layer_count(&self) -> u32 {
        self.layer_count
    }
}

#[derive(Debug)]
pub struct VisionQkvCompilerHandoff {
    selection: VisionQkvStackSelection,
    canonical_manifest_blake3_hex: String,
    semantic_graph_blake3_hex: String,
    manifest_geometry: VisionQkvCompilerManifestGeometry,
    layer_count: usize,
    target_limits: VisionQkvFusedTargetLimits,
    tensor_catalog_len: usize,
}

impl VisionQkvCompilerHandoff {
    #[must_use]
    pub const fn selection(&self) -> &VisionQkvStackSelection {
        &self.selection
    }

    #[must_use]
    pub fn canonical_manifest_blake3_hex(&self) -> &str {
        &self.canonical_manifest_blake3_hex
    }

    #[must_use]
    pub fn semantic_graph_blake3_hex(&self) -> &str {
        &self.semantic_graph_blake3_hex
    }

    #[must_use]
    pub const fn manifest_geometry(&self) -> &VisionQkvCompilerManifestGeometry {
        &self.manifest_geometry
    }

    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.layer_count
    }

    #[must_use]
    pub const fn target_limits(&self) -> VisionQkvFusedTargetLimits {
        self.target_limits
    }

    #[must_use]
    pub const fn tensor_catalog_len(&self) -> usize {
        self.tensor_catalog_len
    }
}

pub(crate) struct ValidatedVisionQkvStackHandoffBinding {
    manifest: VisionStackShardManifest,
    layer_count: usize,
    target: VisionQkvFusedTargetLimits,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionQkvCompilerHandoffErrorCode {
    NonCanonicalManifest,
    InvalidShardDirectory,
    ModelIdentityMismatch,
    ManifestDepthBinding,
    ManifestGeometryBinding,
    ManifestDigestBinding,
    TargetAlignment,
    TargetStorageBindings,
    TargetBindingSize,
    TargetBufferSize,
    ComputeWorkgroupX,
    ComputeWorkgroupY,
    ComputeWorkgroupZ,
    ComputeInvocations,
    ComputeDispatch,
    HostElements,
    InvalidManifest,
    InvalidGeometry,
    CompilerInvariant,
    NoFusedExecution,
}

#[derive(Debug)]
pub struct VisionQkvCompilerHandoffError {
    code: VisionQkvCompilerHandoffErrorCode,
    message: String,
}

impl VisionQkvCompilerHandoffError {
    #[must_use]
    pub const fn code(&self) -> VisionQkvCompilerHandoffErrorCode {
        self.code
    }

    fn new(code: VisionQkvCompilerHandoffErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for VisionQkvCompilerHandoffError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "vision Q/K/V compiler handoff {:?}: {}",
            self.code, self.message
        )
    }
}

impl Error for VisionQkvCompilerHandoffError {}

pub fn compile_vision_qkv_stack_handoff(
    manifest_bytes: &[u8],
    policy: VisionQkvExecutionPolicy,
    capabilities: VisionQkvCompilerCapabilities,
) -> Result<VisionQkvCompilerHandoff, VisionQkvCompilerHandoffError> {
    let manifest = parse_vision_stack_shard_manifest(manifest_bytes).map_err(map_manifest_error)?;
    manifest.plan().map_err(map_manifest_error)?;
    let layer_count = usize::try_from(manifest.layer_count).map_err(|_| {
        handoff_error(
            VisionQkvCompilerHandoffErrorCode::InvalidGeometry,
            "manifest layer count does not fit usize",
        )
    })?;
    let target_limits = target_limits_from_capabilities(capabilities);
    let manifest_geometry = VisionQkvCompilerManifestGeometry {
        tokens: manifest.tokens,
        hidden_size: manifest.hidden_size,
        attention_heads: manifest.attention_heads,
        head_dim: manifest.head_dim,
        intermediate_size: manifest.intermediate_size,
        layer_count: manifest.layer_count,
    };
    let geometry = if policy == VisionQkvExecutionPolicy::Disabled {
        None
    } else {
        Some(plan_manifest_layer_geometry(&manifest)?)
    };
    let mut semantic_graph_blake3_hex = String::new();
    let (mut tensor_catalog,) = (Vec::new(),);
    let tensor_catalog_slot = &mut tensor_catalog;
    let selection = select_vision_qkv_stack_overlay(policy, || {
        let semantic_graph = SemanticGraph::paddleocr_vl_16();
        semantic_graph_blake3_hex = blake3::hash(
            &semantic_graph
                .canonical_bytes()
                .expect("the fixed PaddleOCR-VL semantic graph must serialize"),
        )
        .to_hex()
        .to_string();
        let tensor_catalog = match manifest.oracle {
            VisionStackShardOracle::Synthetic => {
                canonical_synthetic_vision_qkv_tensor_catalog(layer_count, manifest.hidden_size)
                    .expect("validated synthetic manifest geometry must fit the model envelope")
            }
            VisionStackShardOracle::OfficialMpsBf16 => PaddleOcrVl16Schema::tensor_specs(),
        };
        let overlay = build_verified_vision_qkv_stack_overlay(
            &semantic_graph,
            layer_count,
            geometry
                .as_ref()
                .expect("Disabled selection must not invoke its geometry closure"),
            &tensor_catalog,
            target_limits,
        );
        *tensor_catalog_slot = tensor_catalog;
        overlay
    })
    .map_err(map_overlay_error)?;
    Ok(VisionQkvCompilerHandoff {
        selection,
        canonical_manifest_blake3_hex: blake3::hash(manifest_bytes).to_hex().to_string(),
        semantic_graph_blake3_hex,
        manifest_geometry,
        layer_count,
        target_limits,
        tensor_catalog_len: tensor_catalog.len(),
    })
}

pub(crate) fn validate_vision_qkv_stack_handoff_binding(
    handoff: &VisionQkvCompilerHandoff,
    manifest_bytes: &[u8],
    capabilities: VisionQkvCompilerCapabilities,
) -> Result<ValidatedVisionQkvStackHandoffBinding, VisionQkvCompilerHandoffError> {
    let manifest = parse_vision_stack_shard_manifest(manifest_bytes).map_err(map_manifest_error)?;
    manifest.plan().map_err(map_manifest_error)?;
    let layer_count = usize::try_from(manifest.layer_count).map_err(|_| {
        handoff_error(
            VisionQkvCompilerHandoffErrorCode::ManifestDepthBinding,
            "manifest layer count does not fit usize",
        )
    })?;
    if layer_count != handoff.layer_count() {
        return Err(handoff_error(
            VisionQkvCompilerHandoffErrorCode::ManifestDepthBinding,
            "manifest depth differs from the compiler handoff",
        ));
    }
    if !manifest_geometry_matches(&manifest, handoff.manifest_geometry()) {
        return Err(handoff_error(
            VisionQkvCompilerHandoffErrorCode::ManifestGeometryBinding,
            "manifest geometry differs from the compiler handoff",
        ));
    }
    if blake3::hash(manifest_bytes).to_hex().as_str() != handoff.canonical_manifest_blake3_hex() {
        return Err(handoff_error(
            VisionQkvCompilerHandoffErrorCode::ManifestDigestBinding,
            "manifest digest differs from the compiler handoff",
        ));
    }

    let target = target_limits_from_capabilities(capabilities);
    match handoff.selection().outcome() {
        VisionQkvSelectionOutcome::Disabled => {}
        VisionQkvSelectionOutcome::FallbackUnsupportedTarget => {}
        VisionQkvSelectionOutcome::Fused => {}
    }
    let compiled_target = handoff.target_limits();
    if target.min_storage_buffer_offset_alignment
        != compiled_target.min_storage_buffer_offset_alignment
    {
        return Err(handoff_error(
            VisionQkvCompilerHandoffErrorCode::TargetAlignment,
            "runtime storage alignment differs from the compiler handoff",
        ));
    }
    if target.max_storage_buffers_per_shader_stage
        < compiled_target.max_storage_buffers_per_shader_stage
    {
        return Err(handoff_error(
            VisionQkvCompilerHandoffErrorCode::TargetStorageBindings,
            "runtime storage binding count is below the compiler handoff",
        ));
    }
    if target.max_storage_buffer_binding_size < compiled_target.max_storage_buffer_binding_size {
        return Err(handoff_error(
            VisionQkvCompilerHandoffErrorCode::TargetBindingSize,
            "runtime storage binding size is below the compiler handoff",
        ));
    }
    if target.max_buffer_size < compiled_target.max_buffer_size {
        return Err(handoff_error(
            VisionQkvCompilerHandoffErrorCode::TargetBufferSize,
            "runtime buffer size is below the compiler handoff",
        ));
    }
    if target.max_compute_workgroups_per_dimension
        < compiled_target.max_compute_workgroups_per_dimension
    {
        return Err(handoff_error(
            VisionQkvCompilerHandoffErrorCode::ComputeDispatch,
            "runtime dispatch dimension is below the compiler handoff",
        ));
    }

    Ok(ValidatedVisionQkvStackHandoffBinding {
        manifest,
        layer_count,
        target,
    })
}

pub fn prepare_vision_qkv_stack_handoff_execution(
    handoff: &VisionQkvCompilerHandoff,
    manifest_bytes: &[u8],
    capabilities: VisionQkvCompilerCapabilities,
    readback: VisionQkvCompilerReadbackRequest,
) -> Result<VisionQkvPhysicalExecutionSpec, VisionQkvCompilerHandoffError> {
    let ValidatedVisionQkvStackHandoffBinding {
        manifest,
        layer_count,
        target,
    } = validate_vision_qkv_stack_handoff_binding(handoff, manifest_bytes, capabilities)?;
    let overlay = handoff.selection().overlay().ok_or_else(|| {
        handoff_error(
            VisionQkvCompilerHandoffErrorCode::NoFusedExecution,
            "compiler handoff does not contain a fused Q/K/V overlay",
        )
    })?;
    let geometry = plan_manifest_layer_geometry(&manifest)?;
    let prepared_execution =
        prepare_vision_qkv_stack_execution(overlay, layer_count, &geometry, target)
            .map_err(map_prepared_error)?;
    let executor_invocation = prepared_execution.layers()[0].invocation();
    validate_compute_axes(&executor_invocation, capabilities)?;
    let compute_limits = ComputeDispatchLimits {
        max_workgroup_size: capabilities.max_compute_workgroup_size,
        max_invocations_per_workgroup: capabilities.max_compute_invocations_per_workgroup,
        max_workgroups_per_dimension: capabilities.max_compute_workgroups_per_dimension,
    };
    compute_limits
        .validate(&executor_invocation)
        .map_err(|error| {
            handoff_error(
                VisionQkvCompilerHandoffErrorCode::ComputeInvocations,
                error.to_string(),
            )
        })?;
    let workspace = prepared_execution.workspace();
    let readback_layout = plan_vision_qkv_readback_layout(VisionQkvReadbackRequirements {
        semantic_readback_bytes: readback.semantic_readback_bytes,
        scratch_canary_readback_bytes: readback.scratch_canary_readback_bytes,
        qkv_canary_readback_bytes: workspace.canary_readback_bytes(),
        workspace_allocation_bytes: workspace.allocation_bytes(),
        max_buffer_size: capabilities.max_buffer_size,
        max_host_elements: capabilities.max_host_elements,
    })
    .map_err(map_readback_error)?;
    let physical_execution =
        bind_vision_qkv_physical_execution(prepared_execution, readback_layout).map_err(
            |error| {
                handoff_error(
                    VisionQkvCompilerHandoffErrorCode::CompilerInvariant,
                    error.to_string(),
                )
            },
        )?;
    Ok(physical_execution)
}

fn manifest_geometry_matches(
    manifest: &VisionStackShardManifest,
    geometry: &VisionQkvCompilerManifestGeometry,
) -> bool {
    manifest.tokens == geometry.tokens()
        && manifest.hidden_size == geometry.hidden_size()
        && manifest.attention_heads == geometry.attention_heads()
        && manifest.head_dim == geometry.head_dim()
        && manifest.intermediate_size == geometry.intermediate_size()
        && manifest.layer_count == geometry.layer_count()
}

fn plan_manifest_layer_geometry(
    manifest: &VisionStackShardManifest,
) -> Result<pvlc_runtime_core::VisionEncoderLayerPlan, VisionQkvCompilerHandoffError> {
    VisionEncoderLayerGeometry {
        tokens: manifest.tokens,
        hidden_size: manifest.hidden_size,
        attention_heads: manifest.attention_heads,
        head_dim: manifest.head_dim,
        intermediate_size: manifest.intermediate_size,
        layer_norm_epsilon: manifest.layer_norm_epsilon,
        cu_seqlens: &manifest.cu_seqlens,
    }
    .plan()
    .map_err(|error| {
        handoff_error(
            VisionQkvCompilerHandoffErrorCode::InvalidGeometry,
            error.to_string(),
        )
    })
}

const fn target_limits_from_capabilities(
    capabilities: VisionQkvCompilerCapabilities,
) -> VisionQkvFusedTargetLimits {
    VisionQkvFusedTargetLimits {
        min_storage_buffer_offset_alignment: capabilities.min_storage_buffer_offset_alignment,
        max_storage_buffers_per_shader_stage: capabilities.max_storage_buffers_per_shader_stage,
        max_storage_buffer_binding_size: capabilities.max_storage_buffer_binding_size,
        max_buffer_size: capabilities.max_buffer_size,
        max_compute_workgroups_per_dimension: capabilities.max_compute_workgroups_per_dimension,
    }
}

fn validate_compute_axes(
    invocation: &pvlc_runtime_core::InvocationPlan,
    capabilities: VisionQkvCompilerCapabilities,
) -> Result<(), VisionQkvCompilerHandoffError> {
    let codes = [
        VisionQkvCompilerHandoffErrorCode::ComputeWorkgroupX,
        VisionQkvCompilerHandoffErrorCode::ComputeWorkgroupY,
        VisionQkvCompilerHandoffErrorCode::ComputeWorkgroupZ,
    ];
    for (axis, code) in codes.into_iter().enumerate() {
        if invocation.workgroup_size[axis] == 0
            || invocation.workgroup_size[axis] > capabilities.max_compute_workgroup_size[axis]
        {
            return Err(handoff_error(
                code,
                format!("compute workgroup axis {axis} exceeds the supplied capability"),
            ));
        }
        if invocation.dispatch[axis] == 0
            || invocation.dispatch[axis] > capabilities.max_compute_workgroups_per_dimension
        {
            return Err(handoff_error(
                VisionQkvCompilerHandoffErrorCode::ComputeDispatch,
                format!("compute dispatch axis {axis} exceeds the supplied capability"),
            ));
        }
    }
    Ok(())
}

fn map_manifest_error(error: VisionStackShardError) -> VisionQkvCompilerHandoffError {
    let code = match error.code() {
        VisionStackShardErrorCode::NonCanonicalManifest => {
            VisionQkvCompilerHandoffErrorCode::NonCanonicalManifest
        }
        VisionStackShardErrorCode::ModelIdentityMismatch
        | VisionStackShardErrorCode::OfficialIdentityMismatch => {
            VisionQkvCompilerHandoffErrorCode::ModelIdentityMismatch
        }
        VisionStackShardErrorCode::InvalidShardDirectory
        | VisionStackShardErrorCode::WrongShardOrder
        | VisionStackShardErrorCode::LengthMismatch
        | VisionStackShardErrorCode::DigestMismatch => {
            VisionQkvCompilerHandoffErrorCode::InvalidShardDirectory
        }
        VisionStackShardErrorCode::InvalidGeometry => {
            VisionQkvCompilerHandoffErrorCode::InvalidGeometry
        }
        _ => VisionQkvCompilerHandoffErrorCode::InvalidManifest,
    };
    handoff_error(code, error.to_string())
}

fn map_overlay_error(error: VisionQkvStackOverlayError) -> VisionQkvCompilerHandoffError {
    let code = match error.code() {
        VisionQkvStackOverlayErrorCode::UnsupportedTarget => {
            VisionQkvCompilerHandoffErrorCode::TargetStorageBindings
        }
        _ => VisionQkvCompilerHandoffErrorCode::CompilerInvariant,
    };
    handoff_error(code, error.to_string())
}

fn map_prepared_error(error: VisionQkvPreparedExecutionError) -> VisionQkvCompilerHandoffError {
    let code = match error.code() {
        VisionQkvPreparedExecutionErrorCode::TargetAlignment => {
            VisionQkvCompilerHandoffErrorCode::TargetAlignment
        }
        VisionQkvPreparedExecutionErrorCode::TargetStorageBindings => {
            VisionQkvCompilerHandoffErrorCode::TargetStorageBindings
        }
        VisionQkvPreparedExecutionErrorCode::TargetBindingSize => {
            VisionQkvCompilerHandoffErrorCode::TargetBindingSize
        }
        VisionQkvPreparedExecutionErrorCode::TargetBufferSize => {
            VisionQkvCompilerHandoffErrorCode::TargetBufferSize
        }
        VisionQkvPreparedExecutionErrorCode::TargetDispatchLimit => {
            VisionQkvCompilerHandoffErrorCode::ComputeDispatch
        }
        VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow => {
            VisionQkvCompilerHandoffErrorCode::HostElements
        }
        _ => VisionQkvCompilerHandoffErrorCode::CompilerInvariant,
    };
    handoff_error(code, error.to_string())
}

fn map_readback_error(error: pvlc_runtime_core::InvocationError) -> VisionQkvCompilerHandoffError {
    let message = error.to_string();
    let code = if message.contains("host") || message.contains("usize") {
        VisionQkvCompilerHandoffErrorCode::HostElements
    } else if message.contains("buffer") {
        VisionQkvCompilerHandoffErrorCode::TargetBufferSize
    } else {
        VisionQkvCompilerHandoffErrorCode::CompilerInvariant
    };
    handoff_error(code, message)
}

fn handoff_error(
    code: VisionQkvCompilerHandoffErrorCode,
    message: impl Into<String>,
) -> VisionQkvCompilerHandoffError {
    VisionQkvCompilerHandoffError::new(code, message)
}

#[derive(Debug)]
pub enum VisionStackEvidenceError {
    Json(serde_json::Error),
    IntegerConversion,
    ArithmeticOverflow,
    UnexpectedCanaryResults,
    CanaryResultCount,
    InvalidLegacyPlan(String),
}

impl fmt::Display for VisionStackEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "JSON serialization failed: {error}"),
            Self::IntegerConversion => formatter.write_str("integer conversion failed"),
            Self::ArithmeticOverflow => formatter.write_str("evidence arithmetic overflowed"),
            Self::UnexpectedCanaryResults => {
                formatter.write_str("canary results exist without a Q/K/V execution plan")
            }
            Self::CanaryResultCount => {
                formatter.write_str("canary result count differs from the execution plan")
            }
            Self::InvalidLegacyPlan(message) => formatter.write_str(message),
        }
    }
}

impl Error for VisionStackEvidenceError {}

impl From<serde_json::Error> for VisionStackEvidenceError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
struct VisionQkvTargetLimitsEvidence {
    min_storage_buffer_offset_alignment: u32,
    max_storage_buffers_per_shader_stage: u32,
    max_storage_buffer_binding_size: u64,
    max_buffer_size: u64,
    max_compute_workgroups_per_dimension: u32,
}

impl From<VisionQkvFusedTargetLimits> for VisionQkvTargetLimitsEvidence {
    fn from(limits: VisionQkvFusedTargetLimits) -> Self {
        Self {
            min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_buffer_size: limits.max_buffer_size,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }
}

#[derive(Serialize)]
pub struct VisionQkvSelectionEvidence {
    policy: &'static str,
    outcome: &'static str,
    fallback_class: Option<&'static str>,
    manifest_blake3: String,
    #[serde(serialize_with = "serialize_optional_blake3")]
    semantic_graph_blake3: String,
    manifest_geometry: VisionQkvCompilerManifestGeometry,
    target_limits: VisionQkvTargetLimitsEvidence,
    tensor_catalog_len: usize,
    layer_plan_blake3: Vec<String>,
}

fn serialize_optional_blake3<S>(value: &str, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if value.is_empty() {
        serializer.serialize_none()
    } else {
        serializer.serialize_str(value)
    }
}

#[derive(Serialize)]
pub struct VisionQkvEvidenceEnvelope<'a, E> {
    qkv_selection: &'a VisionQkvSelectionEvidence,
    qkv_execution: Option<&'a E>,
}

impl<'a, E> VisionQkvEvidenceEnvelope<'a, E> {
    #[must_use]
    pub const fn qkv_selection(&self) -> &VisionQkvSelectionEvidence {
        self.qkv_selection
    }

    #[must_use]
    pub const fn qkv_execution(&self) -> Option<&E> {
        self.qkv_execution
    }
}

#[derive(Clone)]
pub struct VisionQkvSelectionEvidencePropagation {
    evidence: Rc<VisionQkvSelectionEvidence>,
}

impl VisionQkvSelectionEvidencePropagation {
    #[must_use]
    pub fn opaque_selection_evidence(&self) -> &VisionQkvSelectionEvidence {
        &self.evidence
    }

    pub fn evidence_json(&self) -> Result<String, VisionStackEvidenceError> {
        to_json_string(self.evidence.as_ref())
    }

    #[must_use]
    pub fn additive_begin_evidence<'a, E>(
        &'a self,
        qkv_execution: Option<&'a E>,
    ) -> VisionQkvEvidenceEnvelope<'a, E> {
        self.evidence_envelope(qkv_execution)
    }

    #[must_use]
    pub fn final_diagnostics_evidence<'a, E>(
        &'a self,
        qkv_execution: Option<&'a E>,
    ) -> VisionQkvEvidenceEnvelope<'a, E> {
        self.evidence_envelope(qkv_execution)
    }

    #[must_use]
    pub fn uses_legacy_topology(&self) -> bool {
        self.evidence.outcome != "fused"
    }

    fn evidence_envelope<'a, E>(
        &'a self,
        qkv_execution: Option<&'a E>,
    ) -> VisionQkvEvidenceEnvelope<'a, E> {
        VisionQkvEvidenceEnvelope {
            qkv_selection: &self.evidence,
            qkv_execution,
        }
    }
}

#[must_use]
pub fn build_vision_qkv_selection_evidence_propagation(
    handoff: &VisionQkvCompilerHandoff,
) -> VisionQkvSelectionEvidencePropagation {
    let semantic_graph_blake3 = handoff.semantic_graph_blake3_hex().to_owned();
    let selection = handoff.selection();
    let policy = match selection.policy() {
        VisionQkvExecutionPolicy::Disabled => "disabled",
        VisionQkvExecutionPolicy::Preferred => "preferred",
        VisionQkvExecutionPolicy::Required => "required",
    };
    let outcome = match selection.outcome() {
        VisionQkvSelectionOutcome::Disabled => "disabled",
        VisionQkvSelectionOutcome::Fused => "fused",
        VisionQkvSelectionOutcome::FallbackUnsupportedTarget => "fallback_unsupported_target",
    };
    let fallback_class = selection
        .fallback_error_code()
        .map(|_| "unsupported_target");
    let layer_plan_blake3 = selection
        .overlay()
        .map(|overlay| {
            overlay
                .layers()
                .iter()
                .map(|layer| layer.canonical_plan_blake3_hex().to_owned())
                .collect()
        })
        .unwrap_or_default();
    let evidence = VisionQkvSelectionEvidence {
        policy,
        outcome,
        fallback_class,
        manifest_blake3: handoff.canonical_manifest_blake3_hex().to_owned(),
        semantic_graph_blake3,
        manifest_geometry: *handoff.manifest_geometry(),
        target_limits: handoff.target_limits().into(),
        tensor_catalog_len: handoff.tensor_catalog_len(),
        layer_plan_blake3,
    };
    VisionQkvSelectionEvidencePropagation {
        evidence: Rc::new(evidence),
    }
}

#[derive(Clone, Serialize)]
struct BrowserVisionQkvExecutionWorkspaceEvidence {
    logical_id: &'static str,
    allocation_bytes: u64,
    semantic_base: u64,
    semantic_bytes: u64,
}

#[derive(Clone, Serialize)]
struct BrowserVisionQkvExecutionBindingEvidence {
    binding: u32,
    byte_offset: u64,
    byte_length: u64,
}

#[derive(Clone)]
struct BrowserVisionQkvExecutionCanaryPlanEvidence {
    kind: &'static str,
    plane: Option<u32>,
    byte_offset: u64,
    byte_length: u64,
}

#[derive(Clone)]
pub struct BrowserVisionQkvExecutionEvidencePlan {
    dispatch_count: u32,
    command_buffer_count: u32,
    submission_count: u32,
    map_count: u32,
    workspace: BrowserVisionQkvExecutionWorkspaceEvidence,
    bindings: Vec<BrowserVisionQkvExecutionBindingEvidence>,
    canaries: Vec<BrowserVisionQkvExecutionCanaryPlanEvidence>,
}

#[derive(Serialize)]
struct BrowserVisionQkvExecutionCanaryEvidence<'a> {
    kind: &'a str,
    plane: Option<u32>,
    byte_offset: u64,
    byte_length: u64,
    passed: Option<bool>,
}

#[derive(Serialize)]
struct BrowserVisionQkvExecutionEvidence<'a> {
    dispatch_count: u32,
    command_buffer_count: u32,
    submission_count: u32,
    map_count: u32,
    workspace: &'a BrowserVisionQkvExecutionWorkspaceEvidence,
    bindings: &'a [BrowserVisionQkvExecutionBindingEvidence],
    canaries: Vec<BrowserVisionQkvExecutionCanaryEvidence<'a>>,
}

#[derive(Serialize)]
pub struct BrowserVisionQkvBeginExecutionEvidence<'a> {
    #[serde(flatten)]
    evidence: BrowserVisionQkvExecutionEvidence<'a>,
}

#[derive(Serialize)]
pub struct BrowserVisionQkvFinalExecutionEvidence<'a> {
    #[serde(flatten)]
    evidence: BrowserVisionQkvExecutionEvidence<'a>,
}

impl BrowserVisionQkvExecutionEvidencePlan {
    pub fn from_prepared(
        prepared: Option<&VisionQkvPhysicalExecutionSpec>,
    ) -> Result<Option<Self>, VisionStackEvidenceError> {
        let Some(physical_spec) = prepared else {
            return Ok(None);
        };
        let prepared_execution = physical_spec.prepared_execution();
        let layer_count = u32::try_from(prepared_execution.layer_count())
            .map_err(|_| VisionStackEvidenceError::IntegerConversion)?;
        let dispatch_count = checked_qkv_dispatch_count(layer_count)?;
        let command_buffer_count = checked_qkv_operation_count(layer_count)?;
        let submission_count = command_buffer_count;
        let workspace = prepared_execution.workspace();
        let first_layer = prepared_execution.layers().first().ok_or_else(|| {
            VisionStackEvidenceError::InvalidLegacyPlan(
                "prepared Q/K/V execution has no layers".into(),
            )
        })?;
        let bindings = first_layer
            .attention_bridge()
            .bindings()
            .iter()
            .map(|binding| BrowserVisionQkvExecutionBindingEvidence {
                binding: binding.binding(),
                byte_offset: binding.byte_offset(),
                byte_length: binding.byte_length(),
            })
            .collect();
        let canaries = workspace
            .canaries()
            .iter()
            .map(|canary| {
                let (kind, plane) = serialize_vision_qkv_canary_kind(canary.kind());
                BrowserVisionQkvExecutionCanaryPlanEvidence {
                    kind,
                    plane,
                    byte_offset: canary.byte_offset(),
                    byte_length: canary.byte_length(),
                }
            })
            .collect();
        let workspace = BrowserVisionQkvExecutionWorkspaceEvidence {
            logical_id: "vision-stack-qkv-workspace",
            allocation_bytes: workspace.allocation_bytes(),
            semantic_base: workspace.semantic_base(),
            semantic_bytes: workspace.semantic_bytes(),
        };
        Ok(Some(BrowserVisionQkvExecutionEvidencePlan {
            dispatch_count,
            command_buffer_count,
            submission_count,
            map_count: 1,
            workspace,
            bindings,
            canaries,
        }))
    }

    fn channel_evidence(&self, passed: Vec<Option<bool>>) -> BrowserVisionQkvExecutionEvidence<'_> {
        let canaries = self
            .canaries
            .iter()
            .zip(passed)
            .map(|(canary, passed)| BrowserVisionQkvExecutionCanaryEvidence {
                kind: canary.kind,
                plane: canary.plane,
                byte_offset: canary.byte_offset,
                byte_length: canary.byte_length,
                passed,
            })
            .collect();
        BrowserVisionQkvExecutionEvidence {
            dispatch_count: self.dispatch_count,
            command_buffer_count: self.command_buffer_count,
            submission_count: self.submission_count,
            map_count: self.map_count,
            workspace: &self.workspace,
            bindings: &self.bindings,
            canaries,
        }
    }
}

impl<'a> BrowserVisionQkvBeginExecutionEvidence<'a> {
    #[must_use]
    pub fn from_plan(plan: Option<&'a BrowserVisionQkvExecutionEvidencePlan>) -> Option<Self> {
        plan.map(|plan| {
            let evidence = plan.channel_evidence(vec![None; plan.canaries.len()]);
            BrowserVisionQkvBeginExecutionEvidence { evidence }
        })
    }
}

impl<'a> BrowserVisionQkvFinalExecutionEvidence<'a> {
    pub fn from_verified_plan(
        plan: Option<&'a BrowserVisionQkvExecutionEvidencePlan>,
        canary_results: &[bool],
    ) -> Result<Option<Self>, VisionStackEvidenceError> {
        let Some(plan) = plan else {
            if !canary_results.is_empty() {
                return Err(VisionStackEvidenceError::UnexpectedCanaryResults);
            }
            return Ok(None);
        };
        if canary_results.len() != plan.canaries.len() {
            return Err(VisionStackEvidenceError::CanaryResultCount);
        }
        let passed = canary_results.iter().copied().map(Some).collect();
        let evidence = plan.channel_evidence(passed);
        Ok(Some(BrowserVisionQkvFinalExecutionEvidence { evidence }))
    }
}

fn checked_qkv_dispatch_count(layer_count: u32) -> Result<u32, VisionStackEvidenceError> {
    layer_count
        .checked_mul(10)
        .and_then(|count| count.checked_add(1))
        .ok_or(VisionStackEvidenceError::ArithmeticOverflow)
}

fn checked_qkv_operation_count(layer_count: u32) -> Result<u32, VisionStackEvidenceError> {
    layer_count
        .checked_add(1)
        .ok_or(VisionStackEvidenceError::ArithmeticOverflow)
}

fn serialize_vision_qkv_canary_kind(kind: VisionQkvCanaryKind) -> (&'static str, Option<u32>) {
    match kind {
        VisionQkvCanaryKind::Prefix => ("prefix", None),
        VisionQkvCanaryKind::InternalPadding { plane } => {
            ("internal_padding", u32::try_from(plane).ok())
        }
        VisionQkvCanaryKind::Suffix => ("suffix", None),
    }
}

#[derive(Serialize)]
struct VisionStackQkvSerializedRecord<'a, L, E> {
    #[serde(flatten)]
    legacy: &'a L,
    #[serde(flatten)]
    evidence: E,
}

fn serialize_vision_stack_qkv_record_json<L: Serialize, E: Serialize>(
    legacy: &L,
    evidence: E,
) -> Result<String, VisionStackEvidenceError> {
    to_json_string(&VisionStackQkvSerializedRecord { legacy, evidence })
}

pub fn serialize_vision_stack_qkv_begin_status_json<L: Serialize, E: Serialize>(
    legacy: &L,
    evidence: E,
) -> Result<String, VisionStackEvidenceError> {
    serialize_vision_stack_qkv_record_json(legacy, evidence)
}

pub fn serialize_vision_stack_qkv_final_diagnostics_json<L: Serialize, E: Serialize>(
    legacy: &L,
    evidence: E,
) -> Result<String, VisionStackEvidenceError> {
    serialize_vision_stack_qkv_record_json(legacy, evidence)
}

fn to_json_string<T: Serialize + ?Sized>(value: &T) -> Result<String, VisionStackEvidenceError> {
    serde_json::to_string(value).map_err(VisionStackEvidenceError::from)
}

const LEGACY_CHECKED_SCOPES: [&str; 3] = ["validation", "out_of_memory", "internal"];

#[derive(Clone, Serialize)]
struct VisionStackLegacyPlanRecord {
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
    #[serde(skip_serializing_if = "Option::is_none")]
    scratch_arena_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    main_buffers_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    activation_strategy: Option<VisionStackActivationStrategy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_storage_buffer_offset_alignment: Option<u32>,
    readback_bytes: u64,
    peak_gpu_data_bytes: u64,
    submission_count: u32,
    compute_pass_count: u32,
    dispatch_count: u32,
}

#[derive(Clone, Copy, Serialize)]
struct VisionStackLegacyCapabilitiesRecord {
    min_storage_buffer_offset_alignment: u32,
}

#[derive(Serialize)]
struct VisionStackLegacyStaticLayoutRecord {
    scratch_allocations: Vec<VisionStackLegacyScratchAllocationRecord>,
    scratch_arena_bytes: u64,
    main_buffers_bytes: u64,
    total_activation_bytes: u64,
    physical_buffer_count: usize,
}

#[derive(Serialize)]
struct VisionStackLegacyScratchAllocationRecord {
    stage: &'static str,
    offset: u64,
    size: u64,
    alignment: u64,
    first_write: u32,
    last_use: u32,
}

#[derive(Clone, Copy, Serialize)]
struct VisionStackLegacyMemoryHardeningPlanRecord {
    mode: &'static str,
    scratch_poison_u32: u32,
    prefix_canary_u32: u32,
    suffix_canary_u32: u32,
    storage_alignment: u64,
    guard_bytes: u64,
    logical_scratch_bytes: u64,
    scratch_logical_offset: u64,
    scratch_suffix_offset: u64,
    physical_scratch_bytes: u64,
    semantic_checkpoint_bytes: u64,
    readback_semantic_offset: u64,
    readback_prefix_canary_offset: u64,
    readback_suffix_canary_offset: u64,
    physical_readback_bytes: u64,
    logical_peak_gpu_data_bytes: u64,
    physical_peak_gpu_data_bytes: u64,
}

impl From<&VisionStackMemoryHardeningPlan> for VisionStackLegacyMemoryHardeningPlanRecord {
    fn from(plan: &VisionStackMemoryHardeningPlan) -> Self {
        Self {
            mode: plan.mode().as_str(),
            scratch_poison_u32: VISION_STACK_SCRATCH_POISON_U32,
            prefix_canary_u32: VISION_STACK_PREFIX_CANARY_U32,
            suffix_canary_u32: VISION_STACK_SUFFIX_CANARY_U32,
            storage_alignment: plan.storage_alignment(),
            guard_bytes: plan.guard_bytes(),
            logical_scratch_bytes: plan.logical_scratch_bytes(),
            scratch_logical_offset: plan.scratch_logical_offset(),
            scratch_suffix_offset: plan.scratch_suffix_offset(),
            physical_scratch_bytes: plan.physical_scratch_bytes(),
            semantic_checkpoint_bytes: plan.semantic_checkpoint_bytes(),
            readback_semantic_offset: 0,
            readback_prefix_canary_offset: plan.readback_prefix_canary_offset(),
            readback_suffix_canary_offset: plan.readback_suffix_canary_offset(),
            physical_readback_bytes: plan.physical_readback_bytes(),
            logical_peak_gpu_data_bytes: plan.logical_peak_gpu_data_bytes(),
            physical_peak_gpu_data_bytes: plan.physical_peak_gpu_data_bytes(),
        }
    }
}

#[derive(Clone, Copy, Serialize)]
struct VisionStackLegacyCanaryChecksRecord {
    prefix: bool,
    suffix: bool,
}

#[derive(Clone, Copy, Serialize)]
struct VisionStackLegacyMemoryHardeningDiagnosticsRecord {
    #[serde(flatten)]
    plan: VisionStackLegacyMemoryHardeningPlanRecord,
    canary_checks: VisionStackLegacyCanaryChecksRecord,
}

impl From<&VisionStackMemoryHardeningPlan> for VisionStackLegacyMemoryHardeningDiagnosticsRecord {
    fn from(plan: &VisionStackMemoryHardeningPlan) -> Self {
        Self {
            plan: VisionStackLegacyMemoryHardeningPlanRecord::from(plan),
            canary_checks: VisionStackLegacyCanaryChecksRecord {
                prefix: true,
                suffix: true,
            },
        }
    }
}

#[derive(Serialize)]
pub struct VisionStackLegacyStatusRecord<'a> {
    phase: VisionStackShardProtocolPhase,
    next_shard_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    plan: Option<VisionStackLegacyPlanRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    capabilities: Option<VisionStackLegacyCapabilitiesRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    static_layout: Option<VisionStackLegacyStaticLayoutRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_hardening_plan: Option<VisionStackLegacyMemoryHardeningPlanRecord>,
}

#[derive(Serialize)]
pub struct VisionStackLegacyDiagnosticsRecord<'a> {
    checked_error_scopes: [&'static str; 3],
    captured_errors: Vec<String>,
    queue_wall_time_ns: u64,
    shader_blake3: &'a BTreeMap<KernelId, String>,
    #[serde(flatten)]
    plan: VisionStackLegacyPlanRecord,
    command_buffer_count: u32,
    buffer_allocation_count: u64,
    weight_buffer_count: u32,
    readback_buffer_count: u32,
    map_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    memory_hardening: Option<VisionStackLegacyMemoryHardeningDiagnosticsRecord>,
}

#[expect(clippy::too_many_arguments, reason = "frozen legacy status ABI")]
pub fn build_vision_stack_legacy_status_record<'a>(
    phase: VisionStackShardProtocolPhase,
    next_shard_id: Option<&'a str>,
    plan: &VisionStackShardPlan,
    activation_strategy: VisionStackActivationStrategy,
    activation_layout: Option<&VisionStackActivationLayout>,
    memory_hardening: Option<&VisionStackMemoryHardeningPlan>,
    min_storage_buffer_offset_alignment: u32,
    include_plan: bool,
) -> Result<VisionStackLegacyStatusRecord<'a>, VisionStackEvidenceError> {
    let static_strategy = activation_strategy != VisionStackActivationStrategy::SeparateBuffers;
    validate_legacy_layout_inputs(static_strategy, activation_layout, memory_hardening)?;
    let plan_record = include_plan
        .then(|| {
            build_vision_stack_legacy_plan_record(
                plan,
                activation_strategy,
                activation_layout,
                min_storage_buffer_offset_alignment,
            )
        })
        .transpose()?;
    let capabilities =
        (include_plan && static_strategy).then_some(VisionStackLegacyCapabilitiesRecord {
            min_storage_buffer_offset_alignment,
        });
    let static_layout = if include_plan {
        activation_layout.map(build_vision_stack_legacy_static_layout_record)
    } else {
        None
    };
    let memory_hardening_plan = if include_plan {
        memory_hardening.map(VisionStackLegacyMemoryHardeningPlanRecord::from)
    } else {
        None
    };
    Ok(VisionStackLegacyStatusRecord {
        phase,
        next_shard_id,
        plan: plan_record,
        capabilities,
        static_layout,
        memory_hardening_plan,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn build_vision_stack_legacy_diagnostics_record<'a>(
    plan: &VisionStackShardPlan,
    activation_strategy: VisionStackActivationStrategy,
    activation_layout: Option<&VisionStackActivationLayout>,
    memory_hardening: Option<&VisionStackMemoryHardeningPlan>,
    min_storage_buffer_offset_alignment: u32,
    shader_blake3: &'a BTreeMap<KernelId, String>,
    queue_wall_time_ns: u64,
    buffer_allocation_count: u64,
    weight_buffer_count: u32,
) -> Result<VisionStackLegacyDiagnosticsRecord<'a>, VisionStackEvidenceError> {
    let static_strategy = activation_strategy != VisionStackActivationStrategy::SeparateBuffers;
    validate_legacy_layout_inputs(static_strategy, activation_layout, memory_hardening)?;
    let plan_record = build_vision_stack_legacy_plan_record(
        plan,
        activation_strategy,
        activation_layout,
        min_storage_buffer_offset_alignment,
    )?;
    Ok(VisionStackLegacyDiagnosticsRecord {
        checked_error_scopes: LEGACY_CHECKED_SCOPES,
        captured_errors: Vec::with_capacity(0),
        queue_wall_time_ns: queue_wall_time_ns.max(1),
        shader_blake3,
        plan: plan_record,
        command_buffer_count: plan.submission_count,
        buffer_allocation_count,
        weight_buffer_count,
        readback_buffer_count: 1,
        map_count: 1,
        memory_hardening: memory_hardening
            .map(VisionStackLegacyMemoryHardeningDiagnosticsRecord::from),
    })
}

pub fn serialize_vision_stack_legacy_status_json(
    record: &VisionStackLegacyStatusRecord<'_>,
) -> Result<String, VisionStackEvidenceError> {
    to_json_string(record)
}

pub fn serialize_vision_stack_legacy_diagnostics_json(
    record: &VisionStackLegacyDiagnosticsRecord<'_>,
) -> Result<String, VisionStackEvidenceError> {
    to_json_string(record)
}

fn validate_legacy_layout_inputs(
    static_strategy: bool,
    activation_layout: Option<&VisionStackActivationLayout>,
    memory_hardening: Option<&VisionStackMemoryHardeningPlan>,
) -> Result<(), VisionStackEvidenceError> {
    if static_strategy != activation_layout.is_some() {
        return Err(VisionStackEvidenceError::InvalidLegacyPlan(
            "legacy static activation strategy/layout mismatch".to_owned(),
        ));
    }
    if !static_strategy && memory_hardening.is_some() {
        return Err(VisionStackEvidenceError::InvalidLegacyPlan(
            "legacy separate-buffer plan cannot carry memory hardening".to_owned(),
        ));
    }
    Ok(())
}

fn build_vision_stack_legacy_plan_record(
    plan: &VisionStackShardPlan,
    activation_strategy: VisionStackActivationStrategy,
    activation_layout: Option<&VisionStackActivationLayout>,
    min_storage_buffer_offset_alignment: u32,
) -> Result<VisionStackLegacyPlanRecord, VisionStackEvidenceError> {
    let static_values = activation_layout
        .map(|layout| {
            let activation_buffer_count = u32::try_from(layout.physical_buffer_count)
                .map_err(|_| VisionStackEvidenceError::IntegerConversion)?;
            let peak_resident_shard = plan
                .hidden_bytes
                .max(plan.layer_weight_bytes)
                .max(plan.post_norm_bytes);
            let peak_gpu_data_bytes = layout
                .total_activation_bytes
                .checked_add(plan.readback_bytes)
                .and_then(|bytes| bytes.checked_add(peak_resident_shard))
                .ok_or(VisionStackEvidenceError::ArithmeticOverflow)?;
            Ok::<_, VisionStackEvidenceError>((
                activation_buffer_count,
                layout.total_activation_bytes,
                layout.scratch_arena_bytes,
                layout.main_buffers_bytes,
                peak_gpu_data_bytes,
            ))
        })
        .transpose()?;
    let (
        activation_buffer_count,
        activation_arena_bytes,
        scratch_arena_bytes,
        main_buffers_bytes,
        static_activation_strategy,
        static_alignment,
        peak_gpu_data_bytes,
    ) = match static_values {
        Some((buffer_count, arena_bytes, scratch_bytes, main_bytes, peak_bytes)) => (
            buffer_count,
            arena_bytes,
            Some(scratch_bytes),
            Some(main_bytes),
            Some(activation_strategy),
            Some(min_storage_buffer_offset_alignment),
            peak_bytes,
        ),
        None => (
            plan.activation_buffer_count,
            plan.activation_arena_bytes,
            None,
            None,
            None,
            None,
            plan.peak_gpu_data_bytes,
        ),
    };
    Ok(VisionStackLegacyPlanRecord {
        layer_count: plan.layer_count,
        shard_count: plan.shard_count,
        input_bytes: plan.input_bytes,
        hidden_bytes: plan.hidden_bytes,
        intermediate_bytes: plan.intermediate_bytes,
        layer_weight_bytes: plan.layer_weight_bytes,
        post_norm_bytes: plan.post_norm_bytes,
        transport_bytes: plan.transport_bytes,
        activation_buffer_count,
        activation_arena_bytes,
        scratch_arena_bytes,
        main_buffers_bytes,
        activation_strategy: static_activation_strategy,
        min_storage_buffer_offset_alignment: static_alignment,
        readback_bytes: plan.readback_bytes,
        peak_gpu_data_bytes,
        submission_count: plan.submission_count,
        compute_pass_count: plan.compute_pass_count,
        dispatch_count: plan.dispatch_count,
    })
}

fn build_vision_stack_legacy_static_layout_record(
    layout: &VisionStackActivationLayout,
) -> VisionStackLegacyStaticLayoutRecord {
    VisionStackLegacyStaticLayoutRecord {
        scratch_allocations: layout
            .scratch_allocations
            .iter()
            .map(|allocation| VisionStackLegacyScratchAllocationRecord {
                stage: allocation.stage.as_str(),
                offset: allocation.offset,
                size: allocation.size,
                alignment: allocation.alignment,
                first_write: allocation.first_write,
                last_use: allocation.last_use,
            })
            .collect(),
        scratch_arena_bytes: layout.scratch_arena_bytes,
        main_buffers_bytes: layout.main_buffers_bytes,
        total_activation_bytes: layout.total_activation_bytes,
        physical_buffer_count: layout.physical_buffer_count,
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum VisionQkvWebPhysicalBuffer {
    Workspace,
    Readback,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum VisionQkvWebBindGroupKind {
    FusedQkv,
    Attention,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VisionQkvWebBindingResource {
    Norm1Output { byte_length: u64 },
    QueryWeight { byte_length: u64 },
    QueryBias { byte_length: u64 },
    KeyWeight { byte_length: u64 },
    KeyBias { byte_length: u64 },
    ValueWeight { byte_length: u64 },
    ValueBias { byte_length: u64 },
    WorkspaceRange { byte_offset: u64, byte_length: u64 },
    Uniform { slot: u32, byte_length: u64 },
    CuSeqlens { byte_length: u64 },
    AttentionOutput { byte_length: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvWebBindGroupEntry {
    binding: u32,
    resource: VisionQkvWebBindingResource,
}

impl VisionQkvWebBindGroupEntry {
    #[must_use]
    pub const fn binding(&self) -> u32 {
        self.binding
    }

    #[must_use]
    pub const fn resource(&self) -> &VisionQkvWebBindingResource {
        &self.resource
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VisionQkvWebPhysicalCommand {
    CreateBuffer {
        buffer: VisionQkvWebPhysicalBuffer,
        label: &'static str,
        byte_length: u64,
    },
    CreateBindGroup {
        layer_index: u32,
        kind: VisionQkvWebBindGroupKind,
        label: &'static str,
        uniform_slot: u32,
        entries: Vec<VisionQkvWebBindGroupEntry>,
    },
    CopyBuffer {
        label: &'static str,
        source: VisionQkvWebPhysicalBuffer,
        source_offset: u64,
        destination: VisionQkvWebPhysicalBuffer,
        destination_offset: u64,
        byte_length: u64,
    },
    MapRange {
        label: &'static str,
        buffer: VisionQkvWebPhysicalBuffer,
        byte_range: std::ops::Range<u64>,
    },
}

#[derive(Clone, Copy)]
struct VisionQkvWebFusedInvocationAuthority {
    dispatch_workgroups: [u32; 3],
    uniform_words: [u32; 4],
}

#[derive(Clone)]
pub struct VisionQkvWebPhysicalCommandPlan {
    commands: Vec<VisionQkvWebPhysicalCommand>,
    fused_invocations: Vec<VisionQkvWebFusedInvocationAuthority>,
    executed_phases: Cell<u64>,
    layer_count: usize,
}

impl VisionQkvWebPhysicalCommandPlan {
    #[must_use]
    pub fn commands(&self) -> &[VisionQkvWebPhysicalCommand] {
        &self.commands
    }

    #[must_use]
    pub fn fused_dispatch_workgroups(&self, layer_index: u32) -> Option<[u32; 3]> {
        self.fused_invocation(layer_index)
            .map(|authority| authority.dispatch_workgroups)
    }

    #[must_use]
    pub fn fused_uniform_words(&self, layer_index: u32) -> Option<[u32; 4]> {
        self.fused_invocation(layer_index)
            .map(|authority| authority.uniform_words)
    }

    fn fused_invocation(&self, layer_index: u32) -> Option<&VisionQkvWebFusedInvocationAuthority> {
        let layer_index = usize::try_from(layer_index).ok()?;
        self.fused_invocations.get(layer_index)
    }
}

#[must_use]
pub fn plan_vision_qkv_web_physical_commands(
    physical_spec: &VisionQkvPhysicalExecutionSpec,
) -> VisionQkvWebPhysicalCommandPlan {
    let prepared_execution = physical_spec.prepared_execution();
    let readback_layout = physical_spec.readback_layout();
    let workspace = prepared_execution.workspace();
    let mut commands = Vec::with_capacity(
        2 + prepared_execution.layer_count() * 2 + workspace.canaries().len() + 1,
    );
    let mut fused_invocations = Vec::with_capacity(prepared_execution.layer_count());
    commands.push(VisionQkvWebPhysicalCommand::CreateBuffer {
        buffer: VisionQkvWebPhysicalBuffer::Workspace,
        label: "vision-stack-qkv-workspace",
        byte_length: workspace.allocation_bytes(),
    });
    commands.push(VisionQkvWebPhysicalCommand::CreateBuffer {
        buffer: VisionQkvWebPhysicalBuffer::Readback,
        label: "vision-stack-readback",
        byte_length: readback_layout.total_readback_bytes(),
    });
    for layer in prepared_execution.layers() {
        fused_invocations.push(VisionQkvWebFusedInvocationAuthority {
            dispatch_workgroups: layer.invocation().dispatch,
            uniform_words: layer.uniform_words(),
        });
        let [tokens, input_width, output_width, _] = layer.uniform_words();
        let input_bytes = u64::from(tokens) * u64::from(input_width) * 4;
        let weight_bytes = u64::from(output_width) * u64::from(input_width) * 4;
        let bias_bytes = u64::from(output_width) * 4;
        let layer_index = u32::try_from(layer.layer_index())
            .expect("sealed Q/K/V layer index must fit the Web command ABI");
        commands.push(VisionQkvWebPhysicalCommand::CreateBindGroup {
            layer_index,
            kind: VisionQkvWebBindGroupKind::FusedQkv,
            label: "vision-layer-qkv-fused-bind-group",
            uniform_slot: 1,
            entries: vec![
                web_binding(
                    0,
                    VisionQkvWebBindingResource::Norm1Output {
                        byte_length: input_bytes,
                    },
                ),
                web_binding(
                    1,
                    VisionQkvWebBindingResource::QueryWeight {
                        byte_length: weight_bytes,
                    },
                ),
                web_binding(
                    2,
                    VisionQkvWebBindingResource::QueryBias {
                        byte_length: bias_bytes,
                    },
                ),
                web_binding(
                    3,
                    VisionQkvWebBindingResource::KeyWeight {
                        byte_length: weight_bytes,
                    },
                ),
                web_binding(
                    4,
                    VisionQkvWebBindingResource::KeyBias {
                        byte_length: bias_bytes,
                    },
                ),
                web_binding(
                    5,
                    VisionQkvWebBindingResource::ValueWeight {
                        byte_length: weight_bytes,
                    },
                ),
                web_binding(
                    6,
                    VisionQkvWebBindingResource::ValueBias {
                        byte_length: bias_bytes,
                    },
                ),
                web_binding(
                    7,
                    VisionQkvWebBindingResource::WorkspaceRange {
                        byte_offset: workspace.semantic_base(),
                        byte_length: workspace.semantic_bytes(),
                    },
                ),
                web_binding(
                    8,
                    VisionQkvWebBindingResource::Uniform {
                        slot: 1,
                        byte_length: 16,
                    },
                ),
            ],
        });
        let bridge = layer.attention_bridge().bindings();
        let attention_uniform = layer.attention_uniform_words();
        let cu_seqlens_bytes = u64::from(attention_uniform[3] + 1) * 4;
        let attention_output_bytes = bridge[0].byte_length();
        commands.push(VisionQkvWebPhysicalCommand::CreateBindGroup {
            layer_index,
            kind: VisionQkvWebBindGroupKind::Attention,
            label: "vision-layer-attention-bind-group",
            uniform_slot: 4,
            entries: vec![
                web_binding(
                    0,
                    VisionQkvWebBindingResource::WorkspaceRange {
                        byte_offset: workspace.semantic_base() + bridge[0].byte_offset(),
                        byte_length: bridge[0].byte_length(),
                    },
                ),
                web_binding(
                    1,
                    VisionQkvWebBindingResource::WorkspaceRange {
                        byte_offset: workspace.semantic_base() + bridge[1].byte_offset(),
                        byte_length: bridge[1].byte_length(),
                    },
                ),
                web_binding(
                    2,
                    VisionQkvWebBindingResource::WorkspaceRange {
                        byte_offset: workspace.semantic_base() + bridge[2].byte_offset(),
                        byte_length: bridge[2].byte_length(),
                    },
                ),
                web_binding(
                    3,
                    VisionQkvWebBindingResource::CuSeqlens {
                        byte_length: cu_seqlens_bytes,
                    },
                ),
                web_binding(
                    4,
                    VisionQkvWebBindingResource::AttentionOutput {
                        byte_length: attention_output_bytes,
                    },
                ),
                web_binding(
                    5,
                    VisionQkvWebBindingResource::Uniform {
                        slot: 4,
                        byte_length: 16,
                    },
                ),
            ],
        });
    }
    let mut destination_offset = readback_layout.qkv_canary_offset();
    for canary in workspace.canaries() {
        commands.push(VisionQkvWebPhysicalCommand::CopyBuffer {
            label: "vision-stack-qkv-canary-copy",
            source: VisionQkvWebPhysicalBuffer::Workspace,
            source_offset: canary.byte_offset(),
            destination: VisionQkvWebPhysicalBuffer::Readback,
            destination_offset,
            byte_length: canary.byte_length(),
        });
        destination_offset += canary.byte_length();
    }
    debug_assert_eq!(destination_offset, readback_layout.total_readback_bytes());
    commands.push(VisionQkvWebPhysicalCommand::MapRange {
        label: "vision-stack-readback-map",
        buffer: VisionQkvWebPhysicalBuffer::Readback,
        byte_range: 0..readback_layout.total_readback_bytes(),
    });
    VisionQkvWebPhysicalCommandPlan {
        commands,
        fused_invocations,
        executed_phases: Cell::new(0),
        layer_count: prepared_execution.layer_count(),
    }
}

const fn web_binding(
    binding: u32,
    resource: VisionQkvWebBindingResource,
) -> VisionQkvWebBindGroupEntry {
    VisionQkvWebBindGroupEntry { binding, resource }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionQkvWebPhysicalCommandPhase {
    Start,
    Layer { layer_index: u32 },
    Finish,
}

pub trait VisionQkvWebPhysicalCommandEffectSink {
    type CreatedBuffer;
    type CreatedBindGroup;
    type Error;

    fn apply_create_buffer(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<Self::CreatedBuffer, Self::Error>;
    fn store_created_buffer(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
        created: Self::CreatedBuffer,
    ) -> Result<(), Self::Error>;
    fn apply_create_bind_group(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<Self::CreatedBindGroup, Self::Error>;
    fn store_created_bind_group(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
        created: Self::CreatedBindGroup,
    ) -> Result<(), Self::Error>;
    fn apply_copy_buffer(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<(), Self::Error>;
    fn apply_map_range(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<(), Self::Error>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionQkvWebPhysicalCommandValidationError {
    InvalidTrace,
    InvalidPhase,
}

#[derive(Debug)]
pub enum VisionQkvWebPhysicalCommandExecutionError<E> {
    Validation(VisionQkvWebPhysicalCommandValidationError),
    Sink(E),
}

impl<E> VisionQkvWebPhysicalCommandExecutionError<E> {
    #[must_use]
    pub fn into_sink_error(self) -> Option<E> {
        match self {
            Self::Validation(_) => None,
            Self::Sink(error) => Some(error),
        }
    }
}

pub fn validate_vision_qkv_web_physical_command_dispatches(
    plan: &VisionQkvWebPhysicalCommandPlan,
    dispatches: &[(
        VisionQkvWebPhysicalCommandPhase,
        usize,
        &VisionQkvWebPhysicalCommand,
    )],
) -> Result<(), VisionQkvWebPhysicalCommandValidationError> {
    if dispatches.len() != plan.commands.len() {
        return Err(VisionQkvWebPhysicalCommandValidationError::InvalidTrace);
    }
    for (expected_index, (phase, command_index, command)) in dispatches.iter().enumerate() {
        let expected_command = &plan.commands[expected_index];
        if *command_index != expected_index
            || !std::ptr::eq(*command, expected_command)
            || *phase != vision_qkv_web_physical_command_phase(expected_command)
        {
            return Err(VisionQkvWebPhysicalCommandValidationError::InvalidTrace);
        }
    }
    Ok(())
}

pub fn execute_vision_qkv_web_physical_commands<S: VisionQkvWebPhysicalCommandEffectSink>(
    plan: &VisionQkvWebPhysicalCommandPlan,
    phase: VisionQkvWebPhysicalCommandPhase,
    sink: &mut S,
) -> Result<(), VisionQkvWebPhysicalCommandExecutionError<S::Error>> {
    validate_vision_qkv_web_physical_command_phase(plan, phase)
        .map_err(VisionQkvWebPhysicalCommandExecutionError::Validation)?;
    let commands = plan.commands();
    let dispatches = commands
        .iter()
        .enumerate()
        .map(|(command_index, command)| {
            (
                vision_qkv_web_physical_command_phase(command),
                command_index,
                command,
            )
        })
        .collect::<Vec<_>>();
    validate_vision_qkv_web_physical_command_dispatches(plan, &dispatches)
        .map_err(VisionQkvWebPhysicalCommandExecutionError::Validation)?;
    let phase_bit = vision_qkv_web_physical_phase_bit(phase);
    if plan.executed_phases.get() & phase_bit != 0 {
        return Ok(());
    }
    for (command_phase, command_index, command) in dispatches {
        if command_phase != phase {
            continue;
        }
        match command {
            VisionQkvWebPhysicalCommand::CreateBuffer { .. } => {
                let created = match sink.apply_create_buffer(command_index, command) {
                    Ok(created) => created,
                    Err(error) => {
                        return Err(VisionQkvWebPhysicalCommandExecutionError::Sink(error));
                    }
                };
                if let Err(error) = sink.store_created_buffer(command_index, command, created) {
                    return Err(VisionQkvWebPhysicalCommandExecutionError::Sink(error));
                }
            }
            VisionQkvWebPhysicalCommand::CreateBindGroup { .. } => {
                let created = match sink.apply_create_bind_group(command_index, command) {
                    Ok(created) => created,
                    Err(error) => {
                        return Err(VisionQkvWebPhysicalCommandExecutionError::Sink(error));
                    }
                };
                if let Err(error) = sink.store_created_bind_group(command_index, command, created) {
                    return Err(VisionQkvWebPhysicalCommandExecutionError::Sink(error));
                }
            }
            VisionQkvWebPhysicalCommand::CopyBuffer { .. } => {
                if let Err(error) = sink.apply_copy_buffer(command_index, command) {
                    return Err(VisionQkvWebPhysicalCommandExecutionError::Sink(error));
                }
            }
            VisionQkvWebPhysicalCommand::MapRange { .. } => {
                if let Err(error) = sink.apply_map_range(command_index, command) {
                    return Err(VisionQkvWebPhysicalCommandExecutionError::Sink(error));
                }
            }
        }
    }
    plan.executed_phases
        .set(plan.executed_phases.get() | phase_bit);
    Ok(())
}

fn validate_vision_qkv_web_physical_command_phase(
    plan: &VisionQkvWebPhysicalCommandPlan,
    phase: VisionQkvWebPhysicalCommandPhase,
) -> Result<(), VisionQkvWebPhysicalCommandValidationError> {
    if let VisionQkvWebPhysicalCommandPhase::Layer { layer_index } = phase
        && usize::try_from(layer_index).map_or(true, |index| index >= plan.layer_count)
    {
        return Err(VisionQkvWebPhysicalCommandValidationError::InvalidPhase);
    }
    Ok(())
}

const fn vision_qkv_web_physical_command_phase(
    command: &VisionQkvWebPhysicalCommand,
) -> VisionQkvWebPhysicalCommandPhase {
    match command {
        VisionQkvWebPhysicalCommand::CreateBuffer { .. } => VisionQkvWebPhysicalCommandPhase::Start,
        VisionQkvWebPhysicalCommand::CreateBindGroup { layer_index, .. } => {
            VisionQkvWebPhysicalCommandPhase::Layer {
                layer_index: *layer_index,
            }
        }
        VisionQkvWebPhysicalCommand::CopyBuffer { .. }
        | VisionQkvWebPhysicalCommand::MapRange { .. } => VisionQkvWebPhysicalCommandPhase::Finish,
    }
}

const fn vision_qkv_web_physical_phase_bit(phase: VisionQkvWebPhysicalCommandPhase) -> u64 {
    match phase {
        VisionQkvWebPhysicalCommandPhase::Start => 1,
        VisionQkvWebPhysicalCommandPhase::Layer { layer_index } => 1_u64 << (layer_index + 1),
        VisionQkvWebPhysicalCommandPhase::Finish => 1_u64 << 63,
    }
}

struct VisionStackMappedReadbackGuard<Unmap: FnOnce()> {
    unmap: Option<Unmap>,
}

impl<Unmap: FnOnce()> Drop for VisionStackMappedReadbackGuard<Unmap> {
    fn drop(&mut self) {
        if let Some(unmap) = self.unmap.take() {
            unmap();
        }
    }
}

pub fn with_vision_stack_mapped_readback<T, E, Unmap, Access>(
    map_result: Result<(), E>,
    unmap: Unmap,
    access: Access,
) -> Result<T, E>
where
    Unmap: FnOnce(),
    Access: FnOnce() -> Result<T, E>,
{
    map_result?;
    let guard = VisionStackMappedReadbackGuard { unmap: Some(unmap) };
    let result = access();
    drop(guard);
    result
}
