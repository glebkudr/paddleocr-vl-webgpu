use std::{error::Error, fmt};

use pvlc_ir::{PlanErrorCode, PlanIr, SemanticGraph};
use pvlc_model_schema::{TensorDtype, TensorSpec};
use pvlc_runtime_core::{
    InvocationPlan, KernelId, VISION_QKV_FUSED_STORAGE_BINDING_COUNT, VISION_QKV_FUSED_TILE,
    VisionEncoderLayerPlan, VisionEncoderLayerStage, VisionQkvCanaryKind, VisionQkvExecutionPolicy,
    VisionQkvFusedTargetLimits, VisionQkvReadbackLayout, VisionQkvSelectionOutcome,
};

use crate::vision_qkv::{
    Role, VerifiedVisionQkvFusedDescriptorParts, VisionQkvFusedMismatchKind,
    VisionQkvFusionOptions, VisionQkvPassError, VisionQkvPassErrorCode, VisionQkvPassStatus,
    extract_verified_vision_qkv_fused_descriptor, fuse_vision_qkv, lower_vision_qkv_fragment,
    tensor_physical, tensor_semantic,
};

const VISION_LAYER_COUNT: usize = 27;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionQkvStackOverlayErrorCode {
    SchemaVersion,
    CanonicalEncoding,
    StructuralPlan,
    RewriteProvenance,
    SemanticOrTensorIdentity,
    LayerSetOrOrder,
    ConsumerBridge,
    UnsupportedTarget,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvStackOverlayError {
    code: VisionQkvStackOverlayErrorCode,
    message: String,
}

impl VisionQkvStackOverlayError {
    #[must_use]
    pub const fn code(&self) -> VisionQkvStackOverlayErrorCode {
        self.code
    }

    fn new(code: VisionQkvStackOverlayErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for VisionQkvStackOverlayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "vision Q/K/V stack overlay error {:?}: {}",
            self.code, self.message
        )
    }
}

impl Error for VisionQkvStackOverlayError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedVisionQkvAttentionBinding {
    binding: u32,
    value_id: String,
    buffer_id: String,
    byte_offset: u64,
    byte_length: u64,
}

impl VerifiedVisionQkvAttentionBinding {
    #[must_use]
    pub const fn binding(&self) -> u32 {
        self.binding
    }

    #[must_use]
    pub fn value_id(&self) -> &str {
        &self.value_id
    }

    #[must_use]
    pub fn buffer_id(&self) -> &str {
        &self.buffer_id
    }

    #[must_use]
    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    #[must_use]
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedVisionQkvAttentionBridge {
    bindings: Vec<VerifiedVisionQkvAttentionBinding>,
}

impl VerifiedVisionQkvAttentionBridge {
    #[must_use]
    pub fn bindings(&self) -> &[VerifiedVisionQkvAttentionBinding] {
        &self.bindings
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedVisionQkvLayerDescriptor {
    layer_index: usize,
    canonical_plan_blake3_hex: String,
    invocation: InvocationPlan,
    uniform_words: [u32; 4],
    attention_uniform_words: [u32; 4],
    shared_output_bytes: u64,
    attention_bridge: VerifiedVisionQkvAttentionBridge,
}

impl VerifiedVisionQkvLayerDescriptor {
    #[must_use]
    pub const fn layer_index(&self) -> usize {
        self.layer_index
    }

    #[must_use]
    pub fn canonical_plan_blake3_hex(&self) -> &str {
        &self.canonical_plan_blake3_hex
    }

    #[must_use]
    pub const fn invocation(&self) -> InvocationPlan {
        self.invocation
    }

    #[must_use]
    pub const fn uniform_words(&self) -> [u32; 4] {
        self.uniform_words
    }

    #[must_use]
    pub const fn attention_uniform_words(&self) -> [u32; 4] {
        self.attention_uniform_words
    }

    #[must_use]
    pub const fn shared_output_bytes(&self) -> u64 {
        self.shared_output_bytes
    }

    #[must_use]
    pub const fn attention_bridge(&self) -> &VerifiedVisionQkvAttentionBridge {
        &self.attention_bridge
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedVisionQkvStackOverlay {
    layers: Vec<VerifiedVisionQkvLayerDescriptor>,
    target_limits: VisionQkvFusedTargetLimits,
}

impl VerifiedVisionQkvStackOverlay {
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn layers(&self) -> &[VerifiedVisionQkvLayerDescriptor] {
        &self.layers
    }

    #[must_use]
    pub const fn target_limits(&self) -> VisionQkvFusedTargetLimits {
        self.target_limits
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionQkvPreparedExecutionErrorCode {
    InvalidSyntheticGeometry,
    LayerSetOrOrder,
    CrossLayerDrift,
    Kernel,
    Invocation,
    OutputBytes,
    Dispatch,
    Uniform,
    ConsumerBridge,
    WorkspaceLayout,
    DescriptorMismatch,
    ArithmeticOverflow,
    TargetAlignment,
    TargetStorageBindings,
    TargetBindingSize,
    TargetBufferSize,
    TargetDispatchLimit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvPreparedExecutionError {
    code: VisionQkvPreparedExecutionErrorCode,
    message: String,
}

impl VisionQkvPreparedExecutionError {
    #[must_use]
    pub const fn code(&self) -> VisionQkvPreparedExecutionErrorCode {
        self.code
    }

    fn new(code: VisionQkvPreparedExecutionErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for VisionQkvPreparedExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "vision Q/K/V prepared-execution error {:?}: {}",
            self.code, self.message
        )
    }
}

impl Error for VisionQkvPreparedExecutionError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedVisionQkvCanary {
    kind: VisionQkvCanaryKind,
    byte_offset: u64,
    byte_length: u64,
}

impl PreparedVisionQkvCanary {
    #[must_use]
    pub const fn kind(&self) -> VisionQkvCanaryKind {
        self.kind
    }

    #[must_use]
    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    #[must_use]
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedVisionQkvWorkspace {
    semantic_base: u64,
    semantic_bytes: u64,
    allocation_bytes: u64,
    canary_readback_bytes: u64,
    canaries: Vec<PreparedVisionQkvCanary>,
}

impl PreparedVisionQkvWorkspace {
    #[must_use]
    pub const fn semantic_base(&self) -> u64 {
        self.semantic_base
    }

    #[must_use]
    pub const fn semantic_bytes(&self) -> u64 {
        self.semantic_bytes
    }

    #[must_use]
    pub const fn allocation_bytes(&self) -> u64 {
        self.allocation_bytes
    }

    #[must_use]
    pub const fn canary_readback_bytes(&self) -> u64 {
        self.canary_readback_bytes
    }

    #[must_use]
    pub fn canaries(&self) -> &[PreparedVisionQkvCanary] {
        &self.canaries
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedVisionQkvStackExecution {
    layers: Vec<VerifiedVisionQkvLayerDescriptor>,
    workspace: PreparedVisionQkvWorkspace,
}

impl PreparedVisionQkvStackExecution {
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    #[must_use]
    pub fn layers(&self) -> &[VerifiedVisionQkvLayerDescriptor] {
        &self.layers
    }

    #[must_use]
    pub const fn workspace(&self) -> &PreparedVisionQkvWorkspace {
        &self.workspace
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VisionQkvPhysicalBindingErrorCode {
    WorkspaceAllocationMismatch,
    QkvCanaryReadbackMismatch,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvPhysicalBindingError {
    code: VisionQkvPhysicalBindingErrorCode,
    message: String,
}

impl VisionQkvPhysicalBindingError {
    #[must_use]
    pub const fn code(&self) -> VisionQkvPhysicalBindingErrorCode {
        self.code
    }

    fn new(code: VisionQkvPhysicalBindingErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for VisionQkvPhysicalBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "vision Q/K/V physical-binding error {:?}: {}",
            self.code, self.message
        )
    }
}

impl Error for VisionQkvPhysicalBindingError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvPhysicalExecutionSpec {
    prepared_execution: PreparedVisionQkvStackExecution,
    readback_layout: VisionQkvReadbackLayout,
}

impl VisionQkvPhysicalExecutionSpec {
    #[must_use]
    pub const fn prepared_execution(&self) -> &PreparedVisionQkvStackExecution {
        &self.prepared_execution
    }

    #[must_use]
    pub const fn readback_layout(&self) -> &VisionQkvReadbackLayout {
        &self.readback_layout
    }
}

pub fn canonical_synthetic_vision_qkv_tensor_catalog(
    layer_count: usize,
    hidden_size: u32,
) -> Result<Vec<TensorSpec>, VisionQkvPreparedExecutionError> {
    if layer_count == 0 || layer_count > VISION_LAYER_COUNT || hidden_size == 0 {
        return Err(prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::InvalidSyntheticGeometry,
            format!(
                "synthetic Q/K/V catalog geometry {layer_count}/{hidden_size} is outside the fixed model envelope"
            ),
        ));
    }

    let capacity = layer_count.checked_mul(6).ok_or_else(|| {
        prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
            "synthetic Q/K/V catalog capacity overflowed",
        )
    })?;
    let hidden_size = u64::from(hidden_size);
    let mut catalog = Vec::with_capacity(capacity);
    for layer in 0..layer_count {
        for role in Role::ALL {
            catalog.push(TensorSpec {
                name: tensor_physical(layer, role, "weight"),
                dtype: TensorDtype::BFloat16,
                shape: vec![hidden_size, hidden_size],
                semantic_id: tensor_semantic(layer, role, "weight"),
            });
            catalog.push(TensorSpec {
                name: tensor_physical(layer, role, "bias"),
                dtype: TensorDtype::BFloat16,
                shape: vec![hidden_size],
                semantic_id: tensor_semantic(layer, role, "bias"),
            });
        }
    }
    Ok(catalog)
}

pub fn prepare_vision_qkv_stack_execution(
    overlay: &VerifiedVisionQkvStackOverlay,
    layer_count: usize,
    geometry: &VisionEncoderLayerPlan,
    target: VisionQkvFusedTargetLimits,
) -> Result<PreparedVisionQkvStackExecution, VisionQkvPreparedExecutionError> {
    if layer_count == 0
        || layer_count > VISION_LAYER_COUNT
        || overlay.layer_count() != layer_count
        || overlay
            .layers()
            .iter()
            .enumerate()
            .any(|(index, layer)| layer.layer_index() != index)
    {
        return Err(prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::LayerSetOrOrder,
            "prepared Q/K/V layers must be the exact ascending requested range",
        ));
    }

    let alignment = target.min_storage_buffer_offset_alignment;
    if overlay.target_limits().min_storage_buffer_offset_alignment != alignment
        || alignment < 4
        || !alignment.is_power_of_two()
    {
        return Err(prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::TargetAlignment,
            "prepared Q/K/V alignment does not match the verified overlay",
        ));
    }
    if target.max_storage_buffers_per_shader_stage < VISION_QKV_FUSED_STORAGE_BINDING_COUNT {
        return Err(prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::TargetStorageBindings,
            format!(
                "fused Q/K/V requires {VISION_QKV_FUSED_STORAGE_BINDING_COUNT} storage bindings but the target exposes {}",
                target.max_storage_buffers_per_shader_stage
            ),
        ));
    }

    let query_dispatch = geometry
        .dispatches
        .iter()
        .find(|dispatch| dispatch.stage == VisionEncoderLayerStage::Query)
        .ok_or_else(|| {
            prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::DescriptorMismatch,
                "vision geometry has no query projection dispatch",
            )
        })?;
    let key_dispatch = geometry
        .dispatches
        .iter()
        .find(|dispatch| dispatch.stage == VisionEncoderLayerStage::Key)
        .ok_or_else(|| {
            prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::DescriptorMismatch,
                "vision geometry has no key projection dispatch",
            )
        })?;
    let value_dispatch = geometry
        .dispatches
        .iter()
        .find(|dispatch| dispatch.stage == VisionEncoderLayerStage::Value)
        .ok_or_else(|| {
            prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::DescriptorMismatch,
                "vision geometry has no value projection dispatch",
            )
        })?;
    let attention_dispatch = geometry
        .dispatches
        .iter()
        .find(|dispatch| dispatch.stage == VisionEncoderLayerStage::AttentionContext)
        .ok_or_else(|| {
            prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::DescriptorMismatch,
                "vision geometry has no attention-context dispatch",
            )
        })?;
    if query_dispatch.invocation != key_dispatch.invocation
        || query_dispatch.invocation != value_dispatch.invocation
        || query_dispatch.uniform_words != key_dispatch.uniform_words
        || query_dispatch.uniform_words != value_dispatch.uniform_words
    {
        return Err(prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::DescriptorMismatch,
            "vision geometry Q/K/V projections do not share one executor geometry",
        ));
    }

    let [tokens, input_width, output_width, _] = query_dispatch.uniform_words;
    let input_bytes = checked_prepared_product(&[u64::from(tokens), u64::from(input_width), 4])?;
    let weight_bytes =
        checked_prepared_product(&[u64::from(output_width), u64::from(input_width), 4])?;
    let bias_bytes = checked_prepared_product(&[u64::from(output_width), 4])?;
    let plane_bytes = query_dispatch.invocation.output_bytes;
    let first = overlay.layers().first().ok_or_else(|| {
        prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::LayerSetOrOrder,
            "verified Q/K/V overlay contains no layers",
        )
    })?;
    validate_prepared_layer_geometry(
        first,
        [tokens, input_width, output_width],
        plane_bytes,
        u64::from(alignment),
        target.max_compute_workgroups_per_dimension,
    )?;
    for layer in overlay.layers().iter().skip(1) {
        let bridge_layout_differs = layer
            .attention_bridge()
            .bindings()
            .iter()
            .zip(first.attention_bridge().bindings())
            .any(|(actual, expected)| {
                actual.byte_offset() != expected.byte_offset()
                    || actual.byte_length() != expected.byte_length()
            });
        let coherent_alternate_abi = layer.invocation().output_elements
            != first.invocation().output_elements
            && layer.invocation().output_bytes != first.invocation().output_bytes
            && layer.uniform_words()[3] != first.uniform_words()[3]
            && layer.shared_output_bytes() != first.shared_output_bytes()
            && bridge_layout_differs
            && prepared_layer_execution_abi_is_internally_coherent(
                layer,
                plane_bytes,
                u64::from(alignment),
            )?;
        if coherent_alternate_abi {
            return Err(prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::CrossLayerDrift,
                format!(
                    "verified Q/K/V layer {:02} carries a coherent but different executor ABI",
                    layer.layer_index()
                ),
            ));
        }
        validate_prepared_layer_geometry(
            layer,
            [tokens, input_width, output_width],
            plane_bytes,
            u64::from(alignment),
            target.max_compute_workgroups_per_dimension,
        )?;
        if layer.invocation() != first.invocation()
            || layer.uniform_words() != first.uniform_words()
            || layer.shared_output_bytes() != first.shared_output_bytes()
            || layer
                .attention_bridge()
                .bindings()
                .iter()
                .zip(first.attention_bridge().bindings())
                .any(|(actual, expected)| {
                    actual.binding() != expected.binding()
                        || actual.byte_offset() != expected.byte_offset()
                        || actual.byte_length() != expected.byte_length()
                })
        {
            return Err(prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::CrossLayerDrift,
                format!(
                    "verified Q/K/V layer {:02} differs from the prepared executor geometry",
                    layer.layer_index()
                ),
            ));
        }
    }

    let workspace = prepare_vision_qkv_workspace(first, u64::from(alignment))?;
    for (label, bytes) in [
        ("input", input_bytes),
        ("projection weight", weight_bytes),
        ("projection bias", bias_bytes),
        ("shared output", first.shared_output_bytes()),
    ] {
        if bytes > target.max_storage_buffer_binding_size {
            return Err(prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::TargetBindingSize,
                format!(
                    "fused Q/K/V {label} requires {bytes} binding bytes but the target limit is {}",
                    target.max_storage_buffer_binding_size
                ),
            ));
        }
        if bytes > target.max_buffer_size {
            return Err(prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::TargetBufferSize,
                format!(
                    "fused Q/K/V {label} requires {bytes} buffer bytes but the target limit is {}",
                    target.max_buffer_size
                ),
            ));
        }
    }
    if workspace.allocation_bytes() > target.max_buffer_size {
        return Err(prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::TargetBufferSize,
            format!(
                "guarded Q/K/V workspace requires {} buffer bytes but the target limit is {}",
                workspace.allocation_bytes(),
                target.max_buffer_size
            ),
        ));
    }

    let mut layers = overlay.layers().to_vec();
    for layer in &mut layers {
        layer.attention_uniform_words = attention_dispatch.uniform_words;
    }
    Ok(PreparedVisionQkvStackExecution { layers, workspace })
}

pub fn bind_vision_qkv_physical_execution(
    prepared_execution: PreparedVisionQkvStackExecution,
    readback_layout: VisionQkvReadbackLayout,
) -> Result<VisionQkvPhysicalExecutionSpec, VisionQkvPhysicalBindingError> {
    if readback_layout.workspace_allocation_bytes()
        != prepared_execution.workspace().allocation_bytes()
    {
        return Err(VisionQkvPhysicalBindingError::new(
            VisionQkvPhysicalBindingErrorCode::WorkspaceAllocationMismatch,
            "core readback layout workspace allocation differs from prepared Q/K/V authority",
        ));
    }
    if readback_layout.qkv_canary_readback_bytes()
        != prepared_execution.workspace().canary_readback_bytes()
    {
        return Err(VisionQkvPhysicalBindingError::new(
            VisionQkvPhysicalBindingErrorCode::QkvCanaryReadbackMismatch,
            "core readback layout Q/K/V canary bytes differ from prepared Q/K/V authority",
        ));
    }
    Ok(VisionQkvPhysicalExecutionSpec {
        prepared_execution,
        readback_layout,
    })
}

fn validate_prepared_layer_geometry(
    layer: &VerifiedVisionQkvLayerDescriptor,
    legacy_uniform: [u32; 3],
    plane_bytes: u64,
    alignment: u64,
    max_workgroups_per_dimension: u32,
) -> Result<(), VisionQkvPreparedExecutionError> {
    let invocation = layer.invocation();
    let uniform_words = layer.uniform_words();
    let bridge = layer.attention_bridge().bindings();
    let output_elements = u64::try_from(invocation.output_elements).map_err(|_| {
        prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
            "fused Q/K/V output element count does not fit u64",
        )
    })?;
    let output_bytes = output_elements.checked_mul(4).ok_or_else(|| {
        prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
            "fused Q/K/V output byte count overflowed",
        )
    })?;
    let expected_dispatch = [
        legacy_uniform[2].div_ceil(VISION_QKV_FUSED_TILE),
        legacy_uniform[0].div_ceil(VISION_QKV_FUSED_TILE),
        3,
    ];
    if invocation.kernel != KernelId::VisionQkvFusedF32 {
        return Err(prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::Kernel,
            format!(
                "verified Q/K/V layer {:02} does not use the fused Q/K/V kernel",
                layer.layer_index()
            ),
        ));
    }
    if invocation.workgroup_size != [8, 8, 1] {
        return Err(prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::Invocation,
            format!(
                "verified Q/K/V layer {:02} has the wrong workgroup ABI",
                layer.layer_index()
            ),
        ));
    }
    if invocation.dispatch != expected_dispatch {
        return Err(prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::Dispatch,
            format!(
                "verified Q/K/V layer {:02} has the wrong dispatch geometry",
                layer.layer_index()
            ),
        ));
    }
    if invocation
        .dispatch
        .iter()
        .any(|dimension| *dimension > max_workgroups_per_dimension)
    {
        return Err(prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::TargetDispatchLimit,
            format!(
                "verified Q/K/V layer {:02} dispatch {:?} exceeds target limit {max_workgroups_per_dimension}",
                layer.layer_index(),
                invocation.dispatch
            ),
        ));
    }
    if invocation.output_bytes != layer.shared_output_bytes()
        || output_bytes != layer.shared_output_bytes()
    {
        return Err(prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::OutputBytes,
            format!(
                "verified Q/K/V layer {:02} has inconsistent output bytes",
                layer.layer_index()
            ),
        ));
    }
    if uniform_words[..3] != legacy_uniform {
        return Err(prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::Uniform,
            format!(
                "verified Q/K/V layer {:02} has the wrong logical uniform ABI",
                layer.layer_index()
            ),
        ));
    }
    if bridge.len() != 3 {
        return Err(prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::ConsumerBridge,
            format!(
                "verified Q/K/V layer {:02} must expose exactly three attention bindings",
                layer.layer_index()
            ),
        ));
    }

    let plane_stride_bytes = u64::from(uniform_words[3]).checked_mul(4).ok_or_else(|| {
        prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
            "fused Q/K/V plane stride overflowed",
        )
    })?;
    let expected_shared_bytes = plane_stride_bytes.checked_mul(3).ok_or_else(|| {
        prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
            "fused Q/K/V shared output size overflowed",
        )
    })?;
    if plane_stride_bytes < plane_bytes
        || !plane_stride_bytes.is_multiple_of(alignment)
        || expected_shared_bytes != layer.shared_output_bytes()
    {
        return Err(prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::Uniform,
            "fused Q/K/V physical plane stride is incompatible with the logical planes",
        ));
    }

    let expected_values = ["query", "key", "value"]
        .map(|role| format!("vision.layer.{:02}.{role}", layer.layer_index()));
    let expected_buffer = format!("output.vision.layer.{:02}.qkv", layer.layer_index());
    for (index, binding) in bridge.iter().enumerate() {
        let expected_binding = u32::try_from(index).map_err(|_| {
            prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
                "Q/K/V attention binding index overflowed",
            )
        })?;
        let expected_offset = plane_stride_bytes
            .checked_mul(u64::try_from(index).map_err(|_| {
                prepared_execution_error(
                    VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
                    "Q/K/V attention plane index overflowed",
                )
            })?)
            .ok_or_else(|| {
                prepared_execution_error(
                    VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
                    "Q/K/V attention plane offset overflowed",
                )
            })?;
        binding
            .byte_offset()
            .checked_add(binding.byte_length())
            .ok_or_else(|| {
                prepared_execution_error(
                    VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
                    "Q/K/V attention binding byte range overflowed",
                )
            })?;
        if binding.binding() != expected_binding
            || binding.value_id() != expected_values[index]
            || binding.buffer_id() != expected_buffer
        {
            return Err(prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::ConsumerBridge,
                format!(
                    "verified Q/K/V attention binding {:02}/{index} has the wrong consumer identity",
                    layer.layer_index()
                ),
            ));
        }
        if binding.byte_offset() != expected_offset || binding.byte_length() != plane_bytes {
            return Err(prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::WorkspaceLayout,
                format!(
                    "verified Q/K/V attention binding {:02}/{index} differs from the prepared physical layout",
                    layer.layer_index()
                ),
            ));
        }
    }
    Ok(())
}

fn prepared_layer_execution_abi_is_internally_coherent(
    layer: &VerifiedVisionQkvLayerDescriptor,
    plane_bytes: u64,
    alignment: u64,
) -> Result<bool, VisionQkvPreparedExecutionError> {
    let invocation = layer.invocation();
    let uniform_words = layer.uniform_words();
    let output_elements = u64::try_from(invocation.output_elements).map_err(|_| {
        prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
            "fused Q/K/V output element count does not fit u64",
        )
    })?;
    let output_bytes = output_elements.checked_mul(4).ok_or_else(|| {
        prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
            "fused Q/K/V output byte count overflowed",
        )
    })?;
    let plane_stride_bytes = u64::from(uniform_words[3]).checked_mul(4).ok_or_else(|| {
        prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
            "fused Q/K/V plane stride overflowed",
        )
    })?;
    let shared_output_bytes = plane_stride_bytes.checked_mul(3).ok_or_else(|| {
        prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
            "fused Q/K/V shared output size overflowed",
        )
    })?;
    if output_bytes != invocation.output_bytes
        || invocation.output_bytes != layer.shared_output_bytes()
        || shared_output_bytes != layer.shared_output_bytes()
        || plane_stride_bytes < plane_bytes
        || !plane_stride_bytes.is_multiple_of(alignment)
    {
        return Ok(false);
    }
    let bridge = layer.attention_bridge().bindings();
    if bridge.len() != 3 {
        return Ok(false);
    }
    for (index, binding) in bridge.iter().enumerate() {
        let expected_binding = u32::try_from(index).map_err(|_| {
            prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
                "Q/K/V attention binding index overflowed",
            )
        })?;
        let expected_offset = plane_stride_bytes
            .checked_mul(u64::try_from(index).map_err(|_| {
                prepared_execution_error(
                    VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
                    "Q/K/V attention plane index overflowed",
                )
            })?)
            .ok_or_else(|| {
                prepared_execution_error(
                    VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
                    "Q/K/V attention plane offset overflowed",
                )
            })?;
        if binding.binding() != expected_binding
            || binding.byte_offset() != expected_offset
            || binding.byte_length() != plane_bytes
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn prepare_vision_qkv_workspace(
    layer: &VerifiedVisionQkvLayerDescriptor,
    alignment: u64,
) -> Result<PreparedVisionQkvWorkspace, VisionQkvPreparedExecutionError> {
    let semantic_base = alignment;
    let semantic_bytes = layer.shared_output_bytes();
    let semantic_end = semantic_base.checked_add(semantic_bytes).ok_or_else(|| {
        prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
            "Q/K/V workspace semantic end overflowed",
        )
    })?;
    let allocation_bytes = semantic_end.checked_add(alignment).ok_or_else(|| {
        prepared_execution_error(
            VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
            "Q/K/V workspace allocation overflowed",
        )
    })?;
    let bindings = layer.attention_bridge().bindings();
    let mut canaries = vec![PreparedVisionQkvCanary {
        kind: VisionQkvCanaryKind::Prefix,
        byte_offset: 0,
        byte_length: semantic_base,
    }];
    for (plane, binding) in bindings.iter().enumerate() {
        let slice_end = semantic_base
            .checked_add(binding.byte_offset())
            .and_then(|offset| offset.checked_add(binding.byte_length()))
            .ok_or_else(|| {
                prepared_execution_error(
                    VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
                    "Q/K/V workspace slice end overflowed",
                )
            })?;
        let next_offset = match bindings.get(plane + 1) {
            Some(next) => semantic_base
                .checked_add(next.byte_offset())
                .ok_or_else(|| {
                    prepared_execution_error(
                        VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
                        "Q/K/V workspace next slice offset overflowed",
                    )
                })?,
            None => semantic_end,
        };
        let byte_length = next_offset.checked_sub(slice_end).ok_or_else(|| {
            prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::DescriptorMismatch,
                "Q/K/V logical slices overlap in the guarded workspace",
            )
        })?;
        if byte_length != 0 {
            canaries.push(PreparedVisionQkvCanary {
                kind: VisionQkvCanaryKind::InternalPadding { plane },
                byte_offset: slice_end,
                byte_length,
            });
        }
    }
    canaries.push(PreparedVisionQkvCanary {
        kind: VisionQkvCanaryKind::Suffix,
        byte_offset: semantic_end,
        byte_length: alignment,
    });
    let canary_readback_bytes = canaries.iter().try_fold(0_u64, |total, canary| {
        total.checked_add(canary.byte_length()).ok_or_else(|| {
            prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
                "Q/K/V canary readback size overflowed",
            )
        })
    })?;
    Ok(PreparedVisionQkvWorkspace {
        semantic_base,
        semantic_bytes,
        allocation_bytes,
        canary_readback_bytes,
        canaries,
    })
}

fn checked_prepared_product(factors: &[u64]) -> Result<u64, VisionQkvPreparedExecutionError> {
    factors.iter().try_fold(1_u64, |product, factor| {
        product.checked_mul(*factor).ok_or_else(|| {
            prepared_execution_error(
                VisionQkvPreparedExecutionErrorCode::ArithmeticOverflow,
                "prepared Q/K/V byte-size arithmetic overflowed",
            )
        })
    })
}

fn prepared_execution_error(
    code: VisionQkvPreparedExecutionErrorCode,
    message: impl Into<String>,
) -> VisionQkvPreparedExecutionError {
    VisionQkvPreparedExecutionError::new(code, message)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VisionQkvStackSelection {
    policy: VisionQkvExecutionPolicy,
    outcome: VisionQkvSelectionOutcome,
    overlay: Option<VerifiedVisionQkvStackOverlay>,
    fallback_error_code: Option<VisionQkvStackOverlayErrorCode>,
}

impl VisionQkvStackSelection {
    #[must_use]
    pub const fn policy(&self) -> VisionQkvExecutionPolicy {
        self.policy
    }

    #[must_use]
    pub const fn outcome(&self) -> VisionQkvSelectionOutcome {
        self.outcome
    }

    #[must_use]
    pub const fn overlay(&self) -> Option<&VerifiedVisionQkvStackOverlay> {
        self.overlay.as_ref()
    }

    #[must_use]
    pub const fn fallback_error_code(&self) -> Option<VisionQkvStackOverlayErrorCode> {
        self.fallback_error_code
    }
}

pub fn select_vision_qkv_stack_overlay(
    policy: VisionQkvExecutionPolicy,
    build: impl FnOnce() -> Result<VerifiedVisionQkvStackOverlay, VisionQkvStackOverlayError>,
) -> Result<VisionQkvStackSelection, VisionQkvStackOverlayError> {
    if policy == VisionQkvExecutionPolicy::Disabled {
        return Ok(VisionQkvStackSelection {
            policy,
            outcome: VisionQkvSelectionOutcome::Disabled,
            overlay: None,
            fallback_error_code: None,
        });
    }

    match build() {
        Ok(overlay) => Ok(VisionQkvStackSelection {
            policy,
            outcome: VisionQkvSelectionOutcome::Fused,
            overlay: Some(overlay),
            fallback_error_code: None,
        }),
        Err(error)
            if policy == VisionQkvExecutionPolicy::Preferred
                && error.code() == VisionQkvStackOverlayErrorCode::UnsupportedTarget =>
        {
            Ok(VisionQkvStackSelection {
                policy,
                outcome: VisionQkvSelectionOutcome::FallbackUnsupportedTarget,
                overlay: None,
                fallback_error_code: Some(VisionQkvStackOverlayErrorCode::UnsupportedTarget),
            })
        }
        Err(error) => Err(error),
    }
}

pub fn build_verified_vision_qkv_stack_overlay(
    graph: &SemanticGraph,
    layer_count: usize,
    geometry: &VisionEncoderLayerPlan,
    catalog: &[TensorSpec],
    target: VisionQkvFusedTargetLimits,
) -> Result<VerifiedVisionQkvStackOverlay, VisionQkvStackOverlayError> {
    if layer_count == 0 || layer_count > VISION_LAYER_COUNT {
        return Err(VisionQkvStackOverlayError::new(
            VisionQkvStackOverlayErrorCode::LayerSetOrOrder,
            format!("layer count {layer_count} is outside 1..={VISION_LAYER_COUNT}"),
        ));
    }

    let mut canonical_plans = Vec::with_capacity(layer_count);
    for layer in 0..layer_count {
        let lowered = lower_vision_qkv_fragment(graph, layer, geometry, catalog)
            .map_err(map_lowering_error)?;
        let fused = fuse_vision_qkv(
            &lowered,
            graph,
            VisionQkvFusionOptions {
                enabled: true,
                target,
            },
        )
        .map_err(map_build_fusion_error)?;
        if fused.status != VisionQkvPassStatus::Fused {
            return Err(VisionQkvStackOverlayError::new(
                VisionQkvStackOverlayErrorCode::StructuralPlan,
                format!("layer {layer:02} did not produce a fused PlanIR fragment"),
            ));
        }
        canonical_plans.push(fused.plan.canonical_bytes().map_err(map_plan_error)?);
    }
    let expected_layers = (0..layer_count).collect::<Vec<_>>();
    verify_canonical_vision_qkv_stack_overlay(&expected_layers, &canonical_plans, graph, target)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct VisionQkvAttentionBindingCandidate {
    binding: u32,
    value_id: String,
    buffer_id: String,
    byte_offset: u64,
    byte_length: u64,
}

fn verify_canonical_vision_qkv_stack_overlay(
    expected_layers: &[usize],
    canonical_plans: &[Vec<u8>],
    graph: &SemanticGraph,
    target: VisionQkvFusedTargetLimits,
) -> Result<VerifiedVisionQkvStackOverlay, VisionQkvStackOverlayError> {
    if expected_layers.is_empty()
        || expected_layers.len() != canonical_plans.len()
        || expected_layers.len() > VISION_LAYER_COUNT
        || expected_layers
            .iter()
            .enumerate()
            .any(|(position, layer)| *layer != position || *layer >= VISION_LAYER_COUNT)
    {
        return Err(VisionQkvStackOverlayError::new(
            VisionQkvStackOverlayErrorCode::LayerSetOrOrder,
            "stack layer identities must be the exact ascending range from zero",
        ));
    }

    let mut layers = Vec::with_capacity(expected_layers.len());
    for (&expected_layer, canonical) in expected_layers.iter().zip(canonical_plans) {
        let plan = PlanIr::parse_canonical(canonical).map_err(map_plan_error)?;
        let verified = fuse_vision_qkv(
            &plan,
            graph,
            VisionQkvFusionOptions {
                enabled: true,
                target,
            },
        )
        .map_err(map_verified_fusion_error)?;
        if verified.status != VisionQkvPassStatus::UnchangedAlreadyFused {
            return Err(VisionQkvStackOverlayError::new(
                VisionQkvStackOverlayErrorCode::StructuralPlan,
                format!("layer {expected_layer:02} is not an already-fused PlanIR fragment"),
            ));
        }
        let parts = extract_verified_vision_qkv_fused_descriptor(&verified.plan)
            .map_err(map_verified_fusion_error)?;
        if parts.layer != expected_layer {
            return Err(VisionQkvStackOverlayError::new(
                VisionQkvStackOverlayErrorCode::LayerSetOrOrder,
                format!(
                    "expected layer {expected_layer:02}, but canonical fragment belongs to {:02}",
                    parts.layer
                ),
            ));
        }
        layers.push(descriptor_from_verified_parts(
            &plan,
            parts,
            target.min_storage_buffer_offset_alignment,
        )?);
    }
    Ok(VerifiedVisionQkvStackOverlay {
        layers,
        target_limits: target,
    })
}

fn descriptor_from_verified_parts(
    plan: &PlanIr,
    parts: VerifiedVisionQkvFusedDescriptorParts,
    alignment: u32,
) -> Result<VerifiedVisionQkvLayerDescriptor, VisionQkvStackOverlayError> {
    let candidates = parts
        .outputs
        .iter()
        .enumerate()
        .map(|(binding, output)| VisionQkvAttentionBindingCandidate {
            binding: u32::try_from(binding).unwrap_or(u32::MAX),
            value_id: output.id.as_str().to_owned(),
            buffer_id: output.buffer_id.as_str().to_owned(),
            byte_offset: output.byte_offset,
            byte_length: output.byte_length,
        })
        .collect::<Vec<_>>();
    let bridge = verify_attention_candidates(
        parts.layer,
        parts.shared_output_bytes,
        alignment,
        &candidates,
        None,
    )?;
    Ok(VerifiedVisionQkvLayerDescriptor {
        layer_index: parts.layer,
        canonical_plan_blake3_hex: plan.canonical_blake3_hex().map_err(map_plan_error)?,
        invocation: parts.invocation,
        uniform_words: parts.uniform_words,
        attention_uniform_words: [0; 4],
        shared_output_bytes: parts.shared_output_bytes,
        attention_bridge: bridge,
    })
}

#[cfg(test)]
fn verify_vision_qkv_attention_bridge(
    layer: &VerifiedVisionQkvLayerDescriptor,
    candidates: &[VisionQkvAttentionBindingCandidate],
) -> Result<VerifiedVisionQkvAttentionBridge, VisionQkvStackOverlayError> {
    verify_attention_candidates(
        layer.layer_index,
        layer.shared_output_bytes,
        1,
        candidates,
        Some(&layer.attention_bridge),
    )
}

fn verify_attention_candidates(
    layer: usize,
    shared_output_bytes: u64,
    alignment: u32,
    candidates: &[VisionQkvAttentionBindingCandidate],
    expected: Option<&VerifiedVisionQkvAttentionBridge>,
) -> Result<VerifiedVisionQkvAttentionBridge, VisionQkvStackOverlayError> {
    if candidates.len() != 3 {
        return Err(consumer_bridge(
            "attention bridge must contain exactly three bindings",
        ));
    }
    let expected_values =
        ["query", "key", "value"].map(|role| format!("vision.layer.{layer:02}.{role}"));
    let common_buffer = &candidates[0].buffer_id;
    let mut previous_end = 0_u64;
    for (index, candidate) in candidates.iter().enumerate() {
        let end = candidate
            .byte_offset
            .checked_add(candidate.byte_length)
            .ok_or_else(|| consumer_bridge("attention binding byte range overflows"))?;
        if candidate.binding != u32::try_from(index).unwrap_or(u32::MAX)
            || candidate.value_id != expected_values[index]
            || candidate.buffer_id != *common_buffer
            || candidate.byte_length == 0
            || candidate.byte_length == shared_output_bytes
            || end > shared_output_bytes
            || (index != 0 && candidate.byte_offset < previous_end)
            || (alignment > 1 && candidate.byte_offset % u64::from(alignment) != 0)
        {
            return Err(consumer_bridge(format!(
                "attention binding {index} does not match the verified fused output slice"
            )));
        }
        previous_end = end;
    }

    if let Some(expected) = expected
        && (expected.bindings.len() != candidates.len()
            || expected
                .bindings
                .iter()
                .zip(candidates)
                .any(|(left, right)| {
                    left.binding != right.binding
                        || left.value_id != right.value_id
                        || left.buffer_id != right.buffer_id
                        || left.byte_offset != right.byte_offset
                        || left.byte_length != right.byte_length
                }))
    {
        return Err(consumer_bridge(
            "attention bindings differ from the immutable verified descriptor",
        ));
    }

    Ok(VerifiedVisionQkvAttentionBridge {
        bindings: candidates
            .iter()
            .map(|candidate| VerifiedVisionQkvAttentionBinding {
                binding: candidate.binding,
                value_id: candidate.value_id.clone(),
                buffer_id: candidate.buffer_id.clone(),
                byte_offset: candidate.byte_offset,
                byte_length: candidate.byte_length,
            })
            .collect(),
    })
}

fn map_plan_error(error: pvlc_ir::PlanError) -> VisionQkvStackOverlayError {
    let code = match error.code() {
        PlanErrorCode::UnsupportedSchemaVersion => VisionQkvStackOverlayErrorCode::SchemaVersion,
        PlanErrorCode::UnknownField | PlanErrorCode::NonCanonicalEncoding => {
            VisionQkvStackOverlayErrorCode::CanonicalEncoding
        }
        PlanErrorCode::InvalidRewriteProvenance => {
            VisionQkvStackOverlayErrorCode::RewriteProvenance
        }
        _ => VisionQkvStackOverlayErrorCode::StructuralPlan,
    };
    VisionQkvStackOverlayError::new(code, error.to_string())
}

fn map_lowering_error(error: VisionQkvPassError) -> VisionQkvStackOverlayError {
    let code = match error.code() {
        VisionQkvPassErrorCode::InvalidLayer => VisionQkvStackOverlayErrorCode::LayerSetOrOrder,
        VisionQkvPassErrorCode::SemanticMismatch
        | VisionQkvPassErrorCode::MissingTensor
        | VisionQkvPassErrorCode::DuplicateTensor
        | VisionQkvPassErrorCode::TensorIdentityMismatch
        | VisionQkvPassErrorCode::TensorBindingMismatch => {
            VisionQkvStackOverlayErrorCode::SemanticOrTensorIdentity
        }
        _ => VisionQkvStackOverlayErrorCode::StructuralPlan,
    };
    VisionQkvStackOverlayError::new(code, error.to_string())
}

fn map_build_fusion_error(error: VisionQkvPassError) -> VisionQkvStackOverlayError {
    if error.code() == VisionQkvPassErrorCode::InvalidGeometry {
        return VisionQkvStackOverlayError::new(
            VisionQkvStackOverlayErrorCode::UnsupportedTarget,
            error.to_string(),
        );
    }
    map_verified_fusion_error(error)
}

fn map_verified_fusion_error(error: VisionQkvPassError) -> VisionQkvStackOverlayError {
    let code = match error.fused_mismatch_kind() {
        Some(VisionQkvFusedMismatchKind::Structural) => {
            VisionQkvStackOverlayErrorCode::StructuralPlan
        }
        Some(VisionQkvFusedMismatchKind::SemanticOrTensor) => {
            VisionQkvStackOverlayErrorCode::SemanticOrTensorIdentity
        }
        None => match error.code() {
            VisionQkvPassErrorCode::InvalidProvenance => {
                VisionQkvStackOverlayErrorCode::RewriteProvenance
            }
            VisionQkvPassErrorCode::SemanticMismatch
            | VisionQkvPassErrorCode::MissingTensor
            | VisionQkvPassErrorCode::DuplicateTensor
            | VisionQkvPassErrorCode::TensorIdentityMismatch
            | VisionQkvPassErrorCode::TensorBindingMismatch => {
                VisionQkvStackOverlayErrorCode::SemanticOrTensorIdentity
            }
            VisionQkvPassErrorCode::InvalidLayer => VisionQkvStackOverlayErrorCode::LayerSetOrOrder,
            _ => VisionQkvStackOverlayErrorCode::StructuralPlan,
        },
    };
    VisionQkvStackOverlayError::new(code, error.to_string())
}

fn consumer_bridge(message: impl Into<String>) -> VisionQkvStackOverlayError {
    VisionQkvStackOverlayError::new(VisionQkvStackOverlayErrorCode::ConsumerBridge, message)
}

#[cfg(test)]
mod tests;
