#![allow(dead_code)]

use pvlc_ir::{
    PlanBinding, PlanBindingAccess, PlanBindingResource, PlanBufferId, PlanConsumedNode, PlanDtype,
    PlanExternalValue, PlanIr, PlanNode, PlanNodeId, PlanNodeSnapshot, PlanOutput,
    PlanOutputBuffer, PlanRequirements, PlanRewriteProvenance, PlanTensorResource,
    PlanUniformResource, PlanValueId, SemanticGraph, SemanticId,
};
use pvlc_model_schema::{TensorDtype, TensorSpec};
use pvlc_runtime_core::{
    InvocationPlan, KernelId, VisionEncoderLayerGeometry, VisionEncoderLayerPlan,
    VisionQkvFusedTargetLimits,
};

pub const TOKENS: u32 = 3;
pub const INPUT_WIDTH: u32 = 3;
pub const OUTPUT_WIDTH: u32 = 3;
pub const PLANE_BYTES: u64 = 36;
pub const PASS_ID: &str = "vision-qkv-fusion-v1";
pub const EXPECTED_QUERY_NODE_BLAKE3: &str =
    "71c5bef4c2d30df7356ab7a27d3150ae3256246f350a8f3085b2a4e626fd7c12";
pub const EXPECTED_KEY_NODE_BLAKE3: &str =
    "72142caed540417c95b6e1e8127e14861cae761d09d1f38fd856606edc50ab75";
pub const EXPECTED_VALUE_NODE_BLAKE3: &str =
    "74dc35f8d82579fa9c79fb7d0fe4e10799453be36eed6c80b91ef97aa510875b";
pub const EXPECTED_FUSED_CANONICAL_LEN: usize = 8_410;
pub const EXPECTED_FUSED_BLAKE3: &str =
    "11e7a2ea2a5e602e9f3415e74a2190df6df522452fbafad7d62ceb67318da481";

// This is deliberately a literal oracle. It is not assembled by lowering,
// matching, fusion, or the accepted geometry planner.
pub const EXPECTED_FUSED_CANONICAL: &[u8] = br#"{"schema_version":1,"external_values":[{"id":"vision.layer.00.norm1","dtype":"f32","shape":[3,3],"buffer_id":"activation.vision.layer.00.norm1","byte_offset":0,"byte_length":36}],"nodes":[{"id":"vision.layer.00.qkv_fused","invocation":{"kernel":"vision_qkv_fused_f32","output_elements":48,"output_bytes":192,"workgroup_size":[8,8,1],"dispatch":[1,1,3]},"uniform_words":[3,3,3,16],"bindings":[{"number":0,"access":"read_only_storage","resource":{"kind":"value","value_id":"vision.layer.00.norm1"}},{"number":1,"access":"read_only_storage","resource":{"kind":"tensor","physical_name":"visual.vision_model.encoder.layers.0.self_attn.q_proj.weight","semantic_id":"vision.layer.00.attention.q.weight","dtype":"bf16","shape":[3,3],"storage_format":"f32","buffer_id":"tensor.vision.layer.00.q_weight","byte_offset":0,"byte_length":36}},{"number":2,"access":"read_only_storage","resource":{"kind":"tensor","physical_name":"visual.vision_model.encoder.layers.0.self_attn.q_proj.bias","semantic_id":"vision.layer.00.attention.q.bias","dtype":"bf16","shape":[3],"storage_format":"f32","buffer_id":"tensor.vision.layer.00.q_bias","byte_offset":0,"byte_length":12}},{"number":3,"access":"read_only_storage","resource":{"kind":"tensor","physical_name":"visual.vision_model.encoder.layers.0.self_attn.k_proj.weight","semantic_id":"vision.layer.00.attention.k.weight","dtype":"bf16","shape":[3,3],"storage_format":"f32","buffer_id":"tensor.vision.layer.00.k_weight","byte_offset":0,"byte_length":36}},{"number":4,"access":"read_only_storage","resource":{"kind":"tensor","physical_name":"visual.vision_model.encoder.layers.0.self_attn.k_proj.bias","semantic_id":"vision.layer.00.attention.k.bias","dtype":"bf16","shape":[3],"storage_format":"f32","buffer_id":"tensor.vision.layer.00.k_bias","byte_offset":0,"byte_length":12}},{"number":5,"access":"read_only_storage","resource":{"kind":"tensor","physical_name":"visual.vision_model.encoder.layers.0.self_attn.v_proj.weight","semantic_id":"vision.layer.00.attention.v.weight","dtype":"bf16","shape":[3,3],"storage_format":"f32","buffer_id":"tensor.vision.layer.00.v_weight","byte_offset":0,"byte_length":36}},{"number":6,"access":"read_only_storage","resource":{"kind":"tensor","physical_name":"visual.vision_model.encoder.layers.0.self_attn.v_proj.bias","semantic_id":"vision.layer.00.attention.v.bias","dtype":"bf16","shape":[3],"storage_format":"f32","buffer_id":"tensor.vision.layer.00.v_bias","byte_offset":0,"byte_length":12}},{"number":7,"access":"read_write_storage","resource":{"kind":"output_buffer","buffer_id":"output.vision.layer.00.qkv","byte_length":192}},{"number":8,"access":"uniform","resource":{"kind":"uniform_words","words":[3,3,3,16]}}],"outputs":[{"id":"vision.layer.00.query","dtype":"f32","shape":[3,3],"buffer_id":"output.vision.layer.00.qkv","byte_offset":0,"byte_length":36},{"id":"vision.layer.00.key","dtype":"f32","shape":[3,3],"buffer_id":"output.vision.layer.00.qkv","byte_offset":64,"byte_length":36},{"id":"vision.layer.00.value","dtype":"f32","shape":[3,3],"buffer_id":"output.vision.layer.00.qkv","byte_offset":128,"byte_length":36}],"diagnostic_label":"vision.layer.00.qkv.fused","timestamp_label":null,"source_semantic_ids":["vision.layer.00.qkv"],"rewrite_provenance":{"pass_id":"vision-qkv-fusion-v1","source_semantic_ids":["vision.layer.00.qkv"],"consumed":[{"role":"query","original":{"id":"vision.layer.00.query","invocation":{"kernel":"vision_patch_projection_f32","output_elements":9,"output_bytes":36,"workgroup_size":[8,8,1],"dispatch":[1,1,1]},"uniform_words":[3,3,3,0],"bindings":[{"number":0,"access":"read_only_storage","resource":{"kind":"value","value_id":"vision.layer.00.norm1"}},{"number":1,"access":"read_only_storage","resource":{"kind":"tensor","physical_name":"visual.vision_model.encoder.layers.0.self_attn.q_proj.weight","semantic_id":"vision.layer.00.attention.q.weight","dtype":"bf16","shape":[3,3],"storage_format":"f32","buffer_id":"tensor.vision.layer.00.q_weight","byte_offset":0,"byte_length":36}},{"number":2,"access":"read_only_storage","resource":{"kind":"tensor","physical_name":"visual.vision_model.encoder.layers.0.self_attn.q_proj.bias","semantic_id":"vision.layer.00.attention.q.bias","dtype":"bf16","shape":[3],"storage_format":"f32","buffer_id":"tensor.vision.layer.00.q_bias","byte_offset":0,"byte_length":12}},{"number":3,"access":"read_write_storage","resource":{"kind":"output_buffer","buffer_id":"output.vision.layer.00.query","byte_length":36}},{"number":4,"access":"uniform","resource":{"kind":"uniform_words","words":[3,3,3,0]}}],"outputs":[{"id":"vision.layer.00.query","dtype":"f32","shape":[3,3],"buffer_id":"output.vision.layer.00.query","byte_offset":0,"byte_length":36}],"diagnostic_label":"vision.layer.00.query","timestamp_label":"vision.layer.00.query","source_semantic_ids":["vision.layer.00.qkv"]},"canonical_blake3":"71c5bef4c2d30df7356ab7a27d3150ae3256246f350a8f3085b2a4e626fd7c12"},{"role":"key","original":{"id":"vision.layer.00.key","invocation":{"kernel":"vision_patch_projection_f32","output_elements":9,"output_bytes":36,"workgroup_size":[8,8,1],"dispatch":[1,1,1]},"uniform_words":[3,3,3,0],"bindings":[{"number":0,"access":"read_only_storage","resource":{"kind":"value","value_id":"vision.layer.00.norm1"}},{"number":1,"access":"read_only_storage","resource":{"kind":"tensor","physical_name":"visual.vision_model.encoder.layers.0.self_attn.k_proj.weight","semantic_id":"vision.layer.00.attention.k.weight","dtype":"bf16","shape":[3,3],"storage_format":"f32","buffer_id":"tensor.vision.layer.00.k_weight","byte_offset":0,"byte_length":36}},{"number":2,"access":"read_only_storage","resource":{"kind":"tensor","physical_name":"visual.vision_model.encoder.layers.0.self_attn.k_proj.bias","semantic_id":"vision.layer.00.attention.k.bias","dtype":"bf16","shape":[3],"storage_format":"f32","buffer_id":"tensor.vision.layer.00.k_bias","byte_offset":0,"byte_length":12}},{"number":3,"access":"read_write_storage","resource":{"kind":"output_buffer","buffer_id":"output.vision.layer.00.key","byte_length":36}},{"number":4,"access":"uniform","resource":{"kind":"uniform_words","words":[3,3,3,0]}}],"outputs":[{"id":"vision.layer.00.key","dtype":"f32","shape":[3,3],"buffer_id":"output.vision.layer.00.key","byte_offset":0,"byte_length":36}],"diagnostic_label":"vision.layer.00.key","timestamp_label":"vision.layer.00.key","source_semantic_ids":["vision.layer.00.qkv"]},"canonical_blake3":"72142caed540417c95b6e1e8127e14861cae761d09d1f38fd856606edc50ab75"},{"role":"value","original":{"id":"vision.layer.00.value","invocation":{"kernel":"vision_patch_projection_f32","output_elements":9,"output_bytes":36,"workgroup_size":[8,8,1],"dispatch":[1,1,1]},"uniform_words":[3,3,3,0],"bindings":[{"number":0,"access":"read_only_storage","resource":{"kind":"value","value_id":"vision.layer.00.norm1"}},{"number":1,"access":"read_only_storage","resource":{"kind":"tensor","physical_name":"visual.vision_model.encoder.layers.0.self_attn.v_proj.weight","semantic_id":"vision.layer.00.attention.v.weight","dtype":"bf16","shape":[3,3],"storage_format":"f32","buffer_id":"tensor.vision.layer.00.v_weight","byte_offset":0,"byte_length":36}},{"number":2,"access":"read_only_storage","resource":{"kind":"tensor","physical_name":"visual.vision_model.encoder.layers.0.self_attn.v_proj.bias","semantic_id":"vision.layer.00.attention.v.bias","dtype":"bf16","shape":[3],"storage_format":"f32","buffer_id":"tensor.vision.layer.00.v_bias","byte_offset":0,"byte_length":12}},{"number":3,"access":"read_write_storage","resource":{"kind":"output_buffer","buffer_id":"output.vision.layer.00.value","byte_length":36}},{"number":4,"access":"uniform","resource":{"kind":"uniform_words","words":[3,3,3,0]}}],"outputs":[{"id":"vision.layer.00.value","dtype":"f32","shape":[3,3],"buffer_id":"output.vision.layer.00.value","byte_offset":0,"byte_length":36}],"diagnostic_label":"vision.layer.00.value","timestamp_label":"vision.layer.00.value","source_semantic_ids":["vision.layer.00.qkv"]},"canonical_blake3":"74dc35f8d82579fa9c79fb7d0fe4e10799453be36eed6c80b91ef97aa510875b"}]}}],"outputs":["vision.layer.00.query","vision.layer.00.key","vision.layer.00.value"],"requirements":{"storage_binding_count":8,"uniform_binding_count":1,"required_storage_buffer_offset_alignment":32,"largest_storage_binding_bytes":192,"largest_buffer_bytes":192,"max_workgroup_size":[8,8,1],"max_dispatch":[1,1,3],"required_features":[]}}
"#;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    Query,
    Key,
    Value,
}

impl Role {
    pub const ALL: [Self; 3] = [Self::Query, Self::Key, Self::Value];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Key => "key",
            Self::Value => "value",
        }
    }

    pub const fn letter(self) -> &'static str {
        match self {
            Self::Query => "q",
            Self::Key => "k",
            Self::Value => "v",
        }
    }

    pub const fn physical_projection(self) -> &'static str {
        match self {
            Self::Query => "q_proj",
            Self::Key => "k_proj",
            Self::Value => "v_proj",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndependentLayout {
    pub plane_elements: u64,
    pub plane_bytes: u64,
    pub plane_stride_bytes: u64,
    pub physical_bytes: u64,
    pub offsets: [u64; 3],
    pub uniform_words: [u32; 4],
    pub dispatch: [u32; 3],
}

pub fn independent_layout(alignment: u32) -> IndependentLayout {
    assert!(alignment >= 4 && alignment.is_power_of_two());
    let plane_elements = u64::from(TOKENS) * u64::from(OUTPUT_WIDTH);
    let plane_bytes = plane_elements * 4;
    let alignment = u64::from(alignment);
    let plane_stride_bytes = plane_bytes.div_ceil(alignment) * alignment;
    let physical_bytes = plane_stride_bytes * 3;
    IndependentLayout {
        plane_elements,
        plane_bytes,
        plane_stride_bytes,
        physical_bytes,
        offsets: [0, plane_stride_bytes, plane_stride_bytes * 2],
        uniform_words: [
            TOKENS,
            INPUT_WIDTH,
            OUTPUT_WIDTH,
            u32::try_from(plane_stride_bytes / 4).unwrap(),
        ],
        dispatch: [OUTPUT_WIDTH.div_ceil(8), TOKENS.div_ceil(8), 3],
    }
}

pub fn limits(alignment: u32) -> VisionQkvFusedTargetLimits {
    VisionQkvFusedTargetLimits {
        min_storage_buffer_offset_alignment: alignment,
        max_storage_buffers_per_shader_stage: 8,
        max_storage_buffer_binding_size: 1_u64 << 34,
        max_buffer_size: 1_u64 << 34,
        max_compute_workgroups_per_dimension: 65_535,
    }
}

pub fn larger_limits(alignment: u32) -> VisionQkvFusedTargetLimits {
    VisionQkvFusedTargetLimits {
        min_storage_buffer_offset_alignment: alignment,
        max_storage_buffers_per_shader_stage: 32,
        max_storage_buffer_binding_size: 1_u64 << 38,
        max_buffer_size: 1_u64 << 39,
        max_compute_workgroups_per_dimension: 1_000_000,
    }
}

pub fn node_id(value: &str) -> PlanNodeId {
    PlanNodeId::parse(value)
        .unwrap_or_else(|error| panic!("invalid test node ID {value:?}: {error}"))
}

pub fn value_id(value: &str) -> PlanValueId {
    PlanValueId::parse(value)
        .unwrap_or_else(|error| panic!("invalid test value ID {value:?}: {error}"))
}

pub fn buffer_id(value: &str) -> PlanBufferId {
    PlanBufferId::parse(value)
        .unwrap_or_else(|error| panic!("invalid test buffer ID {value:?}: {error}"))
}

pub fn semantic_id(value: &str) -> SemanticId {
    SemanticId::parse(value)
        .unwrap_or_else(|error| panic!("invalid test semantic ID {value:?}: {error}"))
}

pub fn semantic_source(layer: usize) -> String {
    format!("vision.layer.{layer:02}.qkv")
}

pub fn activation_value(layer: usize) -> String {
    format!("vision.layer.{layer:02}.norm1")
}

pub fn output_value(layer: usize, role: Role) -> String {
    format!("vision.layer.{layer:02}.{}", role.name())
}

pub fn projection_node_id(layer: usize, role: Role) -> String {
    format!("vision.layer.{layer:02}.{}", role.name())
}

pub fn tensor_semantic(layer: usize, role: Role, suffix: &str) -> String {
    format!(
        "vision.layer.{layer:02}.attention.{}.{suffix}",
        role.letter()
    )
}

pub fn tensor_physical(layer: usize, role: Role, suffix: &str) -> String {
    format!(
        "visual.vision_model.encoder.layers.{layer}.self_attn.{}.{suffix}",
        role.physical_projection()
    )
}

pub fn tensor_buffer(layer: usize, role: Role, suffix: &str) -> String {
    format!(
        "tensor.vision.layer.{layer:02}.{}_{}",
        role.letter(),
        suffix
    )
}

pub fn output_buffer(layer: usize, role: Role) -> String {
    format!("output.vision.layer.{layer:02}.{}", role.name())
}

pub fn shared_output_buffer(layer: usize) -> String {
    format!("output.vision.layer.{layer:02}.qkv")
}

pub fn compact_catalog(layer: usize) -> Vec<TensorSpec> {
    let mut out = Vec::new();
    for role in Role::ALL {
        out.push(TensorSpec {
            name: tensor_physical(layer, role, "weight"),
            dtype: TensorDtype::BFloat16,
            shape: vec![u64::from(OUTPUT_WIDTH), u64::from(INPUT_WIDTH)],
            semantic_id: tensor_semantic(layer, role, "weight"),
        });
        out.push(TensorSpec {
            name: tensor_physical(layer, role, "bias"),
            dtype: TensorDtype::BFloat16,
            shape: vec![u64::from(OUTPUT_WIDTH)],
            semantic_id: tensor_semantic(layer, role, "bias"),
        });
    }
    out
}

pub fn compact_layer_plan() -> VisionEncoderLayerPlan {
    VisionEncoderLayerGeometry {
        tokens: TOKENS,
        hidden_size: INPUT_WIDTH,
        attention_heads: 1,
        head_dim: INPUT_WIDTH,
        intermediate_size: 5,
        layer_norm_epsilon: 1.0e-5,
        cu_seqlens: &[0, TOKENS],
    }
    .plan()
    .expect("compact independent vision geometry must be accepted")
}

pub fn official_layer_plan() -> VisionEncoderLayerPlan {
    VisionEncoderLayerGeometry {
        tokens: 1,
        hidden_size: 1_152,
        attention_heads: 16,
        head_dim: 72,
        intermediate_size: 4_304,
        layer_norm_epsilon: 1.0e-5,
        cu_seqlens: &[0, 1],
    }
    .plan()
    .expect("official per-layer geometry must be accepted")
}

fn tensor_resource(
    layer: usize,
    role: Role,
    suffix: &str,
    input_width: u32,
    output_width: u32,
) -> PlanTensorResource {
    let shape = match suffix {
        "weight" => vec![u64::from(output_width), u64::from(input_width)],
        "bias" => vec![u64::from(output_width)],
        _ => panic!("unsupported projection tensor suffix"),
    };
    let elements = shape.iter().copied().product::<u64>();
    PlanTensorResource {
        physical_name: tensor_physical(layer, role, suffix),
        semantic_id: semantic_id(&tensor_semantic(layer, role, suffix)),
        dtype: PlanDtype::BFloat16,
        shape,
        storage_format: PlanDtype::Float32,
        buffer_id: buffer_id(&tensor_buffer(layer, role, suffix)),
        byte_offset: 0,
        byte_length: elements * 4,
    }
}

pub fn legacy_projection_node(
    layer: usize,
    role: Role,
    tokens: u32,
    input_width: u32,
    output_width: u32,
) -> PlanNode {
    let output_elements = u64::from(tokens) * u64::from(output_width);
    let output_bytes = output_elements * 4;
    let output_elements = usize::try_from(output_elements).unwrap();
    let uniform_words = [tokens, input_width, output_width, 0];
    PlanNode {
        id: node_id(&projection_node_id(layer, role)),
        invocation: InvocationPlan {
            kernel: KernelId::VisionPatchProjectionF32,
            output_elements,
            output_bytes,
            workgroup_size: [8, 8, 1],
            dispatch: [output_width.div_ceil(32), tokens.div_ceil(32), 1],
        },
        uniform_words,
        bindings: vec![
            PlanBinding {
                number: 0,
                access: PlanBindingAccess::ReadOnlyStorage,
                resource: PlanBindingResource::Value(value_id(&activation_value(layer))),
            },
            PlanBinding {
                number: 1,
                access: PlanBindingAccess::ReadOnlyStorage,
                resource: PlanBindingResource::Tensor(tensor_resource(
                    layer,
                    role,
                    "weight",
                    input_width,
                    output_width,
                )),
            },
            PlanBinding {
                number: 2,
                access: PlanBindingAccess::ReadOnlyStorage,
                resource: PlanBindingResource::Tensor(tensor_resource(
                    layer,
                    role,
                    "bias",
                    input_width,
                    output_width,
                )),
            },
            PlanBinding {
                number: 3,
                access: PlanBindingAccess::ReadWriteStorage,
                resource: PlanBindingResource::OutputBuffer(PlanOutputBuffer {
                    buffer_id: buffer_id(&output_buffer(layer, role)),
                    byte_length: output_bytes,
                }),
            },
            PlanBinding {
                number: 4,
                access: PlanBindingAccess::Uniform,
                resource: PlanBindingResource::UniformWords(PlanUniformResource {
                    words: uniform_words,
                }),
            },
        ],
        outputs: vec![PlanOutput {
            id: value_id(&output_value(layer, role)),
            dtype: PlanDtype::Float32,
            shape: vec![u64::from(tokens), u64::from(output_width)],
            buffer_id: buffer_id(&output_buffer(layer, role)),
            byte_offset: 0,
            byte_length: output_bytes,
        }],
        diagnostic_label: format!("vision.layer.{layer:02}.{}", role.name()),
        timestamp_label: Some(format!("vision.layer.{layer:02}.{}", role.name())),
        source_semantic_ids: vec![semantic_id(&semantic_source(layer))],
        rewrite_provenance: None,
    }
}

pub fn exact_sentinel_requirements() -> PlanRequirements {
    PlanRequirements {
        storage_binding_count: 2,
        uniform_binding_count: 1,
        required_storage_buffer_offset_alignment: 4,
        largest_storage_binding_bytes: 4,
        largest_buffer_bytes: 4,
        max_workgroup_size: [64, 1, 1],
        max_dispatch: [1, 1, 1],
        required_features: vec![],
    }
}

pub fn exact_unfused_requirements(
    tokens: u32,
    input_width: u32,
    output_width: u32,
) -> PlanRequirements {
    let input_bytes = u64::from(tokens) * u64::from(input_width) * 4;
    let weight_bytes = u64::from(output_width) * u64::from(input_width) * 4;
    let bias_bytes = u64::from(output_width) * 4;
    let output_bytes = u64::from(tokens) * u64::from(output_width) * 4;
    let largest = [input_bytes, weight_bytes, bias_bytes, output_bytes]
        .into_iter()
        .max()
        .unwrap();
    PlanRequirements {
        storage_binding_count: 4,
        uniform_binding_count: 1,
        required_storage_buffer_offset_alignment: 4,
        largest_storage_binding_bytes: largest,
        largest_buffer_bytes: largest,
        max_workgroup_size: [8, 8, 1],
        max_dispatch: [output_width.div_ceil(32), tokens.div_ceil(32), 1],
        required_features: vec![],
    }
}

pub fn exact_fused_requirements(alignment: u32) -> PlanRequirements {
    let layout = independent_layout(alignment);
    PlanRequirements {
        storage_binding_count: 8,
        uniform_binding_count: 1,
        required_storage_buffer_offset_alignment: u64::from(alignment),
        largest_storage_binding_bytes: layout.physical_bytes,
        largest_buffer_bytes: layout.physical_bytes,
        max_workgroup_size: [8, 8, 1],
        max_dispatch: layout.dispatch,
        required_features: vec![],
    }
}

fn logical_value_bytes(plan: &PlanIr, value_id: &PlanValueId) -> u64 {
    plan.external_values
        .iter()
        .find(|value| &value.id == value_id)
        .map(|value| value.byte_length)
        .or_else(|| {
            plan.nodes
                .iter()
                .flat_map(|node| &node.outputs)
                .find(|output| &output.id == value_id)
                .map(|output| output.byte_length)
        })
        .unwrap_or_else(|| panic!("test fixture has a dangling value {value_id}"))
}

pub fn independent_requirements(plan: &PlanIr, alignment: u32) -> PlanRequirements {
    let mut storage_binding_count = 0_u32;
    let mut uniform_binding_count = 0_u32;
    let mut largest_storage_binding_bytes = 0_u64;
    let mut largest_buffer_bytes = plan
        .external_values
        .iter()
        .map(|value| value.byte_offset.checked_add(value.byte_length).unwrap())
        .max()
        .unwrap_or(0);
    let mut max_workgroup_size = [0_u32; 3];
    let mut max_dispatch = [0_u32; 3];

    for node in &plan.nodes {
        let mut node_storage_bindings = 0_u32;
        let mut node_uniform_bindings = 0_u32;
        for binding in &node.bindings {
            let binding_bytes = match &binding.resource {
                PlanBindingResource::Value(value_id) => {
                    node_storage_bindings += 1;
                    logical_value_bytes(plan, value_id)
                }
                PlanBindingResource::Tensor(tensor) => {
                    node_storage_bindings += 1;
                    largest_buffer_bytes = largest_buffer_bytes
                        .max(tensor.byte_offset.checked_add(tensor.byte_length).unwrap());
                    tensor.byte_length
                }
                PlanBindingResource::OutputBuffer(output) => {
                    node_storage_bindings += 1;
                    largest_buffer_bytes = largest_buffer_bytes.max(output.byte_length);
                    output.byte_length
                }
                PlanBindingResource::UniformWords(_) => {
                    node_uniform_bindings += 1;
                    0
                }
            };
            largest_storage_binding_bytes = largest_storage_binding_bytes.max(binding_bytes);
        }
        storage_binding_count = storage_binding_count.max(node_storage_bindings);
        uniform_binding_count = uniform_binding_count.max(node_uniform_bindings);
        for dimension in 0..3 {
            max_workgroup_size[dimension] =
                max_workgroup_size[dimension].max(node.invocation.workgroup_size[dimension]);
            max_dispatch[dimension] =
                max_dispatch[dimension].max(node.invocation.dispatch[dimension]);
        }
    }

    PlanRequirements {
        storage_binding_count,
        uniform_binding_count,
        required_storage_buffer_offset_alignment: u64::from(alignment),
        largest_storage_binding_bytes,
        largest_buffer_bytes,
        max_workgroup_size,
        max_dispatch,
        required_features: vec![],
    }
}

pub fn refresh_requirements(plan: &mut PlanIr, alignment: u32) {
    plan.requirements = independent_requirements(plan, alignment);
}

pub fn unfused_plan_for(layer: usize, tokens: u32, input_width: u32, output_width: u32) -> PlanIr {
    let input_bytes = u64::from(tokens) * u64::from(input_width) * 4;
    let plan = PlanIr {
        schema_version: 1,
        external_values: vec![PlanExternalValue {
            id: value_id(&activation_value(layer)),
            dtype: PlanDtype::Float32,
            shape: vec![u64::from(tokens), u64::from(input_width)],
            buffer_id: buffer_id(&format!("activation.{}", activation_value(layer))),
            byte_offset: 0,
            byte_length: input_bytes,
        }],
        nodes: Role::ALL
            .into_iter()
            .map(|role| legacy_projection_node(layer, role, tokens, input_width, output_width))
            .collect(),
        outputs: Role::ALL
            .into_iter()
            .map(|role| value_id(&output_value(layer, role)))
            .collect(),
        requirements: exact_unfused_requirements(tokens, input_width, output_width),
    };
    assert_eq!(plan.requirements, independent_requirements(&plan, 4));
    plan
}

pub fn compact_unfused_plan() -> PlanIr {
    unfused_plan_for(0, TOKENS, INPUT_WIDTH, OUTPUT_WIDTH)
}

pub fn snapshot(node: &PlanNode) -> PlanNodeSnapshot {
    assert!(node.rewrite_provenance.is_none());
    PlanNodeSnapshot {
        id: node.id.clone(),
        invocation: node.invocation,
        uniform_words: node.uniform_words,
        bindings: node.bindings.clone(),
        outputs: node.outputs.clone(),
        diagnostic_label: node.diagnostic_label.clone(),
        timestamp_label: node.timestamp_label.clone(),
        source_semantic_ids: node.source_semantic_ids.clone(),
    }
}

pub fn expected_consumed(node: &PlanNode, role: Role) -> PlanConsumedNode {
    let canonical_blake3 = match role {
        Role::Query => EXPECTED_QUERY_NODE_BLAKE3,
        Role::Key => EXPECTED_KEY_NODE_BLAKE3,
        Role::Value => EXPECTED_VALUE_NODE_BLAKE3,
    };
    PlanConsumedNode {
        role: role.name().to_owned(),
        original: snapshot(node),
        canonical_blake3: canonical_blake3.to_owned(),
    }
}

pub fn expected_fused_plan(alignment: u32) -> PlanIr {
    let unfused = compact_unfused_plan();
    let layout = independent_layout(alignment);
    let shared_buffer = buffer_id(&shared_output_buffer(0));
    let mut bindings = vec![unfused.nodes[0].bindings[0].clone()];
    for node in &unfused.nodes {
        bindings.push(node.bindings[1].clone());
        bindings.push(node.bindings[2].clone());
    }
    bindings.push(PlanBinding {
        number: 7,
        access: PlanBindingAccess::ReadWriteStorage,
        resource: PlanBindingResource::OutputBuffer(PlanOutputBuffer {
            buffer_id: shared_buffer.clone(),
            byte_length: layout.physical_bytes,
        }),
    });
    bindings.push(PlanBinding {
        number: 8,
        access: PlanBindingAccess::Uniform,
        resource: PlanBindingResource::UniformWords(PlanUniformResource {
            words: layout.uniform_words,
        }),
    });
    for (number, binding) in bindings.iter_mut().enumerate() {
        binding.number = u32::try_from(number).unwrap();
    }

    let outputs = Role::ALL
        .into_iter()
        .zip(layout.offsets)
        .map(|(role, byte_offset)| PlanOutput {
            id: value_id(&output_value(0, role)),
            dtype: PlanDtype::Float32,
            shape: vec![u64::from(TOKENS), u64::from(OUTPUT_WIDTH)],
            buffer_id: shared_buffer.clone(),
            byte_offset,
            byte_length: layout.plane_bytes,
        })
        .collect();
    let provenance = PlanRewriteProvenance {
        pass_id: PASS_ID.to_owned(),
        source_semantic_ids: vec![semantic_id(&semantic_source(0))],
        consumed: unfused
            .nodes
            .iter()
            .zip(Role::ALL)
            .map(|(node, role)| expected_consumed(node, role))
            .collect(),
    };
    let fused = PlanNode {
        id: node_id("vision.layer.00.qkv_fused"),
        invocation: InvocationPlan {
            kernel: KernelId::VisionQkvFusedF32,
            output_elements: usize::try_from(layout.physical_bytes / 4).unwrap(),
            output_bytes: layout.physical_bytes,
            workgroup_size: [8, 8, 1],
            dispatch: layout.dispatch,
        },
        uniform_words: layout.uniform_words,
        bindings,
        outputs,
        diagnostic_label: "vision.layer.00.qkv.fused".to_owned(),
        timestamp_label: None,
        source_semantic_ids: vec![semantic_id(&semantic_source(0))],
        rewrite_provenance: Some(provenance),
    };
    let plan = PlanIr {
        schema_version: 1,
        external_values: unfused.external_values,
        nodes: vec![fused],
        outputs: unfused.outputs,
        requirements: exact_fused_requirements(alignment),
    };
    assert_eq!(
        plan.requirements,
        independent_requirements(&plan, alignment)
    );
    plan
}

pub fn sentinel_node(label: &str, source: &str) -> (PlanExternalValue, PlanNode, PlanValueId) {
    let external_id = value_id(&format!("value.sentinel.{label}.input"));
    let external_buffer = buffer_id(&format!("buffer.sentinel.{label}.input"));
    let output_id = value_id(&format!("value.sentinel.{label}.output"));
    let output_buffer = buffer_id(&format!("buffer.sentinel.{label}.output"));
    let external = PlanExternalValue {
        id: external_id.clone(),
        dtype: PlanDtype::Float32,
        shape: vec![1],
        buffer_id: external_buffer,
        byte_offset: 0,
        byte_length: 4,
    };
    let uniform_words = [1, 0, 0, 0];
    let node = PlanNode {
        id: node_id(&format!("node.sentinel.{label}")),
        invocation: InvocationPlan {
            kernel: KernelId::SiluF32,
            output_elements: 1,
            output_bytes: 4,
            workgroup_size: [64, 1, 1],
            dispatch: [1, 1, 1],
        },
        uniform_words,
        bindings: vec![
            PlanBinding {
                number: 0,
                access: PlanBindingAccess::ReadOnlyStorage,
                resource: PlanBindingResource::Value(external_id),
            },
            PlanBinding {
                number: 1,
                access: PlanBindingAccess::ReadWriteStorage,
                resource: PlanBindingResource::OutputBuffer(PlanOutputBuffer {
                    buffer_id: output_buffer.clone(),
                    byte_length: 4,
                }),
            },
            PlanBinding {
                number: 2,
                access: PlanBindingAccess::Uniform,
                resource: PlanBindingResource::UniformWords(PlanUniformResource {
                    words: uniform_words,
                }),
            },
        ],
        outputs: vec![PlanOutput {
            id: output_id.clone(),
            dtype: PlanDtype::Float32,
            shape: vec![1],
            buffer_id: output_buffer,
            byte_offset: 0,
            byte_length: 4,
        }],
        diagnostic_label: format!("sentinel.{label}"),
        timestamp_label: None,
        source_semantic_ids: vec![semantic_id(source)],
        rewrite_provenance: None,
    };
    (external, node, output_id)
}

pub fn compact_unfused_with_sentinels() -> PlanIr {
    let mut plan = compact_unfused_plan();
    let (before_external, before_node, before_output) =
        sentinel_node("before", "preprocess.resize");
    let (after_external, after_node, after_output) = sentinel_node("after", "preprocess.normalize");
    plan.external_values.insert(0, before_external);
    plan.external_values.push(after_external);
    plan.nodes.insert(0, before_node);
    plan.nodes.push(after_node);
    plan.outputs.insert(0, before_output);
    plan.outputs.push(after_output);
    refresh_requirements(&mut plan, 4);
    plan
}

pub fn unrelated_plan() -> PlanIr {
    let (external, node, output) = sentinel_node("unrelated", "preprocess.resize");
    let plan = PlanIr {
        schema_version: 1,
        external_values: vec![external],
        nodes: vec![node],
        outputs: vec![output],
        requirements: exact_sentinel_requirements(),
    };
    assert_eq!(plan.requirements, independent_requirements(&plan, 4));
    plan
}

pub fn canonical_graph() -> SemanticGraph {
    let graph = SemanticGraph::paddleocr_vl_16();
    graph.verify().expect("built-in SemanticIR must verify");
    graph
}

pub fn append_cluster(plan: &mut PlanIr, layer: usize) {
    let cluster = unfused_plan_for(layer, TOKENS, INPUT_WIDTH, OUTPUT_WIDTH);
    plan.external_values.extend(cluster.external_values);
    plan.nodes.extend(cluster.nodes);
    plan.outputs.extend(cluster.outputs);
    refresh_requirements(plan, 4);
}

pub fn tensor_mut(node: &mut PlanNode, binding: usize) -> &mut PlanTensorResource {
    let PlanBindingResource::Tensor(tensor) = &mut node.bindings[binding].resource else {
        panic!("binding {binding} must be a tensor in the independent fixture");
    };
    tensor
}

pub fn uniform_mut(node: &mut PlanNode) -> &mut PlanUniformResource {
    let index = node.bindings.len() - 1;
    let PlanBindingResource::UniformWords(uniform) = &mut node.bindings[index].resource else {
        panic!("last binding must be uniform words in the independent fixture");
    };
    uniform
}

pub fn output_buffer_mut(node: &mut PlanNode) -> &mut PlanOutputBuffer {
    let index = node.bindings.len() - 2;
    let PlanBindingResource::OutputBuffer(output) = &mut node.bindings[index].resource else {
        panic!("penultimate binding must be the output buffer in the independent fixture");
    };
    output
}
