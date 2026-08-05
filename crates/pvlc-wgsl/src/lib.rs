//! Fixed FP32 WGSL catalog and structural ABI verifier.

use std::{collections::BTreeSet, error::Error, fmt};

use naga::{
    AddressSpace, ArraySize, Module, ScalarKind, ShaderStage, StorageAccess, TypeInner, VectorSize,
    valid::{Capabilities, ValidationFlags, Validator},
};
use pvlc_runtime_core::KernelId;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingKind {
    StorageReadF32,
    StorageReadVec4F32,
    StorageReadF16,
    StorageReadVec4F16,
    StorageReadU32,
    StorageReadWriteF32,
    StorageReadWriteF16,
    StorageReadWriteVec4F16,
    Uniform,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindingSpec {
    pub group: u32,
    pub binding: u32,
    pub kind: BindingKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UniformScalar {
    U32,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UniformFieldSpec {
    pub name: &'static str,
    pub scalar: UniformScalar,
    pub offset: u32,
}

impl UniformFieldSpec {
    #[must_use]
    pub const fn new(name: &'static str, scalar: UniformScalar, offset: u32) -> Self {
        Self {
            name,
            scalar,
            offset,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelSpec {
    pub kernel: KernelId,
    pub entry_point: &'static str,
    pub workgroup_size: [u32; 3],
    pub bindings: &'static [BindingSpec],
    pub uniform_fields: &'static [UniformFieldSpec],
    pub uniform_span: u32,
    pub required_features: &'static [&'static str],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelModule {
    pub spec: KernelSpec,
    pub source: &'static str,
}

impl KernelModule {
    #[must_use]
    pub const fn source_for_build(&self) -> &'static str {
        self.source
    }
}

const GEMM_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadF32),
    binding(1, BindingKind::StorageReadF32),
    binding(2, BindingKind::StorageReadWriteF32),
    binding(3, BindingKind::Uniform),
];
const GEMV_BINDINGS: &[BindingSpec] = GEMM_BINDINGS;
const GEMV_TILED_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadVec4F32),
    binding(1, BindingKind::StorageReadVec4F32),
    binding(2, BindingKind::StorageReadWriteF32),
    binding(3, BindingKind::Uniform),
];
const ADD_BINDINGS: &[BindingSpec] = GEMM_BINDINGS;
const LAYER_NORM_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadF32),
    binding(1, BindingKind::StorageReadF32),
    binding(2, BindingKind::StorageReadF32),
    binding(3, BindingKind::StorageReadWriteF32),
    binding(4, BindingKind::Uniform),
];
const RMS_NORM_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadF32),
    binding(1, BindingKind::StorageReadF32),
    binding(2, BindingKind::StorageReadWriteF32),
    binding(3, BindingKind::Uniform),
];
const RMS_NORM_F16_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadF32),
    binding(1, BindingKind::StorageReadF16),
    binding(2, BindingKind::StorageReadWriteF32),
    binding(3, BindingKind::Uniform),
];
const GEMV_TILED_F16_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadVec4F16),
    binding(1, BindingKind::StorageReadVec4F32),
    binding(2, BindingKind::StorageReadWriteF32),
    binding(3, BindingKind::Uniform),
];
const LINEAR_PROJECTION_F16_WEIGHT_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadVec4F32),
    binding(1, BindingKind::StorageReadVec4F16),
    binding(2, BindingKind::StorageReadVec4F32),
    binding(3, BindingKind::StorageReadWriteF32),
    binding(4, BindingKind::Uniform),
];
const ACTIVATION_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadF32),
    binding(1, BindingKind::StorageReadWriteF32),
    binding(2, BindingKind::Uniform),
];
const ROPE_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadF32),
    binding(1, BindingKind::StorageReadU32),
    binding(2, BindingKind::StorageReadWriteF32),
    binding(3, BindingKind::Uniform),
];
const VISION_ATTENTION_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadF32),
    binding(1, BindingKind::StorageReadF32),
    binding(2, BindingKind::StorageReadF32),
    binding(3, BindingKind::StorageReadU32),
    binding(4, BindingKind::StorageReadWriteF32),
    binding(5, BindingKind::Uniform),
];
const VISION_PATCH_PROJECTION_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadF32),
    binding(1, BindingKind::StorageReadF32),
    binding(2, BindingKind::StorageReadF32),
    binding(3, BindingKind::StorageReadWriteF32),
    binding(4, BindingKind::Uniform),
];
const PROJECTOR_MERGE_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadF32),
    binding(1, BindingKind::StorageReadU32),
    binding(2, BindingKind::StorageReadWriteF32),
    binding(3, BindingKind::Uniform),
];
const PROJECTOR_MERGE_F16_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadVec4F16),
    binding(1, BindingKind::StorageReadU32),
    binding(2, BindingKind::StorageReadWriteVec4F16),
    binding(3, BindingKind::Uniform),
];
const VISION_QKV_FUSED_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadF32),
    binding(1, BindingKind::StorageReadF32),
    binding(2, BindingKind::StorageReadF32),
    binding(3, BindingKind::StorageReadF32),
    binding(4, BindingKind::StorageReadF32),
    binding(5, BindingKind::StorageReadF32),
    binding(6, BindingKind::StorageReadF32),
    binding(7, BindingKind::StorageReadWriteF32),
    binding(8, BindingKind::Uniform),
];
const VISION_QKV_FUSED_F16_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadVec4F32),
    binding(1, BindingKind::StorageReadVec4F16),
    binding(2, BindingKind::StorageReadVec4F32),
    binding(3, BindingKind::StorageReadVec4F16),
    binding(4, BindingKind::StorageReadVec4F32),
    binding(5, BindingKind::StorageReadVec4F16),
    binding(6, BindingKind::StorageReadVec4F32),
    binding(7, BindingKind::StorageReadWriteF32),
    binding(8, BindingKind::Uniform),
];
const LAYER_NORM_F16_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadVec4F16),
    binding(1, BindingKind::StorageReadVec4F16),
    binding(2, BindingKind::StorageReadVec4F16),
    binding(3, BindingKind::StorageReadWriteVec4F16),
    binding(4, BindingKind::Uniform),
];
const LINEAR_PROJECTION_F16_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadVec4F16),
    binding(1, BindingKind::StorageReadVec4F16),
    binding(2, BindingKind::StorageReadVec4F16),
    binding(3, BindingKind::StorageReadWriteVec4F16),
    binding(4, BindingKind::Uniform),
];
const VISION_ATTENTION_F16_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadVec4F16),
    binding(1, BindingKind::StorageReadVec4F16),
    binding(2, BindingKind::StorageReadVec4F16),
    binding(3, BindingKind::StorageReadU32),
    binding(4, BindingKind::StorageReadWriteVec4F16),
    binding(5, BindingKind::Uniform),
];
const ADD_F16_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadVec4F16),
    binding(1, BindingKind::StorageReadVec4F16),
    binding(2, BindingKind::StorageReadWriteVec4F16),
    binding(3, BindingKind::Uniform),
];
const ACTIVATION_F16_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadVec4F16),
    binding(1, BindingKind::StorageReadWriteVec4F16),
    binding(2, BindingKind::Uniform),
];
const VISION_ROPE_2D_F16_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadWriteF16),
    binding(1, BindingKind::StorageReadWriteF16),
    binding(2, BindingKind::StorageReadF32),
    binding(3, BindingKind::StorageReadF32),
    binding(4, BindingKind::Uniform),
];
const VISION_ROPE_2D_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadWriteF32),
    binding(1, BindingKind::StorageReadWriteF32),
    binding(2, BindingKind::StorageReadF32),
    binding(3, BindingKind::StorageReadF32),
    binding(4, BindingKind::Uniform),
];
const DECODER_KV_APPEND_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadF32),
    binding(1, BindingKind::StorageReadF32),
    binding(2, BindingKind::StorageReadWriteF32),
    binding(3, BindingKind::StorageReadWriteF32),
    binding(4, BindingKind::Uniform),
];
const DECODER_GQA_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadF32),
    binding(1, BindingKind::StorageReadF32),
    binding(2, BindingKind::StorageReadF32),
    binding(3, BindingKind::StorageReadWriteF32),
    binding(4, BindingKind::Uniform),
];
const DECODER_MROPE_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadF32),
    binding(1, BindingKind::StorageReadF32),
    binding(2, BindingKind::StorageReadF32),
    binding(3, BindingKind::StorageReadF32),
    binding(4, BindingKind::StorageReadWriteF32),
    binding(5, BindingKind::StorageReadWriteF32),
    binding(6, BindingKind::Uniform),
];
const DECODER_SWIGLU_BINDINGS: &[BindingSpec] = &[
    binding(0, BindingKind::StorageReadF32),
    binding(1, BindingKind::StorageReadF32),
    binding(2, BindingKind::StorageReadWriteF32),
    binding(3, BindingKind::Uniform),
];

const GEMM_UNIFORM: &[UniformFieldSpec] = &[
    uniform("rows", UniformScalar::U32, 0),
    uniform("inner", UniformScalar::U32, 4),
    uniform("columns", UniformScalar::U32, 8),
    uniform("padding", UniformScalar::U32, 12),
];
const GEMV_UNIFORM: &[UniformFieldSpec] = &[
    uniform("rows", UniformScalar::U32, 0),
    uniform("columns", UniformScalar::U32, 4),
    uniform("padding0", UniformScalar::U32, 8),
    uniform("padding1", UniformScalar::U32, 12),
];
const NORM_UNIFORM: &[UniformFieldSpec] = &[
    uniform("rows", UniformScalar::U32, 0),
    uniform("width", UniformScalar::U32, 4),
    uniform("epsilon", UniformScalar::F32, 8),
    uniform("padding", UniformScalar::U32, 12),
];
const ACTIVATION_UNIFORM: &[UniformFieldSpec] = &[
    uniform("length", UniformScalar::U32, 0),
    uniform("padding0", UniformScalar::U32, 4),
    uniform("padding1", UniformScalar::U32, 8),
    uniform("padding2", UniformScalar::U32, 12),
];
const ROPE_UNIFORM: &[UniformFieldSpec] = &[
    uniform("rows", UniformScalar::U32, 0),
    uniform("width", UniformScalar::U32, 4),
    uniform("rotary_dim", UniformScalar::U32, 8),
    uniform("base", UniformScalar::F32, 12),
];
const VISION_ATTENTION_UNIFORM: &[UniformFieldSpec] = &[
    uniform("tokens", UniformScalar::U32, 0),
    uniform("heads", UniformScalar::U32, 4),
    uniform("head_dim", UniformScalar::U32, 8),
    uniform("segments", UniformScalar::U32, 12),
];
const VISION_PATCH_PROJECTION_UNIFORM: &[UniformFieldSpec] = &[
    uniform("patch_count", UniformScalar::U32, 0),
    uniform("input_width", UniformScalar::U32, 4),
    uniform("output_width", UniformScalar::U32, 8),
    uniform("padding", UniformScalar::U32, 12),
];
const PROJECTOR_MERGE_UNIFORM: &[UniformFieldSpec] = &[
    uniform("output_tokens", UniformScalar::U32, 0),
    uniform("hidden_size", UniformScalar::U32, 4),
    uniform("length", UniformScalar::U32, 8),
    uniform("row_stride", UniformScalar::U32, 12),
];
const VISION_QKV_FUSED_UNIFORM: &[UniformFieldSpec] = &[
    uniform("tokens", UniformScalar::U32, 0),
    uniform("input_width", UniformScalar::U32, 4),
    uniform("output_width", UniformScalar::U32, 8),
    uniform("plane_stride_elements", UniformScalar::U32, 12),
];
const VISION_ROPE_2D_UNIFORM: &[UniformFieldSpec] = &[
    uniform("tokens", UniformScalar::U32, 0),
    uniform("heads", UniformScalar::U32, 4),
    uniform("head_dim", UniformScalar::U32, 8),
    uniform("padding", UniformScalar::U32, 12),
];
const DECODER_KV_APPEND_UNIFORM: &[UniformFieldSpec] = &[
    uniform("prefix_tokens", UniformScalar::U32, 0),
    uniform("key_value_heads", UniformScalar::U32, 4),
    uniform("head_dim", UniformScalar::U32, 8),
    uniform("cache_capacity", UniformScalar::U32, 12),
];
const DECODER_GQA_UNIFORM: &[UniformFieldSpec] = &[
    uniform("cache_tokens", UniformScalar::U32, 0),
    uniform("query_heads", UniformScalar::U32, 4),
    uniform("key_value_heads", UniformScalar::U32, 8),
    uniform("head_dim", UniformScalar::U32, 12),
];
const DECODER_MROPE_UNIFORM: &[UniformFieldSpec] = &[
    uniform("position", UniformScalar::U32, 0),
    uniform("rope_capacity", UniformScalar::U32, 4),
    uniform("padding0", UniformScalar::U32, 8),
    uniform("padding1", UniformScalar::U32, 12),
];
const DECODER_SWIGLU_UNIFORM: &[UniformFieldSpec] = &[
    uniform("length", UniformScalar::U32, 0),
    uniform("padding0", UniformScalar::U32, 4),
    uniform("padding1", UniformScalar::U32, 8),
    uniform("padding2", UniformScalar::U32, 12),
];
const DECODER_PREFILL_GQA_UNIFORM: &[UniformFieldSpec] = &[
    uniform("tokens", UniformScalar::U32, 0),
    uniform("query_heads", UniformScalar::U32, 4),
    uniform("key_value_heads", UniformScalar::U32, 8),
    uniform("head_dim", UniformScalar::U32, 12),
];
const DECODER_PREFILL_MROPE_UNIFORM: &[UniformFieldSpec] = &[
    uniform("tokens", UniformScalar::U32, 0),
    uniform("rope_capacity", UniformScalar::U32, 4),
    uniform("padding0", UniformScalar::U32, 8),
    uniform("padding1", UniformScalar::U32, 12),
];
const DECODER_KV_APPEND_RANGE_UNIFORM: &[UniformFieldSpec] = &[
    uniform("tokens", UniformScalar::U32, 0),
    uniform("cache_capacity", UniformScalar::U32, 4),
    uniform("padding0", UniformScalar::U32, 8),
    uniform("padding1", UniformScalar::U32, 12),
];
const DECODER_GQA_SPLIT_UNIFORM: &[UniformFieldSpec] = &[
    uniform("cache_tokens", UniformScalar::U32, 0),
    uniform("chunk_count", UniformScalar::U32, 4),
    uniform("padding0", UniformScalar::U32, 8),
    uniform("padding1", UniformScalar::U32, 12),
];

const fn binding(binding: u32, kind: BindingKind) -> BindingSpec {
    BindingSpec {
        group: 0,
        binding,
        kind,
    }
}

const fn uniform(name: &'static str, scalar: UniformScalar, offset: u32) -> UniformFieldSpec {
    UniformFieldSpec::new(name, scalar, offset)
}

const GEMM_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    rows: u32,
    inner: u32,
    columns: u32,
    padding: u32,
}
@group(0) @binding(0) var<storage, read> left: F32Buffer;
@group(0) @binding(1) var<storage, read> right: F32Buffer;
@group(0) @binding(2) var<storage, read_write> output: F32Buffer;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let column = global_id.x;
    let row = global_id.y;
    if row >= params.rows || column >= params.columns {
        return;
    }
    var accumulator = 0.0;
    for (var depth = 0u; depth < params.inner; depth = depth + 1u) {
        accumulator = accumulator + left.data[row * params.inner + depth] * right.data[depth * params.columns + column];
    }
    output.data[row * params.columns + column] = accumulator;
}
"#;

const GEMV_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    rows: u32,
    columns: u32,
    padding0: u32,
    padding1: u32,
}
@group(0) @binding(0) var<storage, read> matrix: F32Buffer;
@group(0) @binding(1) var<storage, read> vector: F32Buffer;
@group(0) @binding(2) var<storage, read_write> output: F32Buffer;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if row >= params.rows {
        return;
    }
    var accumulator = 0.0;
    for (var column = 0u; column < params.columns; column = column + 1u) {
        accumulator = accumulator + matrix.data[row * params.columns + column] * vector.data[column];
    }
    output.data[row] = accumulator;
}
"#;

const GEMV_TILED_SOURCE: &str = r#"struct Vec4Buffer {
    data: array<vec4<f32>>,
}
struct F32Buffer {
    data: array<f32>,
}
struct Params {
    rows: u32,
    columns: u32,
    padding0: u32,
    padding1: u32,
}
const TILE_ROWS: u32 = 8u;
const THREADS_PER_ROW: u32 = 32u;
const VECTOR_WIDTH: u32 = 4u;
const SHARED_VEC4_CAPACITY: u32 = 768u;
@group(0) @binding(0) var<storage, read> matrix: Vec4Buffer;
@group(0) @binding(1) var<storage, read> vector: Vec4Buffer;
@group(0) @binding(2) var<storage, read_write> output: F32Buffer;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> shared_vector: array<vec4<f32>, 768>;
var<workgroup> partials: array<f32, 256>;
@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let vector_columns = params.columns / VECTOR_WIDTH;
    for (var staged = local_id.x; staged < vector_columns; staged = staged + 256u) {
        shared_vector[staged] = vector.data[staged];
    }
    workgroupBarrier();

    let row_group = local_id.x / THREADS_PER_ROW;
    let lane = local_id.x % THREADS_PER_ROW;
    let row = workgroup_id.x * TILE_ROWS + row_group;
    var partial = 0.0;
    if row < params.rows {
        let row_base = row * vector_columns;
        for (var column = lane; column < vector_columns; column = column + THREADS_PER_ROW) {
            let products = matrix.data[row_base + column] * shared_vector[column];
            partial = partial + products.x;
            partial = partial + products.y;
            partial = partial + products.z;
            partial = partial + products.w;
        }
    }
    partials[local_id.x] = partial;
    workgroupBarrier();

    for (var stride = THREADS_PER_ROW / 2u; stride > 0u; stride = stride >> 1u) {
        if lane < stride {
            partials[local_id.x] = partials[local_id.x] + partials[local_id.x + stride];
        }
        workgroupBarrier();
    }
    if lane == 0u && row < params.rows {
        output.data[row] = partials[local_id.x];
    }
}
"#;

const LAYER_NORM_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    rows: u32,
    width: u32,
    epsilon: f32,
    padding: u32,
}
@group(0) @binding(0) var<storage, read> input: F32Buffer;
@group(0) @binding(1) var<storage, read> weight: F32Buffer;
@group(0) @binding(2) var<storage, read> bias: F32Buffer;
@group(0) @binding(3) var<storage, read_write> output: F32Buffer;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if row >= params.rows {
        return;
    }
    let row_start = row * params.width;
    let first = input.data[row_start];
    var all_equal = true;
    var mean = 0.0;
    for (var column = 0u; column < params.width; column = column + 1u) {
        let value = input.data[row_start + column];
        mean = mean + value;
        if value != first {
            all_equal = false;
        }
    }
    if all_equal {
        for (var column = 0u; column < params.width; column = column + 1u) {
            output.data[row_start + column] = bias.data[column];
        }
        return;
    }
    mean = mean / f32(params.width);
    var variance = 0.0;
    for (var column = 0u; column < params.width; column = column + 1u) {
        let centered = input.data[row_start + column] - mean;
        variance = variance + centered * centered;
    }
    variance = variance / f32(params.width);
    let inverse_stddev = 1.0 / sqrt(variance + params.epsilon);
    for (var column = 0u; column < params.width; column = column + 1u) {
        output.data[row_start + column] = (input.data[row_start + column] - mean) * inverse_stddev * weight.data[column] + bias.data[column];
    }
}
"#;

const RMS_NORM_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    rows: u32,
    width: u32,
    epsilon: f32,
    padding: u32,
}
@group(0) @binding(0) var<storage, read> input: F32Buffer;
@group(0) @binding(1) var<storage, read> weight: F32Buffer;
@group(0) @binding(2) var<storage, read_write> output: F32Buffer;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if row >= params.rows {
        return;
    }
    let row_start = row * params.width;
    var mean_square = 0.0;
    for (var column = 0u; column < params.width; column = column + 1u) {
        let value = input.data[row_start + column];
        mean_square = mean_square + value * value;
    }
    mean_square = mean_square / f32(params.width);
    let inverse_rms = 1.0 / sqrt(mean_square + params.epsilon);
    for (var column = 0u; column < params.width; column = column + 1u) {
        output.data[row_start + column] = input.data[row_start + column] * inverse_rms * weight.data[column];
    }
}
"#;

const SILU_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    length: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}
@group(0) @binding(0) var<storage, read> input: F32Buffer;
@group(0) @binding(1) var<storage, read_write> output: F32Buffer;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_stride = select(params.length, params.padding0, params.padding0 != 0u);
    let index = global_id.x + global_id.y * row_stride;
    if index >= params.length {
        return;
    }
    let value = input.data[index];
    if value < -80.0 {
        output.data[index] = -0.0;
    } else if value > 80.0 {
        output.data[index] = value;
    } else if value < 0.0 {
        let exponential = exp(value);
        output.data[index] = value * exponential / (1.0 + exponential);
    } else {
        output.data[index] = value / (1.0 + exp(-value));
    }
}
"#;

const GELU_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    length: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}
@group(0) @binding(0) var<storage, read> input: F32Buffer;
@group(0) @binding(1) var<storage, read_write> output: F32Buffer;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_stride = select(params.length, params.padding0, params.padding0 != 0u);
    let index = global_id.x + global_id.y * row_stride;
    if index >= params.length {
        return;
    }
    let value = input.data[index];
    let cubic = value * value * value;
    let argument = 0.7978846 * (value + 0.044715 * cubic);
    if argument < -10.0 {
        output.data[index] = -0.0;
    } else if argument > 10.0 {
        output.data[index] = value;
    } else {
        output.data[index] = 0.5 * value * (1.0 + tanh(argument));
    }
}
"#;

const GELU_ERF_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    length: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}
@group(0) @binding(0) var<storage, read> input: F32Buffer;
@group(0) @binding(1) var<storage, read_write> output: F32Buffer;
@group(0) @binding(2) var<uniform> params: Params;
fn erf_approx(value: f32) -> f32 {
    let absolute = abs(value);
    let reciprocal = 1.0 / (1.0 + 0.3275911 * absolute);
    var polynomial = 1.061405429;
    polynomial = -1.453152027 + reciprocal * polynomial;
    polynomial = 1.421413741 + reciprocal * polynomial;
    polynomial = -0.284496736 + reciprocal * polynomial;
    polynomial = 0.254829592 + reciprocal * polynomial;
    let approximation = 1.0 - reciprocal * polynomial * exp(-absolute * absolute);
    return select(approximation, -approximation, value < 0.0);
}
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_stride = select(params.length, params.padding0, params.padding0 != 0u);
    let index = global_id.x + global_id.y * row_stride;
    if index >= params.length {
        return;
    }
    let value = input.data[index];
    let argument = value * 0.7071067811865476;
    let gelu = 0.5 * value * (1.0 + erf_approx(argument));
    output.data[index] = gelu;
}
"#;

const ROPE_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct U32Buffer {
    data: array<u32>,
}
struct Params {
    rows: u32,
    width: u32,
    rotary_dim: u32,
    base: f32,
}
@group(0) @binding(0) var<storage, read> input: F32Buffer;
@group(0) @binding(1) var<storage, read> positions: U32Buffer;
@group(0) @binding(2) var<storage, read_write> output: F32Buffer;
@group(0) @binding(3) var<uniform> params: Params;
fn precise_exp2(value: f32) -> f32 {
    let whole = floor(value);
    let fraction = value - whole;
    var polynomial = 0.0000000070549116;
    polynomial = 0.00000010178086 + fraction * polynomial;
    polynomial = 0.0000013215487 + fraction * polynomial;
    polynomial = 0.000015252734 + fraction * polynomial;
    polynomial = 0.00015403531 + fraction * polynomial;
    polynomial = 0.0013333558 + fraction * polynomial;
    polynomial = 0.009618129 + fraction * polynomial;
    polynomial = 0.05550411 + fraction * polynomial;
    polynomial = 0.2402265 + fraction * polynomial;
    polynomial = 0.6931472 + fraction * polynomial;
    polynomial = 1.0 + fraction * polynomial;
    let scale_bits = u32(i32(whole) + 127) << 23u;
    return bitcast<f32>(scale_bits) * polynomial;
}
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let half = params.rotary_dim / 2u;
    let linear_pair = global_id.x;
    let row = linear_pair / half;
    if row >= params.rows {
        return;
    }
    let pair = linear_pair % half;
    let row_start = row * params.width;
    let first_index = row_start + pair;
    let second_index = row_start + half + pair;
    let exponent = -(2.0 * f32(pair) / f32(params.rotary_dim));
    let inverse_frequency = precise_exp2(log2(params.base) * exponent);
    let angle = f32(positions.data[row]) * inverse_frequency;
    let turns = round(angle * 0.15915494);
    let reduced_high = fma(-turns, 6.28125, angle);
    let reduced_angle = fma(-turns, 0.0019353072, reduced_high);
    let sine = sin(reduced_angle);
    let cosine = cos(reduced_angle);
    let first = input.data[first_index];
    let second = input.data[second_index];
    output.data[first_index] = first * cosine - second * sine;
    output.data[second_index] = second * cosine + first * sine;
}
"#;

const VISION_ATTENTION_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct U32Buffer {
    data: array<u32>,
}
struct Params {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    segments: u32,
}
@group(0) @binding(0) var<storage, read> query: F32Buffer;
@group(0) @binding(1) var<storage, read> key: F32Buffer;
@group(0) @binding(2) var<storage, read> value: F32Buffer;
@group(0) @binding(3) var<storage, read> cu_seqlens: U32Buffer;
@group(0) @binding(4) var<storage, read_write> output: F32Buffer;
@group(0) @binding(5) var<uniform> params: Params;
const QUERY_TILE: u32 = 128u;
const KEY_STEP: u32 = 16u;
const MAX_HEAD_VECTORS: u32 = 18u;
const WORKGROUP_SIZE: u32 = 128u;
const MIN_F32: f32 = -3.402823466e+38;
var<workgroup> key_cache: array<vec4<f32>, 288>;
var<workgroup> value_cache: array<vec4<f32>, 288>;
@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
) {
    let local_index = local_invocation_id.x;
    let head = workgroup_id.y;
    let query_token = workgroup_id.x * QUERY_TILE + local_index;
    let query_valid = query_token < params.tokens && head < params.heads;
    var segment_start = 0u;
    var segment_end = params.tokens;
    if query_valid {
        for (var segment = 0u; segment < params.segments; segment = segment + 1u) {
            let candidate_end = cu_seqlens.data[segment + 1u];
            if query_token < candidate_end {
                segment_start = cu_seqlens.data[segment];
                segment_end = candidate_end;
                break;
            }
        }
    }

    var query_vectors: array<vec4<f32>, 18>;
    for (var vector_index = 0u; vector_index < MAX_HEAD_VECTORS; vector_index = vector_index + 1u) {
        let dimension_base = vector_index * 4u;
        query_vectors[vector_index] = vec4<f32>(0.0);
        if query_valid {
            let query_source_base =
                (query_token * params.heads + head) * params.head_dim + dimension_base;
            if dimension_base + 0u < params.head_dim {
                query_vectors[vector_index].x = query.data[query_source_base + 0u];
            }
            if dimension_base + 1u < params.head_dim {
                query_vectors[vector_index].y = query.data[query_source_base + 1u];
            }
            if dimension_base + 2u < params.head_dim {
                query_vectors[vector_index].z = query.data[query_source_base + 2u];
            }
            if dimension_base + 3u < params.head_dim {
                query_vectors[vector_index].w = query.data[query_source_base + 3u];
            }
        }
    }

    var attention_output: array<vec4<f32>, 18>;
    var running_maximum = MIN_F32;
    var running_denominator = 0.0;
    let attention_scale = inverseSqrt(f32(params.head_dim));

    for (var key_start = 0u; key_start < params.tokens; key_start = key_start + KEY_STEP) {
        for (var cache_index = local_index; cache_index < KEY_STEP * MAX_HEAD_VECTORS; cache_index = cache_index + WORKGROUP_SIZE) {
            let key_slot = cache_index / MAX_HEAD_VECTORS;
            let vector_index = cache_index % MAX_HEAD_VECTORS;
            let key_token = key_start + key_slot;
            let dimension_base = vector_index * 4u;
            var loaded_key = vec4<f32>(0.0);
            var loaded_value = vec4<f32>(0.0);
            if key_token < params.tokens && head < params.heads {
                let key_source_base =
                    (key_token * params.heads + head) * params.head_dim + dimension_base;
                if dimension_base + 0u < params.head_dim {
                    loaded_key.x = key.data[key_source_base + 0u];
                    loaded_value.x = value.data[key_source_base + 0u];
                }
                if dimension_base + 1u < params.head_dim {
                    loaded_key.y = key.data[key_source_base + 1u];
                    loaded_value.y = value.data[key_source_base + 1u];
                }
                if dimension_base + 2u < params.head_dim {
                    loaded_key.z = key.data[key_source_base + 2u];
                    loaded_value.z = value.data[key_source_base + 2u];
                }
                if dimension_base + 3u < params.head_dim {
                    loaded_key.w = key.data[key_source_base + 3u];
                    loaded_value.w = value.data[key_source_base + 3u];
                }
            }
            key_cache[cache_index] = loaded_key;
            value_cache[cache_index] = loaded_value;
        }
        workgroupBarrier();

        var scores: array<f32, 16>;
        var block_maximum = MIN_F32;
        for (var key_slot = 0u; key_slot < KEY_STEP; key_slot = key_slot + 1u) {
            let key_token = key_start + key_slot;
            let valid_key = query_valid && key_token < params.tokens
                && key_token >= segment_start && key_token < segment_end;
            scores[key_slot] = MIN_F32;
            if valid_key {
                scores[key_slot] = 0.0;
                for (var vector_index = 0u; vector_index < MAX_HEAD_VECTORS; vector_index = vector_index + 1u) {
                    scores[key_slot] = scores[key_slot] + dot(query_vectors[vector_index], key_cache[key_slot * MAX_HEAD_VECTORS + vector_index]);
                }
                scores[key_slot] = scores[key_slot] * attention_scale;
                block_maximum = max(block_maximum, scores[key_slot]);
            }
        }

        if block_maximum > MIN_F32 {
            let next_maximum = max(running_maximum, block_maximum);
            var previous_scale = 0.0;
            if running_denominator > 0.0 {
                previous_scale = exp(running_maximum - next_maximum);
            }
            var block_denominator = 0.0;
            for (var key_slot = 0u; key_slot < KEY_STEP; key_slot = key_slot + 1u) {
                let key_token = key_start + key_slot;
                let valid_key = query_valid && key_token < params.tokens
                    && key_token >= segment_start && key_token < segment_end;
                if valid_key {
                    scores[key_slot] = exp(scores[key_slot] - next_maximum);
                } else {
                    scores[key_slot] = 0.0;
                }
                block_denominator = block_denominator + scores[key_slot];
            }
            for (var vector_index = 0u; vector_index < MAX_HEAD_VECTORS; vector_index = vector_index + 1u) {
                var block_weighted = vec4<f32>(0.0);
                for (var key_slot = 0u; key_slot < KEY_STEP; key_slot = key_slot + 1u) {
                    block_weighted = block_weighted
                        + scores[key_slot]
                            * value_cache[key_slot * MAX_HEAD_VECTORS + vector_index];
                }
                attention_output[vector_index] =
                    attention_output[vector_index] * previous_scale + block_weighted;
            }
            running_denominator =
                running_denominator * previous_scale + block_denominator;
            running_maximum = next_maximum;
        }
        workgroupBarrier();
    }

    if query_valid && running_denominator > 0.0 {
        let query_base = (query_token * params.heads + head) * params.head_dim;
        for (var vector_index = 0u; vector_index < MAX_HEAD_VECTORS; vector_index = vector_index + 1u) {
            let dimension_base = vector_index * 4u;
            let normalized = attention_output[vector_index] / running_denominator;
            if dimension_base + 0u < params.head_dim {
                output.data[query_base + dimension_base + 0u] = normalized.x;
            }
            if dimension_base + 1u < params.head_dim {
                output.data[query_base + dimension_base + 1u] = normalized.y;
            }
            if dimension_base + 2u < params.head_dim {
                output.data[query_base + dimension_base + 2u] = normalized.z;
            }
            if dimension_base + 3u < params.head_dim {
                output.data[query_base + dimension_base + 3u] = normalized.w;
            }
        }
    }
}
"#;

const DECODER_KV_APPEND_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    prefix_tokens: u32,
    key_value_heads: u32,
    head_dim: u32,
    cache_capacity: u32,
}
@group(0) @binding(0) var<storage, read> appended_key: F32Buffer;
@group(0) @binding(1) var<storage, read> appended_value: F32Buffer;
@group(0) @binding(2) var<storage, read_write> key_cache: F32Buffer;
@group(0) @binding(3) var<storage, read_write> value_cache: F32Buffer;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    let key_value_width = params.key_value_heads * params.head_dim;
    if linear >= key_value_width || params.prefix_tokens >= params.cache_capacity {
        return;
    }
    let cache_index = params.prefix_tokens * key_value_width + linear;
    key_cache.data[cache_index] = appended_key.data[linear];
    value_cache.data[cache_index] = appended_value.data[linear];
}
"#;

const DECODER_GQA_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    cache_tokens: u32,
    query_heads: u32,
    key_value_heads: u32,
    head_dim: u32,
}
@group(0) @binding(0) var<storage, read> query: F32Buffer;
@group(0) @binding(1) var<storage, read> key_cache: F32Buffer;
@group(0) @binding(2) var<storage, read> value_cache: F32Buffer;
@group(0) @binding(3) var<storage, read_write> output: F32Buffer;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let query_head = global_id.x;
    if query_head >= params.query_heads {
        return;
    }

    let query_heads_per_kv = params.query_heads / params.key_value_heads;
    let key_value_head = query_head / query_heads_per_kv;
    let query_base = query_head * params.head_dim;
    let output_base = query_base;
    let attention_scale = inverseSqrt(f32(params.head_dim));
    var weighted: array<f32, 128>;
    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
        weighted[dimension] = 0.0;
    }
    var maximum = 0.0;
    var denominator = 0.0;
    var first_key = true;

    for (var key_token = 0u; key_token < params.cache_tokens; key_token = key_token + 1u) {
        let key_base =
            (key_token * params.key_value_heads + key_value_head) * params.head_dim;
        var score = 0.0;
        for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
            score = score
                + query.data[query_base + dimension] * key_cache.data[key_base + dimension];
        }
        score = score * attention_scale;

        if first_key {
            maximum = score;
            denominator = 1.0;
            for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
                weighted[dimension] = value_cache.data[key_base + dimension];
            }
            first_key = false;
        } else {
            let next_maximum = max(maximum, score);
            let previous_weight = exp(maximum - next_maximum);
            let current_weight = exp(score - next_maximum);
            denominator = denominator * previous_weight + current_weight;
            for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
                weighted[dimension] = weighted[dimension] * previous_weight
                    + current_weight * value_cache.data[key_base + dimension];
            }
            maximum = next_maximum;
        }
    }

    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
        output.data[output_base + dimension] = weighted[dimension] / denominator;
    }
}
"#;

const DECODER_GQA_SPLIT_PARTIAL_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    cache_tokens: u32,
    chunk_count: u32,
    padding0: u32,
    padding1: u32,
}
const HEAD_DIM: u32 = 128u;
const QUERY_HEADS: u32 = 16u;
const KEY_VALUE_HEADS: u32 = 2u;
const CHUNK_SIZE: u32 = 32u;
const PARTIAL_STRIDE: u32 = 192u;
@group(0) @binding(0) var<storage, read> query: F32Buffer;
@group(0) @binding(1) var<storage, read> key_cache: F32Buffer;
@group(0) @binding(2) var<storage, read> value_cache: F32Buffer;
@group(0) @binding(3) var<storage, read_write> partials: F32Buffer;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> scores: array<f32, 32>;
var<workgroup> maxima: array<f32, 32>;
var<workgroup> weights: array<f32, 32>;
var<workgroup> sums: array<f32, 32>;
@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let query_head = workgroup_id.x / params.chunk_count;
    let chunk = workgroup_id.x % params.chunk_count;
    let query_heads_per_kv = QUERY_HEADS / KEY_VALUE_HEADS;
    let key_value_head = query_head / query_heads_per_kv;
    let query_base = query_head * HEAD_DIM;
    let attention_scale = inverseSqrt(f32(HEAD_DIM));
    let key_token = chunk * CHUNK_SIZE + local_id.x;
    if local_id.x < CHUNK_SIZE && key_token < params.cache_tokens {
        let key_base =
            (key_token * KEY_VALUE_HEADS + key_value_head) * HEAD_DIM;
        var score = 0.0;
        for (var dimension = 0u; dimension < HEAD_DIM; dimension = dimension + 1u) {
            score = score
                + query.data[query_base + dimension] * key_cache.data[key_base + dimension];
        }
        score = score * attention_scale;
        scores[local_id.x] = score;
        maxima[local_id.x] = score;
    } else {
        scores[local_id.x] = -3.402823466e+38;
        maxima[local_id.x] = -3.402823466e+38;
    }
    workgroupBarrier();
    // The chunk maximum reuses the same ascending-pair tree, but reduces
    // the separate maxima copy: the per-key scores must survive intact for
    // the exp(score - chunk_max) weights below.
    for (var stride = CHUNK_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if local_id.x < stride {
            maxima[local_id.x] = max(maxima[local_id.x], maxima[local_id.x + stride]);
        }
        workgroupBarrier();
    }
    let chunk_max = maxima[0];
    if local_id.x < CHUNK_SIZE {
        let weight = exp(scores[local_id.x] - chunk_max);
        weights[local_id.x] = weight;
        sums[local_id.x] = weight;
    }
    workgroupBarrier();
    // The chunk sum reuses the same ascending-pair tree, but reduces the
    // separate sums copy: the per-key weights must survive intact for the
    // weighted-V accumulation below.
    for (var stride = CHUNK_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if local_id.x < stride {
            sums[local_id.x] = sums[local_id.x] + sums[local_id.x + stride];
        }
        workgroupBarrier();
    }
    let chunk_sum = sums[0];
    let partial_base = (query_head * params.chunk_count + chunk) * PARTIAL_STRIDE;
    for (var dim_offset = 0u; dim_offset < 2u; dim_offset = dim_offset + 1u) {
        let dimension = local_id.x * 2u + dim_offset;
        var weighted = 0.0;
        for (var key_in_chunk = 0u; key_in_chunk < CHUNK_SIZE; key_in_chunk = key_in_chunk + 1u) {
            let weight = weights[key_in_chunk];
            let key_token = chunk * CHUNK_SIZE + key_in_chunk;
            let key_base =
                (key_token * KEY_VALUE_HEADS + key_value_head) * HEAD_DIM;
            // Out-of-range keys contribute exp(score - chunk_max) = 0.0, but
            // 0.0 * NaN is NaN, so the tail V read is masked explicitly: the
            // physical cache rows past cache_tokens may hold arbitrary
            // non-finite poison that must never reach the accumulator.
            let masked_value = select(
                0.0,
                value_cache.data[key_base + dimension],
                key_token < params.cache_tokens,
            );
            weighted = weighted
                    + weight * masked_value;
        }
        partials.data[partial_base + dimension] = weighted;
    }
    if local_id.x == 0u {
        partials.data[partial_base + HEAD_DIM] = chunk_max;
        partials.data[partial_base + HEAD_DIM + 1u] = chunk_sum;
    }
}
"#;

const DECODER_GQA_SPLIT_MERGE_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    cache_tokens: u32,
    chunk_count: u32,
    padding0: u32,
    padding1: u32,
}
const HEAD_DIM: u32 = 128u;
const QUERY_HEADS: u32 = 16u;
const PARTIAL_STRIDE: u32 = 192u;
@group(0) @binding(0) var<storage, read> partials: F32Buffer;
@group(0) @binding(1) var<storage, read> key_cache: F32Buffer;
@group(0) @binding(2) var<storage, read> value_cache: F32Buffer;
@group(0) @binding(3) var<storage, read_write> output: F32Buffer;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    if linear >= QUERY_HEADS * HEAD_DIM {
        return;
    }
    let query_head = linear / HEAD_DIM;
    let dimension = linear % HEAD_DIM;

    var maximum = 0.0;
    var denominator = 0.0;
    var weighted = 0.0;
    var first_chunk = true;
    for (var chunk = 0u; chunk < params.chunk_count; chunk = chunk + 1u) {
        let partial_base = (query_head * params.chunk_count + chunk) * PARTIAL_STRIDE;
        let chunk_max = partials.data[partial_base + HEAD_DIM];
        let chunk_sum = partials.data[partial_base + HEAD_DIM + 1u];
        let chunk_value = partials.data[partial_base + dimension];
        if first_chunk {
            maximum = chunk_max;
            denominator = chunk_sum;
            weighted = chunk_value;
            first_chunk = false;
        } else {
            let next_maximum = max(maximum, chunk_max);
            let previous_weight = exp(maximum - next_maximum);
            let current_weight = exp(chunk_max - next_maximum);
            denominator = denominator * previous_weight + current_weight * chunk_sum;
            weighted = weighted * previous_weight + current_weight * chunk_value;
            maximum = next_maximum;
        }
    }
    output.data[linear] = weighted / denominator;
}
"#;

const DECODER_MROPE_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    position: u32,
    rope_capacity: u32,
    padding0: u32,
    padding1: u32,
}
const HEAD_DIM: u32 = 128u;
const HALF_DIM: u32 = 64u;
const FIRST_SECTION_END: u32 = 16u;
const SECOND_SECTION_END: u32 = 40u;
const QUERY_WIDTH: u32 = 2048u;
const KEY_WIDTH: u32 = 256u;
const TOTAL_WIDTH: u32 = 2304u;
@group(0) @binding(0) var<storage, read> query: F32Buffer;
@group(0) @binding(1) var<storage, read> key: F32Buffer;
@group(0) @binding(2) var<storage, read> rope_cos: F32Buffer;
@group(0) @binding(3) var<storage, read> rope_sin: F32Buffer;
@group(0) @binding(4) var<storage, read_write> output_query: F32Buffer;
@group(0) @binding(5) var<storage, read_write> output_key: F32Buffer;
@group(0) @binding(6) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    if linear >= TOTAL_WIDTH {
        return;
    }
    let dim = linear % HEAD_DIM;
    let local = select(dim, dim - HALF_DIM, dim >= HALF_DIM);
    let axis = select(select(0u, 1u, local >= FIRST_SECTION_END), 2u, local >= SECOND_SECTION_END);
    let table_index = (axis * params.rope_capacity + params.position) * HEAD_DIM + dim;
    let partner = select(linear + HALF_DIM, linear - HALF_DIM, dim >= HALF_DIM);
    let sign = select(-1.0, 1.0, dim >= HALF_DIM);
    if linear < QUERY_WIDTH {
        let value = query.data[linear];
        let rotated = value * rope_cos.data[table_index]
            + sign * query.data[partner] * rope_sin.data[table_index];
        output_query.data[linear] = rotated;
    } else {
        let key_index = linear - QUERY_WIDTH;
        let key_partner = partner - QUERY_WIDTH;
        let value = key.data[key_index];
        let rotated = value * rope_cos.data[table_index]
            + sign * key.data[key_partner] * rope_sin.data[table_index];
        output_key.data[key_index] = rotated;
    }
}
"#;

const DECODER_SWIGLU_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    length: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}
@group(0) @binding(0) var<storage, read> gate: F32Buffer;
@group(0) @binding(1) var<storage, read> up: F32Buffer;
@group(0) @binding(2) var<storage, read_write> output: F32Buffer;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= params.length {
        return;
    }
    let gate_value = gate.data[index];
    let up_value = up.data[index];
    let activated = gate_value / (1.0 + exp(-gate_value));
    output.data[index] = activated * up_value;
}
"#;

const DECODER_PREFILL_GQA_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    tokens: u32,
    query_heads: u32,
    key_value_heads: u32,
    head_dim: u32,
}
@group(0) @binding(0) var<storage, read> query: F32Buffer;
@group(0) @binding(1) var<storage, read> key_cache: F32Buffer;
@group(0) @binding(2) var<storage, read> value_cache: F32Buffer;
@group(0) @binding(3) var<storage, read_write> output: F32Buffer;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    let work_items = params.tokens * params.query_heads;
    if linear >= work_items {
        return;
    }

    let query_token = linear / params.query_heads;
    let query_head = linear % params.query_heads;
    let query_heads_per_kv = params.query_heads / params.key_value_heads;
    let key_value_head = query_head / query_heads_per_kv;
    let query_base = (query_token * params.query_heads + query_head) * params.head_dim;
    let attention_scale = inverseSqrt(f32(params.head_dim));
    var weighted: array<f32, 128>;
    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
        weighted[dimension] = 0.0;
    }
    var maximum = 0.0;
    var denominator = 0.0;
    var first_key = true;

    for (var key_token = 0u; key_token <= query_token; key_token = key_token + 1u) {
        let key_base =
            (key_token * params.key_value_heads + key_value_head) * params.head_dim;
        var score = 0.0;
        for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
            score = score
                + query.data[query_base + dimension] * key_cache.data[key_base + dimension];
        }
        score = score * attention_scale;

        if first_key {
            maximum = score;
            denominator = 1.0;
            for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
                weighted[dimension] = value_cache.data[key_base + dimension];
            }
            first_key = false;
        } else {
            let next_maximum = max(maximum, score);
            let previous_weight = exp(maximum - next_maximum);
            let current_weight = exp(score - next_maximum);
            denominator = denominator * previous_weight + current_weight;
            for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
                weighted[dimension] = weighted[dimension] * previous_weight
                    + current_weight * value_cache.data[key_base + dimension];
            }
            maximum = next_maximum;
        }
    }

    for (var dimension = 0u; dimension < params.head_dim; dimension = dimension + 1u) {
        output.data[query_base + dimension] = weighted[dimension] / denominator;
    }
}
"#;

const DECODER_PREFILL_MROPE_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    tokens: u32,
    rope_capacity: u32,
    padding0: u32,
    padding1: u32,
}
const HEAD_DIM: u32 = 128u;
const HALF_DIM: u32 = 64u;
const FIRST_SECTION_END: u32 = 16u;
const SECOND_SECTION_END: u32 = 40u;
const QUERY_WIDTH: u32 = 2048u;
const KEY_WIDTH: u32 = 256u;
const TOTAL_WIDTH: u32 = 2304u;
@group(0) @binding(0) var<storage, read> query: F32Buffer;
@group(0) @binding(1) var<storage, read> key: F32Buffer;
@group(0) @binding(2) var<storage, read> rope_cos: F32Buffer;
@group(0) @binding(3) var<storage, read> rope_sin: F32Buffer;
@group(0) @binding(4) var<storage, read_write> output_query: F32Buffer;
@group(0) @binding(5) var<storage, read_write> output_key: F32Buffer;
@group(0) @binding(6) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    if linear >= params.tokens * TOTAL_WIDTH {
        return;
    }
    let token = linear / TOTAL_WIDTH;
    let within = linear % TOTAL_WIDTH;
    let dim = within % HEAD_DIM;
    let local = select(dim, dim - HALF_DIM, dim >= HALF_DIM);
    let axis = select(select(0u, 1u, local >= FIRST_SECTION_END), 2u, local >= SECOND_SECTION_END);
    let table_index = (axis * params.rope_capacity + token) * HEAD_DIM + dim;
    let partner = select(within + HALF_DIM, within - HALF_DIM, dim >= HALF_DIM);
    let sign = select(-1.0, 1.0, dim >= HALF_DIM);
    if within < QUERY_WIDTH {
        let query_base = token * QUERY_WIDTH;
        let value = query.data[query_base + within];
        let rotated = value * rope_cos.data[table_index]
            + sign * query.data[query_base + partner] * rope_sin.data[table_index];
        output_query.data[query_base + within] = rotated;
    } else {
        let key_base = token * KEY_WIDTH;
        let key_index = within - QUERY_WIDTH;
        let key_partner = partner - QUERY_WIDTH;
        let value = key.data[key_base + key_index];
        let rotated = value * rope_cos.data[table_index]
            + sign * key.data[key_base + key_partner] * rope_sin.data[table_index];
        output_key.data[key_base + key_index] = rotated;
    }
}
"#;

const DECODER_KV_APPEND_RANGE_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    tokens: u32,
    cache_capacity: u32,
    padding0: u32,
    padding1: u32,
}
const KEY_VALUE_WIDTH: u32 = 256u;
@group(0) @binding(0) var<storage, read> appended_key: F32Buffer;
@group(0) @binding(1) var<storage, read> appended_value: F32Buffer;
@group(0) @binding(2) var<storage, read_write> key_cache: F32Buffer;
@group(0) @binding(3) var<storage, read_write> value_cache: F32Buffer;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    if linear >= params.tokens * KEY_VALUE_WIDTH || params.tokens > params.cache_capacity {
        return;
    }
    let token = linear / KEY_VALUE_WIDTH;
    let within = linear % KEY_VALUE_WIDTH;
    let cache_index = token * KEY_VALUE_WIDTH + within;
    key_cache.data[cache_index] = appended_key.data[linear];
    value_cache.data[cache_index] = appended_value.data[linear];
}
"#;

const VISION_PATCH_PROJECTION_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    patch_count: u32,
    input_width: u32,
    output_width: u32,
    padding: u32,
}
@group(0) @binding(0) var<storage, read> input: F32Buffer;
@group(0) @binding(1) var<storage, read> weight: F32Buffer;
@group(0) @binding(2) var<storage, read> bias: F32Buffer;
@group(0) @binding(3) var<storage, read_write> output: F32Buffer;
@group(0) @binding(4) var<uniform> params: Params;
const PROJECTION_TILE_ROWS: u32 = 32u;
const PROJECTION_TILE_COLUMNS: u32 = 32u;
const PROJECTION_TILE_DEPTH: u32 = 32u;
const PROJECTION_ROWS_PER_LANE: u32 = 4u;
const PROJECTION_COLUMNS_PER_LANE: u32 = 4u;
const PROJECTION_WORKGROUP_SIZE: u32 = 64u;
var<workgroup> input_tile: array<f32, 1024>;
var<workgroup> weight_tile: array<f32, 1024>;
@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let local_index = local_id.y * 8u + local_id.x;
    let local_row_base = local_id.y * PROJECTION_ROWS_PER_LANE;
    let local_column_base = local_id.x * PROJECTION_COLUMNS_PER_LANE;
    var initial_bias = vec4<f32>(0.0);
    for (var output_offset = 0u; output_offset < PROJECTION_COLUMNS_PER_LANE; output_offset = output_offset + 1u) {
        let output_column =
            workgroup_id.x * PROJECTION_TILE_COLUMNS + local_column_base + output_offset;
        if output_column < params.output_width {
            initial_bias[output_offset] = bias.data[output_column];
        }
    }
    var accumulator0 = initial_bias;
    var accumulator1 = initial_bias;
    var accumulator2 = initial_bias;
    var accumulator3 = initial_bias;

    for (var depth_base = 0u; depth_base < params.input_width; depth_base = depth_base + PROJECTION_TILE_DEPTH) {
        for (var load_index = local_index; load_index < PROJECTION_TILE_ROWS * PROJECTION_TILE_DEPTH; load_index = load_index + PROJECTION_WORKGROUP_SIZE) {
            let tile_row = load_index / PROJECTION_TILE_DEPTH;
            let tile_depth = load_index % PROJECTION_TILE_DEPTH;
            let input_row = workgroup_id.y * PROJECTION_TILE_ROWS + tile_row;
            let input_depth = depth_base + tile_depth;
            var loaded_input = 0.0;
            if input_row < params.patch_count && input_depth < params.input_width {
                loaded_input = input.data[input_row * params.input_width + input_depth];
            }
            input_tile[load_index] = loaded_input;

            let output_column = workgroup_id.x * PROJECTION_TILE_COLUMNS + tile_row;
            var loaded_weight = 0.0;
            if output_column < params.output_width && input_depth < params.input_width {
                loaded_weight =
                    weight.data[output_column * params.input_width + input_depth];
            }
            weight_tile[tile_depth * PROJECTION_TILE_COLUMNS + tile_row] = loaded_weight;
        }
        workgroupBarrier();

        for (var depth = 0u; depth < PROJECTION_TILE_DEPTH; depth = depth + 1u) {
            let coefficients = vec4<f32>(
                weight_tile[depth * PROJECTION_TILE_COLUMNS + local_column_base + 0u],
                weight_tile[depth * PROJECTION_TILE_COLUMNS + local_column_base + 1u],
                weight_tile[depth * PROJECTION_TILE_COLUMNS + local_column_base + 2u],
                weight_tile[depth * PROJECTION_TILE_COLUMNS + local_column_base + 3u],
            );
            accumulator0 = fma(
                vec4<f32>(input_tile[(local_row_base + 0u) * PROJECTION_TILE_DEPTH + depth]),
                coefficients,
                accumulator0,
            );
            accumulator1 = fma(
                vec4<f32>(input_tile[(local_row_base + 1u) * PROJECTION_TILE_DEPTH + depth]),
                coefficients,
                accumulator1,
            );
            accumulator2 = fma(
                vec4<f32>(input_tile[(local_row_base + 2u) * PROJECTION_TILE_DEPTH + depth]),
                coefficients,
                accumulator2,
            );
            accumulator3 = fma(
                vec4<f32>(input_tile[(local_row_base + 3u) * PROJECTION_TILE_DEPTH + depth]),
                coefficients,
                accumulator3,
            );
        }
        workgroupBarrier();
    }

    for (var output_row_offset = 0u; output_row_offset < PROJECTION_ROWS_PER_LANE; output_row_offset = output_row_offset + 1u) {
        let output_row =
            workgroup_id.y * PROJECTION_TILE_ROWS + local_row_base + output_row_offset;
        var accumulated = accumulator0;
        if output_row_offset == 1u {
            accumulated = accumulator1;
        } else if output_row_offset == 2u {
            accumulated = accumulator2;
        } else if output_row_offset == 3u {
            accumulated = accumulator3;
        }
        if output_row < params.patch_count {
            for (var output_offset = 0u; output_offset < PROJECTION_COLUMNS_PER_LANE; output_offset = output_offset + 1u) {
                let output_column = workgroup_id.x * PROJECTION_TILE_COLUMNS
                    + local_column_base + output_offset;
                if output_column < params.output_width {
                    output.data[output_row * params.output_width + output_column] =
                        accumulated[output_offset];
                }
            }
        }
    }
}
"#;

const PROJECTOR_MERGE_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct U32Buffer {
    data: array<u32>,
}
struct Params {
    output_tokens: u32,
    hidden_size: u32,
    length: u32,
    row_stride: u32,
}
@group(0) @binding(0) var<storage, read> input: F32Buffer;
@group(0) @binding(1) var<storage, read> source_token_indices: U32Buffer;
@group(0) @binding(2) var<storage, read_write> output: F32Buffer;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dispatch_row_stride = select(params.length, params.row_stride, params.row_stride != 0u);
    let index = global_id.x + global_id.y * dispatch_row_stride;
    if index >= params.length {
        return;
    }
    let merged_width = params.hidden_size * 4u;
    let output_token = index / merged_width;
    if output_token >= params.output_tokens {
        return;
    }
    let column = index % merged_width;
    let source_patch = column / params.hidden_size;
    let channel = column % params.hidden_size;
    let source_token = source_token_indices.data[output_token * 4u + source_patch];
    output.data[index] = input.data[source_token * params.hidden_size + channel];
}
"#;

const PROJECTOR_MERGE_F16_SOURCE: &str = r#"enable f16;
struct F16Vec4Buffer {
    data: array<vec4<f16>>,
}
struct U32Buffer {
    data: array<u32>,
}
struct Params {
    output_tokens: u32,
    hidden_size: u32,
    length: u32,
    row_stride: u32,
}
@group(0) @binding(0) var<storage, read> input: F16Vec4Buffer;
@group(0) @binding(1) var<storage, read> source_token_indices: U32Buffer;
@group(0) @binding(2) var<storage, read_write> output: F16Vec4Buffer;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vector_length = params.length / 4u;
    let dispatch_row_stride =
        select(vector_length, params.row_stride, params.row_stride != 0u);
    let vector_index = global_id.x + global_id.y * dispatch_row_stride;
    if vector_index >= vector_length {
        return;
    }
    let hidden_vectors = params.hidden_size / 4u;
    let merged_width_vectors = hidden_vectors * 4u;
    let output_token = vector_index / merged_width_vectors;
    if output_token >= params.output_tokens {
        return;
    }
    let column_vector = vector_index % merged_width_vectors;
    let source_patch = column_vector / hidden_vectors;
    let channel_vector = column_vector % hidden_vectors;
    let source_token =
        source_token_indices.data[output_token * 4u + source_patch];
    output.data[vector_index] =
        input.data[source_token * hidden_vectors + channel_vector];
}
"#;

const ADD_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    length: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}
@group(0) @binding(0) var<storage, read> left: F32Buffer;
@group(0) @binding(1) var<storage, read> right: F32Buffer;
@group(0) @binding(2) var<storage, read_write> output: F32Buffer;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_stride = select(params.length, params.padding0, params.padding0 != 0u);
    let index = global_id.x + global_id.y * row_stride;
    if index >= params.length {
        return;
    }
    output.data[index] = left.data[index] + right.data[index];
}
"#;

const VISION_QKV_FUSED_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    tokens: u32,
    input_width: u32,
    output_width: u32,
    plane_stride_elements: u32,
}
@group(0) @binding(0) var<storage, read> input: F32Buffer;
@group(0) @binding(1) var<storage, read> query_weight: F32Buffer;
@group(0) @binding(2) var<storage, read> query_bias: F32Buffer;
@group(0) @binding(3) var<storage, read> key_weight: F32Buffer;
@group(0) @binding(4) var<storage, read> key_bias: F32Buffer;
@group(0) @binding(5) var<storage, read> value_weight: F32Buffer;
@group(0) @binding(6) var<storage, read> value_bias: F32Buffer;
@group(0) @binding(7) var<storage, read_write> output: F32Buffer;
@group(0) @binding(8) var<uniform> params: Params;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var output_channel = 0u;
    output_channel = global_id.x;
    var token = 0u;
    token = global_id.y;
    var projection = 0u;
    projection = global_id.z;
    if token >= params.tokens || output_channel >= params.output_width || projection >= 3u {
        return;
    }

    var accumulator = value_bias.data[output_channel];
    if projection == 0u {
        accumulator = query_bias.data[output_channel];
    } else if projection == 1u {
        accumulator = key_bias.data[output_channel];
    }
    for (var depth = 0u; depth < params.input_width; depth = depth + 1u) {
        var coefficient = value_weight.data[output_channel * params.input_width + depth];
        if projection == 0u {
            coefficient = query_weight.data[output_channel * params.input_width + depth];
        } else if projection == 1u {
            coefficient = key_weight.data[output_channel * params.input_width + depth];
        }
        accumulator = accumulator + input.data[token * params.input_width + depth] * coefficient;
    }
    let output_index = projection * params.plane_stride_elements + token * params.output_width + output_channel;
    output.data[output_index] = accumulator;
}
"#;

const VISION_ROPE_2D_SOURCE: &str = r#"struct F32Buffer {
    data: array<f32>,
}
struct Params {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    padding: u32,
}
@group(0) @binding(0) var<storage, read_write> query: F32Buffer;
@group(0) @binding(1) var<storage, read_write> key: F32Buffer;
@group(0) @binding(2) var<storage, read> cos_table: F32Buffer;
@group(0) @binding(3) var<storage, read> sin_table: F32Buffer;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_count = params.head_dim / 2u;
    let work_items = params.tokens * params.heads * pair_count;
    let linear_pair = global_id.x;
    if linear_pair >= work_items {
        return;
    }
    let pair = linear_pair % pair_count;
    let linear_head = linear_pair / pair_count;
    let head = linear_head % params.heads;
    let token = linear_head / params.heads;
    let first_index = (token * params.heads + head) * params.head_dim + pair;
    let second_index = first_index + pair_count;
    let cosine = cos_table.data[token * pair_count + pair];
    let sine = sin_table.data[token * pair_count + pair];
    let query_first = query.data[first_index];
    let query_second = query.data[second_index];
    let key_first = key.data[first_index];
    let key_second = key.data[second_index];
    query.data[first_index] = query_first * cosine - query_second * sine;
    query.data[second_index] = query_second * cosine + query_first * sine;
    key.data[first_index] = key_first * cosine - key_second * sine;
    key.data[second_index] = key_second * cosine + key_first * sine;
}
"#;

const RMS_NORM_F16_WEIGHTS_SOURCE: &str = r#"enable f16;
struct F32Buffer {
    data: array<f32>,
}
struct F16Buffer {
    data: array<f16>,
}
struct Params {
    rows: u32,
    width: u32,
    epsilon: f32,
    padding: u32,
}
@group(0) @binding(0) var<storage, read> input: F32Buffer;
@group(0) @binding(1) var<storage, read> weight: F16Buffer;
@group(0) @binding(2) var<storage, read_write> output: F32Buffer;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if row >= params.rows {
        return;
    }
    let row_start = row * params.width;
    var mean_square = 0.0;
    for (var column = 0u; column < params.width; column = column + 1u) {
        let value = input.data[row_start + column];
        mean_square = mean_square + value * value;
    }
    mean_square = mean_square / f32(params.width);
    let inverse_rms = 1.0 / sqrt(mean_square + params.epsilon);
    for (var column = 0u; column < params.width; column = column + 1u) {
        output.data[row_start + column] =
            input.data[row_start + column] * inverse_rms * f32(weight.data[column]);
    }
}
"#;

const GEMV_TILED_F16_WEIGHTS_SOURCE: &str = r#"enable f16;
struct F16Vec4Buffer {
    data: array<vec4<f16>>,
}
struct F32Vec4Buffer {
    data: array<vec4<f32>>,
}
struct F32Buffer {
    data: array<f32>,
}
struct Params {
    rows: u32,
    columns: u32,
    padding0: u32,
    padding1: u32,
}
const TILE_ROWS: u32 = 8u;
const THREADS_PER_ROW: u32 = 32u;
const VECTOR_WIDTH: u32 = 4u;
@group(0) @binding(0) var<storage, read> matrix: F16Vec4Buffer;
@group(0) @binding(1) var<storage, read> vector: F32Vec4Buffer;
@group(0) @binding(2) var<storage, read_write> output: F32Buffer;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> shared_vector: array<vec4<f32>, 768>;
var<workgroup> partials: array<f32, 256>;
@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let vector_columns = params.columns / VECTOR_WIDTH;
    for (var staged = local_id.x; staged < vector_columns; staged = staged + 256u) {
        shared_vector[staged] = vector.data[staged];
    }
    workgroupBarrier();

    let row_group = local_id.x / THREADS_PER_ROW;
    let lane = local_id.x % THREADS_PER_ROW;
    let row = workgroup_id.x * TILE_ROWS + row_group;
    var partial = 0.0;
    if row < params.rows {
        let row_base = row * vector_columns;
        for (var column = lane; column < vector_columns; column = column + THREADS_PER_ROW) {
            let products =
                vec4<f32>(matrix.data[row_base + column]) * shared_vector[column];
            partial = partial + products.x;
            partial = partial + products.y;
            partial = partial + products.z;
            partial = partial + products.w;
        }
    }
    partials[local_id.x] = partial;
    workgroupBarrier();

    for (var stride = THREADS_PER_ROW / 2u; stride > 0u; stride = stride >> 1u) {
        if lane < stride {
            partials[local_id.x] = partials[local_id.x] + partials[local_id.x + stride];
        }
        workgroupBarrier();
    }
    if lane == 0u && row < params.rows {
        output.data[row] = partials[local_id.x];
    }
}
"#;

const LINEAR_PROJECTION_F16_WEIGHTS_SOURCE: &str = r#"enable f16;
struct F32Vec4Buffer {
    data: array<vec4<f32>>,
}
struct F16Vec4Buffer {
    data: array<vec4<f16>>,
}
struct F32Buffer {
    data: array<f32>,
}
struct Params {
    patch_count: u32,
    input_width: u32,
    output_width: u32,
    padding: u32,
}
@group(0) @binding(0) var<storage, read> input: F32Vec4Buffer;
@group(0) @binding(1) var<storage, read> weight: F16Vec4Buffer;
@group(0) @binding(2) var<storage, read> bias: F32Vec4Buffer;
@group(0) @binding(3) var<storage, read_write> output: F32Buffer;
@group(0) @binding(4) var<uniform> params: Params;
const PROJECTION_TILE_ROWS: u32 = 32u;
const PROJECTION_TILE_COLUMNS: u32 = 32u;
const PROJECTION_TILE_DEPTH: u32 = 32u;
const PROJECTION_ROWS_PER_LANE: u32 = 4u;
const PROJECTION_COLUMNS_PER_LANE: u32 = 4u;
const PROJECTION_WORKGROUP_SIZE: u32 = 64u;
const PROJECTION_WEIGHT_LAYOUT_INPUT_MAJOR: u32 = 1u;
var<workgroup> input_tile: array<array<vec4<f32>, 8>, 32>;
var<workgroup> weight_tile: array<array<vec4<f32>, 8>, 32>;

fn read_output_major_weight_component(input_depth: u32, output_column: u32) -> f32 {
    let input_width_vec = params.input_width / 4u;
    let packed_index = output_column * input_width_vec + input_depth / 4u;
    let component = input_depth % 4u;
    let packed = weight.data[packed_index];
    if component == 0u { return f32(packed.x); }
    if component == 1u { return f32(packed.y); }
    if component == 2u { return f32(packed.z); }
    return f32(packed.w);
}

fn read_output_major_weight(input_depth: u32, output_column_base: u32) -> vec4<f32> {
    let packed_output_columns = vec4<f32>(
        read_output_major_weight_component(input_depth, output_column_base + 0u),
        read_output_major_weight_component(input_depth, output_column_base + 1u),
        read_output_major_weight_component(input_depth, output_column_base + 2u),
        read_output_major_weight_component(input_depth, output_column_base + 3u),
    );
    return packed_output_columns;
}

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let local_x = local_id.x;
    let local_y = local_id.y;
    let input_width_vec = params.input_width / 4u;
    let output_width_vec = params.output_width / 4u;
    let global_row_start = workgroup_id.y * PROJECTION_TILE_ROWS;
    let global_column_vec = workgroup_id.x * 8u + local_x;
    let global_column_base = global_column_vec * 4u;
    var accumulators: array<vec4<f32>, 4>;
    for (var row_offset = 0u; row_offset < PROJECTION_ROWS_PER_LANE; row_offset = row_offset + 1u) {
        accumulators[row_offset] = vec4<f32>(0.0);
    }

    for (var depth_base = 0u; depth_base < params.input_width; depth_base = depth_base + PROJECTION_TILE_DEPTH) {
        let input_depth_vec = depth_base / 4u + local_x;
        for (var row_offset = 0u; row_offset < PROJECTION_ROWS_PER_LANE; row_offset = row_offset + 1u) {
            let tile_row = local_y * PROJECTION_ROWS_PER_LANE + row_offset;
            let global_row = global_row_start + tile_row;
            if global_row < params.patch_count && input_depth_vec < input_width_vec {
                input_tile[tile_row][local_x] = input.data[global_row * input_width_vec + input_depth_vec];
            } else {
                input_tile[tile_row][local_x] = vec4<f32>(0.0);
            }
        }

        for (var depth_offset = 0u; depth_offset < PROJECTION_ROWS_PER_LANE; depth_offset = depth_offset + 1u) {
            let tile_depth = local_y * PROJECTION_ROWS_PER_LANE + depth_offset;
            let input_depth = depth_base + tile_depth;
            if input_depth < params.input_width && global_column_vec < output_width_vec {
                if params.padding == PROJECTION_WEIGHT_LAYOUT_INPUT_MAJOR {
                    weight_tile[tile_depth][local_x] = vec4<f32>(weight.data[input_depth * output_width_vec + global_column_vec]);
                } else {
                    weight_tile[tile_depth][local_x] = read_output_major_weight(input_depth, global_column_base);
                }
            } else {
                weight_tile[tile_depth][local_x] = vec4<f32>(0.0);
            }
        }
        workgroupBarrier();

        for (var depth_vector = 0u; depth_vector < 8u; depth_vector = depth_vector + 1u) {
            let coefficient0 = weight_tile[depth_vector * 4u + 0u][local_x];
            let coefficient1 = weight_tile[depth_vector * 4u + 1u][local_x];
            let coefficient2 = weight_tile[depth_vector * 4u + 2u][local_x];
            let coefficient3 = weight_tile[depth_vector * 4u + 3u][local_x];
            for (var row_offset = 0u; row_offset < PROJECTION_ROWS_PER_LANE; row_offset = row_offset + 1u) {
                let activation = input_tile[local_y * 4u + row_offset][depth_vector];
                accumulators[row_offset] = fma(vec4<f32>(activation.x), coefficient0, accumulators[row_offset]);
                accumulators[row_offset] = fma(vec4<f32>(activation.y), coefficient1, accumulators[row_offset]);
                accumulators[row_offset] = fma(vec4<f32>(activation.z), coefficient2, accumulators[row_offset]);
                accumulators[row_offset] = fma(vec4<f32>(activation.w), coefficient3, accumulators[row_offset]);
            }
        }
        workgroupBarrier();
    }

    if global_column_vec < output_width_vec {
        let bias_value = bias.data[global_column_vec];
        for (var row_offset = 0u; row_offset < PROJECTION_ROWS_PER_LANE; row_offset = row_offset + 1u) {
            let output_row = global_row_start + local_y * PROJECTION_ROWS_PER_LANE + row_offset;
            if output_row < params.patch_count {
                let values = accumulators[row_offset] + bias_value;
                let output_base = output_row * params.output_width + global_column_base;
                output.data[output_base + 0u] = values.x;
                output.data[output_base + 1u] = values.y;
                output.data[output_base + 2u] = values.z;
                output.data[output_base + 3u] = values.w;
            }
        }
    }
}
"#;

const VISION_QKV_FUSED_F16_WEIGHTS_SOURCE: &str = r#"enable f16;
struct F32Vec4Buffer {
    data: array<vec4<f32>>,
}
struct F16Vec4Buffer {
    data: array<vec4<f16>>,
}
struct F32Buffer {
    data: array<f32>,
}
struct Params {
    tokens: u32,
    input_width: u32,
    output_width: u32,
    plane_stride_elements: u32,
}
@group(0) @binding(0) var<storage, read> input: F32Vec4Buffer;
@group(0) @binding(1) var<storage, read> query_weight: F16Vec4Buffer;
@group(0) @binding(2) var<storage, read> query_bias: F32Vec4Buffer;
@group(0) @binding(3) var<storage, read> key_weight: F16Vec4Buffer;
@group(0) @binding(4) var<storage, read> key_bias: F32Vec4Buffer;
@group(0) @binding(5) var<storage, read> value_weight: F16Vec4Buffer;
@group(0) @binding(6) var<storage, read> value_bias: F32Vec4Buffer;
@group(0) @binding(7) var<storage, read_write> output: F32Buffer;
@group(0) @binding(8) var<uniform> params: Params;
const QKV_TILE_ROWS: u32 = 16u;
const QKV_TILE_COLUMNS: u32 = 32u;
const QKV_TILE_DEPTH: u32 = 16u;
const QKV_ROWS_PER_LANE: u32 = 2u;
const QKV_WORKGROUP_SIZE: u32 = 64u;
var<workgroup> input_tile: array<array<vec4<f32>, 4>, 16>;
var<workgroup> query_weight_tile: array<array<vec4<f32>, 8>, 16>;
var<workgroup> key_weight_tile: array<array<vec4<f32>, 8>, 16>;
var<workgroup> value_weight_tile: array<array<vec4<f32>, 8>, 16>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let local_x = local_id.x;
    let local_y = local_id.y;
    let input_width_vec = params.input_width / 4u;
    let output_width_vec = params.output_width / 4u;
    let global_row_start = workgroup_id.y * QKV_TILE_ROWS;
    let global_column_vec = workgroup_id.x * (QKV_TILE_COLUMNS / 4u) + local_x;
    let output_column_base = global_column_vec * 4u;
    var query_accumulators: array<vec4<f32>, 2>;
    var key_accumulators: array<vec4<f32>, 2>;
    var value_accumulators: array<vec4<f32>, 2>;
    for (var row_offset = 0u; row_offset < QKV_ROWS_PER_LANE; row_offset = row_offset + 1u) {
        query_accumulators[row_offset] = vec4<f32>(0.0);
        key_accumulators[row_offset] = vec4<f32>(0.0);
        value_accumulators[row_offset] = vec4<f32>(0.0);
    }

    for (var depth_base = 0u; depth_base < params.input_width; depth_base = depth_base + QKV_TILE_DEPTH) {
        let input_depth_vec = depth_base / 4u + local_x;
        for (var row_offset = 0u; row_offset < QKV_ROWS_PER_LANE; row_offset = row_offset + 1u) {
            let tile_row = local_y * QKV_ROWS_PER_LANE + row_offset;
            let global_row = global_row_start + tile_row;
            if local_x < 4u && global_row < params.tokens && input_depth_vec < input_width_vec {
                input_tile[tile_row][local_x] = input.data[global_row * input_width_vec + input_depth_vec];
            } else if local_x < 4u {
                input_tile[tile_row][local_x] = vec4<f32>(0.0);
            }
        }
        for (var depth_offset = 0u; depth_offset < QKV_ROWS_PER_LANE; depth_offset = depth_offset + 1u) {
            let tile_depth = local_y * QKV_ROWS_PER_LANE + depth_offset;
            let input_depth = depth_base + tile_depth;
            if input_depth < params.input_width && global_column_vec < output_width_vec {
                query_weight_tile[tile_depth][local_x] = vec4<f32>(query_weight.data[input_depth * output_width_vec + global_column_vec]);
                key_weight_tile[tile_depth][local_x] = vec4<f32>(key_weight.data[input_depth * output_width_vec + global_column_vec]);
                value_weight_tile[tile_depth][local_x] = vec4<f32>(value_weight.data[input_depth * output_width_vec + global_column_vec]);
            } else {
                query_weight_tile[tile_depth][local_x] = vec4<f32>(0.0);
                key_weight_tile[tile_depth][local_x] = vec4<f32>(0.0);
                value_weight_tile[tile_depth][local_x] = vec4<f32>(0.0);
            }
        }
        workgroupBarrier();

        for (var depth_vector = 0u; depth_vector < 4u; depth_vector = depth_vector + 1u) {
            let query_coefficient0 = query_weight_tile[depth_vector * 4u + 0u][local_x];
            let query_coefficient1 = query_weight_tile[depth_vector * 4u + 1u][local_x];
            let query_coefficient2 = query_weight_tile[depth_vector * 4u + 2u][local_x];
            let query_coefficient3 = query_weight_tile[depth_vector * 4u + 3u][local_x];
            let key_coefficient0 = key_weight_tile[depth_vector * 4u + 0u][local_x];
            let key_coefficient1 = key_weight_tile[depth_vector * 4u + 1u][local_x];
            let key_coefficient2 = key_weight_tile[depth_vector * 4u + 2u][local_x];
            let key_coefficient3 = key_weight_tile[depth_vector * 4u + 3u][local_x];
            let value_coefficient0 = value_weight_tile[depth_vector * 4u + 0u][local_x];
            let value_coefficient1 = value_weight_tile[depth_vector * 4u + 1u][local_x];
            let value_coefficient2 = value_weight_tile[depth_vector * 4u + 2u][local_x];
            let value_coefficient3 = value_weight_tile[depth_vector * 4u + 3u][local_x];
            for (var row_offset = 0u; row_offset < QKV_ROWS_PER_LANE; row_offset = row_offset + 1u) {
                let activation = input_tile[local_y * 4u + row_offset][depth_vector];
                query_accumulators[row_offset] = fma(
                    vec4<f32>(activation.x),
                    query_coefficient0,
                    query_accumulators[row_offset],
                );
                query_accumulators[row_offset] = fma(
                    vec4<f32>(activation.y),
                    query_coefficient1,
                    query_accumulators[row_offset],
                );
                query_accumulators[row_offset] = fma(
                    vec4<f32>(activation.z),
                    query_coefficient2,
                    query_accumulators[row_offset],
                );
                query_accumulators[row_offset] = fma(
                    vec4<f32>(activation.w),
                    query_coefficient3,
                    query_accumulators[row_offset],
                );
                key_accumulators[row_offset] = fma(
                    vec4<f32>(activation.x),
                    key_coefficient0,
                    key_accumulators[row_offset],
                );
                key_accumulators[row_offset] = fma(
                    vec4<f32>(activation.y),
                    key_coefficient1,
                    key_accumulators[row_offset],
                );
                key_accumulators[row_offset] = fma(
                    vec4<f32>(activation.z),
                    key_coefficient2,
                    key_accumulators[row_offset],
                );
                key_accumulators[row_offset] = fma(
                    vec4<f32>(activation.w),
                    key_coefficient3,
                    key_accumulators[row_offset],
                );
                value_accumulators[row_offset] = fma(
                    vec4<f32>(activation.x),
                    value_coefficient0,
                    value_accumulators[row_offset],
                );
                value_accumulators[row_offset] = fma(
                    vec4<f32>(activation.y),
                    value_coefficient1,
                    value_accumulators[row_offset],
                );
                value_accumulators[row_offset] = fma(
                    vec4<f32>(activation.z),
                    value_coefficient2,
                    value_accumulators[row_offset],
                );
                value_accumulators[row_offset] = fma(
                    vec4<f32>(activation.w),
                    value_coefficient3,
                    value_accumulators[row_offset],
                );
            }
        }
        workgroupBarrier();
    }

    var query_bias_value = vec4<f32>(0.0);
    var key_bias_value = vec4<f32>(0.0);
    var value_bias_value = vec4<f32>(0.0);
    if global_column_vec < output_width_vec {
        query_bias_value = query_bias.data[global_column_vec];
        key_bias_value = key_bias.data[global_column_vec];
        value_bias_value = value_bias.data[global_column_vec];
    }
    let query_plane = 0u;
    let key_plane = params.plane_stride_elements;
    let value_plane = params.plane_stride_elements * 2u;
    for (var row_offset = 0u; row_offset < QKV_ROWS_PER_LANE; row_offset = row_offset + 1u) {
        let output_row = global_row_start + local_y * QKV_ROWS_PER_LANE + row_offset;
        if output_row < params.tokens && output_column_base + 3u < params.output_width {
            let query_values = query_accumulators[row_offset] + query_bias_value;
            let key_values = key_accumulators[row_offset] + key_bias_value;
            let value_values = value_accumulators[row_offset] + value_bias_value;
            let output_index = output_row * params.output_width + output_column_base;
            output.data[query_plane + output_index + 0u] = query_values.x;
            output.data[query_plane + output_index + 1u] = query_values.y;
            output.data[query_plane + output_index + 2u] = query_values.z;
            output.data[query_plane + output_index + 3u] = query_values.w;
            output.data[key_plane + output_index + 0u] = key_values.x;
            output.data[key_plane + output_index + 1u] = key_values.y;
            output.data[key_plane + output_index + 2u] = key_values.z;
            output.data[key_plane + output_index + 3u] = key_values.w;
            output.data[value_plane + output_index + 0u] = value_values.x;
            output.data[value_plane + output_index + 1u] = value_values.y;
            output.data[value_plane + output_index + 2u] = value_values.z;
            output.data[value_plane + output_index + 3u] = value_values.w;
        }
    }
}
"#;

const LAYER_NORM_F16_SOURCE: &str = r#"enable f16;
struct F16Vec4Buffer {
    data: array<vec4<f16>>,
}
struct Params {
    rows: u32,
    width: u32,
    epsilon: f32,
    padding: u32,
}
@group(0) @binding(0) var<storage, read> input: F16Vec4Buffer;
@group(0) @binding(1) var<storage, read> weight: F16Vec4Buffer;
@group(0) @binding(2) var<storage, read> bias: F16Vec4Buffer;
@group(0) @binding(3) var<storage, read_write> output: F16Vec4Buffer;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if row >= params.rows {
        return;
    }
    let width_vec = params.width / 4u;
    let row_start_vec = row * width_vec;
    let first = input.data[row_start_vec].x;
    var all_equal = true;
    var mean = 0.0f;
    for (var column_vec = 0u; column_vec < width_vec; column_vec = column_vec + 1u) {
        let value = input.data[row_start_vec + column_vec];
        mean = mean + f32(value.x) + f32(value.y) + f32(value.z) + f32(value.w);
        if any(value != vec4<f16>(first)) {
            all_equal = false;
        }
    }
    if all_equal {
        for (var column_vec = 0u; column_vec < width_vec; column_vec = column_vec + 1u) {
            output.data[row_start_vec + column_vec] = bias.data[column_vec];
        }
        return;
    }
    mean = mean / f32(params.width);
    var variance = 0.0f;
    for (var column_vec = 0u; column_vec < width_vec; column_vec = column_vec + 1u) {
        let centered = vec4<f32>(input.data[row_start_vec + column_vec]) - vec4<f32>(mean);
        variance = variance + dot(centered, centered);
    }
    let inverse_stddev = inverseSqrt(variance / f32(params.width) + params.epsilon);
    for (var column_vec = 0u; column_vec < width_vec; column_vec = column_vec + 1u) {
        let normalized =
            (vec4<f32>(input.data[row_start_vec + column_vec]) - vec4<f32>(mean))
                * vec4<f32>(inverse_stddev);
        let scale = vec4<f32>(weight.data[column_vec]);
        let shift = vec4<f32>(bias.data[column_vec]);
        output.data[row_start_vec + column_vec] = vec4<f16>(normalized * scale + shift);
    }
}
"#;

const LINEAR_PROJECTION_F16_SOURCE: &str = r#"enable f16;
struct F16Vec4Buffer {
    data: array<vec4<f16>>,
}
struct Params {
    patch_count: u32,
    input_width: u32,
    output_width: u32,
    padding: u32,
}
@group(0) @binding(0) var<storage, read> input: F16Vec4Buffer;
@group(0) @binding(1) var<storage, read> weight: F16Vec4Buffer;
@group(0) @binding(2) var<storage, read> bias: F16Vec4Buffer;
@group(0) @binding(3) var<storage, read_write> output: F16Vec4Buffer;
@group(0) @binding(4) var<uniform> params: Params;
const PROJECTION_TILE_ROWS: u32 = 32u;
const PROJECTION_TILE_COLUMNS: u32 = 32u;
const PROJECTION_TILE_DEPTH: u32 = 32u;
const PROJECTION_ROWS_PER_LANE: u32 = 4u;
var<workgroup> input_tile: array<array<vec4<f16>, 8>, 32>;
var<workgroup> weight_tile: array<array<vec4<f16>, 8>, 32>;

@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let local_x = local_id.x;
    let local_y = local_id.y;
    let input_width_vec = params.input_width / 4u;
    let output_width_vec = params.output_width / 4u;
    let global_row_start = workgroup_id.y * PROJECTION_TILE_ROWS;
    let global_column_vec = workgroup_id.x * 8u + local_x;
    var accumulators: array<vec4<f32>, 4>;
    for (var row_offset = 0u; row_offset < PROJECTION_ROWS_PER_LANE; row_offset = row_offset + 1u) {
        accumulators[row_offset] = vec4<f32>(0.0);
    }

    for (var depth_base = 0u; depth_base < params.input_width; depth_base = depth_base + PROJECTION_TILE_DEPTH) {
        let input_depth_vec = depth_base / 4u + local_x;
        for (var row_offset = 0u; row_offset < PROJECTION_ROWS_PER_LANE; row_offset = row_offset + 1u) {
            let tile_row = local_y * PROJECTION_ROWS_PER_LANE + row_offset;
            let global_row = global_row_start + tile_row;
            if global_row < params.patch_count && input_depth_vec < input_width_vec {
                input_tile[tile_row][local_x] = input.data[global_row * input_width_vec + input_depth_vec];
            } else {
                input_tile[tile_row][local_x] = vec4<f16>(0.0h);
            }
        }

        for (var depth_offset = 0u; depth_offset < PROJECTION_ROWS_PER_LANE; depth_offset = depth_offset + 1u) {
            let tile_depth = local_y * PROJECTION_ROWS_PER_LANE + depth_offset;
            let input_depth = depth_base + tile_depth;
            if input_depth < params.input_width && global_column_vec < output_width_vec {
                weight_tile[tile_depth][local_x] = weight.data[input_depth * output_width_vec + global_column_vec];
            } else {
                weight_tile[tile_depth][local_x] = vec4<f16>(0.0h);
            }
        }
        workgroupBarrier();

        for (var depth_vector = 0u; depth_vector < 8u; depth_vector = depth_vector + 1u) {
            let coefficient0 = weight_tile[depth_vector * 4u + 0u][local_x];
            let coefficient1 = weight_tile[depth_vector * 4u + 1u][local_x];
            let coefficient2 = weight_tile[depth_vector * 4u + 2u][local_x];
            let coefficient3 = weight_tile[depth_vector * 4u + 3u][local_x];
            for (var row_offset = 0u; row_offset < PROJECTION_ROWS_PER_LANE; row_offset = row_offset + 1u) {
                let activation = input_tile[local_y * 4u + row_offset][depth_vector];
                accumulators[row_offset] = fma(vec4<f32>(f32(activation.x)), vec4<f32>(coefficient0), accumulators[row_offset]);
                accumulators[row_offset] = fma(vec4<f32>(f32(activation.y)), vec4<f32>(coefficient1), accumulators[row_offset]);
                accumulators[row_offset] = fma(vec4<f32>(f32(activation.z)), vec4<f32>(coefficient2), accumulators[row_offset]);
                accumulators[row_offset] = fma(vec4<f32>(f32(activation.w)), vec4<f32>(coefficient3), accumulators[row_offset]);
            }
        }
        workgroupBarrier();
    }

    if global_column_vec < output_width_vec {
        let bias_value = vec4<f32>(bias.data[global_column_vec]);
        for (var row_offset = 0u; row_offset < PROJECTION_ROWS_PER_LANE; row_offset = row_offset + 1u) {
            let output_row = global_row_start + local_y * PROJECTION_ROWS_PER_LANE + row_offset;
            if output_row < params.patch_count {
                let values = accumulators[row_offset] + bias_value;
                output.data[output_row * output_width_vec + global_column_vec] = vec4<f16>(values);
            }
        }
    }
}
"#;

const VISION_ATTENTION_F16_SOURCE: &str = r#"enable f16;
struct F16Vec4Buffer {
    data: array<vec4<f16>>,
}
struct U32Buffer {
    data: array<u32>,
}
struct Params {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    segments: u32,
}
@group(0) @binding(0) var<storage, read> query: F16Vec4Buffer;
@group(0) @binding(1) var<storage, read> key: F16Vec4Buffer;
@group(0) @binding(2) var<storage, read> value: F16Vec4Buffer;
@group(0) @binding(3) var<storage, read> cu_seqlens: U32Buffer;
@group(0) @binding(4) var<storage, read_write> output: F16Vec4Buffer;
@group(0) @binding(5) var<uniform> params: Params;
const QUERY_TILE: u32 = 128u;
const KEY_STEP: u32 = 16u;
const MAX_HEAD_VECTORS: u32 = 18u;
const WORKGROUP_SIZE: u32 = 128u;
const MIN_F32: f32 = -3.402823466e+38;
var<workgroup> key_cache: array<vec4<f16>, 288>;
var<workgroup> value_cache: array<vec4<f16>, 288>;
@compute @workgroup_size(128, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
) {
    let local_index = local_invocation_id.x;
    let head = workgroup_id.y;
    let head_vectors = params.head_dim / 4u;
    let query_token = workgroup_id.x * QUERY_TILE + local_index;
    let query_valid = query_token < params.tokens && head < params.heads;
    var segment_start = 0u;
    var segment_end = params.tokens;
    if query_valid {
        for (var segment = 0u; segment < params.segments; segment = segment + 1u) {
            let candidate_end = cu_seqlens.data[segment + 1u];
            if query_token < candidate_end {
                segment_start = cu_seqlens.data[segment];
                segment_end = candidate_end;
                break;
            }
        }
    }

    var query_vectors: array<vec4<f16>, 18>;
    for (var vector_index = 0u; vector_index < MAX_HEAD_VECTORS; vector_index = vector_index + 1u) {
        query_vectors[vector_index] = vec4<f16>(0.0h);
        if query_valid && vector_index < head_vectors {
            let query_source =
                (query_token * params.heads + head) * head_vectors + vector_index;
            query_vectors[vector_index] = query.data[query_source];
        }
    }

    var attention_output: array<vec4<f32>, 18>;
    var running_maximum = MIN_F32;
    var running_denominator = 0.0f;
    let attention_scale = inverseSqrt(f32(params.head_dim));

    for (var key_start = 0u; key_start < params.tokens; key_start = key_start + KEY_STEP) {
        for (var cache_index = local_index; cache_index < KEY_STEP * MAX_HEAD_VECTORS; cache_index = cache_index + WORKGROUP_SIZE) {
            let key_slot = cache_index / MAX_HEAD_VECTORS;
            let vector_index = cache_index % MAX_HEAD_VECTORS;
            let key_token = key_start + key_slot;
            var loaded_key = vec4<f16>(0.0h);
            var loaded_value = vec4<f16>(0.0h);
            if key_token < params.tokens && head < params.heads && vector_index < head_vectors {
                let source =
                    (key_token * params.heads + head) * head_vectors + vector_index;
                loaded_key = key.data[source];
                loaded_value = value.data[source];
            }
            key_cache[cache_index] = loaded_key;
            value_cache[cache_index] = loaded_value;
        }
        workgroupBarrier();

        var scores: array<f32, 16>;
        var block_maximum = MIN_F32;
        for (var key_slot = 0u; key_slot < KEY_STEP; key_slot = key_slot + 1u) {
            let key_token = key_start + key_slot;
            let valid_key = query_valid && key_token < params.tokens
                && key_token >= segment_start && key_token < segment_end;
            scores[key_slot] = MIN_F32;
            if valid_key {
                scores[key_slot] = 0.0f;
                for (var vector_index = 0u; vector_index < MAX_HEAD_VECTORS; vector_index = vector_index + 1u) {
                    scores[key_slot] = scores[key_slot] + dot(
                        vec4<f32>(query_vectors[vector_index]),
                        vec4<f32>(key_cache[key_slot * MAX_HEAD_VECTORS + vector_index]),
                    );
                }
                scores[key_slot] = scores[key_slot] * attention_scale;
                block_maximum = max(block_maximum, scores[key_slot]);
            }
        }

        if block_maximum > MIN_F32 {
            let next_maximum = max(running_maximum, block_maximum);
            var previous_scale = 0.0f;
            if running_denominator > 0.0f {
                previous_scale = exp(running_maximum - next_maximum);
            }
            var block_denominator = 0.0f;
            for (var key_slot = 0u; key_slot < KEY_STEP; key_slot = key_slot + 1u) {
                let key_token = key_start + key_slot;
                let valid_key = query_valid && key_token < params.tokens
                    && key_token >= segment_start && key_token < segment_end;
                if valid_key {
                    scores[key_slot] = exp(scores[key_slot] - next_maximum);
                } else {
                    scores[key_slot] = 0.0f;
                }
                block_denominator = block_denominator + scores[key_slot];
            }
            for (var vector_index = 0u; vector_index < MAX_HEAD_VECTORS; vector_index = vector_index + 1u) {
                var block_weighted = vec4<f32>(0.0);
                for (var key_slot = 0u; key_slot < KEY_STEP; key_slot = key_slot + 1u) {
                    block_weighted = block_weighted + scores[key_slot] * vec4<f32>(value_cache[key_slot * MAX_HEAD_VECTORS + vector_index]);
                }
                attention_output[vector_index] =
                    attention_output[vector_index] * previous_scale + block_weighted;
            }
            running_denominator = running_denominator * previous_scale + block_denominator;
            running_maximum = next_maximum;
        }
        workgroupBarrier();
    }

    if query_valid && running_denominator > 0.0f {
        let query_base_vec =
            (query_token * params.heads + head) * head_vectors;
        for (var vector_index = 0u; vector_index < MAX_HEAD_VECTORS; vector_index = vector_index + 1u) {
            if vector_index < head_vectors {
                let normalized = attention_output[vector_index] / running_denominator;
                output.data[query_base_vec + vector_index] = vec4<f16>(normalized);
            }
        }
    }
}
"#;

const ADD_F16_SOURCE: &str = r#"enable f16;
struct F16Vec4Buffer {
    data: array<vec4<f16>>,
}
struct Params {
    length: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}
@group(0) @binding(0) var<storage, read> left: F16Vec4Buffer;
@group(0) @binding(1) var<storage, read> right: F16Vec4Buffer;
@group(0) @binding(2) var<storage, read_write> output: F16Vec4Buffer;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vector_length = (params.length + 3u) / 4u;
    let row_stride = select(vector_length, params.padding0, params.padding0 != 0u);
    let index = global_id.x + global_id.y * row_stride;
    if index >= vector_length {
        return;
    }
    output.data[index] = left.data[index] + right.data[index];
}
"#;

const GELU_TANH_F16_SOURCE: &str = r#"enable f16;
struct F16Vec4Buffer {
    data: array<vec4<f16>>,
}
struct Params {
    length: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}
@group(0) @binding(0) var<storage, read> input: F16Vec4Buffer;
@group(0) @binding(1) var<storage, read_write> output: F16Vec4Buffer;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vector_length = (params.length + 3u) / 4u;
    let row_stride = select(vector_length, params.padding0, params.padding0 != 0u);
    let index = global_id.x + global_id.y * row_stride;
    if index >= vector_length {
        return;
    }
    let value = input.data[index];
    let cubic = value * value * value;
    let argument = vec4<f16>(0.7978846h) * (value + vec4<f16>(0.044715h) * cubic);
    output.data[index] = vec4<f16>(0.5h) * value * (vec4<f16>(1.0h) + tanh(argument));
}
"#;

const GELU_ERF_F16_SOURCE: &str = r#"enable f16;
struct F16Vec4Buffer {
    data: array<vec4<f16>>,
}
struct Params {
    length: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}
@group(0) @binding(0) var<storage, read> input: F16Vec4Buffer;
@group(0) @binding(1) var<storage, read_write> output: F16Vec4Buffer;
@group(0) @binding(2) var<uniform> params: Params;
fn erf_approx(value: vec4<f32>) -> vec4<f32> {
    let absolute = abs(value);
    let reciprocal = vec4<f32>(1.0) /
        (vec4<f32>(1.0) + vec4<f32>(0.3275911) * absolute);
    var polynomial = vec4<f32>(1.061405429);
    polynomial = vec4<f32>(-1.453152027) + reciprocal * polynomial;
    polynomial = vec4<f32>(1.421413741) + reciprocal * polynomial;
    polynomial = vec4<f32>(-0.284496736) + reciprocal * polynomial;
    polynomial = vec4<f32>(0.254829592) + reciprocal * polynomial;
    let approximation = vec4<f32>(1.0) -
        reciprocal * polynomial * exp(-absolute * absolute);
    return select(approximation, -approximation, value < vec4<f32>(0.0));
}
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vector_length = (params.length + 3u) / 4u;
    let row_stride =
        select(vector_length, params.padding0, params.padding0 != 0u);
    let index = global_id.x + global_id.y * row_stride;
    if index >= vector_length {
        return;
    }
    let value = vec4<f32>(input.data[index]);
    let argument = value * vec4<f32>(0.7071067811865476);
    let gelu =
        vec4<f32>(0.5) * value * (vec4<f32>(1.0) + erf_approx(argument));
    output.data[index] = vec4<f16>(gelu);
}
"#;

const VISION_ROPE_2D_F16_SOURCE: &str = r#"enable f16;
struct F16Buffer {
    data: array<f16>,
}
struct F32Buffer {
    data: array<f32>,
}
struct Params {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    padding: u32,
}
@group(0) @binding(0) var<storage, read_write> query: F16Buffer;
@group(0) @binding(1) var<storage, read_write> key: F16Buffer;
@group(0) @binding(2) var<storage, read> cos_table: F32Buffer;
@group(0) @binding(3) var<storage, read> sin_table: F32Buffer;
@group(0) @binding(4) var<uniform> params: Params;
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_count = params.head_dim / 2u;
    let work_items = params.tokens * params.heads * pair_count;
    let linear_pair = global_id.x;
    if linear_pair >= work_items {
        return;
    }
    let pair = linear_pair % pair_count;
    let linear_head = linear_pair / pair_count;
    let head = linear_head % params.heads;
    let token = linear_head / params.heads;
    let first_index = (token * params.heads + head) * params.head_dim + pair;
    let second_index = first_index + pair_count;
    let cosine = cos_table.data[token * pair_count + pair];
    let sine = sin_table.data[token * pair_count + pair];
    let query_first = f32(query.data[first_index]);
    let query_second = f32(query.data[second_index]);
    let key_first = f32(key.data[first_index]);
    let key_second = f32(key.data[second_index]);
    query.data[first_index] = f16(query_first * cosine - query_second * sine);
    query.data[second_index] = f16(query_second * cosine + query_first * sine);
    key.data[first_index] = f16(key_first * cosine - key_second * sine);
    key.data[second_index] = f16(key_second * cosine + key_first * sine);
}
"#;

const FULL_CATALOG: &[KernelModule] = &[
    kernel(
        KernelId::GemmF32,
        [8, 8, 1],
        GEMM_BINDINGS,
        GEMM_UNIFORM,
        GEMM_SOURCE,
    ),
    kernel(
        KernelId::GemvF32,
        [64, 1, 1],
        GEMV_BINDINGS,
        GEMV_UNIFORM,
        GEMV_SOURCE,
    ),
    kernel(
        KernelId::LayerNormF32,
        [64, 1, 1],
        LAYER_NORM_BINDINGS,
        NORM_UNIFORM,
        LAYER_NORM_SOURCE,
    ),
    kernel(
        KernelId::RmsNormF32,
        [64, 1, 1],
        RMS_NORM_BINDINGS,
        NORM_UNIFORM,
        RMS_NORM_SOURCE,
    ),
    kernel(
        KernelId::SiluF32,
        [64, 1, 1],
        ACTIVATION_BINDINGS,
        ACTIVATION_UNIFORM,
        SILU_SOURCE,
    ),
    kernel(
        KernelId::GeluTanhF32,
        [64, 1, 1],
        ACTIVATION_BINDINGS,
        ACTIVATION_UNIFORM,
        GELU_SOURCE,
    ),
    kernel(
        KernelId::RopeNeoxF32,
        [64, 1, 1],
        ROPE_BINDINGS,
        ROPE_UNIFORM,
        ROPE_SOURCE,
    ),
    kernel(
        KernelId::VisionAttentionF32,
        [128, 1, 1],
        VISION_ATTENTION_BINDINGS,
        VISION_ATTENTION_UNIFORM,
        VISION_ATTENTION_SOURCE,
    ),
    kernel(
        KernelId::VisionPatchProjectionF32,
        [8, 8, 1],
        VISION_PATCH_PROJECTION_BINDINGS,
        VISION_PATCH_PROJECTION_UNIFORM,
        VISION_PATCH_PROJECTION_SOURCE,
    ),
    kernel(
        KernelId::AddF32,
        [64, 1, 1],
        ADD_BINDINGS,
        ACTIVATION_UNIFORM,
        ADD_SOURCE,
    ),
    kernel(
        KernelId::GeluErfF32,
        [64, 1, 1],
        ACTIVATION_BINDINGS,
        ACTIVATION_UNIFORM,
        GELU_ERF_SOURCE,
    ),
    kernel(
        KernelId::ProjectorMerge2x2F32,
        [64, 1, 1],
        PROJECTOR_MERGE_BINDINGS,
        PROJECTOR_MERGE_UNIFORM,
        PROJECTOR_MERGE_SOURCE,
    ),
    kernel(
        KernelId::VisionQkvFusedF32,
        [8, 8, 1],
        VISION_QKV_FUSED_BINDINGS,
        VISION_QKV_FUSED_UNIFORM,
        VISION_QKV_FUSED_SOURCE,
    ),
    kernel(
        KernelId::DecoderKvAppendF32,
        [64, 1, 1],
        DECODER_KV_APPEND_BINDINGS,
        DECODER_KV_APPEND_UNIFORM,
        DECODER_KV_APPEND_SOURCE,
    ),
    kernel(
        KernelId::DecoderGqaF32,
        [64, 1, 1],
        DECODER_GQA_BINDINGS,
        DECODER_GQA_UNIFORM,
        DECODER_GQA_SOURCE,
    ),
    kernel(
        KernelId::DecoderGqaSplitPartialF32,
        [64, 1, 1],
        DECODER_GQA_BINDINGS,
        DECODER_GQA_SPLIT_UNIFORM,
        DECODER_GQA_SPLIT_PARTIAL_SOURCE,
    ),
    kernel(
        KernelId::DecoderGqaSplitMergeF32,
        [64, 1, 1],
        DECODER_GQA_BINDINGS,
        DECODER_GQA_SPLIT_UNIFORM,
        DECODER_GQA_SPLIT_MERGE_SOURCE,
    ),
    kernel(
        KernelId::DecoderMropeF32,
        [64, 1, 1],
        DECODER_MROPE_BINDINGS,
        DECODER_MROPE_UNIFORM,
        DECODER_MROPE_SOURCE,
    ),
    kernel(
        KernelId::DecoderSwigluF32,
        [64, 1, 1],
        DECODER_SWIGLU_BINDINGS,
        DECODER_SWIGLU_UNIFORM,
        DECODER_SWIGLU_SOURCE,
    ),
    kernel(
        KernelId::DecoderPrefillGqaF32,
        [64, 1, 1],
        DECODER_GQA_BINDINGS,
        DECODER_PREFILL_GQA_UNIFORM,
        DECODER_PREFILL_GQA_SOURCE,
    ),
    kernel(
        KernelId::DecoderPrefillMropeF32,
        [64, 1, 1],
        DECODER_MROPE_BINDINGS,
        DECODER_PREFILL_MROPE_UNIFORM,
        DECODER_PREFILL_MROPE_SOURCE,
    ),
    kernel(
        KernelId::DecoderKvAppendRangeF32,
        [64, 1, 1],
        DECODER_KV_APPEND_BINDINGS,
        DECODER_KV_APPEND_RANGE_UNIFORM,
        DECODER_KV_APPEND_RANGE_SOURCE,
    ),
    kernel(
        KernelId::GemvTiledF32,
        [256, 1, 1],
        GEMV_TILED_BINDINGS,
        GEMV_UNIFORM,
        GEMV_TILED_SOURCE,
    ),
    fp16_kernel(
        KernelId::RmsNormF16Weights,
        [64, 1, 1],
        RMS_NORM_F16_BINDINGS,
        NORM_UNIFORM,
        RMS_NORM_F16_WEIGHTS_SOURCE,
    ),
    fp16_kernel(
        KernelId::GemvTiledF16Weights,
        [256, 1, 1],
        GEMV_TILED_F16_BINDINGS,
        GEMV_UNIFORM,
        GEMV_TILED_F16_WEIGHTS_SOURCE,
    ),
    fp16_kernel(
        KernelId::LinearProjectionF16Weights,
        [8, 8, 1],
        LINEAR_PROJECTION_F16_WEIGHT_BINDINGS,
        VISION_PATCH_PROJECTION_UNIFORM,
        LINEAR_PROJECTION_F16_WEIGHTS_SOURCE,
    ),
    kernel(
        KernelId::VisionRope2dF32,
        [64, 1, 1],
        VISION_ROPE_2D_BINDINGS,
        VISION_ROPE_2D_UNIFORM,
        VISION_ROPE_2D_SOURCE,
    ),
    fp16_kernel(
        KernelId::VisionQkvFusedF16Weights,
        [8, 8, 1],
        VISION_QKV_FUSED_F16_BINDINGS,
        VISION_QKV_FUSED_UNIFORM,
        VISION_QKV_FUSED_F16_WEIGHTS_SOURCE,
    ),
    fp16_kernel(
        KernelId::LayerNormF16,
        [64, 1, 1],
        LAYER_NORM_F16_BINDINGS,
        NORM_UNIFORM,
        LAYER_NORM_F16_SOURCE,
    ),
    fp16_kernel(
        KernelId::LinearProjectionF16,
        [8, 8, 1],
        LINEAR_PROJECTION_F16_BINDINGS,
        VISION_PATCH_PROJECTION_UNIFORM,
        LINEAR_PROJECTION_F16_SOURCE,
    ),
    fp16_kernel(
        KernelId::VisionAttentionF16,
        [128, 1, 1],
        VISION_ATTENTION_F16_BINDINGS,
        VISION_ATTENTION_UNIFORM,
        VISION_ATTENTION_F16_SOURCE,
    ),
    fp16_kernel(
        KernelId::AddF16,
        [64, 1, 1],
        ADD_F16_BINDINGS,
        ACTIVATION_UNIFORM,
        ADD_F16_SOURCE,
    ),
    fp16_kernel(
        KernelId::GeluTanhF16,
        [64, 1, 1],
        ACTIVATION_F16_BINDINGS,
        ACTIVATION_UNIFORM,
        GELU_TANH_F16_SOURCE,
    ),
    fp16_kernel(
        KernelId::VisionRope2dF16,
        [64, 1, 1],
        VISION_ROPE_2D_F16_BINDINGS,
        VISION_ROPE_2D_UNIFORM,
        VISION_ROPE_2D_F16_SOURCE,
    ),
    fp16_kernel(
        KernelId::ProjectorMerge2x2F16,
        [64, 1, 1],
        PROJECTOR_MERGE_F16_BINDINGS,
        PROJECTOR_MERGE_UNIFORM,
        PROJECTOR_MERGE_F16_SOURCE,
    ),
    fp16_kernel(
        KernelId::GeluErfF16,
        [64, 1, 1],
        ACTIVATION_F16_BINDINGS,
        ACTIVATION_UNIFORM,
        GELU_ERF_F16_SOURCE,
    ),
];

const fn kernel(
    id: KernelId,
    workgroup_size: [u32; 3],
    bindings: &'static [BindingSpec],
    uniform_fields: &'static [UniformFieldSpec],
    source: &'static str,
) -> KernelModule {
    KernelModule {
        spec: KernelSpec {
            kernel: id,
            entry_point: "main",
            workgroup_size,
            bindings,
            uniform_fields,
            uniform_span: 16,
            required_features: &[],
        },
        source,
    }
}

const fn fp16_kernel(
    id: KernelId,
    workgroup_size: [u32; 3],
    bindings: &'static [BindingSpec],
    uniform_fields: &'static [UniformFieldSpec],
    source: &'static str,
) -> KernelModule {
    let mut module = kernel(id, workgroup_size, bindings, uniform_fields, source);
    module.spec.required_features = &["shader_f16"];
    module
}

#[must_use]
pub fn catalog() -> &'static [KernelModule] {
    &FULL_CATALOG[..KernelId::M2_PRIMITIVES.len()]
}

#[must_use]
pub const fn full_catalog() -> &'static [KernelModule] {
    FULL_CATALOG
}

#[must_use]
pub fn module(kernel: KernelId) -> Option<&'static KernelModule> {
    FULL_CATALOG
        .iter()
        .find(|module| module.spec.kernel == kernel)
}

pub fn validate_catalog() -> Result<(), WgslError> {
    validate_modules(FULL_CATALOG)
}

pub fn validate_modules(modules: &[KernelModule]) -> Result<(), WgslError> {
    let mut kernels = BTreeSet::new();
    for module in modules {
        if !kernels.insert(module.spec.kernel) {
            return Err(WgslError::for_kernel(
                WgslErrorCode::DuplicateKernel,
                module.spec.kernel,
                "WGSL catalog contains a duplicate kernel",
            ));
        }
        validate_source_contract(&module.spec, module.source)?;
    }
    Ok(())
}

pub fn validate_source_contract(spec: &KernelSpec, source: &str) -> Result<(), WgslError> {
    validate_source_contract_with_access(spec, source, StorageAccessMode::Declared)
}

/// Builds and validates a deterministic shader variant whose storage bindings
/// all use read-write access. This is useful when distinct verified slices of
/// one physical buffer must share one conservative resource-usage state.
pub fn storage_read_write_variant(spec: &KernelSpec, source: &str) -> Result<String, WgslError> {
    validate_source_contract(spec, source)?;
    let expected_replacements = spec
        .bindings
        .iter()
        .filter(|binding| {
            matches!(
                binding.kind,
                BindingKind::StorageReadF32
                    | BindingKind::StorageReadVec4F32
                    | BindingKind::StorageReadF16
                    | BindingKind::StorageReadVec4F16
                    | BindingKind::StorageReadU32
            )
        })
        .count();
    let declaration = "var<storage, read>";
    let actual_replacements = source.matches(declaration).count();
    if actual_replacements != expected_replacements {
        return Err(binding_error(
            spec,
            format!(
                "shader has {actual_replacements} storage-read declarations but ABI requires {expected_replacements}"
            ),
        ));
    }
    let variant = source.replace(declaration, "var<storage, read_write>");
    validate_source_contract_with_access(spec, &variant, StorageAccessMode::AllReadWrite)?;
    Ok(variant)
}

#[derive(Clone, Copy)]
enum StorageAccessMode {
    Declared,
    AllReadWrite,
}

fn validate_source_contract_with_access(
    spec: &KernelSpec,
    source: &str,
    storage_access: StorageAccessMode,
) -> Result<(), WgslError> {
    let enabled_features = source
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            line.starts_with("enable ").then_some(line)
        })
        .collect::<Vec<_>>();
    let shader_f16 = spec.required_features == ["shader_f16"];
    if shader_f16 {
        if enabled_features != ["enable f16;"] {
            return Err(WgslError::for_kernel(
                WgslErrorCode::ForbiddenFeature,
                spec.kernel,
                "shader-f16 kernels must enable exactly the WGSL f16 extension",
            ));
        }
    } else if !spec.required_features.is_empty() || !enabled_features.is_empty() {
        return Err(WgslError::for_kernel(
            WgslErrorCode::ForbiddenFeature,
            spec.kernel,
            "FP32 baseline shaders may not enable optional WGSL features",
        ));
    }
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(module) => module,
        Err(error) => {
            if source.contains("struct Params") && source.contains("@group(") {
                preflight_text_abi(spec, source)?;
            }
            return Err(WgslError::for_kernel(
                WgslErrorCode::Parse,
                spec.kernel,
                format!("WGSL parse failed: {error}"),
            ));
        }
    };
    preflight_text_abi(spec, source)?;
    validate_entry_point(spec, &module)?;
    validate_bindings(spec, &module, storage_access)?;
    let capabilities = if shader_f16 {
        Capabilities::SHADER_FLOAT16
    } else {
        Capabilities::empty()
    };
    Validator::new(ValidationFlags::all(), capabilities)
        .validate(&module)
        .map_err(|error| {
            WgslError::for_kernel(
                WgslErrorCode::Validation,
                spec.kernel,
                format!("WGSL validation failed: {error}"),
            )
        })?;
    Ok(())
}

fn preflight_text_abi(spec: &KernelSpec, source: &str) -> Result<(), WgslError> {
    let compact: String = source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    if spec.bindings.iter().any(|binding| {
        matches!(
            binding.kind,
            BindingKind::StorageReadF32 | BindingKind::StorageReadWriteF32
        )
    }) && !compact.contains("array<f32>")
    {
        return Err(binding_error(spec, "shader has no f32 storage array"));
    }
    if spec
        .bindings
        .iter()
        .any(|binding| binding.kind == BindingKind::StorageReadVec4F32)
        && !compact.contains("array<vec4<f32>>")
    {
        return Err(binding_error(spec, "shader has no vec4<f32> storage array"));
    }
    if spec
        .bindings
        .iter()
        .any(|binding| binding.kind == BindingKind::StorageReadF16)
        && !compact.contains("array<f16>")
    {
        return Err(binding_error(spec, "shader has no f16 storage array"));
    }
    if spec
        .bindings
        .iter()
        .any(|binding| binding.kind == BindingKind::StorageReadVec4F16)
        && !compact.contains("array<vec4<f16>>")
    {
        return Err(binding_error(spec, "shader has no vec4<f16> storage array"));
    }
    if spec
        .bindings
        .iter()
        .any(|binding| binding.kind == BindingKind::StorageReadWriteF16)
        && !compact.contains("array<f16>")
    {
        return Err(binding_error(spec, "shader has no f16 storage array"));
    }
    if spec
        .bindings
        .iter()
        .any(|binding| binding.kind == BindingKind::StorageReadWriteVec4F16)
        && !compact.contains("array<vec4<f16>>")
    {
        return Err(binding_error(spec, "shader has no vec4<f16> storage array"));
    }
    if spec
        .bindings
        .iter()
        .any(|binding| binding.kind == BindingKind::StorageReadU32)
        && !compact.contains("array<u32>")
    {
        return Err(binding_error(spec, "shader has no u32 storage array"));
    }

    let Some(params_start) = source.find("struct Params") else {
        return Err(uniform_error(spec, "shader has no Params uniform struct"));
    };
    let params = &source[params_start..];
    let Some(open) = params.find('{') else {
        return Err(uniform_error(spec, "Params struct has no body"));
    };
    let Some(close) = params[open + 1..].find('}') else {
        return Err(uniform_error(spec, "Params struct is unterminated"));
    };
    let fields: Vec<_> = params[open + 1..open + 1 + close]
        .split(',')
        .map(str::trim)
        .filter(|field| !field.is_empty())
        .collect();
    if fields.len() != spec.uniform_fields.len() {
        return Err(uniform_error(spec, "Params field count differs from ABI"));
    }
    for (actual, expected) in fields.into_iter().zip(spec.uniform_fields) {
        let Some((name, scalar)) = actual.split_once(':') else {
            return Err(uniform_error(spec, "Params field is malformed"));
        };
        let expected_scalar = match expected.scalar {
            UniformScalar::U32 => "u32",
            UniformScalar::F32 => "f32",
        };
        if name.trim() != expected.name || scalar.trim() != expected_scalar {
            return Err(uniform_error(
                spec,
                "Params field order or type differs from ABI",
            ));
        }
    }
    Ok(())
}

fn validate_entry_point(spec: &KernelSpec, module: &Module) -> Result<(), WgslError> {
    if module.entry_points.len() != 1
        || module.entry_points[0].name != spec.entry_point
        || module.entry_points[0].stage != ShaderStage::Compute
    {
        return Err(WgslError::for_kernel(
            WgslErrorCode::MissingEntryPoint,
            spec.kernel,
            "shader must contain exactly the declared compute entry point",
        ));
    }
    if module.entry_points[0].workgroup_size != spec.workgroup_size {
        return Err(WgslError::for_kernel(
            WgslErrorCode::WorkgroupMismatch,
            spec.kernel,
            "compute workgroup size differs from the kernel ABI",
        ));
    }
    Ok(())
}

fn validate_bindings(
    spec: &KernelSpec,
    module: &Module,
    storage_access: StorageAccessMode,
) -> Result<(), WgslError> {
    let globals: Vec<_> = module
        .global_variables
        .iter()
        .filter(|(_, global)| global.binding.is_some())
        .map(|(_, global)| global)
        .collect();
    if globals.len() != spec.bindings.len() {
        return Err(binding_error(spec, "shader binding count differs from ABI"));
    }
    for expected in spec.bindings {
        let Some(global) = globals.iter().find(|global| {
            global.binding.as_ref().is_some_and(|binding| {
                binding.group == expected.group && binding.binding == expected.binding
            })
        }) else {
            return Err(binding_error(spec, "declared shader binding is missing"));
        };
        match (expected.kind, storage_access) {
            (BindingKind::StorageReadF32, StorageAccessMode::Declared) => {
                validate_storage(spec, module, global, StorageAccess::LOAD, ScalarKind::Float)?
            }
            (BindingKind::StorageReadVec4F32, StorageAccessMode::Declared) => {
                validate_storage_vec4_f32(spec, module, global, StorageAccess::LOAD)?
            }
            (BindingKind::StorageReadF16, StorageAccessMode::Declared) => {
                validate_storage_f16(spec, module, global, StorageAccess::LOAD)?
            }
            (BindingKind::StorageReadVec4F16, StorageAccessMode::Declared) => {
                validate_storage_vec4_f16(spec, module, global, StorageAccess::LOAD)?
            }
            (BindingKind::StorageReadU32, StorageAccessMode::Declared) => {
                validate_storage(spec, module, global, StorageAccess::LOAD, ScalarKind::Uint)?
            }
            (BindingKind::StorageReadF32, StorageAccessMode::AllReadWrite) => validate_storage(
                spec,
                module,
                global,
                StorageAccess::LOAD | StorageAccess::STORE,
                ScalarKind::Float,
            )?,
            (BindingKind::StorageReadVec4F32, StorageAccessMode::AllReadWrite) => {
                validate_storage_vec4_f32(
                    spec,
                    module,
                    global,
                    StorageAccess::LOAD | StorageAccess::STORE,
                )?
            }
            (BindingKind::StorageReadF16, StorageAccessMode::AllReadWrite) => validate_storage_f16(
                spec,
                module,
                global,
                StorageAccess::LOAD | StorageAccess::STORE,
            )?,
            (BindingKind::StorageReadVec4F16, StorageAccessMode::AllReadWrite) => {
                validate_storage_vec4_f16(
                    spec,
                    module,
                    global,
                    StorageAccess::LOAD | StorageAccess::STORE,
                )?
            }
            (BindingKind::StorageReadU32, StorageAccessMode::AllReadWrite) => validate_storage(
                spec,
                module,
                global,
                StorageAccess::LOAD | StorageAccess::STORE,
                ScalarKind::Uint,
            )?,
            (BindingKind::StorageReadWriteF32, _) => validate_storage(
                spec,
                module,
                global,
                StorageAccess::LOAD | StorageAccess::STORE,
                ScalarKind::Float,
            )?,
            (BindingKind::StorageReadWriteF16, _) => validate_storage_f16(
                spec,
                module,
                global,
                StorageAccess::LOAD | StorageAccess::STORE,
            )?,
            (BindingKind::StorageReadWriteVec4F16, _) => validate_storage_vec4_f16(
                spec,
                module,
                global,
                StorageAccess::LOAD | StorageAccess::STORE,
            )?,
            (BindingKind::Uniform, _) => validate_uniform(spec, module, global)?,
        }
    }
    Ok(())
}

fn validate_storage(
    spec: &KernelSpec,
    module: &Module,
    global: &naga::GlobalVariable,
    expected_access: StorageAccess,
    expected_scalar: ScalarKind,
) -> Result<(), WgslError> {
    validate_storage_scalar(spec, module, global, expected_access, expected_scalar, 4, 4)
}

fn validate_storage_f16(
    spec: &KernelSpec,
    module: &Module,
    global: &naga::GlobalVariable,
    expected_access: StorageAccess,
) -> Result<(), WgslError> {
    validate_storage_scalar(
        spec,
        module,
        global,
        expected_access,
        ScalarKind::Float,
        2,
        2,
    )
}

fn validate_storage_scalar(
    spec: &KernelSpec,
    module: &Module,
    global: &naga::GlobalVariable,
    expected_access: StorageAccess,
    expected_scalar: ScalarKind,
    expected_width: u8,
    expected_stride: u32,
) -> Result<(), WgslError> {
    if global.space
        != (AddressSpace::Storage {
            access: expected_access,
        })
    {
        return Err(binding_error(spec, "storage access mode differs from ABI"));
    }
    let TypeInner::Struct { members, .. } = &module.types[global.ty].inner else {
        return Err(binding_error(
            spec,
            "storage binding must use a buffer struct",
        ));
    };
    if members.len() != 1 || members[0].offset != 0 {
        return Err(binding_error(
            spec,
            "storage buffer struct layout differs from ABI",
        ));
    }
    let TypeInner::Array {
        base,
        size: ArraySize::Dynamic,
        stride,
    } = module.types[members[0].ty].inner
    else {
        return Err(binding_error(
            spec,
            "storage buffer must be a packed runtime array",
        ));
    };
    if stride != expected_stride {
        return Err(binding_error(
            spec,
            "storage buffer element stride differs from ABI",
        ));
    }
    let TypeInner::Scalar(scalar) = module.types[base].inner else {
        return Err(binding_error(spec, "storage buffer element must be scalar"));
    };
    if scalar.kind != expected_scalar || scalar.width != expected_width {
        return Err(binding_error(
            spec,
            "storage buffer scalar type differs from ABI",
        ));
    }
    Ok(())
}

fn validate_storage_vec4_f32(
    spec: &KernelSpec,
    module: &Module,
    global: &naga::GlobalVariable,
    expected_access: StorageAccess,
) -> Result<(), WgslError> {
    validate_storage_vec4(spec, module, global, expected_access, 4, 16)
}

fn validate_storage_vec4_f16(
    spec: &KernelSpec,
    module: &Module,
    global: &naga::GlobalVariable,
    expected_access: StorageAccess,
) -> Result<(), WgslError> {
    validate_storage_vec4(spec, module, global, expected_access, 2, 8)
}

fn validate_storage_vec4(
    spec: &KernelSpec,
    module: &Module,
    global: &naga::GlobalVariable,
    expected_access: StorageAccess,
    expected_width: u8,
    expected_stride: u32,
) -> Result<(), WgslError> {
    if global.space
        != (AddressSpace::Storage {
            access: expected_access,
        })
    {
        return Err(binding_error(spec, "storage access mode differs from ABI"));
    }
    let TypeInner::Struct { members, .. } = &module.types[global.ty].inner else {
        return Err(binding_error(
            spec,
            "storage binding must use a buffer struct",
        ));
    };
    if members.len() != 1 || members[0].offset != 0 {
        return Err(binding_error(
            spec,
            "storage buffer struct layout differs from ABI",
        ));
    }
    let TypeInner::Array {
        base,
        size: ArraySize::Dynamic,
        stride,
    } = module.types[members[0].ty].inner
    else {
        return Err(binding_error(
            spec,
            "vec4 storage buffer must be a packed runtime array",
        ));
    };
    if stride != expected_stride {
        return Err(binding_error(
            spec,
            "vec4 storage buffer element stride differs from ABI",
        ));
    }
    let TypeInner::Vector {
        size: VectorSize::Quad,
        scalar,
    } = module.types[base].inner
    else {
        return Err(binding_error(spec, "storage buffer element must be vec4"));
    };
    if scalar.kind != ScalarKind::Float || scalar.width != expected_width {
        return Err(binding_error(
            spec,
            "storage buffer vector type differs from ABI",
        ));
    }
    Ok(())
}

fn validate_uniform(
    spec: &KernelSpec,
    module: &Module,
    global: &naga::GlobalVariable,
) -> Result<(), WgslError> {
    if global.space != AddressSpace::Uniform {
        return Err(uniform_error(spec, "params binding is not uniform"));
    }
    let TypeInner::Struct { members, span } = &module.types[global.ty].inner else {
        return Err(uniform_error(spec, "uniform binding must use a struct"));
    };
    if *span != spec.uniform_span || members.len() != spec.uniform_fields.len() {
        return Err(uniform_error(spec, "uniform struct size differs from ABI"));
    }
    for (member, expected) in members.iter().zip(spec.uniform_fields) {
        let expected_kind = match expected.scalar {
            UniformScalar::U32 => ScalarKind::Uint,
            UniformScalar::F32 => ScalarKind::Float,
        };
        let TypeInner::Scalar(actual) = module.types[member.ty].inner else {
            return Err(uniform_error(spec, "uniform member must be scalar"));
        };
        if member.name.as_deref() != Some(expected.name)
            || member.offset != expected.offset
            || actual.kind != expected_kind
            || actual.width != 4
        {
            return Err(uniform_error(
                spec,
                "uniform member layout differs from ABI",
            ));
        }
    }
    Ok(())
}

fn binding_error(spec: &KernelSpec, message: impl Into<String>) -> WgslError {
    WgslError::for_kernel(WgslErrorCode::BindingMismatch, spec.kernel, message)
}

fn uniform_error(spec: &KernelSpec, message: impl Into<String>) -> WgslError {
    WgslError::for_kernel(WgslErrorCode::UniformLayoutMismatch, spec.kernel, message)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WgslErrorCode {
    Parse,
    Validation,
    MissingEntryPoint,
    WorkgroupMismatch,
    BindingMismatch,
    UniformLayoutMismatch,
    ForbiddenFeature,
    DuplicateKernel,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WgslError {
    code: WgslErrorCode,
    kernel: Option<KernelId>,
    message: String,
}

impl WgslError {
    fn for_kernel(code: WgslErrorCode, kernel: KernelId, message: impl Into<String>) -> Self {
        Self {
            code,
            kernel: Some(kernel),
            message: message.into(),
        }
    }

    #[must_use]
    pub const fn code(&self) -> WgslErrorCode {
        self.code
    }

    #[must_use]
    pub const fn kernel(&self) -> Option<KernelId> {
        self.kernel
    }
}

impl fmt::Display for WgslError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "WGSL error {:?}: {}", self.code, self.message)
    }
}

impl Error for WgslError {}
