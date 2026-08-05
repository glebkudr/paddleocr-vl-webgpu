use std::collections::BTreeSet;

use naga::{
    AddressSpace, ArraySize, Module, ScalarKind, StorageAccess, TypeInner, VectorSize,
    valid::{Capabilities, ValidationFlags, Validator},
};
use pvlc_runtime_core::{KernelId, KernelInvocation};
use pvlc_wgsl::{
    BindingKind, BindingSpec, WgslErrorCode, catalog, storage_read_write_variant, validate_catalog,
    validate_source_contract,
};

fn binding(binding: u32, kind: BindingKind) -> BindingSpec {
    BindingSpec {
        group: 0,
        binding,
        kind,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum HostValue {
    U32(u32),
    F32(f32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TestScalar {
    U32,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TestUniformField {
    name: &'static str,
    scalar: TestScalar,
    offset: u32,
}

const fn uniform(name: &'static str, scalar: TestScalar, offset: u32) -> TestUniformField {
    TestUniformField {
        name,
        scalar,
        offset,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TestBindingKind {
    StorageReadF32,
    StorageReadF16,
    StorageReadVec4F32,
    StorageReadVec4F16,
    StorageReadU32,
    StorageReadWriteF32,
    StorageReadWriteF16,
    StorageReadWriteVec4F16,
    Uniform,
}

fn expected_workgroup(kernel: KernelId) -> [u32; 3] {
    match kernel {
        KernelId::GemmF32
        | KernelId::VisionPatchProjectionF32
        | KernelId::VisionQkvFusedF32
        | KernelId::VisionQkvFusedF16Weights
        | KernelId::LinearProjectionF16 => [8, 8, 1],
        KernelId::GemvTiledF32 | KernelId::GemvTiledF16Weights => [256, 1, 1],
        KernelId::VisionAttentionF32 | KernelId::VisionAttentionF16 => [128, 1, 1],
        KernelId::LinearProjectionF16Weights => [8, 8, 1],
        KernelId::GemvF32
        | KernelId::LayerNormF32
        | KernelId::RmsNormF32
        | KernelId::RmsNormF16Weights
        | KernelId::LayerNormF16
        | KernelId::SiluF32
        | KernelId::GeluTanhF32
        | KernelId::GeluErfF32
        | KernelId::GeluTanhF16
        | KernelId::GeluErfF16
        | KernelId::RopeNeoxF32
        | KernelId::VisionRope2dF32
        | KernelId::VisionRope2dF16
        | KernelId::AddF32
        | KernelId::AddF16
        | KernelId::ProjectorMerge2x2F32
        | KernelId::ProjectorMerge2x2F16
        | KernelId::DecoderKvAppendF32
        | KernelId::DecoderGqaF32
        | KernelId::DecoderGqaSplitPartialF32
        | KernelId::DecoderGqaSplitMergeF32
        | KernelId::DecoderMropeF32
        | KernelId::DecoderSwigluF32
        | KernelId::DecoderPrefillGqaF32
        | KernelId::DecoderPrefillMropeF32
        | KernelId::DecoderKvAppendRangeF32 => [64, 1, 1],
    }
}

fn expected_uniform(kernel: KernelId) -> (Vec<TestUniformField>, u32) {
    use TestScalar::{F32, U32};
    match kernel {
        KernelId::GemmF32 => (
            vec![
                uniform("rows", U32, 0),
                uniform("inner", U32, 4),
                uniform("columns", U32, 8),
                uniform("padding", U32, 12),
            ],
            16,
        ),
        KernelId::GemvF32 | KernelId::GemvTiledF32 | KernelId::GemvTiledF16Weights => (
            vec![
                uniform("rows", U32, 0),
                uniform("columns", U32, 4),
                uniform("padding0", U32, 8),
                uniform("padding1", U32, 12),
            ],
            16,
        ),
        KernelId::LayerNormF32
        | KernelId::RmsNormF32
        | KernelId::RmsNormF16Weights
        | KernelId::LayerNormF16 => (
            vec![
                uniform("rows", U32, 0),
                uniform("width", U32, 4),
                uniform("epsilon", F32, 8),
                uniform("padding", U32, 12),
            ],
            16,
        ),
        KernelId::SiluF32
        | KernelId::GeluTanhF32
        | KernelId::GeluErfF32
        | KernelId::AddF32
        | KernelId::GeluTanhF16
        | KernelId::GeluErfF16
        | KernelId::AddF16 => (
            vec![
                uniform("length", U32, 0),
                uniform("padding0", U32, 4),
                uniform("padding1", U32, 8),
                uniform("padding2", U32, 12),
            ],
            16,
        ),
        KernelId::RopeNeoxF32 => (
            vec![
                uniform("rows", U32, 0),
                uniform("width", U32, 4),
                uniform("rotary_dim", U32, 8),
                uniform("base", F32, 12),
            ],
            16,
        ),
        KernelId::VisionAttentionF32 | KernelId::VisionAttentionF16 => (
            vec![
                uniform("tokens", U32, 0),
                uniform("heads", U32, 4),
                uniform("head_dim", U32, 8),
                uniform("segments", U32, 12),
            ],
            16,
        ),
        KernelId::VisionRope2dF32 | KernelId::VisionRope2dF16 => (
            vec![
                uniform("tokens", U32, 0),
                uniform("heads", U32, 4),
                uniform("head_dim", U32, 8),
                uniform("padding", U32, 12),
            ],
            16,
        ),
        KernelId::VisionPatchProjectionF32
        | KernelId::LinearProjectionF16Weights
        | KernelId::LinearProjectionF16 => (
            vec![
                uniform("patch_count", U32, 0),
                uniform("input_width", U32, 4),
                uniform("output_width", U32, 8),
                uniform("padding", U32, 12),
            ],
            16,
        ),
        KernelId::VisionQkvFusedF32 | KernelId::VisionQkvFusedF16Weights => (
            vec![
                uniform("tokens", U32, 0),
                uniform("input_width", U32, 4),
                uniform("output_width", U32, 8),
                uniform("plane_stride_elements", U32, 12),
            ],
            16,
        ),
        KernelId::ProjectorMerge2x2F32 | KernelId::ProjectorMerge2x2F16 => (
            vec![
                uniform("output_tokens", U32, 0),
                uniform("hidden_size", U32, 4),
                uniform("length", U32, 8),
                uniform("row_stride", U32, 12),
            ],
            16,
        ),
        KernelId::DecoderKvAppendF32 => (
            vec![
                uniform("prefix_tokens", U32, 0),
                uniform("key_value_heads", U32, 4),
                uniform("head_dim", U32, 8),
                uniform("cache_capacity", U32, 12),
            ],
            16,
        ),
        KernelId::DecoderGqaF32 => (
            vec![
                uniform("cache_tokens", U32, 0),
                uniform("query_heads", U32, 4),
                uniform("key_value_heads", U32, 8),
                uniform("head_dim", U32, 12),
            ],
            16,
        ),
        KernelId::DecoderGqaSplitPartialF32 | KernelId::DecoderGqaSplitMergeF32 => (
            vec![
                uniform("cache_tokens", U32, 0),
                uniform("chunk_count", U32, 4),
                uniform("padding0", U32, 8),
                uniform("padding1", U32, 12),
            ],
            16,
        ),
        KernelId::DecoderMropeF32 => (
            vec![
                uniform("position", U32, 0),
                uniform("rope_capacity", U32, 4),
                uniform("padding0", U32, 8),
                uniform("padding1", U32, 12),
            ],
            16,
        ),
        KernelId::DecoderSwigluF32 => (
            vec![
                uniform("length", U32, 0),
                uniform("padding0", U32, 4),
                uniform("padding1", U32, 8),
                uniform("padding2", U32, 12),
            ],
            16,
        ),
        KernelId::DecoderPrefillGqaF32 => (
            vec![
                uniform("tokens", U32, 0),
                uniform("query_heads", U32, 4),
                uniform("key_value_heads", U32, 8),
                uniform("head_dim", U32, 12),
            ],
            16,
        ),
        KernelId::DecoderPrefillMropeF32 => (
            vec![
                uniform("tokens", U32, 0),
                uniform("rope_capacity", U32, 4),
                uniform("padding0", U32, 8),
                uniform("padding1", U32, 12),
            ],
            16,
        ),
        KernelId::DecoderKvAppendRangeF32 => (
            vec![
                uniform("tokens", U32, 0),
                uniform("cache_capacity", U32, 4),
                uniform("padding0", U32, 8),
                uniform("padding1", U32, 12),
            ],
            16,
        ),
    }
}

fn expected_resources(kernel: KernelId) -> Vec<(&'static str, u32, TestBindingKind)> {
    use TestBindingKind::{
        StorageReadF16, StorageReadF32, StorageReadU32, StorageReadVec4F16, StorageReadVec4F32,
        StorageReadWriteF16, StorageReadWriteF32, StorageReadWriteVec4F16, Uniform,
    };
    match kernel {
        KernelId::GemmF32 => vec![
            ("left", 0, StorageReadF32),
            ("right", 1, StorageReadF32),
            ("output", 2, StorageReadWriteF32),
            ("params", 3, Uniform),
        ],
        KernelId::GemvF32 => vec![
            ("matrix", 0, StorageReadF32),
            ("vector", 1, StorageReadF32),
            ("output", 2, StorageReadWriteF32),
            ("params", 3, Uniform),
        ],
        KernelId::GemvTiledF32 => vec![
            ("matrix", 0, StorageReadVec4F32),
            ("vector", 1, StorageReadVec4F32),
            ("output", 2, StorageReadWriteF32),
            ("params", 3, Uniform),
        ],
        KernelId::LayerNormF32 => vec![
            ("input", 0, StorageReadF32),
            ("weight", 1, StorageReadF32),
            ("bias", 2, StorageReadF32),
            ("output", 3, StorageReadWriteF32),
            ("params", 4, Uniform),
        ],
        KernelId::LayerNormF16 => vec![
            ("input", 0, StorageReadVec4F16),
            ("weight", 1, StorageReadVec4F16),
            ("bias", 2, StorageReadVec4F16),
            ("output", 3, StorageReadWriteVec4F16),
            ("params", 4, Uniform),
        ],
        KernelId::RmsNormF32 => vec![
            ("input", 0, StorageReadF32),
            ("weight", 1, StorageReadF32),
            ("output", 2, StorageReadWriteF32),
            ("params", 3, Uniform),
        ],
        KernelId::RmsNormF16Weights => vec![
            ("input", 0, StorageReadF32),
            ("weight", 1, StorageReadF16),
            ("output", 2, StorageReadWriteF32),
            ("params", 3, Uniform),
        ],
        KernelId::GemvTiledF16Weights => vec![
            ("matrix", 0, StorageReadVec4F16),
            ("vector", 1, StorageReadVec4F32),
            ("output", 2, StorageReadWriteF32),
            ("params", 3, Uniform),
        ],
        KernelId::SiluF32 | KernelId::GeluTanhF32 | KernelId::GeluErfF32 => vec![
            ("input", 0, StorageReadF32),
            ("output", 1, StorageReadWriteF32),
            ("params", 2, Uniform),
        ],
        KernelId::GeluTanhF16 | KernelId::GeluErfF16 => vec![
            ("input", 0, StorageReadVec4F16),
            ("output", 1, StorageReadWriteVec4F16),
            ("params", 2, Uniform),
        ],
        KernelId::RopeNeoxF32 => vec![
            ("input", 0, StorageReadF32),
            ("positions", 1, StorageReadU32),
            ("output", 2, StorageReadWriteF32),
            ("params", 3, Uniform),
        ],
        KernelId::VisionAttentionF32 => vec![
            ("query", 0, StorageReadF32),
            ("key", 1, StorageReadF32),
            ("value", 2, StorageReadF32),
            ("cu_seqlens", 3, StorageReadU32),
            ("output", 4, StorageReadWriteF32),
            ("params", 5, Uniform),
        ],
        KernelId::VisionAttentionF16 => vec![
            ("query", 0, StorageReadVec4F16),
            ("key", 1, StorageReadVec4F16),
            ("value", 2, StorageReadVec4F16),
            ("cu_seqlens", 3, StorageReadU32),
            ("output", 4, StorageReadWriteVec4F16),
            ("params", 5, Uniform),
        ],
        KernelId::VisionRope2dF32 => vec![
            ("query", 0, StorageReadWriteF32),
            ("key", 1, StorageReadWriteF32),
            ("cos_table", 2, StorageReadF32),
            ("sin_table", 3, StorageReadF32),
            ("params", 4, Uniform),
        ],
        KernelId::VisionRope2dF16 => vec![
            ("query", 0, StorageReadWriteF16),
            ("key", 1, StorageReadWriteF16),
            ("cos_table", 2, StorageReadF32),
            ("sin_table", 3, StorageReadF32),
            ("params", 4, Uniform),
        ],
        KernelId::VisionPatchProjectionF32 => vec![
            ("input", 0, StorageReadF32),
            ("weight", 1, StorageReadF32),
            ("bias", 2, StorageReadF32),
            ("output", 3, StorageReadWriteF32),
            ("params", 4, Uniform),
        ],
        KernelId::LinearProjectionF16Weights => vec![
            ("input", 0, StorageReadVec4F32),
            ("weight", 1, StorageReadVec4F16),
            ("bias", 2, StorageReadVec4F32),
            ("output", 3, StorageReadWriteF32),
            ("params", 4, Uniform),
        ],
        KernelId::LinearProjectionF16 => vec![
            ("input", 0, StorageReadVec4F16),
            ("weight", 1, StorageReadVec4F16),
            ("bias", 2, StorageReadVec4F16),
            ("output", 3, StorageReadWriteVec4F16),
            ("params", 4, Uniform),
        ],
        KernelId::VisionQkvFusedF32 => vec![
            ("input", 0, StorageReadF32),
            ("query_weight", 1, StorageReadF32),
            ("query_bias", 2, StorageReadF32),
            ("key_weight", 3, StorageReadF32),
            ("key_bias", 4, StorageReadF32),
            ("value_weight", 5, StorageReadF32),
            ("value_bias", 6, StorageReadF32),
            ("output", 7, StorageReadWriteF32),
            ("params", 8, Uniform),
        ],
        KernelId::VisionQkvFusedF16Weights => vec![
            ("input", 0, StorageReadVec4F32),
            ("query_weight", 1, StorageReadVec4F16),
            ("query_bias", 2, StorageReadVec4F32),
            ("key_weight", 3, StorageReadVec4F16),
            ("key_bias", 4, StorageReadVec4F32),
            ("value_weight", 5, StorageReadVec4F16),
            ("value_bias", 6, StorageReadVec4F32),
            ("output", 7, StorageReadWriteF32),
            ("params", 8, Uniform),
        ],
        KernelId::AddF32 => vec![
            ("left", 0, StorageReadF32),
            ("right", 1, StorageReadF32),
            ("output", 2, StorageReadWriteF32),
            ("params", 3, Uniform),
        ],
        KernelId::AddF16 => vec![
            ("left", 0, StorageReadVec4F16),
            ("right", 1, StorageReadVec4F16),
            ("output", 2, StorageReadWriteVec4F16),
            ("params", 3, Uniform),
        ],
        KernelId::ProjectorMerge2x2F32 => vec![
            ("input", 0, StorageReadF32),
            ("source_token_indices", 1, StorageReadU32),
            ("output", 2, StorageReadWriteF32),
            ("params", 3, Uniform),
        ],
        KernelId::ProjectorMerge2x2F16 => vec![
            ("input", 0, StorageReadVec4F16),
            ("source_token_indices", 1, StorageReadU32),
            ("output", 2, StorageReadWriteVec4F16),
            ("params", 3, Uniform),
        ],
        KernelId::DecoderKvAppendF32 => vec![
            ("appended_key", 0, StorageReadF32),
            ("appended_value", 1, StorageReadF32),
            ("key_cache", 2, StorageReadWriteF32),
            ("value_cache", 3, StorageReadWriteF32),
            ("params", 4, Uniform),
        ],
        KernelId::DecoderGqaF32 => vec![
            ("query", 0, StorageReadF32),
            ("key_cache", 1, StorageReadF32),
            ("value_cache", 2, StorageReadF32),
            ("output", 3, StorageReadWriteF32),
            ("params", 4, Uniform),
        ],
        KernelId::DecoderGqaSplitPartialF32 => vec![
            ("query", 0, StorageReadF32),
            ("key_cache", 1, StorageReadF32),
            ("value_cache", 2, StorageReadF32),
            ("partials", 3, StorageReadWriteF32),
            ("params", 4, Uniform),
        ],
        KernelId::DecoderGqaSplitMergeF32 => vec![
            ("partials", 0, StorageReadF32),
            ("key_cache", 1, StorageReadF32),
            ("value_cache", 2, StorageReadF32),
            ("output", 3, StorageReadWriteF32),
            ("params", 4, Uniform),
        ],
        KernelId::DecoderMropeF32 => vec![
            ("query", 0, StorageReadF32),
            ("key", 1, StorageReadF32),
            ("rope_cos", 2, StorageReadF32),
            ("rope_sin", 3, StorageReadF32),
            ("output_query", 4, StorageReadWriteF32),
            ("output_key", 5, StorageReadWriteF32),
            ("params", 6, Uniform),
        ],
        KernelId::DecoderSwigluF32 => vec![
            ("gate", 0, StorageReadF32),
            ("up", 1, StorageReadF32),
            ("output", 2, StorageReadWriteF32),
            ("params", 3, Uniform),
        ],
        KernelId::DecoderPrefillGqaF32 => vec![
            ("query", 0, StorageReadF32),
            ("key_cache", 1, StorageReadF32),
            ("value_cache", 2, StorageReadF32),
            ("output", 3, StorageReadWriteF32),
            ("params", 4, Uniform),
        ],
        KernelId::DecoderPrefillMropeF32 => vec![
            ("query", 0, StorageReadF32),
            ("key", 1, StorageReadF32),
            ("rope_cos", 2, StorageReadF32),
            ("rope_sin", 3, StorageReadF32),
            ("output_query", 4, StorageReadWriteF32),
            ("output_key", 5, StorageReadWriteF32),
            ("params", 6, Uniform),
        ],
        KernelId::DecoderKvAppendRangeF32 => vec![
            ("appended_key", 0, StorageReadF32),
            ("appended_value", 1, StorageReadF32),
            ("key_cache", 2, StorageReadWriteF32),
            ("value_cache", 3, StorageReadWriteF32),
            ("params", 4, Uniform),
        ],
    }
}

fn assert_storage_scalar(
    module: &Module,
    ty: naga::Handle<naga::Type>,
    expected_kind: ScalarKind,
    expected_width: u8,
) {
    let TypeInner::Struct { members, .. } = &module.types[ty].inner else {
        panic!("storage binding must use a one-member runtime-array struct")
    };
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].name.as_deref(), Some("data"));
    assert_eq!(members[0].offset, 0);
    let TypeInner::Array {
        base,
        size: ArraySize::Dynamic,
        stride,
    } = module.types[members[0].ty].inner
    else {
        panic!("storage data must be a tightly packed runtime array")
    };
    assert_eq!(stride, u32::from(expected_width));
    let TypeInner::Scalar(scalar) = module.types[base].inner else {
        panic!("storage array element must be scalar")
    };
    assert_eq!(scalar.kind, expected_kind);
    assert_eq!(scalar.width, expected_width);
}

fn assert_storage_vec4_float(module: &Module, ty: naga::Handle<naga::Type>, expected_width: u8) {
    let TypeInner::Struct { members, .. } = &module.types[ty].inner else {
        panic!("storage binding must use a one-member runtime-array struct")
    };
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].name.as_deref(), Some("data"));
    assert_eq!(members[0].offset, 0);
    let TypeInner::Array {
        base,
        size: ArraySize::Dynamic,
        stride,
    } = module.types[members[0].ty].inner
    else {
        panic!("vec4 storage data must be a tightly packed runtime array")
    };
    assert_eq!(stride, 4 * u32::from(expected_width));
    let TypeInner::Vector {
        size: VectorSize::Quad,
        scalar,
    } = module.types[base].inner
    else {
        panic!("storage array element must be vec4")
    };
    assert_eq!(scalar.kind, ScalarKind::Float);
    assert_eq!(scalar.width, expected_width);
}

fn fixed_type_bytes(module: &Module, ty: naga::Handle<naga::Type>) -> usize {
    match &module.types[ty].inner {
        TypeInner::Scalar(scalar) | TypeInner::Atomic(scalar) => usize::from(scalar.width),
        TypeInner::Vector { size, scalar } => u32::from(*size) as usize * usize::from(scalar.width),
        TypeInner::Array {
            size: ArraySize::Constant(size),
            stride,
            ..
        } => size.get() as usize * *stride as usize,
        TypeInner::Struct { span, .. } => *span as usize,
        other => panic!("unsupported fixed internal type: {other:?}"),
    }
}

fn independently_reflect_and_assert(module_source: &str, kernel: KernelId) -> Module {
    let module = naga::front::wgsl::parse_str(module_source).unwrap();
    assert_eq!(module.entry_points.len(), 1);
    let entry = &module.entry_points[0];
    assert_eq!(entry.name, "main");
    assert_eq!(entry.stage, naga::ShaderStage::Compute);
    assert_eq!(entry.workgroup_size, expected_workgroup(kernel));
    let expected = expected_resources(kernel);
    let mut actual = Vec::new();
    let mut internal = Vec::new();
    for (_, global) in module.global_variables.iter() {
        if let Some(resource) = &global.binding {
            actual.push((resource.binding, global));
        } else {
            internal.push(global);
        }
    }
    assert_eq!(actual.len(), expected.len());
    match kernel {
        KernelId::GemvTiledF32 | KernelId::GemvTiledF16Weights => {
            assert_eq!(
                internal
                    .iter()
                    .map(|global| global.name.as_deref())
                    .collect::<Vec<_>>(),
                [Some("shared_vector"), Some("partials")]
            );
            assert!(
                internal
                    .iter()
                    .all(|global| global.space == AddressSpace::WorkGroup)
            );
        }
        KernelId::VisionAttentionF32 | KernelId::VisionAttentionF16 => {
            assert!(!internal.is_empty());
            assert!(
                internal
                    .iter()
                    .all(|global| global.space == AddressSpace::WorkGroup),
                "vision attention may retain bounded workgroup tiles but no private globals"
            );
            let workgroup_bytes = internal
                .iter()
                .map(|global| fixed_type_bytes(&module, global.ty))
                .sum::<usize>();
            assert!(
                workgroup_bytes <= 16_384,
                "vision attention requires {workgroup_bytes} workgroup bytes"
            );
        }
        KernelId::VisionPatchProjectionF32
        | KernelId::LinearProjectionF16Weights
        | KernelId::LinearProjectionF16 => {
            assert_eq!(
                internal
                    .iter()
                    .map(|global| global.name.as_deref())
                    .collect::<Vec<_>>(),
                [Some("input_tile"), Some("weight_tile")]
            );
            assert!(
                internal
                    .iter()
                .all(|global| global.space == AddressSpace::WorkGroup)
            );
        }
        KernelId::VisionQkvFusedF16Weights => {
            assert_eq!(
                internal
                    .iter()
                    .map(|global| global.name.as_deref())
                    .collect::<Vec<_>>(),
                [
                    Some("input_tile"),
                    Some("query_weight_tile"),
                    Some("key_weight_tile"),
                    Some("value_weight_tile"),
                ]
            );
            assert!(
                internal
                    .iter()
                    .all(|global| global.space == AddressSpace::WorkGroup)
            );
        }
        _ => assert!(internal.is_empty()),
    }
    actual.sort_by_key(|(binding, _)| *binding);

    for ((actual_binding, global), (name, binding, kind)) in actual.into_iter().zip(expected) {
        assert_eq!(actual_binding, binding);
        assert_eq!(global.binding.as_ref().unwrap().group, 0);
        assert_eq!(global.name.as_deref(), Some(name));
        match kind {
            TestBindingKind::StorageReadF32 => {
                assert_eq!(
                    global.space,
                    AddressSpace::Storage {
                        access: StorageAccess::LOAD
                    }
                );
                assert_storage_scalar(&module, global.ty, ScalarKind::Float, 4);
            }
            TestBindingKind::StorageReadF16 => {
                assert_eq!(
                    global.space,
                    AddressSpace::Storage {
                        access: StorageAccess::LOAD
                    }
                );
                assert_storage_scalar(&module, global.ty, ScalarKind::Float, 2);
            }
            TestBindingKind::StorageReadVec4F32 => {
                assert_eq!(
                    global.space,
                    AddressSpace::Storage {
                        access: StorageAccess::LOAD
                    }
                );
                assert_storage_vec4_float(&module, global.ty, 4);
            }
            TestBindingKind::StorageReadVec4F16 => {
                assert_eq!(
                    global.space,
                    AddressSpace::Storage {
                        access: StorageAccess::LOAD
                    }
                );
                assert_storage_vec4_float(&module, global.ty, 2);
            }
            TestBindingKind::StorageReadU32 => {
                assert_eq!(
                    global.space,
                    AddressSpace::Storage {
                        access: StorageAccess::LOAD
                    }
                );
                assert_storage_scalar(&module, global.ty, ScalarKind::Uint, 4);
            }
            TestBindingKind::StorageReadWriteF32 => {
                assert_eq!(
                    global.space,
                    AddressSpace::Storage {
                        access: StorageAccess::LOAD | StorageAccess::STORE
                    }
                );
                assert_storage_scalar(&module, global.ty, ScalarKind::Float, 4);
            }
            TestBindingKind::StorageReadWriteF16 => {
                assert_eq!(
                    global.space,
                    AddressSpace::Storage {
                        access: StorageAccess::LOAD | StorageAccess::STORE
                    }
                );
                assert_storage_scalar(&module, global.ty, ScalarKind::Float, 2);
            }
            TestBindingKind::StorageReadWriteVec4F16 => {
                assert_eq!(
                    global.space,
                    AddressSpace::Storage {
                        access: StorageAccess::LOAD | StorageAccess::STORE
                    }
                );
                assert_storage_vec4_float(&module, global.ty, 2);
            }
            TestBindingKind::Uniform => {
                assert_eq!(global.space, AddressSpace::Uniform);
                let (expected_fields, expected_span) = expected_uniform(kernel);
                let TypeInner::Struct { members, span } = &module.types[global.ty].inner else {
                    panic!("uniform binding must be a struct")
                };
                assert_eq!(*span, expected_span);
                assert_eq!(members.len(), expected_fields.len());
                for (member, expected) in members.iter().zip(expected_fields) {
                    assert_eq!(member.name.as_deref(), Some(expected.name));
                    assert_eq!(member.offset, expected.offset);
                    let TypeInner::Scalar(scalar) = module.types[member.ty].inner else {
                        panic!("uniform member must be scalar")
                    };
                    let expected_kind = match expected.scalar {
                        TestScalar::U32 => ScalarKind::Uint,
                        TestScalar::F32 => ScalarKind::Float,
                    };
                    assert_eq!(scalar.kind, expected_kind);
                    assert_eq!(scalar.width, 4);
                }
            }
        }
    }
    module
}

#[test]
fn catalog_contains_each_m2_fp32_primitive_exactly_once_with_stable_abi() {
    let expected = [
        (
            KernelId::GemmF32,
            [8, 8, 1],
            vec![
                binding(0, BindingKind::StorageReadF32),
                binding(1, BindingKind::StorageReadF32),
                binding(2, BindingKind::StorageReadWriteF32),
                binding(3, BindingKind::Uniform),
            ],
        ),
        (
            KernelId::GemvF32,
            [64, 1, 1],
            vec![
                binding(0, BindingKind::StorageReadF32),
                binding(1, BindingKind::StorageReadF32),
                binding(2, BindingKind::StorageReadWriteF32),
                binding(3, BindingKind::Uniform),
            ],
        ),
        (
            KernelId::LayerNormF32,
            [64, 1, 1],
            vec![
                binding(0, BindingKind::StorageReadF32),
                binding(1, BindingKind::StorageReadF32),
                binding(2, BindingKind::StorageReadF32),
                binding(3, BindingKind::StorageReadWriteF32),
                binding(4, BindingKind::Uniform),
            ],
        ),
        (
            KernelId::RmsNormF32,
            [64, 1, 1],
            vec![
                binding(0, BindingKind::StorageReadF32),
                binding(1, BindingKind::StorageReadF32),
                binding(2, BindingKind::StorageReadWriteF32),
                binding(3, BindingKind::Uniform),
            ],
        ),
        (
            KernelId::SiluF32,
            [64, 1, 1],
            vec![
                binding(0, BindingKind::StorageReadF32),
                binding(1, BindingKind::StorageReadWriteF32),
                binding(2, BindingKind::Uniform),
            ],
        ),
        (
            KernelId::GeluTanhF32,
            [64, 1, 1],
            vec![
                binding(0, BindingKind::StorageReadF32),
                binding(1, BindingKind::StorageReadWriteF32),
                binding(2, BindingKind::Uniform),
            ],
        ),
        (
            KernelId::RopeNeoxF32,
            [64, 1, 1],
            vec![
                binding(0, BindingKind::StorageReadF32),
                binding(1, BindingKind::StorageReadU32),
                binding(2, BindingKind::StorageReadWriteF32),
                binding(3, BindingKind::Uniform),
            ],
        ),
    ];

    assert_eq!(catalog().len(), KernelId::M2_PRIMITIVES.len());
    let unique: BTreeSet<_> = catalog().iter().map(|module| module.spec.kernel).collect();
    assert_eq!(unique.len(), KernelId::M2_PRIMITIVES.len());
    for (kernel, workgroup_size, bindings) in expected {
        let actual = pvlc_wgsl::module(kernel).expect("M2 kernel must be in the full catalog");
        assert_eq!(actual.spec.kernel, kernel);
        assert_eq!(actual.spec.entry_point, "main");
        assert_eq!(actual.spec.workgroup_size, workgroup_size);
        assert_eq!(actual.spec.bindings, bindings);
        let (uniform_fields, uniform_span) = expected_uniform(kernel);
        assert_eq!(actual.spec.uniform_fields.len(), uniform_fields.len());
        for (actual_field, expected_field) in actual.spec.uniform_fields.iter().zip(uniform_fields)
        {
            assert_eq!(actual_field.name, expected_field.name);
            assert_eq!(actual_field.offset, expected_field.offset);
            let actual_scalar = match actual_field.scalar {
                pvlc_wgsl::UniformScalar::U32 => TestScalar::U32,
                pvlc_wgsl::UniformScalar::F32 => TestScalar::F32,
            };
            assert_eq!(actual_scalar, expected_field.scalar);
        }
        assert_eq!(actual.spec.uniform_span, uniform_span);
        assert!(actual.spec.required_features.is_empty());
    }
}

#[test]
fn every_builtin_shader_independently_parses_and_typechecks_in_naga() {
    validate_catalog().expect("the complete generated catalog must satisfy its ABI contract");
    for kernel in catalog() {
        let module = independently_reflect_and_assert(kernel.source, kernel.spec.kernel);
        let capabilities = if kernel.spec.required_features.contains(&"shader_f16") {
            Capabilities::SHADER_FLOAT16
        } else {
            Capabilities::empty()
        };
        Validator::new(ValidationFlags::all(), capabilities)
            .validate(&module)
            .unwrap_or_else(|error| {
                panic!("{} failed Naga validation: {error}", kernel.spec.kernel)
            });
    }
    let attention = pvlc_wgsl::module(KernelId::VisionAttentionF32).unwrap();
    let module = independently_reflect_and_assert(attention.source, attention.spec.kernel);
    Validator::new(ValidationFlags::all(), Capabilities::empty())
        .validate(&module)
        .expect("vision attention failed independent Naga validation");
}

#[test]
fn fp16_shaders_independently_reflect_exact_physical_abis() {
    for kernel in [
        KernelId::RmsNormF16Weights,
        KernelId::GemvTiledF16Weights,
        KernelId::LinearProjectionF16Weights,
        KernelId::VisionQkvFusedF16Weights,
        KernelId::LayerNormF16,
        KernelId::LinearProjectionF16,
        KernelId::VisionAttentionF16,
        KernelId::AddF16,
        KernelId::GeluTanhF16,
        KernelId::VisionRope2dF16,
        KernelId::ProjectorMerge2x2F16,
        KernelId::GeluErfF16,
    ] {
        let kernel = pvlc_wgsl::module(kernel).expect("FP16 kernel must be in the full catalog");
        assert_eq!(kernel.spec.required_features, ["shader_f16"]);
        let module = independently_reflect_and_assert(kernel.source, kernel.spec.kernel);
        Validator::new(ValidationFlags::all(), Capabilities::SHADER_FLOAT16)
            .validate(&module)
            .unwrap_or_else(|error| {
                panic!("{} failed Naga validation: {error}", kernel.spec.kernel)
            });
    }
}

#[test]
fn scalar_activation_shaders_flatten_bounded_two_dimensional_dispatches_without_aliasing() {
    for kernel in [KernelId::SiluF32, KernelId::GeluTanhF32] {
        let source = pvlc_wgsl::module(kernel).unwrap().source;
        assert!(source.contains(
            "let row_stride = select(params.length, params.padding0, params.padding0 != 0u);"
        ));
        assert!(source.contains("let index = global_id.x + global_id.y * row_stride;"));
        assert!(source.contains("if index >= params.length {"));
        assert!(!source.contains("global_id.x %"));
    }
}

#[test]
fn host_uniform_bytes_decode_as_every_independently_reflected_field() {
    let cases: [(KernelInvocation, &[HostValue]); 7] = [
        (
            KernelInvocation::GemmF32 {
                rows: 2,
                inner: 3,
                columns: 9,
                left: vec![1.0; 6],
                right: vec![1.0; 27],
            },
            &[
                HostValue::U32(2),
                HostValue::U32(3),
                HostValue::U32(9),
                HostValue::U32(0),
            ],
        ),
        (
            KernelInvocation::GemvF32 {
                rows: 65,
                columns: 3,
                matrix: vec![1.0; 195],
                vector: vec![1.0; 3],
            },
            &[
                HostValue::U32(65),
                HostValue::U32(3),
                HostValue::U32(0),
                HostValue::U32(0),
            ],
        ),
        (
            KernelInvocation::LayerNormF32 {
                rows: 3,
                width: 5,
                input: vec![1.0; 15],
                weight: vec![1.0; 5],
                bias: vec![0.0; 5],
                epsilon: 0.5,
            },
            &[
                HostValue::U32(3),
                HostValue::U32(5),
                HostValue::F32(0.5),
                HostValue::U32(0),
            ],
        ),
        (
            KernelInvocation::RmsNormF32 {
                rows: 3,
                width: 5,
                input: vec![1.0; 15],
                weight: vec![1.0; 5],
                epsilon: 0.25,
            },
            &[
                HostValue::U32(3),
                HostValue::U32(5),
                HostValue::F32(0.25),
                HostValue::U32(0),
            ],
        ),
        (
            KernelInvocation::SiluF32 {
                values: vec![1.0; 65],
            },
            &[
                HostValue::U32(65),
                HostValue::U32(0),
                HostValue::U32(0),
                HostValue::U32(0),
            ],
        ),
        (
            KernelInvocation::GeluTanhF32 {
                values: vec![1.0; 129],
            },
            &[
                HostValue::U32(129),
                HostValue::U32(0),
                HostValue::U32(0),
                HostValue::U32(0),
            ],
        ),
        (
            KernelInvocation::RopeNeoxF32 {
                rows: 3,
                width: 10,
                rotary_dim: 8,
                positions: vec![0, 1, 7],
                base: 500_000.0,
                values: vec![1.0; 30],
            },
            &[
                HostValue::U32(3),
                HostValue::U32(10),
                HostValue::U32(8),
                HostValue::F32(500_000.0),
            ],
        ),
    ];

    for (invocation, expected_values) in cases {
        let kernel = invocation.kernel_id();
        let module = pvlc_wgsl::module(kernel).unwrap();
        independently_reflect_and_assert(module.source, kernel);
        let bytes = invocation.uniform_bytes().unwrap();
        let (fields, span) = expected_uniform(kernel);
        assert_eq!(bytes.len(), span as usize);
        for ((field, expected), index) in fields.iter().zip(expected_values).zip(0_usize..) {
            let raw: [u8; 4] = bytes[field.offset as usize..field.offset as usize + 4]
                .try_into()
                .unwrap();
            let actual = match field.scalar {
                TestScalar::U32 => HostValue::U32(u32::from_le_bytes(raw)),
                TestScalar::F32 => HostValue::F32(f32::from_le_bytes(raw)),
            };
            assert_eq!(actual, *expected, "uniform field {index}: {}", field.name);
        }
    }
}

#[test]
fn source_contract_rejects_abi_entrypoint_workgroup_and_feature_drift() {
    let module = catalog()
        .iter()
        .find(|module| module.spec.kernel == KernelId::GemmF32)
        .unwrap();

    let wrong_binding = module.source.replacen("@binding(0)", "@binding(7)", 1);
    let error = validate_source_contract(&module.spec, &wrong_binding).unwrap_err();
    assert_eq!(error.code(), WgslErrorCode::BindingMismatch);
    assert_eq!(error.kernel(), Some(KernelId::GemmF32));

    let wrong_access = module
        .source
        .replacen("var<storage, read>", "var<storage, read_write>", 1);
    let error = validate_source_contract(&module.spec, &wrong_access).unwrap_err();
    assert_eq!(error.code(), WgslErrorCode::BindingMismatch);

    let wrong_storage_scalar = module.source.replacen("array<f32>", "array<u32>", 1);
    let error = validate_source_contract(&module.spec, &wrong_storage_scalar).unwrap_err();
    assert_eq!(error.code(), WgslErrorCode::BindingMismatch);

    let wrong_uniform_scalar = module.source.replacen("rows: u32", "rows: f32", 1);
    let error = validate_source_contract(&module.spec, &wrong_uniform_scalar).unwrap_err();
    assert_eq!(error.code(), WgslErrorCode::UniformLayoutMismatch);

    assert!(module.source.contains("rows: u32,\n    inner: u32"));
    let wrong_uniform_order = module.source.replacen(
        "rows: u32,\n    inner: u32",
        "inner: u32,\n    rows: u32",
        1,
    );
    let error = validate_source_contract(&module.spec, &wrong_uniform_order).unwrap_err();
    assert_eq!(error.code(), WgslErrorCode::UniformLayoutMismatch);

    let extra_binding = module.source.replacen(
        "@compute",
        "@group(0) @binding(9) var<storage, read> extra: F32Buffer;\n@compute",
        1,
    );
    let error = validate_source_contract(&module.spec, &extra_binding).unwrap_err();
    assert_eq!(error.code(), WgslErrorCode::BindingMismatch);

    let wrong_entry = module.source.replacen("fn main", "fn renamed", 1);
    let error = validate_source_contract(&module.spec, &wrong_entry).unwrap_err();
    assert_eq!(error.code(), WgslErrorCode::MissingEntryPoint);

    let wrong_workgroup =
        module
            .source
            .replacen("@workgroup_size(8, 8, 1)", "@workgroup_size(4, 8, 1)", 1);
    let error = validate_source_contract(&module.spec, &wrong_workgroup).unwrap_err();
    assert_eq!(error.code(), WgslErrorCode::WorkgroupMismatch);

    let forbidden_feature = format!("enable f16;\n{}", module.source);
    let error = validate_source_contract(&module.spec, &forbidden_feature).unwrap_err();
    assert_eq!(error.code(), WgslErrorCode::ForbiddenFeature);
}

#[test]
fn malformed_wgsl_and_type_errors_are_distinguished_from_contract_drift() {
    let module = &catalog()[0];
    let parse_error = validate_source_contract(&module.spec, "this is not WGSL").unwrap_err();
    assert_eq!(parse_error.code(), WgslErrorCode::Parse);

    let type_error = r#"
struct F32Buffer { data: array<f32> }
struct Params { rows: u32, inner: u32, columns: u32, padding: u32 }
@group(0) @binding(0) var<storage, read> left: F32Buffer;
@group(0) @binding(1) var<storage, read> right: F32Buffer;
@group(0) @binding(2) var<storage, read_write> output: F32Buffer;
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(8, 8, 1)
fn main() { output.data[0] = dpdx(f32(params.rows)); }
"#;
    let parsed = naga::front::wgsl::parse_str(type_error)
        .expect("the negative fixture must reach semantic validation, not fail parsing");
    assert!(
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&parsed)
            .is_err(),
        "derivatives are invalid in a compute entry point"
    );
    let error = validate_source_contract(&module.spec, type_error).unwrap_err();
    assert_eq!(error.code(), WgslErrorCode::Validation);
}

#[test]
fn fp32_baseline_has_no_hidden_optional_feature_or_precision_dependency() {
    for module in catalog() {
        assert!(!module.source.contains("f16"));
        assert!(!module.source.contains("subgroup"));
        assert!(!module.source.contains("atomic<"));
        assert!(!module.source.contains("override "));
        assert!(module.source.is_ascii());
        assert!(module.source.ends_with('\n'));
        assert_eq!(module.source, module.source_for_build());
    }
}

#[test]
fn catalog_lookup_is_total_for_known_ids_and_rejects_unknown_source_claims() {
    for module in catalog() {
        assert_eq!(pvlc_wgsl::module(module.spec.kernel).unwrap(), module);
    }

    let mut duplicate = catalog().to_vec();
    duplicate.push(catalog()[0].clone());
    let error = pvlc_wgsl::validate_modules(&duplicate).unwrap_err();
    assert_eq!(error.code(), WgslErrorCode::DuplicateKernel);
}

#[test]
fn storage_read_write_variant_is_deterministic_for_fixed_vision_kernels_and_preserves_original_source()
 {
    for kernel in [
        KernelId::LayerNormF32,
        KernelId::VisionPatchProjectionF32,
        KernelId::VisionAttentionF32,
        KernelId::GeluTanhF32,
        KernelId::AddF32,
    ] {
        let module = pvlc_wgsl::module(kernel).unwrap();
        let original = module.source.to_owned();
        let first = storage_read_write_variant(&module.spec, module.source).unwrap();
        let second = storage_read_write_variant(&module.spec, module.source).unwrap();

        assert_eq!(module.source, original);
        assert_eq!(first, second);
        assert_ne!(first, original);
        assert!(!first.contains("var<storage, read>;"));
        assert!(first.contains("var<storage, read_write>"));

        let parsed = naga::front::wgsl::parse_str(&first).unwrap();
        Validator::new(ValidationFlags::all(), Capabilities::empty())
            .validate(&parsed)
            .unwrap();
    }
}

#[test]
fn static_vision_shader_blake3_anchors_are_stable() {
    for (name, kernel, expected_blake3) in [
        (
            "add_f32",
            KernelId::AddF32,
            "82dc89c6d3538860903154012c1e02ccb586c2f34b54b2c5d0a6681b55d0375a",
        ),
        (
            "gelu_tanh_f32",
            KernelId::GeluTanhF32,
            "6e8793a16683a469ed3b51da5015b6e8a731970a320f84b494205ab49f4a89b2",
        ),
        (
            "layer_norm_f32",
            KernelId::LayerNormF32,
            "a2746dcc4e84b7ef93fd0d0605c22369a75b19516c4b9b613c06bffb1fadcb93",
        ),
    ] {
        let module = pvlc_wgsl::module(kernel).unwrap();
        let source = storage_read_write_variant(&module.spec, module.source).unwrap();
        assert_eq!(
            blake3::hash(source.as_bytes()).to_hex().to_string(),
            expected_blake3,
            "static all-read_write {name} WGSL hash drifted"
        );
    }
}

#[test]
fn performance_kernel_hashes_leave_the_rejected_scalar_and_reduction_sources() {
    for (kernel, rejected_blake3) in [
        (
            KernelId::VisionAttentionF32,
            "db16ca2882420844ba6a28682b14eb48026da2397a5e2afd89ef496622ff80ca",
        ),
        (
            KernelId::VisionPatchProjectionF32,
            "31ea54eb47cbc5b28e64fa52e56981ce61d8e7b3cc34fcab27c18235b5da045d",
        ),
    ] {
        let module = pvlc_wgsl::module(kernel).unwrap();
        let first = storage_read_write_variant(&module.spec, module.source).unwrap();
        let second = storage_read_write_variant(&module.spec, module.source).unwrap();
        let hash = blake3::hash(first.as_bytes()).to_hex().to_string();
        assert_eq!(
            first, second,
            "{kernel} source transformation must be deterministic"
        );
        assert_ne!(
            hash, rejected_blake3,
            "{kernel} still hashes to the rejected scalar/reduction implementation",
        );
    }
}

#[test]
fn storage_read_write_variant_rejects_semantically_valid_spacing_that_breaks_exact_replacement_count()
 {
    let module = pvlc_wgsl::module(KernelId::VisionPatchProjectionF32).unwrap();
    let spaced = module
        .source
        .replacen("var<storage, read>", "var < storage, read >", 1);
    assert_ne!(spaced, module.source);
    validate_source_contract(&module.spec, &spaced).unwrap();

    let error = storage_read_write_variant(&module.spec, &spaced).unwrap_err();
    assert_eq!(error.code(), WgslErrorCode::BindingMismatch);
}
