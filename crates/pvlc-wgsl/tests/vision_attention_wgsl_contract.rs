use std::collections::BTreeSet;

use naga::{
    AddressSpace, ArraySize, Expression, Handle, Literal, Module, StorageAccess, Type, TypeInner,
};
use pvlc_runtime_core::KernelId;
use pvlc_wgsl::{BindingKind, UniformScalar, catalog, full_catalog, validate_source_contract};

#[test]
fn m3_vision_attention_has_a_fixed_baseline_webgpu_abi() {
    assert_eq!(
        catalog().len(),
        7,
        "the frozen M2 browser catalog is additive"
    );
    assert_eq!(full_catalog().len(), KernelId::ALL.len());
    assert_eq!(
        full_catalog()
            .iter()
            .map(|module| module.spec.kernel)
            .collect::<Vec<_>>(),
        KernelId::ALL
    );
    let unique: BTreeSet<_> = full_catalog()
        .iter()
        .map(|module| module.spec.kernel)
        .collect();
    assert_eq!(unique.len(), full_catalog().len());

    let module = full_catalog()
        .iter()
        .find(|module| module.spec.kernel == KernelId::VisionAttentionF32)
        .unwrap();
    assert_eq!(module.spec.entry_point, "main");
    assert_eq!(module.spec.workgroup_size, [128, 1, 1]);
    assert_eq!(
        module
            .spec
            .bindings
            .iter()
            .map(|binding| binding.kind)
            .collect::<Vec<_>>(),
        vec![
            BindingKind::StorageReadF32,
            BindingKind::StorageReadF32,
            BindingKind::StorageReadF32,
            BindingKind::StorageReadU32,
            BindingKind::StorageReadWriteF32,
            BindingKind::Uniform,
        ]
    );
    assert_eq!(
        module
            .spec
            .uniform_fields
            .iter()
            .map(|field| (field.name, field.scalar, field.offset))
            .collect::<Vec<_>>(),
        vec![
            ("tokens", UniformScalar::U32, 0),
            ("heads", UniformScalar::U32, 4),
            ("head_dim", UniformScalar::U32, 8),
            ("segments", UniformScalar::U32, 12),
        ]
    );
    assert_eq!(module.spec.uniform_span, 16);
    assert!(module.spec.required_features.is_empty());
    validate_source_contract(&module.spec, module.source).unwrap();
}

#[test]
fn reflected_shader_uses_one_query_per_lane_and_bounded_shared_key_value_tiles() {
    let builtin = pvlc_wgsl::module(KernelId::VisionAttentionF32).unwrap();
    let module = naga::front::wgsl::parse_str(builtin.source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    assert!(
        module.functions.is_empty(),
        "the global-read audit requires the complete attention dataflow in the entry point",
    );

    let bound_globals: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(_, global)| global.binding.as_ref().map(|binding| (global, binding)))
        .collect();
    assert_eq!(bound_globals.len(), 6);
    let workgroup_bytes: usize = module
        .global_variables
        .iter()
        .filter(|(_, global)| matches!(global.space, AddressSpace::WorkGroup))
        .map(|(_, global)| fixed_type_bytes(&module, global.ty))
        .sum();
    assert!(
        workgroup_bytes <= 16_384,
        "attention tiles must fit the portable WebGPU workgroup limit, got {workgroup_bytes} bytes"
    );
    assert_eq!(constant_u32(&module, "QUERY_TILE"), 128);
    assert_eq!(constant_u32(&module, "KEY_STEP"), 16);
    assert_eq!(constant_u32(&module, "MAX_HEAD_VECTORS"), 18);
    assert_eq!(constant_u32(&module, "WORKGROUP_SIZE"), 128);
    assert!(
        builtin
            .source
            .contains("let query_token = workgroup_id.x * QUERY_TILE + local_index;")
    );
    assert!(builtin.source.contains("let head = workgroup_id.y;"));
    assert!(builtin.source.contains("key_start = key_start + KEY_STEP"));
    assert_eq!(
        builtin.source.matches("workgroupBarrier();").count(),
        2,
        "a key step needs one overwrite fence and one cache-ready fence, not per-query reductions",
    );
    assert!(builtin.source.contains("var scores: array<f32, 16>;"));
    assert!(builtin.source.contains("var query_vectors: array<vec4<f32>, 18>;"));
    assert!(builtin.source.contains("var attention_output: array<vec4<f32>, 18>;"));
    assert!(!builtin.source.contains("tile_scores"));
    assert!(!builtin.source.contains("reductions"));
    assert!(!builtin.source.contains("query_cache"));
    assert!(builtin.source.contains("if query_token < candidate_end"));
    assert!(
        builtin
            .source
            .contains("key_token >= segment_start && key_token < segment_end")
    );

    let storage: Vec<_> = bound_globals
        .iter()
        .filter(|(global, _)| matches!(global.space, AddressSpace::Storage { .. }))
        .collect();
    assert_eq!(storage.len(), 5, "Q/K/V, cu_seqlens, and output only");
    let writable = storage
        .iter()
        .filter(|(global, _)| {
            global.space
                == (AddressSpace::Storage {
                    access: StorageAccess::LOAD | StorageAccess::STORE,
                })
        })
        .count();
    assert_eq!(writable, 1, "only the final O tensor may be writable");

    let private_global_bytes: usize = module
        .global_variables
        .iter()
        .filter(|(_, global)| matches!(global.space, AddressSpace::Private))
        .map(|(_, global)| fixed_type_bytes(&module, global.ty))
        .sum();
    assert_eq!(
        private_global_bytes, 0,
        "attention must not hide a per-invocation head-sized accumulator in private globals"
    );
    let entry_local_bytes: usize = module.entry_points[0]
        .function
        .local_variables
        .iter()
        .map(|(_, local)| fixed_type_bytes(&module, local.ty))
        .sum();
    let helper_local_bytes: usize = module
        .functions
        .iter()
        .flat_map(|(_, function)| function.local_variables.iter())
        .map(|(_, local)| fixed_type_bytes(&module, local.ty))
        .sum();
    let local_bytes = entry_local_bytes + helper_local_bytes;
    assert!(
        local_bytes <= 768,
        "one-query FlashAttention state must remain bounded, got {local_bytes} bytes"
    );
}

#[test]
fn reflected_shader_loads_q_once_and_reuses_each_shared_k_v_step_for_128_queries() {
    let builtin = pvlc_wgsl::module(KernelId::VisionAttentionF32).unwrap();
    let module = naga::front::wgsl::parse_str(builtin.source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();

    let workgroup_globals = module
        .global_variables
        .iter()
        .filter(|(_, global)| matches!(global.space, AddressSpace::WorkGroup))
        .map(|(_, global)| {
            (
                global.name.as_deref().unwrap_or("<unnamed>"),
                fixed_type_bytes(&module, global.ty),
            )
        })
        .collect::<Vec<_>>();
    let query_tile = constant_u32(&module, "QUERY_TILE") as usize;
    let key_step = constant_u32(&module, "KEY_STEP") as usize;
    let max_head_vectors = constant_u32(&module, "MAX_HEAD_VECTORS") as usize;
    let workgroup_size = constant_u32(&module, "WORKGROUP_SIZE") as usize;
    assert_eq!(max_head_vectors, 18);
    assert_eq!(workgroup_size, 128);
    assert_eq!(query_tile, 128);
    assert_eq!(key_step, 16);
    assert!(
        workgroup_globals.contains(&(
            "key_cache",
            key_step * max_head_vectors * size_of::<[f32; 4]>()
        )),
        "each 16-key step must be loaded once and reused by 128 queries: {workgroup_globals:?}"
    );
    assert!(
        workgroup_globals.contains(&(
            "value_cache",
            key_step * max_head_vectors * size_of::<[f32; 4]>()
        )),
        "each 16-value step must be loaded once and reused by 128 queries: {workgroup_globals:?}"
    );
    assert_eq!(workgroup_globals.len(), 2, "only shared K/V tiles are needed");

    let key_loop_header =
        "for (var key_start = 0u; key_start < params.tokens; key_start = key_start + KEY_STEP)";
    let query_load_header = concat!(
        "for (var vector_index = 0u; ",
        "vector_index < MAX_HEAD_VECTORS; ",
        "vector_index = vector_index + 1u)"
    );
    let key_load_header = concat!(
        "for (var cache_index = local_index; ",
        "cache_index < KEY_STEP * MAX_HEAD_VECTORS; ",
        "cache_index = cache_index + WORKGROUP_SIZE)"
    );
    let query_load = block_span_after(builtin.source, query_load_header);
    let key_loop = block_span_after(builtin.source, key_loop_header);
    let key_load = block_span_after(builtin.source, key_load_header);
    assert!(
        query_load.1 < key_loop.0 && key_loop.0 < key_load.0 && key_load.1 < key_loop.1,
        "Q must load once before the key loop and K/V once inside every key step"
    );
    let query_load_source = &builtin.source[query_load.0..query_load.1];
    assert!(
        query_load_source.contains("query.data[")
            && query_load_source.contains("query_vectors[vector_index] ="),
        "each lane must populate its private Q exactly once",
    );

    let key_load_source = &builtin.source[key_load.0..key_load.1];
    assert!(key_load_source.contains("let key_slot = cache_index / MAX_HEAD_VECTORS;"));
    assert!(key_load_source.contains("let vector_index = cache_index % MAX_HEAD_VECTORS;"));
    assert!(key_load_source.contains("let key_token = key_start + key_slot;"));
    assert!(
        key_load_source.contains("key.data[")
            && key_load_source.contains("value.data[")
            && key_load_source.contains("key_cache[cache_index] = loaded_key;")
            && key_load_source.contains("value_cache[cache_index] = loaded_value;"),
        "cooperative cache population must depend on the real K/V operands",
    );

    let key_loop_source = &builtin.source[key_loop.0..key_loop.1];
    let key_load_end = key_load.1 - key_loop.0;
    let score_start = key_loop_source.find("var scores: array<f32, 16>;")
        .expect("private per-query score block is missing");
    assert!(
        key_loop_source[key_load_end..score_start].contains("workgroupBarrier();"),
        "all K/V cache writes must be visible before the tile's QK and weighted-V work"
    );
    let final_key_use = key_loop_source
        .rfind("key_cache[")
        .expect("cached K is never consumed");
    let final_value_use = key_loop_source
        .rfind("value_cache[")
        .expect("cached V is never consumed");
    let final_cache_use = final_key_use.max(final_value_use);
    assert!(
        key_loop_source[final_cache_use..]
            .rfind("workgroupBarrier();")
            .is_some(),
        "all lanes must finish consuming K/V before the caches are overwritten by the next tile"
    );

    assert!(
        key_loop_source.contains(
            "dot(query_vectors[vector_index], \
             key_cache[key_slot * MAX_HEAD_VECTORS + vector_index])"
        ),
        "the QK loop must consume private Q and shared K",
    );
    assert!(
        key_loop_source.contains(
            "value_cache[key_slot * MAX_HEAD_VECTORS + vector_index]"
        ),
        "weighted-V accumulation must consume shared V",
    );
    let compute_source = &key_loop_source[key_load_end..];
    assert!(
        !compute_source.contains("query.data[")
            && !compute_source.contains("key.data[")
            && !compute_source.contains("value.data["),
        "after cooperative loading, the quadratic loop must not reread Q/K/V globally",
    );
}

fn block_span_after(source: &str, marker: &str) -> (usize, usize) {
    let start = source
        .find(marker)
        .unwrap_or_else(|| panic!("missing WGSL block marker: {marker}"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("WGSL block has no opening brace: {marker}"));
    let mut depth = 0_u32;
    for (offset, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return (start, open + offset + 1);
                }
            }
            _ => {}
        }
    }
    panic!("WGSL block has no closing brace: {marker}");
}

fn constant_u32(module: &Module, name: &str) -> u32 {
    let constant = module
        .constants
        .iter()
        .map(|(_, constant)| constant)
        .find(|constant| constant.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing {name} constant"));
    match module.global_expressions[constant.init] {
        Expression::Literal(Literal::U32(value)) => value,
        ref expression => panic!("{name} is not a u32 literal: {expression:?}"),
    }
}

fn fixed_type_bytes(module: &Module, handle: Handle<Type>) -> usize {
    match &module.types[handle].inner {
        TypeInner::Scalar(scalar) | TypeInner::Atomic(scalar) => usize::from(scalar.width),
        TypeInner::Vector { size, scalar } => u32::from(*size) as usize * usize::from(scalar.width),
        TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => u32::from(*columns) as usize * u32::from(*rows) as usize * usize::from(scalar.width),
        TypeInner::Array {
            size: ArraySize::Constant(size),
            stride,
            ..
        } => size.get() as usize * *stride as usize,
        TypeInner::Struct { span, .. } => *span as usize,
        other => panic!("unsupported function-local type in memory audit: {other:?}"),
    }
}
