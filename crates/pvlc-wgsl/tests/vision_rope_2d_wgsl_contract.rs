use pvlc_runtime_core::KernelId;
use pvlc_wgsl::{BindingKind, UniformScalar, module, validate_source_contract};

#[test]
fn vision_rope_2d_has_the_in_place_qk_and_precomputed_table_abi() {
    let module = module(KernelId::VisionRope2dF32).expect("vision 2D RoPE kernel is registered");

    assert_eq!(module.spec.entry_point, "main");
    assert_eq!(module.spec.workgroup_size, [64, 1, 1]);
    assert_eq!(
        module
            .spec
            .bindings
            .iter()
            .map(|binding| binding.kind)
            .collect::<Vec<_>>(),
        vec![
            BindingKind::StorageReadWriteF32,
            BindingKind::StorageReadWriteF32,
            BindingKind::StorageReadF32,
            BindingKind::StorageReadF32,
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
            ("padding", UniformScalar::U32, 12),
        ]
    );
    assert_eq!(module.spec.uniform_span, 16);
    assert!(module.spec.required_features.is_empty());
    validate_source_contract(&module.spec, module.source).unwrap();
}

#[test]
fn vision_rope_2d_rotates_each_half_pair_once_without_runtime_trigonometry() {
    let source = module(KernelId::VisionRope2dF32).unwrap().source;

    assert!(source.contains("let pair_count = params.head_dim / 2u;"));
    assert!(source.contains("let work_items = params.tokens * params.heads * pair_count;"));
    assert!(source.contains("let linear_pair = global_id.x;"));
    assert!(source.contains("if linear_pair >= work_items {"));
    assert!(source.contains("let pair = linear_pair % pair_count;"));
    assert!(source.contains("let linear_head = linear_pair / pair_count;"));
    assert!(source.contains("let head = linear_head % params.heads;"));
    assert!(source.contains("let token = linear_head / params.heads;"));
    assert!(
        source
            .contains("let first_index = (token * params.heads + head) * params.head_dim + pair;")
    );
    assert!(source.contains("let second_index = first_index + pair_count;"));
    assert!(source.contains("let cosine = cos_table.data[token * pair_count + pair];"));
    assert!(source.contains("let sine = sin_table.data[token * pair_count + pair];"));
    assert!(source.contains("let query_first = query.data[first_index];"));
    assert!(source.contains("let query_second = query.data[second_index];"));
    assert!(source.contains("let key_first = key.data[first_index];"));
    assert!(source.contains("let key_second = key.data[second_index];"));
    assert!(
        source.contains("query.data[first_index] = query_first * cosine - query_second * sine;")
    );
    assert!(
        source.contains("query.data[second_index] = query_second * cosine + query_first * sine;")
    );
    assert!(source.contains("key.data[first_index] = key_first * cosine - key_second * sine;"));
    assert!(source.contains("key.data[second_index] = key_second * cosine + key_first * sine;"));
    assert!(!source.contains("sin("));
    assert!(!source.contains("cos("));
    assert!(!source.contains("pow("));
}
