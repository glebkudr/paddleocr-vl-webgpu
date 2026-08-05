use pvlc_runtime_core::KernelId;
use pvlc_wgsl::{
    BindingKind, UniformScalar, full_catalog, validate_catalog, validate_source_contract,
};

fn module(kernel: KernelId) -> &'static pvlc_wgsl::KernelModule {
    full_catalog()
        .iter()
        .find(|module| module.spec.kernel == kernel)
        .unwrap()
}

#[test]
fn cached_decode_appends_two_kv_planes_without_query_head_expansion() {
    let append = module(KernelId::DecoderKvAppendF32);
    assert_eq!(
        KernelId::DecoderKvAppendF32.as_str(),
        "decoder_kv_append_f32"
    );
    assert_eq!(append.spec.entry_point, "main");
    assert_eq!(append.spec.workgroup_size, [64, 1, 1]);
    assert_eq!(
        append
            .spec
            .bindings
            .iter()
            .map(|binding| (binding.group, binding.binding, binding.kind))
            .collect::<Vec<_>>(),
        [
            (0, 0, BindingKind::StorageReadF32),
            (0, 1, BindingKind::StorageReadF32),
            (0, 2, BindingKind::StorageReadWriteF32),
            (0, 3, BindingKind::StorageReadWriteF32),
            (0, 4, BindingKind::Uniform),
        ]
    );
    assert_eq!(
        append
            .spec
            .uniform_fields
            .iter()
            .map(|field| (field.name, field.scalar, field.offset))
            .collect::<Vec<_>>(),
        [
            ("prefix_tokens", UniformScalar::U32, 0),
            ("key_value_heads", UniformScalar::U32, 4),
            ("head_dim", UniformScalar::U32, 8),
            ("cache_capacity", UniformScalar::U32, 12),
        ]
    );
    assert_eq!(append.spec.uniform_span, 16);
    assert!(append.spec.required_features.is_empty());
    validate_source_contract(&append.spec, append.source).unwrap();
}

#[test]
fn cached_decode_gqa_uses_direct_grouped_cache_bindings_and_fixed_fp32_abi() {
    let attention = module(KernelId::DecoderGqaF32);
    assert_eq!(KernelId::DecoderGqaF32.as_str(), "decoder_gqa_f32");
    assert_eq!(attention.spec.entry_point, "main");
    assert_eq!(attention.spec.workgroup_size, [64, 1, 1]);
    assert_eq!(
        attention
            .spec
            .bindings
            .iter()
            .map(|binding| (binding.group, binding.binding, binding.kind))
            .collect::<Vec<_>>(),
        [
            (0, 0, BindingKind::StorageReadF32),
            (0, 1, BindingKind::StorageReadF32),
            (0, 2, BindingKind::StorageReadF32),
            (0, 3, BindingKind::StorageReadWriteF32),
            (0, 4, BindingKind::Uniform),
        ]
    );
    assert_eq!(
        attention
            .spec
            .uniform_fields
            .iter()
            .map(|field| (field.name, field.scalar, field.offset))
            .collect::<Vec<_>>(),
        [
            ("cache_tokens", UniformScalar::U32, 0),
            ("query_heads", UniformScalar::U32, 4),
            ("key_value_heads", UniformScalar::U32, 8),
            ("head_dim", UniformScalar::U32, 12),
        ]
    );
    assert_eq!(attention.spec.uniform_span, 16);
    assert!(attention.spec.required_features.is_empty());
    validate_source_contract(&attention.spec, attention.source).unwrap();
}

#[test]
fn both_cached_decode_modules_are_unique_and_naga_validated_by_the_full_catalog_gate() {
    assert_eq!(
        full_catalog()
            .iter()
            .filter(|module| {
                matches!(
                    module.spec.kernel,
                    KernelId::DecoderKvAppendF32 | KernelId::DecoderGqaF32
                )
            })
            .count(),
        2
    );
    assert_eq!(
        &KernelId::ALL[13..19],
        [
            KernelId::DecoderKvAppendF32,
            KernelId::DecoderGqaF32,
            KernelId::DecoderGqaSplitPartialF32,
            KernelId::DecoderGqaSplitMergeF32,
            KernelId::DecoderMropeF32,
            KernelId::DecoderSwigluF32,
        ]
    );
    validate_catalog().unwrap();
}
