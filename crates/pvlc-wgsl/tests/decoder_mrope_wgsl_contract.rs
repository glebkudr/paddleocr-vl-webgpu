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
fn decoder_mrope_has_a_fixed_single_token_fp32_webgpu_abi() {
    let mrope = module(KernelId::DecoderMropeF32);
    assert_eq!(KernelId::DecoderMropeF32.as_str(), "decoder_mrope_f32");
    assert_eq!(mrope.spec.entry_point, "main");
    assert_eq!(mrope.spec.workgroup_size, [64, 1, 1]);
    assert_eq!(
        mrope
            .spec
            .bindings
            .iter()
            .map(|binding| (binding.group, binding.binding, binding.kind))
            .collect::<Vec<_>>(),
        [
            (0, 0, BindingKind::StorageReadF32),
            (0, 1, BindingKind::StorageReadF32),
            (0, 2, BindingKind::StorageReadF32),
            (0, 3, BindingKind::StorageReadF32),
            (0, 4, BindingKind::StorageReadWriteF32),
            (0, 5, BindingKind::StorageReadWriteF32),
            (0, 6, BindingKind::Uniform),
        ]
    );
    assert_eq!(
        mrope
            .spec
            .uniform_fields
            .iter()
            .map(|field| (field.name, field.scalar, field.offset))
            .collect::<Vec<_>>(),
        [
            ("position", UniformScalar::U32, 0),
            ("rope_capacity", UniformScalar::U32, 4),
            ("padding0", UniformScalar::U32, 8),
            ("padding1", UniformScalar::U32, 12),
        ]
    );
    assert_eq!(mrope.spec.uniform_span, 16);
    assert!(mrope.spec.required_features.is_empty());
    validate_source_contract(&mrope.spec, mrope.source).unwrap();
}

#[test]
fn decoder_mrope_source_implements_axis_major_section_selection_and_half_dim_rotation() {
    let source = module(KernelId::DecoderMropeF32).source;

    // The PaddleOCR-VL-1.6 decoder topology is pinned at compile time: one
    // decode token, 16 query heads and 2 key-value heads of dim 128, and
    // repeated [t, h, w] sections of 16, 24, 24 (cumulative ends 16 and 40).
    for pinned in [
        "const HEAD_DIM: u32 = 128u;",
        "const HALF_DIM: u32 = 64u;",
        "const FIRST_SECTION_END: u32 = 16u;",
        "const SECOND_SECTION_END: u32 = 40u;",
        "const QUERY_WIDTH: u32 = 2048u;",
        "const KEY_WIDTH: u32 = 256u;",
        "const TOTAL_WIDTH: u32 = 2304u;",
    ] {
        assert!(source.contains(pinned), "missing pinned constant: {pinned}");
    }

    // One thread per rotated element over the concatenated query+key width.
    assert!(source.contains("let linear = global_id.x;"));
    assert!(source.contains("if linear >= TOTAL_WIDTH {"));
    assert!(source.contains("let dim = linear % HEAD_DIM;"));

    // Axis-major [3, T, 128] tables: the repeated [t,h,w,t,h,w] section layout
    // selects the axis from the within-half dimension, and the row is
    // axis * rope_capacity + position. The axis select is pinned whole so the
    // outer value, both boundary conditions, and the nesting cannot drift.
    assert!(source.contains("let local = select(dim, dim - HALF_DIM, dim >= HALF_DIM);"));
    assert!(source.contains(
        "let axis = select(select(0u, 1u, local >= FIRST_SECTION_END), 2u, local >= SECOND_SECTION_END);"
    ));
    assert!(source.contains(
        "let table_index = (axis * params.rope_capacity + params.position) * HEAD_DIM + dim;"
    ));

    // Half-dimension rotation out[d] = x[d] * cos[d] + sign * x[partner] * sin[d]
    // with partner = d +/- HALF_DIM and sign = -1 below the half, +1 above. The
    // joined statements are pinned whole so the join operator cannot flip, and
    // the query/key branch guard is pinned so the split point cannot drift.
    assert!(
        source.contains(
            "let partner = select(linear + HALF_DIM, linear - HALF_DIM, dim >= HALF_DIM);"
        )
    );
    assert!(source.contains("let sign = select(-1.0, 1.0, dim >= HALF_DIM);"));
    assert!(source.contains("if linear < QUERY_WIDTH {"));
    assert!(source.contains("let value = query.data[linear];"));
    assert!(source.contains(
        "let rotated = value * rope_cos.data[table_index]\n            + sign * query.data[partner] * rope_sin.data[table_index];"
    ));
    assert!(source.contains("output_query.data[linear] = rotated;"));
    assert!(source.contains("let key_index = linear - QUERY_WIDTH;"));
    assert!(source.contains("let key_partner = partner - QUERY_WIDTH;"));
    assert!(source.contains("let value = key.data[key_index];"));
    assert!(source.contains(
        "let rotated = value * rope_cos.data[table_index]\n            + sign * key.data[key_partner] * rope_sin.data[table_index];"
    ));
    assert!(source.contains("output_key.data[key_index] = rotated;"));
}

#[test]
fn decoder_mrope_module_is_unique_and_naga_validated_by_the_full_catalog_gate() {
    assert_eq!(
        full_catalog()
            .iter()
            .filter(|module| module.spec.kernel == KernelId::DecoderMropeF32)
            .count(),
        1
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
