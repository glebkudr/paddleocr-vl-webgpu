//! Structural contract for the M7o5 vec4-tiled decode GEMV kernel before
//! production exists (docs/m7o5_tiled_gemv_contract.md).
//!
//! The accepted scalar `gemv_f32` stays byte-for-byte unchanged. The new
//! kernel appends to the catalog, views the same matrix/vector bytes as packed
//! vec4 values, stages at most 3072 scalar columns in portable workgroup
//! storage, and computes eight rows with thirty-two lanes per row.

use pvlc_runtime_core::KernelId;
use pvlc_wgsl::{
    BindingKind, UniformScalar, full_catalog, validate_catalog, validate_source_contract,
};

// Independent caller-owned source authority:
// web/tests/support/m7o5_tiled_gemv_source.mjs.
const GEMV_TILED_F32_SOURCE_BLAKE3: &str =
    "8f39c5ace52d42a8b76b421e7c8a9885a54f38633c79dfa1d937980c5bbe8c81";

fn module(kernel: KernelId) -> &'static pvlc_wgsl::KernelModule {
    full_catalog()
        .iter()
        .find(|module| module.spec.kernel == kernel)
        .unwrap()
}

#[test]
fn gemv_tiled_has_a_fixed_vec4_fp32_webgpu_abi() {
    let tiled = module(KernelId::GemvTiledF32);
    assert_eq!(KernelId::GemvTiledF32.as_str(), "gemv_tiled_f32");
    assert_eq!(tiled.spec.entry_point, "main");
    assert_eq!(tiled.spec.workgroup_size, [256, 1, 1]);
    assert_eq!(
        tiled
            .spec
            .bindings
            .iter()
            .map(|binding| (binding.group, binding.binding, binding.kind))
            .collect::<Vec<_>>(),
        [
            (0, 0, BindingKind::StorageReadVec4F32),
            (0, 1, BindingKind::StorageReadVec4F32),
            (0, 2, BindingKind::StorageReadWriteF32),
            (0, 3, BindingKind::Uniform),
        ]
    );
    assert_eq!(
        tiled
            .spec
            .uniform_fields
            .iter()
            .map(|field| (field.name, field.scalar, field.offset))
            .collect::<Vec<_>>(),
        [
            ("rows", UniformScalar::U32, 0),
            ("columns", UniformScalar::U32, 4),
            ("padding0", UniformScalar::U32, 8),
            ("padding1", UniformScalar::U32, 12),
        ]
    );
    assert_eq!(tiled.spec.uniform_span, 16);
    assert!(tiled.spec.required_features.is_empty());
    validate_source_contract(&tiled.spec, tiled.source).unwrap();
}

#[test]
fn gemv_tiled_source_pins_coalesced_vec4_loads_and_the_exact_tree() {
    let source = module(KernelId::GemvTiledF32).source;

    for pinned in [
        "const TILE_ROWS: u32 = 8u;",
        "const THREADS_PER_ROW: u32 = 32u;",
        "const VECTOR_WIDTH: u32 = 4u;",
        "const SHARED_VEC4_CAPACITY: u32 = 768u;",
        "var<workgroup> shared_vector: array<vec4<f32>, 768>;",
        "var<workgroup> partials: array<f32, 256>;",
    ] {
        assert!(source.contains(pinned), "missing pinned source: {pinned}");
    }

    // At maximum width this is 12 KiB of staged vector plus 1 KiB of
    // partials: 13,312 bytes, within the portable 16 KiB minimum.
    assert_eq!(768 * 16 + 256 * 4, 13_312);
    assert!(13_312 <= 16 * 1024);

    // Threads in one staging round read adjacent vec4 values.
    assert!(source.contains("let vector_columns = params.columns / VECTOR_WIDTH;"));
    assert!(source.contains(
        "for (var staged = local_id.x; staged < vector_columns; staged = staged + 256u) {"
    ));
    assert!(source.contains("shared_vector[staged] = vector.data[staged];"));
    assert!(source.contains("workgroupBarrier();"));

    // One 32-lane row group maps exactly onto one of eight output rows.
    assert!(source.contains("let row_group = local_id.x / THREADS_PER_ROW;"));
    assert!(source.contains("let lane = local_id.x % THREADS_PER_ROW;"));
    assert!(source.contains("let row = workgroup_id.x * TILE_ROWS + row_group;"));
    assert!(source.contains("let row_base = row * vector_columns;"));

    // Lane-strided vec4 matrix reads are coalesced. Every vec4 product is
    // accumulated in the exact x/y/z/w scalar order; dot/fma are forbidden
    // because they do not pin the same arithmetic association.
    assert!(source.contains(
        "for (var column = lane; column < vector_columns; column = column + THREADS_PER_ROW) {"
    ));
    assert!(
        source.contains("let products = matrix.data[row_base + column] * shared_vector[column];")
    );
    for component in ["x", "y", "z", "w"] {
        assert!(
            source.contains(&format!("partial = partial + products.{component};")),
            "missing {component} component accumulation"
        );
    }
    assert!(
        !source.contains("dot("),
        "dot() leaves component association unpinned"
    );
    assert!(
        !source.contains("fma("),
        "fma() changes the two-rounding discipline"
    );

    // Every invocation, including the tail workgroup's out-of-range row
    // groups, publishes a partial and reaches all three lexical barriers.
    // The matrix read is inside the row guard, while the publication and
    // both synchronization sites are outside it.
    assert!(source.contains("if row < params.rows {\n        let row_base"));
    assert!(source.contains(
        "    }\n    partials[local_id.x] = partial;\n    workgroupBarrier();\n\n    for"
    ));
    assert_eq!(
        source.matches("workgroupBarrier();").count(),
        3,
        "staging, partial publication, and reduction-loop barriers are required"
    );

    // The 32 lane partials reduce through the fixed 16/8/4/2/1 tree. The
    // barrier is lexically inside the loop but outside the lane guard, so
    // every invocation participates in every reduction round.
    assert!(
        source.contains(
            "for (var stride = THREADS_PER_ROW / 2u; stride > 0u; stride = stride >> 1u) {"
        )
    );
    assert!(source.contains("if lane < stride {"));
    assert!(
        source.contains(
            "partials[local_id.x] = partials[local_id.x] + partials[local_id.x + stride];"
        )
    );
    assert!(source.contains("        }\n        workgroupBarrier();\n    }\n    if lane == 0u"));
    assert!(source.contains("if lane == 0u && row < params.rows {"));
    assert!(source.contains("output.data[row] = partials[local_id.x];"));
}

#[test]
fn gemv_tiled_appends_without_reindexing_the_accepted_catalog() {
    assert_eq!(
        full_catalog()
            .iter()
            .filter(|module| module.spec.kernel == KernelId::GemvTiledF32)
            .count(),
        1
    );
    assert!(KernelId::ALL.len() >= 23);
    assert_eq!(KernelId::ALL[22], KernelId::GemvTiledF32);
    assert_eq!(full_catalog().len(), KernelId::ALL.len());
    assert_eq!(
        full_catalog()
            .iter()
            .map(|module| module.spec.kernel)
            .collect::<Vec<_>>(),
        KernelId::ALL
    );
    assert_eq!(
        &KernelId::ALL[..22],
        [
            KernelId::GemmF32,
            KernelId::GemvF32,
            KernelId::LayerNormF32,
            KernelId::RmsNormF32,
            KernelId::SiluF32,
            KernelId::GeluTanhF32,
            KernelId::RopeNeoxF32,
            KernelId::VisionAttentionF32,
            KernelId::VisionPatchProjectionF32,
            KernelId::AddF32,
            KernelId::GeluErfF32,
            KernelId::ProjectorMerge2x2F32,
            KernelId::VisionQkvFusedF32,
            KernelId::DecoderKvAppendF32,
            KernelId::DecoderGqaF32,
            KernelId::DecoderGqaSplitPartialF32,
            KernelId::DecoderGqaSplitMergeF32,
            KernelId::DecoderMropeF32,
            KernelId::DecoderSwigluF32,
            KernelId::DecoderPrefillGqaF32,
            KernelId::DecoderPrefillMropeF32,
            KernelId::DecoderKvAppendRangeF32,
        ]
    );
    validate_catalog().unwrap();
}

#[test]
fn gemv_tiled_caller_owned_source_blake3_anchor_is_stable() {
    let module = module(KernelId::GemvTiledF32);
    assert_eq!(
        blake3::hash(module.source.as_bytes()).to_hex().to_string(),
        GEMV_TILED_F32_SOURCE_BLAKE3,
        "production gemv_tiled_f32 differs from the caller-owned browser authority"
    );
}
