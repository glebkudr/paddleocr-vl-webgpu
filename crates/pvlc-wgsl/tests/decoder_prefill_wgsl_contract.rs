use pvlc_runtime_core::KernelId;
use pvlc_wgsl::{
    BindingKind, UniformScalar, full_catalog, storage_read_write_variant, validate_catalog,
    validate_source_contract,
};

// Compile-time catalog pin: the WGSL production step must grow KernelId::ALL
// from the accepted 17 kernels to 22, appending the three M6e7 prefill
// kernels in the pinned order gqa, mrope, append and (M7o2) inserting the two
// split-K decode GQA kernels immediately after decoder_gqa_f32 in the pinned
// order partial, merge. Until the production step lands, this test target
// does not compile: the type ascription reports the missing catalog slots,
// and the slot assertions below pin their exact positions once the array has
// grown.
const GROWN_ALL: &[KernelId] = &KernelId::ALL;
const _: () = {
    assert!(
        matches!(GROWN_ALL[15], KernelId::DecoderGqaSplitPartialF32),
        "kernel 16 of KernelId::ALL must be decoder_gqa_split_partial_f32"
    );
    assert!(
        matches!(GROWN_ALL[16], KernelId::DecoderGqaSplitMergeF32),
        "kernel 17 of KernelId::ALL must be decoder_gqa_split_merge_f32"
    );
    assert!(
        matches!(GROWN_ALL[19], KernelId::DecoderPrefillGqaF32),
        "kernel 20 of KernelId::ALL must be decoder_prefill_gqa_f32"
    );
    assert!(
        matches!(GROWN_ALL[20], KernelId::DecoderPrefillMropeF32),
        "kernel 21 of KernelId::ALL must be decoder_prefill_mrope_f32"
    );
    assert!(
        matches!(GROWN_ALL[21], KernelId::DecoderKvAppendRangeF32),
        "kernel 22 of KernelId::ALL must be decoder_kv_append_range_f32"
    );
    assert!(
        matches!(GROWN_ALL[22], KernelId::GemvTiledF32),
        "kernel 23 of KernelId::ALL must be gemv_tiled_f32"
    );
};

// Byte-exact BLAKE3 anchors of the all-read_write variant sources, recorded
// when the M6e7 WGSL production step landed the three prefill kernels —
// exactly like the accepted static shader anchors in wgsl_contract.rs.
const DECODER_PREFILL_GQA_F32_VARIANT_BLAKE3: &str =
    "4bcfe8776697b050787f1b9d7549c216fcffb6d472d1a6d51607087510eebfc5";
const DECODER_PREFILL_MROPE_F32_VARIANT_BLAKE3: &str =
    "cb1f2a8cfadd08b195062a386df824526500167d2d67fa46aad40a999defad0b";
const DECODER_KV_APPEND_RANGE_F32_VARIANT_BLAKE3: &str =
    "7875e13521922430ba7f29f352e9f2da642cf93951bc3292f1dc3ed7e94988b3";

fn module(kernel: KernelId) -> &'static pvlc_wgsl::KernelModule {
    full_catalog()
        .iter()
        .find(|module| module.spec.kernel == kernel)
        .unwrap()
}

#[test]
fn decoder_prefill_mrope_has_a_fixed_multi_token_fp32_webgpu_abi() {
    let mrope = module(KernelId::DecoderPrefillMropeF32);
    assert_eq!(
        KernelId::DecoderPrefillMropeF32.as_str(),
        "decoder_prefill_mrope_f32"
    );
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
            ("tokens", UniformScalar::U32, 0),
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
fn decoder_prefill_mrope_source_rotates_every_token_row_with_the_row_index_as_position() {
    let source = module(KernelId::DecoderPrefillMropeF32).source;

    // The decoder topology stays pinned at compile time exactly as in the
    // accepted single-token kernel: 16 query heads and 2 key-value heads of
    // dim 128 fused into one 2304-wide row per token, and repeated [t, h, w]
    // sections of 16, 24, 24 (cumulative ends 16 and 40).
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

    // One thread per rotated element over tokens * 2304; the token axis is
    // recovered by dividing the fused row width, and the token index itself
    // is the M-RoPE position (prefill positions are 0..tokens).
    assert!(source.contains("let linear = global_id.x;"));
    assert!(source.contains("if linear >= params.tokens * TOTAL_WIDTH {"));
    assert!(source.contains("let token = linear / TOTAL_WIDTH;"));
    assert!(source.contains("let within = linear % TOTAL_WIDTH;"));
    assert!(source.contains("let dim = within % HEAD_DIM;"));

    // The axis-major [3, capacity, 128] table row is
    // axis * rope_capacity + token; there is no position uniform word — the
    // token row index is the position. The axis select is pinned whole so the
    // outer value, both boundary conditions, and the nesting cannot drift.
    assert!(source.contains("let local = select(dim, dim - HALF_DIM, dim >= HALF_DIM);"));
    assert!(source.contains(
        "let axis = select(select(0u, 1u, local >= FIRST_SECTION_END), 2u, local >= SECOND_SECTION_END);"
    ));
    assert!(
        source
            .contains("let table_index = (axis * params.rope_capacity + token) * HEAD_DIM + dim;")
    );

    // Half-dimension rotation out[d] = x[d] * cos[d] + sign * x[partner] *
    // sin[d] with partner = d +/- HALF_DIM and sign = -1 below the half, +1
    // above — applied per token row, so both the element and its partner are
    // addressed through the per-token row base. The joined statements are
    // pinned whole so the join operator and the row-base placement cannot
    // drift, and the query/key branch guard pins the fused-row split point.
    assert!(
        source.contains(
            "let partner = select(within + HALF_DIM, within - HALF_DIM, dim >= HALF_DIM);"
        )
    );
    assert!(source.contains("let sign = select(-1.0, 1.0, dim >= HALF_DIM);"));
    assert!(source.contains("if within < QUERY_WIDTH {"));
    assert!(source.contains("let query_base = token * QUERY_WIDTH;"));
    assert!(source.contains("let value = query.data[query_base + within];"));
    assert!(source.contains(
        "let rotated = value * rope_cos.data[table_index]\n            + sign * query.data[query_base + partner] * rope_sin.data[table_index];"
    ));
    assert!(source.contains("output_query.data[query_base + within] = rotated;"));
    assert!(source.contains("let key_base = token * KEY_WIDTH;"));
    assert!(source.contains("let key_index = within - QUERY_WIDTH;"));
    assert!(source.contains("let key_partner = partner - QUERY_WIDTH;"));
    assert!(source.contains("let value = key.data[key_base + key_index];"));
    assert!(source.contains(
        "let rotated = value * rope_cos.data[table_index]\n            + sign * key.data[key_base + key_partner] * rope_sin.data[table_index];"
    ));
    assert!(source.contains("output_key.data[key_base + key_index] = rotated;"));
}

#[test]
fn decoder_kv_append_range_has_a_fixed_prefix_zero_fp32_webgpu_abi() {
    let append = module(KernelId::DecoderKvAppendRangeF32);
    assert_eq!(
        KernelId::DecoderKvAppendRangeF32.as_str(),
        "decoder_kv_append_range_f32"
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
    // KV2/D128 are pinned, so the accepted single-token append's
    // key_value_heads/head_dim uniform words are dropped; the range append
    // carries only the token count and the cache capacity, at prefix 0.
    assert_eq!(
        append
            .spec
            .uniform_fields
            .iter()
            .map(|field| (field.name, field.scalar, field.offset))
            .collect::<Vec<_>>(),
        [
            ("tokens", UniformScalar::U32, 0),
            ("cache_capacity", UniformScalar::U32, 4),
            ("padding0", UniformScalar::U32, 8),
            ("padding1", UniformScalar::U32, 12),
        ]
    );
    assert_eq!(append.spec.uniform_span, 16);
    assert!(append.spec.required_features.is_empty());
    validate_source_contract(&append.spec, append.source).unwrap();
}

#[test]
fn decoder_kv_append_range_source_writes_rows_zero_to_tokens_into_compact_cache_planes() {
    let source = module(KernelId::DecoderKvAppendRangeF32).source;

    // The KV width is pinned (2 key-value heads x 128), matching the compact
    // physical cache planes [capacity, 2, 128].
    assert!(source.contains("const KEY_VALUE_WIDTH: u32 = 256u;"));

    // One thread per appended element over tokens * 256; out-of-range threads
    // and an over-capacity token count return before any access, mirroring
    // the accepted single-token append's capacity guard.
    assert!(source.contains("let linear = global_id.x;"));
    assert!(source.contains(
        "if linear >= params.tokens * KEY_VALUE_WIDTH || params.tokens > params.cache_capacity {"
    ));

    // Row-major token placement at prefix 0: cache row = token, column =
    // within-row element, so cache_index carries no prefix term. The store
    // statements are pinned whole so the read/write planes cannot swap.
    assert!(source.contains("let token = linear / KEY_VALUE_WIDTH;"));
    assert!(source.contains("let within = linear % KEY_VALUE_WIDTH;"));
    assert!(source.contains("let cache_index = token * KEY_VALUE_WIDTH + within;"));
    assert!(source.contains("key_cache.data[cache_index] = appended_key.data[linear];"));
    assert!(source.contains("value_cache.data[cache_index] = appended_value.data[linear];"));
}

#[test]
fn decoder_prefill_gqa_has_a_fixed_causal_multi_token_fp32_webgpu_abi() {
    let attention = module(KernelId::DecoderPrefillGqaF32);
    assert_eq!(
        KernelId::DecoderPrefillGqaF32.as_str(),
        "decoder_prefill_gqa_f32"
    );
    assert_eq!(attention.spec.entry_point, "main");
    assert_eq!(attention.spec.workgroup_size, [64, 1, 1]);
    // Q read, K/V read straight from the physical cache planes (the range
    // append runs before attention), one writable output, no cu_seqlens.
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
            ("tokens", UniformScalar::U32, 0),
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
fn decoder_prefill_gqa_source_implements_causal_grouped_online_softmax() {
    let source = module(KernelId::DecoderPrefillGqaF32).source;

    // Causal prefill attention has no cu_seqlens segmentation and no decode
    // cache_tokens word: the bound is purely key_token <= query_token against
    // the physical cache planes, and the token count arrives as `tokens`.
    assert!(!source.contains("cu_seqlens"));
    assert!(!source.contains("segments"));
    assert!(!source.contains("cache_tokens"));

    // One thread per (query token, query head) over tokens * query_heads.
    assert!(source.contains("let linear = global_id.x;"));
    assert!(source.contains("let work_items = params.tokens * params.query_heads;"));
    assert!(source.contains("if linear >= work_items {"));
    assert!(source.contains("let query_token = linear / params.query_heads;"));
    assert!(source.contains("let query_head = linear % params.query_heads;"));

    // The accepted decoder_gqa_f32 grouped mapping: kv_head = q_head /
    // (query_heads / key_value_heads), pinned as whole statements so the
    // division and the grouping order cannot drift. Q rows are
    // [tokens, query_heads, head_dim].
    assert!(
        source.contains("let query_heads_per_kv = params.query_heads / params.key_value_heads;")
    );
    assert!(source.contains("let key_value_head = query_head / query_heads_per_kv;"));
    assert!(source.contains(
        "let query_base = (query_token * params.query_heads + query_head) * params.head_dim;"
    ));

    // Head dim 128 scale pinned exactly as 1/sqrt(head_dim), as in the
    // accepted decoder_gqa_f32 (the fixed-size accumulator is pinned below).
    assert!(source.contains("let attention_scale = inverseSqrt(f32(params.head_dim));"));

    // The causal bound replaces the vision attention segment scan: keys run
    // 0..=query_token over the cache planes laid out
    // [capacity, key_value_heads, head_dim].
    assert!(source.contains(
        "for (var key_token = 0u; key_token <= query_token; key_token = key_token + 1u) {"
    ));
    assert!(source.contains(
        "let key_base =\n            (key_token * params.key_value_heads + key_value_head) * params.head_dim;"
    ));

    // The vision_attention_f32 online-softmax skeleton, pinned whole so the
    // running maximum, the rescaling weights, the denominator update, and the
    // running-state writes cannot drift (a mutant dropping
    // `maximum = next_maximum;` silently breaks the online rescaling).
    assert!(source.contains("var weighted: array<f32, 128>;"));
    assert!(source.contains("var maximum = 0.0;"));
    assert!(source.contains("var denominator = 0.0;"));
    assert!(source.contains("var first_key = true;"));
    assert!(source.contains("var score = 0.0;"));
    assert!(source.contains("score = score * attention_scale;"));
    assert!(source.contains("if first_key {"));
    assert!(source.contains("maximum = score;"));
    assert!(source.contains("denominator = 1.0;"));
    assert!(source.contains("first_key = false;"));
    assert!(source.contains("let next_maximum = max(maximum, score);"));
    assert!(source.contains("let previous_weight = exp(maximum - next_maximum);"));
    assert!(source.contains("let current_weight = exp(score - next_maximum);"));
    assert!(source.contains("denominator = denominator * previous_weight + current_weight;"));
    assert!(source.contains("maximum = next_maximum;"));

    // K/V come from the appended cache planes, and the normalized output row
    // lands at the same [tokens, query_heads, head_dim] base as the query.
    assert!(source.contains(
        "score = score\n                + query.data[query_base + dimension] * key_cache.data[key_base + dimension];"
    ));
    assert!(source.contains("weighted[dimension] = value_cache.data[key_base + dimension];"));
    assert!(source.contains(
        "weighted[dimension] = weighted[dimension] * previous_weight\n                    + current_weight * value_cache.data[key_base + dimension];"
    ));
    assert!(
        source.contains("output.data[query_base + dimension] = weighted[dimension] / denominator;")
    );
}

#[test]
fn prefill_modules_extend_the_full_catalog_in_the_pinned_order_and_pass_the_naga_gate() {
    for kernel in [
        KernelId::DecoderPrefillGqaF32,
        KernelId::DecoderPrefillMropeF32,
        KernelId::DecoderKvAppendRangeF32,
    ] {
        assert_eq!(
            full_catalog()
                .iter()
                .filter(|module| module.spec.kernel == kernel)
                .count(),
            1,
            "{kernel} must appear exactly once in the full catalog"
        );
    }

    // The accepted 17 kernels keep their relative order; the M7o2 split-K
    // kernels are inserted immediately after decoder_gqa_f32 and the three
    // prefill kernels stay appended in the pinned order gqa, mrope, append;
    // the full WGSL catalog mirrors KernelId::ALL exactly.
    assert_eq!(
        &KernelId::ALL[..23],
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
            KernelId::GemvTiledF32,
        ]
    );
    assert_eq!(full_catalog().len(), KernelId::ALL.len());
    assert_eq!(
        full_catalog()
            .iter()
            .map(|module| module.spec.kernel)
            .collect::<Vec<_>>(),
        KernelId::ALL
    );
    validate_catalog().unwrap();
}

#[test]
fn prefill_source_blake3_anchors_are_stable() {
    for (name, kernel, expected_blake3) in [
        (
            "decoder_prefill_gqa_f32",
            KernelId::DecoderPrefillGqaF32,
            DECODER_PREFILL_GQA_F32_VARIANT_BLAKE3,
        ),
        (
            "decoder_prefill_mrope_f32",
            KernelId::DecoderPrefillMropeF32,
            DECODER_PREFILL_MROPE_F32_VARIANT_BLAKE3,
        ),
        (
            "decoder_kv_append_range_f32",
            KernelId::DecoderKvAppendRangeF32,
            DECODER_KV_APPEND_RANGE_F32_VARIANT_BLAKE3,
        ),
    ] {
        let module = module(kernel);
        let variant = storage_read_write_variant(&module.spec, module.source).unwrap();
        assert_eq!(
            blake3::hash(variant.as_bytes()).to_hex().to_string(),
            expected_blake3,
            "static all-read_write {name} WGSL hash drifted"
        );
    }
}
