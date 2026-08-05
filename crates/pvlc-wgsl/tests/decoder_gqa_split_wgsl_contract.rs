//! Structural contract for the M7o2 split-K (flash-decoding) decode GQA
//! kernels before any WGSL production exists
//! (docs/m7o2_split_k_decode_gqa_contract.md).
//!
//! The accepted `decoder_gqa_f32` kernel (one thread per query head, serial
//! online softmax over the whole cache) remains accepted and unchanged. M7o2
//! appends two kernels to the catalog in the pinned order immediately AFTER
//! `decoder_gqa_f32` (the relative order of every other kernel is unchanged):
//!
//! - `decoder_gqa_split_partial_f32` — one workgroup of 64 threads per
//!   (query_head, chunk) with the pinned chunk size 32 keys: 32 score threads
//!   each compute the full 128-dim dot for exactly one chunk key (ascending
//!   hidden order, zero accumulator, the accepted scale 1/sqrt(128)); a
//!   deterministic fixed-shape shared-memory tree reduction over ascending
//!   index pairs yields the chunk maximum and the chunk weight sum; each
//!   thread accumulates the unnormalized weighted V for its two dims over the
//!   chunk keys in ascending key order (exp(score - chunk_max)); the
//!   workgroup writes (weighted_v[128], chunk_max, chunk_sum) into one
//!   192-stride partials plane row at (query_head, chunk);
//! - `decoder_gqa_split_merge_f32` — one work item per (query_head, dim):
//!   merges the ceil(cache_tokens / 32) partials in ascending chunk order
//!   (running maximum/rescaling, the standard split-K merge with a fixed
//!   association), writing the normalized output row.

use pvlc_runtime_core::KernelId;
use pvlc_wgsl::{
    BindingKind, UniformScalar, full_catalog, storage_read_write_variant, validate_catalog,
    validate_source_contract,
};

// Byte-exact BLAKE3 anchors of the all-read_write variant sources. Until the
// WGSL production step lands the two kernels this target does not compile
// (the KernelId variants are unresolved); the production step records the
// measured anchors here at landing, exactly like the accepted M6e7 prefill
// anchors in decoder_prefill_wgsl_contract.rs.
// M7o2 amendment: the partial anchor was re-recorded after the two
// reduction-correctness fixes exposed by the M6e2 persistent native KV
// session contract: (1) the chunk max tree reduces the separate `maxima`
// copy and the chunk sum tree the separate `sums` copy, so the per-key
// scores and weights survive for the weighted-V accumulation, and (2) the
// tail V read is masked with select on key_token < cache_tokens because
// 0.0 * NaN is NaN in a physically poisoned cache. The merge anchor is
// untouched.
const DECODER_GQA_SPLIT_PARTIAL_F32_VARIANT_BLAKE3: &str =
    "9ac5acb6e0ee4fd784a3c216a672275df312dbaf763e038e7182b37fff8732d0";
const DECODER_GQA_SPLIT_MERGE_F32_VARIANT_BLAKE3: &str =
    "870cd376cbef9e4a05d0500809e4076f20c44c84922aad90e682df09ed8a7ced";

fn module(kernel: KernelId) -> &'static pvlc_wgsl::KernelModule {
    full_catalog()
        .iter()
        .find(|module| module.spec.kernel == kernel)
        .unwrap()
}

#[test]
fn decoder_gqa_split_partial_has_a_fixed_fp32_webgpu_abi() {
    let partial = module(KernelId::DecoderGqaSplitPartialF32);
    assert_eq!(
        KernelId::DecoderGqaSplitPartialF32.as_str(),
        "decoder_gqa_split_partial_f32"
    );
    assert_eq!(partial.spec.entry_point, "main");
    assert_eq!(partial.spec.workgroup_size, [64, 1, 1]);
    assert_eq!(
        partial
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
        partial
            .spec
            .uniform_fields
            .iter()
            .map(|field| (field.name, field.scalar, field.offset))
            .collect::<Vec<_>>(),
        [
            ("cache_tokens", UniformScalar::U32, 0),
            ("chunk_count", UniformScalar::U32, 4),
            ("padding0", UniformScalar::U32, 8),
            ("padding1", UniformScalar::U32, 12),
        ]
    );
    assert_eq!(partial.spec.uniform_span, 16);
    assert!(partial.spec.required_features.is_empty());
    validate_source_contract(&partial.spec, partial.source).unwrap();
}

#[test]
fn decoder_gqa_split_merge_has_a_fixed_fp32_webgpu_abi() {
    let merge = module(KernelId::DecoderGqaSplitMergeF32);
    assert_eq!(
        KernelId::DecoderGqaSplitMergeF32.as_str(),
        "decoder_gqa_split_merge_f32"
    );
    assert_eq!(merge.spec.entry_point, "main");
    assert_eq!(merge.spec.workgroup_size, [64, 1, 1]);
    assert_eq!(
        merge
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
        merge
            .spec
            .uniform_fields
            .iter()
            .map(|field| (field.name, field.scalar, field.offset))
            .collect::<Vec<_>>(),
        [
            ("cache_tokens", UniformScalar::U32, 0),
            ("chunk_count", UniformScalar::U32, 4),
            ("padding0", UniformScalar::U32, 8),
            ("padding1", UniformScalar::U32, 12),
        ]
    );
    assert_eq!(merge.spec.uniform_span, 16);
    assert!(merge.spec.required_features.is_empty());
    validate_source_contract(&merge.spec, merge.source).unwrap();
}

#[test]
fn decoder_gqa_split_partial_source_pins_the_deterministic_chunk_softmax() {
    let source = module(KernelId::DecoderGqaSplitPartialF32).source;

    // The decoder topology and the split geometry are pinned as constants:
    // 16 query heads and 2 key-value heads of dim 128, the pinned chunk size
    // 32 keys, and the pinned 192-element partials row stride
    // (128 weighted-V elements + chunk max + chunk sum, padded).
    for pinned in [
        "const HEAD_DIM: u32 = 128u;",
        "const QUERY_HEADS: u32 = 16u;",
        "const KEY_VALUE_HEADS: u32 = 2u;",
        "const CHUNK_SIZE: u32 = 32u;",
        "const PARTIAL_STRIDE: u32 = 192u;",
    ] {
        assert!(source.contains(pinned), "missing pinned constant: {pinned}");
    }

    // One workgroup of 64 threads per (query_head, chunk): the dispatch is
    // [query_heads * chunk_count, 1, 1] and the pair is recovered from the
    // workgroup index with the chunk axis fastest.
    assert!(source.contains("let query_head = workgroup_id.x / params.chunk_count;"));
    assert!(source.contains("let chunk = workgroup_id.x % params.chunk_count;"));

    // The accepted scale 1/sqrt(head_dim) with the pinned head dim.
    assert!(source.contains("let attention_scale = inverseSqrt(f32(HEAD_DIM));"));

    // The 32 score threads each compute the full 128-dim dot for exactly one
    // chunk key: ascending hidden order, zero accumulator, and the accepted
    // two-rounding accumulation of the serial decoder_gqa_f32 (the multiply
    // rounds, then the add rounds — the fma() builtin is a different
    // association and is forbidden). The grouped head mapping is the
    // accepted kv_head = q_head / (query_heads / key_value_heads), pinned
    // with the constant group size.
    assert!(
        !source.contains("fma("),
        "the split partial dot must keep the accepted two-rounding accumulation, not fma()"
    );
    assert!(source.contains("var<workgroup> scores: array<f32, 32>;"));
    assert!(source.contains("let query_heads_per_kv = QUERY_HEADS / KEY_VALUE_HEADS;"));
    assert!(source.contains("let key_value_head = query_head / query_heads_per_kv;"));
    assert!(source.contains("let query_base = query_head * HEAD_DIM;"));
    assert!(source.contains("let key_token = chunk * CHUNK_SIZE + local_id.x;"));
    assert!(source.contains("if local_id.x < CHUNK_SIZE && key_token < params.cache_tokens {"));
    assert!(source.contains("var score = 0.0;"));
    assert!(
        source.contains(
            "for (var dimension = 0u; dimension < HEAD_DIM; dimension = dimension + 1u) {"
        )
    );
    assert!(source.contains(
        "let key_base =\n            (key_token * KEY_VALUE_HEADS + key_value_head) * HEAD_DIM;"
    ));
    assert!(source.contains(
        "score = score\n                + query.data[query_base + dimension] * key_cache.data[key_base + dimension];"
    ));
    assert!(source.contains("score = score * attention_scale;"));
    assert!(source.contains("scores[local_id.x] = score;"));
    // Out-of-range score lanes seed the identity of the max reduction so a
    // tail chunk cannot poison the chunk maximum; their exp(score -
    // chunk_max) then underflows to exactly 0.0, seeding the sum identity.
    assert!(source.contains("scores[local_id.x] = -3.402823466e+38;"));

    // The deterministic fixed-shape shared-memory tree reduction over
    // ascending index pairs (32 lanes, halving stride 16..1, each active
    // lane folding its ascending partner into itself) yields the chunk
    // maximum. M7o2 amendment: the tree reduces a separate `maxima` copy —
    // reducing `scores` in place destroyed the per-key scores before the
    // exp(score - chunk_max) weights were computed from them (the M6e2
    // native KV session exposed the corrupted partials against the CPU
    // oracle).
    assert!(source.contains("var<workgroup> maxima: array<f32, 32>;"));
    assert!(source.contains("maxima[local_id.x] = score;"));
    assert!(source.contains("maxima[local_id.x] = -3.402823466e+38;"));
    assert!(
        source.contains("for (var stride = CHUNK_SIZE / 2u; stride > 0u; stride = stride >> 1u) {")
    );
    assert!(source.contains("workgroupBarrier();"));
    assert!(source.contains("if local_id.x < stride {"));
    assert!(
        source
            .contains("maxima[local_id.x] = max(maxima[local_id.x], maxima[local_id.x + stride]);")
    );
    assert!(
        !source
            .contains("scores[local_id.x] = max(scores[local_id.x], scores[local_id.x + stride]);"),
        "the per-key scores must survive the chunk-max reduction intact"
    );
    assert!(source.contains("let chunk_max = maxima[0];"));

    // The chunk weight sum reuses the same fixed association: ascending key
    // order exp(score - chunk_max) per lane, then the same ascending-pair
    // tree, this time summing. M7o2 amendment: the tree reduces a separate
    // `sums` copy — reducing `weights` in place destroyed the per-key
    // weights before the weighted-V accumulation read them (the M6e2 native
    // KV session exposed the corrupted partials against the CPU oracle).
    assert!(source.contains("var<workgroup> weights: array<f32, 32>;"));
    assert!(source.contains("var<workgroup> sums: array<f32, 32>;"));
    assert!(source.contains("let weight = exp(scores[local_id.x] - chunk_max);"));
    assert!(source.contains("weights[local_id.x] = weight;"));
    assert!(source.contains("sums[local_id.x] = weight;"));
    assert!(source.contains("sums[local_id.x] = sums[local_id.x] + sums[local_id.x + stride];"));
    assert!(
        !source
            .contains("weights[local_id.x] = weights[local_id.x] + weights[local_id.x + stride];"),
        "the per-key weights must survive the chunk-sum reduction intact"
    );
    assert!(source.contains("let chunk_sum = sums[0];"));

    // Each of the 64 threads accumulates the unnormalized weighted V for its
    // two dims over the chunk keys in ascending key order (the fixed
    // split-K association), again with the accepted two-rounding mul+add
    // accumulation, then the workgroup writes
    // (weighted_v[128], chunk_max, chunk_sum) into its partials row.
    assert!(
        source
            .contains("for (var dim_offset = 0u; dim_offset < 2u; dim_offset = dim_offset + 1u) {")
    );
    assert!(source.contains("var weighted = 0.0;"));
    assert!(source.contains(
        "for (var key_in_chunk = 0u; key_in_chunk < CHUNK_SIZE; key_in_chunk = key_in_chunk + 1u) {"
    ));
    assert!(source.contains("let weight = weights[key_in_chunk];"));
    assert!(source.contains("let key_token = chunk * CHUNK_SIZE + key_in_chunk;"));
    assert!(source.contains("let dimension = local_id.x * 2u + dim_offset;"));
    // M7o2 amendment: the M6e2 persistent native KV session admits a
    // physically poisoned (non-finite) cache tail past cache_tokens, and
    // 0.0 * NaN is NaN, so the weight-0 masking of out-of-range keys is not
    // enough — the tail V read is masked explicitly with select, preserving
    // the ascending two-rounding accumulation for every in-range key.
    assert!(source.contains(
        "let masked_value = select(\n                0.0,\n                value_cache.data[key_base + dimension],\n                key_token < params.cache_tokens,\n            );"
    ));
    assert!(source.contains("weighted = weighted\n                    + weight * masked_value;"));
    assert!(source.contains(
        "let partial_base = (query_head * params.chunk_count + chunk) * PARTIAL_STRIDE;"
    ));
    assert!(source.contains("partials.data[partial_base + dimension] = weighted;"));
    assert!(source.contains("partials.data[partial_base + HEAD_DIM] = chunk_max;"));
    assert!(source.contains("partials.data[partial_base + HEAD_DIM + 1u] = chunk_sum;"));
}

#[test]
fn decoder_gqa_split_merge_source_merges_partials_in_ascending_chunk_order() {
    let source = module(KernelId::DecoderGqaSplitMergeF32).source;

    for pinned in [
        "const HEAD_DIM: u32 = 128u;",
        "const QUERY_HEADS: u32 = 16u;",
        "const PARTIAL_STRIDE: u32 = 192u;",
    ] {
        assert!(source.contains(pinned), "missing pinned constant: {pinned}");
    }

    // One work item per (query_head, dim) over the [16, 128] output row.
    assert!(source.contains("let linear = global_id.x;"));
    assert!(source.contains("if linear >= QUERY_HEADS * HEAD_DIM {"));
    assert!(source.contains("let query_head = linear / HEAD_DIM;"));
    assert!(source.contains("let dimension = linear % HEAD_DIM;"));

    // The merge runs over the ceil(cache_tokens / 32) partials in ascending
    // chunk order with the standard split-K running maximum/rescaling and a
    // fixed association: the running state, the first-chunk seed, and the
    // rescaling weights are pinned whole so the merge order cannot drift
    // (a reversed chunk order silently changes the association). The fma()
    // builtin is forbidden here too: every rescaling multiply and add rounds
    // separately, exactly like the accepted serial online softmax.
    assert!(
        !source.contains("fma("),
        "the split merge rescaling must keep the accepted two-rounding accumulation, not fma()"
    );
    assert!(
        source.contains("for (var chunk = 0u; chunk < params.chunk_count; chunk = chunk + 1u) {")
    );
    assert!(source.contains(
        "let partial_base = (query_head * params.chunk_count + chunk) * PARTIAL_STRIDE;"
    ));
    assert!(source.contains("let chunk_max = partials.data[partial_base + HEAD_DIM];"));
    assert!(source.contains("let chunk_sum = partials.data[partial_base + HEAD_DIM + 1u];"));
    assert!(source.contains("let chunk_value = partials.data[partial_base + dimension];"));
    assert!(source.contains("var maximum = 0.0;"));
    assert!(source.contains("var denominator = 0.0;"));
    assert!(source.contains("var weighted = 0.0;"));
    assert!(source.contains("var first_chunk = true;"));
    assert!(source.contains("if first_chunk {"));
    assert!(source.contains("maximum = chunk_max;"));
    assert!(source.contains("denominator = chunk_sum;"));
    assert!(source.contains("weighted = chunk_value;"));
    assert!(source.contains("first_chunk = false;"));
    assert!(source.contains("let next_maximum = max(maximum, chunk_max);"));
    assert!(source.contains("let previous_weight = exp(maximum - next_maximum);"));
    assert!(source.contains("let current_weight = exp(chunk_max - next_maximum);"));
    assert!(
        source
            .contains("denominator = denominator * previous_weight + current_weight * chunk_sum;")
    );
    assert!(
        source.contains("weighted = weighted * previous_weight + current_weight * chunk_value;")
    );
    assert!(source.contains("maximum = next_maximum;"));

    // The normalized output row lands at the same [query_head, dim] base.
    assert!(source.contains("output.data[linear] = weighted / denominator;"));
}

#[test]
fn split_modules_extend_the_full_catalog_after_decoder_gqa_f32_and_pass_the_naga_gate() {
    for kernel in [
        KernelId::DecoderGqaSplitPartialF32,
        KernelId::DecoderGqaSplitMergeF32,
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

    // The accepted 20 kernels keep their relative order; the two split-K
    // kernels are inserted in the pinned order partial, merge immediately
    // after decoder_gqa_f32, and the full WGSL catalog mirrors KernelId::ALL
    // exactly.
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
fn split_source_blake3_anchors_are_stable() {
    for (name, kernel, expected_blake3) in [
        (
            "decoder_gqa_split_partial_f32",
            KernelId::DecoderGqaSplitPartialF32,
            DECODER_GQA_SPLIT_PARTIAL_F32_VARIANT_BLAKE3,
        ),
        (
            "decoder_gqa_split_merge_f32",
            KernelId::DecoderGqaSplitMergeF32,
            DECODER_GQA_SPLIT_MERGE_F32_VARIANT_BLAKE3,
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
