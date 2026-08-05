//! Structural contract for the M7o2 split-K decode GQA step planner before
//! any production exists (docs/m7o2_split_k_decode_gqa_contract.md).
//!
//! `DecoderGqaSplitDescriptor { cache_tokens, query_heads, key_value_heads,
//! head_dim }` plans the deterministic split-K replacement of the accepted
//! serial `decoder_gqa_f32` decode dispatch: the pinned chunk size 32, the
//! partial invocation (one workgroup per (query_head, chunk)), the merge
//! invocation (one work item per (query_head, dim)), the exact partials
//! scratch geometry, and the two identical position-dependent uniform word
//! sets written per step.

use pvlc_runtime_core::{DecoderGqaSplitDescriptor, InvocationErrorCode, InvocationPlan, KernelId};

const CACHE_TOKENS: u32 = 332;
const QUERY_HEADS: u32 = 16;
const KEY_VALUE_HEADS: u32 = 2;
const HEAD_DIM: u32 = 128;

const CHUNK_SIZE: u32 = 32;
const CHUNK_COUNT: u32 = 11; // ceil(332 / 32)
const PARTIAL_STRIDE_F32: u32 = 192;
const PARTIALS_ELEMENTS: usize = QUERY_HEADS as usize * CHUNK_COUNT as usize * 192;
const PARTIALS_BYTES: u64 = PARTIALS_ELEMENTS as u64 * 4;
const OUTPUT_ELEMENTS: usize = QUERY_HEADS as usize * HEAD_DIM as usize;
const OUTPUT_BYTES: u64 = OUTPUT_ELEMENTS as u64 * 4;

fn descriptor(cache_tokens: u32) -> DecoderGqaSplitDescriptor {
    DecoderGqaSplitDescriptor {
        cache_tokens,
        query_heads: QUERY_HEADS,
        key_value_heads: KEY_VALUE_HEADS,
        head_dim: HEAD_DIM,
    }
}

fn assert_descriptor_error(descriptor: DecoderGqaSplitDescriptor, expected: InvocationErrorCode) {
    let error = descriptor.plan().unwrap_err();
    assert_eq!(error.code(), expected, "{error}");
}

fn expected_partial_invocation() -> InvocationPlan {
    InvocationPlan {
        kernel: KernelId::DecoderGqaSplitPartialF32,
        output_elements: PARTIALS_ELEMENTS,
        output_bytes: PARTIALS_BYTES,
        workgroup_size: [64, 1, 1],
        dispatch: [QUERY_HEADS * CHUNK_COUNT, 1, 1],
    }
}

fn expected_merge_invocation() -> InvocationPlan {
    InvocationPlan {
        kernel: KernelId::DecoderGqaSplitMergeF32,
        output_elements: OUTPUT_ELEMENTS,
        output_bytes: OUTPUT_BYTES,
        workgroup_size: [64, 1, 1],
        dispatch: [(QUERY_HEADS * HEAD_DIM).div_ceil(64), 1, 1],
    }
}

#[test]
fn plan_pins_the_exact_split_k_lattice() {
    let plan = descriptor(CACHE_TOKENS).plan().expect("plan");

    assert_eq!(plan.chunk_size, CHUNK_SIZE);
    assert_eq!(plan.chunk_count, CHUNK_COUNT);
    assert_eq!(plan.partial_stride_f32, PARTIAL_STRIDE_F32);
    assert_eq!(plan.partials_elements, PARTIALS_ELEMENTS);
    assert_eq!(plan.partials_bytes, PARTIALS_BYTES);
    assert_eq!(plan.partial_invocation, expected_partial_invocation());
    assert_eq!(plan.merge_invocation, expected_merge_invocation());
    assert_eq!(
        plan.uniform_words,
        [
            [CACHE_TOKENS, CHUNK_COUNT, 0, 0],
            [CACHE_TOKENS, CHUNK_COUNT, 0, 0],
        ]
    );
}

#[test]
fn chunk_count_is_the_ceil_of_cache_tokens_over_the_pinned_chunk_size() {
    for (cache_tokens, chunk_count) in [
        (1, 1),
        (31, 1),
        (32, 1),
        (33, 2),
        (64, 2),
        (332, 11),
        (337, 11),
    ] {
        let plan = descriptor(cache_tokens).plan().expect("plan");
        assert_eq!(
            plan.chunk_count, chunk_count,
            "cache_tokens {cache_tokens} must split into {chunk_count} chunks"
        );
        assert_eq!(plan.chunk_size, CHUNK_SIZE);
        assert_eq!(
            plan.partial_invocation.dispatch,
            [QUERY_HEADS * chunk_count, 1, 1]
        );
        assert_eq!(
            plan.partials_elements,
            QUERY_HEADS as usize * chunk_count as usize * PARTIAL_STRIDE_F32 as usize
        );
        assert_eq!(
            plan.uniform_words,
            [
                [cache_tokens, chunk_count, 0, 0],
                [cache_tokens, chunk_count, 0, 0],
            ]
        );
    }
}

#[test]
fn plan_is_deterministic_and_pure() {
    let first = descriptor(CACHE_TOKENS).plan().expect("first plan");
    let second = descriptor(CACHE_TOKENS).plan().expect("second plan");
    assert_eq!(first, second);
}

#[test]
fn descriptor_rejects_zero_cache_tokens() {
    assert_descriptor_error(descriptor(0), InvocationErrorCode::InvalidDecoderGeometry);
}

#[test]
fn descriptor_rejects_zero_query_heads() {
    assert_descriptor_error(
        DecoderGqaSplitDescriptor {
            query_heads: 0,
            ..descriptor(CACHE_TOKENS)
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
}

#[test]
fn descriptor_rejects_zero_key_value_heads() {
    assert_descriptor_error(
        DecoderGqaSplitDescriptor {
            key_value_heads: 0,
            ..descriptor(CACHE_TOKENS)
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
}

#[test]
fn descriptor_rejects_non_grouped_query_heads() {
    // The accepted grouped mapping kv_head = q_head / (query_heads /
    // key_value_heads) requires an exact integer group size.
    assert_descriptor_error(
        DecoderGqaSplitDescriptor {
            key_value_heads: 3,
            ..descriptor(CACHE_TOKENS)
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderGqaSplitDescriptor {
            query_heads: 8,
            key_value_heads: 16,
            ..descriptor(CACHE_TOKENS)
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
}

#[test]
fn descriptor_rejects_unpinned_head_counts() {
    // Even a valid GQA grouping is geometry drift when it is not the pinned
    // 16 query heads / 2 key-value heads: the split kernels pin the decoder
    // topology (the partials plane, the work-item lattices, and the grouped
    // mapping constants) instead of inferring them.
    assert_descriptor_error(
        DecoderGqaSplitDescriptor {
            query_heads: 8,
            key_value_heads: 2,
            ..descriptor(CACHE_TOKENS)
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderGqaSplitDescriptor {
            query_heads: 16,
            key_value_heads: 4,
            ..descriptor(CACHE_TOKENS)
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
}

#[test]
fn descriptor_rejects_unpinned_head_dim() {
    // The split kernels pin the 128-dim head geometry and the 192-element
    // partials stride; any other head dim is geometry drift.
    for head_dim in [0, 64, 256] {
        assert_descriptor_error(
            DecoderGqaSplitDescriptor {
                head_dim,
                ..descriptor(CACHE_TOKENS)
            },
            InvocationErrorCode::InvalidDecoderGeometry,
        );
    }
}

#[test]
fn descriptor_accepts_the_maximum_admitted_cache_span() {
    // ceil(131040 / 32) = 4095 chunks -> 16 * 4095 = 65520 workgroups, the
    // largest partial dispatch inside the WebGPU single-dispatch bound.
    let plan = descriptor(131_040).plan().expect("maximum admitted span");
    assert_eq!(plan.chunk_count, 4095);
    assert_eq!(plan.partial_invocation.dispatch, [65_520, 1, 1]);
}

#[test]
fn descriptor_rejects_partial_dispatch_overflow() {
    // ceil(131072 / 32) = 4096 chunks -> 16 * 4096 = 65536 workgroups, the
    // first cache span exceeding the 65535 single-dispatch bound.
    assert_descriptor_error(
        descriptor(131_072),
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        descriptor(200_000),
        InvocationErrorCode::InvalidDecoderGeometry,
    );
}

#[test]
fn pinned_descriptor_supplies_the_literal_geometry() {
    let plan = DecoderGqaSplitDescriptor::pinned(CACHE_TOKENS)
        .plan()
        .expect("pinned plan");
    assert_eq!(plan.chunk_size, CHUNK_SIZE);
    assert_eq!(plan.chunk_count, CHUNK_COUNT);
    assert_eq!(plan.partial_stride_f32, PARTIAL_STRIDE_F32);
    assert_eq!(plan.partial_invocation, expected_partial_invocation());
    assert_eq!(plan.merge_invocation, expected_merge_invocation());
    assert_eq!(
        plan.uniform_words,
        [
            [CACHE_TOKENS, CHUNK_COUNT, 0, 0],
            [CACHE_TOKENS, CHUNK_COUNT, 0, 0],
        ]
    );
}

#[test]
fn pinned_descriptor_matches_the_explicit_pinned_geometry() {
    let explicit = descriptor(CACHE_TOKENS).plan().expect("explicit plan");
    let pinned = DecoderGqaSplitDescriptor::pinned(CACHE_TOKENS)
        .plan()
        .expect("pinned plan");
    assert_eq!(explicit, pinned);
}
