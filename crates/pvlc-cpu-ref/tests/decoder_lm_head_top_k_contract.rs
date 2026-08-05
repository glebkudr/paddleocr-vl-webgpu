//! M6c final-norm-to-logits/top-k only; generation and decode-chunk
//! orchestration are intentionally outside this contract.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::UnsafeCell,
    mem::size_of,
};

use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, LmHeadTopKConfig, PINNED_DECODER_VOCAB_SIZE, TopKEntry,
    chunked_lm_head_top_k_f32, pinned_chunked_lm_head_top_k_f32, prefill_last_logits_f32, top_k,
};

const TRACKED_ALLOCATION_SLOTS: usize = 1_024;

struct ThreadCountingAllocator;

struct ThreadAllocationState {
    enabled: bool,
    live_bytes: usize,
    peak_bytes: usize,
    overflowed: bool,
    pointers: [usize; TRACKED_ALLOCATION_SLOTS],
    sizes: [usize; TRACKED_ALLOCATION_SLOTS],
}

impl ThreadAllocationState {
    const fn new() -> Self {
        Self {
            enabled: false,
            live_bytes: 0,
            peak_bytes: 0,
            overflowed: false,
            pointers: [0; TRACKED_ALLOCATION_SLOTS],
            sizes: [0; TRACKED_ALLOCATION_SLOTS],
        }
    }

    fn begin(&mut self) {
        assert!(!self.enabled, "allocation tracking windows may not nest");
        self.live_bytes = 0;
        self.peak_bytes = 0;
        self.overflowed = false;
        self.pointers.fill(0);
        self.sizes.fill(0);
        self.enabled = true;
    }

    fn record_allocation(&mut self, pointer: usize, size: usize) {
        if !self.enabled || size == 0 {
            return;
        }
        let Some(slot) = self.pointers.iter().position(|stored| *stored == 0) else {
            self.overflowed = true;
            return;
        };
        let Some(live_bytes) = self.live_bytes.checked_add(size) else {
            self.overflowed = true;
            return;
        };
        self.pointers[slot] = pointer;
        self.sizes[slot] = size;
        self.live_bytes = live_bytes;
        self.peak_bytes = self.peak_bytes.max(live_bytes);
    }

    fn record_deallocation(&mut self, pointer: usize) {
        if !self.enabled {
            return;
        }
        let Some(slot) = self.pointers.iter().position(|stored| *stored == pointer) else {
            // This allocation predates the measurement window.
            return;
        };
        let size = self.sizes[slot];
        if let Some(live_bytes) = self.live_bytes.checked_sub(size) {
            self.live_bytes = live_bytes;
        } else {
            self.overflowed = true;
        }
        self.pointers[slot] = 0;
        self.sizes[slot] = 0;
    }

    fn finish(&mut self) -> AllocationSnapshot {
        assert!(self.enabled, "allocation tracking window is not active");
        let snapshot = AllocationSnapshot {
            peak_bytes: self.peak_bytes,
            overflowed: self.overflowed,
        };
        self.enabled = false;
        self.pointers.fill(0);
        self.sizes.fill(0);
        snapshot
    }
}

#[derive(Clone, Copy, Debug)]
struct AllocationSnapshot {
    peak_bytes: usize,
    overflowed: bool,
}

thread_local! {
    static THREAD_ALLOCATION_STATE: UnsafeCell<ThreadAllocationState> =
        const { UnsafeCell::new(ThreadAllocationState::new()) };
}

fn with_allocation_state<T>(operation: impl FnOnce(&mut ThreadAllocationState) -> T) -> T {
    THREAD_ALLOCATION_STATE.with(|state| {
        // SAFETY: state is thread-local and allocator callbacks use this
        // synchronous helper, so mutable references cannot overlap.
        operation(unsafe { &mut *state.get() })
    })
}

fn try_with_allocation_state(operation: impl FnOnce(&mut ThreadAllocationState)) {
    let _ = THREAD_ALLOCATION_STATE.try_with(|state| {
        // SAFETY: as above; try_with also avoids TLS teardown panics.
        operation(unsafe { &mut *state.get() });
    });
}

unsafe impl GlobalAlloc for ThreadCountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the exact layout is forwarded to the system allocator.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            try_with_allocation_state(|state| {
                state.record_allocation(pointer as usize, layout.size());
            });
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the exact layout is forwarded to the system allocator.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            try_with_allocation_state(|state| {
                state.record_allocation(pointer as usize, layout.size());
            });
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        try_with_allocation_state(|state| state.record_deallocation(pointer as usize));
        // SAFETY: pointer and layout are the pair supplied by the caller.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: pointer, old layout, and new size are forwarded unchanged.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            try_with_allocation_state(|state| {
                state.record_deallocation(pointer as usize);
                state.record_allocation(new_pointer as usize, new_size);
            });
        }
        new_pointer
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: ThreadCountingAllocator = ThreadCountingAllocator;

fn measure_peak_allocation<T>(operation: impl FnOnce() -> T) -> (T, AllocationSnapshot) {
    with_allocation_state(ThreadAllocationState::begin);
    let output = operation();
    let snapshot = with_allocation_state(ThreadAllocationState::finish);
    (output, snapshot)
}

fn config(hidden_size: usize, vocab_size: usize, k: usize, chunk_size: usize) -> LmHeadTopKConfig {
    LmHeadTopKConfig {
        hidden_size,
        vocab_size,
        k,
        chunk_size,
    }
}

fn full_logits_and_top_k(
    final_norm_one_row: &[f32],
    hidden_size: usize,
    vocab_size: usize,
    k: usize,
    lm_head_weight: &[f32],
) -> (Vec<f32>, Vec<TopKEntry>) {
    let logits = prefill_last_logits_f32(
        final_norm_one_row,
        pvlc_cpu_ref::PrefillLmHeadConfig {
            tokens: 1,
            hidden_size,
            vocab_size,
        },
        lm_head_weight,
    )
    .unwrap();
    let entries = top_k(&logits, k).unwrap();
    (logits, entries)
}

fn weights_for_first_channel(scores: &[f32], hidden_size: usize) -> Vec<f32> {
    let mut weights = vec![0.0_f32; scores.len() * hidden_size];
    for (token_id, score) in scores.iter().copied().enumerate() {
        weights[token_id * hidden_size] = score;
    }
    weights
}

fn hash_f32(values: &[f32]) -> String {
    let mut hasher = blake3::Hasher::new();
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn entry_indices(entries: &[TopKEntry]) -> Vec<usize> {
    entries.iter().map(|entry| entry.index).collect()
}

fn entry_value_bits(entries: &[TopKEntry]) -> Vec<u32> {
    entries.iter().map(|entry| entry.value.to_bits()).collect()
}

fn assert_error(result: Result<Vec<TopKEntry>, CpuRefError>, expected: CpuRefErrorCode) {
    assert_eq!(result.unwrap_err().code(), expected);
}

fn assert_error_without_allocation(
    operation: impl FnOnce() -> Result<Vec<TopKEntry>, CpuRefError>,
    expected: CpuRefErrorCode,
) {
    let (result, allocation) = measure_peak_allocation(operation);
    assert_eq!(result.unwrap_err().code(), expected);
    assert!(!allocation.overflowed, "{allocation:#?}");
    assert_eq!(allocation.peak_bytes, 0, "{allocation:#?}");
}

#[test]
fn literal_output_major_bias_free_dot_matches_fixed_bits_and_blake3() {
    const HIDDEN: usize = 4;
    const VOCAB: usize = 5;
    const FINAL_NORM: [f32; HIDDEN] = [0.2, -1.3, 2.7, 0.0625];
    // Vocabulary output rows, then hidden channels within each row.
    const WEIGHT: [f32; VOCAB * HIDDEN] = [
        0.7, -1.1, 0.3, 2.0, -0.4, 0.9, -1.7, 0.125, 1.2, 0.25, -0.75, 0.6, -2.0, 1.5, 0.5, -0.2,
        0.05, -0.08, 0.13, -0.21,
    ];
    const LOGIT_BITS: [u32; VOCAB] = [
        0x4020_51ec,
        0xc0ba_a148,
        0xc004_a3d8,
        0xbf81_9999,
        0x3ee7_5c28,
    ];
    const LOGITS_BLAKE3: &str = "330604b81e09bab80b697069dbae035d946b5b29c05afebf011a73052afdd6c7";

    let (logits, expected) = full_logits_and_top_k(&FINAL_NORM, HIDDEN, VOCAB, VOCAB, &WEIGHT);
    assert_eq!(
        logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        LOGIT_BITS
    );
    assert_eq!(hash_f32(&logits), LOGITS_BLAKE3);
    assert_eq!(entry_indices(&expected), [0, 4, 3, 2, 1]);
    assert_eq!(
        entry_value_bits(&expected),
        [
            LOGIT_BITS[0],
            LOGIT_BITS[4],
            LOGIT_BITS[3],
            LOGIT_BITS[2],
            LOGIT_BITS[1],
        ]
    );

    let actual =
        chunked_lm_head_top_k_f32(&FINAL_NORM, config(HIDDEN, VOCAB, VOCAB, 2), &WEIGHT).unwrap();
    assert_eq!(actual, expected);

    let zero = chunked_lm_head_top_k_f32(&[0.0; HIDDEN], config(HIDDEN, VOCAB, VOCAB, 2), &WEIGHT)
        .unwrap();
    assert_eq!(entry_indices(&zero), [0, 1, 2, 3, 4]);
    assert!(zero.iter().all(|entry| entry.value == 0.0));
}

#[test]
fn chunk_sizes_around_boundaries_match_full_logits_top_k_exactly() {
    const HIDDEN: usize = 3;
    const SCORES: [f32; 11] = [3.0, 8.0, 1.0, 7.0, 9.0, 5.0, 9.0, -1.0, 4.0, 6.0, 2.0];
    const CHUNK_SIZES: [usize; 7] = [1, 2, 3, 4, 5, 11, 12];
    let final_norm = [1.0_f32, 2.0, 4.0];
    let weights = weights_for_first_channel(&SCORES, HIDDEN);
    let (_, expected) = full_logits_and_top_k(&final_norm, HIDDEN, SCORES.len(), 6, &weights);
    assert_eq!(entry_indices(&expected), [4, 6, 1, 3, 9, 5]);
    assert_eq!(
        entry_value_bits(&expected),
        [
            9.0_f32.to_bits(),
            9.0_f32.to_bits(),
            8.0_f32.to_bits(),
            7.0_f32.to_bits(),
            6.0_f32.to_bits(),
            5.0_f32.to_bits(),
        ]
    );

    for chunk_size in CHUNK_SIZES {
        let actual = chunked_lm_head_top_k_f32(
            &final_norm,
            config(HIDDEN, SCORES.len(), 6, chunk_size),
            &weights,
        )
        .unwrap();
        assert_eq!(actual, expected, "chunk_size={chunk_size}");
    }
}

#[test]
fn more_than_k_equal_maxima_across_chunks_keep_the_globally_smallest_token_ids() {
    let final_norm = [2.0_f32];
    // Six maxima compete for four slots; four different chunks contain at
    // least one winner or loser at the cutoff.
    let weights = [1.0_f32, 5.0, 2.0, 5.0, 5.0, 3.0, 5.0, 4.0, 5.0, 5.0];
    for chunk_size in [1, 2, 3, 4, 5, 7] {
        let actual = chunked_lm_head_top_k_f32(
            &final_norm,
            config(1, weights.len(), 4, chunk_size),
            &weights,
        )
        .unwrap();
        assert_eq!(entry_indices(&actual), [1, 3, 4, 6]);
        assert_eq!(entry_value_bits(&actual), [10.0_f32.to_bits(); 4]);
    }
}

#[test]
fn k_zero_one_and_vocab_match_the_existing_full_logits_baseline() {
    let final_norm = [1.0_f32];
    let weights = [3.0_f32, -1.0, 7.0, 7.0, 2.0];
    for k in [0, 1, weights.len()] {
        let (_, expected) = full_logits_and_top_k(&final_norm, 1, weights.len(), k, &weights);
        let actual =
            chunked_lm_head_top_k_f32(&final_norm, config(1, weights.len(), k, 2), &weights)
                .unwrap();
        assert_eq!(actual, expected, "k={k}");
    }
}

#[test]
fn k_above_vocab_and_zero_chunk_size_are_distinct_fail_closed_errors() {
    let final_norm = [1.0_f32, 2.0];
    let weights = [0.5_f32; 6];
    assert_error(
        chunked_lm_head_top_k_f32(&final_norm, config(2, 3, 4, 1), &weights),
        CpuRefErrorCode::InvalidK,
    );
    assert_error(
        chunked_lm_head_top_k_f32(&final_norm, config(2, 3, 1, 0), &weights),
        CpuRefErrorCode::InvalidTileSize,
    );
}

#[test]
fn generic_geometry_exact_lengths_and_multiplication_overflow_fail_closed() {
    for invalid in [
        config(0, 3, 1, 1),
        config(2, 0, 0, 1),
        config(usize::MAX, 2, 1, 1),
    ] {
        assert_error(
            chunked_lm_head_top_k_f32(&[], invalid, &[]),
            CpuRefErrorCode::DimensionMismatch,
        );
    }
    assert_error(
        chunked_lm_head_top_k_f32(&[0.0; 2], config(2, usize::MAX, 1, 1), &[]),
        CpuRefErrorCode::DimensionMismatch,
    );

    let valid_final_norm = [0.25_f32; 3];
    let valid_weight = [0.5_f32; 12];
    for malformed in [&valid_final_norm[..2], &[0.25_f32; 4][..]] {
        assert_error(
            chunked_lm_head_top_k_f32(malformed, config(3, 4, 2, 2), &valid_weight),
            CpuRefErrorCode::DimensionMismatch,
        );
    }
    for malformed in [&valid_weight[..11], &[0.5_f32; 13][..]] {
        assert_error(
            chunked_lm_head_top_k_f32(&valid_final_norm, config(3, 4, 2, 2), malformed),
            CpuRefErrorCode::DimensionMismatch,
        );
    }
}

#[test]
fn k_zero_still_validates_chunk_shape_lengths_and_finiteness_in_frozen_precedence() {
    let final_norm = [0.25_f32; 3];
    let weight = [0.5_f32; 12];

    assert_error(
        chunked_lm_head_top_k_f32(&final_norm, config(3, 4, 0, 0), &weight),
        CpuRefErrorCode::InvalidTileSize,
    );
    for malformed in [&final_norm[..2], &[0.25_f32; 4][..]] {
        assert_error(
            chunked_lm_head_top_k_f32(malformed, config(3, 4, 0, 2), &weight),
            CpuRefErrorCode::DimensionMismatch,
        );
    }
    for malformed in [&weight[..11], &[0.5_f32; 13][..]] {
        assert_error(
            chunked_lm_head_top_k_f32(&final_norm, config(3, 4, 0, 2), malformed),
            CpuRefErrorCode::DimensionMismatch,
        );
    }

    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut malformed_final_norm = final_norm;
        malformed_final_norm[1] = nonfinite;
        assert_error(
            chunked_lm_head_top_k_f32(&malformed_final_norm, config(3, 4, 0, 2), &weight),
            CpuRefErrorCode::NonFiniteInput,
        );

        let mut malformed_weight = weight;
        malformed_weight[weight.len() / 2] = nonfinite;
        assert_error(
            chunked_lm_head_top_k_f32(&final_norm, config(3, 4, 0, 2), &malformed_weight),
            CpuRefErrorCode::NonFiniteInput,
        );
    }

    assert_error(
        chunked_lm_head_top_k_f32(&[f32::NAN; 2], config(3, 4, 0, 2), &weight),
        CpuRefErrorCode::DimensionMismatch,
    );
    assert_error(
        chunked_lm_head_top_k_f32(&final_norm, config(3, 4, 0, 2), &[f32::NAN; 11]),
        CpuRefErrorCode::DimensionMismatch,
    );
}

#[test]
fn nonfinite_values_at_first_middle_and_last_positions_of_both_operands_are_rejected() {
    let final_norm = [0.25_f32; 3];
    let weight = [0.5_f32; 12];
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for position in [0, final_norm.len() / 2, final_norm.len() - 1] {
            let mut malformed = final_norm;
            malformed[position] = nonfinite;
            assert_error(
                chunked_lm_head_top_k_f32(&malformed, config(3, 4, 2, 2), &weight),
                CpuRefErrorCode::NonFiniteInput,
            );
        }
        for position in [0, weight.len() / 2, weight.len() - 1] {
            let mut malformed = weight;
            malformed[position] = nonfinite;
            assert_error(
                chunked_lm_head_top_k_f32(&final_norm, config(3, 4, 2, 2), &malformed),
                CpuRefErrorCode::NonFiniteInput,
            );
        }
    }
}

#[test]
fn successful_chunked_projection_preserves_both_operands() {
    let final_norm = vec![0.125_f32, -1.5, 2.25, 0.75];
    let weight = (0..36)
        .map(|index| (index as f32 - 17.0) * 0.03125)
        .collect::<Vec<_>>();
    let preserved_final_norm = final_norm.clone();
    let preserved_weight = weight.clone();
    let (_, expected) = full_logits_and_top_k(&final_norm, 4, 9, 5, &weight);

    let actual = chunked_lm_head_top_k_f32(&final_norm, config(4, 9, 5, 4), &weight).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(final_norm, preserved_final_norm);
    assert_eq!(weight, preserved_weight);
}

#[test]
fn pinned_wrapper_rejects_mutually_consistent_h1_and_v1_geometries() {
    const _: [(); 103_424] = [(); PINNED_DECODER_VOCAB_SIZE];

    // These operands are mutually consistent for H=1 and pinned V.
    assert_error(
        pinned_chunked_lm_head_top_k_f32(&[0.0], 1, 64, &vec![0.0; PINNED_DECODER_VOCAB_SIZE]),
        CpuRefErrorCode::DimensionMismatch,
    );

    // These operands are mutually consistent for pinned H and V=1.
    assert_error(
        pinned_chunked_lm_head_top_k_f32(&vec![0.0; 1_024], 1, 64, &vec![0.0; 1_024]),
        CpuRefErrorCode::DimensionMismatch,
    );
}

#[test]
fn peak_temporary_allocation_is_bounded_by_k_plus_chunk_not_vocab() {
    const HIDDEN: usize = 3;
    const VOCAB: usize = 65_537;
    const K: usize = 7;
    const CHUNK: usize = 31;
    const ACCOUNTING_SLACK: usize = 16 * 1_024;

    let final_norm = [1.0_f32, 0.5, -0.25];
    let scores = (0..VOCAB)
        .map(|token_id| (token_id % 101) as f32 - 50.0)
        .collect::<Vec<_>>();
    let weight = weights_for_first_channel(&scores, HIDDEN);
    let (_, expected) = full_logits_and_top_k(&final_norm, HIDDEN, VOCAB, K, &weight);

    let (actual, allocation) = measure_peak_allocation(|| {
        chunked_lm_head_top_k_f32(&final_norm, config(HIDDEN, VOCAB, K, CHUNK), &weight).unwrap()
    });
    assert_eq!(actual, expected);
    assert!(!allocation.overflowed, "{allocation:#?}");

    let bounded_peak = (K + CHUNK) * size_of::<TopKEntry>() * 8 + ACCOUNTING_SLACK;
    assert!(
        allocation.peak_bytes <= bounded_peak,
        "temporary allocation scaled beyond O(k + chunk): {allocation:#?}, bound={bounded_peak}"
    );
    assert!(
        allocation.peak_bytes < VOCAB * size_of::<f32>(),
        "a full vocabulary logits buffer was materialized: {allocation:#?}"
    );
}

#[test]
fn representative_invalid_calls_allocate_nothing_before_failing_closed() {
    let final_norm = [0.25_f32; 3];
    let weight = [0.5_f32; 12];
    let short_final_norm = [0.25_f32; 2];
    let overflow_final_norm = [0.0_f32; 2];
    let empty_weight: [f32; 0] = [];
    let mut nonfinite_weight = weight;
    nonfinite_weight[6] = f32::NAN;

    assert_error_without_allocation(
        || {
            chunked_lm_head_top_k_f32(
                &overflow_final_norm,
                config(2, usize::MAX, 1, 1),
                &empty_weight,
            )
        },
        CpuRefErrorCode::DimensionMismatch,
    );
    assert_error_without_allocation(
        || chunked_lm_head_top_k_f32(&short_final_norm, config(3, 4, 2, 2), &weight),
        CpuRefErrorCode::DimensionMismatch,
    );
    assert_error_without_allocation(
        || chunked_lm_head_top_k_f32(&final_norm, config(3, 4, 2, 2), &nonfinite_weight),
        CpuRefErrorCode::NonFiniteInput,
    );
    assert_error_without_allocation(
        || chunked_lm_head_top_k_f32(&final_norm, config(3, 4, 5, 2), &weight),
        CpuRefErrorCode::InvalidK,
    );
    assert_error_without_allocation(
        || chunked_lm_head_top_k_f32(&final_norm, config(3, 4, 2, 0), &weight),
        CpuRefErrorCode::InvalidTileSize,
    );
}
