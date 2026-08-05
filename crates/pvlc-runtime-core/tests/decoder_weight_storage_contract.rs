//! M7q1 planner contract for weight-only FP16 decoder storage.
//!
//! Decoder arithmetic plans stay F32. This helper derives only the physical
//! byte spans and per-layer dynamic offsets of immutable weight buffers.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use pvlc_runtime_core::{
    DecoderAttentionBlockPlan, DecoderLayerPlan, DecoderLmHeadGeometryDescriptor,
    DecoderLmHeadPlan, DecoderStackGeometryDescriptor, DecoderStackPlan, DecoderStackPrefillPlan,
    DecoderWeightResourceDescriptor, DecoderWeightStorage, DecoderWeightStorageErrorCode,
    InvocationErrorCode, InvocationPlan, KernelId,
};

const LAYERS: u32 = 18;
const HIDDEN: u64 = 1_024;
const QUERY: u64 = 2_048;
const KEY_VALUE: u64 = 256;
const INTERMEDIATE: u64 = 3_072;
const CACHE_CAPACITY: u64 = 337;
const HEAD_DIM: u64 = 128;
const VOCAB: u64 = 103_424;
const F32_LAYER_STRIDES: [u64; 9] = [
    HIDDEN * 4,
    QUERY * HIDDEN * 4,
    KEY_VALUE * HIDDEN * 4,
    KEY_VALUE * HIDDEN * 4,
    HIDDEN * QUERY * 4,
    HIDDEN * 4,
    INTERMEDIATE * HIDDEN * 4,
    INTERMEDIATE * HIDDEN * 4,
    HIDDEN * INTERMEDIATE * 4,
];
const F32_ROPE_TABLE_BYTES: u64 = 3 * CACHE_CAPACITY * HEAD_DIM * 4;
const F32_FINAL_NORM_BYTES: u64 = HIDDEN * 4;
const F32_LM_HEAD_BYTES: u64 = VOCAB * HIDDEN * 4;
const PREFILL_TOKENS: u32 = 332;
const GEOMETRY_PLAN_TOTAL_ALLOCATION_BUDGET: usize = 16 * 1024 * 1024;
const GEOMETRY_PLAN_SINGLE_ALLOCATION_BUDGET: usize = 1024 * 1024;

struct TrackingAllocator;

static TRACK_GEOMETRY_PLAN_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static GEOMETRY_PLAN_ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static GEOMETRY_PLAN_LARGEST_ALLOCATION: AtomicUsize = AtomicUsize::new(0);

fn record_geometry_plan_allocation(bytes: usize) {
    if TRACK_GEOMETRY_PLAN_ALLOCATIONS.load(Ordering::Relaxed) {
        GEOMETRY_PLAN_ALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
        GEOMETRY_PLAN_LARGEST_ALLOCATION.fetch_max(bytes, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_geometry_plan_allocation(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_geometry_plan_allocation(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_geometry_plan_allocation(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator;

struct GeometryPlanAllocationWindow;

impl GeometryPlanAllocationWindow {
    fn start() -> Self {
        GEOMETRY_PLAN_ALLOCATED_BYTES.store(0, Ordering::Relaxed);
        GEOMETRY_PLAN_LARGEST_ALLOCATION.store(0, Ordering::Relaxed);
        TRACK_GEOMETRY_PLAN_ALLOCATIONS.store(true, Ordering::Release);
        Self
    }
}

impl Drop for GeometryPlanAllocationWindow {
    fn drop(&mut self) {
        TRACK_GEOMETRY_PLAN_ALLOCATIONS.store(false, Ordering::Release);
    }
}

fn descriptor(storage: DecoderWeightStorage, with_logits: bool) -> DecoderWeightResourceDescriptor {
    DecoderWeightResourceDescriptor {
        layers: LAYERS,
        f32_layer_weight_stride_bytes: F32_LAYER_STRIDES,
        f32_rope_table_bytes: F32_ROPE_TABLE_BYTES,
        f32_final_norm_weight_bytes: with_logits.then_some(F32_FINAL_NORM_BYTES),
        f32_lm_head_weight_bytes: with_logits.then_some(F32_LM_HEAD_BYTES),
        storage,
    }
}

fn stack_geometry() -> DecoderStackGeometryDescriptor {
    DecoderStackGeometryDescriptor {
        layers: LAYERS,
        hidden_size: HIDDEN as u32,
        intermediate_size: INTERMEDIATE as u32,
        query_heads: 16,
        key_value_heads: 2,
        head_dim: HEAD_DIM as u32,
        rms_norm_epsilon: 1e-5,
        cache_capacity: CACHE_CAPACITY as u32,
    }
}

fn invocation(
    kernel: KernelId,
    output_elements: u32,
    workgroup_size: [u32; 3],
    dispatch: [u32; 3],
) -> InvocationPlan {
    InvocationPlan {
        kernel,
        output_elements: output_elements as usize,
        output_bytes: u64::from(output_elements) * 4,
        workgroup_size,
        dispatch,
    }
}

fn single_row_stage(kernel: KernelId, output_elements: u32) -> InvocationPlan {
    invocation(
        kernel,
        output_elements,
        [64, 1, 1],
        [output_elements.div_ceil(64), 1, 1],
    )
}

fn gemv_stage(output_width: u32) -> InvocationPlan {
    invocation(
        KernelId::GemvTiledF32,
        output_width,
        [256, 1, 1],
        [output_width.div_ceil(8), 1, 1],
    )
}

fn expected_layer_plan() -> DecoderLayerPlan {
    let hidden = HIDDEN as u32;
    let query_width = QUERY as u32;
    let key_value_width = KEY_VALUE as u32;
    let intermediate = INTERMEDIATE as u32;
    let mrope_width = query_width + key_value_width;

    DecoderLayerPlan {
        attention_block: DecoderAttentionBlockPlan {
            hidden_size: hidden,
            query_heads: 16,
            key_value_heads: 2,
            head_dim: HEAD_DIM as u32,
            query_width: query_width as usize,
            key_value_width: key_value_width as usize,
            rope_elements: (3 * CACHE_CAPACITY * HEAD_DIM) as usize,
            cache_capacity: CACHE_CAPACITY as u32,
            rms_norm_epsilon: 1e-5,
            mrope_sections: [16, 24, 24],
            rms_norm_invocation: invocation(KernelId::RmsNormF32, hidden, [64, 1, 1], [1, 1, 1]),
            query_invocation: gemv_stage(query_width),
            key_invocation: gemv_stage(key_value_width),
            value_invocation: gemv_stage(key_value_width),
            output_invocation: gemv_stage(hidden),
            mrope_invocation: single_row_stage(KernelId::DecoderMropeF32, mrope_width),
            residual_invocation: single_row_stage(KernelId::AddF32, hidden),
        },
        intermediate_size: intermediate,
        norm2_invocation: invocation(KernelId::RmsNormF32, hidden, [64, 1, 1], [1, 1, 1]),
        gate_invocation: gemv_stage(intermediate),
        up_invocation: gemv_stage(intermediate),
        swiglu_invocation: single_row_stage(KernelId::DecoderSwigluF32, intermediate),
        down_invocation: gemv_stage(hidden),
        second_residual_invocation: single_row_stage(KernelId::AddF32, hidden),
    }
}

fn prefill_row_stage(kernel: KernelId, output_elements: u32, work_items: u32) -> InvocationPlan {
    invocation(
        kernel,
        output_elements,
        [64, 1, 1],
        [work_items.div_ceil(64), 1, 1],
    )
}

fn prefill_projection_stage(tokens: u32, output_width: u32) -> InvocationPlan {
    invocation(
        KernelId::VisionPatchProjectionF32,
        tokens * output_width,
        [8, 8, 1],
        [output_width.div_ceil(32), tokens.div_ceil(32), 1],
    )
}

fn expected_prefill_stage_invocations(tokens: u32) -> [InvocationPlan; 15] {
    let hidden = HIDDEN as u32;
    let query_width = QUERY as u32;
    let key_value_width = KEY_VALUE as u32;
    let intermediate = INTERMEDIATE as u32;
    let mrope_width = query_width + key_value_width;
    let hidden_elements = tokens * hidden;
    let intermediate_elements = tokens * intermediate;

    [
        prefill_row_stage(KernelId::RmsNormF32, hidden_elements, tokens),
        prefill_projection_stage(tokens, query_width),
        prefill_projection_stage(tokens, key_value_width),
        prefill_projection_stage(tokens, key_value_width),
        prefill_row_stage(
            KernelId::DecoderPrefillMropeF32,
            tokens * mrope_width,
            tokens * mrope_width,
        ),
        prefill_row_stage(
            KernelId::DecoderKvAppendRangeF32,
            2 * CACHE_CAPACITY as u32 * key_value_width,
            tokens * key_value_width,
        ),
        prefill_row_stage(
            KernelId::DecoderPrefillGqaF32,
            tokens * query_width,
            tokens * 16,
        ),
        prefill_projection_stage(tokens, hidden),
        prefill_row_stage(KernelId::AddF32, hidden_elements, hidden_elements),
        prefill_row_stage(KernelId::RmsNormF32, hidden_elements, tokens),
        prefill_projection_stage(tokens, intermediate),
        prefill_projection_stage(tokens, intermediate),
        prefill_row_stage(
            KernelId::DecoderSwigluF32,
            intermediate_elements,
            intermediate_elements,
        ),
        prefill_projection_stage(tokens, hidden),
        prefill_row_stage(KernelId::AddF32, hidden_elements, hidden_elements),
    ]
}

fn expected_prefill_stage_uniform_words(tokens: u32) -> [[u32; 4]; 15] {
    let hidden = HIDDEN as u32;
    let query_width = QUERY as u32;
    let key_value_width = KEY_VALUE as u32;
    let intermediate = INTERMEDIATE as u32;

    [
        [tokens, hidden, 1e-5_f32.to_bits(), 0],
        [tokens, hidden, query_width, 0],
        [tokens, hidden, key_value_width, 0],
        [tokens, hidden, key_value_width, 0],
        [tokens, CACHE_CAPACITY as u32, 0, 0],
        [tokens, CACHE_CAPACITY as u32, 0, 0],
        [tokens, 16, 2, HEAD_DIM as u32],
        [tokens, query_width, hidden, 0],
        [tokens * hidden, 0, 0, 0],
        [tokens, hidden, 1e-5_f32.to_bits(), 0],
        [tokens, hidden, intermediate, 0],
        [tokens, hidden, intermediate, 0],
        [tokens * intermediate, 0, 0, 0],
        [tokens, intermediate, hidden, 0],
        [tokens * hidden, 0, 0, 0],
    ]
}

fn expected_logits_plan() -> DecoderLmHeadPlan {
    let hidden = HIDDEN as u32;
    let vocab = VOCAB as u32;

    DecoderLmHeadPlan {
        hidden_size: hidden,
        vocab_size: vocab,
        final_norm_weight_bytes: F32_FINAL_NORM_BYTES,
        lm_head_weight_bytes: F32_LM_HEAD_BYTES,
        normed_row_bytes: F32_FINAL_NORM_BYTES,
        logits_bytes: VOCAB * 4,
        stage_invocations: [
            invocation(KernelId::RmsNormF32, hidden, [64, 1, 1], [1, 1, 1]),
            gemv_stage(vocab),
        ],
        stage_uniform_words: [[1, hidden, 1e-5_f32.to_bits(), 0], [vocab, hidden, 0, 0]],
    }
}

#[test]
fn geometry_only_planners_match_every_accepted_stack_prefill_and_logits_field() {
    let geometry = stack_geometry();
    let stack = geometry.plan().unwrap();
    assert_eq!(
        stack,
        DecoderStackPlan {
            layers: LAYERS,
            layer_plan: expected_layer_plan(),
            weight_stride_bytes: F32_LAYER_STRIDES,
            cache_stride_bytes: CACHE_CAPACITY * KEY_VALUE * 4,
            hidden_stride_bytes: HIDDEN * 4,
        }
    );

    let prefill = geometry.plan_prefill(PREFILL_TOKENS).unwrap();
    assert_eq!(
        prefill,
        DecoderStackPrefillPlan {
            layers: LAYERS,
            tokens: PREFILL_TOKENS,
            cache_capacity: CACHE_CAPACITY as u32,
            weight_stride_bytes: F32_LAYER_STRIDES,
            cache_stride_bytes: CACHE_CAPACITY * KEY_VALUE * 4,
            hidden_stride_bytes: HIDDEN * 4,
            stage_invocations: expected_prefill_stage_invocations(PREFILL_TOKENS),
            stage_uniform_words: expected_prefill_stage_uniform_words(PREFILL_TOKENS),
        }
    );

    let logits = DecoderLmHeadGeometryDescriptor::pinned().plan().unwrap();
    assert_eq!(logits, expected_logits_plan());
}

#[test]
fn geometry_only_planners_reject_every_invalid_geometry_field_and_token_bound() {
    let base = stack_geometry();
    for invalid in [
        DecoderStackGeometryDescriptor {
            layers: LAYERS - 1,
            ..base
        },
        DecoderStackGeometryDescriptor {
            hidden_size: HIDDEN as u32 - 1,
            ..base
        },
        DecoderStackGeometryDescriptor {
            intermediate_size: INTERMEDIATE as u32 - 1,
            ..base
        },
        DecoderStackGeometryDescriptor {
            query_heads: 15,
            ..base
        },
        DecoderStackGeometryDescriptor {
            key_value_heads: 1,
            ..base
        },
        DecoderStackGeometryDescriptor {
            head_dim: HEAD_DIM as u32 - 1,
            ..base
        },
        DecoderStackGeometryDescriptor {
            rms_norm_epsilon: 1e-6,
            ..base
        },
        DecoderStackGeometryDescriptor {
            cache_capacity: 0,
            ..base
        },
    ] {
        let plan_error = invalid.plan().unwrap_err();
        assert_eq!(
            plan_error.code(),
            InvocationErrorCode::InvalidDecoderGeometry,
            "{invalid:?} returned the wrong plan error: {plan_error}"
        );
        let prefill_error = invalid.plan_prefill(1).unwrap_err();
        assert_eq!(
            prefill_error.code(),
            InvocationErrorCode::InvalidDecoderGeometry,
            "{invalid:?} returned the wrong prefill error: {prefill_error}"
        );
    }
    assert_eq!(base.plan_prefill(1).unwrap().tokens, 1);
    assert_eq!(
        base.plan_prefill(CACHE_CAPACITY as u32).unwrap().tokens,
        CACHE_CAPACITY as u32
    );
    assert!(base.plan_prefill(0).is_err());
    assert!(
        base.plan_prefill(CACHE_CAPACITY as u32 + 1).is_err(),
        "prefill beyond the admitted cache capacity was admitted"
    );

    let lm_head = DecoderLmHeadGeometryDescriptor::pinned();
    for invalid in [
        DecoderLmHeadGeometryDescriptor {
            hidden_size: 0,
            ..lm_head
        },
        DecoderLmHeadGeometryDescriptor {
            vocab_size: 0,
            ..lm_head
        },
        DecoderLmHeadGeometryDescriptor {
            rms_norm_epsilon: 0.0,
            ..lm_head
        },
        DecoderLmHeadGeometryDescriptor {
            rms_norm_epsilon: f32::NAN,
            ..lm_head
        },
        DecoderLmHeadGeometryDescriptor {
            vocab_size: u32::MAX,
            ..lm_head
        },
    ] {
        assert!(invalid.plan().is_err(), "{invalid:?} was admitted");
    }
}

#[test]
fn geometry_only_planners_have_a_bounded_allocation_footprint() {
    let geometry = stack_geometry();
    let allocation_window = GeometryPlanAllocationWindow::start();
    let stack = geometry.plan();
    let prefill = geometry.plan_prefill(PREFILL_TOKENS);
    let logits = DecoderLmHeadGeometryDescriptor::pinned().plan();
    let allocated_bytes = GEOMETRY_PLAN_ALLOCATED_BYTES.load(Ordering::Acquire);
    let largest_allocation = GEOMETRY_PLAN_LARGEST_ALLOCATION.load(Ordering::Acquire);
    drop(allocation_window);

    stack.unwrap();
    prefill.unwrap();
    logits.unwrap();
    assert!(
        allocated_bytes <= GEOMETRY_PLAN_TOTAL_ALLOCATION_BUDGET,
        "geometry-only planning allocated {allocated_bytes} bytes; expected at most \
         {GEOMETRY_PLAN_TOTAL_ALLOCATION_BUDGET}"
    );
    assert!(
        largest_allocation <= GEOMETRY_PLAN_SINGLE_ALLOCATION_BUDGET,
        "geometry-only planning made a {largest_allocation}-byte allocation; expected no \
         allocation larger than {GEOMETRY_PLAN_SINGLE_ALLOCATION_BUDGET}"
    );
}

#[test]
fn weight_storage_scales_only_immutable_weight_bytes() {
    assert_eq!(DecoderWeightStorage::F32.bytes_per_element(), 4);
    assert_eq!(DecoderWeightStorage::F16.bytes_per_element(), 2);

    for elements in [1_u64, 1024, 2048 * 1024, 103_424 * 1024] {
        assert_eq!(
            DecoderWeightStorage::F32.storage_bytes(elements),
            elements.checked_mul(4)
        );
        assert_eq!(
            DecoderWeightStorage::F16.storage_bytes(elements),
            elements.checked_mul(2)
        );
    }

    // Activations and caches are deliberately outside this API. Their
    // accepted F32 sizes are used directly and never precision-scaled.
    let hidden_activation_bytes = 1024_u64 * 4;
    let compact_cache_plane_bytes = 337_u64 * 256 * 4;
    assert_eq!(hidden_activation_bytes, 4096);
    assert_eq!(compact_cache_plane_bytes, 345_088);
}

#[test]
fn fidelity_resource_plan_preserves_all_checkpoint_and_table_bytes() {
    let plan = descriptor(DecoderWeightStorage::F32, true).plan().unwrap();

    assert_eq!(plan.storage, DecoderWeightStorage::F32);
    assert_eq!(plan.layers, LAYERS);
    assert_eq!(plan.layer_weight_stride_bytes, F32_LAYER_STRIDES);
    assert_eq!(
        plan.layer_weight_bulk_bytes,
        F32_LAYER_STRIDES.map(|stride| stride * u64::from(LAYERS))
    );
    assert_eq!(plan.rope_table_bytes, F32_ROPE_TABLE_BYTES);
    assert_eq!(plan.final_norm_weight_bytes, Some(F32_FINAL_NORM_BYTES));
    assert_eq!(plan.lm_head_weight_bytes, Some(F32_LM_HEAD_BYTES));
    assert_eq!(plan.checkpoint_shard_count, 11);
    assert_eq!(plan.f16_checkpoint_shard_count, 0);
    assert_eq!(plan.f32_checkpoint_shard_count, 11);
    assert_eq!(plan.f32_table_shard_count, 2);
    assert!(!plan.requires_shader_f16());
}

#[test]
fn balanced_resource_plan_halves_exactly_eleven_weights_but_not_two_rope_tables() {
    let plan = descriptor(DecoderWeightStorage::F16, true).plan().unwrap();

    assert_eq!(plan.storage, DecoderWeightStorage::F16);
    assert_eq!(
        plan.layer_weight_stride_bytes,
        F32_LAYER_STRIDES.map(|stride| stride / 2)
    );
    assert_eq!(
        plan.layer_weight_bulk_bytes,
        F32_LAYER_STRIDES.map(|stride| stride / 2 * u64::from(LAYERS))
    );
    assert_eq!(plan.rope_table_bytes, F32_ROPE_TABLE_BYTES);
    assert_eq!(plan.final_norm_weight_bytes, Some(F32_FINAL_NORM_BYTES / 2));
    assert_eq!(plan.lm_head_weight_bytes, Some(F32_LM_HEAD_BYTES / 2));
    assert_eq!(plan.checkpoint_shard_count, 11);
    assert_eq!(plan.f16_checkpoint_shard_count, 11);
    assert_eq!(plan.f32_checkpoint_shard_count, 0);
    assert_eq!(plan.f32_table_shard_count, 2);
    assert!(plan.requires_shader_f16());

    // Runtime-owned mutable storage is intentionally not represented by the
    // weight plan and therefore retains its accepted F32 byte widths.
    let hidden_activation_bytes = HIDDEN * 4;
    let compact_cache_plane_bytes = CACHE_CAPACITY * KEY_VALUE * 4;
    let logits_bytes = VOCAB * 4;
    assert_eq!(hidden_activation_bytes, 4_096);
    assert_eq!(compact_cache_plane_bytes, 345_088);
    assert_eq!(logits_bytes, 413_696);
}

#[test]
fn legacy_and_logits_plans_have_closed_world_shard_partitions() {
    let legacy = descriptor(DecoderWeightStorage::F16, false).plan().unwrap();
    assert_eq!(legacy.checkpoint_shard_count, 9);
    assert_eq!(legacy.f16_checkpoint_shard_count, 9);
    assert_eq!(legacy.f32_checkpoint_shard_count, 0);
    assert_eq!(legacy.f32_table_shard_count, 2);
    assert_eq!(legacy.final_norm_weight_bytes, None);
    assert_eq!(legacy.lm_head_weight_bytes, None);

    let logits = descriptor(DecoderWeightStorage::F16, true).plan().unwrap();
    assert_eq!(logits.checkpoint_shard_count, 11);
    assert_eq!(logits.f16_checkpoint_shard_count, 11);
    assert_eq!(logits.f32_table_shard_count, 2);

    for (final_norm, lm_head) in [
        (Some(F32_FINAL_NORM_BYTES), None),
        (None, Some(F32_LM_HEAD_BYTES)),
    ] {
        let mut invalid = descriptor(DecoderWeightStorage::F16, false);
        invalid.f32_final_norm_weight_bytes = final_norm;
        invalid.f32_lm_head_weight_bytes = lm_head;
        let error = invalid.plan().unwrap_err();
        assert_eq!(
            error.code(),
            DecoderWeightStorageErrorCode::IncompleteLogitsWeights
        );
    }
}

#[test]
fn decode_prefill_and_logits_weight_ranges_share_one_offset_authority() {
    for storage in [DecoderWeightStorage::F32, DecoderWeightStorage::F16] {
        let plan = descriptor(storage, true).plan().unwrap();
        for layer in [0, 1, LAYERS - 1] {
            assert_eq!(
                plan.layer_weight_offsets(layer),
                Some(
                    plan.layer_weight_stride_bytes
                        .map(|stride| u64::from(layer) * stride)
                )
            );
            for slot in 0..F32_LAYER_STRIDES.len() {
                assert_eq!(
                    plan.layer_weight_offset(layer, slot),
                    Some(u64::from(layer) * plan.layer_weight_stride_bytes[slot]),
                    "layer {layer}, slot {slot}, storage {storage:?}"
                );
            }
        }
        assert_eq!(plan.layer_weight_offsets(LAYERS), None);
        assert_eq!(plan.layer_weight_offset(LAYERS, 0), None);
        assert_eq!(plan.layer_weight_offset(0, F32_LAYER_STRIDES.len()), None);
        assert_eq!(
            plan.layer_weight_range(LAYERS - 1, 8),
            Some((
                u64::from(LAYERS - 1) * plan.layer_weight_stride_bytes[8],
                plan.layer_weight_stride_bytes[8],
            ))
        );
        assert_eq!(
            plan.final_norm_weight_range(),
            Some((0, plan.final_norm_weight_bytes.unwrap()))
        );
        assert_eq!(
            plan.lm_head_weight_range(),
            Some((0, plan.lm_head_weight_bytes.unwrap()))
        );
    }
}

#[test]
fn f32_plan_offsets_map_to_exact_storage_offsets() {
    for f32_offset in [0_u64, 1024 * 4, 2048 * 1024 * 4, 17 * 3072 * 1024 * 4] {
        assert_eq!(
            DecoderWeightStorage::F32.from_f32_byte_offset(f32_offset),
            Some(f32_offset)
        );
        assert_eq!(
            DecoderWeightStorage::F16.from_f32_byte_offset(f32_offset),
            Some(f32_offset / 2)
        );
    }

    for invalid in [1_u64, 2, 3, 5, u64::MAX] {
        assert_eq!(
            DecoderWeightStorage::F16.from_f32_byte_offset(invalid),
            None,
            "non-F32-aligned planner offset {invalid} was admitted"
        );
    }
}

#[test]
fn storage_arithmetic_fails_closed_on_overflow() {
    assert_eq!(DecoderWeightStorage::F32.storage_bytes(u64::MAX), None);
    assert_eq!(DecoderWeightStorage::F16.storage_bytes(u64::MAX), None);

    let mut descriptor = descriptor(DecoderWeightStorage::F16, false);
    descriptor.layers = u32::MAX;
    descriptor.f32_layer_weight_stride_bytes[8] = u64::MAX - 3;
    let error = descriptor.plan().unwrap_err();
    assert_eq!(error.code(), DecoderWeightStorageErrorCode::Overflow);
}

#[test]
fn weight_storage_validates_exact_finite_little_endian_payloads() {
    let f16 = [0x0000_u16, 0x8000, 0x3c00, 0xbc00, 0x0001, 0x7bff]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(DecoderWeightStorage::F16.validate_finite_bytes(&f16), Ok(6));

    let f32 = [0.0_f32, -0.0, 1.0, -1.0, f32::MIN_POSITIVE]
        .into_iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(DecoderWeightStorage::F32.validate_finite_bytes(&f32), Ok(5));

    assert!(!DecoderWeightStorage::F32.requires_shader_f16());
    assert!(DecoderWeightStorage::F16.requires_shader_f16());
}

#[test]
fn weight_storage_rejects_misalignment_and_nonfinite_values_at_the_exact_index() {
    let misaligned: [(DecoderWeightStorage, Vec<u8>); 2] = [
        (DecoderWeightStorage::F16, vec![0_u8]),
        (DecoderWeightStorage::F32, vec![0_u8; 2]),
    ];
    for (storage, bytes) in misaligned {
        let error = storage.validate_finite_bytes(&bytes).unwrap_err();
        assert_eq!(
            error.code(),
            DecoderWeightStorageErrorCode::ByteLengthNotAligned
        );
        assert_eq!(error.element_index(), None);
    }

    let nonfinite: [(DecoderWeightStorage, Vec<u8>, u64); 2] = [
        (
            DecoderWeightStorage::F16,
            [0x3c00_u16, 0x7c00, 0x0000]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
            1,
        ),
        (
            DecoderWeightStorage::F32,
            [1.0_f32.to_bits(), f32::NAN.to_bits(), 0.0_f32.to_bits()]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
            1,
        ),
    ];
    for (storage, bytes, expected_index) in nonfinite {
        let error = storage.validate_finite_bytes(&bytes).unwrap_err();
        assert_eq!(error.code(), DecoderWeightStorageErrorCode::NonFinite);
        assert_eq!(error.element_index(), Some(expected_index));
    }
}
