use std::panic::{AssertUnwindSafe, catch_unwind};

use super::*;

fn input(
    semantic_readback_bytes: u64,
    canary_readback_bytes: u64,
    workspace_allocation_bytes: u64,
    max_buffer_size: u64,
    max_host_elements: u64,
) -> VisionQkvExecutionAllocationPreflight {
    VisionQkvExecutionAllocationPreflight {
        semantic_readback_bytes,
        canary_readback_bytes,
        workspace_allocation_bytes,
        max_buffer_size,
        max_host_elements,
    }
}

fn assert_validation_error(name: &str, input: VisionQkvExecutionAllocationPreflight) {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        preflight_vision_qkv_execution_allocations(input)
    }));
    let result = outcome.unwrap_or_else(|_| {
        panic!("{name}: allocation/readback preflight panicked instead of returning Validation")
    });
    match result {
        Err(error) => assert_eq!(
            error.code(),
            RuntimeErrorCode::Validation,
            "{name}: unexpected error: {error}"
        ),
        Ok(_) => panic!("{name}: invalid sizes were accepted"),
    }
}

#[test]
fn vision_qkv_allocation_preflight_accepts_exact_adapter_and_host_boundaries() {
    let preflight = preflight_vision_qkv_execution_allocations(input(64, 16, 80, 80, 20))
        .expect("exact adapter and host boundaries must be accepted");
    assert_eq!(preflight.total_readback_bytes, 80);
    assert_eq!(preflight.readback_f32_elements, 20);
    assert_eq!(preflight.workspace_u32_words, 20);
}

#[test]
fn vision_qkv_allocation_preflight_rejects_semantic_plus_canary_tail_over_adapter_limit() {
    assert_validation_error(
        "semantic readback fits but semantic plus canary tail exceeds max_buffer_size",
        input(80, 4, 80, 80, 20),
    );
}

#[test]
fn vision_qkv_allocation_preflight_rejects_workspace_over_adapter_limit() {
    assert_validation_error(
        "guarded workspace exceeds max_buffer_size",
        input(64, 16, 84, 80, 21),
    );
}

#[test]
fn vision_qkv_allocation_preflight_checks_readback_addition_without_panicking() {
    assert_validation_error(
        "semantic plus canary readback addition overflow",
        input(u64::MAX - 3, 4, 4, u64::MAX, u64::MAX / 4),
    );
}

#[test]
fn vision_qkv_allocation_preflight_rejects_non_word_sized_buffers() {
    assert_validation_error(
        "readback bytes are not an exact F32 element count",
        input(64, 1, 64, 80, 20),
    );
    assert_validation_error(
        "workspace bytes are not an exact u32 word count",
        input(64, 16, 79, 80, 20),
    );
}

#[test]
fn vision_qkv_allocation_preflight_enforces_synthetic_32_bit_host_element_limit() {
    let first_unrepresentable_count = u64::from(u32::MAX) + 1;
    let first_unrepresentable_bytes = first_unrepresentable_count * 4;
    let synthetic_32_bit_limit = u64::from(u32::MAX);

    assert_validation_error(
        "readback F32 count exceeds synthetic 32-bit host element limit",
        input(
            first_unrepresentable_bytes,
            0,
            4,
            first_unrepresentable_bytes,
            synthetic_32_bit_limit,
        ),
    );
    assert_validation_error(
        "workspace u32 count exceeds synthetic 32-bit host element limit",
        input(
            4,
            0,
            first_unrepresentable_bytes,
            first_unrepresentable_bytes,
            synthetic_32_bit_limit,
        ),
    );
}
