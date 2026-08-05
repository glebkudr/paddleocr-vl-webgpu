//! Planner contract for the M7o5 decode GEMV replacement before production
//! exists (docs/m7o5_tiled_gemv_contract.md).
//!
//! The kernel is pinned to eight output rows per 256-thread workgroup,
//! thirty-two lanes per row, four scalar columns per vector load, and the
//! three real decoder input widths. The planner owns dispatch geometry and
//! preserves the accepted `{rows, columns, 0, 0}` uniform words.

use pvlc_runtime_core::{GemvTiledDescriptor, InvocationErrorCode};

const TILE_ROWS: u32 = 8;
const THREADS_PER_ROW: u32 = 32;
const VECTOR_WIDTH: u32 = 4;
const WORKGROUP_THREADS: u32 = 256;
const SHARED_CAPACITY: u32 = 3072;
const WORKGROUP_STORAGE_BYTES: u32 = 13_312;
const ADMITTED_COLUMNS: [u32; 3] = [1024, 2048, 3072];

fn descriptor(rows: u32, columns: u32) -> GemvTiledDescriptor {
    GemvTiledDescriptor { rows, columns }
}

fn assert_descriptor_error(descriptor: GemvTiledDescriptor, expected: InvocationErrorCode) {
    let error = descriptor.plan().unwrap_err();
    assert_eq!(error.code(), expected, "{error}");
}

#[test]
fn plan_pins_the_exact_vec4_tiled_lattice() {
    let plan = descriptor(2048, 1024).plan().expect("plan");

    assert_eq!(plan.tile_rows, TILE_ROWS);
    assert_eq!(plan.threads_per_row, THREADS_PER_ROW);
    assert_eq!(plan.vector_width, VECTOR_WIDTH);
    assert_eq!(plan.shared_capacity, SHARED_CAPACITY);
    assert_eq!(plan.workgroup_storage_bytes, WORKGROUP_STORAGE_BYTES);
    assert_eq!(plan.dispatch, [256, 1, 1]);
    assert_eq!(plan.workgroup_size, [WORKGROUP_THREADS, 1, 1]);
    assert_eq!(plan.uniform_words, [2048, 1024, 0, 0]);
    assert_eq!(plan.output_elements, 2048_usize);
    assert_eq!(plan.output_bytes, 2048_u64 * 4);
    assert!(
        plan.workgroup_storage_bytes <= 16 * 1024,
        "the kernel must fit WebGPU's portable minimum workgroup storage"
    );
}

#[test]
fn every_real_decoder_width_has_the_same_tiling_authority() {
    for (columns, rows, workgroups) in [
        (1024, 256, 32),         // K/V
        (1024, 2048, 256),       // Q
        (1024, 3072, 384),       // gate/up
        (1024, 103_424, 12_928), // LM head
        (2048, 1024, 128),       // O
        (3072, 1024, 128),       // down
    ] {
        let plan = descriptor(rows, columns).plan().expect("decoder plan");
        assert_eq!(plan.dispatch, [workgroups, 1, 1]);
        assert_eq!(plan.workgroup_size, [WORKGROUP_THREADS, 1, 1]);
        assert_eq!(plan.uniform_words, [rows, columns, 0, 0]);
        assert_eq!(plan.output_elements, rows as usize);
        assert_eq!(plan.output_bytes, u64::from(rows) * 4);
        assert_eq!(plan.tile_rows * plan.threads_per_row, WORKGROUP_THREADS);
        assert_eq!(columns % plan.vector_width, 0);
    }
}

#[test]
fn dispatch_is_the_ceil_of_rows_over_eight() {
    for (rows, workgroups) in [
        (1, 1),
        (7, 1),
        (8, 1),
        (9, 2),
        (256, 32),
        (1024, 128),
        (2048, 256),
        (3072, 384),
        (103_424, 12_928),
    ] {
        for columns in ADMITTED_COLUMNS {
            let plan = descriptor(rows, columns).plan().expect("plan");
            assert_eq!(plan.dispatch, [workgroups, 1, 1]);
            assert_eq!(plan.uniform_words, [rows, columns, 0, 0]);
        }
    }
}

#[test]
fn plan_is_deterministic_and_pure() {
    let first = descriptor(2048, 1024).plan().expect("first plan");
    let second = descriptor(2048, 1024).plan().expect("second plan");
    assert_eq!(first, second);
}

#[test]
fn descriptor_rejects_zero_rows_and_non_decoder_widths() {
    assert_descriptor_error(
        descriptor(0, 1024),
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    for columns in [0, 4, 512, 1020, 1028, 4096] {
        assert_descriptor_error(
            descriptor(2048, columns),
            InvocationErrorCode::InvalidDecoderGeometry,
        );
    }
}

#[test]
fn descriptor_enforces_the_webgpu_dispatch_limit() {
    let maximum_rows = TILE_ROWS * 65_535;
    let maximum = descriptor(maximum_rows, 1024)
        .plan()
        .expect("maximum admitted rows");
    assert_eq!(maximum.dispatch, [65_535, 1, 1]);
    assert_eq!(maximum.output_elements, maximum_rows as usize);

    for rows in [maximum_rows + 1, u32::MAX] {
        assert_descriptor_error(
            descriptor(rows, 1024),
            InvocationErrorCode::InvalidDecoderGeometry,
        );
    }
}
