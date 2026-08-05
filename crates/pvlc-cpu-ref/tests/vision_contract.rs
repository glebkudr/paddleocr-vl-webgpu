use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use pvlc_cpu_ref::{
    CpuRefErrorCode, KvBlockOrder, LayerNormParameters, LinearParameters, VisionEncoderLayerConfig,
    VisionEncoderLayerParameters, VisionPreprocessConfig, add_interpolated_position_embedding_f32,
    add_vectors_f32, canonical_rgb8_to_patches, linear_f32, materialized_segmented_attention_f32,
    patch_projection_f32, smart_resize_paddleocr_vl, streaming_segmented_attention_f32,
    vision_encoder_layer_identity_rope_f32,
};

const PATCH_SIZE: usize = 14;
const MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const PREPROCESSOR_CONFIG: &[u8] = include_bytes!(concat!(
    "../../../models/snapshots/",
    "66317acc4c9fc17bd154591ce650735cd2855f3e/",
    "preprocessor_config.json"
));
const MODEL_LOCK: &str = include_str!("../../../models/paddleocr-vl-1.6.lock");

#[derive(Clone, Copy, Debug, Default)]
struct AllocationState {
    active: bool,
    current_bytes: usize,
    peak_bytes: usize,
}

thread_local! {
    static ALLOCATION_STATE: Cell<AllocationState> = const { Cell::new(AllocationState {
        active: false,
        current_bytes: 0,
        peak_bytes: 0,
    }) };
}

struct TrackingAllocator;

#[global_allocator]
static GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn record_allocation(bytes: usize) {
    let _ = ALLOCATION_STATE.try_with(|state| {
        let mut value = state.get();
        if value.active {
            value.current_bytes = value.current_bytes.saturating_add(bytes);
            value.peak_bytes = value.peak_bytes.max(value.current_bytes);
            state.set(value);
        }
    });
}

fn record_deallocation(bytes: usize) {
    let _ = ALLOCATION_STATE.try_with(|state| {
        let mut value = state.get();
        if value.active {
            value.current_bytes = value.current_bytes.saturating_sub(bytes);
            state.set(value);
        }
    });
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record_deallocation(layout.size());
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            record_deallocation(layout.size());
            record_allocation(new_size);
        }
        new_pointer
    }
}

fn measure_peak_allocation<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATION_STATE.with(|state| {
        assert!(!state.get().active, "allocation measurements may not nest");
        state.set(AllocationState {
            active: true,
            current_bytes: 0,
            peak_bytes: 0,
        });
    });
    let output = operation();
    let peak = ALLOCATION_STATE.with(|state| {
        let peak = state.get().peak_bytes;
        state.set(AllocationState::default());
        peak
    });
    (output, peak)
}

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:?} expected={expected:?} tolerance={tolerance:?}"
    );
}

fn model_config() -> VisionPreprocessConfig {
    VisionPreprocessConfig {
        patch_size: PATCH_SIZE,
        merge_size: 2,
        min_pixels: 112_896,
        max_pixels: 1_003_520,
    }
}

#[test]
fn smart_resize_matches_the_pinned_processor_at_geometry_boundaries() {
    let config = model_config();
    for (height, width, expected) in [
        (1, 1, (336, 336)),
        (27, 27, (336, 336)),
        (28, 28, (336, 336)),
        (29, 29, (336, 336)),
        (300, 800, (308, 812)),
        (800, 300, (812, 308)),
        // Python uses ties-to-even here: round(350 / 28) == 12.
        (350, 350, (336, 336)),
        (3_000, 3_000, (980, 980)),
    ] {
        assert_eq!(
            smart_resize_paddleocr_vl(height, width, config).unwrap(),
            expected,
            "source geometry {height}x{width}"
        );
    }

    let (height, width) = smart_resize_paddleocr_vl(300, 800, config).unwrap();
    assert_eq!([1, height / PATCH_SIZE, width / PATCH_SIZE], [1, 22, 58]);
    assert_eq!((height / PATCH_SIZE) * (width / PATCH_SIZE), 1_276);
}

#[test]
fn smart_resize_rejects_invalid_geometry_and_impossible_configuration() {
    let config = model_config();
    for (height, width) in [(0, 28), (28, 0), (28, 28 * 201)] {
        assert_eq!(
            smart_resize_paddleocr_vl(height, width, config)
                .unwrap_err()
                .code(),
            CpuRefErrorCode::InvalidImageGeometry
        );
    }

    for invalid in [
        VisionPreprocessConfig {
            patch_size: 0,
            ..config
        },
        VisionPreprocessConfig {
            merge_size: 0,
            ..config
        },
        VisionPreprocessConfig {
            min_pixels: 1_000,
            max_pixels: 999,
            ..config
        },
    ] {
        assert_eq!(
            smart_resize_paddleocr_vl(28, 28, invalid)
                .unwrap_err()
                .code(),
            CpuRefErrorCode::InvalidPreprocessConfig
        );
    }
}

#[test]
fn canonical_rgb8_preprocessing_preserves_patch_channel_and_pixel_order() {
    let height = 28;
    let width = 56;
    let mut rgb = vec![0_u8; height * width * 3];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                rgb[(y * width + x) * 3 + channel] = ((y * 31 + x * 17 + channel * 53) % 256) as u8;
            }
        }
    }
    let config = VisionPreprocessConfig {
        min_pixels: height * width,
        max_pixels: height * width,
        ..model_config()
    };
    let output = canonical_rgb8_to_patches(&rgb, height, width, config).unwrap();

    assert_eq!(
        (output.resized_height, output.resized_width),
        (height, width)
    );
    assert_eq!(output.image_grid_thw, [1, 2, 4]);
    assert_eq!(output.values.len(), 8 * 3 * PATCH_SIZE * PATCH_SIZE);

    for patch_y in 0..2 {
        for patch_x in 0..4 {
            let patch = patch_y * 4 + patch_x;
            for channel in 0..3 {
                for local_y in 0..PATCH_SIZE {
                    for local_x in 0..PATCH_SIZE {
                        let source_y = patch_y * PATCH_SIZE + local_y;
                        let source_x = patch_x * PATCH_SIZE + local_x;
                        let source = rgb[(source_y * width + source_x) * 3 + channel];
                        let output_index =
                            (((patch * 3 + channel) * PATCH_SIZE + local_y) * PATCH_SIZE) + local_x;
                        let expected = 2.0 * f32::from(source) / 255.0 - 1.0;
                        assert_eq!(output.values[output_index], expected);
                    }
                }
            }
        }
    }
}

#[test]
fn canonical_rgb8_bicubic_resize_matches_the_official_python_processor() {
    let height = 29;
    let width = 31;
    let mut rgb = vec![0_u8; height * width * 3];
    for y in 0..height {
        for x in 0..width {
            let pixel = (y * width + x) * 3;
            rgb[pixel] = ((x * 17 + y * 3) % 256) as u8;
            rgb[pixel + 1] = ((x * 5 + y * 29 + 7) % 256) as u8;
            rgb[pixel + 2] = ((x * 11 + y * 13 + 19) % 256) as u8;
        }
    }
    let config = VisionPreprocessConfig {
        min_pixels: 28 * 28,
        max_pixels: 28 * 28,
        ..model_config()
    };
    let output = canonical_rgb8_to_patches(&rgb, height, width, config).unwrap();

    assert_eq!((output.resized_height, output.resized_width), (28, 28));
    assert_eq!(output.image_grid_thw, [1, 2, 2]);
    let mut raw_bits = Vec::with_capacity(output.values.len() * 4);
    for value in &output.values {
        raw_bits.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    assert_eq!(
        blake3::hash(&raw_bits).to_hex().as_str(),
        "d7319132258c207228f7d1c6544e6b5d08bf285ab4deaabdd32335c4ca541994"
    );
    for (index, expected_bits) in [
        (0, 0xbf7d_fdfe),
        (13, 0x3f79_f9fa),
        (196, 0xbf6f_eff0),
        (588, 0xbd88_8888),
        (1_500, 0x3f0b_8b8c),
        (2_351, 0x3f11_9192),
    ] {
        assert_eq!(
            output.values[index].to_bits(),
            expected_bits,
            "index {index}"
        );
    }
}

#[test]
fn production_config_bicubic_upscale_matches_the_pinned_official_processor() {
    // Regeneration oracle:
    //   local snapshot at MODEL_REVISION
    //   PaddleOCRVLImageProcessor(...preprocessor_config.json...).preprocess(return_tensors="np")
    //   source is the deterministic formula below in row-major RGB8 order
    // Hashes bind both the source and every output f32 bit to that invocation.
    assert!(MODEL_LOCK.contains(&format!("revision = {MODEL_REVISION:?}")));
    assert!(MODEL_LOCK.contains(
        "\"preprocessor_config.json\" = { blake3 = \"06a17b64a56e696acc447ca8002286dde7cc2900f57378e478178c39927cf70e\""
    ));
    assert_eq!(
        blake3::hash(PREPROCESSOR_CONFIG).to_hex().as_str(),
        "06a17b64a56e696acc447ca8002286dde7cc2900f57378e478178c39927cf70e"
    );
    let height = 29;
    let width = 31;
    let mut rgb = vec![0_u8; height * width * 3];
    for y in 0..height {
        for x in 0..width {
            let pixel = (y * width + x) * 3;
            rgb[pixel] = ((x * 17 + y * 3) % 256) as u8;
            rgb[pixel + 1] = ((x * 5 + y * 29 + 7) % 256) as u8;
            rgb[pixel + 2] = ((x * 11 + y * 13 + 19) % 256) as u8;
        }
    }
    assert_eq!(
        blake3::hash(&rgb).to_hex().as_str(),
        "cb6acd67be313b63eaf51641695a424a52a0dc63287dc195792ce447a179837d"
    );

    let output = canonical_rgb8_to_patches(&rgb, height, width, model_config()).unwrap();
    assert_eq!((output.resized_height, output.resized_width), (336, 364));
    assert_eq!(output.image_grid_thw, [1, 24, 26]);
    assert_eq!(output.values.len(), 366_912);
    let mut raw_bits = Vec::with_capacity(output.values.len() * 4);
    for value in &output.values {
        raw_bits.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    assert_eq!(
        blake3::hash(&raw_bits).to_hex().as_str(),
        "54cc35abbf418aee73fb775ce963f3f3edee9da33fbcba137b583735706a8a4e"
    );
    for (index, expected_bits) in [
        (0, 0xbf80_0000),
        (13, 0xbf6b_ebec),
        (588, 0xbf67_e7e8),
        (10_000, 0xbea2_a2a2),
        (100_000, 0xbf0b_8b8c),
        (300_000, 0x3de8_e8f0),
        (366_911, 0x3f19_999a),
    ] {
        assert_eq!(
            output.values[index].to_bits(),
            expected_bits,
            "index {index}"
        );
    }
}

#[test]
fn canonical_rgb8_constant_image_remains_constant_across_mandatory_upscale() {
    let output = canonical_rgb8_to_patches(&[0, 128, 255], 1, 1, model_config()).unwrap();
    assert_eq!((output.resized_height, output.resized_width), (336, 336));
    assert_eq!(output.image_grid_thw, [1, 24, 24]);
    for (index, value) in output.values.iter().copied().enumerate() {
        let channel = (index / (PATCH_SIZE * PATCH_SIZE)) % 3;
        let expected = [-1.0, 2.0 * 128.0 / 255.0 - 1.0, 1.0][channel];
        assert_eq!(value, expected);
    }
}

#[test]
fn canonical_rgb8_preprocessing_rejects_wrong_buffer_length() {
    assert_eq!(
        canonical_rgb8_to_patches(&[0_u8; 11], 2, 2, model_config())
            .unwrap_err()
            .code(),
        CpuRefErrorCode::DimensionMismatch
    );
}

#[test]
fn patch_projection_preserves_conv2d_output_input_and_patch_order() {
    let patches = [1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 2.0, 1.0];
    let weights = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 0.0, 2.0];
    let bias = [0.25, -0.5];

    let output = patch_projection_f32(&patches, 2, 1, 2, &weights, &bias, 2).unwrap();

    assert_eq!(output, [30.25, 7.5, 9.25, 2.5]);
}

#[test]
fn patch_projection_rejects_zero_overflow_malformed_and_nonfinite_inputs() {
    let patches = [1.0, 2.0, 3.0, 4.0];
    let weights = [1.0, 2.0, 3.0, 4.0];
    let bias = [0.0];

    for dimensions in [
        (0, 1, 2, 1),
        (1, 0, 2, 1),
        (1, 1, 0, 1),
        (1, 1, 2, 0),
        (1, usize::MAX, 2, 1),
    ] {
        let (patch_count, channels, patch_size, output_width) = dimensions;
        assert_eq!(
            patch_projection_f32(
                &patches,
                patch_count,
                channels,
                patch_size,
                &weights,
                &bias,
                output_width,
            )
            .unwrap_err()
            .code(),
            CpuRefErrorCode::DimensionMismatch
        );
    }

    for (bad_patches, bad_weights, bad_bias) in [
        (&patches[..3], &weights[..], &bias[..]),
        (&patches[..], &weights[..3], &bias[..]),
        (&patches[..], &weights[..], &[][..]),
    ] {
        assert_eq!(
            patch_projection_f32(bad_patches, 1, 1, 2, bad_weights, bad_bias, 1)
                .unwrap_err()
                .code(),
            CpuRefErrorCode::DimensionMismatch
        );
    }

    for (bad_patches, bad_weights, bad_bias) in [
        (&[1.0, f32::NAN, 3.0, 4.0][..], &weights[..], &bias[..]),
        (&patches[..], &[1.0, 2.0, f32::INFINITY, 4.0][..], &bias[..]),
        (&patches[..], &weights[..], &[f32::NEG_INFINITY][..]),
    ] {
        assert_eq!(
            patch_projection_f32(bad_patches, 1, 1, 2, bad_weights, bad_bias, 1)
                .unwrap_err()
                .code(),
            CpuRefErrorCode::NonFiniteInput
        );
    }
}

#[test]
fn dense_linear_reuses_checkpoint_output_major_layout_for_every_row() {
    let input = [1.0, 2.0, 3.0, -1.0, 0.0, 2.0];
    let weight = [1.0, 2.0, 3.0, -1.0, 0.5, 2.0];
    let bias = [0.25, -0.5];

    let output = linear_f32(&input, 2, 3, &weight, &bias, 2).unwrap();

    assert_eq!(output, [14.25, 5.5, 5.25, 4.5]);
}

#[test]
fn dense_linear_and_residual_add_reject_malformed_or_nonfinite_operands() {
    for dimensions in [(0, 3, 2), (2, 0, 2), (2, 3, 0), (usize::MAX, 2, 2)] {
        assert_eq!(
            linear_f32(&[], dimensions.0, dimensions.1, &[], &[], dimensions.2)
                .unwrap_err()
                .code(),
            CpuRefErrorCode::DimensionMismatch
        );
    }
    for (input, weight, bias) in [
        (&[1.0, 2.0][..], &[1.0, 2.0, 3.0][..], &[0.0][..]),
        (&[1.0, 2.0, 3.0][..], &[1.0, 2.0][..], &[0.0][..]),
        (&[1.0, 2.0, 3.0][..], &[1.0, 2.0, 3.0][..], &[][..]),
    ] {
        assert_eq!(
            linear_f32(input, 1, 3, weight, bias, 1).unwrap_err().code(),
            CpuRefErrorCode::DimensionMismatch
        );
    }
    for (input, weight, bias) in [
        (&[1.0, f32::NAN, 3.0][..], &[1.0, 2.0, 3.0][..], &[0.0][..]),
        (
            &[1.0, 2.0, 3.0][..],
            &[1.0, f32::INFINITY, 3.0][..],
            &[0.0][..],
        ),
        (
            &[1.0, 2.0, 3.0][..],
            &[1.0, 2.0, 3.0][..],
            &[f32::NEG_INFINITY][..],
        ),
    ] {
        assert_eq!(
            linear_f32(input, 1, 3, weight, bias, 1).unwrap_err().code(),
            CpuRefErrorCode::NonFiniteInput
        );
    }

    assert_eq!(
        add_vectors_f32(&[1.0, -2.0, 3.0], &[0.5, 2.0, -1.0]).unwrap(),
        [1.5, 0.0, 2.0]
    );
    for (left, right, expected) in [
        (&[][..], &[][..], CpuRefErrorCode::DimensionMismatch),
        (
            &[1.0][..],
            &[1.0, 2.0][..],
            CpuRefErrorCode::DimensionMismatch,
        ),
        (&[f32::NAN][..], &[1.0][..], CpuRefErrorCode::NonFiniteInput),
        (
            &[1.0][..],
            &[f32::INFINITY][..],
            CpuRefErrorCode::NonFiniteInput,
        ),
    ] {
        assert_eq!(add_vectors_f32(left, right).unwrap_err().code(), expected);
    }
}

#[test]
fn position_embedding_uses_builtin_transformers_endpoint_bilinear_geometry() {
    let patch_embeddings = [0.0; 16];
    let source_positions = [0.0, 10.0, 20.0, 30.0];

    let output = add_interpolated_position_embedding_f32(
        &patch_embeddings,
        1,
        &source_positions,
        2,
        2,
        &[[1, 4, 4]],
    )
    .unwrap();

    let expected = [
        0.0,
        10.0 / 3.0,
        20.0 / 3.0,
        10.0,
        20.0 / 3.0,
        10.0,
        40.0 / 3.0,
        50.0 / 3.0,
        40.0 / 3.0,
        50.0 / 3.0,
        20.0,
        70.0 / 3.0,
        20.0,
        70.0 / 3.0,
        80.0 / 3.0,
        30.0,
    ];
    for (actual, expected) in output.iter().zip(expected) {
        assert!((actual - expected).abs() <= 2.0e-6, "{actual} != {expected}");
    }
}

#[test]
fn position_embedding_repeats_temporal_grids_and_keeps_packed_images_in_order() {
    let patch_embeddings = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
    let source_positions = [0.0, 10.0, 20.0, 30.0];

    let output = add_interpolated_position_embedding_f32(
        &patch_embeddings,
        1,
        &source_positions,
        2,
        2,
        &[[2, 1, 2], [1, 2, 1]],
    )
    .unwrap();

    assert_eq!(output, [100.0, 111.0, 102.0, 113.0, 104.0, 125.0]);
}

#[test]
fn position_embedding_rejects_invalid_grids_shapes_overflow_and_nonfinite_values() {
    let embeddings = [1.0, 2.0, 3.0, 4.0];
    let positions = [0.0, 10.0, 20.0, 30.0];
    for (hidden, source_height, source_width, grids) in [
        (0, 2, 2, &[[1, 2, 2]][..]),
        (1, 0, 2, &[[1, 2, 2]][..]),
        (1, 2, 0, &[[1, 2, 2]][..]),
        (1, 2, 2, &[][..]),
        (1, 2, 2, &[[0, 2, 2]][..]),
        (1, 2, 2, &[[1, 0, 2]][..]),
        (1, 2, 2, &[[1, 2, 0]][..]),
        (1, 2, 2, &[[usize::MAX, 2, 2]][..]),
    ] {
        assert_eq!(
            add_interpolated_position_embedding_f32(
                &embeddings,
                hidden,
                &positions,
                source_height,
                source_width,
                grids,
            )
            .unwrap_err()
            .code(),
            CpuRefErrorCode::DimensionMismatch
        );
    }
    assert_eq!(
        add_interpolated_position_embedding_f32(
            &embeddings[..3],
            1,
            &positions,
            2,
            2,
            &[[1, 2, 2]],
        )
        .unwrap_err()
        .code(),
        CpuRefErrorCode::DimensionMismatch
    );
    assert_eq!(
        add_interpolated_position_embedding_f32(
            &embeddings,
            1,
            &positions[..3],
            2,
            2,
            &[[1, 2, 2]],
        )
        .unwrap_err()
        .code(),
        CpuRefErrorCode::DimensionMismatch
    );
    for (bad_embeddings, bad_positions) in [
        (&[1.0, 2.0, f32::NAN, 4.0][..], &positions[..]),
        (&embeddings[..], &[0.0, 10.0, f32::INFINITY, 30.0][..]),
    ] {
        assert_eq!(
            add_interpolated_position_embedding_f32(
                bad_embeddings,
                1,
                bad_positions,
                2,
                2,
                &[[1, 2, 2]],
            )
            .unwrap_err()
            .code(),
            CpuRefErrorCode::NonFiniteInput
        );
    }
}

fn attention_fixture(
    tokens: usize,
    heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = tokens * heads * head_dim;
    let query = (0..len)
        .map(|index| ((index * 17 + 3) as f32 * 0.071).sin())
        .collect();
    let key = (0..len)
        .map(|index| ((index * 29 + 11) as f32 * 0.037).cos())
        .collect();
    let value = (0..len)
        .map(|index| ((index * 13 + 5) as f32 * 0.053).sin() * 2.0)
        .collect();
    (query, key, value)
}

#[test]
fn materialized_attention_matches_a_hand_calculated_two_token_example() {
    let tokens = 2;
    let heads = 1;
    let head_dim = 2;
    let query = [1.0, 0.0, 0.0, 1.0];
    let key = [1.0, 0.0, 0.0, 1.0];
    let value = [10.0, 0.0, 0.0, 20.0];
    let actual = materialized_segmented_attention_f32(
        &query,
        &key,
        &value,
        tokens,
        heads,
        head_dim,
        &[0, tokens],
    )
    .unwrap();
    let raised = (1.0_f64 / 2.0_f64.sqrt()).exp();
    let high = (raised / (raised + 1.0)) as f32;
    let low = (1.0 / (raised + 1.0)) as f32;
    let expected = [10.0 * high, 20.0 * low, 10.0 * low, 20.0 * high];
    assert_eq!(actual.len(), tokens * heads * head_dim);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_close(actual, expected, 2.0e-6);
    }
}

#[test]
fn streaming_attention_matches_materialized_reference_across_required_sequences_and_tails() {
    for tokens in [8, 16, 31, 64, 127, 256] {
        let heads = 2;
        let head_dim = 4;
        let (query, key, value) = attention_fixture(tokens, heads, head_dim);
        let cu_seqlens = [0, tokens];
        let reference = materialized_segmented_attention_f32(
            &query,
            &key,
            &value,
            tokens,
            heads,
            head_dim,
            &cu_seqlens,
        )
        .unwrap();
        let expected_len = tokens * heads * head_dim;
        assert_eq!(reference.len(), expected_len, "materialized S={tokens}");

        for key_tile in [1, 7, 32, 128] {
            let streamed = streaming_segmented_attention_f32(
                &query,
                &key,
                &value,
                tokens,
                heads,
                head_dim,
                &cu_seqlens,
                key_tile,
                KvBlockOrder::Forward,
            )
            .unwrap();
            assert_eq!(streamed.len(), expected_len, "S={tokens} tile={key_tile}");
            for (actual, expected) in streamed.iter().zip(&reference) {
                assert_close(*actual, *expected, 2.0e-5);
            }
        }
    }
}

#[test]
fn streaming_attention_matches_materialized_reference_for_packed_uneven_segments() {
    let tokens = 67;
    let heads = 2;
    let head_dim = 4;
    let boundaries = [0, 3, 31, 67];
    let (query, key, value) = attention_fixture(tokens, heads, head_dim);
    let expected = materialized_segmented_attention_f32(
        &query,
        &key,
        &value,
        tokens,
        heads,
        head_dim,
        &boundaries,
    )
    .unwrap();
    let actual = streaming_segmented_attention_f32(
        &query,
        &key,
        &value,
        tokens,
        heads,
        head_dim,
        &boundaries,
        16,
        KvBlockOrder::Forward,
    )
    .unwrap();
    let expected_len = tokens * heads * head_dim;
    assert_eq!(expected.len(), expected_len);
    assert_eq!(actual.len(), expected_len);
    for (actual, expected) in actual.iter().zip(expected) {
        assert_close(*actual, expected, 2.0e-5);
    }
}

#[test]
fn streaming_attention_is_stable_across_key_block_order() {
    let (query, key, value) = attention_fixture(127, 2, 8);
    let forward = streaming_segmented_attention_f32(
        &query,
        &key,
        &value,
        127,
        2,
        8,
        &[0, 31, 127],
        17,
        KvBlockOrder::Forward,
    )
    .unwrap();
    let reverse = streaming_segmented_attention_f32(
        &query,
        &key,
        &value,
        127,
        2,
        8,
        &[0, 31, 127],
        17,
        KvBlockOrder::Reverse,
    )
    .unwrap();
    let expected_len = 127 * 2 * 8;
    assert_eq!(forward.len(), expected_len);
    assert_eq!(reverse.len(), expected_len);
    for (left, right) in forward.iter().zip(reverse) {
        assert_close(*left, right, 3.0e-5);
    }
}

#[test]
fn segmented_attention_never_mixes_any_pair_of_images() {
    let tokens = 17;
    let heads = 2;
    let head_dim = 4;
    let boundaries = [0, 3, 9, 17];
    let (query, key, value) = attention_fixture(tokens, heads, head_dim);
    let baseline = streaming_segmented_attention_f32(
        &query,
        &key,
        &value,
        tokens,
        heads,
        head_dim,
        &boundaries,
        3,
        KvBlockOrder::Forward,
    )
    .unwrap();
    let expected_len = tokens * heads * head_dim;
    assert_eq!(baseline.len(), expected_len);

    for poisoned_segment in 0..boundaries.len() - 1 {
        let mut poisoned_key = key.clone();
        let mut poisoned_value = value.clone();
        let scalar_start = boundaries[poisoned_segment] * heads * head_dim;
        let scalar_end = boundaries[poisoned_segment + 1] * heads * head_dim;
        for index in scalar_start..scalar_end {
            poisoned_key[index] = poisoned_key[index] * -31.0 + 7.0;
            poisoned_value[index] = poisoned_value[index] * 47.0 - 11.0;
        }
        let poisoned = streaming_segmented_attention_f32(
            &query,
            &poisoned_key,
            &poisoned_value,
            tokens,
            heads,
            head_dim,
            &boundaries,
            3,
            KvBlockOrder::Forward,
        )
        .unwrap();
        assert_eq!(poisoned.len(), expected_len);
        for segment in 0..boundaries.len() - 1 {
            let start = boundaries[segment] * heads * head_dim;
            let end = boundaries[segment + 1] * heads * head_dim;
            if segment == poisoned_segment {
                assert_ne!(&baseline[start..end], &poisoned[start..end]);
            } else {
                assert_eq!(&baseline[start..end], &poisoned[start..end]);
            }
        }
    }
}

#[test]
fn real_resolution_streaming_execution_stays_below_an_independent_allocation_budget() {
    let (query, key, value) = attention_fixture(5_120, 1, 2);
    // The CPU oracle is deliberately synchronous. If this implementation ever
    // gains worker threads, this thread-local allocation probe must be replaced
    // by a process-wide measurement before the memory proof remains valid.
    let (output, peak_bytes) = measure_peak_allocation(|| {
        streaming_segmented_attention_f32(
            &query,
            &key,
            &value,
            5_120,
            1,
            2,
            &[0, 5_120],
            128,
            KvBlockOrder::Forward,
        )
        .unwrap()
    });
    assert_eq!(output.len(), 5_120 * 2);
    assert!(output.iter().all(|value| value.is_finite()));
    let output_bytes = output.len() * size_of::<f32>();
    let allocation_budget = output_bytes + 512 * 1_024;
    assert!(
        peak_bytes <= allocation_budget,
        "streaming attention used {peak_bytes} bytes; budget is {allocation_budget}, while an SxS f32 matrix alone is {} bytes",
        5_120_usize * 5_120 * size_of::<f32>()
    );
}

#[test]
fn composed_layer_never_materializes_a_global_attention_score_matrix() {
    const TOKENS: usize = 2_048;
    const HIDDEN_SIZE: usize = 2;
    const INTERMEDIATE_SIZE: usize = 2;
    const IDENTITY: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    const ZERO_BIAS: [f32; 2] = [0.0, 0.0];
    const NORM_WEIGHT: [f32; 2] = [1.0, 1.0];
    const NORM_BIAS: [f32; 2] = [0.0, 0.0];

    let input = (0..TOKENS * HIDDEN_SIZE)
        .map(|index| (index % 17) as f32 * 0.031_25 - 0.25)
        .collect::<Vec<_>>();
    let linear = LinearParameters {
        weight: &IDENTITY,
        bias: &ZERO_BIAS,
    };
    let norm = LayerNormParameters {
        weight: &NORM_WEIGHT,
        bias: &NORM_BIAS,
    };
    let parameters = VisionEncoderLayerParameters {
        norm1: norm,
        query: linear,
        key: linear,
        value: linear,
        attention_output: linear,
        norm2: norm,
        mlp_fc1: linear,
        mlp_fc2: linear,
    };
    let config = VisionEncoderLayerConfig {
        tokens: TOKENS,
        hidden_size: HIDDEN_SIZE,
        attention_heads: 1,
        head_dim: HIDDEN_SIZE,
        intermediate_size: INTERMEDIATE_SIZE,
        layer_norm_epsilon: 1.0e-5,
        attention_key_tile: 128,
        attention_order: KvBlockOrder::Forward,
    };
    let boundaries = [0, TOKENS];

    let (trace, peak_bytes) = measure_peak_allocation(|| {
        vision_encoder_layer_identity_rope_f32(&input, config, &boundaries, parameters).unwrap()
    });
    assert_eq!(trace.output.len(), TOKENS * HIDDEN_SIZE);
    assert!(trace.output.iter().all(|value: &f32| value.is_finite()));

    let retained_trace_bytes =
        (10 * TOKENS * HIDDEN_SIZE + 2 * TOKENS * INTERMEDIATE_SIZE) * size_of::<f32>();
    let allocation_budget = retained_trace_bytes + 512 * 1_024;
    assert!(
        peak_bytes <= allocation_budget,
        "composed layer used {peak_bytes} bytes; budget is {allocation_budget}, while one SxS f32 score matrix alone is {} bytes",
        TOKENS * TOKENS * size_of::<f32>()
    );
}

#[test]
fn segmented_attention_rejects_malformed_shapes_boundaries_tiles_and_nonfinite_inputs() {
    let (query, key, value) = attention_fixture(4, 1, 2);
    for boundaries in [&[1, 4][..], &[0, 3][..], &[0, 2, 2, 4][..]] {
        assert_eq!(
            streaming_segmented_attention_f32(
                &query,
                &key,
                &value,
                4,
                1,
                2,
                boundaries,
                2,
                KvBlockOrder::Forward,
            )
            .unwrap_err()
            .code(),
            CpuRefErrorCode::InvalidSequenceBoundaries
        );
    }
    assert_eq!(
        streaming_segmented_attention_f32(
            &query,
            &key,
            &value,
            4,
            1,
            2,
            &[0, 4],
            0,
            KvBlockOrder::Forward,
        )
        .unwrap_err()
        .code(),
        CpuRefErrorCode::InvalidTileSize
    );
    assert_eq!(
        materialized_segmented_attention_f32(&query[..7], &key, &value, 4, 1, 2, &[0, 4])
            .unwrap_err()
            .code(),
        CpuRefErrorCode::DimensionMismatch
    );
    let mut nonfinite = query.clone();
    nonfinite[3] = f32::NAN;
    assert_eq!(
        streaming_segmented_attention_f32(
            &nonfinite,
            &key,
            &value,
            4,
            1,
            2,
            &[0, 4],
            2,
            KvBlockOrder::Forward,
        )
        .unwrap_err()
        .code(),
        CpuRefErrorCode::NonFiniteInput
    );
    for tensor in [&key, &value] {
        let mut nonfinite = tensor.clone();
        nonfinite[5] = f32::INFINITY;
        let (bad_key, bad_value) = if std::ptr::eq(tensor, &key) {
            (&nonfinite[..], &value[..])
        } else {
            (&key[..], &nonfinite[..])
        };
        assert_eq!(
            streaming_segmented_attention_f32(
                &query,
                bad_key,
                bad_value,
                4,
                1,
                2,
                &[0, 4],
                2,
                KvBlockOrder::Forward,
            )
            .unwrap_err()
            .code(),
            CpuRefErrorCode::NonFiniteInput
        );
    }
}
