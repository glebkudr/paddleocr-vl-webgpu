use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, MultimodalRopePositions, assemble_multimodal_embeddings_f32,
    decode_mrope_position_ids, image_placeholder_count, mrope_position_ids,
};

const IMAGE_TOKEN: u32 = 99;
const VISION_START_TOKEN: u32 = 98;
const MERGE_SIZE: usize = 2;

fn assert_error<T>(result: Result<T, CpuRefError>, expected: CpuRefErrorCode) {
    let error = result.err().expect("contract mutation must fail closed");
    assert_eq!(error.code(), expected, "{error}");
}

fn assert_positions(actual: &MultimodalRopePositions, expected: [&[i64]; 3], expected_delta: i64) {
    for (axis, expected_axis) in expected.into_iter().enumerate() {
        assert_eq!(actual.position_ids[axis], expected_axis, "axis={axis}");
    }
    assert_eq!(actual.rope_delta, expected_delta);
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn changed_rows(left: &[f32], right: &[f32], width: usize) -> Vec<usize> {
    assert_eq!(left.len(), right.len());
    assert!(width > 0 && left.len().is_multiple_of(width));
    left.chunks_exact(width)
        .zip(right.chunks_exact(width))
        .enumerate()
        .filter_map(|(row, (left, right))| (bits(left) != bits(right)).then_some(row))
        .collect()
}

#[test]
fn placeholder_count_sums_temporal_grids_and_rejects_invalid_or_overflowing_geometry() {
    assert_eq!(image_placeholder_count(&[], MERGE_SIZE).unwrap(), 0);
    assert_eq!(
        image_placeholder_count(&[[2, 4, 6], [1, 2, 8]], MERGE_SIZE).unwrap(),
        16
    );

    for (grids, merge_size) in [
        (vec![[0, 2, 2]], MERGE_SIZE),
        (vec![[1, 0, 2]], MERGE_SIZE),
        (vec![[1, 2, 0]], MERGE_SIZE),
        (vec![[1, 3, 2]], MERGE_SIZE),
        (vec![[1, 2, 3]], MERGE_SIZE),
        (vec![[1, 2, 2]], 0),
        (vec![[usize::MAX, usize::MAX, usize::MAX]], 1),
    ] {
        assert_error(
            image_placeholder_count(&grids, merge_size),
            CpuRefErrorCode::InvalidImageGeometry,
        );
    }
}

#[test]
fn text_only_mrope_is_exact_with_no_padding_left_padding_and_right_padding() {
    let plain = mrope_position_ids(
        &[10, 11, 12],
        None,
        &[],
        IMAGE_TOKEN,
        VISION_START_TOKEN,
        MERGE_SIZE,
    )
    .unwrap();
    assert_positions(&plain, [&[0, 1, 2], &[0, 1, 2], &[0, 1, 2]], 0);

    let right_padded = mrope_position_ids(
        &[10, 11, 0, 0],
        Some(&[1, 1, 0, 0]),
        &[],
        IMAGE_TOKEN,
        VISION_START_TOKEN,
        MERGE_SIZE,
    )
    .unwrap();
    assert_positions(
        &right_padded,
        [&[0, 1, 1, 1], &[0, 1, 1, 1], &[0, 1, 1, 1]],
        -2,
    );

    let left_padded = mrope_position_ids(
        &[0, 0, 10, 11],
        Some(&[0, 0, 1, 1]),
        &[],
        IMAGE_TOKEN,
        VISION_START_TOKEN,
        MERGE_SIZE,
    )
    .unwrap();
    assert_positions(
        &left_padded,
        [&[1, 1, 0, 1], &[1, 1, 0, 1], &[1, 1, 0, 1]],
        -2,
    );
}

#[test]
fn one_image_has_exact_temporal_height_width_positions() {
    let input_ids = [
        10,
        VISION_START_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        11,
    ];
    let actual = mrope_position_ids(
        &input_ids,
        None,
        &[[1, 4, 4]],
        IMAGE_TOKEN,
        VISION_START_TOKEN,
        MERGE_SIZE,
    )
    .unwrap();
    assert_positions(
        &actual,
        [
            &[0, 1, 2, 2, 2, 2, 4],
            &[0, 1, 2, 2, 3, 3, 4],
            &[0, 1, 2, 3, 2, 3, 4],
        ],
        -2,
    );
}

#[test]
fn temporal_image_grid_repeats_spatial_axes_with_constant_image_time() {
    let mut input_ids = vec![10, VISION_START_TOKEN];
    input_ids.extend([IMAGE_TOKEN; 12]);
    input_ids.extend([11, 12]);
    assert_eq!(input_ids.len(), 16);

    let actual = mrope_position_ids(
        &input_ids,
        None,
        &[[2, 4, 6]],
        IMAGE_TOKEN,
        VISION_START_TOKEN,
        MERGE_SIZE,
    )
    .unwrap();
    // Pinned upstream image semantics set second_per_grid_t=0, so both temporal image
    // slices share one coordinate. Video timing/second_per_grid_ts is outside this image-only API.
    assert_positions(
        &actual,
        [
            &[0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 6],
            &[0, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3, 5, 6],
            &[0, 1, 2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4, 5, 6],
        ],
        -9,
    );

    let first_decode = decode_mrope_position_ids(input_ids.len(), 1, actual.rope_delta).unwrap();
    assert_eq!(first_decode, [vec![7], vec![7], vec![7]]);
}

#[test]
fn one_image_mrope_preserves_active_positions_under_left_and_right_padding() {
    let left_padded = mrope_position_ids(
        &[
            0,
            0,
            10,
            VISION_START_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            11,
        ],
        Some(&[0, 0, 1, 1, 1, 1, 1, 1, 1]),
        &[[1, 4, 4]],
        IMAGE_TOKEN,
        VISION_START_TOKEN,
        MERGE_SIZE,
    )
    .unwrap();
    assert_positions(
        &left_padded,
        [
            &[1, 1, 0, 1, 2, 2, 2, 2, 4],
            &[1, 1, 0, 1, 2, 2, 3, 3, 4],
            &[1, 1, 0, 1, 2, 3, 2, 3, 4],
        ],
        -4,
    );

    let right_padded = mrope_position_ids(
        &[
            10,
            VISION_START_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            IMAGE_TOKEN,
            11,
            0,
            0,
        ],
        Some(&[1, 1, 1, 1, 1, 1, 1, 0, 0]),
        &[[1, 4, 4]],
        IMAGE_TOKEN,
        VISION_START_TOKEN,
        MERGE_SIZE,
    )
    .unwrap();
    assert_positions(
        &right_padded,
        [
            &[0, 1, 2, 2, 2, 2, 4, 1, 1],
            &[0, 1, 2, 2, 3, 3, 4, 1, 1],
            &[0, 1, 2, 3, 2, 3, 4, 1, 1],
        ],
        -4,
    );
}

#[test]
fn two_images_preserve_grid_order_across_different_aspect_ratios() {
    let input_ids = [
        10,
        VISION_START_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        11,
        12,
        VISION_START_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        13,
    ];
    let actual = mrope_position_ids(
        &input_ids,
        None,
        &[[1, 2, 4], [1, 4, 2]],
        IMAGE_TOKEN,
        VISION_START_TOKEN,
        MERGE_SIZE,
    )
    .unwrap();
    assert_positions(
        &actual,
        [
            &[0, 1, 2, 2, 4, 5, 6, 7, 7, 9],
            &[0, 1, 2, 2, 4, 5, 6, 7, 8, 9],
            &[0, 1, 2, 3, 4, 5, 6, 7, 7, 9],
        ],
        0,
    );
}

#[test]
fn decode_positions_match_full_recompute_for_first_and_incremental_text_tokens() {
    let prefill = vec![
        10,
        VISION_START_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        IMAGE_TOKEN,
        11,
    ];
    let prefill_positions = mrope_position_ids(
        &prefill,
        None,
        &[[1, 4, 4]],
        IMAGE_TOKEN,
        VISION_START_TOKEN,
        MERGE_SIZE,
    )
    .unwrap();
    assert_eq!(prefill_positions.rope_delta, -2);

    let first_decode =
        decode_mrope_position_ids(prefill.len(), 1, prefill_positions.rope_delta).unwrap();
    let mut with_first_decode = prefill.clone();
    with_first_decode.push(12);
    let first_recomputed = mrope_position_ids(
        &with_first_decode,
        None,
        &[[1, 4, 4]],
        IMAGE_TOKEN,
        VISION_START_TOKEN,
        MERGE_SIZE,
    )
    .unwrap();
    for (axis, decoded_axis) in first_decode.iter().enumerate() {
        assert_eq!(decoded_axis, &[5]);
        assert_eq!(
            decoded_axis[0],
            *first_recomputed.position_ids[axis].last().unwrap()
        );
    }

    let incremental =
        decode_mrope_position_ids(with_first_decode.len(), 2, prefill_positions.rope_delta)
            .unwrap();
    let mut with_three_decode_tokens = with_first_decode;
    with_three_decode_tokens.extend([13, 14]);
    let fully_recomputed = mrope_position_ids(
        &with_three_decode_tokens,
        None,
        &[[1, 4, 4]],
        IMAGE_TOKEN,
        VISION_START_TOKEN,
        MERGE_SIZE,
    )
    .unwrap();
    for (axis, incremental_axis) in incremental.iter().enumerate() {
        assert_eq!(incremental_axis, &[6, 7]);
        assert_eq!(
            incremental_axis.as_slice(),
            &fully_recomputed.position_ids[axis][fully_recomputed.position_ids[axis].len() - 2..]
        );
    }
}

#[test]
fn mrope_rejects_invalid_masks_token_topology_and_decode_overflow() {
    assert_error(
        mrope_position_ids(
            &[10, 11, 12],
            Some(&[1, 1]),
            &[],
            IMAGE_TOKEN,
            VISION_START_TOKEN,
            MERGE_SIZE,
        ),
        CpuRefErrorCode::InvalidSequenceBoundaries,
    );
    assert_error(
        mrope_position_ids(
            &[10, 11, 12],
            Some(&[1, 2, 1]),
            &[],
            IMAGE_TOKEN,
            VISION_START_TOKEN,
            MERGE_SIZE,
        ),
        CpuRefErrorCode::InvalidSequenceBoundaries,
    );
    assert_error(
        mrope_position_ids(
            &[10, 11, 12],
            Some(&[0, 0, 0]),
            &[],
            IMAGE_TOKEN,
            VISION_START_TOKEN,
            MERGE_SIZE,
        ),
        CpuRefErrorCode::AllMasked,
    );

    for (input_ids, grids) in [
        (vec![10, VISION_START_TOKEN, IMAGE_TOKEN], vec![[1, 2, 4]]),
        (vec![10, IMAGE_TOKEN, IMAGE_TOKEN, 11], vec![[1, 2, 4]]),
        (vec![10, VISION_START_TOKEN, 11], vec![]),
        (
            vec![10, VISION_START_TOKEN, IMAGE_TOKEN, 11, IMAGE_TOKEN],
            vec![[1, 2, 4]],
        ),
        (
            vec![10, VISION_START_TOKEN, IMAGE_TOKEN, IMAGE_TOKEN],
            vec![[1, 2, 4], [1, 2, 4]],
        ),
    ] {
        assert_error(
            mrope_position_ids(
                &input_ids,
                None,
                &grids,
                IMAGE_TOKEN,
                VISION_START_TOKEN,
                MERGE_SIZE,
            ),
            CpuRefErrorCode::InvalidSequenceBoundaries,
        );
    }

    assert_error(
        decode_mrope_position_ids(0, 0, 0),
        CpuRefErrorCode::InvalidSequenceBoundaries,
    );
    assert_error(
        decode_mrope_position_ids(usize::MAX, 1, 0),
        CpuRefErrorCode::InvalidSequenceBoundaries,
    );
}

#[test]
fn direct_embedding_assembly_is_bit_exact_sequential_masked_scatter() {
    let hidden_size = 3;
    let input_ids = [10, IMAGE_TOKEN, 11, IMAGE_TOKEN, 12];
    let token_embeddings = (0..input_ids.len() * hidden_size)
        .map(|index| f32::from_bits(0x3f00_0000 + index as u32))
        .collect::<Vec<_>>();
    let projected = [101.0, 102.0, 103.0, 201.0, 202.0, 203.0];
    let actual = assemble_multimodal_embeddings_f32(
        &token_embeddings,
        &projected,
        &input_ids,
        hidden_size,
        IMAGE_TOKEN,
    )
    .unwrap();
    let expected = [
        &token_embeddings[0..3],
        &projected[0..3],
        &token_embeddings[6..9],
        &projected[3..6],
        &token_embeddings[12..15],
    ]
    .concat();
    assert_eq!(bits(&actual), bits(&expected));

    for text_row in [0, 2, 4] {
        let range = text_row * hidden_size..(text_row + 1) * hidden_size;
        assert_eq!(bits(&actual[range.clone()]), bits(&token_embeddings[range]));
    }
    assert_eq!(bits(&actual[3..6]), bits(&projected[0..3]));
    assert_eq!(bits(&actual[9..12]), bits(&projected[3..6]));

    let mut poisoned = projected;
    poisoned[3..6].copy_from_slice(&[-901.0, -902.0, -903.0]);
    let poisoned_output = assemble_multimodal_embeddings_f32(
        &token_embeddings,
        &poisoned,
        &input_ids,
        hidden_size,
        IMAGE_TOKEN,
    )
    .unwrap();
    assert_eq!(changed_rows(&actual, &poisoned_output, hidden_size), [3]);

    let mut swapped = projected;
    for channel in 0..hidden_size {
        swapped.swap(channel, hidden_size + channel);
    }
    let swapped_output = assemble_multimodal_embeddings_f32(
        &token_embeddings,
        &swapped,
        &input_ids,
        hidden_size,
        IMAGE_TOKEN,
    )
    .unwrap();
    assert_eq!(changed_rows(&actual, &swapped_output, hidden_size), [1, 3]);

    let text_ids = [10, 11, 12];
    let text_embeddings = token_embeddings[..text_ids.len() * hidden_size].to_vec();
    let text_only = assemble_multimodal_embeddings_f32(
        &text_embeddings,
        &[],
        &text_ids,
        hidden_size,
        IMAGE_TOKEN,
    )
    .unwrap();
    assert_eq!(bits(&text_only), bits(&text_embeddings));
}

#[test]
fn embedding_assembly_rejects_count_shape_nonfinite_and_overflow_mismatches() {
    let input_ids = [10, IMAGE_TOKEN, 11, IMAGE_TOKEN];
    let token_embeddings = vec![1.0; input_ids.len() * 2];
    let projected = vec![2.0; 4];

    for wrong_projected in [vec![2.0; 2], vec![2.0; 6], vec![2.0; 3]] {
        assert_error(
            assemble_multimodal_embeddings_f32(
                &token_embeddings,
                &wrong_projected,
                &input_ids,
                2,
                IMAGE_TOKEN,
            ),
            CpuRefErrorCode::DimensionMismatch,
        );
    }
    assert_error(
        assemble_multimodal_embeddings_f32(
            &token_embeddings[..token_embeddings.len() - 1],
            &projected,
            &input_ids,
            2,
            IMAGE_TOKEN,
        ),
        CpuRefErrorCode::DimensionMismatch,
    );
    assert_error(
        assemble_multimodal_embeddings_f32(&[1.0], &[], &[10, 11], usize::MAX, IMAGE_TOKEN),
        CpuRefErrorCode::DimensionMismatch,
    );
    assert_error(
        assemble_multimodal_embeddings_f32(&[], &[], &[10], 0, IMAGE_TOKEN),
        CpuRefErrorCode::DimensionMismatch,
    );
    assert_error(
        assemble_multimodal_embeddings_f32(&[], &[], &[], 2, IMAGE_TOKEN),
        CpuRefErrorCode::InvalidSequenceBoundaries,
    );

    let mut nonfinite_tokens = token_embeddings.clone();
    nonfinite_tokens[3] = f32::NAN;
    assert_error(
        assemble_multimodal_embeddings_f32(
            &nonfinite_tokens,
            &projected,
            &input_ids,
            2,
            IMAGE_TOKEN,
        ),
        CpuRefErrorCode::NonFiniteInput,
    );
    let mut nonfinite_projected = projected;
    nonfinite_projected[2] = f32::INFINITY;
    assert_error(
        assemble_multimodal_embeddings_f32(
            &token_embeddings,
            &nonfinite_projected,
            &input_ids,
            2,
            IMAGE_TOKEN,
        ),
        CpuRefErrorCode::NonFiniteInput,
    );
}
