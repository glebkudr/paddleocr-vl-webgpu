use pvlc_cpu_ref::{CpuRefErrorCode, projector_merge_2x2_f32};

fn encoded_packed_input(grids: &[[usize; 3]], hidden_size: usize) -> Vec<f32> {
    let mut values = Vec::new();
    for (image, &[temporal, height, width]) in grids.iter().enumerate() {
        for time in 0..temporal {
            for y in 0..height {
                for x in 0..width {
                    let position = image * 1_000_000 + time * 100_000 + y * 1_000 + x;
                    values.extend(
                        (0..hidden_size).map(|channel| position as f32 + channel as f32 * 0.25),
                    );
                }
            }
        }
    }
    values
}

#[test]
fn projector_merge_2x2_preserves_official_spatial_temporal_and_channel_order() {
    let grids = [[1, 2, 4], [2, 2, 2]];
    let hidden_size = 2;
    let input = encoded_packed_input(&grids, hidden_size);

    let output = projector_merge_2x2_f32(&input, hidden_size, &grids).unwrap();

    // Pinned remote code:
    // (t h p1 w p2) d -> (t h w) (p1 p2 d), with p1 = p2 = 2.
    let expected_source_positions = [
        0, 1, 1_000, 1_001, // image 0, spatial block (0, 0)
        2, 3, 1_002, 1_003, // image 0, spatial block (0, 1)
        1_000_000, 1_000_001, 1_001_000, 1_001_001, // image 1, time 0
        1_100_000, 1_100_001, 1_101_000, 1_101_001, // image 1, time 1
    ];
    let expected = expected_source_positions
        .into_iter()
        .flat_map(|position| {
            (0..hidden_size).map(move |channel| position as f32 + channel as f32 * 0.25)
        })
        .collect::<Vec<_>>();

    assert_eq!(output, expected);
    assert_eq!(output.len(), 4 * 4 * hidden_size);
}

#[test]
fn projector_merge_2x2_orders_multiple_spatial_blocks_inside_each_time_step() {
    let grids = [[2, 4, 4]];
    let input = encoded_packed_input(&grids, 1);

    let output = projector_merge_2x2_f32(&input, 1, &grids).unwrap();

    // The merged token axes are exactly (t, h, w), never (h, w, t).
    let expected_source_positions = [
        0, 1, 1_000, 1_001, // time 0, block (0, 0)
        2, 3, 1_002, 1_003, // time 0, block (0, 1)
        2_000, 2_001, 3_000, 3_001, // time 0, block (1, 0)
        2_002, 2_003, 3_002, 3_003, // time 0, block (1, 1)
        100_000, 100_001, 101_000, 101_001, // time 1, block (0, 0)
        100_002, 100_003, 101_002, 101_003, // time 1, block (0, 1)
        102_000, 102_001, 103_000, 103_001, // time 1, block (1, 0)
        102_002, 102_003, 103_002, 103_003, // time 1, block (1, 1)
    ]
    .map(|position| position as f32);
    assert_eq!(output, expected_source_positions);
}

#[test]
fn projector_merge_2x2_keeps_packed_images_bidirectionally_isolated() {
    let grids = [[1, 2, 2], [1, 2, 4]];
    let hidden_size = 3;
    let input = encoded_packed_input(&grids, hidden_size);
    let baseline = projector_merge_2x2_f32(&input, hidden_size, &grids).unwrap();
    let first_image_input_elements = 4 * hidden_size;
    let first_image_output_elements = 4 * hidden_size;

    let mut first_poisoned = input.clone();
    for value in &mut first_poisoned[..first_image_input_elements] {
        *value += 17.0;
    }
    let first_output = projector_merge_2x2_f32(&first_poisoned, hidden_size, &grids).unwrap();
    assert_ne!(
        &first_output[..first_image_output_elements],
        &baseline[..first_image_output_elements]
    );
    assert_eq!(
        &first_output[first_image_output_elements..],
        &baseline[first_image_output_elements..]
    );

    let mut second_poisoned = input;
    for value in &mut second_poisoned[first_image_input_elements..] {
        *value -= 23.0;
    }
    let second_output = projector_merge_2x2_f32(&second_poisoned, hidden_size, &grids).unwrap();
    assert_eq!(
        &second_output[..first_image_output_elements],
        &baseline[..first_image_output_elements]
    );
    assert_ne!(
        &second_output[first_image_output_elements..],
        &baseline[first_image_output_elements..]
    );
}

#[test]
fn projector_merge_2x2_rejects_invalid_geometry_lengths_overflow_and_nonfinite_values() {
    let valid_grid = [[1, 2, 2]];
    let valid = encoded_packed_input(&valid_grid, 3);

    assert_eq!(
        projector_merge_2x2_f32(&[], 3, &[]).unwrap_err().code(),
        CpuRefErrorCode::InvalidProjectorGeometry
    );
    for grid in [
        [[0, 2, 2]],
        [[1, 0, 2]],
        [[1, 2, 0]],
        [[1, 3, 2]],
        [[1, 2, 3]],
    ] {
        assert_eq!(
            projector_merge_2x2_f32(&valid, 3, &grid)
                .unwrap_err()
                .code(),
            CpuRefErrorCode::InvalidProjectorGeometry,
            "grid={grid:?}"
        );
    }

    assert_eq!(
        projector_merge_2x2_f32(&[], 0, &valid_grid)
            .unwrap_err()
            .code(),
        CpuRefErrorCode::DimensionMismatch
    );
    for malformed in [&valid[..valid.len() - 1], &valid[..]] {
        let input = if malformed.len() == valid.len() {
            let mut oversized = malformed.to_vec();
            oversized.push(0.0);
            oversized
        } else {
            malformed.to_vec()
        };
        assert_eq!(
            projector_merge_2x2_f32(&input, 3, &valid_grid)
                .unwrap_err()
                .code(),
            CpuRefErrorCode::DimensionMismatch
        );
    }

    let individually_valid_but_collectively_overflowing =
        [[usize::MAX / 4, 2, 2], [usize::MAX / 4, 2, 2]];
    assert_eq!(
        projector_merge_2x2_f32(&[], 1, &individually_valid_but_collectively_overflowing)
            .unwrap_err()
            .code(),
        CpuRefErrorCode::InvalidProjectorGeometry
    );
    assert_eq!(
        projector_merge_2x2_f32(&[], 1, &[[usize::MAX, 2, 2]])
            .unwrap_err()
            .code(),
        CpuRefErrorCode::InvalidProjectorGeometry
    );
    assert_eq!(
        projector_merge_2x2_f32(&[], usize::MAX, &valid_grid)
            .unwrap_err()
            .code(),
        CpuRefErrorCode::DimensionMismatch
    );

    for index in [0, valid.len() / 2, valid.len() - 1] {
        for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut poisoned = valid.clone();
            poisoned[index] = nonfinite;
            assert_eq!(
                projector_merge_2x2_f32(&poisoned, 3, &valid_grid)
                    .unwrap_err()
                    .code(),
                CpuRefErrorCode::NonFiniteInput
            );
        }
    }
}
