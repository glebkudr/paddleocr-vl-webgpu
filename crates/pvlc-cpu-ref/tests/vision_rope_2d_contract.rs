use pvlc_cpu_ref::{CpuRefErrorCode, vision_rope_2d_f32};

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2.0e-6,
            "value {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn vision_rope_matches_transformers_height_then_width_half_rotation() {
    // The production head is 72-wide. Eight dimensions make the same mapping
    // visible: H frequencies occupy pairs 0..2, W frequencies pairs 2..4,
    // and rotate_half couples each pair with the corresponding second half.
    let query = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let key = [
        -1.0, -2.0, -3.0, -4.0, 0.5, 1.5, 2.5, 3.5, //
        -9.0, -10.0, -11.0, -12.0, 4.5, 5.5, 6.5, 7.5,
    ];
    let (rotated_query, rotated_key) =
        vision_rope_2d_f32(&query, &key, 1, 2, 8, &[[1, 0]], 10_000.0).unwrap();

    let angle_h0 = 1.0_f32;
    let angle_h1 = 0.01_f32;
    let expected_query_head_0 = [
        1.0 * angle_h0.cos() - 5.0 * angle_h0.sin(),
        2.0 * angle_h1.cos() - 6.0 * angle_h1.sin(),
        3.0,
        4.0,
        5.0 * angle_h0.cos() + 1.0 * angle_h0.sin(),
        6.0 * angle_h1.cos() + 2.0 * angle_h1.sin(),
        7.0,
        8.0,
    ];
    let expected_query_head_1 = [
        9.0 * angle_h0.cos() - 13.0 * angle_h0.sin(),
        10.0 * angle_h1.cos() - 14.0 * angle_h1.sin(),
        11.0,
        12.0,
        13.0 * angle_h0.cos() + 9.0 * angle_h0.sin(),
        14.0 * angle_h1.cos() + 10.0 * angle_h1.sin(),
        15.0,
        16.0,
    ];
    let expected_key_head_0 = [
        -angle_h0.cos() - 0.5 * angle_h0.sin(),
        -2.0 * angle_h1.cos() - 1.5 * angle_h1.sin(),
        -3.0,
        -4.0,
        0.5 * angle_h0.cos() - angle_h0.sin(),
        1.5 * angle_h1.cos() - 2.0 * angle_h1.sin(),
        2.5,
        3.5,
    ];
    let expected_key_head_1 = [
        -9.0 * angle_h0.cos() - 4.5 * angle_h0.sin(),
        -10.0 * angle_h1.cos() - 5.5 * angle_h1.sin(),
        -11.0,
        -12.0,
        4.5 * angle_h0.cos() - 9.0 * angle_h0.sin(),
        5.5 * angle_h1.cos() - 10.0 * angle_h1.sin(),
        6.5,
        7.5,
    ];
    assert_close(&rotated_query[..8], &expected_query_head_0);
    assert_close(&rotated_query[8..], &expected_query_head_1);
    assert_close(&rotated_key[..8], &expected_key_head_0);
    assert_close(&rotated_key[8..], &expected_key_head_1);
}

#[test]
fn vision_rope_uses_width_positions_independently_and_zero_position_is_identity() {
    let query = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let key = query.map(|value| -value);
    let (rotated_query, rotated_key) =
        vision_rope_2d_f32(&query, &key, 2, 1, 8, &[[0, 0], [0, 1]], 10_000.0)
            .unwrap();

    let angle_w0 = 1.0_f32;
    let angle_w1 = 0.01_f32;
    let expected_query = [
        9.0,
        10.0,
        11.0 * angle_w0.cos() - 15.0 * angle_w0.sin(),
        12.0 * angle_w1.cos() - 16.0 * angle_w1.sin(),
        13.0,
        14.0,
        15.0 * angle_w0.cos() + 11.0 * angle_w0.sin(),
        16.0 * angle_w1.cos() + 12.0 * angle_w1.sin(),
    ];
    let expected_key = expected_query.map(|value| -value);
    assert_eq!(&rotated_query[..8], &query[..8]);
    assert_eq!(&rotated_key[..8], &key[..8]);
    assert_close(&rotated_query[8..], &expected_query);
    assert_close(&rotated_key[8..], &expected_key);
}

#[test]
fn vision_rope_rejects_malformed_geometry_base_positions_and_nonfinite_inputs() {
    let values = [0.0; 8];
    for (tokens, heads, head_dim, positions, expected) in [
        (0, 1, 8, &[[0, 0]][..], CpuRefErrorCode::DimensionMismatch),
        (1, 0, 8, &[[0, 0]][..], CpuRefErrorCode::DimensionMismatch),
        (
            1,
            1,
            6,
            &[[0, 0]][..],
            CpuRefErrorCode::InvalidRotaryDimension,
        ),
        (1, 1, 8, &[][..], CpuRefErrorCode::DimensionMismatch),
    ] {
        assert_eq!(
            vision_rope_2d_f32(
                &values,
                &values,
                tokens,
                heads,
                head_dim,
                positions,
                10_000.0,
            )
            .unwrap_err()
            .code(),
            expected
        );
    }

    for base in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            vision_rope_2d_f32(&values, &values, 1, 1, 8, &[[0, 0]], base)
                .unwrap_err()
                .code(),
            CpuRefErrorCode::InvalidRopeBase
        );
    }

    let mut poisoned = values;
    poisoned[3] = f32::NAN;
    assert_eq!(
        vision_rope_2d_f32(&poisoned, &values, 1, 1, 8, &[[0, 0]], 10_000.0)
            .unwrap_err()
            .code(),
        CpuRefErrorCode::NonFiniteInput
    );
    assert_eq!(
        vision_rope_2d_f32(&values, &poisoned, 1, 1, 8, &[[0, 0]], 10_000.0)
            .unwrap_err()
            .code(),
        CpuRefErrorCode::NonFiniteInput
    );
    assert_eq!(
        vision_rope_2d_f32(&values[..7], &values, 1, 1, 8, &[[0, 0]], 10_000.0)
            .unwrap_err()
            .code(),
        CpuRefErrorCode::DimensionMismatch
    );
    assert_eq!(
        vision_rope_2d_f32(&values, &values[..7], 1, 1, 8, &[[0, 0]], 10_000.0)
            .unwrap_err()
            .code(),
        CpuRefErrorCode::DimensionMismatch
    );
}
