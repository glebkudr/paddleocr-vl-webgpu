use proptest::prelude::*;
use pvlc_cpu_ref::{
    CpuRefErrorCode, apply_rope_neox, gelu_pytorch_tanh, gemm_f32, layer_norm_f32, rms_norm_f32,
    silu, softmax_rows_f32, top_k,
};

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:?} expected={expected:?} tolerance={tolerance:?}"
    );
}

#[test]
fn row_major_gemm_handles_asymmetric_shapes_and_zero_inner_dimension() {
    let a = [1.0, 2.0, 3.0, -1.0, 0.5, 4.0];
    let b = [2.0, -1.0, 0.0, 3.0, 1.5, 2.0];
    let output = gemm_f32(&a, 2, 3, &b, 2).unwrap();
    assert_eq!(output, [6.5, 11.0, 4.0, 10.5]);

    assert_eq!(gemm_f32(&[], 3, 0, &[], 4).unwrap(), vec![0.0; 12]);
}

#[test]
fn gemm_accumulates_in_f32_in_a_documented_order() {
    // Left-to-right f32 accumulation is the arithmetic oracle contract. This
    // first fixture differs from f64 accumulation rounded once at the end; the
    // second differs from reverse-order f32 accumulation.
    let b = [1.0_f32, 1.0, 1.0];
    assert_eq!(
        gemm_f32(&[100_000_000.0, 1.0, -100_000_000.0], 1, 3, &b, 1).unwrap(),
        [0.0]
    );
    assert_eq!(
        gemm_f32(&[100_000_000.0, -100_000_000.0, 1.0], 1, 3, &b, 1).unwrap(),
        [1.0]
    );
}

proptest! {
    #[test]
    fn gemm_is_linear_in_the_left_input(
        a in prop::collection::vec(-10.0_f32..10.0, 6),
        c in prop::collection::vec(-10.0_f32..10.0, 6),
        b in prop::collection::vec(-10.0_f32..10.0, 6),
    ) {
        let sum: Vec<_> = a.iter().zip(&c).map(|(x, y)| x + y).collect();
        let lhs = gemm_f32(&sum, 2, 3, &b, 2).unwrap();
        let a_product = gemm_f32(&a, 2, 3, &b, 2).unwrap();
        let c_product = gemm_f32(&c, 2, 3, &b, 2).unwrap();
        for ((lhs, a), c) in lhs.iter().zip(a_product).zip(c_product) {
            prop_assert!((lhs - (a + c)).abs() <= 2.0e-4);
        }
    }
}

fn independent_layer_norm(row: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let mean = row.iter().map(|v| f64::from(*v)).sum::<f64>() / row.len() as f64;
    let variance = row
        .iter()
        .map(|v| {
            let centered = f64::from(*v) - mean;
            centered * centered
        })
        .sum::<f64>()
        / row.len() as f64;
    row.iter()
        .zip(weight)
        .zip(bias)
        .map(|((value, weight), bias)| {
            (((f64::from(*value) - mean) / (variance + f64::from(eps)).sqrt()) * f64::from(*weight)
                + f64::from(*bias)) as f32
        })
        .collect()
}

#[test]
fn layer_norm_matches_an_independent_f64_formula_per_row() {
    let input = [1.0, 2.0, 4.0, 8.0, -3.0, 0.0, 5.0, 7.0];
    let weight = [0.5, 1.0, 1.5, 2.0];
    let bias = [-1.0, 0.0, 0.25, 2.0];
    let actual = layer_norm_f32(&input, 2, 4, &weight, &bias, 1.0e-5).unwrap();
    let expected: Vec<_> = input
        .chunks_exact(4)
        .flat_map(|row| independent_layer_norm(row, &weight, &bias, 1.0e-5))
        .collect();
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert_close(actual, expected, 2.0e-6);
    }
}

#[test]
fn rms_norm_matches_formula_and_preserves_zero_input() {
    let input = [3.0, 4.0, 0.0, 0.0, -2.0, 2.0, -2.0, 2.0];
    let weight = [1.0, 0.5, 2.0, -1.0];
    let actual = rms_norm_f32(&input, 2, 4, &weight, 1.0e-6).unwrap();
    for row in 0..2 {
        let source = &input[row * 4..row * 4 + 4];
        let mean_square = source.iter().map(|x| x * x).sum::<f32>() / 4.0;
        let scale = (mean_square + 1.0e-6).sqrt().recip();
        for column in 0..4 {
            assert_close(
                actual[row * 4 + column],
                source[column] * scale * weight[column],
                1.0e-6,
            );
        }
    }
    assert_eq!(
        rms_norm_f32(&[0.0; 4], 1, 4, &[1.0; 4], 1e-6).unwrap(),
        [0.0; 4]
    );
}

#[test]
fn activation_functions_match_the_model_contract() {
    for value in [-10.0_f32, -1.0, -0.0, 0.5, 3.0, 10.0] {
        let expected_silu = value / (1.0 + (-value).exp());
        assert_close(silu(value), expected_silu, 1.0e-7);

        let expected_gelu = 0.5
            * value
            * (1.0 + (0.797_884_6_f32 * (value + 0.044_715 * value * value * value)).tanh());
        assert_close(gelu_pytorch_tanh(value), expected_gelu, 1.0e-7);
    }
}

#[test]
fn masked_softmax_is_stable_normalized_and_exactly_zero_on_masked_items() {
    let logits = [10_000.0, 9_999.0, -10_000.0, 12.0, 12.0, 7.0];
    let mask = [true, true, false, true, false, true];
    let output = softmax_rows_f32(&logits, 2, 3, Some(&mask)).unwrap();
    assert!(output.iter().all(|value| value.is_finite()));
    assert_eq!(output[2], 0.0);
    assert_eq!(output[4], 0.0);
    assert_close(output[..3].iter().sum(), 1.0, 1.0e-6);
    assert_close(output[3..].iter().sum(), 1.0, 1.0e-6);
    let first_denominator = 1.0 + (-1.0_f32).exp();
    assert_close(output[0], 1.0 / first_denominator, 1.0e-6);
    assert_close(output[1], (-1.0_f32).exp() / first_denominator, 1.0e-6);
    let second_denominator = 1.0 + (-5.0_f32).exp();
    assert_close(output[3], 1.0 / second_denominator, 1.0e-6);
    assert_close(output[5], (-5.0_f32).exp() / second_denominator, 1.0e-6);
}

#[test]
fn unmasked_softmax_matches_independent_formula_and_is_additive_constant_invariant() {
    let logits = [-3.0_f32, 0.5, 8.0, 2.0, -7.0, 2.0];
    let shifted: Vec<_> = logits.iter().map(|value| value + 1_024.0).collect();
    let actual = softmax_rows_f32(&logits, 2, 3, None).unwrap();
    let shifted_actual = softmax_rows_f32(&shifted, 2, 3, None).unwrap();
    for row in 0..2 {
        let source = &logits[row * 3..row * 3 + 3];
        let max = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials: Vec<_> = source.iter().map(|value| (value - max).exp()).collect();
        let denominator: f32 = exponentials.iter().sum();
        for (column, exponential) in exponentials.iter().enumerate() {
            let index = row * 3 + column;
            assert_close(actual[index], *exponential / denominator, 1.0e-7);
            assert_close(actual[index], shifted_actual[index], 1.0e-7);
        }
    }
}

#[test]
fn neox_rope_position_zero_is_identity_and_each_pair_preserves_norm() {
    let original = vec![1.0, 2.0, 3.0, 4.0, 9.0, -2.0, -4.0, 5.0, 7.0, 11.0];
    let mut values = original.clone();
    apply_rope_neox(&mut values, 2, 5, 4, &[0, 17], 500_000.0).unwrap();
    assert_eq!(&values[..5], &original[..5]);
    assert_eq!(values[4], original[4]);
    assert_eq!(values[9], original[9]);

    let angle_pair_0 = 17.0_f32;
    let angle_pair_1 = 17.0_f32 / 500_000.0_f32.sqrt();
    let expected = [
        -2.0 * angle_pair_0.cos() - 5.0 * angle_pair_0.sin(),
        -4.0 * angle_pair_1.cos() - 7.0 * angle_pair_1.sin(),
        5.0 * angle_pair_0.cos() + -2.0 * angle_pair_0.sin(),
        7.0 * angle_pair_1.cos() + -4.0 * angle_pair_1.sin(),
    ];
    for (actual, expected) in values[5..9].iter().copied().zip(expected) {
        assert_close(actual, expected, 1.0e-6);
    }

    for (left, right) in [(5, 7), (6, 8)] {
        let before = original[left] * original[left] + original[right] * original[right];
        let after = values[left] * values[left] + values[right] * values[right];
        assert_close(after, before, 2.0e-5);
    }
}

#[test]
fn top_k_orders_descending_and_breaks_equal_logits_by_smaller_token_id() {
    let entries = top_k(&[0.5, 7.0, 7.0, -2.0, 7.0, 1.0], 4).unwrap();
    assert_eq!(
        entries.iter().map(|entry| entry.index).collect::<Vec<_>>(),
        [1, 2, 4, 5]
    );
    assert_eq!(
        entries.iter().map(|entry| entry.value).collect::<Vec<_>>(),
        [7.0, 7.0, 7.0, 1.0]
    );
    assert!(top_k(&[1.0, 2.0], 0).unwrap().is_empty());
}

#[test]
fn primitive_contracts_reject_bad_dimensions_masks_epsilons_and_nonfinite_values() {
    assert_eq!(
        gemm_f32(&[1.0], 2, 2, &[1.0; 4], 2).unwrap_err().code(),
        CpuRefErrorCode::DimensionMismatch
    );
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            layer_norm_f32(&[1.0; 4], 1, 4, &[1.0; 4], &[0.0; 4], epsilon)
                .unwrap_err()
                .code(),
            CpuRefErrorCode::NonPositiveEpsilon
        );
        assert_eq!(
            rms_norm_f32(&[1.0; 4], 1, 4, &[1.0; 4], epsilon)
                .unwrap_err()
                .code(),
            CpuRefErrorCode::NonPositiveEpsilon
        );
    }
    assert_eq!(
        softmax_rows_f32(&[1.0; 4], 2, 2, Some(&[true; 3]))
            .unwrap_err()
            .code(),
        CpuRefErrorCode::DimensionMismatch
    );
    assert_eq!(
        softmax_rows_f32(&[1.0; 2], 1, 2, Some(&[false, false]))
            .unwrap_err()
            .code(),
        CpuRefErrorCode::AllMasked
    );
    assert_eq!(
        apply_rope_neox(&mut [1.0; 4], 1, 4, 3, &[0], 10_000.0)
            .unwrap_err()
            .code(),
        CpuRefErrorCode::InvalidRotaryDimension
    );
    for base in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            apply_rope_neox(&mut [1.0; 4], 1, 4, 4, &[0], base)
                .unwrap_err()
                .code(),
            CpuRefErrorCode::InvalidRopeBase
        );
    }
    assert_eq!(
        top_k(&[1.0, 2.0], 3).unwrap_err().code(),
        CpuRefErrorCode::InvalidK
    );
    assert_eq!(
        top_k(&[1.0, f32::NAN], 1).unwrap_err().code(),
        CpuRefErrorCode::NonFiniteInput
    );
    for nonfinite in [f32::NEG_INFINITY, f32::INFINITY] {
        assert_eq!(
            top_k(&[1.0, nonfinite], 1).unwrap_err().code(),
            CpuRefErrorCode::NonFiniteInput
        );
    }

    for error in [
        gemm_f32(&[f32::INFINITY], 1, 1, &[1.0], 1).unwrap_err(),
        layer_norm_f32(&[1.0, f32::NAN], 1, 2, &[1.0; 2], &[0.0; 2], 1e-5).unwrap_err(),
        rms_norm_f32(&[1.0, f32::NEG_INFINITY], 1, 2, &[1.0; 2], 1e-5).unwrap_err(),
        softmax_rows_f32(&[1.0, f32::INFINITY], 1, 2, None).unwrap_err(),
        apply_rope_neox(&mut [1.0, f32::NAN], 1, 2, 2, &[1], 10_000.0).unwrap_err(),
    ] {
        assert_eq!(error.code(), CpuRefErrorCode::NonFiniteInput);
    }
}
