use pvlc_cpu_ref::{
    CpuRefErrorCode, LayerNormParameters, LinearParameters, ProjectorParameters, gelu_erf_f32,
    gelu_pytorch_tanh, projector_f32,
};

const HIDDEN_SIZE: usize = 3;
const MERGED_WIDTH: usize = HIDDEN_SIZE * 4;
const OUTPUT_WIDTH: usize = 4;
const GRIDS: [[usize; 3]; 2] = [[1, 2, 2], [1, 2, 4]];
const EPSILON: f32 = 1.0e-5;

#[derive(Clone)]
struct OwnedParameters {
    pre_norm_weight: Vec<f32>,
    pre_norm_bias: Vec<f32>,
    linear1_weight: Vec<f32>,
    linear1_bias: Vec<f32>,
    linear2_weight: Vec<f32>,
    linear2_bias: Vec<f32>,
}

impl OwnedParameters {
    fn fixture() -> Self {
        Self {
            pre_norm_weight: vec![0.75, -1.25, 0.5],
            pre_norm_bias: vec![0.2, -0.4, 0.1],
            linear1_weight: (0..MERGED_WIDTH * MERGED_WIDTH)
                .map(|index| ((index * 13 + 7) % 29) as f32 / 64.0 - 0.22)
                .collect(),
            linear1_bias: (0..MERGED_WIDTH)
                .map(|index| index as f32 / 50.0 - 0.1)
                .collect(),
            linear2_weight: (0..OUTPUT_WIDTH * MERGED_WIDTH)
                .map(|index| ((index * 11 + 3) % 23) as f32 / 48.0 - 0.2)
                .collect(),
            linear2_bias: vec![-0.2, 0.05, 0.3, -0.1],
        }
    }

    fn borrowed(&self) -> ProjectorParameters<'_> {
        ProjectorParameters {
            pre_norm: LayerNormParameters {
                weight: &self.pre_norm_weight,
                bias: &self.pre_norm_bias,
            },
            linear1: LinearParameters {
                weight: &self.linear1_weight,
                bias: &self.linear1_bias,
            },
            linear2: LinearParameters {
                weight: &self.linear2_weight,
                bias: &self.linear2_bias,
            },
        }
    }
}

fn fixture_input() -> Vec<f32> {
    (0..12)
        .flat_map(|row| {
            let row = row as f32;
            [
                row * 0.31 - 0.7,
                (row * 0.47 + 0.2).sin() * 1.3,
                row * row * 0.017 - 0.4,
            ]
        })
        .collect()
}

fn parameter_operand(parameters: &mut OwnedParameters, operand: usize) -> &mut Vec<f32> {
    match operand {
        0 => &mut parameters.pre_norm_weight,
        1 => &mut parameters.pre_norm_bias,
        2 => &mut parameters.linear1_weight,
        3 => &mut parameters.linear1_bias,
        4 => &mut parameters.linear2_weight,
        5 => &mut parameters.linear2_bias,
        _ => panic!("invalid parameter operand {operand}"),
    }
}

fn erf_series(value: f64) -> f64 {
    let mut sum = value;
    let mut power = value;
    let mut factorial = 1.0_f64;
    for n in 1..40 {
        power *= value * value;
        factorial *= n as f64;
        let term = power / (factorial * (2 * n + 1) as f64);
        if n % 2 == 0 {
            sum += term;
        } else {
            sum -= term;
        }
    }
    2.0 / std::f64::consts::PI.sqrt() * sum
}

fn independent_layer_norm(input: &[f32], weight: &[f32], bias: &[f32], width: usize) -> Vec<f32> {
    input
        .chunks_exact(width)
        .flat_map(|row| {
            let mean = row.iter().map(|value| f64::from(*value)).sum::<f64>() / width as f64;
            let variance = row
                .iter()
                .map(|value| (f64::from(*value) - mean).powi(2))
                .sum::<f64>()
                / width as f64;
            row.iter()
                .enumerate()
                .map(|(column, value)| {
                    (((f64::from(*value) - mean) / (variance + f64::from(EPSILON)).sqrt())
                        * f64::from(weight[column])
                        + f64::from(bias[column])) as f32
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn independent_merge(input: &[f32]) -> Vec<f32> {
    let mut output = Vec::new();
    let mut image_offset = 0;
    for &[temporal, height, width] in &GRIDS {
        for time in 0..temporal {
            for merged_y in 0..height / 2 {
                for merged_x in 0..width / 2 {
                    for patch_y in 0..2 {
                        for patch_x in 0..2 {
                            let token = image_offset
                                + time * height * width
                                + (merged_y * 2 + patch_y) * width
                                + merged_x * 2
                                + patch_x;
                            output.extend_from_slice(
                                &input[token * HIDDEN_SIZE..(token + 1) * HIDDEN_SIZE],
                            );
                        }
                    }
                }
            }
        }
        image_offset += temporal * height * width;
    }
    output
}

fn independent_linear(input: &[f32], input_width: usize, weight: &[f32], bias: &[f32]) -> Vec<f32> {
    input
        .chunks_exact(input_width)
        .flat_map(|row| {
            bias.iter()
                .enumerate()
                .map(|(output, bias)| {
                    let dot = row
                        .iter()
                        .enumerate()
                        .map(|(input, value)| {
                            f64::from(*value) * f64::from(weight[output * input_width + input])
                        })
                        .sum::<f64>();
                    (dot + f64::from(*bias)) as f32
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "index={index} actual={actual:?} expected={expected:?} tolerance={tolerance:?}"
        );
    }
}

#[test]
fn projector_matches_an_independent_f64_stage_oracle_and_exact_remote_order() {
    let input = fixture_input();
    let parameters = OwnedParameters::fixture();
    let trace = projector_f32(&input, HIDDEN_SIZE, &GRIDS, parameters.borrowed(), EPSILON).unwrap();

    let expected_pre_norm = independent_layer_norm(
        &input,
        &parameters.pre_norm_weight,
        &parameters.pre_norm_bias,
        HIDDEN_SIZE,
    );
    let expected_merged = independent_merge(&expected_pre_norm);
    let expected_linear1 = independent_linear(
        &expected_merged,
        MERGED_WIDTH,
        &parameters.linear1_weight,
        &parameters.linear1_bias,
    );
    let expected_activation = expected_linear1
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            (0.5 * value * (1.0 + erf_series(value / std::f64::consts::SQRT_2))) as f32
        })
        .collect::<Vec<_>>();
    let expected_output = independent_linear(
        &expected_activation,
        MERGED_WIDTH,
        &parameters.linear2_weight,
        &parameters.linear2_bias,
    );

    assert_eq!(trace.pre_norm.len(), 12 * HIDDEN_SIZE);
    assert_eq!(trace.merged.len(), 3 * MERGED_WIDTH);
    assert_eq!(trace.linear1.len(), 3 * MERGED_WIDTH);
    assert_eq!(trace.activation.len(), 3 * MERGED_WIDTH);
    assert_eq!(trace.output.len(), 3 * OUTPUT_WIDTH);
    assert_close(&trace.pre_norm, &expected_pre_norm, 3.0e-6);
    assert_close(&trace.merged, &expected_merged, 3.0e-6);
    assert_close(&trace.linear1, &expected_linear1, 3.0e-6);
    assert_close(&trace.activation, &expected_activation, 4.0e-6);
    assert_close(&trace.output, &expected_output, 5.0e-6);

    let skipped_norm = independent_merge(&input);
    assert_ne!(trace.merged, skipped_norm);
    let wrong_patch_order = trace
        .pre_norm
        .chunks_exact(MERGED_WIDTH)
        .flat_map(|block| {
            [0, 2, 1, 3]
                .into_iter()
                .flat_map(|patch| {
                    block[patch * HIDDEN_SIZE..(patch + 1) * HIDDEN_SIZE]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_ne!(trace.merged, wrong_patch_order);
    assert_ne!(trace.activation, trace.linear1);
}

#[test]
fn erf_gelu_is_pinned_independently_and_distinguishes_transformers_exact_mode() {
    for value in [-3.0_f32, -1.0, -0.0, 0.5, 1.0, 3.0] {
        let expected = {
            let value = f64::from(value);
            (0.5 * value * (1.0 + erf_series(value / std::f64::consts::SQRT_2))) as f32
        };
        assert!((gelu_erf_f32(value) - expected).abs() <= 2.0e-7);
    }
    assert!((gelu_erf_f32(1.0) - gelu_pytorch_tanh(1.0)).abs() > 1.0e-4);
}

#[test]
fn projector_keeps_every_stage_of_packed_images_bidirectionally_isolated() {
    let parameters = OwnedParameters::fixture();
    let input = fixture_input();
    let baseline =
        projector_f32(&input, HIDDEN_SIZE, &GRIDS, parameters.borrowed(), EPSILON).unwrap();

    let mut first_poisoned = input.clone();
    first_poisoned[1] += 2.0;
    let first = projector_f32(
        &first_poisoned,
        HIDDEN_SIZE,
        &GRIDS,
        parameters.borrowed(),
        EPSILON,
    )
    .unwrap();
    for (baseline_stage, poisoned_stage, first_image_elements) in [
        (&baseline.pre_norm, &first.pre_norm, 4 * HIDDEN_SIZE),
        (&baseline.merged, &first.merged, MERGED_WIDTH),
        (&baseline.linear1, &first.linear1, MERGED_WIDTH),
        (&baseline.activation, &first.activation, MERGED_WIDTH),
        (&baseline.output, &first.output, OUTPUT_WIDTH),
    ] {
        assert_ne!(
            &baseline_stage[..first_image_elements],
            &poisoned_stage[..first_image_elements]
        );
        assert_eq!(
            &baseline_stage[first_image_elements..],
            &poisoned_stage[first_image_elements..]
        );
    }

    let mut second_poisoned = input;
    second_poisoned[4 * HIDDEN_SIZE + 2] -= 3.0;
    let second = projector_f32(
        &second_poisoned,
        HIDDEN_SIZE,
        &GRIDS,
        parameters.borrowed(),
        EPSILON,
    )
    .unwrap();
    for (baseline_stage, poisoned_stage, first_image_elements) in [
        (&baseline.pre_norm, &second.pre_norm, 4 * HIDDEN_SIZE),
        (&baseline.merged, &second.merged, MERGED_WIDTH),
        (&baseline.linear1, &second.linear1, MERGED_WIDTH),
        (&baseline.activation, &second.activation, MERGED_WIDTH),
        (&baseline.output, &second.output, OUTPUT_WIDTH),
    ] {
        assert_eq!(
            &baseline_stage[..first_image_elements],
            &poisoned_stage[..first_image_elements]
        );
        assert_ne!(
            &baseline_stage[first_image_elements..],
            &poisoned_stage[first_image_elements..]
        );
    }
}

#[test]
fn projector_rejects_all_malformed_and_nonfinite_operands_before_execution() {
    let input = fixture_input();
    let valid = OwnedParameters::fixture();
    let invoke = |input: &[f32], parameters: &OwnedParameters, epsilon| {
        projector_f32(input, HIDDEN_SIZE, &GRIDS, parameters.borrowed(), epsilon)
    };

    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            invoke(&input, &valid, epsilon).unwrap_err().code(),
            CpuRefErrorCode::NonPositiveEpsilon
        );
    }
    for invalid_input in [&input[..input.len() - 1], &input[..4]] {
        assert_eq!(
            invoke(invalid_input, &valid, EPSILON).unwrap_err().code(),
            CpuRefErrorCode::DimensionMismatch
        );
    }
    let mut oversized_input = input.clone();
    oversized_input.push(0.0);
    assert_eq!(
        invoke(&oversized_input, &valid, EPSILON)
            .unwrap_err()
            .code(),
        CpuRefErrorCode::DimensionMismatch
    );

    for operand in 0..6 {
        for oversized in [false, true] {
            let mut parameters = valid.clone();
            let values = parameter_operand(&mut parameters, operand);
            if oversized {
                values.push(0.0);
            } else {
                values.pop();
            }
            assert_eq!(
                invoke(&input, &parameters, EPSILON).unwrap_err().code(),
                CpuRefErrorCode::DimensionMismatch,
                "parameter operand={operand} oversized={oversized}"
            );
        }
    }

    for operand in 0..7 {
        let operand_len = match operand {
            0 => input.len(),
            parameter => {
                let mut lengths = valid.clone();
                parameter_operand(&mut lengths, parameter - 1).len()
            }
        };
        for index in [0, operand_len / 2, operand_len - 1] {
            for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut poisoned_input = input.clone();
                let mut poisoned = valid.clone();
                let values = match operand {
                    0 => &mut poisoned_input,
                    parameter => parameter_operand(&mut poisoned, parameter - 1),
                };
                values[index] = nonfinite;
                assert_eq!(
                    invoke(&poisoned_input, &poisoned, EPSILON)
                        .unwrap_err()
                        .code(),
                    CpuRefErrorCode::NonFiniteInput,
                    "operand={operand} index={index} nonfinite={nonfinite:?}"
                );
            }
        }
    }

    for overflowing_grids in [
        vec![[usize::MAX, 2, 2]],
        vec![[usize::MAX / 4, 2, 2], [usize::MAX / 4, 2, 2]],
    ] {
        assert_eq!(
            projector_f32(
                &[],
                HIDDEN_SIZE,
                &overflowing_grids,
                valid.borrowed(),
                EPSILON,
            )
            .unwrap_err()
            .code(),
            CpuRefErrorCode::InvalidProjectorGeometry,
            "overflowing_grids={overflowing_grids:?}"
        );
    }

    assert_eq!(
        projector_f32(&input, HIDDEN_SIZE, &[[1, 3, 2]], valid.borrowed(), EPSILON,)
            .unwrap_err()
            .code(),
        CpuRefErrorCode::InvalidProjectorGeometry
    );
    let mut malformed = valid.clone();
    malformed.linear1_weight.pop();
    let mut also_nonfinite = input;
    also_nonfinite[0] = f32::NAN;
    assert_eq!(
        invoke(&also_nonfinite, &malformed, EPSILON)
            .unwrap_err()
            .code(),
        CpuRefErrorCode::DimensionMismatch,
        "shape validation must precede arithmetic/finiteness checks"
    );
}
