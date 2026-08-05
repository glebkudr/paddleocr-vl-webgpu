use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use pvlc_cpu_ref::{
    CpuRefErrorCode, LayerNormParameters, VisionEncoderStackConfig,
    vision_encoder_stack_identity_rope_f32,
};

fn config() -> VisionEncoderStackConfig {
    VisionEncoderStackConfig {
        tokens: 2,
        hidden_size: 3,
        layers: 4,
        layer_norm_epsilon: 1.0e-5,
    }
}

fn layer_output(layer: usize, input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let scale = 1.0_f32 + (layer + 1) as f32 * 0.125;
            let bias = (index % 3) as f32 * 0.03125 - layer as f32 * 0.015625;
            value * scale + bias
        })
        .collect()
}

fn independent_layer_norm(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());
    for row in 0..rows {
        let values = &input[row * width..(row + 1) * width];
        let mean = values.iter().map(|value| f64::from(*value)).sum::<f64>() / width as f64;
        let variance = values
            .iter()
            .map(|value| (f64::from(*value) - mean).powi(2))
            .sum::<f64>()
            / width as f64;
        let inverse = (variance + f64::from(epsilon)).sqrt().recip();
        for column in 0..width {
            output.push(
                (((f64::from(values[column]) - mean) * inverse * f64::from(weight[column]))
                    + f64::from(bias[column])) as f32,
            );
        }
    }
    output
}

fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        assert!(
            (left - right).abs() <= tolerance,
            "value {index}: {left} != {right}"
        );
    }
}

#[test]
fn stack_chains_every_layer_once_retains_only_selected_depths_and_applies_post_norm() {
    let input = vec![-2.0, 0.5, 3.0, 4.5, -1.25, 0.75];
    let weight = vec![0.75, 1.25, 1.5];
    let bias = vec![-0.125, 0.25, 0.0625];
    let post_norm = LayerNormParameters {
        weight: &weight,
        bias: &bias,
    };
    let calls = Cell::new(0_usize);
    let mut independently_chained = input.clone();
    let mut expected_checkpoints = Vec::new();
    for layer in 0..config().layers {
        independently_chained = layer_output(layer, &independently_chained);
        if [1, 3].contains(&layer) {
            expected_checkpoints.push((layer, independently_chained.clone()));
        }
    }
    let expected_output = independent_layer_norm(
        &independently_chained,
        config().tokens,
        config().hidden_size,
        &weight,
        &bias,
        config().layer_norm_epsilon,
    );

    let trace = vision_encoder_stack_identity_rope_f32(
        &input,
        config(),
        &[1, 3],
        post_norm,
        |layer, current| {
            assert_eq!(layer, calls.get());
            calls.set(calls.get() + 1);
            Ok(layer_output(layer, current))
        },
    )
    .unwrap();

    assert_eq!(calls.get(), 4);
    assert_eq!(trace.executed_layers, 4);
    assert_eq!(trace.checkpoints.len(), 2);
    assert_eq!(trace.retained_checkpoint_elements, 2 * input.len());
    for (actual, (layer, values)) in trace.checkpoints.iter().zip(expected_checkpoints) {
        assert_eq!(actual.layer_index, layer);
        assert_eq!(actual.values, values);
        assert_eq!(trace.checkpoint(layer), Some(actual.values.as_slice()));
    }
    assert_eq!(trace.checkpoint(0), None);
    assert_close(&trace.output, &expected_output, 2.0e-6);
}

#[test]
fn stack_prevalidation_rejects_geometry_input_post_norm_and_checkpoint_drift_before_layer_zero() {
    let input = vec![0.0; 6];
    let weight = vec![1.0; 3];
    let bias = vec![0.0; 3];
    let calls = Cell::new(0_usize);
    let run = |input: &[f32],
               config: VisionEncoderStackConfig,
               checkpoints: &[usize],
               weight: &[f32],
               bias: &[f32]| {
        let result = vision_encoder_stack_identity_rope_f32(
            input,
            config,
            checkpoints,
            LayerNormParameters { weight, bias },
            |_, values| {
                calls.set(calls.get() + 1);
                Ok(values.to_vec())
            },
        );
        assert_eq!(calls.replace(0), 0, "invalid stack executed a layer");
        result.unwrap_err().code()
    };

    for changed in [
        VisionEncoderStackConfig {
            layers: 0,
            ..config()
        },
        VisionEncoderStackConfig {
            tokens: 0,
            ..config()
        },
        VisionEncoderStackConfig {
            hidden_size: 0,
            ..config()
        },
        VisionEncoderStackConfig {
            tokens: usize::MAX,
            hidden_size: 2,
            ..config()
        },
    ] {
        assert_eq!(
            run(&input, changed, &[0], &weight, &bias),
            CpuRefErrorCode::DimensionMismatch
        );
    }
    assert_eq!(
        run(&input[..5], config(), &[0], &weight, &bias),
        CpuRefErrorCode::DimensionMismatch
    );
    let mut nonfinite_input = input.clone();
    nonfinite_input[3] = f32::NAN;
    assert_eq!(
        run(&nonfinite_input, config(), &[0], &weight, &bias),
        CpuRefErrorCode::NonFiniteInput
    );
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(
            run(
                &input,
                VisionEncoderStackConfig {
                    layer_norm_epsilon: epsilon,
                    ..config()
                },
                &[0],
                &weight,
                &bias,
            ),
            CpuRefErrorCode::NonPositiveEpsilon
        );
    }
    assert_eq!(
        run(&input, config(), &[0], &weight[..2], &bias),
        CpuRefErrorCode::DimensionMismatch
    );
    let mut nonfinite_bias = bias.clone();
    nonfinite_bias[1] = f32::INFINITY;
    assert_eq!(
        run(&input, config(), &[0], &weight, &nonfinite_bias),
        CpuRefErrorCode::NonFiniteInput
    );
    for checkpoints in [&[1, 0][..], &[1, 1][..], &[0, 4][..]] {
        assert_eq!(
            run(&input, config(), checkpoints, &weight, &bias),
            CpuRefErrorCode::InvalidCheckpointSelection
        );
    }
}

#[test]
fn stack_rejects_malformed_or_nonfinite_layer_outputs_and_propagates_layer_errors() {
    let input = vec![0.0; 6];
    let weight = vec![1.0; 3];
    let bias = vec![0.0; 3];
    for expected_code in [
        CpuRefErrorCode::DimensionMismatch,
        CpuRefErrorCode::NonFiniteInput,
    ] {
        let calls = Cell::new(0_usize);
        let error = vision_encoder_stack_identity_rope_f32(
            &input,
            config(),
            &[],
            LayerNormParameters {
                weight: &weight,
                bias: &bias,
            },
            |layer, values| {
                calls.set(calls.get() + 1);
                let mut output = values.to_vec();
                if layer == 1 {
                    if expected_code == CpuRefErrorCode::DimensionMismatch {
                        output.pop();
                    } else {
                        output[2] = f32::NEG_INFINITY;
                    }
                }
                Ok(output)
            },
        )
        .unwrap_err();
        assert_eq!(error.code(), expected_code);
        assert_eq!(calls.get(), 2);
    }

    let seen_layers = RefCell::new(Vec::new());
    let error = vision_encoder_stack_identity_rope_f32(
        &input,
        config(),
        &[],
        LayerNormParameters {
            weight: &weight,
            bias: &bias,
        },
        |layer, values| {
            seen_layers.borrow_mut().push(layer);
            if layer == 2 {
                pvlc_cpu_ref::layer_norm_f32(&[1.0], 1, 1, &[1.0], &[0.0], 0.0)
            } else {
                Ok(values.to_vec())
            }
        },
    )
    .unwrap_err();
    assert_eq!(error.code(), CpuRefErrorCode::NonPositiveEpsilon);
    assert_eq!(*seen_layers.borrow(), [0, 1, 2]);
}

#[derive(Default)]
struct LiveOutputs {
    live: Cell<usize>,
    peak: Cell<usize>,
}

struct TrackedLayerOutput {
    values: Vec<f32>,
    live: Rc<LiveOutputs>,
}

impl TrackedLayerOutput {
    fn new(values: Vec<f32>, live: Rc<LiveOutputs>) -> Self {
        let next = live.live.get() + 1;
        live.live.set(next);
        live.peak.set(live.peak.get().max(next));
        Self { values, live }
    }
}

impl AsRef<[f32]> for TrackedLayerOutput {
    fn as_ref(&self) -> &[f32] {
        &self.values
    }
}

impl Drop for TrackedLayerOutput {
    fn drop(&mut self) {
        self.live.live.set(self.live.live.get() - 1);
    }
}

#[test]
fn long_stack_checkpoint_retention_is_bounded_by_the_explicit_selection() {
    let input = vec![0.25; 6];
    let weight = vec![1.0; 3];
    let bias = vec![0.0; 3];
    let live = Rc::new(LiveOutputs::default());
    let trace = vision_encoder_stack_identity_rope_f32(
        &input,
        VisionEncoderStackConfig {
            layers: 128,
            ..config()
        },
        &[0, 64, 127],
        LayerNormParameters {
            weight: &weight,
            bias: &bias,
        },
        |_, values| {
            Ok(TrackedLayerOutput::new(
                values.iter().map(|value| value + 0.001).collect(),
                Rc::clone(&live),
            ))
        },
    )
    .unwrap();
    assert_eq!(trace.executed_layers, 128);
    assert_eq!(trace.checkpoints.len(), 3);
    assert_eq!(trace.retained_checkpoint_elements, 3 * input.len());
    assert_eq!(trace.output.len(), input.len());
    assert_eq!(live.live.get(), 0, "provider output survived stack return");
    assert!(
        live.peak.get() <= 2,
        "streaming stack retained {} provider outputs concurrently",
        live.peak.get()
    );
}
