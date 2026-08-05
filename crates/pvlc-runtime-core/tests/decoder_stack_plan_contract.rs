use pvlc_runtime_core::{
    DecoderAttentionBlockDescriptor, DecoderLayerDescriptor, DecoderLayerStep,
    DecoderStackDescriptor, DecoderStackStep, InvocationErrorCode,
};

const LAYERS: u32 = 18;
const HIDDEN_SIZE: u32 = 1024;
const INTERMEDIATE_SIZE: u32 = 3072;
const QUERY_HEADS: u32 = 16;
const KEY_VALUE_HEADS: u32 = 2;
const HEAD_DIM: u32 = 128;
const CACHE_CAPACITY: u32 = 9;
const RMS_NORM_EPSILON: f32 = 1.0e-5;
const MROPE_SECTIONS: [usize; 3] = [16, 24, 24];

const QUERY_WIDTH: usize = (QUERY_HEADS * HEAD_DIM) as usize;
const KEY_VALUE_WIDTH: usize = (KEY_VALUE_HEADS * HEAD_DIM) as usize;
const ROPE_ELEMENTS: usize = 3 * CACHE_CAPACITY as usize * HEAD_DIM as usize;

const NORM_ELEMENTS: usize = HIDDEN_SIZE as usize;
const Q_WEIGHT_ELEMENTS: usize = QUERY_WIDTH * HIDDEN_SIZE as usize;
const K_WEIGHT_ELEMENTS: usize = KEY_VALUE_WIDTH * HIDDEN_SIZE as usize;
const O_WEIGHT_ELEMENTS: usize = HIDDEN_SIZE as usize * QUERY_WIDTH;
const GATE_WEIGHT_ELEMENTS: usize = INTERMEDIATE_SIZE as usize * HIDDEN_SIZE as usize;
const DOWN_WEIGHT_ELEMENTS: usize = HIDDEN_SIZE as usize * INTERMEDIATE_SIZE as usize;

#[derive(Clone)]
struct Fixture {
    norm1_weight: Vec<f32>,
    q_weight: Vec<f32>,
    k_weight: Vec<f32>,
    v_weight: Vec<f32>,
    o_weight: Vec<f32>,
    mrope_cos: Vec<f32>,
    mrope_sin: Vec<f32>,
    norm2_weight: Vec<f32>,
    gate_weight: Vec<f32>,
    up_weight: Vec<f32>,
    down_weight: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            norm1_weight: finite_values(LAYERS as usize * NORM_ELEMENTS, 0.011),
            q_weight: finite_values(LAYERS as usize * Q_WEIGHT_ELEMENTS, 0.013),
            k_weight: finite_values(LAYERS as usize * K_WEIGHT_ELEMENTS, 0.017),
            v_weight: finite_values(LAYERS as usize * K_WEIGHT_ELEMENTS, 0.019),
            o_weight: finite_values(LAYERS as usize * O_WEIGHT_ELEMENTS, 0.023),
            mrope_cos: finite_values(ROPE_ELEMENTS, 0.029),
            mrope_sin: finite_values(ROPE_ELEMENTS, 0.031),
            norm2_weight: finite_values(LAYERS as usize * NORM_ELEMENTS, 0.037),
            gate_weight: finite_values(LAYERS as usize * GATE_WEIGHT_ELEMENTS, 0.041),
            up_weight: finite_values(LAYERS as usize * GATE_WEIGHT_ELEMENTS, 0.043),
            down_weight: finite_values(LAYERS as usize * DOWN_WEIGHT_ELEMENTS, 0.047),
        }
    }

    fn descriptor(&self) -> DecoderStackDescriptor<'_> {
        DecoderStackDescriptor {
            layers: LAYERS,
            hidden_size: HIDDEN_SIZE,
            intermediate_size: INTERMEDIATE_SIZE,
            query_heads: QUERY_HEADS,
            key_value_heads: KEY_VALUE_HEADS,
            head_dim: HEAD_DIM,
            rms_norm_epsilon: RMS_NORM_EPSILON,
            norm1_weight: &self.norm1_weight,
            q_weight: &self.q_weight,
            k_weight: &self.k_weight,
            v_weight: &self.v_weight,
            o_weight: &self.o_weight,
            mrope_cos: &self.mrope_cos,
            mrope_sin: &self.mrope_sin,
            norm2_weight: &self.norm2_weight,
            gate_weight: &self.gate_weight,
            up_weight: &self.up_weight,
            down_weight: &self.down_weight,
            cache_capacity: CACHE_CAPACITY,
        }
    }
}

fn finite_values(length: usize, scale: f32) -> Vec<f32> {
    (0..length)
        .map(|index| ((index * 31 + 7) as f32 * scale).sin())
        .collect()
}

fn assert_descriptor_error(descriptor: DecoderStackDescriptor<'_>, expected: InvocationErrorCode) {
    let error = descriptor.plan().unwrap_err();
    assert_eq!(error.code(), expected, "{error}");
}

fn hidden_row_fixture() -> Vec<f32> {
    finite_values(HIDDEN_SIZE as usize, 0.053)
}

#[test]
fn descriptor_freezes_pinned_stack_geometry_strides_and_layer_plan() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();

    assert_eq!(plan.layers, LAYERS);
    assert_eq!(plan.layer_plan.attention_block.hidden_size, HIDDEN_SIZE);
    assert_eq!(
        plan.layer_plan.attention_block.cache_capacity,
        CACHE_CAPACITY
    );
    assert_eq!(
        plan.layer_plan.attention_block.mrope_sections,
        MROPE_SECTIONS
    );
    assert_eq!(plan.layer_plan.intermediate_size, INTERMEDIATE_SIZE);

    // Per-layer byte strides of every bulk resource, in the documented
    // dynamic-offset scheme: norm weights, the four projections, the three
    // MLP weights, both cache planes, and the hidden ping-pong slice.
    assert_eq!(
        plan.weight_stride_bytes,
        [
            (NORM_ELEMENTS * 4) as u64,
            (Q_WEIGHT_ELEMENTS * 4) as u64,
            (K_WEIGHT_ELEMENTS * 4) as u64,
            (K_WEIGHT_ELEMENTS * 4) as u64,
            (O_WEIGHT_ELEMENTS * 4) as u64,
            (NORM_ELEMENTS * 4) as u64,
            (GATE_WEIGHT_ELEMENTS * 4) as u64,
            (GATE_WEIGHT_ELEMENTS * 4) as u64,
            (DOWN_WEIGHT_ELEMENTS * 4) as u64,
        ]
    );
    assert_eq!(
        plan.cache_stride_bytes,
        (CACHE_CAPACITY as usize * KEY_VALUE_WIDTH * 4) as u64
    );
    assert_eq!(plan.hidden_stride_bytes, (HIDDEN_SIZE as usize * 4) as u64);

    // Double-move proves the plan is `Copy` rather than `Clone`-only.
    let cloned = plan;
    let copied = cloned;
    assert_eq!(cloned, copied);
}

#[test]
fn stack_plan_reuses_the_accepted_layer_plan() {
    let fixture = Fixture::new();
    let stack = fixture.descriptor().plan().unwrap();
    let layer = DecoderLayerDescriptor {
        attention: DecoderAttentionBlockDescriptor {
            hidden_size: HIDDEN_SIZE,
            query_heads: QUERY_HEADS,
            key_value_heads: KEY_VALUE_HEADS,
            head_dim: HEAD_DIM,
            rms_norm_epsilon: RMS_NORM_EPSILON,
            norm1_weight: &fixture.norm1_weight[..NORM_ELEMENTS],
            q_weight: &fixture.q_weight[..Q_WEIGHT_ELEMENTS],
            k_weight: &fixture.k_weight[..K_WEIGHT_ELEMENTS],
            v_weight: &fixture.v_weight[..K_WEIGHT_ELEMENTS],
            o_weight: &fixture.o_weight[..O_WEIGHT_ELEMENTS],
            mrope_cos: &fixture.mrope_cos,
            mrope_sin: &fixture.mrope_sin,
            cache_capacity: CACHE_CAPACITY,
        },
        intermediate_size: INTERMEDIATE_SIZE,
        norm2_weight: &fixture.norm2_weight[..NORM_ELEMENTS],
        gate_weight: &fixture.gate_weight[..GATE_WEIGHT_ELEMENTS],
        up_weight: &fixture.up_weight[..GATE_WEIGHT_ELEMENTS],
        down_weight: &fixture.down_weight[..DOWN_WEIGHT_ELEMENTS],
    }
    .plan()
    .unwrap();
    assert_eq!(stack.layer_plan, layer);
}

#[test]
fn plan_depends_on_geometry_not_operand_values() {
    let fixture = Fixture::new();
    let same = fixture.descriptor().plan().unwrap();
    let different_values = DecoderStackDescriptor {
        norm1_weight: &finite_values(LAYERS as usize * NORM_ELEMENTS, 0.059),
        q_weight: &finite_values(LAYERS as usize * Q_WEIGHT_ELEMENTS, 0.061),
        k_weight: &finite_values(LAYERS as usize * K_WEIGHT_ELEMENTS, 0.067),
        v_weight: &finite_values(LAYERS as usize * K_WEIGHT_ELEMENTS, 0.071),
        o_weight: &finite_values(LAYERS as usize * O_WEIGHT_ELEMENTS, 0.073),
        mrope_cos: &finite_values(ROPE_ELEMENTS, 0.079),
        mrope_sin: &finite_values(ROPE_ELEMENTS, 0.083),
        norm2_weight: &finite_values(LAYERS as usize * NORM_ELEMENTS, 0.089),
        gate_weight: &finite_values(LAYERS as usize * GATE_WEIGHT_ELEMENTS, 0.097),
        up_weight: &finite_values(LAYERS as usize * GATE_WEIGHT_ELEMENTS, 0.101),
        down_weight: &finite_values(LAYERS as usize * DOWN_WEIGHT_ELEMENTS, 0.103),
        ..fixture.descriptor()
    }
    .plan()
    .unwrap();
    assert_eq!(same, different_values);
}

#[test]
fn descriptor_rejects_layer_count_drift() {
    let fixture = Fixture::new();
    let base = fixture.descriptor();
    for layers in [0, 1, LAYERS - 1, LAYERS + 1, 36] {
        assert_descriptor_error(
            DecoderStackDescriptor { layers, ..base },
            InvocationErrorCode::InvalidDecoderGeometry,
        );
    }
}

#[test]
fn descriptor_rejects_inherited_geometry_drift() {
    let fixture = Fixture::new();
    let base = fixture.descriptor();

    assert_descriptor_error(
        DecoderStackDescriptor {
            hidden_size: HIDDEN_SIZE / 2,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderStackDescriptor {
            intermediate_size: INTERMEDIATE_SIZE / 2,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderStackDescriptor {
            query_heads: 8,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderStackDescriptor {
            key_value_heads: 4,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderStackDescriptor {
            head_dim: 64,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderStackDescriptor {
            rms_norm_epsilon: 1.0e-6,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );

    // Inherited geometry drift outranks a bulk length violation.
    let short_gate = &fixture.gate_weight[..fixture.gate_weight.len() - 1];
    assert_descriptor_error(
        DecoderStackDescriptor {
            intermediate_size: INTERMEDIATE_SIZE / 2,
            gate_weight: short_gate,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
}

#[test]
fn stack_strides_track_the_admitted_cache_capacity() {
    let fixture = Fixture::new();
    let base_plan = fixture.descriptor().plan().unwrap();
    let narrower = DecoderStackDescriptor {
        cache_capacity: CACHE_CAPACITY - 1,
        mrope_cos: &fixture.mrope_cos[..3 * (CACHE_CAPACITY as usize - 1) * HEAD_DIM as usize],
        mrope_sin: &fixture.mrope_sin[..3 * (CACHE_CAPACITY as usize - 1) * HEAD_DIM as usize],
        ..fixture.descriptor()
    }
    .plan()
    .unwrap();

    assert_ne!(base_plan, narrower);
    assert_eq!(
        narrower.cache_stride_bytes,
        ((CACHE_CAPACITY as usize - 1) * KEY_VALUE_WIDTH * 4) as u64
    );
    assert_eq!(narrower.cache_stride_bytes, 8_192);
    assert_eq!(narrower.weight_stride_bytes, base_plan.weight_stride_bytes);
    assert_eq!(narrower.hidden_stride_bytes, base_plan.hidden_stride_bytes);
}

#[test]
fn descriptor_rejects_bulk_weight_length_drift() {
    let fixture = Fixture::new();
    let base = fixture.descriptor();

    let bulk_fields: [usize; 9] = [
        fixture.norm1_weight.len(),
        fixture.q_weight.len(),
        fixture.k_weight.len(),
        fixture.v_weight.len(),
        fixture.o_weight.len(),
        fixture.norm2_weight.len(),
        fixture.gate_weight.len(),
        fixture.up_weight.len(),
        fixture.down_weight.len(),
    ];
    let scales = [
        0.011, 0.013, 0.017, 0.019, 0.023, 0.037, 0.041, 0.043, 0.047,
    ];
    for (field_index, length) in bulk_fields.iter().enumerate() {
        let short = finite_values(length - 1, scales[field_index]);
        let overlong = finite_values(length + 1, scales[field_index]);
        for target in [&short, &overlong] {
            let descriptor = match field_index {
                0 => DecoderStackDescriptor {
                    norm1_weight: target,
                    ..base
                },
                1 => DecoderStackDescriptor {
                    q_weight: target,
                    ..base
                },
                2 => DecoderStackDescriptor {
                    k_weight: target,
                    ..base
                },
                3 => DecoderStackDescriptor {
                    v_weight: target,
                    ..base
                },
                4 => DecoderStackDescriptor {
                    o_weight: target,
                    ..base
                },
                5 => DecoderStackDescriptor {
                    norm2_weight: target,
                    ..base
                },
                6 => DecoderStackDescriptor {
                    gate_weight: target,
                    ..base
                },
                7 => DecoderStackDescriptor {
                    up_weight: target,
                    ..base
                },
                _ => DecoderStackDescriptor {
                    down_weight: target,
                    ..base
                },
            };
            let error = descriptor.plan().unwrap_err();
            assert_eq!(
                error.code(),
                InvocationErrorCode::LengthMismatch,
                "field {field_index} length {}: {error}",
                target.len()
            );
        }
    }

    // The shared rope tables are validated with their full [3, T, 128] shape.
    let overlong_cos = finite_values(ROPE_ELEMENTS + 1, 0.029);
    let overlong_sin = finite_values(ROPE_ELEMENTS + 1, 0.031);
    for (cos, sin) in [
        (
            &fixture.mrope_cos[..ROPE_ELEMENTS - 1],
            &fixture.mrope_sin[..],
        ),
        (
            &fixture.mrope_cos[..],
            &fixture.mrope_sin[..ROPE_ELEMENTS - 1],
        ),
        (&overlong_cos[..], &fixture.mrope_sin[..]),
        (&fixture.mrope_cos[..], &overlong_sin[..]),
    ] {
        assert_descriptor_error(
            DecoderStackDescriptor {
                mrope_cos: cos,
                mrope_sin: sin,
                ..base
            },
            InvocationErrorCode::LengthMismatch,
        );
    }
}

#[test]
fn descriptor_rejects_nonfinite_bulk_operands_beyond_the_first_layer() {
    let fixture = Fixture::new();
    let base = fixture.descriptor();
    let poison_values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

    let bulk_fields: [usize; 9] = [
        fixture.norm1_weight.len(),
        fixture.q_weight.len(),
        fixture.k_weight.len(),
        fixture.v_weight.len(),
        fixture.o_weight.len(),
        fixture.norm2_weight.len(),
        fixture.gate_weight.len(),
        fixture.up_weight.len(),
        fixture.down_weight.len(),
    ];
    let scales = [
        0.011, 0.013, 0.017, 0.019, 0.023, 0.037, 0.041, 0.043, 0.047,
    ];
    for (field_index, length) in bulk_fields.iter().enumerate() {
        // Positions: the first element, the middle of the stack (inside a
        // middle layer's slice), deep inside the last layer's slice, and the
        // final element of the whole bulk.
        for position in [
            0,
            length / 2,
            length - (length / LAYERS as usize) / 2,
            length - 1,
        ] {
            for poison in poison_values {
                let mut target = finite_values(*length, scales[field_index]);
                target[position] = poison;
                let descriptor = match field_index {
                    0 => DecoderStackDescriptor {
                        norm1_weight: &target,
                        ..base
                    },
                    1 => DecoderStackDescriptor {
                        q_weight: &target,
                        ..base
                    },
                    2 => DecoderStackDescriptor {
                        k_weight: &target,
                        ..base
                    },
                    3 => DecoderStackDescriptor {
                        v_weight: &target,
                        ..base
                    },
                    4 => DecoderStackDescriptor {
                        o_weight: &target,
                        ..base
                    },
                    5 => DecoderStackDescriptor {
                        norm2_weight: &target,
                        ..base
                    },
                    6 => DecoderStackDescriptor {
                        gate_weight: &target,
                        ..base
                    },
                    7 => DecoderStackDescriptor {
                        up_weight: &target,
                        ..base
                    },
                    _ => DecoderStackDescriptor {
                        down_weight: &target,
                        ..base
                    },
                };
                let error = descriptor.plan().unwrap_err();
                assert_eq!(
                    error.code(),
                    InvocationErrorCode::NonFiniteInput,
                    "field {field_index} position {position} poison {poison}: {error}"
                );
            }
        }
    }

    // The shared rope tables are finiteness-checked over their full length.
    for (field_index, scale) in [(5_usize, 0.029), (6_usize, 0.031)] {
        for position in [0, ROPE_ELEMENTS / 2, ROPE_ELEMENTS - 1] {
            for poison in poison_values {
                let mut target = finite_values(ROPE_ELEMENTS, scale);
                target[position] = poison;
                let descriptor = if field_index == 5 {
                    DecoderStackDescriptor {
                        mrope_cos: &target,
                        ..base
                    }
                } else {
                    DecoderStackDescriptor {
                        mrope_sin: &target,
                        ..base
                    }
                };
                let error = descriptor.plan().unwrap_err();
                assert_eq!(
                    error.code(),
                    InvocationErrorCode::NonFiniteInput,
                    "rope field {field_index} position {position} poison {poison}: {error}"
                );
            }
        }
    }
}

#[test]
fn descriptor_checks_geometry_before_bulk_lengths_and_lengths_before_finiteness() {
    let fixture = Fixture::new();
    let base = fixture.descriptor();

    // A geometry violation outranks a bulk length violation.
    let short_gate = &fixture.gate_weight[..fixture.gate_weight.len() - 1];
    assert_descriptor_error(
        DecoderStackDescriptor {
            layers: LAYERS + 1,
            gate_weight: short_gate,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );

    // A bulk length violation outranks a bulk finiteness violation.
    let mut short_poisoned_gate = fixture.gate_weight[..fixture.gate_weight.len() - 1].to_vec();
    short_poisoned_gate[0] = f32::NAN;
    assert_descriptor_error(
        DecoderStackDescriptor {
            gate_weight: &short_poisoned_gate,
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
}

#[test]
fn step_plan_delegates_to_the_accepted_layer_step_plan() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();
    let hidden = hidden_row_fixture();

    for position in [0, 3, CACHE_CAPACITY - 1] {
        let stack_step = plan
            .plan_step(
                position,
                &DecoderStackStep {
                    hidden_row: &hidden,
                },
            )
            .unwrap();
        let layer_step = plan
            .layer_plan
            .plan_step(
                position,
                &DecoderLayerStep {
                    hidden_row: &hidden,
                },
            )
            .unwrap();
        assert_eq!(stack_step, layer_step);
    }

    // Position bounds and hidden-row admission are the layer step plan's.
    let error = plan
        .plan_step(
            CACHE_CAPACITY,
            &DecoderStackStep {
                hidden_row: &hidden,
            },
        )
        .unwrap_err();
    assert_eq!(
        error.code(),
        InvocationErrorCode::InvalidDecoderGeometry,
        "{error}"
    );
    let error = plan
        .plan_step(
            3,
            &DecoderStackStep {
                hidden_row: &hidden[..HIDDEN_SIZE as usize - 1],
            },
        )
        .unwrap_err();
    assert_eq!(error.code(), InvocationErrorCode::LengthMismatch, "{error}");
}
