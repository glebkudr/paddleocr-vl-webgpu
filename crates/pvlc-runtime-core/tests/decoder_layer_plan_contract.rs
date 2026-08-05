use pvlc_runtime_core::{
    DecoderAttentionBlockDescriptor, DecoderAttentionBlockStep, DecoderLayerDescriptor,
    DecoderLayerStep, InvocationErrorCode, InvocationPlan, KernelId,
};

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
            norm1_weight: finite_values(HIDDEN_SIZE as usize, 0.011),
            q_weight: finite_values(QUERY_WIDTH * HIDDEN_SIZE as usize, 0.013),
            k_weight: finite_values(KEY_VALUE_WIDTH * HIDDEN_SIZE as usize, 0.017),
            v_weight: finite_values(KEY_VALUE_WIDTH * HIDDEN_SIZE as usize, 0.019),
            o_weight: finite_values(HIDDEN_SIZE as usize * QUERY_WIDTH, 0.023),
            mrope_cos: finite_values(ROPE_ELEMENTS, 0.029),
            mrope_sin: finite_values(ROPE_ELEMENTS, 0.031),
            norm2_weight: finite_values(HIDDEN_SIZE as usize, 0.037),
            gate_weight: finite_values(INTERMEDIATE_SIZE as usize * HIDDEN_SIZE as usize, 0.041),
            up_weight: finite_values(INTERMEDIATE_SIZE as usize * HIDDEN_SIZE as usize, 0.043),
            down_weight: finite_values(HIDDEN_SIZE as usize * INTERMEDIATE_SIZE as usize, 0.047),
        }
    }

    fn descriptor(&self) -> DecoderLayerDescriptor<'_> {
        DecoderLayerDescriptor {
            attention: DecoderAttentionBlockDescriptor {
                hidden_size: HIDDEN_SIZE,
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
                cache_capacity: CACHE_CAPACITY,
            },
            intermediate_size: INTERMEDIATE_SIZE,
            norm2_weight: &self.norm2_weight,
            gate_weight: &self.gate_weight,
            up_weight: &self.up_weight,
            down_weight: &self.down_weight,
        }
    }
}

fn finite_values(length: usize, scale: f32) -> Vec<f32> {
    (0..length)
        .map(|index| ((index * 31 + 7) as f32 * scale).sin())
        .collect()
}

fn assert_descriptor_error(descriptor: DecoderLayerDescriptor<'_>, expected: InvocationErrorCode) {
    let error = descriptor.plan().unwrap_err();
    assert_eq!(error.code(), expected, "{error}");
}

fn stage(kernel: KernelId, output_elements: usize, dispatch_x: u32) -> InvocationPlan {
    InvocationPlan {
        kernel,
        output_elements,
        output_bytes: (output_elements * size_of::<f32>()) as u64,
        workgroup_size: [64, 1, 1],
        dispatch: [dispatch_x, 1, 1],
    }
}

fn tiled_gemv_stage(output_elements: usize, dispatch_x: u32) -> InvocationPlan {
    InvocationPlan {
        kernel: KernelId::GemvTiledF32,
        output_elements,
        output_bytes: (output_elements * size_of::<f32>()) as u64,
        workgroup_size: [256, 1, 1],
        dispatch: [dispatch_x, 1, 1],
    }
}

fn hidden_row_fixture() -> Vec<f32> {
    finite_values(HIDDEN_SIZE as usize, 0.053)
}

#[test]
fn descriptor_freezes_pinned_layer_geometry_and_exact_thirteen_stage_plans() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();

    assert_eq!(plan.attention_block.hidden_size, HIDDEN_SIZE);
    assert_eq!(plan.attention_block.query_heads, QUERY_HEADS);
    assert_eq!(plan.attention_block.key_value_heads, KEY_VALUE_HEADS);
    assert_eq!(plan.attention_block.head_dim, HEAD_DIM);
    assert_eq!(plan.attention_block.query_width, QUERY_WIDTH);
    assert_eq!(plan.attention_block.key_value_width, KEY_VALUE_WIDTH);
    assert_eq!(plan.attention_block.rope_elements, ROPE_ELEMENTS);
    assert_eq!(plan.attention_block.cache_capacity, CACHE_CAPACITY);
    assert_eq!(plan.attention_block.rms_norm_epsilon, RMS_NORM_EPSILON);
    assert_eq!(plan.attention_block.mrope_sections, MROPE_SECTIONS);
    assert_eq!(plan.intermediate_size, INTERMEDIATE_SIZE);

    assert_eq!(
        plan.attention_block.rms_norm_invocation,
        stage(KernelId::RmsNormF32, HIDDEN_SIZE as usize, 1)
    );
    assert_eq!(
        plan.attention_block.query_invocation,
        tiled_gemv_stage(QUERY_WIDTH, 256)
    );
    assert_eq!(
        plan.attention_block.mrope_invocation,
        stage(KernelId::DecoderMropeF32, 2304, 36)
    );
    assert_eq!(
        plan.attention_block.residual_invocation,
        stage(KernelId::AddF32, HIDDEN_SIZE as usize, 16)
    );

    assert_eq!(
        plan.norm2_invocation,
        stage(KernelId::RmsNormF32, HIDDEN_SIZE as usize, 1)
    );
    assert_eq!(
        plan.gate_invocation,
        tiled_gemv_stage(INTERMEDIATE_SIZE as usize, 384)
    );
    assert_eq!(
        plan.up_invocation,
        tiled_gemv_stage(INTERMEDIATE_SIZE as usize, 384)
    );
    assert_eq!(
        plan.swiglu_invocation,
        stage(KernelId::DecoderSwigluF32, INTERMEDIATE_SIZE as usize, 48)
    );
    assert_eq!(
        plan.down_invocation,
        tiled_gemv_stage(HIDDEN_SIZE as usize, 128)
    );
    assert_eq!(
        plan.second_residual_invocation,
        stage(KernelId::AddF32, HIDDEN_SIZE as usize, 16)
    );

    // Double-move proves the plan is `Copy` rather than `Clone`-only.
    let cloned = plan;
    let copied = cloned;
    assert_eq!(cloned, copied);
}

#[test]
fn layer_plan_attention_half_equals_the_standalone_attention_block_plan() {
    let fixture = Fixture::new();
    let layer = fixture.descriptor().plan().unwrap();
    let attention = DecoderAttentionBlockDescriptor {
        hidden_size: HIDDEN_SIZE,
        query_heads: QUERY_HEADS,
        key_value_heads: KEY_VALUE_HEADS,
        head_dim: HEAD_DIM,
        rms_norm_epsilon: RMS_NORM_EPSILON,
        norm1_weight: &fixture.norm1_weight,
        q_weight: &fixture.q_weight,
        k_weight: &fixture.k_weight,
        v_weight: &fixture.v_weight,
        o_weight: &fixture.o_weight,
        mrope_cos: &fixture.mrope_cos,
        mrope_sin: &fixture.mrope_sin,
        cache_capacity: CACHE_CAPACITY,
    }
    .plan()
    .unwrap();
    assert_eq!(layer.attention_block, attention);
}

#[test]
fn plan_depends_on_geometry_not_operand_values() {
    let fixture = Fixture::new();
    let same = fixture.descriptor().plan().unwrap();
    let different_values = DecoderLayerDescriptor {
        attention: DecoderAttentionBlockDescriptor {
            norm1_weight: &finite_values(HIDDEN_SIZE as usize, 0.059),
            q_weight: &finite_values(QUERY_WIDTH * HIDDEN_SIZE as usize, 0.061),
            k_weight: &finite_values(KEY_VALUE_WIDTH * HIDDEN_SIZE as usize, 0.067),
            v_weight: &finite_values(KEY_VALUE_WIDTH * HIDDEN_SIZE as usize, 0.071),
            o_weight: &finite_values(HIDDEN_SIZE as usize * QUERY_WIDTH, 0.073),
            mrope_cos: &finite_values(ROPE_ELEMENTS, 0.079),
            mrope_sin: &finite_values(ROPE_ELEMENTS, 0.083),
            ..fixture.descriptor().attention
        },
        norm2_weight: &finite_values(HIDDEN_SIZE as usize, 0.089),
        gate_weight: &finite_values(INTERMEDIATE_SIZE as usize * HIDDEN_SIZE as usize, 0.097),
        up_weight: &finite_values(INTERMEDIATE_SIZE as usize * HIDDEN_SIZE as usize, 0.101),
        down_weight: &finite_values(HIDDEN_SIZE as usize * INTERMEDIATE_SIZE as usize, 0.103),
        ..fixture.descriptor()
    }
    .plan()
    .unwrap();
    assert_eq!(same, different_values);
}

#[test]
fn descriptor_rejects_mlp_topology_drift() {
    let fixture = Fixture::new();
    let base = fixture.descriptor();

    assert_descriptor_error(
        DecoderLayerDescriptor {
            intermediate_size: INTERMEDIATE_SIZE / 2,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderLayerDescriptor {
            intermediate_size: INTERMEDIATE_SIZE * 2,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderLayerDescriptor {
            intermediate_size: 0,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );

    // Attention-half violations still surface from the composed attention
    // descriptor before any MLP operand is examined.
    assert_descriptor_error(
        DecoderLayerDescriptor {
            attention: DecoderAttentionBlockDescriptor {
                query_heads: 8,
                ..base.attention
            },
            norm2_weight: &fixture.norm2_weight[..fixture.norm2_weight.len() - 1],
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
}

#[test]
fn descriptor_rejects_mlp_weight_length_drift() {
    let fixture = Fixture::new();
    let base = fixture.descriptor();

    assert_descriptor_error(
        DecoderLayerDescriptor {
            norm2_weight: &fixture.norm2_weight[..fixture.norm2_weight.len() - 1],
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderLayerDescriptor {
            gate_weight: &fixture.gate_weight[..fixture.gate_weight.len() - 1],
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderLayerDescriptor {
            up_weight: &fixture.up_weight[..fixture.up_weight.len() - 1],
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderLayerDescriptor {
            down_weight: &fixture.down_weight[..fixture.down_weight.len() - 1],
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );

    let overlong_norm2 = finite_values(HIDDEN_SIZE as usize + 1, 0.037);
    let overlong_gate = finite_values(INTERMEDIATE_SIZE as usize * HIDDEN_SIZE as usize + 1, 0.041);
    let overlong_up = finite_values(INTERMEDIATE_SIZE as usize * HIDDEN_SIZE as usize + 1, 0.043);
    let overlong_down = finite_values(HIDDEN_SIZE as usize * INTERMEDIATE_SIZE as usize + 1, 0.047);
    assert_descriptor_error(
        DecoderLayerDescriptor {
            norm2_weight: &overlong_norm2,
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderLayerDescriptor {
            gate_weight: &overlong_gate,
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderLayerDescriptor {
            up_weight: &overlong_up,
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderLayerDescriptor {
            down_weight: &overlong_down,
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
}

#[test]
fn descriptor_rejects_nonfinite_mlp_operands_at_first_middle_and_last_positions() {
    let fixture = Fixture::new();
    let base = fixture.descriptor();
    let poison_values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

    let weight_fields: [usize; 4] = [
        fixture.norm2_weight.len(),
        fixture.gate_weight.len(),
        fixture.up_weight.len(),
        fixture.down_weight.len(),
    ];
    let scales = [0.037, 0.041, 0.043, 0.047];
    for (field_index, length) in weight_fields.iter().enumerate() {
        for position in [0, length / 2, length - 1] {
            for poison in poison_values {
                let mut target = finite_values(*length, scales[field_index]);
                target[position] = poison;
                let descriptor = match field_index {
                    0 => DecoderLayerDescriptor {
                        norm2_weight: &target,
                        ..base
                    },
                    1 => DecoderLayerDescriptor {
                        gate_weight: &target,
                        ..base
                    },
                    2 => DecoderLayerDescriptor {
                        up_weight: &target,
                        ..base
                    },
                    _ => DecoderLayerDescriptor {
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
}

#[test]
fn layer_step_plan_owns_exact_thirteen_stage_uniform_words_in_chain_order() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();
    let hidden = hidden_row_fixture();
    let step_plan = plan
        .plan_step(
            3,
            &DecoderLayerStep {
                hidden_row: &hidden,
            },
        )
        .unwrap();

    assert_eq!(step_plan.position, 3);
    assert_eq!(
        step_plan.stage_uniform_words,
        [
            // rmsnorm: rows, width, epsilon bits, padding.
            [1, HIDDEN_SIZE, RMS_NORM_EPSILON.to_bits(), 0],
            // linear q: rows, columns, padding, padding.
            [QUERY_WIDTH as u32, HIDDEN_SIZE, 0, 0],
            // linear k.
            [KEY_VALUE_WIDTH as u32, HIDDEN_SIZE, 0, 0],
            // linear v.
            [KEY_VALUE_WIDTH as u32, HIDDEN_SIZE, 0, 0],
            // mrope q/k: position, rope capacity, padding, padding.
            [3, CACHE_CAPACITY, 0, 0],
            // linear o.
            [HIDDEN_SIZE, QUERY_WIDTH as u32, 0, 0],
            // residual add: length, padding, padding, padding.
            [HIDDEN_SIZE, 0, 0, 0],
            // post-attention rmsnorm.
            [1, HIDDEN_SIZE, RMS_NORM_EPSILON.to_bits(), 0],
            // linear gate.
            [INTERMEDIATE_SIZE, HIDDEN_SIZE, 0, 0],
            // linear up.
            [INTERMEDIATE_SIZE, HIDDEN_SIZE, 0, 0],
            // swiglu: length, padding, padding, padding.
            [INTERMEDIATE_SIZE, 0, 0, 0],
            // linear down.
            [HIDDEN_SIZE, INTERMEDIATE_SIZE, 0, 0],
            // second residual add.
            [HIDDEN_SIZE, 0, 0, 0],
        ]
    );

    // Double-move proves the step plan is `Copy` rather than `Clone`-only.
    let cloned = step_plan;
    let copied = cloned;
    assert_eq!(cloned, copied);
}

#[test]
fn layer_step_plan_matches_the_attention_step_plan_on_the_shared_prefix() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();
    let hidden = hidden_row_fixture();
    let layer_step = plan
        .plan_step(
            3,
            &DecoderLayerStep {
                hidden_row: &hidden,
            },
        )
        .unwrap();
    let attention_step = plan
        .attention_block
        .plan_step(
            3,
            &DecoderAttentionBlockStep {
                hidden_row: &hidden,
            },
        )
        .unwrap();
    assert_eq!(
        layer_step.stage_uniform_words[..7],
        attention_step.stage_uniform_words
    );
}

#[test]
fn layer_step_plan_depends_on_position_not_hidden_values() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();
    let hidden = hidden_row_fixture();
    let other_hidden = finite_values(HIDDEN_SIZE as usize, 0.059);

    let base = plan
        .plan_step(
            3,
            &DecoderLayerStep {
                hidden_row: &hidden,
            },
        )
        .unwrap();
    let same = plan
        .plan_step(
            3,
            &DecoderLayerStep {
                hidden_row: &other_hidden,
            },
        )
        .unwrap();
    assert_eq!(base, same);

    let later = plan
        .plan_step(
            4,
            &DecoderLayerStep {
                hidden_row: &hidden,
            },
        )
        .unwrap();
    assert_ne!(base, later);
    assert_eq!(later.position, 4);
    let mut expected_words = base.stage_uniform_words;
    expected_words[4][0] = 4;
    assert_eq!(later.stage_uniform_words, expected_words);
}

#[test]
fn layer_step_plan_tracks_the_admitted_cache_capacity() {
    let fixture = Fixture::new();
    let narrower = DecoderLayerDescriptor {
        attention: DecoderAttentionBlockDescriptor {
            cache_capacity: CACHE_CAPACITY - 1,
            mrope_cos: &fixture.mrope_cos[..3 * (CACHE_CAPACITY as usize - 1) * HEAD_DIM as usize],
            mrope_sin: &fixture.mrope_sin[..3 * (CACHE_CAPACITY as usize - 1) * HEAD_DIM as usize],
            ..fixture.descriptor().attention
        },
        ..fixture.descriptor()
    }
    .plan()
    .unwrap();
    let hidden = hidden_row_fixture();

    let step_plan = narrower
        .plan_step(
            7,
            &DecoderLayerStep {
                hidden_row: &hidden,
            },
        )
        .unwrap();
    assert_eq!(step_plan.position, 7);
    assert_eq!(
        step_plan.stage_uniform_words,
        [
            [1, HIDDEN_SIZE, RMS_NORM_EPSILON.to_bits(), 0],
            [QUERY_WIDTH as u32, HIDDEN_SIZE, 0, 0],
            [KEY_VALUE_WIDTH as u32, HIDDEN_SIZE, 0, 0],
            [KEY_VALUE_WIDTH as u32, HIDDEN_SIZE, 0, 0],
            [7, CACHE_CAPACITY - 1, 0, 0],
            [HIDDEN_SIZE, QUERY_WIDTH as u32, 0, 0],
            [HIDDEN_SIZE, 0, 0, 0],
            [1, HIDDEN_SIZE, RMS_NORM_EPSILON.to_bits(), 0],
            [INTERMEDIATE_SIZE, HIDDEN_SIZE, 0, 0],
            [INTERMEDIATE_SIZE, HIDDEN_SIZE, 0, 0],
            [INTERMEDIATE_SIZE, 0, 0, 0],
            [HIDDEN_SIZE, INTERMEDIATE_SIZE, 0, 0],
            [HIDDEN_SIZE, 0, 0, 0],
        ]
    );

    let error = narrower
        .plan_step(
            8,
            &DecoderLayerStep {
                hidden_row: &hidden,
            },
        )
        .unwrap_err();
    assert_eq!(
        error.code(),
        InvocationErrorCode::InvalidDecoderGeometry,
        "{error}"
    );
}

#[test]
fn layer_step_plan_rejects_hidden_row_length_and_finiteness_drift() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();
    let hidden = hidden_row_fixture();
    let overlong = finite_values(HIDDEN_SIZE as usize + 1, 0.053);

    for row in [&hidden[..HIDDEN_SIZE as usize - 1], &overlong[..]] {
        let error = plan
            .plan_step(3, &DecoderLayerStep { hidden_row: row })
            .unwrap_err();
        assert_eq!(error.code(), InvocationErrorCode::LengthMismatch, "{error}");
    }

    // Length admission precedes finiteness: a short poisoned row is a length
    // error, never a finiteness error.
    let mut short_poisoned = hidden_row_fixture();
    short_poisoned[0] = f32::NAN;
    let error = plan
        .plan_step(
            3,
            &DecoderLayerStep {
                hidden_row: &short_poisoned[..HIDDEN_SIZE as usize - 1],
            },
        )
        .unwrap_err();
    assert_eq!(error.code(), InvocationErrorCode::LengthMismatch, "{error}");

    for poison_index in [0, HIDDEN_SIZE as usize / 2, HIDDEN_SIZE as usize - 1] {
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut target = hidden_row_fixture();
            target[poison_index] = poison;
            let error = plan
                .plan_step(
                    3,
                    &DecoderLayerStep {
                        hidden_row: &target,
                    },
                )
                .unwrap_err();
            assert_eq!(
                error.code(),
                InvocationErrorCode::NonFiniteInput,
                "position {poison_index} poison {poison}: {error}"
            );
        }
    }
}

#[test]
fn layer_step_plan_rejects_positions_outside_the_rope_tables() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();
    let hidden = hidden_row_fixture();

    // Position zero is a valid table row: prefix enforcement belongs to the
    // companion KV cache transition, not to the layer step plan.
    plan.plan_step(
        0,
        &DecoderLayerStep {
            hidden_row: &hidden,
        },
    )
    .unwrap();
    plan.plan_step(
        CACHE_CAPACITY - 1,
        &DecoderLayerStep {
            hidden_row: &hidden,
        },
    )
    .unwrap();
    for position in [CACHE_CAPACITY, CACHE_CAPACITY + 1, u32::MAX] {
        let error = plan
            .plan_step(
                position,
                &DecoderLayerStep {
                    hidden_row: &hidden,
                },
            )
            .unwrap_err();
        assert_eq!(
            error.code(),
            InvocationErrorCode::InvalidDecoderGeometry,
            "position {position}: {error}"
        );
    }
}

#[test]
fn layer_step_plan_checks_position_before_hidden_row_admission() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();
    let hidden = hidden_row_fixture();
    let short = &hidden[..HIDDEN_SIZE as usize - 1];
    let error = plan
        .plan_step(CACHE_CAPACITY, &DecoderLayerStep { hidden_row: short })
        .unwrap_err();
    assert_eq!(
        error.code(),
        InvocationErrorCode::InvalidDecoderGeometry,
        "{error}"
    );
}
