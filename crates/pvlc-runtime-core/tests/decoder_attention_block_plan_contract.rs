use pvlc_runtime_core::{
    DecoderAttentionBlockDescriptor, DecoderAttentionBlockStep, InvocationErrorCode,
    InvocationPlan, KernelId,
};

const HIDDEN_SIZE: u32 = 1024;
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
        }
    }

    fn descriptor(&self) -> DecoderAttentionBlockDescriptor<'_> {
        DecoderAttentionBlockDescriptor {
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
        }
    }
}

fn finite_values(length: usize, scale: f32) -> Vec<f32> {
    (0..length)
        .map(|index| ((index * 31 + 7) as f32 * scale).sin())
        .collect()
}

fn assert_descriptor_error(
    descriptor: DecoderAttentionBlockDescriptor<'_>,
    expected: InvocationErrorCode,
) {
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

#[test]
fn descriptor_freezes_pinned_attention_block_geometry_and_exact_stage_plans() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();

    assert_eq!(plan.hidden_size, HIDDEN_SIZE);
    assert_eq!(plan.query_heads, QUERY_HEADS);
    assert_eq!(plan.key_value_heads, KEY_VALUE_HEADS);
    assert_eq!(plan.head_dim, HEAD_DIM);
    assert_eq!(plan.query_width, QUERY_WIDTH);
    assert_eq!(plan.key_value_width, KEY_VALUE_WIDTH);
    assert_eq!(plan.rope_elements, ROPE_ELEMENTS);
    assert_eq!(plan.cache_capacity, CACHE_CAPACITY);
    assert_eq!(plan.rms_norm_epsilon, RMS_NORM_EPSILON);
    assert_eq!(plan.mrope_sections, MROPE_SECTIONS);

    assert_eq!(
        plan.rms_norm_invocation,
        stage(KernelId::RmsNormF32, HIDDEN_SIZE as usize, 1)
    );
    assert_eq!(
        plan.query_invocation,
        tiled_gemv_stage(QUERY_WIDTH, (QUERY_WIDTH as u32).div_ceil(8))
    );
    assert_eq!(
        plan.key_invocation,
        tiled_gemv_stage(KEY_VALUE_WIDTH, (KEY_VALUE_WIDTH as u32).div_ceil(8))
    );
    assert_eq!(plan.value_invocation, plan.key_invocation);
    assert_eq!(
        plan.output_invocation,
        tiled_gemv_stage(HIDDEN_SIZE as usize, HIDDEN_SIZE.div_ceil(8))
    );
    assert_eq!(
        plan.mrope_invocation,
        stage(
            KernelId::DecoderMropeF32,
            ((QUERY_HEADS + KEY_VALUE_HEADS) * HEAD_DIM) as usize,
            ((QUERY_HEADS + KEY_VALUE_HEADS) * HEAD_DIM).div_ceil(64),
        )
    );
    assert_eq!(
        plan.residual_invocation,
        stage(
            KernelId::AddF32,
            HIDDEN_SIZE as usize,
            HIDDEN_SIZE.div_ceil(64)
        )
    );

    // Double-move proves the plan is `Copy` rather than `Clone`-only.
    let cloned = plan;
    let copied = cloned;
    assert_eq!(cloned, copied);
}

#[test]
fn plan_depends_on_geometry_not_operand_values() {
    let fixture = Fixture::new();
    let same = fixture.descriptor().plan().unwrap();
    let different_values = DecoderAttentionBlockDescriptor {
        norm1_weight: &finite_values(HIDDEN_SIZE as usize, 0.041),
        q_weight: &finite_values(QUERY_WIDTH * HIDDEN_SIZE as usize, 0.043),
        k_weight: &finite_values(KEY_VALUE_WIDTH * HIDDEN_SIZE as usize, 0.047),
        v_weight: &finite_values(KEY_VALUE_WIDTH * HIDDEN_SIZE as usize, 0.049),
        o_weight: &finite_values(HIDDEN_SIZE as usize * QUERY_WIDTH, 0.053),
        mrope_cos: &finite_values(ROPE_ELEMENTS, 0.059),
        mrope_sin: &finite_values(ROPE_ELEMENTS, 0.061),
        ..fixture.descriptor()
    }
    .plan()
    .unwrap();
    assert_eq!(same, different_values);
}

#[test]
fn block_plan_stays_consistent_with_the_companion_kv_session_plan() {
    use pvlc_runtime_core::DecoderKvSessionDescriptor;

    let fixture = Fixture::new();
    let block = fixture.descriptor().plan().unwrap();
    let prefix_tokens = 3;
    let kv_cache_elements = CACHE_CAPACITY as usize * KEY_VALUE_WIDTH;
    let key_cache = finite_values(kv_cache_elements, 0.071);
    let value_cache = finite_values(kv_cache_elements, 0.073);
    let session = DecoderKvSessionDescriptor {
        query_heads: QUERY_HEADS,
        key_value_heads: KEY_VALUE_HEADS,
        head_dim: HEAD_DIM,
        prefix_tokens,
        cache_capacity: CACHE_CAPACITY,
        key_cache: &key_cache,
        value_cache: &value_cache,
    }
    .plan()
    .unwrap();

    assert_eq!(block.cache_capacity, session.cache_capacity);
    assert_eq!(block.key_value_width, session.key_value_width);
    assert_eq!(block.head_dim, session.head_dim);
    assert_eq!(block.query_heads, session.query_heads);
    assert_eq!(block.key_value_heads, session.key_value_heads);
    assert_eq!(block.query_width, session.query_elements);
}

#[test]
fn descriptor_rejects_topology_drift() {
    let fixture = Fixture::new();
    let base = fixture.descriptor();

    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            hidden_size: HIDDEN_SIZE / 2,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            query_heads: 8,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            key_value_heads: 4,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            head_dim: 64,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            head_dim: 256,
            ..base
        },
        InvocationErrorCode::UnsupportedHeadDimension,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            rms_norm_epsilon: 1.0e-6,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            rms_norm_epsilon: 0.0,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            rms_norm_epsilon: -1.0e-5,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            rms_norm_epsilon: f32::NAN,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            rms_norm_epsilon: f32::INFINITY,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            rms_norm_epsilon: f32::NEG_INFINITY,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    // Deliberate contract choice: zero capacity is a geometry violation here,
    // while capacity overflow surfaces as ArithmeticOverflow.
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            cache_capacity: 0,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            cache_capacity: u32::MAX,
            ..base
        },
        InvocationErrorCode::ArithmeticOverflow,
    );
}

#[test]
fn descriptor_rejects_weight_length_drift() {
    let fixture = Fixture::new();
    let base = fixture.descriptor();

    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            norm1_weight: &fixture.norm1_weight[..fixture.norm1_weight.len() - 1],
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            q_weight: &fixture.q_weight[..fixture.q_weight.len() - 1],
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            k_weight: &fixture.k_weight[..fixture.k_weight.len() - 1],
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            v_weight: &fixture.v_weight[..fixture.v_weight.len() - 1],
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            o_weight: &fixture.o_weight[..fixture.o_weight.len() - 1],
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            mrope_cos: &fixture.mrope_cos[..fixture.mrope_cos.len() - 1],
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            mrope_sin: &fixture.mrope_sin[..fixture.mrope_sin.len() - 1],
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );

    let overlong_norm = finite_values(HIDDEN_SIZE as usize + 1, 0.011);
    let overlong_q = finite_values(QUERY_WIDTH * HIDDEN_SIZE as usize + 1, 0.013);
    let overlong_kv = finite_values(KEY_VALUE_WIDTH * HIDDEN_SIZE as usize + 1, 0.017);
    let overlong_o = finite_values(HIDDEN_SIZE as usize * QUERY_WIDTH + 1, 0.023);
    let overlong_rope = finite_values(ROPE_ELEMENTS + 1, 0.029);
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            norm1_weight: &overlong_norm,
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            q_weight: &overlong_q,
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            k_weight: &overlong_kv,
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            v_weight: &overlong_kv,
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            o_weight: &overlong_o,
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            mrope_cos: &overlong_rope,
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            mrope_sin: &overlong_rope,
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
}

#[test]
fn descriptor_rejects_nonfinite_operands_at_first_middle_and_last_positions() {
    let fixture = Fixture::new();
    let base = fixture.descriptor();
    let poison_values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

    let weight_fields: [usize; 7] = [
        fixture.norm1_weight.len(),
        fixture.q_weight.len(),
        fixture.k_weight.len(),
        fixture.v_weight.len(),
        fixture.o_weight.len(),
        fixture.mrope_cos.len(),
        fixture.mrope_sin.len(),
    ];
    let scales = [0.011, 0.013, 0.017, 0.019, 0.023, 0.029, 0.031];
    for (field_index, length) in weight_fields.iter().enumerate() {
        for position in [0, length / 2, length - 1] {
            for poison in poison_values {
                let mut target = finite_values(*length, scales[field_index]);
                target[position] = poison;
                let descriptor = match field_index {
                    0 => DecoderAttentionBlockDescriptor {
                        norm1_weight: &target,
                        ..base
                    },
                    1 => DecoderAttentionBlockDescriptor {
                        q_weight: &target,
                        ..base
                    },
                    2 => DecoderAttentionBlockDescriptor {
                        k_weight: &target,
                        ..base
                    },
                    3 => DecoderAttentionBlockDescriptor {
                        v_weight: &target,
                        ..base
                    },
                    4 => DecoderAttentionBlockDescriptor {
                        o_weight: &target,
                        ..base
                    },
                    5 => DecoderAttentionBlockDescriptor {
                        mrope_cos: &target,
                        ..base
                    },
                    _ => DecoderAttentionBlockDescriptor {
                        mrope_sin: &target,
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
fn descriptor_rejects_capacity_overflow_paths() {
    let fixture = Fixture::new();
    let base = fixture.descriptor();
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            cache_capacity: u32::MAX,
            ..base
        },
        InvocationErrorCode::ArithmeticOverflow,
    );
    assert_descriptor_error(
        DecoderAttentionBlockDescriptor {
            cache_capacity: u32::MAX / 128,
            ..base
        },
        InvocationErrorCode::ArithmeticOverflow,
    );
}

#[test]
fn plan_is_only_produced_by_the_exact_descriptor() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();
    let same = fixture.descriptor().plan().unwrap();
    assert_eq!(plan, same);

    let narrower = DecoderAttentionBlockDescriptor {
        cache_capacity: CACHE_CAPACITY - 1,
        mrope_cos: &fixture.mrope_cos[..3 * (CACHE_CAPACITY as usize - 1) * HEAD_DIM as usize],
        mrope_sin: &fixture.mrope_sin[..3 * (CACHE_CAPACITY as usize - 1) * HEAD_DIM as usize],
        ..fixture.descriptor()
    }
    .plan()
    .unwrap();
    assert_ne!(plan, narrower);
    assert_eq!(
        narrower.rope_elements,
        3 * (CACHE_CAPACITY as usize - 1) * HEAD_DIM as usize
    );
    assert_eq!(narrower.query_invocation, plan.query_invocation);
    assert_eq!(narrower.residual_invocation, plan.residual_invocation);
}

fn hidden_row_fixture() -> Vec<f32> {
    finite_values(HIDDEN_SIZE as usize, 0.037)
}

#[test]
fn block_step_plan_owns_exact_stage_uniform_words_in_chain_order() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();
    let hidden = hidden_row_fixture();
    let step_plan = plan
        .plan_step(
            3,
            &DecoderAttentionBlockStep {
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
        ]
    );

    // Double-move proves the step plan is `Copy` rather than `Clone`-only.
    let cloned = step_plan;
    let copied = cloned;
    assert_eq!(cloned, copied);
}

#[test]
fn block_step_plan_depends_on_position_not_hidden_values() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();
    let hidden = hidden_row_fixture();
    let other_hidden = finite_values(HIDDEN_SIZE as usize, 0.041);

    let base = plan
        .plan_step(
            3,
            &DecoderAttentionBlockStep {
                hidden_row: &hidden,
            },
        )
        .unwrap();
    let same = plan
        .plan_step(
            3,
            &DecoderAttentionBlockStep {
                hidden_row: &other_hidden,
            },
        )
        .unwrap();
    assert_eq!(base, same);

    let later = plan
        .plan_step(
            4,
            &DecoderAttentionBlockStep {
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
fn block_step_plan_rejects_hidden_row_length_and_finiteness_drift() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();
    let hidden = hidden_row_fixture();
    let overlong = finite_values(HIDDEN_SIZE as usize + 1, 0.037);

    for row in [&hidden[..HIDDEN_SIZE as usize - 1], &overlong[..]] {
        let error = plan
            .plan_step(3, &DecoderAttentionBlockStep { hidden_row: row })
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
            &DecoderAttentionBlockStep {
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
                    &DecoderAttentionBlockStep {
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
fn block_step_plan_rejects_positions_outside_the_rope_tables() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();
    let hidden = hidden_row_fixture();

    // Position zero is a valid table row: prefix enforcement belongs to the
    // companion KV cache transition, not to the attention-block step plan.
    plan.plan_step(
        0,
        &DecoderAttentionBlockStep {
            hidden_row: &hidden,
        },
    )
    .unwrap();
    plan.plan_step(
        CACHE_CAPACITY - 1,
        &DecoderAttentionBlockStep {
            hidden_row: &hidden,
        },
    )
    .unwrap();
    for position in [CACHE_CAPACITY, CACHE_CAPACITY + 1, u32::MAX] {
        let error = plan
            .plan_step(
                position,
                &DecoderAttentionBlockStep {
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
fn block_step_plan_tracks_the_admitted_cache_capacity() {
    let fixture = Fixture::new();
    let narrower = DecoderAttentionBlockDescriptor {
        cache_capacity: CACHE_CAPACITY - 1,
        mrope_cos: &fixture.mrope_cos[..3 * (CACHE_CAPACITY as usize - 1) * HEAD_DIM as usize],
        mrope_sin: &fixture.mrope_sin[..3 * (CACHE_CAPACITY as usize - 1) * HEAD_DIM as usize],
        ..fixture.descriptor()
    }
    .plan()
    .unwrap();
    let hidden = hidden_row_fixture();

    let step_plan = narrower
        .plan_step(
            7,
            &DecoderAttentionBlockStep {
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
        ]
    );

    let error = narrower
        .plan_step(
            8,
            &DecoderAttentionBlockStep {
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
fn block_step_plan_checks_position_before_hidden_row_admission() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();
    let hidden = hidden_row_fixture();
    let short = &hidden[..HIDDEN_SIZE as usize - 1];
    let error = plan
        .plan_step(
            CACHE_CAPACITY,
            &DecoderAttentionBlockStep { hidden_row: short },
        )
        .unwrap_err();
    assert_eq!(
        error.code(),
        InvocationErrorCode::InvalidDecoderGeometry,
        "{error}"
    );
}

fn kv_session_plan(prefix_tokens: u32) -> pvlc_runtime_core::DecoderKvSessionPlan {
    kv_session_plan_with_capacity(prefix_tokens, CACHE_CAPACITY)
}

fn kv_session_plan_with_capacity(
    prefix_tokens: u32,
    cache_capacity: u32,
) -> pvlc_runtime_core::DecoderKvSessionPlan {
    use pvlc_runtime_core::DecoderKvSessionDescriptor;

    let kv_cache_elements = cache_capacity as usize * KEY_VALUE_WIDTH;
    let key_cache = finite_values(kv_cache_elements, 0.071);
    let value_cache = finite_values(kv_cache_elements, 0.073);
    DecoderKvSessionDescriptor {
        query_heads: QUERY_HEADS,
        key_value_heads: KEY_VALUE_HEADS,
        head_dim: HEAD_DIM,
        prefix_tokens,
        cache_capacity,
        key_cache: &key_cache,
        value_cache: &value_cache,
    }
    .plan()
    .unwrap()
}

#[test]
fn kv_cache_transition_matches_plan_step_transition_without_host_operands() {
    use pvlc_runtime_core::DecoderKvSessionStep;

    let query = finite_values(QUERY_WIDTH, 0.043);
    let appended_key = finite_values(KEY_VALUE_WIDTH, 0.047);
    let appended_value = finite_values(KEY_VALUE_WIDTH, 0.049);
    let step = DecoderKvSessionStep {
        query: &query,
        appended_key: &appended_key,
        appended_value: &appended_value,
    };

    for (prefix_tokens, cache_capacity, accepted, rejected) in [
        (
            3,
            CACHE_CAPACITY,
            &[3, 4, CACHE_CAPACITY - 1][..],
            &[0, 2][..],
        ),
        (8, CACHE_CAPACITY, &[8][..], &[7][..]),
        (1, 2, &[1][..], &[0][..]),
    ] {
        let plan = kv_session_plan_with_capacity(prefix_tokens, cache_capacity);
        for cache_tokens in accepted {
            assert_eq!(
                plan.plan_cache_transition(*cache_tokens).unwrap(),
                plan.plan_step(*cache_tokens, &step).unwrap(),
                "transition at ({prefix_tokens}, {cache_capacity}) tokens {cache_tokens} must equal the full plan_step transition"
            );
        }
        for cache_tokens in rejected {
            let error = plan.plan_cache_transition(*cache_tokens).unwrap_err();
            assert_eq!(
                error.code(),
                InvocationErrorCode::InvalidDecoderGeometry,
                "({prefix_tokens}, {cache_capacity}) cache_tokens {cache_tokens}: {error}"
            );
        }
        for cache_tokens in [cache_capacity, u32::MAX] {
            let error = plan.plan_cache_transition(cache_tokens).unwrap_err();
            assert_eq!(
                error.code(),
                InvocationErrorCode::InvalidDecoderGeometry,
                "({prefix_tokens}, {cache_capacity}) cache_tokens {cache_tokens}: {error}"
            );
        }
    }
}

#[test]
fn kv_plan_step_error_precedence_is_unchanged_by_transition_delegation() {
    use pvlc_runtime_core::DecoderKvSessionStep;

    let plan = kv_session_plan(3);
    let query = finite_values(QUERY_WIDTH, 0.043);
    let short_query = &query[..QUERY_WIDTH - 1];
    let mut poisoned_key = finite_values(KEY_VALUE_WIDTH, 0.047);
    poisoned_key[0] = f32::NAN;
    let appended_value = finite_values(KEY_VALUE_WIDTH, 0.049);

    // Geometry violations still precede every operand check.
    for cache_tokens in [2, CACHE_CAPACITY] {
        let error = plan
            .plan_step(
                cache_tokens,
                &DecoderKvSessionStep {
                    query: short_query,
                    appended_key: &poisoned_key,
                    appended_value: &appended_value,
                },
            )
            .unwrap_err();
        assert_eq!(
            error.code(),
            InvocationErrorCode::InvalidDecoderGeometry,
            "cache_tokens {cache_tokens}: {error}"
        );
    }

    // With valid geometry, operand length admission still precedes finiteness.
    let error = plan
        .plan_step(
            3,
            &DecoderKvSessionStep {
                query: short_query,
                appended_key: &poisoned_key,
                appended_value: &appended_value,
            },
        )
        .unwrap_err();
    assert_eq!(error.code(), InvocationErrorCode::LengthMismatch, "{error}");
    let error = plan
        .plan_step(
            3,
            &DecoderKvSessionStep {
                query: &query,
                appended_key: &poisoned_key,
                appended_value: &appended_value,
            },
        )
        .unwrap_err();
    assert_eq!(error.code(), InvocationErrorCode::NonFiniteInput, "{error}");
}
