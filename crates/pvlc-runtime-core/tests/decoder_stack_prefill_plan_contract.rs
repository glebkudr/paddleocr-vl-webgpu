use pvlc_runtime_core::{
    DecoderStackDescriptor, DecoderStackPrefillDescriptor, DecoderStackStep, InvocationErrorCode,
    InvocationPlan, KernelId,
};

const LAYERS: u32 = 18;
const HIDDEN_SIZE: u32 = 1024;
const INTERMEDIATE_SIZE: u32 = 3072;
const QUERY_HEADS: u32 = 16;
const KEY_VALUE_HEADS: u32 = 2;
const HEAD_DIM: u32 = 128;
const CACHE_CAPACITY: u32 = 332;
const RMS_NORM_EPSILON: f32 = 1.0e-5;

const QUERY_WIDTH: u32 = QUERY_HEADS * HEAD_DIM;
const KEY_VALUE_WIDTH: u32 = KEY_VALUE_HEADS * HEAD_DIM;
const MROPE_WIDTH: u32 = (QUERY_HEADS + KEY_VALUE_HEADS) * HEAD_DIM;

const NORM_ELEMENTS: usize = HIDDEN_SIZE as usize;
const Q_WEIGHT_ELEMENTS: usize = QUERY_WIDTH as usize * HIDDEN_SIZE as usize;
const K_WEIGHT_ELEMENTS: usize = KEY_VALUE_WIDTH as usize * HIDDEN_SIZE as usize;
const O_WEIGHT_ELEMENTS: usize = HIDDEN_SIZE as usize * QUERY_WIDTH as usize;
const GATE_WEIGHT_ELEMENTS: usize = INTERMEDIATE_SIZE as usize * HIDDEN_SIZE as usize;
const DOWN_WEIGHT_ELEMENTS: usize = HIDDEN_SIZE as usize * INTERMEDIATE_SIZE as usize;
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

    fn descriptor(&self, tokens: u32) -> DecoderStackPrefillDescriptor<'_> {
        DecoderStackPrefillDescriptor {
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
            tokens,
        }
    }

    fn stack_descriptor(&self) -> DecoderStackDescriptor<'_> {
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

fn assert_descriptor_error(
    descriptor: DecoderStackPrefillDescriptor<'_>,
    expected: InvocationErrorCode,
) {
    let error = descriptor.plan().unwrap_err();
    assert_eq!(error.code(), expected, "{error}");
}

// The fifteen prefill stages in chain order, shared by every layer:
// rmsnorm, linear q, linear k, linear v, multi-token mrope q/k, kv range
// append, causal prefill gqa, linear o, residual add, post-attention rmsnorm,
// linear gate, linear up, swiglu, linear down, residual add.
fn expected_stage_invocations(tokens: u32) -> [InvocationPlan; 15] {
    [
        rms_norm_stage(tokens),
        projection_stage(QUERY_WIDTH, HIDDEN_SIZE, tokens),
        projection_stage(KEY_VALUE_WIDTH, HIDDEN_SIZE, tokens),
        projection_stage(KEY_VALUE_WIDTH, HIDDEN_SIZE, tokens),
        prefill_mrope_stage(tokens),
        kv_append_range_stage(tokens),
        prefill_gqa_stage(tokens),
        projection_stage(HIDDEN_SIZE, QUERY_WIDTH, tokens),
        add_stage(tokens, HIDDEN_SIZE),
        rms_norm_stage(tokens),
        projection_stage(INTERMEDIATE_SIZE, HIDDEN_SIZE, tokens),
        projection_stage(INTERMEDIATE_SIZE, HIDDEN_SIZE, tokens),
        swiglu_stage(tokens),
        projection_stage(HIDDEN_SIZE, INTERMEDIATE_SIZE, tokens),
        add_stage(tokens, HIDDEN_SIZE),
    ]
}

fn expected_stage_uniform_words(tokens: u32) -> [[u32; 4]; 15] {
    [
        // rmsnorm: rows, width, epsilon bits, padding.
        [tokens, HIDDEN_SIZE, RMS_NORM_EPSILON.to_bits(), 0],
        // linear q via vision_patch_projection: patches, input width, output
        // width, padding (the accepted multi-token linear ABI).
        [tokens, HIDDEN_SIZE, QUERY_WIDTH, 0],
        // linear k.
        [tokens, HIDDEN_SIZE, KEY_VALUE_WIDTH, 0],
        // linear v.
        [tokens, HIDDEN_SIZE, KEY_VALUE_WIDTH, 0],
        // multi-token mrope q/k: token count, rope capacity, padding, padding.
        [tokens, CACHE_CAPACITY, 0, 0],
        // kv range append: token count, cache capacity, padding, padding.
        [tokens, CACHE_CAPACITY, 0, 0],
        // causal prefill gqa: tokens, query heads, key-value heads, head dim.
        [tokens, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM],
        // linear o.
        [tokens, QUERY_WIDTH, HIDDEN_SIZE, 0],
        // residual add: length, padding, padding, padding.
        [tokens * HIDDEN_SIZE, 0, 0, 0],
        // post-attention rmsnorm.
        [tokens, HIDDEN_SIZE, RMS_NORM_EPSILON.to_bits(), 0],
        // linear gate.
        [tokens, HIDDEN_SIZE, INTERMEDIATE_SIZE, 0],
        // linear up.
        [tokens, HIDDEN_SIZE, INTERMEDIATE_SIZE, 0],
        // swiglu: length, padding, padding, padding.
        [tokens * INTERMEDIATE_SIZE, 0, 0, 0],
        // linear down.
        [tokens, INTERMEDIATE_SIZE, HIDDEN_SIZE, 0],
        // second residual add.
        [tokens * HIDDEN_SIZE, 0, 0, 0],
    ]
}

fn rms_norm_stage(tokens: u32) -> InvocationPlan {
    let elements = tokens * HIDDEN_SIZE;
    InvocationPlan {
        kernel: KernelId::RmsNormF32,
        output_elements: elements as usize,
        output_bytes: u64::from(elements) * 4,
        workgroup_size: [64, 1, 1],
        dispatch: [tokens.div_ceil(64), 1, 1],
    }
}

fn projection_stage(output_width: u32, input_width: u32, tokens: u32) -> InvocationPlan {
    let elements = tokens * output_width;
    let _ = input_width;
    InvocationPlan {
        kernel: KernelId::VisionPatchProjectionF32,
        output_elements: elements as usize,
        output_bytes: u64::from(elements) * 4,
        workgroup_size: [8, 8, 1],
        dispatch: [output_width.div_ceil(32), tokens.div_ceil(32), 1],
    }
}

fn prefill_mrope_stage(tokens: u32) -> InvocationPlan {
    let elements = tokens * MROPE_WIDTH;
    InvocationPlan {
        kernel: KernelId::DecoderPrefillMropeF32,
        output_elements: elements as usize,
        output_bytes: u64::from(elements) * 4,
        workgroup_size: [64, 1, 1],
        dispatch: [elements.div_ceil(64), 1, 1],
    }
}

fn kv_append_range_stage(tokens: u32) -> InvocationPlan {
    // Both physical cache planes are the output resource, exactly as the
    // accepted single-token append plan reports them.
    let elements = 2 * CACHE_CAPACITY * KEY_VALUE_WIDTH;
    InvocationPlan {
        kernel: KernelId::DecoderKvAppendRangeF32,
        output_elements: elements as usize,
        output_bytes: u64::from(elements) * 4,
        workgroup_size: [64, 1, 1],
        dispatch: [(tokens * KEY_VALUE_WIDTH).div_ceil(64), 1, 1],
    }
}

fn prefill_gqa_stage(tokens: u32) -> InvocationPlan {
    let elements = tokens * QUERY_WIDTH;
    InvocationPlan {
        kernel: KernelId::DecoderPrefillGqaF32,
        output_elements: elements as usize,
        output_bytes: u64::from(elements) * 4,
        workgroup_size: [64, 1, 1],
        dispatch: [(tokens * QUERY_HEADS).div_ceil(64), 1, 1],
    }
}

fn add_stage(tokens: u32, width: u32) -> InvocationPlan {
    let elements = tokens * width;
    InvocationPlan {
        kernel: KernelId::AddF32,
        output_elements: elements as usize,
        output_bytes: u64::from(elements) * 4,
        workgroup_size: [64, 1, 1],
        dispatch: [elements.div_ceil(64), 1, 1],
    }
}

fn swiglu_stage(tokens: u32) -> InvocationPlan {
    let elements = tokens * INTERMEDIATE_SIZE;
    InvocationPlan {
        kernel: KernelId::DecoderSwigluF32,
        output_elements: elements as usize,
        output_bytes: u64::from(elements) * 4,
        workgroup_size: [64, 1, 1],
        dispatch: [elements.div_ceil(64), 1, 1],
    }
}

#[test]
fn descriptor_freezes_pinned_prefill_geometry_strides_and_fifteen_stage_plans() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor(3).plan().unwrap();

    assert_eq!(plan.layers, LAYERS);
    assert_eq!(plan.tokens, 3);
    assert_eq!(plan.cache_capacity, CACHE_CAPACITY);
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
        (CACHE_CAPACITY as usize * KEY_VALUE_WIDTH as usize * 4) as u64
    );
    assert_eq!(plan.hidden_stride_bytes, (HIDDEN_SIZE as usize * 4) as u64);

    assert_eq!(plan.stage_invocations, expected_stage_invocations(3));
    assert_eq!(plan.stage_uniform_words, expected_stage_uniform_words(3));

    // Double-move proves the plan is `Copy` rather than `Clone`-only.
    let cloned = plan;
    let copied = cloned;
    assert_eq!(cloned, copied);
}

#[test]
fn prefill_plan_tracks_tokens_for_representative_and_boundary_counts() {
    let fixture = Fixture::new();
    for tokens in [1, 3, 7, CACHE_CAPACITY] {
        let plan = fixture.descriptor(tokens).plan().unwrap();
        assert_eq!(plan.tokens, tokens);
        assert_eq!(plan.stage_invocations, expected_stage_invocations(tokens));
        assert_eq!(
            plan.stage_uniform_words,
            expected_stage_uniform_words(tokens)
        );
    }
}

#[test]
fn prefill_plan_matches_the_accepted_stack_plan_on_shared_statics() {
    let fixture = Fixture::new();
    let prefill = fixture.descriptor(3).plan().unwrap();
    let stack = fixture.stack_descriptor().plan().unwrap();

    assert_eq!(prefill.layers, stack.layers);
    assert_eq!(prefill.weight_stride_bytes, stack.weight_stride_bytes);
    assert_eq!(prefill.cache_stride_bytes, stack.cache_stride_bytes);
    assert_eq!(prefill.hidden_stride_bytes, stack.hidden_stride_bytes);
}

#[test]
fn single_token_prefill_is_consistent_with_the_accepted_decode_step() {
    let fixture = Fixture::new();
    let prefill = fixture.descriptor(1).plan().unwrap();
    let hidden = finite_values(HIDDEN_SIZE as usize, 0.053);
    let decode = fixture
        .stack_descriptor()
        .plan()
        .unwrap()
        .plan_step(
            0,
            &DecoderStackStep {
                hidden_row: &hidden,
            },
        )
        .unwrap();

    let prefill_words = prefill.stage_uniform_words;
    let decode_words = decode.stage_uniform_words;

    // Stages with a token-count/length ABI keep the exact decode words at
    // one token: both rmsnorms, both residuals, and swiglu.
    assert_eq!(prefill_words[0], decode_words[0]);
    assert_eq!(prefill_words[8], decode_words[6]);
    assert_eq!(prefill_words[9], decode_words[7]);
    assert_eq!(prefill_words[12], decode_words[10]);
    assert_eq!(prefill_words[14], decode_words[12]);

    // The multi-token mrope replaces the decode position with the token
    // count; the rope-capacity word is unchanged.
    assert_eq!(prefill_words[4][0], 1);
    assert_eq!(prefill_words[4][1..], decode_words[4][1..]);

    // The seven projections switch from the single-row gemv ABI
    // ([rows, columns, 0, 0]) to the accepted multi-token
    // vision_patch_projection ABI ([patches, input width, output width, 0]);
    // the encoded widths correspond pairwise.
    for (prefill_index, decode_index) in
        [(1, 1), (2, 2), (3, 3), (7, 5), (10, 8), (11, 9), (13, 11)]
    {
        assert_eq!(prefill_words[prefill_index][0], 1);
        assert_eq!(
            prefill_words[prefill_index][1], decode_words[decode_index][1],
            "stage {prefill_index} input width"
        );
        assert_eq!(
            prefill_words[prefill_index][2], decode_words[decode_index][0],
            "stage {prefill_index} output width"
        );
        assert_eq!(prefill_words[prefill_index][3], 0);
    }

    // The token-count kernels admit exactly one work row at one token.
    assert_eq!(prefill_words[5], [1, CACHE_CAPACITY, 0, 0]);
    assert_eq!(
        prefill_words[6],
        [1, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM]
    );
}

#[test]
fn plan_depends_on_geometry_and_tokens_not_operand_values() {
    let fixture = Fixture::new();
    let same = fixture.descriptor(3).plan().unwrap();
    let different_values = DecoderStackPrefillDescriptor {
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
        ..fixture.descriptor(3)
    }
    .plan()
    .unwrap();
    assert_eq!(same, different_values);

    let more_tokens = fixture.descriptor(4).plan().unwrap();
    assert_ne!(same, more_tokens);
}

#[test]
fn descriptor_rejects_token_count_drift() {
    let fixture = Fixture::new();

    // The full capacity is the largest admitted prefill.
    fixture.descriptor(CACHE_CAPACITY).plan().unwrap();

    for tokens in [0, CACHE_CAPACITY + 1, CACHE_CAPACITY * 2, u32::MAX] {
        assert_descriptor_error(
            fixture.descriptor(tokens),
            InvocationErrorCode::InvalidDecoderGeometry,
        );
    }
}

#[test]
fn descriptor_rejects_layer_count_and_inherited_geometry_drift() {
    let fixture = Fixture::new();
    let base = fixture.descriptor(3);

    for layers in [0, 1, LAYERS - 1, LAYERS + 1, 36] {
        assert_descriptor_error(
            DecoderStackPrefillDescriptor { layers, ..base },
            InvocationErrorCode::InvalidDecoderGeometry,
        );
    }
    assert_descriptor_error(
        DecoderStackPrefillDescriptor {
            cache_capacity: 0,
            tokens: 0,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderStackPrefillDescriptor {
            hidden_size: HIDDEN_SIZE / 2,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderStackPrefillDescriptor {
            intermediate_size: INTERMEDIATE_SIZE / 2,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderStackPrefillDescriptor {
            query_heads: 8,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderStackPrefillDescriptor {
            key_value_heads: 4,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderStackPrefillDescriptor {
            head_dim: 64,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderStackPrefillDescriptor {
            rms_norm_epsilon: 1.0e-6,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );

    // Inherited geometry drift outranks a bulk length violation.
    let short_gate = &fixture.gate_weight[..fixture.gate_weight.len() - 1];
    assert_descriptor_error(
        DecoderStackPrefillDescriptor {
            intermediate_size: INTERMEDIATE_SIZE / 2,
            gate_weight: short_gate,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
}

#[test]
fn descriptor_rejects_bulk_weight_length_drift() {
    let fixture = Fixture::new();
    let base = fixture.descriptor(3);

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
                0 => DecoderStackPrefillDescriptor {
                    norm1_weight: target,
                    ..base
                },
                1 => DecoderStackPrefillDescriptor {
                    q_weight: target,
                    ..base
                },
                2 => DecoderStackPrefillDescriptor {
                    k_weight: target,
                    ..base
                },
                3 => DecoderStackPrefillDescriptor {
                    v_weight: target,
                    ..base
                },
                4 => DecoderStackPrefillDescriptor {
                    o_weight: target,
                    ..base
                },
                5 => DecoderStackPrefillDescriptor {
                    norm2_weight: target,
                    ..base
                },
                6 => DecoderStackPrefillDescriptor {
                    gate_weight: target,
                    ..base
                },
                7 => DecoderStackPrefillDescriptor {
                    up_weight: target,
                    ..base
                },
                _ => DecoderStackPrefillDescriptor {
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
            DecoderStackPrefillDescriptor {
                mrope_cos: cos,
                mrope_sin: sin,
                ..base
            },
            InvocationErrorCode::LengthMismatch,
        );
    }
}

#[test]
fn descriptor_rejects_nonfinite_bulk_operands() {
    let fixture = Fixture::new();
    let base = fixture.descriptor(3);
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
        // Positions: the first element, an element deep inside a middle
        // layer's slice, and the final element of the whole bulk.
        for position in [0, length - (length / LAYERS as usize) / 2, length - 1] {
            for poison in poison_values {
                let mut target = finite_values(*length, scales[field_index]);
                target[position] = poison;
                let descriptor = match field_index {
                    0 => DecoderStackPrefillDescriptor {
                        norm1_weight: &target,
                        ..base
                    },
                    1 => DecoderStackPrefillDescriptor {
                        q_weight: &target,
                        ..base
                    },
                    2 => DecoderStackPrefillDescriptor {
                        k_weight: &target,
                        ..base
                    },
                    3 => DecoderStackPrefillDescriptor {
                        v_weight: &target,
                        ..base
                    },
                    4 => DecoderStackPrefillDescriptor {
                        o_weight: &target,
                        ..base
                    },
                    5 => DecoderStackPrefillDescriptor {
                        norm2_weight: &target,
                        ..base
                    },
                    6 => DecoderStackPrefillDescriptor {
                        gate_weight: &target,
                        ..base
                    },
                    7 => DecoderStackPrefillDescriptor {
                        up_weight: &target,
                        ..base
                    },
                    _ => DecoderStackPrefillDescriptor {
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
                    DecoderStackPrefillDescriptor {
                        mrope_cos: &target,
                        ..base
                    }
                } else {
                    DecoderStackPrefillDescriptor {
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
fn descriptor_checks_tokens_before_bulk_lengths_and_lengths_before_finiteness() {
    let fixture = Fixture::new();
    let base = fixture.descriptor(3);
    let short_gate = &fixture.gate_weight[..fixture.gate_weight.len() - 1];

    // A token-count violation outranks a bulk length violation.
    assert_descriptor_error(
        DecoderStackPrefillDescriptor {
            tokens: 0,
            gate_weight: short_gate,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
    assert_descriptor_error(
        DecoderStackPrefillDescriptor {
            tokens: CACHE_CAPACITY + 1,
            gate_weight: short_gate,
            ..base
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );

    // A bulk length violation outranks a bulk finiteness violation.
    let mut short_poisoned_gate = fixture.gate_weight[..fixture.gate_weight.len() - 1].to_vec();
    short_poisoned_gate[0] = f32::NAN;
    assert_descriptor_error(
        DecoderStackPrefillDescriptor {
            gate_weight: &short_poisoned_gate,
            ..base
        },
        InvocationErrorCode::LengthMismatch,
    );
}

#[test]
fn plan_does_not_mutate_descriptor_inputs() {
    let fixture = Fixture::new();
    let fingerprints_before = [
        fingerprint(&fixture.norm1_weight),
        fingerprint(&fixture.q_weight),
        fingerprint(&fixture.k_weight),
        fingerprint(&fixture.v_weight),
        fingerprint(&fixture.o_weight),
        fingerprint(&fixture.mrope_cos),
        fingerprint(&fixture.mrope_sin),
        fingerprint(&fixture.norm2_weight),
        fingerprint(&fixture.gate_weight),
        fingerprint(&fixture.up_weight),
        fingerprint(&fixture.down_weight),
    ];

    let first = fixture.descriptor(3).plan().unwrap();
    let second = fixture.descriptor(3).plan().unwrap();
    assert_eq!(first, second);

    let fingerprints_after = [
        fingerprint(&fixture.norm1_weight),
        fingerprint(&fixture.q_weight),
        fingerprint(&fixture.k_weight),
        fingerprint(&fixture.v_weight),
        fingerprint(&fixture.o_weight),
        fingerprint(&fixture.mrope_cos),
        fingerprint(&fixture.mrope_sin),
        fingerprint(&fixture.norm2_weight),
        fingerprint(&fixture.gate_weight),
        fingerprint(&fixture.up_weight),
        fingerprint(&fixture.down_weight),
    ];
    assert_eq!(fingerprints_before, fingerprints_after);
}

fn fingerprint(values: &[f32]) -> (usize, f64) {
    let sum: f64 = values
        .iter()
        .step_by(4096)
        .map(|value| f64::from(*value))
        .sum();
    (values.len(), sum)
}
