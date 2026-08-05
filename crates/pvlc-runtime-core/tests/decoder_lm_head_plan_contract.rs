use pvlc_runtime_core::{DecoderLmHeadDescriptor, InvocationErrorCode, InvocationPlan, KernelId};

const HIDDEN_SIZE: u32 = 1024;
const VOCAB_SIZE: u32 = 103_424;
const RMS_NORM_EPSILON: f32 = 1.0e-5;

const NORM_ELEMENTS: usize = HIDDEN_SIZE as usize;
const LM_HEAD_ELEMENTS: usize = VOCAB_SIZE as usize * HIDDEN_SIZE as usize;

const LOGITS_BYTES: u64 = VOCAB_SIZE as u64 * 4;
const LM_HEAD_BYTES: u64 = LM_HEAD_ELEMENTS as u64 * 4;
const NORMED_ROW_BYTES: u64 = HIDDEN_SIZE as u64 * 4;

struct Fixture {
    final_norm_weight: Vec<f32>,
    lm_head_weight: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            final_norm_weight: finite_values(NORM_ELEMENTS, 0.011),
            lm_head_weight: finite_values(LM_HEAD_ELEMENTS, 0.013),
        }
    }

    fn descriptor(&self) -> DecoderLmHeadDescriptor<'_> {
        DecoderLmHeadDescriptor {
            hidden_size: HIDDEN_SIZE,
            vocab_size: VOCAB_SIZE,
            rms_norm_epsilon: RMS_NORM_EPSILON,
            final_norm_weight: &self.final_norm_weight,
            lm_head_weight: &self.lm_head_weight,
        }
    }
}

fn finite_values(length: usize, scale: f32) -> Vec<f32> {
    (0..length)
        .map(|index| ((index * 31 + 7) as f32 * scale).sin())
        .collect()
}

fn assert_descriptor_error(descriptor: DecoderLmHeadDescriptor<'_>, expected: InvocationErrorCode) {
    let error = descriptor.plan().unwrap_err();
    assert_eq!(error.code(), expected, "{error}");
}

// The two logits stages in chain order: final rmsnorm of the single current
// hidden row, then the bias-free output-major LM-head GEMV.
fn expected_stage_invocations() -> [InvocationPlan; 2] {
    [
        InvocationPlan {
            kernel: KernelId::RmsNormF32,
            output_elements: HIDDEN_SIZE as usize,
            output_bytes: NORMED_ROW_BYTES,
            workgroup_size: [64, 1, 1],
            dispatch: [1, 1, 1],
        },
        InvocationPlan {
            kernel: KernelId::GemvTiledF32,
            output_elements: VOCAB_SIZE as usize,
            output_bytes: LOGITS_BYTES,
            workgroup_size: [256, 1, 1],
            dispatch: [VOCAB_SIZE.div_ceil(8), 1, 1],
        },
    ]
}

#[test]
fn plan_pins_the_exact_two_stage_logits_lattice() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().expect("plan");

    assert_eq!(plan.hidden_size, HIDDEN_SIZE);
    assert_eq!(plan.vocab_size, VOCAB_SIZE);
    assert_eq!(plan.final_norm_weight_bytes, NORMED_ROW_BYTES);
    assert_eq!(plan.lm_head_weight_bytes, LM_HEAD_BYTES);
    assert_eq!(plan.normed_row_bytes, NORMED_ROW_BYTES);
    assert_eq!(plan.logits_bytes, LOGITS_BYTES);
    assert_eq!(plan.stage_invocations, expected_stage_invocations());
    assert_eq!(
        plan.stage_uniform_words,
        [
            [1, HIDDEN_SIZE, RMS_NORM_EPSILON.to_bits(), 0],
            [VOCAB_SIZE, HIDDEN_SIZE, 0, 0],
        ]
    );
}

#[test]
fn plan_is_deterministic_and_pure() {
    let fixture = Fixture::new();
    let first = fixture.descriptor().plan().expect("first plan");
    let second = fixture.descriptor().plan().expect("second plan");
    assert_eq!(first, second);
}

#[test]
fn descriptor_rejects_zero_hidden_size() {
    let fixture = Fixture::new();
    assert_descriptor_error(
        DecoderLmHeadDescriptor {
            hidden_size: 0,
            ..fixture.descriptor()
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
}

#[test]
fn descriptor_rejects_zero_vocab_size() {
    let fixture = Fixture::new();
    assert_descriptor_error(
        DecoderLmHeadDescriptor {
            vocab_size: 0,
            ..fixture.descriptor()
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
}

#[test]
fn descriptor_rejects_nonpositive_epsilon() {
    let fixture = Fixture::new();
    for epsilon in [0.0_f32, -1.0e-5, f32::NAN, f32::INFINITY] {
        assert_descriptor_error(
            DecoderLmHeadDescriptor {
                rms_norm_epsilon: epsilon,
                ..fixture.descriptor()
            },
            InvocationErrorCode::InvalidDecoderGeometry,
        );
    }
}

#[test]
fn descriptor_rejects_vocab_hidden_overflow() {
    let fixture = Fixture::new();
    assert_descriptor_error(
        DecoderLmHeadDescriptor {
            vocab_size: u32::MAX,
            ..fixture.descriptor()
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
}

#[test]
fn descriptor_rejects_final_norm_length_drift() {
    let fixture = Fixture::new();
    let overlong = finite_values(NORM_ELEMENTS + 1, 0.017);
    for drifted in [
        &fixture.final_norm_weight[..NORM_ELEMENTS - 1],
        &overlong[..],
    ] {
        assert_descriptor_error(
            DecoderLmHeadDescriptor {
                final_norm_weight: drifted,
                ..fixture.descriptor()
            },
            InvocationErrorCode::LengthMismatch,
        );
    }
}

#[test]
fn descriptor_rejects_lm_head_length_drift() {
    let fixture = Fixture::new();
    let overlong = finite_values(LM_HEAD_ELEMENTS + 1, 0.019);
    for drifted in [
        &fixture.lm_head_weight[..LM_HEAD_ELEMENTS - 1],
        &overlong[..],
    ] {
        assert_descriptor_error(
            DecoderLmHeadDescriptor {
                lm_head_weight: drifted,
                ..fixture.descriptor()
            },
            InvocationErrorCode::LengthMismatch,
        );
    }
}

#[test]
fn descriptor_rejects_nonfinite_final_norm_weight() {
    let mut fixture = Fixture::new();
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        fixture.final_norm_weight[NORM_ELEMENTS / 2] = poison;
        assert_descriptor_error(fixture.descriptor(), InvocationErrorCode::NonFiniteInput);
        fixture = Fixture::new();
    }
}

#[test]
fn descriptor_rejects_nonfinite_lm_head_weight() {
    let mut fixture = Fixture::new();
    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        fixture.lm_head_weight[LM_HEAD_ELEMENTS - 1] = poison;
        assert_descriptor_error(fixture.descriptor(), InvocationErrorCode::NonFiniteInput);
        fixture = Fixture::new();
    }
}

#[test]
fn geometry_drift_outranks_operand_length_and_finiteness() {
    // Zero geometry is reported before any bulk operand check, exactly like
    // the accepted prefill descriptor discipline.
    let mut fixture = Fixture::new();
    fixture.final_norm_weight[0] = f32::NAN;
    assert_descriptor_error(
        DecoderLmHeadDescriptor {
            hidden_size: 0,
            final_norm_weight: &fixture.final_norm_weight[..NORM_ELEMENTS - 1],
            ..fixture.descriptor()
        },
        InvocationErrorCode::InvalidDecoderGeometry,
    );
}

#[test]
fn length_drift_outranks_finiteness() {
    let mut fixture = Fixture::new();
    fixture.lm_head_weight[0] = f32::NAN;
    assert_descriptor_error(
        DecoderLmHeadDescriptor {
            lm_head_weight: &fixture.lm_head_weight[..LM_HEAD_ELEMENTS - 1],
            ..fixture.descriptor()
        },
        InvocationErrorCode::LengthMismatch,
    );
}

#[test]
fn plan_rounds_the_gemv_dispatch_for_generic_vocab() {
    // Generic non-decoder hidden widths retain the accepted serial fallback;
    // only the admitted decoder widths 1024/2048/3072 use GemvTiledF32.
    let hidden_size = 64_u32;
    let vocab_size = 100_u32;
    let final_norm_weight = finite_values(hidden_size as usize, 0.011);
    let lm_head_weight = finite_values(vocab_size as usize * hidden_size as usize, 0.013);
    let plan = DecoderLmHeadDescriptor {
        hidden_size,
        vocab_size,
        rms_norm_epsilon: RMS_NORM_EPSILON,
        final_norm_weight: &final_norm_weight,
        lm_head_weight: &lm_head_weight,
    }
    .plan()
    .expect("generic plan");
    assert_eq!(plan.stage_invocations[1].kernel, KernelId::GemvF32);
    assert_eq!(plan.stage_invocations[1].workgroup_size, [64, 1, 1]);
    assert_eq!(plan.stage_invocations[1].dispatch, [2, 1, 1]);
    assert_eq!(plan.stage_uniform_words[1], [100, 64, 0, 0]);
    assert_eq!(plan.logits_bytes, 400);
}

#[test]
fn pinned_descriptor_supplies_the_literal_geometry() {
    let fixture = Fixture::new();
    let plan = DecoderLmHeadDescriptor::pinned(&fixture.final_norm_weight, &fixture.lm_head_weight)
        .plan()
        .expect("pinned plan");
    assert_eq!(plan.hidden_size, 1024);
    assert_eq!(plan.vocab_size, 103_424);
    assert_eq!(plan.stage_invocations, expected_stage_invocations());
}

#[test]
fn pinned_descriptor_never_infers_geometry_from_operand_lengths() {
    // The M5d critic's loophole class: operands whose lengths happen to match
    // a DIFFERENT (hidden, vocab) pair must not be reinterpreted. A final
    // norm of 1023 elements and an LM head of 1023 * 103424 elements are
    // internally consistent for hidden=1023 but are not the pinned geometry.
    let fixture = Fixture::new();
    let wrong_norm = finite_values(NORM_ELEMENTS - 1, 0.017);
    let wrong_head = finite_values(LM_HEAD_ELEMENTS - HIDDEN_SIZE as usize, 0.019);
    let error = DecoderLmHeadDescriptor::pinned(&wrong_norm, &wrong_head)
        .plan()
        .unwrap_err();
    assert_eq!(error.code(), InvocationErrorCode::LengthMismatch, "{error}");
    let _ = &fixture;
}

#[test]
fn pinned_descriptor_rejects_nonfinite_operands() {
    let mut final_norm = finite_values(NORM_ELEMENTS, 0.011);
    final_norm[3] = f32::NAN;
    let lm_head = finite_values(LM_HEAD_ELEMENTS, 0.013);
    let error = DecoderLmHeadDescriptor::pinned(&final_norm, &lm_head)
        .plan()
        .unwrap_err();
    assert_eq!(error.code(), InvocationErrorCode::NonFiniteInput, "{error}");
}
