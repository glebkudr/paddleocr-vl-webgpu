use pvlc_cpu_ref::{
    CpuRefError, CpuRefErrorCode, PINNED_DECODER_VOCAB_SIZE, PrefillLmHeadConfig,
    pinned_prefill_last_logits_f32, prefill_last_logits_f32, top_k,
};

const TOKENS: usize = 3;
const HIDDEN_SIZE: usize = 4;
const VOCAB_SIZE: usize = 5;
const PINNED_HIDDEN_SIZE: usize = 1_024;

const FINAL_NORM: [f32; TOKENS * HIDDEN_SIZE] = [
    0.125,
    -1.75,
    2.5,
    0.333_333_34,
    9.25,
    -0.5,
    1.125,
    -3.75,
    0.2,
    -1.3,
    2.7,
    0.0625,
];

// Rows are vocabulary outputs; columns are hidden channels.
const LM_HEAD_WEIGHT: [f32; VOCAB_SIZE * HIDDEN_SIZE] = [
    0.7, -1.1, 0.3, 2.0, -0.4, 0.9, -1.7, 0.125, 1.2, 0.25, -0.75, 0.6, -2.0, 1.5, 0.5, -0.2, 0.05,
    -0.08, 0.13, -0.21,
];

const EXPECTED_BITS: [u32; VOCAB_SIZE] = [
    0x4020_51ec,
    0xc0ba_a148,
    0xc004_a3d8,
    0xbf81_9999,
    0x3ee7_5c28,
];
const EXPECTED_BLAKE3: &str = "330604b81e09bab80b697069dbae035d946b5b29c05afebf011a73052afdd6c7";

fn config(tokens: usize, hidden_size: usize, vocab_size: usize) -> PrefillLmHeadConfig {
    PrefillLmHeadConfig {
        tokens,
        hidden_size,
        vocab_size,
    }
}

fn compact_config() -> PrefillLmHeadConfig {
    config(TOKENS, HIDDEN_SIZE, VOCAB_SIZE)
}

fn independent_last_row_dot(
    final_norm: &[f32],
    config: PrefillLmHeadConfig,
    lm_head_weight: &[f32],
) -> Vec<f32> {
    let last_row_start = (config.tokens - 1) * config.hidden_size;
    let last_row = &final_norm[last_row_start..last_row_start + config.hidden_size];
    let mut logits = Vec::with_capacity(config.vocab_size);
    for token_id in 0..config.vocab_size {
        let row_start = token_id * config.hidden_size;
        let weight_row = &lm_head_weight[row_start..row_start + config.hidden_size];
        let mut accumulator = 0.0_f32;
        for hidden_index in 0..config.hidden_size {
            accumulator += last_row[hidden_index] * weight_row[hidden_index];
        }
        logits.push(accumulator);
    }
    logits
}

fn hash_f32(values: &[f32]) -> String {
    let mut hasher = blake3::Hasher::new();
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn assert_error(result: Result<Vec<f32>, CpuRefError>, expected: CpuRefErrorCode) {
    assert_eq!(result.unwrap_err().code(), expected);
}

#[test]
fn compact_logits_match_an_independent_literal_oracle_and_preserve_operands() {
    let expected = independent_last_row_dot(&FINAL_NORM, compact_config(), &LM_HEAD_WEIGHT);
    assert_eq!(bits(&expected), EXPECTED_BITS);
    assert_eq!(hash_f32(&expected), EXPECTED_BLAKE3);

    let preserved_final_norm = FINAL_NORM;
    let preserved_weight = LM_HEAD_WEIGHT;
    let actual = prefill_last_logits_f32(&FINAL_NORM, compact_config(), &LM_HEAD_WEIGHT).unwrap();

    assert_eq!(
        actual.len(),
        VOCAB_SIZE,
        "only one vocabulary row is returned"
    );
    assert_eq!(bits(&actual), EXPECTED_BITS);
    assert_eq!(hash_f32(&actual), EXPECTED_BLAKE3);
    assert_eq!(FINAL_NORM, preserved_final_norm);
    assert_eq!(LM_HEAD_WEIGHT, preserved_weight);
}

#[test]
fn only_the_last_prefill_row_contributes_to_the_single_logit_row() {
    let baseline = prefill_last_logits_f32(&FINAL_NORM, compact_config(), &LM_HEAD_WEIGHT).unwrap();

    let mut poisoned_earlier_rows = FINAL_NORM;
    for (index, value) in poisoned_earlier_rows[..(TOKENS - 1) * HIDDEN_SIZE]
        .iter_mut()
        .enumerate()
    {
        *value = if index % 2 == 0 { f32::MAX } else { -f32::MAX };
    }
    let with_earlier_poison =
        prefill_last_logits_f32(&poisoned_earlier_rows, compact_config(), &LM_HEAD_WEIGHT).unwrap();
    assert_eq!(with_earlier_poison, baseline);

    let mut changed_last_row = FINAL_NORM;
    changed_last_row[(TOKENS - 1) * HIDDEN_SIZE + 1] += 0.5;
    let changed =
        prefill_last_logits_f32(&changed_last_row, compact_config(), &LM_HEAD_WEIGHT).unwrap();
    assert_ne!(changed, baseline);
    assert_eq!(
        changed,
        independent_last_row_dot(&changed_last_row, compact_config(), &LM_HEAD_WEIGHT)
    );
}

#[test]
fn lm_head_is_bias_free_and_uses_output_major_weight_rows_without_transpose() {
    let final_norm = [91.0_f32, 92.0, 2.0, 3.0];
    let weight = [1.0_f32, 10.0, 100.0, 1_000.0, -2.0, 4.0];
    let actual = prefill_last_logits_f32(&final_norm, config(2, 2, 3), &weight).unwrap();

    assert_eq!(actual, [32.0, 3_200.0, 8.0]);
    assert_ne!(
        actual,
        [3_002.0, 14.0, 212.0],
        "hidden-major transpose accepted"
    );
    assert_eq!(actual[0], 2.0 * 1.0 + 3.0 * 10.0, "unexpected bias term");
}

#[test]
fn produced_ties_use_the_existing_smaller_token_id_rule() {
    let logits =
        prefill_last_logits_f32(&[999.0_f32, 2.0], config(2, 1, 3), &[3.0, 3.0, 1.0]).unwrap();
    assert_eq!(logits, [6.0, 6.0, 2.0]);
    let entries = top_k(&logits, 2).unwrap();
    assert_eq!(
        entries.iter().map(|entry| entry.index).collect::<Vec<_>>(),
        [0, 1]
    );
}

#[test]
fn generic_geometry_overflow_and_exact_lengths_fail_closed() {
    for invalid in [
        config(0, HIDDEN_SIZE, VOCAB_SIZE),
        config(TOKENS, 0, VOCAB_SIZE),
        config(TOKENS, HIDDEN_SIZE, 0),
        config(usize::MAX, 2, 1),
        config(1, 2, usize::MAX),
    ] {
        assert_error(
            prefill_last_logits_f32(&[], invalid, &[]),
            CpuRefErrorCode::DimensionMismatch,
        );
    }

    let valid_final_norm = vec![0.25_f32; TOKENS * HIDDEN_SIZE];
    let valid_weight = vec![0.5_f32; VOCAB_SIZE * HIDDEN_SIZE];
    for length in [valid_final_norm.len() - 1, valid_final_norm.len() + 1] {
        let malformed = vec![0.25_f32; length];
        assert_error(
            prefill_last_logits_f32(&malformed, compact_config(), &valid_weight),
            CpuRefErrorCode::DimensionMismatch,
        );
    }
    for length in [valid_weight.len() - 1, valid_weight.len() + 1] {
        let malformed = vec![0.5_f32; length];
        assert_error(
            prefill_last_logits_f32(&valid_final_norm, compact_config(), &malformed),
            CpuRefErrorCode::DimensionMismatch,
        );
    }

    let malformed_final_norm = vec![f32::NAN; valid_final_norm.len() - 1];
    let malformed_weight = vec![f32::NAN; valid_weight.len() - 1];
    assert_error(
        prefill_last_logits_f32(&malformed_final_norm, compact_config(), &valid_weight),
        CpuRefErrorCode::DimensionMismatch,
    );
    assert_error(
        prefill_last_logits_f32(&valid_final_norm, compact_config(), &malformed_weight),
        CpuRefErrorCode::DimensionMismatch,
    );
}

#[test]
fn nonfinite_values_are_rejected_at_first_middle_and_last_positions_of_both_operands() {
    let final_norm = vec![0.25_f32; TOKENS * HIDDEN_SIZE];
    let weight = vec![0.5_f32; VOCAB_SIZE * HIDDEN_SIZE];
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for position in [0, final_norm.len() / 2, final_norm.len() - 1] {
            let mut malformed = final_norm.clone();
            malformed[position] = nonfinite;
            assert_error(
                prefill_last_logits_f32(&malformed, compact_config(), &weight),
                CpuRefErrorCode::NonFiniteInput,
            );
        }
        for position in [0, weight.len() / 2, weight.len() - 1] {
            let mut malformed = weight.clone();
            malformed[position] = nonfinite;
            assert_error(
                prefill_last_logits_f32(&final_norm, compact_config(), &malformed),
                CpuRefErrorCode::NonFiniteInput,
            );
        }
    }
}

#[test]
fn malformed_gigantic_output_geometry_is_rejected_before_materialization() {
    assert_error(
        prefill_last_logits_f32(&[1.0], config(1, 1, usize::MAX), &[]),
        CpuRefErrorCode::DimensionMismatch,
    );
}

#[test]
fn pinned_wrapper_freezes_vocab_and_hidden_geometry_and_fails_closed() {
    const _: [(); 103_424] = [(); PINNED_DECODER_VOCAB_SIZE];
    assert_eq!(PINNED_DECODER_VOCAB_SIZE, 103_424);

    assert_error(
        pinned_prefill_last_logits_f32(&[], 0, &[]),
        CpuRefErrorCode::DimensionMismatch,
    );
    assert_error(
        pinned_prefill_last_logits_f32(&[], usize::MAX, &[]),
        CpuRefErrorCode::DimensionMismatch,
    );
    assert_error(
        pinned_prefill_last_logits_f32(&vec![0.0; PINNED_HIDDEN_SIZE - 1], 1, &[]),
        CpuRefErrorCode::DimensionMismatch,
    );
    assert_error(
        pinned_prefill_last_logits_f32(&vec![0.0; PINNED_HIDDEN_SIZE], 1, &[]),
        CpuRefErrorCode::DimensionMismatch,
    );

    // Both operands are mutually length-consistent for an inferred H=1. A
    // pinned wrapper must still require H=1024 rather than accepting them.
    assert_error(
        pinned_prefill_last_logits_f32(&[0.0], 1, &vec![0.0; PINNED_DECODER_VOCAB_SIZE]),
        CpuRefErrorCode::DimensionMismatch,
    );

    // Both operands are mutually length-consistent for an inferred V=1. A
    // pinned wrapper must still require V=103424 rather than accepting them.
    assert_error(
        pinned_prefill_last_logits_f32(
            &vec![0.0; PINNED_HIDDEN_SIZE],
            1,
            &vec![0.0; PINNED_HIDDEN_SIZE],
        ),
        CpuRefErrorCode::DimensionMismatch,
    );
}
