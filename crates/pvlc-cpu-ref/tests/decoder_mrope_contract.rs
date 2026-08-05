use pvlc_cpu_ref::{CpuRefError, CpuRefErrorCode, apply_multimodal_rope_f32};

const TOKENS: usize = 3;
const QUERY_HEADS: usize = 4;
const KEY_VALUE_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const SECTIONS: [usize; 3] = [1, 1, 2];
const QUERY_LEN: usize = TOKENS * QUERY_HEADS * HEAD_DIM;
const KEY_LEN: usize = TOKENS * KEY_VALUE_HEADS * HEAD_DIM;
const RAW_LEN: usize = 3 * TOKENS * HEAD_DIM;
const LENGTHS: [usize; 4] = [QUERY_LEN, KEY_LEN, RAW_LEN, RAW_LEN];

type Inputs = [Vec<f32>; 4];
type Geometry = (usize, usize, usize, usize, [usize; 3]);
const VALID: Geometry = (TOKENS, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM, SECTIONS);

#[rustfmt::skip]
const INVALID_GEOMETRY: [(Geometry, [usize; 4]); 7] = [
    ((0, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM, SECTIONS), [0; 4]),
    ((TOKENS, 0, KEY_VALUE_HEADS, HEAD_DIM, SECTIONS), [0, KEY_LEN, RAW_LEN, RAW_LEN]),
    ((TOKENS, QUERY_HEADS, 0, HEAD_DIM, SECTIONS), [QUERY_LEN, 0, RAW_LEN, RAW_LEN]),
    ((TOKENS, QUERY_HEADS, KEY_VALUE_HEADS, 0, SECTIONS), [0; 4]),
    ((TOKENS, 3, KEY_VALUE_HEADS, HEAD_DIM, SECTIONS), [TOKENS * 3 * HEAD_DIM, KEY_LEN, RAW_LEN, RAW_LEN]),
    ((TOKENS, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM, [1, 1, 1]), LENGTHS),
    ((usize::MAX, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM, SECTIONS), [0; 4]),
];

fn fixture(
    len: usize,
    mul: usize,
    add: usize,
    modulus: usize,
    divisor: f32,
    bias: f32,
) -> Vec<f32> {
    (0..len)
        .map(|index| (((index * mul + add) % modulus) as f32 / divisor) - bias)
        .collect()
}

fn raw_fixture(cosine: bool) -> Vec<f32> {
    let mut values = Vec::with_capacity(RAW_LEN);
    for axis in 0..3 {
        for token in 0..TOKENS {
            for dim in 0..HEAD_DIM {
                values.push(if cosine {
                    0.6 + axis as f32 * 0.2 + token as f32 * 0.03 + dim as f32 * 0.01
                } else {
                    -0.4 - axis as f32 * 0.15 + token as f32 * 0.02 - dim as f32 * 0.005
                });
            }
        }
    }
    values
}

fn fixtures() -> Inputs {
    [
        fixture(QUERY_LEN, 7, 3, 29, 17.0, 0.8),
        fixture(KEY_LEN, 5, 11, 31, 19.0, 0.7),
        raw_fixture(true),
        raw_fixture(false),
    ]
}

fn select_axis_chunks(raw: &[f32], token: usize, sections: [usize; 3]) -> Vec<f32> {
    let mut selected = Vec::with_capacity(HEAD_DIM);
    let mut offset = 0;
    for (chunk, size) in sections.into_iter().cycle().take(6).enumerate() {
        let base = (chunk % 3) * TOKENS * HEAD_DIM + token * HEAD_DIM + offset;
        selected.extend_from_slice(&raw[base..base + size]);
        offset += size;
    }
    selected
}

fn independent_apply_rows(
    input: &[f32],
    heads: usize,
    raw_cos: &[f32],
    raw_sin: &[f32],
    sections: [usize; 3],
) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());
    for token in 0..TOKENS {
        let cos = select_axis_chunks(raw_cos, token, sections);
        let sin = select_axis_chunks(raw_sin, token, sections);
        for head in 0..heads {
            let start = (token * heads + head) * HEAD_DIM;
            let row = &input[start..start + HEAD_DIM];
            for dim in 0..HEAD_DIM {
                let rotated = if dim < HEAD_DIM / 2 {
                    -row[dim + HEAD_DIM / 2]
                } else {
                    row[dim - HEAD_DIM / 2]
                };
                output.push(row[dim] * cos[dim] + rotated * sin[dim]);
            }
        }
    }
    output
}

fn independent_apply(inputs: &Inputs, sections: [usize; 3]) -> (Vec<f32>, Vec<f32>) {
    (
        independent_apply_rows(&inputs[0], QUERY_HEADS, &inputs[2], &inputs[3], sections),
        independent_apply_rows(
            &inputs[1],
            KEY_VALUE_HEADS,
            &inputs[2],
            &inputs[3],
            sections,
        ),
    )
}

fn invoke(inputs: &Inputs, geometry: Geometry) -> Result<(Vec<f32>, Vec<f32>), CpuRefError> {
    let (tokens, query_heads, key_value_heads, head_dim, sections) = geometry;
    apply_multimodal_rope_f32(
        &inputs[0],
        &inputs[1],
        tokens,
        query_heads,
        key_value_heads,
        head_dim,
        &inputs[2],
        &inputs[3],
        sections,
    )
}

fn assert_error<T>(result: Result<T, CpuRefError>, expected: CpuRefErrorCode) {
    match result {
        Err(error) => assert_eq!(error.code(), expected),
        Ok(_) => panic!("expected {expected:?}"),
    }
}

fn combined_digest(query: &[f32], key: &[f32]) -> String {
    let bytes = query
        .iter()
        .chain(key)
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    blake3::hash(&bytes).to_hex().to_string()
}

#[test]
fn tiny_mrope_matches_independent_chunked_axis_loop_and_literal_blake3() {
    let inputs = fixtures();
    let expected = independent_apply(&inputs, SECTIONS);
    assert_eq!(
        combined_digest(&expected.0, &expected.1),
        "7d068d43a4663496e6e48857683f86175319a3ed7760de9340f22d114e07abf5"
    );

    let actual = invoke(&inputs, VALID).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn mrope_distinguishes_axes_and_chunk_order_without_mutating_inputs() {
    let inputs = fixtures();
    let preserved = inputs.clone();
    let expected = independent_apply(&inputs, SECTIONS);

    let mut collapsed = inputs.clone();
    collapsed[2] = inputs[2][..TOKENS * HEAD_DIM].repeat(3);
    collapsed[3] = inputs[3][..TOKENS * HEAD_DIM].repeat(3);
    let axis_zero_only = independent_apply(&collapsed, SECTIONS);
    let wrong_order = independent_apply(&inputs, [1, 2, 1]);
    assert_ne!(expected.0, axis_zero_only.0);
    assert_ne!(expected.1, axis_zero_only.1);
    assert_ne!(expected.0, wrong_order.0);
    assert_ne!(expected.1, wrong_order.1);

    let actual = invoke(&inputs, VALID).unwrap();
    assert_eq!(actual, expected);
    assert_ne!(actual, axis_zero_only);
    assert_ne!(actual, wrong_order);
    assert_eq!(inputs, preserved);
}

#[test]
fn mrope_fail_closes_for_invalid_geometry_lengths_and_nonfinite_operands() {
    for (geometry, lengths) in INVALID_GEOMETRY {
        let inputs = lengths.map(|length| vec![0.0; length]);
        assert_error(
            invoke(&inputs, geometry),
            CpuRefErrorCode::DimensionMismatch,
        );
    }

    for operand in 0..4 {
        for delta in [-1_isize, 1] {
            let mut lengths = LENGTHS;
            lengths[operand] = lengths[operand].checked_add_signed(delta).unwrap();
            let inputs = lengths.map(|length| vec![0.0; length]);
            assert_error(invoke(&inputs, VALID), CpuRefErrorCode::DimensionMismatch);
        }
    }

    let finite = fixtures();
    for operand in 0..4 {
        let len = finite[operand].len();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for index in [0, len / 2, len - 1] {
                let mut inputs = finite.clone();
                inputs[operand][index] = value;
                assert_error(invoke(&inputs, VALID), CpuRefErrorCode::NonFiniteInput);
            }
        }
    }
}
