use pvlc_runtime_core::{
    DecoderCachedGqaInvocation, DecoderCachedGqaStage, InvocationErrorCode, KernelId,
    MAX_DECODER_HEAD_DIM,
};

const QUERY_HEADS: u32 = 16;
const KEY_VALUE_HEADS: u32 = 2;
const HEAD_DIM: u32 = 128;
const PREFIX_TOKENS: u32 = 3;
const CACHE_CAPACITY: u32 = 8;

#[derive(Clone)]
struct Fixture {
    query: Vec<f32>,
    appended_key: Vec<f32>,
    appended_value: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
        let query_width = (QUERY_HEADS * HEAD_DIM) as usize;
        let key_value_width = (KEY_VALUE_HEADS * HEAD_DIM) as usize;
        let cache_elements = CACHE_CAPACITY as usize * key_value_width;
        Self {
            query: finite_values(query_width, 0.013),
            appended_key: finite_values(key_value_width, 0.017),
            appended_value: finite_values(key_value_width, 0.019),
            key_cache: finite_values(cache_elements, 0.023),
            value_cache: finite_values(cache_elements, 0.029),
        }
    }

    fn invocation(&self) -> DecoderCachedGqaInvocation<'_> {
        DecoderCachedGqaInvocation {
            query_heads: QUERY_HEADS,
            key_value_heads: KEY_VALUE_HEADS,
            head_dim: HEAD_DIM,
            prefix_tokens: PREFIX_TOKENS,
            cache_capacity: CACHE_CAPACITY,
            query: &self.query,
            appended_key: &self.appended_key,
            appended_value: &self.appended_value,
            key_cache: &self.key_cache,
            value_cache: &self.value_cache,
        }
    }
}

fn finite_values(length: usize, scale: f32) -> Vec<f32> {
    (0..length)
        .map(|index| ((index * 17 + 5) as f32 * scale).sin())
        .collect()
}

fn assert_error(invocation: DecoderCachedGqaInvocation<'_>, expected: InvocationErrorCode) {
    let error = invocation.plan().unwrap_err();
    assert_eq!(error.code(), expected, "{error}");
}

#[test]
fn pinned_plan_freezes_direct_gqa_geometry_physical_cache_and_ordered_dispatches() {
    let fixture = Fixture::new();
    let plan = fixture.invocation().plan().unwrap();

    let query_elements = (QUERY_HEADS * HEAD_DIM) as usize;
    let key_value_width = (KEY_VALUE_HEADS * HEAD_DIM) as usize;
    let cache_elements = CACHE_CAPACITY as usize * key_value_width;
    let valid_cache_elements = (PREFIX_TOKENS + 1) as usize * key_value_width;

    assert_eq!(MAX_DECODER_HEAD_DIM, HEAD_DIM);
    assert_eq!(plan.cache_tokens_after_append, PREFIX_TOKENS + 1);
    assert_eq!(plan.query_elements, query_elements);
    assert_eq!(plan.key_value_width, key_value_width);
    assert_eq!(plan.cache_elements, cache_elements);
    assert_eq!(plan.valid_cache_elements, valid_cache_elements);
    assert_eq!(plan.cache_bytes, (cache_elements * size_of::<f32>()) as u64);
    assert_eq!(
        plan.attention_bytes,
        (query_elements * size_of::<f32>()) as u64
    );

    assert_eq!(plan.append.stage, DecoderCachedGqaStage::AppendKeyValue);
    assert_eq!(plan.append.invocation.kernel, KernelId::DecoderKvAppendF32);
    assert_eq!(plan.append.invocation.workgroup_size, [64, 1, 1]);
    assert_eq!(plan.append.invocation.dispatch, [4, 1, 1]);
    assert_eq!(plan.append.invocation.output_elements, cache_elements * 2);
    assert_eq!(plan.append.invocation.output_bytes, plan.cache_bytes * 2);
    assert_eq!(
        plan.append.uniform_words,
        [PREFIX_TOKENS, KEY_VALUE_HEADS, HEAD_DIM, CACHE_CAPACITY]
    );

    assert_eq!(plan.attention.stage, DecoderCachedGqaStage::DirectGqa);
    assert_eq!(plan.attention.invocation.kernel, KernelId::DecoderGqaF32);
    assert_eq!(plan.attention.invocation.workgroup_size, [64, 1, 1]);
    assert_eq!(plan.attention.invocation.dispatch, [1, 1, 1]);
    assert_eq!(plan.attention.invocation.output_elements, query_elements);
    assert_eq!(plan.attention.invocation.output_bytes, plan.attention_bytes);
    assert_eq!(
        plan.attention.uniform_words,
        [PREFIX_TOKENS + 1, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM]
    );

    assert_eq!(
        [plan.append.stage, plan.attention.stage,],
        [
            DecoderCachedGqaStage::AppendKeyValue,
            DecoderCachedGqaStage::DirectGqa,
        ]
    );
    assert_eq!(
        plan.cache_elements,
        CACHE_CAPACITY as usize * KEY_VALUE_HEADS as usize * HEAD_DIM as usize,
        "the physical cache must use KV heads, not an expanded query-head repeat"
    );
}

#[test]
fn geometry_rejects_zero_dimensions_invalid_grouping_and_missing_append_capacity() {
    let fixture = Fixture::new();
    let base = fixture.invocation();

    for invocation in [
        DecoderCachedGqaInvocation {
            query_heads: 0,
            ..base
        },
        DecoderCachedGqaInvocation {
            key_value_heads: 0,
            ..base
        },
        DecoderCachedGqaInvocation {
            head_dim: 0,
            ..base
        },
        DecoderCachedGqaInvocation {
            prefix_tokens: 0,
            ..base
        },
        DecoderCachedGqaInvocation {
            cache_capacity: 0,
            ..base
        },
    ] {
        assert_error(invocation, InvocationErrorCode::ZeroDimension);
    }

    for invocation in [
        DecoderCachedGqaInvocation {
            query_heads: 15,
            ..base
        },
        DecoderCachedGqaInvocation {
            query_heads: 2,
            key_value_heads: 4,
            ..base
        },
        DecoderCachedGqaInvocation {
            prefix_tokens: CACHE_CAPACITY,
            ..base
        },
        DecoderCachedGqaInvocation {
            prefix_tokens: CACHE_CAPACITY + 1,
            ..base
        },
    ] {
        assert_error(invocation, InvocationErrorCode::InvalidDecoderGeometry);
    }

    for invocation in [
        DecoderCachedGqaInvocation {
            query_heads: 8,
            ..base
        },
        DecoderCachedGqaInvocation {
            key_value_heads: 1,
            ..base
        },
        DecoderCachedGqaInvocation {
            head_dim: 64,
            ..base
        },
    ] {
        assert_error(invocation, InvocationErrorCode::InvalidDecoderGeometry);
    }

    assert_error(
        DecoderCachedGqaInvocation {
            head_dim: MAX_DECODER_HEAD_DIM + 1,
            ..base
        },
        InvocationErrorCode::UnsupportedHeadDimension,
    );
}

#[test]
fn every_semantic_operand_and_physical_cache_length_is_checked_exactly() {
    let fixture = Fixture::new();

    for operand in 0..5 {
        let original = match operand {
            0 => &fixture.query,
            1 => &fixture.appended_key,
            2 => &fixture.appended_value,
            3 => &fixture.key_cache,
            4 => &fixture.value_cache,
            _ => unreachable!(),
        };
        for values in wrong_lengths(original) {
            let invocation = match operand {
                0 => DecoderCachedGqaInvocation {
                    query: &values,
                    ..fixture.invocation()
                },
                1 => DecoderCachedGqaInvocation {
                    appended_key: &values,
                    ..fixture.invocation()
                },
                2 => DecoderCachedGqaInvocation {
                    appended_value: &values,
                    ..fixture.invocation()
                },
                3 => DecoderCachedGqaInvocation {
                    key_cache: &values,
                    ..fixture.invocation()
                },
                4 => DecoderCachedGqaInvocation {
                    value_cache: &values,
                    ..fixture.invocation()
                },
                _ => unreachable!(),
            };
            assert_error(invocation, InvocationErrorCode::LengthMismatch);
        }
    }
}

fn wrong_lengths(values: &[f32]) -> [Vec<f32>; 2] {
    let mut short = values.to_vec();
    short.pop();
    let mut long = values.to_vec();
    long.push(0.0);
    [short, long]
}

#[test]
fn only_semantically_read_cache_prefix_must_be_finite_and_poison_tail_is_admitted() {
    let fixture = Fixture::new();
    let valid_prefix_elements =
        PREFIX_TOKENS as usize * KEY_VALUE_HEADS as usize * HEAD_DIM as usize;

    for operand in 0..3 {
        let mut query = fixture.query.clone();
        let mut appended_key = fixture.appended_key.clone();
        let mut appended_value = fixture.appended_value.clone();
        match operand {
            0 => query[7] = f32::NAN,
            1 => appended_key[9] = f32::INFINITY,
            2 => appended_value[11] = f32::NEG_INFINITY,
            _ => unreachable!(),
        }
        assert_error(
            DecoderCachedGqaInvocation {
                query: &query,
                appended_key: &appended_key,
                appended_value: &appended_value,
                ..fixture.invocation()
            },
            InvocationErrorCode::NonFiniteInput,
        );
    }

    for poison_key in [true, false] {
        let mut key_cache = fixture.key_cache.clone();
        let mut value_cache = fixture.value_cache.clone();
        let cache = if poison_key {
            &mut key_cache
        } else {
            &mut value_cache
        };
        cache[valid_prefix_elements - 1] = f32::NAN;
        assert_error(
            DecoderCachedGqaInvocation {
                key_cache: &key_cache,
                value_cache: &value_cache,
                ..fixture.invocation()
            },
            InvocationErrorCode::NonFiniteInput,
        );
    }

    let mut key_cache = fixture.key_cache.clone();
    let mut value_cache = fixture.value_cache.clone();
    key_cache[valid_prefix_elements..].fill(f32::from_bits(0x7fc0_51a7));
    value_cache[valid_prefix_elements..].fill(f32::NEG_INFINITY);
    let plan = DecoderCachedGqaInvocation {
        key_cache: &key_cache,
        value_cache: &value_cache,
        ..fixture.invocation()
    }
    .plan()
    .unwrap();
    assert_eq!(
        plan.valid_cache_elements,
        (PREFIX_TOKENS + 1) as usize * KEY_VALUE_HEADS as usize * HEAD_DIM as usize
    );
}

#[test]
fn shader_address_space_overflow_is_rejected_before_operand_lengths() {
    let fixture = Fixture::new();

    assert_error(
        DecoderCachedGqaInvocation {
            query_heads: u32::MAX,
            key_value_heads: 1,
            head_dim: MAX_DECODER_HEAD_DIM,
            query: &[],
            ..fixture.invocation()
        },
        InvocationErrorCode::ArithmeticOverflow,
    );
    assert_error(
        DecoderCachedGqaInvocation {
            query_heads: 1,
            key_value_heads: 1,
            cache_capacity: u32::MAX,
            key_cache: &[],
            value_cache: &[],
            ..fixture.invocation()
        },
        InvocationErrorCode::ArithmeticOverflow,
    );
}
