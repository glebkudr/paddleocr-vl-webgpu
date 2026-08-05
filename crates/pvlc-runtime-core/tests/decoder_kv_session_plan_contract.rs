use pvlc_runtime_core::{
    DecoderCachedGqaInvocation, DecoderCachedGqaStage, DecoderKvSessionDescriptor,
    DecoderKvSessionStep, InvocationErrorCode, KernelId,
};

const QUERY_HEADS: u32 = 16;
const KEY_VALUE_HEADS: u32 = 2;
const HEAD_DIM: u32 = 128;
const PREFIX_TOKENS: u32 = 3;
const CACHE_CAPACITY: u32 = 9;

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
        let query_elements = (QUERY_HEADS * HEAD_DIM) as usize;
        let key_value_width = (KEY_VALUE_HEADS * HEAD_DIM) as usize;
        let cache_elements = CACHE_CAPACITY as usize * key_value_width;
        Self {
            query: finite_values(query_elements, 0.013),
            appended_key: finite_values(key_value_width, 0.017),
            appended_value: finite_values(key_value_width, 0.019),
            key_cache: finite_values(cache_elements, 0.023),
            value_cache: finite_values(cache_elements, 0.029),
        }
    }

    fn descriptor(&self) -> DecoderKvSessionDescriptor<'_> {
        DecoderKvSessionDescriptor {
            query_heads: QUERY_HEADS,
            key_value_heads: KEY_VALUE_HEADS,
            head_dim: HEAD_DIM,
            prefix_tokens: PREFIX_TOKENS,
            cache_capacity: CACHE_CAPACITY,
            key_cache: &self.key_cache,
            value_cache: &self.value_cache,
        }
    }

    fn step(&self) -> DecoderKvSessionStep<'_> {
        DecoderKvSessionStep {
            query: &self.query,
            appended_key: &self.appended_key,
            appended_value: &self.appended_value,
        }
    }
}

fn finite_values(length: usize, scale: f32) -> Vec<f32> {
    (0..length)
        .map(|index| ((index * 31 + 7) as f32 * scale).sin())
        .collect()
}

fn assert_descriptor_error(
    descriptor: DecoderKvSessionDescriptor<'_>,
    expected: InvocationErrorCode,
) {
    let error = descriptor.plan().unwrap_err();
    assert_eq!(error.code(), expected, "{error}");
}

#[test]
fn descriptor_and_step_freeze_compact_pinned_geometry_and_dynamic_uniforms() {
    let fixture = Fixture::new();
    let session = fixture.descriptor().plan().unwrap();
    let step = session.plan_step(PREFIX_TOKENS, &fixture.step()).unwrap();

    let query_elements = (QUERY_HEADS * HEAD_DIM) as usize;
    let key_value_width = (KEY_VALUE_HEADS * HEAD_DIM) as usize;
    let cache_elements = CACHE_CAPACITY as usize * key_value_width;

    assert_eq!(session.initial_cache_tokens, PREFIX_TOKENS);
    assert_eq!(session.cache_capacity, CACHE_CAPACITY);
    assert_eq!(session.query_heads, QUERY_HEADS);
    assert_eq!(session.key_value_heads, KEY_VALUE_HEADS);
    assert_eq!(session.head_dim, HEAD_DIM);
    assert_eq!(session.query_elements, query_elements);
    assert_eq!(session.key_value_width, key_value_width);
    assert_eq!(session.cache_elements, cache_elements);
    assert_eq!(
        session.cache_bytes,
        (cache_elements * size_of::<f32>()) as u64
    );
    assert_eq!(
        session.attention_bytes,
        (query_elements * size_of::<f32>()) as u64
    );
    assert_eq!(
        session.append_invocation.kernel,
        KernelId::DecoderKvAppendF32
    );
    assert_eq!(session.append_invocation.workgroup_size, [64, 1, 1]);
    assert_eq!(session.append_invocation.dispatch, [4, 1, 1]);
    assert_eq!(session.attention_invocation.kernel, KernelId::DecoderGqaF32);
    assert_eq!(session.attention_invocation.workgroup_size, [64, 1, 1]);
    assert_eq!(session.attention_invocation.dispatch, [1, 1, 1]);

    assert_eq!(step.cache_tokens_before, PREFIX_TOKENS);
    assert_eq!(step.cache_tokens_after, PREFIX_TOKENS + 1);
    assert_eq!(step.append.stage, DecoderCachedGqaStage::AppendKeyValue);
    assert_eq!(
        step.append.uniform_words,
        [PREFIX_TOKENS, KEY_VALUE_HEADS, HEAD_DIM, CACHE_CAPACITY]
    );
    assert_eq!(step.attention.stage, DecoderCachedGqaStage::DirectGqa);
    assert_eq!(
        step.attention.uniform_words,
        [PREFIX_TOKENS + 1, QUERY_HEADS, KEY_VALUE_HEADS, HEAD_DIM,]
    );

    let later = session
        .plan_step(PREFIX_TOKENS + 4, &fixture.step())
        .unwrap();
    assert_eq!(later.cache_tokens_before, PREFIX_TOKENS + 4);
    assert_eq!(later.cache_tokens_after, PREFIX_TOKENS + 5);
    assert_eq!(later.append.uniform_words[0], PREFIX_TOKENS + 4);
    assert_eq!(later.attention.uniform_words[0], PREFIX_TOKENS + 5);
    assert_eq!(later.append.invocation, session.append_invocation);
    assert_eq!(later.attention.invocation, session.attention_invocation);

    // M7o2 amendment: every committed transition also carries the exact
    // split-K decode GQA plan (the persistent stack/native steps dispatch the
    // split partial/merge pair; the single-shot authority keeps the accepted
    // serial attention plan above).
    let expected_chunks = (PREFIX_TOKENS + 1).div_ceil(32);
    assert_eq!(step.split_gqa.chunk_size, 32);
    assert_eq!(step.split_gqa.chunk_count, expected_chunks);
    assert_eq!(step.split_gqa.partial_stride_f32, 192);
    assert_eq!(
        step.split_gqa.partial_invocation.kernel,
        KernelId::DecoderGqaSplitPartialF32
    );
    assert_eq!(
        step.split_gqa.partial_invocation.dispatch,
        [16 * expected_chunks, 1, 1]
    );
    assert_eq!(
        step.split_gqa.merge_invocation.kernel,
        KernelId::DecoderGqaSplitMergeF32
    );
    assert_eq!(step.split_gqa.merge_invocation.dispatch, [32, 1, 1]);
    assert_eq!(
        step.split_gqa.uniform_words,
        [
            [PREFIX_TOKENS + 1, expected_chunks, 0, 0],
            [PREFIX_TOKENS + 1, expected_chunks, 0, 0],
        ]
    );
    let later_chunks = (PREFIX_TOKENS + 5).div_ceil(32);
    assert_eq!(later.split_gqa.chunk_count, later_chunks);
    assert_eq!(
        later.split_gqa.uniform_words[0],
        [PREFIX_TOKENS + 5, later_chunks, 0, 0]
    );
}

#[test]
fn one_shot_and_persistent_planners_are_the_same_arithmetic_authority() {
    let query_elements = (QUERY_HEADS * HEAD_DIM) as usize;
    let key_value_width = (KEY_VALUE_HEADS * HEAD_DIM) as usize;
    let query = finite_values(query_elements, 0.013);
    let appended_key = finite_values(key_value_width, 0.017);
    let appended_value = finite_values(key_value_width, 0.019);
    let step_operand = DecoderKvSessionStep {
        query: &query,
        appended_key: &appended_key,
        appended_value: &appended_value,
    };

    for (initial_prefix, cache_capacity) in [(1, 2), (1, 9), (3, 9), (8, 9), (31, 33)] {
        let cache_elements = cache_capacity as usize * key_value_width;
        let key_cache = finite_values(cache_elements, 0.023);
        let value_cache = finite_values(cache_elements, 0.029);
        let session = DecoderKvSessionDescriptor {
            query_heads: QUERY_HEADS,
            key_value_heads: KEY_VALUE_HEADS,
            head_dim: HEAD_DIM,
            prefix_tokens: initial_prefix,
            cache_capacity,
            key_cache: &key_cache,
            value_cache: &value_cache,
        }
        .plan()
        .unwrap();

        let mut valid_prefixes = vec![initial_prefix, cache_capacity - 1];
        if initial_prefix + 1 < cache_capacity {
            valid_prefixes.push(initial_prefix + 1);
        }
        valid_prefixes.sort_unstable();
        valid_prefixes.dedup();
        for prefix_tokens in valid_prefixes {
            let persistent = session.plan_step(prefix_tokens, &step_operand).unwrap();
            let one_shot = DecoderCachedGqaInvocation {
                query_heads: QUERY_HEADS,
                key_value_heads: KEY_VALUE_HEADS,
                head_dim: HEAD_DIM,
                prefix_tokens,
                cache_capacity,
                query: &query,
                appended_key: &appended_key,
                appended_value: &appended_value,
                key_cache: &key_cache,
                value_cache: &value_cache,
            }
            .plan()
            .unwrap();

            assert_eq!(session.query_elements, one_shot.query_elements);
            assert_eq!(session.key_value_width, one_shot.key_value_width);
            assert_eq!(session.cache_elements, one_shot.cache_elements);
            assert_eq!(session.cache_bytes, one_shot.cache_bytes);
            assert_eq!(session.attention_bytes, one_shot.attention_bytes);
            assert_eq!(
                persistent.cache_tokens_after,
                one_shot.cache_tokens_after_append
            );
            assert_eq!(persistent.append, one_shot.append);
            assert_eq!(persistent.attention, one_shot.attention);
        }

        for invalid_prefix in [cache_capacity, cache_capacity + 1] {
            let persistent_error = session
                .plan_step(invalid_prefix, &step_operand)
                .unwrap_err();
            let one_shot_error = DecoderCachedGqaInvocation {
                query_heads: QUERY_HEADS,
                key_value_heads: KEY_VALUE_HEADS,
                head_dim: HEAD_DIM,
                prefix_tokens: invalid_prefix,
                cache_capacity,
                query: &query,
                appended_key: &appended_key,
                appended_value: &appended_value,
                key_cache: &key_cache,
                value_cache: &value_cache,
            }
            .plan()
            .unwrap_err();
            assert_eq!(persistent_error.code(), one_shot_error.code());
            assert_eq!(
                persistent_error.code(),
                InvocationErrorCode::InvalidDecoderGeometry
            );
        }
    }
}

#[test]
fn descriptor_rejects_every_geometry_length_overflow_and_semantic_poison_class() {
    let fixture = Fixture::new();
    let base = fixture.descriptor();

    for descriptor in [
        DecoderKvSessionDescriptor {
            query_heads: 0,
            ..base
        },
        DecoderKvSessionDescriptor {
            key_value_heads: 0,
            ..base
        },
        DecoderKvSessionDescriptor {
            head_dim: 0,
            ..base
        },
        DecoderKvSessionDescriptor {
            prefix_tokens: 0,
            ..base
        },
        DecoderKvSessionDescriptor {
            cache_capacity: 0,
            ..base
        },
    ] {
        assert_descriptor_error(descriptor, InvocationErrorCode::ZeroDimension);
    }

    for descriptor in [
        DecoderKvSessionDescriptor {
            query_heads: 8,
            ..base
        },
        DecoderKvSessionDescriptor {
            key_value_heads: 1,
            ..base
        },
        DecoderKvSessionDescriptor {
            head_dim: 64,
            ..base
        },
        DecoderKvSessionDescriptor {
            prefix_tokens: CACHE_CAPACITY,
            ..base
        },
        DecoderKvSessionDescriptor {
            prefix_tokens: CACHE_CAPACITY + 1,
            ..base
        },
    ] {
        assert_descriptor_error(descriptor, InvocationErrorCode::InvalidDecoderGeometry);
    }
    assert_descriptor_error(
        DecoderKvSessionDescriptor {
            head_dim: HEAD_DIM + 1,
            ..base
        },
        InvocationErrorCode::UnsupportedHeadDimension,
    );
    assert_descriptor_error(
        DecoderKvSessionDescriptor {
            cache_capacity: u32::MAX,
            key_cache: &[],
            value_cache: &[],
            ..base
        },
        InvocationErrorCode::ArithmeticOverflow,
    );

    for cache_operand in 0..2 {
        let original = if cache_operand == 0 {
            &fixture.key_cache
        } else {
            &fixture.value_cache
        };
        for values in wrong_lengths(original) {
            let descriptor = if cache_operand == 0 {
                DecoderKvSessionDescriptor {
                    key_cache: &values,
                    ..base
                }
            } else {
                DecoderKvSessionDescriptor {
                    value_cache: &values,
                    ..base
                }
            };
            assert_descriptor_error(descriptor, InvocationErrorCode::LengthMismatch);
        }
    }

    let semantic_elements = PREFIX_TOKENS as usize * KEY_VALUE_HEADS as usize * HEAD_DIM as usize;
    for key_operand in [true, false] {
        let mut keys = fixture.key_cache.clone();
        let mut values = fixture.value_cache.clone();
        if key_operand {
            keys[semantic_elements - 1] = f32::NAN;
        } else {
            values[semantic_elements - 1] = f32::INFINITY;
        }
        assert_descriptor_error(
            DecoderKvSessionDescriptor {
                key_cache: &keys,
                value_cache: &values,
                ..base
            },
            InvocationErrorCode::NonFiniteInput,
        );
    }
}

#[test]
fn unused_cache_capacity_is_not_semantic_but_rollback_and_exhaustion_are_rejected() {
    let fixture = Fixture::new();
    let semantic_elements = PREFIX_TOKENS as usize * KEY_VALUE_HEADS as usize * HEAD_DIM as usize;
    let mut keys = fixture.key_cache.clone();
    let mut values = fixture.value_cache.clone();
    keys[semantic_elements..].fill(f32::NAN);
    values[semantic_elements..].fill(f32::from_bits(0x7fc0_51a7));
    let descriptor = DecoderKvSessionDescriptor {
        key_cache: &keys,
        value_cache: &values,
        ..fixture.descriptor()
    };
    let plan = descriptor.plan().unwrap();

    let rollback = plan
        .plan_step(PREFIX_TOKENS - 1, &fixture.step())
        .unwrap_err();
    assert_eq!(rollback.code(), InvocationErrorCode::InvalidDecoderGeometry);
    let exhausted = plan.plan_step(CACHE_CAPACITY, &fixture.step()).unwrap_err();
    assert_eq!(
        exhausted.code(),
        InvocationErrorCode::InvalidDecoderGeometry
    );
    let beyond = plan
        .plan_step(CACHE_CAPACITY + 1, &fixture.step())
        .unwrap_err();
    assert_eq!(beyond.code(), InvocationErrorCode::InvalidDecoderGeometry);
}

#[test]
fn every_step_operand_is_exact_and_finite_before_a_dynamic_plan_exists() {
    let fixture = Fixture::new();
    let plan = fixture.descriptor().plan().unwrap();
    let base = fixture.step();

    for operand in 0..3 {
        let original = match operand {
            0 => &fixture.query,
            1 => &fixture.appended_key,
            2 => &fixture.appended_value,
            _ => unreachable!(),
        };
        for values in wrong_lengths(original) {
            let step = match operand {
                0 => DecoderKvSessionStep {
                    query: &values,
                    ..base
                },
                1 => DecoderKvSessionStep {
                    appended_key: &values,
                    ..base
                },
                2 => DecoderKvSessionStep {
                    appended_value: &values,
                    ..base
                },
                _ => unreachable!(),
            };
            let error = plan.plan_step(PREFIX_TOKENS, &step).unwrap_err();
            assert_eq!(error.code(), InvocationErrorCode::LengthMismatch);
        }
    }

    for operand in 0..3 {
        let mut query = fixture.query.clone();
        let mut key = fixture.appended_key.clone();
        let mut value = fixture.appended_value.clone();
        match operand {
            0 => query[7] = f32::NAN,
            1 => key[11] = f32::INFINITY,
            2 => value[13] = f32::NEG_INFINITY,
            _ => unreachable!(),
        }
        let error = plan
            .plan_step(
                PREFIX_TOKENS,
                &DecoderKvSessionStep {
                    query: &query,
                    appended_key: &key,
                    appended_value: &value,
                },
            )
            .unwrap_err();
        assert_eq!(error.code(), InvocationErrorCode::NonFiniteInput);
    }
}

fn wrong_lengths(values: &[f32]) -> [Vec<f32>; 2] {
    let mut short = values.to_vec();
    short.pop();
    let mut long = values.to_vec();
    long.push(0.0);
    [short, long]
}
