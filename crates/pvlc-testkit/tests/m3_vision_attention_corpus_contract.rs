use pvlc_cpu_ref::materialized_segmented_attention_f32;
use pvlc_runtime_core::{KernelId, KernelInvocation};
use pvlc_testkit::{
    M3_VISION_ATTENTION_SEQUENCE_LENGTHS, M3VisionAttentionCorpus, PADDLEOCR_VL_VISION_HEAD_DIM,
    PADDLEOCR_VL_VISION_HEADS, VISION_ATTENTION_FIXTURE_ALGORITHM, m3_vision_attention_corpus,
};

const REQUIRED_CASE_IDS: [&str; 11] = [
    "vision_attention_f32/single-s0008",
    "vision_attention_f32/single-s0016",
    "vision_attention_f32/single-s0031",
    "vision_attention_f32/single-s0064",
    "vision_attention_f32/single-s0127",
    "vision_attention_f32/single-s0256",
    "vision_attention_f32/packed-0003-0011-0031",
    "vision_attention_f32/isolation-baseline",
    "vision_attention_f32/isolation-poison-0",
    "vision_attention_f32/isolation-poison-1",
    "vision_attention_f32/isolation-poison-2",
];

type AttentionParts<'a> = (u32, u32, u32, &'a [f32], &'a [f32], &'a [f32], &'a [u32]);

fn unpack(invocation: &KernelInvocation) -> AttentionParts<'_> {
    let KernelInvocation::VisionAttentionF32 {
        tokens,
        heads,
        head_dim,
        query,
        key,
        value,
        cu_seqlens,
    } = invocation
    else {
        panic!("M3 vision corpus emitted a non-attention invocation")
    };
    (*tokens, *heads, *head_dim, query, key, value, cu_seqlens)
}

#[test]
fn corpus_is_compact_strict_deterministic_and_anchored_independently() {
    assert_eq!(
        M3_VISION_ATTENTION_SEQUENCE_LENGTHS,
        [8, 16, 31, 64, 127, 256]
    );
    assert_eq!(PADDLEOCR_VL_VISION_HEADS, 16);
    assert_eq!(PADDLEOCR_VL_VISION_HEAD_DIM, 72);
    assert_eq!(
        VISION_ATTENTION_FIXTURE_ALGORITHM,
        "affine-mod257-binary-f32-v1"
    );

    let first = m3_vision_attention_corpus().unwrap();
    let second = m3_vision_attention_corpus().unwrap();
    assert_eq!(first, second);
    assert_eq!(first.schema_version, 1);
    assert_eq!(
        first.oracle,
        "pvlc-cpu-ref/materialized-segmented-attention-f32-v1"
    );
    assert_eq!(first.fixture_algorithm, VISION_ATTENTION_FIXTURE_ALGORITHM);
    assert_eq!(
        first
            .cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<Vec<_>>(),
        REQUIRED_CASE_IDS
    );

    let canonical = serde_json::to_vec(&first).unwrap();
    assert_eq!(
        blake3::hash(&canonical),
        blake3::hash(&serde_json::to_vec(&second).unwrap())
    );
    assert!(
        canonical.len() < 24_000_000,
        "the browser artifact must not serialize three full Q/K/V tensors per case"
    );
    let text = std::str::from_utf8(&canonical).unwrap();
    assert!(!text.contains("\"query\""));
    assert!(!text.contains("\"key\""));
    assert!(!text.contains("\"value\""));
    assert_eq!(
        serde_json::from_slice::<M3VisionAttentionCorpus>(&canonical).unwrap(),
        first
    );
    let unknown = text.replacen('{', "{\"unexpected\":true,", 1);
    assert!(serde_json::from_str::<M3VisionAttentionCorpus>(&unknown).is_err());
}

#[test]
fn every_required_sequence_uses_real_model_geometry_and_a_materialized_cpu_oracle() {
    let corpus = m3_vision_attention_corpus().unwrap();
    let singles = &corpus.cases[..M3_VISION_ATTENTION_SEQUENCE_LENGTHS.len()];
    assert_eq!(singles.len(), 6);
    for (index, (case, tokens)) in singles
        .iter()
        .zip([8_u32, 16, 31, 64, 127, 256])
        .enumerate()
    {
        assert_eq!(case.tokens, tokens);
        assert_eq!(case.heads, 16);
        assert_eq!(case.head_dim, 72);
        assert_eq!(case.cu_seqlens, [0, tokens]);
        assert_eq!(case.seed, 101 + index as u32);
        assert_eq!(case.poisoned_segment, None);
        assert!(case.tags.iter().any(|tag| tag == "single-segment"));
        assert!(case.tags.iter().any(|tag| tag == "required-sequence"));
        assert!(case.tags.windows(2).all(|pair| pair[0] < pair[1]));

        let invocation = case.invocation().unwrap();
        let (actual_tokens, heads, head_dim, query, key, value, boundaries) = unpack(&invocation);
        assert_eq!((actual_tokens, heads, head_dim), (tokens, 16, 72));
        assert_eq!(boundaries, [0, tokens]);
        assert_eq!(query.len(), tokens as usize * 16 * 72);
        assert_eq!(key.len(), query.len());
        assert_eq!(value.len(), query.len());
        assert!(
            query
                .iter()
                .chain(key)
                .chain(value)
                .all(|item| item.is_finite())
        );
        assert_eq!(
            invocation.plan().unwrap().output_elements,
            case.expected.len()
        );
        assert_eq!(case.shape, [tokens as usize, 16, 72]);

        let expected = materialized_segmented_attention_f32(
            query,
            key,
            value,
            tokens as usize,
            16,
            72,
            &[0, tokens as usize],
        )
        .unwrap();
        assert_eq!(case.expected, expected);
        assert_eq!(case.policy.max_abs, 1.0e-3);
        assert_eq!(case.policy.max_mean_abs, 2.0e-4);
        assert_eq!(case.policy.max_p99_abs, 6.0e-4);
        assert_eq!(case.policy.max_relative_l2, 3.0e-4);
        assert_eq!(case.policy.min_cosine_similarity, 0.999_99);
        assert_eq!(case.policy.native_max_abs, 3.0e-4);
        assert_eq!(case.policy.native_max_relative_l2, 1.0e-4);
    }

    let first = singles[0].invocation().unwrap();
    let (_, _, _, query, key, value, _) = unpack(&first);
    assert_eq!(&query[..2], &[-0.078_125, 0.187_5]);
    assert_eq!(&key[..2], &[-1.671_875, -1.218_75]);
    assert_eq!(&value[..2], &[-1.187_5, -0.984_375]);
}

#[test]
fn packed_boundaries_and_each_poison_direction_are_observable_without_cross_image_leakage() {
    let corpus = m3_vision_attention_corpus().unwrap();
    let packed = &corpus.cases[6];
    assert_eq!(packed.cu_seqlens, [0, 3, 11, 31]);
    assert_eq!((packed.tokens, packed.heads, packed.head_dim), (31, 16, 72));
    assert!(packed.tags.iter().any(|tag| tag == "packed-segments"));
    let packed_invocation = packed.invocation().unwrap();
    let (_, _, _, query, key, value, boundaries) = unpack(&packed_invocation);
    assert_eq!(boundaries, [0, 3, 11, 31]);
    assert_eq!(
        packed.expected,
        materialized_segmented_attention_f32(query, key, value, 31, 16, 72, &[0, 3, 11, 31],)
            .unwrap()
    );

    let isolation = &corpus.cases[7..];
    assert_eq!(isolation.len(), 4);
    let baseline = &isolation[0];
    assert_eq!(baseline.cu_seqlens, [0, 3, 9, 17]);
    assert_eq!(baseline.poisoned_segment, None);
    assert_eq!(baseline.seed, 301);
    for case in isolation {
        let invocation = case.invocation().unwrap();
        let (tokens, heads, head_dim, query, key, value, boundaries) = unpack(&invocation);
        let boundaries = boundaries
            .iter()
            .map(|boundary| *boundary as usize)
            .collect::<Vec<_>>();
        assert_eq!(
            case.expected,
            materialized_segmented_attention_f32(
                query,
                key,
                value,
                tokens as usize,
                heads as usize,
                head_dim as usize,
                &boundaries,
            )
            .unwrap(),
            "{} must be independently anchored to the materialized oracle",
            case.id
        );
    }
    for (poisoned_segment, poisoned) in isolation[1..].iter().enumerate() {
        assert_eq!(poisoned.seed, baseline.seed);
        assert_eq!(poisoned.cu_seqlens, baseline.cu_seqlens);
        assert_eq!(poisoned.poisoned_segment, Some(poisoned_segment as u32));
        let poisoned_invocation = poisoned.invocation().unwrap();
        assert_eq!(
            poisoned_invocation.kernel_id(),
            KernelId::VisionAttentionF32
        );
        for segment in 0..3 {
            let start = baseline.cu_seqlens[segment] as usize * 16 * 72;
            let end = baseline.cu_seqlens[segment + 1] as usize * 16 * 72;
            if segment == poisoned_segment {
                assert_ne!(
                    &baseline.expected[start..end],
                    &poisoned.expected[start..end]
                );
            } else {
                assert_eq!(
                    &baseline.expected[start..end],
                    &poisoned.expected[start..end]
                );
            }
        }
    }
}
