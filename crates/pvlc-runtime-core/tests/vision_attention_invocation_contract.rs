use pvlc_runtime_core::{
    InvocationErrorCode, InvocationInput, KernelId, KernelInvocation, MAX_VISION_HEAD_DIM,
};

fn invocation(tokens: u32, heads: u32, head_dim: u32, cu_seqlens: Vec<u32>) -> KernelInvocation {
    let elements = (u64::from(tokens) * u64::from(heads) * u64::from(head_dim)) as usize;
    KernelInvocation::VisionAttentionF32 {
        tokens,
        heads,
        head_dim,
        query: vec![0.25; elements],
        key: vec![-0.5; elements],
        value: vec![1.0; elements],
        cu_seqlens,
    }
}

fn assert_error(invocation: KernelInvocation, expected: InvocationErrorCode) {
    assert_eq!(invocation.plan().unwrap_err().code(), expected);
}

#[test]
fn vision_attention_is_a_stable_additive_protocol_without_changing_the_m2_subset() {
    assert_eq!(MAX_VISION_HEAD_DIM, 72);
    assert_eq!(
        KernelId::VisionAttentionF32.as_str(),
        "vision_attention_f32"
    );
    assert_eq!(
        serde_json::to_string(&KernelId::VisionAttentionF32).unwrap(),
        "\"vision_attention_f32\""
    );
    assert_eq!(
        serde_json::from_str::<KernelId>("\"vision_attention_f32\"").unwrap(),
        KernelId::VisionAttentionF32
    );
    assert_eq!(KernelId::M2_PRIMITIVES.len(), 7);
    assert!(!KernelId::M2_PRIMITIVES.contains(&KernelId::VisionAttentionF32));
    let legacy = [
        KernelId::GemmF32,
        KernelId::GemvF32,
        KernelId::LayerNormF32,
        KernelId::RmsNormF32,
        KernelId::SiluF32,
        KernelId::GeluTanhF32,
        KernelId::RopeNeoxF32,
        KernelId::VisionAttentionF32,
        KernelId::VisionPatchProjectionF32,
        KernelId::AddF32,
        KernelId::GeluErfF32,
        KernelId::ProjectorMerge2x2F32,
    ];
    assert!(KernelId::ALL.len() >= legacy.len() + 11);
    assert_eq!(&KernelId::ALL[..legacy.len()], legacy);
    assert_eq!(KernelId::ALL[legacy.len()], KernelId::VisionQkvFusedF32);
    assert_eq!(
        &KernelId::ALL[legacy.len() + 1..legacy.len() + 11],
        [
            KernelId::DecoderKvAppendF32,
            KernelId::DecoderGqaF32,
            KernelId::DecoderGqaSplitPartialF32,
            KernelId::DecoderGqaSplitMergeF32,
            KernelId::DecoderMropeF32,
            KernelId::DecoderSwigluF32,
            KernelId::DecoderPrefillGqaF32,
            KernelId::DecoderPrefillMropeF32,
            KernelId::DecoderKvAppendRangeF32,
            KernelId::GemvTiledF32,
        ]
    );
    assert_eq!(
        &KernelId::ALL[..7],
        KernelId::M2_PRIMITIVES.as_slice(),
        "M3 additions must not reorder the frozen M2 protocol"
    );
    assert_eq!(KernelId::ALL[7], KernelId::VisionAttentionF32);
}

#[test]
fn vision_attention_plan_fixes_cardinality_dispatch_uniforms_and_binding_order() {
    let invocation = invocation(65, 16, 72, vec![0, 17, 65]);
    let plan = invocation.plan().unwrap();
    assert_eq!(plan.kernel, KernelId::VisionAttentionF32);
    assert_eq!(plan.output_elements, 65 * 16 * 72);
    assert_eq!(plan.output_bytes, (65 * 16 * 72 * 4) as u64);
    assert_eq!(plan.workgroup_size, [128, 1, 1]);
    assert_eq!(
        plan.dispatch,
        [1, 16, 1],
        "one workgroup must own 128 independent queries and share each K/V tile"
    );
    assert_eq!(
        invocation.uniform_bytes().unwrap(),
        [65_u32, 16, 72, 2]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );

    let KernelInvocation::VisionAttentionF32 {
        query,
        key,
        value,
        cu_seqlens,
        ..
    } = &invocation
    else {
        unreachable!()
    };
    assert_eq!(
        invocation.inputs(),
        vec![
            InvocationInput::F32(query),
            InvocationInput::F32(key),
            InvocationInput::F32(value),
            InvocationInput::U32(cu_seqlens),
        ]
    );
    assert_eq!(invocation.output_initializer(), None);
}

#[test]
fn vision_attention_tiled_dispatch_covers_query_tile_edges_and_segment_crossings() {
    for tokens in [1, 17, 127, 128, 129, 255, 256, 257] {
        let cu_seqlens = match tokens {
            17 => vec![0, 3, 17],
            257 => vec![0, 129, 257],
            _ => vec![0, tokens],
        };
        let plan = invocation(tokens, 16, 72, cu_seqlens).plan().unwrap();
        assert_eq!(plan.workgroup_size, [128, 1, 1]);
        assert_eq!(plan.dispatch, [tokens.div_ceil(128), 16, 1]);
    }
}

#[test]
fn vision_attention_json_is_strict_and_roundtrips_all_segment_boundaries() {
    let invocation = KernelInvocation::VisionAttentionF32 {
        tokens: 2,
        heads: 1,
        head_dim: 2,
        query: vec![1.0, 0.0, 0.0, 1.0],
        key: vec![1.0, 0.0, 0.0, 1.0],
        value: vec![10.0, 0.0, 0.0, 20.0],
        cu_seqlens: vec![0, 2],
    };
    let canonical = concat!(
        r#"{"kernel":"vision_attention_f32","tokens":2,"heads":1,"head_dim":2,"#,
        r#""query":[1.0,0.0,0.0,1.0],"key":[1.0,0.0,0.0,1.0],"#,
        r#""value":[10.0,0.0,0.0,20.0],"cu_seqlens":[0,2]}"#
    );
    assert_eq!(serde_json::to_string(&invocation).unwrap(), canonical);
    assert_eq!(
        serde_json::from_str::<KernelInvocation>(canonical).unwrap(),
        invocation
    );
    let unknown = canonical.replace(
        "\"cu_seqlens\":[0,2]",
        "\"cu_seqlens\":[0,2],\"causal\":false",
    );
    assert!(serde_json::from_str::<KernelInvocation>(&unknown).is_err());
}

#[test]
fn vision_attention_rejects_zero_oversized_overflown_and_malformed_requests() {
    assert_error(
        invocation(0, 1, 2, vec![0]),
        InvocationErrorCode::ZeroDimension,
    );
    assert_error(
        invocation(2, 0, 2, vec![0, 2]),
        InvocationErrorCode::ZeroDimension,
    );
    assert_error(
        invocation(2, 1, 0, vec![0, 2]),
        InvocationErrorCode::ZeroDimension,
    );
    assert_error(
        invocation(2, 1, MAX_VISION_HEAD_DIM + 1, vec![0, 2]),
        InvocationErrorCode::UnsupportedHeadDimension,
    );

    for boundaries in [
        vec![],
        vec![0],
        vec![1, 2],
        vec![0, 1],
        vec![0, 1, 1, 2],
        vec![0, 3, 2],
    ] {
        assert_error(
            invocation(2, 1, 2, boundaries),
            InvocationErrorCode::InvalidSequenceBoundaries,
        );
    }

    for operand in 0..3 {
        for oversized in [false, true] {
            let KernelInvocation::VisionAttentionF32 {
                tokens,
                heads,
                head_dim,
                mut query,
                mut key,
                mut value,
                cu_seqlens,
            } = invocation(2, 1, 2, vec![0, 2])
            else {
                unreachable!()
            };
            let selected = match operand {
                0 => &mut query,
                1 => &mut key,
                2 => &mut value,
                _ => unreachable!(),
            };
            if oversized {
                selected.push(0.0);
            } else {
                selected.pop();
            }
            assert_error(
                KernelInvocation::VisionAttentionF32 {
                    tokens,
                    heads,
                    head_dim,
                    query,
                    key,
                    value,
                    cu_seqlens,
                },
                InvocationErrorCode::LengthMismatch,
            );
        }
    }

    assert_error(
        KernelInvocation::VisionAttentionF32 {
            tokens: u32::MAX,
            heads: u32::MAX,
            head_dim: MAX_VISION_HEAD_DIM,
            query: vec![],
            key: vec![],
            value: vec![],
            cu_seqlens: vec![0, u32::MAX],
        },
        InvocationErrorCode::ArithmeticOverflow,
    );
}

#[test]
fn vision_attention_rejects_nonfinite_values_in_every_qkv_operand() {
    for operand in 0..3 {
        let KernelInvocation::VisionAttentionF32 {
            tokens,
            heads,
            head_dim,
            mut query,
            mut key,
            mut value,
            cu_seqlens,
        } = invocation(2, 1, 2, vec![0, 2])
        else {
            unreachable!()
        };
        [&mut query, &mut key, &mut value][operand][1] = f32::NAN;
        assert_error(
            KernelInvocation::VisionAttentionF32 {
                tokens,
                heads,
                head_dim,
                query,
                key,
                value,
                cu_seqlens,
            },
            InvocationErrorCode::NonFiniteInput,
        );
    }
}
