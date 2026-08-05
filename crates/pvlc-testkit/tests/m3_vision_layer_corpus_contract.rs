use pvlc_cpu_ref::{
    KvBlockOrder, LayerNormParameters as CpuNorm, LinearParameters as CpuLinear,
    VisionEncoderLayerConfig as CpuConfig, VisionEncoderLayerParameters as CpuParameters,
    VisionEncoderLayerTrace, vision_encoder_layer_identity_rope_f32,
};
use pvlc_runtime_core::{OwnedVisionEncoderLayerInvocation, VisionEncoderLayerStage};
use pvlc_testkit::{
    M3_VISION_LAYER_ATTENTION_HEADS, M3_VISION_LAYER_CU_SEQLENS, M3_VISION_LAYER_HEAD_DIM,
    M3_VISION_LAYER_HIDDEN_SIZE, M3_VISION_LAYER_INTERMEDIATE_SIZE, M3_VISION_LAYER_TOKENS,
    VISION_LAYER_FIXTURE_ALGORITHM, m3_vision_layer_corpus,
};

const CASE_IDS: [&str; 4] = [
    "vision_encoder_layer_identity_rope/baseline",
    "vision_encoder_layer_identity_rope/poison-segment-0",
    "vision_encoder_layer_identity_rope/poison-segment-1",
    "vision_encoder_layer_identity_rope/poison-segment-2",
];

fn cpu_trace(invocation: &OwnedVisionEncoderLayerInvocation) -> VisionEncoderLayerTrace {
    let boundaries = invocation
        .cu_seqlens
        .iter()
        .map(|value| *value as usize)
        .collect::<Vec<_>>();
    let parameters = &invocation.parameters;
    vision_encoder_layer_identity_rope_f32(
        &invocation.input,
        CpuConfig {
            tokens: invocation.tokens as usize,
            hidden_size: invocation.hidden_size as usize,
            attention_heads: invocation.attention_heads as usize,
            head_dim: invocation.head_dim as usize,
            intermediate_size: invocation.intermediate_size as usize,
            layer_norm_epsilon: invocation.layer_norm_epsilon,
            attention_key_tile: 4,
            attention_order: KvBlockOrder::Forward,
        },
        &boundaries,
        CpuParameters {
            norm1: CpuNorm {
                weight: &parameters.norm1.weight,
                bias: &parameters.norm1.bias,
            },
            query: CpuLinear {
                weight: &parameters.query.weight,
                bias: &parameters.query.bias,
            },
            key: CpuLinear {
                weight: &parameters.key.weight,
                bias: &parameters.key.bias,
            },
            value: CpuLinear {
                weight: &parameters.value.weight,
                bias: &parameters.value.bias,
            },
            attention_output: CpuLinear {
                weight: &parameters.attention_output.weight,
                bias: &parameters.attention_output.bias,
            },
            norm2: CpuNorm {
                weight: &parameters.norm2.weight,
                bias: &parameters.norm2.bias,
            },
            mlp_fc1: CpuLinear {
                weight: &parameters.mlp_fc1.weight,
                bias: &parameters.mlp_fc1.bias,
            },
            mlp_fc2: CpuLinear {
                weight: &parameters.mlp_fc2.weight,
                bias: &parameters.mlp_fc2.bias,
            },
        },
    )
    .unwrap()
}

fn trace_stage(trace: &VisionEncoderLayerTrace, stage: VisionEncoderLayerStage) -> &[f32] {
    match stage {
        VisionEncoderLayerStage::Norm1 => &trace.norm1,
        VisionEncoderLayerStage::Query => &trace.query,
        VisionEncoderLayerStage::Key => &trace.key,
        VisionEncoderLayerStage::Value => &trace.value,
        VisionEncoderLayerStage::AttentionContext => &trace.attention_context,
        VisionEncoderLayerStage::AttentionOutput => &trace.attention_output,
        VisionEncoderLayerStage::AttentionResidual => &trace.attention_residual,
        VisionEncoderLayerStage::Norm2 => &trace.norm2,
        VisionEncoderLayerStage::MlpFc1 => &trace.mlp_fc1,
        VisionEncoderLayerStage::MlpActivation => &trace.mlp_activation,
        VisionEncoderLayerStage::MlpOutput => &trace.mlp_output,
        VisionEncoderLayerStage::Output => &trace.output,
    }
}

#[test]
fn corpus_has_one_asymmetric_packed_baseline_and_every_single_segment_poison() {
    assert_eq!(M3_VISION_LAYER_TOKENS, 9);
    assert_eq!(M3_VISION_LAYER_HIDDEN_SIZE, 18);
    assert_eq!(M3_VISION_LAYER_ATTENTION_HEADS, 3);
    assert_eq!(M3_VISION_LAYER_HEAD_DIM, 6);
    assert_eq!(M3_VISION_LAYER_INTERMEDIATE_SIZE, 23);
    assert_eq!(M3_VISION_LAYER_CU_SEQLENS, [0, 2, 5, 9]);
    assert_eq!(
        VISION_LAYER_FIXTURE_ALGORITHM,
        "vision-layer-affine-mod257-binary-f32-v1"
    );

    let corpus = m3_vision_layer_corpus().unwrap();
    assert_eq!(corpus.schema_version, 1);
    assert_eq!(
        corpus.oracle,
        "pvlc-cpu-ref/vision-encoder-layer-identity-rope-f32-v1"
    );
    assert_eq!(corpus.fixture_algorithm, VISION_LAYER_FIXTURE_ALGORITHM);
    assert_eq!(
        corpus
            .cases
            .iter()
            .map(|case| case.id.as_str())
            .collect::<Vec<_>>(),
        CASE_IDS
    );
    assert_eq!(corpus.cases[0].poisoned_segment, None);
    for segment in 0..3_u32 {
        assert_eq!(
            corpus.cases[segment as usize + 1].poisoned_segment,
            Some(segment)
        );
    }
    for case in &corpus.cases {
        assert_eq!(case.policy.max_abs, 2.0e-4);
        assert_eq!(case.policy.max_mean_abs, 2.0e-5);
        assert_eq!(case.policy.max_p99_abs, 1.0e-4);
        assert_eq!(case.policy.max_relative_l2, 1.0e-4);
        assert_eq!(case.policy.min_cosine_similarity, 0.999_99);
        assert_eq!(case.policy.native_max_abs, 2.0e-4);
        assert_eq!(case.policy.native_max_relative_l2, 1.0e-4);
    }
}

#[test]
fn compact_fixture_expansion_is_deterministic_strictly_shaped_and_json_ready() {
    let first = m3_vision_layer_corpus().unwrap();
    let second = m3_vision_layer_corpus().unwrap();
    assert_eq!(first, second);

    let baseline = first.cases[0].invocation().unwrap();
    assert_eq!(baseline.tokens, M3_VISION_LAYER_TOKENS);
    assert_eq!(baseline.hidden_size, M3_VISION_LAYER_HIDDEN_SIZE);
    assert_eq!(baseline.attention_heads, M3_VISION_LAYER_ATTENTION_HEADS);
    assert_eq!(baseline.head_dim, M3_VISION_LAYER_HEAD_DIM);
    assert_eq!(
        baseline.intermediate_size,
        M3_VISION_LAYER_INTERMEDIATE_SIZE
    );
    assert_eq!(baseline.cu_seqlens, M3_VISION_LAYER_CU_SEQLENS);
    assert_eq!(&baseline.input[..2], &[0.640_625, 0.906_25]);
    assert_eq!(
        &baseline.parameters.norm1.weight[..2],
        &[1.173_828_1, 1.187_5]
    );
    assert_eq!(
        &baseline.parameters.norm1.bias[..2],
        &[0.119_140_625, -0.121_093_75]
    );
    assert_eq!(
        &baseline.parameters.query.weight[..2],
        &[-0.067_382_81, -0.050_781_25]
    );
    assert_eq!(baseline.borrowed().plan().unwrap().dispatches.len(), 12);

    let poisoned = first.cases[2].invocation().unwrap();
    let row = M3_VISION_LAYER_HIDDEN_SIZE as usize;
    let start = M3_VISION_LAYER_CU_SEQLENS[1] as usize * row;
    let end = M3_VISION_LAYER_CU_SEQLENS[2] as usize * row;
    assert_eq!(&baseline.input[..start], &poisoned.input[..start]);
    assert_eq!(&baseline.input[end..], &poisoned.input[end..]);
    assert!(
        baseline.input[start..end]
            .iter()
            .zip(&poisoned.input[start..end])
            .all(|(left, right)| left != right)
    );
    assert_eq!(poisoned.input[start], 64.156_25);

    let artifact = serde_json::to_vec(&first).unwrap();
    let text = std::str::from_utf8(&artifact).unwrap();
    assert!(!text.contains("\"input\""));
    assert!(!text.contains("\"weight\""));
    assert!(
        artifact.len() < 250_000,
        "compact corpus grew to {} bytes",
        artifact.len()
    );
}

#[test]
fn every_serialized_checkpoint_matches_the_independent_composed_cpu_oracle() {
    let corpus = m3_vision_layer_corpus().unwrap();
    for case in &corpus.cases {
        let invocation = case.invocation().unwrap();
        let trace = cpu_trace(&invocation);
        for stage in VisionEncoderLayerStage::ALL {
            let expected = case.expected.stage(stage);
            let oracle = trace_stage(&trace, stage);
            assert_eq!(expected, oracle, "{} stage {stage:?}", case.id);
            assert!(expected.iter().all(|value| value.is_finite()));
            let expected_width = if matches!(
                stage,
                VisionEncoderLayerStage::MlpFc1 | VisionEncoderLayerStage::MlpActivation
            ) {
                M3_VISION_LAYER_INTERMEDIATE_SIZE
            } else {
                M3_VISION_LAYER_HIDDEN_SIZE
            } as usize;
            assert_eq!(
                expected.len(),
                M3_VISION_LAYER_TOKENS as usize * expected_width
            );
        }
    }
}

#[test]
fn oracle_outputs_prove_exact_end_to_end_packed_segment_isolation() {
    let corpus = m3_vision_layer_corpus().unwrap();
    let baseline = &corpus.cases[0];
    let row = M3_VISION_LAYER_HIDDEN_SIZE as usize;
    for segment in 0..3_usize {
        let poisoned = &corpus.cases[segment + 1];
        let start = M3_VISION_LAYER_CU_SEQLENS[segment] as usize * row;
        let end = M3_VISION_LAYER_CU_SEQLENS[segment + 1] as usize * row;
        let expected = baseline.expected.stage(VisionEncoderLayerStage::Output);
        let actual = poisoned.expected.stage(VisionEncoderLayerStage::Output);
        assert!(
            expected[start..end]
                .iter()
                .zip(&actual[start..end])
                .any(|(left, right)| left != right),
            "poison {segment} did not affect its own final-output segment"
        );
        for other in 0..3_usize {
            if other == segment {
                continue;
            }
            let other_start = M3_VISION_LAYER_CU_SEQLENS[other] as usize * row;
            let other_end = M3_VISION_LAYER_CU_SEQLENS[other + 1] as usize * row;
            assert_eq!(
                &expected[other_start..other_end],
                &actual[other_start..other_end],
                "poison {segment} leaked into segment {other}"
            );
        }
    }
}
