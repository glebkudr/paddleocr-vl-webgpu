use pvlc_cpu_ref::{
    CpuRefErrorCode, KvBlockOrder, LayerNormParameters, LinearParameters, VisionEncoderLayerConfig,
    VisionEncoderLayerParameters, vision_encoder_layer_identity_rope_f32,
};

const INPUT: [f32; 8] = [1.0, -1.0, 2.0, 0.5, -0.5, 1.5, 0.25, -2.0];

const NORM1_WEIGHT: [f32; 4] = [1.1, 0.9, 1.2, 0.8];
const NORM1_BIAS: [f32; 4] = [0.1, -0.2, 0.05, 0.3];
const QUERY_WEIGHT: [f32; 16] = [
    0.2, -0.1, 0.3, 0.4, -0.5, 0.6, 0.1, -0.2, 0.7, 0.2, -0.4, 0.1, -0.3, 0.5, 0.6, -0.7,
];
const QUERY_BIAS: [f32; 4] = [0.01, -0.02, 0.03, -0.04];
const KEY_WEIGHT: [f32; 16] = [
    -0.4, 0.2, 0.5, 0.1, 0.3, -0.6, 0.2, 0.4, 0.1, 0.7, -0.2, -0.5, 0.6, 0.1, 0.3, -0.4,
];
const KEY_BIAS: [f32; 4] = [-0.03, 0.02, 0.04, -0.01];
const VALUE_WEIGHT: [f32; 16] = [
    0.5, 0.1, -0.3, 0.2, -0.2, 0.4, 0.6, 0.1, 0.3, -0.5, 0.2, 0.7, -0.6, 0.2, 0.1, 0.5,
];
const VALUE_BIAS: [f32; 4] = [0.05, -0.04, 0.02, 0.03];
const ATTENTION_OUTPUT_WEIGHT: [f32; 16] = [
    0.4, -0.2, 0.1, 0.3, -0.1, 0.5, 0.2, -0.4, 0.6, 0.1, -0.5, 0.2, 0.2, -0.3, 0.4, 0.7,
];
const ATTENTION_OUTPUT_BIAS: [f32; 4] = [0.02, -0.01, 0.03, -0.02];
const NORM2_WEIGHT: [f32; 4] = [0.95, 1.05, 0.85, 1.15];
const NORM2_BIAS: [f32; 4] = [-0.05, 0.07, -0.02, 0.04];
const MLP_FC1_WEIGHT: [f32; 12] = [
    0.2, -0.4, 0.6, 0.1, -0.5, 0.3, 0.2, 0.7, 0.4, 0.1, -0.3, 0.5,
];
const MLP_FC1_BIAS: [f32; 3] = [0.01, -0.02, 0.03];
const MLP_FC2_WEIGHT: [f32; 12] = [
    0.3, -0.2, 0.5, -0.4, 0.6, 0.1, 0.2, 0.3, -0.5, 0.7, -0.1, 0.4,
];
const MLP_FC2_BIAS: [f32; 4] = [-0.01, 0.02, -0.03, 0.04];

fn tiny_config() -> VisionEncoderLayerConfig {
    VisionEncoderLayerConfig {
        tokens: 2,
        hidden_size: 4,
        attention_heads: 2,
        head_dim: 2,
        intermediate_size: 3,
        layer_norm_epsilon: 1.0e-5,
        attention_key_tile: 1,
        attention_order: KvBlockOrder::Forward,
    }
}

fn tiny_parameters() -> VisionEncoderLayerParameters<'static> {
    VisionEncoderLayerParameters {
        norm1: LayerNormParameters {
            weight: &NORM1_WEIGHT,
            bias: &NORM1_BIAS,
        },
        query: LinearParameters {
            weight: &QUERY_WEIGHT,
            bias: &QUERY_BIAS,
        },
        key: LinearParameters {
            weight: &KEY_WEIGHT,
            bias: &KEY_BIAS,
        },
        value: LinearParameters {
            weight: &VALUE_WEIGHT,
            bias: &VALUE_BIAS,
        },
        attention_output: LinearParameters {
            weight: &ATTENTION_OUTPUT_WEIGHT,
            bias: &ATTENTION_OUTPUT_BIAS,
        },
        norm2: LayerNormParameters {
            weight: &NORM2_WEIGHT,
            bias: &NORM2_BIAS,
        },
        mlp_fc1: LinearParameters {
            weight: &MLP_FC1_WEIGHT,
            bias: &MLP_FC1_BIAS,
        },
        mlp_fc2: LinearParameters {
            weight: &MLP_FC2_WEIGHT,
            bias: &MLP_FC2_BIAS,
        },
    }
}

fn assert_slice_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}[{index}]: actual={actual:?} expected={expected:?} tolerance={tolerance:?}"
        );
    }
}

#[test]
fn composed_identity_rope_layer_matches_an_independent_tiny_transformer_trace() {
    let trace =
        vision_encoder_layer_identity_rope_f32(&INPUT, tiny_config(), &[0, 2], tiny_parameters())
            .unwrap();

    for (label, actual, expected) in [
        (
            "norm1",
            &trace.norm1[..],
            &[
                0.481_049_54,
                -1.550_993_8,
                1.574_198_1,
                0.207_624_38,
                -0.171_294_12,
                0.998_626_9,
                0.464_340_2,
                -0.844_368_04,
            ][..],
        ),
        (
            "query",
            &trace.query[..],
            &[
                0.816_618_5,
                -1.075_226_2,
                -0.552_380_9,
                -0.160_629_9,
                -0.322_566_66,
                0.880_130_9,
                -0.160_353_39,
                1.380_363_5,
            ][..],
        ),
        (
            "key",
            &trace.key[..],
            &[
                0.275_242_95,
                1.492_800_5,
                -1.416_242_5,
                0.512_740_1,
                0.385_976_3,
                -0.875_443_64,
                1.051_225_3,
                0.464_135_53,
            ][..],
        ),
        (
            "value",
            &trace.value[..],
            &[
                -0.295_309_2,
                0.208_673_98,
                1.399_988_4,
                -0.307_596_47,
                -0.243_960_07,
                0.587_876_86,
                -1.028_891_3,
                -0.043_248_147,
            ][..],
        ),
        (
            "attention_context",
            &trace.attention_context[..],
            &[
                -0.250_849_96,
                0.536_996_4,
                0.726_634_74,
                -0.234_311_7,
                -0.285_935_1,
                0.277_899_74,
                0.382_488_82,
                -0.196_856_41,
            ][..],
        ),
        (
            "attention_output",
            &trace.attention_output[..],
            &[
                -0.185_369_3,
                0.522_634_86,
                -0.476_990_07,
                -0.104_633_18,
                -0.170_762_03,
                0.312_783_72,
                -0.344_386_8,
                -0.145_360_9,
            ][..],
        ),
        (
            "attention_residual",
            &trace.attention_residual[..],
            &[
                0.814_630_7,
                -0.477_365_14,
                1.523_009_9,
                0.395_366_82,
                -0.670_762_06,
                1.812_783_7,
                -0.094_386_786,
                -2.145_361,
            ][..],
        ),
        (
            "norm2",
            &trace.norm2[..],
            &[
                0.279_083_2,
                -1.440_594_9,
                1.106_355_5,
                -0.227_795_69,
                -0.315_467_2,
                1.615_205_5,
                0.087_901_92,
                -1.476_998,
            ][..],
        ),
        (
            "mlp_fc1",
            &trace.mlp_fc1[..],
            &[
                1.283_088_4,
                -0.529_905_9,
                -0.448_230_7,
                -0.794_134_26,
                -0.394_022_88,
                -0.699_535_9,
            ][..],
        ),
        (
            "mlp_activation",
            &trace.mlp_activation[..],
            &[
                1.154_896_9,
                -0.157_980_14,
                -0.146_58,
                -0.169_676_38,
                -0.136_647_18,
                -0.169_418_83,
            ][..],
        ),
        (
            "mlp_output",
            &trace.mlp_output[..],
            &[
                0.294_775_13,
                -0.551_404_83,
                0.226_875_3,
                0.805_593_85,
                -0.118_282_89,
                -0.011_059_646,
                -0.020_220_017,
                -0.132_876_28,
            ][..],
        ),
        (
            "output",
            &trace.output[..],
            &[
                1.109_405_8,
                -1.028_77,
                1.749_885_2,
                1.200_960_6,
                -0.789_045,
                1.801_724_1,
                -0.114_606_805,
                -2.278_237_3,
            ][..],
        ),
    ] {
        assert_slice_close(label, actual, expected, 2.0e-5);
    }
}

#[test]
fn composed_layer_preserves_packed_image_attention_boundaries() {
    let baseline = vision_encoder_layer_identity_rope_f32(
        &INPUT,
        tiny_config(),
        &[0, 1, 2],
        tiny_parameters(),
    )
    .unwrap();
    let mut changed_input = INPUT;
    changed_input[4..].copy_from_slice(&[20.0, -30.0, 40.0, -50.0]);
    let isolated = vision_encoder_layer_identity_rope_f32(
        &changed_input,
        tiny_config(),
        &[0, 1, 2],
        tiny_parameters(),
    )
    .unwrap();
    let joint = vision_encoder_layer_identity_rope_f32(
        &changed_input,
        tiny_config(),
        &[0, 2],
        tiny_parameters(),
    )
    .unwrap();

    assert_eq!(&baseline.output[..4], &isolated.output[..4]);
    assert!(
        baseline.output[..4]
            .iter()
            .zip(&joint.output[..4])
            .any(|(baseline, joint): (&f32, &f32)| (baseline - joint).abs() > 1.0e-3)
    );
}

#[test]
fn composed_layer_rejects_invalid_geometry_boundaries_weights_and_values() {
    for config in [
        VisionEncoderLayerConfig {
            tokens: 0,
            ..tiny_config()
        },
        VisionEncoderLayerConfig {
            attention_heads: 3,
            head_dim: 1,
            ..tiny_config()
        },
        VisionEncoderLayerConfig {
            attention_heads: usize::MAX,
            ..tiny_config()
        },
        VisionEncoderLayerConfig {
            intermediate_size: 0,
            ..tiny_config()
        },
    ] {
        assert_eq!(
            vision_encoder_layer_identity_rope_f32(&INPUT, config, &[0, 2], tiny_parameters(),)
                .unwrap_err()
                .code(),
            CpuRefErrorCode::DimensionMismatch
        );
    }

    for (config, boundaries, expected) in [
        (
            VisionEncoderLayerConfig {
                layer_norm_epsilon: 0.0,
                ..tiny_config()
            },
            &[0, 2][..],
            CpuRefErrorCode::NonPositiveEpsilon,
        ),
        (
            VisionEncoderLayerConfig {
                attention_key_tile: 0,
                ..tiny_config()
            },
            &[0, 2][..],
            CpuRefErrorCode::InvalidTileSize,
        ),
        (
            tiny_config(),
            &[0, 1, 1, 2][..],
            CpuRefErrorCode::InvalidSequenceBoundaries,
        ),
    ] {
        assert_eq!(
            vision_encoder_layer_identity_rope_f32(&INPUT, config, boundaries, tiny_parameters(),)
                .unwrap_err()
                .code(),
            expected
        );
    }

    let malformed = VisionEncoderLayerParameters {
        query: LinearParameters {
            weight: &QUERY_WEIGHT[..QUERY_WEIGHT.len() - 1],
            bias: &QUERY_BIAS,
        },
        ..tiny_parameters()
    };
    assert_eq!(
        vision_encoder_layer_identity_rope_f32(&INPUT, tiny_config(), &[0, 2], malformed)
            .unwrap_err()
            .code(),
        CpuRefErrorCode::DimensionMismatch
    );

    let mut nonfinite_query = QUERY_WEIGHT;
    nonfinite_query[7] = f32::NAN;
    let nonfinite = VisionEncoderLayerParameters {
        query: LinearParameters {
            weight: &nonfinite_query,
            bias: &QUERY_BIAS,
        },
        ..tiny_parameters()
    };
    assert_eq!(
        vision_encoder_layer_identity_rope_f32(&INPUT, tiny_config(), &[0, 2], nonfinite)
            .unwrap_err()
            .code(),
        CpuRefErrorCode::NonFiniteInput
    );
}
