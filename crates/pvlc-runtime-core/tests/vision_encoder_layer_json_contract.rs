use pvlc_runtime_core::{
    OwnedVisionEncoderLayerInvocation, OwnedVisionEncoderLayerParameters,
    OwnedVisionLayerNormParameters, OwnedVisionLinearParameters, VisionEncoderLayerStage,
    VisionLayerReadback, VisionRopeSpecialization,
};

const TOKENS: u32 = 3;
const HIDDEN: u32 = 6;
const HEADS: u32 = 2;
const HEAD_DIM: u32 = 3;
const INTERMEDIATE: u32 = 7;

fn linear(input_width: u32, output_width: u32, seed: f32) -> OwnedVisionLinearParameters {
    OwnedVisionLinearParameters {
        weight: (0..input_width * output_width)
            .map(|index| seed + index as f32 / 128.0)
            .collect(),
        bias: (0..output_width)
            .map(|index| seed / 4.0 - index as f32 / 64.0)
            .collect(),
    }
}

fn fixture() -> OwnedVisionEncoderLayerInvocation {
    OwnedVisionEncoderLayerInvocation {
        tokens: TOKENS,
        hidden_size: HIDDEN,
        attention_heads: HEADS,
        head_dim: HEAD_DIM,
        intermediate_size: INTERMEDIATE,
        layer_norm_epsilon: 1.0e-6,
        input: (0..TOKENS * HIDDEN)
            .map(|index| index as f32 / 32.0 - 0.25)
            .collect(),
        cu_seqlens: vec![0, 1, TOKENS],
        parameters: OwnedVisionEncoderLayerParameters {
            norm1: OwnedVisionLayerNormParameters {
                weight: vec![1.0; HIDDEN as usize],
                bias: vec![0.0; HIDDEN as usize],
            },
            query: linear(HIDDEN, HIDDEN, 0.01),
            key: linear(HIDDEN, HIDDEN, 0.02),
            value: linear(HIDDEN, HIDDEN, 0.03),
            attention_output: linear(HIDDEN, HIDDEN, 0.04),
            norm2: OwnedVisionLayerNormParameters {
                weight: vec![0.75; HIDDEN as usize],
                bias: vec![0.125; HIDDEN as usize],
            },
            mlp_fc1: linear(HIDDEN, INTERMEDIATE, 0.05),
            mlp_fc2: linear(INTERMEDIATE, HIDDEN, 0.06),
        },
    }
}

#[test]
fn owned_layer_json_roundtrips_into_the_same_valid_resident_plan() {
    let fixture = fixture();
    let json = serde_json::to_string(&fixture).unwrap();
    let decoded: OwnedVisionEncoderLayerInvocation = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded, fixture);
    let plan = decoded.borrowed().plan().unwrap();
    assert_eq!(
        plan.dispatches.map(|dispatch| dispatch.stage),
        VisionEncoderLayerStage::ALL
    );
    assert_eq!(plan.rope_specialization, VisionRopeSpecialization::Identity);
}

#[test]
fn layer_transport_is_strict_at_every_public_object_boundary() {
    let fixture = fixture();
    let mut root = serde_json::to_value(&fixture).unwrap();
    root.as_object_mut()
        .unwrap()
        .insert("unexpected".to_owned(), serde_json::json!(true));
    assert!(serde_json::from_value::<OwnedVisionEncoderLayerInvocation>(root).is_err());

    let mut parameters = serde_json::to_value(&fixture).unwrap();
    parameters["parameters"]
        .as_object_mut()
        .unwrap()
        .insert("unexpected".to_owned(), serde_json::json!([]));
    assert!(serde_json::from_value::<OwnedVisionEncoderLayerInvocation>(parameters).is_err());

    let mut linear = serde_json::to_value(&fixture).unwrap();
    linear["parameters"]["query"]
        .as_object_mut()
        .unwrap()
        .insert("unexpected".to_owned(), serde_json::json!(0));
    assert!(serde_json::from_value::<OwnedVisionEncoderLayerInvocation>(linear).is_err());

    let mut norm = serde_json::to_value(&fixture).unwrap();
    norm["parameters"]["norm1"]
        .as_object_mut()
        .unwrap()
        .insert("unexpected".to_owned(), serde_json::json!(0));
    assert!(serde_json::from_value::<OwnedVisionEncoderLayerInvocation>(norm).is_err());
}

#[test]
fn layer_protocol_names_are_stable_for_browser_diagnostics_and_readback_selection() {
    assert_eq!(
        serde_json::to_value(VisionEncoderLayerStage::ALL).unwrap(),
        serde_json::json!([
            "norm1",
            "query",
            "key",
            "value",
            "attention_context",
            "attention_output",
            "attention_residual",
            "norm2",
            "mlp_fc1",
            "mlp_activation",
            "mlp_output",
            "output"
        ])
    );
    assert_eq!(
        serde_json::to_string(&VisionRopeSpecialization::Identity).unwrap(),
        "\"identity\""
    );
    assert_eq!(
        serde_json::to_string(&VisionLayerReadback::AllStages).unwrap(),
        "\"all_stages\""
    );
    assert_eq!(
        serde_json::to_string(&VisionLayerReadback::OutputOnly).unwrap(),
        "\"output_only\""
    );
    assert_eq!(
        serde_json::from_str::<VisionLayerReadback>("\"all_stages\"").unwrap(),
        VisionLayerReadback::AllStages
    );
    assert!(serde_json::from_str::<VisionLayerReadback>("\"debug\"").is_err());
}

#[test]
fn decoded_layer_still_rejects_semantically_invalid_dimensions_and_values() {
    let mut wrong_geometry = fixture();
    wrong_geometry.head_dim = 4;
    assert!(wrong_geometry.borrowed().plan().is_err());

    let mut malformed_boundaries = fixture();
    malformed_boundaries.cu_seqlens = vec![0, 2, 2, TOKENS];
    assert!(malformed_boundaries.borrowed().plan().is_err());

    let mut nonfinite = fixture();
    nonfinite.parameters.mlp_fc2.weight[0] = f32::INFINITY;
    assert!(nonfinite.borrowed().plan().is_err());
}
