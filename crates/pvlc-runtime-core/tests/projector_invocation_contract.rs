use pvlc_runtime_core::{
    InvocationErrorCode, InvocationInput, KernelId, KernelInvocation, OwnedProjectorInvocation,
    OwnedProjectorParameters, OwnedVisionLayerNormParameters, OwnedVisionLinearParameters,
    ProjectorDispatch, ProjectorInvocation, ProjectorParameters, ProjectorReadback, ProjectorStage,
    VisionLayerNormParameters, VisionLinearParameters,
};

const HIDDEN: u32 = 3;
const MERGED: u32 = HIDDEN * 4;
const OUTPUT: u32 = 5;
const EPSILON: f32 = 1.0e-5;
const GRIDS: [[u32; 3]; 2] = [[1, 2, 4], [2, 2, 2]];

#[derive(Clone)]
struct Fixture {
    hidden: u32,
    output: u32,
    epsilon: f32,
    grids: Vec<[u32; 3]>,
    input: Vec<f32>,
    pre_norm_weight: Vec<f32>,
    pre_norm_bias: Vec<f32>,
    linear1_weight: Vec<f32>,
    linear1_bias: Vec<f32>,
    linear2_weight: Vec<f32>,
    linear2_bias: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            hidden: HIDDEN,
            output: OUTPUT,
            epsilon: EPSILON,
            grids: GRIDS.to_vec(),
            input: values(16 * HIDDEN as usize, 1),
            pre_norm_weight: shifted_values(HIDDEN as usize, 2, 1.0),
            pre_norm_bias: values(HIDDEN as usize, 3),
            linear1_weight: values((MERGED * MERGED) as usize, 4),
            linear1_bias: values(MERGED as usize, 5),
            linear2_weight: values((OUTPUT * MERGED) as usize, 6),
            linear2_bias: values(OUTPUT as usize, 7),
        }
    }

    fn tiled_boundary() -> Self {
        let hidden = 1;
        let merged = hidden * 4;
        let output = 33;
        let grids = vec![[1, 12, 12]];
        let input_tokens = 144;
        Self {
            hidden,
            output,
            epsilon: EPSILON,
            grids,
            input: values((input_tokens * hidden) as usize, 41),
            pre_norm_weight: shifted_values(hidden as usize, 42, 1.0),
            pre_norm_bias: values(hidden as usize, 43),
            linear1_weight: values((merged * merged) as usize, 44),
            linear1_bias: values(merged as usize, 45),
            linear2_weight: values((output * merged) as usize, 46),
            linear2_bias: values(output as usize, 47),
        }
    }

    fn invocation(&self) -> ProjectorInvocation<'_> {
        ProjectorInvocation {
            hidden_size: self.hidden,
            output_size: self.output,
            layer_norm_epsilon: self.epsilon,
            input: &self.input,
            image_grid_thw: &self.grids,
            parameters: ProjectorParameters {
                pre_norm: VisionLayerNormParameters {
                    weight: &self.pre_norm_weight,
                    bias: &self.pre_norm_bias,
                },
                linear1: VisionLinearParameters {
                    weight: &self.linear1_weight,
                    bias: &self.linear1_bias,
                },
                linear2: VisionLinearParameters {
                    weight: &self.linear2_weight,
                    bias: &self.linear2_bias,
                },
            },
        }
    }

    fn owned(&self) -> OwnedProjectorInvocation {
        OwnedProjectorInvocation {
            hidden_size: self.hidden,
            output_size: self.output,
            layer_norm_epsilon: self.epsilon,
            input: self.input.clone(),
            image_grid_thw: self.grids.clone(),
            parameters: OwnedProjectorParameters {
                pre_norm: OwnedVisionLayerNormParameters {
                    weight: self.pre_norm_weight.clone(),
                    bias: self.pre_norm_bias.clone(),
                },
                linear1: OwnedVisionLinearParameters {
                    weight: self.linear1_weight.clone(),
                    bias: self.linear1_bias.clone(),
                },
                linear2: OwnedVisionLinearParameters {
                    weight: self.linear2_weight.clone(),
                    bias: self.linear2_bias.clone(),
                },
            },
        }
    }

    fn floating_operand_mut(&mut self, operand: usize) -> &mut Vec<f32> {
        match operand {
            0 => &mut self.input,
            1 => &mut self.pre_norm_weight,
            2 => &mut self.pre_norm_bias,
            3 => &mut self.linear1_weight,
            4 => &mut self.linear1_bias,
            5 => &mut self.linear2_weight,
            6 => &mut self.linear2_bias,
            _ => panic!("projector fixture has seven floating operands"),
        }
    }
}

fn values(length: usize, seed: u32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            let phase = (index as f32 + 1.0) * (seed as f32 + 0.25) * 0.031;
            phase.sin() * 0.2 - phase.cos() * 0.07
        })
        .collect()
}

fn shifted_values(length: usize, seed: u32, shift: f32) -> Vec<f32> {
    values(length, seed)
        .into_iter()
        .map(|value| value + shift)
        .collect()
}

fn assert_dispatch(
    actual: ProjectorDispatch,
    stage: ProjectorStage,
    kernel: KernelId,
    output_elements: usize,
    dispatch: [u32; 3],
    uniform_words: [u32; 4],
) {
    assert_eq!(actual.stage, stage);
    assert_eq!(actual.invocation.kernel, kernel);
    assert_eq!(actual.invocation.output_elements, output_elements);
    assert_eq!(actual.invocation.output_bytes, (output_elements * 4) as u64);
    assert_eq!(actual.invocation.dispatch, dispatch);
    assert_eq!(actual.uniform_words, uniform_words);
}

#[test]
fn projector_kernels_are_additive_without_changing_the_frozen_m2_subset() {
    assert_eq!(
        KernelId::M2_PRIMITIVES,
        [
            KernelId::GemmF32,
            KernelId::GemvF32,
            KernelId::LayerNormF32,
            KernelId::RmsNormF32,
            KernelId::SiluF32,
            KernelId::GeluTanhF32,
            KernelId::RopeNeoxF32,
        ]
    );
    assert!(KernelId::ALL.contains(&KernelId::GeluErfF32));
    assert!(KernelId::ALL.contains(&KernelId::ProjectorMerge2x2F32));
    assert_eq!(KernelId::GeluErfF32.as_str(), "gelu_erf_f32");
    assert_eq!(
        KernelId::ProjectorMerge2x2F32.as_str(),
        "projector_merge_2x2_f32"
    );
}

#[test]
fn primitive_erf_gelu_and_merge_plans_fix_mapping_dispatch_uniforms_and_inputs() {
    let gelu = KernelInvocation::GeluErfF32 {
        values: vec![-3.0, -1.0, -0.0, 0.5, 1.0, 3.0],
    };
    let plan = gelu.plan().unwrap();
    assert_eq!(plan.kernel, KernelId::GeluErfF32);
    assert_eq!(plan.output_elements, 6);
    assert_eq!(plan.output_bytes, 24);
    assert_eq!(plan.workgroup_size, [64, 1, 1]);
    assert_eq!(plan.dispatch, [1, 1, 1]);
    assert_eq!(
        gelu.uniform_bytes().unwrap(),
        [6_u32, 0, 0, 0].map(u32::to_le_bytes).concat()
    );

    let source_token_indices = vec![0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let merge = KernelInvocation::ProjectorMerge2x2F32 {
        output_tokens: 4,
        hidden_size: HIDDEN,
        input: (0..16 * HIDDEN).map(|value| value as f32).collect(),
        source_token_indices: source_token_indices.clone(),
    };
    let plan = merge.plan().unwrap();
    assert_eq!(plan.kernel, KernelId::ProjectorMerge2x2F32);
    assert_eq!(plan.output_elements, 48);
    assert_eq!(plan.output_bytes, 192);
    assert_eq!(plan.workgroup_size, [64, 1, 1]);
    assert_eq!(plan.dispatch, [1, 1, 1]);
    assert_eq!(
        merge.uniform_bytes().unwrap(),
        [4_u32, HIDDEN, 48, 0].map(u32::to_le_bytes).concat()
    );
    assert_eq!(
        merge.inputs(),
        vec![
            InvocationInput::F32(match &merge {
                KernelInvocation::ProjectorMerge2x2F32 { input, .. } => input,
                _ => unreachable!(),
            }),
            InvocationInput::U32(&source_token_indices),
        ]
    );
}

#[test]
fn projector_plan_has_exact_five_stage_resident_topology_and_source_mapping() {
    let fixture = Fixture::new();
    let plan = fixture.invocation().plan().unwrap();
    assert_eq!(plan.input_tokens, 16);
    assert_eq!(plan.output_tokens, 4);
    assert_eq!(plan.merged_width, MERGED);
    assert_eq!(
        plan.source_token_indices,
        [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    );
    assert_eq!(ProjectorStage::ALL.len(), 5);
    assert_eq!(
        plan.dispatches.map(|dispatch| dispatch.stage),
        ProjectorStage::ALL
    );

    assert_dispatch(
        plan.dispatches[0],
        ProjectorStage::PreNorm,
        KernelId::LayerNormF32,
        48,
        [1, 1, 1],
        [16, HIDDEN, EPSILON.to_bits(), 0],
    );
    assert_dispatch(
        plan.dispatches[1],
        ProjectorStage::Merge,
        KernelId::ProjectorMerge2x2F32,
        48,
        [1, 1, 1],
        [4, HIDDEN, 48, 0],
    );
    assert_dispatch(
        plan.dispatches[2],
        ProjectorStage::Linear1,
        KernelId::VisionPatchProjectionF32,
        48,
        [1, 1, 1],
        [4, MERGED, MERGED, 0],
    );
    assert_dispatch(
        plan.dispatches[3],
        ProjectorStage::Activation,
        KernelId::GeluErfF32,
        48,
        [1, 1, 1],
        [48, 0, 0, 0],
    );
    assert_dispatch(
        plan.dispatches[4],
        ProjectorStage::Linear2,
        KernelId::VisionPatchProjectionF32,
        20,
        [1, 1, 1],
        [4, MERGED, OUTPUT, 0],
    );
    assert_eq!(plan.resident_intermediate_bytes, 848);
    assert_eq!(plan.resident_weight_bytes, 908);
    assert_eq!(plan.readback_bytes(ProjectorReadback::OutputOnly), 80);
    assert_eq!(plan.readback_bytes(ProjectorReadback::AllStages), 848);
}

#[test]
fn projector_both_linear_stages_use_32_by_32_tiles_on_both_dispatch_axes() {
    let fixture = Fixture::tiled_boundary();
    let plan = fixture.invocation().plan().unwrap();
    assert_eq!(plan.input_tokens, 144);
    assert_eq!(plan.output_tokens, 36);
    assert_eq!(plan.merged_width, 4);

    assert_dispatch(
        plan.dispatches[2],
        ProjectorStage::Linear1,
        KernelId::VisionPatchProjectionF32,
        144,
        [1, 2, 1],
        [36, 4, 4, 0],
    );
    assert_dispatch(
        plan.dispatches[4],
        ProjectorStage::Linear2,
        KernelId::VisionPatchProjectionF32,
        1_188,
        [2, 2, 1],
        [36, 4, 33, 0],
    );
}

#[test]
fn projector_source_mapping_is_temporal_then_spatial_then_local_for_a_multiblock_grid() {
    let mut fixture = Fixture::new();
    fixture.grids = vec![[2, 4, 4]];
    fixture.input = values(32 * HIDDEN as usize, 31);

    let plan = fixture.invocation().plan().unwrap();
    assert_eq!(plan.input_tokens, 32);
    assert_eq!(plan.output_tokens, 8);
    assert_eq!(
        plan.source_token_indices,
        [
            0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15, 16, 17, 20, 21, 18, 19, 22, 23,
            24, 25, 28, 29, 26, 27, 30, 31,
        ]
    );
}

#[test]
fn merge_mapping_is_a_complete_bounded_permutation_and_fails_closed() {
    let valid_input = (0..48).map(|value| value as f32).collect::<Vec<_>>();
    let valid_mapping = vec![0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    for mapping in [
        valid_mapping[..15].to_vec(),
        {
            let mut value = valid_mapping.clone();
            value.push(0);
            value
        },
        {
            let mut value = valid_mapping.clone();
            value[15] = 16;
            value
        },
        {
            let mut value = valid_mapping.clone();
            value[15] = 14;
            value
        },
    ] {
        let error = KernelInvocation::ProjectorMerge2x2F32 {
            output_tokens: 4,
            hidden_size: HIDDEN,
            input: valid_input.clone(),
            source_token_indices: mapping,
        }
        .plan()
        .unwrap_err();
        assert_eq!(error.code(), InvocationErrorCode::InvalidProjectorGeometry);
    }

    for mut input in [valid_input[..47].to_vec(), {
        let mut value = valid_input.clone();
        value.push(0.0);
        value
    }] {
        assert_eq!(
            KernelInvocation::ProjectorMerge2x2F32 {
                output_tokens: 4,
                hidden_size: HIDDEN,
                input: std::mem::take(&mut input),
                source_token_indices: valid_mapping.clone(),
            }
            .plan()
            .unwrap_err()
            .code(),
            InvocationErrorCode::LengthMismatch
        );
    }
}

#[test]
fn projector_prevalidation_rejects_geometry_shapes_and_every_nonfinite_location() {
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut fixture = Fixture::new();
        fixture.epsilon = epsilon;
        assert_eq!(
            fixture.invocation().plan().unwrap_err().code(),
            InvocationErrorCode::InvalidEpsilon
        );
    }
    for grids in [
        vec![],
        vec![[0, 2, 2]],
        vec![[1, 0, 2]],
        vec![[1, 2, 0]],
        vec![[1, 3, 2]],
        vec![[1, 2, 3]],
    ] {
        let mut fixture = Fixture::new();
        fixture.grids = grids;
        assert_eq!(
            fixture.invocation().plan().unwrap_err().code(),
            InvocationErrorCode::InvalidProjectorGeometry
        );
    }
    let mut overflowing = Fixture::new();
    overflowing.grids = vec![[u32::MAX, 2, 2]];
    assert_eq!(
        overflowing.invocation().plan().unwrap_err().code(),
        InvocationErrorCode::ArithmeticOverflow
    );
    let mut collectively_overflowing = Fixture::new();
    collectively_overflowing.grids = vec![[u32::MAX / 4, 2, 2], [u32::MAX / 4, 2, 2]];
    assert_eq!(
        collectively_overflowing
            .invocation()
            .plan()
            .unwrap_err()
            .code(),
        InvocationErrorCode::ArithmeticOverflow
    );
    for operand in 0..7 {
        for oversized in [false, true] {
            let mut fixture = Fixture::new();
            let values = fixture.floating_operand_mut(operand);
            if oversized {
                values.push(0.0);
            } else {
                values.pop();
            }
            assert_eq!(
                fixture.invocation().plan().unwrap_err().code(),
                InvocationErrorCode::LengthMismatch,
                "operand={operand} oversized={oversized}"
            );
        }
        let operand_len = Fixture::new().floating_operand_mut(operand).len();
        for index in [0, operand_len / 2, operand_len - 1] {
            for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut fixture = Fixture::new();
                fixture.floating_operand_mut(operand)[index] = nonfinite;
                assert_eq!(
                    fixture.invocation().plan().unwrap_err().code(),
                    InvocationErrorCode::NonFiniteInput,
                    "operand={operand} index={index} nonfinite={nonfinite:?}"
                );
            }
        }
    }
    for (hidden, output) in [(0, OUTPUT), (HIDDEN, 0)] {
        let mut fixture = Fixture::new();
        fixture.hidden = hidden;
        fixture.output = output;
        assert_eq!(
            fixture.invocation().plan().unwrap_err().code(),
            InvocationErrorCode::ZeroDimension
        );
    }
}

#[test]
fn owned_projector_json_is_strict_stable_and_borrows_the_same_plan() {
    let fixture = Fixture::new();
    let owned = fixture.owned();
    assert_eq!(
        owned.borrowed().plan().unwrap(),
        fixture.invocation().plan().unwrap()
    );
    let first = serde_json::to_string(&owned).unwrap();
    let decoded: OwnedProjectorInvocation = serde_json::from_str(&first).unwrap();
    assert_eq!(decoded, owned);
    assert_eq!(serde_json::to_string(&decoded).unwrap(), first);

    let mut unknown = serde_json::to_value(&owned).unwrap();
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unknown".into(), serde_json::json!(1));
    assert!(serde_json::from_value::<OwnedProjectorInvocation>(unknown).is_err());
    let mut nested_unknown = serde_json::to_value(&owned).unwrap();
    nested_unknown["parameters"]["linear1"]["unknown"] = serde_json::json!(1);
    assert!(serde_json::from_value::<OwnedProjectorInvocation>(nested_unknown).is_err());
}
