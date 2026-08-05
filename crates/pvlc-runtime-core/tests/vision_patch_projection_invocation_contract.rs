use pvlc_runtime_core::{InvocationErrorCode, InvocationInput, KernelId, KernelInvocation};

fn invocation(patch_count: u32, input_width: u32, output_width: u32) -> KernelInvocation {
    KernelInvocation::VisionPatchProjectionF32 {
        patch_count,
        input_width,
        output_width,
        input: vec![0.25; (u64::from(patch_count) * u64::from(input_width)) as usize],
        weight: vec![-0.5; (u64::from(output_width) * u64::from(input_width)) as usize],
        bias: vec![1.0; output_width as usize],
    }
}

fn assert_error(invocation: KernelInvocation, expected: InvocationErrorCode) {
    assert_eq!(invocation.plan().unwrap_err().code(), expected);
}

#[test]
fn patch_projection_is_an_additive_protocol_after_attention_and_the_frozen_m2_subset() {
    assert_eq!(
        KernelId::VisionPatchProjectionF32.as_str(),
        "vision_patch_projection_f32"
    );
    assert_eq!(
        serde_json::to_string(&KernelId::VisionPatchProjectionF32).unwrap(),
        "\"vision_patch_projection_f32\""
    );
    assert_eq!(
        serde_json::from_str::<KernelId>("\"vision_patch_projection_f32\"").unwrap(),
        KernelId::VisionPatchProjectionF32
    );
    assert_eq!(KernelId::M2_PRIMITIVES.len(), 7);
    assert_eq!(&KernelId::ALL[..7], KernelId::M2_PRIMITIVES.as_slice());
    assert_eq!(KernelId::ALL[7], KernelId::VisionAttentionF32);
    assert_eq!(KernelId::ALL[8], KernelId::VisionPatchProjectionF32);
}

#[test]
fn patch_projection_plan_fixes_output_dispatch_uniforms_and_checkpoint_binding_order() {
    let invocation = invocation(17, 588, 1_152);
    let plan = invocation.plan().unwrap();
    assert_eq!(plan.kernel, KernelId::VisionPatchProjectionF32);
    assert_eq!(plan.output_elements, 17 * 1_152);
    assert_eq!(plan.output_bytes, (17 * 1_152 * 4) as u64);
    assert_eq!(plan.workgroup_size, [8, 8, 1]);
    assert_eq!(
        plan.dispatch,
        [36, 1, 1],
        "64 lanes must cooperatively produce one 32x32 output tile",
    );
    assert_eq!(
        invocation.uniform_bytes().unwrap(),
        [17_u32, 588, 1_152, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );

    let KernelInvocation::VisionPatchProjectionF32 {
        input,
        weight,
        bias,
        ..
    } = &invocation
    else {
        unreachable!()
    };
    assert_eq!(
        invocation.inputs(),
        vec![
            InvocationInput::F32(input),
            InvocationInput::F32(weight),
            InvocationInput::F32(bias),
        ]
    );
    assert_eq!(invocation.output_initializer(), None);
}

#[test]
fn patch_projection_json_is_strict_stable_and_preserves_checkpoint_layout() {
    let invocation = KernelInvocation::VisionPatchProjectionF32 {
        patch_count: 2,
        input_width: 2,
        output_width: 2,
        input: vec![1.0, 2.0, 3.0, 4.0],
        weight: vec![1.0, 2.0, -1.0, 0.5],
        bias: vec![0.25, -0.5],
    };
    let canonical = concat!(
        r#"{"kernel":"vision_patch_projection_f32","patch_count":2,"input_width":2,"output_width":2,"#,
        r#""input":[1.0,2.0,3.0,4.0],"weight":[1.0,2.0,-1.0,0.5],"bias":[0.25,-0.5]}"#
    );
    assert_eq!(serde_json::to_string(&invocation).unwrap(), canonical);
    assert_eq!(
        serde_json::from_str::<KernelInvocation>(canonical).unwrap(),
        invocation
    );
    let unknown = canonical.replace("\"bias\":[0.25,-0.5]", "\"bias\":[0.25,-0.5],\"stride\":14");
    assert!(serde_json::from_str::<KernelInvocation>(&unknown).is_err());
}

#[test]
fn patch_projection_rejects_zero_overflow_and_every_malformed_operand() {
    for dimensions in [(0, 3, 4), (2, 0, 4), (2, 3, 0)] {
        assert_error(
            invocation(dimensions.0, dimensions.1, dimensions.2),
            InvocationErrorCode::ZeroDimension,
        );
    }

    for operand in 0..3 {
        for oversized in [false, true] {
            let KernelInvocation::VisionPatchProjectionF32 {
                patch_count,
                input_width,
                output_width,
                mut input,
                mut weight,
                mut bias,
            } = invocation(2, 3, 4)
            else {
                unreachable!()
            };
            let selected = match operand {
                0 => &mut input,
                1 => &mut weight,
                2 => &mut bias,
                _ => unreachable!(),
            };
            if oversized {
                selected.push(0.0);
            } else {
                selected.pop();
            }
            assert_error(
                KernelInvocation::VisionPatchProjectionF32 {
                    patch_count,
                    input_width,
                    output_width,
                    input,
                    weight,
                    bias,
                },
                InvocationErrorCode::LengthMismatch,
            );
        }
    }

    assert_error(
        KernelInvocation::VisionPatchProjectionF32 {
            patch_count: u32::MAX,
            input_width: u32::MAX,
            output_width: u32::MAX,
            input: vec![],
            weight: vec![],
            bias: vec![],
        },
        InvocationErrorCode::ArithmeticOverflow,
    );
}

#[test]
fn patch_projection_rejects_nonfinite_values_in_input_weight_and_bias() {
    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for operand in 0..3 {
            let KernelInvocation::VisionPatchProjectionF32 {
                patch_count,
                input_width,
                output_width,
                mut input,
                mut weight,
                mut bias,
            } = invocation(2, 3, 4)
            else {
                unreachable!()
            };
            [&mut input, &mut weight, &mut bias][operand][0] = nonfinite;
            assert_error(
                KernelInvocation::VisionPatchProjectionF32 {
                    patch_count,
                    input_width,
                    output_width,
                    input,
                    weight,
                    bias,
                },
                InvocationErrorCode::NonFiniteInput,
            );
        }
    }
}
