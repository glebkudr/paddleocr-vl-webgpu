use pvlc_runtime_core::{InvocationErrorCode, InvocationInput, KernelId, KernelInvocation};

fn invocation(length: usize) -> KernelInvocation {
    KernelInvocation::AddF32 {
        left: vec![0.25; length],
        right: vec![-0.5; length],
    }
}

fn assert_error(invocation: KernelInvocation, expected: InvocationErrorCode) {
    assert_eq!(invocation.plan().unwrap_err().code(), expected);
}

#[test]
fn add_is_an_additive_protocol_after_the_frozen_m2_and_existing_m3_kernels() {
    assert_eq!(KernelId::AddF32.as_str(), "add_f32");
    assert_eq!(
        serde_json::to_string(&KernelId::AddF32).unwrap(),
        "\"add_f32\""
    );
    assert_eq!(
        serde_json::from_str::<KernelId>("\"add_f32\"").unwrap(),
        KernelId::AddF32
    );
    assert_eq!(KernelId::M2_PRIMITIVES.len(), 7);
    assert_eq!(&KernelId::ALL[..7], KernelId::M2_PRIMITIVES.as_slice());
    assert_eq!(KernelId::ALL[7], KernelId::VisionAttentionF32);
    assert_eq!(KernelId::ALL[8], KernelId::VisionPatchProjectionF32);
    assert_eq!(KernelId::ALL[9], KernelId::AddF32);
    assert_eq!(KernelId::ALL[10], KernelId::GeluErfF32);
    assert_eq!(KernelId::ALL[11], KernelId::ProjectorMerge2x2F32);
}

#[test]
fn add_plan_fixes_output_dispatch_uniforms_and_binding_order() {
    let invocation = invocation(65);
    let plan = invocation.plan().unwrap();
    assert_eq!(plan.kernel, KernelId::AddF32);
    assert_eq!(plan.output_elements, 65);
    assert_eq!(plan.output_bytes, 65 * 4);
    assert_eq!(plan.workgroup_size, [64, 1, 1]);
    assert_eq!(plan.dispatch, [2, 1, 1]);
    assert_eq!(
        invocation.uniform_bytes().unwrap(),
        [65_u32, 0, 0, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );

    let KernelInvocation::AddF32 { left, right } = &invocation else {
        unreachable!()
    };
    assert_eq!(
        invocation.inputs(),
        vec![InvocationInput::F32(left), InvocationInput::F32(right)]
    );
    assert_eq!(invocation.output_initializer(), None);
}

#[test]
fn add_json_is_strict_stable_and_preserves_operand_order() {
    let invocation = KernelInvocation::AddF32 {
        left: vec![1.0, -2.0, 3.0],
        right: vec![0.5, 2.0, -1.0],
    };
    let canonical = r#"{"kernel":"add_f32","left":[1.0,-2.0,3.0],"right":[0.5,2.0,-1.0]}"#;
    assert_eq!(serde_json::to_string(&invocation).unwrap(), canonical);
    assert_eq!(
        serde_json::from_str::<KernelInvocation>(canonical).unwrap(),
        invocation
    );
    let unknown = canonical.replace("]}", "],\"scale\":1.0}");
    assert!(serde_json::from_str::<KernelInvocation>(&unknown).is_err());
}

#[test]
fn add_rejects_empty_mismatched_and_nonfinite_operands_with_exact_codes() {
    assert_error(invocation(0), InvocationErrorCode::ZeroDimension);
    for (left, right) in [
        (vec![1.0], vec![]),
        (vec![], vec![1.0]),
        (vec![1.0], vec![1.0, 2.0]),
    ] {
        assert_error(
            KernelInvocation::AddF32 { left, right },
            InvocationErrorCode::LengthMismatch,
        );
    }

    for nonfinite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for operand in 0..2 {
            let (mut left, mut right) = (vec![1.0, 2.0], vec![3.0, 4.0]);
            [&mut left, &mut right][operand][1] = nonfinite;
            assert_error(
                KernelInvocation::AddF32 { left, right },
                InvocationErrorCode::NonFiniteInput,
            );
        }
    }
}
