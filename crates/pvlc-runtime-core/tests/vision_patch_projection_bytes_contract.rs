use pvlc_runtime_core::{
    DecoderWeightStorage, KernelId, VisionPatchProjectionBytesDescriptor,
    VisionPatchProjectionBytesErrorCode,
};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f16_bytes(bits: &[u16]) -> Vec<u8> {
    bits.iter().flat_map(|value| value.to_le_bytes()).collect()
}

#[test]
fn realistic_patch_projection_plan_selects_the_storage_specific_kernel_and_exact_buffers() {
    let descriptor = VisionPatchProjectionBytesDescriptor {
        patch_count: 1_276,
        input_width: 588,
        output_width: 1_152,
        weight_storage: DecoderWeightStorage::F32,
    };
    let f32_plan = descriptor
        .plan()
        .expect("official L3 F32 patch projection must plan");
    assert_eq!(f32_plan.kernel, KernelId::VisionPatchProjectionF32);
    assert_eq!(f32_plan.input_bytes, 1_276 * 588 * 4);
    assert_eq!(f32_plan.weight_bytes, 1_152 * 588 * 4);
    assert_eq!(f32_plan.bias_bytes, 1_152 * 4);
    assert_eq!(f32_plan.output_bytes, 1_276 * 1_152 * 4);
    assert_eq!(f32_plan.dispatch, [36, 40, 1]);
    assert!(!f32_plan.requires_shader_f16());

    let f16_plan = VisionPatchProjectionBytesDescriptor {
        weight_storage: DecoderWeightStorage::F16,
        ..descriptor
    }
    .plan()
    .expect("shared-checkpoint FP16 patch projection must plan");
    assert_eq!(f16_plan.kernel, KernelId::LinearProjectionF16Weights);
    assert_eq!(f16_plan.input_bytes, f32_plan.input_bytes);
    assert_eq!(f16_plan.weight_bytes, f32_plan.weight_bytes / 2);
    assert_eq!(f16_plan.bias_bytes, f32_plan.bias_bytes);
    assert_eq!(f16_plan.output_bytes, f32_plan.output_bytes);
    assert_eq!(f16_plan.dispatch, [36, 40, 1]);
    assert!(f16_plan.requires_shader_f16());
}

#[test]
fn production_vision_token_count_rounds_to_fifty_eight_row_tiles() {
    for weight_storage in [DecoderWeightStorage::F32, DecoderWeightStorage::F16] {
        let plan = VisionPatchProjectionBytesDescriptor {
            patch_count: 1_836,
            input_width: 1_152,
            output_width: 4_304,
            weight_storage,
        }
        .plan()
        .unwrap();
        assert_eq!(plan.dispatch, [135, 58, 1]);
    }
}

#[test]
fn exact_finite_operands_and_capabilities_are_admitted_for_both_storage_modes() {
    let descriptor = VisionPatchProjectionBytesDescriptor {
        patch_count: 2,
        input_width: 3,
        output_width: 4,
        weight_storage: DecoderWeightStorage::F32,
    };
    let f32_plan = descriptor.plan().expect("small F32 plan");
    f32_plan
        .validate_capabilities(false, 65_535)
        .expect("F32 projection must not require shader-f16");
    f32_plan
        .validate_operands(
            &f32_bytes(&[1.0, -2.0, 3.0, 4.0, 5.0, -6.0]),
            &f32_bytes(&[
                0.1, 0.2, 0.3, -0.4, 0.5, 0.6, 0.7, -0.8, 0.9, 1.0, 1.1, -1.2,
            ]),
            &f32_bytes(&[0.25, -0.5, 0.75, 1.0]),
        )
        .expect("exact finite F32 operands must be admitted");

    let f16_descriptor = VisionPatchProjectionBytesDescriptor {
        patch_count: 2,
        input_width: 4,
        output_width: 4,
        weight_storage: DecoderWeightStorage::F16,
    };
    let f16_plan = f16_descriptor.plan().expect("small packed F16 plan");
    f16_plan
        .validate_capabilities(true, 65_535)
        .expect("F16 projection must be admitted when shader-f16 is available");
    f16_plan
        .validate_operands(
            &f32_bytes(&[1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 7.0, -8.0]),
            &f16_bytes(&[
                0x2e66, 0x3266, 0x34cd, 0xb666, 0x3800, 0x38cd, 0x399a, 0xba66, 0x3b33, 0x3c00,
                0x3c66, 0xbccd, 0x3d00, 0x3d33, 0x3d66, 0x3d9a,
            ]),
            &f32_bytes(&[0.25, -0.5, 0.75, 1.0]),
        )
        .expect("exact finite F16 weights with F32 activations and bias must be admitted");
}

#[test]
fn every_operand_requires_exact_byte_length_in_both_storage_modes() {
    let f32_descriptor = VisionPatchProjectionBytesDescriptor {
        patch_count: 2,
        input_width: 3,
        output_width: 4,
        weight_storage: DecoderWeightStorage::F32,
    };
    let f16_descriptor = VisionPatchProjectionBytesDescriptor {
        input_width: 4,
        weight_storage: DecoderWeightStorage::F16,
        ..f32_descriptor
    };
    let bias = f32_bytes(&[0.0; 4]);
    for (plan, input, weights) in [
        (
            f32_descriptor.plan().expect("small F32 plan"),
            f32_bytes(&[1.0, -2.0, 3.0, 4.0, 5.0, -6.0]),
            f32_bytes(&[0.5; 12]),
        ),
        (
            f16_descriptor.plan().expect("small packed F16 plan"),
            f32_bytes(&[1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 7.0, -8.0]),
            f16_bytes(&[0x3800; 16]),
        ),
    ] {
        let exact = [&input[..], &weights[..], &bias[..]];
        for operand in 0..exact.len() {
            for delta in [-1_isize, 1] {
                let mut changed = exact.map(|bytes| bytes.to_vec());
                if delta < 0 {
                    changed[operand].truncate(changed[operand].len() - 1);
                } else {
                    changed[operand].push(0);
                }
                let error = plan
                    .validate_operands(&changed[0], &changed[1], &changed[2])
                    .expect_err("short and oversized operands must fail");
                assert_eq!(
                    error.code(),
                    VisionPatchProjectionBytesErrorCode::LengthMismatch,
                    "storage {:?}, operand {operand}, delta {delta}",
                    plan.weight_storage,
                );
            }
        }
    }
}

#[test]
fn every_non_finite_operand_is_rejected_before_gpu_execution() {
    let descriptor = VisionPatchProjectionBytesDescriptor {
        patch_count: 2,
        input_width: 3,
        output_width: 4,
        weight_storage: DecoderWeightStorage::F32,
    };
    let f32_plan = descriptor.plan().expect("small F32 plan");
    let input = f32_bytes(&[1.0, -2.0, 3.0, 4.0, 5.0, -6.0]);
    let weights = f32_bytes(&[0.5; 12]);
    let bias = f32_bytes(&[0.0; 4]);

    for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for operand in 0..3 {
            let mut changed = [input.clone(), weights.clone(), bias.clone()];
            changed[operand][..4].copy_from_slice(&non_finite.to_le_bytes());
            assert_eq!(
                f32_plan
                    .validate_operands(&changed[0], &changed[1], &changed[2])
                    .expect_err("every non-finite F32 operand must fail")
                    .code(),
                VisionPatchProjectionBytesErrorCode::NonFinite,
                "operand {operand}, value {non_finite:?}",
            );
        }
    }

    let f16_plan = VisionPatchProjectionBytesDescriptor {
        input_width: 4,
        weight_storage: DecoderWeightStorage::F16,
        ..descriptor
    }
    .plan()
    .expect("small packed F16 plan");
    let packed_input = f32_bytes(&[1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 7.0, -8.0]);
    for non_finite_bits in [0x7c00_u16, 0xfc00, 0x7e00] {
        let mut non_finite_f16 = f16_bytes(&[0x3c00; 16]);
        non_finite_f16[..2].copy_from_slice(&non_finite_bits.to_le_bytes());
        assert_eq!(
            f16_plan
                .validate_operands(&packed_input, &non_finite_f16, &bias)
                .expect_err("NaN and infinities in F16 weights must fail")
                .code(),
            VisionPatchProjectionBytesErrorCode::NonFinite,
            "F16 bits {non_finite_bits:#06x}",
        );
    }
}

#[test]
fn fp16_projection_rejects_widths_that_cannot_use_packed_vec4_io() {
    for (input_width, output_width) in [(3, 4), (4, 5), (7, 9)] {
        let error = VisionPatchProjectionBytesDescriptor {
            patch_count: 2,
            input_width,
            output_width,
            weight_storage: DecoderWeightStorage::F16,
        }
        .plan()
        .expect_err("unaligned F16 geometry must not reach the packed shader");
        assert_eq!(
            error.code(),
            VisionPatchProjectionBytesErrorCode::InvalidGeometry,
        );
        assert!(error.to_string().contains("multiple of 4"));
    }

    assert!(
        VisionPatchProjectionBytesDescriptor {
            patch_count: 2,
            input_width: 3,
            output_width: 5,
            weight_storage: DecoderWeightStorage::F32,
        }
        .plan()
        .is_ok(),
        "the scalar F32 compatibility path must retain arbitrary widths",
    );
}

#[test]
fn zero_and_overflowing_geometry_have_stable_error_classes() {
    for (patch_count, input_width, output_width) in
        [(0, 588, 1_152), (1_276, 0, 1_152), (1_276, 588, 0)]
    {
        let error = VisionPatchProjectionBytesDescriptor {
            patch_count,
            input_width,
            output_width,
            weight_storage: DecoderWeightStorage::F32,
        }
        .plan()
        .expect_err("zero geometry must fail");
        assert_eq!(
            error.code(),
            VisionPatchProjectionBytesErrorCode::InvalidGeometry
        );
    }

    let error = VisionPatchProjectionBytesDescriptor {
        patch_count: u32::MAX,
        input_width: u32::MAX,
        output_width: u32::MAX,
        weight_storage: DecoderWeightStorage::F32,
    }
    .plan()
    .expect_err("overflowing geometry must fail");
    assert_eq!(error.code(), VisionPatchProjectionBytesErrorCode::Overflow);
}

#[test]
fn dispatch_rounding_and_browser_capabilities_fail_closed() {
    let rounded_f32 = VisionPatchProjectionBytesDescriptor {
        patch_count: 9,
        input_width: 3,
        output_width: 17,
        weight_storage: DecoderWeightStorage::F32,
    }
    .plan()
    .expect("rounded dispatch plan");
    assert_eq!(rounded_f32.dispatch, [1, 1, 1]);

    let rounded_f16 = VisionPatchProjectionBytesDescriptor {
        patch_count: 9,
        input_width: 4,
        output_width: 36,
        weight_storage: DecoderWeightStorage::F16,
    }
    .plan()
    .expect("rounded four-output FP16 dispatch plan");
    assert_eq!(
        rounded_f16.dispatch,
        [2, 1, 1],
        "36 output columns require two 32-column workgroups",
    );

    let exact_limit_descriptor = VisionPatchProjectionBytesDescriptor {
        patch_count: 1,
        input_width: 1,
        output_width: 65_535 * 32,
        weight_storage: DecoderWeightStorage::F32,
    };
    let exact_limit = exact_limit_descriptor
        .plan()
        .expect("exact browser workgroup limit");
    exact_limit
        .validate_capabilities(false, 65_535)
        .expect("exact workgroup limit must pass");

    let over_limit = VisionPatchProjectionBytesDescriptor {
        output_width: 65_535 * 32 + 1,
        ..exact_limit_descriptor
    }
    .plan()
    .expect("over-limit geometry still has a pure plan");
    assert_eq!(over_limit.dispatch[0], 65_536);
    assert_eq!(
        over_limit
            .validate_capabilities(false, 65_535)
            .expect_err("dispatch beyond the browser limit must fail")
            .code(),
        VisionPatchProjectionBytesErrorCode::DispatchLimitExceeded,
    );

    let exact_fp16_limit_descriptor = VisionPatchProjectionBytesDescriptor {
        patch_count: 1,
        input_width: 4,
        output_width: 65_535 * 32,
        weight_storage: DecoderWeightStorage::F16,
    };
    let exact_fp16_limit = exact_fp16_limit_descriptor
        .plan()
        .expect("exact four-output FP16 browser workgroup limit");
    assert_eq!(exact_fp16_limit.dispatch, [65_535, 1, 1]);
    exact_fp16_limit
        .validate_capabilities(true, 65_535)
        .expect("exact FP16 workgroup limit must pass");

    let over_fp16_limit = VisionPatchProjectionBytesDescriptor {
        output_width: 65_535 * 32 + 4,
        ..exact_fp16_limit_descriptor
    }
    .plan()
    .expect("over-limit FP16 geometry still has a pure plan");
    assert_eq!(over_fp16_limit.dispatch, [65_536, 1, 1]);
    assert_eq!(
        over_fp16_limit
            .validate_capabilities(true, 65_535)
            .expect_err("FP16 dispatch beyond the browser limit must fail")
            .code(),
        VisionPatchProjectionBytesErrorCode::DispatchLimitExceeded,
    );

    assert_eq!(
        rounded_f16
            .validate_capabilities(false, 65_535)
            .expect_err("F16 storage without shader-f16 must fail")
            .code(),
        VisionPatchProjectionBytesErrorCode::MissingShaderF16,
    );
}
