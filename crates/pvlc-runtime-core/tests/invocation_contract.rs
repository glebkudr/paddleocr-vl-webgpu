use pvlc_runtime_core::{InvocationErrorCode, KernelId, KernelInvocation};

fn assert_error(invocation: KernelInvocation, expected: InvocationErrorCode) {
    let error = invocation
        .plan()
        .expect_err("malformed invocation must be rejected before GPU submission");
    assert_eq!(error.code(), expected);
}

#[test]
fn kernel_ids_are_a_stable_cross_runtime_protocol() {
    let cases = [
        (KernelId::GemmF32, "gemm_f32"),
        (KernelId::GemvF32, "gemv_f32"),
        (KernelId::LayerNormF32, "layer_norm_f32"),
        (KernelId::RmsNormF32, "rms_norm_f32"),
        (KernelId::SiluF32, "silu_f32"),
        (KernelId::GeluTanhF32, "gelu_tanh_f32"),
        (KernelId::RopeNeoxF32, "rope_neox_f32"),
    ];

    for (kernel, stable_name) in cases {
        assert_eq!(kernel.as_str(), stable_name);
        assert_eq!(
            serde_json::to_string(&kernel).unwrap(),
            format!("\"{stable_name}\"")
        );
        assert_eq!(
            serde_json::from_str::<KernelId>(&format!("\"{stable_name}\"")).unwrap(),
            kernel
        );
    }
    assert!(serde_json::from_str::<KernelId>("\"gemm\"").is_err());
}

#[test]
fn plans_fix_output_sizes_workgroups_and_boundary_dispatch_rounding() {
    let gemm = KernelInvocation::GemmF32 {
        rows: 2,
        inner: 3,
        columns: 9,
        left: vec![1.0; 6],
        right: vec![1.0; 27],
    }
    .plan()
    .unwrap();
    assert_eq!(gemm.kernel, KernelId::GemmF32);
    assert_eq!(gemm.output_elements, 18);
    assert_eq!(gemm.output_bytes, 72);
    assert_eq!(gemm.workgroup_size, [8, 8, 1]);
    assert_eq!(gemm.dispatch, [2, 1, 1]);

    let gemv = KernelInvocation::GemvF32 {
        rows: 65,
        columns: 3,
        matrix: vec![1.0; 195],
        vector: vec![1.0; 3],
    }
    .plan()
    .unwrap();
    assert_eq!(gemv.output_elements, 65);
    assert_eq!(gemv.workgroup_size, [64, 1, 1]);
    assert_eq!(gemv.dispatch, [2, 1, 1]);

    let layer_norm = KernelInvocation::LayerNormF32 {
        rows: 3,
        width: 5,
        input: vec![1.0; 15],
        weight: vec![1.0; 5],
        bias: vec![0.0; 5],
        epsilon: 1.0e-5,
    }
    .plan()
    .unwrap();
    assert_eq!(layer_norm.output_elements, 15);
    assert_eq!(layer_norm.workgroup_size, [64, 1, 1]);
    assert_eq!(layer_norm.dispatch, [1, 1, 1]);

    let rms_norm = KernelInvocation::RmsNormF32 {
        rows: 65,
        width: 1,
        input: vec![1.0; 65],
        weight: vec![1.0],
        epsilon: 1.0e-6,
    }
    .plan()
    .unwrap();
    assert_eq!(rms_norm.dispatch, [2, 1, 1]);

    for (length, expected_x) in [(1, 1), (63, 1), (64, 1), (65, 2), (129, 3)] {
        let plan = KernelInvocation::SiluF32 {
            values: vec![0.0; length],
        }
        .plan()
        .unwrap();
        assert_eq!(plan.dispatch, [expected_x, 1, 1]);
    }

    let rope = KernelInvocation::RopeNeoxF32 {
        rows: 3,
        width: 10,
        rotary_dim: 8,
        positions: vec![0, 1, 7],
        base: 500_000.0,
        values: vec![1.0; 30],
    }
    .plan()
    .unwrap();
    assert_eq!(rope.output_elements, 30);
    assert_eq!(rope.workgroup_size, [64, 1, 1]);
    assert_eq!(rope.dispatch, [1, 1, 1]);
}

#[test]
fn scalar_activations_tile_across_two_dimensions_past_the_webgpu_x_limit() {
    const LENGTH: usize = 65_535 * 64 + 1;
    const ROW_STRIDE: u32 = 32_768 * 64;
    for gelu in [false, true] {
        let invocation = if gelu {
            KernelInvocation::GeluTanhF32 {
                values: vec![0.0; LENGTH],
            }
        } else {
            KernelInvocation::SiluF32 {
                values: vec![0.0; LENGTH],
            }
        };
        let plan = invocation.plan().unwrap();
        assert_eq!(plan.dispatch, [32_768, 2, 1]);
        assert!(plan.dispatch[0] <= 65_535 && plan.dispatch[1] <= 65_535);
        assert_eq!(
            invocation.uniform_bytes().unwrap(),
            [LENGTH as u32, ROW_STRIDE, 0, 0]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn invocation_json_is_strict_stable_and_location_independent() {
    let cases = [
        (
            KernelInvocation::GemmF32 {
                rows: 1,
                inner: 2,
                columns: 1,
                left: vec![1.0, 2.0],
                right: vec![3.0, 4.0],
            },
            concat!(
                r#"{"kernel":"gemm_f32","rows":1,"inner":2,"columns":1,"#,
                r#""left":[1.0,2.0],"right":[3.0,4.0]}"#
            ),
        ),
        (
            KernelInvocation::GemvF32 {
                rows: 2,
                columns: 1,
                matrix: vec![1.0, 2.0],
                vector: vec![3.0],
            },
            concat!(
                r#"{"kernel":"gemv_f32","rows":2,"columns":1,"#,
                r#""matrix":[1.0,2.0],"vector":[3.0]}"#
            ),
        ),
        (
            KernelInvocation::LayerNormF32 {
                rows: 1,
                width: 2,
                input: vec![1.0, 2.0],
                weight: vec![3.0, 4.0],
                bias: vec![5.0, 6.0],
                epsilon: 0.5,
            },
            concat!(
                r#"{"kernel":"layer_norm_f32","rows":1,"width":2,"#,
                r#""input":[1.0,2.0],"weight":[3.0,4.0],"bias":[5.0,6.0],"epsilon":0.5}"#
            ),
        ),
        (
            KernelInvocation::RmsNormF32 {
                rows: 1,
                width: 2,
                input: vec![1.0, 2.0],
                weight: vec![3.0, 4.0],
                epsilon: 0.5,
            },
            concat!(
                r#"{"kernel":"rms_norm_f32","rows":1,"width":2,"#,
                r#""input":[1.0,2.0],"weight":[3.0,4.0],"epsilon":0.5}"#
            ),
        ),
        (
            KernelInvocation::SiluF32 {
                values: vec![-1.0, 0.0, 1.0],
            },
            r#"{"kernel":"silu_f32","values":[-1.0,0.0,1.0]}"#,
        ),
        (
            KernelInvocation::GeluTanhF32 {
                values: vec![-1.0, 0.0, 1.0],
            },
            r#"{"kernel":"gelu_tanh_f32","values":[-1.0,0.0,1.0]}"#,
        ),
        (
            KernelInvocation::RopeNeoxF32 {
                rows: 1,
                width: 2,
                rotary_dim: 2,
                positions: vec![7],
                base: 10_000.0,
                values: vec![1.0, 2.0],
            },
            concat!(
                r#"{"kernel":"rope_neox_f32","rows":1,"width":2,"rotary_dim":2,"#,
                r#""positions":[7],"base":10000.0,"values":[1.0,2.0]}"#
            ),
        ),
    ];
    for (invocation, canonical) in cases {
        assert_eq!(serde_json::to_string(&invocation).unwrap(), canonical);
        assert_eq!(
            serde_json::from_str::<KernelInvocation>(canonical).unwrap(),
            invocation
        );
    }

    let gemm = concat!(
        r#"{"kernel":"gemm_f32","rows":1,"inner":2,"columns":1,"#,
        r#""left":[1.0,2.0],"right":[3.0,4.0]}"#
    );
    let unknown_field = gemm.replace(
        r#""right":[3.0,4.0]"#,
        r#""right":[3.0,4.0],"transpose":true"#,
    );
    assert!(serde_json::from_str::<KernelInvocation>(&unknown_field).is_err());
    assert!(
        serde_json::from_str::<KernelInvocation>(r#"{"kernel":"unknown","values":[1.0]}"#).is_err()
    );
    assert!(
        serde_json::from_str::<KernelInvocation>(
            r#"{"kernel":"silu_f32","values":[1.0],"values":[2.0]}"#
        )
        .is_err()
    );
    assert!(
        serde_json::from_str::<KernelInvocation>(
            r#"{"kernel":"gemm_f32","rows":1,"inner":1,"columns":1,"left":[1.0]}"#
        )
        .is_err()
    );
}

#[test]
fn host_uniform_bytes_match_the_independent_little_endian_shader_abi() {
    fn words(values: [u32; 4]) -> Vec<u8> {
        values.into_iter().flat_map(u32::to_le_bytes).collect()
    }

    let cases = [
        (
            KernelInvocation::GemmF32 {
                rows: 2,
                inner: 3,
                columns: 9,
                left: vec![1.0; 6],
                right: vec![1.0; 27],
            },
            words([2, 3, 9, 0]),
        ),
        (
            KernelInvocation::GemvF32 {
                rows: 65,
                columns: 3,
                matrix: vec![1.0; 195],
                vector: vec![1.0; 3],
            },
            words([65, 3, 0, 0]),
        ),
        (
            KernelInvocation::LayerNormF32 {
                rows: 3,
                width: 5,
                input: vec![1.0; 15],
                weight: vec![1.0; 5],
                bias: vec![0.0; 5],
                epsilon: 0.5,
            },
            words([3, 5, 0.5_f32.to_bits(), 0]),
        ),
        (
            KernelInvocation::RmsNormF32 {
                rows: 3,
                width: 5,
                input: vec![1.0; 15],
                weight: vec![1.0; 5],
                epsilon: 0.25,
            },
            words([3, 5, 0.25_f32.to_bits(), 0]),
        ),
        (
            KernelInvocation::SiluF32 {
                values: vec![1.0; 65],
            },
            words([65, 0, 0, 0]),
        ),
        (
            KernelInvocation::GeluTanhF32 {
                values: vec![1.0; 129],
            },
            words([129, 0, 0, 0]),
        ),
        (
            KernelInvocation::RopeNeoxF32 {
                rows: 3,
                width: 10,
                rotary_dim: 8,
                positions: vec![0, 1, 7],
                base: 500_000.0,
                values: vec![1.0; 30],
            },
            words([3, 10, 8, 500_000.0_f32.to_bits()]),
        ),
    ];

    for (invocation, expected) in cases {
        assert_eq!(invocation.uniform_bytes().unwrap(), expected);
        assert_eq!(expected.len(), 16);
    }
}

#[test]
fn every_request_is_validated_before_any_gpu_allocation_or_dispatch() {
    assert_error(
        KernelInvocation::GemmF32 {
            rows: 0,
            inner: 1,
            columns: 1,
            left: vec![],
            right: vec![1.0],
        },
        InvocationErrorCode::ZeroDimension,
    );
    assert_error(
        KernelInvocation::GemmF32 {
            rows: 1,
            inner: 2,
            columns: 1,
            left: vec![1.0],
            right: vec![1.0, 2.0],
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_error(
        KernelInvocation::GemvF32 {
            rows: 2,
            columns: 2,
            matrix: vec![1.0; 4],
            vector: vec![1.0],
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_error(
        KernelInvocation::LayerNormF32 {
            rows: 1,
            width: 2,
            input: vec![1.0, 2.0],
            weight: vec![1.0, 1.0],
            bias: vec![0.0],
            epsilon: 1.0e-5,
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_error(
        KernelInvocation::LayerNormF32 {
            rows: 1,
            width: 1,
            input: vec![1.0],
            weight: vec![1.0],
            bias: vec![0.0],
            epsilon: 0.0,
        },
        InvocationErrorCode::InvalidEpsilon,
    );
    assert_error(
        KernelInvocation::RmsNormF32 {
            rows: 1,
            width: 1,
            input: vec![1.0],
            weight: vec![1.0],
            epsilon: f32::INFINITY,
        },
        InvocationErrorCode::InvalidEpsilon,
    );
    assert_error(
        KernelInvocation::SiluF32 { values: vec![] },
        InvocationErrorCode::ZeroDimension,
    );
    assert_error(
        KernelInvocation::GeluTanhF32 {
            values: vec![f32::NAN],
        },
        InvocationErrorCode::NonFiniteInput,
    );
    assert_error(
        KernelInvocation::RopeNeoxF32 {
            rows: 1,
            width: 4,
            rotary_dim: 3,
            positions: vec![0],
            base: 10_000.0,
            values: vec![1.0; 4],
        },
        InvocationErrorCode::InvalidRotaryDimension,
    );
    assert_error(
        KernelInvocation::RopeNeoxF32 {
            rows: 1,
            width: 4,
            rotary_dim: 6,
            positions: vec![0],
            base: 10_000.0,
            values: vec![1.0; 4],
        },
        InvocationErrorCode::InvalidRotaryDimension,
    );
    assert_error(
        KernelInvocation::RopeNeoxF32 {
            rows: 2,
            width: 4,
            rotary_dim: 4,
            positions: vec![0],
            base: 10_000.0,
            values: vec![1.0; 8],
        },
        InvocationErrorCode::LengthMismatch,
    );
    assert_error(
        KernelInvocation::RopeNeoxF32 {
            rows: 1,
            width: 4,
            rotary_dim: 4,
            positions: vec![0],
            base: 0.0,
            values: vec![1.0; 4],
        },
        InvocationErrorCode::InvalidRopeBase,
    );
    assert_error(
        KernelInvocation::GemmF32 {
            rows: u32::MAX,
            inner: 1,
            columns: u32::MAX,
            left: vec![],
            right: vec![],
        },
        InvocationErrorCode::ArithmeticOverflow,
    );
}

#[test]
fn all_floating_inputs_are_checked_not_only_the_primary_buffer() {
    let cases = [
        KernelInvocation::GemmF32 {
            rows: 1,
            inner: 1,
            columns: 1,
            left: vec![f32::NAN],
            right: vec![1.0],
        },
        KernelInvocation::GemmF32 {
            rows: 1,
            inner: 1,
            columns: 1,
            left: vec![1.0],
            right: vec![f32::INFINITY],
        },
        KernelInvocation::GemvF32 {
            rows: 1,
            columns: 1,
            matrix: vec![f32::NAN],
            vector: vec![1.0],
        },
        KernelInvocation::GemvF32 {
            rows: 1,
            columns: 1,
            matrix: vec![1.0],
            vector: vec![f32::NEG_INFINITY],
        },
        KernelInvocation::LayerNormF32 {
            rows: 1,
            width: 1,
            input: vec![f32::NAN],
            weight: vec![1.0],
            bias: vec![0.0],
            epsilon: 1.0e-5,
        },
        KernelInvocation::LayerNormF32 {
            rows: 1,
            width: 1,
            input: vec![1.0],
            weight: vec![f32::NAN],
            bias: vec![0.0],
            epsilon: 1.0e-5,
        },
        KernelInvocation::LayerNormF32 {
            rows: 1,
            width: 1,
            input: vec![1.0],
            weight: vec![1.0],
            bias: vec![f32::INFINITY],
            epsilon: 1.0e-5,
        },
        KernelInvocation::RmsNormF32 {
            rows: 1,
            width: 1,
            input: vec![f32::NEG_INFINITY],
            weight: vec![1.0],
            epsilon: 1.0e-5,
        },
        KernelInvocation::RmsNormF32 {
            rows: 1,
            width: 1,
            input: vec![1.0],
            weight: vec![f32::NAN],
            epsilon: 1.0e-5,
        },
        KernelInvocation::SiluF32 {
            values: vec![f32::NAN],
        },
        KernelInvocation::GeluTanhF32 {
            values: vec![f32::INFINITY],
        },
        KernelInvocation::RopeNeoxF32 {
            rows: 1,
            width: 2,
            rotary_dim: 2,
            positions: vec![0],
            base: 10_000.0,
            values: vec![1.0, f32::INFINITY],
        },
    ];

    for invocation in cases {
        assert_error(invocation, InvocationErrorCode::NonFiniteInput);
    }

    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_error(
            KernelInvocation::LayerNormF32 {
                rows: 1,
                width: 1,
                input: vec![1.0],
                weight: vec![1.0],
                bias: vec![0.0],
                epsilon,
            },
            InvocationErrorCode::InvalidEpsilon,
        );
        assert_error(
            KernelInvocation::RmsNormF32 {
                rows: 1,
                width: 1,
                input: vec![1.0],
                weight: vec![1.0],
                epsilon,
            },
            InvocationErrorCode::InvalidEpsilon,
        );
    }
    for base in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_error(
            KernelInvocation::RopeNeoxF32 {
                rows: 1,
                width: 2,
                rotary_dim: 2,
                positions: vec![0],
                base,
                values: vec![1.0, 2.0],
            },
            InvocationErrorCode::InvalidRopeBase,
        );
    }
}
