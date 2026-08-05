//! M7q1 mixed-precision WGSL contract.
//!
//! Only immutable checkpoint-derived decoder weights use binary16 storage.
//! Generated M-RoPE tables, activations, outputs, workgroup partials, and
//! every accumulator remain F32.

use naga::{Expression, Literal, Module};
use pvlc_runtime_core::KernelId;
use pvlc_wgsl::{BindingKind, full_catalog, validate_catalog, validate_source_contract};

const FP16_KERNELS: [KernelId; 3] = [
    KernelId::RmsNormF16Weights,
    KernelId::GemvTiledF16Weights,
    KernelId::LinearProjectionF16Weights,
];

fn module(kernel: KernelId) -> &'static pvlc_wgsl::KernelModule {
    full_catalog()
        .iter()
        .find(|module| module.spec.kernel == kernel)
        .unwrap_or_else(|| panic!("kernel {kernel} is missing"))
}

fn binding_kinds(kernel: KernelId) -> Vec<BindingKind> {
    module(kernel)
        .spec
        .bindings
        .iter()
        .map(|binding| binding.kind)
        .collect()
}

#[test]
fn fp16_weight_kernels_keep_their_indices_when_later_kernels_append() {
    assert_eq!(KernelId::ALL.len(), 36);
    assert_eq!(&KernelId::ALL[23..26], &FP16_KERNELS);
    assert_eq!(KernelId::ALL[26], KernelId::VisionRope2dF32);
    assert_eq!(
        KernelId::ALL[27],
        KernelId::VisionQkvFusedF16Weights,
        "the new browser kernel must append without renumbering established kernels",
    );
    assert_eq!(
        FP16_KERNELS.map(KernelId::as_str),
        [
            "rms_norm_f16_weights",
            "gemv_tiled_f16_weights",
            "linear_projection_f16_weights",
        ]
    );
    assert_eq!(
        full_catalog()
            .iter()
            .map(|module| module.spec.kernel)
            .collect::<Vec<_>>(),
        KernelId::ALL
    );

    for kernel in FP16_KERNELS {
        let module = module(kernel);
        assert_eq!(module.spec.required_features, ["shader_f16"]);
        assert!(
            module.source.starts_with("enable f16;\n"),
            "{kernel} does not explicitly enable f16"
        );
        validate_source_contract(&module.spec, module.source).unwrap();
    }
    let qkv = module(KernelId::VisionQkvFusedF16Weights);
    assert_eq!(qkv.spec.required_features, ["shader_f16"]);
    assert!(qkv.source.starts_with("enable f16;\n"));
    validate_source_contract(&qkv.spec, qkv.source).unwrap();
    validate_catalog().unwrap();
}

#[test]
fn fp16_weight_abis_keep_activations_and_outputs_in_f32() {
    assert_eq!(
        binding_kinds(KernelId::RmsNormF16Weights),
        [
            BindingKind::StorageReadF32,
            BindingKind::StorageReadF16,
            BindingKind::StorageReadWriteF32,
            BindingKind::Uniform,
        ]
    );
    assert_eq!(
        binding_kinds(KernelId::GemvTiledF16Weights),
        [
            BindingKind::StorageReadVec4F16,
            BindingKind::StorageReadVec4F32,
            BindingKind::StorageReadWriteF32,
            BindingKind::Uniform,
        ]
    );
    assert_eq!(
        binding_kinds(KernelId::LinearProjectionF16Weights),
        [
            BindingKind::StorageReadVec4F32,
            BindingKind::StorageReadVec4F16,
            BindingKind::StorageReadVec4F32,
            BindingKind::StorageReadWriteF32,
            BindingKind::Uniform,
        ]
    );
}

#[test]
fn fp16_loads_are_explicitly_widened_before_f32_accumulation() {
    let rms = module(KernelId::RmsNormF16Weights).source;
    assert!(rms.contains("struct F16Buffer"));
    assert!(rms.contains("data: array<f16>"));
    assert!(rms.contains("f32(weight.data[column])"));
    assert!(rms.contains("var mean_square = 0.0;"));

    let gemv = module(KernelId::GemvTiledF16Weights).source;
    assert!(gemv.contains("data: array<vec4<f16>>"));
    assert!(gemv.contains("shared_vector: array<vec4<f32>"));
    assert!(gemv.contains("partials: array<f32"));
    assert!(gemv.contains("var partial = 0.0;"));
    assert!(gemv.contains("vec4<f32>(matrix.data[row_base + column])"));
    assert!(!gemv.contains("partials: array<f16"));
    assert!(!gemv.contains("var partial: f16"));

    let projection = module(KernelId::LinearProjectionF16Weights).source;
    assert!(projection.contains("data: array<vec4<f16>>"));
    assert!(projection.contains("data: array<vec4<f32>>"));
    assert!(projection.contains("vec4<f32>(weight.data["));
    assert!(projection.contains("fma("));
}

#[test]
fn fp16_linear_projection_keeps_both_checkpoint_layouts_in_the_tiled_kernel() {
    let projection = module(KernelId::LinearProjectionF16Weights).source;
    let reflected = naga::front::wgsl::parse_str(projection).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::SHADER_FLOAT16,
    )
    .validate(&reflected)
    .unwrap();

    assert_eq!(
        module(KernelId::LinearProjectionF16Weights)
            .spec
            .workgroup_size,
        [8, 8, 1],
    );
    assert_eq!(constant_u32(&reflected, "PROJECTION_TILE_ROWS"), 32);
    assert_eq!(constant_u32(&reflected, "PROJECTION_TILE_COLUMNS"), 32);
    assert_eq!(constant_u32(&reflected, "PROJECTION_TILE_DEPTH"), 32);
    assert_eq!(constant_u32(&reflected, "PROJECTION_ROWS_PER_LANE"), 4);
    assert_eq!(constant_u32(&reflected, "PROJECTION_COLUMNS_PER_LANE"), 4);
    assert_eq!(
        constant_u32(&reflected, "PROJECTION_WEIGHT_LAYOUT_INPUT_MAJOR"),
        1,
    );
    assert!(
        projection.contains("if params.padding == PROJECTION_WEIGHT_LAYOUT_INPUT_MAJOR {"),
        "the offline-transposed official stack must select input-major addressing",
    );
    assert!(
        projection.contains("weight.data[input_depth * output_width_vec + global_column_vec]")
            && projection.contains("read_output_major_weight("),
        "input-major must use one contiguous vec4 read while output-major remains exact",
    );
    assert!(
        !projection.contains("return;") && projection.matches("workgroupBarrier();").count() == 2,
        "all 64 lanes must stay in uniform control flow across every depth tile",
    );
}

fn constant_u32(module: &Module, name: &str) -> u32 {
    let constant = module
        .constants
        .iter()
        .map(|(_, constant)| constant)
        .find(|constant| constant.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing {name} constant"));
    match module.global_expressions[constant.init] {
        Expression::Literal(Literal::U32(value)) => value,
        ref expression => panic!("{name} is not a u32 literal: {expression:?}"),
    }
}
