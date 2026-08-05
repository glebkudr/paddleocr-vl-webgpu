use std::collections::BTreeMap;

use naga::{
    AddressSpace, ArraySize, Handle, Type, TypeInner,
    valid::{Capabilities, ValidationFlags, Validator},
};
use pvlc_runtime_core::KernelId;
use pvlc_wgsl::{BindingKind, validate_source_contract};

const ROW_TILE: u32 = 16;
const COLUMN_TILE: u32 = 32;
const DEPTH_TILE: u32 = 16;

fn kernel() -> &'static pvlc_wgsl::KernelModule {
    pvlc_wgsl::module(KernelId::VisionQkvFusedF16Weights).expect("tiled FP16-weight QKV kernel")
}

#[test]
fn fused_fp16_qkv_uses_the_portable_eight_storage_binding_abi() {
    let kernel = kernel();
    assert_eq!(kernel.spec.workgroup_size, [8, 8, 1]);
    assert_eq!(kernel.spec.required_features, ["shader_f16"]);
    assert_eq!(
        kernel
            .spec
            .bindings
            .iter()
            .map(|binding| binding.kind)
            .collect::<Vec<_>>(),
        [
            BindingKind::StorageReadVec4F32,
            BindingKind::StorageReadVec4F16,
            BindingKind::StorageReadVec4F32,
            BindingKind::StorageReadVec4F16,
            BindingKind::StorageReadVec4F32,
            BindingKind::StorageReadVec4F16,
            BindingKind::StorageReadVec4F32,
            BindingKind::StorageReadWriteF32,
            BindingKind::Uniform,
        ],
    );
    assert_eq!(
        kernel.spec.bindings.len() - 1,
        8,
        "the kernel must fit the minimum WebGPU storage-buffer-per-stage limit",
    );
    assert_eq!(
        kernel
            .spec
            .uniform_fields
            .iter()
            .map(|field| (field.name, field.offset))
            .collect::<Vec<_>>(),
        [
            ("tokens", 0),
            ("input_width", 4),
            ("output_width", 8),
            ("plane_stride_elements", 12),
        ],
    );
    validate_source_contract(&kernel.spec, kernel.source).unwrap();
}

#[test]
fn fused_fp16_qkv_reuses_one_input_tile_across_three_input_major_weight_tiles() {
    let kernel = kernel();
    let module = naga::front::wgsl::parse_str(kernel.source).unwrap();
    Validator::new(ValidationFlags::all(), Capabilities::SHADER_FLOAT16)
        .validate(&module)
        .unwrap();

    for (name, expected) in [
        ("QKV_TILE_ROWS", ROW_TILE),
        ("QKV_TILE_COLUMNS", COLUMN_TILE),
        ("QKV_TILE_DEPTH", DEPTH_TILE),
        ("QKV_ROWS_PER_LANE", 2),
        ("QKV_WORKGROUP_SIZE", 64),
    ] {
        assert_eq!(constant_u32(&module, name), expected, "{name} drifted");
    }
    assert_eq!(
        module
            .global_variables
            .iter()
            .filter(|(_, global)| matches!(global.space, AddressSpace::WorkGroup))
            .map(|(_, global)| (
                global.name.as_deref().unwrap_or("<unnamed>"),
                fixed_type_bytes(&module, global.ty),
            ))
            .collect::<BTreeMap<_, _>>(),
        BTreeMap::from([
            ("input_tile", (ROW_TILE * DEPTH_TILE * 4) as usize),
            ("query_weight_tile", (DEPTH_TILE * COLUMN_TILE * 4) as usize,),
            ("key_weight_tile", (DEPTH_TILE * COLUMN_TILE * 4) as usize,),
            ("value_weight_tile", (DEPTH_TILE * COLUMN_TILE * 4) as usize,),
        ]),
        "one 16x16 input tile must be shared by three 16x32 weight tiles",
    );
    assert_eq!(kernel.source.matches("input.data[").count(), 1);
    assert_eq!(kernel.source.matches("query_weight.data[").count(), 1);
    assert_eq!(kernel.source.matches("key_weight.data[").count(), 1);
    assert_eq!(kernel.source.matches("value_weight.data[").count(), 1);
    assert_eq!(kernel.source.matches("workgroupBarrier();").count(), 2);
    assert!(!kernel.source.contains("global_invocation_id"));
    assert!(!kernel.source.contains("workgroup_id.z"));
    assert!(!kernel.source.contains("read_output_major"));
    assert!(
        kernel
            .source
            .contains("input_depth * output_width_vec + global_column_vec"),
        "all matrix reads must use the checkpoint's packed input-major layout",
    );
    for guard in [
        "global_row < params.tokens",
        "local_x < 4u",
        "input_depth < params.input_width",
        "global_column_vec < output_width_vec",
        "output_row < params.tokens",
        "output_column_base + 3u < params.output_width",
    ] {
        assert!(kernel.source.contains(guard), "missing tail guard {guard}");
    }
    assert!(
        !kernel.source.contains("return;"),
        "edge lanes must remain active across both workgroup barriers",
    );
    audit_fused_fp16_qkv_dataflow(kernel.source).unwrap();
}

#[test]
fn fused_fp16_qkv_keeps_accumulation_and_three_output_planes_in_f32() {
    let source = kernel().source;
    for projection in ["query", "key", "value"] {
        assert!(
            source.contains(&format!(
                "var {projection}_accumulators: array<vec4<f32>, 2>;"
            )),
            "{projection} accumulation must remain F32",
        );
        assert!(
            source.contains(&format!("{projection}_accumulators[row_offset] = fma(")),
            "{projection} must consume the common activation tile",
        );
    }
    assert!(source.contains("let query_plane = 0u;"));
    assert!(source.contains("let key_plane = params.plane_stride_elements;"));
    assert!(source.contains("let value_plane = params.plane_stride_elements * 2u;"));
    assert!(source.contains("query_plane + output_index"));
    assert!(source.contains("key_plane + output_index"));
    assert!(source.contains("value_plane + output_index"));
    assert!(!source.contains("array<f16>"));
    assert!(!source.contains("vec4<f16>(0.0"));
}

#[test]
fn input_major_three_plane_indexing_matches_three_independent_projections() {
    let tokens = 5;
    let input_width = 36;
    let output_width = 12;
    let input = (0..tokens * input_width)
        .map(|index| ((index * 17 % 29) as f32 - 14.0) / 8.0)
        .collect::<Vec<_>>();
    let weights = (0..3)
        .map(|projection| {
            (0..input_width * output_width)
                .map(|index| (((index * 11 + projection * 7) % 23) as f32 - 11.0) / 16.0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let biases = (0..3)
        .map(|projection| {
            (0..output_width)
                .map(|column| (projection * 3 + column) as f32 / 32.0)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let expected = (0..3)
        .flat_map(|projection| {
            independent_projection(
                &input,
                &weights[projection],
                &biases[projection],
                tokens,
                input_width,
                output_width,
            )
        })
        .collect::<Vec<_>>();
    let actual = fused_tiled_projection(
        &input,
        [&weights[0], &weights[1], &weights[2]],
        [&biases[0], &biases[1], &biases[2]],
        tokens,
        input_width,
        output_width,
    );
    assert_eq!(actual, expected);
}

#[test]
fn semantic_audit_rejects_disconnected_or_misindexed_production_qkv_dataflow() {
    let source = kernel().source;
    audit_fused_fp16_qkv_dataflow(source).unwrap();
    for (from, to) in [
        (
            "input.data[global_row * input_width_vec + input_depth_vec]",
            "input.data[input_depth_vec]",
        ),
        (
            "query_weight.data[input_depth * output_width_vec + global_column_vec]",
            "query_weight.data[global_column_vec * params.input_width + input_depth]",
        ),
        (
            "key_accumulators[row_offset] = fma(",
            "key_accumulators[row_offset] = key_accumulators[row_offset] + vec4<f32>(",
        ),
        (
            "key_coefficient2,\n                    key_accumulators[row_offset]",
            "query_coefficient2,\n                    key_accumulators[row_offset]",
        ),
        (
            "let value_values = value_accumulators[row_offset] + value_bias_value;",
            "let value_values = value_bias_value;",
        ),
        (
            "output.data[key_plane + output_index + 3u] = key_values.w;",
            "output.data[query_plane + output_index + 3u] = key_values.w;",
        ),
        (
            "output.data[value_plane + output_index + 1u] = value_values.y;",
            "output.data[value_plane + output_index + 1u] = value_values.x;",
        ),
    ] {
        assert!(source.contains(from), "mutation fixture drifted: {from}");
        assert!(
            audit_fused_fp16_qkv_dataflow(&source.replace(from, to)).is_err(),
            "semantic audit accepted mutation {from:?} -> {to:?}",
        );
    }
}

fn independent_projection(
    input: &[f32],
    input_major_weight: &[f32],
    bias: &[f32],
    tokens: usize,
    input_width: usize,
    output_width: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; tokens * output_width];
    for token in 0..tokens {
        for column in 0..output_width {
            let mut value = 0.0;
            for depth in 0..input_width {
                value = input[token * input_width + depth]
                    .mul_add(input_major_weight[depth * output_width + column], value);
            }
            output[token * output_width + column] = value + bias[column];
        }
    }
    output
}

fn fused_tiled_projection(
    input: &[f32],
    input_major_weights: [&[f32]; 3],
    biases: [&[f32]; 3],
    tokens: usize,
    input_width: usize,
    output_width: usize,
) -> Vec<f32> {
    let plane_stride = tokens * output_width;
    let mut output = vec![0.0; plane_stride * 3];
    for row_base in (0..tokens).step_by(ROW_TILE as usize) {
        for column_base in (0..output_width).step_by(COLUMN_TILE as usize) {
            for row in row_base..tokens.min(row_base + ROW_TILE as usize) {
                for column in column_base..output_width.min(column_base + COLUMN_TILE as usize) {
                    let mut values = [0.0; 3];
                    for depth_base in (0..input_width).step_by(DEPTH_TILE as usize) {
                        for depth in depth_base..input_width.min(depth_base + DEPTH_TILE as usize) {
                            let activation = input[row * input_width + depth];
                            for projection in 0..3 {
                                values[projection] = activation.mul_add(
                                    input_major_weights[projection][depth * output_width + column],
                                    values[projection],
                                );
                            }
                        }
                    }
                    for projection in 0..3 {
                        output[projection * plane_stride + row * output_width + column] =
                            values[projection] + biases[projection][column];
                    }
                }
            }
        }
    }
    output
}

fn audit_fused_fp16_qkv_dataflow(source: &str) -> Result<(), String> {
    for required in [
        "input_tile[tile_row][local_x] = input.data[global_row * input_width_vec + input_depth_vec];",
        "query_weight_tile[tile_depth][local_x] = vec4<f32>(query_weight.data[input_depth * output_width_vec + global_column_vec]);",
        "key_weight_tile[tile_depth][local_x] = vec4<f32>(key_weight.data[input_depth * output_width_vec + global_column_vec]);",
        "value_weight_tile[tile_depth][local_x] = vec4<f32>(value_weight.data[input_depth * output_width_vec + global_column_vec]);",
        "let activation = input_tile[local_y * 4u + row_offset][depth_vector];",
        "let query_values = query_accumulators[row_offset] + query_bias_value;",
        "let key_values = key_accumulators[row_offset] + key_bias_value;",
        "let value_values = value_accumulators[row_offset] + value_bias_value;",
    ] {
        if !source.contains(required) {
            return Err(format!("missing fused FP16 QKV dataflow edge {required:?}"));
        }
    }
    for projection in ["query", "key", "value"] {
        for (component_index, component) in ["x", "y", "z", "w"].into_iter().enumerate() {
            let coefficient = format!(
                "let {projection}_coefficient{component_index} = \
                 {projection}_weight_tile[depth_vector * 4u + {component_index}u][local_x];"
            );
            if !source.contains(&coefficient) {
                return Err(format!("missing coefficient edge {coefficient:?}"));
            }
            let fma = format!(
                "{projection}_accumulators[row_offset] = fma(\
                 vec4<f32>(activation.{component}), \
                 {projection}_coefficient{component_index}, \
                 {projection}_accumulators[row_offset],);"
            );
            if !compact_contains(source, &fma) {
                return Err(format!("missing FMA edge {fma:?}"));
            }
            let store = format!(
                "output.data[{projection}_plane + output_index + {component_index}u] = \
                 {projection}_values.{component};"
            );
            if !source.contains(&store) {
                return Err(format!("missing output edge {store:?}"));
            }
        }
    }
    let compute = source
        .split_once("for (var depth_vector = 0u; depth_vector < 4u;")
        .and_then(|(_, tail)| tail.split_once("\n        }\n").map(|(body, _)| body))
        .ok_or_else(|| "missing packed depth-vector compute loop".to_owned())?;
    if compute.contains("input.data[")
        || compute.contains("query_weight.data[")
        || compute.contains("key_weight.data[")
        || compute.contains("value_weight.data[")
    {
        return Err("inner compute loop bypasses shared tiles".to_owned());
    }
    Ok(())
}

fn compact_contains(source: &str, needle: &str) -> bool {
    let compact = |value: &str| {
        value
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>()
    };
    compact(source).contains(&compact(needle))
}

fn constant_u32(module: &naga::Module, name: &str) -> u32 {
    let constant = module
        .constants
        .iter()
        .map(|(_, constant)| constant)
        .find(|constant| constant.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing {name} constant"));
    match module.global_expressions[constant.init] {
        naga::Expression::Literal(naga::Literal::U32(value)) => value,
        ref expression => panic!("{name} is not a u32 literal: {expression:?}"),
    }
}

fn fixed_type_bytes(module: &naga::Module, handle: Handle<Type>) -> usize {
    match &module.types[handle].inner {
        TypeInner::Scalar(scalar) | TypeInner::Atomic(scalar) => scalar.width as usize,
        TypeInner::Vector { size, scalar } => {
            let lanes = match size {
                naga::VectorSize::Bi => 2,
                naga::VectorSize::Tri => 3,
                naga::VectorSize::Quad => 4,
            };
            lanes * scalar.width as usize
        }
        TypeInner::Array { base, size, stride } => {
            let ArraySize::Constant(length) = size else {
                panic!("workgroup arrays must have fixed size")
            };
            assert_eq!(
                fixed_type_bytes(module, *base),
                *stride as usize,
                "unexpected workgroup array padding",
            );
            length.get() as usize * *stride as usize
        }
        other => panic!("unsupported fixed workgroup type {other:?}"),
    }
}
