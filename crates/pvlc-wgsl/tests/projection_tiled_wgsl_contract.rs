use naga::{AddressSpace, ArraySize, Handle, Type, TypeInner};
use pvlc_runtime_core::KernelId;

const TILE: u32 = 32;
const WORKGROUP_LANES: u32 = 64;

#[test]
fn scalar_f32_projection_uses_a_32_by_32_gemm_tile() {
    for kernel in [KernelId::VisionPatchProjectionF32] {
        let builtin = pvlc_wgsl::module(kernel).expect("projection kernel");
        let reflected = naga::front::wgsl::parse_str(builtin.source).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&reflected)
        .unwrap();
        assert!(
            reflected.functions.is_empty(),
            "{kernel} must not hide global reads outside the audited entry-point dataflow",
        );

        assert_eq!(builtin.spec.workgroup_size, [8, 8, 1]);
        assert_eq!(constant_u32(&reflected, "PROJECTION_TILE_ROWS"), TILE);
        assert_eq!(constant_u32(&reflected, "PROJECTION_TILE_COLUMNS"), TILE);
        assert_eq!(constant_u32(&reflected, "PROJECTION_TILE_DEPTH"), TILE);
        assert_eq!(constant_u32(&reflected, "PROJECTION_ROWS_PER_LANE"), 4);
        assert_eq!(constant_u32(&reflected, "PROJECTION_COLUMNS_PER_LANE"), 4);
        assert_eq!(
            constant_u32(&reflected, "PROJECTION_WORKGROUP_SIZE"),
            WORKGROUP_LANES
        );

        let workgroup_globals = reflected
            .global_variables
            .iter()
            .filter(|(_, global)| matches!(global.space, AddressSpace::WorkGroup))
            .map(|(_, global)| {
                (
                    global.name.as_deref().unwrap_or("<unnamed>"),
                    fixed_type_bytes(&reflected, global.ty),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            workgroup_globals,
            [
                ("input_tile", (TILE * TILE * 4) as usize),
                ("weight_tile", (TILE * TILE * 4) as usize),
            ],
            "{kernel} must stage each A/B element once for a 32x32 output tile",
        );
        assert!(
            !builtin.source.contains("return;"),
            "{kernel} cannot let an edge lane exit before a workgroup barrier",
        );
        assert_eq!(
            builtin.source.matches("workgroupBarrier();").count(),
            2,
            "{kernel} needs only load-ready and reuse-safe barriers per K tile",
        );
        let outer_depth_header = concat!(
            "for (var depth_base = 0u; ",
            "depth_base < params.input_width; ",
            "depth_base = depth_base + PROJECTION_TILE_DEPTH)"
        );
        let outer_depth_block = block_span_after(builtin.source, outer_depth_header);
        assert!(builtin.source.contains(
            "for (var load_index = local_index; \
             load_index < PROJECTION_TILE_ROWS * PROJECTION_TILE_DEPTH; \
             load_index = load_index + PROJECTION_WORKGROUP_SIZE)"
        ));
        let load_block = block_span_after(
            builtin.source,
            concat!(
                "for (var load_index = local_index; ",
                "load_index < PROJECTION_TILE_ROWS * PROJECTION_TILE_DEPTH; ",
                "load_index = load_index + PROJECTION_WORKGROUP_SIZE)"
            ),
        );
        let load_source = &builtin.source[load_block.0..load_block.1];
        assert!(
            load_source.contains("input.data[") && load_source.contains("weight.data["),
            "{kernel} shared tiles must be populated from both real operands",
        );
        assert!(
            load_source.contains("input_tile[")
                && load_source.contains("= loaded_input;")
                && load_source.contains("weight_tile[")
                && load_source.contains("= loaded_weight;"),
            "{kernel} cooperative global reads must populate both shared tiles",
        );
        assert!(
            load_source.contains("input_row < params.patch_count")
                && load_source.contains("input_depth < params.input_width")
                && load_source.contains("output_column < params.output_width"),
            "{kernel} cooperative edge loads must be explicitly guarded",
        );
        let depth_block = block_span_after(
            builtin.source,
            "for (var depth = 0u; depth < PROJECTION_TILE_DEPTH; depth = depth + 1u)",
        );
        let depth_source = &builtin.source[depth_block.0..depth_block.1];
        assert!(
            depth_source.contains("input_tile[") && depth_source.contains("weight_tile["),
            "{kernel} dot products must consume both shared tiles",
        );
        assert_eq!(
            depth_source.matches("fma(").count(),
            4,
            "{kernel} must update four output rows, each containing four columns",
        );
        assert!(
            !depth_source.contains("input.data[") && !depth_source.contains("weight.data["),
            "{kernel} inner multiply loop must not retain the rejected global-read path",
        );
        assert!(
            outer_depth_block.0 < load_block.0
                && load_block.1 < depth_block.0
                && depth_block.1 < outer_depth_block.1,
            "{kernel} must load and consume every shared tile inside the complete K traversal",
        );
        let first_barrier = builtin.source[load_block.1..depth_block.0]
            .find("workgroupBarrier();")
            .map(|offset| load_block.1 + offset)
            .unwrap_or_else(|| {
                panic!("{kernel} needs a load-ready barrier before shared-tile multiplication")
            });
        let second_barrier = builtin.source[depth_block.1..outer_depth_block.1]
            .find("workgroupBarrier();")
            .map(|offset| depth_block.1 + offset)
            .unwrap_or_else(|| {
                panic!("{kernel} needs a reuse-safe barrier after shared-tile multiplication")
            });
        assert!(
            load_block.1 < first_barrier
                && first_barrier < depth_block.0
                && depth_block.1 < second_barrier
                && second_barrier < outer_depth_block.1,
            "{kernel} barriers must bracket each shared-tile compute phase",
        );
        for accumulator in 0..4 {
            assert!(
                depth_source.contains(&format!("accumulator{accumulator} = fma(")),
                "{kernel} must update output-row accumulator {accumulator}",
            );
        }
        assert_eq!(
            builtin.source.matches("input.data[").count(),
            1,
            "{kernel} must read A only while cooperatively filling the shared tile",
        );
        assert!(
            builtin.source.contains("fma("),
            "{kernel} must update a 4x4 register tile from shared operands",
        );
        let first_depth_tile = builtin
            .source
            .find("for (var depth_base = 0u;")
            .expect("F32 projection must traverse depth tiles");
        let bias_initialization =
            block_span_after(builtin.source, "for (var output_offset = 0u;");
        assert!(
            bias_initialization.1 < first_depth_tile,
            "{kernel} must seed every accumulator with bias before ordered depth accumulation",
        );
        let bias_source =
            &builtin.source[bias_initialization.0..bias_initialization.1];
        assert!(
            bias_source.contains("if output_column < params.output_width")
                && bias_source
                    .contains("initial_bias[output_offset] = bias.data[output_column]"),
            "{kernel} must tail-guard the bias seed",
        );
        for accumulator in 0..4 {
            assert!(
                builtin
                    .source
                    .contains(&format!("var accumulator{accumulator} = initial_bias;")),
                "{kernel} accumulator {accumulator} must preserve bias-first arithmetic",
            );
        }
        let row_guard = block_span_after(builtin.source, "if output_row < params.patch_count");
        let relative_column_guard = block_span_after(
            &builtin.source[row_guard.0..row_guard.1],
            "if output_column < params.output_width",
        );
        let column_guard = (
            row_guard.0 + relative_column_guard.0,
            row_guard.0 + relative_column_guard.1,
        );
        assert!(
            row_guard.0 < column_guard.0 && column_guard.1 < row_guard.1,
            "{kernel} output stores must be nested under both row and column tail guards",
        );
        let column_guard_source = &builtin.source[column_guard.0..column_guard.1];
        assert!(
            column_guard_source
                .contains("output.data[output_row * params.output_width + output_column]")
        );
        assert!(column_guard_source.contains("accumulated[output_offset]"));
        assert!(
            !column_guard_source.contains("bias.data["),
            "{kernel} must not reassociate bias after the ordered dot product",
        );
    }
}

#[test]
fn tiled_projection_preserves_checkpoint_layout_and_f32_accumulation() {
    let f32 = pvlc_wgsl::module(KernelId::VisionPatchProjectionF32)
        .unwrap()
        .source;
    assert_eq!(f32.matches("weight.data[").count(), 1);
    assert!(f32.contains("weight.data[output_column * params.input_width + input_depth]"));

    let f16 = pvlc_wgsl::module(KernelId::LinearProjectionF16Weights)
        .unwrap()
        .source;
    let reflected = naga::front::wgsl::parse_str(f16).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::SHADER_FLOAT16,
    )
    .validate(&reflected)
    .unwrap();

    audit_packed_projection_dataflow(f16).unwrap();
    assert_eq!(f16.matches("workgroupBarrier();").count(), 2);
    assert!(
        f16.contains("input_tile: array<array<vec4<f32>, 8>, 32>")
            && f16.contains("weight_tile: array<array<vec4<f32>, 8>, 32>")
            && f16.contains("var accumulators: array<vec4<f32>, 4>"),
        "the measured packed kernel deliberately fixes a 32x32 shared tile and 4x4 \
         per-lane register tile; representation changes require a replacement GPU benchmark",
    );

    let workgroup_globals = reflected
        .global_variables
        .iter()
        .filter(|(_, global)| matches!(global.space, AddressSpace::WorkGroup))
        .map(|(_, global)| {
            (
                global.name.as_deref().unwrap_or("<unnamed>"),
                fixed_type_bytes(&reflected, global.ty),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        workgroup_globals,
        [
            ("input_tile", (TILE * TILE * 4) as usize),
            ("weight_tile", (TILE * TILE * 4) as usize),
        ],
        "packed reads must retain one 32x32 F32 shared tile per operand",
    );
    assert!(
        reflected
            .global_variables
            .iter()
            .filter(|(_, global)| matches!(global.space, AddressSpace::WorkGroup))
            .all(|(_, global)| !type_contains_f16(&reflected, global.ty)),
        "packed F16 storage must be widened before entering shared memory",
    );

    let accumulator_types = reflected.entry_points[0]
        .function
        .local_variables
        .iter()
        .filter(|(_, local)| {
            local
                .name
                .as_deref()
                .is_some_and(|name| name.contains("accum"))
        })
        .map(|(_, local)| local.ty)
        .collect::<Vec<_>>();
    assert!(
        !accumulator_types.is_empty()
            && accumulator_types
                .iter()
                .all(|ty| type_contains_f32(&reflected, *ty)
                    && !type_contains_f16(&reflected, *ty)),
        "every accumulator representation must remain F32",
    );
}

#[test]
fn packed_projection_semantic_audit_rejects_broken_indexing_and_disconnected_dataflow() {
    let source = pvlc_wgsl::module(KernelId::LinearProjectionF16Weights)
        .unwrap()
        .source;
    audit_packed_projection_dataflow(source).unwrap();

    let mutations = [
        (
            "let packed_index = output_column * input_width_vec + input_depth / 4u;",
            "let packed_index = 0u;",
        ),
        (
            "let component = input_depth % 4u;",
            "let component = 0u;",
        ),
        (
            "if component == 1u { return f32(packed.y); }",
            "if component == 0u { return f32(packed.y); }",
        ),
        (
            "read_output_major_weight_component(input_depth, output_column_base + 2u),",
            "read_output_major_weight_component(input_depth, output_column_base + 1u),",
        ),
        (
            "return packed_output_columns;",
            "return vec4<f32>(0.0);",
        ),
        (
            "weight_tile[tile_depth][local_x] = vec4<f32>(weight.data[input_depth * output_width_vec + global_column_vec]);",
            "weight_tile[tile_depth][local_x] = vec4<f32>(0.0);",
        ),
        (
            "weight_tile[tile_depth][local_x] = read_output_major_weight(input_depth, global_column_base);",
            "weight_tile[tile_depth][local_x] = vec4<f32>(0.0);",
        ),
        (
            "input_tile[tile_row][local_x] = input.data[global_row * input_width_vec + input_depth_vec];",
            "input_tile[tile_row][local_x] = vec4<f32>(0.0);",
        ),
        (
            "let bias_value = bias.data[global_column_vec];",
            "let bias_value = vec4<f32>(0.0);",
        ),
        (
            "accumulators[row_offset] = fma(vec4<f32>(activation.z), coefficient2, accumulators[row_offset]);",
            "accumulators[row_offset] = accumulators[row_offset];",
        ),
        (
            "let values = accumulators[row_offset] + bias_value;",
            "let values = bias_value;",
        ),
        (
            "output.data[output_base + 3u] = values.w;",
            "output.data[output_base + 3u] = values.x;",
        ),
        (
            "let input_width_vec = params.input_width / 4u;",
            "let input_width_vec = 1u;",
        ),
        (
            "let output_width_vec = params.output_width / 4u;",
            "let output_width_vec = 1u;",
        ),
        (
            "let global_column_vec = workgroup_id.x * 8u + local_x;",
            "let global_column_vec = local_x;",
        ),
        (
            "let global_column_base = global_column_vec * 4u;",
            "let global_column_base = global_column_vec;",
        ),
        (
            "let input_depth_vec = depth_base / 4u + local_x;",
            "let input_depth_vec = local_x;",
        ),
    ];
    for (from, to) in mutations {
        assert!(
            source.contains(from),
            "the mutation fixture drifted from the production shader: {from}",
        );
        let mutated = source.replace(from, to);
        assert!(
            audit_packed_projection_dataflow(&mutated).is_err(),
            "the semantic audit accepted mutation {from:?} -> {to:?}",
        );
    }
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

fn audit_packed_projection_dataflow(source: &str) -> Result<(), String> {
    let component_accessor =
        try_block_span_after(source, "fn read_output_major_weight_component")
            .ok_or_else(|| "missing output-major packed component accessor".to_owned())?;
    let component_accessor = &source[component_accessor.0..component_accessor.1];
    for required in [
        "let input_width_vec = params.input_width / 4u;",
        "let packed_index = output_column * input_width_vec + input_depth / 4u;",
        "let component = input_depth % 4u;",
        "let packed = weight.data[packed_index];",
        "if component == 0u { return f32(packed.x); }",
        "if component == 1u { return f32(packed.y); }",
        "if component == 2u { return f32(packed.z); }",
        "return f32(packed.w);",
    ] {
        if !component_accessor.contains(required) {
            return Err(format!("output-major accessor is missing {required:?}"));
        }
    }

    let wrapper = try_block_span_after(source, "fn read_output_major_weight(")
        .ok_or_else(|| "missing output-major vec4 wrapper".to_owned())?;
    let wrapper = &source[wrapper.0..wrapper.1];
    if !wrapper.contains("let packed_output_columns = vec4<f32>(")
        || !wrapper.contains("return packed_output_columns;")
    {
        return Err("output-major wrapper calls are disconnected from its return value".to_owned());
    }
    for required in [
        "read_output_major_weight_component(input_depth, output_column_base + 0u),",
        "read_output_major_weight_component(input_depth, output_column_base + 1u),",
        "read_output_major_weight_component(input_depth, output_column_base + 2u),",
        "read_output_major_weight_component(input_depth, output_column_base + 3u),",
    ] {
        if !wrapper.contains(required) {
            return Err(format!("output-major vec4 wrapper is missing {required:?}"));
        }
    }

    for required in [
        "let input_width_vec = params.input_width / 4u;",
        "let output_width_vec = params.output_width / 4u;",
        "let global_column_vec = workgroup_id.x * 8u + local_x;",
        "let global_column_base = global_column_vec * 4u;",
        "let input_depth_vec = depth_base / 4u + local_x;",
    ] {
        if !source.contains(required) {
            return Err(format!("packed projection alias is missing {required:?}"));
        }
    }

    let input_major = try_block_span_after(
        source,
        "if params.padding == PROJECTION_WEIGHT_LAYOUT_INPUT_MAJOR",
    )
    .ok_or_else(|| "missing input-major branch".to_owned())?;
    let input_major = &source[input_major.0..input_major.1];
    for required in [
        "weight_tile[tile_depth][local_x] =",
        "vec4<f32>(weight.data[input_depth * output_width_vec + global_column_vec])",
    ] {
        if !input_major.contains(required) {
            return Err(format!("input-major dataflow is missing {required:?}"));
        }
    }

    let compute =
        try_block_span_after(source, "for (var depth_vector = 0u; depth_vector < 8u;")
            .ok_or_else(|| "missing packed depth-vector compute loop".to_owned())?;
    let compute = &source[compute.0..compute.1];
    for required in [
        "let coefficient0 = weight_tile[depth_vector * 4u + 0u][local_x];",
        "let coefficient1 = weight_tile[depth_vector * 4u + 1u][local_x];",
        "let coefficient2 = weight_tile[depth_vector * 4u + 2u][local_x];",
        "let coefficient3 = weight_tile[depth_vector * 4u + 3u][local_x];",
        "let activation = input_tile[local_y * 4u + row_offset][depth_vector];",
        "accumulators[row_offset] = fma(vec4<f32>(activation.x), coefficient0, accumulators[row_offset]);",
        "accumulators[row_offset] = fma(vec4<f32>(activation.y), coefficient1, accumulators[row_offset]);",
        "accumulators[row_offset] = fma(vec4<f32>(activation.z), coefficient2, accumulators[row_offset]);",
        "accumulators[row_offset] = fma(vec4<f32>(activation.w), coefficient3, accumulators[row_offset]);",
    ] {
        if !compute.contains(required) {
            return Err(format!("packed FMA dataflow is missing {required:?}"));
        }
    }
    if compute.contains("input.data[")
        || compute.contains("weight.data[")
        || compute.contains("bias.data[")
    {
        return Err("packed compute loop bypasses shared tiles".to_owned());
    }

    for required in [
        "weight_tile[tile_depth][local_x] = read_output_major_weight(input_depth, global_column_base);",
        "input_tile[tile_row][local_x] = input.data[global_row * input_width_vec + input_depth_vec];",
        "let bias_value = bias.data[global_column_vec];",
        "let values = accumulators[row_offset] + bias_value;",
        "output.data[output_base + 0u] = values.x;",
        "output.data[output_base + 1u] = values.y;",
        "output.data[output_base + 2u] = values.z;",
        "output.data[output_base + 3u] = values.w;",
    ] {
        if !source.contains(required) {
            return Err(format!("packed projection dataflow is missing {required:?}"));
        }
    }
    Ok(())
}

fn type_contains_f16(module: &naga::Module, handle: Handle<Type>) -> bool {
    type_contains_float_width(module, handle, 2)
}

fn type_contains_f32(module: &naga::Module, handle: Handle<Type>) -> bool {
    type_contains_float_width(module, handle, 4)
}

fn type_contains_float_width(
    module: &naga::Module,
    handle: Handle<Type>,
    expected_width: u8,
) -> bool {
    match &module.types[handle].inner {
        TypeInner::Scalar(scalar) | TypeInner::Atomic(scalar) => {
            scalar.kind == naga::ScalarKind::Float && scalar.width == expected_width
        }
        TypeInner::Vector { scalar, .. } | TypeInner::Matrix { scalar, .. } => {
            scalar.kind == naga::ScalarKind::Float && scalar.width == expected_width
        }
        TypeInner::Array { base, .. } => {
            type_contains_float_width(module, *base, expected_width)
        }
        TypeInner::Struct { members, .. } => members
            .iter()
            .any(|member| type_contains_float_width(module, member.ty, expected_width)),
        _ => false,
    }
}

fn fixed_type_bytes(module: &naga::Module, handle: Handle<Type>) -> usize {
    match &module.types[handle].inner {
        TypeInner::Scalar(scalar) | TypeInner::Atomic(scalar) => usize::from(scalar.width),
        TypeInner::Vector { size, scalar } => u32::from(*size) as usize * usize::from(scalar.width),
        TypeInner::Array {
            size: ArraySize::Constant(size),
            stride,
            ..
        } => size.get() as usize * *stride as usize,
        TypeInner::Struct { span, .. } => *span as usize,
        other => panic!("unsupported projection tile type: {other:?}"),
    }
}

fn block_span_after(source: &str, marker: &str) -> (usize, usize) {
    let start = source
        .find(marker)
        .unwrap_or_else(|| panic!("missing WGSL block marker: {marker}"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("WGSL block has no opening brace: {marker}"));
    let mut depth = 0_u32;
    for (offset, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return (start, open + offset + 1);
                }
            }
            _ => {}
        }
    }
    panic!("WGSL block has no closing brace: {marker}");
}

fn try_block_span_after(source: &str, marker: &str) -> Option<(usize, usize)> {
    let start = source.find(marker)?;
    let open = start + source[start..].find('{')?;
    let mut depth = 0_u32;
    for (offset, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((start, open + offset + 1));
                }
            }
            _ => {}
        }
    }
    None
}
