use naga::{AddressSpace, StorageAccess};
use pvlc_runtime_core::KernelId;
use pvlc_wgsl::{
    BindingKind, UniformScalar, full_catalog, storage_read_write_variant, validate_source_contract,
};

fn fused_module() -> &'static pvlc_wgsl::KernelModule {
    pvlc_wgsl::module(KernelId::VisionQkvFusedF32)
        .expect("the appended fused QKV kernel must have one fixed WGSL module")
}

fn replace_once(source: &str, from: &str, to: &str) -> String {
    assert_eq!(
        source.matches(from).count(),
        1,
        "mutation anchor must identify exactly one reviewed construct"
    );
    source.replacen(from, to, 1)
}

fn compact(source: &str) -> String {
    let mut uncommented = String::with_capacity(source.len());
    for line in source.lines() {
        uncommented.push_str(line.split_once("//").map_or(line, |(code, _)| code));
        uncommented.push('\n');
    }

    let mut compact = String::with_capacity(uncommented.len());
    let mut pending_whitespace = false;
    let mut previous_was_identifier = false;
    for character in uncommented.chars() {
        if character.is_whitespace() {
            pending_whitespace = true;
            continue;
        }
        let is_identifier = character.is_ascii_alphanumeric() || character == '_';
        if pending_whitespace && previous_was_identifier && is_identifier {
            compact.push(' ');
        }
        compact.push(character);
        pending_whitespace = false;
        previous_was_identifier = is_identifier;
    }
    compact
}

fn identifier_before(source: &str, marker: &str) -> Result<String, &'static str> {
    let marker = source.find(marker).ok_or("semantic marker is absent")?;
    let identifier: String = source[..marker]
        .chars()
        .rev()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    (!identifier.is_empty())
        .then_some(identifier)
        .ok_or("semantic marker has no identifier")
}

fn replace_identifier(source: &str, from: &str, to: &str) -> String {
    let mut output = String::with_capacity(source.len());
    let mut token = String::new();
    let flush = |output: &mut String, token: &mut String| {
        if token == from {
            output.push_str(to);
        } else {
            output.push_str(token);
        }
        token.clear();
    };
    for character in source.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            token.push(character);
        } else {
            flush(&mut output, &mut token);
            output.push(character);
        }
    }
    flush(&mut output, &mut token);
    output
}

fn canonical_semantic_source(source: &str) -> Result<String, &'static str> {
    let mut source = compact(source);
    let x = identifier_before(&source, "=global_id.x;")?;
    let y = identifier_before(&source, "=global_id.y;")?;
    let z = identifier_before(&source, "=global_id.z;")?;
    let loop_start = source
        .find("for(var ")
        .ok_or("ascending inner loop is absent")?
        + "for(var ".len();
    let depth_end = source[loop_start..]
        .find("=0u;")
        .ok_or("inner loop does not start at zero")?
        + loop_start;
    let depth = source[loop_start..depth_end].to_owned();
    for (from, to) in [
        (x.as_str(), "$x"),
        (y.as_str(), "$y"),
        (z.as_str(), "$z"),
        (depth.as_str(), "$depth"),
    ] {
        source = replace_identifier(&source, from, to);
    }
    Ok(source)
}

fn audit_fused_semantics(source: &str) -> Result<(), &'static str> {
    let source = canonical_semantic_source(source)?;
    let loop_start = source
        .find("for(var $depth=0u;$depth<params.input_width;$depth=$depth+1u)")
        .ok_or("inner accumulation must visit depth in increasing order")?;
    for resource in ["query_bias", "key_bias", "value_bias"] {
        let use_position = source
            .find(&format!("{resource}.data[$x]"))
            .ok_or("projection bias is not indexed by output channel")?;
        if use_position >= loop_start {
            return Err("projection bias must seed the accumulator before the inner loop");
        }
    }
    if !source.contains("input.data[$y*params.input_width+$depth]") {
        return Err("input is not read in token-major row order");
    }
    for resource in ["query_weight", "key_weight", "value_weight"] {
        let direct = format!("{resource}.data[$x*params.input_width+$depth]");
        let helper = format!("{resource}.data[qkv_weight_index($x,$depth)]");
        if !(source.contains(&direct)
            || (source.contains(&helper) && source.contains("fn qkv_weight_index(")))
        {
            return Err("projection weight is not [output, input] row-major");
        }
    }
    let output_formula = "$z*params.plane_stride_elements+$y*params.output_width+$x";
    let direct_output = source.contains(&format!("output.data[{output_formula}]"));
    let aliased_output = identifier_before(&source, &format!("={output_formula};"))
        .is_ok_and(|alias| source.contains(&format!("output.data[{alias}]")));
    let helper_output = source.contains("output.data[qkv_output_index($z,$y,$x)]");
    if !direct_output && !aliased_output && !helper_output {
        return Err("output does not preserve the padded Q/K/V plane mapping");
    }
    let if_routing = source.matches("$z==0u").count() >= 2 && source.matches("$z==1u").count() >= 2;
    let switch_routing = source.contains("switch($z)")
        && source.contains("case 0u:")
        && source.contains("case 1u:")
        && source.contains("case 2u:");
    if !if_routing && !switch_routing {
        return Err("global z does not independently route Q, K, and V");
    }
    if source.contains("$depth*params.output_width+$x")
        || source.contains("var<workgroup>")
        || source.contains("var<private>")
    {
        return Err("projection transposes or repacks data");
    }
    Ok(())
}

#[test]
fn semantic_audit_freezes_row_major_z_routing_and_bias_first_ascending_accumulation() {
    let source = fused_module().source;
    audit_fused_semantics(source).unwrap();

    let compact_source = compact(source);
    let x = identifier_before(&compact_source, "=global_id.x;").unwrap();
    let y = identifier_before(&compact_source, "=global_id.y;").unwrap();
    let z = identifier_before(&compact_source, "=global_id.z;").unwrap();
    let depth_start = compact_source.find("for(var ").unwrap() + "for(var ".len();
    let depth_end = compact_source[depth_start..].find("=0u;").unwrap() + depth_start;
    let depth = compact_source[depth_start..depth_end].to_owned();

    let mut aliases = source.to_owned();
    for (from, to) in [
        (x.as_str(), "arbitrary_lane_alias"),
        (y.as_str(), "arbitrary_row_alias"),
        (z.as_str(), "arbitrary_plane_alias"),
        (depth.as_str(), "arbitrary_inner_alias"),
    ] {
        aliases = replace_identifier(&aliases, from, to);
    }
    audit_fused_semantics(&aliases).expect("harmless local aliases must not change semantics");

    let transpose = compact_source.replacen(
        &format!("query_weight.data[{x}*params.input_width+{depth}]"),
        &format!("query_weight.data[{depth}*params.output_width+{x}]"),
        1,
    );
    let wrong_route = compact_source.replacen(&format!("{z}==0u"), &format!("{z}==2u"), 1);
    let descending = compact_source.replacen(
        &format!("{depth}={depth}+1u"),
        &format!("{depth}={depth}-1u"),
        1,
    );
    let missing_bias = compact_source.replacen("query_bias.data", "value_bias.data", 1);
    let missing_stride = compact_source.replacen(
        &format!("{z}*params.plane_stride_elements+{y}*params.output_width+{x}"),
        &format!("{y}*params.output_width+{x}"),
        1,
    );
    for mutant in [
        transpose,
        wrong_route,
        descending,
        missing_bias,
        missing_stride,
    ] {
        assert_ne!(
            mutant, compact_source,
            "semantic mutation anchor was absent"
        );
        assert!(audit_fused_semantics(&mutant).is_err());
    }
}

#[test]
fn catalog_keeps_the_fixed_fused_qkv_module_after_the_legacy_prefix() {
    let catalog = full_catalog();
    assert_eq!(catalog.len(), KernelId::ALL.len());
    let fused_index = KernelId::ALL
        .iter()
        .position(|kernel| *kernel == KernelId::VisionQkvFusedF32)
        .unwrap();
    assert_eq!(
        catalog[fused_index].spec.kernel,
        KernelId::VisionQkvFusedF32
    );
    assert_eq!(
        catalog[..fused_index]
            .iter()
            .map(|module| module.spec.kernel)
            .collect::<Vec<_>>(),
        KernelId::ALL[..fused_index]
    );

    let module = fused_module();
    assert_eq!(module.spec.entry_point, "main");
    assert_eq!(module.spec.workgroup_size, [8, 8, 1]);
    assert!(module.spec.required_features.is_empty());
    assert!(
        !module
            .source
            .lines()
            .any(|line| line.trim().starts_with("enable "))
    );
    validate_source_contract(&module.spec, module.source).unwrap();
}

#[test]
fn reflected_abi_is_exactly_eight_storage_bindings_then_one_uniform() {
    let module = fused_module();
    assert_eq!(
        module
            .spec
            .bindings
            .iter()
            .map(|binding| (binding.group, binding.binding, binding.kind))
            .collect::<Vec<_>>(),
        vec![
            (0, 0, BindingKind::StorageReadF32),
            (0, 1, BindingKind::StorageReadF32),
            (0, 2, BindingKind::StorageReadF32),
            (0, 3, BindingKind::StorageReadF32),
            (0, 4, BindingKind::StorageReadF32),
            (0, 5, BindingKind::StorageReadF32),
            (0, 6, BindingKind::StorageReadF32),
            (0, 7, BindingKind::StorageReadWriteF32),
            (0, 8, BindingKind::Uniform),
        ]
    );
    assert_eq!(
        module
            .spec
            .uniform_fields
            .iter()
            .map(|field| (field.name, field.scalar, field.offset))
            .collect::<Vec<_>>(),
        vec![
            ("tokens", UniformScalar::U32, 0),
            ("input_width", UniformScalar::U32, 4),
            ("output_width", UniformScalar::U32, 8),
            ("plane_stride_elements", UniformScalar::U32, 12),
        ]
    );
    assert_eq!(module.spec.uniform_span, 16);

    let parsed = naga::front::wgsl::parse_str(module.source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&parsed)
    .unwrap();
    let mut resources: Vec<_> = parsed
        .global_variables
        .iter()
        .filter_map(|(_, global)| {
            global.binding.as_ref().map(|binding| {
                (
                    binding.group,
                    binding.binding,
                    global.name.as_deref().unwrap_or(""),
                    global.space,
                )
            })
        })
        .collect();
    resources.sort_unstable_by_key(|resource| resource.1);
    assert_eq!(resources.len(), 9);
    assert_eq!(
        resources
            .iter()
            .map(|resource| (resource.0, resource.1, resource.2))
            .collect::<Vec<_>>(),
        [
            (0, 0, "input"),
            (0, 1, "query_weight"),
            (0, 2, "query_bias"),
            (0, 3, "key_weight"),
            (0, 4, "key_bias"),
            (0, 5, "value_weight"),
            (0, 6, "value_bias"),
            (0, 7, "output"),
            (0, 8, "params"),
        ]
    );
    for resource in resources.iter().take(7) {
        assert_eq!(
            resource.3,
            AddressSpace::Storage {
                access: StorageAccess::LOAD,
            }
        );
    }
    assert_eq!(
        resources[7].3,
        AddressSpace::Storage {
            access: StorageAccess::LOAD | StorageAccess::STORE,
        }
    );
    assert_eq!(resources[8].3, AddressSpace::Uniform);
    assert!(parsed.global_variables.iter().all(|(_, global)| !matches!(
        global.space,
        AddressSpace::WorkGroup | AddressSpace::Private
    )));
}

#[test]
fn source_contract_rejects_abi_and_optional_feature_mutants_independently_of_hashes() {
    let module = fused_module();
    let source = module.source;
    let mutations = [
        replace_once(source, "@group(0) @binding(1)", "@group(0) @binding(9)"),
        replace_once(
            source,
            "@group(0) @binding(1) var<storage, read>",
            "@group(0) @binding(1) var<storage, read_write>",
        ),
        replace_once(
            source,
            "plane_stride_elements: u32",
            "plane_stride_bytes: u32",
        ),
        replace_once(
            source,
            "@workgroup_size(8, 8, 1)",
            "@workgroup_size(4, 8, 1)",
        ),
        format!("enable f16;\n{source}"),
    ];
    for mutated in mutations {
        assert_ne!(mutated, source);
        assert!(validate_source_contract(&module.spec, &mutated).is_err());
    }
}

#[test]
fn conservative_read_write_variant_preserves_abi_shape_and_fused_semantics() {
    let module = fused_module();
    let variant = storage_read_write_variant(&module.spec, module.source).unwrap();
    audit_fused_semantics(&variant).unwrap();

    let parsed = naga::front::wgsl::parse_str(&variant).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&parsed)
    .unwrap();
    let storage: Vec<_> = parsed
        .global_variables
        .iter()
        .filter(|(_, global)| matches!(global.space, AddressSpace::Storage { .. }))
        .collect();
    assert_eq!(storage.len(), 8);
    assert!(storage.iter().all(|(_, global)| global.space
        == (AddressSpace::Storage {
            access: StorageAccess::LOAD | StorageAccess::STORE,
        })));
    assert_eq!(
        parsed
            .global_variables
            .iter()
            .filter(|(_, global)| global.space == AddressSpace::Uniform)
            .count(),
        1
    );
}
