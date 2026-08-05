use naga::{AddressSpace, StorageAccess};
use pvlc_runtime_core::KernelId;
use pvlc_wgsl::{BindingKind, UniformScalar, full_catalog, validate_source_contract};

fn module(kernel: KernelId) -> &'static pvlc_wgsl::KernelModule {
    full_catalog()
        .iter()
        .find(|module| module.spec.kernel == kernel)
        .unwrap()
}

fn executable_source_without_comments(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut executable = String::with_capacity(source.len());
    let mut index = 0;
    let mut block_depth = 0_u32;
    while index < bytes.len() {
        if block_depth == 0 && bytes[index..].starts_with(b"//") {
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
        } else if bytes[index..].starts_with(b"/*") {
            block_depth += 1;
            index += 2;
        } else if block_depth > 0 && bytes[index..].starts_with(b"*/") {
            block_depth -= 1;
            index += 2;
        } else {
            if block_depth == 0 {
                executable.push(char::from(bytes[index]));
            }
            index += 1;
        }
    }
    assert_eq!(
        block_depth, 0,
        "test helper received an unterminated comment"
    );
    executable
}

#[test]
fn projector_merge_has_a_fixed_u32_mapping_and_fp32_webgpu_abi() {
    let module = module(KernelId::ProjectorMerge2x2F32);
    assert_eq!(module.spec.entry_point, "main");
    assert_eq!(module.spec.workgroup_size, [64, 1, 1]);
    assert_eq!(
        module
            .spec
            .bindings
            .iter()
            .map(|binding| (binding.group, binding.binding, binding.kind))
            .collect::<Vec<_>>(),
        vec![
            (0, 0, BindingKind::StorageReadF32),
            (0, 1, BindingKind::StorageReadU32),
            (0, 2, BindingKind::StorageReadWriteF32),
            (0, 3, BindingKind::Uniform),
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
            ("output_tokens", UniformScalar::U32, 0),
            ("hidden_size", UniformScalar::U32, 4),
            ("length", UniformScalar::U32, 8),
            ("row_stride", UniformScalar::U32, 12),
        ]
    );
    assert_eq!(module.spec.uniform_span, 16);
    assert!(module.spec.required_features.is_empty());
    validate_source_contract(&module.spec, module.source).unwrap();
}

#[test]
fn reflected_merge_shader_only_reorders_through_the_validated_source_map() {
    let builtin = module(KernelId::ProjectorMerge2x2F32);
    let parsed = naga::front::wgsl::parse_str(builtin.source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&parsed)
    .unwrap();

    let mut resources = parsed
        .global_variables
        .iter()
        .filter_map(|(_, global)| {
            global.binding.as_ref().map(|binding| {
                (
                    binding.binding,
                    global.name.as_deref().unwrap_or(""),
                    global.space,
                )
            })
        })
        .collect::<Vec<_>>();
    resources.sort_unstable_by_key(|resource| resource.0);
    assert_eq!(
        resources
            .iter()
            .map(|resource| resource.1)
            .collect::<Vec<_>>(),
        ["input", "source_token_indices", "output", "params"]
    );
    assert_eq!(
        resources[0].2,
        AddressSpace::Storage {
            access: StorageAccess::LOAD
        }
    );
    assert_eq!(
        resources[1].2,
        AddressSpace::Storage {
            access: StorageAccess::LOAD
        }
    );
    assert_eq!(
        resources[2].2,
        AddressSpace::Storage {
            access: StorageAccess::LOAD | StorageAccess::STORE
        }
    );
    assert_eq!(resources[3].2, AddressSpace::Uniform);
    assert!(!builtin.source.contains("var<workgroup>"));
    let executable = executable_source_without_comments(builtin.source);
    let comment_only_probe = executable_source_without_comments(
        "// let source_patch = column / params.hidden_size\n/* source_token_indices.data[output_token * 4u + source_patch] */",
    );
    assert!(!comment_only_probe.contains("source_patch"));
    assert!(executable.contains("let merged_width = params.hidden_size * 4u"));
    assert!(executable.contains("let source_patch = column / params.hidden_size"));
    assert!(executable.contains("source_token_indices.data[output_token * 4u + source_patch]"));
    assert!(executable.contains("input.data[source_token * params.hidden_size + channel]"));
}

#[test]
fn exact_projector_gelu_has_a_separate_abi_and_reviewed_erf_approximation() {
    let exact = module(KernelId::GeluErfF32);
    let tanh = module(KernelId::GeluTanhF32);
    assert_eq!(exact.spec.workgroup_size, [64, 1, 1]);
    assert_eq!(exact.spec.bindings, tanh.spec.bindings);
    assert_eq!(exact.spec.uniform_fields, tanh.spec.uniform_fields);
    assert_eq!(exact.spec.uniform_span, 16);
    assert_ne!(exact.source, tanh.source);
    validate_source_contract(&exact.spec, exact.source).unwrap();

    assert!(exact.source.contains("fn erf_approx(value: f32) -> f32"));
    for coefficient in [
        "0.3275911",
        "0.254829592",
        "-0.284496736",
        "1.421413741",
        "-1.453152027",
        "1.061405429",
    ] {
        assert!(
            exact.source.contains(coefficient),
            "missing reviewed A&S 7.1.26 coefficient {coefficient}"
        );
    }
    assert!(exact.source.contains("exp(-absolute * absolute)"));
    assert!(exact.source.contains("value * 0.7071067811865476"));
    assert!(
        exact
            .source
            .contains("0.5 * value * (1.0 + erf_approx(argument))")
    );
    assert!(!exact.source.contains("tanh("));
    assert!(!exact.source.contains("0.044715"));
}

#[test]
fn projector_shader_abis_fail_closed_on_mapping_or_exact_gelu_drift() {
    let merge = module(KernelId::ProjectorMerge2x2F32);
    let wrong_mapping_type = merge.source.replace("array<u32>", "array<f32>");
    assert!(validate_source_contract(&merge.spec, &wrong_mapping_type).is_err());
    let writable_mapping = merge.source.replace(
        "var<storage, read> source_token_indices",
        "var<storage, read_write> source_token_indices",
    );
    assert!(validate_source_contract(&merge.spec, &writable_mapping).is_err());

    let gelu = module(KernelId::GeluErfF32);
    let wrong_binding = gelu.source.replace("@binding(1)", "@binding(7)");
    assert!(validate_source_contract(&gelu.spec, &wrong_binding).is_err());
    let wrong_workgroup = gelu
        .source
        .replace("@workgroup_size(64, 1, 1)", "@workgroup_size(32, 1, 1)");
    assert!(validate_source_contract(&gelu.spec, &wrong_workgroup).is_err());
}
