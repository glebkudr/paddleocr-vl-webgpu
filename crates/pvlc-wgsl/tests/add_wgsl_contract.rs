use naga::{AddressSpace, StorageAccess};
use pvlc_runtime_core::KernelId;
use pvlc_wgsl::{BindingKind, UniformScalar, full_catalog, validate_source_contract};

#[test]
fn add_has_a_fixed_fp32_webgpu_abi() {
    let module = full_catalog()
        .iter()
        .find(|module| module.spec.kernel == KernelId::AddF32)
        .unwrap();
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
            (0, 1, BindingKind::StorageReadF32),
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
            ("length", UniformScalar::U32, 0),
            ("padding0", UniformScalar::U32, 4),
            ("padding1", UniformScalar::U32, 8),
            ("padding2", UniformScalar::U32, 12),
        ]
    );
    assert_eq!(module.spec.uniform_span, 16);
    assert!(module.spec.required_features.is_empty());
    validate_source_contract(&module.spec, module.source).unwrap();
}

#[test]
fn reflected_add_shader_reads_both_operands_and_writes_only_the_output() {
    let builtin = pvlc_wgsl::module(KernelId::AddF32).unwrap();
    let module = naga::front::wgsl::parse_str(builtin.source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();

    let mut resources = module
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
            .map(|(binding, name, _)| (*binding, *name))
            .collect::<Vec<_>>(),
        vec![(0, "left"), (1, "right"), (2, "output"), (3, "params")]
    );
    for resource in resources.iter().take(2) {
        assert_eq!(
            resource.2,
            AddressSpace::Storage {
                access: StorageAccess::LOAD,
            }
        );
    }
    assert_eq!(
        resources[2].2,
        AddressSpace::Storage {
            access: StorageAccess::LOAD | StorageAccess::STORE,
        }
    );
    assert_eq!(resources[3].2, AddressSpace::Uniform);
    assert!(!builtin.source.contains("var<workgroup>"));
    assert!(builtin.source.contains(
        "let row_stride = select(params.length, params.padding0, params.padding0 != 0u);"
    ));
    assert!(
        builtin
            .source
            .contains("let index = global_id.x + global_id.y * row_stride;")
    );
    assert!(builtin.source.contains("if index >= params.length {"));
    assert!(
        builtin
            .source
            .contains("output.data[index] = left.data[index] + right.data[index]")
    );
}
