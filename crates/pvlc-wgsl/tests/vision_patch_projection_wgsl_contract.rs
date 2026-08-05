use naga::{AddressSpace, StorageAccess};
use pvlc_runtime_core::KernelId;
use pvlc_wgsl::{BindingKind, UniformScalar, full_catalog, validate_source_contract};

#[test]
fn vision_patch_projection_has_a_fixed_checkpoint_native_webgpu_abi() {
    let module = full_catalog()
        .iter()
        .find(|module| module.spec.kernel == KernelId::VisionPatchProjectionF32)
        .unwrap();
    assert_eq!(module.spec.entry_point, "main");
    assert_eq!(module.spec.workgroup_size, [8, 8, 1]);
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
            (0, 3, BindingKind::StorageReadWriteF32),
            (0, 4, BindingKind::Uniform),
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
            ("patch_count", UniformScalar::U32, 0),
            ("input_width", UniformScalar::U32, 4),
            ("output_width", UniformScalar::U32, 8),
            ("padding", UniformScalar::U32, 12),
        ]
    );
    assert_eq!(module.spec.uniform_span, 16);
    assert!(module.spec.required_features.is_empty());
    validate_source_contract(&module.spec, module.source).unwrap();
}

#[test]
fn reflected_patch_shader_tiles_the_existing_output_major_checkpoint_layout() {
    let builtin = pvlc_wgsl::module(KernelId::VisionPatchProjectionF32).unwrap();
    let module = naga::front::wgsl::parse_str(builtin.source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();

    let mut bound_resources: Vec<_> = module
        .global_variables
        .iter()
        .filter_map(|(_, global)| {
            global.binding.as_ref().map(|binding| {
                (
                    binding.binding,
                    binding.group,
                    global.name.as_deref().unwrap_or(""),
                    global.space,
                )
            })
        })
        .collect();
    bound_resources.sort_unstable_by_key(|resource| resource.0);
    assert_eq!(bound_resources.len(), 5);
    for (binding, expected_name) in [
        (0, "input"),
        (1, "weight"),
        (2, "bias"),
        (3, "output"),
        (4, "params"),
    ] {
        assert_eq!(bound_resources[binding].0, binding as u32);
        assert_eq!(bound_resources[binding].1, 0);
        assert_eq!(bound_resources[binding].2, expected_name);
    }
    for resource in bound_resources.iter().take(3) {
        assert_eq!(
            resource.3,
            AddressSpace::Storage {
                access: StorageAccess::LOAD,
            }
        );
    }
    assert_eq!(
        bound_resources[3].3,
        AddressSpace::Storage {
            access: StorageAccess::LOAD | StorageAccess::STORE,
        }
    );
    assert_eq!(bound_resources[4].3, AddressSpace::Uniform);

    let storage: Vec<_> = module
        .global_variables
        .iter()
        .filter(|(_, global)| matches!(global.space, AddressSpace::Storage { .. }))
        .collect();
    assert_eq!(storage.len(), 4, "input, checkpoint weight+bias, output");
    assert_eq!(
        storage
            .iter()
            .filter(|(_, global)| {
                global.space
                    == (AddressSpace::Storage {
                        access: StorageAccess::LOAD | StorageAccess::STORE,
                    })
            })
            .count(),
        1,
        "only output may be writable"
    );
    let internal = module
        .global_variables
        .iter()
        .filter(|(_, global)| global.binding.is_none())
        .map(|(_, global)| {
            (
                global.name.as_deref(),
                global.space,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        internal,
        [
            (Some("input_tile"), AddressSpace::WorkGroup),
            (Some("weight_tile"), AddressSpace::WorkGroup),
        ],
        "the only internal storage must be the cooperatively reused A/B tiles",
    );
    assert!(builtin.source.contains("var initial_bias = vec4<f32>(0.0)"));
    assert!(
        builtin
            .source
            .contains("initial_bias[output_offset] = bias.data[output_column]")
    );
    for accumulator in 0..4 {
        assert!(
            builtin
                .source
                .contains(&format!("var accumulator{accumulator} = initial_bias;"))
        );
    }
    assert!(
        builtin
            .source
            .contains("weight.data[output_column * params.input_width + input_depth]")
    );
    assert!(!builtin.source.contains("return;"));
    assert_eq!(builtin.source.matches("workgroupBarrier();").count(), 2);
}
