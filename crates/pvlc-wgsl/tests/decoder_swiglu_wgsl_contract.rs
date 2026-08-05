use pvlc_runtime_core::KernelId;
use pvlc_wgsl::{
    BindingKind, UniformScalar, full_catalog, validate_catalog, validate_source_contract,
};

fn module(kernel: KernelId) -> &'static pvlc_wgsl::KernelModule {
    full_catalog()
        .iter()
        .find(|module| module.spec.kernel == kernel)
        .unwrap()
}

#[test]
fn decoder_swiglu_has_a_fixed_elementwise_fp32_webgpu_abi() {
    let swiglu = module(KernelId::DecoderSwigluF32);
    assert_eq!(KernelId::DecoderSwigluF32.as_str(), "decoder_swiglu_f32");
    assert_eq!(swiglu.spec.entry_point, "main");
    assert_eq!(swiglu.spec.workgroup_size, [64, 1, 1]);
    assert_eq!(
        swiglu
            .spec
            .bindings
            .iter()
            .map(|binding| (binding.group, binding.binding, binding.kind))
            .collect::<Vec<_>>(),
        [
            (0, 0, BindingKind::StorageReadF32),
            (0, 1, BindingKind::StorageReadF32),
            (0, 2, BindingKind::StorageReadWriteF32),
            (0, 3, BindingKind::Uniform),
        ]
    );
    assert_eq!(
        swiglu
            .spec
            .uniform_fields
            .iter()
            .map(|field| (field.name, field.scalar, field.offset))
            .collect::<Vec<_>>(),
        [
            ("length", UniformScalar::U32, 0),
            ("padding0", UniformScalar::U32, 4),
            ("padding1", UniformScalar::U32, 8),
            ("padding2", UniformScalar::U32, 12),
        ]
    );
    assert_eq!(swiglu.spec.uniform_span, 16);
    assert!(swiglu.spec.required_features.is_empty());
    validate_source_contract(&swiglu.spec, swiglu.source).unwrap();
}

#[test]
fn decoder_swiglu_source_implements_exact_silu_gate_times_up() {
    let source = module(KernelId::DecoderSwigluF32).source;

    // One thread per element; out-of-range threads return before any access.
    assert!(source.contains("let index = global_id.x;"));
    assert!(source.contains("if index >= params.length {"));

    // silu(x) = x / (1 + exp(-x)) applied to the gate, then elementwise
    // multiplication by the up projection: out[i] = silu(gate[i]) * up[i].
    // The full statements are pinned so the sigmoid placement, the division,
    // the negation, and the join cannot drift.
    assert!(source.contains("let gate_value = gate.data[index];"));
    assert!(source.contains("let up_value = up.data[index];"));
    assert!(source.contains("let activated = gate_value / (1.0 + exp(-gate_value));"));
    assert!(source.contains("output.data[index] = activated * up_value;"));
}

#[test]
fn decoder_swiglu_module_is_unique_and_naga_validated_by_the_full_catalog_gate() {
    assert_eq!(
        full_catalog()
            .iter()
            .filter(|module| module.spec.kernel == KernelId::DecoderSwigluF32)
            .count(),
        1
    );
    assert_eq!(
        &KernelId::ALL[13..19],
        [
            KernelId::DecoderKvAppendF32,
            KernelId::DecoderGqaF32,
            KernelId::DecoderGqaSplitPartialF32,
            KernelId::DecoderGqaSplitMergeF32,
            KernelId::DecoderMropeF32,
            KernelId::DecoderSwigluF32,
        ]
    );
    validate_catalog().unwrap();
}
