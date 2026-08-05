//! Structural browser contract for M7q1 weight-only FP16 decoder execution.
//!
//! The real browser gate owns numerical/effect validation. This host-side gate
//! pins the otherwise wasm-only admission and resource-selection branches so a
//! balanced pack cannot silently execute through the F32 ABI.

use std::collections::BTreeMap;

use syn::{ImplItem, Item, visit::Visit};

const WEB_RUNTIME: &str = include_str!("../src/web.rs");
const STACK_RUNTIME: &str = include_str!("../src/web/decoder_stack_session.rs");

#[derive(Default)]
struct PlanAccesses {
    function_calls: BTreeMap<String, usize>,
    method_calls: BTreeMap<String, usize>,
    fields: BTreeMap<String, usize>,
    self_fields: BTreeMap<String, usize>,
    struct_fields: BTreeMap<String, usize>,
}

#[derive(Default)]
struct StoragePipelineMatch {
    matches: usize,
    arms: BTreeMap<String, PlanAccesses>,
}

fn is_weight_storage_discriminant(expression: &syn::Expr) -> bool {
    let syn::Expr::Field(storage) = expression else {
        return false;
    };
    let syn::Member::Named(storage_name) = &storage.member else {
        return false;
    };
    let syn::Expr::Field(plan) = storage.base.as_ref() else {
        return false;
    };
    let syn::Member::Named(plan_name) = &plan.member else {
        return false;
    };
    matches!(
        plan.base.as_ref(),
        syn::Expr::Path(path)
            if path.path.is_ident("self")
                && storage_name == "storage"
                && plan_name == "weight_resource_plan"
    )
}

impl<'ast> Visit<'ast> for StoragePipelineMatch {
    fn visit_expr_match(&mut self, expression: &'ast syn::ExprMatch) {
        if is_weight_storage_discriminant(&expression.expr) {
            self.matches += 1;
            for arm in &expression.arms {
                let syn::Pat::Path(pattern) = &arm.pat else {
                    continue;
                };
                let Some(storage) = pattern.path.segments.last() else {
                    continue;
                };
                if storage.ident == "F32" || storage.ident == "F16" {
                    let mut arm_accesses = PlanAccesses::default();
                    arm_accesses.visit_expr(&arm.body);
                    self.arms.insert(storage.ident.to_string(), arm_accesses);
                }
            }
        }
        syn::visit::visit_expr_match(self, expression);
    }
}

impl<'ast> Visit<'ast> for PlanAccesses {
    fn visit_expr_call(&mut self, expression: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = expression.func.as_ref()
            && let Some(segment) = path.path.segments.last()
        {
            *self
                .function_calls
                .entry(segment.ident.to_string())
                .or_default() += 1;
        }
        syn::visit::visit_expr_call(self, expression);
    }

    fn visit_expr_method_call(&mut self, expression: &'ast syn::ExprMethodCall) {
        *self
            .method_calls
            .entry(expression.method.to_string())
            .or_default() += 1;
        syn::visit::visit_expr_method_call(self, expression);
    }

    fn visit_expr_field(&mut self, expression: &'ast syn::ExprField) {
        if let syn::Member::Named(name) = &expression.member {
            *self.fields.entry(name.to_string()).or_default() += 1;
            if matches!(
                expression.base.as_ref(),
                syn::Expr::Path(path) if path.path.is_ident("self")
            ) {
                *self.self_fields.entry(name.to_string()).or_default() += 1;
            }
        }
        syn::visit::visit_expr_field(self, expression);
    }

    fn visit_field_value(&mut self, field: &'ast syn::FieldValue) {
        if let syn::Member::Named(name) = &field.member {
            *self.struct_fields.entry(name.to_string()).or_default() += 1;
        }
        syn::visit::visit_field_value(self, field);
    }
}

fn parsed_stack_module() -> syn::File {
    syn::parse_file(STACK_RUNTIME).expect("decoder stack module must parse")
}

fn impl_method<'a>(module: &'a syn::File, self_type: &str, name: &str) -> &'a syn::ImplItemFn {
    module
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Impl(item) => Some(item),
            _ => None,
        })
        .filter(|item| {
            matches!(
                item.self_ty.as_ref(),
                syn::Type::Path(path)
                    if path.path.segments.last().is_some_and(
                        |segment| segment.ident == self_type
                    )
            )
        })
        .flat_map(|item| &item.items)
        .find_map(|item| match item {
            ImplItem::Fn(method) if method.sig.ident == name => Some(method),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing {self_type}::{name}"))
}

fn session_method<'a>(module: &'a syn::File, name: &str) -> &'a syn::ImplItemFn {
    impl_method(module, "BrowserDecoderStackSession", name)
}

fn authority_method<'a>(module: &'a syn::File, name: &str) -> &'a syn::ImplItemFn {
    impl_method(module, "DecoderStackSessionAuthority", name)
}

fn free_function<'a>(module: &'a syn::File, name: &str) -> &'a syn::ItemFn {
    module
        .items
        .iter()
        .find_map(|item| match item {
            Item::Fn(function) if function.sig.ident == name => Some(function),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing free function {name}"))
}

fn accesses(block: &syn::Block) -> PlanAccesses {
    let mut accesses = PlanAccesses::default();
    accesses.visit_block(block);
    accesses
}

fn source_section(start: &str, end: &str) -> &'static str {
    STACK_RUNTIME
        .split(start)
        .nth(1)
        .and_then(|tail| tail.split(end).next())
        .unwrap_or_else(|| panic!("missing source section {start} ... {end}"))
}

#[test]
fn browser_device_requests_shader_f16_only_when_the_adapter_exposes_it() {
    assert!(
        WEB_RUNTIME.contains("let adapter_features = adapter.features();")
            && WEB_RUNTIME
                .contains("adapter_features.contains(wgpu::Features::SHADER_F16)"),
        "adapter shader-f16 capability is not checked"
    );
    assert!(
        WEB_RUNTIME.contains("required_features |= wgpu::Features::SHADER_F16"),
        "shader-f16 is not requested on the capable device branch"
    );
    assert!(
        WEB_RUNTIME.contains("let mut required_features = wgpu::Features::empty();"),
        "the F32 fallback device branch disappeared"
    );
}

#[test]
fn pack_precision_is_closed_world_and_controls_bytes_before_gpu_effects() {
    for required in [
        "\"fidelity\" => DecoderWeightStorage::F32",
        "\"balanced\" => DecoderWeightStorage::F16",
        "precision_profile is unsupported",
        "require_pack_f16_finite",
        "require_pack_f32_finite",
        "weight_storage.storage_bytes",
        "weight_storage.from_f32_byte_offset",
    ] {
        assert!(
            STACK_RUNTIME.contains(required),
            "missing FP16 pack admission authority: {required}"
        );
    }
    assert!(
        STACK_RUNTIME.contains("device.features().contains(wgpu::Features::SHADER_F16)"),
        "balanced admission does not bind to the requested device feature"
    );
    assert!(
        STACK_RUNTIME.contains("requires the shader-f16 device feature"),
        "missing fail-closed balanced-pack diagnostic"
    );

    let parser = STACK_RUNTIME
        .split("fn parse_stack_weight_pack(")
        .nth(1)
        .and_then(|tail| tail.split("fn require_stack_cache_operands(").next())
        .expect("pack parser body");
    assert!(
        parser.contains("DecoderWeightStorage"),
        "pack parser does not return the authenticated storage format"
    );

    let module = parsed_stack_module();
    let preparation = accesses(&free_function(&module, "prepare_stack_begin").block);
    assert_eq!(
        preparation
            .function_calls
            .get("check_stack_admission")
            .copied(),
        Some(1)
    );
    assert_eq!(
        preparation
            .function_calls
            .get("parse_stack_weight_pack")
            .copied(),
        Some(1)
    );
    assert_eq!(
        preparation
            .function_calls
            .get("require_weight_storage_device_feature")
            .copied(),
        Some(1)
    );
    assert!(
        preparation
            .struct_fields
            .contains_key("weight_resource_plan"),
        "shared begin preparation does not return the authenticated resource plan"
    );

    let shared = source_section("fn prepare_stack_begin(", "\nasync fn run_begin(");
    let admission = shared.find("check_stack_admission").unwrap();
    let parse = shared.find("parse_stack_weight_pack").unwrap();
    let resource_plan = shared
        .find("DecoderWeightResourceDescriptor")
        .expect("weight resource descriptor");
    let feature = shared
        .find("require_weight_storage_device_feature")
        .expect("weight-storage device feature gate");
    assert!(
        admission < parse && parse < resource_plan && resource_plan < feature,
        "shared begin preparation must parse and plan storage before the shader-f16 gate"
    );

    for method in ["begin", "begin_with_shader_override"] {
        let method_accesses = accesses(&authority_method(&module, method).block);
        assert_eq!(
            method_accesses
                .function_calls
                .get("prepare_stack_begin")
                .copied(),
            Some(1),
            "{method} must use the one shared precision admission path"
        );
        assert!(
            !method_accesses
                .function_calls
                .contains_key("parse_stack_weight_pack"),
            "{method} duplicates or bypasses shared pack parsing"
        );
        assert!(
            !method_accesses
                .function_calls
                .contains_key("require_weight_storage_device_feature"),
            "{method} duplicates or bypasses the shared feature gate"
        );
    }

    let begin = source_section(
        "pub(super) fn begin(",
        "\n    pub(super) fn begin_with_shader_override(",
    );
    let override_begin = source_section(
        "pub(super) fn begin_with_shader_override(",
        "\n    pub(super) fn step(",
    );
    for (name, body) in [
        ("begin", begin),
        ("begin_with_shader_override", override_begin),
    ] {
        let preparation = body.find("prepare_stack_begin(").unwrap();
        let promise = body.find("future_to_promise").unwrap();
        let gpu = body.find("run_begin(").unwrap();
        assert!(
            preparation < promise && promise < gpu,
            "{name} must finish shared precision admission before the async GPU path"
        );
    }
}

#[test]
fn balanced_session_selects_f16_kernels_and_half_sized_weight_ranges() {
    for kernel in [
        "rms_norm_f16_weights",
        "gemv_tiled_f16_weights",
        "linear_projection_f16_weights",
    ] {
        assert!(
            STACK_RUNTIME.contains(kernel),
            "balanced kernel {kernel} is absent"
        );
    }
    for field in [
        "weight_resource_plan: pvlc_runtime_core::DecoderWeightResourcePlan",
        "checkpoint_blake3",
        "rms_norm_f16_pipeline",
        "gemv_tiled_f16_pipeline",
        "prefill_projection_f16_pipeline",
    ] {
        assert!(
            STACK_RUNTIME.contains(field),
            "session does not own {field}"
        );
    }

    let module = parsed_stack_module();
    let session = module
        .items
        .iter()
        .find_map(|item| match item {
            Item::Struct(item) if item.ident == "BrowserDecoderStackSession" => Some(item),
            _ => None,
        })
        .expect("BrowserDecoderStackSession");
    let resource_plan_fields = session
        .fields
        .iter()
        .filter(|field| {
            field
                .ident
                .as_ref()
                .is_some_and(|name| name == "weight_resource_plan")
        })
        .collect::<Vec<_>>();
    assert_eq!(resource_plan_fields.len(), 1);
    assert!(matches!(
        &resource_plan_fields[0].ty,
        syn::Type::Path(path)
            if path.path.segments.last().is_some_and(
                |segment| segment.ident == "DecoderWeightResourcePlan"
            )
    ));

    // One shared plan supplies every immutable allocation/range and every
    // layer offset. Decode and prefill each consume one nine-offset result
    // rather than repeating precision arithmetic at nine call sites.
    let create = accesses(&session_method(&module, "create").block);
    for field in [
        "layer_weight_bulk_bytes",
        "rope_table_bytes",
        "final_norm_weight_bytes",
        "lm_head_weight_bytes",
    ] {
        assert!(
            create.fields.contains_key(field),
            "session create bypasses resource-plan field {field}"
        );
    }
    for field in [
        "rms_norm_f16_weights",
        "gemv_tiled_f16_weights",
        "linear_projection_f16_weights",
    ] {
        assert!(
            create.fields.contains_key(field),
            "session create does not consume the shader source {field}"
        );
    }
    for field in [
        "rms_norm_f16_pipeline",
        "gemv_tiled_f16_pipeline",
        "prefill_projection_f16_pipeline",
    ] {
        assert!(
            create.struct_fields.contains_key(field),
            "session create does not causally construct {field}"
        );
    }

    let selector = accesses(&session_method(&module, "decoder_weight_pipelines").block);
    for field in [
        "storage",
        "rms_norm_pipeline",
        "gemv_tiled_pipeline",
        "prefill_projection_pipeline",
        "rms_norm_f16_pipeline",
        "gemv_tiled_f16_pipeline",
        "prefill_projection_f16_pipeline",
    ] {
        assert!(
            selector.fields.contains_key(field),
            "weight-storage selector does not bind {field}"
        );
    }
    let mut pipeline_match = StoragePipelineMatch::default();
    pipeline_match.visit_block(&session_method(&module, "decoder_weight_pipelines").block);
    assert_eq!(
        pipeline_match.matches, 1,
        "pipeline selector must have one match on self.weight_resource_plan.storage"
    );
    let f32_arm = pipeline_match.arms.get("F32").expect("F32 selector arm");
    let f16_arm = pipeline_match.arms.get("F16").expect("F16 selector arm");
    for field in [
        "rms_norm_pipeline",
        "gemv_tiled_pipeline",
        "prefill_projection_pipeline",
    ] {
        assert!(
            f32_arm.self_fields.contains_key(field),
            "F32 selector arm does not return self.{field}"
        );
        assert!(
            !f16_arm.self_fields.contains_key(field),
            "F16 selector arm incorrectly returns self.{field}"
        );
    }
    for field in [
        "rms_norm_f16_pipeline",
        "gemv_tiled_f16_pipeline",
        "prefill_projection_f16_pipeline",
    ] {
        assert!(
            f16_arm.self_fields.contains_key(field),
            "F16 selector arm does not return self.{field}"
        );
        assert!(
            !f32_arm.self_fields.contains_key(field),
            "F32 selector arm incorrectly returns self.{field}"
        );
    }
    let compact_source = STACK_RUNTIME
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();

    for (method, selected_fields) in [
        ("encode_step", ["rms_norm", "gemv_tiled"].as_slice()),
        (
            "encode_prefill",
            ["rms_norm", "prefill_projection"].as_slice(),
        ),
    ] {
        let method_accesses = accesses(&session_method(&module, method).block);
        assert_eq!(
            method_accesses
                .method_calls
                .get("layer_weight_offsets")
                .copied(),
            Some(1),
            "{method} must obtain all nine offsets from the shared plan once"
        );
        assert!(
            !method_accesses.fields.contains_key("weight_stride_bytes"),
            "{method} still computes offsets from the F32 stack plan"
        );
        assert!(
            !method_accesses
                .method_calls
                .contains_key("from_f32_byte_offset"),
            "{method} duplicates precision scaling outside the plan"
        );
        assert_eq!(
            method_accesses
                .method_calls
                .get("decoder_weight_pipelines")
                .copied(),
            Some(1),
            "{method} must select the authenticated weight pipelines once"
        );
        for field in [
            "rms_norm_pipeline",
            "gemv_tiled_pipeline",
            "prefill_projection_pipeline",
            "rms_norm_f16_pipeline",
            "gemv_tiled_f16_pipeline",
            "prefill_projection_f16_pipeline",
        ] {
            assert!(
                !method_accesses.self_fields.contains_key(field),
                "{method} bypasses the storage selector through self.{field}"
            );
        }
        for field in selected_fields {
            assert!(
                method_accesses.fields.contains_key(*field),
                "{method} does not consume weight_pipelines.{field}"
            );
        }
    }

    let logits = accesses(&free_function(&module, "run_logits").block);
    assert_eq!(
        logits.method_calls.get("final_norm_weight_range").copied(),
        Some(1)
    );
    assert_eq!(
        logits.method_calls.get("lm_head_weight_range").copied(),
        Some(1)
    );
    assert_eq!(
        logits.method_calls.get("decoder_weight_pipelines").copied(),
        Some(1),
        "logits must select the authenticated weight pipelines once"
    );
    for field in [
        "rms_norm_pipeline",
        "gemv_tiled_pipeline",
        "rms_norm_f16_pipeline",
        "gemv_tiled_f16_pipeline",
    ] {
        assert!(
            !logits.fields.contains_key(field),
            "logits bypasses the storage selector through session.{field}"
        );
    }
    for field in ["rms_norm", "gemv_tiled"] {
        assert!(
            logits.fields.contains_key(field),
            "logits does not consume weight_pipelines.{field}"
        );
    }
    assert_eq!(
        compact_source
            .matches("letweight_pipelines=self.decoder_weight_pipelines()?;")
            .count(),
        2,
        "decode and prefill must each bind the authenticated pipeline selection"
    );
    assert_eq!(
        compact_source
            .matches("letweight_pipelines=session.decoder_weight_pipelines()?;")
            .count(),
        2,
        "logits and top-1 must each bind the authenticated pipeline selection"
    );
    assert!(
        compact_source.contains("weight_pipelines.prefill_projection"),
        "prefill does not consume the selected projection pipeline"
    );

    let capability_gate = accesses(&free_function(&module, "validate_stack_capabilities").block);
    for field in [
        "layer_weight_stride_bytes",
        "layer_weight_bulk_bytes",
        "rope_table_bytes",
        "final_norm_weight_bytes",
        "lm_head_weight_bytes",
    ] {
        assert!(
            capability_gate.fields.contains_key(field),
            "capability gate bypasses physical resource-plan field {field}"
        );
    }
    assert!(
        !source_section("fn validate_stack_capabilities(", "\nfn js_object_set(",)
            .contains("stack_plan.weight_stride_bytes"),
        "capability gate still rejects FP16 resources by their logical F32 weight size"
    );
    assert!(STACK_RUNTIME.contains("cache_stride_bytes"));
    assert!(STACK_RUNTIME.contains("rope_table_bytes"));
}

#[test]
fn fp32_session_surface_and_existing_kernels_remain_available() {
    for kernel in [
        "rms_norm_f32",
        "gemv_tiled_f32",
        "decoder_mrope_f32",
        "decoder_prefill_mrope_f32",
        "vision_patch_projection_f32",
    ] {
        assert!(
            STACK_RUNTIME.contains(kernel),
            "accepted F32 kernel {kernel} was removed"
        );
    }
    for method in [
        "pub(super) fn begin(",
        "pub(super) fn step(",
        "pub(super) fn prefill(",
        "pub(super) fn logits(",
        "pub(super) fn finish(",
        "pub(super) fn abort(",
    ] {
        assert!(
            STACK_RUNTIME.contains(method),
            "sealed session operation drifted: {method}"
        );
    }
}

#[test]
fn decoder_prefill_f16_linear_cannot_enter_any_vision_kernel_set() {
    for constant_name in [
        "const VISION_LAYER_KERNELS:",
        "const VISION_QKV_STACK_KERNELS:",
        "const PROJECTOR_KERNELS:",
    ] {
        let body = WEB_RUNTIME
            .split(constant_name)
            .nth(1)
            .and_then(|tail| tail.split("];").next())
            .unwrap_or_else(|| panic!("missing {constant_name}"));
        assert!(body.contains("VisionPatchProjectionF32"));
        assert!(
            !body.contains("LinearProjectionF16Weights"),
            "{constant_name} admits the decoder-only FP16 linear kernel"
        );
    }
    assert!(
        !WEB_RUNTIME.contains("vision_weight_storage: DecoderWeightStorage::F16"),
        "browser vision resource selection changed from F32"
    );
}
