//! Structural contract for the sealed M6e5 persistent browser full decoder
//! layer session module before any production exists.

#[path = "decoder_session_contract_helpers.rs"]
mod helpers;

use std::collections::BTreeSet;

use helpers::*;
use syn::{File, ImplItem, Item, Type, Visibility, visit::Visit};

const WEB_RS: &str = "crates/pvlc-runtime-web/src/web.rs";
const CAUSAL_MODULE: &str = "crates/pvlc-runtime-web/src/web/decoder_full_layer_session.rs";
const AUTHORITY: &str = "DecoderFullLayerSessionAuthority";
const SESSION: &str = "BrowserDecoderFullLayerSession";
const OWNER_FIELD: &str = "decoder_full_layer_session";

#[test]
fn decoder_full_layer_session_is_one_cfg_free_sealed_causal_module() {
    let root = parse(WEB_RS);
    let declarations = root
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Mod(module) if module.ident == OWNER_FIELD => Some(module),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        declarations.len(),
        1,
        "Web runtime must declare decoder_full_layer_session exactly once"
    );
    let module_declaration = declarations[0];
    assert!(
        inherited(&module_declaration.vis)
            && module_declaration.content.is_none()
            && only_doc_attributes(&module_declaration.attrs),
        "decoder_full_layer_session must be one unconditional private out-of-line module"
    );

    let module = parse(CAUSAL_MODULE);
    let mut forbidden = ForbiddenCausalSyntax::default();
    forbidden.visit_file(&module);
    assert_eq!(forbidden.unsafe_blocks, 0, "causal module contains unsafe");
    assert_eq!(
        forbidden.macro_invocations, 0,
        "causal module invokes a macro"
    );
    assert_eq!(
        forbidden.derive_attributes, 0,
        "causal module uses a derive macro"
    );
    assert_eq!(
        forbidden.cfg_attributes, 0,
        "causal module contains cfg/cfg_attr decoy paths"
    );
    assert_eq!(
        forbidden.type_aliases, 0,
        "causal module hides authority behind a type alias"
    );
    assert_eq!(forbidden.statics, 0, "causal module contains global state");
    assert_eq!(
        forbidden.renamed_imports, 0,
        "causal module hides authority behind a renamed import"
    );
    assert_eq!(
        forbidden.nested_modules, 0,
        "causal module must not contain a nested or out-of-line escape module"
    );
    assert_eq!(
        forbidden.implementation_blocks, 2,
        "causal module must contain only the sealed authority and session implementations"
    );

    let authorities = module
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Struct(item) if item.ident == AUTHORITY => Some(item),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        authorities.len(),
        1,
        "sealed DecoderFullLayerSessionAuthority must be declared exactly once"
    );
    let authority = authorities[0];
    assert!(
        restricted_to_parent(&authority.vis)
            && authority.generics.params.is_empty()
            && only_doc_attributes(&authority.attrs),
        "sealed authority must be an unconditional non-generic pub(super) type"
    );
    let authority_fields = authority.fields.iter().collect::<Vec<_>>();
    assert_eq!(
        authority_fields.len(),
        3,
        "sealed authority must own exactly its device, queue, and async session owner"
    );
    let device_field = authority_fields[0];
    assert!(
        inherited(&device_field.vis)
            && only_doc_attributes(&device_field.attrs)
            && device_field
                .ident
                .as_ref()
                .is_some_and(|name| name == "device")
            && exact_wgpu_type(&device_field.ty, "Device"),
        "sealed authority must privately own its exact wgpu::Device first"
    );
    let queue_field = authority_fields[1];
    assert!(
        inherited(&queue_field.vis)
            && only_doc_attributes(&queue_field.attrs)
            && queue_field
                .ident
                .as_ref()
                .is_some_and(|name| name == "queue")
            && exact_wgpu_type(&queue_field.ty, "Queue"),
        "sealed authority must privately own its exact wgpu::Queue second"
    );
    let owner_field = authority_fields[2];
    assert!(
        inherited(&owner_field.vis)
            && only_doc_attributes(&owner_field.attrs)
            && owner_field
                .ident
                .as_ref()
                .is_some_and(|name| name == "owner")
            && exact_async_session_owner(&owner_field.ty, SESSION),
        "sealed authority must privately and directly own exactly crate::AsyncSessionOwner<BrowserDecoderFullLayerSession> last"
    );

    let sessions = module
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Struct(item) if item.ident == SESSION => Some(item),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sessions.len(),
        1,
        "private BrowserDecoderFullLayerSession must be declared exactly once"
    );
    let session = sessions[0];
    assert!(
        inherited(&session.vis)
            && session.generics.params.is_empty()
            && only_doc_attributes(&session.attrs),
        "browser decoder full layer session must be unconditional, private, and non-generic"
    );
    let mut field_class_counts = [0usize; 7];
    for field in &session.fields {
        assert!(
            inherited(&field.vis) && only_doc_attributes(&field.attrs),
            "browser decoder full layer session fields must all be private"
        );
        let Some(class) = classify_session_field(&field.ty) else {
            panic!(
                "browser decoder full layer session field hides state outside the closed persistent algebra"
            )
        };
        field_class_counts[class as usize] += 1;
    }
    assert_eq!(
        field_class_counts[SessionFieldClass::WgpuWebGpuBuffer as usize],
        45,
        "browser decoder full layer session must directly own its forty-five exact wgpu::webgpu::GpuBuffer resources"
    );
    assert_eq!(
        field_class_counts[SessionFieldClass::JsObjectHandle as usize],
        22,
        "browser decoder full layer session must directly own its seven pipeline and fifteen bind-group handles"
    );
    assert_eq!(
        field_class_counts[SessionFieldClass::CoreKvPlan as usize],
        1,
        "browser decoder full layer session must hold exactly one exact core DecoderKvSessionPlan"
    );
    assert_eq!(
        field_class_counts[SessionFieldClass::CoreLayerPlan as usize],
        1,
        "browser decoder full layer session must hold exactly one exact core DecoderLayerPlan"
    );
    assert_eq!(
        field_class_counts[SessionFieldClass::Scalar as usize],
        3,
        "browser decoder full layer session must keep exactly its cache position and poison/ready state in plain scalars"
    );
    assert_eq!(
        field_class_counts[SessionFieldClass::Digest as usize],
        7,
        "browser decoder full layer session must keep exactly the seven canonical shader digests"
    );

    let authority_impls = matching_impls(&module, AUTHORITY);
    assert_eq!(
        authority_impls.len(),
        1,
        "sealed authority must have one inherent implementation"
    );
    let authority_impl = authority_impls[0];
    assert!(
        authority_impl.trait_.is_none()
            && authority_impl.generics.params.is_empty()
            && only_doc_attributes(&authority_impl.attrs),
        "sealed authority implementation must be unconditional and inherent"
    );
    let expected_methods = [
        "abort",
        "begin",
        "begin_with_shader_override",
        "finish",
        "new",
        "shader_sources_json",
        "step",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    let actual_methods = authority_impl
        .items
        .iter()
        .filter_map(|item| match item {
            ImplItem::Fn(function) => {
                assert!(
                    restricted_to_parent(&function.vis)
                        && function.sig.generics.params.is_empty()
                        && only_doc_attributes(&function.attrs),
                    "{} must be an unconditional non-generic pub(super) operation",
                    function.sig.ident
                );
                Some(function.sig.ident.to_string())
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual_methods, expected_methods,
        "sealed authority operation surface drifted"
    );
    let session_impls = matching_impls(&module, SESSION);
    assert_eq!(
        session_impls.len(),
        1,
        "browser decoder full layer session must have one inherent implementation"
    );
    let session_impl = session_impls[0];
    assert!(
        session_impl.trait_.is_none()
            && session_impl.generics.params.is_empty()
            && only_doc_attributes(&session_impl.attrs),
        "browser decoder full layer session implementation must be unconditional and inherent"
    );
    for item in &session_impl.items {
        if let ImplItem::Fn(function) = item {
            assert!(
                inherited(&function.vis)
                    && function.sig.generics.params.is_empty()
                    && only_doc_attributes(&function.attrs),
                "{} must be an unconditional private session operation",
                function.sig.ident
            );
        }
    }

    let begin = authority_impl
        .items
        .iter()
        .find_map(|item| match item {
            ImplItem::Fn(function) if function.sig.ident == "begin" => Some(function),
            _ => None,
        })
        .expect("sealed begin operation is missing");
    let begin_with_override = authority_impl
        .items
        .iter()
        .find_map(|item| match item {
            ImplItem::Fn(function) if function.sig.ident == "begin_with_shader_override" => {
                Some(function)
            }
            _ => None,
        })
        .expect("sealed begin_with_shader_override operation is missing");
    let step = authority_impl
        .items
        .iter()
        .find_map(|item| match item {
            ImplItem::Fn(function) if function.sig.ident == "step" => Some(function),
            _ => None,
        })
        .expect("sealed step operation is missing");
    assert!(
        planner_result_reaches_live_consumer(begin, "plan", 2, "owner"),
        "sealed begin must bind the exact descriptor.plan() results of both the cache and the full layer and feed them to the live session constructor"
    );
    assert!(
        planner_result_reaches_live_consumer(step, "plan_step", 1, "owner"),
        "sealed step must bind the exact full-layer plan_step() result and feed it to the live step executor"
    );
    assert!(
        planner_result_reaches_live_consumer(step, "plan_cache_transition", 1, "owner"),
        "sealed step must bind the exact cache plan_cache_transition() result and feed it to the live step executor"
    );
    let override_plan_calls = count_method_calls(&begin_with_override.block, "plan");
    assert!(
        override_plan_calls == 0
            || (override_plan_calls == 2
                && planner_result_reaches_live_consumer(begin_with_override, "plan", 2, "owner")),
        "sealed begin_with_shader_override must not compute-then-ignore either descriptor.plan()"
    );
    for item in &authority_impl.items {
        let ImplItem::Fn(function) = item else {
            continue;
        };
        let name = function.sig.ident.to_string();
        let plan_calls = count_method_calls(&function.block, "plan");
        let plan_step_calls = count_method_calls(&function.block, "plan_step");
        let transition_calls = count_method_calls(&function.block, "plan_cache_transition");
        match name.as_str() {
            "begin" => {
                assert_eq!(plan_step_calls, 0, "begin replans a step outside step");
                assert_eq!(
                    transition_calls, 0,
                    "begin replans a cache transition outside step"
                );
            }
            "begin_with_shader_override" => {
                assert_eq!(
                    plan_step_calls, 0,
                    "begin_with_shader_override replans a step outside step"
                );
                assert_eq!(
                    transition_calls, 0,
                    "begin_with_shader_override replans a cache transition outside step"
                );
            }
            "step" => {
                assert_eq!(plan_calls, 0, "step replans the session outside begin");
            }
            _ => {
                assert!(
                    plan_calls == 0 && plan_step_calls == 0 && transition_calls == 0,
                    "{name} calls the exact core decoder planner outside its sealed boundary"
                );
            }
        }
    }
    for item in &session_impl.items {
        let ImplItem::Fn(function) = item else {
            continue;
        };
        assert!(
            count_method_calls(&function.block, "plan") == 0
                && count_method_calls(&function.block, "plan_step") == 0
                && count_method_calls(&function.block, "plan_cache_transition") == 0,
            "{} replans the session inside its executor instead of consuming the bound plans",
            function.sig.ident
        );
    }
    for item in &module.items {
        let Item::Fn(function) = item else {
            continue;
        };
        assert!(
            count_method_calls(&function.block, "plan") == 0
                && count_method_calls(&function.block, "plan_step") == 0
                && count_method_calls(&function.block, "plan_cache_transition") == 0,
            "{} recomputes decoder topology outside the sealed planner boundary",
            function.sig.ident
        );
    }
    assert_eq!(
        plan_struct_constructions(&module),
        0,
        "causal module must not hand-construct the exact core decoder plan types"
    );

    for item in &module.items {
        if matches!(
            item,
            Item::Struct(value) if value.ident == AUTHORITY || value.ident == SESSION
        ) || matches!(
            item,
            Item::Impl(value)
                if matches!(
                    terminal_type_name(value.self_ty.as_ref()).as_deref(),
                    Some(AUTHORITY) | Some(SESSION)
                )
        ) {
            continue;
        }
        if let Some(visibility) = item_visibility(item) {
            assert!(
                inherited(visibility),
                "causal module exposes an additional item outside its sealed authority"
            );
        }
    }
}

#[test]
fn web_runtime_has_one_closed_full_layer_owner_and_direct_exact_wasm_delegations() {
    let root = parse(WEB_RS);
    let private_modules = root
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Mod(module) => {
                assert!(
                    inherited(&module.vis)
                        && module.content.is_none()
                        && only_doc_attributes(&module.attrs),
                    "{} must remain a private unconditional out-of-line module",
                    module.ident
                );
                Some(module.ident.to_string())
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        private_modules,
        [
            "decoder_kv_session",
            "decoder_full_layer_session",
            "decoder_layer_session",
            "decoder_stack_session",
            "vision_stack_first_effect",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        "web.rs private module surface drifted"
    );
    for sibling in [
        "crates/pvlc-runtime-web/src/web/vision_stack_first_effect.rs",
        "crates/pvlc-runtime-web/src/web/decoder_kv_session.rs",
        "crates/pvlc-runtime-web/src/web/decoder_layer_session.rs",
    ] {
        let sibling_root = parse(sibling);
        assert!(
            matching_impls(&sibling_root, AUTHORITY).is_empty()
                && matching_impls(&sibling_root, SESSION).is_empty()
                && !named_type_declarations(&sibling_root)
                    .iter()
                    .any(|name| name == AUTHORITY || name == SESSION),
            "{sibling} must not become a decoder full-layer side door"
        );
    }
    assert!(
        matching_impls(
            &parse("crates/pvlc-runtime-web/src/web/vision_stack_first_effect.rs"),
            "WebRuntime"
        )
        .is_empty(),
        "the existing first-effect module must not become a WebRuntime side door"
    );
    assert_eq!(
        public_web_surface(&root),
        [
            "fn:assemble_browser_benchmark_cohort_v1",
            "fn:canonical_vision_encoder_stack_shader_sources_json",
            "fn:validate_browser_benchmark_cohort_plan_v1",
            "struct:WebRuntime",
            "struct:WebVisionQkvStackSelection",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        "web.rs public type/function/module surface drifted"
    );
    assert_eq!(
        public_inherent_methods(&root, "WebRuntime"),
        [
            "abort_decoder_kv_session",
            "abort_decoder_layer_session",
            "abort_decoder_full_layer_session",
            "abort_decoder_stack_session",
            "abort_vision_encoder_stack_sharded",
            "begin_decoder_kv_session",
            "begin_decoder_kv_session_with_shader_override",
            "begin_decoder_layer_session",
            "begin_decoder_layer_session_with_shader_override",
            "begin_decoder_full_layer_session",
            "begin_decoder_full_layer_session_with_shader_override",
            "begin_decoder_stack_session",
            "begin_decoder_stack_session_resident",
            "begin_decoder_stack_session_with_shader_override",
            "begin_vision_encoder_stack_sharded_json",
            "begin_vision_encoder_stack_sharded_resident_with_activation_strategy_and_qkv_selection_json",
            "begin_vision_encoder_stack_sharded_with_activation_strategy_and_memory_hardening_and_qkv_selection_json",
            "begin_vision_encoder_stack_sharded_with_activation_strategy_and_memory_hardening_json",
            "begin_vision_encoder_stack_sharded_with_activation_strategy_and_qkv_selection_json",
            "begin_vision_encoder_stack_sharded_with_activation_strategy_json",
            "blake3_bytes_hex",
            "blake3_hex",
            "capabilities_json",
            "compile_vision_encoder_stack_qkv_selection",
            "configure_vision_encoder_stack_spatial_rope_f32",
            "create",
            "decoder_kv_session_shader_sources_json",
            "decoder_layer_session_shader_sources_json",
            "decoder_full_layer_session_shader_sources_json",
            "decoder_stack_resident_weights_json",
            "decoder_stack_session_shader_sources_json",
            "enqueue_vision_encoder_stack_sharded_layer_json",
            "enqueue_vision_encoder_stack_sharded_resident_layer_json",
            "finish_decoder_kv_session",
            "finish_decoder_layer_session",
            "finish_decoder_full_layer_session",
            "finish_decoder_stack_session",
            "finish_vision_encoder_stack_sharded",
            "finish_vision_encoder_stack_sharded_resident",
            "finish_vision_encoder_stack_sharded_resident_with_projector_f16",
            "has_projector_f16_resident_weights",
            "has_vision_encoder_stack_resident_weights",
            "logits_decoder_stack_session",
            "prefill_decoder_stack_session",
            "preflight_vision_encoder_stack_manifest_shard_json",
            "preflight_vision_encoder_stack_shard_json",
            "prepare_projector_f16_resident_weights",
            "probe_m7q1_precision_admission_json",
            "probe_validation_error_json",
            "projector_shader_sources_json",
            "run_json",
            "run_m7q1_fp16_weight_probe_json",
            "run_projector_bytes",
            "run_projector_f16_resident_bytes",
            "run_projector_json",
            "run_projector_with_shader_override_json",
            "run_vision_encoder_layer_identity_rope_bytes",
            "run_vision_encoder_layer_identity_rope_json",
            "run_vision_encoder_layer_identity_rope_with_shader_override_json",
            "run_vision_encoder_stack_sharded_layer_json",
            "run_vision_patch_projection_bytes",
            "run_with_shader_json",
            "start_vision_encoder_stack_sharded_json",
            "step_decoder_kv_session",
            "step_decoder_layer_session",
            "step_decoder_full_layer_session",
            "step_decoder_stack_session",
            "top1_decoder_stack_session",
            "validate_all_pipelines_json",
            "validate_projector_pipelines_json",
            "validate_vision_attention_pipeline_json",
            "validate_vision_encoder_layer_pipelines_json",
            "vision_encoder_layer_shader_sources_json",
            "vision_encoder_stack_qkv_shader_sources_json",
            "vision_encoder_stack_shader_sources_json",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        "all public WebRuntime methods, including non-wasm impls, must match the closed allowlist"
    );
    let foreign_authority_impls = matching_impls(&root, AUTHORITY).len();
    assert_eq!(
        foreign_authority_impls, 0,
        "web.rs must not extend the sealed decoder full layer authority"
    );
    for alias in root.items.iter().filter_map(|item| match item {
        Item::Type(alias) => Some(alias),
        _ => None,
    }) {
        assert!(
            !type_names_in_type(&alias.ty).contains(AUTHORITY),
            "web.rs must not hide the decoder full layer authority behind a type alias"
        );
    }

    let runtime = root
        .items
        .iter()
        .find_map(|item| match item {
            Item::Struct(item) if item.ident == "WebRuntime" => Some(item),
            _ => None,
        })
        .expect("WebRuntime is missing");
    let owner_fields = runtime
        .fields
        .iter()
        .filter(|field| type_names(field).contains(AUTHORITY))
        .collect::<Vec<_>>();
    assert_eq!(
        owner_fields.len(),
        1,
        "WebRuntime must own exactly one DecoderFullLayerSessionAuthority"
    );
    let owner_field = owner_fields[0];
    assert!(
        inherited(&owner_field.vis)
            && owner_field
                .ident
                .as_ref()
                .is_some_and(|name| name == OWNER_FIELD)
            && terminal_type_name(&owner_field.ty).as_deref() == Some(AUTHORITY)
            && only_doc_attributes(&owner_field.attrs),
        "WebRuntime decoder full layer authority must be one unconditional private named field"
    );

    let expected_delegations = [
        ("abort_decoder_full_layer_session", "abort"),
        ("begin_decoder_full_layer_session", "begin"),
        (
            "begin_decoder_full_layer_session_with_shader_override",
            "begin_with_shader_override",
        ),
        (
            "decoder_full_layer_session_shader_sources_json",
            "shader_sources_json",
        ),
        ("finish_decoder_full_layer_session", "finish"),
        ("step_decoder_full_layer_session", "step"),
    ];
    let wasm_impls = root
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Impl(item)
                if terminal_type_name(item.self_ty.as_ref()).as_deref() == Some("WebRuntime")
                    && has_attribute(&item.attrs, "wasm_bindgen") =>
            {
                Some(item)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        wasm_impls.len(),
        1,
        "WebRuntime must have one wasm_bindgen inherent implementation"
    );
    let wasm_impl = wasm_impls[0];
    for (exported_name, authority_method) in expected_delegations {
        let functions = wasm_impl
            .items
            .iter()
            .filter_map(|item| match item {
                ImplItem::Fn(function) if function.sig.ident == exported_name => Some(function),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            functions.len(),
            1,
            "{exported_name} must be exported exactly once"
        );
        let function = functions[0];
        assert!(
            matches!(function.vis, Visibility::Public(_))
                && cfg_free(&function.attrs)
                && function.sig.generics.params.is_empty(),
            "{exported_name} must be an unconditional non-generic public WASM method"
        );
        assert_direct_authority_call(function, authority_method, OWNER_FIELD);
    }

    let exported_layer_methods = wasm_impl
        .items
        .iter()
        .filter_map(|item| match item {
            ImplItem::Fn(function)
                if function
                    .sig
                    .ident
                    .to_string()
                    .contains("decoder_full_layer_session") =>
            {
                Some(function.sig.ident.to_string())
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        exported_layer_methods,
        expected_delegations
            .into_iter()
            .map(|(name, _)| name.to_owned())
            .collect(),
        "decoder full layer WASM operation allowlist drifted"
    );

    let crate_root = parse("crates/pvlc-runtime-web/src/lib.rs");
    let mut public_web_reexports = BTreeSet::new();
    for item in &crate_root.items {
        let Item::Use(item_use) = item else {
            continue;
        };
        if matches!(item_use.vis, Visibility::Public(_)) {
            collect_use_paths(&item_use.tree, "", &mut public_web_reexports);
        }
    }
    assert_eq!(
        public_web_reexports,
        ["web::WebRuntime", "web::WebVisionQkvStackSelection"]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        "crate root web re-export surface drifted"
    );
}

#[test]
fn sealing_helpers_require_direct_owner_and_allow_only_documentation_attributes() {
    let direct = syn::parse_str::<Type>("crate::AsyncSessionOwner<BrowserDecoderFullLayerSession>")
        .expect("direct owner type must parse");
    let unqualified = syn::parse_str::<Type>("AsyncSessionOwner<BrowserDecoderFullLayerSession>")
        .expect("unqualified owner type must parse");
    let decoy = syn::parse_str::<Type>("decoy::AsyncSessionOwner<BrowserDecoderFullLayerSession>")
        .expect("decoy owner type must parse");
    let wrapped =
        syn::parse_str::<Type>("RefCell<AsyncSessionOwner<BrowserDecoderFullLayerSession>>")
            .expect("wrapped owner type must parse");
    let tuple = syn::parse_str::<Type>(
        "(AsyncSessionOwner<BrowserDecoderFullLayerSession>, ExtraAuthority)",
    )
    .expect("tuple owner type must parse");
    assert!(exact_async_session_owner(&direct, SESSION));
    assert!(!exact_async_session_owner(&unqualified, SESSION));
    assert!(!exact_async_session_owner(&decoy, SESSION));
    assert!(!exact_async_session_owner(&wrapped, SESSION));
    assert!(!exact_async_session_owner(&tuple, SESSION));

    let documented: Item = syn::parse_str("#[doc = \"sealed\"] struct Documented;")
        .expect("documented item must parse");
    let configured: Item = syn::parse_str("#[cfg(target_arch = \"wasm32\")] struct Configured;")
        .expect("configured item must parse");
    let Item::Struct(documented) = documented else {
        panic!("documented fixture is not a struct");
    };
    let Item::Struct(configured) = configured else {
        panic!("configured fixture is not a struct");
    };
    assert!(only_doc_attributes(&documented.attrs));
    assert!(!only_doc_attributes(&configured.attrs));

    let nested = syn::parse_file("mod escape { impl super::WebRuntime {} }")
        .expect("nested-module fixture must parse");
    let mut forbidden = ForbiddenCausalSyntax::default();
    forbidden.visit_file(&nested);
    assert_eq!(forbidden.nested_modules, 1);
    assert_eq!(forbidden.implementation_blocks, 1);
}

#[test]
fn delegation_helper_requires_exact_argument_forwarding() {
    let file = syn::parse_file(
        r#"
        impl WebRuntime {
            pub fn step_decoder_full_layer_session(&self, hidden: Bytes) -> Result {
                self.decoder_full_layer_session.step(hidden)
            }
            pub fn swapped(&self, hidden: Bytes, extra: Bytes) -> Result {
                self.decoder_full_layer_session.step(extra)
            }
        }
        "#,
    )
    .expect("delegation fixture must parse");
    let Item::Impl(implementation) = &file.items[0] else {
        panic!("delegation fixture is not an impl");
    };
    let methods = implementation
        .items
        .iter()
        .filter_map(|item| match item {
            ImplItem::Fn(function) => Some(function),
            _ => None,
        })
        .collect::<Vec<_>>();
    let syn::Stmt::Expr(direct, None) = &methods[0].block.stmts[0] else {
        panic!("direct delegation fixture drifted");
    };
    let syn::Stmt::Expr(swapped, None) = &methods[1].block.stmts[0] else {
        panic!("swapped delegation fixture drifted");
    };
    assert!(exact_authority_call(
        methods[0],
        direct,
        "step",
        OWNER_FIELD
    ));
    assert!(!exact_authority_call(
        methods[1],
        swapped,
        "step",
        OWNER_FIELD
    ));
}

#[test]
fn planner_results_must_reach_live_consumers() {
    let live_begin = parse_impl_method(
        r#"
        impl DecoderFullLayerSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let kv_plan = kv_descriptor.plan()?;
                let layer_plan = layer_descriptor.plan()?;
                let session = BrowserDecoderFullLayerSession::create(&self.device, &self.queue, kv_plan, layer_plan)?;
                let generation = self.owner.begin(session)?;
                Ok(generation.to_string())
            }
        }
        "#,
        "begin",
    );
    assert!(planner_result_reaches_live_consumer(
        &live_begin,
        "plan",
        2,
        "owner"
    ));

    let live_step = parse_impl_method(
        r#"
        impl DecoderFullLayerSessionAuthority {
            fn step(&self, hidden: Bytes) -> Result<String, Error> {
                let (lease, mut session) = self.owner.acquire()?;
                let transition = session.kv_plan.plan_cache_transition(session.cache_tokens)?;
                let step_plan = session.layer_plan.plan_step(transition.cache_tokens_before, &hidden)?;
                session.cache_tokens = transition.cache_tokens_after;
                session.pending_uniform_words = step_plan.stage_uniform_words;
                let outcome = self.owner.complete(lease, session, CompletionAction::Restore);
                Ok(format!("{outcome:?}"))
            }
        }
        "#,
        "step",
    );
    assert!(planner_result_reaches_live_consumer(
        &live_step,
        "plan_step",
        1,
        "owner"
    ));
    assert!(planner_result_reaches_live_consumer(
        &live_step,
        "plan_cache_transition",
        1,
        "owner"
    ));

    let ignored_second_plan = parse_impl_method(
        r#"
        impl DecoderFullLayerSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let kv_plan = kv_descriptor.plan()?;
                let layer_plan = layer_descriptor.plan()?;
                let session = BrowserDecoderFullLayerSession::create(&self.device, &self.queue, kv_plan)?;
                let generation = self.owner.begin(session)?;
                Ok(generation.to_string())
            }
        }
        "#,
        "begin",
    );
    assert!(!planner_result_reaches_live_consumer(
        &ignored_second_plan,
        "plan",
        2,
        "owner"
    ));

    let dead_nested_call = parse_impl_method(
        r#"
        impl DecoderFullLayerSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                if false {
                    descriptor.plan();
                    layer_descriptor.plan();
                }
                fallback()
            }
        }
        "#,
        "begin",
    );
    assert!(!planner_result_reaches_live_consumer(
        &dead_nested_call,
        "plan",
        2,
        "owner"
    ));

    let duplicated_call = parse_impl_method(
        r#"
        impl DecoderFullLayerSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let kv_plan = kv_descriptor.plan()?;
                let second = kv_descriptor.plan()?;
                let layer_plan = layer_descriptor.plan()?;
                let session = BrowserDecoderFullLayerSession::create(&self.device, &self.queue, second, layer_plan)?;
                let generation = self.owner.begin(session)?;
                Ok(generation.to_string())
            }
        }
        "#,
        "begin",
    );
    assert!(!planner_result_reaches_live_consumer(
        &duplicated_call,
        "plan",
        2,
        "owner"
    ));

    // A rebinding that still flows to the live constructor keeps the plan
    // alive: the transitive closure must follow `shadow` back to its init.
    let transitive_shadow = parse_impl_method(
        r#"
        impl DecoderFullLayerSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let kv_plan = kv_descriptor.plan()?;
                let layer_plan = layer_descriptor.plan()?;
                let shadow = layer_plan;
                let session = BrowserDecoderFullLayerSession::create(&self.device, &self.queue, kv_plan, shadow)?;
                let generation = self.owner.begin(session)?;
                Ok(generation.to_string())
            }
        }
        "#,
        "begin",
    );
    assert!(planner_result_reaches_live_consumer(
        &transitive_shadow,
        "plan",
        2,
        "owner"
    ));

    let dead_field_sink = parse_impl_method(
        r#"
        impl DecoderFullLayerSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let kv_plan = kv_descriptor.plan()?;
                let layer_plan = layer_descriptor.plan()?;
                let mut decoy = Decoy::new();
                decoy.plan = layer_plan;
                let session = BrowserDecoderFullLayerSession::create(&self.device, &self.queue, kv_plan)?;
                let generation = self.owner.begin(session)?;
                Ok(generation.to_string())
            }
        }
        "#,
        "begin",
    );
    assert!(!planner_result_reaches_live_consumer(
        &dead_field_sink,
        "plan",
        2,
        "owner"
    ));

    let ignored_step_plan = parse_impl_method(
        r#"
        impl DecoderFullLayerSessionAuthority {
            fn step(&self, hidden: Bytes) -> Result<String, Error> {
                let (lease, mut session) = self.owner.acquire()?;
                let transition = session.kv_plan.plan_cache_transition(session.cache_tokens)?;
                let step_plan = session.layer_plan.plan_step(session.cache_tokens, &hidden)?;
                session.cache_tokens = transition.cache_tokens_after;
                let outcome = self.owner.complete(lease, session, CompletionAction::Restore);
                Ok(format!("{outcome:?}"))
            }
        }
        "#,
        "step",
    );
    assert!(!planner_result_reaches_live_consumer(
        &ignored_step_plan,
        "plan_step",
        1,
        "owner"
    ));

    let ignored_transition = parse_impl_method(
        r#"
        impl DecoderFullLayerSessionAuthority {
            fn step(&self, hidden: Bytes) -> Result<String, Error> {
                let (lease, mut session) = self.owner.acquire()?;
                let transition = session.kv_plan.plan_cache_transition(session.cache_tokens)?;
                let step_plan = session.layer_plan.plan_step(session.cache_tokens, &hidden)?;
                session.cache_tokens = step_plan.position;
                let outcome = self.owner.complete(lease, session, CompletionAction::Restore);
                Ok(format!("{outcome:?}"))
            }
        }
        "#,
        "step",
    );
    assert!(!planner_result_reaches_live_consumer(
        &ignored_transition,
        "plan_cache_transition",
        1,
        "owner"
    ));

    // The accepted prepared-closure idiom: planner calls nested inside one
    // `let prepared = (|| { ... })();` count as nested `let` bindings, and
    // every returned plan flows to the live async tail.
    let live_closure_begin = parse_impl_method(
        r#"
        impl DecoderFullLayerSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<Promise, Error> {
                let prepared = (|| {
                    let kv_plan = kv_descriptor.plan()?;
                    let layer_plan = layer_descriptor.plan()?;
                    Ok((kv_plan, layer_plan, sources()))
                })();
                let (kv_plan, layer_plan, sources) = match prepared {
                    Ok(prepared) => prepared,
                    Err(error) => return reject(error),
                };
                future_to_promise(async move {
                    run_begin(kv_plan, layer_plan, sources).await
                })
            }
        }
        "#,
        "begin",
    );
    assert!(planner_result_reaches_live_consumer(
        &live_closure_begin,
        "plan",
        2,
        "owner"
    ));

    // The same closure idiom with one plan dropped before the return tuple is
    // an ignored-result forgery even though every call is `let`-bound.
    let closure_dropped_plan = parse_impl_method(
        r#"
        impl DecoderFullLayerSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<Promise, Error> {
                let prepared = (|| {
                    let kv_plan = kv_descriptor.plan()?;
                    let layer_plan = layer_descriptor.plan()?;
                    Ok((kv_plan, sources()))
                })();
                let (kv_plan, sources) = match prepared {
                    Ok(prepared) => prepared,
                    Err(error) => return reject(error),
                };
                future_to_promise(async move {
                    run_begin(kv_plan, sources).await
                })
            }
        }
        "#,
        "begin",
    );
    assert!(!planner_result_reaches_live_consumer(
        &closure_dropped_plan,
        "plan",
        2,
        "owner"
    ));

    let forged_plans: File = syn::parse_file(
        r#"
        fn forge() {
            let _ = pvlc_runtime_core::DecoderKvSessionPlan {};
            let _ = pvlc_runtime_core::DecoderKvSessionStepPlan {};
            let _ = pvlc_runtime_core::DecoderLayerPlan {};
            let _ = pvlc_runtime_core::DecoderLayerStepPlan {};
        }
        "#,
    )
    .expect("forged-plan fixture must parse");
    assert_eq!(plan_struct_constructions(&forged_plans), 4);
}

#[test]
fn session_field_algebra_rejects_host_shadows_and_custom_wrappers() {
    for (source, expected) in [
        (
            "wgpu::webgpu::GpuBuffer",
            SessionFieldClass::WgpuWebGpuBuffer,
        ),
        ("js_sys::Object", SessionFieldClass::JsObjectHandle),
        (
            "pvlc_runtime_core::DecoderKvSessionPlan",
            SessionFieldClass::CoreKvPlan,
        ),
        (
            "pvlc_runtime_core::DecoderAttentionBlockPlan",
            SessionFieldClass::CoreAttentionPlan,
        ),
        (
            "pvlc_runtime_core::DecoderLayerPlan",
            SessionFieldClass::CoreLayerPlan,
        ),
        ("bool", SessionFieldClass::Scalar),
        ("u32", SessionFieldClass::Scalar),
        ("u64", SessionFieldClass::Scalar),
        ("usize", SessionFieldClass::Scalar),
        ("[u8; 32]", SessionFieldClass::Digest),
    ] {
        let ty = syn::parse_str::<Type>(source).expect("accepted field type must parse");
        assert_eq!(
            classify_session_field(&ty),
            Some(expected),
            "{source} must stay inside the closed persistent algebra"
        );
    }
    for source in [
        "Vec<f32>",
        "Vec<u8>",
        "Box<[f32]>",
        "Box<wgpu::webgpu::GpuBuffer>",
        "String",
        "std::string::String",
        "std::collections::BTreeMap<u32, u32>",
        "&'static [f32]",
        "&[u8]",
        "std::sync::OnceLock<wgpu::webgpu::GpuBuffer>",
        "std::sync::LazyLock<wgpu::webgpu::GpuBuffer>",
        "std::borrow::Cow<'static, [f32]>",
        "std::rc::Rc<wgpu::webgpu::GpuBuffer>",
        "HostCache",
        "SessionBuffers",
        "wgpu::Buffer",
        "wgpu::ComputePipeline",
        "wgpu::BindGroup",
        "wgpu::Device",
        "wgpu::Queue",
        "crate::AsyncSessionOwner<BrowserDecoderFullLayerSession>",
        "wgpu::webgpu::GpuBuffer<'_>",
        "webgpu::GpuBuffer",
        "wgpu::GpuBuffer",
        "js_sys::Uint8Array",
        "js_sys::Array",
        "Object",
        "[f32; 4]",
        "[u8; 16]",
        "[u16; 32]",
        "[[u8; 32]; 2]",
        // M6e7 amendment: Option<wgpu::webgpu::GpuBuffer> is the accepted
        // optional prefill-extension shape and stays inside the algebra.
        "Option<pvlc_runtime_core::DecoderLayerPlan>",
        "pvlc_runtime_core::DecoderKvSessionStepPlan",
        "pvlc_runtime_core::DecoderAttentionBlockStepPlan",
        "pvlc_runtime_core::DecoderLayerStepPlan",
        "pvlc_runtime_core::DecoderLayerStep<'_>",
        "crate::DecoderKvSessionPlan",
        "super::DecoderKvSessionPlan",
        "crate::DecoderLayerPlan",
        "super::DecoderLayerPlan",
        "<wgpu::webgpu::GpuBuffer as Trait>::Assoc",
    ] {
        let ty = syn::parse_str::<Type>(source).expect("rejected field type must parse");
        assert_eq!(
            classify_session_field(&ty),
            None,
            "{source} must fall outside the closed persistent algebra"
        );
    }

    let session = syn::parse_str::<syn::ItemStruct>(
        r#"
        struct BrowserDecoderFullLayerSession {
            kv_plan: pvlc_runtime_core::DecoderKvSessionPlan,
            layer_plan: pvlc_runtime_core::DecoderLayerPlan,
            cache_tokens: u32,
            poisoned: bool,
            ready: bool,
            rms_norm_shader_blake3: [u8; 32],
            gemv_shader_blake3: [u8; 32],
            mrope_shader_blake3: [u8; 32],
            append_shader_blake3: [u8; 32],
            attention_shader_blake3: [u8; 32],
            swiglu_shader_blake3: [u8; 32],
            residual_shader_blake3: [u8; 32],
            hidden_token_buffer: wgpu::webgpu::GpuBuffer,
            norm1_weight_buffer: wgpu::webgpu::GpuBuffer,
            q_weight_buffer: wgpu::webgpu::GpuBuffer,
            k_weight_buffer: wgpu::webgpu::GpuBuffer,
            v_weight_buffer: wgpu::webgpu::GpuBuffer,
            o_weight_buffer: wgpu::webgpu::GpuBuffer,
            rope_cos_buffer: wgpu::webgpu::GpuBuffer,
            rope_sin_buffer: wgpu::webgpu::GpuBuffer,
            norm2_weight_buffer: wgpu::webgpu::GpuBuffer,
            gate_weight_buffer: wgpu::webgpu::GpuBuffer,
            up_weight_buffer: wgpu::webgpu::GpuBuffer,
            down_weight_buffer: wgpu::webgpu::GpuBuffer,
            norm1_buffer: wgpu::webgpu::GpuBuffer,
            q_projection_buffer: wgpu::webgpu::GpuBuffer,
            k_projection_buffer: wgpu::webgpu::GpuBuffer,
            v_projection_buffer: wgpu::webgpu::GpuBuffer,
            mrope_query_buffer: wgpu::webgpu::GpuBuffer,
            mrope_key_buffer: wgpu::webgpu::GpuBuffer,
            key_cache_buffer: wgpu::webgpu::GpuBuffer,
            value_cache_buffer: wgpu::webgpu::GpuBuffer,
            attention_output_buffer: wgpu::webgpu::GpuBuffer,
            o_projection_buffer: wgpu::webgpu::GpuBuffer,
            attention_residual_buffer: wgpu::webgpu::GpuBuffer,
            norm2_buffer: wgpu::webgpu::GpuBuffer,
            gate_buffer: wgpu::webgpu::GpuBuffer,
            up_buffer: wgpu::webgpu::GpuBuffer,
            activation_buffer: wgpu::webgpu::GpuBuffer,
            down_projection_buffer: wgpu::webgpu::GpuBuffer,
            layer_output_buffer: wgpu::webgpu::GpuBuffer,
            layer_readback_buffer: wgpu::webgpu::GpuBuffer,
            rms_uniform_buffer: wgpu::webgpu::GpuBuffer,
            gemv_q_uniform_buffer: wgpu::webgpu::GpuBuffer,
            gemv_k_uniform_buffer: wgpu::webgpu::GpuBuffer,
            gemv_v_uniform_buffer: wgpu::webgpu::GpuBuffer,
            mrope_uniform_buffer: wgpu::webgpu::GpuBuffer,
            append_uniform_buffer: wgpu::webgpu::GpuBuffer,
            attention_uniform_buffer: wgpu::webgpu::GpuBuffer,
            residual_uniform_buffer: wgpu::webgpu::GpuBuffer,
            rms2_uniform_buffer: wgpu::webgpu::GpuBuffer,
            gemv_gate_uniform_buffer: wgpu::webgpu::GpuBuffer,
            gemv_up_uniform_buffer: wgpu::webgpu::GpuBuffer,
            swiglu_uniform_buffer: wgpu::webgpu::GpuBuffer,
            gemv_down_uniform_buffer: wgpu::webgpu::GpuBuffer,
            residual2_uniform_buffer: wgpu::webgpu::GpuBuffer,
            gemv_o_uniform_buffer: wgpu::webgpu::GpuBuffer,
            rms_norm_pipeline: js_sys::Object,
            gemv_pipeline: js_sys::Object,
            mrope_pipeline: js_sys::Object,
            append_pipeline: js_sys::Object,
            attention_pipeline: js_sys::Object,
            swiglu_pipeline: js_sys::Object,
            residual_pipeline: js_sys::Object,
            rms_bind_group: js_sys::Object,
            gemv_q_bind_group: js_sys::Object,
            gemv_k_bind_group: js_sys::Object,
            gemv_v_bind_group: js_sys::Object,
            mrope_bind_group: js_sys::Object,
            append_bind_group: js_sys::Object,
            attention_bind_group: js_sys::Object,
            gemv_o_bind_group: js_sys::Object,
            residual_bind_group: js_sys::Object,
            rms2_bind_group: js_sys::Object,
            gemv_gate_bind_group: js_sys::Object,
            gemv_up_bind_group: js_sys::Object,
            swiglu_bind_group: js_sys::Object,
            gemv_down_bind_group: js_sys::Object,
            residual2_bind_group: js_sys::Object,
        }
        "#,
    )
    .expect("canonical session fixture must parse");
    let mut counts = [0usize; 7];
    for field in &session.fields {
        let class = classify_session_field(&field.ty).expect("canonical field must classify");
        counts[class as usize] += 1;
    }
    assert_eq!(counts[SessionFieldClass::WgpuWebGpuBuffer as usize], 45);
    assert_eq!(counts[SessionFieldClass::JsObjectHandle as usize], 22);
    assert_eq!(counts[SessionFieldClass::CoreKvPlan as usize], 1);
    assert_eq!(counts[SessionFieldClass::CoreLayerPlan as usize], 1);
    assert_eq!(counts[SessionFieldClass::Scalar as usize], 3);
    assert_eq!(counts[SessionFieldClass::Digest as usize], 7);

    let shadowed = syn::parse_str::<syn::ItemStruct>(
        r#"
        struct BrowserDecoderFullLayerSession {
            cache_tokens: u32,
            kv_plan: pvlc_runtime_core::DecoderKvSessionPlan,
            layer_plan: pvlc_runtime_core::DecoderLayerPlan,
            key_cache_buffer: wgpu::webgpu::GpuBuffer,
            host_cache: Vec<f32>,
        }
        "#,
    )
    .expect("shadowed session fixture must parse");
    assert!(
        shadowed
            .fields
            .iter()
            .any(|field| classify_session_field(&field.ty).is_none()),
        "a Vec<f32> host cache must escape the closed persistent algebra"
    );
}

#[test]
fn decoder_full_layer_authority_has_no_crate_wide_side_doors() {
    let src_root = workspace_path("crates/pvlc-runtime-web/src");
    let mut files = Vec::new();
    collect_source_files(&src_root, &mut files);
    assert!(
        files.len() >= 5,
        "crate source walk missed files: {}",
        files.len()
    );
    for file in files {
        let relative = file
            .strip_prefix(&src_root)
            .expect("source file must live under src")
            .to_path_buf();
        let manifest_relative =
            format!("crates/pvlc-runtime-web/src/{}", relative.to_string_lossy());
        let root = parse(&manifest_relative);
        let is_web_rs = relative == std::path::Path::new("web.rs");
        let is_causal_module = relative
            == std::path::Path::new("web")
                .join("decoder_full_layer_session.rs")
                .as_path();
        if !is_web_rs {
            assert!(
                matching_impls(&root, "WebRuntime").is_empty(),
                "{} extends WebRuntime outside web.rs",
                relative.to_string_lossy()
            );
            assert!(
                !named_type_declarations(&root)
                    .iter()
                    .any(|name| name == "WebRuntime"),
                "{} redeclares WebRuntime outside web.rs",
                relative.to_string_lossy()
            );
        }
        if !is_causal_module {
            for sealed in [AUTHORITY, SESSION] {
                assert!(
                    matching_impls(&root, sealed).is_empty(),
                    "{} extends {sealed} outside its causal module",
                    relative.to_string_lossy()
                );
                assert!(
                    !named_type_declarations(&root)
                        .iter()
                        .any(|name| name == sealed),
                    "{} redeclares {sealed} outside its causal module",
                    relative.to_string_lossy()
                );
            }
        }
        let mut aliases = AliasTypes::default();
        aliases.visit_file(&root);
        for alias in &aliases.0 {
            let names = type_names_in_type(alias);
            assert!(
                !names.contains(AUTHORITY) && !names.contains(SESSION),
                "{} hides the sealed decoder full layer session behind a type alias",
                relative.to_string_lossy()
            );
        }
    }
}

#[test]
fn public_method_allowlist_observes_non_wasm_impls() {
    let file = syn::parse_file(
        r#"
        #[wasm_bindgen]
        impl WebRuntime {
            pub fn exported(&self) {}
        }
        impl WebRuntime {
            pub fn hidden_handle(&self) {}
            fn private_helper(&self) {}
        }
        mod escape {
            impl super::WebRuntime {
                pub fn nested_handle(&self) {}
            }
        }
        "#,
    )
    .expect("public-method fixture must parse");
    assert_eq!(
        public_inherent_methods(&file, "WebRuntime"),
        ["exported", "hidden_handle", "nested_handle"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    );
}
