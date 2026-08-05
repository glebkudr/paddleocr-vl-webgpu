//! Structural contract for the sealed M6e6 persistent browser decoder stack
//! session module before any production exists.

#[path = "decoder_session_contract_helpers.rs"]
mod helpers;

use std::collections::BTreeSet;

use helpers::*;
use syn::{File, ImplItem, Item, Type, Visibility, visit::Visit};

const WEB_RS: &str = "crates/pvlc-runtime-web/src/web.rs";
const CAUSAL_MODULE: &str = "crates/pvlc-runtime-web/src/web/decoder_stack_session.rs";
const AUTHORITY: &str = "DecoderStackSessionAuthority";
const SESSION: &str = "BrowserDecoderStackSession";
const OWNER_FIELD: &str = "decoder_stack_session";

#[test]
fn decoder_stack_session_is_one_cfg_free_sealed_causal_module() {
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
        "Web runtime must declare decoder_stack_session exactly once"
    );
    let module_declaration = declarations[0];
    assert!(
        inherited(&module_declaration.vis)
            && module_declaration.content.is_none()
            && only_doc_attributes(&module_declaration.attrs),
        "decoder_stack_session must be one unconditional private out-of-line module"
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
        forbidden.implementation_blocks, 5,
        "causal module must contain only the sealed authority, session, and resident-cache implementations"
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
        "sealed DecoderStackSessionAuthority must be declared exactly once"
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
        4,
        "sealed authority must own exactly its device, queue, async session owner, and shared resident-weight cache"
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
        "sealed authority must privately and directly own exactly crate::AsyncSessionOwner<BrowserDecoderStackSession> third"
    );
    let resident_cache_field = authority_fields[3];
    assert!(
        inherited(&resident_cache_field.vis)
            && only_doc_attributes(&resident_cache_field.attrs)
            && resident_cache_field
                .ident
                .as_ref()
                .is_some_and(|name| name == "resident_weight_cache")
            && terminal_type_name(&resident_cache_field.ty).as_deref()
                == Some("SharedDecoderStackResidentWeightCache"),
        "sealed authority must privately own the shared resident decoder-weight cache last"
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
        "private BrowserDecoderStackSession must be declared exactly once"
    );
    let session = sessions[0];
    assert!(
        inherited(&session.vis)
            && session.generics.params.is_empty()
            && only_doc_attributes(&session.attrs),
        "browser decoder stack session must be unconditional, private, and non-generic"
    );
    let mut field_class_counts = [0usize; 8];
    for field in &session.fields {
        assert!(
            inherited(&field.vis) && only_doc_attributes(&field.attrs),
            "browser decoder stack session fields must all be private"
        );
        // M6e7 amendment: the prefill extension fields (the optional
        // prefill-only resources and the stored exact core prefill plan) are
        // pinned by the M6e7 prefill module contract; the accepted decode-only
        // persistent algebra counted below is unchanged.
        // M6e8 amendment: the optional logits capability plan (present only
        // on logits-capable sessions) and the optional logits resources are
        // pinned by the M6e8 logits module contract.
        // M7q1 amendment: one exact core weight-resource plan owns the
        // authenticated F32/F16 physical byte algebra. Its storage-specific
        // selectors are pinned by the M7q1 FP16 module contract.
        if optional_session_field_class(&field.ty).is_some()
            || exact_core_plan(&field.ty, "DecoderStackPrefillPlan")
            || exact_core_plan(&field.ty, "DecoderWeightResourcePlan")
            || exact_optional_core_plan(&field.ty, "DecoderLmHeadPlan")
        {
            continue;
        }
        let Some(class) = classify_session_field(&field.ty) else {
            panic!(
                "browser decoder stack session field hides state outside the closed persistent algebra"
            )
        };
        field_class_counts[class as usize] += 1;
    }
    assert_eq!(
        field_class_counts[SessionFieldClass::WgpuWebGpuBuffer as usize],
        47,
        "browser decoder stack session must directly own its forty-seven exact wgpu::webgpu::GpuBuffer resources: the accepted forty-four plus the M7o2 scratch partials plane and the two split stage uniforms"
    );
    assert_eq!(
        field_class_counts[SessionFieldClass::JsObjectHandle as usize],
        24,
        "browser decoder stack session must directly own its eight pipeline and sixteen bind-group handles: the accepted twenty-two minus the retired serial GQA pipeline and bind group, plus the M7o2 split pipelines and bind groups"
    );
    assert_eq!(
        field_class_counts[SessionFieldClass::CoreKvPlan as usize],
        1,
        "browser decoder stack session must hold exactly one exact core DecoderKvSessionPlan"
    );
    assert_eq!(
        field_class_counts[SessionFieldClass::CoreStackPlan as usize],
        1,
        "browser decoder stack session must hold exactly one exact core DecoderStackPlan"
    );
    assert_eq!(
        field_class_counts[SessionFieldClass::Scalar as usize],
        4,
        "browser decoder stack session must keep exactly its cache position, poison/ready state, and resident-byte count in plain scalars"
    );
    assert_eq!(
        field_class_counts[SessionFieldClass::Digest as usize],
        10,
        "browser decoder stack session must keep exactly ten required decode shader digests; optional checkpoint/top-1 digests are pinned by their feature contracts"
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
    // M6e7 amendment: the accepted seven operations plus exactly the sealed
    // prefill operation pinned by the M6e7 prefill module contract.
    // M6e8 amendment: plus exactly the sealed logits operation pinned by the
    // M6e8 logits module contract.
    // M7q1 amendment: plus two small real-device FP16 probe operations. They
    // delegate through this same authority so the causal module exposes no
    // sibling side door.
    let expected_methods = [
        "abort",
        "begin",
        "begin_resident",
        "begin_with_shader_override",
        "finish",
        "logits",
        "new",
        "prefill",
        "probe_m7q1_precision_admission_json",
        "resident_weights_json",
        "run_m7q1_fp16_weight_probe_json",
        "shader_sources_json",
        "step",
        "top1",
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
        "browser decoder stack session must have one inherent implementation"
    );
    let session_impl = session_impls[0];
    assert!(
        session_impl.trait_.is_none()
            && session_impl.generics.params.is_empty()
            && only_doc_attributes(&session_impl.attrs),
        "browser decoder stack session implementation must be unconditional and inherent"
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
    assert_eq!(
        count_method_calls(&begin.block, "plan"),
        0,
        "sealed begin must consume the one shared M7q1 preparation result instead of duplicating planners"
    );
    assert!(
        planner_result_reaches_live_consumer(step, "plan_step", 1, "owner"),
        "sealed step must bind the exact stack plan_step() result and feed it to the live step executor"
    );
    assert!(
        planner_result_reaches_live_consumer(step, "plan_cache_transition", 1, "owner"),
        "sealed step must bind the exact cache plan_cache_transition() result and feed it to the live step executor"
    );
    let prefill = authority_impl
        .items
        .iter()
        .find_map(|item| match item {
            ImplItem::Fn(function) if function.sig.ident == "prefill" => Some(function),
            _ => None,
        })
        .expect("sealed prefill operation is missing");
    assert!(
        planner_result_reaches_live_consumer(prefill, "plan_prefill", 1, "owner"),
        "sealed prefill must bind the payload-free geometry plan_prefill() result and feed it to the live prefill executor"
    );
    let override_plan_calls = count_method_calls(&begin_with_override.block, "plan");
    assert!(
        override_plan_calls == 0
            || (override_plan_calls == 4
                && planner_result_reaches_live_consumer(begin_with_override, "plan", 4, "owner")),
        "sealed begin_with_shader_override must not compute-then-ignore any descriptor.plan()"
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
            "prefill" => {
                assert_eq!(
                    plan_calls, 0,
                    "prefill must use plan_prefill instead of rebuilding operand-validating model weights"
                );
                assert_eq!(plan_step_calls, 0, "prefill replans a step outside step");
                assert_eq!(
                    transition_calls, 0,
                    "prefill replans a cache transition outside step"
                );
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
        if function.sig.ident == "prepare_stack_begin" {
            assert_eq!(
                count_method_calls(&function.block, "plan"),
                7,
                "shared begin preparation must bind both closed precision branches and the one physical resource planner"
            );
            assert_eq!(
                count_method_calls(&function.block, "plan_prefill"),
                1,
                "shared begin preparation must bind the balanced payload-free prefill planner"
            );
            assert_eq!(
                count_method_calls(&function.block, "plan_step"),
                0,
                "shared begin preparation replans a decode step"
            );
            assert_eq!(
                count_method_calls(&function.block, "plan_cache_transition"),
                0,
                "shared begin preparation replans a cache transition"
            );
            continue;
        }
        if function.sig.ident == "prepare_resident_stack_begin" {
            assert_eq!(
                count_method_calls(&function.block, "plan"),
                4,
                "resident begin preparation must bind stack, logits, physical-weight, and cache plans"
            );
            assert_eq!(
                count_method_calls(&function.block, "plan_prefill"),
                1,
                "resident begin preparation must bind the prefill planner"
            );
            assert_eq!(
                count_method_calls(&function.block, "plan_step"),
                0,
                "resident begin preparation replans a decode step"
            );
            assert_eq!(
                count_method_calls(&function.block, "plan_cache_transition"),
                0,
                "resident begin preparation replans a cache transition"
            );
            continue;
        }
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
        if matches!(
            item,
            Item::Struct(value)
                if value.ident == "BrowserDecoderStackResidentWeights"
                    || value.ident == "SharedDecoderStackResidentWeightCache"
        ) || matches!(
            item,
            Item::Fn(value) if value.sig.ident == "shared_resident_weight_cache"
        ) {
            let visibility =
                item_visibility(item).expect("resident cache bridge must have visibility");
            assert!(
                restricted_to_parent(visibility),
                "resident cache bridge must be restricted to the parent WebRuntime module"
            );
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
fn web_runtime_has_one_closed_stack_owner_and_direct_exact_wasm_delegations() {
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
            "decoder_full_layer_session",
            "decoder_kv_session",
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
        "crates/pvlc-runtime-web/src/web/decoder_full_layer_session.rs",
    ] {
        let sibling_root = parse(sibling);
        assert!(
            matching_impls(&sibling_root, AUTHORITY).is_empty()
                && matching_impls(&sibling_root, SESSION).is_empty()
                && !named_type_declarations(&sibling_root)
                    .iter()
                    .any(|name| name == AUTHORITY || name == SESSION),
            "{sibling} must not become a decoder stack side door"
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
            "abort_decoder_full_layer_session",
            "abort_decoder_kv_session",
            "abort_decoder_layer_session",
            "abort_decoder_stack_session",
            "abort_vision_encoder_stack_sharded",
            "begin_decoder_full_layer_session",
            "begin_decoder_full_layer_session_with_shader_override",
            "begin_decoder_kv_session",
            "begin_decoder_kv_session_with_shader_override",
            "begin_decoder_layer_session",
            "begin_decoder_layer_session_with_shader_override",
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
            "decoder_full_layer_session_shader_sources_json",
            "decoder_kv_session_shader_sources_json",
            "decoder_layer_session_shader_sources_json",
            "decoder_stack_resident_weights_json",
            "decoder_stack_session_shader_sources_json",
            "enqueue_vision_encoder_stack_sharded_layer_json",
            "enqueue_vision_encoder_stack_sharded_resident_layer_json",
            "finish_decoder_full_layer_session",
            "finish_decoder_kv_session",
            "finish_decoder_layer_session",
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
            "step_decoder_full_layer_session",
            "step_decoder_kv_session",
            "step_decoder_layer_session",
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
        "web.rs must not extend the sealed decoder stack authority"
    );
    for alias in root.items.iter().filter_map(|item| match item {
        Item::Type(alias) => Some(alias),
        _ => None,
    }) {
        assert!(
            !type_names_in_type(&alias.ty).contains(AUTHORITY),
            "web.rs must not hide the decoder stack authority behind a type alias"
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
        "WebRuntime must own exactly one DecoderStackSessionAuthority"
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
        "WebRuntime decoder stack authority must be one unconditional private named field"
    );

    let expected_delegations = [
        ("abort_decoder_stack_session", "abort"),
        ("begin_decoder_stack_session", "begin"),
        ("begin_decoder_stack_session_resident", "begin_resident"),
        (
            "begin_decoder_stack_session_with_shader_override",
            "begin_with_shader_override",
        ),
        (
            "decoder_stack_resident_weights_json",
            "resident_weights_json",
        ),
        (
            "decoder_stack_session_shader_sources_json",
            "shader_sources_json",
        ),
        ("finish_decoder_stack_session", "finish"),
        ("logits_decoder_stack_session", "logits"),
        ("prefill_decoder_stack_session", "prefill"),
        (
            "probe_m7q1_precision_admission_json",
            "probe_m7q1_precision_admission_json",
        ),
        (
            "run_m7q1_fp16_weight_probe_json",
            "run_m7q1_fp16_weight_probe_json",
        ),
        ("step_decoder_stack_session", "step"),
        ("top1_decoder_stack_session", "top1"),
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

    let exported_stack_methods = wasm_impl
        .items
        .iter()
        .filter_map(|item| match item {
            ImplItem::Fn(function)
                if function
                    .sig
                    .ident
                    .to_string()
                    .contains("decoder_stack_session") =>
            {
                Some(function.sig.ident.to_string())
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        exported_stack_methods,
        expected_delegations
            .into_iter()
            .filter(|(name, _)| name.contains("decoder_stack_session"))
            .map(|(name, _)| name.to_owned())
            .collect(),
        "decoder stack WASM operation allowlist drifted"
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
    let direct = syn::parse_str::<Type>("crate::AsyncSessionOwner<BrowserDecoderStackSession>")
        .expect("direct owner type must parse");
    let unqualified = syn::parse_str::<Type>("AsyncSessionOwner<BrowserDecoderStackSession>")
        .expect("unqualified owner type must parse");
    let decoy = syn::parse_str::<Type>("decoy::AsyncSessionOwner<BrowserDecoderStackSession>")
        .expect("decoy owner type must parse");
    let wrapped = syn::parse_str::<Type>("RefCell<AsyncSessionOwner<BrowserDecoderStackSession>>")
        .expect("wrapped owner type must parse");
    let tuple =
        syn::parse_str::<Type>("(AsyncSessionOwner<BrowserDecoderStackSession>, ExtraAuthority)")
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
            pub fn step_decoder_stack_session(&self, hidden: Bytes) -> Result {
                self.decoder_stack_session.step(hidden)
            }
            pub fn swapped(&self, hidden: Bytes, extra: Bytes) -> Result {
                self.decoder_stack_session.step(extra)
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
        impl DecoderStackSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let kv_plan = kv_descriptor.plan()?;
                let stack_plan = stack_descriptor.plan()?;
                let session = BrowserDecoderStackSession::create(&self.device, &self.queue, kv_plan, stack_plan)?;
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
        impl DecoderStackSessionAuthority {
            fn step(&self, hidden: Bytes) -> Result<String, Error> {
                let (lease, mut session) = self.owner.acquire()?;
                let transition = session.kv_plan.plan_cache_transition(session.cache_tokens)?;
                let step_plan = session.stack_plan.plan_step(transition.cache_tokens_before, &hidden)?;
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
        impl DecoderStackSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let kv_plan = kv_descriptor.plan()?;
                let stack_plan = stack_descriptor.plan()?;
                let session = BrowserDecoderStackSession::create(&self.device, &self.queue, kv_plan)?;
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
        impl DecoderStackSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                if false {
                    descriptor.plan();
                    stack_descriptor.plan();
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
        impl DecoderStackSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let kv_plan = kv_descriptor.plan()?;
                let second = kv_descriptor.plan()?;
                let stack_plan = stack_descriptor.plan()?;
                let session = BrowserDecoderStackSession::create(&self.device, &self.queue, second, stack_plan)?;
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
        impl DecoderStackSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let kv_plan = kv_descriptor.plan()?;
                let stack_plan = stack_descriptor.plan()?;
                let shadow = stack_plan;
                let session = BrowserDecoderStackSession::create(&self.device, &self.queue, kv_plan, shadow)?;
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
        impl DecoderStackSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let kv_plan = kv_descriptor.plan()?;
                let stack_plan = stack_descriptor.plan()?;
                let mut decoy = Decoy::new();
                decoy.plan = stack_plan;
                let session = BrowserDecoderStackSession::create(&self.device, &self.queue, kv_plan)?;
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
        impl DecoderStackSessionAuthority {
            fn step(&self, hidden: Bytes) -> Result<String, Error> {
                let (lease, mut session) = self.owner.acquire()?;
                let transition = session.kv_plan.plan_cache_transition(session.cache_tokens)?;
                let step_plan = session.stack_plan.plan_step(session.cache_tokens, &hidden)?;
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
        impl DecoderStackSessionAuthority {
            fn step(&self, hidden: Bytes) -> Result<String, Error> {
                let (lease, mut session) = self.owner.acquire()?;
                let transition = session.kv_plan.plan_cache_transition(session.cache_tokens)?;
                let step_plan = session.stack_plan.plan_step(session.cache_tokens, &hidden)?;
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
        impl DecoderStackSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<Promise, Error> {
                let prepared = (|| {
                    let kv_plan = kv_descriptor.plan()?;
                    let stack_plan = stack_descriptor.plan()?;
                    Ok((kv_plan, stack_plan, sources()))
                })();
                let (kv_plan, stack_plan, sources) = match prepared {
                    Ok(prepared) => prepared,
                    Err(error) => return reject(error),
                };
                future_to_promise(async move {
                    run_begin(kv_plan, stack_plan, sources).await
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
        impl DecoderStackSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<Promise, Error> {
                let prepared = (|| {
                    let kv_plan = kv_descriptor.plan()?;
                    let stack_plan = stack_descriptor.plan()?;
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
            let _ = pvlc_runtime_core::DecoderStackPlan {};
        }
        "#,
    )
    .expect("forged-plan fixture must parse");
    assert_eq!(plan_struct_constructions(&forged_plans), 5);
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
        (
            "pvlc_runtime_core::DecoderStackPlan",
            SessionFieldClass::CoreStackPlan,
        ),
        ("bool", SessionFieldClass::Scalar),
        ("u32", SessionFieldClass::Scalar),
        ("u64", SessionFieldClass::Scalar),
        ("usize", SessionFieldClass::Scalar),
        ("[u8; 32]", SessionFieldClass::Digest),
        // M6e7 amendment: the prefill extension stores its prefill-only
        // resources as optional session fields so decode-only sessions keep
        // the accepted persistent topology.
        (
            "Option<wgpu::webgpu::GpuBuffer>",
            SessionFieldClass::WgpuWebGpuBuffer,
        ),
        ("Option<js_sys::Object>", SessionFieldClass::JsObjectHandle),
        ("Option<[u8; 32]>", SessionFieldClass::Digest),
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
        "crate::AsyncSessionOwner<BrowserDecoderStackSession>",
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
        "[wgpu::webgpu::GpuBuffer; 18]",
        "[js_sys::Object; 15]",
        "Option<pvlc_runtime_core::DecoderStackPlan>",
        "pvlc_runtime_core::DecoderKvSessionStepPlan",
        "pvlc_runtime_core::DecoderLayerStepPlan",
        "pvlc_runtime_core::DecoderStackStep<'_>",
        "crate::DecoderKvSessionPlan",
        "super::DecoderKvSessionPlan",
        "crate::DecoderStackPlan",
        "super::DecoderStackPlan",
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
        struct BrowserDecoderStackSession {
            kv_plan: pvlc_runtime_core::DecoderKvSessionPlan,
            stack_plan: pvlc_runtime_core::DecoderStackPlan,
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
            hidden_pingpong_buffer: wgpu::webgpu::GpuBuffer,
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
            key_cache_buffer: wgpu::webgpu::GpuBuffer,
            value_cache_buffer: wgpu::webgpu::GpuBuffer,
            norm1_buffer: wgpu::webgpu::GpuBuffer,
            q_projection_buffer: wgpu::webgpu::GpuBuffer,
            k_projection_buffer: wgpu::webgpu::GpuBuffer,
            v_projection_buffer: wgpu::webgpu::GpuBuffer,
            mrope_query_buffer: wgpu::webgpu::GpuBuffer,
            mrope_key_buffer: wgpu::webgpu::GpuBuffer,
            attention_output_buffer: wgpu::webgpu::GpuBuffer,
            o_projection_buffer: wgpu::webgpu::GpuBuffer,
            attention_residual_buffer: wgpu::webgpu::GpuBuffer,
            norm2_buffer: wgpu::webgpu::GpuBuffer,
            gate_buffer: wgpu::webgpu::GpuBuffer,
            up_buffer: wgpu::webgpu::GpuBuffer,
            activation_buffer: wgpu::webgpu::GpuBuffer,
            down_projection_buffer: wgpu::webgpu::GpuBuffer,
            stack_readback_buffer: wgpu::webgpu::GpuBuffer,
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
            swiglu_pipeline: js_sys::Object,
            residual_pipeline: js_sys::Object,
            rms_bind_group: js_sys::Object,
            gemv_q_bind_group: js_sys::Object,
            gemv_k_bind_group: js_sys::Object,
            gemv_v_bind_group: js_sys::Object,
            mrope_bind_group: js_sys::Object,
            append_bind_group: js_sys::Object,
            gemv_o_bind_group: js_sys::Object,
            residual_bind_group: js_sys::Object,
            rms2_bind_group: js_sys::Object,
            gemv_gate_bind_group: js_sys::Object,
            gemv_up_bind_group: js_sys::Object,
            swiglu_bind_group: js_sys::Object,
            gemv_down_bind_group: js_sys::Object,
            residual2_bind_group: js_sys::Object,
            split_partials_buffer: wgpu::webgpu::GpuBuffer,
            split_partial_uniform_buffer: wgpu::webgpu::GpuBuffer,
            split_merge_uniform_buffer: wgpu::webgpu::GpuBuffer,
            split_partial_pipeline: js_sys::Object,
            split_merge_pipeline: js_sys::Object,
            split_partial_bind_group: js_sys::Object,
            split_merge_bind_group: js_sys::Object,
            split_partial_shader_blake3: [u8; 32],
            split_merge_shader_blake3: [u8; 32],
        }
        "#,
    )
    .expect("canonical session fixture must parse");
    let mut counts = [0usize; 8];
    for field in &session.fields {
        let class = classify_session_field(&field.ty).expect("canonical field must classify");
        counts[class as usize] += 1;
    }
    // M7o2 amendment: the canonical fixture carries the mandatory split-K
    // resources, so the accepted counts grow by three buffers, two net
    // handles (the split four minus the retired serial GQA pipeline and bind
    // group), and two digests.
    assert_eq!(counts[SessionFieldClass::WgpuWebGpuBuffer as usize], 47);
    assert_eq!(counts[SessionFieldClass::JsObjectHandle as usize], 24);
    assert_eq!(counts[SessionFieldClass::CoreKvPlan as usize], 1);
    assert_eq!(counts[SessionFieldClass::CoreStackPlan as usize], 1);
    assert_eq!(counts[SessionFieldClass::Scalar as usize], 3);
    assert_eq!(counts[SessionFieldClass::Digest as usize], 9);

    let shadowed = syn::parse_str::<syn::ItemStruct>(
        r#"
        struct BrowserDecoderStackSession {
            cache_tokens: u32,
            kv_plan: pvlc_runtime_core::DecoderKvSessionPlan,
            stack_plan: pvlc_runtime_core::DecoderStackPlan,
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
fn decoder_stack_authority_has_no_crate_wide_side_doors() {
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
                .join("decoder_stack_session.rs")
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
                "{} hides the sealed decoder stack session behind a type alias",
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
