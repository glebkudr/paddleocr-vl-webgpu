//! Structural contract for the M6e7 persistent browser decoder stack prefill
//! operation before any production exists.
//!
//! The accepted M6e6 sealed browser decoder stack session gains exactly one
//! WASM operation, `prefill_decoder_stack_session(hidden_states: Uint8Array)
//! -> Promise`, implemented inside the same cfg-free sealed causal module
//! `web/decoder_stack_session.rs`. This contract pins the prefill-specific
//! surface, the planner dataflow, the zero-prefix admission, the caller-input
//! discipline, and the exact persistent-topology deltas derived from the M6e7
//! contract (docs/m6e7_persistent_browser_decoder_stack_prefill_contract.md,
//! sections "Accepted Boundary", "Kernels", and "Persistent Topology").
//!
//! Persistent-topology derivation, applied to the accepted M6e6 session shape
//! (44 GpuBuffer / 7 pipelines / 15 bind groups / 7 shader digests / 3
//! scalars / 1 DecoderKvSessionPlan / 1 DecoderStackPlan):
//!
//! Buffers 44 -> 56 (+12):
//!   - one hidden-states storage buffer `[cache_capacity, 1024]` f32 (the
//!     prefill input; the accepted decode hidden ping-pong `[2, 1024]`
//!     stays for decode);
//!   - ten multi-token stage intermediates sized by cache_capacity, exactly
//!     the contract's enumeration (norm1 / q / k / v / context / o / norm2 /
//!     gate / up / act); the down projection folds into the residual-add
//!     discipline, so it adds no dedicated buffer;
//!   - one zero bias buffer `[3072]` f32 uploaded once, covering every
//!     projection output width of the bias-free multi-token linear;
//!   - no new uniform or readback buffers: the fifteen prefill stage-uniform
//!     word sets reuse the accepted fifteen per-stage uniform buffers one to
//!     one (prefill and decode never coexist inside one session state), and
//!     the exact 4096-byte prefill output (the final hidden row) reuses the
//!     accepted 4096-byte stack readback buffer.
//!
//! Pipelines 7 -> 11 (+4):
//!   - `decoder_prefill_gqa_f32`, `decoder_prefill_mrope_f32`,
//!     `decoder_kv_append_range_f32` — the three new catalog kernels;
//!   - `vision_patch_projection_f32` — the contract's Kernels section runs
//!     all seven prefill projections on this reused catalog kernel, but the
//!     accepted session owns no pipeline for it (its only projection
//!     pipeline is the single-token `gemv_f32`), so one more pipeline is
//!     mandatory. The contract's "three new pipelines" phrase enumerates
//!     only the new kernels; this test pins the executable topology.
//!
//! Bind groups 15 -> 30 (+15): one bind group per prefill stage (rms1, q, k,
//!   v, prefill mrope, range append, prefill gqa, o, residual, rms2, gate,
//!   up, swiglu, down, residual), each reused across all eighteen layers
//!   through the accepted 256-aligned dynamic-offset scheme; none can be
//!   shared with the accepted fifteen decode groups because prefill
//!   addresses the new multi-token buffers. Total js_sys::Object handles
//!   (pipelines + bind groups) 22 -> 41.
//!
//! Shader digests 7 -> 11 (+4): the session compiles eleven shaders (the
//!   accepted seven plus the projection kernel and the three prefill
//!   kernels), each pinned by a canonical BLAKE3 digest for the
//!   shader-override mechanism.
//!
//! Scalars stay 3 (cache_tokens / poisoned / ready): zero-prefix admission
//!   is derived from `cache_tokens == 0`, so no new scalar state is accepted.
//!
//! Core plans: exactly one DecoderKvSessionPlan, exactly one
//!   DecoderStackPlan, and at most one stored DecoderStackPrefillPlan (the
//!   plan may instead be rebuilt inside each prefill from the same
//!   descriptors; both shapes satisfy the contract).

#[path = "decoder_session_contract_helpers.rs"]
mod helpers;

use std::collections::BTreeSet;

use helpers::*;
use syn::{
    BinOp, Expr, ExprBinary, ExprField, ExprPath, FnArg, ImplItem, ImplItemFn, LitInt, ReturnType,
    Type, Visibility, visit::Visit,
};

const WEB_RS: &str = "crates/pvlc-runtime-web/src/web.rs";
const CAUSAL_MODULE: &str = "crates/pvlc-runtime-web/src/web/decoder_stack_session.rs";
const AUTHORITY: &str = "DecoderStackSessionAuthority";
const SESSION: &str = "BrowserDecoderStackSession";
const OWNER_FIELD: &str = "decoder_stack_session";
const PREFILL_AUTHORITY_METHOD: &str = "prefill";
const PREFILL_WASM_METHOD: &str = "prefill_decoder_stack_session";

/// Collects the terminal identifier of every path (type and expression
/// positions) so module-level references to exact core types and kernel ids
/// can be pinned without depending on formatting.
#[derive(Default)]
struct PathIdents(BTreeSet<String>);

impl<'ast> Visit<'ast> for PathIdents {
    fn visit_path(&mut self, path: &'ast syn::Path) {
        if let Some(segment) = path.segments.last() {
            self.0.insert(segment.ident.to_string());
        }
        syn::visit::visit_path(self, path);
    }
}

/// Detects the zero-prefix admission guard: an equality comparison that
/// mentions `cache_tokens` (as a field or a plain binding) against the
/// literal `0`.
#[derive(Default)]
struct ZeroPrefixAdmission {
    found: bool,
}

impl<'ast> Visit<'ast> for ZeroPrefixAdmission {
    fn visit_expr_binary(&mut self, expression: &'ast ExprBinary) {
        if matches!(expression.op, BinOp::Eq(_)) {
            let operands = [&expression.left, &expression.right];
            let mentions_cache_tokens = operands.iter().any(|side| {
                let mut references = CacheTokensRefs::default();
                references.visit_expr(side);
                references.0
            });
            let has_zero_literal = operands.iter().any(|side| {
                matches!(
                    side.as_ref(),
                    Expr::Lit(lit) if matches!(&lit.lit, syn::Lit::Int(value) if value.base10_digits() == "0")
                )
            });
            if mentions_cache_tokens && has_zero_literal {
                self.found = true;
            }
        }
        syn::visit::visit_expr_binary(self, expression);
    }
}

#[derive(Default)]
struct CacheTokensRefs(bool);

impl<'ast> Visit<'ast> for CacheTokensRefs {
    fn visit_expr_field(&mut self, expression: &'ast ExprField) {
        if matches!(&expression.member, syn::Member::Named(name) if name == "cache_tokens") {
            self.0 = true;
        }
        syn::visit::visit_expr_field(self, expression);
    }

    fn visit_expr_path(&mut self, expression: &'ast ExprPath) {
        if expression.path.is_ident("cache_tokens") {
            self.0 = true;
        }
        syn::visit::visit_expr_path(self, expression);
    }
}

/// Detects the exact 4096 row byte width literal: the `[tokens, 1024]` f32
/// input row (`1024 * 4` bytes) used for byte-length validation and the exact
/// 4096-byte prefill output.
#[derive(Default)]
struct RowByteWidth4096(bool);

impl<'ast> Visit<'ast> for RowByteWidth4096 {
    fn visit_lit_int(&mut self, literal: &'ast LitInt) {
        if literal.base10_digits() == "4096" {
            self.0 = true;
        }
    }
}

/// Counts hand-constructions of the exact core prefill plan type; the causal
/// module must bind the planner result, never forge it.
#[derive(Default)]
struct PrefillPlanConstructions(usize);

impl<'ast> Visit<'ast> for PrefillPlanConstructions {
    fn visit_expr_struct(&mut self, expression: &'ast syn::ExprStruct) {
        if expression
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "DecoderStackPrefillPlan")
        {
            self.0 += 1;
        }
        syn::visit::visit_expr_struct(self, expression);
    }
}

fn is_exact_js_path(ty: &Type, terminal: &str) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == 2
        && path.path.segments[0].ident == "js_sys"
        && matches!(path.path.segments[0].arguments, syn::PathArguments::None)
        && path.path.segments[1].ident == terminal
        && matches!(path.path.segments[1].arguments, syn::PathArguments::None)
}

fn is_uint8_array_reference(ty: &Type) -> bool {
    let Type::Reference(reference) = ty else {
        return false;
    };
    reference.mutability.is_none() && is_exact_js_path(&reference.elem, "Uint8Array")
}

fn returns_js_promise(output: &ReturnType) -> bool {
    let ReturnType::Type(_, ty) = output else {
        return false;
    };
    is_exact_js_path(ty, "Promise")
}

/// Closed persistent-field algebra for the prefill-capable session: the
/// accepted M6e6 classes plus the exact core prefill plan. Host-side storage,
/// indirection, or wrapper types remain outside the algebra.
/// M6e8 amendment: the single optional exact core LM-head plan (the logits
/// capability, present only on logits-capable sessions) joins the algebra and
/// is pinned by the M6e8 logits module contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillFieldClass {
    Accepted(SessionFieldClass),
    CorePrefillPlan,
    CoreWeightResourcePlan,
    OptionalCoreLmHeadPlan,
}

fn classify_prefill_session_field(ty: &Type) -> Option<PrefillFieldClass> {
    if exact_core_plan(ty, "DecoderStackPrefillPlan") {
        return Some(PrefillFieldClass::CorePrefillPlan);
    }
    if exact_core_plan(ty, "DecoderWeightResourcePlan") {
        return Some(PrefillFieldClass::CoreWeightResourcePlan);
    }
    if exact_optional_core_plan(ty, "DecoderLmHeadPlan") {
        return Some(PrefillFieldClass::OptionalCoreLmHeadPlan);
    }
    classify_session_field(ty).map(PrefillFieldClass::Accepted)
}

#[derive(Default)]
struct PrefillTopology {
    buffers: usize,
    handles: usize,
    kv_plans: usize,
    stack_plans: usize,
    prefill_plans: usize,
    weight_resource_plans: usize,
    lm_head_plans: usize,
    scalars: usize,
    digests: usize,
}

fn authority_impl(module: &syn::File) -> &syn::ItemImpl {
    let authority_impls = matching_impls(module, AUTHORITY);
    assert_eq!(
        authority_impls.len(),
        1,
        "sealed authority must have one inherent implementation"
    );
    authority_impls[0]
}

fn authority_method<'a>(authority_impl: &'a syn::ItemImpl, name: &str) -> &'a ImplItemFn {
    authority_impl
        .items
        .iter()
        .find_map(|item| match item {
            ImplItem::Fn(function) if function.sig.ident == name => Some(function),
            _ => None,
        })
        .unwrap_or_else(|| panic!("sealed {name} operation is missing"))
}

#[test]
fn prefill_extends_the_sealed_authority_with_exactly_one_operation() {
    let module = parse(CAUSAL_MODULE);
    let authority_impl = authority_impl(&module);
    let expected_methods = [
        "abort",
        "begin",
        "begin_resident",
        "begin_with_shader_override",
        "finish",
        "logits",
        "new",
        PREFILL_AUTHORITY_METHOD,
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
        "sealed authority operation surface drifted: M6e7 adds exactly prefill, M6e8 adds exactly logits"
    );

    let prefill = authority_method(authority_impl, PREFILL_AUTHORITY_METHOD);
    let typed_arguments = prefill
        .sig
        .inputs
        .iter()
        .filter_map(|argument| match argument {
            FnArg::Typed(typed) => Some(typed),
            FnArg::Receiver(_) => None,
        })
        .collect::<Vec<_>>();
    assert!(
        matches!(prefill.sig.inputs.first(), Some(FnArg::Receiver(_)))
            && typed_arguments.len() == 1
            && is_uint8_array_reference(&typed_arguments[0].ty),
        "sealed prefill must take exactly (&self, hidden_states: &js_sys::Uint8Array)"
    );
    assert!(
        returns_js_promise(&prefill.sig.output),
        "sealed prefill must return js_sys::Promise"
    );

    assert!(
        planner_result_reaches_live_consumer(prefill, "plan_prefill", 1, "owner"),
        "sealed prefill must bind the payload-free DecoderStackGeometryDescriptor::plan_prefill() result and feed it to the live prefill executor"
    );
    assert_eq!(
        count_method_calls(&prefill.block, "plan_step"),
        0,
        "prefill replans a decode step: the decode step planner stays decode-only"
    );
    assert_eq!(
        count_method_calls(&prefill.block, "plan_cache_transition"),
        0,
        "prefill replans a +1 cache transition: it sets cache_tokens = tokens under its own planner authority"
    );

    let mut admission = ZeroPrefixAdmission::default();
    admission.visit_block(&prefill.block);
    assert!(
        admission.found,
        "sealed prefill must reject a non-zero-prefix session through an explicit cache_tokens == 0 admission check"
    );

    let mut paths = PathIdents::default();
    paths.visit_block(&prefill.block);
    assert!(
        paths.0.contains("stack_uint8_to_bytes"),
        "sealed prefill must copy the caller-owned input synchronously through the accepted stack_uint8_to_bytes helper"
    );
    assert!(
        paths.0.contains("stack_bytes_to_f32"),
        "sealed prefill must validate input finiteness through the accepted stack_bytes_to_f32 helper"
    );

    let mut row_width = RowByteWidth4096::default();
    row_width.visit_block(&prefill.block);
    assert!(
        row_width.0,
        "sealed prefill must pin the exact 4096-byte [tokens, 1024] f32 row width used for byte-length validation and the 4096-byte output"
    );
}

#[test]
fn web_runtime_exports_exactly_one_prefill_delegation() {
    let root = parse(WEB_RS);
    let wasm_impls = root
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Impl(item)
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

    let prefill_exports = wasm_impl
        .items
        .iter()
        .filter_map(|item| match item {
            ImplItem::Fn(function) if function.sig.ident == PREFILL_WASM_METHOD => Some(function),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        prefill_exports.len(),
        1,
        "{PREFILL_WASM_METHOD} must be exported exactly once"
    );
    let prefill = prefill_exports[0];
    assert!(
        matches!(prefill.vis, Visibility::Public(_))
            && cfg_free(&prefill.attrs)
            && prefill.sig.generics.params.is_empty(),
        "{PREFILL_WASM_METHOD} must be an unconditional non-generic public WASM method"
    );
    let typed_arguments = prefill
        .sig
        .inputs
        .iter()
        .filter_map(|argument| match argument {
            FnArg::Typed(typed) => Some(typed),
            FnArg::Receiver(_) => None,
        })
        .collect::<Vec<_>>();
    assert!(
        typed_arguments.len() == 1 && is_uint8_array_reference(&typed_arguments[0].ty),
        "{PREFILL_WASM_METHOD} must take exactly one hidden_states: &js_sys::Uint8Array argument"
    );
    assert!(
        returns_js_promise(&prefill.sig.output),
        "{PREFILL_WASM_METHOD} must return js_sys::Promise"
    );
    assert_direct_authority_call(prefill, PREFILL_AUTHORITY_METHOD, OWNER_FIELD);

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
        [
            "abort_decoder_stack_session",
            "begin_decoder_stack_session",
            "begin_decoder_stack_session_resident",
            "begin_decoder_stack_session_with_shader_override",
            "decoder_stack_session_shader_sources_json",
            "finish_decoder_stack_session",
            "logits_decoder_stack_session",
            PREFILL_WASM_METHOD,
            "step_decoder_stack_session",
            "top1_decoder_stack_session",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        "decoder stack WASM operation allowlist drifted: M6e7 grows it by exactly one, M6e8 by exactly one more"
    );
    assert!(
        public_inherent_methods(&root, "WebRuntime").contains(PREFILL_WASM_METHOD),
        "the closed public WebRuntime method allowlist must gain exactly {PREFILL_WASM_METHOD}"
    );

    let runtime = root
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Struct(item) if item.ident == "WebRuntime" => Some(item),
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
        "WebRuntime must keep owning exactly one DecoderStackSessionAuthority"
    );
}

#[test]
fn prefill_session_topology_extends_the_closed_persistent_algebra() {
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
        "causal module must contain only the sealed session/cache implementations"
    );

    let authorities = module
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Struct(item) if item.ident == AUTHORITY => Some(item),
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
            && only_doc_attributes(&authority.attrs)
            && authority.fields.len() == 4,
        "sealed authority must own its device, queue, async session owner, and resident-weight cache"
    );

    let sessions = module
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Struct(item) if item.ident == SESSION => Some(item),
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
    let mut topology = PrefillTopology::default();
    for field in &session.fields {
        assert!(
            inherited(&field.vis) && only_doc_attributes(&field.attrs),
            "browser decoder stack session fields must all be private"
        );
        match classify_prefill_session_field(&field.ty) {
            Some(PrefillFieldClass::Accepted(SessionFieldClass::WgpuWebGpuBuffer)) => {
                topology.buffers += 1;
            }
            Some(PrefillFieldClass::Accepted(SessionFieldClass::JsObjectHandle)) => {
                topology.handles += 1;
            }
            Some(PrefillFieldClass::Accepted(SessionFieldClass::CoreKvPlan)) => {
                topology.kv_plans += 1;
            }
            Some(PrefillFieldClass::Accepted(SessionFieldClass::CoreStackPlan)) => {
                topology.stack_plans += 1;
            }
            Some(PrefillFieldClass::CorePrefillPlan) => {
                topology.prefill_plans += 1;
            }
            Some(PrefillFieldClass::CoreWeightResourcePlan) => {
                topology.weight_resource_plans += 1;
            }
            Some(PrefillFieldClass::OptionalCoreLmHeadPlan) => {
                topology.lm_head_plans += 1;
            }
            Some(PrefillFieldClass::Accepted(SessionFieldClass::Scalar)) => {
                topology.scalars += 1;
            }
            Some(PrefillFieldClass::Accepted(SessionFieldClass::Digest)) => {
                topology.digests += 1;
            }
            Some(PrefillFieldClass::Accepted(
                SessionFieldClass::CoreAttentionPlan | SessionFieldClass::CoreLayerPlan,
            )) => {
                panic!("decoder stack session must not hold attention-block or layer plans")
            }
            None => panic!(
                "browser decoder stack session field {:?} hides state outside the closed persistent algebra",
                field.ident.as_ref().map(|ident| ident.to_string())
            ),
        }
    }
    assert_eq!(
        topology.buffers, 68,
        "prefill-capable session must own the accepted prefill, logits, top-1, and split-K GPU resources"
    );
    assert_eq!(
        topology.handles, 51,
        "prefill-capable session must own the accepted handles plus FP16 and top-1 pipelines"
    );
    assert_eq!(
        topology.kv_plans, 1,
        "session must hold exactly one exact core DecoderKvSessionPlan"
    );
    assert_eq!(
        topology.stack_plans, 1,
        "session must hold exactly one exact core DecoderStackPlan"
    );
    assert!(
        topology.prefill_plans <= 1,
        "session may store at most one exact core DecoderStackPrefillPlan (or rebuild it per prefill from the same descriptors)"
    );
    assert_eq!(
        topology.weight_resource_plans, 1,
        "M7q1 session must hold one exact core DecoderWeightResourcePlan"
    );
    assert_eq!(
        topology.lm_head_plans, 1,
        "M6e8 amendment: session must hold exactly one Option<pvlc_runtime_core::DecoderLmHeadPlan> (the logits capability, absent on legacy sessions)"
    );
    assert_eq!(
        topology.scalars, 4,
        "session must keep cache position, poison/ready state, and authenticated resident-byte count in plain scalars"
    );
    assert_eq!(
        topology.digests, 16,
        "session must keep the accepted prefill, split, tiled, top-1, and checkpoint digests"
    );

    let mut paths = PathIdents::default();
    paths.visit_file(&module);
    for required in [
        "DecoderStackPrefillDescriptor",
        "DecoderStackPrefillPlan",
        "VisionPatchProjectionF32",
        "DecoderPrefillGqaF32",
        "DecoderPrefillMropeF32",
        "DecoderKvAppendRangeF32",
    ] {
        assert!(
            paths.0.contains(required),
            "causal module must reference the exact core name {required}"
        );
    }

    assert_eq!(
        plan_struct_constructions(&module),
        0,
        "causal module must not hand-construct the accepted exact core decoder plan types"
    );
    let mut forged_prefill = PrefillPlanConstructions::default();
    forged_prefill.visit_file(&module);
    assert_eq!(
        forged_prefill.0, 0,
        "causal module must not hand-construct the exact core DecoderStackPrefillPlan"
    );
}

#[test]
fn prefill_contract_helpers_reject_decoys() {
    let accepted_argument: Type =
        syn::parse_str("&js_sys::Uint8Array").expect("accepted argument type must parse");
    assert!(is_uint8_array_reference(&accepted_argument));
    for decoy in [
        "js_sys::Uint8Array",
        "&mut js_sys::Uint8Array",
        "&Uint8Array",
        "&js_sys::Array",
        "&[u8]",
        "js_sys::Promise",
    ] {
        let ty = syn::parse_str::<Type>(decoy).expect("decoy argument type must parse");
        assert!(
            !is_uint8_array_reference(&ty),
            "{decoy} must not satisfy the hidden-states argument pin"
        );
    }
    let accepted_return: Type =
        syn::parse_str("js_sys::Promise").expect("accepted return type must parse");
    let Type::Path(_) = &accepted_return else {
        panic!("accepted return type must be a path");
    };
    assert!(is_exact_js_path(&accepted_return, "Promise"));

    let prefill_plan: Type = syn::parse_str("pvlc_runtime_core::DecoderStackPrefillPlan")
        .expect("prefill plan type must parse");
    assert_eq!(
        classify_prefill_session_field(&prefill_plan),
        Some(PrefillFieldClass::CorePrefillPlan)
    );
    for decoy in [
        "Vec<f32>",
        "Option<pvlc_runtime_core::DecoderStackPrefillPlan>",
        "crate::DecoderStackPrefillPlan",
        "DecoderStackPrefillPlan",
    ] {
        let ty = syn::parse_str::<Type>(decoy).expect("decoy field type must parse");
        assert_eq!(
            classify_prefill_session_field(&ty),
            None,
            "{decoy} must fall outside the closed persistent algebra"
        );
    }

    let guarded = parse_impl_method(
        r#"
        impl DecoderStackSessionAuthority {
            fn prefill(&self, hidden_states: &js_sys::Uint8Array) -> js_sys::Promise {
                let prepared = (|| {
                    let (lease, session) = acquire_stack_session(&self.owner)?;
                    if session.cache_tokens == 0 {
                        admit(session)
                    } else {
                        reject()
                    }
                })();
                prepared
            }
        }
        "#,
        "prefill",
    );
    let mut admission = ZeroPrefixAdmission::default();
    admission.visit_block(&guarded.block);
    assert!(admission.found);

    let unguarded = parse_impl_method(
        r#"
        impl DecoderStackSessionAuthority {
            fn prefill(&self, hidden_states: &js_sys::Uint8Array) -> js_sys::Promise {
                let prepared = (|| {
                    let (lease, session) = acquire_stack_session(&self.owner)?;
                    if session.cache_tokens > 0 {
                        reject()
                    }
                    admit(session)
                })();
                prepared
            }
        }
        "#,
        "prefill",
    );
    let mut admission = ZeroPrefixAdmission::default();
    admission.visit_block(&unguarded.block);
    assert!(
        !admission.found,
        "a non-equality or non-zero cache_tokens comparison must not satisfy the zero-prefix admission pin"
    );

    let forged: syn::File = syn::parse_file(
        r#"
        fn forge() {
            let _ = pvlc_runtime_core::DecoderStackPrefillPlan {};
        }
        "#,
    )
    .expect("forged prefill plan fixture must parse");
    let mut forged_prefill = PrefillPlanConstructions::default();
    forged_prefill.visit_file(&forged);
    assert_eq!(forged_prefill.0, 1);
}
