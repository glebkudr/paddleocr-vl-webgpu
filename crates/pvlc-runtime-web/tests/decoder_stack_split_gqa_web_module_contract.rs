//! Structural contract for the M7o2 split-K decode GQA path in the persistent
//! browser decoder stack session before any production exists
//! (docs/m7o2_split_k_decode_gqa_contract.md).
//!
//! M7o2 adds no WASM operation and no session capability: the accepted
//! sealed authority surface (nine operations, eight exported stack methods)
//! is unchanged. It replaces the per-layer serial `decoder_gqa_f32` dispatch
//! of the decode step with exactly two deterministic dispatches — the split
//! partial pass and the split merge pass, over the same cache buffers — and
//! grows the persistent topology accordingly:
//!
//! Buffers 63 -> 66 (+3, all mandatory: the split path serves every decode
//!   step of every admitted session):
//!   - one scratch partials plane `[16, ceil(cache_capacity / 32), 192]` f32
//!     created once at begin (the split-K partials: 128 weighted-V elements
//!     plus the chunk max and the chunk sum per (query_head, chunk) row);
//!   - two stage-uniform buffers (split_partial, split_merge), written once
//!     per step because the uniform words `[cache_tokens, chunk_count, 0, 0]`
//!     are position-dependent.
//!
//! Pipelines 11 -> 12 (+2 net): `decoder_gqa_split_partial_f32` and
//!   `decoder_gqa_split_merge_f32`, created once at begin through the
//!   accepted pipeline discipline; the serial `decoder_gqa_f32` pipeline and
//!   bind group retire with the replaced dispatch (the kernel itself remains
//!   accepted in the catalog and shader-overridable).
//!
//! Bind groups 33 -> 34 (+2 net), created once at begin through the accepted
//!   256-aligned dynamic-offset scheme: `split_partial` (query = mrope query,
//!   key/value = the compact cache planes, out = the partials plane, uniform)
//!   and `split_merge` (same reads, out = the accepted attention output
//!   buffer, uniform). Total js_sys::Object handles (pipelines + bind
//!   groups) 44 -> 46.
//!
//! Shader digests 11 -> 13 (+2): the session compiles thirteen shaders.
//!
//! Scalars stay 3 and the plan fields are unchanged: the split uniforms are
//!   derived per step under the accepted cache-transition planner authority.
//!
//! One decode step after M7o2: 17 queue writes (the hidden row plus the
//!   thirteen per-stage uniforms, the append uniform, and the two position-
//!   dependent split uniforms, replacing the single attention uniform
//!   write), 16 dispatches per layer (288 over the accepted eighteen layers:
//!   the serial GQA replaced by split partial then split merge), one copy,
//!   one submit, one map — the accepted step discipline otherwise unchanged.

#[path = "decoder_session_contract_helpers.rs"]
mod helpers;

use std::collections::BTreeSet;

use helpers::*;
use pvlc_runtime_core::DecoderGqaSplitDescriptor;
use syn::{Expr, ImplItem, ImplItemFn, Item, Type, visit::Visit};

const WEB_RS: &str = "crates/pvlc-runtime-web/src/web.rs";
const CAUSAL_MODULE: &str = "crates/pvlc-runtime-web/src/web/decoder_stack_session.rs";
const AUTHORITY: &str = "DecoderStackSessionAuthority";
const SESSION: &str = "BrowserDecoderStackSession";
const OWNER_FIELD: &str = "decoder_stack_session";
const ENCODE_STEP: &str = "encode_step";
const SPLIT_PARTIAL_KERNEL: &str = "decoder_gqa_split_partial_f32";
const SPLIT_MERGE_KERNEL: &str = "decoder_gqa_split_merge_f32";
const SPLIT_PARTIAL_KERNEL_CONST: &str = "SPLIT_PARTIAL_KERNEL_NAME";
const SPLIT_MERGE_KERNEL_CONST: &str = "SPLIT_MERGE_KERNEL_NAME";

/// The accepted eleven decoder stack kernel name constants, the two split-K
/// names, and the M7o5 tiled GEMV appended without reindexing them.
const EXPECTED_KERNEL_NAME_IDENTS: [&str; 14] = [
    "RMS_NORM_KERNEL_NAME",
    "GEMV_KERNEL_NAME",
    "MROPE_KERNEL_NAME",
    "APPEND_KERNEL_NAME",
    "ATTENTION_KERNEL_NAME",
    "SWIGLU_KERNEL_NAME",
    "RESIDUAL_KERNEL_NAME",
    "PROJECTION_KERNEL_NAME",
    "PREFILL_MROPE_KERNEL_NAME",
    "KV_APPEND_RANGE_KERNEL_NAME",
    "PREFILL_GQA_KERNEL_NAME",
    SPLIT_PARTIAL_KERNEL_CONST,
    SPLIT_MERGE_KERNEL_CONST,
    "GEMV_TILED_KERNEL_NAME",
];

/// The mandatory M7o2 session fields: one scratch partials plane, two
/// position-dependent stage uniforms, two pipelines, and two bind groups.
const EXPECTED_SPLIT_BUFFER_FIELDS: [&str; 3] = [
    "split_partials_buffer",
    "split_partial_uniform_buffer",
    "split_merge_uniform_buffer",
];
const EXPECTED_SPLIT_HANDLE_FIELDS: [&str; 4] = [
    "split_partial_pipeline",
    "split_merge_pipeline",
    "split_partial_bind_group",
    "split_merge_bind_group",
];

/// Collects every path segment identifier (type and expression positions) so
/// module-level references to exact core kernel ids can be pinned without
/// depending on formatting.
#[derive(Default)]
struct PathIdents(BTreeSet<String>);

impl<'ast> Visit<'ast> for PathIdents {
    fn visit_path(&mut self, path: &'ast syn::Path) {
        for segment in &path.segments {
            self.0.insert(segment.ident.to_string());
        }
        syn::visit::visit_path(self, path);
    }
}

/// Collects every field member name (declarations and `self.field` access
/// positions) so the mandatory M7o2 session resources can be pinned by name.
#[derive(Default)]
struct FieldNames(BTreeSet<String>);

impl<'ast> Visit<'ast> for FieldNames {
    fn visit_expr_field(&mut self, expression: &'ast syn::ExprField) {
        if let syn::Member::Named(name) = &expression.member {
            self.0.insert(name.to_string());
        }
        syn::visit::visit_expr_field(self, expression);
    }

    fn visit_field(&mut self, field: &'ast syn::Field) {
        if let Some(name) = &field.ident {
            self.0.insert(name.to_string());
        }
        syn::visit::visit_field(self, field);
    }
}

/// Counts references to one named field so the exact step-encode discipline
/// (which bind groups, pipelines, and uniform buffers the step touches) can
/// be pinned structurally.
struct FieldRefs<'a> {
    name: &'a str,
    refs: usize,
}

impl<'ast> Visit<'ast> for FieldRefs<'_> {
    fn visit_expr_field(&mut self, expression: &'ast syn::ExprField) {
        if matches!(&expression.member, syn::Member::Named(name) if name == self.name) {
            self.refs += 1;
        }
        syn::visit::visit_expr_field(self, expression);
    }
}

fn count_field_refs(block: &syn::Block, name: &str) -> usize {
    let mut counter = FieldRefs { name, refs: 0 };
    counter.visit_block(block);
    counter.refs
}

/// Counts free-function calls by terminal path identifier inside one body so
/// the exact step-encode discipline (passes / writes / copy / submit) can be
/// pinned against the accepted raw-WebGPU helpers.
struct FreeCallCount<'a> {
    name: &'a str,
    calls: usize,
}

impl<'ast> Visit<'ast> for FreeCallCount<'_> {
    fn visit_expr_call(&mut self, expression: &'ast syn::ExprCall) {
        if let Expr::Path(path) = expression.func.as_ref()
            && path
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == self.name)
        {
            self.calls += 1;
        }
        syn::visit::visit_expr_call(self, expression);
    }
}

fn count_free_calls(block: &syn::Block, name: &str) -> usize {
    let mut counter = FreeCallCount { name, calls: 0 };
    counter.visit_block(block);
    counter.calls
}

/// Closed persistent-field algebra for the M7o2 session: the accepted M6e8
/// classes, unchanged. Host-side storage, indirection, or wrapper types
/// remain outside the algebra.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SplitFieldClass {
    Accepted(SessionFieldClass),
    CorePrefillPlan,
    CoreWeightResourcePlan,
    OptionalCoreLmHeadPlan,
}

fn classify_split_session_field(ty: &Type) -> Option<SplitFieldClass> {
    if exact_core_plan(ty, "DecoderStackPrefillPlan") {
        return Some(SplitFieldClass::CorePrefillPlan);
    }
    if exact_core_plan(ty, "DecoderWeightResourcePlan") {
        return Some(SplitFieldClass::CoreWeightResourcePlan);
    }
    if exact_optional_core_plan(ty, "DecoderLmHeadPlan") {
        return Some(SplitFieldClass::OptionalCoreLmHeadPlan);
    }
    classify_session_field(ty).map(SplitFieldClass::Accepted)
}

#[derive(Default)]
struct SplitTopology {
    plain_buffers: usize,
    optional_buffers: usize,
    plain_handles: usize,
    optional_handles: usize,
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

fn session_method<'a>(module: &'a syn::File, name: &str) -> &'a ImplItemFn {
    let session_impls = matching_impls(module, SESSION);
    assert_eq!(
        session_impls.len(),
        1,
        "browser decoder stack session must have one inherent implementation"
    );
    session_impls[0]
        .items
        .iter()
        .find_map(|item| match item {
            ImplItem::Fn(function) if function.sig.ident == name => Some(function),
            _ => None,
        })
        .unwrap_or_else(|| panic!("session method {name} is missing"))
}

fn expected_authority_methods() -> BTreeSet<String> {
    [
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
    .collect()
}

fn collect_authority_methods(authority_impl: &syn::ItemImpl) -> BTreeSet<String> {
    authority_impl
        .items
        .iter()
        .filter_map(|item| match item {
            ImplItem::Fn(function) => Some(function.sig.ident.to_string()),
            _ => None,
        })
        .collect()
}

fn authority_methods_well_formed(authority_impl: &syn::ItemImpl) -> bool {
    authority_impl.items.iter().all(|item| match item {
        ImplItem::Fn(function) => {
            restricted_to_parent(&function.vis)
                && function.sig.generics.params.is_empty()
                && only_doc_attributes(&function.attrs)
        }
        _ => true,
    })
}

fn expected_stack_wasm_methods() -> BTreeSet<String> {
    [
        "abort_decoder_stack_session",
        "begin_decoder_stack_session",
        "begin_decoder_stack_session_resident",
        "begin_decoder_stack_session_with_shader_override",
        "decoder_stack_session_shader_sources_json",
        "finish_decoder_stack_session",
        "logits_decoder_stack_session",
        "prefill_decoder_stack_session",
        "step_decoder_stack_session",
        "top1_decoder_stack_session",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn module_const<'a>(module: &'a syn::File, name: &str) -> &'a syn::ItemConst {
    module
        .items
        .iter()
        .find_map(|item| match item {
            Item::Const(value) if value.ident == name => Some(value),
            _ => None,
        })
        .unwrap_or_else(|| panic!("module constant {name} is missing"))
}

/// Extracts the string literal of a `const NAME: &str = "..."` constant.
fn const_str_literal(value: &syn::ItemConst) -> String {
    let Expr::Lit(lit) = value.expr.as_ref() else {
        panic!("constant {} must be a literal", value.ident);
    };
    let syn::Lit::Str(text) = &lit.lit else {
        panic!("constant {} must be a string literal", value.ident);
    };
    text.value()
}

/// Extracts the declared element count of a `const NAME: [T; N]` constant
/// without inspecting the elements (kernel name arrays reference other
/// constants).
fn const_array_declared_len(value: &syn::ItemConst) -> String {
    let Type::Array(array) = value.ty.as_ref() else {
        panic!("constant {} must be an array", value.ident);
    };
    let Expr::Lit(length_lit) = &array.len else {
        panic!("constant {} length must be a literal", value.ident);
    };
    let syn::Lit::Int(length) = &length_lit.lit else {
        panic!("constant {} length must be an integer", value.ident);
    };
    length.base10_digits().to_owned()
}

/// Extracts the terminal identifier of every element of a
/// `const NAME: [&str; N] = [CONST_A, CONST_B, ...]` constant so the pinned
/// kernel-name order can be checked through the name constants.
fn const_array_element_idents(value: &syn::ItemConst) -> Vec<String> {
    let Expr::Array(elements) = value.expr.as_ref() else {
        panic!("constant {} must be an array literal", value.ident);
    };
    elements
        .elems
        .iter()
        .map(|element| {
            let Expr::Path(path) = element else {
                panic!("constant {} elements must be paths", value.ident);
            };
            path.path
                .segments
                .last()
                .map(|segment| segment.ident.to_string())
                .unwrap_or_else(|| panic!("constant {} element path is empty", value.ident))
        })
        .collect()
}

#[test]
fn split_gqa_adds_no_new_wasm_operation() {
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
    assert!(
        inherited(&declarations[0].vis)
            && declarations[0].content.is_none()
            && only_doc_attributes(&declarations[0].attrs),
        "decoder_stack_session must stay one unconditional private out-of-line module"
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
        forbidden.implementation_blocks, 5,
        "causal module must contain only the sealed session/cache implementations"
    );

    let authority_impl = authority_impl(&module);
    assert!(
        authority_methods_well_formed(authority_impl),
        "every sealed authority operation must be an unconditional non-generic pub(super) method"
    );
    assert_eq!(
        collect_authority_methods(authority_impl),
        expected_authority_methods(),
        "M7o2 adds no sealed operation: the authority surface stays at the accepted nine"
    );

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
    let exported_stack_methods = wasm_impls[0]
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
        expected_stack_wasm_methods(),
        "M7o2 adds no WASM operation: the exported decoder stack allowlist stays at the accepted eight"
    );
    for method in public_inherent_methods(&root, "WebRuntime") {
        assert!(
            !method.contains("split"),
            "M7o2 must not leak a split-specific public WebRuntime method: {method}"
        );
    }
}

#[test]
fn split_gqa_catalog_grows_by_exactly_two_kernels_in_the_pinned_order() {
    let module = parse(CAUSAL_MODULE);
    assert_eq!(
        const_str_literal(module_const(&module, SPLIT_PARTIAL_KERNEL_CONST)),
        SPLIT_PARTIAL_KERNEL,
        "the split partial kernel name constant drifted"
    );
    assert_eq!(
        const_str_literal(module_const(&module, SPLIT_MERGE_KERNEL_CONST)),
        SPLIT_MERGE_KERNEL,
        "the split merge kernel name constant drifted"
    );
    let kernel_names = module_const(&module, "KERNEL_NAMES");
    assert_eq!(
        const_array_declared_len(kernel_names),
        "14",
        "M7o5 appends tiled GEMV after the accepted M7o2 thirteen-kernel catalog"
    );
    assert_eq!(
        const_array_element_idents(kernel_names),
        EXPECTED_KERNEL_NAME_IDENTS
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        "kernel catalog order drifted: the M7o2 slots stay fixed and tiled GEMV appends"
    );

    let mut paths = PathIdents::default();
    paths.visit_file(&module);
    for required in ["DecoderGqaSplitPartialF32", "DecoderGqaSplitMergeF32"] {
        assert!(
            paths.0.contains(required),
            "causal module must reference the exact M7o2 kernel id {required}"
        );
    }
    let mut field_names = FieldNames::default();
    field_names.visit_file(&module);
    for required in [
        "split_partials_buffer",
        "split_partial_uniform_buffer",
        "split_merge_uniform_buffer",
        "split_partial_pipeline",
        "split_merge_pipeline",
        "split_partial_bind_group",
        "split_merge_bind_group",
    ] {
        assert!(
            field_names.0.contains(required),
            "causal module must own the exact M7o2 resource {required}"
        );
    }
}

#[test]
fn split_gqa_session_topology_extends_the_closed_persistent_algebra() {
    let module = parse(CAUSAL_MODULE);
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
    let mut topology = SplitTopology::default();
    let mut plain_buffer_names = BTreeSet::new();
    let mut plain_handle_names = BTreeSet::new();
    for field in &session.fields {
        assert!(
            inherited(&field.vis) && only_doc_attributes(&field.attrs),
            "browser decoder stack session fields must all be private"
        );
        let name = field
            .ident
            .as_ref()
            .expect("session fields must be named")
            .to_string();
        match classify_split_session_field(&field.ty) {
            Some(SplitFieldClass::Accepted(SessionFieldClass::WgpuWebGpuBuffer)) => {
                if optional_session_field_class(&field.ty).is_some() {
                    topology.optional_buffers += 1;
                } else {
                    topology.plain_buffers += 1;
                    plain_buffer_names.insert(name);
                }
            }
            Some(SplitFieldClass::Accepted(SessionFieldClass::JsObjectHandle)) => {
                if optional_session_field_class(&field.ty).is_some() {
                    topology.optional_handles += 1;
                } else {
                    topology.plain_handles += 1;
                    plain_handle_names.insert(name);
                }
            }
            Some(SplitFieldClass::Accepted(SessionFieldClass::CoreKvPlan)) => {
                topology.kv_plans += 1;
            }
            Some(SplitFieldClass::Accepted(SessionFieldClass::CoreStackPlan)) => {
                topology.stack_plans += 1;
            }
            Some(SplitFieldClass::CorePrefillPlan) => {
                topology.prefill_plans += 1;
            }
            Some(SplitFieldClass::CoreWeightResourcePlan) => {
                topology.weight_resource_plans += 1;
            }
            Some(SplitFieldClass::OptionalCoreLmHeadPlan) => {
                topology.lm_head_plans += 1;
            }
            Some(SplitFieldClass::Accepted(SessionFieldClass::Scalar)) => {
                topology.scalars += 1;
            }
            Some(SplitFieldClass::Accepted(SessionFieldClass::Digest)) => {
                topology.digests += 1;
            }
            Some(SplitFieldClass::Accepted(
                SessionFieldClass::CoreAttentionPlan | SessionFieldClass::CoreLayerPlan,
            )) => {
                panic!("decoder stack session must not hold attention-block or layer plans")
            }
            None => panic!(
                "browser decoder stack session field {name} hides state outside the closed persistent algebra"
            ),
        }
    }
    assert_eq!(
        topology.plain_buffers, 47,
        "the accepted forty-four mandatory buffers plus the scratch partials plane and the two split stage uniforms"
    );
    assert_eq!(
        topology.optional_buffers, 21,
        "the accepted optional prefill, logits, and top-1 buffers are unchanged"
    );
    assert_eq!(
        topology.plain_buffers + topology.optional_buffers,
        68,
        "M7o2 session topology includes the later top-1 persistent buffers"
    );
    for expected in EXPECTED_SPLIT_BUFFER_FIELDS {
        assert!(
            plain_buffer_names.contains(expected),
            "mandatory split buffer field {expected} is missing"
        );
    }
    assert_eq!(
        topology.plain_handles, 24,
        "the accepted twenty-two mandatory handles minus the retired serial GQA pipeline and bind group, plus the two split pipelines and the two split bind groups"
    );
    assert_eq!(
        topology.optional_handles, 27,
        "the accepted optional handles plus FP16 and top-1 pipelines"
    );
    assert_eq!(
        topology.plain_handles + topology.optional_handles,
        51,
        "session topology includes storage-selected FP16 and top-1 pipelines"
    );
    for expected in EXPECTED_SPLIT_HANDLE_FIELDS {
        assert!(
            plain_handle_names.contains(expected),
            "mandatory split handle field {expected} is missing"
        );
    }
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
        "session may store at most one exact core DecoderStackPrefillPlan"
    );
    assert_eq!(
        topology.weight_resource_plans, 1,
        "M7q1 session must hold one exact core DecoderWeightResourcePlan"
    );
    assert_eq!(
        topology.lm_head_plans, 1,
        "session must keep exactly one Option<pvlc_runtime_core::DecoderLmHeadPlan> (the M6e8 logits capability)"
    );
    assert_eq!(
        topology.scalars, 4,
        "session must keep cache position, poison/ready state, and resident-byte count"
    );
    assert_eq!(
        topology.digests, 16,
        "session must keep the accepted decoder, top-1, and checkpoint digests"
    );
}

#[test]
fn split_step_replaces_the_serial_gqa_with_two_deterministic_dispatches() {
    let module = parse(CAUSAL_MODULE);
    let encode_step = session_method(&module, ENCODE_STEP);
    assert_eq!(
        count_free_calls(&encode_step.block, "encode_stack_pass"),
        16,
        "one decode step must encode exactly sixteen passes per layer: the serial GQA replaced by the split partial then the split merge"
    );
    assert_eq!(
        count_free_calls(&encode_step.block, "write_stack_buffer"),
        17,
        "one decode step must perform exactly seventeen queue writes: the hidden row, the fifteen accepted uniform writes minus the retired attention uniform, plus the two position-dependent split uniforms"
    );
    assert_eq!(
        count_free_calls(&encode_step.block, "encode_stack_copy"),
        1,
        "one decode step keeps its single output copy"
    );
    assert_eq!(
        count_free_calls(&encode_step.block, "submit_stack_encoder"),
        1,
        "one decode step keeps its single submission"
    );
    for required in [
        "split_partial_pipeline",
        "split_merge_pipeline",
        "split_partial_bind_group",
        "split_merge_bind_group",
        "split_partial_uniform_buffer",
        "split_merge_uniform_buffer",
    ] {
        assert!(
            count_field_refs(&encode_step.block, required) >= 1,
            "encode_step must dispatch through the exact {required} resource"
        );
    }
    assert_eq!(
        count_field_refs(&encode_step.block, "attention_bind_group"),
        0,
        "encode_step must not dispatch the serial decoder_gqa_f32 anymore: the split partial/merge pair replaces it"
    );
    assert_eq!(
        count_field_refs(&encode_step.block, "attention_uniform_buffer"),
        0,
        "encode_step must not write the retired attention uniform anymore: the two split uniforms carry the position"
    );

    // The web step pins must agree with the shared arithmetic authority: the
    // exact core split planner owns the chunk geometry, the two dispatch
    // shapes, and the uniform words the step writes.
    let plan = DecoderGqaSplitDescriptor::pinned(332)
        .plan()
        .expect("pinned split plan");
    assert_eq!(plan.chunk_size, 32);
    assert_eq!(plan.chunk_count, 11);
    assert_eq!(plan.partial_stride_f32, 192);
    assert_eq!(plan.partials_elements, 16 * 11 * 192);
    assert_eq!(plan.partials_bytes, 16 * 11 * 192 * 4);
    assert_eq!(plan.partial_invocation.dispatch, [176, 1, 1]);
    assert_eq!(plan.partial_invocation.workgroup_size, [64, 1, 1]);
    assert_eq!(plan.merge_invocation.dispatch, [32, 1, 1]);
    assert_eq!(plan.merge_invocation.workgroup_size, [64, 1, 1]);
    assert_eq!(plan.uniform_words, [[332, 11, 0, 0], [332, 11, 0, 0],]);
}

#[test]
fn split_gqa_contract_helpers_reject_decoys() {
    // Kernel catalog decoys: a missing merge kernel, a swapped split order,
    // and a renamed kernel constant must all drift from the pinned catalog.
    let catalog_fixture = |elements: &str, declared: usize| -> syn::File {
        syn::parse_file(&format!(
            "const KERNEL_NAMES: [&str; {declared}] = [{elements}];"
        ))
        .expect("catalog fixture must parse")
    };
    let accepted_elements = EXPECTED_KERNEL_NAME_IDENTS
        .iter()
        .map(|ident| ident.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let accepted_catalog = catalog_fixture(&accepted_elements, 14);
    let accepted_names = module_const(&accepted_catalog, "KERNEL_NAMES");
    assert_eq!(const_array_declared_len(accepted_names), "14");
    assert_eq!(
        const_array_element_idents(accepted_names),
        EXPECTED_KERNEL_NAME_IDENTS
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    );

    let swapped_elements = EXPECTED_KERNEL_NAME_IDENTS
        .iter()
        .enumerate()
        .map(|(index, ident)| match index {
            11 => SPLIT_MERGE_KERNEL_CONST.to_owned(),
            12 => SPLIT_PARTIAL_KERNEL_CONST.to_owned(),
            _ => ident.to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ");
    let swapped_catalog = catalog_fixture(&swapped_elements, 14);
    assert_ne!(
        const_array_element_idents(module_const(&swapped_catalog, "KERNEL_NAMES")),
        EXPECTED_KERNEL_NAME_IDENTS
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        "a swapped split kernel order must be rejected"
    );

    let missing_merge = catalog_fixture(&accepted_elements, 13);
    assert_ne!(
        const_array_declared_len(module_const(&missing_merge, "KERNEL_NAMES")),
        "14",
        "a catalog without the split merge kernel must be rejected"
    );

    let renamed_kernel: syn::File = syn::parse_file(
        r#"
        const SPLIT_PARTIAL_KERNEL_NAME: &str = "decoder_gqa_split_k_partial_f32";
        "#,
    )
    .expect("renamed-kernel fixture must parse");
    assert_ne!(
        const_str_literal(module_const(&renamed_kernel, SPLIT_PARTIAL_KERNEL_CONST)),
        SPLIT_PARTIAL_KERNEL,
        "a drifted split kernel name must be rejected"
    );

    // Authority surface decoys: any additional split-specific operation
    // breaks the accepted nine-operation surface, and a fully pub operation
    // is rejected.
    let extra_method: syn::File = syn::parse_file(
        r#"
        impl DecoderStackSessionAuthority {
            pub(super) fn abort(&self) {}
            pub(super) fn begin(&self) {}
            pub(super) fn begin_with_shader_override(&self) {}
            pub(super) fn finish(&self) {}
            pub(super) fn logits(&self) -> js_sys::Promise {}
            pub(super) fn new() {}
            pub(super) fn prefill(&self) {}
            pub(super) fn shader_sources_json(&self) {}
            pub(super) fn split_step(&self) -> js_sys::Promise {}
            pub(super) fn step(&self) {}
        }
        "#,
    )
    .expect("extra-method fixture must parse");
    let Item::Impl(extra_impl) = &extra_method.items[0] else {
        panic!("extra-method fixture is not an impl");
    };
    assert!(authority_methods_well_formed(extra_impl));
    assert_ne!(
        collect_authority_methods(extra_impl),
        expected_authority_methods(),
        "an additional split-specific sealed operation must break the authority surface"
    );

    let public_method: syn::File = syn::parse_file(
        r#"
        impl DecoderStackSessionAuthority {
            pub fn step(&self) -> js_sys::Promise {}
        }
        "#,
    )
    .expect("public-method fixture must parse");
    let Item::Impl(public_impl) = &public_method.items[0] else {
        panic!("public-method fixture is not an impl");
    };
    assert!(
        !authority_methods_well_formed(public_impl),
        "a fully pub sealed operation must be rejected"
    );

    // Field algebra decoys: a host-side partials copy or a wrongly wrapped
    // scratch plane falls outside the closed persistent algebra.
    for decoy in [
        "Vec<f32>",
        "Option<wgpu::webgpu::GpuBuffer>",
        "Box<wgpu::webgpu::GpuBuffer>",
        "wgpu::Buffer",
    ] {
        let ty = syn::parse_str::<Type>(decoy).expect("decoy field type must parse");
        let classified = classify_split_session_field(&ty);
        if decoy == "Option<wgpu::webgpu::GpuBuffer>" {
            assert_eq!(
                classified,
                Some(SplitFieldClass::Accepted(
                    SessionFieldClass::WgpuWebGpuBuffer
                )),
                "optional buffers keep their accepted class (the M7o2 scratch plane is mandatory, but the accepted optional prefill/logits buffers stay)"
            );
        } else {
            assert_eq!(
                classified, None,
                "{decoy} must fall outside the closed persistent algebra"
            );
        }
    }

    // Step-encode decoys: the serial GQA retained, a missing split pass, and
    // a missing split uniform write must all be detected.
    let serial_gqa_retained: syn::File = syn::parse_file(
        r#"
        impl BrowserDecoderStackSession {
            fn encode_step(&self) -> Result<(), JsValue> {
                write_stack_buffer(queue, &self.hidden_pingpong_buffer, hidden_bytes)?;
                write_stack_buffer(
                    queue,
                    &self.attention_uniform_buffer,
                    bytemuck::cast_slice(&transition.attention.uniform_words),
                )?;
                let encoder = create_stack_encoder(device, "encoder")?;
                for layer in 0..self.stack_plan.layers {
                    encode_stack_pass(&encoder, &self.attention_pipeline, &self.attention_bind_group, dispatch, &[cache_offset, cache_offset])?;
                }
                encode_stack_copy(&encoder, &self.hidden_pingpong_buffer, 0, &self.stack_readback_buffer, 0, hidden_stride)?;
                submit_stack_encoder(queue, &encoder)
            }
        }
        "#,
    )
    .expect("serial-gqa fixture must parse");
    let serial_method = session_method(&serial_gqa_retained, ENCODE_STEP);
    assert!(
        count_field_refs(&serial_method.block, "attention_bind_group") >= 1,
        "a step that still dispatches the serial GQA must be detected"
    );
    assert!(
        count_field_refs(&serial_method.block, "attention_uniform_buffer") >= 1,
        "a step that still writes the retired attention uniform must be detected"
    );
    assert_ne!(
        count_free_calls(&serial_method.block, "encode_stack_pass"),
        16,
        "a step without the split partial/merge pair must not satisfy the sixteen-pass pin"
    );

    let missing_split_write: syn::File = syn::parse_file(
        r#"
        impl BrowserDecoderStackSession {
            fn encode_step(&self) -> Result<(), JsValue> {
                write_stack_buffer(queue, &self.hidden_pingpong_buffer, hidden_bytes)?;
                write_stack_buffer(
                    queue,
                    &self.split_partial_uniform_buffer,
                    bytemuck::cast_slice(&transition.split_partial.uniform_words),
                )?;
                let encoder = create_stack_encoder(device, "encoder")?;
                for layer in 0..self.stack_plan.layers {
                    encode_stack_pass(&encoder, &self.split_partial_pipeline, &self.split_partial_bind_group, dispatch, &[cache_offset, cache_offset])?;
                    encode_stack_pass(&encoder, &self.split_merge_pipeline, &self.split_merge_bind_group, dispatch, &[cache_offset, cache_offset])?;
                }
                encode_stack_copy(&encoder, &self.hidden_pingpong_buffer, 0, &self.stack_readback_buffer, 0, hidden_stride)?;
                submit_stack_encoder(queue, &encoder)
            }
        }
        "#,
    )
    .expect("missing-split-write fixture must parse");
    let missing_write_method = session_method(&missing_split_write, ENCODE_STEP);
    assert_ne!(
        count_free_calls(&missing_write_method.block, "write_stack_buffer"),
        17,
        "a step missing a split uniform write must not satisfy the seventeen-write pin"
    );
    assert_eq!(
        count_field_refs(&missing_write_method.block, "split_merge_uniform_buffer"),
        0,
        "a step that never touches the split merge uniform must be detected"
    );
}
