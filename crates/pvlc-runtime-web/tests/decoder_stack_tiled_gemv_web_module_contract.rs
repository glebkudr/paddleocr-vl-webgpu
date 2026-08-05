//! Structural contract for the M7o5 tiled decode GEMV path in the persistent
//! browser decoder stack session before any production exists
//! (docs/m7o5_tiled_gemv_contract.md).
//!
//! M7o5 adds no WASM operation, no session capability, and no buffer: the
//! accepted sealed authority surface (nine operations, eight exported stack
//! methods) and the accepted buffer topology are unchanged. It replaces the
//! accepted `gemv_f32` dispatches of the decode step and the M6e8 logits
//! call with `gemv_tiled_f32` dispatches over the SAME bind groups, dynamic
//! offsets, and uniform words:
//!
//! Pipelines 12 -> 12 prefill-capable / 8 -> 8 decode-only (net zero): the
//!   session gains exactly one pipeline, `gemv_tiled_f32`, created once at
//!   begin through the accepted pipeline discipline on the accepted GEMV
//!   layout, and the serial `gemv_f32` pipeline retires with the replaced
//!   dispatches — exactly the M7o2 attention-pipeline precedent (a dead
//!   pipeline field cannot stay: the closed persistent algebra forbids
//!   non-doc field attributes, so it cannot be silenced, and the kernel
//!   itself remains accepted in KERNEL_NAMES, the catalog, and the shader
//!   override surface). The diagnostics pipeline_count pins follow: 12
//!   prefill-capable, 8 decode-only.
//!
//! KERNEL_NAMES 13 -> 14: the accepted eleven decoder kernels plus the two
//!   M7o2 split-K names plus `gemv_tiled_f32` appended AT THE END. The
//!   append-at-end order is deliberate: the kernel-name array is the shader
//!   override allowlist and every existing slot keeps its position, so an
//!   override recipe written for any accepted kernel index or name keeps
//!   working; the tiled kernel is the newest surface and sorts last.
//!
//! Bind groups and buffers are unchanged: the tiled kernel shares the
//!   accepted GEMV ABI (0 matrix RO, 1 vector RO, 2 output RW, 3 uniform),
//!   so all seven per-layer projection stages (q, k, v, o, gate, up, down)
//!   and the M6e8 LM-head GEMV keep their accepted bind groups, per-layer
//!   dynamic offsets, and `[rows, columns, 0, 0]` uniform words — only the
//!   pipeline and the planner-owned dispatch shape change.
//!
//! Shader digests 13 -> 14 (+1): the session compiles fourteen shaders.
//!
//! One decode step after M7o5: 17 queue writes and 16 dispatches per layer,
//!   unchanged; the seven projection dispatches run through the tiled
//!   pipeline with the planner's `[ceil(rows / 8), 1, 1]` dispatch. One
//!   logits call keeps its zero writes / two ordered compute passes / one
//!   copy / one submit / one map, with the LM-head GEMV on the tiled
//!   pipeline. The prefill path is untouched: its projections already use
//!   the tiled `vision_patch_projection_f32` kernel.

#[path = "decoder_session_contract_helpers.rs"]
mod helpers;

use std::collections::BTreeSet;

use helpers::*;
use pvlc_runtime_core::GemvTiledDescriptor;
use syn::{Expr, ImplItem, ImplItemFn, Item, Type, visit::Visit};

const WEB_RS: &str = "crates/pvlc-runtime-web/src/web.rs";
const CAUSAL_MODULE: &str = "crates/pvlc-runtime-web/src/web/decoder_stack_session.rs";
const AUTHORITY: &str = "DecoderStackSessionAuthority";
const SESSION: &str = "BrowserDecoderStackSession";
const OWNER_FIELD: &str = "decoder_stack_session";
const ENCODE_STEP: &str = "encode_step";
const ENCODE_PREFILL: &str = "encode_prefill";
const LOGITS_EXECUTOR: &str = "run_logits";
const TILED_KERNEL: &str = "gemv_tiled_f32";
const TILED_KERNEL_CONST: &str = "GEMV_TILED_KERNEL_NAME";
const LEGACY_GEMV_PIPELINE_FIELD: &str = "gemv_pipeline";
const TILED_PIPELINE_FIELD: &str = "gemv_tiled_pipeline";
const SELECTED_TILED_PIPELINE_FIELD: &str = "gemv_tiled";

/// The accepted thirteen decoder stack kernel name constants plus the tiled
/// GEMV name appended at the end.
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
    "SPLIT_PARTIAL_KERNEL_NAME",
    "SPLIT_MERGE_KERNEL_NAME",
    TILED_KERNEL_CONST,
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
/// positions) so the exact pipeline ownership can be pinned by name.
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
/// (which pipeline each stage dispatches through) can be pinned structurally.
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

/// Closed persistent-field algebra for the M7o5 session: the accepted M7o2
/// classes, unchanged. Host-side storage, indirection, or wrapper types
/// remain outside the algebra.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TiledFieldClass {
    Accepted(SessionFieldClass),
    CorePrefillPlan,
    CoreWeightResourcePlan,
    OptionalCoreLmHeadPlan,
}

fn classify_tiled_session_field(ty: &Type) -> Option<TiledFieldClass> {
    if exact_core_plan(ty, "DecoderStackPrefillPlan") {
        return Some(TiledFieldClass::CorePrefillPlan);
    }
    if exact_core_plan(ty, "DecoderWeightResourcePlan") {
        return Some(TiledFieldClass::CoreWeightResourcePlan);
    }
    if exact_optional_core_plan(ty, "DecoderLmHeadPlan") {
        return Some(TiledFieldClass::OptionalCoreLmHeadPlan);
    }
    classify_session_field(ty).map(TiledFieldClass::Accepted)
}

#[derive(Default)]
struct TiledTopology {
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

fn free_function<'a>(module: &'a syn::File, name: &str) -> &'a syn::ItemFn {
    module
        .items
        .iter()
        .find_map(|item| match item {
            Item::Fn(function) if function.sig.ident == name => Some(function),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{name} executor is missing"))
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
fn tiled_gemv_adds_no_new_wasm_operation() {
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
        "M7o5 adds no sealed operation: the authority surface stays at the accepted nine"
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
        "M7o5 adds no WASM operation: the exported decoder stack allowlist stays at the accepted eight"
    );
    for method in public_inherent_methods(&root, "WebRuntime") {
        assert!(
            !method.contains("tiled"),
            "M7o5 must not leak a tiled-specific public WebRuntime method: {method}"
        );
    }
}

#[test]
fn tiled_gemv_catalog_grows_by_exactly_one_kernel_in_the_pinned_order() {
    let module = parse(CAUSAL_MODULE);
    assert_eq!(
        const_str_literal(module_const(&module, TILED_KERNEL_CONST)),
        TILED_KERNEL,
        "the tiled GEMV kernel name constant drifted"
    );
    let kernel_names = module_const(&module, "KERNEL_NAMES");
    assert_eq!(
        const_array_declared_len(kernel_names),
        "14",
        "M7o5 grows the decoder stack kernel catalog to fourteen kernels"
    );
    assert_eq!(
        const_array_element_idents(kernel_names),
        EXPECTED_KERNEL_NAME_IDENTS
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        "kernel catalog order drifted: the accepted thirteen stay in order and the tiled GEMV name appends at the end"
    );

    let mut paths = PathIdents::default();
    paths.visit_file(&module);
    assert!(
        paths.0.contains("GemvTiledF32"),
        "causal module must reference the exact core kernel id GemvTiledF32"
    );
    let mut field_names = FieldNames::default();
    field_names.visit_file(&module);
    assert!(
        field_names.0.contains(TILED_PIPELINE_FIELD),
        "causal module must own the exact {TILED_PIPELINE_FIELD} pipeline"
    );
    assert!(
        !field_names.0.contains(LEGACY_GEMV_PIPELINE_FIELD),
        "the retired serial gemv_f32 pipeline must not survive as a session field (the M7o2 attention-pipeline precedent: dead pipelines are removed, the kernel stays in the catalog)"
    );
}

#[test]
fn tiled_gemv_session_topology_keeps_the_closed_persistent_algebra() {
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
    let mut topology = TiledTopology::default();
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
        match classify_tiled_session_field(&field.ty) {
            Some(TiledFieldClass::Accepted(SessionFieldClass::WgpuWebGpuBuffer)) => {
                if optional_session_field_class(&field.ty).is_some() {
                    topology.optional_buffers += 1;
                } else {
                    topology.plain_buffers += 1;
                }
            }
            Some(TiledFieldClass::Accepted(SessionFieldClass::JsObjectHandle)) => {
                if optional_session_field_class(&field.ty).is_some() {
                    topology.optional_handles += 1;
                } else {
                    topology.plain_handles += 1;
                    plain_handle_names.insert(name);
                }
            }
            Some(TiledFieldClass::Accepted(SessionFieldClass::CoreKvPlan)) => {
                topology.kv_plans += 1;
            }
            Some(TiledFieldClass::Accepted(SessionFieldClass::CoreStackPlan)) => {
                topology.stack_plans += 1;
            }
            Some(TiledFieldClass::CorePrefillPlan) => {
                topology.prefill_plans += 1;
            }
            Some(TiledFieldClass::CoreWeightResourcePlan) => {
                topology.weight_resource_plans += 1;
            }
            Some(TiledFieldClass::OptionalCoreLmHeadPlan) => {
                topology.lm_head_plans += 1;
            }
            Some(TiledFieldClass::Accepted(SessionFieldClass::Scalar)) => {
                topology.scalars += 1;
            }
            Some(TiledFieldClass::Accepted(SessionFieldClass::Digest)) => {
                topology.digests += 1;
            }
            Some(TiledFieldClass::Accepted(
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
        "M7o5 adds no buffer: the accepted forty-four plus the M7o2 scratch partials plane and the two split stage uniforms"
    );
    assert_eq!(
        topology.optional_buffers, 21,
        "the accepted optional prefill, logits, and top-1 buffers are unchanged"
    );
    assert_eq!(
        topology.plain_buffers + topology.optional_buffers,
        68,
        "M7o5 session topology includes the later top-1 buffers"
    );
    assert_eq!(
        topology.plain_handles, 24,
        "eight pipelines (the tiled GEMV replacing the retired serial gemv_f32 pipeline) and sixteen bind groups: the tiled kernel shares the accepted GEMV bind groups"
    );
    assert!(
        plain_handle_names.contains(TILED_PIPELINE_FIELD),
        "the mandatory {TILED_PIPELINE_FIELD} pipeline field is missing"
    );
    assert!(
        !plain_handle_names.contains(LEGACY_GEMV_PIPELINE_FIELD),
        "the retired serial gemv_f32 pipeline must not survive as a session field"
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
        "session must keep canonical shader, top-1, and checkpoint digests"
    );
}

#[test]
fn tiled_step_and_logits_dispatch_through_the_tiled_pipeline() {
    let module = parse(CAUSAL_MODULE);
    let encode_step = session_method(&module, ENCODE_STEP);
    assert_eq!(
        count_free_calls(&encode_step.block, "encode_stack_pass"),
        16,
        "one decode step keeps exactly sixteen passes per layer: only the projection pipelines change"
    );
    assert_eq!(
        count_free_calls(&encode_step.block, "write_stack_buffer"),
        17,
        "one decode step keeps exactly seventeen queue writes: the tiled stages keep the accepted uniform words"
    );
    assert_eq!(
        count_field_refs(&encode_step.block, SELECTED_TILED_PIPELINE_FIELD),
        7,
        "encode_step must dispatch all seven per-layer projections through the storage-selected tiled pipeline"
    );
    assert_eq!(
        count_field_refs(&encode_step.block, LEGACY_GEMV_PIPELINE_FIELD),
        0,
        "encode_step must not dispatch the serial gemv_f32 pipeline anymore"
    );

    let encode_prefill = session_method(&module, ENCODE_PREFILL);
    assert_eq!(
        count_field_refs(&encode_prefill.block, SELECTED_TILED_PIPELINE_FIELD),
        0,
        "prefill uses its storage-selected multi-token projection pipeline rather than decode GEMV"
    );

    let logits = free_function(&module, LOGITS_EXECUTOR);
    assert!(
        count_field_refs(&logits.block, SELECTED_TILED_PIPELINE_FIELD) >= 1,
        "the logits call must dispatch its LM-head GEMV through the storage-selected tiled pipeline"
    );
    assert_eq!(
        count_field_refs(&logits.block, LEGACY_GEMV_PIPELINE_FIELD),
        0,
        "the logits call must not dispatch the serial gemv_f32 pipeline anymore"
    );
    assert_eq!(
        count_free_calls(&logits.block, "encode_stack_pass"),
        2,
        "one logits call keeps its two ordered compute passes"
    );
    assert_eq!(
        count_free_calls(&logits.block, "write_stack_buffer"),
        0,
        "one logits call keeps its zero writes"
    );

    // The web step pins must agree with the shared arithmetic authority: the
    // exact core tiled-GEMV planner owns the dispatch shape, the workgroup
    // size, and the uniform words the step writes.
    let q_projection = GemvTiledDescriptor {
        rows: 2048,
        columns: 1024,
    }
    .plan()
    .expect("tiled Q plan");
    assert_eq!(q_projection.tile_rows, 8);
    assert_eq!(q_projection.threads_per_row, 32);
    assert_eq!(q_projection.vector_width, 4);
    assert_eq!(q_projection.dispatch, [256, 1, 1]);
    assert_eq!(q_projection.workgroup_size, [256, 1, 1]);
    assert_eq!(q_projection.uniform_words, [2048, 1024, 0, 0]);
    let o_projection = GemvTiledDescriptor {
        rows: 1024,
        columns: 2048,
    }
    .plan()
    .expect("tiled O plan");
    assert_eq!(o_projection.dispatch, [128, 1, 1]);
    assert_eq!(o_projection.uniform_words, [1024, 2048, 0, 0]);
    let down_projection = GemvTiledDescriptor {
        rows: 1024,
        columns: 3072,
    }
    .plan()
    .expect("tiled down plan");
    assert_eq!(down_projection.dispatch, [128, 1, 1]);
    assert_eq!(down_projection.uniform_words, [1024, 3072, 0, 0]);
    let lm_head = GemvTiledDescriptor {
        rows: 103_424,
        columns: 1024,
    }
    .plan()
    .expect("tiled LM-head plan");
    assert_eq!(lm_head.dispatch, [12_928, 1, 1]);
    assert_eq!(lm_head.uniform_words, [103_424, 1024, 0, 0]);
}

#[test]
fn tiled_gemv_contract_helpers_reject_decoys() {
    // Kernel catalog decoys: the tiled name missing, misplaced, or renamed
    // must all drift from the pinned catalog.
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

    let misplaced_elements = EXPECTED_KERNEL_NAME_IDENTS
        .iter()
        .enumerate()
        .map(|(index, ident)| match index {
            1 => TILED_KERNEL_CONST.to_owned(),
            13 => "GEMV_KERNEL_NAME".to_owned(),
            _ => ident.to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ");
    let misplaced_catalog = catalog_fixture(&misplaced_elements, 14);
    assert_ne!(
        const_array_element_idents(module_const(&misplaced_catalog, "KERNEL_NAMES")),
        EXPECTED_KERNEL_NAME_IDENTS
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        "a tiled GEMV name anywhere but the end must be rejected"
    );

    let missing_tiled = catalog_fixture(
        &EXPECTED_KERNEL_NAME_IDENTS[..13]
            .iter()
            .map(|ident| ident.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        13,
    );
    assert_ne!(
        const_array_declared_len(module_const(&missing_tiled, "KERNEL_NAMES")),
        "14",
        "a catalog without the tiled GEMV name must be rejected"
    );

    let renamed_kernel: syn::File = syn::parse_file(
        r#"
        const GEMV_TILED_KERNEL_NAME: &str = "gemv_tiled_v2_f32";
        "#,
    )
    .expect("renamed-kernel fixture must parse");
    assert_ne!(
        const_str_literal(module_const(&renamed_kernel, TILED_KERNEL_CONST)),
        TILED_KERNEL,
        "a drifted tiled kernel name must be rejected"
    );

    // Authority surface decoys: any additional tiled-specific operation
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
            pub(super) fn step(&self) {}
            pub(super) fn tiled_step(&self) -> js_sys::Promise {}
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
        "an additional tiled-specific sealed operation must break the authority surface"
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

    // Step-encode decoys: the serial pipeline retained, a missing projection
    // stage, a tiled dispatch leaking into the prefill path, and an extra
    // uniform write must all be detected.
    let serial_gemv_retained: syn::File = syn::parse_file(
        r#"
        impl BrowserDecoderStackSession {
            fn encode_step(&self) -> Result<(), JsValue> {
                let encoder = create_stack_encoder(device, "encoder")?;
                for layer in 0..self.stack_plan.layers {
                    encode_stack_pass(&encoder, &self.gemv_pipeline, &self.gemv_q_bind_group, dispatch, &[q_offset])?;
                    encode_stack_pass(&encoder, &self.gemv_tiled_pipeline, &self.gemv_k_bind_group, dispatch, &[k_offset])?;
                }
                submit_stack_encoder(queue, &encoder)
            }
        }
        "#,
    )
    .expect("serial-gemv fixture must parse");
    let serial_method = session_method(&serial_gemv_retained, ENCODE_STEP);
    assert!(
        count_field_refs(&serial_method.block, LEGACY_GEMV_PIPELINE_FIELD) >= 1,
        "a step that still dispatches the serial gemv_f32 pipeline must be detected"
    );
    assert_ne!(
        count_field_refs(&serial_method.block, TILED_PIPELINE_FIELD),
        7,
        "a step missing projection stages must not satisfy the seven-tiled-dispatch pin"
    );

    let tiled_in_prefill: syn::File = syn::parse_file(
        r#"
        impl BrowserDecoderStackSession {
            fn encode_prefill(&self) -> Result<(), JsValue> {
                let encoder = create_stack_encoder(device, "encoder")?;
                for layer in 0..self.stack_plan.layers {
                    encode_stack_pass(&encoder, &self.gemv_tiled_pipeline, &self.prefill_query_bind_group, dispatch, &[q_offset])?;
                }
                submit_stack_encoder(queue, &encoder)
            }
        }
        "#,
    )
    .expect("tiled-prefill fixture must parse");
    let prefill_method = session_method(&tiled_in_prefill, ENCODE_PREFILL);
    assert!(
        count_field_refs(&prefill_method.block, TILED_PIPELINE_FIELD) >= 1,
        "a tiled dispatch leaking into the prefill path must be detected"
    );

    let extra_write: syn::File = syn::parse_file(
        r#"
        impl BrowserDecoderStackSession {
            fn encode_step(&self) -> Result<(), JsValue> {
                write_stack_buffer(queue, &self.hidden_pingpong_buffer, hidden_bytes)?;
                write_stack_buffer(queue, &self.gemv_q_uniform_buffer, words)?;
                write_stack_buffer(queue, &self.gemv_tiled_uniform_buffer, words)?;
                let encoder = create_stack_encoder(device, "encoder")?;
                for layer in 0..self.stack_plan.layers {
                    encode_stack_pass(&encoder, &self.gemv_tiled_pipeline, &self.gemv_q_bind_group, dispatch, &[q_offset])?;
                }
                submit_stack_encoder(queue, &encoder)
            }
        }
        "#,
    )
    .expect("extra-write fixture must parse");
    let write_method = session_method(&extra_write, ENCODE_STEP);
    assert_ne!(
        count_free_calls(&write_method.block, "write_stack_buffer"),
        17,
        "a step with an extra tiled-specific uniform write must not satisfy the seventeen-write pin: the tiled stages keep the accepted uniform words"
    );

    // Field algebra decoys: a host-side tiled scratch or a wrongly wrapped
    // pipeline handle falls outside the closed persistent algebra.
    for decoy in ["Vec<f32>", "Box<js_sys::Object>", "wgpu::ComputePipeline"] {
        let ty = syn::parse_str::<Type>(decoy).expect("decoy field type must parse");
        assert_eq!(
            classify_tiled_session_field(&ty),
            None,
            "{decoy} must fall outside the closed persistent algebra"
        );
    }
}
