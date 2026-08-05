//! Structural contract for the M6e8 persistent browser decoder stack logits
//! operation before any production exists.
//!
//! The accepted M6e7 sealed browser decoder stack session gains exactly one
//! WASM operation, `logits_decoder_stack_session() -> Promise`, implemented
//! inside the same cfg-free sealed causal module `web/decoder_stack_session.rs`.
//! This contract pins the logits-specific surface, the shared planner authority,
//! the dual pack/descriptor admission, the live-session capability guard, the
//! pure-readout discipline, and the exact persistent-topology deltas derived
//! from the M6e8 contract
//! (docs/m6e8_persistent_browser_decoder_stack_logits_contract.md, sections
//! "Accepted Boundary", "Kernels", and "Pack And Persistent Topology").
//!
//! Dual admission (the M6e6 -> M6e7 13/14-field descriptor precedent):
//!
//! - The accepted 12-section PVLCPK01 pack with the accepted 14-field begin
//!   descriptor (no `vocab_size`) admits a legacy session: the exact M6e7
//!   behaviour, evidence, and runtime topology (56 buffers / 30 bind groups /
//!   14 initial uploads / three begin planner calls) are unchanged. A `logits`
//!   call on a legacy session is rejected by the preflight capability guard
//!   with zero effect (unsupported capability, no poison). The legacy session
//!   must not pay for the logits resources: every M6e8 session field is
//!   Option-wrapped, so a legacy session allocates none of them.
//! - The 14-section pack (the accepted eleven shards plus exactly
//!   `weights.final_layernorm` and `weights.lm_head` appended in that order)
//!   with the 15-field begin descriptor (`vocab_size: 103424` pinned) admits a
//!   logits-capable session: 63 buffers / 33 bind groups / 16 initial uploads /
//!   four begin planner calls (the LM-head plan is bound once at begin through
//!   the shared `DecoderLmHeadDescriptor::plan` authority, conditionally on the
//!   admitted capability; no per-call planning exists).
//! - Closed world is preserved: unknown sections, unknown descriptor fields,
//!   and shard-order drift remain hard rejects.
//!
//! Persistent-topology derivation, applied to the accepted M6e7 session shape
//! (44 plain GpuBuffer + 12 optional prefill GpuBuffer / 22 plain handles + 19
//! optional prefill handles / 11 shader digests / 3 scalars / 1
//! DecoderKvSessionPlan / 1 DecoderStackPlan / 1 DecoderStackPrefillPlan):
//!
//! Buffers 56 -> 63 (+7 optional), created once at begin only on the
//!   logits-capable path and reused by every logits call:
//!   - the two new shared weight buffers: final norm `[1024]` f32 and LM head
//!     `[103424, 1024]` f32 (16 initial uploads total on the logits path: the
//!     accepted 14 plus these two);
//!   - one normed-row intermediate `[1024]` f32;
//!   - one logits storage buffer `[103424]` f32 and one logits readback buffer
//!     `[103424]` f32;
//!   - two static stage-uniform buffers (the rms and gemv word sets are fully
//!     static, so no per-call write exists).
//!
//! Pipelines stay 11: no kernel is added. The final RMSNorm reuses the
//!   accepted `rms_norm_f32` (rows = 1, dispatch [1, 1, 1]) and the LM head
//!   reuses the accepted `gemv_f32` (rows = 103424, columns = 1024, dispatch
//!   [1616, 1, 1]).
//!
//! Bind groups 30 -> 33 (+3 optional), created once at begin on the
//!   logits-capable path through the accepted 256-aligned dynamic-offset
//!   scheme: `prefill_logits_rms` (input = prefill hidden storage at the
//!   admitted last-row offset), `step_logits_rms` (input = hidden ping-pong
//!   slot 0), and `gemv_logits`. Total js_sys::Object handles (pipelines +
//!   bind groups) 41 -> 44.
//!
//! Shader digests stay 11 and scalars stay 3 (cache_tokens / poisoned /
//!   ready): live-session admission derives from `cache_tokens >= 1`, the
//!   capability guard derives from the single optional
//!   `Option<DecoderLmHeadPlan>` field, and the pure readout never moves the
//!   cache position.
//!
//! One logits call performs zero writes, two ordered compute passes (final
//!   RMSNorm, LM-head GEMV), one logits readback copy, one submit, and one
//!   map.

#[path = "decoder_session_contract_helpers.rs"]
mod helpers;

use std::collections::BTreeSet;

use helpers::*;
use pvlc_runtime_core::{DecoderLmHeadDescriptor, KernelId};
use syn::{
    BinOp, Expr, ExprBinary, ExprField, ExprPath, FnArg, GenericArgument, ImplItem, ImplItemFn,
    Item, LitInt, PathArguments, ReturnType, Type, Visibility, visit::Visit,
};

const WEB_RS: &str = "crates/pvlc-runtime-web/src/web.rs";
const CAUSAL_MODULE: &str = "crates/pvlc-runtime-web/src/web/decoder_stack_session.rs";
const AUTHORITY: &str = "DecoderStackSessionAuthority";
const SESSION: &str = "BrowserDecoderStackSession";
const OWNER_FIELD: &str = "decoder_stack_session";
const LOGITS_AUTHORITY_METHOD: &str = "logits";
const LOGITS_WASM_METHOD: &str = "logits_decoder_stack_session";
const LOGITS_EXECUTOR: &str = "run_logits";
const LM_HEAD_PLAN_FIELD: &str = "lm_head_plan";
const PINNED_VOCAB_SIZE: u32 = 103_424;

/// The accepted M6e7 shard order plus exactly the two M6e8 shared logits
/// shards appended at the end, in the contract's fixed order.
const EXPECTED_PACK_SHARD_IDS: [&str; 13] = [
    "weights.input_layernorm",
    "weights.q_proj",
    "weights.k_proj",
    "weights.v_proj",
    "weights.o_proj",
    "weights.mrope_cos",
    "weights.mrope_sin",
    "weights.post_attention_layernorm",
    "weights.gate_proj",
    "weights.up_proj",
    "weights.down_proj",
    "weights.final_layernorm",
    "weights.lm_head",
];

/// The accepted fourteen begin-descriptor fields plus exactly `vocab_size`:
/// the 15-field shape admits the logits capability, the 14-field prefix alone
/// (without `vocab_size`) stays the accepted legacy descriptor.
const EXPECTED_DESCRIPTOR_FIELDS: [&str; 15] = [
    "schema_version",
    "hidden_size",
    "intermediate_size",
    "query_heads",
    "key_value_heads",
    "head_dim",
    "query_width",
    "key_value_width",
    "prefix_tokens",
    "cache_capacity",
    "mrope_sections",
    "rms_norm_epsilon",
    "layers",
    "prefill_tokens",
    "vocab_size",
];

/// The twelve optional prefill buffers accepted in M6e7 plus exactly the
/// seven optional logits buffers: every M6e8 buffer is Option-wrapped so a
/// legacy (12-section) session allocates none of them.
const EXPECTED_OPTIONAL_BUFFER_FIELDS: [&str; 21] = [
    "prefill_hidden_storage_buffer",
    "prefill_norm1_buffer",
    "prefill_query_buffer",
    "prefill_key_buffer",
    "prefill_value_buffer",
    "prefill_context_buffer",
    "prefill_output_buffer",
    "prefill_norm2_buffer",
    "prefill_gate_buffer",
    "prefill_up_buffer",
    "prefill_activation_buffer",
    "prefill_zero_bias_buffer",
    "final_norm_weight_buffer",
    "lm_head_weight_buffer",
    "normed_row_buffer",
    "logits_buffer",
    "logits_readback_buffer",
    "logits_rms_uniform_buffer",
    "logits_gemv_uniform_buffer",
    "top1_result_buffer",
    "top1_readback_buffer",
];

/// The nineteen optional prefill pipelines and bind groups accepted in M6e7
/// plus exactly the three optional logits bind groups.
const EXPECTED_OPTIONAL_HANDLE_FIELDS: [&str; 27] = [
    "rms_norm_f16_pipeline",
    "gemv_tiled_f16_pipeline",
    "prefill_projection_pipeline",
    "prefill_projection_f16_pipeline",
    "prefill_mrope_pipeline",
    "kv_append_range_pipeline",
    "prefill_gqa_pipeline",
    "prefill_rms1_bind_group",
    "prefill_query_bind_group",
    "prefill_key_bind_group",
    "prefill_value_bind_group",
    "prefill_mrope_bind_group",
    "prefill_kv_append_range_bind_group",
    "prefill_gqa_bind_group",
    "prefill_output_bind_group",
    "prefill_residual_bind_group",
    "prefill_rms2_bind_group",
    "prefill_gate_bind_group",
    "prefill_up_bind_group",
    "prefill_swiglu_bind_group",
    "prefill_down_bind_group",
    "prefill_residual2_bind_group",
    "prefill_logits_rms_bind_group",
    "step_logits_rms_bind_group",
    "gemv_logits_bind_group",
    "top1_pipeline",
    "top1_bind_group",
];

/// Collects every path segment identifier (type and expression positions) so
/// module-level references to exact core types can be pinned without
/// depending on formatting or on whether the type is named through an import
/// (`DecoderLmHeadDescriptor::pinned`) or a full path.
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

/// Detects the live-session admission guard: any ordering or equality
/// comparison that mentions `cache_tokens` against the literal `0` or `1`
/// (`cache_tokens == 0`, `cache_tokens >= 1`, `cache_tokens < 1`, and their
/// mirror forms are the accepted shapes).
#[derive(Default)]
struct LogitsAdmission {
    found: bool,
}

impl<'ast> Visit<'ast> for LogitsAdmission {
    fn visit_expr_binary(&mut self, expression: &'ast ExprBinary) {
        if matches!(
            expression.op,
            BinOp::Eq(_) | BinOp::Ne(_) | BinOp::Gt(_) | BinOp::Ge(_) | BinOp::Lt(_) | BinOp::Le(_)
        ) {
            let operands = [&expression.left, &expression.right];
            let mentions_cache_tokens = operands.iter().any(|side| {
                let mut references = CacheTokensRefs::default();
                references.visit_expr(side);
                references.0
            });
            let has_admission_literal = operands.iter().any(|side| {
                matches!(
                    side.as_ref(),
                    Expr::Lit(lit)
                        if matches!(&lit.lit, syn::Lit::Int(value)
                            if value.base10_digits() == "0" || value.base10_digits() == "1")
                )
            });
            if mentions_cache_tokens && has_admission_literal {
                self.found = true;
            }
        }
        syn::visit::visit_expr_binary(self, expression);
    }
}

/// Counts references to one named field so the logits capability guard (the
/// `session.lm_head_plan` check) can be pinned structurally.
struct FieldRefs<'a> {
    name: &'a str,
    refs: usize,
}

impl<'ast> Visit<'ast> for FieldRefs<'_> {
    fn visit_expr_field(&mut self, expression: &'ast ExprField) {
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

/// Counts assignments whose left side is a `cache_tokens` field: the logits
/// readout must never move the admitted cache position.
#[derive(Default)]
struct CacheTokensAssignments(usize);

impl<'ast> Visit<'ast> for CacheTokensAssignments {
    fn visit_expr_assign(&mut self, expression: &'ast syn::ExprAssign) {
        if matches!(
            expression.left.as_ref(),
            Expr::Field(field)
                if matches!(&field.member, syn::Member::Named(name) if name == "cache_tokens")
        ) {
            self.0 += 1;
        }
        syn::visit::visit_expr_assign(self, expression);
    }
}

/// Counts `.plan()` calls nested inside an `if` expression: under dual
/// admission the LM-head planner call must stay conditional on the admitted
/// pack/descriptor capability so the legacy path keeps its accepted three
/// unconditional planner calls.
#[derive(Default)]
struct ConditionalPlanCalls {
    if_depth: usize,
    conditional: usize,
}

impl<'ast> Visit<'ast> for ConditionalPlanCalls {
    fn visit_expr_if(&mut self, expression: &'ast syn::ExprIf) {
        self.if_depth += 1;
        syn::visit::visit_expr_if(self, expression);
        self.if_depth -= 1;
    }

    fn visit_expr_method_call(&mut self, expression: &'ast syn::ExprMethodCall) {
        if expression.method == "plan" && self.if_depth > 0 {
            self.conditional += 1;
        }
        syn::visit::visit_expr_method_call(self, expression);
    }
}

/// Counts hand-constructions of the exact core LM-head plan type; the causal
/// module must bind the planner result, never forge it.
#[derive(Default)]
struct LmHeadPlanConstructions(usize);

impl<'ast> Visit<'ast> for LmHeadPlanConstructions {
    fn visit_expr_struct(&mut self, expression: &'ast syn::ExprStruct) {
        if expression
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "DecoderLmHeadPlan")
        {
            self.0 += 1;
        }
        syn::visit::visit_expr_struct(self, expression);
    }
}

/// Detects the pinned `103424` vocabulary literal used for descriptor
/// validation and buffer sizing (underscore-separated spellings count).
struct PinnedVocabLiteral(bool);

impl<'ast> Visit<'ast> for PinnedVocabLiteral {
    fn visit_lit_int(&mut self, literal: &'ast LitInt) {
        if literal.base10_digits() == "103424" {
            self.0 = true;
        }
    }
}

/// Counts free-function calls by terminal path identifier inside one body so
/// the exact logits execution discipline (passes / copy / submit / map / no
/// writes) can be pinned against the accepted raw-WebGPU helpers.
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

fn returns_js_promise(output: &ReturnType) -> bool {
    let ReturnType::Type(_, ty) = output else {
        return false;
    };
    is_exact_js_path(ty, "Promise")
}

/// The exact `Option<pvlc_runtime_core::...>` capability-plan shape: under
/// dual admission the LM-head plan exists only on logits-capable sessions, so
/// the plain plan type stays outside the persistent algebra.
fn exact_optional_core_plan(ty: &Type, expected: &str) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    if path.qself.is_some()
        || path.path.leading_colon.is_some()
        || path.path.segments.len() != 1
        || path.path.segments[0].ident != "Option"
    {
        return false;
    }
    let PathArguments::AngleBracketed(arguments) = &path.path.segments[0].arguments else {
        return false;
    };
    if arguments.args.len() != 1 {
        return false;
    }
    let Some(GenericArgument::Type(inner)) = arguments.args.first() else {
        return false;
    };
    exact_core_plan(inner, expected)
}

/// Closed persistent-field algebra for the dual-admission session: the
/// accepted M6e7 classes plus exactly one optional exact core LM-head plan
/// (the logits capability). A plain `DecoderLmHeadPlan` would force the
/// capability onto legacy sessions and stays outside the algebra, as do
/// host-side storage, indirection, or wrapper types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LogitsFieldClass {
    Accepted(SessionFieldClass),
    CorePrefillPlan,
    CoreWeightResourcePlan,
    OptionalCoreLmHeadPlan,
}

fn classify_logits_session_field(ty: &Type) -> Option<LogitsFieldClass> {
    if exact_core_plan(ty, "DecoderStackPrefillPlan") {
        return Some(LogitsFieldClass::CorePrefillPlan);
    }
    if exact_core_plan(ty, "DecoderWeightResourcePlan") {
        return Some(LogitsFieldClass::CoreWeightResourcePlan);
    }
    if exact_optional_core_plan(ty, "DecoderLmHeadPlan") {
        return Some(LogitsFieldClass::OptionalCoreLmHeadPlan);
    }
    classify_session_field(ty).map(LogitsFieldClass::Accepted)
}

#[derive(Default)]
struct LogitsTopology {
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
        LOGITS_AUTHORITY_METHOD,
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

fn const_u32_literal(value: &syn::ItemConst) -> String {
    let Expr::Lit(lit) = value.expr.as_ref() else {
        panic!("constant {} must be a literal", value.ident);
    };
    let syn::Lit::Int(integer) = &lit.lit else {
        panic!("constant {} must be an integer literal", value.ident);
    };
    integer.base10_digits().to_owned()
}

/// Extracts the declared element count of a `const NAME: [T; N]` constant
/// without inspecting the elements (kernel name constants reference other
/// constants, unlike the shard/descriptor string literals).
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

/// Extracts the declared element count and the literal elements of a
/// `const NAME: [&str; N] = [...]` constant.
fn const_str_array(value: &syn::ItemConst) -> (String, Vec<String>) {
    let Type::Array(array) = value.ty.as_ref() else {
        panic!("constant {} must be an array", value.ident);
    };
    let Expr::Lit(length_lit) = &array.len else {
        panic!("constant {} length must be a literal", value.ident);
    };
    let syn::Lit::Int(length) = &length_lit.lit else {
        panic!("constant {} length must be an integer", value.ident);
    };
    let Expr::Array(elements) = value.expr.as_ref() else {
        panic!("constant {} must be an array literal", value.ident);
    };
    let items = elements
        .elems
        .iter()
        .map(|element| {
            let Expr::Lit(lit) = element else {
                panic!("constant {} elements must be literals", value.ident);
            };
            let syn::Lit::Str(text) = &lit.lit else {
                panic!("constant {} elements must be strings", value.ident);
            };
            text.value()
        })
        .collect();
    (length.base10_digits().to_owned(), items)
}

#[test]
fn logits_extends_the_sealed_authority_with_exactly_one_operation() {
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
    let authority_impl = authority_impl(&module);
    for item in &authority_impl.items {
        if let ImplItem::Fn(function) = item {
            assert!(
                restricted_to_parent(&function.vis)
                    && function.sig.generics.params.is_empty()
                    && only_doc_attributes(&function.attrs),
                "{} must be an unconditional non-generic pub(super) operation",
                function.sig.ident
            );
        }
    }
    assert_eq!(
        collect_authority_methods(authority_impl),
        expected_authority_methods(),
        "sealed authority operation surface drifted: M6e8 adds exactly logits"
    );

    let logits = authority_method(authority_impl, LOGITS_AUTHORITY_METHOD);
    let typed_arguments = logits
        .sig
        .inputs
        .iter()
        .filter_map(|argument| match argument {
            FnArg::Typed(typed) => Some(typed),
            FnArg::Receiver(_) => None,
        })
        .collect::<Vec<_>>();
    assert!(
        matches!(logits.sig.inputs.first(), Some(FnArg::Receiver(_))) && typed_arguments.is_empty(),
        "sealed logits must take exactly (&self): the current hidden row is read on device"
    );
    assert!(
        returns_js_promise(&logits.sig.output),
        "sealed logits must return js_sys::Promise"
    );

    assert_eq!(
        count_method_calls(&logits.block, "plan"),
        0,
        "logits replans the session: the exact DecoderLmHeadPlan is bound once at begin"
    );
    assert_eq!(
        count_method_calls(&logits.block, "plan_step"),
        0,
        "logits replans a decode step: the decode step planner stays decode-only"
    );
    assert_eq!(
        count_method_calls(&logits.block, "plan_cache_transition"),
        0,
        "logits is a pure readout: it must not plan or apply a cache transition"
    );

    let readout_preflight = free_function(&module, "prepare_logits_readout");
    let mut admission = LogitsAdmission::default();
    admission.visit_block(&readout_preflight.block);
    assert!(
        admission.found,
        "sealed logits must reject a session before its first admitted operation through an explicit cache_tokens comparison against 0 or 1"
    );
    assert!(
        count_field_refs(&readout_preflight.block, LM_HEAD_PLAN_FIELD) >= 1,
        "sealed logits must guard on the session.{LM_HEAD_PLAN_FIELD} capability: a legacy (12-section) session is rejected by the preflight with zero effect, not poisoned"
    );

    let mut assignments = CacheTokensAssignments::default();
    assignments.visit_block(&logits.block);
    assert_eq!(
        assignments.0, 0,
        "sealed logits is a pure readout: it must never assign cache_tokens, so repeated calls stay bit-identical"
    );

    let mut logits_paths = PathIdents::default();
    logits_paths.visit_block(&logits.block);
    for required in ["prepare_logits_readout", LOGITS_EXECUTOR] {
        assert!(
            logits_paths.0.contains(required),
            "sealed logits must keep the accepted acquire/restore zero-effect discipline and delegate the device work to {LOGITS_EXECUTOR}: missing {required}"
        );
    }

    let begin = authority_method(authority_impl, "begin");
    assert_eq!(
        count_method_calls(&begin.block, "plan"),
        0,
        "sealed begin must consume the shared M7q1 preparation instead of duplicating planners"
    );
    let mut begin_paths = PathIdents::default();
    begin_paths.visit_block(&begin.block);
    assert!(
        begin_paths.0.contains("prepare_stack_begin"),
        "sealed begin must delegate to the shared precision-aware preparation authority"
    );
    let prepare = free_function(&module, "prepare_stack_begin");
    let mut conditional_plans = ConditionalPlanCalls::default();
    conditional_plans.visit_block(&prepare.block);
    assert!(
        conditional_plans.conditional >= 1,
        "dual admission: the shared preparation must keep LM-head planning conditional on the admitted pack/descriptor capability"
    );
    let begin_with_override = authority_method(authority_impl, "begin_with_shader_override");
    let override_plan_calls = count_method_calls(&begin_with_override.block, "plan");
    assert!(
        override_plan_calls == 0
            || ((override_plan_calls == 3 || override_plan_calls == 4)
                && planner_result_reaches_live_consumer(
                    begin_with_override,
                    "plan",
                    override_plan_calls,
                    "owner"
                )),
        "sealed begin_with_shader_override must either delegate to begin or bind its three (legacy) / four (logits-capable) planner results without compute-then-ignore"
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
            "begin" | "begin_with_shader_override" => {
                assert_eq!(plan_step_calls, 0, "{name} replans a step outside step");
                assert_eq!(
                    transition_calls, 0,
                    "{name} replans a cache transition outside step"
                );
            }
            "step" => {
                assert_eq!(plan_calls, 0, "step replans the session outside begin");
            }
            "prefill" => {
                assert_eq!(
                    plan_calls, 0,
                    "prefill must not rebuild operand-validating model weights"
                );
                assert_eq!(
                    count_method_calls(&function.block, "plan_prefill"),
                    1,
                    "prefill must bind exactly one payload-free geometry prefill planner result"
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

    let mut forbidden = ForbiddenCausalSyntax::default();
    forbidden.visit_file(&module);
    assert_eq!(forbidden.unsafe_blocks, 0, "causal module contains unsafe");
    assert_eq!(
        forbidden.derive_attributes, 0,
        "causal module uses a derive macro: no Clone/Default side doors"
    );
    for item in matching_impls(&module, AUTHORITY)
        .into_iter()
        .chain(matching_impls(&module, SESSION))
    {
        for impl_item in &item.items {
            if let ImplItem::Fn(function) = impl_item {
                assert!(
                    function.sig.ident != "from_parts",
                    "causal module must not expose a from_parts forgery constructor"
                );
            }
        }
    }
}

#[test]
fn web_runtime_exports_exactly_one_logits_delegation() {
    let root = parse(WEB_RS);
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
        (LOGITS_WASM_METHOD, LOGITS_AUTHORITY_METHOD),
        ("prefill_decoder_stack_session", "prefill"),
        ("step_decoder_stack_session", "step"),
        ("top1_decoder_stack_session", "top1"),
    ];
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

    let logits = wasm_impl
        .items
        .iter()
        .find_map(|item| match item {
            ImplItem::Fn(function) if function.sig.ident == LOGITS_WASM_METHOD => Some(function),
            _ => None,
        })
        .expect("logits export is missing");
    let typed_arguments = logits
        .sig
        .inputs
        .iter()
        .filter_map(|argument| match argument {
            FnArg::Typed(typed) => Some(typed),
            FnArg::Receiver(_) => None,
        })
        .collect::<Vec<_>>();
    assert!(
        typed_arguments.is_empty(),
        "{LOGITS_WASM_METHOD} must take no arguments: the hidden row never crosses the host"
    );
    assert!(
        returns_js_promise(&logits.sig.output),
        "{LOGITS_WASM_METHOD} must return js_sys::Promise"
    );

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
            LOGITS_WASM_METHOD,
            "prefill_decoder_stack_session",
            "step_decoder_stack_session",
            "top1_decoder_stack_session",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        "decoder stack WASM operation allowlist drifted: M6e8 grows it by exactly one"
    );
    assert!(
        public_inherent_methods(&root, "WebRuntime").contains(LOGITS_WASM_METHOD),
        "the closed public WebRuntime method allowlist must gain exactly {LOGITS_WASM_METHOD}"
    );

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
        "WebRuntime must keep owning exactly one DecoderStackSessionAuthority"
    );
}

#[test]
fn logits_session_topology_extends_the_closed_persistent_algebra() {
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
        "causal module must contain only the session/authority and resident-cache implementations"
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
            && only_doc_attributes(&authority.attrs)
            && authority.fields.len() == 4,
        "sealed authority must own exactly its device, queue, async session owner, and shared resident-weight cache"
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
    let mut topology = LogitsTopology::default();
    let mut optional_buffer_names = BTreeSet::new();
    let mut optional_handle_names = BTreeSet::new();
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
        match classify_logits_session_field(&field.ty) {
            Some(LogitsFieldClass::Accepted(SessionFieldClass::WgpuWebGpuBuffer)) => {
                if optional_session_field_class(&field.ty).is_some() {
                    topology.optional_buffers += 1;
                    optional_buffer_names.insert(name);
                } else {
                    topology.plain_buffers += 1;
                }
            }
            Some(LogitsFieldClass::Accepted(SessionFieldClass::JsObjectHandle)) => {
                if optional_session_field_class(&field.ty).is_some() {
                    topology.optional_handles += 1;
                    optional_handle_names.insert(name);
                } else {
                    topology.plain_handles += 1;
                }
            }
            Some(LogitsFieldClass::Accepted(SessionFieldClass::CoreKvPlan)) => {
                topology.kv_plans += 1;
            }
            Some(LogitsFieldClass::Accepted(SessionFieldClass::CoreStackPlan)) => {
                topology.stack_plans += 1;
            }
            Some(LogitsFieldClass::CorePrefillPlan) => {
                topology.prefill_plans += 1;
            }
            Some(LogitsFieldClass::CoreWeightResourcePlan) => {
                topology.weight_resource_plans += 1;
            }
            Some(LogitsFieldClass::OptionalCoreLmHeadPlan) => {
                topology.lm_head_plans += 1;
                assert_eq!(
                    name, LM_HEAD_PLAN_FIELD,
                    "the single logits capability field must be named {LM_HEAD_PLAN_FIELD}"
                );
            }
            Some(LogitsFieldClass::Accepted(SessionFieldClass::Scalar)) => {
                topology.scalars += 1;
            }
            Some(LogitsFieldClass::Accepted(SessionFieldClass::Digest)) => {
                topology.digests += 1;
            }
            Some(LogitsFieldClass::Accepted(
                SessionFieldClass::CoreAttentionPlan | SessionFieldClass::CoreLayerPlan,
            )) => {
                panic!("decoder stack session must not hold attention-block or layer plans")
            }
            None => panic!(
                "browser decoder stack session field hides state outside the closed persistent algebra"
            ),
        }
    }
    assert_eq!(
        topology.plain_buffers, 47,
        "the accepted M6e6 forty-four exact wgpu::webgpu::GpuBuffer resources stay mandatory on both admission paths; M7o2 amendment: plus the three mandatory split-K resources (the scratch partials plane and the two split stage uniforms)"
    );
    assert_eq!(
        topology.optional_buffers, 21,
        "optional prefill/logits buffers plus two lazily allocated GPU top-1 buffers"
    );
    assert_eq!(
        topology.plain_buffers + topology.optional_buffers,
        68,
        "logits-capable session topology plus two lazy GPU top-1 buffers"
    );
    assert_eq!(
        optional_buffer_names,
        EXPECTED_OPTIONAL_BUFFER_FIELDS
            .into_iter()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>(),
        "optional buffer field set drifted: the accepted twelve prefill buffers plus exactly the seven logits buffers"
    );
    assert_eq!(
        topology.plain_handles, 24,
        "the accepted M6e6 handles minus the retired serial GQA pipeline and bind group stay mandatory on both admission paths; M7o2 amendment: plus the two split pipelines and the two split bind groups"
    );
    assert_eq!(
        topology.optional_handles, 27,
        "the accepted optional handles plus the lazy GPU top-1 pipeline and bind group"
    );
    assert_eq!(
        topology.plain_handles + topology.optional_handles,
        51,
        "logits-capable session topology includes the accepted handles and GPU top-1 resources"
    );
    assert_eq!(
        optional_handle_names,
        EXPECTED_OPTIONAL_HANDLE_FIELDS
            .into_iter()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>(),
        "optional handle field set drifted: the accepted nineteen prefill handles plus exactly the three logits bind groups"
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
        "session must hold exactly one Option<pvlc_runtime_core::DecoderLmHeadPlan>: the capability exists only on logits-capable sessions and no per-call planning exists"
    );
    assert_eq!(
        topology.scalars, 4,
        "session keeps cache/poison/ready state plus resident-weight byte telemetry"
    );
    assert_eq!(
        topology.digests, 16,
        "session keeps the accepted decoder digests plus checkpoint and GPU top-1 identities"
    );

    let kernel_count = const_array_declared_len(module_const(&module, "KERNEL_NAMES"));
    assert_eq!(
        kernel_count, "14",
        "M7o5 appends tiled GEMV after the accepted thirteen kernel names"
    );
    assert_eq!(
        const_u32_literal(module_const(&module, "PACK_SECTION_COUNT")),
        "14",
        "the logits-capable PVLCPK01 stack pack declares fourteen sections: the descriptor plus thirteen shards"
    );
    assert_eq!(
        const_u32_literal(module_const(&module, "LEGACY_PACK_SECTION_COUNT")),
        "12",
        "dual admission: the accepted M6e7 twelve-section pack stays a first-class admitted format"
    );
    let (shard_count, shard_ids) = const_str_array(module_const(&module, "PACK_SHARD_IDS"));
    assert_eq!(
        shard_count, "13",
        "PACK_SHARD_IDS must declare exactly thirteen shards"
    );
    assert_eq!(
        shard_ids,
        EXPECTED_PACK_SHARD_IDS
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        "pack shard order drifted: the accepted eleven shards stay in order and weights.final_layernorm / weights.lm_head append at the end"
    );
    let (descriptor_count, descriptor_fields) =
        const_str_array(module_const(&module, "DESCRIPTOR_FIELDS"));
    assert_eq!(
        descriptor_count, "15",
        "the begin descriptor allowlist must declare exactly fifteen fields"
    );
    assert_eq!(
        descriptor_fields,
        EXPECTED_DESCRIPTOR_FIELDS
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        "begin descriptor field set drifted: the accepted fourteen fields stay and vocab_size is added; its presence admits the logits capability"
    );
    let mut vocab = PinnedVocabLiteral(false);
    vocab.visit_file(&module);
    assert!(
        vocab.0,
        "causal module must pin the 103424 vocabulary size for the 15-field descriptor validation and logits buffer sizing"
    );

    let mut paths = PathIdents::default();
    paths.visit_file(&module);
    for required in ["DecoderLmHeadDescriptor", "DecoderLmHeadPlan"] {
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
    let mut forged_lm_head = LmHeadPlanConstructions::default();
    forged_lm_head.visit_file(&module);
    assert_eq!(
        forged_lm_head.0, 0,
        "causal module must not hand-construct the exact core DecoderLmHeadPlan"
    );

    let executor = free_function(&module, LOGITS_EXECUTOR);
    assert_eq!(
        count_free_calls(&executor.block, "encode_stack_pass"),
        2,
        "one logits call must encode exactly two ordered compute passes: the final RMSNorm and the LM-head GEMV"
    );
    assert_eq!(
        count_free_calls(&executor.block, "encode_stack_copy"),
        1,
        "one logits call must encode exactly one copy: the logits storage into the logits readback"
    );
    assert_eq!(
        count_free_calls(&executor.block, "submit_stack_encoder"),
        1,
        "one logits call must submit exactly one command buffer"
    );
    assert_eq!(
        count_free_calls(&executor.block, "map_stack_buffer"),
        1,
        "one logits call must map exactly one buffer: the logits readback"
    );
    assert_eq!(
        count_free_calls(&executor.block, "write_stack_buffer"),
        0,
        "one logits call must perform zero writes: both stage uniforms are static, created once at begin"
    );

    // The web topology pins must agree with the shared arithmetic authority:
    // the exact core LM-head planner is the single source of the stage
    // uniforms, dispatch shapes, and buffer sizes the counts above assume.
    let final_norm_weight = vec![0.0_f32; 1024];
    let lm_head_weight = vec![0.0_f32; PINNED_VOCAB_SIZE as usize * 1024];
    let plan = DecoderLmHeadDescriptor::pinned(&final_norm_weight, &lm_head_weight)
        .plan()
        .expect("pinned LM-head plan");
    assert_eq!(plan.hidden_size, 1024);
    assert_eq!(plan.vocab_size, PINNED_VOCAB_SIZE);
    assert_eq!(plan.final_norm_weight_bytes, 4096);
    assert_eq!(plan.lm_head_weight_bytes, 423_624_704);
    assert_eq!(plan.normed_row_bytes, 4096);
    assert_eq!(plan.logits_bytes, 413_696);
    assert_eq!(plan.stage_invocations.len(), 2);
    assert_eq!(plan.stage_invocations[0].kernel, KernelId::RmsNormF32);
    assert_eq!(plan.stage_invocations[0].dispatch, [1, 1, 1]);
    assert_eq!(plan.stage_invocations[1].kernel, KernelId::GemvTiledF32);
    assert_eq!(plan.stage_invocations[1].workgroup_size, [256, 1, 1]);
    assert_eq!(plan.stage_invocations[1].dispatch, [12_928, 1, 1]);
    assert_eq!(
        plan.stage_uniform_words,
        [
            [1, 1024, 1.0e-5_f32.to_bits(), 0],
            [PINNED_VOCAB_SIZE, 1024, 0, 0],
        ]
    );
}

#[test]
fn logits_contract_helpers_reject_decoys() {
    let accepted_return: Type =
        syn::parse_str("js_sys::Promise").expect("accepted return type must parse");
    assert!(returns_js_promise(&ReturnType::Type(
        Default::default(),
        Box::new(accepted_return.clone())
    )));
    for decoy in ["&js_sys::Promise", "Promise", "js_sys::Uint8Array", "()"] {
        let ty = syn::parse_str::<Type>(decoy).expect("decoy return type must parse");
        assert!(
            !returns_js_promise(&ReturnType::Type(Default::default(), Box::new(ty))),
            "{decoy} must not satisfy the Promise return pin"
        );
    }

    let optional_lm_head_plan: Type =
        syn::parse_str("Option<pvlc_runtime_core::DecoderLmHeadPlan>")
            .expect("optional LM-head plan type must parse");
    assert_eq!(
        classify_logits_session_field(&optional_lm_head_plan),
        Some(LogitsFieldClass::OptionalCoreLmHeadPlan),
        "the logits capability must be storable as the single optional exact core plan"
    );
    for decoy in [
        // A plain plan would force the capability (and its 404 MB of
        // resources) onto legacy sessions.
        "pvlc_runtime_core::DecoderLmHeadPlan",
        "Option<crate::DecoderLmHeadPlan>",
        "Option<Option<pvlc_runtime_core::DecoderLmHeadPlan>>",
        "Option<Vec<f32>>",
        "Vec<f32>",
        "crate::DecoderLmHeadPlan",
        "DecoderLmHeadPlan",
    ] {
        let ty = syn::parse_str::<Type>(decoy).expect("decoy field type must parse");
        assert_eq!(
            classify_logits_session_field(&ty),
            None,
            "{decoy} must fall outside the closed persistent algebra"
        );
    }
    let prefill_plan: Type = syn::parse_str("pvlc_runtime_core::DecoderStackPrefillPlan")
        .expect("prefill plan type must parse");
    assert_eq!(
        classify_logits_session_field(&prefill_plan),
        Some(LogitsFieldClass::CorePrefillPlan),
        "the prefill plan must keep its own accepted class"
    );

    let admitted = parse_impl_method(
        r#"
        impl DecoderStackSessionAuthority {
            fn logits(&self) -> js_sys::Promise {
                let prepared = (|| {
                    let (lease, session) = acquire_stack_session(&self.owner)?;
                    let Some(lm_head_plan) = session.lm_head_plan.as_ref() else {
                        restore_stack_session(&self.owner, lease, session);
                        return Err(unsupported());
                    };
                    if session.cache_tokens >= 1 {
                        admit(session, lm_head_plan)
                    } else {
                        reject()
                    }
                })();
                prepared
            }
        }
        "#,
        "logits",
    );
    let mut admission = LogitsAdmission::default();
    admission.visit_block(&admitted.block);
    assert!(admission.found);
    assert!(
        count_field_refs(&admitted.block, LM_HEAD_PLAN_FIELD) >= 1,
        "the capability-guarded admission fixture must be detected"
    );

    let zero_guarded = parse_impl_method(
        r#"
        impl DecoderStackSessionAuthority {
            fn logits(&self) -> js_sys::Promise {
                let prepared = (|| {
                    let (lease, session) = acquire_stack_session(&self.owner)?;
                    if session.lm_head_plan.is_none() {
                        restore_stack_session(&self.owner, lease, session);
                        return Err(unsupported());
                    }
                    if session.cache_tokens == 0 {
                        reject()
                    }
                    admit(session)
                })();
                prepared
            }
        }
        "#,
        "logits",
    );
    let mut admission = LogitsAdmission::default();
    admission.visit_block(&zero_guarded.block);
    assert!(
        admission.found,
        "the mirror cache_tokens == 0 rejection must satisfy the live-session admission pin"
    );
    assert!(
        count_field_refs(&zero_guarded.block, LM_HEAD_PLAN_FIELD) >= 1,
        "the is_none capability-guard shape must be detected"
    );

    // A logits call that admits a legacy session: no lm_head_plan capability
    // check anywhere, so the zero-effect unsupported-capability rejection is
    // missing even though the cache-position guard is present.
    let capability_blind = parse_impl_method(
        r#"
        impl DecoderStackSessionAuthority {
            fn logits(&self) -> js_sys::Promise {
                let prepared = (|| {
                    let (lease, session) = acquire_stack_session(&self.owner)?;
                    if session.cache_tokens >= 1 {
                        admit(session)
                    } else {
                        reject()
                    }
                })();
                prepared
            }
        }
        "#,
        "logits",
    );
    let mut admission = LogitsAdmission::default();
    admission.visit_block(&capability_blind.block);
    assert!(admission.found);
    assert_eq!(
        count_field_refs(&capability_blind.block, LM_HEAD_PLAN_FIELD),
        0,
        "a logits call that never checks the lm_head_plan capability must be detected: legacy sessions would be admitted instead of rejected with zero effect"
    );

    for source in [
        // No cache position comparison at all.
        r#"
        impl DecoderStackSessionAuthority {
            fn logits(&self) -> js_sys::Promise {
                let prepared = (|| {
                    let (lease, session) = acquire_stack_session(&self.owner)?;
                    if session.lm_head_plan.is_some() {
                        admit(session)
                    } else {
                        reject()
                    }
                })();
                prepared
            }
        }
        "#,
        // A comparison against a different literal admits nothing about the
        // zero-position rejection.
        r#"
        impl DecoderStackSessionAuthority {
            fn logits(&self) -> js_sys::Promise {
                let prepared = (|| {
                    let (lease, session) = acquire_stack_session(&self.owner)?;
                    if session.cache_tokens == 2 {
                        reject()
                    }
                    admit(session)
                })();
                prepared
            }
        }
        "#,
    ] {
        let unguarded = parse_impl_method(source, "logits");
        let mut admission = LogitsAdmission::default();
        admission.visit_block(&unguarded.block);
        assert!(
            !admission.found,
            "a missing or non-admission cache_tokens comparison must not satisfy the live-session admission pin"
        );
    }

    let mutating = parse_impl_method(
        r#"
        impl DecoderStackSessionAuthority {
            fn logits(&self) -> js_sys::Promise {
                let prepared = (|| {
                    let (lease, mut session) = acquire_stack_session(&self.owner)?;
                    session.cache_tokens = session.cache_tokens + 1;
                    admit(session)
                })();
                prepared
            }
        }
        "#,
        "logits",
    );
    let mut assignments = CacheTokensAssignments::default();
    assignments.visit_block(&mutating.block);
    assert_eq!(
        assignments.0, 1,
        "a cache-moving logits forgery must be detected"
    );

    let forged: syn::File = syn::parse_file(
        r#"
        fn forge() {
            let _ = pvlc_runtime_core::DecoderLmHeadPlan {};
        }
        "#,
    )
    .expect("forged LM-head plan fixture must parse");
    let mut forged_lm_head = LmHeadPlanConstructions::default();
    forged_lm_head.visit_file(&forged);
    assert_eq!(forged_lm_head.0, 1);

    let unconditional_plan = parse_impl_method(
        r#"
        impl DecoderStackSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<Promise, Error> {
                let prepared = (|| {
                    let kv_plan = kv_descriptor.plan()?;
                    let stack_plan = stack_descriptor.plan()?;
                    let prefill_plan = prefill_descriptor.plan()?;
                    let lm_head_plan = lm_head_descriptor.plan()?;
                    Ok((kv_plan, stack_plan, prefill_plan, Some(lm_head_plan)))
                })();
                prepared
            }
        }
        "#,
        "begin",
    );
    let mut conditional_plans = ConditionalPlanCalls::default();
    conditional_plans.visit_block(&unconditional_plan.block);
    assert_eq!(
        conditional_plans.conditional, 0,
        "an unconditional LM-head planner call must be detected: the legacy path would plan (and fail) without its shards"
    );
    let conditional_plan = parse_impl_method(
        r#"
        impl DecoderStackSessionAuthority {
            fn begin(&self, descriptor: Descriptor) -> Result<Promise, Error> {
                let prepared = (|| {
                    let kv_plan = kv_descriptor.plan()?;
                    let stack_plan = stack_descriptor.plan()?;
                    let prefill_plan = prefill_descriptor.plan()?;
                    let lm_head_plan = if logits_capable {
                        Some(lm_head_descriptor.plan()?)
                    } else {
                        None
                    };
                    Ok((kv_plan, stack_plan, prefill_plan, lm_head_plan))
                })();
                prepared
            }
        }
        "#,
        "begin",
    );
    let mut conditional_plans = ConditionalPlanCalls::default();
    conditional_plans.visit_block(&conditional_plan.block);
    assert_eq!(
        conditional_plans.conditional, 1,
        "the dual-admission conditional LM-head planner call must satisfy the pin"
    );

    let extra_method: syn::File = syn::parse_file(
        r#"
        impl DecoderStackSessionAuthority {
            pub(super) fn abort(&self) {}
            pub(super) fn begin(&self) {}
            pub(super) fn begin_with_shader_override(&self) {}
            pub(super) fn finish(&self) {}
            pub(super) fn logits(&self) -> js_sys::Promise {}
            pub(super) fn logits_top_k(&self) -> js_sys::Promise {}
            pub(super) fn new() {}
            pub(super) fn prefill(&self) {}
            pub(super) fn shader_sources_json(&self) {}
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
        "an additional logits-adjacent operation must break the sealed authority surface"
    );

    let public_method: syn::File = syn::parse_file(
        r#"
        impl DecoderStackSessionAuthority {
            pub fn logits(&self) -> js_sys::Promise {}
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

    let public_field: syn::File = syn::parse_file(
        r#"
        struct BrowserDecoderStackSession {
            cache_tokens: u32,
            pub logits_buffer: wgpu::webgpu::GpuBuffer,
        }
        "#,
    )
    .expect("public-field fixture must parse");
    let Item::Struct(public_field_session) = &public_field.items[0] else {
        panic!("public-field fixture is not a struct");
    };
    assert!(
        public_field_session
            .fields
            .iter()
            .any(|field| !inherited(&field.vis)),
        "a public session field must be detected"
    );

    let clone_derive = syn::parse_file(
        r#"
        #[derive(Clone)]
        struct BrowserDecoderStackSession {
            cache_tokens: u32,
        }
        "#,
    )
    .expect("clone-derive fixture must parse");
    let mut forbidden = ForbiddenCausalSyntax::default();
    forbidden.visit_file(&clone_derive);
    assert_eq!(
        forbidden.derive_attributes, 1,
        "a Clone derive on the session must be detected"
    );

    let from_parts = syn::parse_file(
        r#"
        impl BrowserDecoderStackSession {
            fn from_parts(cache_tokens: u32) -> Self {
                Self { cache_tokens }
            }
        }
        "#,
    )
    .expect("from-parts fixture must parse");
    let Item::Impl(from_parts_impl) = &from_parts.items[0] else {
        panic!("from-parts fixture is not an impl");
    };
    assert!(
        from_parts_impl.items.iter().any(|item| matches!(
            item,
            ImplItem::Fn(function) if function.sig.ident == "from_parts"
        )),
        "a from_parts forgery constructor must be detectable"
    );

    let shard_fixture = |ids: &str| -> syn::File {
        syn::parse_file(&format!("const PACK_SHARD_IDS: [&str; 13] = [{ids}];"))
            .expect("shard fixture must parse")
    };
    let quoted = EXPECTED_PACK_SHARD_IDS
        .iter()
        .map(|id| format!("\"{id}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let (_, accepted_ids) =
        const_str_array(module_const(&shard_fixture(&quoted), "PACK_SHARD_IDS"));
    assert_eq!(
        accepted_ids,
        EXPECTED_PACK_SHARD_IDS
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    );
    // The two new shards swapped.
    let swapped = EXPECTED_PACK_SHARD_IDS
        .iter()
        .enumerate()
        .map(|(index, id)| match index {
            11 => "\"weights.lm_head\"".to_owned(),
            12 => "\"weights.final_layernorm\"".to_owned(),
            _ => format!("\"{id}\""),
        })
        .collect::<Vec<_>>()
        .join(", ");
    let (_, swapped_ids) =
        const_str_array(module_const(&shard_fixture(&swapped), "PACK_SHARD_IDS"));
    assert_ne!(
        swapped_ids,
        EXPECTED_PACK_SHARD_IDS
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        "a swapped logits shard order must be rejected"
    );
    // The lm_head shard missing (the accepted M6e7 eleven plus final norm).
    let missing = EXPECTED_PACK_SHARD_IDS[..12]
        .iter()
        .map(|id| format!("\"{id}\""))
        .chain(["\"weights.down_proj_extra\"".to_owned()])
        .collect::<Vec<_>>()
        .join(", ");
    let (_, missing_ids) =
        const_str_array(module_const(&shard_fixture(&missing), "PACK_SHARD_IDS"));
    assert_ne!(
        missing_ids,
        EXPECTED_PACK_SHARD_IDS
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        "a pack without the weights.lm_head shard must be rejected"
    );

    let wrong_legacy_count: syn::File = syn::parse_file(
        r#"
        const LEGACY_PACK_SECTION_COUNT: u32 = 13;
        "#,
    )
    .expect("legacy-count fixture must parse");
    assert_ne!(
        const_u32_literal(module_const(
            &wrong_legacy_count,
            "LEGACY_PACK_SECTION_COUNT"
        )),
        "12",
        "a drifted legacy section count must not satisfy the dual-admission pin"
    );

    // vocab_size smuggled into the legacy 14-field descriptor shape: the
    // capability field must exist only as the fifteenth field on top of the
    // intact accepted fourteen.
    let legacy_with_vocab: syn::File = syn::parse_file(
        r#"
        const DESCRIPTOR_FIELDS: [&str; 14] = [
            "schema_version",
            "hidden_size",
            "intermediate_size",
            "query_heads",
            "key_value_heads",
            "head_dim",
            "query_width",
            "key_value_width",
            "prefix_tokens",
            "cache_capacity",
            "mrope_sections",
            "rms_norm_epsilon",
            "layers",
            "vocab_size",
        ];
        "#,
    )
    .expect("legacy-descriptor fixture must parse");
    let (_, legacy_fields) = const_str_array(module_const(&legacy_with_vocab, "DESCRIPTOR_FIELDS"));
    assert_ne!(
        legacy_fields,
        EXPECTED_DESCRIPTOR_FIELDS[..14]
            .iter()
            .map(|field| field.to_string())
            .collect::<Vec<_>>(),
        "a legacy descriptor whose fourteenth field slot carries vocab_size (dropping an accepted field) must be rejected: the accepted fourteen stay intact"
    );

    let descriptor_fixture = |fields: &str| -> syn::File {
        syn::parse_file(&format!(
            "const DESCRIPTOR_FIELDS: [&str; 15] = [{fields}];"
        ))
        .expect("descriptor fixture must parse")
    };
    let without_vocab = EXPECTED_DESCRIPTOR_FIELDS[..14]
        .iter()
        .map(|field| format!("\"{field}\""))
        .chain(["\"vocabulary_size\"".to_owned()])
        .collect::<Vec<_>>()
        .join(", ");
    let (_, decoy_fields) = const_str_array(module_const(
        &descriptor_fixture(&without_vocab),
        "DESCRIPTOR_FIELDS",
    ));
    assert!(
        !decoy_fields.iter().any(|field| field == "vocab_size"),
        "a misspelled vocabulary field must not satisfy the descriptor pin"
    );

    let wrong_vocab: syn::File = syn::parse_file(
        r#"
        fn validate(vocab_size: u32) -> bool {
            vocab_size == 100_000
        }
        "#,
    )
    .expect("wrong-vocab fixture must parse");
    let mut vocab = PinnedVocabLiteral(false);
    vocab.visit_file(&wrong_vocab);
    assert!(
        !vocab.0,
        "a drifted vocabulary literal must not satisfy the pinned 103424 pin"
    );
    let right_vocab: syn::File = syn::parse_file(
        r#"
        fn validate(vocab_size: u32) -> bool {
            vocab_size == 103_424
        }
        "#,
    )
    .expect("right-vocab fixture must parse");
    let mut vocab = PinnedVocabLiteral(false);
    vocab.visit_file(&right_vocab);
    assert!(vocab.0);

    let wrong_discipline: syn::File = syn::parse_file(
        r#"
        async fn run_logits() {
            encode_stack_pass()?;
            submit_stack_encoder()?;
            map_stack_buffer().await?;
        }
        "#,
    )
    .expect("wrong-discipline fixture must parse");
    let Item::Fn(wrong_executor) = &wrong_discipline.items[0] else {
        panic!("wrong-discipline fixture is not a function");
    };
    assert_eq!(
        count_free_calls(&wrong_executor.block, "encode_stack_pass"),
        1,
        "a single-pass logits forgery must not satisfy the two-pass pin"
    );
    assert_eq!(
        count_free_calls(&wrong_executor.block, "encode_stack_copy"),
        0,
        "a logits forgery without the readback copy must be detected"
    );
    let writing_discipline: syn::File = syn::parse_file(
        r#"
        async fn run_logits() {
            encode_stack_pass()?;
            encode_stack_pass()?;
            write_stack_buffer()?;
            encode_stack_copy()?;
            submit_stack_encoder()?;
            map_stack_buffer().await?;
        }
        "#,
    )
    .expect("writing-discipline fixture must parse");
    let Item::Fn(writing_executor) = &writing_discipline.items[0] else {
        panic!("writing-discipline fixture is not a function");
    };
    assert_eq!(
        count_free_calls(&writing_executor.block, "write_stack_buffer"),
        1,
        "a per-call uniform write must be detected: both logits stage uniforms are static"
    );
}
