//! Shared structural helpers for the sealed decoder session module contracts.
//!
//! These helpers are included with `#[path]` from each decoder session module
//! contract test target so the sealed-module, authority, field-algebra, and
//! planner-dataflow rules stay one source of truth.
#![allow(dead_code)]

use std::{collections::BTreeSet, fs};

use syn::{
    Attribute, Expr, ExprMethodCall, Field, File, FnArg, GenericArgument, ImplItem, ImplItemFn,
    Item, Pat, PathArguments, Stmt, Type, TypePath, UseTree, Visibility, visit::Visit,
};

pub(crate) fn workspace_path(relative: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

pub(crate) fn parse(relative: &str) -> File {
    let source = fs::read_to_string(workspace_path(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"));
    syn::parse_file(&source).unwrap_or_else(|error| panic!("parse {relative}: {error}"))
}

pub(crate) fn inherited(visibility: &Visibility) -> bool {
    matches!(visibility, Visibility::Inherited)
}

pub(crate) fn restricted_to_parent(visibility: &Visibility) -> bool {
    matches!(
        visibility,
        Visibility::Restricted(restricted)
            if restricted.in_token.is_none() && restricted.path.is_ident("super")
    )
}

pub(crate) fn has_attribute(attributes: &[Attribute], name: &str) -> bool {
    attributes
        .iter()
        .any(|attribute| attribute.path().is_ident(name))
}

pub(crate) fn cfg_free(attributes: &[Attribute]) -> bool {
    !attributes
        .iter()
        .any(|attribute| attribute.path().is_ident("cfg") || attribute.path().is_ident("cfg_attr"))
}

pub(crate) fn only_doc_attributes(attributes: &[Attribute]) -> bool {
    attributes
        .iter()
        .all(|attribute| attribute.path().is_ident("doc"))
}

#[derive(Default)]
pub(crate) struct ForbiddenCausalSyntax {
    pub(crate) unsafe_blocks: usize,
    pub(crate) macro_invocations: usize,
    pub(crate) derive_attributes: usize,
    pub(crate) cfg_attributes: usize,
    pub(crate) type_aliases: usize,
    pub(crate) statics: usize,
    pub(crate) renamed_imports: usize,
    pub(crate) nested_modules: usize,
    pub(crate) implementation_blocks: usize,
}

impl<'ast> Visit<'ast> for ForbiddenCausalSyntax {
    fn visit_expr_unsafe(&mut self, expression: &'ast syn::ExprUnsafe) {
        self.unsafe_blocks += 1;
        syn::visit::visit_expr_unsafe(self, expression);
    }

    fn visit_macro(&mut self, invocation: &'ast syn::Macro) {
        self.macro_invocations += 1;
        syn::visit::visit_macro(self, invocation);
    }

    fn visit_attribute(&mut self, attribute: &'ast Attribute) {
        if attribute.path().is_ident("derive") {
            self.derive_attributes += 1;
        }
        if attribute.path().is_ident("cfg") || attribute.path().is_ident("cfg_attr") {
            self.cfg_attributes += 1;
        }
        syn::visit::visit_attribute(self, attribute);
    }

    fn visit_item_type(&mut self, item: &'ast syn::ItemType) {
        self.type_aliases += 1;
        syn::visit::visit_item_type(self, item);
    }

    fn visit_item_static(&mut self, item: &'ast syn::ItemStatic) {
        self.statics += 1;
        syn::visit::visit_item_static(self, item);
    }

    fn visit_use_rename(&mut self, rename: &'ast syn::UseRename) {
        self.renamed_imports += 1;
        syn::visit::visit_use_rename(self, rename);
    }

    fn visit_item_mod(&mut self, module: &'ast syn::ItemMod) {
        self.nested_modules += 1;
        syn::visit::visit_item_mod(self, module);
    }

    fn visit_item_impl(&mut self, implementation: &'ast syn::ItemImpl) {
        self.implementation_blocks += 1;
        syn::visit::visit_item_impl(self, implementation);
    }
}

#[derive(Default)]
pub(crate) struct TypeNames(BTreeSet<String>);

impl<'ast> Visit<'ast> for TypeNames {
    fn visit_type_path(&mut self, path: &'ast TypePath) {
        for segment in &path.path.segments {
            self.0.insert(segment.ident.to_string());
        }
        syn::visit::visit_type_path(self, path);
    }
}

pub(crate) fn type_names(field: &Field) -> BTreeSet<String> {
    type_names_in_type(&field.ty)
}

pub(crate) fn type_names_in_type(ty: &Type) -> BTreeSet<String> {
    let mut names = TypeNames::default();
    names.visit_type(ty);
    names.0
}

pub(crate) fn terminal_type_name(ty: &Type) -> Option<String> {
    let Type::Path(path) = ty else {
        return None;
    };
    path.path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}

pub(crate) fn exact_async_session_owner(ty: &Type, session_type: &str) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    if path.qself.is_some()
        || path.path.leading_colon.is_some()
        || path.path.segments.len() != 2
        || path.path.segments[0].ident != "crate"
        || !matches!(path.path.segments[0].arguments, PathArguments::None)
    {
        return false;
    }
    let owner = &path.path.segments[1];
    if owner.ident != "AsyncSessionOwner" {
        return false;
    }
    let PathArguments::AngleBracketed(arguments) = &owner.arguments else {
        return false;
    };
    if arguments.args.len() != 1 {
        return false;
    }
    let Some(GenericArgument::Type(Type::Path(session))) = arguments.args.first() else {
        return false;
    };
    session.qself.is_none()
        && session.path.leading_colon.is_none()
        && session.path.segments.len() == 1
        && session.path.segments[0].ident == session_type
        && matches!(session.path.segments[0].arguments, PathArguments::None)
}

pub(crate) fn exact_wgpu_type(ty: &Type, expected: &str) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    if path.qself.is_some()
        || path.path.leading_colon.is_some()
        || path.path.segments.len() != 2
        || path.path.segments[0].ident != "wgpu"
        || !matches!(path.path.segments[0].arguments, PathArguments::None)
    {
        return false;
    }
    let terminal = &path.path.segments[1];
    terminal.ident == expected && matches!(terminal.arguments, PathArguments::None)
}

pub(crate) fn exact_core_plan(ty: &Type, expected: &str) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    if path.qself.is_some()
        || path.path.leading_colon.is_some()
        || path.path.segments.len() != 2
        || path.path.segments[0].ident != "pvlc_runtime_core"
        || !matches!(path.path.segments[0].arguments, PathArguments::None)
    {
        return false;
    }
    let terminal = &path.path.segments[1];
    terminal.ident == expected && matches!(terminal.arguments, PathArguments::None)
}

/// The exact `Option<pvlc_runtime_core::...>` capability-plan shape: the M6e8
/// logits extension stores its LM-head plan as an optional field so legacy
/// (12-section pack) sessions keep the accepted M6e7 persistent topology.
pub(crate) fn exact_optional_core_plan(ty: &Type, expected: &str) -> bool {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SessionFieldClass {
    WgpuWebGpuBuffer,
    JsObjectHandle,
    CoreKvPlan,
    CoreAttentionPlan,
    Scalar,
    Digest,
    CoreLayerPlan,
    CoreStackPlan,
}

pub(crate) fn exact_wgpu_webgpu_buffer(ty: &Type) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    if path.qself.is_some()
        || path.path.leading_colon.is_some()
        || path.path.segments.len() != 3
        || path.path.segments[0].ident != "wgpu"
        || path.path.segments[1].ident != "webgpu"
        || !matches!(path.path.segments[0].arguments, PathArguments::None)
        || !matches!(path.path.segments[1].arguments, PathArguments::None)
    {
        return false;
    }
    let terminal = &path.path.segments[2];
    terminal.ident == "GpuBuffer" && matches!(terminal.arguments, PathArguments::None)
}

pub(crate) fn exact_js_sys_object(ty: &Type) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    if path.qself.is_some()
        || path.path.leading_colon.is_some()
        || path.path.segments.len() != 2
        || path.path.segments[0].ident != "js_sys"
        || !matches!(path.path.segments[0].arguments, PathArguments::None)
    {
        return false;
    }
    let terminal = &path.path.segments[1];
    terminal.ident == "Object" && matches!(terminal.arguments, PathArguments::None)
}

fn is_digest_array(ty: &Type) -> bool {
    let Type::Array(array) = ty else {
        return false;
    };
    let is_u8_element = matches!(
        array.elem.as_ref(),
        Type::Path(path)
            if path.qself.is_none()
                && path.path.leading_colon.is_none()
                && path.path.segments.len() == 1
                && path.path.segments[0].ident == "u8"
                && matches!(path.path.segments[0].arguments, PathArguments::None)
    );
    let is_digest_len = matches!(
        &array.len,
        Expr::Lit(lit)
            if matches!(&lit.lit, syn::Lit::Int(value) if value.base10_digits() == "32")
    );
    is_u8_element && is_digest_len
}

/// Persistent class of an Option-wrapped accepted session resource: the M6e7
/// prefill extension stores its prefill-only buffers, pipelines, bind groups,
/// and digests as optional fields so a decode-only session keeps the accepted
/// M6e6 persistent topology byte for byte. Only the exact browser handle and
/// digest shapes are seen through the Option; exact core plans and scalars
/// stay non-optional, and `Option` of anything else remains outside the
/// algebra.
pub(crate) fn optional_session_field_class(ty: &Type) -> Option<SessionFieldClass> {
    let Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some()
        || path.path.leading_colon.is_some()
        || path.path.segments.len() != 1
        || path.path.segments[0].ident != "Option"
    {
        return None;
    }
    let PathArguments::AngleBracketed(arguments) = &path.path.segments[0].arguments else {
        return None;
    };
    if arguments.args.len() != 1 {
        return None;
    }
    let Some(GenericArgument::Type(inner)) = arguments.args.first() else {
        return None;
    };
    if exact_wgpu_webgpu_buffer(inner) {
        return Some(SessionFieldClass::WgpuWebGpuBuffer);
    }
    if exact_js_sys_object(inner) {
        return Some(SessionFieldClass::JsObjectHandle);
    }
    if is_digest_array(inner) {
        return Some(SessionFieldClass::Digest);
    }
    None
}

/// Positive closed persistent-field algebra for a sealed browser session: only
/// exact browser handles, exact core plans, plain scalars, and fixed digest
/// arrays. Any host-side storage, indirection, or custom wrapper type is
/// outside the algebra and therefore rejected structurally.
pub(crate) fn classify_session_field(ty: &Type) -> Option<SessionFieldClass> {
    if exact_wgpu_webgpu_buffer(ty) {
        return Some(SessionFieldClass::WgpuWebGpuBuffer);
    }
    if exact_js_sys_object(ty) {
        return Some(SessionFieldClass::JsObjectHandle);
    }
    if exact_core_plan(ty, "DecoderKvSessionPlan") {
        return Some(SessionFieldClass::CoreKvPlan);
    }
    if exact_core_plan(ty, "DecoderAttentionBlockPlan") {
        return Some(SessionFieldClass::CoreAttentionPlan);
    }
    if exact_core_plan(ty, "DecoderLayerPlan") {
        return Some(SessionFieldClass::CoreLayerPlan);
    }
    if exact_core_plan(ty, "DecoderStackPlan") {
        return Some(SessionFieldClass::CoreStackPlan);
    }
    if let Type::Path(path) = ty
        && path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == 1
        && matches!(
            path.path.segments[0].ident.to_string().as_str(),
            "bool" | "u32" | "u64" | "usize"
        )
        && matches!(path.path.segments[0].arguments, PathArguments::None)
    {
        return Some(SessionFieldClass::Scalar);
    }
    if is_digest_array(ty) {
        return Some(SessionFieldClass::Digest);
    }
    optional_session_field_class(ty)
}

#[derive(Default)]
pub(crate) struct NamedTypeDeclarations(Vec<String>);

impl<'ast> Visit<'ast> for NamedTypeDeclarations {
    fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
        self.0.push(item.ident.to_string());
        syn::visit::visit_item_struct(self, item);
    }

    fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
        self.0.push(item.ident.to_string());
        syn::visit::visit_item_enum(self, item);
    }
}

pub(crate) fn named_type_declarations(root: &File) -> Vec<String> {
    let mut collector = NamedTypeDeclarations::default();
    collector.visit_file(root);
    collector.0
}

#[derive(Default)]
pub(crate) struct AliasTypes(pub(crate) Vec<Type>);

impl<'ast> Visit<'ast> for AliasTypes {
    fn visit_item_type(&mut self, item: &'ast syn::ItemType) {
        self.0.push(item.ty.as_ref().clone());
        syn::visit::visit_item_type(self, item);
    }
}

pub(crate) fn collect_source_files(
    directory: &std::path::Path,
    output: &mut Vec<std::path::PathBuf>,
) {
    let mut entries = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("read dir {}: {error}", directory.display()))
        .map(|entry| {
            entry
                .unwrap_or_else(|error| panic!("read entry in {}: {error}", directory.display()))
                .path()
        })
        .collect::<Vec<_>>();
    entries.sort();
    for entry in entries {
        if entry.is_dir() {
            collect_source_files(&entry, output);
        } else if entry.extension().is_some_and(|extension| extension == "rs") {
            output.push(entry);
        }
    }
}

pub(crate) struct MethodCallCount<'a> {
    name: &'a str,
    calls: usize,
}

impl<'ast> Visit<'ast> for MethodCallCount<'_> {
    fn visit_expr_method_call(&mut self, expression: &'ast ExprMethodCall) {
        if expression.method == self.name {
            self.calls += 1;
        }
        syn::visit::visit_expr_method_call(self, expression);
    }
}

pub(crate) fn count_method_calls(block: &syn::Block, name: &str) -> usize {
    let mut counter = MethodCallCount { name, calls: 0 };
    counter.visit_block(block);
    counter.calls
}

#[derive(Default)]
pub(crate) struct UsedIdents(BTreeSet<String>);

impl<'ast> Visit<'ast> for UsedIdents {
    fn visit_expr_path(&mut self, expression: &'ast syn::ExprPath) {
        if expression.qself.is_none()
            && expression.path.leading_colon.is_none()
            && expression.path.segments.len() == 1
        {
            self.0.insert(expression.path.segments[0].ident.to_string());
        }
        syn::visit::visit_expr_path(self, expression);
    }
}

pub(crate) fn used_idents_in_expr(expression: &Expr) -> BTreeSet<String> {
    let mut collector = UsedIdents::default();
    collector.visit_expr(expression);
    collector.0
}

#[derive(Default)]
pub(crate) struct PatternIdents(BTreeSet<String>);

impl<'ast> Visit<'ast> for PatternIdents {
    fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
        self.0.insert(pattern.ident.to_string());
        syn::visit::visit_pat_ident(self, pattern);
    }
}

pub(crate) struct LetBinding {
    names: BTreeSet<String>,
    init_idents: BTreeSet<String>,
    init_has_planner_call: bool,
}

/// Counts method calls that live at the expression level of the visited code,
/// skipping closure bodies and async blocks: a `let prepared = (|| { ...
/// plan() ... })();` init must not count the closure's nested planner calls as
/// its own binding, while `let plan = descriptor.plan().map_err(...)?;` must.
pub(crate) struct DirectMethodCallCount<'a> {
    name: &'a str,
    calls: usize,
}

impl<'ast> Visit<'ast> for DirectMethodCallCount<'_> {
    fn visit_expr_method_call(&mut self, expression: &'ast ExprMethodCall) {
        if expression.method == self.name {
            self.calls += 1;
        }
        syn::visit::visit_expr_method_call(self, expression);
    }

    fn visit_expr_closure(&mut self, _expression: &'ast syn::ExprClosure) {}

    fn visit_expr_async(&mut self, _expression: &'ast syn::ExprAsync) {}
}

pub(crate) struct LetBindings<'a> {
    bindings: Vec<LetBinding>,
    planner_method: &'a str,
}

impl<'ast> Visit<'ast> for LetBindings<'_> {
    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Some(init) = &local.init {
            let mut names = PatternIdents::default();
            names.visit_pat(&local.pat);
            let init_has_planner_call = {
                let mut counter = DirectMethodCallCount {
                    name: self.planner_method,
                    calls: 0,
                };
                counter.visit_expr(&init.expr);
                counter.calls > 0
            };
            self.bindings.push(LetBinding {
                names: names.0,
                init_idents: used_idents_in_expr(&init.expr),
                init_has_planner_call,
            });
        }
        syn::visit::visit_local(self, local);
    }
}

pub(crate) struct FlowSinks<'a> {
    owner_field: &'a str,
    owner_call_args: BTreeSet<String>,
    local_field_assignments: Vec<(String, BTreeSet<String>)>,
}

impl<'ast> Visit<'ast> for FlowSinks<'_> {
    fn visit_expr_method_call(&mut self, expression: &'ast ExprMethodCall) {
        let receiver_is_owner = matches!(
            expression.receiver.as_ref(),
            Expr::Field(field)
                if matches!(field.base.as_ref(), Expr::Path(base) if base.path.is_ident("self"))
                    && matches!(&field.member, syn::Member::Named(name) if name == self.owner_field)
        );
        if receiver_is_owner {
            for argument in &expression.args {
                self.owner_call_args.extend(used_idents_in_expr(argument));
            }
        }
        syn::visit::visit_expr_method_call(self, expression);
    }

    fn visit_expr_assign(&mut self, expression: &'ast syn::ExprAssign) {
        if let Expr::Field(field) = expression.left.as_ref()
            && let Expr::Path(base) = field.base.as_ref()
            && base.qself.is_none()
            && base.path.segments.len() == 1
        {
            self.local_field_assignments.push((
                base.path.segments[0].ident.to_string(),
                used_idents_in_expr(&expression.right),
            ));
        }
        syn::visit::visit_expr_assign(self, expression);
    }
}

/// Structural capability/dataflow proof that an exact core planner result is
/// not computed-then-ignored: every planner call must be bound by a `let`
/// whose binding is transitively consumed by the operation tail, a sealed-owner
/// call, or a live session field assignment. `let` bindings are collected at
/// every nesting depth so the accepted prepared-closure idiom (planner calls
/// inside one `let prepared = (|| { ... })();`) counts each nested binding.
pub(crate) fn planner_result_reaches_live_consumer(
    function: &ImplItemFn,
    planner_method: &str,
    expected_calls: usize,
    owner_field: &str,
) -> bool {
    if count_method_calls(&function.block, planner_method) != expected_calls {
        return false;
    }
    let mut collector = LetBindings {
        bindings: Vec::new(),
        planner_method,
    };
    collector.visit_block(&function.block);
    let bindings = collector.bindings;
    let plan_bindings = bindings
        .iter()
        .filter(|binding| binding.init_has_planner_call)
        .count();
    if plan_bindings != expected_calls {
        return false;
    }
    let mut reachable = BTreeSet::new();
    if let Some(Stmt::Expr(tail, None)) = function.block.stmts.last() {
        reachable.extend(used_idents_in_expr(tail));
    }
    let mut sinks = FlowSinks {
        owner_field,
        owner_call_args: BTreeSet::new(),
        local_field_assignments: Vec::new(),
    };
    sinks.visit_block(&function.block);
    reachable.extend(sinks.owner_call_args.iter().cloned());
    loop {
        let mut grew = false;
        for binding in &bindings {
            if binding.names.iter().any(|name| reachable.contains(name)) {
                for ident in &binding.init_idents {
                    grew |= reachable.insert(ident.clone());
                }
            }
        }
        for (base, right_idents) in &sinks.local_field_assignments {
            if reachable.contains(base) {
                for ident in right_idents {
                    grew |= reachable.insert(ident.clone());
                }
            }
        }
        if !grew {
            break;
        }
    }
    bindings
        .iter()
        .filter(|binding| binding.init_has_planner_call)
        .all(|binding| {
            binding.names.len() == 1
                && binding
                    .names
                    .iter()
                    .next()
                    .is_some_and(|name| reachable.contains(name))
        })
}

#[derive(Default)]
pub(crate) struct PlanStructConstructions(usize);

impl<'ast> Visit<'ast> for PlanStructConstructions {
    fn visit_expr_struct(&mut self, expression: &'ast syn::ExprStruct) {
        if expression.path.segments.last().is_some_and(|segment| {
            segment.ident == "DecoderKvSessionPlan"
                || segment.ident == "DecoderKvSessionStepPlan"
                || segment.ident == "DecoderAttentionBlockPlan"
                || segment.ident == "DecoderAttentionBlockStepPlan"
                || segment.ident == "DecoderLayerPlan"
                || segment.ident == "DecoderLayerStepPlan"
                || segment.ident == "DecoderStackPlan"
        }) {
            self.0 += 1;
        }
        syn::visit::visit_expr_struct(self, expression);
    }
}

pub(crate) fn plan_struct_constructions(root: &File) -> usize {
    let mut collector = PlanStructConstructions::default();
    collector.visit_file(root);
    collector.0
}

pub(crate) fn item_visibility(item: &Item) -> Option<&Visibility> {
    match item {
        Item::Const(value) => Some(&value.vis),
        Item::Enum(value) => Some(&value.vis),
        Item::ExternCrate(value) => Some(&value.vis),
        Item::Fn(value) => Some(&value.vis),
        Item::ForeignMod(_) => None,
        Item::Impl(_) => None,
        Item::Macro(_) => None,
        Item::Mod(value) => Some(&value.vis),
        Item::Static(value) => Some(&value.vis),
        Item::Struct(value) => Some(&value.vis),
        Item::Trait(value) => Some(&value.vis),
        Item::TraitAlias(value) => Some(&value.vis),
        Item::Type(value) => Some(&value.vis),
        Item::Union(value) => Some(&value.vis),
        Item::Use(value) => Some(&value.vis),
        _ => None,
    }
}

pub(crate) fn outer_method_call(expression: &Expr) -> Option<&ExprMethodCall> {
    match expression {
        Expr::Await(value) => outer_method_call(&value.base),
        Expr::Group(value) => outer_method_call(&value.expr),
        Expr::Paren(value) => outer_method_call(&value.expr),
        Expr::Return(value) => value.expr.as_deref().and_then(outer_method_call),
        Expr::Try(value) => outer_method_call(&value.expr),
        Expr::MethodCall(call) => Some(call),
        _ => None,
    }
}

pub(crate) fn forwarded_parameter_names(function: &ImplItemFn) -> Option<Vec<String>> {
    function
        .sig
        .inputs
        .iter()
        .filter_map(|argument| match argument {
            FnArg::Receiver(_) => None,
            FnArg::Typed(argument) => Some(argument),
        })
        .map(|argument| {
            let Pat::Ident(pattern) = argument.pat.as_ref() else {
                return None;
            };
            if pattern.by_ref.is_some() || pattern.mutability.is_some() || pattern.subpat.is_some()
            {
                return None;
            }
            Some(pattern.ident.to_string())
        })
        .collect()
}

pub(crate) fn exact_authority_call(
    function: &ImplItemFn,
    expression: &Expr,
    expected_method: &str,
    authority_field: &str,
) -> bool {
    let Some(call) = outer_method_call(expression) else {
        return false;
    };
    if call.method != expected_method {
        return false;
    }
    let Expr::Field(field) = call.receiver.as_ref() else {
        return false;
    };
    let Expr::Path(receiver) = field.base.as_ref() else {
        return false;
    };
    if !receiver.path.is_ident("self")
        || !matches!(
            &field.member,
            syn::Member::Named(name) if name == authority_field
        )
    {
        return false;
    }
    let Some(parameters) = forwarded_parameter_names(function) else {
        return false;
    };
    call.args.len() == parameters.len()
        && call
            .args
            .iter()
            .zip(parameters)
            .all(|(argument, parameter)| {
                matches!(
                    argument,
                    Expr::Path(path)
                        if path.qself.is_none() && path.path.is_ident(&parameter)
                )
            })
}

pub(crate) fn assert_direct_authority_call(
    function: &ImplItemFn,
    expected_method: &str,
    authority_field: &str,
) {
    assert_eq!(
        function.block.stmts.len(),
        1,
        "{} must contain one direct sealed-authority delegation",
        function.sig.ident
    );
    let Stmt::Expr(expression, None) = &function.block.stmts[0] else {
        panic!(
            "{} must return one direct sealed-authority delegation",
            function.sig.ident
        );
    };
    assert!(
        exact_authority_call(function, expression, expected_method, authority_field),
        "{} must directly forward every argument to self.{authority_field}.{expected_method}",
        function.sig.ident
    );
}

pub(crate) struct MatchingImpls<'ast> {
    type_name: String,
    implementations: Vec<&'ast syn::ItemImpl>,
}

impl<'ast> Visit<'ast> for MatchingImpls<'ast> {
    fn visit_item_impl(&mut self, implementation: &'ast syn::ItemImpl) {
        if terminal_type_name(implementation.self_ty.as_ref()).as_deref()
            == Some(self.type_name.as_str())
        {
            self.implementations.push(implementation);
        }
        syn::visit::visit_item_impl(self, implementation);
    }
}

pub(crate) fn matching_impls<'ast>(root: &'ast File, type_name: &str) -> Vec<&'ast syn::ItemImpl> {
    let mut matches = MatchingImpls {
        type_name: type_name.to_owned(),
        implementations: Vec::new(),
    };
    matches.visit_file(root);
    matches.implementations
}

pub(crate) fn public_inherent_methods(root: &File, type_name: &str) -> BTreeSet<String> {
    matching_impls(root, type_name)
        .into_iter()
        .filter(|implementation| implementation.trait_.is_none())
        .flat_map(|implementation| implementation.items.iter())
        .filter_map(|item| match item {
            ImplItem::Fn(function) if matches!(function.vis, Visibility::Public(_)) => {
                Some(function.sig.ident.to_string())
            }
            _ => None,
        })
        .collect()
}

pub(crate) fn public_web_surface(root: &File) -> BTreeSet<String> {
    root.items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(value) if matches!(value.vis, Visibility::Public(_)) => {
                Some(format!("fn:{}", value.sig.ident))
            }
            Item::Struct(value) if matches!(value.vis, Visibility::Public(_)) => {
                Some(format!("struct:{}", value.ident))
            }
            Item::Enum(value) if matches!(value.vis, Visibility::Public(_)) => {
                Some(format!("enum:{}", value.ident))
            }
            Item::Trait(value) if matches!(value.vis, Visibility::Public(_)) => {
                Some(format!("trait:{}", value.ident))
            }
            Item::Type(value) if matches!(value.vis, Visibility::Public(_)) => {
                Some(format!("type:{}", value.ident))
            }
            Item::Const(value) if matches!(value.vis, Visibility::Public(_)) => {
                Some(format!("const:{}", value.ident))
            }
            Item::Static(value) if matches!(value.vis, Visibility::Public(_)) => {
                Some(format!("static:{}", value.ident))
            }
            Item::Mod(value) if matches!(value.vis, Visibility::Public(_)) => {
                Some(format!("mod:{}", value.ident))
            }
            Item::Use(value) if matches!(value.vis, Visibility::Public(_)) => {
                Some("use".to_owned())
            }
            _ => None,
        })
        .collect()
}

pub(crate) fn collect_use_paths(tree: &UseTree, prefix: &str, output: &mut BTreeSet<String>) {
    match tree {
        UseTree::Path(path) => {
            let next = if prefix.is_empty() {
                path.ident.to_string()
            } else {
                format!("{prefix}::{}", path.ident)
            };
            collect_use_paths(&path.tree, &next, output);
        }
        UseTree::Name(name) => {
            output.insert(format!("{prefix}::{}", name.ident));
        }
        UseTree::Rename(rename) => {
            output.insert(format!("{prefix}::{} as {}", rename.ident, rename.rename));
        }
        UseTree::Glob(_) => {
            output.insert(format!("{prefix}::*"));
        }
        UseTree::Group(group) => {
            for item in &group.items {
                collect_use_paths(item, prefix, output);
            }
        }
    }
}

pub(crate) fn parse_impl_method(source: &str, method: &str) -> ImplItemFn {
    let file = syn::parse_file(source).expect("planner fixture must parse");
    let Item::Impl(implementation) = &file.items[0] else {
        panic!("planner fixture is not an impl");
    };
    implementation
        .items
        .iter()
        .find_map(|item| match item {
            ImplItem::Fn(function) if function.sig.ident == method => Some(function.clone()),
            _ => None,
        })
        .expect("planner fixture method is missing")
}
