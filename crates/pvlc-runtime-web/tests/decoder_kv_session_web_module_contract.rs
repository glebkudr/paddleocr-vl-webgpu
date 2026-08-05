use std::{collections::BTreeSet, fs};

use syn::{
    Attribute, Expr, ExprMethodCall, Field, File, FnArg, GenericArgument, ImplItem, ImplItemFn,
    Item, Pat, PathArguments, Stmt, Type, TypePath, UseTree, Visibility, visit::Visit,
};

fn workspace_path(relative: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn parse(relative: &str) -> File {
    let source = fs::read_to_string(workspace_path(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"));
    syn::parse_file(&source).unwrap_or_else(|error| panic!("parse {relative}: {error}"))
}

fn inherited(visibility: &Visibility) -> bool {
    matches!(visibility, Visibility::Inherited)
}

fn restricted_to_parent(visibility: &Visibility) -> bool {
    matches!(
        visibility,
        Visibility::Restricted(restricted)
            if restricted.in_token.is_none() && restricted.path.is_ident("super")
    )
}

fn has_attribute(attributes: &[Attribute], name: &str) -> bool {
    attributes
        .iter()
        .any(|attribute| attribute.path().is_ident(name))
}

fn cfg_free(attributes: &[Attribute]) -> bool {
    !attributes
        .iter()
        .any(|attribute| attribute.path().is_ident("cfg") || attribute.path().is_ident("cfg_attr"))
}

fn only_doc_attributes(attributes: &[Attribute]) -> bool {
    attributes
        .iter()
        .all(|attribute| attribute.path().is_ident("doc"))
}

#[derive(Default)]
struct ForbiddenCausalSyntax {
    unsafe_blocks: usize,
    macro_invocations: usize,
    derive_attributes: usize,
    cfg_attributes: usize,
    type_aliases: usize,
    statics: usize,
    renamed_imports: usize,
    nested_modules: usize,
    implementation_blocks: usize,
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
struct TypeNames(BTreeSet<String>);

impl<'ast> Visit<'ast> for TypeNames {
    fn visit_type_path(&mut self, path: &'ast TypePath) {
        for segment in &path.path.segments {
            self.0.insert(segment.ident.to_string());
        }
        syn::visit::visit_type_path(self, path);
    }
}

fn type_names(field: &Field) -> BTreeSet<String> {
    type_names_in_type(&field.ty)
}

fn type_names_in_type(ty: &Type) -> BTreeSet<String> {
    let mut names = TypeNames::default();
    names.visit_type(ty);
    names.0
}

fn terminal_type_name(ty: &Type) -> Option<String> {
    let Type::Path(path) = ty else {
        return None;
    };
    path.path
        .segments
        .last()
        .map(|segment| segment.ident.to_string())
}

fn exact_async_session_owner(ty: &Type) -> bool {
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
        && session.path.segments[0].ident == "BrowserDecoderKvSession"
        && matches!(session.path.segments[0].arguments, PathArguments::None)
}

fn exact_wgpu_type(ty: &Type, expected: &str) -> bool {
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

fn exact_core_decoder_plan(ty: &Type) -> bool {
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
    terminal.ident == "DecoderKvSessionPlan" && matches!(terminal.arguments, PathArguments::None)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SessionFieldClass {
    WgpuWebGpuBuffer,
    JsObjectHandle,
    CoreDecoderPlan,
    Scalar,
    Digest,
}

fn exact_wgpu_webgpu_buffer(ty: &Type) -> bool {
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

fn exact_js_sys_object(ty: &Type) -> bool {
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

/// Positive closed persistent-field algebra for `BrowserDecoderKvSession`.
///
/// A persistent browser session may only own exact browser resource handles
/// (`wgpu::webgpu::GpuBuffer` values, opaque `js_sys::Object` pipeline and
/// bind-group handles), the exact core session plan, plain scalars, and fixed
/// digest arrays. Any host-side storage (`Vec`, `Box`, `String`, maps),
/// indirection, or custom wrapper type is outside the algebra and therefore
/// rejected structurally.
fn classify_session_field(ty: &Type) -> Option<SessionFieldClass> {
    if exact_wgpu_webgpu_buffer(ty) {
        return Some(SessionFieldClass::WgpuWebGpuBuffer);
    }
    if exact_js_sys_object(ty) {
        return Some(SessionFieldClass::JsObjectHandle);
    }
    if exact_core_decoder_plan(ty) {
        return Some(SessionFieldClass::CoreDecoderPlan);
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
    if let Type::Array(array) = ty {
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
        if is_u8_element && is_digest_len {
            return Some(SessionFieldClass::Digest);
        }
    }
    None
}

#[derive(Default)]
struct NamedTypeDeclarations(Vec<String>);

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

fn named_type_declarations(root: &File) -> Vec<String> {
    let mut collector = NamedTypeDeclarations::default();
    collector.visit_file(root);
    collector.0
}

#[derive(Default)]
struct AliasTypes(Vec<Type>);

impl<'ast> Visit<'ast> for AliasTypes {
    fn visit_item_type(&mut self, item: &'ast syn::ItemType) {
        self.0.push(item.ty.as_ref().clone());
        syn::visit::visit_item_type(self, item);
    }
}

fn collect_source_files(directory: &std::path::Path, output: &mut Vec<std::path::PathBuf>) {
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

struct MethodCallCount<'a> {
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

fn count_method_calls(block: &syn::Block, name: &str) -> usize {
    let mut counter = MethodCallCount { name, calls: 0 };
    counter.visit_block(block);
    counter.calls
}

#[derive(Default)]
struct UsedIdents(BTreeSet<String>);

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

fn used_idents_in_expr(expression: &Expr) -> BTreeSet<String> {
    let mut collector = UsedIdents::default();
    collector.visit_expr(expression);
    collector.0
}

#[derive(Default)]
struct PatternIdents(BTreeSet<String>);

impl<'ast> Visit<'ast> for PatternIdents {
    fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
        self.0.insert(pattern.ident.to_string());
        syn::visit::visit_pat_ident(self, pattern);
    }
}

struct LetBinding {
    names: BTreeSet<String>,
    init_idents: BTreeSet<String>,
    init_has_planner_call: bool,
}

struct FlowSinks<'a> {
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

/// Structural capability/dataflow proof that the exact core planner result is
/// not computed-then-ignored: the single planner call must be bound by a `let`
/// whose binding is transitively consumed by the operation tail, a sealed-owner
/// call, or a live session field assignment.
fn planner_result_reaches_live_consumer(
    function: &ImplItemFn,
    planner_method: &str,
    owner_field: &str,
) -> bool {
    if count_method_calls(&function.block, planner_method) != 1 {
        return false;
    }
    let mut bindings = Vec::new();
    for statement in &function.block.stmts {
        let Stmt::Local(local) = statement else {
            continue;
        };
        let Some(init) = &local.init else {
            continue;
        };
        let mut names = PatternIdents::default();
        names.visit_pat(&local.pat);
        let init_has_planner_call = {
            let mut counter = MethodCallCount {
                name: planner_method,
                calls: 0,
            };
            counter.visit_expr(&init.expr);
            counter.calls > 0
        };
        bindings.push(LetBinding {
            names: names.0,
            init_idents: used_idents_in_expr(&init.expr),
            init_has_planner_call,
        });
    }
    let plan_bindings = bindings
        .iter()
        .filter(|binding| binding.init_has_planner_call)
        .count();
    if plan_bindings != 1 {
        return false;
    }
    let plan_binding = bindings
        .iter()
        .find(|binding| binding.init_has_planner_call)
        .expect("exactly one plan binding exists");
    if plan_binding.names.len() != 1 {
        return false;
    }
    let plan_name = plan_binding
        .names
        .iter()
        .next()
        .expect("plan binding has one name");

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
    reachable.contains(plan_name)
}

#[derive(Default)]
struct PlanStructConstructions(usize);

impl<'ast> Visit<'ast> for PlanStructConstructions {
    fn visit_expr_struct(&mut self, expression: &'ast syn::ExprStruct) {
        if expression.path.segments.last().is_some_and(|segment| {
            segment.ident == "DecoderKvSessionPlan" || segment.ident == "DecoderKvSessionStepPlan"
        }) {
            self.0 += 1;
        }
        syn::visit::visit_expr_struct(self, expression);
    }
}

fn item_visibility(item: &Item) -> Option<&Visibility> {
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

fn outer_method_call(expression: &Expr) -> Option<&ExprMethodCall> {
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

fn forwarded_parameter_names(function: &ImplItemFn) -> Option<Vec<String>> {
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

fn exact_authority_call(function: &ImplItemFn, expression: &Expr, expected_method: &str) -> bool {
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
            syn::Member::Named(name) if name == "decoder_kv_session"
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

fn assert_direct_authority_call(function: &ImplItemFn, expected_method: &str) {
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
        exact_authority_call(function, expression, expected_method),
        "{} must directly forward every argument to self.decoder_kv_session.{expected_method}",
        function.sig.ident
    );
}

struct MatchingImpls<'ast> {
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

fn matching_impls<'ast>(root: &'ast File, type_name: &str) -> Vec<&'ast syn::ItemImpl> {
    let mut matches = MatchingImpls {
        type_name: type_name.to_owned(),
        implementations: Vec::new(),
    };
    matches.visit_file(root);
    matches.implementations
}

fn public_inherent_methods(root: &File, type_name: &str) -> BTreeSet<String> {
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

fn public_web_surface(root: &File) -> BTreeSet<String> {
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

#[test]
fn decoder_session_is_one_cfg_free_sealed_causal_module() {
    let root = parse("crates/pvlc-runtime-web/src/web.rs");
    let declarations = root
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Mod(module) if module.ident == "decoder_kv_session" => Some(module),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        declarations.len(),
        1,
        "Web runtime must declare decoder_kv_session exactly once"
    );
    let module_declaration = declarations[0];
    assert!(
        inherited(&module_declaration.vis)
            && module_declaration.content.is_none()
            && only_doc_attributes(&module_declaration.attrs),
        "decoder_kv_session must be one unconditional private out-of-line module"
    );

    let module = parse("crates/pvlc-runtime-web/src/web/decoder_kv_session.rs");
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
            Item::Struct(item) if item.ident == "DecoderKvSessionAuthority" => Some(item),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        authorities.len(),
        1,
        "sealed DecoderKvSessionAuthority must be declared exactly once"
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
            && exact_async_session_owner(&owner_field.ty),
        "sealed authority must privately and directly own exactly crate::AsyncSessionOwner<BrowserDecoderKvSession> last"
    );

    let sessions = module
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Struct(item) if item.ident == "BrowserDecoderKvSession" => Some(item),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sessions.len(),
        1,
        "private BrowserDecoderKvSession must be declared exactly once"
    );
    let session = sessions[0];
    assert!(
        inherited(&session.vis)
            && session.generics.params.is_empty()
            && only_doc_attributes(&session.attrs),
        "browser decoder session must be unconditional, private, and non-generic"
    );
    let mut field_class_counts = [0usize; 5];
    for field in &session.fields {
        assert!(
            inherited(&field.vis) && only_doc_attributes(&field.attrs),
            "browser decoder fields must all be private"
        );
        let Some(class) = classify_session_field(&field.ty) else {
            panic!(
                "browser decoder session field hides state outside the closed persistent algebra"
            )
        };
        field_class_counts[class as usize] += 1;
    }
    assert_eq!(
        field_class_counts[SessionFieldClass::WgpuWebGpuBuffer as usize],
        9,
        "browser decoder session must directly own its nine exact wgpu::webgpu::GpuBuffer resources"
    );
    assert_eq!(
        field_class_counts[SessionFieldClass::JsObjectHandle as usize],
        4,
        "browser decoder session must directly own its two pipeline and two bind-group handles"
    );
    assert_eq!(
        field_class_counts[SessionFieldClass::CoreDecoderPlan as usize],
        1,
        "browser decoder session must hold exactly one exact core DecoderKvSessionPlan"
    );
    assert!(
        field_class_counts[SessionFieldClass::Scalar as usize] >= 1,
        "browser decoder session must keep its cache position and poison state in plain scalars"
    );

    let authority_impls = matching_impls(&module, "DecoderKvSessionAuthority");
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
    let session_impls = matching_impls(&module, "BrowserDecoderKvSession");
    assert_eq!(
        session_impls.len(),
        1,
        "browser decoder session must have one inherent implementation"
    );
    let session_impl = session_impls[0];
    assert!(
        session_impl.trait_.is_none()
            && session_impl.generics.params.is_empty()
            && only_doc_attributes(&session_impl.attrs),
        "browser decoder session implementation must be unconditional and inherent"
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
        planner_result_reaches_live_consumer(begin, "plan", "owner"),
        "sealed begin must bind the exact descriptor.plan() result and feed it to the live session constructor"
    );
    assert!(
        planner_result_reaches_live_consumer(step, "plan_step", "owner"),
        "sealed step must bind the exact plan_step() result and feed it to the live step executor"
    );
    let override_plan_calls = count_method_calls(&begin_with_override.block, "plan");
    assert!(
        override_plan_calls == 0
            || (override_plan_calls == 1
                && planner_result_reaches_live_consumer(begin_with_override, "plan", "owner",)),
        "sealed begin_with_shader_override must not compute-then-ignore descriptor.plan()"
    );
    for item in &authority_impl.items {
        let ImplItem::Fn(function) = item else {
            continue;
        };
        let name = function.sig.ident.to_string();
        let plan_calls = count_method_calls(&function.block, "plan");
        let plan_step_calls = count_method_calls(&function.block, "plan_step");
        match name.as_str() {
            "begin" => {
                assert_eq!(plan_step_calls, 0, "begin replans a step outside step");
            }
            "begin_with_shader_override" => {
                assert_eq!(
                    plan_step_calls, 0,
                    "begin_with_shader_override replans a step outside step"
                );
            }
            "step" => {
                assert_eq!(plan_calls, 0, "step replans the session outside begin");
            }
            _ => {
                assert!(
                    plan_calls == 0 && plan_step_calls == 0,
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
                && count_method_calls(&function.block, "plan_step") == 0,
            "{} replans the session inside its executor instead of consuming the bound plan",
            function.sig.ident
        );
    }
    for item in &module.items {
        let Item::Fn(function) = item else {
            continue;
        };
        assert!(
            count_method_calls(&function.block, "plan") == 0
                && count_method_calls(&function.block, "plan_step") == 0,
            "{} recomputes decoder topology outside the sealed planner boundary",
            function.sig.ident
        );
    }
    let mut plan_constructions = PlanStructConstructions::default();
    plan_constructions.visit_file(&module);
    assert_eq!(
        plan_constructions.0, 0,
        "causal module must not hand-construct the exact core decoder plan types"
    );

    for item in &module.items {
        if matches!(
            item,
            Item::Struct(value)
                if value.ident == "DecoderKvSessionAuthority"
                    || value.ident == "BrowserDecoderKvSession"
        ) || matches!(
            item,
            Item::Impl(value)
                if matches!(
                    terminal_type_name(value.self_ty.as_ref()).as_deref(),
                    Some("DecoderKvSessionAuthority") | Some("BrowserDecoderKvSession")
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
fn web_runtime_has_one_closed_owner_and_direct_exact_wasm_delegations() {
    let root = parse("crates/pvlc-runtime-web/src/web.rs");
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
    let first_effect = parse("crates/pvlc-runtime-web/src/web/vision_stack_first_effect.rs");
    assert!(
        matching_impls(&first_effect, "WebRuntime").is_empty()
            && matching_impls(&first_effect, "DecoderKvSessionAuthority").is_empty(),
        "the existing first-effect module must not become a WebRuntime or decoder-authority side door"
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
    let foreign_authority_impls = matching_impls(&root, "DecoderKvSessionAuthority").len();
    assert_eq!(
        foreign_authority_impls, 0,
        "web.rs must not extend the sealed decoder authority"
    );
    for alias in root.items.iter().filter_map(|item| match item {
        Item::Type(alias) => Some(alias),
        _ => None,
    }) {
        assert!(
            !type_names_in_type(&alias.ty).contains("DecoderKvSessionAuthority"),
            "web.rs must not hide decoder authority behind a type alias"
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
        .filter(|field| type_names(field).contains("DecoderKvSessionAuthority"))
        .collect::<Vec<_>>();
    assert_eq!(
        owner_fields.len(),
        1,
        "WebRuntime must own exactly one DecoderKvSessionAuthority"
    );
    let owner_field = owner_fields[0];
    assert!(
        inherited(&owner_field.vis)
            && owner_field
                .ident
                .as_ref()
                .is_some_and(|name| name == "decoder_kv_session")
            && terminal_type_name(&owner_field.ty).as_deref() == Some("DecoderKvSessionAuthority")
            && only_doc_attributes(&owner_field.attrs),
        "WebRuntime decoder authority must be one unconditional private named field"
    );

    let expected_delegations = [
        ("abort_decoder_kv_session", "abort"),
        ("begin_decoder_kv_session", "begin"),
        (
            "begin_decoder_kv_session_with_shader_override",
            "begin_with_shader_override",
        ),
        (
            "decoder_kv_session_shader_sources_json",
            "shader_sources_json",
        ),
        ("finish_decoder_kv_session", "finish"),
        ("step_decoder_kv_session", "step"),
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
        assert_direct_authority_call(function, authority_method);
    }

    let exported_decoder_methods = wasm_impl
        .items
        .iter()
        .filter_map(|item| match item {
            ImplItem::Fn(function)
                if function
                    .sig
                    .ident
                    .to_string()
                    .contains("decoder_kv_session") =>
            {
                Some(function.sig.ident.to_string())
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        exported_decoder_methods,
        expected_delegations
            .into_iter()
            .map(|(name, _)| name.to_owned())
            .collect(),
        "decoder WASM operation allowlist drifted"
    );

    let crate_root = parse("crates/pvlc-runtime-web/src/lib.rs");
    let mut public_web_reexports = BTreeSet::new();
    fn collect_use_paths(tree: &UseTree, prefix: &str, output: &mut BTreeSet<String>) {
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
    let direct = syn::parse_str::<Type>("crate::AsyncSessionOwner<BrowserDecoderKvSession>")
        .expect("direct owner type must parse");
    let unqualified = syn::parse_str::<Type>("AsyncSessionOwner<BrowserDecoderKvSession>")
        .expect("unqualified owner type must parse");
    let decoy = syn::parse_str::<Type>("decoy::AsyncSessionOwner<BrowserDecoderKvSession>")
        .expect("decoy owner type must parse");
    let wrapped = syn::parse_str::<Type>("RefCell<AsyncSessionOwner<BrowserDecoderKvSession>>")
        .expect("wrapped owner type must parse");
    let tuple =
        syn::parse_str::<Type>("(AsyncSessionOwner<BrowserDecoderKvSession>, ExtraAuthority)")
            .expect("tuple owner type must parse");
    assert!(exact_async_session_owner(&direct));
    assert!(!exact_async_session_owner(&unqualified));
    assert!(!exact_async_session_owner(&decoy));
    assert!(!exact_async_session_owner(&wrapped));
    assert!(!exact_async_session_owner(&tuple));

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
            pub async fn step_decoder_kv_session(
                &self,
                query: Bytes,
                key: Bytes,
                value: Bytes,
            ) -> Result {
                self.decoder_kv_session.step(query, key, value).await
            }
            pub async fn swapped(
                &self,
                query: Bytes,
                key: Bytes,
                value: Bytes,
            ) -> Result {
                self.decoder_kv_session.step(query, value, key).await
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
    let Stmt::Expr(direct, None) = &methods[0].block.stmts[0] else {
        panic!("direct delegation fixture drifted");
    };
    let Stmt::Expr(swapped, None) = &methods[1].block.stmts[0] else {
        panic!("swapped delegation fixture drifted");
    };
    assert!(exact_authority_call(methods[0], direct, "step"));
    assert!(!exact_authority_call(methods[1], swapped, "step"));
}

fn parse_impl_method(source: &str, method: &str) -> ImplItemFn {
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

#[test]
fn planner_result_must_reach_a_live_consumer() {
    let live_begin = parse_impl_method(
        r#"
        impl DecoderKvSessionAuthority {
            async fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let plan = descriptor.plan()?;
                let session = BrowserDecoderKvSession::create(&self.device, &self.queue, plan)?;
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
        "owner"
    ));

    let live_step = parse_impl_method(
        r#"
        impl DecoderKvSessionAuthority {
            async fn step(&self, query: Bytes) -> Result<String, Error> {
                let (lease, mut session) = self.owner.acquire()?;
                let step_plan = session.plan.plan_step(session.cache_tokens, &query)?;
                session.cache_tokens = step_plan.cache_tokens_after;
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
        "owner"
    ));

    let ignored_result = parse_impl_method(
        r#"
        impl DecoderKvSessionAuthority {
            async fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let plan = descriptor.plan()?;
                Ok(String::new())
            }
        }
        "#,
        "begin",
    );
    assert!(!planner_result_reaches_live_consumer(
        &ignored_result,
        "plan",
        "owner"
    ));

    let dead_nested_call = parse_impl_method(
        r#"
        impl DecoderKvSessionAuthority {
            async fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                if false {
                    descriptor.plan();
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
        "owner"
    ));

    let duplicated_call = parse_impl_method(
        r#"
        impl DecoderKvSessionAuthority {
            async fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let plan = descriptor.plan()?;
                let second = descriptor.plan()?;
                let session = BrowserDecoderKvSession::create(&self.device, &self.queue, second)?;
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
        "owner"
    ));

    let dead_shadow = parse_impl_method(
        r#"
        impl DecoderKvSessionAuthority {
            async fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let plan = descriptor.plan()?;
                let shadow = plan;
                Ok(String::new())
            }
        }
        "#,
        "begin",
    );
    assert!(!planner_result_reaches_live_consumer(
        &dead_shadow,
        "plan",
        "owner"
    ));

    let dead_field_sink = parse_impl_method(
        r#"
        impl DecoderKvSessionAuthority {
            async fn begin(&self, descriptor: Descriptor) -> Result<String, Error> {
                let plan = descriptor.plan()?;
                let mut decoy = Decoy::new();
                decoy.plan = plan;
                Ok(String::new())
            }
        }
        "#,
        "begin",
    );
    assert!(!planner_result_reaches_live_consumer(
        &dead_field_sink,
        "plan",
        "owner"
    ));

    let forged_plan = syn::parse_file(
        r#"
        fn begin(descriptor: &Descriptor) -> Plan {
            let _ = descriptor;
            pvlc_runtime_core::DecoderKvSessionPlan {
                initial_cache_tokens: 1,
                cache_capacity: 2,
                query_heads: 16,
                key_value_heads: 2,
                head_dim: 128,
                query_elements: 2048,
                key_value_width: 256,
                cache_elements: 512,
                cache_bytes: 2048,
                attention_bytes: 8192,
                append_invocation: invocation(),
                attention_invocation: invocation(),
            }
        }
        "#,
    )
    .expect("forged-plan fixture must parse");
    let mut constructions = PlanStructConstructions::default();
    constructions.visit_file(&forged_plan);
    assert_eq!(constructions.0, 1);
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
            SessionFieldClass::CoreDecoderPlan,
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
        "crate::AsyncSessionOwner<BrowserDecoderKvSession>",
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
        "pvlc_runtime_core::DecoderKvSessionStepPlan",
        "crate::DecoderKvSessionPlan",
        "super::DecoderKvSessionPlan",
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
        struct BrowserDecoderKvSession {
            cache_tokens: u32,
            poisoned: bool,
            ready: bool,
            plan: pvlc_runtime_core::DecoderKvSessionPlan,
            query_buffer: wgpu::webgpu::GpuBuffer,
            appended_key_buffer: wgpu::webgpu::GpuBuffer,
            appended_value_buffer: wgpu::webgpu::GpuBuffer,
            key_cache_buffer: wgpu::webgpu::GpuBuffer,
            value_cache_buffer: wgpu::webgpu::GpuBuffer,
            attention_output_buffer: wgpu::webgpu::GpuBuffer,
            append_uniform_buffer: wgpu::webgpu::GpuBuffer,
            attention_uniform_buffer: wgpu::webgpu::GpuBuffer,
            attention_readback_buffer: wgpu::webgpu::GpuBuffer,
            append_pipeline: js_sys::Object,
            attention_pipeline: js_sys::Object,
            append_bind_group: js_sys::Object,
            attention_bind_group: js_sys::Object,
        }
        "#,
    )
    .expect("canonical session fixture must parse");
    let mut counts = [0usize; 5];
    for field in &session.fields {
        let class = classify_session_field(&field.ty).expect("canonical field must classify");
        counts[class as usize] += 1;
    }
    assert_eq!(counts[SessionFieldClass::WgpuWebGpuBuffer as usize], 9);
    assert_eq!(counts[SessionFieldClass::JsObjectHandle as usize], 4);
    assert_eq!(counts[SessionFieldClass::CoreDecoderPlan as usize], 1);
    assert_eq!(counts[SessionFieldClass::Scalar as usize], 3);

    let shadowed = syn::parse_str::<syn::ItemStruct>(
        r#"
        struct BrowserDecoderKvSession {
            cache_tokens: u32,
            plan: pvlc_runtime_core::DecoderKvSessionPlan,
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
fn decoder_authority_has_no_crate_wide_side_doors() {
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
                .join("decoder_kv_session.rs")
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
            for sealed in ["DecoderKvSessionAuthority", "BrowserDecoderKvSession"] {
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
                !names.contains("DecoderKvSessionAuthority")
                    && !names.contains("BrowserDecoderKvSession"),
                "{} hides the sealed decoder session behind a type alias",
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
