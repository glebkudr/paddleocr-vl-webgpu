//! Thin, structural binding contract between the executable streaming
//! authority and the browser WebGPU adapter.
//!
//! The causal unit tests own ordering, cardinality, cache reuse, and failure
//! semantics. This file proves that the exported wasm method supplies the real
//! stored session, authenticated shard inputs, persistent cache, and exact
//! WebGPU effect closures to that authority.

use std::collections::{BTreeMap, BTreeSet};

use syn::{
    Expr, ExprCall, ExprClosure, FnArg, ImplItem, Item, ItemImpl, Pat, ReturnType, Stmt, Type,
    Visibility, visit::Visit,
};

const WEB_RUNTIME: &str = include_str!("../src/web.rs");

fn parsed_web_module() -> syn::File {
    syn::parse_file(WEB_RUNTIME).expect("web runtime source must parse")
}

#[derive(Clone, Copy)]
struct RuntimeMethod<'a> {
    parent: &'a ItemImpl,
    method: &'a syn::ImplItemFn,
}

fn runtime_methods(module: &syn::File) -> BTreeMap<String, RuntimeMethod<'_>> {
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
                Type::Path(path)
                    if path.path.segments.last().is_some_and(
                        |segment| segment.ident == "WebRuntime"
                    )
            )
        })
        .flat_map(|parent| {
            parent.items.iter().filter_map(move |item| match item {
                ImplItem::Fn(method) => Some((
                    method.sig.ident.to_string(),
                    RuntimeMethod { parent, method },
                )),
                _ => None,
            })
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Callable<'a> {
    Method(&'a syn::ImplItemFn),
    Function(&'a syn::ItemFn),
}

impl<'a> Callable<'a> {
    fn block(self) -> &'a syn::Block {
        match self {
            Self::Method(method) => &method.block,
            Self::Function(function) => &function.block,
        }
    }
}

fn callables<'a>(
    module: &'a syn::File,
    methods: &BTreeMap<String, RuntimeMethod<'a>>,
) -> BTreeMap<String, Callable<'a>> {
    let mut callables = methods
        .iter()
        .map(|(name, method)| (name.clone(), Callable::Method(method.method)))
        .collect::<BTreeMap<_, _>>();
    for item in &module.items {
        if let Item::Fn(function) = item {
            callables
                .entry(function.sig.ident.to_string())
                .or_insert(Callable::Function(function));
        }
    }
    callables
}

#[derive(Default)]
struct Calls {
    names: Vec<String>,
}

impl<'ast> Visit<'ast> for Calls {
    fn visit_expr(&mut self, expression: &'ast Expr) {
        match expression {
            Expr::Call(call) => {
                if let Expr::Path(path) = call.func.as_ref() {
                    if let Some(segment) = path.path.segments.last() {
                        self.names.push(segment.ident.to_string());
                    }
                }
            }
            Expr::MethodCall(call) => self.names.push(call.method.to_string()),
            _ => {}
        }
        syn::visit::visit_expr(self, expression);
    }
}

fn calls(block: &syn::Block) -> Calls {
    let mut calls = Calls::default();
    calls.visit_block(block);
    calls
}

struct FunctionCalls<'ast> {
    target: &'static str,
    found: Vec<&'ast ExprCall>,
}

impl<'ast> Visit<'ast> for FunctionCalls<'ast> {
    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        if matches!(
            call.func.as_ref(),
            Expr::Path(path)
                if path.path.segments.last().is_some_and(
                    |segment| segment.ident == self.target
                )
        ) {
            self.found.push(call);
        }
        syn::visit::visit_expr_call(self, call);
    }
}

struct MethodCalls<'ast> {
    target: &'static str,
    found: Vec<&'ast syn::ExprMethodCall>,
}

impl<'ast> Visit<'ast> for MethodCalls<'ast> {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == self.target {
            self.found.push(call);
        }
        syn::visit::visit_expr_method_call(self, call);
    }
}

fn one_function_call<'a>(block: &'a syn::Block, target: &'static str) -> &'a ExprCall {
    let mut calls = FunctionCalls {
        target,
        found: Vec::new(),
    };
    calls.visit_block(block);
    assert_eq!(calls.found.len(), 1, "expected exactly one `{target}` call",);
    calls.found[0]
}

fn one_method_call<'a>(block: &'a syn::Block, target: &'static str) -> &'a syn::ExprMethodCall {
    let mut calls = MethodCalls {
        target,
        found: Vec::new(),
    };
    calls.visit_block(block);
    assert_eq!(
        calls.found.len(),
        1,
        "expected exactly one `{target}` method call",
    );
    calls.found[0]
}

fn assert_direct_try_statement(block: &syn::Block, target: &str) {
    let matches = block
        .stmts
        .iter()
        .filter(|statement| {
            matches!(
                statement,
                Stmt::Expr(Expr::Try(try_expression), Some(_))
                    if matches!(
                        try_expression.expr.as_ref(),
                        Expr::Call(call)
                            if matches!(
                                call.func.as_ref(),
                                Expr::Path(path)
                                    if path.path.segments.last().is_some_and(
                                        |segment| segment.ident == target
                                    )
                            )
                    )
            )
        })
        .count();
    assert_eq!(
        matches, 1,
        "`{target}` result must be propagated directly with `?`",
    );
}

fn ident(expression: &Expr) -> Option<String> {
    match expression {
        Expr::Path(path) if path.path.segments.len() == 1 => {
            Some(path.path.segments[0].ident.to_string())
        }
        Expr::Paren(paren) => ident(&paren.expr),
        _ => None,
    }
}

fn pat_ident(pattern: &Pat) -> Option<String> {
    match pattern {
        Pat::Ident(ident) => Some(ident.ident.to_string()),
        _ => None,
    }
}

fn field(expression: &Expr) -> Option<(String, String)> {
    let Expr::Field(field) = expression else {
        return None;
    };
    let syn::Member::Named(member) = &field.member else {
        return None;
    };
    Some((ident(&field.base)?, member.to_string()))
}

fn closure(expression: &Expr) -> &ExprClosure {
    match expression {
        Expr::Closure(closure) => closure,
        Expr::Paren(paren) => closure(&paren.expr),
        _ => panic!("authority effect argument is not a closure"),
    }
}

fn assert_exact_method_closure(
    expression: &Expr,
    closure_inputs: &[&str],
    method_name: &str,
    forwarded_arguments: &[&str],
) {
    let closure = closure(expression);
    assert_eq!(
        closure
            .inputs
            .iter()
            .map(pat_ident)
            .collect::<Option<Vec<_>>>()
            .unwrap(),
        closure_inputs,
        "`{method_name}` closure parameters drifted",
    );
    let call = match closure.body.as_ref() {
        Expr::MethodCall(call) => call,
        Expr::Paren(paren) => match paren.expr.as_ref() {
            Expr::MethodCall(call) => call,
            _ => panic!("`{method_name}` closure body is not a direct method call"),
        },
        _ => panic!("`{method_name}` closure body is not a direct method call"),
    };
    assert_eq!(call.method, method_name);
    assert_eq!(ident(&call.receiver), Some("self".to_owned()));
    assert_eq!(
        call.args.iter().map(ident).collect::<Option<Vec<_>>>(),
        Some(
            forwarded_arguments
                .iter()
                .map(|argument| (*argument).to_owned())
                .collect(),
        ),
        "`{method_name}` does not forward the authenticated authority values exactly",
    );
}

#[derive(Default)]
struct Fields {
    names: Vec<String>,
}

impl<'ast> Visit<'ast> for Fields {
    fn visit_expr_field(&mut self, field: &'ast syn::ExprField) {
        if let syn::Member::Named(name) = &field.member {
            self.names.push(name.to_string());
        }
        syn::visit::visit_expr_field(self, field);
    }
}

fn type_idents(value: &Type) -> Vec<String> {
    struct TypeIdents(Vec<String>);
    impl<'ast> Visit<'ast> for TypeIdents {
        fn visit_type_path(&mut self, path: &'ast syn::TypePath) {
            self.0.extend(
                path.path
                    .segments
                    .iter()
                    .map(|segment| segment.ident.to_string()),
            );
            syn::visit::visit_type_path(self, path);
        }
    }
    let mut names = TypeIdents(Vec::new());
    names.visit_type(value);
    names.0
}

fn assert_persistent_streaming_cache_field(module: &syn::File) {
    let runtime = module
        .items
        .iter()
        .find_map(|item| match item {
            Item::Struct(item) if item.ident == "WebRuntime" => Some(item),
            _ => None,
        })
        .expect("missing WebRuntime struct");
    let field = runtime
        .fields
        .iter()
        .find(|field| {
            field
                .ident
                .as_ref()
                .is_some_and(|ident| ident == "vision_stack_streaming_weight_cache")
        })
        .expect("streaming cache must persist on WebRuntime across all 27 layer calls");
    assert_eq!(
        type_idents(&field.ty),
        [
            "RefCell",
            "VisionStackStreamingWeightCache",
            "wgpu",
            "Buffer",
        ],
        "persistent cache must own one typed set of WebGPU weight buffers",
    );
}

fn assert_public_export_signature(binding: RuntimeMethod<'_>) {
    assert!(matches!(binding.method.vis, Visibility::Public(_)));
    assert!(binding.method.sig.asyncness.is_none());
    assert!(
        binding
            .parent
            .attrs
            .iter()
            .any(|attribute| attribute.path().is_ident("wasm_bindgen")),
        "synchronous method is not on a wasm-exported impl",
    );
    assert_eq!(binding.method.sig.inputs.len(), 3);
    assert!(matches!(
        &binding.method.sig.inputs[0],
        FnArg::Receiver(receiver)
            if receiver.mutability.is_none()
                && matches!(
                    receiver.kind,
                    syn::ReceiverKind::Reference(_, _, None)
                )
    ));
    for (argument, expected_name, expected_type_names) in [
        (&binding.method.sig.inputs[1], "shard_id", vec!["str"]),
        (
            &binding.method.sig.inputs[2],
            "bytes",
            vec!["js_sys", "Uint8Array"],
        ),
    ] {
        let FnArg::Typed(argument) = argument else {
            panic!("`{expected_name}` is not a typed argument");
        };
        assert_eq!(pat_ident(&argument.pat), Some(expected_name.to_owned()));
        let Type::Reference(reference) = argument.ty.as_ref() else {
            panic!("`{expected_name}` must be borrowed");
        };
        assert_eq!(type_idents(&reference.elem), expected_type_names);
    }
    let ReturnType::Type(_, output) = &binding.method.sig.output else {
        panic!("synchronous wasm enqueue has no Result return");
    };
    assert_eq!(type_idents(output), ["Result", "String", "JsValue"]);
}

fn assert_exact_public_forwarding(method: &syn::ImplItemFn) {
    assert_eq!(method.block.stmts.len(), 1);
    let Stmt::Expr(Expr::MethodCall(map_err), None) = &method.block.stmts[0] else {
        panic!("wasm enqueue must directly return private result.map_err(js_error)");
    };
    assert_eq!(map_err.method, "map_err");
    assert_eq!(map_err.args.len(), 1);
    assert_eq!(ident(&map_err.args[0]), Some("js_error".to_owned()));
    let Expr::MethodCall(enqueue) = map_err.receiver.as_ref() else {
        panic!("map_err receiver is not the private enqueue call");
    };
    assert_eq!(enqueue.method, "enqueue_vision_stack_sharded_layer");
    assert_eq!(ident(&enqueue.receiver), Some("self".to_owned()));
    assert_eq!(
        enqueue.args.iter().map(ident).collect::<Option<Vec<_>>>(),
        Some(vec!["shard_id".to_owned(), "bytes".to_owned()]),
    );
}

fn reachable_calls(root: &str, callables: &BTreeMap<String, Callable<'_>>) -> BTreeSet<String> {
    assert!(
        callables.contains_key(root),
        "missing call-graph root `{root}`"
    );
    let mut visited = BTreeSet::new();
    let mut pending = vec![root.to_owned()];
    while let Some(name) = pending.pop() {
        if !visited.insert(name.clone()) {
            continue;
        }
        for callee in calls(callables[&name].block()).names {
            if callables.contains_key(&callee) {
                pending.push(callee);
            }
        }
    }
    visited
}

fn reachable_call_count(
    reachable: &BTreeSet<String>,
    callables: &BTreeMap<String, Callable<'_>>,
    target: &str,
) -> usize {
    reachable
        .iter()
        .map(|name| {
            calls(callables[name].block())
                .names
                .iter()
                .filter(|name| name.as_str() == target)
                .count()
        })
        .sum()
}

fn statement_has_awaited_call(statement: &Stmt, target: &str) -> bool {
    struct AwaitedCalls<'name> {
        target: &'name str,
        count: usize,
    }
    impl<'ast> Visit<'ast> for AwaitedCalls<'_> {
        fn visit_expr_await(&mut self, expression: &'ast syn::ExprAwait) {
            let mut calls = Calls::default();
            calls.visit_expr(&expression.base);
            self.count += calls
                .names
                .iter()
                .filter(|name| name.as_str() == self.target)
                .count();
            syn::visit::visit_expr_await(self, expression);
        }
    }
    let mut awaited = AwaitedCalls { target, count: 0 };
    awaited.visit_stmt(statement);
    awaited.count == 1
}

#[test]
fn synchronous_wasm_enqueue_binds_real_session_shard_cache_and_effects_once() {
    let module = parsed_web_module();
    let methods = runtime_methods(&module);
    assert_persistent_streaming_cache_field(&module);

    let public = *methods
        .get("enqueue_vision_encoder_stack_sharded_layer_json")
        .expect("missing synchronous wasm enqueue");
    assert_public_export_signature(public);
    assert_exact_public_forwarding(public.method);

    let private = methods
        .get("enqueue_vision_stack_sharded_layer")
        .expect("missing private streaming adapter")
        .method;
    let private_calls = calls(&private.block);
    assert_direct_try_statement(&private.block, "run_vision_stack_streaming_session_layer");
    let authority = one_function_call(&private.block, "run_vision_stack_streaming_session_layer");
    assert_eq!(authority.args.len(), 7);

    let mut owner_fields = Fields::default();
    owner_fields.visit_expr(&authority.args[0]);
    assert_eq!(
        owner_fields
            .names
            .iter()
            .filter(|name| name.as_str() == "vision_stack_session")
            .count(),
        1,
        "coordinator does not own the real vision-stack session",
    );
    let owner_calls = {
        let mut calls = Calls::default();
        calls.visit_expr(&authority.args[0]);
        calls
    };
    assert_eq!(
        owner_calls
            .names
            .iter()
            .filter(|name| name.as_str() == "borrow")
            .count(),
        1,
    );
    let mut busy_fields = Fields::default();
    busy_fields.visit_expr(&authority.args[1]);
    assert_eq!(
        busy_fields
            .names
            .iter()
            .filter(|name| name.as_str() == "execution_busy")
            .count(),
        1,
    );
    let mut cache_fields = Fields::default();
    cache_fields.visit_expr(&authority.args[2]);
    assert_eq!(
        cache_fields
            .names
            .iter()
            .filter(|name| name.as_str() == "vision_stack_streaming_weight_cache")
            .count(),
        1,
        "coordinator does not receive the persistent WebRuntime cache",
    );
    let cache_calls = {
        let mut calls = Calls::default();
        calls.visit_expr(&authority.args[2]);
        calls
    };
    assert_eq!(
        cache_calls
            .names
            .iter()
            .filter(|name| name.as_str() == "borrow_mut")
            .count(),
        1,
    );

    assert_exact_method_closure(
        &authority.args[3],
        &["session"],
        "validate_vision_stack_streaming_layer",
        &["session", "shard_id", "bytes"],
    );
    assert_exact_method_closure(
        &authority.args[4],
        &["slot", "range"],
        "create_vision_stack_streaming_weight_buffer",
        &["slot", "range"],
    );
    assert_exact_method_closure(
        &authority.args[5],
        &["slot", "range", "resource"],
        "upload_vision_stack_streaming_weight",
        &["bytes", "slot", "range", "resource"],
    );
    assert_exact_method_closure(
        &authority.args[6],
        &["session", "layer_index", "checkpoint_slot", "resources"],
        "encode_and_submit_vision_stack_layer",
        &["session", "layer_index", "checkpoint_slot", "resources"],
    );

    let validation = methods
        .get("validate_vision_stack_streaming_layer")
        .expect("missing authenticated schedule adapter")
        .method;
    assert_eq!(
        validation.block.stmts.len(),
        3,
        "schedule adapter must be a closed authenticate → admitted weight plan → return dataflow",
    );
    let Stmt::Local(admission_local) = &validation.block.stmts[0] else {
        panic!("schedule adapter must first bind authenticated layer/checkpoint");
    };
    let Pat::Tuple(admission_pattern) = &admission_local.pat else {
        panic!("accepted layer/checkpoint are not destructured together");
    };
    assert_eq!(
        admission_pattern
            .elems
            .iter()
            .map(pat_ident)
            .collect::<Option<Vec<_>>>(),
        Some(vec!["layer_index".to_owned(), "checkpoint_slot".to_owned()]),
    );
    let Expr::Try(admission_try) = admission_local.init.as_ref().unwrap().expr.as_ref() else {
        panic!("authentication Result is not propagated with `?`");
    };
    let Expr::MethodCall(admitted) = admission_try.expr.as_ref() else {
        panic!("first binding is not direct shard authentication");
    };
    assert_eq!(admitted.method, "validate_vision_stack_layer");
    assert_eq!(ident(&admitted.receiver), Some("self".to_owned()));
    assert_eq!(
        admitted.args.iter().map(ident).collect::<Option<Vec<_>>>(),
        Some(vec![
            "session".to_owned(),
            "shard_id".to_owned(),
            "bytes".to_owned(),
        ]),
        "streaming validation does not authenticate the exact supplied shard",
    );

    let Stmt::Local(ranges_local) = &validation.block.stmts[1] else {
        panic!("schedule adapter must bind ranges from the admitted browser weight plan");
    };
    assert_eq!(pat_ident(&ranges_local.pat), Some("ranges".to_owned()));
    let Expr::MethodCall(map_ranges) = ranges_local.init.as_ref().unwrap().expr.as_ref() else {
        panic!("admitted weight ranges are not converted directly");
    };
    assert_eq!(map_ranges.method, "map");
    let Expr::Field(ranges_field) = map_ranges.receiver.as_ref() else {
        panic!("streaming ranges do not come from a stored weight plan");
    };
    assert!(matches!(
        &ranges_field.member,
        syn::Member::Named(member) if member == "ranges"
    ));
    let Expr::Field(weight_plan_field) = ranges_field.base.as_ref() else {
        panic!("streaming ranges bypass the session weight plan");
    };
    assert!(matches!(
        &weight_plan_field.member,
        syn::Member::Named(member) if member == "weight_plan"
    ));
    assert_eq!(ident(&weight_plan_field.base), Some("session".to_owned()));
    assert_eq!(map_ranges.args.len(), 1);
    let range_conversion = closure(&map_ranges.args[0]);
    assert_eq!(
        range_conversion
            .inputs
            .iter()
            .map(pat_ident)
            .collect::<Option<Vec<_>>>(),
        Some(vec!["range".to_owned()]),
    );
    let Expr::Call(converted_range) = range_conversion.body.as_ref() else {
        panic!("range conversion does not construct a streaming range");
    };
    assert!(matches!(
        converted_range.func.as_ref(),
        Expr::Path(path)
            if path.path.segments.len() >= 2
                && path.path.segments[path.path.segments.len() - 2].ident
                    == "VisionStackStreamingWeightRange"
                && path.path.segments.last().is_some_and(|segment| segment.ident == "new")
    ));
    assert_eq!(
        converted_range
            .args
            .iter()
            .map(field)
            .collect::<Option<Vec<_>>>(),
        Some(vec![
            ("range".to_owned(), "offset".to_owned()),
            ("range".to_owned(), "bytes".to_owned()),
        ]),
    );

    let Stmt::Expr(Expr::Call(ok), None) = &validation.block.stmts[2] else {
        panic!("schedule adapter does not directly return its schedule");
    };
    assert!(matches!(
        ok.func.as_ref(),
        Expr::Path(path) if path.path.is_ident("Ok")
    ));
    assert_eq!(ok.args.len(), 1);
    let Expr::Call(schedule) = &ok.args[0] else {
        panic!("returned success is not a streaming schedule");
    };
    assert!(matches!(
        schedule.func.as_ref(),
        Expr::Path(path)
            if path.path.segments.len() >= 2
                && path.path.segments[path.path.segments.len() - 2].ident
                    == "VisionStackStreamingLayerSchedule"
                && path.path.segments.last().is_some_and(|segment| segment.ident == "new")
    ));
    assert_eq!(
        schedule.args.iter().map(ident).collect::<Option<Vec<_>>>(),
        Some(vec![
            "layer_index".to_owned(),
            "checkpoint_slot".to_owned(),
            "ranges".to_owned(),
        ]),
    );

    for required in [
        "validate_vision_stack_streaming_layer",
        "create_vision_stack_streaming_weight_buffer",
        "upload_vision_stack_streaming_weight",
        "encode_and_submit_vision_stack_layer",
    ] {
        assert_eq!(
            private_calls
                .names
                .iter()
                .filter(|name| name.as_str() == required)
                .count(),
            1,
            "adapter duplicates or bypasses `{required}` outside its authority closure",
        );
    }

    let allocation = methods
        .get("create_vision_stack_streaming_weight_buffer")
        .expect("missing exact-one-buffer allocation adapter")
        .method;
    let allocation_calls = calls(&allocation.block);
    assert_eq!(
        allocation_calls
            .names
            .iter()
            .filter(|name| name.as_str() == "create_runtime_buffer")
            .count(),
        1,
    );
    assert_eq!(
        allocation_calls
            .names
            .iter()
            .filter(|name| name.as_str() == "length_bytes")
            .count(),
        1,
    );
    assert_eq!(
        allocation_calls
            .names
            .iter()
            .filter(|name| name.as_str() == "create_buffer")
            .count(),
        0,
        "streaming allocation bypasses shared allocation accounting",
    );
    let create_buffer = one_method_call(&allocation.block, "create_runtime_buffer");
    assert_eq!(create_buffer.args.len(), 3);
    let Expr::MethodCall(size_call) = &create_buffer.args[1] else {
        panic!("streaming buffer size is not taken from its supplied range");
    };
    assert_eq!(size_call.method, "length_bytes");
    assert_eq!(ident(&size_call.receiver), Some("range".to_owned()));
    let Expr::Binary(usage) = &create_buffer.args[2] else {
        panic!("streaming buffer usage is not STORAGE | COPY_DST");
    };
    assert!(matches!(usage.op, syn::BinOp::BitOr(_)));
    let usage_path = |expression: &Expr| match expression {
        Expr::Path(path) => Some(
            path.path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect::<Vec<_>>(),
        ),
        _ => None,
    };
    assert_eq!(
        usage_path(&usage.left),
        Some(vec![
            "wgpu".to_owned(),
            "BufferUsages".to_owned(),
            "STORAGE".to_owned(),
        ]),
    );
    assert_eq!(
        usage_path(&usage.right),
        Some(vec![
            "wgpu".to_owned(),
            "BufferUsages".to_owned(),
            "COPY_DST".to_owned(),
        ]),
    );

    let upload = methods
        .get("upload_vision_stack_streaming_weight")
        .expect("missing exact-range upload adapter")
        .method;
    assert_eq!(
        calls(&upload.block)
            .names
            .iter()
            .filter(|name| name.as_str() == "upload_js_range")
            .count(),
        1,
        "one authority upload must write exactly its supplied range/resource",
    );
    let upload_call = one_function_call(&upload.block, "upload_js_range");
    assert_eq!(upload_call.args.len(), 4);
    let mut queue_fields = Fields::default();
    queue_fields.visit_expr(&upload_call.args[0]);
    assert_eq!(queue_fields.names, ["queue"]);
    assert_eq!(ident(&upload_call.args[1]), Some("resource".to_owned()));
    assert_eq!(ident(&upload_call.args[2]), Some("bytes".to_owned()));
    let upload_range_calls = {
        let mut calls = Calls::default();
        calls.visit_expr(&upload_call.args[3]);
        calls
    };
    assert_eq!(
        upload_range_calls
            .names
            .iter()
            .filter(|name| name.as_str() == "vision_stack_streaming_tensor_range")
            .count(),
        1,
    );
    struct ExpressionIdents(Vec<String>);
    impl<'ast> Visit<'ast> for ExpressionIdents {
        fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
            if path.path.segments.len() == 1 {
                self.0.push(path.path.segments[0].ident.to_string());
            }
            syn::visit::visit_expr_path(self, path);
        }
    }
    let mut upload_range_idents = ExpressionIdents(Vec::new());
    upload_range_idents.visit_expr(&upload_call.args[3]);
    assert!(
        upload_range_idents.0.iter().any(|name| name == "range"),
        "upload adapter does not forward the supplied authenticated range",
    );
    let range_converter = module
        .items
        .iter()
        .find_map(|item| match item {
            Item::Fn(function) if function.sig.ident == "vision_stack_streaming_tensor_range" => {
                Some(function)
            }
            _ => None,
        })
        .expect("missing exact streaming-to-WebGPU range conversion");
    assert_eq!(range_converter.block.stmts.len(), 1);
    let Stmt::Expr(Expr::Struct(converted), None) = &range_converter.block.stmts[0] else {
        panic!("streaming range converter does not directly return a tensor range");
    };
    for (member, method) in [("offset", "offset_bytes"), ("bytes", "length_bytes")] {
        let field = converted
            .fields
            .iter()
            .find(|field| {
                matches!(
                    &field.member,
                    syn::Member::Named(name) if name == member
                )
            })
            .unwrap_or_else(|| panic!("converted range has no `{member}`"));
        let Expr::MethodCall(call) = &field.expr else {
            panic!("converted `{member}` is fabricated");
        };
        assert_eq!(call.method, method);
        assert_eq!(ident(&call.receiver), Some("range".to_owned()));
    }

    let legacy = methods
        .get("execute_vision_stack_layer")
        .expect("missing accepted awaited vision encoder")
        .method;
    let legacy_calls = calls(&legacy.block);
    for required in [
        "encode_and_submit_vision_stack_layer",
        "await_queue_completion",
        "destroy_vision_qkv_web_layer_weights",
    ] {
        assert_eq!(
            legacy_calls
                .names
                .iter()
                .filter(|name| name.as_str() == required)
                .count(),
            1,
            "legacy evidence path must use shared encoder then await/destroy once: `{required}`",
        );
    }
    let awaited_statement = legacy
        .block
        .stmts
        .iter()
        .position(|statement| statement_has_awaited_call(statement, "await_queue_completion"))
        .expect("legacy queue completion is not actually awaited");
    let encoded_statement = legacy
        .block
        .stmts
        .iter()
        .position(|statement| {
            let mut calls = Calls::default();
            calls.visit_stmt(statement);
            calls
                .names
                .iter()
                .any(|name| name == "encode_and_submit_vision_stack_layer")
        })
        .unwrap();
    let destroyed_statement = legacy
        .block
        .stmts
        .iter()
        .position(|statement| {
            let mut calls = Calls::default();
            calls.visit_stmt(statement);
            calls
                .names
                .iter()
                .any(|name| name == "destroy_vision_qkv_web_layer_weights")
        })
        .unwrap();
    assert!(
        encoded_statement < awaited_statement && awaited_statement < destroyed_statement,
        "legacy path must submit → await completion → destroy weights",
    );

    let awaited = methods
        .get("run_vision_encoder_stack_sharded_layer_json")
        .expect("the accepted awaited evidence API was removed");
    assert!(awaited.method.sig.asyncness.is_some());
}

#[test]
fn streaming_has_no_barrier_and_finish_has_exactly_one_completion_primitive() {
    let module = parsed_web_module();
    let methods = runtime_methods(&module);
    let callables = callables(&module, &methods);

    let streaming = reachable_calls(
        "enqueue_vision_encoder_stack_sharded_layer_json",
        &callables,
    );
    for forbidden in [
        "await_queue_completion",
        "onSubmittedWorkDone",
        "on_submitted_work_done",
        "map_read",
        "map_async",
        "destroy",
        "destroy_vision_qkv_web_layer_weights",
    ] {
        assert_eq!(
            reachable_call_count(&streaming, &callables, forbidden),
            0,
            "streaming call graph contains `{forbidden}`",
        );
    }
    for (effect, expected) in [
        ("create_buffer", 1),
        ("upload_js_range", 1),
        ("write_buffer", 1),
        ("submit", 1),
    ] {
        assert_eq!(
            reachable_call_count(&streaming, &callables, effect),
            expected,
            "streaming graph has an extra or missing `{effect}` effect site",
        );
    }

    let finish = reachable_calls("finish_vision_encoder_stack_sharded", &callables);
    assert_eq!(
        reachable_call_count(&finish, &callables, "map_read"),
        1,
        "finish must own one semantic readback",
    );
    assert_eq!(
        reachable_call_count(&finish, &callables, "map_async"),
        1,
        "finish must resolve through one underlying completion primitive",
    );
    for forbidden in [
        "await_queue_completion",
        "onSubmittedWorkDone",
        "on_submitted_work_done",
    ] {
        assert_eq!(
            reachable_call_count(&finish, &callables, forbidden),
            0,
            "finish contains a second completion mechanism `{forbidden}`",
        );
    }
}

#[test]
fn manifest_preflight_export_advances_only_the_protocol_and_never_touches_payload_bytes() {
    let module = parsed_web_module();
    let methods = runtime_methods(&module);
    let callables = callables(&module, &methods);
    let public = methods
        .get("preflight_vision_encoder_stack_manifest_shard_json")
        .expect("missing deferred manifest-preflight wasm export");
    assert!(matches!(public.method.vis, Visibility::Public(_)));
    assert!(public.method.sig.asyncness.is_none());
    assert!(
        public
            .parent
            .attrs
            .iter()
            .any(|attribute| attribute.path().is_ident("wasm_bindgen")),
    );
    assert_eq!(public.method.sig.inputs.len(), 2);
    let FnArg::Typed(shard_id) = &public.method.sig.inputs[1] else {
        panic!("manifest preflight shard id is not typed");
    };
    assert_eq!(pat_ident(&shard_id.pat), Some("shard_id".to_owned()));
    let Type::Reference(shard_id_type) = shard_id.ty.as_ref() else {
        panic!("manifest preflight shard id is not borrowed");
    };
    assert_eq!(type_idents(&shard_id_type.elem), ["str"]);
    let ReturnType::Type(_, output) = &public.method.sig.output else {
        panic!("manifest preflight has no Result return");
    };
    assert_eq!(type_idents(output), ["Result", "String", "JsValue"]);

    assert_eq!(public.method.block.stmts.len(), 1);
    let Stmt::Expr(Expr::MethodCall(map_err), None) = &public.method.block.stmts[0] else {
        panic!("manifest preflight must directly return private result.map_err(js_error)");
    };
    assert_eq!(map_err.method, "map_err");
    assert_eq!(map_err.args.len(), 1);
    assert_eq!(ident(&map_err.args[0]), Some("js_error".to_owned()));
    let Expr::MethodCall(private_call) = map_err.receiver.as_ref() else {
        panic!("manifest preflight does not delegate directly");
    };
    assert_eq!(private_call.method, "preflight_vision_stack_manifest_shard");
    assert_eq!(ident(&private_call.receiver), Some("self".to_owned()));
    assert_eq!(
        private_call
            .args
            .iter()
            .map(ident)
            .collect::<Option<Vec<_>>>(),
        Some(vec!["shard_id".to_owned()]),
    );

    let private = methods
        .get("preflight_vision_stack_manifest_shard")
        .expect("missing private deferred preflight adapter")
        .method;
    assert_eq!(
        private.block.stmts.len(),
        4,
        "private adapter must be exactly stored owner -> stored session -> protocol acceptance -> status",
    );

    let Stmt::Local(owner_local) = &private.block.stmts[0] else {
        panic!("private adapter must first bind the stored session owner");
    };
    assert_eq!(pat_ident(&owner_local.pat), Some("owner".to_owned()));
    let Expr::MethodCall(owner_borrow) = owner_local
        .init
        .as_ref()
        .map(|init| init.expr.as_ref())
        .expect("owner binding has no initializer")
    else {
        panic!("owner binding must borrow the WebRuntime session field");
    };
    assert_eq!(owner_borrow.method, "borrow");
    assert!(owner_borrow.args.is_empty());
    assert_eq!(
        field(&owner_borrow.receiver),
        Some(("self".to_owned(), "vision_stack_session".to_owned())),
        "manifest preflight is not bound to the real stored vision-stack session",
    );

    let Stmt::Local(session_local) = &private.block.stmts[1] else {
        panic!("private adapter must next bind the mutable stored session");
    };
    let Pat::Ident(session_pattern) = &session_local.pat else {
        panic!("stored session binding must be a simple identifier");
    };
    assert_eq!(session_pattern.ident, "session");
    assert!(
        session_pattern.mutability.is_some(),
        "protocol cannot advance through an immutable session binding",
    );
    let Expr::Try(session_try) = session_local
        .init
        .as_ref()
        .map(|init| init.expr.as_ref())
        .expect("session binding has no initializer")
    else {
        panic!("missing stored session must propagate with `?`");
    };
    let Expr::MethodCall(session_error) = session_try.expr.as_ref() else {
        panic!("stored session must be required through ok_or_else");
    };
    assert_eq!(session_error.method, "ok_or_else");
    let Expr::MethodCall(stored_mut) = session_error.receiver.as_ref() else {
        panic!("session binding does not call owner.stored_mut()");
    };
    assert_eq!(stored_mut.method, "stored_mut");
    assert_eq!(ident(&stored_mut.receiver), Some("owner".to_owned()));
    assert!(stored_mut.args.is_empty());

    let Stmt::Expr(Expr::Try(accept_try), Some(_)) = &private.block.stmts[2] else {
        panic!("deferred acceptance must be a semicolon statement propagated with `?`");
    };
    let Expr::MethodCall(accept_error) = accept_try.expr.as_ref() else {
        panic!("deferred acceptance error must be mapped before propagation");
    };
    assert_eq!(accept_error.method, "map_err");
    let Expr::MethodCall(accepted) = accept_error.receiver.as_ref() else {
        panic!("map_err receiver must be the deferred protocol acceptance");
    };
    assert_eq!(accepted.method, "accept_deferred_preflight");
    assert_eq!(
        field(&accepted.receiver),
        Some(("session".to_owned(), "protocol".to_owned())),
        "deferred acceptance must advance the stored session protocol directly",
    );
    assert_eq!(
        accepted.args.iter().map(ident).collect::<Option<Vec<_>>>(),
        Some(vec!["shard_id".to_owned()]),
        "manifest preflight must forward only the declared shard id",
    );

    let Stmt::Expr(Expr::MethodCall(status_error), None) = &private.block.stmts[3] else {
        panic!("private adapter must return the advanced session status");
    };
    assert_eq!(status_error.method, "map_err");
    let Expr::Call(status) = status_error.receiver.as_ref() else {
        panic!("tail expression must serialize the same stored session");
    };
    assert!(matches!(
        status.func.as_ref(),
        Expr::Path(path)
            if path.path.segments.last().is_some_and(
                |segment| segment.ident == "vision_stack_status_json"
            )
    ));
    assert_eq!(status.args.len(), 2);
    assert!(matches!(
        &status.args[0],
        Expr::Reference(reference)
            if reference.mutability.is_none()
                && ident(&reference.expr) == Some("session".to_owned())
    ));
    assert!(matches!(
        &status.args[1],
        Expr::Lit(literal)
            if matches!(&literal.lit, syn::Lit::Bool(value) if !value.value)
    ));

    let reachable = reachable_calls(
        "preflight_vision_encoder_stack_manifest_shard_json",
        &callables,
    );
    for forbidden in [
        "inspect_js_vision_stack_f32_shard",
        "blake3_js_bytes",
        "upload_js_range",
        "write_buffer",
        "submit",
    ] {
        assert_eq!(
            reachable_call_count(&reachable, &callables, forbidden),
            0,
            "manifest-only preflight transitively calls `{forbidden}`",
        );
    }
}

#[test]
fn resident_weight_exports_reuse_authenticated_gpu_buffers_without_payload_or_upload_effects() {
    let module = parsed_web_module();
    let methods = runtime_methods(&module);
    let callables = callables(&module, &methods);

    for method_name in [
        "has_vision_encoder_stack_resident_weights",
        "begin_vision_encoder_stack_sharded_resident_with_activation_strategy_and_qkv_selection_json",
        "enqueue_vision_encoder_stack_sharded_resident_layer_json",
        "finish_vision_encoder_stack_sharded_resident",
    ] {
        let method = methods
            .get(method_name)
            .unwrap_or_else(|| panic!("missing resident vision-stack export `{method_name}`"));
        assert!(matches!(method.method.vis, Visibility::Public(_)));
        assert!(
            method
                .parent
                .attrs
                .iter()
                .any(|attribute| attribute.path().is_ident("wasm_bindgen")),
            "`{method_name}` is not exported through wasm_bindgen",
        );
    }

    let resident_check =
        reachable_calls("has_vision_encoder_stack_resident_weights", &callables);
    assert_eq!(
        reachable_call_count(
            &resident_check,
            &callables,
            "vision_stack_resident_weight_key",
        ),
        1,
        "resident lookup must derive the reviewed authenticated manifest identity",
    );
    assert_eq!(
        reachable_call_count(&resident_check, &callables, "is_ready_for"),
        1,
        "resident lookup must bind the derived identity to the persistent cache",
    );
    let resident_begin = reachable_calls(
        "begin_vision_encoder_stack_sharded_resident_with_activation_strategy_and_qkv_selection_json",
        &callables,
    );
    assert_eq!(
        reachable_call_count(
            &resident_begin,
            &callables,
            "vision_stack_resident_weight_key",
        ),
        1,
        "resident begin must derive the same reviewed manifest identity as readiness lookup",
    );
    assert_eq!(
        reachable_call_count(&resident_begin, &callables, "prepare"),
        1,
        "resident begin must prepare or reset the persistent cache before any cold storage",
    );

    for method_name in [
        "enqueue_vision_encoder_stack_sharded_resident_layer_json",
        "finish_vision_encoder_stack_sharded_resident",
    ] {
        let method = methods[method_name].method;
        assert_eq!(
            method.sig.inputs.len(),
            2,
            "`{method_name}` must accept only &self plus shard_id",
        );
        let FnArg::Typed(shard_id) = &method.sig.inputs[1] else {
            panic!("`{method_name}` shard id is not typed");
        };
        assert_eq!(pat_ident(&shard_id.pat), Some("shard_id".to_owned()));
        assert_eq!(type_idents(&shard_id.ty), ["str"]);

        let reachable = reachable_calls(method_name, &callables);
        for forbidden in [
            "inspect_js_vision_stack_shard",
            "upload_js_range",
            "write_buffer",
            "create_uploaded_js_buffer",
            "create_vision_stack_streaming_weight_buffer",
        ] {
            assert_eq!(
                reachable_call_count(&reachable, &callables, forbidden),
                0,
                "resident cache-hit graph `{method_name}` contains `{forbidden}`",
            );
        }
    }

    let resident_layer =
        reachable_calls("enqueue_vision_encoder_stack_sharded_resident_layer_json", &callables);
    assert_eq!(
        reachable_call_count(&resident_layer, &callables, "clone_layer"),
        1,
        "resident layer must resolve one authenticated cached layer",
    );
    assert_eq!(
        reachable_call_count(
            &resident_layer,
            &callables,
            "encode_and_submit_vision_stack_layer",
        ),
        1,
        "resident layer must submit the cached resources through the production encoder",
    );

    let resident_finish =
        reachable_calls("finish_vision_encoder_stack_sharded_resident", &callables);
    assert_eq!(
        reachable_call_count(&resident_finish, &callables, "clone_post_norm"),
        1,
        "resident finish must resolve the authenticated cached post-norm weights",
    );
    assert_eq!(
        reachable_call_count(&resident_finish, &callables, "map_read"),
        1,
        "resident finish must preserve the one final semantic readback",
    );

    let cold_layer = reachable_calls(
        "enqueue_vision_encoder_stack_sharded_layer_json",
        &callables,
    );
    assert_eq!(
        reachable_call_count(
            &cold_layer,
            &callables,
            "run_vision_stack_resident_cold_layer",
        ),
        1,
        "the authenticated payload path must populate resident layers through the causal authority",
    );
    assert_eq!(
        reachable_call_count(&cold_layer, &callables, "inspect_js_vision_stack_shard"),
        1,
        "resident cold population must retain payload digest and finite-value authentication",
    );

    let cold_finish = reachable_calls("finish_vision_encoder_stack_sharded", &callables);
    assert_eq!(
        reachable_call_count(&cold_finish, &callables, "store_post_norm"),
        1,
        "the authenticated post-norm path must complete the resident cache",
    );
    let finish_once = methods["finish_vision_stack_sharded_once"].method;
    let statement_index = |target: &str| {
        finish_once
            .block
            .stmts
            .iter()
            .position(|statement| {
                let mut calls = Calls::default();
                calls.visit_stmt(statement);
                calls.names.iter().any(|name| name == target)
            })
            .unwrap_or_else(|| panic!("finish path does not call `{target}`"))
    };
    let completed = statement_index("complete_browser_error_scoped_operation");
    let checked = statement_index("require_no_uncaptured_errors");
    let committed = statement_index("commit_vision_stack_resident_post_norm");
    assert!(
        completed < checked && checked < committed,
        "post-norm resources became resident before GPU completion and error checks",
    );
}
