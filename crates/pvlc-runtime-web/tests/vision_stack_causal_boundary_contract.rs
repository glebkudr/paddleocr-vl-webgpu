//! Structural M7c2b security boundary.
//!
//! This contract intentionally stays small. Rust privacy and compile-fail
//! checks own authority isolation; behavioral unit tests own event ordering.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use syn::visit::Visit;

const AUTHORITY_TYPES: [&str; 12] = [
    "VisionStackGpuEffectBoundary",
    "VisionStackPostEffectToken",
    "VisionStackOperationEffectBoundary",
    "VisionStackOperationEffectResult",
    "VisionStackOperationFailure",
    "VisionStackOperationTransaction",
    "VisionStackErrorScopeAuthority",
    "VisionStackErrorScopeLedger",
    "VisionStackErrorScopeDrain",
    "VisionStackErrorScopePushFailure",
    "VisionStackErrorScopePopAttempt",
    "VisionStackErrorScopedOperation",
];

fn crate_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn read_required_source(relative: &str) -> String {
    fs::read_to_string(crate_path(relative))
        .unwrap_or_else(|error| panic!("required structural source {relative} is missing: {error}"))
}

#[derive(Default)]
struct ForbiddenExecutionSyntax {
    unsafe_blocks: usize,
    unsafe_functions_or_impls: usize,
    macro_invocations: Vec<String>,
    attributes: Vec<String>,
}

impl<'ast> Visit<'ast> for ForbiddenExecutionSyntax {
    fn visit_expr_unsafe(&mut self, expression: &'ast syn::ExprUnsafe) {
        self.unsafe_blocks += 1;
        syn::visit::visit_expr_unsafe(self, expression);
    }

    fn visit_signature(&mut self, signature: &'ast syn::Signature) {
        if !matches!(signature.safety, syn::Safety::Default) {
            self.unsafe_functions_or_impls += 1;
        }
        syn::visit::visit_signature(self, signature);
    }

    fn visit_item_impl(&mut self, implementation: &'ast syn::ItemImpl) {
        if implementation.unsafety.is_some() {
            self.unsafe_functions_or_impls += 1;
        }
        syn::visit::visit_item_impl(self, implementation);
    }

    fn visit_item_trait(&mut self, item_trait: &'ast syn::ItemTrait) {
        if item_trait.unsafety.is_some() {
            self.unsafe_functions_or_impls += 1;
        }
        syn::visit::visit_item_trait(self, item_trait);
    }

    fn visit_item_foreign_mod(&mut self, foreign_mod: &'ast syn::ItemForeignMod) {
        if foreign_mod.unsafety.is_some() {
            self.unsafe_functions_or_impls += 1;
        }
        syn::visit::visit_item_foreign_mod(self, foreign_mod);
    }

    fn visit_macro(&mut self, invocation: &'ast syn::Macro) {
        self.macro_invocations.push(
            invocation
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string())
                .unwrap_or_else(|| "<empty>".to_owned()),
        );
        syn::visit::visit_macro(self, invocation);
    }

    fn visit_attribute(&mut self, attribute: &'ast syn::Attribute) {
        self.attributes.push(
            attribute
                .path()
                .segments
                .last()
                .map(|segment| segment.ident.to_string())
                .unwrap_or_else(|| "<empty>".to_owned()),
        );
        syn::visit::visit_attribute(self, attribute);
    }
}

fn assert_no_implicit_code_generation_or_unsafe(
    source: &str,
    context: &str,
    allowed_inert_attributes: &[&str],
    allowed_macros: &[&str],
) -> syn::File {
    let syntax = syn::parse_file(source)
        .unwrap_or_else(|error| panic!("{context} must remain parseable Rust: {error}"));
    let mut forbidden = ForbiddenExecutionSyntax::default();
    forbidden.visit_file(&syntax);
    assert_eq!(
        forbidden.unsafe_blocks, 0,
        "{context} contains an unsafe block",
    );
    assert_eq!(
        forbidden.unsafe_functions_or_impls, 0,
        "{context} contains unsafe function/impl authority",
    );
    assert!(
        forbidden
            .macro_invocations
            .iter()
            .all(|name| allowed_macros.contains(&name.as_str())),
        "{context} contains a non-allowlisted macro: {:?}",
        forbidden.macro_invocations,
    );
    assert!(
        forbidden
            .attributes
            .iter()
            .all(|attribute| allowed_inert_attributes.contains(&attribute.as_str())),
        "{context} contains a non-allowlisted attribute: {:?}",
        forbidden.attributes,
    );
    syntax
}

fn is_crate_visible(visibility: &syn::Visibility) -> bool {
    matches!(visibility, syn::Visibility::Restricted(restricted)
        if restricted.path.is_ident("crate"))
}

fn is_super_visible(visibility: &syn::Visibility) -> bool {
    matches!(visibility, syn::Visibility::Restricted(restricted)
        if restricted.path.is_ident("super"))
}

fn exact_private_out_of_line_module<'a>(syntax: &'a syn::File, name: &str) -> &'a syn::ItemMod {
    let modules = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Mod(module) if module.ident == name => Some(module),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(modules.len(), 1, "expected one module declaration {name}");
    let module = modules[0];
    assert!(matches!(module.vis, syn::Visibility::Inherited));
    assert!(
        module.content.is_none(),
        "module {name} must be out-of-line"
    );
    module
}

fn use_tree_mentions(tree: &syn::UseTree, expected: &str) -> bool {
    match tree {
        syn::UseTree::Path(path) => {
            path.ident == expected || use_tree_mentions(&path.tree, expected)
        }
        syn::UseTree::Name(name) => name.ident == expected,
        syn::UseTree::Rename(rename) => rename.ident == expected || rename.rename == expected,
        syn::UseTree::Group(group) => group
            .items
            .iter()
            .any(|item| use_tree_mentions(item, expected)),
        syn::UseTree::Glob(_) => false,
    }
}

fn assert_no_visible_reexport(syntax: &syn::File, module: &str) {
    for item in &syntax.items {
        if let syn::Item::Use(import) = item
            && !matches!(import.vis, syn::Visibility::Inherited)
        {
            assert!(!use_tree_mentions(&import.tree, module));
        }
    }
}

fn type_shape(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(path) if path.qself.is_none() && path.path.leading_colon.is_none() => path
            .path
            .segments
            .iter()
            .map(|segment| {
                let arguments = match &segment.arguments {
                    syn::PathArguments::None => String::new(),
                    syn::PathArguments::AngleBracketed(arguments) => format!(
                        "<{}>",
                        arguments
                            .args
                            .iter()
                            .map(|argument| match argument {
                                syn::GenericArgument::Type(ty) => type_shape(ty),
                                syn::GenericArgument::Lifetime(lifetime) => lifetime.to_string(),
                                _ => "<unsupported>".to_owned(),
                            })
                            .collect::<Vec<_>>()
                            .join(","),
                    ),
                    syn::PathArguments::Parenthesized(_) => "(<unsupported>)".to_owned(),
                };
                format!("{}{arguments}", segment.ident)
            })
            .collect::<Vec<_>>()
            .join("::"),
        syn::Type::Reference(reference) => format!(
            "&{}{}{}",
            reference
                .lifetime
                .as_ref()
                .map_or_else(String::new, |lifetime| format!("{lifetime} ")),
            if reference.mutability.is_some() {
                "mut "
            } else {
                ""
            },
            type_shape(&reference.elem),
        ),
        _ => "<unsupported>".to_owned(),
    }
}

fn generic_shape(generics: &syn::Generics) -> Vec<String> {
    assert!(generics.where_clause.is_none());
    generics
        .params
        .iter()
        .map(|parameter| match parameter {
            syn::GenericParam::Lifetime(parameter)
                if parameter.attrs.is_empty() && parameter.bounds.is_empty() =>
            {
                parameter.lifetime.to_string()
            }
            syn::GenericParam::Type(parameter)
                if parameter.attrs.is_empty()
                    && parameter.bounds.is_empty()
                    && parameter.default.is_none() =>
            {
                parameter.ident.to_string()
            }
            _ => "<unsupported>".to_owned(),
        })
        .collect()
}

fn named_field_shape(fields: &syn::Fields) -> Vec<(String, String)> {
    let syn::Fields::Named(fields) = fields else {
        panic!("authority struct must have named fields");
    };
    let mut shape = fields
        .named
        .iter()
        .map(|field| {
            assert!(matches!(field.vis, syn::Visibility::Inherited));
            assert!(field.attrs.is_empty());
            (
                field.ident.as_ref().unwrap().to_string(),
                type_shape(&field.ty),
            )
        })
        .collect::<Vec<_>>();
    shape.sort();
    shape
}

fn assert_authority_struct_schema(
    syntax: &syn::File,
    name: &str,
    generics: &[&str],
    fields: &[(&str, &str)],
) {
    let definitions = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Struct(item) if item.ident == name => Some(item),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(definitions.len(), 1, "{name} must have one definition");
    let definition = definitions[0];
    assert!(definition.attrs.is_empty());
    assert!(is_crate_visible(&definition.vis));
    assert_eq!(generic_shape(&definition.generics), generics);
    let mut expected = fields
        .iter()
        .map(|(field, ty)| ((*field).to_owned(), (*ty).to_owned()))
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(
        named_field_shape(&definition.fields),
        expected,
        "{name} schema drifted"
    );
}

#[test]
fn causal_authority_is_one_private_closed_macro_free_module() {
    let lib = read_required_source("src/lib.rs");
    let causal = read_required_source("src/vision_stack_causal.rs");
    let lib_syntax = syn::parse_file(&lib).expect("lib.rs must parse");
    let causal_module = exact_private_out_of_line_module(&lib_syntax, "vision_stack_causal");
    assert!(causal_module.attrs.is_empty());
    assert_no_visible_reexport(&lib_syntax, "vision_stack_causal");

    for authority in AUTHORITY_TYPES {
        assert!(
            !lib.contains(&format!("struct {authority}"))
                && !lib.contains(&format!("enum {authority}")),
            "{authority} remains in the monolithic crate root",
        );
    }
    assert!(
        !lib.contains("trait VisionStackFirstWebGpuEffectSink"),
        "the open first-effect extension trait must be removed",
    );

    let syntax =
        assert_no_implicit_code_generation_or_unsafe(
            &causal,
            "causal module",
            &["allow", "derive", "must_use"],
            &["format"],
        );
    let boundaries = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Enum(item) if item.ident == "VisionStackGpuEffectBoundary" => Some(item),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(boundaries.len(), 1);
    assert!(boundaries[0].attrs.is_empty());
    assert!(is_crate_visible(&boundaries[0].vis));
    assert!(boundaries[0].generics.params.is_empty());
    assert!(boundaries[0].generics.where_clause.is_none());
    assert_eq!(
        boundaries[0]
            .variants
            .iter()
            .map(|variant| variant.ident.to_string())
            .collect::<Vec<_>>(),
        ["PreEffect", "PostEffect"],
    );
    assert!(
        boundaries[0]
            .variants
            .iter()
            .all(|variant| variant.attrs.is_empty()
                && matches!(variant.fields, syn::Fields::Unit)
                && variant.discriminant.is_none())
    );

    assert_authority_struct_schema(
        &syntax,
        "VisionStackPostEffectToken",
        &[],
        &[
            ("tracker_id", "u64"),
            ("boundary", "VisionStackGpuEffectBoundary"),
        ],
    );
    assert_authority_struct_schema(
        &syntax,
        "VisionStackOperationEffectBoundary",
        &[],
        &[
            ("tracker", "VisionStackEffectTracker"),
            ("post_effect", "Option<VisionStackPostEffectToken>"),
        ],
    );
    assert_authority_struct_schema(
        &syntax,
        "VisionStackOperationEffectResult",
        &["T", "E"],
        &[
            ("result", "Result<T,E>"),
            ("tracker_id", "u64"),
            ("boundary", "VisionStackGpuEffectBoundary"),
        ],
    );
    assert_authority_struct_schema(
        &syntax,
        "VisionStackOperationFailure",
        &["E"],
        &[
            ("error", "E"),
            ("tracker_id", "u64"),
            ("boundary", "VisionStackGpuEffectBoundary"),
        ],
    );
    assert_authority_struct_schema(
        &syntax,
        "VisionStackOperationTransaction",
        &["Session", "T", "E"],
        &[
            ("original", "Session"),
            ("shadow", "Session"),
            ("outcome", "Result<T,VisionStackOperationFailure<E>>"),
            ("tracker_id", "u64"),
            ("boundary", "VisionStackGpuEffectBoundary"),
        ],
    );
    assert_authority_struct_schema(
        &syntax,
        "VisionStackErrorScopeAuthority",
        &["'a", "Scope"],
        &[
            ("healthy", "&'a Cell<bool>"),
            ("occupied", "&'a Cell<bool>"),
            ("ledger", "VisionStackErrorScopeLedger<Scope>"),
        ],
    );
    assert_authority_struct_schema(
        &syntax,
        "VisionStackErrorScopeLedger",
        &["Scope"],
        &[("scopes", "Vec<Scope>")],
    );
    assert_authority_struct_schema(
        &syntax,
        "VisionStackErrorScopeDrain",
        &["Value", "Error"],
        &[
            ("popped", "Vec<Value>"),
            ("failures", "Vec<Error>"),
            ("remaining", "usize"),
        ],
    );
    assert_authority_struct_schema(
        &syntax,
        "VisionStackErrorScopePushFailure",
        &["Value", "Error"],
        &[
            ("push_error", "Error"),
            ("cleanup", "VisionStackErrorScopeDrain<Value,Error>"),
        ],
    );
    assert_authority_struct_schema(
        &syntax,
        "VisionStackErrorScopedOperation",
        &["T", "Value", "Error"],
        &[
            ("operation", "Result<T,Error>"),
            ("cleanup", "VisionStackErrorScopeDrain<Value,Error>"),
        ],
    );
    let pop_attempts = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Enum(item) if item.ident == "VisionStackErrorScopePopAttempt" => Some(item),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(pop_attempts.len(), 1);
    assert!(is_crate_visible(&pop_attempts[0].vis));
    assert_eq!(generic_shape(&pop_attempts[0].generics), ["Value", "Error"]);
    assert_eq!(
        pop_attempts[0]
            .variants
            .iter()
            .map(|variant| variant.ident.to_string())
            .collect::<Vec<_>>(),
        ["Popped", "NotPopped"],
    );
    let pop_variant_types = pop_attempts[0]
        .variants
        .iter()
        .map(|variant| {
            assert!(variant.attrs.is_empty());
            assert!(variant.discriminant.is_none());
            let syn::Fields::Unnamed(fields) = &variant.fields else {
                panic!("pop classification variants must carry one positional value");
            };
            assert_eq!(fields.unnamed.len(), 1);
            type_shape(&fields.unnamed[0].ty)
        })
        .collect::<Vec<_>>();
    assert_eq!(pop_variant_types, ["Result<Value,Error>", "Error"]);
}

#[test]
fn first_effect_adapter_is_sealed_function_only_and_has_three_raw_entrypoints() {
    let web = read_required_source("src/web.rs");
    let first_effect = read_required_source("src/web/vision_stack_first_effect.rs");
    let web_syntax = syn::parse_file(&web).expect("web.rs must parse");
    let module = exact_private_out_of_line_module(&web_syntax, "vision_stack_first_effect");
    assert!(module.attrs.is_empty());
    assert_no_visible_reexport(&web_syntax, "vision_stack_first_effect");
    assert!(!web.contains("trait VisionStackFirstWebGpuEffectSink"));

    let syntax =
        assert_no_implicit_code_generation_or_unsafe(
            &first_effect,
            "first-effect module",
            &[],
            &[],
        );
    for item in &syntax.items {
        match item {
            syn::Item::Use(import) => {
                assert!(import.attrs.is_empty());
                assert!(matches!(import.vis, syn::Visibility::Inherited));
            }
            syn::Item::Fn(function)
                if function.sig.ident == "run_first_webgpu_effect"
                    && is_super_visible(&function.vis) => {}
            _ => panic!(
                "first-effect module may contain only inherited-private imports and the sealed runner"
            ),
        }
    }

    let runners = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == "run_first_webgpu_effect" => {
                Some(function)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(runners.len(), 1);
    let runner = runners[0];
    assert!(runner.attrs.is_empty());
    assert!(is_super_visible(&runner.vis));
    assert!(runner.sig.constness.is_none());
    assert!(runner.sig.asyncness.is_none());
    assert!(matches!(runner.sig.safety, syn::Safety::Default));
    assert!(runner.sig.abi.is_none());
    assert!(runner.sig.variadic.is_none());
    assert!(runner.sig.generics.params.is_empty());
    assert!(runner.sig.generics.where_clause.is_none());

    let inputs = runner
        .sig
        .inputs
        .iter()
        .map(|input| {
            let syn::FnArg::Typed(input) = input else {
                panic!("sealed runner must not have a receiver");
            };
            assert!(input.attrs.is_empty());
            let syn::Pat::Ident(pattern) = &*input.pat else {
                panic!("sealed runner inputs must be plain identifiers");
            };
            assert!(pattern.attrs.is_empty());
            assert!(pattern.by_ref.is_none());
            assert!(pattern.mutability.is_none());
            assert!(pattern.subpat.is_none());
            (pattern.ident.to_string(), type_shape(&input.ty))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        inputs,
        [
            ("device".to_owned(), "&wgpu::Device".to_owned()),
            (
                "post_effect".to_owned(),
                "&VisionStackPostEffectToken".to_owned(),
            ),
            (
                "effect".to_owned(),
                "PreparedFirstWebGpuEffect<'_>".to_owned(),
            ),
        ],
        "sealed runner authority inputs drifted",
    );
    let syn::ReturnType::Type(_, output) = &runner.sig.output else {
        panic!("sealed runner must return its fallible effect output");
    };
    assert_eq!(
        type_shape(output),
        "Result<FirstWebGpuEffectOutput,BrowserVisionStackError>",
        "sealed runner output drifted",
    );

    #[derive(Default)]
    struct RunnerBody {
        raw_methods: Vec<(String, String, usize)>,
        block_local_items: usize,
    }
    impl<'ast> Visit<'ast> for RunnerBody {
        fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
            let receiver = match &*call.receiver {
                syn::Expr::Path(path)
                    if path.qself.is_none()
                        && path.path.leading_colon.is_none()
                        && path.path.segments.len() == 1 =>
                {
                    path.path.segments[0].ident.to_string()
                }
                _ => "<not-a-plain-binding>".to_owned(),
            };
            self.raw_methods
                .push((call.method.to_string(), receiver, call.args.len()));
            syn::visit::visit_expr_method_call(self, call);
        }

        fn visit_stmt(&mut self, statement: &'ast syn::Stmt) {
            if matches!(statement, syn::Stmt::Item(_)) {
                self.block_local_items += 1;
            }
            syn::visit::visit_stmt(self, statement);
        }
    }
    let mut body = RunnerBody::default();
    body.visit_block(&runner.block);
    body.raw_methods.sort();
    assert_eq!(body.block_local_items, 0);
    assert_eq!(
        body.raw_methods,
        [
            ("call1".to_owned(), "push".to_owned(), 2),
            ("create_compute_pipeline".to_owned(), "device".to_owned(), 1,),
            ("create_shader_module".to_owned(), "device".to_owned(), 1,),
        ],
        "sealed runner raw surface drifted",
    );
}

fn newest_runtime_rlib() -> PathBuf {
    let dependencies = std::env::current_exe()
        .expect("current test executable path")
        .parent()
        .expect("test executable dependency directory")
        .to_path_buf();
    fs::read_dir(&dependencies)
        .expect("read Cargo dependency directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("libpvlc_runtime_web") && name.ends_with(".rlib")
                })
        })
        .max_by_key(|path| {
            fs::metadata(path)
                .and_then(|metadata| metadata.modified())
                .ok()
        })
        .expect("Cargo must build a host pvlc-runtime-web rlib for this integration test")
}

#[test]
fn external_crate_cannot_import_or_construct_causal_authority() {
    let fixture = crate_path("tests/ui/causal_authority_escape.rs");
    let fixture_source = read_required_source("tests/ui/causal_authority_escape.rs");
    let runtime_rlib = newest_runtime_rlib();
    let dependencies = runtime_rlib.parent().unwrap();
    let output = std::env::temp_dir().join(format!(
        "pvlc-causal-authority-escape-{}",
        std::process::id(),
    ));
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let result = Command::new(rustc)
        .arg("--edition=2024")
        .arg("--crate-name=causal_authority_escape")
        .arg(&fixture)
        .arg("--extern")
        .arg(format!("pvlc_runtime_web={}", runtime_rlib.display()))
        .arg("-L")
        .arg(format!("dependency={}", dependencies.display()))
        .arg("-o")
        .arg(&output)
        .output()
        .expect("run compile-fail authority fixture");
    let _ = fs::remove_file(&output);
    assert!(
        !result.status.success(),
        "authority escape fixture compiled"
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        !stderr.contains("can't find crate") && !stderr.contains("incompatible version"),
        "compile-fail harness failed before checking privacy:\n{stderr}",
    );
    assert!(
        stderr.contains("vision_stack_causal") && stderr.contains("private"),
        "causal module was absent instead of structurally private:\n{stderr}",
    );
    for authority in [
        "VisionStackOperationEffectBoundary",
        "VisionStackPostEffectToken",
        "VisionStackOperationEffectResult",
        "VisionStackErrorScopeAuthority",
        "VisionStackErrorScopedOperation",
    ] {
        assert!(
            fixture_source.contains(authority),
            "fixture did not exercise {authority}"
        );
    }
}

#[test]
fn behavioral_journal_tests_are_owned_next_to_the_private_module() {
    let lib = read_required_source("src/lib.rs");
    let lib_syntax = syn::parse_file(&lib).expect("lib.rs must parse");
    let test_module = exact_private_out_of_line_module(&lib_syntax, "vision_stack_causal_tests");
    assert_eq!(test_module.attrs.len(), 1);
    assert!(matches!(&test_module.attrs[0].meta, syn::Meta::List(list)
        if list.path.is_ident("cfg") && list.tokens.to_string() == "test"));

    let tests = read_required_source("src/vision_stack_causal_tests.rs");
    let syntax = syn::parse_file(&tests).expect("causal behavior tests must parse");
    for required in [
        "validation_failure_logs_no_marker_or_effect",
        "first_effect_preparation_failure_is_restored_without_marker_or_raw_call",
        "failed_start_drops_partial_local_pipelines_and_new_begin_rebuilds_from_scratch",
        "success_logs_validate_marker_first_raw_completion",
        "post_effect_failure_is_terminal",
        "duplicate_effect_run_is_rejected",
        "stale_pre_effect_result_is_rejected",
        "completion_policy_covers_every_operation_and_boundary",
        "busy_coordinator_clears_only_terminal_outcomes",
        "cross_tracker_effect_result_is_rejected",
        "every_scope_push_failure_drains_exact_prior_lifo_and_releases_admission",
        "confirmed_pop_normalization_failures_are_recorded_while_lower_scopes_are_drained",
        "every_unconfirmed_pop_retains_exact_top_poisons_and_blocks_raw_reentry",
        "device_scope_admission_is_exclusive_and_empty_drop_is_recoverable",
        "operation_error_after_first_push_is_returned_only_after_full_lifo_drain",
        "dropping_partial_push_operation_or_pop_future_poisons_and_releases_admission",
        "two_stage_pop_adapter_distinguishes_unconfirmed_rejection_from_post_pop_failure",
    ] {
        let functions = syntax
            .items
            .iter()
            .filter_map(|item| match item {
                syn::Item::Fn(function) if function.sig.ident == required => Some(function),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            functions.len(),
            1,
            "missing causal behavior test {required}"
        );
        assert!(
            functions[0]
                .attrs
                .iter()
                .any(|attribute| attribute.path().is_ident("test")),
            "causal behavior scenario {required} is not executable",
        );
        assert!(!functions[0].block.stmts.is_empty());
    }
    for event in ["validate", "marker", "first_raw", "completion"] {
        assert!(tests.contains(event), "behavioral journal omits {event}");
    }
}

fn unique_impl_method<'a>(syntax: &'a syn::File, name: &str) -> &'a syn::ImplItemFn {
    let methods = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Impl(implementation) => Some(implementation),
            _ => None,
        })
        .flat_map(|implementation| implementation.items.iter())
        .filter_map(|item| match item {
            syn::ImplItem::Fn(method) if method.sig.ident == name => Some(method),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(methods.len(), 1, "expected one impl method {name}");
    methods[0]
}

fn web_runtime_methods(syntax: &syn::File) -> BTreeMap<String, &syn::ImplItemFn> {
    let implementations = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Impl(implementation) => Some(implementation),
            _ => None,
        })
        .filter(|implementation| match &*implementation.self_ty {
            syn::Type::Path(path) => path.path.segments.last().is_some_and(|segment| {
                segment.ident == "WebRuntime"
                    && matches!(segment.arguments, syn::PathArguments::None)
            }),
            _ => false,
        })
        .collect::<Vec<_>>();
    assert!(
        !implementations.is_empty(),
        "expected at least one inherent WebRuntime implementation",
    );
    let methods = implementations
        .into_iter()
        .flat_map(|implementation| implementation.items.iter())
        .filter_map(|item| match item {
            syn::ImplItem::Fn(method) => Some((method.sig.ident.to_string(), method)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut unique = BTreeMap::new();
    for (name, method) in methods {
        assert!(
            unique.insert(name.clone(), method).is_none(),
            "duplicate inherent WebRuntime method {name}",
        );
    }
    unique
}

#[derive(Default)]
struct MethodSurface {
    method_calls: Vec<String>,
    function_calls: Vec<String>,
    function_paths: Vec<String>,
    field_reads: Vec<String>,
    path_reads: Vec<String>,
    macro_invocations: usize,
}

fn plain_path(path: &syn::Path) -> String {
    if path.leading_colon.is_some()
        || path
            .segments
            .iter()
            .any(|segment| !matches!(segment.arguments, syn::PathArguments::None))
    {
        return "<non-plain-path>".to_owned();
    }
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

impl<'ast> Visit<'ast> for MethodSurface {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.method_calls.push(call.method.to_string());
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = &*call.func
            && let Some(segment) = path.path.segments.last()
        {
            self.function_calls.push(segment.ident.to_string());
            self.function_paths.push(if path.qself.is_none() {
                plain_path(&path.path)
            } else {
                "<qualified-self-path>".to_owned()
            });
        }
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_expr_field(&mut self, field: &'ast syn::ExprField) {
        if let syn::Member::Named(name) = &field.member {
            self.field_reads.push(name.to_string());
        }
        syn::visit::visit_expr_field(self, field);
    }

    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.qself.is_none()
            && path.path.segments.len() == 1
            && let Some(segment) = path.path.segments.first()
        {
            self.path_reads.push(segment.ident.to_string());
        }
        syn::visit::visit_expr_path(self, path);
    }

    fn visit_macro(&mut self, invocation: &'ast syn::Macro) {
        self.macro_invocations += 1;
        syn::visit::visit_macro(self, invocation);
    }
}

#[derive(Default)]
struct RawDeviceMethodArguments {
    names: Vec<String>,
}

impl<'ast> Visit<'ast> for RawDeviceMethodArguments {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if call.method == "raw_device_method" {
            assert_eq!(call.args.len(), 1);
            let argument = call.args.first().unwrap();
            let syn::Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(name),
                ..
            }) = argument
            else {
                panic!("raw_device_method name must be one literal");
            };
            self.names.push(name.value());
        }
        syn::visit::visit_expr_method_call(self, call);
    }
}

fn method_surface(method: &syn::ImplItemFn) -> MethodSurface {
    let mut surface = MethodSurface::default();
    surface.visit_block(&method.block);
    surface
}

fn expression_surface(expression: &syn::Expr) -> MethodSurface {
    let mut surface = MethodSurface::default();
    surface.visit_expr(expression);
    surface
}

fn reachable_web_runtime_methods<'a>(
    methods: &BTreeMap<String, &'a syn::ImplItemFn>,
    root: &str,
) -> Vec<&'a syn::ImplItemFn> {
    assert!(methods.contains_key(root), "missing WebRuntime root {root}");
    let mut queue = VecDeque::from([root.to_owned()]);
    let mut visited = BTreeSet::new();
    while let Some(name) = queue.pop_front() {
        if !visited.insert(name.clone()) {
            continue;
        }
        let surface = method_surface(methods[&name]);
        for call in surface
            .method_calls
            .into_iter()
            .chain(surface.function_calls)
        {
            if methods.contains_key(&call) && !visited.contains(&call) {
                queue.push_back(call);
            }
        }
    }
    visited.into_iter().map(|name| methods[&name]).collect()
}

fn assert_cpu_only_validation_graph(methods: &BTreeMap<String, &syn::ImplItemFn>, root: &str) {
    const ALLOWED_METHODS: &[&str] = &[
        "accept_execution",
        "as_ref",
        "clear",
        "clear_uncaptured_errors",
        "filter",
        "into",
        "into_inner",
        "is_none",
        "is_some",
        "is_some_and",
        "lock",
        "manifest",
        "map_err",
        "ok_or_else",
        "outcome",
        "push_str",
        "to_owned",
        "to_string",
        "unwrap_or_else",
        "validate_vision_stack_session_authority",
    ];
    const ALLOWED_FUNCTIONS: &[&str] = &[
        "BrowserVisionStackError",
        "BrowserVisionStackStaticPlan::new",
        "Err",
        "Ok",
        "Some",
        "String::from",
        "inspect_js_vision_stack_shard",
    ];
    for method in reachable_web_runtime_methods(methods, root) {
        let surface = method_surface(method);
        assert_eq!(
            surface.macro_invocations, 0,
            "PRE validation graph hides code in a macro through {}",
            method.sig.ident,
        );
        assert!(
            !surface.field_reads.iter().any(|field| matches!(
                field.as_str(),
                "device" | "queue" | "pipelines" | "buffer_allocations" | "submissions"
            )),
            "PRE validation graph reaches WebGPU authority through {}: {:?}",
            method.sig.ident,
            surface.field_reads,
        );
        assert!(
            surface
                .method_calls
                .iter()
                .all(|call| ALLOWED_METHODS.contains(&call.as_str())),
            "PRE validation graph escapes its CPU-only method surface through {}: {:?}",
            method.sig.ident,
            surface.method_calls,
        );
        assert!(
            surface
                .function_paths
                .iter()
                .all(|call| ALLOWED_FUNCTIONS.contains(&call.as_str())),
            "PRE validation graph escapes its CPU-only function surface through {}: {:?}",
            method.sig.ident,
            surface.function_paths,
        );
    }
}

fn assert_cpu_only_completion_graph(methods: &BTreeMap<String, &syn::ImplItemFn>) {
    for method in reachable_web_runtime_methods(methods, "finish_vision_stack_transaction") {
        let surface = method_surface(method);
        assert_eq!(
            surface.macro_invocations, 0,
            "completion graph hides post-transaction work in a macro through {}",
            method.sig.ident,
        );
        assert!(
            surface
                .field_reads
                .iter()
                .all(|field| field == "execution_busy"),
            "completion graph reaches non-CPU runtime authority through {}: {:?}",
            method.sig.ident,
            surface.field_reads,
        );
        assert!(
            surface
                .method_calls
                .iter()
                .all(|call| matches!(call.as_str(), "map_err" | "to_owned" | "to_string")),
            "completion graph escapes its CPU-only method surface through {}: {:?}",
            method.sig.ident,
            surface.method_calls,
        );
        assert!(
            surface.function_paths.iter().all(|call| matches!(
                call.as_str(),
                "Err" | "coordinate_vision_stack_completion_busy"
            )),
            "completion graph escapes its CPU-only function surface through {}: {:?}",
            method.sig.ident,
            surface.function_paths,
        );
    }
}

#[derive(Default)]
struct UnprojectedSelf {
    found: bool,
}

impl<'ast> Visit<'ast> for UnprojectedSelf {
    fn visit_expr_path(&mut self, path: &'ast syn::ExprPath) {
        if path.qself.is_none() && path.path.is_ident("self") {
            self.found = true;
        }
        syn::visit::visit_expr_path(self, path);
    }

    fn visit_expr_field(&mut self, field: &'ast syn::ExprField) {
        if !matches!(&*field.base, syn::Expr::Path(path) if path.path.is_ident("self")) {
            self.visit_expr(&field.base);
        }
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        if matches!(&*call.receiver, syn::Expr::Path(path) if path.path.is_ident("self")) {
            self.found = true;
        } else {
            self.visit_expr(&call.receiver);
        }
        for argument in &call.args {
            self.visit_expr(argument);
        }
    }
}

#[derive(Default)]
struct FullRuntimeAuthorityEscapes {
    free_calls: Vec<String>,
}

impl<'ast> Visit<'ast> for FullRuntimeAuthorityEscapes {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let function = match &*call.func {
            syn::Expr::Path(path) if path.qself.is_none() => plain_path(&path.path),
            _ => "<dynamic-call>".to_owned(),
        };
        if call.args.iter().enumerate().any(|(index, argument)| {
            if function == "collect_vision_stack_session_resources" && index == 1 {
                return false;
            }
            let mut exposure = UnprojectedSelf::default();
            exposure.visit_expr(argument);
            exposure.found
        }) {
            self.free_calls.push(function);
        }
        syn::visit::visit_expr_call(self, call);
    }
}

#[derive(Default)]
struct FullRuntimeAuthorityCaptures {
    struct_fields: Vec<String>,
}

impl<'ast> Visit<'ast> for FullRuntimeAuthorityCaptures {
    fn visit_expr_struct(&mut self, expression: &'ast syn::ExprStruct) {
        for field in &expression.fields {
            let mut exposure = UnprojectedSelf::default();
            exposure.visit_expr(&field.expr);
            if exposure.found {
                self.struct_fields.push(match &field.member {
                    syn::Member::Named(name) => name.to_string(),
                    syn::Member::Unnamed(index) => index.index.to_string(),
                });
            }
        }
        syn::visit::visit_expr_struct(self, expression);
    }
}

fn closure_block(expression: &syn::Expr) -> &syn::Block {
    let syn::Expr::Closure(closure) = expression else {
        panic!("transaction validation must be an explicit closure");
    };
    let syn::Expr::Block(block) = &*closure.body else {
        panic!("transaction validation closure must use an explicit block");
    };
    &block.block
}

fn single_tail_expression<'a>(block: &'a syn::Block, context: &str) -> &'a syn::Expr {
    assert_eq!(
        block.stmts.len(),
        1,
        "{context} must contain exactly one immediate tail expression",
    );
    let syn::Stmt::Expr(expression, None) = &block.stmts[0] else {
        panic!("{context} must contain no statement before its immediate effect");
    };
    expression
}

fn closure_single_tail_expression<'a>(expression: &'a syn::Expr, context: &str) -> &'a syn::Expr {
    single_tail_expression(closure_block(expression), context)
}

fn direct_awaited_method_call<'a>(
    expression: &'a syn::Expr,
    expected: &str,
    context: &str,
) -> &'a syn::ExprMethodCall {
    let syn::Expr::Await(awaited) = expression else {
        panic!("{context} must directly await {expected}");
    };
    let syn::Expr::MethodCall(call) = &*awaited.base else {
        panic!("{context} must directly call {expected}");
    };
    assert_eq!(call.method, expected, "{context} called the wrong method");
    call
}

fn plain_binding(pattern: &syn::Pat, context: &str) -> String {
    let syn::Pat::Ident(binding) = pattern else {
        panic!("{context} must bind one plain local authority");
    };
    assert!(binding.attrs.is_empty());
    assert!(binding.by_ref.is_none());
    assert!(binding.mutability.is_none());
    assert!(binding.subpat.is_none());
    binding.ident.to_string()
}

fn direct_fallible_method_binding<'a>(
    statement: &'a syn::Stmt,
    expected_method: &str,
    context: &str,
) -> (String, &'a syn::ExprMethodCall) {
    let syn::Stmt::Local(local) = statement else {
        panic!("{context} must be a direct local binding");
    };
    let binding = plain_binding(&local.pat, context);
    let initializer = local
        .init
        .as_ref()
        .unwrap_or_else(|| panic!("{context} omitted its initializer"));
    let syn::Expr::Try(propagated) = &*initializer.expr else {
        panic!("{context} must propagate failure immediately with ?");
    };
    let syn::Expr::MethodCall(call) = &*propagated.expr else {
        panic!("{context} must call {expected_method} directly");
    };
    assert_eq!(
        call.method, expected_method,
        "{context} called the wrong method"
    );
    assert!(
        matches!(&*call.receiver, syn::Expr::Path(path) if path.path.is_ident("self")),
        "{context} must use this WebRuntime directly",
    );
    assert!(
        call.args
            .iter()
            .all(|argument| matches!(argument, syn::Expr::Path(_))),
        "{context} evaluates work inside an authority argument",
    );
    (binding, call)
}

fn exact_pre_effect_validation_closure(
    validation_expression: &syn::Expr,
    operation: &str,
    validation_method: &str,
) -> String {
    let syn::Expr::Closure(closure) = validation_expression else {
        panic!("{operation} validation must be one explicit closure");
    };
    assert_eq!(
        closure.inputs.len(),
        1,
        "{operation} validation closure must receive only its transaction shadow",
    );
    let shadow = plain_binding(&closure.inputs[0], operation);
    assert_eq!(shadow, "shadow", "{operation} must name its PRE shadow");
    let validation = closure_block(validation_expression);
    assert_eq!(
        validation.stmts.len(),
        3,
        "{operation} PRE closure must contain only validation, first-effect preparation, and return",
    );
    let (validated, validation_call) = direct_fallible_method_binding(
        &validation.stmts[0],
        validation_method,
        &format!("{operation} validation"),
    );
    assert!(
        !validation_call.args.is_empty(),
        "{operation} validation omitted its shadow session",
    );
    assert!(
        matches!(&validation_call.args[0], syn::Expr::Path(path) if path.path.is_ident(&shadow)),
        "{operation} validates captured state instead of its transaction shadow",
    );
    let (prepared, preparation_call) = direct_fallible_method_binding(
        &validation.stmts[1],
        "prepare_vision_stack_first_error_scope",
        &format!("{operation} first-effect preparation"),
    );
    assert!(
        preparation_call.args.is_empty(),
        "{operation} first-effect preparation accepts unsealed authority arguments",
    );

    let syn::Stmt::Expr(tail, None) = &validation.stmts[2] else {
        panic!("{operation} PRE closure must directly return both authorities");
    };
    let syn::Expr::Call(ok) = tail else {
        panic!("{operation} PRE closure must directly return Ok((validated, prepared))");
    };
    assert!(
        matches!(&*ok.func, syn::Expr::Path(path) if path.path.is_ident("Ok")),
        "{operation} PRE closure return must be the Result::Ok constructor",
    );
    assert_eq!(ok.args.len(), 1);
    let syn::Expr::Tuple(authorities) = &ok.args[0] else {
        panic!("{operation} PRE closure must return one authority pair");
    };
    assert_eq!(authorities.elems.len(), 2);
    for (value, expected) in authorities.elems.iter().zip([&validated, &prepared]) {
        assert!(
            matches!(value, syn::Expr::Path(path) if path.path.is_ident(expected)),
            "{operation} PRE closure returned a substituted authority",
        );
    }
    prepared
}

fn assert_exact_cpu_session_acquisition(statement: &syn::Stmt, operation: &str) {
    let syn::Stmt::Local(local) = statement else {
        panic!("{operation} must begin with one session-acquisition binding");
    };
    let syn::Pat::Tuple(tuple) = &local.pat else {
        panic!("{operation} first binding must be (lease, session)");
    };
    assert_eq!(tuple.elems.len(), 2);
    assert_eq!(plain_binding(&tuple.elems[0], operation), "lease");
    assert_eq!(plain_binding(&tuple.elems[1], operation), "session");
    let initializer = local
        .init
        .as_ref()
        .expect("session acquisition must have an initializer");
    let syn::Expr::Try(propagated) = &*initializer.expr else {
        panic!("{operation} must propagate acquisition failure immediately");
    };
    let syn::Expr::MethodCall(map_error) = &*propagated.expr else {
        panic!("{operation} acquisition must terminate in map_err");
    };
    assert_eq!(map_error.method, "map_err");
    assert_eq!(map_error.args.len(), 1);
    let syn::Expr::Closure(error_mapping) = &map_error.args[0] else {
        panic!("{operation} acquisition error mapping must be an explicit CPU-only closure");
    };
    let syn::Expr::MethodCall(acquire) = &*map_error.receiver else {
        panic!("{operation} must acquire directly from its session owner");
    };
    assert_eq!(acquire.method, "acquire");
    assert!(acquire.args.is_empty());
    let syn::Expr::MethodCall(borrow) = &*acquire.receiver else {
        panic!("{operation} must borrow its session owner directly");
    };
    assert_eq!(borrow.method, "borrow_mut");
    assert!(borrow.args.is_empty());
    let syn::Expr::Field(owner) = &*borrow.receiver else {
        panic!("{operation} acquisition must read only vision_stack_session");
    };
    assert!(matches!(&*owner.base, syn::Expr::Path(path) if path.path.is_ident("self")),);
    assert!(
        matches!(&owner.member, syn::Member::Named(member) if member == "vision_stack_session")
    );

    let acquisition_surface = expression_surface(&initializer.expr);
    assert_eq!(
        acquisition_surface.macro_invocations, 0,
        "{operation} hides pre-transaction work in a macro",
    );
    assert_eq!(
        acquisition_surface.field_reads,
        ["vision_stack_session"],
        "{operation} reads additional runtime authority before the transaction",
    );
    assert!(
        acquisition_surface.function_paths.is_empty(),
        "{operation} calls a free function before the transaction: {:?}",
        acquisition_surface.function_paths,
    );
    for required in ["borrow_mut", "acquire", "map_err"] {
        assert_eq!(
            acquisition_surface
                .method_calls
                .iter()
                .filter(|call| call.as_str() == required)
                .count(),
            1,
            "{operation} acquisition must call .{required}() exactly once",
        );
    }
    assert!(
        acquisition_surface.method_calls.iter().all(|call| matches!(
            call.as_str(),
            "borrow_mut" | "acquire" | "map_err" | "to_owned" | "to_string" | "push_str"
        )),
        "{operation} performs non-CPU work before its transaction: {:?}",
        acquisition_surface.method_calls,
    );
    let error_surface = expression_surface(&error_mapping.body);
    assert_eq!(error_surface.macro_invocations, 0);
    assert!(error_surface.field_reads.is_empty());
    assert!(error_surface.function_paths.is_empty());
    assert!(!error_surface.path_reads.iter().any(|path| path == "self"));
}

fn direct_transaction_statement<'a>(
    method: &'a syn::ImplItemFn,
    operation: &str,
) -> &'a syn::ExprCall {
    assert!(
        method.block.stmts.len() >= 2,
        "{operation} omitted its transaction",
    );
    assert_exact_cpu_session_acquisition(&method.block.stmts[0], operation);
    let syn::Stmt::Local(transaction) = &method.block.stmts[1] else {
        panic!("{operation} transaction must immediately follow acquisition");
    };
    assert_eq!(plain_binding(&transaction.pat, operation), "transaction");
    let initializer = transaction
        .init
        .as_ref()
        .expect("transaction binding must have an initializer");
    let syn::Expr::Await(awaited) = &*initializer.expr else {
        panic!("{operation} must directly await its transaction");
    };
    let syn::Expr::Call(call) = &*awaited.base else {
        panic!("{operation} transaction binding must be a direct function call");
    };
    assert!(
        matches!(&*call.func, syn::Expr::Path(path)
            if path.path.is_ident("run_vision_stack_operation_transaction")),
        "{operation} invokes the wrong transaction constructor",
    );
    assert_eq!(call.args.len(), 3);
    assert!(
        matches!(&call.args[0], syn::Expr::Path(path) if path.path.is_ident("session")),
        "{operation} evaluates work while constructing transaction arg0",
    );
    call
}

fn assert_exact_cpu_completion_suffix(
    method: &syn::ImplItemFn,
    operation: &str,
    async_operation: &str,
    activity: &str,
) {
    assert_eq!(
        method.block.stmts.len(),
        4,
        "{operation} wrapper must contain only acquisition, transaction, completion, and return",
    );
    let syn::Stmt::Local(completion) = &method.block.stmts[2] else {
        panic!("{operation} must complete its transaction immediately");
    };
    let syn::Pat::Tuple(outputs) = &completion.pat else {
        panic!("{operation} completion must bind (outcome, result)");
    };
    assert_eq!(outputs.elems.len(), 2);
    assert_eq!(plain_binding(&outputs.elems[0], operation), "outcome");
    assert_eq!(plain_binding(&outputs.elems[1], operation), "result");
    let completion_initializer = completion
        .init
        .as_ref()
        .expect("completion binding must have an initializer");
    let syn::Expr::Block(completion_block) = &*completion_initializer.expr else {
        panic!("{operation} completion must use one explicit CPU-only block");
    };
    assert_eq!(completion_block.block.stmts.len(), 2);

    let syn::Stmt::Local(owner) = &completion_block.block.stmts[0] else {
        panic!("{operation} completion must borrow its session owner first");
    };
    let syn::Pat::Ident(owner_binding) = &owner.pat else {
        panic!("{operation} completion owner must be one local binding");
    };
    assert_eq!(owner_binding.ident, "owner");
    assert!(owner_binding.mutability.is_some());
    let owner_initializer = owner
        .init
        .as_ref()
        .expect("completion owner must have an initializer");
    let syn::Expr::MethodCall(borrow) = &*owner_initializer.expr else {
        panic!("{operation} completion owner must be a direct borrow");
    };
    assert_eq!(borrow.method, "borrow_mut");
    assert!(borrow.args.is_empty());
    let syn::Expr::Field(session_owner) = &*borrow.receiver else {
        panic!("{operation} completion must borrow vision_stack_session");
    };
    assert!(matches!(&*session_owner.base, syn::Expr::Path(path) if path.path.is_ident("self")),);
    assert!(matches!(&session_owner.member, syn::Member::Named(member)
        if member == "vision_stack_session"));

    let syn::Stmt::Expr(completed, None) = &completion_block.block.stmts[1] else {
        panic!("{operation} completion block must directly return causal completion");
    };
    let syn::Expr::Call(completed) = completed else {
        panic!("{operation} completion must be one direct function call");
    };
    assert!(matches!(&*completed.func, syn::Expr::Path(path)
            if path.path.is_ident("complete_vision_stack_async_operation")),);
    assert_eq!(completed.args.len(), 4);
    assert!(matches!(&completed.args[0], syn::Expr::Reference(reference)
        if reference.mutability.is_some()
            && matches!(&*reference.expr, syn::Expr::Path(path) if path.path.is_ident("owner"))));
    assert!(matches!(&completed.args[1], syn::Expr::Path(path) if path.path.is_ident("lease")));
    assert!(matches!(&completed.args[2], syn::Expr::Path(path)
        if plain_path(&path.path) == format!("VisionStackAsyncOperation::{async_operation}")));
    assert!(matches!(&completed.args[3], syn::Expr::Path(path)
        if path.path.is_ident("transaction")));

    let syn::Stmt::Expr(returned, None) = &method.block.stmts[3] else {
        panic!("{operation} must directly return its normalized completion");
    };
    let syn::Expr::MethodCall(finish) = returned else {
        panic!("{operation} final expression must be finish_vision_stack_transaction");
    };
    assert_eq!(finish.method, "finish_vision_stack_transaction");
    assert!(matches!(&*finish.receiver, syn::Expr::Path(path) if path.path.is_ident("self")));
    assert_eq!(finish.args.len(), 3);
    assert!(matches!(&finish.args[0], syn::Expr::Path(path) if path.path.is_ident("outcome")));
    assert!(matches!(&finish.args[1], syn::Expr::Path(path) if path.path.is_ident("result")));
    assert!(matches!(&finish.args[2], syn::Expr::Lit(literal)
        if matches!(&literal.lit, syn::Lit::Str(value) if value.value() == activity)));
}

#[derive(Default)]
struct PatternIdents {
    names: Vec<String>,
}

impl<'ast> Visit<'ast> for PatternIdents {
    fn visit_pat_ident(&mut self, ident: &'ast syn::PatIdent) {
        self.names.push(ident.ident.to_string());
        syn::visit::visit_pat_ident(self, ident);
    }
}

fn closure_parameter_idents(expression: &syn::Expr, index: usize) -> Vec<String> {
    let syn::Expr::Closure(closure) = expression else {
        panic!("transaction effect must be an explicit closure");
    };
    let pattern = closure
        .inputs
        .iter()
        .nth(index)
        .unwrap_or_else(|| panic!("transaction closure omitted parameter {index}"));
    let mut idents = PatternIdents::default();
    idents.visit_pat(pattern);
    assert!(
        !idents.names.is_empty(),
        "transaction parameter {index} must retain named values",
    );
    idents.names
}

fn pattern_contains_ident(pattern: &syn::Pat, expected: &str) -> bool {
    let mut idents = PatternIdents::default();
    idents.visit_pat(pattern);
    idents.names.iter().any(|name| name == expected)
}

fn closure_parameter_contains_ident(expression: &syn::Expr, index: usize, expected: &str) -> bool {
    closure_parameter_idents(expression, index)
        .iter()
        .any(|name| name == expected)
}

fn typed_input_contains_ident(input: &syn::FnArg, expected: &str) -> bool {
    match input {
        syn::FnArg::Typed(typed) => pattern_contains_ident(&typed.pat, expected),
        syn::FnArg::Receiver(_) => false,
    }
}

#[derive(Default)]
struct MethodCalls<'a> {
    calls: Vec<&'a syn::ExprMethodCall>,
}

#[derive(Default)]
struct PipelineStatePublications {
    gpu_states: usize,
    pipeline_fields: usize,
    exact_bindings: usize,
}

impl<'ast> Visit<'ast> for PipelineStatePublications {
    fn visit_expr_struct(&mut self, expression: &'ast syn::ExprStruct) {
        let is_gpu_state = expression
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "BrowserVisionStackGpuState");
        if is_gpu_state {
            self.gpu_states += 1;
            for field in &expression.fields {
                let syn::Member::Named(member) = &field.member else {
                    continue;
                };
                if member != "pipelines" {
                    continue;
                }
                self.pipeline_fields += 1;
                let syn::Expr::Path(value) = &field.expr else {
                    continue;
                };
                if value.path.is_ident("pipelines") {
                    self.exact_bindings += 1;
                }
            }
        }
        syn::visit::visit_expr_struct(self, expression);
    }
}

impl<'ast> Visit<'ast> for MethodCalls<'ast> {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.calls.push(call);
        syn::visit::visit_expr_method_call(self, call);
    }
}

#[derive(Default)]
struct TransactionCalls {
    count: usize,
}

impl<'ast> Visit<'ast> for TransactionCalls {
    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        let is_transaction =
            match &*call.func {
                syn::Expr::Path(path) => path.path.segments.last().is_some_and(|segment| {
                    segment.ident == "run_vision_stack_operation_transaction"
                }),
                _ => false,
            };
        if is_transaction {
            self.count += 1;
        }
        syn::visit::visit_expr_call(self, call);
    }
}

#[test]
fn pre_transaction_validation_graph_is_closed_and_cpu_only() {
    let web = read_required_source("src/web.rs");
    let syntax = syn::parse_file(&web).expect("web.rs must parse");
    let methods = web_runtime_methods(&syntax);
    for validation in [
        "validate_vision_stack_start",
        "validate_vision_stack_layer",
        "validate_vision_stack_finish",
    ] {
        assert_cpu_only_validation_graph(&methods, validation);
    }
}

#[test]
fn post_transaction_completion_graph_is_closed_and_cpu_only() {
    let web = read_required_source("src/web.rs");
    let syntax = syn::parse_file(&web).expect("web.rs must parse");
    let methods = web_runtime_methods(&syntax);
    assert_cpu_only_completion_graph(&methods);
}

#[test]
fn operation_wrappers_cross_the_transaction_after_one_cpu_acquisition() {
    let web = read_required_source("src/web.rs");
    let syntax = syn::parse_file(&web).expect("web.rs must parse");
    for (operation, validation_method, async_operation, activity) in [
        (
            "start_vision_stack_sharded",
            "validate_vision_stack_start",
            "Start",
            "starting",
        ),
        (
            "run_vision_stack_sharded_layer",
            "validate_vision_stack_layer",
            "Layer",
            "running a layer",
        ),
        (
            "finish_vision_stack_sharded",
            "validate_vision_stack_finish",
            "Finish",
            "finishing",
        ),
    ] {
        let method = unique_impl_method(&syntax, operation);
        let transaction = direct_transaction_statement(method, operation);
        assert_eq!(transaction.args.len(), 3);
        exact_pre_effect_validation_closure(&transaction.args[1], operation, validation_method);
        assert_exact_cpu_completion_suffix(method, operation, async_operation, activity);
    }
}

#[test]
fn browser_resolves_the_fallible_first_effect_before_marking_post_effect() {
    let web = read_required_source("src/web.rs");
    let syntax = syn::parse_file(&web).expect("web.rs must parse");
    let methods = web_runtime_methods(&syntax);
    let prepare = unique_impl_method(&syntax, "prepare_vision_stack_first_error_scope");
    let prepare_surface = method_surface(prepare);
    assert_eq!(
        prepare_surface.macro_invocations, 0,
        "PRE first-effect preparation must not hide raw work in a macro",
    );
    assert_eq!(
        prepare_surface
            .method_calls
            .iter()
            .filter(|call| call.as_str() == "raw_device_method")
            .count(),
        2,
        "first-effect preparation must resolve pushErrorScope and popErrorScope while PRE",
    );
    assert!(
        prepare_surface.method_calls.iter().all(|call| matches!(
            call.as_str(),
            "clear_uncaptured_errors" | "raw_device_method" | "to_owned" | "into"
        )),
        "PRE scope preparation escapes its health/reflection surface: {:?}",
        prepare_surface.method_calls,
    );
    assert_eq!(
        prepare_surface
            .function_paths
            .iter()
            .filter(|call| call.as_str() == "VisionStackErrorScopeAuthority::acquire")
            .count(),
        1,
        "PRE preparation must acquire the behavior-tested serialized scope authority",
    );
    assert!(
        prepare_surface
            .function_calls
            .iter()
            .all(|call| matches!(call.as_str(), "Ok" | "Err" | "acquire")),
        "PRE scope preparation calls an unsealed constructor: {:?}",
        prepare_surface.function_calls,
    );
    let first_prepare_statement = prepare
        .block
        .stmts
        .first()
        .expect("scope preparation must start with serialized admission");
    let syn::Stmt::Local(first_authority) = first_prepare_statement else {
        panic!("scope preparation must bind its serialized authority first");
    };
    let first_initializer = first_authority
        .init
        .as_ref()
        .expect("scope authority binding must have an initializer");
    let syn::Expr::Try(first_admission) = &*first_initializer.expr else {
        panic!("scope admission must propagate rejection with ?");
    };
    let syn::Expr::Call(first_acquire) = &*first_admission.expr else {
        panic!("scope admission must directly call the closed authority constructor");
    };
    assert!(matches!(&*first_acquire.func, syn::Expr::Path(path)
        if plain_path(&path.path) == "VisionStackErrorScopeAuthority::acquire"));
    let mut first_prepare_surface = MethodSurface::default();
    first_prepare_surface.visit_stmt(first_prepare_statement);
    assert_eq!(
        first_prepare_surface
            .function_paths
            .iter()
            .filter(|call| call.as_str() == "VisionStackErrorScopeAuthority::acquire")
            .count(),
        1,
        "serialized scope admission must run before any fallible reflection",
    );
    assert!(
        !first_prepare_surface
            .method_calls
            .iter()
            .any(|call| matches!(
                call.as_str(),
                "clear_uncaptured_errors" | "raw_device_method"
            )),
        "runtime state or fallible raw reflection changed before serialized admission",
    );
    let mut raw_method_arguments = RawDeviceMethodArguments::default();
    raw_method_arguments.visit_block(&prepare.block);
    assert_eq!(
        raw_method_arguments.names,
        ["pushErrorScope", "popErrorScope"],
        "PRE preparation must resolve the two exact distinct raw methods",
    );

    unique_impl_method(&syntax, "push_vision_stack_error_scopes");
    let push_graph = reachable_web_runtime_methods(&methods, "push_vision_stack_error_scopes");
    assert!(
        !push_graph
            .iter()
            .flat_map(|method| {
                let surface = method_surface(method);
                surface
                    .method_calls
                    .into_iter()
                    .chain(surface.function_calls)
            })
            .any(|call| call == "raw_device_method"),
        "post-effect path transitively performs fallible BrowserWebGPU reflection before its first raw call",
    );
    let push_surface = method_surface(unique_impl_method(
        &syntax,
        "push_vision_stack_error_scopes",
    ));
    assert_eq!(
        push_surface.macro_invocations, 0,
        "post-effect scope push must not hide raw work or reflection in a macro",
    );
    for call in &push_surface.function_calls {
        assert!(
            matches!(
                call.as_str(),
                "run_first_webgpu_effect"
                    | "from_str"
                    | "Ok"
                    | "Err"
                    | "after_first_push"
                    | "push_vision_stack_error_scope_or_drain"
                    | "pop_browser_vision_stack_error_scope"
                    | "browser_vision_stack_scope_push_failure"
            ),
            "post-effect scope push can escape its sealed raw-call surface through {call}",
        );
    }
    for call in &push_surface.method_calls {
        assert!(
            matches!(
                call.as_str(),
                "effect_tracker_id"
                    | "effect_boundary"
                    | "call1"
                    | "filter_str"
                    | "map"
                    | "map_err"
                    | "as_str"
                    | "to_owned"
                    | "into"
            ),
            "post-effect scope push can escape its sealed raw-call surface through .{call}()",
        );
    }
    let push_method = unique_impl_method(&syntax, "push_vision_stack_error_scopes");
    let sealed_statements = push_method
        .block
        .stmts
        .iter()
        .enumerate()
        .filter_map(|(index, statement)| {
            let syn::Stmt::Local(local) = statement else {
                return None;
            };
            let init = local.init.as_ref()?;
            let syn::Expr::Try(tried) = &*init.expr else {
                return None;
            };
            let syn::Expr::Call(call) = &*tried.expr else {
                return None;
            };
            matches!(&*call.func, syn::Expr::Path(path)
                if path.path.is_ident("run_first_webgpu_effect"))
            .then_some((index, call))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        sealed_statements.len(),
        1,
        "sealed adapter must be one direct fallible top-level binding",
    );
    let (sealed_index, sealed_call) = sealed_statements[0];
    let statement_surface = |statement: &syn::Stmt| {
        let mut surface = MethodSurface::default();
        surface.visit_stmt(statement);
        surface
    };
    assert!(
        push_method.block.stmts[..sealed_index]
            .iter()
            .map(statement_surface)
            .all(|surface| {
                !surface.method_calls.iter().any(|call| call == "call1")
                    && !surface
                        .function_calls
                        .iter()
                        .any(|call| call == "run_first_webgpu_effect")
            }),
        "raw WebGPU work occurs before the sealed first-effect statement",
    );
    let mut sealed_surface = MethodSurface::default();
    sealed_surface.visit_expr_call(sealed_call);
    assert_eq!(
        sealed_surface
            .function_calls
            .iter()
            .filter(|call| call.as_str() == "run_first_webgpu_effect")
            .count(),
        1,
    );
    assert!(
        !sealed_surface
            .method_calls
            .iter()
            .any(|call| call == "call1"),
        "sealed adapter arguments perform a raw call before adapter entry",
    );
    assert_eq!(
        push_method.block.stmts[sealed_index + 1..]
            .iter()
            .map(statement_surface)
            .flat_map(|surface| surface.method_calls)
            .filter(|call| call == "call1")
            .count(),
        2,
        "the remaining two error scopes must be pushed only after the sealed first effect",
    );

    for (operation, validation_method, operation_once) in [
        (
            "start_vision_stack_sharded",
            "validate_vision_stack_start",
            "start_vision_stack_sharded_once",
        ),
        (
            "run_vision_stack_sharded_layer",
            "validate_vision_stack_layer",
            "run_vision_stack_sharded_layer_once",
        ),
        (
            "finish_vision_stack_sharded",
            "validate_vision_stack_finish",
            "finish_vision_stack_sharded_once",
        ),
    ] {
        let method = unique_impl_method(&syntax, operation);
        let direct_transaction = direct_transaction_statement(method, operation);
        let mut transaction_calls = TransactionCalls::default();
        transaction_calls.visit_block(&method.block);
        assert_eq!(
            transaction_calls.count, 1,
            "{operation} must have one transaction boundary",
        );
        let arguments = direct_transaction.args.iter().collect::<Vec<_>>();
        assert_eq!(arguments.len(), 3);
        let prepared_parameter =
            exact_pre_effect_validation_closure(arguments[1], operation, validation_method);
        assert_cpu_only_validation_graph(&methods, validation_method);
        let validation = expression_surface(arguments[1]);
        let effect = expression_surface(arguments[2]);
        assert_eq!(
            validation.macro_invocations, 0,
            "{operation} validation must not hide preparation/effects in a macro",
        );
        assert_eq!(
            effect.macro_invocations, 0,
            "{operation} effect closure must not hide preparation/effects in a macro",
        );
        assert_eq!(
            validation
                .method_calls
                .iter()
                .filter(|call| call.as_str() == "prepare_vision_stack_first_error_scope")
                .count(),
            1,
            "{operation} must prepare its first raw WebGPU call before boundary marking",
        );
        assert!(
            closure_parameter_contains_ident(arguments[2], 1, &prepared_parameter),
            "{operation} effect closure does not bind the authority returned by validation",
        );
        assert!(
            effect
                .path_reads
                .iter()
                .any(|path| path == &prepared_parameter),
            "{operation} effect closure does not consume the authority returned by validation",
        );
        assert!(
            !effect
                .method_calls
                .iter()
                .chain(&effect.function_calls)
                .any(|call| call == "prepare_vision_stack_first_error_scope"),
            "{operation} moved fallible first-effect preparation after the POST marker",
        );
        let boundary_parameters = closure_parameter_idents(arguments[2], 2);
        assert_eq!(boundary_parameters.len(), 1);
        let run_effect = direct_awaited_method_call(
            closure_single_tail_expression(arguments[2], operation),
            "run_webgpu_effect",
            operation,
        );
        assert!(
            matches!(&*run_effect.receiver, syn::Expr::Path(path)
                if path.path.is_ident(&boundary_parameters[0])),
            "{operation} does not immediately cross the supplied effect boundary",
        );
        assert_eq!(run_effect.args.len(), 1);
        let post_effect_closure = &run_effect.args[0];
        let post_effect_parameters = closure_parameter_idents(post_effect_closure, 0);
        assert_eq!(post_effect_parameters.len(), 1);
        let once_call = direct_awaited_method_call(
            closure_single_tail_expression(post_effect_closure, operation_once),
            operation_once,
            operation_once,
        );
        assert!(
            matches!(&*once_call.receiver, syn::Expr::Path(path) if path.path.is_ident("self")),
            "{operation} effect stage must call this WebRuntime directly",
        );
        assert!(
            once_call
                .args
                .iter()
                .all(|argument| matches!(argument, syn::Expr::Path(_))),
            "{operation} evaluates fallible or effectful arguments after the POST marker",
        );
        for required in [&prepared_parameter, &post_effect_parameters[0]] {
            assert!(
                once_call.args.iter().any(
                    |argument| matches!(argument, syn::Expr::Path(path) if path.path.is_ident(required)),
                ),
                "{operation} does not pass {required} directly to its sealed effect stage",
            );
        }
        assert_eq!(
            effect
                .method_calls
                .iter()
                .filter(|call| call.as_str() == "run_webgpu_effect")
                .count(),
            1,
            "{operation} must cross the causal boundary exactly once",
        );
        let mut effect_method_calls = MethodCalls::default();
        effect_method_calls.visit_expr(arguments[2]);
        let operation_calls = effect_method_calls
            .calls
            .iter()
            .filter(|call| call.method == operation_once)
            .collect::<Vec<_>>();
        assert_eq!(
            operation_calls.len(),
            1,
            "{operation} must invoke its sealed effect stage exactly once",
        );
        assert!(
            operation_calls[0]
                .args
                .iter()
                .any(|argument| expression_surface(argument)
                    .path_reads
                    .iter()
                    .any(|path| path == &prepared_parameter)),
            "{operation} dead-reads the prepared authority instead of passing it to its sealed effect stage",
        );
    }

    for operation in [
        "start_vision_stack_sharded_once",
        "run_vision_stack_sharded_layer_once",
        "finish_vision_stack_sharded_once",
    ] {
        let method = unique_impl_method(&syntax, operation);
        let first = method
            .block
            .stmts
            .first()
            .expect("effectful operation must have a first statement");
        let syn::Stmt::Local(first_local) = first else {
            panic!("{operation} first statement must bind its error-scope guards");
        };
        let first_init = first_local
            .init
            .as_ref()
            .expect("first-effect guard binding must have an initializer");
        let syn::Expr::Try(first_try) = &*first_init.expr else {
            panic!("{operation} must immediately propagate first-effect failure with ?");
        };
        let syn::Expr::Await(first_await) = &*first_try.expr else {
            panic!("{operation} must await scope acquisition and partial-failure cleanup");
        };
        let syn::Expr::MethodCall(first_call) = &*first_await.base else {
            panic!("{operation} first evaluated expression must be the awaited sealed scope push");
        };
        assert_eq!(first_call.method, "push_vision_stack_error_scopes");
        assert!(
            matches!(&*first_call.receiver, syn::Expr::Path(path) if path.path.is_ident("self")),
        );
        assert!(
            first_call
                .args
                .iter()
                .all(|argument| matches!(argument, syn::Expr::Path(_))),
            "{operation} evaluates fallible or effectful scope-push arguments first",
        );
        let mut first_surface = MethodSurface::default();
        first_surface.visit_stmt(first);
        assert_eq!(
            first_surface.method_calls,
            ["push_vision_stack_error_scopes"],
            "{operation} does work before invoking its sealed prepared first effect",
        );
        assert!(first_surface.function_calls.is_empty());
        assert_eq!(first_surface.macro_invocations, 0);
        let prepared_inputs = method
            .sig
            .inputs
            .iter()
            .filter_map(|input| match input {
                syn::FnArg::Typed(typed) => {
                    let mut idents = PatternIdents::default();
                    idents.visit_pat(&typed.pat);
                    idents
                        .names
                        .into_iter()
                        .find(|name| name.contains("prepared"))
                }
                syn::FnArg::Receiver(_) => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            prepared_inputs.len(),
            1,
            "{operation} must receive exactly one prepared first-effect authority",
        );
        assert!(
            first_surface
                .path_reads
                .iter()
                .any(|path| path == &prepared_inputs[0]),
            "{operation} does not pass its prepared authority into the first raw effect",
        );
        assert!(
            first_call
                .args
                .iter()
                .any(|argument| matches!(argument, syn::Expr::Path(path)
                    if path.path.is_ident(&prepared_inputs[0]))),
            "{operation} sealed first call does not consume its prepared authority",
        );
        assert!(
            method
                .sig
                .inputs
                .iter()
                .any(|input| typed_input_contains_ident(input, &prepared_inputs[0])),
            "{operation} lost its named prepared authority",
        );
    }
}

#[test]
fn every_device_error_scope_path_shares_one_raii_admission_and_cleanup_authority() {
    let web = read_required_source("src/web.rs");
    let syntax = syn::parse_file(&web).expect("web.rs must parse");
    let structure = |name: &str| {
        let structures = syntax
            .items
            .iter()
            .filter_map(|item| match item {
                syn::Item::Struct(item) if item.ident == name => Some(item),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(structures.len(), 1, "expected one private struct {name}");
        structures[0]
    };

    let prepared_definition = structure("PreparedVisionStackFirstErrorScope");
    assert!(matches!(
        prepared_definition.vis,
        syn::Visibility::Inherited
    ));
    let prepared = named_field_shape(&prepared_definition.fields);
    assert_eq!(
        prepared,
        [
            (
                "authority".to_owned(),
                "VisionStackErrorScopeAuthority<'a,ScopeKind>".to_owned(),
            ),
            ("pop".to_owned(), "js_sys::Function".to_owned()),
            ("push".to_owned(), "js_sys::Function".to_owned()),
            ("raw_device".to_owned(), "&'a JsValue".to_owned()),
        ],
        "prepared scope authority has more than the one RAII admission and exact reflected functions",
    );
    let guards_definition = structure("BrowserVisionStackErrorScopes");
    assert!(matches!(guards_definition.vis, syn::Visibility::Inherited));
    let guards = named_field_shape(&guards_definition.fields);
    assert_eq!(
        guards,
        [
            (
                "authority".to_owned(),
                "VisionStackErrorScopeAuthority<'a,ScopeKind>".to_owned(),
            ),
            ("pop".to_owned(), "js_sys::Function".to_owned()),
            ("raw_device".to_owned(), "&'a JsValue".to_owned()),
        ],
        "live scope guard has more than the exact RAII authority and classified pop state",
    );
    for (_, ty) in prepared.iter().chain(&guards) {
        assert!(
            !ty.contains("WebRuntime") && !ty.contains("Vec<ScopeKind>"),
            "raw scope authority embeds full runtime or an untracked guard vector: {ty}",
        );
    }

    let runtime = named_field_shape(&structure("WebRuntime").fields);
    assert!(
        runtime.iter().any(|(name, ty)| {
            name == "vision_stack_error_scopes_healthy" && ty == "Cell<bool>"
        }),
        "persistent device scope health is not fail-closed in WebRuntime",
    );
    assert!(
        runtime.iter().any(|(name, ty)| {
            name == "vision_stack_error_scopes_occupied" && ty == "Cell<bool>"
        }),
        "the shared GPUDevice error-scope stack has no serialized admission state",
    );

    let methods = web_runtime_methods(&syntax);
    for removed in [
        "push_error_scopes",
        "pop_error_scopes",
        "pop_vision_stack_error_scopes",
    ] {
        assert!(
            !methods.contains_key(removed),
            "legacy unpoisoned scope helper {removed} still bypasses the common authority",
        );
    }
    assert_eq!(
        web.matches("\"pushErrorScope\"").count(),
        1,
        "pushErrorScope reflection is not centralized",
    );
    assert_eq!(
        web.matches("\"popErrorScope\"").count(),
        1,
        "popErrorScope reflection is not centralized",
    );

    let prepare = unique_impl_method(&syntax, "prepare_vision_stack_first_error_scope");
    let prepare_surface = method_surface(prepare);
    assert_eq!(
        prepare_surface
            .function_paths
            .iter()
            .filter(|call| call.as_str() == "VisionStackErrorScopeAuthority::acquire")
            .count(),
        1,
        "scope preparation does not acquire the single serialized RAII authority",
    );
    assert_eq!(
        prepare_surface
            .method_calls
            .iter()
            .filter(|call| call.as_str() == "clear_uncaptured_errors")
            .count(),
        1,
        "uncaptured errors are not cleared exactly once after admission",
    );
    let clear_owners = methods
        .values()
        .filter(|method| {
            method_surface(method)
                .method_calls
                .iter()
                .any(|call| call == "clear_uncaptured_errors")
        })
        .map(|method| method.sig.ident.to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        clear_owners,
        ["prepare_vision_stack_first_error_scope"],
        "a GPU path can mutate uncaptured-error state before health/admission succeeds",
    );

    for begin in [
        "begin_vision_stack_sharded",
        "begin_vision_stack_sharded_with_qkv_selection",
    ] {
        let method = unique_impl_method(&syntax, begin);
        let first = method.block.stmts.first().expect("begin method is empty");
        let syn::Stmt::Expr(syn::Expr::Try(propagated), Some(_)) = first else {
            panic!("{begin} must propagate the admission gate as its first statement");
        };
        let syn::Expr::Call(admission) = &*propagated.expr else {
            panic!("{begin} first statement must directly call the admission gate");
        };
        assert!(matches!(&*admission.func, syn::Expr::Path(path)
            if path.path.is_ident("require_vision_stack_error_scope_admission_available")));
        let mut first_surface = MethodSurface::default();
        first_surface.visit_stmt(first);
        assert_eq!(
            first_surface.function_calls,
            ["require_vision_stack_error_scope_admission_available"],
            "{begin} does not reject poison/interleaving before busy/session work",
        );
        assert!(
            first_surface
                .method_calls
                .iter()
                .all(|call| matches!(call.as_str(), "to_owned" | "into"))
        );
        assert!(!first_surface.field_reads.iter().any(|field| matches!(
            field.as_str(),
            "execution_busy" | "vision_stack_session" | "uncaptured_errors"
        )));
    }

    let push = unique_impl_method(&syntax, "push_vision_stack_error_scopes");
    assert!(
        push.sig.asyncness.is_some(),
        "partial push cleanup must be awaited before returning the terminal error",
    );
    let push_surface = method_surface(push);
    assert_eq!(
        push_surface
            .function_calls
            .iter()
            .filter(|call| call.as_str() == "push_vision_stack_error_scope_or_drain")
            .count(),
        2,
        "second and third scope pushes must both use the behavior-tested cleanup transition",
    );
    assert_eq!(
        push_surface
            .function_calls
            .iter()
            .filter(|call| call.as_str() == "pop_browser_vision_stack_error_scope")
            .count(),
        2,
        "both partial-push cleanup paths must use the classified browser pop adapter",
    );
    assert!(
        !push_surface
            .method_calls
            .iter()
            .any(|call| call == "raw_device_method"),
        "POST scope push reacquires fallible reflection authority",
    );

    let generic_push = unique_impl_method(&syntax, "push_browser_error_scopes");
    assert!(generic_push.sig.asyncness.is_some());
    let generic_push_surface = method_surface(generic_push);
    assert_eq!(
        generic_push_surface
            .function_calls
            .iter()
            .filter(|call| call.as_str() == "push_vision_stack_error_scope_or_drain")
            .count(),
        3,
        "generic GPU paths do not push all three scopes through partial-cleanup authority",
    );

    #[derive(Default)]
    struct EagerControlFlow {
        awaits: usize,
        macros: usize,
        returns: usize,
        tries: usize,
    }
    impl<'ast> Visit<'ast> for EagerControlFlow {
        fn visit_expr_async(&mut self, _expression: &'ast syn::ExprAsync) {}

        fn visit_expr_closure(&mut self, _expression: &'ast syn::ExprClosure) {}

        fn visit_expr_await(&mut self, expression: &'ast syn::ExprAwait) {
            self.awaits += 1;
            syn::visit::visit_expr_await(self, expression);
        }

        fn visit_macro(&mut self, invocation: &'ast syn::Macro) {
            self.macros += 1;
            syn::visit::visit_macro(self, invocation);
        }

        fn visit_expr_return(&mut self, expression: &'ast syn::ExprReturn) {
            self.returns += 1;
            syn::visit::visit_expr_return(self, expression);
        }

        fn visit_expr_try(&mut self, expression: &'ast syn::ExprTry) {
            self.tries += 1;
            syn::visit::visit_expr_try(self, expression);
        }
    }

    let complete = unique_impl_method(&syntax, "complete_browser_error_scoped_operation");
    assert!(complete.sig.asyncness.is_some());
    let guard_inputs = complete
        .sig
        .inputs
        .iter()
        .filter_map(|input| match input {
            syn::FnArg::Typed(typed)
                if type_shape(&typed.ty).starts_with("BrowserVisionStackErrorScopes<") =>
            {
                Some(plain_binding(&typed.pat, "common completion guard"))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        guard_inputs,
        ["guards"],
        "common completion must own exactly one named live scope guard",
    );
    assert_eq!(
        complete
            .sig
            .inputs
            .iter()
            .filter(|input| typed_input_contains_ident(input, "operation"))
            .count(),
        1,
        "common completion must receive exactly one named lazy operation",
    );
    let complete_surface = method_surface(complete);
    assert_eq!(
        complete_surface
            .function_calls
            .iter()
            .filter(|call| call.as_str() == "run_vision_stack_error_scoped_operation")
            .count(),
        1,
        "normal/error completion bypasses the behavior-tested drain-before-return transition",
    );
    assert_eq!(
        complete_surface
            .function_calls
            .iter()
            .filter(|call| call.as_str() == "pop_browser_vision_stack_error_scope")
            .count(),
        1,
        "common completion does not classify every raw pop",
    );
    let syn::Stmt::Local(completed) = complete
        .block
        .stmts
        .first()
        .expect("common completion omitted the causal runner")
    else {
        panic!("common completion must bind its causal runner as the first statement");
    };
    let completed_initializer = completed
        .init
        .as_ref()
        .expect("common completion causal runner omitted its initializer");
    let syn::Expr::Await(awaited) = &*completed_initializer.expr else {
        panic!("common completion must directly await its causal runner first");
    };
    let syn::Expr::Call(run_scoped) = &*awaited.base else {
        panic!("common completion must directly call its causal runner first");
    };
    assert!(matches!(&*run_scoped.func, syn::Expr::Path(path)
        if path.path.is_ident("run_vision_stack_error_scoped_operation")));
    assert_eq!(
        run_scoped.args.len(),
        3,
        "common completion causal runner must receive authority, operation, and pop adapter only",
    );
    assert!(
        run_scoped.args.first().is_some_and(|argument| {
            matches!(argument, syn::Expr::Field(field)
                if matches!(&*field.base, syn::Expr::Path(path) if path.path.is_ident("guards"))
                    && matches!(&field.member, syn::Member::Named(member) if member == "authority"))
        }),
        "common completion does not transfer the exact caller-owned live scope authority",
    );
    assert!(
        matches!(&run_scoped.args[1], syn::Expr::Path(path) if path.path.is_ident("operation")),
        "common completion substitutes the caller's lazy operation",
    );
    assert!(
        matches!(&run_scoped.args[2], syn::Expr::Closure(_)),
        "common completion must defer raw pop classification to its causal runner",
    );
    for argument in &run_scoped.args {
        let mut eager = EagerControlFlow::default();
        eager.visit_expr(argument);
        assert_eq!(
            (eager.awaits, eager.macros, eager.returns, eager.tries),
            (0, 0, 0, 0),
            "common completion evaluates fallible control flow before transferring authority to its causal runner",
        );
    }

    for (scoped_root, push_method) in [
        (
            "start_vision_stack_sharded_once",
            "push_vision_stack_error_scopes",
        ),
        (
            "run_vision_stack_sharded_layer_once",
            "push_vision_stack_error_scopes",
        ),
        (
            "finish_vision_stack_sharded_once",
            "push_vision_stack_error_scopes",
        ),
        ("run_projector_scoped", "push_browser_error_scopes"),
        ("run_vision_layer_scoped", "push_browser_error_scopes"),
        ("validate_pipeline_source", "push_browser_error_scopes"),
        ("run_source", "push_browser_error_scopes"),
        ("probe_validation_error_json", "push_browser_error_scopes"),
    ] {
        let method = unique_impl_method(&syntax, scoped_root);
        let mut push_positions = Vec::new();
        let mut completion_positions = Vec::new();
        for (index, statement) in method.block.stmts.iter().enumerate() {
            let mut surface = MethodSurface::default();
            surface.visit_stmt(statement);
            if surface.method_calls.iter().any(|call| call == push_method) {
                push_positions.push(index);
            }
            if surface
                .method_calls
                .iter()
                .any(|call| call == "complete_browser_error_scoped_operation")
            {
                completion_positions.push(index);
            }
        }
        assert_eq!(
            push_positions.len(),
            1,
            "{scoped_root} must establish exactly one {push_method} authority",
        );
        assert_eq!(
            completion_positions.len(),
            1,
            "{scoped_root} must transfer exactly one operation to common scoped completion",
        );
        let push_index = push_positions[0];
        let completion_index = completion_positions[0];
        assert_eq!(
            completion_index,
            push_index + 1,
            "{scoped_root} permits an early exit between push and scoped completion transfer",
        );

        let syn::Stmt::Local(pushed) = &method.block.stmts[push_index] else {
            panic!("{scoped_root} must bind its pushed authority directly");
        };
        let authority_binding = plain_binding(&pushed.pat, scoped_root);
        let pushed_initializer = pushed
            .init
            .as_ref()
            .expect("pushed authority binding omitted its initializer");
        let syn::Expr::Try(propagated_push) = &*pushed_initializer.expr else {
            panic!("{scoped_root} must propagate its direct push failure with ?");
        };
        let syn::Expr::Await(awaited_push) = &*propagated_push.expr else {
            panic!("{scoped_root} must directly await its scope push");
        };
        let syn::Expr::MethodCall(push_call) = &*awaited_push.base else {
            panic!("{scoped_root} must directly invoke its scope push");
        };
        assert_eq!(push_call.method, push_method);
        assert!(matches!(&*push_call.receiver, syn::Expr::Path(path)
            if path.path.is_ident("self")));

        let syn::Stmt::Local(completed) = &method.block.stmts[completion_index] else {
            panic!("{scoped_root} must bind scoped completion immediately after push");
        };
        let mut completion = &*completed
            .init
            .as_ref()
            .expect("scoped completion binding omitted its initializer")
            .expr;
        if let syn::Expr::Try(propagated) = completion {
            completion = &propagated.expr;
        }
        let syn::Expr::Await(awaited) = completion else {
            panic!("{scoped_root} must await scoped completion before any later exit");
        };
        let syn::Expr::MethodCall(call) = &*awaited.base else {
            panic!("{scoped_root} must directly invoke scoped completion");
        };
        assert_eq!(call.method, "complete_browser_error_scoped_operation");
        assert!(matches!(&*call.receiver, syn::Expr::Path(path) if path.path.is_ident("self")));
        assert_eq!(
            call.args.len(),
            2,
            "{scoped_root} must transfer only its exact guard and one lazy operation",
        );
        assert!(
            call.args
                .first()
                .is_some_and(|argument| matches!(argument, syn::Expr::Path(path)
                    if path.path.is_ident(&authority_binding)),),
            "{scoped_root} does not transfer its exact pushed authority to common completion",
        );
        assert!(
            matches!(&call.args[1], syn::Expr::Async(_)),
            "{scoped_root} must defer every post-push operation inside one explicit async block",
        );
        for argument in &call.args {
            let mut eager = EagerControlFlow::default();
            eager.visit_expr(argument);
            assert_eq!(
                (eager.awaits, eager.macros, eager.returns, eager.tries),
                (0, 0, 0, 0),
                "{scoped_root} evaluates fallible control flow before transferring the operation future to scoped completion",
            );
        }
    }

    let adapters = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(function)
                if function.sig.ident == "pop_browser_vision_stack_error_scope" =>
            {
                Some(function)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        adapters.len(),
        1,
        "expected one private browser pop adapter"
    );
    assert!(matches!(adapters[0].vis, syn::Visibility::Inherited));
    assert!(adapters[0].sig.asyncness.is_some());
    let mut adapter_surface = MethodSurface::default();
    adapter_surface.visit_block(&adapters[0].block);
    assert_eq!(
        adapter_surface
            .function_calls
            .iter()
            .filter(|call| call.as_str() == "observe_vision_stack_error_scope_pop")
            .count(),
        1,
        "browser pop stages bypass the behavior-tested classification adapter",
    );
}

#[test]
fn sharded_session_pipeline_construction_never_mutates_the_runtime_global_cache() {
    let web = read_required_source("src/web.rs");
    let syntax = syn::parse_file(&web).expect("web.rs must parse");
    let methods = web_runtime_methods(&syntax);
    let reachable = reachable_web_runtime_methods(&methods, "vision_stack_pipeline");
    for method in reachable {
        let surface = method_surface(method);
        assert!(
            !surface.field_reads.iter().any(|field| field == "pipelines"),
            "vision-stack pipeline graph reaches persistent cache field through {}",
            method.sig.ident,
        );
        assert!(
            !surface
                .method_calls
                .iter()
                .any(|call| matches!(call.as_str(), "borrow" | "borrow_mut" | "insert")),
            "vision-stack pipeline graph can leak a partially validated cache entry through {}",
            method.sig.ident,
        );
    }

    let pipeline = unique_impl_method(&syntax, "vision_stack_pipeline");
    assert!(
        !pipeline
            .sig
            .inputs
            .iter()
            .filter_map(|input| match input {
                syn::FnArg::Typed(typed) => match &*typed.pat {
                    syn::Pat::Ident(ident) => Some(ident.ident.to_string()),
                    _ => None,
                },
                syn::FnArg::Receiver(_) => None,
            })
            .any(|input| input == "cached_kernel"),
        "session-local pipeline construction still accepts a persistent-cache key",
    );
    let surface = method_surface(pipeline);
    assert_eq!(
        surface.macro_invocations, 0,
        "vision_stack_pipeline must not hide persistent state access in a macro",
    );
    assert!(
        surface.field_reads.iter().all(|field| field == "device"),
        "vision_stack_pipeline may only read the WebGPU device, not persistent runtime state: {:?}",
        surface.field_reads,
    );
    for call in &surface.function_calls {
        assert!(
            matches!(call.as_str(), "run_first_webgpu_effect" | "Ok" | "Err"),
            "vision_stack_pipeline can escape its sealed local builder through {call}",
        );
    }
    for call in &surface.method_calls {
        assert!(
            matches!(call.as_str(), "map_err" | "to_string" | "to_owned"),
            "vision_stack_pipeline can escape its sealed local builder through .{call}()",
        );
    }
}

#[test]
fn qkv_physical_effect_sink_carries_only_narrow_allocation_authority() {
    let web = read_required_source("src/web.rs");
    let syntax = syn::parse_file(&web).expect("web.rs must parse");
    let structure = |name: &str| {
        let matches = syntax
            .items
            .iter()
            .filter_map(|item| match item {
                syn::Item::Struct(item) if item.ident == name => Some(item),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1, "expected one private struct {name}");
        matches[0]
    };

    let allocation = structure("BrowserVisionQkvAllocationAuthority");
    assert!(matches!(allocation.vis, syn::Visibility::Inherited));
    let allocation_fields = named_field_shape(&allocation.fields);
    assert_eq!(
        allocation_fields,
        [
            ("buffer_allocations".to_owned(), "&'a Cell<u64>".to_owned(),),
            ("device".to_owned(), "&'a wgpu::Device".to_owned()),
            ("queue".to_owned(), "&'a wgpu::Queue".to_owned()),
        ],
        "QKV allocation authority is not the exact typed device/queue/counter projection",
    );
    assert!(
        allocation_fields
            .iter()
            .all(|(_, ty)| !ty.contains("WebRuntime") && !ty.contains("pipelines")),
        "QKV allocation authority embeds full runtime or persistent pipeline authority",
    );

    let sink = structure("BrowserVisionQkvPhysicalCommandEffectSink");
    assert!(matches!(sink.vis, syn::Visibility::Inherited));
    let sink_fields = named_field_shape(&sink.fields);
    assert_eq!(
        sink_fields,
        [
            (
                "allocation".to_owned(),
                "BrowserVisionQkvAllocationAuthority<'a>".to_owned(),
            ),
            (
                "bind_groups".to_owned(),
                "&'b mut BrowserVisionQkvLayerBindGroups".to_owned(),
            ),
            (
                "context".to_owned(),
                "&'a BrowserVisionQkvLayerResolutionContext<'a>".to_owned(),
            ),
            (
                "storage".to_owned(),
                "&'b mut BrowserVisionQkvPhysicalStorage".to_owned(),
            ),
        ],
        "typed QKV effect sink retained authority outside its exact allocation/context/stores",
    );
    assert!(
        sink_fields
            .iter()
            .all(|(_, ty)| !ty.contains("WebRuntime") && !ty.contains("pipelines")),
        "typed QKV effect sink embeds full runtime or persistent pipeline authority",
    );
}

#[test]
fn allocator_reachable_graph_cannot_publish_persistent_pipeline_state() {
    let web = read_required_source("src/web.rs");
    let syntax = syn::parse_file(&web).expect("web.rs must parse");
    let methods = web_runtime_methods(&syntax);
    let allocator = unique_impl_method(&syntax, "allocate_vision_stack_gpu");
    let allocator_graph = reachable_web_runtime_methods(&methods, "allocate_vision_stack_gpu");
    for method in allocator_graph {
        let surface = method_surface(method);
        assert_eq!(
            surface.macro_invocations, 0,
            "allocator graph hides persistent-cache authority in a macro through {}",
            method.sig.ident,
        );
        assert!(
            !surface.field_reads.iter().any(|field| field == "pipelines"),
            "allocator graph reaches persistent pipeline cache through {}",
            method.sig.ident,
        );
        assert!(
            !surface
                .method_calls
                .iter()
                .chain(&surface.function_calls)
                .any(|call| call == "pipeline"),
            "allocator graph bypasses the session-local builder through {}",
            method.sig.ident,
        );
        let mut escapes = FullRuntimeAuthorityEscapes::default();
        escapes.visit_block(&method.block);
        assert!(
            escapes.free_calls.is_empty(),
            "allocator graph exports full WebRuntime authority to a free helper through {}: {:?}",
            method.sig.ident,
            escapes.free_calls,
        );
        let mut captures = FullRuntimeAuthorityCaptures::default();
        captures.visit_block(&method.block);
        assert!(
            captures.struct_fields.is_empty(),
            "allocator graph stores full WebRuntime authority in a struct through {}: {:?}",
            method.sig.ident,
            captures.struct_fields,
        );
    }
    assert_eq!(
        method_surface(allocator).macro_invocations,
        0,
        "allocator must not hide pipeline collection or publication in a macro",
    );
    let pipeline_initializers = allocator
        .block
        .stmts
        .iter()
        .filter_map(|statement| match statement {
            syn::Stmt::Local(local) if pattern_contains_ident(&local.pat, "pipelines") => {
                local.init.as_ref().map(|init| &*init.expr)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        pipeline_initializers.len(),
        1,
        "real Web allocator must have one causally named session pipeline map",
    );
    let syn::Expr::Try(collected) = pipeline_initializers[0] else {
        panic!("session pipeline binding must propagate atomic collector failure with ?");
    };
    let syn::Expr::Call(collector_call) = &*collected.expr else {
        panic!("session pipeline binding must be the direct atomic collector result");
    };
    let syn::Expr::Path(collector) = &*collector_call.func else {
        panic!("session pipeline binding must call the named atomic collector directly");
    };
    assert!(
        collector
            .path
            .is_ident("collect_vision_stack_session_resources"),
        "real Web allocator pipeline binding must come directly from the behavior-tested atomic local collector",
    );
    assert_eq!(
        collector_call.args.len(),
        2,
        "atomic collector must receive prepared specifications and one sealed builder",
    );
    let syn::Expr::Closure(builder) = &collector_call.args[1] else {
        panic!("atomic collector builder must be an explicit closure");
    };
    let syn::Expr::MethodCall(build_call) = &*builder.body else {
        panic!("atomic collector builder body must directly call vision_stack_pipeline");
    };
    assert_eq!(
        build_call.method, "vision_stack_pipeline",
        "atomic collector bypasses the clean session-local pipeline builder",
    );
    assert!(
        matches!(&*build_call.receiver, syn::Expr::Path(path) if path.path.is_ident("self")),
        "atomic collector pipeline builder must use this WebRuntime's clean builder",
    );
    let builder_surface = expression_surface(&collector_call.args[1]);
    assert_eq!(builder_surface.macro_invocations, 0);
    assert_eq!(
        builder_surface.method_calls,
        ["vision_stack_pipeline"],
        "atomic collector closure must perform only the clean pipeline build",
    );
    assert!(builder_surface.function_calls.is_empty());
    assert!(builder_surface.field_reads.is_empty());
    let mut publications = PipelineStatePublications::default();
    publications.visit_block(&allocator.block);
    assert_eq!(
        publications.gpu_states, 1,
        "allocator must construct exactly one BrowserVisionStackGpuState",
    );
    assert_eq!(
        publications.pipeline_fields, 1,
        "the one GPU state must contain exactly one pipeline field",
    );
    assert_eq!(
        publications.exact_bindings, 1,
        "allocator must publish the exact successfully collected local pipeline map into session state",
    );
}
