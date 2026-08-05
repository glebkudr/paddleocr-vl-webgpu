//! Host-side M7c2b API, ownership, and source contract for optimized Web sessions.
//!
//! The wasm implementation is source-inspected on the host while the generic
//! generation/lease transitions are exercised as real Rust code. Browser
//! execution remains the M7c2c gate.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

use syn::ext::IdentExt;

use pvlc_ir::SemanticGraph;
use pvlc_passes::{VisionQkvPhysicalExecutionSpec, VisionQkvStackOverlayErrorCode};
use pvlc_runtime_core::{
    KernelId, VisionEncoderLayerStage, VisionQkvCanaryKind, VisionQkvExecutionPolicy,
    VisionQkvFusedTargetLimits, VisionQkvSelectionOutcome, VisionStackActivationLayout,
    VisionStackActivationStrategy, VisionStackScratchAllocation,
};
use pvlc_runtime_web::{
    BrowserVisionQkvBeginExecutionEvidence, BrowserVisionQkvExecutionEvidencePlan,
    BrowserVisionQkvFinalExecutionEvidence, VisionQkvCompilerCapabilities,
    VisionQkvCompilerHandoffErrorCode, VisionQkvCompilerReadbackRequest, VisionQkvWebBindGroupKind,
    VisionQkvWebBindingResource, VisionQkvWebPhysicalBuffer, VisionQkvWebPhysicalCommand,
    VisionQkvWebPhysicalCommandEffectSink, VisionQkvWebPhysicalCommandPhase,
    VisionQkvWebPhysicalCommandPlan, VisionStackMemoryHardening, VisionStackMemoryHardeningPlan,
    build_vision_qkv_selection_evidence_propagation, build_vision_stack_legacy_diagnostics_record,
    build_vision_stack_legacy_status_record, compile_vision_qkv_stack_handoff,
    execute_vision_qkv_web_physical_commands, plan_vision_qkv_web_physical_commands,
    prepare_vision_qkv_stack_handoff_execution, serialize_vision_stack_legacy_diagnostics_json,
    serialize_vision_stack_legacy_status_json, serialize_vision_stack_qkv_begin_status_json,
    serialize_vision_stack_qkv_final_diagnostics_json,
    validate_vision_qkv_web_physical_command_dispatches, with_vision_stack_mapped_readback,
};

use pvlc_model_schema::PaddleOcrVl16Schema;
use pvlc_pack::{
    VisionStackShardManifest, VisionStackShardOracle, VisionStackShardPlan,
    VisionStackShardProtocolPhase, canonical_vision_stack_shard_manifest_bytes,
    parse_vision_stack_shard_manifest,
};

const WEB_SOURCE: &str = include_str!("../src/web.rs");
const LIB_SOURCE: &str = include_str!("../src/lib.rs");
const WEB_MANIFEST: &str = include_str!("../Cargo.toml");
const NATIVE_SOURCE: &str = include_str!("../../pvlc-runtime-native/src/lib.rs");
const CORE_SOURCE: &str = include_str!("../../pvlc-runtime-core/src/lib.rs");
const PASSES_SOURCE: &str = include_str!("../../pvlc-passes/src/vision_qkv_stack.rs");
const SYNTHETIC_MANIFEST_BYTES: &[u8] =
    include_bytes!("../../../web/runner/data/m3-vision-stack-sharded/manifest.json");
const OFFICIAL_MANIFEST_BYTES: &[u8] =
    include_bytes!("../../../web/runner/data/m3-vision-stack-sharded-official/manifest.json");

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect()
}

fn value_constructor_occurrences(source: &str, type_name: &str) -> usize {
    let marker = format!("{type_name} {{");
    source
        .match_indices(&marker)
        .filter(|(start, _)| {
            let prefix = source[..*start].trim_end();
            let declaration_start = [';', '{', '}']
                .into_iter()
                .filter_map(|boundary| source[..*start].rfind(boundary))
                .max()
                .map_or(0, |index| index + 1);
            let declaration_prefix = &source[declaration_start..*start];
            if declaration_prefix.contains("->") && !declaration_prefix.contains('=') {
                return false;
            }
            let preceding = prefix
                .rsplit(|character: char| !(character == '_' || character.is_ascii_alphanumeric()))
                .find(|token| !token.is_empty());
            !matches!(preceding, Some("struct" | "impl"))
        })
        .count()
}

fn mask_non_newline(bytes: &mut [u8], start: usize, end: usize) {
    for byte in &mut bytes[start..end] {
        if *byte != b'\n' && *byte != b'\r' {
            *byte = b' ';
        }
    }
}

fn mask_rust_string_literal(
    target: &mut [u8],
    opening_quote: usize,
    end: usize,
    closing_delimiter_bytes: usize,
) {
    let content_start = opening_quote + 1;
    let content_end = end
        .checked_sub(closing_delimiter_bytes)
        .expect("closed Rust string delimiter must fit inside its span");
    assert!(content_start <= content_end);
    mask_non_newline(target, content_start, content_end);
}

fn mask_rust_char_literal(target: &mut [u8], source: &[u8], start: usize, end: usize) {
    mask_non_newline(target, start, end);
    let replacement: &[u8] = if source.get(start) == Some(&b'b') {
        b"b'x'"
    } else {
        b"'x'"
    };
    target[start..start + replacement.len()].copy_from_slice(replacement);
}

fn rust_raw_string_span(source: &[u8], start: usize) -> Option<(usize, usize, usize)> {
    let mut cursor = start;
    if source.get(cursor) == Some(&b'b') {
        cursor += 1;
    }
    if source.get(cursor) != Some(&b'r') {
        return None;
    }
    cursor += 1;
    let hashes_start = cursor;
    while source.get(cursor) == Some(&b'#') {
        cursor += 1;
    }
    if source.get(cursor) != Some(&b'"') {
        return None;
    }
    let opening_quote = cursor;
    let hashes = cursor - hashes_start;
    cursor += 1;
    while cursor < source.len() {
        if source[cursor] == b'"'
            && source.get(cursor + 1..cursor + 1 + hashes)
                == Some(&source[hashes_start..hashes_start + hashes])
        {
            return Some((opening_quote, cursor + 1 + hashes, 1 + hashes));
        }
        cursor += 1;
    }
    None
}

fn rust_string_end(source: &[u8], quote: usize) -> Option<usize> {
    let mut cursor = quote + 1;
    let mut escaped = false;
    while cursor < source.len() {
        let byte = source[cursor];
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if byte == b'"' {
            return Some(cursor + 1);
        }
        cursor += 1;
    }
    None
}

fn rust_char_end(source: &[u8], quote: usize) -> Option<usize> {
    let content = quote + 1;
    let mut cursor = content;
    match *source.get(cursor)? {
        b'\n' | b'\r' | b'\'' => return None,
        b'\\' => {
            cursor += 1;
            match *source.get(cursor)? {
                b'x' => {
                    let digits = source.get(cursor + 1..cursor + 3)?;
                    if !digits.iter().all(u8::is_ascii_hexdigit) {
                        return None;
                    }
                    cursor += 3;
                }
                b'u' => {
                    if source.get(cursor + 1) != Some(&b'{') {
                        return None;
                    }
                    cursor += 2;
                    let digits_start = cursor;
                    let mut hexadecimal_digits = 0_usize;
                    while let Some(byte) = source.get(cursor) {
                        match byte {
                            b'}' => break,
                            b'_' => cursor += 1,
                            byte if byte.is_ascii_hexdigit() => {
                                hexadecimal_digits += 1;
                                cursor += 1;
                            }
                            _ => return None,
                        }
                    }
                    if source.get(cursor) != Some(&b'}')
                        || hexadecimal_digits == 0
                        || hexadecimal_digits > 6
                        || cursor == digits_start
                    {
                        return None;
                    }
                    cursor += 1;
                }
                b'n' | b'r' | b't' | b'\\' | b'0' | b'\'' | b'"' => cursor += 1,
                _ => return None,
            }
        }
        first => {
            let width = if first.is_ascii() {
                1
            } else {
                let tail = std::str::from_utf8(&source[cursor..]).ok()?;
                tail.chars().next()?.len_utf8()
            };
            std::str::from_utf8(source.get(cursor..cursor + width)?).ok()?;
            cursor += width;
        }
    }
    (source.get(cursor) == Some(&b'\'')).then_some(cursor + 1)
}

fn mask_dead_if_false_blocks(source: &mut [u8]) {
    let mut cursor = 0_usize;
    while cursor + 2 <= source.len() {
        if source.get(cursor..cursor + 2) != Some(b"if")
            || cursor > 0
                && (source[cursor - 1] == b'_' || source[cursor - 1].is_ascii_alphanumeric())
            || source
                .get(cursor + 2)
                .is_some_and(|byte| *byte == b'_' || byte.is_ascii_alphanumeric())
        {
            cursor += 1;
            continue;
        }
        let mut probe = cursor + 2;
        while source.get(probe).is_some_and(u8::is_ascii_whitespace) {
            probe += 1;
        }
        let parenthesized = source.get(probe) == Some(&b'(');
        if parenthesized {
            probe += 1;
            while source.get(probe).is_some_and(u8::is_ascii_whitespace) {
                probe += 1;
            }
        }
        if source.get(probe..probe + "false".len()) != Some(b"false") {
            cursor += 2;
            continue;
        }
        probe += "false".len();
        while source.get(probe).is_some_and(u8::is_ascii_whitespace) {
            probe += 1;
        }
        if parenthesized {
            if source.get(probe) != Some(&b')') {
                cursor += 2;
                continue;
            }
            probe += 1;
            while source.get(probe).is_some_and(u8::is_ascii_whitespace) {
                probe += 1;
            }
        }
        if source.get(probe) != Some(&b'{') {
            cursor += 2;
            continue;
        }
        let mut depth = 0_usize;
        let mut end = probe;
        while end < source.len() {
            match source[end] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end += 1;
                        break;
                    }
                }
                _ => {}
            }
            end += 1;
        }
        mask_non_newline(source, cursor, end);
        cursor = end;
    }
}

fn live_rust_source(source: &str) -> String {
    let original = source.as_bytes();
    let mut live = original.to_vec();
    let mut cursor = 0_usize;
    while cursor < original.len() {
        if original.get(cursor..cursor + 2) == Some(b"//") {
            let end = original[cursor..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(original.len(), |offset| cursor + offset);
            mask_non_newline(&mut live, cursor, end);
            cursor = end;
            continue;
        }
        if original.get(cursor..cursor + 2) == Some(b"/*") {
            let mut depth = 1_usize;
            let mut end = cursor + 2;
            while end < original.len() && depth > 0 {
                if original.get(end..end + 2) == Some(b"/*") {
                    depth += 1;
                    end += 2;
                } else if original.get(end..end + 2) == Some(b"*/") {
                    depth -= 1;
                    end += 2;
                } else {
                    end += 1;
                }
            }
            mask_non_newline(&mut live, cursor, end);
            cursor = end;
            continue;
        }
        if let Some((opening_quote, end, closing_delimiter_bytes)) =
            rust_raw_string_span(original, cursor)
        {
            mask_rust_string_literal(&mut live, opening_quote, end, closing_delimiter_bytes);
            cursor = end;
            continue;
        }
        let string_quote = match original[cursor] {
            b'"' => Some(cursor),
            b'b' if original.get(cursor + 1) == Some(&b'"') => Some(cursor + 1),
            _ => None,
        };
        if let Some(quote) = string_quote
            && let Some(end) = rust_string_end(original, quote)
        {
            mask_rust_string_literal(&mut live, quote, end, 1);
            cursor = end;
            continue;
        }
        let char_quote = match original[cursor] {
            b'\'' => Some(cursor),
            b'b' if original.get(cursor + 1) == Some(&b'\'') => Some(cursor + 1),
            _ => None,
        };
        if let Some(quote) = char_quote
            && let Some(end) = rust_char_end(original, quote)
        {
            mask_rust_char_literal(&mut live, original, cursor, end);
            cursor = end;
            continue;
        }
        cursor += 1;
    }
    mask_dead_if_false_blocks(&mut live);
    String::from_utf8(live).expect("Rust source masking must preserve UTF-8")
}

fn occurrences(source: &str, needle: &str) -> usize {
    source.match_indices(needle).count()
}

fn source_call_occurrences_named(source: &str, function_name: &str) -> usize {
    source
        .match_indices(function_name)
        .filter(|(start, _)| {
            let start = *start;
            let declaration_prefix = source[..start].trim_end();
            let is_declaration = declaration_prefix.strip_suffix("fn").is_some_and(|before| {
                before
                    .chars()
                    .next_back()
                    .is_none_or(|character| character != '_' && !character.is_ascii_alphanumeric())
            });
            let has_start_boundary = source[..start]
                .chars()
                .next_back()
                .is_none_or(|character| character != '_' && !character.is_ascii_alphanumeric());
            let mut cursor = start + function_name.len();
            let has_end_boundary = source[cursor..]
                .chars()
                .next()
                .is_none_or(|character| character != '_' && !character.is_ascii_alphanumeric());
            while source
                .as_bytes()
                .get(cursor)
                .is_some_and(u8::is_ascii_whitespace)
            {
                cursor += 1;
            }
            !is_declaration
                && has_start_boundary
                && has_end_boundary
                && source.as_bytes().get(cursor) == Some(&b'(')
        })
        .count()
}

fn source_calls_named(source: &str, function_name: &str) -> bool {
    source_call_occurrences_named(source, function_name) != 0
}

fn assert_json_recursively_qkv_free(value: &serde_json::Value, path: &str) {
    match value {
        serde_json::Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                assert_json_recursively_qkv_free(value, &format!("{path}[{index}]"));
            }
        }
        serde_json::Value::Object(fields) => {
            for (key, value) in fields {
                assert!(
                    !key.to_ascii_lowercase().contains("qkv"),
                    "legacy production serializer leaked QKV key {path}.{key}",
                );
                assert_json_recursively_qkv_free(value, &format!("{path}.{key}"));
            }
        }
        _ => {}
    }
}

fn braced_item<'a>(source: &'a str, needle: &str) -> &'a str {
    let start = source
        .find(needle)
        .unwrap_or_else(|| panic!("missing reviewed source item {needle:?}"));
    let brace = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("source item {needle:?} has no body"));
    let mut depth = 0_usize;
    for (offset, byte) in source.as_bytes()[brace..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=brace + offset];
                }
            }
            _ => {}
        }
    }
    panic!("source item {needle:?} has an unterminated body")
}

fn bracketed_item<'a>(source: &'a str, needle: &str) -> &'a str {
    let start = source
        .find(needle)
        .unwrap_or_else(|| panic!("missing reviewed source item {needle:?}"));
    let equals = source[start..]
        .find('=')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("source item {needle:?} has no equals sign"));
    let bracket = source[equals..]
        .find('[')
        .map(|offset| equals + offset)
        .unwrap_or_else(|| panic!("source item {needle:?} has no array"));
    let mut depth = 0_usize;
    for (offset, byte) in source.as_bytes()[bracket..].iter().enumerate() {
        match byte {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=bracket + offset];
                }
            }
            _ => {}
        }
    }
    panic!("source item {needle:?} has an unterminated array")
}

fn balanced_call<'a>(source: &'a str, needle: &str) -> (Vec<&'a str>, usize) {
    let start = source
        .find(needle)
        .unwrap_or_else(|| panic!("missing reviewed call {needle:?}"));
    let open = source[start..]
        .find('(')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("reviewed call {needle:?} has no arguments"));
    let bytes = source.as_bytes();
    let mut arguments = Vec::new();
    let mut argument_start = open + 1;
    let mut parentheses = 0_usize;
    let mut braces = 0_usize;
    let mut brackets = 0_usize;
    let mut string = false;
    let mut escaped = false;
    let mut line_comment = false;
    let mut block_comments = 0_usize;
    let mut index = open + 1;
    while index < bytes.len() {
        let byte = bytes[index];
        let next = bytes.get(index + 1).copied();
        if line_comment {
            line_comment = byte != b'\n';
            index += 1;
            continue;
        }
        if block_comments > 0 {
            if byte == b'/' && next == Some(b'*') {
                block_comments += 1;
                index += 2;
            } else if byte == b'*' && next == Some(b'/') {
                block_comments -= 1;
                index += 2;
            } else {
                index += 1;
            }
            continue;
        }
        if string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                string = false;
            }
            index += 1;
            continue;
        }
        if byte == b'/' && next == Some(b'/') {
            line_comment = true;
            index += 2;
            continue;
        }
        if byte == b'/' && next == Some(b'*') {
            block_comments = 1;
            index += 2;
            continue;
        }
        match byte {
            b'"' => string = true,
            b'(' => parentheses += 1,
            b')' if parentheses == 0 && braces == 0 && brackets == 0 => {
                let final_argument = source[argument_start..index].trim();
                if !final_argument.is_empty() {
                    arguments.push(final_argument);
                }
                return (arguments, index + 1);
            }
            b')' => parentheses -= 1,
            b'{' => braces += 1,
            b'}' => braces -= 1,
            b'[' => brackets += 1,
            b']' => brackets -= 1,
            b',' if parentheses == 0 && braces == 0 && brackets == 0 => {
                arguments.push(source[argument_start..index].trim());
                argument_start = index + 1;
            }
            _ => {}
        }
        index += 1;
    }
    panic!("reviewed call {needle:?} has unbalanced arguments")
}

fn balanced_call_arguments<'a>(source: &'a str, needle: &str) -> Vec<&'a str> {
    balanced_call(source, needle).0
}

fn all_balanced_call_arguments<'a>(source: &'a str, needle: &str) -> Vec<Vec<&'a str>> {
    let mut calls = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = source[cursor..].find(needle) {
        let start = cursor + relative;
        let (arguments, end) = balanced_call(&source[start..], needle);
        calls.push(arguments);
        cursor = start + end;
    }
    calls
}

#[derive(Clone, Copy, Debug)]
struct DirectCallBinding<'a> {
    name: &'a str,
    call_start: usize,
    call_end: usize,
    statement_end: usize,
}

fn direct_call_binding<'a>(
    body: &'a str,
    needle: &str,
    allow_separate_error_propagation: bool,
) -> DirectCallBinding<'a> {
    let binding = call_initializer_binding(body, needle);
    let tail = compact(&body[binding.call_end..binding.statement_end]);
    let directly_propagated =
        tail == "?" || (tail.starts_with(".map_err(") && tail.ends_with(")?"));
    if !directly_propagated {
        assert!(
            allow_separate_error_propagation && tail.is_empty(),
            "{needle:?} result is not propagated with ?/map_err?: {tail:?}",
        );
        let remainder = compact(&body[binding.statement_end + 1..]);
        let plain = format!("{}?;", binding.name);
        let mapped = format!("{}.map_err(", binding.name);
        assert!(
            remainder.starts_with(&plain) || remainder.starts_with(&mapped),
            "{needle:?} raw result binding is not immediately propagated",
        );
        if remainder.starts_with(&mapped) {
            let (_, propagation_end) = balanced_call(&remainder, &mapped);
            assert!(
                remainder[propagation_end..].starts_with("?;"),
                "{needle:?} map_err result is not immediately propagated with ?",
            );
        }
    }
    binding
}

fn plain_call_binding<'a>(body: &'a str, needle: &str) -> DirectCallBinding<'a> {
    let binding = call_initializer_binding(body, needle);
    assert!(
        body[binding.call_end..binding.statement_end]
            .trim()
            .is_empty(),
        "{needle:?} accessor result is transformed before its named binding",
    );
    binding
}

fn call_initializer_binding<'a>(body: &'a str, needle: &str) -> DirectCallBinding<'a> {
    assert_eq!(
        occurrences(body, needle),
        1,
        "reviewed function must contain exactly one {needle:?} call",
    );
    let call_start = body.find(needle).unwrap();
    let call_end = balanced_call(body, needle).1;
    let binding_start = body[..call_start]
        .rfind("let ")
        .unwrap_or_else(|| panic!("{needle:?} result is not directly bound"));
    let previous_boundary = [';', '{', '}']
        .into_iter()
        .filter_map(|character| body[..call_start].rfind(character))
        .max()
        .map_or(0, |index| index + 1);
    assert!(
        binding_start >= previous_boundary,
        "{needle:?} is not the initializer of its binding statement",
    );
    let equals = body[binding_start..call_start]
        .rfind('=')
        .map(|offset| binding_start + offset)
        .unwrap_or_else(|| panic!("{needle:?} binding has no equals sign"));
    let name = body[binding_start + "let ".len()..equals]
        .trim()
        .strip_prefix("mut ")
        .unwrap_or_else(|| body[binding_start + "let ".len()..equals].trim());
    assert!(
        !name.is_empty()
            && !name.starts_with('_')
            && name
                .bytes()
                .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric()),
        "{needle:?} must bind one named local, not discard/destructure {name:?}",
    );
    let initializer_prefix = compact(&body[equals + 1..call_start]);
    let plain_receiver_path = initializer_prefix.strip_suffix('.').is_some_and(|path| {
        !path.is_empty()
            && path.split('.').all(|segment| {
                !segment.is_empty()
                    && segment
                        .bytes()
                        .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
            })
    });
    assert!(
        initializer_prefix.is_empty()
            || initializer_prefix == "self."
            || plain_receiver_path
            || initializer_prefix
                .bytes()
                .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric()),
        "{needle:?} binding wraps or reconstructs the direct call before it starts: {initializer_prefix:?}",
    );
    let statement_end = body[call_end..]
        .find(';')
        .map(|offset| call_end + offset)
        .unwrap_or_else(|| panic!("{needle:?} binding statement is unterminated"));
    DirectCallBinding {
        name,
        call_start,
        call_end,
        statement_end,
    }
}

fn assert_order(body: &str, names: &[&str]) {
    let mut cursor = 0;
    for name in names {
        let relative = body[cursor..]
            .find(name)
            .unwrap_or_else(|| panic!("{name:?} is missing from reviewed body"));
        cursor += relative + name.len();
    }
}

#[derive(Clone, Copy)]
struct SourceFunction<'a> {
    name: &'a str,
    body: &'a str,
}

#[derive(Clone, Copy)]
struct PublicSourceFunction<'a> {
    name: &'a str,
    header: &'a str,
    body: &'a str,
}

fn source_functions(source: &str) -> Vec<SourceFunction<'_>> {
    let mut functions = Vec::new();
    for (start, _) in source.match_indices("fn ") {
        let name_start = start + 3;
        let Some(name_end) = source[name_start..]
            .find(['(', '<'])
            .map(|offset| name_start + offset)
        else {
            continue;
        };
        let name = source[name_start..name_end].trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
        {
            continue;
        }
        let body = braced_item(&source[start..], &source[start..name_end]);
        functions.push(SourceFunction { name, body });
    }
    functions
}

fn public_source_functions(source: &str) -> Vec<PublicSourceFunction<'_>> {
    let mut functions = Vec::new();
    for (fn_start, _) in source.match_indices("fn ") {
        let name_start = fn_start + "fn ".len();
        let Some(name_end) = source[name_start..]
            .find(['(', '<'])
            .map(|offset| name_start + offset)
        else {
            continue;
        };
        let name = source[name_start..name_end].trim();
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
        {
            continue;
        }
        let declaration_start = ['{', '}', ';']
            .into_iter()
            .filter_map(|boundary| source[..fn_start].rfind(boundary))
            .max()
            .map_or(0, |index| index + 1);
        let visibility = &source[declaration_start..fn_start];
        let normalized_visibility = compact(visibility);
        if !identifier_tokens(visibility).contains("pub") {
            continue;
        }
        let Some(public_start) = normalized_visibility.rfind("pub") else {
            continue;
        };
        let public_tail = &normalized_visibility[public_start..];
        if public_tail.starts_with("pub(") {
            continue;
        }
        let body = braced_item(&source[fn_start..], &source[fn_start..name_end]);
        let open = body.find('{').expect("public function body");
        let header_end = fn_start + open;
        functions.push(PublicSourceFunction {
            name,
            header: &source[declaration_start..header_end],
            body,
        });
    }
    functions
}

fn all_braced_items<'a>(source: &'a str, needle: &str) -> Vec<&'a str> {
    let mut items = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = source[cursor..].find(needle) {
        let start = cursor + relative;
        let item = braced_item(&source[start..], needle);
        items.push(item);
        cursor = start + item.len();
    }
    items
}

fn public_inherent_function_names(item: &str) -> BTreeSet<&str> {
    public_source_functions(item)
        .into_iter()
        .map(|function| function.name)
        .collect()
}

fn function_header(function: SourceFunction<'_>) -> &str {
    function
        .body
        .split('{')
        .next()
        .expect("source function has no header")
}

fn identifier_tokens(source: &str) -> BTreeSet<&str> {
    source
        .split(|character: char| !(character == '_' || character.is_ascii_alphanumeric()))
        .filter(|token| !token.is_empty())
        .collect()
}

fn assert_unshadowed_binding(body: &str, binding: DirectCallBinding<'_>) {
    assert_unshadowed_name(body, binding.name);
}

fn assert_unshadowed_name(body: &str, name: &str) {
    let declarations = body
        .match_indices("let ")
        .filter(|(start, _)| {
            let declaration = &body[start + "let ".len()..];
            let end = declaration.find('=').unwrap_or(declaration.len());
            declaration[..end]
                .trim()
                .strip_prefix("mut ")
                .unwrap_or_else(|| declaration[..end].trim())
                == name
        })
        .count();
    assert_eq!(
        declarations, 1,
        "authority binding {} was shadowed or redeclared",
        name,
    );
    let assignments = body
        .match_indices(name)
        .filter(|(start, _)| {
            let previous = body[..*start].chars().next_back();
            let has_identifier_boundary = previous.is_none_or(|character| {
                character != '_' && !character.is_ascii_alphanumeric() && character != '.'
            });
            let tail = body[*start + name.len()..].trim_start();
            let is_assignment = ["<<=", ">>=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^="]
                .into_iter()
                .any(|operator| tail.starts_with(operator))
                || (tail.starts_with('=') && !tail.starts_with("==") && !tail.starts_with("=>"));
            has_identifier_boundary && is_assignment
        })
        .count();
    assert_eq!(assignments, 1, "authority binding {} was reassigned", name,);
    let normalized = compact(body);
    for cosmetic in ["drop(", "dbg!(", "println!(", "tracing::", "log::"] {
        assert!(
            !normalized.contains(&format!("{cosmetic}{name}")),
            "authority binding {} is only consumed cosmetically",
            name,
        );
    }
}

fn named_initializer_binding<'a>(body: &'a str, initializer: &str) -> &'a str {
    assert_eq!(
        occurrences(body, initializer),
        1,
        "reviewed initializer {initializer:?} must occur exactly once",
    );
    let start = body.find(initializer).unwrap();
    let binding_start = body[..start]
        .rfind("let ")
        .unwrap_or_else(|| panic!("{initializer:?} is not directly bound"));
    let previous_boundary = [';', '{', '}']
        .into_iter()
        .filter_map(|character| body[..start].rfind(character))
        .max()
        .map_or(0, |index| index + 1);
    assert!(
        binding_start >= previous_boundary,
        "{initializer:?} is not the initializer of its binding",
    );
    let equals = body[binding_start..start]
        .rfind('=')
        .map(|offset| binding_start + offset)
        .expect("named initializer binding has no equals sign");
    let name = body[binding_start + "let ".len()..equals]
        .trim()
        .strip_prefix("mut ")
        .unwrap_or_else(|| body[binding_start + "let ".len()..equals].trim());
    assert!(
        !name.is_empty()
            && !name.starts_with('_')
            && name
                .bytes()
                .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric()),
        "{initializer:?} must bind one named authority",
    );
    assert_unshadowed_name(body, name);
    name
}

fn flowing_names<'a>(body: &'a str, seed: &'a str) -> BTreeSet<&'a str> {
    let mut flowing = BTreeSet::from([seed]);
    loop {
        let before = flowing.len();
        for statement in body.split(';') {
            let Some(let_start) = statement.rfind("let ") else {
                continue;
            };
            let declaration = &statement[let_start + "let ".len()..];
            let Some(equals) = declaration.find('=') else {
                continue;
            };
            let name = declaration[..equals].trim().trim_start_matches("mut ");
            if name.is_empty()
                || !name
                    .bytes()
                    .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
            {
                continue;
            }
            let initializer = &declaration[equals + 1..];
            if identifier_tokens(initializer)
                .iter()
                .any(|token| flowing.contains(token))
            {
                flowing.insert(name);
            }
        }
        for loop_header in body.split('{') {
            let Some(for_start) = loop_header.rfind("for ") else {
                continue;
            };
            let declaration = &loop_header[for_start + "for ".len()..];
            let Some(in_start) = declaration.find(" in ") else {
                continue;
            };
            let name = declaration[..in_start].trim();
            let iterator = &declaration[in_start + " in ".len()..];
            if name
                .bytes()
                .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
                && identifier_tokens(iterator)
                    .iter()
                    .any(|token| flowing.contains(token))
            {
                flowing.insert(name);
            }
        }
        if flowing.len() == before {
            return flowing;
        }
    }
}

fn struct_field_initializer<'a>(argument: &'a str, field: &str) -> Option<&'a str> {
    let open = argument.find('{')?;
    let bytes = argument.as_bytes();
    let mut brace_depth = 1_usize;
    let mut parentheses = 0_usize;
    let mut brackets = 0_usize;
    let mut string = false;
    let mut escaped = false;
    let mut segment_start = open + 1;
    let mut index = segment_start;
    while index < bytes.len() {
        let byte = bytes[index];
        if string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                string = false;
            }
            index += 1;
            continue;
        }
        match byte {
            b'"' => string = true,
            b'(' => parentheses += 1,
            b')' => parentheses -= 1,
            b'[' => brackets += 1,
            b']' => brackets -= 1,
            b'{' => brace_depth += 1,
            b'}' if brace_depth == 1 => {
                let segment = argument[segment_start..index].trim();
                if let Some(initializer) = segment.strip_prefix(field) {
                    let initializer = initializer.trim_start();
                    if initializer.is_empty() {
                        return Some(segment);
                    }
                    if let Some(initializer) = initializer.strip_prefix(':') {
                        return Some(initializer.trim());
                    }
                }
                return None;
            }
            b'}' => brace_depth -= 1,
            b',' if brace_depth == 1 && parentheses == 0 && brackets == 0 => {
                let segment = argument[segment_start..index].trim();
                if let Some(initializer) = segment.strip_prefix(field) {
                    let initializer = initializer.trim_start();
                    if initializer.is_empty() {
                        return Some(segment);
                    }
                    if let Some(initializer) = initializer.strip_prefix(':') {
                        return Some(initializer.trim());
                    }
                }
                segment_start = index + 1;
            }
            _ => {}
        }
        index += 1;
    }
    None
}

fn assert_physical_spec_sink<'a>(
    functions: &'a [SourceFunction<'a>],
    terminal_sink: &str,
    accessor: &str,
    value_accessor: &str,
    value_sink: &str,
    allowed_argument_indices: &[usize],
    struct_field: Option<&str>,
) -> SourceFunction<'a> {
    let function = unique_function_containing(
        functions,
        &[
            "&VisionQkvPhysicalExecutionSpec",
            terminal_sink,
            accessor,
            value_accessor,
            value_sink,
        ],
    );
    let header = compact(function_header(function));
    assert!(
        header.contains("&VisionQkvPhysicalExecutionSpec"),
        "{terminal_sink} helper does not accept the sealed physical spec",
    );
    for alternate in [
        "&PreparedVisionQkvStackExecution",
        "&VisionQkvReadbackLayout",
        "workspace_allocation_bytes:u64",
        "readback_bytes:u64",
        "qkv_canary_bytes:u64",
        "readback_elements:usize",
    ] {
        assert!(
            !header.contains(alternate),
            "{terminal_sink} helper accepts alternate geometry parameter {alternate}",
        );
    }
    let authority = plain_call_binding(function.body, accessor);
    assert_unshadowed_binding(function.body, authority);
    let authority_flow = flowing_names(function.body, authority.name);
    let physical_value = plain_call_binding(function.body, value_accessor);
    assert_unshadowed_binding(function.body, physical_value);
    assert!(
        authority_flow.contains(physical_value.name),
        "{value_accessor} binding {} is not derived from exact {accessor} authority {}",
        physical_value.name,
        authority.name,
    );
    let physical_value_flow = flowing_names(function.body, physical_value.name);
    let sink_arguments = all_balanced_call_arguments(function.body, value_sink);
    assert!(
        sink_arguments.iter().any(|arguments| {
            allowed_argument_indices.iter().any(|index| {
                let Some(argument) = arguments.get(*index) else {
                    return false;
                };
                let routed_value = match struct_field {
                    Some(field) => struct_field_initializer(argument, field),
                    None => Some(*argument),
                };
                routed_value.is_some_and(|routed_value| {
                    identifier_tokens(routed_value)
                        .iter()
                        .any(|token| physical_value_flow.contains(token))
                })
            })
        }),
        "{value_sink} allowed arguments/field {struct_field:?} do not receive data flowing from exact {value_accessor} binding {}",
        physical_value.name,
    );
    function
}

fn strip_whole_parentheses(mut expression: &str) -> &str {
    loop {
        let bytes = expression.as_bytes();
        if bytes.first() != Some(&b'(') || bytes.last() != Some(&b')') {
            return expression;
        }
        let mut depth = 0_usize;
        let mut closes_early = false;
        for (index, byte) in bytes.iter().enumerate() {
            match byte {
                b'(' => depth += 1,
                b')' => {
                    if depth == 0 {
                        return expression;
                    }
                    depth -= 1;
                    if depth == 0 && index + 1 != bytes.len() {
                        closes_early = true;
                        break;
                    }
                }
                _ => {}
            }
        }
        if closes_early || depth != 0 {
            return expression;
        }
        expression = &expression[1..expression.len() - 1];
    }
}

fn expression_is_one_exact_flow(
    expression: &str,
    flowing: &BTreeSet<&str>,
    allowed_single_argument_resolvers: &[&str],
) -> bool {
    fn exact(
        expression: &str,
        flowing: &BTreeSet<&str>,
        allowed_single_argument_resolvers: &[&str],
    ) -> bool {
        let expression = strip_whole_parentheses(expression);
        if flowing.contains(expression) {
            return true;
        }
        for prefix in ["&mut", "&", "*"] {
            if let Some(inner) = expression.strip_prefix(prefix)
                && exact(inner, flowing, allowed_single_argument_resolvers)
            {
                return true;
            }
        }
        if let Some(inner) = expression.strip_suffix(".clone()")
            && exact(inner, flowing, allowed_single_argument_resolvers)
        {
            return true;
        }
        for resolver in allowed_single_argument_resolvers {
            let prefix = format!("{resolver}(");
            if !expression.starts_with(&prefix) || !expression.ends_with(')') {
                continue;
            }
            let arguments = balanced_call_arguments(expression, resolver);
            let (_, end) = balanced_call(expression, resolver);
            if end == expression.len()
                && arguments.len() == 1
                && exact(arguments[0], flowing, allowed_single_argument_resolvers)
            {
                return true;
            }
        }
        false
    }

    let normalized = compact(expression);
    exact(&normalized, flowing, allowed_single_argument_resolvers)
}

fn exact_flowing_names<'a>(
    body: &'a str,
    seed: &'a str,
    allowed_single_argument_resolvers: &[&str],
) -> BTreeSet<&'a str> {
    let mut flowing = BTreeSet::from([seed]);
    loop {
        let before = flowing.len();
        for statement in body.split(';') {
            let Some(let_start) = statement.rfind("let ") else {
                continue;
            };
            let declaration = &statement[let_start + "let ".len()..];
            let Some(equals) = declaration.find('=') else {
                continue;
            };
            let name = declaration[..equals].trim().trim_start_matches("mut ");
            if name.is_empty()
                || !name
                    .bytes()
                    .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
            {
                continue;
            }
            let initializer = &declaration[equals + 1..];
            if expression_is_one_exact_flow(
                initializer,
                &flowing,
                allowed_single_argument_resolvers,
            ) {
                flowing.insert(name);
            }
        }
        if flowing.len() == before {
            return flowing;
        }
    }
}

fn sink_argument_receives_exact_flow(
    body: &str,
    sink: &str,
    argument_index: usize,
    seed: &str,
    allowed_single_argument_resolvers: &[&str],
) -> bool {
    let flow = exact_flowing_names(body, seed, allowed_single_argument_resolvers);
    all_balanced_call_arguments(body, sink)
        .iter()
        .filter_map(|arguments| arguments.get(argument_index))
        .any(|argument| {
            expression_is_one_exact_flow(argument, &flow, allowed_single_argument_resolvers)
        })
}

fn simple_method_receiver<'a>(body: &'a str, method_call: &str) -> &'a str {
    assert_eq!(occurrences(body, method_call), 1);
    let dot = body.find(method_call).unwrap();
    let end = dot;
    let start = body[..end]
        .rfind(|character: char| !(character == '_' || character.is_ascii_alphanumeric()))
        .map_or(0, |index| index + 1);
    let receiver = &body[start..end];
    assert!(
        !receiver.is_empty()
            && receiver
                .bytes()
                .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric()),
        "{method_call} must use one named receiver, not a reconstructed expression",
    );
    receiver
}

fn assert_exact_struct_field_flow(
    body: &str,
    initializer: &str,
    field: &str,
    seed: &str,
    allowed_single_argument_resolvers: &[&str],
) {
    let field_initializer = struct_field_initializer(initializer, field)
        .unwrap_or_else(|| panic!("typed result omitted {field}"));
    let flow = exact_flowing_names(body, seed, allowed_single_argument_resolvers);
    assert!(
        expression_is_one_exact_flow(field_initializer, &flow, allowed_single_argument_resolvers,),
        "typed result field {field} is not one exact flow from {seed}",
    );
}

fn assert_typed_web_physical_adapter_source(source: &str) {
    let live = live_rust_source(source);
    let ast = syn::parse_file(source).expect("typed Web physical adapters must parse as Rust");
    let functions = source_functions(&live);
    for (name, variant, sink, exact_arguments) in [
        (
            "apply_vision_qkv_web_create_buffer_command",
            "VisionQkvWebPhysicalCommand::CreateBuffer",
            "create_buffer(",
            &[(0_usize, "label", &[][..]), (1, "byte_length", &[][..])][..],
        ),
        (
            "apply_vision_qkv_web_copy_buffer_command",
            "VisionQkvWebPhysicalCommand::CopyBuffer",
            "copy_buffer_to_buffer(",
            &[
                (0_usize, "source", &["self.resolve_buffer"][..]),
                (1, "source_offset", &[][..]),
                (2, "destination", &["self.resolve_buffer"][..]),
                (3, "destination_offset", &[][..]),
                (4, "byte_length", &[][..]),
            ][..],
        ),
    ] {
        let matches = functions
            .iter()
            .filter(|function| function.name == name)
            .collect::<Vec<_>>();
        assert_eq!(
            matches.len(),
            1,
            "missing or duplicate typed adapter {name}"
        );
        let function = matches[0];
        assert!(
            function_header(*function).contains("&VisionQkvWebPhysicalCommand"),
            "{name} accepts raw geometry instead of one typed command",
        );
        assert_eq!(
            occurrences(function.body, variant),
            1,
            "{name} variant drift"
        );
        assert_eq!(
            occurrences(function.body, sink),
            1,
            "{name} sink must occur once"
        );
        for (index, seed, allowed_resolvers) in exact_arguments {
            assert!(
                sink_argument_receives_exact_flow(
                    function.body,
                    sink,
                    *index,
                    seed,
                    allowed_resolvers,
                ),
                "{name} {sink} argument {index} is not exact flow from {seed}",
            );
        }
    }

    let create = functions
        .iter()
        .find(|function| function.name == "apply_vision_qkv_web_create_buffer_command")
        .unwrap();
    ast_unique_tail_struct_expression(
        one_ast_function(&ast, "apply_vision_qkv_web_create_buffer_command"),
        "BrowserVisionQkvCreatedBuffer",
    );
    assert!(
        function_header(*create).contains("BrowserVisionQkvCreatedBuffer"),
        "create-buffer adapter must return its typed logical/physical identity",
    );
    let created_gpu_buffer = plain_call_binding(create.body, "create_buffer(");
    let created = braced_item(
        &create.body[created_gpu_buffer.statement_end..],
        "BrowserVisionQkvCreatedBuffer {",
    );
    assert_exact_struct_field_flow(create.body, created, "logical_buffer", "buffer", &[]);
    assert_exact_struct_field_flow(
        create.body,
        created,
        "gpu_buffer",
        created_gpu_buffer.name,
        &[],
    );

    let bind = functions
        .iter()
        .filter(|function| function.name == "apply_vision_qkv_web_create_bind_group_command")
        .collect::<Vec<_>>();
    assert_eq!(
        bind.len(),
        1,
        "missing or duplicate typed bind-group adapter"
    );
    let bind = bind[0];
    ast_unique_tail_struct_expression(
        one_ast_function(&ast, "apply_vision_qkv_web_create_bind_group_command"),
        "BrowserVisionQkvCreatedBindGroup",
    );
    assert!(function_header(*bind).contains("&VisionQkvWebPhysicalCommand"));
    assert_eq!(
        occurrences(bind.body, "VisionQkvWebPhysicalCommand::CreateBindGroup"),
        1,
    );
    assert_eq!(occurrences(bind.body, "create_bind_group("), 1);
    assert!(
        function_header(*bind).contains("BrowserVisionQkvCreatedBindGroup"),
        "bind-group adapter must return its typed layer/kind/physical identity",
    );
    let descriptor = all_balanced_call_arguments(bind.body, "create_bind_group(");
    assert_eq!(descriptor.len(), 1);
    assert_eq!(descriptor[0].len(), 1);
    let label_initializer = struct_field_initializer(descriptor[0][0], "label")
        .expect("typed bind-group descriptor omitted label");
    let label_flow = exact_flowing_names(bind.body, "label", &["Some"]);
    assert!(expression_is_one_exact_flow(
        label_initializer,
        &label_flow,
        &["Some"],
    ));
    let resolved_entries =
        plain_call_binding(bind.body, "resolve_vision_qkv_web_bind_group_entries(");
    let resolution_arguments =
        balanced_call_arguments(bind.body, "resolve_vision_qkv_web_bind_group_entries(")
            .into_iter()
            .filter(|argument| !argument.is_empty())
            .collect::<Vec<_>>();
    assert_eq!(resolution_arguments.len(), 4);
    assert_eq!(compact(resolution_arguments[0]), "layer_index");
    assert_eq!(compact(resolution_arguments[1]), "kind");
    assert_eq!(compact(resolution_arguments[2]), "uniform_slot");
    assert_eq!(compact(resolution_arguments[3]), "entries");
    let entries_initializer = struct_field_initializer(descriptor[0][0], "entries")
        .expect("typed bind-group descriptor omitted entries");
    let entries_flow = exact_flowing_names(bind.body, resolved_entries.name, &[]);
    assert!(
        expression_is_one_exact_flow(entries_initializer, &entries_flow, &[]),
        "typed bind-group descriptor does not consume the exact exhaustive resolver result",
    );
    let created_bind_group = plain_call_binding(bind.body, "create_bind_group(");
    let created = braced_item(
        &bind.body[created_bind_group.statement_end..],
        "BrowserVisionQkvCreatedBindGroup {",
    );
    assert_exact_struct_field_flow(bind.body, created, "layer_index", "layer_index", &[]);
    assert_exact_struct_field_flow(bind.body, created, "kind", "kind", &[]);
    assert_exact_struct_field_flow(
        bind.body,
        created,
        "bind_group",
        created_bind_group.name,
        &[],
    );

    let map = functions
        .iter()
        .filter(|function| function.name == "apply_vision_qkv_web_map_range_command")
        .collect::<Vec<_>>();
    assert_eq!(map.len(), 1, "missing or duplicate typed map adapter");
    let map = map[0];
    assert!(function_header(*map).contains("&VisionQkvWebPhysicalCommand"));
    assert_eq!(
        occurrences(map.body, "VisionQkvWebPhysicalCommand::MapRange"),
        1,
    );
    assert_eq!(occurrences(map.body, ".slice("), 1);
    assert_eq!(occurrences(map.body, "get_mapped_range("), 1);
    assert!(
        sink_argument_receives_exact_flow(map.body, ".slice(", 0, "byte_range", &[]),
        "typed map adapter reconstructed the exact command range",
    );
    let mapped_receiver = simple_method_receiver(map.body, ".slice(");
    let mapped_buffers = exact_flowing_names(map.body, "buffer", &["self.resolve_buffer"]);
    assert!(
        mapped_buffers.contains(mapped_receiver),
        "typed map adapter mapped a receiver not resolved from the command buffer role",
    );

    let create_buffer_helper = functions
        .iter()
        .filter(|function| function.name == "create_buffer")
        .collect::<Vec<_>>();
    assert_eq!(
        create_buffer_helper.len(),
        1,
        "typed allocation must reuse exactly one existing create_buffer helper",
    );
    let helper = create_buffer_helper[0];
    let helper_header = compact(function_header(*helper));
    for parameter in ["label:&str", "size:u64", "usage:wgpu::BufferUsages"] {
        assert!(
            helper_header.contains(parameter),
            "create_buffer helper lost exact parameter {parameter}",
        );
    }
    assert_eq!(occurrences(helper.body, ".device.create_buffer("), 1);
    let descriptor = balanced_call_arguments(helper.body, ".device.create_buffer(");
    assert_eq!(descriptor.len(), 1);
    for (field, seed, resolvers) in [
        ("label", "label", &["Some"][..]),
        ("size", "size", &[][..]),
        ("usage", "usage", &[][..]),
    ] {
        let initializer = struct_field_initializer(descriptor[0], field)
            .unwrap_or_else(|| panic!("create_buffer descriptor omitted {field}"));
        let flow = BTreeSet::from([seed]);
        assert!(
            expression_is_one_exact_flow(initializer, &flow, resolvers),
            "create_buffer descriptor {field} is not exact flow from {seed}",
        );
    }
    assert_eq!(
        compact(
            struct_field_initializer(descriptor[0], "mapped_at_creation")
                .expect("create_buffer descriptor omitted mapped_at_creation"),
        ),
        "false",
    );

    let normalized = compact(
        &functions
            .iter()
            .filter(|function| {
                function.name.starts_with("apply_vision_qkv_web_")
                    && function.name.ends_with("_command")
            })
            .map(|function| function.body)
            .collect::<String>(),
    );
    for forbidden in [
        "workspace_allocation_bytes:u64",
        "readback_bytes:u64",
        "qkv_canary_bytes:u64",
        "semantic_readback_bytes:u64",
        "scratch_canary_readback_bytes:u64",
        "checked_add(",
        "checked_sub(",
        "saturating_add(",
        "wrapping_add(",
    ] {
        assert!(
            !normalized.contains(forbidden),
            "typed physical adapter graph reconstructs geometry through {forbidden}",
        );
    }
}

fn assert_typed_web_physical_storage_source(source: &str) {
    let live = live_rust_source(source);
    let functions = source_functions(&live);
    for (name, result_type, exact_arguments) in [
        (
            "store_vision_qkv_web_created_buffer",
            "BrowserVisionQkvCreatedBuffer",
            &["created.logical_buffer", "created.gpu_buffer"][..],
        ),
        (
            "store_vision_qkv_web_created_bind_group",
            "BrowserVisionQkvCreatedBindGroup",
            &["(created.layer_index,created.kind)", "created.bind_group"][..],
        ),
    ] {
        let matches = functions
            .iter()
            .filter(|function| function.name == name)
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1, "missing or duplicate typed store {name}");
        let function = matches[0];
        assert!(
            function_header(*function).contains(result_type),
            "{name} does not accept the exact typed adapter result {result_type}",
        );
        assert_eq!(
            occurrences(function.body, ".insert("),
            1,
            "{name} must make exactly one identity-preserving store",
        );
        let arguments = balanced_call_arguments(function.body, ".insert(");
        assert_eq!(
            arguments.len(),
            exact_arguments.len(),
            "{name} store arity drifted",
        );
        for (index, expected) in exact_arguments.iter().enumerate() {
            assert_eq!(
                compact(arguments[index]),
                *expected,
                "{name} store argument {index} changed logical ownership",
            );
        }
    }
}

fn enum_variant_arm<'a>(
    body: &'a str,
    enum_name: &str,
    variant: &str,
    all_variants: &[&str],
) -> &'a str {
    let marker = format!("{enum_name}::{variant}");
    assert_eq!(occurrences(body, &marker), 1, "{marker} match arm drifted");
    let start = body.find(&marker).unwrap();
    let search_start = start + marker.len();
    let end = all_variants
        .iter()
        .filter(|candidate| **candidate != variant)
        .filter_map(|candidate| {
            body[search_start..]
                .find(&format!("{enum_name}::{candidate}"))
                .map(|offset| search_start + offset)
        })
        .min()
        .unwrap_or(body.len());
    &body[start..end]
}

fn assert_typed_web_physical_resolver_source(source: &str) {
    const RESOURCE_ENUM: &str = "VisionQkvWebBindingResource";
    const RESOURCE_VARIANTS: &[&str] = &[
        "Norm1Output",
        "QueryWeight",
        "QueryBias",
        "KeyWeight",
        "KeyBias",
        "ValueWeight",
        "ValueBias",
        "WorkspaceRange",
        "Uniform",
        "CuSeqlens",
        "AttentionOutput",
    ];

    let live = live_rust_source(source);
    let functions = source_functions(&live);
    let buffer_resolvers = functions
        .iter()
        .filter(|function| function.name == "resolve_buffer")
        .collect::<Vec<_>>();
    assert_eq!(
        buffer_resolvers.len(),
        1,
        "physical Workspace/Readback resolution must have one authority",
    );
    let buffer_resolver = buffer_resolvers[0];
    assert!(
        function_header(*buffer_resolver).contains("VisionQkvWebPhysicalBuffer"),
        "resolve_buffer does not accept the exact typed logical role",
    );
    assert_eq!(occurrences(buffer_resolver.body, ".get("), 1);
    let lookup = balanced_call_arguments(buffer_resolver.body, ".get(");
    assert_eq!(lookup.len(), 1);
    assert_eq!(
        compact(lookup[0]),
        "buffer",
        "Workspace/Readback resolver substituted a fixed or alternate lookup key",
    );
    for forbidden in [
        "unwrap_or(",
        "unwrap_or_else(",
        "or_else(",
        "VisionQkvWebPhysicalBuffer::Workspace=>VisionQkvWebPhysicalBuffer::Readback",
        "VisionQkvWebPhysicalBuffer::Readback=>VisionQkvWebPhysicalBuffer::Workspace",
    ] {
        assert!(
            !compact(buffer_resolver.body).contains(forbidden),
            "Workspace/Readback resolver contains fallback/swap {forbidden}",
        );
    }

    let entry_resolvers = functions
        .iter()
        .filter(|function| function.name == "resolve_vision_qkv_web_bind_group_entries")
        .collect::<Vec<_>>();
    assert_eq!(entry_resolvers.len(), 1, "typed entry resolver drifted");
    let entries = entry_resolvers[0];
    let entries_header = function_header(*entries);
    for required in [
        "VisionQkvWebBindGroupKind",
        "VisionQkvWebBindGroupEntry",
        "uniform_slot",
    ] {
        assert!(
            entries_header.contains(required),
            "typed entry resolver omitted {required}",
        );
    }
    for forbidden in [
        ".filter(",
        ".filter_map(",
        ".skip(",
        ".take(",
        ".chain(",
        "_=>",
    ] {
        assert!(
            !compact(entries.body).contains(forbidden),
            "typed entry resolver can omit/add/default entries via {forbidden}",
        );
    }
    assert_eq!(occurrences(entries.body, "entry.binding("), 1);
    assert_eq!(occurrences(entries.body, "entry.resource("), 1);
    assert_eq!(
        occurrences(entries.body, "validate_vision_qkv_web_uniform_slot(",),
        1,
        "entry resolver must cross-check command and Uniform static slots",
    );
    let slot_validation =
        balanced_call_arguments(entries.body, "validate_vision_qkv_web_uniform_slot(");
    assert_eq!(slot_validation.len(), 3);
    assert_eq!(compact(slot_validation[0]), "kind");
    assert_eq!(compact(slot_validation[1]), "uniform_slot");
    assert_eq!(compact(slot_validation[2]), "entries");
    let resolved_resource =
        plain_call_binding(entries.body, "resolve_vision_qkv_web_binding_resource(");
    let resource_arguments =
        balanced_call_arguments(entries.body, "resolve_vision_qkv_web_binding_resource(");
    assert_eq!(resource_arguments.len(), 1);
    assert_eq!(compact(resource_arguments[0]), "entry.resource()");
    let entry_initializer = braced_item(entries.body, "wgpu::BindGroupEntry {");
    assert_eq!(
        compact(
            struct_field_initializer(entry_initializer, "binding")
                .expect("resolved entry omitted binding"),
        ),
        "entry.binding()",
    );
    assert_eq!(
        compact(
            struct_field_initializer(entry_initializer, "resource")
                .expect("resolved entry omitted resource"),
        ),
        resolved_resource.name,
    );

    let slot_validators = functions
        .iter()
        .filter(|function| function.name == "validate_vision_qkv_web_uniform_slot")
        .collect::<Vec<_>>();
    assert_eq!(
        slot_validators.len(),
        1,
        "static uniform-slot validator drifted"
    );
    let slot_validator = slot_validators[0];
    let slot_compact = compact(slot_validator.body);
    assert_eq!(
        occurrences(slot_validator.body, "VisionQkvWebBindGroupKind::FusedQkv",),
        1,
    );
    assert_eq!(
        occurrences(slot_validator.body, "VisionQkvWebBindGroupKind::Attention",),
        1,
    );
    let kind_variants = ["FusedQkv", "Attention"];
    assert!(
        compact(enum_variant_arm(
            slot_validator.body,
            "VisionQkvWebBindGroupKind",
            "FusedQkv",
            &kind_variants,
        ))
        .starts_with("VisionQkvWebBindGroupKind::FusedQkv=>1,"),
    );
    assert!(
        compact(enum_variant_arm(
            slot_validator.body,
            "VisionQkvWebBindGroupKind",
            "Attention",
            &kind_variants,
        ))
        .starts_with("VisionQkvWebBindGroupKind::Attention=>4,"),
    );
    assert_eq!(
        occurrences(slot_validator.body, "VisionQkvWebBindingResource::Uniform",),
        1,
    );
    assert!(
        slot_compact.contains("assert_eq!(uniform_slot,&expected)")
            && slot_compact.contains("assert_eq!(uniform_slot,entry_slot)"),
        "uniform-slot validator does not bind both command kind and Uniform entry slot",
    );

    let resource_resolvers = functions
        .iter()
        .filter(|function| function.name == "resolve_vision_qkv_web_binding_resource")
        .collect::<Vec<_>>();
    assert_eq!(
        resource_resolvers.len(),
        1,
        "symbolic resource resolver drifted"
    );
    let resources = resource_resolvers[0];
    assert!(function_header(*resources).contains(RESOURCE_ENUM));
    assert!(
        !compact(resources.body).contains("_=>") && !compact(resources.body).contains("..}"),
        "symbolic resource resolver contains a default/partial arm",
    );
    let expected_context_bindings = [
        ("Norm1Output", "&self.norm1_output"),
        ("QueryWeight", "&self.query_weight"),
        ("QueryBias", "&self.query_bias"),
        ("KeyWeight", "&self.key_weight"),
        ("KeyBias", "&self.key_bias"),
        ("ValueWeight", "&self.value_weight"),
        ("ValueBias", "&self.value_bias"),
        ("CuSeqlens", "&self.cu_seqlens"),
        ("AttentionOutput", "&self.attention_output"),
    ];
    for (variant, expected_binding) in expected_context_bindings {
        let arm = enum_variant_arm(resources.body, RESOURCE_ENUM, variant, RESOURCE_VARIANTS);
        assert_eq!(
            occurrences(arm, "resolve_vision_qkv_web_context_binding(",),
            1,
            "{variant} bypasses the offset-preserving context binding resolver",
        );
        let arguments = balanced_call_arguments(arm, "resolve_vision_qkv_web_context_binding(");
        assert_eq!(arguments.len(), 2);
        assert_eq!(
            compact(arguments[0]),
            expected_binding,
            "{variant} resolved the wrong physical buffer",
        );
        let length_flow = BTreeSet::from(["byte_length"]);
        assert!(
            expression_is_one_exact_flow(arguments[1], &length_flow, &[]),
            "{variant} context size is not exact flow from its sealed byte_length",
        );
    }
    let workspace_arm = enum_variant_arm(
        resources.body,
        RESOURCE_ENUM,
        "WorkspaceRange",
        RESOURCE_VARIANTS,
    );
    let workspace = braced_item(workspace_arm, "wgpu::BufferBinding {");
    assert!(workspace_arm.contains("VisionQkvWebPhysicalBuffer::Workspace"));
    assert_eq!(
        compact(struct_field_initializer(workspace, "buffer").unwrap()),
        "self.resolve_buffer(&VisionQkvWebPhysicalBuffer::Workspace)",
        "WorkspaceRange resolved a different logical buffer role",
    );
    let workspace_offset = struct_field_initializer(workspace, "offset").unwrap();
    let workspace_offset_flow = BTreeSet::from(["byte_offset"]);
    assert!(expression_is_one_exact_flow(
        workspace_offset,
        &workspace_offset_flow,
        &[],
    ));
    let workspace_size = struct_field_initializer(workspace, "size").unwrap();
    let workspace_size_flow = BTreeSet::from(["byte_length"]);
    assert!(expression_is_one_exact_flow(
        workspace_size,
        &workspace_size_flow,
        &["wgpu::BufferSize::new"],
    ));

    let uniform_arm = enum_variant_arm(resources.body, RESOURCE_ENUM, "Uniform", RESOURCE_VARIANTS);
    let uniform = braced_item(uniform_arm, "wgpu::BufferBinding {");
    assert_eq!(
        compact(struct_field_initializer(uniform, "buffer").unwrap()),
        "self.uniform_buffer",
        "Uniform buffer must be the exact already-borrowed session authority",
    );
    let uniform_offset = struct_field_initializer(uniform, "offset").unwrap();
    assert_eq!(
        compact(uniform_offset),
        "self.resolve_vision_qkv_web_uniform_offset(*slot,self.uniform_stride)",
        "Uniform static offset is not exact flow from the sealed slot",
    );
    let uniform_size = struct_field_initializer(uniform, "size").unwrap();
    let uniform_size_flow = BTreeSet::from(["byte_length"]);
    assert!(expression_is_one_exact_flow(
        uniform_size,
        &uniform_size_flow,
        &["wgpu::BufferSize::new"],
    ));

    let context_helpers = functions
        .iter()
        .filter(|function| function.name == "resolve_vision_qkv_web_context_binding")
        .collect::<Vec<_>>();
    assert_eq!(context_helpers.len(), 1, "context binding resolver drifted");
    let context = context_helpers[0];
    assert!(function_header(*context).contains("VisionStackBufferBinding"));
    assert_eq!(occurrences(context.body, "binding.resource("), 1);
    let context_compact = compact(context.body);
    assert!(
        context_compact.contains("binding.bytes==byte_length"),
        "context resolver does not verify sealed bytes against the physical binding bytes",
    );
    assert!(
        !context.body.contains("wgpu::BufferBinding {"),
        "context resolver reconstructed and could lose the strategy/hardening offset",
    );

    let uniform_offsets = functions
        .iter()
        .filter(|function| function.name == "resolve_vision_qkv_web_uniform_offset")
        .collect::<Vec<_>>();
    assert_eq!(
        uniform_offsets.len(),
        1,
        "static uniform offset resolver drifted"
    );
    let uniform_offset = uniform_offsets[0];
    let uniform_offset_compact = compact(uniform_offset.body);
    assert!(
        uniform_offset_compact.contains("u64::from(slot).checked_mul(uniform_stride)"),
        "static uniform offset is not checked slot × stride",
    );
    assert_eq!(occurrences(uniform_offset.body, "checked_mul("), 1);
    for forbidden in [
        "saturating_mul(",
        "wrapping_mul(",
        "checked_add(",
        "saturating_add(",
        "wrapping_add(",
    ] {
        assert!(
            !uniform_offset_compact.contains(forbidden),
            "static uniform offset uses alternate arithmetic {forbidden}",
        );
    }
}

fn assert_branchless_web_physical_orchestration(body: &str, label: &str) {
    let normalized = compact(body);
    let tokens = identifier_tokens(body);
    for forbidden in ["if", "cfg", "match", "return"] {
        assert!(
            !tokens.contains(forbidden),
            "{label} retained conditional routing token {forbidden}",
        );
    }
    for forbidden in [
        "cfg!(",
        "#[cfg",
        ".commands(",
        ".iter(",
        ".filter(",
        ".filter_map(",
        ".skip(",
        ".take(",
        ".chain(",
        "matchVisionQkvWebPhysicalCommand",
        "|command",
        "|_",
    ] {
        assert!(
            !normalized.contains(&compact(forbidden)),
            "{label} retained caller-owned routing/filter bridge {forbidden}",
        );
    }
}

fn assert_typed_web_physical_executor_source(source: &str) {
    let live = live_rust_source(source);
    let sink_trait = braced_item(&live, "pub trait VisionQkvWebPhysicalCommandEffectSink");
    let trait_methods = sink_trait
        .match_indices("fn ")
        .map(|(start, _)| {
            let name_start = start + "fn ".len();
            let name_end = sink_trait[name_start..]
                .find('(')
                .map(|offset| name_start + offset)
                .expect("effect-sink trait method lost argument list");
            sink_trait[name_start..name_end].trim()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        trait_methods,
        BTreeSet::from([
            "apply_copy_buffer",
            "apply_create_bind_group",
            "apply_create_buffer",
            "apply_map_range",
            "store_created_bind_group",
            "store_created_buffer",
        ]),
        "typed physical effect seam must expose exactly four effects and two linear stores",
    );
    for associated in ["CreatedBuffer", "CreatedBindGroup", "Error"] {
        assert_eq!(
            occurrences(sink_trait, &format!("type {associated}")),
            1,
            "typed physical effect seam lost associated type {associated}",
        );
    }
    for method in &trait_methods {
        let marker = format!("fn {method}");
        let start = sink_trait.find(&marker).unwrap();
        let end = sink_trait[start..]
            .find(';')
            .map(|offset| start + offset)
            .expect("effect-sink trait method lost declaration terminator");
        let header = &sink_trait[start..end];
        for required in [
            "command_index",
            "command: &VisionQkvWebPhysicalCommand",
            "Self::Error",
        ] {
            assert!(
                header.contains(required),
                "{} omitted exact global-index/borrow/error contract {required}",
                method,
            );
        }
    }
    for method_name in ["store_created_buffer", "store_created_bind_group"] {
        let marker = format!("fn {method_name}");
        let start = sink_trait.find(&marker).unwrap();
        let end = sink_trait[start..]
            .find(';')
            .map(|offset| start + offset)
            .unwrap();
        let header = &sink_trait[start..end];
        assert!(
            header.contains("created: Self::Created"),
            "{method_name} must consume its typed result by value",
        );
    }

    let executor_syntax = syn::parse_file(&live).expect("typed executor source must parse");
    let executor_items = executor_syntax
        .items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(function)
                if function.sig.ident == "execute_vision_qkv_web_physical_commands" =>
            {
                Some(function)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(executor_items.len(), 1, "typed top-level executor drifted");
    let executor_item = executor_items[0];
    assert!(matches!(executor_item.vis, syn::Visibility::Public(_)));
    assert!(
        executor_item.sig.constness.is_none()
            && executor_item.sig.asyncness.is_none()
            && matches!(executor_item.sig.safety, syn::Safety::Default)
            && executor_item.sig.abi.is_none()
            && executor_item.sig.variadic.is_none(),
        "typed executor must remain one ordinary safe public Rust function",
    );

    let executors = source_functions(&live)
        .into_iter()
        .filter(|function| function.name == "execute_vision_qkv_web_physical_commands")
        .collect::<Vec<_>>();
    assert_eq!(executors.len(), 1, "typed executor definition drifted");
    let executor_function = executors[0];
    let executor = executor_function.body;
    let executor_header = compact(function_header(executor_function));
    assert!(
        executor_header.contains("VisionQkvWebPhysicalCommandEffectSink"),
        "typed executor is not generic over the exact physical effect seam",
    );
    assert_eq!(
        occurrences(
            executor,
            "validate_vision_qkv_web_physical_command_dispatches(",
        ),
        1,
        "typed executor must prevalidate the complete trace exactly once",
    );
    let first_effect = [
        "sink.apply_create_buffer(",
        "sink.apply_create_bind_group(",
        "sink.apply_copy_buffer(",
        "sink.apply_map_range(",
    ]
    .iter()
    .filter_map(|needle| executor.find(needle))
    .min()
    .expect("typed executor has no effect call");
    assert!(
        executor
            .find("validate_vision_qkv_web_physical_command_dispatches(")
            .unwrap()
            < first_effect,
        "typed executor performed an effect before complete trace validation",
    );
    assert_eq!(
        occurrences(executor, ".commands()"),
        1,
        "only the typed executor may iterate the immutable physical plan, exactly once",
    );
    for forbidden in [
        "visitor(",
        ".filter(",
        ".filter_map(",
        ".skip(",
        ".take(",
        ".chain(",
        "cfg!(",
        "#[cfg",
    ] {
        assert!(
            !compact(executor).contains(&compact(forbidden)),
            "typed executor retained caller-gameable routing bridge {forbidden}",
        );
    }

    const COMMAND_VARIANTS: &[&str] =
        &["CreateBuffer", "CreateBindGroup", "CopyBuffer", "MapRange"];
    for (variant, effect, store) in [
        (
            "CreateBuffer",
            "sink.apply_create_buffer(",
            Some("sink.store_created_buffer("),
        ),
        (
            "CreateBindGroup",
            "sink.apply_create_bind_group(",
            Some("sink.store_created_bind_group("),
        ),
        ("CopyBuffer", "sink.apply_copy_buffer(", None),
        ("MapRange", "sink.apply_map_range(", None),
    ] {
        let arm = enum_variant_arm(
            executor,
            "VisionQkvWebPhysicalCommand",
            variant,
            COMMAND_VARIANTS,
        );
        assert_eq!(
            occurrences(arm, effect),
            1,
            "{variant} must route to its exact effect once",
        );
        let effect_arguments = balanced_call_arguments(arm, effect);
        assert_eq!(effect_arguments.len(), 2);
        assert_eq!(compact(effect_arguments[0]), "command_index");
        assert_eq!(compact(effect_arguments[1]), "command");
        if let Some(store) = store {
            let created = named_initializer_binding(arm, effect);
            assert_eq!(occurrences(arm, store), 1);
            let store_arguments = balanced_call_arguments(arm, store);
            assert_eq!(store_arguments.len(), 3);
            assert_eq!(compact(store_arguments[0]), "command_index");
            assert_eq!(compact(store_arguments[1]), "command");
            assert_eq!(
                compact(store_arguments[2]),
                created,
                "{variant} reconstructed or cloned its typed effect result before storage",
            );
            assert_order(arm, &[effect, store]);
            assert!(!compact(arm).contains(&format!("{created}.clone(")));
        } else {
            assert!(
                !arm.contains("store_created_"),
                "{variant} incorrectly enters a created-result store",
            );
        }
    }

    let wrapper_impls = all_braced_items(
        &live,
        "impl<E> VisionQkvWebPhysicalCommandExecutionError<E>",
    );
    assert_eq!(
        wrapper_impls.len(),
        1,
        "generic physical executor error wrapper implementation drifted",
    );
    let into_sink = source_functions(wrapper_impls[0])
        .into_iter()
        .filter(|method| method.name == "into_sink_error")
        .collect::<Vec<_>>();
    assert_eq!(into_sink.len(), 1);
    let into_sink = into_sink[0];
    assert!(
        compact(function_header(into_sink)).contains("self)->Option<E>"),
        "into_sink_error must consume the wrapper and return the exact generic sink error",
    );
    for forbidden in ["clone(", "to_string(", "format!("] {
        assert!(
            !into_sink.body.contains(forbidden),
            "into_sink_error loses exact non-Clone error identity through {forbidden}",
        );
    }
}

fn assert_browser_execution_evidence_builder_source(source: &str) {
    let live = live_rust_source(source);

    let plan = compact(braced_item(
        &live,
        "pub struct BrowserVisionQkvExecutionEvidencePlan",
    ));
    assert_eq!(
        plan,
        compact(
            r#"pub struct BrowserVisionQkvExecutionEvidencePlan {
                dispatch_count: u32,
                command_buffer_count: u32,
                submission_count: u32,
                map_count: u32,
                workspace: BrowserVisionQkvExecutionWorkspaceEvidence,
                bindings: Vec<BrowserVisionQkvExecutionBindingEvidence>,
                canaries: Vec<BrowserVisionQkvExecutionCanaryPlanEvidence>,
            }"#,
        ),
        "execution evidence plan must remain one closed immutable authority",
    );
    for forbidden in ["Clone", "Copy", "RefCell", "UnsafeCell", "Cell<", "&mut"] {
        assert!(
            !braced_item(&live, "pub struct BrowserVisionQkvExecutionEvidencePlan",)
                .contains(forbidden),
            "execution evidence plan became reconstructible/mutable through {forbidden}",
        );
    }

    for (name, fields) in [
        (
            "BrowserVisionQkvExecutionWorkspaceEvidence",
            &[
                "logical_id:&'staticstr",
                "allocation_bytes:u64",
                "semantic_base:u64",
                "semantic_bytes:u64",
            ][..],
        ),
        (
            "BrowserVisionQkvExecutionBindingEvidence",
            &["binding:u32", "byte_offset:u64", "byte_length:u64"][..],
        ),
        (
            "BrowserVisionQkvExecutionCanaryPlanEvidence",
            &[
                "kind:&'staticstr",
                "plane:Option<u32>",
                "byte_offset:u64",
                "byte_length:u64",
            ][..],
        ),
    ] {
        let declaration = compact(braced_item(&live, &format!("struct {name}")));
        for field in fields {
            assert!(
                declaration.contains(field),
                "{name} omitted exact closed field {field}",
            );
        }
        assert_eq!(
            declaration.matches(':').count(),
            fields.len(),
            "{name} gained an extra serialized or mutable field",
        );
    }

    let channel = compact(braced_item(
        &live,
        "struct BrowserVisionQkvExecutionEvidence<'a>",
    ));
    for field in [
        "dispatch_count:u32",
        "command_buffer_count:u32",
        "submission_count:u32",
        "map_count:u32",
        "workspace:&'aBrowserVisionQkvExecutionWorkspaceEvidence",
        "bindings:&'a[BrowserVisionQkvExecutionBindingEvidence]",
        "canaries:Vec<BrowserVisionQkvExecutionCanaryEvidence<'a>>",
    ] {
        assert!(channel.contains(field), "execution channel omitted {field}");
    }
    assert_eq!(channel.matches(':').count(), 7);
    let channel_canary = compact(braced_item(
        &live,
        "struct BrowserVisionQkvExecutionCanaryEvidence<'a>",
    ));
    for field in [
        "kind:&'astr",
        "plane:Option<u32>",
        "byte_offset:u64",
        "byte_length:u64",
        "passed:Option<bool>",
    ] {
        assert!(channel_canary.contains(field));
    }
    assert_eq!(channel_canary.matches(':').count(), 5);
    for wrapper in [
        "BrowserVisionQkvBeginExecutionEvidence<'a>",
        "BrowserVisionQkvFinalExecutionEvidence<'a>",
    ] {
        let declaration = compact(braced_item(&live, &format!("pub struct {wrapper}")));
        assert!(
            declaration
                .contains("#[serde(flatten)]evidence:BrowserVisionQkvExecutionEvidence<'a>",),
            "{wrapper} must serialize only the sealed common channel view",
        );
        assert_eq!(declaration.matches(':').count(), 1);
    }
    for forbidden in [
        "impl Serialize for BrowserVisionQkvExecutionEvidence",
        "impl Serialize for BrowserVisionQkvBeginExecutionEvidence",
        "impl Serialize for BrowserVisionQkvFinalExecutionEvidence",
    ] {
        assert!(
            !live.contains(forbidden),
            "execution evidence gained a schema-bypassing serializer through {forbidden}",
        );
    }

    let plan_impls = all_braced_items(&live, "impl BrowserVisionQkvExecutionEvidencePlan");
    assert_eq!(plan_impls.len(), 1);
    assert_eq!(
        public_inherent_function_names(plan_impls[0]),
        BTreeSet::from(["from_prepared"]),
        "execution plan must expose only its checked prepared-spec factory",
    );
    let methods = source_functions(plan_impls[0]);
    let from_prepared = methods
        .iter()
        .find(|method| method.name == "from_prepared")
        .unwrap();
    let from_header = compact(function_header(*from_prepared));
    assert!(from_header.contains("Option<&VisionQkvPhysicalExecutionSpec>"));
    assert!(from_header.contains("Result<Option<Self>"));
    for required in [
        ".prepared_execution(",
        ".layer_count(",
        ".workspace(",
        ".layers(",
        ".attention_bridge(",
        ".bindings(",
        ".canaries(",
        ".allocation_bytes(",
        ".semantic_base(",
        ".semantic_bytes(",
        ".binding(",
        ".byte_offset(",
        ".byte_length(",
    ] {
        assert!(
            from_prepared.body.contains(required),
            "prepared evidence factory omitted sealed accessor {required}",
        );
    }
    assert_eq!(
        occurrences(
            from_prepared.body,
            "BrowserVisionQkvExecutionEvidencePlan {"
        ),
        1,
    );
    let plan_initializer = braced_item(
        from_prepared.body,
        "BrowserVisionQkvExecutionEvidencePlan {",
    );
    for field in [
        "dispatch_count",
        "command_buffer_count",
        "submission_count",
        "workspace",
        "bindings",
        "canaries",
    ] {
        assert_eq!(
            compact(struct_field_initializer(plan_initializer, field).unwrap()),
            field,
            "prepared factory substituted or reconstructed plan field {field}",
        );
    }
    assert_eq!(
        compact(struct_field_initializer(plan_initializer, "map_count").unwrap()),
        "1",
    );
    let plan_impl_needle = "impl BrowserVisionQkvExecutionEvidencePlan";
    let raw_impl_start = live
        .find(plan_impl_needle)
        .expect("execution evidence plan implementation is missing");
    let raw_plan_impl_source = braced_item(&source[raw_impl_start..], plan_impl_needle);
    let raw_plan_impl = syn::parse_str::<syn::ItemImpl>(raw_plan_impl_source)
        .expect("execution evidence plan implementation must parse as Rust");
    let raw_from_prepared = raw_plan_impl
        .items
        .iter()
        .find_map(|item| match item {
            syn::ImplItem::Fn(method) if method.sig.ident == "from_prepared" => Some(AstFunction {
                name: &method.sig.ident,
                signature: &method.sig,
                block: &method.block,
                owner: Some(&raw_plan_impl.self_ty),
            }),
            _ => None,
        })
        .expect("execution evidence plan omitted from_prepared");
    let raw_workspace = ast_unique_struct_expression(
        raw_from_prepared,
        "BrowserVisionQkvExecutionWorkspaceEvidence",
    );
    let logical_id = raw_workspace
        .fields
        .iter()
        .find(|field| matches!(&field.member, syn::Member::Named(name) if name == "logical_id"))
        .expect("prepared workspace evidence omitted logical_id");
    assert!(
        matches!(&logical_id.expr, syn::Expr::Lit(literal)
            if matches!(&literal.lit, syn::Lit::Str(value)
                if value.value() == "vision-stack-qkv-workspace")),
        "prepared workspace evidence replaced logical_id",
    );
    let workspace_initializer = braced_item(
        from_prepared.body,
        "BrowserVisionQkvExecutionWorkspaceEvidence {",
    );
    for (field, exact) in [
        ("allocation_bytes", "workspace.allocation_bytes()"),
        ("semantic_base", "workspace.semantic_base()"),
        ("semantic_bytes", "workspace.semantic_bytes()"),
    ] {
        assert_eq!(
            compact(struct_field_initializer(workspace_initializer, field).unwrap()),
            compact(exact),
            "prepared workspace evidence replaced {field}",
        );
    }
    let binding_initializer = braced_item(
        from_prepared.body,
        "BrowserVisionQkvExecutionBindingEvidence {",
    );
    for (field, exact) in [
        ("binding", "binding.binding()"),
        ("byte_offset", "binding.byte_offset()"),
        ("byte_length", "binding.byte_length()"),
    ] {
        assert_eq!(
            compact(struct_field_initializer(binding_initializer, field).unwrap()),
            compact(exact),
        );
    }
    let canary_initializer = braced_item(
        from_prepared.body,
        "BrowserVisionQkvExecutionCanaryPlanEvidence {",
    );
    for (field, exact) in [
        ("kind", "kind"),
        ("plane", "plane"),
        ("byte_offset", "canary.byte_offset()"),
        ("byte_length", "canary.byte_length()"),
    ] {
        assert_eq!(
            compact(struct_field_initializer(canary_initializer, field).unwrap()),
            compact(exact),
        );
    }
    assert!(from_prepared.body.contains("canary.kind()"));
    for forbidden in [
        "clone(",
        "to_owned(",
        "RefCell",
        "Cell<",
        "static mut",
        "constant_execution",
        "default_execution",
    ] {
        assert!(
            !from_prepared.body.contains(forbidden),
            "prepared evidence factory reconstructed mutable/constant authority via {forbidden}",
        );
    }

    let channel_builder = methods
        .iter()
        .find(|method| method.name == "channel_evidence")
        .expect("execution plan needs one private common channel builder");
    assert!(!function_header(*channel_builder).contains("pub "));
    let channel_canary_initializer = braced_item(
        channel_builder.body,
        "BrowserVisionQkvExecutionCanaryEvidence {",
    );
    for (field, exact) in [
        ("kind", "canary.kind"),
        ("plane", "canary.plane"),
        ("byte_offset", "canary.byte_offset"),
        ("byte_length", "canary.byte_length"),
        ("passed", "passed"),
    ] {
        assert_eq!(
            compact(struct_field_initializer(channel_canary_initializer, field).unwrap()),
            compact(exact),
        );
    }
    let channel_initializer =
        braced_item(channel_builder.body, "BrowserVisionQkvExecutionEvidence {");
    for (field, exact) in [
        ("dispatch_count", "self.dispatch_count"),
        ("command_buffer_count", "self.command_buffer_count"),
        ("submission_count", "self.submission_count"),
        ("map_count", "self.map_count"),
        ("workspace", "&self.workspace"),
        ("bindings", "&self.bindings"),
        ("canaries", "canaries"),
    ] {
        assert_eq!(
            compact(struct_field_initializer(channel_initializer, field).unwrap()),
            compact(exact),
            "channel builder replaced immutable plan field {field}",
        );
    }
    let compact_channel_builder = compact(channel_builder.body);
    assert!(compact_channel_builder.contains("self.canaries.iter()"));
    assert!(compact_channel_builder.contains(".zip(passed)"));

    let begin_impls =
        all_braced_items(&live, "impl<'a> BrowserVisionQkvBeginExecutionEvidence<'a>");
    assert_eq!(begin_impls.len(), 1);
    assert_eq!(
        public_inherent_function_names(begin_impls[0]),
        BTreeSet::from(["from_plan"]),
    );
    let begin = source_functions(begin_impls[0])
        .into_iter()
        .find(|method| method.name == "from_plan")
        .unwrap();
    assert!(
        compact(function_header(begin))
            .contains("Option<&'aBrowserVisionQkvExecutionEvidencePlan>")
    );
    assert_eq!(occurrences(begin.body, "plan.channel_evidence("), 1);
    assert!(compact(begin.body).contains("vec![None;plan.canaries.len()]"));
    for forbidden in ["Some(true)", "Some(false)", "canary_results", "clone("] {
        assert!(!begin.body.contains(forbidden));
    }

    let final_impls =
        all_braced_items(&live, "impl<'a> BrowserVisionQkvFinalExecutionEvidence<'a>");
    assert_eq!(final_impls.len(), 1);
    assert_eq!(
        public_inherent_function_names(final_impls[0]),
        BTreeSet::from(["from_verified_plan"]),
    );
    let final_builder = source_functions(final_impls[0])
        .into_iter()
        .find(|method| method.name == "from_verified_plan")
        .unwrap();
    let final_header = compact(function_header(final_builder));
    assert!(final_header.contains("Option<&'aBrowserVisionQkvExecutionEvidencePlan>"));
    assert!(final_header.contains("canary_results:&[bool]"));
    assert!(final_header.contains("Result<Option<Self>"));
    assert!(compact(final_builder.body).contains("canary_results.len()!=plan.canaries.len()"));
    assert!(
        final_builder
            .body
            .contains("canary_results.iter().copied().map(Some)")
    );
    assert_eq!(occurrences(final_builder.body, "plan.channel_evidence("), 1);
    for forbidden in [
        "vec![true",
        "vec![false",
        "repeat(true",
        "repeat(false",
        "clone(",
    ] {
        assert!(!final_builder.body.contains(forbidden));
    }

    assert_eq!(
        value_constructor_occurrences(&live, "BrowserVisionQkvExecutionEvidencePlan"),
        1,
        "only the checked prepared-spec factory may create the immutable evidence plan",
    );
    assert_eq!(
        value_constructor_occurrences(&live, "BrowserVisionQkvBeginExecutionEvidence"),
        1,
    );
    assert_eq!(
        value_constructor_occurrences(&live, "BrowserVisionQkvFinalExecutionEvidence"),
        1,
    );
}

fn assert_selection_evidence_authority_source(lib_source: &str, web_source: &str) {
    let live_lib = live_rust_source(lib_source);
    let live_web = live_rust_source(web_source);
    let combined = format!("{live_lib}\n{live_web}");
    let combined_source = format!("{lib_source}\n{web_source}");

    assert_browser_execution_evidence_builder_source(&combined_source);

    let envelope = braced_item(&live_lib, "pub struct VisionQkvEvidenceEnvelope<");
    let envelope_compact = compact(envelope);
    assert!(
        envelope_compact.contains("qkv_selection:&'aVisionQkvSelectionEvidence"),
        "evidence envelope must borrow the exact immutable selection allocation",
    );
    assert!(
        envelope_compact.contains("qkv_execution:Option<&'aE>"),
        "evidence envelope must borrow the exact channel-specific execution value",
    );
    for forbidden in [
        "pub qkv_selection",
        "pub qkv_execution",
        "skip_serializing_if",
        "skip_serializing",
        "flatten",
        "default",
    ] {
        assert!(
            !envelope.contains(forbidden),
            "evidence envelope exposed or conditionally omitted a closed-schema field via {forbidden}",
        );
    }
    assert_eq!(
        value_constructor_occurrences(&combined, "VisionQkvEvidenceEnvelope"),
        1,
        "only the propagation authority may construct the borrowed evidence envelope",
    );

    let envelope_impls = all_braced_items(&live_lib, "impl<'a, E> VisionQkvEvidenceEnvelope<");
    assert_eq!(envelope_impls.len(), 1);
    assert_eq!(
        envelope_impls
            .iter()
            .flat_map(|implementation| public_inherent_function_names(implementation))
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["qkv_execution", "qkv_selection"]),
        "borrowed envelope must expose only read-only identity accessors",
    );
    for (accessor, return_fragment, expression) in [
        (
            "qkv_selection",
            "&VisionQkvSelectionEvidence",
            "self.qkv_selection",
        ),
        ("qkv_execution", "Option<&E>", "self.qkv_execution"),
    ] {
        let method = envelope_impls
            .iter()
            .flat_map(|implementation| source_functions(implementation))
            .find(|method| method.name == accessor)
            .unwrap();
        assert!(function_header(method).contains(return_fragment));
        let body_expression =
            &method.body[method.body.find('{').unwrap() + 1..method.body.rfind('}').unwrap()];
        assert_eq!(
            compact(body_expression),
            compact(expression),
            "{accessor} reconstructed or replaced its borrowed identity",
        );
    }

    let propagation = braced_item(
        &live_lib,
        "pub struct VisionQkvSelectionEvidencePropagation",
    );
    let propagation_compact = compact(propagation);
    assert!(
        propagation_compact.contains("Rc<VisionQkvSelectionEvidence>")
            || propagation_compact.contains("Arc<VisionQkvSelectionEvidence>"),
        "selection propagation must clone one immutable allocation, not reconstruct its value",
    );
    for forbidden in ["pub evidence", "RefCell", "UnsafeCell", "Cell<", "&mut"] {
        assert!(
            !propagation.contains(forbidden),
            "selection propagation exposed mutable/reconstructible authority via {forbidden}",
        );
    }

    let propagation_impls =
        all_braced_items(&live_lib, "impl VisionQkvSelectionEvidencePropagation");
    assert_eq!(propagation_impls.len(), 1);
    let propagation_methods = propagation_impls
        .iter()
        .flat_map(|implementation| source_functions(implementation))
        .collect::<Vec<_>>();
    assert_eq!(
        propagation_impls
            .iter()
            .flat_map(|implementation| public_inherent_function_names(implementation))
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "additive_begin_evidence",
            "evidence_json",
            "final_diagnostics_evidence",
            "opaque_selection_evidence",
            "uses_legacy_topology",
        ]),
        "selection propagation surface must remain closed and read-only",
    );
    let opaque_accessor = propagation_methods
        .iter()
        .find(|method| method.name == "opaque_selection_evidence")
        .unwrap();
    assert!(
        function_header(*opaque_accessor).contains("&VisionQkvSelectionEvidence"),
        "opaque accessor must borrow the one shared allocation",
    );
    let opaque_expression = &opaque_accessor.body
        [opaque_accessor.body.find('{').unwrap() + 1..opaque_accessor.body.rfind('}').unwrap()];
    assert_eq!(compact(opaque_expression), "&self.evidence");

    let private_envelope_builders = propagation_methods
        .iter()
        .filter(|method| method.name == "evidence_envelope")
        .collect::<Vec<_>>();
    assert_eq!(
        private_envelope_builders.len(),
        1,
        "begin and final channels must share one private borrowed-envelope constructor",
    );
    let envelope_builder = private_envelope_builders[0];
    assert!(
        !function_header(*envelope_builder).contains("pub "),
        "caller must not be able to author an arbitrary evidence envelope",
    );
    let envelope_initializer = braced_item(envelope_builder.body, "VisionQkvEvidenceEnvelope {");
    assert_eq!(
        compact(struct_field_initializer(envelope_initializer, "qkv_selection").unwrap()),
        "&self.evidence",
    );
    assert_eq!(
        compact(struct_field_initializer(envelope_initializer, "qkv_execution").unwrap()),
        "qkv_execution",
    );

    for channel in ["additive_begin_evidence", "final_diagnostics_evidence"] {
        let method = propagation_methods
            .iter()
            .find(|method| method.name == channel)
            .unwrap();
        let header = compact(function_header(*method));
        assert!(
            header.contains("<'a,E>")
                && header.contains("&'aself")
                && header.contains("Option<&'aE>")
                && header.contains("VisionQkvEvidenceEnvelope<'a,E>"),
            "{channel} must generically borrow selection and channel execution for one lifetime",
        );
        assert_eq!(occurrences(method.body, "self.evidence_envelope("), 1);
        let arguments = balanced_call_arguments(method.body, "self.evidence_envelope(");
        assert_eq!(arguments.len(), 1);
        assert_eq!(compact(arguments[0]), "qkv_execution");
        for forbidden in [
            "VisionQkvEvidenceEnvelope {",
            "VisionQkvSelectionEvidence {",
            "clone(",
            "serialize",
            "to_json_string(",
        ] {
            assert!(
                !method.body.contains(forbidden),
                "{channel} reconstructed or prematurely serialized evidence via {forbidden}",
            );
        }
    }

    let factory = braced_item(
        &live_lib,
        "pub fn build_vision_qkv_selection_evidence_propagation(",
    );
    assert!(
        function_header(SourceFunction {
            name: "build_vision_qkv_selection_evidence_propagation",
            body: factory,
        })
        .contains("&VisionQkvCompilerHandoff"),
        "selection evidence factory must accept only the exact compiler handoff",
    );
    let semantic_handoff_calls = factory
        .match_indices("handoff.semantic_graph_blake3_hex(")
        .filter(|(start, _)| {
            factory[..*start]
                .chars()
                .next_back()
                .is_none_or(|character| character != '_' && !character.is_ascii_alphanumeric())
        })
        .count();
    assert_eq!(
        semantic_handoff_calls, 1,
        "semantic identity must be read from the exact handoff parameter",
    );
    let lib_ast =
        syn::parse_file(lib_source).expect("selection evidence source must parse as Rust");
    let factory_ast = one_ast_function(&lib_ast, "build_vision_qkv_selection_evidence_propagation");
    let semantic_call = ast_unique_method_call(factory_ast, "semantic_graph_blake3_hex");
    assert!(
        ast_expr_is_path(&semantic_call.receiver, "handoff") && semantic_call.args.is_empty(),
        "semantic graph identity must come from the exact handoff receiver",
    );
    let semantic = named_initializer_binding(factory, "handoff.semantic_graph_blake3_hex(");
    let evidence_initializer = braced_item(factory, "VisionQkvSelectionEvidence {");
    let semantic_initializer =
        struct_field_initializer(evidence_initializer, "semantic_graph_blake3")
            .expect("selection evidence omitted semantic_graph_blake3");
    let semantic_flow = exact_flowing_names(factory, semantic, &[]);
    assert!(
        expression_is_one_exact_flow(semantic_initializer, &semantic_flow, &[]),
        "semantic graph identity is not exact flow from the supplied handoff",
    );
    assert_eq!(
        value_constructor_occurrences(factory, "VisionQkvSelectionEvidence"),
        1,
    );
    assert_eq!(
        value_constructor_occurrences(factory, "VisionQkvSelectionEvidencePropagation"),
        1,
    );
    assert_eq!(
        value_constructor_occurrences(&combined, "VisionQkvSelectionEvidence"),
        1,
        "selection evidence must have one factory-owned value construction",
    );
    assert_eq!(
        value_constructor_occurrences(&combined, "VisionQkvSelectionEvidencePropagation"),
        1,
        "propagation authority must have one factory-owned value construction",
    );

    let selection = braced_item(&live_web, "pub struct WebVisionQkvStackSelection");
    let selection_compact = compact(selection);
    assert!(selection_compact.contains("handoff:VisionQkvCompilerHandoff"));
    assert!(
        selection_compact.contains("evidence:VisionQkvSelectionEvidencePropagation"),
        "opaque wasm handle must own the exact shared evidence propagation authority",
    );
    let compile = braced_item(
        &live_web,
        "pub fn compile_vision_encoder_stack_qkv_selection(",
    );
    let handoff = named_initializer_binding(compile, "compile_vision_qkv_stack_handoff(");
    let evidence =
        named_initializer_binding(compile, "build_vision_qkv_selection_evidence_propagation(");
    assert_order(
        compile,
        &[
            "compile_vision_qkv_stack_handoff(",
            "build_vision_qkv_selection_evidence_propagation(",
        ],
    );
    let factory_arguments =
        balanced_call_arguments(compile, "build_vision_qkv_selection_evidence_propagation(");
    assert_eq!(factory_arguments.len(), 1);
    assert_eq!(compact(factory_arguments[0]), format!("&{handoff}"));
    let opaque_initializer = braced_item(compile, "WebVisionQkvStackSelection {");
    assert_eq!(
        compact(struct_field_initializer(opaque_initializer, "handoff").unwrap()),
        handoff,
    );
    assert_eq!(
        compact(struct_field_initializer(opaque_initializer, "evidence").unwrap()),
        evidence,
    );

    let opaque_evidence_json = braced_item(&live_web, "pub fn evidence_json(");
    assert_eq!(
        occurrences(opaque_evidence_json, "self.evidence.evidence_json("),
        1,
        "opaque serializer must delegate once to its exact shared authority",
    );
    let opaque_expression = &opaque_evidence_json
        [opaque_evidence_json.find('{').unwrap() + 1..opaque_evidence_json.rfind('}').unwrap()];
    assert!(
        compact(opaque_expression).starts_with("self.evidence.evidence_json()"),
        "opaque serializer's sole return flow is not the exact propagation serializer",
    );
    assert!(
        !opaque_expression.contains(';'),
        "opaque serializer retained a dead/discarded delegated call",
    );
    for forbidden in ["if", "match", "return", "cfg", "let"] {
        assert!(
            !identifier_tokens(opaque_expression).contains(forbidden),
            "opaque serializer conditionally/discardedly delegates through {forbidden}",
        );
    }
    for forbidden in [".map(", ".and_then(", ".or_else(", ".unwrap_or("] {
        assert!(
            !compact(opaque_expression).contains(&compact(forbidden)),
            "opaque serializer replaces its successful delegated value through {forbidden}",
        );
    }
    for forbidden in [
        "self.handoff",
        "semantic_graph_blake3",
        "VisionQkvSelectionEvidence {",
        "serde_json::",
        "to_json_string(",
    ] {
        assert!(
            !opaque_evidence_json.contains(forbidden),
            "opaque serializer reauthored evidence through {forbidden}",
        );
    }

    let session = compact(braced_item(&live_web, "struct BrowserVisionStackSession"));
    assert!(
        session.contains("qkv_selection_evidence:Option<VisionQkvSelectionEvidencePropagation>",),
        "Web session must retain the exact optional propagation clone",
    );
    assert!(
        session
            .contains("qkv_execution_evidence_plan:Option<BrowserVisionQkvExecutionEvidencePlan>"),
        "optimized session must retain one immutable execution-plan evidence authority",
    );
    let begin = braced_item(
        &live_web,
        "fn begin_vision_stack_sharded_with_qkv_selection(",
    );
    let session_evidence = named_initializer_binding(begin, "qkv_selection.evidence.clone(");
    let session_initializer = braced_item(begin, "BrowserVisionStackSession {");
    assert_exact_struct_field_flow(
        begin,
        session_initializer,
        "qkv_selection_evidence",
        session_evidence,
        &["Some"],
    );
    let plan = named_initializer_binding(
        begin,
        "BrowserVisionQkvExecutionEvidencePlan::from_prepared(",
    );
    let plan_arguments = balanced_call_arguments(
        begin,
        "BrowserVisionQkvExecutionEvidencePlan::from_prepared(",
    )
    .into_iter()
    .filter(|argument| !argument.is_empty())
    .collect::<Vec<_>>();
    assert_eq!(plan_arguments.len(), 1);
    assert_eq!(
        compact(plan_arguments[0]),
        "qkv_physical_execution.as_ref()",
        "browser evidence plan was not built from the exact sealed physical spec retained by the session",
    );
    assert_exact_struct_field_flow(
        begin,
        session_initializer,
        "qkv_execution_evidence_plan",
        plan,
        &[],
    );

    let web_functions = source_functions(&live_web);
    let begin_serializer =
        unique_function_containing(&web_functions, &["fn vision_stack_qkv_status_json("]);
    let final_serializer =
        unique_function_containing(&web_functions, &["fn vision_stack_qkv_diagnostics_json("]);
    assert_final_diagnostics_causal_flow_ast(web_source);
    let begin_reachable =
        reachable_functions(&live_web, "begin_vision_stack_sharded_with_qkv_selection")
            .into_iter()
            .map(|function| function.name)
            .collect::<BTreeSet<_>>();
    let final_reachable = reachable_functions(&live_web, "finish_vision_stack_sharded_once")
        .into_iter()
        .map(|function| function.name)
        .collect::<BTreeSet<_>>();
    assert!(begin_reachable.contains("vision_stack_qkv_status_json"));
    assert!(final_reachable.contains("vision_stack_qkv_diagnostics_json"));
    assert!(!begin_reachable.contains("vision_stack_qkv_diagnostics_json"));
    assert!(!final_reachable.contains("vision_stack_qkv_status_json"));
    assert_eq!(occurrences(&live_web, ".additive_begin_evidence("), 1);
    assert_eq!(occurrences(&live_web, ".final_diagnostics_evidence("), 1);
    assert_eq!(occurrences(begin_serializer.body, "to_json_string("), 0);
    assert_eq!(occurrences(final_serializer.body, "to_json_string("), 0);

    let common_functions = source_functions(&live_lib);
    let common_serializer = common_functions
        .iter()
        .find(|function| function.name == "serialize_vision_stack_qkv_record_json")
        .expect("one host-callable QKV record serializer authority");
    let common_record = braced_item(common_serializer.body, "VisionStackQkvSerializedRecord {");
    assert_eq!(
        compact(struct_field_initializer(common_record, "legacy").unwrap()),
        "legacy",
    );
    assert_eq!(
        compact(struct_field_initializer(common_record, "evidence").unwrap()),
        "evidence",
    );
    assert_eq!(occurrences(common_serializer.body, "to_json_string("), 1);
    for forbidden in ["if ", "match ", "serde_json::Value", "json!", "clone("] {
        assert!(
            !common_serializer.body.contains(forbidden),
            "common QKV serializer can replace/reconstruct its inputs via {forbidden}",
        );
    }
    for public_name in [
        "serialize_vision_stack_qkv_begin_status_json",
        "serialize_vision_stack_qkv_final_diagnostics_json",
    ] {
        let channel_serializers = common_functions
            .iter()
            .filter(|function| function.name == public_name)
            .collect::<Vec<_>>();
        assert_eq!(channel_serializers.len(), 1);
        let channel_serializer = channel_serializers[0];
        assert_eq!(
            occurrences(
                channel_serializer.body,
                "serialize_vision_stack_qkv_record_json(",
            ),
            1,
        );
        let arguments = balanced_call_arguments(
            channel_serializer.body,
            "serialize_vision_stack_qkv_record_json(",
        );
        assert_eq!(arguments.len(), 2);
        assert_eq!(compact(arguments[0]), "legacy");
        assert_eq!(compact(arguments[1]), "evidence");
        let expression = &channel_serializer.body[channel_serializer.body.find('{').unwrap() + 1
            ..channel_serializer.body.rfind('}').unwrap()];
        assert_eq!(
            compact(expression),
            "serialize_vision_stack_qkv_record_json(legacy,evidence)",
            "{public_name} conditionally replaced its common serializer result",
        );
    }

    for (
        channel,
        serializer,
        envelope_call,
        execution_builder,
        legacy_builder,
        production_serializer,
    ) in [(
        "begin",
        begin_serializer,
        ".additive_begin_evidence(",
        "BrowserVisionQkvBeginExecutionEvidence::from_plan(",
        Some("build_vision_stack_legacy_status_record("),
        "serialize_vision_stack_qkv_begin_status_json(",
    )] {
        assert_eq!(
            occurrences(
                serializer.body,
                "session.qkv_execution_evidence_plan.as_ref("
            ),
            1,
            "{channel} serializer must borrow the session's exact execution-plan authority",
        );
        let execution = named_initializer_binding(serializer.body, execution_builder);
        let plan_arguments = balanced_call_arguments(serializer.body, execution_builder);
        assert!(
            !plan_arguments.is_empty()
                && compact(plan_arguments[0]).contains("qkv_execution_evidence_plan"),
            "{channel} channel view was not derived from the session plan authority",
        );
        if channel == "begin" {
            assert_eq!(plan_arguments.len(), 1);
        } else {
            assert_eq!(plan_arguments.len(), 2);
            assert_eq!(compact(plan_arguments[1]), "canary_results");
        }
        assert_eq!(occurrences(serializer.body, envelope_call), 1);
        let envelope_arguments = balanced_call_arguments(serializer.body, envelope_call);
        assert_eq!(envelope_arguments.len(), 1);
        assert!(
            compact(envelope_arguments[0]).contains(&format!("{execution}.as_ref()")),
            "{channel} serializer did not feed its exact channel view into the propagation envelope",
        );
        let selection_evidence =
            named_initializer_binding(serializer.body, "session.qkv_selection_evidence.as_ref(");
        assert_eq!(
            occurrences(
                serializer.body,
                &format!("{selection_evidence}{envelope_call}"),
            ),
            1,
            "{channel} serializer used a constant, null, or wrong selection authority",
        );
        let evidence = named_initializer_binding(serializer.body, envelope_call);
        let expected_legacy_argument = if let Some(legacy_builder) = legacy_builder {
            let legacy = named_initializer_binding(serializer.body, legacy_builder);
            format!("&{legacy}")
        } else {
            assert!(
                compact(function_header(serializer))
                    .contains("legacy_diagnostics:&VisionStackLegacyDiagnosticsRecord"),
                "final QKV serializer must borrow the one diagnostics base record built by finish",
            );
            "legacy_diagnostics".to_owned()
        };
        let serialized_json = named_initializer_binding(serializer.body, production_serializer);
        let serialize_arguments = balanced_call_arguments(serializer.body, production_serializer);
        assert_eq!(serialize_arguments.len(), 2);
        assert_eq!(compact(serialize_arguments[0]), expected_legacy_argument);
        assert_eq!(compact(serialize_arguments[1]), evidence);
        assert_unshadowed_binding(
            serializer.body,
            direct_call_binding(serializer.body, production_serializer, false),
        );
        let return_position = serializer.body.rfind("Ok(").expect("channel return");
        assert_eq!(
            occurrences(serializer.body, "Ok("),
            1,
            "{channel} serializer retained an alternate successful early return",
        );
        let return_tail = &serializer.body[return_position..];
        let (return_arguments, return_end) = balanced_call(return_tail, "Ok(");
        assert_eq!(return_arguments.len(), 1);
        assert_eq!(compact(return_arguments[0]), serialized_json);
        assert_eq!(compact(&return_tail[return_end..]), "}");
        for forbidden in [
            "VisionQkvEvidenceEnvelope {",
            "VisionQkvSelectionEvidence {",
            "VisionStackQkvSerializedRecord {",
            "serde_json::json!",
            "HashMap",
            "BTreeMap",
        ] {
            assert!(
                !serializer.body.contains(forbidden),
                "{channel} serializer caller-authored evidence via {forbidden}",
            );
        }
    }
    assert!(
        final_serializer.body.contains("canary_results"),
        "final serializer must derive boolean canary results from mapped verification",
    );
    assert!(
        !begin_serializer.body.contains("canary_results"),
        "begin serializer must not reuse unavailable final verification results",
    );

    let begin_call = named_initializer_binding(begin, "vision_stack_qkv_status_json(");
    assert_unshadowed_binding(
        begin,
        direct_call_binding(begin, "vision_stack_qkv_status_json(", false),
    );
    let begin_return = begin.rfind("Ok(").expect("optimized begin return");
    assert_eq!(
        occurrences(begin, "Ok("),
        1,
        "real optimized begin retained an alternate successful early return",
    );
    let begin_tail = &begin[begin_return..];
    let (begin_return_arguments, begin_return_end) = balanced_call(begin_tail, "Ok(");
    assert_eq!(begin_return_arguments.len(), 1);
    assert_eq!(
        compact(begin_return_arguments[0]),
        begin_call,
        "real optimized begin conditionally replaced its status serializer result",
    );
    assert_eq!(
        compact(&begin_tail[begin_return_end..]),
        "}",
        "real optimized begin retained control/replacement logic after its exact return",
    );
    let finish = braced_item(&live_web, "fn finish_vision_stack_sharded_once(");
    let canary_results = named_initializer_binding(finish, "verify_mapped_qkv_canaries(");
    let legacy_diagnostics =
        named_initializer_binding(finish, "build_vision_stack_legacy_diagnostics_record(");
    let final_call = named_initializer_binding(finish, "vision_stack_qkv_diagnostics_json(");
    assert_unshadowed_binding(
        finish,
        direct_call_binding(finish, "vision_stack_qkv_diagnostics_json(", false),
    );
    let final_call_arguments =
        balanced_call_arguments(finish, "vision_stack_qkv_diagnostics_json(");
    assert_eq!(final_call_arguments.len(), 3);
    assert_eq!(
        compact(final_call_arguments[0]),
        format!("&{legacy_diagnostics}"),
    );
    assert_eq!(
        compact(final_call_arguments[1]),
        "session",
        "final diagnostics must consume the exact already-borrowed session",
    );
    assert_eq!(
        compact(final_call_arguments[2]),
        format!("&{canary_results}"),
    );
    assert_order(
        finish,
        &[
            "verify_mapped_qkv_canaries(",
            "build_vision_stack_legacy_diagnostics_record(",
            "vision_stack_qkv_diagnostics_json(",
        ],
    );
    let raw_finish = braced_item(web_source, "fn finish_vision_stack_sharded_once(");
    let web_ast = syn::parse_file(web_source).expect("Web evidence source must parse as Rust");
    let finish_ast = one_ast_function(&web_ast, "finish_vision_stack_sharded_once");
    let diagnostics_writes = all_balanced_call_arguments(raw_finish, "Reflect::set(")
        .into_iter()
        .filter(|arguments| arguments.len() == 3 && compact(arguments[1]).contains("diagnostics"))
        .collect::<Vec<_>>();
    assert_eq!(
        diagnostics_writes.len(),
        1,
        "actual finish must write diagnostics exactly once",
    );
    assert_eq!(
        ast_direct_call_count(finish_ast.block, "Ok"),
        1,
        "actual finish retained an alternate successful early return",
    );
    let final_flow = exact_flowing_names(raw_finish, final_call, &["JsValue::from_str"]);
    assert!(
        expression_is_one_exact_flow(
            diagnostics_writes[0][2],
            &final_flow,
            &["JsValue::from_str"],
        ),
        "actual finish conditionally replaced its final diagnostics serializer value",
    );

    for forbidden in [
        "VisionQkvSelectionEvidence {",
        "VisionQkvSelectionEvidencePropagation {",
        "build_vision_qkv_selection_evidence_propagation(",
        "semantic_graph_blake3_hex(",
    ] {
        assert_eq!(
            occurrences(&live_web, forbidden),
            usize::from(forbidden == "build_vision_qkv_selection_evidence_propagation("),
            "Web caller reauthored selection evidence via {forbidden}",
        );
    }
}

fn assert_web_physical_effect_sink_source(source: &str) {
    let live = live_rust_source(source);
    let ast = syn::parse_file(source).expect("Web physical effect sink must parse as Rust");
    let implementations = all_braced_items(
        &live,
        "impl VisionQkvWebPhysicalCommandEffectSink for BrowserVisionQkvPhysicalCommandEffectSink",
    );
    assert_eq!(
        implementations.len(),
        1,
        "Web must have one exact typed physical effect-sink implementation",
    );
    let implementation = implementations[0];
    assert_branchless_web_physical_orchestration(implementation, "Web physical effect sink");
    let methods = source_functions(implementation);
    assert_eq!(
        methods
            .iter()
            .map(|method| method.name)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "apply_copy_buffer",
            "apply_create_bind_group",
            "apply_create_buffer",
            "apply_map_range",
            "store_created_bind_group",
            "store_created_buffer",
        ]),
        "Web effect sink must implement exactly four variant effects and two typed stores",
    );

    for (method_name, adapter, returns_created) in [
        (
            "apply_create_buffer",
            "apply_vision_qkv_web_create_buffer_command(",
            true,
        ),
        (
            "apply_create_bind_group",
            "apply_vision_qkv_web_create_bind_group_command(",
            true,
        ),
        (
            "apply_copy_buffer",
            "apply_vision_qkv_web_copy_buffer_command(",
            false,
        ),
        (
            "apply_map_range",
            "apply_vision_qkv_web_map_range_command(",
            false,
        ),
    ] {
        let method = methods
            .iter()
            .find(|method| method.name == method_name)
            .unwrap();
        let ast_method = one_ast_function(&ast, method_name);
        let header = function_header(*method);
        assert!(header.contains("command_index"));
        assert!(header.contains("&VisionQkvWebPhysicalCommand"));
        assert_eq!(
            occurrences(method.body, adapter),
            1,
            "{method_name} must call its exact typed adapter once",
        );
        let arguments = balanced_call_arguments(method.body, adapter);
        assert_eq!(arguments.len(), 1);
        assert_eq!(compact(arguments[0]), "command");
        assert_eq!(occurrences(method.body, "Ok("), 1);
        let ok = balanced_call_arguments(method.body, "Ok(");
        assert_eq!(ok.len(), 1);
        if returns_created {
            let created = named_initializer_binding(method.body, adapter);
            assert_eq!(
                compact(ok[0]),
                created,
                "{method_name} reconstructed the typed adapter result",
            );
            assert!(!compact(method.body).contains(&format!("{created}.clone(")));
        } else {
            assert_eq!(compact(ok[0]), "()");
        }
        for forbidden in [
            "create_buffer",
            "create_bind_group",
            "copy_buffer_to_buffer",
            "get_mapped_range",
            "store_vision_qkv_web_created_buffer",
            "store_vision_qkv_web_created_bind_group",
        ] {
            assert_eq!(
                ast_call_or_method_count(ast_method.block, forbidden),
                0,
                "{method_name} reaches alternate raw/store sink {forbidden}",
            );
        }

        let allowed = reachable_functions(&live, adapter.trim_end_matches('('))
            .into_iter()
            .map(|function| function.name)
            .collect::<BTreeSet<_>>();
        for helper in reachable_helpers_from_fragment(&live, method.body) {
            assert!(
                helper.name == method_name || allowed.contains(helper.name),
                "{method_name} reaches wrapper/filter helper {}",
                helper.name,
            );
            let original = source_functions(source)
                .into_iter()
                .find(|candidate| candidate.name == helper.name)
                .unwrap_or(helper);
            let tokens = identifier_tokens(original.body);
            assert!(
                !tokens.contains("if") && !tokens.contains("cfg"),
                "{method_name} reachable graph conditionally gates {}",
                helper.name,
            );
            assert!(
                !compact(original.body).contains(".commands("),
                "{method_name} reachable graph replans commands in {}",
                helper.name,
            );
        }
    }

    for (method_name, store) in [
        (
            "store_created_buffer",
            "store_vision_qkv_web_created_buffer(",
        ),
        (
            "store_created_bind_group",
            "store_vision_qkv_web_created_bind_group(",
        ),
    ] {
        let method = methods
            .iter()
            .find(|method| method.name == method_name)
            .unwrap();
        let ast_method = one_ast_function(&ast, method_name);
        let header = function_header(*method);
        for parameter in ["command_index", "&VisionQkvWebPhysicalCommand", "created"] {
            assert!(
                header.contains(parameter),
                "{method_name} omitted {parameter}"
            );
        }
        assert_eq!(occurrences(method.body, store), 1);
        let store_arguments = balanced_call_arguments(method.body, store);
        assert_eq!(store_arguments.len(), 1);
        assert_eq!(
            compact(store_arguments[0]),
            "created",
            "{method_name} reconstructed or cloned its linear typed result",
        );
        assert_eq!(occurrences(method.body, "Ok("), 1);
        assert_eq!(
            compact(balanced_call_arguments(method.body, "Ok(")[0]),
            "()"
        );
        assert!(!compact(method.body).contains("created.clone("));
        for forbidden in [
            "apply_vision_qkv_web_create_buffer_command",
            "apply_vision_qkv_web_create_bind_group_command",
            "apply_vision_qkv_web_copy_buffer_command",
            "apply_vision_qkv_web_map_range_command",
            "create_buffer",
            "create_bind_group",
            "copy_buffer_to_buffer",
            "get_mapped_range",
        ] {
            assert_eq!(
                ast_call_or_method_count(ast_method.block, forbidden),
                0,
                "{method_name} reaches alternate effect {forbidden}",
            );
        }
        let allowed = reachable_functions(&live, store.trim_end_matches('('))
            .into_iter()
            .map(|function| function.name)
            .collect::<BTreeSet<_>>();
        for helper in reachable_helpers_from_fragment(&live, method.body) {
            assert!(
                helper.name == method_name || allowed.contains(helper.name),
                "{method_name} reaches wrapper/filter helper {}",
                helper.name,
            );
        }
    }

    let functions = source_functions(&live);
    for (root, typed_phase) in [
        (
            "apply_vision_qkv_web_start_commands",
            "VisionQkvWebPhysicalCommandPhase::Start",
        ),
        (
            "apply_vision_qkv_web_layer_commands",
            "VisionQkvWebPhysicalCommandPhase::Layer{layer_index}",
        ),
        (
            "apply_vision_qkv_web_finish_commands",
            "VisionQkvWebPhysicalCommandPhase::Finish",
        ),
    ] {
        let roots = functions
            .iter()
            .filter(|function| function.name == root)
            .collect::<Vec<_>>();
        assert_eq!(roots.len(), 1, "missing or duplicate Web phase root {root}");
        let phase_root = roots[0];
        let ast_phase_root = one_ast_function(&ast, root);
        assert_branchless_web_physical_orchestration(phase_root.body, root);
        assert!(function_header(*phase_root).contains("&VisionQkvWebPhysicalCommandPlan"));
        let sink = named_initializer_binding(
            phase_root.body,
            "BrowserVisionQkvPhysicalCommandEffectSink {",
        );
        assert_eq!(
            occurrences(phase_root.body, "execute_vision_qkv_web_physical_commands(",),
            1,
        );
        let arguments =
            balanced_call_arguments(phase_root.body, "execute_vision_qkv_web_physical_commands(")
                .into_iter()
                .filter(|argument| !argument.is_empty())
                .collect::<Vec<_>>();
        assert_eq!(arguments.len(), 3);
        assert_eq!(compact(arguments[0]), "plan");
        assert_eq!(compact(arguments[1]), typed_phase);
        assert_eq!(compact(arguments[2]), format!("&mut{sink}"));
        for forbidden in [
            "apply_vision_qkv_web_create_buffer_command",
            "apply_vision_qkv_web_create_bind_group_command",
            "apply_vision_qkv_web_copy_buffer_command",
            "apply_vision_qkv_web_map_range_command",
            "store_vision_qkv_web_created_buffer",
            "store_vision_qkv_web_created_bind_group",
            "create_buffer",
            "create_bind_group",
            "copy_buffer_to_buffer",
            "get_mapped_range",
        ] {
            assert_eq!(
                ast_call_or_method_count(ast_phase_root.block, forbidden),
                0,
                "{root} retained caller-owned adapter/store route {forbidden}",
            );
        }
        for helper in reachable_helpers_from_fragment(&live, phase_root.body) {
            assert_eq!(
                helper.name, root,
                "{root} reaches wrapper/filter helper {} outside the host executor",
                helper.name,
            );
        }
    }
}

fn ast_pattern_binding_count(block: &syn::Block, expected: &str) -> usize {
    struct BindingCounter<'a> {
        expected: &'a str,
        count: usize,
    }
    impl<'ast> syn::visit::Visit<'ast> for BindingCounter<'_> {
        fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
            if ast_ident_is(&pattern.ident, self.expected) {
                self.count += 1;
            }
            syn::visit::visit_pat_ident(self, pattern);
        }
    }

    let mut counter = BindingCounter { expected, count: 0 };
    syn::visit::Visit::visit_block(&mut counter, block);
    counter.count
}

fn exact_reference_parameter_index(
    function: AstFunction<'_>,
    name: &str,
    expected_path: &str,
    expected_type_lifetime: Option<&str>,
    mutable: bool,
) -> usize {
    let typed_inputs = function
        .signature
        .inputs
        .iter()
        .filter_map(|input| match input {
            syn::FnArg::Typed(input) => Some(input),
            syn::FnArg::Receiver(_) => None,
        })
        .enumerate()
        .collect::<Vec<_>>();
    let named = typed_inputs
        .iter()
        .filter(|(_, input)| {
            matches!(
                input.pat.as_ref(),
                syn::Pat::Ident(pattern)
                    if input.attrs.is_empty()
                        && pattern.attrs.is_empty()
                        && ast_ident_is(&pattern.ident, name)
                        && pattern.by_ref.is_none()
                        && pattern.mutability.is_none()
                        && pattern.subpat.is_none()
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        named.len(),
        1,
        "{} must have one exact immutable {name} parameter",
        function.name,
    );
    let (parameter_index, parameter) = *named[0];
    let syn::Type::Reference(reference) = parameter.ty.as_ref() else {
        panic!("{} {name} must be a reference", function.name);
    };
    assert_eq!(reference.mutability.is_some(), mutable);
    assert!(reference.lifetime.is_none());
    assert!(
        ast_type_is_exact_path(&reference.elem, expected_path, expected_type_lifetime,),
        "{} {name} must be exact &{expected_path}",
        function.name,
    );
    let exact_type_count = typed_inputs
        .iter()
        .filter(|(_, input)| {
            matches!(
                input.ty.as_ref(),
                syn::Type::Reference(reference)
                    if reference.mutability.is_some() == mutable
                        && reference.lifetime.is_none()
                        && ast_type_is_exact_path(
                            &reference.elem,
                            expected_path,
                            expected_type_lifetime,
                        )
            )
        })
        .count();
    assert_eq!(
        exact_type_count, 1,
        "{} must not accept a duplicate/decoy &{expected_path}",
        function.name,
    );
    assert_eq!(
        ast_pattern_binding_count(function.block, name),
        0,
        "{} shadows or reconstructs its authenticated {name} parameter",
        function.name,
    );
    parameter_index
}

fn exact_immutable_reference_parameter_index(
    function: AstFunction<'_>,
    name: &str,
    expected_path: &str,
    expected_type_lifetime: Option<&str>,
) -> usize {
    exact_reference_parameter_index(function, name, expected_path, expected_type_lifetime, false)
}

fn assert_web_layer_physical_context_wiring(source: &str) -> usize {
    let live = live_rust_source(source);
    let ast = syn::parse_file(source).expect("Web layer physical context source must parse");
    let functions = source_functions(&live);
    let layer_phases = functions
        .iter()
        .filter(|function| function.name == "apply_vision_qkv_web_layer_commands")
        .collect::<Vec<_>>();
    assert_eq!(
        layer_phases.len(),
        1,
        "Web layer typed-command phase root drifted"
    );
    let layer_phase = layer_phases[0];
    let ast_phase_root = one_ast_function(&ast, "apply_vision_qkv_web_layer_commands");
    assert!(
        ast_phase_root.owner.is_none(),
        "Web layer typed-command phase root must remain a free function",
    );
    let context_parameter_index = exact_immutable_reference_parameter_index(
        ast_phase_root,
        "context",
        "BrowserVisionQkvLayerResolutionContext",
        Some("_"),
    );
    let sink = braced_item(
        layer_phase.body,
        "BrowserVisionQkvPhysicalCommandEffectSink {",
    );
    assert_eq!(
        compact(
            struct_field_initializer(sink, "context")
                .expect("typed layer phase sink omitted its exact context field"),
        ),
        "context",
        "typed layer phase reconstructed or substituted its resolution context",
    );
    assert_eq!(
        ast_call_or_method_count(
            ast_phase_root.block,
            "apply_vision_qkv_web_create_bind_group_command",
        ),
        0,
        "typed layer phase bypassed the common executor/effect sink",
    );

    let implementations = all_braced_items(
        &live,
        "impl VisionQkvWebPhysicalCommandEffectSink for BrowserVisionQkvPhysicalCommandEffectSink",
    );
    assert_eq!(implementations.len(), 1);
    let bind_methods = source_functions(implementations[0])
        .into_iter()
        .filter(|method| method.name == "apply_create_bind_group")
        .collect::<Vec<_>>();
    assert_eq!(
        bind_methods.len(),
        1,
        "typed physical effect sink bind-group method drifted"
    );
    let bind_method = bind_methods[0];
    let ast_bind_method = one_ast_function(&ast, "apply_create_bind_group");
    let _command_parameter_index = exact_immutable_reference_parameter_index(
        ast_bind_method,
        "command",
        "VisionQkvWebPhysicalCommand",
        None,
    );
    let bind_body = compact(bind_method.body);
    let adapter = "self.context.apply_vision_qkv_web_create_bind_group_command(";
    assert_eq!(
        occurrences(&bind_body, adapter),
        1,
        "typed effect sink does not invoke the bind-group adapter on its exact context",
    );
    let arguments = balanced_call_arguments(&bind_body, adapter);
    assert_eq!(arguments.len(), 1);
    assert_eq!(
        compact(arguments[0]),
        "command",
        "typed effect sink substituted the sealed bind-group command",
    );
    context_parameter_index
}

fn assert_web_layer_context_call_argument(
    source: &str,
    context_parameter_index: usize,
    expected_context_binding: &str,
) {
    let calls = all_balanced_call_arguments(source, "apply_vision_qkv_web_layer_commands(");
    assert_eq!(calls.len(), 1, "real fused layer phase call drifted");
    let arguments = &calls[0];
    assert!(
        context_parameter_index < arguments.len(),
        "real fused layer phase call omitted its context argument",
    );
    assert_eq!(
        compact(arguments[context_parameter_index]),
        format!("&{expected_context_binding}"),
        "real fused layer phase call did not pass the exact constructed context at the parameter consumed by the sink",
    );
}

fn authority_type_names(source: &str, type_name: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::from([type_name.to_owned()]);
    loop {
        let before = names.len();
        for (start, _) in source.match_indices("type ") {
            let declaration = &source[start + "type ".len()..];
            let Some(equals) = declaration.find('=') else {
                continue;
            };
            let Some(end) = declaration[equals + 1..].find(';') else {
                continue;
            };
            let alias = declaration[..equals]
                .trim()
                .split(['<', ' ', '\t', '\n'])
                .next()
                .unwrap_or_default();
            if alias.is_empty()
                || !alias
                    .bytes()
                    .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
            {
                continue;
            }
            let target = &declaration[equals + 1..equals + 1 + end];
            let target_tokens = identifier_tokens(target);
            if names
                .iter()
                .any(|name| target_tokens.contains(name.as_str()))
            {
                names.insert(alias.to_owned());
            }
        }
        if names.len() == before {
            return names;
        }
    }
}

fn top_level_semicolon_items(item: &str) -> Vec<&str> {
    let open = item.find('{').expect("reviewed braced item");
    let bytes = item.as_bytes();
    let mut brace_depth = 1_usize;
    let mut parentheses = 0_usize;
    let mut brackets = 0_usize;
    let mut string = false;
    let mut escaped = false;
    let mut start = open + 1;
    let mut items = Vec::new();
    let mut index = start;
    while index < bytes.len() {
        let byte = bytes[index];
        if string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                string = false;
            }
            index += 1;
            continue;
        }
        match byte {
            b'"' => string = true,
            b'(' => parentheses += 1,
            b')' => parentheses -= 1,
            b'[' => brackets += 1,
            b']' => brackets -= 1,
            b'{' => brace_depth += 1,
            b'}' if brace_depth == 1 => break,
            b'}' => brace_depth -= 1,
            b';' if brace_depth == 1 && parentheses == 0 && brackets == 0 => {
                items.push(item[start..=index].trim());
                start = index + 1;
            }
            _ => {}
        }
        index += 1;
    }
    items
}

fn line_started_braced_items<'a>(source: &'a str, prefixes: &[&str]) -> Vec<&'a str> {
    let mut items = Vec::new();
    let mut offset = 0_usize;
    for line in source.split_inclusive('\n') {
        let leading = line.len() - line.trim_start().len();
        let start = offset + leading;
        let trimmed = &source[start..offset + line.len()];
        if prefixes.iter().any(|prefix| trimmed.starts_with(prefix)) {
            let item = braced_item(
                &source[start..],
                prefixes
                    .iter()
                    .find(|prefix| trimmed.starts_with(**prefix))
                    .unwrap(),
            );
            items.push(item);
        }
        offset += line.len();
    }
    items
}

fn implementation_items(source: &str) -> Vec<&str> {
    line_started_braced_items(source, &["impl ", "impl<", "unsafe impl ", "unsafe impl<"])
}

fn is_trait_implementation(header: &str) -> bool {
    header.split_ascii_whitespace().any(|token| token == "for")
}

fn implementation_mentions_authority(header: &str, authority_names: &BTreeSet<String>) -> bool {
    let tokens = identifier_tokens(header);
    authority_names
        .iter()
        .any(|name| tokens.contains(name.as_str()))
}

fn trait_self_mentions_authority(header: &str, authority_names: &BTreeSet<String>) -> bool {
    let words = header.split_ascii_whitespace().collect::<Vec<_>>();
    let Some(for_index) = words.iter().rposition(|word| *word == "for") else {
        return false;
    };
    let self_type = words[for_index + 1..].join(" ");
    let self_tokens = identifier_tokens(&self_type);
    authority_names
        .iter()
        .any(|name| self_tokens.contains(name.as_str()))
}

fn authority_literal(body: &str, authority_names: &BTreeSet<String>) -> bool {
    let normalized = compact(body);
    authority_names
        .iter()
        .any(|name| normalized.contains(&format!("{name}{{")))
}

fn returned_authority(
    header: &str,
    authority_names: &BTreeSet<String>,
    self_is_authority: bool,
) -> bool {
    let Some((_, returned)) = header.split_once("->") else {
        return false;
    };
    let returned_tokens = identifier_tokens(returned);
    authority_names
        .iter()
        .any(|name| returned_tokens.contains(name.as_str()))
        || self_is_authority && returned_tokens.contains("Self")
}

fn assert_no_public_authority_const_or_static(
    source: &str,
    type_name: &str,
    authority_names: &BTreeSet<String>,
) {
    for (start, _) in source.match_indices("pub ") {
        let declaration = &source[start..];
        let normalized_prefix =
            compact(&declaration[..declaration.find(';').unwrap_or(declaration.len())]);
        if normalized_prefix.starts_with("pubconstfn")
            || !(normalized_prefix.starts_with("pubconst")
                || normalized_prefix.starts_with("pubstatic")
                || normalized_prefix.starts_with("pubunsafestatic"))
        {
            continue;
        }
        let end = declaration
            .find(';')
            .unwrap_or_else(|| panic!("public const/static declaration is unterminated"));
        let item = &declaration[..=end];
        let tokens = identifier_tokens(item);
        assert!(
            !authority_names
                .iter()
                .any(|name| tokens.contains(name.as_str()))
                && !authority_literal(item, authority_names),
            "{type_name} escaped through public const/static item {item:?}",
        );
    }
}

fn assert_no_public_construction_trait(
    source: &str,
    type_name: &str,
    authority_names: &BTreeSet<String>,
) {
    let normalized = compact(source);
    for (start, _) in normalized.match_indices("impl") {
        let Some(brace_offset) = normalized[start..].find('{') else {
            continue;
        };
        let header = &normalized[start..start + brace_offset];
        if authority_names
            .iter()
            .any(|name| header.contains(&format!("for{name}")))
        {
            for trait_name in [
                "From<",
                "TryFrom<",
                "Into<",
                "TryInto<",
                "FromStr",
                "Default",
                "AsRef<",
                "AsMut<",
                "Borrow<",
                "BorrowMut<",
                "ToOwned",
                "FromIterator<",
                "Extend<",
                "Deserialize",
                "Build",
                "Builder",
            ] {
                assert!(
                    !header.contains(trait_name),
                    "{type_name} exposed construction through {trait_name}",
                );
            }
        }
    }

    for public_trait in line_started_braced_items(
        source,
        &["pub trait ", "pub unsafe trait ", "pub auto trait "],
    ) {
        let tokens = identifier_tokens(public_trait);
        assert!(
            !authority_names
                .iter()
                .any(|name| tokens.contains(name.as_str())),
            "{type_name} escaped through a custom public trait method or associated item",
        );
    }

    for implementation in implementation_items(source) {
        let header = implementation
            .split('{')
            .next()
            .expect("implementation header");
        let is_trait_implementation = is_trait_implementation(header);
        let self_is_authority = trait_self_mentions_authority(header, authority_names);

        if is_trait_implementation {
            for associated in top_level_semicolon_items(implementation) {
                let tokens = identifier_tokens(associated);
                if !tokens.contains("const")
                    && !tokens.contains("static")
                    && !tokens.contains("type")
                {
                    continue;
                }
                let exposes_named_authority = authority_names
                    .iter()
                    .any(|name| tokens.contains(name.as_str()));
                let exposes_self = self_is_authority && tokens.contains("Self");
                assert!(
                    !exposes_named_authority
                        && !exposes_self
                        && !authority_literal(associated, authority_names),
                    "{type_name} escaped through a trait-associated const/static item",
                );
            }
        }

        if is_trait_implementation {
            for method in source_functions(implementation) {
                assert!(
                    !(returned_authority(
                        function_header(method),
                        authority_names,
                        self_is_authority,
                    ) || authority_literal(method.body, authority_names)
                        || self_is_authority && compact(method.body).contains("Self{")),
                    "{type_name} exposed construction through trait method {}",
                    method.name,
                );
            }
        }
    }
}

fn assert_complete_opaque_authority_surface_live(
    source: &str,
    type_name: &str,
    accessors: &[&str],
    factory: &str,
    returning_accessors: &[&str],
) {
    let structure_start = source
        .find(&format!("pub struct {type_name}"))
        .unwrap_or_else(|| panic!("missing opaque authority {type_name}"));
    let structure = braced_item(source, &format!("pub struct {type_name}"));
    assert!(
        structure.lines().skip(1).all(|line| {
            let trimmed = line.trim_start();
            !trimmed.starts_with("pub ") && !trimmed.starts_with("pub(")
        }),
        "{type_name} authority fields must remain private",
    );
    let derive_start = ['}', ';']
        .into_iter()
        .filter_map(|boundary| source[..structure_start].rfind(boundary))
        .max()
        .map_or(0, |position| position + 1);
    let attributes = format!(
        "{}{}",
        compact(&source[derive_start..structure_start]),
        compact(structure),
    );
    for forbidden in ["Deserialize", "#[serde", "serde("] {
        assert!(
            !attributes.contains(forbidden),
            "{type_name} exposed deserialization through {forbidden}",
        );
    }

    let authority_names = authority_type_names(source, type_name);
    let implementations = implementation_items(source)
        .into_iter()
        .filter(|implementation| {
            let header = implementation
                .split('{')
                .next()
                .expect("implementation header");
            !is_trait_implementation(header)
                && implementation_mentions_authority(header, &authority_names)
        })
        .collect::<Vec<_>>();
    assert!(
        !implementations.is_empty(),
        "{type_name} must have one reviewed accessor implementation",
    );
    let actual_accessors = implementations
        .iter()
        .flat_map(|implementation| public_inherent_function_names(implementation))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual_accessors,
        accessors.iter().copied().collect::<BTreeSet<_>>(),
        "{type_name} public inherent API must be exactly the read-only accessor allowlist",
    );
    for implementation in &implementations {
        let normalized = compact(implementation);
        for prefix in ["pubconst", "pubstatic"] {
            for (start, _) in normalized.match_indices(prefix) {
                let tail = &normalized[start + prefix.len()..];
                assert!(
                    prefix == "pubconst" && tail.starts_with("fn"),
                    "{type_name} exposed associated const/static construction state",
                );
            }
        }
        assert!(
            !normalized.contains("#[wasm_bindgen(constructor)]"),
            "{type_name} exposed a wasm constructor",
        );
    }
    assert!(
        !source.contains(&format!("{type_name}Builder")),
        "{type_name} exposed a public construction builder",
    );
    assert_no_public_authority_const_or_static(source, type_name, &authority_names);
    assert_no_public_construction_trait(source, type_name, &authority_names);

    let allowed = std::iter::once(factory)
        .chain(returning_accessors.iter().copied())
        .collect::<BTreeSet<_>>();
    let public_functions = public_source_functions(source);
    let constructing = public_functions
        .iter()
        .copied()
        .filter(|function| {
            returned_authority(function.header, &authority_names, false)
                || authority_literal(function.body, &authority_names)
        })
        .collect::<Vec<_>>();
    let inherent_constructors = implementations
        .iter()
        .flat_map(|implementation| public_source_functions(implementation))
        .filter(|function| {
            returned_authority(function.header, &authority_names, true)
                || authority_literal(function.body, &authority_names)
                || compact(function.body).contains("Self{")
        })
        .collect::<Vec<_>>();
    assert!(
        constructing.iter().any(|function| function.name == factory)
            || inherent_constructors
                .iter()
                .any(|function| function.name == factory),
        "{type_name} is not constructed by its one sanctioned factory {factory}",
    );
    for function in constructing.into_iter().chain(inherent_constructors) {
        assert!(
            allowed.contains(function.name),
            "{} is an unsanctioned pub async/unsafe/const/free/inherent constructor or return path for {type_name}",
            function.name,
        );
    }
}

fn assert_complete_opaque_authority_surface(
    source: &str,
    type_name: &str,
    accessors: &[&str],
    factory: &str,
    returning_accessors: &[&str],
) {
    let live = live_rust_source(source);
    assert_complete_opaque_authority_surface_live(
        &live,
        type_name,
        accessors,
        factory,
        returning_accessors,
    );
}

fn unique_function_containing<'a>(
    functions: &'a [SourceFunction<'a>],
    required: &[&str],
) -> SourceFunction<'a> {
    let matches = functions
        .iter()
        .copied()
        .filter(|function| {
            required
                .iter()
                .all(|required| function.body.contains(required))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        matches.len(),
        1,
        "expected one function containing {required:?}, found {}",
        matches.len()
    );
    matches[0]
}

fn reachable_functions<'a>(source: &'a str, root: &str) -> Vec<SourceFunction<'a>> {
    let functions = source_functions(source);
    let by_name = functions
        .iter()
        .copied()
        .map(|function| (function.name, function))
        .collect::<BTreeMap<_, _>>();
    assert!(by_name.contains_key(root), "missing source root {root}");

    let mut queue = VecDeque::from([root]);
    let mut visited = BTreeSet::new();
    while let Some(name) = queue.pop_front() {
        if !visited.insert(name) {
            continue;
        }
        let body = by_name[name].body;
        for candidate in by_name.keys().copied() {
            if source_calls_named(body, candidate) {
                queue.push_back(candidate);
            }
        }
    }
    visited.into_iter().map(|name| by_name[name]).collect()
}

fn reachable_helpers_from_fragment<'a>(source: &'a str, fragment: &str) -> Vec<SourceFunction<'a>> {
    let functions = source_functions(source);
    let mut queue = functions
        .iter()
        .enumerate()
        .filter(|(_, function)| source_calls_named(fragment, function.name))
        .map(|(index, _)| index)
        .collect::<VecDeque<_>>();
    let mut visited = BTreeSet::new();
    while let Some(index) = queue.pop_front() {
        if !visited.insert(index) {
            continue;
        }
        for (candidate_index, candidate) in functions.iter().enumerate() {
            if source_calls_named(functions[index].body, candidate.name) {
                queue.push_back(candidate_index);
            }
        }
    }
    visited.into_iter().map(|index| functions[index]).collect()
}

#[derive(Clone, Copy)]
struct AstFunction<'a> {
    name: &'a syn::Ident,
    signature: &'a syn::Signature,
    block: &'a syn::Block,
    owner: Option<&'a syn::Type>,
}

fn collect_ast_functions_from_items<'a>(
    items: &'a [syn::Item],
    functions: &mut Vec<AstFunction<'a>>,
) {
    for item in items {
        match item {
            syn::Item::Fn(function) => functions.push(AstFunction {
                name: &function.sig.ident,
                signature: &function.sig,
                block: &function.block,
                owner: None,
            }),
            syn::Item::Impl(implementation) => {
                for item in &implementation.items {
                    if let syn::ImplItem::Fn(function) = item {
                        functions.push(AstFunction {
                            name: &function.sig.ident,
                            signature: &function.sig,
                            block: &function.block,
                            owner: Some(&implementation.self_ty),
                        });
                    }
                }
            }
            syn::Item::Mod(module) => {
                if let Some((_, items)) = &module.content {
                    collect_ast_functions_from_items(items, functions);
                }
            }
            _ => {}
        }
    }
}

fn ast_functions(file: &syn::File) -> Vec<AstFunction<'_>> {
    let mut functions = Vec::new();
    collect_ast_functions_from_items(&file.items, &mut functions);
    functions
}

fn ast_ident_name(identifier: &syn::Ident) -> String {
    identifier.unraw().to_string()
}

fn ast_ident_is(identifier: &syn::Ident, expected: &str) -> bool {
    identifier.unraw() == expected
}

fn one_ast_function<'a>(file: &'a syn::File, name: &str) -> AstFunction<'a> {
    let matches = ast_functions(file)
        .into_iter()
        .filter(|function| ast_ident_is(function.name, name))
        .collect::<Vec<_>>();
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one AST function {name}, found {}",
        matches.len(),
    );
    matches[0]
}

fn ast_unique_struct_expression<'a>(
    function: AstFunction<'a>,
    expected_path: &str,
) -> &'a syn::ExprStruct {
    struct StructCollector<'ast> {
        expressions: Vec<&'ast syn::ExprStruct>,
    }
    impl<'ast> syn::visit::Visit<'ast> for StructCollector<'ast> {
        fn visit_expr_struct(&mut self, expression: &'ast syn::ExprStruct) {
            self.expressions.push(expression);
            syn::visit::visit_expr_struct(self, expression);
        }
    }

    let mut collector = StructCollector {
        expressions: Vec::new(),
    };
    syn::visit::Visit::visit_block(&mut collector, function.block);
    let matches = collector
        .expressions
        .into_iter()
        .filter(|expression| ast_path_name(&expression.path) == expected_path)
        .collect::<Vec<_>>();
    assert_eq!(
        matches.len(),
        1,
        "{} must construct exactly one {expected_path}",
        function.name,
    );
    matches[0]
}

fn ast_unique_tail_struct_expression<'a>(
    function: AstFunction<'a>,
    expected_path: &str,
) -> &'a syn::ExprStruct {
    let unique = ast_unique_struct_expression(function, expected_path);
    let syn::Expr::Struct(tail) = ast_without_try(ast_tail_expression(function.block)) else {
        panic!(
            "{} must return its {expected_path} as the exact tail expression",
            function.name
        );
    };
    assert_eq!(ast_path_name(&tail.path), expected_path);
    assert!(
        std::ptr::eq(unique, tail),
        "{} discarded the unique {expected_path} before returning another value",
        function.name,
    );
    tail
}

fn ast_unique_method_call<'a>(
    function: AstFunction<'a>,
    expected_method: &str,
) -> &'a syn::ExprMethodCall {
    struct MethodCollector<'ast> {
        calls: Vec<&'ast syn::ExprMethodCall>,
    }
    impl<'ast> syn::visit::Visit<'ast> for MethodCollector<'ast> {
        fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
            self.calls.push(call);
            syn::visit::visit_expr_method_call(self, call);
        }
    }

    let mut collector = MethodCollector { calls: Vec::new() };
    syn::visit::Visit::visit_block(&mut collector, function.block);
    let matches = collector
        .calls
        .into_iter()
        .filter(|call| ast_ident_is(&call.method, expected_method))
        .collect::<Vec<_>>();
    assert_eq!(
        matches.len(),
        1,
        "{} must call {expected_method} exactly once",
        function.name,
    );
    matches[0]
}

fn ast_call_or_method_count(block: &syn::Block, expected: &str) -> usize {
    struct CallCounter<'a> {
        expected: &'a str,
        count: usize,
    }
    impl<'ast> syn::visit::Visit<'ast> for CallCounter<'_> {
        fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
            if ast_expr_path(&call.func).is_some_and(|path| {
                path.segments
                    .last()
                    .is_some_and(|segment| ast_ident_is(&segment.ident, self.expected))
            }) {
                self.count += 1;
            }
            syn::visit::visit_expr_call(self, call);
        }

        fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
            if ast_ident_is(&call.method, self.expected) {
                self.count += 1;
            }
            syn::visit::visit_expr_method_call(self, call);
        }
    }

    let mut counter = CallCounter { expected, count: 0 };
    syn::visit::Visit::visit_block(&mut counter, block);
    counter.count
}

fn ast_path_name(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|segment| ast_ident_name(&segment.ident))
        .collect::<Vec<_>>()
        .join("::")
}

fn ast_expr_path(expr: &syn::Expr) -> Option<&syn::Path> {
    match expr {
        syn::Expr::Path(path) if path.qself.is_none() => Some(&path.path),
        syn::Expr::Group(group) => ast_expr_path(&group.expr),
        syn::Expr::Paren(parenthesized) => ast_expr_path(&parenthesized.expr),
        _ => None,
    }
}

fn ast_expr_is_path(expr: &syn::Expr, expected: &str) -> bool {
    ast_expr_path(expr).is_some_and(|path| ast_path_name(path) == expected)
}

fn ast_expr_field_path(expr: &syn::Expr) -> Option<Vec<String>> {
    match expr {
        syn::Expr::Path(path) if path.qself.is_none() => Some(
            path.path
                .segments
                .iter()
                .map(|segment| ast_ident_name(&segment.ident))
                .collect(),
        ),
        syn::Expr::Field(field) => {
            let mut path = ast_expr_field_path(&field.base)?;
            match &field.member {
                syn::Member::Named(name) => path.push(ast_ident_name(name)),
                syn::Member::Unnamed(index) => path.push(index.index.to_string()),
            }
            Some(path)
        }
        syn::Expr::Group(group) => ast_expr_field_path(&group.expr),
        syn::Expr::Paren(parenthesized) => ast_expr_field_path(&parenthesized.expr),
        syn::Expr::Reference(reference) => ast_expr_field_path(&reference.expr),
        _ => None,
    }
}

fn ast_expr_is_field_path(expr: &syn::Expr, expected: &[&str]) -> bool {
    ast_expr_field_path(expr)
        .is_some_and(|path| path.iter().map(String::as_str).eq(expected.iter().copied()))
}

fn ast_statement_expression(statement: &syn::Stmt) -> Option<&syn::Expr> {
    match statement {
        syn::Stmt::Expr(expression, _) => Some(expression),
        _ => None,
    }
}

fn ast_local_named<'a>(statement: &'a syn::Stmt, expected: &str) -> &'a syn::Local {
    let syn::Stmt::Local(local) = statement else {
        panic!("expected local binding {expected}");
    };
    let syn::Pat::Ident(pattern) = &local.pat else {
        panic!("expected one named local binding {expected}");
    };
    assert!(ast_ident_is(&pattern.ident, expected));
    assert!(pattern.subpat.is_none());
    local
}

fn ast_local_initializer(local: &syn::Local) -> &syn::Expr {
    local
        .init
        .as_ref()
        .map(|initializer| initializer.expr.as_ref())
        .expect("reviewed local must have an initializer")
}

fn ast_call_named<'a>(expr: &'a syn::Expr, expected: &str) -> &'a syn::ExprCall {
    let syn::Expr::Call(call) = expr else {
        panic!("expected direct call {expected}");
    };
    let path = ast_expr_path(&call.func).expect("direct call must use a path");
    assert_eq!(ast_path_name(path), expected);
    call
}

fn ast_method_named<'a>(expr: &'a syn::Expr, expected: &str) -> &'a syn::ExprMethodCall {
    let syn::Expr::MethodCall(call) = expr else {
        panic!("expected method call {expected}");
    };
    assert!(ast_ident_is(&call.method, expected));
    call
}

fn ast_reference_to_path(expr: &syn::Expr, expected: &str) -> bool {
    matches!(expr, syn::Expr::Reference(reference)
        if reference.mutability.is_none() && ast_expr_is_path(&reference.expr, expected))
}

fn type_identifiers(ty: &syn::Type) -> BTreeSet<String> {
    struct TypeIdentifiers {
        names: BTreeSet<String>,
    }

    impl<'ast> syn::visit::Visit<'ast> for TypeIdentifiers {
        fn visit_path_segment(&mut self, segment: &'ast syn::PathSegment) {
            self.names.insert(ast_ident_name(&segment.ident));
            syn::visit::visit_path_segment(self, segment);
        }
    }

    let mut identifiers = TypeIdentifiers {
        names: BTreeSet::new(),
    };
    syn::visit::Visit::visit_type(&mut identifiers, ty);
    identifiers.names
}

fn ast_without_try(expr: &syn::Expr) -> &syn::Expr {
    match expr {
        syn::Expr::Try(expression) => ast_without_try(&expression.expr),
        syn::Expr::Group(group) => ast_without_try(&group.expr),
        syn::Expr::Paren(parenthesized) => ast_without_try(&parenthesized.expr),
        _ => expr,
    }
}

fn ast_map_err_js_error_receiver(expr: &syn::Expr) -> &syn::Expr {
    let call = ast_method_named(expr, "map_err");
    assert!(call.turbofish.is_none());
    assert_eq!(call.args.len(), 1);
    assert!(ast_expr_is_path(call.args.first().unwrap(), "js_error"));
    &call.receiver
}

fn assert_ast_begin_common_tail(
    expression: &syn::Expr,
    expected_strategy: &str,
    expected_hardening: Option<&str>,
) {
    let common = ast_method_named(
        ast_map_err_js_error_receiver(expression),
        "begin_vision_stack_sharded",
    );
    assert!(ast_expr_is_path(&common.receiver, "self"));
    assert_eq!(common.args.len(), 3);
    assert!(ast_expr_is_path(&common.args[0], "manifest_json"));
    assert!(ast_expr_is_path(&common.args[1], expected_strategy));
    match expected_hardening {
        None => assert!(ast_expr_is_path(&common.args[2], "None")),
        Some(binding) => {
            let some = ast_call_named(&common.args[2], "Some");
            assert_eq!(some.args.len(), 1);
            assert!(ast_expr_is_path(some.args.first().unwrap(), binding));
        }
    }
}

fn assert_ast_parse_activation_local(statement: &syn::Stmt) {
    let local = ast_local_named(statement, "activation_strategy");
    let parsed = ast_without_try(ast_local_initializer(local));
    let parser = ast_call_named(
        ast_map_err_js_error_receiver(parsed),
        "parse_vision_stack_activation_strategy",
    );
    assert_eq!(parser.args.len(), 1);
    assert!(ast_expr_is_path(
        parser.args.first().unwrap(),
        "activation_strategy"
    ));
}

fn assert_ast_parse_hardening_local(statement: &syn::Stmt) {
    let local = ast_local_named(statement, "memory_hardening");
    let parsed = ast_without_try(ast_local_initializer(local));
    let parser = ast_method_named(ast_map_err_js_error_receiver(parsed), "parse");
    assert!(ast_expr_is_path(&parser.receiver, "memory_hardening"));
    assert_eq!(parser.args.len(), 0);
    let turbofish = parser
        .turbofish
        .as_ref()
        .expect("memory-hardening parser must bind its exact type");
    assert_eq!(turbofish.args.len(), 1);
    let syn::GenericArgument::Type(ty) = turbofish.args.first().unwrap() else {
        panic!("memory-hardening parser type drifted");
    };
    assert!(type_identifiers(ty).contains("VisionStackMemoryHardening"));
}

fn ast_direct_call_count(block: &syn::Block, expected: &str) -> usize {
    struct CallCounter<'a> {
        expected: &'a str,
        count: usize,
    }
    impl<'ast> syn::visit::Visit<'ast> for CallCounter<'_> {
        fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
            if ast_expr_path(&call.func).is_some_and(|path| ast_path_name(path) == self.expected) {
                self.count += 1;
            }
            syn::visit::visit_expr_call(self, call);
        }
    }
    let mut counter = CallCounter { expected, count: 0 };
    syn::visit::Visit::visit_block(&mut counter, block);
    counter.count
}

fn ast_local_direct_call_binding<'a>(
    block: &'a syn::Block,
    callee: &str,
) -> (&'a syn::Ident, &'a syn::ExprCall) {
    let mut matches = Vec::new();
    for statement in &block.stmts {
        let syn::Stmt::Local(local) = statement else {
            continue;
        };
        let syn::Pat::Ident(pattern) = &local.pat else {
            continue;
        };
        let initializer = ast_without_try(ast_local_initializer(local));
        let syn::Expr::Call(call) = initializer else {
            continue;
        };
        if ast_expr_path(&call.func).is_some_and(|path| ast_path_name(path) == callee) {
            matches.push((&pattern.ident, call));
        }
    }
    assert_eq!(
        matches.len(),
        1,
        "{callee} must initialize exactly one top-level authority binding",
    );
    matches[0]
}

fn ast_tail_expression(block: &syn::Block) -> &syn::Expr {
    ast_statement_expression(
        block
            .stmts
            .last()
            .expect("reviewed function must have a tail expression"),
    )
    .expect("reviewed function must end in an expression")
}

fn ast_tail_direct_call<'a>(block: &'a syn::Block, expected: &str) -> &'a syn::ExprCall {
    ast_call_named(ast_without_try(ast_tail_expression(block)), expected)
}

fn ast_type_nominal_name(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    path.path
        .segments
        .last()
        .map(|segment| ast_ident_name(&segment.ident))
}

fn assert_final_diagnostics_causal_flow_ast(source: &str) {
    let file = syn::parse_file(source).expect("final diagnostics Web source must parse as an AST");
    let diagnostics = one_ast_function(&file, "vision_stack_qkv_diagnostics_json");
    assert_eq!(
        diagnostics.block.stmts.len(),
        2,
        "final diagnostics must bind the session selection once, then tail-match it",
    );

    let mut inputs = diagnostics.signature.inputs.iter();
    for (expected_name, expected_type) in [
        (
            "legacy_diagnostics",
            Some("VisionStackLegacyDiagnosticsRecord"),
        ),
        ("session", Some("BrowserVisionStackSession")),
        ("canary_results", None),
    ] {
        let syn::FnArg::Typed(input) = inputs
            .next()
            .unwrap_or_else(|| panic!("final diagnostics omitted {expected_name}"))
        else {
            panic!("final diagnostics input {expected_name} cannot be self");
        };
        let syn::Pat::Ident(binding) = input.pat.as_ref() else {
            panic!("final diagnostics input {expected_name} must be one named binding");
        };
        assert_eq!(binding.ident, expected_name);
        assert!(
            binding.by_ref.is_none() && binding.mutability.is_none() && binding.subpat.is_none(),
            "final diagnostics input {expected_name} has shadowable pattern syntax",
        );
        let syn::Type::Reference(reference) = input.ty.as_ref() else {
            panic!("final diagnostics input {expected_name} must be an immutable borrow");
        };
        assert!(reference.mutability.is_none());
        if let Some(expected_type) = expected_type {
            assert_eq!(
                ast_type_nominal_name(&reference.elem).as_deref(),
                Some(expected_type),
                "final diagnostics input {expected_name} has the wrong nominal type",
            );
        }
    }
    assert!(inputs.next().is_none());

    let selection = ast_local_named(&diagnostics.block.stmts[0], "selection_option");
    let selection = ast_method_named(ast_local_initializer(selection), "as_ref");
    assert_eq!(selection.args.len(), 0);
    assert!(ast_expr_is_field_path(
        &selection.receiver,
        &["session", "qkv_selection_evidence"],
    ));

    let syn::Expr::Match(channels) = ast_tail_expression(diagnostics.block) else {
        panic!("final diagnostics must tail-match the exact selection_option binding");
    };
    assert!(ast_expr_is_path(&channels.expr, "selection_option"));
    assert_eq!(channels.arms.len(), 2);
    assert_eq!(
        ast_direct_call_count(
            diagnostics.block,
            "BrowserVisionQkvFinalExecutionEvidence::from_verified_plan",
        ),
        1,
    );
    assert_eq!(
        ast_direct_call_count(
            diagnostics.block,
            "crate::serialize_vision_stack_qkv_final_diagnostics_json",
        ),
        1,
    );
    assert_eq!(
        ast_direct_call_count(
            diagnostics.block,
            "crate::serialize_vision_stack_legacy_diagnostics_json",
        ),
        1,
    );
    assert_eq!(ast_direct_call_count(diagnostics.block, "Ok"), 0);

    let mut optimized_seen = false;
    let mut legacy_seen = false;
    for arm in &channels.arms {
        match &arm.pat {
            syn::Pat::TupleStruct(pattern)
                if ast_path_name(&pattern.path) == "Some" && pattern.elems.len() == 1 =>
            {
                assert!(!optimized_seen, "duplicate optimized diagnostics arm");
                optimized_seen = true;
                let syn::Pat::Ident(selection_evidence) = pattern.elems.first().unwrap() else {
                    panic!("optimized diagnostics arm must bind selection_evidence");
                };
                assert_eq!(selection_evidence.ident, "selection_evidence");
                assert!(
                    selection_evidence.by_ref.is_none()
                        && selection_evidence.mutability.is_none()
                        && selection_evidence.subpat.is_none(),
                );
                let syn::Expr::Block(body) = arm.body.as_ref() else {
                    panic!("optimized diagnostics arm must contain its two causal bindings");
                };
                assert_eq!(
                    body.block.stmts.len(),
                    3,
                    "optimized diagnostics arm must bind execution, bind envelope, then serialize",
                );

                let execution = ast_local_named(&body.block.stmts[0], "qkv_execution");
                let execution_call = ast_call_named(
                    ast_without_try(ast_local_initializer(execution)),
                    "BrowserVisionQkvFinalExecutionEvidence::from_verified_plan",
                );
                assert_eq!(execution_call.args.len(), 2);
                let plan = ast_method_named(&execution_call.args[0], "as_ref");
                assert_eq!(plan.args.len(), 0);
                assert!(ast_expr_is_field_path(
                    &plan.receiver,
                    &["session", "qkv_execution_evidence_plan"],
                ));
                assert!(ast_expr_is_path(&execution_call.args[1], "canary_results"));

                let evidence = ast_local_named(&body.block.stmts[1], "evidence");
                let evidence_call = ast_method_named(
                    ast_local_initializer(evidence),
                    "final_diagnostics_evidence",
                );
                assert!(ast_expr_is_path(
                    &evidence_call.receiver,
                    "selection_evidence",
                ));
                assert_eq!(evidence_call.args.len(), 1);
                let execution_ref = ast_method_named(&evidence_call.args[0], "as_ref");
                assert_eq!(execution_ref.args.len(), 0);
                assert!(ast_expr_is_path(&execution_ref.receiver, "qkv_execution"));

                let serializer = ast_tail_direct_call(
                    &body.block,
                    "crate::serialize_vision_stack_qkv_final_diagnostics_json",
                );
                assert_eq!(serializer.args.len(), 2);
                assert!(ast_expr_is_path(&serializer.args[0], "legacy_diagnostics"));
                assert!(ast_expr_is_path(&serializer.args[1], "evidence"));
            }
            syn::Pat::Ident(pattern)
                if ast_ident_is(&pattern.ident, "None")
                    && pattern.by_ref.is_none()
                    && pattern.mutability.is_none()
                    && pattern.subpat.is_none() =>
            {
                assert!(!legacy_seen, "duplicate legacy diagnostics arm");
                legacy_seen = true;
                let expression = match arm.body.as_ref() {
                    syn::Expr::Block(body) => {
                        assert_eq!(body.block.stmts.len(), 1);
                        ast_tail_expression(&body.block)
                    }
                    expression => expression,
                };
                let serializer = ast_call_named(
                    ast_without_try(expression),
                    "crate::serialize_vision_stack_legacy_diagnostics_json",
                );
                assert_eq!(serializer.args.len(), 1);
                assert!(ast_expr_is_path(
                    serializer.args.first().unwrap(),
                    "legacy_diagnostics",
                ));
            }
            _ => panic!("diagnostics match has an open or wrong channel pattern"),
        }
    }
    assert!(optimized_seen && legacy_seen);
}

fn assert_legacy_json_causal_flow_source(source: &str) {
    let file = syn::parse_file(source).expect("legacy Web source must parse as an AST");

    let base = one_ast_function(&file, "begin_vision_encoder_stack_sharded_json");
    assert_eq!(base.block.stmts.len(), 1);
    assert_eq!(ast_direct_call_count(base.block, "Ok"), 0);
    assert_ast_begin_common_tail(
        ast_tail_expression(base.block),
        "VisionStackActivationStrategy::SeparateBuffers",
        None,
    );

    let strategy = one_ast_function(
        &file,
        "begin_vision_encoder_stack_sharded_with_activation_strategy_json",
    );
    assert_eq!(strategy.block.stmts.len(), 2);
    assert_eq!(ast_direct_call_count(strategy.block, "Ok"), 0);
    assert_ast_parse_activation_local(&strategy.block.stmts[0]);
    assert_ast_begin_common_tail(
        ast_tail_expression(strategy.block),
        "activation_strategy",
        None,
    );

    let hardened = one_ast_function(
        &file,
        "begin_vision_encoder_stack_sharded_with_activation_strategy_and_memory_hardening_json",
    );
    assert_eq!(hardened.block.stmts.len(), 4);
    assert_eq!(ast_direct_call_count(hardened.block, "Ok"), 0);
    assert_ast_parse_activation_local(&hardened.block.stmts[0]);
    assert_ast_parse_hardening_local(&hardened.block.stmts[1]);
    let validation = ast_statement_expression(&hardened.block.stmts[2])
        .expect("hardening wrapper must retain its validation branch");
    let syn::Expr::If(validation) = validation else {
        panic!("hardening wrapper validation must remain one exact if branch");
    };
    assert!(validation.else_branch.is_none());
    let syn::Expr::Unary(condition) = validation.cond.as_ref() else {
        panic!("hardening wrapper must negate its exact accepted-strategy match");
    };
    assert!(matches!(condition.op, syn::UnOp::Not(_)));
    let syn::Expr::Macro(accepted_strategies) = condition.expr.as_ref() else {
        panic!("hardening wrapper accepted-strategy condition must remain matches!");
    };
    assert_eq!(ast_path_name(&accepted_strategies.mac.path), "matches");
    let accepted_tokens = accepted_strategies
        .mac
        .tokens
        .to_string()
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    assert_eq!(
        accepted_tokens,
        "activation_strategy,VisionStackActivationStrategy::StaticArenaNoAlias|VisionStackActivationStrategy::StaticArenaAlias",
    );
    assert_eq!(ast_direct_call_count(&validation.then_branch, "Err"), 1);
    assert_eq!(ast_direct_call_count(&validation.then_branch, "Ok"), 0);
    assert_eq!(validation.then_branch.stmts.len(), 1);
    let rejection = ast_statement_expression(&validation.then_branch.stmts[0])
        .expect("hardening wrapper rejection must be a return expression");
    let syn::Expr::Return(rejection) = rejection else {
        panic!("hardening wrapper rejection must return immediately");
    };
    let rejected = ast_call_named(
        rejection
            .expr
            .as_deref()
            .expect("hardening wrapper rejection value"),
        "Err",
    );
    assert_eq!(rejected.args.len(), 1);
    let js_error = ast_call_named(rejected.args.first().unwrap(), "js_error");
    assert_eq!(js_error.args.len(), 1);
    assert!(
        matches!(js_error.args.first().unwrap(), syn::Expr::Lit(literal)
        if matches!(&literal.lit, syn::Lit::Str(message)
            if message.value() == "vision-stack memory hardening requires static_arena_no_alias or static_arena_alias"))
    );
    assert_ast_begin_common_tail(
        ast_tail_expression(hardened.block),
        "activation_strategy",
        Some("memory_hardening"),
    );

    let status = one_ast_function(&file, "vision_stack_status_json");
    assert_eq!(
        status.block.stmts.len(),
        2,
        "legacy status must be one builder followed by one serializer tail",
    );
    let (record, builder) = ast_local_direct_call_binding(
        status.block,
        "crate::build_vision_stack_legacy_status_record",
    );
    assert_eq!(builder.args.len(), 8);
    let phase = ast_method_named(&builder.args[0], "phase");
    assert_eq!(phase.args.len(), 0);
    assert!(ast_expr_is_field_path(
        &phase.receiver,
        &["session", "protocol"],
    ));
    let next_shard = ast_method_named(&builder.args[1], "next_shard_id");
    assert_eq!(next_shard.args.len(), 0);
    assert!(ast_expr_is_field_path(
        &next_shard.receiver,
        &["session", "protocol"],
    ));
    assert!(matches!(&builder.args[2], syn::Expr::Reference(reference)
        if ast_expr_is_field_path(&reference.expr, &["session", "plan"])));
    assert!(ast_expr_is_field_path(
        &builder.args[3],
        &["session", "activation_strategy"],
    ));
    let layout = ast_method_named(&builder.args[4], "as_ref");
    assert_eq!(layout.args.len(), 0);
    assert!(ast_expr_is_field_path(
        &layout.receiver,
        &["session", "activation_layout"],
    ));
    let hardening = ast_method_named(&builder.args[5], "as_ref");
    assert_eq!(hardening.args.len(), 0);
    assert!(ast_expr_is_field_path(
        &hardening.receiver,
        &["session", "memory_hardening"],
    ));
    assert!(ast_expr_is_field_path(
        &builder.args[6],
        &["session", "storage_alignment"],
    ));
    assert!(ast_expr_is_path(&builder.args[7], "include_plan"));
    assert_eq!(
        ast_direct_call_count(
            status.block,
            "crate::build_vision_stack_legacy_status_record",
        ),
        1,
    );
    assert_eq!(
        ast_direct_call_count(
            status.block,
            "crate::serialize_vision_stack_legacy_status_json",
        ),
        1,
    );
    assert_eq!(ast_direct_call_count(status.block, "Ok"), 0);
    let serializer = ast_tail_direct_call(
        status.block,
        "crate::serialize_vision_stack_legacy_status_json",
    );
    assert_eq!(serializer.args.len(), 1);
    assert!(ast_reference_to_path(
        serializer.args.first().unwrap(),
        &record.to_string(),
    ));

    assert_final_diagnostics_causal_flow_ast(source);
    let finish = one_ast_function(&file, "finish_vision_stack_sharded_once");
    let (legacy_diagnostics, diagnostics_builder) = ast_local_direct_call_binding(
        finish.block,
        "crate::build_vision_stack_legacy_diagnostics_record",
    );
    assert_eq!(diagnostics_builder.args.len(), 9);
    assert!(
        matches!(&diagnostics_builder.args[0], syn::Expr::Reference(reference)
        if ast_expr_is_field_path(&reference.expr, &["session", "plan"]))
    );
    assert!(ast_expr_is_field_path(
        &diagnostics_builder.args[1],
        &["session", "activation_strategy"],
    ));
    let diagnostics_layout = ast_method_named(&diagnostics_builder.args[2], "as_ref");
    assert_eq!(diagnostics_layout.args.len(), 0);
    assert!(ast_expr_is_field_path(
        &diagnostics_layout.receiver,
        &["session", "activation_layout"],
    ));
    let diagnostics_hardening = ast_method_named(&diagnostics_builder.args[3], "as_ref");
    assert_eq!(diagnostics_hardening.args.len(), 0);
    assert!(ast_expr_is_field_path(
        &diagnostics_hardening.receiver,
        &["session", "memory_hardening"],
    ));
    assert!(ast_expr_is_field_path(
        &diagnostics_builder.args[4],
        &["session", "storage_alignment"],
    ));
    assert!(
        matches!(&diagnostics_builder.args[5], syn::Expr::Reference(reference)
        if ast_expr_is_field_path(&reference.expr, &["gpu", "shader_blake3"]))
    );
    let queue_time = ast_method_named(&diagnostics_builder.args[6], "max");
    assert!(ast_expr_is_path(&queue_time.receiver, "queue_wall_time_ns"));
    assert_eq!(queue_time.args.len(), 1);
    assert!(
        matches!(queue_time.args.first().unwrap(), syn::Expr::Lit(literal)
        if matches!(&literal.lit, syn::Lit::Int(value) if value.base10_digits() == "1"))
    );
    assert!(ast_expr_is_path(
        &diagnostics_builder.args[7],
        "buffer_allocation_count",
    ));
    assert!(ast_expr_is_path(
        &diagnostics_builder.args[8],
        "weight_buffer_count",
    ));
    let (diagnostics_json, final_serializer) =
        ast_local_direct_call_binding(finish.block, "vision_stack_qkv_diagnostics_json");
    assert_eq!(final_serializer.args.len(), 3);
    assert!(ast_reference_to_path(
        final_serializer.args.first().unwrap(),
        &legacy_diagnostics.to_string(),
    ));
    assert_eq!(
        ast_direct_call_count(
            finish.block,
            "crate::build_vision_stack_legacy_diagnostics_record"
        ),
        1,
    );
    assert_eq!(
        ast_direct_call_count(finish.block, "vision_stack_qkv_diagnostics_json"),
        1,
    );
    assert_eq!(
        ast_direct_call_count(finish.block, "js_sys::Reflect::set"),
        2
    );
    assert_eq!(ast_direct_call_count(finish.block, "Ok"), 1);

    struct ReflectWrites<'a> {
        writes: Vec<&'a syn::ExprCall>,
    }
    impl<'ast> syn::visit::Visit<'ast> for ReflectWrites<'ast> {
        fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
            if ast_expr_path(&call.func)
                .is_some_and(|path| ast_path_name(path) == "js_sys::Reflect::set")
            {
                self.writes.push(call);
            }
            syn::visit::visit_expr_call(self, call);
        }
    }
    let mut writes = ReflectWrites { writes: Vec::new() };
    syn::visit::Visit::visit_block(&mut writes, finish.block);
    let diagnostics_writes = writes
        .writes
        .into_iter()
        .filter(|call| {
            call.args.get(1).is_some_and(|key| {
                let key = match key {
                    syn::Expr::Reference(reference) => reference.expr.as_ref(),
                    expression => expression,
                };
                let syn::Expr::Call(key) = key else {
                    return false;
                };
                ast_expr_path(&key.func)
                    .is_some_and(|path| ast_path_name(path) == "JsValue::from_str")
                    && key.args.first().is_some_and(|argument| {
                        matches!(argument, syn::Expr::Lit(literal)
                            if matches!(&literal.lit, syn::Lit::Str(value) if value.value() == "diagnostics_json"))
                    })
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(diagnostics_writes.len(), 1);
    let value = diagnostics_writes[0]
        .args
        .get(2)
        .expect("diagnostics Reflect::set value");
    let value = match value {
        syn::Expr::Reference(reference) => reference.expr.as_ref(),
        expression => expression,
    };
    let from_str = ast_call_named(value, "JsValue::from_str");
    assert_eq!(from_str.args.len(), 1);
    assert!(ast_reference_to_path(
        from_str.args.first().unwrap(),
        &diagnostics_json.to_string(),
    ));
}

#[test]
fn legacy_json_causal_flow_scanner_rejects_discarded_records_and_constant_returns() {
    const VALID: &str = r#"
impl WebRuntime {
    pub fn begin_vision_encoder_stack_sharded_json(
        &self,
        manifest_json: &str,
    ) -> Result<String, JsValue> {
        self.begin_vision_stack_sharded(
            manifest_json,
            VisionStackActivationStrategy::SeparateBuffers,
            None,
        )
        .map_err(js_error)
    }

    pub fn begin_vision_encoder_stack_sharded_with_activation_strategy_json(
        &self,
        manifest_json: &str,
        activation_strategy: &str,
    ) -> Result<String, JsValue> {
        let activation_strategy =
            parse_vision_stack_activation_strategy(activation_strategy).map_err(js_error)?;
        self.begin_vision_stack_sharded(manifest_json, activation_strategy, None)
            .map_err(js_error)
    }

    pub fn begin_vision_encoder_stack_sharded_with_activation_strategy_and_memory_hardening_json(
        &self,
        manifest_json: &str,
        activation_strategy: &str,
        memory_hardening: &str,
    ) -> Result<String, JsValue> {
        let activation_strategy =
            parse_vision_stack_activation_strategy(activation_strategy).map_err(js_error)?;
        let memory_hardening = memory_hardening
            .parse::<VisionStackMemoryHardening>()
            .map_err(js_error)?;
        if !matches!(
            activation_strategy,
            VisionStackActivationStrategy::StaticArenaNoAlias
                | VisionStackActivationStrategy::StaticArenaAlias
        ) {
            return Err(js_error(
                "vision-stack memory hardening requires static_arena_no_alias or static_arena_alias",
            ));
        }
        self.begin_vision_stack_sharded(
            manifest_json,
            activation_strategy,
            Some(memory_hardening),
        )
        .map_err(js_error)
    }
}

fn vision_stack_status_json(
    session: &BrowserVisionStackSession,
    include_plan: bool,
) -> Result<String, String> {
    let legacy_status = crate::build_vision_stack_legacy_status_record(
        session.protocol.phase(),
        session.protocol.next_shard_id(),
        &session.plan,
        session.activation_strategy,
        session.activation_layout.as_ref(),
        session.memory_hardening.as_ref(),
        session.storage_alignment,
        include_plan,
    )?;
    crate::serialize_vision_stack_legacy_status_json(&legacy_status)
}

fn vision_stack_qkv_diagnostics_json(
    legacy_diagnostics: &VisionStackLegacyDiagnosticsRecord,
    session: &BrowserVisionStackSession,
    canary_results: &CanaryResults,
) -> Result<String, String> {
    let selection_option = session.qkv_selection_evidence.as_ref();
    match selection_option {
        Some(selection_evidence) => {
            let qkv_execution = BrowserVisionQkvFinalExecutionEvidence::from_verified_plan(
                session.qkv_execution_evidence_plan.as_ref(),
                canary_results,
            )?;
            let evidence =
                selection_evidence.final_diagnostics_evidence(qkv_execution.as_ref());
            crate::serialize_vision_stack_qkv_final_diagnostics_json(legacy_diagnostics, evidence)
        }
        None => crate::serialize_vision_stack_legacy_diagnostics_json(legacy_diagnostics),
    }
}

fn finish_vision_stack_sharded_once(
    session: &BrowserVisionStackSession,
) -> Result<JsValue, String> {
    let legacy_diagnostics = crate::build_vision_stack_legacy_diagnostics_record(
        &session.plan,
        session.activation_strategy,
        session.activation_layout.as_ref(),
        session.memory_hardening.as_ref(),
        session.storage_alignment,
        &gpu.shader_blake3,
        queue_wall_time_ns.max(1),
        buffer_allocation_count,
        weight_buffer_count,
    )?;
    let diagnostics_json = vision_stack_qkv_diagnostics_json(
        &legacy_diagnostics,
        session,
        canary_results,
    )?;
    let result = js_sys::Object::new();
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("checkpoint_bytes"),
        &checkpoint_bytes,
    )?;
    js_sys::Reflect::set(
        &result,
        &JsValue::from_str("diagnostics_json"),
        &JsValue::from_str(&diagnostics_json),
    )?;
    Ok(result.into())
}
"#;
    assert_legacy_json_causal_flow_source(VALID);

    for (label, hostile) in [
        (
            "base export returns a constant",
            VALID.replacen(
                "self.begin_vision_stack_sharded(\n            manifest_json,\n            VisionStackActivationStrategy::SeparateBuffers,\n            None,\n        )\n        .map_err(js_error)",
                "Ok(constant_json())",
                1,
            ),
        ),
        (
            "strategy export passes a constant strategy",
            VALID.replacen(
                "self.begin_vision_stack_sharded(manifest_json, activation_strategy, None)",
                "self.begin_vision_stack_sharded(manifest_json, VisionStackActivationStrategy::SeparateBuffers, None)",
                1,
            ),
        ),
        (
            "hardening export has an alternate successful return",
            VALID.replacen(
                "        self.begin_vision_stack_sharded(\n            manifest_json,\n            activation_strategy,\n            Some(memory_hardening),\n        )",
                "        if hostile { return Ok(constant_json()); }\n        self.begin_vision_stack_sharded(\n            manifest_json,\n            activation_strategy,\n            Some(memory_hardening),\n        )",
                1,
            ),
        ),
        (
            "hardening export narrows the accepted legacy strategies",
            VALID.replace(
                "VisionStackActivationStrategy::StaticArenaNoAlias\n                | VisionStackActivationStrategy::StaticArenaAlias",
                "VisionStackActivationStrategy::StaticArenaNoAlias",
            ),
        ),
        (
            "status discards the production serializer",
            VALID.replace(
                "    crate::serialize_vision_stack_legacy_status_json(&legacy_status)\n}",
                "    let _discarded = crate::serialize_vision_stack_legacy_status_json(&legacy_status);\n    Ok(constant_json())\n}",
            ),
        ),
        (
            "status serializes a cached record",
            VALID.replace(
                "crate::serialize_vision_stack_legacy_status_json(&legacy_status)",
                "crate::serialize_vision_stack_legacy_status_json(&cached_status)",
            ),
        ),
        (
            "status calls a same-named Web-local builder",
            VALID.replace(
                "crate::build_vision_stack_legacy_status_record(",
                "web_local::build_vision_stack_legacy_status_record(",
            ),
        ),
        (
            "status calls an unqualified same-named builder",
            VALID.replace(
                "crate::build_vision_stack_legacy_status_record(",
                "build_vision_stack_legacy_status_record(",
            ),
        ),
        (
            "status calls a same-named Web-local serializer",
            VALID.replace(
                "crate::serialize_vision_stack_legacy_status_json(",
                "web_local::serialize_vision_stack_legacy_status_json(",
            ),
        ),
        (
            "status calls an unqualified same-named serializer",
            VALID.replace(
                "crate::serialize_vision_stack_legacy_status_json(",
                "serialize_vision_stack_legacy_status_json(",
            ),
        ),
        (
            "status builder receives a cached plan",
            VALID.replace("        &session.plan,", "        &cached_plan,"),
        ),
        (
            "legacy final branch serializes another record",
            VALID.replace(
                "crate::serialize_vision_stack_legacy_diagnostics_json(legacy_diagnostics)",
                "crate::serialize_vision_stack_legacy_diagnostics_json(other_diagnostics)",
            ),
        ),
        (
            "optimized final branch serializes another record",
            VALID.replace(
                "crate::serialize_vision_stack_qkv_final_diagnostics_json(legacy_diagnostics, evidence)",
                "crate::serialize_vision_stack_qkv_final_diagnostics_json(other_diagnostics, evidence)",
            ),
        ),
        (
            "legacy final arm shadows the authenticated parameter",
            VALID.replace(
                "None => crate::serialize_vision_stack_legacy_diagnostics_json(legacy_diagnostics),",
                "None => {\n            let legacy_diagnostics = build_wrong_record();\n            crate::serialize_vision_stack_legacy_diagnostics_json(legacy_diagnostics)\n        },",
            ),
        ),
        (
            "optimized final arm shadows the authenticated parameter",
            VALID.replace(
                "Some(selection_evidence) => {\n            let qkv_execution",
                "Some(selection_evidence) => {\n            let legacy_diagnostics = build_wrong_record();\n            let qkv_execution",
            ),
        ),
        (
            "optimized evidence pattern shadows the authenticated parameter",
            VALID.replace(
                "Some(selection_evidence) => {",
                "Some(legacy_diagnostics) => {",
            )
                .replace(
                    "selection_evidence.final_diagnostics_evidence(qkv_execution.as_ref())",
                    "legacy_diagnostics.final_diagnostics_evidence(qkv_execution.as_ref())",
                ),
        ),
        (
            "final helper shadows the authenticated parameter before its match",
            VALID.replace(
                "    match selection_option {",
                "    let legacy_diagnostics = build_wrong_record();\n    match selection_option {",
            ),
        ),
        (
            "final arm delegates through an unreviewed replacement helper",
            VALID.replace(
                "crate::serialize_vision_stack_legacy_diagnostics_json(legacy_diagnostics)",
                "serialize_selected_diagnostics(legacy_diagnostics)",
            ),
        ),
        (
            "legacy and optimized final branches are swapped",
            VALID.replace(
                "crate::serialize_vision_stack_qkv_final_diagnostics_json(legacy_diagnostics, evidence)",
                "crate::serialize_vision_stack_legacy_diagnostics_json(legacy_diagnostics)",
            )
            .replace(
                "None => crate::serialize_vision_stack_legacy_diagnostics_json(legacy_diagnostics),",
                "None => crate::serialize_vision_stack_qkv_final_diagnostics_json(legacy_diagnostics, evidence),",
            ),
        ),
        (
            "final diagnostics match gains an open fallback arm",
            VALID.replace(
                "        None => crate::serialize_vision_stack_legacy_diagnostics_json(legacy_diagnostics),",
                "        None => crate::serialize_vision_stack_legacy_diagnostics_json(legacy_diagnostics),\n        _ => Ok(constant_json()),",
            ),
        ),
        (
            "final diagnostics call a same-named Web-local serializer",
            VALID.replace(
                "crate::serialize_vision_stack_legacy_diagnostics_json(",
                "web_local::serialize_vision_stack_legacy_diagnostics_json(",
            ),
        ),
        (
            "final diagnostics call an unqualified same-named serializer",
            VALID.replace(
                "crate::serialize_vision_stack_legacy_diagnostics_json(",
                "serialize_vision_stack_legacy_diagnostics_json(",
            ),
        ),
        (
            "finish passes another diagnostics record",
            VALID.replace(
                "        &legacy_diagnostics,\n        session,",
                "        &cached_diagnostics,\n        session,",
            ),
        ),
        (
            "final diagnostics builder receives a cached plan",
            VALID.replace(
                "let legacy_diagnostics = crate::build_vision_stack_legacy_diagnostics_record(\n        &session.plan,",
                "let legacy_diagnostics = crate::build_vision_stack_legacy_diagnostics_record(\n        &cached_plan,",
            ),
        ),
        (
            "finish calls a same-named Web-local diagnostics builder",
            VALID.replace(
                "crate::build_vision_stack_legacy_diagnostics_record(",
                "web_local::build_vision_stack_legacy_diagnostics_record(",
            ),
        ),
        (
            "finish calls an unqualified same-named diagnostics builder",
            VALID.replace(
                "crate::build_vision_stack_legacy_diagnostics_record(",
                "build_vision_stack_legacy_diagnostics_record(",
            ),
        ),
        (
            "finish writes cached diagnostics JSON",
            VALID.replace(
                "&JsValue::from_str(&diagnostics_json),",
                "&JsValue::from_str(&cached_diagnostics_json),",
            ),
        ),
    ] {
        assert_ne!(hostile, VALID, "legacy causal-flow mutant {label} was a no-op");
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                assert_legacy_json_causal_flow_source(&hostile);
            }))
            .is_err(),
            "legacy causal-flow scanner accepted {label}",
        );
    }
}

#[test]
fn legacy_wasm_exports_and_closed_status_records_remain_frozen() {
    assert_legacy_json_causal_flow_source(WEB_SOURCE);
    for method in [
        "pub fn begin_vision_encoder_stack_sharded_json(",
        "pub fn begin_vision_encoder_stack_sharded_with_activation_strategy_json(",
        "pub fn begin_vision_encoder_stack_sharded_with_activation_strategy_and_memory_hardening_json(",
        "pub fn preflight_vision_encoder_stack_shard_json(",
        "pub async fn start_vision_encoder_stack_sharded_json(",
        "pub async fn run_vision_encoder_stack_sharded_layer_json(",
        "pub async fn finish_vision_encoder_stack_sharded(",
        "pub fn abort_vision_encoder_stack_sharded(&self)",
    ] {
        assert_eq!(
            occurrences(WEB_SOURCE, method),
            1,
            "legacy export drifted: {method}"
        );
    }

    for record in [
        "pub struct VisionStackLegacyStatusRecord",
        "pub struct VisionStackLegacyDiagnosticsRecord",
    ] {
        let declaration = braced_item(LIB_SOURCE, record);
        assert!(
            !declaration.to_ascii_lowercase().contains("qkv"),
            "host-callable legacy record {record} gained a QKV field",
        );
        for forbidden in ["pub phase", "pub plan", "serialize_with", "rename = \"qkv"] {
            assert!(
                !declaration.contains(forbidden),
                "legacy record is caller-forgeable/schema-customizable via {forbidden}",
            );
        }
    }
    for builder in [
        "build_vision_stack_legacy_status_record",
        "build_vision_stack_legacy_diagnostics_record",
    ] {
        let graph = reachable_functions(LIB_SOURCE, builder);
        let source = graph
            .iter()
            .map(|function| function.body)
            .collect::<String>();
        assert!(
            !source.to_ascii_lowercase().contains("qkv"),
            "legacy builder graph {builder} injects QKV data",
        );
        for forbidden in ["serde_json::Value", "HashMap", "json!", "serialize_with"] {
            assert!(
                !source.contains(forbidden),
                "legacy builder graph {builder} can author an open record via {forbidden}",
            );
        }
    }
    for serializer in [
        "serialize_vision_stack_legacy_status_json",
        "serialize_vision_stack_legacy_diagnostics_json",
    ] {
        let functions = source_functions(LIB_SOURCE)
            .into_iter()
            .filter(|function| function.name == serializer)
            .collect::<Vec<_>>();
        assert_eq!(functions.len(), 1);
        let function = functions[0];
        assert_eq!(occurrences(function.body, "to_json_string("), 1);
        let arguments = balanced_call_arguments(function.body, "to_json_string(");
        assert_eq!(arguments.len(), 1);
        assert_eq!(compact(arguments[0]), "record");
        let expression =
            &function.body[function.body.find('{').unwrap() + 1..function.body.rfind('}').unwrap()];
        assert_eq!(
            compact(expression),
            "to_json_string(record)",
            "{serializer} conditionally replaced its exact production record",
        );
    }

    let status_serializer = braced_item(WEB_SOURCE, "fn vision_stack_status_json(");
    assert_eq!(
        occurrences(
            status_serializer,
            "build_vision_stack_legacy_status_record(",
        ),
        1,
    );
    assert_eq!(
        occurrences(
            status_serializer,
            "serialize_vision_stack_legacy_status_json(",
        ),
        1,
    );
    assert_eq!(occurrences(status_serializer, "to_json_string("), 0);
    assert!(!status_serializer.to_ascii_lowercase().contains("qkv"));

    let finish = braced_item(WEB_SOURCE, "fn finish_vision_stack_sharded_once(");
    assert_eq!(
        occurrences(finish, "build_vision_stack_legacy_diagnostics_record(",),
        1,
        "legacy and optimized finish must share one exact diagnostics base record",
    );
    assert_eq!(
        occurrences(finish, "vision_stack_qkv_diagnostics_json(",),
        1,
    );
    assert_eq!(occurrences(finish, "to_json_string("), 0);
    let final_serializer = braced_item(WEB_SOURCE, "fn vision_stack_qkv_diagnostics_json(");
    assert_eq!(
        occurrences(
            final_serializer,
            "serialize_vision_stack_legacy_diagnostics_json(",
        ),
        1,
    );
    assert_eq!(
        occurrences(
            final_serializer,
            "serialize_vision_stack_qkv_final_diagnostics_json(",
        ),
        1,
    );
    assert_eq!(occurrences(final_serializer, "to_json_string("), 0);
}

#[test]
fn production_legacy_serializers_are_recursively_exact_and_qkv_free_for_all_strategies() {
    const SEPARATE_GOLDEN: &str = include_str!("goldens/m7c2b_legacy_separate.jsonl");
    const STATIC_NO_ALIAS_GOLDEN: &str = include_str!("goldens/m7c2b_legacy_static_no_alias.jsonl");
    const STATIC_NO_ALIAS_HARDENED_GOLDEN: &str =
        include_str!("goldens/m7c2b_legacy_static_no_alias_hardened.jsonl");
    const STATIC_ALIAS_GOLDEN: &str = include_str!("goldens/m7c2b_legacy_static_alias.jsonl");
    const STATIC_ALIAS_HARDENED_GOLDEN: &str =
        include_str!("goldens/m7c2b_legacy_static_alias_hardened.jsonl");

    let manifest = parse_vision_stack_shard_manifest(synthetic_manifest_at_depth(1).as_slice())
        .expect("canonical depth-one manifest");
    let plan: VisionStackShardPlan = manifest.plan().expect("legacy plan");
    let static_layout = VisionStackActivationLayout {
        scratch_allocations: vec![VisionStackScratchAllocation {
            stage: VisionEncoderLayerStage::Norm1,
            offset: 0,
            size: plan.hidden_bytes,
            alignment: 32,
            first_write: 0,
            last_use: 3,
        }],
        scratch_arena_bytes: 64,
        main_buffers_bytes: plan.hidden_bytes * 2,
        total_activation_bytes: 64 + plan.hidden_bytes * 2,
        physical_buffer_count: 3,
    };
    let static_peak_gpu_data_bytes = static_layout.total_activation_bytes
        + plan.readback_bytes
        + plan
            .hidden_bytes
            .max(plan.layer_weight_bytes)
            .max(plan.post_norm_bytes);
    let memory_hardening = VisionStackMemoryHardeningPlan::new(
        VisionStackMemoryHardening::PoisonCanary,
        32,
        static_layout.scratch_arena_bytes,
        plan.readback_bytes,
        static_peak_gpu_data_bytes,
    )
    .expect("valid host memory-hardening evidence");
    let shader_blake3 = BTreeMap::from([(KernelId::LayerNormF32, "ab".repeat(32))]);

    for strategy in [
        VisionStackActivationStrategy::SeparateBuffers,
        VisionStackActivationStrategy::StaticArenaNoAlias,
        VisionStackActivationStrategy::StaticArenaAlias,
    ] {
        let (layout, hardenings) = match strategy {
            VisionStackActivationStrategy::SeparateBuffers => (None, vec![None]),
            VisionStackActivationStrategy::StaticArenaNoAlias
            | VisionStackActivationStrategy::StaticArenaAlias => {
                (Some(&static_layout), vec![None, Some(&memory_hardening)])
            }
        };
        for hardening in hardenings {
            let status_record = build_vision_stack_legacy_status_record(
                VisionStackShardProtocolPhase::Preflight,
                manifest.shards.first().map(|shard| shard.id.as_str()),
                &plan,
                strategy,
                layout,
                hardening,
                32,
                true,
            )
            .expect("production legacy status builder");
            let diagnostics_record = build_vision_stack_legacy_diagnostics_record(
                &plan,
                strategy,
                layout,
                hardening,
                32,
                &shader_blake3,
                1,
                17,
                18,
            )
            .expect("production legacy diagnostics builder");
            let status_json = serialize_vision_stack_legacy_status_json(&status_record)
                .expect("production legacy status serializer");
            let diagnostics_json =
                serialize_vision_stack_legacy_diagnostics_json(&diagnostics_record)
                    .expect("production legacy diagnostics serializer");
            let expected_golden = match (strategy, hardening.is_some()) {
                (VisionStackActivationStrategy::SeparateBuffers, false) => SEPARATE_GOLDEN,
                (VisionStackActivationStrategy::StaticArenaNoAlias, false) => {
                    STATIC_NO_ALIAS_GOLDEN
                }
                (VisionStackActivationStrategy::StaticArenaNoAlias, true) => {
                    STATIC_NO_ALIAS_HARDENED_GOLDEN
                }
                (VisionStackActivationStrategy::StaticArenaAlias, false) => STATIC_ALIAS_GOLDEN,
                (VisionStackActivationStrategy::StaticArenaAlias, true) => {
                    STATIC_ALIAS_HARDENED_GOLDEN
                }
                (VisionStackActivationStrategy::SeparateBuffers, true) => {
                    panic!("separate buffers cannot be hardened")
                }
            };
            let actual_bytes = format!("{status_json}\n{diagnostics_json}\n");
            assert_eq!(
                actual_bytes.as_bytes(),
                expected_golden.as_bytes(),
                "{strategy:?} hardening={} changed legacy JSON bytes",
                hardening.is_some(),
            );

            let status: serde_json::Value =
                serde_json::from_str(&status_json).expect("legacy status JSON");
            let diagnostics: serde_json::Value =
                serde_json::from_str(&diagnostics_json).expect("legacy diagnostics JSON");

            assert_json_recursively_qkv_free(&status, &format!("{strategy:?}.status"));
            assert_json_recursively_qkv_free(&diagnostics, &format!("{strategy:?}.diagnostics"));
            let static_strategy = strategy != VisionStackActivationStrategy::SeparateBuffers;
            let expected_status_keys = [
                "phase",
                "next_shard_id",
                "plan",
                "capabilities",
                "static_layout",
                "memory_hardening_plan",
            ]
            .into_iter()
            .filter(|key| {
                !matches!(
                    (*key, static_strategy, hardening.is_some()),
                    ("capabilities" | "static_layout", false, _)
                        | ("memory_hardening_plan", _, false)
                )
            })
            .collect::<BTreeSet<_>>();
            assert_eq!(
                status
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>(),
                expected_status_keys,
                "{strategy:?} legacy status closed schema drifted",
            );
            assert_eq!(status["phase"], "preflight");
            assert_eq!(status["next_shard_id"], manifest.shards[0].id);

            let mut expected_plan_keys = BTreeSet::from([
                "layer_count",
                "shard_count",
                "input_bytes",
                "hidden_bytes",
                "intermediate_bytes",
                "layer_weight_bytes",
                "post_norm_bytes",
                "transport_bytes",
                "activation_buffer_count",
                "activation_arena_bytes",
                "readback_bytes",
                "peak_gpu_data_bytes",
                "submission_count",
                "compute_pass_count",
                "dispatch_count",
            ]);
            if static_strategy {
                expected_plan_keys.extend([
                    "scratch_arena_bytes",
                    "main_buffers_bytes",
                    "activation_strategy",
                    "min_storage_buffer_offset_alignment",
                ]);
            }
            assert_eq!(
                status["plan"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>(),
                expected_plan_keys,
            );
            let diagnostic_keys = diagnostics
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            let mut expected_diagnostic_keys = expected_plan_keys;
            expected_diagnostic_keys.extend([
                "checked_error_scopes",
                "captured_errors",
                "queue_wall_time_ns",
                "shader_blake3",
                "command_buffer_count",
                "buffer_allocation_count",
                "weight_buffer_count",
                "readback_buffer_count",
                "map_count",
            ]);
            if hardening.is_some() {
                expected_diagnostic_keys.insert("memory_hardening");
            }
            assert_eq!(diagnostic_keys, expected_diagnostic_keys);
            assert_eq!(diagnostics["command_buffer_count"], plan.submission_count);
            assert_eq!(diagnostics["buffer_allocation_count"], 17);
            assert_eq!(diagnostics["weight_buffer_count"], 18);
            assert_eq!(diagnostics["readback_buffer_count"], 1);
            assert_eq!(diagnostics["map_count"], 1);
            if let Some(layout) = layout {
                assert_eq!(
                    status["static_layout"],
                    serde_json::json!({
                        "scratch_allocations": [{
                            "stage": "norm1",
                            "offset": 0,
                            "size": plan.hidden_bytes,
                            "alignment": 32,
                            "first_write": 0,
                            "last_use": 3,
                        }],
                        "scratch_arena_bytes": layout.scratch_arena_bytes,
                        "main_buffers_bytes": layout.main_buffers_bytes,
                        "total_activation_bytes": layout.total_activation_bytes,
                        "physical_buffer_count": layout.physical_buffer_count,
                    }),
                );
            }
        }
    }
}

#[test]
fn optimized_wasm_surface_is_additive_opaque_and_wasm_only() {
    assert_selection_evidence_authority_source(LIB_SOURCE, WEB_SOURCE);
    let selection = braced_item(WEB_SOURCE, "pub struct WebVisionQkvStackSelection");
    assert!(
        compact(selection).contains("handoff:VisionQkvCompilerHandoff")
            && compact(selection).contains("evidence:VisionQkvSelectionEvidencePropagation"),
        "opaque wasm handle must own the exact handoff and shared evidence propagation",
    );
    assert!(
        !selection.contains("PreparedVisionQkvStackExecution")
            && !selection.contains("VisionQkvPhysicalExecutionSpec"),
        "opaque compile handle must not own a begin-time physical execution view",
    );
    assert!(
        selection.lines().skip(1).all(|line| {
            let trimmed = line.trim_start();
            !trimmed.starts_with("pub ")
        }),
        "opaque selection authority fields must remain private"
    );

    for method in [
        "pub fn compile_vision_encoder_stack_qkv_selection(",
        "pub fn evidence_json(",
        "pub fn vision_encoder_stack_qkv_shader_sources_json(",
        "pub fn begin_vision_encoder_stack_sharded_with_activation_strategy_and_qkv_selection_json(",
        "pub fn begin_vision_encoder_stack_sharded_with_activation_strategy_and_memory_hardening_and_qkv_selection_json(",
    ] {
        assert_eq!(
            occurrences(WEB_SOURCE, method),
            1,
            "optimized export missing or duplicated: {method}"
        );
    }
    let handle_impls = all_braced_items(WEB_SOURCE, "impl WebVisionQkvStackSelection {");
    let handle_methods = handle_impls
        .iter()
        .flat_map(|implementation| public_inherent_function_names(implementation))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        handle_methods,
        BTreeSet::from(["evidence_json"]),
        "opaque handle must expose only read-only evidence; source reporting belongs to WebRuntime",
    );
    let runtime_methods = all_braced_items(WEB_SOURCE, "impl WebRuntime {")
        .iter()
        .flat_map(|implementation| public_inherent_function_names(implementation))
        .collect::<BTreeSet<_>>();
    for method in [
        "compile_vision_encoder_stack_qkv_selection",
        "vision_encoder_stack_qkv_shader_sources_json",
        "begin_vision_encoder_stack_sharded_with_activation_strategy_and_qkv_selection_json",
        "begin_vision_encoder_stack_sharded_with_activation_strategy_and_memory_hardening_and_qkv_selection_json",
    ] {
        assert!(
            runtime_methods.contains(method),
            "additive wasm method {method} must be implemented by WebRuntime, not the opaque handle",
        );
    }
    assert!(
        LIB_SOURCE.contains("#[cfg(target_arch = \"wasm32\")]\npub use web::{WebRuntime, WebVisionQkvStackSelection};"),
        "opaque class must be exported only behind the wasm32 module gate",
    );
    assert!(!LIB_SOURCE.contains("pub struct WebVisionQkvStackSelection"));
}

#[test]
fn shared_prepared_and_readback_authority_is_opaque_and_accessor_only() {
    let all_authority_sources =
        format!("{PASSES_SOURCE}\n{CORE_SOURCE}\n{LIB_SOURCE}\n{WEB_SOURCE}\n{NATIVE_SOURCE}",);
    assert_complete_opaque_authority_surface(
        &all_authority_sources,
        "PreparedVisionQkvStackExecution",
        &["layer_count", "layers", "workspace"],
        "prepare_vision_qkv_stack_execution",
        &["prepared_execution"],
    );
    assert_complete_opaque_authority_surface(
        &all_authority_sources,
        "VisionQkvPhysicalExecutionSpec",
        &["prepared_execution", "readback_layout"],
        "bind_vision_qkv_physical_execution",
        &["prepare_vision_qkv_stack_handoff_execution"],
    );
    assert_complete_opaque_authority_surface(
        &all_authority_sources,
        "VisionQkvWebPhysicalCommandPlan",
        &[
            "commands",
            "fused_dispatch_workgroups",
            "fused_uniform_words",
        ],
        "plan_vision_qkv_web_physical_commands",
        &[],
    );
    assert_complete_opaque_authority_surface(
        &all_authority_sources,
        "VisionQkvReadbackLayout",
        &[
            "semantic_offset",
            "semantic_readback_bytes",
            "scratch_canary_offset",
            "scratch_canary_readback_bytes",
            "qkv_canary_offset",
            "qkv_canary_readback_bytes",
            "total_readback_bytes",
            "workspace_allocation_bytes",
            "readback_f32_elements",
            "workspace_u32_words",
        ],
        "plan_vision_qkv_readback_layout",
        &["readback_layout"],
    );
    assert_complete_opaque_authority_surface(
        &all_authority_sources,
        "VisionQkvCompilerHandoff",
        &[
            "selection",
            "canonical_manifest_blake3_hex",
            "semantic_graph_blake3_hex",
            "manifest_geometry",
            "layer_count",
            "target_limits",
            "tensor_catalog_len",
        ],
        "compile_vision_qkv_stack_handoff",
        &[],
    );
    assert_complete_opaque_authority_surface(
        &all_authority_sources,
        "VisionQkvCompilerManifestGeometry",
        &[
            "tokens",
            "hidden_size",
            "attention_heads",
            "head_dim",
            "intermediate_size",
            "layer_count",
        ],
        "compile_vision_qkv_stack_handoff",
        &["manifest_geometry"],
    );
    assert_complete_opaque_authority_surface(
        &all_authority_sources,
        "WebVisionQkvStackSelection",
        &["evidence_json"],
        "compile_vision_encoder_stack_qkv_selection",
        &[],
    );
}

#[test]
fn opaque_authority_scanner_rejects_alias_trait_const_and_return_bypasses() {
    const VALID: &str = r#"
pub struct SealedAuthority {
    value: u8,
}

impl SealedAuthority {
    pub fn value(&self) -> u8 {
        self.value
    }
}

pub fn make_sealed_authority() -> SealedAuthority {
    SealedAuthority { value: 1 }
}
"#;

    assert_complete_opaque_authority_surface(
        VALID,
        "SealedAuthority",
        &["value"],
        "make_sealed_authority",
        &[],
    );

    for (label, hostile, expected_panic) in [
        (
            "unsafe trait implementation returning Self",
            r#"
unsafe impl ExternalForge for SealedAuthority {
    fn forge() -> Self {
        Self { value: 2 }
    }
}
"#,
            "trait method forge",
        ),
        (
            "inherent implementation through a public type alias",
            r#"
pub type SealedAlias = SealedAuthority;

impl SealedAlias {
    pub fn new() -> Self {
        Self { value: 2 }
    }
}
"#,
            "public inherent API",
        ),
        (
            "external trait associated authority const",
            r#"
impl ExternalForge for OtherAuthority {
    const INSTANCE: SealedAuthority = SealedAuthority { value: 2 };
}
"#,
            "trait-associated const/static item",
        ),
        (
            "public custom trait authority return",
            r#"
pub trait ExternalForge {
    fn forge() -> SealedAuthority;
}
"#,
            "custom public trait method or associated item",
        ),
        (
            "free public authority const",
            r#"
pub const INSTANCE: SealedAuthority = SealedAuthority { value: 2 };
"#,
            "public const/static item",
        ),
        (
            "public return through a type alias",
            r#"
pub type SealedAlias = SealedAuthority;

pub fn leak_sealed_authority() -> SealedAlias {
    make_sealed_authority()
}
"#,
            "leak_sealed_authority is an unsanctioned",
        ),
    ] {
        let source = format!("{VALID}\n{hostile}");
        let panic = catch_unwind(AssertUnwindSafe(|| {
            assert_complete_opaque_authority_surface(
                &source,
                "SealedAuthority",
                &["value"],
                "make_sealed_authority",
                &[],
            );
        }))
        .unwrap_err();
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("non-string opacity scanner panic");
        assert!(
            message.contains(expected_panic),
            "{label}: wrong opacity rejection: {message}",
        );
    }
}

#[test]
fn live_source_sanitizer_preserves_parseability_offsets_and_masks_literal_matrix() {
    let source = r#####"
fn lifetime_probe<'a, 'b, F>(left: &'a str, transform: F) -> &'a str
where
    F: for<'c> Fn(&'c str) -> &'c str,
{
    'retry: loop {
        if left.is_empty() {
            break 'retry;
        }
        break 'retry;
    }
    let normal = "normal_decoy \"escaped_quote_decoy\" { nested }";
    let multiline = "multiline_decoy \
continuation_decoy";
    let bytes = b"byte_decoy \"quoted_byte_decoy\" { nested }";
    let leading_newline = "
leading_normal_newline_decoy";
    let byte_multiline = b"
leading_byte_newline_decoy";
    let raw = r###"raw_decoy \"# nested \"## { nested }
raw_multiline_decoy"###;
    let raw_leading_newline = r#"
leading_raw_newline_decoy"#;
    let raw_bytes = br##"raw_byte_decoy \"# { nested }
raw_byte_multiline_decoy"##;
    let raw_byte_leading_newline = br#"
leading_raw_byte_newline_decoy"#;
    let characters = ('\'', '\\', '\u{1F980}', '🦀', b'\xFF', b'{', '}');
    if false {
        fake_effect();
        let hidden = "dead_string_decoy { nested }";
    }
    live_effect();
    let _ = (
        transform,
        normal,
        multiline,
        bytes,
        leading_newline,
        byte_multiline,
        raw,
        raw_leading_newline,
        raw_bytes,
        raw_byte_leading_newline,
        characters,
    );
    left
}
"#####;

    let live = live_rust_source(source);
    assert_eq!(
        live.len(),
        source.len(),
        "sanitizing must preserve every byte offset"
    );
    let line_breaks = |value: &str| {
        value
            .bytes()
            .enumerate()
            .filter_map(|(index, byte)| matches!(byte, b'\n' | b'\r').then_some((index, byte)))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        line_breaks(&live),
        line_breaks(source),
        "sanitizing must preserve every CR/LF byte and its offset"
    );
    syn::parse_file(&live).expect("the complete sanitized Rust matrix must remain parseable");

    for live_syntax in ["<'a, 'b, F>", "for<'c>", "'retry:", "break 'retry;"] {
        assert!(
            live.contains(live_syntax),
            "sanitizer corrupted lifetime or label syntax: {live_syntax}"
        );
    }
    for decoy in [
        "normal_decoy",
        "escaped_quote_decoy",
        "multiline_decoy",
        "continuation_decoy",
        "byte_decoy",
        "quoted_byte_decoy",
        "leading_normal_newline_decoy",
        "leading_byte_newline_decoy",
        "raw_decoy",
        "raw_multiline_decoy",
        "leading_raw_newline_decoy",
        "raw_byte_decoy",
        "raw_byte_multiline_decoy",
        "leading_raw_byte_newline_decoy",
        "dead_string_decoy",
        "fake_effect(",
        "1F980",
        "🦀",
        "\\xFF",
        "b'{'",
        "'}'",
    ] {
        assert!(
            !live.contains(decoy),
            "sanitizer retained a literal/dead-code decoy: {decoy}"
        );
    }
    assert!(live.contains("live_effect("));
}

#[test]
fn live_source_sanitizer_and_typed_physical_scanner_reject_every_decoy_form() {
    const VALID: &str = r#"
fn apply_vision_qkv_web_create_buffer_command(
    &self,
    command: &VisionQkvWebPhysicalCommand,
) -> BrowserVisionQkvCreatedBuffer {
    let VisionQkvWebPhysicalCommand::CreateBuffer { buffer, label, byte_length } = command else {
        return;
    };
    let gpu_buffer = self.create_buffer(label, *byte_length, usage);
    BrowserVisionQkvCreatedBuffer { logical_buffer: *buffer, gpu_buffer }
}

fn apply_vision_qkv_web_create_bind_group_command(
    &self,
    command: &VisionQkvWebPhysicalCommand,
) -> BrowserVisionQkvCreatedBindGroup {
    let VisionQkvWebPhysicalCommand::CreateBindGroup {
        layer_index,
        kind,
        label,
        uniform_slot,
        entries,
    } = command else {
        return;
    };
    let gpu_entries = self.resolve_vision_qkv_web_bind_group_entries(
        layer_index,
        kind,
        uniform_slot,
        entries,
    );
    let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
        label: Some(label),
        entries: &gpu_entries,
    });
    BrowserVisionQkvCreatedBindGroup {
        layer_index: *layer_index,
        kind: *kind,
        bind_group,
    }
}

fn apply_vision_qkv_web_copy_buffer_command(
    &self,
    command: &VisionQkvWebPhysicalCommand,
) {
    let VisionQkvWebPhysicalCommand::CopyBuffer {
        source,
        source_offset,
        destination,
        destination_offset,
        byte_length,
        ..
    } = command else {
        return;
    };
    let resolved_source_buffer = self.resolve_buffer(source);
    let source_buffer = resolved_source_buffer.clone();
    let destination_buffer = self.resolve_buffer(destination);
    encoder.copy_buffer_to_buffer(
        source_buffer,
        *source_offset,
        destination_buffer,
        *destination_offset,
        *byte_length,
    );
}

fn apply_vision_qkv_web_map_range_command(
    &self,
    command: &VisionQkvWebPhysicalCommand,
) {
    let VisionQkvWebPhysicalCommand::MapRange { buffer, byte_range, .. } = command else {
        return;
    };
    let mapped_buffer = self.resolve_buffer(buffer);
    let mapped = mapped_buffer.slice(byte_range.clone()).get_mapped_range();
    consume(mapped);
}

fn resolve_buffer(
    &self,
    buffer: &VisionQkvWebPhysicalBuffer,
) -> &GpuBuffer {
    self.buffers.get(buffer).expect("sealed physical buffer was stored")
}

fn validate_vision_qkv_web_uniform_slot(
    kind: &VisionQkvWebBindGroupKind,
    uniform_slot: &u32,
    entries: &[VisionQkvWebBindGroupEntry],
) {
    let expected = match kind {
        VisionQkvWebBindGroupKind::FusedQkv => 1,
        VisionQkvWebBindGroupKind::Attention => 4,
    };
    let entry_slot = entries.iter().find_map(|entry| match entry.resource() {
        VisionQkvWebBindingResource::Uniform { slot, byte_length: _ } => Some(slot),
        _ => None,
    }).expect("sealed bind group omitted Uniform");
    assert_eq!(uniform_slot, &expected);
    assert_eq!(uniform_slot, entry_slot);
}

fn resolve_vision_qkv_web_bind_group_entries(
    &self,
    layer_index: &u32,
    kind: &VisionQkvWebBindGroupKind,
    uniform_slot: &u32,
    entries: &[VisionQkvWebBindGroupEntry],
) -> Vec<wgpu::BindGroupEntry<'_>> {
    validate_vision_qkv_web_uniform_slot(kind, uniform_slot, entries);
    entries.iter().map(|entry| {
        let resolved_resource = self.resolve_vision_qkv_web_binding_resource(entry.resource());
        wgpu::BindGroupEntry {
            binding: entry.binding(),
            resource: resolved_resource,
        }
    }).collect()
}

fn resolve_vision_qkv_web_binding_resource(
    &self,
    resource: &VisionQkvWebBindingResource,
) -> wgpu::BindingResource<'_> {
    match resource {
        VisionQkvWebBindingResource::Norm1Output { byte_length } =>
            self.resolve_vision_qkv_web_context_binding(&self.norm1_output, *byte_length),
        VisionQkvWebBindingResource::QueryWeight { byte_length } =>
            self.resolve_vision_qkv_web_context_binding(&self.query_weight, *byte_length),
        VisionQkvWebBindingResource::QueryBias { byte_length } =>
            self.resolve_vision_qkv_web_context_binding(&self.query_bias, *byte_length),
        VisionQkvWebBindingResource::KeyWeight { byte_length } =>
            self.resolve_vision_qkv_web_context_binding(&self.key_weight, *byte_length),
        VisionQkvWebBindingResource::KeyBias { byte_length } =>
            self.resolve_vision_qkv_web_context_binding(&self.key_bias, *byte_length),
        VisionQkvWebBindingResource::ValueWeight { byte_length } =>
            self.resolve_vision_qkv_web_context_binding(&self.value_weight, *byte_length),
        VisionQkvWebBindingResource::ValueBias { byte_length } =>
            self.resolve_vision_qkv_web_context_binding(&self.value_bias, *byte_length),
        VisionQkvWebBindingResource::WorkspaceRange { byte_offset, byte_length } => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.resolve_buffer(&VisionQkvWebPhysicalBuffer::Workspace),
            offset: *byte_offset,
            size: wgpu::BufferSize::new(*byte_length),
        }),
        VisionQkvWebBindingResource::Uniform { slot, byte_length } => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.uniform_buffer,
            offset: self.resolve_vision_qkv_web_uniform_offset(*slot, self.uniform_stride),
            size: wgpu::BufferSize::new(*byte_length),
        }),
        VisionQkvWebBindingResource::CuSeqlens { byte_length } =>
            self.resolve_vision_qkv_web_context_binding(&self.cu_seqlens, *byte_length),
        VisionQkvWebBindingResource::AttentionOutput { byte_length } =>
            self.resolve_vision_qkv_web_context_binding(&self.attention_output, *byte_length),
    }
}

fn resolve_vision_qkv_web_context_binding(
    &self,
    binding: &VisionStackBufferBinding<'_>,
    byte_length: u64,
) -> wgpu::BindingResource<'_> {
    assert!(binding.bytes == byte_length);
    binding.resource()
}

fn resolve_vision_qkv_web_uniform_offset(
    &self,
    slot: u32,
    uniform_stride: u64,
) -> u64 {
    u64::from(slot)
        .checked_mul(uniform_stride)
        .expect("sealed uniform offset overflowed")
}

fn create_buffer(
    &self,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> GpuBuffer {
    let buffer = self.device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    });
    buffer
}

fn store_vision_qkv_web_created_buffer(
    &mut self,
    created: BrowserVisionQkvCreatedBuffer,
) {
    self.buffers.insert(created.logical_buffer, created.gpu_buffer);
}

fn store_vision_qkv_web_created_bind_group(
    &mut self,
    created: BrowserVisionQkvCreatedBindGroup,
) {
    self.bind_groups
        .insert((created.layer_index, created.kind), created.bind_group);
}
"#;
    assert_typed_web_physical_adapter_source(VALID);
    assert_typed_web_physical_storage_source(VALID);
    assert_typed_web_physical_resolver_source(VALID);

    let lexical = live_rust_source(
        r##"
// create_buffer(label, byte_length, usage)
let text = "copy_buffer_to_buffer(source, source_offset, destination, destination_offset, byte_length)";
let raw = r#"get_mapped_range()"#;
if false {
    create_bind_group(entries);
}
real_effect();
"##,
    );
    for decoy in [
        "create_buffer(",
        "copy_buffer_to_buffer(",
        "get_mapped_range(",
        "create_bind_group(",
    ] {
        assert!(!lexical.contains(decoy), "sanitizer retained {decoy} decoy");
    }
    assert!(lexical.contains("real_effect("));
    syn::parse_file(&format!("fn lexical_probe() {{ {lexical} }}"))
        .expect("sanitized live Rust must remain parseable");
    let literal_matrix = live_rust_source(
        r##"const STRINGS: [&str; 3] = ["plain", r#"raw"#, "with { braces }"];
const BYTES: &[u8] = b"bytes";
const CHAR: char = '{';
const BYTE: u8 = b'}';"##,
    );
    syn::parse_file(&literal_matrix)
        .expect("sanitized string, byte-string, char, and byte-char literals must parse");

    let correct_create = "let gpu_buffer = self.create_buffer(label, *byte_length, usage);";
    for (label, hostile) in [
        (
            "comment decoy plus live wrong sink",
            VALID.replace(
                correct_create,
                "// let gpu_buffer = self.create_buffer(label, *byte_length, usage);\n    let gpu_buffer = self.create_buffer(label, *byte_length + 4, usage);",
            ),
        ),
        (
            "dead branch decoy plus live wrong sink",
            VALID.replace(
                correct_create,
                "if false { let gpu_buffer = self.create_buffer(label, *byte_length, usage); }\n    let gpu_buffer = self.create_buffer(label, *byte_length - 4, usage);",
            ),
        ),
        (
            "tainted local",
            VALID.replace(
                correct_create,
                "let tainted_bytes = *byte_length + 4;\n    let gpu_buffer = self.create_buffer(label, tainted_bytes, usage);",
            ),
        ),
        (
            "conditional combines a second authority",
            VALID.replace(
                correct_create,
                "let selected_bytes = if hostile { *byte_length } else { wrong_length };\n    let gpu_buffer = self.create_buffer(label, selected_bytes, usage);",
            ),
        ),
        (
            "function combines a second authority",
            VALID.replace(
                correct_create,
                "let selected_bytes = choose(wrong_length, *byte_length);\n    let gpu_buffer = self.create_buffer(label, selected_bytes, usage);",
            ),
        ),
        (
            "max combines a second authority",
            VALID.replace(
                correct_create,
                "let gpu_buffer = self.create_buffer(label, max(*byte_length, wrong_length), usage);",
            ),
        ),
        (
            "duplicate live sink",
            VALID.replace(
                correct_create,
                "let gpu_buffer = self.create_buffer(label, *byte_length, usage);\n    let duplicate = self.create_buffer(label, *byte_length, usage);",
            ),
        ),
        (
            "wrong field and argument",
            VALID.replace(
                correct_create,
                "let gpu_buffer = self.create_buffer(label, *source_offset, usage);",
            ),
        ),
    ] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                assert_typed_web_physical_adapter_source(&hostile);
            }))
            .is_err(),
            "typed physical scanner accepted {label}",
        );
    }

    let exact_resolver = "let resolved_source_buffer = self.resolve_buffer(source);\n    let source_buffer = resolved_source_buffer.clone();";
    for (label, poisoned) in [
        (
            "resolver receives two authorities",
            "let resolved_source_buffer = self.resolve_buffer(source, destination);\n    let source_buffer = resolved_source_buffer.clone();",
        ),
        (
            "resolver alias chain is conditionally poisoned",
            "let resolved_source_buffer = self.resolve_buffer(source);\n    let source_buffer = if hostile { resolved_source_buffer.clone() } else { wrong_buffer };",
        ),
        (
            "resolver alias chain calls a combining helper",
            "let resolved_source_buffer = self.resolve_buffer(source);\n    let source_buffer = choose(resolved_source_buffer.clone(), wrong_buffer);",
        ),
    ] {
        let hostile = VALID.replace(exact_resolver, poisoned);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                assert_typed_web_physical_adapter_source(&hostile);
            }))
            .is_err(),
            "typed physical scanner accepted {label}",
        );
    }

    for (label, hostile) in [
        (
            "discarded exact created buffer before wrong returned identity",
            VALID.replace(
                "BrowserVisionQkvCreatedBuffer { logical_buffer: *buffer, gpu_buffer }",
                "let discarded = BrowserVisionQkvCreatedBuffer { logical_buffer: *buffer, gpu_buffer: gpu_buffer.clone() };\n    BrowserVisionQkvCreatedBuffer { logical_buffer: VisionQkvWebPhysicalBuffer::Readback, gpu_buffer }",
            ),
        ),
        (
            "created buffer swaps its logical role",
            VALID.replace(
                "BrowserVisionQkvCreatedBuffer { logical_buffer: *buffer, gpu_buffer }",
                "BrowserVisionQkvCreatedBuffer { logical_buffer: VisionQkvWebPhysicalBuffer::Readback, gpu_buffer }",
            ),
        ),
        (
            "created bind group increments its layer",
            VALID.replace(
                "layer_index: *layer_index,\n        kind: *kind,",
                "layer_index: *layer_index + 1,\n        kind: *kind,",
            ),
        ),
        (
            "created bind group swaps fused and attention kind",
            VALID.replace(
                "layer_index: *layer_index,\n        kind: *kind,",
                "layer_index: *layer_index,\n        kind: VisionQkvWebBindGroupKind::Attention,",
            ),
        ),
        (
            "bind adapter substitutes its static uniform slot",
            VALID.replace(
                "kind,\n        uniform_slot,\n        entries,",
                "kind,\n        &0,\n        entries,",
            ),
        ),
        (
            "map range resolves a different logical buffer",
            VALID.replace(
                "let mapped_buffer = self.resolve_buffer(buffer);",
                "let mapped_buffer = self.resolve_buffer(other_buffer);",
            ),
        ),
    ] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                assert_typed_web_physical_adapter_source(&hostile);
            }))
            .is_err(),
            "typed physical scanner accepted {label}",
        );
    }

    let exact_buffer_store = "self.buffers.insert(created.logical_buffer, created.gpu_buffer);";
    let exact_group_store = ".insert((created.layer_index, created.kind), created.bind_group);";
    for (label, hostile) in [
        (
            "Workspace/Readback store key swap",
            VALID.replace(
                exact_buffer_store,
                "self.buffers.insert(VisionQkvWebPhysicalBuffer::Readback, created.gpu_buffer);",
            ),
        ),
        (
            "FusedQkv/Attention store key swap",
            VALID.replace(
                exact_group_store,
                ".insert((created.layer_index, VisionQkvWebBindGroupKind::Attention), created.bind_group);",
            ),
        ),
        (
            "bind-group store increments the layer key",
            VALID.replace(
                exact_group_store,
                ".insert((created.layer_index + 1, created.kind), created.bind_group);",
            ),
        ),
        (
            "commented correct store decoys for a live wrong key",
            VALID.replace(
                exact_buffer_store,
                "// self.buffers.insert(created.logical_buffer, created.gpu_buffer);\n    self.buffers.insert(VisionQkvWebPhysicalBuffer::Readback, created.gpu_buffer);",
            ),
        ),
        (
            "dead correct store decoys for a live wrong key",
            VALID.replace(
                exact_buffer_store,
                "if false { self.buffers.insert(created.logical_buffer, created.gpu_buffer); }\n    self.buffers.insert(VisionQkvWebPhysicalBuffer::Readback, created.gpu_buffer);",
            ),
        ),
        (
            "duplicate buffer store",
            VALID.replace(
                exact_buffer_store,
                "self.buffers.insert(created.logical_buffer, created.gpu_buffer);\n    self.buffers.insert(created.logical_buffer, created.gpu_buffer);",
            ),
        ),
    ] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                assert_typed_web_physical_storage_source(&hostile);
            }))
            .is_err(),
            "typed physical storage scanner accepted {label}",
        );
    }

    for (label, hostile) in [
        (
            "buffer resolver always selects Readback",
            VALID.replace(
                "self.buffers.get(buffer)",
                "self.buffers.get(&VisionQkvWebPhysicalBuffer::Readback)",
            ),
        ),
        (
            "commented exact buffer lookup decoys for live Readback",
            VALID.replace(
                "self.buffers.get(buffer)",
                "/* self.buffers.get(buffer) */ self.buffers.get(&VisionQkvWebPhysicalBuffer::Readback)",
            ),
        ),
        (
            "query resource resolves the key buffer",
            VALID.replace(
                "self.resolve_vision_qkv_web_context_binding(&self.query_weight, *byte_length)",
                "self.resolve_vision_qkv_web_context_binding(&self.key_weight, *byte_length)",
            ),
        ),
        (
            "workspace offset and size fields are swapped",
            VALID.replace(
                "offset: *byte_offset,\n            size: wgpu::BufferSize::new(*byte_length),",
                "offset: *byte_length,\n            size: wgpu::BufferSize::new(*byte_offset),",
            ),
        ),
        (
            "entry resolver substitutes binding zero",
            VALID.replace("binding: entry.binding(),", "binding: 0,"),
        ),
        (
            "fused static uniform slot aliases attention",
            VALID.replace(
                "VisionQkvWebBindGroupKind::FusedQkv => 1,",
                "VisionQkvWebBindGroupKind::FusedQkv => 4,",
            ),
        ),
        (
            "uniform slot offset is replaced by zero",
            VALID.replace(
                "offset: self.resolve_vision_qkv_web_uniform_offset(*slot, self.uniform_stride),",
                "offset: 0,",
            ),
        ),
        (
            "context resolver discards physical binding offsets",
            VALID.replace(
                "binding.resource()",
                "wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: binding.buffer, offset: 0, size: binding.size })",
            ),
        ),
        (
            "resource match hides behind a default",
            VALID.replace(
                "VisionQkvWebBindingResource::AttentionOutput { byte_length } =>",
                "_ =>",
            ),
        ),
    ] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                assert_typed_web_physical_resolver_source(&hostile);
            }))
            .is_err(),
            "typed physical resolver scanner accepted {label}",
        );
    }
}

#[test]
fn typed_web_effect_sink_scanner_rejects_conditions_wrappers_duplicates_and_reconstruction() {
    const VALID: &str = r#"
fn apply_vision_qkv_web_create_buffer_command(command: &VisionQkvWebPhysicalCommand) -> CreatedBuffer {
    adapter_create_buffer(command)
}
fn apply_vision_qkv_web_create_bind_group_command(command: &VisionQkvWebPhysicalCommand) -> CreatedBindGroup {
    adapter_create_bind_group(command)
}
fn apply_vision_qkv_web_copy_buffer_command(command: &VisionQkvWebPhysicalCommand) {
    adapter_copy_buffer(command)
}
fn apply_vision_qkv_web_map_range_command(command: &VisionQkvWebPhysicalCommand) {
    adapter_map_range(command)
}
fn store_vision_qkv_web_created_buffer(created: CreatedBuffer) {
    typed_buffer_store(created)
}
fn store_vision_qkv_web_created_bind_group(created: CreatedBindGroup) {
    typed_bind_group_store(created)
}

impl VisionQkvWebPhysicalCommandEffectSink for BrowserVisionQkvPhysicalCommandEffectSink<'_> {
    type CreatedBuffer = CreatedBuffer;
    type CreatedBindGroup = CreatedBindGroup;
    type Error = WebError;

    fn apply_create_buffer(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<Self::CreatedBuffer, Self::Error> {
        let created = self.runtime.apply_vision_qkv_web_create_buffer_command(command);
        Ok(created)
    }

    fn store_created_buffer(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
        created: Self::CreatedBuffer,
    ) -> Result<(), Self::Error> {
        self.runtime.store_vision_qkv_web_created_buffer(created);
        Ok(())
    }

    fn apply_create_bind_group(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<Self::CreatedBindGroup, Self::Error> {
        let created = self.runtime.apply_vision_qkv_web_create_bind_group_command(command);
        Ok(created)
    }

    fn store_created_bind_group(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
        created: Self::CreatedBindGroup,
    ) -> Result<(), Self::Error> {
        self.runtime.store_vision_qkv_web_created_bind_group(created);
        Ok(())
    }

    fn apply_copy_buffer(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<(), Self::Error> {
        self.runtime.apply_vision_qkv_web_copy_buffer_command(command);
        Ok(())
    }

    fn apply_map_range(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<(), Self::Error> {
        self.runtime.apply_vision_qkv_web_map_range_command(command);
        Ok(())
    }
}

fn apply_vision_qkv_web_start_commands(
    runtime: &mut WebRuntime,
    plan: &VisionQkvWebPhysicalCommandPlan,
) -> Result<(), WebError> {
    let mut sink = BrowserVisionQkvPhysicalCommandEffectSink { runtime };
    execute_vision_qkv_web_physical_commands(
        plan,
        VisionQkvWebPhysicalCommandPhase::Start,
        &mut sink,
    )
}

fn apply_vision_qkv_web_layer_commands(
    runtime: &mut WebRuntime,
    plan: &VisionQkvWebPhysicalCommandPlan,
    layer_index: u32,
) -> Result<(), WebError> {
    let mut sink = BrowserVisionQkvPhysicalCommandEffectSink { runtime };
    execute_vision_qkv_web_physical_commands(
        plan,
        VisionQkvWebPhysicalCommandPhase::Layer { layer_index },
        &mut sink,
    )
}

fn apply_vision_qkv_web_finish_commands(
    runtime: &mut WebRuntime,
    plan: &VisionQkvWebPhysicalCommandPlan,
) -> Result<(), WebError> {
    let mut sink = BrowserVisionQkvPhysicalCommandEffectSink { runtime };
    execute_vision_qkv_web_physical_commands(
        plan,
        VisionQkvWebPhysicalCommandPhase::Finish,
        &mut sink,
    )
}
"#;
    assert_web_physical_effect_sink_source(VALID);

    let create = "let created = self.runtime.apply_vision_qkv_web_create_buffer_command(command);";
    let copy = "self.runtime.apply_vision_qkv_web_copy_buffer_command(command);";
    let buffer_store = "self.runtime.store_vision_qkv_web_created_buffer(created);";
    let start_execute = r#"execute_vision_qkv_web_physical_commands(
        plan,
        VisionQkvWebPhysicalCommandPhase::Start,
        &mut sink,
    )"#;
    for (label, hostile) in [
        (
            "target cfg gate",
            VALID.replace(
                copy,
                "if cfg!(not(target_arch = \"wasm32\")) { self.runtime.apply_vision_qkv_web_copy_buffer_command(command); }",
            ),
        ),
        (
            "literal condition gate",
            VALID.replace(
                copy,
                "if true { self.runtime.apply_vision_qkv_web_copy_buffer_command(command); }",
            ),
        ),
        (
            "nonliteral condition gate",
            VALID.replace(
                copy,
                "if self.runtime.enabled { self.runtime.apply_vision_qkv_web_copy_buffer_command(command); }",
            ),
        ),
        (
            "match condition gate",
            VALID.replace(
                copy,
                "match self.runtime.enabled { true => self.runtime.apply_vision_qkv_web_copy_buffer_command(command), false => () }",
            ),
        ),
        (
            "dead correct route",
            VALID.replace(
                copy,
                "if false { self.runtime.apply_vision_qkv_web_copy_buffer_command(command); } self.runtime.raw_copy(command);",
            ),
        ),
        (
            "early return before correct route",
            VALID.replace(
                copy,
                "return Ok(()); self.runtime.apply_vision_qkv_web_copy_buffer_command(command);",
            ),
        ),
        (
            "reachable wrapper filter",
            format!(
                "{}\nfn replay_filtered(plan: &VisionQkvWebPhysicalCommandPlan) {{ for command in plan.commands().iter().filter(|_| true) {{ raw(command); }} }}",
                VALID.replace(start_execute, &format!("{start_execute}; replay_filtered(plan)")),
            ),
        ),
        (
            "reachable conditional wrapper",
            format!(
                "{}\nfn conditional_route(command: &VisionQkvWebPhysicalCommand) {{ if hostile {{ raw(command); }} }}",
                VALID.replace(copy, &format!("{copy} conditional_route(command);")),
            ),
        ),
        (
            "duplicate adapter",
            VALID.replace(copy, &format!("{copy} {copy}")),
        ),
        (
            "missing adapter",
            VALID.replace(copy, "drop(command);"),
        ),
        (
            "duplicate typed store",
            VALID.replace(buffer_store, &format!("{buffer_store} {buffer_store}")),
        ),
        (
            "missing typed store",
            VALID.replace(buffer_store, "drop(created);"),
        ),
        (
            "reconstructed command",
            VALID.replace(create, "let created = self.runtime.apply_vision_qkv_web_create_buffer_command(&command.clone());"),
        ),
        (
            "reconstructed created result",
            VALID.replace("Ok(created)\n    }", "Ok(rebuild(created))\n    }"),
        ),
        (
            "cloned store result",
            VALID.replace(buffer_store, "self.runtime.store_vision_qkv_web_created_buffer(created.clone());"),
        ),
        (
            "alternate raw sink",
            VALID.replace(
                copy,
                &format!("{copy} self.runtime.device.create_buffer(raw_descriptor);"),
            ),
        ),
        (
            "alternate raw sink hides call syntax behind a comment",
            VALID.replace(
                copy,
                &format!(
                    "{copy} self.runtime.device.create_buffer /* lexical decoy */ (raw_descriptor);"
                ),
            ),
        ),
        (
            "typed store hides an alternate raw sink behind a comment",
            VALID.replace(
                buffer_store,
                &format!(
                    "{buffer_store} self.runtime.device.create_buffer /* lexical decoy */ (raw_descriptor);"
                ),
            ),
        ),
        (
            "phase root hides an alternate raw sink behind a comment",
            VALID.replace(
                start_execute,
                &format!(
                    "{start_execute}; runtime.device.create_buffer /* lexical decoy */ (raw_descriptor)"
                ),
            ),
        ),
    ] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                assert_web_physical_effect_sink_source(&hostile);
            }))
            .is_err(),
            "Web typed effect-sink scanner accepted {label}",
        );
    }
}

#[test]
fn typed_web_layer_context_wiring_stays_inside_the_effect_sink() {
    const VALID: &str = r#"
impl VisionQkvWebPhysicalCommandEffectSink for BrowserVisionQkvPhysicalCommandEffectSink<'_> {
    fn apply_create_bind_group(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<Self::CreatedBindGroup, Self::Error> {
        let created = self
            .context
            .apply_vision_qkv_web_create_bind_group_command(command);
        Ok(created)
    }
}

fn apply_vision_qkv_web_layer_commands(
    context: &BrowserVisionQkvLayerResolutionContext<'_>,
    plan: &VisionQkvWebPhysicalCommandPlan,
    layer_index: u32,
) -> Result<(), WebError> {
    let mut sink = BrowserVisionQkvPhysicalCommandEffectSink { context };
    execute_vision_qkv_web_physical_commands(
        plan,
        VisionQkvWebPhysicalCommandPhase::Layer { layer_index },
        &mut sink,
    )
}

fn invoke_fused_layer(
    constructed_context: BrowserVisionQkvLayerResolutionContext<'_>,
    alternate_context: BrowserVisionQkvLayerResolutionContext<'_>,
    plan: &VisionQkvWebPhysicalCommandPlan,
    layer_index: u32,
) -> Result<(), WebError> {
    let _ = alternate_context;
    apply_vision_qkv_web_layer_commands(&constructed_context, plan, layer_index)
}
"#;
    let context_parameter_index = assert_web_layer_physical_context_wiring(VALID);
    let invoke = braced_item(VALID, "fn invoke_fused_layer(");
    assert_web_layer_context_call_argument(invoke, context_parameter_index, "constructed_context");

    for (label, hostile) in [
        (
            "phase substitutes another context",
            VALID.replace(
                "BrowserVisionQkvPhysicalCommandEffectSink { context }",
                "BrowserVisionQkvPhysicalCommandEffectSink { context: alternate_context }",
            ),
        ),
        (
            "effect sink invokes another context",
            VALID.replace(
                "self\n            .context\n            .apply_vision_qkv_web_create_bind_group_command(command)",
                "self\n            .alternate_context\n            .apply_vision_qkv_web_create_bind_group_command(command)",
            ),
        ),
        (
            "effect sink reconstructs the sealed command",
            VALID.replace(
                ".apply_vision_qkv_web_create_bind_group_command(command)",
                ".apply_vision_qkv_web_create_bind_group_command(&command.clone())",
            ),
        ),
        (
            "phase bypasses the common executor",
            VALID.replace(
                "let mut sink = BrowserVisionQkvPhysicalCommandEffectSink { context };",
                "context.apply_vision_qkv_web_create_bind_group_command(command);\n    let mut sink = BrowserVisionQkvPhysicalCommandEffectSink { context };",
            ),
        ),
        (
            "phase shadows its authenticated context parameter",
            VALID.replace(
                "let mut sink = BrowserVisionQkvPhysicalCommandEffectSink { context };",
                "let context = alternate_context;\n    let mut sink = BrowserVisionQkvPhysicalCommandEffectSink { context };",
            ),
        ),
        (
            "effect sink shadows and clones its authenticated command",
            VALID.replace(
                "let created = self",
                "let command = &command.clone();\n        let created = self",
            ),
        ),
        (
            "phase accepts an extra exact typed decoy context",
            VALID.replace(
                "context: &BrowserVisionQkvLayerResolutionContext<'_>,\n    plan:",
                "context: &BrowserVisionQkvLayerResolutionContext<'_>,\n    decoy_context: &BrowserVisionQkvLayerResolutionContext<'_>,\n    plan:",
            ),
        ),
        (
            "constructed context is supplied only to an unused duplicate parameter",
            VALID
                .replace(
                    "context: &BrowserVisionQkvLayerResolutionContext<'_>,\n    plan:",
                    "context: &BrowserVisionQkvLayerResolutionContext<'_>,\n    unused_context: &BrowserVisionQkvLayerResolutionContext<'_>,\n    plan:",
                )
                .replace(
                    "apply_vision_qkv_web_layer_commands(&constructed_context, plan, layer_index)",
                    "apply_vision_qkv_web_layer_commands(&alternate_context, &constructed_context, plan, layer_index)",
                ),
        ),
        (
            "phase context uses a type-name-prefix proxy",
            VALID.replacen(
                "context: &BrowserVisionQkvLayerResolutionContext<'_>",
                "context: &BrowserVisionQkvLayerResolutionContextProxy<'_>",
                1,
            ),
        ),
    ] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                assert_web_layer_physical_context_wiring(&hostile);
            }))
            .is_err(),
            "typed Web layer context scanner accepted {label}",
        );
    }

    let receiver_shift = VALID
        .replace(
            "fn apply_vision_qkv_web_layer_commands(\n    context: &BrowserVisionQkvLayerResolutionContext<'_>,",
            "impl Host {\nfn apply_vision_qkv_web_layer_commands(\n    &self,\n    context: &BrowserVisionQkvLayerResolutionContext<'_>,\n    decoy_context: &impl Sized,",
        )
        .replace(
            "    )\n}\n\nfn invoke_fused_layer(",
            "    )\n    }\n}\n\nfn invoke_fused_layer(",
        )
        .replace(
            "    alternate_context: BrowserVisionQkvLayerResolutionContext<'_>,\n    plan:",
            "    alternate_context: BrowserVisionQkvLayerResolutionContext<'_>,\n    host: &Host,\n    plan:",
        )
        .replace(
            "apply_vision_qkv_web_layer_commands(&constructed_context, plan, layer_index)",
            "host.apply_vision_qkv_web_layer_commands(&alternate_context, &constructed_context, plan, layer_index)",
        );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let context_parameter_index = assert_web_layer_physical_context_wiring(&receiver_shift);
            let invoke = braced_item(&receiver_shift, "fn invoke_fused_layer(");
            assert_web_layer_context_call_argument(
                invoke,
                context_parameter_index,
                "constructed_context",
            );
        }))
        .is_err(),
        "typed Web layer context scanner accepted a receiver-shifted method root",
    );

    let wrong_call_context = VALID.replace(
        "apply_vision_qkv_web_layer_commands(&constructed_context, plan, layer_index)",
        "let _ = &constructed_context;\n    apply_vision_qkv_web_layer_commands(&alternate_context, plan, layer_index)",
    );
    let context_parameter_index = assert_web_layer_physical_context_wiring(&wrong_call_context);
    let invoke = braced_item(&wrong_call_context, "fn invoke_fused_layer(");
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            assert_web_layer_context_call_argument(
                invoke,
                context_parameter_index,
                "constructed_context",
            );
        }))
        .is_err(),
        "typed Web layer call accepted an unused constructed context and alternate sink context",
    );
}

#[test]
fn selection_evidence_scanner_rejects_dead_handoff_calls_constants_nulls_and_caller_records() {
    const VALID_LIB: &str = r#"
struct VisionQkvSelectionEvidence {
    semantic_graph_blake3: String,
}

#[derive(Serialize)]
pub struct VisionQkvEvidenceEnvelope<'a, E> {
    qkv_selection: &'a VisionQkvSelectionEvidence,
    qkv_execution: Option<&'a E>,
}

impl<'a, E> VisionQkvEvidenceEnvelope<'a, E> {
    pub fn qkv_selection(&self) -> &VisionQkvSelectionEvidence {
        self.qkv_selection
    }
    pub fn qkv_execution(&self) -> Option<&E> {
        self.qkv_execution
    }
}

pub struct VisionQkvSelectionEvidencePropagation {
    evidence: Rc<VisionQkvSelectionEvidence>,
}

impl VisionQkvSelectionEvidencePropagation {
    pub fn opaque_selection_evidence(&self) -> &VisionQkvSelectionEvidence {
        &self.evidence
    }
    pub fn evidence_json(&self) -> Result<String, EvidenceError> {
        serialize_selection(&self.evidence)
    }
    pub fn additive_begin_evidence<'a, E>(
        &'a self,
        qkv_execution: Option<&'a E>,
    ) -> VisionQkvEvidenceEnvelope<'a, E> {
        self.evidence_envelope(qkv_execution)
    }
    pub fn final_diagnostics_evidence<'a, E>(
        &'a self,
        qkv_execution: Option<&'a E>,
    ) -> VisionQkvEvidenceEnvelope<'a, E> {
        self.evidence_envelope(qkv_execution)
    }
    pub fn uses_legacy_topology(&self) -> bool {
        selection_uses_legacy(&self.evidence)
    }
    fn evidence_envelope<'a, E>(
        &'a self,
        qkv_execution: Option<&'a E>,
    ) -> VisionQkvEvidenceEnvelope<'a, E> {
        VisionQkvEvidenceEnvelope {
            qkv_selection: &self.evidence,
            qkv_execution,
        }
    }
}

pub fn build_vision_qkv_selection_evidence_propagation(
    handoff: &VisionQkvCompilerHandoff,
) -> VisionQkvSelectionEvidencePropagation {
    let semantic_graph_blake3 = handoff.semantic_graph_blake3_hex().to_owned();
    let evidence = VisionQkvSelectionEvidence {
        semantic_graph_blake3,
    };
    VisionQkvSelectionEvidencePropagation {
        evidence: Rc::new(evidence),
    }
}

struct BrowserVisionQkvExecutionWorkspaceEvidence {
    logical_id: &'static str,
    allocation_bytes: u64,
    semantic_base: u64,
    semantic_bytes: u64,
}

struct BrowserVisionQkvExecutionBindingEvidence {
    binding: u32,
    byte_offset: u64,
    byte_length: u64,
}

struct BrowserVisionQkvExecutionCanaryPlanEvidence {
    kind: &'static str,
    plane: Option<u32>,
    byte_offset: u64,
    byte_length: u64,
}

pub struct BrowserVisionQkvExecutionEvidencePlan {
    dispatch_count: u32,
    command_buffer_count: u32,
    submission_count: u32,
    map_count: u32,
    workspace: BrowserVisionQkvExecutionWorkspaceEvidence,
    bindings: Vec<BrowserVisionQkvExecutionBindingEvidence>,
    canaries: Vec<BrowserVisionQkvExecutionCanaryPlanEvidence>,
}

#[derive(Serialize)]
struct BrowserVisionQkvExecutionCanaryEvidence<'a> {
    kind: &'a str,
    plane: Option<u32>,
    byte_offset: u64,
    byte_length: u64,
    passed: Option<bool>,
}

#[derive(Serialize)]
struct BrowserVisionQkvExecutionEvidence<'a> {
    dispatch_count: u32,
    command_buffer_count: u32,
    submission_count: u32,
    map_count: u32,
    workspace: &'a BrowserVisionQkvExecutionWorkspaceEvidence,
    bindings: &'a [BrowserVisionQkvExecutionBindingEvidence],
    canaries: Vec<BrowserVisionQkvExecutionCanaryEvidence<'a>>,
}

#[derive(Serialize)]
pub struct BrowserVisionQkvBeginExecutionEvidence<'a> {
    #[serde(flatten)]
    evidence: BrowserVisionQkvExecutionEvidence<'a>,
}

#[derive(Serialize)]
pub struct BrowserVisionQkvFinalExecutionEvidence<'a> {
    #[serde(flatten)]
    evidence: BrowserVisionQkvExecutionEvidence<'a>,
}

impl BrowserVisionQkvExecutionEvidencePlan {
    pub fn from_prepared(
        prepared: Option<&VisionQkvPhysicalExecutionSpec>,
    ) -> Result<Option<Self>, EvidenceError> {
        let Some(physical_spec) = prepared else {
            return Ok(None);
        };
        let prepared_execution = physical_spec.prepared_execution();
        let layer_count = u32::try_from(prepared_execution.layer_count())?;
        let dispatch_count = checked_qkv_dispatch_count(layer_count)?;
        let command_buffer_count = checked_qkv_operation_count(layer_count)?;
        let submission_count = command_buffer_count;
        let workspace = prepared_execution.workspace();
        let bindings = prepared_execution.layers()[0]
            .attention_bridge()
            .bindings()
            .iter()
            .map(|binding| BrowserVisionQkvExecutionBindingEvidence {
                binding: binding.binding(),
                byte_offset: binding.byte_offset(),
                byte_length: binding.byte_length(),
            })
            .collect();
        let canaries = workspace
            .canaries()
            .iter()
            .map(|canary| {
                let (kind, plane) = serialize_vision_qkv_canary_kind(canary.kind());
                BrowserVisionQkvExecutionCanaryPlanEvidence {
                    kind,
                    plane,
                    byte_offset: canary.byte_offset(),
                    byte_length: canary.byte_length(),
                }
            })
            .collect();
        let workspace = BrowserVisionQkvExecutionWorkspaceEvidence {
            logical_id: "vision-stack-qkv-workspace",
            allocation_bytes: workspace.allocation_bytes(),
            semantic_base: workspace.semantic_base(),
            semantic_bytes: workspace.semantic_bytes(),
        };
        Ok(Some(BrowserVisionQkvExecutionEvidencePlan {
            dispatch_count,
            command_buffer_count,
            submission_count,
            map_count: 1,
            workspace,
            bindings,
            canaries,
        }))
    }

    fn channel_evidence(
        &self,
        passed: Vec<Option<bool>>,
    ) -> BrowserVisionQkvExecutionEvidence<'_> {
        let canaries = self
            .canaries
            .iter()
            .zip(passed)
            .map(|(canary, passed)| BrowserVisionQkvExecutionCanaryEvidence {
                kind: canary.kind,
                plane: canary.plane,
                byte_offset: canary.byte_offset,
                byte_length: canary.byte_length,
                passed,
            })
            .collect();
        BrowserVisionQkvExecutionEvidence {
            dispatch_count: self.dispatch_count,
            command_buffer_count: self.command_buffer_count,
            submission_count: self.submission_count,
            map_count: self.map_count,
            workspace: &self.workspace,
            bindings: &self.bindings,
            canaries,
        }
    }
}

impl<'a> BrowserVisionQkvBeginExecutionEvidence<'a> {
    pub fn from_plan(
        plan: Option<&'a BrowserVisionQkvExecutionEvidencePlan>,
    ) -> Option<Self> {
        plan.map(|plan| {
            let evidence = plan.channel_evidence(vec![None; plan.canaries.len()]);
            BrowserVisionQkvBeginExecutionEvidence { evidence }
        })
    }
}

impl<'a> BrowserVisionQkvFinalExecutionEvidence<'a> {
    pub fn from_verified_plan(
        plan: Option<&'a BrowserVisionQkvExecutionEvidencePlan>,
        canary_results: &[bool],
    ) -> Result<Option<Self>, EvidenceError> {
        let Some(plan) = plan else {
            if !canary_results.is_empty() {
                return Err(EvidenceError::UnexpectedCanaryResults);
            }
            return Ok(None);
        };
        if canary_results.len() != plan.canaries.len() {
            return Err(EvidenceError::CanaryResultCount);
        }
        let passed = canary_results.iter().copied().map(Some).collect();
        let evidence = plan.channel_evidence(passed);
        Ok(Some(BrowserVisionQkvFinalExecutionEvidence { evidence }))
    }
}

#[derive(Serialize)]
struct VisionStackQkvSerializedRecord<'a, L, E> {
    #[serde(flatten)]
    legacy: &'a L,
    #[serde(flatten)]
    evidence: E,
}

fn serialize_vision_stack_qkv_record_json<L: Serialize, E: Serialize>(
    legacy: &L,
    evidence: E,
) -> Result<String, EvidenceError> {
    to_json_string(&VisionStackQkvSerializedRecord { legacy, evidence })
}

pub fn serialize_vision_stack_qkv_begin_status_json<L: Serialize, E: Serialize>(
    legacy: &L,
    evidence: E,
) -> Result<String, EvidenceError> {
    serialize_vision_stack_qkv_record_json(legacy, evidence)
}

pub fn serialize_vision_stack_qkv_final_diagnostics_json<L: Serialize, E: Serialize>(
    legacy: &L,
    evidence: E,
) -> Result<String, EvidenceError> {
    serialize_vision_stack_qkv_record_json(legacy, evidence)
}
"#;
    const VALID_WEB: &str = r#"
pub struct WebVisionQkvStackSelection {
    handoff: VisionQkvCompilerHandoff,
    evidence: VisionQkvSelectionEvidencePropagation,
}

impl WebVisionQkvStackSelection {
    pub fn evidence_json(&self) -> Result<String, JsValue> {
        self.evidence.evidence_json().map_err(js_error)
    }
}

pub fn compile_vision_encoder_stack_qkv_selection(
    bytes: &[u8],
) -> Result<WebVisionQkvStackSelection, JsValue> {
    let handoff = compile_vision_qkv_stack_handoff(bytes)?;
    let evidence = build_vision_qkv_selection_evidence_propagation(&handoff);
    Ok(WebVisionQkvStackSelection { handoff, evidence })
}

struct BrowserVisionStackSession {
    qkv_selection_evidence: Option<VisionQkvSelectionEvidencePropagation>,
    qkv_execution_evidence_plan: Option<BrowserVisionQkvExecutionEvidencePlan>,
}

fn begin_vision_stack_sharded_with_qkv_selection(
    qkv_selection: &WebVisionQkvStackSelection,
) -> Result<String, JsValue> {
    let session_evidence = qkv_selection.evidence.clone();
    let qkv_physical_execution = prepare_physical_execution(&qkv_selection.handoff)?;
    let evidence_plan = BrowserVisionQkvExecutionEvidencePlan::from_prepared(
        qkv_physical_execution.as_ref(),
    )?;
    let session = BrowserVisionStackSession {
        qkv_selection_evidence: Some(session_evidence),
        qkv_execution_evidence_plan: evidence_plan,
    };
    let status = vision_stack_qkv_status_json(&session)?;
    Ok(status)
}

fn vision_stack_qkv_status_json(
    session: &BrowserVisionStackSession,
) -> Result<String, JsValue> {
    let legacy_status = build_vision_stack_legacy_status_record(session)?;
    let qkv_execution = BrowserVisionQkvBeginExecutionEvidence::from_plan(
        session.qkv_execution_evidence_plan.as_ref(),
    );
    let selection_evidence = session.qkv_selection_evidence.as_ref().unwrap();
    let evidence = selection_evidence.additive_begin_evidence(qkv_execution.as_ref());
    let json = serialize_vision_stack_qkv_begin_status_json(&legacy_status, evidence)?;
    Ok(json)
}

fn vision_stack_qkv_diagnostics_json(
    legacy_diagnostics: &VisionStackLegacyDiagnosticsRecord,
    session: &BrowserVisionStackSession,
    canary_results: &BrowserVisionQkvCanaryResults,
) -> Result<String, JsValue> {
    let selection_option = session.qkv_selection_evidence.as_ref();
    match selection_option {
        Some(selection_evidence) => {
            let qkv_execution = BrowserVisionQkvFinalExecutionEvidence::from_verified_plan(
                session.qkv_execution_evidence_plan.as_ref(),
                canary_results,
            )?;
            let evidence =
                selection_evidence.final_diagnostics_evidence(qkv_execution.as_ref());
            crate::serialize_vision_stack_qkv_final_diagnostics_json(legacy_diagnostics, evidence)
        }
        None => crate::serialize_vision_stack_legacy_diagnostics_json(legacy_diagnostics),
    }
}

fn finish_vision_stack_sharded_once(
    session: BrowserVisionStackSession,
) -> Result<JsValue, JsValue> {
    let canary_results = verify_mapped_qkv_canaries(&session)?;
    let legacy_diagnostics = build_vision_stack_legacy_diagnostics_record(&session)?;
    let diagnostics_json = vision_stack_qkv_diagnostics_json(
        &legacy_diagnostics,
        session,
        &canary_results,
    )?;
    let result = Object::new();
    Reflect::set(
        &result,
        &"diagnostics".into(),
        &JsValue::from_str(&diagnostics_json),
    )?;
    Ok(result.into())
}
"#;
    assert_selection_evidence_authority_source(VALID_LIB, VALID_WEB);

    for (label, hostile_lib, hostile_web) in [
        (
            "dead semantic handoff call plus constant",
            VALID_LIB.replace(
                "let semantic_graph_blake3 = handoff.semantic_graph_blake3_hex().to_owned();",
                "let observed = handoff.semantic_graph_blake3_hex().to_owned();\n    let semantic_graph_blake3 = CANONICAL_SEMANTIC_GRAPH_BLAKE3.to_owned();",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "discarded semantic handoff call",
            VALID_LIB.replace(
                "let semantic_graph_blake3 = handoff.semantic_graph_blake3_hex().to_owned();",
                "let _observed = handoff.semantic_graph_blake3_hex();\n    let semantic_graph_blake3 = CANONICAL_SEMANTIC_GRAPH_BLAKE3.to_owned();",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "null semantic identity",
            VALID_LIB.replace(
                "semantic_graph_blake3,",
                "semantic_graph_blake3: None,",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "wrong handoff semantic identity",
            VALID_LIB.replace(
                "handoff.semantic_graph_blake3_hex()",
                "other_handoff.semantic_graph_blake3_hex()",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "nested field named handoff supplies semantic identity",
            VALID_LIB.replace(
                "handoff.semantic_graph_blake3_hex()",
                "other.handoff.semantic_graph_blake3_hex()",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "caller-authored evidence record",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "let session_evidence = qkv_selection.evidence.clone();",
                "let caller = VisionQkvSelectionEvidence { semantic_graph_blake3: constant() };\n    let session_evidence = qkv_selection.evidence.clone();",
            ),
        ),
        (
            "opaque handoff reserialization",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "self.evidence.evidence_json().map_err(js_error)",
                "serialize_handoff(&self.handoff).map_err(js_error)",
            ),
        ),
        (
            "discarded delegated serializer",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "self.evidence.evidence_json().map_err(js_error)",
                "let delegated = self.evidence.evidence_json(); Ok(constant_json())",
            ),
        ),
        (
            "accessor returns alternate evidence",
            VALID_LIB.replace(
                "pub fn opaque_selection_evidence(&self) -> &VisionQkvSelectionEvidence {\n        &self.evidence\n    }",
                "pub fn opaque_selection_evidence(&self) -> &VisionQkvSelectionEvidence {\n        let exact = &self.evidence; alternate_evidence()\n    }",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "factory receives reconstructed handoff",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "build_vision_qkv_selection_evidence_propagation(&handoff)",
                "build_vision_qkv_selection_evidence_propagation(&handoff.clone())",
            ),
        ),
        (
            "begin rebuilds evidence",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "let session_evidence = qkv_selection.evidence.clone();",
                "let session_evidence = build_vision_qkv_selection_evidence_propagation(&qkv_selection.handoff);",
            ),
        ),
        (
            "dead correct begin accessor with constant production selection",
            VALID_LIB.to_owned(),
            format!(
                "{}\nfn dead_begin_accessor(authority: &VisionQkvSelectionEvidencePropagation) {{ let _ = authority.additive_begin_evidence::<serde_json::Value>(None); }}",
                VALID_WEB.replace(
                    "let selection_evidence = session.qkv_selection_evidence.as_ref().unwrap();\n    let evidence = selection_evidence.additive_begin_evidence(qkv_execution.as_ref());",
                    "let selection_evidence = constant_selection_propagation();\n    let evidence = selection_evidence.additive_begin_evidence(qkv_execution.as_ref());",
                )
            ),
        ),
        (
            "null final selection authority",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "let selection_option = session.qkv_selection_evidence.as_ref();",
                "let selection_option = None;",
            ),
        ),
        (
            "equal-by-value reconstructed selection allocation",
            VALID_LIB.replace(
                "qkv_selection: &self.evidence,",
                "qkv_selection: Box::leak(Box::new((*self.evidence).clone())),",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "prepared execution builder observes authority but stores a constant count",
            VALID_LIB.replace(
                "            dispatch_count,\n            command_buffer_count,",
                "            dispatch_count: 31,\n            command_buffer_count,",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "prepared execution builder stores a constant workspace range",
            VALID_LIB.replace(
                "allocation_bytes: workspace.allocation_bytes(),",
                "allocation_bytes: 256,",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "commented exact workspace identity precedes a wrong live identity",
            VALID_LIB.replace(
                "logical_id: \"vision-stack-qkv-workspace\",",
                "// logical_id: \"vision-stack-qkv-workspace\",\n            logical_id: \"wrong-workspace\",",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "execution plan gains mutable divergent cache",
            VALID_LIB.replace(
                "    canaries: Vec<BrowserVisionQkvExecutionCanaryPlanEvidence>,\n}",
                "    canaries: Vec<BrowserVisionQkvExecutionCanaryPlanEvidence>,\n    divergent: RefCell<Option<BrowserVisionQkvExecutionEvidencePlan>>,\n}",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "begin builder hardcodes successful canaries",
            VALID_LIB.replace(
                "vec![None; plan.canaries.len()]",
                "vec![Some(true); plan.canaries.len()]",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "final builder ignores mixed verified results",
            VALID_LIB.replace(
                "canary_results.iter().copied().map(Some).collect()",
                "std::iter::repeat_n(Some(true), plan.canaries.len()).collect()",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "final builder reconstructs an equal-value plan",
            VALID_LIB.replace(
                "let evidence = plan.channel_evidence(passed);",
                "let reconstructed = reconstruct_equal_plan(plan);\n        let evidence = reconstructed.channel_evidence(passed);",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "omitted qkv_execution envelope field",
            VALID_LIB.replace("    qkv_execution: Option<&'a E>,\n", ""),
            VALID_WEB.to_owned(),
        ),
        (
            "conditionally omitted null qkv_execution field",
            VALID_LIB.replace(
                "    qkv_execution: Option<&'a E>,",
                "    #[serde(skip_serializing_if = \"Option::is_none\")]\n    qkv_execution: Option<&'a E>,",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "begin uses final channel accessor",
            VALID_LIB.to_owned(),
            VALID_WEB.replacen(
                ".additive_begin_evidence(qkv_execution.as_ref())",
                ".final_diagnostics_evidence(qkv_execution.as_ref())",
                1,
            ),
        ),
        (
            "final serializes begin channel execution view",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "BrowserVisionQkvFinalExecutionEvidence::from_verified_plan(",
                "BrowserVisionQkvBeginExecutionEvidence::from_plan(",
            ),
        ),
        (
            "final reuses cached begin execution value",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "session.qkv_execution_evidence_plan.as_ref(),\n                canary_results,",
                "session.cached_begin_execution.as_ref(),\n                canary_results,",
            ),
        ),
        (
            "final serializes constant canary results",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "                canary_results,\n            )?;",
                "                constant_canary_results(),\n            )?;",
            ),
        ),
        (
            "dead envelope plus caller-authored production record",
            VALID_LIB.to_owned(),
            VALID_WEB.replacen(
                "let evidence = selection_evidence.additive_begin_evidence(qkv_execution.as_ref());\n    let json = serialize_vision_stack_qkv_begin_status_json(&legacy_status, evidence)?;",
                "let _dead = selection_evidence.additive_begin_evidence(qkv_execution.as_ref());\n    let evidence = caller_authored_record();\n    let json = serialize_vision_stack_qkv_begin_status_json(&legacy_status, evidence)?;",
                1,
            ),
        ),
        (
            "begin envelope authority is mutated with a compound assignment",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "let evidence = selection_evidence.additive_begin_evidence(qkv_execution.as_ref());",
                "let mut evidence = selection_evidence.additive_begin_evidence(qkv_execution.as_ref());\n    evidence += caller_authored_record();",
            ),
        ),
        (
            "duplicate begin serializer",
            VALID_LIB.to_owned(),
            VALID_WEB.replacen(
                "let json = serialize_vision_stack_qkv_begin_status_json(&legacy_status, evidence)?;",
                "let _discarded = serialize_vision_stack_qkv_begin_status_json(&legacy_status, evidence);\n    let json = serialize_vision_stack_qkv_begin_status_json(&legacy_status, evidence)?;",
                1,
            ),
        ),
        (
            "host begin serializer conditionally replaces the common record",
            VALID_LIB.replace(
                "serialize_vision_stack_qkv_record_json(legacy, evidence)\n}\n\npub fn serialize_vision_stack_qkv_final_diagnostics_json",
                "if false { serialize_vision_stack_qkv_record_json(legacy, evidence) } else { Ok(constant_json()) }\n}\n\npub fn serialize_vision_stack_qkv_final_diagnostics_json",
            ),
            VALID_WEB.to_owned(),
        ),
        (
            "actual finish returns begin status value",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "    )?;\n    let result = Object::new();",
                "    )?;\n    let diagnostics_json = vision_stack_qkv_status_json(&session)?;\n    let result = Object::new();",
            ),
        ),
        (
            "optimized begin conditionally replaces actual status JSON",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "Ok(status)",
                "Ok(if false { status } else { constant_json() })",
            ),
        ),
        (
            "optimized finish conditionally replaces actual diagnostics JSON",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "&JsValue::from_str(&diagnostics_json),",
                "&if false { JsValue::from_str(&diagnostics_json) } else { JsValue::from_str(&constant_json()) },",
            ),
        ),
        (
            "optimized begin conditionally returns a constant before its real status tail",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "    let status = vision_stack_qkv_status_json(&session)?;",
                "    if hostile { return Ok(constant_json()); }\n    let status = vision_stack_qkv_status_json(&session)?;",
            ),
        ),
        (
            "begin status serializer conditionally returns a constant before its real tail",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "    let legacy_status = build_vision_stack_legacy_status_record(session)?;",
                "    if hostile { return Ok(constant_json()); }\n    let legacy_status = build_vision_stack_legacy_status_record(session)?;",
            ),
        ),
        (
            "final diagnostics serializer conditionally returns a constant before its real tail",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "    let qkv_execution = BrowserVisionQkvFinalExecutionEvidence::from_verified_plan(",
                "    if hostile { return Ok(constant_json()); }\n    let qkv_execution = BrowserVisionQkvFinalExecutionEvidence::from_verified_plan(",
            ),
        ),
        (
            "finish Reflect flow conditionally returns a constant before its real write",
            VALID_LIB.to_owned(),
            VALID_WEB.replace(
                "    let result = Object::new();\n    Reflect::set(",
                "    let result = Object::new();\n    if hostile { return Ok(constant_result()); }\n    Reflect::set(",
            ),
        ),
    ] {
        assert!(
            hostile_lib != VALID_LIB || hostile_web != VALID_WEB,
            "selection evidence mutant {label} did not change its fixture",
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                assert_selection_evidence_authority_source(&hostile_lib, &hostile_web);
            }))
            .is_err(),
            "selection evidence scanner accepted {label}",
        );
    }
}

fn compiler_capabilities() -> VisionQkvCompilerCapabilities {
    VisionQkvCompilerCapabilities {
        min_storage_buffer_offset_alignment: 32,
        max_storage_buffers_per_shader_stage: 8,
        max_storage_buffer_binding_size: 1_u64 << 34,
        max_buffer_size: 1_u64 << 34,
        max_compute_workgroup_size: [8, 8, 1],
        max_compute_invocations_per_workgroup: 64,
        max_compute_workgroups_per_dimension: 65_535,
        max_host_elements: u64::from(u32::MAX),
    }
}

fn compiler_capabilities_with_alignment(alignment: u32) -> VisionQkvCompilerCapabilities {
    VisionQkvCompilerCapabilities {
        min_storage_buffer_offset_alignment: alignment,
        ..compiler_capabilities()
    }
}

fn synthetic_manifest_at_depth(depth: u32) -> Vec<u8> {
    assert!(matches!(depth, 1 | 3 | 16));
    if depth == 3 {
        return SYNTHETIC_MANIFEST_BYTES.to_vec();
    }
    canonical_manifest_mutant(SYNTHETIC_MANIFEST_BYTES, |manifest| {
        manifest.case_id = format!("m3-vision-stack-sharded-depth-{depth}");
        manifest.layer_count = depth;
        manifest.checkpoint_layers = if depth == 1 {
            vec![0]
        } else {
            vec![0, depth - 1]
        };
        let input = manifest.shards[0].clone();
        let layer_template = manifest.shards[1].clone();
        let post_norm = manifest.shards.last().unwrap().clone();
        manifest.shards = Vec::with_capacity(usize::try_from(depth).unwrap() + 2);
        manifest.shards.push(input);
        for layer in 0..depth {
            let mut shard = layer_template.clone();
            shard.id = format!("weights.vision_layer.{layer:02}");
            shard.layer_index = Some(layer);
            shard.blake3 = format!("{:064x}", u64::from(layer) + 1);
            manifest.shards.push(shard);
        }
        manifest.shards.push(post_norm);
    })
}

fn physical_spec_fixture(
    depth: u32,
    alignment: u32,
    semantic_readback_bytes: u64,
    scratch_canary_readback_bytes: u64,
) -> VisionQkvPhysicalExecutionSpec {
    physical_spec_fixture_for_policy(
        depth,
        alignment,
        semantic_readback_bytes,
        scratch_canary_readback_bytes,
        VisionQkvExecutionPolicy::Required,
    )
}

fn physical_spec_fixture_for_policy(
    depth: u32,
    alignment: u32,
    semantic_readback_bytes: u64,
    scratch_canary_readback_bytes: u64,
    policy: VisionQkvExecutionPolicy,
) -> VisionQkvPhysicalExecutionSpec {
    let manifest = synthetic_manifest_at_depth(depth);
    let capabilities = compiler_capabilities_with_alignment(alignment);
    let handoff = compile_vision_qkv_stack_handoff(&manifest, policy, capabilities)
        .expect("physical-command fixture must compile");
    prepare_vision_qkv_stack_handoff_execution(
        &handoff,
        &manifest,
        capabilities,
        VisionQkvCompilerReadbackRequest {
            semantic_readback_bytes,
            scratch_canary_readback_bytes,
        },
    )
    .expect("physical-command fixture must bind one sealed spec")
}

fn compiler_readback_request() -> VisionQkvCompilerReadbackRequest {
    VisionQkvCompilerReadbackRequest {
        semantic_readback_bytes: 16,
        scratch_canary_readback_bytes: 8,
    }
}

fn canonical_manifest_mutant(
    source: &[u8],
    mutate: impl FnOnce(&mut VisionStackShardManifest),
) -> Vec<u8> {
    let mut manifest = parse_vision_stack_shard_manifest(source)
        .expect("reviewed manifest fixture must be canonical");
    mutate(&mut manifest);
    canonical_vision_stack_shard_manifest_bytes(&manifest)
        .expect("hostile manifest must remain independently canonical")
}

// Caller-owned literal identities for the exact manifest geometry (including tokens,
// hidden width, sequence boundaries, and epsilon): this Web handoff oracle must not
// invoke the passes catalog or overlay builders that produce the values being checked.
const INDEPENDENT_SYNTHETIC_ALIGN32_PLAN_IR: [&str; 16] = [
    "1e60e96f9def5561bb200f93074a717fb168252bc80c02061edd17db28edef4a",
    "e0553a2c5682843eb327c22b730f53b69fea79cbcd0a857ea0346f26ed0ed543",
    "a9431a2ab9cdcfe1fe4ee9632344f17cb1c990a924db01c79d8baf9b2e03062e",
    "df0a78c2b89e5f750cd3daf720832d067fd4aced06be72d6d336578cf0fb95bc",
    "16209515595417a150cc50bee5cbe26cd3fccfd38a4ee462dd81f7512c4604d9",
    "4bdd6bccb2a25c80890e08e3f92f9b7b66ae98954dc969f77ba75d88ca5dd788",
    "f7f90573103a9436f4c5053605247ad8540bb40556b45c8de95781bacae9ad2b",
    "d13d1b9419e291b6814411a0475a90626cd911863dd9f52ebea920f8d622ef60",
    "993c86c94b6c79b177780409c1aa0f47c4b938333901a848159544f355bfd75c",
    "0f22cf45cd03034f336617e8b9a537eb3caabb36ca57ddb5d5b13ea3fc5152bc",
    "b7d51498222180dac9bc3065acb012cacd2bf15d7e182ef128f7be0b151275a3",
    "4912c0d78ff400888791d3bfc167ab04d6e1cd0c9bc80ed376106c0ce5187ded",
    "5abe10d7850ca24e738f88f355a266ccfd4f3829fd86bd9fa62b4959c71f131c",
    "ff976cf8ffe1d46e86199a2f18789fe6d7864a131591dfce80761c3412cd03a2",
    "81578909da40adb8a17fd790a35a394ea04af7c3ecea4bd66a284db73dd49cfe",
    "cec1e94969603c3fd8968d58b281578b595ee110feb068f12fc0b4381ddcc8c6",
];
const INDEPENDENT_SYNTHETIC_ALIGN256_PLAN_IR: [&str; 16] = [
    "e6bf9230a8cfba83d37dde43f76be3c35f95fdbc20ad0ee44abd45bdcfa30947",
    "3afa79f1c07a1292ea04d8e80c5ddd25af10e95ec8c9fbc58caf83c7d8da2387",
    "4b1385068924b1062c74d1019a036102e2d05de0fa9ae7b0bb427969ceac016a",
    "fb92abeffed258644c1cd51c2efb74bbce0456e89b08f771c784a7dd4b609a2f",
    "f7aa911f66a504734d3a3bfcdc64938dd96334ed95386f24d430fa90283bd83f",
    "f95d1710158db8c191d6414ba36a87bf99e2f7adb4d51b3f913f21e4ed56cb4b",
    "5575a2c2d0400eba08030beb53090826d5212540181187025193acbfbd221e1a",
    "004f71bc43d7f129244be4985149737867d7b9eeee48e8d1869460acdf42ab17",
    "246aeb09d0c75f01330dc77028793626cae8e684c591f7ba740e558e643af95e",
    "040b6e9433aed9cb13b0b8b98a302c5cd00fbc0b7d17a9c584344062868327b8",
    "8ffdd91cae961d72bc393f2379ddd3cbb71e0cfca620380edb53cb21653c98e2",
    "ed8e4c0171e61330ac5ea9a89885602dc854b12dbef89e3f4be456df3a3599a2",
    "ede8a2504baa4c56b83c68d1a0835e6c423f488322c81d6d69e2c9d57f647243",
    "8e0761998084b950a0abca2de13e2bae23c4f9e77ff9a63944ef22675b83723c",
    "e01b4b34422718f785fa4d6936e117b1dd6fb67bf1a3ee75a9129e5470f68871",
    "8f2389d84030402378ba29b4f4b83e54258f6d8d3942fc0b462278fe6a3cf66b",
];
const INDEPENDENT_OFFICIAL_ALIGN32_PLAN_IR: [&str; 27] = [
    "5a89ee3b0f288ba7bd0792257c1c2f822bbe8b532ee22238094835a1b04988eb",
    "9838f7dd49acecaaffc35bfae8780bcc34fe0ed5121b96093b53ececdb2e71ee",
    "71190959323001523ac2e8f39ccdc75335ee03d1aade2c16d67864c6d7d19ecf",
    "c4d74f69c4828d9348cc3ba4d5c31db7ece5cb5569c43bde56d254a06e45dd6d",
    "e81f7d2bf346fb3aa33d7d054e6d6ce106b42b288f74ab13c85c398e6b9dfe98",
    "63c3c1b85606791f5e971a6bada3ab21adfafb8d2fd6e6ef5152f7dc8e73bc3f",
    "f8e9ca738ea40a4df8c3d1e35f61bd2c0e0cc5b75c0095fc009dcdbb08a1c809",
    "df7233ca73d1d883e2124af1e96ce4d8706c511dc4d8c1e932f489d356374866",
    "514a176f326fd358a7f71f3889552c8ed3dfb79c0b2dece0c6cfd45cd9cb3343",
    "d92e3ce12ed6f1b1413ac5cdebcc1df7ccff7162be30eff39eb8b8e4fe1f1514",
    "678f911ee1acda92d6ebb20508314ea6265f4f95f2080601c7ac774abccadfcf",
    "26a98d65525751b532b623771f7cb1433c4f43f2fd7e369ef89b1edac7916604",
    "ce0dd82212cfa979a5a1b32861e44d768964473976a51eee22d9b96fb41059d7",
    "78a783b7ae70389d95c783e7b66a6256a0d82c5ef80d979bef592753f614cbb1",
    "a19bea11b6faf80076d201576a2c666a5d6e132310b70e99ebfc0a0df29ef401",
    "858f7da77b62934bde31a06f488311c8c57ca81ceb7aaeae064d7c4699987225",
    "ce57003d70351069f7bfec049bcc38f747fa6e395bbe179a7c8ce1c431696c60",
    "b63cb28e775c4d177bfc6b46956a637e593d39a4defe89ac3956ae46f18e0484",
    "0549e96f67468356f2ff8dc8486a83389662d25e138e184b0f055f6d4a06350e",
    "8f0ab42bf8881894c0647c9b1b39a9d8e5b855d245c67f1cfbfb35f21967d3c9",
    "d17a6855ae371c0de369f15b5b9ebb87f3d7194344ae6c2a344df85af7061a97",
    "d1d2c7190df924c76e0a5b0f41e3b0d84ba84a79769a9ec348b230336263a616",
    "d8976f0ec7d40dcff89803ace2367d280f67388e5b9171afd1328975cd0c7404",
    "a071f0d24b44581e74212d49fbf35405122e6344488d9c3f2decf6876d1e4d15",
    "4cb53d86bc0083fa009fdb19b2457fc2527fb8c5a980be0fb0a8e519ead5e9e5",
    "bcebeccfe3b61f4b41b0c513abe436e03b16389909673364dad2cf218bce05df",
    "80ca53cdb7ae72231e8c8cb6d002fc048684c12881b9f2f5f8f82d04617e5e47",
];
const INDEPENDENT_OFFICIAL_ALIGN256_PLAN_IR: [&str; 27] = [
    "41490b9f70bf3a101b8d2e77399fbe202aa0b1f249a5fd9aaaf1065a6b5c4820",
    "9bd16ba1019f1e40069bfbc02c31ccd82ab09b074ceaa01d2db2420657ceda14",
    "038778aaa5c27e534e7fe8cc2c27108ce16815f98cee8411563bf45f9bce594c",
    "a7149ce0f38885ed83fc255ec20ffdba732a2a1b51a98f769a2edd3c8628555a",
    "2dcab41ca2d5bbb3b24595b6f1682af8082856cedb7bb705638962260c658075",
    "bf5dcc67e855ea763b32275a423fc31b733c182f1886b99b4ab43310e9f2796c",
    "1aed6a50cb780a5b60ecbf3cc054e2fb631cd41b147631b3edca7660171799b4",
    "397c2fd42370f2f2845988fed36e587ff03fcccfebf6c1954a3a9ae6b4471a68",
    "e63291f05d3a303f4fe9042bf1de2343aacca6e74615c44bae518534a9fb98f1",
    "59e5596c6bad8f39f4332aac3581dd6ff028bc3f2e00f62ceb67b181b9893843",
    "b2f57593213acfaaca053bc73e22830be1d1e141e29aa53a945dc5f921d7f5f5",
    "7711740ac640d9c5208a3a73983676560dbc10f7a2a97d5deb85615714d4c783",
    "e444bec898d016857f5412b443220662de54eb6c5a9b23c3ba8b3d2aa61ee583",
    "a7763642f544ff12af51aa2dd22e11aaabf6d2281012a20f845b1c947fbd7685",
    "0928f417a965342d0d72e39a932575aabb3bfd50605765b0341aa271d2125ba8",
    "d3ecda757b391320b0b512dbcba259c27faac71a27787bffe1b16f7d52a8d7f7",
    "159c92a34975feea8f3704aed22ebfe0dff18b44f29d2135bff7077a3a46be3e",
    "bcb276828aca8ec0d1200c11a206f89e24a8016d1a061a727f6485046bdda409",
    "fef9cc6c9e9b43f5b2c6d35ea59481a74d40fc527eac725be103cb10f757b611",
    "344f584c4ebb540ebda7a4c1b6150ef4e3ac36aed9ad985d7b50cdf29d603fcd",
    "e2608b1ca12bbf3282c1483acf6c8d7a061f1e82f34d13c8d57efd7038bc3eda",
    "257b21b4ab16e5b7c57985e2ea922610b943f80fed990f5eec620f503c4a21c1",
    "b4af8a95ea8a760ee1c7b9e5bf0720c8aba0401d31e584351cf1b1bc4533678d",
    "36a8dd14de61cf50f5ed5b00c49dbfe846aa74af506be1ffa8aa3d5ae96b3d3b",
    "d9397ee18b2b12788ce05243867562257f0971b1f4f789b506758c7fa28b82cb",
    "cdc138018fa095b4400851e4af0a8e9872831e7e6173d643bcc2b192e9809d0d",
    "5753b67445789cee887ba841fd49d178f47611d8ff58b9363952673afc9fbd41",
];
struct IndependentHandoffOracle {
    manifest: VisionStackShardManifest,
    semantic_graph_blake3_hex: String,
    target_limits: VisionQkvFusedTargetLimits,
    tensor_catalog_len: usize,
    layer_plan_blake3: Vec<String>,
}

fn independent_handoff_oracle(
    manifest_bytes: &[u8],
    capabilities: VisionQkvCompilerCapabilities,
) -> IndependentHandoffOracle {
    let manifest = parse_vision_stack_shard_manifest(manifest_bytes)
        .expect("independent handoff oracle requires canonical manifest bytes");
    let depth = usize::try_from(manifest.layer_count).expect("manifest depth must fit usize");
    let (literal_plan_ir, tensor_catalog_len) = match (
        manifest.oracle,
        capabilities.min_storage_buffer_offset_alignment,
    ) {
        (VisionStackShardOracle::Synthetic, 32) if matches!(depth, 1 | 3 | 16) => {
            (&INDEPENDENT_SYNTHETIC_ALIGN32_PLAN_IR[..depth], depth * 6)
        }
        (VisionStackShardOracle::Synthetic, 256) if matches!(depth, 1 | 3 | 16) => {
            (&INDEPENDENT_SYNTHETIC_ALIGN256_PLAN_IR[..depth], depth * 6)
        }
        (VisionStackShardOracle::OfficialMpsBf16, 32) if depth == 27 => {
            (&INDEPENDENT_OFFICIAL_ALIGN32_PLAN_IR[..], 620)
        }
        (VisionStackShardOracle::OfficialMpsBf16, 256) if depth == 27 => {
            (&INDEPENDENT_OFFICIAL_ALIGN256_PLAN_IR[..], 620)
        }
        _ => panic!(
            "independent literal PlanIR oracle has no {:?}/depth {depth}/alignment {} fixture",
            manifest.oracle, capabilities.min_storage_buffer_offset_alignment,
        ),
    };
    let target_limits = VisionQkvFusedTargetLimits {
        min_storage_buffer_offset_alignment: capabilities.min_storage_buffer_offset_alignment,
        max_storage_buffers_per_shader_stage: capabilities.max_storage_buffers_per_shader_stage,
        max_storage_buffer_binding_size: capabilities.max_storage_buffer_binding_size,
        max_buffer_size: capabilities.max_buffer_size,
        max_compute_workgroups_per_dimension: capabilities.max_compute_workgroups_per_dimension,
    };
    IndependentHandoffOracle {
        manifest,
        semantic_graph_blake3_hex:
            "2b2556c363545dcef569e3e6d0db01967973a081706c8483e1c5af3c7dc5bf73".to_owned(),
        target_limits,
        tensor_catalog_len,
        layer_plan_blake3: literal_plan_ir
            .iter()
            .map(|identity| (*identity).to_owned())
            .collect(),
    }
}

fn assert_real_handoff_matches_independent_oracle(
    label: &str,
    manifest_bytes: &[u8],
    alignment: u32,
) {
    let capabilities = compiler_capabilities_with_alignment(alignment);
    let oracle = independent_handoff_oracle(manifest_bytes, capabilities);
    let first = compile_vision_qkv_stack_handoff(
        manifest_bytes,
        VisionQkvExecutionPolicy::Required,
        capabilities,
    )
    .unwrap_or_else(|error| panic!("{label}: first real handoff failed: {error}"));
    let second = compile_vision_qkv_stack_handoff(
        manifest_bytes,
        VisionQkvExecutionPolicy::Required,
        capabilities,
    )
    .unwrap_or_else(|error| panic!("{label}: repeated real handoff failed: {error}"));

    for handoff in [&first, &second] {
        assert_eq!(
            handoff.selection().outcome(),
            VisionQkvSelectionOutcome::Fused,
            "{label}: supported Required handoff did not fuse",
        );
        assert_eq!(
            u64::try_from(handoff.layer_count()).unwrap(),
            u64::from(oracle.manifest.layer_count),
            "{label}",
        );
        assert_eq!(
            handoff.canonical_manifest_blake3_hex(),
            blake3::hash(manifest_bytes).to_hex().as_str(),
            "{label}: canonical manifest identity drifted",
        );
        assert_eq!(
            handoff.semantic_graph_blake3_hex(),
            oracle.semantic_graph_blake3_hex,
            "{label}: semantic graph identity drifted",
        );
        assert_eq!(handoff.target_limits(), oracle.target_limits, "{label}");
        assert_eq!(
            handoff.tensor_catalog_len(),
            oracle.tensor_catalog_len,
            "{label}: oracle catalog branch drifted",
        );
        let actual_layer_plan = handoff
            .selection()
            .overlay()
            .expect("supported Required handoff must retain the sealed overlay")
            .layers()
            .iter()
            .map(|layer| layer.canonical_plan_blake3_hex())
            .collect::<Vec<_>>();
        assert_eq!(
            actual_layer_plan,
            oracle
                .layer_plan_blake3
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            "{label}: Web handoff changed the independently prepared ordered PlanIR identities",
        );
    }

    let first_json = build_vision_qkv_selection_evidence_propagation(&first)
        .evidence_json()
        .unwrap_or_else(|error| panic!("{label}: first selection evidence failed: {error}"));
    let second_json = build_vision_qkv_selection_evidence_propagation(&second)
        .evidence_json()
        .unwrap_or_else(|error| panic!("{label}: repeated selection evidence failed: {error}"));
    assert_eq!(
        first_json, second_json,
        "{label}: repeated opaque handoff construction was nondeterministic",
    );
    let evidence: serde_json::Value = serde_json::from_str(&first_json)
        .unwrap_or_else(|error| panic!("{label}: invalid selection evidence JSON: {error}"));
    assert_eq!(
        evidence
            .as_object()
            .expect("selection evidence must be an object")
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "fallback_class",
            "layer_plan_blake3",
            "manifest_blake3",
            "manifest_geometry",
            "outcome",
            "policy",
            "semantic_graph_blake3",
            "target_limits",
            "tensor_catalog_len",
        ]),
        "{label}: selection evidence schema drifted",
    );
    assert_eq!(evidence["policy"], "required", "{label}");
    assert_eq!(evidence["outcome"], "fused", "{label}");
    assert!(evidence["fallback_class"].is_null(), "{label}");
    assert_eq!(
        evidence["manifest_blake3"],
        blake3::hash(manifest_bytes).to_hex().as_str(),
        "{label}",
    );
    assert_eq!(
        evidence["semantic_graph_blake3"], oracle.semantic_graph_blake3_hex,
        "{label}",
    );
    assert_eq!(
        evidence["manifest_geometry"],
        serde_json::json!({
            "tokens": oracle.manifest.tokens,
            "hidden_size": oracle.manifest.hidden_size,
            "attention_heads": oracle.manifest.attention_heads,
            "head_dim": oracle.manifest.head_dim,
            "intermediate_size": oracle.manifest.intermediate_size,
            "layer_count": oracle.manifest.layer_count,
        }),
        "{label}",
    );
    assert_eq!(
        evidence["target_limits"],
        serde_json::json!({
            "min_storage_buffer_offset_alignment": oracle.target_limits.min_storage_buffer_offset_alignment,
            "max_storage_buffers_per_shader_stage": oracle.target_limits.max_storage_buffers_per_shader_stage,
            "max_storage_buffer_binding_size": oracle.target_limits.max_storage_buffer_binding_size,
            "max_buffer_size": oracle.target_limits.max_buffer_size,
            "max_compute_workgroups_per_dimension": oracle.target_limits.max_compute_workgroups_per_dimension,
        }),
        "{label}",
    );
    assert_eq!(
        evidence["tensor_catalog_len"], oracle.tensor_catalog_len,
        "{label}",
    );
    assert_eq!(
        evidence["layer_plan_blake3"],
        serde_json::json!(oracle.layer_plan_blake3),
        "{label}: evidence did not retain the exact ordered independent identities",
    );
}

fn assert_compiler_handoff_error<T: std::fmt::Debug>(
    label: &str,
    expected: VisionQkvCompilerHandoffErrorCode,
    result: Result<T, pvlc_runtime_web::VisionQkvCompilerHandoffError>,
) {
    let error = result.unwrap_err();
    assert_eq!(error.code(), expected, "{label}: {error}");
}

#[test]
fn real_opaque_web_handoff_matrix_is_deterministic_and_exact_at_deep_and_official_shapes() {
    for depth in [1_u32, 3, 16] {
        let manifest = synthetic_manifest_at_depth(depth);
        for alignment in [32_u32, 256] {
            assert_real_handoff_matches_independent_oracle(
                &format!("synthetic depth {depth} alignment {alignment}"),
                &manifest,
                alignment,
            );
        }
    }
    for alignment in [32_u32, 256] {
        assert_real_handoff_matches_independent_oracle(
            &format!("official depth 27 alignment {alignment}"),
            OFFICIAL_MANIFEST_BYTES,
            alignment,
        );
    }
}

#[test]
fn preferred_never_falls_back_for_canonical_structural_or_semantic_manifest_failures() {
    fn unchecked_manifest_bytes(manifest: &VisionStackShardManifest) -> Vec<u8> {
        let mut bytes = serde_json::to_vec(manifest).expect("hostile manifest JSON");
        bytes.push(b'\n');
        bytes
    }

    let mut noncanonical = SYNTHETIC_MANIFEST_BYTES.to_vec();
    noncanonical.push(b' ');

    let mut structural_manifest =
        parse_vision_stack_shard_manifest(SYNTHETIC_MANIFEST_BYTES).unwrap();
    structural_manifest.shards.remove(2);
    let structural = unchecked_manifest_bytes(&structural_manifest);

    let mut semantic_manifest =
        parse_vision_stack_shard_manifest(SYNTHETIC_MANIFEST_BYTES).unwrap();
    semantic_manifest.model_id = "hostile-semantic-model".to_owned();
    let semantic = unchecked_manifest_bytes(&semantic_manifest);

    for (label, manifest, expected) in [
        (
            "noncanonical encoding",
            noncanonical.as_slice(),
            VisionQkvCompilerHandoffErrorCode::NonCanonicalManifest,
        ),
        (
            "structural shard directory",
            structural.as_slice(),
            VisionQkvCompilerHandoffErrorCode::InvalidShardDirectory,
        ),
        (
            "semantic model identity",
            semantic.as_slice(),
            VisionQkvCompilerHandoffErrorCode::ModelIdentityMismatch,
        ),
    ] {
        let required = compile_vision_qkv_stack_handoff(
            manifest,
            VisionQkvExecutionPolicy::Required,
            compiler_capabilities(),
        )
        .unwrap_err();
        let preferred = compile_vision_qkv_stack_handoff(
            manifest,
            VisionQkvExecutionPolicy::Preferred,
            compiler_capabilities(),
        )
        .unwrap_err();
        assert_eq!(required.code(), expected, "{label}: Required error class");
        assert_eq!(preferred.code(), expected, "{label}: Preferred error class");
        assert_eq!(
            preferred.code(),
            required.code(),
            "{label}: Preferred changed a hard compiler failure into another class",
        );
    }
}

#[test]
fn host_compiler_handoff_routes_real_oracles_and_binds_exact_manifest_evidence() {
    let capabilities = compiler_capabilities();
    let synthetic = compile_vision_qkv_stack_handoff(
        SYNTHETIC_MANIFEST_BYTES,
        VisionQkvExecutionPolicy::Required,
        capabilities,
    )
    .expect("real synthetic manifest must compile through the host seam");
    assert_eq!(
        synthetic.selection().outcome(),
        VisionQkvSelectionOutcome::Fused
    );
    assert_eq!(synthetic.layer_count(), 3, "synthetic oracle/depth context");
    assert_eq!(
        synthetic.tensor_catalog_len(),
        18,
        "synthetic oracle/depth 3 must route the exact six-tensors-per-layer catalog",
    );
    assert_eq!(
        synthetic.canonical_manifest_blake3_hex(),
        blake3::hash(SYNTHETIC_MANIFEST_BYTES).to_hex().as_str(),
        "synthetic oracle/depth 3 manifest digest drifted",
    );
    assert_eq!(
        synthetic.semantic_graph_blake3_hex(),
        blake3::hash(
            &SemanticGraph::paddleocr_vl_16()
                .canonical_bytes()
                .expect("independent canonical semantic graph"),
        )
        .to_hex()
        .as_str(),
        "synthetic oracle/depth 3 semantic graph digest drifted",
    );
    assert_eq!(
        synthetic.target_limits(),
        VisionQkvFusedTargetLimits {
            min_storage_buffer_offset_alignment: 32,
            max_storage_buffers_per_shader_stage: 8,
            max_storage_buffer_binding_size: 1_u64 << 34,
            max_buffer_size: 1_u64 << 34,
            max_compute_workgroups_per_dimension: 65_535,
        },
        "synthetic oracle/depth 3 target handoff drifted",
    );
    let synthetic_geometry = synthetic.manifest_geometry();
    assert_eq!(
        (
            synthetic_geometry.tokens(),
            synthetic_geometry.hidden_size(),
            synthetic_geometry.attention_heads(),
            synthetic_geometry.head_dim(),
            synthetic_geometry.intermediate_size(),
            synthetic_geometry.layer_count(),
        ),
        (3, 4, 2, 2, 5, 3),
        "synthetic oracle/depth 3 full manifest geometry drifted",
    );
    let synthetic_overlay = synthetic
        .selection()
        .overlay()
        .expect("Required synthetic handoff must retain its verified overlay");
    assert_eq!(synthetic_overlay.layer_count(), 3);
    assert_eq!(
        synthetic_overlay.layers()[0].invocation().kernel,
        KernelId::VisionQkvFusedF32
    );
    assert_eq!(
        synthetic_overlay.layers()[0].invocation().workgroup_size,
        [8, 8, 1]
    );
    assert_eq!(
        synthetic_overlay.layers()[0].invocation().dispatch,
        [1, 1, 3]
    );
    assert_eq!(synthetic_overlay.layers()[0].uniform_words(), [3, 4, 4, 16]);

    let official = compile_vision_qkv_stack_handoff(
        OFFICIAL_MANIFEST_BYTES,
        VisionQkvExecutionPolicy::Required,
        capabilities,
    )
    .expect("real official manifest must compile through the official catalog branch");
    assert_eq!(
        official.selection().outcome(),
        VisionQkvSelectionOutcome::Fused
    );
    assert_eq!(official.layer_count(), 27, "official oracle/depth context");
    let official_geometry = official.manifest_geometry();
    assert_eq!(
        (
            official_geometry.tokens(),
            official_geometry.hidden_size(),
            official_geometry.attention_heads(),
            official_geometry.head_dim(),
            official_geometry.intermediate_size(),
            official_geometry.layer_count(),
        ),
        (1_276, 1_152, 16, 72, 4_304, 27),
        "official oracle/depth 27 full manifest geometry drifted",
    );
    assert_eq!(
        official.tensor_catalog_len(),
        PaddleOcrVl16Schema::tensor_specs().len(),
        "official oracle/depth 27 did not route the complete official schema catalog",
    );
    assert_ne!(
        official.tensor_catalog_len(),
        27 * 6,
        "official oracle silently used the shape-compatible synthetic catalog",
    );
    assert_eq!(
        official.canonical_manifest_blake3_hex(),
        blake3::hash(OFFICIAL_MANIFEST_BYTES).to_hex().as_str(),
        "official oracle/depth 27 manifest digest drifted",
    );
    let official_layer = &official
        .selection()
        .overlay()
        .expect("Required official handoff must retain its verified overlay")
        .layers()[26];
    assert_eq!(official_layer.layer_index(), 26);
    assert_eq!(official_layer.invocation().workgroup_size, [8, 8, 1]);
    assert_eq!(official_layer.invocation().dispatch, [144, 160, 3]);
    assert_eq!(
        official_layer.uniform_words(),
        [1_276, 1_152, 1_152, 1_469_952],
    );

    let disabled = compile_vision_qkv_stack_handoff(
        OFFICIAL_MANIFEST_BYTES,
        VisionQkvExecutionPolicy::Disabled,
        VisionQkvCompilerCapabilities {
            min_storage_buffer_offset_alignment: 0,
            max_storage_buffers_per_shader_stage: 0,
            max_storage_buffer_binding_size: 0,
            max_buffer_size: 0,
            max_compute_workgroup_size: [0; 3],
            max_compute_invocations_per_workgroup: 0,
            max_compute_workgroups_per_dimension: 0,
            max_host_elements: 0,
        },
    )
    .expect("Disabled policy must remain lazy even for unusable capabilities");
    assert_eq!(
        disabled.selection().outcome(),
        VisionQkvSelectionOutcome::Disabled
    );
    assert!(disabled.selection().overlay().is_none());
    assert_eq!(disabled.selection().fallback_error_code(), None);
    assert_eq!(disabled.tensor_catalog_len(), 0);
    assert!(
        disabled.semantic_graph_blake3_hex().is_empty(),
        "Disabled handoff must not fabricate a semantic graph identity without constructing it",
    );
}

#[test]
fn real_handoffs_serialize_one_borrowed_fused_or_explicit_null_channel_evidence() {
    let capabilities = VisionQkvCompilerCapabilities {
        max_storage_buffers_per_shader_stage: 7,
        ..compiler_capabilities()
    };
    let preferred = compile_vision_qkv_stack_handoff(
        SYNTHETIC_MANIFEST_BYTES,
        VisionQkvExecutionPolicy::Preferred,
        capabilities,
    )
    .expect("Preferred unsupported target must compile one explicit legacy fallback handoff");

    assert_eq!(
        preferred.selection().policy(),
        VisionQkvExecutionPolicy::Preferred,
    );
    assert_eq!(
        preferred.selection().outcome(),
        VisionQkvSelectionOutcome::FallbackUnsupportedTarget,
    );
    assert_eq!(
        preferred.selection().fallback_error_code(),
        Some(VisionQkvStackOverlayErrorCode::UnsupportedTarget),
    );
    assert!(
        preferred.selection().overlay().is_none(),
        "Preferred unsupported target must not retain a partial fused overlay",
    );
    assert_eq!(
        preferred.semantic_graph_blake3_hex(),
        blake3::hash(
            &SemanticGraph::paddleocr_vl_16()
                .canonical_bytes()
                .expect("independent canonical semantic graph"),
        )
        .to_hex()
        .as_str(),
        "Preferred must retain the semantic graph identity constructed before target fallback",
    );
    assert_eq!(
        preferred.canonical_manifest_blake3_hex(),
        blake3::hash(SYNTHETIC_MANIFEST_BYTES).to_hex().as_str(),
    );
    assert_eq!(
        preferred.tensor_catalog_len(),
        18,
        "Preferred fallback must report the exact catalog used by semantic compilation",
    );
    assert_eq!(
        preferred.target_limits(),
        VisionQkvFusedTargetLimits {
            min_storage_buffer_offset_alignment: 32,
            max_storage_buffers_per_shader_stage: 7,
            max_storage_buffer_binding_size: 1_u64 << 34,
            max_buffer_size: 1_u64 << 34,
            max_compute_workgroups_per_dimension: 65_535,
        },
        "Preferred fallback evidence must retain the rejected target exactly",
    );
    let plan_ir_identity_count = preferred
        .selection()
        .overlay()
        .map_or(0, |overlay| overlay.layers().len());
    assert_eq!(
        plan_ir_identity_count, 0,
        "unsupported fallback must expose an empty PlanIR identity sequence",
    );

    let opaque = build_vision_qkv_selection_evidence_propagation(&preferred);
    let begin_session = opaque.clone();
    let final_session = begin_session.clone();
    let begin_fallback = begin_session.additive_begin_evidence::<serde_json::Value>(None);
    let final_fallback = final_session.final_diagnostics_evidence::<serde_json::Value>(None);
    assert!(
        std::ptr::eq(
            opaque.opaque_selection_evidence(),
            begin_fallback.qkv_selection(),
        ),
        "additive begin reconstructed selection evidence instead of sharing its immutable authority",
    );
    assert!(
        std::ptr::eq(
            opaque.opaque_selection_evidence(),
            final_fallback.qkv_selection(),
        ),
        "final diagnostics reconstructed selection evidence instead of sharing its immutable authority",
    );
    assert!(begin_fallback.qkv_execution().is_none());
    assert!(final_fallback.qkv_execution().is_none());
    assert!(
        final_session.uses_legacy_topology(),
        "Preferred unsupported propagation selected an optimized topology",
    );

    let selection: serde_json::Value = serde_json::from_str(
        &opaque
            .evidence_json()
            .expect("production selection evidence serializer"),
    )
    .expect("production selection evidence must be valid JSON");
    assert_eq!(
        selection
            .as_object()
            .expect("selection evidence object")
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "fallback_class",
            "layer_plan_blake3",
            "manifest_blake3",
            "manifest_geometry",
            "outcome",
            "policy",
            "semantic_graph_blake3",
            "target_limits",
            "tensor_catalog_len",
        ]),
        "production selection evidence schema is not exact and closed",
    );
    assert_eq!(selection["policy"], "preferred");
    assert_eq!(selection["outcome"], "fallback_unsupported_target");
    assert_eq!(selection["fallback_class"], "unsupported_target");
    assert_eq!(
        selection["manifest_blake3"],
        blake3::hash(SYNTHETIC_MANIFEST_BYTES).to_hex().as_str(),
    );
    assert_eq!(
        selection["semantic_graph_blake3"],
        blake3::hash(
            &SemanticGraph::paddleocr_vl_16()
                .canonical_bytes()
                .expect("independent canonical semantic graph"),
        )
        .to_hex()
        .as_str(),
    );
    assert_eq!(
        selection["manifest_geometry"],
        serde_json::json!({
            "tokens": 3,
            "hidden_size": 4,
            "attention_heads": 2,
            "head_dim": 2,
            "intermediate_size": 5,
            "layer_count": 3,
        }),
    );
    assert_eq!(
        selection["target_limits"],
        serde_json::json!({
            "min_storage_buffer_offset_alignment": 32,
            "max_storage_buffers_per_shader_stage": 7,
            "max_storage_buffer_binding_size": 1_u64 << 34,
            "max_buffer_size": 1_u64 << 34,
            "max_compute_workgroups_per_dimension": 65_535,
        }),
    );
    assert_eq!(selection["tensor_catalog_len"], 18);
    assert_eq!(selection["layer_plan_blake3"], serde_json::json!([]));
    for forbidden in ["overlay", "workspace", "qkv_execution"] {
        assert!(
            selection.get(forbidden).is_none(),
            "selection-only evidence leaked {forbidden}",
        );
    }

    for (channel, envelope) in [
        (
            "additive begin",
            serde_json::to_value(&begin_fallback)
                .expect("production additive-begin borrowed evidence envelope"),
        ),
        (
            "final diagnostics",
            serde_json::to_value(&final_fallback)
                .expect("production final-diagnostics borrowed evidence envelope"),
        ),
    ] {
        assert_eq!(
            envelope
                .as_object()
                .expect("evidence envelope object")
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["qkv_execution", "qkv_selection"]),
            "{channel} evidence envelope schema drifted",
        );
        assert_eq!(
            envelope["qkv_selection"], selection,
            "{channel} did not propagate the exact selection value",
        );
        assert!(
            envelope["qkv_execution"].is_null(),
            "{channel} fallback must explicitly serialize qkv_execution: null",
        );
        assert!(
            envelope.get("workspace").is_none(),
            "{channel} fallback fabricated a workspace beside null execution",
        );
    }

    let required = compile_vision_qkv_stack_handoff(
        SYNTHETIC_MANIFEST_BYTES,
        VisionQkvExecutionPolicy::Required,
        VisionQkvCompilerCapabilities {
            max_storage_buffers_per_shader_stage: 16,
            ..capabilities
        },
    )
    .expect("Required fused selection evidence");
    let required_layer_plan_blake3 = required
        .selection()
        .overlay()
        .expect("Required evidence must retain its fused overlay")
        .layers()
        .iter()
        .map(|layer| layer.canonical_plan_blake3_hex())
        .collect::<Vec<_>>();
    let required_evidence = build_vision_qkv_selection_evidence_propagation(&required);
    let required_json: serde_json::Value = serde_json::from_str(
        &required_evidence
            .evidence_json()
            .expect("Required selection evidence serializer"),
    )
    .expect("Required selection JSON");
    assert_eq!(required_json["policy"], "required");
    assert_eq!(required_json["outcome"], "fused");
    assert!(required_json["fallback_class"].is_null());
    assert_eq!(required_json["tensor_catalog_len"], 18);
    assert_eq!(
        required_json["layer_plan_blake3"],
        serde_json::json!(required_layer_plan_blake3),
        "fused selection evidence reconstructed or omitted exact ordered PlanIR identities",
    );
    assert!(
        !required_evidence.uses_legacy_topology(),
        "Required fused propagation selected a legacy topology",
    );

    let begin_execution = serde_json::json!({
        "dispatch_count": 31,
        "command_buffer_count": 4,
        "submission_count": 4,
        "map_count": 1,
        "workspace": {
            "logical_id": "vision-stack-qkv-workspace",
            "allocation_bytes": 256,
            "semantic_base": 0,
            "semantic_bytes": 16,
        },
        "bindings": [{
            "binding": 0,
            "logical_buffer": "vision-stack-qkv-workspace",
            "byte_offset": 0,
            "byte_length": 16,
        }],
        "canaries": [{
            "name": "workspace-prefix",
            "byte_offset": 16,
            "byte_length": 8,
            "passed": null,
        }],
    });
    let mut final_execution = begin_execution.clone();
    final_execution["canaries"][0]["passed"] = serde_json::json!(true);
    let required_begin = required_evidence.additive_begin_evidence(Some(&begin_execution));
    let required_final = required_evidence.final_diagnostics_evidence(Some(&final_execution));
    assert!(std::ptr::eq(
        required_begin.qkv_selection(),
        required_evidence.opaque_selection_evidence(),
    ));
    assert!(std::ptr::eq(
        required_final.qkv_selection(),
        required_evidence.opaque_selection_evidence(),
    ));
    assert!(std::ptr::eq(
        required_begin.qkv_execution().unwrap(),
        &begin_execution,
    ));
    assert!(std::ptr::eq(
        required_final.qkv_execution().unwrap(),
        &final_execution,
    ));

    let begin_envelope =
        serde_json::to_value(&required_begin).expect("Required begin evidence envelope");
    let final_envelope =
        serde_json::to_value(&required_final).expect("Required final evidence envelope");
    for (channel, envelope, expected_execution) in [
        ("begin", &begin_envelope, &begin_execution),
        ("final", &final_envelope, &final_execution),
    ] {
        assert_eq!(
            envelope
                .as_object()
                .expect("Required evidence envelope")
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["qkv_execution", "qkv_selection"]),
            "Required {channel} envelope schema drifted",
        );
        assert_eq!(envelope["qkv_selection"], required_json);
        assert_eq!(&envelope["qkv_execution"], expected_execution);
    }
    assert!(begin_envelope["qkv_execution"]["canaries"][0]["passed"].is_null());
    assert_eq!(
        final_envelope["qkv_execution"]["canaries"][0]["passed"],
        true,
    );
    let strip_results = |execution: &serde_json::Value| {
        let mut plan = execution.clone();
        for canary in plan["canaries"].as_array_mut().unwrap() {
            canary.as_object_mut().unwrap().remove("passed");
        }
        plan
    };
    assert_eq!(
        strip_results(&begin_envelope["qkv_execution"]),
        strip_results(&final_envelope["qkv_execution"]),
        "begin and final evidence must derive from one exact immutable execution plan",
    );

    let disabled = compile_vision_qkv_stack_handoff(
        SYNTHETIC_MANIFEST_BYTES,
        VisionQkvExecutionPolicy::Disabled,
        capabilities,
    )
    .expect("Disabled selection evidence");
    let disabled_evidence = build_vision_qkv_selection_evidence_propagation(&disabled);
    let disabled_json: serde_json::Value = serde_json::from_str(
        &disabled_evidence
            .evidence_json()
            .expect("Disabled selection evidence serializer"),
    )
    .expect("Disabled selection JSON");
    assert!(disabled_json["semantic_graph_blake3"].is_null());
    assert_eq!(disabled_json["tensor_catalog_len"], 0);
    assert_eq!(disabled_json["layer_plan_blake3"], serde_json::json!([]));
    assert!(disabled_evidence.uses_legacy_topology());
    for envelope in [
        disabled_evidence.additive_begin_evidence::<serde_json::Value>(None),
        disabled_evidence.final_diagnostics_evidence::<serde_json::Value>(None),
    ] {
        let envelope = serde_json::to_value(&envelope).expect("Disabled borrowed envelope");
        assert_eq!(envelope["qkv_selection"], disabled_json);
        assert!(
            envelope
                .as_object()
                .expect("Disabled evidence envelope")
                .contains_key("qkv_execution"),
            "Disabled envelope omitted explicit qkv_execution null",
        );
        assert!(envelope["qkv_execution"].is_null());
    }
}

#[test]
fn production_execution_evidence_builders_use_real_spec_one_plan_and_mixed_canary_results() {
    let exact_keys = |value: &serde_json::Value, expected: &[&str], label: &str| {
        assert_eq!(
            value
                .as_object()
                .unwrap_or_else(|| panic!("{label} must be an object"))
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            expected.iter().copied().collect::<BTreeSet<_>>(),
            "{label} schema drifted",
        );
    };
    let strip_passed = |execution: &serde_json::Value| {
        let mut plan = execution.clone();
        for canary in plan["canaries"].as_array_mut().unwrap() {
            canary.as_object_mut().unwrap().remove("passed");
        }
        plan
    };

    for policy in [
        VisionQkvExecutionPolicy::Required,
        VisionQkvExecutionPolicy::Preferred,
    ] {
        for (depth, alignment) in [(1_u32, 32_u32), (3, 256)] {
            let manifest = synthetic_manifest_at_depth(depth);
            let capabilities = compiler_capabilities_with_alignment(alignment);
            let handoff = compile_vision_qkv_stack_handoff(&manifest, policy, capabilities)
                .expect("supported Required/Preferred handoff");
            assert_eq!(
                handoff.selection().outcome(),
                VisionQkvSelectionOutcome::Fused
            );
            let physical = physical_spec_fixture_for_policy(depth, alignment, 16, 8, policy);
            let prepared = physical.prepared_execution();
            let workspace = prepared.workspace();
            let plan = BrowserVisionQkvExecutionEvidencePlan::from_prepared(Some(&physical))
                .expect("real physical spec must build execution evidence")
                .expect("fused physical spec must produce a populated plan");
            let begin = BrowserVisionQkvBeginExecutionEvidence::from_plan(Some(&plan))
                .expect("fused plan must produce begin evidence");
            let mixed_results = (0..workspace.canaries().len())
                .map(|index| index % 2 == 0)
                .collect::<Vec<_>>();
            assert!(
                mixed_results.contains(&true) && mixed_results.contains(&false),
                "fixture must contain mixed final canary results",
            );
            let final_evidence = BrowserVisionQkvFinalExecutionEvidence::from_verified_plan(
                Some(&plan),
                &mixed_results,
            )
            .expect("mixed verified canaries must serialize")
            .expect("fused plan must produce final evidence");

            let selection = build_vision_qkv_selection_evidence_propagation(&handoff);
            let legacy_manifest = parse_vision_stack_shard_manifest(&manifest).unwrap();
            let legacy_plan = legacy_manifest.plan().unwrap();
            let legacy_status = build_vision_stack_legacy_status_record(
                VisionStackShardProtocolPhase::Preflight,
                legacy_manifest
                    .shards
                    .first()
                    .map(|shard| shard.id.as_str()),
                &legacy_plan,
                VisionStackActivationStrategy::SeparateBuffers,
                None,
                None,
                alignment,
                true,
            )
            .unwrap();
            let shader_blake3 = BTreeMap::from([(KernelId::LayerNormF32, "cd".repeat(32))]);
            let legacy_diagnostics = build_vision_stack_legacy_diagnostics_record(
                &legacy_plan,
                VisionStackActivationStrategy::SeparateBuffers,
                None,
                None,
                alignment,
                &shader_blake3,
                1,
                17,
                16 * depth + 2,
            )
            .unwrap();
            let begin_envelope = selection.additive_begin_evidence(Some(&begin));
            let final_envelope = selection.final_diagnostics_evidence(Some(&final_evidence));
            let begin_json: serde_json::Value = serde_json::from_str(
                &serialize_vision_stack_qkv_begin_status_json(&legacy_status, begin_envelope)
                    .expect("actual production begin-status serializer"),
            )
            .unwrap();
            let final_json: serde_json::Value = serde_json::from_str(
                &serialize_vision_stack_qkv_final_diagnostics_json(
                    &legacy_diagnostics,
                    final_envelope,
                )
                .expect("actual production final-diagnostics serializer"),
            )
            .unwrap();
            let legacy_begin_value = serde_json::from_str::<serde_json::Value>(
                &serialize_vision_stack_legacy_status_json(&legacy_status).unwrap(),
            )
            .unwrap();
            let mut expected_begin_keys = legacy_begin_value
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            expected_begin_keys.extend(["qkv_selection", "qkv_execution"]);
            assert_eq!(
                begin_json
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>(),
                expected_begin_keys,
                "actual QKV begin serializer did not add exactly two evidence fields",
            );
            let legacy_final_value = serde_json::from_str::<serde_json::Value>(
                &serialize_vision_stack_legacy_diagnostics_json(&legacy_diagnostics).unwrap(),
            )
            .unwrap();
            let mut expected_final_keys = legacy_final_value
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            expected_final_keys.extend(["qkv_selection", "qkv_execution"]);
            assert_eq!(
                final_json
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>(),
                expected_final_keys,
                "actual QKV final serializer did not add exactly two evidence fields",
            );
            for (channel, envelope) in [("begin", &begin_json), ("final", &final_json)] {
                exact_keys(
                    &envelope["qkv_execution"],
                    &[
                        "dispatch_count",
                        "command_buffer_count",
                        "submission_count",
                        "map_count",
                        "workspace",
                        "bindings",
                        "canaries",
                    ],
                    &format!("{policy:?}/{depth}/{alignment} {channel} execution"),
                );
            }
            assert_eq!(
                begin_json["qkv_selection"], final_json["qkv_selection"],
                "begin/final selection allocation serialized different values",
            );
            let begin_execution = &begin_json["qkv_execution"];
            let final_execution = &final_json["qkv_execution"];
            assert_eq!(
                strip_passed(begin_execution),
                strip_passed(final_execution),
                "begin and final reconstructed divergent non-result plans",
            );
            assert_eq!(final_execution["dispatch_count"], 10 * depth + 1);
            assert_eq!(final_execution["command_buffer_count"], depth + 1);
            assert_eq!(final_execution["submission_count"], depth + 1);
            assert_eq!(final_execution["map_count"], 1);
            assert_eq!(
                final_execution["workspace"],
                serde_json::json!({
                    "logical_id": "vision-stack-qkv-workspace",
                    "allocation_bytes": workspace.allocation_bytes(),
                    "semantic_base": workspace.semantic_base(),
                    "semantic_bytes": workspace.semantic_bytes(),
                }),
            );

            let expected_bindings = prepared.layers()[0]
                .attention_bridge()
                .bindings()
                .iter()
                .map(|binding| {
                    serde_json::json!({
                        "binding": binding.binding(),
                        "byte_offset": binding.byte_offset(),
                        "byte_length": binding.byte_length(),
                    })
                })
                .collect::<Vec<_>>();
            assert_eq!(
                final_execution["bindings"],
                serde_json::Value::Array(expected_bindings),
            );

            let begin_canaries = begin_execution["canaries"].as_array().unwrap();
            let final_canaries = final_execution["canaries"].as_array().unwrap();
            assert_eq!(begin_canaries.len(), workspace.canaries().len());
            assert_eq!(final_canaries.len(), workspace.canaries().len());
            for (index, ((prepared_canary, begin_canary), final_canary)) in workspace
                .canaries()
                .iter()
                .zip(begin_canaries)
                .zip(final_canaries)
                .enumerate()
            {
                let (kind, plane) = match prepared_canary.kind() {
                    VisionQkvCanaryKind::Prefix => ("prefix", serde_json::Value::Null),
                    VisionQkvCanaryKind::InternalPadding { plane } => {
                        ("internal_padding", serde_json::json!(plane))
                    }
                    VisionQkvCanaryKind::Suffix => ("suffix", serde_json::Value::Null),
                };
                let expected_common = serde_json::json!({
                    "kind": kind,
                    "plane": plane,
                    "byte_offset": prepared_canary.byte_offset(),
                    "byte_length": prepared_canary.byte_length(),
                });
                for (channel, actual) in [("begin", begin_canary), ("final", final_canary)] {
                    exact_keys(
                        actual,
                        &["kind", "plane", "byte_offset", "byte_length", "passed"],
                        &format!("{channel} canary {index}"),
                    );
                    let mut without_result = actual.clone();
                    without_result.as_object_mut().unwrap().remove("passed");
                    assert_eq!(without_result, expected_common);
                }
                assert!(begin_canary["passed"].is_null());
                assert_eq!(final_canary["passed"], mixed_results[index]);
            }
            assert!(
                BrowserVisionQkvFinalExecutionEvidence::from_verified_plan(
                    Some(&plan),
                    &mixed_results[..mixed_results.len() - 1],
                )
                .is_err(),
                "final builder accepted a truncated canary-result sequence",
            );
        }
    }

    for (label, handoff) in [
        (
            "Disabled",
            compile_vision_qkv_stack_handoff(
                SYNTHETIC_MANIFEST_BYTES,
                VisionQkvExecutionPolicy::Disabled,
                compiler_capabilities(),
            )
            .unwrap(),
        ),
        (
            "unsupported Preferred",
            compile_vision_qkv_stack_handoff(
                SYNTHETIC_MANIFEST_BYTES,
                VisionQkvExecutionPolicy::Preferred,
                VisionQkvCompilerCapabilities {
                    max_storage_buffers_per_shader_stage: 7,
                    ..compiler_capabilities()
                },
            )
            .unwrap(),
        ),
    ] {
        assert!(handoff.selection().overlay().is_none(), "{label} fixture");
        let plan = BrowserVisionQkvExecutionEvidencePlan::from_prepared(None)
            .expect("fallback absence is valid");
        assert!(plan.is_none());
        let begin = BrowserVisionQkvBeginExecutionEvidence::from_plan(plan.as_ref());
        let final_evidence =
            BrowserVisionQkvFinalExecutionEvidence::from_verified_plan(plan.as_ref(), &[])
                .expect("fallback has no canary results");
        assert!(begin.is_none() && final_evidence.is_none());
        let selection = build_vision_qkv_selection_evidence_propagation(&handoff);
        let legacy_manifest = parse_vision_stack_shard_manifest(SYNTHETIC_MANIFEST_BYTES).unwrap();
        let legacy_plan = legacy_manifest.plan().unwrap();
        let legacy_status = build_vision_stack_legacy_status_record(
            VisionStackShardProtocolPhase::Preflight,
            legacy_manifest
                .shards
                .first()
                .map(|shard| shard.id.as_str()),
            &legacy_plan,
            VisionStackActivationStrategy::SeparateBuffers,
            None,
            None,
            32,
            true,
        )
        .unwrap();
        let shader_blake3 = BTreeMap::from([(KernelId::LayerNormF32, "ef".repeat(32))]);
        let legacy_diagnostics = build_vision_stack_legacy_diagnostics_record(
            &legacy_plan,
            VisionStackActivationStrategy::SeparateBuffers,
            None,
            None,
            32,
            &shader_blake3,
            1,
            17,
            18,
        )
        .unwrap();
        let begin_json: serde_json::Value = serde_json::from_str(
            &serialize_vision_stack_qkv_begin_status_json(
                &legacy_status,
                selection.additive_begin_evidence(begin.as_ref()),
            )
            .unwrap(),
        )
        .unwrap();
        let final_json: serde_json::Value = serde_json::from_str(
            &serialize_vision_stack_qkv_final_diagnostics_json(
                &legacy_diagnostics,
                selection.final_diagnostics_evidence(final_evidence.as_ref()),
            )
            .unwrap(),
        )
        .unwrap();
        for (channel, envelope) in [("begin", begin_json), ("final", final_json)] {
            assert!(envelope.get("qkv_selection").is_some());
            assert!(envelope.get("qkv_execution").is_some());
            assert!(
                envelope["qkv_execution"].is_null(),
                "{label} {channel} must serialize explicit qkv_execution:null",
            );
        }
        assert!(
            BrowserVisionQkvFinalExecutionEvidence::from_verified_plan(plan.as_ref(), &[true])
                .is_err(),
            "{label} accepted final canary results without an execution plan",
        );
    }
}

#[test]
fn host_compiler_handoff_prepares_one_sealed_spec_and_maps_every_exact_capability() {
    let capabilities = compiler_capabilities();
    let handoff = compile_vision_qkv_stack_handoff(
        SYNTHETIC_MANIFEST_BYTES,
        VisionQkvExecutionPolicy::Required,
        capabilities,
    )
    .expect("synthetic Required handoff");
    let exact = prepare_vision_qkv_stack_handoff_execution(
        &handoff,
        SYNTHETIC_MANIFEST_BYTES,
        capabilities,
        compiler_readback_request(),
    )
    .expect("every exact capability boundary must produce the sealed physical spec");
    assert_eq!(exact.prepared_execution().layer_count(), 3);
    assert_eq!(
        exact.prepared_execution().workspace().allocation_bytes(),
        256
    );
    assert_eq!(exact.readback_layout().semantic_readback_bytes(), 16);
    assert_eq!(exact.readback_layout().scratch_canary_readback_bytes(), 8);
    assert_eq!(exact.readback_layout().qkv_canary_readback_bytes(), 112);
    assert_eq!(exact.readback_layout().total_readback_bytes(), 136);

    let larger = VisionQkvCompilerCapabilities {
        max_storage_buffers_per_shader_stage: 32,
        max_storage_buffer_binding_size: 1_u64 << 38,
        max_buffer_size: 1_u64 << 39,
        max_compute_workgroup_size: [16, 16, 4],
        max_compute_invocations_per_workgroup: 1_024,
        max_compute_workgroups_per_dimension: 1_000_000,
        max_host_elements: 1_u64 << 32,
        ..capabilities
    };
    prepare_vision_qkv_stack_handoff_execution(
        &handoff,
        SYNTHETIC_MANIFEST_BYTES,
        larger,
        compiler_readback_request(),
    )
    .expect("same alignment with larger non-identity maxima must remain reusable");

    let one_under_host = exact.readback_layout().readback_f32_elements() as u64 - 1;
    let cases = [
        (
            "storage binding count",
            VisionQkvCompilerHandoffErrorCode::TargetStorageBindings,
            VisionQkvCompilerCapabilities {
                max_storage_buffers_per_shader_stage: 7,
                ..capabilities
            },
        ),
        (
            "storage binding bytes",
            VisionQkvCompilerHandoffErrorCode::TargetBindingSize,
            VisionQkvCompilerCapabilities {
                max_storage_buffer_binding_size: 191,
                ..capabilities
            },
        ),
        (
            "buffer bytes",
            VisionQkvCompilerHandoffErrorCode::TargetBufferSize,
            VisionQkvCompilerCapabilities {
                max_buffer_size: 255,
                ..capabilities
            },
        ),
        (
            "workgroup x",
            VisionQkvCompilerHandoffErrorCode::ComputeWorkgroupX,
            VisionQkvCompilerCapabilities {
                max_compute_workgroup_size: [7, 8, 1],
                ..capabilities
            },
        ),
        (
            "workgroup y",
            VisionQkvCompilerHandoffErrorCode::ComputeWorkgroupY,
            VisionQkvCompilerCapabilities {
                max_compute_workgroup_size: [8, 7, 1],
                ..capabilities
            },
        ),
        (
            "workgroup z",
            VisionQkvCompilerHandoffErrorCode::ComputeWorkgroupZ,
            VisionQkvCompilerCapabilities {
                max_compute_workgroup_size: [8, 8, 0],
                ..capabilities
            },
        ),
        (
            "workgroup invocation product",
            VisionQkvCompilerHandoffErrorCode::ComputeInvocations,
            VisionQkvCompilerCapabilities {
                max_compute_invocations_per_workgroup: 63,
                ..capabilities
            },
        ),
        (
            "dispatch dimension",
            VisionQkvCompilerHandoffErrorCode::ComputeDispatch,
            VisionQkvCompilerCapabilities {
                max_compute_workgroups_per_dimension: 2,
                ..capabilities
            },
        ),
        (
            "host element count",
            VisionQkvCompilerHandoffErrorCode::HostElements,
            VisionQkvCompilerCapabilities {
                max_host_elements: one_under_host,
                ..capabilities
            },
        ),
    ];
    for (field, expected, actual) in cases {
        assert_compiler_handoff_error(
            &format!("synthetic oracle/depth 3 one-under {field}"),
            expected,
            prepare_vision_qkv_stack_handoff_execution(
                &handoff,
                SYNTHETIC_MANIFEST_BYTES,
                actual,
                compiler_readback_request(),
            ),
        );
    }
}

#[test]
fn host_compiler_handoff_rejects_depth_geometry_digest_and_alignment_rebinding_independently() {
    let capabilities = compiler_capabilities();
    let handoff = compile_vision_qkv_stack_handoff(
        SYNTHETIC_MANIFEST_BYTES,
        VisionQkvExecutionPolicy::Required,
        capabilities,
    )
    .expect("synthetic Required handoff");

    let depth = canonical_manifest_mutant(SYNTHETIC_MANIFEST_BYTES, |manifest| {
        manifest.layer_count = 2;
        manifest.checkpoint_layers = vec![0];
        manifest.shards.remove(3);
    });
    let geometry = canonical_manifest_mutant(SYNTHETIC_MANIFEST_BYTES, |manifest| {
        manifest.tokens += 1;
        *manifest.cu_seqlens.last_mut().unwrap() = manifest.tokens;
        manifest.shards[0].bytes += u64::from(manifest.hidden_size) * 4;
    });
    let digest_only = canonical_manifest_mutant(SYNTHETIC_MANIFEST_BYTES, |manifest| {
        manifest.compiler_build = "b".repeat(64);
    });
    for (label, expected, manifest) in [
        (
            "cross depth checked before digest",
            VisionQkvCompilerHandoffErrorCode::ManifestDepthBinding,
            depth.as_slice(),
        ),
        (
            "cross geometry checked before digest",
            VisionQkvCompilerHandoffErrorCode::ManifestGeometryBinding,
            geometry.as_slice(),
        ),
        (
            "same depth/geometry byte identity",
            VisionQkvCompilerHandoffErrorCode::ManifestDigestBinding,
            digest_only.as_slice(),
        ),
    ] {
        assert_compiler_handoff_error(
            &format!("synthetic oracle/depth 3 {label}"),
            expected,
            prepare_vision_qkv_stack_handoff_execution(
                &handoff,
                manifest,
                capabilities,
                compiler_readback_request(),
            ),
        );
    }
    assert_compiler_handoff_error(
        "synthetic oracle/depth 3 stale alignment",
        VisionQkvCompilerHandoffErrorCode::TargetAlignment,
        prepare_vision_qkv_stack_handoff_execution(
            &handoff,
            SYNTHETIC_MANIFEST_BYTES,
            VisionQkvCompilerCapabilities {
                min_storage_buffer_offset_alignment: 256,
                ..capabilities
            },
            compiler_readback_request(),
        ),
    );
}

#[test]
fn every_opaque_selection_outcome_is_bound_before_the_web_topology_branch() {
    let capabilities = compiler_capabilities();
    let depth = canonical_manifest_mutant(SYNTHETIC_MANIFEST_BYTES, |manifest| {
        manifest.layer_count = 2;
        manifest.checkpoint_layers = vec![0];
        manifest.shards.remove(3);
    });
    let geometry = canonical_manifest_mutant(SYNTHETIC_MANIFEST_BYTES, |manifest| {
        manifest.tokens += 1;
        *manifest.cu_seqlens.last_mut().unwrap() = manifest.tokens;
        manifest.shards[0].bytes += u64::from(manifest.hidden_size) * 4;
    });
    let digest = canonical_manifest_mutant(SYNTHETIC_MANIFEST_BYTES, |manifest| {
        manifest.compiler_build = "c".repeat(64);
    });

    for (label, policy, compile_capabilities) in [
        ("Disabled", VisionQkvExecutionPolicy::Disabled, capabilities),
        (
            "Preferred unsupported fallback",
            VisionQkvExecutionPolicy::Preferred,
            VisionQkvCompilerCapabilities {
                max_storage_buffers_per_shader_stage: 7,
                ..capabilities
            },
        ),
    ] {
        let handoff = compile_vision_qkv_stack_handoff(
            SYNTHETIC_MANIFEST_BYTES,
            policy,
            compile_capabilities,
        )
        .unwrap_or_else(|error| panic!("{label} handoff must compile: {error}"));

        for (mutation, expected, manifest) in [
            (
                "depth",
                VisionQkvCompilerHandoffErrorCode::ManifestDepthBinding,
                depth.as_slice(),
            ),
            (
                "geometry",
                VisionQkvCompilerHandoffErrorCode::ManifestGeometryBinding,
                geometry.as_slice(),
            ),
            (
                "digest",
                VisionQkvCompilerHandoffErrorCode::ManifestDigestBinding,
                digest.as_slice(),
            ),
        ] {
            assert_compiler_handoff_error(
                &format!("{label} cross-manifest {mutation} binding"),
                expected,
                prepare_vision_qkv_stack_handoff_execution(
                    &handoff,
                    manifest,
                    compile_capabilities,
                    compiler_readback_request(),
                ),
            );
        }

        assert_compiler_handoff_error(
            &format!("{label} exact sealed manifest/target is binding-compatible"),
            VisionQkvCompilerHandoffErrorCode::NoFusedExecution,
            prepare_vision_qkv_stack_handoff_execution(
                &handoff,
                SYNTHETIC_MANIFEST_BYTES,
                compile_capabilities,
                compiler_readback_request(),
            ),
        );

        let larger_compatible = VisionQkvCompilerCapabilities {
            max_storage_buffers_per_shader_stage: compile_capabilities
                .max_storage_buffers_per_shader_stage
                + 1,
            max_storage_buffer_binding_size: compile_capabilities.max_storage_buffer_binding_size
                + 1,
            max_buffer_size: compile_capabilities.max_buffer_size + 1,
            max_compute_workgroups_per_dimension: compile_capabilities
                .max_compute_workgroups_per_dimension
                + 1,
            ..compile_capabilities
        };
        assert_compiler_handoff_error(
            &format!("{label} larger compatible maxima retain the sealed outcome"),
            VisionQkvCompilerHandoffErrorCode::NoFusedExecution,
            prepare_vision_qkv_stack_handoff_execution(
                &handoff,
                SYNTHETIC_MANIFEST_BYTES,
                larger_compatible,
                compiler_readback_request(),
            ),
        );

        let insufficient_targets = [
            (
                "storage buffer alignment identity",
                VisionQkvCompilerHandoffErrorCode::TargetAlignment,
                VisionQkvCompilerCapabilities {
                    min_storage_buffer_offset_alignment: compile_capabilities
                        .min_storage_buffer_offset_alignment
                        * 2,
                    ..compile_capabilities
                },
            ),
            (
                "storage binding count",
                VisionQkvCompilerHandoffErrorCode::TargetStorageBindings,
                VisionQkvCompilerCapabilities {
                    max_storage_buffers_per_shader_stage: compile_capabilities
                        .max_storage_buffers_per_shader_stage
                        - 1,
                    ..compile_capabilities
                },
            ),
            (
                "storage binding bytes",
                VisionQkvCompilerHandoffErrorCode::TargetBindingSize,
                VisionQkvCompilerCapabilities {
                    max_storage_buffer_binding_size: compile_capabilities
                        .max_storage_buffer_binding_size
                        - 1,
                    ..compile_capabilities
                },
            ),
            (
                "buffer bytes",
                VisionQkvCompilerHandoffErrorCode::TargetBufferSize,
                VisionQkvCompilerCapabilities {
                    max_buffer_size: compile_capabilities.max_buffer_size - 1,
                    ..compile_capabilities
                },
            ),
            (
                "dispatch dimension",
                VisionQkvCompilerHandoffErrorCode::ComputeDispatch,
                VisionQkvCompilerCapabilities {
                    max_compute_workgroups_per_dimension: compile_capabilities
                        .max_compute_workgroups_per_dimension
                        - 1,
                    ..compile_capabilities
                },
            ),
        ];
        for (field, expected, actual) in insufficient_targets {
            assert_compiler_handoff_error(
                &format!("{label} rejects stale/insufficient sealed {field}"),
                expected,
                prepare_vision_qkv_stack_handoff_execution(
                    &handoff,
                    SYNTHETIC_MANIFEST_BYTES,
                    actual,
                    compiler_readback_request(),
                ),
            );
        }
    }

    let host_binding = braced_item(
        LIB_SOURCE,
        "pub(crate) fn validate_vision_qkv_stack_handoff_binding(",
    );
    assert_order(
        host_binding,
        &[
            "parse_vision_stack_shard_manifest(",
            "ManifestDepthBinding",
            "ManifestGeometryBinding",
            "ManifestDigestBinding",
            "target_limits_from_capabilities(",
            "selection().outcome()",
        ],
    );
    for field in [
        "min_storage_buffer_offset_alignment",
        "max_storage_buffers_per_shader_stage",
        "max_storage_buffer_binding_size",
        "max_buffer_size",
        "max_compute_workgroups_per_dimension",
    ] {
        assert!(
            host_binding.contains(field),
            "non-fused handoff target binding omitted {field}",
        );
    }
    assert!(
        host_binding.contains("VisionQkvSelectionOutcome::Disabled")
            && host_binding.contains("VisionQkvSelectionOutcome::FallbackUnsupportedTarget")
            && host_binding.contains("VisionQkvSelectionOutcome::Fused"),
        "handoff binding does not exhaustively distinguish all selection outcomes",
    );

    let fused_prepare = braced_item(
        LIB_SOURCE,
        "pub fn prepare_vision_qkv_stack_handoff_execution(",
    );
    assert_eq!(
        occurrences(fused_prepare, "validate_vision_qkv_stack_handoff_binding(",),
        1,
        "fused preparation must reuse the one common manifest/target binding validator",
    );

    let begin = braced_item(
        WEB_SOURCE,
        "fn begin_vision_stack_sharded_with_qkv_selection(",
    );
    let binding = direct_call_binding(begin, "validate_vision_qkv_stack_handoff_binding(", false);
    let arguments = balanced_call_arguments(begin, "validate_vision_qkv_stack_handoff_binding(");
    assert_eq!(arguments.len(), 3);
    assert_eq!(compact(arguments[0]), "handoff");
    assert_eq!(compact(arguments[1]), "manifest_bytes");
    assert_eq!(compact(arguments[2]), "capabilities");
    let topology_branch = begin
        .find("let session = if qkv_outcome")
        .expect("optimized begin must retain one explicit topology branch");
    let browser_preparation = begin
        .find("self.prepare_browser_stack(")
        .expect("optimized begin must prepare its browser session");
    let session_publish = begin
        .find(".begin(session)")
        .expect("optimized begin must publish exactly one validated session");
    assert!(
        binding.call_start < browser_preparation
            && browser_preparation < topology_branch
            && topology_branch < session_publish,
        "opaque rejection can mutate prepared/session state or branch before binding validation",
    );
    assert_eq!(
        occurrences(begin, "validate_vision_qkv_stack_handoff_binding("),
        1,
        "optimized begin must validate its opaque handoff exactly once",
    );
}

#[derive(Default)]
struct RecordingWebPhysicalSink {
    commands: Vec<VisionQkvWebPhysicalCommand>,
}

impl RecordingWebPhysicalSink {
    fn record(&mut self, plan: &VisionQkvWebPhysicalCommandPlan) {
        self.commands.extend_from_slice(plan.commands());
    }
}

#[test]
fn sealed_spec_behaviorally_emits_the_only_exact_web_physical_command_plan() {
    let factory: fn(&VisionQkvPhysicalExecutionSpec) -> VisionQkvWebPhysicalCommandPlan =
        plan_vision_qkv_web_physical_commands;
    for depth in [1_u32, 3] {
        for alignment in [32_u32, 256] {
            for (semantic_readback_bytes, scratch_canary_readback_bytes) in
                [(16_u64, 8_u64), (28, 20)]
            {
                let spec = physical_spec_fixture(
                    depth,
                    alignment,
                    semantic_readback_bytes,
                    scratch_canary_readback_bytes,
                );
                let plan = factory(&spec);
                let mut sink = RecordingWebPhysicalSink::default();
                sink.record(&plan);

                let (semantic_workspace_bytes, canaries, attention_offsets) = match alignment {
                    32 => (
                        192,
                        vec![(0, 32), (80, 16), (144, 16), (208, 16), (224, 32)],
                        [32, 96, 160],
                    ),
                    256 => (
                        768,
                        vec![(0, 256), (304, 208), (560, 208), (816, 208), (1_024, 256)],
                        [256, 512, 768],
                    ),
                    _ => unreachable!(),
                };
                let workspace_bytes = semantic_workspace_bytes + 2 * u64::from(alignment);
                let qkv_canary_bytes = canaries.iter().map(|(_, bytes)| bytes).sum::<u64>();
                let total_readback_bytes =
                    semantic_readback_bytes + scratch_canary_readback_bytes + qkv_canary_bytes;
                assert_eq!(
                    sink.commands.len(),
                    2 + usize::try_from(depth).unwrap() * 2 + canaries.len() + 1,
                    "depth {depth}, alignment {alignment}: physical command cardinality",
                );

                assert_eq!(
                    sink.commands[0],
                    VisionQkvWebPhysicalCommand::CreateBuffer {
                        buffer: VisionQkvWebPhysicalBuffer::Workspace,
                        label: "vision-stack-qkv-workspace",
                        byte_length: workspace_bytes,
                    },
                );
                assert_eq!(
                    sink.commands[1],
                    VisionQkvWebPhysicalCommand::CreateBuffer {
                        buffer: VisionQkvWebPhysicalBuffer::Readback,
                        label: "vision-stack-readback",
                        byte_length: total_readback_bytes,
                    },
                );

                let expected_fused = vec![
                    (
                        0,
                        VisionQkvWebBindingResource::Norm1Output { byte_length: 48 },
                    ),
                    (
                        1,
                        VisionQkvWebBindingResource::QueryWeight { byte_length: 64 },
                    ),
                    (
                        2,
                        VisionQkvWebBindingResource::QueryBias { byte_length: 16 },
                    ),
                    (
                        3,
                        VisionQkvWebBindingResource::KeyWeight { byte_length: 64 },
                    ),
                    (4, VisionQkvWebBindingResource::KeyBias { byte_length: 16 }),
                    (
                        5,
                        VisionQkvWebBindingResource::ValueWeight { byte_length: 64 },
                    ),
                    (
                        6,
                        VisionQkvWebBindingResource::ValueBias { byte_length: 16 },
                    ),
                    (
                        7,
                        VisionQkvWebBindingResource::WorkspaceRange {
                            byte_offset: u64::from(alignment),
                            byte_length: semantic_workspace_bytes,
                        },
                    ),
                    (
                        8,
                        VisionQkvWebBindingResource::Uniform {
                            slot: 1,
                            byte_length: 16,
                        },
                    ),
                ];
                let expected_attention = vec![
                    (
                        0,
                        VisionQkvWebBindingResource::WorkspaceRange {
                            byte_offset: attention_offsets[0],
                            byte_length: 48,
                        },
                    ),
                    (
                        1,
                        VisionQkvWebBindingResource::WorkspaceRange {
                            byte_offset: attention_offsets[1],
                            byte_length: 48,
                        },
                    ),
                    (
                        2,
                        VisionQkvWebBindingResource::WorkspaceRange {
                            byte_offset: attention_offsets[2],
                            byte_length: 48,
                        },
                    ),
                    (
                        3,
                        VisionQkvWebBindingResource::CuSeqlens { byte_length: 12 },
                    ),
                    (
                        4,
                        VisionQkvWebBindingResource::AttentionOutput { byte_length: 48 },
                    ),
                    (
                        5,
                        VisionQkvWebBindingResource::Uniform {
                            slot: 4,
                            byte_length: 16,
                        },
                    ),
                ];
                for layer in 0..depth {
                    let fused_index = 2 + usize::try_from(layer).unwrap() * 2;
                    let attention_index = fused_index + 1;
                    for (
                        command,
                        expected_kind,
                        expected_label,
                        expected_uniform_slot,
                        expected_entries,
                    ) in [
                        (
                            &sink.commands[fused_index],
                            VisionQkvWebBindGroupKind::FusedQkv,
                            "vision-layer-qkv-fused-bind-group",
                            1,
                            &expected_fused,
                        ),
                        (
                            &sink.commands[attention_index],
                            VisionQkvWebBindGroupKind::Attention,
                            "vision-layer-attention-bind-group",
                            4,
                            &expected_attention,
                        ),
                    ] {
                        let VisionQkvWebPhysicalCommand::CreateBindGroup {
                            layer_index,
                            kind,
                            label,
                            uniform_slot,
                            entries,
                        } = command
                        else {
                            panic!("layer {layer}: expected typed bind-group command")
                        };
                        assert_eq!(*layer_index, layer);
                        assert_eq!(*kind, expected_kind);
                        assert_eq!(*label, expected_label);
                        assert_eq!(*uniform_slot, expected_uniform_slot);
                        assert_eq!(
                            entries
                                .iter()
                                .map(|entry| (entry.binding(), entry.resource().clone()))
                                .collect::<Vec<_>>(),
                            *expected_entries,
                        );
                    }
                }

                let mut destination = semantic_readback_bytes + scratch_canary_readback_bytes;
                let copies_start = 2 + usize::try_from(depth).unwrap() * 2;
                for (index, (source_offset, byte_length)) in canaries.iter().copied().enumerate() {
                    assert_eq!(
                        sink.commands[copies_start + index],
                        VisionQkvWebPhysicalCommand::CopyBuffer {
                            label: "vision-stack-qkv-canary-copy",
                            source: VisionQkvWebPhysicalBuffer::Workspace,
                            source_offset,
                            destination: VisionQkvWebPhysicalBuffer::Readback,
                            destination_offset: destination,
                            byte_length,
                        },
                        "depth {depth}, alignment {alignment}: canary copy {index}",
                    );
                    destination += byte_length;
                }
                assert_eq!(destination, total_readback_bytes);
                assert_eq!(
                    sink.commands.last(),
                    Some(&VisionQkvWebPhysicalCommand::MapRange {
                        label: "vision-stack-readback-map",
                        buffer: VisionQkvWebPhysicalBuffer::Readback,
                        byte_range: 0..total_readback_bytes,
                    }),
                    "mapped range must be the exact full sealed readback allocation",
                );

                assert_eq!(
                    plan.commands(),
                    sink.commands.as_slice(),
                    "recording sink must receive the immutable plan without reconstruction",
                );
            }
        }
    }
}

#[test]
fn typed_web_physical_plan_preserves_exact_fused_invocation_authority_for_compact_and_official_shapes()
 {
    for (label, alignment, manifest, layer_count, expected_workgroups, expected_uniform_words) in [
        (
            "compact-32",
            32_u32,
            SYNTHETIC_MANIFEST_BYTES,
            3_u32,
            [1_u32, 1, 3],
            [3_u32, 4, 4, 16],
        ),
        (
            "compact-256",
            256_u32,
            SYNTHETIC_MANIFEST_BYTES,
            3_u32,
            [1_u32, 1, 3],
            [3_u32, 4, 4, 64],
        ),
        (
            "official-32",
            32_u32,
            OFFICIAL_MANIFEST_BYTES,
            27_u32,
            [144_u32, 160, 3],
            [1_276_u32, 1_152, 1_152, 1_469_952],
        ),
        (
            "official-256",
            256_u32,
            OFFICIAL_MANIFEST_BYTES,
            27_u32,
            [144_u32, 160, 3],
            [1_276_u32, 1_152, 1_152, 1_469_952],
        ),
    ] {
        let capabilities = compiler_capabilities_with_alignment(alignment);
        let handoff = compile_vision_qkv_stack_handoff(
            manifest,
            VisionQkvExecutionPolicy::Required,
            capabilities,
        )
        .unwrap_or_else(|error| panic!("{label}: handoff failed: {error}"));
        let spec = prepare_vision_qkv_stack_handoff_execution(
            &handoff,
            manifest,
            capabilities,
            compiler_readback_request(),
        )
        .unwrap_or_else(|error| panic!("{label}: physical spec failed: {error}"));
        let plan = plan_vision_qkv_web_physical_commands(&spec);

        for layer_index in 0..layer_count {
            assert_eq!(
                plan.fused_dispatch_workgroups(layer_index),
                Some(expected_workgroups),
                "{label}: layer {layer_index} lost compiler dispatch authority",
            );
            assert_eq!(
                plan.fused_uniform_words(layer_index),
                Some(expected_uniform_words),
                "{label}: layer {layer_index} lost compiler uniform authority",
            );
        }
        assert_eq!(
            plan.fused_dispatch_workgroups(layer_count),
            None,
            "{label}: out-of-range layer exposed dispatch authority",
        );
        assert_eq!(
            plan.fused_uniform_words(layer_count),
            None,
            "{label}: out-of-range layer exposed uniform authority",
        );
    }
}

type RecordedWebPhysicalDispatch<'a> = (
    VisionQkvWebPhysicalCommandPhase,
    usize,
    &'a VisionQkvWebPhysicalCommand,
);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecordedWebPhysicalEffectKind {
    CreateBuffer,
    CreateBindGroup,
    CopyBuffer,
    MapRange,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecordingFailurePoint {
    Effect(usize),
    Store(usize),
}

#[derive(Debug)]
struct NonCloneWebPhysicalSinkError {
    identity: Rc<()>,
    drop_count: Rc<Cell<usize>>,
}

impl Drop for NonCloneWebPhysicalSinkError {
    fn drop(&mut self) {
        self.drop_count.set(self.drop_count.get() + 1);
    }
}

#[derive(Debug)]
struct RecordedCreatedBuffer(Rc<()>);

#[derive(Debug)]
struct RecordedCreatedBindGroup(Rc<()>);

#[derive(Clone, Debug, Eq, PartialEq)]
enum RecordedWebPhysicalEffect {
    Effect {
        phase: VisionQkvWebPhysicalCommandPhase,
        kind: RecordedWebPhysicalEffectKind,
        command_index: usize,
        command_identity: *const VisionQkvWebPhysicalCommand,
        result_identity: Option<*const ()>,
    },
    Store {
        phase: VisionQkvWebPhysicalCommandPhase,
        kind: RecordedWebPhysicalEffectKind,
        command_index: usize,
        command_identity: *const VisionQkvWebPhysicalCommand,
        result_identity: *const (),
    },
}

struct RecordingWebPhysicalEffectSink {
    phase: VisionQkvWebPhysicalCommandPhase,
    effects: Vec<RecordedWebPhysicalEffect>,
    failure: Option<RecordingFailurePoint>,
    error_identity: Rc<()>,
    error_drop_count: Rc<Cell<usize>>,
}

impl RecordingWebPhysicalEffectSink {
    fn new(phase: VisionQkvWebPhysicalCommandPhase) -> Self {
        Self {
            phase,
            effects: Vec::new(),
            failure: None,
            error_identity: Rc::new(()),
            error_drop_count: Rc::new(Cell::new(0)),
        }
    }

    fn with_failure(
        phase: VisionQkvWebPhysicalCommandPhase,
        failure: RecordingFailurePoint,
    ) -> Self {
        Self {
            failure: Some(failure),
            ..Self::new(phase)
        }
    }

    fn set_phase(&mut self, phase: VisionQkvWebPhysicalCommandPhase) {
        self.phase = phase;
    }

    fn sink_error(&self) -> NonCloneWebPhysicalSinkError {
        NonCloneWebPhysicalSinkError {
            identity: self.error_identity.clone(),
            drop_count: self.error_drop_count.clone(),
        }
    }

    fn record_effect(
        &mut self,
        kind: RecordedWebPhysicalEffectKind,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
        result_identity: Option<*const ()>,
    ) -> Result<(), NonCloneWebPhysicalSinkError> {
        self.effects.push(RecordedWebPhysicalEffect::Effect {
            phase: self.phase,
            kind,
            command_index,
            command_identity: command as *const _,
            result_identity,
        });
        if self.failure == Some(RecordingFailurePoint::Effect(command_index)) {
            return Err(self.sink_error());
        }
        Ok(())
    }

    fn record_store(
        &mut self,
        kind: RecordedWebPhysicalEffectKind,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
        result_identity: *const (),
    ) -> Result<(), NonCloneWebPhysicalSinkError> {
        self.effects.push(RecordedWebPhysicalEffect::Store {
            phase: self.phase,
            kind,
            command_index,
            command_identity: command as *const _,
            result_identity,
        });
        if self.failure == Some(RecordingFailurePoint::Store(command_index)) {
            return Err(self.sink_error());
        }
        Ok(())
    }
}

impl VisionQkvWebPhysicalCommandEffectSink for RecordingWebPhysicalEffectSink {
    type CreatedBuffer = RecordedCreatedBuffer;
    type CreatedBindGroup = RecordedCreatedBindGroup;
    type Error = NonCloneWebPhysicalSinkError;

    fn apply_create_buffer(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<Self::CreatedBuffer, Self::Error> {
        if self.failure == Some(RecordingFailurePoint::Effect(command_index)) {
            self.record_effect(
                RecordedWebPhysicalEffectKind::CreateBuffer,
                command_index,
                command,
                None,
            )?;
            unreachable!()
        }
        let created = RecordedCreatedBuffer(Rc::new(()));
        self.record_effect(
            RecordedWebPhysicalEffectKind::CreateBuffer,
            command_index,
            command,
            Some(Rc::as_ptr(&created.0)),
        )?;
        Ok(created)
    }

    fn store_created_buffer(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
        created: Self::CreatedBuffer,
    ) -> Result<(), Self::Error> {
        self.record_store(
            RecordedWebPhysicalEffectKind::CreateBuffer,
            command_index,
            command,
            Rc::as_ptr(&created.0),
        )
    }

    fn apply_create_bind_group(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<Self::CreatedBindGroup, Self::Error> {
        if self.failure == Some(RecordingFailurePoint::Effect(command_index)) {
            self.record_effect(
                RecordedWebPhysicalEffectKind::CreateBindGroup,
                command_index,
                command,
                None,
            )?;
            unreachable!()
        }
        let created = RecordedCreatedBindGroup(Rc::new(()));
        self.record_effect(
            RecordedWebPhysicalEffectKind::CreateBindGroup,
            command_index,
            command,
            Some(Rc::as_ptr(&created.0)),
        )?;
        Ok(created)
    }

    fn store_created_bind_group(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
        created: Self::CreatedBindGroup,
    ) -> Result<(), Self::Error> {
        self.record_store(
            RecordedWebPhysicalEffectKind::CreateBindGroup,
            command_index,
            command,
            Rc::as_ptr(&created.0),
        )
    }

    fn apply_copy_buffer(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<(), Self::Error> {
        self.record_effect(
            RecordedWebPhysicalEffectKind::CopyBuffer,
            command_index,
            command,
            None,
        )
    }

    fn apply_map_range(
        &mut self,
        command_index: usize,
        command: &VisionQkvWebPhysicalCommand,
    ) -> Result<(), Self::Error> {
        self.record_effect(
            RecordedWebPhysicalEffectKind::MapRange,
            command_index,
            command,
            None,
        )
    }
}

fn expected_web_physical_command_phase(
    command: &VisionQkvWebPhysicalCommand,
) -> VisionQkvWebPhysicalCommandPhase {
    match command {
        VisionQkvWebPhysicalCommand::CreateBuffer { .. } => VisionQkvWebPhysicalCommandPhase::Start,
        VisionQkvWebPhysicalCommand::CreateBindGroup { layer_index, .. } => {
            VisionQkvWebPhysicalCommandPhase::Layer {
                layer_index: *layer_index,
            }
        }
        VisionQkvWebPhysicalCommand::CopyBuffer { .. }
        | VisionQkvWebPhysicalCommand::MapRange { .. } => VisionQkvWebPhysicalCommandPhase::Finish,
    }
}

fn expected_web_physical_dispatch(
    plan: &VisionQkvWebPhysicalCommandPlan,
) -> Vec<RecordedWebPhysicalDispatch<'_>> {
    plan.commands()
        .iter()
        .enumerate()
        .map(|(command_index, command)| {
            (
                expected_web_physical_command_phase(command),
                command_index,
                command,
            )
        })
        .collect()
}

fn assert_web_physical_dispatch_rejected(
    label: &str,
    plan: &VisionQkvWebPhysicalCommandPlan,
    dispatches: &[RecordedWebPhysicalDispatch<'_>],
) {
    assert!(
        validate_vision_qkv_web_physical_command_dispatches(plan, dispatches).is_err(),
        "physical dispatch validator accepted hostile {label}",
    );
}

#[test]
fn typed_effect_executor_behaviorally_applies_and_stores_each_sealed_command_once_in_order() {
    for depth in [1_u32, 3] {
        for alignment in [32_u32, 256] {
            let spec = physical_spec_fixture(depth, alignment, 28, 20);
            let plan = plan_vision_qkv_web_physical_commands(&spec);
            let mut sink =
                RecordingWebPhysicalEffectSink::new(VisionQkvWebPhysicalCommandPhase::Start);
            for phase in std::iter::once(VisionQkvWebPhysicalCommandPhase::Start)
                .chain(
                    (0..depth)
                        .map(|layer_index| VisionQkvWebPhysicalCommandPhase::Layer { layer_index }),
                )
                .chain(std::iter::once(VisionQkvWebPhysicalCommandPhase::Finish))
            {
                sink.set_phase(phase);
                execute_vision_qkv_web_physical_commands(&plan, phase, &mut sink)
                    .expect("sealed physical phase must execute without reconstruction");
            }

            let effect_events = sink
                .effects
                .iter()
                .filter_map(|event| match event {
                    RecordedWebPhysicalEffect::Effect {
                        phase,
                        kind,
                        command_index,
                        command_identity,
                        ..
                    } => Some((phase, kind, command_index, command_identity)),
                    RecordedWebPhysicalEffect::Store { .. } => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(
                effect_events.len(),
                plan.commands().len(),
                "depth {depth}, alignment {alignment}: every sealed command must have one effect",
            );
            assert_eq!(
                effect_events
                    .iter()
                    .map(|(_, _, command_index, _)| **command_index)
                    .collect::<BTreeSet<_>>(),
                (0..plan.commands().len()).collect::<BTreeSet<_>>(),
                "depth {depth}, alignment {alignment}: effect indices are not exhaustive",
            );
            for (expected_index, (phase, kind, command_index, command_identity)) in
                effect_events.iter().enumerate()
            {
                assert_eq!(
                    **command_index, expected_index,
                    "depth {depth}, alignment {alignment}: executor reordered effects",
                );
                assert_eq!(
                    **command_identity,
                    &plan.commands()[expected_index] as *const _,
                    "depth {depth}, alignment {alignment}: command {expected_index} was reconstructed",
                );
                assert_eq!(
                    *phase,
                    &expected_web_physical_command_phase(&plan.commands()[expected_index]),
                    "depth {depth}, alignment {alignment}: command {expected_index} entered the wrong phase",
                );
                let expected_kind = match &plan.commands()[expected_index] {
                    VisionQkvWebPhysicalCommand::CreateBuffer { .. } => {
                        RecordedWebPhysicalEffectKind::CreateBuffer
                    }
                    VisionQkvWebPhysicalCommand::CreateBindGroup { .. } => {
                        RecordedWebPhysicalEffectKind::CreateBindGroup
                    }
                    VisionQkvWebPhysicalCommand::CopyBuffer { .. } => {
                        RecordedWebPhysicalEffectKind::CopyBuffer
                    }
                    VisionQkvWebPhysicalCommand::MapRange { .. } => {
                        RecordedWebPhysicalEffectKind::MapRange
                    }
                };
                assert_eq!(**kind, expected_kind);
            }

            let mut event_index = 0_usize;
            let mut expected_store_count = 0_usize;
            for (command_index, command) in plan.commands().iter().enumerate() {
                let RecordedWebPhysicalEffect::Effect {
                    phase,
                    kind,
                    command_index: actual_index,
                    command_identity,
                    result_identity,
                } = &sink.effects[event_index]
                else {
                    panic!("command {command_index} did not begin with its exact effect")
                };
                assert_eq!(*actual_index, command_index);
                assert_eq!(*command_identity, command as *const _);
                assert_eq!(phase, &expected_web_physical_command_phase(command));
                event_index += 1;
                if matches!(
                    command,
                    VisionQkvWebPhysicalCommand::CreateBuffer { .. }
                        | VisionQkvWebPhysicalCommand::CreateBindGroup { .. }
                ) {
                    expected_store_count += 1;
                    let created_identity =
                        result_identity.expect("created effect omitted its typed result identity");
                    let RecordedWebPhysicalEffect::Store {
                        phase: store_phase,
                        kind: store_kind,
                        command_index: store_index,
                        command_identity: stored_command,
                        result_identity: stored_result,
                    } = &sink.effects[event_index]
                    else {
                        panic!("created command {command_index} was not immediately stored")
                    };
                    assert_eq!(store_phase, phase);
                    assert_eq!(store_kind, kind);
                    assert_eq!(*store_index, command_index);
                    assert_eq!(*stored_command, command as *const _);
                    assert_eq!(*stored_result, created_identity);
                    event_index += 1;
                } else {
                    assert!(result_identity.is_none());
                }
            }
            assert_eq!(event_index, sink.effects.len());
            assert_eq!(
                expected_store_count,
                2 + usize::try_from(depth).unwrap() * 2
            );
            assert_eq!(
                sink.effects
                    .iter()
                    .filter(|event| matches!(event, RecordedWebPhysicalEffect::Store { .. }))
                    .count(),
                expected_store_count,
                "depth {depth}, alignment {alignment}: each typed created result must be stored once",
            );
        }
    }
}

#[test]
fn typed_effect_executor_validates_before_effect_and_preserves_nonclone_errors_with_cutoff() {
    let depth = 3_u32;
    let spec = physical_spec_fixture(depth, 32, 28, 20);
    let plan = plan_vision_qkv_web_physical_commands(&spec);

    let mut invalid =
        RecordingWebPhysicalEffectSink::new(VisionQkvWebPhysicalCommandPhase::Layer {
            layer_index: depth,
        });
    let validation_error = execute_vision_qkv_web_physical_commands(
        &plan,
        VisionQkvWebPhysicalCommandPhase::Layer { layer_index: depth },
        &mut invalid,
    )
    .expect_err("out-of-range layer silently dispatched an empty phase");
    assert!(
        validation_error.into_sink_error().is_none(),
        "pre-effect validation error was misclassified as a sink error",
    );
    assert!(
        invalid.effects.is_empty(),
        "full-stream and phase validation must finish before the first effect",
    );

    let first_layer_index = 2_usize;
    let layer_phase = VisionQkvWebPhysicalCommandPhase::Layer { layer_index: 0 };
    let mut effect_failure = RecordingWebPhysicalEffectSink::with_failure(
        layer_phase,
        RecordingFailurePoint::Effect(first_layer_index),
    );
    let effect_error =
        execute_vision_qkv_web_physical_commands(&plan, layer_phase, &mut effect_failure)
            .expect_err("injected effect failure must propagate");
    assert_eq!(effect_failure.effects.len(), 1);
    assert!(matches!(
        effect_failure.effects[0],
        RecordedWebPhysicalEffect::Effect {
            command_index: 2,
            result_identity: None,
            ..
        }
    ));
    let effect_error = effect_error
        .into_sink_error()
        .expect("effect failure lost its exact sink error");
    assert!(Rc::ptr_eq(
        &effect_error.identity,
        &effect_failure.error_identity,
    ));
    assert_eq!(effect_failure.error_drop_count.get(), 0);
    drop(effect_error);
    assert_eq!(effect_failure.error_drop_count.get(), 1);

    let mut store_failure = RecordingWebPhysicalEffectSink::with_failure(
        layer_phase,
        RecordingFailurePoint::Store(first_layer_index),
    );
    let store_error =
        execute_vision_qkv_web_physical_commands(&plan, layer_phase, &mut store_failure)
            .expect_err("injected typed-store failure must propagate")
            .into_sink_error()
            .expect("typed-store failure lost its exact sink error");
    assert!(Rc::ptr_eq(
        &store_error.identity,
        &store_failure.error_identity,
    ));
    assert_eq!(store_failure.effects.len(), 2);
    assert!(matches!(
        store_failure.effects.as_slice(),
        [
            RecordedWebPhysicalEffect::Effect {
                command_index: 2,
                result_identity: Some(_),
                ..
            },
            RecordedWebPhysicalEffect::Store {
                command_index: 2,
                ..
            }
        ]
    ));
    assert_eq!(store_failure.error_drop_count.get(), 0);
    drop(store_error);
    assert_eq!(store_failure.error_drop_count.get(), 1);

    let finish_start = 2 + usize::try_from(depth).unwrap() * 2;
    let mut finish_failure = RecordingWebPhysicalEffectSink::with_failure(
        VisionQkvWebPhysicalCommandPhase::Finish,
        RecordingFailurePoint::Effect(finish_start),
    );
    let finish_error = execute_vision_qkv_web_physical_commands(
        &plan,
        VisionQkvWebPhysicalCommandPhase::Finish,
        &mut finish_failure,
    )
    .expect_err("finish effect failure must stop the remaining copies/map")
    .into_sink_error()
    .expect("finish effect failure lost its sink error");
    assert_eq!(finish_failure.effects.len(), 1);
    assert!(matches!(
        finish_failure.effects[0],
        RecordedWebPhysicalEffect::Effect { command_index, .. }
            if command_index == finish_start
    ));
    drop(finish_error);
    assert_eq!(finish_failure.error_drop_count.get(), 1);
}

#[test]
fn typed_phase_dispatch_validator_rejects_incomplete_duplicated_reordered_and_cross_phase_streams()
{
    let depth = 3_u32;
    let spec = physical_spec_fixture(depth, 32, 28, 20);
    let plan = plan_vision_qkv_web_physical_commands(&spec);
    let exact = expected_web_physical_dispatch(&plan);

    assert_web_physical_dispatch_rejected("zero-command stream", &plan, &[]);

    let mut skipped = exact.clone();
    skipped.remove(2);
    assert_web_physical_dispatch_rejected("skipped command", &plan, &skipped);

    let mut duplicated = exact.clone();
    duplicated[3] = exact[2];
    assert_web_physical_dispatch_rejected("duplicate command", &plan, &duplicated);

    let mut reordered = exact.clone();
    reordered.swap(2, 3);
    assert_web_physical_dispatch_rejected("reordered commands", &plan, &reordered);

    let mut wrong_layer = exact.clone();
    let layer_position = wrong_layer
        .iter()
        .position(|(phase, _, _)| matches!(phase, VisionQkvWebPhysicalCommandPhase::Layer { .. }))
        .expect("exact trace must contain a layer command");
    let wrong_layer_index = match &wrong_layer[layer_position].0 {
        VisionQkvWebPhysicalCommandPhase::Layer { layer_index } => *layer_index + 1,
        _ => unreachable!(),
    };
    wrong_layer[layer_position].0 = VisionQkvWebPhysicalCommandPhase::Layer {
        layer_index: wrong_layer_index,
    };
    assert_web_physical_dispatch_rejected("wrong-layer command", &plan, &wrong_layer);

    let mut cross_phase = exact.clone();
    let finish_position = cross_phase
        .iter()
        .position(|(phase, _, _)| matches!(phase, VisionQkvWebPhysicalCommandPhase::Finish))
        .expect("exact trace must contain a finish command");
    cross_phase[finish_position].0 = VisionQkvWebPhysicalCommandPhase::Start;
    assert_web_physical_dispatch_rejected("cross-phase command", &plan, &cross_phase);

    let mut extra_cross_phase = exact.clone();
    extra_cross_phase.push((
        VisionQkvWebPhysicalCommandPhase::Finish,
        exact.len(),
        exact[0].2,
    ));
    assert_web_physical_dispatch_rejected("extra cross-phase command", &plan, &extra_cross_phase);

    let reconstructed_command = plan.commands()[2].clone();
    let mut reconstructed = exact.clone();
    reconstructed[2].2 = &reconstructed_command;
    assert_web_physical_dispatch_rejected(
        "equal-value reconstructed command",
        &plan,
        &reconstructed,
    );
}

#[test]
fn web_is_a_leaf_and_consumes_passes_owned_catalog_and_prepared_view() {
    for dependency in [
        "pvlc-ir",
        "pvlc-model-schema",
        "pvlc-pack",
        "pvlc-passes",
        "pvlc-runtime-core",
    ] {
        assert!(
            WEB_MANIFEST.contains(&format!("{dependency} = {{ path =")),
            "host-pure Web compiler dependency {dependency} is missing",
        );
    }

    let compile = braced_item(LIB_SOURCE, "pub fn compile_vision_qkv_stack_handoff(");
    assert_eq!(
        occurrences(compile, "canonical_synthetic_vision_qkv_tensor_catalog("),
        1,
    );
    assert_eq!(
        occurrences(compile, "PaddleOcrVl16Schema::tensor_specs("),
        1
    );
    assert_eq!(occurrences(compile, "SemanticGraph::paddleocr_vl_16("), 1);
    assert_eq!(occurrences(compile, "select_vision_qkv_stack_overlay("), 1);
    let catalog = named_initializer_binding(compile, "match manifest.oracle");
    let synthetic_arm = braced_item(compile, "VisionStackShardOracle::Synthetic =>");
    assert!(synthetic_arm.contains("canonical_synthetic_vision_qkv_tensor_catalog("));
    assert!(!synthetic_arm.contains("PaddleOcrVl16Schema::tensor_specs("));
    let official_arm = braced_item(compile, "VisionStackShardOracle::OfficialMpsBf16 =>");
    assert!(official_arm.contains("PaddleOcrVl16Schema::tensor_specs("));
    assert!(!official_arm.contains("canonical_synthetic_vision_qkv_tensor_catalog("));
    let builder_arguments =
        balanced_call_arguments(compile, "build_verified_vision_qkv_stack_overlay(");
    assert_eq!(builder_arguments.len(), 5);
    assert_eq!(
        compact(builder_arguments[3]),
        format!("&{catalog}"),
        "overlay builder did not consume the exact oracle-selected catalog binding",
    );
    let handoff = compact(braced_item(compile, "VisionQkvCompilerHandoff {"));
    assert!(
        handoff.contains(&format!("tensor_catalog_len:{catalog}.len()")),
        "catalog evidence did not come from the same exact builder catalog binding",
    );
    for required in [
        "parse_vision_stack_shard_manifest(",
        "blake3::hash(",
        "SemanticGraph::paddleocr_vl_16(",
        ".canonical_bytes()",
        "select_vision_qkv_stack_overlay(",
    ] {
        assert!(
            compile.contains(required),
            "compiler seam omitted {required}"
        );
    }
    assert!(!compile.contains("format!(\"visual.vision_model.encoder.layers"));

    let prepare = braced_item(
        LIB_SOURCE,
        "pub fn prepare_vision_qkv_stack_handoff_execution(",
    );
    assert_order(
        prepare,
        &[
            "validate_vision_qkv_stack_handoff_binding(",
            "prepare_vision_qkv_stack_execution(",
            "ComputeDispatchLimits",
            ".validate(",
            "plan_vision_qkv_readback_layout(",
            "bind_vision_qkv_physical_execution(",
        ],
    );
    for required in [
        "max_compute_workgroup_size",
        "max_compute_invocations_per_workgroup",
        "max_compute_workgroups_per_dimension",
        "max_buffer_size",
        "max_host_elements",
        "semantic_readback_bytes",
        "scratch_canary_readback_bytes",
    ] {
        assert!(
            prepare.contains(required),
            "prepare seam omitted {required}"
        );
    }
    for forbidden in ["checked_add(", "checked_mul(", "try_fold(", "/4"] {
        assert!(
            !compact(prepare).contains(forbidden),
            "host compiler seam duplicated core/passes arithmetic via {forbidden}",
        );
    }

    assert!(!WEB_SOURCE.contains("prepare_vision_qkv_stack_execution("));
    assert!(!WEB_SOURCE.contains("plan_vision_qkv_readback_layout("));
    assert!(!WEB_SOURCE.contains("bind_vision_qkv_physical_execution("));
    assert!(NATIVE_SOURCE.contains("prepare_vision_qkv_stack_execution("));
    for (adapter, source) in [("host Web seam", LIB_SOURCE), ("native", NATIVE_SOURCE)] {
        assert!(!source.contains("plan_vision_qkv_fused_geometry"));
        assert!(
            source.contains("ComputeDispatchLimits"),
            "{adapter} lost compute limits"
        );
        assert!(
            source.contains("plan_vision_qkv_readback_layout("),
            "{adapter} lost the common readback planner",
        );
    }
    assert!(!WEB_SOURCE.contains("struct PreparedVisionQkvWorkspace"));
    assert!(!NATIVE_SOURCE.contains("struct PreparedVisionQkvWorkspace"));

    let native_prepare = braced_item(NATIVE_SOURCE, "fn prepare_vision_qkv_execution(");
    assert!(
        native_prepare.contains("ComputeDispatchLimits")
            && compact(native_prepare).contains(".validate(&executor_invocation)"),
        "native prepared execution retained hand-written workgroup-axis or invocation arithmetic",
    );
    assert!(
        !compact(native_prepare).contains("try_fold(1_u32,u32::checked_mul)"),
        "native must not duplicate the common checked invocation-product implementation",
    );
    let native_readback = braced_item(
        NATIVE_SOURCE,
        "fn preflight_vision_qkv_execution_allocations(",
    );
    assert!(
        native_readback.contains("plan_vision_qkv_readback_layout("),
        "native allocation preflight must adapt, return, and consume the common core layout",
    );
    for forbidden in ["checked_add(", "/4", "usize::try_from(", "try_fold("] {
        assert!(
            !compact(native_readback).contains(forbidden),
            "thin native readback adapter duplicated core arithmetic via {forbidden}",
        );
    }
}

#[test]
fn shared_preflight_and_prepared_values_have_one_named_authority_dataflow() {
    let host_prepare = braced_item(
        LIB_SOURCE,
        "pub fn prepare_vision_qkv_stack_handoff_execution(",
    );
    let prepared = direct_call_binding(host_prepare, "prepare_vision_qkv_stack_execution(", false);
    let layout = direct_call_binding(host_prepare, "plan_vision_qkv_readback_layout(", false);
    let physical = direct_call_binding(host_prepare, "bind_vision_qkv_physical_execution(", false);
    assert!(
        prepared.call_start < layout.call_start && layout.call_start < physical.call_start,
        "host handoff did not prepare, plan readback, and seal in one ordered path",
    );
    let physical_arguments =
        balanced_call_arguments(host_prepare, "bind_vision_qkv_physical_execution(");
    assert_eq!(physical_arguments.len(), 2);
    assert_eq!(compact(physical_arguments[0]), prepared.name);
    assert_eq!(compact(physical_arguments[1]), layout.name);
    assert!(
        compact(&host_prepare[physical.statement_end + 1..])
            .contains(&format!("Ok({})", physical.name)),
        "host handoff did not return the exact sealed physical binding",
    );

    let web_compile = braced_item(
        WEB_SOURCE,
        "pub fn compile_vision_encoder_stack_qkv_selection(",
    );
    let compiled = direct_call_binding(web_compile, "compile_vision_qkv_stack_handoff(", false);
    let web_selection = braced_item(web_compile, "WebVisionQkvStackSelection {");
    assert_eq!(
        compact(struct_field_initializer(web_selection, "handoff").unwrap()),
        compiled.name,
        "wasm compile wrapper did not move the exact host handoff into its opaque handle",
    );

    let begin = braced_item(
        WEB_SOURCE,
        "fn begin_vision_stack_sharded_with_qkv_selection(",
    );
    let manifest_bytes = plain_call_binding(begin, ".as_bytes(");
    let handoff = named_initializer_binding(begin, "&qkv_selection.handoff");
    let capabilities = plain_call_binding(begin, "vision_qkv_compiler_capabilities(");
    let readback = named_initializer_binding(begin, "VisionQkvCompilerReadbackRequest {");
    let web_physical =
        direct_call_binding(begin, "prepare_vision_qkv_stack_handoff_execution(", false);
    let web_arguments =
        balanced_call_arguments(begin, "prepare_vision_qkv_stack_handoff_execution(");
    assert_eq!(web_arguments.len(), 4);
    assert_eq!(compact(web_arguments[0]), handoff);
    assert_eq!(compact(web_arguments[1]), manifest_bytes.name);
    assert_eq!(compact(web_arguments[2]), capabilities.name);
    assert_eq!(compact(web_arguments[3]), readback);
    assert!(
        capabilities.call_start < web_physical.call_start
            && manifest_bytes.call_start < web_physical.call_start,
        "wasm begin did not bind actual manifest/capability inputs before the host seam",
    );
    let session = compact(braced_item(begin, "BrowserVisionStackSession {"));
    assert!(
        session.contains(&format!(
            "qkv_physical_execution:Some({})",
            web_physical.name
        )),
        "Web begin did not move the exact host-sealed physical spec into the session",
    );
    for forbidden in [
        "prepare_vision_qkv_stack_execution(",
        "plan_vision_qkv_readback_layout(",
        "bind_vision_qkv_physical_execution(",
        "ComputeDispatchLimits",
    ] {
        assert!(
            !begin.contains(forbidden),
            "wasm begin bypassed the host handoff seam via {forbidden}",
        );
    }

    let native_prepare = braced_item(NATIVE_SOURCE, "fn prepare_vision_qkv_execution(");
    let native_prepared =
        direct_call_binding(native_prepare, "prepare_vision_qkv_stack_execution(", false);
    let native_validation = direct_call_binding(native_prepare, ".validate(", true);
    let native_preflight = direct_call_binding(
        native_prepare,
        "preflight_vision_qkv_execution_allocations(",
        false,
    );
    let native_physical =
        direct_call_binding(native_prepare, "bind_vision_qkv_physical_execution(", false);
    let native_physical_arguments =
        balanced_call_arguments(native_prepare, "bind_vision_qkv_physical_execution(");
    assert_eq!(native_physical_arguments.len(), 2);
    assert_eq!(compact(native_physical_arguments[0]), native_prepared.name);
    assert_eq!(
        compact(native_physical_arguments[1]),
        format!("{}.layout", native_preflight.name),
    );
    assert!(
        native_prepared.call_start < native_validation.call_start
            && native_validation.call_start < native_preflight.call_start
            && native_preflight.call_start < native_physical.call_start,
        "native shared preparation, validation, preflight, and physical binding order drifted",
    );
    let native_result = compact(braced_item(native_prepare, "PreparedVisionQkvExecution {"));
    assert!(
        native_result.contains(&format!(
            "qkv_physical_execution:Some({})",
            native_physical.name
        )),
        "native did not move the exact bound physical spec into executor state",
    );
    let native_readback = braced_item(
        NATIVE_SOURCE,
        "fn preflight_vision_qkv_execution_allocations(",
    );
    let native_layout =
        direct_call_binding(native_readback, "plan_vision_qkv_readback_layout(", false);
    let compatibility_result =
        braced_item(native_readback, "PreparedVisionQkvExecutionAllocations {");
    assert_eq!(
        compact(struct_field_initializer(compatibility_result, "layout").unwrap()),
        native_layout.name,
        "native compatibility adapter did not move the exact common layout binding",
    );

    for (adapter, source) in [("Web", WEB_SOURCE), ("native", NATIVE_SOURCE)] {
        let normalized = compact(source);
        for call in [
            "prepare_vision_qkv_stack_execution(",
            "plan_vision_qkv_readback_layout(",
            ".validate(",
        ] {
            assert!(
                !normalized.contains(&format!("let_={call}")),
                "{adapter} discarded shared authority call {call}",
            );
        }
    }
}

fn assert_borrowed_vision_qkv_web_bind_group_getter(source: &str) {
    let live = live_rust_source(source);
    let functions = source_functions(&live);
    let group_lookup = functions
        .iter()
        .filter(|function| function.name == "get_vision_qkv_web_bind_group")
        .collect::<Vec<_>>();
    assert_eq!(group_lookup.len(), 1);
    let group_lookup_header = compact(function_header(*group_lookup[0]));
    assert!(
        group_lookup_header.contains("<'a>(groups:&'aBrowserVisionQkvLayerBindGroups")
            && group_lookup_header.ends_with("->&'awgpu::BindGroup"),
        "QKV group getter must return a lifetime-bound borrow from the operation-local holder",
    );
    assert_eq!(occurrences(group_lookup[0].body, ".get("), 1);
    let lookup_key = balanced_call_arguments(group_lookup[0].body, ".get(");
    assert_eq!(lookup_key.len(), 1);
    assert_eq!(compact(lookup_key[0]), "&(layer_index,kind)");
    for forbidden in [
        "unwrap_or(",
        "or_else(",
        "FusedQkv=>Attention",
        "Attention=>FusedQkv",
        ".clone(",
        ".cloned(",
        ".copied(",
        ".to_owned(",
        ".into_owned(",
    ] {
        assert!(
            !compact(group_lookup[0].body).contains(forbidden),
            "borrowed QKV group getter contains ownership/fallback bypass {forbidden}",
        );
    }
}

#[test]
fn borrowed_qkv_group_getter_scanner_rejects_owned_and_wrong_key_decoys() {
    const VALID: &str = r#"
fn get_vision_qkv_web_bind_group<'a>(
    groups: &'a BrowserVisionQkvLayerBindGroups,
    layer_index: u32,
    kind: VisionQkvWebBindGroupKind,
) -> &'a wgpu::BindGroup {
    groups.bind_groups.get(&(layer_index, kind)).expect("exact local group")
}
"#;
    assert_borrowed_vision_qkv_web_bind_group_getter(VALID);
    let commented = VALID.replace(
        "groups.bind_groups.get(&(layer_index, kind))",
        "/* groups.bind_groups.get(&(layer_index, kind)).clone() */ groups.bind_groups.get(&(layer_index, kind))",
    );
    assert_borrowed_vision_qkv_web_bind_group_getter(&commented);
    for (label, hostile) in [
        (
            "owned clone return",
            VALID
                .replace("-> &'a wgpu::BindGroup", "-> wgpu::BindGroup")
                .replace(
                    ".expect(\"exact local group\")",
                    ".expect(\"exact local group\").clone()",
                ),
        ),
        (
            "to_owned return",
            VALID.replace(
                ".expect(\"exact local group\")",
                ".expect(\"exact local group\").to_owned()",
            ),
        ),
        (
            "attention fallback",
            VALID.replace(
                ".expect(\"exact local group\")",
                ".or_else(|| groups.bind_groups.get(&(layer_index, VisionQkvWebBindGroupKind::Attention))).expect(\"fallback\")",
            ),
        ),
        (
            "wrong layer key",
            VALID.replace("&(layer_index, kind)", "&(layer_index + 1, kind)"),
        ),
    ] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                assert_borrowed_vision_qkv_web_bind_group_getter(&hostile);
            }))
            .is_err(),
            "borrowed group getter scanner accepted {label}",
        );
    }
}

#[test]
fn web_and_native_store_core_authority_and_physical_adapters_consume_accessors() {
    let live_web = live_rust_source(WEB_SOURCE);
    let live_lib = live_rust_source(LIB_SOURCE);
    let web_session = compact(braced_item(&live_web, "struct BrowserVisionStackSession"));
    assert!(
        web_session.contains("qkv_physical_execution:Option<VisionQkvPhysicalExecutionSpec>",),
        "Web session must own the exact sealed passes physical authority",
    );
    assert!(
        web_session.contains("qkv_physical_commands:Option<VisionQkvWebPhysicalCommandPlan>"),
        "Web session must own the exact immutable typed command plan derived from the sealed spec",
    );
    assert!(!web_session.contains("qkv_execution:Option<PreparedVisionQkvStackExecution>"));
    assert!(!web_session.contains("qkv_readback_layout:Option<VisionQkvReadbackLayout>"));
    let native_execution = compact(braced_item(
        NATIVE_SOURCE,
        "struct PreparedVisionQkvExecution",
    ));
    assert!(
        native_execution.contains("qkv_physical_execution:Option<VisionQkvPhysicalExecutionSpec>",),
        "native prepared state must own the exact sealed passes physical authority",
    );
    assert!(!native_execution.contains("qkv_execution:Option<PreparedVisionQkvStackExecution>"));
    assert!(!native_execution.contains("readback_layout:VisionQkvReadbackLayout"));
    for forbidden in [
        "struct PreparedVisionQkvWorkspace",
        "struct PreparedVisionQkvLayer",
        "struct VisionQkvReadbackPlan",
    ] {
        assert!(
            !WEB_SOURCE.contains(forbidden),
            "Web duplicated {forbidden}"
        );
        assert!(
            !NATIVE_SOURCE.contains(forbidden),
            "native duplicated {forbidden}"
        );
    }

    assert_eq!(
        occurrences(
            &format!("{live_lib}\n{live_web}"),
            "bind_vision_qkv_physical_execution(",
        ),
        1,
        "host Web seam must bind one physical authority and wasm must never reconstruct it",
    );
    assert_eq!(
        occurrences(NATIVE_SOURCE, "bind_vision_qkv_physical_execution("),
        1,
        "native must bind one physical authority at preflight and never reconstruct it",
    );

    let begin = braced_item(
        &live_web,
        "fn begin_vision_stack_sharded_with_qkv_selection(",
    );
    let sealed = direct_call_binding(begin, "prepare_vision_qkv_stack_handoff_execution(", false);
    let commands = plain_call_binding(begin, "plan_vision_qkv_web_physical_commands(");
    assert!(sealed.statement_end < commands.call_start);
    let command_arguments =
        balanced_call_arguments(begin, "plan_vision_qkv_web_physical_commands(");
    assert_eq!(command_arguments.len(), 1);
    assert_eq!(compact(command_arguments[0]), format!("&{}", sealed.name));
    let session_initializer = compact(braced_item(begin, "BrowserVisionStackSession {"));
    assert!(session_initializer.contains(&format!("qkv_physical_execution:Some({})", sealed.name)));
    assert!(
        session_initializer.contains(&format!("qkv_physical_commands:Some({})", commands.name))
    );
    assert_eq!(
        occurrences(begin, "plan_vision_qkv_web_physical_commands("),
        1,
        "optimized begin must derive exactly one typed plan from the exact sealed spec",
    );
    assert_eq!(
        occurrences(&live_lib, "pub fn plan_vision_qkv_web_physical_commands("),
        1,
        "host-compilable typed plan factory must have one definition",
    );

    assert_typed_web_physical_adapter_source(&live_web);
    assert_typed_web_physical_storage_source(&live_web);
    assert_typed_web_physical_resolver_source(&live_web);
    assert_typed_web_physical_executor_source(&live_lib);
    assert_web_physical_effect_sink_source(&live_web);
    let functions = source_functions(&live_web);
    for (operation, phase_helper) in [
        (
            "allocate_vision_stack_gpu",
            "apply_vision_qkv_web_start_commands(",
        ),
        (
            "run_vision_stack_sharded_layer_once",
            "apply_vision_qkv_web_layer_commands(",
        ),
        (
            "finish_vision_stack_sharded_once",
            "apply_vision_qkv_web_finish_commands(",
        ),
    ] {
        let reachable = reachable_functions(&live_web, operation);
        assert_eq!(
            reachable
                .iter()
                .map(|function| source_call_occurrences_named(
                    function.body,
                    phase_helper.trim_end_matches('(')
                ))
                .sum::<usize>(),
            1,
            "real Web {operation} is not linked exactly once to {phase_helper}",
        );
    }

    let gpu_state = compact(braced_item(&live_web, "struct BrowserVisionStackGpuState"));
    let layer_groups = braced_item(&live_web, "struct BrowserVisionQkvLayerBindGroups");
    let layer_groups_compact = compact(layer_groups);
    for persistent in [&web_session, &gpu_state] {
        assert!(
            !persistent.contains("BrowserVisionQkvLayerBindGroups")
                && !persistent.contains("qkv_bind_groups")
                && !persistent.contains("qkv_layer_bind_groups"),
            "QKV layer bind groups escaped operation-local residency",
        );
    }
    for forbidden in ["Clone", "Rc<", "RefCell", "Arc<"] {
        assert!(
            !layer_groups_compact.contains(forbidden),
            "operation-local QKV bind-group holder gained shared/clone ownership via {forbidden}",
        );
    }

    let fused_layer_functions = functions
        .iter()
        .filter(|function| {
            function
                .body
                .contains("apply_vision_qkv_web_layer_commands(")
                && function.body.contains("VisionQkvSelectionOutcome::Fused")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        fused_layer_functions.len(),
        1,
        "real fused layer effect must have one typed command/dispatch authority",
    );
    let fused_layer = fused_layer_functions[0];
    let fused_arm = braced_item(fused_layer.body, "VisionQkvSelectionOutcome::Fused =>");
    let local_groups = plain_call_binding(fused_arm, "BrowserVisionQkvLayerBindGroups::new(");
    let phase_arguments =
        balanced_call_arguments(fused_arm, "apply_vision_qkv_web_layer_commands(");
    assert!(
        phase_arguments
            .iter()
            .any(|argument| compact(argument) == format!("&mut{}", local_groups.name)),
        "typed layer command consumer does not store into the exact operation-local holder",
    );
    let context = braced_item(fused_arm, "BrowserVisionQkvLayerResolutionContext {");
    let context_name =
        named_initializer_binding(fused_arm, "BrowserVisionQkvLayerResolutionContext {");
    for (field, exact) in [
        ("norm1_output", "scratch[0]"),
        ("query_weight", "weight_bindings[2]"),
        ("query_bias", "weight_bindings[3]"),
        ("key_weight", "weight_bindings[4]"),
        ("key_bias", "weight_bindings[5]"),
        ("value_weight", "weight_bindings[6]"),
        ("value_bias", "weight_bindings[7]"),
        ("cu_seqlens", "boundary"),
        ("attention_output", "scratch[4]"),
        ("uniform_buffer", "&gpu.uniform_buffer"),
        ("uniform_stride", "gpu.uniform_stride"),
    ] {
        assert_eq!(
            compact(
                struct_field_initializer(context, field)
                    .unwrap_or_else(|| panic!("layer resolution context omitted {field}")),
            ),
            exact,
            "layer resolution context mapped {field} to the wrong physical binding",
        );
    }
    let context_parameter_index = assert_web_layer_physical_context_wiring(&live_web);
    assert_web_layer_context_call_argument(fused_arm, context_parameter_index, context_name);
    let lookups = all_balanced_call_arguments(fused_arm, "get_vision_qkv_web_bind_group(");
    assert_eq!(
        lookups.len(),
        2,
        "fused branch must fetch exactly two typed groups"
    );
    for (arguments, kind) in lookups.iter().zip(["FusedQkv", "Attention"]) {
        assert_eq!(arguments.len(), 3);
        assert_eq!(compact(arguments[0]), format!("&{}", local_groups.name));
        assert_eq!(compact(arguments[1]), "layer");
        assert_eq!(
            compact(arguments[2]),
            format!("VisionQkvWebBindGroupKind::{kind}"),
        );
    }
    let fused_compact = compact(fused_arm);
    for (name, kind) in [
        ("fused_qkv_bind_group", "FusedQkv"),
        ("attention_bind_group", "Attention"),
    ] {
        assert!(
            fused_compact.contains(&format!(
                "let{name}=get_vision_qkv_web_bind_group(&{},layer,VisionQkvWebBindGroupKind::{kind})",
                local_groups.name,
            )),
            "fused branch did not bind exact {kind} lookup to {name}",
        );
        let set_calls = all_balanced_call_arguments(fused_arm, "set_bind_group(")
            .into_iter()
            .filter(|arguments| {
                arguments
                    .get(1)
                    .is_some_and(|argument| compact(argument) == name)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            set_calls.len(),
            1,
            "{kind} lookup is not bound exactly once"
        );
        assert_eq!(compact(set_calls[0][0]), "0");
        assert_eq!(compact(set_calls[0][2]), "&[]");
    }
    assert_order(
        fused_arm,
        &[
            "KernelId::VisionQkvFusedF32",
            "fused_qkv_bind_group",
            "dispatch_workgroups(",
            "KernelId::VisionAttentionF32",
            "attention_bind_group",
            "dispatch_workgroups(",
        ],
    );
    for forbidden in [
        "VisionEncoderLayerStage::Query",
        "VisionEncoderLayerStage::Key",
        "VisionEncoderLayerStage::Value",
        "vision_q_proj",
        "vision_k_proj",
        "vision_v_proj",
    ] {
        assert!(
            !fused_arm.contains(forbidden),
            "fused branch retained legacy projection work via {forbidden}",
        );
    }
    let drop_groups = fused_arm
        .find(&format!("drop({})", local_groups.name))
        .expect("fused branch does not explicitly drop operation-local QKV groups");
    assert!(
        !fused_layer
            .body
            .contains("destroy_vision_qkv_web_layer_weights("),
        "shared encoder destroyed streaming-owned weights",
    );
    let legacy_layer = functions
        .iter()
        .find(|function| function.name == "execute_vision_stack_layer")
        .expect("missing legacy layer owner");
    let legacy_encode =
        all_balanced_call_arguments(legacy_layer.body, "encode_and_submit_vision_stack_layer(");
    let legacy_await = all_balanced_call_arguments(legacy_layer.body, "await_queue_completion(");
    let legacy_destroy =
        all_balanced_call_arguments(legacy_layer.body, "destroy_vision_qkv_web_layer_weights(");
    assert_eq!(
        legacy_encode.len(),
        1,
        "legacy layer encoded more than once"
    );
    assert_eq!(
        legacy_await.len(),
        1,
        "legacy layer awaited more than one queue completion",
    );
    assert_eq!(
        legacy_destroy.len(),
        1,
        "legacy layer must release exactly one current-layer weight set",
    );
    assert_eq!(
        legacy_encode[0]
            .iter()
            .map(|argument| compact(argument))
            .collect::<Vec<_>>(),
        ["session", "layer", "checkpoint_slot", "&weights"],
        "legacy layer did not encode its exact created weight set",
    );
    assert_eq!(
        legacy_await[0]
            .iter()
            .map(|argument| compact(argument))
            .collect::<Vec<_>>(),
        ["&self.queue"],
        "legacy layer did not await the execution queue",
    );
    assert_eq!(
        legacy_destroy[0]
            .iter()
            .map(|argument| compact(argument))
            .collect::<Vec<_>>(),
        ["&weights"],
        "legacy layer did not release the same weight set it encoded",
    );
    assert_order(
        legacy_layer.body,
        &[
            "encode_and_submit_vision_stack_layer(",
            "await_queue_completion(",
            "destroy_vision_qkv_web_layer_weights(",
        ],
    );
    let submit = fused_arm
        .find("submit_command_buffers(")
        .expect("fused shared encoder did not submit its command buffer");
    assert!(
        submit < drop_groups,
        "operation-local QKV groups were released before submission",
    );

    assert_borrowed_vision_qkv_web_bind_group_getter(WEB_SOURCE);
    let live_web_compact = compact(&live_web);
    for forbidden in [
        ".prepared_execution(",
        ".readback_layout(",
        ".allocation_bytes(",
        ".qkv_canary_offset(",
        ".qkv_canary_readback_bytes(",
        "prepare_vision_qkv_stack_execution(",
        "plan_vision_qkv_readback_layout(",
        "plan_vision_qkv_fused_geometry(",
    ] {
        assert!(
            !live_web_compact.contains(forbidden),
            "Web bypasses the typed command plan through raw authority {forbidden}",
        );
    }

    let native_execute = reachable_functions(NATIVE_SOURCE, "execute_vision_stack_once_optimized");
    let mut native_physical = Vec::new();
    native_physical.push(assert_physical_spec_sink(
        &native_execute,
        "create_buffer(",
        ".prepared_execution(",
        ".allocation_bytes(",
        "create_buffer(",
        &[1],
        None,
    ));
    native_physical.push(assert_physical_spec_sink(
        &native_execute,
        "create_buffer(",
        ".readback_layout(",
        ".total_readback_bytes(",
        "create_buffer(",
        &[1],
        None,
    ));
    native_physical.push(assert_physical_spec_sink(
        &native_execute,
        "create_bind_group(",
        ".prepared_execution(",
        ".bindings(",
        "create_bind_group(",
        &[0],
        Some("entries"),
    ));
    native_physical.push(assert_physical_spec_sink(
        &native_execute,
        "copy_buffer_to_buffer(",
        ".readback_layout(",
        ".qkv_canary_offset(",
        "copy_buffer_to_buffer(",
        &[3],
        None,
    ));
    native_physical.push(assert_physical_spec_sink(
        &native_execute,
        "copy_buffer_to_buffer(",
        ".prepared_execution(",
        ".canaries(",
        "copy_buffer_to_buffer(",
        &[1, 4],
        None,
    ));
    native_physical.push(assert_physical_spec_sink(
        &native_execute,
        "get_mapped_range(",
        ".readback_layout(",
        ".total_readback_bytes(",
        ".slice(",
        &[0],
        None,
    ));

    for (adapter, source, roots) in [("native", NATIVE_SOURCE, native_physical)] {
        let mut reviewed = BTreeSet::new();
        for root in roots {
            for function in reachable_functions(source, root.name) {
                reviewed.insert(function.name);
            }
        }
        let functions = source_functions(source)
            .into_iter()
            .map(|function| (function.name, function))
            .collect::<BTreeMap<_, _>>();
        let physical_source = reviewed
            .iter()
            .map(|name| functions[name].body)
            .collect::<String>();
        let normalized = compact(&physical_source);
        for forbidden in [
            "PreparedVisionQkvWorkspace",
            "PreparedVisionQkvCanary",
            "VisionQkvReadbackPlan",
            "prepare_vision_qkv_stack_execution(",
            "plan_vision_qkv_readback_layout(",
            "plan_vision_qkv_fused_geometry(",
            "checked_add(",
            "checked_mul(",
            "try_fold(",
            "usize::try_from(",
            "/4",
        ] {
            assert!(
                !normalized.contains(forbidden),
                "{adapter} physical sink graph reconstructed sealed authority via {forbidden}",
            );
        }
    }
}

#[test]
fn both_runtime_adapters_import_the_one_core_qkv_canary_word() {
    for (adapter, source) in [("Web", WEB_SOURCE), ("native", NATIVE_SOURCE)] {
        assert!(
            source.contains("pvlc_runtime_core") && source.contains("VISION_QKV_CANARY_U32"),
            "{adapter} did not import the core canary authority",
        );
        assert!(
            !compact(source)
                .replace('_', "")
                .to_ascii_lowercase()
                .contains("0x7fc051a7"),
            "{adapter} retained a local QKV canary literal",
        );
    }
}

#[test]
fn disabled_selection_is_lazy_and_compile_never_performs_gpu_or_session_effects() {
    let runtime_source = live_rust_source(&format!("{LIB_SOURCE}\n{WEB_SOURCE}"));
    let reachable = reachable_functions(
        &runtime_source,
        "compile_vision_encoder_stack_qkv_selection",
    );
    let compile = unique_function_containing(&reachable, &["select_vision_qkv_stack_overlay("]);
    assert_order(
        compile.body,
        &[
            "parse_vision_stack_shard_manifest(",
            "select_vision_qkv_stack_overlay(",
        ],
    );
    let selector_binding =
        direct_call_binding(compile.body, "select_vision_qkv_stack_overlay(", false);
    let selector_arguments =
        balanced_call_arguments(compile.body, "select_vision_qkv_stack_overlay(");
    assert_eq!(selector_arguments.len(), 2);
    assert_eq!(compact(selector_arguments[0]), "policy");
    let lazy_builder = selector_arguments[1];
    assert!(
        compact(lazy_builder).starts_with("||{") || compact(lazy_builder).starts_with("move||{"),
        "compiler construction must be the selector's actual lazy closure argument",
    );
    for lazy_constructor in [
        "SemanticGraph::paddleocr_vl_16",
        "tensor_specs",
        "canonical_synthetic_vision_qkv_tensor_catalog",
        "build_verified_vision_qkv_stack_overlay",
    ] {
        assert!(
            lazy_builder.contains(lazy_constructor),
            "selector lazy closure omitted {lazy_constructor}",
        );
        assert_eq!(
            occurrences(compile.body, lazy_constructor),
            occurrences(lazy_builder, lazy_constructor),
            "{lazy_constructor} escaped the behavior-tested lazy selector closure",
        );
    }
    let handoff = braced_item(compile.body, "VisionQkvCompilerHandoff {");
    assert_eq!(
        compact(struct_field_initializer(handoff, "selection").unwrap()),
        selector_binding.name,
        "compiler handoff reconstructed the selector result instead of storing the exact lazy outcome",
    );
    let reachable_source = reachable
        .iter()
        .map(|function| function.body)
        .collect::<String>();
    for forbidden in [
        "vision_stack_session.borrow_mut()",
        "execution_busy",
        "create_buffer(",
        "queue.write_buffer(",
        "create_shader_module(",
        "push_error_scope(",
        "pipelines.borrow_mut()",
        "buffer_allocations.set(",
        "submissions.set(",
    ] {
        assert!(
            !reachable_source.contains(forbidden),
            "compiler handoff or a transitively called helper performed effect {forbidden}"
        );
    }
    assert!(
        !reachable_source.contains("prepare_vision_qkv_stack_execution("),
        "compile must not perform begin-time prepared-workspace planning",
    );
    let wasm_compile = braced_item(
        WEB_SOURCE,
        "pub fn compile_vision_encoder_stack_qkv_selection(",
    );
    assert_eq!(
        occurrences(wasm_compile, "compile_vision_qkv_stack_handoff("),
        1,
        "wasm compile must delegate exactly once to the behavior-tested host seam",
    );
    for forbidden in [
        "VisionStackShardOracle::",
        "canonical_synthetic_vision_qkv_tensor_catalog(",
        "PaddleOcrVl16Schema::tensor_specs(",
        "build_verified_vision_qkv_stack_overlay(",
    ] {
        assert!(
            !wasm_compile.contains(forbidden),
            "wasm compile duplicated host compiler routing via {forbidden}",
        );
    }
}

#[test]
fn optimized_begin_finishes_all_binding_and_size_checks_before_session_storage() {
    let begin = braced_item(
        WEB_SOURCE,
        "fn begin_vision_stack_sharded_with_qkv_selection(",
    );
    assert_order(
        begin,
        &[
            ".as_bytes(",
            "&qkv_selection.handoff",
            "vision_qkv_compiler_capabilities(",
            "VisionQkvCompilerReadbackRequest {",
            "prepare_vision_qkv_stack_handoff_execution(",
            "vision_stack_qkv_status_json(",
            ".begin(session)",
            "execution_busy.set(true)",
        ],
    );
    assert_eq!(
        occurrences(begin, ".begin("),
        1,
        "optimized begin must commit exactly one session generation",
    );
    assert_eq!(
        occurrences(begin, "execution_busy.set(true)"),
        1,
        "optimized begin must set busy exactly once",
    );
    let before_store = begin
        .split(".begin(session)")
        .next()
        .expect("session storage boundary");
    for required in [
        "qkv_selection.handoff",
        "vision_qkv_compiler_capabilities(",
        "VisionQkvCompilerReadbackRequest",
        "semantic_readback_bytes",
        "scratch_canary_readback_bytes",
        "prepare_vision_qkv_stack_handoff_execution(",
    ] {
        assert!(
            before_store.contains(required),
            "begin did not preflight {required}"
        );
    }
    for forbidden in [
        ".begin(",
        ".complete(",
        ".abort(",
        "stored_mut(",
        "execution_busy.set(",
        "pipelines.borrow_mut()",
        "buffer_allocations.set(",
        "submissions.set(",
    ] {
        assert!(
            !before_store.contains(forbidden),
            "optimized begin mutated session state before commit via {forbidden}",
        );
    }
    for forbidden in [
        "create_buffer(",
        "queue.write_buffer(",
        "create_shader_module(",
        "submit(",
    ] {
        assert!(
            !begin.contains(forbidden),
            "optimized begin performed WebGPU effect {forbidden}"
        );
    }
    let runtime_source = format!("{LIB_SOURCE}\n{WEB_SOURCE}");
    let reachable = reachable_functions(
        &runtime_source,
        "begin_vision_stack_sharded_with_qkv_selection",
    );
    let reachable_source = reachable
        .iter()
        .map(|function| function.body)
        .collect::<String>();
    let handoff_preflight = unique_function_containing(
        &reachable,
        &[
            "prepare_vision_qkv_stack_execution(",
            "ComputeDispatchLimits",
            "plan_vision_qkv_readback_layout(",
            "bind_vision_qkv_physical_execution(",
        ],
    );
    for required in [
        "max_compute_workgroup_size",
        "max_compute_invocations_per_workgroup",
        "max_compute_workgroups_per_dimension",
        ".validate(",
        "semantic_readback_bytes",
        "scratch_canary_readback_bytes",
        "qkv_canary_readback_bytes",
        "workspace_allocation_bytes",
        "max_buffer_size",
        "max_host_elements",
    ] {
        assert!(
            handoff_preflight.body.contains(required),
            "optimized begin host handoff preflight omitted {required}",
        );
    }
    for forbidden in [
        "create_buffer(",
        "queue.write_buffer(",
        "create_shader_module(",
        "push_error_scope(",
        "submit_command_buffers(",
        "queue.submit(",
        "map_async(",
        "pipelines.borrow_mut()",
        "buffer_allocations.set(",
        "submissions.set(",
    ] {
        assert!(
            !reachable_source.contains(forbidden),
            "optimized begin transitively performed WebGPU effect {forbidden}",
        );
    }
    let hidden_helpers = reachable
        .iter()
        .filter(|function| function.name != "begin_vision_stack_sharded_with_qkv_selection")
        .map(|function| function.body)
        .collect::<String>();
    for forbidden in [
        "vision_stack_session",
        "execution_busy",
        "AsyncSessionOwner",
        "borrow_mut()",
        ".begin(",
        ".complete(",
    ] {
        assert!(
            !hidden_helpers.contains(forbidden),
            "optimized begin helper hid session-owner mutation {forbidden}",
        );
    }
}

fn ast_type_is_exact_path(
    ty: &syn::Type,
    expected_path: &str,
    expected_lifetime: Option<&str>,
) -> bool {
    let syn::Type::Path(path) = ty else {
        return false;
    };
    if path.qself.is_some() || ast_path_name(&path.path) != expected_path {
        return false;
    }
    let Some(last) = path.path.segments.last() else {
        return false;
    };
    if path
        .path
        .segments
        .iter()
        .take(path.path.segments.len().saturating_sub(1))
        .any(|segment| !matches!(segment.arguments, syn::PathArguments::None))
    {
        return false;
    }
    match (expected_lifetime, &last.arguments) {
        (None, syn::PathArguments::None) => true,
        (Some(expected), syn::PathArguments::AngleBracketed(arguments))
            if arguments.args.len() == 1 =>
        {
            matches!(arguments.args.first().unwrap(), syn::GenericArgument::Lifetime(lifetime)
                if lifetime.ident == expected)
        }
        _ => false,
    }
}

#[test]
fn mapped_readback_has_one_range_and_unmaps_every_successful_mapping_exit() {
    let runtime_source = format!("{LIB_SOURCE}\n{WEB_SOURCE}");
    let reachable = reachable_functions(&runtime_source, "finish_vision_stack_sharded");
    let finish = unique_function_containing(&reachable, &["get_mapped_range(", "qkv", "canar"]);
    assert_eq!(
        occurrences(finish.body, "map_read(") + occurrences(finish.body, "map_async("),
        1
    );
    assert_eq!(occurrences(finish.body, "get_mapped_range("), 1);
    assert!(
        !compact(finish.body).contains("get_mapped_range(..)"),
        "mapped range must be explicit and exact"
    );
    let map_call = if finish.body.contains("map_read(") {
        "map_read("
    } else {
        "map_async("
    };
    assert_eq!(
        occurrences(finish.body, "let map_result ="),
        1,
        "finish must bind exactly one real mapping outcome",
    );
    let map_binding = finish.body.find("let map_result =").unwrap();
    let map = finish.body.find(map_call).unwrap();
    let map_end = balanced_call(finish.body, map_call).1;
    let map_statement_end = finish.body[map_end..]
        .find(';')
        .map(|offset| map_end + offset)
        .expect("map result binding must end in a statement");
    assert!(
        map_binding < map,
        "finish does not bind the real mapping call outcome",
    );
    if map_call == "map_read(" {
        let equals = finish.body[map_binding..map]
            .find('=')
            .map(|offset| map_binding + offset)
            .unwrap();
        assert!(
            finish.body[equals + 1..map].trim().is_empty(),
            "finish wraps or reconstructs the map_read outcome before binding",
        );
    }
    assert_eq!(
        compact(&finish.body[map_end..map_statement_end]),
        ".await",
        "finish must bind the direct awaited mapping outcome without ?, Ok, or reconstruction",
    );
    let mapped_range = finish.body.find("get_mapped_range(").unwrap();
    assert!(map < mapped_range);
    let cleanup = finish
        .body
        .find("with_vision_stack_mapped_readback(")
        .expect("mapped access must use the behavior-tested cleanup scope");
    assert!(
        map_statement_end < cleanup && cleanup < mapped_range,
        "cleanup scope must start after the map attempt and enclose getMappedRange",
    );
    assert_eq!(
        occurrences(finish.body, "with_vision_stack_mapped_readback("),
        1
    );
    let cleanup_arguments =
        balanced_call_arguments(finish.body, "with_vision_stack_mapped_readback(");
    assert_eq!(
        cleanup_arguments.len(),
        3,
        "mapped cleanup must receive map result, unmap closure, and access closure",
    );
    assert_eq!(
        compact(cleanup_arguments[0]),
        "map_result",
        "mapped cleanup must consume the exact bound map outcome, not Ok or reconstruction",
    );
    assert!(
        cleanup_arguments[1].contains(".unmap()")
            && !cleanup_arguments[1].contains("get_mapped_range("),
        "sole physical unmap is not supplied as the cleanup argument",
    );
    assert!(
        cleanup_arguments[2].contains("get_mapped_range(")
            && !cleanup_arguments[2].contains(".unmap()"),
        "getMappedRange is not lexically enclosed by the access-and-verify argument",
    );
    let reachable_source = reachable
        .iter()
        .map(|function| function.body)
        .collect::<String>();
    assert_eq!(
        occurrences(&reachable_source, ".unmap()"),
        1,
        "real finish call graph must supply exactly one physical unmap to the cleanup scope",
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MappedProbeError {
    MapRejected,
    GetMappedRange,
    Verification,
}

#[test]
fn mapped_cleanup_scope_unmaps_exactly_once_for_every_post_map_exit() {
    for (label, map_result, body_result, expected_body_calls, expected_unmaps, expected_events) in [
        ("success", Ok(()), Ok(7_u32), 1, 1, &["access", "unmap"][..]),
        (
            "getMappedRange failure",
            Ok(()),
            Err(MappedProbeError::GetMappedRange),
            1,
            1,
            &["access", "unmap"][..],
        ),
        (
            "verification failure",
            Ok(()),
            Err(MappedProbeError::Verification),
            1,
            1,
            &["access", "unmap"][..],
        ),
        (
            "map rejection",
            Err(MappedProbeError::MapRejected),
            Ok(7_u32),
            0,
            0,
            &[][..],
        ),
    ] {
        let body_calls = Cell::new(0_u32);
        let unmaps = Cell::new(0_u32);
        let events = RefCell::new(Vec::new());
        let result = with_vision_stack_mapped_readback(
            map_result,
            || {
                unmaps.set(unmaps.get() + 1);
                events.borrow_mut().push("unmap");
            },
            || {
                body_calls.set(body_calls.get() + 1);
                events.borrow_mut().push("access");
                body_result
            },
        );
        assert_eq!(body_calls.get(), expected_body_calls, "{label}");
        assert_eq!(unmaps.get(), expected_unmaps, "{label}");
        assert_eq!(&*events.borrow(), expected_events, "{label}");
        assert_eq!(result, map_result.and(body_result), "{label}");
    }

    let unmaps = Cell::new(0_u32);
    let events = RefCell::new(Vec::new());
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = with_vision_stack_mapped_readback(
            Ok::<(), MappedProbeError>(()),
            || {
                unmaps.set(unmaps.get() + 1);
                events.borrow_mut().push("unmap");
            },
            || -> Result<(), MappedProbeError> {
                events.borrow_mut().push("access");
                panic!("mapped verification panic")
            },
        );
    }));
    assert!(panic.is_err());
    assert_eq!(unmaps.get(), 1, "panic after map success must still unmap");
    assert_eq!(
        &*events.borrow(),
        &["access", "unmap"],
        "panic cleanup ran before mapped access or more than once",
    );
}

#[test]
fn optimized_source_report_is_exactly_six_kernels_while_legacy_remains_five() {
    let legacy = bracketed_item(WEB_SOURCE, "const VISION_LAYER_KERNELS:");
    assert_eq!(occurrences(legacy, "KernelId::"), 5);
    assert!(!legacy.contains("VisionQkvFusedF32"));

    let optimized = bracketed_item(WEB_SOURCE, "const VISION_QKV_STACK_KERNELS:");
    assert_eq!(occurrences(optimized, "KernelId::"), 6);
    for kernel in [
        "AddF32",
        "GeluTanhF32",
        "LayerNormF32",
        "VisionAttentionF32",
        "VisionPatchProjectionF32",
        "VisionQkvFusedF32",
    ] {
        assert!(
            optimized.contains(kernel),
            "optimized source set lost {kernel}"
        );
    }
}
