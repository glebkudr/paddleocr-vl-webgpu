//! Sealed persistent browser decoder KV session authority.
//!
//! The authority privately owns the exact `wgpu::Device`, the exact `wgpu::Queue`,
//! and one `crate::AsyncSessionOwner<BrowserDecoderKvSession>`. Every operation
//! validates its inputs and the exact core plan before the first GPU effect,
//! pushes the three checked error scopes, executes the exact phase topology with
//! raw WebGPU calls that surface every thrown error as a `Result`, and drains the
//! scopes LIFO. Any post-effect failure poisons the stored session terminally; a
//! cancelled generation drains its scopes only after the newer in-flight lease
//! clears. No `unsafe`, no macros, no host-side compute shadow.

use pvlc_runtime_core::{DecoderKvSessionDescriptor, DecoderKvSessionStep};
use serde_json::{Map, Value};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

use super::ScopeKind;
use crate::vision_stack_causal::{
    VisionStackErrorScopePopAttempt, observe_vision_stack_error_scope_pop,
};
use crate::{AsyncSessionOwner, CompletionAction};

const CHECKED_SCOPES: [ScopeKind; 3] = [
    ScopeKind::Internal,
    ScopeKind::OutOfMemory,
    ScopeKind::Validation,
];
const CHECKED_SCOPE_NAMES: [&str; 3] = ["validation", "out_of_memory", "internal"];
const APPEND_KERNEL_NAME: &str = "decoder_kv_append_f32";
const ATTENTION_KERNEL_NAME: &str = "decoder_gqa_f32";
const ENTRY_POINT: &str = "main";
const BUFFER_QUERY: &str = "decoder-kv-session-query";
const BUFFER_APPENDED_KEY: &str = "decoder-kv-session-appended-key";
const BUFFER_APPENDED_VALUE: &str = "decoder-kv-session-appended-value";
const BUFFER_KEY_CACHE: &str = "decoder-kv-session-key-cache";
const BUFFER_VALUE_CACHE: &str = "decoder-kv-session-value-cache";
const BUFFER_ATTENTION_OUTPUT: &str = "decoder-kv-session-attention-output";
const BUFFER_APPEND_UNIFORM: &str = "decoder-kv-session-append-uniform";
const BUFFER_ATTENTION_UNIFORM: &str = "decoder-kv-session-attention-uniform";
const BUFFER_ATTENTION_READBACK: &str = "decoder-kv-session-attention-readback";
const BUFFER_FINISH_READBACK: &str = "decoder-kv-session-finish-readback";
const DESCRIPTOR_FIELDS: [&str; 6] = [
    "schema_version",
    "query_heads",
    "key_value_heads",
    "head_dim",
    "prefix_tokens",
    "cache_capacity",
];

/// Sealed owner of the persistent decoder KV session lifecycle.
pub(super) struct DecoderKvSessionAuthority {
    device: wgpu::Device,
    queue: wgpu::Queue,
    owner: crate::AsyncSessionOwner<BrowserDecoderKvSession>,
}

struct BrowserDecoderKvSession {
    plan: pvlc_runtime_core::DecoderKvSessionPlan,
    cache_tokens: u32,
    poisoned: bool,
    ready: bool,
    append_shader_blake3: [u8; 32],
    attention_shader_blake3: [u8; 32],
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

struct DecoderShaderSources {
    append: String,
    attention: String,
}

struct StepOperands {
    query_bytes: Vec<u8>,
    appended_key_bytes: Vec<u8>,
    appended_value_bytes: Vec<u8>,
}

struct ParsedDescriptor {
    query_heads: u32,
    key_value_heads: u32,
    head_dim: u32,
    prefix_tokens: u32,
    cache_capacity: u32,
}

fn js_decoder_error(parts: &[&str]) -> JsValue {
    let mut message = String::new();
    for part in parts {
        message.push_str(part);
    }
    super::js_error(message)
}

fn js_error_text(value: &JsValue) -> String {
    if let Some(message) = js_sys::Reflect::get(value, &JsValue::from_str("message"))
        .ok()
        .and_then(|message| message.as_string())
    {
        return message;
    }
    if let Some(text) = value.as_string() {
        return text;
    }
    "unclassified WebGPU error".to_owned()
}

fn raw_decoder_device(device: &wgpu::Device) -> Result<&JsValue, JsValue> {
    device.as_webgpu().map(AsRef::as_ref).ok_or_else(|| {
        js_decoder_error(&["decoder KV session device has no browser WebGPU handle"])
    })
}

fn raw_decoder_queue(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<JsValue, JsValue> {
    let raw_device = raw_decoder_device(device)?;
    let registered =
        js_sys::Reflect::get(raw_device, &JsValue::from_str("queue")).map_err(|error| {
            js_decoder_error(&["cannot access GPUDevice.queue: ", &js_error_text(&error)])
        })?;
    let raw_queue: &JsValue = queue.as_webgpu().map(AsRef::as_ref).ok_or_else(|| {
        js_decoder_error(&["decoder KV session queue has no browser WebGPU handle"])
    })?;
    if !js_sys::Object::is(&registered, raw_queue) {
        return Err(js_decoder_error(&[
            "decoder KV session queue handle is not the exact device queue",
        ]));
    }
    Ok(registered)
}

fn raw_decoder_method(handle: &JsValue, name: &str) -> Result<js_sys::Function, JsValue> {
    js_sys::Reflect::get(handle, &JsValue::from_str(name))
        .map_err(|error| {
            js_decoder_error(&[
                "cannot access WebGPU method ",
                name,
                ": ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<js_sys::Function>()
        .map_err(|_| js_decoder_error(&["WebGPU member ", name, " is not callable"]))
}

async fn push_decoder_error_scope(device: &wgpu::Device, scope: ScopeKind) -> Result<(), JsValue> {
    let raw = raw_decoder_device(device)?;
    let push = raw_decoder_method(raw, "pushErrorScope")?;
    push.call1(raw, &JsValue::from_str(scope.filter_str()))
        .map(|_| ())
        .map_err(|error| {
            js_decoder_error(&[
                "cannot push ",
                scope.as_str(),
                " WebGPU error scope: ",
                &js_error_text(&error),
            ])
        })
}

async fn pop_decoder_error_scope(
    device: &wgpu::Device,
    scope: ScopeKind,
) -> Result<Option<String>, JsValue> {
    let raw = raw_decoder_device(device)?;
    let pop = raw_decoder_method(raw, "popErrorScope")?;
    let invocation = pop
        .call0(raw)
        .map_err(|error| {
            js_decoder_error(&[
                "cannot invoke popErrorScope for ",
                scope.as_str(),
                " scope: ",
                &js_error_text(&error),
            ])
        })
        .and_then(|pending| {
            pending.dyn_into::<js_sys::Promise>().map_err(|_| {
                js_decoder_error(&[
                    "popErrorScope for ",
                    scope.as_str(),
                    " scope did not return a Promise",
                ])
            })
        });
    let attempt = observe_vision_stack_error_scope_pop(
        invocation,
        |promise| async move {
            JsFuture::from(promise).await.map_err(|error| {
                js_decoder_error(&[
                    "popErrorScope Promise for ",
                    scope.as_str(),
                    " scope rejected: ",
                    &js_error_text(&error),
                ])
            })
        },
        |value| {
            if value.is_null() || value.is_undefined() {
                Ok(None)
            } else {
                Ok(Some(js_error_text(&value)))
            }
        },
    )
    .await;
    match attempt {
        VisionStackErrorScopePopAttempt::Popped(result) => result,
        VisionStackErrorScopePopAttempt::NotPopped(error) => Err(error),
    }
}

async fn drain_decoder_error_scopes(
    device: &wgpu::Device,
    ledger: &mut Vec<ScopeKind>,
) -> (Vec<String>, Vec<JsValue>) {
    let mut captures = Vec::new();
    let mut failures = Vec::new();
    while let Some(scope) = ledger.pop() {
        match pop_decoder_error_scope(device, scope).await {
            Ok(Some(message)) => captures.push(message),
            Ok(None) => {}
            Err(error) => failures.push(error),
        }
    }
    (captures, failures)
}

fn drain_appended_message(
    error: JsValue,
    captures: Vec<String>,
    failures: Vec<JsValue>,
) -> JsValue {
    let mut message = js_error_text(&error);
    for captured in &captures {
        message.push_str("; captured WebGPU error: ");
        message.push_str(captured);
    }
    for failure in &failures {
        message.push_str("; scope cleanup failure: ");
        message.push_str(&js_error_text(failure));
    }
    super::js_error(message)
}

fn captured_failure_message(captures: Vec<String>, failures: Vec<JsValue>) -> JsValue {
    let mut message = String::from("decoder KV session captured WebGPU errors:");
    for captured in &captures {
        message.push_str(" [");
        message.push_str(captured);
        message.push(']');
    }
    for failure in &failures {
        message.push_str("; scope cleanup failure: ");
        message.push_str(&js_error_text(failure));
    }
    super::js_error(message)
}

async fn yield_decoder_event_loop() {
    let set_timeout =
        match js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("setTimeout")) {
            Ok(value) => match value.dyn_into::<js_sys::Function>() {
                Ok(function) => function,
                Err(_) => return,
            },
            Err(_) => return,
        };
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let _ = set_timeout.call1(&js_sys::global(), &resolve);
    });
    let _ = JsFuture::from(promise).await;
}

async fn wait_decoder_owner_idle(owner: &AsyncSessionOwner<BrowserDecoderKvSession>) {
    while owner.is_in_flight() {
        yield_decoder_event_loop().await;
    }
}

fn poison_stored_session(owner: &AsyncSessionOwner<BrowserDecoderKvSession>) {
    if let Some(mut session) = owner.stored_mut() {
        session.poisoned = true;
    }
}

fn mark_session_ready(owner: &AsyncSessionOwner<BrowserDecoderKvSession>) {
    if let Some(mut session) = owner.stored_mut() {
        session.ready = true;
    }
}

fn check_decoder_admission(
    owner: &AsyncSessionOwner<BrowserDecoderKvSession>,
) -> Result<(), JsValue> {
    if owner.stored().is_some_and(|session| session.poisoned) {
        return Err(js_decoder_error(&[
            "decoder KV session is terminally poisoned",
        ]));
    }
    if owner.is_busy() {
        return Err(js_decoder_error(&[
            "decoder KV session is already active or busy with another operation",
        ]));
    }
    Ok(())
}

fn acquire_decoder_session(
    owner: &AsyncSessionOwner<BrowserDecoderKvSession>,
) -> Result<(crate::AsyncSessionLease, BrowserDecoderKvSession), JsValue> {
    {
        let Some(session) = owner.stored() else {
            return Err(js_decoder_error(&["no ready decoder KV session"]));
        };
        if session.poisoned {
            return Err(js_decoder_error(&[
                "decoder KV session is terminally poisoned",
            ]));
        }
        if !session.ready {
            return Err(js_decoder_error(&["no ready decoder KV session"]));
        }
    }
    owner
        .acquire()
        .map_err(|_| js_decoder_error(&["no stored decoder KV session"]))
}

fn parse_decoder_descriptor_json(json: &str) -> Result<ParsedDescriptor, JsValue> {
    let value: Value = serde_json::from_str(json).map_err(|error| {
        js_decoder_error(&[
            "invalid decoder KV session descriptor json: ",
            &error.to_string(),
        ])
    })?;
    let object = value.as_object().ok_or_else(|| {
        js_decoder_error(&["invalid decoder KV session descriptor: expected an object"])
    })?;
    for key in object.keys() {
        if !DESCRIPTOR_FIELDS.contains(&key.as_str()) {
            return Err(js_decoder_error(&[
                "invalid decoder KV session descriptor: unknown field ",
                key,
            ]));
        }
    }
    let schema_version = required_descriptor_u32(object, "schema_version")?;
    if schema_version != 1 {
        return Err(js_decoder_error(&[
            "invalid decoder KV session descriptor schema version",
        ]));
    }
    Ok(ParsedDescriptor {
        query_heads: required_descriptor_u32(object, "query_heads")?,
        key_value_heads: required_descriptor_u32(object, "key_value_heads")?,
        head_dim: required_descriptor_u32(object, "head_dim")?,
        prefix_tokens: required_descriptor_u32(object, "prefix_tokens")?,
        cache_capacity: required_descriptor_u32(object, "cache_capacity")?,
    })
}

fn required_descriptor_u32(object: &Map<String, Value>, key: &str) -> Result<u32, JsValue> {
    let value = object.get(key).ok_or_else(|| {
        js_decoder_error(&["invalid decoder KV session descriptor: missing field ", key])
    })?;
    let integer = value.as_u64().ok_or_else(|| {
        js_decoder_error(&[
            "invalid decoder KV session descriptor: field ",
            key,
            " must be an unsigned integer",
        ])
    })?;
    u32::try_from(integer).map_err(|_| {
        js_decoder_error(&[
            "invalid decoder KV session descriptor: field ",
            key,
            " is out of range",
        ])
    })
}

fn decoder_uint8_to_bytes(value: &js_sys::Uint8Array) -> Result<Vec<u8>, JsValue> {
    if !value.is_instance_of::<js_sys::Uint8Array>() {
        return Err(js_decoder_error(&[
            "decoder KV operand must be a Uint8Array view",
        ]));
    }
    if value.byte_length() == 0 {
        return Ok(Vec::new());
    }
    Ok(value.to_vec())
}

fn decoder_bytes_to_f32(bytes: &[u8], label: &str) -> Result<Vec<f32>, JsValue> {
    if !bytes.len().is_multiple_of(4) {
        return Err(js_decoder_error(&[
            label,
            " bytes are not a whole little-endian f32 sequence",
        ]));
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for word in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes([word[0], word[1], word[2], word[3]]));
    }
    Ok(values)
}

fn canonical_decoder_sources() -> Result<DecoderShaderSources, JsValue> {
    let append = pvlc_wgsl::module(pvlc_runtime_core::KernelId::DecoderKvAppendF32)
        .ok_or_else(|| js_decoder_error(&["canonical decoder append kernel is missing"]))?;
    let attention = pvlc_wgsl::module(pvlc_runtime_core::KernelId::DecoderGqaF32)
        .ok_or_else(|| js_decoder_error(&["canonical decoder GQA kernel is missing"]))?;
    Ok(DecoderShaderSources {
        append: append.source.to_owned(),
        attention: attention.source.to_owned(),
    })
}

fn source_blake3(source: &str) -> [u8; 32] {
    *blake3::hash(source.as_bytes()).as_bytes()
}

fn blake3_hex(digest: &[u8; 32]) -> String {
    blake3::Hash::from_bytes(*digest).to_hex().to_string()
}

fn validate_decoder_capabilities(
    device: &wgpu::Device,
    plan: &pvlc_runtime_core::DecoderKvSessionPlan,
) -> Result<(), JsValue> {
    let limits = device.limits();
    if limits.max_storage_buffers_per_shader_stage < 4 {
        return Err(js_decoder_error(&[
            "decoder KV session requires four storage buffers per shader stage",
        ]));
    }
    let dispatch_limits = pvlc_runtime_core::ComputeDispatchLimits {
        max_workgroup_size: [
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z,
        ],
        max_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
    };
    for invocation in [plan.append_invocation, plan.attention_invocation] {
        dispatch_limits.validate(&invocation).map_err(|error| {
            js_decoder_error(&[
                "decoder KV session exceeds adapter dispatch limits: ",
                &error.to_string(),
            ])
        })?;
    }
    let key_value_row_bytes = u64::try_from(plan.key_value_width)
        .ok()
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| js_decoder_error(&["decoder KV row byte size overflowed"]))?;
    for (label, bytes) in [
        ("decoder session query", plan.attention_bytes),
        (
            "decoder session appended key/value row",
            key_value_row_bytes,
        ),
        ("decoder session compact cache", plan.cache_bytes),
        ("decoder session attention output", plan.attention_bytes),
    ] {
        if bytes > limits.max_storage_buffer_binding_size {
            return Err(js_decoder_error(&[
                label,
                " exceeds the adapter storage buffer binding limit",
            ]));
        }
    }
    let cache_readback_bytes = plan
        .cache_bytes
        .checked_mul(2)
        .ok_or_else(|| js_decoder_error(&["decoder KV cache readback byte size overflowed"]))?;
    if plan.attention_bytes > limits.max_buffer_size
        || cache_readback_bytes > limits.max_buffer_size
    {
        return Err(js_decoder_error(&[
            "decoder KV session readback exceeds the adapter buffer size limit",
        ]));
    }
    Ok(())
}

fn js_object_set(object: &js_sys::Object, key: &str, value: &JsValue) -> Result<(), JsValue> {
    js_sys::Reflect::set(object, &JsValue::from_str(key), value)
        .map(|_| ())
        .map_err(|error| {
            js_decoder_error(&[
                "cannot build WebGPU descriptor field ",
                key,
                ": ",
                &js_error_text(&error),
            ])
        })
}

fn create_decoder_buffer(
    device: &JsValue,
    label: &str,
    size: u64,
    usage: u32,
) -> Result<wgpu::webgpu::GpuBuffer, JsValue> {
    let descriptor = js_sys::Object::new();
    js_object_set(&descriptor, "label", &JsValue::from_str(label))?;
    js_object_set(&descriptor, "size", &JsValue::from_f64(size as f64))?;
    js_object_set(&descriptor, "usage", &JsValue::from_f64(f64::from(usage)))?;
    js_object_set(&descriptor, "mappedAtCreation", &JsValue::FALSE)?;
    raw_decoder_method(device, "createBuffer")?
        .call1(device, &descriptor)
        .map_err(|error| {
            js_decoder_error(&[
                "decoder KV session buffer creation failed: ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<wgpu::webgpu::GpuBuffer>()
        .map_err(|_| js_decoder_error(&["createBuffer did not return a GPUBuffer"]))
}

fn create_decoder_pipeline(
    device: &JsValue,
    kernel: &str,
    source: &str,
) -> Result<js_sys::Object, JsValue> {
    let shader_descriptor = js_sys::Object::new();
    js_object_set(&shader_descriptor, "label", &JsValue::from_str(kernel))?;
    js_object_set(&shader_descriptor, "code", &JsValue::from_str(source))?;
    let shader = raw_decoder_method(device, "createShaderModule")?
        .call1(device, &shader_descriptor)
        .map_err(|error| {
            js_decoder_error(&[
                "decoder KV shader module creation failed: ",
                &js_error_text(&error),
            ])
        })?;
    let compute = js_sys::Object::new();
    js_object_set(&compute, "module", &shader)?;
    js_object_set(&compute, "entryPoint", &JsValue::from_str(ENTRY_POINT))?;
    let pipeline_descriptor = js_sys::Object::new();
    js_object_set(&pipeline_descriptor, "label", &JsValue::from_str(kernel))?;
    js_object_set(&pipeline_descriptor, "layout", &JsValue::from_str("auto"))?;
    js_object_set(&pipeline_descriptor, "compute", &compute)?;
    raw_decoder_method(device, "createComputePipeline")?
        .call1(device, &pipeline_descriptor)
        .map_err(|error| {
            js_decoder_error(&[
                "decoder KV compute pipeline creation failed: ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<js_sys::Object>()
        .map_err(|_| js_decoder_error(&["createComputePipeline did not return an object"]))
}

fn create_decoder_bind_group(
    device: &JsValue,
    label: &str,
    pipeline: &js_sys::Object,
    entries: [(&wgpu::webgpu::GpuBuffer, u64); 5],
) -> Result<js_sys::Object, JsValue> {
    let layout = raw_decoder_method(pipeline, "getBindGroupLayout")?
        .call1(pipeline, &JsValue::from_f64(0.0))
        .map_err(|error| {
            js_decoder_error(&[
                "decoder KV bind group layout request failed: ",
                &js_error_text(&error),
            ])
        })?;
    let entry_array = js_sys::Array::new();
    for (binding, (buffer, size)) in entries.iter().enumerate() {
        let resource = js_sys::Object::new();
        js_object_set(&resource, "buffer", buffer.as_ref())?;
        js_object_set(&resource, "offset", &JsValue::from_f64(0.0))?;
        js_object_set(&resource, "size", &JsValue::from_f64(*size as f64))?;
        let entry = js_sys::Object::new();
        js_object_set(&entry, "binding", &JsValue::from_f64(binding as f64))?;
        js_object_set(&entry, "resource", &resource)?;
        entry_array.push(&entry);
    }
    let descriptor = js_sys::Object::new();
    js_object_set(&descriptor, "label", &JsValue::from_str(label))?;
    js_object_set(&descriptor, "layout", &layout)?;
    js_object_set(&descriptor, "entries", &entry_array)?;
    raw_decoder_method(device, "createBindGroup")?
        .call1(device, &descriptor)
        .map_err(|error| {
            js_decoder_error(&[
                "decoder KV bind group creation failed: ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<js_sys::Object>()
        .map_err(|_| js_decoder_error(&["createBindGroup did not return an object"]))
}

fn write_decoder_buffer(
    queue: &JsValue,
    buffer: &wgpu::webgpu::GpuBuffer,
    bytes: &[u8],
) -> Result<(), JsValue> {
    let data = js_sys::Uint8Array::from(bytes);
    raw_decoder_method(queue, "writeBuffer")?
        .call3(queue, buffer.as_ref(), &JsValue::from_f64(0.0), &data)
        .map(|_| ())
        .map_err(|error| {
            js_decoder_error(&["decoder KV queue write failed: ", &js_error_text(&error)])
        })
}

fn create_decoder_encoder(device: &JsValue, label: &str) -> Result<JsValue, JsValue> {
    let descriptor = js_sys::Object::new();
    js_object_set(&descriptor, "label", &JsValue::from_str(label))?;
    raw_decoder_method(device, "createCommandEncoder")?
        .call1(device, &descriptor)
        .map_err(|error| {
            js_decoder_error(&[
                "decoder KV command encoder creation failed: ",
                &js_error_text(&error),
            ])
        })
}

fn encode_decoder_pass(
    encoder: &JsValue,
    pipeline: &js_sys::Object,
    bind_group: &js_sys::Object,
    dispatch: [u32; 3],
) -> Result<(), JsValue> {
    let pass = raw_decoder_method(encoder, "beginComputePass")?
        .call0(encoder)
        .map_err(|error| {
            js_decoder_error(&[
                "decoder KV compute pass begin failed: ",
                &js_error_text(&error),
            ])
        })?;
    raw_decoder_method(&pass, "setPipeline")?
        .call1(&pass, pipeline)
        .map_err(|error| {
            js_decoder_error(&["decoder KV setPipeline failed: ", &js_error_text(&error)])
        })?;
    raw_decoder_method(&pass, "setBindGroup")?
        .call2(&pass, &JsValue::from_f64(0.0), bind_group)
        .map_err(|error| {
            js_decoder_error(&["decoder KV setBindGroup failed: ", &js_error_text(&error)])
        })?;
    raw_decoder_method(&pass, "dispatchWorkgroups")?
        .call3(
            &pass,
            &JsValue::from_f64(f64::from(dispatch[0])),
            &JsValue::from_f64(f64::from(dispatch[1])),
            &JsValue::from_f64(f64::from(dispatch[2])),
        )
        .map_err(|error| {
            js_decoder_error(&["decoder KV dispatch failed: ", &js_error_text(&error)])
        })?;
    raw_decoder_method(&pass, "end")?
        .call0(&pass)
        .map(|_| ())
        .map_err(|error| {
            js_decoder_error(&[
                "decoder KV compute pass end failed: ",
                &js_error_text(&error),
            ])
        })
}

fn encode_decoder_copy(
    encoder: &JsValue,
    source: &wgpu::webgpu::GpuBuffer,
    source_offset: u64,
    destination: &wgpu::webgpu::GpuBuffer,
    destination_offset: u64,
    bytes: u64,
) -> Result<(), JsValue> {
    raw_decoder_method(encoder, "copyBufferToBuffer")?
        .call5(
            encoder,
            source.as_ref(),
            &JsValue::from_f64(source_offset as f64),
            destination.as_ref(),
            &JsValue::from_f64(destination_offset as f64),
            &JsValue::from_f64(bytes as f64),
        )
        .map(|_| ())
        .map_err(|error| {
            js_decoder_error(&["decoder KV buffer copy failed: ", &js_error_text(&error)])
        })
}

fn submit_decoder_encoder(queue: &JsValue, encoder: &JsValue) -> Result<(), JsValue> {
    let command = raw_decoder_method(encoder, "finish")?
        .call0(encoder)
        .map_err(|error| {
            js_decoder_error(&[
                "decoder KV command encoder finish failed: ",
                &js_error_text(&error),
            ])
        })?;
    let commands = js_sys::Array::new();
    commands.push(&command);
    raw_decoder_method(queue, "submit")?
        .call1(queue, &commands)
        .map(|_| ())
        .map_err(|error| {
            js_decoder_error(&[
                "decoder KV queue submission failed: ",
                &js_error_text(&error),
            ])
        })
}

async fn await_decoder_queue_completion(queue: &JsValue) -> Result<(), JsValue> {
    let pending = raw_decoder_method(queue, "onSubmittedWorkDone")?
        .call0(queue)
        .map_err(|error| {
            js_decoder_error(&[
                "decoder KV queue completion request failed: ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<js_sys::Promise>()
        .map_err(|_| js_decoder_error(&["onSubmittedWorkDone did not return a Promise"]))?;
    JsFuture::from(pending).await.map(|_| ()).map_err(|error| {
        js_decoder_error(&[
            "decoder KV queue completion rejected: ",
            &js_error_text(&error),
        ])
    })
}

async fn map_decoder_buffer(buffer: &JsValue, bytes: u64) -> Result<(), JsValue> {
    let pending = raw_decoder_method(buffer, "mapAsync")?
        .call3(
            buffer,
            &JsValue::from_f64(1.0),
            &JsValue::from_f64(0.0),
            &JsValue::from_f64(bytes as f64),
        )
        .map_err(|error| {
            js_decoder_error(&["decoder KV map request failed: ", &js_error_text(&error)])
        })?
        .dyn_into::<js_sys::Promise>()
        .map_err(|_| js_decoder_error(&["mapAsync did not return a Promise"]))?;
    JsFuture::from(pending).await.map(|_| ()).map_err(|error| {
        js_decoder_error(&[
            "decoder KV buffer mapping rejected: ",
            &js_error_text(&error),
        ])
    })
}

fn read_decoder_mapped(buffer: &JsValue, bytes: u64) -> Result<Vec<u8>, JsValue> {
    let range = raw_decoder_method(buffer, "getMappedRange")?
        .call2(
            buffer,
            &JsValue::from_f64(0.0),
            &JsValue::from_f64(bytes as f64),
        )
        .map_err(|error| {
            js_decoder_error(&[
                "decoder KV mapped range read failed: ",
                &js_error_text(&error),
            ])
        })?;
    Ok(js_sys::Uint8Array::new(&range).to_vec())
}

fn unmap_decoder_buffer(buffer: &JsValue) -> Result<(), JsValue> {
    raw_decoder_method(buffer, "unmap")?
        .call0(buffer)
        .map(|_| ())
        .map_err(|error| {
            js_decoder_error(&["decoder KV buffer unmap failed: ", &js_error_text(&error)])
        })
}

fn buffer_usage(parts: &[wgpu::BufferUsages]) -> u32 {
    let mut usage = 0;
    for part in parts {
        usage |= part.bits();
    }
    usage
}

fn checked_u64_bytes(elements: usize, label: &str) -> Result<u64, JsValue> {
    u64::try_from(elements)
        .ok()
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| js_decoder_error(&[label, " byte size overflowed"]))
}

fn checked_usize(bytes: u64, label: &str) -> Result<usize, JsValue> {
    usize::try_from(bytes).map_err(|_| js_decoder_error(&[label, " is too large"]))
}

impl BrowserDecoderKvSession {
    fn create(
        device: &JsValue,
        plan: pvlc_runtime_core::DecoderKvSessionPlan,
        sources: &DecoderShaderSources,
    ) -> Result<BrowserDecoderKvSession, JsValue> {
        let storage_copy_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let cache_usage = storage_copy_dst | wgpu::BufferUsages::COPY_SRC;
        let key_value_row_bytes = checked_u64_bytes(plan.key_value_width, "decoder KV row")?;
        let query_buffer = create_decoder_buffer(
            device,
            BUFFER_QUERY,
            plan.attention_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let appended_key_buffer = create_decoder_buffer(
            device,
            BUFFER_APPENDED_KEY,
            key_value_row_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let appended_value_buffer = create_decoder_buffer(
            device,
            BUFFER_APPENDED_VALUE,
            key_value_row_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let key_cache_buffer = create_decoder_buffer(
            device,
            BUFFER_KEY_CACHE,
            plan.cache_bytes,
            buffer_usage(&[cache_usage]),
        )?;
        let value_cache_buffer = create_decoder_buffer(
            device,
            BUFFER_VALUE_CACHE,
            plan.cache_bytes,
            buffer_usage(&[cache_usage]),
        )?;
        let attention_output_buffer = create_decoder_buffer(
            device,
            BUFFER_ATTENTION_OUTPUT,
            plan.attention_bytes,
            buffer_usage(&[wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC]),
        )?;
        let append_uniform_buffer = create_decoder_buffer(
            device,
            BUFFER_APPEND_UNIFORM,
            16,
            buffer_usage(&[wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST]),
        )?;
        let attention_uniform_buffer = create_decoder_buffer(
            device,
            BUFFER_ATTENTION_UNIFORM,
            16,
            buffer_usage(&[wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST]),
        )?;
        let attention_readback_buffer = create_decoder_buffer(
            device,
            BUFFER_ATTENTION_READBACK,
            plan.attention_bytes,
            buffer_usage(&[wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST]),
        )?;
        let append_pipeline = create_decoder_pipeline(device, APPEND_KERNEL_NAME, &sources.append)?;
        let attention_pipeline =
            create_decoder_pipeline(device, ATTENTION_KERNEL_NAME, &sources.attention)?;
        let append_bind_group = create_decoder_bind_group(
            device,
            "decoder-kv-session-append-bind-group",
            &append_pipeline,
            [
                (&appended_key_buffer, key_value_row_bytes),
                (&appended_value_buffer, key_value_row_bytes),
                (&key_cache_buffer, plan.cache_bytes),
                (&value_cache_buffer, plan.cache_bytes),
                (&append_uniform_buffer, 16),
            ],
        )?;
        let attention_bind_group = create_decoder_bind_group(
            device,
            "decoder-kv-session-attention-bind-group",
            &attention_pipeline,
            [
                (&query_buffer, plan.attention_bytes),
                (&key_cache_buffer, plan.cache_bytes),
                (&value_cache_buffer, plan.cache_bytes),
                (&attention_output_buffer, plan.attention_bytes),
                (&attention_uniform_buffer, 16),
            ],
        )?;
        Ok(BrowserDecoderKvSession {
            plan,
            cache_tokens: plan.initial_cache_tokens,
            poisoned: false,
            ready: false,
            append_shader_blake3: source_blake3(&sources.append),
            attention_shader_blake3: source_blake3(&sources.attention),
            query_buffer,
            appended_key_buffer,
            appended_value_buffer,
            key_cache_buffer,
            value_cache_buffer,
            attention_output_buffer,
            append_uniform_buffer,
            attention_uniform_buffer,
            attention_readback_buffer,
            append_pipeline,
            attention_pipeline,
            append_bind_group,
            attention_bind_group,
        })
    }

    fn upload_initial_caches(
        &self,
        queue: &JsValue,
        key_cache_bytes: &[u8],
        value_cache_bytes: &[u8],
    ) -> Result<(), JsValue> {
        write_decoder_buffer(queue, &self.key_cache_buffer, key_cache_bytes)?;
        write_decoder_buffer(queue, &self.value_cache_buffer, value_cache_bytes)
    }

    fn encode_step(
        &self,
        device: &JsValue,
        queue: &JsValue,
        step_plan: &pvlc_runtime_core::DecoderKvSessionStepPlan,
        query_bytes: &[u8],
        appended_key_bytes: &[u8],
        appended_value_bytes: &[u8],
    ) -> Result<(), JsValue> {
        write_decoder_buffer(queue, &self.query_buffer, query_bytes)?;
        write_decoder_buffer(queue, &self.appended_key_buffer, appended_key_bytes)?;
        write_decoder_buffer(queue, &self.appended_value_buffer, appended_value_bytes)?;
        write_decoder_buffer(
            queue,
            &self.append_uniform_buffer,
            bytemuck::cast_slice(&step_plan.append.uniform_words),
        )?;
        write_decoder_buffer(
            queue,
            &self.attention_uniform_buffer,
            bytemuck::cast_slice(&step_plan.attention.uniform_words),
        )?;
        let encoder = create_decoder_encoder(device, "decoder-kv-session-step-encoder")?;
        encode_decoder_pass(
            &encoder,
            &self.append_pipeline,
            &self.append_bind_group,
            step_plan.append.invocation.dispatch,
        )?;
        encode_decoder_pass(
            &encoder,
            &self.attention_pipeline,
            &self.attention_bind_group,
            step_plan.attention.invocation.dispatch,
        )?;
        encode_decoder_copy(
            &encoder,
            &self.attention_output_buffer,
            0,
            &self.attention_readback_buffer,
            0,
            self.plan.attention_bytes,
        )?;
        submit_decoder_encoder(queue, &encoder)
    }

    fn encode_finish(
        &self,
        device: &JsValue,
        queue: &JsValue,
    ) -> Result<wgpu::webgpu::GpuBuffer, JsValue> {
        let readback_bytes = self
            .plan
            .cache_bytes
            .checked_mul(2)
            .ok_or_else(|| js_decoder_error(&["decoder KV finish readback overflowed"]))?;
        let readback = create_decoder_buffer(
            device,
            BUFFER_FINISH_READBACK,
            readback_bytes,
            buffer_usage(&[wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST]),
        )?;
        let encoder = create_decoder_encoder(device, "decoder-kv-session-finish-encoder")?;
        encode_decoder_copy(
            &encoder,
            &self.key_cache_buffer,
            0,
            &readback,
            0,
            self.plan.cache_bytes,
        )?;
        encode_decoder_copy(
            &encoder,
            &self.value_cache_buffer,
            0,
            &readback,
            self.plan.cache_bytes,
            self.plan.cache_bytes,
        )?;
        submit_decoder_encoder(queue, &encoder)?;
        Ok(readback)
    }
}

fn shader_blake3_json(append: &[u8; 32], attention: &[u8; 32]) -> Value {
    let mut hashes = Map::new();
    hashes.insert(
        APPEND_KERNEL_NAME.to_owned(),
        Value::from(blake3_hex(append).as_str()),
    );
    hashes.insert(
        ATTENTION_KERNEL_NAME.to_owned(),
        Value::from(blake3_hex(attention).as_str()),
    );
    Value::Object(hashes)
}

fn checked_scopes_json() -> Value {
    let mut scopes = Vec::new();
    for name in CHECKED_SCOPE_NAMES {
        scopes.push(Value::from(name));
    }
    Value::Array(scopes)
}

fn json_text(value: Value) -> Result<String, JsValue> {
    serde_json::to_string(&value).map_err(|error| {
        js_decoder_error(&["cannot serialize decoder diagnostics: ", &error.to_string()])
    })
}

fn creation_diagnostics_json(
    plan: &pvlc_runtime_core::DecoderKvSessionPlan,
    append_digest: &[u8; 32],
    attention_digest: &[u8; 32],
) -> Result<String, JsValue> {
    let mut root = Map::new();
    root.insert("schema_version".to_owned(), Value::from(1u64));
    root.insert(
        "initial_cache_tokens".to_owned(),
        Value::from(u64::from(plan.initial_cache_tokens)),
    );
    root.insert(
        "cache_capacity".to_owned(),
        Value::from(u64::from(plan.cache_capacity)),
    );
    root.insert(
        "query_heads".to_owned(),
        Value::from(u64::from(plan.query_heads)),
    );
    root.insert(
        "key_value_heads".to_owned(),
        Value::from(u64::from(plan.key_value_heads)),
    );
    root.insert("head_dim".to_owned(), Value::from(u64::from(plan.head_dim)));
    root.insert("checked_error_scopes".to_owned(), checked_scopes_json());
    root.insert("captured_errors".to_owned(), Value::Array(Vec::new()));
    root.insert(
        "shader_blake3".to_owned(),
        shader_blake3_json(append_digest, attention_digest),
    );
    root.insert("buffer_count".to_owned(), Value::from(9u64));
    root.insert("pipeline_count".to_owned(), Value::from(2u64));
    root.insert("bind_group_count".to_owned(), Value::from(2u64));
    root.insert("initial_cache_write_count".to_owned(), Value::from(2u64));
    json_text(Value::Object(root))
}

fn step_diagnostics_json(
    session: &BrowserDecoderKvSession,
    step_plan: &pvlc_runtime_core::DecoderKvSessionStepPlan,
) -> Result<String, JsValue> {
    let mut root = Map::new();
    root.insert("schema_version".to_owned(), Value::from(1u64));
    root.insert(
        "cache_tokens_before".to_owned(),
        Value::from(u64::from(step_plan.cache_tokens_before)),
    );
    root.insert(
        "cache_tokens_after".to_owned(),
        Value::from(u64::from(step_plan.cache_tokens_after)),
    );
    root.insert("checked_error_scopes".to_owned(), checked_scopes_json());
    root.insert("captured_errors".to_owned(), Value::Array(Vec::new()));
    root.insert(
        "shader_blake3".to_owned(),
        shader_blake3_json(
            &session.append_shader_blake3,
            &session.attention_shader_blake3,
        ),
    );
    root.insert("queue_write_count".to_owned(), Value::from(5u64));
    root.insert("command_encoder_count".to_owned(), Value::from(1u64));
    root.insert("compute_pass_count".to_owned(), Value::from(2u64));
    root.insert("dispatch_count".to_owned(), Value::from(2u64));
    root.insert("copy_count".to_owned(), Value::from(1u64));
    root.insert("command_buffer_count".to_owned(), Value::from(1u64));
    root.insert("submission_count".to_owned(), Value::from(1u64));
    root.insert("map_count".to_owned(), Value::from(1u64));
    root.insert(
        "readback_bytes".to_owned(),
        Value::from(session.plan.attention_bytes),
    );
    let mut effects = Vec::new();
    for (ordinal, kind) in [
        "queue_write",
        "queue_write",
        "queue_write",
        "queue_write",
        "queue_write",
        "dispatch_append",
        "dispatch_gqa",
        "copy_attention",
        "submit",
        "map_attention",
    ]
    .into_iter()
    .enumerate()
    {
        let mut effect = Map::new();
        effect.insert("ordinal".to_owned(), Value::from(ordinal as u64 + 1));
        effect.insert("kind".to_owned(), Value::from(kind));
        effects.push(Value::Object(effect));
    }
    root.insert("effects".to_owned(), Value::Array(effects));
    json_text(Value::Object(root))
}

fn finish_diagnostics_json(session: &BrowserDecoderKvSession) -> Result<String, JsValue> {
    let readback_bytes = session
        .plan
        .cache_bytes
        .checked_mul(2)
        .ok_or_else(|| js_decoder_error(&["decoder KV finish readback overflowed"]))?;
    let mut root = Map::new();
    root.insert("schema_version".to_owned(), Value::from(1u64));
    root.insert("checked_error_scopes".to_owned(), checked_scopes_json());
    root.insert("captured_errors".to_owned(), Value::Array(Vec::new()));
    root.insert("buffer_allocation_count".to_owned(), Value::from(1u64));
    root.insert("queue_write_count".to_owned(), Value::from(0u64));
    root.insert("command_encoder_count".to_owned(), Value::from(1u64));
    root.insert("compute_pass_count".to_owned(), Value::from(0u64));
    root.insert("dispatch_count".to_owned(), Value::from(0u64));
    root.insert("copy_count".to_owned(), Value::from(2u64));
    root.insert("command_buffer_count".to_owned(), Value::from(1u64));
    root.insert("submission_count".to_owned(), Value::from(1u64));
    root.insert("map_count".to_owned(), Value::from(1u64));
    root.insert("readback_bytes".to_owned(), Value::from(readback_bytes));
    json_text(Value::Object(root))
}

fn decoder_step_result(attention: Vec<u8>, diagnostics: String) -> Result<JsValue, JsValue> {
    let result = js_sys::Object::new();
    let attention_bytes = js_sys::Uint8Array::from(attention.as_slice());
    js_object_set(&result, "attention_bytes", &attention_bytes)?;
    js_object_set(
        &result,
        "diagnostics_json",
        &JsValue::from_str(&diagnostics),
    )?;
    Ok(result.into())
}

fn decoder_finish_result(
    key_cache: Vec<u8>,
    value_cache: Vec<u8>,
    diagnostics: String,
) -> Result<JsValue, JsValue> {
    let result = js_sys::Object::new();
    let key_bytes = js_sys::Uint8Array::from(key_cache.as_slice());
    let value_bytes = js_sys::Uint8Array::from(value_cache.as_slice());
    js_object_set(&result, "key_cache_bytes", &key_bytes)?;
    js_object_set(&result, "value_cache_bytes", &value_bytes)?;
    js_object_set(
        &result,
        "diagnostics_json",
        &JsValue::from_str(&diagnostics),
    )?;
    Ok(result.into())
}

async fn run_begin(
    owner: &AsyncSessionOwner<BrowserDecoderKvSession>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plan: pvlc_runtime_core::DecoderKvSessionPlan,
    key_cache_bytes: &[u8],
    value_cache_bytes: &[u8],
    sources: DecoderShaderSources,
) -> Result<String, JsValue> {
    validate_decoder_capabilities(device, &plan)?;
    let mut ledger: Vec<ScopeKind> = Vec::new();
    for scope in CHECKED_SCOPES {
        if let Err(error) = push_decoder_error_scope(device, scope).await {
            ledger.push(scope);
            let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
            return Err(drain_appended_message(error, captures, failures));
        }
        ledger.push(scope);
    }
    let raw_device = raw_decoder_device(device)?;
    let raw_queue = raw_decoder_queue(device, queue)?;
    let append_digest = source_blake3(&sources.append);
    let attention_digest = source_blake3(&sources.attention);
    let session = match BrowserDecoderKvSession::create(raw_device, plan, &sources) {
        Ok(session) => session,
        Err(error) => {
            let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    if let Err(error) =
        session.upload_initial_caches(&raw_queue, key_cache_bytes, value_cache_bytes)
    {
        let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
        return Err(drain_appended_message(error, captures, failures));
    }
    let generation = match owner.begin(session) {
        Ok(generation) => generation,
        Err(_busy) => {
            let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
            return Err(drain_appended_message(
                js_decoder_error(&["decoder KV session is already active"]),
                captures,
                failures,
            ));
        }
    };
    let mut captures = Vec::new();
    let mut failures = Vec::new();
    let mut stale = false;
    while let Some(scope) = ledger.last().copied() {
        if stale {
            wait_decoder_owner_idle(owner).await;
            stale = false;
        }
        match pop_decoder_error_scope(device, scope).await {
            Ok(Some(message)) => captures.push(message),
            Ok(None) => {}
            Err(error) => failures.push(error),
        }
        ledger.pop();
        if owner.generation() != Some(generation) {
            stale = true;
        }
    }
    if owner.generation() != Some(generation) {
        return Err(js_decoder_error(&[
            "decoder KV session begin is stale: its generation was cancelled",
        ]));
    }
    if !failures.is_empty() || !captures.is_empty() {
        poison_stored_session(owner);
        return Err(captured_failure_message(captures, failures));
    }
    mark_session_ready(owner);
    creation_diagnostics_json(&plan, &append_digest, &attention_digest)
}

fn restore_decoder_session(
    owner: &AsyncSessionOwner<BrowserDecoderKvSession>,
    lease: crate::AsyncSessionLease,
    session: BrowserDecoderKvSession,
) {
    let _ = owner.complete(lease, session, CompletionAction::Restore);
}

impl DecoderKvSessionAuthority {
    pub(super) fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            owner: crate::AsyncSessionOwner::new(),
        }
    }

    pub(super) fn abort(&self) {
        let _ = self.owner.cancel_and_release();
    }

    pub(super) fn shader_sources_json(&self) -> Result<String, JsValue> {
        let sources = canonical_decoder_sources()?;
        let append_source = serde_json::to_string(&sources.append).map_err(|error| {
            js_decoder_error(&[
                "cannot serialize decoder append source: ",
                &error.to_string(),
            ])
        })?;
        let attention_source = serde_json::to_string(&sources.attention).map_err(|error| {
            js_decoder_error(&[
                "cannot serialize decoder attention source: ",
                &error.to_string(),
            ])
        })?;
        let mut json =
            String::from("{\"schema_version\":1,\"sources\":{\"decoder_kv_append_f32\":");
        json.push_str(&append_source);
        json.push_str(",\"decoder_gqa_f32\":");
        json.push_str(&attention_source);
        json.push_str("},\"shader_blake3\":{\"decoder_kv_append_f32\":\"");
        json.push_str(&blake3_hex(&source_blake3(&sources.append)));
        json.push_str("\",\"decoder_gqa_f32\":\"");
        json.push_str(&blake3_hex(&source_blake3(&sources.attention)));
        json.push_str("\"}}");
        Ok(json)
    }

    pub(super) fn begin(
        &self,
        descriptor_json: &str,
        key_cache: &js_sys::Uint8Array,
        value_cache: &js_sys::Uint8Array,
    ) -> js_sys::Promise {
        let prepared = (|| {
            check_decoder_admission(&self.owner)?;
            let parsed = parse_decoder_descriptor_json(descriptor_json)?;
            let key_cache_bytes = decoder_uint8_to_bytes(key_cache)?;
            let value_cache_bytes = decoder_uint8_to_bytes(value_cache)?;
            let key_f32 = decoder_bytes_to_f32(&key_cache_bytes, "initial K cache")?;
            let value_f32 = decoder_bytes_to_f32(&value_cache_bytes, "initial V cache")?;
            let descriptor = DecoderKvSessionDescriptor {
                query_heads: parsed.query_heads,
                key_value_heads: parsed.key_value_heads,
                head_dim: parsed.head_dim,
                prefix_tokens: parsed.prefix_tokens,
                cache_capacity: parsed.cache_capacity,
                key_cache: key_f32.as_slice(),
                value_cache: value_f32.as_slice(),
            };
            let plan = descriptor.plan().map_err(|error| {
                js_decoder_error(&[
                    "invalid decoder KV session descriptor geometry or initial cache operands: ",
                    &error.to_string(),
                ])
            })?;
            let sources = canonical_decoder_sources()?;
            Ok((plan, key_cache_bytes, value_cache_bytes, sources))
        })();
        let (plan, key_cache_bytes, value_cache_bytes, sources) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return js_sys::Promise::reject(&error),
        };
        let owner = self.owner.clone();
        let device = self.device.clone();
        let queue = self.queue.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            run_begin(
                &owner,
                &device,
                &queue,
                plan,
                &key_cache_bytes,
                &value_cache_bytes,
                sources,
            )
            .await
            .map(|diagnostics| JsValue::from_str(&diagnostics))
        })
    }

    pub(super) fn begin_with_shader_override(
        &self,
        descriptor_json: &str,
        key_cache: &js_sys::Uint8Array,
        value_cache: &js_sys::Uint8Array,
        kernel: &str,
        source: &str,
    ) -> js_sys::Promise {
        if kernel != APPEND_KERNEL_NAME && kernel != ATTENTION_KERNEL_NAME {
            return js_sys::Promise::reject(&js_decoder_error(&[
                "unknown decoder shader override kernel: ",
                kernel,
            ]));
        }
        if source.is_empty() {
            return js_sys::Promise::reject(&js_decoder_error(&[
                "decoder shader override source must not be empty",
            ]));
        }
        let prepared = (|| {
            check_decoder_admission(&self.owner)?;
            let parsed = parse_decoder_descriptor_json(descriptor_json)?;
            let key_cache_bytes = decoder_uint8_to_bytes(key_cache)?;
            let value_cache_bytes = decoder_uint8_to_bytes(value_cache)?;
            let key_f32 = decoder_bytes_to_f32(&key_cache_bytes, "initial K cache")?;
            let value_f32 = decoder_bytes_to_f32(&value_cache_bytes, "initial V cache")?;
            let descriptor = DecoderKvSessionDescriptor {
                query_heads: parsed.query_heads,
                key_value_heads: parsed.key_value_heads,
                head_dim: parsed.head_dim,
                prefix_tokens: parsed.prefix_tokens,
                cache_capacity: parsed.cache_capacity,
                key_cache: key_f32.as_slice(),
                value_cache: value_f32.as_slice(),
            };
            let plan = descriptor.plan().map_err(|error| {
                js_decoder_error(&[
                    "invalid decoder KV session descriptor geometry or initial cache operands: ",
                    &error.to_string(),
                ])
            })?;
            let mut sources = canonical_decoder_sources()?;
            if kernel == APPEND_KERNEL_NAME {
                sources.append = source.to_owned();
            } else {
                sources.attention = source.to_owned();
            }
            Ok((plan, key_cache_bytes, value_cache_bytes, sources))
        })();
        let (plan, key_cache_bytes, value_cache_bytes, sources) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return js_sys::Promise::reject(&error),
        };
        let owner = self.owner.clone();
        let device = self.device.clone();
        let queue = self.queue.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            run_begin(
                &owner,
                &device,
                &queue,
                plan,
                &key_cache_bytes,
                &value_cache_bytes,
                sources,
            )
            .await
            .map(|diagnostics| JsValue::from_str(&diagnostics))
        })
    }

    pub(super) fn step(
        &self,
        query: &js_sys::Uint8Array,
        appended_key: &js_sys::Uint8Array,
        appended_value: &js_sys::Uint8Array,
    ) -> js_sys::Promise {
        let prepared = (|| {
            let (lease, session) = acquire_decoder_session(&self.owner)?;
            let query_bytes = match decoder_uint8_to_bytes(query) {
                Ok(bytes) => bytes,
                Err(error) => {
                    restore_decoder_session(&self.owner, lease, session);
                    return Err(error);
                }
            };
            let appended_key_bytes = match decoder_uint8_to_bytes(appended_key) {
                Ok(bytes) => bytes,
                Err(error) => {
                    restore_decoder_session(&self.owner, lease, session);
                    return Err(error);
                }
            };
            let appended_value_bytes = match decoder_uint8_to_bytes(appended_value) {
                Ok(bytes) => bytes,
                Err(error) => {
                    restore_decoder_session(&self.owner, lease, session);
                    return Err(error);
                }
            };
            let query_f32 = match decoder_bytes_to_f32(&query_bytes, "step query") {
                Ok(value) => value,
                Err(error) => {
                    restore_decoder_session(&self.owner, lease, session);
                    return Err(error);
                }
            };
            let appended_key_f32 =
                match decoder_bytes_to_f32(&appended_key_bytes, "step appended key") {
                    Ok(value) => value,
                    Err(error) => {
                        restore_decoder_session(&self.owner, lease, session);
                        return Err(error);
                    }
                };
            let appended_value_f32 =
                match decoder_bytes_to_f32(&appended_value_bytes, "step appended value") {
                    Ok(value) => value,
                    Err(error) => {
                        restore_decoder_session(&self.owner, lease, session);
                        return Err(error);
                    }
                };
            let step_inputs = DecoderKvSessionStep {
                query: query_f32.as_slice(),
                appended_key: appended_key_f32.as_slice(),
                appended_value: appended_value_f32.as_slice(),
            };
            let step_plan = match session.plan.plan_step(session.cache_tokens, &step_inputs) {
                Ok(step_plan) => step_plan,
                Err(error) => {
                    restore_decoder_session(&self.owner, lease, session);
                    return Err(js_decoder_error(&[
                        "invalid decoder KV step operands or capacity: ",
                        &error.to_string(),
                    ]));
                }
            };
            Ok((
                lease,
                session,
                step_plan,
                StepOperands {
                    query_bytes,
                    appended_key_bytes,
                    appended_value_bytes,
                },
            ))
        })();
        let (lease, session, step_plan, operands) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return js_sys::Promise::reject(&error),
        };
        let owner = self.owner.clone();
        let device = self.device.clone();
        let queue = self.queue.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            run_step(&owner, &device, &queue, lease, session, step_plan, operands).await
        })
    }

    pub(super) fn finish(&self) -> js_sys::Promise {
        let prepared = acquire_decoder_session(&self.owner);
        let (lease, session) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return js_sys::Promise::reject(&error),
        };
        let owner = self.owner.clone();
        let device = self.device.clone();
        let queue = self.queue.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            run_finish(&owner, &device, &queue, lease, session).await
        })
    }
}

async fn run_step(
    owner: &AsyncSessionOwner<BrowserDecoderKvSession>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lease: crate::AsyncSessionLease,
    mut session: BrowserDecoderKvSession,
    step_plan: pvlc_runtime_core::DecoderKvSessionStepPlan,
    operands: StepOperands,
) -> Result<JsValue, JsValue> {
    let StepOperands {
        query_bytes,
        appended_key_bytes,
        appended_value_bytes,
    } = operands;
    session.poisoned = true;
    let mut ledger: Vec<ScopeKind> = Vec::new();
    for scope in CHECKED_SCOPES {
        if let Err(error) = push_decoder_error_scope(device, scope).await {
            ledger.push(scope);
            let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
            restore_decoder_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
        ledger.push(scope);
    }
    let raw_device = match raw_decoder_device(device) {
        Ok(raw) => raw,
        Err(error) => {
            restore_decoder_session(owner, lease, session);
            return Err(error);
        }
    };
    let raw_queue = match raw_decoder_queue(device, queue) {
        Ok(raw) => raw,
        Err(error) => {
            restore_decoder_session(owner, lease, session);
            return Err(error);
        }
    };
    if let Err(error) = session.encode_step(
        raw_device,
        &raw_queue,
        &step_plan,
        &query_bytes,
        &appended_key_bytes,
        &appended_value_bytes,
    ) {
        let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
        restore_decoder_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if let Err(error) = await_decoder_queue_completion(&raw_queue).await {
        let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
        restore_decoder_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if let Err(error) = map_decoder_buffer(
        session.attention_readback_buffer.as_ref(),
        session.plan.attention_bytes,
    )
    .await
    {
        let _ = unmap_decoder_buffer(session.attention_readback_buffer.as_ref());
        let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
        restore_decoder_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if owner.generation() != Some(lease.generation()) {
        let _ = unmap_decoder_buffer(session.attention_readback_buffer.as_ref());
        wait_decoder_owner_idle(owner).await;
        let _ = drain_decoder_error_scopes(device, &mut ledger).await;
        let _ = owner.complete(lease, session, CompletionAction::Finish);
        return Err(js_decoder_error(&[
            "decoder KV session step is stale: its generation was cancelled",
        ]));
    }
    let attention = match read_decoder_mapped(
        session.attention_readback_buffer.as_ref(),
        session.plan.attention_bytes,
    ) {
        Ok(attention) => attention,
        Err(error) => {
            let _ = unmap_decoder_buffer(session.attention_readback_buffer.as_ref());
            let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
            restore_decoder_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    if let Err(error) = unmap_decoder_buffer(session.attention_readback_buffer.as_ref()) {
        let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
        restore_decoder_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    session.cache_tokens = step_plan.cache_tokens_after;
    session.poisoned = false;
    let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
    if !failures.is_empty() || !captures.is_empty() {
        session.poisoned = true;
        restore_decoder_session(owner, lease, session);
        return Err(captured_failure_message(captures, failures));
    }
    let diagnostics = match step_diagnostics_json(&session, &step_plan) {
        Ok(diagnostics) => diagnostics,
        Err(error) => {
            restore_decoder_session(owner, lease, session);
            return Err(error);
        }
    };
    let _ = owner.complete(lease, session, CompletionAction::Restore);
    decoder_step_result(attention, diagnostics)
}

async fn run_finish(
    owner: &AsyncSessionOwner<BrowserDecoderKvSession>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lease: crate::AsyncSessionLease,
    mut session: BrowserDecoderKvSession,
) -> Result<JsValue, JsValue> {
    session.poisoned = true;
    let mut ledger: Vec<ScopeKind> = Vec::new();
    for scope in CHECKED_SCOPES {
        if let Err(error) = push_decoder_error_scope(device, scope).await {
            ledger.push(scope);
            let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
            restore_decoder_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
        ledger.push(scope);
    }
    let raw_device = match raw_decoder_device(device) {
        Ok(raw) => raw,
        Err(error) => {
            restore_decoder_session(owner, lease, session);
            return Err(error);
        }
    };
    let raw_queue = match raw_decoder_queue(device, queue) {
        Ok(raw) => raw,
        Err(error) => {
            restore_decoder_session(owner, lease, session);
            return Err(error);
        }
    };
    let readback = match session.encode_finish(raw_device, &raw_queue) {
        Ok(readback) => readback,
        Err(error) => {
            let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
            restore_decoder_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    let readback_bytes = match session
        .plan
        .cache_bytes
        .checked_mul(2)
        .ok_or_else(|| js_decoder_error(&["decoder KV finish readback overflowed"]))
    {
        Ok(bytes) => bytes,
        Err(error) => {
            restore_decoder_session(owner, lease, session);
            return Err(error);
        }
    };
    if let Err(error) = await_decoder_queue_completion(&raw_queue).await {
        let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
        restore_decoder_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if let Err(error) = map_decoder_buffer(readback.as_ref(), readback_bytes).await {
        let _ = unmap_decoder_buffer(readback.as_ref());
        let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
        restore_decoder_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if owner.generation() != Some(lease.generation()) {
        let _ = unmap_decoder_buffer(readback.as_ref());
        wait_decoder_owner_idle(owner).await;
        let _ = drain_decoder_error_scopes(device, &mut ledger).await;
        let _ = owner.complete(lease, session, CompletionAction::Finish);
        return Err(js_decoder_error(&[
            "decoder KV session finish is stale: its generation was cancelled",
        ]));
    }
    let mapped = match read_decoder_mapped(readback.as_ref(), readback_bytes) {
        Ok(mapped) => mapped,
        Err(error) => {
            let _ = unmap_decoder_buffer(readback.as_ref());
            let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
            restore_decoder_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    if let Err(error) = unmap_decoder_buffer(readback.as_ref()) {
        let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
        restore_decoder_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    let cache_elements = match checked_usize(session.plan.cache_bytes, "decoder cache") {
        Ok(bytes) => bytes,
        Err(error) => {
            restore_decoder_session(owner, lease, session);
            return Err(error);
        }
    };
    let key_cache = mapped[..cache_elements].to_vec();
    let value_cache = mapped[cache_elements..].to_vec();
    let (captures, failures) = drain_decoder_error_scopes(device, &mut ledger).await;
    if !failures.is_empty() || !captures.is_empty() {
        session.poisoned = true;
        restore_decoder_session(owner, lease, session);
        return Err(captured_failure_message(captures, failures));
    }
    let diagnostics = match finish_diagnostics_json(&session) {
        Ok(diagnostics) => diagnostics,
        Err(error) => {
            restore_decoder_session(owner, lease, session);
            return Err(error);
        }
    };
    let _ = owner.complete(lease, session, CompletionAction::Finish);
    decoder_finish_result(key_cache, value_cache, diagnostics)
}
