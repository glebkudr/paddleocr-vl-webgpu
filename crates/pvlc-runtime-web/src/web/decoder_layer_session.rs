//! Sealed persistent browser decoder attention-block session authority.
//!
//! The authority privately owns the exact `wgpu::Device`, the exact `wgpu::Queue`,
//! and one `crate::AsyncSessionOwner<BrowserDecoderLayerSession>`. Every operation
//! validates its inputs, the PVLCPK01 weight pack, and the exact core plans before
//! the first GPU effect, pushes the three checked error scopes, executes the exact
//! phase topology with raw WebGPU calls that surface every thrown error as a
//! `Result`, and drains the scopes LIFO. Any post-effect failure poisons the stored
//! session terminally; a cancelled generation drains its scopes only after the
//! newer in-flight lease clears. No `unsafe`, no macros, no host-side compute
//! shadow.

use pvlc_runtime_core::{
    DecoderAttentionBlockDescriptor, DecoderAttentionBlockStep, DecoderKvSessionDescriptor,
};
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
const RMS_NORM_KERNEL_NAME: &str = "rms_norm_f32";
const GEMV_KERNEL_NAME: &str = "gemv_f32";
const MROPE_KERNEL_NAME: &str = "decoder_mrope_f32";
const APPEND_KERNEL_NAME: &str = "decoder_kv_append_f32";
const ATTENTION_KERNEL_NAME: &str = "decoder_gqa_f32";
const RESIDUAL_KERNEL_NAME: &str = "add_f32";
const KERNEL_NAMES: [&str; 6] = [
    RMS_NORM_KERNEL_NAME,
    GEMV_KERNEL_NAME,
    MROPE_KERNEL_NAME,
    APPEND_KERNEL_NAME,
    ATTENTION_KERNEL_NAME,
    RESIDUAL_KERNEL_NAME,
];
const ENTRY_POINT: &str = "main";
const BUFFER_HIDDEN_TOKEN: &str = "decoder-layer-session-hidden-token";
const BUFFER_NORM1_WEIGHT: &str = "decoder-layer-session-norm1-weight";
const BUFFER_Q_WEIGHT: &str = "decoder-layer-session-q-weight";
const BUFFER_K_WEIGHT: &str = "decoder-layer-session-k-weight";
const BUFFER_V_WEIGHT: &str = "decoder-layer-session-v-weight";
const BUFFER_O_WEIGHT: &str = "decoder-layer-session-o-weight";
const BUFFER_ROPE_COS: &str = "decoder-layer-session-rope-cos";
const BUFFER_ROPE_SIN: &str = "decoder-layer-session-rope-sin";
const BUFFER_NORM1: &str = "decoder-layer-session-norm1";
const BUFFER_Q_PROJECTION: &str = "decoder-layer-session-q-projection";
const BUFFER_K_PROJECTION: &str = "decoder-layer-session-k-projection";
const BUFFER_V_PROJECTION: &str = "decoder-layer-session-v-projection";
const BUFFER_MROPE_QUERY: &str = "decoder-layer-session-mrope-query";
const BUFFER_MROPE_KEY: &str = "decoder-layer-session-mrope-key";
const BUFFER_KEY_CACHE: &str = "decoder-layer-session-key-cache";
const BUFFER_VALUE_CACHE: &str = "decoder-layer-session-value-cache";
const BUFFER_ATTENTION_OUTPUT: &str = "decoder-layer-session-attention-output";
const BUFFER_O_PROJECTION: &str = "decoder-layer-session-o-projection";
const BUFFER_ATTENTION_RESIDUAL: &str = "decoder-layer-session-attention-residual";
const BUFFER_LAYER_READBACK: &str = "decoder-layer-session-layer-readback";
const BUFFER_RMS_UNIFORM: &str = "decoder-layer-session-rms-uniform";
const BUFFER_GEMV_Q_UNIFORM: &str = "decoder-layer-session-gemv-q-uniform";
const BUFFER_GEMV_K_UNIFORM: &str = "decoder-layer-session-gemv-k-uniform";
const BUFFER_GEMV_V_UNIFORM: &str = "decoder-layer-session-gemv-v-uniform";
const BUFFER_MROPE_UNIFORM: &str = "decoder-layer-session-mrope-uniform";
const BUFFER_APPEND_UNIFORM: &str = "decoder-layer-session-append-uniform";
const BUFFER_ATTENTION_UNIFORM: &str = "decoder-layer-session-attention-uniform";
const BUFFER_GEMV_O_UNIFORM: &str = "decoder-layer-session-gemv-o-uniform";
const BUFFER_RESIDUAL_UNIFORM: &str = "decoder-layer-session-residual-uniform";
const BUFFER_FINISH_KEY_READBACK: &str = "decoder-layer-session-finish-key-readback";
const BUFFER_FINISH_VALUE_READBACK: &str = "decoder-layer-session-finish-value-readback";
const DESCRIPTOR_FIELDS: [&str; 11] = [
    "schema_version",
    "hidden_size",
    "query_heads",
    "key_value_heads",
    "head_dim",
    "query_width",
    "key_value_width",
    "prefix_tokens",
    "cache_capacity",
    "mrope_sections",
    "rms_norm_epsilon",
];
const PINNED_LAYER_QUERY_WIDTH: u32 = 2048;
const PINNED_LAYER_KEY_VALUE_WIDTH: u32 = 256;
const PACK_MAGIC: [u8; 8] = *b"PVLCPK01";
const PACK_HEADER_BYTES: usize = 32;
const PACK_DIRECTORY_FIXED_BYTES: usize = 56;
const PACK_DIRECTORY_ENTRY_ALIGNMENT: usize = 8;
const PACK_MAX_ALIGNMENT: u64 = 4096;
const PACK_VERSION: u32 = 1;
const PACK_SECTION_COUNT: u32 = 8;
const PACK_DESCRIPTOR_SECTION_ID: &str = "ir.decoder_layer_00_attention_block";
const PACK_SHARD_IDS: [&str; 7] = [
    "weights.input_layernorm",
    "weights.q_proj",
    "weights.k_proj",
    "weights.v_proj",
    "weights.o_proj",
    "weights.mrope_cos",
    "weights.mrope_sin",
];
const PACK_MODEL_ID: &str = "PaddlePaddle/PaddleOCR-VL-1.6";
const PACK_MODEL_REVISION: &str = "66317acc4c9fc17bd154591ce650735cd2855f3e";
const PACK_MANIFEST_FIELDS: [&str; 7] = [
    "compiler_build",
    "compiler_model_abi",
    "context_limit",
    "model_id",
    "model_revision",
    "precision_profile",
    "resolution_buckets",
];
const PACK_DESCRIPTOR_FIELDS: [&str; 15] = [
    "schema_version",
    "oracle",
    "case_id",
    "model_revision",
    "hidden_size",
    "query_heads",
    "key_value_heads",
    "head_dim",
    "query_width",
    "key_value_width",
    "prefix_tokens",
    "cache_capacity",
    "rms_norm_epsilon",
    "mrope_sections",
    "shards",
];
const PACK_DESCRIPTOR_ORACLES: [&str; 2] = ["synthetic", "official_l3"];
const UNIFORM_BUFFER_BYTES: u64 = 16;

/// Sealed owner of the persistent decoder attention-block session lifecycle.
pub(super) struct DecoderLayerSessionAuthority {
    device: wgpu::Device,
    queue: wgpu::Queue,
    owner: crate::AsyncSessionOwner<BrowserDecoderLayerSession>,
}

struct BrowserDecoderLayerSession {
    kv_plan: pvlc_runtime_core::DecoderKvSessionPlan,
    attention_plan: pvlc_runtime_core::DecoderAttentionBlockPlan,
    cache_tokens: u32,
    poisoned: bool,
    ready: bool,
    rms_norm_shader_blake3: [u8; 32],
    gemv_shader_blake3: [u8; 32],
    mrope_shader_blake3: [u8; 32],
    append_shader_blake3: [u8; 32],
    attention_shader_blake3: [u8; 32],
    residual_shader_blake3: [u8; 32],
    hidden_token_buffer: wgpu::webgpu::GpuBuffer,
    norm1_weight_buffer: wgpu::webgpu::GpuBuffer,
    q_weight_buffer: wgpu::webgpu::GpuBuffer,
    k_weight_buffer: wgpu::webgpu::GpuBuffer,
    v_weight_buffer: wgpu::webgpu::GpuBuffer,
    o_weight_buffer: wgpu::webgpu::GpuBuffer,
    rope_cos_buffer: wgpu::webgpu::GpuBuffer,
    rope_sin_buffer: wgpu::webgpu::GpuBuffer,
    norm1_buffer: wgpu::webgpu::GpuBuffer,
    q_projection_buffer: wgpu::webgpu::GpuBuffer,
    k_projection_buffer: wgpu::webgpu::GpuBuffer,
    v_projection_buffer: wgpu::webgpu::GpuBuffer,
    mrope_query_buffer: wgpu::webgpu::GpuBuffer,
    mrope_key_buffer: wgpu::webgpu::GpuBuffer,
    key_cache_buffer: wgpu::webgpu::GpuBuffer,
    value_cache_buffer: wgpu::webgpu::GpuBuffer,
    attention_output_buffer: wgpu::webgpu::GpuBuffer,
    o_projection_buffer: wgpu::webgpu::GpuBuffer,
    attention_residual_buffer: wgpu::webgpu::GpuBuffer,
    layer_readback_buffer: wgpu::webgpu::GpuBuffer,
    rms_uniform_buffer: wgpu::webgpu::GpuBuffer,
    gemv_q_uniform_buffer: wgpu::webgpu::GpuBuffer,
    gemv_k_uniform_buffer: wgpu::webgpu::GpuBuffer,
    gemv_v_uniform_buffer: wgpu::webgpu::GpuBuffer,
    mrope_uniform_buffer: wgpu::webgpu::GpuBuffer,
    append_uniform_buffer: wgpu::webgpu::GpuBuffer,
    attention_uniform_buffer: wgpu::webgpu::GpuBuffer,
    gemv_o_uniform_buffer: wgpu::webgpu::GpuBuffer,
    residual_uniform_buffer: wgpu::webgpu::GpuBuffer,
    rms_norm_pipeline: js_sys::Object,
    gemv_pipeline: js_sys::Object,
    mrope_pipeline: js_sys::Object,
    append_pipeline: js_sys::Object,
    attention_pipeline: js_sys::Object,
    residual_pipeline: js_sys::Object,
    rms_bind_group: js_sys::Object,
    gemv_q_bind_group: js_sys::Object,
    gemv_k_bind_group: js_sys::Object,
    gemv_v_bind_group: js_sys::Object,
    mrope_bind_group: js_sys::Object,
    append_bind_group: js_sys::Object,
    attention_bind_group: js_sys::Object,
    gemv_o_bind_group: js_sys::Object,
    residual_bind_group: js_sys::Object,
}

struct LayerShaderSources {
    rms_norm: String,
    gemv: String,
    mrope: String,
    append: String,
    attention: String,
    residual: String,
}

struct LayerShaderDigests {
    rms_norm: [u8; 32],
    gemv: [u8; 32],
    mrope: [u8; 32],
    append: [u8; 32],
    attention: [u8; 32],
    residual: [u8; 32],
}

struct LayerBeginOperands {
    key_cache_bytes: Vec<u8>,
    value_cache_bytes: Vec<u8>,
    norm1_weight_bytes: Vec<u8>,
    q_weight_bytes: Vec<u8>,
    k_weight_bytes: Vec<u8>,
    v_weight_bytes: Vec<u8>,
    o_weight_bytes: Vec<u8>,
    rope_cos_bytes: Vec<u8>,
    rope_sin_bytes: Vec<u8>,
}

struct LayerStepOperands {
    transition: pvlc_runtime_core::DecoderKvSessionStepPlan,
    step_plan: pvlc_runtime_core::DecoderAttentionBlockStepPlan,
    hidden_bytes: Vec<u8>,
}

struct ParsedDescriptor {
    hidden_size: u32,
    query_heads: u32,
    key_value_heads: u32,
    head_dim: u32,
    prefix_tokens: u32,
    cache_capacity: u32,
    rms_norm_epsilon: f32,
}

struct ParsedWeightPack {
    norm1_weight: Vec<u8>,
    q_weight: Vec<u8>,
    k_weight: Vec<u8>,
    v_weight: Vec<u8>,
    o_weight: Vec<u8>,
    rope_cos: Vec<u8>,
    rope_sin: Vec<u8>,
}

struct PackDirectoryEntry {
    offset: u64,
    byte_length: u64,
    alignment: u64,
    digest: [u8; 32],
}

fn js_layer_error(parts: &[&str]) -> JsValue {
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

fn raw_layer_device(device: &wgpu::Device) -> Result<&JsValue, JsValue> {
    device.as_webgpu().map(AsRef::as_ref).ok_or_else(|| {
        js_layer_error(&["decoder layer session device has no browser WebGPU handle"])
    })
}

fn raw_layer_queue(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<JsValue, JsValue> {
    let raw_device = raw_layer_device(device)?;
    let registered =
        js_sys::Reflect::get(raw_device, &JsValue::from_str("queue")).map_err(|error| {
            js_layer_error(&["cannot access GPUDevice.queue: ", &js_error_text(&error)])
        })?;
    let raw_queue: &JsValue = queue.as_webgpu().map(AsRef::as_ref).ok_or_else(|| {
        js_layer_error(&["decoder layer session queue has no browser WebGPU handle"])
    })?;
    if !js_sys::Object::is(&registered, raw_queue) {
        return Err(js_layer_error(&[
            "decoder layer session queue handle is not the exact device queue",
        ]));
    }
    Ok(registered)
}

fn raw_layer_method(handle: &JsValue, name: &str) -> Result<js_sys::Function, JsValue> {
    js_sys::Reflect::get(handle, &JsValue::from_str(name))
        .map_err(|error| {
            js_layer_error(&[
                "cannot access WebGPU method ",
                name,
                ": ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<js_sys::Function>()
        .map_err(|_| js_layer_error(&["WebGPU member ", name, " is not callable"]))
}

async fn push_layer_error_scope(device: &wgpu::Device, scope: ScopeKind) -> Result<(), JsValue> {
    let raw = raw_layer_device(device)?;
    let push = raw_layer_method(raw, "pushErrorScope")?;
    push.call1(raw, &JsValue::from_str(scope.filter_str()))
        .map(|_| ())
        .map_err(|error| {
            js_layer_error(&[
                "cannot push ",
                scope.as_str(),
                " WebGPU error scope: ",
                &js_error_text(&error),
            ])
        })
}

async fn pop_layer_error_scope(
    device: &wgpu::Device,
    scope: ScopeKind,
) -> Result<Option<String>, JsValue> {
    let raw = raw_layer_device(device)?;
    let pop = raw_layer_method(raw, "popErrorScope")?;
    let invocation = pop
        .call0(raw)
        .map_err(|error| {
            js_layer_error(&[
                "cannot invoke popErrorScope for ",
                scope.as_str(),
                " scope: ",
                &js_error_text(&error),
            ])
        })
        .and_then(|pending| {
            pending.dyn_into::<js_sys::Promise>().map_err(|_| {
                js_layer_error(&[
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
                js_layer_error(&[
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

async fn drain_layer_error_scopes(
    device: &wgpu::Device,
    ledger: &mut Vec<ScopeKind>,
) -> (Vec<String>, Vec<JsValue>) {
    let mut captures = Vec::new();
    let mut failures = Vec::new();
    while let Some(scope) = ledger.pop() {
        match pop_layer_error_scope(device, scope).await {
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
    let mut message = String::from("decoder layer session captured WebGPU errors:");
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

async fn yield_layer_event_loop() {
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

async fn wait_layer_owner_idle(owner: &AsyncSessionOwner<BrowserDecoderLayerSession>) {
    while owner.is_in_flight() {
        yield_layer_event_loop().await;
    }
}

fn poison_stored_session(owner: &AsyncSessionOwner<BrowserDecoderLayerSession>) {
    if let Some(mut session) = owner.stored_mut() {
        session.poisoned = true;
    }
}

fn mark_session_ready(owner: &AsyncSessionOwner<BrowserDecoderLayerSession>) {
    if let Some(mut session) = owner.stored_mut() {
        session.ready = true;
    }
}

fn check_layer_admission(
    owner: &AsyncSessionOwner<BrowserDecoderLayerSession>,
) -> Result<(), JsValue> {
    if owner.stored().is_some_and(|session| session.poisoned) {
        return Err(js_layer_error(&[
            "decoder layer session is terminally poisoned",
        ]));
    }
    if owner.is_busy() {
        return Err(js_layer_error(&[
            "decoder layer session is already active or busy with another operation",
        ]));
    }
    Ok(())
}

fn acquire_layer_session(
    owner: &AsyncSessionOwner<BrowserDecoderLayerSession>,
) -> Result<(crate::AsyncSessionLease, BrowserDecoderLayerSession), JsValue> {
    {
        let Some(session) = owner.stored() else {
            return Err(js_layer_error(&["no ready decoder layer session"]));
        };
        if session.poisoned {
            return Err(js_layer_error(&[
                "decoder layer session is terminally poisoned",
            ]));
        }
        if !session.ready {
            return Err(js_layer_error(&["no ready decoder layer session"]));
        }
    }
    owner
        .acquire()
        .map_err(|_| js_layer_error(&["no stored decoder layer session"]))
}

fn restore_layer_session(
    owner: &AsyncSessionOwner<BrowserDecoderLayerSession>,
    lease: crate::AsyncSessionLease,
    session: BrowserDecoderLayerSession,
) {
    let _ = owner.complete(lease, session, CompletionAction::Restore);
}

fn parse_layer_descriptor_json(json: &str) -> Result<ParsedDescriptor, JsValue> {
    let value: Value = serde_json::from_str(json).map_err(|error| {
        js_layer_error(&[
            "invalid decoder layer session descriptor json: ",
            &error.to_string(),
        ])
    })?;
    let object = value.as_object().ok_or_else(|| {
        js_layer_error(&["invalid decoder layer session descriptor: expected an object"])
    })?;
    for key in object.keys() {
        if !DESCRIPTOR_FIELDS.contains(&key.as_str()) {
            return Err(js_layer_error(&[
                "invalid decoder layer session descriptor: unknown field ",
                key,
            ]));
        }
    }
    let schema_version = required_descriptor_u32(object, "schema_version")?;
    if schema_version != 1 {
        return Err(js_layer_error(&[
            "invalid decoder layer session descriptor schema version",
        ]));
    }
    let hidden_size = required_descriptor_u32(object, "hidden_size")?;
    let query_heads = required_descriptor_u32(object, "query_heads")?;
    let key_value_heads = required_descriptor_u32(object, "key_value_heads")?;
    let head_dim = required_descriptor_u32(object, "head_dim")?;
    let query_width = required_descriptor_u32(object, "query_width")?;
    let key_value_width = required_descriptor_u32(object, "key_value_width")?;
    for (field, actual, pinned) in [
        (
            "hidden_size",
            hidden_size,
            pvlc_runtime_core::PINNED_DECODER_HIDDEN_SIZE,
        ),
        (
            "query_heads",
            query_heads,
            pvlc_runtime_core::PINNED_DECODER_QUERY_HEADS,
        ),
        (
            "key_value_heads",
            key_value_heads,
            pvlc_runtime_core::PINNED_DECODER_KEY_VALUE_HEADS,
        ),
        (
            "head_dim",
            head_dim,
            pvlc_runtime_core::MAX_DECODER_HEAD_DIM,
        ),
        ("query_width", query_width, PINNED_LAYER_QUERY_WIDTH),
        (
            "key_value_width",
            key_value_width,
            PINNED_LAYER_KEY_VALUE_WIDTH,
        ),
    ] {
        if actual != pinned {
            return Err(js_layer_error(&[
                "invalid decoder layer session descriptor geometry: field ",
                field,
                " drifted from the pinned decoder value",
            ]));
        }
    }
    require_descriptor_mrope_sections(object)?;
    let rms_norm_epsilon = required_descriptor_epsilon(object)?;
    Ok(ParsedDescriptor {
        hidden_size,
        query_heads,
        key_value_heads,
        head_dim,
        prefix_tokens: required_descriptor_u32(object, "prefix_tokens")?,
        cache_capacity: required_descriptor_u32(object, "cache_capacity")?,
        rms_norm_epsilon,
    })
}

fn required_descriptor_u32(object: &Map<String, Value>, key: &str) -> Result<u32, JsValue> {
    let value = object.get(key).ok_or_else(|| {
        js_layer_error(&[
            "invalid decoder layer session descriptor: missing field ",
            key,
        ])
    })?;
    let integer = value.as_u64().ok_or_else(|| {
        js_layer_error(&[
            "invalid decoder layer session descriptor: field ",
            key,
            " must be an unsigned integer",
        ])
    })?;
    u32::try_from(integer).map_err(|_| {
        js_layer_error(&[
            "invalid decoder layer session descriptor: field ",
            key,
            " is out of range",
        ])
    })
}

fn require_descriptor_mrope_sections(object: &Map<String, Value>) -> Result<(), JsValue> {
    let value = object.get("mrope_sections").ok_or_else(|| {
        js_layer_error(&["invalid decoder layer session descriptor: missing field mrope_sections"])
    })?;
    let array = value.as_array().ok_or_else(|| {
        js_layer_error(&[
            "invalid decoder layer session descriptor: field mrope_sections must be an array",
        ])
    })?;
    let pinned = pvlc_runtime_core::PINNED_DECODER_MROPE_SECTIONS;
    if array.len() != pinned.len() {
        return Err(js_layer_error(&[
            "invalid decoder layer session descriptor: mrope_sections drifted from the pinned [16, 24, 24]",
        ]));
    }
    for (index, section) in array.iter().enumerate() {
        let actual = section.as_u64().ok_or_else(|| {
            js_layer_error(&[
                "invalid decoder layer session descriptor: mrope_sections entries must be unsigned integers",
            ])
        })?;
        if actual != pinned[index] as u64 {
            return Err(js_layer_error(&[
                "invalid decoder layer session descriptor: mrope_sections drifted from the pinned [16, 24, 24]",
            ]));
        }
    }
    Ok(())
}

fn required_descriptor_epsilon(object: &Map<String, Value>) -> Result<f32, JsValue> {
    let value = object.get("rms_norm_epsilon").ok_or_else(|| {
        js_layer_error(&[
            "invalid decoder layer session descriptor: missing field rms_norm_epsilon",
        ])
    })?;
    let epsilon = value.as_f64().ok_or_else(|| {
        js_layer_error(&[
            "invalid decoder layer session descriptor: field rms_norm_epsilon must be a number",
        ])
    })? as f32;
    if epsilon != pvlc_runtime_core::PINNED_DECODER_RMS_NORM_EPSILON {
        return Err(js_layer_error(&[
            "invalid decoder layer session descriptor: rms_norm_epsilon drifted from the pinned decoder value",
        ]));
    }
    Ok(epsilon)
}

fn layer_uint8_to_bytes(value: &js_sys::Uint8Array) -> Result<Vec<u8>, JsValue> {
    if !value.is_instance_of::<js_sys::Uint8Array>() {
        return Err(js_layer_error(&[
            "decoder layer session operand must be a Uint8Array view",
        ]));
    }
    if value.byte_length() == 0 {
        return Ok(Vec::new());
    }
    Ok(value.to_vec())
}

fn layer_pack_to_bytes(value: &js_sys::Uint8Array) -> Result<Vec<u8>, JsValue> {
    if !value.is_instance_of::<js_sys::Uint8Array>() {
        return Err(js_layer_error(&[
            "decoder layer weight pack must be a Uint8Array view",
        ]));
    }
    if value.byte_length() == 0 {
        return Ok(Vec::new());
    }
    Ok(value.to_vec())
}

fn layer_bytes_to_f32(bytes: &[u8], label: &str) -> Result<Vec<f32>, JsValue> {
    if !bytes.len().is_multiple_of(4) {
        return Err(js_layer_error(&[
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

fn canonical_layer_sources() -> Result<LayerShaderSources, JsValue> {
    let rms_norm = pvlc_wgsl::module(pvlc_runtime_core::KernelId::RmsNormF32)
        .ok_or_else(|| js_layer_error(&["canonical decoder layer RMS-norm kernel is missing"]))?;
    let gemv = pvlc_wgsl::module(pvlc_runtime_core::KernelId::GemvF32)
        .ok_or_else(|| js_layer_error(&["canonical decoder layer GEMV kernel is missing"]))?;
    let mrope = pvlc_wgsl::module(pvlc_runtime_core::KernelId::DecoderMropeF32)
        .ok_or_else(|| js_layer_error(&["canonical decoder layer M-RoPE kernel is missing"]))?;
    let append = pvlc_wgsl::module(pvlc_runtime_core::KernelId::DecoderKvAppendF32)
        .ok_or_else(|| js_layer_error(&["canonical decoder layer append kernel is missing"]))?;
    let attention = pvlc_wgsl::module(pvlc_runtime_core::KernelId::DecoderGqaF32)
        .ok_or_else(|| js_layer_error(&["canonical decoder layer GQA kernel is missing"]))?;
    let residual = pvlc_wgsl::module(pvlc_runtime_core::KernelId::AddF32)
        .ok_or_else(|| js_layer_error(&["canonical decoder layer residual kernel is missing"]))?;
    Ok(LayerShaderSources {
        rms_norm: rms_norm.source.to_owned(),
        gemv: gemv.source.to_owned(),
        mrope: mrope.source.to_owned(),
        append: append.source.to_owned(),
        attention: attention.source.to_owned(),
        residual: residual.source.to_owned(),
    })
}

fn source_blake3(source: &str) -> [u8; 32] {
    *blake3::hash(source.as_bytes()).as_bytes()
}

fn blake3_hex(digest: &[u8; 32]) -> String {
    blake3::Hash::from_bytes(*digest).to_hex().to_string()
}

fn layer_shader_digests(sources: &LayerShaderSources) -> LayerShaderDigests {
    LayerShaderDigests {
        rms_norm: source_blake3(&sources.rms_norm),
        gemv: source_blake3(&sources.gemv),
        mrope: source_blake3(&sources.mrope),
        append: source_blake3(&sources.append),
        attention: source_blake3(&sources.attention),
        residual: source_blake3(&sources.residual),
    }
}

fn validate_layer_capabilities(
    device: &wgpu::Device,
    kv_plan: &pvlc_runtime_core::DecoderKvSessionPlan,
    attention_plan: &pvlc_runtime_core::DecoderAttentionBlockPlan,
) -> Result<(), JsValue> {
    let limits = device.limits();
    if limits.max_storage_buffers_per_shader_stage < 6 {
        return Err(js_layer_error(&[
            "decoder layer session requires six storage buffers per shader stage",
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
    for invocation in [
        &attention_plan.rms_norm_invocation,
        &attention_plan.query_invocation,
        &attention_plan.key_invocation,
        &attention_plan.value_invocation,
        &attention_plan.mrope_invocation,
        &kv_plan.append_invocation,
        &kv_plan.attention_invocation,
        &attention_plan.output_invocation,
        &attention_plan.residual_invocation,
    ] {
        dispatch_limits.validate(invocation).map_err(|error| {
            js_layer_error(&[
                "decoder layer session exceeds adapter dispatch limits: ",
                &error.to_string(),
            ])
        })?;
    }
    let hidden_bytes =
        checked_u64_bytes(attention_plan.hidden_size as usize, "decoder hidden row")?;
    let query_bytes = checked_u64_bytes(attention_plan.query_width, "decoder query row")?;
    let key_value_bytes =
        checked_u64_bytes(attention_plan.key_value_width, "decoder key/value row")?;
    let rope_bytes = checked_u64_bytes(attention_plan.rope_elements, "decoder rope table")?;
    let q_weight_bytes = checked_u64_weight_bytes(
        attention_plan.query_width,
        attention_plan.hidden_size as usize,
        "decoder query weight",
    )?;
    let k_weight_bytes = checked_u64_weight_bytes(
        attention_plan.key_value_width,
        attention_plan.hidden_size as usize,
        "decoder key weight",
    )?;
    for (label, bytes) in [
        ("decoder layer query weight", q_weight_bytes),
        ("decoder layer key weight", k_weight_bytes),
        ("decoder layer rope table", rope_bytes),
        ("decoder layer compact cache", kv_plan.cache_bytes),
        ("decoder layer attention output", kv_plan.attention_bytes),
        ("decoder layer hidden row", hidden_bytes),
        ("decoder layer query row", query_bytes),
        ("decoder layer key/value row", key_value_bytes),
    ] {
        if bytes > limits.max_storage_buffer_binding_size {
            return Err(js_layer_error(&[
                label,
                " exceeds the adapter storage buffer binding limit",
            ]));
        }
    }
    if q_weight_bytes > limits.max_buffer_size || kv_plan.cache_bytes > limits.max_buffer_size {
        return Err(js_layer_error(&[
            "decoder layer session allocations exceed the adapter buffer size limit",
        ]));
    }
    Ok(())
}

fn js_object_set(object: &js_sys::Object, key: &str, value: &JsValue) -> Result<(), JsValue> {
    js_sys::Reflect::set(object, &JsValue::from_str(key), value)
        .map(|_| ())
        .map_err(|error| {
            js_layer_error(&[
                "cannot build WebGPU descriptor field ",
                key,
                ": ",
                &js_error_text(&error),
            ])
        })
}

fn create_layer_buffer(
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
    raw_layer_method(device, "createBuffer")?
        .call1(device, &descriptor)
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer session buffer creation failed: ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<wgpu::webgpu::GpuBuffer>()
        .map_err(|_| js_layer_error(&["createBuffer did not return a GPUBuffer"]))
}

fn create_layer_pipeline(
    device: &JsValue,
    kernel: &str,
    source: &str,
) -> Result<js_sys::Object, JsValue> {
    let shader_descriptor = js_sys::Object::new();
    js_object_set(&shader_descriptor, "label", &JsValue::from_str(kernel))?;
    js_object_set(&shader_descriptor, "code", &JsValue::from_str(source))?;
    let shader = raw_layer_method(device, "createShaderModule")?
        .call1(device, &shader_descriptor)
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer shader module creation failed: ",
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
    raw_layer_method(device, "createComputePipeline")?
        .call1(device, &pipeline_descriptor)
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer compute pipeline creation failed: ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<js_sys::Object>()
        .map_err(|_| js_layer_error(&["createComputePipeline did not return an object"]))
}

fn create_layer_bind_group(
    device: &JsValue,
    label: &str,
    pipeline: &js_sys::Object,
    entries: &[(&wgpu::webgpu::GpuBuffer, u64)],
) -> Result<js_sys::Object, JsValue> {
    let layout = raw_layer_method(pipeline, "getBindGroupLayout")?
        .call1(pipeline, &JsValue::from_f64(0.0))
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer bind group layout request failed: ",
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
    raw_layer_method(device, "createBindGroup")?
        .call1(device, &descriptor)
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer bind group creation failed: ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<js_sys::Object>()
        .map_err(|_| js_layer_error(&["createBindGroup did not return an object"]))
}

fn write_layer_buffer(
    queue: &JsValue,
    buffer: &wgpu::webgpu::GpuBuffer,
    bytes: &[u8],
) -> Result<(), JsValue> {
    let data = js_sys::Uint8Array::from(bytes);
    raw_layer_method(queue, "writeBuffer")?
        .call3(queue, buffer.as_ref(), &JsValue::from_f64(0.0), &data)
        .map(|_| ())
        .map_err(|error| {
            js_layer_error(&["decoder layer queue write failed: ", &js_error_text(&error)])
        })
}

fn create_layer_encoder(device: &JsValue, label: &str) -> Result<JsValue, JsValue> {
    let descriptor = js_sys::Object::new();
    js_object_set(&descriptor, "label", &JsValue::from_str(label))?;
    raw_layer_method(device, "createCommandEncoder")?
        .call1(device, &descriptor)
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer command encoder creation failed: ",
                &js_error_text(&error),
            ])
        })
}

fn encode_layer_pass(
    encoder: &JsValue,
    pipeline: &js_sys::Object,
    bind_group: &js_sys::Object,
    dispatch: [u32; 3],
) -> Result<(), JsValue> {
    let pass = raw_layer_method(encoder, "beginComputePass")?
        .call0(encoder)
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer compute pass begin failed: ",
                &js_error_text(&error),
            ])
        })?;
    raw_layer_method(&pass, "setPipeline")?
        .call1(&pass, pipeline)
        .map_err(|error| {
            js_layer_error(&["decoder layer setPipeline failed: ", &js_error_text(&error)])
        })?;
    raw_layer_method(&pass, "setBindGroup")?
        .call2(&pass, &JsValue::from_f64(0.0), bind_group)
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer setBindGroup failed: ",
                &js_error_text(&error),
            ])
        })?;
    raw_layer_method(&pass, "dispatchWorkgroups")?
        .call3(
            &pass,
            &JsValue::from_f64(f64::from(dispatch[0])),
            &JsValue::from_f64(f64::from(dispatch[1])),
            &JsValue::from_f64(f64::from(dispatch[2])),
        )
        .map_err(|error| {
            js_layer_error(&["decoder layer dispatch failed: ", &js_error_text(&error)])
        })?;
    raw_layer_method(&pass, "end")?
        .call0(&pass)
        .map(|_| ())
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer compute pass end failed: ",
                &js_error_text(&error),
            ])
        })
}

fn encode_layer_copy(
    encoder: &JsValue,
    source: &wgpu::webgpu::GpuBuffer,
    source_offset: u64,
    destination: &wgpu::webgpu::GpuBuffer,
    destination_offset: u64,
    bytes: u64,
) -> Result<(), JsValue> {
    raw_layer_method(encoder, "copyBufferToBuffer")?
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
            js_layer_error(&["decoder layer buffer copy failed: ", &js_error_text(&error)])
        })
}

fn submit_layer_encoder(queue: &JsValue, encoder: &JsValue) -> Result<(), JsValue> {
    let command = raw_layer_method(encoder, "finish")?
        .call0(encoder)
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer command encoder finish failed: ",
                &js_error_text(&error),
            ])
        })?;
    let commands = js_sys::Array::new();
    commands.push(&command);
    raw_layer_method(queue, "submit")?
        .call1(queue, &commands)
        .map(|_| ())
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer queue submission failed: ",
                &js_error_text(&error),
            ])
        })
}

async fn await_layer_queue_completion(queue: &JsValue) -> Result<(), JsValue> {
    let pending = raw_layer_method(queue, "onSubmittedWorkDone")?
        .call0(queue)
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer queue completion request failed: ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<js_sys::Promise>()
        .map_err(|_| js_layer_error(&["onSubmittedWorkDone did not return a Promise"]))?;
    JsFuture::from(pending).await.map(|_| ()).map_err(|error| {
        js_layer_error(&[
            "decoder layer queue completion rejected: ",
            &js_error_text(&error),
        ])
    })
}

async fn map_layer_buffer(buffer: &JsValue, bytes: u64) -> Result<(), JsValue> {
    let pending = raw_layer_method(buffer, "mapAsync")?
        .call3(
            buffer,
            &JsValue::from_f64(1.0),
            &JsValue::from_f64(0.0),
            &JsValue::from_f64(bytes as f64),
        )
        .map_err(|error| {
            js_layer_error(&["decoder layer map request failed: ", &js_error_text(&error)])
        })?
        .dyn_into::<js_sys::Promise>()
        .map_err(|_| js_layer_error(&["mapAsync did not return a Promise"]))?;
    JsFuture::from(pending).await.map(|_| ()).map_err(|error| {
        js_layer_error(&[
            "decoder layer buffer mapping rejected: ",
            &js_error_text(&error),
        ])
    })
}

fn read_layer_mapped(buffer: &JsValue, bytes: u64) -> Result<Vec<u8>, JsValue> {
    let range = raw_layer_method(buffer, "getMappedRange")?
        .call2(
            buffer,
            &JsValue::from_f64(0.0),
            &JsValue::from_f64(bytes as f64),
        )
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer mapped range read failed: ",
                &js_error_text(&error),
            ])
        })?;
    Ok(js_sys::Uint8Array::new(&range).to_vec())
}

fn unmap_layer_buffer(buffer: &JsValue) -> Result<(), JsValue> {
    raw_layer_method(buffer, "unmap")?
        .call0(buffer)
        .map(|_| ())
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer buffer unmap failed: ",
                &js_error_text(&error),
            ])
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
        .ok_or_else(|| js_layer_error(&[label, " byte size overflowed"]))
}

fn checked_u64_weight_bytes(rows: usize, columns: usize, label: &str) -> Result<u64, JsValue> {
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| js_layer_error(&[label, " element count overflowed"]))?;
    checked_u64_bytes(elements, label)
}

fn pack_u16_le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn pack_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn pack_u64_le(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

fn pack_align_up(value: usize, alignment: usize) -> usize {
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + alignment - remainder
    }
}

fn pack_usize(value: u64, label: &str) -> Result<usize, JsValue> {
    usize::try_from(value)
        .map_err(|_| js_layer_error(&["decoder layer weight pack ", label, " offset is too large"]))
}

fn pack_slice<'a>(
    bytes: &'a [u8],
    start: usize,
    end: usize,
    label: &str,
) -> Result<&'a [u8], JsValue> {
    if start > end || end > bytes.len() {
        return Err(js_layer_error(&[
            "decoder layer weight pack ",
            label,
            " range exceeds the file",
        ]));
    }
    Ok(&bytes[start..end])
}

fn pack_padding_is_zero(bytes: &[u8]) -> bool {
    bytes.iter().all(|byte| *byte == 0)
}

fn pack_hex_nibble(byte: u8, label: &str) -> Result<u8, JsValue> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(js_layer_error(&[
            "decoder layer weight pack ",
            label,
            " is not lowercase BLAKE3 hex",
        ])),
    }
}

fn pack_hex_digest(hex: &str, label: &str) -> Result<[u8; 32], JsValue> {
    let digits = hex.as_bytes();
    if digits.len() != 64 {
        return Err(js_layer_error(&[
            "decoder layer weight pack ",
            label,
            " is not a 64-digit BLAKE3 hex string",
        ]));
    }
    let mut digest = [0u8; 32];
    for (index, slot) in digest.iter_mut().enumerate() {
        let high = pack_hex_nibble(digits[index * 2], label)?;
        let low = pack_hex_nibble(digits[index * 2 + 1], label)?;
        *slot = (high << 4) | low;
    }
    Ok(digest)
}

fn pack_json_object(payload: &[u8], label: &str) -> Result<Map<String, Value>, JsValue> {
    let text = std::str::from_utf8(payload).map_err(|error| {
        js_layer_error(&[
            "decoder layer weight pack ",
            label,
            " UTF-8 is invalid: ",
            &error.to_string(),
        ])
    })?;
    if !text.ends_with('\n') || text.ends_with("\n\n") {
        return Err(js_layer_error(&[
            "decoder layer weight pack ",
            label,
            " is not newline-terminated canonical JSON",
        ]));
    }
    let value: Value = serde_json::from_str(text).map_err(|error| {
        js_layer_error(&[
            "decoder layer weight pack ",
            label,
            " JSON is invalid: ",
            &error.to_string(),
        ])
    })?;
    value.as_object().cloned().ok_or_else(|| {
        js_layer_error(&[
            "decoder layer weight pack ",
            label,
            " must be a JSON object",
        ])
    })
}

fn require_pack_exact_keys(
    object: &Map<String, Value>,
    expected: &[&str],
    label: &str,
) -> Result<(), JsValue> {
    for key in object.keys() {
        if !expected.contains(&key.as_str()) {
            return Err(js_layer_error(&[
                "decoder layer weight pack ",
                label,
                " has unknown field ",
                key,
            ]));
        }
    }
    for key in expected {
        if !object.contains_key(*key) {
            return Err(js_layer_error(&[
                "decoder layer weight pack ",
                label,
                " is missing field ",
                key,
            ]));
        }
    }
    Ok(())
}

fn pack_json_u64(object: &Map<String, Value>, key: &str, label: &str) -> Result<u64, JsValue> {
    object.get(key).and_then(Value::as_u64).ok_or_else(|| {
        js_layer_error(&[
            "decoder layer weight pack ",
            label,
            " field ",
            key,
            " must be an unsigned integer",
        ])
    })
}

fn pack_json_str<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a str, JsValue> {
    object.get(key).and_then(Value::as_str).ok_or_else(|| {
        js_layer_error(&[
            "decoder layer weight pack ",
            label,
            " field ",
            key,
            " must be a string",
        ])
    })
}

fn validate_pack_manifest(payload: &[u8]) -> Result<(), JsValue> {
    let manifest = pack_json_object(payload, "manifest")?;
    require_pack_exact_keys(&manifest, &PACK_MANIFEST_FIELDS, "manifest")?;
    if pack_json_str(&manifest, "model_id", "manifest")? != PACK_MODEL_ID {
        return Err(js_layer_error(&[
            "decoder layer weight pack manifest model_id drifted",
        ]));
    }
    if pack_json_str(&manifest, "model_revision", "manifest")? != PACK_MODEL_REVISION {
        return Err(js_layer_error(&[
            "decoder layer weight pack manifest model_revision drifted",
        ]));
    }
    if pack_json_u64(&manifest, "compiler_model_abi", "manifest")? != 1 {
        return Err(js_layer_error(&[
            "decoder layer weight pack manifest compiler_model_abi drifted",
        ]));
    }
    pack_hex_digest(
        pack_json_str(&manifest, "compiler_build", "manifest")?,
        "manifest compiler_build",
    )?;
    if pack_json_str(&manifest, "precision_profile", "manifest")? != "fidelity" {
        return Err(js_layer_error(&[
            "decoder layer weight pack manifest precision_profile drifted",
        ]));
    }
    if pack_json_u64(&manifest, "context_limit", "manifest")? == 0 {
        return Err(js_layer_error(&[
            "decoder layer weight pack manifest context_limit drifted",
        ]));
    }
    let buckets = manifest
        .get("resolution_buckets")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            js_layer_error(&[
                "decoder layer weight pack manifest resolution_buckets must be an array",
            ])
        })?;
    if buckets.is_empty() {
        return Err(js_layer_error(&[
            "decoder layer weight pack manifest resolution_buckets drifted",
        ]));
    }
    for bucket in buckets {
        let pair = bucket.as_array().ok_or_else(|| {
            js_layer_error(&["decoder layer weight pack manifest resolution bucket must be a pair"])
        })?;
        if pair.len() != 2
            || pair
                .iter()
                .any(|side| side.as_u64().is_none_or(|side| side == 0))
        {
            return Err(js_layer_error(&[
                "decoder layer weight pack manifest resolution bucket drifted",
            ]));
        }
    }
    Ok(())
}

fn expected_shard_bytes(shard_index: usize, cache_capacity: u32) -> Result<u64, JsValue> {
    let hidden = pvlc_runtime_core::PINNED_DECODER_HIDDEN_SIZE;
    let query_width = PINNED_LAYER_QUERY_WIDTH;
    let key_value_width = PINNED_LAYER_KEY_VALUE_WIDTH;
    match shard_index {
        0 => checked_u64_bytes(hidden as usize, "decoder layer norm weight"),
        1 => checked_u64_weight_bytes(
            query_width as usize,
            hidden as usize,
            "decoder layer query weight",
        ),
        2 | 3 => checked_u64_weight_bytes(
            key_value_width as usize,
            hidden as usize,
            "decoder layer key/value weight",
        ),
        4 => checked_u64_weight_bytes(
            hidden as usize,
            query_width as usize,
            "decoder layer output weight",
        ),
        _ => {
            let elements = 3usize
                .checked_mul(cache_capacity as usize)
                .and_then(|value| {
                    value.checked_mul(pvlc_runtime_core::MAX_DECODER_HEAD_DIM as usize)
                })
                .ok_or_else(|| {
                    js_layer_error(&["decoder layer rope table element count overflowed"])
                })?;
            checked_u64_bytes(elements, "decoder layer rope table")
        }
    }
}

fn require_pack_f32_finite(payload: &[u8], shard_id: &str) -> Result<(), JsValue> {
    for word in payload.chunks_exact(4) {
        if !f32::from_le_bytes([word[0], word[1], word[2], word[3]]).is_finite() {
            return Err(js_layer_error(&[
                "decoder layer weight pack shard ",
                shard_id,
                " contains a nonfinite f32 payload element",
            ]));
        }
    }
    Ok(())
}

fn validate_pack_descriptor(
    payload: &[u8],
    expected_prefix_tokens: u32,
    expected_cache_capacity: u32,
) -> Result<Map<String, Value>, JsValue> {
    let descriptor = pack_json_object(payload, "descriptor")?;
    require_pack_exact_keys(&descriptor, &PACK_DESCRIPTOR_FIELDS, "descriptor")?;
    if pack_json_u64(&descriptor, "schema_version", "descriptor")? != 1 {
        return Err(js_layer_error(&[
            "decoder layer weight pack descriptor schema version drifted",
        ]));
    }
    let oracle = pack_json_str(&descriptor, "oracle", "descriptor")?;
    if !PACK_DESCRIPTOR_ORACLES.contains(&oracle) {
        return Err(js_layer_error(&[
            "decoder layer weight pack descriptor oracle is unsupported",
        ]));
    }
    if pack_json_str(&descriptor, "case_id", "descriptor")?.is_empty() {
        return Err(js_layer_error(&[
            "decoder layer weight pack descriptor case_id is empty",
        ]));
    }
    if pack_json_str(&descriptor, "model_revision", "descriptor")? != PACK_MODEL_REVISION {
        return Err(js_layer_error(&[
            "decoder layer weight pack descriptor model_revision drifted",
        ]));
    }
    for (field, pinned) in [
        ("hidden_size", pvlc_runtime_core::PINNED_DECODER_HIDDEN_SIZE),
        ("query_heads", pvlc_runtime_core::PINNED_DECODER_QUERY_HEADS),
        (
            "key_value_heads",
            pvlc_runtime_core::PINNED_DECODER_KEY_VALUE_HEADS,
        ),
        ("head_dim", pvlc_runtime_core::MAX_DECODER_HEAD_DIM),
        ("query_width", PINNED_LAYER_QUERY_WIDTH),
        ("key_value_width", PINNED_LAYER_KEY_VALUE_WIDTH),
    ] {
        if pack_json_u64(&descriptor, field, "descriptor")? != u64::from(pinned) {
            return Err(js_layer_error(&[
                "decoder layer weight pack descriptor geometry field ",
                field,
                " drifted from the pinned decoder value",
            ]));
        }
    }
    let prefix_tokens = pack_json_u64(&descriptor, "prefix_tokens", "descriptor")?;
    let cache_capacity = pack_json_u64(&descriptor, "cache_capacity", "descriptor")?;
    if prefix_tokens == 0 || cache_capacity <= prefix_tokens {
        return Err(js_layer_error(&[
            "decoder layer weight pack descriptor cache geometry is invalid",
        ]));
    }
    if prefix_tokens != u64::from(expected_prefix_tokens)
        || cache_capacity != u64::from(expected_cache_capacity)
    {
        return Err(js_layer_error(&[
            "decoder layer weight pack descriptor cache geometry does not match the session descriptor",
        ]));
    }
    let epsilon = descriptor
        .get("rms_norm_epsilon")
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            js_layer_error(&[
                "decoder layer weight pack descriptor rms_norm_epsilon must be a number",
            ])
        })? as f32;
    if epsilon != pvlc_runtime_core::PINNED_DECODER_RMS_NORM_EPSILON {
        return Err(js_layer_error(&[
            "decoder layer weight pack descriptor rms_norm_epsilon drifted",
        ]));
    }
    let sections = descriptor
        .get("mrope_sections")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            js_layer_error(&[
                "decoder layer weight pack descriptor mrope_sections must be an array",
            ])
        })?;
    let pinned = pvlc_runtime_core::PINNED_DECODER_MROPE_SECTIONS;
    if sections.len() != pinned.len()
        || sections
            .iter()
            .enumerate()
            .any(|(index, section)| section.as_u64() != Some(pinned[index] as u64))
    {
        return Err(js_layer_error(&[
            "decoder layer weight pack descriptor mrope_sections drifted",
        ]));
    }
    let shards = descriptor
        .get("shards")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            js_layer_error(&["decoder layer weight pack descriptor shards must be an object"])
        })?;
    require_pack_exact_keys(shards, &PACK_SHARD_IDS, "descriptor shards")?;
    Ok(shards.clone())
}

fn parse_layer_weight_pack(
    pack_bytes: &[u8],
    expected_prefix_tokens: u32,
    expected_cache_capacity: u32,
) -> Result<ParsedWeightPack, JsValue> {
    if pack_bytes.len() < PACK_HEADER_BYTES {
        return Err(js_layer_error(&[
            "decoder layer weight pack is shorter than its header",
        ]));
    }
    if pack_bytes[..8] != PACK_MAGIC {
        return Err(js_layer_error(&[
            "decoder layer weight pack magic mismatch",
        ]));
    }
    if pack_u32_le(pack_bytes, 8) != PACK_VERSION {
        return Err(js_layer_error(&[
            "decoder layer weight pack version is unsupported",
        ]));
    }
    let manifest_length = u64::from(pack_u32_le(pack_bytes, 12));
    let directory_length = u64::from(pack_u32_le(pack_bytes, 16));
    if pack_u32_le(pack_bytes, 20) != PACK_SECTION_COUNT {
        return Err(js_layer_error(&[
            "decoder layer weight pack section count drifted",
        ]));
    }
    let pack_length = pack_u64_le(pack_bytes, 24);
    if pack_length != pack_bytes.len() as u64 {
        return Err(js_layer_error(&[
            "decoder layer weight pack declared length drifted",
        ]));
    }
    let manifest_end = pack_usize(
        u64::try_from(PACK_HEADER_BYTES)
            .ok()
            .and_then(|header| header.checked_add(manifest_length))
            .ok_or_else(|| {
                js_layer_error(&["decoder layer weight pack manifest length overflowed"])
            })?,
        "manifest",
    )?;
    let directory_end = pack_usize(
        u64::try_from(manifest_end)
            .ok()
            .and_then(|offset| offset.checked_add(directory_length))
            .ok_or_else(|| {
                js_layer_error(&["decoder layer weight pack directory length overflowed"])
            })?,
        "directory",
    )?;
    if directory_end > pack_bytes.len() {
        return Err(js_layer_error(&[
            "decoder layer weight pack prefix exceeds the file",
        ]));
    }
    validate_pack_manifest(pack_slice(
        pack_bytes,
        PACK_HEADER_BYTES,
        manifest_end,
        "manifest",
    )?)?;

    let expected_ids: [&str; 8] = [
        PACK_DESCRIPTOR_SECTION_ID,
        PACK_SHARD_IDS[0],
        PACK_SHARD_IDS[1],
        PACK_SHARD_IDS[2],
        PACK_SHARD_IDS[3],
        PACK_SHARD_IDS[4],
        PACK_SHARD_IDS[5],
        PACK_SHARD_IDS[6],
    ];
    let mut entries: Vec<PackDirectoryEntry> = Vec::new();
    let mut cursor = manifest_end;
    for (index, expected_id) in expected_ids.iter().enumerate() {
        let fixed_end = cursor + PACK_DIRECTORY_FIXED_BYTES;
        if fixed_end > directory_end {
            return Err(js_layer_error(&[
                "decoder layer weight pack directory fixed entry is truncated",
            ]));
        }
        let id_length = usize::from(pack_u16_le(pack_bytes, cursor));
        let kind = pack_bytes[cursor + 2];
        let reserved = pack_bytes[cursor + 3];
        let alignment = u64::from(pack_u32_le(pack_bytes, cursor + 4));
        let offset = pack_u64_le(pack_bytes, cursor + 8);
        let byte_length = pack_u64_le(pack_bytes, cursor + 16);
        let mut digest = [0u8; 32];
        digest.copy_from_slice(pack_slice(
            pack_bytes,
            cursor + 24,
            cursor + 56,
            "directory entry digest",
        )?);
        if reserved != 0 {
            return Err(js_layer_error(&[
                "decoder layer weight pack directory reserved byte is nonzero",
            ]));
        }
        let expected_kind = if index == 0 { 1 } else { 2 };
        if kind != expected_kind {
            return Err(js_layer_error(&[
                "decoder layer weight pack section ",
                expected_id,
                " kind drifted",
            ]));
        }
        if alignment == 0 || alignment > PACK_MAX_ALIGNMENT || !alignment.is_power_of_two() {
            return Err(js_layer_error(&[
                "decoder layer weight pack section ",
                expected_id,
                " alignment is invalid",
            ]));
        }
        let id_end = fixed_end + id_length;
        let entry_end = cursor
            + pack_align_up(
                PACK_DIRECTORY_FIXED_BYTES + id_length,
                PACK_DIRECTORY_ENTRY_ALIGNMENT,
            );
        if entry_end > directory_end || id_end > entry_end {
            return Err(js_layer_error(&[
                "decoder layer weight pack directory entry is truncated",
            ]));
        }
        let id = std::str::from_utf8(pack_slice(
            pack_bytes,
            fixed_end,
            id_end,
            "directory entry id",
        )?)
        .map_err(|error| {
            js_layer_error(&[
                "decoder layer weight pack directory entry id UTF-8 is invalid: ",
                &error.to_string(),
            ])
        })?;
        if id != *expected_id {
            return Err(js_layer_error(&[
                "decoder layer weight pack section order drifted at ",
                expected_id,
            ]));
        }
        if !pack_padding_is_zero(pack_slice(
            pack_bytes,
            id_end,
            entry_end,
            "directory padding",
        )?) {
            return Err(js_layer_error(&[
                "decoder layer weight pack directory entry ",
                expected_id,
                " contains nonzero padding",
            ]));
        }
        entries.push(PackDirectoryEntry {
            offset,
            byte_length,
            alignment,
            digest,
        });
        cursor = entry_end;
    }
    if cursor != directory_end {
        return Err(js_layer_error(&[
            "decoder layer weight pack directory length drifted",
        ]));
    }

    let mut previous_end = directory_end;
    let mut payloads: Vec<&[u8]> = Vec::new();
    for (index, entry) in entries.iter().enumerate() {
        let alignment = pack_usize(entry.alignment, "section alignment")?;
        let expected_offset = pack_align_up(previous_end, alignment);
        let offset = pack_usize(entry.offset, "section offset")?;
        let byte_length = pack_usize(entry.byte_length, "section length")?;
        let section_end = offset.checked_add(byte_length).ok_or_else(|| {
            js_layer_error(&["decoder layer weight pack section length overflowed"])
        })?;
        if offset != expected_offset || section_end > pack_bytes.len() {
            return Err(js_layer_error(&[
                "decoder layer weight pack section ",
                expected_ids[index],
                " layout is invalid",
            ]));
        }
        if !pack_padding_is_zero(pack_slice(
            pack_bytes,
            previous_end,
            offset,
            "section padding",
        )?) {
            return Err(js_layer_error(&[
                "decoder layer weight pack section ",
                expected_ids[index],
                " leading alignment contains nonzero padding",
            ]));
        }
        let payload = pack_slice(pack_bytes, offset, section_end, "section payload")?;
        if blake3::hash(payload).as_bytes() != &entry.digest {
            return Err(js_layer_error(&[
                "decoder layer weight pack section ",
                expected_ids[index],
                " BLAKE3 digest mismatch",
            ]));
        }
        payloads.push(payload);
        previous_end = section_end;
    }
    if previous_end != pack_bytes.len() {
        return Err(js_layer_error(&[
            "decoder layer weight pack has trailing or missing section bytes",
        ]));
    }

    let shards =
        validate_pack_descriptor(payloads[0], expected_prefix_tokens, expected_cache_capacity)?;
    let mut shard_payloads: Vec<&[u8]> = Vec::new();
    for (shard_index, shard_id) in PACK_SHARD_IDS.iter().enumerate() {
        let entry = &entries[shard_index + 1];
        let pin = shards
            .get(*shard_id)
            .and_then(Value::as_object)
            .ok_or_else(|| {
                js_layer_error(&[
                    "decoder layer weight pack shard ",
                    shard_id,
                    " pin must be an object",
                ])
            })?;
        require_pack_exact_keys(pin, &["bytes", "blake3"], "shard pin")?;
        let expected_bytes = expected_shard_bytes(shard_index, expected_cache_capacity)?;
        if pack_json_u64(pin, "bytes", "shard pin")? != expected_bytes
            || entry.byte_length != expected_bytes
        {
            return Err(js_layer_error(&[
                "decoder layer weight pack shard ",
                shard_id,
                " declared length drifted",
            ]));
        }
        if !entry.byte_length.is_multiple_of(4) {
            return Err(js_layer_error(&[
                "decoder layer weight pack shard ",
                shard_id,
                " is not f32 aligned",
            ]));
        }
        let pinned_digest = pack_json_str(pin, "blake3", "shard pin")?;
        if pack_hex_digest(pinned_digest, "shard pin blake3")? != entry.digest {
            return Err(js_layer_error(&[
                "decoder layer weight pack shard ",
                shard_id,
                " BLAKE3 pin drifted",
            ]));
        }
        let payload = payloads[shard_index + 1];
        require_pack_f32_finite(payload, shard_id)?;
        shard_payloads.push(payload);
    }
    Ok(ParsedWeightPack {
        norm1_weight: shard_payloads[0].to_vec(),
        q_weight: shard_payloads[1].to_vec(),
        k_weight: shard_payloads[2].to_vec(),
        v_weight: shard_payloads[3].to_vec(),
        o_weight: shard_payloads[4].to_vec(),
        rope_cos: shard_payloads[5].to_vec(),
        rope_sin: shard_payloads[6].to_vec(),
    })
}

impl BrowserDecoderLayerSession {
    fn create(
        device: &JsValue,
        kv_plan: pvlc_runtime_core::DecoderKvSessionPlan,
        attention_plan: pvlc_runtime_core::DecoderAttentionBlockPlan,
        sources: &LayerShaderSources,
    ) -> Result<BrowserDecoderLayerSession, JsValue> {
        let storage_copy_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let copy_src_usage = storage_copy_dst | wgpu::BufferUsages::COPY_SRC;
        let readback_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let hidden_bytes =
            checked_u64_bytes(attention_plan.hidden_size as usize, "decoder hidden row")?;
        let query_bytes = checked_u64_bytes(attention_plan.query_width, "decoder query row")?;
        let key_value_bytes =
            checked_u64_bytes(attention_plan.key_value_width, "decoder key/value row")?;
        let rope_bytes = checked_u64_bytes(attention_plan.rope_elements, "decoder rope table")?;
        let q_weight_bytes = checked_u64_weight_bytes(
            attention_plan.query_width,
            attention_plan.hidden_size as usize,
            "decoder query weight",
        )?;
        let k_weight_bytes = checked_u64_weight_bytes(
            attention_plan.key_value_width,
            attention_plan.hidden_size as usize,
            "decoder key/value weight",
        )?;
        let o_weight_bytes = checked_u64_weight_bytes(
            attention_plan.hidden_size as usize,
            attention_plan.query_width,
            "decoder output weight",
        )?;
        let hidden_token_buffer = create_layer_buffer(
            device,
            BUFFER_HIDDEN_TOKEN,
            hidden_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let norm1_weight_buffer = create_layer_buffer(
            device,
            BUFFER_NORM1_WEIGHT,
            hidden_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let q_weight_buffer = create_layer_buffer(
            device,
            BUFFER_Q_WEIGHT,
            q_weight_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let k_weight_buffer = create_layer_buffer(
            device,
            BUFFER_K_WEIGHT,
            k_weight_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let v_weight_buffer = create_layer_buffer(
            device,
            BUFFER_V_WEIGHT,
            k_weight_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let o_weight_buffer = create_layer_buffer(
            device,
            BUFFER_O_WEIGHT,
            o_weight_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let rope_cos_buffer = create_layer_buffer(
            device,
            BUFFER_ROPE_COS,
            rope_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let rope_sin_buffer = create_layer_buffer(
            device,
            BUFFER_ROPE_SIN,
            rope_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let norm1_buffer = create_layer_buffer(
            device,
            BUFFER_NORM1,
            hidden_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let q_projection_buffer = create_layer_buffer(
            device,
            BUFFER_Q_PROJECTION,
            query_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let k_projection_buffer = create_layer_buffer(
            device,
            BUFFER_K_PROJECTION,
            key_value_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let v_projection_buffer = create_layer_buffer(
            device,
            BUFFER_V_PROJECTION,
            key_value_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let mrope_query_buffer = create_layer_buffer(
            device,
            BUFFER_MROPE_QUERY,
            query_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let mrope_key_buffer = create_layer_buffer(
            device,
            BUFFER_MROPE_KEY,
            key_value_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let key_cache_buffer = create_layer_buffer(
            device,
            BUFFER_KEY_CACHE,
            kv_plan.cache_bytes,
            buffer_usage(&[copy_src_usage]),
        )?;
        let value_cache_buffer = create_layer_buffer(
            device,
            BUFFER_VALUE_CACHE,
            kv_plan.cache_bytes,
            buffer_usage(&[copy_src_usage]),
        )?;
        let attention_output_buffer = create_layer_buffer(
            device,
            BUFFER_ATTENTION_OUTPUT,
            kv_plan.attention_bytes,
            buffer_usage(&[copy_src_usage]),
        )?;
        let o_projection_buffer = create_layer_buffer(
            device,
            BUFFER_O_PROJECTION,
            hidden_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let attention_residual_buffer = create_layer_buffer(
            device,
            BUFFER_ATTENTION_RESIDUAL,
            hidden_bytes,
            buffer_usage(&[copy_src_usage]),
        )?;
        let layer_readback_buffer = create_layer_buffer(
            device,
            BUFFER_LAYER_READBACK,
            hidden_bytes,
            buffer_usage(&[readback_usage]),
        )?;
        let rms_uniform_buffer = create_layer_buffer(
            device,
            BUFFER_RMS_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let gemv_q_uniform_buffer = create_layer_buffer(
            device,
            BUFFER_GEMV_Q_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let gemv_k_uniform_buffer = create_layer_buffer(
            device,
            BUFFER_GEMV_K_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let gemv_v_uniform_buffer = create_layer_buffer(
            device,
            BUFFER_GEMV_V_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let mrope_uniform_buffer = create_layer_buffer(
            device,
            BUFFER_MROPE_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let append_uniform_buffer = create_layer_buffer(
            device,
            BUFFER_APPEND_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let attention_uniform_buffer = create_layer_buffer(
            device,
            BUFFER_ATTENTION_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let gemv_o_uniform_buffer = create_layer_buffer(
            device,
            BUFFER_GEMV_O_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let residual_uniform_buffer = create_layer_buffer(
            device,
            BUFFER_RESIDUAL_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let rms_norm_pipeline =
            create_layer_pipeline(device, RMS_NORM_KERNEL_NAME, &sources.rms_norm)?;
        let gemv_pipeline = create_layer_pipeline(device, GEMV_KERNEL_NAME, &sources.gemv)?;
        let mrope_pipeline = create_layer_pipeline(device, MROPE_KERNEL_NAME, &sources.mrope)?;
        let append_pipeline = create_layer_pipeline(device, APPEND_KERNEL_NAME, &sources.append)?;
        let attention_pipeline =
            create_layer_pipeline(device, ATTENTION_KERNEL_NAME, &sources.attention)?;
        let residual_pipeline =
            create_layer_pipeline(device, RESIDUAL_KERNEL_NAME, &sources.residual)?;
        let mut session = BrowserDecoderLayerSession {
            kv_plan,
            attention_plan,
            cache_tokens: kv_plan.initial_cache_tokens,
            poisoned: false,
            ready: false,
            rms_norm_shader_blake3: source_blake3(&sources.rms_norm),
            gemv_shader_blake3: source_blake3(&sources.gemv),
            mrope_shader_blake3: source_blake3(&sources.mrope),
            append_shader_blake3: source_blake3(&sources.append),
            attention_shader_blake3: source_blake3(&sources.attention),
            residual_shader_blake3: source_blake3(&sources.residual),
            hidden_token_buffer,
            norm1_weight_buffer,
            q_weight_buffer,
            k_weight_buffer,
            v_weight_buffer,
            o_weight_buffer,
            rope_cos_buffer,
            rope_sin_buffer,
            norm1_buffer,
            q_projection_buffer,
            k_projection_buffer,
            v_projection_buffer,
            mrope_query_buffer,
            mrope_key_buffer,
            key_cache_buffer,
            value_cache_buffer,
            attention_output_buffer,
            o_projection_buffer,
            attention_residual_buffer,
            layer_readback_buffer,
            rms_uniform_buffer,
            gemv_q_uniform_buffer,
            gemv_k_uniform_buffer,
            gemv_v_uniform_buffer,
            mrope_uniform_buffer,
            append_uniform_buffer,
            attention_uniform_buffer,
            gemv_o_uniform_buffer,
            residual_uniform_buffer,
            rms_norm_pipeline,
            gemv_pipeline,
            mrope_pipeline,
            append_pipeline,
            attention_pipeline,
            residual_pipeline,
            rms_bind_group: js_sys::Object::default(),
            gemv_q_bind_group: js_sys::Object::default(),
            gemv_k_bind_group: js_sys::Object::default(),
            gemv_v_bind_group: js_sys::Object::default(),
            mrope_bind_group: js_sys::Object::default(),
            append_bind_group: js_sys::Object::default(),
            attention_bind_group: js_sys::Object::default(),
            gemv_o_bind_group: js_sys::Object::default(),
            residual_bind_group: js_sys::Object::default(),
        };
        session.create_bind_groups(device)?;
        Ok(session)
    }

    /// Creates the nine persistent bind groups from the exact session buffer
    /// and pipeline fields after the session owns them.
    fn create_bind_groups(&mut self, device: &JsValue) -> Result<(), JsValue> {
        let hidden_bytes = checked_u64_bytes(
            self.attention_plan.hidden_size as usize,
            "decoder hidden row",
        )?;
        let query_bytes = checked_u64_bytes(self.attention_plan.query_width, "decoder query row")?;
        let key_value_bytes =
            checked_u64_bytes(self.attention_plan.key_value_width, "decoder key/value row")?;
        let rope_bytes =
            checked_u64_bytes(self.attention_plan.rope_elements, "decoder rope table")?;
        let q_weight_bytes = checked_u64_weight_bytes(
            self.attention_plan.query_width,
            self.attention_plan.hidden_size as usize,
            "decoder query weight",
        )?;
        let k_weight_bytes = checked_u64_weight_bytes(
            self.attention_plan.key_value_width,
            self.attention_plan.hidden_size as usize,
            "decoder key/value weight",
        )?;
        let o_weight_bytes = checked_u64_weight_bytes(
            self.attention_plan.hidden_size as usize,
            self.attention_plan.query_width,
            "decoder output weight",
        )?;
        self.rms_bind_group = create_layer_bind_group(
            device,
            "decoder-layer-session-rms-bind-group",
            &self.rms_norm_pipeline,
            &[
                (&self.hidden_token_buffer, hidden_bytes),
                (&self.norm1_weight_buffer, hidden_bytes),
                (&self.norm1_buffer, hidden_bytes),
                (&self.rms_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.gemv_q_bind_group = create_layer_bind_group(
            device,
            "decoder-layer-session-gemv-q-bind-group",
            &self.gemv_pipeline,
            &[
                (&self.q_weight_buffer, q_weight_bytes),
                (&self.norm1_buffer, hidden_bytes),
                (&self.q_projection_buffer, query_bytes),
                (&self.gemv_q_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.gemv_k_bind_group = create_layer_bind_group(
            device,
            "decoder-layer-session-gemv-k-bind-group",
            &self.gemv_pipeline,
            &[
                (&self.k_weight_buffer, k_weight_bytes),
                (&self.norm1_buffer, hidden_bytes),
                (&self.k_projection_buffer, key_value_bytes),
                (&self.gemv_k_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.gemv_v_bind_group = create_layer_bind_group(
            device,
            "decoder-layer-session-gemv-v-bind-group",
            &self.gemv_pipeline,
            &[
                (&self.v_weight_buffer, k_weight_bytes),
                (&self.norm1_buffer, hidden_bytes),
                (&self.v_projection_buffer, key_value_bytes),
                (&self.gemv_v_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.mrope_bind_group = create_layer_bind_group(
            device,
            "decoder-layer-session-mrope-bind-group",
            &self.mrope_pipeline,
            &[
                (&self.q_projection_buffer, query_bytes),
                (&self.k_projection_buffer, key_value_bytes),
                (&self.rope_cos_buffer, rope_bytes),
                (&self.rope_sin_buffer, rope_bytes),
                (&self.mrope_query_buffer, query_bytes),
                (&self.mrope_key_buffer, key_value_bytes),
                (&self.mrope_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.append_bind_group = create_layer_bind_group(
            device,
            "decoder-layer-session-append-bind-group",
            &self.append_pipeline,
            &[
                (&self.mrope_key_buffer, key_value_bytes),
                (&self.v_projection_buffer, key_value_bytes),
                (&self.key_cache_buffer, self.kv_plan.cache_bytes),
                (&self.value_cache_buffer, self.kv_plan.cache_bytes),
                (&self.append_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.attention_bind_group = create_layer_bind_group(
            device,
            "decoder-layer-session-attention-bind-group",
            &self.attention_pipeline,
            &[
                (&self.mrope_query_buffer, query_bytes),
                (&self.key_cache_buffer, self.kv_plan.cache_bytes),
                (&self.value_cache_buffer, self.kv_plan.cache_bytes),
                (&self.attention_output_buffer, self.kv_plan.attention_bytes),
                (&self.attention_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.gemv_o_bind_group = create_layer_bind_group(
            device,
            "decoder-layer-session-gemv-o-bind-group",
            &self.gemv_pipeline,
            &[
                (&self.o_weight_buffer, o_weight_bytes),
                (&self.attention_output_buffer, self.kv_plan.attention_bytes),
                (&self.o_projection_buffer, hidden_bytes),
                (&self.gemv_o_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.residual_bind_group = create_layer_bind_group(
            device,
            "decoder-layer-session-residual-bind-group",
            &self.residual_pipeline,
            &[
                (&self.hidden_token_buffer, hidden_bytes),
                (&self.o_projection_buffer, hidden_bytes),
                (&self.attention_residual_buffer, hidden_bytes),
                (&self.residual_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        Ok(())
    }

    fn upload_initial_operands(
        &self,
        queue: &JsValue,
        operands: &LayerBeginOperands,
    ) -> Result<(), JsValue> {
        write_layer_buffer(queue, &self.key_cache_buffer, &operands.key_cache_bytes)?;
        write_layer_buffer(queue, &self.value_cache_buffer, &operands.value_cache_bytes)?;
        write_layer_buffer(
            queue,
            &self.norm1_weight_buffer,
            &operands.norm1_weight_bytes,
        )?;
        write_layer_buffer(queue, &self.q_weight_buffer, &operands.q_weight_bytes)?;
        write_layer_buffer(queue, &self.k_weight_buffer, &operands.k_weight_bytes)?;
        write_layer_buffer(queue, &self.v_weight_buffer, &operands.v_weight_bytes)?;
        write_layer_buffer(queue, &self.o_weight_buffer, &operands.o_weight_bytes)?;
        write_layer_buffer(queue, &self.rope_cos_buffer, &operands.rope_cos_bytes)?;
        write_layer_buffer(queue, &self.rope_sin_buffer, &operands.rope_sin_bytes)
    }

    fn encode_step(
        &self,
        device: &JsValue,
        queue: &JsValue,
        transition: &pvlc_runtime_core::DecoderKvSessionStepPlan,
        step_plan: &pvlc_runtime_core::DecoderAttentionBlockStepPlan,
        hidden_bytes: &[u8],
    ) -> Result<(), JsValue> {
        write_layer_buffer(queue, &self.hidden_token_buffer, hidden_bytes)?;
        write_layer_buffer(
            queue,
            &self.rms_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[0]),
        )?;
        write_layer_buffer(
            queue,
            &self.gemv_q_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[1]),
        )?;
        write_layer_buffer(
            queue,
            &self.gemv_k_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[2]),
        )?;
        write_layer_buffer(
            queue,
            &self.gemv_v_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[3]),
        )?;
        write_layer_buffer(
            queue,
            &self.mrope_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[4]),
        )?;
        write_layer_buffer(
            queue,
            &self.append_uniform_buffer,
            bytemuck::cast_slice(&transition.append.uniform_words),
        )?;
        write_layer_buffer(
            queue,
            &self.attention_uniform_buffer,
            bytemuck::cast_slice(&transition.attention.uniform_words),
        )?;
        write_layer_buffer(
            queue,
            &self.gemv_o_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[5]),
        )?;
        write_layer_buffer(
            queue,
            &self.residual_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[6]),
        )?;
        let encoder = create_layer_encoder(device, "decoder-layer-session-step-encoder")?;
        encode_layer_pass(
            &encoder,
            &self.rms_norm_pipeline,
            &self.rms_bind_group,
            self.attention_plan.rms_norm_invocation.dispatch,
        )?;
        encode_layer_pass(
            &encoder,
            &self.gemv_pipeline,
            &self.gemv_q_bind_group,
            self.attention_plan.query_invocation.dispatch,
        )?;
        encode_layer_pass(
            &encoder,
            &self.gemv_pipeline,
            &self.gemv_k_bind_group,
            self.attention_plan.key_invocation.dispatch,
        )?;
        encode_layer_pass(
            &encoder,
            &self.gemv_pipeline,
            &self.gemv_v_bind_group,
            self.attention_plan.value_invocation.dispatch,
        )?;
        encode_layer_pass(
            &encoder,
            &self.mrope_pipeline,
            &self.mrope_bind_group,
            self.attention_plan.mrope_invocation.dispatch,
        )?;
        encode_layer_pass(
            &encoder,
            &self.append_pipeline,
            &self.append_bind_group,
            self.kv_plan.append_invocation.dispatch,
        )?;
        encode_layer_pass(
            &encoder,
            &self.attention_pipeline,
            &self.attention_bind_group,
            self.kv_plan.attention_invocation.dispatch,
        )?;
        encode_layer_pass(
            &encoder,
            &self.gemv_pipeline,
            &self.gemv_o_bind_group,
            self.attention_plan.output_invocation.dispatch,
        )?;
        encode_layer_pass(
            &encoder,
            &self.residual_pipeline,
            &self.residual_bind_group,
            self.attention_plan.residual_invocation.dispatch,
        )?;
        let hidden_row_bytes = checked_u64_bytes(
            self.attention_plan.hidden_size as usize,
            "decoder hidden row",
        )?;
        encode_layer_copy(
            &encoder,
            &self.attention_residual_buffer,
            0,
            &self.layer_readback_buffer,
            0,
            hidden_row_bytes,
        )?;
        submit_layer_encoder(queue, &encoder)
    }

    fn encode_finish(
        &self,
        device: &JsValue,
        queue: &JsValue,
    ) -> Result<(wgpu::webgpu::GpuBuffer, wgpu::webgpu::GpuBuffer), JsValue> {
        let readback_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let key_readback = create_layer_buffer(
            device,
            BUFFER_FINISH_KEY_READBACK,
            self.kv_plan.cache_bytes,
            buffer_usage(&[readback_usage]),
        )?;
        let value_readback = create_layer_buffer(
            device,
            BUFFER_FINISH_VALUE_READBACK,
            self.kv_plan.cache_bytes,
            buffer_usage(&[readback_usage]),
        )?;
        let encoder = create_layer_encoder(device, "decoder-layer-session-finish-encoder")?;
        encode_layer_copy(
            &encoder,
            &self.key_cache_buffer,
            0,
            &key_readback,
            0,
            self.kv_plan.cache_bytes,
        )?;
        encode_layer_copy(
            &encoder,
            &self.value_cache_buffer,
            0,
            &value_readback,
            0,
            self.kv_plan.cache_bytes,
        )?;
        submit_layer_encoder(queue, &encoder)?;
        Ok((key_readback, value_readback))
    }
}

fn session_shader_digests(session: &BrowserDecoderLayerSession) -> LayerShaderDigests {
    LayerShaderDigests {
        rms_norm: session.rms_norm_shader_blake3,
        gemv: session.gemv_shader_blake3,
        mrope: session.mrope_shader_blake3,
        append: session.append_shader_blake3,
        attention: session.attention_shader_blake3,
        residual: session.residual_shader_blake3,
    }
}

fn shader_blake3_json(digests: &LayerShaderDigests) -> Value {
    let mut hashes = Map::new();
    hashes.insert(
        RMS_NORM_KERNEL_NAME.to_owned(),
        Value::from(blake3_hex(&digests.rms_norm).as_str()),
    );
    hashes.insert(
        GEMV_KERNEL_NAME.to_owned(),
        Value::from(blake3_hex(&digests.gemv).as_str()),
    );
    hashes.insert(
        MROPE_KERNEL_NAME.to_owned(),
        Value::from(blake3_hex(&digests.mrope).as_str()),
    );
    hashes.insert(
        APPEND_KERNEL_NAME.to_owned(),
        Value::from(blake3_hex(&digests.append).as_str()),
    );
    hashes.insert(
        ATTENTION_KERNEL_NAME.to_owned(),
        Value::from(blake3_hex(&digests.attention).as_str()),
    );
    hashes.insert(
        RESIDUAL_KERNEL_NAME.to_owned(),
        Value::from(blake3_hex(&digests.residual).as_str()),
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
        js_layer_error(&[
            "cannot serialize decoder layer diagnostics: ",
            &error.to_string(),
        ])
    })
}

fn creation_diagnostics_json(
    kv_plan: &pvlc_runtime_core::DecoderKvSessionPlan,
    attention_plan: &pvlc_runtime_core::DecoderAttentionBlockPlan,
    digests: &LayerShaderDigests,
) -> Result<String, JsValue> {
    let mut root = Map::new();
    root.insert("schema_version".to_owned(), Value::from(1u64));
    root.insert(
        "initial_cache_tokens".to_owned(),
        Value::from(u64::from(kv_plan.initial_cache_tokens)),
    );
    root.insert(
        "cache_capacity".to_owned(),
        Value::from(u64::from(kv_plan.cache_capacity)),
    );
    root.insert(
        "hidden_size".to_owned(),
        Value::from(u64::from(attention_plan.hidden_size)),
    );
    root.insert(
        "query_heads".to_owned(),
        Value::from(u64::from(kv_plan.query_heads)),
    );
    root.insert(
        "key_value_heads".to_owned(),
        Value::from(u64::from(kv_plan.key_value_heads)),
    );
    root.insert(
        "head_dim".to_owned(),
        Value::from(u64::from(kv_plan.head_dim)),
    );
    root.insert("checked_error_scopes".to_owned(), checked_scopes_json());
    root.insert("captured_errors".to_owned(), Value::Array(Vec::new()));
    root.insert("shader_blake3".to_owned(), shader_blake3_json(digests));
    root.insert("buffer_count".to_owned(), Value::from(29u64));
    root.insert("pipeline_count".to_owned(), Value::from(6u64));
    root.insert("bind_group_count".to_owned(), Value::from(9u64));
    root.insert("initial_upload_count".to_owned(), Value::from(9u64));
    json_text(Value::Object(root))
}

fn step_diagnostics_json(
    session: &BrowserDecoderLayerSession,
    transition: &pvlc_runtime_core::DecoderKvSessionStepPlan,
) -> Result<String, JsValue> {
    let mut root = Map::new();
    root.insert("schema_version".to_owned(), Value::from(1u64));
    root.insert(
        "cache_tokens_before".to_owned(),
        Value::from(u64::from(transition.cache_tokens_before)),
    );
    root.insert(
        "cache_tokens_after".to_owned(),
        Value::from(u64::from(transition.cache_tokens_after)),
    );
    root.insert("checked_error_scopes".to_owned(), checked_scopes_json());
    root.insert("captured_errors".to_owned(), Value::Array(Vec::new()));
    root.insert(
        "shader_blake3".to_owned(),
        shader_blake3_json(&session_shader_digests(session)),
    );
    root.insert("queue_write_count".to_owned(), Value::from(10u64));
    root.insert("command_encoder_count".to_owned(), Value::from(1u64));
    root.insert("compute_pass_count".to_owned(), Value::from(9u64));
    root.insert("dispatch_count".to_owned(), Value::from(9u64));
    root.insert("copy_count".to_owned(), Value::from(1u64));
    root.insert("command_buffer_count".to_owned(), Value::from(1u64));
    root.insert("submission_count".to_owned(), Value::from(1u64));
    root.insert("map_count".to_owned(), Value::from(1u64));
    let hidden_row_bytes = checked_u64_bytes(
        session.attention_plan.hidden_size as usize,
        "decoder hidden row",
    )?;
    root.insert("readback_bytes".to_owned(), Value::from(hidden_row_bytes));
    let mut effects = Vec::new();
    for (ordinal, kind) in [
        "queue_write",
        "queue_write",
        "queue_write",
        "queue_write",
        "queue_write",
        "queue_write",
        "queue_write",
        "queue_write",
        "queue_write",
        "queue_write",
        "dispatch_rms_norm",
        "dispatch_query",
        "dispatch_key",
        "dispatch_value",
        "dispatch_mrope",
        "dispatch_append",
        "dispatch_gqa",
        "dispatch_output",
        "dispatch_residual",
        "copy_residual",
        "submit",
        "map_residual",
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

fn finish_diagnostics_json(session: &BrowserDecoderLayerSession) -> Result<String, JsValue> {
    let readback_bytes = session
        .kv_plan
        .cache_bytes
        .checked_mul(2)
        .ok_or_else(|| js_layer_error(&["decoder layer finish readback overflowed"]))?;
    let mut root = Map::new();
    root.insert("schema_version".to_owned(), Value::from(1u64));
    root.insert("checked_error_scopes".to_owned(), checked_scopes_json());
    root.insert("captured_errors".to_owned(), Value::Array(Vec::new()));
    root.insert("buffer_allocation_count".to_owned(), Value::from(2u64));
    root.insert("queue_write_count".to_owned(), Value::from(0u64));
    root.insert("command_encoder_count".to_owned(), Value::from(1u64));
    root.insert("compute_pass_count".to_owned(), Value::from(0u64));
    root.insert("dispatch_count".to_owned(), Value::from(0u64));
    root.insert("copy_count".to_owned(), Value::from(2u64));
    root.insert("command_buffer_count".to_owned(), Value::from(1u64));
    root.insert("submission_count".to_owned(), Value::from(1u64));
    root.insert("map_count".to_owned(), Value::from(2u64));
    root.insert("readback_bytes".to_owned(), Value::from(readback_bytes));
    json_text(Value::Object(root))
}

fn layer_step_result(residual: Vec<u8>, diagnostics: String) -> Result<JsValue, JsValue> {
    let result = js_sys::Object::new();
    let residual_bytes = js_sys::Uint8Array::from(residual.as_slice());
    js_object_set(&result, "residual_bytes", &residual_bytes)?;
    js_object_set(
        &result,
        "diagnostics_json",
        &JsValue::from_str(&diagnostics),
    )?;
    Ok(result.into())
}

fn layer_finish_result(
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
    owner: &AsyncSessionOwner<BrowserDecoderLayerSession>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    kv_plan: pvlc_runtime_core::DecoderKvSessionPlan,
    attention_plan: pvlc_runtime_core::DecoderAttentionBlockPlan,
    operands: LayerBeginOperands,
    sources: LayerShaderSources,
) -> Result<String, JsValue> {
    validate_layer_capabilities(device, &kv_plan, &attention_plan)?;
    let mut ledger: Vec<ScopeKind> = Vec::new();
    for scope in CHECKED_SCOPES {
        if let Err(error) = push_layer_error_scope(device, scope).await {
            ledger.push(scope);
            let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
            return Err(drain_appended_message(error, captures, failures));
        }
        ledger.push(scope);
    }
    let raw_device = raw_layer_device(device)?;
    let raw_queue = raw_layer_queue(device, queue)?;
    let digests = layer_shader_digests(&sources);
    let session =
        match BrowserDecoderLayerSession::create(raw_device, kv_plan, attention_plan, &sources) {
            Ok(session) => session,
            Err(error) => {
                let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
                return Err(drain_appended_message(error, captures, failures));
            }
        };
    if let Err(error) = session.upload_initial_operands(&raw_queue, &operands) {
        let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
        return Err(drain_appended_message(error, captures, failures));
    }
    let generation = match owner.begin(session) {
        Ok(generation) => generation,
        Err(_busy) => {
            let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
            return Err(drain_appended_message(
                js_layer_error(&["decoder layer session is already active"]),
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
            wait_layer_owner_idle(owner).await;
            stale = false;
        }
        match pop_layer_error_scope(device, scope).await {
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
        return Err(js_layer_error(&[
            "decoder layer session begin is stale: its generation was cancelled",
        ]));
    }
    if !failures.is_empty() || !captures.is_empty() {
        poison_stored_session(owner);
        return Err(captured_failure_message(captures, failures));
    }
    mark_session_ready(owner);
    creation_diagnostics_json(&kv_plan, &attention_plan, &digests)
}

async fn run_step(
    owner: &AsyncSessionOwner<BrowserDecoderLayerSession>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lease: crate::AsyncSessionLease,
    mut session: BrowserDecoderLayerSession,
    operands: LayerStepOperands,
) -> Result<JsValue, JsValue> {
    let LayerStepOperands {
        transition,
        step_plan,
        hidden_bytes,
    } = operands;
    session.poisoned = true;
    let mut ledger: Vec<ScopeKind> = Vec::new();
    for scope in CHECKED_SCOPES {
        if let Err(error) = push_layer_error_scope(device, scope).await {
            ledger.push(scope);
            let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
            restore_layer_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
        ledger.push(scope);
    }
    let raw_device = match raw_layer_device(device) {
        Ok(raw) => raw,
        Err(error) => {
            restore_layer_session(owner, lease, session);
            return Err(error);
        }
    };
    let raw_queue = match raw_layer_queue(device, queue) {
        Ok(raw) => raw,
        Err(error) => {
            restore_layer_session(owner, lease, session);
            return Err(error);
        }
    };
    let hidden_row_bytes = match checked_u64_bytes(
        session.attention_plan.hidden_size as usize,
        "decoder hidden row",
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            restore_layer_session(owner, lease, session);
            return Err(error);
        }
    };
    if let Err(error) = session.encode_step(
        raw_device,
        &raw_queue,
        &transition,
        &step_plan,
        &hidden_bytes,
    ) {
        let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
        restore_layer_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if let Err(error) = await_layer_queue_completion(&raw_queue).await {
        let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
        restore_layer_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if let Err(error) =
        map_layer_buffer(session.layer_readback_buffer.as_ref(), hidden_row_bytes).await
    {
        let _ = unmap_layer_buffer(session.layer_readback_buffer.as_ref());
        let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
        restore_layer_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if owner.generation() != Some(lease.generation()) {
        let _ = unmap_layer_buffer(session.layer_readback_buffer.as_ref());
        wait_layer_owner_idle(owner).await;
        let _ = drain_layer_error_scopes(device, &mut ledger).await;
        let _ = owner.complete(lease, session, CompletionAction::Finish);
        return Err(js_layer_error(&[
            "decoder layer session step is stale: its generation was cancelled",
        ]));
    }
    let residual = match read_layer_mapped(session.layer_readback_buffer.as_ref(), hidden_row_bytes)
    {
        Ok(residual) => residual,
        Err(error) => {
            let _ = unmap_layer_buffer(session.layer_readback_buffer.as_ref());
            let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
            restore_layer_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    if let Err(error) = unmap_layer_buffer(session.layer_readback_buffer.as_ref()) {
        let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
        restore_layer_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    session.cache_tokens = transition.cache_tokens_after;
    session.poisoned = false;
    let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
    if !failures.is_empty() || !captures.is_empty() {
        session.poisoned = true;
        restore_layer_session(owner, lease, session);
        return Err(captured_failure_message(captures, failures));
    }
    let diagnostics = match step_diagnostics_json(&session, &transition) {
        Ok(diagnostics) => diagnostics,
        Err(error) => {
            restore_layer_session(owner, lease, session);
            return Err(error);
        }
    };
    let _ = owner.complete(lease, session, CompletionAction::Restore);
    layer_step_result(residual, diagnostics)
}

async fn run_finish(
    owner: &AsyncSessionOwner<BrowserDecoderLayerSession>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lease: crate::AsyncSessionLease,
    mut session: BrowserDecoderLayerSession,
) -> Result<JsValue, JsValue> {
    session.poisoned = true;
    let mut ledger: Vec<ScopeKind> = Vec::new();
    for scope in CHECKED_SCOPES {
        if let Err(error) = push_layer_error_scope(device, scope).await {
            ledger.push(scope);
            let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
            restore_layer_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
        ledger.push(scope);
    }
    let raw_device = match raw_layer_device(device) {
        Ok(raw) => raw,
        Err(error) => {
            restore_layer_session(owner, lease, session);
            return Err(error);
        }
    };
    let raw_queue = match raw_layer_queue(device, queue) {
        Ok(raw) => raw,
        Err(error) => {
            restore_layer_session(owner, lease, session);
            return Err(error);
        }
    };
    let (key_readback, value_readback) = match session.encode_finish(raw_device, &raw_queue) {
        Ok(readbacks) => readbacks,
        Err(error) => {
            let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
            restore_layer_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    let cache_bytes = session.kv_plan.cache_bytes;
    if let Err(error) = await_layer_queue_completion(&raw_queue).await {
        let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
        restore_layer_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if let Err(error) = map_layer_buffer(key_readback.as_ref(), cache_bytes).await {
        let _ = unmap_layer_buffer(key_readback.as_ref());
        let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
        restore_layer_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if owner.generation() != Some(lease.generation()) {
        let _ = unmap_layer_buffer(key_readback.as_ref());
        wait_layer_owner_idle(owner).await;
        let _ = drain_layer_error_scopes(device, &mut ledger).await;
        let _ = owner.complete(lease, session, CompletionAction::Finish);
        return Err(js_layer_error(&[
            "decoder layer session finish is stale: its generation was cancelled",
        ]));
    }
    let key_cache = match read_layer_mapped(key_readback.as_ref(), cache_bytes) {
        Ok(mapped) => mapped,
        Err(error) => {
            let _ = unmap_layer_buffer(key_readback.as_ref());
            let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
            restore_layer_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    if let Err(error) = unmap_layer_buffer(key_readback.as_ref()) {
        let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
        restore_layer_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if let Err(error) = map_layer_buffer(value_readback.as_ref(), cache_bytes).await {
        let _ = unmap_layer_buffer(value_readback.as_ref());
        let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
        restore_layer_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    let value_cache = match read_layer_mapped(value_readback.as_ref(), cache_bytes) {
        Ok(mapped) => mapped,
        Err(error) => {
            let _ = unmap_layer_buffer(value_readback.as_ref());
            let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
            restore_layer_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    if let Err(error) = unmap_layer_buffer(value_readback.as_ref()) {
        let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
        restore_layer_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    let (captures, failures) = drain_layer_error_scopes(device, &mut ledger).await;
    if !failures.is_empty() || !captures.is_empty() {
        session.poisoned = true;
        restore_layer_session(owner, lease, session);
        return Err(captured_failure_message(captures, failures));
    }
    let diagnostics = match finish_diagnostics_json(&session) {
        Ok(diagnostics) => diagnostics,
        Err(error) => {
            restore_layer_session(owner, lease, session);
            return Err(error);
        }
    };
    let _ = owner.complete(lease, session, CompletionAction::Finish);
    layer_finish_result(key_cache, value_cache, diagnostics)
}

impl DecoderLayerSessionAuthority {
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
        let sources = canonical_layer_sources()?;
        let rms_norm_source = serde_json::to_string(&sources.rms_norm).map_err(|error| {
            js_layer_error(&[
                "cannot serialize decoder layer RMS-norm source: ",
                &error.to_string(),
            ])
        })?;
        let gemv_source = serde_json::to_string(&sources.gemv).map_err(|error| {
            js_layer_error(&[
                "cannot serialize decoder layer GEMV source: ",
                &error.to_string(),
            ])
        })?;
        let mrope_source = serde_json::to_string(&sources.mrope).map_err(|error| {
            js_layer_error(&[
                "cannot serialize decoder layer M-RoPE source: ",
                &error.to_string(),
            ])
        })?;
        let append_source = serde_json::to_string(&sources.append).map_err(|error| {
            js_layer_error(&[
                "cannot serialize decoder layer append source: ",
                &error.to_string(),
            ])
        })?;
        let attention_source = serde_json::to_string(&sources.attention).map_err(|error| {
            js_layer_error(&[
                "cannot serialize decoder layer attention source: ",
                &error.to_string(),
            ])
        })?;
        let residual_source = serde_json::to_string(&sources.residual).map_err(|error| {
            js_layer_error(&[
                "cannot serialize decoder layer residual source: ",
                &error.to_string(),
            ])
        })?;
        let mut json = String::from("{\"schema_version\":1,\"sources\":{\"rms_norm_f32\":");
        json.push_str(&rms_norm_source);
        json.push_str(",\"gemv_f32\":");
        json.push_str(&gemv_source);
        json.push_str(",\"decoder_mrope_f32\":");
        json.push_str(&mrope_source);
        json.push_str(",\"decoder_kv_append_f32\":");
        json.push_str(&append_source);
        json.push_str(",\"decoder_gqa_f32\":");
        json.push_str(&attention_source);
        json.push_str(",\"add_f32\":");
        json.push_str(&residual_source);
        json.push_str("},\"shader_blake3\":{\"rms_norm_f32\":\"");
        json.push_str(&blake3_hex(&source_blake3(&sources.rms_norm)));
        json.push_str("\",\"gemv_f32\":\"");
        json.push_str(&blake3_hex(&source_blake3(&sources.gemv)));
        json.push_str("\",\"decoder_mrope_f32\":\"");
        json.push_str(&blake3_hex(&source_blake3(&sources.mrope)));
        json.push_str("\",\"decoder_kv_append_f32\":\"");
        json.push_str(&blake3_hex(&source_blake3(&sources.append)));
        json.push_str("\",\"decoder_gqa_f32\":\"");
        json.push_str(&blake3_hex(&source_blake3(&sources.attention)));
        json.push_str("\",\"add_f32\":\"");
        json.push_str(&blake3_hex(&source_blake3(&sources.residual)));
        json.push_str("\"}}");
        Ok(json)
    }

    pub(super) fn begin(
        &self,
        descriptor_json: &str,
        pack: &js_sys::Uint8Array,
        key_cache: &js_sys::Uint8Array,
        value_cache: &js_sys::Uint8Array,
    ) -> js_sys::Promise {
        let prepared = (|| {
            check_layer_admission(&self.owner)?;
            let parsed = parse_layer_descriptor_json(descriptor_json)?;
            let pack_bytes = layer_pack_to_bytes(pack)?;
            let weight_pack =
                parse_layer_weight_pack(&pack_bytes, parsed.prefix_tokens, parsed.cache_capacity)?;
            let key_cache_bytes = layer_uint8_to_bytes(key_cache)?;
            let value_cache_bytes = layer_uint8_to_bytes(value_cache)?;
            let key_f32 = layer_bytes_to_f32(&key_cache_bytes, "initial K cache")?;
            let value_f32 = layer_bytes_to_f32(&value_cache_bytes, "initial V cache")?;
            let norm1_f32 =
                layer_bytes_to_f32(&weight_pack.norm1_weight, "weights.input_layernorm")?;
            let q_f32 = layer_bytes_to_f32(&weight_pack.q_weight, "weights.q_proj")?;
            let k_f32 = layer_bytes_to_f32(&weight_pack.k_weight, "weights.k_proj")?;
            let v_f32 = layer_bytes_to_f32(&weight_pack.v_weight, "weights.v_proj")?;
            let o_f32 = layer_bytes_to_f32(&weight_pack.o_weight, "weights.o_proj")?;
            let cos_f32 = layer_bytes_to_f32(&weight_pack.rope_cos, "weights.mrope_cos")?;
            let sin_f32 = layer_bytes_to_f32(&weight_pack.rope_sin, "weights.mrope_sin")?;
            let kv_descriptor = DecoderKvSessionDescriptor {
                query_heads: parsed.query_heads,
                key_value_heads: parsed.key_value_heads,
                head_dim: parsed.head_dim,
                prefix_tokens: parsed.prefix_tokens,
                cache_capacity: parsed.cache_capacity,
                key_cache: key_f32.as_slice(),
                value_cache: value_f32.as_slice(),
            };
            let kv_plan = kv_descriptor.plan().map_err(|error| {
                js_layer_error(&[
                    "invalid decoder layer session descriptor geometry or initial cache operands: ",
                    &error.to_string(),
                ])
            })?;
            let attention_descriptor = DecoderAttentionBlockDescriptor {
                hidden_size: parsed.hidden_size,
                query_heads: parsed.query_heads,
                key_value_heads: parsed.key_value_heads,
                head_dim: parsed.head_dim,
                rms_norm_epsilon: parsed.rms_norm_epsilon,
                norm1_weight: norm1_f32.as_slice(),
                q_weight: q_f32.as_slice(),
                k_weight: k_f32.as_slice(),
                v_weight: v_f32.as_slice(),
                o_weight: o_f32.as_slice(),
                mrope_cos: cos_f32.as_slice(),
                mrope_sin: sin_f32.as_slice(),
                cache_capacity: parsed.cache_capacity,
            };
            let attention_plan = attention_descriptor.plan().map_err(|error| {
                js_layer_error(&[
                    "invalid decoder layer session descriptor geometry or weight operands: ",
                    &error.to_string(),
                ])
            })?;
            let sources = canonical_layer_sources()?;
            let operands = LayerBeginOperands {
                key_cache_bytes,
                value_cache_bytes,
                norm1_weight_bytes: weight_pack.norm1_weight,
                q_weight_bytes: weight_pack.q_weight,
                k_weight_bytes: weight_pack.k_weight,
                v_weight_bytes: weight_pack.v_weight,
                o_weight_bytes: weight_pack.o_weight,
                rope_cos_bytes: weight_pack.rope_cos,
                rope_sin_bytes: weight_pack.rope_sin,
            };
            Ok((kv_plan, attention_plan, operands, sources))
        })();
        let (kv_plan, attention_plan, operands, sources) = match prepared {
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
                kv_plan,
                attention_plan,
                operands,
                sources,
            )
            .await
            .map(|diagnostics| JsValue::from_str(&diagnostics))
        })
    }

    pub(super) fn begin_with_shader_override(
        &self,
        descriptor_json: &str,
        pack: &js_sys::Uint8Array,
        key_cache: &js_sys::Uint8Array,
        value_cache: &js_sys::Uint8Array,
        kernel: &str,
        source: &str,
    ) -> js_sys::Promise {
        if !KERNEL_NAMES.contains(&kernel) {
            return js_sys::Promise::reject(&js_layer_error(&[
                "unknown decoder layer shader override kernel: ",
                kernel,
            ]));
        }
        if source.is_empty() {
            return js_sys::Promise::reject(&js_layer_error(&[
                "decoder layer shader override source must not be empty",
            ]));
        }
        let prepared = (|| {
            check_layer_admission(&self.owner)?;
            let parsed = parse_layer_descriptor_json(descriptor_json)?;
            let pack_bytes = layer_pack_to_bytes(pack)?;
            let weight_pack =
                parse_layer_weight_pack(&pack_bytes, parsed.prefix_tokens, parsed.cache_capacity)?;
            let key_cache_bytes = layer_uint8_to_bytes(key_cache)?;
            let value_cache_bytes = layer_uint8_to_bytes(value_cache)?;
            let key_f32 = layer_bytes_to_f32(&key_cache_bytes, "initial K cache")?;
            let value_f32 = layer_bytes_to_f32(&value_cache_bytes, "initial V cache")?;
            let norm1_f32 =
                layer_bytes_to_f32(&weight_pack.norm1_weight, "weights.input_layernorm")?;
            let q_f32 = layer_bytes_to_f32(&weight_pack.q_weight, "weights.q_proj")?;
            let k_f32 = layer_bytes_to_f32(&weight_pack.k_weight, "weights.k_proj")?;
            let v_f32 = layer_bytes_to_f32(&weight_pack.v_weight, "weights.v_proj")?;
            let o_f32 = layer_bytes_to_f32(&weight_pack.o_weight, "weights.o_proj")?;
            let cos_f32 = layer_bytes_to_f32(&weight_pack.rope_cos, "weights.mrope_cos")?;
            let sin_f32 = layer_bytes_to_f32(&weight_pack.rope_sin, "weights.mrope_sin")?;
            let kv_descriptor = DecoderKvSessionDescriptor {
                query_heads: parsed.query_heads,
                key_value_heads: parsed.key_value_heads,
                head_dim: parsed.head_dim,
                prefix_tokens: parsed.prefix_tokens,
                cache_capacity: parsed.cache_capacity,
                key_cache: key_f32.as_slice(),
                value_cache: value_f32.as_slice(),
            };
            let kv_plan = kv_descriptor.plan().map_err(|error| {
                js_layer_error(&[
                    "invalid decoder layer session descriptor geometry or initial cache operands: ",
                    &error.to_string(),
                ])
            })?;
            let attention_descriptor = DecoderAttentionBlockDescriptor {
                hidden_size: parsed.hidden_size,
                query_heads: parsed.query_heads,
                key_value_heads: parsed.key_value_heads,
                head_dim: parsed.head_dim,
                rms_norm_epsilon: parsed.rms_norm_epsilon,
                norm1_weight: norm1_f32.as_slice(),
                q_weight: q_f32.as_slice(),
                k_weight: k_f32.as_slice(),
                v_weight: v_f32.as_slice(),
                o_weight: o_f32.as_slice(),
                mrope_cos: cos_f32.as_slice(),
                mrope_sin: sin_f32.as_slice(),
                cache_capacity: parsed.cache_capacity,
            };
            let attention_plan = attention_descriptor.plan().map_err(|error| {
                js_layer_error(&[
                    "invalid decoder layer session descriptor geometry or weight operands: ",
                    &error.to_string(),
                ])
            })?;
            let mut sources = canonical_layer_sources()?;
            if kernel == RMS_NORM_KERNEL_NAME {
                sources.rms_norm = source.to_owned();
            } else if kernel == GEMV_KERNEL_NAME {
                sources.gemv = source.to_owned();
            } else if kernel == MROPE_KERNEL_NAME {
                sources.mrope = source.to_owned();
            } else if kernel == APPEND_KERNEL_NAME {
                sources.append = source.to_owned();
            } else if kernel == ATTENTION_KERNEL_NAME {
                sources.attention = source.to_owned();
            } else {
                sources.residual = source.to_owned();
            }
            let operands = LayerBeginOperands {
                key_cache_bytes,
                value_cache_bytes,
                norm1_weight_bytes: weight_pack.norm1_weight,
                q_weight_bytes: weight_pack.q_weight,
                k_weight_bytes: weight_pack.k_weight,
                v_weight_bytes: weight_pack.v_weight,
                o_weight_bytes: weight_pack.o_weight,
                rope_cos_bytes: weight_pack.rope_cos,
                rope_sin_bytes: weight_pack.rope_sin,
            };
            Ok((kv_plan, attention_plan, operands, sources))
        })();
        let (kv_plan, attention_plan, operands, sources) = match prepared {
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
                kv_plan,
                attention_plan,
                operands,
                sources,
            )
            .await
            .map(|diagnostics| JsValue::from_str(&diagnostics))
        })
    }

    pub(super) fn step(&self, hidden_token: &js_sys::Uint8Array) -> js_sys::Promise {
        let prepared = (|| {
            let (lease, session) = acquire_layer_session(&self.owner)?;
            let hidden_bytes = match layer_uint8_to_bytes(hidden_token) {
                Ok(bytes) => bytes,
                Err(error) => {
                    restore_layer_session(&self.owner, lease, session);
                    return Err(error);
                }
            };
            let hidden_f32 = match layer_bytes_to_f32(&hidden_bytes, "step hidden row") {
                Ok(value) => value,
                Err(error) => {
                    restore_layer_session(&self.owner, lease, session);
                    return Err(error);
                }
            };
            let step_input = DecoderAttentionBlockStep {
                hidden_row: hidden_f32.as_slice(),
            };
            let transition = match session.kv_plan.plan_cache_transition(session.cache_tokens) {
                Ok(transition) => transition,
                Err(error) => {
                    restore_layer_session(&self.owner, lease, session);
                    return Err(js_layer_error(&[
                        "invalid decoder layer step cache capacity: ",
                        &error.to_string(),
                    ]));
                }
            };
            let step_plan = match session
                .attention_plan
                .plan_step(session.cache_tokens, &step_input)
            {
                Ok(step_plan) => step_plan,
                Err(error) => {
                    restore_layer_session(&self.owner, lease, session);
                    return Err(js_layer_error(&[
                        "invalid decoder layer step hidden row: ",
                        &error.to_string(),
                    ]));
                }
            };
            Ok((
                lease,
                session,
                LayerStepOperands {
                    transition,
                    step_plan,
                    hidden_bytes,
                },
            ))
        })();
        let (lease, session, operands) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return js_sys::Promise::reject(&error),
        };
        let owner = self.owner.clone();
        let device = self.device.clone();
        let queue = self.queue.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            run_step(&owner, &device, &queue, lease, session, operands).await
        })
    }

    pub(super) fn finish(&self) -> js_sys::Promise {
        let prepared = acquire_layer_session(&self.owner);
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
