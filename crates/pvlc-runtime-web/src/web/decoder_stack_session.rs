//! Sealed persistent browser decoder stack session authority.
//!
//! The authority privately owns the exact `wgpu::Device`, the exact `wgpu::Queue`,
//! and one `crate::AsyncSessionOwner<BrowserDecoderStackSession>`. Every
//! operation validates its inputs, the PVLCPK01 stack weight pack, and the exact
//! core plans before the first GPU effect, pushes the three checked error scopes,
//! executes the exact phase topology with raw WebGPU calls that surface every
//! thrown error as a `Result`, and drains the scopes LIFO. Any post-effect
//! failure poisons the stored session terminally; a cancelled generation drains
//! its scopes only after the newer in-flight lease clears. No `unsafe`, no
//! macros, no host-side compute shadow.

use pvlc_runtime_core::{
    DecoderKvSessionDescriptor, DecoderLmHeadDescriptor, DecoderLmHeadGeometryDescriptor,
    DecoderStackDescriptor, DecoderStackGeometryDescriptor, DecoderStackPrefillDescriptor,
    DecoderStackStep, DecoderWeightResourceDescriptor, DecoderWeightStorage,
    LINEAR_PROJECTION_TILE,
};
use serde_json::{Map, Value};
use std::{cell::RefCell, rc::Rc};
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
const GEMV_TILED_KERNEL_NAME: &str = "gemv_tiled_f32";
const RMS_NORM_F16_KERNEL_NAME: &str = "rms_norm_f16_weights";
const GEMV_TILED_F16_KERNEL_NAME: &str = "gemv_tiled_f16_weights";
const PREFILL_PROJECTION_F16_KERNEL_NAME: &str = "linear_projection_f16_weights";
const MROPE_KERNEL_NAME: &str = "decoder_mrope_f32";
const APPEND_KERNEL_NAME: &str = "decoder_kv_append_f32";
const ATTENTION_KERNEL_NAME: &str = "decoder_gqa_f32";
const SWIGLU_KERNEL_NAME: &str = "decoder_swiglu_f32";
const RESIDUAL_KERNEL_NAME: &str = "add_f32";
const PROJECTION_KERNEL_NAME: &str = "vision_patch_projection_f32";
const PREFILL_MROPE_KERNEL_NAME: &str = "decoder_prefill_mrope_f32";
const KV_APPEND_RANGE_KERNEL_NAME: &str = "decoder_kv_append_range_f32";
const PREFILL_GQA_KERNEL_NAME: &str = "decoder_prefill_gqa_f32";
const SPLIT_PARTIAL_KERNEL_NAME: &str = "decoder_gqa_split_partial_f32";
const SPLIT_MERGE_KERNEL_NAME: &str = "decoder_gqa_split_merge_f32";
const KERNEL_NAMES: [&str; 14] = [
    RMS_NORM_KERNEL_NAME,
    GEMV_KERNEL_NAME,
    MROPE_KERNEL_NAME,
    APPEND_KERNEL_NAME,
    ATTENTION_KERNEL_NAME,
    SWIGLU_KERNEL_NAME,
    RESIDUAL_KERNEL_NAME,
    PROJECTION_KERNEL_NAME,
    PREFILL_MROPE_KERNEL_NAME,
    KV_APPEND_RANGE_KERNEL_NAME,
    PREFILL_GQA_KERNEL_NAME,
    SPLIT_PARTIAL_KERNEL_NAME,
    SPLIT_MERGE_KERNEL_NAME,
    GEMV_TILED_KERNEL_NAME,
];
const ENTRY_POINT: &str = "main";
const BUFFER_HIDDEN_PINGPONG: &str = "decoder-stack-session-hidden-pingpong";
const BUFFER_NORM1_WEIGHT: &str = "decoder-stack-session-norm1-weight";
const BUFFER_Q_WEIGHT: &str = "decoder-stack-session-q-weight";
const BUFFER_K_WEIGHT: &str = "decoder-stack-session-k-weight";
const BUFFER_V_WEIGHT: &str = "decoder-stack-session-v-weight";
const BUFFER_O_WEIGHT: &str = "decoder-stack-session-o-weight";
const BUFFER_ROPE_COS: &str = "decoder-stack-session-rope-cos";
const BUFFER_ROPE_SIN: &str = "decoder-stack-session-rope-sin";
const BUFFER_NORM2_WEIGHT: &str = "decoder-stack-session-norm2-weight";
const BUFFER_GATE_WEIGHT: &str = "decoder-stack-session-gate-weight";
const BUFFER_UP_WEIGHT: &str = "decoder-stack-session-up-weight";
const BUFFER_DOWN_WEIGHT: &str = "decoder-stack-session-down-weight";
const BUFFER_KEY_CACHE: &str = "decoder-stack-session-key-cache";
const BUFFER_VALUE_CACHE: &str = "decoder-stack-session-value-cache";
const BUFFER_NORM1: &str = "decoder-stack-session-norm1";
const BUFFER_Q_PROJECTION: &str = "decoder-stack-session-q-projection";
const BUFFER_K_PROJECTION: &str = "decoder-stack-session-k-projection";
const BUFFER_V_PROJECTION: &str = "decoder-stack-session-v-projection";
const BUFFER_MROPE_QUERY: &str = "decoder-stack-session-mrope-query";
const BUFFER_MROPE_KEY: &str = "decoder-stack-session-mrope-key";
const BUFFER_ATTENTION_OUTPUT: &str = "decoder-stack-session-attention-output";
const BUFFER_O_PROJECTION: &str = "decoder-stack-session-o-projection";
const BUFFER_ATTENTION_RESIDUAL: &str = "decoder-stack-session-attention-residual";
const BUFFER_NORM2: &str = "decoder-stack-session-norm2";
const BUFFER_GATE: &str = "decoder-stack-session-gate";
const BUFFER_UP: &str = "decoder-stack-session-up";
const BUFFER_ACTIVATION: &str = "decoder-stack-session-activation";
const BUFFER_DOWN_PROJECTION: &str = "decoder-stack-session-down-projection";
const BUFFER_STACK_READBACK: &str = "decoder-stack-session-stack-readback";
const BUFFER_RMS_UNIFORM: &str = "decoder-stack-session-rms-uniform";
const BUFFER_GEMV_Q_UNIFORM: &str = "decoder-stack-session-gemv-q-uniform";
const BUFFER_GEMV_K_UNIFORM: &str = "decoder-stack-session-gemv-k-uniform";
const BUFFER_GEMV_V_UNIFORM: &str = "decoder-stack-session-gemv-v-uniform";
const BUFFER_MROPE_UNIFORM: &str = "decoder-stack-session-mrope-uniform";
const BUFFER_APPEND_UNIFORM: &str = "decoder-stack-session-append-uniform";
const BUFFER_ATTENTION_UNIFORM: &str = "decoder-stack-session-attention-uniform";
const BUFFER_RESIDUAL_UNIFORM: &str = "decoder-stack-session-residual-uniform";
const BUFFER_RMS2_UNIFORM: &str = "decoder-stack-session-rms2-uniform";
const BUFFER_GEMV_GATE_UNIFORM: &str = "decoder-stack-session-gemv-gate-uniform";
const BUFFER_GEMV_UP_UNIFORM: &str = "decoder-stack-session-gemv-up-uniform";
const BUFFER_SWIGLU_UNIFORM: &str = "decoder-stack-session-swiglu-uniform";
const BUFFER_GEMV_DOWN_UNIFORM: &str = "decoder-stack-session-gemv-down-uniform";
const BUFFER_RESIDUAL2_UNIFORM: &str = "decoder-stack-session-residual2-uniform";
const BUFFER_GEMV_O_UNIFORM: &str = "decoder-stack-session-gemv-o-uniform";
const BUFFER_FINISH_KEY_READBACK: &str = "decoder-stack-session-finish-key-readback";
const BUFFER_FINISH_VALUE_READBACK: &str = "decoder-stack-session-finish-value-readback";
const BUFFER_PREFILL_HIDDEN_STORAGE: &str = "decoder-stack-session-prefill-hidden-storage";
const BUFFER_PREFILL_NORM1: &str = "decoder-stack-session-prefill-norm1";
const BUFFER_PREFILL_QUERY: &str = "decoder-stack-session-prefill-query";
const BUFFER_PREFILL_KEY: &str = "decoder-stack-session-prefill-key";
const BUFFER_PREFILL_VALUE: &str = "decoder-stack-session-prefill-value";
const BUFFER_PREFILL_CONTEXT: &str = "decoder-stack-session-prefill-context";
const BUFFER_PREFILL_OUTPUT: &str = "decoder-stack-session-prefill-output";
const BUFFER_PREFILL_NORM2: &str = "decoder-stack-session-prefill-norm2";
const BUFFER_PREFILL_GATE: &str = "decoder-stack-session-prefill-gate";
const BUFFER_PREFILL_UP: &str = "decoder-stack-session-prefill-up";
const BUFFER_PREFILL_ACTIVATION: &str = "decoder-stack-session-prefill-activation";
const BUFFER_PREFILL_ZERO_BIAS: &str = "decoder-stack-session-prefill-zero-bias";
const BUFFER_FINAL_NORM_WEIGHT: &str = "decoder-stack-session-final-norm-weight";
const BUFFER_LM_HEAD: &str = "decoder-stack-session-lm-head";
const BUFFER_NORMED_ROW: &str = "decoder-stack-session-normed-row";
const BUFFER_LOGITS: &str = "decoder-stack-session-logits";
const BUFFER_LOGITS_READBACK: &str = "decoder-stack-session-logits-readback";
const BUFFER_LOGITS_RMS_UNIFORM: &str = "decoder-stack-session-logits-rms-uniform";
const BUFFER_LOGITS_GEMV_UNIFORM: &str = "decoder-stack-session-logits-gemv-uniform";
const BUFFER_TOP1_RESULT: &str = "decoder-stack-session-top1-result";
const BUFFER_TOP1_READBACK: &str = "decoder-stack-session-top1-readback";
const TOP1_KERNEL_NAME: &str = "decoder_logits_top1_f32";
const TOP1_RESULT_BYTES: u64 = 8;
const BUFFER_SPLIT_PARTIALS: &str = "decoder-stack-session-split-partials";
const BUFFER_SPLIT_PARTIAL_UNIFORM: &str = "decoder-stack-session-split-partial-uniform";
const BUFFER_SPLIT_MERGE_UNIFORM: &str = "decoder-stack-session-split-merge-uniform";
const PREFILL_HIDDEN_WIDTH: u64 = 1024;
const PREFILL_QUERY_WIDTH: u64 = 2048;
const PREFILL_KEY_VALUE_WIDTH: u64 = 256;
const PREFILL_INTERMEDIATE_WIDTH: u64 = 3072;
const PREFILL_ZERO_BIAS_BYTES: u64 = 12288;
const PINNED_STACK_VOCAB_SIZE: u32 = 103_424;
const DESCRIPTOR_FIELDS: [&str; 15] = [
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
const PINNED_STACK_QUERY_WIDTH: u32 = 2048;
const PINNED_STACK_KEY_VALUE_WIDTH: u32 = 256;
const PACK_MAGIC: [u8; 8] = *b"PVLCPK01";
const PACK_HEADER_BYTES: usize = 32;
const PACK_DIRECTORY_FIXED_BYTES: usize = 56;
const PACK_DIRECTORY_ENTRY_ALIGNMENT: usize = 8;
const PACK_MAX_ALIGNMENT: u64 = 4096;
const PACK_VERSION: u32 = 1;
const PACK_SECTION_COUNT: u32 = 14;
const LEGACY_PACK_SECTION_COUNT: u32 = 12;
const PACK_DESCRIPTOR_SECTION_ID: &str = "ir.decoder_stack_00";
const PACK_SHARD_IDS: [&str; 13] = [
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
const LEGACY_PACK_SHARD_COUNT: usize = 11;
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
const PACK_DESCRIPTOR_FIELDS: [&str; 17] = [
    "schema_version",
    "oracle",
    "case_id",
    "model_revision",
    "layers",
    "hidden_size",
    "intermediate_size",
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
const PACK_BALANCED_DESCRIPTOR_FIELDS: [&str; 20] = [
    "schema_version",
    "oracle",
    "case_id",
    "model_revision",
    "layers",
    "hidden_size",
    "intermediate_size",
    "query_heads",
    "key_value_heads",
    "head_dim",
    "query_width",
    "key_value_width",
    "prefix_tokens",
    "cache_capacity",
    "rms_norm_epsilon",
    "mrope_sections",
    "checkpoint_blake3",
    "checkpoint_bytes",
    "weight_storage",
    "shards",
];
const PACK_DESCRIPTOR_ORACLES: [&str; 2] = ["synthetic", "official_l3"];
const UNIFORM_BUFFER_BYTES: u64 = 16;
const READ_ONLY_STORAGE: &str = "read-only-storage";
const READ_WRITE_STORAGE: &str = "storage";
const UNIFORM_BINDING: &str = "uniform";
const RMS_LAYOUT_ENTRIES: [(u32, &str, bool); 4] = [
    (0, READ_ONLY_STORAGE, true),
    (1, READ_ONLY_STORAGE, true),
    (2, READ_WRITE_STORAGE, false),
    (3, UNIFORM_BINDING, false),
];
const GEMV_LAYOUT_ENTRIES: [(u32, &str, bool); 4] = [
    (0, READ_ONLY_STORAGE, true),
    (1, READ_ONLY_STORAGE, false),
    (2, READ_WRITE_STORAGE, false),
    (3, UNIFORM_BINDING, false),
];
const MROPE_LAYOUT_ENTRIES: [(u32, &str, bool); 7] = [
    (0, READ_ONLY_STORAGE, false),
    (1, READ_ONLY_STORAGE, false),
    (2, READ_ONLY_STORAGE, false),
    (3, READ_ONLY_STORAGE, false),
    (4, READ_WRITE_STORAGE, false),
    (5, READ_WRITE_STORAGE, false),
    (6, UNIFORM_BINDING, false),
];
const APPEND_LAYOUT_ENTRIES: [(u32, &str, bool); 5] = [
    (0, READ_ONLY_STORAGE, false),
    (1, READ_ONLY_STORAGE, false),
    (2, READ_WRITE_STORAGE, true),
    (3, READ_WRITE_STORAGE, true),
    (4, UNIFORM_BINDING, false),
];
const ATTENTION_LAYOUT_ENTRIES: [(u32, &str, bool); 5] = [
    (0, READ_ONLY_STORAGE, false),
    (1, READ_ONLY_STORAGE, true),
    (2, READ_ONLY_STORAGE, true),
    (3, READ_WRITE_STORAGE, false),
    (4, UNIFORM_BINDING, false),
];
const SWIGLU_LAYOUT_ENTRIES: [(u32, &str, bool); 4] = [
    (0, READ_ONLY_STORAGE, false),
    (1, READ_ONLY_STORAGE, false),
    (2, READ_WRITE_STORAGE, false),
    (3, UNIFORM_BINDING, false),
];
const RESIDUAL_LAYOUT_ENTRIES: [(u32, &str, bool); 4] = [
    (0, READ_ONLY_STORAGE, true),
    (1, READ_ONLY_STORAGE, false),
    (2, READ_WRITE_STORAGE, true),
    (3, UNIFORM_BINDING, false),
];
const PROJECTION_LAYOUT_ENTRIES: [(u32, &str, bool); 5] = [
    (0, READ_ONLY_STORAGE, false),
    (1, READ_ONLY_STORAGE, true),
    (2, READ_ONLY_STORAGE, false),
    (3, READ_WRITE_STORAGE, false),
    (4, UNIFORM_BINDING, false),
];
const TOP1_LAYOUT_ENTRIES: [(u32, &str, bool); 2] = [
    (0, READ_ONLY_STORAGE, false),
    (1, READ_WRITE_STORAGE, false),
];

const TOP1_SHADER_SOURCE: &str = r#"
const VOCAB_SIZE: u32 = 103424u;
const WORKGROUP_SIZE: u32 = 256u;

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;

var<workgroup> best_keys: array<i32, 256>;
var<workgroup> best_ids: array<u32, 256>;

fn total_order_key(value: f32) -> i32 {
    let bits = bitcast<i32>(value);
    return bits ^ i32(u32(bits >> 31) >> 1u);
}

fn is_better(candidate_key: i32, candidate_id: u32, current_key: i32, current_id: u32) -> bool {
    return candidate_id != 0xffffffffu &&
        (current_id == 0xffffffffu ||
         candidate_key > current_key ||
         (candidate_key == current_key && candidate_id < current_id));
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    var best_key = bitcast<i32>(0x80000000u);
    var best_id = 0xffffffffu;
    var token_id = lane;
    loop {
        if (token_id >= VOCAB_SIZE) {
            break;
        }
        let key = total_order_key(logits[token_id]);
        if (is_better(key, token_id, best_key, best_id)) {
            best_key = key;
            best_id = token_id;
        }
        token_id += WORKGROUP_SIZE;
    }
    best_keys[lane] = best_key;
    best_ids[lane] = best_id;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            let other_key = best_keys[lane + stride];
            let other_id = best_ids[lane + stride];
            if (is_better(other_key, other_id, best_keys[lane], best_ids[lane])) {
                best_keys[lane] = other_key;
                best_ids[lane] = other_id;
            }
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (lane == 0u) {
        result[0] = best_ids[0];
        result[1] = bitcast<u32>(logits[best_ids[0]]);
    }
}
"#;

pub(super) struct BrowserDecoderStackResidentWeights {
    key: String,
    checkpoint_blake3: [u8; 32],
    resident_bytes: u64,
    norm1: wgpu::webgpu::GpuBuffer,
    q: wgpu::webgpu::GpuBuffer,
    k: wgpu::webgpu::GpuBuffer,
    v: wgpu::webgpu::GpuBuffer,
    o: wgpu::webgpu::GpuBuffer,
    norm2: wgpu::webgpu::GpuBuffer,
    gate: wgpu::webgpu::GpuBuffer,
    up: wgpu::webgpu::GpuBuffer,
    down: wgpu::webgpu::GpuBuffer,
    final_norm: Option<wgpu::webgpu::GpuBuffer>,
    lm_head: Option<wgpu::webgpu::GpuBuffer>,
}

impl Clone for BrowserDecoderStackResidentWeights {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            checkpoint_blake3: self.checkpoint_blake3,
            resident_bytes: self.resident_bytes,
            norm1: self.norm1.clone(),
            q: self.q.clone(),
            k: self.k.clone(),
            v: self.v.clone(),
            o: self.o.clone(),
            norm2: self.norm2.clone(),
            gate: self.gate.clone(),
            up: self.up.clone(),
            down: self.down.clone(),
            final_norm: self.final_norm.clone(),
            lm_head: self.lm_head.clone(),
        }
    }
}

pub(super) struct SharedDecoderStackResidentWeightCache {
    slot: Rc<RefCell<Option<BrowserDecoderStackResidentWeights>>>,
}

impl Clone for SharedDecoderStackResidentWeightCache {
    fn clone(&self) -> Self {
        Self {
            slot: Rc::clone(&self.slot),
        }
    }
}

impl SharedDecoderStackResidentWeightCache {
    fn get(&self) -> Option<BrowserDecoderStackResidentWeights> {
        self.slot.borrow().clone()
    }

    fn replace(&self, value: BrowserDecoderStackResidentWeights) {
        self.slot.replace(Some(value));
    }
}

pub(super) fn shared_resident_weight_cache() -> SharedDecoderStackResidentWeightCache {
    SharedDecoderStackResidentWeightCache {
        slot: Rc::new(RefCell::new(None)),
    }
}

/// Sealed owner of the persistent decoder stack session lifecycle.
pub(super) struct DecoderStackSessionAuthority {
    device: wgpu::Device,
    queue: wgpu::Queue,
    owner: crate::AsyncSessionOwner<BrowserDecoderStackSession>,
    resident_weight_cache: SharedDecoderStackResidentWeightCache,
}

struct BrowserDecoderStackSession {
    kv_plan: pvlc_runtime_core::DecoderKvSessionPlan,
    stack_plan: pvlc_runtime_core::DecoderStackPlan,
    weight_resource_plan: pvlc_runtime_core::DecoderWeightResourcePlan,
    checkpoint_blake3: Option<[u8; 32]>,
    resident_weight_bytes: u64,
    cache_tokens: u32,
    poisoned: bool,
    ready: bool,
    rms_norm_shader_blake3: [u8; 32],
    gemv_shader_blake3: [u8; 32],
    gemv_tiled_shader_blake3: [u8; 32],
    mrope_shader_blake3: [u8; 32],
    append_shader_blake3: [u8; 32],
    attention_shader_blake3: [u8; 32],
    swiglu_shader_blake3: [u8; 32],
    residual_shader_blake3: [u8; 32],
    hidden_pingpong_buffer: wgpu::webgpu::GpuBuffer,
    norm1_weight_buffer: wgpu::webgpu::GpuBuffer,
    q_weight_buffer: wgpu::webgpu::GpuBuffer,
    k_weight_buffer: wgpu::webgpu::GpuBuffer,
    v_weight_buffer: wgpu::webgpu::GpuBuffer,
    o_weight_buffer: wgpu::webgpu::GpuBuffer,
    rope_cos_buffer: wgpu::webgpu::GpuBuffer,
    rope_sin_buffer: wgpu::webgpu::GpuBuffer,
    norm2_weight_buffer: wgpu::webgpu::GpuBuffer,
    gate_weight_buffer: wgpu::webgpu::GpuBuffer,
    up_weight_buffer: wgpu::webgpu::GpuBuffer,
    down_weight_buffer: wgpu::webgpu::GpuBuffer,
    key_cache_buffer: wgpu::webgpu::GpuBuffer,
    value_cache_buffer: wgpu::webgpu::GpuBuffer,
    norm1_buffer: wgpu::webgpu::GpuBuffer,
    q_projection_buffer: wgpu::webgpu::GpuBuffer,
    k_projection_buffer: wgpu::webgpu::GpuBuffer,
    v_projection_buffer: wgpu::webgpu::GpuBuffer,
    mrope_query_buffer: wgpu::webgpu::GpuBuffer,
    mrope_key_buffer: wgpu::webgpu::GpuBuffer,
    attention_output_buffer: wgpu::webgpu::GpuBuffer,
    o_projection_buffer: wgpu::webgpu::GpuBuffer,
    attention_residual_buffer: wgpu::webgpu::GpuBuffer,
    norm2_buffer: wgpu::webgpu::GpuBuffer,
    gate_buffer: wgpu::webgpu::GpuBuffer,
    up_buffer: wgpu::webgpu::GpuBuffer,
    activation_buffer: wgpu::webgpu::GpuBuffer,
    down_projection_buffer: wgpu::webgpu::GpuBuffer,
    stack_readback_buffer: wgpu::webgpu::GpuBuffer,
    rms_uniform_buffer: wgpu::webgpu::GpuBuffer,
    gemv_q_uniform_buffer: wgpu::webgpu::GpuBuffer,
    gemv_k_uniform_buffer: wgpu::webgpu::GpuBuffer,
    gemv_v_uniform_buffer: wgpu::webgpu::GpuBuffer,
    mrope_uniform_buffer: wgpu::webgpu::GpuBuffer,
    append_uniform_buffer: wgpu::webgpu::GpuBuffer,
    attention_uniform_buffer: wgpu::webgpu::GpuBuffer,
    residual_uniform_buffer: wgpu::webgpu::GpuBuffer,
    rms2_uniform_buffer: wgpu::webgpu::GpuBuffer,
    gemv_gate_uniform_buffer: wgpu::webgpu::GpuBuffer,
    gemv_up_uniform_buffer: wgpu::webgpu::GpuBuffer,
    swiglu_uniform_buffer: wgpu::webgpu::GpuBuffer,
    gemv_down_uniform_buffer: wgpu::webgpu::GpuBuffer,
    residual2_uniform_buffer: wgpu::webgpu::GpuBuffer,
    gemv_o_uniform_buffer: wgpu::webgpu::GpuBuffer,
    rms_norm_pipeline: js_sys::Object,
    gemv_tiled_pipeline: js_sys::Object,
    rms_norm_f16_pipeline: Option<js_sys::Object>,
    gemv_tiled_f16_pipeline: Option<js_sys::Object>,
    mrope_pipeline: js_sys::Object,
    append_pipeline: js_sys::Object,
    swiglu_pipeline: js_sys::Object,
    residual_pipeline: js_sys::Object,
    rms_bind_group: js_sys::Object,
    gemv_q_bind_group: js_sys::Object,
    gemv_k_bind_group: js_sys::Object,
    gemv_v_bind_group: js_sys::Object,
    mrope_bind_group: js_sys::Object,
    append_bind_group: js_sys::Object,
    gemv_o_bind_group: js_sys::Object,
    residual_bind_group: js_sys::Object,
    rms2_bind_group: js_sys::Object,
    gemv_gate_bind_group: js_sys::Object,
    gemv_up_bind_group: js_sys::Object,
    swiglu_bind_group: js_sys::Object,
    gemv_down_bind_group: js_sys::Object,
    residual2_bind_group: js_sys::Object,
    prefill_plan: pvlc_runtime_core::DecoderStackPrefillPlan,
    prefill_projection_shader_blake3: Option<[u8; 32]>,
    prefill_mrope_shader_blake3: Option<[u8; 32]>,
    kv_append_range_shader_blake3: Option<[u8; 32]>,
    prefill_gqa_shader_blake3: Option<[u8; 32]>,
    prefill_hidden_storage_buffer: Option<wgpu::webgpu::GpuBuffer>,
    prefill_norm1_buffer: Option<wgpu::webgpu::GpuBuffer>,
    prefill_query_buffer: Option<wgpu::webgpu::GpuBuffer>,
    prefill_key_buffer: Option<wgpu::webgpu::GpuBuffer>,
    prefill_value_buffer: Option<wgpu::webgpu::GpuBuffer>,
    prefill_context_buffer: Option<wgpu::webgpu::GpuBuffer>,
    prefill_output_buffer: Option<wgpu::webgpu::GpuBuffer>,
    prefill_norm2_buffer: Option<wgpu::webgpu::GpuBuffer>,
    prefill_gate_buffer: Option<wgpu::webgpu::GpuBuffer>,
    prefill_up_buffer: Option<wgpu::webgpu::GpuBuffer>,
    prefill_activation_buffer: Option<wgpu::webgpu::GpuBuffer>,
    prefill_zero_bias_buffer: Option<wgpu::webgpu::GpuBuffer>,
    prefill_projection_pipeline: Option<js_sys::Object>,
    prefill_projection_f16_pipeline: Option<js_sys::Object>,
    prefill_mrope_pipeline: Option<js_sys::Object>,
    kv_append_range_pipeline: Option<js_sys::Object>,
    prefill_gqa_pipeline: Option<js_sys::Object>,
    prefill_rms1_bind_group: Option<js_sys::Object>,
    prefill_query_bind_group: Option<js_sys::Object>,
    prefill_key_bind_group: Option<js_sys::Object>,
    prefill_value_bind_group: Option<js_sys::Object>,
    prefill_mrope_bind_group: Option<js_sys::Object>,
    prefill_kv_append_range_bind_group: Option<js_sys::Object>,
    prefill_gqa_bind_group: Option<js_sys::Object>,
    prefill_output_bind_group: Option<js_sys::Object>,
    prefill_residual_bind_group: Option<js_sys::Object>,
    prefill_rms2_bind_group: Option<js_sys::Object>,
    prefill_gate_bind_group: Option<js_sys::Object>,
    prefill_up_bind_group: Option<js_sys::Object>,
    prefill_swiglu_bind_group: Option<js_sys::Object>,
    prefill_down_bind_group: Option<js_sys::Object>,
    prefill_residual2_bind_group: Option<js_sys::Object>,
    lm_head_plan: Option<pvlc_runtime_core::DecoderLmHeadPlan>,
    final_norm_weight_buffer: Option<wgpu::webgpu::GpuBuffer>,
    lm_head_weight_buffer: Option<wgpu::webgpu::GpuBuffer>,
    normed_row_buffer: Option<wgpu::webgpu::GpuBuffer>,
    logits_buffer: Option<wgpu::webgpu::GpuBuffer>,
    logits_readback_buffer: Option<wgpu::webgpu::GpuBuffer>,
    top1_result_buffer: Option<wgpu::webgpu::GpuBuffer>,
    top1_readback_buffer: Option<wgpu::webgpu::GpuBuffer>,
    top1_pipeline: Option<js_sys::Object>,
    top1_bind_group: Option<js_sys::Object>,
    top1_shader_blake3: Option<[u8; 32]>,
    logits_rms_uniform_buffer: Option<wgpu::webgpu::GpuBuffer>,
    logits_gemv_uniform_buffer: Option<wgpu::webgpu::GpuBuffer>,
    prefill_logits_rms_bind_group: Option<js_sys::Object>,
    step_logits_rms_bind_group: Option<js_sys::Object>,
    gemv_logits_bind_group: Option<js_sys::Object>,
    split_partials_buffer: wgpu::webgpu::GpuBuffer,
    split_partial_uniform_buffer: wgpu::webgpu::GpuBuffer,
    split_merge_uniform_buffer: wgpu::webgpu::GpuBuffer,
    split_partial_pipeline: js_sys::Object,
    split_merge_pipeline: js_sys::Object,
    split_partial_bind_group: js_sys::Object,
    split_merge_bind_group: js_sys::Object,
    split_partial_shader_blake3: [u8; 32],
    split_merge_shader_blake3: [u8; 32],
}

struct StackShaderSources {
    rms_norm: String,
    gemv: String,
    gemv_tiled: String,
    rms_norm_f16_weights: String,
    gemv_tiled_f16_weights: String,
    linear_projection_f16_weights: String,
    mrope: String,
    append: String,
    attention: String,
    swiglu: String,
    residual: String,
    projection: String,
    prefill_mrope: String,
    kv_append_range: String,
    prefill_gqa: String,
    split_partial: String,
    split_merge: String,
}

struct StackShaderDigests {
    rms_norm: [u8; 32],
    gemv: [u8; 32],
    gemv_tiled: [u8; 32],
    mrope: [u8; 32],
    append: [u8; 32],
    attention: [u8; 32],
    swiglu: [u8; 32],
    residual: [u8; 32],
    split_partial: [u8; 32],
    split_merge: [u8; 32],
}

struct StackBeginOperands {
    upload_initial_cache: bool,
    key_cache_bytes: Vec<u8>,
    value_cache_bytes: Vec<u8>,
    norm1_weight_bytes: Vec<u8>,
    q_weight_bytes: Vec<u8>,
    k_weight_bytes: Vec<u8>,
    v_weight_bytes: Vec<u8>,
    o_weight_bytes: Vec<u8>,
    rope_cos_bytes: Vec<u8>,
    rope_sin_bytes: Vec<u8>,
    norm2_weight_bytes: Vec<u8>,
    gate_weight_bytes: Vec<u8>,
    up_weight_bytes: Vec<u8>,
    down_weight_bytes: Vec<u8>,
    final_norm_weight_bytes: Option<Vec<u8>>,
    lm_head_weight_bytes: Option<Vec<u8>>,
}

struct PreparedStackBegin {
    kv_plan: pvlc_runtime_core::DecoderKvSessionPlan,
    stack_plan: pvlc_runtime_core::DecoderStackPlan,
    weight_resource_plan: pvlc_runtime_core::DecoderWeightResourcePlan,
    checkpoint_blake3: Option<[u8; 32]>,
    prefill_plan: pvlc_runtime_core::DecoderStackPrefillPlan,
    lm_head_plan: Option<pvlc_runtime_core::DecoderLmHeadPlan>,
    prefill_capable: bool,
    operands: StackBeginOperands,
    sources: StackShaderSources,
}

struct StackStepOperands {
    transition: pvlc_runtime_core::DecoderKvSessionStepPlan,
    step_plan: pvlc_runtime_core::DecoderLayerStepPlan,
    hidden_bytes: Vec<u8>,
}

struct StackPrefillOperands {
    prefill_plan: pvlc_runtime_core::DecoderStackPrefillPlan,
    hidden_bytes: Vec<u8>,
}

struct StackBindGroupLayouts {
    rms: js_sys::Object,
    gemv: js_sys::Object,
    mrope: js_sys::Object,
    append: js_sys::Object,
    attention: js_sys::Object,
    swiglu: js_sys::Object,
    residual: js_sys::Object,
}

struct DecoderWeightPipelines<'a> {
    rms_norm: &'a js_sys::Object,
    gemv_tiled: &'a js_sys::Object,
    prefill_projection: Option<&'a js_sys::Object>,
}

struct ParsedDescriptor {
    hidden_size: u32,
    intermediate_size: u32,
    query_heads: u32,
    key_value_heads: u32,
    head_dim: u32,
    prefix_tokens: u32,
    cache_capacity: u32,
    rms_norm_epsilon: f32,
    prefill_tokens: u32,
    vocab_size: Option<u32>,
}

struct ParsedWeightPack {
    weight_storage: DecoderWeightStorage,
    checkpoint_blake3: Option<[u8; 32]>,
    norm1_weight: Vec<u8>,
    q_weight: Vec<u8>,
    k_weight: Vec<u8>,
    v_weight: Vec<u8>,
    o_weight: Vec<u8>,
    rope_cos: Vec<u8>,
    rope_sin: Vec<u8>,
    norm2_weight: Vec<u8>,
    gate_weight: Vec<u8>,
    up_weight: Vec<u8>,
    down_weight: Vec<u8>,
    final_norm_weight: Option<Vec<u8>>,
    lm_head_weight: Option<Vec<u8>>,
}

struct ValidatedPackDescriptor {
    shards: Map<String, Value>,
    checkpoint_blake3: Option<[u8; 32]>,
}

struct M7q1CheckpointPartition {
    f16_checkpoint_shards: u32,
    f32_rope_table_shards: u32,
}

struct M7q1RmsProbe {
    rows: u32,
    width: u32,
    epsilon: f32,
    input: Vec<f32>,
    weight_f16: Vec<u16>,
}

struct M7q1GemvProbe {
    rows: u32,
    columns: u32,
    matrix_f16: Vec<u16>,
    vector: Vec<f32>,
}

struct M7q1LinearProbe {
    tokens: u32,
    input_width: u32,
    output_width: u32,
    input: Vec<f32>,
    weight_f16: Vec<u16>,
    bias: Vec<f32>,
}

struct M7q1VisionProbe {
    patches: u32,
    input_width: u32,
    output_width: u32,
    input: Vec<f32>,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

struct M7q1WeightProbe {
    precision_profile: String,
    checkpoint_blake3: String,
    checkpoint_partition: M7q1CheckpointPartition,
    rms: M7q1RmsProbe,
    gemv: M7q1GemvProbe,
    linear: M7q1LinearProbe,
    linear_input_major: M7q1LinearProbe,
    vision: M7q1VisionProbe,
}

struct PackDirectoryEntry {
    offset: u64,
    byte_length: u64,
    alignment: u64,
    digest: [u8; 32],
}

fn js_stack_error(parts: &[&str]) -> JsValue {
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

fn raw_stack_device(device: &wgpu::Device) -> Result<&JsValue, JsValue> {
    device.as_webgpu().map(AsRef::as_ref).ok_or_else(|| {
        js_stack_error(&["decoder stack session device has no browser WebGPU handle"])
    })
}

fn raw_stack_queue(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<JsValue, JsValue> {
    let raw_device = raw_stack_device(device)?;
    let registered =
        js_sys::Reflect::get(raw_device, &JsValue::from_str("queue")).map_err(|error| {
            js_stack_error(&["cannot access GPUDevice.queue: ", &js_error_text(&error)])
        })?;
    let raw_queue: &JsValue = queue.as_webgpu().map(AsRef::as_ref).ok_or_else(|| {
        js_stack_error(&["decoder stack session queue has no browser WebGPU handle"])
    })?;
    if !js_sys::Object::is(&registered, raw_queue) {
        return Err(js_stack_error(&[
            "decoder stack session queue handle is not the exact device queue",
        ]));
    }
    Ok(registered)
}

fn raw_stack_method(handle: &JsValue, name: &str) -> Result<js_sys::Function, JsValue> {
    js_sys::Reflect::get(handle, &JsValue::from_str(name))
        .map_err(|error| {
            js_stack_error(&[
                "cannot access WebGPU method ",
                name,
                ": ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<js_sys::Function>()
        .map_err(|_| js_stack_error(&["WebGPU member ", name, " is not callable"]))
}

async fn push_stack_error_scope(device: &wgpu::Device, scope: ScopeKind) -> Result<(), JsValue> {
    let raw = raw_stack_device(device)?;
    let push = raw_stack_method(raw, "pushErrorScope")?;
    push.call1(raw, &JsValue::from_str(scope.filter_str()))
        .map(|_| ())
        .map_err(|error| {
            js_stack_error(&[
                "cannot push ",
                scope.as_str(),
                " WebGPU error scope: ",
                &js_error_text(&error),
            ])
        })
}

async fn pop_stack_error_scope(
    device: &wgpu::Device,
    scope: ScopeKind,
) -> Result<Option<String>, JsValue> {
    let raw = raw_stack_device(device)?;
    let pop = raw_stack_method(raw, "popErrorScope")?;
    let invocation = pop
        .call0(raw)
        .map_err(|error| {
            js_stack_error(&[
                "cannot invoke popErrorScope for ",
                scope.as_str(),
                " scope: ",
                &js_error_text(&error),
            ])
        })
        .and_then(|pending| {
            pending.dyn_into::<js_sys::Promise>().map_err(|_| {
                js_stack_error(&[
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
                js_stack_error(&[
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

async fn drain_stack_error_scopes(
    device: &wgpu::Device,
    ledger: &mut Vec<ScopeKind>,
) -> (Vec<String>, Vec<JsValue>) {
    let mut captures = Vec::new();
    let mut failures = Vec::new();
    while let Some(scope) = ledger.pop() {
        match pop_stack_error_scope(device, scope).await {
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
    let mut message = String::from("decoder stack session captured WebGPU errors:");
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

async fn yield_stack_event_loop() {
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

async fn wait_stack_owner_idle(owner: &AsyncSessionOwner<BrowserDecoderStackSession>) {
    while owner.is_in_flight() {
        yield_stack_event_loop().await;
    }
}

fn poison_stored_session(owner: &AsyncSessionOwner<BrowserDecoderStackSession>) {
    if let Some(mut session) = owner.stored_mut() {
        session.poisoned = true;
    }
}

fn mark_session_ready(owner: &AsyncSessionOwner<BrowserDecoderStackSession>) {
    if let Some(mut session) = owner.stored_mut() {
        session.ready = true;
    }
}

fn check_stack_admission(
    owner: &AsyncSessionOwner<BrowserDecoderStackSession>,
) -> Result<(), JsValue> {
    if owner.stored().is_some_and(|session| session.poisoned) {
        return Err(js_stack_error(&[
            "decoder stack session is terminally poisoned",
        ]));
    }
    if owner.is_busy() {
        return Err(js_stack_error(&[
            "decoder stack session is already active or busy with another operation",
        ]));
    }
    Ok(())
}

fn acquire_stack_session(
    owner: &AsyncSessionOwner<BrowserDecoderStackSession>,
) -> Result<(crate::AsyncSessionLease, BrowserDecoderStackSession), JsValue> {
    {
        let Some(session) = owner.stored() else {
            return Err(js_stack_error(&["no ready decoder stack session"]));
        };
        if session.poisoned {
            return Err(js_stack_error(&[
                "decoder stack session is terminally poisoned",
            ]));
        }
        if !session.ready {
            return Err(js_stack_error(&["no ready decoder stack session"]));
        }
    }
    owner
        .acquire()
        .map_err(|_| js_stack_error(&["no stored decoder stack session"]))
}

fn restore_stack_session(
    owner: &AsyncSessionOwner<BrowserDecoderStackSession>,
    lease: crate::AsyncSessionLease,
    session: BrowserDecoderStackSession,
) {
    let _ = owner.complete(lease, session, CompletionAction::Restore);
}

fn prepare_logits_readout(
    owner: &AsyncSessionOwner<BrowserDecoderStackSession>,
    operation: &str,
) -> Result<
    (
        crate::AsyncSessionLease,
        BrowserDecoderStackSession,
        bool,
        u64,
    ),
    JsValue,
> {
    let (lease, session) = acquire_stack_session(owner)?;
    if session.lm_head_plan.is_none() {
        restore_stack_session(owner, lease, session);
        return Err(js_stack_error(&[
            "decoder stack session does not admit the ",
            operation,
            " operation",
        ]));
    }
    let has_admitted_operation = if session.prefill_hidden_storage_buffer.is_some() {
        session.cache_tokens >= 1
    } else {
        session.cache_tokens > session.kv_plan.initial_cache_tokens
    };
    if !has_admitted_operation {
        restore_stack_session(owner, lease, session);
        return Err(js_stack_error(&[
            "decoder stack session ",
            operation,
            " requires an admitted prefill or decode step",
        ]));
    }
    let from_prefill = session.prefill_hidden_storage_buffer.is_some()
        && session.cache_tokens == session.prefill_plan.tokens;
    let last_row_offset = if from_prefill {
        match u64::from(session.prefill_plan.tokens.saturating_sub(1)).checked_mul(4096) {
            Some(offset) => offset,
            None => {
                restore_stack_session(owner, lease, session);
                return Err(js_stack_error(&[
                    "decoder stack session ",
                    operation,
                    " hidden row offset overflowed",
                ]));
            }
        }
    } else {
        0
    };
    Ok((lease, session, from_prefill, last_row_offset))
}

fn parse_stack_descriptor_json(json: &str) -> Result<ParsedDescriptor, JsValue> {
    let value: Value = serde_json::from_str(json).map_err(|error| {
        js_stack_error(&[
            "invalid decoder stack session descriptor json: ",
            &error.to_string(),
        ])
    })?;
    let object = value.as_object().ok_or_else(|| {
        js_stack_error(&["invalid decoder stack session descriptor: expected an object"])
    })?;
    for key in object.keys() {
        if !DESCRIPTOR_FIELDS.contains(&key.as_str()) {
            return Err(js_stack_error(&[
                "invalid decoder stack session descriptor: unknown field ",
                key,
            ]));
        }
    }
    let schema_version = required_descriptor_u32(object, "schema_version")?;
    if schema_version != 1 {
        return Err(js_stack_error(&[
            "invalid decoder stack session descriptor schema version",
        ]));
    }
    let layers = required_descriptor_u32(object, "layers")?;
    let hidden_size = required_descriptor_u32(object, "hidden_size")?;
    let intermediate_size = required_descriptor_u32(object, "intermediate_size")?;
    let query_heads = required_descriptor_u32(object, "query_heads")?;
    let key_value_heads = required_descriptor_u32(object, "key_value_heads")?;
    let head_dim = required_descriptor_u32(object, "head_dim")?;
    let query_width = required_descriptor_u32(object, "query_width")?;
    let key_value_width = required_descriptor_u32(object, "key_value_width")?;
    for (field, actual, pinned) in [
        ("layers", layers, pvlc_runtime_core::PINNED_DECODER_LAYERS),
        (
            "hidden_size",
            hidden_size,
            pvlc_runtime_core::PINNED_DECODER_HIDDEN_SIZE,
        ),
        (
            "intermediate_size",
            intermediate_size,
            pvlc_runtime_core::PINNED_DECODER_INTERMEDIATE_SIZE,
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
        ("query_width", query_width, PINNED_STACK_QUERY_WIDTH),
        (
            "key_value_width",
            key_value_width,
            PINNED_STACK_KEY_VALUE_WIDTH,
        ),
    ] {
        if actual != pinned {
            return Err(js_stack_error(&[
                "invalid decoder stack session descriptor geometry: field ",
                field,
                " drifted from the pinned decoder value",
            ]));
        }
    }
    require_descriptor_mrope_sections(object)?;
    let rms_norm_epsilon = required_descriptor_epsilon(object)?;
    let prefix_tokens = required_descriptor_u32(object, "prefix_tokens")?;
    let cache_capacity = required_descriptor_u32(object, "cache_capacity")?;
    let prefill_tokens = optional_descriptor_u32(object, "prefill_tokens")?;
    if prefill_tokens > 0 && prefix_tokens != 0 {
        return Err(js_stack_error(&[
            "invalid decoder stack session descriptor: prefill requires a zero-prefix cache",
        ]));
    }
    let vocab_size = match object.get("vocab_size") {
        Some(_) => {
            let vocab_size = required_descriptor_u32(object, "vocab_size")?;
            if vocab_size != PINNED_STACK_VOCAB_SIZE {
                return Err(js_stack_error(&[
                    "invalid decoder stack session descriptor: vocab_size drifted from the pinned decoder value",
                ]));
            }
            Some(vocab_size)
        }
        None => None,
    };
    Ok(ParsedDescriptor {
        hidden_size,
        intermediate_size,
        query_heads,
        key_value_heads,
        head_dim,
        prefix_tokens,
        cache_capacity,
        rms_norm_epsilon,
        prefill_tokens,
        vocab_size,
    })
}

fn optional_descriptor_u32(object: &Map<String, Value>, key: &str) -> Result<u32, JsValue> {
    match object.get(key) {
        Some(_) => required_descriptor_u32(object, key),
        None => Ok(0),
    }
}

fn required_descriptor_u32(object: &Map<String, Value>, key: &str) -> Result<u32, JsValue> {
    let value = object.get(key).ok_or_else(|| {
        js_stack_error(&[
            "invalid decoder stack session descriptor: missing field ",
            key,
        ])
    })?;
    let integer = value.as_u64().ok_or_else(|| {
        js_stack_error(&[
            "invalid decoder stack session descriptor: field ",
            key,
            " must be an unsigned integer",
        ])
    })?;
    u32::try_from(integer).map_err(|_| {
        js_stack_error(&[
            "invalid decoder stack session descriptor: field ",
            key,
            " is out of range",
        ])
    })
}

fn require_descriptor_mrope_sections(object: &Map<String, Value>) -> Result<(), JsValue> {
    let value = object.get("mrope_sections").ok_or_else(|| {
        js_stack_error(&["invalid decoder stack session descriptor: missing field mrope_sections"])
    })?;
    let array = value.as_array().ok_or_else(|| {
        js_stack_error(&[
            "invalid decoder stack session descriptor: field mrope_sections must be an array",
        ])
    })?;
    let pinned = pvlc_runtime_core::PINNED_DECODER_MROPE_SECTIONS;
    if array.len() != pinned.len() {
        return Err(js_stack_error(&[
            "invalid decoder stack session descriptor: mrope_sections drifted from the pinned [16, 24, 24]",
        ]));
    }
    for (index, section) in array.iter().enumerate() {
        let actual = section.as_u64().ok_or_else(|| {
            js_stack_error(&[
                "invalid decoder stack session descriptor: mrope_sections entries must be unsigned integers",
            ])
        })?;
        if actual != pinned[index] as u64 {
            return Err(js_stack_error(&[
                "invalid decoder stack session descriptor: mrope_sections drifted from the pinned [16, 24, 24]",
            ]));
        }
    }
    Ok(())
}

fn required_descriptor_epsilon(object: &Map<String, Value>) -> Result<f32, JsValue> {
    let value = object.get("rms_norm_epsilon").ok_or_else(|| {
        js_stack_error(&[
            "invalid decoder stack session descriptor: missing field rms_norm_epsilon",
        ])
    })?;
    let epsilon = value.as_f64().ok_or_else(|| {
        js_stack_error(&[
            "invalid decoder stack session descriptor: field rms_norm_epsilon must be a number",
        ])
    })? as f32;
    if epsilon != pvlc_runtime_core::PINNED_DECODER_RMS_NORM_EPSILON {
        return Err(js_stack_error(&[
            "invalid decoder stack session descriptor: rms_norm_epsilon drifted from the pinned decoder value",
        ]));
    }
    Ok(epsilon)
}

fn stack_uint8_to_bytes(value: &js_sys::Uint8Array) -> Result<Vec<u8>, JsValue> {
    if !value.is_instance_of::<js_sys::Uint8Array>() {
        return Err(js_stack_error(&[
            "decoder stack session operand must be a Uint8Array view",
        ]));
    }
    if value.byte_length() == 0 {
        return Ok(Vec::new());
    }
    Ok(value.to_vec())
}

fn stack_pack_to_bytes(value: &js_sys::Uint8Array) -> Result<Vec<u8>, JsValue> {
    if !value.is_instance_of::<js_sys::Uint8Array>() {
        return Err(js_stack_error(&[
            "decoder stack weight pack must be a Uint8Array view",
        ]));
    }
    if value.byte_length() == 0 {
        return Ok(Vec::new());
    }
    Ok(value.to_vec())
}

fn stack_bytes_to_f32(bytes: &[u8], label: &str) -> Result<Vec<f32>, JsValue> {
    if !bytes.len().is_multiple_of(4) {
        return Err(js_stack_error(&[
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

fn canonical_stack_sources() -> Result<StackShaderSources, JsValue> {
    let rms_norm = pvlc_wgsl::module(pvlc_runtime_core::KernelId::RmsNormF32)
        .ok_or_else(|| js_stack_error(&["canonical decoder stack RMS-norm kernel is missing"]))?;
    let gemv = pvlc_wgsl::module(pvlc_runtime_core::KernelId::GemvF32)
        .ok_or_else(|| js_stack_error(&["canonical decoder stack GEMV kernel is missing"]))?;
    let gemv_tiled = pvlc_wgsl::module(pvlc_runtime_core::KernelId::GemvTiledF32)
        .ok_or_else(|| js_stack_error(&["canonical decoder stack tiled GEMV kernel is missing"]))?;
    let rms_norm_f16_weights = pvlc_wgsl::module(pvlc_runtime_core::KernelId::RmsNormF16Weights)
        .ok_or_else(|| {
            js_stack_error(&["canonical decoder stack FP16 RMS-norm kernel is missing"])
        })?;
    let gemv_tiled_f16_weights =
        pvlc_wgsl::module(pvlc_runtime_core::KernelId::GemvTiledF16Weights).ok_or_else(|| {
            js_stack_error(&["canonical decoder stack FP16 tiled GEMV kernel is missing"])
        })?;
    let linear_projection_f16_weights =
        pvlc_wgsl::module(pvlc_runtime_core::KernelId::LinearProjectionF16Weights).ok_or_else(
            || js_stack_error(&["canonical decoder stack FP16 prefill linear kernel is missing"]),
        )?;
    let mrope = pvlc_wgsl::module(pvlc_runtime_core::KernelId::DecoderMropeF32)
        .ok_or_else(|| js_stack_error(&["canonical decoder stack M-RoPE kernel is missing"]))?;
    let append = pvlc_wgsl::module(pvlc_runtime_core::KernelId::DecoderKvAppendF32)
        .ok_or_else(|| js_stack_error(&["canonical decoder stack append kernel is missing"]))?;
    let attention = pvlc_wgsl::module(pvlc_runtime_core::KernelId::DecoderGqaF32)
        .ok_or_else(|| js_stack_error(&["canonical decoder stack GQA kernel is missing"]))?;
    let swiglu = pvlc_wgsl::module(pvlc_runtime_core::KernelId::DecoderSwigluF32)
        .ok_or_else(|| js_stack_error(&["canonical decoder stack SwiGLU kernel is missing"]))?;
    let residual = pvlc_wgsl::module(pvlc_runtime_core::KernelId::AddF32)
        .ok_or_else(|| js_stack_error(&["canonical decoder stack residual kernel is missing"]))?;
    let projection = pvlc_wgsl::module(pvlc_runtime_core::KernelId::VisionPatchProjectionF32)
        .ok_or_else(|| {
            js_stack_error(&["canonical decoder stack prefill projection kernel is missing"])
        })?;
    let prefill_mrope = pvlc_wgsl::module(pvlc_runtime_core::KernelId::DecoderPrefillMropeF32)
        .ok_or_else(|| {
            js_stack_error(&["canonical decoder stack prefill M-RoPE kernel is missing"])
        })?;
    let kv_append_range = pvlc_wgsl::module(pvlc_runtime_core::KernelId::DecoderKvAppendRangeF32)
        .ok_or_else(|| {
        js_stack_error(&["canonical decoder stack KV range-append kernel is missing"])
    })?;
    let prefill_gqa = pvlc_wgsl::module(pvlc_runtime_core::KernelId::DecoderPrefillGqaF32)
        .ok_or_else(|| {
            js_stack_error(&["canonical decoder stack prefill GQA kernel is missing"])
        })?;
    let split_partial = pvlc_wgsl::module(pvlc_runtime_core::KernelId::DecoderGqaSplitPartialF32)
        .ok_or_else(|| {
        js_stack_error(&["canonical decoder stack split partial kernel is missing"])
    })?;
    let split_merge = pvlc_wgsl::module(pvlc_runtime_core::KernelId::DecoderGqaSplitMergeF32)
        .ok_or_else(|| {
            js_stack_error(&["canonical decoder stack split merge kernel is missing"])
        })?;
    Ok(StackShaderSources {
        rms_norm: rms_norm.source.to_owned(),
        gemv: gemv.source.to_owned(),
        gemv_tiled: gemv_tiled.source.to_owned(),
        rms_norm_f16_weights: rms_norm_f16_weights.source.to_owned(),
        gemv_tiled_f16_weights: gemv_tiled_f16_weights.source.to_owned(),
        linear_projection_f16_weights: linear_projection_f16_weights.source.to_owned(),
        mrope: mrope.source.to_owned(),
        append: append.source.to_owned(),
        attention: attention.source.to_owned(),
        swiglu: swiglu.source.to_owned(),
        residual: residual.source.to_owned(),
        projection: projection.source.to_owned(),
        prefill_mrope: prefill_mrope.source.to_owned(),
        kv_append_range: kv_append_range.source.to_owned(),
        prefill_gqa: prefill_gqa.source.to_owned(),
        split_partial: split_partial.source.to_owned(),
        split_merge: split_merge.source.to_owned(),
    })
}

fn source_blake3(source: &str) -> [u8; 32] {
    *blake3::hash(source.as_bytes()).as_bytes()
}

fn blake3_hex(digest: &[u8; 32]) -> String {
    blake3::Hash::from_bytes(*digest).to_hex().to_string()
}

fn resident_weight_key(
    checkpoint_blake3: Option<[u8; 32]>,
    plan: &pvlc_runtime_core::DecoderWeightResourcePlan,
) -> Option<String> {
    let checkpoint = checkpoint_blake3?;
    let bytes = plan
        .layer_weight_bulk_bytes
        .iter()
        .try_fold(0_u64, |total, value| total.checked_add(*value))?
        .checked_add(plan.final_norm_weight_bytes.unwrap_or(0))?
        .checked_add(plan.lm_head_weight_bytes.unwrap_or(0))?;
    let mut key = blake3_hex(&checkpoint);
    key.push(':');
    key.push_str(match plan.storage {
        DecoderWeightStorage::F32 => "f32",
        DecoderWeightStorage::F16 => "f16",
    });
    key.push(':');
    key.push_str(&bytes.to_string());
    Some(key)
}

fn resident_weight_bytes(
    plan: &pvlc_runtime_core::DecoderWeightResourcePlan,
) -> Result<u64, JsValue> {
    plan.layer_weight_bulk_bytes
        .iter()
        .try_fold(0_u64, |total, bytes| total.checked_add(*bytes))
        .and_then(|total| total.checked_add(plan.final_norm_weight_bytes.unwrap_or(0)))
        .and_then(|total| total.checked_add(plan.lm_head_weight_bytes.unwrap_or(0)))
        .ok_or_else(|| js_stack_error(&["decoder resident weight byte size overflowed"]))
}

fn stack_shader_digests(sources: &StackShaderSources) -> StackShaderDigests {
    StackShaderDigests {
        rms_norm: source_blake3(&sources.rms_norm),
        gemv: source_blake3(&sources.gemv),
        gemv_tiled: source_blake3(&sources.gemv_tiled),
        mrope: source_blake3(&sources.mrope),
        append: source_blake3(&sources.append),
        attention: source_blake3(&sources.attention),
        swiglu: source_blake3(&sources.swiglu),
        residual: source_blake3(&sources.residual),
        split_partial: source_blake3(&sources.split_partial),
        split_merge: source_blake3(&sources.split_merge),
    }
}

fn validate_stack_capabilities(
    device: &wgpu::Device,
    kv_plan: &pvlc_runtime_core::DecoderKvSessionPlan,
    stack_plan: &pvlc_runtime_core::DecoderStackPlan,
    weight_resource_plan: &pvlc_runtime_core::DecoderWeightResourcePlan,
    prefill_plan: &pvlc_runtime_core::DecoderStackPrefillPlan,
    prefill_capable: bool,
    lm_head_plan: Option<&pvlc_runtime_core::DecoderLmHeadPlan>,
) -> Result<(), JsValue> {
    let limits = device.limits();
    if limits.max_storage_buffers_per_shader_stage < 6 {
        return Err(js_stack_error(&[
            "decoder stack session requires six storage buffers per shader stage",
        ]));
    }
    if limits.min_storage_buffer_offset_alignment > 256 {
        return Err(js_stack_error(&[
            "decoder stack session requires a storage buffer offset alignment of at most 256 bytes",
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
        &stack_plan.layer_plan.attention_block.rms_norm_invocation,
        &stack_plan.layer_plan.attention_block.query_invocation,
        &stack_plan.layer_plan.attention_block.key_invocation,
        &stack_plan.layer_plan.attention_block.value_invocation,
        &stack_plan.layer_plan.attention_block.mrope_invocation,
        &kv_plan.append_invocation,
        &kv_plan.attention_invocation,
        &stack_plan.layer_plan.attention_block.output_invocation,
        &stack_plan.layer_plan.attention_block.residual_invocation,
        &stack_plan.layer_plan.norm2_invocation,
        &stack_plan.layer_plan.gate_invocation,
        &stack_plan.layer_plan.up_invocation,
        &stack_plan.layer_plan.swiglu_invocation,
        &stack_plan.layer_plan.down_invocation,
        &stack_plan.layer_plan.second_residual_invocation,
    ] {
        dispatch_limits.validate(invocation).map_err(|error| {
            js_stack_error(&[
                "decoder stack session exceeds adapter dispatch limits: ",
                &error.to_string(),
            ])
        })?;
    }
    let hidden_bytes = checked_u64_bytes(
        stack_plan.layer_plan.attention_block.hidden_size as usize,
        "decoder hidden row",
    )?;
    let query_bytes = checked_u64_bytes(
        stack_plan.layer_plan.attention_block.query_width,
        "decoder query row",
    )?;
    let key_value_bytes = checked_u64_bytes(
        stack_plan.layer_plan.attention_block.key_value_width,
        "decoder key/value row",
    )?;
    let intermediate_bytes = checked_u64_bytes(
        stack_plan.layer_plan.intermediate_size as usize,
        "decoder intermediate row",
    )?;
    let rope_bytes = weight_resource_plan.rope_table_bytes;
    let cache_bulk = u64::from(stack_plan.layers)
        .checked_mul(stack_plan.cache_stride_bytes)
        .ok_or_else(|| js_stack_error(&["decoder stack bulk byte size overflowed"]))?;
    let weight_strides = weight_resource_plan.layer_weight_stride_bytes;
    let weight_bulks = weight_resource_plan.layer_weight_bulk_bytes;
    for (label, bytes) in [
        ("decoder stack norm1 weight slice", weight_strides[0]),
        ("decoder stack query weight slice", weight_strides[1]),
        ("decoder stack key weight slice", weight_strides[2]),
        ("decoder stack value weight slice", weight_strides[3]),
        ("decoder stack output weight slice", weight_strides[4]),
        ("decoder stack norm2 weight slice", weight_strides[5]),
        ("decoder stack gate weight slice", weight_strides[6]),
        ("decoder stack up weight slice", weight_strides[7]),
        ("decoder stack down weight slice", weight_strides[8]),
        ("decoder stack rope table", rope_bytes),
        (
            "decoder stack compact cache slice",
            stack_plan.cache_stride_bytes,
        ),
        ("decoder stack attention output", kv_plan.attention_bytes),
        ("decoder stack hidden row", hidden_bytes),
        ("decoder stack query row", query_bytes),
        ("decoder stack key/value row", key_value_bytes),
        ("decoder stack intermediate row", intermediate_bytes),
        ("decoder stack norm1 weight bulk", weight_bulks[0]),
        ("decoder stack query weight bulk", weight_bulks[1]),
        ("decoder stack key weight bulk", weight_bulks[2]),
        ("decoder stack value weight bulk", weight_bulks[3]),
        ("decoder stack output weight bulk", weight_bulks[4]),
        ("decoder stack norm2 weight bulk", weight_bulks[5]),
        ("decoder stack gate weight bulk", weight_bulks[6]),
        ("decoder stack up weight bulk", weight_bulks[7]),
        ("decoder stack down weight bulk", weight_bulks[8]),
        ("decoder stack compact cache bulk", cache_bulk),
    ] {
        if bytes > limits.max_storage_buffer_binding_size {
            return Err(js_stack_error(&[
                label,
                " exceeds the adapter storage buffer binding limit",
            ]));
        }
    }
    for (label, bytes) in [
        ("decoder stack norm1 weight bulk", weight_bulks[0]),
        ("decoder stack query weight bulk", weight_bulks[1]),
        ("decoder stack key weight bulk", weight_bulks[2]),
        ("decoder stack value weight bulk", weight_bulks[3]),
        ("decoder stack output weight bulk", weight_bulks[4]),
        ("decoder stack norm2 weight bulk", weight_bulks[5]),
        ("decoder stack gate weight bulk", weight_bulks[6]),
        ("decoder stack up weight bulk", weight_bulks[7]),
        ("decoder stack down weight bulk", weight_bulks[8]),
        ("decoder stack compact cache bulk", cache_bulk),
    ] {
        if bytes > limits.max_buffer_size {
            return Err(js_stack_error(&[
                label,
                " exceeds the adapter buffer size limit",
            ]));
        }
    }
    // The split-K decode attention dispatch and scratch plane must fit the
    // adapter: the partial dispatch grows with the cache span, and the
    // partials plane is one storage buffer.
    let split_partial_workgroups =
        u64::from(kv_plan.cache_capacity.div_ceil(32)).saturating_mul(16);
    if split_partial_workgroups > u64::from(limits.max_compute_workgroups_per_dimension) {
        return Err(js_stack_error(&[
            "decoder stack session split partial dispatch exceeds the adapter workgroup limit",
        ]));
    }
    if kv_plan.split_partials_bytes > limits.max_storage_buffer_binding_size
        || kv_plan.split_partials_bytes > limits.max_buffer_size
    {
        return Err(js_stack_error(&[
            "decoder stack split partials plane exceeds the adapter buffer limits",
        ]));
    }
    if prefill_capable {
        for invocation in &prefill_plan.stage_invocations {
            dispatch_limits.validate(invocation).map_err(|error| {
                js_stack_error(&[
                    "decoder stack session prefill exceeds adapter dispatch limits: ",
                    &error.to_string(),
                ])
            })?;
        }
        let capacity = u64::from(kv_plan.cache_capacity);
        let mut prefill_hidden_bytes = [0u64; 1];
        let mut prefill_query_bytes = [0u64; 1];
        let mut prefill_key_value_bytes = [0u64; 1];
        let mut prefill_intermediate_bytes = [0u64; 1];
        for (target, width) in [
            (&mut prefill_hidden_bytes, PREFILL_HIDDEN_WIDTH),
            (&mut prefill_query_bytes, PREFILL_QUERY_WIDTH),
            (&mut prefill_key_value_bytes, PREFILL_KEY_VALUE_WIDTH),
            (&mut prefill_intermediate_bytes, PREFILL_INTERMEDIATE_WIDTH),
        ] {
            target[0] = capacity
                .checked_mul(width)
                .and_then(|value| value.checked_mul(4))
                .ok_or_else(|| {
                    js_stack_error(&["decoder stack prefill stage byte size overflowed"])
                })?;
        }
        for (label, bytes) in [
            (
                "decoder stack prefill hidden storage",
                prefill_hidden_bytes[0],
            ),
            ("decoder stack prefill query stage", prefill_query_bytes[0]),
            (
                "decoder stack prefill key/value stage",
                prefill_key_value_bytes[0],
            ),
            (
                "decoder stack prefill intermediate stage",
                prefill_intermediate_bytes[0],
            ),
            ("decoder stack prefill zero bias", PREFILL_ZERO_BIAS_BYTES),
        ] {
            if bytes > limits.max_storage_buffer_binding_size {
                return Err(js_stack_error(&[
                    label,
                    " exceeds the adapter storage buffer binding limit",
                ]));
            }
            if bytes > limits.max_buffer_size {
                return Err(js_stack_error(&[
                    label,
                    " exceeds the adapter buffer size limit",
                ]));
            }
        }
    }
    if let Some(plan) = lm_head_plan {
        let final_norm_weight_bytes = weight_resource_plan.final_norm_weight_bytes.ok_or_else(
            || {
                js_stack_error(&[
                    "decoder stack physical final norm weight is missing from the resource plan",
                ])
            },
        )?;
        let lm_head_weight_bytes = weight_resource_plan.lm_head_weight_bytes.ok_or_else(|| {
            js_stack_error(&[
                "decoder stack physical LM head weight is missing from the resource plan",
            ])
        })?;
        for invocation in &plan.stage_invocations {
            dispatch_limits.validate(invocation).map_err(|error| {
                js_stack_error(&[
                    "decoder stack session logits exceeds adapter dispatch limits: ",
                    &error.to_string(),
                ])
            })?;
        }
        for (label, bytes) in [
            ("decoder stack final norm weight", final_norm_weight_bytes),
            ("decoder stack LM head weight", lm_head_weight_bytes),
            ("decoder stack normed row", plan.normed_row_bytes),
            ("decoder stack logits", plan.logits_bytes),
        ] {
            if bytes > limits.max_storage_buffer_binding_size {
                return Err(js_stack_error(&[
                    label,
                    " exceeds the adapter storage buffer binding limit",
                ]));
            }
            if bytes > limits.max_buffer_size {
                return Err(js_stack_error(&[
                    label,
                    " exceeds the adapter buffer size limit",
                ]));
            }
        }
    }
    Ok(())
}

fn js_object_set(object: &js_sys::Object, key: &str, value: &JsValue) -> Result<(), JsValue> {
    js_sys::Reflect::set(object, &JsValue::from_str(key), value)
        .map(|_| ())
        .map_err(|error| {
            js_stack_error(&[
                "cannot build WebGPU descriptor field ",
                key,
                ": ",
                &js_error_text(&error),
            ])
        })
}

fn create_stack_buffer(
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
    raw_stack_method(device, "createBuffer")?
        .call1(device, &descriptor)
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack session buffer creation failed: ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<wgpu::webgpu::GpuBuffer>()
        .map_err(|_| js_stack_error(&["createBuffer did not return a GPUBuffer"]))
}

/// Creates a buffer with its fully static initial contents written at
/// creation time (`mappedAtCreation`): the two logits stage-uniform buffers
/// carry their pinned word sets from the exact core plan and are never
/// queue-written, so one logits call performs zero writes.
fn create_stack_buffer_initialized(
    device: &JsValue,
    label: &str,
    size: u64,
    usage: u32,
    initial: &[u8],
) -> Result<wgpu::webgpu::GpuBuffer, JsValue> {
    let descriptor = js_sys::Object::new();
    js_object_set(&descriptor, "label", &JsValue::from_str(label))?;
    js_object_set(&descriptor, "size", &JsValue::from_f64(size as f64))?;
    js_object_set(&descriptor, "usage", &JsValue::from_f64(f64::from(usage)))?;
    js_object_set(&descriptor, "mappedAtCreation", &JsValue::TRUE)?;
    let buffer = raw_stack_method(device, "createBuffer")?
        .call1(device, &descriptor)
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack session buffer creation failed: ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<wgpu::webgpu::GpuBuffer>()
        .map_err(|_| js_stack_error(&["createBuffer did not return a GPUBuffer"]))?;
    let range = raw_stack_method(buffer.as_ref(), "getMappedRange")?
        .call0(buffer.as_ref())
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack session static buffer map failed: ",
                &js_error_text(&error),
            ])
        })?;
    js_sys::Uint8Array::new(&range).copy_from(initial);
    unmap_stack_buffer(buffer.as_ref())?;
    Ok(buffer)
}

fn create_stack_bind_group_layout(
    device: &JsValue,
    entries: &[(u32, &str, bool)],
) -> Result<js_sys::Object, JsValue> {
    let entry_array = js_sys::Array::new();
    for (binding, buffer_type, dynamic) in entries {
        let buffer = js_sys::Object::new();
        js_object_set(&buffer, "type", &JsValue::from_str(buffer_type))?;
        js_object_set(&buffer, "hasDynamicOffset", &JsValue::from_bool(*dynamic))?;
        let entry = js_sys::Object::new();
        js_object_set(&entry, "binding", &JsValue::from_f64(f64::from(*binding)))?;
        js_object_set(&entry, "visibility", &JsValue::from_f64(4.0))?;
        js_object_set(&entry, "buffer", &buffer)?;
        entry_array.push(&entry);
    }
    let descriptor = js_sys::Object::new();
    js_object_set(&descriptor, "entries", &entry_array)?;
    raw_stack_method(device, "createBindGroupLayout")?
        .call1(device, &descriptor)
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack bind group layout creation failed: ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<js_sys::Object>()
        .map_err(|_| js_stack_error(&["createBindGroupLayout did not return an object"]))
}

fn create_stack_pipeline(
    device: &JsValue,
    kernel: &str,
    source: &str,
    layout: &js_sys::Object,
) -> Result<js_sys::Object, JsValue> {
    let layouts = js_sys::Array::new();
    layouts.push(layout);
    let layout_descriptor = js_sys::Object::new();
    js_object_set(&layout_descriptor, "bindGroupLayouts", &layouts)?;
    let pipeline_layout = raw_stack_method(device, "createPipelineLayout")?
        .call1(device, &layout_descriptor)
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack pipeline layout creation failed: ",
                &js_error_text(&error),
            ])
        })?;
    let shader_descriptor = js_sys::Object::new();
    js_object_set(&shader_descriptor, "label", &JsValue::from_str(kernel))?;
    js_object_set(&shader_descriptor, "code", &JsValue::from_str(source))?;
    let shader = raw_stack_method(device, "createShaderModule")?
        .call1(device, &shader_descriptor)
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack shader module creation failed: ",
                &js_error_text(&error),
            ])
        })?;
    let compute = js_sys::Object::new();
    js_object_set(&compute, "module", &shader)?;
    js_object_set(&compute, "entryPoint", &JsValue::from_str(ENTRY_POINT))?;
    let pipeline_descriptor = js_sys::Object::new();
    js_object_set(&pipeline_descriptor, "label", &JsValue::from_str(kernel))?;
    js_object_set(&pipeline_descriptor, "layout", &pipeline_layout)?;
    js_object_set(&pipeline_descriptor, "compute", &compute)?;
    raw_stack_method(device, "createComputePipeline")?
        .call1(device, &pipeline_descriptor)
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack compute pipeline creation failed: ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<js_sys::Object>()
        .map_err(|_| js_stack_error(&["createComputePipeline did not return an object"]))
}

fn create_stack_bind_group(
    device: &JsValue,
    label: &str,
    layout: &js_sys::Object,
    entries: &[(&wgpu::webgpu::GpuBuffer, u64)],
) -> Result<js_sys::Object, JsValue> {
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
    js_object_set(&descriptor, "layout", layout)?;
    js_object_set(&descriptor, "entries", &entry_array)?;
    raw_stack_method(device, "createBindGroup")?
        .call1(device, &descriptor)
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack bind group creation failed: ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<js_sys::Object>()
        .map_err(|_| js_stack_error(&["createBindGroup did not return an object"]))
}

fn write_stack_buffer(
    queue: &JsValue,
    buffer: &wgpu::webgpu::GpuBuffer,
    bytes: &[u8],
) -> Result<(), JsValue> {
    let data = js_sys::Uint8Array::from(bytes);
    raw_stack_method(queue, "writeBuffer")?
        .call3(queue, buffer.as_ref(), &JsValue::from_f64(0.0), &data)
        .map(|_| ())
        .map_err(|error| {
            js_stack_error(&["decoder stack queue write failed: ", &js_error_text(&error)])
        })
}

fn create_stack_encoder(device: &JsValue, label: &str) -> Result<JsValue, JsValue> {
    let descriptor = js_sys::Object::new();
    js_object_set(&descriptor, "label", &JsValue::from_str(label))?;
    raw_stack_method(device, "createCommandEncoder")?
        .call1(device, &descriptor)
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack command encoder creation failed: ",
                &js_error_text(&error),
            ])
        })
}

fn encode_stack_pass(
    encoder: &JsValue,
    pipeline: &js_sys::Object,
    bind_group: &js_sys::Object,
    dispatch: [u32; 3],
    dynamic_offsets: &[u64],
) -> Result<(), JsValue> {
    let pass = raw_stack_method(encoder, "beginComputePass")?
        .call0(encoder)
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack compute pass begin failed: ",
                &js_error_text(&error),
            ])
        })?;
    raw_stack_method(&pass, "setPipeline")?
        .call1(&pass, pipeline)
        .map_err(|error| {
            js_stack_error(&["decoder stack setPipeline failed: ", &js_error_text(&error)])
        })?;
    let offsets = js_sys::Array::new();
    for offset in dynamic_offsets {
        offsets.push(&JsValue::from_f64(*offset as f64));
    }
    raw_stack_method(&pass, "setBindGroup")?
        .call3(&pass, &JsValue::from_f64(0.0), bind_group, &offsets)
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack setBindGroup failed: ",
                &js_error_text(&error),
            ])
        })?;
    raw_stack_method(&pass, "dispatchWorkgroups")?
        .call3(
            &pass,
            &JsValue::from_f64(f64::from(dispatch[0])),
            &JsValue::from_f64(f64::from(dispatch[1])),
            &JsValue::from_f64(f64::from(dispatch[2])),
        )
        .map_err(|error| {
            js_stack_error(&["decoder stack dispatch failed: ", &js_error_text(&error)])
        })?;
    raw_stack_method(&pass, "end")?
        .call0(&pass)
        .map(|_| ())
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack compute pass end failed: ",
                &js_error_text(&error),
            ])
        })
}

fn encode_stack_copy(
    encoder: &JsValue,
    source: &wgpu::webgpu::GpuBuffer,
    source_offset: u64,
    destination: &wgpu::webgpu::GpuBuffer,
    destination_offset: u64,
    bytes: u64,
) -> Result<(), JsValue> {
    raw_stack_method(encoder, "copyBufferToBuffer")?
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
            js_stack_error(&["decoder stack buffer copy failed: ", &js_error_text(&error)])
        })
}

fn submit_stack_encoder(queue: &JsValue, encoder: &JsValue) -> Result<(), JsValue> {
    let command = raw_stack_method(encoder, "finish")?
        .call0(encoder)
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack command encoder finish failed: ",
                &js_error_text(&error),
            ])
        })?;
    let commands = js_sys::Array::new();
    commands.push(&command);
    raw_stack_method(queue, "submit")?
        .call1(queue, &commands)
        .map(|_| ())
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack queue submission failed: ",
                &js_error_text(&error),
            ])
        })
}

async fn await_stack_queue_completion(queue: &JsValue) -> Result<(), JsValue> {
    let pending = raw_stack_method(queue, "onSubmittedWorkDone")?
        .call0(queue)
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack queue completion request failed: ",
                &js_error_text(&error),
            ])
        })?
        .dyn_into::<js_sys::Promise>()
        .map_err(|_| js_stack_error(&["onSubmittedWorkDone did not return a Promise"]))?;
    JsFuture::from(pending).await.map(|_| ()).map_err(|error| {
        js_stack_error(&[
            "decoder stack queue completion rejected: ",
            &js_error_text(&error),
        ])
    })
}

async fn map_stack_buffer(buffer: &JsValue, bytes: u64) -> Result<(), JsValue> {
    let pending = raw_stack_method(buffer, "mapAsync")?
        .call3(
            buffer,
            &JsValue::from_f64(1.0),
            &JsValue::from_f64(0.0),
            &JsValue::from_f64(bytes as f64),
        )
        .map_err(|error| {
            js_stack_error(&["decoder stack map request failed: ", &js_error_text(&error)])
        })?
        .dyn_into::<js_sys::Promise>()
        .map_err(|_| js_stack_error(&["mapAsync did not return a Promise"]))?;
    JsFuture::from(pending).await.map(|_| ()).map_err(|error| {
        js_stack_error(&[
            "decoder stack buffer mapping rejected: ",
            &js_error_text(&error),
        ])
    })
}

fn read_stack_mapped(buffer: &JsValue, bytes: u64) -> Result<Vec<u8>, JsValue> {
    let range = raw_stack_method(buffer, "getMappedRange")?
        .call2(
            buffer,
            &JsValue::from_f64(0.0),
            &JsValue::from_f64(bytes as f64),
        )
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack mapped range read failed: ",
                &js_error_text(&error),
            ])
        })?;
    Ok(js_sys::Uint8Array::new(&range).to_vec())
}

fn unmap_stack_buffer(buffer: &JsValue) -> Result<(), JsValue> {
    raw_stack_method(buffer, "unmap")?
        .call0(buffer)
        .map(|_| ())
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack buffer unmap failed: ",
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
        .ok_or_else(|| js_stack_error(&[label, " byte size overflowed"]))
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
        .map_err(|_| js_stack_error(&["decoder stack weight pack ", label, " offset is too large"]))
}

fn pack_slice<'a>(
    bytes: &'a [u8],
    start: usize,
    end: usize,
    label: &str,
) -> Result<&'a [u8], JsValue> {
    if start > end || end > bytes.len() {
        return Err(js_stack_error(&[
            "decoder stack weight pack ",
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
        _ => Err(js_stack_error(&[
            "decoder stack weight pack ",
            label,
            " is not lowercase BLAKE3 hex",
        ])),
    }
}

fn pack_hex_digest(hex: &str, label: &str) -> Result<[u8; 32], JsValue> {
    let digits = hex.as_bytes();
    if digits.len() != 64 {
        return Err(js_stack_error(&[
            "decoder stack weight pack ",
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
        js_stack_error(&[
            "decoder stack weight pack ",
            label,
            " UTF-8 is invalid: ",
            &error.to_string(),
        ])
    })?;
    if !text.ends_with('\n') || text.ends_with("\n\n") {
        return Err(js_stack_error(&[
            "decoder stack weight pack ",
            label,
            " is not newline-terminated canonical JSON",
        ]));
    }
    let value: Value = serde_json::from_str(text).map_err(|error| {
        js_stack_error(&[
            "decoder stack weight pack ",
            label,
            " JSON is invalid: ",
            &error.to_string(),
        ])
    })?;
    value.as_object().cloned().ok_or_else(|| {
        js_stack_error(&[
            "decoder stack weight pack ",
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
            return Err(js_stack_error(&[
                "decoder stack weight pack ",
                label,
                " has unknown field ",
                key,
            ]));
        }
    }
    for key in expected {
        if !object.contains_key(*key) {
            return Err(js_stack_error(&[
                "decoder stack weight pack ",
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
        js_stack_error(&[
            "decoder stack weight pack ",
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
        js_stack_error(&[
            "decoder stack weight pack ",
            label,
            " field ",
            key,
            " must be a string",
        ])
    })
}

fn validate_pack_manifest(payload: &[u8]) -> Result<DecoderWeightStorage, JsValue> {
    let manifest = pack_json_object(payload, "manifest")?;
    require_pack_exact_keys(&manifest, &PACK_MANIFEST_FIELDS, "manifest")?;
    if pack_json_str(&manifest, "model_id", "manifest")? != PACK_MODEL_ID {
        return Err(js_stack_error(&[
            "decoder stack weight pack manifest model_id drifted",
        ]));
    }
    if pack_json_str(&manifest, "model_revision", "manifest")? != PACK_MODEL_REVISION {
        return Err(js_stack_error(&[
            "decoder stack weight pack manifest model_revision drifted",
        ]));
    }
    if pack_json_u64(&manifest, "compiler_model_abi", "manifest")? != 1 {
        return Err(js_stack_error(&[
            "decoder stack weight pack manifest compiler_model_abi drifted",
        ]));
    }
    pack_hex_digest(
        pack_json_str(&manifest, "compiler_build", "manifest")?,
        "manifest compiler_build",
    )?;
    let weight_storage = match pack_json_str(&manifest, "precision_profile", "manifest")? {
        "fidelity" => DecoderWeightStorage::F32,
        "balanced" => DecoderWeightStorage::F16,
        _ => {
            return Err(js_stack_error(&[
                "decoder stack weight pack manifest precision_profile is unsupported",
            ]));
        }
    };
    if pack_json_u64(&manifest, "context_limit", "manifest")? == 0 {
        return Err(js_stack_error(&[
            "decoder stack weight pack manifest context_limit drifted",
        ]));
    }
    let buckets = manifest
        .get("resolution_buckets")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            js_stack_error(&[
                "decoder stack weight pack manifest resolution_buckets must be an array",
            ])
        })?;
    if buckets.is_empty() {
        return Err(js_stack_error(&[
            "decoder stack weight pack manifest resolution_buckets drifted",
        ]));
    }
    for bucket in buckets {
        let pair = bucket.as_array().ok_or_else(|| {
            js_stack_error(&["decoder stack weight pack manifest resolution bucket must be a pair"])
        })?;
        if pair.len() != 2
            || pair
                .iter()
                .any(|side| side.as_u64().is_none_or(|side| side == 0))
        {
            return Err(js_stack_error(&[
                "decoder stack weight pack manifest resolution bucket drifted",
            ]));
        }
    }
    Ok(weight_storage)
}

fn expected_shard_bytes(
    shard_index: usize,
    cache_capacity: u32,
    weight_storage: DecoderWeightStorage,
) -> Result<u64, JsValue> {
    if weight_storage.storage_bytes(1).is_none() {
        return Err(js_stack_error(&[
            "decoder stack weight storage has no addressable element size",
        ]));
    }
    let hidden = u64::from(pvlc_runtime_core::PINNED_DECODER_HIDDEN_SIZE);
    // The two shared logits operands sit at the end of the admitted shard
    // order: the pinned final norm `[1024]` and the pinned LM head
    // `[103424, 1024]`, both f32 and not layer-major.
    if shard_index == LEGACY_PACK_SHARD_COUNT {
        return weight_storage
            .storage_bytes(hidden)
            .ok_or_else(|| js_stack_error(&["decoder stack final norm byte size overflowed"]));
    }
    if shard_index == LEGACY_PACK_SHARD_COUNT + 1 {
        let elements = u64::from(PINNED_STACK_VOCAB_SIZE)
            .checked_mul(hidden)
            .ok_or_else(|| js_stack_error(&["decoder stack LM head element count overflowed"]))?;
        return weight_storage
            .storage_bytes(elements)
            .ok_or_else(|| js_stack_error(&["decoder stack LM head byte size overflowed"]));
    }
    let layers = u64::from(pvlc_runtime_core::PINNED_DECODER_LAYERS);
    let intermediate = u64::from(pvlc_runtime_core::PINNED_DECODER_INTERMEDIATE_SIZE);
    let query_width = u64::from(PINNED_STACK_QUERY_WIDTH);
    let key_value_width = u64::from(PINNED_STACK_KEY_VALUE_WIDTH);
    let per_layer = match shard_index {
        0 | 7 => Some(hidden),
        1 => query_width.checked_mul(hidden),
        2 | 3 => key_value_width.checked_mul(hidden),
        4 => hidden.checked_mul(query_width),
        8 | 9 => intermediate.checked_mul(hidden),
        10 => hidden.checked_mul(intermediate),
        _ => {
            let elements = 3u64
                .checked_mul(u64::from(cache_capacity))
                .and_then(|value| {
                    value.checked_mul(u64::from(pvlc_runtime_core::MAX_DECODER_HEAD_DIM))
                })
                .ok_or_else(|| {
                    js_stack_error(&["decoder stack rope table element count overflowed"])
                })?;
            return elements
                .checked_mul(4)
                .ok_or_else(|| js_stack_error(&["decoder stack rope table byte size overflowed"]));
        }
    };
    let elements = per_layer
        .and_then(|value| value.checked_mul(layers))
        .ok_or_else(|| js_stack_error(&["decoder stack shard element count overflowed"]))?;
    let physical_bytes = weight_storage
        .storage_bytes(elements)
        .ok_or_else(|| js_stack_error(&["decoder stack shard byte size overflowed"]))?;
    let f32_bytes = elements
        .checked_mul(4)
        .ok_or_else(|| js_stack_error(&["decoder stack F32 shard byte size overflowed"]))?;
    if weight_storage.from_f32_byte_offset(f32_bytes) != Some(physical_bytes) {
        return Err(js_stack_error(&[
            "decoder stack shard precision offset mapping drifted",
        ]));
    }
    Ok(physical_bytes)
}

fn require_pack_f32_finite(payload: &[u8], shard_id: &str) -> Result<(), JsValue> {
    for word in payload.chunks_exact(4) {
        if !f32::from_le_bytes([word[0], word[1], word[2], word[3]]).is_finite() {
            return Err(js_stack_error(&[
                "decoder stack weight pack shard ",
                shard_id,
                " contains a nonfinite f32 payload element",
            ]));
        }
    }
    Ok(())
}

fn require_pack_f16_finite(payload: &[u8], shard_id: &str) -> Result<(), JsValue> {
    for word in payload.chunks_exact(2) {
        let bits = u16::from_le_bytes([word[0], word[1]]);
        if bits & 0x7c00 == 0x7c00 {
            return Err(js_stack_error(&[
                "decoder stack weight pack shard ",
                shard_id,
                " contains a nonfinite f16 payload element",
            ]));
        }
    }
    Ok(())
}

fn validate_pack_descriptor(
    payload: &[u8],
    expected_prefix_tokens: u32,
    expected_cache_capacity: u32,
    prefill_capable: bool,
    shard_ids: &[&str],
    weight_storage: DecoderWeightStorage,
) -> Result<ValidatedPackDescriptor, JsValue> {
    let descriptor = pack_json_object(payload, "descriptor")?;
    let descriptor_fields: &[&str] = match weight_storage {
        DecoderWeightStorage::F32 => &PACK_DESCRIPTOR_FIELDS,
        DecoderWeightStorage::F16 => &PACK_BALANCED_DESCRIPTOR_FIELDS,
    };
    require_pack_exact_keys(&descriptor, descriptor_fields, "descriptor")?;
    if pack_json_u64(&descriptor, "schema_version", "descriptor")? != 1 {
        return Err(js_stack_error(&[
            "decoder stack weight pack descriptor schema version drifted",
        ]));
    }
    let oracle = pack_json_str(&descriptor, "oracle", "descriptor")?;
    if !PACK_DESCRIPTOR_ORACLES.contains(&oracle) {
        return Err(js_stack_error(&[
            "decoder stack weight pack descriptor oracle is unsupported",
        ]));
    }
    if pack_json_str(&descriptor, "case_id", "descriptor")?.is_empty() {
        return Err(js_stack_error(&[
            "decoder stack weight pack descriptor case_id is empty",
        ]));
    }
    if pack_json_str(&descriptor, "model_revision", "descriptor")? != PACK_MODEL_REVISION {
        return Err(js_stack_error(&[
            "decoder stack weight pack descriptor model_revision drifted",
        ]));
    }
    for (field, pinned) in [
        ("layers", pvlc_runtime_core::PINNED_DECODER_LAYERS),
        ("hidden_size", pvlc_runtime_core::PINNED_DECODER_HIDDEN_SIZE),
        (
            "intermediate_size",
            pvlc_runtime_core::PINNED_DECODER_INTERMEDIATE_SIZE,
        ),
        ("query_heads", pvlc_runtime_core::PINNED_DECODER_QUERY_HEADS),
        (
            "key_value_heads",
            pvlc_runtime_core::PINNED_DECODER_KEY_VALUE_HEADS,
        ),
        ("head_dim", pvlc_runtime_core::MAX_DECODER_HEAD_DIM),
        ("query_width", PINNED_STACK_QUERY_WIDTH),
        ("key_value_width", PINNED_STACK_KEY_VALUE_WIDTH),
    ] {
        if pack_json_u64(&descriptor, field, "descriptor")? != u64::from(pinned) {
            return Err(js_stack_error(&[
                "decoder stack weight pack descriptor geometry field ",
                field,
                " drifted from the pinned decoder value",
            ]));
        }
    }
    let prefix_tokens = pack_json_u64(&descriptor, "prefix_tokens", "descriptor")?;
    let cache_capacity = pack_json_u64(&descriptor, "cache_capacity", "descriptor")?;
    if (prefix_tokens == 0 && !prefill_capable) || cache_capacity <= prefix_tokens {
        return Err(js_stack_error(&[
            "decoder stack weight pack descriptor cache geometry is invalid",
        ]));
    }
    if prefix_tokens != u64::from(expected_prefix_tokens)
        || cache_capacity != u64::from(expected_cache_capacity)
    {
        return Err(js_stack_error(&[
            "decoder stack weight pack descriptor cache geometry does not match the session descriptor",
        ]));
    }
    let epsilon = descriptor
        .get("rms_norm_epsilon")
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            js_stack_error(&[
                "decoder stack weight pack descriptor rms_norm_epsilon must be a number",
            ])
        })? as f32;
    if epsilon != pvlc_runtime_core::PINNED_DECODER_RMS_NORM_EPSILON {
        return Err(js_stack_error(&[
            "decoder stack weight pack descriptor rms_norm_epsilon drifted",
        ]));
    }
    let sections = descriptor
        .get("mrope_sections")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            js_stack_error(&[
                "decoder stack weight pack descriptor mrope_sections must be an array",
            ])
        })?;
    let pinned = pvlc_runtime_core::PINNED_DECODER_MROPE_SECTIONS;
    if sections.len() != pinned.len()
        || sections
            .iter()
            .enumerate()
            .any(|(index, section)| section.as_u64() != Some(pinned[index] as u64))
    {
        return Err(js_stack_error(&[
            "decoder stack weight pack descriptor mrope_sections drifted",
        ]));
    }
    let checkpoint_blake3 = match weight_storage {
        DecoderWeightStorage::F32 => None,
        DecoderWeightStorage::F16 => {
            if pack_json_str(&descriptor, "weight_storage", "descriptor")? != "f16" {
                return Err(js_stack_error(&[
                    "decoder stack weight pack descriptor weight_storage drifted",
                ]));
            }
            if pack_json_u64(&descriptor, "checkpoint_bytes", "descriptor")? == 0 {
                return Err(js_stack_error(&[
                    "decoder stack weight pack descriptor checkpoint_bytes drifted",
                ]));
            }
            Some(pack_hex_digest(
                pack_json_str(&descriptor, "checkpoint_blake3", "descriptor")?,
                "descriptor checkpoint_blake3",
            )?)
        }
    };
    let shards = descriptor
        .get("shards")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            js_stack_error(&["decoder stack weight pack descriptor shards must be an object"])
        })?;
    require_pack_exact_keys(shards, shard_ids, "descriptor shards")?;
    Ok(ValidatedPackDescriptor {
        shards: shards.clone(),
        checkpoint_blake3,
    })
}

fn parse_stack_weight_pack(
    pack_bytes: &[u8],
    expected_prefix_tokens: u32,
    expected_cache_capacity: u32,
    prefill_capable: bool,
) -> Result<ParsedWeightPack, JsValue> {
    if pack_bytes.len() < PACK_HEADER_BYTES {
        return Err(js_stack_error(&[
            "decoder stack weight pack is shorter than its header",
        ]));
    }
    if pack_bytes[..8] != PACK_MAGIC {
        return Err(js_stack_error(&[
            "decoder stack weight pack magic mismatch",
        ]));
    }
    if pack_u32_le(pack_bytes, 8) != PACK_VERSION {
        return Err(js_stack_error(&[
            "decoder stack weight pack version is unsupported",
        ]));
    }
    let manifest_length = u64::from(pack_u32_le(pack_bytes, 12));
    let directory_length = u64::from(pack_u32_le(pack_bytes, 16));
    let section_count = pack_u32_le(pack_bytes, 20);
    if section_count != LEGACY_PACK_SECTION_COUNT && section_count != PACK_SECTION_COUNT {
        return Err(js_stack_error(&[
            "decoder stack weight pack section count drifted",
        ]));
    }
    // Dual admission: the accepted M6e7 twelve-section pack admits the legacy
    // session (its eleven shards, in order); the M6e8 fourteen-section pack
    // appends exactly the two shared logits shards at the end.
    let shard_ids: &[&str] = if section_count == PACK_SECTION_COUNT {
        &PACK_SHARD_IDS
    } else {
        &PACK_SHARD_IDS[..LEGACY_PACK_SHARD_COUNT]
    };
    let pack_length = pack_u64_le(pack_bytes, 24);
    if pack_length != pack_bytes.len() as u64 {
        return Err(js_stack_error(&[
            "decoder stack weight pack declared length drifted",
        ]));
    }
    let manifest_end = pack_usize(
        u64::try_from(PACK_HEADER_BYTES)
            .ok()
            .and_then(|header| header.checked_add(manifest_length))
            .ok_or_else(|| {
                js_stack_error(&["decoder stack weight pack manifest length overflowed"])
            })?,
        "manifest",
    )?;
    let directory_end = pack_usize(
        u64::try_from(manifest_end)
            .ok()
            .and_then(|offset| offset.checked_add(directory_length))
            .ok_or_else(|| {
                js_stack_error(&["decoder stack weight pack directory length overflowed"])
            })?,
        "directory",
    )?;
    if directory_end > pack_bytes.len() {
        return Err(js_stack_error(&[
            "decoder stack weight pack prefix exceeds the file",
        ]));
    }
    let weight_storage = validate_pack_manifest(pack_slice(
        pack_bytes,
        PACK_HEADER_BYTES,
        manifest_end,
        "manifest",
    )?)?;

    let mut expected_ids: Vec<&str> = Vec::new();
    expected_ids.push(PACK_DESCRIPTOR_SECTION_ID);
    expected_ids.extend_from_slice(shard_ids);
    let mut entries: Vec<PackDirectoryEntry> = Vec::new();
    let mut cursor = manifest_end;
    for (index, expected_id) in expected_ids.iter().enumerate() {
        let fixed_end = cursor + PACK_DIRECTORY_FIXED_BYTES;
        if fixed_end > directory_end {
            return Err(js_stack_error(&[
                "decoder stack weight pack directory fixed entry is truncated",
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
            return Err(js_stack_error(&[
                "decoder stack weight pack directory reserved byte is nonzero",
            ]));
        }
        let expected_kind = if index == 0 { 1 } else { 2 };
        if kind != expected_kind {
            return Err(js_stack_error(&[
                "decoder stack weight pack section ",
                expected_id,
                " kind drifted",
            ]));
        }
        if alignment == 0 || alignment > PACK_MAX_ALIGNMENT || !alignment.is_power_of_two() {
            return Err(js_stack_error(&[
                "decoder stack weight pack section ",
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
            return Err(js_stack_error(&[
                "decoder stack weight pack directory entry is truncated",
            ]));
        }
        let id = std::str::from_utf8(pack_slice(
            pack_bytes,
            fixed_end,
            id_end,
            "directory entry id",
        )?)
        .map_err(|error| {
            js_stack_error(&[
                "decoder stack weight pack directory entry id UTF-8 is invalid: ",
                &error.to_string(),
            ])
        })?;
        if id != *expected_id {
            return Err(js_stack_error(&[
                "decoder stack weight pack section order drifted at ",
                expected_id,
            ]));
        }
        if !pack_padding_is_zero(pack_slice(
            pack_bytes,
            id_end,
            entry_end,
            "directory padding",
        )?) {
            return Err(js_stack_error(&[
                "decoder stack weight pack directory entry ",
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
        return Err(js_stack_error(&[
            "decoder stack weight pack directory length drifted",
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
            js_stack_error(&["decoder stack weight pack section length overflowed"])
        })?;
        if offset != expected_offset || section_end > pack_bytes.len() {
            return Err(js_stack_error(&[
                "decoder stack weight pack section ",
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
            return Err(js_stack_error(&[
                "decoder stack weight pack section ",
                expected_ids[index],
                " leading alignment contains nonzero padding",
            ]));
        }
        let payload = pack_slice(pack_bytes, offset, section_end, "section payload")?;
        if blake3::hash(payload).as_bytes() != &entry.digest {
            return Err(js_stack_error(&[
                "decoder stack weight pack section ",
                expected_ids[index],
                " BLAKE3 digest mismatch",
            ]));
        }
        payloads.push(payload);
        previous_end = section_end;
    }
    if previous_end != pack_bytes.len() {
        return Err(js_stack_error(&[
            "decoder stack weight pack has trailing or missing section bytes",
        ]));
    }

    let ValidatedPackDescriptor {
        shards,
        checkpoint_blake3,
    } = validate_pack_descriptor(
        payloads[0],
        expected_prefix_tokens,
        expected_cache_capacity,
        prefill_capable,
        shard_ids,
        weight_storage,
    )?;
    let mut shard_payloads: Vec<&[u8]> = Vec::new();
    for (shard_index, shard_id) in shard_ids.iter().enumerate() {
        let entry = &entries[shard_index + 1];
        let pin = shards
            .get(*shard_id)
            .and_then(Value::as_object)
            .ok_or_else(|| {
                js_stack_error(&[
                    "decoder stack weight pack shard ",
                    shard_id,
                    " pin must be an object",
                ])
            })?;
        let rope_table = shard_index == 5 || shard_index == 6;
        match weight_storage {
            DecoderWeightStorage::F32 => {
                require_pack_exact_keys(pin, &["bytes", "blake3"], "shard pin")?;
            }
            DecoderWeightStorage::F16 => {
                require_pack_exact_keys(pin, &["bytes", "blake3", "dtype"], "shard pin")?;
                let expected_dtype = if rope_table { "f32" } else { "f16" };
                if pack_json_str(pin, "dtype", "shard pin")? != expected_dtype {
                    return Err(js_stack_error(&[
                        "decoder stack weight pack shard ",
                        shard_id,
                        " dtype drifted",
                    ]));
                }
            }
        }
        let expected_bytes =
            expected_shard_bytes(shard_index, expected_cache_capacity, weight_storage)?;
        if pack_json_u64(pin, "bytes", "shard pin")? != expected_bytes
            || entry.byte_length != expected_bytes
        {
            return Err(js_stack_error(&[
                "decoder stack weight pack shard ",
                shard_id,
                " declared length drifted",
            ]));
        }
        let shard_storage = if rope_table {
            DecoderWeightStorage::F32
        } else {
            weight_storage
        };
        if !entry
            .byte_length
            .is_multiple_of(u64::from(shard_storage.bytes_per_element()))
        {
            return Err(js_stack_error(&[
                "decoder stack weight pack shard ",
                shard_id,
                " is not aligned to its authenticated storage element",
            ]));
        }
        let pinned_digest = pack_json_str(pin, "blake3", "shard pin")?;
        if pack_hex_digest(pinned_digest, "shard pin blake3")? != entry.digest {
            return Err(js_stack_error(&[
                "decoder stack weight pack shard ",
                shard_id,
                " BLAKE3 pin drifted",
            ]));
        }
        let payload = payloads[shard_index + 1];
        match shard_storage {
            DecoderWeightStorage::F32 => require_pack_f32_finite(payload, shard_id)?,
            DecoderWeightStorage::F16 => require_pack_f16_finite(payload, shard_id)?,
        }
        shard_payloads.push(payload);
    }
    let logits_capable = shard_ids.len() == PACK_SHARD_IDS.len();
    Ok(ParsedWeightPack {
        weight_storage,
        checkpoint_blake3,
        norm1_weight: shard_payloads[0].to_vec(),
        q_weight: shard_payloads[1].to_vec(),
        k_weight: shard_payloads[2].to_vec(),
        v_weight: shard_payloads[3].to_vec(),
        o_weight: shard_payloads[4].to_vec(),
        rope_cos: shard_payloads[5].to_vec(),
        rope_sin: shard_payloads[6].to_vec(),
        norm2_weight: shard_payloads[7].to_vec(),
        gate_weight: shard_payloads[8].to_vec(),
        up_weight: shard_payloads[9].to_vec(),
        down_weight: shard_payloads[10].to_vec(),
        final_norm_weight: if logits_capable {
            Some(shard_payloads[LEGACY_PACK_SHARD_COUNT].to_vec())
        } else {
            None
        },
        lm_head_weight: if logits_capable {
            Some(shard_payloads[LEGACY_PACK_SHARD_COUNT + 1].to_vec())
        } else {
            None
        },
    })
}

fn require_stack_cache_operands(
    bytes: &[u8],
    label: &str,
    prefix_tokens: u32,
    cache_capacity: u32,
) -> Result<Vec<f32>, JsValue> {
    let key_value_width = u64::from(PINNED_STACK_KEY_VALUE_WIDTH);
    let layers = u64::from(pvlc_runtime_core::PINNED_DECODER_LAYERS);
    let expected_bytes = layers
        .checked_mul(u64::from(cache_capacity))
        .and_then(|value| value.checked_mul(key_value_width))
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| js_stack_error(&[label, " byte size overflowed"]))?;
    if bytes.len() as u64 != expected_bytes {
        return Err(js_stack_error(&[
            label,
            " byte length does not match the exact layer-major bulk cache geometry",
        ]));
    }
    let values = stack_bytes_to_f32(bytes, label)?;
    let plane_elements = u64::from(cache_capacity)
        .checked_mul(key_value_width)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| js_stack_error(&[label, " plane element count overflowed"]))?;
    let semantic_elements = u64::from(prefix_tokens)
        .checked_mul(key_value_width)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| js_stack_error(&[label, " semantic element count overflowed"]))?;
    for plane in 0..pvlc_runtime_core::PINNED_DECODER_LAYERS as usize {
        let base = plane
            .checked_mul(plane_elements)
            .ok_or_else(|| js_stack_error(&[label, " plane offset overflowed"]))?;
        for element in &values[base..base + semantic_elements] {
            if !element.is_finite() {
                return Err(js_stack_error(&[
                    label,
                    " contains a nonfinite f32 cache element",
                ]));
            }
        }
    }
    Ok(values)
}

impl BrowserDecoderStackSession {
    fn create(
        device: &JsValue,
        kv_plan: pvlc_runtime_core::DecoderKvSessionPlan,
        stack_plan: pvlc_runtime_core::DecoderStackPlan,
        weight_resource_plan: pvlc_runtime_core::DecoderWeightResourcePlan,
        checkpoint_blake3: Option<[u8; 32]>,
        prefill_plan: pvlc_runtime_core::DecoderStackPrefillPlan,
        sources: &StackShaderSources,
        prefill_capable: bool,
        lm_head_plan: Option<pvlc_runtime_core::DecoderLmHeadPlan>,
        resident_weights: Option<&BrowserDecoderStackResidentWeights>,
    ) -> Result<
        (
            BrowserDecoderStackSession,
            Option<BrowserDecoderStackResidentWeights>,
        ),
        JsValue,
    > {
        let storage_copy_dst = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let copy_src_usage = storage_copy_dst | wgpu::BufferUsages::COPY_SRC;
        let readback_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let hidden_bytes = stack_plan.hidden_stride_bytes;
        let pingpong_bytes = hidden_bytes.checked_mul(2).ok_or_else(|| {
            js_stack_error(&["decoder stack hidden ping-pong byte size overflowed"])
        })?;
        let query_bytes = checked_u64_bytes(
            stack_plan.layer_plan.attention_block.query_width,
            "decoder query row",
        )?;
        let key_value_bytes = checked_u64_bytes(
            stack_plan.layer_plan.attention_block.key_value_width,
            "decoder key/value row",
        )?;
        let intermediate_bytes = checked_u64_bytes(
            stack_plan.layer_plan.intermediate_size as usize,
            "decoder intermediate row",
        )?;
        let rope_bytes = weight_resource_plan.rope_table_bytes;
        let [
            norm1_weight_bytes,
            q_weight_bytes,
            k_weight_bytes,
            v_weight_bytes,
            o_weight_bytes,
            norm2_weight_bytes,
            gate_weight_bytes,
            up_weight_bytes,
            down_weight_bytes,
        ] = weight_resource_plan.layer_weight_bulk_bytes;
        let cache_bytes = u64::from(stack_plan.layers)
            .checked_mul(stack_plan.cache_stride_bytes)
            .ok_or_else(|| js_stack_error(&["decoder stack compact cache byte size overflowed"]))?;
        let prefill_capacity = u64::from(kv_plan.cache_capacity);
        let prefill_bytes = |width: u64, label: &str| {
            prefill_capacity
                .checked_mul(width)
                .and_then(|value| value.checked_mul(4))
                .ok_or_else(|| js_stack_error(&[label, " byte size overflowed"]))
        };
        let prefill_hidden_bytes =
            prefill_bytes(PREFILL_HIDDEN_WIDTH, "decoder stack prefill hidden stage")?;
        let prefill_query_bytes =
            prefill_bytes(PREFILL_QUERY_WIDTH, "decoder stack prefill query stage")?;
        let prefill_key_value_bytes = prefill_bytes(
            PREFILL_KEY_VALUE_WIDTH,
            "decoder stack prefill key/value stage",
        )?;
        let prefill_intermediate_bytes = prefill_bytes(
            PREFILL_INTERMEDIATE_WIDTH,
            "decoder stack prefill intermediate stage",
        )?;
        let hidden_pingpong_buffer = create_stack_buffer(
            device,
            BUFFER_HIDDEN_PINGPONG,
            pingpong_bytes,
            buffer_usage(&[copy_src_usage]),
        )?;
        let resident_weight_bytes = resident_weight_bytes(&weight_resource_plan)?;
        let norm1_weight_buffer = match resident_weights {
            Some(weights) => weights.norm1.clone(),
            None => create_stack_buffer(
                device,
                BUFFER_NORM1_WEIGHT,
                norm1_weight_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?,
        };
        let q_weight_buffer = match resident_weights {
            Some(weights) => weights.q.clone(),
            None => create_stack_buffer(
                device,
                BUFFER_Q_WEIGHT,
                q_weight_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?,
        };
        let k_weight_buffer = match resident_weights {
            Some(weights) => weights.k.clone(),
            None => create_stack_buffer(
                device,
                BUFFER_K_WEIGHT,
                k_weight_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?,
        };
        let v_weight_buffer = match resident_weights {
            Some(weights) => weights.v.clone(),
            None => create_stack_buffer(
                device,
                BUFFER_V_WEIGHT,
                v_weight_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?,
        };
        let o_weight_buffer = match resident_weights {
            Some(weights) => weights.o.clone(),
            None => create_stack_buffer(
                device,
                BUFFER_O_WEIGHT,
                o_weight_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?,
        };
        let rope_cos_buffer = create_stack_buffer(
            device,
            BUFFER_ROPE_COS,
            rope_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let rope_sin_buffer = create_stack_buffer(
            device,
            BUFFER_ROPE_SIN,
            rope_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let norm2_weight_buffer = match resident_weights {
            Some(weights) => weights.norm2.clone(),
            None => create_stack_buffer(
                device,
                BUFFER_NORM2_WEIGHT,
                norm2_weight_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?,
        };
        let gate_weight_buffer = match resident_weights {
            Some(weights) => weights.gate.clone(),
            None => create_stack_buffer(
                device,
                BUFFER_GATE_WEIGHT,
                gate_weight_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?,
        };
        let up_weight_buffer = match resident_weights {
            Some(weights) => weights.up.clone(),
            None => create_stack_buffer(
                device,
                BUFFER_UP_WEIGHT,
                up_weight_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?,
        };
        let down_weight_buffer = match resident_weights {
            Some(weights) => weights.down.clone(),
            None => create_stack_buffer(
                device,
                BUFFER_DOWN_WEIGHT,
                down_weight_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?,
        };
        let key_cache_buffer = create_stack_buffer(
            device,
            BUFFER_KEY_CACHE,
            cache_bytes,
            buffer_usage(&[copy_src_usage]),
        )?;
        let value_cache_buffer = create_stack_buffer(
            device,
            BUFFER_VALUE_CACHE,
            cache_bytes,
            buffer_usage(&[copy_src_usage]),
        )?;
        let norm1_buffer = create_stack_buffer(
            device,
            BUFFER_NORM1,
            hidden_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let q_projection_buffer = create_stack_buffer(
            device,
            BUFFER_Q_PROJECTION,
            query_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let k_projection_buffer = create_stack_buffer(
            device,
            BUFFER_K_PROJECTION,
            key_value_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let v_projection_buffer = create_stack_buffer(
            device,
            BUFFER_V_PROJECTION,
            key_value_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let mrope_query_buffer = create_stack_buffer(
            device,
            BUFFER_MROPE_QUERY,
            query_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let mrope_key_buffer = create_stack_buffer(
            device,
            BUFFER_MROPE_KEY,
            key_value_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let attention_output_buffer = create_stack_buffer(
            device,
            BUFFER_ATTENTION_OUTPUT,
            kv_plan.attention_bytes,
            buffer_usage(&[copy_src_usage]),
        )?;
        let o_projection_buffer = create_stack_buffer(
            device,
            BUFFER_O_PROJECTION,
            hidden_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let attention_residual_buffer = create_stack_buffer(
            device,
            BUFFER_ATTENTION_RESIDUAL,
            hidden_bytes,
            buffer_usage(&[copy_src_usage]),
        )?;
        let norm2_buffer = create_stack_buffer(
            device,
            BUFFER_NORM2,
            hidden_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let gate_buffer = create_stack_buffer(
            device,
            BUFFER_GATE,
            intermediate_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let up_buffer = create_stack_buffer(
            device,
            BUFFER_UP,
            intermediate_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let activation_buffer = create_stack_buffer(
            device,
            BUFFER_ACTIVATION,
            intermediate_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let down_projection_buffer = create_stack_buffer(
            device,
            BUFFER_DOWN_PROJECTION,
            hidden_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let stack_readback_buffer = create_stack_buffer(
            device,
            BUFFER_STACK_READBACK,
            hidden_bytes,
            buffer_usage(&[readback_usage]),
        )?;
        let rms_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_RMS_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let gemv_q_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_GEMV_Q_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let gemv_k_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_GEMV_K_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let gemv_v_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_GEMV_V_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let mrope_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_MROPE_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let append_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_APPEND_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let attention_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_ATTENTION_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let residual_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_RESIDUAL_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let rms2_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_RMS2_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let gemv_gate_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_GEMV_GATE_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let gemv_up_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_GEMV_UP_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let swiglu_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_SWIGLU_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let gemv_down_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_GEMV_DOWN_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let residual2_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_RESIDUAL2_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let gemv_o_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_GEMV_O_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let split_partials_buffer = create_stack_buffer(
            device,
            BUFFER_SPLIT_PARTIALS,
            kv_plan.split_partials_bytes,
            buffer_usage(&[storage_copy_dst]),
        )?;
        let split_partial_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_SPLIT_PARTIAL_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let split_merge_uniform_buffer = create_stack_buffer(
            device,
            BUFFER_SPLIT_MERGE_UNIFORM,
            UNIFORM_BUFFER_BYTES,
            buffer_usage(&[uniform_usage]),
        )?;
        let layouts = StackBindGroupLayouts {
            rms: create_stack_bind_group_layout(device, &RMS_LAYOUT_ENTRIES)?,
            gemv: create_stack_bind_group_layout(device, &GEMV_LAYOUT_ENTRIES)?,
            mrope: create_stack_bind_group_layout(device, &MROPE_LAYOUT_ENTRIES)?,
            append: create_stack_bind_group_layout(device, &APPEND_LAYOUT_ENTRIES)?,
            attention: create_stack_bind_group_layout(device, &ATTENTION_LAYOUT_ENTRIES)?,
            swiglu: create_stack_bind_group_layout(device, &SWIGLU_LAYOUT_ENTRIES)?,
            residual: create_stack_bind_group_layout(device, &RESIDUAL_LAYOUT_ENTRIES)?,
        };
        let rms_norm_pipeline = create_stack_pipeline(
            device,
            RMS_NORM_KERNEL_NAME,
            &sources.rms_norm,
            &layouts.rms,
        )?;
        let gemv_tiled_pipeline = create_stack_pipeline(
            device,
            GEMV_TILED_KERNEL_NAME,
            &sources.gemv_tiled,
            &layouts.gemv,
        )?;
        let (rms_norm_f16_pipeline, gemv_tiled_f16_pipeline) =
            if weight_resource_plan.storage == DecoderWeightStorage::F16 {
                (
                    Some(create_stack_pipeline(
                        device,
                        RMS_NORM_F16_KERNEL_NAME,
                        &sources.rms_norm_f16_weights,
                        &layouts.rms,
                    )?),
                    Some(create_stack_pipeline(
                        device,
                        GEMV_TILED_F16_KERNEL_NAME,
                        &sources.gemv_tiled_f16_weights,
                        &layouts.gemv,
                    )?),
                )
            } else {
                (None, None)
            };
        let mrope_pipeline =
            create_stack_pipeline(device, MROPE_KERNEL_NAME, &sources.mrope, &layouts.mrope)?;
        let append_pipeline =
            create_stack_pipeline(device, APPEND_KERNEL_NAME, &sources.append, &layouts.append)?;
        let swiglu_pipeline =
            create_stack_pipeline(device, SWIGLU_KERNEL_NAME, &sources.swiglu, &layouts.swiglu)?;
        let residual_pipeline = create_stack_pipeline(
            device,
            RESIDUAL_KERNEL_NAME,
            &sources.residual,
            &layouts.residual,
        )?;
        let split_partial_pipeline = create_stack_pipeline(
            device,
            SPLIT_PARTIAL_KERNEL_NAME,
            &sources.split_partial,
            &layouts.attention,
        )?;
        let split_merge_pipeline = create_stack_pipeline(
            device,
            SPLIT_MERGE_KERNEL_NAME,
            &sources.split_merge,
            &layouts.attention,
        )?;
        let mut prefill_hidden_storage_buffer = None;
        let mut prefill_norm1_buffer = None;
        let mut prefill_query_buffer = None;
        let mut prefill_key_buffer = None;
        let mut prefill_value_buffer = None;
        let mut prefill_context_buffer = None;
        let mut prefill_output_buffer = None;
        let mut prefill_norm2_buffer = None;
        let mut prefill_gate_buffer = None;
        let mut prefill_up_buffer = None;
        let mut prefill_activation_buffer = None;
        let mut prefill_zero_bias_buffer = None;
        let mut prefill_projection_pipeline = None;
        let mut prefill_projection_f16_pipeline = None;
        let mut prefill_mrope_pipeline = None;
        let mut kv_append_range_pipeline = None;
        let mut prefill_gqa_pipeline = None;
        let mut projection_layout = None;
        if prefill_capable {
            let hidden_storage = create_stack_buffer(
                device,
                BUFFER_PREFILL_HIDDEN_STORAGE,
                prefill_hidden_bytes,
                buffer_usage(&[copy_src_usage]),
            )?;
            let norm1 = create_stack_buffer(
                device,
                BUFFER_PREFILL_NORM1,
                prefill_hidden_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?;
            let query = create_stack_buffer(
                device,
                BUFFER_PREFILL_QUERY,
                prefill_query_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?;
            let key = create_stack_buffer(
                device,
                BUFFER_PREFILL_KEY,
                prefill_key_value_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?;
            let value = create_stack_buffer(
                device,
                BUFFER_PREFILL_VALUE,
                prefill_key_value_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?;
            let context = create_stack_buffer(
                device,
                BUFFER_PREFILL_CONTEXT,
                prefill_query_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?;
            let output = create_stack_buffer(
                device,
                BUFFER_PREFILL_OUTPUT,
                prefill_hidden_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?;
            let norm2 = create_stack_buffer(
                device,
                BUFFER_PREFILL_NORM2,
                prefill_hidden_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?;
            let gate = create_stack_buffer(
                device,
                BUFFER_PREFILL_GATE,
                prefill_intermediate_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?;
            let up = create_stack_buffer(
                device,
                BUFFER_PREFILL_UP,
                prefill_intermediate_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?;
            let activation = create_stack_buffer(
                device,
                BUFFER_PREFILL_ACTIVATION,
                prefill_intermediate_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?;
            let zero_bias = create_stack_buffer(
                device,
                BUFFER_PREFILL_ZERO_BIAS,
                PREFILL_ZERO_BIAS_BYTES,
                buffer_usage(&[storage_copy_dst]),
            )?;
            let layout = create_stack_bind_group_layout(device, &PROJECTION_LAYOUT_ENTRIES)?;
            let projection_pipeline = create_stack_pipeline(
                device,
                PROJECTION_KERNEL_NAME,
                &sources.projection,
                &layout,
            )?;
            if weight_resource_plan.storage == DecoderWeightStorage::F16 {
                prefill_projection_f16_pipeline = Some(create_stack_pipeline(
                    device,
                    PREFILL_PROJECTION_F16_KERNEL_NAME,
                    &sources.linear_projection_f16_weights,
                    &layout,
                )?);
            }
            let mrope_pipeline = create_stack_pipeline(
                device,
                PREFILL_MROPE_KERNEL_NAME,
                &sources.prefill_mrope,
                &layouts.mrope,
            )?;
            let append_range_pipeline = create_stack_pipeline(
                device,
                KV_APPEND_RANGE_KERNEL_NAME,
                &sources.kv_append_range,
                &layouts.append,
            )?;
            let gqa_pipeline = create_stack_pipeline(
                device,
                PREFILL_GQA_KERNEL_NAME,
                &sources.prefill_gqa,
                &layouts.attention,
            )?;
            prefill_hidden_storage_buffer = Some(hidden_storage);
            prefill_norm1_buffer = Some(norm1);
            prefill_query_buffer = Some(query);
            prefill_key_buffer = Some(key);
            prefill_value_buffer = Some(value);
            prefill_context_buffer = Some(context);
            prefill_output_buffer = Some(output);
            prefill_norm2_buffer = Some(norm2);
            prefill_gate_buffer = Some(gate);
            prefill_up_buffer = Some(up);
            prefill_activation_buffer = Some(activation);
            prefill_zero_bias_buffer = Some(zero_bias);
            prefill_projection_pipeline = Some(projection_pipeline);
            prefill_mrope_pipeline = Some(mrope_pipeline);
            kv_append_range_pipeline = Some(append_range_pipeline);
            prefill_gqa_pipeline = Some(gqa_pipeline);
            projection_layout = Some(layout);
        }
        let mut final_norm_weight_buffer = None;
        let mut lm_head_weight_buffer = None;
        let mut normed_row_buffer = None;
        let mut logits_buffer = None;
        let mut logits_readback_buffer = None;
        let mut logits_rms_uniform_buffer = None;
        let mut logits_gemv_uniform_buffer = None;
        if let Some(plan) = lm_head_plan.as_ref() {
            let final_norm_weight_bytes =
                weight_resource_plan
                    .final_norm_weight_bytes
                    .ok_or_else(|| {
                        js_stack_error(&[
                            "decoder stack logits final-norm resource range is missing",
                        ])
                    })?;
            let lm_head_weight_bytes =
                weight_resource_plan.lm_head_weight_bytes.ok_or_else(|| {
                    js_stack_error(&["decoder stack logits LM-head resource range is missing"])
                })?;
            // The logits persistent resources, created once at begin and
            // reused by every later logits call: the two shared weights, the
            // normed-row intermediate, the logits storage and readback, and
            // the two static stage uniforms (written at creation, never
            // queue-written).
            final_norm_weight_buffer = Some(match resident_weights {
                Some(weights) => weights.final_norm.clone().ok_or_else(|| {
                    js_stack_error(&["decoder resident weights are missing the final-norm buffer"])
                })?,
                None => create_stack_buffer(
                    device,
                    BUFFER_FINAL_NORM_WEIGHT,
                    final_norm_weight_bytes,
                    buffer_usage(&[storage_copy_dst]),
                )?,
            });
            lm_head_weight_buffer = Some(match resident_weights {
                Some(weights) => weights.lm_head.clone().ok_or_else(|| {
                    js_stack_error(&["decoder resident weights are missing the LM-head buffer"])
                })?,
                None => create_stack_buffer(
                    device,
                    BUFFER_LM_HEAD,
                    lm_head_weight_bytes,
                    buffer_usage(&[storage_copy_dst]),
                )?,
            });
            normed_row_buffer = Some(create_stack_buffer(
                device,
                BUFFER_NORMED_ROW,
                plan.normed_row_bytes,
                buffer_usage(&[storage_copy_dst]),
            )?);
            logits_buffer = Some(create_stack_buffer(
                device,
                BUFFER_LOGITS,
                plan.logits_bytes,
                buffer_usage(&[copy_src_usage]),
            )?);
            logits_readback_buffer = Some(create_stack_buffer(
                device,
                BUFFER_LOGITS_READBACK,
                plan.logits_bytes,
                buffer_usage(&[readback_usage]),
            )?);
            logits_rms_uniform_buffer = Some(create_stack_buffer_initialized(
                device,
                BUFFER_LOGITS_RMS_UNIFORM,
                UNIFORM_BUFFER_BYTES,
                buffer_usage(&[uniform_usage]),
                bytemuck::cast_slice(&plan.stage_uniform_words[0]),
            )?);
            logits_gemv_uniform_buffer = Some(create_stack_buffer_initialized(
                device,
                BUFFER_LOGITS_GEMV_UNIFORM,
                UNIFORM_BUFFER_BYTES,
                buffer_usage(&[uniform_usage]),
                bytemuck::cast_slice(&plan.stage_uniform_words[1]),
            )?);
        }
        let mut session = BrowserDecoderStackSession {
            kv_plan,
            stack_plan,
            weight_resource_plan,
            checkpoint_blake3,
            resident_weight_bytes,
            cache_tokens: if prefill_capable {
                0
            } else {
                kv_plan.initial_cache_tokens
            },
            poisoned: false,
            ready: false,
            rms_norm_shader_blake3: source_blake3(&sources.rms_norm),
            gemv_shader_blake3: source_blake3(&sources.gemv),
            gemv_tiled_shader_blake3: source_blake3(&sources.gemv_tiled),
            mrope_shader_blake3: source_blake3(&sources.mrope),
            append_shader_blake3: source_blake3(&sources.append),
            attention_shader_blake3: source_blake3(&sources.attention),
            swiglu_shader_blake3: source_blake3(&sources.swiglu),
            residual_shader_blake3: source_blake3(&sources.residual),
            hidden_pingpong_buffer,
            norm1_weight_buffer,
            q_weight_buffer,
            k_weight_buffer,
            v_weight_buffer,
            o_weight_buffer,
            rope_cos_buffer,
            rope_sin_buffer,
            norm2_weight_buffer,
            gate_weight_buffer,
            up_weight_buffer,
            down_weight_buffer,
            key_cache_buffer,
            value_cache_buffer,
            norm1_buffer,
            q_projection_buffer,
            k_projection_buffer,
            v_projection_buffer,
            mrope_query_buffer,
            mrope_key_buffer,
            attention_output_buffer,
            o_projection_buffer,
            attention_residual_buffer,
            norm2_buffer,
            gate_buffer,
            up_buffer,
            activation_buffer,
            down_projection_buffer,
            stack_readback_buffer,
            rms_uniform_buffer,
            gemv_q_uniform_buffer,
            gemv_k_uniform_buffer,
            gemv_v_uniform_buffer,
            mrope_uniform_buffer,
            append_uniform_buffer,
            attention_uniform_buffer,
            residual_uniform_buffer,
            rms2_uniform_buffer,
            gemv_gate_uniform_buffer,
            gemv_up_uniform_buffer,
            swiglu_uniform_buffer,
            gemv_down_uniform_buffer,
            residual2_uniform_buffer,
            gemv_o_uniform_buffer,
            rms_norm_pipeline,
            gemv_tiled_pipeline,
            rms_norm_f16_pipeline,
            gemv_tiled_f16_pipeline,
            mrope_pipeline,
            append_pipeline,
            swiglu_pipeline,
            residual_pipeline,
            rms_bind_group: js_sys::Object::default(),
            gemv_q_bind_group: js_sys::Object::default(),
            gemv_k_bind_group: js_sys::Object::default(),
            gemv_v_bind_group: js_sys::Object::default(),
            mrope_bind_group: js_sys::Object::default(),
            append_bind_group: js_sys::Object::default(),
            gemv_o_bind_group: js_sys::Object::default(),
            residual_bind_group: js_sys::Object::default(),
            rms2_bind_group: js_sys::Object::default(),
            gemv_gate_bind_group: js_sys::Object::default(),
            gemv_up_bind_group: js_sys::Object::default(),
            swiglu_bind_group: js_sys::Object::default(),
            gemv_down_bind_group: js_sys::Object::default(),
            residual2_bind_group: js_sys::Object::default(),
            prefill_plan,
            prefill_projection_shader_blake3: if prefill_capable {
                Some(source_blake3(&sources.projection))
            } else {
                None
            },
            prefill_mrope_shader_blake3: if prefill_capable {
                Some(source_blake3(&sources.prefill_mrope))
            } else {
                None
            },
            kv_append_range_shader_blake3: if prefill_capable {
                Some(source_blake3(&sources.kv_append_range))
            } else {
                None
            },
            prefill_gqa_shader_blake3: if prefill_capable {
                Some(source_blake3(&sources.prefill_gqa))
            } else {
                None
            },
            prefill_hidden_storage_buffer,
            prefill_norm1_buffer,
            prefill_query_buffer,
            prefill_key_buffer,
            prefill_value_buffer,
            prefill_context_buffer,
            prefill_output_buffer,
            prefill_norm2_buffer,
            prefill_gate_buffer,
            prefill_up_buffer,
            prefill_activation_buffer,
            prefill_zero_bias_buffer,
            prefill_projection_pipeline,
            prefill_projection_f16_pipeline,
            prefill_mrope_pipeline,
            kv_append_range_pipeline,
            prefill_gqa_pipeline,
            prefill_rms1_bind_group: None,
            prefill_query_bind_group: None,
            prefill_key_bind_group: None,
            prefill_value_bind_group: None,
            prefill_mrope_bind_group: None,
            prefill_kv_append_range_bind_group: None,
            prefill_gqa_bind_group: None,
            prefill_output_bind_group: None,
            prefill_residual_bind_group: None,
            prefill_rms2_bind_group: None,
            prefill_gate_bind_group: None,
            prefill_up_bind_group: None,
            prefill_swiglu_bind_group: None,
            prefill_down_bind_group: None,
            prefill_residual2_bind_group: None,
            lm_head_plan,
            final_norm_weight_buffer,
            lm_head_weight_buffer,
            normed_row_buffer,
            logits_buffer,
            logits_readback_buffer,
            top1_result_buffer: None,
            top1_readback_buffer: None,
            top1_pipeline: None,
            top1_bind_group: None,
            top1_shader_blake3: None,
            logits_rms_uniform_buffer,
            logits_gemv_uniform_buffer,
            prefill_logits_rms_bind_group: None,
            step_logits_rms_bind_group: None,
            gemv_logits_bind_group: None,
            split_partials_buffer,
            split_partial_uniform_buffer,
            split_merge_uniform_buffer,
            split_partial_pipeline,
            split_merge_pipeline,
            split_partial_bind_group: js_sys::Object::default(),
            split_merge_bind_group: js_sys::Object::default(),
            split_partial_shader_blake3: source_blake3(&sources.split_partial),
            split_merge_shader_blake3: source_blake3(&sources.split_merge),
        };
        session.create_bind_groups(device, &layouts)?;
        if let Some(layout) = projection_layout.as_ref() {
            session.create_prefill_bind_groups(device, &layouts, layout)?;
        }
        if session.lm_head_plan.is_some() {
            session.create_logits_bind_groups(device, &layouts)?;
        }
        let resident_candidate = if resident_weights.is_none() {
            match (
                resident_weight_key(session.checkpoint_blake3, &session.weight_resource_plan),
                session.checkpoint_blake3,
            ) {
                (Some(key), Some(checkpoint_blake3)) => Some(BrowserDecoderStackResidentWeights {
                    key,
                    checkpoint_blake3,
                    resident_bytes: session.resident_weight_bytes,
                    norm1: session.norm1_weight_buffer.clone(),
                    q: session.q_weight_buffer.clone(),
                    k: session.k_weight_buffer.clone(),
                    v: session.v_weight_buffer.clone(),
                    o: session.o_weight_buffer.clone(),
                    norm2: session.norm2_weight_buffer.clone(),
                    gate: session.gate_weight_buffer.clone(),
                    up: session.up_weight_buffer.clone(),
                    down: session.down_weight_buffer.clone(),
                    final_norm: session.final_norm_weight_buffer.clone(),
                    lm_head: session.lm_head_weight_buffer.clone(),
                }),
                _ => None,
            }
        } else {
            None
        };
        Ok((session, resident_candidate))
    }

    /// Creates the fifteen persistent bind groups from the exact session buffer,
    /// pipeline, and layout fields after the session owns its resources. Every
    /// dynamically offset entry binds its per-layer slice (one hidden stride,
    /// one weight stride, one cache stride) so `dynamicOffset + entrySize`
    /// never exceeds the buffer; static entries bind the whole buffer.
    fn create_bind_groups(
        &mut self,
        device: &JsValue,
        layouts: &StackBindGroupLayouts,
    ) -> Result<(), JsValue> {
        let hidden_bytes = self.stack_plan.hidden_stride_bytes;
        let query_bytes = checked_u64_bytes(
            self.stack_plan.layer_plan.attention_block.query_width,
            "decoder query row",
        )?;
        let key_value_bytes = checked_u64_bytes(
            self.stack_plan.layer_plan.attention_block.key_value_width,
            "decoder key/value row",
        )?;
        let intermediate_bytes = checked_u64_bytes(
            self.stack_plan.layer_plan.intermediate_size as usize,
            "decoder intermediate row",
        )?;
        let rope_bytes = self.weight_resource_plan.rope_table_bytes;
        let weight_strides = self.weight_resource_plan.layer_weight_stride_bytes;
        let cache_stride = self.stack_plan.cache_stride_bytes;
        self.rms_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-rms-bind-group",
            &layouts.rms,
            &[
                (&self.hidden_pingpong_buffer, hidden_bytes),
                (&self.norm1_weight_buffer, weight_strides[0]),
                (&self.norm1_buffer, hidden_bytes),
                (&self.rms_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.gemv_q_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-gemv-q-bind-group",
            &layouts.gemv,
            &[
                (&self.q_weight_buffer, weight_strides[1]),
                (&self.norm1_buffer, hidden_bytes),
                (&self.q_projection_buffer, query_bytes),
                (&self.gemv_q_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.gemv_k_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-gemv-k-bind-group",
            &layouts.gemv,
            &[
                (&self.k_weight_buffer, weight_strides[2]),
                (&self.norm1_buffer, hidden_bytes),
                (&self.k_projection_buffer, key_value_bytes),
                (&self.gemv_k_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.gemv_v_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-gemv-v-bind-group",
            &layouts.gemv,
            &[
                (&self.v_weight_buffer, weight_strides[3]),
                (&self.norm1_buffer, hidden_bytes),
                (&self.v_projection_buffer, key_value_bytes),
                (&self.gemv_v_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.mrope_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-mrope-bind-group",
            &layouts.mrope,
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
        self.append_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-append-bind-group",
            &layouts.append,
            &[
                (&self.mrope_key_buffer, key_value_bytes),
                (&self.v_projection_buffer, key_value_bytes),
                (&self.key_cache_buffer, cache_stride),
                (&self.value_cache_buffer, cache_stride),
                (&self.append_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.gemv_o_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-gemv-o-bind-group",
            &layouts.gemv,
            &[
                (&self.o_weight_buffer, weight_strides[4]),
                (&self.attention_output_buffer, self.kv_plan.attention_bytes),
                (&self.o_projection_buffer, hidden_bytes),
                (&self.gemv_o_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.residual_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-residual-bind-group",
            &layouts.residual,
            &[
                (&self.hidden_pingpong_buffer, hidden_bytes),
                (&self.o_projection_buffer, hidden_bytes),
                (&self.attention_residual_buffer, hidden_bytes),
                (&self.residual_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.rms2_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-rms2-bind-group",
            &layouts.rms,
            &[
                (&self.attention_residual_buffer, hidden_bytes),
                (&self.norm2_weight_buffer, weight_strides[5]),
                (&self.norm2_buffer, hidden_bytes),
                (&self.rms2_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.gemv_gate_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-gemv-gate-bind-group",
            &layouts.gemv,
            &[
                (&self.gate_weight_buffer, weight_strides[6]),
                (&self.norm2_buffer, hidden_bytes),
                (&self.gate_buffer, intermediate_bytes),
                (&self.gemv_gate_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.gemv_up_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-gemv-up-bind-group",
            &layouts.gemv,
            &[
                (&self.up_weight_buffer, weight_strides[7]),
                (&self.norm2_buffer, hidden_bytes),
                (&self.up_buffer, intermediate_bytes),
                (&self.gemv_up_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.swiglu_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-swiglu-bind-group",
            &layouts.swiglu,
            &[
                (&self.gate_buffer, intermediate_bytes),
                (&self.up_buffer, intermediate_bytes),
                (&self.activation_buffer, intermediate_bytes),
                (&self.swiglu_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.gemv_down_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-gemv-down-bind-group",
            &layouts.gemv,
            &[
                (&self.down_weight_buffer, weight_strides[8]),
                (&self.activation_buffer, intermediate_bytes),
                (&self.down_projection_buffer, hidden_bytes),
                (&self.gemv_down_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.residual2_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-residual2-bind-group",
            &layouts.residual,
            &[
                (&self.attention_residual_buffer, hidden_bytes),
                (&self.down_projection_buffer, hidden_bytes),
                (&self.hidden_pingpong_buffer, hidden_bytes),
                (&self.residual2_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        // The split-K decode attention groups reuse the accepted attention
        // layout (query read, per-layer cache slices, output, uniform): the
        // partial pass writes the scratch partials plane; the merge pass
        // reads it through the static read-only slot and writes the accepted
        // attention output.
        self.split_partial_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-split-partial-bind-group",
            &layouts.attention,
            &[
                (&self.mrope_query_buffer, query_bytes),
                (&self.key_cache_buffer, cache_stride),
                (&self.value_cache_buffer, cache_stride),
                (
                    &self.split_partials_buffer,
                    self.kv_plan.split_partials_bytes,
                ),
                (&self.split_partial_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        self.split_merge_bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-split-merge-bind-group",
            &layouts.attention,
            &[
                (
                    &self.split_partials_buffer,
                    self.kv_plan.split_partials_bytes,
                ),
                (&self.key_cache_buffer, cache_stride),
                (&self.value_cache_buffer, cache_stride),
                (&self.attention_output_buffer, self.kv_plan.attention_bytes),
                (&self.split_merge_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?;
        Ok(())
    }

    /// Creates the fifteen persistent prefill bind groups from the exact
    /// prefill session buffer fields after the session owns its resources,
    /// mirroring the accepted bind-group discipline: every dynamically offset
    /// entry binds its per-layer slice (one weight stride, one cache stride)
    /// and every multi-token stage entry binds its capacity-sized slice, so
    /// `dynamicOffset + entrySize` never exceeds the buffer.
    fn create_prefill_bind_groups(
        &mut self,
        device: &JsValue,
        layouts: &StackBindGroupLayouts,
        projection_layout: &js_sys::Object,
    ) -> Result<(), JsValue> {
        let capacity = u64::from(self.kv_plan.cache_capacity);
        let stage_bytes = |width: u64, label: &str| {
            capacity
                .checked_mul(width)
                .and_then(|value| value.checked_mul(4))
                .ok_or_else(|| js_stack_error(&[label, " byte size overflowed"]))
        };
        let hidden_bytes = stage_bytes(PREFILL_HIDDEN_WIDTH, "decoder stack prefill hidden stage")?;
        let query_bytes = stage_bytes(PREFILL_QUERY_WIDTH, "decoder stack prefill query stage")?;
        let key_value_bytes = stage_bytes(
            PREFILL_KEY_VALUE_WIDTH,
            "decoder stack prefill key/value stage",
        )?;
        let intermediate_bytes = stage_bytes(
            PREFILL_INTERMEDIATE_WIDTH,
            "decoder stack prefill intermediate stage",
        )?;
        let rope_bytes = self.weight_resource_plan.rope_table_bytes;
        let weight_strides = self.weight_resource_plan.layer_weight_stride_bytes;
        let cache_stride = self.stack_plan.cache_stride_bytes;
        let Some(hidden_storage) = self.prefill_hidden_storage_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(norm1) = self.prefill_norm1_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(query) = self.prefill_query_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(key) = self.prefill_key_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(value) = self.prefill_value_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(context) = self.prefill_context_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(output) = self.prefill_output_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(norm2) = self.prefill_norm2_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(gate) = self.prefill_gate_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(up) = self.prefill_up_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(activation) = self.prefill_activation_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(zero_bias) = self.prefill_zero_bias_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        self.prefill_rms1_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-rms1-bind-group",
            &layouts.rms,
            &[
                (hidden_storage, hidden_bytes),
                (&self.norm1_weight_buffer, weight_strides[0]),
                (norm1, hidden_bytes),
                (&self.rms_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.prefill_query_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-query-bind-group",
            projection_layout,
            &[
                (norm1, hidden_bytes),
                (&self.q_weight_buffer, weight_strides[1]),
                (zero_bias, PREFILL_ZERO_BIAS_BYTES),
                (context, query_bytes),
                (&self.gemv_q_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.prefill_key_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-key-bind-group",
            projection_layout,
            &[
                (norm1, hidden_bytes),
                (&self.k_weight_buffer, weight_strides[2]),
                (zero_bias, PREFILL_ZERO_BIAS_BYTES),
                (key, key_value_bytes),
                (&self.gemv_k_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.prefill_value_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-value-bind-group",
            projection_layout,
            &[
                (norm1, hidden_bytes),
                (&self.v_weight_buffer, weight_strides[3]),
                (zero_bias, PREFILL_ZERO_BIAS_BYTES),
                (value, key_value_bytes),
                (&self.gemv_v_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.prefill_mrope_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-mrope-bind-group",
            &layouts.mrope,
            &[
                (context, query_bytes),
                (key, key_value_bytes),
                (&self.rope_cos_buffer, rope_bytes),
                (&self.rope_sin_buffer, rope_bytes),
                (query, query_bytes),
                (norm1, key_value_bytes),
                (&self.mrope_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.prefill_kv_append_range_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-kv-append-range-bind-group",
            &layouts.append,
            &[
                (norm1, key_value_bytes),
                (value, key_value_bytes),
                (&self.key_cache_buffer, cache_stride),
                (&self.value_cache_buffer, cache_stride),
                (&self.append_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.prefill_gqa_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-gqa-bind-group",
            &layouts.attention,
            &[
                (query, query_bytes),
                (&self.key_cache_buffer, cache_stride),
                (&self.value_cache_buffer, cache_stride),
                (context, query_bytes),
                (&self.attention_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.prefill_output_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-output-bind-group",
            projection_layout,
            &[
                (context, query_bytes),
                (&self.o_weight_buffer, weight_strides[4]),
                (zero_bias, PREFILL_ZERO_BIAS_BYTES),
                (output, hidden_bytes),
                (&self.gemv_o_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.prefill_residual_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-residual-bind-group",
            &layouts.residual,
            &[
                (hidden_storage, hidden_bytes),
                (output, hidden_bytes),
                (norm1, hidden_bytes),
                (&self.residual_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.prefill_rms2_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-rms2-bind-group",
            &layouts.rms,
            &[
                (norm1, hidden_bytes),
                (&self.norm2_weight_buffer, weight_strides[5]),
                (norm2, hidden_bytes),
                (&self.rms2_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.prefill_gate_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-gate-bind-group",
            projection_layout,
            &[
                (norm2, hidden_bytes),
                (&self.gate_weight_buffer, weight_strides[6]),
                (zero_bias, PREFILL_ZERO_BIAS_BYTES),
                (gate, intermediate_bytes),
                (&self.gemv_gate_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.prefill_up_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-up-bind-group",
            projection_layout,
            &[
                (norm2, hidden_bytes),
                (&self.up_weight_buffer, weight_strides[7]),
                (zero_bias, PREFILL_ZERO_BIAS_BYTES),
                (up, intermediate_bytes),
                (&self.gemv_up_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.prefill_swiglu_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-swiglu-bind-group",
            &layouts.swiglu,
            &[
                (gate, intermediate_bytes),
                (up, intermediate_bytes),
                (activation, intermediate_bytes),
                (&self.swiglu_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.prefill_down_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-down-bind-group",
            projection_layout,
            &[
                (activation, intermediate_bytes),
                (&self.down_weight_buffer, weight_strides[8]),
                (zero_bias, PREFILL_ZERO_BIAS_BYTES),
                (output, hidden_bytes),
                (&self.gemv_down_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.prefill_residual2_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-residual2-bind-group",
            &layouts.residual,
            &[
                (norm1, hidden_bytes),
                (output, hidden_bytes),
                (hidden_storage, hidden_bytes),
                (&self.residual2_uniform_buffer, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        Ok(())
    }

    /// Creates the three persistent logits bind groups from the exact logits
    /// session buffer fields after the session owns its resources, mirroring
    /// the accepted bind-group discipline: the final-RMS input entry binds one
    /// hidden-row slice (the prefill hidden storage last row, or the hidden
    /// ping-pong slot 0) selected by a dynamic offset at dispatch; every
    /// static entry binds its whole buffer.
    fn create_logits_bind_groups(
        &mut self,
        device: &JsValue,
        layouts: &StackBindGroupLayouts,
    ) -> Result<(), JsValue> {
        let Some(plan) = self.lm_head_plan else {
            return Err(js_stack_error(&[
                "decoder stack session is not logits capable",
            ]));
        };
        let Some(final_norm) = self.final_norm_weight_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not logits capable",
            ]));
        };
        let Some(lm_head) = self.lm_head_weight_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not logits capable",
            ]));
        };
        let Some(normed_row) = self.normed_row_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not logits capable",
            ]));
        };
        let Some(logits) = self.logits_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not logits capable",
            ]));
        };
        let Some(rms_uniform) = self.logits_rms_uniform_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not logits capable",
            ]));
        };
        let Some(gemv_uniform) = self.logits_gemv_uniform_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not logits capable",
            ]));
        };
        let final_norm_weight_bytes = self
            .weight_resource_plan
            .final_norm_weight_bytes
            .ok_or_else(|| {
                js_stack_error(&["decoder stack session final-norm resource range is missing"])
            })?;
        let lm_head_weight_bytes =
            self.weight_resource_plan
                .lm_head_weight_bytes
                .ok_or_else(|| {
                    js_stack_error(&["decoder stack session LM-head resource range is missing"])
                })?;
        let hidden_stride = self.stack_plan.hidden_stride_bytes;
        // A logits-capable session without the prefill capability never
        // dispatches the prefill-source group; its input entry binds the
        // hidden ping-pong so the group still references an owned buffer.
        let prefill_input = self
            .prefill_hidden_storage_buffer
            .as_ref()
            .unwrap_or(&self.hidden_pingpong_buffer);
        self.prefill_logits_rms_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-prefill-logits-rms-bind-group",
            &layouts.rms,
            &[
                (prefill_input, hidden_stride),
                (final_norm, final_norm_weight_bytes),
                (normed_row, plan.normed_row_bytes),
                (rms_uniform, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.step_logits_rms_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-step-logits-rms-bind-group",
            &layouts.rms,
            &[
                (&self.hidden_pingpong_buffer, hidden_stride),
                (final_norm, final_norm_weight_bytes),
                (normed_row, plan.normed_row_bytes),
                (rms_uniform, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        self.gemv_logits_bind_group = Some(create_stack_bind_group(
            device,
            "decoder-stack-session-gemv-logits-bind-group",
            &layouts.gemv,
            &[
                (lm_head, lm_head_weight_bytes),
                (normed_row, plan.normed_row_bytes),
                (logits, plan.logits_bytes),
                (gemv_uniform, UNIFORM_BUFFER_BYTES),
            ],
        )?);
        Ok(())
    }

    fn upload_initial_operands(
        &self,
        queue: &JsValue,
        operands: &StackBeginOperands,
        upload_resident_weights: bool,
    ) -> Result<(), JsValue> {
        if operands.upload_initial_cache {
            write_stack_buffer(queue, &self.key_cache_buffer, &operands.key_cache_bytes)?;
            write_stack_buffer(queue, &self.value_cache_buffer, &operands.value_cache_bytes)?;
        }
        write_stack_buffer(queue, &self.rope_cos_buffer, &operands.rope_cos_bytes)?;
        write_stack_buffer(queue, &self.rope_sin_buffer, &operands.rope_sin_bytes)?;
        if upload_resident_weights {
            write_stack_buffer(
                queue,
                &self.norm1_weight_buffer,
                &operands.norm1_weight_bytes,
            )?;
            write_stack_buffer(queue, &self.q_weight_buffer, &operands.q_weight_bytes)?;
            write_stack_buffer(queue, &self.k_weight_buffer, &operands.k_weight_bytes)?;
            write_stack_buffer(queue, &self.v_weight_buffer, &operands.v_weight_bytes)?;
            write_stack_buffer(queue, &self.o_weight_buffer, &operands.o_weight_bytes)?;
            write_stack_buffer(
                queue,
                &self.norm2_weight_buffer,
                &operands.norm2_weight_bytes,
            )?;
            write_stack_buffer(queue, &self.gate_weight_buffer, &operands.gate_weight_bytes)?;
            write_stack_buffer(queue, &self.up_weight_buffer, &operands.up_weight_bytes)?;
            write_stack_buffer(queue, &self.down_weight_buffer, &operands.down_weight_bytes)?;
        }
        if let Some(zero_bias) = self.prefill_zero_bias_buffer.as_ref() {
            let bias_bytes = usize::try_from(PREFILL_ZERO_BIAS_BYTES).map_err(|_| {
                js_stack_error(&["decoder stack prefill zero bias byte size overflowed"])
            })?;
            let mut zeros = Vec::with_capacity(bias_bytes);
            zeros.resize(bias_bytes, 0);
            write_stack_buffer(queue, zero_bias, &zeros)?;
        }
        if upload_resident_weights
            && let Some(final_norm_bytes) = operands.final_norm_weight_bytes.as_ref()
        {
            let Some(final_norm) = self.final_norm_weight_buffer.as_ref() else {
                return Err(js_stack_error(&[
                    "decoder stack session logits resources are missing",
                ]));
            };
            write_stack_buffer(queue, final_norm, final_norm_bytes)?;
        }
        if upload_resident_weights
            && let Some(lm_head_bytes) = operands.lm_head_weight_bytes.as_ref()
        {
            let Some(lm_head) = self.lm_head_weight_buffer.as_ref() else {
                return Err(js_stack_error(&[
                    "decoder stack session logits resources are missing",
                ]));
            };
            write_stack_buffer(queue, lm_head, lm_head_bytes)?;
        }
        Ok(())
    }

    fn decoder_weight_pipelines(&self) -> Result<DecoderWeightPipelines<'_>, JsValue> {
        match self.weight_resource_plan.storage {
            DecoderWeightStorage::F32 => Ok(DecoderWeightPipelines {
                rms_norm: &self.rms_norm_pipeline,
                gemv_tiled: &self.gemv_tiled_pipeline,
                prefill_projection: self.prefill_projection_pipeline.as_ref(),
            }),
            DecoderWeightStorage::F16 => {
                let rms_norm = self.rms_norm_f16_pipeline.as_ref().ok_or_else(|| {
                    js_stack_error(&["decoder stack FP16 RMS-norm pipeline is missing"])
                })?;
                let gemv_tiled = self.gemv_tiled_f16_pipeline.as_ref().ok_or_else(|| {
                    js_stack_error(&["decoder stack FP16 tiled GEMV pipeline is missing"])
                })?;
                Ok(DecoderWeightPipelines {
                    rms_norm,
                    gemv_tiled,
                    prefill_projection: self.prefill_projection_f16_pipeline.as_ref(),
                })
            }
        }
    }

    fn ensure_top1_resources(&mut self, device: &JsValue) -> Result<(), JsValue> {
        if self.top1_result_buffer.is_some()
            && self.top1_readback_buffer.is_some()
            && self.top1_pipeline.is_some()
            && self.top1_bind_group.is_some()
        {
            return Ok(());
        }
        let plan = self
            .lm_head_plan
            .as_ref()
            .ok_or_else(|| js_stack_error(&["decoder stack session does not admit GPU top-1"]))?;
        let logits = self.logits_buffer.as_ref().ok_or_else(|| {
            js_stack_error(&["decoder stack session logits resources are missing"])
        })?;
        let storage_copy_src = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let readback_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let result = create_stack_buffer(
            device,
            BUFFER_TOP1_RESULT,
            TOP1_RESULT_BYTES,
            buffer_usage(&[storage_copy_src]),
        )?;
        let readback = create_stack_buffer(
            device,
            BUFFER_TOP1_READBACK,
            TOP1_RESULT_BYTES,
            buffer_usage(&[readback_usage]),
        )?;
        let layout = create_stack_bind_group_layout(device, &TOP1_LAYOUT_ENTRIES)?;
        let pipeline =
            create_stack_pipeline(device, TOP1_KERNEL_NAME, TOP1_SHADER_SOURCE, &layout)?;
        let bind_group = create_stack_bind_group(
            device,
            "decoder-stack-session-top1-bind-group",
            &layout,
            &[(logits, plan.logits_bytes), (&result, TOP1_RESULT_BYTES)],
        )?;
        self.top1_result_buffer = Some(result);
        self.top1_readback_buffer = Some(readback);
        self.top1_pipeline = Some(pipeline);
        self.top1_bind_group = Some(bind_group);
        self.top1_shader_blake3 = Some(source_blake3(TOP1_SHADER_SOURCE));
        Ok(())
    }

    fn encode_step(
        &self,
        device: &JsValue,
        queue: &JsValue,
        transition: &pvlc_runtime_core::DecoderKvSessionStepPlan,
        step_plan: &pvlc_runtime_core::DecoderLayerStepPlan,
        hidden_bytes: &[u8],
    ) -> Result<(), JsValue> {
        write_stack_buffer(queue, &self.hidden_pingpong_buffer, hidden_bytes)?;
        write_stack_buffer(
            queue,
            &self.rms_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[0]),
        )?;
        write_stack_buffer(
            queue,
            &self.gemv_q_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[1]),
        )?;
        write_stack_buffer(
            queue,
            &self.gemv_k_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[2]),
        )?;
        write_stack_buffer(
            queue,
            &self.gemv_v_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[3]),
        )?;
        write_stack_buffer(
            queue,
            &self.mrope_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[4]),
        )?;
        write_stack_buffer(
            queue,
            &self.gemv_o_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[5]),
        )?;
        write_stack_buffer(
            queue,
            &self.residual_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[6]),
        )?;
        write_stack_buffer(
            queue,
            &self.rms2_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[7]),
        )?;
        write_stack_buffer(
            queue,
            &self.gemv_gate_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[8]),
        )?;
        write_stack_buffer(
            queue,
            &self.gemv_up_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[9]),
        )?;
        write_stack_buffer(
            queue,
            &self.swiglu_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[10]),
        )?;
        write_stack_buffer(
            queue,
            &self.gemv_down_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[11]),
        )?;
        write_stack_buffer(
            queue,
            &self.residual2_uniform_buffer,
            bytemuck::cast_slice(&step_plan.stage_uniform_words[12]),
        )?;
        write_stack_buffer(
            queue,
            &self.append_uniform_buffer,
            bytemuck::cast_slice(&transition.append.uniform_words),
        )?;
        write_stack_buffer(
            queue,
            &self.split_partial_uniform_buffer,
            bytemuck::cast_slice(&transition.split_gqa.uniform_words[0]),
        )?;
        write_stack_buffer(
            queue,
            &self.split_merge_uniform_buffer,
            bytemuck::cast_slice(&transition.split_gqa.uniform_words[1]),
        )?;
        let encoder = create_stack_encoder(device, "decoder-stack-session-step-encoder")?;
        let hidden_stride = self.stack_plan.hidden_stride_bytes;
        let cache_stride = self.stack_plan.cache_stride_bytes;
        let weight_pipelines = self.decoder_weight_pipelines()?;
        for layer in 0..self.stack_plan.layers {
            let layer_index = u64::from(layer);
            let pingpong_in = hidden_stride.checked_mul(layer_index % 2).ok_or_else(|| {
                js_stack_error(&["decoder stack hidden ping-pong offset overflowed"])
            })?;
            let pingpong_out = hidden_stride
                .checked_mul((layer_index + 1) % 2)
                .ok_or_else(|| {
                    js_stack_error(&["decoder stack hidden ping-pong offset overflowed"])
                })?;
            let [
                norm1_offset,
                q_offset,
                k_offset,
                v_offset,
                o_offset,
                norm2_offset,
                gate_offset,
                up_offset,
                down_offset,
            ] = self
                .weight_resource_plan
                .layer_weight_offsets(layer)
                .ok_or_else(|| {
                    js_stack_error(&["decoder stack physical weight offset is unavailable"])
                })?;
            let cache_offset = cache_stride
                .checked_mul(layer_index)
                .ok_or_else(|| js_stack_error(&["decoder stack cache offset overflowed"]))?;
            encode_stack_pass(
                &encoder,
                weight_pipelines.rms_norm,
                &self.rms_bind_group,
                self.stack_plan
                    .layer_plan
                    .attention_block
                    .rms_norm_invocation
                    .dispatch,
                &[pingpong_in, norm1_offset],
            )?;
            encode_stack_pass(
                &encoder,
                weight_pipelines.gemv_tiled,
                &self.gemv_q_bind_group,
                self.stack_plan
                    .layer_plan
                    .attention_block
                    .query_invocation
                    .dispatch,
                &[q_offset],
            )?;
            encode_stack_pass(
                &encoder,
                weight_pipelines.gemv_tiled,
                &self.gemv_k_bind_group,
                self.stack_plan
                    .layer_plan
                    .attention_block
                    .key_invocation
                    .dispatch,
                &[k_offset],
            )?;
            encode_stack_pass(
                &encoder,
                weight_pipelines.gemv_tiled,
                &self.gemv_v_bind_group,
                self.stack_plan
                    .layer_plan
                    .attention_block
                    .value_invocation
                    .dispatch,
                &[v_offset],
            )?;
            encode_stack_pass(
                &encoder,
                &self.mrope_pipeline,
                &self.mrope_bind_group,
                self.stack_plan
                    .layer_plan
                    .attention_block
                    .mrope_invocation
                    .dispatch,
                &[],
            )?;
            encode_stack_pass(
                &encoder,
                &self.append_pipeline,
                &self.append_bind_group,
                self.kv_plan.append_invocation.dispatch,
                &[cache_offset, cache_offset],
            )?;
            encode_stack_pass(
                &encoder,
                &self.split_partial_pipeline,
                &self.split_partial_bind_group,
                transition.split_gqa.partial_invocation.dispatch,
                &[cache_offset, cache_offset],
            )?;
            encode_stack_pass(
                &encoder,
                &self.split_merge_pipeline,
                &self.split_merge_bind_group,
                transition.split_gqa.merge_invocation.dispatch,
                &[cache_offset, cache_offset],
            )?;
            encode_stack_pass(
                &encoder,
                weight_pipelines.gemv_tiled,
                &self.gemv_o_bind_group,
                self.stack_plan
                    .layer_plan
                    .attention_block
                    .output_invocation
                    .dispatch,
                &[o_offset],
            )?;
            encode_stack_pass(
                &encoder,
                &self.residual_pipeline,
                &self.residual_bind_group,
                self.stack_plan
                    .layer_plan
                    .attention_block
                    .residual_invocation
                    .dispatch,
                &[pingpong_in, 0],
            )?;
            encode_stack_pass(
                &encoder,
                weight_pipelines.rms_norm,
                &self.rms2_bind_group,
                self.stack_plan.layer_plan.norm2_invocation.dispatch,
                &[0, norm2_offset],
            )?;
            encode_stack_pass(
                &encoder,
                weight_pipelines.gemv_tiled,
                &self.gemv_gate_bind_group,
                self.stack_plan.layer_plan.gate_invocation.dispatch,
                &[gate_offset],
            )?;
            encode_stack_pass(
                &encoder,
                weight_pipelines.gemv_tiled,
                &self.gemv_up_bind_group,
                self.stack_plan.layer_plan.up_invocation.dispatch,
                &[up_offset],
            )?;
            encode_stack_pass(
                &encoder,
                &self.swiglu_pipeline,
                &self.swiglu_bind_group,
                self.stack_plan.layer_plan.swiglu_invocation.dispatch,
                &[],
            )?;
            encode_stack_pass(
                &encoder,
                weight_pipelines.gemv_tiled,
                &self.gemv_down_bind_group,
                self.stack_plan.layer_plan.down_invocation.dispatch,
                &[down_offset],
            )?;
            encode_stack_pass(
                &encoder,
                &self.residual_pipeline,
                &self.residual2_bind_group,
                self.stack_plan
                    .layer_plan
                    .second_residual_invocation
                    .dispatch,
                &[0, pingpong_out],
            )?;
        }
        encode_stack_copy(
            &encoder,
            &self.hidden_pingpong_buffer,
            0,
            &self.stack_readback_buffer,
            0,
            hidden_stride,
        )?;
        submit_stack_encoder(queue, &encoder)
    }

    fn encode_prefill(
        &self,
        device: &JsValue,
        queue: &JsValue,
        prefill_plan: &pvlc_runtime_core::DecoderStackPrefillPlan,
        hidden_bytes: &[u8],
    ) -> Result<(), JsValue> {
        let Some(hidden_storage) = self.prefill_hidden_storage_buffer.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let weight_pipelines = self.decoder_weight_pipelines()?;
        let Some(projection_pipeline) = weight_pipelines.prefill_projection else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(mrope_pipeline) = self.prefill_mrope_pipeline.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(append_range_pipeline) = self.kv_append_range_pipeline.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(gqa_pipeline) = self.prefill_gqa_pipeline.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(rms1_bind_group) = self.prefill_rms1_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(query_bind_group) = self.prefill_query_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(key_bind_group) = self.prefill_key_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(value_bind_group) = self.prefill_value_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(mrope_bind_group) = self.prefill_mrope_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(append_range_bind_group) = self.prefill_kv_append_range_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(gqa_bind_group) = self.prefill_gqa_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(output_bind_group) = self.prefill_output_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(residual_bind_group) = self.prefill_residual_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(rms2_bind_group) = self.prefill_rms2_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(gate_bind_group) = self.prefill_gate_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(up_bind_group) = self.prefill_up_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(swiglu_bind_group) = self.prefill_swiglu_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(down_bind_group) = self.prefill_down_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        let Some(residual2_bind_group) = self.prefill_residual2_bind_group.as_ref() else {
            return Err(js_stack_error(&[
                "decoder stack session is not prefill capable",
            ]));
        };
        write_stack_buffer(queue, hidden_storage, hidden_bytes)?;
        write_stack_buffer(
            queue,
            &self.rms_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[0]),
        )?;
        write_stack_buffer(
            queue,
            &self.gemv_q_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[1]),
        )?;
        write_stack_buffer(
            queue,
            &self.gemv_k_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[2]),
        )?;
        write_stack_buffer(
            queue,
            &self.gemv_v_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[3]),
        )?;
        write_stack_buffer(
            queue,
            &self.mrope_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[4]),
        )?;
        write_stack_buffer(
            queue,
            &self.append_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[5]),
        )?;
        write_stack_buffer(
            queue,
            &self.attention_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[6]),
        )?;
        write_stack_buffer(
            queue,
            &self.gemv_o_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[7]),
        )?;
        write_stack_buffer(
            queue,
            &self.residual_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[8]),
        )?;
        write_stack_buffer(
            queue,
            &self.rms2_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[9]),
        )?;
        write_stack_buffer(
            queue,
            &self.gemv_gate_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[10]),
        )?;
        write_stack_buffer(
            queue,
            &self.gemv_up_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[11]),
        )?;
        write_stack_buffer(
            queue,
            &self.swiglu_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[12]),
        )?;
        write_stack_buffer(
            queue,
            &self.gemv_down_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[13]),
        )?;
        write_stack_buffer(
            queue,
            &self.residual2_uniform_buffer,
            bytemuck::cast_slice(&prefill_plan.stage_uniform_words[14]),
        )?;
        let encoder = create_stack_encoder(device, "decoder-stack-session-prefill-encoder")?;
        let cache_stride = prefill_plan.cache_stride_bytes;
        for layer in 0..prefill_plan.layers {
            let layer_index = u64::from(layer);
            let [
                norm1_offset,
                q_offset,
                k_offset,
                v_offset,
                o_offset,
                norm2_offset,
                gate_offset,
                up_offset,
                down_offset,
            ] = self
                .weight_resource_plan
                .layer_weight_offsets(layer)
                .ok_or_else(|| {
                    js_stack_error(&["decoder stack physical weight offset is unavailable"])
                })?;
            let cache_offset = cache_stride
                .checked_mul(layer_index)
                .ok_or_else(|| js_stack_error(&["decoder stack cache offset overflowed"]))?;
            encode_stack_pass(
                &encoder,
                weight_pipelines.rms_norm,
                rms1_bind_group,
                prefill_plan.stage_invocations[0].dispatch,
                &[0, norm1_offset],
            )?;
            encode_stack_pass(
                &encoder,
                projection_pipeline,
                query_bind_group,
                prefill_plan.stage_invocations[1].dispatch,
                &[q_offset],
            )?;
            encode_stack_pass(
                &encoder,
                projection_pipeline,
                key_bind_group,
                prefill_plan.stage_invocations[2].dispatch,
                &[k_offset],
            )?;
            encode_stack_pass(
                &encoder,
                projection_pipeline,
                value_bind_group,
                prefill_plan.stage_invocations[3].dispatch,
                &[v_offset],
            )?;
            encode_stack_pass(
                &encoder,
                mrope_pipeline,
                mrope_bind_group,
                prefill_plan.stage_invocations[4].dispatch,
                &[],
            )?;
            encode_stack_pass(
                &encoder,
                append_range_pipeline,
                append_range_bind_group,
                prefill_plan.stage_invocations[5].dispatch,
                &[cache_offset, cache_offset],
            )?;
            encode_stack_pass(
                &encoder,
                gqa_pipeline,
                gqa_bind_group,
                prefill_plan.stage_invocations[6].dispatch,
                &[cache_offset, cache_offset],
            )?;
            encode_stack_pass(
                &encoder,
                projection_pipeline,
                output_bind_group,
                prefill_plan.stage_invocations[7].dispatch,
                &[o_offset],
            )?;
            encode_stack_pass(
                &encoder,
                &self.residual_pipeline,
                residual_bind_group,
                prefill_plan.stage_invocations[8].dispatch,
                &[0, 0],
            )?;
            encode_stack_pass(
                &encoder,
                weight_pipelines.rms_norm,
                rms2_bind_group,
                prefill_plan.stage_invocations[9].dispatch,
                &[0, norm2_offset],
            )?;
            encode_stack_pass(
                &encoder,
                projection_pipeline,
                gate_bind_group,
                prefill_plan.stage_invocations[10].dispatch,
                &[gate_offset],
            )?;
            encode_stack_pass(
                &encoder,
                projection_pipeline,
                up_bind_group,
                prefill_plan.stage_invocations[11].dispatch,
                &[up_offset],
            )?;
            encode_stack_pass(
                &encoder,
                &self.swiglu_pipeline,
                swiglu_bind_group,
                prefill_plan.stage_invocations[12].dispatch,
                &[],
            )?;
            encode_stack_pass(
                &encoder,
                projection_pipeline,
                down_bind_group,
                prefill_plan.stage_invocations[13].dispatch,
                &[down_offset],
            )?;
            encode_stack_pass(
                &encoder,
                &self.residual_pipeline,
                residual2_bind_group,
                prefill_plan.stage_invocations[14].dispatch,
                &[0, 0],
            )?;
        }
        let hidden_stride = prefill_plan.hidden_stride_bytes;
        let last_row_offset = u64::from(prefill_plan.tokens.saturating_sub(1))
            .checked_mul(hidden_stride)
            .ok_or_else(|| js_stack_error(&["decoder stack prefill readback offset overflowed"]))?;
        encode_stack_copy(
            &encoder,
            hidden_storage,
            last_row_offset,
            &self.stack_readback_buffer,
            0,
            hidden_stride,
        )?;
        submit_stack_encoder(queue, &encoder)
    }

    fn encode_finish(
        &self,
        device: &JsValue,
        queue: &JsValue,
    ) -> Result<(wgpu::webgpu::GpuBuffer, wgpu::webgpu::GpuBuffer), JsValue> {
        let readback_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let cache_bytes = u64::from(self.stack_plan.layers)
            .checked_mul(self.stack_plan.cache_stride_bytes)
            .ok_or_else(|| js_stack_error(&["decoder stack compact cache byte size overflowed"]))?;
        let key_readback = create_stack_buffer(
            device,
            BUFFER_FINISH_KEY_READBACK,
            cache_bytes,
            buffer_usage(&[readback_usage]),
        )?;
        let value_readback = create_stack_buffer(
            device,
            BUFFER_FINISH_VALUE_READBACK,
            cache_bytes,
            buffer_usage(&[readback_usage]),
        )?;
        let encoder = create_stack_encoder(device, "decoder-stack-session-finish-encoder")?;
        encode_stack_copy(
            &encoder,
            &self.key_cache_buffer,
            0,
            &key_readback,
            0,
            cache_bytes,
        )?;
        encode_stack_copy(
            &encoder,
            &self.value_cache_buffer,
            0,
            &value_readback,
            0,
            cache_bytes,
        )?;
        submit_stack_encoder(queue, &encoder)?;
        Ok((key_readback, value_readback))
    }
}

fn session_shader_digests(session: &BrowserDecoderStackSession) -> StackShaderDigests {
    StackShaderDigests {
        rms_norm: session.rms_norm_shader_blake3,
        gemv: session.gemv_shader_blake3,
        gemv_tiled: session.gemv_tiled_shader_blake3,
        mrope: session.mrope_shader_blake3,
        append: session.append_shader_blake3,
        attention: session.attention_shader_blake3,
        swiglu: session.swiglu_shader_blake3,
        residual: session.residual_shader_blake3,
        split_partial: session.split_partial_shader_blake3,
        split_merge: session.split_merge_shader_blake3,
    }
}

fn insert_blake3(hashes: &mut Map<String, Value>, name: &str, digest: &[u8; 32]) {
    hashes.insert(name.to_owned(), Value::from(blake3_hex(digest).as_str()));
}

fn shader_blake3_json(digests: &StackShaderDigests) -> Value {
    let mut hashes = Map::new();
    insert_blake3(&mut hashes, RMS_NORM_KERNEL_NAME, &digests.rms_norm);
    insert_blake3(&mut hashes, GEMV_KERNEL_NAME, &digests.gemv);
    insert_blake3(&mut hashes, MROPE_KERNEL_NAME, &digests.mrope);
    insert_blake3(&mut hashes, APPEND_KERNEL_NAME, &digests.append);
    insert_blake3(&mut hashes, ATTENTION_KERNEL_NAME, &digests.attention);
    insert_blake3(&mut hashes, SWIGLU_KERNEL_NAME, &digests.swiglu);
    insert_blake3(&mut hashes, RESIDUAL_KERNEL_NAME, &digests.residual);
    insert_blake3(&mut hashes, GEMV_TILED_KERNEL_NAME, &digests.gemv_tiled);
    insert_blake3(
        &mut hashes,
        SPLIT_PARTIAL_KERNEL_NAME,
        &digests.split_partial,
    );
    insert_blake3(&mut hashes, SPLIT_MERGE_KERNEL_NAME, &digests.split_merge);
    Value::Object(hashes)
}

fn extend_shader_blake3_with_prefill(base: Value, digests: [&[u8; 32]; 4]) -> Value {
    let Value::Object(mut hashes) = base else {
        return base;
    };
    insert_blake3(&mut hashes, PROJECTION_KERNEL_NAME, digests[0]);
    insert_blake3(&mut hashes, PREFILL_MROPE_KERNEL_NAME, digests[1]);
    insert_blake3(&mut hashes, KV_APPEND_RANGE_KERNEL_NAME, digests[2]);
    insert_blake3(&mut hashes, PREFILL_GQA_KERNEL_NAME, digests[3]);
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
        js_stack_error(&[
            "cannot serialize decoder stack diagnostics: ",
            &error.to_string(),
        ])
    })
}

fn creation_diagnostics_json(
    kv_plan: &pvlc_runtime_core::DecoderKvSessionPlan,
    stack_plan: &pvlc_runtime_core::DecoderStackPlan,
    digests: &StackShaderDigests,
    sources: &StackShaderSources,
    prefill_capable: bool,
    logits_capable: bool,
    resident_weight_cache_hit: bool,
    initial_cache_uploaded: bool,
) -> Result<String, JsValue> {
    let mut root = Map::new();
    root.insert("schema_version".to_owned(), Value::from(1u64));
    root.insert(
        "initial_cache_tokens".to_owned(),
        Value::from(if prefill_capable {
            0
        } else {
            u64::from(kv_plan.initial_cache_tokens)
        }),
    );
    root.insert(
        "cache_capacity".to_owned(),
        Value::from(u64::from(kv_plan.cache_capacity)),
    );
    root.insert(
        "layers".to_owned(),
        Value::from(u64::from(stack_plan.layers)),
    );
    root.insert(
        "hidden_size".to_owned(),
        Value::from(u64::from(stack_plan.layer_plan.attention_block.hidden_size)),
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
    if prefill_capable {
        root.insert(
            "shader_blake3".to_owned(),
            extend_shader_blake3_with_prefill(
                shader_blake3_json(digests),
                [
                    &source_blake3(&sources.projection),
                    &source_blake3(&sources.prefill_mrope),
                    &source_blake3(&sources.kv_append_range),
                    &source_blake3(&sources.prefill_gqa),
                ],
            ),
        );
        root.insert("buffer_count".to_owned(), Value::from(59u64));
        root.insert("pipeline_count".to_owned(), Value::from(12u64));
        root.insert("bind_group_count".to_owned(), Value::from(31u64));
        root.insert("initial_upload_count".to_owned(), Value::from(14u64));
    } else {
        root.insert("shader_blake3".to_owned(), shader_blake3_json(digests));
        root.insert("buffer_count".to_owned(), Value::from(47u64));
        root.insert("pipeline_count".to_owned(), Value::from(8u64));
        root.insert("bind_group_count".to_owned(), Value::from(16u64));
        root.insert("initial_upload_count".to_owned(), Value::from(13u64));
    }
    if logits_capable {
        // The logits-capable topology adds the two shared weights, the
        // normed row, the logits storage and readback, and the two static
        // Preserve the accepted logits topology counters; GPU top-1 is
        // reported additively below so existing logits evidence remains
        // comparable.
        for (key, delta) in [
            ("buffer_count", 7u64),
            ("pipeline_count", 0u64),
            ("bind_group_count", 3u64),
            (
                "initial_upload_count",
                if resident_weight_cache_hit {
                    0u64
                } else {
                    2u64
                },
            ),
        ] {
            let current = root.get(key).and_then(Value::as_u64).ok_or_else(|| {
                js_stack_error(&["decoder stack session creation diagnostics drifted"])
            })?;
            root.insert(key.to_owned(), Value::from(current + delta));
        }
        root.insert("logits_capable".to_owned(), Value::from(true));
    }
    if resident_weight_cache_hit {
        let cold_weight_uploads = 9u64;
        let current = root
            .get("initial_upload_count")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                js_stack_error(&["decoder stack session creation diagnostics drifted"])
            })?;
        root.insert(
            "initial_upload_count".to_owned(),
            Value::from(current.saturating_sub(cold_weight_uploads)),
        );
    }
    if !initial_cache_uploaded {
        let current = root
            .get("initial_upload_count")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                js_stack_error(&["decoder stack session creation diagnostics drifted"])
            })?;
        root.insert(
            "initial_upload_count".to_owned(),
            Value::from(current.saturating_sub(2)),
        );
    }
    json_text(Value::Object(root))
}

fn step_diagnostics_json(
    session: &BrowserDecoderStackSession,
    transition: &pvlc_runtime_core::DecoderKvSessionStepPlan,
) -> Result<String, JsValue> {
    let mut root = Map::new();
    root.insert("schema_version".to_owned(), Value::from(1u64));
    if let Some(checkpoint_blake3) = session.checkpoint_blake3.as_ref() {
        root.insert(
            "checkpoint_blake3".to_owned(),
            Value::from(blake3_hex(checkpoint_blake3)),
        );
    }
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
    root.insert("queue_write_count".to_owned(), Value::from(17u64));
    root.insert("command_encoder_count".to_owned(), Value::from(1u64));
    let pass_count = u64::from(session.stack_plan.layers)
        .checked_mul(16)
        .ok_or_else(|| js_stack_error(&["decoder stack pass count overflowed"]))?;
    root.insert("compute_pass_count".to_owned(), Value::from(pass_count));
    root.insert("dispatch_count".to_owned(), Value::from(pass_count));
    root.insert("copy_count".to_owned(), Value::from(1u64));
    root.insert("command_buffer_count".to_owned(), Value::from(1u64));
    root.insert("submission_count".to_owned(), Value::from(1u64));
    root.insert("map_count".to_owned(), Value::from(1u64));
    root.insert(
        "readback_bytes".to_owned(),
        Value::from(session.stack_plan.hidden_stride_bytes),
    );
    let stage_kinds = [
        "dispatch_rms_norm",
        "dispatch_query",
        "dispatch_key",
        "dispatch_value",
        "dispatch_mrope",
        "dispatch_append",
        "dispatch_split_partial",
        "dispatch_split_merge",
        "dispatch_output",
        "dispatch_residual",
        "dispatch_post_norm",
        "dispatch_gate",
        "dispatch_up",
        "dispatch_swiglu",
        "dispatch_down",
        "dispatch_residual_2",
    ];
    let mut kinds: Vec<&str> = Vec::new();
    kinds.extend(["queue_write"; 17]);
    for _ in 0..session.stack_plan.layers {
        for kind in stage_kinds {
            kinds.push(kind);
        }
    }
    kinds.push("copy_stack_output");
    kinds.push("submit");
    kinds.push("map_stack_output");
    let mut effects = Vec::new();
    for (ordinal, kind) in kinds.into_iter().enumerate() {
        let mut effect = Map::new();
        effect.insert("ordinal".to_owned(), Value::from(ordinal as u64 + 1));
        effect.insert("kind".to_owned(), Value::from(kind));
        effects.push(Value::Object(effect));
    }
    root.insert("effects".to_owned(), Value::Array(effects));
    json_text(Value::Object(root))
}

fn prefill_diagnostics_json(
    session: &BrowserDecoderStackSession,
    prefill_plan: &pvlc_runtime_core::DecoderStackPrefillPlan,
) -> Result<String, JsValue> {
    let (Some(projection), Some(prefill_mrope), Some(kv_append_range), Some(prefill_gqa)) = (
        session.prefill_projection_shader_blake3.as_ref(),
        session.prefill_mrope_shader_blake3.as_ref(),
        session.kv_append_range_shader_blake3.as_ref(),
        session.prefill_gqa_shader_blake3.as_ref(),
    ) else {
        return Err(js_stack_error(&[
            "decoder stack session is not prefill capable",
        ]));
    };
    let mut root = Map::new();
    root.insert("schema_version".to_owned(), Value::from(1u64));
    root.insert(
        "tokens".to_owned(),
        Value::from(u64::from(prefill_plan.tokens)),
    );
    root.insert(
        "cache_capacity".to_owned(),
        Value::from(u64::from(prefill_plan.cache_capacity)),
    );
    root.insert("cache_tokens_before".to_owned(), Value::from(0u64));
    root.insert(
        "cache_tokens_after".to_owned(),
        Value::from(u64::from(prefill_plan.tokens)),
    );
    root.insert("checked_error_scopes".to_owned(), checked_scopes_json());
    root.insert("captured_errors".to_owned(), Value::Array(Vec::new()));
    root.insert(
        "shader_blake3".to_owned(),
        extend_shader_blake3_with_prefill(
            shader_blake3_json(&session_shader_digests(session)),
            [projection, prefill_mrope, kv_append_range, prefill_gqa],
        ),
    );
    root.insert("queue_write_count".to_owned(), Value::from(16u64));
    root.insert("command_encoder_count".to_owned(), Value::from(1u64));
    let pass_count = u64::from(prefill_plan.layers)
        .checked_mul(15)
        .ok_or_else(|| js_stack_error(&["decoder stack prefill pass count overflowed"]))?;
    root.insert("compute_pass_count".to_owned(), Value::from(pass_count));
    root.insert("dispatch_count".to_owned(), Value::from(pass_count));
    root.insert("copy_count".to_owned(), Value::from(1u64));
    root.insert("command_buffer_count".to_owned(), Value::from(1u64));
    root.insert("submission_count".to_owned(), Value::from(1u64));
    root.insert("map_count".to_owned(), Value::from(1u64));
    root.insert(
        "readback_bytes".to_owned(),
        Value::from(prefill_plan.hidden_stride_bytes),
    );
    let stage_kinds = [
        "dispatch_rms_norm",
        "dispatch_query",
        "dispatch_key",
        "dispatch_value",
        "dispatch_prefill_mrope",
        "dispatch_kv_append_range",
        "dispatch_prefill_gqa",
        "dispatch_output",
        "dispatch_residual",
        "dispatch_post_norm",
        "dispatch_gate",
        "dispatch_up",
        "dispatch_swiglu",
        "dispatch_down",
        "dispatch_residual_2",
    ];
    let mut kinds: Vec<&str> = Vec::new();
    kinds.extend(["queue_write"; 16]);
    for _ in 0..prefill_plan.layers {
        for kind in stage_kinds {
            kinds.push(kind);
        }
    }
    kinds.push("copy_stack_output");
    kinds.push("submit");
    kinds.push("map_stack_output");
    let mut effects = Vec::new();
    for (ordinal, kind) in kinds.into_iter().enumerate() {
        let mut effect = Map::new();
        effect.insert("ordinal".to_owned(), Value::from(ordinal as u64 + 1));
        effect.insert("kind".to_owned(), Value::from(kind));
        effects.push(Value::Object(effect));
    }
    root.insert("effects".to_owned(), Value::Array(effects));
    json_text(Value::Object(root))
}

fn finish_diagnostics_json(session: &BrowserDecoderStackSession) -> Result<String, JsValue> {
    let cache_bytes = u64::from(session.stack_plan.layers)
        .checked_mul(session.stack_plan.cache_stride_bytes)
        .ok_or_else(|| js_stack_error(&["decoder stack compact cache byte size overflowed"]))?;
    let readback_bytes = cache_bytes
        .checked_mul(2)
        .ok_or_else(|| js_stack_error(&["decoder stack finish readback overflowed"]))?;
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

fn stack_step_result(output: Vec<u8>, diagnostics: String) -> Result<JsValue, JsValue> {
    let result = js_sys::Object::new();
    let output_bytes = js_sys::Uint8Array::from(output.as_slice());
    js_object_set(&result, "output_bytes", &output_bytes)?;
    js_object_set(
        &result,
        "diagnostics_json",
        &JsValue::from_str(&diagnostics),
    )?;
    Ok(result.into())
}

fn logits_diagnostics_json(session: &BrowserDecoderStackSession) -> Result<String, JsValue> {
    let Some(plan) = session.lm_head_plan else {
        return Err(js_stack_error(&[
            "decoder stack session is not logits capable",
        ]));
    };
    let mut root = Map::new();
    root.insert("schema_version".to_owned(), Value::from(1u64));
    root.insert(
        "cache_tokens".to_owned(),
        Value::from(u64::from(session.cache_tokens)),
    );
    root.insert("checked_error_scopes".to_owned(), checked_scopes_json());
    root.insert("captured_errors".to_owned(), Value::Array(Vec::new()));
    root.insert(
        "shader_blake3".to_owned(),
        shader_blake3_json(&session_shader_digests(session)),
    );
    root.insert("queue_write_count".to_owned(), Value::from(0u64));
    root.insert("command_encoder_count".to_owned(), Value::from(1u64));
    root.insert("compute_pass_count".to_owned(), Value::from(2u64));
    root.insert("dispatch_count".to_owned(), Value::from(2u64));
    root.insert("copy_count".to_owned(), Value::from(1u64));
    root.insert("command_buffer_count".to_owned(), Value::from(1u64));
    root.insert("submission_count".to_owned(), Value::from(1u64));
    root.insert("map_count".to_owned(), Value::from(1u64));
    root.insert("readback_bytes".to_owned(), Value::from(plan.logits_bytes));
    let kinds = [
        "dispatch_final_rms_norm",
        "dispatch_lm_head_gemv",
        "copy_logits_output",
        "submit",
        "map_logits_output",
    ];
    let mut effects = Vec::new();
    for (ordinal, kind) in kinds.into_iter().enumerate() {
        let mut effect = Map::new();
        effect.insert("ordinal".to_owned(), Value::from(ordinal as u64 + 1));
        effect.insert("kind".to_owned(), Value::from(kind));
        effects.push(Value::Object(effect));
    }
    root.insert("effects".to_owned(), Value::Array(effects));
    json_text(Value::Object(root))
}

fn stack_logits_result(output: Vec<u8>, diagnostics: String) -> Result<JsValue, JsValue> {
    let result = js_sys::Object::new();
    let logits_bytes = js_sys::Uint8Array::from(output.as_slice());
    js_object_set(&result, "logits_bytes", &logits_bytes)?;
    js_object_set(
        &result,
        "diagnostics_json",
        &JsValue::from_str(&diagnostics),
    )?;
    Ok(result.into())
}

fn top1_diagnostics_json(
    session: &BrowserDecoderStackSession,
    queue_wall_time_ns: u64,
) -> Result<String, JsValue> {
    let Some(top1_digest) = session.top1_shader_blake3 else {
        return Err(js_stack_error(&[
            "decoder stack session is not GPU top-1 capable",
        ]));
    };
    let mut root = Map::new();
    root.insert("schema_version".to_owned(), Value::from(1u64));
    root.insert(
        "cache_tokens".to_owned(),
        Value::from(u64::from(session.cache_tokens)),
    );
    root.insert("checked_error_scopes".to_owned(), checked_scopes_json());
    root.insert("captured_errors".to_owned(), Value::Array(Vec::new()));
    let mut shaders = match shader_blake3_json(&session_shader_digests(session)) {
        Value::Object(shaders) => shaders,
        _ => Map::new(),
    };
    insert_blake3(&mut shaders, TOP1_KERNEL_NAME, &top1_digest);
    root.insert("shader_blake3".to_owned(), Value::Object(shaders));
    root.insert(
        "queue_wall_time_ns".to_owned(),
        Value::from(queue_wall_time_ns),
    );
    root.insert("queue_write_count".to_owned(), Value::from(0u64));
    root.insert("command_encoder_count".to_owned(), Value::from(1u64));
    root.insert("compute_pass_count".to_owned(), Value::from(3u64));
    root.insert("dispatch_count".to_owned(), Value::from(3u64));
    root.insert("copy_count".to_owned(), Value::from(1u64));
    root.insert("command_buffer_count".to_owned(), Value::from(1u64));
    root.insert("submission_count".to_owned(), Value::from(1u64));
    root.insert("map_count".to_owned(), Value::from(1u64));
    root.insert("readback_bytes".to_owned(), Value::from(TOP1_RESULT_BYTES));
    root.insert("full_logits_readback_elided".to_owned(), Value::from(true));
    json_text(Value::Object(root))
}

fn stack_top1_result(token_id: u32, value: f32, diagnostics: String) -> Result<JsValue, JsValue> {
    let result = js_sys::Object::new();
    js_object_set(&result, "token_id", &JsValue::from_f64(f64::from(token_id)))?;
    js_object_set(&result, "value", &JsValue::from_f64(f64::from(value)))?;
    js_object_set(
        &result,
        "diagnostics_json",
        &JsValue::from_str(&diagnostics),
    )?;
    Ok(result.into())
}

fn stack_finish_result(
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

fn probe_m7q1_precision_admission_json(
    precision_profile: &str,
    shader_f16_available: bool,
) -> Result<String, JsValue> {
    let storage = match precision_profile {
        "fidelity" => DecoderWeightStorage::F32,
        "balanced" => DecoderWeightStorage::F16,
        _ => {
            return Err(js_stack_error(&[
                "M7q1 precision probe profile is unsupported",
            ]));
        }
    };
    let requires_shader_f16 = storage.requires_shader_f16();
    let admitted = !requires_shader_f16 || shader_f16_available;
    let mut effects = Map::new();
    effects.insert("buffer_creations".to_owned(), Value::from(0u64));
    effects.insert("queue_writes".to_owned(), Value::from(0u64));
    effects.insert("submissions".to_owned(), Value::from(0u64));
    let mut report = Map::new();
    report.insert(
        "precision_profile".to_owned(),
        Value::from(precision_profile),
    );
    report.insert("admitted".to_owned(), Value::from(admitted));
    report.insert("effects".to_owned(), Value::Object(effects));
    if !admitted {
        report.insert(
            "error".to_owned(),
            Value::from("balanced precision requires shader-f16"),
        );
    }
    json_text(Value::Object(report))
}

fn m7q1_f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn m7q1_f16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn m7q1_probe_elements(dimensions: &[u32], label: &str) -> Result<usize, JsValue> {
    let mut elements = 1_u64;
    for dimension in dimensions {
        if *dimension == 0 {
            return Err(js_stack_error(&["M7q1 ", label, " has a zero dimension"]));
        }
        elements = elements
            .checked_mul(u64::from(*dimension))
            .ok_or_else(|| js_stack_error(&["M7q1 ", label, " element count overflowed"]))?;
    }
    usize::try_from(elements)
        .map_err(|_| js_stack_error(&["M7q1 ", label, " element count is not addressable"]))
}

fn require_m7q1_f32(values: &[f32], expected: usize, label: &str) -> Result<(), JsValue> {
    if values.len() != expected {
        return Err(js_stack_error(&[
            "M7q1 ",
            label,
            " length drifted from its geometry",
        ]));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(js_stack_error(&[
            "M7q1 ",
            label,
            " contains a nonfinite f32 value",
        ]));
    }
    Ok(())
}

fn require_m7q1_f16(values: &[u16], expected: usize, label: &str) -> Result<(), JsValue> {
    if values.len() != expected {
        return Err(js_stack_error(&[
            "M7q1 ",
            label,
            " length drifted from its geometry",
        ]));
    }
    if values.iter().any(|bits| bits & 0x7c00 == 0x7c00) {
        return Err(js_stack_error(&[
            "M7q1 ",
            label,
            " contains a nonfinite f16 value",
        ]));
    }
    Ok(())
}

fn m7q1_object<'a>(
    value: &'a Value,
    expected_keys: &[&str],
    label: &str,
) -> Result<&'a Map<String, Value>, JsValue> {
    let object = value
        .as_object()
        .ok_or_else(|| js_stack_error(&["M7q1 ", label, " must be an object"]))?;
    if object.len() != expected_keys.len()
        || expected_keys.iter().any(|key| !object.contains_key(*key))
    {
        return Err(js_stack_error(&["M7q1 ", label, " keys drifted"]));
    }
    Ok(object)
}

fn m7q1_field<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a Value, JsValue> {
    object
        .get(key)
        .ok_or_else(|| js_stack_error(&["M7q1 ", label, " is missing field ", key]))
}

fn m7q1_string(object: &Map<String, Value>, key: &str, label: &str) -> Result<String, JsValue> {
    m7q1_field(object, key, label)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| js_stack_error(&["M7q1 ", label, " field ", key, " must be a string"]))
}

fn m7q1_u32(object: &Map<String, Value>, key: &str, label: &str) -> Result<u32, JsValue> {
    let integer = m7q1_field(object, key, label)?.as_u64().ok_or_else(|| {
        js_stack_error(&[
            "M7q1 ",
            label,
            " field ",
            key,
            " must be an unsigned integer",
        ])
    })?;
    u32::try_from(integer)
        .map_err(|_| js_stack_error(&["M7q1 ", label, " field ", key, " is out of range"]))
}

fn m7q1_f32(object: &Map<String, Value>, key: &str, label: &str) -> Result<f32, JsValue> {
    let number = m7q1_field(object, key, label)?
        .as_f64()
        .ok_or_else(|| js_stack_error(&["M7q1 ", label, " field ", key, " must be a number"]))?;
    let narrowed = number as f32;
    if !narrowed.is_finite() {
        return Err(js_stack_error(&[
            "M7q1 ",
            label,
            " field ",
            key,
            " is outside finite f32",
        ]));
    }
    Ok(narrowed)
}

fn m7q1_f32_array(
    object: &Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<Vec<f32>, JsValue> {
    let values = m7q1_field(object, key, label)?
        .as_array()
        .ok_or_else(|| js_stack_error(&["M7q1 ", label, " field ", key, " must be an array"]))?;
    let mut parsed = Vec::with_capacity(values.len());
    for value in values {
        let number = value
            .as_f64()
            .ok_or_else(|| js_stack_error(&["M7q1 ", label, " array is not numeric"]))?;
        let narrowed = number as f32;
        if !narrowed.is_finite() {
            return Err(js_stack_error(&[
                "M7q1 ",
                label,
                " array contains a value outside finite f32",
            ]));
        }
        parsed.push(narrowed);
    }
    Ok(parsed)
}

fn m7q1_u16_array(
    object: &Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<Vec<u16>, JsValue> {
    let values = m7q1_field(object, key, label)?
        .as_array()
        .ok_or_else(|| js_stack_error(&["M7q1 ", label, " field ", key, " must be an array"]))?;
    let mut parsed = Vec::with_capacity(values.len());
    for value in values {
        let integer = value
            .as_u64()
            .and_then(|integer| u16::try_from(integer).ok())
            .ok_or_else(|| js_stack_error(&["M7q1 ", label, " array contains a non-u16 value"]))?;
        parsed.push(integer);
    }
    Ok(parsed)
}

fn parse_m7q1_linear_probe(
    object: &Map<String, Value>,
    label: &str,
) -> Result<M7q1LinearProbe, JsValue> {
    Ok(M7q1LinearProbe {
        tokens: m7q1_u32(object, "tokens", label)?,
        input_width: m7q1_u32(object, "input_width", label)?,
        output_width: m7q1_u32(object, "output_width", label)?,
        input: m7q1_f32_array(object, "input", label)?,
        weight_f16: m7q1_u16_array(object, "weight_f16", label)?,
        bias: m7q1_f32_array(object, "bias", label)?,
    })
}

fn parse_m7q1_weight_probe(fixture_json: &str) -> Result<M7q1WeightProbe, JsValue> {
    let value: Value = serde_json::from_str(fixture_json).map_err(|error| {
        js_stack_error(&["invalid M7q1 FP16 weight probe JSON: ", &error.to_string()])
    })?;
    let root = m7q1_object(
        &value,
        &[
            "precision_profile",
            "checkpoint_blake3",
            "checkpoint_partition",
            "rms",
            "gemv",
            "linear",
            "linear_input_major",
            "vision",
        ],
        "weight probe",
    )?;
    let checkpoint_partition = m7q1_object(
        m7q1_field(root, "checkpoint_partition", "weight probe")?,
        &["f16_checkpoint_shards", "f32_rope_table_shards"],
        "checkpoint partition",
    )?;
    let rms = m7q1_object(
        m7q1_field(root, "rms", "weight probe")?,
        &["rows", "width", "epsilon", "input", "weight_f16"],
        "RMS probe",
    )?;
    let gemv = m7q1_object(
        m7q1_field(root, "gemv", "weight probe")?,
        &["rows", "columns", "matrix_f16", "vector"],
        "GEMV probe",
    )?;
    let linear = m7q1_object(
        m7q1_field(root, "linear", "weight probe")?,
        &[
            "tokens",
            "input_width",
            "output_width",
            "input",
            "weight_f16",
            "bias",
        ],
        "linear probe",
    )?;
    let linear_input_major = m7q1_object(
        m7q1_field(root, "linear_input_major", "weight probe")?,
        &[
            "tokens",
            "input_width",
            "output_width",
            "input",
            "weight_f16",
            "bias",
        ],
        "input-major linear probe",
    )?;
    let vision = m7q1_object(
        m7q1_field(root, "vision", "weight probe")?,
        &[
            "patches",
            "input_width",
            "output_width",
            "input",
            "weight",
            "bias",
        ],
        "vision probe",
    )?;

    Ok(M7q1WeightProbe {
        precision_profile: m7q1_string(root, "precision_profile", "weight probe")?,
        checkpoint_blake3: m7q1_string(root, "checkpoint_blake3", "weight probe")?,
        checkpoint_partition: M7q1CheckpointPartition {
            f16_checkpoint_shards: m7q1_u32(
                checkpoint_partition,
                "f16_checkpoint_shards",
                "checkpoint partition",
            )?,
            f32_rope_table_shards: m7q1_u32(
                checkpoint_partition,
                "f32_rope_table_shards",
                "checkpoint partition",
            )?,
        },
        rms: M7q1RmsProbe {
            rows: m7q1_u32(rms, "rows", "RMS probe")?,
            width: m7q1_u32(rms, "width", "RMS probe")?,
            epsilon: m7q1_f32(rms, "epsilon", "RMS probe")?,
            input: m7q1_f32_array(rms, "input", "RMS probe")?,
            weight_f16: m7q1_u16_array(rms, "weight_f16", "RMS probe")?,
        },
        gemv: M7q1GemvProbe {
            rows: m7q1_u32(gemv, "rows", "GEMV probe")?,
            columns: m7q1_u32(gemv, "columns", "GEMV probe")?,
            matrix_f16: m7q1_u16_array(gemv, "matrix_f16", "GEMV probe")?,
            vector: m7q1_f32_array(gemv, "vector", "GEMV probe")?,
        },
        linear: parse_m7q1_linear_probe(linear, "linear probe")?,
        linear_input_major: parse_m7q1_linear_probe(
            linear_input_major,
            "input-major linear probe",
        )?,
        vision: M7q1VisionProbe {
            patches: m7q1_u32(vision, "patches", "vision probe")?,
            input_width: m7q1_u32(vision, "input_width", "vision probe")?,
            output_width: m7q1_u32(vision, "output_width", "vision probe")?,
            input: m7q1_f32_array(vision, "input", "vision probe")?,
            weight: m7q1_f32_array(vision, "weight", "vision probe")?,
            bias: m7q1_f32_array(vision, "bias", "vision probe")?,
        },
    })
}

fn validate_m7q1_weight_probe(probe: &M7q1WeightProbe) -> Result<(), JsValue> {
    if probe.precision_profile != "balanced" {
        return Err(js_stack_error(&[
            "M7q1 weight probe precision profile drifted",
        ]));
    }
    pack_hex_digest(&probe.checkpoint_blake3, "M7q1 checkpoint_blake3")?;
    if probe.checkpoint_partition.f16_checkpoint_shards != 11
        || probe.checkpoint_partition.f32_rope_table_shards != 2
    {
        return Err(js_stack_error(&[
            "M7q1 weight probe checkpoint partition drifted",
        ]));
    }

    let rms_elements = m7q1_probe_elements(&[probe.rms.rows, probe.rms.width], "RMS input")?;
    require_m7q1_f32(&probe.rms.input, rms_elements, "RMS input")?;
    require_m7q1_f16(
        &probe.rms.weight_f16,
        probe.rms.width as usize,
        "RMS weight",
    )?;
    if !probe.rms.epsilon.is_finite() || probe.rms.epsilon <= 0.0 {
        return Err(js_stack_error(&["M7q1 RMS epsilon is invalid"]));
    }

    if (probe.gemv.columns != 1024 && probe.gemv.columns != 2048 && probe.gemv.columns != 3072)
        || !probe.gemv.columns.is_multiple_of(4)
    {
        return Err(js_stack_error(&[
            "M7q1 GEMV columns are outside the tiled decoder lattice",
        ]));
    }
    let gemv_matrix = m7q1_probe_elements(&[probe.gemv.rows, probe.gemv.columns], "GEMV matrix")?;
    require_m7q1_f16(&probe.gemv.matrix_f16, gemv_matrix, "GEMV matrix")?;
    require_m7q1_f32(
        &probe.gemv.vector,
        probe.gemv.columns as usize,
        "GEMV vector",
    )?;

    let linear_input = m7q1_probe_elements(
        &[probe.linear.tokens, probe.linear.input_width],
        "linear input",
    )?;
    let linear_weight = m7q1_probe_elements(
        &[probe.linear.output_width, probe.linear.input_width],
        "linear weight",
    )?;
    require_m7q1_f32(&probe.linear.input, linear_input, "linear input")?;
    require_m7q1_f16(&probe.linear.weight_f16, linear_weight, "linear weight")?;
    require_m7q1_f32(
        &probe.linear.bias,
        probe.linear.output_width as usize,
        "linear bias",
    )?;
    let input_major_linear_input = m7q1_probe_elements(
        &[
            probe.linear_input_major.tokens,
            probe.linear_input_major.input_width,
        ],
        "input-major linear input",
    )?;
    let input_major_linear_weight = m7q1_probe_elements(
        &[
            probe.linear_input_major.input_width,
            probe.linear_input_major.output_width,
        ],
        "input-major linear weight",
    )?;
    require_m7q1_f32(
        &probe.linear_input_major.input,
        input_major_linear_input,
        "input-major linear input",
    )?;
    require_m7q1_f16(
        &probe.linear_input_major.weight_f16,
        input_major_linear_weight,
        "input-major linear weight",
    )?;
    require_m7q1_f32(
        &probe.linear_input_major.bias,
        probe.linear_input_major.output_width as usize,
        "input-major linear bias",
    )?;
    if probe.linear_input_major.tokens != probe.linear.tokens
        || probe.linear_input_major.input_width != probe.linear.input_width
        || probe.linear_input_major.output_width != probe.linear.output_width
        || probe.linear_input_major.input != probe.linear.input
        || probe.linear_input_major.bias != probe.linear.bias
    {
        return Err(js_stack_error(&[
            "M7q1 input-major linear probe logical inputs drifted",
        ]));
    }
    for output in 0..probe.linear.output_width as usize {
        for input in 0..probe.linear.input_width as usize {
            let output_major =
                probe.linear.weight_f16[output * probe.linear.input_width as usize + input];
            let input_major = probe.linear_input_major.weight_f16
                [input * probe.linear.output_width as usize + output];
            if input_major != output_major {
                return Err(js_stack_error(&[
                    "M7q1 input-major linear probe weight transpose drifted",
                ]));
            }
        }
    }

    let vision_input = m7q1_probe_elements(
        &[probe.vision.patches, probe.vision.input_width],
        "vision input",
    )?;
    let vision_weight = m7q1_probe_elements(
        &[probe.vision.output_width, probe.vision.input_width],
        "vision weight",
    )?;
    require_m7q1_f32(&probe.vision.input, vision_input, "vision input")?;
    require_m7q1_f32(&probe.vision.weight, vision_weight, "vision weight")?;
    require_m7q1_f32(
        &probe.vision.bias,
        probe.vision.output_width as usize,
        "vision bias",
    )?;
    Ok(())
}

fn create_m7q1_uploaded_buffer(
    device: &JsValue,
    queue: &JsValue,
    label: &str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
) -> Result<wgpu::webgpu::GpuBuffer, JsValue> {
    let buffer = create_stack_buffer(
        device,
        label,
        bytes.len() as u64,
        buffer_usage(&[usage | wgpu::BufferUsages::COPY_DST]),
    )?;
    write_stack_buffer(queue, &buffer, bytes)?;
    Ok(buffer)
}

async fn read_m7q1_probe_output(
    buffer: &wgpu::webgpu::GpuBuffer,
    bytes: u64,
    label: &str,
) -> Result<Vec<f32>, JsValue> {
    map_stack_buffer(buffer.as_ref(), bytes).await?;
    let mapped = match read_stack_mapped(buffer.as_ref(), bytes) {
        Ok(mapped) => mapped,
        Err(error) => {
            let _ = unmap_stack_buffer(buffer.as_ref());
            return Err(error);
        }
    };
    unmap_stack_buffer(buffer.as_ref())?;
    stack_bytes_to_f32(&mapped, label)
}

fn m7q1_f32_json(values: Vec<f32>, label: &str) -> Result<Value, JsValue> {
    let mut output = Vec::with_capacity(values.len());
    for value in values {
        if !value.is_finite() {
            return Err(js_stack_error(&[
                "M7q1 ",
                label,
                " contains a nonfinite GPU result",
            ]));
        }
        output.push(Value::from(f64::from(value)));
    }
    Ok(Value::Array(output))
}

async fn run_m7q1_fp16_weight_probe_json(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fixture_json: &str,
) -> Result<String, JsValue> {
    if !device.features().contains(wgpu::Features::SHADER_F16) {
        return Err(js_stack_error(&[
            "M7q1 FP16 weight probe requires shader-f16",
        ]));
    }
    let probe = parse_m7q1_weight_probe(fixture_json)?;
    validate_m7q1_weight_probe(&probe)?;
    let raw_device = raw_stack_device(device)?;
    let raw_queue = raw_stack_queue(device, queue)?;
    let sources = canonical_stack_sources()?;
    let storage_usage = wgpu::BufferUsages::STORAGE;
    let output_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let readback_usage = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
    let uniform_usage = wgpu::BufferUsages::UNIFORM;

    let rms_layout = create_stack_bind_group_layout(raw_device, &RMS_LAYOUT_ENTRIES)?;
    let gemv_layout = create_stack_bind_group_layout(raw_device, &GEMV_LAYOUT_ENTRIES)?;
    let projection_layout = create_stack_bind_group_layout(raw_device, &PROJECTION_LAYOUT_ENTRIES)?;
    let rms_pipeline = create_stack_pipeline(
        raw_device,
        RMS_NORM_F16_KERNEL_NAME,
        &sources.rms_norm_f16_weights,
        &rms_layout,
    )?;
    let gemv_pipeline = create_stack_pipeline(
        raw_device,
        GEMV_TILED_F16_KERNEL_NAME,
        &sources.gemv_tiled_f16_weights,
        &gemv_layout,
    )?;
    let linear_pipeline = create_stack_pipeline(
        raw_device,
        PREFILL_PROJECTION_F16_KERNEL_NAME,
        &sources.linear_projection_f16_weights,
        &projection_layout,
    )?;
    let vision_pipeline = create_stack_pipeline(
        raw_device,
        PROJECTION_KERNEL_NAME,
        &sources.projection,
        &projection_layout,
    )?;

    let rms_input_bytes = m7q1_f32_bytes(&probe.rms.input);
    let rms_weight_bytes = m7q1_f16_bytes(&probe.rms.weight_f16);
    let rms_uniform_words = [
        probe.rms.rows,
        probe.rms.width,
        probe.rms.epsilon.to_bits(),
        0,
    ];
    let rms_uniform_bytes = bytemuck::cast_slice(&rms_uniform_words);
    let rms_output_bytes = (probe.rms.input.len() as u64) * 4;
    let rms_input = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-rms-input",
        &rms_input_bytes,
        storage_usage,
    )?;
    let rms_weight = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-rms-weight-f16",
        &rms_weight_bytes,
        storage_usage,
    )?;
    let rms_uniform = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-rms-uniform",
        &rms_uniform_bytes,
        uniform_usage,
    )?;
    let rms_output = create_stack_buffer(
        raw_device,
        "m7q1-rms-output",
        rms_output_bytes,
        buffer_usage(&[output_usage]),
    )?;
    let rms_readback = create_stack_buffer(
        raw_device,
        "m7q1-rms-readback",
        rms_output_bytes,
        buffer_usage(&[readback_usage]),
    )?;
    let rms_bind_group = create_stack_bind_group(
        raw_device,
        "m7q1-rms-bind-group",
        &rms_layout,
        &[
            (&rms_input, rms_input_bytes.len() as u64),
            (&rms_weight, rms_weight_bytes.len() as u64),
            (&rms_output, rms_output_bytes),
            (&rms_uniform, 16),
        ],
    )?;

    let gemv_matrix_bytes = m7q1_f16_bytes(&probe.gemv.matrix_f16);
    let gemv_vector_bytes = m7q1_f32_bytes(&probe.gemv.vector);
    let gemv_uniform_words = [probe.gemv.rows, probe.gemv.columns, 0, 0];
    let gemv_uniform_bytes = bytemuck::cast_slice(&gemv_uniform_words);
    let gemv_output_bytes = u64::from(probe.gemv.rows) * 4;
    let gemv_matrix = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-gemv-matrix-f16",
        &gemv_matrix_bytes,
        storage_usage,
    )?;
    let gemv_vector = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-gemv-vector",
        &gemv_vector_bytes,
        storage_usage,
    )?;
    let gemv_uniform = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-gemv-uniform",
        gemv_uniform_bytes,
        uniform_usage,
    )?;
    let gemv_output = create_stack_buffer(
        raw_device,
        "m7q1-gemv-output",
        gemv_output_bytes,
        buffer_usage(&[output_usage]),
    )?;
    let gemv_readback = create_stack_buffer(
        raw_device,
        "m7q1-gemv-readback",
        gemv_output_bytes,
        buffer_usage(&[readback_usage]),
    )?;
    let gemv_bind_group = create_stack_bind_group(
        raw_device,
        "m7q1-gemv-bind-group",
        &gemv_layout,
        &[
            (&gemv_matrix, gemv_matrix_bytes.len() as u64),
            (&gemv_vector, gemv_vector_bytes.len() as u64),
            (&gemv_output, gemv_output_bytes),
            (&gemv_uniform, 16),
        ],
    )?;

    let linear_input_bytes = m7q1_f32_bytes(&probe.linear.input);
    let linear_weight_bytes = m7q1_f16_bytes(&probe.linear.weight_f16);
    let linear_bias_bytes = m7q1_f32_bytes(&probe.linear.bias);
    let linear_uniform_words = [
        probe.linear.tokens,
        probe.linear.input_width,
        probe.linear.output_width,
        0,
    ];
    let linear_uniform_bytes = bytemuck::cast_slice(&linear_uniform_words);
    let linear_output_bytes =
        u64::from(probe.linear.tokens) * u64::from(probe.linear.output_width) * 4;
    let linear_input = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-linear-input",
        &linear_input_bytes,
        storage_usage,
    )?;
    let linear_weight = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-linear-weight-f16",
        &linear_weight_bytes,
        storage_usage,
    )?;
    let linear_bias = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-linear-bias",
        &linear_bias_bytes,
        storage_usage,
    )?;
    let linear_uniform = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-linear-uniform",
        linear_uniform_bytes,
        uniform_usage,
    )?;
    let linear_output = create_stack_buffer(
        raw_device,
        "m7q1-linear-output",
        linear_output_bytes,
        buffer_usage(&[output_usage]),
    )?;
    let linear_readback = create_stack_buffer(
        raw_device,
        "m7q1-linear-readback",
        linear_output_bytes,
        buffer_usage(&[readback_usage]),
    )?;
    let linear_bind_group = create_stack_bind_group(
        raw_device,
        "m7q1-linear-bind-group",
        &projection_layout,
        &[
            (&linear_input, linear_input_bytes.len() as u64),
            (&linear_weight, linear_weight_bytes.len() as u64),
            (&linear_bias, linear_bias_bytes.len() as u64),
            (&linear_output, linear_output_bytes),
            (&linear_uniform, 16),
        ],
    )?;
    let linear_input_major_weight_bytes = m7q1_f16_bytes(&probe.linear_input_major.weight_f16);
    let linear_input_major_uniform_words = [
        probe.linear_input_major.tokens,
        probe.linear_input_major.input_width,
        probe.linear_input_major.output_width,
        1,
    ];
    let linear_input_major_uniform_bytes = bytemuck::cast_slice(&linear_input_major_uniform_words);
    let linear_input_major_weight = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-linear-input-major-weight-f16",
        &linear_input_major_weight_bytes,
        storage_usage,
    )?;
    let linear_input_major_uniform = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-linear-input-major-uniform",
        linear_input_major_uniform_bytes,
        uniform_usage,
    )?;
    let linear_input_major_output = create_stack_buffer(
        raw_device,
        "m7q1-linear-input-major-output",
        linear_output_bytes,
        buffer_usage(&[output_usage]),
    )?;
    let linear_input_major_readback = create_stack_buffer(
        raw_device,
        "m7q1-linear-input-major-readback",
        linear_output_bytes,
        buffer_usage(&[readback_usage]),
    )?;
    let linear_input_major_bind_group = create_stack_bind_group(
        raw_device,
        "m7q1-linear-input-major-bind-group",
        &projection_layout,
        &[
            (&linear_input, linear_input_bytes.len() as u64),
            (
                &linear_input_major_weight,
                linear_input_major_weight_bytes.len() as u64,
            ),
            (&linear_bias, linear_bias_bytes.len() as u64),
            (&linear_input_major_output, linear_output_bytes),
            (&linear_input_major_uniform, 16),
        ],
    )?;

    let vision_input_bytes = m7q1_f32_bytes(&probe.vision.input);
    let vision_weight_bytes = m7q1_f32_bytes(&probe.vision.weight);
    let vision_bias_bytes = m7q1_f32_bytes(&probe.vision.bias);
    let vision_uniform_words = [
        probe.vision.patches,
        probe.vision.input_width,
        probe.vision.output_width,
        0,
    ];
    let vision_uniform_bytes = bytemuck::cast_slice(&vision_uniform_words);
    let vision_output_bytes =
        u64::from(probe.vision.patches) * u64::from(probe.vision.output_width) * 4;
    let vision_input = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-vision-input",
        &vision_input_bytes,
        storage_usage,
    )?;
    let vision_weight = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-vision-weight-f32",
        &vision_weight_bytes,
        storage_usage,
    )?;
    let vision_bias = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-vision-bias",
        &vision_bias_bytes,
        storage_usage,
    )?;
    let vision_uniform = create_m7q1_uploaded_buffer(
        raw_device,
        &raw_queue,
        "m7q1-vision-uniform",
        vision_uniform_bytes,
        uniform_usage,
    )?;
    let vision_output = create_stack_buffer(
        raw_device,
        "m7q1-vision-output",
        vision_output_bytes,
        buffer_usage(&[output_usage]),
    )?;
    let vision_readback = create_stack_buffer(
        raw_device,
        "m7q1-vision-readback",
        vision_output_bytes,
        buffer_usage(&[readback_usage]),
    )?;
    let vision_bind_group = create_stack_bind_group(
        raw_device,
        "m7q1-vision-bind-group",
        &projection_layout,
        &[
            (&vision_input, vision_input_bytes.len() as u64),
            (&vision_weight, vision_weight_bytes.len() as u64),
            (&vision_bias, vision_bias_bytes.len() as u64),
            (&vision_output, vision_output_bytes),
            (&vision_uniform, 16),
        ],
    )?;

    let encoder = create_stack_encoder(raw_device, "m7q1-fp16-weight-probe-encoder")?;
    encode_stack_pass(
        &encoder,
        &rms_pipeline,
        &rms_bind_group,
        [probe.rms.rows.div_ceil(64), 1, 1],
        &[0, 0],
    )?;
    encode_stack_pass(
        &encoder,
        &gemv_pipeline,
        &gemv_bind_group,
        [probe.gemv.rows.div_ceil(8), 1, 1],
        &[0],
    )?;
    encode_stack_pass(
        &encoder,
        &linear_pipeline,
        &linear_bind_group,
        [
            probe.linear.output_width.div_ceil(
                DecoderWeightStorage::F16.linear_projection_output_columns_per_workgroup(),
            ),
            probe.linear.tokens.div_ceil(LINEAR_PROJECTION_TILE),
            1,
        ],
        &[0],
    )?;
    encode_stack_pass(
        &encoder,
        &linear_pipeline,
        &linear_input_major_bind_group,
        [
            probe.linear_input_major.output_width.div_ceil(
                DecoderWeightStorage::F16.linear_projection_output_columns_per_workgroup(),
            ),
            probe
                .linear_input_major
                .tokens
                .div_ceil(LINEAR_PROJECTION_TILE),
            1,
        ],
        &[0],
    )?;
    encode_stack_pass(
        &encoder,
        &vision_pipeline,
        &vision_bind_group,
        [
            probe.vision.output_width.div_ceil(LINEAR_PROJECTION_TILE),
            probe.vision.patches.div_ceil(LINEAR_PROJECTION_TILE),
            1,
        ],
        &[0],
    )?;
    for (output, readback, bytes) in [
        (&rms_output, &rms_readback, rms_output_bytes),
        (&gemv_output, &gemv_readback, gemv_output_bytes),
        (&linear_output, &linear_readback, linear_output_bytes),
        (
            &linear_input_major_output,
            &linear_input_major_readback,
            linear_output_bytes,
        ),
        (&vision_output, &vision_readback, vision_output_bytes),
    ] {
        encode_stack_copy(&encoder, output, 0, readback, 0, bytes)?;
    }
    submit_stack_encoder(&raw_queue, &encoder)?;
    await_stack_queue_completion(&raw_queue).await?;

    let rms_values =
        read_m7q1_probe_output(&rms_readback, rms_output_bytes, "M7q1 RMS output").await?;
    let gemv_values =
        read_m7q1_probe_output(&gemv_readback, gemv_output_bytes, "M7q1 GEMV output").await?;
    let linear_values =
        read_m7q1_probe_output(&linear_readback, linear_output_bytes, "M7q1 linear output").await?;
    let linear_input_major_values = read_m7q1_probe_output(
        &linear_input_major_readback,
        linear_output_bytes,
        "M7q1 input-major linear output",
    )
    .await?;
    let vision_values =
        read_m7q1_probe_output(&vision_readback, vision_output_bytes, "M7q1 vision output").await?;
    let mut resources = Map::new();
    resources.insert(
        "f16_checkpoint_shards".to_owned(),
        Value::from(probe.checkpoint_partition.f16_checkpoint_shards),
    );
    resources.insert("f32_checkpoint_shards".to_owned(), Value::from(0u64));
    resources.insert(
        "f32_rope_table_shards".to_owned(),
        Value::from(probe.checkpoint_partition.f32_rope_table_shards),
    );
    resources.insert(
        "checkpoint_weight_element_bytes".to_owned(),
        Value::from(2u64),
    );
    resources.insert("rope_table_element_bytes".to_owned(), Value::from(4u64));
    resources.insert("activation_element_bytes".to_owned(), Value::from(4u64));
    resources.insert("cache_element_bytes".to_owned(), Value::from(4u64));
    resources.insert("output_element_bytes".to_owned(), Value::from(4u64));
    resources.insert("accumulator_dtype".to_owned(), Value::from("f32"));

    let mut decoder_pipeline_trace = Vec::new();
    decoder_pipeline_trace.push(Value::from(RMS_NORM_F16_KERNEL_NAME));
    decoder_pipeline_trace.push(Value::from(GEMV_TILED_F16_KERNEL_NAME));
    decoder_pipeline_trace.push(Value::from(PREFILL_PROJECTION_F16_KERNEL_NAME));
    let mut vision_pipeline_trace = Vec::new();
    vision_pipeline_trace.push(Value::from(PROJECTION_KERNEL_NAME));
    let mut device_features = Vec::new();
    device_features.push(Value::from("shader-f16"));

    let mut outputs = Map::new();
    outputs.insert("rms".to_owned(), m7q1_f32_json(rms_values, "RMS output")?);
    outputs.insert(
        "gemv".to_owned(),
        m7q1_f32_json(gemv_values, "GEMV output")?,
    );
    outputs.insert(
        "linear".to_owned(),
        m7q1_f32_json(linear_values, "linear output")?,
    );
    outputs.insert(
        "linear_input_major".to_owned(),
        m7q1_f32_json(linear_input_major_values, "input-major linear output")?,
    );
    outputs.insert(
        "vision".to_owned(),
        m7q1_f32_json(vision_values, "vision output")?,
    );

    let mut effects = Map::new();
    effects.insert("buffer_creations".to_owned(), Value::from(26u64));
    effects.insert("queue_writes".to_owned(), Value::from(16u64));
    effects.insert("submissions".to_owned(), Value::from(1u64));

    let mut report = Map::new();
    report.insert("precision_profile".to_owned(), Value::from("balanced"));
    report.insert("device_features".to_owned(), Value::Array(device_features));
    report.insert(
        "checkpoint_blake3".to_owned(),
        Value::from(probe.checkpoint_blake3),
    );
    report.insert("resource_plan".to_owned(), Value::Object(resources));
    report.insert(
        "decoder_pipeline_trace".to_owned(),
        Value::Array(decoder_pipeline_trace),
    );
    report.insert(
        "vision_pipeline_trace".to_owned(),
        Value::Array(vision_pipeline_trace),
    );
    report.insert("vision_weight_storage".to_owned(), Value::from("f32"));
    report.insert("outputs".to_owned(), Value::Object(outputs));
    report.insert("effects".to_owned(), Value::Object(effects));
    json_text(Value::Object(report))
}

fn require_weight_storage_device_feature(
    device: &wgpu::Device,
    weight_resource_plan: &pvlc_runtime_core::DecoderWeightResourcePlan,
) -> Result<(), JsValue> {
    if weight_resource_plan.requires_shader_f16()
        && !device.features().contains(wgpu::Features::SHADER_F16)
    {
        return Err(js_stack_error(&[
            "decoder stack balanced weight storage requires the shader-f16 device feature",
        ]));
    }
    Ok(())
}

fn prepare_stack_begin(
    owner: &AsyncSessionOwner<BrowserDecoderStackSession>,
    device: &wgpu::Device,
    descriptor_json: &str,
    pack: &js_sys::Uint8Array,
    key_cache: &js_sys::Uint8Array,
    value_cache: &js_sys::Uint8Array,
) -> Result<PreparedStackBegin, JsValue> {
    check_stack_admission(owner)?;
    let parsed = parse_stack_descriptor_json(descriptor_json)?;
    let prefill_capable = parsed.prefill_tokens > 0;
    let pack_bytes = stack_pack_to_bytes(pack)?;
    let weight_pack = parse_stack_weight_pack(
        &pack_bytes,
        parsed.prefix_tokens,
        parsed.cache_capacity,
        prefill_capable,
    )?;
    let logits_capable = match (parsed.vocab_size, weight_pack.final_norm_weight.is_some()) {
        (Some(_), true) => true,
        (None, false) => false,
        _ => {
            return Err(js_stack_error(&[
                "decoder stack session logits capability drifted between the descriptor and the weight pack",
            ]));
        }
    };
    // Shard payloads are owned by the parsed pack, so the monolithic input
    // copy can be released before any optional F32 planner conversion.
    drop(pack_bytes);

    let key_cache_bytes = stack_uint8_to_bytes(key_cache)?;
    let value_cache_bytes = stack_uint8_to_bytes(value_cache)?;
    let key_f32 = require_stack_cache_operands(
        &key_cache_bytes,
        "initial K cache",
        parsed.prefix_tokens,
        parsed.cache_capacity,
    )?;
    let value_f32 = require_stack_cache_operands(
        &value_cache_bytes,
        "initial V cache",
        parsed.prefix_tokens,
        parsed.cache_capacity,
    )?;
    let cache_plane_elements = u64::from(parsed.cache_capacity)
        .checked_mul(u64::from(PINNED_STACK_KEY_VALUE_WIDTH))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| js_stack_error(&["initial cache plane element count overflowed"]))?;
    let admitted_prefix_tokens = if prefill_capable {
        parsed
            .prefill_tokens
            .min(parsed.cache_capacity.saturating_sub(1))
    } else {
        parsed.prefix_tokens
    };
    let kv_plan = DecoderKvSessionDescriptor {
        query_heads: parsed.query_heads,
        key_value_heads: parsed.key_value_heads,
        head_dim: parsed.head_dim,
        prefix_tokens: admitted_prefix_tokens,
        cache_capacity: parsed.cache_capacity,
        key_cache: key_f32[..cache_plane_elements].as_ref(),
        value_cache: value_f32[..cache_plane_elements].as_ref(),
    }
    .plan()
    .map_err(|error| {
        js_stack_error(&[
            "invalid decoder stack session descriptor geometry or initial cache operands: ",
            &error.to_string(),
        ])
    })?;
    let admitted_prefill_tokens = if prefill_capable {
        parsed.prefill_tokens
    } else {
        1
    };
    let geometry = DecoderStackGeometryDescriptor {
        layers: pvlc_runtime_core::PINNED_DECODER_LAYERS,
        hidden_size: parsed.hidden_size,
        intermediate_size: parsed.intermediate_size,
        query_heads: parsed.query_heads,
        key_value_heads: parsed.key_value_heads,
        head_dim: parsed.head_dim,
        rms_norm_epsilon: parsed.rms_norm_epsilon,
        cache_capacity: parsed.cache_capacity,
    };

    let (stack_plan, prefill_plan, lm_head_plan) = match weight_pack.weight_storage {
        DecoderWeightStorage::F32 => {
            // Fidelity keeps the accepted operand-validating planners. This
            // branch is intentionally unchanged semantically; only balanced
            // storage uses the payload-free authority below.
            let norm1_f32 =
                stack_bytes_to_f32(&weight_pack.norm1_weight, "weights.input_layernorm")?;
            let q_f32 = stack_bytes_to_f32(&weight_pack.q_weight, "weights.q_proj")?;
            let k_f32 = stack_bytes_to_f32(&weight_pack.k_weight, "weights.k_proj")?;
            let v_f32 = stack_bytes_to_f32(&weight_pack.v_weight, "weights.v_proj")?;
            let o_f32 = stack_bytes_to_f32(&weight_pack.o_weight, "weights.o_proj")?;
            let cos_f32 = stack_bytes_to_f32(&weight_pack.rope_cos, "weights.mrope_cos")?;
            let sin_f32 = stack_bytes_to_f32(&weight_pack.rope_sin, "weights.mrope_sin")?;
            let norm2_f32 = stack_bytes_to_f32(
                &weight_pack.norm2_weight,
                "weights.post_attention_layernorm",
            )?;
            let gate_f32 = stack_bytes_to_f32(&weight_pack.gate_weight, "weights.gate_proj")?;
            let up_f32 = stack_bytes_to_f32(&weight_pack.up_weight, "weights.up_proj")?;
            let down_f32 = stack_bytes_to_f32(&weight_pack.down_weight, "weights.down_proj")?;
            let stack_plan = DecoderStackDescriptor {
                layers: geometry.layers,
                hidden_size: geometry.hidden_size,
                intermediate_size: geometry.intermediate_size,
                query_heads: geometry.query_heads,
                key_value_heads: geometry.key_value_heads,
                head_dim: geometry.head_dim,
                rms_norm_epsilon: geometry.rms_norm_epsilon,
                norm1_weight: norm1_f32.as_slice(),
                q_weight: q_f32.as_slice(),
                k_weight: k_f32.as_slice(),
                v_weight: v_f32.as_slice(),
                o_weight: o_f32.as_slice(),
                mrope_cos: cos_f32.as_slice(),
                mrope_sin: sin_f32.as_slice(),
                norm2_weight: norm2_f32.as_slice(),
                gate_weight: gate_f32.as_slice(),
                up_weight: up_f32.as_slice(),
                down_weight: down_f32.as_slice(),
                cache_capacity: geometry.cache_capacity,
            }
            .plan()
            .map_err(|error| {
                js_stack_error(&[
                    "invalid decoder stack session descriptor geometry or weight operands: ",
                    &error.to_string(),
                ])
            })?;
            let prefill_plan = DecoderStackPrefillDescriptor {
                layers: geometry.layers,
                hidden_size: geometry.hidden_size,
                intermediate_size: geometry.intermediate_size,
                query_heads: geometry.query_heads,
                key_value_heads: geometry.key_value_heads,
                head_dim: geometry.head_dim,
                rms_norm_epsilon: geometry.rms_norm_epsilon,
                norm1_weight: norm1_f32.as_slice(),
                q_weight: q_f32.as_slice(),
                k_weight: k_f32.as_slice(),
                v_weight: v_f32.as_slice(),
                o_weight: o_f32.as_slice(),
                mrope_cos: cos_f32.as_slice(),
                mrope_sin: sin_f32.as_slice(),
                norm2_weight: norm2_f32.as_slice(),
                gate_weight: gate_f32.as_slice(),
                up_weight: up_f32.as_slice(),
                down_weight: down_f32.as_slice(),
                cache_capacity: geometry.cache_capacity,
                tokens: admitted_prefill_tokens,
            }
            .plan()
            .map_err(|error| {
                js_stack_error(&[
                    "invalid decoder stack session prefill geometry or weight operands: ",
                    &error.to_string(),
                ])
            })?;
            let lm_head_plan = if logits_capable {
                let (Some(final_norm_bytes), Some(lm_head_bytes)) = (
                    weight_pack.final_norm_weight.as_ref(),
                    weight_pack.lm_head_weight.as_ref(),
                ) else {
                    return Err(js_stack_error(&[
                        "decoder stack session logits weight pack shards are missing",
                    ]));
                };
                let final_norm_f32 =
                    stack_bytes_to_f32(final_norm_bytes, "weights.final_layernorm")?;
                let lm_head_f32 = stack_bytes_to_f32(lm_head_bytes, "weights.lm_head")?;
                Some(
                    DecoderLmHeadDescriptor::pinned(
                        final_norm_f32.as_slice(),
                        lm_head_f32.as_slice(),
                    )
                    .plan()
                    .map_err(|error| {
                        js_stack_error(&[
                            "invalid decoder stack session logits geometry or weight operands: ",
                            &error.to_string(),
                        ])
                    })?,
                )
            } else {
                None
            };
            (stack_plan, prefill_plan, lm_head_plan)
        }
        DecoderWeightStorage::F16 => {
            let stack_plan = geometry.plan().map_err(|error| {
                js_stack_error(&[
                    "invalid balanced decoder stack session geometry: ",
                    &error.to_string(),
                ])
            })?;
            let prefill_plan = geometry
                .plan_prefill(admitted_prefill_tokens)
                .map_err(|error| {
                    js_stack_error(&[
                        "invalid balanced decoder stack prefill geometry: ",
                        &error.to_string(),
                    ])
                })?;
            let lm_head_plan = if logits_capable {
                Some(
                    DecoderLmHeadGeometryDescriptor::pinned()
                        .plan()
                        .map_err(|error| {
                            js_stack_error(&[
                                "invalid balanced decoder stack logits geometry: ",
                                &error.to_string(),
                            ])
                        })?,
                )
            } else {
                None
            };
            (stack_plan, prefill_plan, lm_head_plan)
        }
    };

    let f32_rope_table_bytes = checked_u64_bytes(
        stack_plan.layer_plan.attention_block.rope_elements,
        "decoder rope table",
    )?;
    let weight_resource_plan = DecoderWeightResourceDescriptor {
        layers: stack_plan.layers,
        f32_layer_weight_stride_bytes: stack_plan.weight_stride_bytes,
        f32_rope_table_bytes,
        f32_final_norm_weight_bytes: lm_head_plan
            .as_ref()
            .map(|plan| plan.final_norm_weight_bytes),
        f32_lm_head_weight_bytes: lm_head_plan.as_ref().map(|plan| plan.lm_head_weight_bytes),
        storage: weight_pack.weight_storage,
    }
    .plan()
    .map_err(|error| {
        js_stack_error(&[
            "invalid decoder stack physical weight resource plan: ",
            &error.to_string(),
        ])
    })?;
    require_weight_storage_device_feature(device, &weight_resource_plan)?;

    let actual_layer_weight_bytes = [
        weight_pack.norm1_weight.len() as u64,
        weight_pack.q_weight.len() as u64,
        weight_pack.k_weight.len() as u64,
        weight_pack.v_weight.len() as u64,
        weight_pack.o_weight.len() as u64,
        weight_pack.norm2_weight.len() as u64,
        weight_pack.gate_weight.len() as u64,
        weight_pack.up_weight.len() as u64,
        weight_pack.down_weight.len() as u64,
    ];
    if actual_layer_weight_bytes != weight_resource_plan.layer_weight_bulk_bytes
        || weight_pack.rope_cos.len() as u64 != weight_resource_plan.rope_table_bytes
        || weight_pack.rope_sin.len() as u64 != weight_resource_plan.rope_table_bytes
        || weight_pack
            .final_norm_weight
            .as_ref()
            .map(|bytes| bytes.len() as u64)
            != weight_resource_plan.final_norm_weight_bytes
        || weight_pack
            .lm_head_weight
            .as_ref()
            .map(|bytes| bytes.len() as u64)
            != weight_resource_plan.lm_head_weight_bytes
    {
        return Err(js_stack_error(&[
            "decoder stack authenticated shard bytes drifted from the physical resource plan",
        ]));
    }

    let checkpoint_blake3 = weight_pack.checkpoint_blake3;
    let sources = canonical_stack_sources()?;
    let operands = StackBeginOperands {
        upload_initial_cache: true,
        key_cache_bytes,
        value_cache_bytes,
        norm1_weight_bytes: weight_pack.norm1_weight,
        q_weight_bytes: weight_pack.q_weight,
        k_weight_bytes: weight_pack.k_weight,
        v_weight_bytes: weight_pack.v_weight,
        o_weight_bytes: weight_pack.o_weight,
        rope_cos_bytes: weight_pack.rope_cos,
        rope_sin_bytes: weight_pack.rope_sin,
        norm2_weight_bytes: weight_pack.norm2_weight,
        gate_weight_bytes: weight_pack.gate_weight,
        up_weight_bytes: weight_pack.up_weight,
        down_weight_bytes: weight_pack.down_weight,
        final_norm_weight_bytes: weight_pack.final_norm_weight,
        lm_head_weight_bytes: weight_pack.lm_head_weight,
    };
    Ok(PreparedStackBegin {
        kv_plan,
        stack_plan,
        weight_resource_plan,
        checkpoint_blake3,
        prefill_plan,
        lm_head_plan,
        prefill_capable,
        operands,
        sources,
    })
}

fn prepare_resident_stack_begin(
    owner: &AsyncSessionOwner<BrowserDecoderStackSession>,
    resident_weight_cache: &SharedDecoderStackResidentWeightCache,
    device: &wgpu::Device,
    descriptor_json: &str,
    rope_cos: &js_sys::Uint8Array,
    rope_sin: &js_sys::Uint8Array,
) -> Result<PreparedStackBegin, JsValue> {
    check_stack_admission(owner)?;
    let parsed = parse_stack_descriptor_json(descriptor_json)?;
    if parsed.prefix_tokens != 0 || parsed.prefill_tokens == 0 {
        return Err(js_stack_error(&[
            "resident decoder begin requires a zero-prefix prefill descriptor",
        ]));
    }
    if parsed.vocab_size != Some(PINNED_STACK_VOCAB_SIZE) {
        return Err(js_stack_error(&[
            "resident decoder begin requires the pinned logits vocabulary",
        ]));
    }
    let resident = resident_weight_cache.get().ok_or_else(|| {
        js_stack_error(&[
            "resident decoder weights are not ready; run one authenticated full begin first",
        ])
    })?;
    let geometry = DecoderStackGeometryDescriptor {
        layers: pvlc_runtime_core::PINNED_DECODER_LAYERS,
        hidden_size: parsed.hidden_size,
        intermediate_size: parsed.intermediate_size,
        query_heads: parsed.query_heads,
        key_value_heads: parsed.key_value_heads,
        head_dim: parsed.head_dim,
        rms_norm_epsilon: parsed.rms_norm_epsilon,
        cache_capacity: parsed.cache_capacity,
    };
    let stack_plan = geometry.plan().map_err(|error| {
        js_stack_error(&[
            "invalid resident decoder stack geometry: ",
            &error.to_string(),
        ])
    })?;
    let prefill_plan = geometry
        .plan_prefill(parsed.prefill_tokens)
        .map_err(|error| {
            js_stack_error(&[
                "invalid resident decoder prefill geometry: ",
                &error.to_string(),
            ])
        })?;
    let lm_head_plan = DecoderLmHeadGeometryDescriptor::pinned()
        .plan()
        .map_err(|error| {
            js_stack_error(&[
                "invalid resident decoder logits geometry: ",
                &error.to_string(),
            ])
        })?;
    let f32_rope_table_bytes = checked_u64_bytes(
        stack_plan.layer_plan.attention_block.rope_elements,
        "resident decoder rope table",
    )?;
    let weight_resource_plan = DecoderWeightResourceDescriptor {
        layers: stack_plan.layers,
        f32_layer_weight_stride_bytes: stack_plan.weight_stride_bytes,
        f32_rope_table_bytes,
        f32_final_norm_weight_bytes: Some(lm_head_plan.final_norm_weight_bytes),
        f32_lm_head_weight_bytes: Some(lm_head_plan.lm_head_weight_bytes),
        storage: DecoderWeightStorage::F16,
    }
    .plan()
    .map_err(|error| {
        js_stack_error(&[
            "invalid resident decoder physical weight plan: ",
            &error.to_string(),
        ])
    })?;
    require_weight_storage_device_feature(device, &weight_resource_plan)?;
    let expected_key = resident_weight_key(Some(resident.checkpoint_blake3), &weight_resource_plan)
        .ok_or_else(|| js_stack_error(&["resident decoder cache key is unavailable"]))?;
    if resident.key != expected_key {
        return Err(js_stack_error(&[
            "resident decoder weights do not match the requested model geometry",
        ]));
    }
    let rope_cos_bytes = stack_uint8_to_bytes(rope_cos)?;
    let rope_sin_bytes = stack_uint8_to_bytes(rope_sin)?;
    for (bytes, label) in [
        (&rope_cos_bytes, "resident decoder cosine table"),
        (&rope_sin_bytes, "resident decoder sine table"),
    ] {
        if bytes.len() as u64 != weight_resource_plan.rope_table_bytes {
            return Err(js_stack_error(&[
                label,
                " byte length does not match the requested cache capacity",
            ]));
        }
        let values = stack_bytes_to_f32(bytes, label)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(js_stack_error(&[label, " contains a nonfinite f32 value"]));
        }
    }
    let empty_cache_elements = usize::try_from(
        u64::from(parsed.cache_capacity)
            .checked_mul(u64::from(PINNED_STACK_KEY_VALUE_WIDTH))
            .ok_or_else(|| js_stack_error(&["resident decoder cache plane size overflowed"]))?,
    )
    .map_err(|_| js_stack_error(&["resident decoder cache plane is not addressable"]))?;
    let mut empty_cache = Vec::with_capacity(empty_cache_elements);
    empty_cache.resize(empty_cache_elements, 0.0_f32);
    let kv_plan = DecoderKvSessionDescriptor {
        query_heads: parsed.query_heads,
        key_value_heads: parsed.key_value_heads,
        head_dim: parsed.head_dim,
        prefix_tokens: parsed
            .prefill_tokens
            .min(parsed.cache_capacity.saturating_sub(1)),
        cache_capacity: parsed.cache_capacity,
        key_cache: empty_cache.as_slice(),
        value_cache: empty_cache.as_slice(),
    }
    .plan()
    .map_err(|error| {
        js_stack_error(&[
            "invalid resident decoder cache geometry: ",
            &error.to_string(),
        ])
    })?;
    Ok(PreparedStackBegin {
        kv_plan,
        stack_plan,
        weight_resource_plan,
        checkpoint_blake3: Some(resident.checkpoint_blake3),
        prefill_plan,
        lm_head_plan: Some(lm_head_plan),
        prefill_capable: true,
        operands: StackBeginOperands {
            upload_initial_cache: false,
            key_cache_bytes: Vec::new(),
            value_cache_bytes: Vec::new(),
            norm1_weight_bytes: Vec::new(),
            q_weight_bytes: Vec::new(),
            k_weight_bytes: Vec::new(),
            v_weight_bytes: Vec::new(),
            o_weight_bytes: Vec::new(),
            rope_cos_bytes,
            rope_sin_bytes,
            norm2_weight_bytes: Vec::new(),
            gate_weight_bytes: Vec::new(),
            up_weight_bytes: Vec::new(),
            down_weight_bytes: Vec::new(),
            final_norm_weight_bytes: None,
            lm_head_weight_bytes: None,
        },
        sources: canonical_stack_sources()?,
    })
}

async fn run_begin(
    owner: &AsyncSessionOwner<BrowserDecoderStackSession>,
    resident_weight_cache: &SharedDecoderStackResidentWeightCache,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    kv_plan: pvlc_runtime_core::DecoderKvSessionPlan,
    stack_plan: pvlc_runtime_core::DecoderStackPlan,
    weight_resource_plan: pvlc_runtime_core::DecoderWeightResourcePlan,
    checkpoint_blake3: Option<[u8; 32]>,
    prefill_plan: pvlc_runtime_core::DecoderStackPrefillPlan,
    lm_head_plan: Option<pvlc_runtime_core::DecoderLmHeadPlan>,
    prefill_capable: bool,
    operands: StackBeginOperands,
    sources: StackShaderSources,
) -> Result<String, JsValue> {
    validate_stack_capabilities(
        device,
        &kv_plan,
        &stack_plan,
        &weight_resource_plan,
        &prefill_plan,
        prefill_capable,
        lm_head_plan.as_ref(),
    )?;
    let mut ledger: Vec<ScopeKind> = Vec::new();
    for scope in CHECKED_SCOPES {
        if let Err(error) = push_stack_error_scope(device, scope).await {
            ledger.push(scope);
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            return Err(drain_appended_message(error, captures, failures));
        }
        ledger.push(scope);
    }
    let raw_device = raw_stack_device(device)?;
    let raw_queue = raw_stack_queue(device, queue)?;
    let digests = stack_shader_digests(&sources);
    let logits_capable = lm_head_plan.is_some();
    let expected_resident_key = resident_weight_key(checkpoint_blake3, &weight_resource_plan);
    let initial_cache_uploaded = operands.upload_initial_cache;
    let resident_weights = resident_weight_cache
        .get()
        .filter(|weights| Some(weights.key.as_str()) == expected_resident_key.as_deref());
    let (session, resident_candidate) = match BrowserDecoderStackSession::create(
        raw_device,
        kv_plan,
        stack_plan,
        weight_resource_plan,
        checkpoint_blake3,
        prefill_plan,
        &sources,
        prefill_capable,
        lm_head_plan,
        resident_weights.as_ref(),
    ) {
        Ok(created) => created,
        Err(error) => {
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    if let Err(error) =
        session.upload_initial_operands(&raw_queue, &operands, resident_weights.is_none())
    {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        return Err(drain_appended_message(error, captures, failures));
    }
    let generation = match owner.begin(session) {
        Ok(generation) => generation,
        Err(_busy) => {
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            return Err(drain_appended_message(
                js_stack_error(&["decoder stack session is already active"]),
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
            wait_stack_owner_idle(owner).await;
            stale = false;
        }
        match pop_stack_error_scope(device, scope).await {
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
        return Err(js_stack_error(&[
            "decoder stack session begin is stale: its generation was cancelled",
        ]));
    }
    if !failures.is_empty() || !captures.is_empty() {
        poison_stored_session(owner);
        return Err(captured_failure_message(captures, failures));
    }
    if let Some(candidate) = resident_candidate {
        resident_weight_cache.replace(candidate);
    }
    mark_session_ready(owner);
    creation_diagnostics_json(
        &kv_plan,
        &stack_plan,
        &digests,
        &sources,
        prefill_capable,
        logits_capable,
        resident_weights.is_some(),
        initial_cache_uploaded,
    )
}

async fn run_step(
    owner: &AsyncSessionOwner<BrowserDecoderStackSession>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lease: crate::AsyncSessionLease,
    mut session: BrowserDecoderStackSession,
    operands: StackStepOperands,
) -> Result<JsValue, JsValue> {
    let StackStepOperands {
        transition,
        step_plan,
        hidden_bytes,
    } = operands;
    session.poisoned = true;
    let mut ledger: Vec<ScopeKind> = Vec::new();
    for scope in CHECKED_SCOPES {
        if let Err(error) = push_stack_error_scope(device, scope).await {
            ledger.push(scope);
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            restore_stack_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
        ledger.push(scope);
    }
    let raw_device = match raw_stack_device(device) {
        Ok(raw) => raw,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    let raw_queue = match raw_stack_queue(device, queue) {
        Ok(raw) => raw,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    let hidden_row_bytes = session.stack_plan.hidden_stride_bytes;
    if let Err(error) = session.encode_step(
        raw_device,
        &raw_queue,
        &transition,
        &step_plan,
        &hidden_bytes,
    ) {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if let Err(error) = await_stack_queue_completion(&raw_queue).await {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if let Err(error) =
        map_stack_buffer(session.stack_readback_buffer.as_ref(), hidden_row_bytes).await
    {
        let _ = unmap_stack_buffer(session.stack_readback_buffer.as_ref());
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if owner.generation() != Some(lease.generation()) {
        let _ = unmap_stack_buffer(session.stack_readback_buffer.as_ref());
        wait_stack_owner_idle(owner).await;
        let _ = drain_stack_error_scopes(device, &mut ledger).await;
        let _ = owner.complete(lease, session, CompletionAction::Finish);
        return Err(js_stack_error(&[
            "decoder stack session step is stale: its generation was cancelled",
        ]));
    }
    let output = match read_stack_mapped(session.stack_readback_buffer.as_ref(), hidden_row_bytes) {
        Ok(output) => output,
        Err(error) => {
            let _ = unmap_stack_buffer(session.stack_readback_buffer.as_ref());
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            restore_stack_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    if let Err(error) = unmap_stack_buffer(session.stack_readback_buffer.as_ref()) {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    session.cache_tokens = transition.cache_tokens_after;
    session.poisoned = false;
    let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
    if !failures.is_empty() || !captures.is_empty() {
        session.poisoned = true;
        restore_stack_session(owner, lease, session);
        return Err(captured_failure_message(captures, failures));
    }
    let diagnostics = match step_diagnostics_json(&session, &transition) {
        Ok(diagnostics) => diagnostics,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    let _ = owner.complete(lease, session, CompletionAction::Restore);
    stack_step_result(output, diagnostics)
}

async fn run_prefill(
    owner: &AsyncSessionOwner<BrowserDecoderStackSession>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lease: crate::AsyncSessionLease,
    mut session: BrowserDecoderStackSession,
    operands: StackPrefillOperands,
) -> Result<JsValue, JsValue> {
    let StackPrefillOperands {
        prefill_plan,
        hidden_bytes,
    } = operands;
    session.poisoned = true;
    let mut ledger: Vec<ScopeKind> = Vec::new();
    for scope in CHECKED_SCOPES {
        if let Err(error) = push_stack_error_scope(device, scope).await {
            ledger.push(scope);
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            restore_stack_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
        ledger.push(scope);
    }
    let raw_device = match raw_stack_device(device) {
        Ok(raw) => raw,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    let raw_queue = match raw_stack_queue(device, queue) {
        Ok(raw) => raw,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    let hidden_row_bytes = session.stack_plan.hidden_stride_bytes;
    if let Err(error) = session.encode_prefill(raw_device, &raw_queue, &prefill_plan, &hidden_bytes)
    {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if let Err(error) = await_stack_queue_completion(&raw_queue).await {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if let Err(error) =
        map_stack_buffer(session.stack_readback_buffer.as_ref(), hidden_row_bytes).await
    {
        let _ = unmap_stack_buffer(session.stack_readback_buffer.as_ref());
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if owner.generation() != Some(lease.generation()) {
        let _ = unmap_stack_buffer(session.stack_readback_buffer.as_ref());
        wait_stack_owner_idle(owner).await;
        let _ = drain_stack_error_scopes(device, &mut ledger).await;
        let _ = owner.complete(lease, session, CompletionAction::Finish);
        return Err(js_stack_error(&[
            "decoder stack session prefill is stale: its generation was cancelled",
        ]));
    }
    let output = match read_stack_mapped(session.stack_readback_buffer.as_ref(), hidden_row_bytes) {
        Ok(output) => output,
        Err(error) => {
            let _ = unmap_stack_buffer(session.stack_readback_buffer.as_ref());
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            restore_stack_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    if let Err(error) = unmap_stack_buffer(session.stack_readback_buffer.as_ref()) {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    session.cache_tokens = prefill_plan.tokens;
    session.poisoned = false;
    let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
    if !failures.is_empty() || !captures.is_empty() {
        session.poisoned = true;
        restore_stack_session(owner, lease, session);
        return Err(captured_failure_message(captures, failures));
    }
    let diagnostics = match prefill_diagnostics_json(&session, &prefill_plan) {
        Ok(diagnostics) => diagnostics,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    let _ = owner.complete(lease, session, CompletionAction::Restore);
    stack_step_result(output, diagnostics)
}

async fn run_logits(
    owner: &AsyncSessionOwner<BrowserDecoderStackSession>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lease: crate::AsyncSessionLease,
    mut session: BrowserDecoderStackSession,
    from_prefill: bool,
    last_row_offset: u64,
) -> Result<JsValue, JsValue> {
    session.poisoned = true;
    let mut ledger: Vec<ScopeKind> = Vec::new();
    for scope in CHECKED_SCOPES {
        if let Err(error) = push_stack_error_scope(device, scope).await {
            ledger.push(scope);
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            restore_stack_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
        ledger.push(scope);
    }
    let raw_device = match raw_stack_device(device) {
        Ok(raw) => raw,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    let raw_queue = match raw_stack_queue(device, queue) {
        Ok(raw) => raw,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    // One logits call: zero writes (both stage uniforms are static), two
    // ordered compute passes (the final RMSNorm over the admitted hidden row,
    // then the LM-head GEMV), one logits readback copy, one submit, one map.
    let encoded = (|| {
        let plan = session.lm_head_plan.as_ref().ok_or_else(|| {
            js_stack_error(&["decoder stack session does not admit the logits operation"])
        })?;
        let logits = session.logits_buffer.as_ref().ok_or_else(|| {
            js_stack_error(&["decoder stack session logits resources are missing"])
        })?;
        let readback = session.logits_readback_buffer.as_ref().ok_or_else(|| {
            js_stack_error(&["decoder stack session logits resources are missing"])
        })?;
        let rms_bind_group = if from_prefill {
            session.prefill_logits_rms_bind_group.as_ref()
        } else {
            session.step_logits_rms_bind_group.as_ref()
        }
        .ok_or_else(|| js_stack_error(&["decoder stack session logits resources are missing"]))?;
        let gemv_bind_group = session.gemv_logits_bind_group.as_ref().ok_or_else(|| {
            js_stack_error(&["decoder stack session logits resources are missing"])
        })?;
        let (final_norm_weight_offset, _) = session
            .weight_resource_plan
            .final_norm_weight_range()
            .ok_or_else(|| {
                js_stack_error(&["decoder stack session final-norm resource range is missing"])
            })?;
        let (lm_head_weight_offset, _) = session
            .weight_resource_plan
            .lm_head_weight_range()
            .ok_or_else(|| {
                js_stack_error(&["decoder stack session LM-head resource range is missing"])
            })?;
        let weight_pipelines = session.decoder_weight_pipelines()?;
        let encoder = create_stack_encoder(raw_device, "decoder-stack-session-logits-encoder")?;
        encode_stack_pass(
            &encoder,
            weight_pipelines.rms_norm,
            rms_bind_group,
            plan.stage_invocations[0].dispatch,
            // The accepted RMS layout dynamically offsets the input row and
            // the weight; the logits weight slice always starts at zero.
            &[last_row_offset, final_norm_weight_offset],
        )?;
        encode_stack_pass(
            &encoder,
            weight_pipelines.gemv_tiled,
            gemv_bind_group,
            plan.stage_invocations[1].dispatch,
            // The accepted GEMV layout dynamically offsets the matrix; the
            // shared LM-head matrix always starts at zero.
            &[lm_head_weight_offset],
        )?;
        encode_stack_copy(&encoder, logits, 0, readback, 0, plan.logits_bytes)?;
        submit_stack_encoder(&raw_queue, &encoder)
    })();
    if let Err(error) = encoded {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if let Err(error) = await_stack_queue_completion(&raw_queue).await {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    let Some(readback_raw) = session
        .logits_readback_buffer
        .as_ref()
        .map(|buffer| AsRef::<JsValue>::as_ref(buffer).clone())
    else {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(
            js_stack_error(&["decoder stack session logits resources are missing"]),
            captures,
            failures,
        ));
    };
    let Some(logits_bytes) = session.lm_head_plan.as_ref().map(|plan| plan.logits_bytes) else {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(
            js_stack_error(&["decoder stack session does not admit the logits operation"]),
            captures,
            failures,
        ));
    };
    if let Err(error) = map_stack_buffer(&readback_raw, logits_bytes).await {
        let _ = unmap_stack_buffer(&readback_raw);
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if owner.generation() != Some(lease.generation()) {
        let _ = unmap_stack_buffer(&readback_raw);
        wait_stack_owner_idle(owner).await;
        let _ = drain_stack_error_scopes(device, &mut ledger).await;
        let _ = owner.complete(lease, session, CompletionAction::Finish);
        return Err(js_stack_error(&[
            "decoder stack session logits is stale: its generation was cancelled",
        ]));
    }
    let output = match read_stack_mapped(&readback_raw, logits_bytes) {
        Ok(output) => output,
        Err(error) => {
            let _ = unmap_stack_buffer(&readback_raw);
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            restore_stack_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    if let Err(error) = unmap_stack_buffer(&readback_raw) {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    session.poisoned = false;
    let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
    if !failures.is_empty() || !captures.is_empty() {
        session.poisoned = true;
        restore_stack_session(owner, lease, session);
        return Err(captured_failure_message(captures, failures));
    }
    let diagnostics = match logits_diagnostics_json(&session) {
        Ok(diagnostics) => diagnostics,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    let _ = owner.complete(lease, session, CompletionAction::Restore);
    stack_logits_result(output, diagnostics)
}

async fn run_top1(
    owner: &AsyncSessionOwner<BrowserDecoderStackSession>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lease: crate::AsyncSessionLease,
    mut session: BrowserDecoderStackSession,
    from_prefill: bool,
    last_row_offset: u64,
) -> Result<JsValue, JsValue> {
    session.poisoned = true;
    let mut ledger: Vec<ScopeKind> = Vec::new();
    for scope in CHECKED_SCOPES {
        if let Err(error) = push_stack_error_scope(device, scope).await {
            ledger.push(scope);
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            restore_stack_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
        ledger.push(scope);
    }
    let raw_device = match raw_stack_device(device) {
        Ok(raw) => raw,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    let raw_queue = match raw_stack_queue(device, queue) {
        Ok(raw) => raw,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    if let Err(error) = session.ensure_top1_resources(raw_device) {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    let encoded = (|| {
        let plan = session.lm_head_plan.as_ref().ok_or_else(|| {
            js_stack_error(&["decoder stack session does not admit the GPU top-1 operation"])
        })?;
        let rms_bind_group = if from_prefill {
            session.prefill_logits_rms_bind_group.as_ref()
        } else {
            session.step_logits_rms_bind_group.as_ref()
        }
        .ok_or_else(|| js_stack_error(&["decoder stack session logits resources are missing"]))?;
        let gemv_bind_group = session.gemv_logits_bind_group.as_ref().ok_or_else(|| {
            js_stack_error(&["decoder stack session logits resources are missing"])
        })?;
        let top1_pipeline = session.top1_pipeline.as_ref().ok_or_else(|| {
            js_stack_error(&["decoder stack session GPU top-1 pipeline is missing"])
        })?;
        let top1_bind_group = session.top1_bind_group.as_ref().ok_or_else(|| {
            js_stack_error(&["decoder stack session GPU top-1 bind group is missing"])
        })?;
        let top1_result = session.top1_result_buffer.as_ref().ok_or_else(|| {
            js_stack_error(&["decoder stack session GPU top-1 result is missing"])
        })?;
        let top1_readback = session.top1_readback_buffer.as_ref().ok_or_else(|| {
            js_stack_error(&["decoder stack session GPU top-1 readback is missing"])
        })?;
        let (final_norm_weight_offset, _) = session
            .weight_resource_plan
            .final_norm_weight_range()
            .ok_or_else(|| {
                js_stack_error(&["decoder stack session final-norm resource range is missing"])
            })?;
        let (lm_head_weight_offset, _) = session
            .weight_resource_plan
            .lm_head_weight_range()
            .ok_or_else(|| {
                js_stack_error(&["decoder stack session LM-head resource range is missing"])
            })?;
        let weight_pipelines = session.decoder_weight_pipelines()?;
        let encoder = create_stack_encoder(raw_device, "decoder-stack-session-top1-encoder")?;
        encode_stack_pass(
            &encoder,
            weight_pipelines.rms_norm,
            rms_bind_group,
            plan.stage_invocations[0].dispatch,
            &[last_row_offset, final_norm_weight_offset],
        )?;
        encode_stack_pass(
            &encoder,
            weight_pipelines.gemv_tiled,
            gemv_bind_group,
            plan.stage_invocations[1].dispatch,
            &[lm_head_weight_offset],
        )?;
        encode_stack_pass(&encoder, top1_pipeline, top1_bind_group, [1, 1, 1], &[])?;
        encode_stack_copy(
            &encoder,
            top1_result,
            0,
            top1_readback,
            0,
            TOP1_RESULT_BYTES,
        )?;
        submit_stack_encoder(&raw_queue, &encoder)
    })();
    if let Err(error) = encoded {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    let queue_started_ms = js_sys::Date::now();
    if let Err(error) = await_stack_queue_completion(&raw_queue).await {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    let queue_wall_time_ns =
        ((js_sys::Date::now() - queue_started_ms).max(0.0) * 1_000_000.0).round() as u64;
    let Some(readback_raw) = session
        .top1_readback_buffer
        .as_ref()
        .map(|buffer| AsRef::<JsValue>::as_ref(buffer).clone())
    else {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(
            js_stack_error(&["decoder stack session GPU top-1 readback is missing"]),
            captures,
            failures,
        ));
    };
    if let Err(error) = map_stack_buffer(&readback_raw, TOP1_RESULT_BYTES).await {
        let _ = unmap_stack_buffer(&readback_raw);
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if owner.generation() != Some(lease.generation()) {
        let _ = unmap_stack_buffer(&readback_raw);
        wait_stack_owner_idle(owner).await;
        let _ = drain_stack_error_scopes(device, &mut ledger).await;
        let _ = owner.complete(lease, session, CompletionAction::Finish);
        return Err(js_stack_error(&[
            "decoder stack session GPU top-1 is stale: its generation was cancelled",
        ]));
    }
    let output = match read_stack_mapped(&readback_raw, TOP1_RESULT_BYTES) {
        Ok(output) => output,
        Err(error) => {
            let _ = unmap_stack_buffer(&readback_raw);
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            restore_stack_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    if let Err(error) = unmap_stack_buffer(&readback_raw) {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if output.len() != TOP1_RESULT_BYTES as usize {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(
            js_stack_error(&["decoder stack GPU top-1 readback is truncated"]),
            captures,
            failures,
        ));
    }
    let mut token_bytes = [0_u8; 4];
    token_bytes.copy_from_slice(&output[0..4]);
    let mut value_bytes = [0_u8; 4];
    value_bytes.copy_from_slice(&output[4..8]);
    let token_id = u32::from_le_bytes(token_bytes);
    let value = f32::from_le_bytes(value_bytes);
    if token_id >= PINNED_STACK_VOCAB_SIZE || !value.is_finite() {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(
            js_stack_error(&["decoder stack GPU top-1 returned an invalid token or score"]),
            captures,
            failures,
        ));
    }
    session.poisoned = false;
    let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
    if !failures.is_empty() || !captures.is_empty() {
        session.poisoned = true;
        restore_stack_session(owner, lease, session);
        return Err(captured_failure_message(captures, failures));
    }
    let diagnostics = match top1_diagnostics_json(&session, queue_wall_time_ns) {
        Ok(diagnostics) => diagnostics,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    let _ = owner.complete(lease, session, CompletionAction::Restore);
    stack_top1_result(token_id, value, diagnostics)
}

async fn run_finish(
    owner: &AsyncSessionOwner<BrowserDecoderStackSession>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    lease: crate::AsyncSessionLease,
    mut session: BrowserDecoderStackSession,
) -> Result<JsValue, JsValue> {
    session.poisoned = true;
    let mut ledger: Vec<ScopeKind> = Vec::new();
    for scope in CHECKED_SCOPES {
        if let Err(error) = push_stack_error_scope(device, scope).await {
            ledger.push(scope);
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            restore_stack_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
        ledger.push(scope);
    }
    let raw_device = match raw_stack_device(device) {
        Ok(raw) => raw,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    let raw_queue = match raw_stack_queue(device, queue) {
        Ok(raw) => raw,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    let (key_readback, value_readback) = match session.encode_finish(raw_device, &raw_queue) {
        Ok(readbacks) => readbacks,
        Err(error) => {
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            restore_stack_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    let cache_bytes = u64::from(session.stack_plan.layers)
        .checked_mul(session.stack_plan.cache_stride_bytes)
        .ok_or_else(|| js_stack_error(&["decoder stack compact cache byte size overflowed"]))?;
    if let Err(error) = await_stack_queue_completion(&raw_queue).await {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if let Err(error) = map_stack_buffer(key_readback.as_ref(), cache_bytes).await {
        let _ = unmap_stack_buffer(key_readback.as_ref());
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if owner.generation() != Some(lease.generation()) {
        let _ = unmap_stack_buffer(key_readback.as_ref());
        wait_stack_owner_idle(owner).await;
        let _ = drain_stack_error_scopes(device, &mut ledger).await;
        let _ = owner.complete(lease, session, CompletionAction::Finish);
        return Err(js_stack_error(&[
            "decoder stack session finish is stale: its generation was cancelled",
        ]));
    }
    let key_cache = match read_stack_mapped(key_readback.as_ref(), cache_bytes) {
        Ok(mapped) => mapped,
        Err(error) => {
            let _ = unmap_stack_buffer(key_readback.as_ref());
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            restore_stack_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    if let Err(error) = unmap_stack_buffer(key_readback.as_ref()) {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    if let Err(error) = map_stack_buffer(value_readback.as_ref(), cache_bytes).await {
        let _ = unmap_stack_buffer(value_readback.as_ref());
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    let value_cache = match read_stack_mapped(value_readback.as_ref(), cache_bytes) {
        Ok(mapped) => mapped,
        Err(error) => {
            let _ = unmap_stack_buffer(value_readback.as_ref());
            let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
            restore_stack_session(owner, lease, session);
            return Err(drain_appended_message(error, captures, failures));
        }
    };
    if let Err(error) = unmap_stack_buffer(value_readback.as_ref()) {
        let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
        restore_stack_session(owner, lease, session);
        return Err(drain_appended_message(error, captures, failures));
    }
    let (captures, failures) = drain_stack_error_scopes(device, &mut ledger).await;
    if !failures.is_empty() || !captures.is_empty() {
        session.poisoned = true;
        restore_stack_session(owner, lease, session);
        return Err(captured_failure_message(captures, failures));
    }
    let diagnostics = match finish_diagnostics_json(&session) {
        Ok(diagnostics) => diagnostics,
        Err(error) => {
            restore_stack_session(owner, lease, session);
            return Err(error);
        }
    };
    let _ = owner.complete(lease, session, CompletionAction::Finish);
    stack_finish_result(key_cache, value_cache, diagnostics)
}

impl DecoderStackSessionAuthority {
    pub(super) fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        resident_weight_cache: SharedDecoderStackResidentWeightCache,
    ) -> Self {
        Self {
            device,
            queue,
            owner: crate::AsyncSessionOwner::new(),
            resident_weight_cache,
        }
    }

    pub(super) fn abort(&self) {
        let _ = self.owner.cancel_and_release();
    }

    pub(super) fn resident_weights_json(&self) -> Result<String, JsValue> {
        let cache = self.resident_weight_cache.get();
        let mut root = Map::new();
        root.insert("schema_version".to_owned(), Value::from(1u64));
        root.insert("ready".to_owned(), Value::from(cache.is_some()));
        root.insert(
            "resident_weight_bytes".to_owned(),
            Value::from(
                cache
                    .as_ref()
                    .map(|weights| weights.resident_bytes)
                    .unwrap_or(0),
            ),
        );
        if let Some(weights) = cache.as_ref() {
            root.insert("key".to_owned(), Value::from(weights.key.as_str()));
        }
        json_text(Value::Object(root))
    }

    pub(super) fn probe_m7q1_precision_admission_json(
        &self,
        precision_profile: &str,
        shader_f16_available: bool,
    ) -> Result<String, JsValue> {
        probe_m7q1_precision_admission_json(precision_profile, shader_f16_available)
    }

    pub(super) async fn run_m7q1_fp16_weight_probe_json(
        &self,
        fixture_json: &str,
    ) -> Result<String, JsValue> {
        run_m7q1_fp16_weight_probe_json(&self.device, &self.queue, fixture_json).await
    }

    pub(super) fn shader_sources_json(&self) -> Result<String, JsValue> {
        let sources = canonical_stack_sources()?;
        let rms_norm_source = serde_json::to_string(&sources.rms_norm).map_err(|error| {
            js_stack_error(&[
                "cannot serialize decoder stack RMS-norm source: ",
                &error.to_string(),
            ])
        })?;
        let gemv_source = serde_json::to_string(&sources.gemv).map_err(|error| {
            js_stack_error(&[
                "cannot serialize decoder stack GEMV source: ",
                &error.to_string(),
            ])
        })?;
        let gemv_tiled_source = serde_json::to_string(&sources.gemv_tiled).map_err(|error| {
            js_stack_error(&[
                "cannot serialize decoder stack tiled GEMV source: ",
                &error.to_string(),
            ])
        })?;
        let mrope_source = serde_json::to_string(&sources.mrope).map_err(|error| {
            js_stack_error(&[
                "cannot serialize decoder stack M-RoPE source: ",
                &error.to_string(),
            ])
        })?;
        let append_source = serde_json::to_string(&sources.append).map_err(|error| {
            js_stack_error(&[
                "cannot serialize decoder stack append source: ",
                &error.to_string(),
            ])
        })?;
        let attention_source = serde_json::to_string(&sources.attention).map_err(|error| {
            js_stack_error(&[
                "cannot serialize decoder stack attention source: ",
                &error.to_string(),
            ])
        })?;
        let swiglu_source = serde_json::to_string(&sources.swiglu).map_err(|error| {
            js_stack_error(&[
                "cannot serialize decoder stack SwiGLU source: ",
                &error.to_string(),
            ])
        })?;
        let residual_source = serde_json::to_string(&sources.residual).map_err(|error| {
            js_stack_error(&[
                "cannot serialize decoder stack residual source: ",
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
        json.push_str(",\"decoder_swiglu_f32\":");
        json.push_str(&swiglu_source);
        json.push_str(",\"add_f32\":");
        json.push_str(&residual_source);
        json.push_str(",\"gemv_tiled_f32\":");
        json.push_str(&gemv_tiled_source);
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
        json.push_str("\",\"decoder_swiglu_f32\":\"");
        json.push_str(&blake3_hex(&source_blake3(&sources.swiglu)));
        json.push_str("\",\"add_f32\":\"");
        json.push_str(&blake3_hex(&source_blake3(&sources.residual)));
        json.push_str("\",\"gemv_tiled_f32\":\"");
        json.push_str(&blake3_hex(&source_blake3(&sources.gemv_tiled)));
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
        let prepared = match prepare_stack_begin(
            &self.owner,
            &self.device,
            descriptor_json,
            pack,
            key_cache,
            value_cache,
        ) {
            Ok(prepared) => prepared,
            Err(error) => return js_sys::Promise::reject(&error),
        };
        let PreparedStackBegin {
            kv_plan,
            stack_plan,
            weight_resource_plan,
            checkpoint_blake3,
            prefill_plan,
            lm_head_plan,
            prefill_capable,
            operands,
            sources,
        } = prepared;
        let owner = self.owner.clone();
        let device = self.device.clone();
        let queue = self.queue.clone();
        let resident_weight_cache = self.resident_weight_cache.clone();
        return wasm_bindgen_futures::future_to_promise(async move {
            run_begin(
                &owner,
                &resident_weight_cache,
                &device,
                &queue,
                kv_plan,
                stack_plan,
                weight_resource_plan,
                checkpoint_blake3,
                prefill_plan,
                lm_head_plan,
                prefill_capable,
                operands,
                sources,
            )
            .await
            .map(|diagnostics| JsValue::from_str(&diagnostics))
        });
        /*
        let prepared = (|| {
            check_stack_admission(&self.owner)?;
            let parsed = parse_stack_descriptor_json(descriptor_json)?;
            let prefill_capable = parsed.prefill_tokens > 0;
            let pack_bytes = stack_pack_to_bytes(pack)?;
            let weight_pack = parse_stack_weight_pack(
                &pack_bytes,
                parsed.prefix_tokens,
                parsed.cache_capacity,
                prefill_capable,
            )?;
            // Dual admission: the descriptor and the pack must agree on the
            // logits capability. The 15-field descriptor with the pinned
            // vocab_size pairs with the 14-section pack; the accepted legacy
            // shapes pair with each other; every other combination is a
            // fail-closed rejection before the first GPU effect.
            let logits_capable = match (parsed.vocab_size, weight_pack.final_norm_weight.is_some())
            {
                (Some(_), true) => true,
                (None, false) => false,
                _ => {
                    return Err(js_stack_error(&[
                        "decoder stack session logits capability drifted between the descriptor and the weight pack",
                    ]));
                }
            };
            // The raw pack bytes are dead once the shard payloads are owned by
            // the parsed pack; dropping them before the f32 conversion vecs
            // are built keeps the wasm32 begin peak at ~2x the pack size
            // instead of ~3x (the M6e8 1.44 GB pack would exceed 4 GB).
            drop(pack_bytes);
            let key_cache_bytes = stack_uint8_to_bytes(key_cache)?;
            let value_cache_bytes = stack_uint8_to_bytes(value_cache)?;
            let key_f32 = require_stack_cache_operands(
                &key_cache_bytes,
                "initial K cache",
                parsed.prefix_tokens,
                parsed.cache_capacity,
            )?;
            let value_f32 = require_stack_cache_operands(
                &value_cache_bytes,
                "initial V cache",
                parsed.prefix_tokens,
                parsed.cache_capacity,
            )?;
            let norm1_f32 =
                stack_bytes_to_f32(&weight_pack.norm1_weight, "weights.input_layernorm")?;
            let q_f32 = stack_bytes_to_f32(&weight_pack.q_weight, "weights.q_proj")?;
            let k_f32 = stack_bytes_to_f32(&weight_pack.k_weight, "weights.k_proj")?;
            let v_f32 = stack_bytes_to_f32(&weight_pack.v_weight, "weights.v_proj")?;
            let o_f32 = stack_bytes_to_f32(&weight_pack.o_weight, "weights.o_proj")?;
            let cos_f32 = stack_bytes_to_f32(&weight_pack.rope_cos, "weights.mrope_cos")?;
            let sin_f32 = stack_bytes_to_f32(&weight_pack.rope_sin, "weights.mrope_sin")?;
            let norm2_f32 = stack_bytes_to_f32(
                &weight_pack.norm2_weight,
                "weights.post_attention_layernorm",
            )?;
            let gate_f32 = stack_bytes_to_f32(&weight_pack.gate_weight, "weights.gate_proj")?;
            let up_f32 = stack_bytes_to_f32(&weight_pack.up_weight, "weights.up_proj")?;
            let down_f32 = stack_bytes_to_f32(&weight_pack.down_weight, "weights.down_proj")?;
            let cache_plane_elements = u64::from(parsed.cache_capacity)
                .checked_mul(u64::from(PINNED_STACK_KEY_VALUE_WIDTH))
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| js_stack_error(&["initial cache plane element count overflowed"]))?;
            let admitted_prefix_tokens = if prefill_capable {
                parsed
                    .prefill_tokens
                    .min(parsed.cache_capacity.saturating_sub(1))
            } else {
                parsed.prefix_tokens
            };
            let kv_descriptor = DecoderKvSessionDescriptor {
                query_heads: parsed.query_heads,
                key_value_heads: parsed.key_value_heads,
                head_dim: parsed.head_dim,
                prefix_tokens: admitted_prefix_tokens,
                cache_capacity: parsed.cache_capacity,
                key_cache: key_f32[..cache_plane_elements].as_ref(),
                value_cache: value_f32[..cache_plane_elements].as_ref(),
            };
            let kv_plan = kv_descriptor.plan().map_err(|error| {
                js_stack_error(&[
                    "invalid decoder stack session descriptor geometry or initial cache operands: ",
                    &error.to_string(),
                ])
            })?;
            let stack_descriptor = DecoderStackDescriptor {
                layers: pvlc_runtime_core::PINNED_DECODER_LAYERS,
                hidden_size: parsed.hidden_size,
                intermediate_size: parsed.intermediate_size,
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
                norm2_weight: norm2_f32.as_slice(),
                gate_weight: gate_f32.as_slice(),
                up_weight: up_f32.as_slice(),
                down_weight: down_f32.as_slice(),
                cache_capacity: parsed.cache_capacity,
            };
            let stack_plan = stack_descriptor.plan().map_err(|error| {
                js_stack_error(&[
                    "invalid decoder stack session descriptor geometry or weight operands: ",
                    &error.to_string(),
                ])
            })?;
            let admitted_prefill_tokens = if prefill_capable {
                parsed.prefill_tokens
            } else {
                1
            };
            let prefill_descriptor = DecoderStackPrefillDescriptor {
                layers: pvlc_runtime_core::PINNED_DECODER_LAYERS,
                hidden_size: parsed.hidden_size,
                intermediate_size: parsed.intermediate_size,
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
                norm2_weight: norm2_f32.as_slice(),
                gate_weight: gate_f32.as_slice(),
                up_weight: up_f32.as_slice(),
                down_weight: down_f32.as_slice(),
                cache_capacity: parsed.cache_capacity,
                tokens: admitted_prefill_tokens,
            };
            let prefill_plan = prefill_descriptor.plan().map_err(|error| {
                js_stack_error(&[
                    "invalid decoder stack session prefill geometry or weight operands: ",
                    &error.to_string(),
                ])
            })?;
            // The LM-head plan is bound once at begin, conditionally on the
            // admitted capability: the legacy path keeps its three accepted
            // planner calls and never touches the logits operands.
            let lm_head_plan = if logits_capable {
                let (Some(final_norm_bytes), Some(lm_head_bytes)) = (
                    weight_pack.final_norm_weight.as_ref(),
                    weight_pack.lm_head_weight.as_ref(),
                ) else {
                    return Err(js_stack_error(&[
                        "decoder stack session logits weight pack shards are missing",
                    ]));
                };
                let final_norm_f32 =
                    stack_bytes_to_f32(final_norm_bytes, "weights.final_layernorm")?;
                let lm_head_f32 = stack_bytes_to_f32(lm_head_bytes, "weights.lm_head")?;
                let lm_head_descriptor = DecoderLmHeadDescriptor::pinned(
                    final_norm_f32.as_slice(),
                    lm_head_f32.as_slice(),
                );
                Some(lm_head_descriptor.plan().map_err(|error| {
                    js_stack_error(&[
                        "invalid decoder stack session logits geometry or weight operands: ",
                        &error.to_string(),
                    ])
                })?)
            } else {
                None
            };
            // The f32 conversion vecs exist only for planner validation and
            // are dropped before the upload phase, keeping the wasm32 begin
            // peak near 2x the pack size.
            drop(key_f32);
            drop(value_f32);
            drop(norm1_f32);
            drop(q_f32);
            drop(k_f32);
            drop(v_f32);
            drop(o_f32);
            drop(cos_f32);
            drop(sin_f32);
            drop(norm2_f32);
            drop(gate_f32);
            drop(up_f32);
            drop(down_f32);
            let sources = canonical_stack_sources()?;
            let operands = StackBeginOperands {
                upload_initial_cache: true,
                key_cache_bytes,
                value_cache_bytes,
                norm1_weight_bytes: weight_pack.norm1_weight,
                q_weight_bytes: weight_pack.q_weight,
                k_weight_bytes: weight_pack.k_weight,
                v_weight_bytes: weight_pack.v_weight,
                o_weight_bytes: weight_pack.o_weight,
                rope_cos_bytes: weight_pack.rope_cos,
                rope_sin_bytes: weight_pack.rope_sin,
                norm2_weight_bytes: weight_pack.norm2_weight,
                gate_weight_bytes: weight_pack.gate_weight,
                up_weight_bytes: weight_pack.up_weight,
                down_weight_bytes: weight_pack.down_weight,
                final_norm_weight_bytes: weight_pack.final_norm_weight,
                lm_head_weight_bytes: weight_pack.lm_head_weight,
            };
            Ok((
                kv_plan,
                stack_plan,
                prefill_plan,
                lm_head_plan,
                prefill_capable,
                operands,
                sources,
            ))
        })();
        let (kv_plan, stack_plan, prefill_plan, lm_head_plan, prefill_capable, operands, sources) =
            match prepared {
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
                stack_plan,
                prefill_plan,
                lm_head_plan,
                prefill_capable,
                operands,
                sources,
            )
            .await
            .map(|diagnostics| JsValue::from_str(&diagnostics))
        })
        */
    }

    pub(super) fn begin_resident(
        &self,
        descriptor_json: &str,
        rope_cos: &js_sys::Uint8Array,
        rope_sin: &js_sys::Uint8Array,
    ) -> js_sys::Promise {
        let prepared = match prepare_resident_stack_begin(
            &self.owner,
            &self.resident_weight_cache,
            &self.device,
            descriptor_json,
            rope_cos,
            rope_sin,
        ) {
            Ok(prepared) => prepared,
            Err(error) => return js_sys::Promise::reject(&error),
        };
        let PreparedStackBegin {
            kv_plan,
            stack_plan,
            weight_resource_plan,
            checkpoint_blake3,
            prefill_plan,
            lm_head_plan,
            prefill_capable,
            operands,
            sources,
        } = prepared;
        let owner = self.owner.clone();
        let device = self.device.clone();
        let queue = self.queue.clone();
        let resident_weight_cache = self.resident_weight_cache.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            run_begin(
                &owner,
                &resident_weight_cache,
                &device,
                &queue,
                kv_plan,
                stack_plan,
                weight_resource_plan,
                checkpoint_blake3,
                prefill_plan,
                lm_head_plan,
                prefill_capable,
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
            return js_sys::Promise::reject(&js_stack_error(&[
                "unknown decoder stack shader override kernel: ",
                kernel,
            ]));
        }
        if source.is_empty() {
            return js_sys::Promise::reject(&js_stack_error(&[
                "decoder stack shader override source must not be empty",
            ]));
        }
        let mut prepared = match prepare_stack_begin(
            &self.owner,
            &self.device,
            descriptor_json,
            pack,
            key_cache,
            value_cache,
        ) {
            Ok(prepared) => prepared,
            Err(error) => return js_sys::Promise::reject(&error),
        };
        if kernel == RMS_NORM_KERNEL_NAME {
            prepared.sources.rms_norm = source.to_owned();
        } else if kernel == GEMV_KERNEL_NAME {
            prepared.sources.gemv = source.to_owned();
        } else if kernel == GEMV_TILED_KERNEL_NAME {
            prepared.sources.gemv_tiled = source.to_owned();
        } else if kernel == MROPE_KERNEL_NAME {
            prepared.sources.mrope = source.to_owned();
        } else if kernel == APPEND_KERNEL_NAME {
            prepared.sources.append = source.to_owned();
        } else if kernel == ATTENTION_KERNEL_NAME {
            prepared.sources.attention = source.to_owned();
        } else if kernel == SWIGLU_KERNEL_NAME {
            prepared.sources.swiglu = source.to_owned();
        } else if kernel == RESIDUAL_KERNEL_NAME {
            prepared.sources.residual = source.to_owned();
        } else if kernel == PROJECTION_KERNEL_NAME {
            prepared.sources.projection = source.to_owned();
        } else if kernel == PREFILL_MROPE_KERNEL_NAME {
            prepared.sources.prefill_mrope = source.to_owned();
        } else if kernel == KV_APPEND_RANGE_KERNEL_NAME {
            prepared.sources.kv_append_range = source.to_owned();
        } else if kernel == SPLIT_PARTIAL_KERNEL_NAME {
            prepared.sources.split_partial = source.to_owned();
        } else if kernel == SPLIT_MERGE_KERNEL_NAME {
            prepared.sources.split_merge = source.to_owned();
        } else {
            prepared.sources.prefill_gqa = source.to_owned();
        }
        let PreparedStackBegin {
            kv_plan,
            stack_plan,
            weight_resource_plan,
            checkpoint_blake3,
            prefill_plan,
            lm_head_plan,
            prefill_capable,
            operands,
            sources,
        } = prepared;
        let owner = self.owner.clone();
        let device = self.device.clone();
        let queue = self.queue.clone();
        let resident_weight_cache = self.resident_weight_cache.clone();
        return wasm_bindgen_futures::future_to_promise(async move {
            run_begin(
                &owner,
                &resident_weight_cache,
                &device,
                &queue,
                kv_plan,
                stack_plan,
                weight_resource_plan,
                checkpoint_blake3,
                prefill_plan,
                lm_head_plan,
                prefill_capable,
                operands,
                sources,
            )
            .await
            .map(|diagnostics| JsValue::from_str(&diagnostics))
        });
        /*
        let prepared = (|| {
            check_stack_admission(&self.owner)?;
            let parsed = parse_stack_descriptor_json(descriptor_json)?;
            let prefill_capable = parsed.prefill_tokens > 0;
            let pack_bytes = stack_pack_to_bytes(pack)?;
            let weight_pack = parse_stack_weight_pack(
                &pack_bytes,
                parsed.prefix_tokens,
                parsed.cache_capacity,
                prefill_capable,
            )?;
            // Dual admission: the descriptor and the pack must agree on the
            // logits capability. The 15-field descriptor with the pinned
            // vocab_size pairs with the 14-section pack; the accepted legacy
            // shapes pair with each other; every other combination is a
            // fail-closed rejection before the first GPU effect.
            let logits_capable = match (parsed.vocab_size, weight_pack.final_norm_weight.is_some())
            {
                (Some(_), true) => true,
                (None, false) => false,
                _ => {
                    return Err(js_stack_error(&[
                        "decoder stack session logits capability drifted between the descriptor and the weight pack",
                    ]));
                }
            };
            // The raw pack bytes are dead once the shard payloads are owned by
            // the parsed pack; dropping them before the f32 conversion vecs
            // are built keeps the wasm32 begin peak at ~2x the pack size
            // instead of ~3x (the M6e8 1.44 GB pack would exceed 4 GB).
            drop(pack_bytes);
            let key_cache_bytes = stack_uint8_to_bytes(key_cache)?;
            let value_cache_bytes = stack_uint8_to_bytes(value_cache)?;
            let key_f32 = require_stack_cache_operands(
                &key_cache_bytes,
                "initial K cache",
                parsed.prefix_tokens,
                parsed.cache_capacity,
            )?;
            let value_f32 = require_stack_cache_operands(
                &value_cache_bytes,
                "initial V cache",
                parsed.prefix_tokens,
                parsed.cache_capacity,
            )?;
            let norm1_f32 =
                stack_bytes_to_f32(&weight_pack.norm1_weight, "weights.input_layernorm")?;
            let q_f32 = stack_bytes_to_f32(&weight_pack.q_weight, "weights.q_proj")?;
            let k_f32 = stack_bytes_to_f32(&weight_pack.k_weight, "weights.k_proj")?;
            let v_f32 = stack_bytes_to_f32(&weight_pack.v_weight, "weights.v_proj")?;
            let o_f32 = stack_bytes_to_f32(&weight_pack.o_weight, "weights.o_proj")?;
            let cos_f32 = stack_bytes_to_f32(&weight_pack.rope_cos, "weights.mrope_cos")?;
            let sin_f32 = stack_bytes_to_f32(&weight_pack.rope_sin, "weights.mrope_sin")?;
            let norm2_f32 = stack_bytes_to_f32(
                &weight_pack.norm2_weight,
                "weights.post_attention_layernorm",
            )?;
            let gate_f32 = stack_bytes_to_f32(&weight_pack.gate_weight, "weights.gate_proj")?;
            let up_f32 = stack_bytes_to_f32(&weight_pack.up_weight, "weights.up_proj")?;
            let down_f32 = stack_bytes_to_f32(&weight_pack.down_weight, "weights.down_proj")?;
            let cache_plane_elements = u64::from(parsed.cache_capacity)
                .checked_mul(u64::from(PINNED_STACK_KEY_VALUE_WIDTH))
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| js_stack_error(&["initial cache plane element count overflowed"]))?;
            let admitted_prefix_tokens = if prefill_capable {
                parsed
                    .prefill_tokens
                    .min(parsed.cache_capacity.saturating_sub(1))
            } else {
                parsed.prefix_tokens
            };
            let kv_descriptor = DecoderKvSessionDescriptor {
                query_heads: parsed.query_heads,
                key_value_heads: parsed.key_value_heads,
                head_dim: parsed.head_dim,
                prefix_tokens: admitted_prefix_tokens,
                cache_capacity: parsed.cache_capacity,
                key_cache: key_f32[..cache_plane_elements].as_ref(),
                value_cache: value_f32[..cache_plane_elements].as_ref(),
            };
            let kv_plan = kv_descriptor.plan().map_err(|error| {
                js_stack_error(&[
                    "invalid decoder stack session descriptor geometry or initial cache operands: ",
                    &error.to_string(),
                ])
            })?;
            let stack_descriptor = DecoderStackDescriptor {
                layers: pvlc_runtime_core::PINNED_DECODER_LAYERS,
                hidden_size: parsed.hidden_size,
                intermediate_size: parsed.intermediate_size,
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
                norm2_weight: norm2_f32.as_slice(),
                gate_weight: gate_f32.as_slice(),
                up_weight: up_f32.as_slice(),
                down_weight: down_f32.as_slice(),
                cache_capacity: parsed.cache_capacity,
            };
            let stack_plan = stack_descriptor.plan().map_err(|error| {
                js_stack_error(&[
                    "invalid decoder stack session descriptor geometry or weight operands: ",
                    &error.to_string(),
                ])
            })?;
            let admitted_prefill_tokens = if prefill_capable {
                parsed.prefill_tokens
            } else {
                1
            };
            let prefill_descriptor = DecoderStackPrefillDescriptor {
                layers: pvlc_runtime_core::PINNED_DECODER_LAYERS,
                hidden_size: parsed.hidden_size,
                intermediate_size: parsed.intermediate_size,
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
                norm2_weight: norm2_f32.as_slice(),
                gate_weight: gate_f32.as_slice(),
                up_weight: up_f32.as_slice(),
                down_weight: down_f32.as_slice(),
                cache_capacity: parsed.cache_capacity,
                tokens: admitted_prefill_tokens,
            };
            let prefill_plan = prefill_descriptor.plan().map_err(|error| {
                js_stack_error(&[
                    "invalid decoder stack session prefill geometry or weight operands: ",
                    &error.to_string(),
                ])
            })?;
            // The LM-head plan is bound once at begin, conditionally on the
            // admitted capability: the legacy path keeps its three accepted
            // planner calls and never touches the logits operands.
            let lm_head_plan = if logits_capable {
                let (Some(final_norm_bytes), Some(lm_head_bytes)) = (
                    weight_pack.final_norm_weight.as_ref(),
                    weight_pack.lm_head_weight.as_ref(),
                ) else {
                    return Err(js_stack_error(&[
                        "decoder stack session logits weight pack shards are missing",
                    ]));
                };
                let final_norm_f32 =
                    stack_bytes_to_f32(final_norm_bytes, "weights.final_layernorm")?;
                let lm_head_f32 = stack_bytes_to_f32(lm_head_bytes, "weights.lm_head")?;
                let lm_head_descriptor = DecoderLmHeadDescriptor::pinned(
                    final_norm_f32.as_slice(),
                    lm_head_f32.as_slice(),
                );
                Some(lm_head_descriptor.plan().map_err(|error| {
                    js_stack_error(&[
                        "invalid decoder stack session logits geometry or weight operands: ",
                        &error.to_string(),
                    ])
                })?)
            } else {
                None
            };
            // The f32 conversion vecs exist only for planner validation and
            // are dropped before the upload phase, keeping the wasm32 begin
            // peak near 2x the pack size.
            drop(key_f32);
            drop(value_f32);
            drop(norm1_f32);
            drop(q_f32);
            drop(k_f32);
            drop(v_f32);
            drop(o_f32);
            drop(cos_f32);
            drop(sin_f32);
            drop(norm2_f32);
            drop(gate_f32);
            drop(up_f32);
            drop(down_f32);
            let mut sources = canonical_stack_sources()?;
            if kernel == RMS_NORM_KERNEL_NAME {
                sources.rms_norm = source.to_owned();
            } else if kernel == GEMV_KERNEL_NAME {
                sources.gemv = source.to_owned();
            } else if kernel == GEMV_TILED_KERNEL_NAME {
                sources.gemv_tiled = source.to_owned();
            } else if kernel == MROPE_KERNEL_NAME {
                sources.mrope = source.to_owned();
            } else if kernel == APPEND_KERNEL_NAME {
                sources.append = source.to_owned();
            } else if kernel == ATTENTION_KERNEL_NAME {
                sources.attention = source.to_owned();
            } else if kernel == SWIGLU_KERNEL_NAME {
                sources.swiglu = source.to_owned();
            } else if kernel == RESIDUAL_KERNEL_NAME {
                sources.residual = source.to_owned();
            } else if kernel == PROJECTION_KERNEL_NAME {
                sources.projection = source.to_owned();
            } else if kernel == PREFILL_MROPE_KERNEL_NAME {
                sources.prefill_mrope = source.to_owned();
            } else if kernel == KV_APPEND_RANGE_KERNEL_NAME {
                sources.kv_append_range = source.to_owned();
            } else if kernel == SPLIT_PARTIAL_KERNEL_NAME {
                sources.split_partial = source.to_owned();
            } else if kernel == SPLIT_MERGE_KERNEL_NAME {
                sources.split_merge = source.to_owned();
            } else {
                sources.prefill_gqa = source.to_owned();
            }
            let operands = StackBeginOperands {
                upload_initial_cache: true,
                key_cache_bytes,
                value_cache_bytes,
                norm1_weight_bytes: weight_pack.norm1_weight,
                q_weight_bytes: weight_pack.q_weight,
                k_weight_bytes: weight_pack.k_weight,
                v_weight_bytes: weight_pack.v_weight,
                o_weight_bytes: weight_pack.o_weight,
                rope_cos_bytes: weight_pack.rope_cos,
                rope_sin_bytes: weight_pack.rope_sin,
                norm2_weight_bytes: weight_pack.norm2_weight,
                gate_weight_bytes: weight_pack.gate_weight,
                up_weight_bytes: weight_pack.up_weight,
                down_weight_bytes: weight_pack.down_weight,
                final_norm_weight_bytes: weight_pack.final_norm_weight,
                lm_head_weight_bytes: weight_pack.lm_head_weight,
            };
            Ok((
                kv_plan,
                stack_plan,
                prefill_plan,
                lm_head_plan,
                prefill_capable,
                operands,
                sources,
            ))
        })();
        let (kv_plan, stack_plan, prefill_plan, lm_head_plan, prefill_capable, operands, sources) =
            match prepared {
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
                stack_plan,
                prefill_plan,
                lm_head_plan,
                prefill_capable,
                operands,
                sources,
            )
            .await
            .map(|diagnostics| JsValue::from_str(&diagnostics))
        })
        */
    }

    pub(super) fn step(&self, hidden_token: &js_sys::Uint8Array) -> js_sys::Promise {
        let prepared = (|| {
            let (lease, session) = acquire_stack_session(&self.owner)?;
            let hidden_bytes = match stack_uint8_to_bytes(hidden_token) {
                Ok(bytes) => bytes,
                Err(error) => {
                    restore_stack_session(&self.owner, lease, session);
                    return Err(error);
                }
            };
            let hidden_f32 = match stack_bytes_to_f32(&hidden_bytes, "step hidden row") {
                Ok(value) => value,
                Err(error) => {
                    restore_stack_session(&self.owner, lease, session);
                    return Err(error);
                }
            };
            let step_input = DecoderStackStep {
                hidden_row: hidden_f32.as_slice(),
            };
            let transition = match session.kv_plan.plan_cache_transition(session.cache_tokens) {
                Ok(transition) => transition,
                Err(error) => {
                    restore_stack_session(&self.owner, lease, session);
                    return Err(js_stack_error(&[
                        "invalid decoder stack step cache capacity: ",
                        &error.to_string(),
                    ]));
                }
            };
            let step_plan = match session
                .stack_plan
                .plan_step(session.cache_tokens, &step_input)
            {
                Ok(step_plan) => step_plan,
                Err(error) => {
                    restore_stack_session(&self.owner, lease, session);
                    return Err(js_stack_error(&[
                        "invalid decoder stack step hidden row: ",
                        &error.to_string(),
                    ]));
                }
            };
            Ok((
                lease,
                session,
                StackStepOperands {
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

    /// Admits one prompt prefill on a zero-prefix session: the caller-owned
    /// token-major hidden states `[tokens, 1024]` are copied synchronously,
    /// validated for the exact admitted byte length and finiteness, and the
    /// exact core prefill plan is re-derived under the payload-free geometry
    /// authority. Checkpoint bytes were admitted once at session begin, so
    /// prefill never allocates zero-valued model-sized placeholder weights.
    pub(super) fn prefill(&self, hidden_states: &js_sys::Uint8Array) -> js_sys::Promise {
        let prepared = (|| {
            let (lease, session) = acquire_stack_session(&self.owner)?;
            if session.cache_tokens == 0 {
                let hidden_bytes = match stack_uint8_to_bytes(hidden_states) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        restore_stack_session(&self.owner, lease, session);
                        return Err(error);
                    }
                };
                let hidden_f32 = match stack_bytes_to_f32(&hidden_bytes, "prefill hidden states") {
                    Ok(values) => values,
                    Err(error) => {
                        restore_stack_session(&self.owner, lease, session);
                        return Err(error);
                    }
                };
                let Some(expected_bytes) = u64::from(session.prefill_plan.tokens).checked_mul(4096)
                else {
                    restore_stack_session(&self.owner, lease, session);
                    return Err(js_stack_error(&[
                        "decoder stack prefill hidden states byte size overflowed",
                    ]));
                };
                if hidden_bytes.len() as u64 != expected_bytes {
                    restore_stack_session(&self.owner, lease, session);
                    return Err(js_stack_error(&[
                        "decoder stack prefill hidden states byte length does not match the admitted token count",
                    ]));
                }
                if !hidden_f32.iter().all(|element| element.is_finite()) {
                    restore_stack_session(&self.owner, lease, session);
                    return Err(js_stack_error(&[
                        "decoder stack prefill hidden states contain a nonfinite f32 element",
                    ]));
                }
                let attention = session.stack_plan.layer_plan.attention_block;
                let prefill_plan = match (DecoderStackGeometryDescriptor {
                    layers: session.stack_plan.layers,
                    hidden_size: attention.hidden_size,
                    intermediate_size: session.stack_plan.layer_plan.intermediate_size,
                    query_heads: attention.query_heads,
                    key_value_heads: attention.key_value_heads,
                    head_dim: attention.head_dim,
                    rms_norm_epsilon: attention.rms_norm_epsilon,
                    cache_capacity: session.prefill_plan.cache_capacity,
                })
                .plan_prefill(session.prefill_plan.tokens)
                {
                    Ok(plan) => plan,
                    Err(error) => {
                        restore_stack_session(&self.owner, lease, session);
                        return Err(js_stack_error(&[
                            "invalid decoder stack prefill geometry: ",
                            &error.to_string(),
                        ]));
                    }
                };
                Ok((
                    lease,
                    session,
                    StackPrefillOperands {
                        prefill_plan,
                        hidden_bytes,
                    },
                ))
            } else {
                restore_stack_session(&self.owner, lease, session);
                Err(js_stack_error(&[
                    "decoder stack session prefill requires a zero-prefix cache position",
                ]))
            }
        })();
        let (lease, session, operands) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return js_sys::Promise::reject(&error),
        };
        let owner = self.owner.clone();
        let device = self.device.clone();
        let queue = self.queue.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            run_prefill(&owner, &device, &queue, lease, session, operands).await
        })
    }

    /// Reads the current hidden row on device into the full f32 vocabulary
    /// logits. Admission is fail-closed before the first GPU effect: a legacy
    /// (12-section) session is rejected with zero effect, and a logits-capable
    /// session is rejected until its first admitted prefill or decode step.
    /// The operation is a pure readout: it never moves `cache_tokens`, never
    /// writes, and repeated calls are bit-identical.
    pub(super) fn logits(&self) -> js_sys::Promise {
        let prepared = prepare_logits_readout(&self.owner, "logits");
        let (lease, session, from_prefill, last_row_offset) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return js_sys::Promise::reject(&error),
        };
        let owner = self.owner.clone();
        let device = self.device.clone();
        let queue = self.queue.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            run_logits(
                &owner,
                &device,
                &queue,
                lease,
                session,
                from_prefill,
                last_row_offset,
            )
            .await
        })
    }

    /// Computes final RMSNorm, the LM-head projection and exact greedy top-1
    /// in one command buffer. Only the selected token id and f32 score (8
    /// bytes) cross to the CPU; the full vocabulary row remains on the GPU.
    pub(super) fn top1(&self) -> js_sys::Promise {
        let prepared = prepare_logits_readout(&self.owner, "GPU top-1");
        let (lease, session, from_prefill, last_row_offset) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return js_sys::Promise::reject(&error),
        };
        let owner = self.owner.clone();
        let device = self.device.clone();
        let queue = self.queue.clone();
        wasm_bindgen_futures::future_to_promise(async move {
            run_top1(
                &owner,
                &device,
                &queue,
                lease,
                session,
                from_prefill,
                last_row_offset,
            )
            .await
        })
    }

    pub(super) fn finish(&self) -> js_sys::Promise {
        let prepared = acquire_stack_session(&self.owner);
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
